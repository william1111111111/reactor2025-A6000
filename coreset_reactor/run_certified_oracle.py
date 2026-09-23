"""Read-only frozen checkpoints -> cached VAL predictions -> certified Select-10.

This is a GT oracle on a fixed diagnostic subset or full VAL571. No
inference-safe selector, new weights or conditional-distribution claims.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from coreset_reactor.certified_select import solve_select, geometry, selection_value, extension, quality_feasible
from coreset_reactor.evaluate import official_prediction
from coreset_reactor.fast_frc import candidate_frc
from coreset_reactor.oracle_select import Candidate, SharedQualityPool
from coreset_reactor.student_spread_probe import expand_predictions
from coreset_reactor.teacher_cache import sha256_file
from framework.metrics import compute_s_mse, compute_FRVar
from framework.metrics.FRC import _func

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'coreset_reactor/reports/fixed_teacher_distillation'
SOURCES = [('basic',RUNS/'t10_student_pilot10e/all_session_val571_cached.json'),
           ('distance',RUNS/'t10_student_rel_pilot10e/all_session_val571_fast.json'),
           ('direction',RUNS/'t10_student_direction_pilot10e/all_session_val571_fast.json')]


def atomic_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def build_pool(soft, labels):
    """Keep the original ten baseline instances; merge all other exact copies."""
    candidates,origins,lookup=[],[],{}
    raw_count=None
    for scale in (1.,1.5):
        for model,(pred,label) in enumerate(zip(soft,labels)):
            official=official_prediction(expand_predictions(pred,scale)).contiguous()
            for slot,trajectory in enumerate(official):
                digest=hashlib.sha256(trajectory.numpy().tobytes()).hexdigest()
                origin={'model':label,'slot':slot,'scale':scale}
                if digest in lookup and not (model==0 and scale==1):
                    origins[lookup[digest]].append(origin)
                else:
                    lookup.setdefault(digest,len(candidates))
                    candidates.append(trajectory)
                    origins.append([origin])
        if scale==1:
            raw_count=len(candidates)
    return torch.stack(candidates),origins,raw_count,len(lookup)


def write_report(output,config,records):
    scope = 'full VAL571' if config['contexts']==571 else f"VAL subset/shard ({config['contexts']})"
    lines=[f'# Quality-constrained Select-10: {scope}','',
           '**GT oracle only; no inference method or newly trained model.**','',
           f"Completed {len(records)}/{config['contexts']} fixed VAL contexts; seed={config['eval_seed']}.",
           'Baseline: Basic-KD student, ten original predictions. Center750; all-session opposite-role GT.',
           'Per-context FRC >= baseline and exact FRD <= baseline; no policy slack.',
           'R30: original predictions from Basic/Distance/Direction students; E60 adds scale1.5 outputs.',
           'These are separate pools. Not the earlier CoReSet M1/route-checkpoint experiment.',
           'Equal official trajectories are deduplicated except baseline instances needed to preserve its ten-output contract.',
           '', '| Pool | Mean baseline FRDiv | Mean feasible LB | Mean certified UB | Mean gap | FRC baseline / selected | FRD baseline / selected |',
           '|---|---:|---:|---:|---:|---:|---:|']
    summary={}
    for name in ('raw30','expanded60'):
        rr=[r[name] for r in records]
        if not rr:
            continue
        baseline={k:float(np.mean([r['baseline'][k] for r in rr])) for k in ('FRC','exact_FRD','FRDiv')}
        selected={k:float(np.mean([r['selected_metrics'][k] for r in rr])) for k in ('FRC','exact_FRD','FRDiv')}
        upper=float(np.mean([r['upper_bound'] for r in rr])); lower=selected['FRDiv']
        summary[name]={'contexts':len(rr),'baseline':baseline,'selected':selected,
                       'upper_bound_mean':upper,'absolute_gap_mean':upper-lower,
                       'unique_pool_size_mean':float(np.mean([r['pool_size'] for r in rr])),
                       'bounds_width_quantiles':np.quantile([r['absolute_gap'] for r in rr],[0,.5,.9,1]).tolist(),
                       'selected_frdiv_quantiles':np.quantile([r['lower_bound'] for r in rr],[0,.5,.9,1]).tolist(),
                       'upper_frdiv_quantiles':np.quantile([r['upper_bound'] for r in rr],[0,.5,.9,1]).tolist(),
                       'positive_gain_contexts':sum(r['lower_bound']>r['baseline']['FRDiv']+1e-8 for r in rr),
                       'selected_canonical_composition':dict(sum((Counter(r['canonical_source_counts']) for r in rr),Counter()))}
        lines.append(f"| {name} | {baseline['FRDiv']:.6f} | {lower:.6f} | {upper:.6f} | {upper-lower:.6f} | "
                     f"{baseline['FRC']:.6f} / {selected['FRC']:.6f} | {baseline['exact_FRD']:.6f} / {selected['exact_FRD']:.6f} |")
    lines+=['','## Interpretation','',
            '- LB is an actual ten-trajectory set passing both exact-quality checks. It is never a fractional output.',
            '- UB comes from a concave tangent and an independently evaluated LP dual bound, not the optimizer primal value.',
            '- Float64 numerical allowance is reported per context; this is not formal interval-arithmetic certification.',
            '- A wide gap means the optimum remains unresolved. Low LB alone does not prove generator insufficiency.',
            '- Bounds apply only to this finite pool and these per-context constraints; not all possible reactions,',
            '  not a looser dataset-average-quality constraint. Partial reports cover completed contexts only.',
            '- GT defines candidate quality only for this oracle. No inference-time quality guarantee is claimed.',
            '- Full distributions, source composition, solver status and hashes are in summary.json and per-context JSON.',
            '- Numeric feasibility tolerances: FRC1e-10, FRD1e-8; these are not percentage quality slack.',
            '- No training: temporal-loss budget 0. Cached predictions: no fresh neural inference in this run.',
            '  Solver/metric wall times are measured separately; this is not an end-to-end deployment latency benchmark.']
    atomic_json(output/'summary.json',{'config':config,'completed':len(records),'pools':summary,'per_context':records})
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def validate_reuse_config(previous,current):
    """Only orchestration/subset can differ; numerical solver and inputs freeze."""
    for key in ('checkpoint_sources','eval_seed','metric_workers','mip_seconds',
                'baseline','primary_pool','analysis_pool','quality_policy'):
        if previous[key]!=current[key]:
            raise ValueError('Reuse incompatibility: '+key)
    for path,digest in previous['source_hashes'].items():
        if path=='coreset_reactor/run_certified_oracle.py':
            continue  # orchestration now supports shards/reuse; validate actual sets below
        if current['source_hashes'].get(path)!=digest:
            raise ValueError('Numerical source changed: '+path)


def validate_cached_record(record,trajectories,origins,raw_count,frc,frd):
    """Recompute feasibility, diversity and the saved upper-bound expression."""
    if record['candidate_origins']!=origins or frc.shape!=(len(trajectories),) or frd.shape!=frc.shape:
        raise ValueError('Reuse candidate order/shape mismatch')
    if not np.isfinite(frc).all() or not np.isfinite(frd).all():
        raise ValueError('Nonfinite cached quality')
    for name,size in (('raw30',raw_count),('expanded60',len(trajectories))):
        result=record[name]; selected=result['selected']; base=list(range(10))
        if result['pool_size']!=size or not quality_feasible(selected,frc[:size],frd[:size],base):
            raise ValueError('Reused selection not quality feasible')
        p=trajectories[:size]
        points=p.flatten(1).double().numpy()/np.sqrt(p.shape[1]*p.shape[2])
        gram,norms,distance=geometry(points)
        for field,ids in (('baseline',base),('selected_metrics',selected)):
            recomputed={'FRC':float(frc[ids].sum()),'exact_FRD':float(frd[ids].sum()),
                        'FRDiv':selection_value(distance,ids)}
            for key,value in recomputed.items():
                np.testing.assert_allclose(result[field][key],value,rtol=0,atol=1e-9)
        cert=result['certificate']; x=np.array(cert['tangent_point'])
        value,g=extension(x,gram,norms,10)
        intercept=value-g@x
        if cert['kind']=='cardinality_tangent':
            raw=intercept+np.sort(g)[-10:].sum()
        elif cert['kind']=='quality_lp_dual_tangent':
            thresholds=np.array([frc[base].sum(),frd[base].sum()])
            scales=np.maximum(1,np.abs(thresholds))
            matrix=np.stack([-frc[:size],frd[:size]])/scales[:,None]
            bound=np.array([-thresholds[0],thresholds[1]])/scales
            lam=np.array(cert['lambda']); nu=cert['nu']
            if np.any(lam<0):
                raise ValueError('Invalid cached dual multipliers')
            raw=intercept+lam@bound+nu*10+np.maximum(g-matrix.T@lam-nu,0).sum()
        else:
            raise ValueError('Unknown certificate')
        np.testing.assert_allclose(result['upper_bound'],raw+cert['numeric_allowance'],rtol=0,atol=1e-9)
        if result['upper_bound']+1e-10<result['lower_bound']:
            raise ValueError('Invalid cached bound interval')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--contexts',type=int,default=24)
    parser.add_argument('--eval-seed',type=int,default=1234)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--mip-seconds',type=float,default=5.)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--context-start',type=int,default=0)
    parser.add_argument('--context-stop',type=int)
    parser.add_argument('--reuse-dir',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    if not 1<=args.contexts<=571:
        raise ValueError('Need 1..571 contexts')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Choose a new output or explicit --resume')
    args.output.mkdir(parents=True,exist_ok=True)
    values=[]; predictions=[]; provenance=[]
    for label,path in SOURCES:
        data=json.loads(path.read_text())
        pred_path=path.with_suffix('.predictions.pt')
        checkpoint=Path(data['checkpoint'])
        digest=sha256_file(checkpoint)
        if digest!=data['checkpoint_sha256']:
            raise ValueError(f'Frozen checkpoint mismatch: {label}')
        payload=torch.load(pred_path,map_location='cpu',weights_only=False)
        if payload['checkpoint_sha256']!=digest or payload['clip_ids']!=[r['clip_id'] for r in data['rows']]:
            raise ValueError('Prediction provenance mismatch')
        if values:
            for key in ('clip_id','target_cache_key','gt_count'):
                if [r[key] for r in values[0]['rows']]!=[r[key] for r in data['rows']]:
                    raise ValueError('Sources are not matched on '+key)
        if len(data['rows'])!=571:
            raise ValueError('Expected full571 cache')
        values.append(data); predictions.append(payload['soft_predictions'])
        provenance.append({'label':label,'source':str(path),'source_sha256':sha256_file(path),
                           'checkpoint':str(checkpoint),'checkpoint_sha256':digest,
                           'prediction_sha256':sha256_file(pred_path),
                           'parameters':data['speed']['parameters']})
    all_indices=sorted(np.random.default_rng(args.eval_seed).choice(571,args.contexts,replace=False).tolist())
    stop=args.contexts if args.context_stop is None else args.context_stop
    if not 0<=args.context_start<stop<=args.contexts:
        raise ValueError('Invalid context shard range')
    indices=all_indices[args.context_start:stop]
    config={'contexts':len(indices),'population_contexts':args.contexts,
            'context_start':args.context_start,'context_stop':stop,
            'reuse_dir':str(args.reuse_dir.resolve()) if args.reuse_dir else None,
            'eval_seed':args.eval_seed,'indices':indices,
            'clip_ids':[values[0]['rows'][i]['clip_id'] for i in indices],
            'checkpoint_sources':provenance,'metric_workers':args.workers,'mip_seconds':args.mip_seconds,
            'git_commit_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_hashes':{p:sha256_file(ROOT/p) for p in (
                'coreset_reactor/certified_select.py','coreset_reactor/run_certified_oracle.py',
                'coreset_reactor/oracle_select.py','coreset_reactor/fast_frc.py',
                'coreset_reactor/student_spread_probe.py','mam_reactor/tools/compute_frd_resumable.py')},
            'baseline':'basic KD frozen outputs, not historical CoReSet M1',
            'primary_pool':'three raw frozen students (30)', 'analysis_pool':'raw + scale1.5 (60)',
            'quality_policy':'per-context zero slack FRC and exact FRD',
            'not_inference_safe':True,'new_training':False}
    if (args.output/'config.json').exists():
        if json.loads((args.output/'config.json').read_text())!=config:
            raise ValueError('Resume config/provenance mismatch')
    else:
        atomic_json(args.output/'config.json',config)
    cache=RUNS/'val571_postprocessed_gt_cache'
    # Validate and adopt the existing pilot before starting any remaining work.
    if args.reuse_dir:
        previous=json.loads((args.reuse_dir/'config.json').read_text())
        validate_reuse_config(previous,config)
        for index in indices:
            source=args.reuse_dir/f'context_{index:04d}.json'
            destination=args.output/source.name
            if not source.exists() or destination.exists():
                continue
            record=json.loads(source.read_text()); row=values[0]['rows'][index]
            if record['clip_id']!=row['clip_id'] or record['target_cache_key']!=row['target_cache_key']:
                raise ValueError('Reused GT context mismatch')
            target_file=cache/f"{row['target_cache_key']}.pt"
            if sha256_file(target_file)!=record['gt_cache_sha256']:
                raise ValueError('Reused GT tensor changed')
            p,origins,raw_count,_=build_pool([v[index] for v in predictions],[v[0] for v in SOURCES])
            with np.load(args.reuse_dir/f'quality_{index:04d}.npz',allow_pickle=False) as quality:
                frc,frd=quality['frc'],quality['frd']
                if quality['target_cache_key'].item()!=row['target_cache_key']:
                    raise ValueError('Reused quality cache identity mismatch')
            target=torch.load(target_file,map_location='cpu',weights_only=False)['targets']
            np.testing.assert_allclose(candidate_frc(target.numpy(),p.numpy()),frc,rtol=1e-5,atol=1e-6)
            validate_cached_record(record,p,origins,raw_count,frc,frd)
            import shutil
            shutil.copyfile(args.reuse_dir/f'quality_{index:04d}.npz',args.output/f'quality_{index:04d}.npz')
            record['reused_from']={'directory':str(args.reuse_dir.resolve()),'record_sha256':sha256_file(source)}
            atomic_json(destination,record)
    records=[]; pool=None
    for index in indices:
        file=args.output/f'context_{index:04d}.json'
        if file.exists():
            record=json.loads(file.read_text())
            if record['clip_id']!=values[0]['rows'][index]['clip_id']:
                raise ValueError('Resume context mismatch')
            records.append(record)
    done={r['val_index'] for r in records}
    write_report(args.output,config,records)
    print(json.dumps({'reused_or_resumed':len(done),'shard_total':len(indices)}),flush=True)
    try:
        for position,index in enumerate(indices):
            result_file=args.output/f'context_{index:04d}.json'
            row=values[0]['rows'][index]
            if index in done:
                continue
            soft=[p[index] for p in predictions]
            trajectories,origins,raw_count,distinct=build_pool(soft,[v[0] for v in SOURCES])
            if any(p.shape!=soft[0].shape for p in soft) or soft[0].shape[0]!=10:
                raise ValueError('Ten-output/length contract mismatch')
            target_file=cache/f"{row['target_cache_key']}.pt"
            payload=torch.load(target_file,map_location='cpu',weights_only=False)
            target=payload['targets']
            if payload['fingerprint']!=row['target_cache_key'] or target.shape!=(row['gt_count'],soft[0].shape[1],25):
                raise ValueError('GT cache mismatch')
            if not torch.isfinite(target).all():
                raise ValueError('GT contains nonfinite values')
            quality_file=args.output/f'quality_{index:04d}.npz'
            metric_seconds=0.
            if quality_file.exists():
                with np.load(quality_file,allow_pickle=False) as stored:
                    frc,frd=stored['frc'],stored['frd']
                    if stored['target_cache_key'].item()!=row['target_cache_key']:
                        raise ValueError('Saved quality cache mismatch')
            else:
                if pool is None:
                    pool=SharedQualityPool(args.workers,60,750,max_targets=max(r['gt_count'] for r in values[0]['rows']))
                started=time.perf_counter()
                frc=candidate_frc(target.numpy(),trajectories.numpy())
                if position==0:
                    reference=np.array([float(_func(target,p[None])) for p in trajectories[:10]])
                    np.testing.assert_allclose(frc[:10],reference,rtol=1e-5,atol=1e-6)
                candidates=[Candidate(p,str(i),'certified_pool',[(str(i),i)]) for i,p in enumerate(trajectories)]
                legacy_frc,frd=pool.score(candidates,target)
                np.testing.assert_allclose(frc,legacy_frc,rtol=1e-5,atol=1e-6)
                metric_seconds=time.perf_counter()-started
                temporary=quality_file.with_suffix('.tmp.npz')
                np.savez(temporary,frc=frc,frd=frd,target_cache_key=row['target_cache_key'])
                temporary.replace(quality_file)
            np.testing.assert_allclose(frc[:10],row['frc'],rtol=1e-5,atol=1e-6)
            record={'clip_id':row['clip_id'],'val_index':index,'gt_count':row['gt_count'],
                    'target_cache_key':row['target_cache_key'],'gt_cache_sha256':sha256_file(target_file),
                    'metric_seconds':metric_seconds,'unique_trajectory_count':distinct,
                    'candidate_origins':origins}
            for name,size in (('raw30',raw_count),('expanded60',len(trajectories))):
                p=trajectories[:size]
                points=p.flatten(1).double().numpy()/np.sqrt(p.shape[1]*p.shape[2])
                solved=solve_select(points,frc[:size],frd[:size],list(range(10)),args.mip_seconds)
                selected=solved['selected']
                np.testing.assert_allclose(solved['lower_bound'],float(compute_s_mse([p[selected]])),rtol=1e-5,atol=1e-6)
                solved['selected_metrics']['FRVar']=float(compute_FRVar([p[selected]]))
                solved['pool_size']=size
                solved['canonical_source_counts']=dict(Counter(origins[i][0]['model'] for i in selected))
                solved['canonical_scale_counts']=dict(Counter(str(origins[i][0]['scale']) for i in selected))
                solved['selected_origins']=[origins[i] for i in selected]
                record[name]=solved
            if record['expanded60']['upper_bound']+1e-8<record['raw30']['lower_bound']:
                raise RuntimeError('Expanded-pool upper bound excludes valid raw-pool solution')
            atomic_json(result_file,record)
            records.append(record); records.sort(key=lambda r:r['val_index'])
            write_report(args.output,config,records)
            print(json.dumps({'completed':len(records),'total':len(indices),'val_index':index,
                              **{n:{k:record[n][k] for k in ('lower_bound','upper_bound','absolute_gap')} for n in ('raw30','expanded60')}}),flush=True)
    finally:
        if pool is not None:
            pool.close()
    for item in provenance:
        if sha256_file(Path(item['checkpoint']))!=item['checkpoint_sha256']:
            raise RuntimeError('Frozen checkpoint changed')
    print('Requested subset/shard complete; all checkpoint hashes unchanged.',flush=True)


if __name__=='__main__':
    main()
