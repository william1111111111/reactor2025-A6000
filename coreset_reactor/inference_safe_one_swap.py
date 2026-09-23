"""Session-grouped inference-only one-swap quality-delta selector."""
from __future__ import annotations
import argparse, json, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor

from coreset_reactor.certified_select import geometry, selection_value
from coreset_reactor.inference_safe_selector import atomic_json, balanced_session_folds, trajectory_summary
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.run_certified_oracle import RUNS, SOURCES, build_pool


def compact(summary):
    # mean, std and mean absolute velocity from the fixed 275-D summary.
    return summary[np.r_[0:50,100:125]]


def pair_features(candidate, source, metadata, distance):
    result=[]; pairs=[]; base=list(range(10))
    base_value=selection_value(distance,base)
    for old in base:
        for new in range(10,60):
            trial=[i for i in base if i!=old]+[new]
            gain=selection_value(distance,trial)-base_value
            a,b=compact(candidate[new]),compact(candidate[old])
            result.append(np.r_[a,b,a-b,compact(source),metadata[new],metadata[old],
                                metadata[new]-metadata[old],gain])
            pairs.append((old,new,gain))
    result=np.asarray(result,dtype=np.float32)
    if result.shape!=(500,346): raise AssertionError(result.shape)
    return result,pairs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--trees',type=int,default=128)
    p.add_argument('--jobs',type=int,default=8);p.add_argument('--seed',type=int,default=20260921)
    p.add_argument('--train-pairs-per-context',type=int,default=150)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True); started=time.monotonic()
    values=[];soft=[]
    for _,path in SOURCES:
        r=json.loads(path.read_text());t=torch.load(path.with_suffix('.predictions.pt'),map_location='cpu',weights_only=False)
        if t['checkpoint_sha256']!=r['checkpoint_sha256']: raise ValueError('provenance mismatch')
        values.append(r);soft.append(t['soft_predictions'])
    clips=[r['clip_id'] for r in values[0]['rows']]
    data=PairedReactionDataset(Path('/public/zhaowenjie/react2025_new/data'),'val',750,crop_mode='center',
        normalization_dir=Path('/home/zhaowenjie/react2025_new/mam_reactor/external/FaceVerse'))
    oracle=RUNS/'certified_select10_val571_seed1234'; quality={}
    for s in oracle.glob('shard_*'):
        for q in s.glob('quality_*.npz'): quality[int(q.stem.split('_')[1])]=q
    if set(quality)!=set(range(571)): raise ValueError('quality cache incomplete')
    features=[];pairs=[];targets=[];qualities=[];distances=[];origins=[];sessions=[]
    models=('basic','distance','direction')
    for i in range(571):
        sample=data[i]
        if sample['clip_id']!=clips[i]:raise ValueError('order mismatch')
        length=int(sample['source_lengths']); speaker=trajectory_summary(sample['speaker_emotion'][:length].numpy())
        pool,org,raw,_=build_pool([v[i] for v in soft],[v[0] for v in SOURCES])
        if len(pool)!=60 or raw!=30:raise ValueError('E60 contract changed')
        cand=np.stack([trajectory_summary(x.numpy()) for x in pool]); meta=np.zeros((60,15),np.float32)
        for j,o in enumerate(org):
            z=o[0];meta[j,models.index(z['model'])]=1;meta[j,3+z['slot']]=1;meta[j,13]=z['scale']==1.5;meta[j,14]=length/750
        points=pool.flatten(1).double().numpy()/np.sqrt(pool.shape[1]*pool.shape[2]);dist=geometry(points)[2]
        f,pr=pair_features(cand,speaker,meta,dist)
        with np.load(quality[i],allow_pickle=False) as q: y=np.stack((q['frc'],q['frd']),1)
        dy=np.array([[y[new,0]-y[old,0],y[new,1]-y[old,1]] for old,new,_ in pr])
        features.append(f);pairs.append(pr);targets.append(dy);qualities.append(y);distances.append(dist);origins.append(org);sessions.append(sample['session_id'])
    X,Y=np.stack(features),np.stack(targets); counts=Counter(sessions)
    groups=balanced_session_folds(sorted(counts),counts,5)
    fold=np.array([next(k for k,g in enumerate(groups) if s in g) for s in sessions])
    pred=np.full_like(Y,np.nan);eps={};fold_info=[]
    for test in range(5):
        cal=(test+1)%5;tr=np.flatnonzero((fold!=test)&(fold!=cal));ca=np.flatnonzero(fold==cal);te=np.flatnonzero(fold==test)
        rng=np.random.default_rng(args.seed+test)
        chosen=np.stack([rng.choice(500,args.train_pairs_per_context,replace=False) for _ in tr])
        tx=np.concatenate([X[c,chosen[n]] for n,c in enumerate(tr)]);ty=np.concatenate([Y[c,chosen[n]] for n,c in enumerate(tr)])
        mean,std=ty.mean(0),ty.std(0);m=ExtraTreesRegressor(n_estimators=args.trees,min_samples_leaf=12,max_features=.75,max_depth=24,n_jobs=args.jobs,random_state=args.seed+test)
        m.fit(tx,(ty-mean)/std)
        cp=m.predict(X[ca].reshape(-1,346))*std+mean;tp=m.predict(X[te].reshape(-1,346))*std+mean
        pred[te]=tp.reshape(len(te),500,2);res=np.abs(Y[ca].reshape(-1,2)-cp)
        eps[test]={str(q):np.quantile(res,q,axis=0).tolist()
                   for q in (.25,.3,.35,.4,.45,.5,.75,.9,.95,.99)}
        fold_info.append({'fold':test,'test_sessions':groups[test],'calibration_sessions':groups[cal],
                          'train_contexts':len(tr),'test_contexts':len(te)})
        print(json.dumps(fold_info[-1]),flush=True)
    if not np.isfinite(pred).all():raise ValueError('missing predictions')
    policies={'margin_0':None,'margin_q25':'0.25','margin_q30':'0.3','margin_q35':'0.35',
              'margin_q40':'0.4','margin_q45':'0.45','margin_q50':'0.5',
              'margin_q75':'0.75','margin_q90':'0.9','margin_q95':'0.95','margin_q99':'0.99'}
    records=[]
    for i in range(571):
        y=qualities[i];base=list(range(10));bm={'FRC':float(y[:10,0].sum()),'exact_FRD':float(y[:10,1].sum()),'FRDiv':selection_value(distances[i],base)}
        item={'clip_id':clips[i],'session':sessions[i],'fold':int(fold[i]),'baseline':bm,'policies':{}}
        for name,q in policies.items():
            e=np.zeros(2) if q is None else np.asarray(eps[int(fold[i])][q])
            feasible=np.flatnonzero((pred[i,:,0]-e[0]>=0)&(pred[i,:,1]+e[1]<=0)&(X[i,:,-1]>0))
            # Final X column is diversity gain. Deterministic baseline fallback.
            if len(feasible):
                chosen=int(feasible[np.argmax(X[i,feasible,-1])]);old,new,gain=pairs[i][chosen];selected=[k for k in base if k!=old]+[new]
            else:selected=base;old=new=None
            frc=float(y[selected,0].sum());frd=float(y[selected,1].sum());div=selection_value(distances[i],selected)
            item['policies'][name]={'selected':selected,'old':old,'new':new,'FRC':frc,'exact_FRD':frd,'FRDiv':div,
                'strict_quality_pass':bool(frc>=bm['FRC']-1e-10 and frd<=bm['exact_FRD']+1e-8),'eps_FRC':float(e[0]),'eps_FRD':float(e[1])}
        records.append(item)
        if (i+1)%25==0:atomic_json(args.output/'progress.json',{'completed':i+1,'total':571})
    def agg(name):
        s=[r['policies'][name] for r in records];b=[r['baseline'] for r in records]
        sm={k:float(np.mean([x[k] for x in s])) for k in ('FRC','exact_FRD','FRDiv')};bm={k:float(np.mean([x[k] for x in b])) for k in sm}
        changed=[x['new'] is not None for x in s]
        return {'baseline':bm,'selected':sm,'delta':{k:sm[k]-bm[k] for k in sm},'changed_contexts':sum(changed),
                'strict_quality_pass_contexts':sum(x['strict_quality_pass'] for x in s),
                'positive_FRDiv_contexts':sum(x['FRDiv']>y['FRDiv']+1e-12 for x,y in zip(s,b))}
    summary={'scope':'VAL571 session-grouped five-fold one-swap cross-fit; center750/all-session; not TEST',
             'inference_inputs':'speaker/candidate summaries, candidate provenance and prediction-only diversity gain; no GT/session/clip identity',
             'quality_prediction':{'delta_FRC_spearman':float(spearmanr(Y[:,:,0].ravel(),pred[:,:,0].ravel()).statistic),
                                   'delta_FRD_spearman':float(spearmanr(Y[:,:,1].ravel(),pred[:,:,1].ravel()).statistic),
                                   'delta_FRC_MAE':float(np.abs(Y[:,:,0]-pred[:,:,0]).mean()),
                                   'delta_FRD_MAE':float(np.abs(Y[:,:,1]-pred[:,:,1]).mean())},
             'folds':fold_info,'calibration_eps':eps,'policies':{k:agg(k) for k in policies},
             'elapsed_seconds':time.monotonic()-started,
             'limitations':['development-set cross-fit','one replacement maximum','empirical margins are not formal guarantees','matched center750 protocol']}
    atomic_json(args.output/'summary.json',summary);atomic_json(args.output/'records.json',{'records':records})
    lines=['# Inference-safe one-swap selector','',summary['scope'],'','| Policy | FRC | exact FRD | FRDiv | strict pass | changed |','|---|---:|---:|---:|---:|---:|']
    b=summary['policies']['margin_0']['baseline'];lines.append(f"| baseline | {b['FRC']:.6f} | {b['exact_FRD']:.6f} | {b['FRDiv']:.6f} | 571/571 | 0 |")
    for name in policies:
        a=summary['policies'][name];v=a['selected'];lines.append(f"| {name} | {v['FRC']:.6f} | {v['exact_FRD']:.6f} | {v['FRDiv']:.6f} | {a['strict_quality_pass_contexts']}/571 | {a['changed_contexts']} |")
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
