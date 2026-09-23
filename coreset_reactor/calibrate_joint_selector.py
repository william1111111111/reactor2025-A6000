"""TRAIN-session calibration of a frozen scorer and joint replacement policy."""
import json
import time
import subprocess
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from coreset_reactor.train_inference_safe_selector import (
    load_train_shards, load_val_sources, load_val_quality, make_pairs_with_length)
from coreset_reactor.inference_safe_selector import trajectory_summary, balanced_session_folds, atomic_json
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.run_certified_oracle import RUNS
from coreset_reactor.certified_select import selection_value
from coreset_reactor.teacher_cache import sha256_file

BASE=list(range(10))
OLD=np.repeat(np.arange(10),50)
NEW=np.tile(np.arange(10,60),10)
LEGAL=(OLD[:,None]!=OLD[None,:]) & (NEW[:,None]!=NEW[None,:])


def select(pred,distance,eps):
    """GT-free, exact diversity objective among <=2 predicted-feasible swaps."""
    gain=2/90*(distance[NEW,:10].sum(1)-distance[NEW,OLD]-distance[OLD,:10].sum(1))
    feasible=(pred[:,0]>=eps[0]) & (pred[:,1]<=-eps[1]) & (gain>1e-12)
    selected=BASE
    best=0.
    if feasible.any():
        a=int(np.argmax(np.where(feasible,gain,-np.inf)))
        best=gain[a]
        selected=[k for k in BASE if k!=OLD[a]]+[int(NEW[a])]
    cross=2/90*(distance[NEW[:,None],NEW[None,:]]-distance[NEW[:,None],OLD[None,:]]
                -distance[OLD[:,None],NEW[None,:]]+distance[OLD[:,None],OLD[None,:]])
    feasible=LEGAL & (pred[:,0,None]+pred[None,:,0]>=2*eps[0]) & (pred[:,1,None]+pred[None,:,1]<=-2*eps[1])
    gains=np.where(feasible,gain[:,None]+gain[None,:]+cross,-np.inf)
    a,b=np.unravel_index(np.argmax(gains),gains.shape)
    if gains[a,b]>best+1e-12:
        selected=[k for k in BASE if k not in (OLD[a],OLD[b])]+[int(NEW[a]),int(NEW[b])]
    assert len(selected)==10 and len(set(selected))==10
    assert selection_value(distance,selected)>=selection_value(distance,BASE)-1e-10
    return selected


def audit(selected,y,distance):
    c,d=y[:,0],y[:,1]
    dc=float(c[selected].sum()-c[:10].sum())
    dd=float(d[selected].sum()-d[:10].sum())
    return dict(selected=selected,FRC=float(c[selected].sum()),exact_FRD=float(d[selected].sum()),
        FRDiv=selection_value(distance,selected),delta_FRC=dc,delta_FRD=dd,
        gain=selection_value(distance,selected)-selection_value(distance,BASE),
        strict=bool(dc>=-1e-10 and dd<=1e-8),changed=selected!=BASE)


def aggregate(rows):
    result={k:float(np.mean([r[k] for r in rows])) for k in ('FRC','exact_FRD','FRDiv','delta_FRC','delta_FRD','gain')}
    result.update(contexts=len(rows),strict=sum(r['strict'] for r in rows),changed=sum(r['changed'] for r in rows))
    return result


def dataset(split):
    return PairedReactionDataset(Path('/public/zhaowenjie/react2025_new/data'),split,750,crop_mode='center',
        normalization_dir=Path('/home/zhaowenjie/react2025_new/mam_reactor/external/FaceVerse'))


def main():
    torch.set_num_threads(2)
    output=RUNS/'joint_selector_train_calibration_v1'
    output.mkdir(exist_ok=False)
    started=time.monotonic()
    soft,paths,clips,manifests=load_train_shards([])
    data=dataset('train')
    sessions=[Path(clip).parts[1] for clip in clips]
    counts=Counter(sessions)
    groups=balanced_session_folds(sorted(counts),counts,5)
    calibration=set(groups[0])
    rng=np.random.default_rng(20260921)
    xs,ys,heldout=[],[],[]
    for i in range(len(data)):
        sample=data[i]
        assert clips[i]==sample['clip_id']
        length=int(sample['source_lengths'])
        speaker=trajectory_summary(sample['speaker_emotion'][:length].numpy())
        _,_,dist,x,_=make_pairs_with_length([v[i] for v in soft],speaker,length)
        with np.load(paths[i]) as q:
            assert str(q['clip_id'])==clips[i]
            y=np.stack((q['frc'],q['frd']),1)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        if sessions[i] in calibration:
            heldout.append((clips[i],x,y,dist))
        else:
            chosen=rng.choice(500,150,replace=False)
            xs.append(x[chosen]); ys.append((y[NEW]-y[OLD])[chosen])
        if (i+1)%200==0: print('TRAIN features',i+1,flush=True)
    del soft
    x,y=np.concatenate(xs),np.concatenate(ys)
    mean,std=y.mean(0),y.std(0).clip(min=1e-6)
    model=ExtraTreesRegressor(n_estimators=96,min_samples_leaf=12,max_features=.75,max_depth=24,n_jobs=8,random_state=20260921)
    print('Fit frozen scorer',x.shape,flush=True)
    fit_start=time.monotonic()
    model.fit(x,(y-mean)/std)
    fit_seconds=time.monotonic()-fit_start
    preds=[model.predict(x)*std+mean for _,x,_,_ in heldout]
    residual=np.concatenate([np.abs(y[NEW]-y[OLD]-p) for (_,_,y,_),p in zip(heldout,preds)])
    margins={'zero':np.zeros(2)}
    for q in (.25,.5,.75,.9,.95,.99): margins[f'q{q:g}']=np.quantile(residual,q,axis=0)
    cal={name:[] for name in [*margins,'baseline']}
    for (_,_,y,dist),pred in zip(heldout,preds):
        for name,eps in margins.items(): cal[name].append(audit(select(pred,dist,eps),y,dist))
        cal['baseline'].append(audit(BASE,y,dist))
    summaries={name:aggregate(rows) for name,rows in cal.items()}
    def choose(predicate):
        eligible=[name for name,s in summaries.items() if predicate(s)]
        return max(eligible,key=lambda name:(summaries[name]['gain'],-summaries[name]['changed']))
    chosen={'cal_zero_violations':choose(lambda s:s['strict']==s['contexts']),
            'cal_average_quality':choose(lambda s:s['delta_FRC']>=-1e-10 and s['delta_FRD']<=1e-8),
            'cal_95pct_pass':choose(lambda s:s['strict']/s['contexts']>=.95)}
    # Freeze decisions and serialized scorer before opening VAL for this run.
    torch.save(dict(model=model,mean=mean,std=std,margins=margins,chosen=chosen),output/'selector.pt')
    protocol=dict(chosen=chosen,calibration=summaries,calibration_sessions=sorted(calibration),
        fit_contexts=len(xs),calibration_contexts=len(heldout),seed=20260921,
        scorer_sha256=sha256_file(output/'selector.pt'),fit_seconds=fit_seconds,
        script_sha256=sha256_file(Path(__file__)),git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        limitations=['TRAIN trajectory cache is float16; quality labels predate quantization.',
                    'Empirical policy tuning, not formal coverage guarantee; generator trained on TRAIN.',
                    'VAL571 is reused development data, center750/all-session protocol.'])
    atomic_json(output/'frozen_protocol.json',protocol)
    print('TRAIN policy choices',chosen,flush=True)
    valsoft,valclips=load_val_sources(); paths=load_val_quality(); data=dataset('val')
    rows=[]; inference_seconds=0.
    for i in range(len(data)):
        sample=data[i]; assert sample['clip_id']==valclips[i]
        length=int(sample['source_lengths'])
        speaker=trajectory_summary(sample['speaker_emotion'][:length].numpy())
        t=time.monotonic()
        _,_,dist,x,_=make_pairs_with_length([v[i] for v in valsoft],speaker,length)
        pred=model.predict(x)*std+mean
        selections={'baseline':BASE,'zero_margin':select(pred,dist,margins['zero'])}
        for label,name in chosen.items(): selections[label]=BASE if name=='baseline' else select(pred,dist,margins[name])
        inference_seconds+=time.monotonic()-t
        with np.load(paths[i]) as q: y=np.stack((q['frc'],q['frd']),1)
        rows.append(dict(clip_id=valclips[i],policies={name:audit(ids,y,dist) for name,ids in selections.items()}))
        if (i+1)%100==0: print('VAL',i+1,flush=True)
    summary={name:aggregate([row['policies'][name] for row in rows]) for name in rows[0]['policies']}
    atomic_json(output/'summary.json',dict(protocol=protocol,policies=summary,records=rows,
        elapsed_seconds=time.monotonic()-started,feature_scoring_selection_seconds_per_context=inference_seconds/len(rows)))
    lines=['# Frozen scorer with TRAIN session policy calibration','','Scorer is never refit after calibration. Policy choices were saved before VAL evaluation. Empirical TRAIN tuning does not guarantee per-context quality on VAL.','','| Policy | FRC | exact FRD | FRDiv | Strict contexts | Changed |','|---|---:|---:|---:|---:|---:|']
    for name,s in summary.items(): lines.append(f"| {name} | {s['FRC']:.6f} | {s['exact_FRD']:.6f} | {s['FRDiv']:.6f} | {s['strict']}/{s['contexts']} | {s['changed']} |")
    lines+=['','TRAIN choices: '+json.dumps(chosen),'',*protocol['limitations']]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
