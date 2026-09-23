"""Audit cached one-swap policies and exact one-swap oracle; no training."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from coreset_reactor.train_inference_safe_selector import load_val_sources, load_val_quality, make_pairs_with_length
from coreset_reactor.inference_safe_selector import trajectory_summary
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.run_certified_oracle import RUNS, build_pool
from coreset_reactor.certified_select import geometry, selection_value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--learned',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    root = RUNS / 'train_selector_e60_center750_val571_v2'
    records = json.loads((root/'records.json').read_text())['records']
    soft, clips = load_val_sources()
    qualities = load_val_quality()
    if args.learned:
        scorer=torch.load(root/'selector.pt',map_location='cpu',weights_only=False)
        scorer['model'].n_jobs=8
        dataset=PairedReactionDataset(Path('/public/zhaowenjie/react2025_new/data'),'val',750,crop_mode='center',normalization_dir=Path('/home/zhaowenjie/react2025_new/mam_reactor/external/FaceVerse'))
    rows = []
    for i, record in enumerate(records):
        assert record['clip_id'] == clips[i]
        pool, _, _, _ = build_pool([x[i].float() for x in soft], ['basic','distance','direction'])
        distance = geometry(pool.flatten(1).double().numpy()/np.sqrt(pool.shape[1]*25))[2]
        with np.load(qualities[i]) as q:
            c, d = q['frc'], q['frd']
        base = list(range(10))
        best = base
        value = selection_value(distance, base)
        np.testing.assert_allclose(value, record['baseline']['FRDiv'], atol=1e-9)
        for old in base:
            for new in range(10, len(pool)):
                trial = [k for k in base if k != old]+[new]
                if c[new]-c[old] >= -1e-10 and d[new]-d[old] <= 1e-8:
                    gain = selection_value(distance, trial)
                    if gain > value+1e-12:
                        best, value = trial, gain
        policies = {}
        for name, policy in record['policies'].items():
            selected = policy['selected'] if policy['FRDiv'] > record['baseline']['FRDiv']+1e-12 else base
            policies[name] = selected
        policies['one_swap_GT_oracle'] = best
        # Enumerate all two-replacement sets through pairs of one-swap moves.
        # The cross term corrects the two independently measured gains.
        old = np.repeat(np.arange(10), len(pool)-10)
        new = np.tile(np.arange(10,len(pool)),10)
        gain = 2/90*(distance[new,:10].sum(1)-distance[new,old]-distance[old,:10].sum(1))
        dc, dd = c[new]-c[old], d[new]-d[old]
        cross = 2/90*(distance[new[:,None],new[None,:]]-distance[new[:,None],old[None,:]]
                       -distance[old[:,None],new[None,:]]+distance[old[:,None],old[None,:]])
        scores = gain[:,None]+gain[None,:]+cross
        feasible = ((old[:,None]!=old[None,:]) & (new[:,None]!=new[None,:]) &
                    (dc[:,None]+dc[None,:]>=-1e-10) & (dd[:,None]+dd[None,:]<=1e-8))
        scores[~feasible] = -np.inf
        a,b = np.unravel_index(np.argmax(scores),scores.shape)
        selected = [k for k in base if k not in (old[a],old[b])]+[int(new[a]),int(new[b])]
        direct = selection_value(distance,selected)
        np.testing.assert_allclose(direct-record['baseline']['FRDiv'],scores[a,b],atol=1e-9)
        policies['up_to_two_swap_GT_oracle'] = selected if direct>value+1e-12 else best
        if args.learned:
            sample=dataset[i]
            assert sample['clip_id']==clips[i]
            length=int(sample['source_lengths'])
            speaker=trajectory_summary(sample['speaker_emotion'][:length].numpy())
            _,_,_,features,_=make_pairs_with_length([x[i].float() for x in soft],speaker,length)
            pred=scorer['model'].predict(features)*scorer['std']+scorer['mean']
            np.testing.assert_allclose(features[:,-1],gain,atol=1e-7)
            legal=(old[:,None]!=old[None,:]) & (new[:,None]!=new[None,:])
            for name,eps in [('zero',np.zeros(2)),('q50',np.asarray(scorer['calibration_eps']['0.5'])),('q75',np.asarray(scorer['calibration_eps']['0.75']))]:
                single_ok=(pred[:,0]>=eps[0]) & (pred[:,1]<=-eps[1]) & (gain>1e-12)
                chosen=base
                improvement=0.
                if single_ok.any():
                    j=int(np.argmax(np.where(single_ok,gain,-np.inf)))
                    improvement=gain[j]
                    chosen=[k for k in base if k!=old[j]]+[int(new[j])]
                feasible=legal & (pred[:,0,None]+pred[None,:,0]>=2*eps[0]) & (pred[:,1,None]+pred[None,:,1]<=-2*eps[1])
                gains=np.where(feasible,gain[:,None]+gain[None,:]+cross,-np.inf)
                a,b=np.unravel_index(np.argmax(gains),gains.shape)
                if gains[a,b]>improvement+1e-12:
                    chosen=[k for k in base if k not in (old[a],old[b])]+[int(new[a]),int(new[b])]
                policies['learned_two_'+name]=chosen
        metrics = {}
        for name, selected in policies.items():
            metrics[name] = dict(FRC=float(c[selected].sum()), exact_FRD=float(d[selected].sum()),
                FRDiv=selection_value(distance,selected), changed=selected!=base,
                strict=bool(c[selected].sum()>=c[:10].sum()-1e-10 and d[selected].sum()<=d[:10].sum()+1e-8))
        rows.append(dict(clip_id=clips[i], baseline=record['baseline'], policies=metrics))
        if (i+1)%100==0: print(i+1, flush=True)
    baseline = {k:float(np.mean([r['baseline'][k] for r in rows])) for k in ('FRC','exact_FRD','FRDiv')}
    summary = {}
    for name in rows[0]['policies']:
        values = [r['policies'][name] for r in rows]
        summary[name] = {k:float(np.mean([v[k] for v in values])) for k in baseline}
        summary[name].update(strict=sum(v['strict'] for v in values),changed=sum(v['changed'] for v in values))
    out=RUNS/('selector_diversity_learned_two' if args.learned else 'selector_diversity_diagnosis')
    out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(baseline=baseline,policies=summary,records=rows),indent=2))
    lines=['# One-swap diversity diagnosis','','VAL571, existing center750/all-session protocol. Cached policies receive a prediction-only positive-diversity gate. Oracle uses GT and is diagnostic only.','','| Method | FRC | exact FRD | FRDiv | strict contexts | changed |','|---|---:|---:|---:|---:|---:|']
    for name,v in [('baseline',dict(**baseline,strict=571,changed=0)),*summary.items()]:
        lines.append(f"| {name} | {v['FRC']:.6f} | {v['exact_FRD']:.6f} | {v['FRDiv']:.6f} | {v['strict']} | {v['changed']} |")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
