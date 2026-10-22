"""Replay frozen VAL predictions against legacy and full-align-then-crop GT.

No training or model inference. Shards retain per-context GT and exact scores;
aggregate only after every original context is present and baseline parity holds.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

from coreset_reactor.crop_target_alignment import align_targets_then_crop
from coreset_reactor.fast_frc import candidate_frc
from coreset_reactor.oracle_select import _rolling_frd
from framework.metrics import compute_s_mse, compute_FRVar
from framework.modules.post_processor import Processor

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--aggregate', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    baseline_path = ROOT/'coreset_reactor/reports/mode_reference_student_seed1/basic/val571_full.json'
    predictions_path = baseline_path.with_suffix('.predictions.pt')
    old_cache = ROOT/'coreset_reactor/reports/fixed_teacher_distillation/val571_postprocessed_gt_cache'
    baseline = json.loads(baseline_path.read_text())
    payload = torch.load(predictions_path, map_location='cpu')
    assert payload['checkpoint_sha256'] == baseline['checkpoint_sha256']
    assert payload['clip_ids'] == [r['clip_id'] for r in baseline['rows']]
    assert len(payload['clip_ids']) == 571
    predictions = payload['official_predictions']
    asset = ROOT/'mam_reactor'
    assert digest(asset/'pretrained_models/post_processor/checkpoint.pth') == baseline['cache_protocol']['postprocessor_sha256']
    args.output.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        rows = [json.loads((args.output/'rows'/f'{i:04d}.json').read_text()) for i in range(571)]
        assert [r['clip_id'] for r in rows] == payload['clip_ids']
        before = {k: float(np.mean([r['before'][k] for r in rows])) for k in ('FRC', 'exact_FRD')}
        after = {k: float(np.mean([r['after'][k] for r in rows])) for k in before}
        for k, v in [('FRDiv', float(compute_s_mse(predictions))), ('FRVar', float(compute_FRVar(predictions)))]:
            before[k] = after[k] = v
        for k in before:
            np.testing.assert_allclose(before[k], baseline['metrics'][k], rtol=1e-6, atol=1e-6)
        result = {'contexts':571, 'checkpoint_sha256':payload['checkpoint_sha256'],
                  'baseline':str(baseline_path), 'predictions_sha256':digest(predictions_path),
                  'before':before, 'after':after,
                  'delta':{k:after[k]-before[k] for k in before},
                  'relative_percent':{k:100*(after[k]-before[k])/abs(before[k]) for k in before},
                  'changed_contexts':sum(r['gt_changed'] for r in rows),
                  'cropped_contexts':sum(r['source_length']>r['frames'] for r in rows),
                  'protocol':'Same archived center750 K10 predictions and all-session reference membership. Official full-duration GT conversion, then identical source crop.',
                  'limitation':'Nonpaired session GT has no verified frame-level semantic alignment. Not full-sequence official TEST.',
                  'rows':rows}
        write(args.output/'comparison.json', result)
        print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2), flush=True)
        return
    assert 0 <= args.shard < args.shards
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    provenance = {'baseline_sha256':digest(baseline_path), 'predictions_sha256':digest(predictions_path),
                  'checkpoint_sha256':payload['checkpoint_sha256'], 'torch_version':torch.__version__,
                  'script_sha256':digest(Path(__file__)),
                  'alignment_sha256':digest(Path(__file__).with_name('crop_target_alignment.py')),
                  'shard':args.shard,'shards':args.shards,'device':str(device)}
    prov_path = args.output/f'provenance_{args.shard}.json'
    if prov_path.exists():
        assert json.loads(prov_path.read_text()) == provenance, 'Resume provenance changed'
    else:
        write(prov_path, provenance)
    (args.output/'rows').mkdir(exist_ok=True)
    (args.output/'targets').mkdir(exist_ok=True)
    processor = Processor(cfg_dir=str(asset), ckpt_dir=str(asset/'pretrained_models/post_processor'),
                          clip_len_test=1000, device=device, num_preds=10)
    facial = ROOT/'data/val/facial-attributes'
    file_hashes = {}
    started = time.monotonic()
    with ThreadPoolExecutor(args.workers) as pool, torch.inference_mode():
        for i in range(args.shard, len(predictions), args.shards):
            result_path = args.output/'rows'/f'{i:04d}.json'
            if result_path.exists():
                continue
            clip = payload['clip_ids'][i]
            source = facial/(clip+'.npy')
            source_lengths = [np.load(ROOT/'data/val'/folder/(clip+'.npy'), mmap_mode='r').shape[0]
                              for folder in ('facial-attributes','audio-features','coefficients')]
            total = min(source_lengths)
            start = max(0, total-750)//2
            prediction = predictions[i]
            frames = prediction.shape[1]
            assert prediction.shape == (10, min(750, total), 25)
            paired = facial/(clip.replace('speaker/','listener/',1)+'.npy')
            paths = [paired]+[v for v in sorted(paired.parent.glob('*.npy')) if v != paired]
            original_row = baseline['rows'][i]
            assert len(paths) == original_row['gt_count']
            old_payload = torch.load(old_cache/(original_row['target_cache_key']+'.pt'),map_location='cpu')
            assert old_payload['fingerprint'] == original_row['target_cache_key']
            old = old_payload['targets']
            raw = [torch.from_numpy(np.load(v).astype(np.float32)) for v in paths]
            target_path = args.output/'targets'/f'{i:04d}.pt'
            if target_path.exists():
                new = torch.load(target_path, map_location='cpu')['targets']
            else:
                new = align_targets_then_crop(processor,prediction,raw,source_length=total,
                    crop_start=start,device=device,seed=baseline['cache_protocol']['eval_seed'],clip_id=clip).cpu()
                tmp = target_path.with_suffix('.tmp')
                torch.save({'targets':new,'clip_id':clip,'source_length':total,'crop_start':start},tmp)
                os.replace(tmp,target_path)
            assert new.shape == old.shape
            before_frc = candidate_frc(old.numpy(),prediction.numpy())
            np.testing.assert_allclose(before_frc, original_row['frc'], rtol=1e-6, atol=1e-7)
            after_frc = candidate_frc(new.numpy(),prediction.numpy())
            # Exact CPU recurrence; threads release the GIL inside numba DTW.
            def score(target):
                return float(sum(pool.map(lambda pred: _rolling_frd(target.numpy(),pred.numpy()), prediction)))
            before_frd = score(old)
            np.testing.assert_allclose(before_frd, original_row['exact_FRD'], rtol=1e-6, atol=1e-6)
            changed = not torch.equal(old,new)
            after_frd = score(new) if changed else before_frd
            for path in paths:
                if path not in file_hashes:
                    file_hashes[path] = digest(path)
            row = {'index':i,'clip_id':clip,'source_length':total,'source_feature_lengths':source_lengths,
                   'crop_start':start,'frames':frames,'gt_count':len(paths),
                   'gt_files':[(str(v),file_hashes[v]) for v in paths],
                   'gt_changed':changed,'paired_raw_length':len(raw[0]),
                   'before':{'FRC':float(before_frc.sum()),'exact_FRD':before_frd},
                   'after':{'FRC':float(after_frc.sum()),'exact_FRD':after_frd}}
            write(result_path,row)
            print(json.dumps({'shard':args.shard,'index':i,'elapsed_s':time.monotonic()-started,
                              'before':row['before'],'after':row['after']}),flush=True)
    write(args.output/f'done_{args.shard}.json',{'completed':True,'elapsed_s':time.monotonic()-started})


if __name__ == '__main__':
    main()
