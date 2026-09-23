"""Twenty fixed-reference draws on frozen VAL predictions (not TEST reproduction)."""
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.fast_frc import candidate_frc
from tools.compute_frd_resumable import rolling_dtw


def distances(pred, targets):
    return np.asarray([sum(w * rolling_dtw(np.ascontiguousarray(pred[:, a:b]),
                                         np.ascontiguousarray(gt[:, a:b]))
                           for a, b, w in ((0, 15, 1/15), (15, 17, 1), (17, 25, 1/8)))
                       for gt in targets])


def stats(values):
    return dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)),
                minimum=float(np.min(values)), maximum=float(np.max(values)))


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent / 'reports/fixed_teacher_distillation'
    source = root / 't10_student_direction_pilot10e/all_session_val571_full.json'
    original = json.loads(source.read_text())
    cached = torch.load(source.with_suffix('.predictions.pt'), map_location='cpu')
    assert cached['checkpoint_sha256'] == original['checkpoint_sha256']
    assert cached['clip_ids'] == [r['clip_id'] for r in original['rows']]
    output = root / 'direction_gt_sampling20_val571'
    output.mkdir(exist_ok=False)
    previous = json.loads((root / 'direction_sampled10_matched_val571/rows.json').read_text())
    accumulated, started = [], time.monotonic()
    rolling_dtw(np.zeros((2, 1)), np.zeros((2, 1)))
    with ThreadPoolExecutor(max_workers=10) as executor:
        for index, (row, tensor) in enumerate(zip(original['rows'], cached['official_predictions'])):
            payload = torch.load(root / 'val571_postprocessed_gt_cache' / (row['target_cache_key'] + '.pt'), map_location='cpu')
            assert payload['fingerprint'] == row['target_cache_key']
            targets, pred = payload['targets'].numpy(), tensor.numpy()
            assert len(targets) == row['gt_count']
            frc = np.stack([candidate_frc(gt[None], pred) for gt in targets], axis=1)
            frd = np.stack(list(executor.map(lambda p: distances(p, targets), pred)))
            np.testing.assert_allclose(frc.max(1).sum(), sum(row['frc']), atol=1e-5)
            np.testing.assert_allclose(frd.min(1).sum(), row['exact_FRD'], atol=1e-5)
            selections, scores = [], []
            others = list(range(1, len(targets))) or [0]
            for seed in range(1234, 1254):
                rng = random.Random(int.from_bytes(hashlib.sha256(f'{seed}:{row["clip_id"]}'.encode()).digest()[:8], 'big'))
                chosen = [0] + (rng.sample(others, 9) if len(others) >= 9 else rng.choices(others, k=9))
                selections.append(chosen)
                scores.append([frc[:, chosen].max(1).sum(), frd[:, chosen].min(1).sum()])
            assert previous[index]['clip_id'] == row['clip_id']
            assert selections[0] == previous[index]['indices']
            np.testing.assert_allclose(scores[0], [previous[index]['FRC'], previous[index]['exact_FRD']], atol=1e-5)
            np.savez_compressed(output / f'context_{index:04d}.npz', frc=frc, frd=frd,
                                selections=selections, scores=scores, clip_id=row['clip_id'],
                                target_cache_key=row['target_cache_key'])
            accumulated.append(scores)
            values = np.asarray(accumulated)
            means = values.mean(0)
            summary = {'completed': index+1, 'total': len(original['rows']),
                       'finished': index+1 == len(original['rows']), 'seeds': list(range(1234,1254)),
                       'protocol': 'Frozen Direction-KD, VAL571 center750, cached GT, paired+9 repeated20; sampling-only diagnostic, not official TEST.',
                       'checkpoint_sha256': original['checkpoint_sha256'],
                       'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       'FRC_across_draw_means': stats(means[:,0]),
                       'exact_FRD_across_draw_means': stats(means[:,1]),
                       'per_draw_means': means.tolist(),
                       'context_FRC_std_quantiles': np.quantile(values[:,:,0].std(1,ddof=1),[0,.5,.9,1]).tolist(),
                       'context_FRD_std_quantiles': np.quantile(values[:,:,1].std(1,ddof=1),[0,.5,.9,1]).tolist(),
                       'unchanged_full_VAL_FRDiv': original['metrics']['FRDiv'],
                       'all_session_full_VAL': original['metrics'],
                       'elapsed_seconds': time.monotonic()-started}
            temp = output / 'summary.tmp'
            temp.write_text(json.dumps(summary, indent=2))
            temp.replace(output / 'summary.json')
            if (index+1)%25 == 0:
                print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
