"""Controlled reference-count ablation, not a full official TEST reproduction."""
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.fast_frc import candidate_frc
from coreset_reactor.oracle_select import Candidate, SharedQualityPool, _quality_terms


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent / 'reports/fixed_teacher_distillation'
    source = root / 't10_student_direction_pilot10e/all_session_val571_full.json'
    output = root / 'direction_sampled10_matched_val571'
    output.mkdir(exist_ok=False)
    original = json.loads(source.read_text())
    cached = torch.load(source.with_suffix('.predictions.pt'), map_location='cpu')
    assert cached['checkpoint_sha256'] == original['checkpoint_sha256']
    assert cached['clip_ids'] == [r['clip_id'] for r in original['rows']]
    pool = SharedQualityPool(8, 10, 750, max_targets=10)
    rows = []
    started = time.monotonic()
    try:
        for i, (row, pred) in enumerate(zip(original['rows'], cached['official_predictions'])):
            payload = torch.load(root / 'val571_postprocessed_gt_cache' / (row['target_cache_key'] + '.pt'), map_location='cpu')
            assert payload['fingerprint'] == row['target_cache_key']
            targets = payload['targets']
            assert len(targets) == row['gt_count']
            # Cache builder puts the same-name paired reference at index zero.
            seed = int.from_bytes(hashlib.sha256(('1234:' + row['clip_id']).encode()).digest()[:8], 'big')
            rng = random.Random(seed)
            others = list(range(1, len(targets))) or [0]
            chosen = [0] + (rng.sample(others, 9) if len(others) >= 9 else rng.choices(others, k=9))
            sampled = targets[chosen]
            all_frc = float(candidate_frc(targets.numpy(), pred.numpy()).sum())
            np.testing.assert_allclose(all_frc, sum(row['frc']), atol=1e-5)
            candidates = [Candidate(pred[k], str(k), 'direction', [('direction', k)]) for k in range(10)]
            frc, frd = _quality_terms(candidates, sampled, workers=8, engine='rolling', shared_pool=pool)
            sampled_frc, sampled_frd = float(frc.sum()), float(frd.sum())
            assert sampled_frc <= all_frc + 1e-5
            assert sampled_frd >= row['exact_FRD'] - 1e-5
            rows.append({'clip_id': row['clip_id'], 'indices': chosen, 'FRC': sampled_frc,
                         'exact_FRD': sampled_frd, 'all_FRC': all_frc, 'all_exact_FRD': row['exact_FRD']})
            summary = {'completed': len(rows), 'total': len(original['rows']),
                       'finished': len(rows) == len(original['rows']),
                       'protocol': 'VAL571 center750; fixed predictions and postprocessed GT; paired plus random9; per-context seed1234. Not official TEST/full-sequence reproduction.',
                       'checkpoint_sha256': original['checkpoint_sha256'],
                       'all_session': original['metrics'],
                       'sampled10_partial': {key: float(np.mean([r[key] for r in rows])) for key in ('FRC', 'exact_FRD')},
                       'elapsed_seconds': time.monotonic() - started}
            temp = output / 'summary.tmp'
            temp.write_text(json.dumps(summary, indent=2))
            temp.replace(output / 'summary.json')
            if (i + 1) % 25 == 0:
                print(json.dumps(summary), flush=True)
        (output / 'rows.json').write_text(json.dumps(rows))
    finally:
        pool.close()


if __name__ == '__main__':
    main()
