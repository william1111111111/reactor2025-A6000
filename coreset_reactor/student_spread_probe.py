"""Frozen-student residual expansion: diagnostic VAL sweep, not new training."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from coreset_reactor.evaluate import official_prediction
from coreset_reactor.fast_frc import candidate_frc
from coreset_reactor.oracle_select import Candidate, SharedQualityPool
from coreset_reactor.teacher_cache import sha256_file
from framework.metrics import compute_FRVar, compute_s_mse


def expand_predictions(prediction, scale):
    """Expand around the per-frame slot mean, then restore channel domains."""
    if prediction.ndim != 3 or prediction.shape[-1] != 25:
        raise ValueError('Expected [K,T,25]')
    if not math.isfinite(scale) or scale < 1 or not torch.isfinite(prediction).all():
        raise ValueError('Need finite trajectories and scale >= 1')
    if scale == 1:
        return prediction.clone()  # Exact identity, including expression bits.
    center = prediction.mean(0, keepdim=True)
    result = center + scale * (prediction - center)
    result[..., :15] = result[..., :15].clamp(0, 1)
    result[..., 15:17] = result[..., 15:17].clamp(-1, 1)
    expression = result[..., 17:25].clamp_min(0)
    result[..., 17:25] = expression / expression.sum(-1, keepdim=True).clamp_min(1e-12)
    return result


def save_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def calibrate_spread(predictions, target, max_scale=4, tolerance=1e-4):
    """One global VAL-tuned scale, using diversity only (never per-context GT).

    A coarse bracket followed by bounded refinement; retain the best measured
    point. This is not a proof of global optimality or held-out generalization.
    """
    history = []

    def measure(scale):
        value = float(np.mean([float(compute_s_mse([
            official_prediction(expand_predictions(p, scale))])) for p in predictions]))
        history.append({'scale': scale, 'FRDiv': value})
        return value

    lo, hi = 1., 1.
    low_value = measure(lo)
    if low_value > target + tolerance:
        raise ValueError('Target below baseline; expansion cannot calibrate it')
    high_value = low_value
    while high_value < target-tolerance and hi < max_scale:
        lo, low_value = hi, high_value
        hi = min(hi*1.5, max_scale)
        high_value = measure(hi)
        if high_value < low_value-tolerance:
            raise ValueError('Nonmonotone calibration bracket; inspect manually')
    if high_value < target-tolerance:
        raise ValueError(f'Target not reached by max_scale={max_scale}')
    for _ in range(14):
        best = min(history, key=lambda x: (abs(x['FRDiv']-target), x['scale']))
        if abs(best['FRDiv']-target) <= tolerance:
            break
        mid = (lo+hi)/2
        value = measure(mid)
        if value < target:
            lo = mid
        else:
            hi = mid
    best = min(history, key=lambda x: (abs(x['FRDiv']-target), x['scale']))
    return best['scale'], {'target':target, 'tolerance':tolerance, 'max_scale':max_scale,
                           'achieved_FRDiv':best['FRDiv'], 'history':history,
                           'selection':'global VAL output diversity only; no per-context GT selection'}


def report(output, result):
    lines = ['# Frozen student residual expansion', '',
             'Full VAL571 speaker contexts; center750; all-session GT; same cached GT.',
             'No training or checkpoint changes. Expansion precedes AU rounding.',
             'AU clipped [0,1]; VA clipped [-1,1]; expression clipped nonnegative and renormalized.',
             '', '| Scale | FRC | exact FRD | FRDiv | FRVar | soft FRDiv |',
             '|---|---:|---:|---:|---:|---:|']
    for item in result['sweep']:
        m = item['metrics']
        frd = m.get('exact_FRD')
        frd_text = f'{frd:.6f}' if isinstance(frd, (int, float)) else 'not evaluated'
        lines.append(f"| {item['scale']:g} | {m['FRC']:.6f} | {frd_text} | "
                     f"{m['FRDiv']:.6f} | {m['FRVar']:.6f} | {m['soft_FRDiv']:.6f} |")
    lines += ['', f"Nearest tested scale to target {result['target_FRDiv']}: {result['nearest_scale']}",
              'Scale is selected using VAL FRDiv, so this is a validation-tuned diagnostic, not a held-out test result.',
              'No per-context GT-based scale selection. All ten outputs use the same global scale.',
              'Increasing distances is not evidence of learning new behavioral modes.',
              'Per-context metrics, clipping diagnostics, source hashes and deltas are in sweep.json.',
              f"Frozen checkpoint SHA256: {result['checkpoint_sha256']}"]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Completed full evaluation JSON')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scales', type=float, nargs='+', default=[1, 1.1, 1.2, 1.3, 1.5, 1.75, 2])
    parser.add_argument('--target', type=float, default=.16)
    parser.add_argument('--calibrate-target', action='store_true',
                        help='Replace grid with identity and a globally calibrated scale (max 4)')
    parser.add_argument('--exact-nearest', action='store_true')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = json.loads(args.source.read_text())
    checkpoint = Path(source['checkpoint'])
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != source['checkpoint_sha256']:
        raise ValueError('Checkpoint no longer matches frozen predictions')
    pred_path = args.source.with_suffix('.predictions.pt')
    payload = torch.load(pred_path, map_location='cpu', weights_only=False)
    if payload['checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('Prediction checkpoint mismatch')
    rows = source['rows']
    if len(rows) != 571 or payload['clip_ids'] != [r['clip_id'] for r in rows]:
        raise ValueError('Expected matching ordered full VAL571')
    soft = payload['soft_predictions']
    if len(soft) != len(rows) or any(p.shape[0] != 10 for p in soft):
        raise ValueError('Need exactly ten predictions per context')
    calibration = None
    if args.calibrate_target:
        with torch.inference_mode():
            scale, calibration = calibrate_spread(soft, args.target)
        args.scales = sorted(set([1., scale]))
        print('Spread calibration '+json.dumps(calibration), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    output_json = args.output/'sweep.json'
    identity = {'checkpoint_sha256': checkpoint_hash,
                'prediction_sha256': sha256_file(pred_path),
                'source_sha256': sha256_file(args.source),
                'script_sha256': sha256_file(Path(__file__)),
                'target_FRDiv': args.target, 'scales': args.scales}

    def targets(row, pred):
        key = row['target_cache_key']
        cached = torch.load(args.cache/f'{key}.pt', map_location='cpu', weights_only=False)
        target = cached['targets']
        if cached['fingerprint'] != key or target.shape != (row['gt_count'], pred.shape[1], 25):
            raise ValueError('GT cache identity/shape mismatch')
        if not torch.isfinite(target).all():
            raise ValueError('Nonfinite GT')
        return target

    if output_json.exists():
        result = json.loads(output_json.read_text())
        if any(result[k] != v for k, v in identity.items()):
            raise ValueError('Existing output belongs to a different experiment')
    else:
        if 1 not in args.scales or len(set(args.scales)) != len(args.scales):
            raise ValueError('Scales must be unique and include identity')
        records = [[] for _ in args.scales]
        started = time.monotonic()
        with torch.inference_mode():
            for index, (row, pred) in enumerate(zip(rows, soft), 1):
                target = targets(row, pred)
                for scale, dest in zip(args.scales, records):
                    tick = time.perf_counter()
                    transformed = expand_predictions(pred, scale)
                    latency = time.perf_counter()-tick
                    official = official_prediction(transformed)
                    frc = candidate_frc(target.numpy(), official.numpy())
                    if scale == 1:
                        np.testing.assert_allclose(frc, row['frc'], rtol=1e-5, atol=1e-6)
                    raw = pred.mean(0, keepdim=True) + scale*(pred-pred.mean(0, keepdim=True))
                    dest.append({'clip_id': row['clip_id'], 'target_cache_key': row['target_cache_key'],
                                 'FRC': float(frc.sum()), 'FRC_per_expert': frc.tolist(),
                                 'FRDiv': float(compute_s_mse([official])),
                                 'FRVar': float(compute_FRVar([official])),
                                 'soft_FRDiv': float(compute_s_mse([transformed])),
                                 'au_clipped_fraction': float(((raw[..., :15]<0)|(raw[..., :15]>1)).float().mean()),
                                 'va_clipped_fraction': float((raw[..., 15:17].abs()>1).float().mean()),
                                 'expression_negative_fraction': float((raw[..., 17:25]<0).float().mean()),
                                 'transform_seconds': latency})
                if index % 25 == 0 or index == len(rows):
                    save_json(args.output/'fast.progress.json', {'completed': index, 'total': len(rows)})
                    print(f'[spread-fast] {index}/{len(rows)}', flush=True)
        sweep = []
        baseline = records[args.scales.index(1)]
        for scale, rr in zip(args.scales, records):
            metrics = {k: float(np.mean([r[k] for r in rr])) for k in ('FRC', 'FRDiv', 'FRVar', 'soft_FRDiv')}
            if scale == 1:
                metrics['exact_FRD'] = source['metrics']['exact_FRD']
                for k in ('FRC', 'FRDiv', 'FRVar', 'soft_FRDiv'):
                    np.testing.assert_allclose(metrics[k], source['metrics'][k], rtol=1e-5, atol=1e-6)
            delta = {k: {'mean': metrics[k]-float(np.mean([r[k] for r in baseline])),
                         'quantiles': np.quantile([r[k]-b[k] for r,b in zip(rr,baseline)], [0,.1,.5,.9,1]).tolist()}
                     for k in ('FRC','FRDiv')}
            sweep.append({'scale':scale, 'metrics':metrics, 'delta_vs_identity':delta, 'rows':rr})
        nearest = min(sweep, key=lambda x: (abs(x['metrics']['FRDiv']-args.target), x['scale']))
        result = {**identity, 'source':str(args.source.resolve()), 'checkpoint':str(checkpoint),
                  'git_commit_sha':subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  'protocol':source['protocol'], 'sweep':sweep, 'nearest_scale':nearest['scale'],
                  'fast_seconds':time.monotonic()-started, 'scope':'VAL-tuned post-hoc diagnostic; no training',
                  'spread_calibration':calibration}
        save_json(output_json, result)
        report(args.output, result)
        print(json.dumps([{ 'scale':x['scale'], **x['metrics']} for x in sweep], indent=2), flush=True)
    if args.exact_nearest:
        chosen = next(x for x in result['sweep'] if x['scale'] == result['nearest_scale'])
        if 'exact_FRD' not in chosen['metrics']:
            partial = args.output/'exact.partial.json'
            done = json.loads(partial.read_text()) if partial.exists() else []
            for index, record in enumerate(done):
                if record['clip_id'] != rows[index]['clip_id'] or record['scale'] != chosen['scale']:
                    raise ValueError('Resume mismatch')
            pool = SharedQualityPool(args.workers, 10, max(p.shape[1] for p in soft),
                                     max_targets=max(r['gt_count'] for r in rows))
            try:
                for index in range(len(done), len(rows)):
                    pred = official_prediction(expand_predictions(soft[index], chosen['scale']))
                    candidates = [Candidate(p, f'spread_{k}', 'student_spread', [('student',k)]) for k,p in enumerate(pred)]
                    _, frd = pool.score(candidates, targets(rows[index], pred))
                    done.append({'clip_id':rows[index]['clip_id'], 'scale':chosen['scale'],
                                 'exact_FRD':float(frd.sum()), 'FRD_per_expert':frd.tolist()})
                    save_json(partial, done)
                    if len(done)%25 == 0 or len(done)==len(rows):
                        print(f'[spread-exact scale={chosen["scale"]}] {len(done)}/{len(rows)}', flush=True)
            finally:
                pool.close()
            chosen['metrics']['exact_FRD'] = float(np.mean([r['exact_FRD'] for r in done]))
            for row, frd_row in zip(chosen['rows'], done):
                row.update({k:v for k,v in frd_row.items() if k not in ('clip_id','scale')})
            save_json(output_json, result)
            report(args.output, result)
    if sha256_file(checkpoint) != checkpoint_hash:
        raise RuntimeError('Frozen checkpoint changed during experiment')
    print('Checkpoint hash unchanged.', flush=True)


if __name__ == '__main__':
    main()
