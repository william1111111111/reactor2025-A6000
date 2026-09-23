"""Session-grouped cross-fitted quality scorer and prediction-only Select-10.

GT quality is used only for fit/calibration labels and final auditing.  Test
selection sees speaker features, frozen candidate trajectories and provenance.
This is an exploratory VAL cross-fit, not a held-out TEST claim.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor

from coreset_reactor.certified_select import geometry, improve_swaps, selection_value
from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.run_certified_oracle import RUNS, SOURCES, build_pool
from coreset_reactor.teacher_cache import sha256_file


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def trajectory_summary(value: np.ndarray) -> np.ndarray:
    """Fixed inference-only statistics for one [T,25] trajectory."""
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 25 or not np.isfinite(value).all():
        raise ValueError('Expected finite [T,25] trajectory')
    velocity = np.abs(np.diff(value, axis=0)).mean(0) if len(value) > 1 else np.zeros(25)
    parts = [value.mean(0), value.std(0), value.min(0), value.max(0), velocity,
             value[0], value[-1]]
    for part in np.array_split(value, 4):
        parts.append(part.mean(0))
    result = np.concatenate(parts).astype(np.float32)
    if result.shape != (275,):
        raise AssertionError(result.shape)
    return result


def balanced_session_folds(sessions: list[str], counts: dict[str, int], folds: int) -> list[list[str]]:
    groups = [[] for _ in range(folds)]
    loads = [0] * folds
    for session in sorted(sessions, key=lambda x: (-counts[x], x)):
        fold = min(range(folds), key=lambda i: (loads[i], i))
        groups[fold].append(session)
        loads[fold] += counts[session]
    if sorted(sum(groups, [])) != sorted(sessions):
        raise AssertionError('Session fold construction failed')
    return groups


def robust_arrays(pred_frc: np.ndarray, pred_frd: np.ndarray,
                  baseline: list[int], eps_frc: float, eps_frd: float) -> tuple[np.ndarray, np.ndarray]:
    """Encode pred(delta) +/- eps*L1(delta) as additive set constraints."""
    c, d = pred_frc.copy(), pred_frd.copy()
    mask = np.zeros(len(c), dtype=bool)
    mask[baseline] = True
    c[mask] += eps_frc
    c[~mask] -= eps_frc
    d[mask] -= eps_frd
    d[~mask] += eps_frd
    return c, d


def aggregate(records: list[dict], policy: str) -> dict:
    selected = [r['policies'][policy] for r in records]
    baseline = [r['baseline'] for r in records]
    def mean(key, values):
        return float(np.mean([v[key] for v in values]))
    strict = np.array([v['strict_quality_pass'] for v in selected], dtype=bool)
    source = Counter()
    changes = []
    for record, value in zip(records, selected):
        source.update(record['origins'][i][0]['model'] for i in value['selected'])
        changes.append(len(set(value['selected']) - set(range(10))))
    return {
        'contexts': len(records),
        'baseline': {k: mean(k, baseline) for k in ('FRC','exact_FRD','FRDiv')},
        'selected': {k: mean(k, selected) for k in ('FRC','exact_FRD','FRDiv')},
        'delta': {k: mean(k, selected)-mean(k, baseline) for k in ('FRC','exact_FRD','FRDiv')},
        'strict_quality_pass_contexts': int(strict.sum()),
        'strict_quality_pass_fraction': float(strict.mean()),
        'positive_FRDiv_contexts': int(sum(v['FRDiv'] > b['FRDiv'] + 1e-12
                                           for v,b in zip(selected,baseline))),
        'mean_replacements': float(np.mean(changes)),
        'selected_source_counts': dict(source),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--trees', type=int, default=96)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--target-mode', choices=('absolute','baseline_relative'), default='absolute')
    args = parser.parse_args()
    if args.output.exists() or args.folds < 3 or args.trees < 8 or args.jobs < 1:
        raise ValueError('Invalid configuration or output already exists')
    args.output.mkdir(parents=True)
    started = time.monotonic()

    # Frozen predictions and per-candidate GT labels are loaded independently.
    values, soft = [], []
    for label, path in SOURCES:
        report = json.loads(path.read_text())
        payload = torch.load(path.with_suffix('.predictions.pt'), map_location='cpu', weights_only=False)
        if payload['checkpoint_sha256'] != report['checkpoint_sha256']:
            raise ValueError('Prediction provenance mismatch')
        values.append(report)
        soft.append(payload['soft_predictions'])
    clip_ids = [r['clip_id'] for r in values[0]['rows']]
    if len(clip_ids) != 571 or any([r['clip_id'] for r in v['rows']] != clip_ids for v in values[1:]):
        raise ValueError('Source contexts do not match')

    data_root = Path('/public/zhaowenjie/react2025_new/data')
    dataset = PairedReactionDataset(
        data_root, 'val', 750, crop_mode='center',
        normalization_dir=Path('/home/zhaowenjie/react2025_new/mam_reactor/external/FaceVerse'))
    context_files = {}
    quality_files = {}
    oracle_root = RUNS/'certified_select10_val571_seed1234'
    for shard in oracle_root.glob('shard_*'):
        for path in shard.glob('context_*.json'):
            index = int(path.stem.split('_')[1]); context_files[index] = path
        for path in shard.glob('quality_*.npz'):
            index = int(path.stem.split('_')[1]); quality_files[index] = path
    if set(context_files) != set(range(571)) or set(quality_files) != set(range(571)):
        raise ValueError('Incomplete Oracle cache')

    features, qualities, distances, origins, contexts = [], [], [], [], []
    for index in range(571):
        sample = dataset[index]
        if sample['clip_id'] != clip_ids[index]:
            raise ValueError('Dataset order mismatch')
        length = int(sample['source_lengths'])
        speaker = trajectory_summary(sample['speaker_emotion'][:length].numpy())
        pool, pool_origins, raw_count, _ = build_pool([p[index] for p in soft], [v[0] for v in SOURCES])
        if len(pool) != 60 or raw_count != 30:
            raise ValueError('This prototype requires the fixed E60 pool')
        candidate = np.stack([trajectory_summary(p.numpy()) for p in pool])
        source = np.repeat(speaker[None], 60, axis=0)
        interaction = np.concatenate((candidate[:, :25]-source[:, :25],
                                      np.abs(candidate[:, :25]-source[:, :25])), axis=1)
        metadata = np.zeros((60,15), dtype=np.float32)
        for row, item in enumerate(pool_origins):
            origin = item[0]
            metadata[row, ('basic','distance','direction').index(origin['model'])] = 1
            metadata[row, 3+int(origin['slot'])] = 1
            metadata[row, 13] = float(origin['scale'] == 1.5)
            metadata[row, 14] = length/750.
        # No clip/session ID, GT count, GT descriptor or quality enters X.
        x = np.concatenate((candidate, source, interaction, metadata), axis=1)
        if x.shape != (60,615) or not np.isfinite(x).all():
            raise ValueError('Invalid inference feature matrix')
        with np.load(quality_files[index], allow_pickle=False) as q:
            quality = np.stack((q['frc'],q['frd']), axis=1)
        record = json.loads(context_files[index].read_text())
        if record['clip_id'] != clip_ids[index] or quality.shape != (60,2):
            raise ValueError('Quality/context cache mismatch')
        points = pool.flatten(1).double().numpy()/np.sqrt(pool.shape[1]*pool.shape[2])
        distance = geometry(points)[2]
        features.append(x); qualities.append(quality); distances.append(distance)
        origins.append(pool_origins)
        contexts.append({'clip_id':clip_ids[index], 'session':sample['session_id'],
                         'oracle_selected':record['expanded60']['selected']})
    X, Y = np.stack(features), np.stack(qualities)
    if not np.isfinite(Y).all():
        raise ValueError('Nonfinite labels')

    counts = Counter(c['session'] for c in contexts)
    fold_sessions = balanced_session_folds(sorted(counts), counts, args.folds)
    context_fold = np.array([next(i for i,g in enumerate(fold_sessions) if c['session'] in g)
                             for c in contexts])
    model_targets = Y.copy()
    if args.target_mode == 'baseline_relative':
        model_targets -= Y[:,:10].mean(1, keepdims=True)
    predictions = np.full_like(model_targets, np.nan, dtype=np.float64)
    calibration_eps = {}
    fold_records = []
    for test_fold in range(args.folds):
        calibration_fold = (test_fold+1) % args.folds
        train_contexts = np.flatnonzero((context_fold != test_fold) & (context_fold != calibration_fold))
        calibration_contexts = np.flatnonzero(context_fold == calibration_fold)
        test_contexts = np.flatnonzero(context_fold == test_fold)
        train_x = X[train_contexts].reshape(-1,X.shape[-1])
        train_y = model_targets[train_contexts].reshape(-1,2)
        mean, std = train_y.mean(0), train_y.std(0)
        model = ExtraTreesRegressor(
            n_estimators=args.trees, min_samples_leaf=12, max_features=.7,
            max_depth=22, n_jobs=args.jobs, random_state=args.seed+test_fold)
        model.fit(train_x, (train_y-mean)/std)
        cal_pred = (model.predict(X[calibration_contexts].reshape(-1,X.shape[-1]))*std+mean).reshape(len(calibration_contexts),60,2)
        test_pred = (model.predict(X[test_contexts].reshape(-1,X.shape[-1]))*std+mean).reshape(len(test_contexts),60,2)
        if args.target_mode == 'baseline_relative':
            # Baseline-centering is prediction-only and makes its set sum
            # exactly zero, matching the baseline-relative quality contract.
            cal_pred -= cal_pred[:,:10].mean(1,keepdims=True)
            test_pred -= test_pred[:,:10].mean(1,keepdims=True)
        predictions[test_contexts] = test_pred
        residual = np.abs(model_targets[calibration_contexts]-cal_pred).reshape(-1,2)
        calibration_eps[test_fold] = {
            str(q): np.quantile(residual,q,axis=0).tolist() for q in (.5,.75,.9,.95,.99)
        }
        fold_records.append({'fold':test_fold,'test_sessions':fold_sessions[test_fold],
                             'calibration_sessions':fold_sessions[calibration_fold],
                             'train_contexts':len(train_contexts),'calibration_contexts':len(calibration_contexts),
                             'test_contexts':len(test_contexts)})
        print(json.dumps(fold_records[-1]), flush=True)
    if not np.isfinite(predictions).all():
        raise ValueError('Incomplete cross-fitted predictions')

    policies = ['margin_0','margin_q50','margin_q75','margin_q90','margin_q95','margin_q99']
    quantile = {'margin_0':None,'margin_q50':'0.5','margin_q75':'0.75',
                'margin_q90':'0.9','margin_q95':'0.95','margin_q99':'0.99'}
    records = []
    baseline_ids = list(range(10))
    for index in range(571):
        true_c,true_d = Y[index,:,0],Y[index,:,1]
        base = {'FRC':float(true_c[:10].sum()),'exact_FRD':float(true_d[:10].sum()),
                'FRDiv':selection_value(distances[index],baseline_ids)}
        item = {'clip_id':contexts[index]['clip_id'],'session':contexts[index]['session'],
                'fold':int(context_fold[index]),'baseline':base,'origins':origins[index], 'policies':{}}
        for name in policies:
            q = quantile[name]
            eps = (np.zeros(2) if q is None else
                   np.array(calibration_eps[int(context_fold[index])][q]))
            c,d = robust_arrays(predictions[index,:,0],predictions[index,:,1],baseline_ids,*eps)
            selected = improve_swaps(distances[index],c,d,baseline_ids)
            actual = {'selected':selected,'FRC':float(true_c[selected].sum()),
                      'exact_FRD':float(true_d[selected].sum()),
                      'FRDiv':selection_value(distances[index],selected),
                      'strict_quality_pass':bool(true_c[selected].sum() >= true_c[:10].sum()-1e-10 and
                                                 true_d[selected].sum() <= true_d[:10].sum()+1e-8),
                      'eps_FRC':float(eps[0]),'eps_FRD':float(eps[1])}
            item['policies'][name] = actual
        records.append(item)
        if (index+1)%25==0:
            atomic_json(args.output/'progress.json',{'completed':index+1,'total':571})

    flat_y, flat_p = model_targets.reshape(-1,2), predictions.reshape(-1,2)
    summary = {
        'scope':'VAL571 exploratory session-grouped five-fold cross-fit; center750/all-session labels; not official TEST',
        'inference_inputs':'speaker emotion summary, candidate trajectory summary, model/slot/scale provenance, source length; no GT/session/clip identity',
        'quality_labels':'cached all-session per-candidate FRC and exact FRD; labels unavailable to test selector',
        'quality_target_mode':args.target_mode,
        'candidate_pool':'E60 frozen Basic/Distance/Direction KD, raw+scale1.5',
        'folds':fold_records,'fold_sessions':fold_sessions,'calibration_eps':calibration_eps,
        'quality_prediction':{
            'FRC_spearman':float(spearmanr(flat_y[:,0],flat_p[:,0]).statistic),
            'FRD_spearman':float(spearmanr(flat_y[:,1],flat_p[:,1]).statistic),
            'FRC_MAE':float(np.abs(flat_y[:,0]-flat_p[:,0]).mean()),
            'FRD_MAE':float(np.abs(flat_y[:,1]-flat_p[:,1]).mean())},
        'policies':{name:aggregate(records,name) for name in policies},
        'oracle_reference':json.loads((oracle_root/'summary.json').read_text())['pools']['expanded60'],
        'parameters_per_fold':None,
        'selector_training_seconds':time.monotonic()-started,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_sha256':sha256_file(Path(__file__)),
        'limitations':['cross-fitted development-set diagnostic, not independent TEST',
                       'session holdout prevents same-session GT-pool leakage but does not replace external validation',
                       'quality margins are empirical calibration quantiles, not formal coverage guarantees',
                       'center750 timing protocol retained for matched comparison']}
    atomic_json(args.output/'summary.json',summary)
    torch.save({'predicted_quality':torch.from_numpy(predictions),'true_quality':torch.from_numpy(Y),
                'context_fold':torch.from_numpy(context_fold)},args.output/'crossfit_predictions.pt')
    atomic_json(args.output/'records.json',{'records':records})
    lines=['# Inference-safe Select-10 cross-fit','',summary['scope'],'',
           '| Policy | FRC | exact FRD | FRDiv | strict pass | mean replacements |',
           '|---|---:|---:|---:|---:|---:|']
    base=summary['policies']['margin_0']['baseline']
    lines.append(f"| Basic-KD baseline | {base['FRC']:.6f} | {base['exact_FRD']:.6f} | {base['FRDiv']:.6f} | 571/571 | 0 |")
    for name in policies:
        p=summary['policies'][name]; m=p['selected']
        lines.append(f"| {name} | {m['FRC']:.6f} | {m['exact_FRD']:.6f} | {m['FRDiv']:.6f} | {p['strict_quality_pass_contexts']}/571 | {p['mean_replacements']:.3f} |")
    lines += ['',f"Quality prediction Spearman: FRC {summary['quality_prediction']['FRC_spearman']:.4f}, FRD {summary['quality_prediction']['FRD_spearman']:.4f}.",
              '', 'No GT, session identity or clip identity is available to the selector on its held-out fold.',
              'This is an exploratory cross-fit on VAL, not an independent test result or formal quality guarantee.']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'quality_prediction':summary['quality_prediction'],'policies':summary['policies']},indent=2),flush=True)


if __name__ == '__main__':
    main()
