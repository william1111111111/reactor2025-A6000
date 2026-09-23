"""Evaluate frozen KD and relational KD on exactly the same full VAL inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'coreset_reactor/reports/fixed_teacher_distillation'
BASELINE = RUNS / 't10_student_pilot10e'
RELATION = RUNS / 't10_student_rel_pilot10e'


def evaluate(run, name, device, exact=False):
    output = run / name
    if output.exists():
        return json.loads(output.read_text())
    command = [sys.executable, '-m', 'coreset_reactor.eval_expert_student_fast',
               '--checkpoint', str(run / 'student.pth'),
               '--data-root', '/public/zhaowenjie/react2025_new/data',
               '--asset-mam-root', '/home/zhaowenjie/react2025_new/mam_reactor',
               '--output', str(output), '--device', device,
               '--target-cache', str(RUNS / 'val571_postprocessed_gt_cache')]
    if exact:
        command += ['--exact-frd', '--metric-workers', '16']
    with output.with_suffix('.log').open('a') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(output.read_text())


def write_report(base, new, baseline_full=None):
    if [r['clip_id'] for r in base['rows']] != [r['clip_id'] for r in new['rows']]:
        raise ValueError('VAL context mismatch')
    if [r['target_cache_key'] for r in base['rows']] != [r['target_cache_key'] for r in new['rows']]:
        raise ValueError('Processed GT cache keys differ')
    bc, nc = [json.loads((p / 'config.json').read_text()) for p in (BASELINE, RELATION)]
    for key in ('epochs', 'batch_size', 'workers', 'lr', 'seed', 'expert_embedding_dim',
                'cache_manifest', 'initialize_backbone'):
        if bc[key] != nc[key]:
            raise ValueError(f'Training config differs: {key}')
    metrics = ('FRC', 'FRDiv', 'FRVar', 'soft_FRDiv')
    comparisons = {k: {'baseline': base['metrics'][k], 'relation': new['metrics'][k],
                       'delta': new['metrics'][k] - base['metrics'][k]} for k in metrics}
    if baseline_full and isinstance(new['metrics']['exact_FRD'], (float, int)):
        comparisons['exact_FRD'] = {'baseline': baseline_full['metrics']['exact_FRD'],
                                   'relation': new['metrics']['exact_FRD'],
                                   'delta': new['metrics']['exact_FRD'] - baseline_full['metrics']['exact_FRD']}
    for metric in ('FRC', 'FRDiv'):
        def value(row):
            return sum(row['frc']) if metric == 'FRC' else row['FRDiv']
        delta = np.array([value(n) - value(b) for n, b in zip(new['rows'], base['rows'])])
        rng = np.random.default_rng(1234)
        # Context bootstrap is descriptive; session correlations limit inference.
        means = delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(1)
        comparisons[metric]['context_bootstrap_delta_ci95'] = np.quantile(means, [.025, .975]).tolist()
        comparisons[metric]['per_context_delta_quantiles'] = np.quantile(delta, [0, .1, .5, .9, 1]).tolist()
    report = {'metrics': comparisons, 'provenance': json.loads((RELATION/'provenance.json').read_text()),
              'scope': 'full VAL571 speaker sources; center750; all-session GT; same processed GT cache',
              'limitations': ['single seed', 'context bootstrap does not account for session correlations',
                             'latencies are diagnostics, not a dedicated controlled benchmark'],
              'baseline_checkpoint_sha256': base['checkpoint_sha256'],
              'relation_checkpoint_sha256': new['checkpoint_sha256']}
    (RELATION/'matched_comparison.json').write_text(json.dumps(report, indent=2))
    lines = ['# KD versus teacher-relation KD', '', report['scope'], '',
             '| Metric | Basic KD | Relation KD | Delta |', '|---|---:|---:|---:|']
    for metric, row in comparisons.items():
        lines.append(f"| {metric} | {row['baseline']:.6f} | {row['relation']:.6f} | {row['delta']:+.6f} |")
    lines += ['', 'Both students use the same architecture, seed=1, teacher cache, teacher-0 initialization,',
              'batch=4, 10 epochs and 8300 updates. Only the relation loss differs.',
              'Teacher soft trajectory pair distances are matched by AU/VA/expression group.',
              'TRAIN-only initial gradient calibration sets the auxiliary weight.',
              '', 'Exact FRD: ' + ('complete' if 'exact_FRD' in comparisons else 'pending'),
              'FRC/FRD increases or decreases must be checked jointly before claiming a quality-preserving improvement.',
              'No inference noise, additional generator parameters, or teacher access is used by the student.',
              'Results are single-seed VAL analysis, not official TEST scores.']
    (ROOT/'reports/student_relation_comparison.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(comparisons, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:6')
    args = parser.parse_args()
    baseline = evaluate(BASELINE, 'all_session_val571_cached.json', args.device)
    # Ensure baseline output reproduces the earlier frozen result before reuse.
    reference = json.loads((BASELINE/'all_session_val571_fast.json').read_text())
    for metric in ('FRC', 'FRDiv', 'FRVar'):
        if abs(baseline['metrics'][metric] - reference['metrics'][metric]) > 1e-4:
            raise ValueError(f'Baseline reproduction failed for {metric}')
    deadline = time.monotonic() + 3 * 3600
    while not (RELATION/'train_metrics.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Relation training did not complete within three hours')
        time.sleep(30)
    new = evaluate(RELATION, 'all_session_val571_fast.json', args.device)
    write_report(baseline, new)
    new_full = evaluate(RELATION, 'all_session_val571_full.json', args.device, exact=True)
    baseline_full_path = BASELINE/'all_session_val571_full.json'
    baseline_full = json.loads(baseline_full_path.read_text()) if baseline_full_path.exists() else None
    write_report(baseline, new_full, baseline_full)


if __name__ == '__main__':
    main()
