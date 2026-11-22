"""Exercise reference organization, both teacher losses, caching, and student KD.

Synthetic feature files and tiny CPU models keep this independent of REACT data.
This checks execution of the training path, not paper-level model convergence.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    work = args.output or Path(tempfile.mkdtemp(prefix='coreset-smoke-'))
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    data = work / 'data'
    rng = np.random.default_rng(1)
    for role in ['speaker', 'listener']:
        for index in range(10):
            frames = 20 + 3 * index
            facial = rng.random((frames, 25), dtype=np.float32)
            facial[:, :15] = np.round(facial[:, :15])
            facial[:, 15:17] = 2 * facial[:, 15:17] - 1
            facial[:, 17:] /= facial[:, 17:].sum(1, keepdims=True)
            for kind, value in [('audio-features', rng.standard_normal((frames, 768)).astype('float32')),
                                ('facial-attributes', facial),
                                ('coefficients', rng.standard_normal((frames, 58)).astype('float32'))]:
                directory = data / 'train' / kind / role / 'session0'
                directory.mkdir(parents=True, exist_ok=True)
                np.save(directory / f'clip{index:02d}.npy', value)
    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'OPENBLAS_NUM_THREADS': '2', 'PYTHONPATH': str(ROOT),
           'PYTHONDONTWRITEBYTECODE': '1'}
    stages = []

    def run(label, *arguments):
        command = [sys.executable, *map(str, arguments)]
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        (work / f'{label}.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f'{label} failed; see {work / (label + ".log")}\n{result.stderr[-3000:]}')
        stages.append(label)
        print(label + ': passed', flush=True)

    assignment = work / 'assignment.json'
    run('references', '-m', 'coreset_reactor.prepare_references', '--data-root', data, '--output', assignment)
    common = ['--data-dir', data, '--fixed-pair-manifest', assignment,
              '--fixed-pair-alignment-policy', 'relative_time_masked',
              '--epochs', 50, '--batch-size', 2, '--workers', 0, '--clip-length', 32,
              '--hidden-dim', 32, '--temporal-layers', 1, '--temporal-heads', 4,
              '--graph-dim', 16, '--graph-heads', 4, '--graph-layers', 1,
              '--device', 'cpu', '--precision', 'fp32', '--max-train-steps', 1,
              '--snapshot-steps', 1]
    run('paired_teacher', '-m', 'regnn.train_conditional_regnn', *common,
        '--fixed-pair-rank', 0, '--run-dir', work / 'teacher0')
    run('reference_teacher', '-m', 'coreset_reactor.train_mode_reference_teacher', *common,
        '--fixed-pair-rank', 1, '--run-dir', work / 'teacher1')
    q0 = work / 'teacher0/conditional-step000001-seed1.pth'
    q1 = work / 'teacher1/conditional-step000001-seed1.pth'
    run('cache', '-m', 'coreset_reactor.teacher_cache', '--data-root', data,
        '--teacher-checkpoints', q0, *([q1] * 9), '--output-dir', work / 'cache',
        '--device', 'cpu', '--workers', 0, '--batch-size', 2, '--clip-length', 32,
        '--crop-view-mode', 'stratified', '--crop-views', 3)
    run('student', '-m', 'coreset_reactor.train_expert_student', '--data-root', data,
        '--cache-manifest', work / 'cache/manifest.json', '--run-dir', work / 'student',
        '--device', 'cpu', '--workers', 0, '--batch-size', 2, '--epochs', 1,
        '--max-steps', 1, '--initialize-backbone', q0, '--relation-grad-ratio', .25,
        '--relation-objective', 'centered_residual')
    run('export', ROOT / 'scripts/export_checkpoint.py', '--source', work / 'student/student.pth',
        '--output', work / 'student_release.pth', '--normalization-dir', ROOT / 'regnn/assets')
    checkpoint = torch.load(work / 'student_release.pth', map_location='cpu', weights_only=True)
    assert checkpoint['global_step'] == 1
    cache = json.loads((work / 'cache/manifest.json').read_text())
    assert cache['num_experts'] == 10 and cache['contexts'] > cache['source_contexts']
    report = {'passed': stages, 'synthetic_data': True, 'device': 'cpu',
              'expert_count': 10, 'cache_contexts': cache['contexts'], 'student_steps': 1}
    (work / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'report': str(work / 'summary.json')}))


if __name__ == '__main__':
    main()
