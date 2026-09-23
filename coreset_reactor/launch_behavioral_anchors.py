"""Two GPU queues: train ten matched anchors, then evaluate full VAL571."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'coreset_reactor/reports/anchor_behavioral_pairs_full50_seed1'
DATA = '/public/zhaowenjie/react2025_new/data'
ASSETS = '/home/zhaowenjie/react2025_new/mam_reactor'


def worker(gpu, ranks):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               MAM_REACTOR_ROOT=ASSETS, OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', PYTHONPATH=str(ROOT / 'mam_reactor') + ':' + str(ROOT))
    for rank in ranks:
        model = RUN / f'model_{rank:02d}'
        command = [sys.executable, '-m', 'regnn.train_conditional_regnn',
                   '--data-dir', DATA, '--run-dir', str(model),
                   '--epochs', '50', '--batch-size', '32', '--workers', '8',
                   '--precision', 'bf16', '--seed', '1', '--fixed-pair-seed', '1',
                   '--fixed-pair-rank', str(rank), '--fixed-pair-count', '10',
                   '--fixed-pair-manifest', str(RUN / 'targets.json')]
        with (RUN / f'model_{rank:02d}.log').open('a') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)


def main():
    recovery = '--recover-even-queue' in sys.argv
    parallel_remaining = '--parallel-remaining' in sys.argv
    source_files = sorted((ROOT / 'mam_reactor/regnn').glob('*.py')) + [Path(__file__), ROOT / 'coreset_reactor/behavioral_pairs.py']
    provenance = {'git_commit_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                  'manifest_sha256': hashlib.sha256((RUN / 'targets.json').read_bytes()).hexdigest(),
                  'gpus': [5, 5], 'models': 10, 'epochs': 50, 'seed': 1}
    (RUN / ('recovery_provenance.json' if recovery else 'provenance.json')).write_text(json.dumps(provenance, indent=2))
    if parallel_remaining:
        # Ranks 0/1 are already running. Each remaining rank gets an
        # independent process so several models can share one A6000.
        assignments = {2: 1, 3: 1, 4: 1, 5: 1,
                       6: 5, 7: 5, 8: 6, 9: 6}
        provenance['parallel_assignments'] = assignments
        (RUN / 'parallel_provenance.json').write_text(
            json.dumps(provenance, indent=2))
        with concurrent.futures.ThreadPoolExecutor(len(assignments)) as executor:
            jobs = [executor.submit(worker, gpu, [rank])
                    for rank, gpu in assignments.items()]
            for job in jobs:
                job.result()
        while not all((RUN / f'model_{rank:02d}/TRAINING_COMPLETE').exists()
                      for rank in range(10)):
            time.sleep(30)
    elif recovery:
        worker(5, range(0, 10, 2))
        while not all((RUN / f'model_{rank:02d}/TRAINING_COMPLETE').exists() for rank in range(10)):
            time.sleep(30)
    else:
        with concurrent.futures.ThreadPoolExecutor(2) as executor:
            jobs = [executor.submit(worker, 5, range(0, 10, 2)),
                    executor.submit(worker, 5, range(1, 10, 2))]
            for job in jobs:
                job.result()
    command = [sys.executable, '-m', 'coreset_reactor.anchor_pair_ensemble',
               '--output-root', str(RUN), '--data-root', DATA,
               '--cache', '/home/zhaowenjie/react2025_new/coreset_reactor/cache/train_descriptors_v2.pt',
               '--asset-mam-root', ASSETS, '--device', 'cuda:5',
               '--max-examples', '571', '--output', str(RUN / 'full_val571.json'),
               '--report', str(RUN / 'full_val571.md')]
    with (RUN / 'evaluation.log').open('x') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
