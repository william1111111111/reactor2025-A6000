"""Ten fresh Mam anchors, 50 optimizer steps each, fixed farthest GTs."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'coreset_reactor/reports/anchor_farthest_50steps_seed1'


def worker(gpu, ranks, step_limit=50):
    template = ROOT / 'coreset_reactor/reports/anchor_behavioral_pairs_full50_seed1/model_00/config.json'
    config = json.loads(template.read_text())['training_config']
    for rank in ranks:
        settings = dict(config, run_dir=str(RUN / f'model_{rank:02d}'),
                        fixed_pair_rank=rank, fixed_pair_manifest=str(RUN / 'targets.json'),
                        max_train_steps=step_limit)
        settings.pop('fixed_pair_manifest_sha256', None)
        command = [sys.executable, '-m', 'regnn.train_conditional_regnn']
        for key, value in settings.items():
            if value is not None:
                command.extend(['--' + key.replace('_', '-'), str(value)])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', MAM_REACTOR_ROOT='/home/zhaowenjie/react2025_new/mam_reactor',
                   PYTHONPATH=str(ROOT / 'mam_reactor') + ':' + str(ROOT))
        with (RUN / f'model_{rank:02d}.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        done = json.loads((RUN / f'model_{rank:02d}/TRAINING_COMPLETE').read_text())
        expected = step_limit or 5150
        assert done['global_steps'] == expected
        print(f'GPU{gpu} model{rank}: completed {expected} steps', flush=True)


def main():
    files = [ROOT / 'coreset_reactor/behavioral_pairs.py', Path(__file__),
             *sorted((ROOT / 'mam_reactor/regnn').glob('*.py'))]
    provenance = {'steps_per_model': 50, 'models': 10, 'initialization': 'fresh seed1',
                  'scheduler': 'original cosine T_max50 epochs; stop after50 updates',
                  'git_sha': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  'target_sha256': hashlib.sha256((RUN/'targets.json').read_bytes()).hexdigest()}
    (RUN/'provenance.json').write_text(json.dumps(provenance,indent=2))
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        jobs = [executor.submit(worker, 1, range(0,10,2)), executor.submit(worker,6,range(1,10,2))]
        for job in jobs:
            job.result()


if __name__ == '__main__':
    main()
