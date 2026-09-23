"""Fresh 50-epoch farthest-target anchors and matched all-session evaluation."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from coreset_reactor import train_farthest_smoke as training


def main():
    root = training.ROOT
    previous = root / 'coreset_reactor/reports/anchor_farthest_50steps_seed1'
    baseline = root / 'coreset_reactor/reports/anchor_behavioral_pairs_full50_seed1'
    run = root / 'coreset_reactor/reports/anchor_farthest_full50_seed1'
    run.mkdir(exist_ok=False)
    shutil.copyfile(previous / 'targets.json', run / 'targets.json')
    training.RUN = run
    assignments = {0:1, 1:1, 2:1, 3:1, 4:5, 5:5, 6:5, 7:5, 8:6, 9:6}
    sources = list((root/'mam_reactor/regnn').glob('*.py')) + [Path(__file__), Path(training.__file__)]
    provenance = {
        'git_sha': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'source_hashes': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'manifest_sha256': hashlib.sha256((run/'targets.json').read_bytes()).hexdigest(),
        'baseline': str(baseline), 'assignments': assignments,
        'initialization': 'fresh seed1; no partial-epoch resume',
        'epochs_per_model': 50, 'updates_per_model': 5150,
        'changed_training_factor': 'fixed GT identities only',
    }
    (run/'provenance.json').write_text(json.dumps(provenance,indent=2))
    with concurrent.futures.ThreadPoolExecutor(10) as executor:
        jobs = [executor.submit(training.worker,gpu,[rank],0) for rank,gpu in assignments.items()]
        for job in jobs:
            job.result()
    # Verify every model matches its baseline rank, except target identity/output.
    allowed = {'run_dir','fixed_pair_manifest','fixed_pair_manifest_sha256'}
    for rank in assignments:
        old = json.loads((baseline/f'model_{rank:02d}/config.json').read_text())
        new = json.loads((run/f'model_{rank:02d}/config.json').read_text())
        assert old['model_config'] == new['model_config']
        assert {k:v for k,v in old['training_config'].items() if k not in allowed} == {k:v for k,v in new['training_config'].items() if k not in allowed}
    command = [sys.executable,'-m','coreset_reactor.anchor_pair_ensemble',
               '--output-root',str(run),'--data-root','/public/zhaowenjie/react2025_new/data',
               '--cache','/home/zhaowenjie/react2025_new/coreset_reactor/cache/train_descriptors_v2.pt',
               '--asset-mam-root','/home/zhaowenjie/react2025_new/mam_reactor',
               '--output',str(run/'all_session_val571.json'),
               '--report',str(run/'all_session_val571.md'),'--all-session-gt','--device','cuda:5']
    with (run/'evaluation.log').open('x') as log:
        subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    old = json.loads((baseline/'all_session_val571.json').read_text())['metrics']
    new = json.loads((run/'all_session_val571.json').read_text())['metrics']
    fields = ['FRC','exact_FRD','FRDiv','FRVar','TLCC','gt_descriptor_coverage']
    comparison = {k:{'baseline':old[k],'farthest':new[k],'delta':new[k]-old[k]} for k in fields}
    (run/'matched_comparison.json').write_text(json.dumps(comparison,indent=2))
    print(json.dumps(comparison,indent=2),flush=True)


if __name__ == '__main__':
    main()
