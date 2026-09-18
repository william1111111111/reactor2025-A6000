import json
from pathlib import Path

root=Path('/public/zhaowenjie/react2025_new/mam_reactor/runs/official25_comparison_20260917_v3')
scores={}
for name in ['baseline','joint']:
    metrics=json.loads((root/name/'metrics.json').read_text())
    frd=json.loads((root/name/'frd_exact/frd_result.json').read_text())
    assert metrics['full_test'] and metrics['num_examples']==1142
    assert frd['status']=='complete' and frd['samples']==1142 and frd['tasks']==11420
    scores[name]={key:metrics[key] for key in ['FRC','FRDiv','FRVar','TLCC']}
    scores[name]['FRD']=frd['FRD']
delta={key:scores['joint'][key]-scores['baseline'][key] for key in scores['joint']}
result={'split':'test','samples':1142,'output_dim':25,'seed':1234,'scores':scores,'joint_minus_baseline':delta,
    'notes':['Offline full-length official evaluation, 10 predictions per sample.',
             'GT regenerated with official postprocessor in local sample order; both models use identical cached GT.',
             'FRD is exact unconstrained DTW, not a surrogate or sample estimate.',
             'Official TLCC evaluates only query 0 (frozen anchor); constant-channel behavior can yield boundary 49. Do not interpret as learned-query synchronization.',
             '58-D outputs are excluded. FRDvs is not implemented in the provided official code.']}
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Official 25-D evaluation','', 'Full test: 1,142 samples, 10 predictions/sample, seed 1234.','',
       '| Metric | 25-D baseline | Joint model (25-D only) | Joint - baseline |',
       '|---|---:|---:|---:|']
for key in ['FRC','FRD','FRDiv','FRVar','TLCC']:
    lines.append(f'| {key} | {scores["baseline"][key]:.6f} | {scores["joint"][key]:.6f} | {delta[key]:+.6f} |')
lines += ['', *result['notes']]
(root/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(result,indent=2))
