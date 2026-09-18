import json
from pathlib import Path
base=Path('/public/zhaowenjie/react2025_new/mam_reactor/runs/anchor_comparison_20260917')
scores={}
for name in ['old','new']:
    m=json.loads((base/name/'metrics.json').read_text())
    f=json.loads((base/name/'frd_exact/frd_result.json').read_text())
    assert m['samples']==1142 and f['status']=='complete' and f['samples']==1142
    scores[name]={k:m[k] for k in ['FRC','FRDiv','FRVar','TLCC']}
    scores[name]['FRD']=f['FRD']
delta={k:scores['new'][k]-scores['old'][k] for k in scores['old']}
(base/'comparison.json').write_text(json.dumps({'scores':scores,'new_minus_old':delta},indent=2)+'\n')
lines=['# New vs old anchor: full test (1142 samples, 25D only)','', '| Metric | Old | New | New - old |','|---|---:|---:|---:|']
for k in ['FRC','FRD','FRDiv','FRVar','TLCC']:
    lines.append(f'| {k} | {scores["old"][k]:.6f} | {scores["new"][k]:.6f} | {delta[k]:+.6f} |')
lines += ['', 'Deterministic anchors repeat one prediction ten times; FRDiv is zero by construction.', 'TLCC retains official constant-channel limitations. FRD is exact DTW.', 'Same full-length test sequences, GT cache, seed, clipping and AU rounding; only 25D evaluated.']
(base/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
(base/'EVALUATION_COMPLETE').write_text('complete\n')
print('\n'.join(lines))
