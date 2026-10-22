# Completed alpha=1.50 evaluation and FRC attribution

2026-10-12. Full VAL1124, frozen count4 student, all-session cached GT.
Same official AU rounding and Mam-Reactor exact rolling DTW as baseline.
Exploratory VAL calibration; no independent TEST claim.

| Metric | Original | alpha=1.50 |
|---|---:|---:|
| FRC (sum of ten best-GT CCC scores) | 0.655348190 | 0.703201093 |
| exact FRD | 86.025762868 | 93.945012816 |
| FRDiv | 0.132445923 | 0.168621240 |
| FRVar | 0.058830553 | 0.071231729 |
| soft FRDiv | 0.058582043 | 0.104122866 |

FRD increases by 7.919249948 (~9.21% worse). This is NOT a zero-slack
quality-preserving improvement, even though FRC and diversity both improve.
No old checkpoint or baseline is replaced.

## Matching attribution

Baseline FRC: 0.655348190.
Scaled predictions evaluated against the same GT indices selected by baseline:
0.692372901. Scaled predictions with fresh all-GT matching: 0.703201093.
Mean fraction of candidates changing their argmax GT: 0.189501779.
Approximately 77.4% of the aggregate FRC improvement is already present with
the old GT assignments held fixed. This comparison does not prove which
channel or training mechanism causes the improvement.

CCC depends on variance and mean agreement, not only Pearson correlation.
Increased amplitude can correct under-dispersed predictions, but clipping,
expression normalization and AU rounding can also change correlations and means.
DTW measures absolute trajectory discrepancy, so rising CCC does not require
falling FRD.

FRDvs, FRRea and FRSyn/TLCC were not evaluated for this calibrated arm.
The local Mam-Reactor FRDvs entry point is a placeholder, so it cannot supply
a valid score. Never substitute an old model's scores or rename soft FRDiv.

Full per-context artifacts: results.json, attribution.json, exact_frd_rows.jsonl.
Exact FRD completed 1124/1124; elapsed metric loop 570.65 seconds.
First-context exact FRD passed comparison to the archived baseline.
