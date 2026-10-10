> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# KD versus teacher-relation KD

full VAL571 speaker sources; center750; all-session GT; same processed GT cache

| Metric | Basic KD | Relation KD | Delta |
|---|---:|---:|---:|
| FRC | 1.178483 | 1.123105 | -0.055378 |
| FRDiv | 0.060729 | 0.108257 | +0.047528 |
| FRVar | 0.043209 | 0.045989 | +0.002779 |
| soft_FRDiv | 0.029321 | 0.060312 | +0.030992 |
| exact_FRD | 85.709485 | 91.051643 | +5.342158 |

Both students use the same architecture, seed=1, teacher cache, teacher-0 initialization,
batch=4, 10 epochs and 8300 updates. Only the relation loss differs.
Teacher soft trajectory pair distances are matched by AU/VA/expression group.
TRAIN-only initial gradient calibration sets the auxiliary weight.

Exact FRD: complete
FRC/FRD increases or decreases must be checked jointly before claiming a quality-preserving improvement.
No inference noise, additional generator parameters, or teacher access is used by the student.
Results are single-seed VAL analysis, not official TEST scores.
