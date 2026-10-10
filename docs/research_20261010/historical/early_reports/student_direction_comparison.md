> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Matched direction-versus-distance distillation

full VAL571 speaker inputs; center750; all-session opposite-role GT; AU-rounded; vectorized scalar-parity FRC

| Model | FRC | exact FRD | FRDiv | FRVar | Train seconds | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Basic KD | 1.178483 | 85.709485 | 0.060729 | 0.043209 | not recorded | 3803201 |
| Distance KD | 1.123105 | 91.051643 | 0.108257 | 0.045989 | 1073.2 | 3803201 |
| Centered-direction KD | 1.167020 | 85.283965 | 0.066293 | 0.043262 | 1077.7 | 3803201 |

Same backbone, seed=1, initialization, cache, source schedule, batch4, 10 epochs / 8300 updates.
First-batch KD loss matches exactly across arms. Auxiliary gradients calibrated on TRAIN only.
No added parameters, teacher calls or expensive exact temporal-GT computations during training.
Historical teacher label alignment is unchanged; this isolates the student objective only.
Basic-KD FRD is reused from its earlier full evaluation (no cache fingerprints); checkpoint path, context and per-slot FRC parity checked.
Report full quality-diversity curves before attributing a gain to mode preservation.
Measured evaluation inference timing is diagnostic, not a controlled efficiency benchmark.
