> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Matched REACT2024 student loss deletion

Frozen count4 teachers. Full VAL1124, all-session GT, exact FRD, AU rounding.
Single seed. Same initialization, TRAIN cache/views/order, 10 epochs / 15850 updates.
Residual scale/coefficient frozen from the retained full model TRAIN calibration.
No backbone change, teacher change, inference noise or rescaling.

| Arm | Velocity | Residual | FRC | exact FRD | FRDiv | FRVar |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.05 | 0.213747 | 0.655348 | 86.025763 | 0.132445 | 0.058831 |
| minimal | 0.0 | 0.000000 | 0.656854 | 88.964175 | 0.135939 | 0.060444 |
| velocity | 0.05 | 0.000000 | 0.655127 | 88.886196 | 0.135477 | 0.060246 |
| residual | 0.0 | 0.213747 | 0.656394 | 86.033383 | 0.132724 | 0.058911 |

The full arm is a retained completed run, not retrained in this sweep.
Deletion candidates are not automatically promoted; report quality/diversity tradeoffs honestly.
No claim that frozen calibration can be removed across datasets; minimal KD needs none.
Velocity is still computed for logging when weighted zero; this is objective deletion, not a runtime optimization.
