> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Matched REACT2025 student loss deletion

fixed effective-count8 mode-reference teacher bank. Full VAL571, all-session GT, exact FRD, AU rounding.
Single seed. Same initialization, TRAIN cache/views/order, 10 epochs / 8300 updates.
Residual scale/coefficient frozen from the retained full model TRAIN calibration.
No backbone change, teacher change, inference noise or rescaling.

| Arm | Velocity | Residual | FRC | exact FRD | FRDiv | FRVar |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.05 | 0.329149 | 1.444718 | 97.540968 | 0.170190 | 0.067888 |
| minimal | 0.0 | 0.000000 | 1.454643 | 99.714605 | 0.170338 | 0.069951 |
| velocity | 0.05 | 0.000000 | 1.454373 | 99.752944 | 0.170335 | 0.069890 |
| residual | 0.0 | 0.329149 | 1.445507 | 97.463281 | 0.170130 | 0.067918 |

The full arm is a retained completed run, not retrained in this sweep.
The velocity-only arm is also retained when running REACT2025; matched hashes and evaluation checked.
Deletion candidates are not automatically promoted; report quality/diversity tradeoffs honestly.
No claim that frozen calibration can be removed across datasets; minimal KD needs none.
Velocity is still computed for logging when weighted zero; this is objective deletion, not a runtime optimization.
