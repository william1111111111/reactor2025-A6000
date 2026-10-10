> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# REACT2024 mode-reference teacher → student

Full VAL1124; center750; all-session opposite-role GT; AU round; exact DTW-FRD.
Not native neighbour-GT/FRDvs, not official TEST. Single seed; no inference scaling.

| Arm | FRC | exact FRD | FRDiv | FRVar |
|---|---:|---:|---:|---:|
| old_teacher | 0.620408 | 87.107812 | 0.157259 | 0.061897 |
| new_teacher | 0.657408 | 85.626572 | 0.146408 | 0.058906 |
| old_contrast | 0.630941 | 86.373361 | 0.136166 | 0.059413 |
| new_basic | 0.681457 | 89.499561 | 0.130556 | 0.059306 |
| new_contrast | 0.671193 | 85.730786 | 0.127287 | 0.057549 |

Teacher pilot: 600 updates, unchanged T1 target manifests and paired anchor.
Students: matched initialization/source views/order, 10 epochs / 15850 updates.
All-reference teacher CCC adds compute; equal updates are not equal training time.
See per-arm config, source snapshots, and random_frc100 for provenance, speed and random-reference FRC.
Historical teacher lacks initial/source hashes; do not claim bitwise matched retraining.
