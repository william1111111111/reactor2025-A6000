> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# REACT2024 accepted count4 teacher → student

Full VAL1124, all-session GT, center750, AU rounded, exact DTW-FRD.
Single seed, not native neighbour-GT / official TEST. No inference rescaling.
Count4 teacher failed the original strict gate; user explicitly accepted the tradeoff.
Matched contrast KD: identical initialization and 6340 source/crop views, 10 epochs / 15850 updates.
TRAIN-only scale/gradient calibration repeated for new targets. All old weights retained.

| Model | FRC | exact FRD | FRDiv | FRVar |
|---|---:|---:|---:|---:|
| teacher_count8 | 0.657408 | 85.626572 | 0.146408 | 0.058906 |
| teacher_count4 | 0.648193 | 86.151776 | 0.152807 | 0.060486 |
| student_count8 | 0.671193 | 85.730786 | 0.127287 | 0.057549 |
| student_count4 | 0.655348 | 86.025763 | 0.132445 | 0.058831 |
