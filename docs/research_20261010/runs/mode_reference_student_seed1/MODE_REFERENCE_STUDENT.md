> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Mode-reference teacher distillation

REACT2025, full VAL571, all-session GT, official AU rounding and exact FRD.
Single seed; exploratory VAL, not official sampled-ten TEST.

| Model | FRC | exact FRD | FRDiv | FRVar |
|---|---:|---:|---:|---:|
| New teacher | 1.370792 | 95.772670 | 0.190244 | 0.069174 |
| Old basic | 1.122329 | 100.690347 | 0.189725 | 0.068813 |
| Old contrast | 1.121615 | 96.617181 | 0.183968 | 0.065987 |
| New basic | 1.454373 | 99.752944 | 0.170335 | 0.069890 |
| New contrast | 1.444718 | 97.540968 | 0.170190 | 0.067888 |

Same initialization, all 3320 source/crop views and 8300-update schedule verified.
Basic KD and signed centered-residual KD; weights/scales calibrated on TRAIN only.
The student has 3,803,201 parameters; ten outputs without teachers at inference.
No inference scaling, teacher averaging or GT input. Old weights are retained.
Training speed, peak memory and coefficients: per-arm config/provenance JSON.
Inference timing is diagnostic on shared GPUs, not a controlled speed benchmark.
