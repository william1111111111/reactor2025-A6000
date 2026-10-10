> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# REACT2025 key component ablations

Full VAL571 / all-session GT / exact FRD. Single seed; matched initialization and source schedules.
Teacher and student comparisons are separate. No outcome-based promotion or discarded negative results.

| Stage | Arm | FRC | exact FRD | FRDiv | FRVar |
|---|---|---:|---:|---:|---:|
| Teacher | Mode prior | 1.370792 | 95.772670 | 0.190244 | 0.069174 |
| Teacher | Uniform prior | 2.038626 | 95.410771 | 0.091645 | 0.070391 |
| Student | Minimal KD | 1.454643 | 99.714605 | 0.170338 | 0.069951 |
| Student | Direct same fixed GT | 1.066966 | 97.427886 | 0.177390 | 0.064775 |

Teacher control: corrected CCC in both arms; same fixed targets/q0/600 updates, only prior changes.
Student control: same teacher0 initialization, 8300 updates, MSE+CCC; only supervision targets change.
Direct GT is no output distillation, not a claim of no pretrained teacher initialization.
All 3320 source views checked; 300 targets bitwise checked against production teacher preprocessing.
Session-cluster bootstrap and per-session deltas are in summary.json; not training-seed uncertainty.
Training costs/checkpoints/source hashes are retained. Different supervision has different compute cost.
