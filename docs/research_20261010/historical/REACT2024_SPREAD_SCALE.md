> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# REACT2024 inference-time spread scaling

Date: 2026-09-22. Distance-KD seed 1, 10-epoch student; no retraining.
Evaluation uses all 1,124 official VAL directions and the native React2024
FRC/S_MSE/FRVar/FRDvs implementation. Teacher inference remains at its frozen
residual/style scales of 3.0.

## Method

For each context and frame, let `mu` be the mean of the student's ten output
slots. We replace each slot with `mu + alpha * (prediction - mu)`, then clip
AU/VA/expression channels to their valid domains, renormalize expression, and
apply the existing AU rounding. A single global alpha is used for all contexts;
there is no GT-based per-context choice. This only amplifies the directions
already present in the student outputs; it does not create new learned modes.

## Results

| Model / scale | FRC | S_MSE | FRVar | FRDvs |
|---|---:|---:|---:|---:|
| Mam-Reactor teacher | 0.451339 | 0.101503 | 0.084147 | 0.204071 |
| Distance-KD, alpha=1.00 | 0.483342 | 0.107258 | 0.070031 | 0.159101 |
| Distance-KD, alpha=1.50 | 0.457099 | 0.148280 | 0.069689 | 0.158709 |
| Distance-KD, alpha=1.60 | 0.452758 | 0.153923 | 0.069628 | 0.158695 |
| Distance-KD, alpha=1.65 | 0.450797 | 0.156561 | 0.069607 | 0.158717 |
| Distance-KD, alpha=1.70 | 0.448765 | 0.159094 | 0.069594 | 0.158752 |
| Distance-KD, alpha=1.75 | 0.446792 | 0.161544 | 0.069584 | 0.158797 |
| Distance-KD, alpha=2.00 | 0.438160 | 0.172689 | 0.069606 | 0.159173 |

At alpha=1.60, S_MSE rises about 43.6% over the unscaled student and FRC
remains just above the teacher, but FRC is 0.0306 below the unscaled student.
At alpha=1.75, S_MSE reaches 0.1615 (about 50.6% above the unscaled student),
while FRC falls to 0.4468, below the teacher's 0.4513. FRVar and FRDvs barely
move across this sweep, so the S_MSE gain is primarily increased pairwise
amplitude, not evidence of improved temporal behavior or learned diversity.

Conclusion: yes, this is the remembered inference-time coefficient trick,
and it raises the benchmark's within-set S_MSE. It has a clear FRC trade-off;
alpha=1.60 is the more conservative VAL operating point, while reaching about
0.16 S_MSE costs some FRC relative to the teacher. These coefficients were
examined on VAL and are not held-out test results. `FRDvs` here is the native
React2024 metric, not exact temporal DTW-FRD.

## Provenance

- Student checkpoint SHA-256: `bd270f84b6dd83de8ac62e2f639f012ea454dbc0e19e52bf32f151cc2c7f361c`
- Teacher checkpoint SHA-256: `8625c64bc2f92558478fcf5c96208f68d87a3c09ef85c004ad44c8fdd1c0c627`
- Full sweep JSONs: `spread_scale_val_full.json` and
  `spread_scale_val_refine_full.json`.
- Evaluation script SHA-256: `c088866b0dfb75a98a9e318234e4fd55a442d31bb4c16e057f833cd80f7f13e4`.
- Server copies are under
  `$LOCAL_PATH/distance_kd025_seed1_10ep`.
