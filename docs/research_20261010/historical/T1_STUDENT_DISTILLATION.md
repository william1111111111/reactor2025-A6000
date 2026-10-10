> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# T1 Pure Max-FRDiv teacher → shared student

REACT2025 full VAL571, all-session GT, center750, AU-rounded; local analysis

| Method | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar | Teacher FRDiv retained |
|---|---:|---:|---:|---:|---:|
| teacher | 1.072215 | 96.813896 | 0.205415 | 0.068097 | 100.0% |
| basic | 1.122329 | 100.690347 | 0.189725 | 0.068813 | 92.4% |
| distance | 1.110616 | 100.095737 | 0.189301 | 0.067937 | 92.2% |
| contrast | 1.121615 | 96.617181 | 0.183968 | 0.065987 | 89.6% |
| au_decision | 1.130168 | 100.263240 | 0.190479 | 0.068266 | 92.7% |

Teacher FRC/FRDiv/FRVar were reproduced using the student evaluator (maximum FRC difference 3.5e-7).
Teacher exact FRD reuses its frozen full-VAL571 result; every student exact FRD was recomputed against all session GTs.

## Interpretation

Largest student FRDiv: `au_decision` = 0.190479 (92.7% retained).
The highest-diversity student satisfying both T1 quality point-estimate thresholds is `contrast`: FRC 1.121615, exact FRD 96.617181, FRDiv 0.183968.
These are VAL comparisons from one seed; closeness in FRDiv alone does not establish lossless distillation.
The old Prototype/KMeans teacher and its 0.0607 Basic-KD student are different target-generation experiments and are not the matched baseline here.

## Matched training

- One 3,803,201-parameter shared student, ten expert IDs; teacher ensemble: 37,201,290 parameters.
- T1 rank-0 checkpoint initialization; seed 1, 3,320 fixed TRAIN views, batch 4, 10 epochs / 8,300 updates.
- AdamW lr=1e-4, betas=(0.9,0.99), weight decay=1e-4; cosine schedule, BF16, clip norm 1.
- Base KD: MSE + CCC + 0.05 velocity against soft teacher trajectories.
- Distance KD matches 45 teacher pair distances within AU/VA/expression groups.
- Contrast KD matches signed slot residuals about the set mean. It is exactly equivalent to matching pairwise difference vectors up to 2K/(K−1).
- AU-decision KD is a follow-up motivated by the first three VAL results: confidence-weighted BCE against rounded TRAIN teacher AUs. Its weight is also calibrated on TRAIN only; it uses no inference scaling.
- Auxiliary weights calibrated to 0.25 of initial KD gradient norm, using 32 TRAIN scale examples and 16 TRAIN gradient probes.
- Student state hash, observed source order, teacher cache and evaluation GT keys matched across arms.
- No inference spread scaling or added noise. All ten outputs come from the shared student.

| Arm | Auxiliary weight | Training seconds | Updates/s | Inference ms (diagnostic) |
|---|---:|---:|---:|---:|
| basic | 0.000000 | 1087.0 | 7.64 | 36.78 |
| distance | 0.127617 | 1080.2 | 7.68 | 20.04 |
| contrast | 0.269761 | 1080.2 | 7.68 | 20.08 |
| au_decision | 0.146184 | 1079.9 | 7.69 | 19.90 |

## Convergence on full VAL571

| Arm | Epoch | FRC | FRDiv | Soft FRDiv |
|---|---:|---:|---:|---:|
| basic | 5 | 1.097365 | 0.189026 | 0.097971 |
| basic | 10 | 1.122329 | 0.189725 | 0.096505 |
| distance | 5 | 1.065997 | 0.190582 | 0.102846 |
| distance | 10 | 1.110616 | 0.189301 | 0.099199 |
| contrast | 5 | 1.112358 | 0.182224 | 0.089749 |
| contrast | 10 | 1.121615 | 0.183968 | 0.090682 |
| au_decision | 5 | 1.114296 | 0.187934 | 0.102349 |
| au_decision | 10 | 1.130168 | 0.190479 | 0.100474 |

## Diversity components

AU/VA/expression contributions sum to official FRDiv. Dynamic contrast removes each slot’s time-average residual.

| Method | AU | VA | Expression | Dynamic contrast | Soft FRDiv before AU rounding |
|---|---:|---:|---:|---:|---:|
| teacher | 0.171005 | 0.011130 | 0.023282 | 0.130476 | 0.109424 |
| basic | 0.157435 | 0.011122 | 0.021169 | 0.125183 | 0.096506 |
| distance | 0.156414 | 0.010370 | 0.022517 | 0.122475 | 0.099200 |
| contrast | 0.153413 | 0.010006 | 0.020550 | 0.119171 | 0.090682 |
| au_decision | 0.158014 | 0.011184 | 0.021282 | 0.125052 | 0.100475 |

## Provenance and limits

Full numerical diagnostics and paired session-bootstrap intervals are saved in the local `analysis.json`.
This is single-seed VAL exploration, not a TEST score.
The T1 teacher was a 600-step-per-model pilot with a quality cost versus T0; retaining T1 quality does not establish T0-level quality.
Base commit plus exact training-file hashes are recorded because this experiment used worktree changes.

- Teacher reference file SHA256: `99ef6c50d4868f0e3eee4db70d84a60d25b71212bf638178a65916379ae853bc`.
- basic: checkpoint `9ae95f39936bed601e8326588ba52a46388645508e31a9d524524b055680404d`, initial state `65ee1a21436be9326370bf035836dee49c177325397b996769749133f24f2dd7`, source order `25d047b93ce072e09f73a1848f0a5b449b5302d4fc75cde568c522979152647a`.
- distance: checkpoint `58c32c6a73f8c3928211c0ad72a39c5d20f41e4eb32d8a591b6fd0cf18e2fd1f`, initial state `65ee1a21436be9326370bf035836dee49c177325397b996769749133f24f2dd7`, source order `25d047b93ce072e09f73a1848f0a5b449b5302d4fc75cde568c522979152647a`.
- contrast: checkpoint `5ef5a3a3a69e0314e1d643286a48c3f8bd98a0f935a41304c3ecc35fab8a7d21`, initial state `65ee1a21436be9326370bf035836dee49c177325397b996769749133f24f2dd7`, source order `25d047b93ce072e09f73a1848f0a5b449b5302d4fc75cde568c522979152647a`.
- au_decision: checkpoint `ab802fa917acabde1b4f98a4dc042ff59230b113fb3c17f1a94f9850000ac515`, initial state `65ee1a21436be9326370bf035836dee49c177325397b996769749133f24f2dd7`, source order `25d047b93ce072e09f73a1848f0a5b449b5302d4fc75cde568c522979152647a`.
