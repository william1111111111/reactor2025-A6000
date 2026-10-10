> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# REACT2024 diversity diagnosis

2026-09-26. Read-only diagnostics of frozen weights and fixed TRAIN targets.
Full eligible TRAIN: 6,340 directional records / 184 opposite-role groups.
Full VAL: 1,124 inputs. No TEST data, no changes to existing training targets.

## Fixed GT reselection is not the first priority

All REACT2024 TRAIN GT clips here are 750 frames. AU values are already binary;
official AU rounding does not change their target-set distances. All sets below
use the same trajectory alignment, representation and context weighting.

| Paired-preserving TRAIN set | Mean FRDiv |
|---|---:|
| Paired + random nine, analytic expectation | 0.140486300 |
| Existing frozen T1 targets | 0.165024833 |
| Original + greedy + 16 random starts, each with 1-swap refinement | 0.165046151 |

The gain in the aggregate is 0.000021318, about 0.0129%. The original is retained
as a feasible candidate. This is a stronger heuristic search, **not a certified
optimum or proof that all other selection objectives are exhausted**. Newly
found candidate sets are saved only as diagnostic artifacts, not installed.

For context, the frozen REACT2025 TRAIN T1 set is 0.273745 under a standardized
750-frame representation. This is a different data population; neither number
is a hard bound on learned VAL outputs, and TRAIN/VAL values must not be divided
and called an exact transfer efficiency.

## Full-VAL loss is mainly in AU, not only rounding

The columns are additive components of official prediction-pair FRDiv:

| Model | AU | VA | Expression | Total |
|---|---:|---:|---:|---:|
| Old T1 teacher | 0.137557 | 0.003064 | 0.016639 | 0.157260 |
| New count8 teacher | 0.127612 | 0.002695 | 0.016101 | 0.146409 |
| New basic student | 0.113271 | 0.002553 | 0.014732 | 0.130557 |
| New contrast student | 0.110857 | 0.002093 | 0.014338 | 0.127287 |

About 91.6% of the old-to-new teacher drop, and 87.6% of the new-teacher-to-
contrast-student drop, is attributable arithmetically to the AU component.
This is metric decomposition, not causal attribution to a training mechanism.

| Model | Temporal-mean component | Zero-time-mean residual | Soft FRDiv before rounding |
|---|---:|---:|---:|
| Old T1 teacher | 0.034002 | 0.123258 | 0.078555 |
| New count8 teacher | 0.030524 | 0.115885 | 0.070042 |
| New basic student | 0.019785 | 0.110771 | 0.057754 |
| New contrast student | 0.020536 | 0.106751 | 0.055538 |

Both temporal-mean separation and temporal residual separation shrink. Soft
trajectories also shrink, so AU rounding alone cannot explain the result.
Float64 decomposition differs from the saved official float32 totals by less
than 1e-6. This is numerical precision, not a changed metric protocol.

## Fixed effective-reference count is not cross-dataset invariant

Prior overlap is the mean Bhattacharyya affinity between the nine specialist
reference distributions: mean over slot pairs of sum_j sqrt(p_kj * p_lj).
It is in [0,1], not a fraction of identical selected trajectories.

| TRAIN diagnostic | REACT2025 count8 | REACT2024 count8 | REACT2024 count4 |
|---|---:|---:|---:|
| Context-weighted pool size | 97.694 | 39.558 | 39.558 |
| Mean prior mass on fixed GT | 0.518407 | 0.452729 | 0.656025 |
| Mean between-slot prior overlap | 0.281542 | 0.458473 | 0.280806 |

Count4 on REACT2024 approximately matches count8's REACT2025 prior overlap.
This supports a **single controlled count8-to-count4 ablation**, not a claim
that the count is already proven to cause diversity loss. All references remain
in the soft CCC calculation; effective count is entropy, not hard top-k.

Cross-dataset priors are compared at a common 750-frame reference length with
each dataset's own raw TRAIN descriptor scale. This does not assert exact
runtime priors for short REACT2025 inputs. Expected descriptor displacement is
actually lower in REACT2024 (0.136887 vs 0.570185 at count8); therefore do not
claim that its semantic radius is simply larger from 8/N alone. Distribution
overlap is the specific signal motivating the next experiment.

## Bounded follow-up protocol

- Retain every frozen target manifest and the paired anchor; change only
  effective-reference count 8 to 4, with temperature 0.05 unchanged.
- Fresh nine-specialist training, seed1, 600 updates, same initialization,
  source/crop schedule, optimizer, architecture and loss coefficients.
- Full VAL1124, same processed-GT cache, official AU rounding and exact FRD.
- Only distill if teacher FRC >= 0.6574084104, exact FRD <= 85.6265717740,
  and FRDiv > 0.1464078575. No quality slack.
- If the gate passes, repeat the retained contrast KD recipe with identical
  student initialization/input views/schedule, 10 epochs / 15,850 updates.
  Student acceptance requires FRC >= 0.6711928350, exact FRD <= 85.7307860464,
  and FRDiv > 0.1272866130.
- Do not overwrite current weights or promote a candidate failing the gate.

The follow-up is exploratory single-seed VAL research, not an independent TEST
claim. Diagnostic completion does not imply that the follow-up has succeeded.
