> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# M3 Oracle Select-10: existing candidate-pool upper bound

This is a GT-oracle offline diagnostic. Ground truth is used to score and select the final ten; it is not an inference method.

- Split/scope: `VAL deterministic same-session pilot; GT-oracle selection, not inference`, `64 contexts`
- Pool construction: `80.00` candidates/context before selection, `80.00` after exact deduplication
- Main candidate checkpoints: `8`; analysis-only mode-adapter checkpoints: `1`; ignored non-main checkpoints: `17`
- Baseline: `m1_seed_20260918`; selection size: `10`
- Candidate quality engine: `rolling`, workers: `16`
- Source SHA: `e37be25319bb2dcc04012539486f433f19e65a1e9b1b71c99edea6b4758c93aa`

## Available checkpoints

| label | family | seed | route weight | main pool | SHA-256 (prefix) |
|---|---:|---:|---:|---:|---:|
| m1_seed_20260918 | m1 | 20260918 | 0.1 | yes | `469aa1e0c64a` |
| m1_seed_20260918_w0.05 | m1 | 20260918 | 0.05 | no | `0aacdd389356` |
| m1_seed_20260918_w0.125 | m1 | 20260918 | 0.125 | no | `89c141db00a6` |
| m1_seed_20260918_w0.15 | m1 | 20260918 | 0.15 | no | `6c38dae9b069` |
| m1_seed_20260918_w0.2 | m1 | 20260918 | 0.2 | no | `cea250166aa2` |
| m1_seed_20260918_w0.25 | m1 | 20260918 | 0.25 | no | `adb5a093a7f3` |
| m1_seed_20260918_w0.5 | m1 | 20260918 | 0.5 | no | `0e2ab868f307` |
| m1_seed_20260918_w1 | m1 | 20260918 | 1.0 | no | `83bd24e5e6e1` |
| m1_seed_20260919 | m1 | 20260919 | 0.1 | yes | `114597fe585d` |
| m1_seed_20260919_w1 | m1 | 20260919 | 1.0 | no | `f9fd17d87194` |
| m1_seed_20260920 | m1 | 20260920 | 0.1 | yes | `0aff336abe38` |
| m1_seed_20260920_w1 | m1 | 20260920 | 1.0 | no | `da285c97aa71` |
| mode_adapter_seed_20260918 | mode_adapter | 20260918 |  | no | `a69485eaf2de` |
| other_seed_20260918_m1_checkpoint_smoke_20260919 | other | 20260918 |  | no | `d9d403dd33f3` |
| other_seed_20260918_m1_pilot_600step_val24_seed20260918 | other | 20260918 |  | no | `3f63b60ee370` |
| other_seed_20260918_m1_smoke_20260918 | other | 20260918 |  | no | `5874226da941` |
| other_seed_20260918_m1_v2_600step_val64_seed20260918 | other | 20260918 |  | no | `44c967e08e8e` |
| other_seed_20260918_m1_v2_smoke_20260918 | other | 20260918 |  | no | `25250d660263` |
| other_seed_20260919_m1_v2_600step_val64_seed20260919 | other | 20260919 |  | no | `2d0df944c792` |
| other_seed_20260920_m1_confirm2500_b1_seed20260920 | other | 20260920 |  | no | `973d9a206c48` |
| other_seed_20260920_m1_v2_600step_val64_seed20260920 | other | 20260920 |  | no | `cbad3e3ad287` |
| route_0.005 | route | 20260918 | 0.005 | yes | `2617619ca8e8` |
| route_0.010 | route | 20260918 | 0.01 | yes | `0abfce797463` |
| route_0.015 | route | 20260918 | 0.015 | yes | `f0c7645408c3` |
| route_0.020 | route | 20260918 | 0.02 | yes | `9815e6150ee8` |
| route_0.050 | route | 20260918 | 0.05 | yes | `e278508738ea` |

## Main selection results

Values are means across contexts. `FRC` is higher-is-better; exact `FRD` is lower-is-better.

| method | FRC | exact FRD | FRDiv | FRVar | strict feasible contexts |
|---|---:|---:|---:|---:|---:|
| baseline | 0.750812 | 82.047779 | 0.012697 | 0.035116 | 0 |
| max_FRDiv_unconstrained | 0.790013 | 82.937493 | 0.034481 | 0.039151 | 0 |
| quality_constrained_FRC_slack_0.25pct | 0.816581 | 80.943000 | 0.030962 | 0.038404 | 64 |
| quality_constrained_FRC_slack_0.5pct | 0.815991 | 80.916165 | 0.031041 | 0.038444 | 64 |
| quality_only | 0.971429 | 82.832959 | 0.016969 | 0.040760 | 0 |
| strict_quality_constrained | 0.817863 | 80.928443 | 0.030908 | 0.038401 | 64 |

### Relative to matched M1

| method | ΔFRC mean/p10/p90 | ΔFRD mean/p10/p90 | ΔFRDiv mean/p10/p90 |
|---|---:|---:|---:|
| max_FRDiv_unconstrained | 0.039200 / -0.120183 / 0.208685 | 0.889714 / -4.605771 / 5.516981 | 0.021783 / 0.011409 / 0.034295 |
| quality_constrained_FRC_slack_0.25pct | 0.065769 / -0.001783 / 0.189305 | -1.104779 / -3.644807 / -0.024698 | 0.018265 / 0.009107 / 0.027056 |
| quality_constrained_FRC_slack_0.5pct | 0.065178 / -0.005013 / 0.189305 | -1.131615 / -4.120554 / -0.016654 | 0.018344 / 0.009584 / 0.027221 |
| quality_only | 0.220617 / 0.066079 / 0.488391 | 0.785179 / -3.902860 / 6.635546 | 0.004272 / -0.002064 / 0.012305 |
| strict_quality_constrained | 0.067050 / 0.000389 / 0.189305 | -1.119337 / -3.903003 / -0.024698 | 0.018211 / 0.009016 / 0.026854 |

## Full-VAL distribution and pool ceiling

- `baseline`: FRC=0.750812 (p10 0.233399, p90 1.381438), FRDiv=0.012697 (p10 0.005184, p90 0.018747), FRVar=0.035116 (p10 0.009815, p90 0.060550), exact_FRD=82.047779 (p10 66.937589, p90 100.464382)
- `max_FRDiv_unconstrained`: FRC=0.790013 (p10 0.242197, p90 1.413760), FRDiv=0.034481 (p10 0.017958, p90 0.050266), FRVar=0.039151 (p10 0.013929, p90 0.062867), exact_FRD=82.937493 (p10 68.254326, p90 100.285472)
- `quality_constrained_FRC_slack_0.25pct`: FRC=0.816581 (p10 0.278675, p90 1.411605), FRDiv=0.030962 (p10 0.016982, p90 0.042400), FRVar=0.038404 (p10 0.013955, p90 0.062384), exact_FRD=80.943000 (p10 65.645935, p90 99.189301)
- `quality_constrained_FRC_slack_0.5pct`: FRC=0.815991 (p10 0.278675, p90 1.411605), FRDiv=0.031041 (p10 0.017113, p90 0.042448), FRVar=0.038444 (p10 0.013955, p90 0.062384), exact_FRD=80.916165 (p10 65.645935, p90 99.189301)
- `quality_only`: FRC=0.971429 (p10 0.344917, p90 1.619983), FRDiv=0.016969 (p10 0.006699, p90 0.026294), FRVar=0.040760 (p10 0.013230, p90 0.068439), exact_FRD=82.832959 (p10 66.589691, p90 103.393220)
- `strict_quality_constrained`: FRC=0.817863 (p10 0.278675, p90 1.411605), FRDiv=0.030908 (p10 0.016809, p90 0.042317), FRVar=0.038401 (p10 0.013955, p90 0.062384), exact_FRD=80.928443 (p10 65.645935, p90 99.189301)

- Maximum single pairwise diversity in a context: mean `0.054302`, p90 `0.081949`.
- Maximum 10-set FRDiv available in the pool (unconstrained): mean `0.034481`.

## Phase B attribution

The strict selected set draws from `4.28` distinct checkpoints per context on average (median `4`, min `3`, max `7`).

Selected-candidate source composition (canonical source; byte-identical aliases are retained in JSON):

| source | selected count | mean FRC contribution | mean exact-FRD contribution | mean min distance to selected peers |
|---|---:|---:|---:|---:|
| m1_seed_20260918 | 39 | 0.097367 | 7.534200 | 0.011400 |
| m1_seed_20260919 | 132 | 0.079361 | 8.247302 | 0.015365 |
| m1_seed_20260920 | 178 | 0.084523 | 7.972832 | 0.012961 |
| route_0.005 | 13 | 0.088920 | 7.319956 | 0.008977 |
| route_0.010 | 13 | 0.123052 | 8.235905 | 0.006341 |
| route_0.015 | 13 | 0.078544 | 7.883476 | 0.004757 |
| route_0.020 | 39 | 0.077118 | 8.267601 | 0.005134 |
| route_0.050 | 213 | 0.076248 | 8.218923 | 0.011600 |

Behavioral statistics of selected candidates by source family:

- `m1`: count `349`, AU mean `0.1402`, 602-D descriptor L2 `5.8099` (mean `0.0860`, std `0.2204`), AU active fraction `0.1402`, VA mean `[-0.1287 0.1408]`, velocity `0.0089`, latency proxy `0.0039`.
- `route`: count `291`, AU mean `0.1174`, 602-D descriptor L2 `5.6899` (mean `0.0817`, std `0.2168`), AU active fraction `0.1174`, VA mean `[-0.1531 0.1383]`, velocity `0.0093`, latency proxy `0.0038`.

## Interpretation guardrails

- This is an upper-bound diagnostic, not a deployable selector.
- The strict quality-constrained row never relaxes FRC or exact FRD; slack rows are separate diagnostics.
- Checkpoint/data paths, trajectories, and per-context identifiers are intentionally omitted from this tracked report. The complete per-context selection ledger is local-only JSON.
