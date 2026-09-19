# M3 Oracle Select-10: existing candidate-pool upper bound

This is a GT-oracle offline diagnostic. Ground truth is used to score and select the final ten; it is not an inference method.

- Split/scope: `VAL deterministic same-session pilot; GT-oracle selection, not inference`, `571 contexts`
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
| baseline | 0.810111 | 82.480135 | 0.013611 | 0.039200 | 0 |
| max_FRDiv_unconstrained | 0.829212 | 82.756349 | 0.036974 | 0.042754 | 0 |
| quality_constrained_FRC_slack_0.25pct | 0.862802 | 81.090919 | 0.033638 | 0.042239 | 571 |
| quality_constrained_FRC_slack_0.5pct | 0.861680 | 81.112051 | 0.033735 | 0.042264 | 571 |
| quality_only | 1.009611 | 82.453890 | 0.017302 | 0.044157 | 0 |
| strict_quality_constrained | 0.864074 | 81.107519 | 0.033547 | 0.042234 | 571 |

### Relative to matched M1

| method | ΔFRC mean/p10/p90 | ΔFRD mean/p10/p90 | ΔFRDiv mean/p10/p90 |
|---|---:|---:|---:|
| max_FRDiv_unconstrained | 0.019100 / -0.126106 / 0.166898 | 0.276214 / -4.319363 / 4.920304 | 0.023363 / 0.012542 / 0.035253 |
| quality_constrained_FRC_slack_0.25pct | 0.052691 / -0.002025 / 0.162572 | -1.389215 / -4.201571 / -0.024877 | 0.020027 / 0.009314 / 0.031300 |
| quality_constrained_FRC_slack_0.5pct | 0.051568 / -0.004949 / 0.162572 | -1.368083 / -3.980427 / -0.020474 | 0.020123 / 0.009482 / 0.031300 |
| quality_only | 0.199500 / 0.060789 / 0.424417 | -0.026244 / -5.551017 / 5.994933 | 0.003690 / -0.004148 / 0.012667 |
| strict_quality_constrained | 0.053963 / 0.000485 / 0.162572 | -1.372615 / -3.980427 / -0.023619 | 0.019936 / 0.009253 / 0.031296 |

## Full-VAL distribution and pool ceiling

- `baseline`: FRC=0.810111 (p10 0.219691, p90 1.541036), FRDiv=0.013611 (p10 0.006639, p90 0.019126), FRVar=0.039200 (p10 0.015716, p90 0.062109), exact_FRD=82.480135 (p10 64.892858, p90 100.585429)
- `max_FRDiv_unconstrained`: FRC=0.829212 (p10 0.243092, p90 1.520684), FRDiv=0.036974 (p10 0.020426, p90 0.053042), FRVar=0.042754 (p10 0.019610, p90 0.064295), exact_FRD=82.756349 (p10 66.319093, p90 99.936306)
- `quality_constrained_FRC_slack_0.25pct`: FRC=0.862802 (p10 0.255797, p90 1.618926), FRDiv=0.033638 (p10 0.018157, p90 0.048200), FRVar=0.042239 (p10 0.019157, p90 0.063670), exact_FRD=81.090919 (p10 64.278546, p90 98.942184)
- `quality_constrained_FRC_slack_0.5pct`: FRC=0.861680 (p10 0.254581, p90 1.612827), FRDiv=0.033735 (p10 0.018573, p90 0.048999), FRVar=0.042264 (p10 0.019157, p90 0.063824), exact_FRD=81.112051 (p10 64.240302, p90 98.949335)
- `quality_only`: FRC=1.009611 (p10 0.360207, p90 1.821062), FRDiv=0.017302 (p10 0.007450, p90 0.028369), FRVar=0.044157 (p10 0.018028, p90 0.067602), exact_FRD=82.453890 (p10 64.613768, p90 99.134908)
- `strict_quality_constrained`: FRC=0.864074 (p10 0.256873, p90 1.623356), FRDiv=0.033547 (p10 0.018117, p90 0.048200), FRVar=0.042234 (p10 0.019251, p90 0.064113), exact_FRD=81.107519 (p10 64.278546, p90 98.910687)

- Maximum single pairwise diversity in a context: mean `0.058670`, p90 `0.087328`.
- Maximum 10-set FRDiv available in the pool (unconstrained): mean `0.036974`.

## Phase B attribution

The strict selected set draws from `4.20` distinct checkpoints per context on average (median `4`, min `2`, max `7`).

Selected-candidate source composition (canonical source; byte-identical aliases are retained in JSON):

| source | selected count | mean FRC contribution | mean exact-FRD contribution | mean min distance to selected peers |
|---|---:|---:|---:|---:|
| m1_seed_20260918 | 324 | 0.094351 | 7.859233 | 0.012604 |
| m1_seed_20260919 | 1174 | 0.090066 | 8.223782 | 0.016568 |
| m1_seed_20260920 | 1574 | 0.090390 | 8.032738 | 0.012996 |
| route_0.005 | 99 | 0.087428 | 7.861487 | 0.010322 |
| route_0.010 | 74 | 0.102069 | 8.150425 | 0.007674 |
| route_0.015 | 122 | 0.080413 | 8.362513 | 0.006310 |
| route_0.020 | 372 | 0.079216 | 8.197061 | 0.007131 |
| route_0.050 | 1971 | 0.080831 | 8.126231 | 0.012938 |

Behavioral statistics of selected candidates by source family:

- `m1`: count `3072`, AU mean `0.1498`, 602-D descriptor L2 `5.9305` (mean `0.0895`, std `0.2243`), AU active fraction `0.1498`, VA mean `[-0.1169 0.1385]`, velocity `0.0095`, latency proxy `0.0041`.
- `route`: count `2638`, AU mean `0.1201`, 602-D descriptor L2 `5.7086` (mean `0.0830`, std `0.2172`), AU active fraction `0.1201`, VA mean `[-0.1547 0.1367]`, velocity `0.0098`, latency proxy `0.0037`.

## Interpretation guardrails

- This is an upper-bound diagnostic, not a deployable selector.
- The strict quality-constrained row never relaxes FRC or exact FRD; slack rows are separate diagnostics.
- Checkpoint/data paths, trajectories, and per-context identifiers are intentionally omitted from this tracked report. The complete per-context selection ledger is local-only JSON.
