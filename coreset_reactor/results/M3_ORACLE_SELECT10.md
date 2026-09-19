# M3 Oracle Select-10: existing candidate-pool upper bound

This is a GT-oracle offline diagnostic. Ground truth is used to score and select the final ten; it is not an inference method.

- Split/scope: `VAL deterministic same-session pilot; GT-oracle selection, not inference`, `64 contexts`
- Pool construction: `80.00` candidates/context before selection, `80.00` after exact deduplication
- Main candidate checkpoints: `8`; analysis-only mode-adapter checkpoints: `1`; ignored non-main checkpoints: `17`
- Baseline: `m1_seed_20260918`; selection size: `10`
- Source SHA: `a0709eae899b6500179882d9bfa68d8cc9cdb57612cf28e6d2d4deed816ce931`

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
| baseline | 0.752348 | 82.143721 | 0.012697 | 0.035116 | 0 |
| max_FRDiv_unconstrained | 0.786666 | 83.007383 | 0.034480 | 0.039151 | 0 |
| quality_constrained_FRC_slack_0.25pct | 0.814471 | 80.990922 | 0.030797 | 0.038300 | 64 |
| quality_constrained_FRC_slack_0.5pct | 0.813393 | 80.999127 | 0.030787 | 0.038314 | 64 |
| quality_only | 0.967080 | 82.833479 | 0.016597 | 0.040558 | 0 |
| strict_quality_constrained | 0.815314 | 80.946035 | 0.030601 | 0.038308 | 64 |

### Relative to matched M1

| method | ΔFRC mean/p10/p90 | ΔFRD mean/p10/p90 | ΔFRDiv mean/p10/p90 |
|---|---:|---:|---:|
| max_FRDiv_unconstrained | 0.034318 / -0.138862 / 0.217870 | 0.863661 / -4.689302 / 5.308466 | 0.021783 / 0.011409 / 0.034277 |
| quality_constrained_FRC_slack_0.25pct | 0.062123 / -0.002261 / 0.182129 | -1.152799 / -4.038796 / -0.012065 | 0.018099 / 0.008794 / 0.027334 |
| quality_constrained_FRC_slack_0.5pct | 0.061046 / -0.004164 / 0.182129 | -1.144594 / -4.038796 / -0.012065 | 0.018090 / 0.008819 / 0.027285 |
| quality_only | 0.214732 / 0.063982 / 0.445952 | 0.689758 / -4.294723 / 6.911436 | 0.003900 / -0.003849 / 0.011780 |
| strict_quality_constrained | 0.062966 / 0.000231 / 0.182129 | -1.197686 / -4.038796 / -0.013716 | 0.017903 / 0.008794 / 0.027280 |

## Full-VAL distribution and pool ceiling

- `baseline`: FRC=0.752348 (p10 0.228546, p90 1.388374), FRDiv=0.012697 (p10 0.005184, p90 0.018747), FRVar=0.035116 (p10 0.009815, p90 0.060550), exact_FRD=82.143721 (p10 66.933537, p90 100.718496)
- `max_FRDiv_unconstrained`: FRC=0.786666 (p10 0.248167, p90 1.396871), FRDiv=0.034480 (p10 0.017958, p90 0.050266), FRVar=0.039151 (p10 0.013929, p90 0.062867), exact_FRD=83.007383 (p10 68.075686, p90 100.252323)
- `quality_constrained_FRC_slack_0.25pct`: FRC=0.814471 (p10 0.283001, p90 1.420269), FRDiv=0.030797 (p10 0.016735, p90 0.042742), FRVar=0.038300 (p10 0.013748, p90 0.061992), exact_FRD=80.990922 (p10 66.104914, p90 99.169958)
- `quality_constrained_FRC_slack_0.5pct`: FRC=0.813393 (p10 0.283335, p90 1.420269), FRDiv=0.030787 (p10 0.016805, p90 0.042710), FRVar=0.038314 (p10 0.013748, p90 0.061963), exact_FRD=80.999127 (p10 66.104914, p90 99.169958)
- `quality_only`: FRC=0.967080 (p10 0.355763, p90 1.611669), FRDiv=0.016597 (p10 0.006681, p90 0.026568), FRVar=0.040558 (p10 0.012154, p90 0.068598), exact_FRD=82.833479 (p10 66.434486, p90 103.673379)
- `strict_quality_constrained`: FRC=0.815314 (p10 0.283171, p90 1.420269), FRDiv=0.030601 (p10 0.016655, p90 0.042733), FRVar=0.038308 (p10 0.013748, p90 0.062041), exact_FRD=80.946035 (p10 66.104914, p90 99.169958)

- Maximum single pairwise diversity in a context: mean `0.054302`, p90 `0.081949`.
- Maximum 10-set FRDiv available in the pool (unconstrained): mean `0.034480`.

## Phase B attribution

Selected-candidate source composition (canonical source; byte-identical aliases are retained in JSON):

| source | selected count | mean FRC contribution | mean exact-FRD contribution | mean min distance to selected peers |
|---|---:|---:|---:|---:|
| m1_seed_20260918 | 41 | 0.102939 | 7.498070 | 0.009752 |
| m1_seed_20260919 | 129 | 0.080091 | 8.268005 | 0.015543 |
| m1_seed_20260920 | 173 | 0.082928 | 7.983417 | 0.013160 |
| route_0.005 | 14 | 0.073244 | 7.765985 | 0.008165 |
| route_0.010 | 12 | 0.119887 | 7.800396 | 0.006871 |
| route_0.015 | 17 | 0.056526 | 8.024199 | 0.003379 |
| route_0.020 | 34 | 0.084121 | 8.246572 | 0.006076 |
| route_0.050 | 220 | 0.077255 | 8.210447 | 0.012041 |

Behavioral statistics of selected candidates by source family:

- `m1`: count `343`, AU mean `0.1399`, 602-D descriptor L2 `5.8102` (mean `0.0860`, std `0.2204`), AU active fraction `0.1399`, VA mean `[-0.1277 0.1424]`, velocity `0.0088`, latency proxy `0.0039`.
- `route`: count `297`, AU mean `0.1169`, 602-D descriptor L2 `5.6803` (mean `0.0814`, std `0.2165`), AU active fraction `0.1169`, VA mean `[-0.1581 0.1369]`, velocity `0.0094`, latency proxy `0.0036`.

## Interpretation guardrails

- This is an upper-bound diagnostic, not a deployable selector.
- The strict quality-constrained row never relaxes FRC or exact FRD; slack rows are separate diagnostics.
- Checkpoint/data paths, trajectories, and per-context identifiers are intentionally omitted from this tracked report. The complete per-context selection ledger is local-only JSON.
