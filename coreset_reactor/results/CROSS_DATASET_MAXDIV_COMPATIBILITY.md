# Cross-dataset Max-FRDiv compatibility diagnostic

This is a TRAIN-only, read-only diagnostic. No model was retrained and no
VAL/TEST GT was used. Both datasets use the same implementation:

- every cross-listener GT is processed by `relative_time_resample_reaction(..., 750)`;
- membership distance is mean squared 25-D trajectory distance;
- T0/T1 membership comes from the frozen rank-00 manifests;
- a frozen paired ConditionalREGNN anchor provides source-conditioned distances;
- compatibility FRD is the weighted rolling unconstrained DTW on selected q1--q9.

## Frozen teacher results being explained

| Dataset | Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ |
|---|---|---:|---:|---:|
| React2025 | T0 random | 1.373 | 84.93 | 0.0364 |
| React2025 | T1 pure Max-FRDiv | 1.072 | 96.82 | 0.2054 |
| React2024 | T0 random | 0.244 | 83.54 | 0.0425 |
| React2024 | T1 pure Max-FRDiv | 0.620 | 87.11 | 0.1573 |

## Main compatibility comparison

| Dataset | Arm | Target FRDiv | Anchor deviation | Normalized CI | TailRate90 | Anchor rolling FRD | Descriptor compatibility |
|---|---|---:|---:|---:|---:|---:|---:|
| React2024 | T0 | 0.140615 | 0.130996 | 0.9450928 | 0.1168 | 8.139143 | 0.199998 |
| React2024 | T1 | 0.165025 | 0.146087 | 1.0587754 | 0.1887 | 8.539516 | 0.212586 |
| React2025 | T0 | 0.200888 | 0.177765 | 0.8933819 | 0.0525 | 14.518108 | 0.247872 |
| React2025 | T1 | 0.273745 | 0.229372 | 1.1680969 | 0.2411 | 17.166345 | 0.285728 |

Changes from T0 to T1:

| Dataset | Δ target FRDiv | Δ anchor deviation | Δ CI | Δ TailRate90 | Δ anchor rolling FRD |
|---|---:|---:|---:|---:|---:|
| React2024 | +0.024410 | +0.015092 | +0.113683 | +0.0719 | +0.400373 |
| React2025 | +0.072858 | +0.051607 | +0.274715 | +0.1886 | +2.648238 |

The key normalized result is not the raw FRDiv scale. Relative to the session
median pairwise distance, R25 T1 has CI `1.1681` versus R24 T1 `1.0588`, and
24.1% of its selected specialists are beyond the session P90 compared with
18.9% for R24. More importantly, the T1-minus-T0 increase is about 2.4x
larger in CI and about 2.6x larger in TailRate90 on R25.

The session geometry is also heavier-tailed on R25:

| Dataset | mean P90/median | median P90/median | P90/median P90 |
|---|---:|---:|---:|
| React2024 | 1.2296 | 1.2185 | 1.3464 |
| React2025 | 1.3402 | 1.3070 | 1.4117 |

## Rank diagnosis

R25's high-risk T1 ranks are concentrated around q6/q8/q2/q1: their mean
session percentiles are approximately `0.744/0.746/0.733/0.694`. R24's ranks
are much less extreme and cluster around `0.49--0.62`; no single specialist
has the same tail concentration.

This is consistent with the quality results: R25 Max-FRDiv transfers more
diversity, but a larger fraction of that diversity is outside the
source-conditioned neighborhood, so FRC falls and exact FRD rises. R24 also
moves away from its anchor, but the move is milder and its teacher remains in
a more compatible conditional region.

## Compatibility-gated target simulation

This is target-level only: alternatives are ranked by frozen-anchor trajectory
distance, the nearest Q30/Q50/Q70/Q90 pool is retained, and the same paired
Max-FRDiv selector chooses nine specialists. No teacher is trained.

| Dataset | Gate | Target FRDiv | CI | TailRate90 |
|---|---|---:|---:|---:|
| React2024 | Q30 | 0.126680 | 0.808698 | 0.0468 |
| React2024 | Q50 | 0.137949 | 0.865779 | 0.0537 |
| React2024 | Q70 | 0.148144 | 0.928920 | 0.0756 |
| React2024 | Q90 | 0.158229 | 1.003323 | 0.1112 |
| React2025 | Q30 | 0.186202 | 0.720005 | 0.0004 |
| React2025 | Q50 | 0.208006 | 0.811044 | 0.0034 |
| React2025 | Q70 | 0.226931 | 0.899685 | 0.0133 |
| React2025 | Q90 | 0.250511 | 1.026725 | 0.0432 |

The gate strongly removes the compatibility tail, especially on R25. However,
none of these simple gates reaches `1.3 × T0 target FRDiv`; the next teacher
experiment should therefore not be started until a better conditional gate or
an explicit quality constraint is designed.

## Scientific conclusion

The data support the central hypothesis:

> Session diversity and conditional diversity are different quantities.

Pure Max-FRDiv is much more aggressive on React2025. It selects GTs that are
farther from the paired anchor after session normalization and much more often
fall in the session-distance tail. This explains why R25 gets a large FRDiv
gain together with a quality loss. React2024's selected set is also more
diverse, but its conditionality penalty is substantially smaller, allowing the
teacher FRC to improve rather than collapse.

This is a data/target-geometry explanation, not evidence that the model
capacity differs. The correct next route is an anchor-compatible Max-FRDiv
teacher, but only after defining a gate that preserves meaningful diversity.

Per-context Spearman correlations with teacher ΔFRC/ΔFRD are not claimed in
this report: the frozen R24/R25 evaluation JSONs contain only aggregate
metrics and no per-context teacher rows. The current report does contain all
per-source target/anchor rows needed to attach that evaluation without
retraining.

## Provenance

| Field | React2024 | React2025 |
|---|---|---|
| TRAIN contexts | 6,340 | 3,320 |
| data fingerprint | `31d153d80303e6b2a4bdfbffff061d7aa88b20467574b5314c65655abd71e7c8` | `b18fb0e894fe4607da2f209981631445add001be4795bf1c27e496ec5710aae9` |
| T0 manifest SHA256 | `0bad9ba285ad8824436497f493e83871a535f0055c973613873dd245d4f8f880` | `c2bb033ddf6edc3ed0f4be1e5c3b1f830b92cc560582e599a625d7d3671d88aa` |
| T1 manifest SHA256 | `c06b3fe5c435582a654fc3df92430c0a83cf9321563b4763c1ee92e6bd3b0968` | `a68ec7bcf8f93ebbb331c351fd02fa0c40ed02ea127a615ca29a836f89b10b52` |
| frozen anchor checkpoint SHA256 | `cc596668b02efc1b2a304fc8a32b1150eabb35eecbc27beeb4e7d8afac62a7bc` | `9d0190a7cb48703e001dc3f092d84ac7de17bea5e2412604f3871099d1c3ebb6` |
| diagnostic output | `/tmp/cross_dataset_full.json` | `/tmp/cross_dataset_full.json` |

