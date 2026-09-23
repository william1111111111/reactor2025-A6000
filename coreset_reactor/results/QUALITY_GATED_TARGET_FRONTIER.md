# Quality-gated Max-FRDiv TRAIN target frontier

This is a TRAIN-only target-selection audit. The frozen T0/T1 experiments are
unchanged. No VAL GT is used to set thresholds or select targets.

The audit uses 1,660 speaker-source contexts from 17 TRAIN sessions. For each
context, the native paired target is kept fixed. The remaining same-session
listener GTs are filtered by either TRAIN-scaled 602-D descriptor RMS distance
or relative-time trajectory MSE to the paired GT. The existing paired
farthest-sum plus one-swap selector then chooses nine alternatives inside the
legal pool. Descriptor prototypes are used only for deterministic q1--q9
Hungarian slot assignment.

## Frontier

| Gate | Target FRDiv | vs T0 | Mean descriptor distance to paired | Mean trajectory distance to paired | AU range | VA range | Expression modes |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0 random paired+9 | 0.197589 | 1.000x | 1.273933 | 0.198094 | 0.992972 | 1.459490 | 4.202 |
| DESC Q30 | 0.200650 | 1.015x | 1.092100 | 0.185367 | 0.993253 | 1.496699 | 3.580 |
| DESC Q40 | 0.211365 | 1.070x | 1.127782 | 0.194081 | 0.996466 | 1.517474 | 3.869 |
| DESC Q50 | **0.220407** | **1.115x** | 1.163170 | 0.202039 | 0.995984 | 1.526859 | 4.080 |
| DESC Q60 | 0.228393 | 1.156x | 1.197840 | 0.208918 | 0.997831 | 1.526889 | 4.267 |
| DESC Q70 | **0.236899** | **1.199x** | 1.237790 | 0.216669 | 0.998594 | 1.531368 | 4.471 |
| DESC Q80 | **0.246159** | **1.246x** | 1.285720 | 0.225221 | 0.998112 | 1.545151 | 4.636 |
| TRAJ Q30 | 0.179191 | 0.907x | 1.159365 | 0.163861 | 0.990442 | 1.435472 | 3.745 |
| TRAJ Q40 | 0.192312 | 0.973x | 1.193355 | 0.173074 | 0.994177 | 1.452206 | 4.073 |
| TRAJ Q50 | 0.204411 | 1.035x | 1.227259 | 0.182330 | 0.996225 | 1.475209 | 4.307 |
| TRAJ Q60 | 0.215136 | 1.089x | 1.262104 | 0.191027 | 0.997590 | 1.496828 | 4.483 |
| TRAJ Q70 | 0.225622 | 1.142x | 1.297136 | 0.200156 | 0.998635 | 1.506814 | 4.675 |
| TRAJ Q80 | 0.237161 | 1.200x | 1.343936 | 0.210826 | 0.999036 | 1.518337 | 4.878 |
| T1 pure Max-FRDiv paired+9 | 0.278971 | 1.412x | 1.556085 | 0.253270 | 1.000000 | 1.546922 | 4.896 |

Every gate retained more than nine legal alternatives on every context; no
minimum-count widening or random fallback occurred. Mean eligible pool sizes
for DESC Q50/Q70/Q80 were 48.61/67.44/77.10 alternatives.

## Pilot candidates

The first short matched teacher pilot should use:

```text
T0        random paired+9
G1        DESC_Q50
G2        DESC_Q70
G3        DESC_Q80
T1        pure Max-FRDiv paired+9
```

These points give the smoothest monotone descriptor-gated frontier and cover
target FRDiv approximately 0.22, 0.24, and 0.25 without mixing the two gate
families. The trajectory family remains a separate diagnostic; it is not
silently merged into the descriptor family.

## Provenance

The generated local manifest contains target IDs and is not committed. The
aggregate audit used:

- frozen T0/T1 source manifest SHA256:
  `13d2ad79aa558fcd7fc524088d6d3feb81c6508e8079237d1272a2c4fb670b77`
- current selector source SHA256:
  `db9d0c7e9a344fdd3bd905090c2ea23548f3eb4951a02011807bcbf362a716af`
- generated local audit manifest SHA256:
  `52cf07fab53bf23506879bd94db5fc878d5ffe02ca9b65fd39e41d96de3c0d63`
- alignment: `relative_time_masked`
- membership distance: official trajectory-space mean squared 25-D distance
  after `relative_time_resample_reaction`

The quality-gated generator is implemented in
`coreset_reactor/quality_gated_maxfrdiv.py`; its minimum-candidate guard and
deterministic tie handling are covered by unit tests.

For the matched teacher manifest, the same policy was generated for both
source roles (3,320 contexts total). The pooled target-set FRDiv values used by
the training manifests were:

```text
T0        0.200888
DESC_Q50 0.221019
DESC_Q70 0.235931
DESC_Q80 0.244072
T1        0.273745
```
