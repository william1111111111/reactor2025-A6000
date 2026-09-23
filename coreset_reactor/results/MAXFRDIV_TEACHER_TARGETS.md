# Max-FRDiv teacher target-set audit

This is an aggregate-only TRAIN report. Raw target IDs, manifests, trajectories,
and checkpoints remain local and are not committed.

## Frozen protocol

- split: TRAIN only
- source role: speaker; target role: same-session listener
- contexts: 1,660
- sessions: 17
- target length for membership selection: 750 frames
- cross-listener normalization: `relative_time_resample_reaction`
- membership distance: official 25-D trajectory mean squared distance
- rank 0: the true same-basename paired target for T0 and T1
- q1--q9 identity: TRAIN-only global descriptor prototypes with Hungarian assignment
- T1/T2 membership: deterministic farthest-sum plus one-swap approximation
- no VAL GT was used

The exact MILP was also probed on one 95-GT TRAIN session. With a 30-second
limit it did not reach an optimal status, so the complete manifest does not call
the heuristic a theoretical optimum. The retained heuristic is a valid lower
bound. The probe took 34.56 seconds and returned no exact incumbent from the
solver before the limit; its feasible farthest lower bound was `0.290039`.

## Target-set statistics

| Teacher set | FRDiv | Descriptor pair distance | AU range | VA range | Mean distinct expression modes |
|---|---:|---:|---:|---:|---:|
| T0: random paired + 9 | 0.197589 | 1.272094 | 0.992972 | 1.459490 | 4.202 |
| T1: paired-constrained approximate Max-FRDiv + 9 | **0.278971** | **1.677804** | 1.000000 | **1.546922** | **4.896** |
| T2: unconstrained approximate Max-FRDiv 10 | 0.282918 | 1.699305 | 1.000000 | 1.555712 | 4.823 |

- T1/T0 FRDiv ratio: **1.4119**
- T2/T1 FRDiv ratio: **1.0141**

T1 therefore provides a substantial target-support increase while preserving
the paired anchor. T2 adds only a further 1.4% over T1, which supports keeping
T1 as the main teacher experiment and T2 as a ceiling diagnostic.

## Status

The target-set phase is complete. No teacher or student was trained from these
manifests yet. The next controlled step is the matched 600-step T0-vs-T1
ConditionalREGNN teacher pilot with `fixed_pair_alignment_policy=relative_time_masked`
and the same optimizer, initialization, source schedule, and temporal budget.
