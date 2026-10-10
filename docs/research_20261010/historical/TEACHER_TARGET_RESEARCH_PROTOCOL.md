> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Fixed-teacher target diagnosis and two controlled 600-step pilots

Protocol frozen before either new VAL result is available, 2026-09-25.

## Diagnosis

Evaluate every TRAIN speaker input (1660 contexts), center750, using existing
T0/T1 checkpoints at600/1200/2400 steps. Report fixed-GT soft and rounded FRC,
all-session rounded FRC with training-relative alignment, and fixed/all-session
FRC with the official deterministic target processor. All fixed GTs are present
in the TRAIN candidate pool. The reverse-role TRAIN inputs are excluded here
to match the speaker-input direction of VAL; both roles remain in training.

Use exact training GT trajectories as predictions for an identity check.
Compare same-GT raw identity scores against the same GT after official
processing, and also allow all-session matching. Constant channels have zero
CCC, so the identity score is not assumed to be10.

Three disjoint shards collectively cover all1660 inputs. An interrupted
preliminary single-worker pass has no aggregate result and is not analyzed.

## T2R: supported modes with reuse penalties

Retain the original TRAIN-only nine prototypes and q0 paired anchor. For each
source-role/session, solve a rectangular assignment of nine distinct other
GTs to the nine slots. Cost = normalized prototype distance +0.5 normalized
five-neighbor density distance +0.5 per-slot prior-use count / uniform expected
use. Source processing order is a stable seed1 hash. The resulting full3320
fixed mappings are immutable during training.

This is a combined target-selection policy, not a component-level ablation.
Density and reuse are not certified speaker-response compatibility labels.
T2R is distinct from the historical unconstrained-GT oracle named T2.

Primary comparison: matched T0_600. Strict guard requires FRC >= T0,
exact FRD <= T0, FRDiv > T0. Show T1_600 as an additional frontier reference.

## T1P: isolate target representation

Retain all original T1 fixed GT identities, roles and training views. For
q1--q9 only, replace relative-time interpolation with the existing official
VAE target processor, cached using TRAIN GT identities and source output
lengths. q0 retains native paired supervision. Cache seeds are stable per
GT identity/output length, so this is not identical to the evaluator's
context-dependent random seed. No VAL or TEST target enters training.

Primary comparison: matched T1_600. Strict guard requires FRC >= T1,
exact FRD <= T1, FRDiv >= T1. Improvement in quality with a diversity loss
is a tradeoff, not a strict win.

## Shared controls

ConditionalREGNN graph2 linear head; seed1; batch32; BF16; AdamW1e-4;
original50-epoch cosine schedule;600 actual updates; unchanged MSE +1-CCC
and masked velocity loss. Each arm reuses its verified native q0 checkpoint
and trains nine specialists, costing5400 new optimizer updates. Full ten-model
budget including the reused anchor remains6000 updates, equal to its baseline.
Check initial-state and actual source/crop-order hashes against the matched
600-step reference. No new student training.

Evaluate all new ensembles on all571 VAL contexts with the existing evaluator,
all-session opposite-role GT cache, official AU rounding and exact rolling
FRD. Report all arms, session-bootstrap differences, and seen/unseen-session
strata. Those strata also differ in GT-pool size, so their raw metric gaps
cannot be attributed exclusively to session generalization.

All ten outputs remain; no output averaging, subset selector or post-hoc scaling.
Record parameter count, actual update budget, training throughput and inference
latency. GPUs1/5/6 host concurrent diagnostics, so timing is diagnostic rather
than an isolated performance benchmark. No VAL-based policy-weight sweep.
