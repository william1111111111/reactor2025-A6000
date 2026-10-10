> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Teacher upgrade: evidence and proposed next experiments

Status: planning/read-only audit on 2026-09-24. No new teacher training or
oracle evaluation was launched by this audit. Student weights/recipes are
frozen in [the recipe bank](STUDENT_RECIPE_BANK.md).

Execution update (2026-09-25): priority 1 is now complete; see
[the six-point full-VAL571 convergence report](TEACHER_CONVERGENCE.md).
T0 improves with additional updates, but no longer-trained T1 point preserves
all three baseline metrics. T1_1200 raises FRC and FRDiv while worsening exact
FRD by 0.836899; T1_2400 also loses FRC. Preserve T1_600 as the strict reference,
T1_1200 as an explicit tradeoff point, and T0_2400 as a stronger quality family.
These are single-seed VAL results, not independent confirmation. A next
diagnostic can pool T0_2400 quality candidates with T1_600/T1_1200 specialists;
it has not been run, and no new student distillation has started.

## What the current evidence does and does not establish

| Teacher | Updates/model | FRC ↑ | exact FRD ↓ | FRDiv ↑ |
|---|---:|---:|---:|---:|
| T0 random fixed paired+9 | 600 | 1.373332 | 84.929128 | 0.036415 |
| T1 pure Max-FRDiv fixed paired+9 | 600 | 1.072215 | 96.813896 | 0.205415 |

Both use ten independent 3,720,129-parameter models. The nominal schedule is
50 epochs, but the actual frozen checkpoints are a 600-update prefix (103
batches/epoch, stopping partway through epoch 6). They are not 50-epoch models.
Some TRAIN CCC curves are still improving; this alone does not establish VAL
underfitting. An older behavioral-target experiment showed worse VAL FRC at
50 than at 20 epochs, under a different target/evaluation protocol. It is a
warning against assuming monotonic gains, not a numeric baseline for T1.

The quality-gated 600-step experiment already tried DESC_Q50/Q70/Q80. Q50
gave FRC=1.337285, FRD=88.289176, FRDiv=0.107139, still worse quality than
its matched T0. Thus another fine descriptor-quantile sweep is not the main
next experiment. The cross-dataset audit also found larger source-anchor
deviation for R25 T1: target diversity is not necessarily source-compatible
diversity. Anchor proximity remains a proxy, not a psychological quality label.

## Priority 1: teacher convergence, not a new architecture

Proposed matched grid: T0/T1 at 600, 1,200 and 2,400 optimizer updates, seed 1.
Keep each fixed GT manifest, rank identity, initialization, source/crop order,
batch32, BF16, AdamW, 50-epoch cosine schedule, alignment and loss identical.
Retain all ten ranks and all inspected checkpoints; report extra compute.
Do not change GT selection, architecture and optimization simultaneously.

Use the established vectorized full-VAL571 all-session FRC/FRDiv evaluator
for learning curves, then exact rolling FRD for nominated checkpoints and
their matched controls. All selection uses VAL only; TEST stays untouched.
Publish the entire inspected curve and do not describe VAL-selected peaks as
independent confirmation. Additional seeds come after this one-seed diagnosis.

Success is not merely higher FRC. Report both frozen T1 and matched T0 as
references, require exact FRD, and distinguish strict T1 Pareto improvements
from explicit quality/diversity tradeoff points. A practical first question is
whether T1 can retain about 0.20 diversity while improving its quality; this
is a question, not a predicted result or a relaxed strict-Pareto claim.

### Resume pitfall found during the audit

The current trainer saves epoch/model/optimizer/scheduler but not within-epoch
position or RNG state. Resume reconstructs global_step as epoch × batch count.
For the epoch-6/step-600 checkpoint it would reconstruct 618 and skip the last
18 batches of that epoch; it is not an exact continuation. Before this study,
either implement and test exact-step/RNG/sampler resume, or rerun matched
prefixes from the same initialization. Do not silently call the existing
epoch-only resume a matched 600→1200 continuation. No fix was applied here.

## Priority 2: combine quality teachers and diversity specialists

If longer training plateaus or damages diversity, diagnose the already-trained
T0+T1 pool (20 candidates/context, canonical slot0 anchor retained). A Q50
family can be added only after locating/verifying its actual checkpoint files;
this audit verified T0/T1 locally, not Q50 weights.

Run a separate GT-oracle Select-10 diagnostic under strict frozen T1 FRC/FRD
constraints, keeping the complete T1 set as a feasible baseline. Also report
the stronger T0-reference frontier separately; it may have no feasible set.
Use deterministic selection with swap refinement and call the result an
achieved feasible bound, not a proven combinatorial optimum.

This tests whether an ensemble can supply a quality core plus diverse
specialists without retraining. Do not average trajectories: arithmetic
averaging can erase the very modes to be distilled. The historical M3 pool
used CoReSet TCN checkpoints, not this T0/T1 teacher pool, so its numbers do
not answer this new question.

VAL-GT oracle selection is diagnostic only, never an inference method or a
source of student training targets. If worthwhile, construct teacher labels
using TRAIN GT only; learn an inference-safe selector if a deployable teacher
is needed. Canonicalize roles with TRAIN-defined prototypes before KD to
avoid making a student's slot imitate unrelated roles across contexts.

## Priority 3: source-compatible specialist targets

Only if the first two diagnoses support it: replace pure distance extremeness
with source-conditioned, supported behavioral targets. Preserve paired rank0
and stable specialist identities. Investigate a small mode-specific set of
compatible GTs rather than forcing each specialist to fit an arbitrary extreme
trajectory for every context. Keep the expensive temporal budget matched.
Use TRAIN-only cross-fitted compatibility estimates; no VAL/TEST labels.

This is not the already-tested global descriptor-nearest quantile gate. In
particular, a single paired reaction is not the sole definition of a valid
response; penalizing all departures from it can erase legitimate modes.
Do not change cross-listener velocity supervision casually: the present
relative-time policy masks it for non-synchronous pairs.

## Teacher → student confirmation

After a teacher actually improves, regenerate its frozen TRAIN cache and reuse
S-basic/S-residual, followed by matched S-continuation/S-logit if warranted.
Keep student budget/initialization fixed across teacher variants. Better
teacher FRDiv does not guarantee a constant student retention ratio: verify
quality, retained diversity, slot identity and model size together.

References: [T0/T1 pilot](MAXFRDIV_TEACHER_PILOT.md),
[quality-gated teachers](QUALITY_GATED_TEACHER_FULL.md),
[cross-dataset compatibility](CROSS_DATASET_MAXDIV_COMPATIBILITY.md),
[older convergence warning](M3_BEHAVIORAL_PAIRED_ANCHORS.md).
