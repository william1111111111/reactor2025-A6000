> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Learned common-VA correction: frozen protocol

Frozen before TRAIN-label generation or model fitting, 2026-09-25.

## Question

Can a small inference-safe model predict part of the GT-oracle common VA
correction while preserving T1 teacher diversity by construction?

## Data separation

- Generate labels only from the 1660 REACT2025 TRAIN speaker contexts.
- For each TRAIN source, use all opposite-role listener GTs from its session,
  with the existing deterministic learned target processor and eval seed1234.
- Freeze T1_600 predictions. Generate the same finite proposal family as the
  completed VAL oracle: zero, GT-free T0_2400 centroid, and optimized common VA
  correction at scales0.25/0.50/1.00.
- Select the highest-FRC proposal satisfying processed FRC>=baseline and exact
  FRD<=baseline per TRAIN context. Baseline fallback is always available.
- No VAL/TEST GT, VAL oracle shifts, VAL oracle choices or VAL metric deltas may
  enter labels, fitting, thresholds, normalization or early stopping.

The TRAIN labels remain a finite local oracle, not psychological ground truth.
TRAIN and VAL identities/sessions are not assumed interchangeable.

## Inference inputs and model

Inputs contain only quantities available without listener GT:

- speaker audio768, facial attributes25 and normalized3DMM58;
- frozen T1 prediction mean25 and standard deviation25 at each frame;
- valid source length.

Use separate linear projections to32/16/16/16/16 channels, concatenate, then
three residual temporal Conv1d blocks (hidden64, kernel5, dilations1/2/4,
GroupNorm, GELU, dropout0.1). A frame head predicts two common-VA offsets with
`0.35*tanh`; a pooled head predicts one context gate logit. Approximately
0.2M parameters; exact count must be reported.

The predicted offset is projected once into the common feasible interval for
all ten T1 outputs. Never clip slots independently. AU/expression are bitwise
unchanged and all ten predictions receive the same VA offset, including q0.
Thus pairwise geometry/FRDiv is structurally preserved up to float precision.
q0 weights are unchanged, but q0 VA output may change.

## Training

- Seed20260925; all1660 TRAIN contexts; center750; fixed20 epochs; batch8.
- AdamW lr3e-4, betas(0.9,0.99), weight decay1e-4; cosine schedule; BF16;
  gradient clip1. No early stopping or VAL inspection.
- Shift loss: valid-prefix SmoothL1(beta0.02) to the selected TRAIN correction.
- Velocity loss:0.1 times adjacent-frame SmoothL1 on the correction.
- Gate loss:0.2 times BCE, target1 iff TRAIN strict oracle selects nonbaseline.
- Zero padded output before loss/application. Record actual source order and
  initialization hashes. No teacher/student checkpoint is overwritten.

## Fixed inference policies

Report four policies on full VAL571:

1. baseline (zero correction);
2. ungated predicted correction;
3. gate probability>=0.50;
4. **primary** gate probability>=0.80;
5. conservative diagnostic gate probability>=0.90.

Threshold0.80 is fixed here, not selected on VAL. Report all policies even if
another threshold looks better. No quality model, GT, DTW/FRC score, T0 model,
oracle proposal or post-hoc coefficient is available at inference.

## Evaluation and stopping

Use the unchanged full VAL571 all-session processed-GT cache, official AU
rounding, exact rolling FRD and raw-GT diagnostic. This is the established
all-session extension, not official sampled-10 scoring. Compare against the
exact frozen T1_600 rows and cache keys.

Primary success requires mean FRC>=T1, exact FRD<=T1, FRDiv equal within1e-6,
and raw-GT FRC>=T1. Also report per-context quality pass rate, gate rate,
session-bootstrap intervals, per-session results, gate precision/recall against
the post-hoc VAL oracle only as a diagnostic, parameter count, training cost
and inference overhead. The learned method has no per-context safety guarantee.

One seed/full VAL is exploratory evidence, not independent TEST confirmation.
If the primary policy fails, retain it as a negative/frontier result and do not
tune thresholds or loss weights in this run.
