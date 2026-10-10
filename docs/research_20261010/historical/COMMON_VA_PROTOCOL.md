> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Frozen common-VA quality-correction diagnostic

Protocol fixed before results, 2026-09-25. No teacher/student training.

Use frozen T1_600 ten-teacher outputs as the only baseline. AU and expression
remain bitwise unchanged. Add one shared, time-dependent VA offset to all ten
outputs, including q0. q0 weights remain frozen but its VA output is modified;
this is explicitly not an unchanged-q0-output comparison.

Project the common offset into `[-1 - min_k y_k, 1 - max_k y_k]` for each time
and VA coordinate. Never clip individual predictions independently. This
preserves all pairwise differences and therefore mathematical FRDiv. Verify
actual float32 invariance to 1e-6; quality constraints have no slack.

Five finite candidate sets per context, always exactly ten reactions:

1. Baseline, zero correction.
2. GT-free correction toward the T0_2400 ensemble's VA mean, coefficient1.
3--5. GT-oracle FRC correction at scales0.25/0.50/1.00. Use at most16 linear
control points per VA channel, zero initialization, Adam lr0.03, 100 steps,
all-session processed GT, and keep the highest-FRC iterate. No teacher weights
are optimized. This is a local finite-family oracle, not a proven upper bound
or global optimum. All oracle choices depend on VAL GT and are not deployable.

Compute exact FRD for all five sets with the existing Mam rolling recurrence.
Share invariant AU/expression DTW terms across variants; test parity first.
FRC uses the established vectorized native-parity metric. GT caches, input
ordering, source lengths, AU rounding and all-session matching remain fixed.

Report the GT-free fixed correction independently. The strict oracle chooses
the highest FRC set subject to per-context FRC>=baseline and FRD<=baseline;
zero correction is always available. Also report the FRC-only oracle and all
three scales, including failures. No threshold/scale changes after inspection.

First evaluate a deterministic64-context pilot (seed1234, selected from571),
then complete full VAL571 with the exact same proposal family. Reuse pilot
rows rather than double-counting them. Full VAL is not independent confirmation
because proposals are optimized using each context's GT. Retain every delta
and baseline fallback, distribution, per-session changes, raw-GT diagnostic,
FRVar and computation/provenance. No TEST or student targets are generated.

The GT-free arm needs two ten-model ensembles:74,402,580 parameters in total,
unless the correction is later distilled; do not call its inference free.
This run reuses cached predictions, so correction timing excludes those model
forwards. The oracle adds per-context optimization and five exact-FRD scores.
