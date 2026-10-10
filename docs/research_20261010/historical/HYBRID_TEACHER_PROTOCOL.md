> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# T1H: raw fixed-GT supervision plus weak processed-target auxiliary loss

Frozen before new training/evaluation, 2026-09-25. REACT2025, seed1.

Keep the T1 Pure Max-FRDiv TRAIN target identities. Reuse the exact T1_600
native paired q0; train nine specialists from the original initialization.
The model, source/crop schedule, batch32, BF16, AdamW1e-4 and 50-epoch cosine
prefix remain unchanged. Each specialist receives exactly 600 updates.

Use one prespecified weight, 0.10:

`L = L_original_raw + 0.10 * (MSE_processed + 1 - CCC_processed)`.

Do not dilute or replace the original loss. Retain its original velocity mask.
No auxiliary velocity loss, target interpolation between representations,
new GT draws, model parameters, output scaling, or student training.
Reuse the previous TRAIN-only processed cache (fixed per-GT/length seeds,
not identical to the evaluator's per-context seeds). q0 is unchanged.

Training uses one backbone forward and two inexpensive pointwise/CCC loss
evaluations per source. This adds loss arithmetic but no additional temporal
GT identities, SoftDTW or model forward. Record throughput honestly.

Before launch: test zero-weight loss/gradient equivalence, exact additive-loss
gradients, padding masks, and unchanged real-data speaker/source/raw-target
views. Audit checkpoint initial-state and actual source/crop hashes after
each model completes. Fail on any mismatch.

Run full VAL571 with the unchanged all-session opposite-role GT cache,
official AU rounding, existing learned target processor, and exact rolling
FRD. This is the established all-session extension, not official random-10
GT sampling. Also score the same predictions against raw training-aligned GT.

The strict guard relative to T1_600 requires all four:

- processed all-session FRC >= baseline;
- exact FRD <= baseline;
- FRDiv >= baseline;
- raw-target all-session diagnostic FRC >= baseline.

No slack or weight search after inspecting results. Report all metrics,
session-bootstrap paired differences, source hashes, parameter count and
compute; retain negative results and old weights. A failed pilot is not
promoted. This is a single-seed, repeatedly observed VAL experiment, not an
independent TEST confirmation.
