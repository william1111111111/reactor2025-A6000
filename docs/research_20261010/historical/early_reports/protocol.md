> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# REACT2025 fixed-teacher evaluation protocol

Protocol lock date: 2026-09-20

## Scope

The current workstream is REACT2025, offline, 25D facial-attribute generation. Predictions contain 15 AU channels, 2 VA channels, and 8 expression channels. A ten-teacher ensemble contributes one complete trajectory per teacher; trajectories and parameters are not averaged.

Two evaluation modes are kept separate:

1. **Public-code-compatible sampled-10 diagnostic**: paired same-basename GT plus nine deterministic same-session opposite-role GTs, followed by the public post-processor. This mirrors the local public loader's ten-target construction closely, but uses VAL center-750 and is not an official challenge-server TEST score.
2. **All-session-GT diagnostic (requested primary local analysis)**: every same-session opposite-role GT is available to each prediction. Current full VAL scope is 571 speaker inputs, center-750. GT count ranges from 13 to 113 (mean about 71.6). This changes the candidate reference set and therefore must never be mixed numerically with sampled-10 results.

Generation receives speaker inputs only. Replacing or reordering evaluation GT must not change generated predictions.

## Target processing

- Predictions are 25D and AUs 0--14 are rounded before FRC/FRD, following the local evaluation path.
- Variable-length GT is transformed to prediction length by the frozen Mam-Reactor `Processor` and emotion-VAE checkpoint.
- The stochastic post-processor is made deterministic with a SHA-256-derived seed from `(eval_seed, clip_id, target-postprocessor)`.
- All compared methods use the same source list, GT path ordering, `eval_seed=1234`, post-processor checkpoint and center-750 input contract.
- Post-processed GT may be cached across checkpoints/epochs because processing depends on GT, output shape, seed and clip identity, not prediction values. Cache keys must include all of those fields plus checkpoint/protocol fingerprint.

## Metric code contract

| Local code name | Implementation and aggregation | Direction | Reporting name |
|---|---|---:|---|
| `FRC` | For each prediction, mean CCC over 25 channels against every available GT; take the best GT; sum over 10 predictions; then average contexts | higher | FRC |
| `FRD` / `exact_FRD` | For each prediction, minimum exact DTW over available GTs; AU/VA/expression groups weighted `1/15`, `1`, `1/8`; sum over 10; average contexts | lower | exact FRD |
| `smse` | Mean ordered pairwise squared Euclidean distance among predictions, normalized by flattened dimension | higher diversity | FRDiv (with explicit note that code function is `compute_s_mse`) |
| `FRVar` | Mean temporal variance over context, slot and channel | diagnostic; larger is not automatically better | FRVar |
| `TLCC` | Absolute peak-lag offset against speaker; current function returns after slot 0 | lower | TLCC (slot-0 implementation) |

`FRRea` is not evaluated without the video/rendering assets. Missing video metrics are written as `not_evaluated`, never zero.

## Fast FRC implementation lock

Future all-session FRC evaluations use `coreset_reactor/fast_frc.py` and `coreset_reactor/anchor_frc_curve.py`:

- vectorize the complete `[prediction, GT, channel]` CCC tensor;
- retain the local Mam-Reactor convention that a constant-channel correlation becomes zero;
- do not reduce the VAL contexts or GT set;
- cache deterministic post-processed targets across checkpoint epochs;
- compare the first real context of every evaluated checkpoint against `framework.metrics.FRC._func` with `rtol=1e-5`, `atol=1e-6`;
- record elapsed time and fail on non-finite values.

Synthetic benchmark: scalar kernel 4.58 s, vectorized kernel 0.154 s for K=10, N=113, T=750, D=25 (29.7x kernel speedup; maximum absolute error `1.04e-17`). This is not claimed as whole-evaluator speedup.

## Comparison lock

For matched comparisons, keep fixed:

- architecture and initialization where applicable;
- speaker record schedule and crop/view seed;
- assignment manifest and manifest hash;
- optimizer, scheduler, precision, batch size, epochs/optimizer steps;
- VAL speaker list, center-750 crop, GT list/order, post-processor seed;
- AU rounding and exact metric implementation.

Each report must include parameter count, teacher/student count, temporal-loss budget, training model-steps, inference hardware/precision, warm-up/synchronization policy, latency, peak memory, commit SHA, source hashes, checkpoint hashes, manifest hash, and whether the result is sampled-10 or all-session-GT.

## Current baseline labels

- `B0`: existing single-model mechanism producing ten reactions (or ten identical copies if deterministic; disclose which).
- `T10-random-legacy`: ten historical deterministic-random fixed-pair teachers with diffusion-compatible target fill.
- `T10-prototype-legacy`: paired plus nine prototype-selected teachers, same legacy target handling.
- `T10-farthest-legacy`: paired plus nine descriptor-farthest teachers, same legacy target handling; fair result requires 50 epochs.
- Future `S-direct`, `S-KD`, and `S-KD-rel` must use the corrected explicit alignment/mask contract and must not be compared as though their labels were processed identically to the legacy teachers without an additional matched control.

## Selection discipline

Checkpoint/epoch selection uses VAL only. TEST is not used for pairing, fitting descriptors, choosing losses, selecting epochs, or tuning student weights. All-session-GT is an analysis protocol requested for this local research and should be reported alongside, not substituted silently for, sampled-10/public-code-compatible evaluation.
