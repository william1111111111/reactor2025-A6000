> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Frozen common-VA correction: quality versus invariant diversity

REACT2025; 571 VAL contexts; all-session opposite-role GT; same processed-GT cache; official AU rounding and exact rolling FRD. Not official random-10 GT sampling.

| Method | FRC ↑ | exact FRD ↓ | FRDiv | FRVar | Raw-GT FRC ↑ |
|---|---:|---:|---:|---:|---:|
| baseline | 1.072308 | 96.808993 | 0.205423 | 0.068098 | 1.033339 |
| quality_centroid | 1.082875 | 98.408748 | 0.205423 | 0.068651 | 1.049255 |
| oracle_025 | 1.123025 | 96.692283 | 0.205423 | 0.068396 | 1.049055 |
| oracle_050 | 1.168232 | 99.135146 | 0.205423 | 0.069108 | 1.059425 |
| oracle_100 | 1.209266 | 110.847792 | 0.205423 | 0.071774 | 1.062231 |
| FRC_only_oracle | 1.209266 | 110.847792 | 0.205423 | 0.071774 | 1.062231 |
| strict_oracle | 1.112117 | 95.911353 | 0.205423 | 0.068460 | 1.045459 |

Only `quality_centroid` is GT-free, with fixed coefficient1. Every oracle arm uses VAL GT and is diagnostic only. Selection is over complete ten-prediction sets, never a mixture of independently shifted slots.

## Decision

The restricted oracle finds a feasible improvement: FRC +0.039809 (+3.71%),
exact FRD -0.897640 (-0.93%), with unchanged diversity to numerical precision.
It applies a correction in261/571 contexts (45.7%) and retains the original
set in310/571 (54.3%). Processed FRC and exact FRD are non-worsening in every
context, not merely on average. This establishes an achievable correction in
this finite family, not the global maximum or an inference-safe improvement.

The GT-free mean transfer fails the quality criterion: FRD increases by1.599754.
Unconstrained FRC optimization also fails badly on FRD (110.847792). The useful
direction is a correction with quality control, not always transferring a
quality-model mean or always maximizing FRC.

Raw-GT FRC improves in aggregate from1.033339 to1.045459, but decreases on
31/571 contexts. Raw-GT FRC was not a selection constraint; the per-context
safety statement applies only to the prespecified processed FRC and exact FRD.

A possible follow-up is to generate correction/safety labels using TRAIN GT
only, then learn a small source-conditioned correction/gate. Its inference
inputs must exclude GT, GT matches and oracle metric scores. Structural
FRDiv invariance can remain enforced, but learned FRC/FRD safety cannot be
assumed and needs separate validation. No such model was trained here.

## Strict safety and distribution

Strict selection requires per-context FRC >= baseline and exact FRD <= baseline, with no quality slack. Baseline fallback is always eligible. It maximizes FRC only within the finite proposed family, not over all possible corrections.

Strict selected sources: `{"baseline": 310, "oracle_025": 127, "oracle_050": 83, "oracle_100": 22, "quality_centroid": 29}`.

| Method / metric | Mean delta | P10 / median / P90 | Session bootstrap95% CI |
|---|---:|---|---|
| quality_centroid / FRC | +0.010567 | -0.019517 / +0.009176 / +0.043746 | [+0.003075, +0.018240] |
| quality_centroid / exact_FRD | +1.599754 | -2.826695 / +2.133114 / +5.966672 | [-0.367033, +2.506401] |
| quality_centroid / raw_FRC | +0.015916 | -0.018184 / +0.014282 / +0.051345 | [+0.003583, +0.029560] |
| FRC_only_oracle / FRC | +0.136958 | +0.080854 / +0.125213 / +0.212287 | [+0.121424, +0.174687] |
| FRC_only_oracle / exact_FRD | +14.038799 | +5.658216 / +13.535096 / +22.697668 | [+12.406550, +15.343560] |
| FRC_only_oracle / raw_FRC | +0.028891 | -0.016299 / +0.025554 / +0.076558 | [+0.018765, +0.039522] |
| strict_oracle / FRC | +0.039809 | +0.000000 / +0.000000 / +0.135558 | [+0.024423, +0.074459] |
| strict_oracle / exact_FRD | -0.897640 | -2.570350 / +0.000000 / +0.000000 | [-1.828992, -0.484137] |
| strict_oracle / raw_FRC | +0.012119 | +0.000000 / +0.000000 / +0.045937 | [+0.007144, +0.022309] |

Bootstrap describes variation across evaluated sessions, not generalization of an Oracle to unseen GT. Full per-session metrics and context ledgers are retained locally.

## Invariance, scope and cost

Maximum centered-geometry change: 4.58e-08; each context's native FRDiv change is asserted below1e-6. AU/expression are bitwise unchanged. Common VA shifts stay in[-1,1] without independent clipping.

q0 VA output is also corrected; its model weights remain frozen. An unchanged-q0-output guarantee is not claimed. Raw-GT FRC is diagnostic and is not a constraint used to select these Oracle sets.

Baseline uses ten3,720,129-parameter teachers. The GT-free correction uses an additional ten-model T0_2400 ensemble:74,402,580 parameters total, final output10. No weights are trained. The oracle optimizes at most32 per-context control parameters for100 Adam steps; these are GT-dependent, not deployable model weights.

Cached-prediction correction overhead: 0.460ms/context; oracle proposal optimization: 0.166s/context; exact scoring of five sets: 1.514s/context.

These timings exclude baseline/quality model forward passes and are not end-to-end inference latency. Exact DTW reuses invariant AU/expression costs and uses the same nonnegative-group pruning and rolling recurrence as Mam. Parity is tested and the baseline FRD is reproduced for every context.

Four unit tests and a two-context real-data smoke test passed. All571 GT cache
fingerprints were recomputed from the original protocol, target paths, sizes
and modification times and match the frozen evaluation. Means of native
per-context FRDiv scores differ from the prior float32 aggregate by less than
0.000001; no scoring definition was changed.

Base commit: `c421f509f8cb7997e9341592cae2eb1e721b0e8e` plus frozen source hashes. Summary implementation SHA256: `edceccda4a27a2872ac5974ffcc472a417a6ae2b6081b81049912cfe5b116265`.

No checkpoint, TRAIN target or student cache is replaced. This is a local diagnostic, not a new official TEST result.

See [the prespecified protocol](COMMON_VA_PROTOCOL.md).
