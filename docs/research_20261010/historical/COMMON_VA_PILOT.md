> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Frozen common-VA correction: quality versus invariant diversity

REACT2025; 64 VAL contexts; all-session opposite-role GT; same processed-GT cache; official AU rounding and exact rolling FRD. Not official random-10 GT sampling.

| Method | FRC ↑ | exact FRD ↓ | FRDiv | FRVar | Raw-GT FRC ↑ |
|---|---:|---:|---:|---:|---:|
| baseline | 1.119153 | 96.648585 | 0.203003 | 0.069049 | 1.058569 |
| quality_centroid | 1.129156 | 98.451932 | 0.203003 | 0.069574 | 1.071736 |
| oracle_025 | 1.171804 | 96.707066 | 0.203003 | 0.069337 | 1.074393 |
| oracle_050 | 1.217258 | 99.203876 | 0.203003 | 0.070050 | 1.084689 |
| oracle_100 | 1.254873 | 110.843926 | 0.203003 | 0.072753 | 1.086792 |
| FRC_only_oracle | 1.254873 | 110.843926 | 0.203003 | 0.072753 | 1.086792 |
| strict_oracle | 1.153138 | 95.803008 | 0.203003 | 0.069293 | 1.068757 |

Only `quality_centroid` is GT-free, with fixed coefficient1. Every oracle arm uses VAL GT and is diagnostic only. Selection is over complete ten-prediction sets, never a mixture of independently shifted slots.

## Strict safety and distribution

Strict selection requires per-context FRC >= baseline and exact FRD <= baseline, with no quality slack. Baseline fallback is always eligible. It maximizes FRC only within the finite proposed family, not over all possible corrections.

Strict selected sources: `{"baseline": 38, "oracle_025": 13, "oracle_050": 8, "oracle_100": 1, "quality_centroid": 4}`.

| Method / metric | Mean delta | P10 / median / P90 | Session bootstrap95% CI |
|---|---:|---|---|
| quality_centroid / FRC | +0.010003 | -0.020572 / +0.009717 / +0.037427 | [+0.000905, +0.018951] |
| quality_centroid / exact_FRD | +1.803346 | -2.150025 / +2.324666 / +5.717086 | [+0.066920, +2.605761] |
| quality_centroid / raw_FRC | +0.013167 | -0.015331 / +0.010795 / +0.040868 | [+0.000190, +0.025300] |
| FRC_only_oracle / FRC | +0.135720 | +0.084019 / +0.117548 / +0.211775 | [+0.118161, +0.182863] |
| FRC_only_oracle / exact_FRD | +14.195341 | +6.014714 / +14.024821 / +20.173898 | [+11.508146, +15.336944] |
| FRC_only_oracle / raw_FRC | +0.028224 | -0.010055 / +0.025459 / +0.066354 | [+0.017416, +0.042737] |
| strict_oracle / FRC | +0.033984 | +0.000000 / +0.000000 / +0.130169 | [+0.017389, +0.076990] |
| strict_oracle / exact_FRD | -0.845577 | -1.558423 / +0.000000 / +0.000000 | [-2.228710, -0.289912] |
| strict_oracle / raw_FRC | +0.010188 | +0.000000 / +0.000000 / +0.033505 | [+0.005473, +0.021800] |

Bootstrap describes variation across evaluated sessions, not generalization of an Oracle to unseen GT. Full per-session metrics and context ledgers are retained locally.

## Invariance, scope and cost

Maximum centered-geometry change: 4.47e-08; each context's native FRDiv change is asserted below1e-6. AU/expression are bitwise unchanged. Common VA shifts stay in[-1,1] without independent clipping.

q0 VA output is also corrected; its model weights remain frozen. An unchanged-q0-output guarantee is not claimed. Raw-GT FRC is diagnostic and is not a constraint used to select these Oracle sets.

Baseline uses ten3,720,129-parameter teachers. The GT-free correction uses an additional ten-model T0_2400 ensemble:74,402,580 parameters total, final output10. No weights are trained. The oracle optimizes at most32 per-context control parameters for100 Adam steps; these are GT-dependent, not deployable model weights.

Cached-prediction correction overhead: 0.463ms/context; oracle proposal optimization: 0.166s/context; exact scoring of five sets: 1.847s/context.

These timings exclude baseline/quality model forward passes and are not end-to-end inference latency. Exact DTW reuses invariant AU/expression costs and uses the same nonnegative-group pruning and rolling recurrence as Mam. Parity is tested and the baseline FRD is reproduced for every context.

Base commit: `c421f509f8cb7997e9341592cae2eb1e721b0e8e` plus frozen source hashes. Summary implementation SHA256: `edceccda4a27a2872ac5974ffcc472a417a6ae2b6081b81049912cfe5b116265`.

No checkpoint, TRAIN target or student cache is replaced. This is a local diagnostic, not a new official TEST result.

See [the prespecified protocol](COMMON_VA_PROTOCOL.md).
