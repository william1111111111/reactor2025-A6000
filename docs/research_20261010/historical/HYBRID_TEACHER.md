> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# T1H: preserve raw GT plus a weak processed-target auxiliary

REACT2025, seed1, 600 updates per teacher, full VAL571. All-session opposite-role GT with official AU rounding and learned target processing; not official random-10 GT sampling.

| Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar | Raw-GT diagnostic FRC ↑ |
|---|---:|---:|---:|---:|---:|
| T1 raw baseline | 1.072308 | 96.808993 | 0.205423 | 0.068098 | 1.033339 |
| T1P processed replacement | 1.144855 | 97.750267 | 0.203711 | 0.068681 | 0.962444 |
| T1H raw + 0.10 processed | 1.083011 | 96.233200 | 0.202933 | 0.066297 | 1.040024 |

Strict four-metric guard: **FAIL**.

Individual guards: FRC=PASS, exact_FRD=PASS, FRDiv=FAIL, raw_FRC=PASS.

| Difference vs T1 | Mean | Session-bootstrap 95% CI | Context improvement fraction |
|---|---:|---|---:|
| FRC | +0.010703 | [+0.004835, +0.016847] | 0.6252 |
| exact_FRD | -0.575794 | [-0.765820, -0.192637] | 0.7391 |
| FRDiv | -0.002490 | [-0.003064, -0.001755] | 0.0946 |

Raw-GT diagnostic FRC change: +0.006685. This diagnostic is supplementary, not an alternative scoring protocol chosen to favor the result.

## Interpretation

Do not promote this pilot: at least one prespecified guard fails. Report the quality/diversity tradeoff rather than claiming a universal improvement. No weight sweep or additional training followed inspection.

Unlike processed-target replacement, the hybrid improves both processed and
raw-target FRC, while also reducing exact FRD. It is retained as a
quality-oriented frontier candidate, not as a replacement for the original
high-diversity T1 bank. FRC improves on 62.5% of contexts and exact FRD on
73.9%; FRDiv improves on only 9.5%. This is not a strict Pareto improvement.

Post-hoc decomposition attributes approximately 80.5% of the net official
FRDiv decrease to AU channels. Dynamic residual diversity falls, while the
static-offset component slightly increases. The decrease also occurs before
AU rounding, so it is not solely a rounding-threshold artifact. This describes
the outputs; it does not prove a causal mechanism for the auxiliary loss.

A focused next ablation would keep the same raw supervision and test whether
excluding AU from the processed auxiliary preserves AU temporal diversity.
That proposal remains untested; do not infer that it retains the quality gain.

## Controls and compute

The same T1 TRAIN GT identities, native paired anchor, architecture, initialization, actual source/crop schedule, batch32, BF16, AdamW1e-4 and original 50-epoch cosine prefix were retained. Initial-state/source hashes passed for all nine new specialists. No student training or old-checkpoint replacement.

`L = L_raw + 0.10 * (MSE_processed + 1 - CCC_processed)`. Raw supervision is not diluted; the original cross-listener velocity mask is retained; no processed velocity term. Targets are packed internally only for loss transport; predictions remain 25-D.

Nine new models ×600 updates =5400 new updates; with the reused q0 the matched ensemble budget is6000 updates. One backbone forward per source, with one additional inexpensive loss evaluation on the same GT identity. No extra SoftDTW, parameters, or inference-time processor for the model. The evaluator target processor is unchanged.

Parameters: 37,201,290 across ten teachers. Candidate/output size10; selector overhead0.
Mean training updates/s/model: 6.406. Training wall time: 297.2s.
Inference latency: 36.99ms/context; GT cache hits: 571/571. These concurrent-device timings are diagnostic, not isolated speed benchmarks.

The previous 5240-entry TRAIN GT/length cache is reused. Its original preprocessing cost is not included in optimizer throughput; cache seeds differ from evaluation-context seeds. No VAL GT enters training.

Base commit: `c421f509f8cb7997e9341592cae2eb1e721b0e8e` plus recorded uncommitted-source hashes. Analysis SHA256: `c30bfd39b67facfa288f5586209dc6ee4ace55caf4464a1afe841eb04642230f`.

Full per-context diagnostics and checkpoints remain local. This is a single-seed VAL-selected study, not a held-out TEST claim.

See [the protocol](HYBRID_TEACHER_PROTOCOL.md).

## Post-hoc diversity attribution

Computed after the primary experiment. Channel contributions use the full 25-channel denominator and sum to FRDiv. Static means between-slot temporal-mean offsets; dynamic means the remaining time-varying slot residual. These are descriptive, not causal ablations.

The decomposition uses float64 accumulation. Its totals differ from the
evaluator's float32 results by less than 0.000001; primary guards use the
unchanged evaluator results above.

| Representation / component | T1 | T1H | Change |
|---|---:|---:|---:|
| official_predictions / AU_total | 0.171013 | 0.169009 | -0.002004 |
| official_predictions / VA_total | 0.011130 | 0.011050 | -0.000080 |
| official_predictions / expression_total | 0.023281 | 0.022875 | -0.000406 |
| official_predictions / FRDiv | 0.205424 | 0.202934 | -0.002490 |
| official_predictions / dynamic_FRDiv | 0.130479 | 0.126718 | -0.003761 |
| soft_predictions / AU_total | 0.075018 | 0.072551 | -0.002467 |
| soft_predictions / VA_total | 0.011130 | 0.011050 | -0.000080 |
| soft_predictions / expression_total | 0.023281 | 0.022875 | -0.000406 |
| soft_predictions / FRDiv | 0.109429 | 0.106475 | -0.002953 |
| soft_predictions / dynamic_FRDiv | 0.055526 | 0.052724 | -0.002802 |

Attribution source SHA256: `ed9b83af980c91cd75b225c6b8bc236bae496ca43165169860f276d85a8b7744`.
