> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Teacher target research: TRAIN fit, supported GTs, and target representation

REACT2025; full VAL571; all-session opposite-role GT; AU rounding; exact rolling FRD. This uses the established all-session extension, not the official random-10 GT sampling contract.

| Arm, 600 updates/model | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar |
|---|---:|---:|---:|---:|
| T0 | 1.373312 | 84.928192 | 0.036411 | 0.036259 |
| T1 | 1.072308 | 96.808993 | 0.205423 | 0.068098 |
| T2R | 1.501668 | 85.737791 | 0.036828 | 0.041974 |
| T1P | 1.144855 | 97.750267 | 0.203711 | 0.068681 |

T0: random fixed paired+9; T1: pure Max-FRDiv; T2R: supported prototypes/reuse penalty; T1P: unchanged T1 target IDs, official-processed TRAIN specialist targets.

## Predeclared guard outcomes

- T2R: strict guard FAIL against T0.
- T1P: strict guard FAIL against T1.

All arms retain exactly ten outputs, the same architecture and 600-step budget. A higher FRC with lost diversity is not a strict win.

| Contrast | ΔFRC [session bootstrap95% CI] | ΔFRD [95% CI] | ΔFRDiv [95% CI] |
|---|---|---|---|
| T2R_vs_T0 | +0.128356 [+0.057835, +0.180918] | +0.809599 [-0.823919, +1.918898] | +0.000417 [-0.001043, +0.001550] |
| T1P_vs_T1 | +0.072547 [+0.043785, +0.110127] | +0.941274 [+0.062498, +1.482741] | -0.001712 [-0.005756, +0.001316] |

## Full TRAIN fixed-target diagnosis

All 1660 TRAIN speaker inputs, deterministic center source crops, eval-mode teachers. Values are sums of ten per-prediction CCC contributions, then context means. TRAIN fit is not held-out quality.

| Arm / steps | Fixed raw soft | Fixed raw rounded | All-session raw rounded | Fixed processed | All-session processed | VAL all-session processed |
|---|---:|---:|---:|---:|---:|---:|
| T0_600 | 0.361509 | 0.355156 | 1.563624 | 0.218003 | 1.457026 | 1.373312 |
| T0_1200 | 0.486386 | 0.454453 | 1.726146 | 0.302702 | 1.617869 | 1.453088 |
| T0_2400 | 0.698306 | 0.642290 | 1.840305 | 0.400867 | 1.744405 | 1.524074 |
| T1_600 | 1.189887 | 1.114271 | 1.539064 | 0.414772 | 1.251763 | 1.072308 |
| T1_1200 | 2.107342 | 2.001542 | 2.379973 | 0.645839 | 1.450607 | 1.133364 |
| T1_2400 | 3.201632 | 3.082168 | 3.469671 | 0.793643 | 1.465680 | 1.024060 |

The fixed TRAIN GT is explicitly in the TRAIN candidate pool. Best-GT FRC >= fixed-GT FRC is asserted for every prediction under each representation. These identities are not in VAL; this is expected split separation.

T1 fits its fixed raw targets increasingly well from 600 to 2400 updates, but held-out processed FRC does not improve monotonically. This rules out a simple claim that the fixed mapping cannot be fitted; it does not establish which part of the remaining gap is caused by representation, memorization, or other generalization effects.

## Exact-GT identity control

| Target family | GT vs own training representation | Same identity after official processing | Best all-session processed identity |
|---|---:|---:|---:|
| T0 | 7.922683 | 0.997120 | 1.343542 |
| T1 | 8.391863 | 1.080005 | 1.429912 |

Here the predictor is replaced by the actual training GT. The comparison isolates the combined temporal/representation change from model fitting and split generalization. Constant channels have CCC0, so a self-match need not equal10. A low score is not automatically proof that any evaluator implementation is incorrect.

| Arm / steps | Fixed GT is best: raw | Fixed GT is best: processed |
|---|---:|---:|
| T0_600 | 0.0233 | 0.0137 |
| T0_1200 | 0.0277 | 0.0152 |
| T0_2400 | 0.0555 | 0.0250 |
| T1_600 | 0.4555 | 0.1335 |
| T1_1200 | 0.6270 | 0.2208 |
| T1_2400 | 0.7258 | 0.2995 |

## GT reuse and session strata

| Family | Distinct specialist GT/rank (min–max) | Effective GT/rank (min–max) |
|---|---:|---:|
| T2R | 2184–2622 | 2035.3–2480.6 |
| T0 | 2083–2122 | 1850.1–1892.1 |
| T1 | 77–119 | 41.6–63.9 |

T2R TRAIN target-set FRDiv: 0.184056.

| Arm | Session stratum | Contexts | FRC | exact FRD | FRDiv |
|---|---|---:|---:|---:|---:|
| T0 | seen_session | 235 | 1.031723 | 96.159629 | 0.035339 |
| T0 | unseen_session | 336 | 1.612221 | 77.072871 | 0.037161 |
| T1 | seen_session | 235 | 0.751920 | 110.011770 | 0.200500 |
| T1 | unseen_session | 336 | 1.296389 | 87.574908 | 0.208867 |
| T2R | seen_session | 235 | 1.099179 | 95.487714 | 0.036588 |
| T2R | unseen_session | 336 | 1.783170 | 78.918648 | 0.036996 |
| T1P | seen_session | 235 | 0.831827 | 110.272152 | 0.197421 |
| T1P | unseen_session | 336 | 1.363788 | 88.992401 | 0.208110 |

The unseen sessions are session2/session10/session18. GT-pool sizes differ across strata, so this is descriptive, not an isolated causal test of session novelty.

## Post-hoc VAL representation check

The same 571 predictions are rescored against relative-time raw GT and native paired raw GT. These are diagnostic scores, not a replacement for the official-processed protocol or new selection criteria.

| Arm | All-session raw representation FRC | Native paired raw FRC | Official-processed all-session FRC |
|---|---:|---:|---:|
| T0 | 1.465193 | 0.300162 | 1.373312 |
| T1 | 1.033339 | 0.133205 | 1.072308 |
| T2R | 1.566737 | 0.326731 | 1.501668 |
| T1P | 0.962444 | 0.103707 | 1.144855 |

## Interpretation and next decision

Neither pilot is promoted: both fail their predeclared zero-slack guards. T2R improves FRC but does not retain the high-diversity T1 operating point, and its exact FRD is worse than T0. This does not isolate whether density support or reduced reuse caused the FRC gain.

T1P improves official-processed FRC but reduces raw-target VAL FRC and slightly reduces FRDiv while worsening exact FRD. Thus matching the target representation alone is not a demonstrated improvement in original-trajectory fidelity. Do not replace the teacher bank based on its FRC alone.

A fixed TRAIN identity is eligible in the TRAIN all-session search; it need not be the best-scoring identity after processing. TRAIN identities are not inserted into VAL. Both optimization/generalization and target representation must be considered; these diagnostics do not prove that the evaluator is wrong or that reuse alone caused the T1 gap.

The next bounded experiment, if pursued, should retain the T1 GT identities and raw-trajectory supervision, adding only a weak processed-target auxiliary term rather than replacing the target. Keep source schedule, optimizer updates, and paired anchor fixed; preregister processed-FRC, exact-FRD, FRDiv and raw-target diagnostic guards. This is a proposal, not a completed or known-successful method. No additional teacher or student training was launched after these two pilots.

## Compute and provenance

Each arm has ten 3,720,129-parameter teachers (37,201,290 total). q0 is reused from its exact matched baseline; nine new models ×600 updates =5400 new optimizer steps per arm. Including q0, each full ensemble budget is 6000 updates. Batch32, BF16, AdamW1e-4, the original 50-epoch cosine prefix, loss, source schedule and initial state remain matched.

| Arm | Mean training updates/s/model | Inference ms/context | Cache hits |
|---|---:|---:|---:|
| T2R | 4.617 | 85.22 | 571/571 |
| T1P | 4.718 | 57.73 | 571/571 |

Pool/output size10; one exact temporal target per source per teacher; selector overhead0. Concurrent GPU work makes latency/throughput diagnostic, not a clean benchmark. T1P also requires a one-time TRAIN VAE preprocessing cache; it does not run that processor at inference.

T1P caches 5240 TRAIN GT/length combinations. Preprocessing wall time was not separately recorded, so optimizer-budget equality must not be interpreted as equality of total preparation cost.

Initial parameter and actual source/crop hashes passed. All VAL target-cache keys and input order match their references. Checkpoint/data identities and per-context outputs stay local. Aggregate conclusions are single-seed and VAL-selected. Existing student recipes and weights remain unchanged.

See [the predeclared protocol](TEACHER_TARGET_RESEARCH_PROTOCOL.md).

Base commit: `c421f509f8cb7997e9341592cae2eb1e721b0e8e`. The pilots also record hashes of the uncommitted source files; the commit alone does not identify this research implementation.

Analysis source SHA256: `c9d6b18957a03ecd0b969888fd7b015bda228ca96c0180fda460deaa8ab05284`.

Merged TRAIN diagnostic SHA256: `6baace62f734cf70f82daba1fb46e42a1148b05076ee2163e530f0ab8d409727`.
