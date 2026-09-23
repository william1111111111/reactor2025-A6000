# Quality-gated Max-FRDiv teacher: full matched result

## Conclusion

Source-compatible target filtering converts part of the session diversity into
learned diversity, but the present fixed-target teacher still has a quality
tradeoff. The best operating point in this sweep is `DESC_Q50`:

- FRDiv: `0.036411 → 0.107139` (`+194.3%`, 2.94x)
- FRC: `1.373284 → 1.337285` (`−2.62%`)
- exact FRD: `84.926861 → 88.289176` (`+3.362`, lower is better)
- GT descriptor coverage: `1.138256 → 1.074316` (improved)

It is a useful diversity frontier knee, but it does not meet the pre-defined
strict or near-Pareto safety criteria. No student, KD, spread scaling, or
architecture change was introduced.

## Matched protocol

All five arms used 10 independent ConditionalREGNN models with the same seed,
initialization, source schedule, optimizer, batch, precision, loss, formal
50-epoch schedule and 600-step prefix. The only changed factor was the fixed
TRAIN target manifest.

Evaluation is full VAL571 with center750, deterministic postprocessing, official
AU rounding, exact rolling FRD, and all same-session opposite-role GT. The
evaluator still outputs exactly ten predictions.

| Arm | TRAIN target FRDiv | VAL FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | GT coverage ↓ | validity ↓ | utilization ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 random paired+9 | 0.200888 | 1.373284 | 84.926861 | 0.036411 | 0.036258 | 1.138256 | 0.929190 | 0.650088 |
| DESC_Q50 | 0.221019 | 1.337285 | 88.289176 | 0.107139 | 0.048853 | 1.074316 | 0.926192 | 0.769002 |
| DESC_Q70 | 0.235931 | 1.190626 | 88.443215 | 0.125165 | 0.049953 | 1.078707 | 0.955689 | 0.785639 |
| DESC_Q80 | 0.244072 | 1.198924 | 89.796506 | 0.145586 | 0.054441 | 1.081409 | 0.971591 | 0.764623 |
| T1 pure Max-FRDiv | 0.273745 | 1.072329 | 96.817232 | 0.205433 | 0.068095 | 1.118582 | 1.086842 | 0.764098 |

Relative to T0, the diversity gains are `+194.3%`, `+243.8%`, `+299.8%`, and
`+464.2%` for Q50/Q70/Q80/T1. FRC drops are `−2.62%`, `−13.30%`, `−12.70%`,
and `−21.91%` respectively.

## Specialization

| Arm | Slot-pair FRDiv | Descriptor centroid distance | AU range | VA mean range | Distinct expression modes |
|---|---:|---:|---:|---:|---:|
| T0 | 0.036411 | 0.052351 | 0.085288 | 0.126814 | 1.716 |
| DESC_Q50 | 0.107139 | 0.127985 | 0.194828 | 0.266506 | 4.180 |
| DESC_Q70 | 0.125166 | 0.141982 | 0.239548 | 0.351209 | 4.496 |
| DESC_Q80 | 0.145586 | 0.153600 | 0.257183 | 0.295497 | 4.825 |
| T1 | 0.205434 | 0.196661 | 0.244760 | 0.404446 | 4.609 |

The monotone increase in slot-pair diversity and descriptor-centroid distance
shows that the result is learned specialization, not only metric noise.

## Per-rank quality diagnosis

The following arrays are ordered by fixed rank `q0...q9`. Rank zero remains the
true paired anchor in every arm.

| Arm | Per-rank FRC | Per-rank exact FRD |
|---|---|---|
| T0 | 0.1226, 0.1232, 0.1437, 0.1490, 0.1519, 0.1421, 0.1272, 0.1419, 0.1340, 0.1378 | 8.261, 8.255, 8.517, 8.662, 8.499, 8.328, 8.456, 8.540, 8.492, 8.916 |
| DESC_Q50 | 0.1226, 0.1369, 0.1483, 0.1570, 0.1536, 0.1531, 0.1150, 0.0975, 0.1371, 0.1163 | 8.261, 8.886, 9.155, 9.951, 8.897, 8.570, 8.682, 8.536, 8.617, 8.734 |
| DESC_Q70 | 0.1226, 0.1418, 0.1026, 0.1585, 0.1632, 0.1421, 0.0729, 0.0908, 0.1078, 0.0882 | 8.261, 9.037, 9.011, 9.702, 8.886, 8.329, 8.255, 8.876, 8.849, 9.237 |
| DESC_Q80 | 0.1226, 0.1319, 0.0811, 0.1604, 0.1730, 0.1583, 0.0758, 0.0898, 0.1069, 0.0992 | 8.260, 8.794, 8.985, 9.264, 9.576, 8.607, 8.647, 8.843, 9.280, 9.540 |
| T1 | 0.1226, 0.1044, 0.0885, 0.1059, 0.1684, 0.1327, 0.0655, 0.1203, 0.0806, 0.0836 | 8.262, 9.660, 9.657, 9.986, 9.665, 9.590, 9.018, 10.551, 9.765, 10.664 |

Q50 is the clearest knee: it retains useful quality across most ranks while
increasing specialization. Q70/Q80 begin to expose more low-quality specialist
ranks, and pure T1 is the strongest quality-diversity tradeoff.

## Safety criteria

| Criterion | Requirement | Best observed | Result |
|---|---|---|---|
| Strict FRC | FRC ≥ T0 | Q50 `1.337285` | FAIL |
| Strict exact FRD | exact FRD ≤ T0 | Q50 `88.289176` | FAIL |
| Strict diversity | FRDiv > T0 | Q50 `0.107139` | PASS |
| Near-Pareto FRC | relative drop ≤ 0.5% | Q50 drop `2.62%` | FAIL |
| Near-Pareto exact FRD | increase ≤ 1.0 | Q50 increase `3.362` | FAIL |
| Near-Pareto diversity | gain ≥ 50% | Q50 gain `194.3%` | PASS |

## Decision

The quality-gated target family is a valid conditional-diversity mechanism, but
this global descriptor gate alone does not produce a quality-safe operating
point. Per the stopping rule, do not continue a finer quantile sweep as the
next main experiment. The next research question should be source-conditioned
compatibility or quality-gated/mode-specific supervision, while preserving the
frozen T0 baseline.

## Provenance

- repository commit: `220727c7fa6f7c766a200a00189db97f5f4f8d06`
- source snapshot SHA256: `51f892fe8f307ae95f5906098e53eb911edecb318f02116966206cd78033ec01`
- quality target manifest SHA256: `e39d2e8fc3569906ce7c5a50a4413452f0345ddbe426417519488cf96dd1a82a`
- rank manifest index SHA256: `7cb704bdf8342918c153cc5f4a0b7700ba573b16b826b6f09f14d193e7394deb`
- formal-run provenance SHA256: `199f5ed9423f4bd45bb2defa4213a4c39cb04ebbb8ece7a913bebfcbc6084e18`
- data fingerprint: `e39d2e8fc3569906ce7c5a50a4413452f0345ddbe426417519488cf96dd1a82a`
- training: 5 arms × 10 ranks, batch 32, BF16, AdamW `1e-4`, 600 steps,
  formal 50-epoch cosine schedule, GPUs 1/5/6
