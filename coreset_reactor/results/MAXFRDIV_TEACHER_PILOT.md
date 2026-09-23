# Matched fixed-teacher Max-FRDiv pilot

## Executive result

This is a strict matched **600-step safety pilot** comparing two fixed-teacher
target policies. The experiment was stopped at the safety gate.

T1 (paired anchor plus an approximate maximum-FRDiv set of nine TRAIN targets)
transfers substantially more diversity and produces much stronger slot
specialization than T0 (paired anchor plus nine random TRAIN targets). However,
T1 also loses quality on the official full VAL571 evaluation:

- FRC: `1.072215` vs `1.373332` for T0 (`-0.301117`)
- exact FRD: `96.813896` vs `84.929128` for T0 (`+11.884768`; lower is better)
- FRDiv: `0.205415` vs `0.036415` for T0 (`+0.169001`, `+464.1%`)

Therefore T1 is a clear **diversity-transfer and specialization result**, but
not a quality-safe teacher. The planned full teacher run and student
distillation were not started.

## Matched protocol

The only changed factor was the fixed TRAIN target selection policy.

| Item | Setting |
|---|---|
| Arms | T0 random fixed targets; T1 approximate Max-FRDiv fixed targets |
| Models | 10 independent models per arm, one fixed rank per model |
| Total models | 20 |
| Training budget | 600 optimizer steps (formal 50-epoch schedule prefix) |
| Dataset records | 3,320 source-role records |
| Source roles | Both deterministic source/listener roles covered |
| Batch size | 32 |
| Precision | BF16 |
| Optimizer | AdamW, lr `1e-4`, weight decay `1e-4`, betas `(0.9, 0.99)` |
| LR schedule | Cosine, `T_max=50`, final lr `0.05x` initial lr |
| Gradient clipping | `1.0` |
| Seed | `1` |
| Alignment | `relative_time_masked` |
| Evaluation | Full VAL571, official 750-frame center crop, all same-session GT matching |

All arms used the same initialization, source schedule, optimizer, seed,
training budget, and evaluation subset. The evaluator ran with 16 metric
workers on CUDA. The output contains exactly ten predictions: one from each
rank/model.

## Aggregate VAL571 result

| Arm | TRAIN target-set FRDiv | VAL FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | GT coverage ↓ | prediction validity ↓ | candidate utilization ↑ | GT→teacher transfer* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 random | 0.197589 | 1.373332 | 84.929128 | 0.036415 | 0.036260 | 1.138303 | 0.929214 | 0.650263 | 0.184296 |
| T1 Max-FRDiv | 0.278971 | 1.072215 | 96.813896 | 0.205415 | 0.068097 | 1.118545 | 1.086761 | 0.764098 | 0.736332 |
| T1 − T0 | +0.081382 | −0.301117 | +11.884768 | +0.169001 | +0.031836 | −0.019758 | +0.157547 | +0.113835 | +0.552037 |

`GT→teacher transfer` is the VAL output FRDiv divided by the TRAIN fixed
target-set FRDiv. It is a cross-split diagnostic, not a same-context identity.

## Per-rank quality

| Rank | T0 FRC | T1 FRC | T0 exact FRD | T1 exact FRD |
|---:|---:|---:|---:|---:|
| 0 | 0.122617 | 0.122625 | 8.263193 | 8.262084 |
| 1 | 0.123194 | 0.104364 | 8.256000 | 9.662353 |
| 2 | 0.143686 | 0.088468 | 8.514449 | 9.657520 |
| 3 | 0.149000 | 0.105901 | 8.661811 | 9.986434 |
| 4 | 0.151844 | 0.168344 | 8.501097 | 9.665810 |
| 5 | 0.142136 | 0.132628 | 8.327890 | 9.590165 |
| 6 | 0.127187 | 0.065396 | 8.456660 | 9.017307 |
| 7 | 0.141892 | 0.120332 | 8.540048 | 10.548595 |
| 8 | 0.133985 | 0.080523 | 8.491841 | 9.765551 |
| 9 | 0.137790 | 0.083634 | 8.916139 | 10.658076 |

The T1 diversity gain is not caused by one isolated rank; it is accompanied by
quality degradation across most ranks.

## Specialization and diversity diagnostics

| Diagnostic | T0 | T1 |
|---|---:|---:|
| Slot-pair FRDiv | 0.036415 | 0.205416 |
| Descriptor centroid distance | 0.052349 | 0.196641 |
| AU activation range | 0.085329 | 0.244736 |
| VA mean range | 0.126966 | 0.404067 |
| Distinct expression modes | 1.716287 | 4.609457 |

The result supports the intended mechanism: Max-FRDiv teacher targets create
distinct behavioral roles rather than merely adding small perturbations. The
transfer ratio of `0.736` also shows that a large part of the target diversity
survives into VAL predictions, but the quality cost is too large.

## Pre-registered safety gate

| Gate | Requirement | Observed | Result |
|---|---:|---:|---|
| Diversity output | T1 FRDiv > T0 FRDiv | `0.205415 > 0.036415` | PASS |
| Relative diversity gain | ≥ 10% | `+464.1%` | PASS |
| FRC safety | T1 FRC ≥ T0 FRC − 0.02 | `1.072215 < 1.353332` | FAIL |
| exact-FRD safety | T1 exact FRD ≤ T0 exact FRD + 2.0 | `96.813896 > 86.929128` | FAIL |

## Provenance

The raw checkpoints, datasets, and local reports are intentionally not part of
the repository. The following hashes identify the frozen inputs and the
sanitized pilot record:

| Artifact | SHA256 / identifier |
|---|---|
| Training source commit | `316989c768c79e2f34cca54c25e5fb38c5a6f3f3` |
| Current evaluator commit | `63dca25` |
| Frozen source snapshot | `d7cc2dda27ad555ec8d58c7d4dc3eb0bdba1b6e1046b95c469a7e65f030dd757` |
| Frozen target manifest file | `13d2ad79aa558fcd7fc524088d6d3feb81c6508e8079237d1272a2c4fb670b77` |
| Frozen target manifest internal hash | `9eb9b24cfefae2c22b82b11e3d4572dd91c0aa89daa2fd494b771321e924859d` |
| Manifest index | `3cbc7c1ef39cf287e12c8a0c20e237cadda5a1b227d9fd81c04dc12bf65d37e2` |
| Pilot provenance | `cd63e386dc66a14d8893d8eac09dd399606033a814d773ad5b3a06186aeac568` |

## Decision

Do not promote T1 to the full teacher experiment in its current form. The next
research step should constrain or gate the diversity teacher objective so that
the observed specialization is retained while restoring FRC and exact FRD.
