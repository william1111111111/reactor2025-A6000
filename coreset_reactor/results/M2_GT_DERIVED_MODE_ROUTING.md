# CoReSet-Reactor M2-v2: GT-derived behavioral mode routing

M2-v2 tests whether explicit slot identity, rather than extra model capacity,
is sufficient to specialize the ten M1 outputs. The 172,122-parameter
ParallelTCN, optimizer, source schedule, exact-GT draws and M1 losses are
unchanged. The only added term is

```text
L = L_M1 + lambda_route L_route
```

where slot k is routed toward same-session GT descriptors belonging to global
TRAIN mode k. There is no mode adapter, diversity repulsion or inference-time
GT access.

## TRAIN-only routing artifact

The existing 1,660 normalized 602-D TRAIN descriptors are deterministically
partitioned into ten capacity-balanced medoid modes. Every mode contains 166
GTs. Fifteen of 17 TRAIN sessions support all ten modes and two support nine.
The assignment converges in three iterations and has SHA-256
`52dd2e739ec908eab08bd100d361b81f2245d80c91c281ecb2c9e3f886d0a5c8`.
VAL does not enter clustering.

The shared initialization SHA is
`841d6368118e521a6b995871b7a61c08169764293284da57cc3c57d4ce22862e`
and the 2500-step schedule SHA is
`8280076a2a3e43c17eefcb02b5bc0869476fb7c7267c3d9ae5e2d5d92366cbd5`,
both identical to frozen M1. Routing is zero for steps 1-500, ramps linearly
to its target over steps 501-1250, then remains constant.

## Deterministic VAL64 frontier

All values use seed `20260918`, evaluation seed `1234`, 2500 steps and the
same 64 contexts. M1 FRC/FRD were re-evaluated under the same deterministic
post-processor protocol rather than copied from the older metrics file.

| Route weight | FRC ↑ | exact FRD ↓ | FRDiv ↑ | GT coverage ↓ | Validity ↓ | Utilization ↑ |
|---|---:|---:|---:|---:|---:|---:|
| M1 / 0 | 0.750812 | 82.047779 | 0.012697 | 1.091207 | 0.783672 | 0.710938 |
| 0.02 | **0.751248** | **81.914404** | 0.014407 | 1.079829 | 0.774546 | 0.706250 |
| 0.05 | 0.750834 | 82.056537 | **0.017687** | **1.063547** | **0.765733** | **0.718750** |

Weight 0.02 passes the strict four-part pilot gate versus M1: FRC improves by
0.000436, exact FRD improves by 0.133375, FRDiv improves by 0.001710 (13.5%)
and coverage distance improves by 0.011377. Weight 0.05 provides a larger
FRDiv gain (39.3%) but gives back 0.008758 exact FRD, so it is a useful
frontier point rather than the default operating point.

## Full VAL571 confirmation for weight 0.02

| Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | GT coverage ↓ | Validity ↓ | Utilization ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | **0.810111** | 82.480136 | 0.013611 | 0.039200 | 1.095160 | 0.785032 | **0.769352** |
| M2-v2 / 0.02 | 0.805749 | **82.360061** | **0.015685** | **0.039502** | **1.083657** | **0.776017** | 0.756567 |
| Delta | -0.004363 | -0.120075 | +0.002074 | +0.000302 | -0.011503 | -0.009015 | -0.012785 |

The full split preserves the FRD, diversity, coverage and validity gains.
FRC decreases by 0.54%, so this is a near-Pareto result under a small quality
tolerance, not a strict zero-slack full-VAL win.

## Specialization and direct route alignment

The first table uses all 571 contexts. Weight 0.05 is included to show the
strength-versus-balance tradeoff.

| Metric | M1 | route 0.02 | route 0.05 |
|---|---:|---:|---:|
| Slot-pair FRDiv | 0.013611 | 0.015685 | 0.019571 |
| Descriptor-centroid distance | 0.066041 | 0.080016 | 0.115249 |
| Mean AU activation range | 0.023965 | 0.034782 | 0.054346 |
| VA valence mean range | 0.061130 | 0.083246 | 0.159102 |
| VA arousal mean range | 0.055809 | 0.065682 | 0.090540 |
| Effective slots | **9.286** | 8.410 | 7.982 |
| Mean utilized slots/context | **7.694** | 7.566 | 7.573 |
| Contexts using all ten slots | **24.5%** | 22.6% | 16.1% |

For a direct identifiability check, each VAL GT is assigned to its nearest
TRAIN-derived mode centroid. For every supported mode, the diagnostic ranks
all ten prediction slots by soft-min distance to that mode's GTs.

| Alignment metric | M1 | route 0.02 | Delta |
|---|---:|---:|---:|
| Correct slot is top-1 | 11.1% | **39.2%** | +28.1 pp |
| Mean reciprocal rank | 0.314 | **0.574** | +0.260 |
| Own-mode soft-min distance ↓ | 0.8773 | **0.8607** | -0.0166 |
| Diagonal margin ↑ | -0.0109 | **-0.0042** | +0.0066 |

This confirms that the increased centroid separation is learned behavioral
role alignment rather than undirected output noise. It is not complete:
the mean diagonal margin remains slightly negative, one mode has only 5.2%
top-1 alignment, and responsibility entropy decreases. Distinct dominant
expression count also remains one; the observed specialization is mainly in
AU intensity/dynamics and VA behavior.

## Decision

The central M2-v2 hypothesis is supported: explicit GT-derived routing makes
the existing slot-specific head substantially more identifiable without any
new model parameters. Weight 0.02 is the current candidate because it passes
the strict VAL64 gate and retains most gains on full VAL571. It is not yet a
fully frozen replacement for M1 because full-VAL FRC drops slightly and slot
responsibility becomes less balanced.

Weight 0.10 was not run: weight 0.05 already shows monotonic concentration and
misses strict FRD, so a stronger point would not test the current bottleneck.
No three-seed replication or adaptive sampler experiment is claimed here.
Raw reports, checkpoints, per-context records, sample identifiers, datasets
and local paths are not published.
