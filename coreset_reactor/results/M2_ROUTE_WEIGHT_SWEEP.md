# M2-v2 low-weight routing sweep

This is a single-seed matched sweep after the GT-derived behavioral routing
implementation in `fb96f35`. Every arm uses the frozen M1 architecture with
172,122 parameters, the same initialization, 2500-step source schedule,
paired/exact GT draws, TRAIN-only balanced mode assignment and deterministic
VAL protocol. Only the final routing weight changes. The routing schedule is
zero for steps 1-500, ramps to its target by step 1250, and remains constant.

## VAL64 frontier

M1 was re-evaluated with the same deterministic target post-processor. Delta
values are relative to M1; lower is better for FRD and the two coverage
distances.

| Route weight | FRC | ΔFRC | exact FRD | ΔFRD | FRDiv | ΔFRDiv | GT coverage | Δcoverage | Validity | Utilization |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 / 0 | 0.750812 | — | 82.047779 | — | 0.012697 | — | 1.091207 | — | 0.783672 | 0.710938 |
| 0.005 | 0.751374 | +0.000562 | 81.985902 | -0.061877 | 0.013006 | +0.000309 | 1.088716 | -0.002491 | 0.781153 | 0.723437 |
| 0.010 | **0.751470** | **+0.000658** | 81.920399 | -0.127381 | 0.013412 | +0.000715 | 1.085909 | -0.005298 | 0.778647 | 0.718750 |
| 0.015 | 0.751614 | **+0.000802** | **81.907642** | **-0.140137** | **0.013901** | **+0.001203** | 1.082901 | -0.008306 | 0.776471 | 0.717187 |
| 0.020 | 0.751248 | -0.000436 | 81.914404 | -0.133375 | 0.014407 | +0.001710 | 1.079829 | -0.011377 | 0.774546 | 0.706250 |
| 0.050 | 0.750834 | +0.000021 | 82.056537 | +0.008758 | **0.017687** | **+0.004990** | **1.063547** | **-0.027659** | **0.765733** | 0.718750 |

All five routing points pass the strict VAL64 FRC/FRD/FRDiv/coverage gate.
The larger weights buy more descriptor diversity, but the 0.05 point already
shows the quality tradeoff that motivated the low-weight sweep. Weight 0.015
is the best VAL64 quality/diversity frontier point among the low weights.

## Full VAL571 confirmation for the selected 0.015 point

| Metric | M1 | Route 0.015 | Delta |
|---|---:|---:|---:|
| FRC ↑ | 0.810111 | 0.807115 | -0.002996 |
| exact FRD ↓ | 82.480136 | **82.375193** | **-0.104943** |
| FRDiv ↑ | 0.013611 | **0.015051** | **+0.001440** |
| FRVar ↑ | 0.039200 | 0.039444 | +0.000244 |
| GT coverage ↓ | 1.095160 | **1.086864** | **-0.008296** |
| Prediction validity ↓ | 0.785032 | **0.777992** | **-0.007040** |
| Candidate utilization ↑ | 0.769352 | 0.761821 | -0.007531 |

The full split keeps the FRD, diversity and coverage gains. FRC decreases by
0.0030 (0.37%), which is materially smaller than the 0.02 point's 0.00436
decrease, but it is still not a strict zero-slack full-VAL win.

## Full-VAL role alignment

The same direct alignment diagnostic assigns each VAL GT to its nearest
TRAIN-derived behavioral centroid and ranks the ten slots for each mode.

| Alignment metric | M1 | Route 0.015 | Delta |
|---|---:|---:|---:|
| Correct slot top-1 | 11.1% | **33.7%** | +22.6 pp |
| Mean reciprocal rank | 0.314 | **0.529** | +0.215 |
| Own-mode soft-min distance ↓ | 0.8773 | **0.8643** | -0.0129 |
| Diagonal margin ↑ | -0.0109 | **-0.0053** | +0.0055 |
| Slot-pair FRDiv | 0.013611 | **0.015051** | +10.6% |
| Descriptor-centroid distance | 0.066041 | **0.074948** | +13.5% |
| Effective slots | 9.286 | 8.696 | -0.590 |
| Mean utilized slots/context | 7.694 | 7.618 | -0.075 |
| Contexts using all ten slots | 24.5% | 24.2% | -0.4 pp |

Compared with route 0.02, weight 0.015 retains the identity gain while
recovering slot balance: effective slots rise from 8.41 to 8.70 and all-ten
usage returns from 22.6% to 24.2%.

## Decision

The current M2-v2 candidate is `lambda_route=0.015`. It is the strongest
low-weight point in the matched VAL64 sweep and the best tested compromise on
full VAL571: FRC loss is reduced, FRD and coverage improve, and direct
slot-to-mode alignment remains strong. The full-VAL FRC result is still a
small regression, so this should be described as a near-Pareto operating point,
not a universal strict win.

The 0.10 point remains intentionally unrun; 0.05 already establishes the
stronger-routing concentration trend. No three-seed replication or adaptive
sampler result is claimed by this sweep.

Raw reports, checkpoints, per-context records, sample identifiers, datasets
and local paths are not published.
