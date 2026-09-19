# CoReSet-Reactor M2: lightweight mode adapter v1

This is a controlled negative result. The frozen baseline is M1 B3 with static
descriptor weight 0.10, 2500 steps and seed `20260918`. M2 changes only the
architecture by adding one learned 16-D code per slot and a shared nonlinear
64-D residual adapter at the control-point features:

```text
u_k(t) = GELU(W_h H(t) + W_z z_k)
raw_k_M2(t) = raw_k_M1(t) + W_o u_k(t)
```

`W_o` weight and bias are zero-initialized, so M1 and M2 produce exactly equal
outputs before the first update. The adapter adds 11,065 parameters to the
172,122-parameter M1 model. No loss, sampler, TCN block, training budget or
evaluation subset changes. The full schedule SHA and shared-initialization SHA
match the frozen M1 run exactly.

## 600-step safety gate

The safety run uses the first 600 steps of the planned 2500-step schedule, not
a separately generated 600-step schedule.

| Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | GT coverage distance ↓ |
|---|---:|---:|---:|---:|
| M1 | 0.654740 | 85.708847 | 0.018275 | 1.104191 |
| M2 adapter | 0.657863 | 87.907772 | 0.015079 | 1.109756 |
| M2 minus M1 | +0.003123 | +2.198925 | -0.003197 | +0.005565 |

Training is numerically stable and FRC does not degrade, but FRD, FRDiv and
coverage are already worse. Full-VAL specialization also moves in the wrong
direction: mean slot-pair FRDiv falls from 0.01902 to 0.01613 and mean
descriptor-centroid distance falls from 0.10308 to 0.08385.

## 2500-step matched result

| Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | GT coverage distance ↓ | Validity distance ↓ | Candidate utilization ↑ |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 0.750812 | 82.047779 | 0.012697 | 1.091207 | 0.783672 | 0.710938 |
| M2 adapter | 0.736246 | 81.753210 | 0.011478 | 1.087159 | 0.778094 | 0.679688 |
| M2 minus M1 | **-0.014567** | -0.294569 | **-0.001219** | -0.004047 | -0.005578 | -0.031250 |

M2 improves FRD and the two descriptor distances slightly, but fails the FRC
and FRDiv requirements.

## Full-VAL specialization

| Metric | M1 | M2 adapter | Delta |
|---|---:|---:|---:|
| FRDiv | 0.013611 | 0.012780 | -0.000831 |
| Mean descriptor-centroid distance | 0.066041 | 0.062835 | -0.003206 |
| Effective slots | 9.286 | 8.223 | -1.064 |
| Mean utilized slots/context | 7.694 | 7.326 | -0.368 |
| Contexts using all 10 slots | 24.5% | 17.3% | -7.2 pp |
| Contexts using at most 5 slots | 17.7% | 23.8% | +6.1 pp |
| Mean AU activation range | 0.02396 | 0.01933 | -0.00463 |
| Distinct dominant expressions | 1 | 1 | 0 |

The q6 responsibility share rises from 18.0% to 28.9%. Thus the adapter is not
merely neutral: under the unchanged M1 objective it learns a largely shared
correction and makes slot responsibility more concentrated.

Training throughput decreases by about 4.0% (21.87 to 20.98 samples/s), while
peak GPU memory is effectively unchanged. The final 500-step average training
loss is almost identical (0.93975 M1 versus 0.93942 M2), so lower diversity is
not explained by failed optimization of the existing objective.

## Decision

Mode-adapter v1 fails the preregistered quality/diversity/specialization gate.
No three-seed replication, full-VAL quality run, specialization loss or
adaptive sampler experiment is authorized by this result. A follow-up should
first change how slot identity is made identifiable; simply adding capacity
behind a permutation-symmetric objective is insufficient.

This is a deterministic development-set experiment, not an official TEST
result or significance claim. Raw reports, checkpoints, sample identifiers,
datasets and local paths are not published.
