# CoReSet-Reactor M2 preflight: GT ceiling and slot responsibility

This diagnostic freezes the M1 operating point at B3, static descriptor weight
0.10, 2500 steps, seed `20260918`. It evaluates all 571 VAL contexts from 20
sessions with ten outputs. The checkpoint SHA-256 is
`469aa1e0c64ad07b0391105c9a02458f071be28da9fd2c3446b63b283d3733c9`.

For the real-GT pool diagnostic, every same-session listener trajectory is
deterministically center-cropped or linearly stretched to the context's valid
source length. Selection uses the same normalized squared sequence distance
as official FRDiv. Random-10 is seeded per context; k-medoids uses deterministic
BUILD plus PAM-style swaps; farthest-10 uses greedy farthest-sum plus swap
refinement and is an approximation, not a claimed combinatorial optimum.

## Real-GT diversity ceiling

| Set | Context-weighted FRDiv | Session-balanced FRDiv | Median | P10 | P90 | Ratio to model |
|---|---:|---:|---:|---:|---:|---:|
| Frozen B3 predictions | 0.013611 | 0.014007 | — | — | — | 1.00x |
| Random real GT-10 | 0.203625 | 0.206547 | 0.210052 | 0.158219 | 0.236765 | 14.96x |
| k-medoids real GT-10 | 0.226351 | 0.219801 | 0.230381 | 0.201007 | 0.237912 | 16.63x |
| Farthest-sum real GT-10 | 0.265977 | 0.237573 | 0.277236 | 0.206300 | 0.304022 | 19.54x |
| Every real GT in the pool | 0.204326 | 0.207091 | 0.212109 | 0.169472 | 0.230165 | 15.01x |

Every real-GT construction exceeds the prediction FRDiv in every one of the
571 contexts. The conclusion also holds separately for the 458 full-length
contexts and the 113 short contexts, so it is not caused by short-sequence
interpolation. The available same-session pool has a large diversity ceiling;
the model is not close to it.

## Prediction-slot responsibility

Each real GT is assigned to its nearest prediction using the same TRAIN-scaled
602-D descriptor space as the M1 all-seen loss.

| Slot | Global responsibility | Contexts with nonzero responsibility | Mean FRDiv to other slots | Mean VA |
|---:|---:|---:|---:|---:|
| q0 | 6.82% | 65.1% | 0.01336 | (-0.170, 0.165) |
| q1 | 5.39% | 67.6% | 0.01664 | (-0.151, 0.150) |
| q2 | 5.63% | 69.0% | 0.01652 | (-0.158, 0.142) |
| q3 | 9.15% | 79.9% | 0.01290 | (-0.121, 0.151) |
| q4 | 11.11% | 79.2% | 0.01245 | (-0.173, 0.114) |
| q5 | 8.13% | 74.6% | 0.01283 | (-0.139, 0.141) |
| q6 | 17.97% | 94.4% | 0.01372 | (-0.182, 0.110) |
| q7 | 14.37% | 84.1% | 0.01239 | (-0.157, 0.154) |
| q8 | 13.42% | 84.2% | 0.01208 | (-0.178, 0.148) |
| q9 | 8.01% | 71.3% | 0.01323 | (-0.164, 0.154) |

This is not classic global dead-slot collapse. Global responsibility has 9.29
effective slots; each context uses 7.69 slots on average (median 8), and 24.5%
of contexts use all ten. However, 17.7% use at most five slots, and q6 has the
largest responsibility in 14 of 20 sessions.

The larger weakness is semantic homogeneity rather than complete inactivity:

- every slot has the same dominant expression category;
- the mean per-AU activation-rate range across slots is only 0.024;
- the two VA means span only 0.061 and 0.056 across slots;
- pairwise slot FRDiv occupies a narrow, low range of 0.00975–0.01904;
- normalized descriptor-centroid distances occupy 0.0453–0.0848, with q7/q8
  the closest pair under both sequence and descriptor measurements.

## Decision

The GT pool contains 15–20 times the model's diversity, while the existing
heads are used but do not acquire strongly separated, stable semantic roles.
The next controlled experiment should therefore keep the frozen M1 losses and
budget, add only lightweight learned mode codes, and compare against the exact
frozen B3 configuration. Adaptive exact-GT sampling should remain deferred
until mode specialization is tested.

This remains a development-set diagnostic, not an official TEST result. The
same-session GTs are pool-diversity proxies and are not verified reactions to
the identical stimulus. No checkpoint, dataset, local path, sample identifier,
per-context record or raw responsibility matrix is included here.
