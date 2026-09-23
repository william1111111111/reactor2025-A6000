# Dataset GT oracle ceiling (REACT2025 VAL)

This diagnostic treats each VAL listener directory as one independent session
pool. Every listener trajectory in a session is converted to 750 frames using
the same center-crop/linear-stretch rule used by the existing GT ceiling code.
For a ten-element subset, FRDiv is the ordered pairwise mean of squared
Euclidean distances divided by `25*750`.

The random value is the exact expectation of a uniformly random ten-subset
(the all-pair mean); 10,000 seeded draws are also recorded in the machine JSON.
Farthest is the existing deterministic greedy farthest-sum plus one-swap lower
bound. Exact uses binary `x_i` and `z_ij=x_i*x_j` variables and SciPy/HiGHS or
HiGHS-native MILP.

## Per-session results

| Session | N | Random expectation | Farthest lower bound | Best known | Exact status |
|---|---:|---:|---:|---:|---|
| session0 | 14 | 0.199402 | 0.222283 | 0.222283 | exact |
| session1 | 14 | 0.210338 | 0.229802 | 0.229802 | exact |
| session10 | 113 | 0.169472 | 0.277236 | 0.277236 | not proved |
| session11 | 13 | 0.187757 | 0.206300 | 0.206300 | exact |
| session12 | 13 | 0.186592 | 0.203843 | 0.203843 | exact |
| session13 | 14 | 0.206026 | 0.230327 | 0.230327 | exact |
| session14 | 14 | 0.234977 | 0.253431 | 0.253431 | exact |
| session15 | 14 | 0.226753 | 0.242792 | 0.242792 | exact |
| session16 | 14 | 0.219670 | 0.241608 | 0.241608 | exact |
| session17 | 14 | 0.237113 | 0.259020 | 0.259020 | exact |
| session18 | 111 | 0.223036 | 0.304022 | 0.304022 | not proved |
| session19 | 14 | 0.230165 | 0.244608 | 0.244608 | exact |
| session2 | 112 | 0.212109 | 0.296140 | 0.296140 | not proved |
| session21 | 13 | 0.209958 | 0.228939 | 0.228939 | exact |
| session3 | 14 | 0.227866 | 0.242694 | 0.242894 | exact |
| session4 | 14 | 0.230244 | 0.251326 | 0.251326 | exact |
| session5 | 14 | 0.171021 | 0.201611 | 0.201611 | exact |
| session6 | 14 | 0.130543 | 0.151169 | 0.151169 | exact |
| session7 | 14 | 0.233021 | 0.251501 | 0.251501 | exact |
| session9 | 14 | 0.187029 | 0.203616 | 0.203616 | exact |

## Summary

- 20 VAL sessions; the actual VAL partition has 17 sessions with 13–14
  listener GTs and 3 sessions with 111–113 listener GTs. It therefore does not
  have 20 pools of roughly 90–100 items; that larger pool shape applies to the
  TRAIN partition.
- Session-balanced random-10 expectation: **0.206655**.
- Session-balanced farthest lower bound: **0.237113**.
- Session-balanced best-known lower bound: **0.237123**.
- 17/20 sessions were proven exact. Their exact mean is **0.227357**.
- For the three large sessions, 600-second HiGHS warm-start runs still did not
  prove optimality. Their valid best-known values are 0.277236, 0.304022 and
  0.296140. Consequently, **0.237123 is a strict lower bound for the full
  VAL session-balanced exact ceiling, not the exact full mean**.
- On the 17 solved sessions, farthest-sum is already extremely close to exact:
  mean farthest/exact ratio **0.999952**. The only visible improvement is
  session3: 0.242694 → 0.242894.

This VAL session-level oracle is not numerically interchangeable with the
previous context-conditioned 571-context ceiling: the latter re-aligns each
session pool to each source length and weights contexts, while this experiment
uses one 750-frame pool per session. GPU acceleration is not expected to help
the exact MILP branch-and-bound; the bottleneck is combinatorial CPU solving.

Machine-readable result: local ignored `results/dataset_gt_oracle_val_v4.json`.
