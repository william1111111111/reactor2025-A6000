# CoReSet-Reactor M1: post-hoc gradient diagnostics

This summary uses seed `20260918` and the final checkpoint of each listed
arm. Five fixed training-schedule probes (positions 1, 150, 300, 450 and 600)
were re-evaluated with valid source prefixes in the model, SoftDTW and
descriptor computations. The numbers are medians of gradient L2 norms across
those five probes. They are **not** gradients measured during training at
those steps, and they do not establish causation.

| Final checkpoint | Paired gradient norm | Weighted SoftDTW gradient norm | Weighted descriptor-block gradient norm | Median descriptor/paired ratio |
|---|---:|---:|---:|---:|
| B1 | 0.205807 | 0.069631 | not active | not applicable |
| B3 / 1.0 | 0.237726 | 0.045822 | 0.421169 | 1.850484 |
| B3 / 0.5 | 0.242251 | 0.055659 | 0.280617 | 0.883550 |
| B3 / 0.25 | 0.210279 | 0.050618 | 0.139918 | 0.514206 |

For B3 / 1.0, the descriptor-to-paired ratio exceeds one in all five
same-seed probes; across the three original B3 / 1.0 seeds it exceeds one in
14 of 15 probes (median ratio 2.17). One short-source probe was included in
each of the four same-seed groups, and the corrected valid-length path was
used. The load-penalty value was zero in all five same-seed probes for each
B3 weight. This suggests the descriptor block can dominate local sensitivity
at full weight, but a training-time gradient trace or intervention would be
needed to establish why FRC changes.

`--descriptor-cover` weights coverage, validity and load together. No
sample-level diagnostic, checkpoint, dataset, path or exact per-run Git SHA
is included. The original run configs did not record `git_commit_sha`.
