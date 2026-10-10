> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# CoReSet-Reactor M1 single-seed quality frontier

Local-only aggregate of the deterministic VAL pilot: seed `20260918`, 600
training steps, batch size 2, `eval_seed=1234`, and the same 64 VAL contexts.
All rows use ten reaction predictions. B1 has no descriptor block; B3 uses
paired + 17 sampled same-session exact GTs and all-seen descriptor supervision.
The 0.05, 0.15, 0.20 and ramp rows were run on `cuda:5` in this follow-up.

| B3 descriptor policy | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | GT coverage distance ↓ | Four strict criteria |
|---|---:|---:|---:|---:|---:|---|
| B1 (no descriptor) | 0.688476 | 88.613668 | 0.007848 | 0.032886 | 1.246347 | baseline |
| static 0.05 | 0.700214 | 86.445197 | 0.011299 | 0.034815 | 1.166816 | pass |
| static 0.10 | 0.690499 | 84.664277 | 0.017054 | 0.036089 | 1.113882 | pass |
| static 0.125 | 0.680634 | 84.252299 | 0.019896 | 0.036482 | 1.093196 | fail: FRC |
| static 0.15 | 0.673672 | 83.728105 | 0.021529 | 0.036678 | 1.081519 | fail: FRC |
| static 0.20 | 0.653684 | 83.103936 | 0.025867 | 0.036656 | 1.064957 | fail: FRC |
| ramp 0 to 0.15 | 0.663118 | 84.111712 | 0.013152 | 0.037094 | 1.098378 | fail: FRC |

The strict pilot criterion is FRC >= B1, exact FRD <= B1, FRDiv > B1 and GT
coverage distance < B1. Among passing policies, static 0.10 gives more
FRDiv/coverage improvement but only a +0.002023 FRC margin; static 0.05 gives
a larger +0.011738 FRC margin with a smaller diversity gain. The tested ramp
(weight zero through step 120, linear increase to 0.15 at step 360, then
constant) is worse than static 0.15 on all four strict metrics. The applied
weights were checked in its training curve at steps 1, 120, 240, 360 and 600.

This is **one seed and one 64-context VAL subset**, not a full official TEST
score or a generalization claim. Same-session GTs are not verified
same-stimulus reactions. The 0.05 run began before source-provenance capture
was added; older 0.10/0.125/B1 runs also lack exact run-time SHA. New 0.15,
0.20 and ramp configs record a source snapshot hash. Their `git_commit_sha`
is null because the local CoReSet source is not tracked by the enclosing Git
repository; the recorded `git_head_sha` alone does not identify the executed
source. Raw per-context metrics, checkpoint and training curves remain in
their ignored local run folders and were not uploaded.
