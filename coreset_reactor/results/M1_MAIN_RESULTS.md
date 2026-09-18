# CoReSet-Reactor M1: B1 versus B3-full

This is a deterministic REACT2025 VAL pilot, not the full official TEST result.
Each arm trained for 600 steps (batch size 2) with the same model initialization,
source schedule, exact-GT draws and seed within a run. B1 uses the paired GT and
17 random same-session temporal GTs. B3-full uses the same expensive targets
plus descriptors of every distinct TRAIN listener recording in that session
(94–101 GTs per session). The v2 descriptor cache uses each raw recording's
entire trajectory. Evaluation uses 64 of 571 VAL contexts selected with
`eval_seed=1234`, ten predictions, the paired GT plus nine deterministic
same-session GTs, AU-rounded predictions, the existing target post-processor
and repository metric functions. All three seeds use the same VAL subset.

| Training seed | Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | GT coverage distance ↓ |
|---:|---|---:|---:|---:|---:|
| 20260918 | B1 | 0.688476 | 88.613668 | 0.007848 | 1.246347 |
| 20260918 | B3-full (weight 1.0) | 0.379560 | 88.008084 | 0.066818 | 0.997448 |
| 20260919 | B1 | 0.793772 | 92.437204 | 0.008688 | 1.239925 |
| 20260919 | B3-full (weight 1.0) | 0.400295 | 86.573902 | 0.066094 | 1.004390 |
| 20260920 | B1 | 0.778911 | 91.851143 | 0.010628 | 1.233422 |
| 20260920 | B3-full (weight 1.0) | 0.353831 | 85.687768 | 0.060211 | 0.998440 |

Mean B3-full minus B1: **FRC -0.375825**, exact FRD -4.210753,
FRDiv +0.055320, GT coverage distance -0.239806. FRVar rises by 0.005978,
prediction-validity distance falls by 0.292855, and candidate utilization
rises by 0.232812. Thus full-strength descriptor supervision improves the
diversity/coverage diagnostics but fails the strict FRC non-degradation
criterion in every seed. This does **not** establish an M1 win or authorize M2.

Same-session listener recordings are weak proxies, not verified reactions to
the identical stimulus. The result is a development-set pilot; no full TEST
score or significance claim is made. Local regression tests: 9 passed; this
repository does not have a recorded GitHub CI run for these experiments.

Provenance limitation: the original run configs did **not** record a
`git_commit_sha`, so an exact per-run SHA is unavailable. Reference source
commits for review are `1b022557528a7d1061e913deea907a47e6fbdc2a`
(corrected training/evaluation) and
`3b96c88ee895bfc512ea5cbb40c149721cb363eb` (diagnostic alignment and
single-arm CLI controls, not a new training objective). These review commits
are not substitutes for an
automatically captured run-time SHA. No checkpoint, dataset, sample identifier,
per-context record or absolute local path is included here.
