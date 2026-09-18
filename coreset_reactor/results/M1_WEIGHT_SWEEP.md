# CoReSet-Reactor M1: descriptor-block weight pilot

Only the four explicitly listed groups are summarized here. Training seed
`20260918`, 600 steps, batch size 2, `eval_seed=1234`, and the identical
64-context VAL subset are held fixed. Apart from the weight, run configs match.
B1 has no descriptor loss even though its shared config stores a default
descriptor weight. `--descriptor-cover` scales the **entire** descriptor block,
not just coverage:

`L = L_paired + 0.1 L_SoftDTW + w_d (L_cover + 0.5 L_valid + 0.05 L_load)`.

| Arm / `w_d` | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | GT coverage distance ↓ | prediction-validity distance ↓ | candidate utilization ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 / none | 0.688476 | 88.613668 | 0.007848 | 0.032886 | 1.246347 | 0.999317 | 0.554688 |
| B3 / 1.0 | 0.379560 | 88.008084 | 0.066818 | 0.044599 | 0.997448 | 0.700421 | 0.801562 |
| B3 / 0.5 | 0.499706 | 84.792324 | 0.047400 | 0.039334 | 1.022879 | 0.726507 | 0.775000 |
| B3 / 0.25 | 0.619862 | 83.015613 | 0.030158 | 0.036473 | 1.054832 | 0.763596 | 0.720312 |

Reducing the descriptor-block weight restores some FRC while keeping FRD,
FRDiv and descriptor coverage better than B1 in this seed. Neither 0.5 nor
0.25 reaches B1 FRC, so neither satisfies all four strict M1 criteria.
The 0.5 and 0.25 rows are **one-seed pilots**, not multi-seed confirmation.
Results for other weights are intentionally outside this requested export.

| Arm | Training compute samples/s ↑ | Mean model-forward latency per 10-reaction set (ms) ↓ |
|---|---:|---:|
| B1 | 24.871 | 1.180 |
| B3 / 1.0 | 21.640 | 1.178 |
| B3 / 0.5 | 21.355 | 1.626 |
| B3 / 0.25 | 22.454 | 1.653 |

Training throughput excludes file I/O, CPU preprocessing and host-to-device
transfer. Forward latency excludes target post-processing and metric
computation. These measurements are not end-to-end throughput benchmarks;
the speed differences should not be treated as controlled performance effects.

All arms use the same 172,122-parameter model and corrected valid-prefix and
AU-rounding evaluation contract. The run configs did not capture a
`git_commit_sha`; exact per-run SHA is unavailable. See
`M1_MAIN_RESULTS.md` for the source-provenance limitation. No raw report,
checkpoint, dataset, per-context metric or sample identifier is published.
