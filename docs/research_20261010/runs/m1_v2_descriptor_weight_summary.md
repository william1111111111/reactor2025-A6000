> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# CoReSet-Reactor M1 descriptor-weight follow-up (local only)

The original three-seed, 600-step/64-context VAL pilot with descriptor-cover
weight 1.0 improved B3 diversity and descriptor coverage but substantially
reduced FRC versus the same-seed B1 baseline. After updating `diagnostics.py`
to use valid source prefixes for the model, SoftDTW and descriptors, post-hoc
gradient probes on the three final B3 checkpoints found descriptor/paired
gradient-norm median 2.17 (14 of 15 probes above 1). Two probes included
short sources. This is a final-checkpoint sensitivity measurement, not a
history of training gradients or causal proof of the FRC loss.

All weight variants below use the same initial model seed, 600-step source
schedule, 64 VAL contexts, v2 full-raw-GT descriptor cache and evaluation
contract as B1 seed 20260918. Only B3's descriptor-cover coefficient changes;
the paired, temporal, validity and load coefficients remain fixed.

| Arm / descriptor-cover weight | FRC ↑ | exact FRD ↓ | FRDiv ↑ | GT coverage distance ↓ |
|---|---:|---:|---:|---:|
| B1 / no descriptor loss | 0.688476 | 88.613668 | 0.007848 | 1.246347 |
| B3 / 1.0 | 0.379560 | 88.008084 | 0.066818 | 0.997448 |
| B3 / 0.5 | 0.499706 | 84.792324 | 0.047400 | 1.022879 |
| B3 / 0.25 | 0.619862 | 83.015613 | 0.030158 | 1.054832 |
| B3 / 0.125 | 0.680634 | 84.252299 | 0.019896 | 1.093196 |
| B3 / 0.10 | 0.690499 | 84.664277 | 0.017054 | 1.113882 |

Weight 0.10 passes all four strict within-seed criteria on seed 20260918,
but the FRC margin is only +0.002023. The same weight was therefore
replicated on the two other original seeds, with B1 held fixed:

| Seed | ΔFRC | Δexact FRD | ΔFRDiv | ΔGT coverage distance | All four pass? |
|---|---:|---:|---:|---:|---|
| 20260918 | +0.002023 | -3.949391 | +0.009206 | -0.132464 | yes |
| 20260919 | -0.014908 | -4.301017 | +0.009731 | -0.121807 | no |
| 20260920 | +0.012398 | -5.428149 | +0.006703 | -0.130238 | yes |
| Mean | -0.000162 | -4.559519 | +0.008547 | -0.128170 | no robust zero-slack pass |

Interpretation: reducing descriptor weight markedly restores FRC while
retaining improvements in exact FRD, FRDiv and GT-descriptor coverage. The
0.10 setting is near the quality boundary, but fails strict FRC non-degradation
in one of three seeds. This is a promising tradeoff, not a confirmed M1 win.
The weight was selected using the same 64-context VAL set, so further tuning
there would introduce selection bias. A separately frozen held-out VAL subset
or a predeclared confirmation protocol is needed before making a stronger
claim. This remains a development pilot, not the full official TEST score.

No adaptive sampling, global-context model change, overgeneration or M2
training was started. Checkpoints, per-context metrics and this report remain
local-only and are not intended for GitHub.
