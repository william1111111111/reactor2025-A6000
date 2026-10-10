> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# T1 student: quality-preserving AU refinement

REACT2025 full VAL571; all-session GT; center750; AU rounded; exact rolling FRD

| Method | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar | T1 FRDiv retained |
|---|---:|---:|---:|---:|---:|
| T1_teacher | 1.072215 | 96.813896 | 0.205415 | 0.068097 | 100.0% |
| contrast_epoch10 | 1.121615 | 96.617181 | 0.183968 | 0.065987 | 89.6% |
| control | 1.133005 | 96.705597 | 0.187008 | 0.065888 | 91.0% |
| au_decision | 1.133438 | 96.644603 | 0.186752 | 0.065592 | 90.9% |
| au_logit | 1.132541 | 96.474092 | 0.186153 | 0.065210 | 90.6% |

## Experiment contract

- Frozen starting point: the previous 10-epoch centered-residual student. Every new arm starts from exactly those weights.
- Three arms: unchanged contrast-KD control; contrast KD + confidence-weighted AU decisions; contrast KD + signed AU log-odds contrasts.
- All arms: five more epochs / 4,150 updates, seed 1, batch 4, lr 2e-5 → 1e-6 cosine, BF16, clip norm 1.
- AdamW is reset identically in every arm: betas (0.9,0.99), weight decay 1e-4. This is matched fine-tuning, not an optimizer resume.
- Frozen base loss: MSE + (1-CCC) + 0.05 velocity + 0.2697612303 centered-residual loss, with unchanged TRAIN scales.
- Extra loss weights use 16 fixed TRAIN probes and target 0.25 of the initial primary-loss gradient norm. No VAL labels enter training/calibration.
- AU logit contrast: clip probabilities to [0.001,0.999], subtract the ten-slot logit mean, match detached teacher contrasts with Huber loss.
- Same 3.803M parameters, ten outputs, teacher cache, initial-state hash and observed source-order hash. No inference noise/scaling or teacher calls.
- Each student now has 12,450 total training updates. Comparisons against epoch10 alone are NOT budget-matched; the continuation control is the causal comparator.

## Selection and caution

Highest FRDiv: `control`.
Highest FRDiv satisfying both frozen T1 quality point estimates: `control`.
Highest FRDiv satisfying both frozen parent-student quality point estimates: `au_logit`.
Arms preserving both parent-student quality point estimates: contrast_epoch10, au_logit.
No quality slack is used. Passing the teacher thresholds does not imply passing the stronger parent-student thresholds.
This is a single-seed adaptive VAL study, not proof of statistical equivalence or a TEST result.

## Deltas versus matched continuation control

| Arm | ΔFRC [session bootstrap 95% CI] | ΔFRD [95% CI] | ΔFRDiv [95% CI] |
|---|---|---|---|
| control | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| au_decision | +0.000433 [+0.000145, +0.000744] | -0.060994 [-0.094984, -0.040226] | -0.000256 [-0.000381, -0.000133] |
| au_logit | -0.000464 [-0.002048, +0.001577] | -0.231504 [-0.347619, -0.147667] | -0.000855 [-0.001070, -0.000657] |

Both AU auxiliary objectives fail to improve FRDiv over the equal-budget continuation control.
Any diversity gain versus epoch10 therefore cannot be attributed to these new objectives.

## AU fidelity and diversity components

| Arm | AU FRDiv | VA FRDiv | Expression FRDiv | AU decision disagreement | AU contrast cosine |
|---|---:|---:|---:|---:|---:|
| contrast_epoch10 | 0.153413 | 0.010006 | 0.020550 | 8.84% | 0.839204 |
| control | 0.155444 | 0.010332 | 0.021232 | 8.32% | 0.854543 |
| au_decision | 0.155051 | 0.010357 | 0.021345 | 8.26% | 0.855245 |
| au_logit | 0.154900 | 0.010233 | 0.021021 | 8.29% | 0.858596 |

### Teacher-relative AU errors

These are disagreement rates with teacher predictions, NOT accuracy against real GT. Margin-bin rates pool all valid evaluated frames.

| Arm | Teacher-on missed | Teacher-off falsely activated | Disagreement at teacher margin < 0.05 |
|---|---:|---:|---:|
| contrast_epoch10 | 19.67% | 4.99% | 42.76% |
| control | 18.26% | 4.80% | 42.38% |
| au_decision | 18.25% | 4.72% | 42.33% |
| au_logit | 18.27% | 4.76% | 42.36% |

## Compute and provenance

| Arm | Extra weight | Updates | Train seconds | Updates/s | Diagnostic inference ms |
|---|---:|---:|---:|---:|---:|
| control | 0.000000 | 4150 | 532.4 | 7.80 | 20.00 |
| au_decision | 0.099834 | 4150 | 533.8 | 7.77 | 20.15 |
| au_logit | 0.102627 | 4150 | 531.9 | 7.80 | 20.15 |

No expensive temporal-GT loss or inference selector was added. Selector overhead is zero.
All original/new weights are retained. Per-context metrics, distributions, teacher-margin-bin errors and hashes are saved locally in analysis.json.
Teacher metrics are reused from the preceding parity-verified evaluation. Its 600-step T1 ensemble already sacrifices quality versus T0; do not conflate their quality references.

- control: checkpoint `e15c8219c0442a572630d42e55899106b67d8468f6a5c1a92caa49c372ab3720`, initial state `b4dec995b9fc0254d61648517b2abbab5c248da9b6940adb7125b46e97173bde`, source order `9c5013ce081878d35d52668ebf5496ba839dcb20c2fd6c9c7ccf4e7ec8735a5f`.
- au_decision: checkpoint `1a886a4af1c234fca47fa3d8bf7f64c0e56c0a2db4e0743d57fc851a7b5c78a5`, initial state `b4dec995b9fc0254d61648517b2abbab5c248da9b6940adb7125b46e97173bde`, source order `9c5013ce081878d35d52668ebf5496ba839dcb20c2fd6c9c7ccf4e7ec8735a5f`.
- au_logit: checkpoint `c867910592225a845750c600fd00c92cbcf45e6c012350666bcdf9716565e328`, initial state `b4dec995b9fc0254d61648517b2abbab5c248da9b6940adb7125b46e97173bde`, source order `9c5013ce081878d35d52668ebf5496ba839dcb20c2fd6c9c7ccf4e7ec8735a5f`.
