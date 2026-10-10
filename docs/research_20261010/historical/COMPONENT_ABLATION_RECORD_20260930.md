> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Teacher and student component ablation record

Recorded 2026-09-30. This consolidates the completed REACT2024/2025 student
loss-deletion studies and the corrected REACT2025 key-component controls.
It records observed tradeoffs, not a claim that every component is indispensable.
All old checkpoints and negative/frontier results remain retained.

## Main conclusion

The evidence supports different roles for the retained components:

- Diverse fixed GT targets supply distinct training reactions. Historical
  random-vs-Max-FRDiv teacher evidence is indexed below; this is not yet a
  random-target deletion from the entire final method.
- A mode-conditioned reference prior preserves diversity during multi-reference
  teacher training, at a cost in FRC compared with uniform references.
- Teacher-output KD substantially improves student FRC relative to direct
  fixed-GT supervision under the matched minimal-loss protocol. Direct GT has
  better FRD and slightly higher diversity, so KD is not a Pareto improvement.
- Centered-residual KD primarily improves student FRD in the measured settings;
  it is not required to obtain high FRDiv.
- The tested velocity term shows little incremental benefit. Removing it is
  the recommended simplification, not a statistically established universal rule.

推荐的论文表述：各组件承担不同职责，共同形成质量与多样性的平衡。
不写成“模块缺一不可”，也不把中心化差异蒸馏写成多样性的唯一来源。

## Evaluation and matching contract

All tables use full VAL, all opposite-role same-session GT, deterministic
target post-processing, AU rounding and exact DTW-FRD. REACT2025 has 571
contexts; converted REACT2024 has 1124. This is not an official sampled-ten
TEST reproduction, and converted REACT2024 all-session scoring is not its
native neighbour-GT evaluation protocol. Do not compare these scores directly
with published challenge leaderboards or between datasets as equal difficulty.

Seed is 1. Within each comparison, initialization, source views/order, model
and stage-specific training budget are matched and audited. The backbone is
unchanged ConditionalREGNN with graph=2; these studies do not establish that
graph removal is lossless. Student parameter count is 3,803,201, with ten outputs
and no teachers, GT, inference noise, selector or output rescaling at inference.

Student training uses batch 4, AdamW lr 1e-4, betas (0.9,0.99), weight decay
1e-4, gradient clip 1, BF16 and a ten-epoch cosine schedule. REACT2025 uses
3320 cached TRAIN views / 8300 updates; REACT2024 uses 6340 / 15850 updates.
The REACT2025 teacher prior uses effective count8, while the accepted
REACT2024 teacher uses count4; these are not identical cross-dataset recipes.

## Student loss deletion

Base loss means MSE + (1-CCC). Velocity weight is 0.05 when enabled.
The residual term matches signed slot deviations around the ten-slot mean,
normalized by fixed TRAIN-derived AU/VA/expression scales. It penalizes both
too little and too much separation, rather than repelling predictions.

The residual weight/scales were frozen from each dataset's existing full-model
TRAIN calibration: weight 0.3291488264202468 for REACT2025 and
0.21374727996059517 for REACT2024. They were not recalibrated after deleting
velocity. These numbers are not universal hyperparameters or current gradient
ratios. The minimal-loss arms require no residual calibration.

### REACT2025 full VAL571

| Arm | Velocity | Residual | FRC up | exact FRD down | FRDiv up | FRVar |
|---|---|---|---:|---:|---:|---:|
| Full | Yes | Yes | 1.444718 | 97.540968 | 0.170190 | 0.067888 |
| Minimal | No | No | 1.454643 | 99.714605 | 0.170338 | 0.069951 |
| Velocity only | Yes | No | 1.454373 | 99.752944 | 0.170335 | 0.069890 |
| Residual only | No | Yes | 1.445507 | 97.463281 | 0.170130 | 0.067918 |

Deleting velocity from Full improves FRC by 0.000789 and FRD by 0.077686,
but reduces FRDiv by 0.000060. This is not a strict three-metric win.
Full and Velocity-only are retained completed runs; Minimal and Residual-only
were newly trained. Their matching audits passed.

### REACT2024 full VAL1124

| Arm | Velocity | Residual | FRC up | exact FRD down | FRDiv up | FRVar |
|---|---|---|---:|---:|---:|---:|
| Full | Yes | Yes | 0.655348 | 86.025763 | 0.132445 | 0.058831 |
| Minimal | No | No | 0.656854 | 88.964175 | 0.135939 | 0.060444 |
| Velocity only | Yes | No | 0.655127 | 88.886196 | 0.135477 | 0.060246 |
| Residual only | No | Yes | 0.656394 | 86.033383 | 0.132724 | 0.058911 |

Deleting velocity from Full improves FRC by 0.001046 and FRDiv by 0.000279,
but worsens FRD by 0.007620. This is also not a strict three-metric win.
Full is retained; the other three arms were newly trained.
These are objective deletions: velocity was still computed for logging when
weighted zero, so no training-speed improvement is claimed from its deletion.

## Corrected teacher prior ablation

Both teacher banks use the corrected CCC gradient at constant predictions.
The paired q0 is identical. The nine specialists each train for 600 updates,
batch32, with the same initialization, source/crop schedule, fixed Max-FRDiv
targets, graph2 model, fixed-target MSE, masked velocity and temperature0.05.
Only the reference prior changes from mode-conditioned to uniform. Both banks
were freshly evaluated with the current evaluator; no old uniform-v1 score was
substituted. Each teacher has 3,720,129 parameters; the bank has 37,201,290.

| Teacher prior | FRC up | exact FRD down | FRDiv up | FRVar |
|---|---:|---:|---:|---:|
| Mode-conditioned | 1.370792 | 95.772670 | 0.190244 | 0.069174 |
| Uniform | 2.038626 | 95.410771 | 0.091645 | 0.070391 |

Uniform minus Mode:

| Metric | Mean delta | Session-cluster bootstrap 95% interval |
|---|---:|---|
| FRC | +0.667834 | [+0.528643, +0.801175] |
| exact FRD | -0.361899 | [-2.403709, +1.173301] |
| FRDiv | -0.098598 | [-0.113551, -0.085686] |

All 571 contexts lose diversity without the mode prior; mean FRDiv falls about
51.8%. Uniform references increase FRC and slightly improve mean FRD, although
the FRD interval spans zero. The interpretation is preservation of distinct
teacher behavior, not an across-the-board quality benefit from the prior.

## Direct real GT versus teacher output supervision

Both students use the same initialization and MSE+(1-CCC), with neither
velocity nor residual loss. Exactly the current Max-FRDiv target identities
are used; this is not the historical KMeans/diffusion-legacy direct-GT study.
All 3320 source views were checked, and 300 real-target tensors were checked
bitwise against production teacher dataset preprocessing. Rank0 uses native
paired cropping; ranks1-9 use the current relative-time alignment. Both caches
store targets in FP16 and train against FP32-converted values.

The direct arm reads real trajectories, not teacher predictions. It still
uses the same pretrained teacher0 backbone initialization as KD. Thus this is
an ablation of output distillation, not a teacher-free-from-scratch experiment.

| Student targets | FRC up | exact FRD down | FRDiv up | FRVar |
|---|---:|---:|---:|---:|
| Teacher outputs, Minimal KD | 1.454643 | 99.714605 | 0.170338 | 0.069951 |
| Same fixed real GT | 1.066966 | 97.427886 | 0.177390 | 0.064775 |

Direct minus Minimal KD:

| Metric | Mean delta | Session-cluster bootstrap 95% interval |
|---|---:|---|
| FRC | -0.387677 | [-0.446677, -0.295797] |
| exact FRD | -2.286719 | [-3.971230, -0.767244] |
| FRDiv | +0.007052 | [+0.001741, +0.009516] |

Direct supervision reduces FRC about 26.7%, while improving FRD and diversity.
This supports an FRC benefit from learned teacher outputs, not a necessity of
distillation for producing diverse reactions. Comparing Direct GT with the
Residual-only student additionally changes the loss, so it is an operating-point
comparison, not a single-component ablation.

## Frozen prediction random reference check

Full REACT2025 VAL571, 100 draws per protocol, identical draws across students.
Predictions and processed GT are fixed. Only FRC reference sampling changes;
FRD was not recomputed, and FRDiv is unchanged by GT sampling.

| Student | All-session FRC | Paired plus random9 FRC mean and std | Uniform10 FRC mean and std |
|---|---:|---:|---:|
| Full | 1.444718 | 1.024583 ± 0.006425 | 1.019943 ± 0.007364 |
| Minimal | 1.454643 | 1.027358 ± 0.006522 | 1.022223 ± 0.007567 |
| Velocity only | 1.454373 | 1.026989 ± 0.006513 | 1.021893 ± 0.007578 |
| Residual only | 1.445507 | 1.025062 ± 0.006422 | 1.020433 ± 0.007353 |

This sampling-only diagnostic is not a complete official TEST reproduction.

## Interpretation boundaries and remaining controls

The working simplified student recipe is Base + centered residual, without
velocity. This is a recommendation, not a checkpoint replacement or automatic
promotion. No weights were deleted or overwritten.

All studies are seed1 on repeatedly inspected VAL. Session-cluster bootstrap
intervals describe variation across sessions conditional on that training run;
they do not measure training-seed uncertainty or remove validation-selection
bias. No SOTA, universal indispensability or zero-slack domination claim is made.

Per-stage update budgets are matched, not end-to-end compute: KD also incurs
teacher training and cache generation. Shared-server inference timings are
diagnostic, not controlled speed benchmarks. Complete speed/config records
remain in the run summaries.

Remaining controls include random rather than Max-FRDiv targets under the
entire final teacher recipe, confirmation on independent seeds/held-out data,
and the two key-component controls on REACT2024. These have not been launched
by this documentation update. Historical fixed-reference and graph experiments
must not be silently treated as matched deletions from the final method.

## Evidence and local provenance

Persistent run root: `$RUN_ROOT/`.
Each listed run contains `summary.json`, `experiment.json`, source snapshots,
matching audits, configs, logs, prediction ledgers and retained checkpoints.
The source Git SHA was `c421f509f8cb7997e9341592cae2eb1e721b0e8e`, with local
changes; per-file snapshot hashes are necessary to identify the executed code.

| Run directory | Summary SHA256 |
|---|---|
| `react2024_student_loss_pruning_seed1` | `ac7f7855ad9e9e5116caa3af2e108d4ee356f10e8b15d21ad6db9889da4d77aa` |
| `react2025_student_loss_pruning_seed1` | `cbba1a08ffbf07ca5035dac12b601e8cc4cc4178f544526bc46e1b060d846b9a` |
| `react2025_key_component_ablations_seed1` | `6a42d7536211600e767d8dd560733a370d3189302aa9812481ff7c1581e87e55` |

| Retained checkpoint relative to run root | SHA256 |
|---|---|
| `react2024_student_loss_pruning_seed1/residual/student.pth` | `0c41afe63a4e56ad4ec0f7722870119ee8bf209616e0b23157199f53ddd32eb5` |
| `react2025_student_loss_pruning_seed1/residual/student.pth` | `84df1ea5cd51196747fa662421653ff5c8789324221a52529053a6612e1935b4` |
| `react2025_key_component_ablations_seed1/direct_student/student.pth` | `33c040fea67ec74a45f421b95d23e6e0486aeffcae16d0b4ec5d2c734bc549e5` |

Historical evidence, with its original scope preserved:
[Max-FRDiv teacher pilot](MAXFRDIV_TEACHER_PILOT.md),
[latent-reference teacher study](LATENT_REFERENCE_TEACHERS.md),
[old direct-GT student](DIRECT_GT_STUDENT.md), and
[retained student recipes](STUDENT_RECIPE_BANK.md).

This record contains aggregate results only, not dataset samples, raw per-clip
predictions or weights. Recording it does not authorize a GitHub upload.
