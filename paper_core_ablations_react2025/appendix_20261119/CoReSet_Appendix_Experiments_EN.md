# Supplementary Experiments for CoReSet

This appendix reports completed supplementary experiments on REACT2025. It separates the newly retrained reference-organization study from the existing fixed-teacher distillation study. All numerical tables are populated from saved result files; no new training or evaluation is performed during document assembly.

## A. Evaluation Protocol and Statistical Reporting

Unless otherwise stated, evaluation uses all 1,142 REACT2025 validation contexts: 571 in each direction, grouped into 20 sessions. Each model generates Q = 10 AFRs. Full-duration sequences are reconstructed from 750-frame windows with stride 375 and power-4 center-weighted overlap fusion. Network inference uses FP32 with TF32 disabled. Reference-based quality scores use the fixed AU-rounding and query-0 postprocessing recipe, while set-structure metrics use continuous outputs before these operations.

Means ± SD refer to the mean and sample standard deviation across three student training seeds. Confidence intervals instead use paired session-cluster bootstrap with 4,000 resamples and bootstrap seed 20261113. Both directions from a session are resampled together. These are unadjusted, metricwise 95% CIs and do not estimate training-seed uncertainty. CIs below are reproduced from each experiment’s saved bootstrap summary, rather than recomputed during assembly.

### A.1. Metric definitions

- FRC ↑ and FRD ↓ are reference-matching metrics; FRDiv ↑ measures within-set output diversity; FRVar ↑ measures temporal variation. FRSyn ↓ is stored under the key `TLCC` in the result files.
- PairRelErr ↓ is the mean squared error between teacher and student pairwise difference vectors, averaged over AU/VA/EX after scaling by fixed TRAIN-derived group scales. It is not normalized by each teacher bank’s energy.
- Spearman ↑ compares the ranks of the 45 unordered pairwise distances in the teacher and student sets; the global distances are full-duration, 25-channel mean squared trajectory distances. Outputs remain aligned by expert identity.
- CoverageDist ↓ averages, over reference AFRs, the minimum exact DTW distance to any generated AFR.
- Diversity retention below is the ratio of the mean official student FRDiv to the mean official teacher FRDiv, not a mean of per-context ratios.

For a DTW matrix D with generated outputs in rows and references in columns,

$$\mathrm{CoverageDist}=rac{1}{R}\sum_{r=1}^{R}\min_q D_{q,r},\qquad \mathrm{FRD}=\sum_{q=1}^{Q}\min_r D_{q,r}.$$

SupportDist equals FRD / 10 for Q = 10 and is not separate evidence. Likewise, the corresponding pairwise-difference MSE is 20/9 times centered-residual MSE; the two are not independent mechanism tests. Undefined distance correlations are treated as missing rather than zero.

## B. Reference Selection × Expert Assignment

The four cells are RR (random selection/random assignment), RH (random/Hungarian), FR (farthest-sum/random), and FH (farthest-sum/Hungarian). Each bank contains the same frozen paired teacher at q = 0 and nine newly trained specialists. Specialist initialization, source order, architecture, optimizer batch size 32, seed 1, and 600-update budgets are matched. Random organization is fixed before training rather than reshuffled each epoch. Target-specific reference learning is retained in every cell.

Each student is trained for four epochs and 8,820 optimizer updates with batch size 4, using identical initialization and the same 8,820 TRAIN views from 3,320 source contexts. All four cells use identical activation checkpointing and graph batch chunks of 16. This changes dropout draw order relative to the older recipe, so FH here is not an exact replay of the legacy A0 checkpoint. Comparisons in this section are within the freshly matched four-cell study.

**Table S1. Teacher ensembles: full validation quality metrics.**

| ID | Selection | Assignment | FRC ↑ | FRD ↓ | FRDiv ↑ | FRVar ↑ | FRSyn ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RR | Random | Random | 1.530767 | 126.987847 | 0.032743 | 0.042741 | 45.853765 |
| RH | Random | Hungarian | 1.468984 | 126.409614 | 0.072647 | 0.042651 | 45.853765 |
| FR | Farthest-sum | Random | 1.476223 | 129.863372 | 0.044167 | 0.046862 | 45.853765 |
| FH | Farthest-sum | Hungarian | 1.047369 | 133.137275 | 0.189570 | 0.057323 | 45.853765 |

**Table S2. Distilled students: full validation quality metrics.**

| ID | Selection | Assignment | FRC ↑ | FRD ↓ | FRDiv ↑ | FRVar ↑ | FRSyn ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RR | Random | Random | 1.547885 | 128.780203 | 0.023506 | 0.043103 | 45.748687 |
| RH | Random | Hungarian | 1.479229 | 128.864954 | 0.065621 | 0.042045 | 45.844133 |
| FR | Farthest-sum | Random | 1.504418 | 131.724147 | 0.032673 | 0.047617 | 45.982487 |
| FH | Farthest-sum | Hungarian | 1.097915 | 133.977783 | 0.177803 | 0.056571 | 45.954466 |

**Table S3. Student set-structure agreement and descriptive diversity retention.**

| ID | PairRelErr ↓ | Spearman ↑ | Student / teacher FRDiv |
| --- | --- | --- | --- |
| RR | 0.028047 | 0.749589 | 71.79% |
| RH | 0.049089 | 0.930419 | 90.33% |
| FR | 0.038460 | 0.813769 | 73.98% |
| FH | 0.164875 | 0.928972 | 93.79% |

### B.1. Main effects and interactions

The selection main effect is (FR + FH − RR − RH)/2; the assignment main effect is (RH + FH − RR − FR)/2; the interaction is FH − FR − RH + RR. Positive changes are favorable for FRC, FRDiv, and Spearman; negative changes are favorable for FRD and PairRelErr.

**Table S4. Predeclared factorial contrasts with paired 95% CIs.**

| Output | Metric | Contrast | Difference | 95% CI |
| --- | --- | --- | --- | --- |
| teacher | FRC ↑ | Farthest-sum − Random | -0.238079 | [-0.280359, -0.146306] |
| teacher | FRC ↑ | Hungarian − Random | -0.245319 | [-0.290021, -0.158646] |
| teacher | FRC ↑ | FH − FR − RH + RR | -0.367071 | [-0.421456, -0.243061] |
| teacher | FRD ↓ | Farthest-sum − Random | +4.801593 | [+1.337534, +9.682591] |
| teacher | FRD ↓ | Hungarian − Random | +1.347835 | [-0.706680, +4.282098] |
| teacher | FRD ↓ | FH − FR − RH + RR | +3.852136 | [-0.059312, +8.779646] |
| teacher | FRDiv ↑ | Farthest-sum − Random | +0.064173 | [+0.061236, +0.067336] |
| teacher | FRDiv ↑ | Hungarian − Random | +0.092654 | [+0.086703, +0.098263] |
| teacher | FRDiv ↑ | FH − FR − RH + RR | +0.105500 | [+0.101878, +0.109502] |
| student | FRC ↑ | Farthest-sum − Random | -0.212391 | [-0.257329, -0.124620] |
| student | FRC ↑ | Hungarian − Random | -0.237579 | [-0.285585, -0.148270] |
| student | FRC ↑ | FH − FR − RH + RR | -0.337848 | [-0.392343, -0.221164] |
| student | FRD ↓ | Farthest-sum − Random | +4.028386 | [+0.817424, +8.427932] |
| student | FRD ↓ | Hungarian − Random | +1.169194 | [-0.596706, +3.606923] |
| student | FRD ↓ | FH − FR − RH + RR | +2.168884 | [-1.645926, +6.925833] |
| student | FRDiv ↑ | Farthest-sum − Random | +0.060674 | [+0.058376, +0.062952] |
| student | FRDiv ↑ | Hungarian − Random | +0.093622 | [+0.087953, +0.098906] |
| student | FRDiv ↑ | FH − FR − RH + RR | +0.103015 | [+0.099063, +0.107414] |
| student | PairRelErr ↓ | Farthest-sum − Random | +0.063099 | [+0.060063, +0.070280] |
| student | PairRelErr ↓ | Hungarian − Random | +0.073729 | [+0.070040, +0.080571] |
| student | PairRelErr ↓ | FH − FR − RH + RR | +0.105373 | [+0.098706, +0.118614] |
| student | Spearman ↑ | Farthest-sum − Random | +0.031366 | [+0.015545, +0.039378] |
| student | Spearman ↑ | Hungarian − Random | +0.148017 | [+0.132886, +0.165617] |
| student | Spearman ↑ | FH − FR − RH + RR | -0.065628 | [-0.083450, -0.052800] |

**Table S5. Simple effects on student FRDiv.**

| Contrast | Difference | 95% CI |
| --- | --- | --- |
| FR − RR | +0.009167 | [+0.008521, +0.009484] |
| FH − RH | +0.112182 | [+0.107879, +0.116664] |
| RH − RR | +0.042115 | [+0.038112, +0.045543] |
| FH − FR | +0.145130 | [+0.137551, +0.152433] |

Both components increase teacher and student diversity, with a positive student-FRDiv interaction of 0.103015 (95% CI [0.099063, 0.107414]). FH reaches student FRDiv 0.177803 and retains 93.79% of its teacher FRDiv. These gains involve a trade-off: FH has lower FRC and higher FRD point estimates than RR. The farthest-sum FRD main-effect CI is above zero; the Hungarian FRD main-effect and FRD interaction CIs include zero. Therefore, favorable diversity interaction does not establish a favorable interaction across all quality metrics.

FH has the highest PairRelErr, so this study does not show a PairRelErr reduction. Each cell is evaluated against its own teacher bank, whose spread and targets differ. Hungarian yields higher distance-rank Spearman under both selection rules, but improved rank agreement does not imply lower error for every aspect of geometry. This is a one-training-seed factorial study.

## C. Three-Seed Main-Metric Comparison

A1 uses trajectory-only knowledge distillation; A0 adds centered-difference supervision. The same frozen teacher bank is used for all three student seeds. Within each seed, the paired methods match initialization, source order, architecture, velocity supervision, and the 8,820-update budget. These are legacy fixed-bank checkpoints, not the freshly trained cells in Section B.

**Table S6. Absolute validation scores for each student seed.**

| Seed | Method | FRC ↑ | FRD ↓ | FRDiv ↑ | FRVar ↑ | FRSyn ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | A1 | 1.101718 | 136.159593 | 0.179193 | 0.058117 | 45.803853 |
| 1 | A0 | 1.101366 | 134.225506 | 0.178959 | 0.057093 | 46.035026 |
| 2 | A1 | 1.100051 | 136.400236 | 0.179000 | 0.057045 | 45.735552 |
| 2 | A0 | 1.094219 | 135.542083 | 0.181248 | 0.057255 | 46.471103 |
| 3 | A1 | 1.094508 | 136.217732 | 0.179820 | 0.057730 | 46.224168 |
| 3 | A0 | 1.089029 | 134.389843 | 0.178394 | 0.056475 | 46.175131 |

**Table S7. Mean ± sample SD across the three student seeds.**

| Method | FRC ↑ | FRD ↓ | FRDiv ↑ | FRVar ↑ | FRSyn ↓ |
| --- | --- | --- | --- | --- | --- |
| A1 | 1.098759 ± 0.003775 | 136.259187 ± 0.125563 | 0.179338 ± 0.000429 | 0.057631 ± 0.000543 | 45.921191 ± 0.264599 |
| A0 | 1.094871 ± 0.006194 | 134.719144 ± 0.717407 | 0.179534 ± 0.001511 | 0.056941 ± 0.000412 | 46.227087 ± 0.222633 |

**Table S8. Within-seed paired changes, A0 − A1.**

| Seed | Metric | Difference | 95% CI |
| --- | --- | --- | --- |
| 1 | FRC ↑ | -0.000352 | [-0.006446, +0.007650] |
| 1 | FRD ↓ | -1.934087 | [-2.332308, -1.278786] |
| 1 | FRDiv ↑ | -0.000234 | [-0.000746, +0.000231] |
| 1 | FRVar ↑ | -0.001024 | [-0.001345, -0.000576] |
| 1 | FRSyn ↓ | +0.231173 | [+0.051330, +0.508012] |
| 2 | FRC ↑ | -0.005832 | [-0.008597, -0.001560] |
| 2 | FRD ↓ | -0.858153 | [-1.323387, -0.349692] |
| 2 | FRDiv ↑ | +0.002248 | [+0.001226, +0.002837] |
| 2 | FRVar ↑ | +0.000210 | [-0.000010, +0.000486] |
| 2 | FRSyn ↓ | +0.735552 | [+0.434421, +1.169788] |
| 3 | FRC ↑ | -0.005479 | [-0.010463, -0.000970] |
| 3 | FRD ↓ | -1.827889 | [-2.339571, -1.188948] |
| 3 | FRDiv ↑ | -0.001426 | [-0.002491, -0.000276] |
| 3 | FRVar ↑ | -0.001255 | [-0.001391, -0.001126] |
| 3 | FRSyn ↓ | -0.049037 | [-0.391185, +0.215925] |

A0 lowers FRD in all three seeds, and every corresponding FRD CI is below zero. FRC point estimates are lower in all three seeds; the seed-1 CI includes zero, whereas the seed-2 and seed-3 CIs are below zero. FRDiv, FRVar, and FRSyn do not show uniform improvement across seeds. The evidence supports a repeatable FRD benefit under this fixed teacher bank rather than uniform superiority across metrics.

## D. Validation Set-Structure Preservation and Reference Coverage

This section extends the main-paper set-preservation table with seedwise values, sample SDs, and paired CIs. All set metrics use unrounded, full-duration soft trajectories and preserve expert identities. The reference pool contains all available opposite-role AFRs in the same session.

**Table S9. Seedwise set metrics and coverage.**

| Seed | Method | PairRelErr ↓ | Spearman ↑ | Distance MAE ↓ | CoverageDist ↓ |
| --- | --- | --- | --- | --- | --- |
| 1 | A1 | 0.189633 | 0.922715 | 0.013183 | 15.715697 |
| 1 | A0 | 0.164420 | 0.928376 | 0.012698 | 15.512427 |
| 2 | A1 | 0.180098 | 0.928002 | 0.012501 | 15.744527 |
| 2 | A0 | 0.158839 | 0.934213 | 0.011678 | 15.598892 |
| 3 | A1 | 0.184902 | 0.922404 | 0.013261 | 15.729969 |
| 3 | A0 | 0.163372 | 0.929745 | 0.012347 | 15.492377 |

**Table S10. Three-seed mean ± sample SD for set metrics and coverage.**

| Method | PairRelErr ↓ | Spearman ↑ | Distance MAE ↓ | CoverageDist ↓ |
| --- | --- | --- | --- | --- |
| A1 | 0.184878 ± 0.004768 | 0.924374 ± 0.003146 | 0.012981 ± 0.000418 | 15.730064 ± 0.014415 |
| A0 | 0.162210 ± 0.002966 | 0.930778 ± 0.003052 | 0.012241 ± 0.000518 | 15.534565 ± 0.056603 |

**Table S11. Paired changes in set metrics and coverage, A0 − A1.**

| Seed | Metric | Difference | 95% CI |
| --- | --- | --- | --- |
| 1 | PairRelErr ↓ | -0.025214 | [-0.031261, -0.020313] |
| 1 | Spearman ↑ | +0.005660 | [+0.001779, +0.007586] |
| 1 | Distance MAE ↓ | -0.000485 | [-0.001100, +0.000004] |
| 1 | CoverageDist ↓ | -0.203270 | [-0.220136, -0.173050] |
| 2 | PairRelErr ↓ | -0.021259 | [-0.025027, -0.018887] |
| 2 | Spearman ↑ | +0.006211 | [+0.004772, +0.007714] |
| 2 | Distance MAE ↓ | -0.000823 | [-0.000994, -0.000595] |
| 2 | CoverageDist ↓ | -0.145635 | [-0.179981, -0.097631] |
| 3 | PairRelErr ↓ | -0.021530 | [-0.025314, -0.019559] |
| 3 | Spearman ↑ | +0.007341 | [+0.005228, +0.008630] |
| 3 | Distance MAE ↓ | -0.000914 | [-0.001236, -0.000492] |
| 3 | CoverageDist ↓ | -0.237591 | [-0.275854, -0.206639] |

Comparing three-seed means, A0 reduces TRAIN-scaled PairRelErr by 12.26%. PairRelErr and CoverageDist decrease, and distance-rank Spearman increases, with favorable CIs in each seed. This supports teacher-set relationship preservation and nearest-reference coverage under the stated reference protocol; it does not establish coverage of more verified same-stimulus human reactions.

**Table S12. Validation channel-group diagnostics: three-seed mean ± sample SD.**

| Group | Method | PairRelErr ↓ | Spearman ↑ | Distance MAE ↓ |
| --- | --- | --- | --- | --- |
| AU | A1 | 0.183761 ± 0.002524 | 0.915368 ± 0.002940 | 0.015774 ± 0.000527 |
| AU | A0 | 0.182363 ± 0.001018 | 0.919470 ± 0.002802 | 0.014985 ± 0.000746 |
| VA | A1 | 0.193921 ± 0.007361 | 0.890309 ± 0.005112 | 0.025064 ± 0.000913 |
| VA | A0 | 0.142044 ± 0.004118 | 0.920826 ± 0.003174 | 0.021015 ± 0.000501 |
| EX | A1 | 0.176951 ± 0.007682 | 0.892204 ± 0.010826 | 0.011089 ± 0.000419 |
| EX | A0 | 0.162225 ± 0.006937 | 0.905893 ± 0.002823 | 0.010541 ± 0.000084 |

**Table S13. Exploratory per-context association between paired changes in PairRelErr and CoverageDist.**

| Seed | Pearson of paired changes | Spearman of paired changes |
| --- | --- | --- |
| 1 | 0.012994 | 0.153667 |
| 2 | 0.037740 | -0.044663 |
| 3 | 0.140360 | 0.156326 |

The weak and varying per-context associations are exploratory; simultaneous improvements in aggregate metrics do not isolate a causal link from improved set-structure agreement to improved reference coverage.

## E. Additional Evaluation of the Uncentered A3 Control

A3 uses uncentered group-weighted MSE and is reported at seed 1. It is a fixed-coefficient control; it should not be described as a newly isolated scalar-distance or direction-only ablation. The comparisons below use matching seed-1 outputs.

**Table S14. Seed-1 quality metrics for A1, A3, and A0.**

| Method | FRC ↑ | FRD ↓ | FRDiv ↑ | FRVar ↑ | FRSyn ↓ |
| --- | --- | --- | --- | --- | --- |
| A1 | 1.101718 | 136.159593 | 0.179193 | 0.058117 | 45.803853 |
| A3 | 1.088426 | 132.571447 | 0.178139 | 0.056107 | 46.125219 |
| A0 | 1.101366 | 134.225506 | 0.178959 | 0.057093 | 46.035026 |

**Table S15. Seed-1 structure and coverage metrics for A1, A3, and A0.**

| Method | PairRelErr ↓ | Spearman ↑ | Distance MAE ↓ | CoverageDist ↓ |
| --- | --- | --- | --- | --- |
| A1 | 0.189633 | 0.922715 | 0.013183 | 15.715697 |
| A3 | 0.170328 | 0.922506 | 0.012914 | 15.383815 |
| A0 | 0.164420 | 0.928376 | 0.012698 | 15.512427 |

**Table S16. Matched seed-1 changes, A0 − A3.**

| Metric | Difference | 95% CI |
| --- | --- | --- |
| FRC ↑ | +0.012940 | [+0.007895, +0.016882] |
| FRD ↓ | +1.654058 | [+1.212257, +1.950571] |
| FRDiv ↑ | +0.000819 | [+0.000344, +0.001224] |
| PairRelErr ↓ | -0.005909 | [-0.006342, -0.005019] |
| Spearman ↑ | +0.005870 | [+0.003179, +0.008349] |
| Distance MAE ↓ | -0.000216 | [-0.000357, -0.000054] |
| CoverageDist ↓ | +0.128612 | [+0.100912, +0.154718] |

A0 has lower PairRelErr and higher distance-rank Spearman than A3, whereas A3 has lower FRD and CoverageDist. Thus, the centered objective does not dominate the uncentered control on all reference-matching metrics.

## F. Balanced Optimal Transport on the Same DTW Matrices

We additionally solve unregularized balanced optimal transport with uniform generated-output and reference masses on the same full-duration DTW matrices. This differs from nearest-neighbor coverage by constraining both marginals.

$$\min_{\pi\ge0}\sum_{q,r}\pi_{q,r}D_{q,r},\qquad \sum_r\pi_{q,r}=1/Q,\quad \sum_q\pi_{q,r}=1/R.$$

**Table S17. Balanced OT cost on all 1,142 validation contexts; lower is better.**

| Method / seed | Balanced OT ↓ |
| --- | --- |
| A1_seed1 | 17.053982 |
| A0_seed1 | 17.027952 |
| A3_seed1 | 16.872887 |
| A1_seed2 | 17.204111 |
| A0_seed2 | 17.091290 |
| A1_seed3 | 17.113461 |
| A0_seed3 | 16.940623 |

**Table S18. Three-seed balanced OT mean ± sample SD.**

| Method | Balanced OT ↓ |
| --- | --- |
| A1 | 17.123851 ± 0.075602 |
| A0 | 17.019955 ± 0.075651 |

**Table S19. Paired balanced OT differences.**

| Comparison | Difference | 95% CI |
| --- | --- | --- |
| A0_seed1 − A1_seed1 | -0.026030 | [-0.072061, +0.071822] |
| A0_seed2 − A1_seed2 | -0.112821 | [-0.167555, -0.055612] |
| A0_seed3 − A1_seed3 | -0.172837 | [-0.192946, -0.145173] |
| A0_seed1 − A3_seed1 | +0.155065 | [+0.102556, +0.197537] |
| A3_seed1 − A1_seed1 | -0.181095 | [-0.230971, -0.081385] |

The A0−A1 CI includes zero in seed 1 and is below zero in seeds 2 and 3. A3 has lower OT cost than A0 in seed 1. Hence, evidence for uniform-mass whole-set matching is weaker than the nearest-neighbor CoverageDist evidence and does not support uniform superiority over A3.

## G. Model Size and Inference Efficiency

Measurements use an NVIDIA RTX A6000, PyTorch 2.1.0+cu121, FP32, TF32 disabled, batch size 1, T = 750 frames, and Q = 10 outputs. Four fixed real inputs are used, with ten warm-up passes and thirty timed repeats per model per input (120 timed observations per model). Timings are interleaved and synchronized. Inputs are preloaded on the GPU; feature extraction, I/O, host/device transfer, overlap fusion, AU postprocessing, and metric computation are excluded. The ten teachers run sequentially; the student decodes ten slots together. Peak allocated memory is measured in separate one-arm processes. The server is shared rather than exclusively reserved.

**Table S20. Network-forward efficiency for generating ten AFRs.**

| Model | Parameters (M) | Dense FLOPs (G) | Mean (ms) | Median (ms) | P95 (ms) | Peak allocated (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| 10-teacher ensemble | 37.201290 | 116.390400 | 36.965 | 36.983 | 37.051 | 220.84 |
| Shared student | 3.803201 | 47.593446 | 19.611 | 19.611 | 19.721 | 632.52 |

The student reduces deployment parameters by 89.78% and counted dense FLOPs by 59.11%, and achieves a 1.88× speedup in mean network-forward time. Its measured peak allocated memory is higher, not lower. Dense FLOPs count two operations per multiply-accumulate and include all linear projections and QK/AV attention products for all ten decodes; they exclude elementwise, nonlinear, and normalization operators. These results are not end-to-end system latency or online-response measurements.

## H. Training-View Difference Reconstruction Diagnostics

The following diagnostics use all 8,820 cached TRAIN views at seed 1, with unrounded soft outputs and identical valid-frame masks. They measure fit to the training teacher set and must be distinguished from the validation results in Section D.

**Table S21. TRAIN reconstruction diagnostics; negative relative change favors A0.**

| Group | Metric | A1 | A0 | (A0 − A1) / A1 |
| --- | --- | --- | --- | --- |
| AU | centered_residual_mse | 0.009519 | 0.009451 | -0.72% |
| AU | pair_distance_mae | 0.026163 | 0.026432 | +1.03% |
| VA | centered_residual_mse | 0.010951 | 0.007442 | -32.05% |
| VA | pair_distance_mae | 0.040255 | 0.033572 | -16.60% |
| EX | centered_residual_mse | 0.006577 | 0.005908 | -10.17% |
| EX | pair_distance_mae | 0.024834 | 0.023038 | -7.23% |

The training diagnostic uses root-mean-squared trajectory distances for pair_distance_mae. This is a different distance unit from the mean-squared trajectory Distance MAE in Section D; their numerical magnitudes should not be compared directly.

Centered-residual MSE decreases in all three channel groups. Pair-distance MAE improves for VA and expression, but increases slightly for AU. This supports centered-difference reconstruction rather than an improvement in every possible relation metric.

## I. Qualitative Figures and Their Interpretation

**Figure S1.** Teacher and student quality metrics in the matched four-cell study. [PDF](figures/S1_four_cell_metrics.pdf) · [PNG](figures/S1_four_cell_metrics.png)

**Figure S2.** Main effects and interaction with session-bootstrap CIs. [PDF](figures/S2_factorial_effects.pdf) · [PNG](figures/S2_factorial_effects.png)

**Figure S3.** A2 and CoReSet outputs for the same frozen validation input; selected slots are illustrative. [PDF](figures/S3_A2_CoReSet_trajectories.pdf) · [PNG](figures/S3_A2_CoReSet_trajectories.png)

**Figure S4.** Within-set difference reconstruction on a selected training case; aggregate TRAIN metrics are reported separately in Section H. [PDF](figures/S4_training_difference_transfer.pdf) · [PNG](figures/S4_training_difference_transfer.png)

**Figure S5.** Selected same-session references and generated reactions shown at relative time; recordings are not asserted to be synchronized. [PDF](figures/S5_reference_and_generated_set.pdf) · [PNG](figures/S5_reference_and_generated_set.png)

**Figure S6.** Rendered A2/CoReSet reactions with a shared identity and frozen rendering pipeline. [PDF](figures/S6_rendered_reaction_sequences.pdf) · [PNG](figures/S6_rendered_reaction_sequences.png)

The validation illustration uses the fixed case with manifest index 65 and length 2,162 frames. The trajectory comparison and reference-matching illustration use different selected slot subsets and should not be treated as random samples of all outputs. In the reference illustration, displayed references are chosen by a VA-based one-to-one match, exclusion of the largest-error pair, and selection of a diverse triplet; this post hoc display selection is separate from the expert-aligned quantitative evaluation. The rendering pipeline maps actual model AFR25 predictions through the same frozen TRAIN-trained AFR25-to-FaceVerse58 bridge and renderer, with the same identity image for both methods. The model itself predicts AFR attributes, not face pixels. No human perceptual study is reported.

## J. Scope and Reproducibility

The four-cell reference-organization study has one newly trained seed per cell; the A0/A1 distillation study has three student seeds with one fixed teacher bank. The latter does not establish robustness to teacher-bank retraining. VAL was used in prior development. Same-session opposite-role references are not verified independent reactions to an identical stimulus, so coverage should not be interpreted as direct human-reaction coverage for the same stimulus. The frozen numerical sources, inputs, aggregate files, and figure assets used here are listed with SHA-256 hashes in `source_manifest.json`.

The reported results support diversity gains from reference organization, improved same-bank difference reconstruction and validation FRD with centered-difference distillation, and a more compact student with faster network forward passes. They also expose trade-offs in FRC, FRD, absolute cross-bank relation error, and peak allocated memory. These trade-offs are part of the results, not omitted exceptions.
