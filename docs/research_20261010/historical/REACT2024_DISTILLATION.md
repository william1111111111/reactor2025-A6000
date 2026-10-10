> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# REACT2024 fixed-teacher distillation

Run date: 2026-09-22. Single seed; all 10-epoch training runs and full-VAL
evaluations completed. The official TEST split was not used for model selection.

## Protocol

- Teacher: React2024 Mam-Reactor `query_seed1_final`, epoch 30,
  checkpoint SHA-256 `8625c64bc2f92558478fcf5c96208f68d87a3c09ef85c004ad44c8fdd1c0c627`.
- Teacher inference uses residual and style scales of 3.0, the frozen
  operating point recorded in the existing React2024 evaluation. Re-evaluated
  teacher FRC/S_MSE/FRVar/FRDvs match that saved evaluation within `6e-8`.
- Training teacher cache: all 3,186 TRAIN contexts, 10 raw soft trajectories
  per context, fixed view epoch 0. Cache manifest hash:
  `bb164c54e286d240c1fc51afc4f501aaf383d2be302bcc9609d668506bcbc90b`.
- Student initialization: React2024 anchor epoch 50,
  SHA-256 `53fd5aff68af6725fc8a40fccdd0c5b9b6a9588812cf5ae3e742ab9c2e0404b7`.
- All students use the same 3,803,201-parameter expert-conditioned REGNN,
  seed 1, batch 4, 10 epochs (7,970 updates), AdamW `1e-4`, identical cache,
  initialization and data schedule. Basic-KD uses the trajectory loss only;
  Distance-KD adds teacher slot-distance matching; Direction-KD adds
  centered slot-direction matching. The requested relation-gradient ratio is
  0.25; calibrated weights were 0.755385 and 0.359820 respectively.
- Evaluation: all 1,124 official React2024 VAL directions, the existing
  `neighbour_emotion_val.npy` FRC matching, and the native React2024
  S_MSE/FRVar/FRDvs calculations. AUs are rounded exactly as in the saved
  teacher evaluation.

## Full-VAL results

S_MSE is the benchmark's within-set pairwise prediction MSE (larger means
more spread within the 10 outputs). FRDvs is reported using the native
React2024 implementation; it is not the temporal DTW-FRD metric.

| Model | FRC | S_MSE | FRVar | FRDvs |
|---|---:|---:|---:|---:|
| Mam-Reactor teacher | 0.451339 | 0.101503 | 0.084147 | 0.204071 |
| Basic-KD | 0.482803 | 0.078991 | 0.063534 | 0.147760 |
| Distance-KD | **0.483342** | **0.107258** | 0.070031 | 0.159101 |
| Direction-KD | 0.484768 | 0.078257 | 0.064943 | 0.151331 |

| Student delta vs teacher | ΔFRC | ΔS_MSE | ΔFRVar | ΔFRDvs |
|---|---:|---:|---:|---:|
| Basic-KD | +0.031463 | -0.022512 | -0.020613 | -0.056311 |
| Distance-KD | +0.032003 | **+0.005755** | -0.014117 | -0.044969 |
| Direction-KD | **+0.033429** | -0.023246 | -0.019204 | -0.052740 |

The most promising balanced result is Distance-KD: it improves FRC and
slightly raises within-set S_MSE over the teacher. Basic-KD and Direction-KD
have marginally higher FRC, but their within-set S_MSE is lower. Thus the
distance relation term appears useful for retaining teacher-set spread; it
does not improve every variability statistic. This is a one-seed result, not
yet a multi-seed claim.

## Limits and artifacts

- Exact temporal DTW-FRD was not run for these VAL predictions; this report
  covers the native fast React2024 metrics above.
- Inference timing was collected on shared GPUs and is not a controlled
  latency comparison.
- Checkpoints, per-run configs, provenance, logs and JSON evaluations are on
  the server under
  `$LOCAL_PATH/coreset_distillation_react2024`.
- Source scripts added for this experiment are
  `react2024_teacher_cache.py` and `eval_react2024_distilled_student.py`;
  the training entry point is `train_expert_student.py`. Run provenance
  records the base Git SHA and hashes of the exact source files used.
