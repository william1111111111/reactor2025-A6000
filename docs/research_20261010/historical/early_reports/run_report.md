> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Fixed-teacher/distillation implementation run report

Date: 2026-09-20

## Completed in this round

### Repository/protocol audit

- Added `reports/repo_audit.md` and `reports/protocol.md`.
- Locked the distinction between sampled-10 and the user-requested full VAL571 all-session-GT diagnostic.
- Locked the vectorized/cached fast-FRC implementation with scalar Mam-Reactor parity checks.
- Recorded the cross-listener speaker-tail-fill limitation of the legacy teacher runs.

### Fixed-random assignment

- Added `coreset_reactor/fixed_pair_manifest.py`.
- Built a real TRAIN-only K=10 manifest from 3,320 records.
- Manifest path: `coreset_reactor/reports/fixed_teacher_distillation/manifests/fixed_random_train_seed1.json`.
- Manifest size: 39 MiB; it remains local under the ignored experiment directory.
- Hash: `dada1b0213fa1431e548fb4976dbfab9b11c9805fe85debad649b1f7fc636825`.
- Every source has ten unique legal targets; duplicate slots: 0.

### Shared expert-conditioned student

- Refactored `ConditionalREGNN` into parameter-identical `encode_conditions` and `decode_context` methods. Existing 3,720,129-parameter checkpoint loaded strictly with all keys matching.
- Added `mam_reactor/regnn/expert_student.py::ExpertConditionalREGNN`.
- Speaker conditions are encoded once. A learnable expert embedding modulates the encoded sequence, and all experts share the complete decoder/graph head.
- No teacher module or teacher checkpoint path is part of the deployment model.
- K=2 student parameter count: 3,802,689, versus 7,440,258 parameters for two independent teachers.

### Corrected cross-listener supervision path

- Added explicit `--fixed-pair-alignment-policy` with backward-compatible `diffusion_legacy` and corrected `relative_time_masked` choices.
- The corrected path keeps native pairs on their real aligned prefix. Cross-listener targets are mapped by relative time: AU uses nearest-neighbor resampling, VA/expression use linear interpolation, and expression probabilities are renormalized.
- Cross-listener pointwise velocity supervision is disabled while masked MSE/CCC remains active.
- A real TRAIN rank-1 sample produced finite `[750,25]`, `velocity_supervision=False`, and expression-sum error `1.79e-7`.

### Teacher cache and K=2 real-data smoke

- Added `coreset_reactor/teacher_cache.py` and `coreset_reactor/train_expert_student.py`.
- Cache generation uses teacher `eval()` plus `inference_mode()`, fixed TRAIN view epoch 0, and records sample/view/start/length, teacher hashes, preprocessing version and sampler hash.
- First cache command exited 1 before reading samples because `MAM_REACTOR_ROOT` was absent. The empty output directory was preserved; no data/checkpoint was overwritten.
- Corrected command exited 0 and cached four real TRAIN inputs from independent teacher ranks 0 and 1 at epoch 5.
- Cache path: `coreset_reactor/reports/fixed_teacher_distillation/k2_cache_smoke_real4_v2`.
- Cache manifest hash: `056a1d863430b819f76362b5b0175b07b5269e1147742c973dc5bc49e4024855`.
- Teacher checkpoint hashes: `b543f7...a83a16`, `e0b924...04358`.
- Four-step real-data K=2 student smoke exited 0; all losses and gradients were finite.
- Student path: `coreset_reactor/reports/fixed_teacher_distillation/k2_student_smoke_real4/student.pth`.
- Student-only CPU inference, without loading either teacher, produced finite `[1,2,750,25]` output.

This smoke validates the interface and data chain only. Four samples/four updates do not establish validation quality or successful full-data distillation.

## Automated tests

Command:

```bash
PYTHONPATH=mam_reactor:. $LOCAL_PATH/python \
  -m pytest -q \
  coreset_reactor/tests/test_expert_student.py \
  coreset_reactor/tests/test_fixed_pair_manifest.py \
  coreset_reactor/tests/test_anchor_fixed_pairs.py \
  coreset_reactor/tests/test_fixed_pair_ensemble.py
```

Focused new-path result: **14 passed**, exit 0. The complete
`coreset_reactor/tests` suite also passed: **39 passed**, exit 0.

Verified behaviors include deterministic order-independent assignment, explicit duplicate marking for `M_i<K`, native-pair preservation, invalid manifest rejection, one speaker-encoding call for K outputs, expert-embedding gradient, no teacher modules in the student, and micro-overfitting two distinct teacher targets for the same input. The microfit reached MSE below `2e-4` and slot separation above `0.4`; this proves wiring, not generalization.

## Existing long run retained

The previously authorized K=10 farthest legacy teacher run remains active and was not restarted. Latest observed completed epochs:

```text
models 00--03: 29
models 04--07: 28
models 08--09: 50 (complete)
```

Its full matched evaluation must wait until all ten teachers finish. The short 50-optimizer-step run remains excluded from fair 50-epoch conclusions.

## Not yet completed

- Full TRAIN K=2 teacher cache and meaningful K=2 student training/evaluation.
- Corrected mask-aware S-direct baseline.
- K=10 student cache/training.
- S-KD-rel (deliberately deferred until basic KD evidence exists).
- Student sampled-10 and all-session-GT VAL571 metrics.
- Matched latency, peak memory and throughput benchmark.
- Video FRRea (assets unavailable in this numeric data root).

No improvement claim is made from the smoke run.
