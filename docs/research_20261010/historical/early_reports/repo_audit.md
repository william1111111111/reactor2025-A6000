> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# REACT2025 fixed-teacher/distillation repository audit

Audit date: 2026-09-20 (Asia/Shanghai)

## Repository and runtime

- Repository: `$SOURCE_ROOT`
- Branch/HEAD at audit start: `main`, `8b6c67a5c753de907ad9d5f64cb23eb21f2e1c04`
- Remote: `https://github.com/william1111111111/reactor2025-A6000.git`
- Worktree is intentionally dirty: fixed-target training/evaluation changes and local experiment launchers are not committed yet.
- Python 3.10.12; PyTorch 2.1.0+cu121; CUDA runtime 12.1.
- Eight NVIDIA RTX A6000 GPUs are visible. At audit time GPUs 1, 5, and 6 were occupied by the previously authorized ten-teacher matched run; other GPUs may belong to unrelated users/tasks and are not assumed available.
- Free space at audit time: `/tmp` 688 GiB, `/home` 396 GiB, `/public` 4.2 TiB.

## Local data and assets

- Data root: `$PROJECT_ROOT/data` (126 GiB).
- TRAIN: 3,320 facial-attribute files, 3,320 audio-feature files, 3,320 coefficient files, 17 sessions, both `speaker` and `listener` roles.
- VAL: 1,142 files per numeric modality. Current center-750 experiments use the 571 `speaker` records as inputs and same-session `listener` records as GT.
- TEST: 1,142 files per numeric modality. It has not been used for selecting the current fixed targets.
- Local data root does not contain a `videos` directory; video realism/rendering evaluation is therefore `not_evaluated` in this workstream.
- Facial attributes are variable-length `[T,25]`. Sample TRAIN lengths include 691--1021 frames.
- Speaker conditions used by the active model: audio 768D, facial attributes 25D, normalized FaceVerse coefficients 58D.
- FaceVerse normalization: `$LOCAL_PATH/{mean_face,std_face}.npy`.
- Evaluation post-processor: `$LOCAL_PATH/checkpoint.pth`, SHA-256 `95d3d12737ed4157d80756b31613c014085807f846b381f8f3aa82afd1f61c5a`.

No subject identity field was found in the current numeric directory contract. The stable local identity is `role/session/basename`; the manifest records `speaker_id_if_available=null` rather than inventing a subject ID.

## Active architecture and training path

The active fixed-target teacher is the custom offline `ConditionalREGNN`, not Trans-VAE or PerFRDiff:

- Model: `mam_reactor/regnn/conditional_model.py::ConditionalREGNN`.
- Dataset: `mam_reactor/regnn/conditional_data.py::ConditionalReactionDataset`.
- Trainer: `mam_reactor/regnn/train_conditional_regnn.py`.
- Inputs are full center/random 750-frame audio/facial/3DMM sequences. The Transformer is bidirectional, so this configuration is **offline** and must not be reported as online.
- Output is one deterministic 25D listener reaction. The active ten-teacher set obtains ten predictions by loading ten independently trained teachers; outputs are stacked and never averaged.
- Current teacher parameter count: 3,720,129 each (37,201,290 for ten loaded models).
- Current loss: masked MSE + `1.0 * (1-CCC)` + `0.05 * velocity MSE`.
- Optimizer: AdamW, LR `1e-4`, weight decay `1e-4`, betas `(0.9,0.99)`, gradient clipping 1.0.
- Scheduler: cosine annealing over 50 epochs, minimum LR `5e-6`.
- Precision/batch: BF16, batch 32, drop-last, 103 optimizer steps per full epoch.
- Listener 3DMM supervision is disabled in this experiment; 58D is a speaker condition only.

Known reusable anchor: `$LOCAL_PATH/conditional_regnn_offline_anchor_epoch0050.pth`, SHA-256 `9d0190a7cb48703e001dc3f092d84ac7de17bea5e2412604f3871099d1c3ebb6`.

## Fixed-pair state

Three local assignment families currently exist:

1. Historical paired-plus-nine deterministic random teachers (`anchor_paired_plus9_diffusion_full50_seed1`). The mapping is reproducible from seed and code, but was not serialized as a rich manifest before training.
2. Global-descriptor prototype representatives (`anchor_behavioral_pairs_full50_seed1`).
3. Descriptor farthest-first targets (`anchor_farthest_full50_seed1`), currently training to 50 epochs. This is a research comparison, not proof of ten natural semantic modes.

The new `fixed_random` builder is `coreset_reactor/fixed_pair_manifest.py`. Its real TRAIN manifest is local at `coreset_reactor/reports/fixed_teacher_distillation/manifests/fixed_random_train_seed1.json` (39 MiB, ignored by Git). It contains all 3,320 source records, their legal same-session opposite-role candidates, lengths, ten assignments, duplicate flags, preprocessing version, and hash. Manifest hash: `dada1b0213fa1431e548fb4976dbfab9b11c9805fe85debad649b1f7fc636825`. All TRAIN sources had enough candidates, so duplicate slots are zero.

## Important alignment finding

The currently running historical fixed-teacher experiments intentionally follow the earlier user-requested diffusion behavior: selected cross-listener targets reuse the speaker crop offset, and a short target is filled with the cropped speaker-emotion tail (`diffusion_listener_crop`). This reproduces the local diffusion loader, but the filled tail is not valid cross-listener supervision.

Therefore:

- existing random/prototype/farthest runs are retained as **legacy diffusion-compatible comparisons**;
- they are not silently re-labelled as mask-correct cross-listener training;
- the new manifest declares `alignment_policy=relative_time_masked` for the subsequent student/distillation implementation;
- future corrected comparisons must use independent target length/mask and must not compute velocity loss across invalid/fill frames.

## Existing checkpoints and evaluations

- The historical random and prototype ten-teacher runs have complete epoch-50 checkpoints.
- The farthest run was already launched before this task document and is allowed to finish; no duplicate full run was started.
- A 50-optimizer-step farthest smoke run exists and is not treated as comparable to 50 epochs.
- Full VAL571 all-session-GT results exist for the historical random/prototype and 50-step farthest runs. This all-session mode is a local upper-coverage diagnostic, not identical to the public loader's paired-plus-nine sampling protocol.
- A fast all-session FRC evaluator now vectorizes all candidate/GT CCC comparisons and reuses deterministic post-processed GT across epochs. Synthetic parity against the Mam-Reactor scalar implementation was exact to about `1e-17` and about 30x faster for the CCC kernel. Real evaluation keeps a per-run scalar parity check.

## Gaps before student claims

- No shared expert-conditioned student has yet completed a real-data K=2 distillation closed loop.
- No teacher-output cache with checkpoint/view/sampler hashes exists yet.
- No student-only isolation test exists yet.
- No student parameter/latency/peak-memory comparison exists yet.
- No video FRRea path can run with the current local numeric-only data root.
- The local TLCC implementation returns inside the prediction loop and therefore measures only prediction slot 0. Results must retain the code name `TLCC` and disclose this behavior; they must not be presented as a ten-slot synchronization aggregate.

## Verified commands in this audit

```bash
$LOCAL_PATH/python \
  -m coreset_reactor.fixed_pair_manifest \
  --data-root $PROJECT_ROOT/data \
  --output coreset_reactor/reports/fixed_teacher_distillation/manifests/fixed_random_train_seed1.json \
  --assignment-seed 1 --k 10

PYTHONPATH=mam_reactor:. $LOCAL_PATH/python \
  -m pytest -q \
  coreset_reactor/tests/test_fixed_pair_manifest.py \
  coreset_reactor/tests/test_anchor_fixed_pairs.py \
  coreset_reactor/tests/test_fixed_pair_ensemble.py
```

Observed result: manifest creation exited 0; eight tests passed.
