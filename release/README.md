# CoReSet

**Complementary Reference-guided AFR Set Distillation** for multiple appropriate
facial reaction generation. This self-contained release includes reference
organization, teacher training, teacher-output caching, student distillation,
the feature dataset, forward/inference code, and the REACT2025 **A0, seed 1**
student used for the paper's main result.

[中文说明](README_zh.md) · [Data format](DATASET.md) ·
[Checkpoint and protocol](MODEL_CARD.md) · [Validation](VALIDATION.md)

## Setup

Use Python 3.10. The verified runtime uses PyTorch 2.1.0, NumPy 1.26.4,
SciPy 1.15.3 and scikit-learn 1.7.2. Install the PyTorch build for your
CUDA/CPU environment, then run from this directory:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e '.[test]'
```

The release runs independently of the historical research directories.
Inference needs only PyTorch, NumPy and the included student checkpoint;
it consumes pre-extracted speaker features. Teacher training and reference
organization also use SciPy and scikit-learn.

## Pretrained inference

The single included weight is
[`checkpoints/coreset_react2025_a0_seed1.pth`](checkpoints/coreset_react2025_a0_seed1.pth)
(15.25 MB, FP32, **3,803,201 parameters**). Normalization statistics are
included in the checkpoint. No teacher models or reference reactions enter
the inference path.

Check your installation using synthetic features:

```bash
python -m coreset_reactor.inference --demo --device cpu --output demo.npz
```

For a real sequence, supply aligned, pre-extracted audio, facial attributes
and raw 3DMM coefficients:

```bash
python -m coreset_reactor.inference \
  --audio /path/to/audio_features.npy \
  --facial-attributes /path/to/speaker_facial_attributes.npy \
  --coefficients /path/to/speaker_coefficients.npy \
  --checkpoint checkpoints/coreset_react2025_a0_seed1.pth \
  --device cuda:0 --batch-size 1 --output reaction_set.npz
```

Input dimensions are `[T,768]`, `[T,25]` and `[T,58]` (also `[T,1,58]`
for 3DMM). Audio and facial features are used as stored. Raw 3DMM is
normalized once with the bundled mean/std; pass `--normalized-3dmm` only
if it was already normalized.

The NPZ contains `prediction` (continuous `[10,T,25]` AFRs for rendering and
structure analysis), `paper_prediction` (the evaluation output after AU
rounding and the fixed query-0 transition postprocess), and `expert_ids`.
Long inputs use 750-frame windows, stride 375 and normalized triangular
weights raised to power 4, matching the paper's inference recipe. The
transition postprocess and its effect are described in [MODEL_CARD.md](MODEL_CARD.md).

## Model forward

```python
import torch
from coreset_reactor.inference import load_student

model, metadata = load_student(device="cpu")
audio = torch.randn(1, 64, 768)
facial = torch.randn(1, 64, 25)
normalized_3dmm = torch.randn(1, 64, 58)
lengths = torch.tensor([64])
with torch.inference_mode():
    outputs = model.generate(audio, facial, normalized_3dmm, lengths)
print(outputs["prediction"].shape)  # [1, 10, 64, 25]
```

`generate` encodes the source once, conditions on ten expert embeddings,
and decodes the ten outputs together. `model(..., expert_ids)` accepts one
expert ID per batch element. Forward inputs must fit the model's 750-frame
capacity; the CLI handles full-length input with window fusion. Channel
order is 15 AUs, two valence/arousal values and eight expression probabilities.

## Training

Prepare the REACT2025 features using [DATASET.md](DATASET.md), then run:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_react2025.sh /path/to/react2025 runs/react2025_seed1
```

This executes the full recipe sequentially on one GPU:

1. Train the paired teacher q0 and nine target-specific teachers, each for
   600 updates (batch 32, AdamW, learning rate and weight decay `1e-4`). The
   original cosine schedule has a 50-epoch horizon and stops at update 600.
2. Freeze all ten teachers and cache FP16 AFR targets for deterministic
   head/center/tail TRAIN views. Short recordings contribute one view;
   recordings longer than 750 frames contribute three.
3. Initialize the shared student backbone from q0 and distill for four
   epochs (batch 4, AdamW, learning rate `1e-4`). The loss combines trajectory
   MSE/CCC/velocity matching with signed, slot-centered residual matching.
   Group scales and the auxiliary coefficient are calibrated on TRAIN only,
   targeting an initial gradient ratio of 0.25.
4. Export the trained student to a weights-only checkpoint.

The paper TRAIN release contains 3,320 bidirectional source contexts and
8,820 cached views, yielding 8,820 student updates over four epochs.
[`configs/react2025_reference_assignment.json`](configs/react2025_reference_assignment.json)
preserves the paper's exact ten-reference identities and expert ordering.
[`configs/react2025_main.json`](configs/react2025_main.json) records the
original run's hyperparameters, checkpoint hashes, calibrated scales and
archived metrics. A fresh training run recalibrates its coefficient on TRAIN;
it does not select an output or tune a coefficient on VAL.

Historical provenance: the archived A0 run used an earlier 600-update paired
checkpoint for student initialization and a later q0 snapshot in the cached
teacher bank. Their hashes differ and are recorded separately in
`configs/react2025_main.json`. The fresh-training script uses its newly trained
q0 as initializer by default. To supply a separate historical-compatible
initializer, set `CORESET_INITIALIZER=/path/to/paired_teacher.pth` when launching
the script. The released A0 parameter state reproduces the archived model;
fresh training is not claimed to be bitwise identical to that historical run.

To build assignments for a different compatible TRAIN feature set:

```bash
python -m coreset_reactor.prepare_references \
  --data-root /path/to/react2025 --output runs/reference_assignment.json
```

This fits nine TRAIN descriptor prototypes and applies paired-constrained
farthest-sum selection followed by Hungarian assignment. Point
`assignment` in the training script to that file. The released frozen file
is the appropriate choice for reproducing the paper's reference identities.

## Checks

```bash
python -m pytest -q
python scripts/smoke_test.py
```

The smoke test generates synthetic feature files and small CPU models. It
executes reference organization, paired and target-specific teacher training,
ten-output caching, centered-difference student training and export. It tests
the execution path rather than training convergence.

## Layout

| Path | Purpose |
| --- | --- |
| `regnn/conditional_data.py` | Bidirectional feature dataset, crop/mask/target alignment |
| `regnn/conditional_model.py` | Multimodal temporal encoder and relation-aware decoder |
| `regnn/expert_student.py` | Shared expert-conditioned student and parallel forward |
| `regnn/train_conditional_regnn.py` | Paired teacher training |
| `coreset_reactor/prepare_references.py` | Farthest-sum selection and Hungarian assignment |
| `coreset_reactor/train_mode_reference_teacher.py` | Target-specific multi-reference teacher training |
| `coreset_reactor/teacher_cache.py` | Frozen teacher-output generation |
| `coreset_reactor/train_expert_student.py` | Trajectory and centered-difference distillation |
| `coreset_reactor/relation_distillation.py` | Masked set-structure losses |
| `coreset_reactor/inference.py` | Student loading, sequence inference and CLI |
| `checkpoints/` | One REACT2025 main-result student weight |

Source and weight provenance is recorded in `SOURCE_MANIFEST.json` and
`configs/react2025_main.json`. Code is provided under the [MIT license](LICENSE).
REACT/MARS data and the original feature extraction assets retain their
respective terms; obtain the dataset from its providers. The included
58-dimensional normalization statistics reproduce the FaceVerse scaling
used by the original feature loader.
