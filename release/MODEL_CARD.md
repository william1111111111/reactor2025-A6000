# REACT2025 main-result student

| Field | Value |
| --- | --- |
| Method | Full CoReSet, A0 |
| Student seed | 1 |
| Parameters | 3,803,201 |
| Weight file | `checkpoints/coreset_react2025_a0_seed1.pth` |
| Size | 15,252,479 bytes, FP32 |
| Optimizer updates | 8,820, four student epochs |
| Outputs | Ten expert-conditioned 25-D AFR sequences |

The weight export contains the unchanged model tensors and configuration,
normalization statistics and compact inference metadata. It omits the
optimizer and machine-specific run paths. Every tensor is compared against
the original checkpoint during export. The exported file loads with
`torch.load(..., weights_only=True)` and strict `load_state_dict`.

```text
Original checkpoint SHA-256:
24c53d5af11eb8df8b4100cb0a4c027a0ea6a899b289dac8e1cbddb56ca6da4d

Released checkpoint SHA-256:
639a9bb260ff56ac413d9162b283cce728f77a8fbd60a78ee3b5d94f8fb83459

Parameter-state SHA-256 (ordered names and tensor bytes):
dc87e0ceba933f530c23b0a14da1853f731d7210ba497d5886d9c43ac7af6cbc
```

## Archived paper result

| FRC ↑ | FRD ↓ | FRDiv ↑ | FRVar ↑ | TLCC ↓ |
| ---: | ---: | ---: | ---: | ---: |
| 1.1013659645 | 134.2255057332 | 0.1789588281 | 0.0570926070 | 46.0350262697 |

These are the archived main-result metrics, not scores produced by the
installation smoke test. The protocol is full REACT2025 VAL1142, both
directions equally weighted, complete sequences, all opposite-role
same-session references, the original official target Processor, official
AU rounding and exact, unconstrained DTW. It is distinct from a hidden
challenge TEST evaluation or a sampled-ten-reference evaluation.

The evaluation Processor aligns reference targets for metric computation;
it is not required to run the student. The source/result hashes and the
original teacher bank hashes are in `configs/react2025_main.json`.

## Training provenance

The archived student's initialization checksum and the ordered teacher-cache
bank checksums are stored independently in `configs/react2025_main.json`.
In that historical run the initializer was an earlier 600-update q0 checkpoint,
distinct from the later 600-update q0 snapshot used in the teacher cache.
The fresh-training script supports `CORESET_INITIALIZER` for a separate
initializer and defaults to its newly trained q0. Only the final A0 student
weight is bundled in this release.

## Forward and full-sequence inference

The network uses source audio (768-D), facial attributes (25-D), and
normalized 3DMM coefficients (58-D). It predicts 15 sigmoid AUs, two tanh
VA values and eight softmax expression probabilities. It is an offline,
bidirectional temporal model. Each forward call accepts up to 750 frames;
valid-prefix masks handle shorter input.

For a longer sequence, the paper recipe evaluates 750-frame windows with
stride 375 and a final window aligned to the sequence tail. At each frame,
overlapping predictions are fused using positive triangular center weights
raised to power four and normalized by their sum. Inference is FP32 with
TF32 disabled, matching the archived inference configuration.

The CLI saves two outputs:

- `prediction`: soft, fused AFRs for rendering and within-set analysis.
- `paper_prediction`: round the 15 AU channels, then apply the archived
  fixed query-0 transition postprocess. If every source feature channel
  varies and a query-0 AU is constant, one interior frame is flipped at the
  corresponding source AU extremum, with a 49-frame boundary margin.
  Sequences of 98 frames or fewer receive AU rounding only. This fixed
  output convention is exposed separately from the neural network forward.

The main-result row above includes that postprocess. It affects evaluation
of constant AU trajectories, including TLCC. Reproducing the table requires
the complete inference and target-processing protocol, as well as the
weight. Fresh training or different numerical runtimes can vary; the bundled
weight is the exact released parameter state for the archived result.
