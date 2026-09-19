# Ten fixed-pair independent models

Each source has ten deterministic, distinct same-session TRAIN GT paths. Model k always uses rank k. After path selection, the target follows the original diffusion `ReactionDataset.__getitem__` crop and short-tail-fill path without a separate target transform.

- Training: `2500` steps/model, `25000` total model-steps, single seed `20260918`
- Objective: `paired_cost + 0.1 * grouped_banded_SoftDTW on same fixed pair`
- Parameters: `143097` per model, `1430970` total
- Evaluation: deterministic VAL `571`; 10 CUDA-stream ensemble outputs; exact rolling FRD with `16` workers

- Matched initialization hash: `98e805df58a13f28670379eb08a3b967d0fa61be7f7757126abf05aab0f8bea6`; source schedule hash: `c01544d007d219720b8f283d8eb8d47e0284dee73d4eeb20fd8c9e2b0bd2fabc`
- Distinct fixed-target mapping hashes: `10`; source snapshot: `7d967c2d5208f0eeb5e687313aa8ba61499411f5b869cf4c339b3bb214d004e8`

| method | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | GT coverage ↓ |
|---|---:|---:|---:|---:|---:|
| frozen M1 | 0.810111 | 82.480135 | 0.013611 | 0.039200 | 1.095160 |
| fixed-pair ensemble | 0.545194 | 88.513177 | 0.034755 | 0.037998 | 1.147073 |

Deltas versus M1: ΔFRC `-0.264918`, Δexact-FRD `+6.033043`, ΔFRDiv `+0.021144`, ΔFRVar `-0.001202`, ΔGT-coverage `+0.051913`.

Additional diagnostics: prediction validity `0.922339`, candidate utilization `0.624694`, cluster coverage `0.073409`.

Forty real-data target audits (10 fixed ranks × 4 sources) matched the original diffusion crop and speaker-tail-fill output element-for-element.

Verdict: fixed one-to-one specialization creates substantially different outputs, but it is not quality-safe. The diversity gain accompanies lower FRC, higher exact FRD, and worse GT coverage.

The ten-model row is not compute-matched to M1: it uses ten independent backbones and ten times the model-step budget. Same-session GTs remain weak pair proxies rather than verified reactions to the identical stimulus.
