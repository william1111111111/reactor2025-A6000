# CoReSet-Reactor (Milestone 1)

Independent parallel TCN for conditional reaction-set compression. It does not reuse the Mam-Reactor model. The only shared components are the existing paired feature loader, official metric/target-processing code, and differentiable SoftDTW loss kernel.

Milestone 1 compares the same K=10 TCN and optimizer from the same initialization:

- B0: paired + 9 random same-session exact GTs.
- B1: paired + 17 random exact GTs.
- B3: B1 plus descriptor coverage/validity over every distinct same-session TRAIN listener GT (at least 90 per session).

The 602-D descriptor is computed from every frame of each raw TRAIN listener recording and cached once (v2 cache). Only the sampled expensive temporal targets use a fixed 750-frame crop or stretch. The TCN projects audio/face/3DMM to 64/32/32, has five depthwise dilated blocks, predicts all 10 control-point streams in one forward pass, interpolates logits, then applies AU sigmoid, VA tanh and expression softmax. Short source clips are processed at their real length; their padded tails are excluded from descriptor and temporal losses.

From the `react2025_new` root (with a compatible Python environment and the dataset available):

```bash
.venv/bin/python -m coreset_reactor.train --device cuda:5 --run-name m1_initial
```

For a contract smoke test, add `--steps 2 --eval-examples 1 --run-name m1_smoke`.
For independent replications, set `--seed` and keep the config's `eval_seed` fixed so the same VAL subset is used.

Outputs stay local under `coreset_reactor/reports/`. `cache/` and `reports/` are ignored by Git. No automatic upload or push exists.

The data assumption is limited: same-session recordings are not verified reactions to the identical stimulus. The deterministic VAL pilot calls the repository's official metric functions and target post-processor after rounding prediction AU channels, but is not the full official TEST score. No GT enters inference. The B3-vs-B1 preliminary criterion requires better descriptor coverage and FRDiv without FRC or exact FRD degradation.

Experiment reports, cached descriptors, datasets, and checkpoints are local-only and are not distributed in this source repository. Training and evaluation also require the FaceVerse statistics and post-processor checkpoint referenced by the existing Mam-Reactor code.
