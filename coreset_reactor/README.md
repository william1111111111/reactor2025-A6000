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
To test a B3 descriptor weight against an existing same-seed B1 run, use `--arm B3 --descriptor-cover 0.5` with the same `--seed`, `--steps`, and `--eval-examples`.
For a quality-first schedule, add `--descriptor-cover 0.15 --descriptor-warmup-fraction 0.2 --descriptor-ramp-end-fraction 0.6`: the whole descriptor block has zero weight through 20% of steps, rises linearly to 0.15 at 60%, then stays there. Each training-curve row records the applied `descriptor_weight`.

New run configs record `git_head_sha`, `git_commit_sha` only when the relevant source is tracked and clean, and a `source_snapshot_sha256` across local CoReSet/Mam-Reactor Python source. A null `git_commit_sha` means the HEAD alone does not identify the executed source (for example, when CoReSet is untracked).
For matched longer-run comparisons, use the same `--seed`, `--steps`, `--eval-examples`, and `--device` in each arm. `--save-checkpoint-steps 600` writes an evaluation-only checkpoint without interrupting the training RNG state; run configs also record hashes of the initial model and fixed source schedule. Intermediate checkpoints can be evaluated after training with `python -m coreset_reactor.evaluate --checkpoint ... --output ... --max-examples 64 --eval-seed 1234 --metric-workers 4 --device cuda:5`. The evaluation seed also fixes the target post-processor's stochastic latent independently for each context. NumPy-based metric IPC keeps full-VAL evaluation from exhausting file descriptors while preserving the official per-context formulas.

Detailed outputs stay local under `coreset_reactor/reports/`. `cache/` and `reports/` are ignored by Git. Selected sanitized aggregate pilot summaries—including the [matched 2500-step comparison](results/M1_MATCHED_LONG_RUN.md) and its [deterministic three-seed/full-VAL confirmation](results/M1_DETERMINISTIC_CONFIRMATION.md)—are published in [`results/`](results/); there is no automatic upload or push from training.

The data assumption is limited: same-session recordings are not verified reactions to the identical stimulus. The deterministic VAL pilot calls the repository's official metric functions and target post-processor after rounding prediction AU channels, but is not the full official TEST score. No GT enters inference. The B3-vs-B1 preliminary criterion requires better descriptor coverage and FRDiv without FRC or exact FRD degradation.

After freezing M1, run the read-only M2 preflight with `python -m coreset_reactor.m2_diagnostics --checkpoint ... --output ... --max-examples 571 --eval-seed 1234 --device cuda:5`. It compares random, k-medoids and approximate max-FRDiv real-GT sets, then measures descriptor-nearest responsibility and AU/VA/expression patterns for all ten prediction slots. The sanitized [full-VAL diagnostic summary](results/M2_GT_CEILING_SLOT_DIAGNOSTICS.md) is published without per-context identifiers.

The first controlled mode-adapter experiment is reproducible with `python -m coreset_reactor.train --config coreset_reactor/configs/m2_mode_adapter.json --arm B3 --save-checkpoint-steps 600 --run-name m2_mode_adapter_v1 --device cuda:5`. `--stop-after-step 600` provides a safety run using the exact prefix of the planned 2500-step schedule, unlike a separately generated 600-step schedule. The [single-seed v1 result](results/M2_MODE_ADAPTER_V1.md) is negative: the zero-initialized adapter preserves the M1 function at initialization but does not improve diversity or specialization after training.

M2-v2 keeps the 172,122-parameter M1 architecture and adds only TRAIN-derived behavioral mode routing. `configs/m2_mode_routing.json` deterministically partitions the 1,660 TRAIN descriptors into ten capacity-balanced medoid modes, fixes slot k to mode k, and ramps a 0.015 routed descriptor loss from 20% to 50% of training. Missing session modes are skipped; VAL descriptors never enter clustering. The [single-seed matched result](results/M2_GT_DERIVED_MODE_ROUTING.md) reports the original 0.02/0.05 points; the [low-weight sweep](results/M2_ROUTE_WEIGHT_SWEEP.md) selects 0.015 as the current candidate.

Raw experiment reports, cached descriptors, datasets, and checkpoints are local-only and are not distributed in this source repository. Training and evaluation also require the FaceVerse statistics and post-processor checkpoint referenced by the existing Mam-Reactor code.
