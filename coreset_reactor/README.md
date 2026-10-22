# CoReSet-Reactor (Milestone 1)

Independent parallel TCN for conditional reaction-set compression. It does not reuse the Mam-Reactor model. The only shared components are the existing paired feature loader, official metric/target-processing code, and differentiable SoftDTW loss kernel.

Milestone 1 compares the same K=10 TCN and optimizer from the same initialization:

- B0: paired + 9 random same-session exact GTs.
- B1: paired + 17 random exact GTs.
- B3: B1 plus descriptor coverage/validity over every distinct same-session TRAIN listener GT (at least 90 per session).

The 602-D descriptor is computed from every frame of each raw TRAIN listener recording and cached once (v2 cache). Only the sampled expensive temporal targets use a fixed 750-frame crop or stretch. The TCN projects audio/face/3DMM to 64/32/32, has five depthwise dilated blocks, predicts all 10 control-point streams in one forward pass, interpolates logits, then applies AU sigmoid, VA tanh and expression softmax. Short source clips are processed at their real length; their padded tails are excluded from descriptor and temporal losses.

From the repository root (with a compatible Python environment and the dataset available):

```bash
.venv/bin/python -m coreset_reactor.train --device cuda:5 --run-name m1_initial
```

For a contract smoke test, add `--steps 2 --eval-examples 1 --run-name m1_smoke`.
For independent replications, set `--seed` and keep the config's `eval_seed` fixed so the same VAL subset is used.
To test a B3 descriptor weight against an existing same-seed B1 run, use `--arm B3 --descriptor-cover 0.5` with the same `--seed`, `--steps`, and `--eval-examples`.
For a quality-first schedule, add `--descriptor-cover 0.15 --descriptor-warmup-fraction 0.2 --descriptor-ramp-end-fraction 0.6`: the whole descriptor block has zero weight through 20% of steps, rises linearly to 0.15 at 60%, then stays there. Each training-curve row records the applied `descriptor_weight`.

New run configs record `git_head_sha`, `git_commit_sha` only when the relevant source is tracked and clean, and a `source_snapshot_sha256` across local CoReSet/Mam-Reactor Python source. A null `git_commit_sha` means the HEAD alone does not identify the executed source (for example, when CoReSet is untracked).
For matched longer-run comparisons, use the same `--seed`, `--steps`, `--eval-examples`, and `--device` in each arm. `--save-checkpoint-steps 600` writes an evaluation-only checkpoint without interrupting the training RNG state; run configs also record hashes of the initial model and fixed source schedule. Intermediate checkpoints can be evaluated after training with `python -m coreset_reactor.evaluate --checkpoint ... --output ... --max-examples 64 --device cuda:5`.

Detailed outputs stay local under `coreset_reactor/reports/`. `cache/` and
`reports/` are ignored by Git. Only accepted aggregate tables are published
under `published_results/`; training never uploads or pushes automatically.

The data assumption is limited: same-session recordings are not verified reactions to the identical stimulus. The deterministic VAL pilot calls the repository's official metric functions and target post-processor after rounding prediction AU channels, but is not the full official TEST score. No GT enters inference. The B3-vs-B1 preliminary criterion requires better descriptor coverage and FRDiv without FRC or exact FRD degradation.

Raw experiment reports, cached descriptors, datasets, and checkpoints are local-only and are not distributed in this source repository. Training and evaluation also require the FaceVerse statistics and post-processor checkpoint referenced by the existing Mam-Reactor code.

## Center-window GT alignment audit

`eval_expert_student_fast` evaluates all VAL speaker contexts but only the
center 750 input frames (or the full clip when shorter). Its historical
`legacy_compress` policy converts each complete GT directly to that window's
length. To retain the source window's position on the complete timeline, pass
`--target-alignment full_then_crop`: the official Processor first converts GT
to the complete source duration, then the evaluator selects the same center
window as the input. This policy uses a separate versioned cache key including
the source duration and crop offset. The legacy default remains available for
reproducing frozen experiment runners and old scores.

Neither policy makes these scores full-sequence official TEST results.
Nonpaired same-session references have no verified frame-level semantic
correspondence; full-duration conversion followed by cropping is an explicit
evaluation convention, not evidence of such correspondence.

`compare_crop_target_alignment` reuses the frozen basic student's saved K10
predictions, verifies the archived per-context FRC and exact FRD against the
old cache, and computes both metrics against the newly aligned targets. It
does not train or run the student again. Launch nonoverlapping shards with
`--shard N --shards 4 --device cuda:N --output PATH`, then run the same module
with `--aggregate --output PATH` after all four shards finish. Results and
target arrays remain local under the selected report directory.

## Full-sequence, both-direction, all-session comparison

`eval_full_sequence_all_session` evaluates the frozen REACT2025 basic student
and the archived Mam-Reactor offline evidence checkpoint on identical VAL
inputs and identical postprocessed GT tensors. All same-session opposite-role
references are retained. Each model predicts nonoverlapping offline 750-frame
chunks, which are concatenated and trimmed to the original sequence length
before evaluation. These are full-length outputs, not unlimited-context model
forward passes. The report separates speaker-to-listener, listener-to-speaker,
and their equally weighted union.

Run `--prepare --output PATH` once to freeze all 1142 contexts, source/GT file
hashes, model hashes and evaluator source hashes. Run six shards with
`--shard N --shards 6 --output PATH` and a distinct `CUDA_VISIBLE_DEVICES` for
each, then `--aggregate --output PATH`. Per-context prediction/GT arrays and
metrics persist so interrupted shards resume without redoing completed work.

`exact_dtw_cuda` implements the same unconstrained DTW recurrence with FP32
inputs and FP64 accumulation, separately for the official AU/VA/expression
groups. It uses no downsampling, temporal band or approximate alignment.
CPU/GPU parity tests are in `tests/test_full_sequence_all_session.py`.
This is still a VAL experiment with deliberately all-session GT, rather than
the official hidden test or the released loader's sampled GT policy.

### Required protocol for metric ablations

REACT2025 ablations that report FRC, exact FRD, FRDiv, FRVar, or TLCC/FRSyn
must use complete full-sequence VAL1142: 571 contexts in each direction, all
opposite-role same-session GT, official Processor behavior, and official AU
rounding. Single-direction and center-750 result tables must not be published
or drive an operating-point claim. The
enforced contract is documented in `EVALUATION_CONTRACT.md`; the active
multi-view evaluator is `eval_multiview_ablation_bidir_all_session.py`.
The completed six-arm aggregate is published in
`../published_results/REACT2025_BIDIRECTIONAL_OFFICIAL5_ABLATIONS_20261022.md`.
The selected final `p=4` path is evaluated by
`eval_multiview_p4_bidir_all_session.py`. After official AU rounding it adds
one source-aligned transition frame to every constant query-0 AU in eligible
contexts. This remains a documented output postprocess; the six-arm fusion
table keeps its raw predictions so the power ablation is not confounded.
