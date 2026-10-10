> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Teacher-relation distillation pilot

This experiment tests whether one shared student can recover more of the
prototype teacher set's diversity without losing basic-KD quality.

Frozen baseline: `t10_student_pilot10e/student.pth`, full VAL571 all-session GT
FRC 1.1784827273, FRDiv 0.0607287697, FRVar 0.0432094932. Exact FRD was already
being evaluated when this trial started. These are local VAL results, not
official TEST scores. Prototype T10 FRDiv is 0.1183250919.

## Single intervention

For all 45 pairs of the ten outputs, calculate valid-frame RMS trajectory
distance separately over AU (15 channels), VA (2), and expression (8).
Keep soft AU outputs in the training objective. Match student distances to
detached teacher distances with an absolute-error loss. Both contraction and
overshoot are penalized. There is no unbounded pairwise repulsion.

Each group's normalization is frozen to its mean teacher distance on the
first 32 ordered TRAIN cache records, floored at 0.01. No VAL GT enters
normalization or loss-weight choice. The resulting scales are
`[0.33919975, 0.33802897, 0.30582511]`.

Four initial TRAIN batches (16 records) calibrate the auxiliary weight to
approximately one quarter of the basic-KD parameter gradient norm. The
chosen fixed weight is `0.2824869836202808`; measured probe ratios are
0.21594, 0.26691, 0.28279, 0.23510. The calibration restores Torch RNG states
and does not iterate or advance the training DataLoader. Initial batch KD
loss is exactly 0.9414509534835815 in both runs, providing an initialization,
input-order and dropout-state spot check.

The full objective is the existing MSE + (1-CCC) + 0.05 velocity MSE, plus
the calibrated weighted relation loss. Architecture and parameter count
(3,803,201) are unchanged. The relation term adds computation but no new
expensive temporal-GT evaluations. It is a teacher-output-space objective.

## Matched training

- Same seed 1, teacher-0 epoch-50 backbone initialization, expert embeddings.
- Same full TRAIN cache (3,320 inputs, fixed view epoch 0, 10 teachers).
- Same batch 4, workers 4, AdamW LR 1e-4 and original cosine schedule.
- Same 10 epochs / 8,300 updates, BF16, clipping 1.0.
- New run: `coreset_reactor/reports/fixed_teacher_distillation/t10_student_rel_pilot10e`.
- Calibration, actual source hashes, base Git SHA and immutable input hashes
  are recorded in that run's `provenance.json`.
- Atomic epoch snapshots and final deployable student weights are saved.
- Training seconds and peak allocated GPU memory are recorded at completion.

The run uses legacy prototype teachers for an isolated comparison with the
existing KD experiment. This does not retroactively correct their legacy
cross-listener label construction.

## Evaluation and artifacts

`python -m coreset_reactor.run_relation_evaluation --device cuda:6` is queued.
It re-evaluates the frozen baseline, checks its FRC/FRDiv/FRVar against the
existing result within 1e-4, waits for relational KD training, then evaluates
the new model on all 571 VAL speaker contexts, center750, all opposite-role
session GT, eval_seed=1234. The full GT set is unchanged.

A shared persistent GT cache keys each context on ordered GT paths and file
metadata, length, seed, frozen postprocessor checkpoint hash, source hashes,
runtime and output count. Each metric run checks vectorized FRC against the
scalar Mam-Reactor function on its first real sample. Per-context results,
soft/rounded prediction sets, soft/rounded FRDiv, per-slot FRC, checkpoint
hashes and diagnostic inference timing are saved. Timing is not a controlled
speed benchmark.

After fast metrics, exact FRD runs on all 571 contexts with 16 workers. The
queue writes `reports/student_relation_comparison.md` and the run's
`matched_comparison.json`, including paired context delta distributions and
descriptive bootstrap intervals (these do not account for session correlation).

Success requires checking quality and diversity jointly against basic KD.
A higher training pair distance alone is not success. Full student quality
metrics and retained T10 diversity must both be reported, even for a negative
result. No follow-up weight sweep or architecture change is auto-launched.

## Verification so far

41 existing/new unit tests passed, including teacher stop-gradient,
padding invariance, pair-permutation invariance and overshoot penalty.
Training has started with finite losses and gradients. Full VAL results
for this new experiment are pending; the queue will write the comparison.
