> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Latent-reference teacher supervision (preregistered probe)

Status: **completed; corrected mode-prior teacher candidate accepted for the
requested operating point**. REACT2025, seed1, 600 updates per teacher, unchanged
paired q0 reused. Full VAL571 all-session GT, not official sampled-ten scoring.

| Evaluated bank | FRC ↑ | exact FRD ↓ | FRDiv | FRVar |
|---|---:|---:|---:|---:|
| Matched T0 fixed random | 1.373312 | 84.928192 | 0.036411 | 0.036259 |
| Matched T1 fixed Max-FRDiv | 1.072308 | 96.808993 | 0.205423 | 0.068098 |
| Uniform reference (exploratory v1) | 2.038602 | 95.408503 | 0.091648 | 0.070389 |
| **Mode prior, corrected v2** | **1.370792** | **95.772670** | **0.190244** | **0.069174** |

The accepted candidate passes FRC>=1.10, FRDiv in [0.17,0.20], and FRD<=T1.
It is not a Pareto domination of T1 because FRDiv is modestly lower, and it does
not dominate T0 on FRD. Raw-GT FRC independently rises from 1.033339 to 1.347741.
Corrected-run session-bootstrap difference CIs versus T1 are FRC
[+0.23675,+0.33019], FRD [-1.72460,-0.07400], and FRDiv [-0.01852,-0.01126].
FRC improves on 96.8% of contexts; FRD improves on 65.8%. No per-context
zero-slack quality guarantee or multi-seed/independent-test claim is made.

Candidate artifacts: `reports/mode_reference_teacher_v2_seed1/`. Final weights
are `rank_01` through `rank_09/conditional-step000600-seed1.pth`, together with
the unchanged `teacher_convergence_2400_seed1/arms/T1/rank_00/conditional-step000600-seed1.pth`.
The final full evaluator was rerun, not reused; all 571 cache keys and clip IDs
match the baseline. Numerical/audit/unit tests: **23 passed**.

The accepted bank needs 6,000 total optimizer updates including reused q0
(5,400 new updates), but incurs approximately 2.46x historical T1 training time
at the same update count. The full exploration used three nine-specialist runs
(16,200 updates) plus two two-step smoke tests; this is not hidden as one run.
Inference remains ten unchanged teachers, 37.20M parameters and about 37.3ms.
No student training or architecture changes were made.

## Research question

Does treating the temporal reference as a latent variable improve quality while
keeping each teacher anchored to its original diverse behavioral target?

For teacher k, fixed T1 target y_k, same-session opposite-role TRAIN set Y_s,
prediction p_k, and mean-channel CCC c(p,y):

```
q_tau(p, Y_s) = tau * log( mean_y exp(c(p,y) / tau) )
L_k = MSE(p_k, y_k) + 1 - q_tau(p_k, Y_s)
      + unchanged 0.05 * masked_velocity_loss
tau = 0.05
```

Only specialists q1..q9 change their CCC reference rule. The original MSE target,
GT membership, target alignment and cross-listener velocity mask stay unchanged.
The original native paired q0 is reused bit-for-bit. No new parameters, inference
selection, GT-conditioned forward pass, output mixing or explicit FRDiv loss.

The normalized log-sum-exp lies between the average and maximum reference CCC.
Its gradient is a softmax-weighted mixture of reference gradients. It becomes
ordinary single-reference CCC when only one reference is available. This is a
standard latent-reference relaxation; its use here is a testable mechanism,
not yet a claim of algorithmic novelty.

Equivalently, this is the entropy-regularized latent-reference objective

```
q_tau(p, Y_s) = max_{pi in probability simplex}
    sum_j pi_j * c(p, y_j) - tau * KL(pi || Uniform(Y_s))
pi*_j = softmax_j(c(p, y_j) / tau)
```

Substituting the maximizing posterior gives the normalized log-sum-exp above.
Thus the loss is not assigning a new hard GT at inference; it marginalizes a
TRAIN-only nuisance reference under a uniform prior. This variational identity
is standard mathematics, not a new theorem. It makes the hypothesis and the
mean-CCC / hard-best-reference ablations explicit.

## Controls and costs

- Baselines: frozen matched T0_600 and T1_600.
- Same seed1, model/initialization, source/crop schedule, batch32, BF16,
  AdamW1e-4/betas(.9,.99)/wd1e-4/clip1, cosine schedule over 50 nominal epochs.
- 600 actual updates, not 50 completed epochs; 9 new models, one reused q0.
- Model size 3,720,129 each; final ten-model bank 37,201,290 parameters.
- One fixed MSE target per prediction. CCC now compares **all TRAIN-session GT**
  instead of one. This is additional temporal comparison work, not free compute.
  No expensive DTW is used in training. Report throughput and peak memory.
- Raw TRAIN references use the original AU-nearest / VA-expression-linear
  relative-time resampler at each source's valid length, with no padded frames.
- The validation postprocessor is not used to prepare these training references.

Numeric target: full VAL571 FRC >= 1.10, FRDiv in [0.17,0.20], exact FRD no worse
than T1_600. Do not tune the temperature on VAL after observing this probe.
The existing deterministic all-session processed-GT cache is reused; this is
the project's all-session extension, **not** official sampled-ten-GT evaluation.
Session-cluster bootstrap and context/session distributions accompany means.

If there is a positive result, subsequent controls must separate soft aggregation
from proper session conditioning and verify that diverse behavior remains.
If it fails, report the failure; do not substitute a favorable subset.

## Checks before the full run

- 8 unit tests passed across conditional assignment and latent-reference loss.
- CCC agrees with the existing official-parity vectorized metric, including
  constant channels; aggregation bounds and duplicate-reference invariance pass.
- Single-reference loss and gradient reduce to the original teacher loss.
- Padding exclusion test passes.
- Six real TRAIN examples spanning two epochs preserve source features, crops,
  raw fixed targets, lengths and velocity masks exactly.
- Two-step GPU smoke test has the exact baseline initialization hash; finite
  loss/gradient; peak allocated memory 6.91 GiB.
- Full run checks initialization/source schedule hashes again at 600 updates.

Artifacts: `reports/latent_reference_teacher_seed1/`; previous checkpoints remain
unchanged. Git base c421f509 with per-source hashes and code snapshot stored in
the new run. No GitHub push or external annotation is part of this experiment.

## Paper positioning and limitations

Oracle-based ensemble learning already exists, for example
[Stochastic Multiple Choice Learning (Lee et al., 2016)](https://arxiv.org/abs/1606.07839).
Multiple appropriate facial responses and reaction distributions are also
established, e.g. [ReactFace](https://arxiv.org/abs/2305.15748).
Neither a best-match operator nor ten experts by itself is a novel contribution.
The present empirical question is narrower: can fixed diverse mode anchoring
and latent temporal-reference supervision relieve conflicting supervision?

A positive metric result alone would not prove this explanation. Required next
controls include replacing the soft maximum by mean CCC, breaking same-session
conditioning, and checking input dependence. Same-session appropriateness does
not establish exact source-specific temporal causality. The all-session FRC
protocol must remain clearly distinguished from official sampled-ten scoring.

## First result: uniform prior does not meet the goal

| Full VAL571 | T1_600 | Uniform-reference |
|---|---:|---:|
| Processed all-session FRC | 1.072308 | 2.038602 |
| FRDiv | 0.205423 | 0.091648 |
| Raw all-session FRC | 1.033339 | 1.982976 |
| FRVar | 0.068098 | 0.070389 |
| exact FRD | 96.808993 | 95.408503 |

The substantial FRC gain also appears with raw references, but the loss of
diversity rejects this as the requested operating point. This does not prove
mode collapse by itself: semantic/output-distribution diagnostics are required.
Initialization and observed source/crop hashes passed for all nine retrained
specialists. Training was approximately 1.75x the historical T1 wall time at
the same update count; these were concurrent, non-isolated measurements.

All 571 contexts used the original cache keys and clip order. 98.1% of contexts
improved FRC, but **every context lost FRDiv**. Session-cluster bootstrap 95% CIs
for differences against T1 are FRC [+0.7804,+1.1245], FRD [-3.3428,+0.3865],
FRDiv [-0.1256,-0.1018]. Thus FRD is better in its observed mean, not a
statistically established improvement across sessions. Single-seed results do
not establish training-seed robustness.

The FP32 full evaluator took 688.5 seconds using cached GT and rolling-DTW workers;
mean prediction latency 44.3ms (35.4ms in the preceding FRC-only pass), both
non-isolated measurements. The bank has 37,201,290 parameters with no selector.

An exact pairwise-distance decomposition in float64 separates between-teacher
temporal-mean differences from zero-time-mean residual differences. Recomputed
totals agree with the original float32 official kernel within 1e-6.

| Output diagnostic | T1 | Uniform-reference |
|---|---:|---:|
| FRDiv time-mean component | 0.074945 | 0.031695 |
| FRDiv temporal-residual component | 0.130479 | 0.059953 |
| AU weighted FRDiv component | 0.171013 | 0.081866 |
| VA weighted FRDiv component | 0.011130 | 0.002806 |
| Expression weighted FRDiv component | 0.023281 | 0.006976 |
| Mean AU activation range | 0.576137 | 0.378216 |
| Valence mean range | 0.864964 | 0.371976 |
| Arousal mean range | 0.403376 | 0.254662 |
| Distinct dominant expressions (mean per context) | 4.607706 | 3.866900 |

Both temporal shape diversity and between-teacher mean/semantic differences
shrink. This supports, but does not by itself prove, the mode-identity erosion
hypothesis. The data do not justify attributing the drop to temporal smoothing
alone. All checkpoints, raw-GT checks and context/session results are retained.

## Exploratory follow-up: mode-preserving reference prior

Same objective family, replacing Uniform(Y_s) with p_mode(j | fixed GT):

```
q_tau(p,Y_s,p_mode) = tau * log sum_j p_mode(j) exp(CCC(p,y_j)/tau)
                   = max_pi E_pi[CCC] - tau KL(pi || p_mode)
```

The reference kernel uses time-order-invariant 50D mean/std descriptors of
AU/VA/expression, scaled by full TRAIN feature std (floor 0.03). It is centered
on each teacher's unchanged fixed T1 GT. An adaptive kernel bandwidth gives
prior entropy log(8) (effective eight references), using only TRAIN distances.
Exact descriptor duplicates retain symmetry even if effective support cannot
be made as small as eight. CCC still scores all session references, not only
eight selected trajectories. Temperature remains 0.05.

The uniform prior reproduces the previous method; a point-mass prior recovers
fixed-reference CCC. These limits, normalization, entropy and duplicate behavior
are unit tested. This makes one controlled family with the same fixed MSE and
inference architecture, not a teacher-output mixture. Effective count8 is one
prespecified follow-up after observing uniform-prior loss of diversity, not a
held-out confirmation or a claim of optimal hyperparameters.

Code: `train_mode_reference_teacher.py`; artifacts
`reports/mode_reference_teacher_seed1/`. The original uniform run has its own
immutable code snapshot and all old weights remain retained.

Diagnostics from the original mode-prior run (retained separately from v2):

| Full VAL571 | T1 | Uniform prior | Mode prior (effective8) |
|---|---:|---:|---:|
| Processed FRC | 1.072308 | 2.038602 | 1.370734 |
| exact FRD | 96.808993 | 95.408503 | 95.771380 |
| Raw FRC | 1.033339 | 1.982976 | 1.347655 |
| FRDiv | 0.205423 | 0.091648 | 0.190257 |
| Time-mean diversity component | 0.074945 | 0.031695 | 0.070775 |
| Temporal-residual diversity component | 0.130479 | 0.059953 | 0.119482 |
| Mean AU activation range | 0.576137 | 0.378216 | 0.568062 |
| Valence mean range | 0.864964 | 0.371976 | 0.857929 |
| Arousal mean range | 0.403376 | 0.254662 | 0.363239 |
| Distinct dominant expressions | 4.607706 | 3.866900 | 4.618214 |

This is consistent with mode priors retaining behavioral separation while
permitting a less rigid temporal reference. It is stronger evidence than merely
adding output noise, but remains a single-seed exploratory result on repeatedly
used validation data. Exact FRD and the corrected-kernel reproduction have now
passed; the final corrected metrics are given at the top of this report.

For this original mode-prior bank, session-bootstrap difference CIs against T1
are FRC [+0.2367,+0.3302] and exact FRD [-1.7268,-0.0731]. FRC improves on 96.8%
of contexts and in 19/20 session means (session0 is worse). FRD improves on 66.2%
of contexts. These are conditional, single-seed, unadjusted exploratory intervals,
not evidence of a universal per-context Pareto improvement. Mean FRDiv falls by
0.01517 but remains inside the prespecified [0.17,0.20] band.

## Numerical boundary audit and reproduction

After the first two banks finished training, an additional boundary test found
that the initial CCC implementation masked gradients at exactly constant
prediction channels. Forward CCC was correctly zero, but for a varying target
its covariance derivative should remain available. The issue was corrected by
masking only constant **reference** channels, not constant predictions.
The new test matches the original Mam loss's nonzero gradient at constant
prediction (test norm 0.3090 instead of the old erroneous zero).

The earlier single-reference gradient parity test covered nonconstant inputs;
it did not establish this boundary property. Full VAL571 prediction banks of
T1 and the uniform probe had zero exactly-constant-channel fraction, including
a BF16-quantized FP32-output proxy, but this does **not** prove the branch was
never exercised during training. Consequently the promising mode-prior bank was
retrained with the corrected kernel in
`reports/mode_reference_teacher_v2_seed1/`, without overwriting the original.

The reproduction compares all nine model configurations and every final model
tensor. Only exact equality plus unchanged model/evaluator code permits reuse
of the original evaluation with explicit provenance; otherwise the full
evaluator is rerun. Existing measured values remain attributed to the original
evaluated checkpoints. This guard required both the weight check and fresh full
exact FRD evaluation before accepting the corrected candidate.

All nine corrected-run models completed with matching initialization and
source/crop hashes. Their final tensors are not bitwise identical to the first
run, so **no evaluation reuse occurred**; a new full evaluator was launched.
Fast full-VAL571 values reproduce the original operating point closely:
FRC 1.370792 and FRDiv 0.190244. Existing repeated BF16 training runs also show
small non-bitwise differences, so any difference cannot automatically be
attributed to the boundary fix alone.

A real CUDA autocast audit confirmed that AU/VA branches are BF16 but expression
softmax is FP32; concatenation makes the final 25D prediction FP32 even in the
original model. Both original and new CCC therefore receive FP32 predictions;
this is not a BF16-to-FP32 output-precision intervention.

Interleaved warmed inference on GPU1, four fixed VAL inputs, 12 repetitions each,
batch1/FP32 and including output copy: T1 37.18ms, corrected mode-prior 37.33ms
(means). Both have 37,201,290 parameters and zero selector overhead. This is a
paired timing check on a shared server, not an exclusive hardware benchmark.

Measured training cost over the nine specialists: original T1 averaged 92.76s
per 600 updates (206.98 samples/s), corrected mode-prior 228.43s (84.05 samples/s).
Average peak allocated memory was 6.663GiB versus 6.983GiB. This is about 2.46x
training wall time at identical update count, not an equal-FLOP or equal-time
comparison. Preprocessing/startup and shared-server load also affect timing.
The unchanged loss budget means one fixed MSE target and no training DTW;
it does **not** mean unchanged CCC-reference comparisons or total compute.

All nine specialists improve their full-VAL FRC; the reused q0 stays identical:

| Slot | T1 FRC contribution | Corrected mode prior |
|---|---:|---:|
| q0 | 0.122621 | 0.122621 |
| q1 | 0.104350 | 0.149371 |
| q2 | 0.088463 | 0.111847 |
| q3 | 0.105838 | 0.144459 |
| q4 | 0.168322 | 0.204855 |
| q5 | 0.132647 | 0.192437 |
| q6 | 0.065489 | 0.103714 |
| q7 | 0.120333 | 0.135194 |
| q8 | 0.080577 | 0.105426 |
| q9 | 0.083667 | 0.100866 |

Thus the aggregate gain is not a single cherry-picked teacher replacement.

## Reference-use mechanism diagnostic (TRAIN64 only)

Using 64 deterministic TRAIN speaker contexts (seed1234, center crops), frozen
models were inspected without optimization. q0 is excluded here. For each of
the nine specialists, the posterior over raw same-session GT was computed using
its training objective. Distances use the same TRAIN-scaled 50D descriptor.
These are post-hoc training-set diagnostics, not validation metrics.

| TRAIN diagnostic | T1 fixed | Uniform reference | Corrected mode prior |
|---|---:|---:|---:|
| Soft prediction-to-fixed-mode descriptor MSE | 1.3974 | 1.4769 | 1.3849 |
| Rounded prediction-to-fixed-mode descriptor MSE | 1.4154 | 1.6934 | 1.4435 |
| Fixed-target trajectory MSE | 0.12745 | 0.14786 | 0.13198 |
| Fixed-target CCC (per specialist) | 0.12713 | 0.04097 | 0.10433 |
| Best raw session CCC (per specialist) | 0.16558 | 0.23206 | 0.18906 |
| Posterior expected mode displacement | 0.0000 | 2.5718 | 0.5558 |
| Posterior effective references | 1.00 | 37.55 | 5.62 |
| Posterior mass on original fixed GT | 1.0000 | 0.0167 | 0.6610 |
| Posterior argmax is original fixed GT | 100% | 2.95% | 84.90% |

The mode prior allows reference flexibility without typically abandoning the
original teacher's behavioral target. Uniform references place almost no mass
on that original target and exhibit much larger behavioral displacement. This
supports the proposed explanation, but the posterior properties partly reflect
the imposed prior by construction; they are not independent causal proof.
Rounded mode-descriptor error is slightly worse than T1, so the evidence is
mode preservation in broad terms, not exact recovery of every target statistic.

## Scope of interpretation

This is not a proof of identifiable semantic/temporal disentanglement. CCC also
depends on means and variances; the retained fixed MSE penalizes the full
trajectory, not only a semantic mode. The 50D statistics are operational behavior
proxies, not human-annotated semantic ground truth. What is demonstrated is a
controlled change to reference supervision, with measurable mode preservation
and a favorable requested operating point on this validation protocol.

No student was retrained, no graph/backbone architecture changed, and no LLM
annotation or inference-time GT/selector was introduced. Existing student routes
and every baseline checkpoint remain available. Further paper-level work should
separately test additional seeds/held-out data and targeted reference-prior
ablations; the present exploratory validation results are not a SOTA claim.
