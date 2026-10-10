> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Efficient mode-preserving reaction-set distillation: research plan

Status: 2026-09-20. Local exploratory research, not a novelty or publication claim.
All existing checkpoints remain frozen. No data, weights or reports are pushed.

## Central research question

Can a shared student preserve the **conditional behavioral structure** of a
multi-expert reaction set, rather than merely its average response or total
variance, while reducing deployment cost?

The intended backbone is TRAIN-derived behavioral representatives -> fixed
experts -> teacher-aligned shared student -> ten valid reaction trajectories.
The output is a deterministic conditional set, not a calibrated probability
distribution. Ten constructed prototypes are not ten proven natural emotions.

## Evidence and the unresolved issue

All numbers below use full VAL571 speaker contexts, center750, all-session GT.
This is the requested local diagnostic, NOT official challenge-server TEST.

| Model | FRC | exact FRD | FRDiv |
|---|---:|---:|---:|
| Prototype T10 | 1.105138 | 83.160916 | 0.118325 |
| Basic KD | 1.178483 | 85.709485 | 0.060729 |
| Distance KD | 1.123105 | 91.051643 | 0.108257 |
| Distance KD, spread 1.5 | 1.023985 | 102.648340 | 0.155754 |
| Basic KD, calibrated spread 2.406005859375 | 1.018655 | pending | 0.155757 |

The last two points have effectively the same observed diversity. Their FRC
difference is only about 0.00533, before exact FRD and independent-repeat
checks. The relation term cannot be justified from uncalibrated FRDiv alone.
The Basic-KD exact-FRD reference is from an older evaluator without stored GT
fingerprints; cached replay verifies context/path and FRC parity, but final
paper-quality comparisons should regenerate all arms under one frozen version.

## Why amplitude is not enough

For K flattened trajectories of dimension D, define r_k = y_k - mean_j(y_j).
The official squared-distance diversity has the algebraic identity

    FRDiv(Y) = 2 / ((K-1) D) * sum_k ||r_k||^2.

Thus, before domain projection or AU rounding, multiplying residuals by s
multiplies FRDiv by s^2 without adding a new behavioral direction. This is an
algebraic diagnostic, not evidence that any higher-FRDiv set is less realistic.
For the actual reported metric, clipping, probability renormalization and AU
rounding must still be applied and measured; the scale law then need not hold.

We separate three questions:

1. **Support:** do TRAIN-derived teachers cover plausible, context-relevant GT?
2. **Structure:** does compression preserve distinctions between their modes?
3. **Amplitude:** how much of each distinction can be expressed without
   unacceptable quality degradation?

## Experiment R1 (running): scale-invariant centered-direction supervision

Keep MSE + (1-CCC) + 0.05 velocity KD unchanged. Replace only the auxiliary
distance-matching term with a cosine loss on centered slot residuals.

For each sample and AU/VA/expression group:

    R_S[k,t,d] = S[k,t,d] - mean_j S[j,t,d]
    R_T[k,t,d] = T[k,t,d] - mean_j T[j,t,d]
    L_direction = mean_group(1 - cosine(vec(R_S), vec(stopgrad(R_T))))

Only common valid frames are used. Flat teacher groups are skipped. Numerical
floors keep gradients finite even for collapsed students. The existing KD term
still teaches absolute trajectories, magnitudes and common response; the new
auxiliary term does not require matching teacher spread. It does not guarantee
quality, forbid collapse on its own, or invent previously unseen modes.

This is teacher-index-aligned: jointly permuting student and teacher slots is
invariant; permuting only one side is not. Do not call it order-free matching.
The cosine objective itself is an established ingredient; research value must
come from the task-specific decomposition, mechanism evidence and evaluation.

Matched contract:

- Backbone and 3,803,201 parameters unchanged; same seed=1 and teacher0 init.
- Same 3,320 TRAIN cached inputs, crops and teacher outputs, batch4/workers4.
- AdamW LR1e-4, original cosine schedule, BF16, clipping1, 10 epochs / 8300 steps.
- TRAIN-only four-batch gradient calibration targets auxiliary/KD norm ratio .25.
- Fixed auxiliary weight 0.1016126346; observed ratios .20924/.24715/.26121/.25292.
- First training-batch KD loss equals both earlier arms exactly: 0.9414509535.
- No extra teacher inference or exact temporal-GT comparisons in training;
  report actual wall time and peak memory rather than claiming identical FLOPs.
- Independent deployment/checkpoint/optimizer files and epoch snapshots.
- Run: `coreset_reactor/reports/fixed_teacher_distillation/t10_student_direction_pilot10e`.
- Queue: `python -m coreset_reactor.run_direction_evaluation --device cuda:5`.
- Output: `reports/student_direction_comparison.md` after full-VAL fast results,
  then updated after exact FRD. No automatic new training after this experiment.

Tests cover scale/shared-offset invariance, response to wrong/sign-reversed
patterns, teacher stop-gradient, padding, collapsed student, flat teacher,
slot ordering, old-objective equivalence and existing pipeline tests (47 pass).

## Experiment R0 (running): necessary strong post-processing control

Calibrate ONE global scale on Basic-KD VAL predictions to FRDiv=0.155754,
within tolerance1e-4. No per-context GT-dependent scale. Reuse exactly the same
cached all-session GT, AU rounding and metric code. Selected scale is
2.406005859375; full-VAL FRC=1.018655 and FRDiv=0.155757. Exact FRD is running.

Artifacts: `coreset_reactor/reports/fixed_teacher_distillation/t10_basic_spread_matched0155754/`.
Calibration is explicitly VAL-tuned, not a held-out result. Freeze the scale
before a new evaluation split. Reject comparisons with materially different
diversity; provide attainable quality-diversity curves as well as raw points.

## Conditional extensions (not launched)

### R2: data-supported behavioral experts, not outlier experts

Measure representative quality and GT coverage under random/prototype/farthest
target policies. Existing prototypes already use TRAIN-only global clustering,
local representative assignment and a small local-density penalty. Reuse
those controls before adding more machinery. Test whether different target
policies change cross-context slot identity, not only pair distances.

Only if teacher artifacts dominate: add TRAIN-only reliability weights based
on descriptor support and teacher-to-GT agreement. A quality estimator must
be calibrated on held-out TRAIN groups or out-of-fold teacher predictions,
not merely teacher memorization on its own training records. No VAL/TEST GT
input to inference or mode construction. Do not present heuristic weights as
calibrated probabilities. Keep unweighted controls.

### R3: learned, context-dependent amplitude after structure is demonstrated

If R1 improves quality at matched diversity, test an explicit common-response
plus centered-residual factorization, with a small bounded speaker-conditioned
amplitude gate. The gate changes residual strength, not slot identity. Train
from supported teacher/GT trajectories; do not reward arbitrary spread.

Controls: fixed gain, globally VAL-calibrated gain, learned speaker-only gain.
Report whether the learned gate beats the fixed gain on held-out contexts.
Do not turn this into a new backbone plus new loss plus new data experiment.

### R4: within-mode uncertainty only after the deterministic set is useful

Noise is not automatically a source of semantic diversity: a fixed teacher
trajectory for every (speaker, mode) leaves no target variation for noise to
explain. First supply multiple TRAIN-supported targets within each mode;
then test a small conditional latent model with appropriate distributional
training. Compare against the deterministic student and matched output spread.
This is optional, and no stochastic/32-64-candidate model is started now.

## Evidence required before turning this into a paper

- Main ablation: direct GT student / Basic KD / distance KD / direction KD,
  each with and without the same class of globally calibrated post-processing.
- Match initialization, source order, crop/GT handling, output count and budgets.
- Evaluate quality at matched diversity and diversity at matched quality.
  Preserve exact FRC and exact FRD constraints if claiming zero-slack success.
  A successful tradeoff is not a quality-preserving win.
- Distinguish generalization across interaction contexts from memorizing
  session appearance. Split/bootstrap by session where appropriate; do not
  treat correlated clips as independent evidence.
- GT coverage/validity, mode responsibilities, expression/AU/VA pattern
  separation, temporal smoothness and saturation statistics. Video blind
  assessments of appropriateness/synchrony require an actual rendering study.
- Freeze the validation-selected operating point, then evaluate held-out
  data; confirm core comparisons with independent seeds where feasible.
- Public-code-compatible/reference-pool evaluation and all-session diagnostics
  must be reported separately. Do not relabel local VAL as official TEST.
- The current shared student inherits the legacy prototype teacher cache.
  Cross-listener timing/fill/masks need their own corrected matched controls
  before a publication claim about true temporal response correspondence.
- Measure end-to-end ten-output inference latency/throughput/memory under the
  same hardware, precision, batching and warmup; include rendering only if
  claimed. Shared encoding does not remove ten-slot decoder cost. Disclose
  all teacher training/cache cost as well as student deployment savings.

## Decision rules

1. If distance KD is beaten by Basic KD plus scaling at matched diversity,
   record a negative result and do not claim the relation loss is essential.
2. If direction KD is no better after matched-diversity calibration, stop
   expanding auxiliary losses and revisit teacher support/direct GT supervision.
3. If direction KD shifts the quality-diversity frontier and improves actual
   GT coverage, proceed to R2/R3 one at a time, then independent validation.
4. Absolute FRDiv 0.16 is not an acceptance criterion by itself.

## Related work and positioning (preliminary, not exhaustive)

- Park et al., *Relational Knowledge Distillation*, CVPR2019: relation matching
  is established, so neither distance matching nor the general RKD idea is new.
  https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html
- Ma et al., *Multi-Objective Diverse Human Motion Prediction With Knowledge
  Distillation*, CVPR2022: inference-time accuracy/diversity balancing already
  exists; a diversity knob alone is not our proposed contribution.
  https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Multi-Objective_Diverse_Human_Motion_Prediction_With_Knowledge_Distillation_CVPR_2022_paper.html
- Luo et al., *ReactFace*, TVCG (arXiv version revised2024): multiple
  appropriate speaker-conditioned facial reactions is the task context, not
  a new task introduced by this project. Online/offline access must be matched.
  https://arxiv.org/abs/2305.15748

Tentative framing: **Behavior-structured, mode-preserving distillation for
efficient multiple facial reaction generation**. This is a research direction,
not a finalized method name or a claim that current evidence suffices to publish.
