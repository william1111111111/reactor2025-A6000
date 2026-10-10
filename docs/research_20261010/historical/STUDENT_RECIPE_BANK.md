> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Frozen student recipes for future teacher upgrades

Frozen on 2026-09-24. Weights, configs, source snapshots and evaluation ledgers
are retained in persistent storage. No checkpoint was overwritten or deleted.
The local-only registry records absolute artifact paths and verified SHA256s.

Update 2026-09-30: the table below is the historical 2026-09-24 recipe bank,
not the latest teacher/student operating points. See the
[consolidated component ablations](COMPONENT_ABLATION_RECORD_20260930.md) for
new mode-reference teachers, two-dataset velocity/residual deletion studies
and the current proposed Base + centered-residual student without velocity.
Historical checkpoints and their original results are unchanged.

## Retained operating points

All rows: REACT2025 full VAL571, all-session GT, official AU rounding, exact
rolling FRD. These are single-seed local VAL results, not official sampled-10
challenge TEST scores. Each student has 3,803,201 parameters and ten outputs.

| ID | Role | Training | FRC ↑ | exact FRD ↓ | FRDiv ↑ |
|---|---|---|---:|---:|---:|
| S-basic | mandatory KD comparator | basic KD, 10 epochs | 1.122329 | 100.690347 | 0.189725 |
| S-residual | main first-stage recipe | centered-residual KD, 10 epochs | 1.121615 | 96.617181 | 0.183968 |
| S-continuation | matched fine-tuning control | S-residual + 5 epochs, unchanged loss | 1.133005 | 96.705597 | 0.187008 |
| S-logit | current conservative refinement candidate | S-residual + 5 epochs AU logit contrast | 1.132541 | 96.474092 | 0.186153 |

S-logit improves all three point estimates versus S-residual, but has lower
FRDiv than the equal-budget S-continuation. Its incremental effect is a
quality/diversity tradeoff, not evidence that the new AU loss raises diversity.
S-continuation preserves frozen T1 teacher quality, but its FRD is slightly
worse than S-residual. S-basic has higher diversity but is not quality-safe
relative to T1. Other previous students remain retained as negative/frontier
controls; this table is an index, not a deletion/retention filter.

## Recipe contracts

- Shared ExpertConditionalREGNN; ten canonical expert IDs, no noise or
  inference scaling; no teacher network needed at student inference.
- Base loss: MSE + (1-CCC) + 0.05 velocity against soft teacher predictions.
- S-residual adds teacher-aligned signed residuals around the ten-slot mean,
  normalized by fixed TRAIN-derived AU/VA/expression scales.
- Original residual coefficient: 0.2697612302992961; it came from a TRAIN
  gradient-ratio calibration, not a universal weight for all future teachers.
- First stage: seed 1, batch 4, AdamW lr=1e-4, betas=(0.9,0.99), wd=1e-4,
  10 epochs / 8,300 updates, cosine decay, BF16, clip norm 1.
- Refinement: identical optimizer reset, five epochs / 4,150 more updates,
  lr=2e-5, same source order across arms. S-logit uses clipped AU log-odds,
  slot centering and Huber loss; its original auxiliary weight is
  0.10262709644833162, also calibrated on TRAIN only.

## Reuse with a stronger teacher

1. Freeze the new teacher checkpoints and hash a new TRAIN-only prediction
   cache. Never substitute VAL-GT oracle outputs into student training.
2. Preserve slot identity across contexts. Do not average different teachers
   into one trajectory or shuffle expert targets inconsistently.
3. Compare old/new teacher targets with identical student initialization,
   source views/order, update budget and deterministic evaluation protocol.
   Changing to each teacher's own anchor initialization would be a second
   experimental factor and must be a separate arm.
4. Reuse the calibration procedure, not blindly the old teacher's numeric
   scales/coefficients. Calibration may read TRAIN only.
5. First compare S-basic and S-residual at ten epochs. If refining, include
   both S-continuation and S-logit at the same total 15-epoch budget.
6. Report teacher and student FRC/FRD/FRDiv together. FRDiv retention alone is
   not a measure of semantic correctness or a guarantee of quality.

## Checkpoint SHA256s

- S-basic: `9ae95f39936bed601e8326588ba52a46388645508e31a9d524524b055680404d`
- S-residual: `5ef5a3a3a69e0314e1d643286a48c3f8bd98a0f935a41304c3ecc35fab8a7d21`
- S-continuation: `e15c8219c0442a572630d42e55899106b67d8468f6a5c1a92caa49c372ab3720`
- S-logit: `c867910592225a845750c600fd00c92cbcf45e6c012350666bcdf9716565e328`

Evidence: [initial KD comparison](T1_STUDENT_DISTILLATION.md) and
[matched refinement](T1_STUDENT_REFINEMENT.md).
