> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# CVPR2027 exploratory conditioning audit

Interpretation correction (2026-09-27): see PROTOCOL_AUTHORITY_CORRECTION.md.
The following is the historical first-audit protocol, not a mandate to replace
the paper-defined all-session reference set with code-level sampled references.

Registered 2026-09-27 before new intervention scores. Question: do frozen
reaction sets carry context-specific information beyond session priors, and
does all-session FRC detect it? Reused VAL only, not independent TEST or human
semantic validation. Existing weights, official metrics and caches stay read-only.

1. Audit REACT2025 mode-reference contrast student and REACT2024 count4 minimal
   student. Verify archived prediction/checkpoint/source hashes.
2. Derange complete predicted sets within role/session/exact-length groups using
   fixed seeds 2701,2702,2703. Exclude singleton groups from matched comparisons.
   Marginal diversity is exactly preserved. For deterministic source-only models
   this equals swapping inputs, subject to fresh-inference parity checks.
3. Compare all-session FRC, deterministic paired+9 FRC (diagnostic, NOT the
   challenge's fixed reference manifest), and synchronized paired-crop CCC,
   including slot0. Use 2000 paired session-cluster bootstrap resamples.
   Three intervention seeds are NOT three independent model-training seeds.
4. Run at most 96 deterministic session-balanced examples per dataset on GPU5
   (GPU1 fallback, leave occupied GPU6 untouched). Interventions: original,
   same-group donor, cyclic temporal shift, temporal-mean audio, temporal-mean
   visual streams. Keep weights, length and expert IDs fixed. Mean/shift inputs
   may be out-of-distribution; sensitivity is not semantic appropriateness.
5. Conditional information with weak all-session sensitivity motivates improving
   supervision/evaluation. No signal motivates checking alignment/conditioning
   before teacher retraining. No blind parameter sweep, fabricated human labels,
   remote API/media upload, or training is included in this first bounded audit.

Budget: two cached full-VAL audits plus <=2*96*5 forward passes and smoke checks.
New versioned outputs go in this directory's runs/. Missing/mismatched sources or
caches fail closed. The original project is read-only; its /public symlink target
did not permit apply_patch writes in this environment, even after escalation.

## Analysis clarification after initial runs

The REACT2024 conversion encodes P1/P2 as separate directional session names.
Donor matching correctly remains within direction. For final uncertainty,
`analyze.py` merges the two roles of a recording before cluster bootstrap
(19 recording clusters, rather than the preliminary 38 directional groups).
No point estimate, sample selection, prediction, or original result is changed.
The original execution protocol hash is preserved in each run manifest.
The final analysis additionally verifies processed-GT multiset equality within
matching groups, and reports non-anchor slots separately as exploratory analysis.
