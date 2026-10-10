> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Conditional slot assignment: TRAIN-only falsification probe

Status: **not promoted to teacher training**. REACT2025, both TRAIN roles,
3,320 contexts, 17 sessions. No VAL labels or metrics used.

The T1 Max-FRDiv target set and native paired q0 are kept exactly unchanged;
only q1..q9 are reassigned by Hungarian matching. Each slot has a separate
source-conditioned ridge descriptor predictor (source mean/std audio, emotion,
3DMM; fold-fitted scaling/PCA64; ridge mean-loss coefficient 0.1).
Regression is four-fold cross-fitted by session, keeping both roles together.
Existing T1 global TRAIN descriptor statistics/prototypes are inherited; these
are not refitted per fold. Assignment adds 0.25 times frozen prototype cost.

The shuffled-source control deranges predictions within each source role/session.
A constant-slot predictor is a second, offline diagnostic control.

| TRAIN out-of-fold diagnostic | Value |
|---|---:|
| Conditional descriptor MSE | 3.568169 |
| Shuffled-condition descriptor MSE | 3.567238 |
| Constant-slot descriptor MSE | 1.235448 |
| Conditional reassigned non-anchor slots | 32.26% |
| Shuffled reassigned non-anchor slots | 31.96% |
| Constant reassigned non-anchor slots | 25.14% |

There is no credible positive input-conditioned prediction signal in this probe.
Assignment changes alone do not establish learnability. Eighteen GPU teacher
retrains were therefore **not launched**. This falsifies this specific pooled
feature/ridge implementation, not all conditional assignment methods.

## Mathematical negative control

For a fixed target set and squared-distance bijection,
`sum_k ||y_pi(k) - (z_k + h(x))||²`, all terms containing the common shift
`h(x)` are independent of the permutation. A shared additive conditional shift
cannot change the matching. A unit test verifies this; mode-specific conditional
functions are required for an actual intervention.

Four tests passed: common-shift invariance, mode-specific set preservation,
heldout-session regression target isolation, and sham derangement integrity.
Detailed manifests, input hashes, fold results and OOF predictions are in
`reports/conditional_slot_seed1/`. Git base c421f509; dirty source SHA recorded
in the run summary. No new weights, inference changes or VAL claims.

## Next preregistered mechanism probe

**Latent-reference temporal supervision**: retain each T1 teacher's original
fixed-GT MSE (and unchanged velocity mask), but replace its single-reference CCC
with a normalized log-sum-exp over raw opposite-role TRAIN-session references.
Each reference is relative-time resampled to the valid source length. Temperature
0.05, unchanged total CCC coefficient 1.0. No extra prediction heads, no GT at
inference, no output blending and no direct FRDiv objective.

Hypothesis: a diverse target defines a behavioral mode but is not necessarily
the only admissible temporal reference. The fixed MSE preserves mode anchoring;
latent-reference CCC relaxes the arbitrary unique-reference temporal penalty.
This is a hypothesis, not a novelty or performance claim.

First compare one complete nine-specialist + reused q0 bank to matched T1_600
and T0_600. Same initialization, optimizer, source/crop schedule, 600 steps and
batch32; verify hashes. One MSE target remains per prediction, but CCC dot-product
comparisons increase from one to all TRAIN-session GT and that additional cost
must be reported. No DTW is added to training. Evaluate full VAL571 all-session
FRC/FRDiv and exact FRD using the existing deterministic cache. Follow with a
conditioning/aggregation ablation only if this first probe shows useful signal.
