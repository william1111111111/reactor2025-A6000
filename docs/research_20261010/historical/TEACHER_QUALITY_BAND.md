> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Teacher quality at FRDiv 0.17–0.20

REACT2025; deterministic all-session opposite-role GT, existing processed-GT cache, official AU rounding, exact rolling DTW. FRC is the established sum over ten outputs. This is not official sampled-ten-GT evaluation.

A fixed 30-arm VAL64 grid chooses at most three global recipes. Parameters are fixed across contexts; inference uses speaker-conditioned teacher outputs only. q0 remains the paired anchor. No new teacher training. VAL507 excludes pilot contexts but shares sessions and has been used in earlier studies: exploratory evidence, not an independent TEST claim.

| Scope | Recipe | FRC ↑ | exact FRD ↓ | FRDiv | FRVar | Raw-GT FRC ↑ | Goal pass |
|---|---|---:|---:|---:|---:|---:|---|
| VAL64 | baseline | 1.119153 | 96.648585 | 0.203003 | 0.069049 | 1.058569 | False |
| VAL64 | T1_600_replace_q9 | 1.191887 | 94.513099 | 0.192572 | 0.066284 | 1.125129 | True |
| VAL64 | T1_1200_blend0.25 | 1.164969 | 88.878036 | 0.170636 | 0.064897 | 1.130947 | True |
| VAL64 | T1_600_replace_q2 | 1.190646 | 95.754409 | 0.193900 | 0.067123 | 1.132276 | True |
| VAL507 | baseline | 1.066395 | 96.829242 | 0.205729 | 0.067978 | 1.030154 | False |
| VAL507 | T1_600_replace_q9 | 1.141459 | 94.520175 | 0.195137 | 0.065540 | 1.103911 | True |
| VAL507 | T1_1200_blend0.25 | 1.163174 | 89.385251 | 0.175584 | 0.065327 | 1.138957 | True |
| VAL507 | T1_600_replace_q2 | 1.142403 | 95.730735 | 0.196821 | 0.066308 | 1.113976 | True |
| VAL571 | baseline | 1.072308 | 96.808993 | 0.205423 | 0.068098 | 1.033339 | False |
| VAL571 | T1_600_replace_q9 | 1.147111 | 94.519382 | 0.194850 | 0.065623 | 1.106289 | True |
| VAL571 | T1_1200_blend0.25 | 1.163376 | 89.328401 | 0.175030 | 0.065279 | 1.138059 | True |
| VAL571 | T1_600_replace_q2 | 1.147810 | 95.733388 | 0.196493 | 0.066399 | 1.116027 | True |

## Frozen global recipes

Blend: for q1–q9, y=(1-alpha)*T1 + alpha*T0_2400 before AU rounding; q0 uses T1 unchanged. Replacement: use the matching-rank T0_2400 model at one fixed slot and T1 models elsewhere. Different banks have different fixed training targets, so blend is output fusion, not weight interpolation.

- `T1_600_replace_q9`: 10 models, 37,201,290 parameters; recipe/checkpoint hashes in the local inference manifest.

- `T1_1200_blend0.25`: 19 models, 70,682,451 parameters; recipe/checkpoint hashes in the local inference manifest.

- `T1_600_replace_q2`: 10 models, 37,201,290 parameters; recipe/checkpoint hashes in the local inference manifest.

## Distribution and uncertainty

| Full VAL arm / metric | Paired delta | Session-bootstrap 95% CI |
|---|---:|---|
| T1_600_replace_q9 / FRC | +0.074803 | [+0.052347, +0.102386] |
| T1_600_replace_q9 / exact_FRD | -2.289611 | [-2.992344, -1.970169] |
| T1_600_replace_q9 / FRDiv | -0.010573 | [-0.012263, -0.009295] |
| T1_600_replace_q9 / raw_FRC | +0.072950 | [+0.052319, +0.095657] |
| T1_1200_blend0.25 / FRC | +0.091068 | [+0.057174, +0.136432] |
| T1_1200_blend0.25 / exact_FRD | -7.480593 | [-9.595169, -5.846857] |
| T1_1200_blend0.25 / FRDiv | -0.030394 | [-0.033372, -0.025158] |
| T1_1200_blend0.25 / raw_FRC | +0.104720 | [+0.063133, +0.152890] |
| T1_600_replace_q2 / FRC | +0.075502 | [+0.056014, +0.087200] |
| T1_600_replace_q2 / exact_FRD | -1.075605 | [-1.749215, -0.669472] |
| T1_600_replace_q2 / FRDiv | -0.008930 | [-0.009663, -0.008246] |
| T1_600_replace_q2 / raw_FRC | +0.082688 | [+0.060225, +0.098245] |

Full context metrics, per-session summaries, quantiles and input hashes are retained locally. Baseline FRC/FRD is reproduced for every context before scoring recipes.

Budget: zero new training updates / temporal-loss calls; internal model count 10 for replacement or 19 for blend; final output count10. Replacement has the same parameter count and forward count as baseline. Blend costs nine extra teacher forwards. Runtime selector overhead is zero (fixed recipe).

Cached-output composition: 0.334 ms/recipe/context on CPU, excluding model inference and evaluation. No new end-to-end GPU latency benchmark is claimed.

## Actual model inference verification

After metric evaluation, `teacher_quality_ensemble.FixedTeacherEnsemble` loaded the actual checkpoints and generated eight fixed VAL inputs on GPU5. Both q9 replacement and 25% blend reproduced cached evaluation trajectories **bitwise**, maximum absolute error0.0. The forward interface accepts only speaker audio, emotion, 3DMM and source length.

With three warmup forwards, FP32 batch1 inference including the output CPU copy took 34.58ms/context for q9 replacement and 64.08ms for blend (same eight inputs). These are short diagnostic measurements on a shared server, excluding feature loading; not full pipeline benchmarks.

Historical optimization budgets also differ: the q9 replacement uses nine T1_600 checkpoints plus one T0_2400 checkpoint (7,800 total source-model updates versus6,000 for T1_600); the blend uses ten T1_1200 plus nine T0_2400 checkpoints (33,600 total). No new updates were performed in this experiment. Do not attribute these results to an equal-training-budget architectural improvement.

The q9 recipe is the cost-preserving operating point selected from the pilot; the blend is a higher-compute quality alternative. Both satisfy the mean FRC/diversity goal on VAL507 as well as VAL571. This experiment has not retrained a student or incorporated the previous learned VA corrector.

Persistent source checkpoint hashes were verified for all29 distinct models referenced by the three recipes. The local `inference_recipes.json` retains every checkpoint path/hash; `FixedTeacherEnsemble(..., checkpoint_root=...)` supports relocation to the persistent convergence archive.

Base git SHA `c421f509f8cb7997e9341592cae2eb1e721b0e8e`; experiment source SHA256 `95e27d15a6758682edbac57737e1a99a76e2404759e97e567e8bb513cbc11f67`. The full-VAL result is validation-selected and cannot establish independent generalization.
