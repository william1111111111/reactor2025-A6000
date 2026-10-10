> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Matched T0/T1 teacher convergence: 600 → 1200 → 2400 updates

REACT2025 full VAL571, center750, deterministic all-session GT, AU rounded, exact rolling FRD

| Teacher | Updates/model | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar |
|---|---:|---:|---:|---:|---:|
| T0 | 600 | 1.373312 | 84.928192 | 0.036411 | 0.036259 |
| T1 | 600 | 1.072308 | 96.808993 | 0.205423 | 0.068098 |
| T0 | 1200 | 1.453088 | 84.920676 | 0.043459 | 0.040753 |
| T1 | 1200 | 1.133364 | 97.645893 | 0.226178 | 0.078040 |
| T0 | 2400 | 1.524074 | 84.696737 | 0.062817 | 0.047203 |
| T1 | 2400 | 1.024060 | 98.647957 | 0.238767 | 0.078252 |

## Main decision

Strict zero-slack T1 improvements over the matched 600-update baseline: none.
Highest-FRDiv T1 checkpoint preserving baseline FRC and exact FRD: `T1_600`.
T0 point-estimate improvements over its own matched 600-update baseline: T0_1200, T0_2400.
Do not equate outperforming T1_600 with reaching T0 quality. The matched T0 curve is shown at every budget.
Any selected checkpoint is VAL-selected and single-seed; this is not independent confirmation.

## Controlled protocol

- Ten independent ConditionalREGNN teachers per arm; 3,720,129 parameters each, 37,201,290 per ensemble.
- One fixed TRAIN target per source per teacher; original T0/T1 manifests are SHA256-verified. Rank0 remains the true paired anchor.
- Seed1, batch32, BF16, AdamW lr1e-4, betas(0.9,0.99), wd1e-4, clip norm1, original 50-epoch cosine schedule.
- MSE + (1-CCC) + 0.05 velocity, retaining original relative_time_masked policy: non-synchronous cross-listener velocity supervision remains disabled.
- Fresh uninterrupted runs from the original initialization; no weight-only warmstart, scheduler reset or epoch-only resume.
- Step snapshots do not consume RNG, update the optimizer or advance the scheduler. Initial states and observed source/crop-order hashes match across all20 teachers at every milestone.
- Each model sees exactly600/1200/2400 updates at the reported checkpoints. The nominal50-epoch schedule is not a claim of50 completed epochs.
- Twenty runs ×2400 updates =48000 new optimizer updates total. Relative to the600-step teacher point, the final point uses4× training updates; no increased-budget gain is called an architectural gain.
- All six ensemble points receive full571-context exact FRD. No best-rank checkpoint mixing, inference scaling, noise or selector.
- Original teachers and frozen student recipes remain untouched. This experiment does not retrain students.

## T0 learning-curve deltas versus fresh T0_600

| Updates | ΔFRC [session bootstrap95% CI] | ΔFRD [95% CI] | ΔFRDiv [95% CI] |
|---:|---|---|---|
| 1200 | +0.079776 [-0.013412, +0.133454] | -0.007515 [-1.168418, +0.947976] | +0.007048 [+0.004523, +0.009534] |
| 2400 | +0.150762 [+0.045444, +0.219880] | -0.231455 [-1.475609, +0.499783] | +0.026406 [+0.018903, +0.034320] |

## T1 learning-curve deltas versus fresh T1_600

| Updates | ΔFRC [session bootstrap95% CI] | ΔFRD [95% CI] | ΔFRDiv [95% CI] |
|---:|---|---|---|
| 1200 | +0.061056 [+0.025333, +0.107187] | +0.836899 [-0.421893, +1.635381] | +0.020755 [+0.017694, +0.024961] |
| 2400 | -0.048248 [-0.097203, +0.018464] | +1.838964 [+0.549861, +3.303724] | +0.033343 [+0.028386, +0.039655] |

## Per-rank FRC contribution

| Arm / updates | q0 … q9 |
|---|---|
| T0_600 | 0.12261, 0.12319, 0.14371, 0.14900, 0.15186, 0.14216, 0.12717, 0.14190, 0.13394, 0.13777 |
| T1_600 | 0.12262, 0.10435, 0.08846, 0.10584, 0.16832, 0.13265, 0.06549, 0.12033, 0.08058, 0.08367 |
| T0_1200 | 0.13422, 0.13291, 0.14161, 0.15384, 0.15663, 0.15778, 0.13516, 0.14255, 0.14155, 0.15684 |
| T1_1200 | 0.13426, 0.11853, 0.09530, 0.12047, 0.15484, 0.14994, 0.08840, 0.10552, 0.09296, 0.07315 |
| T0_2400 | 0.13782, 0.14950, 0.16396, 0.16099, 0.15641, 0.14548, 0.14095, 0.15661, 0.15387, 0.15847 |
| T1_2400 | 0.13778, 0.09964, 0.09005, 0.10481, 0.13026, 0.13971, 0.07967, 0.09266, 0.07639, 0.07310 |

## Output diversity composition

| Arm / updates | AU | VA | Expression | Dynamic component | FRDiv P10 / P50 / P90 |
|---|---:|---:|---:|---:|---|
| T0_600 | 0.033171 | 0.001427 | 0.001813 | 0.029258 | 0.027526 / 0.036191 / 0.045166 |
| T1_600 | 0.171013 | 0.011130 | 0.023281 | 0.130479 | 0.177910 / 0.201942 / 0.238539 |
| T0_1200 | 0.038746 | 0.001731 | 0.002981 | 0.035467 | 0.030687 / 0.042104 / 0.057225 |
| T1_1200 | 0.186979 | 0.011353 | 0.027848 | 0.151199 | 0.201126 / 0.224872 / 0.254690 |
| T0_2400 | 0.054656 | 0.002530 | 0.005632 | 0.050736 | 0.038424 / 0.059913 / 0.090391 |
| T1_2400 | 0.192964 | 0.012723 | 0.033080 | 0.153272 | 0.207254 / 0.240900 / 0.267735 |

| T1 interval | Total diversity change | Dynamic change | Static change |
|---|---:|---:|---:|
| 600_to_1200 | +0.020755 | +0.020720 | +0.000035 |
| 1200_to_2400 | +0.012588 | +0.002073 | +0.010515 |

Static means the temporal mean of each slot's deviation from the ensemble mean; dynamic is its time-varying residual. This algebraic decomposition does not certify semantic correctness or source compatibility.

## Frozen600 reproduction

| Arm | ΔFRC | Δexact FRD | ΔFRDiv | Bitwise-identical rank checkpoints |
|---|---:|---:|---:|---:|
| T0 | -0.00001948 | -0.00093622 | -0.00000377 | 0/10 |
| T1 | +0.00009288 | -0.00490279 | +0.00000814 | 0/10 |

Fresh600 is the causal baseline for the learning curve; parameter-level differences and all raw hashes are retained in the local ledger.

## Compute / provenance

| Arm / updates | Mean updates/s/model | Sum training seconds | Diagnostic ensemble inference ms |
|---|---:|---:|---:|
| T0_600 | 6.47 | 927.8 | 37.15 |
| T1_600 | 6.47 | 927.3 | 36.99 |
| T0_1200 | 6.52 | 1841.7 | 37.21 |
| T1_1200 | 6.52 | 1840.9 | 37.25 |
| T0_2400 | 6.54 | 3667.9 | 37.22 |
| T1_2400 | 6.55 | 3666.1 | 37.09 |

Candidate pool/output size:10. Temporal target budget:one fixed target per source/model. Selector overhead:zero.
Base commit: `c421f509f8cb7997e9341592cae2eb1e721b0e8e`; working-source hashes, target manifest hashes, initialization hashes and actual source/crop-order hashes are stored locally.
Weights, target identities, trajectories and per-context evaluation data remain local; this report is aggregate-only.
