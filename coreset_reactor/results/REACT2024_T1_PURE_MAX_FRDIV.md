# React2024 T1 pure Max-FRDiv teacher experiment

## Protocol

- Dataset: converted React2024, 25-D facial attributes.
- Training target: one true paired listener plus nine fixed, distinct
  same-session listener GTs selected by the TRAIN pure Max-FRDiv objective.
- T0 control: same paired anchor plus nine fixed same-session GTs from the
  matched random target manifest.
- Models: ten independent ConditionalREGNN models per arm, one output per
  model; same seed, initialization, source schedule, optimizer and training
  budget.
- Training: 600 optimizer steps, nominal 50-epoch cosine schedule, batch 32,
  BF16, `relative_time_masked` alignment, 3,720,129 parameters/model.
- Strict training eligibility: only sessions with at least ten distinct GTs;
  the 18 RECOLA directional TRAIN contexts with nine GTs were excluded from
  training and recorded in provenance.
- Evaluation: full React2024 VAL (1,124 source contexts), center 750-frame
  crops, all same-session opposite-role VAL GTs. The evaluation target is not
  the fixed training GT path.
- Evaluation seed: `1234`.

## Full VAL results

| arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | TLCC |
|---|---:|---:|---:|---:|---:|
| T0 random fixed GT | 0.244023 | 83.538740 | 0.042493 | 0.017756 | 49.000000 |
| T1 pure Max-FRDiv | **0.620408** | 87.107812 | **0.157259** | **0.061897** | 49.000000 |
| T1 − T0 | **+0.376386** | +3.569072 | **+0.114766** | **+0.044142** | 0 |

Additional aggregate diagnostics:

| arm | GT descriptor coverage ↓ | prediction validity ↓ | candidate utilization |
|---|---:|---:|---:|
| T0 | 1.198458 | 1.273650 | 0.139858 |
| T1 | **1.097981** | **1.051754** | **0.434431** |

The TRAIN target-set statistics were:

| target set | FRDiv | descriptor pair distance |
|---|---:|---:|
| T0 | 0.140743 | 0.868272 |
| T1 | **0.165025** | **0.949767** |
| T2 | 0.166194 | 0.953254 |

## Interpretation

Under this matched 600-step pilot, pure Max-FRDiv target selection strongly
improves the diversity of the ten-model output set and also improves the
all-session descriptor coverage. FRC rises substantially relative to the
matched random fixed-target control. The quality trade-off is visible in
exact FRD: it increases by 3.57 under the same evaluation protocol, so this
is not a strict quality-and-diversity Pareto win yet.

This is a pilot result, not a full 50-epoch convergence claim. The native
React2024 evaluator's separate S_MSE/FRDvs protocol was not used here; the
reported metrics are the repository's matched all-session GT protocol used
for the fixed-pair teacher experiments.

## Provenance

- Training run: `/tmp/react2024_t1_runs_600_v3`
- T0 JSON: `/tmp/react2024_t0_full.json`
- T1 JSON: `/tmp/react2024_t1_full.json`
- Data/target-manifest source SHA: `7c114cc7101beea7f706bee9199cbd0fe859694b1ff35fe6b47e9cc44f6a1af9`
- Base Git commit at launch: `21a9809151b5fc59403cdf8bac0ebd1a43824207`
