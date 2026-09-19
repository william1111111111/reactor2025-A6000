# CoReSet-Reactor M1: deterministic confirmation and full-VAL result

This report supersedes the final-checkpoint FRC and exact-FRD values in
`M1_MATCHED_LONG_RUN.md`. The prediction-only metrics there remain unchanged.

The reason for the correction is that the repository target post-processor's
EmotionVAE samples a latent even in evaluation mode. The prior standalone
evaluation did not seed that sampling, so FRC and FRD could shift slightly
between invocations. Evaluation now derives one target-alignment seed from
`eval_seed` and the context identifier, restores Python/NumPy/Torch RNG state
after every call, and evaluates the unchanged official per-context FRC/FRD
formulas. Full-VAL multiprocessing serializes NumPy arrays instead of Torch
tensors to avoid exhausting the process file-descriptor limit.

## Matched three-seed VAL64 confirmation

Every run uses 2500 steps, batch size 2, static descriptor weight 0.10 for B3,
and the same initialization, source schedule, exact-GT draws and evaluation
contexts within each paired seed. Evaluation uses `eval_seed=1234`, ten
predictions, AU rounding, and paired plus nine deterministic same-session GTs.

| Seed | Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | GT coverage distance ↓ |
|---:|---|---:|---:|---:|---:|
| 20260918 | B1 | 0.722075 | 87.083040 | 0.006085 | 1.252561 |
| 20260918 | B3 / 0.10 | 0.750812 | 82.047779 | 0.012697 | 1.091207 |
| 20260919 | B1 | 0.770694 | 90.186702 | 0.007326 | 1.244740 |
| 20260919 | B3 / 0.10 | 0.780813 | 83.695377 | 0.014876 | 1.097090 |
| 20260920 | B1 | 0.826106 | 88.678179 | 0.007411 | 1.240434 |
| 20260920 | B3 / 0.10 | 0.835753 | 82.995800 | 0.012721 | 1.088500 |

Mean B3 minus B1: **FRC +0.016168**, **exact FRD -5.736322**,
**FRDiv +0.006491** (1.935 times B1), and **GT coverage distance
-0.153646**. Every seed independently passes all four zero-slack M1 pilot
criteria.

## Full VAL571 confirmation

Following the preregistered order, only seed `20260918` was expanded after the
three-seed VAL64 check passed. Both arms use all 571 VAL contexts under the
same deterministic protocol.

| Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | GT coverage distance ↓ | Prediction validity distance ↓ | Candidate utilization ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 0.798295 | 87.817214 | 0.006505 | 0.033869 | 1.242311 | 0.957168 | 0.565849 |
| B3 / 0.10 | 0.810111 | 82.480136 | 0.013611 | 0.039200 | 1.095160 | 0.785032 | 0.769352 |
| B3 minus B1 | **+0.011816** | **-5.337079** | **+0.007106** | +0.005330 | **-0.147151** | -0.172137 | +0.203503 |

FRDiv is 2.092 times B1. GT coverage and prediction-validity distances improve
for all 571 paired contexts. Candidate utilization improves for 83.5%, ties
for 10.9%, and is lower for 5.6%. Cluster coverage improves for 27.7% and ties
for the remainder. The full-VAL comparison therefore passes all four M1 pilot
criteria without quality slack.

Reproducibility checks:

- Repeating seed-20260918 B1 VAL64 independently on two GPUs produced exactly
  identical aggregate and per-context metrics.
- Four-worker IPC-safe aggregation produced exactly the same VAL64 metrics as
  serial evaluation.
- The focused local suite passes 13 tests.

This remains a development-set result, not the official full TEST protocol or
a significance claim. The same-session listener recordings used as additional
GTs are proxies, not verified reactions to the identical stimulus. Only one
seed has been evaluated on all 571 contexts. No checkpoint, dataset, sample
identifier, per-context record or absolute local path is included here.
