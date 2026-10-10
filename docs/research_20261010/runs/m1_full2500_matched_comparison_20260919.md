> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# CoReSet-Reactor M1: matched single-seed longer-run comparison

Three arms were trained **from scratch** with seed `20260918`, 2500 steps,
batch size 2, the same `cuda:5` device, model initialization, fixed source
schedule, sampled exact-GT policy (paired + 17), and 64-context VAL subset
selected by `eval_seed=1234`. The only objective difference is that B3 adds
the all-seen GT descriptor block at static weight 0.05 or 0.10. Run configs
have identical `source_snapshot_sha256`, `initial_model_sha256` and
`schedule_sha256`. All three checkpoints at step 600 were saved *inside the
2500-step runs* and evaluated afterward, without interrupting training RNG.
They are not the older standalone 600-step runs, whose schedules differ.

2500 steps draw 5000 source samples, about 3.01 epoch-equivalents from 1660
TRAIN contexts; due to replacement, 1581 distinct source contexts appear.

| Step | Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | GT coverage distance ↓ | Candidate utilization ↑ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 600 | B1 | 0.704084 | 90.278069 | 0.010113 | 0.032253 | 1.237027 | 0.590625 |
| 600 | B3 / 0.05 | 0.703394 | 87.855617 | 0.012253 | 0.033643 | 1.158808 | 0.573438 |
| 600 | B3 / 0.10 | 0.657844 | 85.748596 | 0.018275 | 0.033606 | 1.104191 | 0.645313 |
| 2500 | B1 | 0.724627 | 87.412197 | 0.006085 | 0.030008 | 1.252561 | 0.528125 |
| 2500 | B3 / 0.05 | 0.746225 | 83.502979 | 0.008794 | 0.033122 | 1.130717 | 0.614063 |
| 2500 | B3 / 0.10 | 0.749149 | 82.162475 | 0.012697 | 0.035116 | 1.091207 | 0.710938 |

At step 600, neither B3 weight meets strict zero-slack FRC non-degradation
against matched B1. At step 2500, both weights meet all four strict M1 pilot
criteria (FRC >= B1, exact FRD <= B1, FRDiv > B1, coverage distance < B1).
B3 / 0.10 minus B1 at step 2500: FRC +0.024522, exact FRD -5.249721,
FRDiv +0.006613 (2.087 times B1), coverage distance -0.161355.

Across the same training trajectories, FRDiv **falls** from step 600 to 2500
for B1 (0.010113 to 0.006085), B3 / 0.05 (0.012253 to 0.008794), and
B3 / 0.10 (0.018275 to 0.012697), while FRC rises in all three. Therefore
this comparison does not support the claim that insufficient training alone
explains low absolute FRDiv. The descriptor objective preserves a relative
diversity and GT-coverage benefit, but this TCN remains far below the
Mam-Reactor P4 full-TEST FRDiv; that cross-model number uses a different
architecture, pretraining history and evaluation split and is **not** a
matched training-budget comparison.

This is one seed and one deterministic VAL64 pilot, not full official TEST or
a significance claim. Same-session GT recordings are not verified reactions
to the same stimulus. `git_commit_sha` is null because CoReSet is untracked in
the local enclosing repository; the run configs record the matching source
snapshot hash instead. Raw per-context metrics, checkpoints, training curves
and data remain local in ignored `coreset_reactor/reports/` folders and were
not uploaded.
