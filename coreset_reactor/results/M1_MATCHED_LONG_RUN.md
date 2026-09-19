# CoReSet-Reactor M1: matched single-seed longer-run comparison

Three arms were trained from scratch with seed `20260918`, 2500 steps, batch
size 2, the same GPU, model initialization, fixed source schedule, sampled
exact-GT policy (paired + 17), and 64-context VAL subset selected by
`eval_seed=1234`. The only objective difference is that B3 adds the all-seen
GT descriptor block at static weight 0.05 or 0.10. The run configs have
identical source-snapshot, initial-model and schedule hashes.

The step-600 checkpoints were saved inside the 2500-step runs and evaluated
afterward without interrupting the training RNG. They are not the older
standalone 600-step runs, whose schedules differ. At 2500 steps the fixed
schedule draws 5000 source samples—about 3.01 epoch-equivalents—from 1660
TRAIN contexts and includes 1581 distinct source contexts.

| Step | Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | GT coverage distance ↓ | Candidate utilization ↑ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 600 | B1 | 0.704084 | 90.278069 | 0.010113 | 0.032253 | 1.237027 | 0.590625 |
| 600 | B3 / 0.05 | 0.703394 | 87.855617 | 0.012253 | 0.033643 | 1.158808 | 0.573438 |
| 600 | B3 / 0.10 | 0.657844 | 85.748596 | 0.018275 | 0.033606 | 1.104191 | 0.645313 |
| 2500 | B1 | 0.724627 | 87.412197 | 0.006085 | 0.030008 | 1.252561 | 0.528125 |
| 2500 | B3 / 0.05 | 0.746225 | 83.502979 | 0.008794 | 0.033122 | 1.130717 | 0.614063 |
| 2500 | B3 / 0.10 | 0.749149 | 82.162475 | 0.012697 | 0.035116 | 1.091207 | 0.710938 |

At step 600, neither B3 weight meets strict zero-slack FRC non-degradation
against the matched B1. At step 2500, both weights meet all four strict pilot
criteria: FRC >= B1, exact FRD <= B1, FRDiv > B1 and coverage distance < B1.
B3 / 0.10 minus B1 at step 2500 is FRC +0.024522, exact FRD -5.249721,
FRDiv +0.006613 (2.087 times B1), and coverage distance -0.161355.

FRDiv falls from step 600 to 2500 for B1 (0.010113 to 0.006085), B3 / 0.05
(0.012253 to 0.008794), and B3 / 0.10 (0.018275 to 0.012697), while FRC rises
in all three. This comparison therefore does not support the claim that
insufficient training alone explains the low absolute FRDiv. The descriptor
objective preserves a relative diversity and GT-coverage benefit, but this
TCN remains far below the Mam-Reactor P4 full-TEST FRDiv. That cross-model
number uses a different architecture, pretraining history and evaluation
split and is not a matched training-budget comparison.

This is one seed and one deterministic VAL64 pilot, not a full official TEST
or significance claim. Same-session GT recordings are not verified reactions
to the same stimulus. The runs have no exact run-time Git SHA; matching source
snapshot hashes were captured instead. No checkpoint, raw report, training
curve, sample identifier, per-context metric, dataset or local path is
published here.
