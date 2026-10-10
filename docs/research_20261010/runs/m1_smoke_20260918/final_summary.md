> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# CoReSet-Reactor Milestone 1

## Research question

Does all-seen descriptor coverage beat random 10/18-GT training under the same TCN, optimizer, initialization, source schedule, and seed?

TRAIN steps per arm: 2; batch: 2; VAL pilot contexts: 1.

| Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | GT coverage ↓ | validity ↓ | utilization ↑ | train samples/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.100330 | 179.674417 | 0.060571 | 0.010330 | 2.029515 | 2.059234 | 0.400 | 25.082 |
| B1 | 0.099216 | 180.104603 | 0.060756 | 0.010301 | 2.026511 | 2.054549 | 0.400 | 20.842 |
| B3 | 0.105035 | 187.017286 | 0.066485 | 0.010643 | 1.943421 | 2.024032 | 0.400 | 21.998 |

## Interpretation

B3 vs B1: ΔFRC=+0.005819, Δexact FRD=+6.912684, ΔFRDiv=+0.005729, ΔGT coverage=-0.083090.
Milestone-1 hypothesis: not supported on all three quality/coverage criteria. This is a small development pilot, not a significance claim or permission to enter Milestone 2.

## Boundaries and next step

The 90+ targets are same-session recordings, not verified reactions to the same stimulus. TRAIN descriptors are cached once per distinct listener recording; unpaired temporal matching uses sampled GT only. No explicit prediction-separation loss was used. Official metric formulas and target post-processor were called on a deterministic VAL subset; this is not the full official TEST protocol.

Before Milestone 2, repeat with more steps, multiple seeds and a larger frozen VAL set; inspect descriptor scaling and collapse if B3 fails.
