> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# CoReSet-Reactor Milestone 1

## Research question

Does all-seen descriptor coverage beat random 10/18-GT training under the same TCN, optimizer, initialization, source schedule, and seed?

TRAIN steps per arm: 600; batch: 2; VAL pilot contexts: 24.

| Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | GT coverage ↓ | validity ↓ | utilization ↑ | cluster coverage ↑ | train samples/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.707039 | 91.083195 | 0.001464 | 0.011921 | 1.176802 | 0.879329 | 0.617 | 0.052 | 23.741 |
| B1 | 0.709263 | 91.150931 | 0.001497 | 0.011995 | 1.176544 | 0.879775 | 0.646 | 0.052 | 23.593 |
| B3 | 0.444228 | 85.684274 | 0.024274 | 0.016843 | 1.009212 | 0.676842 | 0.792 | 0.073 | 21.411 |

## Interpretation

B3 vs B1: ΔFRC=-0.265035, Δexact FRD=-5.466657, ΔFRDiv=+0.022776, ΔGT coverage=-0.167332.
Milestone-1 hypothesis: not supported on all three quality/coverage criteria. This is a small development pilot, not a significance claim or permission to enter Milestone 2.

## Boundaries and next step

The 90+ targets are same-session recordings, not verified reactions to the same stimulus. TRAIN descriptors are cached once per distinct listener recording; unpaired temporal matching uses sampled GT only. No explicit prediction-separation loss was used. Official metric formulas and target post-processor were called on a deterministic VAL subset; this is not the full official TEST protocol. Cluster coverage is a non-official 12-medoid neighborhood diagnostic.

Before Milestone 2, repeat with more steps, multiple seeds and a larger frozen VAL set; inspect descriptor scaling and collapse if B3 fails.
