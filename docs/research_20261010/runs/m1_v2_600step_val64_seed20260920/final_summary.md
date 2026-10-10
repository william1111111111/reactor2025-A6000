> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# CoReSet-Reactor Milestone 1

## Research question

Does all-seen descriptor coverage beat random 10/18-GT training under the same TCN, optimizer, initialization, source schedule, and seed?

TRAIN steps per arm: 600; batch: 2; VAL pilot contexts: 64.

| Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | GT coverage ↓ | validity ↓ | utilization ↑ | cluster coverage ↑ | train samples/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.779013 | 92.046746 | 0.010436 | 0.039162 | 1.234407 | 0.994181 | 0.581 | 0.042 | 25.997 |
| B1 | 0.778911 | 91.851143 | 0.010628 | 0.039089 | 1.233422 | 0.993885 | 0.553 | 0.042 | 24.005 |
| B3 | 0.353831 | 85.687768 | 0.060211 | 0.041504 | 0.998440 | 0.688820 | 0.802 | 0.069 | 21.446 |

## Interpretation

B3 vs B1: ΔFRC=-0.425080, Δexact FRD=-6.163374, ΔFRDiv=+0.049583, ΔGT coverage=-0.234982.
Zero-slack criteria: coverage_improves=pass, FRC_non_degraded=fail, exact_FRD_non_degraded=pass, FRDiv_improves=pass.
Milestone-1 hypothesis: not supported on all four coverage/quality/diversity criteria. This is a small development pilot, not a significance claim or permission to enter Milestone 2.

## Boundaries and next step

The 90+ targets are same-session recordings, not verified reactions to the same stimulus. TRAIN descriptors are cached once per distinct listener recording; full raw GTs are used for descriptors, while sampled fixed-length GTs are used for unpaired temporal matching. Valid source prefixes are used for prediction losses and metrics; official metrics round prediction AUs. No explicit prediction-separation loss was used. Official metric formulas and target post-processor were called on a deterministic VAL subset; this is not the full official TEST protocol. Cluster coverage is a non-official 12-medoid neighborhood diagnostic.

Before Milestone 2, repeat with more steps, multiple seeds and a larger frozen VAL set; inspect descriptor scaling and collapse if B3 fails.
