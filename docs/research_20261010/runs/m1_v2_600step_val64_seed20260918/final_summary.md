> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# CoReSet-Reactor Milestone 1

## Research question

Does all-seen descriptor coverage beat random 10/18-GT training under the same TCN, optimizer, initialization, source schedule, and seed?

TRAIN steps per arm: 600; batch: 2; VAL pilot contexts: 64.

| Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | GT coverage ↓ | validity ↓ | utilization ↑ | cluster coverage ↑ | train samples/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.686313 | 88.483660 | 0.007836 | 0.032796 | 1.246730 | 0.999029 | 0.548 | 0.039 | 26.266 |
| B1 | 0.688476 | 88.613668 | 0.007848 | 0.032886 | 1.246347 | 0.999317 | 0.555 | 0.039 | 24.871 |
| B3 | 0.379560 | 88.008084 | 0.066818 | 0.044599 | 0.997448 | 0.700421 | 0.802 | 0.068 | 21.640 |

## Interpretation

B3 vs B1: ΔFRC=-0.308916, Δexact FRD=-0.605584, ΔFRDiv=+0.058970, ΔGT coverage=-0.248899.
Zero-slack criteria: coverage_improves=pass, FRC_non_degraded=fail, exact_FRD_non_degraded=pass, FRDiv_improves=pass.
Milestone-1 hypothesis: not supported on all four coverage/quality/diversity criteria. This is a small development pilot, not a significance claim or permission to enter Milestone 2.

## Boundaries and next step

The 90+ targets are same-session recordings, not verified reactions to the same stimulus. TRAIN descriptors are cached once per distinct listener recording; full raw GTs are used for descriptors, while sampled fixed-length GTs are used for unpaired temporal matching. Valid source prefixes are used for prediction losses and metrics; official metrics round prediction AUs. No explicit prediction-separation loss was used. Official metric formulas and target post-processor were called on a deterministic VAL subset; this is not the full official TEST protocol. Cluster coverage is a non-official 12-medoid neighborhood diagnostic.

Before Milestone 2, repeat with more steps, multiple seeds and a larger frozen VAL set; inspect descriptor scaling and collapse if B3 fails.
