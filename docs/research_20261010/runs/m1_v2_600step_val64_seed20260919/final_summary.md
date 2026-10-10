> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# CoReSet-Reactor Milestone 1

## Research question

Does all-seen descriptor coverage beat random 10/18-GT training under the same TCN, optimizer, initialization, source schedule, and seed?

TRAIN steps per arm: 600; batch: 2; VAL pilot contexts: 64.

| Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | GT coverage ↓ | validity ↓ | utilization ↑ | cluster coverage ↑ | train samples/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.795093 | 92.245089 | 0.008699 | 0.039080 | 1.239621 | 0.988216 | 0.559 | 0.040 | 27.327 |
| B1 | 0.793772 | 92.437204 | 0.008688 | 0.038899 | 1.239925 | 0.989605 | 0.553 | 0.040 | 24.995 |
| B3 | 0.400295 | 86.573902 | 0.066094 | 0.042704 | 1.004390 | 0.715000 | 0.756 | 0.066 | 22.955 |

## Interpretation

B3 vs B1: ΔFRC=-0.393477, Δexact FRD=-5.863302, ΔFRDiv=+0.057406, ΔGT coverage=-0.235535.
Zero-slack criteria: coverage_improves=pass, FRC_non_degraded=fail, exact_FRD_non_degraded=pass, FRDiv_improves=pass.
Milestone-1 hypothesis: not supported on all four coverage/quality/diversity criteria. This is a small development pilot, not a significance claim or permission to enter Milestone 2.

## Boundaries and next step

The 90+ targets are same-session recordings, not verified reactions to the same stimulus. TRAIN descriptors are cached once per distinct listener recording; full raw GTs are used for descriptors, while sampled fixed-length GTs are used for unpaired temporal matching. Valid source prefixes are used for prediction losses and metrics; official metrics round prediction AUs. No explicit prediction-separation loss was used. Official metric formulas and target post-processor were called on a deterministic VAL subset; this is not the full official TEST protocol. Cluster coverage is a non-official 12-medoid neighborhood diagnostic.

Before Milestone 2, repeat with more steps, multiple seeds and a larger frozen VAL set; inspect descriptor scaling and collapse if B3 fails.
