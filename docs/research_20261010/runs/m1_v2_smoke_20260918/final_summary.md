> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# CoReSet-Reactor Milestone 1

## Research question

Does all-seen descriptor coverage beat random 10/18-GT training under the same TCN, optimizer, initialization, source schedule, and seed?

TRAIN steps per arm: 2; batch: 2; VAL pilot contexts: 1.

| Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | GT coverage ↓ | validity ↓ | utilization ↑ | cluster coverage ↑ | train samples/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.171455 | 191.531241 | 0.327079 | 0.081825 | 2.326752 | 2.305487 | 0.400 | 0.000 | 27.540 |
| B1 | 0.171587 | 192.073995 | 0.326293 | 0.081380 | 2.323130 | 2.302086 | 0.400 | 0.000 | 23.078 |
| B3 | 0.189015 | 198.968556 | 0.336176 | 0.082356 | 2.218553 | 2.263239 | 0.400 | 0.000 | 23.904 |

## Interpretation

B3 vs B1: ΔFRC=+0.017428, Δexact FRD=+6.894561, ΔFRDiv=+0.009883, ΔGT coverage=-0.104577.
Zero-slack criteria: coverage_improves=pass, FRC_non_degraded=pass, exact_FRD_non_degraded=fail, FRDiv_improves=pass.
Milestone-1 hypothesis: not supported on all four coverage/quality/diversity criteria. This is a small development pilot, not a significance claim or permission to enter Milestone 2.

## Boundaries and next step

The 90+ targets are same-session recordings, not verified reactions to the same stimulus. TRAIN descriptors are cached once per distinct listener recording; full raw GTs are used for descriptors, while sampled fixed-length GTs are used for unpaired temporal matching. Valid source prefixes are used for prediction losses and metrics; official metrics round prediction AUs. No explicit prediction-separation loss was used. Official metric formulas and target post-processor were called on a deterministic VAL subset; this is not the full official TEST protocol. Cluster coverage is a non-official 12-medoid neighborhood diagnostic.

Before Milestone 2, repeat with more steps, multiple seeds and a larger frozen VAL set; inspect descriptor scaling and collapse if B3 fails.
