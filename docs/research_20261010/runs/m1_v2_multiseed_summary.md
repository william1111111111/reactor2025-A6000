> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# CoReSet-Reactor Milestone 1: corrected three-seed VAL pilot

Protocol: B0 (paired + 9 random GT), B1 (paired + 17 random GT), and B3
(B1 exact targets plus all-session full-raw-GT descriptor supervision) used
the same architecture, initialization, source schedule, optimizer, and seed
within each run. Each arm trained for 600 steps with batch size 2. Seeds were
20260918, 20260919, and 20260920. All evaluations used the identical 64-context
deterministic VAL subset (evaluation seed 1234), 10 predictions, 10 aligned GTs,
the existing target post-processor and metric functions. Prediction AU channels
were rounded before official metrics. This is not the full official TEST score.

The v2 descriptor cache contains 1,660 distinct TRAIN listener recordings in
17 sessions (94–101 GTs per session). Descriptors use every frame of each raw
GT; sampled expensive temporal targets still use a fixed 750-frame crop/stretch.
The model and losses use only valid source prefixes. Full-split length audit:
TRAIN 282/1660 (16.99%) and VAL 113/571 (19.79%) sources/paired reactions are
shorter than 750 frames. Seven of the selected 64 VAL contexts are short.

| Seed | B1 FRC | B3 FRC | ΔFRC | Δexact FRD | ΔFRDiv | ΔGT coverage distance |
|---|---:|---:|---:|---:|---:|---:|
| 20260918 | 0.688476 | 0.379560 | -0.308916 | -0.605584 | +0.058970 | -0.248899 |
| 20260919 | 0.793772 | 0.400295 | -0.393477 | -5.863302 | +0.057406 | -0.235535 |
| 20260920 | 0.778911 | 0.353831 | -0.425080 | -6.163374 | +0.049583 | -0.234982 |
| Mean delta | | | -0.375824 | -4.210753 | +0.055320 | -0.239805 |

Lower exact FRD and GT coverage distance are better; higher FRC and FRDiv are
better. B3 consistently improves diversity and GT-descriptor coverage, and
also improves exact FRD in this pilot, but its large FRC loss fails the
zero-slack quality constraint in all three seeds. Thus Milestone 1 does not
support advancing to adaptive sampling or overgeneration yet. The result is a
quality/diversity tradeoff, not a win for all-seen supervision under the current
loss weights. The cause of the FRC drop remains to be diagnosed.

Same-session GTs are weak proxies, not verified reactions to an identical
stimulus. The original FRC implementation emitted zero-variance warnings for
some AU channels but returned finite scores in all runs. Checkpoints, metrics,
per-context data, caches, and this report remain local-only; none is intended
for GitHub.
