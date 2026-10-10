> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Direct fixed-ten-GT shared student

## Question

Can the 10-slot shared student be trained directly against the fixed ten real
GT trajectories, without reading any teacher prediction?

## Matched protocol

- REACT2025 TRAIN, 3,320 source contexts.
- Fixed target manifest: paired GT plus nine KMeans-mode representatives.
- Each slot always receives the same manifest rank.
- Target processing is exactly `diffusion_listener_crop` / the historical
  `diffusion_legacy` fixed-pair path. Thirty target tensors spanning the first,
  middle and last TRAIN examples were checked bitwise against ten independent
  `ConditionalReactionDataset` rank loaders.
- One 3,803,201-parameter `ExpertConditionalREGNN`, not ten models.
- Teacher-0 backbone initialization, seed 1, batch 4, AdamW 1e-4, CCC weight
  1.0, velocity weight 0.05, gradient clip 1.0, 10 epochs / 8,300 updates.
- The matched Basic-KD control uses the same model and optimization contract;
  only its targets are cached teacher predictions.
- Evaluation: full VAL571, center-750 source, deterministic post-processing,
  AU rounding, all opposite-role GT from the same session, exact FRD.

## Result

| Method | FRC up | exact FRD down | FRDiv up | FRVar up |
|---|---:|---:|---:|---:|
| Basic KD student | 1.178483 | 85.709485 | 0.060729 | 0.043209 |
| Direct fixed-ten-GT student | **1.249694** | **82.960411** | 0.056467 | 0.042467 |
| Direct GT - Basic KD | **+0.071212** | **-2.749074** | -0.004261 | -0.000742 |

Relative to Basic KD, direct GT improves FRC by 6.04% and exact FRD by 3.21%,
while reducing FRDiv by 7.02%. The context-bootstrap 95% interval for the FRC
delta is `[+0.05096, +0.09176]` (descriptive only; contexts within sessions are
correlated).

## Conclusion

Direct fixed-ten-GT supervision is the stronger quality operating point. It
does not dominate Basic KD because the teacher predictions preserve slightly
more output diversity. Distillation is therefore not required to train the
shared student, but it currently acts as a diversity-preserving target
smoother. A clean next experiment is direct GT plus a weak slot-relation or
mode-preservation term, keeping this direct-GT quality point as the baseline.

This is a single-seed VAL analysis, not an official TEST submission.
