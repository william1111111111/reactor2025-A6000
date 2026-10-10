> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Mam anchor ensemble: true pair plus nine fixed targets

Ten independent ConditionalREGNN anchors share architecture, initialization seed, source/crop schedule and optimizer settings. Rank 0 uses the true same-basename paired listener; ranks 1-9 use deterministic distinct same-session listeners. Once selected, every target follows the original diffusion `ReactionDataset` crop offset and short-target speaker-emotion tail-fill path.

- Checkpoint epoch label: `6`; step cap/model: `600` (0 means uncapped); batch `32`, seed `1`, `6000` total model-steps
- Optimizer: AdamW lr `0.0001`, weight decay `0.0001`, cosine decay; BF16 `True`
- Loss: MSE + `1.0` × (1-CCC) + `0.05` × velocity MSE
- Parameters: `3583679` per model, `35836790` total
- Evaluation: deterministic VAL `571`, exact rolling FRD, `16` metric workers

| method | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | TLCC | GT coverage ↓ |
|---|---:|---:|---:|---:|---:|---:|
| frozen M1 | 0.810111 | 82.480135 | 0.013611 | 0.039200 | n/a | 1.095160 |
| paired + nine fixed anchors | 1.348490 | 85.689153 | 0.036882 | 0.035675 | 49.000000 | 1.135109 |

Deltas versus M1: ΔFRC `+0.538378`, Δexact-FRD `+3.209018`, ΔFRDiv `+0.023270`, ΔFRVar `-0.003525`, ΔGT-coverage `+0.039949`.

Additional diagnostics: prediction validity `0.925342`, candidate utilization `0.652014`, cluster coverage `0.073263`.

TLCC uses the repository's official implementation, which returns after its first prediction; it is reproduced for completeness but is not a ten-output aggregate.

This is not compute-matched to M1: ten full ConditionalREGNN backbones are trained and retained.

Interpret results against the matched random-fixed-target ten-anchor baseline; target selection alone does not guarantee improved prediction quality.
