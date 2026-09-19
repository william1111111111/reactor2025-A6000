# Mam anchor ensemble: true pair plus nine fixed targets

Ten independent ConditionalREGNN anchors share architecture, initialization seed, source/crop schedule and optimizer settings. Rank 0 uses the true same-basename paired listener; ranks 1-9 use deterministic distinct same-session listeners. Once selected, every target follows the original diffusion `ReactionDataset` crop offset and short-target speaker-emotion tail-fill path.

- Training: `50` epochs/model, batch `32`, seed `1`, `51500` total model-steps
- Optimizer: AdamW lr `0.0001`, weight decay `0.0001`, cosine decay; BF16 `True`
- Loss: MSE + `1.0` × (1-CCC) + `0.05` × velocity MSE
- Parameters: `3720129` per model, `37201290` total
- Evaluation: deterministic VAL `571`, exact rolling FRD, `16` metric workers
- Base HEAD: `0f7e26019c46f30cc8632a0796367798f19a4fae`
- Executed relevant-source snapshot SHA-256: `73982c5c13a9632b9e0cd83c5e285c493972306209746d1c07ffdd92752df283`
- Published relevant-source snapshot SHA-256: `b5e837229f0ef8c1b075f05ce307c1881d156706d69cffaf354a663ffcfe44a6`
- Epoch-50 checkpoint-manifest SHA-256: `e6dd96173fd098c30f37e56bf4f036f07c0c96c69b7b0377f731dad0bf27cc4d`

| method | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | TLCC | GT coverage ↓ |
|---|---:|---:|---:|---:|---:|---:|
| frozen M1 | 0.810111 | 82.480135 | 0.013611 | 0.039200 | n/a | 1.095160 |
| paired + nine fixed anchors | 0.806542 | 95.141523 | 0.066710 | 0.054319 | 49.000000 | 1.147389 |

Deltas versus M1: ΔFRC `-0.003570`, Δexact-FRD `+12.661389`, ΔFRDiv `+0.053098`, ΔFRVar `+0.015119`, ΔGT-coverage `+0.052229`.

Additional diagnostics: prediction validity `1.008993`, candidate utilization `0.533800`, cluster coverage `0.077496`.

TLCC uses the repository's official implementation, which returns after its
first prediction; it is reproduced for completeness but is not a ten-output
aggregate.

This is not compute-matched to M1: ten full ConditionalREGNN backbones are trained and retained.

The matched VAL64 pilot showed the same shape: FRC `0.760295`, exact FRD
`95.507956`, FRDiv `0.063854`, FRVar `0.050649`, and TLCC `49.0`.

Verdict: fixed one-to-one target specialization produces substantially more
diverse reactions, but it is not quality-safe. FRDiv rises by `0.053098`
(`4.90×` M1), while exact FRD worsens by `12.661389` and descriptor coverage
worsens by `0.052229`. The small FRC change does not rescue that trade-off.
