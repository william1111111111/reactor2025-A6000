> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Milestone-1 failure analysis: 600-step, 24-context VAL pilot

This is a local development result, not a full official TEST result or a claim of statistical significance. B0, B1 and B3 used the same 172,122-parameter TCN, initial weights, source/crop schedule, batch size 2, AdamW settings and seed. Only GT-use policy changed. Each model saw 600 updates. The 24 VAL contexts and paired-plus-nine selected GTs were fixed across arms. Official FRC, exact FRD, FRDiv and FRVar functions and the existing target post-processor were called.

| Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | full-session GT descriptor coverage ↓ | validity ↓ | utilization ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0: random 10 exact | 0.707039 | 91.083195 | 0.001464 | 0.011921 | 1.176802 | 0.879329 | 0.617 |
| B1: random 18 exact | 0.709263 | 91.150931 | 0.001497 | 0.011995 | 1.176544 | 0.879775 | 0.646 |
| B3: all-GT descriptor + random 18 exact | 0.444228 | 85.684274 | 0.024274 | 0.016843 | 1.009212 | 0.676842 | 0.792 |

Relative to B1, B3 improves descriptor coverage by 14.2%, validity by 23.1%, and exact FRD by 6.0%; FRDiv is 16.2× larger and utilization increases from 0.646 to 0.792. But FRC falls by 0.265035 (37.4%). Thus the central hypothesis is **not established as a quality-preserving win**. More diverse predictions are not automatically good reactions under the official CCC-based FRC metric. B0 and B1 are nearly identical, so this run also does not establish a benefit from increasing random exact GTs from 10 to 18.

The non-official 12-medoid cluster-neighborhood coverage diagnostic is 0.052 for B0/B1 and 0.073 for B3. Its threshold is heuristic, so it is supporting context rather than an independently validated success metric.

## Loss diagnostics

Last-100-step means: B1 paired loss 0.83214, B3 paired loss 0.86446; B3 cover 0.77179 and validity 0.36509 (validity weight 0.5). The load-balance hinge was zero throughout the five sampled B3 gradient probes. Weighted unpaired SoftDTW scalar values were small, but loss value alone does not measure gradient strength.

The frozen B3 checkpoint was probed on five deterministic training minibatches (steps 1, 150, 300, 450, 600). Mean parameter-gradient norms were approximately 0.256 for paired, 0.050 for weighted unpaired SoftDTW, and 0.404 for the weighted descriptor objective. Thus the descriptor gradient was about 1.6× the paired gradient at these probes; it is plausible that descriptor matching dominated the quality-preserving aligned signal. This is a diagnostic inference, not proof of a unique cause. See `m1_B3_seed20260918/gradient_diagnostics.json` and the B1 counterpart.

The 602-D descriptor includes statistics and eight coarse temporal segments, not a precise phase-aligned CCC target. Its TRAIN-only scaler had 97 dimensions at the 0.03 standard-deviation floor (28 global, 57 segment, 2 VA, 10 AU). B3's lower descriptor distance together with substantially lower FRC indicates that this descriptor's notion of validity does not guarantee CCC quality. The paired term only asks some prediction to fit the aligned paired GT, while FRC scores **every** prediction against its best of ten GTs. This objective/metric mismatch is a likely mechanism for the quality drop.

Data provenance is another major limitation: the 94–101 cached GTs per TRAIN session are distinct same-session recordings, not verified reactions to the identical stimulus. All 1660 listener files from 17 TRAIN sessions were cached once. The VAL coverage diagnostic likewise uses same-session recordings. Pulling predictions toward their statistics may improve broad session coverage without improving stimulus-specific reaction fidelity.

## Efficiency and decision

B3 processed 21.41 training samples/s versus B1's 23.59 on the local A6000 GPU 5 (about 9.2% slower), with roughly 775 MB peak allocated GPU memory. Forward-only K=10 latency was about 1.2 ms per 750-frame set in this pilot; it excludes data loading and target processing. No L=20 or Mam-Reactor throughput comparison is claimed at this milestone.

Do **not** enter Milestone 2 yet. The next within-Milestone-1 test should pre-register a small descriptor-loss weight sweep on the same frozen VAL contexts, inspect per-candidate CCC/descriptor distance, and repeat with additional seeds. If FRC still collapses, revise the descriptor or quality constraint rather than adding cluster/hard/under-used sampling. Larger or held-out evaluation is needed before a positive paper claim.
