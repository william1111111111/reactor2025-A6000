# T0 graph-layer ablation

This is a matched React2025 T0 experiment.  The only retraining variable is
`graph_layers`: the baseline uses two learned relation blocks and the ablation
uses zero blocks.  Both use the same T0 fixed-pair manifests, seed `1`, source
schedule, 50-epoch formal schedule, 600 optimizer steps, batch size `32`,
BF16, AdamW (`lr=1e-4`, weight decay `1e-4`, betas `(0.9, 0.99)`), cosine
decay, gradient clipping `1.0`, and the same all-session GT VAL571 evaluator.

The baseline numbers are the frozen T0 graph=2 run from
`maxfrdiv_teacher_pilot_600_v3`.  The no-graph numbers are from ten newly
trained graph=0 models, one per fixed T0 rank.  No checkpoint or dataset is
stored in this report.

## Step-0 initialization check

With the same seed, graph=2 and graph=0 have identical non-graph parameter
state hashes:

```text
ee493d1d4809f15e879024e36744986e257625811958d5a8ddaa2204d2c7c569
```

Because `RelationGraphBlock.to_residual` is zero initialized, the two models
also produced bitwise-identical step-0 predictions (`max_abs_diff=0.0`).

## Full VAL571 all-session GT

| model | graph blocks | params/model | FRC | exact FRD | FRDiv | FRVar | GT descriptor coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0 matched baseline | 2 | 3,720,129 | 1.373332 | 84.929128 | 0.036415 | 0.036260 | 1.138303 |
| T0 no-graph retrain | 0 | 3,583,679 | 1.348490 | 85.689153 | 0.036882 | 0.035675 | 1.135109 |

Relative to graph=2, graph=0 changes are:

```text
FRC       -0.024842  (-1.81%)
exact FRD +0.760025
FRDiv     +0.000467  (+1.28%)
params    -136,450  (-3.67%)
```

The no-graph model preserves FRDiv and stays within `+1` exact-FRD, but it
does not meet the strict FRC criterion of no more than `0.5%` degradation:
`1.348490 < 1.366465`.

## Post-hoc graph bypass

As a separate diagnostic, the trained graph=2 checkpoints were evaluated twice
without retraining: once with their normal graph path and once with the graph
blocks bypassed.  This isolates the correction learned by the graph blocks.

| inference path | FRC | exact FRD | FRDiv | GT descriptor coverage | candidate utilization |
|---|---:|---:|---:|---:|---:|
| trained graph=2, normal | 1.373332 | 84.929128 | 0.036415 | 1.138303 | 0.650263 |
| same checkpoint, graph bypass | 1.380108 | 142.738623 | 0.039572 | 1.260559 | 0.525744 |

The bypass raises diversity slightly but severely damages temporal quality and
descriptor validity.  Therefore the graph refinement is not redundant after
training: its main contribution in this T0 setting is quality/temporal
correction, while the fixed target policy controls most of the diversity.

## Provenance

```text
data fingerprint: 13d2ad79aa558fcd7fc524088d6d3feb81c6508e8079237d1272a2c4fb670b77
T0 manifest source: c2bb033ddf6edc3ed0f4be1e5c3b1f830b92cc560582e599a625d7d3671d88aa
no-graph run: /public/zhaowenjie/react2025_new/coreset_reactor/reports/graph_ablation_t0_600/no_graph_ensemble
normal baseline: /public/zhaowenjie/react2025_new/coreset_reactor/reports/maxfrdiv_teacher_pilot_600_v3
evaluator scope: VAL571, center750, all same-session opposite-role GT
```

The graph bypass support is implemented in `anchor_pair_ensemble.py`; the
parallel pilot launcher now records `--graph-layers` and can select arms with
`--arms`.
