# reactor2025-A6000

Public source snapshot for the REACT2024/REACT2025 teacher-student
distillation route.

## Current REACT2025 route

The current student uses deterministic head/center/tail teacher views for
training clips longer than 750 frames. At inference time it applies 50%
overlapping windows and center-weighted full-sequence fusion. The selected
`p=4` fusion path uses the documented query-0 constant-AU transition
postprocess after official AU rounding.

All published REACT2025 metrics follow one contract:

- complete VAL1142, with 571 contexts in each direction;
- complete source sequences, scored after window stitching;
- every opposite-role GT from the same session;
- official Processor behavior, AU rounding, FRC, exact FRD, FRDiv, FRVar,
  and TLCC/FRSyn definitions.

The only intentional difference from the released validation loader is the
all-session GT pool. See
[`coreset_reactor/EVALUATION_CONTRACT.md`](coreset_reactor/EVALUATION_CONTRACT.md)
for the enforced protocol.

## Repository layout

- [`coreset_reactor/`](coreset_reactor/): distillation, full-sequence inference,
  official-metric evaluation, and protocol tests.
- [`mam_reactor/`](mam_reactor/): model and official metric runtime used by the
  distillation code.
- [`published_results/`](published_results/): accepted aggregate result tables.

The current machine-readable official-five aggregates are in
[`published_results/current_official5_results.json`](published_results/current_official5_results.json):
REACT2024 alpha=1.5 has FRSyn 44.849644128 and official FRDvs 0.128923953
on VAL1124, and the current
REACT2025 multi-view `p=4` route has FRSyn 45.803852890 on bidirectional
VAL1142.

Datasets, checkpoints, predictions, per-context rows, caches, and local run
directories are excluded. Single-direction REACT2025 performance tables are
not published.
