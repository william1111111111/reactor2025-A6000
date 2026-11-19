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

## REACT2025 core distillation ablations

The matched A0/A1/A2/A3 experiments, three-seed A0/A1 repeats, and
Fixed/Uniform/Target-specific reference-teacher controls are published in
[`paper_core_ablations_react2025/`](paper_core_ablations_react2025/). Every
formal row uses complete bidirectional VAL1142 and records raw and fixed
query-0-postprocessed output separately. No new REACT2024 training or
evaluation is included in this bundle.

- [`MULTISEED_REPORT.md`](paper_core_ablations_react2025/MULTISEED_REPORT.md):
  three-seed A0 versus trajectory-only A1 results.
- [`SECOND_PRIORITY_REPORT.md`](paper_core_ablations_react2025/SECOND_PRIORITY_REPORT.md):
  direct-reference A2 and uncentered group-MSE A3 controls.
- [`REFERENCE_CONTROLS_REPORT.md`](paper_core_ablations_react2025/REFERENCE_CONTROLS_REPORT.md):
  matched reference-prior teachers and their students.
- [`CLAIM_EVIDENCE.md`](paper_core_ablations_react2025/CLAIM_EVIDENCE.md):
  supported claims, tradeoffs, and evidence boundaries.

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

## CoReSet supplementary appendix

The completed REACT2025 supplementary experiments are collected in
[`paper_core_ablations_react2025/appendix_20261119/`](paper_core_ablations_react2025/appendix_20261119/):

- [English Markdown appendix](paper_core_ablations_react2025/appendix_20261119/CoReSet_Appendix_Experiments_EN.md)
- [中文结果整理](paper_core_ablations_react2025/appendix_20261119/CoReSet_附录实验结果_中文.md)

The appendix contains 21 result tables, six figure groups, experimental
settings, paired confidence intervals, and source-file hashes. It covers
the reference-selection/assignment 2×2 ablation, three-seed student
distillation comparisons, set preservation and reference coverage, the A3
control, balanced OT, inference efficiency, and qualitative diagnostics.
Reported trade-offs and statistical scope are retained.
