# Evaluation contract for ablations

Every REACT2025 ablation table that reports the repository's reaction metrics
must use one frozen protocol:

- complete VAL1142: 571 speaker-to-listener and 571 listener-to-speaker
  contexts, averaged with equal context weight;
- complete source duration, with predictions evaluated after stitching rather
  than averaging per-window scores;
- every processed opposite-role GT from the same session;
- official Processor behavior and AU rounding;
- FRC, exact FRD, FRDiv, FRVar, and official TLCC/FRSyn definitions.

A single-direction subset, a cropped 750-frame view, a sampled-GT pool, or a
surrogate FRD must not be published as an ablation result. It cannot be used
to rank an ablation, select a paper operating point, or populate a result
table. Direction-specific values may be shown only alongside the VAL1142
average produced from the same run.

`eval_multiview_ablation_bidir_all_session.py` is the active evaluator for the
multi-view and boundary-fusion ablations. Its preparation and aggregation
steps reject any population other than 571 contexts in each direction, and
aggregation rejects rows missing one of the five metrics.

REACT2024 keeps its native complete VAL1124 population and alpha=1.5 operating
point. The REACT2025 bidirectional requirement must not be simulated by
duplicating REACT2024 samples.
