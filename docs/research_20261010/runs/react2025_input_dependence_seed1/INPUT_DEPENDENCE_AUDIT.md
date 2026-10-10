> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# Frozen student input dependence audit

REACT2025 full VAL571. TRAIN-only templates/retrieval. Frozen student, no new training.
All-session AU-rounded FRC/exact FRD. Native paired CCC is a separate diagnostic, not official FRC.
Donor controls use only the equal-length750 cohort; baseline comparisons use exactly those contexts.

| Arm | N | FRC | exact FRD | FRDiv | Native paired mean CCC | Change MSE |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 571 | 1.445507 | 97.463281 | 0.170130 | 0.023024 | 0.000000 |
| audio_train_mean | 571 | 1.040990 | 107.261897 | 0.184895 | 0.008205 | 0.048966 |
| visual_train_mean | 571 | 1.463702 | 96.862408 | 0.176463 | 0.018947 | 0.014322 |
| temporal_blocks | 571 | 1.347071 | 97.413746 | 0.170761 | 0.013755 | 0.006553 |
| global_train_templates | 571 | 0.635839 | 105.207555 | 0.298558 | 0.001041 | 0.206040 |
| train_nearest_set | 571 | 0.790439 | 107.080023 | 0.276032 | 0.002138 | 0.144361 |
| same_session_donor | 458 | 1.346905 | 101.159770 | 0.162625 | 0.010958 | 0.021957 |
| cross_session_donor | 458 | 1.289366 | 102.561947 | 0.162625 | 0.011053 | 0.034280 |

Same-session aggregate invariance does not establish input neglect. Inspect paired/context changes and cross-session controls.
Motion-energy lag is descriptive, not official TLCC/FRSyn and not proof of appropriate responding.
Mean replacement and temporal permutation can be out of distribution; these are interventions, not retrained architecture ablations.
No TEST metrics were computed. See split audit before any independent-test claim.
Session bootstrap intervals in summary.json are conditional on one training run; no human-rating results claimed.
