> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 不同蒸馏方式：完整 VAL571 结果

Matched pretrained single-seed VAL571, not scratch/independent TEST. Full all-session references.

| 方法 | FRC ↑ | FRD ↓ | FRDiv ↑ | soft FRDiv |
|---|---:|---:|---:|---:|
| kd_basic | 1.464839 | 100.774094 | 0.171190 | 0.086928 |
| kd_residual | 1.447309 | 97.606049 | 0.173210 | 0.086741 |
| kd_typed | 1.461345 | 100.295472 | 0.173361 | 0.088327 |
| kd_assignment | 1.483930 | 100.578430 | 0.174516 | 0.089496 |
| kd_dynamic | 1.453440 | 95.718811 | 0.167214 | 0.084457 |

20 session clusters, 5000 bootstrap; no seed uncertainty/multiplicity correction. Continuous energy diagnostics are not proof of valid mode coverage; teacher training cost extra.

全部五个预定比较和区间见 analysis.json；不自动据此宣称创新成立。
