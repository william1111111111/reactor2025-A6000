> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 学生结构与蒸馏组合消融结果

Single training seed; complete VAL571/all-session. No additional seeds and no independent TEST.

| 方法 | FRC ↑ | FRD ↓ | FRDiv ↑ | FRVar | soft FRDiv |
|---|---:|---:|---:|---:|---:|
| kd_basic | 1.464839 | 100.774094 | 0.171190 | 0.070300 | 0.086928 |
| kd_assignment | 1.483930 | 100.578430 | 0.174516 | 0.072014 | 0.089496 |
| kd_dynamic | 1.453440 | 95.718811 | 0.167214 | 0.066872 | 0.084457 |
| kd_assignment_dynamic | 1.471408 | 96.182289 | 0.166247 | 0.067881 | 0.081832 |
| kd_multihead | 1.396709 | 96.999068 | 0.188895 | 0.068924 | 0.100102 |

完整四个对照、四格交互及区间见 analysis.json；推理微基准见 BENCHMARK.json。

5000 bootstrap draws over 20 sessions; no training-seed uncertainty or multiple-testing correction. Architecture has common pretrained weights, not identical complete initial functions. Interaction is joint-assignment-dynamic+basic; its sign alone does not establish Pareto improvement.
