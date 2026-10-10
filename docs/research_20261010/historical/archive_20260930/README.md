> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# REACT 多样反应生成研究归档

归档日期：2026 年 9 月 30 日。用途：冻结当前研究思路、实验结果、参数、代码和权重定位，供论文写作与后续复现使用。当前阶段收敛实验，不启动新训练、不删除旧权重、不自动发布 GitHub。

当前论文主线是 **多样化固定 GT → 模式保持的多参考教师 → 单模型学生蒸馏**。这不同于早期 CoReSet 的 TCN 覆盖监督，也不同于 Mam-Reactor P4 的原始主模型。当前模型类仍为 ConditionalREGNN / ExpertConditionalREGNN，保留 graph=2；不能写成已经验证无损的纯 Transformer。

## 如何阅读

1. [方法与全部关键参数](METHOD_AND_PARAMETERS.md)：问题定义、公式、架构、数据处理、教师与学生训练和评估口径。
2. [主结果与历史研究脉络](RESULTS_AND_HISTORY.md)：当前主结果、组件消融、输入依赖诊断、负结果、历史路线及勘误。
3. [复现与文件定位](REPRODUCIBILITY.md)：环境、入口、权重、代码版本、数据与报告边界。
4. 完整归档根目录的 `ALL_REPORTS.md`：所有扫描到的原始 Markdown 报告，可直接点击原文。
5. `metric_index.csv` / `metric_records.jsonl`：从原始 JSON 字典提取的指标索引，包含原文件和 JSON pointer。它是检索账本，**不是将所有行视为同口径的排行榜**；逐 context / rank 数组完整保存在原始 JSON 中。
6. `configs_and_provenance.jsonl`：配置、实验协议、启动记录、summary 和 manifest 的原始内容。缺失字段保持缺失，不按当前默认值补写历史参数。
7. `weights_and_caches.csv`：checkpoint、优化器状态及张量缓存位置。checkpoint 有 SHA256；大张量缓存一般仅登记路径与大小，不能声称这些缓存都已校验哈希。

推荐长期归档位置：

`$ARCHIVE_ROOT/`

当前仓库工作树位于 `$SOURCE_ROOT`。因此原始文本报告和源码会另外复制到上面的持久目录，不能只依赖 `/tmp`。

## 当前建议冻结的学生

| 数据集与口径 | FRC ↑ | exact FRD ↓ | FRDiv | FRVar |
|---|---:|---:|---:|---:|
| REACT2025 VAL571 all-session | 1.445507 | 97.463281 | 0.170130 | 0.067918 |
| 转换版 REACT2024 VAL1124 all-session | 0.656394 | 86.033383 | 0.132724 | 0.058911 |

两者均为 Base KD + centered-residual KD，学生 velocity 权重为 0。每个学生 3,803,201 参数，输出十条 25D 反应，不依赖推理 GT、十个教师、随机噪声、selector 或输出幅度缩放。2025 教师参考先验 effective count=8；2024 为用户接受的 count=4 折中点，并非同一超参数的盲迁移。

这里的“冻结”是论文整理的工作决策，不表示删除其他 operating point，也不表示每项指标严格最优。Minimal KD 有更高 FRC；当前残差项主要换来更好的 FRD。所有 full、minimal、velocity、residual 及历史高多样性权重均保留。

## 归档完整性和隐私

本归档覆盖 `scope.json` 列明的 CoReSet 工作树、持久报告、相关历史临时实验和 Mam 来源文档。它保存扫描时可读的支持格式文件，不声称恢复已删除文件、远端未下载实验或所有无关研究分支。

`evidence/` 是原始文本证据的逐文件副本；`file_manifest.jsonl` 给出原路径、复制后位置、大小与 SHA256。源码快照包括尚未提交的代码；仅 Git HEAD 不足以描述运行状态。历史报告中的“当前”“未启动”只代表其撰写时点，最新状态以本归档及对应新 run 为准。

权重和大缓存**原地保留、建立索引，不重复复制**。因此这是完整的研究记录归档，不是包含全部数据与权重的独立离线恢复镜像。特别是原地位于 `/tmp` 的旧权重仍有临时存储风险；长期主线权重位于 `/public`。

此归档含本机路径、样本标识、未发表源码和逐样本结果，属于 **LOCAL ONLY**。不含 SSH 私钥，不上传原始数据，不自动 git add/commit/push。未来发布须另导出经过审查的聚合结果，而非整包推送。

## 结论边界

- 本地 REACT2025 TEST 与 VAL 内容相同，作为已知数据条件保留；只计一套评估，不构造虚假的独立 TEST 结果。
- 多次使用 VAL 选择方案，当前核心路线为 seed1。历史 M1 的三 seed 结果不能当作最终教师学生路线的三 seed 验证。
- all-session 与随机十 GT、原生 REACT2024 指标及 challenge 在线协议分开报告；不直接横比排行榜。
- Session bootstrap 是给定训练 run 的 session 变异，不是训练 seed 方差。
- 不宣称所有模块缺一不可，不宣称严格 Pareto 全面胜出，不把高 FRDiv 自动解释成合理反应或真实交互因果。
- 未完成的独立划分、外部强基线、人评、多 seed 等是证据范围限制，不再自动触发新实验。
