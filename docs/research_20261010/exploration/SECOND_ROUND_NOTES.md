> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 第二轮执行记录

任务依据：用户明确要求按 REACT2025 论文的 all-session 定义主评测。
旧goal文字中的“固定十参考”不再作为目标。详见 PROTOCOL_AUTHORITY_CORRECTION.md。

## 已执行

- 四组 fixed/uniform/mode/shuffled 的匹配控制已实现，8项训练单元测试通过（含汇聚变分恒等式）。
- 每组 rank1 的2步GPU smoke通过，初始化和数据日程digest一致。
- 36个正式600步作业已全部完成，结果写入 runs/matched_supervision_v1/train。
  VERIFIED.json与CHECKPOINT_AUDIT.json均通过：实际优化器步数、参数有限性、源码、
  manifest以及共同配置全部核验。GPU训练句柄19047已正常退出。
- 完整VAL571推理已完成，INFERENCE_AUDIT.json确认输出有限、顺序匹配、四组q0与
  历史归档q0逐元素一致。精确FRC/FRD评分已完成571条，句柄78570正常退出。
  matched_analysis.py已正常完成，完整结果与研究决策见SECOND_ROUND_REPORT.md。
- eval_preflight/VERIFIED.json：571上下文全参考缓存及历史FRC/FRDiv/FRVar复核通过；
  索引0/140/285的精确FRD与历史值一致。这不是新模型效果结果。

## REACT2024原生协议核验

论文：[REACT2024](https://arxiv.org/html/2401.05166v1)。
其VAL含553 NoXI+9 RECOLA对，双方向对应1124条输入；与当前转换inventory一致。
它不是REACT2025/MARS的同受控session标签定义。

原生代码依据：
[matrix_split.py](https://github.com/reactmultimodalchallenge/baseline_react2024/blob/main/tool/matrix_split.py)、
[FRC.py](https://github.com/reactmultimodalchallenge/baseline_react2024/blob/main/metric/FRC.py)。
按CSV顺序抽取全局矩阵的两个方向块，再按各输入的邻接行选参考；不能直接用同录制
池代替邻接矩阵。这里只核验2024的独立协议，不改变2025主评测。

本地审计：runs/native2024_protocol_v1/result.json（含文件SHA与审计脚本）。

- data_indices.csv：2962对，无重复；全局矩阵却为8616×8616，而两方向应为5924×5924。
- 旧val.csv：839对，含277 UDIVA；剔除UDIVA才是562对。
- neighbour_emotion_val.npy：1038×1038，既非旧CSV的1678×1678，也非当前1124×1124。
- 当前VAL的562对都能在全局索引表中找到，但这不能补全矩阵缺失的行含义。

结论：现有文件不是已证明匹配的一套原生2024评测资产。不能截断矩阵、猜偏移或
将旧同录制池结果标为native官方分数。需要配套矩阵/索引来源后才能重评分。
该问题不妨碍当前2025监督对照。

进一步限定：全局矩阵大于索引表所覆盖的两方向行数，本身不证明矩阵错误；索引表
可能只覆盖其子集。缺失的是可追溯的完整行对应关系，不能猜测其为前缀。
官方2023仓库data目录另经GitHub API核查，公开条目仅train.csv与val.csv，没有配套
data_indices.csv或全局矩阵可供交叉验证。这里结论是“尚不能认证原生重评分”，
而非“已证明官方矩阵有错”。

## 实现与论文表述的边界

2025参考集合采用论文定义；本地Processor在长度不等时经编码—解码得到25D目标，
等长时直接使用目标，随后在25D上算FRC/FRD。论文第4节描述latent-space比较。
本轮固定同一缓存以控制变量，但不宣称代码每一项等同论文文本。

归档TLCC代码在候选循环内部return，只评q0；本轮q0完全共享。不能把这一数值
作为四组同步性有无改善的证据。完整同步性评估需单独明确常数通道及十候选聚合，
不静默修改后仍称原代码分数。

附加诊断在新模型评测前固定：TRAIN均值/标准差50D尺度下的双向最近距离及能量分数，
均低为好。该描述子与mode prior同类，仅作辅助，不当作独立时序覆盖证据。
主比较用session聚类bootstrap5000次，seed270927，按context加权；仅单训练seed，
区间不含训练随机性，不作多重检验后显著性宣称。

## 训练完成后的执行

按顺序运行 `.venv/bin/python` 对应原项目环境中的：

1. `matched_eval.py --phase infer --device cuda:5`（已完成，不重复）
2. `matched_eval.py --phase score --workers 16`（已完成，不重复）
3. `matched_analysis.py`（已完成，不重复）

推理只输入source；共享q0只算一次，四组逐元素核验；完整all-session缓存逐文件校验。
全部三组预定对照、质量—多样性权衡和负结果已在报告中解释。mode未达到继续扩训
条件，下一步建议先验证匹配多样性水平的质量边界，不以单一FRC提升宣称成功。
