> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# CVPR2027 第一轮探索：有条件信号，但当前集合评测存在盲区

> 2026-09-27更正：本报告关于“all-session只是项目扩展”和把置乱不变性解释为
> 评测问题的表述已撤回或收窄。REACT2025论文明确使用同session全部恰当反应，
> 同组交换不降分不能据此判定集合不恰当。数值实验保留；解释与后续计划以
> [论文协议更正说明](PROTOCOL_AUTHORITY_CORRECTION.md)为准。

日期：2026-09-27。状态：本轮预定审计完成；没有新训练，没有替换模型。

## 决策

**继续探索“弱多参考监督下的条件反应集合学习”，但不继续以 all-session
FRC 上涨作为足够证据。** 首先补齐原生评测和条件关联验证，再做监督方式的
匹配实验。不能把先前岭回归分配探针失败解读为整个生成器没有条件信息。

目前不是 CVPR 主会完整贡献：没有新方法胜出、独立测试、人类恰当性判断，
也没有证明训练监督错配是当前性能的因果来源。

## 1. 实际完成的工作

- 原项目 `$LOCAL_PATH/react2025_new` 实际映射到 `$PROJECT_ROOT`。
  写入目标受限，因此本轮代码和新产物独立存放在本目录，原项目只读。
- 核验两套学生权重、归档预测、模型源码快照哈希；严格加载 state_dict。
  当前主目录缺少部分新模型模块，直接从实验快照导入，未恢复或覆盖主目录。
- REACT2025：mode-reference contrast student，VAL571 中 476 条可在同 session、
  同源长度组内置乱；95 条单例被明确排除。20 个 session。
- REACT2024：count4 minimal student，全部 VAL1124 可置乱。
  38 个方向组来自 19 个录制；置乱不跨方向，最终 bootstrap 合并同录制 P1/P2。
- 每套数据做 2701/2702/2703 三个固定置乱种子。不是三个独立训练 seed。
- 每套数据另做 96 条 session 均衡的新 GPU5 推理，共五种条件：原始、同组 donor、
  所有输入时间循环平移、音频时间均值、视觉两流时间均值。
- 两套各 96 条的原始及 donor 新输出与相应历史软输出 **最大绝对误差均为 0**。
  GPU5 单进程峰值 allocated 669,115,392 bytes；本轮未使用 GPU1/6，已无新任务占卡。
- 7 项单元测试通过，包含常量通道下的官方标量 FRC 一致性、多样性恒等式、
  无自交换与组/长度约束、参考抽样、输入扰动，以及共同参考池置乱不变性。
- 首次沙箱 GPU smoke 因无法访问 CUDA 失败；获授权后宿主 GPU5 smoke 与正式审计成功。

## 2. 完整匹配群体的结果

交换一整组十个预测，保留每个预测的全部时间轨迹和模式。
“置乱后”为三个置乱种子的均值；各次置乱的结果都保留在 JSON。

| 数据与数量 | 指标 | 正确输入 | 同组交换后 |
|---|---|---:|---:|
| REACT2025，476 | all-session FRC | 1.372480 | 1.372921 |
| REACT2025，476 | 配对槽位 q0 的同步 CCC | 0.059602 | 0.011228 |
| REACT2025，476 | 十槽平均同步 CCC | 0.020969 | 0.011812 |
| REACT2025，476 | FRDiv 等价方差形式 | 0.164290 | 0.164290 |
| REACT2024，1124 | all-session FRC | 0.656854 | 0.656854 |
| REACT2024，1124 | 配对槽位 q0 的同步 CCC | 0.025768 | -0.000420 |
| REACT2024，1124 | 十槽平均同步 CCC | 0.003593 | 0.000476 |
| REACT2024，1124 | FRDiv 等价方差形式 | 0.135939 | 0.135939 |

CCC 用真实 paired listener 在 source-only 中心裁剪后的有效同步帧，而不是把整段
listener 重采样后当作同步真值。q0 有配对监督，因此首先作为条件关联诊断。
FRC 是十候选求和，q0 CCC 是单候选，两者不可按绝对大小直接比较。

种子2701的“置乱减原始”q0 CCC差值与2000次录制/session聚类bootstrap区间：

- 2025：-0.047878，95%区间 [-0.062981, -0.036841]，20 clusters。
- 2024：-0.026030，95%区间 [-0.031944, -0.020287]，19 clusters。

其他两个置乱种子同向，最终统计见 `runs/final_analysis_v1/summary.json`。
区间是探索性、未经多重比较校正的区间，不能替代跨训练 seed 与独立测试。
原始 cached/GPU JSON 的2024初步区间按38方向组计算；应引用最终19录制组区间。

补充的确定性 paired+9 参考抽样 FRC：2025为0.962480→0.955715，
2024为0.484076→0.479738。它不是官方冻结 reference manifest，也不是2024原生
neighbor-mask协议；仅说明缩小参考池可能恢复一部分配对敏感性。

FRDiv 此处用 `2 * variance_across_candidates(ddof=1)` 的代数等价形式；与旧
float32 cdist 实现有约1e-6量级差异，不应当作性能变化。

## 3. 为什么 all-session 分数会这样？

对同一组内的每个 source，如果评测参考集合完全相同，记作 R_s，则

`score = mean_i h(G(x_i), R_s)`。

组内任意置换 pi 都满足

`mean_i h(G(x_pi(i)), R_s) = mean_i h(G(x_i), R_s)`。

这是直接的置换恒等式，不是新理论。它说明这种组内共同参考池评价无法从均值上
识别输入—输出对应关系被破坏；也不要求模型忽略输入。

本轮逐目标轨迹哈希核验：2024的38/38个匹配组，处理后的GT多重集合完全相同。
因此该数据上的 all-session 不变性是结构性结果，并非“模型极其稳健”。

2025的29个匹配组没有一个满足处理后GT逐字节完全相同，故不能声称其精确不变；
实测只是非常接近不变。处理器与对齐细节对差异的具体贡献尚未逐项归因。

边界：这里验证的是**本项目 all-session 扩展协议**。不能写成已推翻 REACT
官方协议或所有论文结果。原生协议仍待单独核验。未重算 FRD，不报告新 FRD。

## 4. 新推理的模态与时间诊断

以下为各自96条固定诊断子集的 q0 同步 CCC，与上表群体不同：

| 条件 | REACT2025 | REACT2024 |
|---|---:|---:|
| 原始 | 0.061183 | 0.027851 |
| 同组 donor | 0.014405 | -0.003826 |
| 输入时间循环平移 | -0.017422 | -0.007771 |
| 音频仅保留时间均值 | 0.042449 | 0.025059 |
| 视觉仅保留时间均值 | 0.028192 | 0.007922 |

2025音频和视觉时间信息都影响q0的配对一致性；2024视觉影响更明确，音频均值化
差值区间包含0，不能称两套数据都有显著音频时序效应。循环平移和时间均值化
可能离开训练分布，不是自然语言语义反事实，也不是人类行为的因果实验。

只取96条时 donor 集合未必是这个子集自身的排列，因此不要求它的平均 FRDiv
完全不变；严格分布保持性质只适用于上节完整匹配群体。

## 5. 对 CVPR2027 选题的影响

建议将研究问题收紧为：**扩展弱参考带来行为覆盖时，如何保持可验证的条件关联？**

候选题名：Conditional Correspondence-Preserving Reaction Set Learning from
Weak References。这是工作题名，不是已经完成的方法或新颖性声明。

建议三个证据层次：

1. 诊断层：公开可复现的组内交换审计，区分组分布质量与条件对应质量。
   再扩展到原生协议与至少两个强外部基线，不能只研究自己的评分扩展。
2. 方法层：同步配对监督负责可观测的时序关联；弱参考负责行为支持范围，而非
   无条件等同为逐帧正确标签。比较当前软参考聚合、动态信息保持、打乱条件控制。
   尚未证明具体哪种建模最优，不先堆LLM、DPO或新的大骨干。
3. 评价层：独立的人类恰当性、多有效答案与时间判断。q1..q9的低paired CCC不能
   用来宣判其错误，因为一个真实观察并不覆盖所有合理反应。也不能强迫所有候选
   复制q0来刷关联分数。需要同时量化模式覆盖与条件有效性。

## 6. 下一轮最小实验与停止条件（尚未启动）

- 先恢复/核验原生2025冻结10-reference清单和2024 neighbor协议，保持K=10、
  AU处理、source方向、裁剪和split一致。把本地与VAL重合的TEST排除出独立测试声明。
- 冻结训练初始化、source日程、参考池和预算，做四臂小实验：现有软参考监督；
  去掉非配对参考的控制；弱参考仅监督时间不敏感行为描述、配对负责动态；
  同一新增机制但打乱其source条件的假干预。方法细节需先写清，避免不可识别的
  共享加性匹配偏置。新增损失与算力单独计量。
- 能否同时保持模式覆盖、改善条件关联、避免真实同步和自然度恶化，是门槛。
  若收益只存在于all-session FRC、单一VAL或人工强制增加输出敏感性，停止扩大训练。
- 可以先准备100–200段分层人工评测材料，但不得生成“已经人工验证”的标签。
  条件交换不必天然构成不恰当样本，评分允许多个合适反应和静默反应。
- 若上述证据支持，再做多seed、真正独立的保留录制/参与者和外部基线。
  蒸馏先作为效率扩展，不自动算独立创新。

## 7. 文献边界

条件指标能识别“输出逼真但与输入无关”的问题早有研究，不能把这种一般现象
当作首次发现：[Conditional Frechet Inception Distance](https://arxiv.org/abs/2103.11521)。
本轮新增的是本项目两个模型的可复现证据，而非已证明的领域级新颖性。

多恰当反应的任务与评价定义已有
[What, Why and How?](https://arxiv.org/abs/2302.06514)，并有
[REACT2024](https://arxiv.org/abs/2401.05166) 的挑战协议。需尊重一对多答案，
不能用单一配对误差代替完整恰当性评价。

[UniLS, CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Chu_UniLS_End-to-End_Audio-Driven_Avatars_for_Unified_Listening_and_Speaking_CVPR_2026_paper.html)
区分内在运动先验与音频调节；[Avatar Forcing, CVPR2026](https://taekyungki.github.io/AvatarForcing/)
已使用缺少用户条件的生成结果开展偏好学习；
[ReactDiff, ACM MM2025](https://arxiv.org/abs/2510.04712)已有时序先验与蒸馏实验。
我们的潜在差异必须落在弱参考的监督可靠性与条件关联保持，并由实验支撑。

## 8. 产物与复现

- 运行实现：`audit.py`；分析：`analyze.py`；测试：`test_audit.py`。
- 协议：`PROTOCOL.md`，末尾明确记录统计聚类修正。
- 完整置乱：`runs/react2025_cached_v1`、`runs/react2024_cached_v1`。
- 新推理：`runs/react2025_gpu_v1`、`runs/react2024_gpu_v1`，含五种输入的软预测。
- 最终统计：`runs/final_analysis_v1/summary.json`。逐context记录和donor映射可追溯。
- 原项目 git status 与开始时一致；无提交、推送、数据上传或服务修改。

从本目录运行，例如（输出目录必须尚不存在）：

```bash
$LOCAL_PATH/python -m unittest -v test_audit
$LOCAL_PATH/python audit.py --year 2025 --mode cached --output runs/react2025_cached_replication
$LOCAL_PATH/python audit.py --year 2024 --mode gpu --examples 96 --device cuda:5 --output runs/react2024_gpu_replication
```

GPU命令需要宿主CUDA权限。复现也应先检查卡占用。无跨模型挑选checkpoint，
无新参数搜索；本轮是证伪和研究路线选择，不是新SOTA宣称。
