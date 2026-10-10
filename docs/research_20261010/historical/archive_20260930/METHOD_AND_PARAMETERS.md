> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 方法与实验参数

本文对应 2026 年 9 月 30 日建议冻结的论文路线。旧路线完整保留于原始报告；下面的公式不追溯替换旧实验定义。

## 问题与方法主线

输入为说话者的整段音频特征、25D 面部属性与 58D 3DMM 条件；输出十条合理且不同的 listener 25D 反应。数据中同 session 有多条可接受反应，但它们不一定与当前 speaker stimulus 在真实时间上同步。

核心矛盾是：只拟合一个固定非同步 GT 容易损伤质量；完全自由地追逐最好匹配参考又容易丢失十个输出的行为身份。方案保留固定 GT 的模式锚定，同时放松 temporal reference，然后将十个教师压缩进一个按 expert ID 条件化的学生。

```text
TRAIN 同 session GT
  → 同名 paired q0 加九条固定 Max-FRDiv GT
  → 十个教师保持固定模式锚点
  → q1 到 q9 使用模式先验约束的多参考 CCC
  → 固定 TRAIN 输入视图与十教师软输出缓存
  → 共享学生加十个 expert ID
  → 推理只用 speaker 条件 输出十条反应
```

暂用描述性名称：Mode-preserving multi-reference teacher distillation。不是已确认独占的新算法名称。多假设学习、log-sum-exp、KL 变分恒等式和 relational KD 均是已有工具；论文贡献应落在任务矛盾、模式与参考的分离以及可复现实证。

## 固定 GT 选择

- q0 是真实同名 paired listener，不用随机 GT 替代。
- q1 到 q9 在 TRAIN 对向角色同 session 池内选择九条不同 GT；成员集合按轨迹 FRDiv 最大化，强制包含 q0。
- 默认 Max-FRDiv 为 farthest/swap 等启发式可行解，不自动称全局最优。只有求解器证实 optimal 的小规模实验才叫 exact。
- descriptor prototypes 用于规范 q1 到 q9 的身份排序，不把 descriptor distance 冒充轨迹 FRDiv。
- 每条 TRAIN 输入的目标路径固定；教师 source crop 可以按预设 epoch/seed 改变。固定路径不等于每次迭代都用同一时间片。
- T0 对照为 q0 加九条固定 random GT。当前最终教师路线尚未完成“仅将 Max-FRDiv 改成 random，其余最终算法全相同”的端到端消融；历史 T0/T1 仅支持其自身合同。

主要实现：`maxfrdiv_teacher_manifest.py`、`prepare_maxfrdiv_teacher_manifests.py`、Mam `regnn/conditional_data.py`。

## 教师目标与模式先验

第 k 个教师输出 p_k，固定目标 y_k，同 session TRAIN 参考集合 Y_s。教师 q1 到 q9 的目标为：

```text
L_teacher = MSE_valid(p_k, y_k)
          + [1 - q_tau(p_k, Y_s, rho_k)]
          + 0.05 L_velocity_masked

q_tau = tau log sum_j rho_k(j) exp(CCC(p_k, y_j) / tau)
tau = 0.05
```

CCC 在每条轨迹各通道上计算后平均，不是正式集合 FRC 的十条求和。常数通道的评估值按官方约定为零；当前训练采用修正后的数值稳定梯度实现。历史 v1 保留但不能替代 corrected control。

rho_k 以固定 y_k 为中心：先提取各 GT 每通道时间均值与标准差，组成 50D descriptor；使用全部 TRAIN descriptor 的逐维 std 标准化（下限 0.03）；对 source 有效长度下对齐后的参考计算标准化平方距离，构造 Gibbs 先验。

```text
rho_k(j) ∝ exp(-D(phi(y_k), phi(y_j)) / bandwidth_k)
H(rho_k) = log(effective_count)  （可达到时）
```

带宽用 TRAIN 距离二分确定；相同 descriptor 的重复参考保留对称性，不能为凑指定数量任意破坏平局。2025 effective count=8；2024 接受版本为 4。**不是仅抽取 8 或 4 条 GT**：CCC 仍计算整个 TRAIN session 参考池。

等价形式为 `max_pi E_pi[CCC] - tau KL(pi || rho_k)`。均匀 rho 回到 uniform-reference；点质量 rho 回到 fixed-reference。这是解释机制的标准变分恒等式，不是新的数学定理。

q0 使用原 paired anchor，不加上述多参考项。q1 到 q9 保留 fixed-target MSE、目标集合及原 velocity mask。跨 listener 非同步目标不强加同时间速度监督；mask 以实际数据 loader 为准。

## 教师架构和训练参数

| 项目 | 当前记录 |
|---|---|
| 模型 | ConditionalREGNN；每教师一个输出，十模型集成 |
| 输入维度 | audio 768，speaker emotion 25，speaker 3DMM 58 |
| 条件融合 | 三路 LayerNorm → Linear 到 256 → GELU；拼接后融合回 256 |
| 时间骨干 | sinusoidal position encoding；4 层 Transformer，8 heads，hidden 256 |
| 输出 refinement | graph=2，graph_dim=64，graph_heads=4；25 attribute token attention，无固定解剖拓扑 |
| 输出头 | linear；不是已经替换成 MLP 的实验 |
| 输出约束 | 15 AU sigmoid，2 VA tanh，8 expression softmax |
| Dropout | 0.1 |
| 每教师参数 | 3,720,129 |
| 十教师参数 | 37,201,290 |
| Seed | 1 |
| Batch | 32 |
| Optimizer | AdamW；lr=1e-4；betas=(0.9,0.99)；weight_decay=1e-4 |
| Gradient clip | global norm 1 |
| Precision | BF16 autocast |
| LR schedule | epoch cosine，名义 T_max=50，eta_min=5e-6 |
| 实际训练预算 | 当前参考教师每个 600 optimizer updates；不是完成 50 epochs |
| Clip | 最多 750 帧，valid prefix mask；padding 不进入 loss |
| TRAIN views | REACT2025 为 3320 双角色 source records；2024 学生缓存为 6340 合格 views |
| Listener 3DMM supervision | 当前路线关闭；只预测 25D，58D 仅作为 speaker 条件 |
| 训练 DTW | 当前教师不使用 SoftDTW；多参考开销来自 CCC |

2025 九个新 specialist 为 5400 updates；加复用 q0 的预算口径为 6000。原始记录中 mode-reference 每 update 的训练时间约为历史 T1 的 2.46 倍，不能用“同 600 updates”声称同 FLOPs。探索过的 uniform/v1/v2 训练成本应另记，不能都隐藏进最终单 run。

2024 转换 TRAIN 排除了仅有九条 GT 的 18 个 RECOLA directional contexts，原因是严格十个不同目标的资格要求；不是因为 VAL 表现删样本。其 session allowlist 和 target manifest 随原记录归档。

## 对齐和数据流

训练与评估必须区分：

| 环节 | 对齐与目标 |
|---|---|
| paired q0 TRAIN | 原始同名配对，native 同步裁剪和有效 mask |
| specialist TRAIN | 当前 `relative_time_masked`，AU nearest，VA/expression linear 并处理合法输出；不是旧 diffusion short-tail-fill |
| 多参考 TRAIN CCC | 使用整个原始 TRAIN 参考池，对齐到每条 source 的有效长度；不调用 VAL learned postprocessor |
| 学生 TRAIN | 固定缓存 view_epoch/source_seed，逐样本核对 clip/start/length/view ID；教师输出以 FP16 存储，读取后 float32 |
| VAL source | speaker-only center crop，最大 750，短 source 只评有效前缀；3DMM 使用 FaceVerse mean/std |
| VAL all-session GT | 对向角色同 session 全部 listener；各预测分别做 best-reference 匹配 |
| VAL GT 长度处理 | 现有官方 learned postprocessor；等长可直通，不等长使用 learned VAE；每 context 固定随机种子和缓存 |

不能把 VAL GT 处理概括成一律线性伸缩。历史 dataset ceiling 的统一 750 center/linear 操作是另一个实验，也不能用来描述当前 FRC/FRD target processing。

## 共享学生和损失

学生类 `ExpertConditionalREGNN`，共享上述 backbone，用 10×64 的 expert embedding 经 `Linear(64,256) → GELU → Linear(256,256)` 产生 slot offset：

```text
H = shared_encoder(speaker)
H_k(t) = H(t) + modulation(embedding(k))
S_k = shared_decoder(H_k)
```

一次编码后对十个 slot 展开解码；不是十份学生 backbone。参数总数 3,803,201。相对十教师参数缩减约 89.8%，但解码仍有十个 slot，不能据此声称推理恰好十倍加速。

当前建议简化损失：

```text
L_student = MSE_valid(S, stopgrad(T))
          + [1 - CCC_valid(S, stopgrad(T))]
          + lambda_R L_centered_residual

R_S,k = S_k - mean_l S_l
R_T,k = T_k - mean_l T_l
L_centered_residual = mean_group mean_valid((R_S - R_T)^2 / scale_group^2)
```

分组为 AU15 / VA2 / expression8，三组等权平均。loss 匹配有符号、带幅度的 slot 差异，过度扩张同样受罚，不是 pairwise repulsion。它按固定教师 ID 对齐，不是无序 Hungarian set matching。centered residual 与 pairwise difference-vector MSE 具有标准代数关系，不能称为新定理。

学生训练参数：

| 参数 | REACT2025 | 转换版 REACT2024 |
|---|---:|---:|
| Seed | 1 | 1 |
| 固定 TRAIN views | 3320 | 6340 |
| Epochs | 10 | 10 |
| 实际 updates | 8300 | 15850 |
| Batch / workers | 4 / 4 | 4 / 4 |
| 初始化 | 固定 T1 paired q0 backbone + 同 seed expert-ID 模块 | 本数据集固定 T1 paired q0 backbone + 同 seed模块 |
| AdamW lr / decay | 1e-4 / 1e-4 | 1e-4 / 1e-4 |
| AdamW betas | 0.9 / 0.99 | 0.9 / 0.99 |
| Scheduler | 10-epoch cosine，eta_min=5e-6 | 同左 |
| Precision / clip | BF16 / 1 | BF16 / 1 |
| MSE / CCC weight | 1 / 1 | 1 / 1 |
| 简化版 velocity weight | 0 | 0 |
| 保留 Full 的 velocity weight | 0.05 | 0.05 |
| centered-residual weight | 0.3291488264202468 | 0.21374727996059517 |
| AU scale | 0.32778552174568176 | 0.2697319984436035 |
| VA scale | 0.34031346440315247 | 0.18466883897781372 |
| Expression scale | 0.25885310769081116 | 0.20045259594917297 |

scale 来自最多 32 个 TRAIN 缓存样本的教师 slot pair 距离；初始梯度校准在其中 16 个样本、四个 batch 上以目标辅助/KD norm 比 0.25 确定系数并截顶 1。**0.25 是校准目标，不是最终 loss 系数。** 删除 velocity 时继承 Full 的 scale 与系数，不重新调优，也不声称删除后即时梯度比仍为 0.25。新教师应重新按 TRAIN 校准，不能机械复用旧数值。

Direct-GT matched control 仍使用同一个预训练 q0 初始化，只将缓存 target 从教师输出改成相同固定真实 GT；因此是 output KD 消融，不是完全 teacher-free from scratch。

## 指标定义与加速

设每个 context 十条 AU-rounded 预测 p_i，处理后的同 session GT 为 y_j：

```text
FRC(context) = sum_i max_j mean_channel CCC(p_i, y_j)
FRD(context) = sum_i min_j official_weighted_DTW(p_i, y_j)
FRDiv(context) = mean_{i != l} ||p_i-p_l||^2 / (25T)
```

FRC 是十条 contribution 的**和**，可以大于 1；不是“超过 100% 的相关性”。FRC 与 FRD 独立匹配各自最优 GT，不要求选中训练固定 GT、同一 GT 或一一匹配。正式均值按 context 加权，不等同 session-balanced mean。FRDiv 不依赖参考 GT，固定预测下改成 random10 不会改变 FRDiv。

FRD 的 group weights 为 AU15 的 DTW×1/15、VA2 的 DTW×1、expression8 的 DTW×1/8；并非简单25D逐帧MSE。FRVar 对每条预测沿时间使用 torch.var 的默认样本方差，再平均 context、slot、channel。历史 TLCC/FRSyn 记录原样保存；当前诊断指出 legacy TLCC 的循环提前 return / 常数轨迹问题，故不能将固定 49 或新 motion-energy proxy 当作已验证的同步质量。当前主表不补造第五项结果。

当前快速评估复用向量化、scalar-parity FRC；Numba rolling exact DTW；共享数组与常驻 CPU workers；GT/预测缓存；GPU 模型推理并按 context 分片。不是近似 DTW，不通过减少 GT 池“加速”。固定 `eval_seed=1234`，完整 571 / 1124 context，cache key 含 context/length/处理器及 GT provenance。

随机 GT 对照固定预测与处理后全部参考，做 100 个 seed 的完整 VAL571 FRC；paired+9 和 uniform10 分开。此检查只改变 sampling，不重算 FRD，不是完整官方 TEST 复现。REACT2024 当前 all-session 是转换格式上的统一分析协议，不是其原生 neighbour-GT / S_MSE leaderboard 协议。

## 计算成本与部署边界

2025 Residual-only 学生记录训练 1072.35 秒、峰值 allocated 6,020,531,200 bytes；2024 为 2048.98 秒和同峰值。全 VAL 推理记录约 20.03 / 20.97 ms 每 context；2025 十教师约 37.09 ms。上述来自共享服务器、FP32 评估且无独立 warmup，不是受控速度 benchmark。完整端到端成本还包括十教师训练、TRAIN cache、GT postprocessing 与 CPU exact DTW。

当前模型使用整段 context；本归档不宣称严格在线因果推理。无 VLM 标签、无推理噪声、无后处理增幅、无 GT selector，未启动 32/64-candidate 模型。它们是历史诊断或未实施计划，不属于当前核心算法。
