> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 下一轮：不同蒸馏方式的证据、适配与最小对照

2026-09-28。用户要求保留消融、考察不同蒸馏方式。
本文是文献及本地实现核对后的研究设计，**不是新实验结果，也没有启动新训练**。
继续现有 T1 参考选择→specialist 教师→共享十输出学生路线，不改为剪枝或扩散学生。

## 1. 已经做过什么，不能重新包装

| 方法 | 本地已有证据 | 下一步的身份 |
|---|---|---|
| 轨迹输出 KD | MSE + (1−CCC) + 0.05 velocity，T1/mode 两种教师均有 | 必需基线 |
| 槽位距离 KD | T1 教师下 45 对距离，AU/VA/expression 分组 | 已有关系对照；若用于 mode 教师需重新匹配训练，不能挪旧数值 |
| centered-residual KD | T1 和 mode 教师都做过；新四组受控结果已完成 | 保留现有方法，检验其机制 |
| AU decision KD | 对教师 AU 取整后做置信度加权 BCE | 已做，不是新的 soft Bernoulli 蒸馏 |
| AU-logit 精修 | T1 residual 后再训 4150 步，有同预算 continuation | 已做；相对 continuation 没有 FRDiv 优势 |

T1 历史 distance 的 FRD 100.095737，高于 residual 的 96.617181；不能把距离项当作
未经试验的“新方向”。这些是不同教师/旧日程的数据，不与最新 mode 学生直接排榜。
已有 centered-residual 与全 pairwise difference-vector 匹配仅相差常数；改个名字、
重写成 pairwise 形式、单独再加一阶 velocity，都不构成新机制。

## 2. 论文怎么做，哪些能迁移

以下只把实际核对过的论文作为依据，不宣称覆盖所有 2026 文献。

| 方向与论文 | 已有思想 | 对我们的适配边界 |
|---|---|---|
| [Hinton 等，输出蒸馏](https://arxiv.org/abs/1503.02531) | 用教师软目标传递知识 | 当前 25D 不是一个分类分布，不可整体 softmax 后做 KL |
| [RKD，CVPR2019](https://arxiv.org/abs/1904.05068) | 样本之间的距离/角度关系 | 我们现有项是同输入的输出槽位关系；属于适配，不是原版 RKD 的逐项复现 |
| [Ensemble Distribution Distillation](https://arxiv.org/abs/1905.00076) | 学 ensemble 预测的分布，而非只保留平均 | 原方法侧重分类/不确定性；不能把十个刻意分工的 specialist 自动解释为概率校准的随机 ensemble |
| [回归/分类 ensemble distribution 框架](https://arxiv.org/abs/2002.11531) | 保留 ensemble 的不确定性分解 | 说明回归分布蒸馏已有先例；不等于我们固定十条反应的条件分布已被认证 |
| [WKD，NeurIPS2024](https://arxiv.org/html/2412.08139v1) | 用 Wasserstein 传递类别关系或特征分布 | 分类/检测方法；整段反应的集合 OT 是任务适配，不应冒称原版 WKD 或新发明 OT |
| [FitNets](https://arxiv.org/abs/1412.6550) | 中间层提示/特征模仿 | 可做强通用对照，但十教师 hidden 空间不能不经核对就平均 |
| [KDTF，CVPR2025](https://arxiv.org/html/2503.22201v1) | 将多模态的个体内/个体间 latent 知识分开蒸馏给少模态学生 | 我们不是多行人、也未限制学生输入模态；借鉴语义分层，不照搬其任务叙事和 KL |

同方向 [ReactDiff，Cheng Luo 等](https://arxiv.org/html/2510.04712v1) 已含时间运动学与
面部动作约束，§4.2 可核对。因此“给反应加速度/时间约束”本身不能当作新颖性。
该版全文没有检索到 distill，不延续先前未经确认的“附录蒸馏”描述。
其 FaceVerse/ARKit blendshape 关系也不能直接套到我们 15 AU 的通道编号上。

## 3. 最值得优先比较的三个新方向

### A. 按输出类型区分的软目标蒸馏：先补一个合适的通用基线

已核对实际模型 `constrain_reaction`：前 15 维 sigmoid AU，中间 2 维 tanh VA，
后 8 维 softmax expression。因此可在原 basic KD 上增加分组 soft-distribution 项：

- AU：逐通道 Bernoulli KL，保留教师软概率；不是 15 类互斥分类，也不是教师取整 BCE。
- expression：8 类 categorical KL；只在这 8 维内归一化。
- VA：仍按连续量回归，不制造没有定义的概率分布；沿用原 MSE/CCC。
- 第一轮固定温度 1，用同一 TRAIN 梯度标定辅助项强度；不在 VAL 搜温度。

这是现有输出语义支持的基线，不应单独称为 CVPR 创新。风险是教师不可靠的极端
概率被放大；先检查合法范围、有限梯度、mask 和 teacher detach，完整报告结果。

### B. q0 保持对应，其余九条按整段匹配：检验槽位绑定是否有代价

问题：现有 KD 要求学生第 k 条模仿教师第 k 条，但主评测不看候选编号。
这只是一个动机，不代表现有槽位绑定已经被证明有害；教师分工可能反而帮助训练。

最小实现选择 **整段 Hungarian 一对一匹配**，先不同时加 Sinkhorn、MMD 和时序新项：

1. q0 仍与配对教师 q0 对应；只是固定损失对应，不冻结共享学生参数。
2. 每个输入独立构造其余 9×9 代价矩阵，用原基础轨迹 KD 的同一定义
   `MSE+(1−CCC)+0.05 velocity` 计算整段有效帧代价，固定分配后反传选中代价。
3. 一条预测和一条教师反应一对一使用，各九条都参与；不允许所有候选只追同一教师。
4. 同一段轨迹内不能逐帧更换教师，避免拼接成不存在的反应。
5. 对照只改变 assignment：原始 slot KD 与整段 set KD；不同时更换教师/通道权重。

这等价于等质量有限集合上的硬匹配基线，不是新的 OT 理论。若 later 用 soft transport，
用 [Sinkhorn divergence 的去偏定义](https://proceedings.mlr.press/v89/feydy19a.html)
并说明自项，不能把未经去偏的熵正则 OT 当成完全相同的距离。
不能先把教师轨迹做 transport 加权平均再称“保留多峰”；平均可能再次抹平差异。

集合蒸馏风险：随输入改变的 assignment 可能破坏跨窗口槽位身份；本轮只能研究
离线 clip，不能宣称解决在线长视频一致性。也只是在复制十教师的经验集合，不保证
覆盖所有有效 GT。若保留固定 slot 基础损失再加集合项，则不是纯置换不变对照，须另命名。

### C. 动态残差蒸馏：区分“偏置不同”与“变化过程不同”

当前 residual 主要改善 FRD，而连续多样性没有明确改善。值得检验它保留的是否主要
是静态槽位偏置。下面是待验证的任务适配假设，不是已经证明的缺陷。

对每条有效反应先减同一时刻的十槽位均值，再减该槽位有效时间均值：

`r_k(t) = y_k(t) − mean_j y_j(t)`，`d_k(t) = r_k(t) − mean_valid_time r_k(t)`。

对 student/teacher 的 `d` 按现有 AU/VA/expression TRAIN 尺度匹配，保留 basic KD
监督整体水平。第一轮只加这一项，与原 residual 分开比较；之后才能讨论多尺度
窗口或输入—反应时延。该项排除时间常量的槽位差异，但仍是误差分量重加权，
**不天然构成新颖算法**；必须证明保留下来的差异是合理动态而非抖动。

必要后续消融：原 residual / dynamic-only / static-only，保持辅助梯度预算一致。
不能用本来就有的 velocity 项换名；不能用打乱时间的 mean/std 描述子代替时间结构。
输入相关性检查是辅助证据，不否定 all-session 主评测，也不要求所有合理反应都逐帧
复现唯一配对听者。

## 4. 暂缓的路线

- 中间特征 KD：有价值的后续基线；需要新缓存和层对应核验。共同预训练初始化并不
  保证 fine-tune 后的教师 hidden 坐标仍完全一致。若加入训练用投影头，额外训练成本
  单列；推理结构不变。不能平均十教师特征后把分工蒸没。
- MMD/energy/set distribution：与 B 同属集合匹配家族；第一轮只做一种可解释的
  assignment 基线，不把多个核/温度一起搜索。低维描述子匹配不能等同整个时序分布。
- diffusion/consistency distillation：当前教师是确定性 specialists，切换会改变教师和
  生成范式，不是本轮同教师蒸馏对照。
- 仅匹配 ensemble 均值：可作坍缩负控，但不作为多反应主方案。

## 5. 建议下一轮矩阵（待实施，非已运行）

保留最新四组完整结果。新建独立 run，固定当前 mode teacher cache，不改选择策略：

| 编号 | 训练定义 | 唯一主问题 |
|---|---|---|
| B0 | 原 basic KD | 基线 |
| B1 | 原 residual KD | 现有方法 |
| N1 | B0 + typed soft KL | 利用正确的输出概率语义是否有效 |
| N2 | q0-fixed、九条 whole-clip assignment KD | 固定槽位绑定是否影响可压缩性 |
| N3 | B0 + dynamic residual KD | 动态差异监督是否优于全部 residual |

B0/B1 只有在新代码实际重放、loss 等价、初始化/数据日程/dropout RNG 一致性均通过
时才复用本轮模型；否则在新目录重跑同预算控制，不能仅看配置相似就认定公平。
旧 T1 distance/AU/logit 是历史证据，不能混入此 mode 教师的受控矩阵；需要时独立重跑。

共同要求：

- 相同 3,803,201 参数、共同配对预训练初始化、K=10、3320 TRAIN views、8300 步；
  不以“纯随机初始化”描述，训练总成本与仅学生更新数分别统计。
- 所有归一化、辅助系数、assignment 规则仅用 TRAIN 冻结。辅助项统一使用固定
  TRAIN 探针的初始梯度比例标定，不能逐方法在 VAL 挑最优系数。更换教师需重标定。
- 先单测 + 小步 smoke：mask/padding、零有效长度拒绝、teacher detach、有限梯度、
  同输出自损失、q0 不参与重排、整段重排不跨帧、assignment 一对一。
- VAL571、论文 all-session GT、十候选全参考精确 FRD、不改变 AU 取整主评测；
  FRC、FRD、FRDiv、FRVar、soft FRDiv 全报，不只选择有利指标。
- 额外记录连续/静态/动态槽位差异、教师拟合、assignment 使用与稳定性；这些不是
  新的官方主指标。后续有效模式覆盖需独立诊断与人评，不从 FRDiv 自动推导。
- session bootstrap 仍只刻画该 seed 的 VAL 不确定性；方法筛查后再用多 seed、
  预先固定的强度网格和独立 TEST 检验质量—多样性边界。记录探索性选择轨迹。
- 使用 GPU1/5 的持久服务可复用已有机制；本次仅归档和调研，没有新启动 GPU 作业。

## 6. 投稿层面的判断

最有希望的研究问题是：**如何压缩同一刺激下的多种合理反应，而不把合理差异变成
静态偏置、平均反应或纯幅度扩张？** 这比“换一种 KD loss”更贴近数据与任务。
但当前尚无证据证明 N1/N2/N3 优于现有方法。若仅改变权衡位置或 AU 阈值附近的输出，
应保留负结果，不把标准 KL/匹配/中心化运算包装成已成立的 CVPR2027 核心贡献。
