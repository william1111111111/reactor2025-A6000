# 当前技术路线：质量约束下的候选池压缩

更新日期：2026-09-21。本文只发布技术路线，不包含数据、权重、样本标识或本机运行信息。下述已实现模块部分仍在本地实验中，本次文档提交不代表对应源码已经发布。

## 核心问题

内部生成多于 10 条候选，最终输出固定 10 条：在不降低 FRC、不增加 exact FRD 的条件下，能获得多少真实的多样性收益？先用 GT oracle 测量现有有限候选池的可达范围，再决定投入生成器还是 selector。

当前路线为：固定行为专家 → 专家条件学生蒸馏 → 冻结候选池 → 质量约束 Select-10 → 条件成立后开发 inference-safe selector。

## 现有基础与边界

- 早期 CoReSet M1 的 All-Seen descriptor supervision、M2 的 TRAIN-derived behavioral routing 是前序研究；不再以细粒度 routing weight sweep 为主要方向。
- 当前学生分支比较 Basic KD、Distance KD、Centered-direction KD。增加轨迹幅度可以提高 FRDiv，但并不自动保持质量。
- 当前 Select-10 的 matched baseline 是 Basic-KD 学生原始 10 条输出，不是早期 CoReSet M1，也不是 Mam-Reactor P4。
- 模型及权重冻结。当前阶段不训练新的 32/64-candidate generator，不加普通 prediction-prediction repulsion。

## 第一阶段：有限候选池的 oracle 诊断

### 候选池与评估合同

- R30：三个学生各提供 10 条原始预测。
- E60：R30 加上三个学生各自的 1.5 倍 soft residual expansion；先做合法通道投影，再按指标协议 round AU。
- 合并完全相同的候选，同时保留 baseline 的 10 个实例，保证 baseline 集合始终可行。
- 固定输出 K=10；按每个 context 的 all-session opposite-role GT 独立匹配质量，不限制为训练时选定的 GT。
- 固定 center-750、eval_seed=1234 与 GT cache。这是本地 VAL all-session 诊断，不应标作官方 TEST 结果，也不应与抽取 10 个参考 GT 的结果混用。
- GT 仅供 oracle 诊断使用，不能作为部署时 selector 的输入。

对每个 context、每条候选预计算官方口径 FRC contribution、exact-FRD contribution，以及 prediction-prediction FRDiv 距离。FRC 和 FRD 分别匹配 GT，不要求它们选中同一条参考轨迹。

主约束逐 context 成立：FRC(selected) ≥ FRC(baseline)，exact FRD(selected) ≤ exact FRD(baseline)。不以 dataset-average 补偿局部质量损失，不为了获得漂亮数字放宽约束。浮点可行性容差与实验质量 slack 必须明确区分。

### 可行下界与数值上界

将官方处理后的轨迹展平并除以 sqrt(25T)，记为 u_i。令 G_ij=u_iᵀu_j，n_i=||u_i||²。二值选择向量 z 满足 Σz_i=K 时，官方 FRDiv 可写为：

```text
F(z) = [2K nᵀz - 2 zᵀGz] / [K(K-1)]
cᵀz ≥ cᵀz_baseline
dᵀz ≤ dᵀz_baseline
```

实际可行集合给出下界：baseline 初始化、deterministic swap refinement、限时整数优化；任何输出都重新验证质量及目标值。

将 z 放松到 [0,1]，得到线性约束下的凹目标最大化。近似求解器返回的 primal objective 本身不能充当上界。使用凹函数切平面及线性优化的对偶可行值构造独立重算的上界。

具体地，令 g=∇F(x)，质量约束写作 Az≤b。任意 λ≥0、自由变量 ν 给出：

```text
UB = F(x) - gᵀx + λᵀb + νK
     + Σ_i max(0, g_i - (Aᵀλ)_i - ν)
```

当前实现采用 float64 并增加数值 allowance，因此称为数值上界诊断，不声称是区间算术意义上的形式化证书。该上界仅针对当前有限候选池和逐 context 质量约束，不是所有生成器的多样性天花板。凹松弛与对偶界属于已有优化工具，不单独作为新理论贡献。

### 扩大验证与归因

VAL24 pilot 已完成；下一验证范围为 full VAL571。在完整汇总、检查遗漏与重复 context 之前，不把部分结果称为 full-VAL 结论。

报告同时给出 baseline、可行 FRDiv 下界、数值上界及 gap；FRC、exact FRD、FRVar；逐 context 分布；选中候选的模型及变换来源。检查是否形成 high-quality core + diverse specialists，而不是只观察均值上升。

## 评估加速与公平性

- 复用 Mam-Reactor 的 Numba rolling exact DTW，保持距离定义不变，不用近似 DTW 替换 exact FRD。
- 缓存冻结模型预测与 GT；FRC 向量化并与原实现核对。
- exact FRD 使用常驻多进程、共享缓存和细粒度候选任务；不同 context 分片并发，限制每个进程的 BLAS/OpenMP 线程，避免过度订阅。
- 分别报告指标耗时与组合优化耗时。GPU 适合模型推理，但缓存后的当前瓶颈不一定是 GPU 工作负载。
- 固定 seed、评估子集与预处理；训练对照固定初始化（适用时）、source schedule、GT draws 和预算。记录代码版本、权重及关键输入哈希。
- 原有实验的 legacy diffusion-compatible target 处理与未来可能修正的时间对齐协议须单独标记，不能混作同一公平对照。

## 第二阶段：inference-safe 质量约束选择（计划）

只有 oracle 收益得到验证后才启动。质量评分器只看 speaker/context 与 candidate feature；GT 可以构造 TRAIN teacher target，但不能进入推理输入。先做简单 deterministic greedy，不优先使用复杂 subset network。

探索 baseline-relative robust selection。令 a=z-z_baseline，预测质量为 c_hat、d_hat，候选同时误差界为 eps_c、eps_d：

```text
c_hatᵀa - Σ_i eps_c_i |a_i| ≥ 0
d_hatᵀa + Σ_i eps_d_i |a_i| ≤ 0
```

在同时覆盖事件成立时，上述约束保证相对于 baseline 的质量不退化；未替换候选的误差自动抵消，baseline 永远可行。这仍是研究设想，尚未完成 scorer、独立 calibration 或风险保证验证。必须处理 session 相关性、选择偏差及可能过度保守的问题，不能把边际覆盖误写成逐样本绝对保证。

## 后续决策

- 若有限池严格质量上界仍低：优先改善 generator 的候选支持范围，再考虑 32-candidate 模型。
- 若 oracle 下界显著提升且 gap 小：优先学习质量评分与 Select-10，并测量 oracle-to-inference gap。
- 若 gap 大：先改善优化诊断，不能将启发式未找到更好集合解释为生成器无能力。
- 只有上述证据成立后，才开展集合蒸馏、overcomplete generator 或语义 prototype；不提前做全面 VLM 标注。

任何扩展都报告候选数、参数量、昂贵 temporal-loss budget、训练吞吐、推理延迟及 selector 开销。最终论文主张应建立在独立验证和可复现的质量—多样性结果上，而不是仅凭 oracle 或幅度扩张结果。
