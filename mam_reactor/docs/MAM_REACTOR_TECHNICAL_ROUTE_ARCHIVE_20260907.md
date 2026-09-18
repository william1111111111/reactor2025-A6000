# Mam-Reactor 技术路线归档

**归档日期：2026-09-07**  
**项目代号：Mam-Reactor**  
**代码/历史文档名称：EQR-Mamba（Emotion-Query Residual Mamba）**

## 0. 归档结论

Mam-Reactor 的核心路线已经收敛为：

> 用 Conditional-REGNN 提供稳定的确定性 reaction anchor，再用显式 emotion
> queries、残差式 Bi-Mamba 和多目标 set alignment 生成一组互补反应；在线场景
> 通过固定窗口和输出尾部拼接实现低延迟推理。

当前必须区分四种状态，不能把它们的数值混成一个“最终模型”：

| 状态 | 路线 | 当前结论 |
|---|---|---|
| **离线证据主线** | 750 帧全序列、冻结 anchor、10 个候选、balanced Sinkhorn、isolated VA | 已有完整 1,142 样本全量结果，可作为当前论文主结果候选 |
| **target-supported 新实现** | 独立 sigmoid support gate、predicted-support q9、banded Soft-DTW | 已实现、单测和 smoke 通过；尚无三 seed 长训练和正式全量评估，不能继承旧主结果 |
| **官方 online-window 扩展** | 60 帧 speaker context → 30 帧 listener output | 已完成 full-test 诊断，但 FRC 明显低于离线主线，暂不提升为主线 |
| **严格历史因果诊断** | 固定 30 帧输出，只使用当前边界之前的 speaker 历史 | h=750 结果有潜力，但 exact FRD 更差，暂作为 online 研究分支 |

## 1. 任务与输入输出

目标是从 speaker 的多模态行为生成多个合理、自然且有语义差异的 listener facial
reactions。输入不读取视频，而读取预提取特征：

| 输入 | 形状 | 含义 |
|---|---:|---|
| speaker audio | `[B,T,768]` | 语音特征 |
| speaker facial attributes | `[B,T,25]` | 15 AU、VA、8 类表情概率 |
| speaker 3DMM | `[B,T,58]` | FaceVerse 系数 |
| valid length | `[B]` | 有效帧长度及 padding mask |

标准离线输出为 `[B,10,T,25]`；在线输出为固定长度的 30 帧块，多个块可以按时间
顺序拼接。25 维输出继续按 AU、VA 和 expression 三个 channel group 解释。

## 2. 当前推荐的模型结构

```text
speaker audio / AU-VA-expression / 3DMM
                    │
                    ▼
        Conditional-REGNN deterministic anchor
                    │
       context + style + emotion support gate
                    │
          8 explicit emotion queries
                    │
       query-conditioned bidirectional Mamba
                    │
      dynamic residual + static-style residual
                    │
       q0 anchor + q1..q8 emotion + q9 mixture
                    │
                10 × 25D reactions
```

### 2.1 P0：确定性 reaction anchor

Conditional-REGNN 的职责是提供 high-appropriateness、低方差的条件反应，而不是
承担第二条独立的图结构创新主线。它包含：

- 三路特征投影和融合；
- 四层 temporal Transformer encoder；
- 对 25 个 facial attributes 做 learned relation reasoning 的 relation block；
- paired listener target 上的 CCC、MSE 和 velocity supervision。

anchor 训练完成后冻结。`q0` 直接使用 anchor prediction，不经过 query gate 或残差，
因此候选多样性不能通过牺牲 q0 来换取。

### 2.2 Query residual generation

- `q1`–`q8` 是八个显式 emotion queries，对应八个 expression 方向；
- `q9` 是 context-supported mixture query；
- emotion embedding 维度为 64；
- 主体为两层 query-conditioned Bi-Mamba，`d_model=128`、`d_state=16`、
  `d_conv=4`、`expand=2`；
- 每个 query 同时产生逐帧 dynamic residual 和全局 static-style residual；
- residual 受 `tanh` 限幅，并通过 AU/VA/expression channel multiplier 控制推理幅度。

### 2.3 Target set 与 set alignment

训练 target set 的语义是：

- target index 0 是与 speaker 共享 basename、共享 crop 起点的 paired target，负责
  严格的逐帧质量；
- 其余 target 来自同 session listener pool，只表达 session 语义和一对多分布，
  不假设独立录制之间逐帧对齐；
- target 选择使用 deterministic epoch shuffle，保存 target path、length、duplicate
  和 availability 信息。

当前离线证据主线使用 balanced Sinkhorn，同时作用于 active style、sequence 和 FRD
transport。channel diversity budget 的主设置为 `0.15`，epoch 3 采用 isolated VA
calibration；该校准只更新 `to_residual` 的 VA 行和 bias，共 258 个 scalar。

### 2.4 Loss 组织

总目标按三组组织：

```python
loss = (
    loss_appropriateness
    + loss_semantic_allocation
    + loss_distribution_risk
    + residual_regularization
)
```

其中包含 paired-anchor quality、query identity、target-supported compatibility、
multi-target set alignment、channel diversity、FRD transport 和 isolated VA
preservation。旧主结果使用 diagonal FRD surrogate；新 target-supported 实现才默认
使用 banded Soft-DTW。训练 surrogate 不能称为 official exact FRD，正式 FRD 由
rolling engine 计算。

## 3. 离线证据主线

### 3.1 选定结果

主结果 checkpoint：
`regnn/runs/260814_eqr_channelot_jointisolatedva_bfit_e3_b16_seed1/emotion-query-mamba-epoch0003-seed1.pth`

协议为 1,142 个 full-test 样本、每个样本 10 个 predictions 和 10 个 official
targets、seed 1234、full variable-length sequence、rolling exact FRD。

| 指标 | 结果 | 方向 |
|---|---:|---|
| FRC | **0.853953** | 越高越好 |
| exact FRD | **151.620708** | 越低越好 |
| FRDiv | **0.152125** | 越高越好 |
| FRVar | **0.060050** | 越高越好 |
| semantic top-1 | **0.866462** | 越高越好 |
| distinct emotion modes | **6.931699** | 越高越好 |

这条证据路线的实际 checkpoint 配置仍是
`legacy_relative_softmax + gate_floor=1.0 + diagonal surrogate`，虽然 q9 使用
`predicted_support`。因此不能把它写成已验证的 target-supported 新方法结果。

### 3.2 已获得的因果证据

| 对照 | 主要观察 |
|---|---|
| 去掉 Stage-1 anchor pretraining | 输出严重 collapse：FRC 0.158635、FRDiv 0.008827、semantic 0.132443、modes 1.065674；低 FRD 不能解释为质量提升 |
| row-wise Softmax → hard Hungarian → balanced Sinkhorn | exact FRD 约由 154.25 降至 152.92 再降至 151.62；说明 balanced transport 改善分布级对齐，但不是所有指标都统一变好 |
| pre-alignment warm start 下 K=1 → K=10 targets | FRDiv 0.047798 → 0.049125、modes 7.281961 → 7.389667，exact FRD 改善 0.9638，但 FRC 和 FRVar 下降；多目标收益是 Pareto trade-off |
| dynamic-only / static-only residual | dynamic-only 保留较多多样性（FRDiv 0.114749），static-only 仅 0.043054；两者职责不同，不能删掉 dynamic residual |

上述结果都是单 seed local development evidence。它们支持模块作用和失败模式分析，
不支持无条件的“所有指标均提升”或统计显著性表述。

## 4. Online 路线

### 4.1 60→30 fixed-window geometry

官方 online 扩展采用：

- 每次模型调用最多看 60 帧 speaker context；
- 只保留窗口最后 30 帧作为 listener output/supervision；
- 多块 rollout 的输入跨度为 `60 + (N-1)×30`，仅拼接每块的最后 30 帧；
- listener targets 不进入 model forward；
- 当前实现允许模型看到完整的当前 30 帧 interval，因此这是
  **PerFRDiff-compatible chunk-online**，不是严格逐帧 causal。

### 4.2 Full-test online 结果

| 路线 | FRC | exact FRD | FRDiv | FRVar | semantic | modes | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| offline anchor + online EQR | 0.383438 | 146.615423 | 0.152870 | 0.041292 | 0.884851 | 7.078809 | 完成，未提升 |
| online-adapted anchor + online EQR | 0.312943 | 142.827395 | 0.152843 | 0.031818 | 0.896563 | 7.172504 | 完成，未提升 |

两条 online-window 路线保持了多样性和语义区分，但 FRC/FRVar 明显弱于离线主线；
FRD 较低也不能单独作为优越性证据。online 训练的初始化、监督尾部和推理协议还没有
形成与离线主线同等级的三 seed、B_cal/B_confirm 证据链。

### 4.3 严格历史因果诊断

固定输出 30 帧，把窗口末端设为当前 output boundary，只允许使用 boundary 之前的
speaker history，不读取未来 speaker 或 listener GT。`history=750` 的 full-test 结果为：

| history | past context | FRC | exact FRD | FRDiv | FRVar | semantic | modes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 750 | 720 + 当前输出区间 | 0.862451 | 159.485956 | 0.155495 | 0.068463 | 0.878722 | 7.029772 |

它相对离线主结果提高了 FRC、FRDiv 和 FRVar，但 exact FRD 变差；因此当前最安全的
结论是“长历史有助于相关性和动态覆盖，但引入分布距离代价”，而不是 online 已经
超过 offline。

## 5. 配置与证据边界

### 5.1 三个容易混淆的配置

| 配置/路径 | 语义 | 能否引用旧主结果 |
|---|---|---|
| `regnn/configs/emotion_query_mamba_final.json` | 新 target-supported implementation default；`target_support`、banded Soft-DTW | **不能**，只有实现、单测和 smoke 证据 |
| `260814_eqr_channelot_jointisolatedva_bfit_e3_b16_seed1` | 当前 offline full-test evidence main；legacy gate、diagonal surrogate | **可以**，仅限其完整协议和单 seed 范围 |
| `260903_mam_reactor_*` | online-window、causal-history 和训练探索 | **不能**替代 offline 主结果；必须按各自协议单独标注 |

### 5.2 数据与评估边界

- 当前 B_fit 为 2,146 个样本，full-test 为 1,142 个样本；
- 本地 `val/` 与 `test/` 内容审计为逐字节一致；
- 所有现有数字应称为 local development evidence，不得写成 hidden challenge test、
  leaderboard 或 SOTA；
- target-supported 新配置尚未完成 seeds `1,42,2026` 的 B_fit/B_cal/B_confirm；
- `paper/PAPER_CLAIM_AUDIT.md` 当前结论为 FAIL，原因包括数值错误、progressive
  ablation 混用不兼容协议、no-anchor 行的 VA 标记错误，以及 Huang/MReactor 原始
  对比证据缺失。

## 6. 归档后的建议路线

### P0：论文 offline 主线

继续以现有 full-test evidence main 为基准，先修正 claim audit 中的表格和协议标记。
论文中可以主张：anchor 防止 collapse、balanced transport 改善 distribution-level
alignment、multi-target supervision 带来有限但可观测的 coverage 增益，以及动态/静态
residual 的互补作用。

### P1：target-supported 方法闭环

如果论文要正式宣称 target-supported compatibility 或 banded Soft-DTW，必须从统一
配置重新跑 `B_fit → B_cal → B_confirm`，至少 seeds `1,42,2026`，并重新生成 official
metrics、exact FRD 和原始 JSON；旧 checkpoint 数值只能作为 legacy baseline。

### P2：online 研究分支

先明确投稿目标是“60→30 chunk-online”还是“strict causal history”。两者不能合并成
同一指标表。online 分支下一轮应：

1. 固定一个训练初始化策略和一个 rollout geometry；
2. 让 B_cal 负责选择 history、residual scale 和 checkpoint，B_confirm 只运行一次；
3. 保留 strict causal/no-future-input 的机器可审计字段；
4. 将当前位于 `/tmp/react2025-joint-va-main-ablation-20260817` 的 online 实现整理进
   受版本控制的项目路径，避免 archive 依赖临时 worktree。

## 7. Artifact 索引

### 方法与复现

- [原始 EQR-Mamba 技术路线](EMOTION_QUERY_MAMBA_TECHNICAL_ROUTE.md)
- [EQR-Mamba 可复现协议](EMOTION_QUERY_MAMBA_REPRODUCIBILITY.md)
- [历史实验旅程](EMOTION_QUERY_MAMBA_EXPERIMENT_JOURNEY.md)
- [Joint-Isolated-VA 与模块消融](EQR_JOINTVA_ABLATION_RESULTS_20260817.json)
- [Alignment strategy ablation](ALIGNMENT_STRATEGY_ABLATION_RESULTS.md)
- [Target-set size / pre-alignment warm start](TARGET_SET_SIZE_PREALIGN_WARM_RESULTS_20260902.md)
- [Anchor 与 diversity budget ablation](ANCHOR_AND_DIVERSITY_BUDGET_ABLATIONS_20260901.md)

### 主结果与 online/causal artifact

- 主 checkpoint：`regnn/runs/260814_eqr_channelot_jointisolatedva_bfit_e3_b16_seed1/`
- online offline-anchor checkpoint：`regnn/runs/260903_mam_reactor_online_frozen_offline_anchor_window60_tail30_joint_bfit_e10_b64_seed1/`
- online-adapted-anchor checkpoint：`regnn/runs/260903_mam_reactor_online_window60_tail30_joint_bfit_e10_b64_seed1/`
- causal history h=750：`regnn/runs/260903_mam_reactor_causal_history_h750_to30_full_test/`
- 训练成本：[TRAINING_COST_PROFILE_20260820.md](../../profile_output/TRAINING_COST_PROFILE_20260820.md)
- 推理成本：[REGNN_MAM_REACTOR_INFERENCE_PROFILE_20260818.md](../../profile_output/REGNN_MAM_REACTOR_INFERENCE_PROFILE_20260818.md)
- claim audit：[PAPER_CLAIM_AUDIT.md](../../paper/PAPER_CLAIM_AUDIT.md)

### SHA-256 快照

| artifact | SHA-256 |
|---|---|
| new target-supported config | `d8a511525de8f423606e2da736629b405e96331e173c962b385afff7fe5342ae` |
| offline evidence-main config | `565d39b2c68600a9378cf9bb8bec2a4589f3b6666b7532146846f1189764f864` |
| online offline-anchor config | `d22dc05bc83ec7d703bc6b0fb092625e7ecd69774486963d4af0e23b3f5d04a1` |
| online offline-anchor checkpoint | `b7705ce5ff22c80867d0e063b846604d9a8bc165115f28aa8f515a48c5d3b81a` |
| online-adapted-anchor config | `749a18e2239ef14bb7294a13fd06fd441fd7182aa7cd97f330d6c14fd7cd949b` |
| online-adapted-anchor checkpoint | `d66b5ef8564caceadbdb21fa9512fc7aabddace2ee1d2193209f3b0b247fbc2f` |
