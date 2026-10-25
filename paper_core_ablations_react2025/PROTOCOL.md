# CoReSet 核心消融预注册协议（REACT2025）

本目录只新增 REACT2025 实验。REACT2024 的既有主结果、checkpoint 和评估记录保持冻结，不在本轮重训、重评或补种子。

## 第一批问题

第一批只比较以下两组：

- **A0 Full**：基础轨迹 KD、velocity loss 与中心化差异损失。
- **A1 Trajectory KD**：与 A0 使用相同教师 AFR、学生架构、初始化、训练视图、样本顺序、优化器、更新预算和推理配置，只令中心化差异损失权重为零。

中心化差异损失在每个样本、每个有效时间位置沿 10 个 teacher/student slot 中心化。AU15、VA2、EX8 三组等权：

```text
R_S[g] = S[g] - mean_over_slots(S[g])
R_T[g] = T[g] - mean_over_slots(T[g])
L_ctr = mean_over_groups(mean_valid((R_S[g] - R_T[g])^2) / scale[g]^2)
```

教师输出固定且不接收梯度；slot 按 teacher ID 对应，不做排列匹配。A0 的权重只用固定 TRAIN 缓存的初始梯度按目标比例 0.25 校准，不读取 VAL。校准在 forked RNG 中执行，训练样本顺序应与 A1 一致。

## Seed 1 固定训练条件

- 教师缓存：确定性的 head/center/tail 多时间视图；长序列 3 个视图，短序列 1 个视图，共 8,820 个 context。
- 教师数：10。
- 学生：ExpertConditionalREGNN，3,803,201 参数。
- 初始化：同一个 rank-00 T1 backbone checkpoint。
- seed：1。
- batch size：4。
- optimizer：AdamW，学习率 `1e-4`，weight decay `1e-4`，betas `(0.9, 0.99)`。
- scheduler：4 epoch cosine，终点为初始学习率的 0.05。
- 训练预算：4 epoch、8,820 updates。
- 基础损失：MSE + CCC + `0.05 × velocity`。

## 正式评估

- 数据集：REACT2025。
- 完整双方向 VAL1142：speaker→listener 571 条，listener→speaker 571 条，context 等权。
- 每条使用完整源序列；750 帧窗口、375 步长、归一化中心权重 `p=4`，拼接整段后评分。
- 每条使用同 session 全部对向角色 GT；固定官方 Processor、GT 顺序、seed 和 AU rounding。
- 指标：FRC、exact FRD、FRDiv、FRVar、官方 TLCC/FRSyn。
- 每组分别记录原始 p=4 输出和固定 query-0 常量 AU 单转换后处理结果。不得按组搜索后处理。

单方向、中心 750 帧、子集和近似 FRD 只允许做接口冒烟，不进入正式结果或模型选择。

## 差异迁移诊断

A0/A1 在完整 8,820-context 固定 TRAIN 教师缓存上诊断。教师和两个学生使用相同输入视图、teacher ID、有效帧与 slot 顺序。报告：

1. AU/VA/EX 分组中心化残差误差及固定教师能量归一化值；单独统计近零教师能量。
2. AU/VA/EX 分组教师—学生成对 RMS 距离误差。
3. 未取整软输出 FRDiv 及相对教师的保留比例；近零教师 FRDiv 不形成比值。
4. 按固定哈希规则预选的 6 个长视图示例，保存 Teacher/A1/A0 的 10×10 分组距离矩阵和完整 AU/VA 软轨迹。

诊断只说明教师集合内部差异的迁移程度；最终价值仍由针对真实 GT 的 VAL1142 FRC/FRD/FRDiv 等指标判断。

## 后续顺序

Seed 1 的 A0/A1 完整评估和诊断完成后，才依次进行 A2、A3、A0/A1 seed 2/3，以及教师 reference 对照。本协议不依据第一批结果更换初始化、训练预算、推理参数或评估方法。

## 第二批预定配置（在 A0/A1 完成后冻结）

- **A3 Group-weighted KD**：使用与 A0 完全相同的教师缓存、AU/VA/EX 尺度 `[0.3264721930, 0.3286206722, 0.2594182789]` 和系数 `0.2908728908`。辅助项为 `mean_groups(mean_valid((S[g]-T[g])²) / scale[g]²)`；唯一公式变化是不沿 slot 减均值。不重新梯度校准或读取 VAL。
- **A2 Direct reference + velocity**：在 A1 的训练配方中只把 10 个教师输出目标替换为相同 teacher ID 原本指定的 10 个固定真实参考；rank 0 使用同步原生裁剪，rank 1–9 使用教师训练时相同的 `relative_time_masked` 对齐。保留 `velocity_weight=0.05`，作为严格单变量目标来源对照。
- **A2/A1 no-velocity 补充对照**：真实参考并非逐帧同步时，velocity 的适用性不明确。因此另外训练一个 `velocity_weight=0` 的教师目标 KD 和一个 `velocity_weight=0` 的 Direct reference；两侧同时删除 velocity，只比较目标来源。

三组均保持 seed 1、完整初始 state、8,820 个多时间视图及顺序、4 epoch、8,820 updates、batch 4、AdamW 与最终 p=4 双方向 VAL1142 评估。不根据 A0/A1 结果修改权重或选择配置。

## A0/A1 重复种子预定配置

第二批 seed 1 控制完成后，A0/A1 使用预定 seed 2、3 各重训一次。每个 seed 内两组固定相同教师缓存、完整初始 state、样本顺序、velocity、优化器和 8,820 updates；A0 仍以 TRAIN 初始梯度比例 0.25 校准中心化差异项，A1 关系项权重为零。两个 seed 不因中途结果取消或替换。

每个 checkpoint 均做相同的完整双方向 VAL1142 原始及固定后处理评估。每个训练 seed 内单独做 4,000 次 session-cluster bootstrap；三个训练 seed 的均值和样本标准差另行报告，二者不混用。具体冻结值见 `configs/A0_A1_seed_repeats_frozen.json`。

## Reference 教师及对应学生预定配置

三个教师库均采用相同的 T1 固定真实目标成员、teacher ID、初始 state、seed 1、输入顺序、batch 32、学习率、velocity 和每位教师 600 updates；query 0 是三组共享的真实配对教师。rank 1–9 唯一计划内变化是教师 CCC 参考规则：

- **Fixed**：只用指定真实参考计算 MSE 与 CCC。
- **Uniform**：MSE 仍固定为同一指定参考，CCC 对同 session 全参考采用均匀先验的 log-sum-exp。
- **Target-specific**：MSE 仍固定为同一指定参考，CCC 对同 session 全参考采用围绕指定目标的预定先验。

教师三组在同一次完整双方向 VAL1142 流程中评分。对应学生使用相同 seed-1 架构、初始化、8,820 个有序多时间视图、4 epoch、8,820 updates 和 A0 中心化差异目标。为使“学生损失相同”是数值层面的严格控制，三组共同冻结 A0 的 AU/VA/EX 尺度 `[0.3264721930, 0.3286206722, 0.2594182789]` 及系数 `0.2908728908`，不按教师库重新校准。Target-specific 学生复用已匹配的 A0 seed 1；Fixed 与 Uniform 学生按同一配置新训。
