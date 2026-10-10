> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 学生结构与蒸馏组合消融执行协议

2026年9月30日，用户授权同时执行结构消融与损失组合实验，暂不增加训练随机种子。
只新增两组完整训练，原项目与历史实验只读。所有结果包括负结果、smoke与失败记录均保留。

## 比较范围

新增 `kd_assignment_dynamic`：基础KD的整段q0固定9×9 Hungarian匹配，加动态残差项。
基础项和动态项使用同一个重排后的教师目标。匹配代价仍仅为基础KD，不加入动态代价；
q0只是固定对应，不冻结参数。动态系数固定沿用1.0，AU、VA、expression尺度沿用旧TRAIN校准。
不另做系数搜索、梯度重标定、逐帧匹配或完整残差叠加。旧动态组初始辅助梯度占比
2.7%至4.8%不能当作组合组的测量值，也不声称所有组辅助梯度相同。

新增 `kd_multihead`：相同共享编码器与关系图模块，无专家ID与context偏移，
将一个共享256到25线性头替换成10个独立256到25线性头，再分别经过相同共享关系图。
十个头均从共同配对预训练线性头复制，保留预训练编码器和关系图全部权重；
使用固定槽位基础KD，与旧 `kd_basic` 比较，不同时更换损失或教师。
模型构造先走原expert初始化再转换，不额外消耗训练RNG。不同结构的完整初始函数不相同，
不能称为完全相同初始化；只核对共同预训练权重、数据日程和训练预算。
两结构参数量接近但不强行填充无用参数；记录实际参数量与占比。图模块仍共享且存在，
本轮不是图消融或模型剪枝。十头起始相同，但独立参数由不同教师槽位监督分化。

复用 `kd_basic`、`kd_assignment`、`kd_dynamic` 三组完整结果前，必须用本轮训练路径
分别精确重放头两步的日志、模型、优化器、scheduler、CPU/CUDA/NumPy/Python/loader RNG。
任一不一致停止，不静默复用或自动扩展实验。保留旧residual/typed结果但不作为本轮主对照。

## 共同训练条件

共同mode教师TRAIN缓存，3320个固定source/crop/view，K为10，seed为1，batch为4。
各8300更新、10 epoch，AdamW学习率1e-4、betas为0.9和0.99、weight decay为1e-4，
cosine降至5e-6，BF16、梯度裁剪1、DataLoader seed2709271，4个worker。
不从已训练学生续训，不使用验证集选checkpoint或调超参数，全部取8300步最终模型。
GPU只用1与5，6和其他作业不动。启动前核对空闲显存；不终止其他作业。

准备时审核完整教师分片SHA、全部3320输入view及教师输出语义，共同权重与初始RNG；
冻结本轮及历史源码哈希、历史对照权重和逐样本评测哈希。代码、协议按内容寻址保存副本。
先单测、三组精确重放、两组两步GPU smoke、三条VAL端到端全参考精确评分，再正式训练。
每epoch原子保存模型、优化器、scheduler及全部RNG；每100步记录进度；失败停止，拒绝覆盖。
独立systemd服务最长6小时，单作业最长2小时，不自动重启或悄悄从部分步恢复。

## 评测与预定比较

全部VAL571，保持论文all-session参考集合、center750、25D Processor与AU取整、
每输入十预测的全参考精确FRD。十预测不是十GT。不改动旧评分函数。
合并前核对clip顺序、session、GT数量与缓存哈希、评分源码、训练数据日程。
结构改变需要新的模型加载器；验证expert路径与旧模型加载器的输出逐元素一致。

完整报告FRC、FRD、FRDiv、FRVar和softFRDiv，附连续AU/VA/expression的总、静态、
动态与velocity能量诊断。能量不等于有效模式覆盖。本轮仍是单训练种子VAL，不是独立TEST。
保留20个session聚类5000次bootstrap，不涉及新增训练种子，区间不包含训练随机性，未校正多重检验。

预定：十头减基础、组合减基础、组合减仅匹配、组合减仅动态；四格交互项为
`组合 - 仅匹配 - 仅动态 + 基础`。所有指标与方向全部报告，不能用交互项单独证明Pareto改进。

在GPU5上对已训练基础、十头、组合模型用相同batch1、十输出、750帧、FP32推理，
每模型5次warmup、30次同步计时，报告中位数、四分位数、峰值allocated显存及GPU信息。
计时不含读盘/评分，参数压缩不等于加速；这是合成固定长度输入的微基准，非真实端到端吞吐。
性能剖析报告PyTorch可统计的FLOPs及算子覆盖限制，不将缺失算子当作0真实成本。
