> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 现有方法投稿前对照：共享学生直接训练与蒸馏

2026-09-27，用户授权执行现有方法验证，不新增剪枝或架构。
本轮是第一项决定性对照，不代表已完成多seed、外部基线或独立TEST。

## 固定与分组

共同3,803,201参数ExpertConditionalREGNN，十个输出。所有组采用同一个历史配对q0
初始化骨干、seed1初始化槽位模块；不是随机从零训练。初始化成本是共同前置成本。
共同3320 TRAIN固定source/crop views（与原模式教师缓存逐条核验），batch4，10epoch，
8300优化步，AdamW1e-4，betas(.9,.99)，wd1e-4，epoch cosine至5%，BF16，clip norm1。
独立于模型RNG的相同DataLoader generator固定顺序；因此不声称复现旧学生逐位权重。

- direct_fixed：共享学生直接学习T1的配对q0和九个固定Max-FRDiv参考。
- direct_mode：相同raw目标，q1..9使用现有mode reference CCC；q0使用普通配对CCC。
- kd_basic：共享学生学习冻结模式教师的十槽位soft预测；原MSE+CCC+.05 velocity。
- kd_residual：kd_basic加现有centered-residual关系损失；从相同缓存、相同初始化的
  历史TRAIN校准复用尺度及权重，核对cache/init哈希，所有组不重新调参。

直接监督的非配对参考按现有实现相对时间重采样至当前共同有效长度，禁用其速度监督；
q0保留真实配对时间。蒸馏目标在同一source时间轴生成，全部槽位可使用速度项。
这种差别属于raw监督与teacher监督的定义，不伪称四组目标/计算量相同。
direct_mode汇聚和九个教师的mode规则一致：TRAIN mean/std50D，effective count8，tau.05。
所有参考选择/先验只读TRAIN。直接组不会把teacher_predictions送入loss；加载它们仅为
共用view审计与统一数据路径，输出不参与直接组优化。推理不输入GT或session标签。

## 预算和执行

四组各8300更新，总33200更新；GPU1/5各一个作业，6不占用。先四组各2步smoke，
核验实际优化器步数、有限梯度、相同初始化/顺序、输出K10及损失单测。失败停止队列。
训练、推理、全量精确评分、统计顺序执行，全部产物写新runs/student_method_v1。
每作业记录实际时间、峰值显存、source schedule哈希、完整优化器checkpoint与源码SHA。
不覆盖历史文件、不依据VAL挑训练步数、不自动搜索超参数。后续seed/独立测试另列待办。

## 评测和决策

完整VAL571，论文all-session参考集合；复用已认证同一GT缓存及25D评分实现。
FRC、精确FRD、FRDiv、FRVar、soft FRDiv，逐样本逐候选保存。
同第二轮的session聚类bootstrap5000次；只代表该seed的VAL，不代表独立TEST。
四个预定对照：direct_mode−direct_fixed、kd_basic−direct_mode、
kd_residual−kd_basic、kd_residual−direct_mode。完整报告，不以某一个指标判胜。

教师构建/缓存生成成本额外存在；即使学生更新数相同也不能宣称总训练算力相同。
9.78倍参数压缩的参照是十教师，不是同大小直接学生；报告必须分开。
若直接学生与蒸馏方案相近，蒸馏必要性不成立；若残差仅改善FRD而牺牲多样性，
只能报告取舍，不能写“关系损失提升多样性”。不支持的模块不作为论文核心贡献。
2024 native协议资产尚未认证，本轮不把旧同录制池分数当正式跨数据集证据。
