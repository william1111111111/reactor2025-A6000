> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 不同蒸馏方式：冻结执行协议 v1

2026-09-28，用户授权执行三种新蒸馏。独立目录 runs/distillation_methods_v1，旧消融只读。
不是新剪枝/新架构/扩散蒸馏，不依据验证集选参数或提前停止。

## 分组与共同条件

新组 kd_typed / kd_assignment / kd_dynamic；历史对照 kd_basic / kd_residual。
共同 mode teacher TRAIN cache、3320 固定 views、配对预训练初始化、seed1、
3,803,201 参数、十候选、batch4、8300 更新/10 epoch、AdamW1e-4、(.9,.99)、wd1e-4、
cosine 到5e-6、BF16、clip1、独立 DataLoader seed2709271。不是纯随机从零训练。

- typed：原基础KD + lambda*(平均15维Bernoulli KL + 8类categorical KL)/2；
  温度1，概率clamp1e-5，expression归一化，VA仍由基础回归监督，不对25D整体softmax。
- assignment：q0固定对应；其余9×9整段Hungarian一对一；每个输入独立，不逐帧匹配。
  assignment代价在FP32/no_grad中计算，与基础MSE+(1−CCC)+.05velocity同一定义。
  MSE/velocity保留全batch有效帧分母，CCC保留等候选权重；选择后用原BF16路径
  的基础criterion反传。只有匹配的离散选择使用FP32，不承诺BF16代价逐位相同。
- dynamic：原基础KD + lambda*动态残差MSE；先减同帧槽位均值，再减有效时间均值，
  排除静态槽位偏置；AU/VA/expression仍用旧TRAIN尺度，不额外叠加原residual项。

typed/dynamic使用与旧residual相同的32 TRAIN抽样索引、前16条梯度探针(batch4)、
初始模型train模式和固定dropout种子。lambda=min(1, .25*median(base_grad/aux_grad))。
用独立进程，不消耗正式训练RNG；记录各probe实际梯度比例，不声称所有步都保持.25。
旧residual保持原冻结0.3291488264202468，不重新调优。assignment没有辅助系数。

## 复用对照的硬性条件

新训练代码分别对basic/residual执行两步，逐项重放恢复run的头两步loss/grad日志；
对照旧student_method_v1/smoke的实际两步模型state、优化器state和torch/CUDA RNG，
核对相同初始化、source/view摘要、DataLoader RNG。任何不一致停止，不默默复用。
新三组完整训练还必须与旧两组8300步source/view摘要完全一致。
若复用失败，先报告原因，在新目录制定重跑控制的修订，不直接跳过核验。

## 执行与可追溯性

prepare核验已有PREPARED及全部cache shard SHA、所有3320输入view一致、
teacher有限值/输出语义与长度、共同初始模型。冻结新代码/协议和旧源码SHA。
TRAIN校准→基线精确重放→三个新组各两步smoke→三样本全链路评测→正式训练。
每epoch原子checkpoint，记录完整optimizer/scheduler/CPU CUDA NumPy Python loader RNG。
两个GPU工作槽只用1/5；6及其他任务不动。各作业2小时上限，独立服务6小时总上限，
失败停止、无自动重试、拒绝覆盖输出。训练/缓存成本另计，不把参数比例称为速度比。

## 评测与预定对照

新三组全部VAL571，复用认证all-session GT缓存、center750、原25D Processor/AU取整，
每条十预测全参考精确FRD；不换成sampled10参考。FRC、FRD、FRDiv、FRVar、softFRDiv全报。
通过逐条clip ID、session、GT数量、GT缓存哈希和评分源码SHA一致性后合并旧baseline结果。
旧direct两组只作背景，不作为本轮新增蒸馏主对照。

预定：typed−basic、assignment−basic、dynamic−basic、dynamic−residual、residual−basic。
按20 session聚类bootstrap5000、seed270927，完整区间，单seed VAL，未校正多重检验。
附加每context连续槽位残差的总/静态/动态能量和有效帧velocity能量（均按三组通道），
均为辅助诊断，不声称官方覆盖指标；训练记录assignment变更比例和9×9使用计数。
不根据FRDiv alone宣称有效模式覆盖、创新成立、独立TEST或CVPR投稿完成。
