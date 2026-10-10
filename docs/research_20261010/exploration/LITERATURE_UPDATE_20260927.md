> Sanitized historical record. Dates/status are those of the source; see the publication README for current scope and completed follow-ups. Local path variables are placeholders, not downloadable assets.

# 文献与选题边界补查（2026-09-27）

定位：围绕完整有效参考集合的质量—多样性学习，不再以“all-session非官方”
或“同session置乱说明参考错误”为动机。以下仅是读到的证据，不是穷尽性新颖性检索。

## 直接相关工作

| 工作 | 已核验贡献/范围 | 对当前选题的约束 |
|---|---|---|
| [PerFRDiff，ACM MM2024](https://github.com/xk0720/PerFRDiff) | 从听者历史行为学习个性化权重偏移，编辑通用反应生成模型 | 不能把个人风格或权重适配本身当作新贡献 |
| [ReactDiff，Cheng Luo等，ACM MM2025](https://arxiv.org/html/2510.04712v1) | 扩散生成加入时序运动学与AU依赖；附录C.6已有模型蒸馏和效率—质量比较 | 平滑、AU约束、蒸馏不是空白；新方法需要明确监督机制证据 |
| [ReactDiff，Jiaming Li等，Neural Networks 2025](https://arxiv.org/html/2505.14151v1) | 模态内/模态间注意与潜空间条件扩散 | 与上一行不同论文、不同作者和代码；不能混用结果 |
| [REACT2026](https://arxiv.org/abs/2606.07935) | MARS上扩展个性化、Big-Five和EEG信号 | 个性化或额外模态已成挑战方向，不能仅靠增加输入建立新颖性 |
| MReactor，CVPR2026 Findings | 已核验标题、作者及Findings类别；本次PDF取回失败，尚未核验方法公式 | 保留为必须精读的近邻；不能仅凭“层次认知解耦”标题宣称它解决/未解决模式保持 |

MReactor类别证据来自[作者主页](https://pquochuy.github.io/publications/)及
[作者实验室](https://machine-listening.eecs.qmul.ac.uk/publications/)；其PDF链接位于
CVF的CVPR2026F目录。部分索引省略Findings，不能据此写成CVPR主会。

ReactDiff两篇的出版区分：Cheng Luo等的论文页列出ACM MM2025 DOI；Jiaming Li等的
[出版商页面](https://www.sciencedirect.com/science/article/pii/S0893608025004769)列为Neural Networks。
本轮没有复制论文表格或将不同数据/协议的数字与本项目VAL直接排名。

## 主会论文呈现了什么证据

[DiffListener，CVPR2025 Highlight](https://nuo1wang.github.io/DiffListener/)将显式运动生成与
隐式运动细化结合，并处理视频渲染质量。它的任务输出和视觉证据比25D属性评分更丰富；
这不意味着我们必须更换整套渲染骨干，但若冲击主会，不能只展示一个CCC损失消融表。
此处已读项目页与论文摘要，未完成全文消融复核。

[Timeline Control，CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_Exploring_Timeline_Control_for_Facial_Motion_Generation_CVPR_2025_paper.html)
围绕明确的时间控制接口，同时设计动作区间标注与条件生成。可借鉴的是问题定义、
方法与验证一一对应；不能把它的受控面部运动生成直接称为REACT基线。

## all-session的补充依据

2025主依据仍是其论文第3、4节。另核验了
[2026官方仓库](https://github.com/reactmultimodalchallenge/baseline_react2026)：数据和评估说明
也明确以同session不同参与者的全部反应为GT。这仅是连续性的旁证，不是把2026实现
倒推成2025所有计算细节都一致，更不意味着可以把VAL重新命名为独立TEST。

## 我们的待证假设与决策分支

以下是研究推断，不是上述文献或当前实验已经证明的结论：

- 学习完整有效集合时，统一质量聚合可能吸引多个输出到少数高分模式。
  当前fixed/uniform/mode/shuffled对照用于区分“集合监督有效”与“模式对应真的必要”。
- 若mode同时优于uniform与shuffled，且质量—多样性没有明显退化，下一步应验证
  第二骨干、多训练seed和不使用同类描述子的独立行为覆盖，而不是立即堆更大模型。
- 若mode与shuffled相近，说明语义/行为模式匹配的解释不足；不能继续包装成模式认知。
- 若只提升FRC且显著压低多样性，报告为工作点移动，不算新方法全面胜出。
- 若改进主要来自正确的CCC梯度，贡献应归因于实现修正，不能算模式监督创新。
- 即使pilot成功，CVPR2027仍需严格可比的强外部基线、独立测试、合理的人类多答案
  评价及视觉/时序验证。蒸馏可作部署扩展，不能替代核心研究问题。

可能的工作题名：Mode-Preserving Learning from Sets of Appropriate Reactions。
当前只是可证伪研究问题；不声称首创、不声称达到投稿完成状态。
