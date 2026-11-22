# CoReSet 开源代码

本目录包含论文 REACT2025 主结果对应的完整训练与推理路径：

- Dataset：双方向特征加载、750 帧裁剪、补零与有效帧掩码。
- 教师训练：Farthest-sum 参考选择、Hungarian 分配、目标特定的多参考监督。
- 学生训练：冻结教师输出缓存、轨迹蒸馏、中心化差异蒸馏。
- 模型前向：共享编码器、专家条件、十条 AFR 并行生成。
- 一个主结果权重：完整 CoReSet **A0、seed 1**，3,803,201 个参数。

完整说明与各模块位置见 [README.md](README.md)。

## 安装与快速推理

在本目录中执行，推荐 Python 3.10：

```bash
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e '.[test]'
python -m coreset_reactor.inference --demo --device cpu --output demo.npz
```

`--demo` 使用合成特征检查环境。真实输入需要预提取的音频特征 `[T,768]`、
说话者面部属性 `[T,25]` 和 3DMM 系数 `[T,58]`。示例：

```bash
python -m coreset_reactor.inference \
  --audio /path/to/audio.npy \
  --facial-attributes /path/to/facial_attributes.npy \
  --coefficients /path/to/coefficients.npy \
  --device cuda:0 --output reaction_set.npz
```

输出中 `prediction` 为连续的 `[10,T,25]` AFR 集合，适合渲染；
`paper_prediction` 为 AU 取整并应用论文固定后处理的评测输出。
权重和长序列推理协议见 [MODEL_CARD.md](MODEL_CARD.md)。

## 训练

按照 [DATASET.md](DATASET.md) 准备数据，然后执行：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_react2025.sh /path/to/react2025 runs/react2025_seed1
```

脚本依次训练十个教师、生成多视图缓存、训练完整学生并导出权重。
原始主结果的 3,320 组参考分配已固定保存在 `configs/` 中。
小规模训练流程检查可用 `python scripts/smoke_test.py`，无需下载数据。

发布权重的每个参数张量均与原论文 checkpoint 完全一致；导出时移除了优化器
及本地路径等元信息。论文记录的主结果为 FRC **1.1014**、FRD **134.2255**、
FRDiv **0.1790**、FRVar **0.0571**、TLCC **46.04**。对应完整 VAL1142、
双方向、完整序列和全部同 session 对侧参考的评测协议。
