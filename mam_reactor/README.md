# Mam-Reactor 独立归档与运行目录

> 此 GitHub 副本只包含源码和配置；下文描述的是原始本地归档。权重、数据集、缓存和完整训练结果没有上传，原始机器的绝对路径需按实际环境调整。

这个目录是 Mam-Reactor 的可运行副本：代码、启动脚本、配置、代表性权重和已完成结果都放在这里。训练数据不复制，运行时通过 `REACT_DATA_DIR` 指向外部 REACT2025 数据目录。

## 直接启动

先确认使用的是包含 `torch`、`mamba-ssm`、`torchaudio` 和 `decord` 的环境；当前仓库默认使用 `../.venv/bin/python`。依赖清单是 [`requirements.txt`](requirements.txt) 和 [`requirements_regnn.txt`](requirements_regnn.txt)。

```bash
cd /home/zhengshiyi/react2025/mam_reactor
REACT_DATA_DIR=/home/zhengshiyi/react/data ./verify.sh
REACT_DATA_DIR=/home/zhengshiyi/react/data MAM_REACTOR_GPU_ID=6 ./train.sh
```

`train.sh` 默认复现当前有完整指标支撑的 offline evidence 路线：冻结 Conditional-REGNN anchor，使用 EQR warm-start，在 `B_fit` 上训练 3 个 epoch。每次默认写入新的 `runs/offline_时间戳/`，不会覆盖已有结果。

推理默认使用 offline 权重：

```bash
REACT_DATA_DIR=/home/zhengshiyi/react/data ./infer.sh
```

切换到 60→30 online-window 路线：

```bash
REACT_DATA_DIR=/home/zhengshiyi/react/data \
MAM_REACTOR_VARIANT=online_frozen_offline_anchor ./infer.sh
```

可用的 `MAM_REACTOR_VARIANT` 是：

- `offline_evidence`
- `online_frozen_offline_anchor`
- `online_adapted_anchor`

推理会生成 `metrics.json` 和 `results_frd.pt`。默认优先复用原仓库的官方 target cache；也可以显式设置 `REACT_TARGET_CACHE_RESULTS_PT`。如果没有 target cache，则需要通过 `MAM_REACTOR_ASSET_PROJECT_DIR` 指向包含 `pretrained_models/post_processor/checkpoint.pth` 的完整评估工程。

精确 FRD 单独执行，支持中断后续跑：

```bash
./run_exact_frd.sh runs/inference_offline_时间戳/results_frd.pt \
  runs/inference_offline_时间戳/frd_resumable
```

## 目录说明

```text
mam_reactor/
├── train.sh / infer.sh / run_exact_frd.sh / verify.sh
├── launch/                         # 训练参数入口
├── regnn/                          # Mam-Reactor 与评估代码
├── framework/ dataset/             # 评估所需代码，不含数据
├── configs/                        # 官方评估配置
├── regnn/configs/                  # Mam-Reactor 配置与 session split
├── checkpoints/                    # anchor、warm-start、offline/online 权重
├── assets/                         # 训练所需 style descriptor cache
├── results/                        # 已完成训练、推理和精确 FRD 结果
└── tools/                          # 精确 FRD 计算程序
```

权重和结果的逐文件 SHA-256 在 `MANIFEST.sha256`。原始 `regnn/runs/` 保持不变；大体积的原始 `results_frd.pt` 和训练/测试数据不重复放入本目录，已有指标、FRD 结果 JSON、日志和来源路径均已保存。

结果速览见 [`results/RESULTS_SUMMARY.md`](results/RESULTS_SUMMARY.md)，来源边界见 [`PROVENANCE.md`](PROVENANCE.md)。

## 重要边界

默认训练启动脚本对应的是已经产生完整 offline 指标的 legacy-compatible evidence 配置。`regnn/configs/emotion_query_mamba_final.json` 保留了 target-supported 新配置，但该配置目前只有实现/试验边界，没有可转移的完整三 seed 官方结果。online 和 strict-causal 结果也按独立 protocol 保存，不能和 offline 指标直接混报。
