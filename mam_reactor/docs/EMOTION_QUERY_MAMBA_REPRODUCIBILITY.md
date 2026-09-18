# EQR-Mamba 可复现协议

## 1. 适用范围

本协议对应 **Anchor-Preserving Emotion-Query Residual Mamba** 新实现。最终配置文件
`regnn/configs/emotion_query_mamba_final.json` 的 SHA-256 为：

```text
d8a511525de8f423606e2da736629b405e96331e173c962b385afff7fe5342ae
```

该配置当前状态是 implementation default：已完成单元测试与 GPU smoke，但尚未完成
三 seed 长训练和官方全量评估。配置中的 `candidate_checkpoint` 与
`candidate_metrics` 因此保持 `null`。

## 2. 环境

已验证基础环境为 Python 3.10、PyTorch `2.1.0+cu121`、NumPy `1.26.4`、
`mamba-ssm 1.2.0.post1`。先安装官方 baseline 依赖，再安装
`requirements_regnn.txt`。Mamba CUDA 扩展必须与 PyTorch/CUDA ABI 一致。

训练脚本是单进程单卡；设置 `CUDA_VISIBLE_DEVICES` 时最多暴露两张 GPU，本协议的
smoke 和推荐实验均只使用一张。

## 3. 数据、split 与泄露边界

`--data-dir` 指向包含 `train/val/test` 的 REACT2025 数据目录：

```text
audio-features/{speaker,listener}/session*/...npy
facial-attributes/{speaker,listener}/session*/...npy
coefficients/{speaker,listener}/session*/...npy
```

本地 `val/` 和 `test/` 内容逐字节一致。现有结果只能称为 local development
evidence，不得表述为隐藏测试、challenge leaderboard 或 SOTA。

正式实验必须从 train 按 session 或 person-session 构造三个互斥子集：

- `B_fit`：训练参数；
- `B_cal`：选择 checkpoint、gate threshold、Soft-DTW 参数和 residual multiplier；
- `B_confirm`：冻结全部选择后只运行一次最终确认。

禁止用 `B_confirm` 反复调参。至少运行 seeds `1, 42, 2026`，报告均值、标准差和每个
seed 的原始 JSON。

## 4. target-set 可复现性

训练默认 `target_selection_mode=deterministic_epoch_shuffle`。每个 epoch 开始调用
`dataset.set_epoch(epoch)`；选择只依赖 global seed、epoch 和 source path hash，和
DataLoader worker 调度无关；source 与 non-paired target crop 也使用独立的稳定 hash。
validation/test 选择固定。

每个 batch 返回 `target_paths`、`target_lengths`、`target_is_duplicate` 和
`target_available_mask`。发布 run 必须保存每轮：

- `target_pool_unique_pairs`；
- `target_pool_pair_count`；
- `target_pool_coverage`；
- `target_available_fraction`；
- `target_duplicate_fraction`。

## 5. 外部 artifacts 与 SHA-256

历史 warm-start 链保留用于核对，但不把旧结果转移到新模型：

| artifact | SHA-256 | 角色 |
| --- | --- | --- |
| Conditional-REGNN epoch 50 | `9d0190a7cb48703e001dc3f092d84ac7de17bea5e2412604f3871099d1c3ebb6` | frozen anchor |
| style cache | `ac996b032321a6d824c74143d1353ecea1b6e887bd94736f693925817c46755c` | train-only descriptors |
| legacy query-wise-CVaR epoch 3 | `efae7a1d83fd63bfdc2e42198b1a539be91c78a0a87f8255f595af5db5631bd2` | initialization only |

旧 checkpoint 来自 `gate_floor=1.0`、legacy compatibility/q9 语义和 diagonal FRD
surrogate。新训练会重新学习 target-supported compatibility。每次训练后的
`config.json`、checkpoint 和 prediction/target cache 都必须重新计算 SHA-256：

```bash
sha256sum "$REACT_FINAL_CHECKPOINT" \
  regnn/configs/emotion_query_mamba_final.json \
  "$REACT_STYLE_CACHE" \
  "$REACT_EVAL_DIR/results_frd.pt"
```

## 6. 训练与 smoke

设置环境变量后运行默认配置：

```bash
export CUDA_VISIBLE_DEVICES=0
export REACT_PYTHON=/absolute/path/to/react2025/.venv/bin/python
export REACT_DATA_DIR=/absolute/path/to/data
export REACT_ANCHOR_CHECKPOINT=/absolute/path/to/anchor.pth
export REACT_WARMSTART_CHECKPOINT=/absolute/path/to/legacy_or_new_eqr.pth
export REACT_STYLE_CACHE=/absolute/path/to/train_reaction_style_v1.pt
export REACT_RUN_DIR=/absolute/path/to/a_new_run_directory

launch/train_emotion_query_mamba_final_stage.sh
```

每次必须使用新 `REACT_RUN_DIR`；训练器检测到已有 `train_metrics.jsonl` 或
`TRAINING_COMPLETE` 会拒绝覆盖。快速 smoke 使用完全相同的机制，只跑一个 batch：

```bash
export REACT_RUN_DIR=/absolute/path/to/a_new_smoke_directory
launch/smoke_train_eqr_mamba.sh
```

## 7. 消融

全部 case 定义在 `regnn/configs/eqr_mamba_ablation_matrix.json`。例如：

```bash
export EQR_ABLATION=top3_compatibility_gate
export REACT_RUN_DIR=/absolute/path/to/a_new_top3_run
launch/train_eqr_mamba_ablation.sh
```

可用 case 覆盖 gate floor、target-supported gate、top-3、uniform/predicted q9、
fixed/epoch-shuffled target、diagonal/Soft-DTW、uni/bi-Mamba、generic/explicit queries。
每次只改变矩阵列出的因素。

每个实验至少填写：FRC、exact FRD、FRDiv、FRVar、semantic query accuracy、mean
active query count、target match coverage、每 query FRD excess、inference time、
trainable parameter count 和 peak GPU memory。未运行的字段保持 `null`，不得估计。

## 8. checkpoint compatibility

`EmotionQueryMambaConfig.from_checkpoint_payload()` 对缺少新字段的旧 payload 自动使用：

```text
compatibility_mode = legacy_relative_softmax
mixture_mode = predicted_support
gate_warmup_steps = 0
gate_top_m = 0
```

本轮没有增加 trainable tensor，因此旧 state dict 可 strict-load 到按旧 payload 构造的
模型，旧 q0 与旧生成行为保持可复算。旧 checkpoint 作为新配置 warm-start 也可加载，
但 gate、target sampling、loss 和 FRD surrogate 的语义已经变化，不能把它称为无差别
resume。

新 checkpoint 含 `compatibility_mode`、`mixture_mode` 与 gate/Soft-DTW training config；
未更新的旧代码不能直接解析这些新增配置字段。数据选择改变也会使同一个 epoch 的训练
batch target set 不再等同于旧实现。

## 9. 官方非 FRD 指标

不要覆盖 checkpoint 中的 gate floor：

```bash
python -m regnn.eval_emotion_query_mamba_official_test \
  --checkpoint "$REACT_FINAL_CHECKPOINT" \
  --data-dir "$REACT_DATA_DIR" \
  --metrics-json "$REACT_EVAL_DIR/metrics.json" \
  --residual-scale 0.95 \
  --style-residual-scale 0.95 \
  --au-residual-multiplier 1.8 \
  --va-residual-multiplier 0.5 \
  --expression-residual-multiplier 1.6 \
  --selection-seed 1234
```

该命令记录 FRC、FRDiv、FRVar、官方 FRSyn、semantic diagnostics、generation time 和
peak GPU memory。若只做诊断，可显式请求 Robust FRSyn；它不是官方指标。

## 10. exact FRD

先导出与非 FRD 评估一致的预测/GT cache，再运行 rolling DTW：

```bash
python -m regnn.export_emotion_query_mamba_official_frd_cache \
  --checkpoint "$REACT_FINAL_CHECKPOINT" \
  --data-dir "$REACT_DATA_DIR" \
  --source-metrics-json "$REACT_EVAL_DIR/metrics.json" \
  --results-pt "$REACT_EVAL_DIR/results_frd.pt" \
  --residual-scale 0.95 \
  --style-residual-scale 0.95 \
  --au-residual-multiplier 1.8 \
  --va-residual-multiplier 0.5 \
  --expression-residual-multiplier 1.6 \
  --selection-seed 1234

python tools/compute_frd_resumable.py \
  --results "$REACT_EVAL_DIR/results_frd.pt" \
  --output-dir "$REACT_EVAL_DIR/frd_resumable" \
  --workers 16 \
  --engine rolling
```

`regnn/frd_surrogate.py` 的 banded Soft-DTW 只用于训练，绝不能命名为 official exact
FRD。prediction cache 通常很大，不进入 Git。

## 11. 完整性检查

```bash
python -m pytest -q \
  regnn/tests/test_conditional_regnn.py \
  regnn/tests/test_query_mamba.py \
  regnn/tests/test_emotion_query_mamba.py \
  regnn/tests/test_frd_surrogate.py
```

发布前还必须执行：JSON 解析、shell syntax、Python compile、旧 checkpoint strict-load、
单 GPU one-batch smoke，以及敏感信息/大文件审计。具体通过数由实际命令输出填写，不在
文档中预先写死。
