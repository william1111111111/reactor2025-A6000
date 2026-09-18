# 来源与复现边界

归档日期：2026-09-07。

代码快照来自 `/tmp/react2025-joint-va-main-ablation-20260817` 的 Mam-Reactor 联合训练/评估版本；官方评估依赖的 `framework/`、`dataset/`、根目录 `configs/` 和 FaceVerse mean/std 来自 `/home/zhengshiyi/react2025`。归档内的 `regnn/conditional_data.py` 只增加了 `MAM_REACTOR_ROOT` 路径优先级，用于让独立目录在数据目录位于其他位置时仍能找到 FaceVerse 归一化文件，模型和损失逻辑未改动。

训练数据仍放在外部，由 `REACT_DATA_DIR` 指定。默认推理会尝试使用原仓库的 `../regnn/cache/official_test1142_targets_full_seed1234.pt`；该官方 target cache 约 1.9 GB，因此不重复复制。如果把本目录移到其他位置，需要显式设置 `REACT_TARGET_CACHE_RESULTS_PT`，或提供完整官方评估工程。

本目录保留的是最终代表性权重，不是所有历史 checkpoint：

- offline evidence：Conditional-REGNN epoch 50、EQR legacy warm-start epoch 3、Mam-Reactor epoch 3；
- online frozen anchor：Mam-Reactor epoch 10；
- online adapted anchor：Conditional-REGNN epoch 30 和 Mam-Reactor epoch 10。

所有复制进来的文件均列在 `MANIFEST.sha256`。`results/` 中的 JSON、日志和 FRD resumable 状态来自原始 run；推理过程中产生的大型 `results_frd.pt` 不预置，但可由 `infer.sh` 生成。
