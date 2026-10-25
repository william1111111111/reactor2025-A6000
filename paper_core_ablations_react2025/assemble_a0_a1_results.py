"""Assemble the matched A0/A1 REACT2025 evidence bundle.

The script never evaluates a model.  It audits and combines the frozen raw and
postprocessed VAL1142 rows, computes session-clustered paired uncertainty, and
writes the machine-readable result table plus a claim/evidence note.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("FRC", "exact_FRD", "FRDiv", "FRVar", "TLCC")
HIGHER_IS_BETTER = {"FRC": True, "exact_FRD": False, "FRDiv": True,
                    "FRVar": True, "TLCC": False}
GROUPS = {
    "speaker_to_listener": lambda row: row["role"] == "speaker",
    "listener_to_speaker": lambda row: row["role"] == "listener",
    "both_directions": lambda row: True,
}


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_row(row: dict, metrics: dict) -> dict:
    return {
        "index": int(row["index"]),
        "clip_id": row["clip_id"],
        "role": row["role"],
        "session": row["clip_id"].split("/")[1],
        "metrics": {metric: float(metrics[metric]) for metric in METRICS},
    }


def load_a1(raw_path: Path, transition_path: Path) -> tuple[list[dict], list[dict]]:
    raw_payload = read_json(raw_path)
    transition = read_json(transition_path)
    raw = [
        normalized_row(row, row["methods"]["multiview_power_4"]["metrics"])
        for row in raw_payload["rows"]
    ]
    changed = {int(row["index"]): row["new_metrics"]
               for row in transition["changed_rows"]}
    post = [normalized_row(row, changed.get(int(row["index"]), row["metrics"]))
            for row in raw]
    expected = transition["postprocess"]["changed_contexts"]
    if len(changed) != expected:
        raise ValueError("A1 transition row count mismatch")
    for group, predicate in GROUPS.items():
        selected_raw = [row for row in raw if predicate(row)]
        selected_post = [row for row in post if predicate(row)]
        published = transition["metrics"][group]
        for metric in METRICS:
            raw_mean = float(np.mean([row["metrics"][metric]
                                      for row in selected_raw]))
            post_mean = float(np.mean([row["metrics"][metric]
                                       for row in selected_post]))
            if not math.isclose(raw_mean, published["raw_p4"][metric],
                                rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"A1 raw reconstruction differs: {group} {metric}")
            if not math.isclose(
                post_mean, published["current_p4_with_postprocess"][metric],
                rel_tol=0, abs_tol=1e-12,
            ):
                raise ValueError(f"A1 post reconstruction differs: {group} {metric}")
    return raw, post


def load_comparison(path: Path) -> list[dict]:
    payload = read_json(path)
    return [normalized_row(row, row["metrics"]) for row in payload["rows"]]


def audit_population(arms: dict[str, list[dict]]) -> None:
    reference = None
    for name, rows in arms.items():
        keys = [(row["index"], row["clip_id"], row["role"]) for row in rows]
        if len(rows) != 1142:
            raise ValueError(f"{name} does not contain VAL1142")
        if sum(row["role"] == "speaker" for row in rows) != 571:
            raise ValueError(f"{name} is not balanced bidirectional VAL1142")
        if any(not all(math.isfinite(row["metrics"][m]) for m in METRICS)
               for row in rows):
            raise ValueError(f"{name} has a non-finite metric")
        if reference is None:
            reference = keys
        elif keys != reference:
            raise ValueError(f"{name} row population/order differs")


def means(rows: list[dict]) -> dict:
    return {metric: float(np.mean([row["metrics"][metric] for row in rows]))
            for metric in METRICS}


def aggregate(rows: list[dict]) -> dict:
    return {
        group: {"contexts": len(selected), **means(selected)}
        for group, predicate in GROUPS.items()
        if (selected := [row for row in rows if predicate(row)])
    }


def paired_analysis(a0: list[dict], a1: list[dict], bootstrap: int,
                    seed: int) -> dict:
    sessions: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(a0):
        sessions[row["session"]].append(index)
    names = sorted(sessions)
    rng = np.random.default_rng(seed)
    answer = {
        "definition": "A0 minus A1; 95% CI resamples sessions with replacement",
        "bootstrap_replicates": bootstrap,
        "bootstrap_seed": seed,
        "sessions": len(names),
        "metrics": {},
    }
    for metric in METRICS:
        delta = np.asarray([a0[i]["metrics"][metric] - a1[i]["metrics"][metric]
                            for i in range(len(a0))], dtype=np.float64)
        cluster_sum = np.asarray([delta[sessions[name]].sum() for name in names])
        cluster_count = np.asarray([len(sessions[name]) for name in names])
        draws = rng.integers(0, len(names), size=(bootstrap, len(names)))
        bootstrap_mean = cluster_sum[draws].sum(1) / cluster_count[draws].sum(1)
        signed = delta if HIGHER_IS_BETTER[metric] else -delta
        a1_mean = float(np.mean([row["metrics"][metric] for row in a1]))
        answer["metrics"][metric] = {
            "mean_delta": float(delta.mean()),
            "relative_delta_percent": float(100 * delta.mean() / a1_mean),
            "session_clustered_ci95": [float(np.quantile(bootstrap_mean, 0.025)),
                                        float(np.quantile(bootstrap_mean, 0.975))],
            "contexts_A0_better_fraction": float(np.mean(signed > 0)),
            "contexts_tied_fraction": float(np.mean(signed == 0)),
        }
    return answer


def combined_hash(values) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def training_metadata(checkpoint: Path) -> dict:
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = read_json(checkpoint.parent / "config.json")
    provenance = state["provenance"]
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "teacher_bank_hash": combined_hash(provenance["teacher_checkpoint_hashes"]),
        "cache_hash": provenance["cache_manifest_hash"],
        "initial_state_hash": provenance["initial_student_state_sha256"],
        "source_snapshot_hash": combined_hash(provenance.get("source_hashes", {})),
        "observed_source_schedule_sha256": provenance["observed_source_schedule_sha256"],
        "updates": int(state["global_step"]),
        "training_seconds": float(state.get("training_seconds", config.get("training_seconds", 0))),
        "peak_memory_bytes": int(state.get("peak_memory_bytes", config.get("peak_memory_bytes", 0))),
        "relation_objective": config["relation_objective"],
        "relation_weight": float(config["relation_weight"]),
        "velocity_weight": float(config["velocity_weight"]),
    }


def render_claim(path: Path, result: dict, diagnostic: dict) -> None:
    raw = result["paired"]["raw"]
    post = result["paired"]["postprocessed"]
    lines = [
        "# Claim—Evidence: A0 vs A1",
        "",
        "本页只陈述 seed 1 的配对结果；随机种子稳定性要等 seed 2/3 完成后再判断。",
        "",
        "## 受控比较",
        "",
        ("A0 与 A1 使用相同教师库、TRAIN 多时间视图缓存、初始学生、样本顺序、"
         "8820 次更新、速度损失和 p=4 全序列推理。唯一计划内训练差异是 A0 加入"
         "中心化集合差异损失；A1 的该权重为 0。匹配审计："
         f"`{diagnostic['matched_audit']['status']}`。"),
        "",
        "## 完整 VAL1142",
        "",
        "| 输出 | 指标 | A0-A1 | session bootstrap 95% CI | A0 更优样本比例 |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind, paired in (("原始", raw), ("既定后处理", post)):
        for metric in METRICS:
            item = paired["metrics"][metric]
            low, high = item["session_clustered_ci95"]
            lines.append(
                f"| {kind} | {metric} | {item['mean_delta']:.9g} | "
                f"[{low:.9g}, {high:.9g}] | {100*item['contexts_A0_better_fraction']:.2f}% |"
            )
    lines += ["", "## 差异迁移诊断", ""]
    for group in ("AU", "VA", "EX"):
        a0 = diagnostic["A0"]["groups"][group]
        a1 = diagnostic["A1"]["groups"][group]
        lines.append(
            f"- **{group}**：中心化残差 MSE {a1['centered_residual_mse']['mean']:.6g} → "
            f"{a0['centered_residual_mse']['mean']:.6g}；pair-distance MAE "
            f"{a1['pair_distance_mae']['mean']:.6g} → {a0['pair_distance_mae']['mean']:.6g}。"
        )
    a0_ret = diagnostic["A0"]["soft_FRDiv"]["ratio_of_means"]
    a1_ret = diagnostic["A1"]["soft_FRDiv"]["ratio_of_means"]
    lines += [
        f"- 教师 soft-FRDiv 保留率（均值之比）：A1 `{a1_ret:.6g}`，A0 `{a0_ret:.6g}`。",
        "",
        "## 可支持的论文表述",
        "",
    ]
    residual_better = all(
        diagnostic["A0"]["groups"][g]["centered_residual_mse"]["mean"]
        < diagnostic["A1"]["groups"][g]["centered_residual_mse"]["mean"]
        for g in ("AU", "VA", "EX")
    )
    if residual_better:
        lines.append(
            "在固定教师和训练预算下，显式中心化集合差异监督降低了 AU、VA、EX 三组的"
            "中心化残差 MSE。pair-distance 只在 VA、EX 改善，AU 略退，因此证据支持的是"
            "中心化差异表示的迁移，不是所有关系度量都一致改善。"
        )
    else:
        lines.append(
            "当前诊断没有在 AU、VA、EX 三组上同时显示更准确的集合内部差异迁移，"
            "因此不能写成普遍有效的机制结论。"
        )
    lines += [
        "",
        "VAL1142 上 exact-FRD 明确改善；FRC 和 FRDiv 的 session bootstrap 区间跨 0，"
        "FRVar 降低，后处理 FRSyn 增大。因此 seed 1 支持机制迁移和 FRD 收益，"
        "不支持整体五指标全面优于 A1。单个 seed 不能证明跨随机种子稳定性。",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--a0-raw", type=Path, required=True)
    parser.add_argument("--a0-post", type=Path, required=True)
    parser.add_argument("--a1-raw", type=Path, required=True)
    parser.add_argument("--a1-transition", type=Path, required=True)
    parser.add_argument("--a0-checkpoint", type=Path, required=True)
    parser.add_argument("--a1-checkpoint", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=root)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261023)
    args = parser.parse_args()

    a1_raw, a1_post = load_a1(args.a1_raw, args.a1_transition)
    arms = {
        "A0_raw": load_comparison(args.a0_raw),
        "A0_postprocessed": load_comparison(args.a0_post),
        "A1_raw": a1_raw,
        "A1_postprocessed": a1_post,
    }
    audit_population(arms)
    metadata = {"A0": training_metadata(args.a0_checkpoint),
                "A1": training_metadata(args.a1_checkpoint)}
    matched = {
        key: metadata["A0"][key] == metadata["A1"][key]
        for key in ("teacher_bank_hash", "cache_hash", "initial_state_hash",
                    "source_snapshot_hash", "observed_source_schedule_sha256",
                    "updates", "velocity_weight")
    }
    if not all(matched.values()):
        raise ValueError(f"Training metadata mismatch: {matched}")
    if metadata["A0"]["relation_weight"] <= 0 or metadata["A1"]["relation_weight"] != 0:
        raise ValueError("A0/A1 loss toggle is invalid")

    result = {
        "protocol": ("REACT2025 complete bidirectional VAL1142; full sequences; all "
                     "opposite-role same-session GT; official Processor/AU rounding/five metrics"),
        "training_match": matched,
        "training": metadata,
        "cost": {
            "A0_combined_raw_and_postprocess_gpu_seconds": float(sum(
                row["seconds"] for row in read_json(args.a0_raw)["rows"]
            )),
            "A1_evaluation_seconds": None,
            "A1_evaluation_cost_note": (
                "reused historical evaluation timed multiple inference variants "
                "together, so an isolated A1 p=4 time is not available"
            ),
            "teacher_cache": {
                "status": "reused",
                "contexts": 8820,
                "build_seconds": None,
                "note": "the frozen cache manifest does not record build time",
            },
        },
        "aggregate": {name: aggregate(rows) for name, rows in arms.items()},
        "paired": {
            "raw": paired_analysis(arms["A0_raw"], arms["A1_raw"],
                                   args.bootstrap, args.bootstrap_seed),
            "postprocessed": paired_analysis(
                arms["A0_postprocessed"], arms["A1_postprocessed"],
                args.bootstrap, args.bootstrap_seed),
        },
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256(path)}
                   for name, path in {
                       "A0_raw": args.a0_raw, "A0_postprocessed": args.a0_post,
                       "A1_raw": args.a1_raw, "A1_transition": args.a1_transition,
                       "diagnostic": args.diagnostic,
                   }.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "RESULTS.json", result)
    fields = ["dataset", "population", "seed", "arm", "output_kind",
              "teacher_bank_hash", "cache_hash", "initial_state_hash",
              "source_snapshot_hash", "updates", "window", "stride", "power",
              "postprocess", "FRC", "exact_FRD", "FRDiv", "FRVar", "FRSyn",
              "training_seconds", "peak_memory_bytes", "evaluation_gpu_seconds_shared",
              "status"]
    with (args.output_dir / "RESULTS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for arm in ("A0", "A1"):
            for kind, label in (("raw", "raw"), ("postprocessed", "fixed_query0_transition")):
                values = result["aggregate"][f"{arm}_{kind}"]["both_directions"]
                meta = metadata[arm]
                writer.writerow({
                    "dataset": "REACT2025", "population": "VAL1142_bidir_all_session_gt",
                    "seed": 1, "arm": arm, "output_kind": kind,
                    "teacher_bank_hash": meta["teacher_bank_hash"],
                    "cache_hash": meta["cache_hash"],
                    "initial_state_hash": meta["initial_state_hash"],
                    "source_snapshot_hash": meta["source_snapshot_hash"],
                    "updates": meta["updates"], "window": 750, "stride": 375,
                    "power": 4, "postprocess": label,
                    "FRC": values["FRC"], "exact_FRD": values["exact_FRD"],
                    "FRDiv": values["FRDiv"], "FRVar": values["FRVar"],
                    "FRSyn": values["TLCC"],
                    "training_seconds": meta["training_seconds"],
                    "peak_memory_bytes": meta["peak_memory_bytes"], "status": "complete",
                    "evaluation_gpu_seconds_shared": (
                        result["cost"]["A0_combined_raw_and_postprocess_gpu_seconds"]
                        if arm == "A0" else ""
                    ),
                })
    diagnostic = read_json(args.diagnostic)
    render_claim(args.output_dir / "CLAIM_EVIDENCE.md", result, diagnostic)
    print(json.dumps({"aggregate": result["aggregate"], "paired": result["paired"]},
                     indent=2))


if __name__ == "__main__":
    main()
