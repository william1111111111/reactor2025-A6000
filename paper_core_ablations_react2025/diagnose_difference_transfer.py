"""Matched soft-AFR difference-transfer diagnosis for A0 versus A1.

This reads only the frozen REACT2025 TRAIN teacher cache.  It does not use VAL
targets and does not round AUs.  Teacher IDs remain in their cached order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from coreset_reactor.eval_multiview_boundary_student import load_model
from coreset_reactor.teacher_cache import sha256_file
from coreset_reactor.train_expert_student import TeacherCacheDataset


GROUPS = {"AU": (0, 15), "VA": (15, 17), "EX": (17, 25)}
ENERGY_EPSILON = 1e-8
FRDIV_EPSILON = 1e-10


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def soft_frdiv(value: torch.Tensor) -> float:
    """Official S_MSE formula for one unrounded [K,T,C] context."""
    flat = value.float().reshape(value.shape[0], -1)
    squared = torch.cdist(flat, flat).square().sum()
    denominator = value.shape[0] * (value.shape[0] - 1) * flat.shape[1]
    return float(squared / denominator)


def distance_matrices(value: torch.Tensor) -> dict[str, torch.Tensor]:
    answer = {}
    for name, (start, end) in GROUPS.items():
        flat = value[..., start:end].float().reshape(value.shape[0], -1)
        answer[name] = torch.cdist(flat, flat) / math.sqrt(flat.shape[1])
    return answer


def context_diagnostics(student: torch.Tensor, teacher: torch.Tensor) -> dict:
    if student.shape != teacher.shape or student.ndim != 3:
        raise ValueError("Expected matching [K,T,25] soft trajectories")
    if student.shape[0] < 2 or student.shape[-1] != 25:
        raise ValueError("Expected multiple teacher-aligned 25-channel slots")
    student_matrices = distance_matrices(student)
    teacher_matrices = distance_matrices(teacher)
    upper = torch.triu_indices(student.shape[0], student.shape[0], 1)
    groups = {}
    for name, (start, end) in GROUPS.items():
        s = student[..., start:end].float()
        t = teacher[..., start:end].float()
        residual_error = (s - s.mean(0, keepdim=True)) - (t - t.mean(0, keepdim=True))
        centered_mse = float(residual_error.square().mean())
        teacher_energy = float((t - t.mean(0, keepdim=True)).square().mean())
        pair_error = (
            student_matrices[name][upper[0], upper[1]]
            - teacher_matrices[name][upper[0], upper[1]]
        )
        groups[name] = {
            "centered_residual_mse": centered_mse,
            "teacher_centered_energy": teacher_energy,
            "normalized_centered_residual_mse": centered_mse / max(
                teacher_energy, ENERGY_EPSILON
            ),
            "teacher_energy_near_zero": teacher_energy <= ENERGY_EPSILON,
            "pair_distance_mae": float(pair_error.abs().mean()),
            "pair_distance_rmse": float(pair_error.square().mean().sqrt()),
        }
    teacher_diversity = soft_frdiv(teacher)
    student_diversity = soft_frdiv(student)
    return {
        "groups": groups,
        "soft_FRDiv": student_diversity,
        "teacher_soft_FRDiv": teacher_diversity,
        "soft_FRDiv_retention": (
            student_diversity / teacher_diversity
            if teacher_diversity > FRDIV_EPSILON
            else None
        ),
    }


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Diagnostic distribution is empty or non-finite")
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
    }


def summarize(rows: list[dict], arm: str) -> dict:
    result = {"contexts": len(rows), "groups": {}}
    for group in GROUPS:
        records = [row[arm]["groups"][group] for row in rows]
        result["groups"][group] = {
            key: distribution([record[key] for record in records])
            for key in (
                "centered_residual_mse",
                "normalized_centered_residual_mse",
                "pair_distance_mae",
                "pair_distance_rmse",
            )
        }
        result["groups"][group]["near_zero_teacher_energy_contexts"] = sum(
            record["teacher_energy_near_zero"] for record in records
        )
    teacher = [row[arm]["teacher_soft_FRDiv"] for row in rows]
    student = [row[arm]["soft_FRDiv"] for row in rows]
    retention = [
        row[arm]["soft_FRDiv_retention"]
        for row in rows
        if row[arm]["soft_FRDiv_retention"] is not None
    ]
    result["soft_FRDiv"] = {
        "teacher": distribution(teacher),
        "student": distribution(student),
        "ratio_of_means": float(np.mean(student) / np.mean(teacher)),
        "context_retention": distribution(retention),
        "near_zero_teacher_contexts_excluded_from_ratio": len(rows) - len(retention),
    }
    return result


def matched_audit(a0_path: Path, a1_path: Path) -> dict:
    states = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in (("A0", a0_path), ("A1", a1_path))
    }
    configs = {
        name: json.loads((path.parent / "config.json").read_text())
        for name, path in (("A0", a0_path), ("A1", a1_path))
    }
    provenance = {name: states[name]["provenance"] for name in states}
    checks = {
        "cache_manifest_hash": provenance["A0"]["cache_manifest_hash"]
        == provenance["A1"]["cache_manifest_hash"],
        "teacher_checkpoint_hashes": provenance["A0"]["teacher_checkpoint_hashes"]
        == provenance["A1"]["teacher_checkpoint_hashes"],
        "initialization_checkpoint_hash": provenance["A0"]["initialization_checkpoint_hash"]
        == provenance["A1"]["initialization_checkpoint_hash"],
        "initial_student_state_sha256": provenance["A0"]["initial_student_state_sha256"]
        == provenance["A1"]["initial_student_state_sha256"],
        "observed_source_schedule_sha256": provenance["A0"]["observed_source_schedule_sha256"]
        == provenance["A1"]["observed_source_schedule_sha256"],
        "global_step": states["A0"]["global_step"] == states["A1"]["global_step"],
        "epochs": configs["A0"]["epochs"] == configs["A1"]["epochs"],
        "batch_size": configs["A0"]["batch_size"] == configs["A1"]["batch_size"],
        "lr": configs["A0"]["lr"] == configs["A1"]["lr"],
        "seed": configs["A0"]["seed"] == configs["A1"]["seed"],
        "velocity_weight": configs["A0"]["velocity_weight"]
        == configs["A1"]["velocity_weight"],
    }
    if not all(checks.values()):
        raise ValueError(f"A0/A1 matched audit failed: {checks}")
    if configs["A0"]["relation_objective"] != "centered_residual":
        raise ValueError("A0 must use centered_residual")
    if not configs["A0"]["relation_weight"] > 0 or configs["A1"]["relation_weight"] != 0:
        raise ValueError("A0/A1 relation weights do not implement the planned toggle")
    return {
        "status": "passed",
        "checks": checks,
        "only_planned_loss_toggle": {
            "A0_relation_objective": configs["A0"]["relation_objective"],
            "A0_relation_weight": configs["A0"]["relation_weight"],
            "A1_relation_weight": configs["A1"]["relation_weight"],
        },
        "checkpoint_sha256": {
            "A0": sha256_file(a0_path),
            "A1": sha256_file(a1_path),
        },
        "cache_manifest_hash": provenance["A0"]["cache_manifest_hash"],
        "initial_state_hash": provenance["A0"]["initial_student_state_sha256"],
        "source_schedule_hash": provenance["A0"]["observed_source_schedule_sha256"],
        "updates": states["A0"]["global_step"],
    }


def fixed_example_indices(rows: list[dict], count: int = 6) -> list[int]:
    candidates = []
    for index, row in enumerate(rows):
        if int(row["length"]) < 750:
            continue
        key = f"{row['sample_id']}|{row.get('view_id', 0)}|{row['start']}"
        candidates.append((hashlib.sha256(key.encode()).hexdigest(), index, row["sample_id"]))
    chosen, recordings = [], set()
    for _, index, sample_id in sorted(candidates):
        if sample_id in recordings:
            continue
        recordings.add(sample_id)
        chosen.append(index)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError("Not enough deterministic long-view examples")
    return chosen


def save_example_figures(directory: Path, number: int, payload: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ("teacher", "A1", "A0")
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    vmax = max(
        float(payload[method]["distance_matrices"][group].max())
        for method in methods
        for group in GROUPS
    )
    for row, group in enumerate(GROUPS):
        for column, method in enumerate(methods):
            image = axes[row, column].imshow(
                payload[method]["distance_matrices"][group].numpy(),
                vmin=0,
                vmax=vmax,
                cmap="viridis",
            )
            axes[row, column].set_title(f"{method} {group}")
            axes[row, column].set_xlabel("slot")
            axes[row, column].set_ylabel("slot")
    fig.colorbar(image, ax=axes, shrink=0.7, label="RMS distance")
    fig.savefig(directory / f"example_{number:02d}_distance_matrices.png", dpi=160)
    plt.close(fig)

    teacher = payload["teacher"]["trajectory"]
    au_channel = int(teacher[..., :15].var((0, 1)).argmax())
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True, constrained_layout=True)
    for row, method in enumerate(methods):
        trajectory = payload[method]["trajectory"]
        for slot in range(trajectory.shape[0]):
            axes[row, 0].plot(trajectory[slot, :, au_channel].numpy(), linewidth=0.7)
            axes[row, 1].plot(trajectory[slot, :, 15].numpy(), linewidth=0.7)
        axes[row, 0].set_title(f"{method}: AU{au_channel}")
        axes[row, 1].set_title(f"{method}: valence")
        axes[row, 0].set_ylabel("soft value")
    axes[-1, 0].set_xlabel("frame")
    axes[-1, 1].set_xlabel("frame")
    fig.savefig(directory / f"example_{number:02d}_soft_trajectories.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--a0", type=Path, required=True)
    parser.add_argument("--a1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-contexts", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    figures = args.output / "figures"
    figures.mkdir()

    torch.set_num_threads(2)
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    audit = matched_audit(args.a0, args.a1)
    data = TeacherCacheDataset(args.data_root, args.cache_manifest)
    model_a0 = load_model(args.a0, device)
    model_a1 = load_model(args.a1, device)
    selected = fixed_example_indices(data.rows)
    selected_set = set(selected)
    rows, examples = [], {}
    limit = min(len(data), args.max_contexts or len(data))
    started = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, limit, args.batch_size):
            indices = list(range(offset, min(offset + args.batch_size, limit)))
            batch = default_collate([data[index] for index in indices])
            inputs = [batch[key].to(device) for key in (
                "speaker_audio", "speaker_emotion", "speaker_3dmm", "length"
            )]
            teacher = batch["teacher_predictions"].to(device)
            output_a0 = model_a0.generate(*inputs)
            output_a1 = model_a1.generate(*inputs)
            if not torch.equal(output_a0["valid_mask"], output_a1["valid_mask"]):
                raise ValueError("A0/A1 valid masks differ")
            mask = output_a0["valid_mask"]
            for local, index in enumerate(indices):
                if not torch.equal(mask[local], mask[local, :1].expand_as(mask[local])):
                    raise ValueError("Slots do not share valid frames")
                length = int(mask[local, 0].sum())
                values = {
                    "teacher": teacher[local, :, :length].float().cpu(),
                    "A0": output_a0["prediction"][local, :, :length].float().cpu(),
                    "A1": output_a1["prediction"][local, :, :length].float().cpu(),
                }
                cached = data.rows[index]
                row = {
                    "index": index,
                    "sample_id": cached["sample_id"],
                    "view_id": int(cached.get("view_id", 0)),
                    "start": int(cached["start"]),
                    "length": length,
                    "A0": context_diagnostics(values["A0"], values["teacher"]),
                    "A1": context_diagnostics(values["A1"], values["teacher"]),
                }
                rows.append(row)
                if index in selected_set:
                    examples[index] = {
                        "metadata": {key: row[key] for key in (
                            "index", "sample_id", "view_id", "start", "length"
                        )},
                        **{
                            method: {
                                "trajectory": values[method],
                                "distance_matrices": distance_matrices(values[method]),
                            }
                            for method in ("teacher", "A1", "A0")
                        },
                    }
            if len(rows) % 200 == 0 or len(rows) == limit:
                print(json.dumps({"completed": len(rows), "total": limit}), flush=True)

    if len(rows) != limit:
        raise ValueError("Diagnostic context count mismatch")
    if args.max_contexts is None and len(rows) != 8820:
        raise ValueError("Formal difference diagnosis requires all 8,820 cached contexts")
    if args.max_contexts is None and set(examples) != selected_set:
        raise ValueError("A fixed example was not evaluated")

    summaries = {arm: summarize(rows, arm) for arm in ("A0", "A1")}
    summary = {
        "protocol": (
            "REACT2025 frozen TRAIN teacher cache; identical cached views, teacher IDs, "
            "valid frames and slot order; soft AFRs without AU rounding or postprocess"
        ),
        "matched_audit": audit,
        "energy_epsilon": ENERGY_EPSILON,
        "soft_FRDiv_epsilon": FRDIV_EPSILON,
        "A0": summaries["A0"],
        "A1": summaries["A1"],
        "A0_minus_A1_mean": {
            group: {
                metric: summaries["A0"]["groups"][group][metric]["mean"]
                - summaries["A1"]["groups"][group][metric]["mean"]
                for metric in (
                    "centered_residual_mse",
                    "normalized_centered_residual_mse",
                    "pair_distance_mae",
                    "pair_distance_rmse",
                )
            }
            for group in GROUPS
        },
        "fixed_example_indices": selected,
        "contexts": len(rows),
        "seconds": time.monotonic() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "source_sha256": sha256_file(Path(__file__)),
        "cache_manifest_sha256": sha256_file(args.cache_manifest),
    }
    write_json(args.output / "summary.json", summary)
    with (args.output / "per_context.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    for number, index in enumerate(selected):
        if index not in examples:
            continue
        payload = examples[index]
        torch.save(payload, args.output / f"example_{number:02d}.pt")
        save_example_figures(figures, number, payload)
    print(json.dumps({"matched": audit["status"], "contexts": len(rows),
                      "A0": summary["A0"], "A1": summary["A1"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
