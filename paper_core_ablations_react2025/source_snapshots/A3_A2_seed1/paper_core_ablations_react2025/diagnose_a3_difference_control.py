"""Compare A0/A1/A3 against the same frozen TRAIN teacher AFR cache."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from coreset_reactor.eval_multiview_boundary_student import load_model
from coreset_reactor.train_expert_student import TeacherCacheDataset
from paper_core_ablations_react2025.diagnose_difference_transfer import (
    GROUPS,
    context_diagnostics,
    sha256_file,
    summarize,
    write_json,
)


def audit(checkpoints: dict[str, Path]) -> dict:
    states = {name: torch.load(path, map_location="cpu", weights_only=False)
              for name, path in checkpoints.items()}
    configs = {name: json.loads((path.parent / "config.json").read_text())
               for name, path in checkpoints.items()}
    provenance = {name: states[name]["provenance"] for name in states}
    common = {}
    for key in (
        "cache_manifest_hash", "teacher_checkpoint_hashes",
        "initialization_checkpoint_hash", "initial_student_state_sha256",
        "observed_source_schedule_sha256",
    ):
        common[key] = len({json.dumps(provenance[name][key], sort_keys=True)
                           for name in checkpoints}) == 1
    common["global_step"] = len({states[name]["global_step"] for name in checkpoints}) == 1
    for key in ("epochs", "batch_size", "lr", "seed", "velocity_weight"):
        common[key] = len({configs[name][key] for name in checkpoints}) == 1
    if not all(common.values()):
        raise ValueError(f"A0/A1/A3 matching failed: {common}")
    if configs["A0"]["relation_objective"] != "centered_residual":
        raise ValueError("A0 objective mismatch")
    if configs["A1"]["relation_weight"] != 0:
        raise ValueError("A1 relation weight must be zero")
    if configs["A3"]["relation_objective"] != "group_weighted_mse":
        raise ValueError("A3 objective mismatch")
    if configs["A0"]["relation_weight"] != configs["A3"]["relation_weight"]:
        raise ValueError("A0/A3 coefficients differ")
    a0_cal = states["A0"]["relation_calibration"]
    a3_cal = states["A3"]["relation_calibration"]
    if a0_cal["scale"] != a3_cal["scale"] or a0_cal["weight"] != a3_cal["weight"]:
        raise ValueError("A0/A3 TRAIN scales or weights differ")
    architecture_sources = (
        "mam_reactor/regnn/expert_student.py",
        "mam_reactor/regnn/conditional_model.py",
    )
    architecture_match = all(
        provenance["A0"]["source_hashes"][key]
        == provenance["A3"]["source_hashes"][key]
        for key in architecture_sources
    )
    if not architecture_match:
        raise ValueError("A0/A3 architecture source changed")
    return {
        "status": "passed",
        "common_checks": common,
        "architecture_sources_match": architecture_match,
        "only_A0_A3_formula_change": {
            "A0": "centered_residual",
            "A3": "group_weighted_mse",
            "scale": a0_cal["scale"],
            "weight": a0_cal["weight"],
        },
        "checkpoint_sha256": {name: sha256_file(path)
                              for name, path in checkpoints.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--a0", type=Path, required=True)
    parser.add_argument("--a1", type=Path, required=True)
    parser.add_argument("--a3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    checkpoints = {"A0": args.a0, "A1": args.a1, "A3": args.a3}
    matched = audit(checkpoints)
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = TeacherCacheDataset(args.data_root, args.cache_manifest)
    models = {name: load_model(path, device) for name, path in checkpoints.items()}
    rows = []
    started = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(data), args.batch_size):
            indices = list(range(offset, min(offset + args.batch_size, len(data))))
            batch = default_collate([data[index] for index in indices])
            inputs = [batch[key].to(device) for key in (
                "speaker_audio", "speaker_emotion", "speaker_3dmm", "length",
            )]
            teacher = batch["teacher_predictions"].to(device)
            outputs = {name: model.generate(*inputs) for name, model in models.items()}
            reference_mask = outputs["A0"]["valid_mask"]
            if not all(torch.equal(value["valid_mask"], reference_mask)
                       for value in outputs.values()):
                raise ValueError("A0/A1/A3 masks differ")
            for local, index in enumerate(indices):
                if not torch.equal(
                    reference_mask[local],
                    reference_mask[local, :1].expand_as(reference_mask[local]),
                ):
                    raise ValueError("Slots do not share valid frames")
                length = int(reference_mask[local, 0].sum())
                teacher_value = teacher[local, :, :length].float().cpu()
                cached = data.rows[index]
                row = {
                    "index": index,
                    "sample_id": cached["sample_id"],
                    "view_id": int(cached.get("view_id", 0)),
                    "start": int(cached["start"]),
                    "length": length,
                }
                for name in models:
                    student = outputs[name]["prediction"][local, :, :length].float().cpu()
                    row[name] = context_diagnostics(student, teacher_value)
                rows.append(row)
            if len(rows) % 200 == 0 or len(rows) == len(data):
                print(json.dumps({"completed": len(rows), "total": len(data)}), flush=True)
    if len(rows) != 8820:
        raise ValueError("Formal A3 diagnostic requires all 8,820 contexts")
    summaries = {name: summarize(rows, name) for name in models}
    summary = {
        "protocol": (
            "REACT2025 frozen TRAIN cache; matched inputs/teacher IDs/valid frames; "
            "unrounded soft AFRs; A3 differs from A0 only by slot centering"
        ),
        "matched_audit": matched,
        **summaries,
        "A0_minus_A3_mean": {
            group: {
                metric: summaries["A0"]["groups"][group][metric]["mean"]
                - summaries["A3"]["groups"][group][metric]["mean"]
                for metric in (
                    "centered_residual_mse", "normalized_centered_residual_mse",
                    "pair_distance_mae", "pair_distance_rmse",
                )
            }
            for group in GROUPS
        },
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
    print(json.dumps({"matched": matched["status"], **summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
