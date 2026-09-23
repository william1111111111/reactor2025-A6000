"""Matched REACT2024 VAL evaluation for Mam-Reactor teacher and KD student."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MAM_ROOT = ROOT / "mam_reactor"
sys.path.insert(0, str(MAM_ROOT))

from regnn.conditional_model import ConditionalREGNN, ConditionalREGNNConfig  # noqa: E402
from regnn.emotion_query_mamba import (  # noqa: E402
    EmotionQueryMambaConfig,
    EmotionQueryResidualMamba,
)
from regnn.expert_student import ExpertConditionalREGNN, ExpertStudentConfig  # noqa: E402


def expand_student_predictions(prediction: torch.Tensor, scale: float) -> torch.Tensor:
    """Post-hoc expand K predictions around their per-frame mean."""
    if prediction.ndim != 3 or prediction.shape[-1] != 25:
        raise ValueError("Expected [K,T,25] predictions")
    if not np.isfinite(scale) or scale < 1:
        raise ValueError("student spread scales must be finite and >= 1")
    if scale == 1:
        return prediction.clone()
    center = prediction.mean(dim=0, keepdim=True)
    expanded = center + scale * (prediction - center)
    expanded[..., :15] = expanded[..., :15].clamp(0, 1)
    expanded[..., 15:17] = expanded[..., 15:17].clamp(-1, 1)
    expression = expanded[..., 17:25].clamp_min(0)
    expanded[..., 17:25] = expression / expression.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--react2024-project", type=Path, required=True,
                        help="Directory containing common.py, evaluate_query.py, and metrics.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--residual-scale", type=float, default=3.0)
    parser.add_argument("--style-residual-scale", type=float, default=3.0)
    parser.add_argument("--student-spread-scales", type=float, nargs="+", default=[1.0],
                        help="VAL-only post-hoc spread factors around the 10-slot mean")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {args.output}")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if (1.0 not in args.student_spread_scales
            or len(set(args.student_spread_scales)) != len(args.student_spread_scales)
            or any(not np.isfinite(scale) or scale < 1 for scale in args.student_spread_scales)):
        raise ValueError("student-spread-scales must be unique, finite, >=1, and include 1")

    # The existing REACT2024 project owns official split ordering, feature
    # paths, neighbour masks, and metric implementations.
    sys.path.insert(0, str(args.react2024_project.parent.resolve()))
    package = args.react2024_project.name
    common = __import__(f"{package}.common", fromlist=["OUT"])
    query_eval = __import__(f"{package}.evaluate_query", fromlist=["load_source"])
    metric_module = __import__(f"{package}.metrics", fromlist=["frc"])

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(4)

    student_state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    student_config = student_state["student_config"]
    backbone_config = ConditionalREGNNConfig(**student_config["backbone"])
    student = ExpertConditionalREGNN(ExpertStudentConfig(
        backbone=backbone_config,
        num_experts=int(student_config["num_experts"]),
        expert_embedding_dim=int(student_config["expert_embedding_dim"]),
    ))
    student.load_state_dict(student_state["state_dict"], strict=True)
    student.eval().to(device)

    teacher_state = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    anchor = ConditionalREGNN(ConditionalREGNNConfig(**teacher_state["anchor_model_config"]))
    query_config = EmotionQueryMambaConfig.from_checkpoint_payload(
        dict(teacher_state["emotion_query_model_config"])
    )
    teacher = EmotionQueryResidualMamba(anchor, query_config)
    teacher.load_state_dict(teacher_state["state_dict"], strict=True)
    teacher.eval().to(device)

    split = "val"
    manifest = common.read(common.OUT / "MANIFEST.json")
    directions = manifest["splits"][split]["directions"]
    if not directions:
        raise ValueError("Official validation split is empty")
    stats_root = common.OUT / "external" / "FaceVerse"
    mean = torch.from_numpy(np.load(stats_root / "mean_face.npy")).float()
    std = torch.from_numpy(np.load(stats_root / "std_face.npy")).float().clamp_min(1e-8)
    neighbours = np.load(common.OUT / f"neighbour_emotion_{split}.npy")
    if neighbours.shape != (len(directions), len(directions)):
        raise ValueError(f"Neighbour matrix shape mismatch: {neighbours.shape}")

    teacher_predictions = np.empty((len(directions), 10, 750, 25), dtype=np.float32)
    student_predictions_by_scale = {
        scale: np.empty_like(teacher_predictions) for scale in args.student_spread_scales
    }
    targets = np.empty((len(directions), 750, 25), dtype=np.float32)
    teacher_seconds = student_seconds = 0.0
    batch_size = args.batch_size
    with torch.inference_mode():
        for start in range(0, len(directions), batch_size):
            stop = min(start + batch_size, len(directions))
            batch = directions[start:stop]
            sources = [query_eval.load_source(split, item["source"], mean, std)
                       for item in batch]
            lengths = [min(x[0].shape[0], x[1].shape[0], x[2].shape[0]) for x in sources]
            if any(length != 750 for length in lengths):
                raise ValueError(f"Expected 750-frame validation input; got {lengths}")
            inputs = [torch.stack([x[index] for x in sources]).to(device)
                      for index in range(3)]
            length_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
            tick = time.perf_counter()
            teacher_output = teacher(
                *inputs, length_tensor,
                residual_scale=args.residual_scale,
                style_residual_scale=args.style_residual_scale,
            )["prediction"].float()
            teacher_output[..., :15] = torch.round(teacher_output[..., :15])
            teacher_rows = teacher_output.cpu().numpy()
            teacher_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            student_output = student.generate(*inputs, length_tensor)["prediction"].float()
            student_seconds += time.perf_counter() - tick

            for offset, (item, teacher_row) in enumerate(zip(batch, teacher_rows)):
                teacher_predictions[start + offset] = teacher_row
                for scale, destination in student_predictions_by_scale.items():
                    scaled = expand_student_predictions(student_output[offset], scale)
                    scaled[..., :15] = torch.round(scaled[..., :15])
                    destination[start + offset] = scaled.cpu().numpy()
                targets[start + offset] = query_eval.load_emotion(split, item["target"])
            if stop % 100 < batch_size or stop == len(directions):
                print(f"[react2024-distill-val] {stop}/{len(directions)}", flush=True)

    teacher_metrics = {
        "FRC": metric_module.frc(teacher_predictions, targets, neighbours, device),
        **metric_module.diversity_metrics(teacher_predictions),
    }
    student_spread_sweep = []
    for scale, predictions in student_predictions_by_scale.items():
        metrics = {
            "FRC": metric_module.frc(predictions, targets, neighbours, device),
            **metric_module.diversity_metrics(predictions),
        }
        student_spread_sweep.append({"scale": scale, "metrics": metrics})
    student_metrics = next(
        entry["metrics"] for entry in student_spread_sweep if entry["scale"] == 1.0
    )
    baseline_metrics = student_metrics
    for entry in student_spread_sweep:
        entry["delta_vs_scale_1"] = {
            key: entry["metrics"][key] - baseline_metrics[key]
            for key in baseline_metrics
        }
    reference_path = common.OUT / "training" / "query_eval" / (
        "val_emotion-query-mamba-epoch0030-seed1.json"
    )
    teacher_reference = None
    if reference_path.is_file():
        saved_reference = common.read(reference_path)
        teacher_reference = saved_reference.get("Mam_Reactor")
    deltas = {key: student_metrics[key] - teacher_metrics[key]
              for key in teacher_metrics}
    payload = {
        "protocol": "REACT2024 official VAL directions/neighbour matrix; native source-only Mam-Reactor generation; AU-rounded predictions",
        "split": split,
        "contexts": len(directions),
        "K": 10,
        "teacher_inference": {
            "residual_scale": args.residual_scale,
            "style_residual_scale": args.style_residual_scale,
            "AU_rounding": True,
        },
        "teacher": {
            "checkpoint": str(args.teacher_checkpoint.resolve()),
            "checkpoint_sha256": common.sha256(args.teacher_checkpoint),
            "epoch": int(teacher_state["epoch"]),
            "metrics": teacher_metrics,
            "existing_official_eval_reference": teacher_reference,
            "inference_seconds": teacher_seconds,
        },
        "student": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": common.sha256(args.checkpoint),
            "parameters": sum(p.numel() for p in student.parameters()),
            "metrics": student_metrics,
            "inference_seconds": student_seconds,
        },
        "student_spread_sweep": student_spread_sweep,
        "delta_student_minus_teacher": deltas,
        "train_provenance": student_state.get("provenance"),
        "teacher_cache_manifest_hash": student_state.get("cache_manifest_hash"),
        "notes": [
            "FRC, S_MSE, FRVar, and FRDvs use the existing REACT2024 metric implementation.",
            "Validation only; the official TEST split was not used for training or selection.",
            "Exact DTW-FRD is not computed in this fast matched VAL pass.",
            "Student spread scaling is inference-only, centered on the per-frame mean of its ten slots, and tuned/reported on VAL.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"teacher": teacher_metrics, "student": student_metrics,
                      "delta": deltas}, indent=2), flush=True)


if __name__ == "__main__":
    main()
