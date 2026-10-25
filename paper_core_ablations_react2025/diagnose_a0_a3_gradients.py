"""Measure matched base/auxiliary gradient norms for A0 and A3.

The probe uses the same 16 frozen TRAIN contexts as the A0 calibration.  It
reconstructs the common initial state and then reads every epoch snapshot.
No validation data or metric is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from coreset_reactor.relation_distillation import distillation_relation
from coreset_reactor.train_expert_student import TeacherCacheDataset, set_seed
from mam_reactor.regnn.conditional_model import ConditionalREGNNConfig, ConditionalREGNNLoss
from mam_reactor.regnn.expert_student import ExpertConditionalREGNN, ExpertStudentConfig


def state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def make_model(data: TeacherCacheDataset, device: torch.device) -> ExpertConditionalREGNN:
    backbone = ConditionalREGNNConfig(**data.manifest["model_config"])
    return ExpertConditionalREGNN(ExpertStudentConfig(
        backbone=backbone,
        num_experts=data.manifest["num_experts"],
        expert_embedding_dim=64,
    )).to(device)


def initial_model(data: TeacherCacheDataset, initializer: Path,
                  device: torch.device) -> ExpertConditionalREGNN:
    # This is the exact construction order in train_expert_student.py.
    set_seed(1)
    model = make_model(data, device)
    saved = torch.load(initializer, map_location="cpu", weights_only=False)
    expected = asdict(model.config.backbone)
    actual = dict(saved["model_config"])
    for key, value in expected.items():
        actual.setdefault(key, value)
    if actual != expected:
        raise ValueError("Backbone initialization config mismatch")
    model.backbone.load_state_dict(saved["state_dict"], strict=True)
    return model


def snapshot_model(data: TeacherCacheDataset, checkpoint: Path,
                   device: torch.device) -> ExpertConditionalREGNN:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_model(data, device)
    model.load_state_dict(state["state_dict"], strict=True)
    return model


def norm(grads) -> float:
    values = [gradient.float().square().sum() for gradient in grads
              if gradient is not None]
    return float(torch.sqrt(torch.stack(values).sum()))


def probe(model: torch.nn.Module, data: TeacherCacheDataset, indices: list[int],
          objective: str, scale: torch.Tensor, weight: float,
          device: torch.device, batch_size: int) -> dict:
    model.train()
    criterion = ConditionalREGNNLoss(ccc_weight=1.0, velocity_weight=0.05)
    parameters = tuple(model.parameters())
    rows = []
    for offset in range(0, len(indices), batch_size):
        batch = default_collate([data[index] for index in indices[offset:offset + batch_size]])
        inputs = [batch[key].to(device) for key in (
            "speaker_audio", "speaker_emotion", "speaker_3dmm", "length",
        )]
        teacher = batch["teacher_predictions"].to(device)
        # Fix dropout masks across arms, objectives and checkpoints.
        torch.manual_seed(20261023 + offset)
        torch.cuda.manual_seed_all(20261023 + offset)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            output = model.generate(*inputs)
            prediction, mask = output["prediction"], output["valid_mask"]
            base = criterion(
                prediction.flatten(0, 1), teacher.flatten(0, 1), mask.flatten(0, 1)
            )["loss"]
        auxiliary = distillation_relation(
            prediction, teacher, mask, scale, objective
        )[0]
        base_grads = torch.autograd.grad(
            base, parameters, retain_graph=True, allow_unused=True)
        auxiliary_grads = torch.autograd.grad(
            auxiliary, parameters, retain_graph=False, allow_unused=True)
        base_norm = norm(base_grads)
        auxiliary_norm = norm(auxiliary_grads)
        rows.append({
            "indices": indices[offset:offset + batch_size],
            "base_loss": float(base.detach()),
            "auxiliary_loss": float(auxiliary.detach()),
            "base_gradient_norm": base_norm,
            "auxiliary_gradient_norm": auxiliary_norm,
            "weighted_auxiliary_gradient_norm": weight * auxiliary_norm,
            "weighted_auxiliary_to_base_ratio": weight * auxiliary_norm / max(base_norm, 1e-12),
        })
    answer = {"batches": rows}
    for key in (
        "base_loss", "auxiliary_loss", "base_gradient_norm",
        "auxiliary_gradient_norm", "weighted_auxiliary_gradient_norm",
        "weighted_auxiliary_to_base_ratio",
    ):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        answer[key] = {"mean": float(values.mean()), "median": float(np.median(values))}
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--a0-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = TeacherCacheDataset(args.data_root, args.cache_manifest)
    a0_state = torch.load(args.a0_dir / "student.pth", map_location="cpu", weights_only=False)
    calibration = a0_state["relation_calibration"]
    indices = calibration["sample_indices"][:16]
    scale = torch.tensor(calibration["scale"], device=device)
    weight = float(calibration["weight"])
    expected_initial = a0_state["provenance"]["initial_student_state_sha256"]

    common = initial_model(data, args.initializer, device)
    observed_initial = state_digest(common)
    if observed_initial != expected_initial:
        raise ValueError("Reconstructed initial state does not match A0 provenance")
    results = {}
    for arm, objective in (("A0", "centered_residual"), ("A3", "group_weighted_mse")):
        results[arm] = {
            "initial": probe(common, data, indices, objective, scale, weight,
                             device, args.batch_size)
        }
        run_dir = args.a0_dir if arm == "A0" else args.a3_dir
        for epoch in range(1, 5):
            checkpoint = run_dir / f"student_epoch{epoch:04d}.pth"
            model = snapshot_model(data, checkpoint, device)
            results[arm][f"epoch_{epoch}"] = probe(
                model, data, indices, objective, scale, weight,
                device, args.batch_size)
            del model
    output = {
        "protocol": (
            "Same 16 frozen TRAIN contexts, batch size 4, fixed dropout seeds, "
            "TF32 disabled; base is MSE+CCC+0.05*velocity; auxiliary is raw "
            "A0 centered residual or A3 group-weighted MSE."
        ),
        "initial_state_sha256": observed_initial,
        "sample_indices": indices,
        "relation_scale": calibration["scale"],
        "relation_weight": weight,
        "results": results,
    }
    write_json(args.output, output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
