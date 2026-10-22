"""Train one expert-conditioned student from a fixed teacher-output cache."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import hashlib
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import default_collate
from coreset_reactor.relation_distillation import distillation_relation, slot_distances

ROOT = Path(__file__).resolve().parents[1]
MAM_ROOT = ROOT / "mam_reactor"
sys.path.insert(0, str(MAM_ROOT))

from regnn.conditional_data import ConditionalReactionDataset
from regnn.conditional_model import ConditionalREGNNConfig, ConditionalREGNNLoss
from regnn.expert_student import ExpertConditionalREGNN, ExpertStudentConfig


class TeacherCacheDataset(Dataset):
    def __init__(self, data_root: Path, cache_manifest: Path):
        self.cache_manifest_path = cache_manifest.resolve()
        self.manifest = json.loads(self.cache_manifest_path.read_text())
        self.source = ConditionalReactionDataset(
            str(data_root.resolve()), "train", clip_length=self.manifest["clip_length"],
            training=True, target_mode="paired", seed=self.manifest["source_seed"],
            session_allowlist=self.manifest.get('session_allowlist'),
            crop_view_mode=self.manifest.get("crop_view_mode", "random"),
            crop_view_count=self.manifest.get("crop_views", 1),
        )
        self.source.set_epoch(self.manifest["view_epoch"])
        self.rows = []
        for shard in self.manifest["shards"]:
            self.rows.extend(torch.load(self.cache_manifest_path.parent / shard,
                                        map_location="cpu"))
        if len(self.rows) != self.manifest["contexts"]:
            raise ValueError("Cache context count mismatch")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        source = self.source[(row["sample_index"], row.get("view_id", 0))]
        checks = (
            source["clip_id"] == row["sample_id"],
            int(source["start"]) == row["start"],
            int(source["length"]) == row["length"],
            int(source["crop_view_id"]) == row["view_id"],
        )
        if not all(checks):
            raise ValueError(f"Cached teacher/input view mismatch: {row['sample_id']}")
        return {
            "speaker_audio": source["speaker_audio"],
            "speaker_emotion": source["speaker_emotion"],
            "speaker_3dmm": source["speaker_3dmm"],
            "length": source["length"],
            "sample_id": row["sample_id"],
            "teacher_predictions": row["teacher_predictions"].float(),
        }


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def source_commit_sha() -> str:
    """Record the source revision when available, including staged deployments."""
    configured = os.environ.get("CORESET_GIT_COMMIT")
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable (source hashes are recorded separately)"


def validate_frozen_calibration(provenance, cache_hash, initializer_hash, state_hash, objective):
    """Reuse TRAIN-only calibration only for the identical cache and initial state."""
    expected = {'cache_manifest_hash': cache_hash,
                'initialization_checkpoint_hash': initializer_hash,
                'initial_student_state_sha256': state_hash}
    for key, value in expected.items():
        if provenance[key] != value:
            raise ValueError(f'Frozen relation calibration mismatch: {key}')
    calibration = provenance['relation_calibration']
    if calibration['objective'] != objective:
        raise ValueError('Frozen relation objective mismatch')
    scale = np.asarray(calibration['scale'], dtype=np.float64)
    weight = float(calibration['weight'])
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Invalid frozen relation scales')
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError('Invalid frozen relation weight')
    return scale.tolist(), weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--expert-embedding-dim", type=int, default=64)
    parser.add_argument("--initialize-backbone", type=Path)
    parser.add_argument('--velocity-weight', type=float, default=.05,
                        help='0 removes velocity supervision; default preserves the original recipe')
    parser.add_argument('--frozen-relation-provenance', type=Path,
                        help='Reuse same-cache/same-init TRAIN calibration instead of recalibrating')
    parser.add_argument("--relation-grad-ratio", type=float, default=0.0,
                        help="Calibrate relation weight on TRAIN initial gradients; 0 disables it")
    parser.add_argument('--relation-objective', choices=('distance', 'centered_direction', 'centered_residual', 'au_decision'),
                        default='distance', help='Only the auxiliary objective changes; model and KD stay fixed')
    parser.add_argument('--print-every', type=int, default=1)
    parser.add_argument('--save-every', type=int, default=0,
                        help='Retain epoch snapshots for matched convergence checks')
    args = parser.parse_args()
    if not np.isfinite(args.velocity_weight) or args.velocity_weight < 0:
        raise ValueError('velocity-weight must be finite and nonnegative')
    if args.frozen_relation_provenance and args.relation_grad_ratio != 0:
        raise ValueError('Choose frozen calibration OR a fresh relation-grad-ratio')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    data = TeacherCacheDataset(args.data_root, args.cache_manifest)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    backbone = ConditionalREGNNConfig(**data.manifest["model_config"])
    config = ExpertStudentConfig(
        backbone=backbone, num_experts=data.manifest["num_experts"],
        expert_embedding_dim=args.expert_embedding_dim,
    )
    model = ExpertConditionalREGNN(config).to(device)
    initialization_hash = None
    if args.initialize_backbone is not None:
        from coreset_reactor.teacher_cache import sha256_file
        initialization = torch.load(args.initialize_backbone, map_location="cpu")
        saved_config = dict(initialization["model_config"])
        expected_config = asdict(backbone)
        for key, value in expected_config.items():
            saved_config.setdefault(key, value)
        if saved_config != expected_config:
            raise ValueError("Backbone initialization config mismatch")
        model.backbone.load_state_dict(initialization["state_dict"], strict=True)
        initialization_hash = sha256_file(args.initialize_backbone)
    state_digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        state_digest.update(name.encode())
        state_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    criterion = ConditionalREGNNLoss(ccc_weight=1.0, velocity_weight=args.velocity_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4, betas=(.9, .99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.lr * .05,
    )
    relation_scale, relation_weight, calibration = None, 0.0, {}
    if args.relation_grad_ratio < 0:
        raise ValueError('relation-grad-ratio must be nonnegative')
    if args.frozen_relation_provenance:
        from coreset_reactor.teacher_cache import sha256_file
        frozen = json.loads(args.frozen_relation_provenance.read_text())
        scales, relation_weight = validate_frozen_calibration(
            frozen, data.manifest['cache_manifest_hash'], initialization_hash,
            state_digest.hexdigest(), args.relation_objective)
        relation_scale = torch.tensor(scales, device=device, dtype=torch.float32)
        calibration = {'mode': 'frozen_train_only_calibration', 'scale': scales,
                       'weight': relation_weight, 'objective': args.relation_objective,
                       'reference_provenance_sha256': sha256_file(args.frozen_relation_provenance),
                       'note': 'No current gradient probes; inherited weight is not a claim of current 0.25 gradient ratio.'}
        print('Frozen relation calibration ' + json.dumps(calibration), flush=True)
    elif args.relation_grad_ratio:
        # Use only the fixed TRAIN cache. Calibration must leave training RNG,
        # dropout, sampler and initialization identical to the baseline.
        distances = []
        calibration_indices = np.random.default_rng(args.seed).choice(
            len(data.rows), size=min(32, len(data.rows)), replace=False).tolist()
        for index in calibration_indices:
            row = data.rows[index]
            target = row['teacher_predictions'].float()[None]
            mask = (torch.arange(target.shape[2])[None, None] < row['length']).expand(target.shape[:3])
            distances.append(slot_distances(target, mask))
        relation_scale = torch.cat(distances).mean((0, 1)).clamp_min(.01).to(device)
        probes = []
        with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
            probe_indices = calibration_indices[:16]
            for offset in range(0, len(probe_indices), args.batch_size):
                batch = default_collate([data[i] for i in probe_indices[offset:offset + args.batch_size]])
                inputs = [batch[key].to(device) for key in (
                    'speaker_audio', 'speaker_emotion', 'speaker_3dmm', 'length')]
                teacher = batch['teacher_predictions'].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == 'cuda'):
                    output = model.generate(*inputs)
                    prediction, mask = output['prediction'], output['valid_mask']
                    kd = criterion(prediction.flatten(0, 1), teacher.flatten(0, 1), mask.flatten(0, 1))['loss']
                rel = distillation_relation(prediction, teacher, mask, relation_scale,
                                            args.relation_objective)[0]
                parameters = tuple(model.parameters())
                def grad_norm(value, retain_graph):
                    grads = torch.autograd.grad(value, parameters, retain_graph=retain_graph, allow_unused=True)
                    return float(torch.sqrt(sum(g.float().square().sum() for g in grads if g is not None)))
                kd_norm, rel_norm = grad_norm(kd, True), grad_norm(rel, False)
                probes.append({'kd_loss': float(kd.detach()), 'relation_loss': float(rel.detach()),
                               'kd_gradient': kd_norm, 'relation_gradient': rel_norm})
        relation_weight = min(1.0, args.relation_grad_ratio * float(np.median([
            p['kd_gradient'] / max(p['relation_gradient'], 1e-12) for p in probes])))
        calibration = {'train_scale_examples': len(calibration_indices), 'gradient_probe_examples': len(probe_indices),
                       'sample_indices': calibration_indices,
                       'scale': relation_scale.tolist(), 'weight': relation_weight,
                       'target_gradient_ratio': args.relation_grad_ratio, 'probes': probes,
                       'objective': args.relation_objective}
        for p in probes:
            p['weighted_gradient_ratio'] = relation_weight * p['relation_gradient'] / p['kd_gradient']
        print('Relation calibration ' + json.dumps(calibration), flush=True)
    provenance = {
        'git_commit_sha': source_commit_sha(),
        'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [Path(__file__), ROOT/'coreset_reactor/relation_distillation.py',
                           MAM_ROOT/'regnn/expert_student.py', MAM_ROOT/'regnn/conditional_model.py']},
        'relation_calibration': calibration,
        'velocity_weight': args.velocity_weight,
        'target_kind': data.manifest.get('target_kind', 'teacher_predictions'),
        'cache_manifest_hash': data.manifest['cache_manifest_hash'],
        'initialization_checkpoint_hash': initialization_hash,
        'initial_student_state_sha256': state_digest.hexdigest(),
        'teacher_checkpoint_hashes': data.manifest['teacher_checkpoint_hashes'],
    }
    (args.run_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    (args.run_dir / 'config.json').write_text(json.dumps({**vars(args),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'relation_weight': relation_weight}, default=str, indent=2))
    started = time.monotonic()
    global_step, history = 0, []
    schedule_digest = hashlib.sha256()
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            schedule_digest.update(json.dumps(batch['sample_id'], separators=(',', ':')).encode())
            inputs = [batch[key].to(device, non_blocking=True) for key in (
                "speaker_audio", "speaker_emotion", "speaker_3dmm", "length",
            )]
            teacher = batch["teacher_predictions"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                output = model.generate(*inputs)
                prediction, mask = output["prediction"], output["valid_mask"]
                flat_prediction = prediction.flatten(0, 1)
                flat_teacher = teacher.flatten(0, 1)
                flat_mask = mask.flatten(0, 1)
                losses = criterion(flat_prediction, flat_teacher, flat_mask)
            if relation_weight:
                rel, student_distance, teacher_distance = distillation_relation(
                    prediction, teacher, mask, relation_scale, args.relation_objective)
                losses['kd_loss'] = losses['loss']
                losses['relation_loss'] = rel
                losses['student_pair_distance'] = student_distance
                losses['teacher_pair_distance'] = teacher_distance
                losses['loss'] = losses['loss'] + relation_weight * rel
            if not torch.isfinite(losses['loss']):
                raise RuntimeError(f'Nonfinite training loss at step {global_step}')
            losses["loss"].backward()
            grad = clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            row = {"epoch": epoch + 1, "step": global_step,
                   **{key: float(value.detach()) for key, value in losses.items()},
                   "gradient_norm": float(grad)}
            history.append(row)
            if global_step == 1 or global_step % args.print_every == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
            if args.max_steps and global_step >= args.max_steps: break
        scheduler.step()
        if args.relation_grad_ratio or args.save_every:
            # Atomic epoch snapshots protect long runs; do not overwrite baseline.
            snapshot = {
                'state_dict': model.state_dict(),
                'student_config': {'backbone': asdict(config.backbone),
                                   'num_experts': config.num_experts,
                                   'expert_embedding_dim': config.expert_embedding_dim},
                'global_step': global_step, 'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'relation_calibration': calibration, 'provenance': provenance,
                'rng_state': {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
                              'python': random.getstate(),
                              'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None},
            }
            temporary = args.run_dir / 'latest_epoch.pth.tmp'
            torch.save(snapshot, temporary)
            os.replace(temporary, args.run_dir / 'latest_epoch.pth')
            if args.save_every and (epoch + 1) % args.save_every == 0:
                atomic_path = args.run_dir / f'student_epoch{epoch + 1:04d}.pth'
                torch.save(snapshot, atomic_path.with_suffix('.pth.tmp'))
                os.replace(atomic_path.with_suffix('.pth.tmp'), atomic_path)
        if args.max_steps and global_step >= args.max_steps: break
    # Deployment checkpoint contains student state and immutable hashes only.
    provenance['observed_source_schedule_sha256'] = schedule_digest.hexdigest()
    (args.run_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    checkpoint = {
        "state_dict": model.state_dict(),
        "student_config": {
            "backbone": asdict(config.backbone),
            "num_experts": config.num_experts,
            "expert_embedding_dim": config.expert_embedding_dim,
        },
        "teacher_checkpoint_hashes": data.manifest["teacher_checkpoint_hashes"],
        "cache_manifest_hash": data.manifest["cache_manifest_hash"],
        "initialization_checkpoint_hash": initialization_hash,
        "global_step": global_step,
        "relation_calibration": calibration,
        "provenance": provenance,
    }
    training_state = {
        **checkpoint,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    temporary = args.run_dir / "student.pth.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, args.run_dir / "student.pth")
    temporary = args.run_dir / "training_state.pth.tmp"
    torch.save(training_state, temporary)
    os.replace(temporary, args.run_dir / "training_state.pth")
    (args.run_dir / "train_metrics.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.run_dir / "config.json").write_text(json.dumps({
        **vars(args), "cache_manifest": str(args.cache_manifest.resolve()),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "relation_weight": relation_weight,
        "training_seconds": time.monotonic() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
    }, default=str, indent=2) + "\n")


if __name__ == "__main__":
    main()
