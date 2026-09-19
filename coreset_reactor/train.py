"""Milestone-1 B0/B1/B3 controlled comparison; no Mam-Reactor model reuse."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from coreset_reactor.dataset import TrainSetData
from coreset_reactor.descriptor import reaction_descriptor
from coreset_reactor.evaluate import evaluate_model
from coreset_reactor.losses import (descriptor_distance, descriptor_set_loss,
                                   paired_cost, softmin, unpaired_softdtw_cost)
from coreset_reactor.model import ParallelTCN
from coreset_reactor.sampler import exact_reference_indices


ROOT = Path(__file__).resolve().parents[1]


def descriptor_weight(step: int, steps: int, target: float,
                      schedule: dict | None = None) -> float:
    """Return a constant weight or a zero/warmup/linear-ramp schedule."""
    if schedule is None:
        return target
    if schedule.get("kind") != "linear":
        raise ValueError("unsupported descriptor schedule")
    start = steps * schedule["warmup_fraction"]
    end = steps * schedule["ramp_end_fraction"]
    progress = min(1.0, max(0.0, (step - start) / (end - start)))
    return target * progress


def source_provenance() -> dict:
    """Record a source fingerprint, and a commit only for tracked clean code."""
    files = sorted((ROOT / "coreset_reactor").glob("*.py"))
    files += sorted((ROOT / "mam_reactor").rglob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    git_head_sha = None
    git_commit_sha = None
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True)
        git_head_sha = head.stdout.strip()
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "coreset_reactor/train.py"],
            cwd=ROOT, capture_output=True).returncode == 0
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "coreset_reactor", "mam_reactor"],
            cwd=ROOT, capture_output=True, text=True, check=True)
        if tracked and not status.stdout.strip():
            git_commit_sha = git_head_sha
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"git_head_sha": git_head_sha, "git_commit_sha": git_commit_sha,
            "source_snapshot_sha256": digest.hexdigest(),
            "source_file_count": len(files)}


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fixed_schedule(data: TrainSetData, steps: int, batch_size: int,
                   seed: int) -> list[tuple[str, list[int]]]:
    by_session: dict[str, list[int]] = {}
    for index, path in enumerate(data.dataset.records):
        by_session.setdefault(path.parent.name, []).append(index)
    sessions = sorted(by_session)
    weights = [len(by_session[s]) for s in sessions]
    rng = random.Random(seed)
    return [(session, [rng.choice(by_session[session]) for _ in range(batch_size)])
            for session in rng.choices(sessions, weights=weights, k=steps)]


def train_one(arm: str, config: dict, data: TrainSetData,
              schedule: list[tuple[str, list[int]]],
              initial: dict[str, torch.Tensor], device: torch.device,
              reports: Path) -> tuple[dict, dict]:
    """Same architecture/init/schedule/optimizer; only GT policy varies."""
    seed = config["seed"]
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    model_config = config["model"]
    model = ParallelTCN(**model_config).to(device)
    model.load_state_dict(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scaler = data.scaler.to(device)
    target_count = 10 if arm == "B0" else 18
    use_all = arm == "B3"
    report = reports / f"m1_{arm}_seed{seed}"
    report.mkdir(parents=True, exist_ok=True)
    save_json(report / "config.json", {**config, "arm": arm,
                                       "exact_targets_this_arm": target_count,
                                       "all_gt_descriptor_loss": use_all})
    usage: Counter[int] = Counter()
    times: list[float] = []
    rows: list[dict] = []
    state = data.state
    model.train()
    for step, (session, indices) in enumerate(schedule, 1):
        data.dataset.set_epoch(step - 1)
        samples = [data.dataset[index] for index in indices]
        batch = default_collate(samples)
        session_indices = state["sessions"][session]
        sampled_ids = []
        for sample in samples:
            paired = data.paired_index_for(sample)
            selected = exact_reference_indices(session_indices, paired,
                                               target_count, seed, step,
                                               sample["clip_id"])
            sampled_ids.append(selected)
            usage[paired] += 1
            usage.update(selected)
        target = state["reactions"][torch.tensor(sampled_ids)].float().to(device)
        inputs = (batch["speaker_audio"].to(device),
                  batch["speaker_emotion"].to(device),
                  batch["speaker_3dmm"].to(device))
        source_lengths = batch["source_lengths"]
        paired = batch["paired_target"].to(device)
        pair_lengths = batch["pair_lengths"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        pred = model(*inputs, lengths=source_lengths)
        aligned = softmin(paired_cost(pred, paired, pair_lengths), 1,
                          config["softmin_temperature"]).mean()
        unaligned_cost = unpaired_softdtw_cost(
            pred, target, frames=config["dtw_frames"],
            gamma=config["dtw_gamma"], band_ratio=config["dtw_band_ratio"],
            prediction_lengths=source_lengths)
        unaligned = softmin(unaligned_cost, 1,
                            config["softmin_temperature"]).mean()
        descriptor_terms = None
        weight = (descriptor_weight(step, len(schedule),
                                    config["loss"]["descriptor_cover"],
                                    config.get("descriptor_schedule"))
                  if use_all else 0.0)
        total = aligned + config["loss"]["unaligned_sequence"] * unaligned
        if use_all:
            gt_desc = state["descriptors"][session_indices].to(device)
            gt_desc = gt_desc[None].expand(len(samples), -1, -1)
            pred_desc = scaler(torch.cat([
                reaction_descriptor(pred[index:index + 1, :, :int(length)],
                                    config["descriptor_segments"])
                for index, length in enumerate(source_lengths.tolist())
            ], dim=0))
            distance = descriptor_distance(pred_desc, gt_desc)
            descriptor_terms = descriptor_set_loss(
                distance, config["softmin_temperature"],
                config["loss"]["descriptor_valid"],
                config["loss"]["load_balance"],
                config["loss"]["max_prediction_usage"])
            total = total + weight * descriptor_terms["total"]
        if not torch.isfinite(total):
            raise FloatingPointError(f"nonfinite loss for {arm} step {step}")
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config["grad_clip_norm"])
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"nonfinite gradient for {arm} step {step}")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        row = {"step": step, "loss": float(total.detach()),
               "paired": float(aligned.detach()),
               "unaligned": float(unaligned.detach()),
               "cover": float(descriptor_terms["cover"].detach()) if descriptor_terms else 0.0,
               "valid": float(descriptor_terms["valid"].detach()) if descriptor_terms else 0.0,
               "load": float(descriptor_terms["load"].detach()) if descriptor_terms else 0.0,
               "descriptor_weight": weight,
               "gradient_norm": float(grad_norm), "iteration_s": elapsed}
        rows.append(row)
        if step == 1 or step % 10 == 0 or step == len(schedule):
            print(f"{arm} {step}/{len(schedule)} loss={row['loss']:.4f} "
                  f"step_s={elapsed:.3f}", flush=True)
        if step in config.get("save_checkpoint_steps", ()):
            torch.save({"arm": arm, "seed": seed, "steps": step,
                        "model_config": model_config,
                        "model": {name: value.detach().cpu().clone()
                                  for name, value in model.state_dict().items()},
                        "provenance": config["provenance"]},
                       report / f"checkpoint_step{step:06d}.pt")
    with (report / "training_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checkpoint = {"arm": arm, "seed": seed, "steps": len(schedule),
                  "model_config": model_config, "model": model.cpu().state_dict(),
                  "optimizer": optimizer.state_dict(), "sampler_usage": dict(usage),
                  "descriptor_cache_path": str(config["cache_path"])}
    torch.save(checkpoint, report / "final_checkpoint.pt")
    model.to(device)
    frequencies = sorted(usage.items(), key=lambda item: (-item[1], item[0]))
    sampled_counts = list(usage.values())
    sampler_stats = {"policy": "paired + deterministic random distinct same-session GT",
                     "paired_draws": len(schedule) * len(schedule[0][1]),
                     "random_draws": len(schedule) * len(schedule[0][1]) * (target_count - 1),
                     "unique_gt_used": len(usage),
                     "usage_min_selected": min(sampled_counts),
                     "usage_max": max(sampled_counts),
                     "usage_mean_selected": float(np.mean(sampled_counts)),
                     "most_selected": [{"id": state["ids"][i], "count": n}
                                       for i, n in frequencies[:10]],
                     "least_selected": [{"id": state["ids"][i], "count": n}
                                        for i, n in frequencies[-10:]],
                     "hardness_distribution": "not active in milestone 1",
                     "cluster_selection_distribution": "not active in milestone 1"}
    save_json(report / "sampler_stats.json", sampler_stats)
    metrics, inference_speed = evaluate_model(
        model, Path(config["data_root"]), scaler, device,
        config["eval_examples"], seed=config["eval_seed"], metric_workers=1)
    save_json(report / "metrics.json", metrics)
    warm = times[1:] if len(times) > 1 else times
    speed = {**inference_speed,
             "training_iteration_s_mean": float(np.mean(warm)),
             "training_samples_per_s": float(config["batch_size"] / np.mean(warm)),
             "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))
                                      if device.type == "cuda" else None,
             "training_steps": len(schedule), "batch_size": config["batch_size"]}
    save_json(report / "speed.json", speed)
    (report / "final_summary.md").write_text(
        f"# {arm}: CoReSet-Reactor Milestone 1\n\n"
        f"VAL pilot contexts: {metrics['contexts']}; exact GT policy: paired + 9 same-session.\n\n"
        f"FRC {metrics['FRC']:.6f}; exact FRD {metrics['exact_FRD']:.6f}; "
        f"FRDiv {metrics['FRDiv']:.6f}; FRVar {metrics['FRVar']:.6f}.\n\n"
        f"Full-session GT descriptor coverage {metrics['gt_descriptor_coverage']:.6f}; "
        f"prediction validity {metrics['prediction_validity']:.6f}; "
        f"candidate utilization {metrics['candidate_utilization']:.3f}.\n\n"
        "This VAL pilot is not an official full-TEST score. Same-session GTs are weak "
        "proxies, not verified same-stimulus reactions.\n")
    return metrics, speed


def milestone1_criteria(b1: dict, b3: dict) -> dict[str, bool]:
    """Zero-slack quality constraints with a strictly positive diversity gain."""
    return {"coverage_improves": b3["gt_descriptor_coverage"] < b1["gt_descriptor_coverage"],
            "FRC_non_degraded": b3["FRC"] >= b1["FRC"],
            "exact_FRD_non_degraded": b3["exact_FRD"] <= b1["exact_FRD"],
            "FRDiv_improves": b3["FRDiv"] > b1["FRDiv"]}


def comparison_report(results: dict[str, tuple[dict, dict]], config: dict,
                      output: Path) -> None:
    lines = ["# CoReSet-Reactor Milestone 1", "",
             "## Research question", "",
             "Does all-seen descriptor coverage beat random 10/18-GT training "
             "under the same TCN, optimizer, initialization, source schedule, and seed?", "",
             f"TRAIN steps per arm: {config['steps']}; batch: {config['batch_size']}; "
             f"VAL pilot contexts: {config['eval_examples']}.", "",
             "| Arm | FRC ↑ | exact FRD ↓ | FRDiv ↑ | FRVar ↑ | GT coverage ↓ | validity ↓ | utilization ↑ | cluster coverage ↑ | train samples/s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("B0", "B1", "B3"):
        m, s = results[arm]
        lines.append(f"| {arm} | {m['FRC']:.6f} | {m['exact_FRD']:.6f} | "
                     f"{m['FRDiv']:.6f} | {m['FRVar']:.6f} | "
                     f"{m['gt_descriptor_coverage']:.6f} | {m['prediction_validity']:.6f} | "
                     f"{m['candidate_utilization']:.3f} | {m['cluster_coverage']:.3f} | "
                     f"{s['training_samples_per_s']:.3f} |")
    b1, b3 = results["B1"][0], results["B3"][0]
    criteria = milestone1_criteria(b1, b3)
    verdict = ("preliminary support" if all(criteria.values()) else
               "not supported on all four coverage/quality/diversity criteria")
    lines += ["", "## Interpretation", "",
              f"B3 vs B1: ΔFRC={b3['FRC']-b1['FRC']:+.6f}, "
              f"Δexact FRD={b3['exact_FRD']-b1['exact_FRD']:+.6f}, "
              f"ΔFRDiv={b3['FRDiv']-b1['FRDiv']:+.6f}, "
              f"ΔGT coverage={b3['gt_descriptor_coverage']-b1['gt_descriptor_coverage']:+.6f}.",
              "Zero-slack criteria: " + ", ".join(
                  f"{name}={'pass' if passed else 'fail'}"
                  for name, passed in criteria.items()) + ".",
              f"Milestone-1 hypothesis: {verdict}. This is a small development pilot, "
              "not a significance claim or permission to enter Milestone 2.", "",
              "## Boundaries and next step", "",
              "The 90+ targets are same-session recordings, not verified reactions to the "
              "same stimulus. TRAIN descriptors are cached once per distinct listener "
              "recording; full raw GTs are used for descriptors, while sampled fixed-length "
              "GTs are used for unpaired temporal matching. Valid source prefixes are used "
              "for prediction losses and metrics; official metrics round prediction AUs. No explicit "
              "prediction-separation loss was used. Official metric formulas and "
              "target post-processor were called on a deterministic VAL subset; this "
              "is not the full official TEST protocol. Cluster coverage is a "
              "non-official 12-medoid neighborhood diagnostic.", "",
              "Before Milestone 2, repeat with more steps, multiple seeds and a larger "
              "frozen VAL set; inspect descriptor scaling and collapse if B3 fails.\n"]
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "coreset_reactor" / "configs" / "baseline.json")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-examples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save-checkpoint-steps", nargs="+", type=int)
    parser.add_argument("--arm", choices=("B0", "B1", "B3"))
    parser.add_argument("--descriptor-cover", type=float)
    parser.add_argument("--descriptor-warmup-fraction", type=float)
    parser.add_argument("--descriptor-ramp-end-fraction", type=float)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--run-name", default="m1_initial")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.steps is not None:
        config["steps"] = args.steps
    if args.eval_examples is not None:
        config["eval_examples"] = args.eval_examples
    if args.seed is not None:
        config["seed"] = args.seed
    if args.descriptor_cover is not None:
        if args.descriptor_cover < 0:
            raise ValueError("descriptor-cover must be nonnegative")
        config["loss"]["descriptor_cover"] = args.descriptor_cover
    if ((args.descriptor_warmup_fraction is None) !=
            (args.descriptor_ramp_end_fraction is None)):
        raise ValueError("both descriptor schedule fractions are required")
    if args.descriptor_warmup_fraction is not None:
        start = args.descriptor_warmup_fraction
        end = args.descriptor_ramp_end_fraction
        if not 0 <= start < end <= 1:
            raise ValueError("descriptor schedule requires 0 <= warmup < ramp end <= 1")
        config["descriptor_schedule"] = {"kind": "linear",
                                         "warmup_fraction": start,
                                         "ramp_end_fraction": end}
    if config["steps"] < 1 or config["eval_examples"] < 1:
        raise ValueError("steps and eval examples must be positive")
    if args.save_checkpoint_steps is not None:
        points = sorted(set(args.save_checkpoint_steps))
        if any(point < 1 or point >= config["steps"] for point in points):
            raise ValueError("intermediate checkpoint steps must be within the run")
        config["save_checkpoint_steps"] = points
    config["data_root"] = str(ROOT / "data")
    config["cache_path"] = str(ROOT / "coreset_reactor" / "cache" / "train_descriptors_v2.pt")
    config["device"] = args.device
    config["provenance"] = source_provenance()
    torch.set_num_threads(4)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
    data = TrainSetData(Path(config["data_root"]), Path(config["cache_path"]),
                        frames=config["model_frames"],
                        segments=config["descriptor_segments"], seed=seed)
    schedule = fixed_schedule(data, config["steps"], config["batch_size"], seed)
    initial = copy.deepcopy(ParallelTCN(**config["model"]).state_dict())
    config["schedule_sha256"] = hashlib.sha256(
        json.dumps(schedule, separators=(",", ":")).encode()).hexdigest()
    initial_digest = hashlib.sha256()
    for name, value in sorted(initial.items()):
        initial_digest.update(name.encode())
        initial_digest.update(str((tuple(value.shape), value.dtype)).encode())
        initial_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    config["initial_model_sha256"] = initial_digest.hexdigest()
    reports = ROOT / "coreset_reactor" / "reports" / args.run_name
    reports.mkdir(parents=True, exist_ok=False)
    save_json(reports / "config.json", config)
    results = {}
    arms = (args.arm,) if args.arm else ("B0", "B1", "B3")
    for arm in arms:
        results[arm] = train_one(arm, config, data, schedule, initial,
                                 device, reports)
    if args.arm is None:
        comparison_report(results, config, reports / "final_summary.md")
        print(f"Milestone 1 report: {reports / 'final_summary.md'}", flush=True)
    else:
        print(f"Single-arm report: {reports / f'm1_{args.arm}_seed{seed}' / 'final_summary.md'}",
              flush=True)


if __name__ == "__main__":
    main()
