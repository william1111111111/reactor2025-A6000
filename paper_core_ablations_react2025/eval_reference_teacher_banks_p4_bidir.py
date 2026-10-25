"""Evaluate three matched teacher banks on formal REACT2025 VAL1142."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.eval_overlap_save_student import powered_center_overlap
from coreset_reactor.eval_student_window_gap import WINDOW, score, window_starts
from coreset_reactor.evaluate import official_prediction
from coreset_reactor.query0_au_transition import stabilize_query0_au_transitions
from coreset_reactor.eval_multiview_p4_bidir_all_session import (
    BASE,
    METRICS,
    frozen,
    official_tlcc,
    sha256,
    write_json,
)
from coreset_reactor.teacher_cache import load_teacher


ROOT = Path(__file__).resolve().parents[1]
BANKS = ("fixed", "uniform", "target_specific")


def prepare(output: Path, bank_audit_path: Path) -> None:
    if (output / "manifest.json").exists():
        raise FileExistsError("Use the existing manifest or a fresh output")
    audit = json.loads(bank_audit_path.read_text())
    if audit["status"] != "passed" or set(audit["banks"]) != set(BANKS):
        raise ValueError("Reference teacher-bank audit is incomplete")
    base_path = BASE / "manifest.json"
    base = json.loads(base_path.read_text())
    entries = base["entries"]
    if len(entries) != 1142 or sum(row["role"] == "speaker" for row in entries) != 571:
        raise ValueError("Expected balanced bidirectional VAL1142")
    banks = {}
    for name in BANKS:
        value = audit["banks"][name]
        paths = [Path(path) for path in value["paths"]]
        if len(paths) != 10:
            raise ValueError(f"{name} does not contain ten teachers")
        frozen_paths = [frozen(path) for path in paths]
        if [item[1] for item in frozen_paths] != value["checkpoint_hashes"]:
            raise ValueError(f"{name} checkpoint hashes differ from audit")
        banks[name] = {
            "checkpoints": frozen_paths,
            "bank_hash": value["bank_hash"],
        }
    if len({banks[name]["checkpoints"][0][1] for name in BANKS}) != 1:
        raise ValueError("The three teacher banks must share query 0")
    implementations = [
        Path(__file__).resolve(),
        ROOT / "coreset_reactor/eval_multiview_p4_bidir_all_session.py",
        ROOT / "coreset_reactor/eval_overlap_save_student.py",
        ROOT / "coreset_reactor/eval_student_window_gap.py",
        ROOT / "coreset_reactor/evaluate.py",
        ROOT / "coreset_reactor/query0_au_transition.py",
        ROOT / "coreset_reactor/exact_dtw_cuda.py",
        ROOT / "coreset_reactor/fast_frc.py",
        ROOT / "coreset_reactor/teacher_cache.py",
    ]
    official = ROOT / "mam_reactor/framework/metrics"
    official_sources = [
        official / "FRC.py", official / "FRD.py", official / "S_MSE.py",
        official / "FRVar.py", official / "TLCC.py",
        ROOT / "mam_reactor/framework/modules/post_processor.py",
    ]
    manifest = {
        "schema_version": 1,
        "protocol": (
            "REACT2025 full VAL1142, both directions, full sequences, K10, all "
            "opposite-role same-session GT, official Processor/AU rounding/five "
            "metrics; raw and fixed query-0-transition outputs; overlap p=4"
        ),
        "only_official_protocol_extension": "all same-session GT",
        "bank_audit": frozen(bank_audit_path),
        "base_manifest": frozen(base_path),
        "banks": banks,
        "implementations": [frozen(path) for path in implementations],
        "official_sources": [frozen(path) for path in official_sources],
        "window": WINDOW, "stride": 375, "power": 4.0, "tf32": False,
        "entries": entries,
    }
    write_json(output / "manifest.json", manifest)
    (output / "rows").mkdir(exist_ok=True)


def generate_unique_teacher_windows(models, streams, starts, valid_lengths,
                                    device, batch_size=8):
    padded = []
    for stream in streams:
        items = []
        for start, valid in zip(starts, valid_lengths):
            item = stream[start:start + valid]
            if valid < WINDOW:
                item = torch.nn.functional.pad(item, (0, 0, 0, WINDOW - valid))
            items.append(item)
        padded.append(torch.stack(items))
    lengths = torch.tensor(valid_lengths, dtype=torch.long)
    outputs = {digest: [] for digest in models}
    for begin in range(0, len(starts), batch_size):
        batch = [item[begin:begin + batch_size].to(device) for item in padded]
        batch_lengths = lengths[begin:begin + batch_size].to(device)
        for digest, model in models.items():
            prediction = model(*batch, batch_lengths)["prediction"]
            outputs[digest].append(prediction.detach().float().cpu())
    result = {digest: torch.cat(chunks) for digest, chunks in outputs.items()}
    expected = (len(starts), WINDOW, 25)
    if any(tuple(value.shape) != expected for value in result.values()):
        raise ValueError("Unexpected teacher window shape")
    return result


def grouped(rows: list[dict], bank: str, kind: str) -> dict:
    def one(selected):
        return {
            "contexts": len(selected),
            **{metric: float(np.mean([
                row["banks"][bank][f"{kind}_metrics"][metric] for row in selected
            ])) for metric in METRICS},
        }
    return {
        "speaker_to_listener": one([row for row in rows if row["role"] == "speaker"]),
        "listener_to_speaker": one([row for row in rows if row["role"] == "listener"]),
        "both_directions": one(rows),
    }


def aggregate(output: Path, manifest: dict) -> None:
    rows = [json.loads((output / "rows" / f"{entry['index']:04d}.json").read_text())
            for entry in manifest["entries"]]
    if [row["clip_id"] for row in rows] != [entry["clip_id"] for entry in manifest["entries"]]:
        raise ValueError("Teacher evaluation row order differs from manifest")
    summary = {}
    for bank in BANKS:
        summary[bank] = {}
        for kind in ("raw", "postprocessed"):
            groups = grouped(rows, bank, kind)
            normalized_rows = [{
                "index": row["index"], "clip_id": row["clip_id"],
                "role": row["role"], "length": row["length"],
                "gt_count": row["gt_count"],
                "metrics": row["banks"][bank][f"{kind}_metrics"],
                "postprocess_changed": row["banks"][bank]["postprocess_changed"],
                "seconds_shared_three_banks": row["seconds"],
            } for row in rows]
            payload = {
                "protocol": manifest["protocol"], "bank": bank,
                "bank_hash": manifest["banks"][bank]["bank_hash"],
                "output_kind": kind, "metrics": groups,
                "manifest_sha256": sha256(output / "manifest.json"),
                "rows": normalized_rows,
            }
            write_json(output / f"comparison_{bank}_{kind}.json", payload)
            summary[bank][kind] = groups
    write_json(output / "comparison.json", {
        "protocol": manifest["protocol"],
        "manifest_sha256": sha256(output / "manifest.json"),
        "metrics": summary,
        "changed_contexts": {
            bank: sum(row["banks"][bank]["postprocess_changed"] for row in rows)
            for bank in BANKS
        },
        "shared_evaluation_seconds": float(sum(row["seconds"] for row in rows)),
    })
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank-audit", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--max-contexts", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        if args.bank_audit is None:
            raise ValueError("--bank-audit is required with --prepare")
        prepare(args.output, args.bank_audit)
        return
    manifest = json.loads((args.output / "manifest.json").read_text())
    if args.aggregate:
        aggregate(args.output, manifest)
        return
    for path, digest in [manifest["bank_audit"], manifest["base_manifest"]]:
        if sha256(Path(path)) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    for value in manifest["banks"].values():
        for path, digest in value["checkpoints"]:
            if sha256(Path(path)) != digest:
                raise ValueError(f"Frozen checkpoint changed: {path}")
    for key in ("implementations", "official_sources"):
        for path, digest in manifest[key]:
            if sha256(Path(path)) != digest:
                raise ValueError(f"Frozen source changed: {path}")
    if not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard")

    torch.set_num_threads(2)
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    unique = {}
    for value in manifest["banks"].values():
        for path, digest in value["checkpoints"]:
            if digest not in unique:
                unique[digest] = load_teacher(Path(path), device)[0]
    bank_hashes = {
        name: [digest for _, digest in manifest["banks"][name]["checkpoints"]]
        for name in BANKS
    }
    mean = torch.from_numpy(np.load(
        ROOT / "mam_reactor/external/FaceVerse/mean_face.npy"
    ).astype("float32")).reshape(58)
    std = torch.from_numpy(np.load(
        ROOT / "mam_reactor/external/FaceVerse/std_face.npy"
    ).astype("float32")).reshape(58).clamp_min(1e-8)
    provenance = {
        "manifest_sha256": sha256(args.output / "manifest.json"),
        "shard": args.shard, "shards": args.shards,
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "seed": 1234, "tf32": False,
        "unique_teacher_models": len(unique),
    }
    provenance_path = args.output / f"provenance_{args.shard}.json"
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("Teacher evaluation provenance mismatch")
    else:
        write_json(provenance_path, provenance)

    entries = sorted(manifest["entries"], key=lambda item: (-item["length"], item["clip_id"]))
    schedule = entries[args.shard::args.shards]
    completed = 0
    started = time.monotonic()
    with torch.inference_mode():
        for entry in schedule:
            row_path = args.output / "rows" / f"{entry['index']:04d}.json"
            if row_path.exists():
                continue
            tick = time.monotonic()
            stored = torch.load(BASE / "arrays" / f"{entry['index']:04d}.pt",
                                map_location="cpu", weights_only=False)
            if stored["clip_id"] != entry["clip_id"]:
                raise ValueError("Base tensor clip mismatch")
            target = stored["targets"].float().contiguous()
            length = entry["length"]
            streams = [torch.from_numpy(np.load(path).astype("float32")).reshape(length, -1)
                       for path, _ in entry["source_files"]]
            source_face = streams[1]
            streams[2] = (streams[2] - mean) / std
            starts = window_starts(length)
            valid_lengths = [min(WINDOW, length - start) for start in starts]
            window_outputs = generate_unique_teacher_windows(
                unique, streams, starts, valid_lengths, device)
            bank_results = {}
            for bank in BANKS:
                windows = torch.stack([window_outputs[digest]
                                       for digest in bank_hashes[bank]], dim=1)
                prediction = official_prediction(powered_center_overlap(
                    windows, starts, length, manifest["power"]))
                raw_metrics = score(prediction, target)
                raw_metrics["TLCC"] = official_tlcc(prediction, source_face)
                raw_metrics = {metric: raw_metrics[metric] for metric in METRICS}
                postprocessed = stabilize_query0_au_transitions(prediction, source_face)
                changed = not torch.equal(postprocessed, prediction)
                if changed:
                    post_metrics = score(postprocessed, target)
                    post_metrics["TLCC"] = official_tlcc(postprocessed, source_face)
                    post_metrics = {metric: post_metrics[metric] for metric in METRICS}
                else:
                    post_metrics = dict(raw_metrics)
                bank_results[bank] = {
                    "raw_metrics": raw_metrics,
                    "postprocessed_metrics": post_metrics,
                    "postprocess_changed": changed,
                }
            if len({value["postprocess_changed"] for value in bank_results.values()}) != 1:
                raise ValueError("Shared query-0 postprocess changed different bank populations")
            row = {
                "index": entry["index"], "clip_id": entry["clip_id"],
                "role": entry["role"], "length": length,
                "gt_count": int(target.shape[0]), "banks": bank_results,
                "seconds": time.monotonic() - tick,
            }
            write_json(row_path, row)
            completed += 1
            print(json.dumps({
                "index": entry["index"], "role": entry["role"],
                "frames": length, "completed": completed,
                "total_shard": len(schedule), "seconds": row["seconds"],
                "elapsed_s": time.monotonic() - started,
            }), flush=True)
            if args.max_contexts and completed >= args.max_contexts:
                break


if __name__ == "__main__":
    main()
