"""Build A2 real-reference targets on the frozen 8,820 multi-view schedule."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from coreset_reactor.build_matched_real_gt_cache import real_target
from coreset_reactor.teacher_cache import atomic_torch_save, sha256_file
from regnn.conditional_data import ConditionalReactionDataset


VIEW_KEYS = (
    "sample_index", "sample_id", "view_id", "view_epoch", "start", "length",
    "generated_length",
)


def build(data_root: Path, source_manifest_path: Path,
          fixed_manifests: list[Path], output: Path) -> dict:
    source_manifest_path = source_manifest_path.resolve()
    data_root = data_root.resolve()
    source = json.loads(source_manifest_path.read_text())
    if source["split"] != "train" or source["num_experts"] != 10:
        raise ValueError("Expected a ten-slot REACT2025 TRAIN cache")
    if source["contexts"] != 8820 or source.get("source_contexts") != 3320:
        raise ValueError("Expected the frozen 8,820-context multi-view cache")
    if source.get("crop_view_mode") != "stratified" or source.get("crop_views") != 3:
        raise ValueError("Expected the frozen stratified head/center/tail views")
    if len(fixed_manifests) != 10:
        raise ValueError("Ten fixed-rank manifests are required")

    mappings = []
    manifest_hashes = []
    for path in fixed_manifests:
        path = path.resolve()
        manifest = json.loads(path.read_text())
        if manifest["split"] != "train":
            raise ValueError("GT identity selection must be TRAIN-only")
        mappings.append(manifest["mapping"])
        manifest_hashes.append(sha256_file(path))

    dataset_args = dict(
        clip_length=source["clip_length"], training=True, target_mode="paired",
        seed=source["source_seed"],
        crop_view_mode=source["crop_view_mode"],
        crop_view_count=source["crop_views"],
    )
    dataset = ConditionalReactionDataset(str(data_root), "train", **dataset_args)
    dataset.set_epoch(source["view_epoch"])
    rank_loaders = [
        ConditionalReactionDataset(
            str(data_root), "train", **dataset_args, fixed_pair_rank=rank,
            fixed_pair_count=10, fixed_pair_seed=1,
            fixed_pair_manifest=str(fixed_manifests[rank].resolve()),
            fixed_pair_alignment_policy="relative_time_masked",
        )
        for rank in range(10)
    ]
    for loader in rank_loaders:
        loader.set_epoch(source["view_epoch"])

    output.mkdir(parents=True, exist_ok=False)
    audit_indices = set(torch.linspace(0, source["contexts"] - 1, 30).long().tolist())
    raw_cache: dict[str, torch.Tensor] = {}
    raw_hashes: dict[str, str] = {}
    view_digest = hashlib.sha256()
    completed = parity = 0
    for shard_name in source["shards"]:
        old_rows = torch.load(
            source_manifest_path.parent / shard_name, map_location="cpu",
            weights_only=False,
        )
        new_rows = []
        for row in old_rows:
            index = int(row["sample_index"])
            view_id = int(row["view_id"])
            sample = dataset[(index, view_id)]
            observed = (
                sample["clip_id"], int(sample["start"]), int(sample["length"]),
                int(sample["crop_view_id"]),
            )
            expected = (row["sample_id"], row["start"], row["length"], view_id)
            if observed != expected:
                raise ValueError(f"Frozen multi-view mismatch: {expected} != {observed}")
            source_key = sample["clip_id"] + ".npy"
            selected = [mappings[rank][source_key][rank] for rank in range(10)]
            if len(set(selected)) != 10:
                raise ValueError("The ten fixed target identities must be distinct")
            if any(mappings[rank][source_key] != selected for rank in range(10)):
                raise ValueError("Rank manifests disagree on ordered target identities")

            targets = []
            for rank, relative in enumerate(selected):
                path = dataset.emotion_dir / relative
                if relative not in raw_cache:
                    raw_cache[relative] = dataset._load_tensor(path)
                    raw_hashes[relative] = sha256_file(path)
                target = real_target(
                    raw_cache[relative], rank, row["start"], row["length"],
                    source["clip_length"],
                )
                if completed in audit_indices:
                    check = rank_loaders[rank][(index, view_id)]
                    if int(check["length"]) != row["length"]:
                        raise ValueError("Production target valid length differs")
                    for key in ("speaker_audio", "speaker_emotion", "speaker_3dmm"):
                        torch.testing.assert_close(check[key], sample[key], rtol=0, atol=0)
                    torch.testing.assert_close(check["target"], target, rtol=0, atol=0)
                    parity += 1
                targets.append(target)
            value = torch.stack(targets)
            if value.shape != (10, source["clip_length"], 25):
                raise ValueError("Unexpected real-reference target shape")
            if not torch.isfinite(value).all():
                raise ValueError("Non-finite real-reference target")
            metadata = {key: row[key] for key in VIEW_KEYS}
            view_digest.update(json.dumps(
                [metadata[key] for key in VIEW_KEYS], separators=(",", ":"),
            ).encode())
            new_rows.append({
                **metadata,
                "teacher_seed": None,
                "teacher_predictions": value.half(),
                "target_kind": "real_gt",
                "fixed_gt_ids": selected,
            })
            completed += 1
        atomic_torch_save(new_rows, output / shard_name)
        print(json.dumps({"completed": completed, "total": source["contexts"]}),
              flush=True)
    if completed != source["contexts"] or parity != 300:
        raise ValueError(f"Incomplete build/audit: contexts={completed}, parity={parity}")

    manifest = {
        **source,
        "teacher_checkpoint_hashes": [],
        "teacher_generation": "NONE: fixed real GT; legacy tensor key for trainer compatibility",
        "target_kind": "real_gt",
        "target_alignment": "rank0 native synchronized crop; ranks1-9 relative_time_masked",
        "source_view_manifest_sha256": sha256_file(source_manifest_path),
        "fixed_gt_manifest_sha256": manifest_hashes,
        "real_gt_files_sha256": raw_hashes,
        "view_audit_sha256": view_digest.hexdigest(),
        "target_precision": "FP16 storage, FP32 training like matched KD cache",
        "production_target_parity_examples": parity,
        "builder_sha256": sha256_file(Path(__file__)),
    }
    manifest.pop("cache_manifest_hash", None)
    manifest["cache_manifest_hash"] = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--fixed-manifests", type=Path, nargs=10, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.data_root, args.source_manifest, args.fixed_manifests, args.output)


if __name__ == "__main__":
    main()
