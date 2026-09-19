"""M2 preflight: real-GT diversity ceiling and prediction-slot responsibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from coreset_reactor.dataset import fixed_reaction, full_reaction_descriptor
from coreset_reactor.descriptor import DescriptorScaler, reaction_descriptor
from coreset_reactor.losses import descriptor_distance
from coreset_reactor.model import ParallelTCN
from coreset_reactor.paired_data import PairedReactionDataset


ROOT = Path(__file__).resolve().parents[1]
MAM_ROOT = ROOT / "mam_reactor"


def stable_rng(seed: int, name: str, purpose: str) -> random.Random:
    digest = hashlib.sha256(f"{purpose}|{seed}|{name}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def normalized_squared_distances(reactions: torch.Tensor) -> torch.Tensor:
    """Official-FRDiv pairwise terms for [N,T,25] reactions."""
    if reactions.ndim != 3 or reactions.shape[-1] != 25:
        raise ValueError("reactions must have shape [N,T,25]")
    flat = reactions.float().reshape(reactions.shape[0], -1)
    distance = torch.cdist(flat, flat).square() / flat.shape[-1]
    distance.fill_diagonal_(0)
    return distance


def frdiv_from_distances(distance: torch.Tensor,
                         indices: list[int] | None = None) -> float:
    """Mean ordered off-diagonal distance, exactly matching compute_s_mse."""
    if indices is not None:
        chosen = torch.tensor(indices, dtype=torch.long, device=distance.device)
        distance = distance.index_select(0, chosen).index_select(1, chosen)
    count = distance.shape[0]
    if distance.shape != (count, count) or count < 2:
        raise ValueError("distance must be square with at least two entries")
    return float(distance.sum() / (count * (count - 1)))


def kmedoids_indices(distance: torch.Tensor, count: int,
                     max_swaps: int = 20) -> list[int]:
    """Deterministic PAM-style medoids under the official FRDiv distance."""
    distance = distance.detach().cpu()
    size = distance.shape[0]
    if distance.shape != (size, size) or not 1 < count <= size:
        raise ValueError("invalid distance matrix or medoid count")
    medoids = [int(distance.sum(0).argmin())]
    nearest = distance[:, medoids[0]].clone()
    while len(medoids) < count:
        candidates = [index for index in range(size) if index not in medoids]
        candidate_cost = torch.stack([
            torch.minimum(nearest, distance[:, index]).sum()
            for index in candidates
        ])
        chosen = candidates[int(candidate_cost.argmin())]
        medoids.append(chosen)
        nearest = torch.minimum(nearest, distance[:, chosen])

    for _ in range(max_swaps):
        current_cost = float(distance[:, medoids].amin(1).sum())
        best_cost = current_cost
        best_swap: tuple[int, int] | None = None
        outside = [index for index in range(size) if index not in medoids]
        for position in range(count):
            kept = medoids[:position] + medoids[position + 1:]
            kept_nearest = distance[:, kept].amin(1)
            for candidate in outside:
                cost = float(torch.minimum(kept_nearest,
                                           distance[:, candidate]).sum())
                if cost < best_cost - 1e-9:
                    best_cost = cost
                    best_swap = position, candidate
        if best_swap is None:
            break
        medoids[best_swap[0]] = best_swap[1]
    return sorted(medoids)


def farthest_sum_indices(distance: torch.Tensor, count: int,
                         max_swaps: int = 20) -> list[int]:
    """Greedy farthest-sum plus swap refinement for approximate max-FRDiv."""
    distance = distance.detach().cpu()
    size = distance.shape[0]
    if distance.shape != (size, size) or not 1 < count <= size:
        raise ValueError("invalid distance matrix or selection count")
    upper_mask = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    upper = torch.where(upper_mask, distance,
                        torch.full_like(distance, -torch.inf))
    first_flat = int(upper.argmax())
    selected = [first_flat // size, first_flat % size]
    while len(selected) < count:
        candidates = [index for index in range(size) if index not in selected]
        gains = torch.tensor([float(distance[index, selected].sum())
                              for index in candidates])
        selected.append(candidates[int(gains.argmax())])

    def score(indices: list[int]) -> float:
        chosen = distance[indices][:, indices]
        return float(chosen.sum())

    for _ in range(max_swaps):
        current_score = score(selected)
        best_score = current_score
        best_swap: tuple[int, int] | None = None
        outside = [index for index in range(size) if index not in selected]
        for position in range(count):
            kept = selected[:position] + selected[position + 1:]
            base = score(kept)
            for candidate in outside:
                candidate_score = base + 2 * float(distance[candidate, kept].sum())
                if candidate_score > best_score + 1e-9:
                    best_score = candidate_score
                    best_swap = position, candidate
        if best_swap is None:
            break
        selected[best_swap[0]] = best_swap[1]
    return sorted(selected)


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _session_balanced(records: list[dict], key: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record["session_id"]].append(record[key])
    return _mean([_mean(values) for values in grouped.values()])


@torch.inference_mode()
def run_diagnostics(checkpoint_path: Path, data_root: Path, cache_path: Path,
                    device: torch.device, max_examples: int = 571,
                    seed: int = 1234, set_size: int = 10) -> dict:
    dataset = PairedReactionDataset(
        data_root, "val", 750, crop_mode="center",
        normalization_dir=MAM_ROOT / "external" / "FaceVerse")
    selected = sorted(random.Random(seed).sample(
        range(len(dataset)), min(max_examples, len(dataset))))
    if not selected:
        raise ValueError("no VAL contexts selected")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = ParallelTCN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if model.num_predictions != set_size:
        raise ValueError("diagnostic set size must match model num_predictions")

    cache = torch.load(cache_path, map_location="cpu")
    scaler = DescriptorScaler(cache["descriptor_mean"],
                              cache["descriptor_std"]).to(device)
    facial = data_root / "val" / "facial-attributes" / "listener"
    paths_by_session = {
        directory.name: sorted(directory.glob("*.npy"))
        for directory in sorted(facial.iterdir()) if directory.is_dir()
    }

    raw_cache: dict[str, list[torch.Tensor]] = {}
    descriptor_cache: dict[str, torch.Tensor] = {}
    ceiling_cache: dict[tuple[str, int], dict] = {}
    context_records: list[dict] = []
    slot_count = model.num_predictions
    global_responsibility = torch.zeros(slot_count, dtype=torch.long)
    context_responsibilities: list[torch.Tensor] = []
    session_responsibility: dict[str, torch.Tensor] = defaultdict(
        lambda: torch.zeros(slot_count, dtype=torch.long))
    session_gt_total: dict[str, int] = defaultdict(int)
    nonzero_contexts = torch.zeros(slot_count, dtype=torch.long)
    assigned_distance_sum = torch.zeros(slot_count)
    assigned_distance_count = torch.zeros(slot_count, dtype=torch.long)
    pairwise_prediction_sum = torch.zeros(slot_count, slot_count)
    descriptor_centroid_sum = torch.zeros(slot_count, 602)
    au_sum = torch.zeros(slot_count, 15)
    va_mean_sum = torch.zeros(slot_count, 2)
    va_std_sum = torch.zeros(slot_count, 2)
    expression_sum = torch.zeros(slot_count, 8)
    prediction_frdiv: list[float] = []
    start_time = time.perf_counter()

    def session_data(session: str) -> tuple[list[torch.Tensor], torch.Tensor]:
        if session not in raw_cache:
            paths = paths_by_session.get(session, [])
            if len(paths) < set_size:
                raise ValueError(f"session has fewer than {set_size} listener GTs")
            raw = [torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))
                   for path in paths]
            if not all(value.ndim == 2 and value.shape[1] == 25 and
                       torch.isfinite(value).all() for value in raw):
                raise ValueError("invalid real listener GT")
            descriptors = torch.stack([full_reaction_descriptor(value)
                                       for value in raw]).to(device)
            raw_cache[session] = raw
            descriptor_cache[session] = scaler(descriptors).cpu()
        return raw_cache[session], descriptor_cache[session]

    def ceiling_data(session: str, length: int) -> dict:
        key = session, length
        if key not in ceiling_cache:
            raw, _ = session_data(session)
            reactions = torch.stack([fixed_reaction(value, length)
                                     for value in raw]).to(device)
            pairwise = normalized_squared_distances(reactions).cpu()
            medoids = kmedoids_indices(pairwise, set_size)
            farthest = farthest_sum_indices(pairwise, set_size)
            ceiling_cache[key] = {
                "pairwise": pairwise,
                "kmedoids": medoids,
                "farthest_sum_swap": farthest,
                "kmedoids_FRDiv": frdiv_from_distances(pairwise, medoids),
                "farthest_FRDiv": frdiv_from_distances(pairwise, farthest),
                "all_gt_FRDiv": frdiv_from_distances(pairwise),
            }
        return ceiling_cache[key]

    for offset, index in enumerate(selected, start=1):
        sample = dataset[index]
        length = int(sample["source_lengths"])
        session = sample["session_id"]
        raw, gt_descriptors = session_data(session)
        inputs = (sample["speaker_audio"][None, :length].to(device),
                  sample["speaker_emotion"][None, :length].to(device),
                  sample["speaker_3dmm"][None, :length].to(device))
        prediction = model(*inputs)[0]
        prediction_descriptors = scaler(reaction_descriptor(prediction))
        distances = descriptor_distance(
            prediction_descriptors[None], gt_descriptors.to(device)[None])[0]
        assignment = distances.argmin(0).cpu()
        counts = torch.bincount(assignment, minlength=slot_count)
        responsibility = counts.float() / len(raw)
        global_responsibility += counts
        context_responsibilities.append(responsibility)
        session_responsibility[session] += counts
        session_gt_total[session] += len(raw)
        nonzero_contexts += counts.gt(0)
        for slot in range(slot_count):
            mask = assignment == slot
            if mask.any():
                assigned_distance_sum[slot] += float(distances[slot, mask.to(device)].sum())
                assigned_distance_count[slot] += int(mask.sum())

        official = prediction.detach().cpu().clone()
        official[..., :15] = torch.round(official[..., :15])
        prediction_pairwise = normalized_squared_distances(official)
        pairwise_prediction_sum += prediction_pairwise
        current_frdiv = frdiv_from_distances(prediction_pairwise)
        prediction_frdiv.append(current_frdiv)
        descriptor_centroid_sum += prediction_descriptors.detach().cpu()
        au_sum += official[..., :15].mean(1)
        va_mean_sum += prediction.detach().cpu()[..., 15:17].mean(1)
        va_std_sum += prediction.detach().cpu()[..., 15:17].std(1, unbiased=False)
        expression_sum += prediction.detach().cpu()[..., 17:25].mean(1)

        ceiling = ceiling_data(session, length)
        rng = stable_rng(seed, sample["clip_id"], "random-real-gt-10")
        random_indices = rng.sample(range(len(raw)), set_size)
        random_frdiv = frdiv_from_distances(ceiling["pairwise"], random_indices)
        context_records.append({
            "clip_id": sample["clip_id"],
            "session_id": session,
            "source_length": length,
            "real_gt_count": len(raw),
            "prediction_FRDiv": current_frdiv,
            "random_GT10_FRDiv": random_frdiv,
            "kmedoids_GT10_FRDiv": ceiling["kmedoids_FRDiv"],
            "farthest_GT10_FRDiv": ceiling["farthest_FRDiv"],
            "all_GT_FRDiv": ceiling["all_gt_FRDiv"],
            "slot_responsibility": responsibility.tolist(),
            "utilized_slots": int(counts.gt(0).sum()),
        })
        if offset % 25 == 0 or offset == len(selected):
            elapsed = time.perf_counter() - start_time
            print(f"contexts={offset}/{len(selected)} elapsed_s={elapsed:.1f}",
                  flush=True)

    contexts = len(context_records)
    responsibility_matrix = torch.stack(context_responsibilities)
    global_share = global_responsibility.float() / global_responsibility.sum()
    entropy = float(-(global_share.clamp_min(1e-12).log() * global_share).sum())
    centroid = descriptor_centroid_sum / contexts
    centroid_pairwise = torch.cdist(centroid, centroid) / math.sqrt(centroid.shape[-1])
    pairwise_prediction = pairwise_prediction_sum / contexts
    slot_records = []
    for slot in range(slot_count):
        slot_records.append({
            "slot": slot,
            "global_responsibility": float(global_share[slot]),
            "mean_context_responsibility": float(responsibility_matrix[:, slot].mean()),
            "std_context_responsibility": float(responsibility_matrix[:, slot].std(unbiased=False)),
            "nonzero_context_fraction": float(nonzero_contexts[slot] / contexts),
            "mean_assigned_descriptor_distance": (
                float(assigned_distance_sum[slot] / assigned_distance_count[slot])
                if assigned_distance_count[slot] else None),
            "mean_pairwise_FRDiv_to_other_slots": float(
                pairwise_prediction[slot].sum() / (slot_count - 1)),
            "AU_activation_rate": (au_sum[slot] / contexts).tolist(),
            "VA_mean": (va_mean_sum[slot] / contexts).tolist(),
            "VA_temporal_std": (va_std_sum[slot] / contexts).tolist(),
            "expression_distribution": (expression_sum[slot] / contexts).tolist(),
            "dominant_expression": int(expression_sum[slot].argmax()),
            "descriptor_centroid": centroid[slot].tolist(),
        })

    session_records = []
    for session in sorted(session_responsibility):
        share = session_responsibility[session].float() / session_gt_total[session]
        session_records.append({
            "session_id": session,
            "contexts": sum(record["session_id"] == session
                            for record in context_records),
            "slot_responsibility": share.tolist(),
        })

    ceiling_keys = ("random_GT10_FRDiv", "kmedoids_GT10_FRDiv",
                    "farthest_GT10_FRDiv", "all_GT_FRDiv")
    ceiling_summary = {
        key: {
            "context_weighted_mean": _mean([record[key] for record in context_records]),
            "session_balanced_mean": _session_balanced(context_records, key),
            "median": float(np.median([record[key] for record in context_records])),
            "p10": float(np.quantile([record[key] for record in context_records], .1)),
            "p90": float(np.quantile([record[key] for record in context_records], .9)),
        }
        for key in ceiling_keys
    }
    utilized = [record["utilized_slots"] for record in context_records]
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return {
        "protocol": {
            "split": "val",
            "contexts": contexts,
            "sessions": len(session_records),
            "eval_seed": seed,
            "set_size": set_size,
            "gt_temporal_contract": "center crop long GT; linear stretch short GT to source valid length",
            "selection_distance": "official FRDiv normalized squared sequence distance",
            "kmedoids": "deterministic greedy BUILD plus PAM-style swaps",
            "max_frdiv": "deterministic farthest-sum greedy plus swaps; approximate, not combinatorial optimum",
            "responsibility": "hard nearest prediction in TRAIN-scaled 602-D descriptor space",
            "checkpoint_sha256": checkpoint_sha,
        },
        "gt_diversity_ceiling": ceiling_summary,
        "prediction": {
            "FRDiv_context_weighted_mean": _mean(prediction_frdiv),
            "FRDiv_session_balanced_mean": _session_balanced(
                context_records, "prediction_FRDiv"),
            "mean_utilized_slots": _mean(utilized),
            "median_utilized_slots": float(np.median(utilized)),
            "min_utilized_slots": min(utilized),
            "max_utilized_slots": max(utilized),
            "global_responsibility": global_share.tolist(),
            "effective_slots": math.exp(entropy),
            "responsibility_entropy": entropy,
            "pairwise_FRDiv_matrix": pairwise_prediction.tolist(),
            "descriptor_centroid_distance_matrix": centroid_pairwise.tolist(),
            "slots": slot_records,
            "sessions": session_records,
        },
        "per_context": context_records,
        "runtime_seconds": time.perf_counter() - start_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--cache", type=Path,
                        default=ROOT / "coreset_reactor" / "cache" /
                        "train_descriptors_v2.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=571)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--set-size", type=int, default=10)
    parser.add_argument("--device", default="cuda:5")
    args = parser.parse_args()
    result = run_diagnostics(args.checkpoint, args.data_root, args.cache,
                             torch.device(args.device), args.max_examples,
                             args.eval_seed, args.set_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
