"""TRAIN-only same-session GT cache and independent source/paired loader."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from coreset_reactor.paired_data import PairedReactionDataset
from coreset_reactor.descriptor import reaction_descriptor, DescriptorScaler


CACHE_VERSION = 2


def full_reaction_descriptor(raw: torch.Tensor, segments: int = 8) -> torch.Tensor:
    """Describe every frame of one raw GT, independently of exact-loss cropping."""
    if raw.ndim != 2 or raw.shape[1] != 25 or len(raw) < 2 * segments:
        raise ValueError("raw GT must be [T,25] with at least two frames per segment")
    if not torch.isfinite(raw).all():
        raise ValueError("raw GT contains nonfinite values")
    return reaction_descriptor(raw, segments)


def fixed_reaction(raw: torch.Tensor, frames: int) -> torch.Tensor:
    """Center-crop long GT; linearly stretch short GT without zero padding."""
    if raw.ndim != 2 or raw.shape[1] != 25 or len(raw) == 0:
        raise ValueError("GT reaction must be nonempty [T,25]")
    if len(raw) >= frames:
        start = (len(raw) - frames) // 2
        return raw[start:start + frames].contiguous()
    return F.interpolate(raw.T[None], size=frames, mode="linear",
                         align_corners=True)[0].T.contiguous()


def build_train_cache(data_root: Path, path: Path, frames: int = 750,
                      segments: int = 8) -> dict:
    """Compute each distinct TRAIN listener descriptor once, not each epoch."""
    facial = data_root / "train" / "facial-attributes"
    paths = sorted((facial / "listener").glob("*/*.npy"))
    if not paths:
        raise FileNotFoundError(f"no TRAIN listener GT under {facial}")
    ids: list[str] = []
    sessions: dict[str, list[int]] = defaultdict(list)
    reactions: list[torch.Tensor] = []
    raw_descriptors: list[torch.Tensor] = []
    for index, source in enumerate(paths):
        relative = source.relative_to(facial).with_suffix("")
        identifier = relative.as_posix()
        raw = torch.from_numpy(np.load(source, allow_pickle=False).astype(np.float32))
        if not torch.isfinite(raw).all():
            raise ValueError(f"nonfinite GT in {source}")
        ids.append(identifier)
        sessions[relative.parts[1]].append(index)
        raw_descriptors.append(full_reaction_descriptor(raw, segments))
        reactions.append(fixed_reaction(raw, frames))
    values = torch.stack(reactions)
    descriptors = torch.stack(raw_descriptors)
    mean = descriptors.mean(0)
    std = descriptors.std(0, unbiased=False).clamp_min(.03)
    normalized = (descriptors - mean) / std
    state = {"version": CACHE_VERSION, "split": "train", "frames": frames,
             "segments": segments, "data_root": str(data_root.resolve()),
             "descriptor_source": "full_raw_listener_gt",
             "ids": ids, "sessions": dict(sessions),
             "reactions": values.half(), "descriptors": normalized.float(),
             "descriptor_mean": mean.float(), "descriptor_std": std.float()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return state


class TrainSetData:
    def __init__(self, data_root: Path, cache_path: Path,
                 frames: int = 750, segments: int = 8, seed: int = 123):
        if cache_path.exists():
            state = torch.load(cache_path, map_location="cpu")
        else:
            state = build_train_cache(data_root, cache_path, frames, segments)
        for key, value in (("version", CACHE_VERSION), ("split", "train"),
                           ("frames", frames), ("segments", segments),
                           ("data_root", str(data_root.resolve())),
                           ("descriptor_source", "full_raw_listener_gt")):
            if state.get(key) != value:
                raise ValueError(f"cache {key} mismatch: {state.get(key)} != {value}")
        self.state = state
        self.id_to_index = {name: index for index, name in enumerate(state["ids"])}
        self.dataset = PairedReactionDataset(
            data_root, "train", frames, crop_mode="random", seed=seed,
            normalization_dir=data_root.parent / "mam_reactor" / "external" / "FaceVerse")
        self.scaler = DescriptorScaler(state["descriptor_mean"], state["descriptor_std"])
        self.frames = frames

    def indices_for(self, sample: dict) -> list[int]:
        session = sample["session_id"]
        indices = self.state["sessions"][session]
        if len(indices) < 90:
            raise ValueError(f"expected 90+ distinct GTs in {session}, got {len(indices)}")
        return indices

    def paired_index_for(self, sample: dict) -> int:
        identifier = sample["clip_id"].replace("speaker/", "listener/", 1)
        return self.id_to_index[identifier]
