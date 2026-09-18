import hashlib
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from regnn.query_mamba import (
    STYLE_DESCRIPTOR_DIM,
    STYLE_DESCRIPTOR_VERSION,
)


def crop_and_pad(
    value: torch.Tensor,
    start: int,
    length: int,
) -> torch.Tensor:
    value = value[start:start + length]
    if value.shape[0] < length:
        padding = value.new_zeros(length - value.shape[0], value.shape[1])
        value = torch.cat((value, padding), dim=0)
    return value


class ConditionalReactionDataset(Dataset):
    """Efficient diffusion-condition dataset for Conditional-REGNN.

    Unlike ``ReactionDataset``, this loader never opens video files. It reads
    only the three speaker conditions consumed by diffusion and the listener
    25-D target.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        clip_length: int = 750,
        training: bool = True,
        target_mode: str = "paired",
        num_targets: int = 10,
        crop_mode: str = "center",
        style_cache_path: Optional[str] = None,
        seed: int = 1,
        target_selection_mode: str = "deterministic_epoch_shuffle",
        session_allowlist: Optional[Sequence[str]] = None,
        crop_stride: int = 1,
        load_listener_3dmm: bool = False,
    ):
        super().__init__()
        if clip_length <= 0:
            raise ValueError("clip_length must be positive")
        if target_mode not in {
            "paired",
            "session",
            "session_style",
            "session_cropped",
        }:
            raise ValueError(
                "target_mode must be paired, session, session_style, or "
                "session_cropped"
            )
        if crop_mode not in {"start", "center"}:
            raise ValueError("crop_mode must be start or center")
        if target_selection_mode not in {
            "deterministic_epoch_shuffle",
            "fixed_first",
        }:
            raise ValueError(
                "target_selection_mode must be "
                "deterministic_epoch_shuffle or fixed_first"
            )

        self.root_dir = Path(root_dir)
        self.split = split
        self.data_dir = self.root_dir / split
        self.audio_dir = self.data_dir / "audio-features"
        self.emotion_dir = self.data_dir / "facial-attributes"
        self.coefficient_dir = self.data_dir / "coefficients"
        self.clip_length = int(clip_length)
        self.training = bool(training)
        self.target_mode = target_mode
        self.num_targets = int(num_targets)
        self.crop_mode = crop_mode
        self.seed = int(seed)
        self.target_selection_mode = target_selection_mode
        self.crop_stride = int(crop_stride)
        self.load_listener_3dmm = load_listener_3dmm
        if load_listener_3dmm and target_mode not in {"paired", "session_cropped"}:
            raise ValueError("Listener 3DMM requires paired or session_cropped targets")
        if self.crop_stride <= 0:
            raise ValueError("crop_stride must be positive")
        self.session_allowlist = (
            tuple(sorted(set(session_allowlist)))
            if session_allowlist is not None
            else None
        )
        if self.session_allowlist is not None and not self.session_allowlist:
            raise ValueError("session_allowlist cannot be empty")
        # A shared tensor keeps persistent DataLoader workers synchronized
        # when the parent process advances the epoch.
        self._epoch = torch.zeros((), dtype=torch.long).share_memory_()
        self.style_descriptors: Optional[Dict[str, torch.Tensor]] = None
        if target_mode == "session_style":
            if style_cache_path is None:
                raise ValueError(
                    "session_style requires style_cache_path"
                )
            style_cache = torch.load(
                Path(style_cache_path),
                map_location="cpu",
            )
            if (
                int(style_cache["version"]) != STYLE_DESCRIPTOR_VERSION
                or int(style_cache["descriptor_dim"])
                != STYLE_DESCRIPTOR_DIM
                or style_cache["split"] != split
            ):
                raise ValueError("Style cache metadata does not match dataset")
            self.style_descriptors = {
                path: descriptor
                for path, descriptor in zip(
                    style_cache["paths"],
                    style_cache["descriptors"],
                )
            }

        package_root = os.environ.get("MAM_REACTOR_ROOT")
        faceverse_roots = []
        if package_root:
            faceverse_roots.append(Path(package_root) / "external" / "FaceVerse")
        faceverse_roots.append(self.root_dir.parent / "external" / "FaceVerse")
        faceverse_root = next(
            (path for path in faceverse_roots if (path / "mean_face.npy").is_file()),
            faceverse_roots[0],
        )
        mean_path = faceverse_root / "mean_face.npy"
        std_path = faceverse_root / "std_face.npy"
        if not mean_path.is_file() or not std_path.is_file():
            raise FileNotFoundError(
                "FaceVerse mean/std are required for diffusion-compatible "
                "3DMM normalization"
            )
        self.mean_face = torch.from_numpy(
            np.load(mean_path).astype(np.float32)
        )
        self.std_face = torch.from_numpy(
            np.load(std_path).astype(np.float32)
        ).clamp_min(1e-8)

        records = sorted(
            path.relative_to(self.emotion_dir)
            for path in self.emotion_dir.glob("*/*/*.npy")
        )
        if self.session_allowlist is not None:
            allowed_sessions = set(self.session_allowlist)
            records = [
                record
                for record in records
                if record.parts[1] in allowed_sessions
            ]
        if not records:
            raise RuntimeError(
                "No facial-attribute files found for the requested split "
                f"under {self.emotion_dir}"
            )
        self.records = records
        self.session_targets: Dict[tuple, List[Path]] = {}
        for record in records:
            key = (record.parts[0], record.parts[1])
            self.session_targets.setdefault(key, []).append(record)
        self.target_pool_pair_count = sum(
            1 + len(self._session_alternatives(record))
            for record in self.records
        )

    def __len__(self) -> int:
        return len(self.records)

    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic but epoch-dependent same-session subset."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch.fill_(int(epoch))

    @staticmethod
    def _opposite_role(role: str) -> str:
        if role == "speaker":
            return "listener"
        if role == "listener":
            return "speaker"
        raise ValueError(f"Unknown dyadic role: {role}")

    def _paired_target(self, record: Path) -> Path:
        return Path(
            self._opposite_role(record.parts[0]),
            record.parts[1],
            record.name,
        )

    def _session_alternatives(self, record: Path) -> List[Path]:
        paired = self._paired_target(record)
        candidates = self.session_targets[
            (paired.parts[0], paired.parts[1])
        ]
        return [path for path in candidates if path != paired]

    def _stable_random(
        self,
        record: Path,
        purpose: str,
        include_epoch: bool,
        view_id: int = 0,
    ) -> random.Random:
        epoch = self.epoch if include_epoch and self.training else 0
        payload = (
            f"{self.seed}|{epoch}|{record.as_posix()}|{purpose}"
        ).encode("utf-8")
        # Preserve the historical view-0 hash exactly so the control path is
        # bitwise reproducible. Replacement draws beyond the first receive a
        # deterministic extra crop view without changing target selection.
        if view_id > 0:
            payload += f"|view={view_id}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def _session_target_selection(
        self,
        record: Path,
    ) -> tuple[List[Path], List[bool], List[bool]]:
        """Return paths, duplicate flags, and structural availability.

        The paired target is always index 0. Repeated placeholders are kept
        to preserve a fixed K but are explicitly unavailable to set losses.
        """
        paired = self._paired_target(record)
        alternatives = self._session_alternatives(record)
        if self.target_selection_mode == "fixed_first":
            ordered = list(alternatives)
        else:
            ordered = list(alternatives)
            self._stable_random(
                record,
                purpose="target-selection",
                include_epoch=True,
            ).shuffle(ordered)

        required = self.num_targets - 1
        selected = ordered[:required]
        duplicate_flags = [False] * len(selected)
        available = [True] * len(selected)
        if len(selected) < required:
            repeat_pool = ordered if ordered else [paired]
            repeat_index = 0
            while len(selected) < required:
                selected.append(repeat_pool[repeat_index % len(repeat_pool)])
                duplicate_flags.append(True)
                available.append(False)
                repeat_index += 1
        return (
            [paired, *selected],
            [False, *duplicate_flags],
            [True, *available],
        )

    def _session_target_paths(self, record: Path) -> List[Path]:
        paths, _, _ = self._session_target_selection(record)
        return paths

    def _session_style_target_paths(self, record: Path) -> List[Path]:
        paths, _, _ = self._session_target_selection(record)
        return paths[1:]

    @staticmethod
    def _load_tensor(path: Path) -> torch.Tensor:
        if path.suffix == ".npy":
            return torch.from_numpy(np.load(path).astype(np.float32))
        raise ValueError(f"Unsupported condition file: {path}")

    def _choose_start(
        self,
        record: Path,
        total_length: int,
        view_id: int = 0,
    ) -> int:
        if total_length <= self.clip_length:
            return 0
        if self.training:
            step_count = (total_length - self.clip_length) // self.crop_stride
            return self._stable_random(
                record,
                purpose="source-crop",
                include_epoch=True,
                view_id=view_id,
            ).randint(0, step_count) * self.crop_stride
        if self.crop_mode == "center":
            return (total_length - self.clip_length) // 2
        return 0

    def _choose_unpaired_target_start(
        self,
        record: Path,
        target_path: Path,
        total_length: int,
        view_id: int = 0,
    ) -> int:
        """Crop independent recordings without reusing the source offset."""
        if total_length <= self.clip_length:
            return 0
        if self.training:
            step_count = (total_length - self.clip_length) // self.crop_stride
            return self._stable_random(
                record,
                purpose=f"target-crop|{target_path.as_posix()}",
                include_epoch=True,
                view_id=view_id,
            ).randint(0, step_count) * self.crop_stride
        if self.crop_mode == "center":
            return (total_length - self.clip_length) // 2
        return 0

    def __getitem__(
        self,
        index: Union[int, Tuple[int, int]],
    ) -> Dict[str, object]:
        crop_view_id = 0
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError("Dataset tuple index must be (index, view_id)")
            index, crop_view_id = int(index[0]), int(index[1])
            if crop_view_id < 0:
                raise ValueError("crop view id must be non-negative")
        record = self.records[index]
        relative_without_suffix = record.with_suffix("")
        audio = self._load_tensor(
            self.audio_dir / relative_without_suffix.with_suffix(".npy")
        )
        speaker_emotion = self._load_tensor(self.emotion_dir / record)
        speaker_3dmm = self._load_tensor(
            self.coefficient_dir / relative_without_suffix.with_suffix(".npy")
        ).squeeze()
        if speaker_3dmm.ndim != 2 or speaker_3dmm.shape[-1] != 58:
            raise ValueError(
                "Speaker 3DMM must reduce to [T,58], got "
                f"{tuple(speaker_3dmm.shape)} for {record}"
            )
        speaker_3dmm = (speaker_3dmm - self.mean_face) / self.std_face

        source_length = min(
            audio.shape[0],
            speaker_emotion.shape[0],
            speaker_3dmm.shape[0],
        )
        start = self._choose_start(
            record,
            source_length,
            view_id=crop_view_id,
        )
        valid_source_length = min(
            self.clip_length,
            max(0, source_length - start),
        )
        result: Dict[str, object] = {
            "speaker_audio": crop_and_pad(audio, start, self.clip_length),
            "speaker_emotion": crop_and_pad(
                speaker_emotion, start, self.clip_length
            ),
            "speaker_3dmm": crop_and_pad(
                speaker_3dmm, start, self.clip_length
            ),
            "length": torch.tensor(valid_source_length, dtype=torch.long),
            "clip_id": relative_without_suffix.as_posix(),
            "start": torch.tensor(start, dtype=torch.long),
            "crop_view_id": torch.tensor(crop_view_id, dtype=torch.long),
        }

        if self.target_mode in {"paired", "session_style"}:
            target_path = self._paired_target(record)
            target = self._load_tensor(self.emotion_dir / target_path)
            if self.load_listener_3dmm:
                coefficients = self._load_tensor(
                    self.coefficient_dir / target_path.with_suffix(".npy")
                ).squeeze()
                if coefficients.shape != (target.shape[0], 58):
                    raise ValueError(f"Unaligned listener 3DMM: {target_path}: {coefficients.shape} vs {target.shape}")
                if not torch.isfinite(coefficients).all():
                    raise ValueError(f"Nonfinite listener 3DMM: {target_path}")
                coefficients = (coefficients - self.mean_face) / self.std_face
                result["target_3dmm"] = crop_and_pad(coefficients, start, self.clip_length)
            valid_length = min(
                valid_source_length,
                max(0, target.shape[0] - start),
            )
            result["target"] = crop_and_pad(
                target, start, self.clip_length
            )
            result["length"] = torch.tensor(valid_length, dtype=torch.long)
            if self.target_mode == "session_style":
                assert self.style_descriptors is not None
                style_paths = self._session_style_target_paths(record)
                result["target_styles"] = torch.stack(
                    [
                        self.style_descriptors[path.as_posix()]
                        for path in style_paths
                    ],
                    dim=0,
                )
        elif self.target_mode == "session_cropped":
            (
                target_paths,
                target_is_duplicate,
                target_structurally_available,
            ) = self._session_target_selection(record)
            targets = []
            targets_3dmm = []
            target_lengths = []
            target_available = []
            target_starts = []
            for target_index, target_path in enumerate(target_paths):
                target = self._load_tensor(self.emotion_dir / target_path)
                if self.load_listener_3dmm:
                    coefficients = self._load_tensor(
                        self.coefficient_dir / target_path.with_suffix(".npy")
                    ).squeeze()
                    if coefficients.shape != (target.shape[0], 58):
                        raise ValueError(f"Unaligned listener 3DMM: {target_path}: {coefficients.shape} vs {target.shape}")
                    if not torch.isfinite(coefficients).all():
                        raise ValueError(f"Nonfinite listener 3DMM: {target_path}")
                    coefficients = (coefficients - self.mean_face) / self.std_face
                target_start = (
                    start
                    if target_index == 0
                    else self._choose_unpaired_target_start(
                        record,
                        target_path,
                        target.shape[0],
                        view_id=crop_view_id,
                    )
                )
                valid_length = min(
                    self.clip_length,
                    max(0, target.shape[0] - target_start),
                )
                if target_index == 0:
                    valid_length = min(valid_source_length, valid_length)
                targets.append(
                    crop_and_pad(
                        target,
                        target_start,
                        self.clip_length,
                    )
                )
                target_lengths.append(valid_length)
                if self.load_listener_3dmm:
                    targets_3dmm.append(crop_and_pad(coefficients, target_start, self.clip_length))
                target_starts.append(target_start)
                target_available.append(
                    target_structurally_available[target_index]
                    and valid_length > 0
                )
            result["targets"] = torch.stack(targets, dim=0)
            if self.load_listener_3dmm:
                result["targets_3dmm"] = torch.stack(targets_3dmm, dim=0)
            result["target_lengths"] = torch.tensor(
                target_lengths,
                dtype=torch.long,
            )
            result["target_available_mask"] = torch.tensor(
                target_available,
                dtype=torch.bool,
            )
            result["target_is_duplicate"] = torch.tensor(
                target_is_duplicate,
                dtype=torch.bool,
            )
            result["target_paths"] = [
                target_path.as_posix() for target_path in target_paths
            ]
            result["target_starts"] = torch.tensor(
                target_starts,
                dtype=torch.long,
            )
            result["target"] = targets[0]
            result["length"] = torch.tensor(
                target_lengths[0],
                dtype=torch.long,
            )
        else:
            result["targets"] = [
                self._load_tensor(self.emotion_dir / target_path)
                for target_path in self._session_target_paths(record)
            ]
        return result


def conditional_eval_collate(
    batch: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    if len(batch) != 1:
        raise ValueError("FRC evaluation currently requires batch_size=1")
    item = batch[0]
    return {
        "speaker_audio": item["speaker_audio"].unsqueeze(0),
        "speaker_emotion": item["speaker_emotion"].unsqueeze(0),
        "speaker_3dmm": item["speaker_3dmm"].unsqueeze(0),
        "length": item["length"].unsqueeze(0),
        "clip_id": [item["clip_id"]],
        "targets": [item["targets"]],
    }
