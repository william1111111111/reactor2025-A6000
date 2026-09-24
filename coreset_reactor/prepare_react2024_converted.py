"""Convert the public React2024 baseline layout to Mam-Reactor's feature layout.

The React2024 release stores emotions as CSV, 3DMM as ``3D_FV_files`` and
audio as WAV.  The fixed-pair Mam-Reactor experiments consume the converted
layout used by ``ConditionalReactionDataset``.  This converter keeps the raw
role pairing and creates two directional sessions per source session so that
the opposite-role GT pool never mixes the two directions.

Audio embeddings are produced by the same custom Wav2Vec2 path used by the
Mam-Reactor preprocessing code: convolutional features are interpolated to
the emotion-frame count before the encoder, yielding 768D frame features.
"""
from __future__ import annotations

import argparse
import csv
import contextlib
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor


def _load_emotion(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) != 25:
            raise ValueError(f"invalid React2024 emotion header: {path}")
        for row in reader:
            if not row or row[0].startswith("."):
                continue
            rows.append([float(value) for value in row[:25]])
    value = np.asarray(rows, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 25 or not len(value):
        raise ValueError(f"invalid React2024 emotion array: {path}: {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"nonfinite React2024 emotion array: {path}")
    return value


def _load_3dmm(path: Path, frames: int) -> np.ndarray:
    value = np.load(path, allow_pickle=False).astype(np.float32)
    if value.ndim == 3 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 2 or value.shape[1] != 58 or not len(value):
        raise ValueError(f"invalid React2024 3DMM array: {path}: {value.shape}")
    value = value[:frames]
    if len(value) != frames or not np.isfinite(value).all():
        raise ValueError(f"unaligned/nonfinite React2024 3DMM: {path}")
    return value


def _safe_session(corpus: str, group: str, source_role: str) -> str:
    value = f"{corpus}__{group}__{source_role}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _role_map(raw_group: Path, emotion_roles: list[str]) -> dict[str, str]:
    """Map P1/P2 emotion roles to the corresponding video/audio directory."""
    if set(emotion_roles) != {"P1", "P2"}:
        raise ValueError(f"expected P1/P2 emotion roles in {raw_group}")
    if (raw_group / "Expert_video").is_dir() and (raw_group / "Novice_video").is_dir():
        return {"P1": "Expert_video", "P2": "Novice_video"}
    candidates = sorted(
        path.name for path in raw_group.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if len(candidates) != 2:
        raise ValueError(f"cannot infer two video roles in {raw_group}: {candidates}")
    return {"P1": candidates[0], "P2": candidates[1]}


def _find_by_stem(root: Path, stem: str, suffix: str) -> Path:
    matches = [path for path in root.rglob(f"{stem}{suffix}")
               if not path.name.startswith(".")]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {stem}{suffix} under {root}, got {matches}")
    return matches[0]


def _save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value.astype(np.float32, copy=False))


def enumerate_records(raw_root: Path, split: str) -> list[dict]:
    root = raw_root / split
    emotion_root = root / "Emotion"
    records = []
    for corpus in sorted(path for path in emotion_root.iterdir() if path.is_dir()):
        for group in sorted(path for path in corpus.iterdir() if path.is_dir()):
            role_dirs = {
                role: group / role
                for role in ("P1", "P2")
                if (group / role).is_dir()
            }
            if set(role_dirs) != {"P1", "P2"}:
                continue
            feature_group = root / "3D_FV_files" / corpus.name / group.name
            audio_group = root / "Audio_files" / corpus.name / group.name
            video_group = root / "Video_files" / corpus.name / group.name
            role_to_video = _role_map(video_group, list(role_dirs))
            for source_role in ("P1", "P2"):
                target_role = "P2" if source_role == "P1" else "P1"
                session = _safe_session(corpus.name, group.name, source_role)
                for emotion_path in sorted(role_dirs[source_role].glob("*.csv")):
                    if emotion_path.name.startswith("."):
                        continue
                    stem = emotion_path.stem
                    target_emotion = role_dirs[target_role] / emotion_path.name
                    if not target_emotion.is_file():
                        # A few release records are present for one role only;
                        # they cannot form a paired conditional example and
                        # are excluded symmetrically from the converted split.
                        continue
                    video_role = role_to_video[source_role]
                    target_video_role = role_to_video[target_role]
                    try:
                        audio = _find_by_stem(audio_group / video_role, stem, ".wav")
                        target_audio = _find_by_stem(
                            audio_group / target_video_role, stem, ".wav")
                        coeff = _find_by_stem(feature_group / video_role, stem, ".npy")
                        target_coeff = _find_by_stem(
                            feature_group / target_video_role, stem, ".npy")
                    except FileNotFoundError:
                        # The release contains a small number of incomplete
                        # rows; omit them rather than inventing a feature.
                        continue
                    records.append({
                        "split": split, "corpus": corpus.name, "group": group.name,
                        "source_role": source_role, "target_role": target_role,
                        "session": session, "stem": stem,
                        "emotion": emotion_path, "target_emotion": target_emotion,
                        "audio": audio, "target_audio": target_audio,
                        "coeff": coeff, "target_coeff": target_coeff,
                    })
    if not records:
        raise FileNotFoundError(f"no React2024 records under {emotion_root}")
    return records


def convert_static(raw_root: Path, output_root: Path, splits: tuple[str, ...]) -> list[dict]:
    all_records = []
    for split in splits:
        records = enumerate_records(raw_root, split)
        for row in records:
            source = _load_emotion(row["emotion"])
            target = _load_emotion(row["target_emotion"])
            coeff = _load_3dmm(row["coeff"], min(len(source), len(target)))
            frames = min(len(source), len(target), len(coeff))
            source = source[:frames]
            target = target[:frames]
            relative = Path(row["session"]) / f"{row['stem']}.npy"
            facial = output_root / split / "facial-attributes"
            _save_array(facial / "speaker" / relative, source)
            _save_array(facial / "listener" / relative, target)
            _save_array(output_root / split / "coefficients" / "speaker" / relative, coeff[:frames])
            # The target 3DMM is not consumed by the fixed 25D experiment, but
            # keeping it makes the converted layout complete and auditable.
            _save_array(output_root / split / "coefficients" / "listener" / relative,
                        _load_3dmm(row["target_coeff"], frames))
            row["converted_relative"] = relative.as_posix()
            row["frames"] = frames
            all_records.append(row)
    return all_records


def extract_audio(records: list[dict], output_root: Path, model_root: Path,
                  device: str, shard: int, shards: int, threads: int) -> None:
    from framework.feature_extractor.wav2vec import Wav2VecModel
    from safetensors.torch import load_file
    import librosa
    torch.set_num_threads(threads)
    device_obj = torch.device(device)
    if device_obj.type == "cuda":
        torch.cuda.set_device(device_obj)
    # Transformers 4.37 serializes the weight-normalized positional
    # convolution as ``weight_g/weight_v`` while the parametrized module in
    # this repository exposes ``original0/original1``.  Instantiate from the
    # config and map those two tensors explicitly; otherwise the loader emits
    # a warning and silently leaves the positional convolution random.
    encoder = Wav2VecModel(
        Wav2Vec2Config.from_pretrained(str(model_root), local_files_only=True)
    )
    state = {
        key.removeprefix("wav2vec2."): value
        for key, value in load_file(str(model_root / "model.safetensors")).items()
    }
    state["encoder.pos_conv_embed.conv.parametrizations.weight.original0"] = state.pop(
        "encoder.pos_conv_embed.conv.weight_g"
    )
    state["encoder.pos_conv_embed.conv.parametrizations.weight.original1"] = state.pop(
        "encoder.pos_conv_embed.conv.weight_v"
    )
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if set(missing) != {"masked_spec_embed"} or set(unexpected) != {"lm_head.bias", "lm_head.weight"}:
        raise RuntimeError(f"unexpected Wav2Vec state mismatch: missing={missing}, unexpected={unexpected}")
    encoder = encoder.eval().to(device_obj)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        str(model_root), local_files_only=True)
    selected = records[shard::shards]
    for number, row in enumerate(selected, 1):
        relative = Path(row["converted_relative"])
        for role, audio_key in (("speaker", "audio"), ("listener", "target_audio")):
            output = output_root / row["split"] / "audio-features" / role / relative
            if output.is_file():
                continue
            waveform, sample_rate = librosa.load(row[audio_key], sr=16000)
            input_values = np.asarray(
                extractor(waveform, sampling_rate=sample_rate).input_values[0],
                dtype=np.float32,
            )
            tensor = torch.from_numpy(input_values)[None].to(device_obj)
            with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
                embedding = encoder(
                    tensor, seq_len=int(row["frames"]), output_hidden_states=True,
                ).last_hidden_state.squeeze(0).float().cpu().numpy()
            if embedding.shape != (int(row["frames"]), 768):
                raise ValueError(f"unexpected audio feature shape for {row[audio_key]}: {embedding.shape}")
            _save_array(output, embedding)
        if number == 1 or number % 25 == 0 or number == len(selected):
            print(f"[react2024-audio] shard={shard}/{shards} {number}/{len(selected)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("invalid shard")
    if args.static_only and args.audio_only:
        raise ValueError("static-only and audio-only are mutually exclusive")
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    splits = tuple(args.splits)
    inventory_path = output_root / "react2024_conversion_inventory.json"
    if not args.audio_only:
        records = convert_static(raw_root, output_root, splits)
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = [{key: (str(value) if isinstance(value, Path) else value)
                         for key, value in row.items()} for row in records]
        inventory_path.write_text(json.dumps({
            "dataset": "REACT2024", "raw_root": str(raw_root),
            "output_root": str(output_root), "records": serializable,
        }, indent=2) + "\n")
    else:
        payload = json.loads(inventory_path.read_text())
        records = payload["records"]
        records = [{key: Path(value) if key in {"emotion", "target_emotion", "audio", "target_audio", "coeff", "target_coeff"}
                    else value for key, value in row.items()} for row in records]
    if not args.static_only:
        extract_audio(records, output_root, args.model_root, args.device,
                      args.shard, args.shards, args.threads)


if __name__ == "__main__":
    main()
