"""Paired-only REACT features. No legacy imports, alternatives or target pools."""
import hashlib
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class PairedReactionDataset(Dataset):
    """Fixed aligned crops of speaker -> paired listener, using directory splits.

    source_lengths and pair_lengths refer to the returned crop, not full files.
    Source cropping depends only on source information. A crop with no paired
    overlap is rejected explicitly, never silently resampled using the target.
    Facial attributes are used as stored; only 3DMM receives FaceVerse scaling.
    """
    def __init__(self, root_dir, split, clip_length=128, crop_mode='center', seed=123,
                 normalization_dir=None):
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val or test')
        if clip_length < 1 or crop_mode not in ('start', 'center', 'random', 'tail'):
            raise ValueError('invalid clip_length or crop_mode')
        self.root = Path(root_dir)
        self.split = split
        self.directory = self.root / split
        self.clip_length, self.crop_mode, self.seed = clip_length, crop_mode, seed
        self.epoch = 0
        self.records = sorted((self.directory / 'facial-attributes' / 'speaker').glob('*/*.npy'))
        if not self.records:
            raise FileNotFoundError(f'no speaker facial attributes in {self.directory}')
        norm = Path(normalization_dir) if normalization_dir else self.root.parent / 'external' / 'FaceVerse'
        self.mean = self._load(norm / 'mean_face.npy')
        self.std = self._load(norm / 'std_face.npy').clamp_min(1e-8)
        if self.mean.shape != (58,) or self.std.shape != (58,):
            raise ValueError('FaceVerse mean/std must be [58]')

    @staticmethod
    def _load(path):
        value = torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))
        if not torch.isfinite(value).all():
            raise ValueError(f'nonfinite feature in {path}')
        return value

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        path = self.records[index]
        relative = path.relative_to(self.directory / 'facial-attributes')
        session = relative.parts[1]
        audio = self._load(self.directory / 'audio-features' / relative)
        emotion = self._load(path)
        face = self._load(self.directory / 'coefficients' / relative)
        if face.ndim == 3 and face.shape[1] == 1:
            face = face[:, 0, :]
        target = self._load(self.directory / 'facial-attributes' / 'listener' / session / path.name)
        for value, width in ((audio, 768), (emotion, 25), (face, 58), (target, 25)):
            if value.ndim != 2 or value.shape[1] != width:
                raise ValueError(f'invalid feature shape {tuple(value.shape)} for {relative}')
        face = (face - self.mean) / self.std
        source_total = min(len(audio), len(emotion), len(face))
        if source_total < 1:
            raise ValueError(f'empty source: {relative}')
        max_start = max(0, source_total - self.clip_length)
        if self.crop_mode == 'random':
            payload = f'{self.seed}|{self.epoch}|{relative.as_posix()}'.encode()
            rng = random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big'))
            start = rng.randint(0, max_start)
        elif self.crop_mode == 'center':
            start = max_start // 2
        elif self.crop_mode == 'tail':
            # Intentionally short source crop for padding diagnostics.
            start = max(0, source_total - max(1, self.clip_length // 2))
        else:
            start = 0
        source_length = min(self.clip_length, source_total - start)
        pair_length = min(source_length, max(0, len(target) - start))
        if pair_length < 1:
            raise ValueError(f'no paired overlap for {relative} at source-only crop {start}')

        def padded(value, length):
            out = torch.zeros(self.clip_length, value.shape[-1], dtype=torch.float32)
            out[:length] = value[start:start+length]
            return out

        return dict(speaker_audio=padded(audio, source_length),
                    speaker_emotion=padded(emotion, source_length),
                    speaker_3dmm=padded(face, source_length),
                    source_lengths=torch.tensor(source_length),
                    paired_target=padded(target, pair_length),
                    pair_lengths=torch.tensor(pair_length),
                    clip_id=relative.with_suffix('').as_posix(), session_id=session,
                    crop_start=torch.tensor(start), source_total_length=torch.tensor(source_total),
                    target_total_length=torch.tensor(len(target)))


def paired_model_inputs(batch):
    """Only source lengths enter the unchanged HiRPNet.forward API."""
    return {**{key: batch[key] for key in ('speaker_audio', 'speaker_emotion', 'speaker_3dmm')},
            'lengths': batch['source_lengths']}


def pair_valid_mask(batch):
    lengths = batch['pair_lengths']
    if ((lengths < 1) | (lengths > batch['source_lengths'])).any():
        raise ValueError('pair_lengths must satisfy 1 <= pair <= source')
    return torch.arange(batch['paired_target'].shape[1], device=lengths.device)[None] < lengths[:, None]
