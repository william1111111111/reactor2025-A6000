"""Independent, parallel TCN backbone for ten 25-channel reactions."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 5, padding=2 * dilation,
                                   dilation=dilation, groups=width)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # value: [B,T,H]
        residual = value
        value = self.norm(value).transpose(1, 2)
        value = self.pointwise(self.depthwise(value)).transpose(1, 2)
        return residual + self.dropout(F.gelu(value))


class ParallelTCN(nn.Module):
    def __init__(self, num_predictions: int = 10, hidden_dim: int = 128,
                 control_point_stride: int = 4, dilations=(1, 2, 4, 8, 16),
                 dropout: float = 0.1):
        super().__init__()
        if num_predictions < 1 or control_point_stride < 1:
            raise ValueError("num_predictions and control_point_stride must be positive")
        if hidden_dim != 128:
            raise ValueError("v1 uses 64+32+32=128 feature channels")
        self.num_predictions = num_predictions
        self.control_point_stride = control_point_stride
        self.audio = nn.Linear(768, 64)
        self.face = nn.Linear(25, 32)
        self.mm = nn.Linear(58, 32)
        self.blocks = nn.Sequential(*(TemporalBlock(128, d, dropout) for d in dilations))
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, num_predictions * 25))

    @staticmethod
    def activate(raw: torch.Tensor) -> torch.Tensor:
        # Match the repository's AU / VA / expression channel activation.
        return torch.cat((raw[..., :15].sigmoid(), raw[..., 15:17].tanh(),
                          raw[..., 17:25].softmax(dim=-1)), dim=-1)

    def forward(self, speaker_audio: torch.Tensor, speaker_emotion: torch.Tensor,
                speaker_3dmm: torch.Tensor) -> torch.Tensor:
        # Inputs: [B,T,768], [B,T,25], [B,T,58]. Output: [B,K,T,25].
        if speaker_audio.shape[:2] != speaker_emotion.shape[:2] or \
           speaker_audio.shape[:2] != speaker_3dmm.shape[:2]:
            raise ValueError("speaker modalities must share [B,T]")
        batch, frames = speaker_audio.shape[:2]
        hidden = torch.cat((self.audio(speaker_audio),
                            self.face(speaker_emotion),
                            self.mm(speaker_3dmm)), dim=-1)
        hidden = self.blocks(hidden)
        if self.control_point_stride > 1:
            hidden = F.avg_pool1d(hidden.transpose(1, 2),
                                  self.control_point_stride,
                                  stride=self.control_point_stride,
                                  ceil_mode=True).transpose(1, 2)
        controls = self.head(hidden).reshape(batch, -1, self.num_predictions, 25)
        # Interpolate raw logits first; only then apply channel activations.
        raw = controls.permute(0, 2, 3, 1).reshape(batch * self.num_predictions, 25, -1)
        if raw.shape[-1] != frames:
            raw = F.interpolate(raw, size=frames, mode="linear", align_corners=True)
        raw = raw.transpose(1, 2).reshape(batch, self.num_predictions, frames, 25)
        return self.activate(raw)
