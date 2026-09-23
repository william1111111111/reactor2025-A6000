"""One shared ConditionalREGNN conditioned by a lightweight expert ID."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from regnn.conditional_model import ConditionalREGNN, ConditionalREGNNConfig


@dataclass(frozen=True)
class ExpertStudentConfig:
    backbone: ConditionalREGNNConfig = field(default_factory=ConditionalREGNNConfig)
    num_experts: int = 10
    expert_embedding_dim: int = 64


class ExpertConditionalREGNN(nn.Module):
    """Shared encoder/decoder; expert ID modulates the encoded time sequence."""

    def __init__(self, config: ExpertStudentConfig):
        super().__init__()
        if config.num_experts < 1 or config.expert_embedding_dim < 1:
            raise ValueError("num_experts and expert_embedding_dim must be positive")
        self.config = config
        self.backbone = ConditionalREGNN(config.backbone)
        self.expert_embedding = nn.Embedding(
            config.num_experts, config.expert_embedding_dim,
        )
        self.expert_modulation = nn.Sequential(
            nn.Linear(config.expert_embedding_dim, config.backbone.hidden_dim),
            nn.GELU(),
            nn.Linear(config.backbone.hidden_dim, config.backbone.hidden_dim),
        )

    def _condition(
        self,
        context: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        expert_ids = torch.as_tensor(
            expert_ids, device=context.device, dtype=torch.long,
        )
        if expert_ids.shape != (context.shape[0],):
            raise ValueError("expert_ids must have shape [B]")
        if bool(((expert_ids < 0) | (expert_ids >= self.config.num_experts)).any()):
            raise ValueError("expert_ids out of range")
        offset = self.expert_modulation(self.expert_embedding(expert_ids))
        return context + offset.unsqueeze(1)

    def forward(
        self,
        speaker_audio: torch.Tensor,
        speaker_emotion: torch.Tensor,
        speaker_3dmm: torch.Tensor,
        lengths: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context, valid_mask = self.backbone.encode_conditions(
            speaker_audio, speaker_emotion, speaker_3dmm, lengths,
        )
        return self.backbone.decode_context(
            self._condition(context, expert_ids), valid_mask,
        )

    def generate(
        self,
        speaker_audio: torch.Tensor,
        speaker_emotion: torch.Tensor,
        speaker_3dmm: torch.Tensor,
        lengths: torch.Tensor,
        expert_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode speakers once and decode requested expert slots in chunks."""
        context, valid_mask = self.backbone.encode_conditions(
            speaker_audio, speaker_emotion, speaker_3dmm, lengths,
        )
        batch, frames, hidden = context.shape
        if expert_ids is None:
            expert_ids = torch.arange(
                self.config.num_experts, device=context.device,
            )
        expert_ids = torch.as_tensor(
            expert_ids, device=context.device, dtype=torch.long,
        )
        if expert_ids.ndim != 1:
            raise ValueError("generate expert_ids must be a one-dimensional list")
        count = expert_ids.numel()
        expanded_context = context[:, None].expand(
            batch, count, frames, hidden,
        ).reshape(batch * count, frames, hidden)
        expanded_mask = valid_mask[:, None].expand(
            batch, count, frames,
        ).reshape(batch * count, frames)
        expanded_ids = expert_ids[None].expand(batch, count).reshape(-1)
        output = self.backbone.decode_context(
            self._condition(expanded_context, expanded_ids), expanded_mask,
        )
        result = {}
        for key, value in output.items():
            if value.shape[0] == batch * count:
                result[key] = value.reshape(batch, count, *value.shape[1:])
            else:
                result[key] = value
        result["expert_ids"] = expert_ids
        return result
