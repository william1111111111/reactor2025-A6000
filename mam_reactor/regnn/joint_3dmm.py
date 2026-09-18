"""Joint listener geometry supervision for the nine learned reaction queries.

Geometry is normalized with the same FaceVerse statistics as speaker inputs.
Each 58-D target shares a path/crop/mask with its corresponding 25-D target.
The fixed q0 anchor has no geometry head and is deliberately excluded.
"""
import torch
from regnn.emotion_query_mamba import masked_sinkhorn_transport


def pairwise_mse(prediction, target, mask):
    prediction, target = prediction.float(), target.float()
    mask = mask.to(prediction)
    # [B,Q,K], without materializing [B,Q,K,T,D].
    squared = (
        torch.einsum("bqt,bkt->bqk", prediction.square().sum(-1), mask)
        + (target.square().sum(-1) * mask).sum(-1)[:, None]
        - 2 * torch.einsum("bqtd,bktd->bqk", prediction, target * mask[..., None])
    )
    return squared.clamp_min(0) / (mask.sum(-1)[:, None].clamp_min(1) * prediction.shape[-1])


def listener_3dmm_loss(prediction, emotion_prediction, target, emotion_target, mask):
    mse = pairwise_mse(prediction, target, mask)
    velocity_mask = mask[..., 1:] & mask[..., :-1]
    velocity = pairwise_mse(
        prediction[:, :, 1:] - prediction[:, :, :-1],
        target[:, :, 1:] - target[:, :, :-1], velocity_mask,
    )
    with torch.no_grad():
        # A joint matching cost couples the geometry and emotion of each query.
        cost = mse.detach() + pairwise_mse(emotion_prediction, emotion_target, mask)
        transport = masked_sinkhorn_transport(cost, mask.any(-1), 0.1, 32)
        weights = mask.any(-1).any(-1).to(mse)
    def reduce(values):
        return ((values * transport).sum((1, 2)) * weights).sum() / weights.sum().clamp_min(1)
    return {"listener_3dmm_mse": reduce(mse), "listener_3dmm_velocity": reduce(velocity)}
