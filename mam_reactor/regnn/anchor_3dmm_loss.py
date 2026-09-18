"""Paired, frame-masked normalized listener coefficient loss."""
import torch


def geometry_loss(prediction, target, valid_mask):
    if prediction.shape != target.shape or prediction.shape[-1] != 58:
        raise ValueError("Prediction and target must share [B,T,58]")
    prediction, target = prediction.float(), target.float()
    mask = valid_mask.to(prediction)[..., None]
    mse = ((prediction - target).square() * mask).sum() / (mask.sum().clamp_min(1) * 58)
    delta = (prediction[:,1:] - prediction[:,:-1]) - (target[:,1:] - target[:,:-1])
    pair_mask = (valid_mask[:,1:] & valid_mask[:,:-1]).to(prediction)[...,None]
    velocity = (delta.square() * pair_mask).sum() / (pair_mask.sum().clamp_min(1) * 58)
    return {"geometry_mse": mse, "geometry_velocity": velocity}
