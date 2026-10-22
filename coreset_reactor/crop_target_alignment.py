"""Explicit full-timeline GT alignment for cropped source evaluation.

Nonpaired session references have no frame-level correspondence. This retains
the official learned duration conversion, then selects the source's window;
it does not claim to establish semantic synchronization between recordings.
"""
import torch

from coreset_reactor.evaluate import _deterministic_postprocess


def align_targets_then_crop(processor, prediction, raw_targets, *,
                            source_length, crop_start, device, seed, clip_id):
    frames = prediction.shape[1]
    if source_length < 1 or crop_start < 0 or crop_start + frames > source_length:
        raise ValueError('Crop must lie inside the complete source timeline')
    # Processor reads only prediction shape, never its values. A zero-stride
    # view supplies the complete length without generating missing predictions.
    shape_only = prediction[:, :1].expand(-1, source_length, -1)
    aligned = _deterministic_postprocess(
        processor, shape_only, raw_targets, device, seed, clip_id)
    expected = (len(raw_targets), source_length, prediction.shape[-1])
    if tuple(aligned.shape) != expected:
        raise ValueError(f'Unexpected full GT shape: {aligned.shape} != {expected}')
    cropped = aligned[:, crop_start:crop_start + frames].contiguous()
    if not torch.isfinite(cropped).all():
        raise ValueError('Nonfinite aligned GT')
    return cropped
