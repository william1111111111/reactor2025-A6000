"""Batched unconstrained DTW: official FP32 inputs, FP64 recurrence, no band.

One CUDA block per prediction/reference/channel-group pair; three anti-diagonals
are sufficient for the exact recurrence. No temporal resampling or pruning.
"""
import math
import numpy as np
from numba import cuda, float64


@cuda.jit
def _dtw_groups(pred, target, scratch, result):
    pair = cuda.blockIdx.x
    thread = cuda.threadIdx.x
    stride = cuda.blockDim.x
    group = pair % 3
    target_id = (pair // 3) % target.shape[0]
    pred_id = (pair // 3) // target.shape[0]
    n, m = pred.shape[1], target.shape[1]
    start, end = 0, 15
    if group == 1:
        start, end = 15, 17
    elif group == 2:
        start, end = 17, 25
    for j in range(thread, m + 1, stride):
        for row in range(3):
            scratch[pair, row, j] = math.inf
    if thread == 0:
        scratch[pair, 0, 0] = 0.0
    cuda.syncthreads()
    # Diagonal 1 consists only of infinite boundary cells.
    for diagonal in range(2, n + m + 1):
        current = diagonal % 3
        previous = (diagonal - 1) % 3
        older = (diagonal - 2) % 3
        for j in range(thread, m + 1, stride):
            i = diagonal - j
            value = math.inf
            if j > 0 and i > 0 and i <= n:
                cost = 0.0
                for feature in range(start, end):
                    diff = float64(pred[pred_id, i - 1, feature]) - float64(target[target_id, j - 1, feature])
                    cost += diff * diff
                value = cost + min(scratch[pair, previous, j],
                                   scratch[pair, previous, j - 1],
                                   scratch[pair, older, j - 1])
            scratch[pair, current, j] = value
        cuda.syncthreads()
    if thread == 0:
        result[pred_id, target_id, group] = math.sqrt(scratch[pair, (n + m) % 3, m])


def pairwise_frd(prediction, targets):
    pred = np.ascontiguousarray(prediction, dtype=np.float32)
    target = np.ascontiguousarray(targets, dtype=np.float32)
    if pred.ndim != 3 or target.ndim != 3 or pred.shape[2] != 25 or target.shape[2] != 25:
        raise ValueError('Expected [K,T,25] and [R,U,25]')
    if min(pred.shape[:2] + target.shape[:2]) < 1 or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError('Empty/nonfinite DTW input')
    count = pred.shape[0] * target.shape[0] * 3
    scratch = cuda.device_array((count, 3, target.shape[1] + 1), dtype=np.float64)
    result = cuda.device_array((pred.shape[0], target.shape[0], 3), dtype=np.float64)
    _dtw_groups[count, 128](cuda.to_device(pred), cuda.to_device(target), scratch, result)
    values = result.copy_to_host()
    return values[:, :, 0] / 15.0 + values[:, :, 1] + values[:, :, 2] / 8.0
