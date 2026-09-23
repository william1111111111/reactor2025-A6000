"""Vectorized Mam-Reactor FRC, retaining constant-channel correlation=0."""
import numpy as np


def candidate_frc(target, prediction):
    def stats(value):
        # Contiguous time axis reproduces original per-channel reductions.
        a = np.ascontiguousarray(np.asarray(value).transpose(0, 2, 1))
        mean, var, std = a.mean(-1), a.var(-1), a.std(-1)
        centered = a.astype(np.float64)
        centered -= centered.mean(-1, keepdims=True)
        return mean, var, std, centered
    tm, tv, ts, tc = stats(target)
    pm, pv, ps, pc = stats(prediction)
    cross = np.einsum('kdt,ndt->knd', pc, tc, optimize=True)
    scale = np.sqrt((pc * pc).sum(-1))[:, None, :] * np.sqrt((tc * tc).sum(-1))[None, :, :]
    corr = np.divide(cross, scale, out=np.zeros_like(cross), where=scale != 0)
    corr = np.clip(corr, -1, 1)
    # Original numpy scalar power promotes the squared mean difference.
    delta = (tm[None] - pm[:, None]).astype(np.float64)
    denominator = (tv[None] + pv[:, None]).astype(np.float64) + delta ** 2
    ccc = 2 * corr * ts[None] * ps[:, None] / (denominator + 1e-8)
    return ccc.mean(-1).max(-1)
