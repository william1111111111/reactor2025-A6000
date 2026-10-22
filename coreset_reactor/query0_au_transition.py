"""Minimal output stabilization for constant query-0 AU trajectories."""
from __future__ import annotations

import torch


def stabilize_query0_au_transitions(
    prediction: torch.Tensor,
    source_face: torch.Tensor,
    margin: int = 49,
) -> torch.Tensor:
    """Give each constant query-0 AU one source-aligned transition frame."""
    if (
        prediction.ndim != 3
        or prediction.shape[0] != 10
        or prediction.shape[2] != 25
    ):
        raise ValueError("prediction must be [10,T,25]")
    if source_face.shape != prediction.shape[1:]:
        raise ValueError("source_face must be [T,25] and match prediction")
    if margin < 0 or prediction.shape[1] <= 2 * margin:
        raise ValueError("margin leaves no interior transition frame")
    if (
        not torch.isfinite(prediction).all()
        or not torch.isfinite(source_face).all()
    ):
        raise ValueError("prediction and source_face must be finite")

    result = prediction.clone()
    source_range = source_face.amax(dim=0) - source_face.amin(dim=0)
    # A constant source dimension fixes official TLCC at its boundary lag, so
    # changing the response in those contexts would only perturb quality.
    if torch.any(source_range == 0):
        return result

    query0_aus = result[0, :, :15]
    constant = query0_aus.amax(dim=0) == query0_aus.amin(dim=0)
    interior = source_face[margin:prediction.shape[1] - margin, :15]
    for channel in torch.nonzero(constant, as_tuple=False).flatten().tolist():
        value = float(query0_aus[0, channel])
        if value not in (0.0, 1.0):
            raise ValueError(
                "query-0 AUs must be officially rounded before transition"
            )
        if value == 0.0:
            local_index = int(torch.argmax(interior[:, channel]))
            replacement = 1.0
        else:
            local_index = int(torch.argmin(interior[:, channel]))
            replacement = 0.0
        result[0, margin + local_index, channel] = replacement
    return result
