"""Official REACT2024 inter-condition diversity metric.

Restored from ``reactmultimodalchallenge/baseline_react2024`` commit
``0c3ce587d36f05b26b604eea89c5261544b72847``.
"""
import torch


def compute_FRDvs(preds):
    # preds: (N, 10, 750, 25)
    preds_ = preds.reshape(preds.shape[0], preds.shape[1], -1)
    preds_ = preds_.transpose(0, 1)
    # preds_: (10, N, 750*25)
    dist = torch.pow(torch.cdist(preds_, preds_), 2)
    # dist: (10, N, N)
    dist = torch.sum(dist) / (
        preds.shape[0] * (preds.shape[0] - 1) * preds.shape[1]
    )
    return dist / preds_.shape[-1]
