import time
import numpy as np
import multiprocessing as mp
import torch
from framework.metrics import *


def compute_temporal_s_mse(preds):
    """Compute K10 S-MSE after removing each sequence's temporal mean."""
    if not preds:
        raise ValueError("preds must be non-empty")
    centered_preds = []
    for pred in preds:
        pred = torch.as_tensor(pred).float()
        centered_preds.append(
            pred - pred.mean(dim=1, keepdim=True)
        )
    return compute_s_mse(centered_preds)


def _func(target, pred):
    # target: (10, l, dim)
    # pred: (10, l, dim)
    num_preds = pred.shape[0]
    mean_mae_sum = 0
    for i in range(num_preds):
        mae_list = []
        for j in range(num_preds):
            mae = np.mean(np.abs(target[j][:, 15:].numpy() - pred[i][:, 15:].numpy()))
            mae_list.append(mae)
        mean_mae_sum += np.mean(mae_list)
    return mean_mae_sum / num_preds


def compute_MAE(preds, targets, p=4):
    MAE_list = []
    with mp.Pool(processes=p) as pool:
        MAE_list += pool.starmap(_func, zip(targets, preds))
    return np.mean(MAE_list)


def compute_metrics(speaker_inputs,
                    listener_predictions,
                    listener_targets,
                    threads=4,
                    metric_names=None):

    available_metrics = {
        'FRC': (compute_FRC, (listener_predictions, listener_targets), {'p': threads}),
        'FRD': (compute_FRD, (listener_predictions, listener_targets), {'p': threads}),
        'TLCC': (compute_TLCC, (listener_predictions, speaker_inputs),  {'p': threads}),
        'smse': (compute_s_mse, (listener_predictions,), {}),
        'temporal_smse': (compute_temporal_s_mse, (listener_predictions,), {}),
        'FRVar': (compute_FRVar, (listener_predictions,), {}),
        # 'MAE': (compute_MAE, (listener_predictions, listener_targets), {'p': threads}),
    }

    if metric_names is None:
        metric_names = list(available_metrics)
    else:
        metric_names = list(metric_names)
        unknown = [
            name for name in metric_names
            if name not in available_metrics
        ]
        if unknown:
            raise ValueError(
                f"Unknown metrics {unknown}; available metrics are "
                f"{list(available_metrics)}"
            )
    metrics = {
        name: available_metrics[name]
        for name in metric_names
    }

    results = {}
    for name, (func, args, kwargs) in metrics.items():
        t0 = time.perf_counter()
        value = func(*args, **kwargs)
        if hasattr(value, 'item'):
            value = value.item()
        elapsed = time.perf_counter() - t0

        results[name] = value
        results[f"{name}_time"] = elapsed
        print(f"{name:6s} = {value:.6f}  (time: {elapsed:.4f}s)")

    return results
