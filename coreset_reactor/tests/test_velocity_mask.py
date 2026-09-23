import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mam_reactor"))
from regnn.conditional_model import ConditionalREGNNLoss


def test_cross_listener_velocity_can_be_disabled_without_disabling_quality():
    prediction = torch.zeros(2, 4, 25, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[1, 1::2] = 1.0
    mask = torch.ones(2, 4, dtype=torch.bool)
    criterion = ConditionalREGNNLoss(ccc_weight=0, velocity_weight=1)
    enabled = criterion(prediction, target, mask, torch.tensor([True, True]))
    masked = criterion(prediction, target, mask, torch.tensor([True, False]))
    assert masked["mse"] == enabled["mse"]
    assert masked["velocity"] < enabled["velocity"]
