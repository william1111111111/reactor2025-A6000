import pytest
import torch
from coreset_reactor.student_spread_probe import expand_predictions, calibrate_spread
from coreset_reactor.evaluate import official_prediction
from framework.metrics import compute_s_mse


def valid_predictions():
    torch.manual_seed(123)
    x = torch.randn(10, 12, 25)
    return torch.cat((x[..., :15].sigmoid(), x[..., 15:17].tanh(),
                      x[..., 17:25].softmax(-1)), -1)


def test_identity_no_mutation_and_domains():
    x = valid_predictions()
    backup = x.clone()
    assert torch.equal(expand_predictions(x, 1), x)
    y = expand_predictions(x, 2)
    assert torch.equal(x, backup)
    assert torch.isfinite(y).all()
    assert ((y[..., :15]>=0)&(y[..., :15]<=1)).all()
    assert (y[..., 15:17].abs()<=1).all()
    assert (y[..., 17:25]>=0).all()
    torch.testing.assert_close(y[..., 17:25].sum(-1), torch.ones(10,12))
    order = torch.randperm(10)
    torch.testing.assert_close(expand_predictions(x[order], 2), y[order])


def test_unclipped_square_distance_scaling():
    x = valid_predictions()
    x = x.mean(0, keepdim=True) + .01*(x-x.mean(0, keepdim=True))
    y = expand_predictions(x, 1.2)
    before = (x[:,None]-x[None,:]).square().mean()
    after = (y[:,None]-y[None,:]).square().mean()
    torch.testing.assert_close(after, before*1.2**2)
    for scale in (.5, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            expand_predictions(x, scale)


def test_global_scale_calibration():
    x = valid_predictions()
    target = float(compute_s_mse([official_prediction(expand_predictions(x, 1.5))]))
    scale, record = calibrate_spread([x], target)
    assert abs(record['achieved_FRDiv']-target) <= 1e-4
    assert 1 <= scale <= 4
    assert len(record['history']) >= 2
