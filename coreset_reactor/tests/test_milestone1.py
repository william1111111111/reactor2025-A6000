import torch

from coreset_reactor.descriptor import DescriptorScaler, reaction_descriptor
from coreset_reactor.losses import descriptor_distance, descriptor_set_loss, unpaired_softdtw_cost
from coreset_reactor.model import ParallelTCN
from coreset_reactor.sampler import exact_reference_indices


def test_parallel_head_and_channel_contract():
    for stride in (1, 4):
        model = ParallelTCN(control_point_stride=stride)
        out = model(torch.randn(2, 21, 768), torch.randn(2, 21, 25),
                    torch.randn(2, 21, 58))
        assert out.shape == (2, 10, 21, 25)
        assert ((out[..., :15] >= 0) & (out[..., :15] <= 1)).all()
        assert ((out[..., 15:17] >= -1) & (out[..., 15:17] <= 1)).all()
        assert torch.allclose(out[..., 17:].sum(-1), torch.ones(2, 10, 21))
        out.mean().backward()
        assert model.head[1].weight.grad is not None


def test_descriptor_shape_and_coverage_gradients():
    pred = torch.rand(2, 10, 32, 25, requires_grad=True)
    gt = torch.rand(2, 91, 32, 25)
    scaler = DescriptorScaler(torch.zeros(602), torch.ones(602))
    pd = scaler(reaction_descriptor(pred))
    gd = scaler(reaction_descriptor(gt))
    assert pd.shape == (2, 10, 602)
    assert gd.shape == (2, 91, 602)
    distance = descriptor_distance(pd, gd)
    terms = descriptor_set_loss(distance, .1, .5, .05, .35)
    terms["total"].backward()
    assert torch.isfinite(pred.grad).all()
    assert distance.shape == (2, 10, 91)


def test_random_exact_draws_are_nested_and_distinct():
    pool = list(range(95))
    b0 = exact_reference_indices(pool, 0, 10, 7, 3, "speaker/session0/example")
    b1 = exact_reference_indices(pool, 0, 18, 7, 3, "speaker/session0/example")
    assert b0 == b1[:9]
    assert len(set(b1)) == 17
    assert 0 not in b1


def test_sampled_softdtw_gradient():
    raw = torch.randn(1, 2, 24, 25, requires_grad=True)
    pred = ParallelTCN.activate(raw)
    gt = ParallelTCN.activate(torch.randn(1, 3, 24, 25))
    cost = unpaired_softdtw_cost(pred, gt, frames=8)
    assert cost.shape == (1, 2, 3)
    cost.mean().backward()
    assert torch.isfinite(raw.grad).all()
