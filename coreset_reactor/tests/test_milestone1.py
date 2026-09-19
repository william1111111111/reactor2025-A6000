import random

import numpy as np
import torch

from coreset_reactor.dataset import (CACHE_VERSION, build_train_cache,
                                     fixed_reaction, full_reaction_descriptor)
from coreset_reactor.descriptor import DescriptorScaler, reaction_descriptor
from coreset_reactor.evaluate import (_deterministic_postprocess,
                                      exact_quality_metrics,
                                      official_prediction)
from coreset_reactor.losses import descriptor_distance, descriptor_set_loss, unpaired_softdtw_cost
from coreset_reactor.model import ParallelTCN
from coreset_reactor.sampler import exact_reference_indices
from coreset_reactor.train import descriptor_weight, milestone1_criteria


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


def test_official_au_rounding_keeps_continuous_diagnostics():
    pred = torch.full((2, 13, 25), .25)
    pred[1, :, :15] = .75
    before = pred.clone()
    official = official_prediction(pred)
    assert torch.equal(official[0, :, :15], torch.zeros_like(official[0, :, :15]))
    assert torch.equal(official[1, :, :15], torch.ones_like(official[1, :, :15]))
    assert torch.equal(official[..., 15:], pred[..., 15:])
    assert torch.equal(pred, before)


def test_serial_quality_metrics_match_official_per_context_formulas():
    from framework.metrics.FRC import _func as frc_one
    from framework.metrics.FRD import _func as frd_one
    torch.manual_seed(19)
    predictions = [torch.rand(2, 8, 25), torch.rand(2, 7, 25)]
    targets = [torch.rand(2, 8, 25), torch.rand(2, 7, 25)]
    frc, frd = exact_quality_metrics(predictions, targets, workers=1)
    assert np.isclose(frc, np.mean([frc_one(target, pred)
                                    for pred, target in zip(predictions, targets)]))
    assert np.isclose(frd, np.mean([frd_one(target, pred)
                                    for pred, target in zip(predictions, targets)]))


def test_numpy_ipc_quality_metrics_match_serial_formulas():
    torch.manual_seed(23)
    predictions = [torch.rand(2, 8, 25), torch.rand(2, 7, 25)]
    targets = [torch.rand(2, 8, 25), torch.rand(2, 7, 25)]
    serial = exact_quality_metrics(predictions, targets, workers=1)
    parallel = exact_quality_metrics(predictions, targets, workers=2)
    assert np.allclose(parallel, serial)


def test_postprocessor_rng_is_deterministic_and_restored():
    class StochasticProcessor:
        def forward(self, predictions, targets):
            value = random.random() + np.random.rand() + float(torch.rand(()))
            return [torch.full_like(predictions[0], value)]

    random.seed(29)
    np.random.seed(29)
    torch.manual_seed(29)
    expected_next = (random.random(), np.random.rand(), float(torch.rand(())))
    random.seed(29)
    np.random.seed(29)
    torch.manual_seed(29)
    pred = torch.zeros(2, 4, 25)
    args = (StochasticProcessor(), pred, [pred[0]], torch.device("cpu"),
            1234, "speaker/session/example")
    first = _deterministic_postprocess(*args)
    second = _deterministic_postprocess(*args)
    observed_next = (random.random(), np.random.rand(), float(torch.rand(())))
    assert torch.equal(first, second)
    assert np.allclose(observed_next, expected_next)


def test_train_cache_describes_full_raw_gt(tmp_path):
    raw = torch.zeros(48, 25)
    raw[:8, 0] = 1
    raw[-8:, 0] = 1
    facial = tmp_path / "train" / "facial-attributes" / "listener" / "session0"
    facial.mkdir(parents=True)
    np.save(facial / "example.npy", raw.numpy())
    state = build_train_cache(tmp_path, tmp_path / "cache.pt", frames=32)
    assert state["version"] == CACHE_VERSION
    assert state["descriptor_source"] == "full_raw_listener_gt"
    assert torch.allclose(state["descriptor_mean"], full_reaction_descriptor(raw))
    assert not torch.allclose(state["descriptor_mean"],
                              reaction_descriptor(fixed_reaction(raw, 32)))


def test_short_source_ignores_padded_input():
    torch.manual_seed(7)
    model = ParallelTCN(dropout=0).eval()
    audio = torch.randn(2, 21, 768)
    emotion = torch.randn(2, 21, 25)
    face = torch.randn(2, 21, 58)
    padded = model(audio, emotion, face, lengths=torch.tensor([13, 21]))
    unpadded = model(audio[:1, :13], emotion[:1, :13], face[:1, :13])
    assert torch.allclose(padded[:1, :, :13], unpadded, atol=1e-6)
    assert torch.equal(padded[0, :, 13:], torch.zeros_like(padded[0, :, 13:]))


def test_softdtw_ignores_padded_prediction_tail():
    torch.manual_seed(8)
    pred = torch.rand(1, 2, 24, 25)
    gt = torch.rand(1, 3, 24, 25)
    masked = unpaired_softdtw_cost(pred, gt, frames=8,
                                   prediction_lengths=torch.tensor([13]))
    unpadded = unpaired_softdtw_cost(pred[:, :, :13], gt, frames=8)
    assert torch.allclose(masked, unpadded, atol=1e-6)


def test_milestone1_requires_positive_frdiv_gain():
    b1 = {"gt_descriptor_coverage": 1.0, "FRC": .8,
          "exact_FRD": 150.0, "FRDiv": .15}
    b3 = {"gt_descriptor_coverage": .9, "FRC": .8,
          "exact_FRD": 149.0, "FRDiv": .14}
    assert not all(milestone1_criteria(b1, b3).values())
    b3["FRDiv"] = .16
    assert all(milestone1_criteria(b1, b3).values())


def test_descriptor_weight_warmup_and_ramp():
    schedule = {"kind": "linear", "warmup_fraction": .2,
                "ramp_end_fraction": .6}
    assert descriptor_weight(1, 600, .15, schedule) == 0
    assert descriptor_weight(120, 600, .15, schedule) == 0
    assert abs(descriptor_weight(240, 600, .15, schedule) - .075) < 1e-12
    assert descriptor_weight(360, 600, .15, schedule) == .15
    assert descriptor_weight(600, 600, .15, schedule) == .15
    assert descriptor_weight(120, 600, .1) == .1
