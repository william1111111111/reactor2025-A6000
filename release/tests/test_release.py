import json
from pathlib import Path

import pytest
import torch

from coreset_reactor.inference import (DEFAULT_CHECKPOINT, center_weights,
    load_student, paper_postprocess, prepare_conditions, window_starts)
from coreset_reactor.prepare_references import paired_farthest_indices
from coreset_reactor.relation_distillation import centered_residual_loss

torch.set_num_threads(2)


def test_main_weight_forward_masks_and_slots():
    model, metadata = load_student()
    assert sum(p.numel() for p in model.parameters()) == 3803201
    assert metadata['source_checkpoint_sha256'] == '24c53d5af11eb8df8b4100cb0a4c027a0ea6a899b289dac8e1cbddb56ca6da4d'
    torch.manual_seed(1)
    inputs = [torch.randn(2, 17, d) for d in [768, 25, 58]]
    lengths = torch.tensor([17, 9])
    with torch.inference_mode():
        output = model.generate(*inputs, lengths)
        single = model(*inputs, lengths, torch.tensor([2, 2]))
    y = output['prediction']
    assert y.shape == (2, 10, 17, 25)
    assert torch.isfinite(y).all()
    assert not y[1, :, 9:].any()
    assert ((y[..., :15] >= 0) & (y[..., :15] <= 1)).all()
    assert (y[..., 15:17].abs() <= 1).all()
    torch.testing.assert_close(y[0, :, :, 17:].sum(-1), torch.ones(10, 17))
    torch.testing.assert_close(single['prediction'], y[:, 2], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('length', [1, 17, 750, 751, 1500, 1701])
def test_full_timeline_has_positive_fusion_weights(length):
    starts = window_starts(length)
    denominator = torch.zeros(length)
    weight = center_weights(750).pow(4)
    for start in starts:
        valid = min(750, length - start)
        denominator[start:start + valid] += weight[:valid]
    assert (denominator > 0).all()
    assert starts[-1] == max(0, length - 750)


def test_centered_loss_common_shift_and_target_teacher_gradient():
    torch.manual_seed(2)
    target = torch.randn(2, 10, 8, 25, requires_grad=True)
    prediction = target.detach().clone().requires_grad_()
    mask = torch.ones(2, 10, 8, dtype=torch.bool)
    scales = [0.3, 0.2, 0.4]
    shift = torch.randn(2, 1, 8, 25)
    loss = centered_residual_loss(prediction + shift, target, mask, scales)
    assert loss < 1e-10
    loss.backward()
    assert target.grad is None
    assert prediction.grad is not None


def test_input_alignment_and_normalization():
    streams = prepare_conditions(torch.ones(9, 768), torch.ones(8, 25),
                                 torch.full((10, 1, 58), 3.0), torch.ones(58), torch.full((58,), 2.0))
    assert [len(v) for v in streams] == [8, 8, 8]
    assert torch.equal(streams[2], torch.ones(8, 58))
    with pytest.raises(ValueError):
        prepare_conditions(torch.ones(0, 768), torch.ones(8, 25), torch.ones(8, 58),
                           torch.ones(58), torch.ones(58))


def test_reference_selection_keeps_pair_and_unique_members():
    import numpy as np
    values = np.arange(15, dtype=float)
    distance = (values[:, None] - values[None, :]) ** 2
    selected = paired_farthest_indices(distance, 7)
    assert selected[0] == 7 and len(set(selected)) == len(selected) == 10


def test_frozen_mapping_both_directions():
    path = DEFAULT_CHECKPOINT.parents[1] / 'configs/react2025_reference_assignment.json'
    value = json.loads(path.read_text())
    assert len(value['mapping']) == 3320
    roles = set()
    for source, targets in value['mapping'].items():
        role, session, basename = source.split('/')
        roles.add(role)
        opposite = 'listener' if role == 'speaker' else 'speaker'
        assert targets[0] == f'{opposite}/{session}/{basename}'
        assert len(set(targets)) == 10
        assert all(t.startswith(f'{opposite}/{session}/') for t in targets)
    assert roles == {'speaker', 'listener'}


def test_paper_postprocess_preserves_soft_input():
    soft = torch.full((10, 120, 25), .2)
    soft[..., 15:17] = -.5
    soft[..., 17:] = 1 / 8
    before = soft.clone()
    face = torch.arange(120, dtype=torch.float32)[:, None].expand(-1, 25)
    result = paper_postprocess(soft, face)
    assert torch.equal(soft, before)
    assert (result[0, :, :15] == 1).sum() == 15
    assert (result[1:, :, :15] == 0).all()
    assert torch.equal(result[..., 15:], soft[..., 15:])
