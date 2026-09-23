import pytest
import torch
from coreset_reactor.relation_distillation import centered_direction_loss, distillation_relation, relation_loss


def tensors():
    torch.manual_seed(173)
    teacher = torch.rand(2, 4, 8, 25)
    student = torch.rand_like(teacher)
    mask = torch.ones(2, 4, 8, dtype=torch.bool)
    mask[1, :, 5:] = False
    return student, teacher, mask


def test_direction_invariance_and_discriminates_wrong_patterns():
    p, t, m = tensors()
    assert centered_direction_loss(t, t, m) < 1e-6
    center = t.mean(1, keepdim=True)
    assert centered_direction_loss(center+3*(t-center), t, m) < 1e-6
    assert centered_direction_loss(2*center-t, t, m) > 1.99
    shift = torch.randn(2, 1, 8, 25)
    before = centered_direction_loss(p, t, m)
    torch.testing.assert_close(centered_direction_loss(p+shift, t, m), before)
    torch.testing.assert_close(centered_direction_loss(p[:,[2,0,3,1]], t[:,[2,0,3,1]], m), before)
    assert centered_direction_loss(t[:,[1,2,3,0]], t, m) > .1


def test_padding_teacher_detach_and_collapse_gradients():
    p, t, m = tensors()
    before = centered_direction_loss(p, t, m)
    p[1, :, 5:] = float('nan')
    t[1, :, 5:] = float('nan')
    p.requires_grad_(); t.requires_grad_()
    after = centered_direction_loss(p, t, m)
    torch.testing.assert_close(before, after)
    after.backward()
    assert t.grad is None
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert p.grad[1,:,5:].abs().sum() == 0
    p = torch.zeros_like(p, requires_grad=True)
    centered_direction_loss(p, t, m).backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_flat_teacher_and_mask_contract_and_old_objective():
    p, t, m = tensors()
    p.requires_grad_()
    flat = torch.ones_like(t)
    loss = centered_direction_loss(p, flat, m)
    assert loss == 0
    loss.backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() == 0
    torch.testing.assert_close(distillation_relation(p,t,m,[1,2,3])[0], relation_loss(p,t,m,[1,2,3])[0])
    m[0,0,-1] = False
    with pytest.raises(ValueError):
        centered_direction_loss(p,t,m)
