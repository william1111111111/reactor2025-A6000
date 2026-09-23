import pytest
import torch
from coreset_reactor.relation_distillation import relation_loss, slot_distances


def test_matched_teacher_distances_and_overshoot():
    torch.manual_seed(14)
    teacher = torch.rand(2, 4, 8, 25, requires_grad=True)
    mask = torch.ones(2, 4, 8, dtype=torch.bool)
    matched = relation_loss(teacher, teacher, mask, [1, 1, 1])[0]
    assert matched == 0
    contracted = (teacher.detach() * .5).requires_grad_()
    loss = relation_loss(contracted, teacher, mask, [1, 1, 1])[0]
    loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(contracted.grad).all() and contracted.grad.abs().sum() > 0
    assert relation_loss(teacher * 2, teacher, mask, [1, 1, 1])[0] > 0


def test_padding_and_slot_permutation_invariance():
    torch.manual_seed(12)
    student, teacher = torch.rand(2, 4, 8, 25), torch.rand(2, 4, 8, 25)
    mask = torch.ones(2, 4, 8, dtype=torch.bool)
    mask[1, :, 5:] = False
    before = relation_loss(student, teacher, mask, [1, 2, 3])[0]
    student[1, :, 5:] = float('nan')
    teacher[1, :, 5:] = 1234
    assert torch.equal(before, relation_loss(student, teacher, mask, [1, 2, 3])[0])
    order = [2, 0, 3, 1]
    assert torch.allclose(before, relation_loss(student[:, order], teacher[:, order],
                                               mask[:, order], [1, 2, 3])[0])
    with pytest.raises(ValueError):
        slot_distances(student, torch.zeros_like(mask))
