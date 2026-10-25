import pytest
import torch
from coreset_reactor.relation_distillation import relation_loss, slot_distances
from coreset_reactor.relation_distillation import (
    GROUPS,
    au_decision_loss,
    centered_residual_loss,
    group_weighted_mse_loss,
)
from coreset_reactor.relation_distillation import au_logit_contrast_loss


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


def test_centered_residual_equals_pairwise_vector_error():
    torch.manual_seed(25)
    p, t = torch.randn(2, 5, 9, 25), torch.randn(2, 5, 9, 25)
    mask = torch.ones(2, 5, 9, dtype=torch.bool)
    mask[1, :, 6:] = False
    scale = torch.tensor([0.3, 0.8, 0.2])
    actual = centered_residual_loss(p, t, mask, scale)
    expected = []
    for sample, length in enumerate((9, 6)):
        for g, (start, end) in enumerate(GROUPS):
            pairs = []
            for i in range(5):
                for j in range(i + 1, 5):
                    error = (p[sample, i] - p[sample, j]) - (t[sample, i] - t[sample, j])
                    pairs.append(error[:length, start:end].square().mean())
            expected.append(torch.stack(pairs).mean() * 4 / 10 / scale[g].square())
    torch.testing.assert_close(actual, torch.stack(expected).mean())


def test_centered_residual_shift_padding_direction_and_gradients():
    torch.manual_seed(26)
    teacher = torch.rand(2, 4, 8, 25, requires_grad=True)
    mask = torch.ones(2, 4, 8, dtype=torch.bool)
    mask[1, :, 5:] = False
    shifted = teacher.detach() + torch.rand(2, 1, 8, 25)
    assert centered_residual_loss(shifted, teacher, mask, [1, 1, 1]) < 1e-12
    reflected = 2 * teacher.detach().mean(1, keepdim=True) - teacher.detach()
    # Reflection preserves pairwise distances but reverses teacher behavior.
    assert relation_loss(reflected, teacher, mask, [1, 1, 1])[0] < 1e-6
    assert centered_residual_loss(reflected, teacher, mask, [1, 1, 1]) > .01
    student = (teacher.detach() * .5).requires_grad_()
    original = centered_residual_loss(student, teacher, mask, [1, 1, 1])
    padded = student.detach().clone()
    padded[1, :, 5:] = float('nan')
    torch.testing.assert_close(original, centered_residual_loss(padded, teacher, mask, [1, 1, 1]))
    original.backward()
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all() and student.grad.abs().sum() > 0
    assert not student.grad[1, :, 5:].any()


def test_group_weighted_mse_is_same_scaled_formula_without_centering():
    torch.manual_seed(9)
    teacher = torch.randn(2, 4, 5, 25, requires_grad=True)
    student = (teacher.detach() + torch.randn_like(teacher) * .2).requires_grad_()
    mask = torch.ones(2, 4, 5, dtype=torch.bool)
    mask[1, :, 3:] = False
    scale = torch.tensor([.5, .75, 1.25])
    actual = group_weighted_mse_loss(student, teacher, mask, scale)
    manual = []
    error = student - teacher.detach()
    for group, (start, end) in enumerate(GROUPS):
        value = (error[..., start:end].square() * mask[..., None]).sum((1, 2, 3))
        value = value / (mask.sum((1, 2)) * (end - start))
        manual.append(value / scale[group].square())
    torch.testing.assert_close(actual, torch.stack(manual, -1).mean())
    actual.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None


def test_group_weighted_mse_retains_common_slot_shift():
    teacher = torch.zeros(1, 3, 4, 25)
    mask = torch.ones(1, 3, 4, dtype=torch.bool)
    shifted = teacher + .25
    assert group_weighted_mse_loss(shifted, teacher, mask, [1, 1, 1]) > .01
    assert centered_residual_loss(shifted, teacher, mask, [1, 1, 1]) < 1e-12


def test_au_decision_gradient_and_mask():
    teacher = torch.full((1, 2, 6, 25), .8, requires_grad=True)
    teacher.data[:, 0, :, :15] = .2
    mask = torch.ones(1, 2, 6, dtype=torch.bool)
    mask[:, :, 4:] = False
    student = torch.full_like(teacher, .5, requires_grad=True)
    loss = au_decision_loss(student, teacher, mask)
    loss.backward()
    assert teacher.grad is None
    assert (student.grad[:, 0, :4, :15] > 0).all()
    assert (student.grad[:, 1, :4, :15] < 0).all()
    assert not student.grad[..., 15:].any()
    assert not student.grad[:, :, 4:].any()
    padded = student.detach().clone()
    padded[:, :, 4:] = float('nan')
    torch.testing.assert_close(loss, au_decision_loss(padded, teacher, mask))
    uncertain = torch.full_like(teacher, .5)
    assert au_decision_loss(student, uncertain, mask) == 0


def test_au_logit_contrast_common_shift_and_finite_target():
    torch.manual_seed(71)
    logits = torch.randn(2, 4, 6, 25) * .5
    teacher = logits.sigmoid()
    mask = torch.ones(2, 4, 6, dtype=torch.bool)
    shifted = (logits + .3).sigmoid()
    assert au_logit_contrast_loss(shifted, teacher, mask) < 1e-12
    assert au_logit_contrast_loss(teacher, teacher, mask) == 0
    assert au_logit_contrast_loss((logits*2).sigmoid(), teacher, mask) > .001
    # Reflecting contrasts preserves their distances, not their teacher identity.
    assert au_logit_contrast_loss((-logits).sigmoid(), teacher, mask) > .01


def test_au_logit_contrast_mask_and_gradients():
    torch.manual_seed(72)
    teacher = torch.rand(2, 4, 6, 25, requires_grad=True)
    prediction = (.1 + .8 * torch.rand_like(teacher)).requires_grad_()
    mask = torch.ones(2, 4, 6, dtype=torch.bool)
    mask[1, :, 3:] = False
    loss = au_logit_contrast_loss(prediction, teacher, mask)
    padded = prediction.detach().clone()
    padded[1, :, 3:] = float('nan')
    torch.testing.assert_close(loss, au_logit_contrast_loss(padded, teacher, mask))
    loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[..., :15].abs().sum() > 0
    assert not prediction.grad[..., 15:].any()
    assert not prediction.grad[1, :, 3:].any()
    mask[0, 0, 0] = False
    with pytest.raises(ValueError):
        au_logit_contrast_loss(prediction, teacher, mask)
