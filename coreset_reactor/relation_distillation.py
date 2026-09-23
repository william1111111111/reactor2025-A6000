"""Preserve teacher slot distances, measured on valid soft trajectories."""
import torch

GROUPS = ((0, 15), (15, 17), (17, 25))


def slot_distances(prediction, mask):
    """[B,K,T,25] -> [B,K*(K-1)/2,3] RMS AU/VA/expression distances.

    The teacher defines a finite target distance; excessive separation is
    penalized as well as insufficient separation. No AU rounding in training.
    """
    if prediction.ndim != 4 or prediction.shape[-1] != 25:
        raise ValueError('Expected [B,K,T,25]')
    if mask.shape != prediction.shape[:3] or mask.dtype != torch.bool:
        raise ValueError('Expected boolean [B,K,T] mask')
    if prediction.shape[1] < 2 or not mask.any(-1).all():
        raise ValueError('Need at least two slots with valid frames')
    a, b = torch.triu_indices(prediction.shape[1], prediction.shape[1], 1,
                             device=prediction.device)
    valid = mask[:, a] & mask[:, b]
    if not valid.any(-1).all():
        raise ValueError('Slot pairs need overlapping valid frames')
    # Mask before subtraction so arbitrary/NaN padding cannot enter the loss.
    x = prediction.float().masked_fill(~mask[..., None], 0)
    delta = (x[:, a] - x[:, b]).masked_fill(~valid[..., None], 0)
    result = []
    for start, end in GROUPS:
        mse = delta[..., start:end].square().sum((-1, -2))
        mse = mse / (valid.sum(-1) * (end - start))
        result.append(torch.sqrt(mse + 1e-8))
    return torch.stack(result, -1)


def relation_loss(prediction, teacher, mask, scale):
    if prediction.shape != teacher.shape:
        raise ValueError('Teacher/student shapes differ')
    scale = torch.as_tensor(scale, device=prediction.device, dtype=torch.float32)
    if scale.shape != (3,) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Need three positive fixed TRAIN scales')
    student_distance = slot_distances(prediction, mask)
    teacher_distance = slot_distances(teacher.detach(), mask).detach()
    loss = ((student_distance - teacher_distance).abs() / scale).mean()
    return loss, student_distance.mean().detach(), teacher_distance.mean().detach()


def centered_direction_loss(prediction, teacher, mask, epsilon=1e-8):
    """Compare teacher-aligned slot residual directions, not their amplitude.

    Each AU/VA/expression group is centered across slots at each valid frame.
    Cosine similarity spans K x valid-time x group-channels per context.
    A shared time-varying reaction cancels; positive groupwise residual scaling
    leaves the loss unchanged away from the numerical floor. Slot IDs retain
    their teacher correspondence: this is NOT permutation-free set matching.
    """
    if prediction.shape != teacher.shape or prediction.ndim != 4 or prediction.shape[-1] != 25:
        raise ValueError('Expected matching [B,K,T,25] predictions and teachers')
    if prediction.shape[1] < 2 or mask.shape != prediction.shape[:3] or mask.dtype != torch.bool:
        raise ValueError('Need >=2 slots and boolean [B,K,T] masks')
    if not torch.equal(mask, mask[:, :1].expand_as(mask)) or not mask.any(-1).all():
        raise ValueError('Direction loss requires the same nonempty valid frames for every slot')
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    valid = mask[..., None]
    student = prediction.float().masked_fill(~valid, 0)
    target = teacher.detach().float().masked_fill(~valid, 0)
    student = student-student.mean(1, keepdim=True)
    target = target-target.mean(1, keepdim=True)
    losses, usable = [], []
    for start, end in GROUPS:
        p, t = student[..., start:end], target[..., start:end]
        count = mask.sum((1, 2))*(end-start)
        pe = p.square().sum((1, 2, 3))/count
        te = t.square().sum((1, 2, 3))/count
        cross = (p*t).sum((1, 2, 3))/count
        cosine = cross/(pe.clamp_min(epsilon)*te.clamp_min(epsilon)).sqrt()
        losses.append(1-cosine.clamp(-1, 1))
        usable.append(te > epsilon)
    loss = torch.stack(losses, -1)
    weight = torch.stack(usable, -1).to(loss)
    return (loss*weight).sum()/weight.sum().clamp_min(1)


def distillation_relation(prediction, teacher, mask, scale, objective='distance'):
    if objective == 'distance':
        return relation_loss(prediction, teacher, mask, scale)
    if objective != 'centered_direction':
        raise ValueError(f'Unknown relation objective: {objective}')
    loss = centered_direction_loss(prediction, teacher, mask)
    # Keep the existing absolute-distance diagnostics without an extra graph.
    with torch.no_grad():
        pd = slot_distances(prediction, mask).mean()
        td = slot_distances(teacher, mask).mean()
    return loss, pd, td
