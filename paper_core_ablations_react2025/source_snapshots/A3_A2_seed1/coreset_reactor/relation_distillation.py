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


def centered_residual_loss(prediction, teacher, mask, scale):
    """Match signed slot contrasts with fixed TRAIN AU/VA/expression scales.

    For slot errors e_k, mean_{i<j} ||e_i-e_j||^2 equals
    2K/(K-1) * mean_k ||e_k-mean(e)||^2. Thus this matches all
    teacher pairwise difference *vectors* in O(K), preserving orientation
    and magnitude. A common reaction shift cancels. It is not repulsion:
    expansion beyond the teacher's contrast is penalized too.
    """
    if prediction.shape != teacher.shape or prediction.ndim != 4 or prediction.shape[-1] != 25:
        raise ValueError('Expected matching [B,K,T,25] predictions and teachers')
    if prediction.shape[1] < 2 or mask.shape != prediction.shape[:3] or mask.dtype != torch.bool:
        raise ValueError('Need >=2 slots and boolean [B,K,T] masks')
    if not torch.equal(mask, mask[:, :1].expand_as(mask)) or not mask.any(-1).all():
        raise ValueError('Residual loss requires the same nonempty valid frames for every slot')
    scale = torch.as_tensor(scale, device=prediction.device, dtype=torch.float32)
    if scale.shape != (3,) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Need three positive fixed TRAIN scales')
    p = prediction.float().masked_fill(~mask[..., None], 0)
    t = teacher.detach().float().masked_fill(~mask[..., None], 0)
    error = (p - p.mean(1, keepdim=True)) - (t - t.mean(1, keepdim=True))
    losses = []
    for group, (start, end) in enumerate(GROUPS):
        per_context = error[..., start:end].square().sum((1, 2, 3))
        per_context = per_context / (mask.sum((1, 2)) * (end - start))
        losses.append(per_context / scale[group].square())
    return torch.stack(losses, -1).mean()


def group_weighted_mse_loss(prediction, teacher, mask, scale):
    """Matched A3 control: scaled AU/VA/EX MSE without slot centering.

    This intentionally retains the same group boundaries, TRAIN-only scales,
    valid-frame normalization and coefficient as centered_residual_loss.  The
    sole formula change is that the per-time slot mean is not subtracted.
    """
    if prediction.shape != teacher.shape or prediction.ndim != 4 or prediction.shape[-1] != 25:
        raise ValueError('Expected matching [B,K,T,25] predictions and teachers')
    if prediction.shape[1] < 2 or mask.shape != prediction.shape[:3] or mask.dtype != torch.bool:
        raise ValueError('Need >=2 slots and boolean [B,K,T] masks')
    if not torch.equal(mask, mask[:, :1].expand_as(mask)) or not mask.any(-1).all():
        raise ValueError('Group-weighted MSE requires the same nonempty valid frames for every slot')
    scale = torch.as_tensor(scale, device=prediction.device, dtype=torch.float32)
    if scale.shape != (3,) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Need three positive fixed TRAIN scales')
    error = (prediction.float() - teacher.detach().float()).masked_fill(
        ~mask[..., None], 0)
    losses = []
    for group, (start, end) in enumerate(GROUPS):
        per_context = error[..., start:end].square().sum((1, 2, 3))
        per_context = per_context / (mask.sum((1, 2)) * (end - start))
        losses.append(per_context / scale[group].square())
    return torch.stack(losses, -1).mean()


def au_decision_loss(prediction, teacher, mask):
    """Confidence-weighted distillation of teacher AU rounding decisions.

    Soft trajectory KD remains the primary loss. Only the teacher defines
    the Bernoulli decisions; no evaluation GT or prediction repulsion enters.
    Teacher probabilities near 0.5 receive little weight. A common mask is
    not required here because this objective is pointwise.
    """
    if prediction.shape != teacher.shape or prediction.ndim != 4 or prediction.shape[-1] != 25:
        raise ValueError('Expected matching [B,K,T,25] predictions and teachers')
    if mask.shape != prediction.shape[:3] or mask.dtype != torch.bool or not mask.any(-1).all():
        raise ValueError('Need nonempty boolean [B,K,T] masks')
    p = prediction[..., :15].float().masked_fill(~mask[..., None], .5)
    t = teacher.detach()[..., :15].float().masked_fill(~mask[..., None], .5)
    confidence = (2*t-1).abs() * mask[..., None]
    loss = torch.nn.functional.binary_cross_entropy(
        p.clamp(1e-5, 1-1e-5), t.round(), reduction='none')
    return ((loss*confidence).sum((1, 2, 3)) /
            confidence.sum((1, 2, 3)).clamp_min(1e-8)).mean()


def au_logit_contrast_loss(prediction, teacher, mask, epsilon=1e-3):
    """Match signed AU log-odds contrasts to finite, detached teacher targets.

    Centering across expert IDs removes the shared logit offset. Huber loss
    bounds sensitivity to saturated teacher probabilities. This does not
    reward arbitrary separation: teacher-matched contrasts minimize the loss.
    The ordinary trajectory and residual KD still supervise the shared mean.
    """
    if prediction.shape != teacher.shape or prediction.ndim != 4 or prediction.shape[-1] != 25:
        raise ValueError('Expected matching [B,K,T,25] predictions and teachers')
    if prediction.shape[1] < 2 or mask.shape != prediction.shape[:3] or mask.dtype != torch.bool:
        raise ValueError('Need >=2 slots and boolean [B,K,T] masks')
    if not torch.equal(mask, mask[:, :1].expand_as(mask)) or not mask.any(-1).all():
        raise ValueError('Logit contrast requires common nonempty valid frames')
    if not 0 < epsilon < .5:
        raise ValueError('epsilon must lie in (0, .5)')
    p = prediction[..., :15].float().masked_fill(~mask[..., None], .5)
    t = teacher.detach()[..., :15].float().masked_fill(~mask[..., None], .5)
    p, t = torch.logit(p, eps=epsilon), torch.logit(t, eps=epsilon)
    p, t = p - p.mean(1, keepdim=True), t - t.mean(1, keepdim=True)
    loss = torch.nn.functional.smooth_l1_loss(p, t, reduction='none')
    return ((loss * mask[..., None]).sum((1, 2, 3)) /
            (mask.sum((1, 2)) * 15)).mean()


def distillation_relation(prediction, teacher, mask, scale, objective='distance'):
    if objective == 'distance':
        return relation_loss(prediction, teacher, mask, scale)
    if objective == 'centered_residual':
        loss = centered_residual_loss(prediction, teacher, mask, scale)
    elif objective == 'group_weighted_mse':
        loss = group_weighted_mse_loss(prediction, teacher, mask, scale)
    elif objective == 'centered_direction':
        loss = centered_direction_loss(prediction, teacher, mask)
    elif objective == 'au_decision':
        loss = au_decision_loss(prediction, teacher, mask)
    else:
        raise ValueError(f'Unknown relation objective: {objective}')
    # Keep the existing absolute-distance diagnostics without an extra graph.
    with torch.no_grad():
        pd = slot_distances(prediction, mask).mean()
        td = slot_distances(teacher, mask).mean()
    return loss, pd, td
