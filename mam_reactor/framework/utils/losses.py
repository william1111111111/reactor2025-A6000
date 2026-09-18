from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from framework.motion_diffusion.qbvm import QualityBudgetedVarianceObjective


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class VAELoss(nn.Module):
    def __init__(self, kl_p: float = 0.0002,
                 w_emo: float = None,
                 w_exp: float = None,
                 w_rot: float = None,
                 w_tran: float = None,
                 **kwargs):
        super(VAELoss, self).__init__()
        self.mse = nn.MSELoss(reduce=True, size_average=True)
        self.kl_loss = KLLoss()
        self.kl_p = kl_p

        if w_emo is None:
            w_emo = 1.0
        if w_exp is None:
            w_exp = 1.0
        if w_rot is None:
            w_rot = 10.0
        if w_tran is None:
            w_tran = 10.0
        self.w_emo = w_emo
        self.w_exp = w_exp
        self.w_rot = w_rot
        self.w_tran = w_tran

    def forward(self, gt_emotions, gt_3dmms, pred_emotions, pred_3dmms, distribution):
        """ List
        gt_emotion; gt_3dmm; pred_emotion; pred_3dmm
        """
        bsz = len(gt_emotions)
        rec_emotion_loss = 0
        rec_param_loss = 0
        for gt_emotion, gt_3dmm, pred_emotion, pred_3dmm in zip(
                gt_emotions, gt_3dmms, pred_emotions, pred_3dmms):
            gt_emotion = gt_emotion.to(pred_emotion.get_device())
            gt_3dmm = gt_3dmm.to(pred_3dmm.get_device())

            exp_part = self.w_exp * self.mse(pred_3dmm[:, :52], gt_3dmm[:, :52])  # expression
            rot_part = self.w_rot * self.mse(pred_3dmm[:, 52:55], gt_3dmm[:, 52:55])  # rotation
            tran_part = self.w_tran * self.mse(pred_3dmm[:, 55:], gt_3dmm[:, 55:])  # translation
            rec_param_loss = rec_param_loss + (exp_part + rot_part + tran_part)
            rec_emotion_loss = rec_emotion_loss + self.w_emo * self.mse(pred_emotion, gt_emotion)

        rec_emotion_loss = rec_emotion_loss / bsz
        rec_param_loss = rec_param_loss / bsz
        rec_loss = rec_emotion_loss + rec_param_loss

        mu_ref = torch.zeros_like(distribution[0].loc).to(gt_emotion.get_device())
        scale_ref = torch.ones_like(distribution[0].scale).to(gt_emotion.get_device())
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)

        kld_loss = 0
        for t in range(len(distribution)):
            kld_loss += self.kl_loss(distribution[t], distribution_ref)
        kld_loss = kld_loss / len(distribution)

        loss = rec_loss + self.kl_p * kld_loss
        return loss, rec_loss, rec_emotion_loss, rec_param_loss, kld_loss

    def __repr__(self):
        return "VAELoss()"


def div_loss(Y_1_list, Y_2_list):
    loss = 0.0
    B = len(Y_1_list)
    for y1, y2 in zip(Y_1_list, Y_2_list):
        y1_flat = y1.view(1, -1)
        y2_flat = y2.view(1, -1)
        Y = torch.cat([y1_flat, y2_flat], dim=0)
        dist2 = F.pdist(Y, 2) ** 2
        loss += (-dist2 / 100).exp().mean()
    loss = loss / B
    return loss


def div_loss_v2(Y_1, Y_2):
    loss = 0.0
    b,t,c = Y_1.shape
    Y_g = torch.cat([Y_1.view(b,1,-1), Y_2.view(b,1,-1)], dim = 1)
    for Y in Y_g:
        dist = F.pdist(Y, 2) ** 2
        loss += (-dist /  100).exp().mean()
    loss /= b
    return loss


def TemporalLoss(Y):
    diff = Y[:, 1:, :] - Y[:, :-1, :]
    t_loss = torch.mean(torch.norm(diff, dim=2, p=2) ** 2)
    return t_loss


def L1Loss(prediction, target, reduction="min", **kwargs):
    # prediction has shape of [batch_size, num_preds, features]
    # target has shape of [batch_size, num_preds, features]
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    # manual implementation of L1 loss
    loss = (torch.abs(prediction - target)).mean(dim=-1)

    # reduce across multiple predictions
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(dim=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


def MSELoss(prediction, target, reduction="mean", **kwargs):
    assert len(prediction.shape) == len(target.shape), "prediction and target must have the same shape"
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"

    loss = ((prediction - target) ** 2).mean(dim=-1)

    # reduce across multiple predictions
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "min":
        loss = loss.min(dim=-1)[0].mean()
    else:
        raise NotImplementedError("reduction {} not implemented".format(reduction))
    return loss


def MSELossApt(prediction, target, reduction="mean",
               w_au=1.0, w_va=5.0, w_em=2.0, **kwargs):
    assert len(prediction.shape) == 3, "Only works with predictions of shape [batch_size, num_preds, features]"
    loss_au = F.mse_loss(prediction[:, :, :15], target[:, :, :15], reduction=reduction)
    loss_va = F.mse_loss(prediction[:, :, 15:17], target[:, :, 15:17], reduction=reduction)
    loss_em = F.mse_loss(prediction[:, :, 17:], target[:, :, 17:], reduction=reduction)
    loss = loss_au * w_au + loss_va * w_va + loss_em * w_em

    losses_dict = {"loss": loss, "loss_au": loss_au, "loss_va": loss_va, "loss_em": loss_em}
    return losses_dict


class DiffusionLoss:
    def __init__(self,
                 losses_type='MSELoss',
                 n_preds=10,
                 w_au=1.0,  # action unit
                 w_va=5.0,  # valence and arousal
                 w_em=2.0,  # emotion
                 **kwargs):
        self.loss_type = losses_type
        self.n_preds = n_preds
        self.w_au = w_au
        self.w_va = w_va
        self.w_em = w_em
        self.is_flow_matching = kwargs.get('is_flow_matching', False)
        self.is_3dmm = kwargs.get('is_3dmm', False)
        if not self.is_flow_matching:
            raise ValueError("The paired baseline requires is_flow_matching=True.")

        self.expected_dim = 83 if self.is_3dmm else 25
        self.fm_group_balanced = bool(
            kwargs.get("fm_group_balanced", self.is_3dmm)
        )
        self.fm_group_weights = tuple(
            float(value)
            for value in kwargs.get(
                "fm_group_weights", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            )
        )
        self.fm_robust_3dmm = bool(
            kwargs.get("fm_robust_3dmm", False)
        )
        self.fm_normalize_group_weights = bool(
            kwargs.get("fm_normalize_group_weights", True)
        )
        if len(self.fm_group_weights) != 6:
            raise ValueError("fm_group_weights must contain six values.")
        if sum(self.fm_group_weights) <= 0:
            raise ValueError("fm_group_weights must have positive total weight.")

        self.qbvm_enabled = bool(kwargs.get("qbvm_enabled", False))
        self.qbvm_start_epoch = int(kwargs.get("qbvm_start_epoch", 0))
        self.qbvm_warmup_epochs = int(kwargs.get("qbvm_warmup_epochs", 0))
        self.current_epoch = 0
        self.qbvm = None
        if self.qbvm_enabled:
            if not self.is_3dmm:
                raise ValueError("QBVM is defined for the 83D paired Flow.")
            self.qbvm = QualityBudgetedVarianceObjective(
                variance_weight=float(kwargs.get("qbvm_variance_weight", 0.01)),
                group_weights=kwargs.get(
                    "qbvm_group_weights",
                    [0.30, 0.05, 0.05, 0.45, 0.10, 0.05],
                ),
                group_scales=kwargs.get(
                    "qbvm_group_scales",
                    [0.279849, 0.163874, 0.210920, 0.414361, 0.432518, 0.379946],
                ),
                variance_epsilon=float(
                    kwargs.get("qbvm_variance_epsilon", 1e-4)
                ),
                energy_floor=float(kwargs.get("qbvm_energy_floor", 0.05)),
                velocity_ratio_limit=float(
                    kwargs.get("qbvm_velocity_ratio_limit", 1.20)
                ),
                acceleration_ratio_limit=float(
                    kwargs.get("qbvm_acceleration_ratio_limit", 1.20)
                ),
                range_ratio_limit=float(
                    kwargs.get("qbvm_range_ratio_limit", 1.20)
                ),
                velocity_penalty_weight=float(
                    kwargs.get("qbvm_velocity_penalty_weight", 0.05)
                ),
                acceleration_penalty_weight=float(
                    kwargs.get("qbvm_acceleration_penalty_weight", 0.10)
                ),
                range_penalty_weight=float(
                    kwargs.get("qbvm_range_penalty_weight", 0.05)
                ),
                ccc_minimum=float(kwargs.get("qbvm_ccc_minimum", 0.0)),
                ccc_penalty_weight=float(
                    kwargs.get("qbvm_ccc_penalty_weight", 0.0)
                ),
                use_flow_time_weight=bool(
                    kwargs.get("qbvm_use_flow_time_weight", True)
                ),
            )

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _qbvm_ramp(self):
        if not self.qbvm_enabled or self.current_epoch < self.qbvm_start_epoch:
            return 0.0
        if self.qbvm_warmup_epochs <= 0:
            return 1.0
        elapsed = self.current_epoch - self.qbvm_start_epoch + 1
        return min(1.0, elapsed / self.qbvm_warmup_epochs)

    def __call__(self, output_decoder):
        prediction_emotion = output_decoder["prediction_emotion"]
        target_emotion = output_decoder["target_emotion"]
        if (
                prediction_emotion.shape[-1] != self.expected_dim
                or target_emotion.shape[-1] != self.expected_dim
        ):
            raise ValueError(
                f"Flow{self.expected_dim} loss expects {self.expected_dim} features; "
                f"got {prediction_emotion.shape[-1]} and {target_emotion.shape[-1]}."
            )

        loss_au = F.mse_loss(
            prediction_emotion[..., :15], target_emotion[..., :15]
        )
        loss_va = F.mse_loss(
            prediction_emotion[..., 15:17], target_emotion[..., 15:17]
        )
        loss_emotion = F.mse_loss(
            prediction_emotion[..., 17:25], target_emotion[..., 17:25]
        )

        if self.is_3dmm:
            loss_3dmm = (
                F.smooth_l1_loss
                if self.fm_robust_3dmm
                else F.mse_loss
            )
            loss_expression = loss_3dmm(
                prediction_emotion[..., 25:77],
                target_emotion[..., 25:77],
            )
            loss_rotation = loss_3dmm(
                prediction_emotion[..., 77:80],
                target_emotion[..., 77:80],
            )
            loss_translation = loss_3dmm(
                prediction_emotion[..., 80:83],
                target_emotion[..., 80:83],
            )
            if self.fm_group_balanced:
                component_losses = (
                    loss_au,
                    loss_va,
                    loss_emotion,
                    loss_expression,
                    loss_rotation,
                    loss_translation,
                )
                denominator = (
                    sum(self.fm_group_weights)
                    if self.fm_normalize_group_weights
                    else 1.0
                )
                loss = sum(
                    weight * component_loss
                    for weight, component_loss in zip(
                        self.fm_group_weights, component_losses
                    )
                ) / denominator
            else:
                loss = F.mse_loss(prediction_emotion, target_emotion)
        else:
            # Exact rollback behavior: one unweighted MSE over all 25D.
            loss = F.mse_loss(prediction_emotion, target_emotion)
            loss_expression = loss_au
            loss_rotation = loss.new_zeros(())
            loss_translation = loss.new_zeros(())

        losses_dict = {
            "loss": loss,
            "loss_fm": loss,
            # Preserve the trainer's historical logging key names.
            "loss_em": loss_expression,
            "loss_angle": loss_rotation,
            "loss_trans": loss_translation,
            "loss_au": loss_au,
            "loss_va": loss_va,
            "loss_em_": loss_emotion,
            "loss_fea": loss.new_zeros(()),
        }

        ramp = self._qbvm_ramp()
        if self.qbvm_enabled:
            required = (
                "prediction_clean",
                "target_clean",
                "valid_mask",
                "flow_time",
            )
            missing = [key for key in required if key not in output_decoder]
            if missing:
                raise KeyError(
                    "QBVM requires Flow endpoint metadata: " + ", ".join(missing)
                )
            qbvm_output = self.qbvm(
                prediction=output_decoder["prediction_clean"],
                target=output_decoder["target_clean"],
                valid_mask=output_decoder["valid_mask"],
                flow_time=output_decoder["flow_time"],
            )
            qbvm_loss = ramp * qbvm_output["loss"]
            losses_dict["loss"] = losses_dict["loss"] + qbvm_loss
            losses_dict["loss_qbvm"] = qbvm_loss
            losses_dict["qbvm_ramp"] = loss.new_tensor(ramp)
            for key, value in qbvm_output.items():
                if key == "loss":
                    continue
                losses_dict[f"qbvm_{key}"] = value

        return losses_dict

class EmotionVAELoss:
    def __init__(self, w_au, w_va, w_em, w_kld, **kwargs):
        self.w_au = w_au
        self.w_va = w_va
        self.w_em = w_em
        self.w_kld = w_kld

        self.au_criterion = nn.BCEWithLogitsLoss(reduction="none")  # "mean"
        self.va_criterion = nn.MSELoss(reduction="mean")
        self.em_criterion = nn.KLDivLoss(reduction="none")  # batchmean
        self.kld_criterion = KLLoss()

    def kld_loss(self, mu, logvar, reduction='batchmean'):
        kld_element = 1 + logvar - mu.pow(2) - logvar.exp()
        if reduction == 'sum':
            # sum over B, L, D
            return -0.5 * torch.sum(kld_element)
        elif reduction == 'batchmean':
            # sum over L,D, then mean over batch
            kld = -0.5 * torch.sum(kld_element, dim=(1, 2))  # [B]
            return torch.mean(kld)  # scalar
        elif reduction == 'mean':
            # mean over all elements
            return -0.5 * torch.mean(kld_element)
        else:
            raise ValueError("Unknown reduction")

    def __call__(self, predictions, targets, distribution, mask=None):
        au_logits, va_logits, emotion_logits = predictions
        au_targets, va_targets, emotion_targets = \
            targets[:, :, :15], targets[:, :, 15:17], targets[:, :, 17:]

        au_loss = self.au_criterion(au_logits, au_targets)  # AUs
        va_loss = self.va_criterion(va_logits, va_targets)  # valence and arousal
        emotion_logits = rearrange(F.log_softmax(emotion_logits, dim=-1), "b l d -> (b l) d")
        emotion_targets = rearrange(emotion_targets, "b l d -> (b l) d")
        em_loss = self.em_criterion(emotion_logits, emotion_targets)

        if mask is not None:
            au_mask = mask  # [B, L, 1]
            au_loss = (au_loss * au_mask).sum() / (au_mask.sum() * au_logits.shape[-1])
            em_mask = rearrange(mask, "b l d -> (b l) d")  # [BxL, 1]
            em_loss = (em_loss * em_mask).sum() / em_mask.sum()

        mu_ref = torch.zeros_like(distribution.loc).to(au_logits)
        scale_ref = torch.ones_like(distribution.scale).to(au_logits)
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        kld_loss = self.kld_criterion(distribution, distribution_ref)

        loss = self.w_kld * kld_loss + \
               self.w_au * au_loss + \
               self.w_va * va_loss + \
               self.w_em * em_loss

        return loss, kld_loss, au_loss, va_loss, em_loss
