"""
Conditional Flow Matching (CFM) implementation.
Based on: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.

Provides the same interface as DecoderLatentDiffusion for drop-in replacement.

Data centering: x_0 is zero-centered (subtract mean) before FM operations,
then de-centered after sampling. This helps training stability when
some feature dimensions have large non-zero means.
We only subtract mean (not divide by std) to avoid amplifying near-zero-std dims.
"""

import hashlib
import os
import numpy as np
import torch
import torch.nn as nn


class ConditionalFlowMatching:
    """
    Conditional Flow Matching with optimal transport conditional paths.

    Uses linear interpolation path: x_t = (1 - (1 - sigma_min) * t) * x_0_c + t * noise
    Target velocity: v = noise - (1 - sigma_min) * x_0_c
    where x_0_c = x_0 - data_mean (zero-centered)
    """

    def __init__(self, cfg, num_inference_steps=10):
        self.sigma_min = cfg.get("sigma_min", 1e-4)
        self.num_inference_steps = num_inference_steps
        self.noise_std = cfg.get("noise_std", 1)
        self.k = cfg.num_preds

        # Load data mean for zero-centering. New experiments may pin an
        # explicit, versioned artifact so legacy checkpoints keep using the
        # centering convention they were trained with.
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        nfeats = cfg.get("nfeats", 25)
        configured_mean_path = cfg.get("data_mean_path", None)
        if configured_mean_path:
            mean_path = os.fspath(configured_mean_path)
            if not os.path.isabs(mean_path):
                mean_path = os.path.join(project_root, mean_path)
            if not os.path.exists(mean_path):
                raise FileNotFoundError(
                    f"Configured Flow data mean does not exist: {mean_path}"
                )
        else:
            # Legacy fallback is intentionally retained for old checkpoints.
            mean_path = os.path.join(project_root, f"data_mean_{nfeats}.npy")
            if not os.path.exists(mean_path):
                mean_path = os.path.join(project_root, "data_mean.npy")

        if os.path.exists(mean_path):
            data_mean = np.load(mean_path)
            if data_mean.shape != (nfeats,):
                raise ValueError(
                    f"Expected Flow mean shape ({nfeats},), got "
                    f"{data_mean.shape} from {mean_path}"
                )
            if not np.isfinite(data_mean).all():
                raise ValueError(f"Flow mean contains NaN/Inf: {mean_path}")
            self.data_mean = torch.from_numpy(
                data_mean.astype(np.float32, copy=False)
            )
            with open(mean_path, "rb") as handle:
                mean_sha256 = hashlib.sha256(handle.read()).hexdigest()
            self.data_mean_path = mean_path
            self.data_mean_sha256 = mean_sha256
            self.use_centering = True
            print(f"[FlowMatching] Loaded data mean from {mean_path} "
                  f"(shape={self.data_mean.shape}, "
                  f"sha256={mean_sha256[:12]}) for zero-centering")
        else:
            self.data_mean = None
            self.data_mean_path = None
            self.data_mean_sha256 = None
            self.use_centering = False
            print("[FlowMatching] WARNING: No data mean file found, "
                  "running without centering")

    def _center(self, x):
        """Zero-center: x_c = x - mean."""
        if not self.use_centering:
            return x
        return x - self.data_mean.to(device=x.device, dtype=x.dtype)

    def _decenter(self, x):
        """Reverse centering: x = x_c + mean."""
        if not self.use_centering:
            return x
        return x + self.data_mean.to(device=x.device, dtype=x.dtype)

    def forward_process(self, x_0, t, noise=None):
        """
        Compute noisy sample and target velocity via linear interpolation.
        x_0 is first zero-centered.

        x_t = (1 - (1 - sigma_min) * t) * x_0_c + t * noise
        target_v = noise - (1 - sigma_min) * x_0_c

        Args:
            x_0: clean data (original space), shape (bs, seq_len, feat_dim)
            t: continuous timestep in [0, 1], shape (bs,)
            noise: optional pre-generated noise

        Returns:
            x_t: noisy sample (in centered space)
            target_v: target velocity field (in centered space)
        """
        # Zero-center x_0
        x_0_c = self._center(x_0)

        if noise is None:
            noise = torch.randn_like(x_0_c) * self.noise_std

        # Expand t for broadcasting: (bs,) -> (bs, 1, 1)
        t_expand = t[:, None, None]

        alpha = 1.0 - (1.0 - self.sigma_min) * t_expand
        x_t = alpha * x_0_c + t_expand * noise

        target_v = noise - (1.0 - self.sigma_min) * x_0_c

        return x_t, target_v

    def sample_timesteps(self, batch_size, device):
        """Sample t ~ U(0, 1) for training."""
        return torch.rand(batch_size, device=device)

    def denoise(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Training interface - matches DecoderLatentDiffusion.denoise() return format.

        Args:
            model: denoiser network
            x_start: clean data (original space) (bs, seq_len, feat_dim)
            t: continuous timesteps in [0, 1], shape (bs,)
            model_kwargs: conditioning dict
            noise: optional noise

        Returns:
            dict with velocity prediction/target in centered space, plus the
            differentiable predicted-clean endpoint in original data space.
        """
        if model_kwargs is None:
            model_kwargs = {}

        x_t, target_v = self.forward_process(x_start, t, noise)

        # Model forward: x_t is in centered space
        model_output = model(x_t, t, model_kwargs)
        t_expand = t[:, None, None]
        predicted_clean_centered = x_t - t_expand * model_output
        predicted_clean = self._decenter(predicted_clean_centered)

        results = {
            "prediction_emotion": model_output,
            "target_emotion": target_v,
            "prediction_clean": predicted_clean,
            "target_clean": x_start,
            # Keep the same leading shape as other outputs so the matcher can
            # reshape every training tensor from [B*K,T,*] to [B,K,T,*].
            "flow_time": t_expand.expand(-1, x_start.shape[1], 1),
        }
        return results

    def euler_sample_loop_progressive(
            self,
            matcher,
            model,
            noise=None,
            model_kwargs=None,
            device=None,
            shape=None,
            **kwargs,
    ):
        """
        Euler ODE solver for sampling - matches ddim_sample_loop_progressive interface.

        Integrates from t=1 (noise) to t=0 (clean data) using Euler steps.
        Output is de-centered back to original data space.

        Args:
            matcher: the matcher (for compatibility)
            model: denoiser network with forward_with_cond_scale
            noise: optional starting noise
            model_kwargs: conditioning dict
            device: target device
            shape: output shape (bs, seq_len, feat_dim)

        Yields:
            dict with "sample_enc" and "decoded_prediction" (in original space)
        """
        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs = model_kwargs.copy()

        if device is None:
            device = next(model.parameters()).device

        assert isinstance(shape, (tuple, list))

        if noise is not None:
            x = noise
        else:
            x = torch.randn(size=shape, device=device) * self.noise_std

        dt = 1.0 / self.num_inference_steps

        for i in range(self.num_inference_steps):
            t_val = 1.0 - i * dt
            t_batch = torch.full((shape[0],), t_val, device=device, dtype=x.dtype)

            with torch.no_grad():
                # Get velocity prediction with classifier-free guidance
                v = model.forward_with_cond_scale(x, t_batch, model_kwargs)

            # Euler step: x_{t-dt} = x_t - dt * v(x_t, t)
            x = x - dt * v

        # De-center back to original data space
        x = self._decenter(x)

        out = {
            "decoded_prediction": x,
            "sample_enc": x,
        }
        yield out
