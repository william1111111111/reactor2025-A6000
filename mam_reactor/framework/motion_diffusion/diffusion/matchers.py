"""
Code adapted from:
https://github.com/BarqueroGerman/BeLFusion
"""
from pathlib import Path
import hydra
import torch
import torch.nn as nn
import os
from einops import rearrange
from omegaconf import DictConfig
from hydra.utils import instantiate
from framework.motion_diffusion.diffusion.diffusion_decoder.transformer_denoiser import TransformerDenoiser, \
    lengths_to_mask
from framework.motion_diffusion.diffusion.rnn import LatentEmbedder
from framework.utils.util import from_pretrained_checkpoint, save_checkpoint


class BaseLatentModel(nn.Module):
    def __init__(self, cfg, emb_preprocessing=False, freeze_encoder=True, **kwargs):
        super(BaseLatentModel, self).__init__()
        self.emb_preprocessing = emb_preprocessing
        self.freeze_encoder = freeze_encoder
        def_dtype = torch.get_default_dtype()

        self.audio_encoder = instantiate(cfg.audio_encoder)
        if cfg.latent_embedder is not None:
            self.latent_embedder = instantiate(cfg.latent_embedder)
            model_path = os.path.join(hydra.utils.get_original_cwd(), cfg.latent_embedder.checkpoint_path)
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            self.latent_embedder.load_state_dict(state_dict)
            print(f"Successfully loaded latent embedder from {model_path}")
        else:
            self.latent_embedder = LatentEmbedder()

        if self.freeze_encoder:  # freeze modules
            for para in self.latent_embedder.parameters():
                para.requires_grad = False

        torch.set_default_dtype(def_dtype)
        self.init_params = None

    def deepcopy(self):
        assert self.init_params is not None, "Cannot deepcopy LatentUNetMatcher if init_params is None."
        # I can't deep copy this class. I need to do this trick to make the deepcopy of everything
        model_copy = self.__class__(**self.init_params)
        weights_path = f'weights_temp_{id(model_copy)}.pt'
        torch.save(self.state_dict(), weights_path)
        model_copy.load_state_dict(torch.load(weights_path))
        os.remove(weights_path)
        return model_copy

    def preprocess(self, emb):
        stats = self.embed_emotion_stats
        if stats is None:
            return emb  # when no checkpoint was loaded, there is no stats.

        if "standardize" in self.emb_preprocessing:
            return (emb - stats["mean"]) / torch.sqrt(stats["var"])
        elif "normalize" in self.emb_preprocessing:
            return 2 * (emb - stats["min"]) / (stats["max"] - stats["min"]) - 1
        elif "none" in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")

    def undo_preprocess(self, emb):
        stats = self.embed_emotion_stats
        if stats is None:
            return emb  # when no checkpoint was loaded, there is no stats.

        if "standardize" in self.emb_preprocessing:
            return torch.sqrt(stats["var"]) * emb + stats["mean"]
        elif "normalize" in self.emb_preprocessing:
            return (emb + 1) * (stats["max"] - stats["min"]) / 2 + stats["min"]
        elif "none" in self.emb_preprocessing.lower():
            return emb
        else:
            raise NotImplementedError(f"Error on the embedding preprocessing value: '{self.emb_preprocessing}'")

    def forward(self, pred, timesteps, seq_em):
        raise NotImplementedError("This is an abstract class.")

    # override checkpointing
    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def cuda(self):
        return self.to(torch.device("cuda"))

    # override eval and train
    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()


class DecoderLatentMatcher(BaseLatentModel):
    def __init__(self,
                 conf: DictConfig = None,
                 module_dict_cfg: DictConfig = None,
                 stage: str = 'fit',
                 task: str = 'online',
                 **kwargs):
        cfg = conf.args
        super(DecoderLatentMatcher, self).__init__(
            module_dict_cfg,
            emb_preprocessing=cfg.emb_preprocessing,
            freeze_encoder=cfg.freeze_encoder,
            **kwargs,
        )

        self.stage = stage
        self.task = task
        self.token_len = cfg.token_len
        self.window_size = cfg.get("window_size", 30)
        self.s_ratio = cfg.get("s_ratio", 2)
        self.emotion_dim = cfg.get("nfeats", 25)
        if self.emotion_dim not in (25, 83):
            raise ValueError(
                "The paired Flow matcher supports only 25D facial attributes "
                "or 83D facial attributes plus 3DMM; "
                f"got {self.emotion_dim}."
            )
        self.encode_emotion = cfg.get("encode_emotion", False)
        self.encode_3dmm = cfg.get("encode_3dmm", False)

        self.init_params = {
            "task": task,
            "window_size": self.window_size,
            "encode_emotion": self.encode_emotion,
            "encode_3dmm": self.encode_3dmm,
            "ablation_skip_connection": cfg.get("ablation_skip_connection", True),
            "nfeats": self.emotion_dim,
            "latent_dim": cfg.get("latent_dim", 512),
            "ff_size": cfg.get("ff_size", 1024),
            "num_layers": cfg.get("num_layers", 6),
            "num_heads": cfg.get("num_heads", 4),
            "dropout": cfg.get("dropout", 0.1),
            "normalize_before": cfg.get("normalize_before", False),
            "activation": cfg.get("activation", "gelu"),
            "flip_sin_to_cos": cfg.get("flip_sin_to_cos", True),
            "return_intermediate_dec": cfg.get("return_intermediate_dec", False),
            "position_embedding": cfg.get("position_embedding", "learned"),
            "arch": cfg.get("arch", "trans_enc"),
            "freq_shift": cfg.get("freq_shift", 0),
            "time_encoded_dim": cfg.get("time_encoded_dim", 64),
            "s_audio_dim": cfg.get("s_audio_dim", 768),
            "s_audio_scale": cfg.get("s_audio_scale", cfg.get("latent_dim", 512) ** -0.5),
            "s_emotion_dim": cfg.get("s_emotion_dim", 25),
            "s_3dmm_dim": cfg.get("s_3dmm_dim", 58),
            "concat": cfg.get("concat", "concat_first"),
            "condition_concat": cfg.get("condition_concat", "token_concat"),
            "guidance_scale": cfg.get("guidance_scale", 7.5),
            "fast_guidance_one": cfg.get("fast_guidance_one", False),
            "s_audio_enc_drop_prob": cfg.get("s_audio_enc_drop_prob", 0.2),
            "s_latent_embed_drop_prob": cfg.get("s_latent_embed_drop_prob", 0.2),
            "s_3dmm_enc_drop_prob": cfg.get("s_3dmm_enc_drop_prob", 0.2),
            "s_emotion_enc_drop_prob": cfg.get("s_emotion_enc_drop_prob", 1.0),
            "past_l_emotion_drop_prob": cfg.get("past_l_emotion_drop_prob", 1.0),
        }
        self.use_past_frames = cfg.get("use_past_frames", False)

        self.model = TransformerDenoiser(**self.init_params)

        self.diffusion_type = conf.scheduler.get("diffusion_type", "flow_matching")
        if self.diffusion_type != "flow_matching":
            raise ValueError(
                "Flow25 baseline only supports diffusion_type='flow_matching', "
                f"got {self.diffusion_type!r}."
            )
        scheduler_nfeats = conf.scheduler.get("nfeats", 25)
        if scheduler_nfeats != self.emotion_dim:
            raise ValueError(
                "diffusion decoder and scheduler feature dimensions must "
                f"match, got {self.emotion_dim} and {scheduler_nfeats}."
            )

        from .flow_matching import ConditionalFlowMatching
        self.decoder_diffusion = ConditionalFlowMatching(
            conf.scheduler,
            conf.scheduler.get("num_inference_timesteps", 10),
        )
        print(
            "[DecoderLatentMatcher] Using Conditional Flow Matching "
            f"({self.emotion_dim}D)"
        )
        self.num_preds = conf.scheduler.num_preds

    def _forward(
            self,
            speaker_audio_input=None,
            speaker_emotion_input=None,
            speaker_3dmm_input=None,
            listener_emotion_input=None,
            past_listener_emotion=None,
            motion_length=None,
    ):
        with torch.no_grad():
            s_audio_encodings = self.audio_encoder._encode(speaker_audio_input)
            s_audio_encodings = s_audio_encodings.repeat_interleave(self.num_preds, dim=0)

            # This condition is structurally discarded when its configured
            # drop probability is 1.0. Avoid both the unnecessary 750-step
            # frozen GRUCell pass and its unavailable BF16 CUDA kernel.
            if self.model.s_latent_embed_drop_prob >= 1.0:
                s_latent_embed = None
            else:
                # CUDA GRUCell in the installed PyTorch build has no BF16
                # implementation, so keep the frozen extractor in FP32.
                with torch.autocast(
                        device_type=speaker_emotion_input.device.type,
                        enabled=False,
                ):
                    s_latent_embed = self.latent_embedder.encode(
                        speaker_emotion_input.float()
                    ).unsqueeze(1)
                s_latent_embed = s_latent_embed.repeat_interleave(
                    self.num_preds, dim=0
                )
            # shape: (batch_size * num_preds, 1, ...)

            # s_3dmm_encodings = self.latent_3dmm_embedder.get_encodings(speaker_3dmm_input)
            s_3dmm_encodings = speaker_3dmm_input.repeat_interleave(self.num_preds, dim=0)
            # shape: (bs * num_preds, s_w, ...)

            s_emotion_encodings = speaker_emotion_input.repeat_interleave(self.num_preds, dim=0)
            # shape: (bs * num_preds, s_w, ...)

            past_listener_emotion = past_listener_emotion.repeat_interleave(
                self.num_preds, dim=0) if past_listener_emotion is not None else None
            # shape: (bs * num_preds, l_w, ...)

            motion_length = motion_length.repeat_interleave(
                self.num_preds, dim=0) if motion_length is not None else None

            model_kwargs = {
                "speaker_audio_encodings": s_audio_encodings,
                "speaker_latent_embed": s_latent_embed,
                "speaker_3dmm_encodings": s_3dmm_encodings,
                "speaker_emotion_encodings": s_emotion_encodings,
                "past_listener_emotion": past_listener_emotion,
                "motion_length": motion_length,
            }

        if self.stage == "test":
            bs, l, _ = s_audio_encodings.shape  # bz * num_preds
            with torch.no_grad():
                output = [output for output in self.decoder_diffusion.euler_sample_loop_progressive(
                    matcher=self,
                    model=self.model,
                    model_kwargs=model_kwargs,
                    shape=(bs, self.window_size if self.task == "online" else l, self.emotion_dim),
                )][-1]

            output_listener_emotion = output["sample_enc"]
            output_listener_emotion = rearrange(output_listener_emotion,
                                                "(b n) w d -> b n w d", n=self.num_preds)
            output_whole = {"prediction_emotion": output_listener_emotion}

        else:
            if (
                    listener_emotion_input is None
                    or listener_emotion_input.shape[-1] != self.emotion_dim
            ):
                actual = None if listener_emotion_input is None else listener_emotion_input.shape[-1]
                raise ValueError(
                    "Paired Flow target dimension mismatch: expected "
                    f"{self.emotion_dim}, got {actual}."
                )
            listener_emotion_input = listener_emotion_input.repeat_interleave(self.num_preds, dim=0)
            x_start_selected = listener_emotion_input  # (bs * num_preds, l_w, ...)

            # Standard CFM: t ~ Uniform(0, 1), paired listener endpoint only.
            t = self.decoder_diffusion.sample_timesteps(x_start_selected.shape[0], x_start_selected.device)
            output_whole = self.decoder_diffusion.denoise(
                self.model,
                x_start_selected,
                t,
                model_kwargs=model_kwargs,
            )
            output_mask = None
            if motion_length is not None:  # offline task zero masking
                device = x_start_selected.device
                output_mask = lengths_to_mask(motion_length, device=device, max_len=x_start_selected.shape[1])
                output_whole["prediction_emotion"] = (output_whole["prediction_emotion"]
                                                      * output_mask.float().unsqueeze(-1))

            output_whole = {k: v.view(-1, self.num_preds, *output_whole[k].shape[1:]) for k, v in output_whole.items()}
            if output_mask is None:
                output_mask = torch.ones(
                    x_start_selected.shape[:2],
                    device=x_start_selected.device,
                    dtype=torch.bool,
                )
            output_whole["valid_mask"] = output_mask.view(
                -1, self.num_preds, output_mask.shape[-1]
            )
        return output_whole

    def forward(self, **kwargs):
        return self._forward(**kwargs)


class LatentMatcher(nn.Module):
    def __init__(self,
                 task: str = "online",
                 stage: str = "fit",
                 device: str = None,
                 diffusion_decoder: DictConfig = None,
                 latent_embedder: DictConfig = None,
                 audio_encoder: DictConfig = None,
                 resumed_training: bool = False,
                 test_checkpoint: str = "best",
                 **kwargs):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.task = task
        self.stage = stage
        self.kwargs = kwargs

        module_dict_cfg = DictConfig(
            {"latent_embedder": latent_embedder,
             "audio_encoder": audio_encoder,}
        )

        self.diffusion_decoder_cfg = diffusion_decoder
        nfeats = diffusion_decoder.args.get("nfeats", 25)
        if nfeats not in (25, 83):
            raise ValueError(
                f"Paired Flow supports nfeats=25 or 83, got {nfeats}."
            )
        self.nfeats = nfeats
        self.diffusion_decoder = DecoderLatentMatcher(self.diffusion_decoder_cfg,
                                                      task=task,
                                                      stage=stage,
                                                      module_dict_cfg=module_dict_cfg,
                                                      **kwargs)
        load_ckpt = False
        want_last = False
        want_best = False
        want_epoch = None

        if resumed_training:
            load_ckpt = True
            want_last = True
        if stage == "test":
            load_ckpt = True
            test_checkpoint = str(test_checkpoint).strip().lower()
            if test_checkpoint == "best":
                want_best = True
            elif test_checkpoint == "last":
                want_last = True
            elif test_checkpoint.isdigit() and int(test_checkpoint) > 0:
                want_epoch = int(test_checkpoint)
            else:
                raise ValueError(
                    "test_checkpoint must be 'best', 'last', or a positive "
                    f"epoch number; got {test_checkpoint!r}"
                )

        if load_ckpt:
            ckpt_path = self.get_ckpt_path(
                self.diffusion_decoder.model,
                runid="resume_runid",
                epoch=want_epoch,
                best=want_best,
                last=want_last,
            )
            from_pretrained_checkpoint(str(ckpt_path), self.diffusion_decoder.model, device)

    def forward(
            self,
            speaker_audio_input=None,
            speaker_emotion_input=None,
            speaker_3dmm_input=None,
            listener_emotion_input=None,
            past_listener_emotion=None,
            motion_length=None,
            listener_3dmm_input=None,
            **kwargs,
    ):
        listener_target = listener_emotion_input
        if self.nfeats == 83:
            if listener_emotion_input is None and listener_3dmm_input is None:
                listener_target = None
            elif listener_emotion_input is None or listener_3dmm_input is None:
                raise ValueError(
                    "Flow83 requires both paired listener 25D facial "
                    "attributes and paired listener 58D 3DMM."
                )
            else:
                if listener_emotion_input.shape[:-1] != listener_3dmm_input.shape[:-1]:
                    raise ValueError(
                        "Flow83 listener target shapes are incompatible: "
                        f"{tuple(listener_emotion_input.shape)} vs "
                        f"{tuple(listener_3dmm_input.shape)}."
                    )
                listener_target = torch.cat(
                    (listener_emotion_input, listener_3dmm_input), dim=-1
                )

        outputs = self.diffusion_decoder.forward(
            speaker_audio_input=speaker_audio_input,
            speaker_emotion_input=speaker_emotion_input,
            speaker_3dmm_input=speaker_3dmm_input,
            listener_emotion_input=listener_target,
            past_listener_emotion=past_listener_emotion,
            motion_length=motion_length,
        )
        # outputs['prediction_emotion']: (bz, num_preds, s_w, emotion_dim)
        return outputs

    def get_ckpt_path(self, model, runid="current_runid", epoch=None, best=False, last=False):
        ckpt_dir = Path(hydra.utils.to_absolute_path(self.kwargs.get("ckpt_dir")))
        run_id = Path(self.kwargs.get(runid))
        ckpt_dir = str(ckpt_dir / run_id / model.get_model_name())
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_path = None
        if epoch is not None:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{epoch}.pth")
        if best:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint_best.pth")
        if last:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint_last.pth")
        assert ckpt_path is not None, "No checkpoint path is provided."
        return ckpt_path

    def save_ckpt(self, optimizer, epoch=None, best=False, last=False, best_loss=float("inf")):
        model = self.diffusion_decoder.model
        # torch.compile wraps the denoiser and prefixes state-dict keys with
        # `_orig_mod.`. Save the underlying module so compiled training
        # checkpoints remain loadable by both eager and compiled inference.
        checkpoint_model = getattr(model, "_orig_mod", model)
        ckpt_path = self.get_ckpt_path(
            checkpoint_model, epoch=epoch, best=best, last=last
        )
        save_checkpoint(
            ckpt_path,
            checkpoint_model,
            optimizer,
            epoch=epoch,
            best_loss=best_loss,
        )
