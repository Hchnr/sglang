# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from vllm-omni-pwm d3e9aad33e861b69be16c4f341695e1cc12b3314.
"""Native PWM WAM inference with request-owned joint flow state.

Adapted from vLLM-Omni's PWM runtime. Only immutable model components live on the
runtime; noise, conditioning, CFG branches and UniPC histories belong to each
call. VIDEO and ACTION always run jointly, including action-only presentation.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.multimodal_gen.runtime.models.dits.cosmos_pwm_wam.positions import (
    position_ids,
    rotary,
)

from .assets import (
    snapshot_asset_files,
    validate_action_registry,
    validate_tokenizer,
    validate_vae_config,
)
from .request import PwmRequest, prepare_request


def initial_joint_state(
    request: PwmRequest, clean: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Independent per-modality RandomState streams plus fixed coordinates.

    The first VIDEO frame and every absent ACTION slot remain equal to their
    clean reference throughout predictor/corrector updates. Inactive slots are
    removed before attention and before joint CFG normalization, not merely at
    serialization time.
    """
    video = torch.from_numpy(
        np.random.RandomState(request.seed)
        .standard_normal(tuple(clean.shape))
        .astype(np.float32)
    ).to(clean.device)
    action = torch.from_numpy(
        np.random.RandomState(request.seed)
        .standard_normal((request.action_length, 128))
        .astype(np.float32)
    ).to(clean.device)
    active = torch.tensor(request.mask, dtype=torch.bool, device=clean.device)
    video[:, :1] = clean[:, :1]
    action.masked_fill_(~active, 0)
    fixed_video = torch.zeros_like(video, dtype=torch.bool)
    fixed_video[:, :1] = True
    fixed = torch.cat((fixed_video.flatten(), (~active).expand_as(action).flatten()))
    reference = torch.cat((clean.float().flatten(), torch.zeros_like(action).flatten()))
    latent = torch.cat((video.flatten(), action.flatten()))
    return latent, fixed, reference


def guided_velocity(
    conditional: Tensor, unconditional: Tensor, guidance: float, normalize: bool
) -> Tensor:
    """Normalize once across the concatenated VIDEO/ACTION velocity."""
    predicted = unconditional + guidance * (conditional - unconditional)
    if normalize:
        predicted = predicted * (
            torch.norm(conditional) / (torch.norm(predicted) + 1e-8)
        ).clamp(0, 1)
    return predicted


class PwmWamRuntime(nn.Module):
    """Single-device BF16 three-tower model and FP32 joint sampler."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        vae_path: str,
        tokenizer_path: str,
        device: str | torch.device = "cuda",
        attention_backend: str = "sdpa",
        enable_und_cache: bool = False,
    ):
        super().__init__()
        if attention_backend != "sdpa":
            raise ValueError("PWM WAM currently supports eager SDPA only")
        if type(enable_und_cache) is not bool:
            raise ValueError("enable_und_cache must be a boolean")
        self.enable_und_cache = enable_und_cache
        from diffusers import AutoencoderKLWan
        from transformers import AutoTokenizer

        from sglang.multimodal_gen.runtime.models.dits.cosmos_pwm_wam.transformer import (
            PwmWamTransformer,
        )
        from sglang.multimodal_gen.runtime.models.dits.cosmos_pwm_wam.vae import (
            preserve_checkpoint_rms,
        )

        checkpoint = Path(checkpoint).expanduser().resolve()
        self.checkpoint_path = str(checkpoint)
        config_bytes = (checkpoint / "config.json").read_bytes()
        # This identifies the exported configuration, not the weight contents.
        self.checkpoint_config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        self.device = torch.device(device)
        config = json.loads(config_bytes)["model"]["config"]
        self.spec = config["action128_spec"]
        validate_action_registry(self.spec)
        self.transformer = PwmWamTransformer.from_pretrained(
            checkpoint, device=self.device, dtype=torch.bfloat16, load_mode="serial"
        )
        contract = {
            "action128_spec": self.spec,
            "vision_spatial_position_mode": "action128_seam",
            "spatial_coordinate_version": "pixel_seam_patch_corner_v1",
        }
        digest = hashlib.blake2b(
            json.dumps(contract, sort_keys=True).encode(), digest_size=16
        ).digest()
        normalizer = self.transformer.action_normalization
        if bytes(normalizer.contract_digest.cpu().tolist()) != digest:
            raise ValueError("PWM Action128 normalization contract digest mismatch")
        self.contract_fingerprint = hashlib.blake2b(
            json.dumps(self.spec, sort_keys=True).encode(), digest_size=16
        ).hexdigest()
        # The export may have serialized statistics in the model parameter dtype.
        # Config transforms are authoritative; recover their FP32 values exactly.
        for name in ("scale", "bias", "normalization_mask"):
            dtype = torch.bool if name == "normalization_mask" else torch.float32
            value = torch.tensor(
                [entry[name] for entry in self.spec["transforms"]],
                dtype=dtype,
                device=self.device,
            )
            if value.shape != getattr(normalizer, name).shape:
                raise ValueError("PWM Action128 normalization shape mismatch")
            setattr(normalizer, name, value)
        if (
            not torch.isfinite(normalizer.scale).all()
            or not torch.isfinite(normalizer.bias).all()
        ):
            raise ValueError("Non-finite PWM Action128 transforms")
        if (normalizer.scale[normalizer.normalization_mask] <= 0).any():
            raise ValueError("PWM Action128 normalization scale must be positive")
        vae, loading_info = AutoencoderKLWan.from_pretrained(
            vae_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            output_loading_info=True,
        )
        if any(
            loading_info.get(key)
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError(f"Incomplete PWM VAE checkpoint: {loading_info}")
        validate_vae_config(vae.config)
        self.vae = vae.to(self.device).eval()
        preserve_checkpoint_rms(self.vae)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True
        )
        validate_tokenizer(self.tokenizer)
        self.asset_file_sha256 = snapshot_asset_files(vae_path, tokenizer_path)
        self.eval()

    def text_ids(self, caption: str) -> Tensor:
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": caption}],
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=False,
            return_dict=False,
        )
        if self.tokenizer.bos_token_id is not None:
            ids = [self.tokenizer.bos_token_id] + ids
        ids += [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        ]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def vae_scale(self) -> tuple[Tensor, Tensor]:
        mean = torch.tensor(
            self.vae.config.latents_mean, device=self.device, dtype=torch.bfloat16
        ).view(1, 48, 1, 1, 1)
        inverse = 1 / torch.tensor(
            self.vae.config.latents_std, device=self.device, dtype=torch.bfloat16
        ).view(1, 48, 1, 1, 1)
        return mean, inverse

    def encode_observation(self, request: PwmRequest) -> Tensor:
        pixels = request.pixels[None].to(self.device, dtype=torch.float32) / 127.5 - 1
        mean, inverse = self.vae_scale()
        encoded = self.vae.encode(pixels.to(torch.bfloat16)).latent_dist.mode()
        first = ((encoded - mean) * inverse)[0].float()
        clean = first.new_zeros(
            (48, (request.video_frames - 1) // 4 + 1, *first.shape[-2:])
        )
        clean[:, :1] = first
        return clean

    def _new_scheduler(self, request: PwmRequest):
        from sglang.multimodal_gen.runtime.models.dits.cosmos_pwm_wam.scheduler import (
            FlowUniPCMultistepScheduler,
        )

        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            use_dynamic_shifting=False,
            use_cpu_coefficients=True,
            use_legacy_corrector=True,
        )
        scheduler.set_timesteps(request.steps, device=self.device, shift=request.shift)
        return scheduler

    def sample(self, request: PwmRequest, clean: Tensor) -> tuple[Tensor, Tensor]:
        latent, fixed, reference = initial_joint_state(request, clean)
        video_size = clean.numel()
        action_shape = (request.action_length, 128)
        active = torch.tensor(request.mask, device=self.device, dtype=torch.bool)
        video_indices = torch.arange(1, clean.shape[1], device=self.device)
        action_indices = torch.arange(request.action_length, device=self.device)
        projection = torch.full(
            (request.action_length,),
            request.projection_id,
            dtype=torch.long,
            device=self.device,
        )
        branches = []
        captions = (request.caption,)
        if request.guidance != 1:
            captions += (request.negative_prompt,)
        for caption in captions:
            ids = self.text_ids(caption)
            grid = (
                clean.shape[1],
                (clean.shape[2] + 1) // 2,
                (clean.shape[3] + 1) // 2,
            )
            positions = position_ids(
                len(ids),
                grid,
                request.action_length,
                request.video_fps,
                request.action_fps,
                seam=request.seam,
            )
            positions = tuple(p.to(self.device) for p in positions)
            ropes = tuple(rotary(p, dtype=torch.bfloat16) for p in positions)
            action_ropes = tuple(
                rotary(p, dtype=torch.bfloat16, action=True) for p in positions
            )
            branches.append((ids, ropes, action_ropes))
        use_und_cache = (
            self.enable_und_cache
            if request.enable_und_cache is None
            else request.enable_und_cache
        )
        # Text is independent of diffusion time/noise. Only this request owns
        # these caches; CFG branches cannot share conditioning, and an unused
        # unconditional branch is never evaluated just to populate a cache.
        und_caches = [None] * len(branches)
        scheduler = self._new_scheduler(request)
        generator = torch.Generator(device=self.device).manual_seed(request.seed)

        def velocity(branch_index: int, timestep: Tensor) -> Tensor:
            ids, rope, action_rope = branches[branch_index]
            if use_und_cache and und_caches[branch_index] is None:
                und_caches[branch_index] = self.transformer.build_und_cache(
                    ids, rope[0]
                )
            streams = self.transformer.encode_inputs(
                ids,
                latent[:video_size].reshape(clean.shape),
                latent[video_size:].reshape(action_shape),
                timestep.reshape(1),
                projection,
                video_indices,
                action_indices,
                action_dim_mask=active,
            )
            hidden = self.transformer.forward_hidden(
                streams, rope, action_rope, und_cache=und_caches[branch_index]
            )
            vv, va = self.transformer.decode_outputs(
                hidden,
                tuple(clean.shape),
                projection,
                video_indices,
                action_dim_mask=active,
            )
            return torch.cat((vv.flatten(), va.flatten())).masked_fill(fixed, 0)

        for timestep in scheduler.timesteps:
            latent = torch.where(fixed, reference, latent)
            conditional = velocity(0, timestep)
            needs_cfg = request.guidance != 1
            if needs_cfg and request.guidance_interval is not None:
                # Cosmos applies the interval to UniPC's raw 0..1000 timestep,
                # before the network converts time into its normalized range.
                lo, hi = request.guidance_interval
                needs_cfg = lo < timestep.item() < hi
            predicted = (
                guided_velocity(
                    conditional,
                    velocity(1, timestep),
                    request.guidance,
                    request.normalize_cfg,
                )
                if needs_cfg
                else conditional
            )
            latent = scheduler.step(
                model_output=predicted,
                timestep=timestep,
                sample=latent[None],
                generator=generator,
                return_dict=False,
            )[0].squeeze(0)
            latent = torch.where(fixed, reference, latent).float()
        return latent[:video_size].reshape(clean.shape), latent[video_size:].reshape(
            action_shape
        )

    def _finalize(
        self, request: PwmRequest, video: Tensor, normalized: Tensor
    ) -> dict[str, Any]:
        normalized = normalized.float()
        normalizer = self.transformer.action_normalization
        domain = request.normalization_id
        physical = normalized.clone()
        transform_mask = normalizer.normalization_mask[domain]
        physical[:, transform_mask] = (
            physical[:, transform_mask] - normalizer.bias[domain, transform_mask]
        ) / normalizer.scale[domain, transform_mask]
        active = torch.tensor(request.mask, device=self.device, dtype=torch.bool)
        normalized = normalized.masked_fill(~active, 0)
        physical = physical.masked_fill(~active, 0)
        if not torch.isfinite(normalized).all() or not torch.isfinite(physical).all():
            raise RuntimeError("Non-finite PWM WAM actions")
        frames = None
        if request.return_video:
            mean, inverse = self.vae_scale()
            decoded = self.vae.decode(
                (video[None].to(torch.bfloat16) / inverse) + mean, return_dict=False
            )[0]
            if not torch.isfinite(decoded).all():
                raise RuntimeError("Non-finite PWM WAM video")
            # Cosmos WanVAE.decode restores the FP32 input latent dtype before
            # action_server scales pixels; Diffusers leaves its output in BF16.
            frames = (
                (((decoded[0].float().permute(1, 2, 3, 0) + 1) / 2).clamp(0, 1) * 255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
        return {
            "actions": physical.cpu().numpy(),
            "normalized_actions": normalized.cpu().numpy(),
            "action_domain": request.domain,
            "action_dim_mask": request.mask,
            "action_fps": request.action_fps,
            "action_timestamps_seconds": (
                torch.arange(request.action_length, dtype=torch.float64)
                / request.action_fps
            ).tolist(),
            "action_units": "raw_physical_or_source_units",
            "seed": request.seed,
            "conditioning_caption_sha256": hashlib.sha256(
                request.caption.encode()
            ).hexdigest(),
            "contract_fingerprint": self.contract_fingerprint,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_config_sha256": self.checkpoint_config_sha256,
            "loader_info": dict(self.transformer.load_info),
            "asset_file_sha256": dict(self.asset_file_sha256),
            "video_frames": frames,
            "video_fps": request.video_fps,
            "parameters": {
                "num_inference_steps": request.steps,
                "num_frames": request.video_frames,
                "guidance": request.guidance,
                "shift": request.shift,
                "enable_und_cache": (
                    self.enable_und_cache
                    if request.enable_und_cache is None
                    else request.enable_und_cache
                ),
            },
        }

    @torch.inference_mode()
    def forward(self, raw: dict) -> dict[str, Any]:
        # The shared SGLang platform disables cuDNN SDPA for other pipelines.
        # This checkpoint was validated with PyTorch's full backend selection;
        # preserve that policy for WAM and restore the caller's flags on exit.
        with sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
                SDPBackend.CUDNN_ATTENTION,
            ]
        ):
            return self._forward(raw)

    def _forward(self, raw: dict) -> dict[str, Any]:
        # This eager baseline records completed GPU work, including the small
        # synchronization cost at stage boundaries, rather than enqueue times.
        def timestamp() -> float:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            return time.perf_counter()

        started = timestamp()
        request = prepare_request(raw, self.spec)
        prepared = timestamp()
        clean = self.encode_observation(request)
        encoded = timestamp()
        video, normalized = self.sample(request, clean)
        sampled = timestamp()
        result = self._finalize(request, video, normalized)
        finalized = timestamp()
        result["timing_seconds"] = {
            "prepare": prepared - started,
            "vae_encode": encoded - prepared,
            "sample": sampled - encoded,
            "finalize": finalized - sampled,
            "total": finalized - started,
        }
        result["elapsed_seconds"] = finalized - started
        return result
