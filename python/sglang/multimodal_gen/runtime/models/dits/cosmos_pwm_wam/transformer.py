# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from vllm-omni-pwm d3e9aad33e861b69be16c4f341695e1cc12b3314.
"""Native BF16 PWM three-tower model with the original HF parameter tree.

Inference is single-sample SDPA. Model state contains weights only; request-owned
noise, solver history and conditioning are supplied by the pipeline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import Tensor, nn

Rope = tuple[Tensor, Tensor]
Streams = tuple[Tensor, Tensor, Tensor]


def mask_actions(actions: Tensor, mask: Tensor | None) -> Tensor:
    """Apply the caller's slot mask before projections or solver integration."""
    if mask is None:
        return actions
    if mask.dtype != torch.bool or mask.shape != (128,):
        raise ValueError("Action128 slot mask must have shape [128] and bool dtype")
    return actions.masked_fill(~mask.to(device=actions.device), 0)


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate checkpoint JSON key: {key}")
        result[key] = value
    return result


class PwmRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        value = x.float()
        value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * value.to(x.dtype)


def rotate(x: Tensor, rope: Rope) -> Tensor:
    left, right = x.chunk(2, dim=-1)
    return x * rope[0][:, None] + torch.cat((-right, left), dim=-1) * rope[1][:, None]


def attention(q: Tensor, k: Tensor, v: Tensor, *, causal: bool = False) -> Tensor:
    mask = torch.ones((q.shape[0], k.shape[0]), dtype=torch.bool, device=q.device)
    if causal:
        mask = mask.tril()
    result = F.scaled_dot_product_attention(
        q.transpose(0, 1)[None],
        k.transpose(0, 1)[None],
        v.transpose(0, 1)[None],
        attn_mask=mask[None, None],
        enable_gqa=True,
        scale=q.shape[-1] ** -0.5,
    )
    return result[0].transpose(0, 1).flatten(1)


class PwmMLP(nn.Module):
    def __init__(self, width: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, intermediate, bias=False)
        self.up_proj = nn.Linear(width, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PwmAttentionWeights(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        video: bool = False,
        heads: int = 32,
        kv_heads: int = 8,
        dim: int = 128,
    ):
        super().__init__()
        self.heads, self.kv_heads, self.dim = heads, kv_heads, dim
        for suffix in ("", "_moe_gen") if video else ("",):
            for name, output in (
                ("q_proj", heads * dim),
                ("k_proj", kv_heads * dim),
                ("v_proj", kv_heads * dim),
            ):
                setattr(self, name + suffix, nn.Linear(width, output, bias=False))
            setattr(self, "o_proj" + suffix, nn.Linear(heads * dim, width, bias=False))
            setattr(self, "q_norm" + suffix, PwmRMSNorm(dim))
            setattr(self, "k_norm" + suffix, PwmRMSNorm(dim))

    def project(
        self, x: Tensor, suffix: str = "", *, normalize_key: bool = True
    ) -> tuple[Tensor, Tensor, Tensor]:
        q = getattr(self, "q_proj" + suffix)(x)
        k = getattr(self, "k_proj" + suffix)(x)
        v = getattr(self, "v_proj" + suffix)(x)
        q = q.reshape(-1, self.heads, self.dim)
        k = k.reshape(-1, self.kv_heads, self.dim)
        v = v.reshape(-1, self.kv_heads, self.dim)
        key = getattr(self, "k_norm" + suffix)(k) if normalize_key else k
        return getattr(self, "q_norm" + suffix)(q), key, v


class PwmActionBlock(PwmAttentionWeights):
    def __init__(self, width: int = 1280, intermediate: int = 3840, **kwargs):
        super().__init__(width, **kwargs)
        self.input_layernorm = PwmRMSNorm(width)
        self.post_attention_layernorm = PwmRMSNorm(width)
        self.gate_proj = nn.Linear(width, intermediate, bias=False)
        self.up_proj = nn.Linear(width, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, width, bias=False)

    def mlp(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PwmThreeTowerLayer(nn.Module):
    def __init__(
        self,
        width: int = 4096,
        intermediate: int = 12288,
        action_width: int = 1280,
        action_intermediate: int = 3840,
        heads: int = 32,
        kv_heads: int = 8,
        dim: int = 128,
    ):
        super().__init__()
        self.self_attn = PwmAttentionWeights(
            width, video=True, heads=heads, kv_heads=kv_heads, dim=dim
        )
        self.input_layernorm = PwmRMSNorm(width)
        self.input_layernorm_moe_gen = PwmRMSNorm(width)
        self.post_attention_layernorm = PwmRMSNorm(width)
        self.post_attention_layernorm_moe_gen = PwmRMSNorm(width)
        self.mlp = PwmMLP(width, intermediate)
        self.mlp_moe_gen = PwmMLP(width, intermediate)
        self.action_block_moe_gen_action = PwmActionBlock(
            action_width, action_intermediate, heads=heads, kv_heads=kv_heads, dim=dim
        )

    def forward(
        self,
        streams: Streams,
        rope: tuple[Rope, Rope, Rope],
        action_rope: tuple[Rope, Rope, Rope],
    ) -> Streams:
        und, video, action = streams
        attn, act = self.self_attn, self.action_block_moe_gen_action
        # Every tower reads the same pre-update layer state.
        qu, raw_ku, vu = attn.project(self.input_layernorm(und), normalize_key=False)
        ku = attn.k_norm(raw_ku)
        qv, kv, vv = attn.project(self.input_layernorm_moe_gen(video), "_moe_gen")
        qa, ka, va = act.project(act.input_layernorm(action))
        keys, values = (ku, kv, ka), (vu, vv, va)
        yu = attention(rotate(qu, rope[0]), rotate(ku, rope[0]), vu, causal=True)
        yv = attention(
            rotate(qv, rope[1]),
            torch.cat([rotate(k, r) for k, r in zip(keys, rope)]),
            torch.cat(values),
        )
        ya = attention(
            rotate(qa, action_rope[2]),
            torch.cat([rotate(k, r) for k, r in zip(keys, action_rope)]),
            torch.cat(values),
        )
        action = action + act.o_proj(ya)
        action = action + act.mlp(act.post_attention_layernorm(action))
        und = und + attn.o_proj(yu)
        und = und + self.mlp(self.post_attention_layernorm(und))
        video = video + attn.o_proj_moe_gen(yv)
        video = video + self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(video))
        return und, video, action


class PwmDomainLinear(nn.Module):
    def __init__(self, input_width: int, output_width: int):
        super().__init__()
        self.input_width, self.output_width = input_width, output_width
        self.fc = nn.Embedding(64, input_width * output_width)
        self.bias = nn.Embedding(64, output_width)

    def forward(self, x: Tensor, domain_ids: Tensor) -> Tensor:
        weights = self.fc(domain_ids).view(-1, self.input_width, self.output_width)
        return torch.bmm(x[:, None], weights).squeeze(1) + self.bias(domain_ids)


class PwmTimeEmbedding(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width)
        )
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(128, dtype=torch.float32, device="cpu")
            / 128
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # A module-wide dtype conversion must not round this unlearned FP32
        # constant. Meta checkpoint loading also reconstructs it from CPU.
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(128, dtype=torch.float32, device="cpu")
            / 128
        )
        self.frequencies = frequencies.to(self.frequencies.device)
        return self

    def forward(self, time: Tensor) -> Tensor:
        # FP32 time arithmetic remains independent of any outer stage autocast.
        with torch.autocast(device_type=time.device.type, enabled=False):
            phase = time.float()[:, None] * self.frequencies.to(time.device)[None]
            encoded = torch.cat((phase.cos(), phase.sin()), dim=-1)
            first, last = self.mlp[0], self.mlp[2]
            hidden = F.silu(F.linear(encoded, first.weight.float(), first.bias.float()))
            return F.linear(hidden, last.weight.float(), last.bias.float())


class PwmTextStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(151936, 4096)
        self.layers = nn.ModuleList([PwmThreeTowerLayer() for _ in range(36)])
        self.norm = PwmRMSNorm(4096)
        self.norm_moe_gen = PwmRMSNorm(4096)
        self.norm_moe_gen_action = PwmRMSNorm(1280)


class PwmLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = PwmTextStack()
        # Retained for strict checkpoint coverage; WAM does not emit text logits.
        self.lm_head = nn.Linear(4096, 151936, bias=False)


class PwmNormalizationBuffers(nn.Module):
    def __init__(self, num_domains: int):
        super().__init__()
        self.register_buffer(
            "scale", torch.empty(num_domains, 128, dtype=torch.float32)
        )
        self.register_buffer("bias", torch.empty(num_domains, 128, dtype=torch.float32))
        self.register_buffer(
            "normalization_mask", torch.empty(num_domains, 128, dtype=torch.bool)
        )
        self.register_buffer("contract_digest", torch.empty(16, dtype=torch.uint8))


class PwmWamTransformer(nn.Module):
    def __init__(self, num_domains: int):
        super().__init__()
        self.language_model = PwmLanguageModel()
        self.vae2llm = nn.Linear(192, 4096)
        self.llm2vae = nn.Linear(4096, 192)
        self.time_embedder = PwmTimeEmbedding(4096)
        self.action_time_embedder = PwmTimeEmbedding(1280)
        self.action2llm = PwmDomainLinear(128, 1280)
        self.llm2action = PwmDomainLinear(1280, 128)
        self.action_modality_embed = nn.Parameter(torch.zeros(1280))
        self.action_normalization = PwmNormalizationBuffers(num_domains)

    @classmethod
    def from_pretrained(
        cls,
        path: Path,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        load_mode: str = "serial",
    ):
        """Strictly load the native, sharded HF artifact without Cosmos imports."""
        if load_mode != "serial":
            raise ValueError("PWM first supports serial checkpoint loading only")
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("PWM supports BF16 weights or FP32 diagnostic loading")
        path = Path(path).resolve()
        config = json.loads(
            (path / "config.json").read_text(), object_pairs_hook=_unique_json_object
        )["model"]["config"]
        required = {
            "tower_structure": "three_tower",
            "action_tower_preset": "width_only_1b",
            "action_tower_execution": "native",
            "vision_gen": True,
            "action_gen": True,
            "sound_gen": False,
            "video_temporal_causal": False,
            "joint_attn_implementation": "two_way",
            "max_action_dim": 128,
            "action_tower_projection_mode": "routed",
            "num_embodiment_domains": 64,
        }
        for name, expected_value in required.items():
            if config.get(name) != expected_value:
                raise ValueError(f"PWM requires {name}={expected_value!r}")
        # These change inference semantics without changing tensor shapes. A
        # valid shard index alone cannot establish compatibility with this core.
        for section, fields in (
            (
                "vlm_config",
                {
                    "use_system_prompt": False,
                    "model_name": "Qwen/Qwen3-VL-8B-Instruct",
                },
            ),
            (
                "rectified_flow_inference_config",
                {
                    "num_train_timesteps": 1000,
                    "scheduler_type": "unipc",
                    "shift": 1,
                    "use_dynamic_shifting": False,
                },
            ),
        ):
            settings = config.get(section)
            if not isinstance(settings, dict):
                raise ValueError(f"PWM requires explicit {section}")  # noqa: TRY004 -- Keep the checkpoint validation error contract.
            for name, expected_value in fields.items():
                if settings.get(name) != expected_value:
                    raise ValueError(
                        f"PWM requires {section}.{name}={expected_value!r}"
                    )
        model_instance = config["vlm_config"].get("model_instance", {})
        vlm_settings = model_instance.get("config", {})
        for name, expected_value in {
            "qk_norm_for_text": True,
            "include_visual": False,
        }.items():
            if vlm_settings.get(name) != expected_value:
                raise ValueError(
                    f"PWM requires vlm_config.model_instance.config.{name}={expected_value!r}"
                )
        expert = config.get("diffusion_expert_config", {})
        for name, expected_value in {
            "vision_spatial_position_mode": "action128_seam",
            "vision_temporal_position_mode": "latent_index",
            "patch_spatial": 2,
            "unified_3d_mrope_temporal_modality_margin": 15000,
            "unified_3d_mrope_reset_spatial_ids": True,
            "timestep_range": 1.0,
            "base_fps": 24,
            "enable_fps_modulation": True,
            "enable_media_modality_embedding": False,
            "enable_vision_modality_embeddings": False,
        }.items():
            if expert.get(name) != expected_value:
                raise ValueError(
                    f"PWM requires diffusion_expert_config.{name}={expected_value!r}"
                )
        with torch.device("meta"):
            model = cls(len(config["action128_spec"]["registry"]["domains"]))
        expected = model.state_dict()
        index = json.loads(
            (path / "model.safetensors.index.json").read_text(),
            object_pairs_hook=_unique_json_object,
        )["weight_map"]
        if not isinstance(index, dict) or any(
            not key.startswith("model.net.") for key in index
        ):
            raise ValueError("PWM checkpoint keys must use the model.net. prefix")
        source = {key.removeprefix("model.net."): shard for key, shard in index.items()}
        if set(expected) != set(source):
            raise ValueError(
                f"HF tensor mismatch: missing={set(expected) - set(source)}, extra={set(source) - set(expected)}"
            )
        parameters = set(dict(model.named_parameters()))
        if any(not isinstance(shard, str) for shard in source.values()):
            raise ValueError("PWM shard index must contain filenames")
        shards = sorted(set(source.values()))
        loaded_bytes = 0
        for shard in shards:
            shard_path = (path / shard).resolve()
            if not shard_path.is_relative_to(path):
                raise ValueError(
                    "PWM checkpoint shard must be inside the artifact directory"
                )
            names = {name for name, filename in source.items() if filename == shard}
            with safe_open(shard_path, framework="pt", device="cpu") as tensors:
                if set(tensors.keys()) != {"model.net." + name for name in names}:
                    raise ValueError(f"PWM shard contents disagree with index: {shard}")
                for name in sorted(names):
                    value = tensors.get_tensor("model.net." + name)
                    if value.shape != expected[name].shape:
                        raise ValueError(
                            f"HF shape mismatch for {name}: {value.shape} != {expected[name].shape}"
                        )
                    target_dtype = dtype if name in parameters else expected[name].dtype
                    if name in parameters and value.dtype != torch.bfloat16:
                        raise ValueError(
                            f"PWM checkpoint parameter {name} must be BF16"
                        )
                    if name not in parameters and value.dtype != expected[name].dtype:
                        # Legacy checkpoints rounded normalizer transforms to BF16.
                        # The runtime verifies their digest before restoring exact FP32
                        # transforms from action128_spec, never from rounded buffers.
                        legacy_transform = name in (
                            "action_normalization.scale",
                            "action_normalization.bias",
                        )
                        if not legacy_transform or value.dtype != torch.bfloat16:
                            raise ValueError(f"PWM buffer dtype mismatch for {name}")
                    value = value.to(device=device, dtype=target_dtype)
                    owner, _, leaf = name.rpartition(".")
                    module = model.get_submodule(owner) if owner else model
                    setattr(
                        module,
                        leaf,
                        (
                            nn.Parameter(value, requires_grad=False)
                            if name in parameters
                            else value
                        ),
                    )
                    loaded_bytes += value.numel() * value.element_size()
        model.load_info = {
            "mode": load_mode,
            "tensor_count": len(source),
            "shard_count": len(shards),
            "loaded_tensor_bytes": loaded_bytes,
        }
        return model.eval()

    def encode_inputs(
        self,
        text_ids: Tensor,
        video_latents: Tensor,
        actions: Tensor,
        timestep: Tensor,
        projection_ids: Tensor,
        noisy_video_frames: Tensor,
        noisy_action_frames: Tensor,
        *,
        action_dim_mask: Tensor | None = None,
    ) -> Streams:
        """Embed one packed WAM sample, preserving the reference operation order.

        Video layout is C,T,H,W; timestep is the sampler's unscaled scalar.
        Caller supplies noisy indices explicitly so conditioning never gets a
        timestep embedding. Projection IDs are not normalization-domain IDs.
        """
        channels, frames, height, width = video_latents.shape
        if channels != 48 or actions.ndim != 2 or actions.shape[1] != 128:
            raise ValueError("PWM requires 48-channel video latents and Action128")
        if timestep.numel() != 1:
            raise ValueError("Native WAM currently supports one timestep per request")
        dtype = self.vae2llm.weight.dtype
        padded = F.pad(video_latents, (0, width % 2, 0, height % 2))
        hp, wp = (height + 1) // 2, (width + 1) // 2
        patches = padded.reshape(channels, frames, hp, 2, wp, 2)
        patches = patches.permute(1, 2, 4, 3, 5, 0).reshape(-1, 192)
        video = self.vae2llm(patches.to(dtype))
        scaled_time = timestep.reshape(1).float() * 0.001
        video_time = self.time_embedder(scaled_time).to(dtype)
        spatial = torch.arange(hp * wp, device=video.device)
        noisy_tokens = (noisy_video_frames[:, None] * (hp * wp) + spatial).flatten()
        video[noisy_tokens] = video[noisy_tokens] + video_time
        action = self.action2llm(
            mask_actions(actions, action_dim_mask).to(dtype), projection_ids
        )
        action = action + self.action_modality_embed[None]
        action_time = self.action_time_embedder(scaled_time).to(dtype)
        action[noisy_action_frames] = action[noisy_action_frames] + action_time
        return self.language_model.model.embed_tokens(text_ids), video, action

    def decode_outputs(
        self,
        streams: Streams,
        video_shape: tuple[int, int, int, int],
        projection_ids: Tensor,
        noisy_video_frames: Tensor,
        *,
        action_dim_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Decode final normalized hidden states into video/action velocities."""
        _, frames, height, width = video_shape
        hp, wp = (height + 1) // 2, (width + 1) // 2
        video = (
            streams[1].reshape(frames, hp * wp, -1)[noisy_video_frames].flatten(0, 1)
        )
        patches = self.llm2vae(video).reshape(-1, hp, wp, 2, 2, 48)
        decoded = patches.permute(5, 0, 1, 3, 2, 4).reshape(48, -1, hp * 2, wp * 2)
        velocity = decoded.new_zeros(video_shape)
        velocity[:, noisy_video_frames] = decoded[:, :, :height, :width]
        action = self.llm2action(streams[2], projection_ids)
        return velocity, mask_actions(action, action_dim_mask)

    def forward_hidden(
        self,
        streams: Streams,
        rope: tuple[Rope, Rope, Rope],
        action_rope: tuple[Rope, Rope, Rope],
        und_cache=None,
        attention_metadata=None,
    ) -> Streams:
        if und_cache is not None or attention_metadata is not None:
            raise ValueError(
                "PWM eager SDPA does not support caching or packed attention metadata"
            )
        for layer in self.language_model.model.layers:
            streams = layer(streams, rope, action_rope)
        stack = self.language_model.model
        return (
            stack.norm(streams[0]),
            stack.norm_moe_gen(streams[1]),
            stack.norm_moe_gen_action(streams[2]),
        )
