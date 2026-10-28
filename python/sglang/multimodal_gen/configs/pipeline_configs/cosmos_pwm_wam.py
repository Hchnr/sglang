# SPDX-License-Identifier: Apache-2.0
"""Native three-tower Cosmos PWM WAM serving configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


def is_cosmos_pwm_wam_package(model_path: str) -> bool:
    """Identify the local Action128 three-tower artifact without name heuristics."""
    root = Path(model_path).expanduser()
    if not (root / "model.safetensors.index.json").is_file():
        return False
    try:
        config = json.loads((root / "config.json").read_text())
        model = config["model"]["config"]
        return (
            config.get("architectures") == ["Cosmos3OmniModel"]
            and model.get("tower_structure") == "three_tower"
            and model.get("action_tower_preset") == "width_only_1b"
            and model.get("action_tower_execution") == "native"
            and model.get("action_gen") is True
            and model.get("vision_gen") is True
            and model.get("sound_gen") is False
            and isinstance(model.get("action128_spec"), dict)
            and model.get("diffusion_expert_config", {}).get(
                "vision_spatial_position_mode"
            )
            == "action128_seam"
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


# PipelineConfig's constructor/CLI/serialization contract uses dataclasses.
@dataclass
class CosmosPWMWAMPipelineConfig(PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.VLA_ACTION
    should_use_guidance: bool = False
    enable_autocast: bool = False
    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    vae_decode_precision: str | None = "bf16"
    vae_path: str | None = None
    tokenizer_path: str | None = None
    enable_und_cache: bool = False
    default_num_inference_steps: int = 30
    action_dim: int = 128

    def supports_dynamic_batching(self) -> bool:
        return False

    def supports_native_grouped_requests(self) -> bool:
        return False

    def validate_server_args(self, server_args: Any) -> None:
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
        from sglang.multimodal_gen.runtime.server_args import Backend

        if not isinstance(self.enable_und_cache, bool):
            raise TypeError("Cosmos PWM WAM enable_und_cache must be a boolean")
        if server_args.backend == Backend.DIFFUSERS:
            raise ValueError("Cosmos PWM WAM requires --backend sglang")
        if server_args.disagg_role != RoleType.MONOLITHIC:
            raise ValueError("Cosmos PWM WAM supports monolithic execution only")
        if (
            any(
                value not in (None, 1)
                for value in (
                    server_args.num_gpus,
                    server_args.nnodes,
                    server_args.tp_size,
                    server_args.sp_degree,
                    server_args.dp_size,
                    server_args.ulysses_degree,
                    server_args.ring_degree,
                    server_args.kv_gather_degree,
                    server_args.cfg_parallel_degree,
                )
            )
            or server_args.enable_cfg_parallel
        ):
            raise ValueError("Cosmos PWM WAM supports one GPU with TP=SP=DP=1")
        if server_args.batching_max_size != 1:
            raise ValueError("Cosmos PWM WAM requires --batching-max-size 1")
        if server_args.enable_torch_compile or server_args.enable_breakable_cuda_graph:
            raise ValueError("Cosmos PWM WAM currently supports eager execution only")
        if server_args.regional_compile:
            raise ValueError("Cosmos PWM WAM does not support regional_compile")
        if server_args.cache_dit_config is not None:
            raise ValueError("Cosmos PWM WAM does not support cache_dit_config")
        if server_args.enable_attention_backend_autotune:
            raise ValueError(
                "Cosmos PWM WAM does not support attention backend autotuning"
            )
        if (
            server_args.quantization
            or server_args.component_quantizations
            or server_args.kv_cache_quant_config.enabled
        ):
            raise ValueError("Cosmos PWM WAM does not support quantization")
        if (
            server_args.dit_cpu_offload
            or server_args.dit_layerwise_offload
            or server_args.text_encoder_cpu_offload
            or server_args.image_encoder_cpu_offload
            or server_args.vae_cpu_offload
            or server_args.cpu_offload_components
            or server_args.layerwise_offload_components
            or server_args.component_residency
            or server_args.use_fsdp_inference
        ):
            raise ValueError(
                "Cosmos PWM WAM does not support offload/FSDP; set "
                "--dit-cpu-offload false --text-encoder-cpu-offload false "
                "--image-encoder-cpu-offload false --vae-cpu-offload false "
                "--dit-layerwise-offload false --use-fsdp-inference false"
            )
        if server_args.attention_backend not in (None, "torch_sdpa", "sdpa"):
            raise ValueError("Cosmos PWM WAM requires --attention-backend torch_sdpa")
        if server_args.component_attention_backends:
            raise ValueError(
                "Cosmos PWM WAM does not accept component attention overrides"
            )
        if (
            server_args.component_paths
            or server_args.component_weights_paths
            or server_args.component_precisions
            or server_args.transformer_weights_path
        ):
            raise ValueError(
                "Cosmos PWM WAM loads its complete artifact; configure VAE and "
                "tokenizer using pipeline_config.vae_path/tokenizer_path"
            )
        if server_args.lora_path or server_args.comfyui_mode:
            raise ValueError("Cosmos PWM WAM does not support LoRA or ComfyUI")
        if self.dit_precision != "bf16" or self.vae_precision != "bf16":
            raise ValueError("Cosmos PWM WAM requires BF16 model and VAE precision")
        if self.vae_decode_precision not in (None, "bf16"):
            raise ValueError("Cosmos PWM WAM requires BF16 VAE decode precision")
        if server_args.warmup_mode != "off":
            raise ValueError(
                "Cosmos PWM WAM requires --warmup-mode off; warm up using an explicit WAM request"
            )
        if not self.vae_path or not self.tokenizer_path:
            raise ValueError(
                "Cosmos PWM WAM requires pipeline_config.vae_path and "
                "pipeline_config.tokenizer_path"
            )

    def action_metadata(self, server_args: Any) -> dict[str, Any]:
        return {
            "object": "action.metadata",
            "model": server_args.served_model_name,
            "policy_family": "cosmos_pwm_wam",
            "input": {"modalities": ["image", "video"], "state_dim": None},
            "output": {
                "action_type": "continuous",
                "action_dim": 128,
                "dtype": "float32",
                "units": "source",
                "joint_video": True,
            },
            "defaults": {
                "action_mode": "wam",
                "num_inference_steps": 30,
                "guidance_scale": 1.0,
                "flow_shift": 10.0,
                "return_video": True,
                "enable_und_cache": self.enable_und_cache,
            },
            "capabilities": {
                "action_modes": ["wam"],
                "batch_inputs": False,
                "multiple_candidates": False,
                "cuda_graph": False,
                "exact_prefix_cache": False,
                "request_local_und_cache": True,
                "cross_request_und_cache": False,
                "realtime_websocket": False,
                "openpi_websocket": False,
            },
        }


def register() -> None:
    from sglang.multimodal_gen.configs.sample.cosmos_pwm_wam import (
        CosmosPWMWAMSamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=CosmosPWMWAMSamplingParams,
        pipeline_config_cls=CosmosPWMWAMPipelineConfig,
        model_detectors=[lambda name: name.lower() == "cosmospwmwampipeline"],
        pipeline_config_registry_entries={
            "CosmosPWMWAMPipeline": (
                CosmosPWMWAMPipelineConfig,
                CosmosPWMWAMSamplingParams,
            )
        },
    )
