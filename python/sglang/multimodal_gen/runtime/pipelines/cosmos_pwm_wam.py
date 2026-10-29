# SPDX-License-Identifier: Apache-2.0
"""Execute three-tower PWM WAM inside the SGLang generation worker."""

from pathlib import Path
from typing import ClassVar

import torch

from sglang.multimodal_gen.configs.pipeline_configs.cosmos_pwm_wam import (
    CosmosPWMWAMPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.cosmos_pwm_wam import (
    CosmosPWMWAMSamplingParams,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class CosmosPWMWAMStage(PipelineStage):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        raw = batch.extra["cosmos_pwm_wam"]
        payload = self.runtime.forward(raw)
        payload["request_id"] = batch.request_id
        payload["action_mode"] = "wam"
        payload.setdefault("parameters", {})["num_inference_steps"] = raw["num_steps"]
        return OutputBatch(output=[payload], metrics=batch.metrics)


class CosmosPWMWAMPipeline(ComposedPipelineBase):
    pipeline_name = "CosmosPWMWAMPipeline"
    pipeline_config_cls = CosmosPWMWAMPipelineConfig
    sampling_params_cls = CosmosPWMWAMSamplingParams
    _required_config_modules: ClassVar[list[str]] = []

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError("Cosmos PWM WAM supports monolithic execution only")

    def load_modules(self, server_args: ServerArgs, loaded_modules=None):
        config = server_args.pipeline_config
        if not isinstance(config, CosmosPWMWAMPipelineConfig):
            raise TypeError("CosmosPWMWAMPipeline requires CosmosPWMWAMPipelineConfig")
        config.validate_server_args(server_args)
        if loaded_modules is not None:
            return loaded_modules
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_pwm_wam.runtime import (
            PwmWamRuntime,
        )

        return {
            "wam_runtime": PwmWamRuntime(
                Path(self.model_path),
                vae_path=config.vae_path,
                tokenizer_path=config.tokenizer_path,
                device=f"cuda:{torch.cuda.current_device()}",
                attention_backend=config.runtime_attention_backend(server_args),
                enable_torch_compile=server_args.enable_torch_compile,
                enable_und_cache=config.enable_und_cache,
            )
        }

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(CosmosPWMWAMStage(self.get_module("wam_runtime")), "pwm_wam")


EntryClass = CosmosPWMWAMPipeline
