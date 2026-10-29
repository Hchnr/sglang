# SPDX-License-Identifier: Apache-2.0
"""Transport the WAM contract through native SGLang action requests."""

from __future__ import annotations

import copy
import json
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sglang.multimodal_gen.configs.sample.action import ActionSamplingParams


@dataclass
class CosmosPWMWAMSamplingParams(ActionSamplingParams):
    num_inference_steps: int = 30
    seed: int = field(default_factory=lambda: secrets.randbelow(2**31))
    prompt: str = ""
    vision_path: str | None = None
    prompt_path: str | None = None
    action_domain: str | None = None
    action_fps: float | None = None
    fps: float | None = None
    action_horizon: int | None = None
    num_frames: int | None = None
    image_size: int | None = None
    guidance_scale: float = 1.0
    flow_shift: float = 10.0
    return_video: bool = True
    enable_und_cache: bool | None = None
    action_dim_mask: list[bool] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    wam_request: dict[str, Any] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )

    def _validate(self) -> None:
        super()._validate()
        if self.num_outputs_per_prompt != 1 or isinstance(self.seed, list):
            raise ValueError("Cosmos PWM WAM supports one observation and one output")
        if not isinstance(self.prompt, str):
            raise TypeError("Cosmos PWM WAM prompt must be a string")
        if self.enable_sequence_shard:
            raise ValueError("Cosmos PWM WAM does not support sequence sharding")
        if not isinstance(self.return_video, bool):
            raise TypeError("return_video must be a boolean")
        if self.enable_und_cache is not None and not isinstance(
            self.enable_und_cache, bool
        ):
            raise TypeError("enable_und_cache must be a boolean or None")
        if self.wam_request is not None and not isinstance(self.wam_request, dict):
            raise ValueError("wam_request must be an object")

    def build_wam_request(self) -> dict[str, Any]:
        raw = copy.deepcopy(self.wam_request) if self.wam_request is not None else {}
        if raw.get("model_mode", "wam") != "wam":
            raise ValueError("Cosmos PWM WAM only supports model_mode='wam'")
        raw["model_mode"] = "wam"
        if (
            self.enable_und_cache is not None
            and "enable_und_cache" in raw
            and raw["enable_und_cache"] is not self.enable_und_cache
        ):
            raise ValueError(
                "Conflicting enable_und_cache in wam_request and sampling params"
            )
        fields = {
            "prompt": self.prompt,
            "prompt_path": self.prompt_path,
            "vision_path": self.vision_path,
            "fps": self.fps,
            "action_chunk_size": self.action_horizon,
            "num_frames": self.num_frames,
            "image_size": self.image_size,
            "num_steps": self.num_inference_steps,
            "guidance": self.guidance_scale,
            "shift": self.flow_shift,
            "enable_und_cache": self.enable_und_cache,
        }
        for name, value in fields.items():
            if (
                name == "prompt"
                and not value
                and (raw.get("prompt_path") or self.prompt_path)
            ):
                continue
            if value is not None:
                raw.setdefault(name, value)
        if raw.get("seed") is None:
            raw["seed"] = self.seed
        raw_extra = raw.setdefault("extra", {})
        if not isinstance(raw_extra, dict):
            raise TypeError("WAM extra must be an object")
        for key, value in self.extra.items():
            raw_extra.setdefault(key, copy.deepcopy(value))
        action128 = raw_extra.setdefault("action128", {})
        if not isinstance(action128, dict):
            raise TypeError("extra.action128 must be an object")
        for name, value in (
            ("action_domain", self.action_domain),
            ("action_fps", self.action_fps),
            ("action_dim_mask", self.action_dim_mask),
            ("return_video", self.return_video),
        ):
            if value is not None:
                action128.setdefault(name, value)
        return raw

    def build_request_extra(self) -> dict[str, Any]:
        return {"cosmos_pwm_wam": self.build_wam_request()}

    def _validate_python_fields(self) -> set[str]:
        # The shared factory records user keys after constructing the model's
        # dataclass. HTTP creates this class directly with an already lowered raw
        # request, so it has no explicit convenience fields to validate here.
        explicit = set(self.__dict__.get("_explicit_fields", ()))
        top_level = {
            "prompt": "prompt",
            "prompt_path": "prompt_path",
            "vision_path": "vision_path",
            "seed": "seed",
            "fps": "fps",
            "action_horizon": "action_chunk_size",
            "num_frames": "num_frames",
            "image_size": "image_size",
            "num_inference_steps": "num_steps",
            "guidance_scale": "guidance",
            "flow_shift": "shift",
            "enable_und_cache": "enable_und_cache",
        }
        action_fields = {
            "action_domain",
            "action_fps",
            "action_dim_mask",
            "return_video",
        }
        framework_fields = {
            "request_id",
            "num_outputs_per_prompt",
            "enable_sequence_shard",
        }
        unsupported = explicit - (
            set(top_level) | action_fields | framework_fields | {"wam_request", "extra"}
        )
        if unsupported:
            raise ValueError(
                f"Unsupported Cosmos PWM WAM Python fields: {sorted(unsupported)}"
            )
        # This factory escape hatch is removed before _explicit_fields is set.
        if self.__dict__.get("diffusers_kwargs") is not None:
            raise ValueError("Cosmos PWM WAM does not support diffusers_kwargs")
        raw = self.wam_request or {}
        raw_extra = raw.get("extra") or {}
        if not isinstance(raw_extra, dict) or not isinstance(self.extra, dict):
            raise TypeError("WAM extra must be an object")
        raw_action = raw_extra.get("action128") or {}
        python_action = self.extra.get("action128") or {}
        if not isinstance(raw_action, dict) or not isinstance(python_action, dict):
            raise TypeError("extra.action128 must be an object")

        def check(name, source, key):
            value = getattr(self, name)
            if (
                name in explicit
                and value is not None
                and key in source
                and value != source[key]
            ):
                raise ValueError(
                    f"Conflicting {name} in Python fields and wam_request/extra"
                )

        for name, key in top_level.items():
            check(name, raw, key)
        if "extra" in explicit and "extra" in raw and self.extra != raw["extra"]:
            raise ValueError("Conflicting extra in Python fields and wam_request")
        for name in action_fields:
            check(name, raw_action, name)
            check(name, python_action, name)
        if (
            ("prompt" in explicit and "prompt" not in raw and raw.get("prompt_path"))
            or (
                "prompt_path" in explicit
                and self.prompt_path is not None
                and "prompt_path" not in raw
                and raw.get("prompt")
            )
            or ({"prompt", "prompt_path"} <= explicit and self.prompt_path is not None)
        ):
            raise ValueError(
                "Provide prompt or prompt_path from one Python input source"
            )
        return explicit & framework_fields

    def _adjust(self, server_args) -> None:
        framework_fields = self._validate_python_fields()
        super()._adjust(server_args)
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_pwm_wam.request import (
            validate_request,
        )

        config = json.loads((Path(server_args.model_path) / "config.json").read_text())
        spec = config["model"]["config"]["action128_spec"]
        raw = self.build_wam_request()
        raw.setdefault("enable_und_cache", server_args.pipeline_config.enable_und_cache)
        self.wam_request = validate_request(raw, spec)
        self.enable_und_cache = self.wam_request["enable_und_cache"]
        self.seed = self.wam_request["seed"]
        self.num_inference_steps = self.wam_request["num_steps"]
        # Normalization freezes paths, defaults and seed into the canonical raw
        # request. Later adjustments must validate that one source, not compare
        # its normalized values against the original convenience arguments.
        self._explicit_fields = framework_fields | {"wam_request"}
