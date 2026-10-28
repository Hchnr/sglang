# SPDX-License-Identifier: Apache-2.0
"""WAM registry and joint action/video protocol bookkeeping, without model weights."""

import asyncio
import copy
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from starlette.requests import ClientDisconnect

from sglang.multimodal_gen.configs.pipeline_configs.cosmos_pwm_wam import (
    CosmosPWMWAMPipelineConfig,
    is_cosmos_pwm_wam_package,
)
from sglang.multimodal_gen.configs.sample.cosmos_pwm_wam import (
    CosmosPWMWAMSamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.action import api, protocol
from sglang.multimodal_gen.runtime.entrypoints.action.cosmos_pwm_wam import (
    ActionVideoStore,
    lower_pwm_wam_request,
    pwm_wam_video_response,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# CPU tensor tests need the diffusion lane's optional dependencies.
register_cuda_ci(est_time=15, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")


class TestCosmosPWMWAMServing(CustomTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.spec = {
            "registry": {
                "domains": ["unit-source"],
                "domain_projection_ids": [0],
                "embodiments": ["unit-robot"],
                "domain_embodiments": [0],
            }
        }
        self.config = {
            "architectures": ["Cosmos3OmniModel"],
            "model": {
                "config": {
                    "tower_structure": "three_tower",
                    "action_tower_preset": "width_only_1b",
                    "action_tower_execution": "native",
                    "action_gen": True,
                    "vision_gen": True,
                    "sound_gen": False,
                    "action128_spec": self.spec,
                    "diffusion_expert_config": {
                        "vision_spatial_position_mode": "action128_seam"
                    },
                }
            },
        }
        (self.root / "config.json").write_text(json.dumps(self.config))
        (self.root / "model.safetensors.index.json").write_text('{"weight_map":{}}')
        Image.new("RGB", (32, 32)).save(self.root / "input.png")
        self.raw = {
            "model_mode": "wam",
            "vision_path": str(self.root / "input.png"),
            "prompt": "Pick up the cube",
            "seed": 17,
            "fps": 15,
            "action_chunk_size": 100,
            "extra": {
                "action128": {
                    "action_domain": "unit-source",
                    "action_dim_mask": [True, False] * 64,
                    "action_fps": 50,
                    "return_video": False,
                }
            },
        }
        self.server_args = SimpleNamespace(
            model_path=str(self.root),
            model_id=None,
            backend="sglang",
            pipeline_class_name="CosmosPWMWAMPipeline",
            pipeline_config=CosmosPWMWAMPipelineConfig(),
            served_model_name="pwm-test",
            output_path=None,
            comfyui_mode=False,
            attention_backend_config=None,
            enable_trace=False,
        )

    def test_native_artifact_routes_without_diffusers_index(self):
        """A neutral artifact directory must not fall back to ordinary Cosmos3."""
        from sglang.multimodal_gen import registry

        registry.get_model_info.cache_clear()
        registry._get_config_info.cache_clear()
        self.addCleanup(registry.get_model_info.cache_clear)
        self.addCleanup(registry._get_config_info.cache_clear)
        info = registry.get_model_info(str(self.root), backend="sglang")
        self.assertEqual(info.pipeline_cls.pipeline_name, "CosmosPWMWAMPipeline")
        self.assertIs(info.pipeline_config_cls, CosmosPWMWAMPipelineConfig)
        self.assertIs(info.sampling_param_cls, CosmosPWMWAMSamplingParams)
        self.config["model"]["config"]["tower_structure"] = "two_tower"
        (self.root / "config.json").write_text(json.dumps(self.config))
        self.assertFalse(is_cosmos_pwm_wam_package(str(self.root)))

    def test_python_and_http_share_independent_video_action_rates(self):
        """50 Hz actions over two seconds require 33 video frames at 15 FPS."""
        python_params = SamplingParams.from_user_sampling_params_args(
            str(self.root), self.server_args, wam_request=self.raw
        )
        http_params = protocol.build_action_sampling_params(
            {
                "input": {
                    "task": self.raw["prompt"],
                    "observation": {
                        "image": self.raw["vision_path"],
                    },
                },
                "parameters": {
                    "seed": 17,
                    "fps": 15,
                    "action_horizon": 100,
                    "extra": self.raw["extra"],
                },
            },
            self.server_args,
        )
        expected = python_params.build_request_extra()["cosmos_pwm_wam"]
        self.assertEqual(http_params.build_request_extra()["cosmos_pwm_wam"], expected)
        self.assertEqual(expected["num_frames"], 33)
        self.assertEqual(expected["action_chunk_size"], 100)
        self.assertFalse(expected["extra"]["action128"]["return_video"])
        self.assertNotIn("num_frames", self.raw)

    def test_prompt_file_and_random_seed_are_frozen_before_queueing(self):
        raw = copy.deepcopy(self.raw)
        raw.pop("prompt")
        raw.pop("seed")
        prompt_path = self.root / "prompt.txt"
        prompt_path.write_text("Move left")
        raw["prompt_path"] = str(prompt_path)
        params = CosmosPWMWAMSamplingParams(wam_request=raw)
        params._adjust(self.server_args)
        first = params.build_request_extra()
        prompt_path.write_text("Move right")
        params._adjust(self.server_args)
        self.assertEqual(params.build_request_extra(), first)
        self.assertEqual(first["cosmos_pwm_wam"]["prompt"], "Move left")
        direct = CosmosPWMWAMSamplingParams(
            vision_path=self.raw["vision_path"],
            prompt_path=str(prompt_path),
            extra=self.raw["extra"],
        )
        direct._adjust(self.server_args)
        self.assertEqual(direct.wam_request["prompt"], "Move right")

    def test_und_cache_default_and_request_overrides_reach_worker_contract(self):
        for default in (False, True):
            self.server_args.pipeline_config.enable_und_cache = default
            inherited = protocol.build_action_sampling_params(
                {"wam_request": self.raw}, self.server_args
            )
            self.assertIs(inherited.wam_request["enable_und_cache"], default)
            self.server_args.pipeline_config.enable_und_cache = not default
            inherited._adjust(self.server_args)
            self.assertIs(inherited.wam_request["enable_und_cache"], default)
            self.server_args.pipeline_config.enable_und_cache = default
            for enabled in (False, True):
                python_params = SamplingParams.from_user_sampling_params_args(
                    str(self.root),
                    self.server_args,
                    wam_request=self.raw,
                    enable_und_cache=enabled,
                )
                http_params = protocol.build_action_sampling_params(
                    {"wam_request": self.raw, "runtime": {"enable_und_cache": enabled}},
                    self.server_args,
                )
                direct_raw = {**self.raw, "enable_und_cache": enabled}
                raw_params = protocol.build_action_sampling_params(
                    direct_raw, self.server_args
                )
                expected = python_params.build_request_extra()
                self.assertEqual(http_params.build_request_extra(), expected)
                self.assertEqual(raw_params.build_request_extra(), expected)
                self.assertIs(expected["cosmos_pwm_wam"]["enable_und_cache"], enabled)
            metadata = self.server_args.pipeline_config.action_metadata(
                self.server_args
            )
            self.assertIs(metadata["defaults"]["enable_und_cache"], default)
            self.assertTrue(metadata["capabilities"]["request_local_und_cache"])
            self.assertFalse(metadata["capabilities"]["cross_request_und_cache"])
            self.assertFalse(metadata["capabilities"]["exact_prefix_cache"])
        self.assertNotIn("enable_und_cache", self.raw)

    def test_und_cache_override_rejects_ambiguous_or_nonboolean_values(self):
        for wrapped in (False, True):
            raw = {**self.raw, "enable_und_cache": False}
            payload = {"wam_request": raw} if wrapped else raw
            payload["runtime"] = {"enable_und_cache": False}
            with (
                self.subTest(wrapped=wrapped),
                self.assertRaisesRegex(ValueError, "both"),
            ):
                lower_pwm_wam_request(payload)
        for value in ("false", 0, None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                lower_pwm_wam_request(
                    {"wam_request": self.raw, "runtime": {"enable_und_cache": value}}
                )
        with self.assertRaises(TypeError):
            CosmosPWMWAMSamplingParams(enable_und_cache="false")

    def test_python_rejects_unused_controls_before_building_a_worker_request(self):
        for fields in (
            {"generator_device": "cpu"},
            {"save_output": True},
            {"return_file_paths_only": True},
            {"max_sequence_length": 8},
            {"profile": False},
            {"perf_dump_path": "unused.json"},
            {"data_type": "video"},
            {"no_override_protected_fields": True},
            {"diffusers_kwargs": {}},
        ):
            with (
                self.subTest(fields=fields),
                self.assertRaisesRegex(ValueError, "[Uu]nsupported|does not support"),
            ):
                SamplingParams.from_user_sampling_params_args(
                    str(self.root), self.server_args, wam_request=self.raw, **fields
                )
        for fields in (
            {"width": 999},
            {"height": 999},
            {"runtime": {"enable_und_cache": True}},
        ):
            with self.subTest(fields=fields), self.assertRaises(TypeError):
                SamplingParams.from_user_sampling_params_args(
                    str(self.root), self.server_args, wam_request=self.raw, **fields
                )

    def test_python_rejects_conflicting_raw_and_convenience_values(self):
        raw = {**self.raw, "guidance": 3.0, "num_steps": 5}
        for fields in (
            {"seed": 19},
            {"guidance_scale": 5.0},
            {"num_inference_steps": 7},
            {"prompt": "A different task"},
            {"action_fps": 20},
            {"return_video": True},
            {"extra": {"action128": {"return_video": True}}},
            {"prompt_path": str(self.root / "unused.txt")},
        ):
            with (
                self.subTest(fields=fields),
                self.assertRaisesRegex(
                    ValueError, "[Cc]onflicting|one Python input source"
                ),
            ):
                SamplingParams.from_user_sampling_params_args(
                    str(self.root), self.server_args, wam_request=raw, **fields
                )
        same = SamplingParams.from_user_sampling_params_args(
            str(self.root),
            self.server_args,
            wam_request=raw,
            seed=17,
            guidance_scale=3.0,
            return_video=False,
        )
        self.assertEqual(same.wam_request["seed"], 17)
        self.assertEqual(same.wam_request["guidance"], 3.0)

    def test_python_convenience_values_are_consumed_and_frozen(self):
        prompt_path = self.root / "prompt.txt"
        prompt_path.write_text("Move right")
        params = SamplingParams.from_user_sampling_params_args(
            str(self.root),
            self.server_args,
            request_id="python-wam",
            prompt_path=str(prompt_path),
            vision_path=os.path.relpath(self.root / "input.png"),
            seed=23,
            fps=15,
            action_horizon=100,
            num_frames=33,
            image_size=32,
            num_inference_steps=7,
            guidance_scale=2.0,
            flow_shift=5.0,
            enable_und_cache=True,
            extra=self.raw["extra"],
            return_video=False,
        )
        expected = params.build_request_extra()
        raw = expected["cosmos_pwm_wam"]
        self.assertEqual(params.request_id, "python-wam")
        self.assertEqual(raw["prompt"], "Move right")
        self.assertEqual(raw["vision_path"], str(self.root / "input.png"))
        self.assertEqual(
            {
                key: raw[key]
                for key in (
                    "seed",
                    "fps",
                    "action_chunk_size",
                    "num_frames",
                    "image_size",
                    "num_steps",
                    "guidance",
                    "shift",
                    "enable_und_cache",
                )
            },
            {
                "seed": 23,
                "fps": 15,
                "action_chunk_size": 100,
                "num_frames": 33,
                "image_size": 32,
                "num_steps": 7,
                "guidance": 2.0,
                "shift": 5.0,
                "enable_und_cache": True,
            },
        )
        self.assertFalse(raw["extra"]["action128"]["return_video"])
        prompt_path.write_text("Changed after admission")
        params._adjust(self.server_args)
        self.assertEqual(params.build_request_extra(), expected)

    def test_unsupported_optimization_settings_fail_during_server_resolution(self):
        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        kwargs = {
            "model_path": str(self.root),
            "backend": "sglang",
            "pipeline_class_name": "CosmosPWMWAMPipeline",
            "pipeline_config": {
                "vae_path": str(self.root),
                "tokenizer_path": str(self.root),
                "enable_und_cache": True,
            },
            "attention_backend": "torch_sdpa",
            "num_gpus": 1,
            "tp_size": 1,
            "sp_degree": 1,
            "batching_max_size": 1,
            "enable_cfg_parallel": False,
            "enable_torch_compile": False,
            "enable_breakable_cuda_graph": False,
            "dit_cpu_offload": False,
            "text_encoder_cpu_offload": False,
            "image_encoder_cpu_offload": False,
            "vae_cpu_offload": False,
            "dit_layerwise_offload": False,
            "use_fsdp_inference": False,
            "warmup_mode": "off",
        }
        resolved = ServerArgs.from_kwargs(**kwargs)
        self.assertTrue(resolved.pipeline_config.enable_und_cache)
        self.assertIsNone(resolved.cache_dit_config)
        self.assertFalse(resolved.regional_compile)
        self.assertFalse(resolved.enable_attention_backend_autotune)
        for field, value, message in (
            ("cache_dit_config", {}, "cache_dit_config"),
            ("regional_compile", True, "regional_compile"),
            ("enable_attention_backend_autotune", True, "autotuning"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                ServerArgs.from_kwargs(**kwargs, **{field: value})

    def test_unsupported_action_fields_fail_instead_of_disappearing(self):
        for parameters in (
            {"action_mode": "policy"},
            {"state": [0.0]},
            {"width": 480},
            {"num_outputs_per_prompt": 2},
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                lower_pwm_wam_request({"parameters": parameters})

    def test_raw_wrappers_cannot_hide_unsupported_runtime_or_sampling_fields(self):
        for wrapped in (False, True):
            for field in ("cuda_graph", "prefix_cache"):
                payload = {"wam_request": self.raw} if wrapped else dict(self.raw)
                payload["runtime"] = {field: True}
                with (
                    self.subTest(wrapped=wrapped, field=field),
                    self.assertRaisesRegex(ValueError, f"runtime.{field}"),
                ):
                    lower_pwm_wam_request(payload)
        with self.assertRaisesRegex(ValueError, "inside wam_request"):
            lower_pwm_wam_request({"wam_request": self.raw, "seed": 19})
        with self.assertRaisesRegex(ValueError, "envelope fields"):
            lower_pwm_wam_request({"input": {}, "parameters": {}, "n": 2})
        with self.assertRaisesRegex(ValueError, "supplied twice"):
            lower_pwm_wam_request({"input": {"seed": 17}, "parameters": {"seed": 19}})
        self.assertEqual(
            lower_pwm_wam_request(
                {
                    "wam_request": self.raw,
                    "request_id": "client-request",
                    "runtime": {"response_format": "raw", "prefix_cache": False},
                }
            ),
            self.raw,
        )

    def test_http_retains_action128_metadata_and_downloadable_joint_video(self):
        """JSON serialization must retain normalized values and same-rollout video."""
        app = FastAPI()
        app.state.server_args = self.server_args
        store = ActionVideoStore(self.root / "outputs")
        self.addCleanup(store.close)
        app.state.pwm_wam_video_store = store
        app.include_router(api.router)
        output = {
            "actions": np.ones((100, 128), dtype=np.float32),
            "normalized_actions": np.zeros((100, 128), dtype=np.float32),
            "action_mode": "wam",
            "action_domain": "unit-source",
            "action_dim_mask": [True, False] * 64,
            "action_fps": 50,
            "action_timestamps_seconds": list(np.arange(100) / 50),
            "action_units": "source",
            "seed": 17,
            "contract_fingerprint": "unit-contract",
            "checkpoint_config_sha256": "unit-config-digest",
            "asset_file_sha256": {"vae/config.json": "unit-vae-digest"},
            "loader_info": {"mode": "strict", "tensor_count": 12},
            "video_frames": np.zeros((33, 32, 32, 3), dtype=np.uint8),
            "video_fps": 15,
        }

        async def forward(_request):
            return OutputBatch(output=[output])

        def encode(path, _frames, **_kwargs):
            Path(path).write_bytes(b"unit-mp4")

        with (
            patch.object(protocol.async_scheduler_client, "forward", forward),
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.utils._save_video_ffmpeg",
                encode,
            ),
            TestClient(app) as client,
        ):
            response = client.post(
                "/v1/actions/generations", json={"wam_request": self.raw}
            )
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()["data"][0]
            self.assertEqual(data["action"]["shape"], [100, 128])
            self.assertEqual(data["action128"]["seed"], 17)
            self.assertEqual(
                data["action128"]["checkpoint_config_sha256"], "unit-config-digest"
            )
            self.assertEqual(data["action128"]["loader_info"]["tensor_count"], 12)
            self.assertEqual(
                data["action128"]["asset_file_sha256"]["vae/config.json"],
                "unit-vae-digest",
            )
            self.assertEqual(data["action128"]["normalized_actions"][0], [0.0] * 128)
            self.assertEqual(data["video"]["shape"], [33, 32, 32, 3])
            self.assertEqual(client.get(data["video"]["url"]).content, b"unit-mp4")
            video_id = data["video"]["id"]
            self.assertEqual(
                client.delete(f"/v1/actions/videos/{video_id}").status_code, 200
            )
            self.assertEqual(client.get(data["video"]["url"]).status_code, 404)
            raw_response = client.post(
                "/v1/actions/generations",
                json={"wam_request": self.raw, "runtime": {"response_format": "raw"}},
            )
            self.assertEqual(raw_response.status_code, 200, raw_response.text)
            self.assertNotIn("video_frames", raw_response.json())
            self.assertEqual(raw_response.json()["normalized_actions"][0], [0.0] * 128)
        self.assertFalse(store.directory.exists())

    def test_legacy_facade_schema_admission_and_shutdown(self):
        """Legacy clients retain SampleOutputs and a busy response instead of queueing."""
        from sglang.multimodal_gen.runtime.entrypoints.action.pwm_compat import (
            create_app,
        )

        entered, release = threading.Event(), threading.Event()
        actions = np.zeros((100, 128), dtype=np.float32)

        def run(parameters):
            entered.set()
            if not release.wait(timeout=10):
                raise RuntimeError("test generator release timed out")
            return {
                "actions": actions,
                "normalized_actions": actions,
                "action_mode": "wam",
                "action_domain": "unit-source",
                "action_dim_mask": [True, False] * 64,
                "seed": parameters["wam_request"]["seed"],
                "action_units": "raw_physical_or_source_units",
            }

        generator = SimpleNamespace(
            server_args=self.server_args,
            generate_action=run,
            shutdown=lambda: None,
        )
        app = create_app(generator, output_dir=self.root / "legacy")
        with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as pool:
            self.assertEqual(client.get("/models").json(), ["pwm-test"])
            self.assertEqual(client.get("/domains").json()["domains"], ["unit-source"])
            invalid = copy.deepcopy(self.raw)
            invalid["extra"]["action128"]["action_dim_mask"] = []
            self.assertEqual(client.post("/generate", json=invalid).status_code, 422)
            pending = pool.submit(client.post, "/generate", json=self.raw)
            try:
                self.assertTrue(entered.wait(timeout=10))
                busy = client.post("/generate", json=self.raw)
                self.assertEqual(busy.status_code, 503)
                self.assertEqual(busy.headers["retry-after"], "1")
            finally:
                release.set()
            response = pending.result(timeout=10)
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(body["status"], "success")
            self.assertIn("name", body["args"])
            self.assertEqual(body["outputs"][0]["files"], [])
            self.assertEqual(body["outputs"][0]["content"]["action128"]["seed"], 17)
            self.assertIn(
                "normalized_action", body["outputs"][0]["content"]["action128"]
            )
            self.assertEqual(client.get("/outputs/config.json").status_code, 404)
        self.assertFalse(app.state.pwm_wam_video_store.directory.exists())

    def test_video_download_survives_deletion_and_closes_on_disconnect(self):
        """An admitted download owns its file even after the URL is deleted."""
        app = FastAPI()
        app.state.server_args = self.server_args
        store = ActionVideoStore(self.root / "downloads")
        self.addCleanup(store.close)
        app.state.pwm_wam_video_store = store
        app.include_router(api.router)
        content = b"video-bytes" * 20000

        def encode(path, _frames, **_kwargs):
            Path(path).write_bytes(content)

        with (
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.utils._save_video_ffmpeg",
                encode,
            ),
            TestClient(app) as client,
        ):
            video = store.save_output(
                {"video_frames": np.zeros((1, 2, 2, 3), np.uint8), "video_fps": 20}
            )["video"]
            response = pwm_wam_video_response(store, video["id"])
            received = bytearray()

            async def receive():
                return {"type": "http.request"}

            async def send(message):
                if message["type"] == "http.response.body":
                    if not received:
                        store.delete(video["id"])
                        self.assertEqual(client.get(video["url"]).status_code, 404)
                    received.extend(message.get("body", b""))

            scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
            asyncio.run(response(scope, receive, send))
            self.assertEqual(received, content)
            self.assertTrue(response._stream.closed)

            # Cancellation before even starting the body must close the handle.
            video = store.save_output(
                {"video_frames": np.zeros((1, 2, 2, 3), np.uint8), "video_fps": 20}
            )["video"]
            response = pwm_wam_video_response(store, video["id"])

            async def disconnect(_message):
                raise asyncio.CancelledError

            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(response(scope, receive, disconnect))
            self.assertTrue(response._stream.closed)

            response = pwm_wam_video_response(store, video["id"])

            async def broken_send(message):
                if message["type"] == "http.response.body":
                    raise OSError("client socket closed during download")

            with self.assertRaises(ClientDisconnect):
                asyncio.run(response(scope, receive, broken_send))
            self.assertTrue(response._stream.closed)

    def test_other_wam_pipelines_keep_their_existing_action_schema(self):
        """The common WAM mode name does not opt other models into Action128."""
        args = SimpleNamespace(
            pipeline_config=SimpleNamespace(default_num_inference_steps=30),
            served_model_name="other-wam-model",
        )
        response = protocol.action_generation_response(
            {"actions": np.zeros((4, 7), dtype=np.float32), "action_mode": "wam"},
            args,
        )
        self.assertEqual(response["data"][0]["action"]["shape"], [4, 7])
        self.assertNotIn("action128", response["data"][0])


if __name__ == "__main__":
    unittest.main()
