# SPDX-License-Identifier: Apache-2.0
"""Derived WAM attention/masking invariants and strict artifact bookkeeping."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm
from safetensors.torch import save_file
from torch import nn

from sglang.multimodal_gen.runtime.models.dits.cosmos_pwm_wam.positions import (
    position_ids,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos_pwm_wam.scheduler import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos_pwm_wam.transformer import (
    PwmDomainLinear,
    PwmThreeTowerLayer,
    PwmTimeEmbedding,
    PwmWamTransformer,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos_pwm_wam.vae import (
    preserve_checkpoint_rms,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# CPU tensor tests need the diffusion lane's optional dependencies.
register_cuda_ci(est_time=8, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")


class TestPwmModelContracts(CustomTestCase):
    @torch.inference_mode()
    def test_three_tower_visibility_and_causal_und(self):
        """Gen changes cannot reach UND, but VIDEO and ACTION exchange information."""
        torch.manual_seed(1214)
        layer = PwmThreeTowerLayer(
            width=16,
            intermediate=32,
            action_width=8,
            action_intermediate=16,
            heads=2,
            kv_heads=1,
            dim=8,
        ).eval()
        streams = (torch.randn(3, 16), torch.randn(4, 16), torch.randn(2, 8))
        rope = tuple((torch.ones(len(x), 8), torch.zeros(len(x), 8)) for x in streams)
        original = layer(streams, rope, rope)
        action_changed = (streams[0], streams[1], -streams[2])
        video_changed = (streams[0], -streams[1], streams[2])
        from_action = layer(action_changed, rope, rope)
        from_video = layer(video_changed, rope, rope)
        torch.testing.assert_close(original[0], from_action[0], atol=0, rtol=0)
        torch.testing.assert_close(original[0], from_video[0], atol=0, rtol=0)
        self.assertGreater((original[1] - from_action[1]).abs().max().item(), 1e-5)
        self.assertGreater((original[2] - from_video[2]).abs().max().item(), 1e-5)
        future = streams[0].clone()
        future[-1] = -future[-1]
        changed = layer((future, *streams[1:]), rope, rope)
        torch.testing.assert_close(original[0][:-1], changed[0][:-1], atol=0, rtol=0)

    @torch.inference_mode()
    def test_inactive_actions_never_reach_projections_or_velocity(self):
        """Regression: the old native backend masked outputs only, leaking noise into active slots."""
        torch.manual_seed(1214)
        model = PwmWamTransformer.__new__(PwmWamTransformer)
        nn.Module.__init__(model)
        model.vae2llm = nn.Linear(192, 8)
        model.llm2vae = nn.Linear(8, 192)
        model.time_embedder = PwmTimeEmbedding(8)
        model.action_time_embedder = PwmTimeEmbedding(4)
        model.action2llm = PwmDomainLinear(128, 4)
        model.llm2action = PwmDomainLinear(4, 128)
        model.action_modality_embed = nn.Parameter(torch.zeros(4))
        model.language_model = nn.Module()
        model.language_model.model = nn.Module()
        model.language_model.model.embed_tokens = nn.Embedding(16, 8)
        mask = torch.zeros(128, dtype=torch.bool)
        mask[[0, 5, 127]] = True
        action = torch.randn(3, 128).masked_fill(~mask, 0)
        polluted = action.masked_fill(~mask, float("nan"))
        video = torch.randn(48, 2, 3, 5)
        args = (torch.tensor([1, 2]), video)
        tail = (
            torch.tensor([500]),
            torch.zeros(3, dtype=torch.long),
            torch.tensor([1]),
            torch.arange(3),
        )
        clean_streams = model.encode_inputs(*args, action, *tail, action_dim_mask=mask)
        observed = model.encode_inputs(*args, polluted, *tail, action_dim_mask=mask)
        for clean, actual in zip(clean_streams, observed):
            torch.testing.assert_close(clean, actual, atol=0, rtol=0)
        velocity_video, velocity_action = model.decode_outputs(
            observed, tuple(video.shape), tail[1], tail[2], action_dim_mask=mask
        )
        self.assertEqual(torch.count_nonzero(velocity_action[:, ~mask]).item(), 0)
        self.assertEqual(torch.count_nonzero(velocity_video[:, :1]).item(), 0)
        self.assertTrue(torch.isfinite(velocity_action).all().item())
        self.assertEqual(tuple(velocity_video.shape), tuple(video.shape))

    def test_action_video_time_alignment_and_seam(self):
        """50 Hz actions and 15 FPS video share time zero despite different token grids."""
        text, video, action = position_ids(7, (9, 3, 2), 100, 15, 50, seam=1.5)
        self.assertEqual(text.shape, (3, 7))
        self.assertEqual(video[0, 0].item(), 15007)
        self.assertEqual(action[0, 0].item(), 15007)
        # One compressed video frame is 4/15 seconds, independent of action FPS.
        self.assertAlmostEqual((video[0, 6] - video[0, 0]).item(), 1.6, delta=0.001)
        self.assertAlmostEqual((action[0, 50] - action[0, 0]).item(), 6, delta=0.001)
        torch.testing.assert_close(
            video[1, :6], torch.tensor([1.5, 1.5, 0.5, 0.5, -0.5, -0.5])
        )
        self.assertEqual(torch.count_nonzero(action[1:]).item(), 0)

    def test_time_embedding_keeps_fp32_under_bf16_autocast(self):
        """A generic module dtype cast must not round nonpersistent time frequencies."""
        torch.manual_seed(1214)
        layer = PwmTimeEmbedding(8).to(torch.bfloat16)
        self.assertEqual(layer.frequencies.dtype, torch.float32)
        time = torch.tensor([0.0, 0.125, 1.0])
        expected = layer(time)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = layer(time)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_unipc_constant_flow_and_reused_schedule_reset(self):
        """A constant vector field integrates analytically; A/B/A cannot retain solver history."""
        scheduler = FlowUniPCMultistepScheduler(
            use_cpu_coefficients=True, use_legacy_corrector=True
        )
        initial = torch.linspace(-2, 2, 12).reshape(1, 12)
        velocity = torch.linspace(-0.5, 0.5, 12)
        results = []
        for steps, shift, scale in ((5, 5.0, 1.0), (10, 3.0, 2.0), (5, 5.0, 1.0)):
            scheduler.set_timesteps(steps, device="cpu", shift=shift)
            sigma_start = scheduler.sigmas[0].clone()
            state = initial.clone()
            for time in scheduler.timesteps:
                state = scheduler.step(
                    velocity * scale, time, state, return_dict=False
                )[0]
            torch.testing.assert_close(
                state, initial - sigma_start * velocity * scale, atol=2e-6, rtol=2e-6
            )
            results.append(state)
        torch.testing.assert_close(results[0], results[2], atol=0, rtol=0)

    def test_vae_normalization_preserves_checkpoint_bf16_arithmetic(self):
        """A Diffusers upgrade must not silently upcast the conditioning norm."""
        torch.manual_seed(1214)
        module = nn.Sequential(WanRMS_norm(4, images=False, bias=True)).to(
            torch.bfloat16
        )
        value = torch.randn(2, 4, 3, 2, 2).to(torch.bfloat16)
        norm = module[0]
        # Frozen checkpoint convention: normalize in the input dtype.
        expected = (
            torch.nn.functional.normalize(value, dim=1) * norm.scale * norm.gamma
            + norm.bias
        )
        self.assertEqual(preserve_checkpoint_rms(module), 1)
        torch.testing.assert_close(module(value), expected, atol=0, rtol=0)


class TinyArtifactTransformer(PwmWamTransformer):
    """Synthetic tensors exercise the strict loader without any model checkpoint."""

    def __init__(self, num_domains):
        nn.Module.__init__(self)
        self.projection = nn.Linear(4, 3, bias=False)
        self.register_buffer("contract_digest", torch.empty(16, dtype=torch.uint8))


class TestPwmArtifactValidation(CustomTestCase):
    def test_complete_key_shape_and_dtype_contract(self):
        """Reject corrupted indexes and shard extras rather than silently dropping learned weights."""
        config = {
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
            "vlm_config": {
                "use_system_prompt": False,
                "model_name": "Qwen/Qwen3-VL-8B-Instruct",
                "model_instance": {
                    "config": {"qk_norm_for_text": True, "include_visual": False},
                },
            },
            "rectified_flow_inference_config": {
                "num_train_timesteps": 1000,
                "scheduler_type": "unipc",
                "shift": 1,
                "use_dynamic_shifting": False,
            },
            "action128_spec": {"registry": {"domains": ["test"]}},
            "diffusion_expert_config": {
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
            },
        }
        tensors = {
            "model.net.projection.weight": torch.arange(12)
            .reshape(3, 4)
            .to(torch.bfloat16),
            "model.net.contract_digest": torch.arange(16, dtype=torch.uint8),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"model": {"config": config}}))
            index = root / "model.safetensors.index.json"
            mapping = {key: "weights.safetensors" for key in tensors}
            index.write_text(json.dumps({"weight_map": mapping}))
            save_file(tensors, root / "weights.safetensors")
            model = TinyArtifactTransformer.from_pretrained(root, device="cpu")
            torch.testing.assert_close(
                model.projection.weight,
                tensors["model.net.projection.weight"],
                atol=0,
                rtol=0,
            )
            self.assertEqual(model.contract_digest.dtype, torch.uint8)
            for section, name, value in (
                ("vlm_config", "use_system_prompt", True),
                ("vlm_config", "model_name", "Qwen/another-model"),
                ("rectified_flow_inference_config", "scheduler_type", "edm"),
                ("rectified_flow_inference_config", "num_train_timesteps", 500),
                ("rectified_flow_inference_config", "use_dynamic_shifting", True),
            ):
                incompatible = copy.deepcopy(config)
                incompatible[section][name] = value
                (root / "config.json").write_text(
                    json.dumps({"model": {"config": incompatible}})
                )
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(ValueError, f"{section}.{name}"),
                ):
                    TinyArtifactTransformer.from_pretrained(root, device="cpu")
            for name, value in (("qk_norm_for_text", False), ("include_visual", True)):
                incompatible = copy.deepcopy(config)
                incompatible["vlm_config"]["model_instance"]["config"][name] = value
                (root / "config.json").write_text(
                    json.dumps({"model": {"config": incompatible}})
                )
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                    TinyArtifactTransformer.from_pretrained(root, device="cpu")
            (root / "config.json").write_text(json.dumps({"model": {"config": config}}))
            for variant, error in (
                (
                    {
                        "weight_map": {
                            "model.net.contract_digest": "weights.safetensors"
                        }
                    },
                    "HF tensor mismatch",
                ),
                (
                    {
                        "weight_map": {
                            **mapping,
                            "model.net.extra": "weights.safetensors",
                        }
                    },
                    "HF tensor mismatch",
                ),
            ):
                with self.subTest(error=error, variant=variant):
                    index.write_text(json.dumps(variant))
                    with self.assertRaisesRegex(ValueError, error):
                        TinyArtifactTransformer.from_pretrained(root, device="cpu")
            index.write_text(json.dumps({"weight_map": mapping}))
            save_file(
                {**tensors, "hidden_extra": torch.zeros(1)},
                root / "weights.safetensors",
            )
            with self.assertRaisesRegex(ValueError, "shard contents"):
                TinyArtifactTransformer.from_pretrained(root, device="cpu")
            save_file(
                {**tensors, "model.net.projection.weight": torch.zeros(3, 4)},
                root / "weights.safetensors",
            )
            with self.assertRaisesRegex(ValueError, "must be BF16"):
                TinyArtifactTransformer.from_pretrained(root, device="cpu")
            save_file(
                {
                    **tensors,
                    "model.net.projection.weight": torch.zeros(
                        4, 3, dtype=torch.bfloat16
                    ),
                },
                root / "weights.safetensors",
            )
            with self.assertRaisesRegex(ValueError, "HF shape mismatch"):
                TinyArtifactTransformer.from_pretrained(root, device="cpu")
            save_file(
                {**tensors, "model.net.contract_digest": torch.zeros(16)},
                root / "weights.safetensors",
            )
            with self.assertRaisesRegex(ValueError, "buffer dtype mismatch"):
                TinyArtifactTransformer.from_pretrained(root, device="cpu")
            index.write_text(
                json.dumps(
                    {"weight_map": {key: "../weights.safetensors" for key in tensors}}
                )
            )
            with self.assertRaisesRegex(ValueError, "inside the artifact directory"):
                TinyArtifactTransformer.from_pretrained(root, device="cpu")
            index.write_text('{"weight_map":{"same":"a","same":"b"}}')
            with self.assertRaisesRegex(ValueError, "Duplicate checkpoint JSON key"):
                TinyArtifactTransformer.from_pretrained(root, device="cpu")


if __name__ == "__main__":
    unittest.main()
