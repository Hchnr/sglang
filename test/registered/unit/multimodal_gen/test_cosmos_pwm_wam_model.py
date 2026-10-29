# SPDX-License-Identifier: Apache-2.0
"""Derived WAM attention/masking invariants and strict artifact bookkeeping."""

import copy
import gc
import json
import tempfile
import unittest
import weakref
from functools import partial
from pathlib import Path
from unittest.mock import patch

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
    PwmRMSNorm,
    PwmThreeTowerLayer,
    PwmTimeEmbedding,
    PwmWamTransformer,
    attention,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos_pwm_wam.vae import (
    preserve_checkpoint_rms,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# CPU tensor tests need the diffusion lane's optional dependencies.
register_cuda_ci(est_time=8, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")


class TestPwmModelContracts(CustomTestCase):
    @staticmethod
    def tiny_hidden_model(dtype):
        model = PwmWamTransformer.__new__(PwmWamTransformer)
        nn.Module.__init__(model)
        model.language_model = nn.Module()
        stack = nn.Module()
        model.language_model.model = stack
        stack.embed_tokens = nn.Embedding(32, 16)
        stack.layers = nn.ModuleList(
            [
                PwmThreeTowerLayer(
                    width=16,
                    intermediate=32,
                    action_width=8,
                    action_intermediate=16,
                    heads=2,
                    kv_heads=1,
                    dim=8,
                )
                for _ in range(3)
            ]
        )
        stack.norm = PwmRMSNorm(16)
        stack.norm_moe_gen = PwmRMSNorm(16)
        stack.norm_moe_gen_action = PwmRMSNorm(8)
        return model.to(dtype).eval()

    @torch.inference_mode()
    def test_request_und_cache_matches_uncached_cfg_branches(self):
        """UND-only prefill preserves every layer's pre-update KV for both CFG branches."""
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(1214)
                model = self.tiny_hidden_model(dtype)
                stack = model.language_model.model
                branches = []
                # A/B/A checks both unequal CFG lengths and same-length requests
                # with different contents; token count is never a cache identity.
                for ids in ([2, 4, 6, 8, 10], [1, 3], [12, 14, 16, 18, 20]):
                    ids = torch.tensor(ids)
                    streams = (
                        stack.embed_tokens(ids),
                        torch.randn(4, 16, dtype=dtype),
                        torch.randn(2, 8, dtype=dtype),
                    )
                    ropes = []
                    for _ in range(2):
                        phases = [torch.randn(len(x), 4).repeat(1, 2) for x in streams]
                        ropes.append(
                            tuple(
                                (p.cos().to(dtype), p.sin().to(dtype)) for p in phases
                            )
                        )
                    rope, action_rope = ropes
                    branches.append(
                        (
                            streams,
                            rope,
                            action_rope,
                            model.build_und_cache(ids, rope[0]),
                        )
                    )

                snapshots = [
                    tuple(t.clone() for kv in cache.layers for t in kv)
                    + (cache.hidden.clone(),)
                    for _, _, _, cache in branches
                ]
                for branch_index in (0, 1, 2, 1, 0):
                    original, rope, action_rope, cache = branches[branch_index]
                    streams = (
                        original[0],
                        torch.randn_like(original[1]),
                        torch.randn_like(original[2]),
                    )
                    before = tuple(x.clone() for x in streams)
                    expected = model.forward_hidden(streams, rope, action_rope)
                    # Cached execution must actually skip UND projection/MLP,
                    # not just produce equal outputs through the old path.
                    calls = []

                    def record_call(module, args, calls=calls):
                        calls.append(module)

                    handles = [
                        module.register_forward_pre_hook(record_call)
                        for layer in stack.layers
                        for module in (layer.self_attn.q_proj, layer.mlp)
                    ]
                    try:
                        actual = model.forward_hidden(
                            streams, rope, action_rope, und_cache=cache
                        )
                    finally:
                        for handle in handles:
                            handle.remove()
                    self.assertEqual(calls, [])
                    for left, right in zip(actual, expected):
                        torch.testing.assert_close(left, right, atol=0, rtol=0)
                    for left, right in zip(streams, before):
                        torch.testing.assert_close(left, right, atol=0, rtol=0)
                    cached_tensors = tuple(t for kv in cache.layers for t in kv) + (
                        cache.hidden,
                    )
                    for left, right in zip(cached_tensors, snapshots[branch_index]):
                        torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_request_und_cache_has_no_model_owned_state_or_grad_graph(self):
        """Ending a request releases its KV even when the model continues serving."""
        torch.manual_seed(1214)
        model = self.tiny_hidden_model(torch.float32)
        state_keys = tuple(model.state_dict())
        ids = torch.tensor([2, 4, 6])
        rope = (torch.ones(3, 8), torch.zeros(3, 8))
        # Caller grad mode is intentionally enabled: prefill must not retain the
        # entire text tower's autograd graph alongside the small KV cache.
        cache = model.build_und_cache(ids, rope)
        tensors = tuple(t for kv in cache.layers for t in kv) + (cache.hidden,)
        for tensor in tensors:
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)
        references = [weakref.ref(t) for t in tensors]
        del tensor, tensors, cache
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual(tuple(model.state_dict()), state_keys)

    @torch.inference_mode()
    def test_compiled_layers_preserve_cfg_cache_and_checkpoint_state(self):
        """Static graphs must specialize on both CFG lengths and optional UND KV."""
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        torch.manual_seed(1214)
        model = self.tiny_hidden_model(torch.bfloat16)
        oracle = copy.deepcopy(model)
        saved = {name: value.clone() for name, value in model.state_dict().items()}
        graphs = []

        def record_graph(graph, inputs):
            graphs.append(graph)
            return graph.forward

        # Exercise Dynamo's real fullgraph tracing on CPU; GPU Inductor/kernel
        # numerical parity is covered by explicit-asset validation separately.
        with patch("torch.compile", partial(torch.compile, backend=record_graph)):
            model.configure_inference(enable_torch_compile=True)
        self.assertTrue(model.enable_torch_compile)
        branches = []
        for lengths in ((5, 4, 2), (2, 6, 3)):
            ids = torch.arange(lengths[0]) + 1
            streams = (
                model.language_model.model.embed_tokens(ids),
                torch.randn(lengths[1], 16, dtype=torch.bfloat16),
                torch.randn(lengths[2], 8, dtype=torch.bfloat16),
            )
            phases = [torch.randn(length, 4).repeat(1, 2) for length in lengths]
            rope = tuple((x.cos().bfloat16(), x.sin().bfloat16()) for x in phases)
            before = len(graphs)
            cache = model.build_und_cache(ids, rope[0])
            self.assertGreater(len(graphs), before, "UND prefill bypassed compilation")
            branches.append((streams, rope, cache))
        traced_variants = set()
        for index in (0, 1, 0):
            streams, rope, cache = branches[index]
            expected = oracle.forward_hidden(streams, rope, rope)
            for reuse in (False, True):
                before = len(graphs)
                actual = model.forward_hidden(
                    streams, rope, rope, und_cache=cache if reuse else None
                )
                if (index, reuse) not in traced_variants:
                    self.assertGreater(
                        len(graphs), before, "Denoising layer bypassed compilation"
                    )
                    traced_variants.add((index, reuse))
                for observed, reference in zip(actual, expected):
                    torch.testing.assert_close(observed, reference, atol=0, rtol=0)
        self.assertEqual(tuple(model.state_dict()), tuple(saved))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, saved[name], atol=0, rtol=0)
        model.configure_inference(enable_torch_compile=False)
        self.assertFalse(model.enable_torch_compile)
        streams, rope, _ = branches[0]
        for observed, reference in zip(
            model.forward_hidden(streams, rope, rope),
            oracle.forward_hidden(streams, rope, rope),
        ):
            torch.testing.assert_close(observed, reference, atol=0, rtol=0)

    @torch.inference_mode()
    def test_flash3_metadata_stays_with_its_cfg_branch(self):
        """UND is causal; both generation towers attend the joint unpadded KV."""
        from sglang.multimodal_gen.runtime.models.dits.cosmos_pwm_wam import execution

        torch.manual_seed(1214)
        model = self.tiny_hidden_model(torch.bfloat16)
        oracle = copy.deepcopy(model)

        def cpu_kernel(q, k, v, cu_q, cu_k, causal):
            # Mock only the GPU kernel boundary. Reject an incorrect varlen
            # description before computing the reference attention on CPU.
            self.assertEqual(cu_q.dtype, torch.int32)
            self.assertEqual(cu_k.dtype, torch.int32)
            self.assertEqual(cu_q.tolist(), [0, q.shape[0]])
            self.assertEqual(cu_k.tolist(), [0, k.shape[0]])
            return attention(q, k, v, causal=causal).reshape(q.shape)

        with (
            patch.object(execution, "validate_flash3"),
            patch.object(execution, "_flash3", side_effect=cpu_kernel),
        ):
            model.configure_inference(attention_backend="flash3")
            branches = []
            for lengths in ((5, 4, 2), (2, 6, 3)):
                ids = torch.arange(lengths[0]) + 1
                streams = (
                    model.language_model.model.embed_tokens(ids),
                    torch.randn(lengths[1], 16, dtype=torch.bfloat16),
                    torch.randn(lengths[2], 8, dtype=torch.bfloat16),
                )
                rope = tuple(
                    (torch.ones(n, 8).bfloat16(), torch.zeros(n, 8).bfloat16())
                    for n in lengths
                )
                metadata = model.build_attention_metadata(lengths, device="cpu")
                branches.append((ids, streams, rope, metadata))
            for index in (0, 1, 0):
                ids, streams, rope, metadata = branches[index]
                expected = oracle.forward_hidden(streams, rope, rope)
                cache = model.build_und_cache(ids, rope[0], attention_metadata=metadata)
                for reuse in (False, True):
                    actual = model.forward_hidden(
                        streams,
                        rope,
                        rope,
                        und_cache=cache if reuse else None,
                        attention_metadata=metadata,
                    )
                    for observed, reference in zip(actual, expected):
                        torch.testing.assert_close(observed, reference, atol=0, rtol=0)
            ids, streams, rope, metadata = branches[0]
            other = branches[1][-1]
            with self.assertRaisesRegex(ValueError, "length"):
                model.forward_hidden(streams, rope, rope, attention_metadata=other)
            with self.assertRaisesRegex(ValueError, "length"):
                model.build_und_cache(ids, rope[0], attention_metadata=other)
            with self.assertRaisesRegex(ValueError, "SDPA"):
                oracle.forward_hidden(streams, rope, rope, attention_metadata=metadata)

    @torch.inference_mode()
    def test_execution_errors_do_not_silently_select_another_backend(self):
        """Rejected policies and compiler errors must not report successful eager fallback."""
        model = self.tiny_hidden_model(torch.bfloat16)
        for settings, message in (
            ({"attention_backend": "auto"}, "attention_backend"),
            ({"enable_torch_compile": "false"}, "boolean"),
            ({"attention_backend": "flash3"}, "CUDA BF16"),
        ):
            with (
                self.subTest(settings=settings),
                self.assertRaisesRegex(ValueError, message),
            ):
                model.configure_inference(**settings)
            self.assertEqual(model.attention_backend, "sdpa")
            self.assertFalse(model.enable_torch_compile)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        limits = (
            torch._dynamo.config.recompile_limit,
            torch._dynamo.config.accumulated_recompile_limit,
        )

        def fail_compile(graph, inputs):
            raise RuntimeError("intentional compiler failure")

        with patch("torch.compile", partial(torch.compile, backend=fail_compile)):
            model.configure_inference(enable_torch_compile=True)
        ids = torch.tensor([1, 2, 3])
        rope = (torch.ones(3, 8).bfloat16(), torch.zeros(3, 8).bfloat16())
        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed, "intentional compiler failure"
        ):
            model.build_und_cache(ids, rope)
        self.assertEqual(
            (
                torch._dynamo.config.recompile_limit,
                torch._dynamo.config.accumulated_recompile_limit,
            ),
            limits,
        )

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
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.embed_tokens = nn.Embedding(3, 4)
        self.language_model.lm_head = nn.Linear(4, 3, bias=False)
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
            "model.net.language_model.model.embed_tokens.weight": torch.arange(12)
            .reshape(3, 4)
            .to(torch.bfloat16),
            # Deliberately distinct: the unused head must remain lossless, not
            # be retied to or substituted for the live embedding table.
            "model.net.language_model.lm_head.weight": -torch.arange(12)
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
            state = model.state_dict()
            self.assertEqual(
                set(state), {key.removeprefix("model.net.") for key in tensors}
            )
            for name, actual in state.items():
                torch.testing.assert_close(
                    actual, tensors["model.net." + name], atol=0, rtol=0
                )
            self.assertNotEqual(
                model.language_model.lm_head.weight.data_ptr(),
                model.language_model.model.embed_tokens.weight.data_ptr(),
            )
            self.assertEqual(model.load_info["loaded_tensor_bytes"], 88)
            self.assertEqual(model.load_info["cuda_resident_tensor_bytes"], 0)
            self.assertEqual(model.load_info["unused_text_head_cpu_bytes"], 24)

            # Meta is a non-CPU destination available in the CPU lane. This
            # catches accidentally retaining the used embedding on CPU along
            # with the text head, without requiring a GPU allocation.
            placed = TinyArtifactTransformer.from_pretrained(root, device="meta")
            self.assertEqual(placed.language_model.lm_head.weight.device.type, "cpu")
            self.assertEqual(
                placed.language_model.model.embed_tokens.weight.device.type, "meta"
            )
            self.assertEqual(placed.projection.weight.device.type, "meta")
            self.assertEqual(placed.load_info["cuda_resident_tensor_bytes"], 0)
            torch.testing.assert_close(
                placed.language_model.lm_head.weight,
                tensors["model.net.language_model.lm_head.weight"],
                atol=0,
                rtol=0,
            )

            for invalid, error in (
                (torch.zeros(3, 4), "must be BF16"),
                (torch.zeros(4, 3, dtype=torch.bfloat16), "HF shape mismatch"),
            ):
                save_file(
                    {**tensors, "model.net.language_model.lm_head.weight": invalid},
                    root / "weights.safetensors",
                )
                with (
                    self.subTest(unused_head_error=error),
                    self.assertRaisesRegex(ValueError, error),
                ):
                    TinyArtifactTransformer.from_pretrained(root, device="cpu")
            save_file(tensors, root / "weights.safetensors")
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
