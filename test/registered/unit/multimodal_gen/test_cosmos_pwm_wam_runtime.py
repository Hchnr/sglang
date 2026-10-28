# SPDX-License-Identifier: Apache-2.0
"""WAM media/caption protocol and joint solver invariants, without model weights."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_pwm_wam.request import (
    load_request,
    prepare_request,
    read_observation,
    validate_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_pwm_wam.runtime import (
    PwmWamRuntime,
    guided_velocity,
    initial_joint_state,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from torch import nn

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _VelocityModel(nn.Module):
    """Weight-loading boundary: expose nonzero fixed-coordinate velocities."""

    def __init__(self):
        super().__init__()
        self.observations = []

    def encode_inputs(self, ids, video, action, *args, action_dim_mask):
        self.observations.append((video.clone(), action.clone()))
        return ids, video, action

    def forward_hidden(self, streams, rope, action_rope):
        return streams

    def decode_outputs(self, streams, *args, action_dim_mask):
        ids, video, action = streams
        factor = float(ids[0])
        return torch.full_like(video, factor), torch.full_like(action, 2 * factor)


class _PerturbingScheduler:
    """A solver may perturb fixed coordinates; the runtime must restore them."""

    timesteps = torch.tensor([900, 500, 100])

    def step(self, *, model_output, sample, **kwargs):
        return (sample - model_output / 10 + 0.25,)


class _TinyRuntime(PwmWamRuntime):
    def __init__(self):
        nn.Module.__init__(self)
        self.device = torch.device("cpu")
        self.transformer = _VelocityModel()
        self.contract_fingerprint = "test"

    def text_ids(self, caption):
        return torch.tensor([1 if caption else -1])

    def _new_scheduler(self, request):
        return _PerturbingScheduler()


class TestPwmRuntimeContracts(CustomTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.image = self.directory / "observation.png"
        Image.fromarray(np.full((32, 48, 3), 31, dtype=np.uint8)).save(self.image)
        self.spec = {
            "registry": {
                "domains": ["test", "other"],
                "embodiments": ["arm"],
                "domain_embodiments": [0, 0],
                "domain_projection_ids": [1, 0],
            }
        }
        mask = [False] * 128
        for slot in (0, 5, 127):
            mask[slot] = True
        self.raw = {
            "model_mode": "wam",
            "vision_path": str(self.image),
            "prompt": "Move the block",
            "seed": 17,
            "extra": {
                "action128": {
                    "action_domain": "test",
                    "action_dim_mask": mask,
                    "return_video": False,
                }
            },
        }

    def test_independent_action_rate_owns_horizon_and_video_rounding(self):
        """The action rate, not the display rate, determines duration and T=4k+1."""
        self.raw.update(fps=15, action_chunk_size=100)
        self.raw["extra"]["action128"]["action_fps"] = 50
        request = prepare_request(self.raw, self.spec)
        self.assertEqual((request.video_frames, request.action_length), (33, 100))
        self.assertEqual((request.normalization_id, request.projection_id), (0, 1))
        caption = json.loads(request.caption)
        self.assertEqual(caption["duration"], "2s")
        self.assertEqual(caption["action_frequency_hz"], 50)
        self.raw["action_chunk_size"] = 101
        with self.assertRaisesRegex(ValueError, "horizon"):
            prepare_request(self.raw, self.spec)

    def test_caption_geometry_uses_resized_canvas_without_mutating_request(self):
        """Caption overrides must track real multi-camera seams, retaining task data."""
        Image.fromarray(np.zeros((48, 96, 3), dtype=np.uint8)).save(self.image)
        self.raw.update(
            image_size=32,
            prompt=json.dumps(
                {
                    "command": "Move",
                    "task_title": "A task",
                    "task_plan": ["retain this"],
                    "resolution": {"H": 1, "W": 1},
                    "custom": "retained",
                }
            ),
        )
        self.raw["extra"]["action128"].update(
            selected_roles=["main", "left", "right"],
            camera_layout=[
                ["main", "top", 96, 24],
                ["left", "bottom_left", 48, 24],
                ["right", "bottom_right", 48, 24],
            ],
        )
        original = copy.deepcopy(self.raw)
        request = prepare_request(self.raw, self.spec)
        caption = json.loads(request.caption)
        self.assertEqual(request.pixels.shape, (3, 1, 32, 64))
        self.assertEqual(request.seam, 0.5)
        self.assertEqual(caption["resolution"], {"W": 64, "H": 32})
        self.assertEqual(caption["camera_layout"][0], ["main", "top", 64, 16])
        self.assertEqual(caption["task_plan"], ["retain this"])
        self.assertEqual(caption["custom"], "retained")
        self.assertEqual(self.raw, original)

    def test_admission_freezes_seed_and_file_prompt_without_fetching_media(self):
        """Queued execution cannot resample a seed or re-read a changed prompt file."""
        prompt = self.directory / "caption.txt"
        prompt.write_text("Original caption")
        raw = copy.deepcopy(self.raw)
        raw.pop("prompt")
        raw.pop("seed")
        raw.update(prompt_path="caption.txt", vision_path="observation.png")
        path = self.directory / "request.json"
        path.write_text(json.dumps(raw))
        admitted = validate_request(load_request(path), self.spec)
        prompt.write_text("Changed caption")
        request = prepare_request(admitted, self.spec)
        self.assertEqual(request.seed, admitted["seed"])
        self.assertEqual(json.loads(request.caption)["command"], "Original caption")
        self.assertEqual((request.steps, request.guidance, request.shift), (30, 1, 10))
        raw.update(vision_path="https://example.invalid/observation.mp4", prompt="Move")
        with patch(
            "requests.get", side_effect=AssertionError("admission fetched media")
        ):
            validated = validate_request(raw, self.spec)
        self.assertEqual(validated["vision_path"], raw["vision_path"])

    def test_video_and_url_inputs_use_only_the_first_rgb_frame(self):
        """Video observations must equal the same image observation at frame zero."""
        import av

        first = np.full((32, 48, 3), 31, dtype=np.uint8)
        video = self.directory / "observation.mkv"
        with av.open(str(video), "w") as container:
            stream = container.add_stream("ffv1", rate=24)
            stream.width, stream.height, stream.pix_fmt = 48, 32, "bgr0"
            for pixels in (first, np.full_like(first, 240)):
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        torch.testing.assert_close(
            read_observation(video), read_observation(self.image)
        )

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size):
                yield video.read_bytes()

        with patch("requests.get", return_value=Response()):
            downloaded = read_observation("https://example.invalid/frame.mkv")
        torch.testing.assert_close(downloaded, read_observation(self.image))

    def test_rejects_requests_that_would_silently_change_conditioning(self):
        """The no-state, one-output contract must fail before GPU admission."""
        for invalid in (
            {"model_mode": "av"},
            {"condition_frame_indexes_vision": [1]},
            {"action_path": "state.npy"},
            {"autoregressive": True},
            {"prompt_upsampling": True},
            {"skip_text_tokens_for_cfg": True},
            {"sampler": "edm"},
            {"sigma_max": 1},
            {"num_outputs": 2},
            {"num_frames": 7},
            {"negative_prompt": 0},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_request({**self.raw, **invalid}, self.spec)
        for invalid_mask in ([False] * 128, [1] * 128):
            raw = copy.deepcopy(self.raw)
            raw["extra"]["action128"]["action_dim_mask"] = invalid_mask
            with self.assertRaisesRegex(ValueError, "action_dim_mask"):
                validate_request(raw, self.spec)

    def test_noise_streams_and_fixed_coordinates(self):
        """VIDEO/ACTION reset NumPy independently; holes and conditioning are fixed."""
        request = prepare_request(self.raw, self.spec)
        clean = torch.zeros(48, 3, 2, 2)
        clean[:, :1] = 7
        latent, fixed, reference = initial_joint_state(request, clean)
        noise = np.random.RandomState(request.seed).standard_normal((16, 128))
        action = latent[clean.numel() :].reshape(16, 128)
        active = torch.tensor(request.mask)
        np.testing.assert_array_equal(
            action[:, active], noise[:, request.mask].astype(np.float32)
        )
        self.assertEqual(torch.count_nonzero(action[:, ~active]).item(), 0)
        torch.testing.assert_close(latent[fixed], reference[fixed], atol=0, rtol=0)

    def test_cfg_norm_is_joint_and_cannot_amplify(self):
        """Independent modality normalization gives a different direction/magnitude."""
        conditional = torch.tensor([3.0, 4.0])
        unconditional = torch.tensor([-3.0, 4.0])
        actual = guided_velocity(conditional, unconditional, 2, True)
        expected = torch.tensor([9.0, 4.0]) * (5 / 97**0.5)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.norm(), conditional.norm())
        unscaled = guided_velocity(conditional, unconditional, 0.5, True)
        torch.testing.assert_close(unscaled, torch.tensor([0.0, 4.0]))

    def test_joint_solver_preserves_masks_and_request_isolation(self):
        """Fixed slots survive every solver update and A/B/A cannot leak histories."""
        self.raw.update(guidance=3, normalize_cfg=True)
        request = prepare_request(self.raw, self.spec)
        runtime = _TinyRuntime()
        clean = torch.zeros(48, 3, 2, 2)
        clean[:, :1] = 7
        video_a, action_a = runtime.sample(request, clean)
        request_b = prepare_request({**self.raw, "seed": 31}, self.spec)
        _, action_b = runtime.sample(request_b, clean)
        video_again, action_again = runtime.sample(request, clean)
        torch.testing.assert_close(video_a, video_again, atol=0, rtol=0)
        torch.testing.assert_close(action_a, action_again, atol=0, rtol=0)
        self.assertFalse(torch.equal(action_a, action_b))
        active = torch.tensor(request.mask)
        for video, action in [*runtime.transformer.observations, (video_a, action_a)]:
            torch.testing.assert_close(video[:, :1], clean[:, :1], atol=0, rtol=0)
            self.assertEqual(torch.count_nonzero(action[:, ~active]).item(), 0)
        # Fixed-coordinate velocities cannot enter the denominator of joint CFG.
        latent, _, _ = initial_joint_state(request, clean)
        initial = latent[clean.numel() :].reshape(request.action_length, 128)
        torch.testing.assert_close(action_a[:, active], initial[:, active] + 0.15)

    def test_guidance_interval_uses_raw_unipc_time_and_open_boundaries(self):
        """C's interval covers the raw 0..1000 timestep, excluding both endpoints."""
        self.raw.update(guidance=2, guidance_interval=[100, 900], sigma_max=80)
        request = prepare_request(self.raw, self.spec)
        clean = torch.zeros(48, 3, 2, 2)
        runtime = _TinyRuntime()
        _, action = runtime.sample(request, clean)
        latent, _, _ = initial_joint_state(request, clean)
        initial = latent[clean.numel() :].reshape(request.action_length, 128)
        active = torch.tensor(request.mask)
        # Two conditional updates (+.05) and one CFG update (-.35).
        torch.testing.assert_close(action[:, active], initial[:, active] - 0.25)

    def test_finalization_uses_domain_stats_and_fp32_video_conversion(self):
        """Projection id is not a normalizer index; source VAE returns FP32 pixels."""
        self.raw["extra"]["action128"]["return_video"] = True
        request = prepare_request(self.raw, self.spec)
        runtime = _TinyRuntime()
        scale = torch.full((2, 128), 2.0)
        bias = torch.full((2, 128), 1.0)
        scale[1], bias[1] = 3, 10
        normalization_mask = torch.zeros(2, 128, dtype=torch.bool)
        normalization_mask[:, 0] = True
        runtime.transformer.action_normalization = SimpleNamespace(
            scale=scale, bias=bias, normalization_mask=normalization_mask
        )
        decoded = torch.tensor([0.01171875, 0.25, -0.25], dtype=torch.bfloat16).view(
            1, 3, 1, 1, 1
        )
        runtime.vae = SimpleNamespace(decode=lambda *args, **kwargs: (decoded,))
        runtime.vae_scale = lambda: (torch.tensor(0), torch.tensor(1))
        result = runtime._finalize(
            request, torch.zeros(48, 1, 1, 1), torch.full((16, 128), 5.0)
        )
        self.assertEqual(result["actions"][0, 0], 2)
        self.assertEqual(result["actions"][0, 5], 5)
        self.assertEqual(result["actions"][0, 1], 0)
        np.testing.assert_array_equal(result["video_frames"][0, 0, 0], [128, 159, 95])


if __name__ == "__main__":
    unittest.main()
