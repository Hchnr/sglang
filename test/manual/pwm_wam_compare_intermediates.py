# SPDX-License-Identifier: Apache-2.0
"""Run native WAM and compare explicit Cosmos reference intermediates.

Run with SGLang's environment; no source-framework import is required. The
reference directory comes from pwm_wam_capture_reference.py. Exact equality and
numerical errors are reported separately, without choosing an application-level
tolerance. An explicit frozen-clean-latent option isolates the model and sampler
from VAE differences; the report records this substitution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--enable-reference-sdpa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match production WAM's local SDPA policy; disable only to reproduce the old platform-default drift",
    )
    parser.add_argument("--reference-video", type=Path)
    parser.add_argument("--use-reference-clean-latents", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference_manifest = json.loads((args.reference / "manifest.json").read_text())
    if reference_manifest["status"] != "complete":
        raise ValueError("Reference capture did not complete")
    request_digest = hashlib.sha256(args.request.read_bytes()).hexdigest()
    if request_digest != reference_manifest["request_sha256"]:
        raise ValueError("Reference capture used a different request")

    import diffusers
    import numpy as np
    import torch
    import transformers
    from safetensors.torch import load_file, save_file
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_pwm_wam import (
        runtime as runtime_module,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_pwm_wam.request import (
        load_request,
        prepare_request,
    )

    torch.cuda.set_device(0)
    observed = {}
    metrics = {}
    cache = {}

    def capture(name, value):
        observed[name] = value.detach().cpu().contiguous().clone()

    def reference(filename, key):
        if filename not in cache:
            cache[filename] = load_file(args.reference / f"{filename}.safetensors")
        return cache[filename][key]

    def compare(name, expected):
        actual = observed[name]
        if actual.shape != expected.shape:
            metrics[name] = {
                "shape_mismatch": [list(actual.shape), list(expected.shape)]
            }
            return
        actual64, expected64 = actual.double(), expected.double()
        delta = actual64 - expected64
        metrics[name] = {
            "shape": list(actual.shape),
            "actual_dtype": str(actual.dtype),
            "expected_dtype": str(expected.dtype),
            "exact": bool(torch.equal(actual, expected)),
            "finite": bool(
                torch.isfinite(actual).all() and torch.isfinite(expected).all()
            ),
            "max_abs": delta.abs().max().item() if delta.numel() else 0,
            "mae": delta.abs().mean().item() if delta.numel() else 0,
            "rmse": delta.square().mean().sqrt().item() if delta.numel() else 0,
            "relative_l2": (delta.norm() / expected64.norm().clamp_min(1e-12)).item(),
        }

    started = time.perf_counter()
    runtime = runtime_module.PwmWamRuntime(
        args.checkpoint, vae_path=str(args.vae), tokenizer_path=str(args.tokenizer)
    )
    request = prepare_request(load_request(args.request), runtime.spec)
    capture("pixels", request.pixels)
    compare(
        "pixels",
        reference("request_preparation", "request_preparation.batch.video.0.0")[:, :1],
    )
    prepared = json.loads((args.reference / "request_preparation.json").read_text())
    if request.caption != prepared["batch"]["ai_caption"][0]:
        raise ValueError("Conditioning captions differ")
    capture("token_ids", runtime.text_ids(request.caption))
    compare("token_ids", reference("text_embedding", "text_embedding.inputs.0"))

    original_initial = runtime_module.initial_joint_state

    def capture_initial(*inputs, **kwargs):
        result = original_initial(*inputs, **kwargs)
        capture("initial_noise", result[0])
        return result

    runtime_module.initial_joint_state = capture_initial
    original_positions = runtime_module.position_ids
    position_calls = 0

    def capture_positions(*inputs, **kwargs):
        nonlocal position_calls
        result = original_positions(*inputs, **kwargs)
        if position_calls == 0:
            for stream, value in zip(("und", "video", "action"), result):
                capture("positions." + stream, value)
        position_calls += 1
        return result

    runtime_module.position_ids = capture_positions
    model = runtime.transformer
    original_forward = model.forward_hidden
    hidden_calls = 0

    def capture_hidden(streams, rope, action_rope, *inputs, **kwargs):
        nonlocal hidden_calls
        if hidden_calls == 0:
            for name, ropes in (("rope", rope), ("action_rope", action_rope)):
                for stream, pair in enumerate(ropes):
                    for component, value in enumerate(pair):
                        capture(f"positions.{name}.{stream}.{component}", value)
        hidden_calls += 1
        return original_forward(streams, rope, action_rope, *inputs, **kwargs)

    model.forward_hidden = capture_hidden
    seen = set()

    def hook(name):
        def save(module, inputs, result):
            if name in seen:
                return
            seen.add(name)
            if name.startswith("layer_"):
                for stream, before, after in zip(
                    ("causal_seq", "full_only_seq", "action_seq"), inputs[0], result
                ):
                    capture(f"{name}.inputs.{stream}", before)
                    capture(f"{name}.output.{stream}", after)
            else:
                capture(name + ".output", result)

        return save

    for index in (0, 35):
        model.language_model.model.layers[index].register_forward_hook(
            hook(f"layer_{index:02d}")
        )
    for name in ("time_embedder", "action_time_embedder", "vae2llm", "action2llm"):
        getattr(model, name).register_forward_hook(hook(name))
    original_decode = model.decode_outputs

    def capture_decode(*inputs, **kwargs):
        result = original_decode(*inputs, **kwargs)
        if "first_velocity" not in observed:
            capture("first_velocity", torch.cat([value.flatten() for value in result]))
        return result

    model.decode_outputs = capture_decode

    def sdpa_flags():
        return {
            name: getattr(torch.backends.cuda, name + "_sdp_enabled")()
            for name in ("cudnn", "flash", "mem_efficient", "math")
        }

    flags_before = sdpa_flags()
    attention_context = (
        sdpa_kernel(
            [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        )
        if args.enable_reference_sdpa
        else nullcontext()
    )
    with torch.inference_mode(), attention_context:
        clean = runtime.encode_observation(request)
        capture("clean_latents", clean)
        if args.use_reference_clean_latents:
            clean = reference("preparation", "preparation.output.1.x0_tokens_vision.0")[
                0
            ].to(runtime.device)
        video, actions = runtime.sample(request, clean)
        capture("final_video_latents", video)
        capture("final_actions", actions)
        if args.reference_video is not None:
            result = runtime._finalize(request, video, actions)
            capture("decoded_video", torch.from_numpy(result["video_frames"]))
    flags_after = sdpa_flags()
    if flags_before != flags_after:
        raise AssertionError("Diagnostic leaked SDPA backend state")
    compare(
        "clean_latents",
        reference("preparation", "preparation.output.1.x0_tokens_vision.0")[0],
    )
    compare("initial_noise", reference("preparation", "preparation.output.4.0"))
    compare("first_velocity", reference("first_velocity", "first_velocity.0"))
    compare("final_video_latents", reference("samples", "samples.vision.0")[0])
    compare("final_actions", reference("samples", "samples.action.0"))
    if args.reference_video is not None:
        compare(
            "decoded_video",
            torch.from_numpy(np.load(args.reference_video, allow_pickle=False)),
        )
    for name in observed:
        if name.startswith("positions."):
            compare(name, reference("positions", name))
        elif name.startswith("layer_") or name.endswith(".output"):
            compare(name, reference(name.split(".")[0], name))
    save_file(observed, args.output / "native.safetensors")
    report = {
        "reference_manifest": reference_manifest,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "elapsed_seconds": time.perf_counter() - started,
        "request_sha256": request_digest,
        "enable_reference_sdpa": args.enable_reference_sdpa,
        "use_reference_clean_latents": args.use_reference_clean_latents,
        "sdpa_flags_before": flags_before,
        "sdpa_flags_after": flags_after,
        "metrics": metrics,
    }
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "exact": sum(m.get("exact", False) for m in metrics.values()),
                "comparisons": len(metrics),
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
