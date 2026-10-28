# SPDX-License-Identifier: Apache-2.0
"""Explicit-asset GPU validation for native PWM WAM (outside public CI).

Run with --help. Each request is saved losslessly, with timings and environment
metadata. Repeating A/B/A request paths checks worker state isolation. Reference
comparison reports errors without choosing an application-specific tolerance.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np


def array_metrics(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError("Non-finite validation output")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    norm = np.linalg.norm(expected.astype(np.float64))
    return {
        "shape": list(actual.shape),
        "exact": bool(np.array_equal(actual, expected)),
        "max_abs": float(np.max(np.abs(delta))),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "relative_l2": float(np.linalg.norm(delta) / max(norm, 1e-12)),
    }


def compare_reference(result, reference):
    """Accept a saved native result directory or Cosmos validation worker output."""
    reference = Path(reference)
    if (reference / "result.json").is_file():
        expected = json.loads((reference / "result.json").read_text())
        arrays = np.load(reference / "arrays.npz", allow_pickle=False)
        expected.update({key: arrays[key] for key in arrays.files})
    else:
        lines = [
            json.loads(line)
            for line in (reference / "response.jsonl").read_text().splitlines()
        ]
        responses = [line for line in lines if "result" in line]
        if len(responses) != 1:
            raise ValueError("Reference must contain exactly one completed request")
        expected = dict(responses[0]["result"])
        expected["actions"] = expected.pop("action")
        expected["normalized_actions"] = expected.pop("normalized_action")
        if (reference / "video.npy").is_file():
            expected["video_frames"] = np.load(
                reference / "video.npy", allow_pickle=False
            )
    discrete = (
        "action_domain",
        "action_dim_mask",
        "action_fps",
        "action_units",
        "seed",
        "action_timestamps_seconds",
        "conditioning_caption_sha256",
    )
    for field in discrete:
        if result[field] != expected[field]:
            raise AssertionError(f"Reference contract mismatch: {field}")
    active = np.asarray(result["action_dim_mask"], dtype=bool)
    metrics = {
        key: array_metrics(
            np.asarray(result[key])[:, active], np.asarray(expected[key])[:, active]
        )
        for key in ("actions", "normalized_actions")
    }
    if (
        result.get("video_frames") is not None
        and expected.get("video_frames") is not None
    ):
        metrics["video"] = array_metrics(
            result["video_frames"], expected["video_frames"]
        )
    return metrics


def validate_result(result):
    action = np.asarray(result["actions"])
    normalized = np.asarray(result["normalized_actions"])
    mask = np.asarray(result["action_dim_mask"], dtype=bool)
    assert action.ndim == 2 and action.shape[1] == 128
    assert normalized.shape == action.shape and mask.shape == (128,)
    assert np.isfinite(action).all() and np.isfinite(normalized).all()
    assert np.count_nonzero(action[:, ~mask]) == 0
    assert np.count_nonzero(normalized[:, ~mask]) == 0
    np.testing.assert_allclose(
        result["action_timestamps_seconds"],
        np.arange(len(action), dtype=np.float64) / result["action_fps"],
        rtol=0,
        atol=0,
    )
    frames = result.get("video_frames")
    if frames is not None:
        assert frames.dtype == np.uint8 and frames.ndim == 4 and frames.shape[-1] == 3


def save_result(result, directory):
    from sglang.multimodal_gen.runtime.entrypoints.action.cosmos_pwm_wam import (
        numpy_to_lists,
    )

    directory.mkdir(parents=True, exist_ok=False)
    arrays = {
        key: value for key, value in result.items() if isinstance(value, np.ndarray)
    }
    np.savez(directory / "arrays.npz", **arrays)
    metadata = {key: value for key, value in result.items() if key not in arrays}
    (directory / "result.json").write_text(
        json.dumps(numpy_to_lists(metadata), indent=2)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--request", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=("runtime", "python"), default="python")
    parser.add_argument(
        "--reference",
        type=Path,
        help="Compare the first request against saved reference",
    )
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    args.output.mkdir(parents=True, exist_ok=False)

    import torch

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_pwm_wam.request import (
        load_request,
    )

    torch.cuda.set_device(0)
    started = time.perf_counter()
    generator = None
    if args.route == "runtime":
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_pwm_wam.runtime import (
            PwmWamRuntime,
        )

        runtime = PwmWamRuntime(
            args.checkpoint, vae_path=str(args.vae), tokenizer_path=str(args.tokenizer)
        )
        run = runtime.forward
    else:
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            DiffGenerator,
        )

        generator = DiffGenerator.from_pretrained(
            model_path=str(args.checkpoint),
            pipeline_class_name="CosmosPWMWAMPipeline",
            pipeline_config={
                "vae_path": str(args.vae),
                "tokenizer_path": str(args.tokenizer),
            },
            backend="sglang",
            attention_backend="torch_sdpa",
            num_gpus=1,
            tp_size=1,
            sp_degree=1,
            batching_max_size=1,
            enable_cfg_parallel=False,
            enable_torch_compile=False,
            enable_breakable_cuda_graph=False,
            dit_cpu_offload=False,
            text_encoder_cpu_offload=False,
            image_encoder_cpu_offload=False,
            vae_cpu_offload=False,
            dit_layerwise_offload=False,
            use_fsdp_inference=False,
            warmup_mode="off",
        )
        run = lambda request: generator.generate_action(
            sampling_params_kwargs={"wam_request": request}
        )
    torch.cuda.synchronize()
    summary = {
        "route": args.route,
        "load_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "diffusers", "transformers", "safetensors")
        },
        "requests": [],
    }
    previous = {}
    try:
        for index, request_path in enumerate(args.request * args.repeat):
            raw = load_request(request_path)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = run(raw)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            validate_result(result)
            directory = args.output / f"{index:03d}"
            save_result(result, directory)
            row = {
                "request": str(request_path),
                "elapsed_seconds": elapsed,
                "parent_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "parent_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "timing_seconds": result.get("timing_seconds"),
            }
            # GPU memory for the Python route lives in the worker, not the parent.
            if index == 0 and args.reference:
                row["reference"] = compare_reference(result, args.reference)
            key = json.dumps(raw, sort_keys=True)
            if key in previous:
                baseline = previous[key]
                row["repeat"] = {
                    field: array_metrics(result[field], baseline[field])
                    for field in ("actions", "normalized_actions", "video_frames")
                    if result.get(field) is not None
                }
                if not all(item["exact"] for item in row["repeat"].values()):
                    raise AssertionError("Fixed-request repeat changed across requests")
            else:
                previous[key] = result
            summary["requests"].append(row)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(row), flush=True)
    finally:
        if generator is not None:
            generator.shutdown()


if __name__ == "__main__":
    main()
