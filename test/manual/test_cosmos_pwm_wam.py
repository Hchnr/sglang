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
import sys
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
    parser.add_argument(
        "--und-cache",
        choices=("default", "off", "on", "compare"),
        default="default",
        help="Pin request-local UND reuse, or compare off/on/on/off for each request",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Untimed rounds of the first request for each selected cache mode",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="After timing, capture the first request for each cache mode (runtime only)",
    )
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        parser.error("--repeat must be positive and --warmup nonnegative")
    if args.profile and args.route != "runtime":
        parser.error(
            "--profile requires --route runtime; the Python worker is separate"
        )
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
        "argv": sys.argv,
        "und_cache": args.und_cache,
        "warmup_rounds": args.warmup,
        "load_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "diffusers", "transformers", "safetensors")
        },
        "requests": [],
    }
    cache_modes = {
        "default": (None,),
        "off": (False,),
        "on": (True,),
        "compare": (False, True, True, False),
    }[args.und_cache]

    def request_variant(path, cache):
        raw = load_request(path)
        if args.und_cache == "compare" and raw.get("seed") is None:
            raise ValueError("UND cache comparisons require an explicit fixed seed")
        if cache is not None:
            raw["enable_und_cache"] = cache
        return raw

    def comparison_key(raw):
        key_fields = dict(raw)
        if args.und_cache == "compare":
            key_fields.pop("enable_und_cache", None)
        return json.dumps(key_fields, sort_keys=True)

    previous = {}
    try:
        for _ in range(args.warmup):
            for cache in dict.fromkeys(cache_modes):
                validate_result(run(request_variant(args.request[0], cache)))
        entries = [
            (path, cache)
            for path in args.request * args.repeat
            for cache in cache_modes
        ]
        for index, (request_path, cache) in enumerate(entries):
            raw = request_variant(request_path, cache)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = run(raw)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            validate_result(result)
            if (
                cache is not None
                and result.get("parameters", {}).get("enable_und_cache") is not cache
            ):
                raise AssertionError("Requested UND cache mode was not executed")
            directory = args.output / f"{index:03d}"
            save_result(result, directory)
            row = {
                "request": str(request_path),
                "enable_und_cache": result.get("parameters", {}).get(
                    "enable_und_cache"
                ),
                "elapsed_seconds": elapsed,
                "parent_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "parent_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "timing_seconds": result.get("timing_seconds"),
            }
            # GPU memory for the Python route lives in the worker, not the parent.
            if index == 0 and args.reference:
                row["reference"] = compare_reference(result, args.reference)
            key = comparison_key(raw)
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
        if args.profile:
            from torch.profiler import ProfilerActivity, profile

            summary["profiles"] = []
            for cache in dict.fromkeys(cache_modes):
                raw = request_variant(args.request[0], cache)
                label = "default" if cache is None else ("on" if cache else "off")
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                ) as prof:
                    result = run(raw)
                    torch.cuda.synchronize()
                validate_result(result)
                baseline = previous[comparison_key(raw)]
                exact = {
                    field: array_metrics(result[field], baseline[field])
                    for field in ("actions", "normalized_actions", "video_frames")
                    if result.get(field) is not None
                }
                if not all(value["exact"] for value in exact.values()):
                    raise AssertionError("Profiling changed request output")
                save_result(result, args.output / f"profile_{label}")
                trace = args.output / f"profile_{label}.json"
                prof.export_chrome_trace(str(trace))
                (args.output / f"profile_{label}_operators.txt").write_text(
                    prof.key_averages().table(
                        sort_by="self_device_time_total", row_limit=40
                    )
                )
                summary["profiles"].append({"path": str(trace), "parity": exact})
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    finally:
        if generator is not None:
            generator.shutdown()


if __name__ == "__main__":
    main()
