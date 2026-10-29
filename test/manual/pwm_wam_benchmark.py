# SPDX-License-Identifier: Apache-2.0
"""Paired Cosmos/native WAM measurements with lossless, untimed output capture.

Private assets are explicit. ``--mode pair --print-plan`` prints an ABBA launch
plan without touching CUDA. Child processes save first requests, warmups and
measured samples separately. The common wall-time boundary starts before request
preparation and ends when physical actions and optional uint8 video are on CPU;
JSON/NPZ serialization is outside it. Python-worker IPC is included and labeled.

The ``leaf`` route uses the existing offline namespace/snapshot launcher. Its
optional NVIDIA FA3 adapter is diagnostic only, never a production backend.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(json_value(value), indent=2, allow_nan=False) + "\n"
    )


def git_state(root):
    def command(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments], text=True
        ).strip()

    return {
        "root": str(root),
        "head": command("rev-parse", "HEAD"),
        "tracked_status": command("status", "--porcelain", "--untracked-files=no"),
        "diff_sha256": hashlib.sha256(command("diff", "HEAD").encode()).hexdigest(),
    }


def source_files(root, prefixes):
    paths = set()
    for prefix in prefixes:
        path = root / prefix
        paths.update(path.rglob("*.py") if path.is_dir() else [path])
    return {str(path.relative_to(root)): digest(path) for path in sorted(paths)}


def overlay_modules(code_root):
    """Record optional external imports without hashing multi-GB extensions."""
    roots = [
        Path(item).resolve()
        for item in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if item
    ]
    result = {}
    for name, module in tuple(sys.modules.items()):
        value = getattr(module, "__file__", None)
        if not value:
            continue
        path = Path(value)
        if not any(path.is_relative_to(root) for root in roots) or path.is_relative_to(
            code_root
        ):
            continue
        resolved = path.resolve()
        if resolved.is_file():
            result[name] = {
                "path": str(path),
                "resolved_path": str(resolved),
                "bytes": resolved.stat().st_size,
            }
            if path.suffix == ".py":
                result[name]["sha256"] = digest(path)
    return result


def gpu_status(gpu):
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=uuid,name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,clocks.sm",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return {"time_unix": time.time(), "gpu_csv": result, "processes_csv": processes}


def freeze_requests(paths, output):
    inputs = output / "inputs"
    inputs.mkdir()
    result = []
    for index, path in enumerate(paths):
        raw = json.loads(path.read_text())
        if type(raw.get("seed")) is not int:
            raise ValueError(
                "Every benchmark request requires an explicit integer seed"
            )
        # One execution setting is supplied by the implementation options, never
        # allowed to hide inside a request and override the selected comparator.
        if "enable_und_cache" in raw:
            raise ValueError("Put enable_und_cache in benchmark options, not requests")
        for key in ("vision_path", "prompt_path"):
            if raw.get(key) is None:
                continue
            value = str(raw[key])
            if "://" in value:
                raise ValueError(
                    "Freeze remote benchmark inputs locally before running"
                )
            asset = Path(value).expanduser()
            if not asset.is_absolute():
                asset = path.parent / asset
            frozen = inputs / f"{index:03d}_{key}{asset.suffix}"
            shutil.copyfile(asset, frozen)
            raw[key] = str(frozen.resolve())
        request = inputs / f"{index:03d}.json"
        write_json(request, raw)
        result.append(request)
    return result


def native_fa3_adapter():
    """Adapt only the external kernel provider, with identical explicit inputs."""
    from flash_attn_3_nv.flash_attn_interface import flash_attn_varlen_func

    from sglang.multimodal_gen.runtime.models.dits.cosmos_pwm_wam import execution

    def run(q, k, v, cu_q, cu_k, causal):
        value = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=q.shape[0],
            max_seqlen_k=k.shape[0],
            softmax_scale=q.shape[-1] ** -0.5,
            causal=causal,
            num_splits=1,
        )
        return value[0] if isinstance(value, tuple) else value

    execution._run_flash3 = run
    return {
        "kind": "diagnostic NVIDIA FA3 dependency adapter; not production SGLang kernel",
        "module_path": __import__(
            flash_attn_varlen_func.__module__, fromlist=["x"]
        ).__file__,
        "num_splits": 1,
        "adapter_file_sha256": digest(__file__),
    }


def cosmos_transformers5_compat():
    """Restore an unrelated Edge-processor re-export, not WAM computation."""
    from transformers.image_utils import get_image_size
    from transformers.models.qwen3_vl import video_processing_qwen3_vl

    if hasattr(video_processing_qwen3_vl, "get_image_size"):
        raise ValueError("Compatibility alias requested but the export already exists")
    video_processing_qwen3_vl.get_image_size = get_image_size
    return {
        "purpose": "TF5 removed a helper re-export imported by the unused Cosmos Edge processor",
        "alias": "transformers.models.qwen3_vl.video_processing_qwen3_vl.get_image_size",
        "target": "transformers.image_utils.get_image_size",
        "target_file": inspect.getfile(get_image_size),
        "target_function_sha256": hashlib.sha256(
            inspect.getsource(get_image_size).encode()
        ).hexdigest(),
        "adapter_file_sha256": digest(__file__),
    }


class CosmosRunner:
    def __init__(self, args, options):
        import torch

        os.environ["COSMOS_TRAINING"] = "0"
        sys.path.insert(0, str(args.cosmos_repo))
        os.chdir(args.cosmos_repo)
        from cosmos_framework.inference.args import OmniSetupOverrides
        from cosmos_framework.inference.inference import OmniInference
        from omegaconf import OmegaConf

        allowed = {
            "attention_backend",
            "enable_torch_compile",
            "enable_und_cache",
            "reuse_metadata",
            "projected_und_cache_path",
        }
        if set(options) - allowed:
            raise ValueError(f"Unknown Cosmos options: {set(options) - allowed}")
        required = {
            "attention_backend",
            "enable_torch_compile",
            "enable_und_cache",
            "reuse_metadata",
        }
        if not required <= set(options):
            raise ValueError(f"Cosmos options require {sorted(required)}")
        self.options = options
        self.scratch = tempfile.TemporaryDirectory(prefix="sglang-cosmos-benchmark-")
        for key in tuple(os.environ):
            if key.startswith("TORCHELASTIC_") or key in (
                "MASTER_ADDR",
                "MASTER_PORT",
                "GROUP_RANK",
                "ROLE_RANK",
            ):
                os.environ.pop(key, None)
        os.environ.update(
            RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", LOCAL_WORLD_SIZE="1"
        )
        store = torch.distributed.TCPStore("127.0.0.1", 0, 1, True)
        torch.distributed.init_process_group("nccl", store=store, rank=0, world_size=1)
        overrides = [
            "model.config.ema.enabled=false",
            "model.config.action_tower_init=checkpoint",
            "model.config.fixed_sequence_length.enabled=false",
            "model.config.action_tower_und_short_tokens=0",
            "model.config.precision=bfloat16",
            f"model.config.action_tower_attention_backend={options['attention_backend']}",
            f"model.config.vlm_config.tokenizer.pretrained_model_name={json.dumps(str(args.tokenizer))}",
        ]
        if args.cosmos_vae:
            overrides.append(
                f"model.config.tokenizer.vae_path={json.dumps(str(args.cosmos_vae))}"
            )
        setup = OmniSetupOverrides(
            checkpoint_path=str(args.checkpoint),
            config_file=str(args.checkpoint / "config.json"),
            output_dir=Path(self.scratch.name),
            experiment_overrides=overrides,
            use_ema_weights=False,
            use_torch_compile=options["enable_torch_compile"],
            compiled_region="language",
            compile_dynamic=False,
            use_cuda_graphs=False,
            guardrails=False,
            dp_shard_size=1,
            dp_replicate_size=1,
            cp_size=1,
            cfgp_size=1,
        ).build_setup(world_size=1)
        self.model = OmniInference.create(setup).model.eval()
        if options.get("projected_und_cache_path"):
            if not options["enable_und_cache"]:
                raise ValueError("projected UND adapter requires enable_und_cache")
            path = Path(options["projected_und_cache_path"])
            spec = importlib.util.spec_from_file_location(
                "pwm_reference_und_adapter", path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.install_projected_und_cache(self.model)
        self.spec = OmegaConf.to_container(
            self.model.config.action128_spec, resolve=True
        )
        self.model.net.action_normalization.validate_contract()
        self.model.net.action_normalization._reset_transforms()

    def __call__(self, raw):
        import torch
        from cosmos_framework.inference.action_serving import prepare_wam_request
        from cosmos_framework.inference.args import OmniSampleOverrides
        from cosmos_framework.utils import misc

        with (
            tempfile.TemporaryDirectory(dir=self.scratch.name) as directory,
            torch.inference_mode(),
        ):
            prepared, batch = prepare_wam_request(
                OmniSampleOverrides.model_validate(raw),
                self.spec,
                self.model.config,
                Path(directory),
                self.model.input_video_key,
            )
            batch = misc.to(batch, device="cuda")
            with torch._dynamo.config.patch(
                recompile_limit=512, accumulated_recompile_limit=4096
            ):
                samples = self.model.generate_samples_from_batch(
                    batch,
                    seed=[prepared.seed],
                    num_steps=prepared.num_steps,
                    guidance=prepared.guidance,
                    shift=prepared.shift,
                    guidance_interval=prepared.sample_args["guidance_interval"],
                    normalize_cfg=prepared.sample_args["normalize_cfg"],
                    sigma_max=prepared.sample_args["sigma_max"],
                    has_negative_prompt="neg_ai_caption" in batch,
                    reuse_small_action_metadata=self.options["reuse_metadata"],
                    reuse_small_action_und=self.options["enable_und_cache"],
                )
            normalized = samples["action"][0].float()
            domain = self.spec["registry"]["domains"].index(prepared.action_domain)
            physical = self.model.net.action_normalization.denormalize(
                normalized, domain
            )
            active = torch.tensor(
                prepared.action_dim_mask, device=normalized.device, dtype=torch.bool
            )
            video = None
            if prepared.return_video:
                decoded = ((self.model.decode(samples["vision"][0]) + 1) / 2).clamp(
                    0, 1
                )
                video = (
                    (decoded[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()
                )
            return {
                "actions": physical.masked_fill(~active, 0).cpu().numpy(),
                "normalized_actions": normalized.masked_fill(~active, 0).cpu().numpy(),
                "video_frames": video,
                "video_fps": prepared.sample_args["fps"],
                "action_domain": prepared.action_domain,
                "action_dim_mask": prepared.action_dim_mask,
                "action_fps": prepared.action_fps,
                "action_timestamps_seconds": batch["sequence_plan"][
                    0
                ].action_timestamps_seconds,
                "action_units": "raw_physical_or_source_units",
                "seed": prepared.seed,
                "conditioning_caption_sha256": hashlib.sha256(
                    batch["ai_caption"][0].encode()
                ).hexdigest(),
            }

    def close(self):
        import torch

        torch.distributed.destroy_process_group()
        self.scratch.cleanup()


def make_native(args, options):
    allowed = {"attention_backend", "enable_torch_compile", "enable_und_cache"}
    if set(options) != allowed:
        raise ValueError(f"Native options must contain exactly {sorted(allowed)}")
    if args.native_route == "python":
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            DiffGenerator,
        )

        generator = DiffGenerator.from_pretrained(
            model_path=str(args.checkpoint),
            backend="sglang",
            pipeline_class_name="CosmosPWMWAMPipeline",
            pipeline_config={
                "vae_path": str(args.vae),
                "tokenizer_path": str(args.tokenizer),
                "enable_und_cache": options["enable_und_cache"],
            },
            attention_backend=options["attention_backend"],
            enable_torch_compile=options["enable_torch_compile"],
            num_gpus=1,
            tp_size=1,
            sp_degree=1,
            batching_max_size=1,
            enable_cfg_parallel=False,
            enable_breakable_cuda_graph=False,
            dit_cpu_offload=False,
            text_encoder_cpu_offload=False,
            image_encoder_cpu_offload=False,
            vae_cpu_offload=False,
            dit_layerwise_offload=False,
            use_fsdp_inference=False,
            warmup_mode="off",
        )
        return lambda raw: generator.generate_action(
            {"wam_request": raw}
        ), generator.shutdown
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_pwm_wam.runtime import (
        PwmWamRuntime,
    )

    runtime = PwmWamRuntime(
        args.checkpoint,
        vae_path=str(args.vae),
        tokenizer_path=str(args.tokenizer),
        **options,
    )
    return runtime.forward, lambda: None


def run_child(args):
    import numpy as np
    import torch
    from test_cosmos_pwm_wam import validate_result

    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output / "harness.py")
    shutil.copyfile(args.options, args.output / "options.json")
    options = json.loads(args.options.read_text())
    for key in ("enable_torch_compile", "enable_und_cache", "reuse_metadata"):
        if key in options and type(options[key]) is not bool:
            raise ValueError(f"{key} must be a JSON boolean")
    adapter_path = options.get("projected_und_cache_path")
    if adapter_path:
        adapter_path = Path(adapter_path).expanduser()
        if not adapter_path.is_absolute():
            adapter_path = args.options.parent / adapter_path
        options["projected_und_cache_path"] = str(adapter_path.resolve())
    root = Path(__file__).resolve().parents[2]
    code_root = args.cosmos_repo if args.implementation == "cosmos" else root
    prefixes = (
        (
            "cosmos_framework/model",
            "cosmos_framework/data",
            "cosmos_framework/inference",
            "cosmos_framework/configs",
            "cosmos_framework/utils",
        )
        if args.implementation == "cosmos"
        else (
            "python/sglang/multimodal_gen/runtime/models/dits/cosmos_pwm_wam",
            "python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos_pwm_wam",
        )
    )
    git_root = code_root
    if args.implementation == "native" and args.native_route == "leaf":
        code_root = Path(sys.modules["sglang"].__path__[0])
        prefixes = tuple(prefix.removeprefix("python/sglang/") for prefix in prefixes)
        prefixes += ("multimodal_gen/runtime/models/schedulers/base.py",)
    hashes = source_files(code_root, prefixes)
    manifest = {
        "implementation": args.implementation,
        "native_route": args.native_route,
        "command": shlex.join([sys.executable, *sys.argv]),
        "python": sys.version,
        "options": options,
        "options_sha256": digest(args.options),
        "code": git_state(git_root),
        "executed_leaf_root": str(code_root),
        "leaf_sha256": hashes,
        "checkpoint_config_sha256": digest(args.checkpoint / "config.json"),
        "harness_sha256": digest(__file__),
        "status": "started",
        "samples": [],
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "PYTHONPATH",
                "OMP_NUM_THREADS",
                "COSMOS_FA3_PROVIDER",
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
            )
        },
        "versions": {},
        "timing_scope": "prepare through CPU output arrays, excludes NPZ/JSON capture; Python route includes worker IPC",
        "cosmos_generate_dynamo_config": {
            "recompile_limit": 512,
            "accumulated_recompile_limit": 4096,
        }
        if args.implementation == "cosmos"
        else None,
        "cache_policy": "Per-implementation disk caches shared across ABBA blocks; first is per-process/per-case, not a guaranteed cold disk compile.",
    }
    for name in (
        "torch",
        "diffusers",
        "transformers",
        "triton",
        "safetensors",
        "msgspec",
        "flash-attn-3-nv",
        "sgl-kernel",
    ):
        try:
            manifest["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            manifest["versions"][name] = None
    if adapter_path:
        manifest["projected_und_adapter_sha256"] = digest(
            options["projected_und_cache_path"]
        )
    write_json(args.output / "manifest.json", manifest)
    close = lambda: None
    try:
        if args.cosmos_transformers5_compat:
            if args.implementation != "cosmos":
                raise ValueError("Cosmos TF5 compatibility only applies to Cosmos")
            manifest["diagnostic_transformers_adapter"] = cosmos_transformers5_compat()
        if args.nv_fa3_adapter:
            if args.implementation != "native" or args.native_route != "leaf":
                raise ValueError(
                    "NVIDIA adapter is allowed only in the offline native leaf route"
                )
            manifest["diagnostic_fa3_adapter"] = native_fa3_adapter()
        started = time.perf_counter()
        if args.implementation == "cosmos":
            runner = CosmosRunner(args, options)
            run, close = runner, runner.close
        else:
            run, close = make_native(args, options)
        worker_route = args.implementation == "native" and args.native_route == "python"
        if not worker_route:
            torch.cuda.synchronize()
        manifest["load_seconds"] = time.perf_counter() - started
        manifest["torch_cuda"] = torch.version.cuda
        manifest["cudnn"] = torch.backends.cudnn.version()
        manifest["numerics_process_scope"] = (
            "parent only; worker policy comes from results"
            if worker_route
            else "model process"
        )
        manifest["numerics"] = {
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "bf16_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        }
        for case, request in enumerate(args.request):
            raw = json.loads(request.read_text())
            phases = ["first"] + ["warmup"] * args.warmup + ["measured"] * args.repeat
            for number, phase in enumerate(phases):
                if not worker_route:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                before = gpu_status(args.gpu)
                started = time.perf_counter()
                result = run(copy.deepcopy(raw))
                if not worker_route:
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                after = gpu_status(args.gpu)
                validate_result(result)
                folder = args.output / f"case_{case:03d}_{number:03d}_{phase}"
                folder.mkdir()
                arrays = {
                    key: value
                    for key, value in result.items()
                    if isinstance(value, np.ndarray)
                }
                np.savez(folder / "arrays.npz", **arrays)
                write_json(
                    folder / "result.json",
                    {key: value for key, value in result.items() if key not in arrays},
                )
                row = {
                    "case": case,
                    "phase": phase,
                    "sample": number,
                    "directory": str(folder),
                    "request": str(request),
                    "request_sha256": digest(request),
                    "elapsed_seconds": elapsed,
                    "gpu_before": before,
                    "gpu_after": after,
                    "timing_seconds": result.get("timing_seconds"),
                    "parameters": result.get("parameters"),
                    "peak_allocated_bytes": None
                    if worker_route
                    else torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": None
                    if worker_route
                    else torch.cuda.max_memory_reserved(),
                }
                manifest["samples"].append(row)
                write_json(args.output / "manifest.json", manifest)
                print(
                    json.dumps(
                        {
                            key: row[key]
                            for key in ("case", "phase", "sample", "elapsed_seconds")
                        }
                    ),
                    flush=True,
                )
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["dynamo_counters_scope"] = (
            "parent only; not worker compilation evidence"
            if args.native_route == "python" and args.implementation == "native"
            else "model process"
        )
        manifest["dynamo_counters"] = json_value(dict(torch._dynamo.utils.counters))
        manifest["compile_config"] = {
            name: json_value(getattr(torch._dynamo.config, name, None))
            for name in (
                "suppress_errors",
                "recompile_limit",
                "accumulated_recompile_limit",
                "dynamic_shapes",
            )
        }
        manifest["inductor_config"] = {
            name: json_value(getattr(torch._inductor.config, name, None))
            for name in ("max_autotune", "coordinate_descent_tuning", "compile_threads")
        }
        manifest["overlay_modules"] = overlay_modules(code_root)
        manifest["leaf_files_unchanged"] = source_files(code_root, prefixes) == hashes
        write_json(args.output / "manifest.json", manifest)
        close()
    if not manifest["leaf_files_unchanged"]:
        raise RuntimeError("Source files changed during benchmark; discard this block")


def summarize(output):
    import numpy as np
    from test_cosmos_pwm_wam import array_metrics, compare_reference

    blocks = []
    baseline = {}
    reference_by_impl = {}
    request_by_case = {}
    timings = {}
    identity_by_impl = {}
    comparisons = []
    for folder in sorted(path for path in output.glob("block_*") if path.is_dir()):
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["status"] != "complete" or not manifest["leaf_files_unchanged"]:
            raise ValueError(f"Incomplete or changed-source benchmark: {folder}")
        impl = manifest["implementation"]
        identity = {
            key: manifest.get(key)
            for key in (
                "options",
                "leaf_sha256",
                "checkpoint_config_sha256",
                "versions",
                "python",
                "torch_cuda",
                "cudnn",
                "native_route",
            )
        }
        previous = identity_by_impl.setdefault(impl, identity)
        if identity != previous:
            raise ValueError(
                f"Implementation or environment changed across blocks: {impl}"
            )
        blocks.append(
            {
                "directory": str(folder),
                "implementation": impl,
                "load_seconds": manifest["load_seconds"],
                "versions": manifest["versions"],
                "python": manifest["python"],
                "torch_cuda": manifest["torch_cuda"],
                "cudnn": manifest["cudnn"],
                "numerics": manifest["numerics"],
                "options": manifest["options"],
            }
        )
        for sample in manifest["samples"]:
            case = sample["case"]
            request_hash = sample.get("request_sha256")
            if request_by_case.setdefault(case, request_hash) != request_hash:
                raise ValueError(f"Request changed across blocks: case {case}")
            path = Path(sample["directory"])
            arrays = dict(np.load(path / "arrays.npz", allow_pickle=False))
            result = {**json.loads((path / "result.json").read_text()), **arrays}
            if impl == "cosmos":
                baseline.setdefault(case, path)
            reference_by_impl.setdefault((impl, case), path)
            if sample["phase"] == "measured":
                timings.setdefault((impl, case), []).append(sample["elapsed_seconds"])
            row = {
                "block": folder.name,
                "case": case,
                "phase": sample["phase"],
                "sample": sample["sample"],
            }
            for key, reference in (
                ("same_implementation", reference_by_impl[(impl, case)]),
                ("versus_cosmos", baseline[case]),
            ):
                metrics = compare_reference(result, reference)
                expected = dict(np.load(reference / "arrays.npz", allow_pickle=False))
                metrics["full_width"] = {
                    field: array_metrics(arrays[field], expected[field])
                    for field in ("actions", "normalized_actions")
                }
                if ("video_frames" in arrays) != ("video_frames" in expected):
                    raise ValueError("Video delivery differs between implementations")
                row[key] = metrics
            comparisons.append(row)
    environment_keys = ("python", "versions", "torch_cuda", "cudnn")
    same_env = bool(blocks) and all(
        all(block[key] == blocks[0][key] for key in environment_keys)
        for block in blocks
    )
    rows = []
    for case in sorted({case for _, case in timings}):
        row = {"case": case}
        for impl in ("cosmos", "native"):
            values = timings[(impl, case)]
            row[impl] = {
                "samples_seconds": values,
                "count": len(values),
                "median_seconds": statistics.median(values),
                "min_seconds": min(values),
                "max_seconds": max(values),
                "mean_seconds": statistics.mean(values),
            }
        row["cosmos_over_native_median_ratio"] = (
            row["cosmos"]["median_seconds"] / row["native"]["median_seconds"]
        )
        rows.append(row)
    report = {
        "same_recorded_software_environment": same_env,
        "blocks": blocks,
        "latency": rows,
        "output_comparisons": comparisons,
        "accuracy_policy": "Report measured errors; no task-specific action tolerance is invented. All saved samples, including warmup, are compared.",
    }
    write_json(output / "comparison.json", report)
    return report


def pair(args):
    script = Path(__file__).resolve()
    launch = script.with_name("pwm_wam_isolated_native.py")
    commands = []
    for block, impl in enumerate(
        ("cosmos", "native", "native", "cosmos") * args.rounds
    ):
        destination = args.output / f"block_{block:02d}_{impl}"
        python = args.cosmos_python if impl == "cosmos" else args.native_python
        command = [
            str(python),
            str(script),
            "--mode",
            "run",
            "--implementation",
            impl,
            "--output",
            str(destination),
            "--options",
            str(args.output / f"{impl}_options.json"),
        ]
        if impl == "native" and args.native_route == "leaf":
            command = [
                str(python),
                str(launch),
                "--snapshot",
                str(args.output / "native_snapshot"),
                "--script",
                str(script),
                "--",
                *command[2:],
            ]
        for name in (
            "checkpoint",
            "vae",
            "tokenizer",
            "cosmos_repo",
            "native_route",
            "gpu",
            "warmup",
            "repeat",
        ):
            command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        if args.cosmos_vae:
            command += ["--cosmos-vae", str(args.cosmos_vae)]
        if impl == "cosmos" and args.cosmos_transformers5_compat:
            command.append("--cosmos-transformers5-compat")
        if impl == "native" and args.nv_fa3_adapter:
            command.append("--nv-fa3-adapter")
        for index in range(len(args.request)):
            command += ["--request", str(args.output / "inputs" / f"{index:03d}.json")]
        env = {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "LD_LIBRARY_PATH": "",
            "OMP_NUM_THREADS": str(args.threads),
            "PYTHONDONTWRITEBYTECODE": "1",
            "COSMOS_TRAINING": "0",
            "TORCHINDUCTOR_CACHE_DIR": str(args.output / "compile_cache" / impl),
            "TRITON_CACHE_DIR": str(args.output / "triton_cache" / impl),
            "PYTHONPATH": args.cosmos_pythonpath
            if impl == "cosmos"
            else args.native_pythonpath,
        }
        commands.append(
            {"implementation": impl, "command": command, "environment": env}
        )
    if args.print_plan:
        print(json.dumps(commands, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    freeze_requests(args.request, args.output)
    for impl, path in (
        ("cosmos", args.cosmos_options),
        ("native", args.native_options),
    ):
        value = json.loads(path.read_text())
        if value.get("projected_und_cache_path"):
            adapter = Path(value["projected_und_cache_path"]).expanduser()
            value["projected_und_cache_path"] = str(
                (adapter if adapter.is_absolute() else path.parent / adapter).resolve()
            )
        write_json(args.output / f"{impl}_options.json", value)
    write_json(args.output / "commands.json", commands)
    for index, item in enumerate(commands):
        status = gpu_status(args.gpu)
        if status["processes_csv"]:
            raise RuntimeError(
                "Assigned GPU has a compute process; benchmark did not launch"
            )
        write_json(args.output / f"gpu_before_block_{index:02d}.json", status)
        with (args.output / f"block_{index:02d}.log").open("w") as log:
            subprocess.run(
                item["command"],
                env={**os.environ, **item["environment"]},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        summarize(args.output) if index >= 1 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pair", "run", "summarize"), default="pair")
    parser.add_argument("--implementation", choices=("cosmos", "native"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vae", type=Path)
    parser.add_argument("--cosmos-vae", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--cosmos-repo", type=Path)
    parser.add_argument("--request", type=Path, action="append", default=[])
    parser.add_argument("--options", type=Path)
    parser.add_argument("--cosmos-options", type=Path)
    parser.add_argument("--native-options", type=Path)
    parser.add_argument("--cosmos-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--native-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cosmos-pythonpath", default="")
    parser.add_argument("--native-pythonpath", default="")
    parser.add_argument(
        "--native-route", choices=("leaf", "runtime", "python"), default="leaf"
    )
    parser.add_argument("--nv-fa3-adapter", action="store_true")
    parser.add_argument("--cosmos-transformers5-compat", action="store_true")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    args.request = [path.expanduser().resolve() for path in args.request]
    if args.mode == "summarize":
        summarize(args.output)
        return
    for name in ("checkpoint", "vae", "tokenizer", "cosmos_repo"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if not args.request or args.repeat < 1 or args.warmup < 0 or args.rounds < 1:
        parser.error("Provide requests, positive repeat/rounds and nonnegative warmup")
    if args.mode == "run":
        if args.implementation is None or args.options is None:
            parser.error("run requires --implementation and --options")
        run_child(args)
    else:
        if args.cosmos_options is None or args.native_options is None:
            parser.error("pair requires --cosmos-options and --native-options")
        pair(args)


if __name__ == "__main__":
    main()
