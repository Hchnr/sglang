# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Capture current Cosmos WAM intermediates in its own Python environment.

This diagnostic imports the explicitly selected source repository, never SGLang's
production runtime. Tensor files use safetensors; JSON describes nested values.
Hooks observe the original operations without replacing their implementations.

Example (run with the Cosmos environment's Python):
    CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH='' OMP_NUM_THREADS=8 \
        python test/manual/pwm_wam_capture_reference.py \
        --source-repo ../cosmos-framework-pwm \
        --checkpoint ../vllm-omni-pwm/artifacts/pwm-wam-hf \
        --request artifacts/pwm_wam_1028/request_smoke.json \
        --output artifacts/pwm_wam_1028/intermediate_reference

Capture layout follows vllm-omni-pwm's capture_native_fixture.py, extended with
request preparation, explicit stream positions, and post-mask first velocity.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--attention-backend", choices=("sdpa", "flash3"), default="sdpa"
    )
    args = parser.parse_args()
    source, checkpoint, request_path, output = (
        value.resolve()
        for value in (args.source_repo, args.checkpoint, args.request, args.output)
    )
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be empty to prevent mixing fixture runs")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))
    os.chdir(source)
    os.environ["COSMOS_TRAINING"] = "0"

    import diffusers
    import torch
    import transformers
    from cosmos_framework.data.generator.sequence_packing.runtime import from_all_seq
    from cosmos_framework.inference.action_serving import prepare_wam_request
    from cosmos_framework.inference.args import OmniSampleOverrides, OmniSetupOverrides
    from cosmos_framework.inference.inference import OmniInference
    from cosmos_framework.model.generator.mot import small_action_tower
    from cosmos_framework.utils import misc
    from omegaconf import OmegaConf
    from safetensors.torch import save_file

    def save_tree(name, value):
        tensors = {}

        def encode(item, key):
            if isinstance(item, torch.Tensor):
                tensors[key] = item.detach().cpu().contiguous().clone()
                return {
                    "tensor": key,
                    "shape": list(item.shape),
                    "stride": list(item.stride()),
                    "dtype": str(item.dtype),
                }
            if dataclasses.is_dataclass(item):
                return {
                    field.name: encode(
                        getattr(item, field.name), key + "." + field.name
                    )
                    for field in dataclasses.fields(item)
                }
            if isinstance(item, dict):
                return {str(k): encode(v, key + "." + str(k)) for k, v in item.items()}
            if isinstance(item, (tuple, list)):
                return [encode(v, key + "." + str(i)) for i, v in enumerate(item)]
            if item is None or isinstance(item, (str, int, float, bool)):
                return item
            if isinstance(item, Path):
                return str(item)
            return {"type": str(type(item))}

        structure = encode(value, name)
        save_file(tensors, str(output / f"{name}.safetensors"))
        (output / f"{name}.json").write_text(json.dumps(structure, indent=2) + "\n")

    def source_git(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(source), *arguments], text=True
        ).strip()

    raw = json.loads(request_path.read_text())
    (output / "request.json").write_text(json.dumps(raw, indent=2) + "\n")
    manifest = {
        "source_repo": str(source),
        "source_commit": source_git("rev-parse", "HEAD"),
        "source_dirty": source_git("status", "--porcelain", "--untracked-files=no"),
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": hashlib.sha256(
            (checkpoint / "config.json").read_bytes()
        ).hexdigest(),
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
        "attention_backend": args.attention_backend,
        "precision": "bfloat16",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "pid": os.getpid(),
        "status": "started",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for key in tuple(os.environ):
        if key.startswith("TORCHELASTIC_") or key in (
            "MASTER_ADDR",
            "MASTER_PORT",
            "GROUP_RANK",
            "ROLE_RANK",
        ):
            os.environ.pop(key, None)
    os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", LOCAL_WORLD_SIZE="1")
    store = torch.distributed.TCPStore("127.0.0.1", 0, 1, True)
    torch.distributed.init_process_group("nccl", store=store, rank=0, world_size=1)
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="pwm-reference-") as scratch:
            setup = OmniSetupOverrides(
                checkpoint_path=str(checkpoint),
                config_file=str(checkpoint / "config.json"),
                output_dir=Path(scratch),
                experiment_overrides=[
                    "model.config.ema.enabled=false",
                    "model.config.action_tower_init=checkpoint",
                    "model.config.fixed_sequence_length.enabled=false",
                    "model.config.action_tower_und_short_tokens=0",
                    f"model.config.action_tower_attention_backend={args.attention_backend}",
                    "model.config.precision=bfloat16",
                ],
                use_ema_weights=False,
                use_torch_compile=False,
                use_cuda_graphs=False,
                guardrails=False,
                dp_shard_size=1,
                dp_replicate_size=1,
                cp_size=1,
                cfgp_size=1,
            ).build_setup(world_size=1)
            model = OmniInference.create(setup).model.eval()
            manifest["numerics"] = {
                "cudnn_version": torch.backends.cudnn.version(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_sdpa": torch.backends.cuda.cudnn_sdp_enabled(),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            spec = OmegaConf.to_container(model.config.action128_spec, resolve=True)
            model.net.action_normalization.validate_contract()
            model.net.action_normalization._reset_transforms()
            seen = set()

            def hook(name):
                def capture(module, inputs, kwargs, result):
                    if name not in seen:
                        seen.add(name)
                        save_tree(
                            name, {"inputs": inputs, "kwargs": kwargs, "output": result}
                        )

                return capture

            for name in (
                "time_embedder",
                "action_time_embedder",
                "vae2llm",
                "action2llm",
            ):
                getattr(model.net, name).register_forward_hook(
                    hook(name), with_kwargs=True
                )
            model.net.register_forward_hook(hook("network"), with_kwargs=True)
            model.net.language_model.model.embed_tokens.register_forward_hook(
                hook("text_embedding"), with_kwargs=True
            )
            original_prepare = model._prepare_inference_data

            def capture_prepare(*inputs, **kwargs):
                result = original_prepare(*inputs, **kwargs)
                save_tree("preparation", {"output": result})
                save_tree(
                    "token_ids",
                    {
                        "conditional": torch.tensor(result[2][0]),
                        "unconditional": torch.tensor(result[3][0]),
                    },
                )
                return result

            model._prepare_inference_data = capture_prepare
            original_velocity = model._get_velocity

            def capture_velocity(*inputs, **kwargs):
                first = "first_velocity" not in seen
                if first:
                    seen.add("first_velocity")
                    save_tree(
                        "first_velocity_input",
                        {"noise_x": kwargs["noise_x"], "timestep": kwargs["timestep"]},
                    )
                result = original_velocity(*inputs, **kwargs)
                if first:
                    save_tree("first_velocity", result)
                return result

            model._get_velocity = capture_velocity
            original_compact = small_action_tower.compact_pack

            def capture_compact(pack, positions, position_ids, *inputs, **kwargs):
                result = original_compact(
                    pack, positions, position_ids, *inputs, **kwargs
                )
                if "positions" not in seen:
                    seen.add("positions")
                    positions_split = from_all_seq(position_ids.transpose(0, 1), pack)
                    nu = int(pack["_num_causal_tokens"])
                    save_tree(
                        "positions",
                        {
                            "all": position_ids,
                            "und": positions_split["causal_seq"][:nu].transpose(0, 1),
                            "video": positions_split["full_only_seq"][
                                result["_small_vi"]
                            ].transpose(0, 1),
                            "action": positions_split["full_only_seq"][
                                pack["_action_indices"]
                            ].transpose(0, 1),
                            "rope": result["_small_rope"],
                            "action_rope": result["_small_action_rope"],
                        },
                    )
                return result

            small_action_tower.compact_pack = capture_compact
            original_layer = small_action_tower.small_layer_forward
            layer_count = 0

            def capture_layer(layer, pack, **kwargs):
                nonlocal layer_count
                index = layer_count
                layer_count += 1
                names = ("causal_seq", "full_only_seq", "action_seq")
                captured = (
                    {name: pack[name].detach().cpu().clone() for name in names}
                    if index in (0, 35)
                    else None
                )
                result = original_layer(layer, pack, **kwargs)
                if captured is not None:
                    save_tree(
                        f"layer_{index:02d}",
                        {
                            "inputs": captured,
                            "output": {name: result[0][name] for name in names},
                        },
                    )
                return result

            small_action_tower.small_layer_forward = capture_layer
            prepared, batch = prepare_wam_request(
                OmniSampleOverrides.model_validate(raw),
                spec,
                model.config,
                Path(scratch),
                model.input_video_key,
            )
            save_tree("request_preparation", {"prepared": prepared, "batch": batch})
            batch = misc.to(batch, device="cuda")
            with torch.inference_mode():
                samples = model.generate_samples_from_batch(
                    batch,
                    seed=[prepared.seed],
                    num_steps=prepared.num_steps,
                    guidance=prepared.guidance,
                    shift=prepared.shift,
                    guidance_interval=prepared.sample_args["guidance_interval"],
                    normalize_cfg=prepared.sample_args["normalize_cfg"],
                    sigma_max=prepared.sample_args["sigma_max"],
                    has_negative_prompt="neg_ai_caption" in batch,
                    reuse_small_action_metadata=True,
                    reuse_small_action_und=False,
                )
                save_tree("samples", samples)
                normalized = samples["action"][0].float()
                domain = spec["registry"]["domains"].index(prepared.action_domain)
                physical = model.net.action_normalization.denormalize(
                    normalized, domain
                )
                mask = torch.tensor(
                    prepared.action_dim_mask, device=normalized.device, dtype=torch.bool
                )
                save_tree(
                    "actions",
                    {
                        "normalized": normalized.masked_fill(~mask, 0),
                        "physical": physical.masked_fill(~mask, 0),
                    },
                )
            manifest.update(
                status="complete",
                elapsed_seconds=time.perf_counter() - started,
                layer_calls=layer_count,
            )
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "output": str(output),
                        "status": "complete",
                        "elapsed_seconds": manifest["elapsed_seconds"],
                    }
                )
            )
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
