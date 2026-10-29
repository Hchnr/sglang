# SPDX-License-Identifier: Apache-2.0
"""Replay native WAM layer zero in any Torch environment, using frozen inputs.

This loads only that layer's checkpoint tensors and imports the standalone native
core file. It neither imports Cosmos nor initializes the SGLang serving stack.
Use the same command in the reference and target Python environments to isolate
Torch/library effects from model-port differences.
"""

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    target = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/multimodal_gen/runtime/models/dits/cosmos_pwm_wam/transformer.py"
    )
    spec = importlib.util.spec_from_file_location("pwm_native_core", target)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    prefix = "model.net.language_model.model.layers.0."
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    selected = {key: shard for key, shard in index.items() if key.startswith(prefix)}
    state = {}
    for shard in set(selected.values()):
        with safe_open(
            args.checkpoint / shard, framework="pt", device="cuda"
        ) as tensors:
            for name, filename in selected.items():
                if filename == shard:
                    state[name.removeprefix(prefix)] = tensors.get_tensor(name)
    with torch.device("meta"):
        layer = core.PwmThreeTowerLayer()
    layer.load_state_dict(state, assign=True, strict=True)
    layer.eval()
    fixture = load_file(args.reference / "layer_00.safetensors", device="cuda")
    positions = load_file(args.reference / "positions.safetensors", device="cuda")
    stream_names = ("causal_seq", "full_only_seq", "action_seq")
    streams = tuple(fixture["layer_00.inputs." + name] for name in stream_names)
    rope, action_rope = (
        tuple(
            tuple(
                positions[f"positions.{name}.{stream}.{component}"]
                for component in (0, 1)
            )
            for stream in range(3)
        )
        for name in ("rope", "action_rope")
    )
    observed = {}

    def hook(name):
        def capture(module, inputs, result):
            if isinstance(result, torch.Tensor):
                observed[name] = result.detach().cpu().contiguous().clone()

        return capture

    for name, module in layer.named_modules():
        if name:
            module.register_forward_hook(hook(name))
    original_attention = core.attention
    attention_calls = 0

    def capture_attention(*inputs, **kwargs):
        nonlocal attention_calls
        index = attention_calls
        attention_calls += 1
        for component, value in zip(("q", "k", "v"), inputs):
            observed[f"attention.{index}.{component}"] = (
                value.detach().cpu().contiguous().clone()
            )
        result = original_attention(*inputs, **kwargs)
        observed[f"attention.{index}.output"] = (
            result.detach().cpu().contiguous().clone()
        )
        return result

    core.attention = capture_attention
    with torch.inference_mode():
        results = layer(streams, rope, action_rope)
    metrics = {}
    for name, result in zip(stream_names, results):
        observed["output." + name] = result.detach().cpu().contiguous()
        expected = fixture["layer_00.output." + name]
        delta = result.double() - expected.double()
        metrics[name] = {
            "exact": bool(torch.equal(result, expected)),
            "max_abs": delta.abs().max().item(),
            "mae": delta.abs().mean().item(),
            "relative_l2": (delta.norm() / expected.double().norm()).item(),
        }
    save_file(observed, args.output / "intermediates.safetensors")
    report = {"torch": torch.__version__, "metrics": metrics}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
