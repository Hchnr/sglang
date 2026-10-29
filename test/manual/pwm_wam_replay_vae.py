# SPDX-License-Identifier: Apache-2.0
"""Isolate native VAE arithmetic across Torch/Diffusers environments.

Uses frozen preprocessed pixels and final latents, never imports Cosmos or the
serving runtime. Captures the first encoder leaves plus raw/scaled latents and
decoded pixels to locate library-version differences.
"""

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cudnn-benchmark", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import diffusers
    import torch
    from diffusers import AutoencoderKLWan
    from safetensors.torch import load_file, save_file

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    target = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/multimodal_gen/runtime/models/dits/cosmos_pwm_wam/vae.py"
    )
    spec = importlib.util.spec_from_file_location("pwm_native_vae", target)
    compatibility = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compatibility)
    vae = (
        AutoencoderKLWan.from_pretrained(
            args.vae, torch_dtype=torch.bfloat16, local_files_only=True
        )
        .to("cuda")
        .eval()
    )
    compatibility.preserve_checkpoint_rms(vae)
    inputs = load_file(args.reference / "request_preparation.safetensors")
    preparation = load_file(args.reference / "preparation.safetensors")
    samples = load_file(args.reference / "samples.safetensors")
    observed = {}
    layouts = {}

    def capture(name, value):
        layouts[name] = {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "contiguous": value.is_contiguous(),
            "channels_last_3d": value.ndim == 5
            and value.is_contiguous(memory_format=torch.channels_last_3d),
        }
        observed[name] = value.detach().cpu().contiguous().clone()

    def hook(name):
        def save(module, inputs, output):
            if (
                name not in observed
                and len(observed) < 20
                and isinstance(output, torch.Tensor)
            ):
                capture(name, output)

        return save

    for name, module in vae.encoder.named_modules():
        if name and not list(module.children()):
            module.register_forward_hook(hook("encoder." + name))
    first_conv = vae.encoder.down_blocks[0].resnets[0].conv1

    def capture_first_conv(module, inputs):
        capture("first_conv.input", inputs[0])
        capture("first_conv.weight", module.weight)
        capture("first_conv.bias", module.bias)
        if len(inputs) > 1 and inputs[1] is not None:
            capture("first_conv.cache", inputs[1])

    first_conv.register_forward_pre_hook(capture_first_conv)
    pixels = (
        inputs["request_preparation.batch.video.0.0"][:, :1][None].to(
            "cuda", dtype=torch.float32
        )
        / 127.5
        - 1
    )
    mean = torch.tensor(
        vae.config.latents_mean, device="cuda", dtype=torch.bfloat16
    ).view(1, 48, 1, 1, 1)
    inverse = 1 / torch.tensor(
        vae.config.latents_std, device="cuda", dtype=torch.bfloat16
    ).view(1, 48, 1, 1, 1)
    with torch.inference_mode():
        encoded = vae.encode(pixels.to(torch.bfloat16)).latent_dist.mode()
        capture("raw_encoded", encoded)
        capture("scaled_encoded", ((encoded - mean) * inverse).float())
        decoded = vae.decode(
            samples["samples.vision.0"].to("cuda", dtype=torch.bfloat16) / inverse
            + mean,
            return_dict=False,
        )[0]
        capture("decoded", decoded)
    expected = preparation["preparation.output.1.x0_tokens_vision.0"][:, :, :1]
    delta = observed["scaled_encoded"].double() - expected.double()
    report = {
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "exact": bool(torch.equal(observed["scaled_encoded"], expected)),
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / expected.double().norm()).item(),
        "captured": list(observed),
        "layouts": layouts,
    }
    save_file(observed, args.output / "intermediates.safetensors")
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
