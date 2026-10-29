# SPDX-License-Identifier: Apache-2.0
"""Startup contracts for external WAM assets; these do not identify weights."""

import hashlib
import math
from pathlib import Path


def validate_action_registry(spec: dict) -> None:
    registry = spec["registry"]
    domains, embodiments = registry["domains"], registry["embodiments"]
    for name, values in (("domains", domains), ("embodiments", embodiments)):
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"Invalid PWM Action128 registry {name}")
    for name, limit in (
        ("domain_projection_ids", 64),
        ("domain_embodiments", len(embodiments)),
    ):
        values = registry[name]
        if (
            not isinstance(values, list)
            or len(values) != len(domains)
            or any(type(value) is not int or not 0 <= value < limit for value in values)
        ):
            raise ValueError(f"Invalid PWM Action128 registry {name}")
    if len(spec["transforms"]) != len(domains):
        raise ValueError("PWM Action128 transforms must match registry domains")


def validate_vae_config(config) -> None:
    expected = {
        "z_dim": 48,
        "patch_size": 2,
        "scale_factor_spatial": 16,
        "scale_factor_temporal": 4,
        "in_channels": 12,
        "out_channels": 12,
        "base_dim": 160,
        "decoder_base_dim": 256,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "temperal_downsample": [False, True, True],
        "is_residual": True,
        "clip_output": False,
        "attn_scales": [],
        "dropout": 0.0,
    }
    for name, value in expected.items():
        actual = config.get(name)
        if isinstance(actual, tuple):
            actual = list(actual)
        if actual != value:
            raise ValueError(f"PWM VAE requires {name}={value!r}, got {actual!r}")
    for name in ("latents_mean", "latents_std"):
        values = config.get(name)
        if (
            not isinstance(values, (list, tuple))
            or len(values) != 48
            or any(
                not isinstance(x, (float, int)) or not math.isfinite(x) for x in values
            )
            or (name == "latents_std" and any(x <= 0 for x in values))
        ):
            raise ValueError(
                f"PWM VAE requires 48 finite {name} values and positive std"
            )


def validate_tokenizer(tokenizer) -> None:
    # Measured from the source checkpoint tokenizer using its Transformers
    # environment. Vocabulary size differs from the padded 151936-row embedding.
    if tokenizer.vocab_size != 151643 or len(tokenizer) != 151669:
        raise ValueError("PWM requires the checkpoint's Qwen vocabulary (151643+26)")
    if (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) != (
        None,
        151645,
        151643,
    ):
        raise ValueError("PWM tokenizer BOS/EOS/PAD IDs do not match the checkpoint")
    for token, token_id in {
        "<|endoftext|>": 151643,
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|vision_start|>": 151652,
        "<|vision_end|>": 151653,
        "<|image_pad|>": 151655,
        "<|video_pad|>": 151656,
    }.items():
        if tokenizer.convert_tokens_to_ids(token) != token_id:
            raise ValueError(f"PWM tokenizer special token mismatch: {token}")
    # Pin both the empty unconditional branch and the actual single-user chat
    # template path; checking IDs alone could miss a changed template renderer.
    for prompt, content_ids in (
        ("", []),
        (
            "WAM tokenizer probe: move cup.",
            [54, 1402, 45958, 21730, 25, 3271, 10525, 13],
        ),
    ):
        messages = [{"role": "user", "content": prompt}]
        options = {"add_generation_prompt": True, "add_vision_id": False}
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
        expected = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, **options
        )
        if rendered != expected or ids != [
            151644,
            872,
            198,
            *content_ids,
            151645,
            198,
            151644,
            77091,
            198,
        ]:
            raise ValueError("PWM tokenizer chat template/tokenization probe mismatch")


def snapshot_asset_files(vae_path: str, tokenizer_path: str) -> dict[str, str]:
    """Hash small configuration/tokenization files once, excluding model weights."""
    files = {"vae/config.json": Path(vae_path) / "config.json"}
    tokenizer_root = Path(tokenizer_path)
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    ):
        path = tokenizer_root / name
        if path.is_file():
            files[f"tokenizer/{name}"] = path
    for path in sorted((tokenizer_root / "chat_templates").glob("*.jinja")):
        files[f"tokenizer/chat_templates/{path.name}"] = path
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in files.items()
    }
