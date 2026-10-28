# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from vllm-omni-pwm d3e9aad33e861b69be16c4f341695e1cc12b3314.
"""The Action128 WAM request contract, shared by Python and HTTP execution.

Adapted from the native PWM implementation in vLLM-Omni. Media materialization
and defaults follow the current Cosmos action-serving contract. No training
framework or GPU model is imported by this module.
"""

import copy
import hashlib
import json
import math
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import msgspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch import Tensor


class PwmRequest(msgspec.Struct, kw_only=True):
    caption: str
    negative_prompt: str
    pixels: Tensor
    domain: str
    normalization_id: int
    projection_id: int
    mask: list[bool]
    video_fps: float
    action_fps: float
    action_length: int
    video_frames: int
    seam: float | None
    seed: int
    steps: int
    guidance: float
    shift: float
    normalize_cfg: bool
    guidance_interval: tuple[float, float] | None
    return_video: bool
    enable_und_cache: bool | None = None


def canvas_seam(roles, layout, height=None, width=None):
    if (
        not isinstance(roles, list)
        or not 1 <= len(roles) <= 3
        or len(set(roles)) != len(roles)
    ):
        raise ValueError("selected_roles must contain one to three unique roles")
    if len(roles) == 1:
        if layout is not None:
            raise ValueError("Single-camera layout must be null")
        return None
    locations = (
        ["top", "bottom"] if len(roles) == 2 else ["top", "bottom_left", "bottom_right"]
    )
    if not isinstance(layout, list) or len(layout) != len(roles):
        raise ValueError("camera_layout must match selected_roles")
    for role, location, item in zip(roles, locations, layout):
        if not isinstance(item, list) or len(item) != 4 or item[:2] != [role, location]:
            raise ValueError("Invalid camera_layout role or location")
        if any(type(x) is not int or x <= 0 for x in item[2:]):
            raise ValueError("Camera dimensions must be positive integer pixels")
    # Admission validates declared geometry without reading remote media.
    # Materialization also checks the declaration against the decoded pixels.
    height = layout[0][3] + layout[1][3] if height is None else height
    width = layout[0][2] if width is None else width
    seam = layout[0][3]
    if layout[0][2] != width or not 0 < seam < height:
        raise ValueError("Top camera does not match canvas")
    if (
        any(row[3] != height - seam for row in layout[1:])
        or sum(row[2] for row in layout[1:]) != width
    ):
        raise ValueError("Bottom cameras do not match canvas")
    return seam / 32


@contextmanager
def _local_media(value: str | Path, base_dir: Path | None):
    """Keep a downloaded observation alive only while its first frame is read."""
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("vision_path must be a local file or HTTP(S) URL")
    parsed = urlparse(str(value))
    if parsed.scheme in ("http", "https"):
        import requests

        with tempfile.TemporaryDirectory(prefix="sglang-pwm-input-") as directory:
            path = Path(directory) / ("observation" + Path(parsed.path).suffix)
            try:
                with requests.get(str(value), stream=True, timeout=60) as response:
                    response.raise_for_status()
                    with path.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            output.write(chunk)
            except requests.RequestException as exc:
                raise ValueError("Cannot download WAM observation") from exc
            yield path
        return
    if parsed.scheme:
        raise ValueError("vision_path only supports local files and HTTP(S) URLs")
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.is_file():
        raise ValueError(f"Missing WAM observation: {path}")
    yield path


def read_observation(value: str | Path, *, base_dir: Path | None = None) -> Tensor:
    """Decode exactly the first RGB frame, preserving the source canvas pixels."""
    with _local_media(value, base_dir) as path:
        try:
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB")).copy()
        except UnidentifiedImageError:
            import av

            try:
                with av.open(str(path)) as container:
                    if not container.streams.video:
                        raise ValueError("WAM observation has no video stream")
                    frame = next(container.decode(video=0), None)
                    if frame is None:
                        raise ValueError("WAM observation video contains no frames")
                    array = frame.to_ndarray(format="rgb24")
            except av.FFmpegError as exc:
                raise ValueError(f"Cannot decode WAM observation: {path}") from exc
        except OSError as exc:
            raise ValueError(f"Cannot decode WAM observation: {path}") from exc
    return torch.from_numpy(array).permute(2, 0, 1)[:, None].contiguous()


def load_request(path: str | Path) -> dict:
    """Resolve file-backed request paths relative to the request JSON directory."""
    path = Path(path).expanduser().resolve()
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("WAM request JSON must contain one object")  # noqa: TRY004 -- Request validation uses ValueError for HTTP 400.
    for name in ("vision_path", "prompt_path"):
        value = raw.get(name)
        if value is not None and not urlparse(str(value)).scheme:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                raw[name] = str(path.parent / candidate)
    return raw


def validate_request(raw: dict, spec: dict, *, base_dir: Path | None = None) -> dict:
    """Resolve defaults and validate before queueing, without reading media.

    The returned seed and prompt are frozen so admission and GPU execution see
    the same request. Observation decoding happens once in prepare_request.
    """
    if not isinstance(raw, dict):
        raise ValueError("WAM request must be an object")  # noqa: TRY004 -- Request validation uses ValueError for HTTP 400.
    fixed_defaults = {
        "duration_template": "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS.",
        "resolution_template": "This video is of {height}x{width} resolution.",
        "inverse_duration_template": "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS.",
        "inverse_resolution_template": "This video is not of {height}x{width} resolution.",
        "negative_metadata_mode": "none",
        "negative_prompt_keep_metadata": True,
        "aspect_ratio": "16,9",
        "video_save_quality": 10,
        "image_save_quality": 95,
    }
    supported = {
        "model_mode",
        "model",
        "name",
        "domain_name",
        "output_dir",
        "extra",
        "prompt",
        "prompt_path",
        "negative_prompt",
        "vision_path",
        "seed",
        "num_steps",
        "guidance",
        "shift",
        "guidance_interval",
        "normalize_cfg",
        "fps",
        "action_chunk_size",
        "num_frames",
        "image_size",
        "num_outputs",
        "n",
        "num_outputs_per_prompt",
        "raw_action_dim",
        "sampler",
        "sigma_max",
        "condition_frame_indexes_vision",
        "condition_frame_indexes_action",
        "action_path",
        "state_path",
        "robot_state",
        "tensors_file",
        "pickle_file",
        "autoregressive",
        "enable_sound",
        "native_prompt_upsampling",
        "prompt_upsampling",
        "skip_text_tokens_for_cfg",
        "defaults_file",
        "enable_cuda_graph",
        "enable_breakable_cuda_graph",
        "enable_prefix_cache",
        "enable_torch_compile",
        "enable_und_cache",
    } | fixed_defaults.keys()
    unknown = set(raw) - supported
    if unknown:
        raise ValueError(f"Unsupported WAM request fields: {sorted(unknown)}")
    if "enable_und_cache" in raw and type(raw["enable_und_cache"]) is not bool:
        raise ValueError("enable_und_cache must be a boolean")
    for name, default in fixed_defaults.items():
        if raw.get(name) is not None and raw[name] != default:
            raise ValueError(f"Native WAM requires the default {name}={default!r}")
    for field in (
        "action_path",
        "state_path",
        "robot_state",
        "condition_frame_indexes_action",
        "tensors_file",
        "pickle_file",
    ):
        if raw.get(field) not in (None, []):
            raise ValueError(f"Native no-state WAM does not accept {field}")
    if raw.get("model_mode") != "wam":
        raise ValueError("Native PWM requires model_mode='wam'")
    for flag in (
        "autoregressive",
        "enable_sound",
        "native_prompt_upsampling",
        "prompt_upsampling",
        "skip_text_tokens_for_cfg",
        "enable_cuda_graph",
        "enable_breakable_cuda_graph",
        "enable_prefix_cache",
        "enable_torch_compile",
    ):
        if raw.get(flag):
            raise ValueError(f"Native WAM does not support {flag}")
    if raw.get("sampler") not in (None, "unipc"):
        raise ValueError("Native WAM currently supports the UniPC sampler only")
    # Cosmos' WAM defaults include the inert EDM parameter 80.0. Keep that
    # default round-trip, but do not pretend to implement custom EDM schedules.
    if raw.get("sigma_max") not in (None, 80, 80.0):
        raise ValueError("Custom sigma_max requires EDM; native WAM uses UniPC")
    if raw.get("defaults_file") is not None:
        raise ValueError("Apply defaults_file values to the request before submission")
    if any(
        raw.get(name) not in (None, 1)
        for name in ("num_outputs", "n", "num_outputs_per_prompt")
    ) or raw.get("raw_action_dim") not in (
        None,
        128,
    ):
        raise ValueError("Native WAM requires one Action128 output per request")
    if raw.get("condition_frame_indexes_vision") not in (None, [0]):
        raise ValueError("Native WAM conditions on vision frame zero only")
    raw = copy.deepcopy(raw)
    extra = raw.get("extra") or {}
    if not isinstance(extra, dict) or not isinstance(extra.get("action128", {}), dict):
        raise ValueError("extra.action128 must be an object")  # noqa: TRY004 -- Request validation uses ValueError for HTTP 400.
    ext = copy.deepcopy(extra.get("action128", {}))
    allowed = {
        "action_domain",
        "action_dim_mask",
        "action_fps",
        "selected_roles",
        "camera_layout",
        "task_title",
        "return_video",
    }
    if set(ext) - allowed:
        raise ValueError(f"Unknown Action128 extensions: {set(ext) - allowed}")
    domain = ext.get("action_domain")
    registry = spec["registry"]
    if domain not in registry["domains"]:
        raise ValueError("Unknown Action128 source domain")
    mask = ext.get("action_dim_mask")
    if (
        not isinstance(mask, list)
        or len(mask) != 128
        or any(type(x) is not bool for x in mask)
        or not any(mask)
    ):
        raise ValueError("action_dim_mask must contain 128 booleans and an active slot")
    try:
        fps = float(24 if raw.get("fps") is None else raw["fps"])
        action_fps = float(fps if ext.get("action_fps") is None else ext["action_fps"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid video or action FPS") from exc
    length = 16 if raw.get("action_chunk_size") is None else raw["action_chunk_size"]
    if (
        not math.isfinite(fps)
        or not 0 < fps <= 30
        or not math.isfinite(action_fps)
        or not 0 < action_fps <= 512
    ):
        raise ValueError("Invalid video or action FPS")
    if type(length) is not int or length < 1 or length / action_fps > 2:
        raise ValueError("Action horizon must be positive and at most two seconds")
    duration = length / action_fps
    frames = 4 * math.ceil(math.ceil(duration * fps - 1e-8) / 4) + 1
    if raw.get("num_frames") not in (None, frames):
        raise ValueError(f"num_frames must equal {frames}")
    seed = raw.get("seed")
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**31 - 1)
    if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be an integer between 0 and 2**31 - 1")
    prompt = raw.get("prompt")
    if prompt is None and raw.get("prompt_path") is not None:
        if not isinstance(raw["prompt_path"], (str, Path)):
            raise ValueError("prompt_path must be a local text file")
        path = Path(raw["prompt_path"]).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        try:
            prompt = path.read_text().strip()
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Cannot read WAM prompt: {path}") from exc
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A nonempty prompt is required")
    steps = 30 if raw.get("num_steps") is None else raw["num_steps"]
    try:
        guidance = float(1 if raw.get("guidance") is None else raw["guidance"])
        shift = float(10 if raw.get("shift") is None else raw["shift"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid diffusion sampling parameters") from exc
    if (
        type(steps) is not int
        or steps < 1
        or not math.isfinite(guidance)
        or not 0 <= guidance <= 7
        or not math.isfinite(shift)
        or not 0 < shift <= 100
    ):
        raise ValueError("Invalid diffusion sampling parameters")
    interval = raw.get("guidance_interval")
    if interval is not None and (
        not isinstance(interval, (list, tuple))
        or len(interval) != 2
        or not all(
            isinstance(x, (float, int))
            and not isinstance(x, bool)
            and math.isfinite(x)
            and x >= 0
            for x in interval
        )
        or interval[0] > interval[1]
    ):
        raise ValueError("guidance_interval must be a finite ordered pair")
    negative_prompt = (
        "" if raw.get("negative_prompt") is None else raw["negative_prompt"]
    )
    if not isinstance(negative_prompt, str):
        raise ValueError("negative_prompt must be text")  # noqa: TRY004 -- Request validation uses ValueError for HTTP 400.
    normalize_cfg = raw.get("normalize_cfg")
    normalize_cfg = False if normalize_cfg is None else normalize_cfg
    return_video = ext.get("return_video", True)
    if not isinstance(normalize_cfg, bool) or not isinstance(return_video, bool):
        raise ValueError("normalize_cfg and return_video must be booleans")  # noqa: TRY004 -- Request validation uses ValueError for HTTP 400.
    vision_path = raw.get("vision_path")
    if not isinstance(vision_path, (str, Path)) or not str(vision_path):
        raise ValueError("vision_path is required for WAM")
    parsed = urlparse(str(vision_path))
    if parsed.scheme and parsed.scheme not in ("http", "https"):
        raise ValueError("vision_path only supports local files and HTTP(S) URLs")
    if not parsed.scheme:
        vision_path = Path(vision_path).expanduser()
        if not vision_path.is_absolute() and base_dir is not None:
            vision_path = base_dir / vision_path
        if not vision_path.is_file():
            raise ValueError(f"Missing WAM observation: {vision_path}")
        vision_path = vision_path.resolve()
    image_size = raw.get("image_size")
    if image_size is not None and (
        type(image_size) is not int or image_size <= 0 or image_size % 16
    ):
        raise ValueError("image_size must be a positive multiple of 16")
    roles = ext.get("selected_roles", ["main"])
    if (
        not isinstance(roles, list)
        or not 1 <= len(roles) <= 3
        or not all(isinstance(role, str) and role for role in roles)
        or len(set(roles)) != len(roles)
    ):
        raise ValueError("selected_roles must contain one to three unique role names")
    layout = ext.get("camera_layout")
    if len(roles) == 1 and layout is not None:
        raise ValueError("Single-camera layout must be null")
    if len(roles) > 1 and (not isinstance(layout, list) or len(layout) != len(roles)):
        raise ValueError("camera_layout must match selected_roles")
    canvas_seam(roles, layout)
    raw.update(
        vision_path=str(vision_path),
        prompt=prompt,
        seed=seed,
        fps=fps,
        action_chunk_size=length,
        num_frames=frames,
        num_steps=steps,
        guidance=guidance,
        shift=shift,
        negative_prompt=negative_prompt,
        normalize_cfg=normalize_cfg,
        guidance_interval=list(interval) if interval is not None else None,
    )
    ext.update(action_fps=action_fps, selected_roles=roles, return_video=return_video)
    raw["extra"] = {**extra, "action128": ext}
    return raw


def prepare_request(
    raw: dict, spec: dict, *, base_dir: Path | None = None
) -> PwmRequest:
    raw = validate_request(raw, spec, base_dir=base_dir)
    ext = raw["extra"]["action128"]
    registry = spec["registry"]
    domain, mask = ext["action_domain"], ext["action_dim_mask"]
    fps, action_fps = raw["fps"], ext["action_fps"]
    length, frames = raw["action_chunk_size"], raw["num_frames"]
    duration, seed = length / action_fps, raw["seed"]
    prompt = raw["prompt"]
    pixels = read_observation(raw.get("vision_path"), base_dir=base_dir)
    height, width = pixels.shape[-2:]
    roles, layout = ext.get("selected_roles", ["main"]), ext.get("camera_layout")
    canvas_seam(roles, layout, height, width)
    target_h = raw.get("image_size")
    if target_h is None:
        target_h = 16 * math.ceil(height / 16)
    if type(target_h) is not int or target_h <= 0 or target_h % 16:
        raise ValueError("image_size must be a positive multiple of 16")
    target_w = max(16, 16 * round(width * target_h / height / 16))
    if (target_h, target_w) != (height, width):
        pixels = (
            F.interpolate(
                pixels.permute(1, 0, 2, 3).float(),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 0, 2, 3)
        )
        if layout is not None:
            seam = round(layout[0][3] * target_h / height)
            layout[0][2:] = [target_w, seam]
            if len(layout) == 2:
                layout[1][2:] = [target_w, target_h - seam]
            else:
                left = round(layout[1][2] * target_w / width)
                layout[1][2:] = [left, target_h - seam]
                layout[2][2:] = [target_w - left, target_h - seam]
        height, width = target_h, target_w
    seam = canvas_seam(roles, layout, height, width)
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        caption = None
    title = ext.get("task_title") or raw.get("name") or "action128_wam"
    if isinstance(caption, dict):
        title = caption.get("task_title", title)
        prompt = caption.get("command", caption.get("source_task_description", prompt))
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("WAM caption command must be nonempty text")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("WAM task_title must be nonempty text")
    domain_id = registry["domains"].index(domain)
    formatted = {
        "task_title": title,
        "command": prompt,
        "embodiment": registry["embodiments"][
            registry["domain_embodiments"][domain_id]
        ],
        "duration": f"{duration:g}s",
        "fps": fps,
        "action_frequency_hz": action_fps,
        "resolution": {"W": width, "H": height},
        "aspect_ratio": f"{width // math.gcd(width, height)}:{height // math.gcd(width, height)}",
    }
    if layout is not None:
        formatted["camera_layout"] = layout
    draw = (
        int.from_bytes(
            hashlib.blake2b(
                json.dumps(seed, sort_keys=True).encode(), digest_size=16
            ).digest(),
            "big",
        )
        / 2**128
    )
    if draw < 0.9:
        formatted["task_plan"] = []
    if isinstance(caption, dict):
        for key in (
            "embodiment",
            "duration",
            "fps",
            "action_frequency_hz",
            "resolution",
            "aspect_ratio",
            "camera_layout",
        ):
            if key in formatted:
                caption[key] = formatted[key]
            else:
                caption.pop(key, None)
        caption.setdefault("command", prompt)
        caption.setdefault("task_title", title)
    else:
        caption = formatted
    return PwmRequest(
        caption=json.dumps(caption, ensure_ascii=False),
        negative_prompt=raw["negative_prompt"],
        pixels=pixels,
        domain=domain,
        normalization_id=domain_id,
        projection_id=registry["domain_projection_ids"][domain_id],
        mask=mask,
        video_fps=fps,
        action_fps=action_fps,
        action_length=length,
        video_frames=frames,
        seam=seam,
        seed=seed,
        steps=raw["num_steps"],
        guidance=raw["guidance"],
        shift=raw["shift"],
        normalize_cfg=raw["normalize_cfg"],
        guidance_interval=(
            tuple(raw["guidance_interval"])
            if raw["guidance_interval"] is not None
            else None
        ),
        return_video=ext["return_video"],
        enable_und_cache=raw.get("enable_und_cache"),
    )
