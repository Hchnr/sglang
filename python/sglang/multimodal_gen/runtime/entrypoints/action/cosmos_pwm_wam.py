# SPDX-License-Identifier: Apache-2.0
"""Model-owned WAM request lowering and bounded video attachment storage."""

from __future__ import annotations

import copy
import math
import os
import tempfile
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
from sglang.multimodal_gen.configs.sample.cosmos_pwm_wam import (
    CosmosPWMWAMSamplingParams,
)
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse


def lower_pwm_wam_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Lower native action or original Cosmos input to one WAM request."""
    if not isinstance(payload, dict):
        raise TypeError("WAM request must be an object")
    runtime = payload.get("runtime") or {}
    if not isinstance(runtime, dict):
        raise TypeError("runtime must be an object")
    unsupported_runtime = set(runtime) - {
        "output_format",
        "response_format",
        "return_timing",
        "prefix_cache",
        "cuda_graph",
    }
    if unsupported_runtime:
        raise ValueError(
            f"Unsupported WAM runtime fields: {sorted(unsupported_runtime)}"
        )
    for name in ("prefix_cache", "cuda_graph"):
        if runtime.get(name) not in (None, False, "false"):
            raise ValueError(f"Cosmos PWM WAM does not support runtime.{name}")
    if "wam_request" in payload:
        unknown = set(payload) - {"wam_request", "runtime", "model", "id", "request_id"}
        if unknown:
            raise ValueError(
                "Sampling fields belong inside wam_request; unsupported wrapper "
                f"fields: {sorted(unknown)}"
            )
        raw = payload["wam_request"]
        if not isinstance(raw, dict):
            raise ValueError("wam_request must be an object")
        return copy.deepcopy(raw)
    if "model_mode" in payload:
        return copy.deepcopy(payload)

    unknown = set(payload) - {
        "input",
        "parameters",
        "runtime",
        "model",
        "id",
        "request_id",
    }
    if unknown:
        raise ValueError(f"Unsupported WAM envelope fields: {sorted(unknown)}")
    inputs = payload.get("input", {})
    parameters = payload.get("parameters", {})
    if not isinstance(inputs, dict) or not isinstance(parameters, dict):
        raise TypeError("WAM input and parameters must be objects")
    observation = inputs.get("observation", {})
    if not isinstance(observation, dict):
        raise TypeError("WAM observation must be an object")
    options = {}
    for source in (
        {
            key: value
            for key, value in inputs.items()
            if key not in ("observation", "task")
        },
        observation,
        parameters,
    ):
        duplicate = options.keys() & source.keys()
        if duplicate:
            raise ValueError(f"WAM fields supplied twice: {sorted(duplicate)}")
        options.update(source)
    mode = options.pop("action_mode", "wam")
    if mode != "wam":
        raise ValueError("Cosmos PWM WAM requires action_mode='wam'")
    raw = {"model_mode": "wam"}
    aliases = {
        "num_inference_steps": "num_steps",
        "guidance_scale": "guidance",
        "flow_shift": "shift",
        "action_horizon": "action_chunk_size",
    }
    fields = {
        "prompt",
        "prompt_path",
        "vision_path",
        "num_steps",
        "guidance",
        "shift",
        "seed",
        "fps",
        "num_frames",
        "image_size",
        "action_chunk_size",
        "extra",
        "negative_prompt",
        "normalize_cfg",
        "guidance_interval",
        "name",
    }
    action_fields = {
        "action_domain",
        "action_fps",
        "action_dim_mask",
        "selected_roles",
        "camera_layout",
        "task_title",
        "return_video",
    }
    action128 = {}
    for key, value in options.items():
        name = aliases.get(key, key)
        if name in fields:
            if name in raw:
                raise ValueError(f"WAM parameter {name} was supplied twice")
            raw[name] = copy.deepcopy(value)
        elif name in action_fields:
            action128[name] = copy.deepcopy(value)
        elif name in ("image", "image_path", "input_reference", "video", "video_path"):
            if "vision_path" in raw:
                raise ValueError("WAM accepts exactly one observation canvas")
            raw["vision_path"] = value
        else:
            raise ValueError(f"Unsupported WAM field: {key}")
    if "prompt" not in raw and "prompt_path" not in raw:
        raw["prompt"] = inputs.get("task", inputs.get("prompt", ""))
    extra = raw.setdefault("extra", {})
    if not isinstance(extra, dict) or not isinstance(extra.get("action128", {}), dict):
        raise TypeError("extra.action128 must be an object")
    duplicate = set(action128) & set(extra.get("action128", {}))
    if duplicate:
        raise ValueError(
            f"WAM Action128 fields were supplied twice: {sorted(duplicate)}"
        )
    extra.setdefault("action128", {}).update(action128)
    return raw


def build_pwm_wam_action_sampling_params(payload, server_args):
    params = CosmosPWMWAMSamplingParams(
        request_id=payload.get("request_id") or payload.get("id"),
        wam_request=lower_pwm_wam_request(payload),
    )
    params._adjust(server_args)
    return params


def pwm_wam_response_metadata(output: dict[str, Any], *, preserve_numpy=False) -> dict:
    metadata = {
        name: output[name]
        for name in (
            "action_domain",
            "action_dim_mask",
            "action_fps",
            "action_timestamps_seconds",
            "action_units",
            "normalized_actions",
            "seed",
            "contract_fingerprint",
            "conditioning_caption_sha256",
            "timing_seconds",
            "checkpoint_path",
            "checkpoint_config_sha256",
            "asset_file_sha256",
            "loader_info",
        )
        if name in output
    }
    if not preserve_numpy:
        metadata = numpy_to_lists(metadata)
    return metadata


def numpy_to_lists(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: numpy_to_lists(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [numpy_to_lists(item) for item in value]
    return value


class _VideoDownloadResponse(StreamingResponse):
    """Own an already-open video through normal completion or client disconnect."""

    def __init__(self, stream, filename: str):
        self._stream = stream

        async def chunks():
            try:
                while chunk := await run_in_threadpool(stream.read, 64 * 1024):
                    yield chunk
            finally:
                stream.close()

        super().__init__(
            chunks(),
            media_type="video/mp4",
            headers={
                "content-length": str(os.fstat(stream.fileno()).st_size),
                "content-disposition": f'attachment; filename="{filename}"',
            },
        )

    async def __call__(self, scope, receive, send):
        # StreamingResponse does not run background tasks if send raises. The
        # response also owns closure when the iterator never starts or is paused.
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._stream.close()


def pwm_wam_video_response(store: ActionVideoStore, video_id: str):
    stream = store.open(video_id)
    try:
        return _VideoDownloadResponse(stream, f"{video_id}.mp4")
    except BaseException:
        stream.close()
        raise


class ActionVideoStore:
    """One server's temporary videos, bounded by count and one-hour retention.

    Files are removed on access after expiry, eviction, explicit deletion, or
    normal process exit. A restarted server cannot access earlier files.
    """

    def __init__(self, directory: Path, *, max_entries=128, ttl_seconds=3600):
        if max_entries < 1 or ttl_seconds <= 0:
            raise ValueError("Video retention count and duration must be positive")
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = tempfile.TemporaryDirectory(
            prefix="pwm-actions-", dir=directory
        )
        self.directory = Path(self._directory.name)
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, tuple[Path, float]] = OrderedDict()
        self._lock = RLock()

    def close(self) -> None:
        with self._lock:
            self._entries.clear()
            self._directory.cleanup()

    def _prune(self) -> None:
        now = time.time()
        for video_id, (path, expiry) in list(self._entries.items()):
            if expiry <= now:
                path.unlink(missing_ok=True)
                del self._entries[video_id]

    def save_output(self, output: dict[str, Any]) -> dict[str, Any]:
        response = dict(output)
        frames = response.pop("video_frames", None)
        if frames is None:
            return response
        frames = np.asarray(frames)
        fps = float(response["video_fps"])
        if (
            frames.dtype != np.uint8
            or frames.ndim != 4
            or frames.shape[-1] != 3
            or min(frames.shape) <= 0
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError(
                "WAM video requires uint8 [T,H,W,3] frames and positive FPS"
            )
        from sglang.multimodal_gen.runtime.entrypoints.utils import _save_video_ffmpeg

        video_id = uuid.uuid4().hex
        path = self.directory / f"{video_id}.mp4"
        try:
            _save_video_ffmpeg(str(path), list(frames), fps=fps, quality=10)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        expires_at = time.time() + self.ttl_seconds
        with self._lock:
            self._prune()
            while len(self._entries) >= self.max_entries:
                _, (old_path, _) = self._entries.popitem(last=False)
                old_path.unlink(missing_ok=True)
            self._entries[video_id] = (path, expires_at)
        response["video"] = {
            "id": video_id,
            "url": f"/v1/actions/videos/{video_id}/content",
            "mime_type": "video/mp4",
            "fps": fps,
            "shape": list(frames.shape),
            "expires_at": int(expires_at),
        }
        return response

    def resolve(self, video_id: str) -> Path:
        with self._lock:
            self._prune()
            if video_id not in self._entries:
                raise KeyError(video_id)
            return self._entries[video_id][0]

    def open(self, video_id: str):
        """Acquire a read handle before deletion/eviction can unlink the path."""
        with self._lock:
            return self.resolve(video_id).open("rb")

    def delete(self, video_id: str) -> None:
        with self._lock:
            self._prune()
            path, _ = self._entries.pop(video_id)
            path.unlink(missing_ok=True)

    def list_files(self) -> list[str]:
        with self._lock:
            self._prune()
            return [f"{video_id}.mp4" for video_id in self._entries]
