# SPDX-License-Identifier: Apache-2.0
"""Cosmos /generate compatibility facade backed exclusively by SGLang workers.

Run ``python -m sglang.multimodal_gen.runtime.entrypoints.action.pwm_compat
--help`` for the standalone HTTP launcher. This facade owns its DiffGenerator;
the main SGLang server's /generate route is not replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from sglang.multimodal_gen.runtime.entrypoints.action.cosmos_pwm_wam import (
    ActionVideoStore,
    lower_pwm_wam_request,
    numpy_to_lists,
    pwm_wam_response_metadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_pwm_wam.request import (
    validate_request,
)
from starlette.responses import FileResponse


def create_app(generator, *, output_dir: str | Path = "outputs/pwm-wam") -> FastAPI:
    """Serve one native generator using Cosmos request and SampleOutputs shapes."""
    server_args = generator.server_args
    checkpoint = Path(server_args.model_path)
    config = json.loads((checkpoint / "config.json").read_text())
    spec = config["model"]["config"]["action128_spec"]
    fingerprint = hashlib.blake2b(
        json.dumps(spec, sort_keys=True).encode(), digest_size=16
    ).hexdigest()
    model_name = server_args.served_model_name
    store = ActionVideoStore(Path(output_dir))
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            try:
                generator.shutdown()
            finally:
                store.close()

    app = FastAPI(title="Cosmos PWM WAM on SGLang", lifespan=lifespan)
    app.state.pwm_wam_video_store = store

    @app.get("/healthz")
    def healthz():
        return {"status": "ready"}

    @app.get("/models")
    def models():
        return [model_name]

    @app.get("/info")
    def info():
        return {
            "model_name": model_name,
            "models": [model_name],
            "checkpoint_path": str(checkpoint),
            "backend": "sglang",
            "pipeline": "CosmosPWMWAMPipeline",
            "output_dir": str(output_dir),
            "output_retention_seconds": store.ttl_seconds,
            "max_retained_videos": store.max_entries,
        }

    @app.get("/domains")
    def domains():
        return {
            "domains": spec["registry"]["domains"],
            "action_dim": 128,
            "contract_fingerprint": fingerprint,
        }

    @app.get("/outputs")
    def outputs():
        return {"files": store.list_files()}

    @app.get("/outputs/{filename}")
    def output_file(filename: str):
        if not re.fullmatch(r"[0-9a-f]{32}\.mp4", filename):
            raise HTTPException(status_code=404, detail="Output not found")
        try:
            path = store.resolve(filename.removesuffix(".mp4"))
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="Output expired or not found"
            ) from exc
        return FileResponse(path, media_type="video/mp4", filename=filename)

    @app.delete("/outputs/{filename}")
    def delete_output(filename: str):
        if not re.fullmatch(r"[0-9a-f]{32}\.mp4", filename):
            raise HTTPException(status_code=404, detail="Output not found")
        try:
            store.delete(filename.removesuffix(".mp4"))
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="Output expired or not found"
            ) from exc
        return {"file": filename, "deleted": True}

    @app.post("/generate")
    def generate(payload: dict):
        if payload.get("model") not in (None, model_name):
            raise HTTPException(
                status_code=404, detail=f"Unknown model: {payload['model']}"
            )
        if not lock.acquire(blocking=False):
            raise HTTPException(
                status_code=503,
                detail="Model is busy; retry after the current request",
                headers={"Retry-After": "1"},
            )
        try:
            try:
                raw = lower_pwm_wam_request(payload)
                raw = validate_request(raw, spec)
            except (ValueError, TypeError, OSError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            started = time.monotonic()
            result = generator.generate_action({"wam_request": raw})
            result = store.save_output(result)
            metadata = pwm_wam_response_metadata(result)
            if "normalized_actions" in metadata:
                metadata["normalized_action"] = metadata.pop("normalized_actions")
            metadata.update(
                checkpoint_path=str(checkpoint),
                elapsed_seconds=time.monotonic() - started,
            )
            files = []
            if result.get("video") is not None:
                files.append(f"{result['video']['id']}.mp4")
            raw.setdefault("name", f"wam_{uuid.uuid4().hex}")
            return {
                "args": numpy_to_lists(raw),
                "status": "success",
                "message": None,
                "stack_trace": None,
                "outputs": [
                    {
                        "content": {
                            "action": numpy_to_lists(result["actions"]),
                            "action128": metadata,
                        },
                        "files": files,
                    }
                ],
            }
        finally:
            lock.release()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--model-name", default="cosmos-pwm-wam")
    parser.add_argument("--output-dir", default="outputs/pwm-wam")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    import uvicorn
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

    generator = DiffGenerator.from_pretrained(
        model_path=args.model_path,
        backend="sglang",
        pipeline_class_name="CosmosPWMWAMPipeline",
        served_model_name=args.model_name,
        pipeline_config={
            "vae_path": args.vae_path,
            "tokenizer_path": args.tokenizer_path,
        },
        attention_backend="torch_sdpa",
        num_gpus=1,
        base_gpu_id=args.gpu_id,
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
    try:
        app = create_app(generator, output_dir=args.output_dir)
    except BaseException:
        generator.shutdown()
        raise
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
