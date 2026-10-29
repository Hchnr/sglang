# SPDX-License-Identifier: Apache-2.0
"""Compare multiple original-pixel native rollouts in an isolated Python runtime.

Use pwm_wam_isolated_native.py to launch this script with a frozen native leaf
snapshot. The model loads once; each case is compared against an independently
saved Cosmos response, without injecting reference latents or changing weights.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        nargs=3,
        action="append",
        required=True,
        metavar=("NAME", "REQUEST", "REFERENCE"),
    )
    parser.add_argument("--min-free-gpu-gib", type=float, default=60)
    parser.add_argument("--wait-for-gpu-seconds", type=float, default=0)
    parser.add_argument("--gpu-ready-marker", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from test_cosmos_pwm_wam import compare_reference, validate_result

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_pwm_wam.request import (
        load_request,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_pwm_wam.runtime import (
        PwmWamRuntime,
    )

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if "," in visible:
        raise ValueError("This diagnostic requires exactly one visible GPU")
    deadline = time.monotonic() + args.wait_for_gpu_seconds
    last_log = 0
    while True:
        info = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={visible}",
                    "--query-gpu=uuid,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            .strip()
            .split(",")
        )
        uuid, free_mib = info[0].strip(), float(info[1])
        rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines()
        occupants = [row for row in rows if uuid in row]
        if not occupants and free_mib >= args.min_free_gpu_gib * 1024:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Allocated GPU did not become available before the deadline"
            )
        if time.monotonic() - last_log >= 30:
            print(
                json.dumps(
                    {
                        "status": "waiting_for_gpu",
                        "gpu": visible,
                        "free_mib": free_mib,
                        "occupants": occupants,
                    }
                ),
                flush=True,
            )
            last_log = time.monotonic()
        time.sleep(5)
    if args.gpu_ready_marker is not None:
        args.gpu_ready_marker.write_text(
            json.dumps({"status": "loading", "time": time.time(), "gpu": visible})
            + "\n"
        )
    print("GPU available; loading frozen native leaf snapshot", flush=True)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < args.min_free_gpu_gib * 1024**3:
        raise RuntimeError("Allocated GPU was occupied during diagnostic imports")
    started = time.perf_counter()
    runtime = PwmWamRuntime(
        args.checkpoint,
        vae_path=str(args.vae),
        tokenizer_path=str(args.tokenizer),
        enable_und_cache=False,
    )
    report = {
        "torch": torch.__version__,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "load_seconds": time.perf_counter() - started,
        "enable_und_cache": False,
        "reference_latent_substitution": False,
        "cases": [],
    }
    report_path = args.output / "comparison.json"
    for name, request, reference in args.case:
        raw = load_request(Path(request))
        result = runtime.forward(raw)
        validate_result(result)
        target = args.output / name
        target.mkdir()
        arrays = {
            key: value for key, value in result.items() if isinstance(value, np.ndarray)
        }
        np.savez(target / "arrays.npz", **arrays)
        metadata = {key: value for key, value in result.items() if key not in arrays}
        (target / "result.json").write_text(json.dumps(metadata, indent=2) + "\n")
        row = {
            "name": name,
            "request": request,
            "reference": reference,
            "metrics": compare_reference(result, reference),
        }
        report["cases"].append(row)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
