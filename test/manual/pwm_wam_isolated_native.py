# SPDX-License-Identifier: Apache-2.0
"""Run a native WAM diagnostic with exact leaf files in another Python environment.

This offline-only launcher creates namespace packages instead of importing the
unrelated SGLang serving stack. No leaf operation is replaced. An immutable copy
of the native leaf files prevents concurrent development from changing a run.
The caller must supply optional dependencies through an external PYTHONPATH
overlay; this tool never installs packages or edits either source checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import runpy
import shlex
import sys
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    if "--output" not in arguments:
        parser.error("the diagnostic script must provide --output")
    output = Path(arguments[arguments.index("--output") + 1]).resolve()
    snapshot = args.snapshot.resolve()
    source = Path(__file__).resolve().parents[2] / "python/sglang"
    prefixes = (
        "multimodal_gen/runtime/models/dits/cosmos_pwm_wam",
        "multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos_pwm_wam",
    )
    if not snapshot.exists():
        snapshot.mkdir(parents=True)
        files = [source / "multimodal_gen/runtime/models/schedulers/base.py"]
        for prefix in prefixes:
            files.extend(sorted((source / prefix).glob("*.py")))
        for path in files:
            target = snapshot / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    files = sorted(snapshot.rglob("*.py"))
    hashes = {
        str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    original_argv = list(sys.argv)
    packages = (
        "sglang",
        "sglang.multimodal_gen",
        "sglang.multimodal_gen.runtime",
        "sglang.multimodal_gen.runtime.models",
        "sglang.multimodal_gen.runtime.models.dits",
        "sglang.multimodal_gen.runtime.models.schedulers",
        "sglang.multimodal_gen.runtime.pipelines_core",
        "sglang.multimodal_gen.runtime.pipelines_core.stages",
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages",
    )
    for name in packages:
        if name in sys.modules:
            raise RuntimeError(f"Diagnostic isolation requires an unimported {name}")
        module = types.ModuleType(name)
        module.__path__ = [str(snapshot.joinpath(*name.split(".")[1:]))]
        sys.modules[name] = module
    manifest = {
        "purpose": "offline same-environment native leaf validation; no serving stack",
        "python": sys.version,
        "executable": sys.executable,
        "command": shlex.join([sys.executable, *original_argv]),
        "cwd": str(Path.cwd()),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH",
                "OMP_NUM_THREADS",
                "PYTHONPATH",
                "PYTHONDONTWRITEBYTECODE",
            )
        },
        "snapshot": str(snapshot),
        "diagnostic_sha256": hashlib.sha256(args.script.read_bytes()).hexdigest(),
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "leaf_sha256": hashes,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "diffusers", "transformers", "msgspec", "safetensors")
        },
        "status": "started",
    }
    try:
        sys.argv = [str(args.script.resolve()), *arguments]
        runpy.run_path(str(args.script.resolve()), run_name="__main__")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["loaded_leaf_modules"] = {
            name: str(module.__file__)
            for name, module in sys.modules.items()
            if name.startswith("sglang.") and getattr(module, "__file__", None)
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "isolation_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
