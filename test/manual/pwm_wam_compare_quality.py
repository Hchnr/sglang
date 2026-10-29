# SPDX-License-Identifier: Apache-2.0
"""Compare native compiled WAM with source compiled/eager lossless outputs.

This is a CPU-only diagnostic for explicit private-asset runs. It compares three
pairs per case, retains every video frame's fixed SSIM/PSNR scores, and reports
action errors without inventing an application-specific action tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from test_cosmos_pwm_wam import array_metrics, compare_reference, validate_result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_result(path):
    with np.load(path / "arrays.npz", allow_pickle=False) as arrays:
        result = {
            **json.loads((path / "result.json").read_text()),
            **{key: arrays[key] for key in arrays.files},
        }
    validate_result(result)
    return result


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def video_quality(actual, expected):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    if (
        actual.dtype != np.uint8
        or expected.dtype != np.uint8
        or actual.shape != expected.shape
    ):
        raise ValueError("Video comparison requires equal uint8 THWC arrays")
    frames = []
    for index, (target, reference) in enumerate(zip(actual, expected, strict=True)):
        # Exact frames avoid redundant filtering; these values are mathematical
        # identities, including the infinite PSNR for zero mean squared error.
        exact = np.array_equal(target, reference)
        ssim = (
            1.0
            if exact
            else structural_similarity(
                reference,
                target,
                win_size=11,
                gaussian_weights=True,
                sigma=1.5,
                data_range=255,
                channel_axis=-1,
                use_sample_covariance=False,
                K1=0.01,
                K2=0.03,
            )
        )
        psnr = (
            float("inf")
            if exact
            else peak_signal_noise_ratio(reference, target, data_range=255)
        )
        frames.append({"frame": index, "ssim": float(ssim), "psnr_db": float(psnr)})
    scores = {
        key: np.asarray([frame[key] for frame in frames]) for key in ("ssim", "psnr_db")
    }
    psnr = scores["psnr_db"]
    finite_psnr = psnr[np.isfinite(psnr)]
    relative_mse = np.power(10.0, -psnr / 10.0).mean()
    return {
        "frames": frames,
        "global_psnr_db": float(-10.0 * np.log10(relative_mse))
        if relative_mse
        else float("inf"),
        "finite_frame_psnr_mean": float(finite_psnr.mean())
        if len(finite_psnr)
        else float("inf"),
        "perfect_frame_count": int(np.isposinf(psnr).sum()),
        "summary": {
            key: {
                "mean": float(values.mean()),
                "min": float(values.min()),
                "worst_frame": int(values.argmin()),
            }
            for key, values in scores.items()
        },
        "all_frames_pass": bool(
            (scores["ssim"] >= 0.92).all() and (scores["psnr_db"] >= 24).all()
        ),
        "mean_pass": bool(
            scores["ssim"].mean() >= 0.92 and scores["psnr_db"].mean() >= 24
        ),
    }


def compare(actual, expected, reference_path):
    metrics = compare_reference(actual, reference_path)
    metrics["full_width"] = {
        key: array_metrics(actual[key], expected[key])
        for key in ("actions", "normalized_actions")
    }
    metrics["video_quality"] = video_quality(
        actual["video_frames"], expected["video_frames"]
    )
    return metrics


def source_cases(path):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete" or not manifest["leaf_files_unchanged"]:
        raise ValueError(f"Incomplete or changed-source run: {path}")
    rows = {}
    for row in manifest["samples"]:
        rows.setdefault(row["case"], []).append(row)
    return manifest, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--native-stride", type=int, default=4)
    parser.add_argument("--source-compiled", type=Path, required=True)
    parser.add_argument("--source-eager", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import skimage

    compiled_manifest, compiled = source_cases(args.source_compiled)
    eager_manifest, eager = source_cases(args.source_eager)
    if compiled.keys() != eager.keys() or args.native_stride < 1:
        raise ValueError(
            "Source case sets must match and native stride must be positive"
        )
    expected_options = {
        "attention_backend": "flash3",
        "enable_und_cache": False,
        "reuse_metadata": True,
    }
    for manifest, enabled in ((compiled_manifest, True), (eager_manifest, False)):
        if manifest["options"] != {**expected_options, "enable_torch_compile": enabled}:
            raise ValueError(
                "Unexpected source variant: isolate only the compile setting"
            )
    for key in (
        "leaf_sha256",
        "checkpoint_config_sha256",
        "versions",
        "python",
        "torch_cuda",
        "cudnn",
        "harness_sha256",
        "numerics",
    ):
        if compiled_manifest[key] != eager_manifest[key]:
            raise ValueError(f"Source variants differ beyond compile: {key}")
    report = {
        "diagnostic_sha256": digest(__file__),
        "source_manifest_sha256": {
            "compiled": digest(args.source_compiled / "manifest.json"),
            "eager": digest(args.source_eager / "manifest.json"),
        },
        "native_directory": str(args.native_directory.resolve()),
        "native_stride": args.native_stride,
        "skimage_version": skimage.__version__,
        "source_versions": compiled_manifest["versions"],
        "source_numerics": compiled_manifest["numerics"],
        "ssim_parameters": {
            "win_size": 11,
            "gaussian_weights": True,
            "sigma": 1.5,
            "data_range": 255,
            "channel_axis": -1,
            "use_sample_covariance": False,
            "K1": 0.01,
            "K2": 0.03,
            "boundary": "reflect filtering, truncate=3.5; SSIM averages exclude five boundary pixels; RGB channel mean",
        },
        "psnr_parameters": {
            "data_range": 255,
            "domain": "all RGB pixels; no border crop",
        },
        "video_thresholds": {"ssim": 0.92, "psnr_db": 24.0},
        "scope": "Video thresholds do not establish action accuracy. Source compile drift is a comparator, not an acceptance tolerance. Timings are not performance evidence.",
        "rows": [],
    }
    for case in sorted(compiled):
        cpath = Path(compiled[case][0]["directory"])
        epath = Path(eager[case][0]["directory"])
        npath = args.native_directory / f"{case * args.native_stride:03d}"
        if compiled[case][0]["request_sha256"] != eager[case][0]["request_sha256"]:
            raise ValueError(f"Source variants used different requests: {case}")
        source_compiled, source_eager, native = (
            load_result(path) for path in (cpath, epath, npath)
        )
        parameters = native["parameters"]
        required = {
            "attention_backend": "flash3",
            "enable_torch_compile": True,
            "fp32_matmul_allow_tf32": False,
            "fp32_cudnn_allow_tf32": False,
        }
        if any(parameters.get(key) != value for key, value in required.items()):
            raise ValueError(
                f"Native output lacks the intended IEEE compiled policy: {case}"
            )
        row = {
            "case": case,
            "request": compiled[case][0]["request"],
            "directories": {
                "native": str(npath),
                "source_compiled": str(cpath),
                "source_eager": str(epath),
            },
            "native_parameters": parameters,
            "source_repeat_stability": {},
        }
        for label, samples, first in (
            ("compiled", compiled[case], cpath),
            ("eager", eager[case], epath),
        ):
            row["source_repeat_stability"][label] = [
                compare_reference(load_result(Path(sample["directory"])), first)
                for sample in samples
            ]
        for label, actual, expected, path in (
            ("native_vs_source_compiled", native, source_compiled, cpath),
            ("source_compiled_vs_source_eager", source_compiled, source_eager, epath),
            ("native_vs_source_eager", native, source_eager, epath),
        ):
            row[label] = compare(actual, expected, path)
            print(
                json.dumps(
                    {
                        "case": case,
                        "pair": label,
                        "active_action": row[label]["actions"],
                        "video_summary": finite_json(
                            row[label]["video_quality"]["summary"]
                        ),
                    }
                ),
                flush=True,
            )
        report["rows"].append(row)
        args.output.write_text(
            json.dumps(finite_json(report), indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
