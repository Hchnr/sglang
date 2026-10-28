# SPDX-License-Identifier: Apache-2.0
"""Report framewise SSIM/PSNR for saved lossless WAM video comparisons.

This measures video quality only. Its fixed thresholds do not establish action
accuracy or cross-environment numerical parity. No resizing or subsampling is
performed, and every frame plus the mean/minimum is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def json_number(value):
    return float(value) if math.isfinite(value) else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import skimage
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    inputs = json.loads(args.comparison.read_text())
    report = {
        "comparison_sha256": hashlib.sha256(args.comparison.read_bytes()).hexdigest(),
        "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "skimage_version": skimage.__version__,
        "native_environment": inputs["native_environment"],
        "reference_environment": inputs["reference_environment"],
        "ssim_parameters": {
            "win_size": 11,
            "gaussian_weights": True,
            "sigma": 1.5,
            "data_range": 255,
            "channel_axis": -1,
            "use_sample_covariance": False,
            "K1": 0.01,
            "K2": 0.03,
            "boundary": "skimage Gaussian filter: mode=reflect, truncate=3.5; spatial mean crops five boundary pixels; RGB channels averaged",
        },
        "psnr_parameters": {
            "data_range": 255,
            "domain": "all RGB pixels in the full frame, no border crop",
        },
        "thresholds": {"ssim": 0.92, "psnr_db": 24.0},
        "scope": "Video quality only; no action tolerance or cross-environment exact-parity claim",
        "rows": [],
    }
    for row in inputs["rows"]:
        path = Path(row["native_directory"]) / "arrays.npz"
        expected_path = Path(row["reference_directory"]) / "video.npy"
        with np.load(path, allow_pickle=False) as arrays:
            actual = arrays["video_frames"]
        expected = np.load(expected_path, allow_pickle=False)
        if (
            actual.shape != expected.shape
            or actual.dtype != np.uint8
            or expected.dtype != np.uint8
        ):
            raise ValueError("Video comparison requires equal uint8 THWC arrays")
        frames = []
        for index, (native_frame, source_frame) in enumerate(
            zip(actual, expected, strict=True)
        ):
            ssim = structural_similarity(
                source_frame,
                native_frame,
                win_size=11,
                gaussian_weights=True,
                sigma=1.5,
                data_range=255,
                channel_axis=-1,
                use_sample_covariance=False,
                K1=0.01,
                K2=0.03,
            )
            psnr = peak_signal_noise_ratio(source_frame, native_frame, data_range=255)
            frames.append({"frame": index, "ssim": float(ssim), "psnr_db": float(psnr)})
        scores = {
            key: np.asarray([frame[key] for frame in frames])
            for key in ("ssim", "psnr_db")
        }
        result = {
            "name": row["name"],
            "shape": list(actual.shape),
            "native_directory": row["native_directory"],
            "reference_directory": row["reference_directory"],
            "frames": [
                {
                    key: json_number(value) if key != "frame" else value
                    for key, value in frame.items()
                }
                for frame in frames
            ],
            "summary": {
                key: {
                    "mean": json_number(values.mean()),
                    "min": json_number(values.min()),
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
        report["rows"].append(result)
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "frames"}
            ),
            flush=True,
        )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
