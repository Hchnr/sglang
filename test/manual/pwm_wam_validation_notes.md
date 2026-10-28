# PWM WAM numerical validation notes, 2026-10-28

The current Cosmos reference and the frozen native WAM leaf implementation were evaluated with the same Python 3.13 / Torch 2.10.0+cu130 / cuDNN 9.15.1 environment. All six original-pixel scenarios produce exactly equal full-128 physical actions, normalized actions, and lossless uint8 video. No reference latent substitution was used, and UND cache was disabled. This separates the previously observed cross-environment VAE convolution drift from the native model port. It does not claim cross-environment bitwise equality.

Evidence paths below are relative to `artifacts/pwm_wam_1028/same_environment/` unless otherwise specified. Private assets and generated tensor files are not committed.

## Isolation and provenance

- The Cosmos checkout and its environment were not modified; its tracked git status remains clean.
- The only missing native dependency, msgspec 0.22.0 for CPython 3.13, was installed into `/tmp/sglang-pwm-cosmos313-overlay`, selected through `PYTHONPATH`.
- `test/manual/pwm_wam_isolated_native.py` creates namespace packages that skip unrelated SGLang serving-package initializers. All actual request, model, VAE, scheduler and runtime leaf modules are unmodified byte copies in `native_leaf_snapshot/`. This is an offline correctness experiment, not a deployment configuration or serving-performance measurement.
- `native_matrix_gpu1/isolation_manifest.json` records the complete command, environment, Python/library versions, every leaf SHA256 and the exact loaded module paths. `native_matrix_gpu1_command.json` records the outer invocation; `native_matrix_gpu1_gpu.jsonl` records GPU process occupancy.
- GPU 0 was initially assigned, but an external process appeared during imports. The first attempt failed loading with OOM; a later guarded retry terminated only its own child when another GPU process appeared. Those failed attempts remain separate. GPU 1 was subsequently assigned and was empty before both successful runs. Successful GPU-1 process audits contain only the diagnostic process; GPU 1 was released after each run.

## Six original-pixel scenarios

`native_matrix_gpu1/comparison.json` contains final-output metrics for smoke, default 30 steps, DROID independent frame rates with CFG, Libero full resolution, RoboCasa single camera, and RoboTwin joint actions. Each physical-action, normalized-action, and video comparison is exact with maximum absolute error zero. The original comparison checks active action slots; `native_matrix_gpu1/full_width_comparison.json` independently verifies all 128 slots and confirms the reference inactive coordinates are zero. Each case retains `arrays.npz` and `result.json`.

## Fresh smoke trajectory

`reference_smoke/` is a new current-Cosmos capture with every UniPC step and decoded video, from the original request pixels. `native_smoke/comparison.json` matches the recorded source backend flags and compares 52 tensor boundaries under the same environment. Pixels, token IDs, clean VAE latents, both rotary embeddings, all three tower states at layers 0 and 35, every step's guided velocity, final actions and decoded video are exact.

49 of 52 entire tensors are exact. The only remaining differences are 812 fixed conditioning-frame coordinates in step 2 input/result and final video latents, with maximum absolute difference 1.1920928955078125e-7. `native_smoke/roundoff_location.json` shows zero differing generated coordinates. `native_smoke/fixed_coordinate_check.json` confirms the native conditioning frame equals its clean reference exactly, while the source solver retains tiny FP32 roundoff there. This follows the native contract that explicitly restores fixed coordinates after solver updates. Every generated video coordinate, action coordinate, and final uint8 pixel remains exact. No threshold was used to relabel these three nonexact tensors.

The fresh source command and numerical flags are in `reference_smoke/manifest.json`; native command, environment, source hashes and matching flags are in `native_smoke/isolation_manifest.json` and `native_smoke/comparison.json`. `reference_smoke_command.json` and `native_smoke_command.json` preserve launch arguments.

## Cross-environment video quality

`../final_video_quality.json` separately compares the final deployed Torch 2.14 / cuDNN 9.24 videos to current-Cosmos Torch 2.10 videos. It retains every frame, mean, minimum and worst-frame index for six scenarios plus a repeated Libero request. SSIM uses RGB channel means, Gaussian window 11, sigma 1.5, population covariance and data range 255; skimage uses reflect filtering and excludes the five-pixel boundary when averaging SSIM. PSNR uses every RGB pixel of the full frame and data range 255. No resizing or subsampling is performed.

All frames pass the fixed video-only gates SSIM >= 0.92 and PSNR >= 24 dB. The global worst frame is DROID frame 32: SSIM 0.9327855989185974, PSNR 24.912618667486427 dB. This video-quality result establishes no action tolerance and is not a claim of cross-environment exact parity. The reproducible CPU command is:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 .venv/bin/python test/manual/pwm_wam_video_quality.py --comparison artifacts/pwm_wam_1028/final_matrix_comparison.json --output artifacts/pwm_wam_1028/final_video_quality.json
```
