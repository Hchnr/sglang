# PWM WAM numerical validation notes, 2026-10-28

The saved Cosmos references and the frozen native WAM leaf implementation were evaluated with the same Python 3.13 / Torch 2.10.0+cu130 / cuDNN 9.15.1 environment. All six original-pixel scenarios produce exactly equal full-128 physical actions, normalized actions, and lossless uint8 video. No reference latent substitution was used, and UND cache was disabled. This separates the previously observed cross-environment VAE convolution drift from the native model port. It does not claim cross-environment bitwise equality. The reference checkout advanced during the investigation; the revision audit below distinguishes the saved matrix from the fresh smoke capture.

Evidence paths below are relative to `artifacts/pwm_wam_1028/same_environment/` unless otherwise specified. Private assets and generated tensor files are not committed.

## Isolation and provenance

- The Cosmos checkout and its environment were not modified; its tracked git status remains clean.
- The only missing native dependency, msgspec 0.22.0 for CPython 3.13, was installed into `/tmp/sglang-pwm-cosmos313-overlay`, selected through `PYTHONPATH`.
- `test/manual/pwm_wam_isolated_native.py` creates namespace packages that skip unrelated SGLang serving-package initializers. All actual request, model, VAE, scheduler and runtime leaf modules are unmodified byte copies in `native_leaf_snapshot/`. This is an offline correctness experiment, not a deployment configuration or serving-performance measurement.
- `native_matrix_gpu1/isolation_manifest.json` records the complete command, environment, Python/library versions, every leaf SHA256 and the exact loaded module paths. `native_matrix_gpu1_command.json` records the outer invocation; `native_matrix_gpu1_gpu.jsonl` records GPU process occupancy.
- GPU 0 was initially assigned, but an external process appeared during imports. The first attempt failed loading with OOM; a later guarded retry terminated only its own child when another GPU process appeared. Those failed attempts remain separate. GPU 1 was subsequently assigned and was empty before both successful runs. Successful GPU-1 process audits contain only the diagnostic process; GPU 1 was released after each run.

## Reference revision audit

The initial repository freeze was `3d2aba0fe5e293f505d7ec52cb6b2eba8cc95d06`. The early intermediate capture explicitly records that revision. The later fresh `reference_smoke/manifest.json` explicitly records `42bea124d7fd19c949b6047a49fa80cb0746ec36`, with a clean tracked worktree. Neither revision should be described as the execution revision of every saved reference.

The six earlier JSONL worker outputs record Torch and precision, but do not contain a per-run source commit. Their global provenance names the initial freeze. The following HEAD values are reconstructed from each stderr log's first model timestamp and the source checkout's reflog, not independent attestations of the revision at Python import time. All timestamps are on 2026-10-28, UTC+08:00.

| Saved reference | First model log | Reconstructed HEAD at that time |
| --- | --- | --- |
| smoke | 16:53:44 | `3d2aba0fe5e293f505d7ec52cb6b2eba8cc95d06` |
| Libero full | 17:25:53 | `3fcce4065d91d58e3bf94f531397195d95449a43` |
| DROID independent rates/CFG | 17:26:54 | `d9626e9791238de2f64b1525b0dd9751e7d83cb6` |
| RoboTwin joint | 17:27:54 | `d9626e9791238de2f64b1525b0dd9751e7d83cb6` |
| RoboCasa single | 17:28:36 | `d9626e9791238de2f64b1525b0dd9751e7d83cb6` |
| default 30 steps | 17:44:02 | `d0b77c5e3239c3405db1607881d719932cbb53d5` |

Libero's response was saved at 17:26:29, after HEAD advanced to `d9626e9` at 17:26:12. Exact per-run import revisions cannot be recovered from these older outputs. `source_revision_audit.json` retains the timestamps, response hashes, full revision IDs, changed-path list and this limitation.

The audit compares the initial freeze with the fresh smoke revision. All files under `cosmos_framework/model`, `data`, `configs` and `utils`, plus the inference entrypoint, argument schemas, common inference helpers, defaults and `action_serving.py`, are byte-identical; none of the 12 intervening commits touches these scopes. The artifact audit records matching git blob IDs and SHA256 for 20 concrete model, request, packing, normalization, VAE and solver files. The reference worker's current SHA256 also matches its initial frozen hash. The 23 changed paths comprise the serving graph runner/server, new opt-in VIDEO sparse-attention/FFN-cache implementations, and diagnostic tools/tests. Both reference entrypoints directly call `OmniInference.create`, `prepare_wam_request` and `generate_samples_from_batch`; they do not use `action_server` or install its graph/sparse/FFN runners. The fresh capture explicitly selects SDPA with compile, graphs and UND reuse disabled.

Consequently, this revision drift does not change the numerical path exercised by the six saved references, and does not itself require rerunning the six cases. This is a source-code equivalence assessment for the exercised path, not a claim that the two full repositories or their optimized serving modes are identical. The source checkout and environment remain unmodified by this investigation.

## Six original-pixel scenarios

`native_matrix_gpu1/comparison.json` contains final-output metrics for smoke, default 30 steps, DROID independent frame rates with CFG, Libero full resolution, RoboCasa single camera, and RoboTwin joint actions. Each physical-action, normalized-action, and video comparison is exact with maximum absolute error zero. The original comparison checks active action slots; `native_matrix_gpu1/full_width_comparison.json` independently verifies all 128 slots and confirms the reference inactive coordinates are zero. Each case retains `arrays.npz` and `result.json`.

## Fresh smoke trajectory

`reference_smoke/` is a fresh Cosmos `42bea124` capture with every UniPC step and decoded video, from the original request pixels. `native_smoke/comparison.json` matches the recorded source backend flags and compares 52 tensor boundaries under the same environment. Pixels, token IDs, clean VAE latents, both rotary embeddings, all three tower states at layers 0 and 35, every step's guided velocity, final actions and decoded video are exact.

49 of 52 entire tensors are exact. The only remaining differences are 812 fixed conditioning-frame coordinates in step 2 input/result and final video latents, with maximum absolute difference 1.1920928955078125e-7. `native_smoke/roundoff_location.json` shows zero differing generated coordinates. `native_smoke/fixed_coordinate_check.json` confirms the native conditioning frame equals its clean reference exactly, while the source solver retains tiny FP32 roundoff there. This follows the native contract that explicitly restores fixed coordinates after solver updates. Every generated video coordinate, action coordinate, and final uint8 pixel remains exact. No threshold was used to relabel these three nonexact tensors.

The fresh source command and numerical flags are in `reference_smoke/manifest.json`; native command, environment, source hashes and matching flags are in `native_smoke/isolation_manifest.json` and `native_smoke/comparison.json`. `reference_smoke_command.json` and `native_smoke_command.json` preserve launch arguments.

## Cross-environment video quality

`../final_video_quality.json` separately compares the final deployed Torch 2.14 / cuDNN 9.24 videos to the saved Cosmos Torch 2.10 videos audited above. It retains every frame, mean, minimum and worst-frame index for six scenarios plus a repeated Libero request. SSIM uses RGB channel means, Gaussian window 11, sigma 1.5, population covariance and data range 255; skimage uses reflect filtering and excludes the five-pixel boundary when averaging SSIM. PSNR uses every RGB pixel of the full frame and data range 255. No resizing or subsampling is performed.

All frames pass the fixed video-only gates SSIM >= 0.92 and PSNR >= 24 dB. The global worst frame is DROID frame 32: SSIM 0.9327855989185974, PSNR 24.912618667486427 dB. This video-quality result establishes no action tolerance and is not a claim of cross-environment exact parity. The reproducible CPU command is:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 .venv/bin/python test/manual/pwm_wam_video_quality.py --comparison artifacts/pwm_wam_1028/final_matrix_comparison.json --output artifacts/pwm_wam_1028/final_video_quality.json
```

## Target-environment UND-cache matrix

The actual SGLang runtime in Torch 2.14.1 / Diffusers 0.37.0 / Transformers 5.19.0 was also tested on the same six unique requests, with cache disabled/enabled/enabled/disabled for each case. The existing `test/manual/test_cosmos_pwm_wam.py` harness used `--route runtime --und-cache compare --warmup 1`. One untimed warmup per mode preceded 24 saved rollouts. All 54 comparisons of full-128 physical actions, normalized actions and uint8 video were exact, with maximum absolute difference zero. These cases exercise CFG=1, conditional/unconditional CFG, independent action/video rates, several action domains, 30 default solver steps and video output. Cache remains disabled by default.

Evidence is in `../und_cache_matrix/summary.json`, `../und_cache_matrix/acceptance.json`, and result directories `000` through `023`. `../und_cache_matrix_command.json` preserves every argument and environment setting. `../und_cache_matrix_provenance.json` and `../und_cache_matrix_diff.patch` record the HEAD, working-tree diff and leaf-file hashes; every leaf hash was unchanged after the run. The GPU-1 audit contains only the diagnostic process, and GPU 1 was released afterward. This matrix establishes output parity; its single-pass timing data is not a standalone performance claim.
