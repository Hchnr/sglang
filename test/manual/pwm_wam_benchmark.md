`pwm_wam_benchmark.py` is an explicit-asset, offline correctness and latency
comparison. It is outside public CI. Run it only on an assigned, idle GPU.
It never installs dependencies or edits Cosmos / vLLM source checkouts.

The default order is two complete Cosmos/native/native/Cosmos process rounds.
Each process loads once. For each request it records one first request, three
warmups, and eight measured requests. Every output is saved losslessly and
compared, including first and warmup requests. Only measured requests contribute
to the latency summary. First requests are not claimed to have cold disk caches:
Inductor and Triton caches are isolated per implementation but shared across
process rounds. Load time is separate.

The common timed scope starts before request/media preparation and ends after
physical actions and optional uint8 video reach CPU. NPZ / JSON serialization,
validation, and `nvidia-smi` telemetry are outside the timer. The native `python`
route includes worker IPC. The `runtime` route executes the real runtime directly.
The `leaf` route snapshots native source and bypasses unrelated package startup;
it is a diagnostic, not a production deployment measurement.

Use explicit options files. A standard source server baseline is:

```json
{"attention_backend":"flash3","enable_torch_compile":true,"enable_und_cache":false,"reuse_metadata":true}
```

A native configuration with request-local UND reuse is:

```json
{"attention_backend":"flash3","enable_torch_compile":true,"enable_und_cache":true}
```

Do not weaken the baseline to manufacture a speedup. `reuse_metadata` remains
true. If comparing an explicitly supported source UND-cache variant, set
`enable_und_cache` and supply `projected_und_cache_path` to its implementation;
this optional source adapter is hashed and labeled, not silently installed.
Record the source revision and why the selected configuration is appropriate.

For a target-environment comparison, use the same Python executable and installed
Torch / CUDA / cuDNN stack for both implementations. Cosmos may need an external
`PYTHONPATH` overlay of missing dependencies. Keep that overlay separate from
both repositories and their environments. The harness records loaded overlay
Python-file hashes and extension paths/sizes, package versions, source hashes,
checkpoint-config hash, commands, compile settings, and telemetry. A config hash
is not a weight hash; preserve the separately audited artifact provenance.

```bash
python test/manual/pwm_wam_benchmark.py \
  --checkpoint /assets/pwm-wam-hf \
  --vae /assets/Cosmos3-Nano/vae \
  --cosmos-vae /assets/wan22_vae/Wan2.2_VAE.pth \
  --tokenizer /assets/Cosmos3-Nano-VLM \
  --cosmos-repo /repos/cosmos-framework-pwm \
  --cosmos-python /env/sglang/bin/python \
  --native-python /env/sglang/bin/python \
  --cosmos-pythonpath /tmp/cosmos-dependency-overlay:/repos/cosmos-framework-pwm \
  --native-pythonpath /repos/sglang/python \
  --cosmos-options /inputs/cosmos-options.json \
  --native-options /inputs/native-options.json \
  --request /inputs/request.json \
  --native-route python --gpu 2 --output /results/paired
```

Add `--print-plan` to inspect the eight commands without loading CUDA, reading
assets, or creating the output directory. Repeat `--request` for additional
cases. Each request must have an explicit seed; media and prompt files are copied
into the result directory before either implementation runs. Remote inputs must
be downloaded first. Per-request `enable_und_cache` is rejected so it cannot
silently override comparator options. Child processes explicitly clear
`LD_LIBRARY_PATH`; compile failures propagate. The coordinator refuses to start a
block while the assigned GPU has any compute process. Before/after request
telemetry must still be reviewed for external interference during a block.

Two explicit diagnostic compatibility switches are available:

- `--cosmos-transformers5-compat` restores the removed
  `video_processing_qwen3_vl.get_image_size` re-export from
  `transformers.image_utils`. Cosmos imports it through an unused Edge visual
  processor; WAM still executes its original preprocessing and model files.
  The helper and adapter hashes are recorded. This is not a general claim that
  arbitrary Cosmos paths support Transformers 5.
- `--nv-fa3-adapter`, restricted to native `leaf`, routes its private FA3 provider
  boundary to the existing `flash_attn_3_nv` extension. Q/K/V, cumulative lengths,
  scale, causal flag, and `num_splits=1` are unchanged. This allows a same-Cosmos-
  environment leaf diagnostic without installing SGLang kernels. It is not the
  production SGLang FA3 backend.

`--mode run --implementation cosmos|native --options OPTIONS` runs one child
block and is useful for smoke validation before the full process sequence.
`--mode summarize --output /results/paired` rebuilds `comparison.json` from saved
arrays. It reports full-width and active-action errors, video errors, repeat
stability, and all raw timings. It rejects contract mismatches and changed source
files. It does not invent a task-specific action tolerance or treat video quality
as a substitute for action correctness.

Dynamo counters describe the model process for direct/leaf runs. For the Python
worker route they are explicitly labeled parent-only and are not evidence of
worker compilation. Use actual result backend/compile flags, worker logs, source
hashes, and a separate runtime profiler trace when compiled-kernel evidence is
needed. Do not compare parent CUDA allocator statistics to worker memory.
