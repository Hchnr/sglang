# 三塔 PWM WAM 接入 SGLang：结论

更新（18:55）：同环境六场景原像素对照已完成，动作和视频全部逐位一致；请求内 UND 缓存的无视频 ABBA 实测平均延迟下降 3.15%。缓存开启后的多场景视频、服务验证及显存优化仍在进行。

原生接入与功能验证已完成：真实 checkpoint 由 SGLang worker 直接执行 UND、VIDEO、ACTION 三塔及联合 UniPC，Python、原生 action HTTP 和 Cosmos `/generate` 兼容入口均可交付动作及可选视频。

**同环境原生移植对齐通过，跨环境仍存在已定位的 VAE BF16 卷积差异。** 目标环境六种请求的 active physical action relative L2 为 0.007525–0.016698，视频每帧均通过 SSIM≥0.92、PSNR≥24 dB。未放宽动作阈值或替换生产输入，这些误差不能转为机器人任务成功率结论。当前 eager 接入和 UND 缓存均未超过已有 Cosmos FA3+compile 的性能。

目标：单 GPU/B1 的原生 SGLang WAM 推理、Action128 动作及可选未来视频、可复现的正确性和性能基线。实施记录见 [record_1028.md](record_1028.md)，设计见 [plan.md](plan.md)。

## 当前已确认

- 接入位置为主仓库 `sglang.multimodal_gen`，无需独立 Omni 引擎。
- 已适配固定 HF artifact、原生三塔、SGLang worker/API，修正与当前 Cosmos 的过程 mask、媒体和 seed 差异；复用代码保留来源归属。
- 已实现 `CosmosPWMWAMPipeline`，在 SGLang worker 内直接执行三塔和联合采样。生产代码无 Cosmos/vLLM import 或外部推理进程转发。
- Python `generate_action()`、原生 `/v1/actions/generations`、独立 Cosmos `/generate` facade 均已真实跑通，动作与可选视频可以交付。
- 隔离环境采用仓库目标 Torch 2.14.1、Transformers 5.19.0、Diffusers 0.37.0；source reference 为当前 Cosmos `3d2aba0fe5e293f505d7ec52cb6b2eba8cc95d06`、Torch 2.10.0/cuDNN 9.15.1.9。目标 cuDNN 为 9.24.0.43。
- 7 个分片、1,214 个模型 tensor 严格加载；tensor bytes 32,364,191,504，分片文件 bytes 32,364,372,752，已保存全部 SHA-256。
- 使用固定 `../vllm-omni-pwm/artifacts/pwm-wam-hf`（来源 iter_000008600），VAE `/share/project/eai_pwm/models/Cosmos3-Nano/vae`，tokenizer `/share/project/eai_pwm/models/Cosmos3-Nano-VLM`。未重新导出、覆盖资产或选择训练 latest。

Review 可从 [pipeline](../python/sglang/multimodal_gen/runtime/pipelines/cosmos_pwm_wam.py)、[三塔核心](../python/sglang/multimodal_gen/runtime/models/dits/cosmos_pwm_wam/transformer.py)、[请求合同](../python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos_pwm_wam/request.py)、[联合 runtime](../python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos_pwm_wam/runtime.py)、[HTTP adapter](../python/sglang/multimodal_gen/runtime/entrypoints/action/cosmos_pwm_wam.py) 和 [compat facade](../python/sglang/multimodal_gen/runtime/entrypoints/action/pwm_compat.py) 开始。

## 验收状态

| 项目 | 状态 |
| --- | --- |
| 严格权重加载与合同验证 | 实权重成功；shape/dtype/索引/合同/资产/registry 负例已测试 |
| 原生三塔和联合 UniPC | 同环境六场景原始像素输入：全宽动作、normalized actions、视频逐位一致；fresh smoke 生成坐标全轨迹一致，固定条件帧末位差异见下文 |
| 新增 CPU 测试 | 38/38：9 model +17 runtime +12 serving |
| 既有回归 | Pi05 API 10、Cosmos3 相关 18 +4 subtests、registry 11 均通过 |
| 目标环境对当前 Cosmos 原始像素 e2e | 七请求/六场景完成，shape/离散合同一致、inactive 槽位零、输出有限；每帧视频质量通过，跨环境 strict bitwise 不成立 |
| SGLang worker / Python / HTTP | 最终三入口动作逐值一致，13 项 metadata/资产摘要相同；原生与 facade MP4 逐字节一致（86,268 bytes） |
| 错误与生命周期 | 原生 400/facade 422 后恢复；四并发 1×200+3×503、Retry-After=1 后恢复；下载/删除/404、SIGTERM 文件和目录清理通过 |
| 状态隔离 | Libero → 多 domain/shape/seed → Libero，动作/normalized actions/视频首尾逐元素一致 |
| 性能与显存基线 | 同卡重复 Cosmos SDPA /FA3+compile /SGLang Python 完成，见下文 |
| 部署文档与可复现命令 | [部署说明](../examples/diffusion/cosmos_pwm_wam/README.md) 已提供并实际启动验证 |
| 静态检查 | 22 份新 Python 文件 Ruff、format、isort 通过，`git diff --check` 干净 |

三份单元测试在 CPU 张量上执行，但依赖 Diffusers。既有 hosted CPU CI 只安装 `python[dev]`，因此注册到已有安装 diffusion 依赖的 `base-b-test-diffusion-unit-1-gpu-h100`。CI discovery/validation/filter 已确认三文件均启用、无 skip；没有增加全局 CPU 依赖。私有 checkpoint 不加入公共 CI。

## 数值结论与边界

当前 Cosmos 和 native leaf snapshot 在相同 Python 3.13 / Torch 2.10 / cuDNN 9.15 下，六种原始像素输入的全宽 Action128、normalized actions、无损 uint8 视频全部逐位一致，max abs=0。该实验没有注入 reference latent。诊断加载器只跳过无关 serving 包初始化，实际计算文件是记录 SHA-256 的原生 snapshot；它证明移植语义，不替代目标 Torch 2.14 的 SGLang worker 验证。

另外重新捕获的 smoke 逐步对照有 52 个边界、49 个整张量 exact；另三个张量只在固定条件帧的 812 个坐标上有 max abs `1.1920929e-7`，**生成视频和动作坐标的差异为零**。native 每步恢复固定坐标为 clean，而 source solver 在这些坐标保留 FP32 舍入；不把这三张量标为 exact。完整方法和证据索引见 [数值验证说明](../test/manual/pwm_wam_validation_notes.md)。

已修复 SGLang 平台导入禁用 cuDNN SDPA 带来的模型偏差：仅在 WAM forward 中启用 source 使用的 SDPA 后端集合，正常/异常退出均恢复原状态。

冻结输入/权重的 **layer 0 独立回放** 在 source 与目标 Torch 环境均匹配 source。**3-step smoke 注入 reference clean latent** 后，首步 velocity、layer 0/35 的三塔输入/输出和最终 physical/normalized actions 逐元素一致，video latent relative L2 为 1.49e-8、max abs 为 1.192e-7；该完整采样输出的 decoded uint8 max abs 3、MAE 0.07887。这是隔离模型/solver 的证据，**不等于原始像素端到端通过，也不能推广为整个请求矩阵的隔离验证结果**。

VAE 首差异出现在同值、同 shape/stride/NCDHW layout 的 BF16 residual conv1，首卷积、norm 和 SiLU 已精确一致。该卷积 283 个元素不同、max abs 0.0078125；target clean latent max abs 0.0234375、relative L2 0.004737。严格相同 source latent 的纯 VAE decoder 回放，BF16 输出 max abs 0.021484375、relative L2 0.00260766。多种局部 benchmark/deterministic/TF32/layout 设置不能消除差异，因此没有强塞旧 cuDNN 或放宽阈值。

最终原像素对照如下；动作仅统计 active 槽位，relative L2 以 float64 计算 `||actual-reference|| / ||reference||`，视频误差来自无损 uint8 RGB、单位为 0–255 像素值。不同槽位/领域的物理单位不同，max abs 不是通用任务容差。

| 请求 | 动作 shape | physical relL2 / max abs | normalized relL2 / max abs | 视频 MAE / max abs |
| --- | --- | --- | --- | --- |
| smoke：双相机，3 steps | 4×128 | 0.007525 / 0.008380 | 0.008812 / 0.011094 | 1.32025 / 119 |
| Libero：960×640，2 秒 | 40×128 | 0.008457 / 0.016636 | 0.010640 / 0.026497 | 0.58901 / 132 |
| Droid：三相机，50 Hz /15 FPS，CFG norm | 100×128 | 0.011079 / 0.029596 | 0.014091 / 0.030003 | 2.23442 / 231 |
| Robotwin：joint 非连续 mask | 4×128 | 0.015039 / 0.029537 | 0.012985 / 0.015975 | 0.58108 / 58 |
| Robocasa：单相机 | 8×128 | 0.016698 / 0.031995 | 0.023762 / 0.035519 | 0.93525 / 135 |
| 缺省 30 steps /guidance 1 /shift 10 | 40×128 | 0.013101 / 0.034328 | 0.013450 / 0.034328 | 1.02238 / 198 |

完整 MAE/RMSE、shape、合同与重放结果见 [final_matrix_comparison.json](../artifacts/pwm_wam_1028/final_matrix_comparison.json)，定位命令见 [numerical_diagnosis.md](../artifacts/pwm_wam_1028/numerical_diagnosis.md)。跨环境视频的所有帧均满足 SSIM≥0.92、PSNR≥24 dB；最差 DROID frame32 为 SSIM 0.9327856、PSNR 24.912619 dB，参数及逐帧数值见 [final_video_quality.json](../artifacts/pwm_wam_1028/final_video_quality.json)。跨环境严格原像素一致、业务动作容差和机器人闭环质量没有据此宣称通过。

## 性能与显存

同一 GPU2/H100 80GB、Libero 960×640、40 actions /41 video frames、5 steps、guidance 1.5、shift 5、seed 42、`return_video=false`。各路径顺序运行六次，首轮单列，warm median 取后五次。source metadata reuse 开启、UND cache 关闭；SGLang 为 eager SDPA。计时包含请求前处理/联合推理/动作后处理，不含模型加载、VAE decode、MP4 编码或 HTTP；Python 路径还包含 worker 通信。

| 路径 | 首请求 s | warm median s | 计时边界 |
| --- | --- | --- | --- |
| 当前 Cosmos SDPA | 4.512 | 3.430 | reference worker 内完整请求 |
| 当前 Cosmos FA3 + compile | 21.835 | 2.233 | reference worker 内完整请求，首轮含编译 |
| SGLang Python / worker | 4.443 | 3.239 | `generate_action()`；其中 worker median 3.230、sample median 3.195 |
| 同环境 SGLang direct runtime | 4.418 | 3.228 | 相同模型/采样实现，直接调用 runtime |

初始 SGLang eager 请求比已有 FA3+compile baseline 约慢 **45%**。相较 source SDPA 的差值不能单独归因于框架，因为 Torch/cuDNN 和计算路径不同。FA3、compile 尚未迁入；独立 UND 缓存收益见下文。

新增请求内 UND 缓存采用同 GPU2 的三组 ABBA，固定上述 workload、各模式预热一次，正式样本每种 6 次。未缓存平均 **3.257093 s**，缓存平均 **3.154454 s**，平均延迟下降 **3.1512%**；两者 median 分别为 3.244553 /3.154283 s。12 次动作/normalized actions 全部逐元素一致；另两次 profiler 输出也一致，profile 耗时不计入统计。缓存额外峰值 allocated 为 **36,954,112 bytes**。监督日志确认测试 GPU 没有外部并发，命令、输出、trace 与汇总见 [und_cache_validation/comparison.json](../artifacts/pwm_wam_1028/und_cache_validation/comparison.json)。这组测量基于 `4736596b7c`，不包含后续 CPU 文本头显存优化，也不表示超过 Cosmos FA3+compile。

缓存 trace 的 self CUDA 时间主要为 GEMM 38.16%、cuDNN attention 28.08%，其余为逐元素、cast、norm 等。单 trace 未建立 overlap 收益；自动分析器基于 `nvjet` 名称给出的 FP8 提示不适用于本 BF16 模型。已保留未缓存 trace、缓存 trace、算子表和三表分析，后续 FA3/compile/norm/RoPE 优化应逐项验证舍入、完整动作与视频，不能直接套用其他模型的融合。

这是共享节点上的小样本基线，没有锁定整机独占资源，也没有置信区间/统计显著性结论。收尾时其他 GPU 作业已出现，未操作这些外部进程。

同环境 direct runtime 与 Python/worker 的动作逐元素一致，六次重复也一致。两条路径 warm median 相差约 11.2 ms；Python 同请求的调用耗时减 worker 耗时，中位数约 10.1 ms。direct runtime 峰值 allocated **34,770,594,816 bytes（34.77 GB /32.38 GiB）**、reserved 35,383,148,544 bytes；这是上述无解码 workload 的 allocator 指标，不含 CUDA context/驱动开销，也不代表最大视频 shape。Python 父进程 `parent_peak_allocated_bytes=0` 不代表 worker 不占显存。完整六次结果见 [benchmark/summary.json](../artifacts/pwm_wam_1028/benchmark/summary.json)。

原尺寸带视频矩阵重放 Python e2e 5.001 s、worker 4.618 s，其中 sample 3.181 s、finalize/decode 1.402 s。HTTP 小 smoke 的带视频单次请求为 native 0.433 s、facade 0.468 s，包含 MP4 编码及响应传输；未单独测量 MP4 编码，也未做 HTTP 压测。这些都不是无视频 benchmark 的同口径样本，不据此比较框架速度。

共享 NFS/import/spawn 使初始化波动明显：最终矩阵 Python 初始化 103.3 s、benchmark 初始化 181.4 s；不等于纯权重加载时间。最终 direct runtime 初始化 14.43 s；早期小 shape smoke 加载 11.23 s、峰值 allocated 34.36 GB，不外推最大画布容量。

## 范围、环境与复现

暂不支持其他 action 模式、state/action 条件输入、多候选、动态 batching、TP/SP/DP/CFG 并行、offload、量化、LoRA、compile、CUDA graph、prefix cache、realtime/OpenPI WebSocket；不支持的控制明确拒绝。上述优化属于原计划 P6，不计入首版完成范围。

本仓库 `.venv` 安装目标核心包，并通过 `pwm_validation_base.pth` 只读追加相邻环境的第三方 site-packages（如 msgspec、FastAPI、uvicorn）。没有追加 Cosmos/vLLM 仓库源码或修改其环境。**这次不是干净容器安装验证**；主要版本/路径见 `environment.json`，完整 294 项首选 distribution 见 `environment_all_packages.json`。命令清空 `LD_LIBRARY_PATH` 避免宿主 Torch 库混入。服务返回的九份资产摘要只覆盖 VAE 配置/tokenizer 文件，不包含模型/VAE 权重。

CPU 回归命令：

```bash
env -u CUDA_VISIBLE_DEVICES LD_LIBRARY_PATH='' PYTHONPATH=python \
  .venv/bin/python -m unittest discover \
  -s test/registered/unit/multimodal_gen -p 'test_cosmos_pwm_wam_*.py' -v
```

实权重 Python benchmark（输出目录须尚不存在）：

```bash
env LD_LIBRARY_PATH='' PYTHONPATH=python CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 \
  .venv/bin/python test/manual/test_cosmos_pwm_wam.py \
  --route python --checkpoint ../vllm-omni-pwm/artifacts/pwm-wam-hf \
  --vae /share/project/eai_pwm/models/Cosmos3-Nano/vae \
  --tokenizer /share/project/eai_pwm/models/Cosmos3-Nano-VLM \
  --request artifacts/pwm_wam_1028/benchmark/request.json --repeat 6 \
  --output artifacts/pwm_wam_1028/review_benchmark
```

重复 `--request` 可跑多场景；`--route runtime` 测直接调用与实际 GPU allocator 统计；`--reference` 比较已有输出。HTTP 启动脚本、测试脚本及关闭检查见 [HTTP 复现说明](../artifacts/pwm_wam_1028/http_validation/REPRODUCE.md)。source 工具只在离线验证中显式导入当前 Cosmos，不属于生产依赖。

## 已保留的复现材料

- `artifacts/pwm_wam_1028/`：当前 reference、原始 NumPy 输出、分层 tensor、固定请求矩阵、真实 HTTP/视频、性能日志。
- `environment.json` / `provenance.json` / `weight_sha256.json`：实际依赖、来源与权重摘要。
- `numerical_diagnosis.md`：所有分层诊断命令、环境失败原因和精确一致/不一致边界。
- `test/manual/test_cosmos_pwm_wam.py`：实权重 runtime/Python 验证；`test/manual/pwm_wam_*.py`：source 捕获与层/VAE 回放。

SGLang 起点 `990ee89877f668304761d7551d803ed6b93d1673`，分支 `pwm_infer`；native 核心来源 `vllm-omni-pwm` `d3e9aad33e861b69be16c4f341695e1cc12b3314`。主要本地 checkpoint：`0d1efb0380`（范围）、`92e50e09b2`（三塔）、`629925c278`（请求/采样）、`39dc0c1c49`（服务）、`a2dd639bda`（SDPA/合同）、`9a2a260afe`（诊断）、`66643efca1`（下载）、`0224fcb2ba`（资产/CI）。收尾提交通过 `git log 990ee89877..HEAD --oneline` 查看。

私有权重、大体积输出和 `.venv` 未加入 Git；完整证据保留于本机 `artifacts/pwm_wam_1028/`。没有推送远端、创建 PR、修改其他 checkout 或停止外部 GPU 作业。
