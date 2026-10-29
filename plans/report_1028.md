# 三塔 PWM WAM 接入 SGLang：结论

> 2026-10-29 最新状态：生产代码 `e9be9ec135` 已修复编译数值漂移。相同软件环境、GPU与请求下，新一轮 C/S/S/C 的稳定态 runtime 中位数由 Cosmos **2.288877 s** 降至 SGLang **2.140096 s**，延迟下降 **6.50%**；两端均 compile+FA3、有效FP32 IEEE策略一致。真实Python worker正在重新验收，尚未用直接runtime替代完整worker结论。

> 质量：未替换的生产代码六场景134帧视频均与同环境Cosmos FA3 eager逐像素一致；4场景完整动作一致，DROID/Robotwin各一值的FP32差异分别最多1.164e-10/5.960e-8。28请求cache/回访66数组比较exact；另12请求新进程DROID-first与smoke-first交叉36数组exact。参考 [质量汇总](../artifacts/pwm_wam_1029/final_boundary_quality/acceptance_summary.json) 与 [首次shape交叉](../artifacts/pwm_wam_1029/final_boundary_droid_first/cross_first_shape_comparison.json)。修复前12.27%/11.88%的数据只保留为历史，不能作为最终版本收益。

## 10/28 历史阶段结论

以下保留原始功能与eager阶段证据；其中“尚未迁入/不支持compile”、41项测试、旧路径和性能数字均属于该时点，当前配置见 [部署说明](../examples/diffusion/cosmos_pwm_wam/README.md)，实施进展见 [record_1028.md](record_1028.md)。

已完成本轮单GPU/B1范围的原生WAM接入、正确性验证和两项保守优化。同环境Cosmos-vs-native六场景的动作及视频逐位一致，目标环境的缓存开关/三入口互比也逐位一致。UND缓存相对未缓存路径实测延迟下降 **3.15%**（原尺寸无视频decode）和 **13.14%**（带视频短请求），GPU模型权重减少 **1.159 GiB**。最终审查发现的Python参数静默忽略问题已修复并经真实worker验证；所有本任务服务和worker均已退出。

原生接入与功能验证已完成：真实 checkpoint 由 SGLang worker 直接执行 UND、VIDEO、ACTION 三塔及联合 UniPC，Python、原生 action HTTP 和 Cosmos `/generate` 兼容入口均可交付动作及可选视频。

**同环境原生移植对齐通过，跨环境仍存在已定位的 VAE BF16 卷积差异。** 目标环境六种请求的 active physical action relative L2 为 0.007525–0.016698，视频每帧均通过 SSIM≥0.92、PSNR≥24 dB。未放宽动作阈值或替换生产输入，这些误差不能转为机器人任务成功率结论。当前 eager 接入和 UND 缓存均未超过已有 Cosmos FA3+compile 的性能。

目标：单 GPU/B1 的原生 SGLang WAM 推理、Action128 动作及可选未来视频、可复现的正确性和性能基线。实施记录见 [record_1028.md](record_1028.md)，设计见 [plan.md](plan.md)。

## 当前已确认

- 接入位置为主仓库 `sglang.multimodal_gen`，无需独立 Omni 引擎。
- 已适配固定 HF artifact、原生三塔、SGLang worker/API，修正与当前 Cosmos 的过程 mask、媒体和 seed 差异；复用代码保留来源归属。
- 已实现 `CosmosPWMWAMPipeline`，在 SGLang worker 内直接执行三塔和联合采样。生产代码无 Cosmos/vLLM import 或外部推理进程转发。
- Python `generate_action()`、原生 `/v1/actions/generations`、独立 Cosmos `/generate` facade 均已真实跑通，动作与可选视频可以交付。
- 隔离环境采用仓库目标 Torch 2.14.1、Transformers 5.19.0、Diffusers 0.37.0，目标 cuDNN 为 9.24.0.43；source 使用 Torch 2.10.0/cuDNN 9.15.1.9。Cosmos 初始 revision 为 `3d2aba0f`，外部开发期间推进到 fresh smoke 使用的 `42bea124`。早期六场景未各自记录 commit，只能按日志/reflog重建；相关实际数值调用路径源码等值，具体证据和局限见下文，不把所有运行伪称为同一 HEAD。
- 7 个分片、1,214 个模型 tensor 严格加载；tensor bytes 32,364,191,504，分片文件 bytes 32,364,372,752，已保存全部 SHA-256。
- 使用固定 `../vllm-omni-pwm/artifacts/pwm-wam-hf`（来源 iter_000008600），VAE `/share/project/eai_pwm/models/Cosmos3-Nano/vae`，tokenizer `/share/project/eai_pwm/models/Cosmos3-Nano-VLM`。未重新导出、覆盖资产或选择训练 latest。

Review 可从 [pipeline](../python/sglang/multimodal_gen/runtime/pipelines/cosmos_pwm_wam.py)、[三塔核心](../python/sglang/multimodal_gen/runtime/models/dits/cosmos_pwm_wam/transformer.py)、[请求合同](../python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos_pwm_wam/request.py)、[联合 runtime](../python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos_pwm_wam/runtime.py)、[HTTP adapter](../python/sglang/multimodal_gen/runtime/entrypoints/action/cosmos_pwm_wam.py) 和 [compat facade](../python/sglang/multimodal_gen/runtime/entrypoints/action/pwm_compat.py) 开始。

## 验收状态

| 项目 | 状态 |
| --- | --- |
| 严格权重加载与合同验证 | 实权重成功；shape/dtype/索引/合同/资产/registry 负例已测试 |
| 原生三塔和联合 UniPC | 同环境六场景原始像素输入：全宽动作、normalized actions、视频逐位一致；fresh smoke 生成坐标全轨迹一致，固定条件帧末位差异见下文 |
| 新增 CPU 测试 | 41/41：9 model +17 runtime +15 serving；最后接口补丁新增3项，serving全套已重测 |
| 既有完整回归 | Pi05 API、Cosmos3、registry 全套 pytest：163 passed，59 subtests passed |
| 目标环境对当前 Cosmos 原始像素 e2e | 七请求/六场景完成，shape/离散合同一致、inactive 槽位零、输出有限；每帧视频质量通过，跨环境 strict bitwise 不成立 |
| SGLang worker / Python / HTTP | 最终三入口动作逐值一致，13 项 metadata/资产摘要相同；原生与 facade MP4 逐字节一致（86,268 bytes） |
| 最终参数校验 | Python未支持控制、raw/直接参数冲突均入队前拒绝；12组非法请求0入队，2次有效raw/direct请求动作、normalized、视频与旧基线exact |
| 错误与生命周期 | 原生 400/facade 422 后恢复；四并发 1×200+3×503、Retry-After=1 后恢复；下载/删除/404、SIGTERM 文件和目录清理通过 |
| 状态隔离 | Libero → 多 domain/shape/seed → Libero，动作/normalized actions/视频首尾逐元素一致 |
| 请求内 UND 缓存 | 目标环境六场景×off/on/on/off，共24次 rollout；54组完整动作/normalized/video数组比较全部 exact |
| 显存优化 | 1214个checkpoint张量完整保留，used权重CUDA、unused文本头CPU；CUDA权重精确减少1,244,659,712 bytes，真实输出与优化前一致 |
| 性能与显存基线 | 同卡重复 Cosmos SDPA /FA3+compile /SGLang Python 完成，见下文 |
| 部署文档与可复现命令 | [部署说明](../examples/diffusion/cosmos_pwm_wam/README.md) 已提供并实际启动验证 |
| 静态检查 | 28份新增或修改的Python文件通过仓库pre-commit选定Ruff规则、format、isort；`git diff --check` 干净 |

三份单元测试在 CPU 张量上执行，但依赖 Diffusers。既有 hosted CPU CI 只安装 `python[dev]`，因此注册到已有安装 diffusion 依赖的 `base-b-test-diffusion-unit-1-gpu-h100`。CI discovery/validation/filter 已确认三文件均启用、无 skip；没有增加全局 CPU 依赖。私有 checkpoint 不加入公共 CI。

缓存和显存优化后的真实三入口复测共11组动作数组、7份MP4，均与旧结果一致，开关与继承值、下载/删除/404和关闭清理通过，见 [cross_route_summary.json](../artifacts/pwm_wam_1028/und_cache_serving/cross_route_summary.json)。随后修复Python参数合同：仅允许WAM实际消费的字段，拒绝`diffusers_kwargs`以及raw/外层不同值冲突；同值重复兼容，规范化后的请求仍可再次校验。最后两次真实worker请求继续精确一致，12类错误均未入队，证据见 [python_admission/summary.json](../artifacts/pwm_wam_1028/und_cache_serving/python_admission/summary.json)。HTTP的`runtime`包装与Python直接采样字段已在部署说明中区分。

## 数值结论与边界

当前 Cosmos 和 native leaf snapshot 在相同 Python 3.13 / Torch 2.10 / cuDNN 9.15 下，六种原始像素输入的全宽 Action128、normalized actions、无损 uint8 视频全部逐位一致，max abs=0。该实验没有注入 reference latent。诊断加载器只跳过无关 serving 包初始化，实际计算文件是记录 SHA-256 的原生 snapshot；它证明移植语义，不替代目标 Torch 2.14 的 SGLang worker 验证。

正确性 reference 选择 eager SDPA；FA3+compile 是单独性能对照，未宣称其与 SDPA 逐位一致。Cosmos HEAD 在任务期间有外部提交，最新 smoke 的 manifest 明确记录 `42bea124`。对初始与后续相关 revision 的只读审计确认 model/data/configs/utils、实际 inference entrypoint/args/defaults/action_serving 未变化，20个关键文件 blob/SHA-256 一致；变化集中在未调用的服务优化及诊断工具。早期各场景 revision 的归属由 stderr 时间与 reflog 重建，证据强度低于逐运行 manifest，详见 [source_revision_audit.json](../artifacts/pwm_wam_1028/same_environment/source_revision_audit.json)。原 `final_matrix_comparison.json` 的 `reference_source_commit=3d2aba0f` 只代表初始 session revision，不能作为每个场景的执行 revision；保留原文件以维持视频质量报告记录的SHA，逐场景来源以该审计为准。

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
| 目标 Torch2.14 SGLang direct runtime | 4.418 | 3.228 | 相同模型/采样实现，直接调用 runtime |

初始 SGLang eager 请求比已有 FA3+compile baseline 约慢 **45%**。相较 source SDPA 的差值不能单独归因于框架，因为 Torch/cuDNN 和计算路径不同。FA3、compile 尚未迁入；独立 UND 缓存收益见下文。

新增请求内 UND 缓存采用同 GPU2 的三组 ABBA，固定上述 workload、各模式预热一次，正式样本每种 6 次。未缓存平均 **3.257093 s**，缓存平均 **3.154454 s**，平均延迟下降 **3.1512%**；两者 median 分别为 3.244553 /3.154283 s。12 次动作/normalized actions 全部逐元素一致；另两次 profiler 输出也一致，profile 耗时不计入统计。缓存额外峰值 allocated 为 **36,954,112 bytes**。监督日志确认测试 GPU 没有外部并发，命令、输出、trace 与汇总见 [und_cache_validation/comparison.json](../artifacts/pwm_wam_1028/und_cache_validation/comparison.json)。这组测量基于 `4736596b7c`，不包含后续 CPU 文本头显存优化，也不表示超过 Cosmos FA3+compile。

另测短 smoke（3 steps、CFG1、4 actions/5帧、带视频 decode）：GPU1、每种模式预热3次，5组ABBA，每种10个样本。off平均 **0.313904 s**，on平均 **0.272644 s**，延迟下降 **13.1443%**；动作、normalized actions和视频全部 exact。该版本包含CPU文本头优化，两组采用相同权重布局；计时仍不含加载、MP4和HTTP。前一轮质量矩阵的单组smoke曾出现缓存慢17%的波动，因此补了本轮重复验证，没有删除先前样本；单轮矩阵不作为性能结论。完整数据见 [und_cache_short_request/comparison.json](../artifacts/pwm_wam_1028/und_cache_short_request/comparison.json)。缓存收益依赖请求，默认仍关闭，可显式开启。

独立显存优化将未使用的 `language_model.lm_head.weight` 保留在CPU，继续严格检查key/shape/dtype，并无损保留权重值；不删除参数、不绑定embedding、不改变推理计算。loader未执行全权重数值比对或有限性扫描。模型加载总字节仍为32,364,191,504，CUDA部分31,119,531,792，精确减少 **1,244,659,712 bytes（1.159 GiB）**。GPU3的匹配ABBA全宽动作与旧基线exact；cache-off/on峰值allocated分别为 **33,523,971,072 /33,560,925,184 bytes**。跨卡峰值比旧结果额外少1 MiB，这部分未归因于权重，也未据此声称延迟收益。加载字段不含VAE和激活，设备明细与对照见 [unused_head_cpu_validation/comparison.json](../artifacts/pwm_wam_1028/unused_head_cpu_validation/comparison.json)。

缓存 trace 的 self CUDA 时间主要为 GEMM 38.16%、cuDNN attention 28.08%，其余为逐元素、cast、norm 等。单 trace 未建立 overlap 收益；自动分析器基于 `nvjet` 名称给出的 FP8 提示不适用于本 BF16 模型。已保留未缓存 trace、缓存 trace、算子表和三表分析，后续 FA3/compile/norm/RoPE 优化应逐项验证舍入、完整动作与视频，不能直接套用其他模型的融合。

这是共享节点上的小样本基线，没有锁定整机独占资源，也没有置信区间/统计显著性结论。收尾时其他 GPU 作业已出现，未操作这些外部进程。

目标Torch2.14环境的 direct runtime 与 Python/worker 动作逐元素一致，六次重复也一致。两条路径 warm median 相差约 11.2 ms；Python 同请求的调用耗时减 worker 耗时，中位数约 10.1 ms。初始direct runtime峰值allocated **34,770,594,816 bytes（34.77 GB /32.38 GiB）**、reserved 35,383,148,544 bytes；这是上述无解码workload在显存优化前的allocator指标，不含CUDA context/驱动开销，也不代表最大视频shape。Python 父进程 `parent_peak_allocated_bytes=0` 不代表 worker 不占显存。完整六次结果见 [benchmark/summary.json](../artifacts/pwm_wam_1028/benchmark/summary.json)。

原尺寸带视频矩阵重放 Python e2e 5.001 s、worker 4.618 s，其中 sample 3.181 s、finalize/decode 1.402 s。HTTP 小 smoke 的带视频单次请求为 native 0.433 s、facade 0.468 s，包含 MP4 编码及响应传输；未单独测量 MP4 编码，也未做 HTTP 压测。这些都不是无视频 benchmark 的同口径样本，不据此比较框架速度。

共享 NFS/import/spawn 使初始化波动明显：最终矩阵 Python 初始化 103.3 s、benchmark 初始化 181.4 s；不等于纯权重加载时间。最终 direct runtime 初始化 14.43 s；早期小 shape smoke 加载 11.23 s、峰值 allocated 34.36 GB，不外推最大画布容量。

## 范围、环境与复现

支持范围为固定三塔 `width_only_1b` WAM、单GPU/B1、BF16网络/FP32联合采样、Action128动作及可选视频。请求内UND缓存默认关闭，显式true开启；缓存仅在同一请求的同一CFG分支复用，失败或结束即释放，false保留未缓存数值基线。未使用文本输出头在CPU，参与推理的权重常驻GPU。

暂不支持其他 action 模式、state/action 条件输入、多候选、动态 batching、TP/SP/DP/CFG 并行、动态offload、量化、LoRA、compile、CUDA graph、跨请求prefix cache、realtime/OpenPI WebSocket；不支持的控制明确拒绝。本轮完成请求内复用和显存优化，其余能力按实际需求另行扩展，没有改变步数、分辨率、CFG或量化精度换取速度。

本仓库 `.venv` 安装目标核心包，并通过 `pwm_validation_base.pth` 只读追加相邻环境的第三方 site-packages（如 msgspec、FastAPI、uvicorn）。没有追加 Cosmos/vLLM 仓库源码或修改其环境。**这次不是干净容器安装验证**；主要版本/路径见 `environment.json`，完整 294 项首选 distribution 见 `environment_all_packages.json`。命令清空 `LD_LIBRARY_PATH` 避免宿主 Torch 库混入。服务返回的九份资产摘要只覆盖 VAE 配置/tokenizer 文件，不包含模型/VAE 权重。

CPU 回归命令：

```bash
env -u CUDA_VISIBLE_DEVICES LD_LIBRARY_PATH='' PYTHONPATH=python \
  .venv/bin/python -m unittest discover \
  -s test/registered/unit/multimodal_gen -p 'test_cosmos_pwm_wam_*.py' -v

env -u CUDA_VISIBLE_DEVICES LD_LIBRARY_PATH='' PYTHONPATH=python \
  .venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_pi05_action_api.py \
  python/sglang/multimodal_gen/test/unit/test_cosmos3.py \
  python/sglang/multimodal_gen/test/unit/test_registry.py
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

缓存ABBA复现：将上述命令改为 `--route runtime --und-cache compare --warmup 1 --repeat 3`，使用新的输出目录；额外 `--profile` 在正式计时之后为两种模式各保存一份trace，并比较其输出。请求必须固定seed。正式性能数据还设置 `FLASHINFER_DISABLE_VERSION_CHECK=1`、`SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1`；各实验的完整argv和环境均在对应artifact目录保存。

## 已保留的复现材料

- `artifacts/pwm_wam_1028/`：当前 reference、原始 NumPy 输出、分层 tensor、固定请求矩阵、真实 HTTP/视频、性能日志。
- `environment.json` / `provenance.json` / `weight_sha256.json`：实际依赖、来源与权重摘要。
- `numerical_diagnosis.md`：所有分层诊断命令、环境失败原因和精确一致/不一致边界。
- `test/manual/test_cosmos_pwm_wam.py`：实权重 runtime/Python 验证；`test/manual/pwm_wam_*.py`：source 捕获与层/VAE 回放。

SGLang 起点 `990ee89877f668304761d7551d803ed6b93d1673`，分支 `pwm_infer`；native 核心来源 `vllm-omni-pwm` `d3e9aad33e861b69be16c4f341695e1cc12b3314`。主要本地 checkpoint：`0d1efb0380`（范围）、`92e50e09b2`（三塔）、`629925c278`（请求/采样）、`39dc0c1c49`（服务）、`a2dd639bda`（SDPA/合同）、`9a2a260afe`（诊断）、`66643efca1`（下载）、`0224fcb2ba`（资产/CI）。收尾提交通过 `git log 990ee89877..HEAD --oneline` 查看。

后续实现/验证提交：`4736596b7c`（请求内UND缓存与ABBA工具）、`7f8d26bff6`（同环境与视频质量诊断）、`29495faae6`（未使用文本头CPU常驻）、`455d63f168`（source revision审计说明）、`f616e8eb13`（Python参数严格校验与回归）。最终本地检查日志为 `final_admission_style.log`，完整既有回归为 `final_existing_optimized.log`，最终新增接口回归为 `und_cache_serving/python_admission_cpu.log`。这些是本机验证，没有宣称远端CI已运行。

私有权重、大体积输出和 `.venv` 未加入 Git；完整证据保留于本机 `artifacts/pwm_wam_1028/`。没有推送远端、创建 PR、修改其他 checkout 或停止外部 GPU 作业。
