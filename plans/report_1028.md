# 三塔 PWM WAM 接入 SGLang：结论

2026-10-29 最终结论：指定checkpoint的单GPU/B1原生WAM接入、质量验证和性能验收已完成。相对同时开启compile与FlashAttention 3的Cosmos Baseline，稳定态runtime延迟下降 **6.50%**，真实Python worker（含通信）下降 **7.51%**。六场景134帧视频与同环境源FA3 eager逐像素一致；动作误差和支持边界如下。本文取代此前eager阶段结论，旧结果和失败实验继续保留于实施记录、Git历史和artifacts。

10/29补测本地vLLM-Omni PWM：同卡/同软件栈、两端compile+FA3+UND cache，vLLM runtime中位 **1.954786s**、SGLang **2.136539s**，SGLang延迟高 **9.30%**，目前尚未打平。原样V/S的动作采样数学也有差别；速度与输出差异在下文分别报告，不将此结果解释为等质量实现的速度排名。

生产数值修复为 `e9be9ec135`，后续提交仅涉及测试和文档。完整决策与提交过程见 [record_1028.md](record_1028.md)，原始设计见 [plan.md](plan.md)，实际部署命令见 [README](../examples/diffusion/cosmos_pwm_wam/README.md)。

## 最终性能对照

固定同一 H100 80GB/GPU2、单请求/B1，Libero 960×640、40 actions/41 video frames、5 steps、CFG 1.5、shift 5、seed 42。`return_video=false` 仅关闭最终 decode/交付，**仍执行完整 VIDEO/ACTION 联合去噪**。

| 比较路径 | Cosmos compile+FA3 | SGLang compile+FA3+UND cache | 延迟下降 |
| --- | ---: | ---: | ---: |
| 直接生产 runtime | 2.288877 s | 2.140096 s | **6.50%** |
| 实际 SGLang Python worker（含 IPC） | 2.313668 s | 2.139875 s | **7.51%** |

两组各自完整C/S/S/C，均为四个独立加载进程，每端16个measured样本、中位数汇总。每进程首次请求单列、预热3次、测量8次，保留全部离群值。共8个有效进程、96次请求，其中64次正式测量。Cosmos两组均调用原始直接推理接口；第二组的通信成本计入SGLang实际worker。

runtime速度比为1.06952×，平均值C/S分别2.313672/2.163292 s；worker速度比为1.08122×，平均值C/S分别2.330083/2.146196 s。原始结果见 [runtime对照](../artifacts/pwm_wam_1029/final_boundary_runtime/comparison.json)、[worker对照](../artifacts/pwm_wam_1029/final_boundary_worker_retry/comparison.json) 和 [完整统计/逐样本质量汇总](../artifacts/pwm_wam_1029/final_boundary_summary.json)。

| 对照组/实现 | 两次构建耗时 s | 两次首请求 s |
| --- | --- | --- |
| runtime /Cosmos | 54.366 /39.764 | 31.579 /10.496 |
| runtime /SGLang | 39.673 /31.592 | 24.731 /7.670 |
| worker组 /Cosmos | 30.946 /29.030 | 28.065 /13.499 |
| worker组 /SGLang | 129.550 /78.400 | 24.763 /11.175 |

构建包含工具中的模型/worker初始化，受NFS与导入影响；不是纯权重加载或稳定态推理。每组各实现从新编译缓存开始，组内后续进程共享其磁盘缓存，因此不能把后一次首请求称作完全冷启动。首次编译耗时不混入稳定态样本。两组在不同时段运行，不能用SGLang两个中位数相减估计IPC成本；逐请求wall减runtime自计时的中位差为direct0.495ms、worker10.711ms，其中worker差值包含协议、调度及IPC。

计时从媒体/请求准备开始，到物理动作到达CPU结束；不含模型加载、NPZ/JSON保存、MP4编码或HTTP。Python路径另含真实worker通信。该结果是指定请求的稳定态延迟，不代表吞吐、P99、所有分辨率或所有domain的收益。

公平性与实际执行配置：

- 两端使用同一Python 3.12.3/Torch 2.14.1+cu130、CUDA 13.0、cuDNN 9.24、Triton 3.8、Transformers 5.19、Diffusers 0.37及固定权重/输入，未改外部checkout。
- Cosmos实际原始代码启用compile+FA3、metadata reuse=true、UND cache=false、CUDA graph=false；保持源服务默认的UND配置，未关闭compile或FA3制造慢Baseline。源可选QKV融合不是本次基线原有开启项。
- SGLang使用已有SGL FA3 kernel，显式`ver=3`；动态fullgraph编译与请求内UND复用开启。两端都是BF16网络/FP32 solver，步数、CFG、分辨率、seed和联合生成合同不变。
- 有效FP32 matmul/cuDNN TF32均为false；两端BF16 reduced-precision reduction均为true。SGLang请求结束/异常后恢复调用方flags；请求外caller为true不能代替请求内精度证据。
- 每块记录GPU进程、功率/时钟、源码和harness SHA、依赖/扩展来源、完整首次/预热/测量输出。未锁定整机独占，也未计算统计显著性或置信区间。

源在目标Transformers5环境仅恢复未使用Edge processor依赖的`get_image_size`旧re-export；WAM数学路径不变。该只读兼容overlay及函数SHA由manifest记录。源FA3仍为其原`flash_attn_3_nv`，原生为`sgl_kernel.flash_attn`，没有把两者伪称为同一个provider。

源有效精度另经 [只读审计](../artifacts/pwm_wam_1029/final_boundary_runtime/cosmos_effective_precision_audit.json)：620份manifest源码SHA匹配，构造函数显式关闭两项TF32，实际请求/模型/VAE/solver路径未见内层重新开启；日志和生成图确认compile、FA3及完整三塔联合生成，未发现减步数/CFG等捷径。

最终GPU监督从空闲开始、结束后再次清空。容器无法把NVML的host PID与/proc的namespace PID一一映射；worker采用parent/worker进程树与GPU fd、最多两个CUDA context及稳定GPU PID集合的关联审计。该能力边界和零样本监督误判/全新重跑均在record中说明，未把中止轮计入统计。

## 与本地 vLLM-Omni PWM 的补充对照（10/29）

**当前尚未打平：SGLang比本地vLLM原样native runtime慢9.30%，每请求约多182ms。** 测前采用中位延迟差≤5%的工程观察口径，本次未达到；这个口径不是统计等价检验。此前Cosmos对照的6.50%/7.51%不外推为vLLM结论。

固定同一H10080GB/GPU2、BF16/B1、checkpoint、原始输入及10-active Action128 mask。请求仍为Libero960×640、40 actions/41 frames、5 steps、CFG1.5、shift5、seed42；不做最终视频decode，但保留完整三塔联合生成。双方真实native runtime均开启compile+FA3+请求内UND cache，在同一S目标环境（Torch2.14.1、CUDA13、cuDNN9.24、Triton3.8、Diffusers0.37、Transformers5.19）运行；有效双TF32=False、BF16 reduced-precision reduction=True、deterministic=False。

完整V/S/S/V四个独立加载进程，每块1次first+3次warmup+8次measured，共48请求，双方各16正式样本；全部离群值保留。

| 原样直接runtime | 正式样本 | 中位延迟 | 平均延迟 | 最小 / 最大 |
| --- | ---: | ---: | ---: | ---: |
| vLLM-Omni PWM | 16 | 1.954786s | 1.961072s | 1.938106 / 2.060070s |
| SGLang PWM | 16 | 2.136539s | 2.141599s | 2.129162 / 2.193755s |

四块中位数按顺序为V1.955783、S2.139609、S2.131493、V1.951740s，反向顺序没有消除差距。外层同步计时从raw请求开始，到CPU动作返回结束，V原有CPU列表构造成本留在调用内，字段适配/数组保存位于计时外；没有使用V在视频decode前记录的内部elapsed冒充完整wall time。本次是runtime比较，不是HTTP、完整engine吞吐或SGLang Python worker比较；V单卡engine为inline UniProc，不能假设与S worker有相同IPC边界。

| 进程块 | runtime构建 | 首次请求（含编译/缓存加载） |
| --- | ---: | ---: |
| V0 | 153.605s | 190.627s |
| S1 | 12.304s | 30.143s |
| S2 | 11.179s | 8.773s |
| V3 | 11.120s | 6.552s |

import、模型构建、首请求与稳定态分开记录；NFS和磁盘缓存状态影响前者。两种实现分别从新编译目录开始，各自后续进程共享对应缓存，不宣称四次完全冷启动。vLLM使用真实package及其bundled FA3，S使用已有SGL FA3；没有namespace隔离、provider替换、数学补丁或外部环境修改。V的FP8、近似velocity cache、跨请求UND cache、CUDA graph、融合QKV/gate-up及多GPU均关闭，与原有普通BF16 compile+FA3配置对照。历史可选融合/graph证据收益很小或有数值差异，未堆叠为另一未经当前质量验证的Baseline。

源码中几项相关区别：

- V只在最终输出屏蔽inactive action槽位，S/Cosmos在初始噪声、网络边界和逐步采样中持续屏蔽。因此相同JSON不代表相同数学。
- V使用默认静态整层compile；S使用共享动态prefill/gen，并保留原ATen mean及BF16中间舍入。S这些数值边界由此前DROID回归证明必要，不能仅为延迟取消。
- FA3实际provider、UND缓存保存内容及CPU返回容器也不同。这次未做单因素性能消融，不能把182ms全归因于其中一项。

正式序列中，V的24次完整动作/normalized彼此exact，两个fresh进程间也exact；S的24次完整数组同样彼此exact，且均对同环境Cosmos FA3 eager主场景exact，另经 [原始数组独立核验](../artifacts/pwm_wam_1029/vllm_comparison/formal_runtime/final_audit.json) 确认。V对该锚点active physical relative L2=0.459180059、normalized=0.576815950，不是动作数值等价。

另在GPU0、独立编译上下文对同一主尺寸**带视频**请求执行两次原样V推理。完整动作/normalized/video自身重复exact；对当前S最终输出及Cosmos FA3 eager，active physical relative L2均为 **0.460072202**（max_abs1.253084064、RMSE0.247408028），normalized relative L2为 **0.578084861**。视频全部41帧通过既有门槛，最低SSIM **0.946424944**、PSNR **25.810457791dB**，pixel MAE1.777122065；离散合同一致且inactive输出全零。这个独立上下文与formal序列的V输出有区别，未混为同一次运行或谎报formal两个进程漂移。**动作差异显著，视频通过不能作为动作数值等价或机器人业务质量相同的证据；本次也未分离mask/编译/provider对差异的因果。**

证据与复现入口：

- [正式逐样本统计与动作比较](../artifacts/pwm_wam_1029/vllm_comparison/formal_runtime/comparison.json)、[命令/执行计划](../artifacts/pwm_wam_1029/vllm_comparison/formal_runtime/plan.json)；各block manifest保存实际argv/env、配置、provider及源码SHA。
- [独立审计PASS](../artifacts/pwm_wam_1029/vllm_comparison/formal_runtime/independent_audit.json)、[主任务独立重算](../artifacts/pwm_wam_1029/vllm_comparison/formal_runtime/root_recheck.json)：44个冻结文件当前SHA匹配，32个正式耗时全部纳入，记录的TF32、deterministic及BF16 reduction开关一致，四进程exit0，GPU2每块起止空闲。约2秒GPU采样最大间隔3秒，不声称精确host/namespace PID映射或整机独占。
- [视频/动作检查](../artifacts/pwm_wam_1029/vllm_comparison/quality_libero/findings.md)、[逐帧指标](../artifacts/pwm_wam_1029/vllm_comparison/quality_libero/comparison.json)、[独立NumPy重算](../artifacts/pwm_wam_1029/vllm_comparison/quality_libero/root_recheck.json)、[只读源码/历史审计](../artifacts/pwm_wam_1029/vllm_native_source_audit.json)。资产元数据hash不替代既有权重SHA。

V checkout为`d3e9aad33e861b69be16c4f341695e1cc12b3314`且有1280项既有staged变更，PWM模型/tools本身无局部改动；以上结论绑定实际文件SHA，不宣称全仓clean HEAD。本轮只新增本机比较证据和文档，没有修改两端生产代码或外部checkout；所有比较进程正常退出，GPU0/2释放，收尾四卡compute列表为空。

## 最终质量与状态隔离

正确性锚点为**同环境、同FA3的Cosmos eager**，用于保留原模型逐算子数值语义；性能Baseline继续使用Cosmos compile+FA3。选择该锚点前，已独立验证native FA3 eager与源FA3 eager的DROID视频精确一致，确认移植、VAE和FA3合同，而不是为最终候选临时挑选oracle。

最终未替换的生产代码结果如下，所有动作均比较完整128维：

| 请求 | 动作数/帧数 | physical最大绝对差 | normalized最大绝对差 | 无损视频 |
| --- | ---: | ---: | ---: | --- |
| 双相机smoke，3 steps | 4 /5 | 0 | 0 | 逐像素exact |
| Libero 960×640，5 steps | 40 /41 | 0 | 0 | 逐像素exact |
| DROID，50Hz动作/15FPS、CFG norm | 100 /33 | 1.164153218e-10 | 7.450580597e-9 | 逐像素exact |
| Robotwin joint，非连续mask | 4 /5 | 5.960464478e-8 | 5.960464478e-8 | 逐像素exact |
| Robocasa单相机 | 8 /9 | 0 | 0 | 逐像素exact |
| 缺省30 steps | 40 /41 | 0 | 0 | 逐像素exact |

共134帧，SSIM=1、PSNR=∞；固定逐帧门槛仍为SSIM≥0.92、PSNR≥24dB，未放宽。DROID/Robotwin各有一个FP32动作值存在上述末位差异，不宣称全矩阵动作bitwise；也不把数值验证转化为机器人闭环成功率结论。完整active/full-width误差、逐帧得分和三组source/native比较见 [comparison.json](../artifacts/pwm_wam_1029/final_boundary_quality/comparison.json)，简表见 [acceptance_summary.json](../artifacts/pwm_wam_1029/final_boundary_quality/acceptance_summary.json)。

10/29再次独立复核原始NPZ和当前源码SHA，结果与上述报告一致，见 [correctness recheck](../artifacts/pwm_wam_1029/correctness_recheck_20261029/audit.json)。DROID差异坐标为`[52,2]`：physical相差2个FP32 ULP，normalized相差1个；Robotwin为`[3,11]`，两数组均相差1个ULP。其余元素一致，inactive槽位全0；domain/mask/频率/seed/时间戳/caption摘要相同。ULP仅描述观察到的差异，不是新设定的业务容差。本次是对已保存GPU原始输出的CPU重算与当前生产代码身份核验，没有重新运行GPU推理或改变原验收门槛。

每个场景off/on/on/off，之后回访Libero，共28请求；22组重复/回访、66个完整动作/normalized/video数组比较全部exact。另一全新进程与编译缓存以DROID-first→smoke→DROID运行12请求，36数组与上述smoke-first矩阵exact，见 [首次shape交叉](../artifacts/pwm_wam_1029/final_boundary_droid_first/cross_first_shape_comparison.json)。最终runtime两个进程的 [24次主尺寸动作和normalized](../artifacts/pwm_wam_1029/final_boundary_runtime/source_eager_main_comparison.json)，以及两个真实worker的24次输出，包含首次/预热/正式样本，共48次完整数组均对源FA3 eager和最终质量矩阵主场景双锚点exact；逐样本证据在完整统计中。

源compile自身在DROID相对源eager存在明显漂移，最差SSIM0.875546、PSNR20.1516dB，未通过同一视频门槛。因此不能用源compiled输出替代原始数值语义，也不能把它当作允许本实现漂移的容差。最终native相对该compiled输出的差异仍披露在完整比较中，未修改Baseline数学或隐藏该结果。

源compiled主尺寸动作在不同新磁盘编译缓存组之间也不是bitwise，同组首次/预热/重复则稳定；源码620文件、参数、软件和TF32策略相同。该差异见 [源编译重复性](../artifacts/pwm_wam_1029/final_boundary_cosmos_compiled_reproducibility.json)。它没有被用于放宽本实现门槛；最终native四个fresh进程仍对原eager动作exact。

## 实际接入与修正

模型位于主仓库 `sglang.multimodal_gen`，在SGLang worker内原生执行UND/VIDEO/ACTION三塔和联合UniPC。生产代码无Cosmos/vLLM import、外部推理进程或HTTP转发。SGLang有独立Omni项目用于多阶段编排，但本任务无需把底层WAM推理放入该项目；最初源码/文档调研依据保留在plan中。

- [模型核心](../python/sglang/multimodal_gen/runtime/models/dits/cosmos_pwm_wam/transformer.py)：严格HF参数树、三塔mask/位置/domain逻辑、SDPA/显式FA3、动态编译、请求内UND缓存。
- [私有编译边界](../python/sglang/multimodal_gen/runtime/models/dits/cosmos_pwm_wam/execution.py)：复用已有FA3；RMS仅把原ATen mean保留在编译边界外，没有新CUDA kernel或近似归一化。
- [联合runtime](../python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/cosmos_pwm_wam/runtime.py)：请求状态、FP32 solver、动作/视频联合采样、局部数值策略与原始合同。
- [pipeline](../python/sglang/multimodal_gen/runtime/pipelines/cosmos_pwm_wam.py)、[native HTTP adapter](../python/sglang/multimodal_gen/runtime/entrypoints/action/cosmos_pwm_wam.py)、[兼容facade](../python/sglang/multimodal_gen/runtime/entrypoints/action/pwm_compat.py)：共用原生worker，保留动作/normalized、seed/domain和可选视频。

此次性能工作补齐FA3、metadata复用和compile，并发现、修正了两个编译数值问题：Inductor自动归约改变FP32求和次序；mm+residual→addmm重写省掉原BF16 GEMM中间舍入。最终保留原ATen mean，两个编译函数同时使用`emulate_precision_casts=True, pattern_matcher=False`。实际生成prefill/gen图分别保留4/9个mean边界、7/14个mm，addmm均0，见 [生成图检查](../artifacts/pwm_wam_1029/final_boundary_runtime/production_compiled_boundaries.json)。SiLU屏障、deterministic/strict候选未带来所需改善，未进入生产代码。

UND缓存仅复用同一请求、同一CFG分支的条件；false保留每步重新计算路径，失败或结束即释放。类默认cache=false，Hopper部署recipe显式开启FA3+compile+cache。未使用文本输出头继续严格加载并保留CPU，CUDA checkpoint权重减少1,244,659,712 bytes（1.159GiB），不是删权重或绑定embedding。

## 测试与服务验证

| 项目 | 结果与证据 |
| --- | --- |
| WAM单元测试 | 46项：model14、runtime17、serving15；新增真实CPU Dynamo执行、8种dtype/shape/stride归约合同及fake/alias验证 |
| 新数值回归有效性 | 旧RMS forward实际红灯/当前绿灯；不是仅mock调用计数，见 [红绿记录](../artifacts/pwm_wam_1029/mean_regression/red_green_summary.json) |
| CPU环境独立性 | 最终runtime+serving32passed+65subtests；仅隔离配置测试的外部显存查询，CUDA未初始化，失败/修复日志均保留 |
| 既有公共回归 | 此轮先前Cosmos3/Pi05/registry完整163passed+59subtests；末次RMS修复后重跑上述WAM46项及全部实权重最终验证，不把旧公共回归伪称为新HEAD远端CI |
| 真实native HTTP与facade | 最终8请求、16项动作/normalized比较exact，8个MP4逐字节一致；实际worker FA3+compile、双TF32=false |
| 服务错误/资源清理 | 非法cache字段native400/facade422后恢复；下载/删除/404、SIGTERM留存视频清理通过，上述HTTP/facade服务进程全部退出/GPU1清空 |
| 静态检查 | 修改Python的仓库选定Ruff规则、format、isort以及`git diff --check`通过 |

HTTP视频均5帧176×256@20FPS，85,023bytes；实际加载`sgl_kernel/flash_ops.abi3.so`由worker maps核对。服务证据见 [acceptance.md](../artifacts/pwm_wam_1029/final_boundary_serving/acceptance.md) 与 [跨入口比较](../artifacts/pwm_wam_1029/final_boundary_serving/cross_service_summary.json)。该smoke验证服务合同，不替代原尺寸质量矩阵。原生监督脚本曾过早检查自然退出的编译子进程，保留原始snapshot，宽限轮询后确认全部退出；没有额外kill外部进程。

单元测试使用CPU张量，但依赖Diffusers，因此注册在现有diffusion unit lane，不要求公共CI获得私有checkpoint。本机验证没有宣称远端CI已运行。

## 资产、环境与复现

固定HF artifact为相邻`../vllm-omni-pwm/artifacts/pwm-wam-hf`，来源iter_000008600，没有自动选择训练latest或重新导出覆盖。7分片、1,214个tensor，tensor bytes32,364,191,504；原始权重SHA见 [weight_sha256.json](../artifacts/pwm_wam_1028/weight_sha256.json)。配置hash不代表权重hash。

最终本机路径：

- SGLang VAE：`/share-mixed/project/eai_pwm/models/Cosmos3-Nano/vae`。
- tokenizer：`/share-mixed/project/eai_pwm/models/Cosmos3-Nano-VLM`。
- Cosmos原始VAE：`/share-mixed/project/eai_pwm/models/wan22_vae/Wan2.2_VAE.pth`。
- Cosmos源码：`../cosmos-framework-pwm`，最终对照revision `0cc924cb`；各运行逐文件SHA前后一致，不把任务期间所有旧运行伪称为同一HEAD。

`.venv`采用目标核心版本，并以只读`.pth`追加相邻环境部分第三方依赖；不是干净容器安装验证。Cosmos目标环境兼容依赖使用独立overlay，没有把训练源码加入生产环境。清空`LD_LIBRARY_PATH`避免系统Torch库混入；完整依赖/源码/扩展路径和命令以各运行manifest为准。

CPU回归（本机物理GPU只有0..3，因此9表示无可见GPU，同时兼容现有test_utils端口计算）：

```bash
env CUDA_VISIBLE_DEVICES=9 LD_LIBRARY_PATH='' PYTHONPATH=python \
  .venv/bin/python -m unittest discover \
  -s test/registered/unit/multimodal_gen -p 'test_cosmos_pwm_wam_*.py' -v
```

GPU手工验证使用 [test_cosmos_pwm_wam.py](../test/manual/test_cosmos_pwm_wam.py)，公平性能对照使用 [pwm_wam_benchmark.py](../test/manual/pwm_wam_benchmark.py) 及其 [说明](../test/manual/pwm_wam_benchmark.md)。最终原始argv/env分别在`final_boundary_quality/command.json`、`final_boundary_droid_first/command.json`、`final_boundary_runtime/launch_block*.json`和服务provenance中，输出目录需新建。

支持范围：固定Nano `width_only_1b`、单GPU/B1、BF16网络/VAE、FP32联合UniPC、Action128及可选未来视频。暂不支持其他action模式、state/action条件输入、多候选、动态batching、TP/SP/DP/CFG并行、offload、量化、LoRA、CUDA graph、跨请求prefix cache、realtime/OpenPI WebSocket。不支持的控制明确拒绝；显式FA3依赖/硬件或编译失败直接报错，不静默换后端。

## 历史证据与提交

初版只接通eager路径，相对compile+FA3 Baseline更慢，不能算最终加速目标完成。10/29修复前曾测得12.27% runtime/11.88% worker收益，但DROID视频质量失败；这些数字不是当前实现的最终收益。其全部数据保留在`fair_ieee/`、`fair_worker_ieee/`、`quality_matrix_ieee_retry/`。编译归约、SiLU、strict、no-pattern候选及scope纠正分别在`quality_droid_*`、`quality_mean_silu_candidate/`和`diagnostic_scope_correction.md`。

10/28跨Torch/cuDNN环境的BF16 VAE偏差、分层诊断和旧HTTP/性能证据保留在`artifacts/pwm_wam_1028/`；早期报告原文可从Git历史查看。同环境最终验证不能推广为任意软件版本的bitwise保证。

本轮主要提交：`495e9cc5bc`/`0192b1edf3`/`fdd689092d`（FA3与compile接线）、`ee7c4c35fa`（公平基准工具）、`fc60830b1d`（共享动态编译边界）、`2f499c22b3`（请求内TF32策略）、`e9be9ec135`（保留归约/BF16舍入）、`9235da1b95`（数值回归/部署）、`e41e1e5f06`（CPU测试硬件查询隔离）。详细历史见record及`git log 990ee89877..HEAD --oneline`。

本任务的服务、worker及验证进程均已退出；收尾GPU compute进程列表为空、视频store清空。私有权重、数组、编译缓存和`.venv`未加入Git，证据保留在本机artifacts。没有推送远端、创建PR、修改其他checkout或停止外部GPU作业。
