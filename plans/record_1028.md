# 三塔 PWM WAM 接入 SGLang：实施记录

## 目标与执行约定

在当前 SGLang 主仓库原生执行三塔 WAM，首版范围为 Nano `width_only_1b`、单 GPU/B1、BF16 网络/FP32 联合 UniPC，支持 Action128 动作与可选未来视频。以当前 Cosmos 行为为 reference，复用已有 native 实现并补齐已发现语义差异。详细设计见 [plan.md](plan.md)。

用户已授权实现、测试、保守决策与频繁本地提交；尽量不打断用户。此任务不自动推送远端、不修改其他 checkout、不停止已有其他进程。结论持续更新于 [report_1028.md](report_1028.md)。所有时间为 Asia/Shanghai。

## 2026-10-29：重新打开性能验收

用户明确指出接入必须带来性能收益。之前把功能接入、数值验证和相对自身 eager 的缓存收益作为完成标准过低；10/28 结果已证实仍慢于同时开启 compile 与 FlashAttention 3 的 Cosmos Baseline。继续实施与验证，不能以保留该差距为由结束。

### 决策 D12：以优化后的 Cosmos 为验收对象

- 固定同一 artifact、真实输入、步数/CFG/seed 与动作输出；保留完整联合 VIDEO/ACTION 去噪，不使用减少采样步数、分辨率、精度或跳过视频 latent 等捷径。
- 优先补齐 SGLang 原生可选 FA3 和 torch.compile；eager SDPA 留作正确性 reference。显式选择 FA3 时依赖/硬件不支持必须报错，不得静默 fallback。
- 基准两端同时开启 compile 与 FA3，在同软件环境/同一空闲 GPU 测量；记录首轮、预热、全部稳定态样本和输出误差。真实 Python worker 另做完整链路验证。
- 基础性能达标后才报告收益；缓存对自身的收益不能替代相对 Cosmos 的收益。若实测仍慢，依据 profile 继续定位，保留所有反向结果。
- 并行分工：模型后端/编译、对照工具、独立审查与模型单测；主任务负责 runtime/配置接线、正式 GPU 对照、文档与提交。开始时四张 H100 均空闲，GPU0/1 给组件验证、GPU2 给正式性能对照。

### 决策 D13：生产后端与诊断替代边界明确区分

生产 FA3 使用 SGLang 已有 kernel 的显式 `ver=3`，不依赖 Cosmos/vLLM；命名采用 `flash3`，避免通用 ServerArgs 把 `fa3` / `fa4` 都归一化为 `fa` 后丢失用户要求的版本。metadata 每请求/CFG 分支构造一次。compile 使用 fullgraph 静态每层/UND prefill，失败直接抛出，保留未编译路径。runtime、Python/HTTP pipeline、兼容 facade 和手工 harness 均显式传入配置，响应参数回显实际选择。

目标 S 环境固定 Torch 2.14.1/Transformers 5.19，源 C 环境是 Torch 2.10/Transformers 4.57。优先尝试同 2.14 环境运行真实 Cosmos reference；若需要在 C2.10 环境以 provider adapter 运行 native leaf，仅作为数学诊断，并记录 adapter/source SHA，不把它称为生产 SGLang worker。真实 worker 另测，不以父进程 Dynamo counters 证明子进程编译。

初查把 Cosmos 的 QKV 融合源码误当成已测 baseline 的有效配置，进一步核对已纠正：它由 `action_server.py --fuse-qkv` 显式开启，10/28 的 reference worker 未调用该入口。当前先不混入 QKV/solver/量化/近似缓存优化，待 FA3+compile 公平实测后决定。

阶段验证：目标 S2.14/H100 已实际调用 SGL FA3，输出有限。serving 单测 15 passed +43 subtests，runtime 单测 17 passed +22 subtests；首次 serving 冷导入耗时 277.71 s，不是推理性能。Ruff 选定规则、format、isort 与 diff 检查通过。FA3 整层 GPU 编译及全模型性能仍在进行，尚未宣称达标。

### 决策 D14：以全模型输出决定编译边界

- 组件首次第二 shape 的 cache on/off 出现舍入差异，固定 `emulate_precision_casts=True` 后小组件通过；失败日志保留在 `artifacts/pwm_wam_1029/component_fa3/check_layers_default_precision.*`。
- 全模型预备运行（非正式性能验收）随后仍发现静态编译的 cache on/off 动作 relative L2=0.008012、max_abs=0.028799，手工 harness 如期中止，没有把失败轮当成通过。首个 off 总耗时 2.066831 s、on 2.000552 s，仅用于定位；不能据此宣称输出验收或最终性能收益。证据 `preliminary_compile_cache_compare/`。
- 改为编译路径 cache on/off 共用相同的 UND prefill 与 VIDEO/ACTION generation 编译函数。未缓存时每层重新计算 UND，缓存时跨采样步复用同样的结果；eager SDPA oracle 保留原计算路径。启用动态序列长度，避免不同 caption 长度耗尽静态图上限。新边界组件覆盖两 shape/CFG/A→B→A 与连续68种长度，三塔 cache 开关 exact；还需全模型复核。
- 独立 CPU 审查：36层复用图，没有每层放大；动态候选66种caption长度+回访仅3图，469个request tensor弱引用全部释放。旧静态候选64种变体后会明确报错，未采用无限加大上限或静默退回 eager。
- 第一次全模型预备运行仅加载后失败：原 `/share/project/eai_pwm/models/...` 已迁移至 `/share-mixed/project/eai_pwm/models/...`。未更换资产内容；重试使用已存在路径。runtime 增加本地 VAE/tokenizer 配置文件预检，避免缺少路径时先加载32GB模型。
- 当前 Cosmos 在 S 的同一 Torch2.14/Transformers5 环境已真实完成 FA3+compile 短请求。只为未使用的Edge processor旧导入位置恢复 `get_image_size` re-export；记录adapter函数/文件SHA，不改模型/采样数学或外部checkout。正式大请求同环境ABBA对照继续进行。

### 决策 D15：性能对照必须匹配有效 FP32 策略

新动态共享 prefill 的全模型主请求 gate 通过：四次 off/on/on/off 的全部 Action128 和 normalized arrays exact，source SHA 前后一致（`quality_main_cache/`）。同软件环境的首组正式样本随后被独立审查发现有效数值策略差异：Cosmos 的 FP32 matmul/cuDNN TF32=False，SGLang 平台默认=True；时间嵌入确有 FP32 GEMM，不能假设这一差异无影响。

保留 `fair_runtime/block_00` 至 `block_03` 为各自默认 policy 的诊断组，不作为同精度提速验收。原生 runtime 在单请求作用域关闭两项 TF32，正常返回与异常后恢复 caller 设置；response parameters 回显实际作用域内 flags。扩展既有作用域回归，以外部 True→请求内 False→成功/异常恢复 True 验证，不以父进程 caller flags 充当 worker 有效精度。

随后另采有效 IEEE 策略一致的 C/S/S/C，对照 compile+FA3、同软件栈/同输入/同GPU；保留两组全部数据，不删除第一次 cache gate/默认policy benchmark。C 原生配置不改，只有 S 修正其默认数值策略。采样工具在各完整组结束前保持同一SHA，汇总分析再增强有效/外部policy和harness身份检查。

### 决策 D16：性能通过不能覆盖质量矩阵失败

- 同有效IEEE策略的正式runtime对照共16 measured/实现：C median 2.299706 s、S median 2.017423 s（延迟下降12.2747%、1.13992×）。真实Python worker对照：C 2.296959 s、S 2.024113 s（含worker IPC）；首轮/3轮warmup/全部原始样本保留。两端均compile+FA3，无视频decode但完整联合生成；分别见 `fair_ieee/` 和 `fair_worker_ieee/`。
- 显式SGL FA3 provider的IEEE主请求4次与六场景+回访28次，全部cache off/on/回访的全宽动作、normalized与uint8视频exact。源码前后一致，实际FP32 flags均False。真实provider审计为sgl-kernel；未使用community fallback或Cosmos训练代码。
- 原生与Cosmos跨编译上下文不是bitwise。S formal两次fresh-process相互exact，但独立质量进程/worker/不同首次shape会有差异；发现归一化reduction的自动调优R0_BLOCK在2048/4096之间不同，而归一化后完整generation源码相同。此为可证明配置差异，尚不能据缓存文件把所有输出差异归因于它。
- 同环境6场景对照发现DROID case2末帧失败：native compile vs Cosmos compile 最差SSIM=0.891880、PSNR=20.4875 dB；Cosmos自身compile vs eager也失败（0.875546/20.1516）；native compile vs Cosmos eager为0.932534/22.728 dB，仍未达到固定PSNR24 dB。其余5场景逐帧通过。源自身漂移不能充当本实现通过门槛的证据，未调低阈值。
- 整体验收保持未完成，继续GPU0区分native FA3 eager、SDPA eager和编译路径的DROID输出，研究确定性归约等保守数值策略。已通过速度测量不替代此质量工作；所有失败和早期策略样本继续保留。
- CPU新模型13、runtime17、serving15合计45项；原有Cosmos3/Pi05/registry回归163 passed +59 subtests。IEEE作用域新增断言已用旧forward实际红灯/当前绿灯验证，证据 `ieee_review/`。

### 决策 D17：隔离编译归约与激活，不降低质量门槛

- DROID同环境最小诊断已排除媒体/VAE/FA3移植语义问题：native FA3 eager与Cosmos FA3 eager的33帧视频逐像素exact；active physical仅1个值差1.164153218e-10，normalized差7.450580597e-9。重复请求exact。native SDPA eager相对Cosmos FA3 eager也逐帧通过原门槛（最差0.934650/25.1064dB），但不是bitwise。证据 `quality_droid_eager_diagnosis/`。
- `deterministic=True` 和再加 `numerics='strict'` 两个显式编译诊断均失败：最差SSIM0.913014、PSNR21.7381dB。选项真实生效且repeat exact，但不能作为修复。未修改生产配置来伪称通过。
- 仅在诊断进程为RMSNorm的原始 `Tensor.mean(-1, keepdim=True)` 加ATen custom-op编译边界，保持所有其余表达式不变，DROID全部33帧通过：最差SSIM0.979479、PSNR32.2117dB；动作仍有relL2约0.023，未宣称精确。证据 `quality_droid_aten_mean/`。下一步单独验证原F.silu的编译边界及主尺寸性能，候选未验收前不改生产。
- 证实不同编译历史的generation图源码归一化后相同，但归约autotune R0_BLOCK2048/4096不同，方差求和的浮点次序可不同。该证据不等于所有漂移已证明源自此kernel，也未见请求Tensor泄漏证据。完整缓存/行号见 `compile_autotune_diagnosis.{md,json}`。
- 实际优化服务验证完成：native HTTP与Cosmos facade各4次off/on/on/off，共8请求的动作/normalized与视频均exact；实际worker为FA3+compile、两个FP32 TF32开关False。视频下载、删除/404、native400与facade422错误后恢复正常，两服务优雅退出、GPU1清空、输出store清理，证据 `serving_ieee/`。其数学路径为当前2f499实现，不能据此宣称后续新编译算子已通过同样验证。

### 决策 D18：保留 BF16 中间舍入，再验收性能

- mean-only 与 mean+SiLU 的 DROID 动作、normalized 和视频三数组逐元素相同；SiLU 编译屏障真实生效，但没有改善该案例，主尺寸候选测试还从2.14350 s变慢为2.19079 s，因此不引入该算子。该轮是单进程候选筛选，不替代正式对照。证据 `quality_mean_silu_candidate/`。
- 定位到第二处编译数值变化：原始 `residual + linear(x)` 先产出 BF16 GEMM 结果，再进行残差加法；Inductor post-grad 的 mm+add→addmm 重写省掉该中间舍入。`emulate_precision_casts=True` 并不阻止这条重写；不能仅凭开启该选项宣称保留逐算子语义。
- 诊断仅保留原 ATen mean 边界，再显式关闭 `pattern_matcher`。DROID 与同环境 Cosmos FA3 eager 的33帧视频恢复逐像素exact，physical最大绝对差恢复为1.164153218e-10，与native eager相同。保持原质量门槛，下一步落地最小修复、完整六场景及不同首次shape复核，并重测正式runtime/worker性能。证据 `quality_droid_no_patterns/`；旧12.27%性能数字不作为修复后版本的最终收益。
- 原始质量语义参考使用同环境同FA3的Cosmos eager，性能对照继续启用Cosmos compile+FA3；同时披露该source compile自身的数值漂移，不通过修改其配置制造更慢的Baseline。
- 候选artifact的部分 `command.json` 沿用父脚本的“No compile/model monkeypatch”描述，实际诊断有进程内替换RMSNorm/编译选项（SiLU轮还替换F.silu）。这是诊断而非生产执行；补充说明纠正范围，不覆盖历史原始数据。最终生产验证必须使用未替换的原生代码。
- 最小修复已提交 `e9be9ec135`。候选主尺寸Libero的全128维动作、normalized与Cosmos FA3 eager精确一致，生成图保留独立mm（gen14处/prefill7处）、addmm为0。生产代码冻结后开始新的28请求质量矩阵和7个fresh进程性能/worker序列，尚未提前引用候选耗时作最终验收。
- 新增mean回归采用真实CPU Dynamo图执行和opcheck，覆盖8种2D/3D、FP32/BF16、连续/非连续输入，验证原eager输出、FP32归约、fake metadata与非alias。模型14项通过，旧forward红灯/新forward绿灯。CPU进程用`CUDA_VISIBLE_DEVICES=9`（本机只有0..3）绕过test_utils对空字符串的索引问题，device_count=0且CUDA未初始化，不占GPU0；不再需要之前的临时环境读取patch。
- 部署文档改为显式推荐Hopper FA3+compile+UND缓存，并保留eager对照方法；类默认值未隐式切换。数值策略和首次编译成本写入说明。

### 决策 D19：用最终生产代码重新验收

- `e9be9ec135` 冻结后，GPU0运行未替换的生产runtime：6场景各off/on/on/off，再回访Libero，共28请求。22组重复/回访、66个完整Action128/normalized/video数组全部exact，前后源码SHA一致，无GPU0外部并发。`final_boundary_quality/command.json` 和 `acceptance_summary.json` 保存证据。
- 相对同环境Cosmos FA3 eager，6场景视频全部逐像素exact（共134帧，SSIM=1/PSNR∞）。4场景全宽动作/normalized exact；DROID仅一个值physical差1.164153218e-10、normalized差7.450580597e-9；Robotwin仅一个值physical/normalized差5.960464478e-8。保留这些FP32末位差异，不宣称所有动作bitwise，也不由此推断机器人闭环成功率。
- 相对Cosmos compile仍保留原有DROID差异：源compile自身对源eager未通过原固定视频门槛。未改源数学/配置、未降低SSIM0.92/PSNR24dB门槛；完整三组对照均保留在 `final_boundary_quality/comparison.json`。
- 为覆盖此前首次shape导致的漂移，再以全新进程/编译缓存在GPU3执行DROID-first→smoke→DROID，各off/on/on/off；与GPU0 smoke-first矩阵交叉比较，不使用artifact monkeypatch。
- 新增model14项通过。最终CPU重跑初次为runtime+serving31pass/1fail：配置测试的自动显存查询实际尝试CUDA，而不是生产推理回归。仅在该unit测试隔离外部 `current_platform.get_available_gpu_memory` 查询（返回80GiB），真实ServerArgs解析/拒绝逻辑保留；重跑32passed+65subtests，device_count0、CUDA未初始化。原失败日志和修复后日志均在 `mean_regression/`，最终WAM单测合计46项。
- DROID-first生产验证已完成：12请求、36个数组与GPU0矩阵逐元素exact，独立进程/独立编译缓存、不同首次shape的漂移未再出现。主尺寸正式runtime C/S/S/C各16 measured，C median2.288876745 s、S median2.140096151 s，延迟下降6.5002%；两次native fresh process包含first/warm/measured的24组完整动作/normalized均与Cosmos FA3 eager精确一致。实际生成图prefill/gen分别4/9个原mean边界、7/14个mm、addmm均0，见 `final_boundary_runtime/production_compiled_boundaries.json`。worker组继续独立测量，未提前作为通过。

### 决策 D20：保留监督失败，不把它混作模型或性能样本

- 最终native HTTP/facade均已通过，各4请求共16项动作/normalized比较、8个MP4逐字节exact（85,023bytes，5帧176×256@20FPS）。非法cache字段400/422后恢复、删除后404、SIGTERM清理额外留存视频通过；实际FA3扩展maps和两个有效TF32=False均核对。22份生产文件与`e9be9ec135`相同，所有我方服务/worker退出、GPU1归零。证据 `final_boundary_serving/`。
- native服务的artifact supervisor第一次在parent退出后立即检查编译子进程，短暂仍有自然清理中的child而断言失败；保留initial脚本/snapshot/log，不重写历史。随后确认无需额外信号便全部退出，加入45秒清理宽限、完成兼容facade验证。不是生产数学或视频store故障。
- 最终worker性能组的首个S块在分布式初始化时被artifact supervisor误中止（0采样输出）：真实Python入口正常创建parent CUDA context及worker context，计数>1的旧规则将第二个context误判为外部进程。保留`final_boundary_worker/`为监督失败证据，不纳入性能统计；在新目录`final_boundary_worker_retry/`重新执行完整C/S/S/C，不改生产/harness。
- 本容器的NVML返回host PID，而/proc只提供本namespace PID，无法严谨一一映射。保守监督改为GPU起止空闲、runtime/C最多1个context、Python最多2个；接纳第二个前须有namespace parent+worker两份GPU fd证据，之后PID集合保持稳定，替换/超限只终止自己的进程组。每2秒保存两侧PID/命令/PPid/NSpid/fd和GPU显存/时钟，明确监测能力边界，不伪称获得host权限或精确PID映射。

### 决策 D21：关闭指定单GPU/B1范围的最终验收

- `final_boundary_worker_retry/` 全新C/S/S/C完整完成，各方16 measured，C median2.313667792 s、S worker median2.139874681 s，延迟下降7.5116%、1.081216×。8个最终有效进程（runtime与worker各4个）均exit0、生产源码SHA不变，96次请求中64次正式测量；GPU审计无超限/替换，全部退出后GPU空闲。
- 最终runtime C2.288876745/S2.140096151 s（下降6.5002%）和上述worker数据分别来自独立同卡配对组。48次native first/warm/measured的完整128维动作及normalized对Cosmos FA3 eager、最终quality主场景双锚点全部exact。完整原始样本、构建/首次请求、源码和版本汇总 `final_boundary_summary.json`；不把worker略低于direct的中位数解读成通信零成本，逐请求wall−runtime.total中位差分别0.495/10.711ms，后者包含协议/调度/IPC。
- 源有效精度独立审计完成：620份manifest源文件匹配，构造函数明确关闭TF32；实际请求/网络/VAE/solver及provider无内层重新开启，compile/FA3/metadata复用/完整联合latent都有callsite和生成图证据。`final_boundary_runtime/cosmos_effective_precision_audit.json` 仅只读审计，不假装是逐算子capture。
- 另外保留源compiled自身的跨编译缓存漂移：相同源码/软件/策略/请求，不同fresh磁盘cache组的完整主尺寸动作有差异，同组重复稳定。`final_boundary_cosmos_compiled_reproducibility.json`记录，不由此放宽本实现质量门槛。原生四fresh进程保持对eager动作exact。
- 当前支持范围的原生接入、质量和加速目标已完成；最终报告和部署recipe按实际支持能力重写。46项WAM单测、六场景134帧、不同首次shape、实际runtime/worker、两HTTP入口均有证据。动作矩阵两个单值末位差异、非干净容器环境和未运行远端CI等边界明确保留，未推广为业务闭环成功率或所有shape/硬件收益。
- 所有本任务服务/worker/验证进程退出，收尾GPU compute列表为空、视频stores清空。只在本分支频繁本地提交，没有远端推送、外部checkout写入或停止外部作业；源码最后数值修复为`e9be9ec135`，之后均tests/docs变更。

## 2026-10-28 16:43：开始实施

- SGLang 起点：`990ee89877f668304761d7551d803ed6b93d1673`，分支 `pwm_infer`；只有先前生成的 `plans/plan.md` 未跟踪。
- Cosmos reference：`3d2aba0fe5e293f505d7ec52cb6b2eba8cc95d06`。
- native 复用来源：同级 `vllm-omni-pwm`，HEAD `d3e9aad33e861b69be16c4f341695e1cc12b3314`；不修改该 checkout。
- 四张 H100 80GB 当前均空闲。保守选择 GPU0 供主验证进程使用；子任务先做 CPU 验证，GPU 使用统一分配。
- 已定位可读 native HF artifact `../vllm-omni-pwm/artifacts/pwm-wam-hf`、Wan VAE、Cosmos3-Nano-VLM tokenizer 与历史请求；先用该固定产物验证迁移，不选择训练 latest、不重新导出覆盖资产。
- 测试约定：使用 `write-sglang-test`、`test/README.md`、`.claude/rules/unit-test-admission.md`；新增测试保护合同/数值/生命周期，不照抄实现断言。
- 并行分工：模型/位置/VAE/UniPC；请求与采样；SGLang registry/pipeline/action API；主任务负责环境、reference/e2e、文档和本地提交。

### 决策 D01：先隔离兼容性，再优化

首版 SDPA/eager，不迁入 vLLM scheduler/worker、FP8、Graph 或分布式执行。保留兼容 UniPC 系数/corrector 与 Wan RMS 修复；补齐 inactive action 逐步 mask、媒体/prompt 输入和缺省 seed 差异。跨请求缓存暂不支持。优化不得改变当前模型合同。

### 决策 D02：复用环境资产，不修改已有环境

宿主 Python 为 3.12/Torch 2.12 dev，缺少 diffusion 依赖。旁边 native 环境具有 Torch 2.13、Diffusers 0.40、Transformers 5.14.1；与目标仓库声明仍不同。创建独立验证环境或依赖 overlay，记录实际版本和差异，不在 reference/native 环境直接升级包。后续优先尝试目标版本；任何版本限制明确写入结论，不隐藏兼容性缺口。

## 2026-10-28 17:05：原生核心与参考产物

- 新增纯 PyTorch 三塔核心、位置编码、严格分片 loader、兼容 UniPC 和 Wan RMS；生产代码没有 Cosmos/vLLM 导入或外部 worker 转发。
- loader 对 index/shard/model 的 key、shape、dtype 和路径逐项检查；Action128 的输入与 velocity 屏蔽无效槽位。时间编码常量保持 FP32。
- 小型三塔在 FP32/BF16、两种 token geometry 下与移植来源的 SDPA 分支逐元素一致。7 项模型合同测试已写入 CPU CI；完整执行待独立依赖环境安装完成，当前只宣称静态 Ruff/AST 与独立核心对照通过。
- 当前 Cosmos HF reference 已完成 `request_smoke.json`：GPU0、Torch 2.10.0+cu130、BF16/SDPA、3 steps、5 frames、4 actions、256 高画布。保存 `artifacts/pwm_wam_1028/reference_smoke/{response.jsonl,video.npy,stderr.log}`；联合采样日志约 4.27 秒，含首次运行成本，不能作性能结论。
- 独立 `.venv` 安装期间 Torch 目录尚未写完，过早导入触发系统库/NVSHMEM 混用；暂停使用该环境直到安装完成。共享 NFS 上大型依赖写入耗时，参考环境不受修改。
- 新增手工验证入口 `test/manual/test_cosmos_pwm_wam.py`，显式接收资产路径，保存无损动作/视频数组、版本、耗时，支持 runtime/Python 两条路径与固定请求重复性、reference 误差报告。私有权重不加入公共 CI。

### 决策 D03：合同正确性优先

先前 native 实现仅在最终返回时清零无效槽位，与当前 Cosmos 的每步 mask 不同。此接入修正初始 noise、模型输入、velocity 和 solver 固定坐标；本次必须重做数值 reference，不能引用历史 native 的通过结论。

## 进行中的工作

## 2026-10-28 17:14：请求与采样接入

- 核心提交：`92e50e09b2`。本阶段新增共用 WAM 校验、第一帧图片/视频/URL 输入、相机 canvas resize/seam、caption、独立视频/动作频率与随机 seed 冻结。
- 原生 runtime 使用 request-owned 联合 FP32 latent、每请求新 UniPC，始终联合 VIDEO/ACTION；`return_video=false` 仅关闭解码交付。分别返回物理/normalized actions、domain、mask、时间戳和合同指纹。
- 测试覆盖非连续 mask、独立噪声、联合 CFG norm、A/B/A 状态隔离、normalize/projection domain 区别、媒体/URL 和 prompt 文件。阶段耗时使用 CUDA 同步边界，后续报告包含这一测量成本。
- 当前源 `guidance_interval` 比较的是 solver 的 0..1000 时间（非网络 rescale 后时间）；UniPC 忽略 EDM `sigma_max`，首版仅接受缺省/80，其他值明确拒绝。
- 目标环境 Torch 2.14.1 / Transformers 5.19.0 / Diffusers 0.37.0 导入成功。公共测试导入发现缺 `datasets`，pipeline 导入缺 `addict`；统一补包，无 monkeypatch 绕过导入。此前不完整 Torch 导入失败已消除。
- 中间量 reference 捕获首次在 CUBLAS 初始化失败，保存输入后独立排查；同 checkpoint 的原始 smoke reference 已成功，两者分开记账。

### 决策 D04：保留原生入口与薄兼容服务

提供 SGLang `generate_action` 和 `/v1/actions/generations`，另以独立 `pwm_compat` 进程保留 Cosmos `/generate`、`/models`、`/info`、`/domains`、`/healthz`、`/outputs` 形状；全部通过同一原生 SGLang worker，不调用 Cosmos 服务。原生 action 输入错误沿用 400，兼容入口 422。视频临时附件限定随机 ID、最多 128 个、保留一小时，正常退出清理。兼容层不复制 Cosmos 全部 defaults/output 持久化系统，差异在部署说明中列明。

## 2026-10-28 17:24：CPU 回归与第一轮实权重结果

- 请求/采样提交：`629925c278`。
- CPU：模型 7/7、runtime 10/10、新服务/registry/兼容层 6/6；现有 Pi05 API 10/10、Cosmos3 相关 18 tests +4 subtests、registry 11/11。没有用 mock 代替三塔数值测试；HTTP 单元测试仅 mock 外部调度与编码边界。
- 测试必须 `env -u CUDA_VISIBLE_DEVICES LD_LIBRARY_PATH='' PYTHONPATH=python ...`：既有 `test_utils.py` 对空 CUDA_VISIBLE_DEVICES 使用 `[0]`，因此不能设为空字符串；测试本身显式使用 CPU 张量。
- 实权重 runtime smoke 在目标 Torch 2.14.1 成功，重复请求的动作与原始视频逐元素一致。加载 11.23 s；冷请求 4.628 s；第二次 0.320 s（sample 0.274 s），峰值 allocated 34.36 GB。仅是 256 高、4 action、5 frame、3 step 的小请求，不外推真实部署性能。
- 对当前 Cosmos/Torch 2.10 smoke：active physical max abs 0.015625、MAE 0.002693、relative L2 0.008685；normalized relative L2 0.009665；uint8 video MAE 1.358、RMSE 6.556、max abs 128。**尚不判数值验收通过**，正在定位首个中间量分歧与跨版本 BF16 影响。
- 证据：`artifacts/pwm_wam_1028/native_smoke_v2/summary.json`、`000/arrays.npz`、`001/arrays.npz`。当前 Cosmos 分层捕获已成功，失败原因是继承的 LD_LIBRARY_PATH 混入系统 Torch 库；清空后同 GPU BF16 linear 与完整捕获均成功。
- 真实 SGLang Python worker、native HTTP 与兼容 HTTP 正在分别验证，不将内核 runtime 成功当作服务接入完成。

实现与验证尚未完成。每阶段补充改动、命令/结果、失败原因、修正、提交和剩余风险。

## 2026-10-28 17:40：真实服务、矩阵与差异定位

- 接线提交 `39dc0c1c49`。真实 `DiffGenerator`/worker 与直接 runtime 的动作、normalized actions、uint8 视频逐元素一致；第二次请求 Python end-to-end 0.329 s、worker 0.318 s。
- 原生 HTTP 与 Cosmos facade：动作/normalized actions 完全一致，视频 MP4 逐字节一致；开关视频动作不变，下载/删除/404 通过。原生错误 400 与 facade 错误 422 后均恢复；facade 四个并发请求为 200/503/503/503，返回 Retry-After=1，之后合法请求恢复。
- 真实退出发现原生 HTTP 临时目录残留，补 action router lifespan 显式清理并验证目录删除；Pi05 API 10 项再次通过。补 wrapper 忽略外层采样字段的拒绝规则，避免静默吞参数。
- 更新后的 runtime CPU 13/13 通过，覆盖局部 SDPA 状态在正常/异常路径恢复、严格字段/布局校验与 provenance；新部署文档 `examples/diffusion/cosmos_pwm_wam/README.md` 已写，CLI 参数经真实解析验证，最后服务重测使用相同命令。
- 原尺寸 Libero（960×640、40 actions/41 frames）、Droid（50 Hz actions/15 FPS、100 actions）、Robotwin joint 非连续 mask、单相机 Robocasa 均原生跑通；A/B/C/D/A 的 actions、normalized actions、视频首尾逐元素一致。每个请求当前 Cosmos reference 也已生成。首轮跨环境误差保存在 `matrix_comparison_before_sdpa_fix.json`，不作为最终通过结果。

### 决策 D05：修正模型局部 attention 默认值

SGLang `runtime/platforms/cuda.py` 导入会禁用 cuDNN SDPA，改变了 source SDPA 的算法选择。冻结同层输入和权重后，纯 native core 在 Torch 2.10 和 2.14 的默认 all-backend SDPA 下均与 source 精确一致；使用 SGLang 默认时出现偏差。现在仅在 WAM forward 作用域启用四种 SDPA backend，结束/异常后恢复原状态，不修改全平台默认。该修正令 UND layer 0/35 与 source 精确一致。

VAE 仍有独立偏差：pixels、首卷积、norm1、SiLU 相同，首次分歧出现在早期 residual conv1；正在核对卷积算法/库版本与 layout。C 使用 cuDNN 9.15.1.9、S 为 9.24.0.43。benchmark=True 不能消除差异，因此未据此改生产设置。

### 决策 D06：保留同卡、同请求基线并避开外部作业

GPU0 后续出现本 namespace 不可见的外部 PID 329992，占约 45 GB，不操作它。主验证转 GPU2，分层诊断 GPU1、HTTP GPU3。

GPU2 当前 Cosmos 基线（Libero 960×640、T=40、5 steps、CFG=1.5、shift=5、无视频解码，metadata reuse 开启）：六次请求排除首轮后，SDPA median 3.430 s；FA3+compile median 2.233 s，首次 21.835 s 含编译。等待最终 SGLang 同卡测量；不宣称接入即提速。基线日志与响应位于 `artifacts/pwm_wam_1028/benchmark/`。

## 2026-10-28 17:56：数值诊断收敛与最后一轮验证

- 修复/文档提交 `a2dd639bda`。新增四个独立 manual 诊断工具，Ruff/check/format 全通过：source 捕获、native 中间量比较、纯 layer 回放、纯 VAE 回放。它们明确区分实际像素端到端与诊断用 reference-clean 注入，不进入生产依赖。
- VAE 首差异确认为同值、同 shape/stride、同 contiguous NCDHW input/weight 的早期 BF16 conv1；首卷积、norm、SiLU 均精确一致。该单卷积 283 个元素不同，max abs 0.0078125。benchmark/deterministic/TF32/layout 控制没有消除差异，不据此改全局设置或放宽验收阈值。
- 3-step smoke 注入 source clean latent 后：首 velocity、layer 0/35 的三塔输入/输出、最终物理/normalized actions 均逐元素一致；最终 video latent max abs 1.192e-7、relative L2 1.49e-8。该完整采样结果的 decode uint8 MAE 0.07887、max abs 3；这里最终 video latent 仍有上述 FP32 末位差异，不是严格相同 decode 输入。**这是隔离验证，不是原始像素 e2e strict parity 通过，也不推广为整个请求矩阵的隔离验证结果。**
- source/target VAE-only 证据分别为 `vae_layout_torch210` / `vae_layout_torch214`，cuDNN 91501 / 92400。目标 Torch 编译要求 cuDNN 9.24，不强行注入旧 9.15 库。详细命令与限制见 `artifacts/pwm_wam_1028/numerical_diagnosis.md`。
- 七个权重分片 SHA-256 审计完成：总计 32,364,372,752 bytes，文件 `artifacts/pwm_wam_1028/weight_sha256.json`；实际依赖位置/版本在 `environment.json`。独立环境 python-dateutil 已从 3.0 调整为仓库要求的 2.9.0.post0。
- 最终审查补启动资产/配置 guards，避免错误 VAE/tokenizer、改变的推理配置或越界 projection ID 静默进入 GPU；资产配置 hash 不等同于完整权重绑定。
- 视频下载用锁内打开的句柄，避免下载与 DELETE/过期清理竞态；正常发送、未开始 body 即取消、ASGI send 异常均验证句柄关闭，已打开后删除仍能读完，新 GET 返回 404。服务 CPU 更新为 9/9。
- 最后 GPU 矩阵、最终 HTTP 与同卡 SGLang benchmark 进行中；当前结论仍明确保留跨 Torch/cuDNN 的 VAE e2e 差异。

## 2026-10-28 18:08：最终矩阵与下载生命周期验证

- 诊断工具提交 `9a2a260afe`。最终 Python/worker 矩阵七次请求全部成功，覆盖 smoke、原尺寸 Libero、Droid 独立频率/CFG norm、Robotwin 非连续 mask、Robocasa 单相机、缺省 30 steps 和 Libero 重放。跨 domain/shape/seed 后 Libero 的 actions、normalized actions、uint8 视频仍逐元素一致；证据 `final_python_matrix/summary.json`。
- 最终原生服务按 README CLI 启动成功：wrapper、标准 action envelope、raw JSON 三种输入动作一致；非法 wrapper seed、runtime.cuda_graph、顶层 n 均返回 400；开关视频不改变动作。真实视频下载/删除/404 通过，保留视频后 SIGTERM 确认文件和临时目录都清理。
- 下载生命周期修复单独提交：锁内打开句柄，DELETE/过期清理不会使已开始的下载失效；ASGI 正常完成、发送异常及 body 尚未启动时取消都关闭句柄。9 项 serving CPU 测试通过，最终 native HTTP 重测通过。兼容接口最终复测仍在进行。
- 独立验证环境 `.venv` 的核心包安装在本仓库；`pwm_validation_base.pth` 只读追加相邻环境的第三方 site-packages（包括 msgspec/FastAPI/uvicorn），不包含 Cosmos/vLLM 仓库代码。`environment.json` 记录主要包实际位置，补充 `environment_all_packages.json` 记录完整首选 distribution 版本/路径。这不是干净容器全量安装的验证，保留这一环境边界。
- 共享 NFS 使最后几个进程的 Python import/spawn 明显变慢；此时间与 GPU 请求延迟分开记录，不作为模型性能结论。

### 决策 D07：保持测试依赖与现有 CI 分组一致

31 项测试的张量计算在 CPU 上执行，但模型测试需要 Diffusers。既有 hosted CPU workflow 只安装 `python[dev]`，不包含该可选依赖；因此将三份测试注册到已有安装 diffusion 依赖的 `base-b-test-diffusion-unit-1-gpu-h100`，沿用其 `register_cuda_ci` 入口，不给所有 CPU CI 增加大型依赖，也不以 skip 隐藏测试。实际测试仍在 CPU 上运行，私有实权重 GPU 矩阵保留 manual harness。

## 2026-10-28 18:20：实现与验证收尾

- 生命周期提交 `66643efca1`；资产/配置 guards、依赖正确的 CI 注册和风格修正提交 `0224fcb2ba`。收尾检查将其余五份新文件的 import 分组对齐仓库 isort，仅导入分组/空行变化。
- 31 项 CPU 张量测试再次全部通过（2.289 s 测试体）；后续仅 CI 注册与导入分组调整。已有 CI collect/validate/filter 确认三文件全进入目标 diffusion unit suite，无 skip。22 份新 Python 文件 Ruff、format、isort 全通过；`git diff --check` 干净。结果 `final_style_check.log`。
- 最终 raw-pixel matrix 七请求六场景对当前 Cosmos：所有 shape/离散合同相等、所有 inactive 维为零；active physical relative L2 0.007525–0.016698，normalized 0.008812–0.023762，video MAE 0.58108–2.23442。完整指标 `final_matrix_comparison.json`；不宣称 strict E2E parity。
- 数值独立复核收紧证据措辞：跨 Torch 精确三塔回放指 layer 0；完整网络+solver 的精确动作证据限于 3-step smoke 的 reference-clean 注入；该实验 decode 前最终 latent 仍有 1.192e-7 差异。严格同 source latent 的纯 VAE decoder BF16 输出 max abs 0.021484375、relative L2 0.00260766。
- 最终 Python /native HTTP /facade 的同 smoke 动作和 normalized actions 逐值相同，13 项 metadata/资产摘要相同；两 HTTP 的 MP4 逐字节一致，86,268 bytes。错误恢复、四并发 1×200+3×503/Retry-After、视频开关、下载/删除/404、两服务保留视频后 SIGTERM 清理全部通过。复现命令、脚本、原始响应、数组和日志见 `http_validation/REPRODUCE.md` 与 `http_validation/final/`。
- 同 GPU2 六次顺序性能对照：当前 Cosmos SDPA warm median 3.429874 s，FA3+compile 2.232923 s；SGLang Python 3.239146 s，direct runtime 3.227952 s。Python 比已有 FA3+compile 慢 45.1%；本次不宣称提速。Python 同请求调用耗时减 worker 耗时的中位数 10.1 ms，两个入口动作精确一致、各自重复精确一致。
- direct runtime 峰值 allocated 34,770,594,816 bytes、reserved 35,383,148,544 bytes；计时 960×640、40 actions、41 video frames、5 steps、guidance=1.5、shift=5、seed=42、不解码视频。完整数据见 `benchmark/summary.json`。带视频矩阵/HTTP 的耗时分开列于报告，没有把 Python 父进程零显存误报为 worker 显存。
- 全部本任务 GPU 推理与 HTTP 服务/worker 已退出。共享节点收尾观察到其他外部 GPU 作业；未触碰它们，性能未宣称独占节点或统计显著性。
- 最终报告补齐支持范围、文件落点、环境 overlay、资产摘要边界、数值未验收项、完整复现入口与本地提交。私有数据/权重/环境未加入 Git；无远端推送、无其他 checkout 改动。

## 2026-10-28 18:34：继续完整目标的正确性与性能验收

上一阶段属于 progress：原生实现、服务验证和性能基线有实际代码及证据，但完整目标尚未完成；原始像素数值差异与单请求优化仍需推进。当前 HEAD 为 `d38b24d0d4`，工作区原先干净，未发现遗留本任务运行进程。

### 决策 D08：先消除确定不依赖采样状态的重复计算

UND 只对自身做 causal attention，不读取 VIDEO/ACTION，也不读取 timestep。因此同请求内每个 CFG 分支可以预计算 UND 各层 raw K/V 和最终 hidden；保留 raw K，在 VIDEO/ACTION 消费时仍执行各自原有 k_norm/RoPE。缓存为 caller-owned NamedTuple，不写 model/global；条件和无条件分支分别构建，未使用的 CFG 分支不构建。异常/返回后释放，不做跨请求缓存。

新增 `enable_und_cache`，启动配置为默认来源，request 可严格 bool 覆盖；HTTP runtime 字段与 raw 同名同时提供时拒绝。准确性和性能证据齐备之前默认关闭，oracle 路径继续可用。新增 CPU 测试覆盖多层 FP32/BF16 全三流 exact、UND 实际计算次数、缓存不变性和释放、CFG interval/unused branch、异常后重建。

手动 harness 增加同进程 off/on/on/off 对照、显式 warmup 和额外 profiler capture；profile 耗时不混入正式 latency，输出逐元素检查不能省略。使用 diffusion performance/benchmark-profile 技能，以同 GPU、固定请求、保存动作/视频和源版本为条件。

### 决策 D09：用同库环境隔离移植误差与跨库误差

在当前 Cosmos Python 3.13/Torch 2.10 环境运行原生 leaf 实现，以原始像素对照当前 Cosmos 全轨迹。只在 `/tmp` 安装缺少的 msgspec wheel，不修改原环境；诊断工具用 namespace shells 避开无关 SGLang 父包初始化，加载逐文件 SHA-256 记录的原生 leaf snapshot，不修改其计算实现。该实验只证明 leaf 实现在同环境下的语义，不替代目标环境真实 SGLang worker 的部署验证。

GPU0 初查空闲，导入期间外部任务占约 50 GB，第一次诊断加载 OOM；本任务退出并保留日志，没有采样结果或性能数据。外部任务随后自行结束，复查空闲后才重试。其他三卡有外部任务，未操作它们。

18:41 更新：UND cache 版本 CPU 验证完成，model 9/9、runtime 17/17、serving 12/12，共38项通过。配置/API/内部cache与诊断benchmark工具单独本地提交；默认关闭，GPU一致性和性能暂不宣称通过。第二次同环境诊断遇外部新任务，supervisor检测到第二GPU进程后仅终止本任务，避免第二次OOM；脚本/环境/原生leaf snapshot SHA与资源审计保留。

## 2026-10-28 18:55：同环境正确性闭环与 UND 缓存实测

- 请求内 UND 复用及显式开关提交 `4736596b7c`，当前默认关闭，保留逐步骤重新计算的数值基线。模型、runtime、serving CPU 测试合计 38 项通过。
- 当前 Cosmos 与未改写数学运算的 native leaf snapshot 在相同 Python 3.13 / Torch 2.10 / cuDNN 9.15 下，六个原始像素场景的全宽 Action128、normalized actions、uint8 视频全部逐位一致。没有注入 reference clean latent；隔离只跳过无关 serving 包初始化，离线诊断不等同于目标环境部署。命令、leaf SHA 和加载路径见 `same_environment/native_matrix_gpu1/isolation_manifest.json`。
- 新采集 smoke 的 52 个过程边界中 49 个整张量精确一致。另 3 个张量仅在固定条件帧的 812 个坐标有 `1.1920929e-7` 的 FP32 末位差异；生成视频坐标和动作坐标的差异均为零。native 每步将固定坐标恢复为 clean，source solver 在这些坐标保留舍入误差。没有将这三张量重新标为 exact。详见 `same_environment/native_smoke/{roundoff_location,fixed_coordinate_check}.json`。
- 目标 Torch 2.14 / cuDNN 9.24 对 source 2.10 / 9.15 的六场景及 Libero 重放视频，每帧均通过固定 SSIM≥0.92、PSNR≥24 dB。最差 DROID frame 32 为 0.9327856 / 24.912619 dB。方法与逐帧值见 `final_video_quality.json`；此结论不定义动作单位容差或机器人任务成功率。
- GPU2/H10080GB 固定 Libero 960×640、40 actions/41 frames、5 steps、CFG1.5、shift5、seed42、无 decode：各模式预热一次，执行三组 off/on/on/off。off 六次平均 3.257093 s，on 六次平均 3.154454 s，均值延迟下降 **3.1512%**；全部动作及 normalized actions exact。额外两次 profiler 输出也 exact。缓存峰值 allocated 增加 36,954,112 bytes；后续独立 CPU-head 优化不包含在这组数据中。
- 本组运行加载的是 `4736596b7c`；监督器确认目标 GPU 仅有该推理进程，无外部进程冲突，退出码 0。完整命令、GPU 审计、12 份输出、两份 trace 和统计保存在 `und_cache_validation/`。没有根据带 profiler 的耗时计算提速。
- 使用 `llm-torch-profiler-analysis` 的单 trace 分析整理两条 native trace。算子表显示 GEMM 38.16%、cuDNN attention 28.08%，余下主要为逐元素、cast 和 norm。自动分析器把 `nvjet` 模式提示为 FP8 替换候选，但本模型实际 BF16，**该提示不适用**；未据此引入量化或声称可直接复用 FP8 kernel。单 trace 不能建立可重叠执行的收益结论。
- 多场景 cache-on 视频与真实 Python/HTTP/facade 验证、未使用文本输出头的 CPU 常驻显存实测继续进行。仍不声称已超过当前 Cosmos FA3+compile 基线。

## 2026-10-28：未使用文本输出头的独立显存验证

- WAM 不生成文本 logits。严格 loader 继续校验并保留独立的 `language_model.lm_head.weight`，仅将它留在 CPU；没有删除 key、留下 meta 参数或与实际使用的 embedding 绑定。全部 1,214 个 checkpoint 张量及完整 `state_dict` 仍保留，参与 WAM 推理的参数均在 CUDA。UND 缓存数学路径及默认关闭策略不变。
- `loaded_tensor_bytes` 仍表示全部加载张量，共 32,364,191,504 bytes；新增 `cuda_resident_tensor_bytes=31,119,531,792` 与 `unused_text_head_cpu_bytes=1,244,659,712`。实际 CUDA checkpoint 权重精确减少 **1,244,659,712 bytes（1.1591796875 GiB）**，不是按权重文件大小估算。VAE 与激活不计入这些模型加载字段。
- GPU3/H10080GB 一次加载执行现有手工 harness：`--route runtime --und-cache compare --warmup 1 --repeat 1`，无 profiler。输入为既有 `benchmark/request.json`（Libero 960×640、40 actions、41 frames、5 steps、CFG1.5、shift5、seed42、无视频 decode）。四次 off/on/on/off 的全宽 `[40,128]` physical/normalized arrays 均与 `4736596b7c` 的匹配 ABBA 输出逐元素一致，离散合同相等，inactive 槽位全部为零。本项没有比较 decoded 视频。
- 峰值 allocated：cache off 33,523,971,072 bytes，on 33,560,925,184 bytes。相对之前 GPU2 的匹配输出，均减少 1,245,708,288 bytes，即文本头大小再加 1 MiB。额外 1 MiB 是本次跨 GPU 运行的分配差异，未归因于模型权重；也不根据一轮跨卡测量宣称延迟收益。
- 设备审计包装器只在原 runtime 初始化返回后读取参数设备、字节数和加载信息，然后恢复原初始化函数；没有修改权重或 forward。全 runtime（含 VAE）只有该文本头参数位于 CPU，其余参数均 CUDA。成功进程的 namespace PID2972677 对应 NVML PID721834；监督器未发现该卡第二进程，退出码 0，GPU3 已释放。首次尝试因 PID 命名空间映射被自己的监督器误判并停止，尚无推理输出，保留在 `unused_head_cpu_validation/attempt_01_pid_namespace/`；未停止其他任务。
- 复现命令、git/diff/source SHA、初始化设备明细、峰值与全部四份输出位于 `artifacts/pwm_wam_1028/unused_head_cpu_validation/`；`comparison.json` 分开记录权重减少量与实际峰值减少量。模型 CPU 单测 9/9 通过（0.548 s），新增检查覆盖完整 state_dict 无损/不绑定、used embedding 仍去指定设备、CPU head 的 shape/dtype 损坏仍拒绝，以及加载字节分类。仓库 isort、选定 Ruff 规则、format 和 diff 检查通过。

## 2026-10-28 19:15：完整回归、服务对照与最终审查

- 六个目标环境场景完成24次off/on/on/off rollout，54组全宽physical/normalized/video比较全部exact；source leaf前后SHA一致。证据 `und_cache_matrix/{summary,acceptance}.json`。Python真实worker四次、native HTTP四次、compat三次也通过开关及继承参数验证，11组动作数组与先前最终结果一致，7份MP4逐字节一致，下载/删除/404及服务关闭清理成功；证据 `und_cache_serving/cross_route_summary.json` 和两入口各自 `shutdown.json`。
- 短请求质量矩阵单组ABBA出现缓存慢17%的波动，没有删掉该数据。为解决此疑点，GPU1对同一3-step带视频smoke每mode预热3次，再执行5组ABBA。off十次平均0.313904 s、on十次平均0.272644 s，改善13.1443%；全部动作、normalized和uint8视频exact。目标卡无外部并发，进程正常退出；证据 `und_cache_short_request/comparison.json`。默认缓存仍关闭，部署可显式开启，不外推所有请求的提速比例。
- 最终优化后的新增CPU回归38/38通过；已有Pi05、Cosmos3、registry使用pytest完整运行，163 passed、59 subtests passed。先前用unittest尝试收集pytest函数返回“0 tests”，不计为通过；已用正确runner重跑。日志 `final_cpu_optimized.log`、`final_existing_optimized.log`。
- 28份Python文件通过仓库pre-commit选定Ruff规则F401/F821/F823/UP037、format、isort与diff检查。全规则Ruff最初还报告已有registry旧typing以及与独立isort配置的排序差异，没有为了此任务重构无关代码；实际hook结果在 `final_optimized_hooks.log`，原完整诊断也保留。
- Cosmos外部HEAD在验证期间推进：初始3d2aba，fresh smoke manifest明确42bea124；早期六场景缺逐run commit stamp，不能全归属初始HEAD。只读审计保存20份关键文件blob/SHA与相关源码树等值证据，变化是未调用服务优化/诊断工具。stderr+reflog的逐场景revision归属只标“重建”，证据在 `same_environment/source_revision_audit.json`，说明提交455d63f168。没有修改或回退外部checkout。

### 决策 D10：保留完整权重审计，只让未使用文本头留在CPU

显存优化独立提交29495faae6，全部checkpoint张量仍严格检查，只有WAM不计算的独立lm_head留CPU；不绑embedding、不改变网络精度。精确CUDA权重减少量与跨卡峰值分配差异分开报告。UND缓存仍是显式opt-in，保留直接数值基线。

### 决策 D11：修复最终审查发现的Python参数合同缺口

HTTP adapter已有白名单，但Python generate_action直接创建模型sampling params。ActionSamplingParams中generator_device/save_output/return_file_paths_only/max_sequence_length等字段此前可被接受却不进入WAM推理，raw与外层显式seed/guidance等冲突也可能被setdefault静默忽略。width/height/runtime并非该独立dataclass字段，本来会TypeError，最初审查推断已纠正。README一处runtime.enable_und_cache包装适用于HTTP，Python应使用直接采样字段。

由WAM sampling层基于显式传参来源拒绝不支持字段和双来源冲突，检查factory单独处理的diffusers_kwargs；不改通用action模型行为。不改变数学路径，已完成GPU数值证据继续有效；补定点CPU负例、有效参数正例及一次真实Pythonworker验证。完整目标在修复和复核完成前保持active。

## 2026-10-28 19:30：最终接口修复验证及交付

- Python参数合同补丁独立提交 `f616e8eb13`，只修改WAM sampling、对应serving测试及部署README。未消费的通用字段显式报错；factory单独处理的diffusers_kwargs也被拒绝；raw/直接字段不同值冲突拒绝，同值重复兼容，canonicalization后的重复调整稳定。没有修改通用action模型或三塔数学路径。
- 最后serving CPU 15/15通过（3.297s），新增3个测试方法覆盖真实factory、未支持字段、双来源冲突、相对路径/prompt文件/extra规范化及重复调整。结合此前model9/runtime17，本轮新增回归合计41项均通过。已有完整Pi05/Cosmos3/registry回归为163 passed +59 subtests passed。
- 最后一次真实Python worker复测：12类非法请求均在入队前拒绝，记录的queued_requests全部0；随后raw-off与完整direct-kwargs-on两次有效请求成功，physical actions、normalized actions、uint8视频与彼此及旧基线逐元素一致。证据 `und_cache_serving/python_admission/{summary,shutdown}.json`。父进程、worker和resource tracker均已退出。
- 最后28份Python变更的仓库Ruff hook规则、format、isort、git diff检查全部通过，结果 `final_admission_style.log`。报告本地链接检查通过。未因修复输入校验而重复已经通过且计算路径未变化的GPU数值矩阵。
- 独立审查完成：生产路径原生执行、请求状态隔离、完整权重保留、视频生命周期和参数错误处理均无剩余阻塞。报告修正了“全权重数值检查”的过强措辞，区分key/shape/dtype校验与原值保留；明确早期comparison文件内source commit字段只是初始revision，不覆盖逐场景来源审计。
- 本轮单GPU/B1原生接入、同环境数值对齐、跨环境误差定位/视频质量验证，以及有实测收益的请求内UND复用和显存优化已完成。缓存默认仍关闭；保留未缓存路径、source reference、ABBA/profile/过程对齐工具，供后续优化回归。跨Torch/cuDNN的动作差异及尚未超过已测Cosmos FA3+compile性能均如实保留，不宣称机器人闭环质量或未实现的多卡/compile能力。
- 最终交付入口为 `plans/report_1028.md`、本记录与 `examples/diffusion/cosmos_pwm_wam/README.md`。所有代码分批提交到本地pwm_infer，私有资产和大体积证据留在ignored artifacts；无远端push/PR、无其他checkout修改、无外部进程终止。所有自建服务已停止、临时服务视频清理，完整验证输出保留供review。
