# 三塔 PWM WAM 接入 SGLang：实施记录

## 目标与执行约定

在当前 SGLang 主仓库原生执行三塔 WAM，首版范围为 Nano `width_only_1b`、单 GPU/B1、BF16 网络/FP32 联合 UniPC，支持 Action128 动作与可选未来视频。以当前 Cosmos 行为为 reference，复用已有 native 实现并补齐已发现语义差异。详细设计见 [plan.md](plan.md)。

用户已授权实现、测试、保守决策与频繁本地提交；尽量不打断用户。此任务不自动推送远端、不修改其他 checkout、不停止已有其他进程。结论持续更新于 [report_1028.md](report_1028.md)。所有时间为 Asia/Shanghai。

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

提供 SGLang `generate_action` 和 `/v1/actions/generate`，另以独立 `pwm_compat` 进程保留 Cosmos `/generate`、`/models`、`/info`、`/domains`、`/healthz`、`/outputs` 形状；全部通过同一原生 SGLang worker，不调用 Cosmos 服务。视频临时附件限定随机 ID、最多 128 个、保留一小时，正常退出清理。兼容层不复制 Cosmos 全部 defaults/output 持久化系统，差异在部署说明中列明。

实现与验证尚未完成。每阶段补充改动、命令/结果、失败原因、修正、提交和剩余风险。
