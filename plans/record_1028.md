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

## 进行中的工作

实现与验证尚未完成。每阶段补充改动、命令/结果、失败原因、修正、提交和剩余风险。
