# 三塔 PWM WAM 接入 SGLang：结论

状态：实施中，尚未通过完整验收。此文件会随实际结果更新；不将历史 vLLM/Cosmos 测试作为本次通过证据。

目标：单 GPU/B1 的原生 SGLang WAM 推理、Action128 动作及可选未来视频、可复现的正确性和性能基线。实施记录见 [record_1028.md](record_1028.md)，设计见 [plan.md](plan.md)。

## 当前已确认

- 接入位置为主仓库 `sglang.multimodal_gen`，无需独立 Omni 引擎。
- 已有 native 三塔代码和固定 HF artifact 可复用；必须适配 SGLang worker/API，并修正与当前 Cosmos 的 mask、媒体和 seed 差异。
- 当前存在空闲 H100 80GB 可用于验证；依赖环境仍在准备。

## 验收状态

| 项目 | 状态 |
| --- | --- |
| 严格权重加载与合同验证 | 待实现/验证 |
| 原生三塔和联合 UniPC | 待实现/验证 |
| 单元测试与现有 API 回归 | 待执行 |
| 当前 Cosmos 数值与视频对照 | 待执行 |
| SGLang worker / Python / HTTP | 待实现/验证 |
| 性能与显存基线 | 待执行 |
| 部署文档与可复现命令 | 待整理 |

没有完成结论或性能收益声明。
