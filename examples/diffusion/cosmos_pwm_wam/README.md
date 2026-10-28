# Cosmos PWM 三塔 WAM 推理

`CosmosPWMWAMPipeline` 在 SGLang 的 diffusion worker 中执行 UND、VIDEO、ACTION
三个独立专家塔，通过联合 flow matching 生成未来视频与 Action128 动作。
使用本仓库的 `DiffGenerator`、scheduler 和 action API；运行时不依赖 Cosmos
训练仓库、vLLM 或另一个 Omni 服务。

## 资产与环境

在独立服务环境中安装本仓库的 diffusion 依赖：

```bash
pip install -e 'python[diffusion]'
```

准备三个固定的本地目录：

- **模型 artifact**：原生 HF 导出的 `config.json`、
  `model.safetensors.index.json` 和索引引用的全部 safetensors 分片。
  配置必须包含 `tower_structure=three_tower`、
  `action_tower_preset=width_only_1b`、`action_tower_execution=native`、
  `action128_spec` 以及 `action128_seam` 空间合同。
- **Wan 2.2 VAE**：与训练 checkpoint 匹配、可由 `AutoencoderKLWan` 加载的本地
  Diffusers 目录；48 个 latent channels、4 倍时间与 16 倍空间压缩。
- **Tokenizer**：训练时使用的 tokenizer 文件、词表与 chat template。

使用 Cosmos 源仓库 `cosmos_framework/scripts/export_model.py` 的原生 HF 导出产物。
普通 Cosmos Diffusers pipeline 的转换产物不能代替独立 ACTION 塔的完整权重。
训练和服务环境分别维护；不要为了在线加载而引入训练框架的依赖。

加载器严格核对索引、权重名称、shape、dtype 和 Action128 合同摘要。模型路径指向
启动时选定的 artifact，不解析训练目录的 `latest`，也不热切换权重。所有三个目录
必须与同一次导出的模型兼容；任意 Wan VAE 或任意 Qwen tokenizer 不能替换它们。

## 启动原生 action 服务

创建 `wam_pipeline.json`，将路径替换为实际资产目录：

```json
{
  "vae_path": "/models/pwm-vae",
  "tokenizer_path": "/models/pwm-tokenizer"
}
```

```bash
sglang serve \
  --model-path /models/pwm-hf \
  --pipeline-class-name CosmosPWMWAMPipeline \
  --pipeline-config-path wam_pipeline.json \
  --backend sglang --served-model-name cosmos-pwm-wam \
  --num-gpus 1 --tp-size 1 --sp-degree 1 --batching-max-size 1 \
  --attention-backend torch_sdpa --enable-cfg-parallel false \
  --enable-torch-compile false --enable-breakable-cuda-graph false \
  --dit-cpu-offload false --dit-layerwise-offload false \
  --text-encoder-cpu-offload false --image-encoder-cpu-offload false \
  --vae-cpu-offload false --use-fsdp-inference false --warmup-mode off \
  --host 127.0.0.1 --port 30000
```

完整模型常驻单张 GPU。启动后用一个真实 WAM 请求完成预热。
`GET /v1/actions/metadata` 返回该 pipeline 的能力和默认参数。

## 构造请求

以下代码生成可用于 Python、原生 HTTP 和兼容服务的 `request.json`。将 domain
替换为 artifact `action128_spec.registry.domains` 中与观测/动作定义一致的完整名称，
将 `active_slots` 替换为调用方实际提供的 Action128 槽位。不能根据 normalization
mask 推断动作是否存在；有效槽位可以不连续。

```python
import json
from pathlib import Path

active_slots = list(range(10))  # 示例；按真实 Action128 布局填写
request = {
    "model_mode": "wam",
    "name": "example",
    "vision_path": "/data/observation.png",
    "prompt": "Pick up the cup and place it on the plate.",
    "image_size": 256,
    "fps": 20,
    "action_chunk_size": 40,
    "num_steps": 30,
    "guidance": 1.0,
    "shift": 10.0,
    "seed": 42,
    "extra": {
        "action128": {
            "action_domain": "<full source domain from the artifact>",
            "action_dim_mask": [slot in active_slots for slot in range(128)],
            "action_fps": 20,
            "selected_roles": ["main"],
            "camera_layout": None,
            "return_video": True,
        }
    },
}
Path("request.json").write_text(json.dumps(request, indent=2))
```

`vision_path` 接受图片、视频或 HTTP(S) URL，只读取第一帧。多摄像头输入应先拼成
一个 canvas，并提供真实的 `selected_roles` 和 `camera_layout`。两视角采用
`top/bottom`，三视角采用 `top/bottom_left/bottom_right`；每项为
`[role, location, width, height]`。布局尺寸必须与原始 canvas 一致，resize 后
caption 与空间位置会一起更新。

主要采样参数：

| 参数 | 默认值与语义 |
| --- | --- |
| `num_steps` | 30，正整数 |
| `guidance` | 1.0，范围 `[0, 7]` |
| `shift` | 10.0，范围 `(0, 100]` |
| `seed` | 省略或 null 时每个请求生成一个随机 seed；结果返回实际 seed |
| `fps` | 24，最大 30 Hz；控制视频时间位置和播放速度 |
| `extra.action128.action_fps` | 默认等于 `fps`；最大 512 Hz，独立控制动作时间位置 |
| `action_chunk_size` | 16；动作时长为 `action_chunk_size / action_fps`，不能超过 2 秒 |
| `num_frames` | 自动计算为 `4 * ceil(ceil(duration * fps - 1e-8) / 4) + 1`；显式传入必须一致 |
| `image_size` | canvas 高度，须为 16 的正整数倍；省略时将原图高度向上对齐到 16 |
| `normalize_cfg` | false；开启时对 VIDEO 和 ACTION 拼接后的速度统一做范数缩放 |
| `guidance_interval` | null；可填 `[lo, hi]`，按 UniPC 的原始 `0..1000` timestep 判断，两个端点均不包含 |
| `extra.action128.return_video` | true；false 仅省去最终 VAE decode 和视频保存，仍联合生成 VIDEO/ACTION latent |

支持 `negative_prompt`、纯文本或 JSON `prompt`、本地 `prompt_path`。JSON caption
保留任务字段，运行时用实际观测覆盖 embodiment、时长、FPS、尺寸和布局字段。
Python 的 `load_request()` 会以 JSON 所在目录解析相对媒体/提示词路径；HTTP 中的
本地路径相对于服务进程，建议使用服务器可访问的绝对路径或 URL。

未知字段与不支持的采样控制会报错。`sigma_max` 只接受 null 或 Cosmos 默认值 80，
UniPC 不使用该 EDM 参数。caption template、负向 metadata、aspect ratio 以及
save quality 兼容字段只接受源 WAM 的缺省值或 null；不支持自定义这些字段。
HTTP 视频采用缺省 `video_save_quality=10`。历史 `model`、`domain_name` 和
`output_dir` 可以随请求保留；`domain_name` 不选择 Action128 domain，客户端
`output_dir` 不控制服务端文件写入位置。

## Python API

```python
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_pwm_wam.request import load_request


def main():
    generator = DiffGenerator.from_pretrained(
        model_path="/models/pwm-hf",
        backend="sglang",
        pipeline_class_name="CosmosPWMWAMPipeline",
        pipeline_config={
            "vae_path": "/models/pwm-vae",
            "tokenizer_path": "/models/pwm-tokenizer",
        },
        num_gpus=1, tp_size=1, sp_degree=1, batching_max_size=1,
        attention_backend="torch_sdpa", enable_cfg_parallel=False,
        enable_torch_compile=False, enable_breakable_cuda_graph=False,
        dit_cpu_offload=False, dit_layerwise_offload=False,
        text_encoder_cpu_offload=False, image_encoder_cpu_offload=False,
        vae_cpu_offload=False, use_fsdp_inference=False, warmup_mode="off",
    )
    try:
        result = generator.generate_action({"wam_request": load_request("request.json")})
        print(result["actions"].shape)  # float32 [action_chunk_size, 128]
        print(result["timing_seconds"])
        # result["video_frames"]: optional uint8 [T, H, W, 3] CPU NumPy array
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
```

结果同时包含物理/源单位的 `actions`、`normalized_actions`、动作 domain、槽位 mask、
动作时间戳、实际 seed 与 caption 摘要。所有未激活槽位在模型输入、速度、solver 状态
和最终结果中都保持为零。

`checkpoint_path` 是启动时固定 artifact 的绝对路径；
`checkpoint_config_sha256` 只标识 config 文件，**不是权重 hash**。
`contract_fingerprint` 标识 Action128 合同，`loader_info` 记录加载的张量数、分片数
和字节数。审计具体权重版本时应另外保留导出 manifest 和权重校验和。

`timing_seconds` 分为 `prepare`、`vae_encode`、`sample`、`finalize` 和 `total`。
GPU 阶段边界执行同步，因此包含同步成本；不包含排队、HTTP 传输或 MP4 编码时间。

## HTTP 与 Cosmos 兼容入口

原生 API 可以直接包裹上述请求：

```python
import json
import requests

raw = json.load(open("request.json"))
response = requests.post(
    "http://127.0.0.1:30000/v1/actions/generations",
    json={"model": "cosmos-pwm-wam", "wam_request": raw},
    timeout=600,
)
response.raise_for_status()
result = response.json()
actions = result["data"][0]["action"]["values"]
video = result["data"][0].get("video")
```

也支持标准 action envelope 的 `input.observation.vision_path`、`input.task`，以及
`parameters` 中的 `action_domain`、`action_dim_mask`、`action_fps`、`action_horizon`、
`num_inference_steps`、`guidance_scale`、`flow_shift` 等参数。使用 `action_domain`
传完整源 domain；它不同于普通 Cosmos3 的 `domain_name` 标签。

视频以 `data[0].video.url` 附件返回，可从
`GET /v1/actions/videos/{id}/content` 下载，通过
`DELETE /v1/actions/videos/{id}` 删除。附件最多保留 128 个或 1 小时，服务重启后
失效；应及时保存需要长期保留的结果。

已有 Cosmos `/generate` 客户端可启动独立兼容 facade：

```bash
python -m sglang.multimodal_gen.runtime.entrypoints.action.pwm_compat \
  --model-path /models/pwm-hf --vae-path /models/pwm-vae \
  --tokenizer-path /models/pwm-tokenizer --model-name cosmos-pwm-wam \
  --host 127.0.0.1 --port 8000 --output-dir outputs/pwm-wam
curl -H 'Content-Type: application/json' --data-binary @request.json \
  http://127.0.0.1:8000/generate
```

facade 提供 `/healthz`、`/models`、`/info`、`/domains` 和 `/outputs`，返回 Cosmos
`SampleOutputs` 结构；动作在 `outputs[0].content.action`，元数据在
`outputs[0].content.action128`。视频文件从 `/outputs/{filename}` 下载。
它自行持有一个 SGLang worker；不需要同时启动原生 HTTP 服务。忙时返回 503，
客户端可在当前请求结束后重试。

## 边界与验证

当前支持单 GPU、单 observation、单输出、无状态 WAM、BF16 网络/VAE 与 FP32
联合 UniPC solver。暂不支持其他 action 模式、机器人 state/action 条件输入、
多候选、动态 batching、TP/SP/DP/CFG 并行、offload、量化、LoRA、compile、CUDA
graph、prefix cache、realtime/OpenPI WebSocket。错误请求不会变更下次请求的 seed、
mask、条件或 solver 历史。媒体需要下载/解码，因此真实媒体与声明布局的匹配检查
在 worker 中完成；纯参数和声明布局在入队前校验。

CPU 合同测试：

```bash
python test/registered/unit/multimodal_gen/test_cosmos_pwm_wam_model.py
python test/registered/unit/multimodal_gen/test_cosmos_pwm_wam_runtime.py
python test/registered/unit/multimodal_gen/test_cosmos_pwm_wam_serving.py
```

有私有资产时，使用手动 harness 验证真正的原生 Python/worker 链路：

```bash
python test/manual/test_cosmos_pwm_wam.py \
  --checkpoint /models/pwm-hf --vae /models/pwm-vae \
  --tokenizer /models/pwm-tokenizer --request request.json \
  --route python --repeat 2 --output outputs/pwm-validation
```

对照当前 Cosmos checkpoint 时，使用 `test/manual/pwm_wam_capture_reference.py` 和
`test/manual/pwm_wam_compare_intermediates.py` 保存并比较 pixels、caption/token IDs、
positions/RoPE、初始 noise、VAE latent、三塔 hidden states、速度和最终输出。
不同 Torch、CUDA、attention 实现可能产生浮点差异；记录依赖版本与逐阶段误差，
不能用 CPU 合同测试或旧 checkpoint 的报告替代当前资产的 GPU 数值验证。
