# SPDX-License-Identifier: Apache-2.0
"""Private compile boundary for WAM's use of the existing SGLang FA3 kernel."""

import torch
from torch import Tensor


def _run_flash3(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_q: Tensor,
    cu_k: Tensor,
    causal: bool,
) -> Tensor:
    from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func

    return flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=q.shape[0],
        max_seqlen_k=k.shape[0],
        softmax_scale=q.shape[-1] ** -0.5,
        causal=causal,
        num_splits=1,
        ver=3,
    )


@torch.library.custom_op("sglang::pwm_wam_flash3", mutates_args=())
def _flash3(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_q: Tensor,
    cu_k: Tensor,
    causal: bool,
) -> Tensor:
    # Keep the kernel loader and platform checks outside Dynamo's full layer
    # graph. The implementation remains SGLang's existing FA3 op.
    return _run_flash3(q, k, v, cu_q, cu_k, causal)


@_flash3.register_fake
def _flash3_fake(q, k, v, cu_q, cu_k, causal):
    return q.new_empty((*q.shape[:-1], v.shape[-1]))


def flash3_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool = False,
    cu_seqlens: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    if cu_seqlens is None:
        raise ValueError("PWM FlashAttention 3 requires request attention metadata")
    return _flash3(q, k, v, *cu_seqlens, causal).flatten(1)


def validate_flash3(device: torch.device, dtype: torch.dtype) -> None:
    """Fail during setup if FA3 is unavailable; never select another backend."""
    if device.type != "cuda" or dtype != torch.bfloat16:
        raise ValueError("PWM FlashAttention 3 requires CUDA BF16 weights")
    if torch.cuda.get_device_capability(device)[0] != 9:
        raise ValueError("PWM FlashAttention 3 currently requires a Hopper GPU")
    # Exercise the actual loader/kernel now, before accepting a request. The
    # architecture check also prevents the shared wrapper's FA2 fallback.
    with torch.inference_mode():
        q = torch.zeros((1, 32, 128), dtype=dtype, device=device)
        kv = torch.zeros((1, 8, 128), dtype=dtype, device=device)
        cu = torch.tensor([0, 1], dtype=torch.int32, device=device)
        _flash3(q, kv, kv, cu, cu, True)
