"""Wan self-attention through the pre-attention monokernel (local study code, not upstream).

With XFUSER_MONOKERNEL=1, a fused AITER INT8 Q/K/V projection (XFUSER_WAN_FUSE_QKV=1) and
Attention A2A active, one persistent launch per call runs the projection with the Q/K prep
epilogue, exchanges each role's amax across the Ulysses ranks, quantizes to per-tensor E4M3 and
moves the bytes with device-initiated SDMA (sdma_mono.py of the monokernel study, found through
XFUSER_MONOKERNEL_SCRIPTS). The packed MHA then reads the received bytes as it reads Attention
A2A's.

XFUSER_MONOKERNEL_DEFER (default 1), XFUSER_MONOKERNEL_V_CHUNKS (default 1) and
XFUSER_MONOKERNEL_V_TILE_AMAX (default 0) select SdmaMono's defer, v_chunks and v_tile_amax.
"""
import os
import sys
from typing import Optional

import torch

MONOKERNEL = os.environ.get("XFUSER_MONOKERNEL") == "1"
_SCRIPTS = os.environ.get("XFUSER_MONOKERNEL_SCRIPTS", "/workspace/monokernel-study/scripts")
HEAD_DIM = 128
_STATE = {}
_COLUMNS = {}


def _state(rows, in_features, heads, eps):
    key = (rows, in_features, heads, eps)
    state = _STATE.get(key)
    if state is None:
        if _SCRIPTS not in sys.path:
            sys.path.append(_SCRIPTS)
        from sdma_mono import SdmaMono
        from yunchang.globals import PROCESS_GROUP

        from xfuser.core.distributed import get_ulysses_parallel_rank, get_ulysses_parallel_world_size

        mono = SdmaMono(
            get_ulysses_parallel_rank(), get_ulysses_parallel_world_size(), rows, in_features, heads=heads,
            head_dim=HEAD_DIM, eps=eps, group=PROCESS_GROUP.ULYSSES_PG,
            defer=os.environ.get("XFUSER_MONOKERNEL_DEFER", "1") == "1",
            v_chunks=int(os.environ.get("XFUSER_MONOKERNEL_V_CHUNKS", "1")),
            v_tile_amax=os.environ.get("XFUSER_MONOKERNEL_V_TILE_AMAX") == "1",
        )
        state = {
            "mono": mono,
            "generation": 0,
            "planar": torch.empty(3 * rows * heads * HEAD_DIM, dtype=torch.bfloat16, device="cuda"),
            "partials": torch.empty(2, rows, 3 * heads, dtype=torch.float32, device="cuda"),
        }
        _STATE[key] = state
    return state


def _columns(weight_scale, bias, norm_q_weight, norm_k_weight):
    """B_scale of the fused Q/K/V GEMM, once per module: column scales, biases, then the norm
    weights, times 128 ** -0.5 for the epilogue's Hadamard (ones for V). The parameters live as
    long as the model, so their addresses name the module."""
    key = (weight_scale.data_ptr(), bias.data_ptr(), norm_q_weight.data_ptr(), norm_k_weight.data_ptr())
    columns = _COLUMNS.get(key)
    if columns is None:
        columns = torch.cat([
            weight_scale,
            bias.float(),
            norm_q_weight.float() * HEAD_DIM**-0.5,
            norm_k_weight.float() * HEAD_DIM**-0.5,
            torch.ones(bias.numel() // 3, device=bias.device),
        ])
        _COLUMNS[key] = columns
    return columns


@torch.library.custom_op("xfuser::wan_monokernel_attention", mutates_args=())
def wan_monokernel_attention(
    x_quantized: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    cos_pairs: torch.Tensor,
    sin_pairs: torch.Tensor,
    heads: int,
    eps: float,
    valid_kv_len: Optional[int],
    profile: str,
) -> torch.Tensor:
    """The monokernel, then the packed MHA: [1, S_global, heads / world, 128] BF16."""
    from xfuser.core.attention.backends.aiter_mha_v4.attention_a2a import trim_aiter_mha_v4_a2a_padding
    from xfuser.core.distributed.attention_a2a import get_fused_a2a_codecs, launch_attention_a2a_packed

    rows, in_features = x_quantized.shape
    state = _state(rows, in_features, heads, eps)
    mono = state["mono"]
    state["generation"] += 1
    mono.run(x_quantized, weight, state["planar"], x_scale.reshape(-1),
             _columns(weight_scale, bias, norm_q_weight, norm_k_weight), cos_pairs.reshape(-1),
             sin_pairs.reshape(-1), state["partials"], state["generation"])
    shape = (1, mono.world * rows, heads // mono.world, HEAD_DIM)
    query, key, value = (mono.received(role).view(shape) for role in range(3))
    scales = tuple(mono.scale(role) for role in range(3))
    key, value, scales = trim_aiter_mha_v4_a2a_padding(key, value, scales, valid_kv_len, get_fused_a2a_codecs(),
                                                       profile)
    return launch_attention_a2a_packed(query, key, value, *scales, profile, HEAD_DIM**-0.5).contiguous()


@wan_monokernel_attention.register_fake
def _(x_quantized, x_scale, weight, weight_scale, bias, norm_q_weight, norm_k_weight, cos_pairs, sin_pairs, heads,
      eps, valid_kv_len, profile):
    from xfuser.core.distributed import get_ulysses_parallel_world_size

    world = get_ulysses_parallel_world_size()
    return x_quantized.new_empty((1, world * x_quantized.shape[0], heads // world, HEAD_DIM), dtype=torch.bfloat16)
