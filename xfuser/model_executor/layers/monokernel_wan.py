"""Wan self-attention through the pre-attention monokernel (local study code, not upstream).

With a fused AITER INT8 Q/K/V projection (XFUSER_WAN_FUSE_QKV=1) and Attention A2A active, one
persistent launch per call runs the projection with the Q/K prep epilogue, exchanges what the
profile's scales need across the Ulysses ranks, quantizes into the profile's packed layout and
moves the bytes with device-initiated SDMA. The packed MHA then reads the received bytes as it
reads Attention A2A's. The kernels are the monokernel study's (XFUSER_MONOKERNEL_SCRIPTS):

- XFUSER_MONOKERNEL=1: sdma_mono.py, per-tensor E4M3 Q/K/V (profile e4m3-e4m3);
- XFUSER_MONOKERNEL=2: sdma_mono2.py, every Attention A2A profile.

XFUSER_MONOKERNEL_DEFER (default 1), XFUSER_MONOKERNEL_V_CHUNKS (version 1, default 1) and
XFUSER_MONOKERNEL_V_TILE_AMAX (default 0 for version 1, 1 for version 2) select SdmaMono's
defer, v_chunks and v_tile_amax.
"""
import os
import sys
from typing import Optional

import torch

VERSION = os.environ.get("XFUSER_MONOKERNEL", "")
MONOKERNEL = VERSION in ("1", "2")
# Attention A2A profiles whose packed MHA inputs the monokernel produces, and the Q/K Hadamard
# placements its GEMM epilogue reproduces; anything else takes the ordinary Attention A2A path.
if VERSION == "2":
    MONOKERNEL_PROFILES = ("e4m3-e4m3", "int8-e4m3", "mxfp8-e4m3", "e4m3-mxfp6", "mxfp6-e4m3", "mxfp6-mxfp6",
                           "mxfp4-mxfp4", "mxfp6-mxfp4")
    MONOKERNEL_HADAMARD = ("preprocess", "none")
else:
    MONOKERNEL_PROFILES = ("e4m3-e4m3",)
    MONOKERNEL_HADAMARD = ("preprocess",)
_SCRIPTS = os.environ.get("XFUSER_MONOKERNEL_SCRIPTS", "/workspace/monokernel-study/scripts")
HEAD_DIM = 128
_STATE = {}
_COLUMNS = {}


def _state(rows, in_features, heads, eps, profile, valid_kv_len, hadamard):
    key = (rows, in_features, heads, eps, profile, valid_kv_len, hadamard)
    state = _STATE.get(key)
    if state is None:
        if _SCRIPTS not in sys.path:
            sys.path.append(_SCRIPTS)
        from yunchang.globals import PROCESS_GROUP

        from xfuser.core.distributed import get_ulysses_parallel_rank, get_ulysses_parallel_world_size

        common = dict(heads=heads, head_dim=HEAD_DIM, eps=eps, group=PROCESS_GROUP.ULYSSES_PG,
                      defer=os.environ.get("XFUSER_MONOKERNEL_DEFER", "1") == "1")
        rank, world = get_ulysses_parallel_rank(), get_ulysses_parallel_world_size()
        if VERSION == "2":
            from sdma_mono2 import SdmaMono

            mono = SdmaMono(rank, world, rows, in_features, profile=profile, valid_kv_len=valid_kv_len,
                            hadamard=hadamard,
                            v_tile_amax=os.environ.get("XFUSER_MONOKERNEL_V_TILE_AMAX", "1") == "1", **common)
        else:
            from sdma_mono import SdmaMono

            mono = SdmaMono(rank, world, rows, in_features,
                            v_chunks=int(os.environ.get("XFUSER_MONOKERNEL_V_CHUNKS", "1")),
                            v_tile_amax=os.environ.get("XFUSER_MONOKERNEL_V_TILE_AMAX") == "1", **common)
        state = {
            "mono": mono,
            "generation": 0,
            "planar": torch.empty(3 * rows * heads * HEAD_DIM, dtype=torch.bfloat16, device="cuda"),
            "partials": torch.empty(2, rows, 3 * heads, dtype=torch.float32, device="cuda"),
        }
        _STATE[key] = state
    return state


def _columns(weight_scale, bias, norm_q_weight, norm_k_weight, hadamard):
    """B_scale of the fused Q/K/V GEMM, once per module: column scales, biases, then the norm
    weights, times 128 ** -0.5 for the epilogue's Hadamard (ones for V). The parameters live as
    long as the model, so their addresses name the module."""
    key = (weight_scale.data_ptr(), bias.data_ptr(), norm_q_weight.data_ptr(), norm_k_weight.data_ptr(), hadamard)
    columns = _COLUMNS.get(key)
    if columns is None:
        gain = HEAD_DIM**-0.5 if hadamard else 1.0
        columns = torch.cat([
            weight_scale,
            bias.float(),
            norm_q_weight.float() * gain,
            norm_k_weight.float() * gain,
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
    from xfuser.core.attention.backends.aiter_mha_v4.attention_a2a import (
        packed_output_views,
        trim_aiter_mha_v4_a2a_padding,
    )
    from xfuser.core.distributed import attention_a2a, get_ulysses_parallel_world_size
    from xfuser.core.distributed.attention_a2a import (
        get_fused_a2a_codecs,
        get_fused_a2a_hadamard_placement,
        launch_attention_a2a_packed,
    )

    rows, in_features = x_quantized.shape
    world = get_ulysses_parallel_world_size()
    hadamard = get_fused_a2a_hadamard_placement() == "preprocess"
    codecs = get_fused_a2a_codecs()
    # Direct K/V profiles receive K and V trimmed to valid_kv_len, as AITER's push sends them.
    direct = None
    if VERSION == "2":
        direct = attention_a2a._select_direct_valid_kv_len(attention_a2a._ATTENTION_A2A_CONFIG, valid_kv_len, rows,
                                                           world)
    state = _state(rows, in_features, heads, eps, profile, direct, hadamard)
    mono = state["mono"]
    state["generation"] += 1
    mono.run(x_quantized, weight, state["planar"], x_scale.reshape(-1),
             _columns(weight_scale, bias, norm_q_weight, norm_k_weight, hadamard), cos_pairs.reshape(-1),
             sin_pairs.reshape(-1), state["partials"], state["generation"])
    shape = (1, world * rows, heads // world, HEAD_DIM)
    if VERSION == "2":
        (query, key, value), scales = packed_output_views(
            tuple(mono.payload(role) for role in range(3)), tuple(mono.scale(role) for role in range(3)), shape,
            codecs, valid_kv_len=direct)
    else:
        query, key, value = (mono.received(role).view(shape) for role in range(3))
        scales = tuple(mono.scale(role) for role in range(3))
    key, value, scales = trim_aiter_mha_v4_a2a_padding(key, value, scales, valid_kv_len, codecs, profile)
    return launch_attention_a2a_packed(query, key, value, *scales, profile, HEAD_DIM**-0.5).contiguous()


@wan_monokernel_attention.register_fake
def _(x_quantized, x_scale, weight, weight_scale, bias, norm_q_weight, norm_k_weight, cos_pairs, sin_pairs, heads,
      eps, valid_kv_len, profile):
    from xfuser.core.distributed import get_ulysses_parallel_world_size

    world = get_ulysses_parallel_world_size()
    return x_quantized.new_empty((1, world * x_quantized.shape[0], heads // world, HEAD_DIM), dtype=torch.bfloat16)
