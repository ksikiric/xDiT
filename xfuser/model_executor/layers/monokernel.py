"""Self-attention input through AITER's pre-attention monokernel.

With a fused AITER INT8 or MXFP6 Q/K/V projection and Attention A2A active, one persistent
launch per call runs the projection with the model's Q/K prep, quantizes into the Attention
A2A profile and sends the Ulysses all-to-all from inside the kernel
(``PreAttentionMonokernel``, on its INT8 or A6W6 host GEMM). The packed MHA then reads the
received buffers as it reads Attention A2A's.

A model supplies its prep (norm, RoPE, Hadamard), its projection's bias and Q/K norm weights,
and per-token RoPE tables [rows, 64]. XFUSER_MONOKERNEL=1 enables it.
"""
import os
from typing import Optional

import torch

from xfuser.logger import init_logger

logger = init_logger(__name__)

MONOKERNEL = os.environ.get("XFUSER_MONOKERNEL", "") in ("1", "2")
# Q/K Hadamard placements the GEMM epilogue reproduces; others take the Attention A2A path.
MONOKERNEL_HADAMARD = ("preprocess", "none")
try:
    from aiter.ops.flydsl.attention_a2a_profiles import ATTENTION_A2A_PROFILES

    MONOKERNEL_PROFILES = tuple(ATTENTION_A2A_PROFILES)
except ImportError:
    MONOKERNEL_PROFILES = ()
HEAD_DIM = 128
# Ranks own whole 128-token tiles of the tiled K/V layouts (MXFP4 and MXFP6 K, MXFP4 and
# packed MXFP6 V).
TILE_TOKENS = 128
# The session's call counter is host state, so Inductor runs the op between CUDA graph
# partitions instead of capturing it.
_HOST_STATE = (torch.Tag.cudagraph_unsafe,) if hasattr(torch.Tag, "cudagraph_unsafe") else ()
_SESSIONS = {}
_COLUMNS = {}


def monokernel_rows_multiple(profile: str) -> int:
    """Rows per rank the profile's monokernel layouts need."""
    from aiter.ops.flydsl.attention_a2a_profiles import attention_a2a_profile

    _, k, v = attention_a2a_profile(profile).codecs
    return TILE_TOKENS if k in ("mxfp4", "mxfp6") or v in ("mxfp4", "mxfp6_p") else 1


def _session(rows, in_features, heads, prep, eps, profile, valid_kv_len, gemm):
    key = (rows, in_features, heads, prep, eps, profile, valid_kv_len, gemm)
    session = _SESSIONS.get(key)
    if session is None:
        from aiter.ops.flydsl.pre_attention_monokernel import PreAttentionMonokernel
        from yunchang.globals import PROCESS_GROUP

        from xfuser.core.distributed import get_ulysses_parallel_rank, get_ulysses_parallel_world_size

        session = PreAttentionMonokernel(
            rank=get_ulysses_parallel_rank(),
            world_size=get_ulysses_parallel_world_size(),
            rows=rows,
            K=in_features,
            heads=heads,
            profile=profile,
            group=PROCESS_GROUP.ULYSSES_PG,
            prep=prep,
            valid_kv_len=valid_kv_len,
            eps=eps,
            gemm=gemm,
        )
        _SESSIONS[key] = session
        logger.info(
            "monokernel session: %s host, %d rows per rank, %d heads, K %d, profile %s, %s",
            gemm, rows, heads, in_features, profile, prep,
        )
    return session


def _columns(weight_scale, bias, norm_q_weight, norm_k_weight, heads, hadamard, gemm):
    """Per column of the fused Q/K/V weight: scales (INT8; ones for the A6W6 host, whose
    MFMA applies the block scales), biases, then the norm weights times 128 ** -0.5 for the
    epilogue's Hadamard (ones for V). A per-head norm weight [128] is repeated over the
    heads. The parameters live as long as the model, so their addresses name the module."""
    tensors = (weight_scale, bias, norm_q_weight, norm_k_weight)
    key = tuple(None if t is None else t.data_ptr() for t in tensors) + (hadamard, gemm)
    columns = _COLUMNS.get(key)
    if columns is None:
        if gemm == "a6w6":
            parts = [torch.ones(3 * heads * HEAD_DIM, device=weight_scale.device)]
        else:
            parts = [weight_scale.float().reshape(-1)]
        if bias is not None:
            parts.append(bias.float().reshape(-1))
        if norm_q_weight is not None:
            gain = HEAD_DIM**-0.5 if hadamard else 1.0
            for weight in (norm_q_weight, norm_k_weight):
                weight = weight.float().reshape(-1)
                if weight.numel() == HEAD_DIM:
                    weight = weight.repeat(heads)
                parts.append(weight * gain)
            parts.append(torch.ones(heads * HEAD_DIM, device=weight_scale.device))
        columns = torch.cat(parts).contiguous()
        _COLUMNS[key] = columns
    return columns


@torch.library.custom_op("xfuser::monokernel_attention", mutates_args=(), tags=_HOST_STATE)
def monokernel_attention(
    x_quantized: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    norm_q_weight: Optional[torch.Tensor],
    norm_k_weight: Optional[torch.Tensor],
    cos_pairs: Optional[torch.Tensor],
    sin_pairs: Optional[torch.Tensor],
    heads: int,
    norm: str,
    rope: str,
    hadamard: bool,
    eps: float,
    valid_kv_len: Optional[int],
    profile: str,
    rows: int,
    in_features: int,
    gemm: str,
) -> torch.Tensor:
    """The monokernel, then the packed MHA: [1, S_global, heads / world, 128] BF16.

    ``norm``: "rms_across_heads", "rms_per_head" or "none"; ``rope``: "interleaved",
    "rotate_half" or "none" (see AITER's PrepSpec). ``gemm``: "int8" takes INT8 activations
    with their row scales and the INT8 weight with its column scales; "a6w6" takes the
    activations as ``xfuser::mxfp6_quantize`` packs them and the weight packed in the A6W6
    host's row order (``xFuserMXFP6Linear.weight_packed_host``, ``weight_scale_host``).
    """
    from aiter.ops.flydsl.kernels.gemm_a8w8_int8_4wave import PrepSpec

    from xfuser.core.attention.backends.aiter_mha_v4.attention_a2a import (
        packed_output_views,
        trim_aiter_mha_v4_a2a_padding,
    )
    from xfuser.core.distributed import attention_a2a, get_ulysses_parallel_world_size
    from xfuser.core.distributed.attention_a2a import get_fused_a2a_codecs, launch_attention_a2a_packed

    world = get_ulysses_parallel_world_size()
    codecs = get_fused_a2a_codecs()
    prep = PrepSpec(norm=norm, rope=rope, bias=bias is not None, hadamard=hadamard)
    # Direct K/V profiles receive K and V trimmed to valid_kv_len, as AITER's push sends them.
    direct = attention_a2a._select_direct_valid_kv_len(
        attention_a2a._ATTENTION_A2A_CONFIG, valid_kv_len, rows, world
    )
    session = _session(rows, in_features, heads, prep, eps, profile, direct, gemm)
    columns = _columns(weight_scale, bias, norm_q_weight, norm_k_weight, heads, hadamard, gemm)
    results = session(
        x_quantized,
        x_scale.reshape(-1),
        weight,
        columns,
        None if cos_pairs is None else cos_pairs.reshape(-1),
        None if sin_pairs is None else sin_pairs.reshape(-1),
        weight_scale=weight_scale if gemm == "a6w6" else None,
    )
    shape = (1, world * rows, heads // world, HEAD_DIM)
    (query, key, value), scales = packed_output_views(
        tuple(result.payload for result in results),
        tuple(result.scale for result in results),
        shape,
        codecs,
        valid_kv_len=direct,
    )
    key, value, scales = trim_aiter_mha_v4_a2a_padding(key, value, scales, valid_kv_len, codecs, profile)
    return launch_attention_a2a_packed(query, key, value, *scales, profile, HEAD_DIM**-0.5).contiguous()


@monokernel_attention.register_fake
def _(x_quantized, x_scale, weight, weight_scale, bias, norm_q_weight, norm_k_weight, cos_pairs, sin_pairs, heads,
      norm, rope, hadamard, eps, valid_kv_len, profile, rows, in_features, gemm):
    from xfuser.core.distributed import get_ulysses_parallel_world_size

    world = get_ulysses_parallel_world_size()
    return x_quantized.new_empty((1, world * rows, heads // world, HEAD_DIM), dtype=torch.bfloat16)
