"""Feature-gated MoRI fused all-to-all integration for Wan USP."""

import os

import torch
import torch.distributed as dist


_FUSED_A2A_MODE = int(os.environ.get("XFUSER_FUSED_A2A", "0"))
_FUSED_A2A_QUANT = os.environ.get("FUSED_A2A_QUANT", "0") == "1"
if _FUSED_A2A_QUANT:
    os.environ["FUSED_A2A_HADAMARD"] = "1"
if _FUSED_A2A_MODE not in (0, 1, 2):
    raise ValueError("XFUSER_FUSED_A2A must be 0, 1, or 2")

_MORI_GROUP_KEY = None
_MORI_CPU_GROUP = None
_OP_CACHE = {}


def get_fused_a2a_mode():
    """Return 0 for RCCL, 1 for transport-only, or 2 for full fusion."""
    return _FUSED_A2A_MODE


def _group_ranks(group):
    if hasattr(dist, "get_process_group_ranks"):
        return tuple(dist.get_process_group_ranks(group))
    return tuple(
        dist.get_global_rank(group, rank)
        for rank in range(dist.get_world_size(group))
    )


def _init_mori(group, ranks):
    global _MORI_GROUP_KEY, _MORI_CPU_GROUP

    group_key = (id(group), ranks)
    if _MORI_GROUP_KEY == group_key:
        return
    if _MORI_GROUP_KEY is not None:
        raise RuntimeError(
            "XFUSER_FUSED_A2A supports one live MoRI Ulysses group per process"
        )

    import mori.shmem as ms

    # Only this Ulysses group's members participate. This is intentionally not the
    # sequence-parallel CPU group, whose membership differs when ring degree > 1.
    # Members enter this lazy initialization together on their first fused call;
    # non-members need not participate in subgroup creation.
    _MORI_CPU_GROUP = dist.new_group(
        ranks=list(ranks), backend="gloo", use_local_synchronization=True
    )
    torch._C._distributed_c10d._register_process_group("mori", _MORI_CPU_GROUP)
    ms.shmem_torch_process_group_init("mori")
    _MORI_GROUP_KEY = group_key


def _get_ops(group, rank, shape, dtype, device):
    from aiter.ops.flydsl.kernels.fused_a2a_intranode_op import (
        FusedA2AIntraNodeOp,
        FusedA2AOutIntraNodeOp,
    )

    ranks = _group_ranks(group)
    group_key = (id(group), ranks)
    device_key = (device.type, device.index)
    b, s_local, h_total, d = shape
    key = (group_key, rank, device_key, dtype, b, s_local, h_total, d)
    ops = _OP_CACHE.get(key)
    if ops is None:
        _init_mori(group, ranks)
        in_op = FusedA2AIntraNodeOp(
            rank=rank,
            world_size=len(ranks),
            shape=shape,
            dtype=dtype,
            fuse_norm_rope=_FUSED_A2A_MODE == 2,
            quant=_FUSED_A2A_QUANT,
            return_mode="bf16" if _FUSED_A2A_QUANT else None,
        )
        out_op = FusedA2AOutIntraNodeOp(
            rank=rank,
            world_size=len(ranks),
            shape=(b, h_total // len(ranks), len(ranks) * s_local, d),
            dtype=dtype,
        )
        ops = (in_op, out_op)
        _OP_CACHE[key] = ops
    return ops


def fused_a2a_input(
    query,
    key,
    value,
    group,
    rank,
    norm_q=None,
    norm_k=None,
    cos=None,
    sin=None,
):
    """Run the fused in-hop from USP head-major views."""
    sequence_major = tuple(tensor.transpose(1, 2) for tensor in (query, key, value))
    if not all(tensor.is_contiguous() for tensor in sequence_major):
        raise ValueError(
            "fused A2A requires Q/K/V backed by contiguous [B,S_local,H,D] tensors"
        )

    q, k, v = sequence_major
    in_op, _ = _get_ops(group, rank, tuple(q.shape), q.dtype, q.device)
    outputs = in_op(q, k, v, norm_q, norm_k, cos, sin)
    b, s_local, h_total, d = q.shape
    world_size = dist.get_world_size(group)
    output_shape = (b, h_total // world_size, world_size * s_local, d)
    return tuple(output.view(output_shape) for output in outputs)


def fused_a2a_output(output, group, rank, return_sequence_major=False):
    """Run the fused out-hop and return the requested processor boundary layout."""
    if not output.is_contiguous():
        raise ValueError("fused A2A requires contiguous [B,H_local,S_full,D] output")

    b, h_local, s_full, d = output.shape
    world_size = dist.get_world_size(group)
    s_local = s_full // world_size
    h_total = h_local * world_size
    _, out_op = _get_ops(
        group,
        rank,
        (b, s_local, h_total, d),
        output.dtype,
        output.device,
    )
    head_major = out_op(output).view(b, h_total, s_local, d)
    if not return_sequence_major:
        return head_major
    return head_major.transpose(1, 2).contiguous()


# Raw-pointer kernel launches and MoRI uint8->int64 buffer views must stay outside Dynamo tracing for all fused modes.
fused_a2a_input = torch.compiler.disable(fused_a2a_input)
fused_a2a_output = torch.compiler.disable(fused_a2a_output)


# Cached outputs are persistent symmetric buffers. A same-key launch is safe only
# after the prior attention consumer has finished reading them on its stream.
