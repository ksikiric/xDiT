"""Feature-gated MoRI fused all-to-all integration for Wan USP."""

import os

import torch
import torch.distributed as dist


_FUSED_A2A_MODE = int(os.environ.get("XFUSER_FUSED_A2A", "0"))
_FUSED_A2A_QUANT = os.environ.get("FUSED_A2A_QUANT", "0") == "1"
_FUSED_A2A_INTERLEAVE = os.environ.get("XFUSER_FUSED_A2A_INTERLEAVE", "0") == "1"
if _FUSED_A2A_QUANT:
    os.environ["FUSED_A2A_HADAMARD"] = "1"
if _FUSED_A2A_MODE not in (0, 1, 2):
    raise ValueError("XFUSER_FUSED_A2A must be 0, 1, or 2")

_FUSED_A2A_CODECS = tuple(
    os.environ.get(f"FUSED_A2A_CODEC_{role}", os.environ.get("FUSED_A2A_CODEC", "e4m3"))
    for role in "QKV"
)
_FUSED_A2A_PACKED = (
    _FUSED_A2A_MODE == 1
    and _FUSED_A2A_QUANT
    and _FUSED_A2A_CODECS == ("mxfp8", "mxfp8", "e4m3")
    and os.environ.get("FUSED_A2A_QUANT_RETURN", "fp8") == "fp8"
)
if _FUSED_A2A_PACKED:
    # All three roles must use the consumer ABI, not the default wire layout.
    for _role in "QKV":
        os.environ[f"FUSED_A2A_V4_OUTPUT_{_role}"] = "1"
    os.environ["FUSED_A2A_V4_OUTPUT"] = "1"
    os.environ["FUSED_A2A_SPLIT"] = "1"

_MORI_GROUP_KEY = None
_MORI_CPU_GROUP = None
_OP_CACHE = {}


def get_fused_a2a_mode():
    """Return 0 for RCCL, 1 for transport-only, or 2 for full fusion."""
    return _FUSED_A2A_MODE


def use_fused_a2a_packed():
    return _FUSED_A2A_PACKED


def use_fused_a2a_interleave():
    return _FUSED_A2A_INTERLEAVE and _FUSED_A2A_SIDESTREAM and _FUSED_A2A_PACKED


@torch.compiler.disable
def fused_a2a_input_role(input, role, group, rank, pending=None):
    """Submit one already-normalized sequence-major role without a compute join."""
    if not use_fused_a2a_interleave():
        raise RuntimeError("per-role input requires packed sidestream interleave")
    if role == 0:
        in_op, _ = _get_ops(group, rank, tuple(input.shape), input.dtype, input.device)
        side = _input_side_stream(input.device)
        consumer_done = _INPUT_CONSUMER_DONE.get(input.device)
        if consumer_done is not None:
            side.wait_event(consumer_done)
        pending = {"op": in_op, "side": side, "inputs": [], "next_role": 0}
    if pending is None or pending["next_role"] != role:
        raise ValueError("interleave requires Q, K, V in order")
    side = pending["side"]
    producer_done = torch.cuda.Event()
    producer_done.record(torch.cuda.current_stream(input.device))
    side.wait_event(producer_done)
    # Raw-pointer launchers do not inform the caching allocator about side reads.
    input.record_stream(side)
    pending["inputs"].append(input)
    pending["outputs"] = pending["op"].submit_role(role, input, stream=side)
    pending["next_role"] += 1
    if pending["op"]._epoch <= 1:
        print(
            f"[XFUSER_FUSED_A2A_INTERLEAVE rank={rank}] role={'QKV'[role]} "
            f"epoch={pending['op']._epoch} side={side.cuda_stream}",
            flush=True,
        )
    return pending


def fused_a2a_pad_multiple(world_size):
    if _FUSED_A2A_PACKED or os.environ.get("FUSED_A2A_PAD128", "0") == "1":
        return world_size * 32
    return world_size


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


def _get_ops(group, rank, shape, dtype, device, softmax_scale=None):
    from aiter.ops.flydsl.kernels.fused_a2a_intranode_op import (
        FusedA2AIntraNodeOp,
        FusedA2AOutIntraNodeOp,
    )

    ranks = _group_ranks(group)
    group_key = (id(group), ranks)
    device_key = (device.type, device.index)
    b, s_local, h_total, d = shape
    if softmax_scale is None:
        softmax_scale = d ** -0.5
    key = (group_key, rank, device_key, dtype, b, s_local, h_total, d, softmax_scale)
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
            return_mode="fp8" if _FUSED_A2A_PACKED else "bf16" if _FUSED_A2A_QUANT else None,
            softmax_scale=softmax_scale if _FUSED_A2A_PACKED else None,
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
    softmax_scale=None,
    pending=None,
):
    """Run the fused in-hop from USP head-major views."""
    sequence_major = tuple(tensor.transpose(1, 2) for tensor in (query, key, value))
    if not all(tensor.is_contiguous() for tensor in sequence_major):
        raise ValueError(
            "fused A2A requires Q/K/V backed by contiguous [B,S_local,H,D] tensors"
        )

    q, k, v = sequence_major
    in_op, _ = _get_ops(group, rank, tuple(q.shape), q.dtype, q.device, softmax_scale)
    if pending is not None:
        if pending["op"] is not in_op or pending["next_role"] != 3:
            raise ValueError("interleaved input must finish the same op's Q/K/V trio")
        transport_done = torch.cuda.Event()
        transport_done.record(pending["side"])
        torch.cuda.current_stream(q.device).wait_event(transport_done)
        outputs = pending["outputs"]
    else:
        outputs = in_op(q, k, v, norm_q, norm_k, cos, sin)
    b, s_local, h_total, d = q.shape
    world_size = dist.get_world_size(group)
    if _FUSED_A2A_PACKED:
        outputs, (q_scales, k_scales, v_scales) = outputs
        output_shape = (b, world_size * s_local, h_total // world_size, d)
        scale_shape = (*output_shape[:-1], d // 32)
        return (
            tuple(output.view(output_shape) for output in outputs),
            (q_scales.view(scale_shape), k_scales.view(scale_shape), v_scales),
        )
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
