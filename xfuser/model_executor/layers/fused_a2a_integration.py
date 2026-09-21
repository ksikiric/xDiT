"""Feature-gated MoRI fused all-to-all integration for Wan USP."""

import os

import torch
import torch.distributed as dist
from torch._higher_order_ops.effects import _EffectType, _register_effectful_op, with_effects
from torch.fx.node import has_side_effect


_FUSED_A2A_MODE = int(os.environ.get("XFUSER_FUSED_A2A", "0"))
_FUSED_A2A_QUANT = os.environ.get("FUSED_A2A_QUANT", "0") == "1"
_FUSED_A2A_SIDESTREAM = os.environ.get("XFUSER_FUSED_A2A_SIDESTREAM", "0") == "1"
_FUSED_A2A_INTERLEAVE = os.environ.get("XFUSER_FUSED_A2A_INTERLEAVE", "0") == "1"
_FUSED_A2A_HADAMARD_PLACEMENT = os.environ.get(
    "XFUSER_FUSED_A2A_HADAMARD_PLACEMENT", "transport"
).lower()
if _FUSED_A2A_HADAMARD_PLACEMENT not in (
    "transport",
    "preprocess",
    "epilogue",
    "none",
):
    raise ValueError(
        "XFUSER_FUSED_A2A_HADAMARD_PLACEMENT must be transport, preprocess, "
        f"epilogue, or none; got {_FUSED_A2A_HADAMARD_PLACEMENT!r}"
    )
if _FUSED_A2A_QUANT:
    # Apply the transform exactly once. The preprocess and epilogue variants
    # run before transport; AITER must not rotate packed V4 Q/K again.
    _transport_hadamard = _FUSED_A2A_HADAMARD_PLACEMENT == "transport"
    os.environ["FUSED_A2A_HADAMARD"] = str(int(_transport_hadamard))
    os.environ["FUSED_A2A_V4_HADAMARD"] = str(int(_transport_hadamard))
if _FUSED_A2A_MODE not in (0, 1, 2):
    raise ValueError("XFUSER_FUSED_A2A must be 0, 1, or 2")

_FUSED_A2A_CODECS = tuple(
    os.environ.get(f"FUSED_A2A_CODEC_{role}", os.environ.get("FUSED_A2A_CODEC", "e4m3"))
    for role in "QKV"
)
_FUSED_A2A_PACKED_MXFP8 = (
    _FUSED_A2A_MODE == 1
    and _FUSED_A2A_QUANT
    and _FUSED_A2A_CODECS == ("mxfp8", "mxfp8", "e4m3")
    and os.environ.get("FUSED_A2A_QUANT_RETURN", "fp8") == "fp8"
)
_FUSED_A2A_PACKED_F4F4 = (
    _FUSED_A2A_MODE == 1
    and _FUSED_A2A_QUANT
    and _FUSED_A2A_CODECS == ("mxfp4", "mxfp4", "mxfp4")
    and os.environ.get("FUSED_A2A_QUANT_RETURN", "fp8") == "fp8"
)
_FUSED_A2A_PACKED = _FUSED_A2A_PACKED_MXFP8 or _FUSED_A2A_PACKED_F4F4
if (
    _FUSED_A2A_MODE
    and _FUSED_A2A_HADAMARD_PLACEMENT in ("preprocess", "epilogue")
    and not _FUSED_A2A_PACKED_MXFP8
):
    raise ValueError(
        "preprocess/epilogue Hadamard placement requires packed MXFP8 transport"
    )
if _FUSED_A2A_PACKED:
    # All three roles must use the consumer ABI, not the default wire layout.
    for _role in "QKV":
        os.environ[f"FUSED_A2A_V4_OUTPUT_{_role}"] = "1"
    os.environ["FUSED_A2A_V4_OUTPUT"] = "1"
    os.environ["FUSED_A2A_SPLIT"] = "1"

_FUSED_A2A_INTERLEAVE_BF16 = (
    _FUSED_A2A_MODE == 1
    and _FUSED_A2A_QUANT
    and os.environ.get("FUSED_A2A_QUANT_RETURN", "fp8") == "bf16"
    and all(codec in ("e4m3", "int8", "mxfp4", "mxfp6", "mxfp8") for codec in _FUSED_A2A_CODECS)
)
if _FUSED_A2A_INTERLEAVE and _FUSED_A2A_SIDESTREAM and _FUSED_A2A_INTERLEAVE_BF16:
    os.environ["FUSED_A2A_SPLIT"] = "1"

_MORI_GROUP_KEY = None
_MORI_CPU_GROUP = None
_OP_CACHE = {}
_INPUT_SIDE_STREAMS = {}
_INPUT_CONSUMER_DONE = {}
_INPUT_PENDING = {}
_FUSED_A2A_COLLECTIVE = (
    _FUSED_A2A_PACKED_MXFP8 and _FUSED_A2A_SIDESTREAM and _FUSED_A2A_INTERLEAVE
)


def _register_input_collective():
    supported = {
        ("2.9.1+gitff65f5b", "e63c384da344ad04e68ad01481fbd130adc73bce"),
        # amdsiloai/pytorch-xdit:v26.9. The private collective/wait signatures
        # and Python-wrapper implementations match the reference build.
        ("2.9.1+gitff65f5b", "bcfe9233b739a9ef700f8deec0a7495ed258fcdc"),
    }
    actual = (torch.__version__, torch.version.git_version)
    if actual not in supported:
        raise RuntimeError(
            "xfuser input collective private Inductor shim requires one of "
            f"{sorted(supported)}; found {actual}. Disable fused sidestream "
            "interleave or use a validated image."
        )
    from torch._inductor import config, ir
    from torch._inductor.lowering import (
        add_layout_constraint, constrain_to_fx_strides, register_lowering,
    )

    config.reorder_for_compute_comm_overlap = True
    lib = torch.library.Library("xfuser", "FRAGMENT")
    lib.define(
        "fused_a2a_input_collective(Tensor input, Tensor lifetime, Tensor[] previous, "
        "int role, str group_name, int rank, int world_size) -> Tensor[]",
        tags=(torch.Tag.cudagraph_unsafe,),
    )
    lib.define("fused_a2a_input_wait(Tensor input) -> Tensor", tags=(torch.Tag.cudagraph_unsafe,))

    def submit(input, lifetime, previous, role, group_name, rank, world_size):
        group = dist.distributed_c10d._resolve_process_group(group_name)
        handle = (group_name, rank, input.device.index)
        pending = _INPUT_PENDING.get(handle)
        if role == 0 and pending is not None:
            raise ValueError("previous interleaved input has not been consumed")
        pending = _submit_input_role(input, role, group, rank, pending)
        _INPUT_PENDING[handle] = pending
        op = pending["op"]
        parity = (op._epoch - (role == 2)) % 2
        b, s, h, d = input.shape
        shape = (b, s * world_size, h // world_size, d)
        payload = op.outputs_sets[parity][role].view(shape)
        scale = op.scales_sets[parity][role]
        if role < 2:
            scale = scale.view(*shape[:-1], d // 32)
        done = torch.cuda.Event()
        done.record(pending["side"])
        for tensor in (payload, scale):
            _INPUT_COLLECTIVE_WAITS[tensor.data_ptr()] = (done, handle, role)
        return [payload, scale]

    def submit_fake(input, lifetime, previous, role, group_name, rank, world_size):
        b, s, h, d = input.shape
        shape = (b, s * world_size, h // world_size, d)
        return [
            input.new_empty(shape, dtype=torch.uint8),
            input.new_empty((*shape[:-1], d // 32), dtype=torch.uint8) if role < 2
            else input.new_empty((1,), dtype=torch.float32),
        ]

    def wait(input):
        done, handle, role = _INPUT_COLLECTIVE_WAITS.pop(input.data_ptr())
        torch.cuda.current_stream(input.device).wait_event(done)
        if role == 2 and handle in _INPUT_PENDING:
            del _INPUT_PENDING[handle]
        return input

    lib.impl("fused_a2a_input_collective", submit, "CUDA")
    lib.impl("fused_a2a_input_wait", wait, "CUDA")
    torch.library.register_fake("xfuser::fused_a2a_input_collective", submit_fake)
    torch.library.register_fake("xfuser::fused_a2a_input_wait", lambda input: input)
    submit_op = torch.ops.xfuser.fused_a2a_input_collective.default
    wait_op = torch.ops.xfuser.fused_a2a_input_wait.default
    for op in (submit_op, wait_op):
        add_layout_constraint(op, constrain_to_fx_strides)
        has_side_effect(op)

    @register_lowering(submit_op, type_promotion_kind=None)
    def lower_submit(input, lifetime, previous, role, group_name, rank, world_size):
        # Like _dtensor.shard_dim_alltoall, allocation is owned by the collective.
        # Two source bindings match _WaitKernel's output-indexed volatile reads.
        outputs = ir._CollectiveKernel.create_out_of_place(
            submit_op, input, lifetime, previous, role, group_name, rank, world_size,
        )
        return [ir.TensorBox.create(output) for output in outputs]

    @register_lowering(wait_op, type_promotion_kind=None)
    def lower_wait(input):
        if config.cpp_wrapper:
            raise RuntimeError(
                "fused A2A Tier-2 wait lowering does not support Inductor's "
                "C++ wrapper; use the default Python wrapper"
            )
        ir._WaitKernel.create_wait(wait_op, input)
        return input

    return lib


_INPUT_COLLECTIVE_WAITS = {}
_INPUT_COLLECTIVE_LIB = _register_input_collective() if _FUSED_A2A_COLLECTIVE else None


def _input_side_stream(device):
    if not (_FUSED_A2A_SIDESTREAM and (_FUSED_A2A_PACKED or use_fused_a2a_interleave())):
        return None
    if device not in _INPUT_SIDE_STREAMS:
        _INPUT_SIDE_STREAMS[device] = torch.cuda.Stream(device=device)
        print(
            f"[XFUSER_FUSED_A2A_SIDESTREAM rank={dist.get_rank()}] "
            f"device={device} compute={torch.cuda.current_stream(device).cuda_stream} "
            f"side={_INPUT_SIDE_STREAMS[device].cuda_stream} "
            f"hadamard={_FUSED_A2A_HADAMARD_PLACEMENT} codecs={_FUSED_A2A_CODECS} "
            f"tier2={_FUSED_A2A_COLLECTIVE}",
            flush=True,
        )
    return _INPUT_SIDE_STREAMS[device]


@torch.library.custom_op(
    "xfuser::fused_a2a_consumer_done", mutates_args=(), tags=(torch.Tag.cudagraph_unsafe,),
)
def fused_a2a_input_consumer_done(device: torch.device) -> None:
    if not (_FUSED_A2A_SIDESTREAM and (_FUSED_A2A_PACKED or use_fused_a2a_interleave())):
        return
    done = torch.cuda.Event()
    done.record(torch.cuda.current_stream(device))
    _INPUT_CONSUMER_DONE[device] = done


@fused_a2a_input_consumer_done.register_fake
def _fused_a2a_consumer_done_fake(device):
    return None


_register_effectful_op(torch.ops.xfuser.fused_a2a_consumer_done.default, _EffectType.ORDERED)
has_side_effect(torch.ops.xfuser.fused_a2a_consumer_done.default)


def get_fused_a2a_mode():
    """Return 0 for RCCL, 1 for transport-only, or 2 for full fusion."""
    return _FUSED_A2A_MODE


def get_fused_a2a_hadamard_placement():
    """Return where Q/K Hadamard is applied for the current process."""
    return _FUSED_A2A_HADAMARD_PLACEMENT


def use_fused_a2a_packed():
    return _FUSED_A2A_PACKED


def use_fused_a2a_packed_f4f4():
    return _FUSED_A2A_PACKED_F4F4


def use_fused_a2a_interleave():
    return (
        _FUSED_A2A_INTERLEAVE
        and _FUSED_A2A_SIDESTREAM
        and (_FUSED_A2A_PACKED or _FUSED_A2A_INTERLEAVE_BF16)
    )


def fused_a2a_input_role(input, role, group, rank, pending=None):
    """Submit one role, retaining only a traceable handle outside the opaque op."""
    if _FUSED_A2A_COLLECTIVE:
        previous = [] if pending is None else list(pending)
        outputs = torch.ops.xfuser.fused_a2a_input_collective.default(
            input, input, previous, role, group.group_name, rank, dist.get_world_size(group),
        )
        return (*previous, *outputs)
    handle = (group.group_name, rank, input.device.index)
    if role not in (0, 1, 2) or (role != 0 and pending != (*handle, role)):
        raise ValueError("interleave requires Q, K, V in order")
    _fused_a2a_submit_role(input, role, group.group_name, rank)
    return (*handle, role + 1)


@torch.library.custom_op(
    "xfuser::fused_a2a_submit_role",
    mutates_args=(),
    tags=(torch.Tag.cudagraph_unsafe,),
)
def _fused_a2a_submit_role(
    input: torch.Tensor, role: int, group_name: str, rank: int,
) -> None:
    # The ordered effect owns peer buffers, handshake state and host parity. CUDA
    # graph replay would bypass those host updates and the runtime pending bridge.
    group = dist.distributed_c10d._resolve_process_group(group_name)
    handle = (group_name, rank, input.device.index)
    pending = _INPUT_PENDING.get(handle)
    if role == 0 and pending is not None:
        raise ValueError("previous interleaved input has not been consumed")
    _INPUT_PENDING[handle] = _submit_input_role(input, role, group, rank, pending)


@_fused_a2a_submit_role.register_fake
def _fused_a2a_submit_role_fake(input, role, group_name, rank):
    return None


if not _FUSED_A2A_COLLECTIVE:
    _register_effectful_op(
        torch.ops.xfuser.fused_a2a_submit_role.default, _EffectType.ORDERED,
    )
# PyTorch 2.9 FX DCE must retain both the original no-return node and AOT's
# token wrapper; otherwise Inductor silently removes the ordered submissions.
has_side_effect(torch.ops.xfuser.fused_a2a_submit_role.default)
has_side_effect(with_effects)


def _submit_input_role(input, role, group, rank, pending=None):
    """Submit one already-normalized sequence-major role without a compute join."""
    if not use_fused_a2a_interleave():
        raise RuntimeError("per-role input requires quantized sidestream interleave")
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


def _fused_a2a_input_runtime(
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
    side_stream = _input_side_stream(q.device)
    if pending is not None:
        if pending != (group.group_name, rank, q.device.index, 3):
            raise ValueError("interleaved input handle must finish this group's Q/K/V trio")
        pending = _INPUT_PENDING.pop(pending[:3])
        if pending["op"] is not in_op or pending["next_role"] != 3:
            raise ValueError("interleaved input must finish the same op's Q/K/V trio")
        transport_done = torch.cuda.Event()
        transport_done.record(pending["side"])
        torch.cuda.current_stream(q.device).wait_event(transport_done)
        outputs = pending["outputs"]
    elif side_stream is None:
        outputs = in_op(q, k, v, norm_q, norm_k, cos, sin)
    else:
        compute_stream = torch.cuda.current_stream(q.device)
        consumer_done = _INPUT_CONSUMER_DONE.get(q.device)
        if consumer_done is not None:
            # Conservatively drain the previous consumer, not just the reused parity.
            side_stream.wait_event(consumer_done)
        producer_done = torch.cuda.Event()
        # Include lazy op initialization as well as Q/K/V and norm/RoPE production.
        producer_done.record(compute_stream)
        side_stream.wait_event(producer_done)
        outputs = in_op(q, k, v, norm_q, norm_k, cos, sin, stream=side_stream)
        transport_done = torch.cuda.Event()
        transport_done.record(side_stream)
        compute_stream.wait_event(transport_done)
    b, s_local, h_total, d = q.shape
    world_size = dist.get_world_size(group)
    if _FUSED_A2A_PACKED:
        outputs, (q_scales, k_scales, v_scales) = outputs
        output_shape = (b, world_size * s_local, h_total // world_size, d)
        scale_shape = (*output_shape[:-1], d // 32)
        if _FUSED_A2A_PACKED_F4F4:
            from aiter.ops.mha_v4 import mxfp4_k_view, mxfp4_v_view

            q_raw, k_raw, v_raw = outputs
            q_scales = q_scales.view(scale_shape)
            k_scales = k_scales.view(scale_shape)
            sequence, heads = output_shape[1:3]
            v_scales = v_scales.view(b, heads, ((sequence + 127) // 128) * 512)
            # K/V expose the tiled consumer ABI, not contiguous packed BSHD.
            return (
                (
                    q_raw.view(*output_shape[:-1], d // 2),
                    mxfp4_k_view(k_raw, k_scales),
                    mxfp4_v_view(v_raw, v_scales, sequence),
                ),
                (q_scales, k_scales, v_scales),
            )
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


def _owned_transport_tensor(tensor):
    # Custom-op results must own storage: cached symmetric buffers alias across
    # epochs, and tiled packed K/V consumers also read their backing padding.
    size = tensor.untyped_storage().nbytes() // tensor.element_size()
    storage = tensor.as_strided((size,), (1,), 0).clone()
    return storage.as_strided(tensor.shape, tensor.stride(), tensor.storage_offset())


def fused_a2a_input(
    query, key, value, group, rank, norm_q=None, norm_k=None, cos=None, sin=None,
    softmax_scale=None, pending=None,
):
    if _FUSED_A2A_COLLECTIVE and pending is not None:
        if len(pending) != 6:
            raise ValueError("collective input requires all three role outputs")
        outputs = [torch.ops.xfuser.fused_a2a_input_wait.default(tensor) for tensor in pending]
        return (outputs[0], outputs[2], outputs[4]), (outputs[1], outputs[3], outputs[5])
    if pending is not None and pending != (group.group_name, rank, query.device.index, 3):
        raise ValueError("interleaved input handle must finish this group's Q/K/V trio")
    outputs = _fused_a2a_wait(
        query, key, value, group.group_name, rank, dist.get_world_size(group),
        norm_q, norm_k, cos, sin, softmax_scale, pending is not None,
    )
    if _FUSED_A2A_PACKED:
        return tuple(outputs[:3]), tuple(outputs[3:])
    return tuple(outputs)


@torch.library.custom_op(
    "xfuser::fused_a2a_wait", mutates_args=(), tags=(torch.Tag.cudagraph_unsafe,),
)
def _fused_a2a_wait(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    group_name: str, rank: int, world_size: int,
    norm_q: torch.Tensor | None, norm_k: torch.Tensor | None,
    cos: torch.Tensor | None, sin: torch.Tensor | None,
    softmax_scale: float | None, interleaved: bool,
) -> list[torch.Tensor]:
    group = dist.distributed_c10d._resolve_process_group(group_name)
    pending = (group_name, rank, query.device.index, 3) if interleaved else None
    outputs = _fused_a2a_input_runtime(
        query, key, value, group, rank, norm_q, norm_k, cos, sin,
        softmax_scale, pending,
    )
    if _FUSED_A2A_PACKED:
        outputs = (*outputs[0], *outputs[1])
    return [_owned_transport_tensor(tensor) for tensor in outputs]


@_fused_a2a_wait.register_fake
def _fused_a2a_wait_fake(
    query, key, value, group_name, rank, world_size,
    norm_q, norm_k, cos, sin, softmax_scale, interleaved,
):
    b, h, s, d = query.shape
    shape = (b, s * world_size, h // world_size, d)
    if not _FUSED_A2A_PACKED:
        return [query.new_empty((b, h // world_size, s * world_size, d)) for _ in range(3)]
    scales = [query.new_empty((*shape[:-1], d // 32), dtype=torch.uint8) for _ in range(2)]
    if _FUSED_A2A_PACKED_F4F4:
        from aiter.ops.mha_v4 import mxfp4_k_view, mxfp4_v_view

        tiles = (shape[1] + 127) // 128
        size = b * shape[2] * tiles * 8192
        q = query.new_empty((*shape[:-1], d // 2), dtype=torch.uint8)
        k = mxfp4_k_view(query.new_empty((size,), dtype=torch.uint8), scales[1])
        vs = query.new_empty((b, shape[2], tiles * 512), dtype=torch.uint8)
        v = mxfp4_v_view(query.new_empty((size + 64,), dtype=torch.uint8), vs, shape[1])
        return [q, k, v, *scales, vs]
    return [query.new_empty(shape, dtype=torch.uint8) for _ in range(3)] + [
        *scales, query.new_empty((1,), dtype=torch.float32),
    ]


_fused_a2a_output_runtime = fused_a2a_output


def fused_a2a_output(output, group, rank, return_sequence_major=False):
    return _fused_a2a_output_launch(
        output, group.group_name, rank, dist.get_world_size(group), return_sequence_major,
    )


@torch.library.custom_op(
    "xfuser::fused_a2a_output", mutates_args=(), tags=(torch.Tag.cudagraph_unsafe,),
)
def _fused_a2a_output_launch(
    output: torch.Tensor, group_name: str, rank: int, world_size: int,
    return_sequence_major: bool,
) -> torch.Tensor:
    group = dist.distributed_c10d._resolve_process_group(group_name)
    return _fused_a2a_output_runtime(output, group, rank, return_sequence_major).clone()


@_fused_a2a_output_launch.register_fake
def _fused_a2a_output_fake(output, group_name, rank, world_size, return_sequence_major):
    b, h, s, d = output.shape
    shape = (b, s // world_size, h * world_size, d) if return_sequence_major else (
        b, h * world_size, s // world_size, d,
    )
    return output.new_empty(shape)


for _op in (torch.ops.xfuser.fused_a2a_wait.default, torch.ops.xfuser.fused_a2a_output.default):
    if not _FUSED_A2A_COLLECTIVE or _op == torch.ops.xfuser.fused_a2a_output.default:
        _register_effectful_op(_op, _EffectType.ORDERED)
    has_side_effect(_op)


# Cached outputs are persistent symmetric buffers. A same-key launch is safe only
# after the prior attention consumer has finished reading them on its stream.
