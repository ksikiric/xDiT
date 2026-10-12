"""Measure the attainable MHA-output -> Ulysses A2A -> to_out GEMM savings.

Launch through xdit so the benchmark uses the same distributed launcher/port:

    XDIT_RUNNER_MODULE=benchmarks.quantized_output_handoff \
      xdit --ulysses_degree 8 --sequence 75776 --heads 40

The benchmark deliberately keeps MHA itself out of the timed region. Every
variant starts from the same logical BF16 MHA output. The prepacked upper bound
models an MHA epilogue that directly emits the transport payload and scales.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

import aiter

from xfuser.model_executor.layers.fp8_linear import (
    quantize_weight_to_fp8_blockscale_plain,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ulysses_degree", "--ulysses-degree", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=75776)
    parser.add_argument("--valid_sequence", "--valid-sequence", type=int, default=75600)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head_dim", "--head-dim", type=int, default=128)
    parser.add_argument("--out_features", "--out-features", type=int, default=5120)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/quantized_output_handoff/results.json"),
    )
    return parser


def _inverse_a2a(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Mirror xDiT's Ulysses output A2A for one BHSD tensor."""
    batch, heads_local, sequence, width = tensor.shape
    if sequence % world_size:
        raise ValueError(f"sequence {sequence} must be divisible by world size {world_size}")
    sequence_local = sequence // world_size
    send = tensor.permute(2, 0, 1, 3).contiguous()
    received = torch.empty_like(send)
    dist.all_to_all_single(received.view(-1), send.view(-1))
    return (
        received.reshape(
            world_size,
            sequence_local,
            batch,
            heads_local,
            width,
        )
        .permute(2, 0, 3, 1, 4)
        .reshape(
            batch,
            heads_local * world_size,
            sequence_local,
            width,
        )
    )


def _sequence_major_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """BHSD -> [B*S, H*D], matching Wan's input to attn.to_out[0]."""
    batch, heads, sequence, width = tensor.shape
    return (
        tensor.transpose(1, 2)
        .contiguous()
        .view(
            batch * sequence,
            heads * width,
        )
    )


def _quantize_per_head(
    tensor: torch.Tensor,
    quantize,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each 128-wide MHA output vector independently."""
    batch, heads, sequence, width = tensor.shape
    flat = tensor.reshape(batch * heads * sequence, width)
    payload, scales = quantize(flat, quant_dtype=fp8_dtype)
    return (
        payload.view(batch, heads, sequence, width),
        scales.view(batch, heads, sequence, -1),
    )


def _pack_a2a_send(
    payload: torch.Tensor,
    scales: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, int, int, int]:
    """Pack each peer's FP8 rows and FP32 scales into one equal-size byte chunk."""
    batch, heads_local, sequence, width = payload.shape
    scale_width = scales.shape[-1]
    sequence_local = sequence // world_size
    payload_chunks = payload.permute(2, 0, 1, 3).contiguous().view(torch.uint8).reshape(world_size, -1)
    scale_chunks = scales.permute(2, 0, 1, 3).contiguous().view(torch.uint8).reshape(world_size, -1)
    return (
        torch.cat((payload_chunks, scale_chunks), dim=1),
        payload_chunks.shape[1],
        sequence_local,
        scale_width,
    )


def _packed_inverse_a2a(
    send: torch.Tensor,
    *,
    payload_bytes: int,
    sequence_local: int,
    scale_width: int,
    batch: int,
    heads_local: int,
    head_dim: int,
    world_size: int,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One byte A2A, then expose payload/scales in destination BHSD order."""
    received = torch.empty_like(send)
    dist.all_to_all_single(received.view(-1), send.view(-1))
    payload_chunks = (
        received[:, :payload_bytes]
        .contiguous()
        .view(fp8_dtype)
        .reshape(
            world_size,
            sequence_local,
            batch,
            heads_local,
            head_dim,
        )
    )
    scale_chunks = (
        received[:, payload_bytes:]
        .contiguous()
        .view(torch.float32)
        .reshape(
            world_size,
            sequence_local,
            batch,
            heads_local,
            scale_width,
        )
    )
    payload = payload_chunks.permute(2, 0, 3, 1, 4).reshape(
        batch,
        heads_local * world_size,
        sequence_local,
        head_dim,
    )
    scales = scale_chunks.permute(2, 0, 3, 1, 4).reshape(
        batch,
        heads_local * world_size,
        sequence_local,
        scale_width,
    )
    return payload, scales


def _rank_max_ms(local_ms: float, device: torch.device) -> float:
    value = torch.tensor(local_ms, dtype=torch.float64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value.item()


def _measure(
    fn,
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> float:
    result = None
    for _ in range(warmup):
        result = fn()
    del result
    torch.cuda.synchronize(device)
    dist.barrier()
    start = time.perf_counter()
    for _ in range(iterations):
        result = fn()
    torch.cuda.synchronize(device)
    local_ms = (time.perf_counter() - start) * 1000.0 / iterations
    del result
    return _rank_max_ms(local_ms, device)


def _all_ranks_true(value: bool, device: torch.device) -> bool:
    flag = torch.tensor(int(value), dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def main() -> None:
    args = _parser().parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.ulysses_degree:
        raise ValueError(f"launcher world size {world_size} != --ulysses_degree {args.ulysses_degree}")
    if args.heads % world_size:
        raise ValueError("heads must be divisible by the Ulysses degree")
    if args.sequence % world_size:
        raise ValueError("sequence must be divisible by the Ulysses degree")
    if args.head_dim != 128:
        raise ValueError("this experiment targets the hd128 MHA/GEMM contract")

    batch = 1
    heads_local = args.heads // world_size
    sequence_local = args.sequence // world_size
    in_features = args.heads * args.head_dim
    fp8_dtype = aiter.dtypes.fp8
    quantize = aiter.get_hip_quant(aiter.QuantType.per_1x128)

    torch.manual_seed(1200 + rank)
    mha_output = torch.randn(
        batch,
        heads_local,
        args.sequence,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )

    torch.manual_seed(42)
    weight = torch.randn(
        args.out_features,
        in_features,
        device=device,
        dtype=torch.bfloat16,
    )
    weight_fp8, weight_scale = quantize_weight_to_fp8_blockscale_plain(weight)
    del weight

    def prequantized_gemm(payload_matrix, scale_matrix):
        return aiter.gemm_a8w8_blockscale(
            payload_matrix,
            weight_fp8,
            scale_matrix,
            weight_scale,
        )

    bf16_received = _inverse_a2a(mha_output, world_size)
    bf16_matrix = _sequence_major_matrix(bf16_received)
    reference_payload, reference_scales = quantize(
        bf16_matrix,
        quant_dtype=fp8_dtype,
    )

    source_payload, source_scales = _quantize_per_head(
        mha_output,
        quantize,
        fp8_dtype,
    )
    payload_received = _inverse_a2a(source_payload, world_size)
    scales_received = _inverse_a2a(source_scales, world_size)
    payload_matrix = _sequence_major_matrix(payload_received)
    scale_matrix = _sequence_major_matrix(scales_received)

    payload_equal = _all_ranks_true(
        torch.equal(
            payload_matrix.view(torch.uint8),
            reference_payload.view(torch.uint8),
        ),
        device,
    )
    scales_equal = _all_ranks_true(
        torch.equal(scale_matrix, reference_scales),
        device,
    )

    packed_send, payload_bytes, packed_sequence_local, scale_width = _pack_a2a_send(
        source_payload, source_scales, world_size
    )
    packed_payload, packed_scales = _packed_inverse_a2a(
        packed_send,
        payload_bytes=payload_bytes,
        sequence_local=packed_sequence_local,
        scale_width=scale_width,
        batch=batch,
        heads_local=heads_local,
        head_dim=args.head_dim,
        world_size=world_size,
        fp8_dtype=fp8_dtype,
    )
    packed_payload_matrix = _sequence_major_matrix(packed_payload)
    packed_scale_matrix = _sequence_major_matrix(packed_scales)
    packed_equal = _all_ranks_true(
        torch.equal(
            packed_payload_matrix.view(torch.uint8),
            reference_payload.view(torch.uint8),
        )
        and torch.equal(packed_scale_matrix, reference_scales),
        device,
    )

    normal_output = torch.ops.xfuser.fp8_blockscale_gemm(
        bf16_matrix,
        weight_fp8,
        weight_scale,
    )
    prequantized_output = prequantized_gemm(
        reference_payload,
        reference_scales,
    )
    gemm_max_abs = (normal_output - prequantized_output).abs().max()
    dist.all_reduce(gemm_max_abs, op=dist.ReduceOp.MAX)
    gemm_equal = _all_ranks_true(
        torch.equal(normal_output, prequantized_output),
        device,
    )
    del normal_output, prequantized_output

    def current_pipeline():
        received = _inverse_a2a(mha_output, world_size)
        matrix = _sequence_major_matrix(received)
        return torch.ops.xfuser.fp8_blockscale_gemm(
            matrix,
            weight_fp8,
            weight_scale,
        )

    def quantized_two_collective_pipeline():
        payload, scales = _quantize_per_head(
            mha_output,
            quantize,
            fp8_dtype,
        )
        payload = _sequence_major_matrix(_inverse_a2a(payload, world_size))
        scales = _sequence_major_matrix(_inverse_a2a(scales, world_size))
        return prequantized_gemm(payload, scales)

    def quantized_packed_pipeline():
        payload, scales = _quantize_per_head(
            mha_output,
            quantize,
            fp8_dtype,
        )
        send, q_bytes, s_local, s_width = _pack_a2a_send(
            payload,
            scales,
            world_size,
        )
        payload, scales = _packed_inverse_a2a(
            send,
            payload_bytes=q_bytes,
            sequence_local=s_local,
            scale_width=s_width,
            batch=batch,
            heads_local=heads_local,
            head_dim=args.head_dim,
            world_size=world_size,
            fp8_dtype=fp8_dtype,
        )
        return prequantized_gemm(
            _sequence_major_matrix(payload),
            _sequence_major_matrix(scales),
        )

    def fused_epilogue_upper_bound():
        payload, scales = _packed_inverse_a2a(
            packed_send,
            payload_bytes=payload_bytes,
            sequence_local=packed_sequence_local,
            scale_width=scale_width,
            batch=batch,
            heads_local=heads_local,
            head_dim=args.head_dim,
            world_size=world_size,
            fp8_dtype=fp8_dtype,
        )
        return prequantized_gemm(
            _sequence_major_matrix(payload),
            _sequence_major_matrix(scales),
        )

    benchmarks = {
        "component_bf16_output_a2a_ms": lambda: _inverse_a2a(
            mha_output,
            world_size,
        ),
        "component_dynamic_quant_ms": lambda: _quantize_per_head(
            mha_output,
            quantize,
            fp8_dtype,
        ),
        "component_fp8_payload_a2a_ms": lambda: _inverse_a2a(
            source_payload,
            world_size,
        ),
        "component_scale_a2a_ms": lambda: _inverse_a2a(
            source_scales,
            world_size,
        ),
        "component_packed_payload_scale_a2a_ms": lambda: _packed_inverse_a2a(
            packed_send,
            payload_bytes=payload_bytes,
            sequence_local=packed_sequence_local,
            scale_width=scale_width,
            batch=batch,
            heads_local=heads_local,
            head_dim=args.head_dim,
            world_size=world_size,
            fp8_dtype=fp8_dtype,
        ),
        "component_normal_quantized_gemm_ms": lambda: torch.ops.xfuser.fp8_blockscale_gemm(
            bf16_matrix,
            weight_fp8,
            weight_scale,
        ),
        "component_prequantized_gemm_ms": lambda: prequantized_gemm(
            reference_payload,
            reference_scales,
        ),
        "pipeline_current_bf16_a2a_ms": current_pipeline,
        "pipeline_quantized_two_collective_ms": quantized_two_collective_pipeline,
        "pipeline_quantized_packed_ms": quantized_packed_pipeline,
        "pipeline_fused_epilogue_upper_bound_ms": fused_epilogue_upper_bound,
    }
    timings = {
        name: _measure(
            fn,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        for name, fn in benchmarks.items()
    }

    baseline = timings["pipeline_current_bf16_a2a_ms"]
    for name in (
        "pipeline_quantized_two_collective_ms",
        "pipeline_quantized_packed_ms",
        "pipeline_fused_epilogue_upper_bound_ms",
    ):
        timings[f"{name}_speedup_percent"] = (baseline - timings[name]) / baseline * 100.0

    result = {
        "configuration": {
            "world_size": world_size,
            "batch": batch,
            "heads": args.heads,
            "heads_local": heads_local,
            "sequence": args.sequence,
            "valid_sequence": args.valid_sequence,
            "sequence_local": sequence_local,
            "head_dim": args.head_dim,
            "in_features": in_features,
            "out_features": args.out_features,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "correctness": {
            "payload_byte_exact": payload_equal,
            "scales_exact": scales_equal,
            "packed_transport_exact": packed_equal,
            "gemm_output_exact": gemm_equal,
            "gemm_output_max_abs": gemm_max_abs.item(),
        },
        "bytes_per_rank_per_call": {
            "bf16_payload": mha_output.numel() * mha_output.element_size(),
            "fp8_payload": source_payload.numel() * source_payload.element_size(),
            "scale_payload": source_scales.numel() * source_scales.element_size(),
            "packed_total": packed_send.numel() * packed_send.element_size(),
        },
        "rank_max_timings_ms": timings,
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
