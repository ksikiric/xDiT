"""Validate MORI attention-output codecs against canonical GEMM quantizers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

import aiter
from benchmarks.quantized_output_handoff import (
    _all_ranks_true,
    _inverse_a2a,
    _measure,
    _sequence_major_matrix,
)


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ulysses_degree", "--ulysses-degree", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head_dim", "--head-dim", type=int, default=128)
    parser.add_argument("--out_features", "--out-features", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--codecs",
        nargs="+",
        choices=("e4m3", "mxfp4", "mxfp6"),
        default=("e4m3", "mxfp4", "mxfp6"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/quantized_output_codecs/results.json"),
    )
    return parser


def _reference_quant(input_2d, codec):
    if codec == "e4m3":
        quantize = aiter.get_hip_quant(aiter.QuantType.per_1x128)
        return quantize(input_2d, quant_dtype=aiter.dtypes.fp8)
    if codec == "mxfp4":
        quantize = aiter.get_hip_quant(aiter.QuantType.per_1x32)
        return quantize(input_2d, quant_dtype=aiter.dtypes.fp4x2, shuffle=True)

    from aiter.ops.gemm_op_a6w6 import (
        mxfp6_gemm_pack_size,
        quant_mxfp6_gemm_out,
    )

    packed_size, scale_size = mxfp6_gemm_pack_size(*input_2d.shape)
    packed = torch.zeros(packed_size, dtype=torch.uint8, device=input_2d.device)
    scale = torch.zeros(scale_size, dtype=torch.uint8, device=input_2d.device)
    return quant_mxfp6_gemm_out(input_2d, packed, scale)


def _result_operands(result, codec, rows, features):
    if codec == "e4m3":
        return (
            result.payload.view(aiter.dtypes.fp8).view(rows, features),
            result.scale.view(rows, features // 128),
        )
    if codec == "mxfp4":
        return (
            result.payload.view(aiter.dtypes.fp4x2),
            result.scale.view(aiter.dtypes.fp8_e8m0),
        )
    return result.payload, result.scale


def _gemm_outputs(input_2d, actual, expected, codec, out_features):
    rows, features = input_2d.shape
    torch.manual_seed(47)
    weight = torch.randn(
        out_features,
        features,
        dtype=torch.bfloat16,
        device=input_2d.device,
    )
    if codec == "e4m3":
        from xfuser.model_executor.layers.fp8_linear import (
            quantize_weight_to_fp8_blockscale_plain,
        )

        weight_q, weight_scale = quantize_weight_to_fp8_blockscale_plain(weight)

        def run(pair):
            return aiter.gemm_a8w8_blockscale(pair[0], weight_q, pair[1], weight_scale)

    elif codec == "mxfp4":
        from aiter.ops.shuffle import shuffle_weight

        quantize = aiter.get_hip_quant(aiter.QuantType.per_1x32)
        weight_q, weight_scale = quantize(
            weight,
            quant_dtype=aiter.dtypes.fp4x2,
            shuffle=True,
        )
        weight_q = shuffle_weight(weight_q, layout=(16, 16))

        def run(pair):
            return aiter.gemm_a4w4(
                pair[0],
                weight_q,
                pair[1],
                weight_scale,
                bpreshuffle=True,
            )

    else:
        from aiter.ops.gemm_op_a6w4 import (
            gemm_a6w4,
            mxfp4_gemm_pack_size,
            quant_mxfp4_gemm_hip_out,
        )
        from aiter.ops.gemm_op_a6w6 import (
            gemm_a6w6,
            quant_mxfp6_gemm,
        )

        weight6, weight6_scale = quant_mxfp6_gemm(weight)
        weight4_size, weight4_scale_size = mxfp4_gemm_pack_size(out_features, features)
        weight4 = torch.zeros(weight4_size, dtype=torch.uint8, device=input_2d.device)
        weight4_scale = torch.zeros(weight4_scale_size, dtype=torch.uint8, device=input_2d.device)
        quant_mxfp4_gemm_hip_out(weight, weight4, weight4_scale)

        def run_a6w6(pair):
            return gemm_a6w6(
                pair[0],
                weight6,
                pair[1],
                weight6_scale,
                rows,
                out_features,
                features,
            )

        def run_a6w4(pair):
            return gemm_a6w4(
                pair[0],
                weight4,
                pair[1],
                weight4_scale,
                rows,
                out_features,
                features,
            )

        return {
            "a6w6": (run_a6w6(actual), run_a6w6(expected)),
            "a6w4": (run_a6w4(actual), run_a6w4(expected)),
        }

    return {"default": (run(actual), run(expected))}


def main():
    args = _parser().parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != args.ulysses_degree:
        raise ValueError("launcher world size does not match --ulysses_degree")
    if args.heads % world_size or args.sequence % world_size:
        raise ValueError("heads and sequence must divide by the Ulysses degree")

    import mori.shmem as ms

    cpu_group = dist.new_group(backend="gloo")
    group_name = "attention_output_codec_test"
    torch._C._distributed_c10d._register_process_group(group_name, cpu_group)
    ms.shmem_torch_process_group_init(group_name)
    try:
        from aiter.ops.flydsl.attention_output_a2a_intranode import (
            AttentionOutputA2AIntraNodeOp,
        )

        torch.manual_seed(1700 + rank)
        source = torch.randn(
            1,
            args.sequence,
            args.heads // world_size,
            args.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        reference_bhsd = _inverse_a2a(source.transpose(1, 2), world_size)
        reference_matrix = _sequence_major_matrix(reference_bhsd)
        rows, features = reference_matrix.shape
        results = {}
        for codec in args.codecs:
            op = AttentionOutputA2AIntraNodeOp(
                rank=rank,
                world_size=world_size,
                shape=tuple(source.shape),
                group=cpu_group,
                codec=codec,
            )
            actual_result = op(source)
            actual = _result_operands(
                actual_result,
                codec,
                rows,
                features,
            )
            expected = _reference_quant(reference_matrix, codec)
            payload_exact = _all_ranks_true(
                torch.equal(
                    actual[0].view(torch.uint8),
                    expected[0].view(torch.uint8),
                ),
                device,
            )
            scales_exact = _all_ranks_true(
                torch.equal(
                    actual[1].view(torch.uint8),
                    expected[1].view(torch.uint8),
                ),
                device,
            )
            corner_source = torch.full_like(source, -0.0)
            corner_source[:, 0, 0] = torch.linspace(
                -1.0,
                1.0,
                args.head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            corner_source[:, 1, 0, ::2] = torch.finfo(torch.bfloat16).tiny
            corner_source[:, 1, 0, 1::2] = -torch.finfo(torch.bfloat16).tiny
            corner_source[:, 2, 0, 0] = torch.finfo(torch.bfloat16).max
            corner_reference = _sequence_major_matrix(
                _inverse_a2a(
                    corner_source.transpose(1, 2),
                    world_size,
                )
            )
            corner_expected = _reference_quant(corner_reference, codec)
            corner_result = op(corner_source)
            corner_actual = _result_operands(
                corner_result,
                codec,
                rows,
                features,
            )
            corner_payload_exact = _all_ranks_true(
                torch.equal(
                    corner_actual[0].view(torch.uint8),
                    corner_expected[0].view(torch.uint8),
                ),
                device,
            )
            corner_scales_exact = _all_ranks_true(
                torch.equal(
                    corner_actual[1].view(torch.uint8),
                    corner_expected[1].view(torch.uint8),
                ),
                device,
            )
            gemms = _gemm_outputs(
                reference_matrix,
                actual,
                expected,
                codec,
                args.out_features,
            )
            gemm_results = {}
            for name, (actual_output, expected_output) in gemms.items():
                max_abs = (actual_output - expected_output).abs().max()
                dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
                gemm_results[name] = {
                    "exact": _all_ranks_true(
                        torch.equal(actual_output, expected_output),
                        device,
                    ),
                    "max_abs": max_abs.item(),
                }
            results[codec] = {
                "payload_exact": payload_exact,
                "scales_exact": scales_exact,
                "corner_payload_exact": corner_payload_exact,
                "corner_scales_exact": corner_scales_exact,
                "gemms": gemm_results,
                "mori_ms": _measure(
                    lambda: op(source),
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=device,
                ),
            }

        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(results, indent=2), flush=True)
        dist.barrier()
    finally:
        ms.shmem_finalize()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
