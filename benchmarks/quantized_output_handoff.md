# Quantized attention-output handoff experiment

This experiment estimates the benefit of keeping an MHA output quantized
through the Ulysses output all-to-all and consuming it directly in the
attention `to_out` FP8 GEMM.

## Tested paths

1. Current: BF16 output A2A, then the ordinary dynamic-quantized FP8 GEMM.
2. Quantized, two collectives: dynamic per-1x128 E4M3 quantization, separate
   payload and scale A2As, then a prequantized GEMM.
3. Quantized, packed collective: dynamic quantization, pack payload and scales
   into one byte A2A, unpack, then a prequantized GEMM.
4. Fused-epilogue upper bound: prebuilt payload/scales in one byte A2A, then a
   prequantized GEMM. This excludes standalone output quantization and packing.

The standalone benchmark is launched through `xdit`:

```bash
XDIT_RUNNER_MODULE=benchmarks.quantized_output_handoff \
xdit \
  --ulysses_degree 8 \
  --sequence 75776 \
  --valid_sequence 75600 \
  --heads 40 \
  --head_dim 128 \
  --out_features 5120 \
  --warmup 3 \
  --iterations 20
```

## Correctness

At both a small control shape and the Wan 720x1280 production shape:

- Quantization before versus after A2A produced byte-identical FP8 payloads.
- Dynamic per-1x128 scales were exactly equal.
- Packed payload/scale transport was exactly equal.
- Ordinary and prequantized GEMMs produced bit-identical BF16 outputs.

## Full-shape rank-max timings

- BF16 output A2A: 0.4212 ms
- Dynamic output quantization: 0.0249 ms
- FP8 payload A2A: 0.2791 ms
- Scale A2A: 0.0342 ms
- One packed payload+scale A2A: 0.2409 ms
- Ordinary quantized GEMM: 0.5764 ms
- Prequantized GEMM: 0.5481 ms

Combined path:

- Current BF16 A2A + ordinary GEMM: 1.0808 ms
- Quantized two-collective path: 0.9762 ms, 9.68% faster locally
- Quantized packed path: 1.0160 ms, 6.00% faster locally
- Fused-epilogue upper bound: 0.8620 ms, 20.25% faster locally

Raw result:

`benchmark_results/quantized_output_handoff_2026-10-12/wan-720x1280/results.json`

## Wan2.2 E2E

Two independent process launches per path, two iterations per launch:

- BF16 output path: 32.3765 s and 32.2134 s
- Quantized RCCL handoff: 32.3261 s and 32.3231 s

Means:

- BF16: 32.2950 s
- Quantized: 32.3246 s
- Quantized handoff: 0.09% slower

The run-to-run range of the BF16 baseline was larger than the measured
difference. Therefore the xDiT-only quantized RCCL handoff has no demonstrated
E2E benefit.

## Decision

The isolated output segment has optimization headroom, but it is too small a
fraction of E2E latency for the current prototype to establish a worthwhile
gain. The fused-MHA upper bound roughly doubles the isolated saving, but is
still likely sub-percent E2E.

Do not start the multi-format production implementation based on these results
alone. Proceed only if a true MHA-epilogue plus direct-layout MORI prototype can
demonstrate a stable gain above the team's minimum acceptance threshold.

## Two-GPU fused quantization + MORI prototype

The follow-up prototype keeps the BF16 MHA output but replaces the standalone
quantizer and RCCL output collective with one FlyDSL kernel that dynamically
quantizes each 128-element head vector and writes its E4M3 payload and FP32
scale directly into peer symmetric memory in the destination GEMM layout.

The two-GPU shape preserves the production per-rank byte volume:

- Source on each rank: `[1, 18944, 20, 128]` BF16 BSHD.
- Destination on each rank: `[1, 9472, 40, 128]` E4M3 BSHD.
- BF16 traffic per rank: 96,993,280 bytes.
- E4M3 payload plus scales: 50,012,160 bytes.

Correctness passed byte-for-byte for payloads and scales, including four
generations of double-buffer reuse, one deliberately delayed rank, and
zero/tiny/outlier rows. The final prequantized GEMM output was bit-identical to
the ordinary quantizing GEMM.

Two independent 512-block process launches measured:

- Current BF16 RCCL plus ordinary GEMM: `1.6718 ms`, `1.6662 ms`.
- Quantized MORI plus prequantized GEMM: `1.0798 ms`, `1.0801 ms`.
- Mean saving: `0.5890 ms`, or `35.29%` of this isolated output segment.
- Fused quantize-plus-MORI hop alone: `0.5268 ms` in both launches.

A 128-block launch produced `1.0798 ms` for the complete path, so increasing
the grid to 512 blocks did not help this output kernel at world size two. The
operator therefore retains the 128-block default.

Raw artifacts:

- `benchmark_results/quantized_output_mori_2026-10-22/production-volume/`
- `benchmark_results/quantized_output_mori_2026-10-22/production-volume-repeat/`
- `benchmark_results/quantized_output_mori_2026-10-22/production-volume-blocks128/`

This establishes a component-level gain, not an E2E claim. Eight-GPU transport
and E2E testing are still required because world-size topology and contention
can materially change the communication result.

## Two-GPU E2E result

The operator was then integrated before xDiT's packed-MHA BSHD-to-BHSD
transpose and connected directly to the FP8 block-scale `to_out` GEMM.
Matched eager Wan2.2 runs used 720x1280, 81 frames, 40 denoising steps,
Ulysses degree two, AITER FP8 attention, quantized input A2A, and AITER FP8
GEMMs.

- BF16 RCCL output hop: `216.9246 s`.
- Quantized MORI output hop: `207.6177 s`.
- Saving: `9.3070 s`, or `4.29%` lower E2E latency.

The first untimed/warmup-like iterations moved consistently as well:
`229.20 s` to `217.78 s`.

A production-shape in-process oracle compared the fused result against BF16
RCCL followed by AITER's canonical per-1x128 quantizer. Payload bytes and FP32
scales were exact. The baseline, an independent baseline repeat, and the
corrected MORI run also produced byte-identical encoded videos:

`ed7887b3a08fd461195da16b7704311b6d56c527489fe004938e6257b8a322a6`

Artifacts:

- `benchmark_results/output_mori_e2e_2gpu_2026-10-22/baseline-aiter-eager/`
- `benchmark_results/output_mori_e2e_2gpu_2026-10-22/baseline-aiter-eager-repeat/`
- `benchmark_results/output_mori_e2e_2gpu_2026-10-22/mori-aiter-eager-v3/`
- `benchmark_results/output_mori_e2e_2gpu_2026-10-22/production-parity-check-v3/`

The first compiled attempt exposed a Torch 2.13 metadata incompatibility in
the custom Tier-2 input collective: Inductor's communication cost model expects
the process-group name to be the final constant argument, while xDiT placed
`valid_kv_len` last. Reordering those two metadata fields, without changing
runtime transport behavior, restored compilation.

Matched compiled runs measured:

- BF16 RCCL output hop: `195.0868 s`.
- Quantized MORI output hop: `189.1500 s`.
- Saving: `5.9368 s`, or `3.04%` lower E2E latency.

The compiled baseline and MORI videos were byte-identical:

`2a27871f4d520fe4817b6ac75a714b14bf7aea3ada8ecc519897892d5238f149`

Compiled artifacts:

- `benchmark_results/output_mori_e2e_2gpu_2026-10-23/baseline-aiter-compile/`
- `benchmark_results/output_mori_e2e_2gpu_2026-10-23/mori-aiter-compile/`

## Eight-GPU compiled result

One matched compiled iteration per path measured:

- BF16 RCCL output hop: `45.1888 s`.
- Quantized MORI output hop: `44.6428 s`.
- Saving: `0.5460 s`, or `1.21%` lower E2E latency.

An eight-GPU production-volume component check remained byte-exact for the
payload, FP32 scales, double-buffer reuse, corner cases, and final GEMM output.
Its isolated output pipeline measured `1.0940 ms` for BF16 RCCL plus ordinary
GEMM and `0.7603 ms` for quantized MORI plus prequantized GEMM.

The two independently generated E2E videos did not have matching hashes. Since
only one iteration per path was requested, no baseline repeat was collected to
separate ordinary eight-GPU run-to-run nondeterminism from a path-dependent
difference. The byte-exact component check rules out the output transport and
GEMM handoff themselves as the source.

Artifacts:

- `benchmark_results/output_mori_e2e_8gpu_2026-10-23/baseline-aiter-compile/`
- `benchmark_results/output_mori_e2e_8gpu_2026-10-23/mori-aiter-compile/`
- `benchmark_results/output_mori_e2e_8gpu_2026-10-23/payload-parity/`

## Chunked output-pipeline experiment

At the eight-GPU production shape, splitting the prequantized `to_out` GEMM
into two equal M chunks added only `0.56%` GEMM latency; four chunks added
`33.53%` and was rejected.

A two-stream, two-chunk component prototype then overlapped transport of the
second chunk with GEMM of the first. Payloads and scales remained byte-exact,
but the measured improvement was small:

- Whole-tensor MORI plus GEMM: `0.7101 ms`.
- Actual two-chunk pipeline: `0.6966 ms`.
- Saving: `0.0136 ms`, or `1.91%` of the output segment.

Across roughly 1,600 self-attention calls this projects to about `22 ms`, less
than `0.05%` E2E. The result is much lower than the perfect-overlap bound
because each chunk pays its own transport launches and synchronization, while
the communication kernel also contends with GEMM. Source chunk preparation
was outside the timed region, so this is already optimistic.

Artifacts:

- `benchmark_results/output_mori_e2e_8gpu_2026-10-26/chunked-gemm-gate/`
- `benchmark_results/output_mori_e2e_8gpu_2026-10-26/chunked-pipeline-actual/`

## Output codec and GEMM precision matrix

The output operator was extended to emit the activation contract selected by
the concrete `to_out` GEMM:

- FP8 A8W8: E4M3 with FP32 per-1x128 scales.
- MXFP4 A4W4: natural packed E2M1 with shuffled E8M0 per-1x32 scales.
- MXFP6 A6W6 and A6W4: Hadamard-rotated FP6 with packed E8M0 per-1x32 scales.

At the production eight-GPU shape, all payloads, scales, and resulting GEMM
outputs matched their canonical quantizers and GEMMs bit-for-bit. Pure-format
baseline and MORI videos were byte-identical.

One compiled E2E iteration per path measured:

- FP8: `45.1888 -> 44.6428 s`, `+1.21%`.
- MXFP4: `28.5162 -> 28.7286 s`, `-0.75%`.
- A6W6: `29.7537 -> 31.7071 s`, `-6.57%`.
- A6W4: `29.2349 -> 31.2249 s`, `-6.81%`.

The direct output-MORI kernel itself measured approximately:

- E4M3: `0.163 ms`.
- MXFP4: `0.272 ms`.
- MXFP6: `0.916 ms`.

MXFP6 regresses because producing the GEMM's rotated, tiled FP6 physical
layout inside the communication kernel is much more expensive than E4M3
conversion. MXFP4 is close to neutral at E2E and does not establish a gain.

The FP8/MXFP4 hybrid schedule measured `34.4784 s`. Two baseline runs measured
`35.3356 s` and `34.6520 s`; their `0.684 s` range is larger than the
candidate's `0.515 s` improvement over the baseline mean, so no stable hybrid
uplift is demonstrated. Hybrid baseline repeats also produced different video
hashes, indicating existing run-to-run nondeterminism in that schedule.

Artifacts:

- `benchmark_results/output_mori_codecs_2026-10-26/`
- `benchmark_results/output_mori_gemm_matrix_8gpu_2026-10-26/`
