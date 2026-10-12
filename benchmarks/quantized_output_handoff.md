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
