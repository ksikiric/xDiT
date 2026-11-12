import functools
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice

_BLOCK_M = 256
_BLOCK_N = 256
_K_TILE = 128
_PRESHUFFLE_LAYOUT = (16, 16)
# Read once at import: an environment read per call would land inside the traced region.
_FUSED_BIAS = os.environ.get("XFUSER_AITER_INT8_FUSED_BIAS") == "1"
_SHARED_QUANT = os.environ.get("XFUSER_AITER_INT8_SHARED_QUANT") == "1"


@dataclass(frozen=True)
class AiterInt8Variant:
    name: str
    persistent: bool


def supports_aiter_int8_linear(in_features: int, out_features: int) -> bool:
    return (
        in_features >= 2 * _K_TILE
        and in_features % _K_TILE == 0
        and out_features % _PRESHUFFLE_LAYOUT[0] == 0
    )


def select_aiter_int8_variant(
    m: int,
    n: int,
    k: int,
    cu_num: int,
) -> AiterInt8Variant:
    """Choose measured Wan specializations, otherwise the default 4-wave path."""
    tiles = math.ceil(m / _BLOCK_M) * math.ceil(n / _BLOCK_N)
    if k == 5120 and n in (5120, 13824) and tiles > cu_num:
        return AiterInt8Variant("wan_k5120_persistent", persistent=True)
    if k == 13824 and n == 5120:
        return AiterInt8Variant("wan_k13824", persistent=False)
    return AiterInt8Variant("default", persistent=False)


@triton.jit
def _quantize_rows_int8_kernel(
    output,
    scale_output,
    source,
    columns: tl.constexpr,
    source_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    mask = column < columns
    values = tl.load(
        source + row * source_stride + column,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=0)
    scale = amax / 127.0
    reciprocal = tl.where(amax > 0.0, 1.0 / scale, 0.0)
    quantized = libdevice.rint(values * reciprocal)
    quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0)
    tl.store(scale_output + row, scale)
    tl.store(
        output + row * columns + column,
        quantized.to(tl.int8),
        mask=mask,
        cache_modifier=".cs",
    )


def _quantize_rows_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = tensor.contiguous()
    quantized = torch.empty_like(tensor, dtype=torch.int8)
    scale = torch.empty(
        (tensor.shape[0],),
        dtype=torch.float32,
        device=tensor.device,
    )
    _quantize_rows_int8_kernel[(tensor.shape[0],)](
        quantized,
        scale,
        tensor,
        tensor.shape[1],
        tensor.stride(0),
        BLOCK=triton.next_power_of_2(tensor.shape[1]),
    )
    return quantized, scale


@functools.lru_cache(maxsize=16)
def _compile_4wave(k: int, persistent: bool, bias: bool = False):
    from aiter.jit.utils.chip_info import get_cu_num
    from aiter.ops.flydsl.kernels.gemm_a8w8_int8_4wave import (
        compile_int8_gemm_4w,
    )

    return compile_int8_gemm_4w(
        K=k,
        xcd_swizzle=8,
        num_cu=get_cu_num() if persistent else 0,
        bias=bias,
    )


def _launch_int8_gemm(
    x_quantized: torch.Tensor,
    x_scale: torch.Tensor,
    weight_preshuffled: torch.Tensor,
    column_scales: torch.Tensor,
    bias: bool,
) -> torch.Tensor:
    import flydsl.expr as fx

    from aiter.jit.utils.chip_info import get_cu_num
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    m, k = x_quantized.shape
    n, weight_k = weight_preshuffled.shape
    if weight_k != k:
        raise ValueError(f"AITER INT8 GEMM K mismatch: {k} and {weight_k}")
    variant = select_aiter_int8_variant(m, n, k, get_cu_num())
    executable = _compile_4wave(k, variant.persistent, bias)
    output = torch.empty(
        (m, n),
        dtype=torch.bfloat16,
        device=x_quantized.device,
    )
    _run_compiled(
        executable,
        x_quantized.view(-1),
        weight_preshuffled.view(-1),
        output.view(-1),
        x_scale.reshape(-1),
        column_scales.reshape(-1),
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )
    return output


@torch.library.custom_op("xfuser::aiter_int8_gemm", mutates_args=())
def _aiter_int8_gemm(
    input_2d: torch.Tensor,
    weight_preshuffled: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    if input_2d.dim() != 2 or weight_preshuffled.dim() != 2:
        raise ValueError("AITER INT8 GEMM operands must be two-dimensional")
    if input_2d.dtype != torch.bfloat16:
        raise TypeError("AITER INT8 GEMM activation must use torch.bfloat16")
    x_quantized, x_scale = _quantize_rows_int8(input_2d)
    return _launch_int8_gemm(x_quantized, x_scale, weight_preshuffled, weight_scale, False)


@_aiter_int8_gemm.register_fake
def _(
    input_2d: torch.Tensor,
    weight_preshuffled: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (input_2d.shape[0], weight_preshuffled.shape[0]),
        dtype=torch.bfloat16,
        device=input_2d.device,
    )


@torch.library.custom_op("xfuser::aiter_int8_quantize_rows", mutates_args=())
def _aiter_int8_quantize_rows(input_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _quantize_rows_int8(input_2d)


@_aiter_int8_quantize_rows.register_fake
def _(input_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(input_2d, dtype=torch.int8),
        input_2d.new_empty((input_2d.shape[0],), dtype=torch.float32),
    )


@torch.library.custom_op("xfuser::aiter_int8_gemm_quantized", mutates_args=())
def _aiter_int8_gemm_quantized(
    x_quantized: torch.Tensor,
    x_scale: torch.Tensor,
    weight_preshuffled: torch.Tensor,
    column_scales: torch.Tensor,
    bias: bool,
) -> torch.Tensor:
    """``column_scales`` holds the N weight scales, then N FP32 biases when ``bias``."""
    return _launch_int8_gemm(x_quantized, x_scale, weight_preshuffled, column_scales, bias)


@_aiter_int8_gemm_quantized.register_fake
def _(
    x_quantized: torch.Tensor,
    x_scale: torch.Tensor,
    weight_preshuffled: torch.Tensor,
    column_scales: torch.Tensor,
    bias: bool,
) -> torch.Tensor:
    return torch.empty(
        (x_quantized.shape[0], weight_preshuffled.shape[0]),
        dtype=torch.bfloat16,
        device=x_quantized.device,
    )


def quantize_shared_input(linears, input: torch.Tensor):
    """Quantize ``input`` once for AITER INT8 linears that all read it, or return None."""
    if not _SHARED_QUANT or not all(
        isinstance(m, xFuserAiterInt8Linear) and hasattr(m, "weight_int8") for m in linears
    ):
        return None
    return torch.ops.xfuser.aiter_int8_quantize_rows(
        input.reshape(-1, input.shape[-1]).contiguous()
    )


class xFuserAiterInt8Linear(nn.Module):
    """Drop-in BF16 Linear backed by the gfx950 4-wave true-INT8 GEMM."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dtype = torch.bfloat16 if dtype is None else dtype
        if dtype != torch.bfloat16:
            raise TypeError("AITER INT8 linear supports only torch.bfloat16")
        if not supports_aiter_int8_linear(in_features, out_features):
            raise ValueError(
                "AITER INT8 linear requires K >= 256, K % 128 == 0, "
                f"and N % 16 == 0; got N={out_features}, K={in_features}"
            )

        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def _is_fsdp_managed_parameter(parameter) -> bool:
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:
            return False
        return isinstance(parameter, DTensor)

    def _install_quantized_state(
        self,
        weight_preshuffled: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> None:
        if self.weight is not None and self._is_fsdp_managed_parameter(self.weight):
            raise RuntimeError("Quantize AITER INT8 weights before fully_shard")
        if hasattr(self, "weight_int8"):
            delattr(self, "weight_int8")
        if hasattr(self, "weight_scale"):
            delattr(self, "weight_scale")
        if hasattr(self, "column_scales_bias"):
            delattr(self, "column_scales_bias")
        if self.weight is not None:
            delattr(self, "weight")
            self.register_parameter("weight", None)
        self.register_parameter(
            "weight_int8",
            nn.Parameter(weight_preshuffled.detach(), requires_grad=False),
        )
        self.register_buffer(
            "weight_scale",
            weight_scale.detach(),
            persistent=True,
        )

    def load_and_quantize_weights(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        from aiter.ops.shuffle import shuffle_weight

        expected = (self.out_features, self.in_features)
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"AITER INT8 weight must have shape {expected}, got {tuple(weight.shape)}"
            )
        if weight.dtype != torch.bfloat16:
            raise TypeError("AITER INT8 weights must use torch.bfloat16")
        target = torch.device(device) if device is not None else weight.device
        if target.type == "meta":
            raise RuntimeError("AITER INT8 weights cannot be quantized on meta")
        full_weight = weight.detach().to(target)
        weight_quantized, weight_scale = _quantize_rows_int8(full_weight)
        weight_preshuffled = shuffle_weight(
            weight_quantized,
            layout=_PRESHUFFLE_LAYOUT,
        ).contiguous()
        self._install_quantized_state(weight_preshuffled, weight_scale)
        if bias is not None and self.bias is not None:
            self.bias = nn.Parameter(bias.detach().to(target))
        if _FUSED_BIAS and self.bias is not None:
            self.register_buffer(
                "column_scales_bias",
                torch.cat([self.weight_scale, self.bias.detach().float()]),
                persistent=False,
            )

    def forward(
        self,
        input: torch.Tensor,
        quantized: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """``quantized`` is ``input`` already quantized by ``quantize_shared_input``."""
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"AITER INT8 input has {input.shape[-1]} features, "
                f"expected {self.in_features}"
            )
        if input.dtype != torch.bfloat16:
            raise TypeError("AITER INT8 activations must use torch.bfloat16")
        if not hasattr(self, "weight_int8"):
            self.load_and_quantize_weights(self.weight)

        original_shape = input.shape
        fused_bias = hasattr(self, "column_scales_bias")
        if quantized is None and not fused_bias:
            input_2d = input.reshape(-1, self.in_features).contiguous()
            output = torch.ops.xfuser.aiter_int8_gemm(
                input_2d,
                self.weight_int8,
                self.weight_scale,
            )
        else:
            if quantized is None:
                quantized = torch.ops.xfuser.aiter_int8_quantize_rows(
                    input.reshape(-1, self.in_features).contiguous()
                )
            output = torch.ops.xfuser.aiter_int8_gemm_quantized(
                *quantized,
                self.weight_int8,
                self.column_scales_bias if fused_bias else self.weight_scale,
                fused_bias,
            )
        if self.bias is not None and not fused_bias:
            output = output + self.bias
        return output.reshape(*original_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
