import torch
import torch.nn as nn
import math
from typing import Optional

try:
    from aiter.ops.gemm_op_a6w6 import (
        gemm_a6w6,
        mxfp6_gemm_pack_size,
        quant_mxfp6_gemm,
        quant_mxfp6_gemm_out,
    )
except ImportError:
    pass  # Error will be thrown in base_model.py if mxfp6 gemms are enabled but AITER is not available.


@torch.library.custom_op("mylib::mxfp6_gemm", mutates_args=())
def _mxfp6_gemm(
    a: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    N: int,
    K: int,
) -> torch.Tensor:
    # activation quantize+pack is fused (single Triton pass) -> cheap enough per-call
    a_packed, a_scale = quant_mxfp6_gemm(a)
    return gemm_a6w6(a_packed, w_packed, a_scale, w_scale, a.shape[0], N, K)


@_mxfp6_gemm.register_fake
def _(a: torch.Tensor, w_packed: torch.Tensor, w_scale: torch.Tensor, N: int, K: int) -> torch.Tensor:
    return torch.empty(a.shape[0], N, dtype=torch.bfloat16, device=a.device)


@torch.library.custom_op("mylib::mxfp6_gemm_packed", mutates_args=())
def _mxfp6_gemm_packed(
    a_packed: torch.Tensor,
    a_scale: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    M: int,
    N: int,
    K: int,
) -> torch.Tensor:
    return gemm_a6w6(a_packed, w_packed, a_scale, w_scale, M, N, K)


@_mxfp6_gemm_packed.register_fake
def _(
    a_packed: torch.Tensor,
    a_scale: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    M: int,
    N: int,
    K: int,
) -> torch.Tensor:
    return torch.empty(M, N, dtype=torch.bfloat16, device=a_packed.device)


class xFuserMXFP6Linear(nn.Module):
    """
    Custom Linear layer using MXFP6 (E2M3, per-1x32 blockscale) GEMM.

    Drop-in replacement for nn.Linear. Accuracy is ~equivalent to FP8
    (cos ~0.999 vs bf16) while the GEMM is substantially faster, so it is a
    strong replacement for FP8 in the precision-sensitive layers.
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
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

    def load_and_quantize_weights(
        self, weights: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> None:
        with torch.no_grad():
            if self.weight is None:
                self.weight = nn.Parameter(
                    torch.empty_like(weights, device=weights.device, dtype=weights.dtype)
                )
            self.weight.data.copy_(weights.data)
            if bias is not None and self.bias is not None:
                self.bias.data.copy_(bias.data)
        self._quantize_weights()

    def _quantize_weights(self) -> None:
        if self.weight is None:
            raise RuntimeError(
                "Cannot quantize: weight parameter is None. "
                "Call load_and_quantize_weights() or reset_parameters() first."
            )
        # pack weight once at load time (mxfp6 tile layout + e8m0 scales)
        weight_packed, weight_scale = quant_mxfp6_gemm(self.weight)
        self.register_buffer("weight_packed", weight_packed, persistent=True)
        self.register_buffer("weight_scale", weight_scale, persistent=True)
        delattr(self, "weight")
        self.register_parameter("weight", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "weight_packed"):
            self._quantize_weights()
        original_shape = input.shape
        input_2d = input.view(-1, self.in_features)
        a_packed, a_scale = self.pack_activation(input_2d)
        output = self.forward_packed_2d(a_packed, a_scale, input_2d.shape[0], input.dtype)
        return output.view(*original_shape[:-1], self.out_features)

    def pack_activation(self, input_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return quant_mxfp6_gemm(input_2d)

    def activation_pack_size(self, rows: int) -> tuple[int, int]:
        return mxfp6_gemm_pack_size(rows, self.in_features)

    def pack_activation_out(
        self,
        input_2d: torch.Tensor,
        packed: torch.Tensor,
        packed_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return quant_mxfp6_gemm_out(input_2d, packed, packed_scale)

    def forward_packed_2d(
        self,
        a_packed: torch.Tensor,
        a_scale: torch.Tensor,
        M: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if not hasattr(self, "weight_packed"):
            self._quantize_weights()
        output = torch.ops.mylib.mxfp6_gemm_packed(
            a_packed,
            a_scale,
            self.weight_packed,
            self.weight_scale,
            M,
            self.out_features,
            self.in_features,
        ).to(output_dtype)
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
