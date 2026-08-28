import math

import torch
from torch import nn

# Preflight preserves and reports arbitrary import-time dependency failures.
try:
    from aiter.ops.gemm_op_a6w6 import (
        gemm_a6w6,
        mxfp6_gemm_pack_size,
        quant_mxfp6_gemm,
        quant_mxfp6_gemm_out,
    )
except Exception as exc:  # noqa: BLE001
    _AITER_MXFP6_IMPORT_ERROR = exc
    gemm_a6w6 = None
    mxfp6_gemm_pack_size = None
    quant_mxfp6_gemm = None
    quant_mxfp6_gemm_out = None
else:
    _AITER_MXFP6_IMPORT_ERROR = None


_SUPPORTED_COMPUTE_DTYPES = frozenset({torch.bfloat16, torch.float16})
_PACK_TILE = 256
_PACK_K_TILE = 128
_PACK_K_GUARD_TILES = 2
_PACKED_TILE_BYTES = 24576
_SCALE_TILE_BYTES = 1024
# Provenance for AITER's mxfp6_c0c1_256_padk2 layout with the mandatory
# 32-element Hadamard rotation. Increment on any persistent layout change.
_PACK_LAYOUT_ABI_VERSION = 1
_DTYPE_TO_CODE = {torch.bfloat16: 1, torch.float16: 2}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}


def _require_aiter_api(name: str, value):
    if callable(value):
        return value
    detail = (
        f": {type(_AITER_MXFP6_IMPORT_ERROR).__name__}: " f"{_AITER_MXFP6_IMPORT_ERROR}"
        if _AITER_MXFP6_IMPORT_ERROR is not None
        else ""
    )
    raise RuntimeError(f"AITER MXFP6 API {name} is unavailable{detail}")


def _abi_pack_sizes(rows, in_features):
    """The public AITER A6W6 C0/C1 packed-layout element counts."""

    padded_rows = ((rows + _PACK_TILE - 1) // _PACK_TILE) * _PACK_TILE
    padded_k = ((in_features + _PACK_K_TILE - 1) // _PACK_K_TILE) * _PACK_K_TILE
    row_tiles = padded_rows // _PACK_TILE
    k_tiles_with_guards = padded_k // _PACK_K_TILE + _PACK_K_GUARD_TILES
    return (
        row_tiles * k_tiles_with_guards * _PACKED_TILE_BYTES,
        row_tiles * k_tiles_with_guards * _SCALE_TILE_BYTES,
    )


def _expected_pack_sizes(rows: int, in_features: int) -> tuple[int, int]:
    abi_sizes = _abi_pack_sizes(rows, in_features)
    if callable(mxfp6_gemm_pack_size):
        sizes = mxfp6_gemm_pack_size(rows, in_features)
    else:
        sizes = abi_sizes
    if not isinstance(sizes, tuple) or len(sizes) != 2:
        raise RuntimeError(
            "AITER mxfp6_gemm_pack_size must return "
            "(packed_elements, scale_elements)"
        )
    packed_size, scale_size = (int(sizes[0]), int(sizes[1]))
    if packed_size < 0 or scale_size < 0:
        raise RuntimeError(
            "AITER mxfp6_gemm_pack_size returned negative element counts: " f"{sizes}"
        )
    if (packed_size, scale_size) != abi_sizes:
        raise RuntimeError(
            "AITER mxfp6_gemm_pack_size does not match the required "
            f"C0/C1+Hadamard ABI: got {(packed_size, scale_size)}, "
            f"expected {abi_sizes}"
        )
    return packed_size, scale_size


def _is_compiling() -> bool:
    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    return bool(is_compiling and is_compiling())


def _validate_compute_matrix(
    tensor: torch.Tensor,
    *,
    name: str,
    expected_rows: int | None = None,
    expected_features: int | None = None,
) -> None:
    if tensor.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D [rows, features] tensor, got "
            f"shape {tuple(tensor.shape)}"
        )
    if expected_rows is not None and tensor.shape[0] != expected_rows:
        raise ValueError(f"{name} has {tensor.shape[0]} rows, expected {expected_rows}")
    if expected_features is not None and tensor.shape[1] != expected_features:
        raise ValueError(
            f"{name} has last dimension {tensor.shape[1]}, "
            f"expected {expected_features}"
        )
    if tensor.dtype not in _SUPPORTED_COMPUTE_DTYPES:
        supported = ", ".join(str(dtype) for dtype in _SUPPORTED_COMPUTE_DTYPES)
        raise TypeError(f"{name} must use one of ({supported}), got {tensor.dtype}")
    if tensor.device.type == "meta":
        raise RuntimeError(f"{name} cannot be packed from the meta device")


def _validate_packed_pair(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    rows: int,
    in_features: int,
    name: str,
) -> None:
    if rows < 0:
        raise ValueError(f"{name} rows must be non-negative, got {rows}")
    if packed.ndim != 1 or scale.ndim != 1:
        raise ValueError(
            f"{name} packed tensors must both be 1-D, got "
            f"{tuple(packed.shape)} and {tuple(scale.shape)}"
        )
    if packed.dtype is not torch.uint8 or scale.dtype is not torch.uint8:
        raise TypeError(
            f"{name} packed tensors must both use torch.uint8, got "
            f"{packed.dtype} and {scale.dtype}"
        )
    if not packed.is_contiguous() or not scale.is_contiguous():
        raise ValueError(f"{name} packed values and scales must both be contiguous")
    if packed.device != scale.device:
        raise ValueError(
            f"{name} packed values and scales must share a device, got "
            f"{packed.device} and {scale.device}"
        )
    expected_packed, expected_scale = _expected_pack_sizes(rows, in_features)
    actual = (packed.numel(), scale.numel())
    expected = (expected_packed, expected_scale)
    if actual != expected:
        raise ValueError(
            f"{name} packed tensors have wrong sizes: got {actual}, "
            f"expected {expected} for logical shape ({rows}, {in_features})"
        )


def _packed_provenance(
    out_features: int,
    in_features: int,
    compute_dtype: torch.dtype,
    *,
    device,
) -> torch.Tensor:
    try:
        dtype_code = _DTYPE_TO_CODE[compute_dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported MXFP6 compute dtype: {compute_dtype}") from exc
    return torch.tensor(
        [
            _PACK_LAYOUT_ABI_VERSION,
            out_features,
            in_features,
            dtype_code,
        ],
        dtype=torch.int64,
        device=device,
    )


def _validate_packed_provenance(
    provenance: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    name: str,
) -> torch.dtype:
    if provenance.ndim != 1 or provenance.numel() != 4:
        raise ValueError(f"{name} provenance must be a 4-element 1-D tensor")
    if provenance.dtype is not torch.int64:
        raise TypeError(f"{name} provenance must use torch.int64")
    if not provenance.is_contiguous():
        raise ValueError(f"{name} provenance must be contiguous")
    if provenance.device.type == "meta":
        raise RuntimeError(f"{name} provenance cannot be validated on meta")
    abi_version, logical_n, logical_k, dtype_code = (
        int(value) for value in provenance.tolist()
    )
    if abi_version != _PACK_LAYOUT_ABI_VERSION:
        raise ValueError(
            f"{name} layout ABI version {abi_version} does not match "
            f"required C0/C1+Hadamard ABI {_PACK_LAYOUT_ABI_VERSION}"
        )
    if (logical_n, logical_k) != (out_features, in_features):
        raise ValueError(
            f"{name} logical shape ({logical_n}, {logical_k}) does not match "
            f"destination ({out_features}, {in_features})"
        )
    try:
        return _CODE_TO_DTYPE[dtype_code]
    except KeyError as exc:
        raise ValueError(f"{name} has unknown compute dtype code {dtype_code}") from exc


@torch.library.custom_op("xfuser::mxfp6_pack", mutates_args=())
def _mxfp6_pack(input_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_compute_matrix(input_2d, name="MXFP6 activation")
    if input_2d.shape[0] == 0:
        return (
            torch.empty(0, dtype=torch.uint8, device=input_2d.device),
            torch.empty(0, dtype=torch.uint8, device=input_2d.device),
        )
    pack = _require_aiter_api(
        "aiter.ops.gemm_op_a6w6.quant_mxfp6_gemm",
        quant_mxfp6_gemm,
    )
    packed, scale = pack(input_2d)
    _validate_packed_pair(
        packed,
        scale,
        rows=input_2d.shape[0],
        in_features=input_2d.shape[1],
        name="MXFP6 activation",
    )
    return packed, scale


@_mxfp6_pack.register_fake
def _(input_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    packed_size, scale_size = _abi_pack_sizes(input_2d.shape[0], input_2d.shape[1])
    return (
        torch.empty(packed_size, dtype=torch.uint8, device=input_2d.device),
        torch.empty(scale_size, dtype=torch.uint8, device=input_2d.device),
    )


def _run_aiter_gemm(
    activation_packed: torch.Tensor,
    weight_packed: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    rows: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    if rows < 0:
        raise ValueError(f"MXFP6 GEMM rows must be non-negative, got {rows}")
    if rows == 0:
        return torch.empty(
            (0, out_features),
            dtype=torch.bfloat16,
            device=activation_packed.device,
        )
    gemm = _require_aiter_api(
        "aiter.ops.gemm_op_a6w6.gemm_a6w6",
        gemm_a6w6,
    )
    output = gemm(
        activation_packed,
        weight_packed,
        activation_scale,
        weight_scale,
        rows,
        out_features,
        in_features,
    )
    if output.ndim != 2 or tuple(output.shape) != (rows, out_features):
        raise RuntimeError(
            "AITER gemm_a6w6 returned shape "
            f"{tuple(output.shape)}, expected ({rows}, {out_features})"
        )
    if output.dtype is not torch.bfloat16:
        raise RuntimeError(
            "AITER gemm_a6w6 must return torch.bfloat16, " f"got {output.dtype}"
        )
    if output.device != activation_packed.device:
        raise RuntimeError(
            "AITER gemm_a6w6 returned output on "
            f"{output.device}, expected {activation_packed.device}"
        )
    return output.contiguous()


@torch.library.custom_op("xfuser::mxfp6_gemm", mutates_args=())
def _mxfp6_gemm(
    input_2d: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    _validate_compute_matrix(
        input_2d,
        name="MXFP6 activation",
        expected_features=in_features,
    )
    _validate_packed_pair(
        weight_packed,
        weight_scale,
        rows=out_features,
        in_features=in_features,
        name="MXFP6 weight",
    )
    if input_2d.device != weight_packed.device:
        raise ValueError(
            "MXFP6 activation and packed weight must share a device, got "
            f"{input_2d.device} and {weight_packed.device}"
        )
    if input_2d.shape[0] == 0:
        return torch.empty(
            (0, out_features),
            dtype=torch.bfloat16,
            device=input_2d.device,
        )
    pack = _require_aiter_api(
        "aiter.ops.gemm_op_a6w6.quant_mxfp6_gemm",
        quant_mxfp6_gemm,
    )
    activation_packed, activation_scale = pack(input_2d)
    _validate_packed_pair(
        activation_packed,
        activation_scale,
        rows=input_2d.shape[0],
        in_features=in_features,
        name="MXFP6 activation",
    )
    return _run_aiter_gemm(
        activation_packed,
        weight_packed,
        activation_scale,
        weight_scale,
        input_2d.shape[0],
        out_features,
        in_features,
    )


@_mxfp6_gemm.register_fake
def _(
    input_2d: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    return torch.empty(
        (input_2d.shape[0], out_features),
        dtype=torch.bfloat16,
        device=input_2d.device,
    )


@torch.library.custom_op("xfuser::mxfp6_gemm_packed", mutates_args=())
def _mxfp6_gemm_packed(
    activation_packed: torch.Tensor,
    weight_packed: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    rows: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    if rows < 0:
        raise ValueError(f"MXFP6 GEMM rows must be non-negative, got {rows}")
    _validate_packed_pair(
        activation_packed,
        activation_scale,
        rows=rows,
        in_features=in_features,
        name="MXFP6 activation",
    )
    _validate_packed_pair(
        weight_packed,
        weight_scale,
        rows=out_features,
        in_features=in_features,
        name="MXFP6 weight",
    )
    if activation_packed.device != weight_packed.device:
        raise ValueError(
            "MXFP6 packed activation and weight must share a device, got "
            f"{activation_packed.device} and {weight_packed.device}"
        )
    return _run_aiter_gemm(
        activation_packed,
        weight_packed,
        activation_scale,
        weight_scale,
        rows,
        out_features,
        in_features,
    )


@_mxfp6_gemm_packed.register_fake
def _(
    activation_packed: torch.Tensor,
    weight_packed: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    rows: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    return torch.empty(
        (rows, out_features),
        dtype=torch.bfloat16,
        device=activation_packed.device,
    )


class xFuserMXFP6Linear(nn.Module):
    """Inference-only MXFP6 linear using AITER's packed A6W6 ABI."""

    _version = 2

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                "MXFP6 logical feature dimensions must be positive, got "
                f"in_features={in_features}, out_features={out_features}"
            )
        # A packed A6W6 tensor is one flat byte blob, so its logical dimensions
        # must remain explicit rather than being inferred from parameter shape.
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.logical_in_features = self.in_features
        self.logical_out_features = self.out_features
        dtype = torch.bfloat16 if dtype is None else dtype
        if dtype not in _SUPPORTED_COMPUTE_DTYPES:
            raise TypeError(
                "MXFP6 linear dtype must be torch.bfloat16 or torch.float16, "
                f"got {dtype}"
            )
        self._compute_dtype = dtype
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

    def _is_fsdp_managed_parameter(self, parameter) -> bool:
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:
            return False
        return isinstance(parameter, DTensor)

    def _remove_packed_state(self) -> None:
        if hasattr(self, "weight_packed"):
            delattr(self, "weight_packed")
        if hasattr(self, "weight_scale"):
            delattr(self, "weight_scale")
        if hasattr(self, "weight_packed_provenance"):
            delattr(self, "weight_packed_provenance")

    def _install_packed_state(
        self,
        weight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
        provenance: torch.Tensor,
    ) -> None:
        _validate_packed_pair(
            weight_packed,
            weight_scale,
            rows=self.out_features,
            in_features=self.in_features,
            name="MXFP6 weight",
        )
        provenance_dtype = _validate_packed_provenance(
            provenance,
            out_features=self.out_features,
            in_features=self.in_features,
            name="MXFP6 weight",
        )
        if provenance.device != weight_packed.device:
            raise ValueError(
                "MXFP6 packed weight provenance must share the packed device, "
                f"got {provenance.device} and {weight_packed.device}"
            )
        if provenance_dtype != self._compute_dtype:
            raise TypeError(
                f"MXFP6 packed provenance dtype {provenance_dtype} does not "
                f"match layer compute dtype {self._compute_dtype}"
            )
        self._remove_packed_state()
        if self.weight is not None:
            delattr(self, "weight")
            self.register_parameter("weight", None)
        self.register_parameter(
            "weight_packed",
            nn.Parameter(weight_packed.detach(), requires_grad=False),
        )
        self.register_buffer(
            "weight_scale",
            weight_scale.detach(),
            persistent=True,
        )
        self.register_buffer(
            "weight_packed_provenance",
            provenance.detach(),
            persistent=True,
        )

    def _validate_full_weight(self, weight: torch.Tensor) -> None:
        if weight.ndim != 2 or tuple(weight.shape) != (
            self.out_features,
            self.in_features,
        ):
            raise ValueError(
                "MXFP6 full-precision weight must have shape "
                f"({self.out_features}, {self.in_features}), got "
                f"{tuple(weight.shape)}"
            )
        if weight.dtype not in _SUPPORTED_COMPUTE_DTYPES:
            raise TypeError(
                "MXFP6 full-precision weight must use torch.bfloat16 or "
                f"torch.float16, got {weight.dtype}"
            )
        if weight.device.type == "meta":
            raise RuntimeError("MXFP6 cannot quantize a weight on the meta device")

    def _validate_bias(self, bias: torch.Tensor, *, dtype: torch.dtype) -> None:
        if bias.ndim != 1 or tuple(bias.shape) != (self.out_features,):
            raise ValueError(
                f"MXFP6 bias must have shape ({self.out_features},), got "
                f"{tuple(bias.shape)}"
            )
        if bias.dtype != dtype:
            raise TypeError(
                f"MXFP6 bias dtype {bias.dtype} does not match weight dtype {dtype}"
            )
        if bias.device.type == "meta":
            raise RuntimeError("MXFP6 cannot load a bias from the meta device")

    def load_and_quantize_weights(
        self,
        weights: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        device: torch.device | None = None,
    ) -> None:
        self._validate_full_weight(weights)
        if bias is not None:
            if self.bias is None:
                raise ValueError("MXFP6 layer was constructed without a bias")
            self._validate_bias(bias, dtype=weights.dtype)
        target = torch.device(device) if device is not None else weights.device
        if target.type == "meta":
            raise RuntimeError("MXFP6 quantization target cannot be the meta device")
        full_weight = weights.detach().to(device=target)
        self._compute_dtype = full_weight.dtype
        if self.weight is not None and self._is_fsdp_managed_parameter(self.weight):
            raise RuntimeError(
                "MXFP6 full-precision weight cannot be quantized after FSDP "
                "wrapping; quantize before fully_shard."
            )
        if self.weight is not None:
            delattr(self, "weight")
        self.register_parameter("weight", nn.Parameter(full_weight))
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().to(device=target))
        self._quantize_weights()

    def _quantize_weights(self) -> None:
        if self.weight is None:
            raise RuntimeError(
                "Cannot quantize MXFP6 weight: full-precision weight is absent. "
                "Load full-precision state before quantization."
            )
        if self._is_fsdp_managed_parameter(self.weight):
            raise RuntimeError(
                "MXFP6 full-precision weight cannot transition to packed state "
                "after FSDP wrapping; quantize before fully_shard."
            )
        self._validate_full_weight(self.weight)
        pack_out = _require_aiter_api(
            "aiter.ops.gemm_op_a6w6.quant_mxfp6_gemm_out",
            quant_mxfp6_gemm_out,
        )
        packed_size, scale_size = _expected_pack_sizes(
            self.out_features, self.in_features
        )
        # AITER may intentionally leave guard/padding bytes untouched. Persistent
        # state must therefore start from deterministic caller-owned storage.
        weight_packed = torch.zeros(
            packed_size, dtype=torch.uint8, device=self.weight.device
        )
        weight_scale = torch.zeros(
            scale_size, dtype=torch.uint8, device=self.weight.device
        )
        pack_out(self.weight, weight_packed, weight_scale)
        _validate_packed_pair(
            weight_packed,
            weight_scale,
            rows=self.out_features,
            in_features=self.in_features,
            name="MXFP6 weight",
        )
        self._compute_dtype = self.weight.dtype
        provenance = _packed_provenance(
            self.out_features,
            self.in_features,
            self._compute_dtype,
            device=weight_packed.device,
        )
        self._install_packed_state(weight_packed, weight_scale, provenance)

    @staticmethod
    def _destination_device(current, incoming) -> torch.device:
        return incoming.device if current.device.type == "meta" else current.device

    def _materialize_meta_bias(
        self,
        state_dict,
        prefix: str,
        destination_device: torch.device,
        destination_dtype: torch.dtype,
    ) -> None:
        bias_key = prefix + "bias"
        if (
            bias_key in state_dict
            and self.bias is not None
            and self.bias.device.type == "meta"
        ):
            incoming = state_dict[bias_key]
            self.bias = nn.Parameter(
                torch.empty(
                    incoming.shape,
                    dtype=destination_dtype,
                    device=destination_device,
                )
            )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        weight_key = prefix + "weight"
        packed_key = prefix + "weight_packed"
        scale_key = prefix + "weight_scale"
        provenance_key = prefix + "weight_packed_provenance"
        has_packed = packed_key in state_dict
        has_scale = scale_key in state_dict
        has_provenance = provenance_key in state_dict
        if len({has_packed, has_scale, has_provenance}) != 1:
            raise RuntimeError(
                "MXFP6 packed state requires weight_packed, weight_scale, and "
                "weight_packed_provenance"
            )
        if has_packed and weight_key in state_dict:
            raise RuntimeError(
                "MXFP6 checkpoint cannot contain both full and packed weights"
            )

        if has_packed:
            incoming_packed = state_dict[packed_key]
            incoming_scale = state_dict[scale_key]
            incoming_provenance = state_dict[provenance_key]
            _validate_packed_pair(
                incoming_packed,
                incoming_scale,
                rows=self.out_features,
                in_features=self.in_features,
                name="MXFP6 checkpoint weight",
            )
            provenance_dtype = _validate_packed_provenance(
                incoming_provenance,
                out_features=self.out_features,
                in_features=self.in_features,
                name="MXFP6 checkpoint weight",
            )
            if (
                incoming_provenance.device != incoming_packed.device
                or incoming_scale.device != incoming_packed.device
            ):
                raise ValueError(
                    "MXFP6 checkpoint packed values, scales, and provenance "
                    "must share a device"
                )
            if provenance_dtype != self._compute_dtype:
                raise TypeError(
                    f"MXFP6 checkpoint compute dtype {provenance_dtype} does "
                    f"not match destination dtype {self._compute_dtype}"
                )
            bias_key = prefix + "bias"
            if bias_key in state_dict:
                incoming_bias = state_dict[bias_key]
                if incoming_bias.dtype != provenance_dtype:
                    raise TypeError(
                        f"MXFP6 checkpoint bias dtype {incoming_bias.dtype} "
                        f"does not match packed compute dtype {provenance_dtype}"
                    )
            current = (
                self.weight_packed if hasattr(self, "weight_packed") else self.weight
            )
            if self._is_fsdp_managed_parameter(current):
                raise RuntimeError(
                    "MXFP6 packed state cannot be loaded after FSDP wrapping; "
                    "load the packed checkpoint before fully_shard."
                )
            destination_device = self._destination_device(current, incoming_packed)
            self._remove_packed_state()
            if self.weight is not None:
                delattr(self, "weight")
                self.register_parameter("weight", None)
            self.register_parameter(
                "weight_packed",
                nn.Parameter(
                    torch.empty(
                        incoming_packed.shape,
                        dtype=incoming_packed.dtype,
                        device=destination_device,
                    ),
                    requires_grad=False,
                ),
            )
            self.register_buffer(
                "weight_scale",
                torch.empty(
                    incoming_scale.shape,
                    dtype=incoming_scale.dtype,
                    device=destination_device,
                ),
                persistent=True,
            )
            self.register_buffer(
                "weight_packed_provenance",
                torch.empty(
                    incoming_provenance.shape,
                    dtype=incoming_provenance.dtype,
                    device=destination_device,
                ),
                persistent=True,
            )
            self._materialize_meta_bias(
                state_dict,
                prefix,
                destination_device,
                provenance_dtype,
            )
        elif weight_key in state_dict:
            incoming_weight = state_dict[weight_key]
            if incoming_weight.ndim != 2 or tuple(incoming_weight.shape) != (
                self.out_features,
                self.in_features,
            ):
                raise RuntimeError(
                    "MXFP6 full-precision checkpoint weight has shape "
                    f"{tuple(incoming_weight.shape)}, expected "
                    f"({self.out_features}, {self.in_features})"
                )
            if hasattr(self, "weight_packed"):
                if self._is_fsdp_managed_parameter(self.weight_packed):
                    raise RuntimeError(
                        "MXFP6 full-precision state cannot replace an "
                        "FSDP-managed packed parameter; load the full-precision "
                        "checkpoint before fully_shard."
                    )
                destination_device = self._destination_device(
                    self.weight_packed, incoming_weight
                )
                self._remove_packed_state()
                delattr(self, "weight")
                self.register_parameter(
                    "weight",
                    nn.Parameter(
                        torch.empty(
                            incoming_weight.shape,
                            dtype=self._compute_dtype,
                            device=destination_device,
                        )
                    ),
                )
            elif self.weight.device.type == "meta":
                destination_device = incoming_weight.device
                delattr(self, "weight")
                self.register_parameter(
                    "weight",
                    nn.Parameter(
                        torch.empty(
                            incoming_weight.shape,
                            dtype=self._compute_dtype,
                            device=destination_device,
                        )
                    ),
                )
            else:
                destination_device = self.weight.device
            self._materialize_meta_bias(
                state_dict,
                prefix,
                destination_device,
                self._compute_dtype,
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if has_packed:
            loaded_dtype = _validate_packed_provenance(
                self.weight_packed_provenance,
                out_features=self.out_features,
                in_features=self.in_features,
                name="MXFP6 loaded weight",
            )
            if self.weight_packed_provenance.device != self.weight_packed.device:
                raise ValueError(
                    "MXFP6 loaded provenance and packed weight must share a device"
                )
            actual_dtype = self.bias.dtype if self.bias is not None else loaded_dtype
            if actual_dtype != loaded_dtype:
                raise TypeError(
                    f"MXFP6 loaded bias dtype {actual_dtype} does not match "
                    f"packed compute dtype {loaded_dtype}"
                )
            self._compute_dtype = actual_dtype
        elif weight_key in state_dict:
            if self.weight.dtype not in _SUPPORTED_COMPUTE_DTYPES:
                raise TypeError(
                    "MXFP6 destination weight has unsupported dtype "
                    f"{self.weight.dtype} after state loading"
                )
            if self.bias is not None and self.bias.dtype != self.weight.dtype:
                raise TypeError(
                    f"MXFP6 destination bias dtype {self.bias.dtype} does not "
                    f"match loaded weight dtype {self.weight.dtype}"
                )
            self._compute_dtype = self.weight.dtype

    def _ensure_packed_weight(self) -> None:
        if not hasattr(self, "weight_packed"):
            self._quantize_weights()
        _validate_packed_pair(
            self.weight_packed,
            self.weight_scale,
            rows=self.out_features,
            in_features=self.in_features,
            name="MXFP6 weight",
        )
        provenance = self.weight_packed_provenance
        if (
            provenance.ndim != 1
            or provenance.numel() != 4
            or provenance.dtype is not torch.int64
            or not provenance.is_contiguous()
        ):
            raise ValueError(
                "MXFP6 packed weight provenance must remain a contiguous "
                "4-element torch.int64 tensor"
            )
        if provenance.device != self.weight_packed.device:
            raise ValueError("MXFP6 packed weight and provenance must share a device")

    def _validate_activation(self, input_2d: torch.Tensor) -> None:
        _validate_compute_matrix(
            input_2d,
            name="MXFP6 activation",
            expected_features=self.in_features,
        )
        if input_2d.dtype != self._compute_dtype:
            raise TypeError(
                f"MXFP6 activation dtype {input_2d.dtype} does not match "
                f"packed weight compute dtype {self._compute_dtype}"
            )
        if input_2d.shape[0] == 0:
            storage = (
                self.weight_packed if hasattr(self, "weight_packed") else self.weight
            )
            if input_2d.device != storage.device:
                raise ValueError(
                    f"MXFP6 activation is on {input_2d.device}, but weight "
                    f"storage is on {storage.device}"
                )
            return
        self._ensure_packed_weight()
        if input_2d.device != self.weight_packed.device:
            raise ValueError(
                f"MXFP6 activation is on {input_2d.device}, but packed weight "
                f"is on {self.weight_packed.device}"
            )

    def pack_activation(
        self, input_2d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dynamically quantize and pack one logical 2-D activation."""

        self._validate_activation(input_2d)
        return torch.ops.xfuser.mxfp6_pack(input_2d)

    def activation_pack_size(self, rows: int) -> tuple[int, int]:
        if rows < 0:
            raise ValueError(f"MXFP6 activation rows must be non-negative, got {rows}")
        return _expected_pack_sizes(rows, self.in_features)

    def pack_activation_out(
        self,
        input_2d: torch.Tensor,
        packed: torch.Tensor,
        packed_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack into reusable AITER output buffers for shared-projection callers."""

        self._validate_activation(input_2d)
        _validate_packed_pair(
            packed,
            packed_scale,
            rows=input_2d.shape[0],
            in_features=self.in_features,
            name="MXFP6 activation output",
        )
        if packed.device != input_2d.device:
            raise ValueError(
                f"MXFP6 activation output buffers are on {packed.device}, "
                f"but activation is on {input_2d.device}"
            )
        if input_2d.shape[0] == 0:
            return packed, packed_scale
        pack_out = _require_aiter_api(
            "aiter.ops.gemm_op_a6w6.quant_mxfp6_gemm_out",
            quant_mxfp6_gemm_out,
        )
        pack_out(input_2d, packed, packed_scale)
        return packed, packed_scale

    def forward_packed_2d(
        self,
        activation_packed: torch.Tensor,
        activation_scale: torch.Tensor,
        rows: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run GEMM from a reusable activation pack without repacking it."""

        if rows < 0:
            raise ValueError(
                f"MXFP6 packed forward rows must be non-negative, got {rows}"
            )
        if output_dtype not in _SUPPORTED_COMPUTE_DTYPES:
            raise TypeError(
                "MXFP6 packed forward output dtype must be torch.bfloat16 or "
                f"torch.float16, got {output_dtype}"
            )
        if output_dtype != self._compute_dtype:
            raise TypeError(
                f"MXFP6 output dtype {output_dtype} does not match packed "
                f"weight compute dtype {self._compute_dtype}"
            )
        if _is_compiling():
            if (
                activation_packed.ndim != 1
                or activation_scale.ndim != 1
                or activation_packed.dtype is not torch.uint8
                or activation_scale.dtype is not torch.uint8
                or activation_packed.device != activation_scale.device
                or not activation_packed.is_contiguous()
                or not activation_scale.is_contiguous()
            ):
                raise ValueError(
                    "MXFP6 compiled packed activation must be matching 1-D "
                    "torch.uint8 value and scale tensors"
                )
        else:
            _validate_packed_pair(
                activation_packed,
                activation_scale,
                rows=rows,
                in_features=self.in_features,
                name="MXFP6 activation",
            )
        if rows == 0:
            storage = (
                self.weight_packed if hasattr(self, "weight_packed") else self.weight
            )
            if activation_packed.device != storage.device:
                raise ValueError(
                    "MXFP6 packed activation and weight storage must share a "
                    f"device, got {activation_packed.device} and {storage.device}"
                )
            output = torch.empty(
                (0, self.out_features),
                dtype=output_dtype,
                device=activation_packed.device,
            )
            return self._add_bias(output, output_dtype)

        self._ensure_packed_weight()
        if activation_packed.device != self.weight_packed.device:
            raise ValueError(
                "MXFP6 packed activation and weight must share a device, got "
                f"{activation_packed.device} and {self.weight_packed.device}"
            )
        output = torch.ops.xfuser.mxfp6_gemm_packed(
            activation_packed,
            self.weight_packed,
            activation_scale,
            self.weight_scale,
            rows,
            self.out_features,
            self.in_features,
        ).to(output_dtype)
        return self._add_bias(output, output_dtype)

    def _add_bias(
        self,
        output: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.bias is not None:
            if self.bias.device != output.device:
                raise ValueError(
                    f"MXFP6 bias is on {self.bias.device}, but output is on "
                    f"{output.device}"
                )
            if self.bias.dtype != output_dtype:
                raise TypeError(
                    f"MXFP6 bias dtype {self.bias.dtype} does not match "
                    f"output dtype {output_dtype}"
                )
            output = output + self.bias
        return output.contiguous()

    def _apply(self, fn, recurse=True):
        storage = self.weight_packed if hasattr(self, "weight_packed") else self.weight
        probe = torch.empty(
            0,
            dtype=self._compute_dtype,
            device=storage.device,
        )
        converted = fn(probe)
        if converted.dtype not in _SUPPORTED_COMPUTE_DTYPES:
            raise TypeError(
                "MXFP6 layers only support torch.bfloat16 or torch.float16, "
                f"got conversion target {converted.dtype}"
            )
        if hasattr(self, "weight_packed") and converted.dtype != self._compute_dtype:
            raise TypeError(
                "MXFP6 packed weights cannot be converted between compute "
                f"dtypes ({self._compute_dtype} to {converted.dtype}); "
                "reload and repack full-precision weights instead"
            )
        result = super()._apply(fn, recurse=recurse)
        if self.weight is not None:
            self._compute_dtype = self.weight.dtype
            if self.bias is not None and self.bias.dtype != self._compute_dtype:
                raise TypeError(
                    "MXFP6 weight and bias dtypes diverged after conversion"
                )
        return result

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim == 0:
            raise ValueError("MXFP6 linear input must have at least one dimension")
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"MXFP6 linear input has last dimension {input.shape[-1]}, "
                f"expected {self.in_features}"
            )
        original_shape = input.shape
        input_2d = input.reshape(-1, self.in_features)
        activation_packed, activation_scale = self.pack_activation(input_2d)
        output = self.forward_packed_2d(
            activation_packed,
            activation_scale,
            input_2d.shape[0],
            input.dtype,
        )
        return output.reshape(*original_shape[:-1], self.out_features).contiguous()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )
