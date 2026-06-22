from typing import Callable, List, Optional, Type, TypeVar, FrozenSet

from xfuser.core.distributed.attention_backend import AttentionBackendType, env_info

T = TypeVar("T", bound="AttentionSchedule")
G = TypeVar("G", bound="GemmPrecisionSchedule")

_GEMM_HIGH_SYNONYMS: FrozenSet[str] = frozenset(
    {"FP8", "TRUE", "1", "HIGH", "HP", "YES"}
)
_GEMM_LOW_SYNONYMS: FrozenSet[str] = frozenset(
    {"FP4", "MXFP4", "NVFP4", "FALSE", "0", "LOW", "LP", "NO"}
)


class AttentionSchedule:
    """
    Per-step attention schedule defined by an explicit list of backends.
    backends[i] is the backend used at step i; len(backends) equals total_steps.
    """

    def __init__(self, backends: List[AttentionBackendType]):
        if not backends:
            raise ValueError("AttentionSchedule requires at least one step.")
        self.backends = list(backends)
        self.total_steps = len(self.backends)

    @classmethod
    def from_comma_delimited_string(cls: Type[T], s: str) -> T:
        """
        Create an AttentionSchedule from a comma-delimited string of backend names.
        Each element is interpreted as an AttentionBackendType name (case-insensitive).
        Example: "FLASH_3,FLASH_3_FP8,FLASH_3_FP8,FLASH_3"
        """
        if not s or not s.strip():
            raise ValueError("Comma-delimited string must contain at least one backend name.")
        valid_names = [e.name for e in AttentionBackendType]
        backends: List[AttentionBackendType] = []
        for token in s.split(","):
            name = token.strip().upper()
            if not name:
                raise ValueError("Empty backend name in comma-delimited string.")
            try:
                backends.append(AttentionBackendType[name])
            except KeyError:
                raise ValueError(
                    f"Unknown attention backend '{token.strip()}'. "
                    f"Valid names: {', '.join(valid_names)}."
                ) from None
        return cls(backends)

    def get_backend(self, step: int) -> AttentionBackendType:
        if step < 0 or step >= len(self.backends):
            raise IndexError(f"Step {step} out of range [0, {len(self.backends)}).")
        return self.backends[step]



def create_hybrid_attn_schedule(
    num_high_precision_steps: int,
    low_precision_backend: AttentionBackendType,
    high_precision_backend: AttentionBackendType,
    total_steps: int,
    check_compat: Optional[Callable[[AttentionBackendType], None]] = None,
) -> AttentionSchedule:
    """
    Create a hybrid attention schedule: high-precision attention in the middle, low-precision attention at start/end.
    If check_compat is provided, it is called for both backends before returning (e.g. to validate
    compatibility with the current parallel config); it may raise.
    """
    if check_compat is not None:
        check_compat(low_precision_backend)
        check_compat(high_precision_backend)

    num_low_precision_steps = total_steps - 2 * num_high_precision_steps
    if num_low_precision_steps < 0:
        raise ValueError(
            f"total_steps ({total_steps}) must be >= 2 * num_high_precision_steps ({2 * num_high_precision_steps})."
        )
    backends = (
        [high_precision_backend] * num_high_precision_steps
        + [low_precision_backend] * num_low_precision_steps
        + [high_precision_backend] * num_high_precision_steps
    )
    return AttentionSchedule(backends)


class GemmPrecisionSchedule:
    """
    Per-step GEMM precision schedule defined by an explicit list of booleans.
    use_high_precision[i] indicates whether step i should use high precision GEMM.
    """

    def __init__(self, use_high_precision_schedule: List[bool]):
        if not use_high_precision_schedule:
            raise ValueError("GemmPrecisionSchedule requires at least one step.")
        self.use_high_precision_schedule = list(use_high_precision_schedule)

    @property
    def total_steps(self) -> int:
        return len(self.use_high_precision_schedule)

    @classmethod
    def from_comma_delimited_string(cls: Type[G], s: str) -> G:
        """
        Build a schedule from comma-separated tokens (case-insensitive).

        High-precision (FP8 GEMM path in hybrid layers): FP8, TRUE, 1, HIGH, HP, YES
        Low-precision (FP4 / MXFP4 path): FP4, MXFP4, NVFP4, FALSE, 0, LOW, LP, NO

        Example: ``fp4,fp4,fp8,fp8`` — asymmetric schedules are allowed; length must
        match ``num_inference_steps * cfg_step_multiplier`` at runtime (same as hybrid attention).
        """
        if not s or not s.strip():
            raise ValueError("Comma-delimited GEMM schedule must contain at least one token.")
        schedule: List[bool] = []
        for token in s.split(","):
            name = token.strip().upper()
            if not name:
                raise ValueError("Empty token in comma-delimited GEMM schedule string.")
            if name in _GEMM_HIGH_SYNONYMS:
                schedule.append(True)
            elif name in _GEMM_LOW_SYNONYMS:
                schedule.append(False)
            else:
                raise ValueError(
                    f"Unknown GEMM schedule token {token.strip()!r}. "
                    "Use FP8 or FP4 (synonyms: TRUE/FALSE, 1/0, HIGH/LOW, HP/LP, YES/NO)."
                ) from None
        return cls(schedule)

    def is_high_precision(self, step: int) -> bool:
        if step < 0 or step >= len(self.use_high_precision_schedule):
            raise IndexError(f"Step {step} out of range [0, {len(self.use_high_precision_schedule)}).")
        return self.use_high_precision_schedule[step]


def create_hybrid_gemm_schedule(
    num_high_precision_steps: int,
    total_steps: int,
) -> GemmPrecisionSchedule:
    """
    Create a hybrid GEMM schedule: high-precision GEMM at start/end, low-precision GEMM in the middle.
    """
    num_low_precision_steps = total_steps - 2 * num_high_precision_steps
    if num_low_precision_steps < 0:
        raise ValueError(
            f"total_steps ({total_steps}) must be >= 2 * num_high_precision_steps ({2 * num_high_precision_steps})."
        )
    schedule = (
        [True] * num_high_precision_steps
        + [False] * num_low_precision_steps
        + [True] * num_high_precision_steps
    )
    return GemmPrecisionSchedule(schedule)
