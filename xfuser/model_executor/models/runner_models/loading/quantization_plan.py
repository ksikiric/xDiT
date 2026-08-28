"""Backend-neutral quantization target planning for one model run."""

from typing import Optional

from xfuser.core.utils.runner_utils import log


def apply_fp8_override_cli_to_settings(config, settings) -> None:
    """Apply optional CLI FP8 override patterns (per-slot) to model settings."""

    def _parse_csv_patterns(raw: Optional[str]) -> Optional[tuple[str, ...]]:
        if raw is None or not raw.strip():
            return None
        patterns = tuple(p.strip() for p in raw.split(",") if p.strip())
        return patterns or None

    if config.fp8_precision_override_prefix_patterns is not None:
        settings.fp8_precision_overrides = _parse_csv_patterns(
            config.fp8_precision_override_prefix_patterns
        )
    if config.fp8_precision_override_suffix_patterns is not None:
        settings.fp8_precision_override_suffixes = _parse_csv_patterns(
            config.fp8_precision_override_suffix_patterns
        )


class QuantizationPlan:
    """Resolve declared FP8, FP4, FP6, and INT8 targets from one runner."""

    def __init__(self, model) -> None:
        self.model = model

    def module_list(self, format_name: str = "fp8") -> list[str]:
        settings = self.model.settings
        if format_name == "fp8":
            targets = list(settings.fp8_gemm_module_list or ())
            if self.model.config.use_fp8_text_encoder:
                targets += list(settings.fp8_text_encoder_module_list or ())
            return targets
        if format_name == "fp4":
            return list(settings.fp4_gemm_module_list or ())
        if format_name == "fp6":
            # Pure FP6 is intentionally defined by the existing runner
            # declarations: every target that would have been FP4 or FP8 becomes
            # MXFP6. Keep an explicit field available for future model-local
            # targets without requiring it for the historical union.
            return list(
                dict.fromkeys(
                    list(getattr(settings, "fp4_gemm_module_list", None) or ())
                    + list(getattr(settings, "fp8_gemm_module_list", None) or ())
                    + list(getattr(settings, "fp6_gemm_module_list", None) or ())
                )
            )
        if format_name == "int8":
            return list(settings.int8_gemm_module_list or ())
        raise ValueError(f"unsupported quantization target format: {format_name}")

    def targets_for(self, component_name: str, format_name: str = "fp8") -> list[str]:
        prefix = f"{component_name}."
        return [
            "" if target == component_name else target[len(prefix) :]
            for target in self.module_list(format_name)
            if target == component_name or target.startswith(prefix)
        ]

    def log_fp8_overrides(self) -> None:
        """Log FP8 precision overrides once when the FP4 plan is materialized."""
        settings = self.model.settings
        prefixes = settings.fp8_precision_overrides
        suffixes = settings.fp8_precision_override_suffixes
        if getattr(self.model.config, "use_fp6_only", False):
            return
        override_format = (
            "MXFP6" if getattr(self.model.config, "use_fp6_gemms", False) else "FP8"
        )
        if prefixes:
            log(
                f"The following layers will be quantized to {override_format}, "
                "to maintain output quality: "
                f"{prefixes} (prefix match)"
            )
        if suffixes:
            log(
                f"The following layers will be quantized to {override_format}, "
                "to maintain output quality: "
                f"{suffixes} (suffix match)"
            )
