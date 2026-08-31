"""Dependency-light MXFP6 CLI, load-contract, and backend capability tests."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOADING = ROOT / "xfuser/model_executor/models/runner_models/loading"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    contracts = _load_module(LOADING / "contracts.py", "mxfp6_contracts_under_test")
    backends = _load_module(LOADING / "format_backends.py", "mxfp6_backends_under_test")
    return SimpleNamespace(contracts=contracts, backends=backends)


def _config(**overrides):
    values = {
        "use_fp8_gemms": False,
        "use_fp4_gemms": False,
        "use_fp6_gemms": False,
        "use_int8_gemms": False,
        "use_hybrid_gemm_schedule": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("flags", "expected_format"),
    [
        ({"use_fp6_gemms": True}, "FP6"),
        ({"use_fp4_gemms": True, "use_fp6_gemms": True}, "FP4_FP6"),
    ],
)
def test_runtime_selects_explicit_aiter_only_fp6_contracts(
    modules, flags, expected_format
):
    requested, backend = modules.contracts.select_runtime_quantization(
        _config(**flags),
        aiter_fp8_active=False,
        cuda_active=False,
    )

    assert requested.name == expected_format
    assert backend.name == "AITER"


@pytest.mark.parametrize(
    "offload_flag",
    [
        "enable_model_cpu_offload",
        "enable_sequential_cpu_offload",
        "enable_group_cpu_offload",
    ],
)
def test_mixed_fp4_fp6_rejects_every_cpu_offload_mode(modules, offload_flag):
    contracts = modules.contracts
    config = _config(
        use_fp4_gemms=True,
        use_fp6_gemms=True,
        **{offload_flag: True},
    )

    with pytest.raises(contracts.UnsupportedLoadContract) as refusal:
        contracts.assert_offload_is_compatible_with_format(
            config,
            requested_format=contracts.QuantizationFormat.FP4_FP6,
            selected_backend=contracts.QuantizationBackend.AITER,
        )

    message = str(refusal.value)
    assert f"--{offload_flag}" in message
    assert "primary MXFP4 packing path is unsupported" in message


@pytest.mark.parametrize(
    "offload_flag",
    [
        "enable_model_cpu_offload",
        "enable_sequential_cpu_offload",
        "enable_group_cpu_offload",
    ],
)
def test_pure_fp6_does_not_reject_supported_cpu_offload_modes(modules, offload_flag):
    contracts = modules.contracts
    config = _config(use_fp6_gemms=True, **{offload_flag: True})

    contracts.assert_offload_is_compatible_with_format(
        config,
        requested_format=contracts.QuantizationFormat.FP6,
        selected_backend=contracts.QuantizationBackend.AITER,
    )


@pytest.mark.parametrize(
    ("flags", "cuda", "reason"),
    [
        ({"use_fp6_gemms": True, "use_fp8_gemms": True}, False, "FP8"),
        ({"use_fp6_gemms": True, "use_int8_gemms": True}, False, "INT8"),
        (
            {"use_fp6_gemms": True, "use_hybrid_gemm_schedule": True},
            False,
            "hybrid",
        ),
        ({"use_fp6_gemms": True}, True, "CUDA"),
    ],
)
def test_runtime_rejects_ambiguous_or_unsafe_fp6_combinations(
    modules, flags, cuda, reason
):
    with pytest.raises(modules.contracts.UnsupportedLoadContract, match=reason):
        modules.contracts.select_runtime_quantization(
            _config(**flags),
            aiter_fp8_active=False,
            cuda_active=cuda,
        )


def test_runner_declaration_keeps_fp6_pairs_aiter_only(modules):
    c = modules.contracts
    capabilities = SimpleNamespace(
        use_fp8_gemms=False,
        use_fp4_gemms=True,
        use_fp6_gemms=True,
        use_int8_gemms=False,
    )

    declaration = c.LoadDeclaration.for_runner(capabilities)

    assert (c.QuantizationFormat.FP6, c.QuantizationBackend.AITER) in (
        declaration.quantization_contracts
    )
    assert (c.QuantizationFormat.FP4_FP6, c.QuantizationBackend.AITER) in (
        declaration.quantization_contracts
    )
    assert (c.QuantizationFormat.FP6, c.QuantizationBackend.TORCHAO) not in (
        declaration.quantization_contracts
    )
    assert (
        c.QuantizationFormat.FP4_FP6,
        c.QuantizationBackend.TORCHAO,
    ) not in declaration.quantization_contracts


def test_runner_without_fp4_capability_declares_only_pure_fp6(modules):
    c = modules.contracts
    declaration = c.LoadDeclaration.for_runner(
        SimpleNamespace(
            use_fp8_gemms=False,
            use_fp4_gemms=False,
            use_fp6_gemms=True,
            use_int8_gemms=False,
        )
    )

    assert (c.QuantizationFormat.FP6, c.QuantizationBackend.AITER) in (
        declaration.quantization_contracts
    )
    assert (c.QuantizationFormat.FP4_FP6, c.QuantizationBackend.AITER) not in (
        declaration.quantization_contracts
    )


def test_core_declares_fp6_capabilities_and_limits_opt_in_to_audited_wan():
    base_path = ROOT / "xfuser/model_executor/models/runner_models/base_model.py"
    tree = ast.parse(base_path.read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    capability_fields = {
        node.target.id
        for node in classes["ModelCapabilities"].body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    setting_fields = {
        node.target.id
        for node in classes["ModelSettings"].body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "use_fp6_gemms" in capability_fields
    assert "use_fp6_only" not in capability_fields
    assert "fp6_gemm_module_list" in setting_fields

    opted_in = set()
    runners = ROOT / "xfuser/model_executor/models/runner_models"
    for path in runners.glob("*.py"):
        classes = (
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef)
        )
        for class_node in classes:
            for call in (
                node
                for node in ast.walk(class_node)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ModelCapabilities"
            ):
                for keyword in call.keywords:
                    if (
                        keyword.arg == "use_fp6_gemms"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                    ):
                        opted_in.add(f"{path.name}:{class_node.name}:{keyword.arg}")
    assert opted_in == {
        f"wan.py:{class_name}:use_fp6_gemms"
        for class_name in (
            "xFuserWan21I2VModel",
            "xFuserWan22DistilledI2VModel",
            "xFuserWan21T2VModel",
            "xFuserWan22TI2VModel",
        )
    }


@pytest.mark.parametrize(
    "missing",
    [
        "quant_mxfp6_gemm",
        "gemm_a6w6",
        "mxfp6_gemm_pack_size",
        "quant_mxfp6_gemm_out",
    ],
)
def test_mxfp6_probe_requires_exact_a6w6_surface(modules, monkeypatch, missing):
    b = modules.backends
    api = SimpleNamespace(
        quant_mxfp6_gemm=lambda value: value,
        gemm_a6w6=lambda *args: object(),
        mxfp6_gemm_pack_size=lambda rows, features: (rows, features),
        quant_mxfp6_gemm_out=lambda *args: object(),
    )
    setattr(api, missing, None)
    monkeypatch.delenv("AITER_TRITON_ONLY", raising=False)
    monkeypatch.setattr(b, "import_module", lambda name: api)

    available, reason = b._probe_aiter_mxfp6_apis(
        gcn_arch_probe=lambda: "gfx950:sramecc+:xnack-"
    )

    assert not available
    assert f"aiter.ops.gemm_op_a6w6.{missing}" in reason


def test_mxfp6_probe_preserves_import_reason_and_rejects_triton_only(
    modules, monkeypatch
):
    b = modules.backends
    monkeypatch.delenv("AITER_TRITON_ONLY", raising=False)

    def broken_import(name):
        raise RuntimeError("broken A6W6 extension")

    monkeypatch.setattr(b, "import_module", broken_import)
    available, reason = b._probe_aiter_mxfp6_apis(gcn_arch_probe=lambda: "gfx950")
    assert not available
    assert reason == (
        "AITER MXFP6 import probe failed: RuntimeError: broken A6W6 extension"
    )

    monkeypatch.setenv("AITER_TRITON_ONLY", "1")
    available, reason = b._probe_aiter_mxfp6_apis(gcn_arch_probe=lambda: "gfx950")
    assert not available
    assert reason == (
        "AITER MXFP6 requires the ASM backend, but " "AITER_TRITON_ONLY=1 disables ASM"
    )


@pytest.mark.parametrize(
    ("arch", "available"),
    [("gfx950", True), ("gfx950:sramecc+:xnack-", True), ("gfx942", False)],
)
def test_mxfp6_probe_requires_gfx950(modules, monkeypatch, arch, available):
    b = modules.backends
    api = SimpleNamespace(
        quant_mxfp6_gemm=lambda value: value,
        gemm_a6w6=lambda *args: object(),
        mxfp6_gemm_pack_size=lambda rows, features: (rows, features),
        quant_mxfp6_gemm_out=lambda *args: object(),
    )
    monkeypatch.delenv("AITER_TRITON_ONLY", raising=False)
    monkeypatch.setattr(b, "import_module", lambda name: api)

    result, reason = b._probe_aiter_mxfp6_apis(gcn_arch_probe=lambda: arch)

    assert result is available
    assert (reason is None) is available
    if not available:
        assert arch in reason


def test_mxfp6_capability_preserves_probe_reason(modules):
    b = modules.backends
    capabilities = b.probe_format_backend_capabilities(
        cuda_probe=lambda: False,
        hip_probe=lambda: True,
        cuda_capability_probe=lambda: None,
        mxfp4_probe=lambda: (False, "MXFP4 unavailable"),
        mxfp6_probe=lambda: (
            False,
            "AITER MXFP6 import probe failed: ImportError: missing extension",
        ),
        nvfp4_probe=lambda: pytest.fail("must not probe NVFP4 on ROCm"),
        int8_probe=lambda: pytest.fail("must not probe INT8 on ROCm"),
        diffusers_probe=lambda kind: pytest.fail("must not probe Diffusers"),
        fsdp_probe=lambda kind: pytest.fail(
            "must not probe FSDP for an unavailable backend"
        ),
    )

    assert not capabilities.aiter_mxfp6
    assert capabilities.aiter_mxfp6_reason == (
        "AITER MXFP6 import probe failed: ImportError: missing extension"
    )


def _contract(modules, format_name, backend_name="AITER"):
    c = modules.contracts
    return c.LoadContract(
        requested_format=getattr(c.QuantizationFormat, format_name),
        selected_backend=getattr(c.QuantizationBackend, backend_name),
        materialization_mode=c.MaterializationMode.EAGER,
    )


def test_pure_and_mixed_adapters_select_without_native_streaming(modules):
    b = modules.backends
    capabilities = b.FormatBackendCapabilities(
        aiter_mxfp4=True,
        aiter_mxfp6=True,
    )

    pure = b.select_format_backend(_contract(modules, "FP6"), capabilities=capabilities)
    mixed = b.select_format_backend(
        _contract(modules, "FP4_FP6"), capabilities=capabilities
    )
    remainder = b.select_mxfp6_backend(
        _contract(modules, "FP4_FP6"), capabilities=capabilities
    )

    assert isinstance(pure, b.AiterMxfp6BackendAdapter)
    assert pure.uses_native_transformer_streaming is False
    assert isinstance(mixed, b.AiterMxfp4BackendAdapter)
    assert mixed.use_fp6_for_overrides is True
    assert isinstance(remainder, b.AiterMxfp6BackendAdapter)
    assert remainder.format is modules.contracts.QuantizationFormat.FP6


def test_mxfp6_fsdp_placement_reuses_non_float_parameter_capability(modules):
    b = modules.backends
    adapter = b.AiterMxfp6BackendAdapter(
        backend=modules.contracts.QuantizationBackend.AITER,
        format_=modules.contracts.QuantizationFormat.FP6,
    )
    capabilities = b.FormatBackendCapabilities(
        aiter_mxfp6=True,
        aiter_mxfp6_fsdp=False,
        aiter_mxfp6_fsdp_reason="non-float parameters unsupported",
    )

    with pytest.raises(
        modules.contracts.UnsupportedLoadContract,
        match="AITER MXFP6 packed weight.*non-float",
    ):
        b.validate_format_fsdp_placement(
            _contract(modules, "FP6"),
            adapter,
            capabilities=capabilities,
            required=True,
        )


def test_cli_fp6_flags_validate_before_model_initialization():
    pytest.importorskip("torch")
    from xfuser.config import FlexibleArgumentParser
    from xfuser.config.args import xFuserArgs

    parser = xFuserArgs.add_runner_args(
        FlexibleArgumentParser(description="MXFP6 CLI test")
    )
    parsed = parser.parse_args(["--model", "test/model", "--use_fp6_gemms"])
    assert parsed.use_fp4_gemms is False
    assert parsed.use_fp6_gemms is True
    assert not hasattr(parsed, "use_fp6_only")

    xFuserArgs(use_fp6_gemms=True)._validate_gemm_quantization_flags()
    xFuserArgs(
        use_fp4_gemms=True,
        use_fp6_gemms=True,
    )._validate_gemm_quantization_flags()
    with pytest.raises(ValueError, match="cannot be combined"):
        xFuserArgs(
            use_fp6_gemms=True,
            use_fp8_gemms=True,
        )._validate_gemm_quantization_flags()
