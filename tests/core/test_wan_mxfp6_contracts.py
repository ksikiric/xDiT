"""Dependency-light contracts for the audited Wan MXFP6 runners."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
WAN_PATH = ROOT / "xfuser/model_executor/models/runner_models/wan.py"
CONTRACTS_PATH = (
    ROOT / "xfuser/model_executor/models/runner_models/loading/contracts.py"
)
WAN_CLASSES = {
    node.name: node
    for node in ast.parse(WAN_PATH.read_text()).body
    if isinstance(node, ast.ClassDef)
}

AUDITED_WAN_RUNNERS = (
    "xFuserWan21I2VModel",
    "xFuserWan22I2VModel",
    "xFuserWan22DistilledI2VModel",
    "xFuserWan21T2VModel",
    "xFuserWan22T2VModel",
    "xFuserWan22TI2VModel",
)


def _assignment_values(class_name, attribute):
    values = []
    for statement in WAN_CLASSES[class_name].body:
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target] if isinstance(statement, ast.AnnAssign) else []
        )
        if any(
            isinstance(target, ast.Name) and target.id == attribute
            for target in targets
        ):
            values.append(statement.value)
    return values


def _effective_assignment(class_name, attribute):
    values = _assignment_values(class_name, attribute)
    if values:
        return values[-1]
    for base in WAN_CLASSES[class_name].bases:
        if isinstance(base, ast.Name) and base.id in WAN_CLASSES:
            try:
                return _effective_assignment(base.id, attribute)
            except LookupError:
                pass
    raise LookupError(f"{class_name} has no effective {attribute}")


def _keyword(call, name, default=None):
    value = next((item.value for item in call.keywords if item.arg == name), None)
    return default if value is None else ast.literal_eval(value)


def _capabilities(class_name):
    call = _effective_assignment(class_name, "capabilities")
    return SimpleNamespace(
        use_fp8_gemms=_keyword(call, "use_fp8_gemms", False),
        use_fp4_gemms=_keyword(call, "use_fp4_gemms", False),
        use_fp6_gemms=_keyword(call, "use_fp6_gemms", False),
        use_fp6_only=_keyword(call, "use_fp6_only", False),
        use_int8_gemms=_keyword(call, "use_int8_gemms", False),
        fully_shard_degree=_keyword(call, "fully_shard_degree", False),
    )


def _load_contracts():
    spec = importlib.util.spec_from_file_location(
        "wan_mxfp6_contracts_under_test", CONTRACTS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def contracts():
    return _load_contracts()


@pytest.mark.parametrize("runner_name", AUDITED_WAN_RUNNERS)
def test_audited_wan_runners_accept_both_fp6_contracts(contracts, runner_name):
    declaration = contracts.LoadDeclaration.for_runner(_capabilities(runner_name))

    for format_name in ("FP4_FP6", "FP6"):
        contract = contracts.select_load_contract(
            requested_format=getattr(contracts.QuantizationFormat, format_name),
            selected_backend=contracts.QuantizationBackend.AITER,
            materialization_mode=contracts.MaterializationMode.EAGER,
            declaration=declaration,
            fsdp_strategy={},
            runner_name=runner_name,
        )
        assert contract.requested_format.name == format_name
        assert contract.selected_backend is contracts.QuantizationBackend.AITER


@pytest.mark.parametrize("format_name", ["FP4_FP6", "FP6"])
def test_unopted_wan_vace_rejects_fp6_contracts(contracts, format_name):
    runner_name = "xFuserWan21VACEModel"
    declaration = contracts.LoadDeclaration.for_runner(_capabilities(runner_name))

    with pytest.raises(
        contracts.UnsupportedLoadContract,
        match=rf"{format_name}.*{runner_name}",
    ):
        contracts.select_load_contract(
            requested_format=getattr(contracts.QuantizationFormat, format_name),
            selected_backend=contracts.QuantizationBackend.AITER,
            materialization_mode=contracts.MaterializationMode.EAGER,
            declaration=declaration,
            fsdp_strategy={},
            runner_name=runner_name,
        )


def test_wan21_t2v_effective_capabilities_are_the_final_declaration():
    declarations = _assignment_values("xFuserWan21T2VModel", "capabilities")

    assert len(declarations) == 2
    assert _keyword(declarations[0], "use_fp6_gemms") is None
    assert _keyword(declarations[0], "use_fp6_only") is None
    assert _keyword(declarations[-1], "use_fp6_gemms") is True
    assert _keyword(declarations[-1], "use_fp6_only") is True


@pytest.mark.parametrize(
    ("runner_name", "override_blocks", "override_suffixes"),
    [
        (
            "xFuserWan21I2VModel",
            tuple(map(str, range(10))) + tuple(map(str, range(30, 40))),
            None,
        ),
        (
            "xFuserWan21T2VModel",
            tuple(map(str, range(10))) + tuple(map(str, range(30, 40))),
            None,
        ),
        (
            "xFuserWan22TI2VModel",
            ("0", "1", "28", "29"),
            (".net.0.proj", ".net.2"),
        ),
    ],
)
def test_wan_fp6_opt_in_preserves_override_leaves(
    runner_name, override_blocks, override_suffixes
):
    settings = _effective_assignment(runner_name, "settings")

    assert _keyword(settings, "fp4_gemm_module_list") == ["transformer.blocks"]
    assert _keyword(settings, "fp8_gemm_module_list") == ["transformer.blocks"]
    assert _keyword(settings, "fp6_gemm_module_list") is None
    assert (
        tuple(
            item.rstrip(".") for item in _keyword(settings, "fp8_precision_overrides")
        )
        == override_blocks
    )
    assert _keyword(settings, "fp8_precision_override_suffixes") == override_suffixes


@pytest.mark.parametrize("runner_name", ["xFuserWan22I2VModel", "xFuserWan22T2VModel"])
def test_wan22_keeps_transformer_2_as_the_fp6_only_mixed_remainder(runner_name):
    customize = next(
        node
        for node in WAN_CLASSES[runner_name].body
        if isinstance(node, ast.FunctionDef) and node.name == "_customize_settings"
    )
    updates = {}
    for statement in ast.walk(customize):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Attribute)
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "self"
            and target.value.attr == "settings"
        ):
            try:
                updates[target.attr] = ast.literal_eval(statement.value)
            except (ValueError, TypeError):
                pass

    assert updates["fp8_gemm_module_list"] == [
        "transformer.blocks",
        "transformer_2.blocks",
    ]
    assert updates["fp8_precision_overrides"] is None
    assert "fp4_gemm_module_list" not in updates
    assert "fp6_gemm_module_list" not in updates
