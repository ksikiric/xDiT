"""Eager and blockwise ownership tests for pure and mixed MXFP6 modes."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from xfuser.model_executor.models.runner_models.loading import placement, shard
from xfuser.model_executor.models.runner_models.loading.backend_selection import (
    QuantizationBackends,
)
from xfuser.model_executor.models.runner_models.loading.blockwise_ownership import (
    record_blockwise_ownership,
)
from xfuser.model_executor.models.runner_models.loading.contracts import (
    MaterializationMode,
    QuantizationBackend,
    QuantizationFormat,
)
from xfuser.model_executor.models.runner_models.loading.quantization_ledger import (
    QuantizationLedger,
)
from xfuser.model_executor.models.runner_models.loading.quantization_plan import (
    QuantizationPlan,
)


class RecordingAdapter:
    def __init__(self):
        self.module_calls = []
        self.block_calls = []

    def convert_module(self, module, **kwargs):
        self.module_calls.append((module, kwargs))

    def convert_block(self, block, **kwargs):
        self.block_calls.append((block, kwargs))


def _settings():
    return SimpleNamespace(
        fp4_gemm_module_list=["transformer.blocks"],
        fp8_gemm_module_list=[
            "transformer.blocks",
            "transformer_2.blocks",
        ],
        fp6_gemm_module_list=None,
        fp8_text_encoder_module_list=None,
        fp8_precision_overrides=("0.attn",),
        fp8_precision_override_suffixes=(".ff.proj",),
        fp8_gemm_include_suffixes=None,
        int8_gemm_module_list=None,
        fsdp_strategy={},
    )


def _config(*, mixed=False, pure=False):
    return SimpleNamespace(
        use_fp4_gemms=not pure,
        use_fp8_gemms=False,
        use_fp6_gemms=mixed or pure,
        use_int8_gemms=False,
        use_fp8_text_encoder=False,
        use_hybrid_gemm_schedule=False,
        enable_model_cpu_offload=False,
        enable_sequential_cpu_offload=False,
        enable_group_cpu_offload=False,
        fully_shard_degree=1,
    )


def _targets_for(plan, component_name, format_name):
    return tuple(plan.targets_for(component_name, format_name))


def _loader(*, mixed=False, pure=False):
    primary_module = SimpleNamespace()
    secondary_module = SimpleNamespace()
    pipe = SimpleNamespace(
        transformer=SimpleNamespace(blocks=primary_module),
        transformer_2=SimpleNamespace(blocks=secondary_module),
    )
    model = SimpleNamespace(
        config=_config(mixed=mixed, pure=pure),
        settings=_settings(),
        pipe=pipe,
    )
    plan = QuantizationPlan(model)
    primary = RecordingAdapter()
    fp6 = RecordingAdapter()
    fp8 = RecordingAdapter()
    format_name = QuantizationFormat.FP6 if pure else QuantizationFormat.FP4_FP6
    contract = SimpleNamespace(
        requested_format=format_name,
        selected_backend=QuantizationBackend.AITER,
        materialization_mode=MaterializationMode.EAGER,
    )
    backends = SimpleNamespace(
        format=primary,
        fp6=(primary if pure else fp6),
        blockwise_fp8=fp8,
        format_entries=lambda: tuple(plan.module_list("fp6" if pure else "fp4")),
        format_targets_for=lambda component: _targets_for(
            plan, component, "fp6" if pure else "fp4"
        ),
    )
    return (
        SimpleNamespace(
            model=model,
            load_contract=contract,
            quantization_plan=plan,
            quantization_ledger=QuantizationLedger(
                descriptor_components={"transformer", "transformer_2"}
            ),
            backends=backends,
        ),
        primary,
        fp6,
        fp8,
    )


def test_pure_fp6_targets_are_the_stable_fp4_fp8_union():
    loader, _, _, _ = _loader(pure=True)

    assert loader.quantization_plan.module_list("fp6") == [
        "transformer.blocks",
        "transformer_2.blocks",
    ]


@pytest.mark.parametrize("pure", [False, True])
def test_backend_selection_assigns_primary_and_remainder_owners_without_fp8(
    pure,
):
    loader, primary, fp6, fp8 = _loader(mixed=not pure, pure=pure)
    selected = QuantizationBackends(loader)
    selected.__dict__["format"] = primary
    selected.__dict__["fp6"] = primary if pure else fp6
    selected.__dict__["blockwise_fp8"] = fp8
    loader.backends = selected

    assert selected.transformer_adapter("transformer") == (
        primary,
        ("blocks",),
    )
    assert selected.transformer_adapter("transformer_2") == (
        primary if pure else fp6,
        ("blocks",),
    )
    assert selected.uses_blockwise_fp8() is False
    assert (
        selected.places_torchao_tensor_subclass_under_fsdp2(
            SimpleNamespace(backend=QuantizationBackend.TORCHAO)
        )
        is False
    )


def test_eager_mixed_mode_routes_overrides_and_fp8_only_target_to_mxfp6(
    monkeypatch,
):
    loader, primary, fp6, fp8 = _loader(mixed=True)
    monkeypatch.setattr(placement, "log", lambda message: None)

    placement.setup_mxfp4_gemms(loader, local_rank=2)

    assert len(primary.module_calls) == 1
    primary_module, kwargs = primary.module_calls[0]
    assert primary_module is loader.model.pipe.transformer.blocks
    assert kwargs["fp8_layers"] == ("0.attn",)
    assert kwargs["fp8_suffix_layers"] == (".ff.proj",)
    assert kwargs["device"] == "cuda:2"
    assert fp6.module_calls == [
        (
            loader.model.pipe.transformer_2.blocks,
            {"device": "cuda:2"},
        )
    ]
    assert fp8.module_calls == []


def test_eager_mixed_offload_evicts_primary_overrides_and_fp6_remainder(
    monkeypatch,
):
    loader, primary, fp6, fp8 = _loader(mixed=True)
    loader.model.config.enable_model_cpu_offload = True
    monkeypatch.setattr(placement, "log", lambda message: None)

    placement.setup_mxfp4_gemms(loader, local_rank=2)

    assert primary.module_calls[0][1]["offload_to_cpu"] is True
    assert fp6.module_calls[0][1]["offload_to_cpu"] is True
    assert fp8.module_calls == []


def test_eager_pure_mode_routes_each_union_target_once(monkeypatch):
    loader, primary, fp6, fp8 = _loader(pure=True)
    monkeypatch.setattr(placement, "log", lambda message: None)

    placement.setup_mxfp6_gemms(loader, local_rank=1)

    assert [call[0] for call in primary.module_calls] == [
        loader.model.pipe.transformer.blocks,
        loader.model.pipe.transformer_2.blocks,
    ]
    assert all(call[1]["fp8_layers"] is None for call in primary.module_calls)
    assert all(call[1]["fp8_suffix_layers"] is None for call in primary.module_calls)
    assert all(call[1]["device"] == "cuda:1" for call in primary.module_calls)
    assert fp6.module_calls == []
    assert fp8.module_calls == []


def test_blockwise_mixed_mode_uses_fp6_for_fp8_only_component():
    loader, primary, fp6, fp8 = _loader(mixed=True)

    quantize = shard.build_block_quantize_fn(
        loader, "transformer_2", ["blocks"], local_rank=3
    )
    block = object()
    quantize(block, 0)

    assert primary.block_calls == []
    assert len(fp6.block_calls) == 1
    called_block, kwargs = fp6.block_calls[0]
    assert called_block is block
    assert kwargs["device"] == "cuda:3"
    assert kwargs["filter_fn"](object(), "any.linear")
    assert fp8.block_calls == []


def test_blockwise_pure_mode_uses_primary_fp6_adapter_for_both_components():
    loader, primary, fp6, fp8 = _loader(pure=True)

    for component_name in ("transformer", "transformer_2"):
        quantize = shard.build_block_quantize_fn(
            loader, component_name, ["blocks"], local_rank=0
        )
        quantize(object(), 0)

    assert len(primary.block_calls) == 2
    assert fp6.block_calls == []
    assert fp8.block_calls == []


def test_blockwise_mixed_ledger_records_fp6_remainder_as_format_owned(
    monkeypatch,
):
    monkeypatch.setitem(
        record_blockwise_ownership.__globals__, "log", lambda message: None
    )
    ledger = QuantizationLedger()
    adapter = SimpleNamespace(format=QuantizationFormat.FP4_FP6)
    descriptor = SimpleNamespace(
        materialization_mode="blockwise",
        log_message=lambda: "mixed blockwise",
    )

    record_blockwise_ownership(
        ledger,
        adapter,
        "transformer",
        ("blocks.0.attn",),
        ("blocks",),
        descriptor,
        fp4_gemms=True,
        mxfp6_targets=("blocks",),
    )

    assert ledger.streaming_targets == {
        "transformer.blocks",
        "transformer.blocks.0.attn",
    }
    assert ledger.fp8_streaming_targets == set()


def test_no_fp6_flags_preserve_fp8_remainder_routing():
    loader, primary, fp6, fp8 = _loader()
    loader.model.config.use_fp6_gemms = False
    loader.load_contract.requested_format = QuantizationFormat.FP4

    quantize = shard.build_block_quantize_fn(
        loader, "transformer_2", ["blocks"], local_rank=4
    )
    quantize(object(), 0)

    assert primary.block_calls == []
    assert fp6.block_calls == []
    assert len(fp8.block_calls) == 1
