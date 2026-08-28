"""MXFP6 layer lifecycle tests with the local AITER ABI mocked on CPU."""

from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def runtime():
    torch = pytest.importorskip("torch", reason="PyTorch is required for MXFP6 tests")
    from xfuser.model_executor.layers import mxfp6_linear

    return SimpleNamespace(torch=torch, module=mxfp6_linear)


@pytest.fixture
def fake_aiter(runtime, monkeypatch):
    torch = runtime.torch
    module = runtime.module
    calls = {"packs": [], "gemms": [], "pack_out": []}

    def pack_size(rows, in_features):
        return tuple(int(value) for value in module._abi_pack_sizes(rows, in_features))

    def pack(tensor):
        calls["packs"].append(tensor)
        packed_size, scale_size = pack_size(*tensor.shape)
        return (
            torch.zeros(packed_size, dtype=torch.uint8, device=tensor.device),
            torch.ones(scale_size, dtype=torch.uint8, device=tensor.device),
        )

    def pack_out(tensor, packed, scale):
        calls["pack_out"].append((tensor, packed, scale))
        packed.zero_()
        scale.fill_(1)
        return packed, scale

    def gemm(a_packed, w_packed, a_scale, w_scale, rows, out_features, in_features):
        calls["gemms"].append(
            (
                a_packed,
                w_packed,
                a_scale,
                w_scale,
                rows,
                out_features,
                in_features,
            )
        )
        # Deliberately return a non-contiguous view: the xFuser ABI promises a
        # contiguous logical output even when AITER slices a padded matrix.
        storage = torch.zeros(
            (rows, out_features * 2),
            dtype=torch.bfloat16,
            device=a_packed.device,
        )
        return storage[:, ::2]

    monkeypatch.setattr(module, "mxfp6_gemm_pack_size", pack_size)
    monkeypatch.setattr(module, "quant_mxfp6_gemm", pack)
    monkeypatch.setattr(module, "quant_mxfp6_gemm_out", pack_out)
    monkeypatch.setattr(module, "gemm_a6w6", gemm)
    return calls


def _layer(runtime, *, bias=False, device=None, dtype=None):
    return runtime.module.xFuserMXFP6Linear(
        8,
        4,
        bias=bias,
        device=device,
        dtype=dtype or runtime.torch.bfloat16,
    )


def test_custom_ops_and_packed_registration(runtime, fake_aiter):
    torch = runtime.torch
    layer = _layer(runtime)

    layer._quantize_weights()

    assert callable(torch.ops.xfuser.mxfp6_gemm)
    assert callable(torch.ops.xfuser.mxfp6_gemm_packed)
    assert callable(torch.ops.xfuser.mxfp6_pack)
    assert layer.logical_in_features == 8
    assert layer.logical_out_features == 4
    assert layer.weight is None
    assert not layer.weight_packed.requires_grad
    assert dict(layer.named_parameters())["weight_packed"] is layer.weight_packed
    assert dict(layer.named_buffers())["weight_scale"] is layer.weight_scale
    assert (
        dict(layer.named_buffers())["weight_packed_provenance"]
        is layer.weight_packed_provenance
    )
    assert torch.equal(
        layer.weight_packed_provenance,
        torch.tensor([runtime.module._PACK_LAYOUT_ABI_VERSION, 4, 8, 1]),
    )
    assert set(layer.state_dict()) == {
        "weight_packed",
        "weight_scale",
        "weight_packed_provenance",
    }


def test_custom_op_fake_registrations_expose_logical_shapes(runtime):
    torch = runtime.torch
    fake_tensor = pytest.importorskip("torch._subclasses.fake_tensor")
    module = runtime.module

    with fake_tensor.FakeTensorMode():
        activation = torch.empty((3, 8), dtype=torch.bfloat16)
        activation_packed, activation_scale = torch.ops.xfuser.mxfp6_pack(activation)
        weight_packed_size, weight_scale_size = module._abi_pack_sizes(4, 8)
        weight_packed = torch.empty(weight_packed_size, dtype=torch.uint8)
        weight_scale = torch.empty(weight_scale_size, dtype=torch.uint8)
        output = torch.ops.xfuser.mxfp6_gemm_packed(
            activation_packed,
            weight_packed,
            activation_scale,
            weight_scale,
            3,
            4,
            8,
        )

    assert activation_packed.shape == (module._abi_pack_sizes(3, 8)[0],)
    assert activation_scale.shape == (module._abi_pack_sizes(3, 8)[1],)
    assert output.shape == (3, 4)
    assert output.dtype is torch.bfloat16


def test_compiled_forward_accepts_dynamic_two_dimensional_pack_rows(
    runtime, fake_aiter
):
    torch = runtime.torch
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    layer = _layer(runtime)
    layer._quantize_weights()
    compiled = torch.compile(
        layer,
        backend="eager",
        dynamic=True,
        fullgraph=True,
    )

    first = compiled(torch.ones((2, 8), dtype=torch.bfloat16))
    second = compiled(torch.ones((5, 8), dtype=torch.bfloat16))

    assert first.shape == (2, 4)
    assert second.shape == (5, 4)
    assert first.is_contiguous() and second.is_contiguous()


def test_forward_preserves_leading_dims_bias_and_exact_operand_order(
    runtime, fake_aiter
):
    torch = runtime.torch
    layer = _layer(runtime, bias=True)
    with torch.no_grad():
        layer.bias.fill_(0.5)
    layer._quantize_weights()

    output = layer(torch.ones((2, 3, 8), dtype=torch.bfloat16))

    assert output.shape == (2, 3, 4)
    assert output.is_contiguous()
    assert torch.equal(output, torch.full_like(output, 0.5))
    assert fake_aiter["packs"][-1].shape == (6, 8)
    (
        activation_packed,
        weight_packed,
        activation_scale,
        weight_scale,
        rows,
        out_features,
        in_features,
    ) = fake_aiter["gemms"][-1]
    assert activation_packed.dtype is torch.uint8
    assert weight_packed is layer.weight_packed
    assert activation_scale.dtype is torch.uint8
    assert weight_scale is layer.weight_scale
    assert (rows, out_features, in_features) == (6, 4, 8)


def test_reusable_activation_pack_out_and_packed_forward(runtime, fake_aiter):
    torch = runtime.torch
    layer = _layer(runtime)
    layer._quantize_weights()
    activation = torch.ones((5, 8), dtype=torch.bfloat16)
    packed_size, scale_size = layer.activation_pack_size(5)
    packed = torch.empty(packed_size, dtype=torch.uint8)
    scale = torch.empty(scale_size, dtype=torch.uint8)

    returned = layer.pack_activation_out(activation, packed, scale)
    output = layer.forward_packed_2d(packed, scale, rows=5, output_dtype=torch.bfloat16)

    assert returned == (packed, scale)
    packed_call = fake_aiter["pack_out"][-1]
    assert packed_call[0] is activation
    assert packed_call[1] is packed
    assert packed_call[2] is scale
    assert output.shape == (5, 4)
    assert output.is_contiguous()


def test_full_and_packed_state_materialize_fresh_meta_destinations(runtime, fake_aiter):
    torch = runtime.torch
    full_source = _layer(runtime, bias=True)
    packed_source = _layer(runtime, bias=True)
    packed_source.load_state_dict(full_source.state_dict())
    packed_source._quantize_weights()

    full_destination = _layer(runtime, bias=True, device="meta")
    packed_destination = _layer(runtime, bias=True, device="meta")
    full_result = full_destination.load_state_dict(full_source.state_dict())
    packed_result = packed_destination.load_state_dict(packed_source.state_dict())

    assert not full_result.missing_keys and not full_result.unexpected_keys
    assert full_destination.weight.device.type == "cpu"
    assert full_destination.bias.device.type == "cpu"
    assert torch.equal(full_destination.weight, full_source.weight)
    assert not packed_result.missing_keys and not packed_result.unexpected_keys
    assert packed_destination.weight is None
    assert packed_destination.weight_packed.device.type == "cpu"
    assert packed_destination.weight_scale.device.type == "cpu"
    assert packed_destination.weight_packed_provenance.device.type == "cpu"
    assert packed_destination.bias.device.type == "cpu"
    assert torch.equal(packed_destination.weight_packed, packed_source.weight_packed)


def test_packed_size_and_fsdp_transition_errors_are_explicit(
    runtime, fake_aiter, monkeypatch
):
    torch = runtime.torch
    layer = _layer(runtime)
    layer._quantize_weights()
    bad_state = layer.state_dict()
    bad_state["weight_packed"] = torch.empty(1, dtype=torch.uint8)

    with pytest.raises(ValueError, match="wrong sizes"):
        _layer(runtime).load_state_dict(bad_state)

    packed_source = _layer(runtime)
    packed_source._quantize_weights()
    full_source = _layer(runtime)
    monkeypatch.setattr(layer, "_is_fsdp_managed_parameter", lambda parameter: True)
    with pytest.raises(RuntimeError, match="packed state cannot be loaded after FSDP"):
        layer.load_state_dict(packed_source.state_dict())
    with pytest.raises(
        RuntimeError,
        match="full-precision state cannot replace an FSDP-managed packed parameter",
    ):
        layer.load_state_dict(full_source.state_dict())


def test_dimension_and_dtype_errors_precede_backend_calls(runtime, fake_aiter):
    torch = runtime.torch
    layer = _layer(runtime)
    layer._quantize_weights()

    with pytest.raises(ValueError, match="last dimension"):
        layer(torch.ones((2, 7), dtype=torch.bfloat16))
    with pytest.raises(TypeError, match="MXFP6 activation.*got torch.float32"):
        layer(torch.ones((2, 8), dtype=torch.float32))


def test_persistent_pack_initializes_unwritten_guard_bytes(runtime, monkeypatch):
    torch = runtime.torch
    module = runtime.module
    layer = _layer(runtime)

    def partial_pack_out(weight, packed, scale):
        packed[0] = 17
        scale[0] = 23
        return packed, scale

    monkeypatch.setattr(module, "quant_mxfp6_gemm_out", partial_pack_out)
    layer._quantize_weights()

    assert layer.weight_packed[0].item() == 17
    assert layer.weight_scale[0].item() == 23
    assert torch.count_nonzero(layer.weight_packed[1:]).item() == 0
    assert torch.count_nonzero(layer.weight_scale[1:]).item() == 0


def test_zero_row_paths_skip_aiter_and_preserve_shapes_and_bias(runtime, fake_aiter):
    torch = runtime.torch
    layer = _layer(runtime, bias=True)

    empty = torch.empty((2, 0, 8), dtype=torch.bfloat16)
    output = layer(empty)
    input_2d = empty.reshape(0, 8)
    packed, scale = layer.pack_activation(input_2d)
    packed_out = torch.empty(0, dtype=torch.uint8)
    scale_out = torch.empty(0, dtype=torch.uint8)
    layer.pack_activation_out(input_2d, packed_out, scale_out)
    output_packed = layer.forward_packed_2d(
        packed,
        scale,
        rows=0,
        output_dtype=torch.bfloat16,
    )
    weight_packed_size, weight_scale_size = runtime.module._abi_pack_sizes(4, 8)
    weight_packed = torch.empty(weight_packed_size, dtype=torch.uint8)
    weight_scale = torch.empty(weight_scale_size, dtype=torch.uint8)
    output_direct = torch.ops.xfuser.mxfp6_gemm(
        input_2d,
        weight_packed,
        weight_scale,
        4,
        8,
    )
    output_direct_packed = torch.ops.xfuser.mxfp6_gemm_packed(
        packed,
        weight_packed,
        scale,
        weight_scale,
        0,
        4,
        8,
    )

    assert output.shape == (2, 0, 4)
    assert output_packed.shape == (0, 4)
    assert output_direct.shape == output_direct_packed.shape == (0, 4)
    assert all(
        tensor.is_contiguous()
        for tensor in (output, output_packed, output_direct, output_direct_packed)
    )
    assert packed.numel() == scale.numel() == 0
    assert fake_aiter["packs"] == []
    assert fake_aiter["gemms"] == []
    assert fake_aiter["pack_out"] == []
    with pytest.raises(ValueError, match="non-negative"):
        layer.forward_packed_2d(
            packed,
            scale,
            rows=-1,
            output_dtype=torch.bfloat16,
        )


def test_packed_values_scales_and_provenance_require_contiguous_paired_state(
    runtime, fake_aiter
):
    torch = runtime.torch
    module = runtime.module
    packed_size, scale_size = module._abi_pack_sizes(4, 8)
    noncontiguous = torch.empty(packed_size * 2, dtype=torch.uint8)[::2]
    scale = torch.empty(scale_size, dtype=torch.uint8)

    with pytest.raises(ValueError, match="contiguous"):
        module._validate_packed_pair(
            noncontiguous,
            scale,
            rows=4,
            in_features=8,
            name="test",
        )
    with pytest.raises(ValueError, match="share a device"):
        module._validate_packed_pair(
            torch.empty(packed_size, dtype=torch.uint8),
            torch.empty(scale_size, dtype=torch.uint8, device="meta"),
            rows=4,
            in_features=8,
            name="test",
        )

    layer = _layer(runtime)
    layer._quantize_weights()
    with pytest.raises(ValueError, match="contiguous"):
        layer.forward_packed_2d(
            noncontiguous,
            scale,
            rows=4,
            output_dtype=torch.bfloat16,
        )
    bad_state = layer.state_dict()
    bad_state["weight_packed_provenance"] = torch.empty((8,), dtype=torch.int64)[::2]
    with pytest.raises(ValueError, match="provenance must be contiguous"):
        _layer(runtime).load_state_dict(bad_state)


def test_public_pack_size_must_match_layout_abi(runtime, fake_aiter, monkeypatch):
    module = runtime.module
    expected_packed, expected_scale = module._abi_pack_sizes(4, 8)
    monkeypatch.setattr(
        module,
        "mxfp6_gemm_pack_size",
        lambda rows, in_features: (expected_packed + 1, expected_scale),
    )

    with pytest.raises(RuntimeError, match=r"C0/C1\+Hadamard ABI"):
        _layer(runtime)._quantize_weights()


def test_packed_provenance_rejects_missing_or_colliding_logical_state(
    runtime, fake_aiter
):
    torch = runtime.torch
    source = _layer(runtime)
    source._quantize_weights()

    missing = source.state_dict()
    del missing["weight_packed_provenance"]
    with pytest.raises(RuntimeError, match="requires.*provenance"):
        _layer(runtime).load_state_dict(missing)

    colliding = source.state_dict()
    destination = runtime.module.xFuserMXFP6Linear(
        8,
        5,
        bias=False,
        dtype=torch.bfloat16,
    )
    assert runtime.module._abi_pack_sizes(5, 8) == runtime.module._abi_pack_sizes(4, 8)
    with pytest.raises(ValueError, match="logical shape"):
        destination.load_state_dict(colliding)

    bad_abi = source.state_dict()
    bad_abi["weight_packed_provenance"] = bad_abi["weight_packed_provenance"].clone()
    bad_abi["weight_packed_provenance"][0] += 1
    with pytest.raises(ValueError, match="layout ABI version"):
        _layer(runtime).load_state_dict(bad_abi)


def test_dtype_construction_loading_and_packed_conversion_are_explicit(
    runtime, fake_aiter
):
    torch = runtime.torch
    default_layer = runtime.module.xFuserMXFP6Linear(8, 4, bias=False)
    assert default_layer.weight.dtype is torch.bfloat16
    with pytest.raises(TypeError, match="dtype must be"):
        runtime.module.xFuserMXFP6Linear(
            8,
            4,
            bias=False,
            dtype=torch.float32,
        )

    fp16_source = _layer(runtime, dtype=torch.float16)
    bf16_destination = _layer(runtime)
    bf16_destination.load_state_dict(fp16_source.state_dict())
    assert bf16_destination.weight.dtype is torch.bfloat16
    assert bf16_destination._compute_dtype is torch.bfloat16

    fp16_source._quantize_weights()
    with pytest.raises(TypeError, match="checkpoint compute dtype"):
        _layer(runtime).load_state_dict(fp16_source.state_dict())

    packed = _layer(runtime, bias=True)
    packed._quantize_weights()
    original_bias_dtype = packed.bias.dtype
    with pytest.raises(TypeError, match="cannot be converted"):
        packed.to(dtype=torch.float16)
    assert packed.bias.dtype is original_bias_dtype
    assert packed._compute_dtype is torch.bfloat16
    assert packed.to("cpu") is packed


def test_fp6_walker_preserves_parent_fqn_filter_bias_and_replacement_count(
    runtime, fake_aiter
):
    torch = runtime.torch
    from xfuser.core.utils.runner_utils import quantize_linear_layers_to_fp6

    model = torch.nn.Module()
    model.block = torch.nn.Module()
    model.block.keep = torch.nn.Linear(8, 4, bias=True, dtype=torch.bfloat16)
    model.block.keep.eval()
    model.block.skip = torch.nn.Linear(8, 4, bias=False, dtype=torch.bfloat16)
    expected_bias = model.block.keep.bias.detach().clone()
    seen = []

    replaced = quantize_linear_layers_to_fp6(
        model,
        parent_name="transformer",
        filter_fn=lambda module, fqn: seen.append(fqn) or fqn.endswith(".keep"),
        device=torch.device("cpu"),
        offload_to_cpu=True,
    )

    assert replaced == 1
    assert seen == ["transformer.block.keep", "transformer.block.skip"]
    assert isinstance(model.block.keep, runtime.module.xFuserMXFP6Linear)
    assert isinstance(model.block.skip, torch.nn.Linear)
    assert torch.equal(model.block.keep.bias, expected_bias)
    assert model.block.keep.training is False
    assert model.block.keep.weight_packed.device.type == "cpu"
    assert model.block.keep.weight_scale.device.type == "cpu"


def test_fp6_walker_restores_source_parameter_ownership_after_pack_failure(
    runtime, monkeypatch
):
    torch = runtime.torch
    from xfuser.core.utils.runner_utils import quantize_linear_layers_to_fp6

    model = torch.nn.Module()
    model.proj = torch.nn.Linear(8, 4, bias=True, dtype=torch.bfloat16)
    original_weight = model.proj.weight
    original_bias = model.proj.bias
    monkeypatch.setattr(
        runtime.module,
        "quant_mxfp6_gemm_out",
        lambda *args: (_ for _ in ()).throw(RuntimeError("pack failed")),
    )

    with pytest.raises(RuntimeError, match="pack failed"):
        quantize_linear_layers_to_fp6(model)

    assert isinstance(model.proj, torch.nn.Linear)
    assert model.proj.weight is original_weight
    assert model.proj.bias is original_bias


def test_fp6_walker_install_failure_keeps_source_and_alias_intact(runtime, fake_aiter):
    torch = runtime.torch
    from xfuser.core.utils.runner_utils import quantize_linear_layers_to_fp6

    class RejectingParent(torch.nn.Module):
        def __setattr__(self, name, value):
            if name == "proj" and isinstance(value, runtime.module.xFuserMXFP6Linear):
                raise RuntimeError("install failed")
            super().__setattr__(name, value)

    model = RejectingParent()
    source = torch.nn.Linear(8, 4, bias=True, dtype=torch.bfloat16)
    model.proj = source
    model.alias = source
    original_weight = source.weight
    original_bias = source.bias

    with pytest.raises(RuntimeError, match="install failed"):
        quantize_linear_layers_to_fp6(model)

    assert model.proj is source
    assert model.alias is source
    assert source.weight is original_weight
    assert source.bias is original_bias


def test_mxfp4_walker_explicitly_replaces_only_overrides_with_mxfp6(
    runtime, fake_aiter
):
    torch = runtime.torch
    from xfuser.core.utils.runner_utils import quantize_linear_layers_to_fp4

    model = torch.nn.Module()
    model.sensitive = torch.nn.Linear(8, 4, bias=True, dtype=torch.bfloat16)
    model.untouched = torch.nn.Linear(8, 4, bias=False, dtype=torch.bfloat16)

    quantize_linear_layers_to_fp4(
        model,
        fp8_layers=("sensitive",),
        filter_fn=lambda module, fqn: fqn == "sensitive",
        use_fp6_for_overrides=True,
        offload_to_cpu=True,
    )

    assert isinstance(model.sensitive, runtime.module.xFuserMXFP6Linear)
    assert model.sensitive.weight_packed.device.type == "cpu"
    assert isinstance(model.untouched, torch.nn.Linear)


def test_mxfp4_override_failure_keeps_source_and_alias_intact(runtime, monkeypatch):
    torch = runtime.torch
    from xfuser.core.utils.runner_utils import quantize_linear_layers_to_fp4

    model = torch.nn.Module()
    source = torch.nn.Linear(8, 4, bias=True, dtype=torch.bfloat16)
    model.sensitive = source
    model.alias = source
    original_weight = source.weight
    original_bias = source.bias
    monkeypatch.setattr(
        runtime.module,
        "quant_mxfp6_gemm_out",
        lambda *args: (_ for _ in ()).throw(RuntimeError("pack failed")),
    )

    with pytest.raises(RuntimeError, match="pack failed"):
        quantize_linear_layers_to_fp4(
            model,
            fp8_layers=("sensitive",),
            filter_fn=lambda module, fqn: fqn == "sensitive",
            use_fp6_for_overrides=True,
        )

    assert model.sensitive is source
    assert model.alias is source
    assert source.weight is original_weight
    assert source.bias is original_bias
