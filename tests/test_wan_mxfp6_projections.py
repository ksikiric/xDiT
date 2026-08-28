"""CPU-only tests for Wan's shared MXFP6 projection activation packs."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
WAN_TRANSFORMER = ROOT / "xfuser/model_executor/models/transformers/transformer_wan.py"
PROJECTION_METHODS = {
    "_run_shared_mxfp6_projections",
    "_get_qkv_projections",
    "_get_added_kv_projections",
}


def _load_projection_processor(torch, mxfp6_type):
    """Load only the projection helpers, without importing Diffusers or AITER."""

    tree = ast.parse(WAN_TRANSFORMER.read_text())
    processor = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "xFuserWanAttnProcessor"
    )
    methods = [
        node
        for node in processor.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in PROJECTION_METHODS
    ]
    isolated = ast.ClassDef(
        name="ProjectionProcessor",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[isolated], type_ignores=[]))
    namespace = {
        "WanAttention": object,
        "torch": torch,
        "xFuserMXFP6Linear": mxfp6_type,
    }
    exec(compile(module, str(WAN_TRANSFORMER), "exec"), namespace)  # noqa: S102
    return namespace["ProjectionProcessor"]


@pytest.fixture(scope="module")
def runtime():
    torch = pytest.importorskip(
        "torch", reason="PyTorch is required for Wan projection tests"
    )

    class FakeMXFP6Linear(torch.nn.Module):
        def __init__(
            self,
            in_features,
            out_features,
            *,
            offset=0.0,
            bias=True,
            compute_dtype=None,
        ):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self._compute_dtype = compute_dtype or torch.float32
            matrix = (
                torch.arange(
                    out_features * in_features,
                    dtype=torch.float32,
                ).reshape(out_features, in_features)
                + offset
            ) / 10
            self.register_buffer("matrix", matrix)
            self.weight_packed = self.matrix
            if bias:
                bias_value = (
                    torch.arange(out_features, dtype=torch.float32) + offset
                ) / 7
                self.register_buffer("bias_value", bias_value)
            else:
                self.bias_value = None
            self.forward_calls = 0
            self.pack_calls = 0
            self.gemm_calls = 0
            self.pack_shapes = []

        def pack_activation(self, input_2d):
            self.pack_calls += 1
            self.pack_shapes.append(tuple(input_2d.shape))
            scale = torch.empty(0, dtype=torch.uint8, device=input_2d.device)
            return input_2d, scale

        def forward_packed_2d(
            self,
            activation_packed,
            activation_scale,
            rows,
            output_dtype,
        ):
            del activation_scale
            self.gemm_calls += 1
            assert rows == activation_packed.shape[0]
            output = torch.nn.functional.linear(
                activation_packed,
                self.matrix,
                self.bias_value,
            ).to(output_dtype)
            padded = torch.empty(
                (rows, self.out_features * 2),
                dtype=output.dtype,
                device=output.device,
            )
            padded[:, ::2] = output
            return padded[:, ::2]

        def forward(self, input_tensor):
            self.forward_calls += 1
            original_shape = input_tensor.shape
            input_features = self.matrix.shape[1]
            input_2d = input_tensor.reshape(-1, input_features)
            packed, scale = self.pack_activation(input_2d)
            output = self.forward_packed_2d(
                packed,
                scale,
                input_2d.shape[0],
                input_tensor.dtype,
            )
            return output.reshape(*original_shape[:-1], self.out_features).contiguous()

    class OrdinaryProjection(torch.nn.Module):
        def __init__(
            self,
            in_features,
            out_features,
            *,
            offset=0.0,
            bias=True,
        ):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            matrix = (
                torch.arange(
                    out_features * in_features,
                    dtype=torch.float32,
                ).reshape(out_features, in_features)
                + offset
            ) / 10
            self.register_buffer("matrix", matrix)
            if bias:
                bias_value = (
                    torch.arange(out_features, dtype=torch.float32) + offset
                ) / 7
                self.register_buffer("bias_value", bias_value)
            else:
                self.bias_value = None
            self.calls = 0

        def forward(self, input_tensor):
            self.calls += 1
            return torch.nn.functional.linear(
                input_tensor,
                self.matrix,
                self.bias_value,
            )

    processor_type = _load_projection_processor(torch, FakeMXFP6Linear)
    return SimpleNamespace(
        torch=torch,
        FakeMXFP6Linear=FakeMXFP6Linear,
        OrdinaryProjection=OrdinaryProjection,
        processor=processor_type(),
    )


def _noncontiguous_input(torch, leading_shape, in_features):
    element_count = in_features
    for dimension in leading_shape:
        element_count *= dimension
    contiguous = torch.arange(element_count, dtype=torch.float32).reshape(
        *leading_shape, in_features
    )
    if len(leading_shape) < 2:
        return contiguous
    return contiguous.transpose(0, 1).contiguous().transpose(0, 1)


def _assert_projection(runtime, output, input_tensor, projection):
    torch = runtime.torch
    expected = torch.nn.functional.linear(
        input_tensor,
        projection.matrix,
        projection.bias_value,
    )
    torch.testing.assert_close(output, expected)
    assert output.shape == (
        *input_tensor.shape[:-1],
        projection.out_features,
    )
    assert output.dtype == input_tensor.dtype
    assert output.device == input_tensor.device
    assert output.is_contiguous()


def test_unfused_self_attention_packs_once_for_three_gemms(runtime):
    hidden_states = _noncontiguous_input(runtime.torch, (3, 2, 1), 4)
    assert not hidden_states.is_contiguous()
    query = runtime.FakeMXFP6Linear(4, 3, offset=1.0)
    key = runtime.FakeMXFP6Linear(4, 3, offset=2.0)
    value = runtime.FakeMXFP6Linear(4, 3, offset=3.0)
    attn = SimpleNamespace(
        fused_projections=False,
        cross_attention_dim_head=None,
        to_q=query,
        to_k=key,
        to_v=value,
    )

    outputs = runtime.processor._get_qkv_projections(attn, hidden_states, None)

    assert [layer.pack_calls for layer in (query, key, value)] == [1, 0, 0]
    assert [layer.gemm_calls for layer in (query, key, value)] == [1, 1, 1]
    assert [layer.forward_calls for layer in (query, key, value)] == [0, 0, 0]
    assert query.pack_shapes == [(6, 4)]
    for output, projection in zip(outputs, (query, key, value)):
        _assert_projection(runtime, output, hidden_states, projection)


def test_unfused_cross_attention_keeps_q_independent_and_shares_kv(runtime):
    torch = runtime.torch
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    encoder_hidden_states = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
    query = runtime.FakeMXFP6Linear(4, 2, offset=1.0)
    key = runtime.FakeMXFP6Linear(4, 2, offset=2.0)
    value = runtime.FakeMXFP6Linear(4, 2, offset=3.0)
    attn = SimpleNamespace(
        fused_projections=False,
        cross_attention_dim_head=2,
        to_q=query,
        to_k=key,
        to_v=value,
    )

    outputs = runtime.processor._get_qkv_projections(
        attn, hidden_states, encoder_hidden_states
    )

    assert query.forward_calls == 1
    assert [layer.pack_calls for layer in (query, key, value)] == [1, 1, 0]
    assert [layer.gemm_calls for layer in (query, key, value)] == [1, 1, 1]
    assert [layer.forward_calls for layer in (key, value)] == [0, 0]
    assert key.pack_shapes == [(10, 4)]
    _assert_projection(runtime, outputs[0], hidden_states, query)
    _assert_projection(runtime, outputs[1], encoder_hidden_states, key)
    _assert_projection(runtime, outputs[2], encoder_hidden_states, value)


def test_unfused_i2v_added_kv_packs_once_for_two_gemms(runtime):
    image_states = _noncontiguous_input(runtime.torch, (2, 3), 4)
    key = runtime.FakeMXFP6Linear(4, 3, offset=2.0)
    value = runtime.FakeMXFP6Linear(4, 3, offset=3.0)
    attn = SimpleNamespace(
        fused_projections=False,
        add_k_proj=key,
        add_v_proj=value,
    )

    outputs = runtime.processor._get_added_kv_projections(attn, image_states)

    assert [layer.pack_calls for layer in (key, value)] == [1, 0]
    assert [layer.gemm_calls for layer in (key, value)] == [1, 1]
    assert [layer.forward_calls for layer in (key, value)] == [0, 0]
    for output, projection in zip(outputs, (key, value)):
        _assert_projection(runtime, output, image_states, projection)


def test_fused_projection_paths_remain_unchanged(runtime):
    torch = runtime.torch
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    encoder_hidden_states = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
    image_states = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)

    self_qkv = runtime.OrdinaryProjection(4, 6, offset=1.0)
    self_attn = SimpleNamespace(
        fused_projections=True,
        cross_attention_dim_head=None,
        to_qkv=self_qkv,
    )
    self_outputs = runtime.processor._get_qkv_projections(
        self_attn, hidden_states, None
    )
    assert self_qkv.calls == 1
    assert [output.shape[-1] for output in self_outputs] == [2, 2, 2]

    cross_q = runtime.OrdinaryProjection(4, 2, offset=2.0)
    cross_kv = runtime.OrdinaryProjection(4, 4, offset=3.0)
    cross_attn = SimpleNamespace(
        fused_projections=True,
        cross_attention_dim_head=2,
        to_q=cross_q,
        to_kv=cross_kv,
    )
    cross_outputs = runtime.processor._get_qkv_projections(
        cross_attn, hidden_states, encoder_hidden_states
    )
    assert cross_q.calls == 1
    assert cross_kv.calls == 1
    assert [output.shape[-1] for output in cross_outputs] == [2, 2, 2]

    added_kv = runtime.OrdinaryProjection(4, 4, offset=4.0)
    added_attn = SimpleNamespace(
        fused_projections=True,
        to_added_kv=added_kv,
    )
    added_outputs = runtime.processor._get_added_kv_projections(
        added_attn, image_states
    )
    assert added_kv.calls == 1
    assert [output.shape[-1] for output in added_outputs] == [2, 2]


def test_non_mxfp6_unfused_paths_keep_ordinary_projection_calls(runtime):
    torch = runtime.torch
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    encoder_hidden_states = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)

    self_layers = [
        runtime.OrdinaryProjection(4, 2, offset=float(index)) for index in range(3)
    ]
    self_attn = SimpleNamespace(
        fused_projections=False,
        cross_attention_dim_head=None,
        to_q=self_layers[0],
        to_k=self_layers[1],
        to_v=self_layers[2],
    )
    runtime.processor._get_qkv_projections(self_attn, hidden_states, None)
    assert [layer.calls for layer in self_layers] == [1, 1, 1]

    cross_layers = [
        runtime.OrdinaryProjection(4, 2, offset=float(index + 3)) for index in range(3)
    ]
    cross_attn = SimpleNamespace(
        fused_projections=False,
        cross_attention_dim_head=2,
        to_q=cross_layers[0],
        to_k=cross_layers[1],
        to_v=cross_layers[2],
    )
    runtime.processor._get_qkv_projections(
        cross_attn, hidden_states, encoder_hidden_states
    )
    assert [layer.calls for layer in cross_layers] == [1, 1, 1]

    added_layers = [
        runtime.OrdinaryProjection(4, 2, offset=float(index + 6)) for index in range(2)
    ]
    added_attn = SimpleNamespace(
        fused_projections=False,
        add_k_proj=added_layers[0],
        add_v_proj=added_layers[1],
    )
    runtime.processor._get_added_kv_projections(added_attn, encoder_hidden_states)
    assert [layer.calls for layer in added_layers] == [1, 1]


@pytest.mark.parametrize("difference", ["features", "dtype", "device"])
def test_incompatible_mxfp6_group_falls_back_without_partial_sharing(
    runtime, difference
):
    torch = runtime.torch
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    layers = [runtime.FakeMXFP6Linear(4, 2, offset=float(index)) for index in range(3)]
    if difference == "features":
        layers[-1].in_features = 5
    elif difference == "dtype":
        layers[-1]._compute_dtype = torch.float64
    else:
        layers[-1].weight_packed = SimpleNamespace(device=torch.device("meta"))
    attn = SimpleNamespace(
        fused_projections=False,
        cross_attention_dim_head=None,
        to_q=layers[0],
        to_k=layers[1],
        to_v=layers[2],
    )

    outputs = runtime.processor._get_qkv_projections(attn, hidden_states, None)

    assert [layer.forward_calls for layer in layers] == [1, 1, 1]
    assert [layer.pack_calls for layer in layers] == [1, 1, 1]
    assert [layer.gemm_calls for layer in layers] == [1, 1, 1]
    for output, projection in zip(outputs, layers):
        _assert_projection(runtime, output, hidden_states, projection)


def test_shared_projection_helper_is_fullgraph_compile_compatible(runtime):
    torch = runtime.torch
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")

    class CompileProjection(runtime.FakeMXFP6Linear):
        def pack_activation(self, input_2d):
            return input_2d, input_2d.new_empty((0,))

        def forward_packed_2d(
            self,
            activation_packed,
            activation_scale,
            rows,
            output_dtype,
        ):
            del activation_scale, rows
            return torch.nn.functional.linear(
                activation_packed,
                self.matrix,
                self.bias_value,
            ).to(output_dtype)

    first = CompileProjection(4, 2, offset=1.0)
    second = CompileProjection(4, 3, offset=2.0)

    def project(input_tensor):
        return runtime.processor._run_shared_mxfp6_projections(
            input_tensor, (first, second)
        )

    compiled = torch.compile(
        project,
        backend="eager",
        dynamic=True,
        fullgraph=True,
    )
    outputs = compiled(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))

    assert [output.shape for output in outputs] == [(2, 3, 2), (2, 3, 3)]
    assert all(output.is_contiguous() for output in outputs)
