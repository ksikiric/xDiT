import pytest
import torch

from xfuser.model_executor.layers.aiter_int8_linear import (
    _quantize_rows_int8,
    xFuserAiterInt8Linear,
)


def _gfx950_available() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    try:
        import aiter  # noqa: F401
        from aiter.ops.quant import per_token_quant_i8_rne_hip  # noqa: F401
    except ImportError:
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


pytestmark = pytest.mark.skipif(
    not _gfx950_available(),
    reason="AITER true-INT8 linear requires ROCm gfx950",
)


@torch.no_grad()
def test_aiter_int8_quantizer_uses_round_to_nearest():
    values = torch.zeros(
        (1, 256),
        device="cuda",
        dtype=torch.bfloat16,
    )
    values[0, :9] = torch.tensor(
        [20.5, 21.5, 62.5, 63.5, -20.5, -21.5, -62.5, -63.5, 127.0],
        device="cuda",
        dtype=torch.bfloat16,
    )
    quantized, scale = _quantize_rows_int8(values)

    assert scale.item() == 1.0
    assert quantized[0, :9].cpu().tolist() == [
        20,
        22,
        62,
        64,
        -20,
        -22,
        -62,
        -64,
        127,
    ]


@torch.no_grad()
def test_aiter_int8_linear_shape_accuracy_and_determinism():
    generator = torch.Generator(device="cuda").manual_seed(17)
    m, k, n = 131, 512, 272
    x = torch.randn(
        (2, m, k),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        (n, k),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    bias = torch.randn(
        (n,),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    layer = xFuserAiterInt8Linear(
        k,
        n,
        bias=True,
        device="meta",
        dtype=torch.bfloat16,
    )
    layer.load_and_quantize_weights(weight, bias, device="cuda")

    output = layer(x)
    repeat = layer(x)
    compiled = torch.compile(layer, fullgraph=True)
    compiled_output = compiled(x)
    reference = torch.nn.functional.linear(x, weight, bias)

    assert output.shape == (2, m, n)
    assert output.dtype == torch.bfloat16
    assert torch.equal(output, repeat)
    assert torch.equal(output, compiled_output)
    assert torch.isfinite(output.float()).all()
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(),
        reference.float().flatten(),
        dim=0,
    )
    assert cosine > 0.999
    torch._dynamo.reset()


@torch.no_grad()
def test_aiter_int8_linear_zero_activation_returns_bias():
    k, n = 512, 272
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16)
    layer = xFuserAiterInt8Linear(
        k,
        n,
        bias=True,
        device="meta",
        dtype=torch.bfloat16,
    )
    layer.load_and_quantize_weights(weight, bias, device="cuda")
    output = layer(torch.zeros((4, k), device="cuda", dtype=torch.bfloat16))

    assert torch.isfinite(output.float()).all()
    assert torch.equal(output, bias.expand_as(output))
