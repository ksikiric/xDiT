from xfuser.model_executor.layers.aiter_int8_linear import (
    select_aiter_int8_variant,
    supports_aiter_int8_linear,
)


def test_aiter_int8_dimension_contract():
    assert supports_aiter_int8_linear(512, 272)
    assert supports_aiter_int8_linear(5120, 13824)
    assert not supports_aiter_int8_linear(128, 256)
    assert not supports_aiter_int8_linear(320, 256)
    assert not supports_aiter_int8_linear(512, 270)


def test_aiter_int8_variant_selection():
    assert (
        select_aiter_int8_variant(9450, 5120, 5120, 256).name == "wan_k5120_persistent"
    )
    assert select_aiter_int8_variant(32, 5120, 5120, 256).name == "default"
    assert select_aiter_int8_variant(9450, 5120, 13824, 256).name == "wan_k13824"
    assert select_aiter_int8_variant(9450, 4096, 4096, 256).name == "default"
