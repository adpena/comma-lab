from __future__ import annotations

import brotli
import numpy as np

from tac.boundary_math.witness_crosstensor_codec import (
    CODE_TRANSFORM_FRAME_DELTA_MOD256,
    CODE_TRANSFORM_RAW,
    decode_base_quantized,
    decode_code_quantized,
    derive_base_permutation_plan,
    derive_code_transform_plan,
    encode_base_quantized,
    encode_code_quantized,
    quantized_base,
)
from tac.witness_dsl.gauge import (
    GaugeComponent,
    WitnessCrossTensorCoderGauge,
    witness_cross_tensor_coder_byte_close_flags,
)


def _params() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(17)
    return {
        "a.weight": rng.normal(size=(12, 7)).astype(np.float32),
        "b.weight": np.repeat(np.arange(9, dtype=np.float32)[:, None], 11, axis=1),
        "bias": rng.normal(size=13).astype(np.float32),
    }


def test_base_storage_permutation_roundtrips_canonical_int8() -> None:
    params = _params()
    order = tuple(params)
    plan = derive_base_permutation_plan(params, order)
    raw = encode_base_quantized(params, order, plan.transposed_names)
    got = decode_base_quantized(raw, order, {k: v.shape for k, v in params.items()}, plan.transposed_names)
    expected = quantized_base(params, order)
    assert got.keys() == expected.keys()
    assert all(np.array_equal(got[name], expected[name]) for name in order)
    assert plan.selected_brotli_bytes <= plan.baseline_brotli_bytes
    assert len(brotli.compress(raw, quality=11)) == plan.selected_brotli_bytes


def test_code_frame_delta_mod256_roundtrips_extreme_int8() -> None:
    q = np.array([[-128, 127], [127, -128], [-127, 126], [126, -127]], dtype=np.int8)
    raw = encode_code_quantized(q, CODE_TRANSFORM_FRAME_DELTA_MOD256)
    got = decode_code_quantized(raw, q.shape, CODE_TRANSFORM_FRAME_DELTA_MOD256)
    assert np.array_equal(got, q)


def test_raw_code_roundtrip() -> None:
    q = np.arange(24, dtype=np.int8).reshape(6, 4)
    raw = encode_code_quantized(q, CODE_TRANSFORM_RAW)
    assert np.array_equal(decode_code_quantized(raw, q.shape, CODE_TRANSFORM_RAW), q)


def test_code_plan_is_default_preserving_when_delta_loses() -> None:
    rng = np.random.default_rng(93)
    code = rng.normal(size=(20, 5)).astype(np.float32)
    plan = derive_code_transform_plan(code)
    assert plan.selected_brotli_bytes <= plan.baseline_brotli_bytes
    assert plan.transform in {CODE_TRANSFORM_RAW, CODE_TRANSFORM_FRAME_DELTA_MOD256}
    assert plan.exact_unique_rows <= 20
    assert plan.exact_unique_pairs <= 10


def test_decode_rejects_trailing_bytes() -> None:
    params = _params()
    order = tuple(params)
    raw = encode_base_quantized(params, order, ()) + b"x"
    try:
        decode_base_quantized(raw, order, {k: v.shape for k, v in params.items()}, ())
    except ValueError as exc:
        assert "unconsumed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("trailing byte was not rejected")


def test_dsl_byte_close_chart_is_default_preserving_and_uses_real_flag() -> None:
    assert GaugeComponent.WITNESS_CROSS_TENSOR_CODER.value == "witness_cross_tensor_coder"
    assert witness_cross_tensor_coder_byte_close_flags(WitnessCrossTensorCoderGauge.IDENTITY) == ()
    assert witness_cross_tensor_coder_byte_close_flags(WitnessCrossTensorCoderGauge.AUTO_LOSSLESS) == (
        "--cross-tensor-codec",
        "auto_lossless",
    )
