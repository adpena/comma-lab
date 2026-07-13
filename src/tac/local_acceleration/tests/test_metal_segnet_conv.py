from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tac.local_acceleration import metal_segnet_conv as msc


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _load_benchmark_module():
    path = Path(__file__).resolve().parents[4] / "experiments/bench_custom_metal_segnet_conv.py"
    spec = importlib.util.spec_from_file_location("bench_custom_metal_segnet_conv_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pointwise_numpy_reference_matches_float32_matmul() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, 3, 5, 11), dtype=np.float32)
    weight = rng.standard_normal((11, 13), dtype=np.float32)
    bias = rng.standard_normal(13, dtype=np.float32)
    actual = msc.pointwise_1x1_numpy_fp32(x, weight, bias)
    expected = (x.reshape(-1, 11) @ weight + bias).reshape(2, 3, 5, 13)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_pointwise_numpy_reference_is_byte_deterministic() -> None:
    rng = np.random.default_rng(23)
    x = rng.uniform(-2, 2, size=(1, 4, 7, 9)).astype(np.float16)
    weight = rng.uniform(-1, 1, size=(9, 17)).astype(np.float16)
    first = msc.pointwise_1x1_numpy_fp32(x, weight)
    second = msc.pointwise_1x1_numpy_fp32(x, weight)
    assert _sha(first) == _sha(second)
    assert np.array_equal(first, second)


def test_pointwise_oihw_to_kn_preserves_every_input_channel() -> None:
    weight_oihw = np.arange(13 * 11, dtype=np.float32).reshape(13, 11, 1, 1)
    actual = msc._pointwise_weight_kn_from_oihw(weight_oihw)
    expected = weight_oihw[:, :, 0, 0].T
    assert actual.shape == (11, 13)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("variant", ["int8", "int4"])
def test_pointwise_quantization_is_deterministic_and_bounded(variant: str) -> None:
    rng = np.random.default_rng(11)
    weight = rng.uniform(-3, 3, size=(15, 19)).astype(np.float32)
    quantize = (
        msc.quantize_pointwise_int8
        if variant == "int8"
        else msc.quantize_pointwise_int4
    )
    dequantize = (
        msc.dequantize_pointwise_int8
        if variant == "int8"
        else msc.dequantize_pointwise_int4
    )
    first = quantize(weight)
    second = quantize(weight)
    assert first.variant == variant
    assert first.values.tobytes() == second.values.tobytes()
    assert first.scales.tobytes() == second.scales.tobytes()
    reconstructed = dequantize(first)
    assert reconstructed.shape == weight.shape
    qmax = 127 if variant == "int8" else 7
    bound = np.max(np.abs(weight), axis=0) / np.float32(qmax) / np.float32(2)
    assert np.all(np.max(np.abs(reconstructed - weight), axis=0) <= bound + 1e-6)


def test_zero_weight_channels_quantize_exactly_to_zero() -> None:
    weight = np.zeros((5, 4), dtype=np.float32)
    for packet, dequantize in (
        (msc.quantize_pointwise_int8(weight), msc.dequantize_pointwise_int8),
        (msc.quantize_pointwise_int4(weight), msc.dequantize_pointwise_int4),
    ):
        np.testing.assert_array_equal(packet.scales, np.ones(4, dtype=np.float32))
        np.testing.assert_array_equal(dequantize(packet), weight)


def test_signed_int4_pack_roundtrip_odd_count() -> None:
    values = np.array([-8, -7, -1, 0, 1, 6, 7], dtype=np.int8)
    packed = msc.pack_signed_int4(values)
    assert packed.dtype == np.uint8
    assert packed.size == 4
    np.testing.assert_array_equal(
        msc.unpack_signed_int4(packed, count=values.size),
        values,
    )


def test_depthwise_numpy_reference_known_stride_two_case() -> None:
    x = np.arange(1 * 4 * 5 * 2, dtype=np.float32).reshape(1, 4, 5, 2)
    weight = np.ones((2, 3, 3), dtype=np.float32)
    weight[1] *= np.float32(2.0)
    actual = msc.depthwise_conv2d_numpy_fp32(
        x,
        weight,
        stride=2,
        padding=1,
    )
    expected = np.zeros((1, 2, 3, 2), dtype=np.float32)
    for oh in range(2):
        for ow in range(3):
            for kh in range(3):
                ih = oh * 2 + kh - 1
                for kw in range(3):
                    iw = ow * 2 + kw - 1
                    if 0 <= ih < 4 and 0 <= iw < 5:
                        expected[:, oh, ow, :] += x[:, ih, iw, :] * weight[:, kh, kw]
    np.testing.assert_array_equal(actual, expected)


def test_depthwise_numpy_reference_is_byte_deterministic() -> None:
    rng = np.random.default_rng(31)
    x = rng.standard_normal((2, 7, 9, 4), dtype=np.float32).astype(np.float16)
    weight = rng.standard_normal((4, 5, 5), dtype=np.float32).astype(np.float16)
    first = msc.depthwise_conv2d_numpy_fp32(x, weight, stride=1, padding=2)
    second = msc.depthwise_conv2d_numpy_fp32(x, weight, stride=1, padding=2)
    assert _sha(first) == _sha(second)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off"])
def test_custom_segnet_conv_flag_defaults_off(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(msc.CUSTOM_SEGNET_CONV_FLAG, raising=False)
    else:
        monkeypatch.setenv(msc.CUSTOM_SEGNET_CONV_FLAG, value)
    assert not msc.custom_segnet_conv_env_requested()


@pytest.mark.parametrize("value", ["1", "TRUE", "yes", "On"])
def test_custom_segnet_conv_truthy_flag(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(msc.CUSTOM_SEGNET_CONV_FLAG, value)
    monkeypatch.setattr(msc, "metal_segnet_conv_backend_available", lambda: True)
    assert msc.custom_segnet_conv_env_requested()
    assert msc.custom_segnet_conv_enabled()


def test_kernel_sources_preserve_program_contract() -> None:
    for variant in sorted(msc.VALID_POINTWISE_WEIGHT_VARIANTS):
        source = msc._pointwise_source(variant)
        assert "simdgroup_half8x8" in source
        assert "simdgroup_float8x8" in source
        assert "simdgroup_multiply_accumulate" in source
        assert "atomic" not in source
    assert "float(w[weight_idx])" in msc._pointwise_source("int8")
    assert "weight_idx >> 1" in msc._pointwise_source("int4")
    assert "atomic" not in msc._DEPTHWISE_SOURCE


def test_signature_is_default_off_forward_only_and_names_n600_gate() -> None:
    signature = msc.custom_segnet_conv_signature()
    assert signature["built"] is True
    assert signature["default_enabled"] is False
    assert signature["score_claim"] is False
    assert signature["promotion_eligible"] is False
    assert signature["vjp"] == "not-implemented-forward-only-fail-closed"
    assert signature["promotion_gate"] == msc.N600_FIDELITY_GATE
    assert signature["pointwise"]["weight_variants"] == ["fp16", "int4", "int8"]


def test_amdahl_uses_measured_time_substitution_not_mac_fraction() -> None:
    bench = _load_benchmark_module()
    inventory = {
        "full_segnet": {
            "by_kind": {
                "pointwise-1x1": {"mac_fraction": 0.5},
                "depthwise": {"mac_fraction": 0.2},
            }
        }
    }
    pointwise = [
        {
            "shape": {"occurrences": 1},
            "native": {"mlx-native-fp32": {"median_ms": 2.0}},
            "custom": {
                "fp16": {"timing": {"median_ms": 1.0}},
                "int8": {"timing": {"median_ms": 0.5}},
                "int4": {"timing": {"median_ms": 4.0}},
            },
        }
    ]
    depthwise = [
        {
            "shape": {"occurrences": 1},
            "native": {"mlx-native-fp32": {"median_ms": 1.0}},
            "custom": {"fp16": {"timing": {"median_ms": 0.5}}},
        }
    ]
    full_forward = {
        "native": {"timing": {"median_ms": 10.0}},
        "arms": {
            "fp16": {"direct_speedup_vs_native": 1.1},
            "int8": {"direct_speedup_vs_native": 1.2},
            "int4": {"direct_speedup_vs_native": 0.8},
        },
    }
    result = bench._compose_amdahl(
        inventory,
        pointwise,
        depthwise,
        full_forward,
    )
    fp16 = result["arms"]["fp16"]
    assert fp16["derived_full_forward_amdahl_speedup"] == pytest.approx(10.0 / 8.5)
    assert fp16["derived_mac_share_model_speedup"] == pytest.approx(1.0 / 0.65)
    assert fp16["derived_full_forward_amdahl_speedup"] != pytest.approx(
        fp16["derived_mac_share_model_speedup"]
    )
    assert fp16["derived_full_forward_amdahl_refusal"] is None


def test_amdahl_refuses_when_isolated_native_time_exceeds_full_wall() -> None:
    bench = _load_benchmark_module()
    inventory = {
        "full_segnet": {
            "by_kind": {
                "pointwise-1x1": {"mac_fraction": 0.5},
                "depthwise": {"mac_fraction": 0.2},
            }
        }
    }
    pointwise = [
        {
            "shape": {"occurrences": 1},
            "native": {"mlx-native-fp32": {"median_ms": 2.0}},
            "custom": {
                variant: {"timing": {"median_ms": 1.0}}
                for variant in ("fp16", "int8", "int4")
            },
        }
    ]
    depthwise = [
        {
            "shape": {"occurrences": 1},
            "native": {"mlx-native-fp32": {"median_ms": 1.0}},
            "custom": {"fp16": {"timing": {"median_ms": 0.5}}},
        }
    ]
    full_forward = {
        "native": {"timing": {"median_ms": 2.0}},
        "arms": {
            variant: {"direct_speedup_vs_native": 1.0}
            for variant in ("fp16", "int8", "int4")
        },
    }
    result = bench._compose_amdahl(
        inventory,
        pointwise,
        depthwise,
        full_forward,
    )
    assert result["arms"]["fp16"]["derived_full_forward_amdahl_speedup"] is None
    assert "exceeds" in result["arms"]["fp16"]["derived_full_forward_amdahl_refusal"]


def _require_metal() -> None:
    if not msc.metal_segnet_conv_backend_available():
        pytest.skip("evaluated MLX Metal device unavailable")


@pytest.mark.parametrize("variant", ["fp16", "int8", "int4"])
def test_pointwise_metal_matches_numpy_and_is_deterministic(variant: str) -> None:
    _require_metal()
    import mlx.core as mx

    rng = np.random.default_rng(101)
    x_np = rng.uniform(-2, 2, size=(1, 3, 5, 11)).astype(np.float16)
    weight_kn = rng.uniform(-1, 1, size=(11, 13)).astype(np.float32)
    x = mx.array(x_np, dtype=mx.float16)
    if variant == "fp16":
        runtime_weight = mx.array(weight_kn.astype(np.float16), dtype=mx.float16)
        expected_weight = weight_kn.astype(np.float16)
        kwargs: dict[str, object] = {}
    elif variant == "int8":
        packet = msc.quantize_pointwise_int8(weight_kn)
        runtime_weight = mx.array(packet.values, dtype=mx.int8)
        expected_weight = msc.dequantize_pointwise_int8(packet).astype(np.float16)
        kwargs = {
            "scales": mx.array(packet.scales, dtype=mx.float32),
            "cout": packet.cout,
        }
    else:
        packet = msc.quantize_pointwise_int4(weight_kn)
        runtime_weight = mx.array(packet.values, dtype=mx.uint8)
        expected_weight = msc.dequantize_pointwise_int4(packet).astype(np.float16)
        kwargs = {
            "scales": mx.array(packet.scales, dtype=mx.float32),
            "cout": packet.cout,
        }
    first = msc.pointwise_1x1_metal(
        x,
        runtime_weight,
        variant=variant,
        **kwargs,
    )
    second = msc.pointwise_1x1_metal(
        x,
        runtime_weight,
        variant=variant,
        **kwargs,
    )
    mx.eval(first, second)
    first_np = np.asarray(first)
    second_np = np.asarray(second)
    expected = msc.pointwise_1x1_numpy_fp32(x_np, expected_weight)
    np.testing.assert_allclose(first_np, expected, rtol=5e-3, atol=5e-3)
    assert np.array_equal(first_np, second_np)


@pytest.mark.parametrize("kernel,stride", [(3, 1), (5, 2)])
def test_depthwise_metal_matches_numpy_and_is_deterministic(kernel: int, stride: int) -> None:
    _require_metal()
    import mlx.core as mx

    rng = np.random.default_rng(103 + kernel)
    x_np = rng.uniform(-2, 2, size=(1, 9, 11, 7)).astype(np.float16)
    weight_np = rng.uniform(-1, 1, size=(7, kernel, kernel, 1)).astype(np.float16)
    x = mx.array(x_np, dtype=mx.float16)
    weight = mx.array(weight_np, dtype=mx.float16)
    first = msc.depthwise_conv2d_metal(
        x,
        weight,
        stride=stride,
        padding=kernel // 2,
    )
    second = msc.depthwise_conv2d_metal(
        x,
        weight,
        stride=stride,
        padding=kernel // 2,
    )
    mx.eval(first, second)
    first_np = np.asarray(first)
    second_np = np.asarray(second)
    expected = msc.depthwise_conv2d_numpy_fp32(
        x_np,
        weight_np,
        stride=stride,
        padding=kernel // 2,
    )
    np.testing.assert_allclose(first_np, expected, rtol=3e-4, atol=3e-4)
    assert np.array_equal(first_np, second_np)
