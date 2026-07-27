from __future__ import annotations

import numpy as np
import pytest

from tac.witness_dsl import g105_public_wire_quantization_surface_v1 as subject
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    ExactV9SemanticRootError,
    V9PolarFourierConfigV1,
    V9RuntimeConfigV1,
    V9TensorDTypeV1,
    Y1WireCodecV1,
)


def _production_state() -> tuple[
    V9RuntimeConfigV1,
    dict[str, np.ndarray],
    np.ndarray,
]:
    """Real G111/G105 n600 tensor census: 71,159 model values plus Y1[600,32]."""

    basis = V9PolarFourierConfigV1(
        n_scales=4,
        n_orient0=6,
        f0=2.0,
        base=2.0,
        n_iso=4,
        max_freq=64.0,
    )
    config = V9RuntimeConfigV1(
        input_dim=basis.input_dim,
        hidden_dim=96,
        hidden_layer_count=4,
        modulation_dim=32,
        softmax_temp=0.31,
        hosc_beta=3.177,
        hosc_omega=1.0,
        chroma=True,
        film_per_layer=False,
        film_concat_code=False,
        basis=basis,
    )
    rng = np.random.default_rng(115)

    def values(shape: tuple[int, ...], scale: float) -> np.ndarray:
        return rng.normal(0.0, scale, size=shape).astype(np.float32)

    params = {
        "in_proj.weight": values((96, config.input_dim), 0.08),
        "in_proj.bias": values((96,), 0.03),
        "film.weight": values((2 * 96 * 4, 32), 0.02),
        "film.bias": values((2 * 96 * 4,), 0.02),
        "out_sdf.weight": values((5, 96), 0.08),
        "out_sdf.bias": values((5,), 0.03),
        "out_tex.weight": values((3, 96), 0.08),
        "out_tex.bias": np.zeros((3,), dtype=np.float32),
        "palette": values((5, 3), 0.25),
    }
    for layer_index in range(4):
        params[f"hidden.{layer_index}.weight"] = values((96, 96), 0.05)
        params[f"hidden.{layer_index}.bias"] = values((96,), 0.02)
    y1 = values((600, 32), 0.12)
    return config, params, y1


def test_numpy_authority_is_production_shape_exact_g105_roundtrip() -> None:
    config, params, y1 = _production_state()
    first = subject.compile_g105_public_wire_quantization_surface_numpy(
        config=config,
        params=params,
        y1_code=y1,
    )
    second = subject.compile_g105_public_wire_quantization_surface_numpy(
        config=config,
        params=params,
        y1_code=y1,
    )

    assert config.input_dim == 80
    assert sum(int(np.prod(plan.shape)) for plan in first.receipt.tensor_plans) == 71_159
    assert first.receipt.y1_plan.shape == (600, 32)
    assert first.receipt.y1_plan.quantized_bytes == 38_400
    assert first.receipt.packet_bytes == 111_840
    assert first.packet == second.packet
    assert first.receipt.to_dict() == second.receipt.to_dict()
    assert first.receipt.receipt_sha256 == second.receipt.receipt_sha256
    assert first.receipt.parse_reencode_identical is True
    assert first.receipt.to_dict()["authority"] == subject.AUTHORITY
    assert first.receipt.to_dict()["candidate_or_score_claim"] is False
    assert first.receipt.to_dict()["pointer_moved"] is False

    for plan in first.receipt.tensor_plans:
        if plan.wire_name.endswith(".weight"):
            assert plan.dtype is V9TensorDTypeV1.INT8
            assert plan.symmetric_limit == 127
        else:
            assert plan.dtype is V9TensorDTypeV1.INT16_LE
            assert plan.symmetric_limit == 32767
        assert -64 <= plan.scale_exponent <= 63
    assert first.receipt.y1_plan.dtype is V9TensorDTypeV1.INT16_LE
    assert first.receipt.y1_plan.symmetric_limit == 32767
    assert next(plan for plan in first.receipt.tensor_plans if plan.wire_name == "out_tex.bias").scale_exponent == -32
    assert next(plan for plan in first.receipt.tensor_plans if plan.wire_name == "hidden.weight").source_keys == tuple(
        f"hidden.{index}.weight" for index in range(4)
    )


@pytest.mark.parametrize(
    "codec",
    [Y1WireCodecV1.RAW_I16_LE, Y1WireCodecV1.DELTA_RICE_BEST_K],
)
def test_mlx_ste_forward_is_bit_exact_parsed_dequant_and_gradient_is_live(
    codec: Y1WireCodecV1,
) -> None:
    mx = pytest.importorskip("mlx.core")
    config, params_np, y1_np = _production_state()
    params_mx = {name: mx.array(values, dtype=mx.float32) for name, values in params_np.items()}
    y1_mx = mx.array(y1_np, dtype=mx.float32)

    mlx_state = subject.g105_public_wire_quantize_ste_mlx(
        config=config,
        params=params_mx,
        y1_code=y1_mx,
        y1_wire_codec=codec,
    )
    numpy_authority = subject.compile_g105_public_wire_quantization_surface_numpy(
        config=config,
        params=params_np,
        y1_code=y1_np,
        y1_wire_codec=codec,
    )
    mx.eval(*mlx_state.params.values(), mlx_state.y1_code)
    assert mlx_state.receipt.to_dict() == numpy_authority.receipt.to_dict()
    for name, expected in numpy_authority.params.items():
        assert np.array_equal(np.asarray(mlx_state.params[name]), expected)
    assert np.array_equal(np.asarray(mlx_state.y1_code), numpy_authority.y1_code)

    differentiable = {
        "in_proj.weight": params_mx["in_proj.weight"],
        "in_proj.bias": params_mx["in_proj.bias"],
        "palette": params_mx["palette"],
        "y1_code": y1_mx,
    }

    def loss(selected):
        current = dict(params_mx)
        current["in_proj.weight"] = selected["in_proj.weight"]
        current["in_proj.bias"] = selected["in_proj.bias"]
        current["palette"] = selected["palette"]
        quantized = subject.g105_public_wire_quantize_ste_mlx(
            config=config,
            params=current,
            y1_code=selected["y1_code"],
            y1_wire_codec=codec,
        )
        return (
            mx.sum(quantized.params["in_proj.weight"])
            + mx.sum(quantized.params["in_proj.bias"])
            + mx.sum(quantized.params["palette"])
            + mx.sum(quantized.y1_code)
        )

    gradients = mx.grad(loss)(differentiable)
    mx.eval(*gradients.values())
    for name, gradient in gradients.items():
        observed = np.asarray(gradient)
        assert observed.shape == np.asarray(differentiable[name]).shape
        assert np.array_equal(observed, np.ones_like(observed))


def test_canonical_tie_rounding_and_symmetric_ranges_survive_mlx_ste() -> None:
    mx = pytest.importorskip("mlx.core")
    config, params, y1 = _production_state()
    params["in_proj.weight"].flat[:5] = np.array(
        [127.0, 0.5, 1.5, 2.5, -1.5],
        dtype=np.float32,
    )
    params["in_proj.bias"].flat[:5] = np.array(
        [32767.0, 0.5, 1.5, 2.5, -1.5],
        dtype=np.float32,
    )
    params["palette"].flat[:5] = np.array(
        [32767.0, 0.5, 1.5, 2.5, -1.5],
        dtype=np.float32,
    )
    y1.flat[:5] = np.array(
        [32767.0, 0.5, 1.5, 2.5, -1.5],
        dtype=np.float32,
    )
    authority = subject.compile_g105_public_wire_quantization_surface_numpy(
        config=config,
        params=params,
        y1_code=y1,
    )
    mlx_state = subject.g105_public_wire_quantize_ste_mlx(
        config=config,
        params={name: mx.array(values, dtype=mx.float32) for name, values in params.items()},
        y1_code=mx.array(y1, dtype=mx.float32),
    )
    mx.eval(*mlx_state.params.values(), mlx_state.y1_code)

    expected_ties = np.array([127.0, 0.0, 2.0, 2.0, -2.0], dtype=np.float32)
    assert np.array_equal(authority.params["in_proj.weight"].flat[:5], expected_ties)
    assert np.array_equal(
        authority.params["in_proj.bias"].flat[:5],
        np.array(
            [32767.0, 0.0, 2.0, 2.0, -2.0],
            dtype=np.float32,
        ),
    )
    assert np.array_equal(
        authority.params["palette"].flat[:5],
        np.array(
            [32767.0, 0.0, 2.0, 2.0, -2.0],
            dtype=np.float32,
        ),
    )
    assert np.array_equal(
        authority.y1_code.flat[:5],
        np.array(
            [32767.0, 0.0, 2.0, 2.0, -2.0],
            dtype=np.float32,
        ),
    )
    assert np.array_equal(
        np.asarray(mlx_state.params["in_proj.weight"]).flat[:5],
        authority.params["in_proj.weight"].flat[:5],
    )
    assert np.array_equal(
        np.asarray(mlx_state.params["palette"]).flat[:5],
        authority.params["palette"].flat[:5],
    )
    assert np.array_equal(
        np.asarray(mlx_state.y1_code).flat[:5],
        authority.y1_code.flat[:5],
    )


def test_full_production_census_refuses_noncanonical_range_and_extra_state() -> None:
    config, params, y1 = _production_state()
    with_extra = dict(params)
    with_extra["pose_carrier.dxi"] = np.zeros((600, 6), dtype=np.float32)
    with pytest.raises(
        subject.G105PublicWireQuantizationError,
        match=r"extra=.*pose_carrier",
    ):
        subject.compile_g105_public_wire_quantization_surface_numpy(
            config=config,
            params=with_extra,
            y1_code=y1,
        )

    too_large = dict(params)
    too_large["in_proj.weight"] = params["in_proj.weight"].copy()
    too_large["in_proj.weight"].flat[0] = np.finfo(np.float32).max
    with pytest.raises(
        ExactV9SemanticRootError,
        match="cannot be represented",
    ):
        subject.compile_g105_public_wire_quantization_surface_numpy(
            config=config,
            params=too_large,
            y1_code=y1,
        )
