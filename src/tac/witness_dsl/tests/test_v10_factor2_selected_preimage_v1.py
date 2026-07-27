from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    realize_factor2_uint8_scorer_plane,
)
from tac.witness_dsl.v10_factor2_selected_preimage_v1 import (
    CAMERA_HW,
    SCORER_HW,
    V10Factor2SelectedPreimageError,
    build_mlx_factor2_gather_plan,
    build_numpy_factor2_gather_plan,
    realize_factor2_scorer_plane_mlx,
    realize_factor2_uint8_numpy,
)


def _structured_target() -> np.ndarray:
    rows = np.arange(SCORER_HW[0], dtype=np.uint16)[:, None]
    columns = np.arange(SCORER_HW[1], dtype=np.uint16)[None, :]
    return np.stack(
        (
            np.broadcast_to((rows * 17 + columns * 3) & 255, SCORER_HW),
            np.broadcast_to((rows * 5 + columns * 29) & 255, SCORER_HW),
            np.broadcast_to((rows * 31 + columns * 7) & 255, SCORER_HW),
        ),
        axis=2,
    ).astype(np.uint8)


def test_numpy_realization_is_the_certified_public_factor2_point() -> None:
    target = _structured_target()
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_HW[0],
        camera_w=CAMERA_HW[1],
        scorer_h=SCORER_HW[0],
        scorer_w=SCORER_HW[1],
    )
    expected = realize_factor2_uint8_scorer_plane(operator, target)
    actual = realize_factor2_uint8_numpy(target)
    assert np.array_equal(actual, expected)
    verification = operator.verify_factor2_uint8(actual, target)
    assert verification.certified_exact
    assert verification.numerator_exact


def test_numpy_gather_plan_owns_exactly_four_taps_per_scorer_sample() -> None:
    _operator, indices, valid = build_numpy_factor2_gather_plan()
    assert indices.dtype == np.int32
    assert valid.dtype == np.bool_
    assert int(np.count_nonzero(valid)) == SCORER_HW[0] * SCORER_HW[1] * 4
    counts = np.bincount(indices[valid], minlength=SCORER_HW[0] * SCORER_HW[1])
    assert np.array_equal(counts, np.full(counts.shape, 4, dtype=counts.dtype))


def test_numpy_realization_rejects_non_uint8_boundary() -> None:
    with pytest.raises(V10Factor2SelectedPreimageError, match="uint8"):
        realize_factor2_uint8_numpy(np.zeros((*SCORER_HW, 3), dtype=np.float32))


def test_mlx_forward_is_byte_exact_and_gather_is_differentiable() -> None:
    mx = pytest.importorskip("mlx.core")
    target = _structured_target()
    plan = build_mlx_factor2_gather_plan(mlx_module=mx)
    source = mx.array(target.astype(np.float32))
    camera = realize_factor2_scorer_plane_mlx(
        source,
        mlx_module=mx,
        plan=plan,
        ste_round=True,
    )
    mx.eval(camera)
    host = np.asarray(camera)
    assert host.dtype == np.float32
    assert np.array_equal(host.astype(np.uint8), realize_factor2_uint8_numpy(target))

    gradient = mx.grad(
        lambda value: mx.sum(
            realize_factor2_scorer_plane_mlx(
                value,
                mlx_module=mx,
                plan=plan,
                ste_round=True,
            )
        )
    )(source)
    mx.eval(gradient)
    assert np.array_equal(
        np.asarray(gradient),
        np.full((*SCORER_HW, 3), 4.0, dtype=np.float32),
    )


def test_public_receiver_factor2_function_matches_canonical_helper() -> None:
    path = (
        Path(__file__).resolve().parents[4]
        / "submissions"
        / "robust_current"
        / "g110_two_layer_receiver"
        / "inflate.py"
    )
    spec = importlib.util.spec_from_file_location("_g110_public_inflate_parity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = _structured_target()
    assert np.array_equal(module._realize_factor2(target), realize_factor2_uint8_numpy(target))
