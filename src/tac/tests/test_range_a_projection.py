# SPDX-License-Identifier: MIT
"""Tests for tac.boundary_math.range_a_projection (#520, SPEC_v10 P3).

Behavior-verifying (not constant-checking): the projector's EXACTNESS vs the #519 reference,
idempotence, shape handling, MLX-numpy parity, the guarded hook's default-off byte-identity +
fail-closed, and the DSL lever registry holding the flag.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math import range_a_projection as rap


def test_self_test_matches_519_reference():
    st = rap.projector_self_test()
    # EXACT projector: A(X - PX) == 0 to fp64 round-off (the validated #519 residual).
    assert st["max_A_of_ker_residual"] <= rap.KER_RESIDUAL_CEILING
    assert st["matches_reference"] is True
    # reproduce the #519 residual to within an order of magnitude of the recorded constant
    assert st["max_A_of_ker_residual"] < 1e-13
    assert st["idempotence_residual"] < 1e-11
    # exact blind row/col counts (blind_coordinate 106/874, 140/1164).
    assert st["n_exact_blind_rows"] == 106
    assert st["n_exact_blind_cols"] == 140
    # a range(A^T) element is fully seen; a blind-row image is fully blind.
    assert abs(st["seen_space_blind_fraction"]) < 1e-10
    assert st["blind_row_image_blind_fraction"] == pytest.approx(1.0, abs=1e-10)


def test_exactness_A_of_ker_residual_direct():
    # Directly assert A(X - P(X)) ~ 0 on a random camera frame (the defining property).
    from tac.through_r.flip_inverse import resize_matrix_1d
    dr = resize_matrix_1d(rap.CAMERA_H, rap.SEG_H, "bilinear", align_corners=False,
                          antialias=False, dtype=np.float64)
    dc = resize_matrix_1d(rap.CAMERA_W, rap.SEG_W, "bilinear", align_corners=False,
                          antialias=False, dtype=np.float64)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((rap.CAMERA_H, rap.CAMERA_W))
    px = rap.apply_projection(x, out_dtype=np.float64, compute_dtype=np.float64)
    resid = float(np.abs(dr @ (x - px) @ dc.T).max())
    assert resid < rap.KER_RESIDUAL_CEILING


@pytest.mark.parametrize("shape", [
    (874, 1164),
    (874, 1164, 3),
    (2, 874, 1164, 3),
    (3, 874, 1164),
])
def test_shape_preserved_and_idempotent(shape):
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(shape) * 40 + 128).astype(np.float32)
    p = rap.apply_projection(x)
    assert p.shape == x.shape
    assert p.dtype == np.float32
    # idempotence: P(P(x)) == P(x) to fp32 round-off
    pp = rap.apply_projection(p)
    assert float(np.abs(pp - p).max()) < 1e-3


def test_projection_removes_ker_but_keeps_seen():
    # A range(A^T) frame must be (near) UNCHANGED by the projector.
    from tac.through_r.flip_inverse import resize_matrix_1d
    dr = resize_matrix_1d(rap.CAMERA_H, rap.SEG_H, "bilinear", align_corners=False,
                          antialias=False, dtype=np.float64)
    dc = resize_matrix_1d(rap.CAMERA_W, rap.SEG_W, "bilinear", align_corners=False,
                          antialias=False, dtype=np.float64)
    rng = np.random.default_rng(7)
    seen = dr.T @ rng.standard_normal((rap.SEG_H, rap.SEG_W)) @ dc  # in range(A^T)
    p = rap.apply_projection(seen, out_dtype=np.float64, compute_dtype=np.float64)
    assert float(np.abs(p - seen).max()) < 1e-9


def test_bad_shape_raises():
    with pytest.raises(rap.RangeAProjectionError):
        rap.apply_projection(np.zeros((100, 100), dtype=np.float32))


def test_mlx_numpy_parity():
    pytest.importorskip("mlx.core")
    rng = np.random.default_rng(11)
    x = (rng.standard_normal((874, 1164, 3)) * 40 + 128).astype(np.float32)
    p_np = rap.apply_projection(x, compute_dtype=np.float32)
    p_mx = np.array(rap.apply_projection_mlx(x))
    # fp32 einsum accumulation differs between backends; parity is relative (MLX is a
    # non-authority training-time twin — the numpy fp64 path is the authority).
    rel = float(np.abs(p_mx - p_np).max() / (np.abs(p_np).max() + 1e-9))
    assert rel < 1.5e-2


def test_hook_default_off_returns_same_object():
    x = np.zeros((874, 1164, 3), dtype=np.float32)
    out = rap.maybe_project_render_target(x, enabled=False)
    assert out is x  # byte-identity by construction


def test_hook_enabled_projects():
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((874, 1164, 3)) * 40 + 128).astype(np.float32)
    out = rap.maybe_project_render_target(x, enabled=True, cadence="post_render")
    assert out.shape == x.shape
    # a projection actually changed the (non-range(A)) input
    assert float(np.abs(out - x).max()) > 1e-3


def test_hook_bad_cadence_raises_when_enabled():
    x = np.zeros((874, 1164, 3), dtype=np.float32)
    with pytest.raises(rap.RangeAProjectionError):
        rap.maybe_project_render_target(x, enabled=True, cadence="nope")
    # but a bad cadence is IGNORED when disabled (no-op path)
    assert rap.maybe_project_render_target(x, enabled=False, cadence="nope") is x


def test_hook_bad_backend_raises_when_enabled():
    x = np.zeros((874, 1164, 3), dtype=np.float32)
    with pytest.raises(rap.RangeAProjectionError):
        rap.maybe_project_render_target(x, enabled=True, backend="jax")


def test_dsl_lever_holds_flag():
    from tac.witness_dsl.curriculum_dsl import RangeAProjection
    lv = RangeAProjection()
    assert lv.overrides["--range-a-projection"] is True
    assert lv.overrides["--range-a-projection-cadence"] == "post_render"
    assert RangeAProjection("every_step").overrides["--range-a-projection-cadence"] == "every_step"
    with pytest.raises(ValueError):
        RangeAProjection("bad")


def test_lever_registry_maps_the_flags():
    from tac.witness_dsl.lever_registry import completeness
    c = completeness()
    for flag in ("--range-a-projection", "--range-a-projection-cadence"):
        assert flag in c.mapped, f"{flag} should be DSL-mapped"
        assert flag not in c.unmapped
        assert flag not in c.stale
