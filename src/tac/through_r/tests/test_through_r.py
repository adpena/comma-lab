# SPDX-License-Identifier: MIT
"""Tests for the canonical through-R harness (CANONICALIZATION UNIT 1, #388).

Coverage:
  * resolution_chain: constants vs upstream-read values; WH-vs-HW transposition guard;
    render_grid_to_camera_uint8 shape/dtype/range + camera passthrough + uniform-preserve;
    describe() provenance; verify_against_upstream fail-closed parity.
  * harness: n600 toy-refusal + subset-reason; backend-refusal (no proxy fallback);
    input-space normalization; load_gt_lstars; end-to-end gt_f1 -> lstars reproduction
    (d_seg == 0 by construction) + per-class sum identity + determinism (scorer-gated).
  * scaffold_assembler: no_offset == argmax; parity vs the inc1a delegation shim;
    Inc1aAssemblerError-is-ScaffoldAssemblerError alias; duplicate-home refusal; determinism.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tac.through_r import resolution_chain as rc
from tac.through_r.harness import (
    DEFAULT_GT_CACHE,
    ThroughRHarnessError,
    _to_camera_uint8_frames,
    load_frozen_segnet,
    load_gt_lstars,
    measure_through_r,
)
from tac.through_r.scaffold_assembler import (
    BC_MODES,
    Inc1aAssemblerError,
    ScaffoldAssemblerError,
    compose_partition,
)


# ---------------------------------------------------------------- resolution chain constants
def test_pinned_hw_wh_orderings_are_distinct_and_correct():
    # WH tuples (upstream constant-name convention).
    assert rc.CAMERA_SIZE_WH == (1164, 874)
    assert rc.SCORER_INPUT_SIZE_WH == (512, 384)
    # HW forms (the resize targets / array shapes).
    assert rc.CAMERA_HW == (874, 1164)
    assert rc.SEG_HW == (384, 512)
    assert (rc.CAMERA_H, rc.CAMERA_W) == (874, 1164)
    assert (rc.SEG_H, rc.SEG_W) == (384, 512)
    # the transposition guard: HW must NOT equal WH (crossing them is the bug class).
    assert rc.CAMERA_HW != rc.CAMERA_SIZE_WH
    assert rc.SEG_HW != rc.SCORER_INPUT_SIZE_WH
    assert rc.SEQ_LEN == 2


def test_lstars_grid_matches_seg_hw_convention():
    # The stored GT argmax is (N, 384, 512) = (N, SEG_H, SEG_W): confirms the HW convention.
    assert (rc.SEG_H, rc.SEG_W) == (384, 512)


def test_verify_against_upstream_matches_pins():
    try:
        up = rc.verify_against_upstream()
    except Exception as e:  # upstream import may be unavailable in some envs
        pytest.skip(f"upstream not importable: {e}")
    assert up["camera_size"] == rc.CAMERA_SIZE_WH
    assert up["segnet_model_input_size"] == rc.SCORER_INPUT_SIZE_WH
    assert up["seq_len"] == rc.SEQ_LEN


def test_verify_against_upstream_fail_closed_on_drift(monkeypatch):
    # If upstream ever re-pins, verification RAISES (loud), never drifts silently.
    monkeypatch.setattr(
        rc, "read_upstream_constants",
        lambda: {"camera_size": (999, 111), "segnet_model_input_size": (512, 384), "seq_len": 2},
    )
    with pytest.raises(rc.ResolutionChainError):
        rc.verify_against_upstream()


def test_describe_provenance_dump_has_chain_and_hazard():
    d = rc.describe()
    assert d["schema"] == "through_r_resolution_chain.v1"
    assert d["orderings"]["camera_hw"] == [874, 1164]
    assert d["orderings"]["seg_hw"] == [384, 512]
    assert any("bicubic" in step for step in d["chain"])
    assert "transposition_hazard" in d
    assert "camera_res_placement_149" in d


# ---------------------------------------------------------------- R operator (first half)
def test_render_grid_to_camera_uint8_shape_dtype_range():
    rng = np.random.default_rng(0)
    render = rng.uniform(0.0, 255.0, size=(48, 64, 3)).astype(np.float32)
    cam = rc.render_grid_to_camera_uint8(render)
    assert cam.shape == (rc.CAMERA_H, rc.CAMERA_W, 3)
    assert cam.dtype == np.uint8
    assert cam.min() >= 0 and cam.max() <= 255


def test_render_grid_camera_passthrough_is_identity():
    cam_in = np.full((rc.CAMERA_H, rc.CAMERA_W, 3), 123, dtype=np.uint8)
    cam_out = rc.render_grid_to_camera_uint8(cam_in)
    assert np.array_equal(cam_in, cam_out)


def test_render_grid_uniform_value_is_preserved():
    # A constant render grid resizes to a constant camera frame (bicubic of a constant).
    render = np.full((32, 40, 3), 100.0, dtype=np.float32)
    cam = rc.render_grid_to_camera_uint8(render)
    assert np.all(cam == 100)


def test_render_grid_bad_shape_raises():
    with pytest.raises(rc.ResolutionChainError):
        rc.render_grid_to_camera_uint8(np.zeros((10, 10), dtype=np.float32))
    with pytest.raises(rc.ResolutionChainError):
        rc.render_grid_to_camera_uint8(np.zeros((10, 10, 4), dtype=np.float32))


def test_contest_faithful_R_numpy_returns_scorer_res():
    render = np.full((1, 24, 32, 3), 128.0, dtype=np.float32)  # NHWC
    out = rc.contest_faithful_R_numpy(render)
    assert out.shape == (1, rc.SEG_H, rc.SEG_W, 3)
    assert np.isfinite(out).all()


# ---------------------------------------------------------------- harness refusal gates
def _tiny_camera_frames(n=3):
    return [np.zeros((rc.CAMERA_H, rc.CAMERA_W, 3), dtype=np.uint8) for _ in range(n)]


def test_measure_refuses_non_n600_without_reason():
    lstars = np.zeros((3, rc.SEG_H, rc.SEG_W), dtype=np.int64)
    with pytest.raises(ThroughRHarnessError):
        measure_through_r(_tiny_camera_frames(3), lstars=lstars, pairs="n600")


def test_measure_refuses_unknown_backend_before_scorer():
    lstars = np.zeros((3, rc.SEG_H, rc.SEG_W), dtype=np.int64)
    with pytest.raises(ThroughRHarnessError):
        measure_through_r(
            _tiny_camera_frames(3), lstars=lstars, backend="mlx",
            allow_subset_reason="dev",
        )


def test_measure_refuses_bad_pairs_protocol():
    lstars = np.zeros((3, rc.SEG_H, rc.SEG_W), dtype=np.int64)
    with pytest.raises(ThroughRHarnessError):
        measure_through_r(_tiny_camera_frames(3), lstars=lstars, pairs="n96")


def test_measure_refuses_empty():
    with pytest.raises(ThroughRHarnessError):
        measure_through_r([], lstars=np.zeros((0, rc.SEG_H, rc.SEG_W), np.int64))


def test_load_frozen_segnet_refuses_non_authority_backend():
    with pytest.raises(ThroughRHarnessError):
        load_frozen_segnet("mps")
    with pytest.raises(ThroughRHarnessError):
        load_frozen_segnet("mlx")


# ---------------------------------------------------------------- input-space normalization
def test_to_camera_uint8_camera_space_validates():
    frames = _tiny_camera_frames(2)
    out, resolved = _to_camera_uint8_frames(frames, "camera-uint8")
    assert resolved == "camera-uint8" and len(out) == 2
    assert all(f.shape == (rc.CAMERA_H, rc.CAMERA_W, 3) for f in out)


def test_to_camera_uint8_camera_space_rejects_render_grid():
    with pytest.raises(ThroughRHarnessError):
        _to_camera_uint8_frames([np.zeros((48, 64, 3), np.float32)], "camera-uint8")


def test_to_camera_uint8_render_grid_converts():
    out, resolved = _to_camera_uint8_frames([np.full((32, 40, 3), 50.0, np.float32)], "render-grid")
    assert resolved == "render-grid"
    assert out[0].shape == (rc.CAMERA_H, rc.CAMERA_W, 3) and out[0].dtype == np.uint8


def test_to_camera_uint8_auto_detects_mixed():
    frames = [
        np.zeros((rc.CAMERA_H, rc.CAMERA_W, 3), np.uint8),  # camera
        np.full((32, 40, 3), 10.0, np.float32),  # render grid
    ]
    out, resolved = _to_camera_uint8_frames(frames, "auto")
    assert resolved == "mixed" and len(out) == 2


def test_to_camera_uint8_bad_input_space_raises():
    with pytest.raises(ThroughRHarnessError):
        _to_camera_uint8_frames(_tiny_camera_frames(1), "whatever")


# ---------------------------------------------------------------- gt cache loader
def test_load_gt_lstars_missing_cache_raises():
    with pytest.raises(ThroughRHarnessError):
        load_gt_lstars("does/not/exist.npz")


def test_load_gt_lstars_present_cache_shape():
    if not os.path.exists(DEFAULT_GT_CACHE):
        pytest.skip("gt_n600 cache not present")
    ls = load_gt_lstars(DEFAULT_GT_CACHE, n=4)
    assert ls.shape == (4, rc.SEG_H, rc.SEG_W) and ls.dtype == np.int64


# ---------------------------------------------------------------- end-to-end (scorer-gated)
def _load_cache_or_skip(n=2):
    if not os.path.exists(DEFAULT_GT_CACHE):
        pytest.skip("gt_n600 cache not present")
    try:
        seg = load_frozen_segnet("cpu-torch")
    except Exception as e:  # upstream SegNet weights unavailable
        pytest.skip(f"frozen SegNet unavailable: {e}")
    z = np.load(DEFAULT_GT_CACHE, mmap_mode="r")
    f1 = [np.asarray(z["gt_f1"][i]) for i in range(n)]
    lstars = np.asarray(z["lstars"][:n]).astype(np.int64)
    return seg, f1, lstars


def test_end_to_end_gt_frame_reproduces_lstars_dseg_zero():
    # Feeding gt_f1 through the harness must reproduce lstars EXACTLY (they were computed by
    # this same SegNet on gt_f1) -> d_seg == 0 by construction. The strongest correctness test.
    seg, f1, lstars = _load_cache_or_skip(2)
    res = measure_through_r(
        f1, lstars=lstars, backend="cpu-torch", input_space="camera-uint8",
        allow_subset_reason="unit-test 2-frame reproduction", segnet=seg, verdict_batch=1,
    )
    assert res.agg_dseg == pytest.approx(0.0, abs=1e-9)
    assert res.total_flips == 0
    assert res.n_frames == 2 and not res.is_n600
    assert res.subset_reason.startswith("unit-test")
    assert res.backend == "cpu-torch" and res.input_space == "camera-uint8"
    # per-class sum identity holds by construction.
    assert res.total_flips / res.total_pixels == pytest.approx(res.agg_dseg)


def test_end_to_end_chunking_is_bit_identical():
    seg, f1, lstars = _load_cache_or_skip(2)
    a = measure_through_r(
        f1, lstars=lstars, input_space="camera-uint8", allow_subset_reason="chunk A",
        segnet=seg, verdict_batch=1, return_realized=True,
    )
    b = measure_through_r(
        f1, lstars=lstars, input_space="camera-uint8", allow_subset_reason="chunk B",
        segnet=seg, verdict_batch=0, return_realized=True,  # single-batch
    )
    assert np.array_equal(a.realized, b.realized)
    assert a.agg_dseg == pytest.approx(b.agg_dseg, abs=1e-12)


# ---------------------------------------------------------------- scaffold assembler
H, W, K = 12, 16, 5


def _phi(seed=0):
    return np.random.default_rng(seed).standard_normal((H, W, K)).astype(np.float32)


def test_no_offset_equals_argmax_of_stack():
    phi = _phi(1)
    res = compose_partition(phi_hwk=phi, bc_mode="no_offset")
    assert np.array_equal(res.partition, np.argmax(phi, axis=-1))
    assert res.reconcile_iters == 0


def test_assembler_parity_vs_inc1a_shim():
    # The inc1a shim must produce byte-identical partitions to the canonical through_r home.
    from tac.inc1a_harness.composite_assembler import compose_partition as inc1a_compose

    phi = _phi(2)
    a = compose_partition(phi_hwk=phi, bc_mode="no_offset").partition
    b = inc1a_compose(phi_hwk=phi, bc_mode="no_offset").partition
    assert np.array_equal(a, b)


def test_inc1a_error_is_scaffold_error_alias():
    from tac.inc1a_harness.composite_assembler import Inc1aAssemblerError as ShimErr

    assert Inc1aAssemblerError is ScaffoldAssemblerError
    assert ShimErr is ScaffoldAssemblerError


def test_duplicate_geometric_home_raises():
    from tac.through_r.scaffold_assembler import CarrierField, assemble_fields

    f = np.zeros((H, W), np.float32)
    carriers = [
        CarrierField(class_id=1, phi_hw=f, name="a", geometric_home="G2"),
        CarrierField(class_id=1, phi_hw=f, name="b", geometric_home="G2"),
    ]
    with pytest.raises(ScaffoldAssemblerError):
        assemble_fields(carriers, shape=(H, W), complement_class=0)


def test_bc_modes_include_386_dispatcher():
    for m in ("no_offset", "menon", "ot_newton", "flip_weighted", "flip_median"):
        assert m in BC_MODES


def test_assembler_is_deterministic():
    phi = _phi(3)
    a = compose_partition(phi_hwk=phi, bc_mode="no_offset").partition
    b = compose_partition(phi_hwk=phi, bc_mode="no_offset").partition
    assert np.array_equal(a, b)
