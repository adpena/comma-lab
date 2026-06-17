# SPDX-License-Identifier: MIT
"""NO-FAKE behavior tests for the partition-store realization gate (Phase 1).

The realization gate's entire purpose is to MEASURE the real realized d_seg of a
painted partition through the exact SegNet eval chain — never assume d_seg=0.  These
tests verify BEHAVIOR not constants:

  - the pure realization helpers (upsample / paint / boundary / dilate / d_seg) do
    what their names claim on hand-checkable inputs;
  - ``d_seg`` is the real argmax-flip rate (a no-op compare FAILS the test);
  - the exact-chain forward ``_segnet_argmax_camera`` is wired to the REAL SegNet
    when the checkpoint is present, returns the (384,512) argmax, and a perturbed
    frame can produce a DIFFERENT argmax (it is not a constant) — the realized d_seg
    is genuinely measured, not hard-coded to 0.

The heavy SegNet integration test is skipped when the checkpoint / torch is absent
so the suite stays runnable in CI without contest assets, but it is NOT a synthetic
fixture: when present it runs the real frozen scorer through the exact chain.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))

# Load the experiment module by path (it lives in experiments/, not a package).
_EXP_PATH = REPO / "experiments" / "partition_store_realization_gate.py"
_spec = importlib.util.spec_from_file_location("_psr_gate", _EXP_PATH)
psr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(psr)


# ---------------------------------------------------------------------------
# Pure-numpy realization helpers — hand-checkable behavior
# ---------------------------------------------------------------------------
def test_d_seg_is_real_flip_rate_not_noop():
    a = np.array([[0, 1], [2, 3]], dtype=np.int64)
    b = np.array([[0, 1], [2, 4]], dtype=np.int64)  # one differing pixel of 4
    assert psr.d_seg(a, b) == pytest.approx(0.25)
    assert psr.d_seg(a, a) == 0.0
    # a no-op compare (always-equal) would report 0 here — this catches that fake.
    c = np.zeros_like(a)
    assert psr.d_seg(a, c) > 0.0


def test_upsample_labels_nearest_preserves_classes_and_shape():
    lab = np.array([[0, 1], [2, 3]], dtype=np.int64)
    up = psr._upsample_labels_nearest(lab, 4, 4)
    assert up.shape == (4, 4)
    # nearest upsample must not invent classes outside the source alphabet.
    assert set(np.unique(up).tolist()).issubset({0, 1, 2, 3})
    # top-left quadrant maps to source[0,0]=0.
    assert up[0, 0] == 0
    assert up[-1, -1] == 3


def test_paint_flat_assigns_canonical_color_per_class():
    label = np.array([[0, 1], [1, 0]], dtype=np.int64)
    mu = np.array([[10.0, 20.0, 30.0], [200.0, 100.0, 50.0]], dtype=np.float64)
    frame = psr._paint_flat(label, mu)
    assert frame.dtype == np.uint8
    assert frame.shape == (2, 2, 3)
    assert tuple(frame[0, 0]) == (10, 20, 30)  # class 0
    assert tuple(frame[0, 1]) == (200, 100, 50)  # class 1


def test_boundary_mask_marks_class_transitions():
    # a vertical seam between class 0 (left) and class 1 (right).
    lab = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int64)
    bm = psr._boundary_mask_seg(lab)
    # the two columns straddling the seam are boundary; the outer columns are not.
    assert bm[:, 1].all() and bm[:, 2].all()
    assert not bm[:, 0].any() and not bm[:, 3].any()
    # a uniform map has NO boundary.
    assert not psr._boundary_mask_seg(np.zeros((3, 3), dtype=np.int64)).any()


def test_dilate_majority_idempotent_on_clean_blocks_but_smooths_specks():
    # clean 2x2 blocks: majority(incl self) leaves them unchanged (honest no-op).
    clean = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int64)
    assert np.array_equal(psr._dilate_majority(clean, 1, 5), clean)
    # an isolated speck inside a region IS removed by majority smoothing.
    speck = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int64
    )
    out = psr._dilate_majority(speck, 1, 5)
    assert out[1, 1] == 0  # the lone class-1 pixel is outvoted by class-0 neighbours


def test_dilate_majority_zero_iters_is_identity():
    lab = np.array([[0, 1], [2, 3]], dtype=np.int64)
    assert np.array_equal(psr._dilate_majority(lab, 0, 5), lab)


# ---------------------------------------------------------------------------
# Real eval-chain integration — the NO-FAKE realized-d_seg proof
# ---------------------------------------------------------------------------
def _segnet_or_skip():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not available")
    ckpt = REPO / "upstream" / "models" / "segnet.safetensors"
    if not ckpt.exists():
        pytest.skip("contest SegNet checkpoint not present")
    from tac.boundary_math.seg_core import load_real_segnet

    return load_real_segnet("cpu")


def test_segnet_argmax_camera_returns_seg_grid_argmax():
    seg = _segnet_or_skip()
    frame = np.full((psr.CAMERA_H, psr.CAMERA_W, 3), 100, dtype=np.uint8)
    am = psr._segnet_argmax_camera(seg, frame)
    assert am.shape == (psr.SEG_H, psr.SEG_W)  # exact eval grid 384x512
    assert am.min() >= 0 and am.max() < psr.N_CLASSES


def test_realized_dseg_is_measured_not_assumed_zero():
    """The core NO-FAKE proof: painting the stored partition and running it through
    the EXACT chain produces a NON-ZERO realized d_seg (the store does NOT trivially
    realize at d_seg=0).  A fake that returned 0 by construction would fail here."""
    seg = _segnet_or_skip()
    # Build a stored partition L* from a real SegNet forward on a textured frame,
    # then realize it with a flat per-class color and measure realized d_seg.
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(psr.CAMERA_H, psr.CAMERA_W, 3), dtype=np.uint8)
    lstar = psr._segnet_argmax_camera(seg, base)  # (384,512) stored target
    # canonical color = per-class mean of the base frame (camera res argmax).
    am_cam = psr._upsample_labels_nearest(lstar, psr.CAMERA_H, psr.CAMERA_W)
    mu = np.zeros((psr.N_CLASSES, 3))
    for c in range(psr.N_CLASSES):
        m = am_cam == c
        mu[c] = base[m].mean(axis=0) if m.any() else 128.0
    painted = psr._paint_flat(am_cam, mu)
    realized = psr._segnet_argmax_camera(seg, painted)
    dd = psr.d_seg(realized, lstar)
    # The realization is lossy through resize+SegNet: realized d_seg is > 0.  If this
    # were 0 the operator's "interiors survive trivially" hypothesis would hold; the
    # gate exists precisely because it does NOT.  This is the core NO-FAKE proof: the
    # realized d_seg is genuinely measured through the chain (not hard-coded to 0).
    assert 0.0 < dd <= 1.0
    # And the realized argmax must NOT be bit-identical to the stored L* (proves the
    # measure is a real comparison through the resize/SegNet chain, not a tautology).
    assert not np.array_equal(realized, lstar)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
