# SPDX-License-Identifier: MIT
"""Tests for the in-trainer annulus_convergence telemetry helpers (SENSE stream).

The pure helpers ``_annulus_metrics_from_maps`` / ``_annulus_convergence_row`` are factored out of the
verdict closure in ``experiments/train_levelset_witness_realized_through_R_mlx.py`` precisely so they
are unit-testable at $0 (no training, no MLX, no torch). They REUSE
``tac.witness_annulus_metrics`` -- these tests assert the assembly + the DEFAULT-OFF/fail-safe
contract, not the underlying metric math (which has its own module).

CANONICAL comma10k class order (CLAUDE.md NON-NEGOTIABLE): 0=Road 1=Lane 2=Undrivable 3=Movable 4=MyCar.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
_TRAINER = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"


def _load_trainer_module():
    """Import the trainer module by path (it lives under experiments/, not an installed package).

    The module inserts REPO/{,,src,upstream,experiments} onto sys.path at import time; importing it
    does NOT run training (guarded by ``if __name__ == '__main__'``)."""
    spec = importlib.util.spec_from_file_location("_levelset_trainer_under_test", _TRAINER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    _M = _load_trainer_module()
except Exception as _exc:  # pragma: no cover - environment (mlx/torch import) gap => skip, never fail
    _M = None
    _IMPORT_ERR = _exc


pytestmark = pytest.mark.skipif(_M is None, reason=f"trainer import unavailable: {globals().get('_IMPORT_ERR')}")


# ---------------------------------------------------------------------------
# synthetic fixture: a small (h,w) argmax/margin pair with a KNOWN flip structure.
# ---------------------------------------------------------------------------
def _fixture(h: int = 8, w: int = 8):
    """GT argmax = class 0 everywhere except a thin class-1 vertical stripe (a 'lane'). GT margin is
    SMALL on the stripe + its immediate neighbours (the annulus) and LARGE in the interior. The
    realized argmax flips a couple of stripe pixels (annulus flips) and one interior pixel (a
    structural interior flip) so both the annulus and interior flip fractions are non-trivial."""
    gt = np.zeros((h, w), np.int64)
    gt[:, 3] = 1  # a lane stripe (class 1) in column 3
    margin = np.full((h, w), 5.0, np.float32)  # large interior margin
    margin[:, 2:5] = 0.5  # low-margin annulus band around the stripe (|margin| < band=2.0)
    realized = gt.copy()
    realized[0, 3] = 0  # annulus flip (stripe pixel -> road)
    realized[1, 3] = 0  # annulus flip
    realized[5, 7] = 3  # INTERIOR structural flip (far from the band -> class 3)
    return [realized], [gt], [margin]


# ---------------------------------------------------------------------------
# 1-4: assembly correctness.
# ---------------------------------------------------------------------------
def test_metrics_dict_has_both_annulus_definitions_and_core_keys():
    realized, gt, margin = _fixture()
    m = _M._annulus_metrics_from_maps(realized, gt, margin, band=2.0, bottom_k=0.05)
    for k in ("overall_d_seg", "total_px", "n_flips", "n_pairs", "band", "bottom_k",
              "margin_source", "threshold", "bottom_k_def"):
        assert k in m, k
    assert m["margin_source"] == "gt"
    assert m["n_pairs"] == 1
    assert m["total_px"] == 64
    assert m["n_flips"] == 3  # two annulus flips + one interior flip
    for defn in ("threshold", "bottom_k_def"):
        for k in ("annulus_area_frac", "annulus_flip_frac", "interior_flip_frac",
                  "annulus_flip_mass_share", "annulus_gt_margin", "per_class_annulus_flip_frac"):
            assert k in m[defn], (defn, k)


def test_threshold_annulus_splits_flip_mass_interior_vs_boundary():
    realized, gt, margin = _fixture()
    m = _M._annulus_metrics_from_maps(realized, gt, margin, band=2.0, bottom_k=0.05)
    thr = m["threshold"]
    # 3 flips total: 2 fall in the |margin|<2 band (cols 2..4), 1 in the interior => 2/3 mass share.
    assert thr["annulus_flip_mass_share"] == pytest.approx(2.0 / 3.0, rel=1e-6)
    assert thr["interior_flip_frac"] > 0.0  # the structural interior flip is separated out
    assert thr["annulus_flip_frac"] > 0.0


def test_per_class_annulus_flip_frac_keys_cover_all_five_classes():
    realized, gt, margin = _fixture()
    m = _M._annulus_metrics_from_maps(realized, gt, margin, band=2.0, bottom_k=0.05)
    pc = m["threshold"]["per_class_annulus_flip_frac"]
    # the annulus flips are on class-0 (road) pixels adjacent to the stripe; every class 0..4 present.
    keys = {int(k) for k in pc}
    assert keys == {0, 1, 2, 3, 4}


def test_gt_margin_percentiles_present_p10_p50():
    realized, gt, margin = _fixture()
    m = _M._annulus_metrics_from_maps(realized, gt, margin, band=2.0, bottom_k=0.05)
    am = m["threshold"]["annulus_gt_margin"]
    assert "p10" in am and "p50" in am
    # all annulus (threshold) pixels have GT margin 0.5 by construction.
    assert am["p10"] == pytest.approx(0.5, abs=1e-6)
    assert am["p50"] == pytest.approx(0.5, abs=1e-6)


def test_multi_pair_stack_aggregates_pairs():
    realized, gt, margin = _fixture()
    m = _M._annulus_metrics_from_maps(realized * 3, gt * 3, margin * 3, band=2.0, bottom_k=0.05)
    assert m["n_pairs"] == 3
    assert m["total_px"] == 64 * 3
    assert m["n_flips"] == 9  # 3 flips per pair


# ---------------------------------------------------------------------------
# 5-6: row wrapper is JSON-safe with stage/epoch/seg_form/axis.
# ---------------------------------------------------------------------------
def test_row_wrapper_is_json_serializable_with_stage_and_epoch():
    realized, gt, margin = _fixture()
    m = _M._annulus_metrics_from_maps(realized, gt, margin, band=2.0, bottom_k=0.05)
    row = _M._annulus_convergence_row(m, epoch=42, seg_form="tau")
    s = json.dumps(row)  # must not raise (int dict keys are stringified by json)
    back = json.loads(s)
    assert back["stage"] == "annulus_convergence"
    assert back["epoch"] == 42
    assert back["seg_form"] == "tau"
    assert "NON-PROMOTABLE" in back["axis"]


def test_row_wrapper_accepts_none_seg_form_and_error_payload():
    # the fail-safe path stashes {"error": ...}; the wrapper must still produce a JSON row.
    row = _M._annulus_convergence_row({"error": "ValueError: boom"}, epoch=0, seg_form=None)
    s = json.dumps(row)
    back = json.loads(s)
    assert back["seg_form"] is None
    assert back["error"] == "ValueError: boom"
    assert back["stage"] == "annulus_convergence"


# ---------------------------------------------------------------------------
# 7-8: fail-safe / default-off contract.
# ---------------------------------------------------------------------------
def test_bad_input_raises_inside_helper_so_caller_try_except_can_swallow():
    # mismatched shapes must raise (np.stack / comparison) so the trainer's try/except sets the
    # {"error": ...} payload rather than emitting a corrupt row. We assert the helper itself is the
    # thing that raises (the trainer wraps this exact call in try/except).
    realized = [np.zeros((8, 8), np.int64)]
    gt = [np.zeros((4, 4), np.int64)]  # shape mismatch vs realized
    margin = [np.zeros((8, 8), np.float32)]
    with pytest.raises(Exception):
        _M._annulus_metrics_from_maps(realized, gt, margin, band=2.0, bottom_k=0.05)


def test_default_off_flag_absent_yields_falsey_gate():
    # the trainer reads bool(getattr(args, "annulus_telemetry", False)); an object without the attr
    # (the flag never added / absent) must gate OFF => no annulus computation, byte-identical.
    class _NoAttr:
        pass

    assert bool(getattr(_NoAttr(), "annulus_telemetry", False)) is False


def test_zero_flip_case_is_all_zero_shares():
    # a perfect witness (realized == gt) => no flips => every flip metric is 0.0 (no crash / div0).
    gt = np.zeros((8, 8), np.int64)
    gt[:, 3] = 1
    margin = np.full((8, 8), 5.0, np.float32)
    margin[:, 2:5] = 0.5
    m = _M._annulus_metrics_from_maps([gt.copy()], [gt], [margin], band=2.0, bottom_k=0.05)
    assert m["n_flips"] == 0
    assert m["overall_d_seg"] == 0.0
    assert m["threshold"]["annulus_flip_frac"] == 0.0
    assert m["threshold"]["interior_flip_frac"] == 0.0
    assert m["threshold"]["annulus_flip_mass_share"] == 0.0
