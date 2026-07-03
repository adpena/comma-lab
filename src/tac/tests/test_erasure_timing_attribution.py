"""Tests for tools/erasure_timing_attribution.py (#253 erasure-timing probe).

Fast pure-function tests of the metric engine + guards (NO heavy render/SegNet).
Validates: the CANONICAL comma10k class order (anti-luma-sort regression), the
per-class d_seg decomposition + recall math, connected-component island survival,
#218 persistence-binned survival monotonicity sanity, the NO-MPS authority guard,
the R-operator shape/determinism, stage/epoch filename parsing, and compare-mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO, _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import tools.erasure_timing_attribution as et  # noqa: E402


# ---------------------------------------------------------------------------
# canonical class order (anti-luma-sort regression -- a 3x-recurring bug)
# ---------------------------------------------------------------------------
def test_canonical_class_order_is_measured_comma10k_not_luma_sort():
    assert et.CANONICAL_CLASS_NAMES == {
        0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar",
    }
    # The FORBIDDEN luma-sort of class_values=[41,76,90,124,161] gives
    # [Road, Lane, MyCar, Undrivable, Movable] -- assert we are NOT that.
    luma_sort = {0: "Road", 1: "Lane", 2: "MyCar", 3: "Undrivable", 4: "Movable"}
    assert et.CANONICAL_CLASS_NAMES != luma_sort
    assert et.CANONICAL_CLASS_NAMES[2] == "Undrivable"  # NOT MyCar
    assert et.CANONICAL_CLASS_NAMES[4] == "MyCar"       # NOT Movable


def test_rare_classes_are_lane_and_movable():
    assert et.RARE_CLASSES == (1, 3)
    assert et.CANONICAL_CLASS_NAMES[1] == "Lane"
    assert et.CANONICAL_CLASS_NAMES[3] == "Movable"


# ---------------------------------------------------------------------------
# per-class d_seg decomposition + recall math
# ---------------------------------------------------------------------------
def _finalize(acc: et._Accum, n_pairs: int = 1) -> dict:
    return acc.finalize(n_pairs)


def test_per_class_flip_mass_sums_to_total_dseg():
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 5, size=(32, 40)).astype(np.int64)
    realized = gt.copy()
    # corrupt a known set of pixels across classes
    realized[gt == 0] = 2  # Road -> Undrivable flips
    realized[5:8, 5:8] = (gt[5:8, 5:8] + 1) % 5
    acc = et._Accum()
    acc.update_seg(realized, gt, gt)  # witness_am == gt (irrelevant here)
    row = _finalize(acc)
    per = row["per_class_flip_mass"]
    total_flip = int(np.count_nonzero(realized != gt))
    # counts sum to total flips
    assert sum(v["flip_count"] for v in per.values()) == total_flip
    # frac_of_dseg sums to 1.0 (when there are flips)
    assert row["d_seg_realized"] == pytest.approx(total_flip / gt.size)
    assert sum(v["frac_of_dseg"] for v in per.values()) == pytest.approx(1.0)


def test_dseg_zero_when_realized_equals_gt_no_div_by_zero():
    gt = np.full((16, 16), 2, np.int64)
    acc = et._Accum()
    acc.update_seg(gt.copy(), gt.copy(), gt)
    row = _finalize(acc)
    assert row["d_seg_realized"] == 0.0
    assert all(v["frac_of_dseg"] == 0.0 for v in row["per_class_flip_mass"].values())


def test_recall_math_matches_manual():
    gt = np.array([[1, 1, 3, 3], [1, 2, 3, 0]], np.int64)
    realized = np.array([[1, 2, 3, 0], [1, 2, 3, 0]], np.int64)
    acc = et._Accum()
    acc.update_seg(realized, gt, gt)
    rec = _finalize(acc)["recall"]
    # Lane(1): support=3 (positions (0,0),(0,1),(1,0)); correct where realized==1: (0,0),(1,0)=2
    assert rec["Lane"]["support_px"] == 3
    assert rec["Lane"]["correct_px"] == 2
    assert rec["Lane"]["recall"] == pytest.approx(2 / 3)
    # Movable(3): support=3 ((0,2),(0,3),(1,2)); correct where realized==3: (0,2),(1,2)=2
    assert rec["Movable"]["recall"] == pytest.approx(2 / 3)


def test_recall_perfect_and_zero_and_none_support():
    gt = np.array([[1, 1], [0, 0]], np.int64)  # no Movable(3) present
    acc = et._Accum()
    acc.update_seg(gt.copy(), gt.copy(), gt)  # perfect
    rec = _finalize(acc)["recall"]
    assert rec["Lane"]["recall"] == 1.0
    assert rec["Movable"]["recall"] is None  # zero support -> None (no div-by-zero)


# ---------------------------------------------------------------------------
# connected-component island survival
# ---------------------------------------------------------------------------
def test_island_cc_counts_and_half_survival():
    gt = np.full((20, 20), 2, np.int64)  # Undrivable background
    gt[2:4, 2:4] = 1   # Lane island A (size 4)
    gt[2:4, 10:12] = 1  # Lane island B (size 4)
    realized = gt.copy()
    realized[2:4, 10:12] = 2  # island B erased (Lane->Undrivable)
    acc = et._Accum()
    acc.update_islands_cc(realized, gt, fine_thresh=32)
    cc = _finalize(acc)["island_cc"]["Lane"]
    assert cc["n_islands"] == 2
    assert cc["n_fine_islands"] == 2          # both size-4 <= 32
    assert cc["fine_island_survival_rate"] == pytest.approx(0.5)  # A alive, B dead
    assert cc["mean_fine_island_recall"] == pytest.approx(0.5)    # (1.0 + 0.0)/2


def test_island_cc_all_alive_when_realized_correct():
    gt = np.full((16, 16), 2, np.int64)
    gt[1:3, 1:3] = 3   # Movable island
    gt[5:7, 5:7] = 3   # another
    acc = et._Accum()
    acc.update_islands_cc(gt.copy(), gt, fine_thresh=32)
    cc = _finalize(acc)["island_cc"]["Movable"]
    assert cc["n_islands"] == 2
    assert cc["fine_island_survival_rate"] == 1.0


def test_island_cc_fine_threshold_excludes_large():
    gt = np.full((30, 30), 2, np.int64)
    gt[0:2, 0:2] = 1     # fine island (size 4)
    gt[10:20, 10:20] = 1  # large island (size 100)
    acc = et._Accum()
    acc.update_islands_cc(gt.copy(), gt, fine_thresh=32)
    cc = _finalize(acc)["island_cc"]["Lane"]
    assert cc["n_islands"] == 2
    assert cc["n_fine_islands"] == 1  # only the size-4 one is "fine"


# ---------------------------------------------------------------------------
# #218 persistence-binned survival
# ---------------------------------------------------------------------------
def test_persistence_bins_survival_full_when_all_correct():
    rng = np.random.default_rng(1)
    gt = np.full((40, 40), 0, np.int64)
    gt[rng.random((40, 40)) < 0.15] = 1  # scattered Lane pixels
    gt[rng.random((40, 40)) < 0.05] = 3  # scattered Movable pixels
    margin = rng.random((40, 40)).astype(np.float32)
    acc = et._Accum()
    acc.update_islands_persistence(gt.copy(), gt, margin)  # realized == gt
    pers = _finalize(acc)["island_persistence"]
    assert pers is not None
    assert pers["n_features_total"] > 0
    for b in pers["bins"]:
        if b["survival"] is not None:
            assert b["survival"] == pytest.approx(1.0)  # perfect -> all features survive


def test_persistence_survival_in_unit_interval_and_low_pers_key():
    rng = np.random.default_rng(2)
    gt = np.full((48, 48), 2, np.int64)
    gt[rng.random((48, 48)) < 0.2] = 1
    gt[rng.random((48, 48)) < 0.05] = 3
    realized = gt.copy()
    # erase a chunk of Lane -> survival < 1 in some bin
    realized[gt == 1] = 2
    margin = rng.random((48, 48)).astype(np.float32)
    acc = et._Accum()
    acc.update_islands_persistence(realized, gt, margin)
    pers = _finalize(acc)["island_persistence"]
    assert "low_pers_finest_survival" in pers
    for b in pers["bins"]:
        assert b["survival"] is None or (0.0 <= b["survival"] <= 1.0)


# ---------------------------------------------------------------------------
# witness-internal SDF + SURVIVE-R (pre-R) diagnostics
# ---------------------------------------------------------------------------
def test_witness_sdf_and_pre_r_gaps():
    gt = np.array([[1, 1, 2, 2], [0, 0, 4, 4]], np.int64)
    realized = gt.copy()
    realized[0, 0] = 2                    # 1 realized flip
    witness_am = gt.copy(); witness_am[0, 0:2] = 3  # 2 witness-sdf flips (gt[0,0:2]==[1,1])
    pre_r = gt.copy()                     # 0 pre-R flips
    acc = et._Accum()
    acc.update_seg(realized, witness_am, gt, pre_r=pre_r)
    row = _finalize(acc)
    assert row["d_seg_realized"] == pytest.approx(1 / gt.size)
    assert row["d_seg_witness_sdf_internal"] == pytest.approx(2 / gt.size)
    assert row["d_seg_pre_R"] == pytest.approx(0.0)
    # realized (1 flip) - pre_R (0 flips) = R roundtrip erasure gap
    assert row["r_roundtrip_erasure_gap"] == pytest.approx(1 / gt.size)


def test_pre_r_none_when_not_measured():
    gt = np.zeros((8, 8), np.int64)
    acc = et._Accum()
    acc.update_seg(gt.copy(), gt.copy(), gt)  # no pre_r
    row = _finalize(acc)
    assert row["d_seg_pre_R"] is None
    assert row["r_roundtrip_erasure_gap"] is None


# ---------------------------------------------------------------------------
# NO-MPS authority guard
# ---------------------------------------------------------------------------
class _FakeParam:
    def __init__(self, dev: str):
        import torch
        self.device = torch.device(dev)


class _FakeSegnet:
    def __init__(self, dev: str):
        self._p = [_FakeParam(dev)]

    def parameters(self):
        return iter(self._p)


def test_no_mps_guard_rejects_mps_accepts_cpu():
    et._assert_segnet_cpu(_FakeSegnet("cpu"))  # ok
    with pytest.raises(RuntimeError, match="CPU-torch"):
        et._assert_segnet_cpu(_FakeSegnet("mps"))


# ---------------------------------------------------------------------------
# R operator: shape + determinism (contest-faithful bicubic -> camera uint8)
# ---------------------------------------------------------------------------
def test_R_to_camera_shape_dtype_deterministic():
    rng = np.random.default_rng(3)
    rgb = rng.random((384 * 512, 3)).astype(np.float32) * 255.0
    a = et._R_to_camera(rgb, 384, 512, et.CAMERA_H, et.CAMERA_W)
    b = et._R_to_camera(rgb, 384, 512, et.CAMERA_H, et.CAMERA_W)
    assert a.shape == (et.CAMERA_H, et.CAMERA_W, 3)
    assert a.dtype == np.uint8
    assert np.array_equal(a, b)  # deterministic
    assert a.min() >= 0 and a.max() <= 255


# ---------------------------------------------------------------------------
# stage/epoch filename parsing
# ---------------------------------------------------------------------------
def test_parse_stage_epoch_preserved_stage_ckpt():
    tag, ep = et._parse_stage_epoch(Path("levelset_ckpt_stageMuonStart_ep500.npz"), None)
    assert tag == "stageMuonStart"
    assert ep == 500
    tag2, ep2 = et._parse_stage_epoch(Path("levelset_ckpt_stageCE_ep3000.npz"), None)
    assert tag2 == "stageCE" and ep2 == 3000


def test_parse_stage_epoch_ema_best_uses_stem_and_cfg_epoch():
    tag, ep = et._parse_stage_epoch(Path("levelset_witness_ema_BEST.npz"), 1001)
    assert tag == "ema_BEST"   # stem minus the levelset_witness_ prefix
    assert ep == 1001          # from __epoch fallback


# ---------------------------------------------------------------------------
# compare-mode: per-lever attribution diff aligned by stage_tag
# ---------------------------------------------------------------------------
def _mkrow(stage: str, epoch: int, d_seg: float, lane_rec: float) -> dict:
    return {
        "stage_tag": stage, "epoch": epoch, "d_seg_realized": d_seg,
        "recall": {"Lane": {"recall": lane_rec}, "Movable": {"recall": 0.9}},
        "island_cc": {"Lane": {"fine_island_survival_rate": 0.2}},
        "island_persistence": {"low_pers_finest_survival": 0.8},
    }


def test_compare_curves_aligns_and_diffs_by_stage():
    a = [_mkrow("stageCE", 3000, 0.010, 0.70), _mkrow("stageMuonStart", 5000, 0.005, 0.80)]
    b = [_mkrow("stageCE", 3000, 0.011, 0.68), _mkrow("stageMuonStart", 5000, 0.004, 0.85)]
    diff = et.compare_curves(a, b, "runA", "runB")
    assert diff["mode"] == "compare"
    per = {d["stage_tag"]: d for d in diff["per_stage"]}
    assert set(per) == {"stageCE", "stageMuonStart"}
    assert per["stageMuonStart"]["delta_d_seg_realized"] == pytest.approx(-0.001)
    assert per["stageMuonStart"]["delta_lane_recall"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# text summary renders with the canonical class-order banner
# ---------------------------------------------------------------------------
def test_render_text_curve_has_header_and_class_order():
    gt = np.full((16, 16), 2, np.int64)
    gt[0:2, 0:2] = 1
    acc = et._Accum()
    acc.update_seg(gt.copy(), gt.copy(), gt)
    acc.update_islands_cc(gt.copy(), gt, 32)
    row = acc.finalize(1)
    row.update({"stage_tag": "ema_BEST", "epoch": 1001,
                "island_persistence": {"low_pers_finest_survival": 0.8}})
    txt = et.render_text_curve([row])
    assert "ERASURE-TIMING CURVE" in txt
    assert "0=Road 1=Lane 2=Undrivable 3=Movable 4=MyCar" in txt
