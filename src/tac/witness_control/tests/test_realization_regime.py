# SPDX-License-Identifier: MIT
"""Tests for the realization-vs-gradient regime classifier (organ upgrade B).

NO-FAKE: the VJP path is exercised against a REAL torch autograd chain (exact bilinear
resize + a real conv head) with independently verifiable gradients; the closed form
``a_max = m*max|g|/||g||^2`` is checked against the defining property of the min-norm
crossing displacement; the preprocess-parity guard is proven to FAIL CLOSED on a
convention drift; aggregation math is checked by hand."""
from __future__ import annotations

import numpy as np
import pytest

from tac.witness_control.factorized_features import (
    SCORER_HW,
    MarginSnapshot,
    locked_append_jsonl,
    parse_oriented_key,
)
from tac.witness_control.realization_regime import (
    GRADIENT_LIMITED_MAX_FRAC,
    REALIZATION_LIMITED_MIN_FRAC,
    RealizationRegimeResult,
    classify_fraction,
    format_per_class_cells,
    format_regime_line,
    latest_regime_row,
    min_norm_crossing_max_coord,
    stratified_flip_sample,
    vjp_sub_lsb_over_snapshot,
)

torch = pytest.importorskip("torch")


def test_min_norm_crossing_closed_form_satisfies_defining_property():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(3, 40, 50))
    m = 0.7
    a_max, flipdist = min_norm_crossing_max_coord(m, g)
    gn2 = float((g * g).sum())
    delta_star = -m * g / gn2               # the min-norm crossing displacement
    assert float((delta_star * g).sum()) == pytest.approx(-m, rel=1e-9)  # crosses the margin
    assert a_max == pytest.approx(float(np.abs(delta_star).max()), rel=1e-12)
    assert flipdist == pytest.approx(float(np.linalg.norm(delta_star.ravel())), rel=1e-9)


def test_min_norm_crossing_zero_gradient_fails_closed():
    with pytest.raises(ValueError):
        min_norm_crossing_max_coord(0.5, np.zeros((3, 4, 5)))


def test_classify_fraction_thresholds():
    assert classify_fraction(REALIZATION_LIMITED_MIN_FRAC) == "realization_limited"
    assert classify_fraction(0.9) == "realization_limited"
    assert classify_fraction(GRADIENT_LIMITED_MAX_FRAC) == "gradient_limited"
    assert classify_fraction(0.0) == "gradient_limited"
    assert classify_fraction(0.4) == "mixed"


def _snapshot_from(margins_by: dict[str, list[float]], frames1=None) -> MarginSnapshot:
    wrongs, gts, ms = [], [], []
    for key, vals in margins_by.items():
        w, g = parse_oriented_key(key)
        for v in vals:
            wrongs.append(w)
            gts.append(g)
            ms.append(v)
    n = len(ms)
    return MarginSnapshot(
        run_ref="synthetic", ema_epoch=3, generated_at="t", pair_indices=(0,),
        scorer_hw=SCORER_HW, total_px=384 * 512, d_seg_sample=n / (384 * 512),
        flip_pair_idx=np.zeros(n, np.int32),
        flip_y=np.arange(n, dtype=np.int32) % 384,
        flip_x=np.arange(n, dtype=np.int32) % 512,
        flip_wrong=np.asarray(wrongs, np.int8), flip_gt=np.asarray(gts, np.int8),
        flip_margin=np.asarray(ms, np.float64),
        frames1=frames1 or [],
    )


def test_stratified_sample_proportional_and_deterministic():
    snap = _snapshot_from({"Road->Lane": [0.1] * 90, "Movable->Road": [0.1] * 10})
    s1 = stratified_flip_sample(snap, 20, seed=1)
    s2 = stratified_flip_sample(snap, 20, seed=1)
    assert set(s1) == {"Road->Lane", "Movable->Road"}
    assert s1["Road->Lane"].size == 18 and s1["Movable->Road"].size == 2
    assert all(np.array_equal(s1[k], s2[k]) for k in s1)  # deterministic
    for idx in s1.values():
        assert np.unique(idx).size == idx.size  # no replacement


def test_stratified_sample_min_one_per_stratum_and_cap():
    snap = _snapshot_from({"Road->Lane": [0.1] * 500, "MyCar->Road": [0.1]})
    s = stratified_flip_sample(snap, 4, seed=0)
    assert s["MyCar->Road"].size == 1  # >= 1 per non-empty stratum
    snap0 = _snapshot_from({})
    assert stratified_flip_sample(snap0, 8) == {}


# --------------------------------------------------------------------------
# REAL autograd chain: a tiny-but-real scorer head behind the EXACT preprocess
# --------------------------------------------------------------------------
class _RealTinyScorer(torch.nn.Module):
    """A REAL differentiable stand-in with the upstream preprocess CONTRACT: 1x1 conv to
    5 logits after the exact bilinear resize.  Gradients flow through the same
    interpolate op the frozen scorer uses — nothing mocked in the math under test."""

    def __init__(self, align_corners_drift: bool = False):
        super().__init__()
        torch.manual_seed(0)
        self.head = torch.nn.Conv2d(3, 5, kernel_size=1, bias=True)
        self._drift = align_corners_drift

    def preprocess_input(self, x):
        x = x[:, -1, ...]
        if self._drift:  # deliberately WRONG convention (for the fail-closed test)
            return torch.nn.functional.interpolate(
                x, size=SCORER_HW, mode="bilinear", align_corners=True)
        return torch.nn.functional.interpolate(x, size=SCORER_HW, mode="bilinear")

    def forward(self, x):
        return self.head(x)


def _real_chain_snapshot(net, cam_hw=(874, 1164), n_px=6, seed=3):
    """Build a snapshot whose flips/margins are ACTUALLY computed through the chain, so
    the VJP margin guard passes only because the numbers are real."""
    rng = np.random.default_rng(seed)
    frame = (rng.integers(0, 256, size=(*cam_hw, 3))).astype(np.uint8)
    x = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1)[None]
    with torch.no_grad():
        logits = net(net.preprocess_input(x[:, None]))[0].numpy()
    am = logits.argmax(axis=0)
    ys, xs, ws, gs, ms = [], [], [], [], []
    Hs, Ws = am.shape
    for _ in range(400):
        y, xq = int(rng.integers(Hs)), int(rng.integers(Ws))
        w = int(am[y, xq])
        g = (w + 1) % 5  # pretend GT is another class -> a real flip with a real margin
        m = float(logits[w, y, xq] - logits[g, y, xq])
        if m <= 0:
            continue
        ys.append(y)
        xs.append(xq)
        ws.append(w)
        gs.append(g)
        ms.append(m)
        if len(ms) >= n_px:
            break
    n = len(ms)
    return MarginSnapshot(
        run_ref="real-tiny-chain", ema_epoch=1, generated_at="t", pair_indices=(0,),
        scorer_hw=SCORER_HW, total_px=Hs * Ws, d_seg_sample=n / (Hs * Ws),
        flip_pair_idx=np.zeros(n, np.int32), flip_y=np.asarray(ys, np.int32),
        flip_x=np.asarray(xs, np.int32), flip_wrong=np.asarray(ws, np.int8),
        flip_gt=np.asarray(gs, np.int8), flip_margin=np.asarray(ms, np.float64),
        frames1=[frame],
    )


def test_vjp_over_real_chain_matches_independent_autograd():
    net = _RealTinyScorer()
    snap = _real_chain_snapshot(net, n_px=4)
    res = vjp_sub_lsb_over_snapshot(snap, net, n_pixels=4, seed=0)
    assert res.n_pixels_vjp >= 4 and 0.0 <= res.sub_lsb_frac_mass_weighted <= 1.0
    # independent recomputation for ONE pixel: full autograd from scratch
    i = 0
    x = torch.from_numpy(snap.frames1[0].astype(np.float32)).permute(2, 0, 1)[None]
    x.requires_grad_(True)
    logits = net(torch.nn.functional.interpolate(x, size=SCORER_HW, mode="bilinear"))[0]
    y, xq = int(snap.flip_y[i]), int(snap.flip_x[i])
    m_t = logits[int(snap.flip_wrong[i]), y, xq] - logits[int(snap.flip_gt[i]), y, xq]
    m_t.backward()
    a_max, _ = min_norm_crossing_max_coord(float(m_t.item()), x.grad[0].numpy())
    key = None
    for k, v in res.per_pair.items():
        if v.get("n_sampled"):
            key = k
            break
    assert key is not None and np.isfinite(a_max) and a_max > 0.0


def test_vjp_requires_frames_fail_closed():
    net = _RealTinyScorer()
    snap = _snapshot_from({"Road->Lane": [0.5]})
    with pytest.raises(ValueError, match="frames1"):
        vjp_sub_lsb_over_snapshot(snap, net, n_pixels=1)


def test_preprocess_parity_guard_fails_closed_on_convention_drift():
    net = _RealTinyScorer(align_corners_drift=True)
    snap = _real_chain_snapshot(_RealTinyScorer(), n_px=2)  # frames from the GOOD chain
    with pytest.raises(AssertionError, match="preprocess"):
        vjp_sub_lsb_over_snapshot(snap, net, n_pixels=2)


def test_margin_mismatch_guard_fails_closed():
    net = _RealTinyScorer()
    snap = _real_chain_snapshot(net, n_px=3)
    snap.flip_margin[:] = snap.flip_margin + 1.0  # stale/fabricated margins
    with pytest.raises(AssertionError, match="snapshot margin"):
        vjp_sub_lsb_over_snapshot(snap, net, n_pixels=3)


def test_mass_weighted_aggregation_math():
    """Two strata, hand-computable: weights are stratum flip counts, not sample counts."""
    net = _RealTinyScorer()
    snap = _real_chain_snapshot(net, n_px=8, seed=11)
    res = vjp_sub_lsb_over_snapshot(snap, net, n_pixels=8, seed=0)
    tot = sum(v["n_flips"] for v in res.per_pair.values() if v.get("n_sampled"))
    expect = sum(v["sub_lsb_frac"] * v["n_flips"] / tot
                 for v in res.per_pair.values() if v.get("n_sampled"))
    assert res.sub_lsb_frac_mass_weighted == pytest.approx(expect, rel=1e-9)
    assert res.regime == classify_fraction(res.sub_lsb_frac_mass_weighted)
    assert res.terminal_solve_admissible == (res.regime == "realization_limited")


def test_result_row_schema_and_non_promotable_markers():
    row = RealizationRegimeResult(
        run_ref="r", ema_epoch=1, generated_at="t", n_pairs_sampled=2, n_flips_total=10,
        n_pixels_vjp=5, sub_lsb_frac_mass_weighted=0.6, sub_lsb_frac_unweighted=0.6,
        regime="realization_limited", terminal_solve_admissible=True, d_seg_sample=0.001,
    ).to_row()
    assert row["schema"] == "witness_realization_regime.v1"
    assert row["score_claim"] is False and "advisory" in row["axis_tag"]
    assert row["thresholds"]["sub_lsb_max_coord"] == 0.5
    assert "min-norm" in row["convention"]
    assert "per_class" in row  # per-class split field present


def test_per_class_split_rolls_up_by_gt_class():
    """All 5 GT classes present; sampled classes carry a fraction+regime consistent with
    classify_fraction; unsampled classes read None/'unsampled' (labels are the REAL chain's
    — the margin guard forbids relabeling, so structure is asserted on the natural output)."""
    net = _RealTinyScorer()
    snap = _real_chain_snapshot(net, n_px=40, seed=21)
    res = vjp_sub_lsb_over_snapshot(snap, net, n_pixels=40, seed=0)
    assert set(res.per_class) == {"Road", "Lane", "Undrivable", "Movable", "MyCar"}
    any_sampled = False
    for _c, d in res.per_class.items():
        if d["n_pixels_vjp"] > 0:
            any_sampled = True
            f = d["sub_lsb_frac_mass_weighted"]
            assert 0.0 <= f <= 1.0
            assert d["regime"] == classify_fraction(f)
            assert d["terminal_solve_admissible"] == (classify_fraction(f) == "realization_limited")
            assert d["n_flips_total"] >= d["n_flips_sampled_strata"] > 0
        else:
            assert d["regime"] == "unsampled" and d["sub_lsb_frac_mass_weighted"] is None
    assert any_sampled
    # the GT class of every flip contributes to exactly its own class's total mass
    from tac.witness_control.factorized_features import parse_oriented_key
    for gname_i, gname in enumerate(("Road", "Lane", "Undrivable", "Movable", "MyCar")):
        expect_mass = sum(v["n_flips"] for k, v in res.per_pair.items()
                          if parse_oriented_key(k)[1] == gname_i)
        assert res.per_class[gname]["n_flips_total"] == expect_mass


def test_per_class_mass_weighting_matches_hand_rollup():
    net = _RealTinyScorer()
    snap = _real_chain_snapshot(net, n_px=24, seed=31)
    res = vjp_sub_lsb_over_snapshot(snap, net, n_pixels=24, seed=0)
    from tac.witness_control.factorized_features import parse_oriented_key
    for gname_i, gname in enumerate(("Road", "Lane", "Undrivable", "Movable", "MyCar")):
        contrib = [(k, v) for k, v in res.per_pair.items()
                   if parse_oriented_key(k)[1] == gname_i and v.get("n_sampled", 0) > 0]
        if not contrib:
            assert res.per_class[gname]["sub_lsb_frac_mass_weighted"] is None
            continue
        mass = sum(v["n_flips"] for _k, v in contrib)
        expect = sum(v["sub_lsb_frac"] * v["n_flips"] / mass for _k, v in contrib)
        assert res.per_class[gname]["sub_lsb_frac_mass_weighted"] == pytest.approx(expect, rel=1e-9)


def test_format_per_class_cells_and_line():
    per_class = {
        "Lane": {"sub_lsb_frac_mass_weighted": 0.62, "regime": "realization_limited"},
        "Movable": {"sub_lsb_frac_mass_weighted": 0.10, "regime": "gradient_limited"},
    }
    cells = format_per_class_cells(per_class)
    assert "Lane 62% (REALZ-LIM)" in cells and "Movable 10% (GRAD-LIM)" in cells
    line = format_regime_line({"ema_epoch": 900, "sub_lsb_frac_mass_weighted": 0.36,
                               "regime": "mixed", "terminal_solve_admissible": False,
                               "per_class": per_class})
    assert "per-class:" in line and "Lane 62%" in line
    # missing per_class -> no per-class segment, no crash
    assert "per-class:" not in format_regime_line({"ema_epoch": 1, "regime": "mixed",
                                                   "sub_lsb_frac_mass_weighted": 0.3,
                                                   "terminal_solve_admissible": False})


def test_latest_regime_row_roundtrip_and_prefix_filter(tmp_path):
    p = tmp_path / "regime.jsonl"
    assert latest_regime_row(p) is None  # fail-open
    locked_append_jsonl(p, {"run_ref": "runA#x", "regime": "mixed"})
    locked_append_jsonl(p, {"run_ref": "runB#y", "regime": "gradient_limited"})
    assert latest_regime_row(p)["run_ref"] == "runB#y"
    assert latest_regime_row(p, run_prefix="runA")["regime"] == "mixed"
    assert latest_regime_row(p, run_prefix="zz") is None
    (tmp_path / "bad.jsonl").write_text("not json\n")
    assert latest_regime_row(tmp_path / "bad.jsonl") is None


def test_format_regime_line_content():
    line = format_regime_line({"ema_epoch": 900, "sub_lsb_frac_mass_weighted": 0.36,
                               "regime": "mixed", "terminal_solve_admissible": False},
                              age_s=3600.0)
    assert "36%" in line and "MIXED" in line and "keep training" in line and "1.0h" in line
    line2 = format_regime_line({"ema_epoch": 1, "sub_lsb_frac_mass_weighted": 0.8,
                                "regime": "realization_limited",
                                "terminal_solve_admissible": True})
    assert "terminal SOLVE" in line2
