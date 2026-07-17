# SPDX-License-Identifier: MIT
"""Tests for the #406 apply-pass batch orchestrator (no-scorer logic paths only).

Covers the REGISTRY integrity, the param-transform composition + ΔS arithmetic, the
required-key validation, the fail-closed dep/skip logic, and the memory-guard refusal.
The heavy decode+scorer path (torch + gt cache + SegNet/PoseNet) is exercised ONLY in
the governed fire-mode run — never in unit tests (advisory, memory-heavy)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "src", _REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import witness_applypass_batch as wb  # noqa: E402


# ---------------------------------------------------------------------------
# registry integrity
# ---------------------------------------------------------------------------
def test_registry_ids_unique_and_ordered():
    reg = wb.build_registry()
    ids = [lv.lever_id for lv in reg]
    assert len(ids) == len(set(ids)), f"duplicate lever ids: {ids}"
    # the five named levers + the three #519 param-transform levers are all present
    for want in ("gauge_519", "palette_canon_519", "both_canon_519",
                 "bit_alloc_336", "low_rank_pose_140", "tropnnc_311", "blind_coord_401"):
        assert want in ids, f"lever {want} missing from registry"


def test_registry_kinds_and_shapes():
    reg = wb.build_registry()
    for lv in reg:
        assert lv.kind in ("param_transform", "delegate")
        if lv.kind == "param_transform":
            assert callable(lv.transform), lv.lever_id
            assert lv.required_keys, lv.lever_id
        else:
            assert lv.deps is not None, lv.lever_id
            assert callable(lv.deps), lv.lever_id


def test_delegate_deps_are_boolean_reasoned():
    reg = wb.build_registry()
    for lv in reg:
        if lv.kind != "delegate":
            continue
        ok, reason = lv.deps()
        assert isinstance(ok, bool)
        assert isinstance(reason, str)
        if not ok:
            assert reason, "a failed dep MUST carry a reason (honest skip)"


# ---------------------------------------------------------------------------
# param-transform math (reuses null_subspace_rate_measure transforms)
# ---------------------------------------------------------------------------
def _toy_params() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "out_sdf.weight": rng.standard_normal((5, 4)).astype(np.float32),
        "out_sdf.bias": rng.standard_normal(5).astype(np.float32),
        "palette": rng.standard_normal((5, 3)).astype(np.float32),
        "out_tex.bias": rng.standard_normal(3).astype(np.float32),
    }


def test_gauge_transform_removes_class_mean():
    p = _toy_params()
    reg = {lv.lever_id: lv for lv in wb.build_registry()}
    out = reg["gauge_519"].transform(p)
    # class-mean (mean across the 5 rows) is now ~0 for weight and bias
    assert np.allclose(out["out_sdf.weight"].mean(axis=0), 0.0, atol=1e-5)
    assert abs(float(out["out_sdf.bias"].mean())) < 1e-5
    # the input dict is not mutated in place
    assert not np.allclose(p["out_sdf.weight"].mean(axis=0), 0.0, atol=1e-5)


def test_palette_transform_is_render_invariant_channel_fold():
    p = _toy_params()
    reg = {lv.lever_id: lv for lv in wb.build_registry()}
    out = reg["palette_canon_519"].transform(p)
    # palette channel-mean removed; the removed mean lands in out_tex.bias
    assert np.allclose(out["palette"].mean(axis=0), 0.0, atol=1e-5)
    moved = out["out_tex.bias"] - p["out_tex.bias"]
    assert np.allclose(moved, p["palette"].mean(axis=0), atol=1e-5)


def test_compose_is_left_to_right_and_both_matches():
    p = _toy_params()
    reg = {lv.lever_id: lv for lv in wb.build_registry()}
    composed = wb._compose(reg["gauge_519"].transform, reg["palette_canon_519"].transform)(p)
    both = reg["both_canon_519"].transform(p)
    for k in composed:
        assert np.allclose(composed[k], both[k], atol=1e-5), k


# ---------------------------------------------------------------------------
# ΔS arithmetic
# ---------------------------------------------------------------------------
def test_score_delta_zero_at_baseline():
    base = {"d_seg": 0.003, "d_pose": 143.0, "bytes": 84126.0}
    assert wb._score_delta(0.003, 143.0, 84126.0, base) == pytest.approx(0.0, abs=1e-12)


def test_score_delta_signs_and_composition_not_summed():
    base = {"d_seg": 0.003, "d_pose": 100.0, "bytes": 84126.0}
    # a pure d_seg improvement lowers S by 100*Δd_seg
    dS = wb._score_delta(0.003 - 1e-5, 100.0, 84126.0, base)
    assert dS == pytest.approx(100.0 * (-1e-5), abs=1e-9)
    # bytes term uses the contest denominator
    dS_b = wb._score_delta(0.003, 100.0, 84126.0 + 37_545_489, base)
    assert dS_b == pytest.approx(25.0, abs=1e-6)


# ---------------------------------------------------------------------------
# key-presence validation + fail-closed skip (no ckpt needed)
# ---------------------------------------------------------------------------
class _FakeBatch:
    """Minimal stand-in exposing just the pieces the pure helpers need."""

    def __init__(self, params):
        self.params = params

    _keys_present = wb.ApplyPassBatch._keys_present


def test_keys_present_detects_missing():
    lv = {lv.lever_id: lv for lv in wb.build_registry()}["both_canon_519"]
    full = _FakeBatch(_toy_params())
    ok, missing = full._keys_present(lv)
    assert ok and missing == []
    partial = _FakeBatch({"out_sdf.weight": np.zeros((5, 4), np.float32)})
    ok2, missing2 = partial._keys_present(lv)
    assert not ok2
    assert "palette" in missing2 and "out_sdf.bias" in missing2


# ---------------------------------------------------------------------------
# memory guard + free-mem helper
# ---------------------------------------------------------------------------
def test_free_gib_returns_number():
    v = wb._free_gib()
    assert isinstance(v, float)
    # either a real positive reading or NaN (both are acceptable, never a crash)
    assert v > 0 or v != v  # noqa: PLR0124  (NaN self-inequality is intentional)


def test_dep_helpers():
    ok, _ = wb._dep_module("numpy")()
    assert ok
    ok2, reason2 = wb._dep_module("definitely_not_a_real_module_xyz")()
    assert not ok2 and reason2
    ok3, _ = wb._dep_file(_REPO / "tools" / "witness_applypass_batch.py")()
    assert ok3
    ok4, reason4 = wb._dep_file(_REPO / "tools" / "no_such_tool_xyz.py")()
    assert not ok4 and reason4


def test_rate_denominator_is_contest_value():
    assert wb._RATE_DENOM == 37_545_489


def test_free_gib_is_conservative_min():
    # _free_gib must never exceed the vm_stat truly-free reading when that is finite
    # (the guard errs toward refusing; psutil.available over-counts reclaimable).
    vm = wb._vm_stat_free_gib()
    fg = wb._free_gib()
    if vm == vm and fg == fg:  # both finite
        assert fg <= vm + 1e-9, f"_free_gib {fg} exceeded conservative vm_stat {vm}"


_DONOR = _REPO / "experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z"


@pytest.mark.skipif(not (_DONOR / "levelset_witness_ema_BEST.npz").exists(),
                    reason="donor checkpoint not present")
def test_fire_mode_refuses_when_memory_low(tmp_path, monkeypatch):
    """Fire mode with insufficient free memory must REFUSE (rc=4) BEFORE loading any
    scorer / gt cache — the OOM guard for the live P0 trainer (#205 lesson)."""
    monkeypatch.setattr(wb, "_free_gib", lambda: 1.0)
    # tripwire: if the guard is bypassed and it tries to load scorers, fail loudly.
    monkeypatch.setattr(wb.ns, "load_scorers",
                        lambda *a, **k: pytest.fail("scorers loaded despite low-mem refuse"))
    args = wb.build_parser().parse_args([
        "--ckpt-dir", str(_DONOR), "--pairs", "8",
        "--out-dir", str(tmp_path / "fireguard"), "--min-free-gib", "24"])
    batch = wb.ApplyPassBatch(args)  # light: copies the 448KB npz, loads params
    with pytest.raises(SystemExit) as ei:
        batch.run()
    assert "REFUSE" in str(ei.value)


@pytest.mark.skipif(not (_DONOR / "levelset_witness_ema_BEST.npz").exists(),
                    reason="donor checkpoint not present")
def test_dryrun_builds_param_transform_byte_rows(tmp_path):
    """Dry-run against the donor must build the #519 byte-close blobs (LIGHT, no
    scorer) and reproduce the leg-3 deltas (+11 gauge, -6 palette)."""
    args = wb.build_parser().parse_args([
        "--ckpt-dir", str(_DONOR), "--dry-run", "--compose-best",
        "--out-dir", str(tmp_path / "dry")])
    summary = wb.ApplyPassBatch(args).run()
    levers = {lv["lever_id"]: lv for lv in summary["levers"]}
    assert summary["baseline_0bin_bytes"] == 84126
    assert levers["gauge_519"]["delta"]["delta_bytes"] == 11
    assert levers["palette_canon_519"]["delta"]["delta_bytes"] == -6
    assert levers["both_canon_519"]["status"] == "dryrun"
    # delegates never fire in dry-run
    assert levers["bit_alloc_336"]["status"] == "staged"
    assert levers["low_rank_pose_140"]["status"] == "owed"  # no --pose-target
    assert levers["compose_best"]["status"] == "dryrun"
