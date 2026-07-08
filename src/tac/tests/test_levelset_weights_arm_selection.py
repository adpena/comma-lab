# SPDX-License-Identifier: MIT
"""B1 (confound-F1) — the level-set byte-close/eval weights-arm selection.

The byte-close/eval gains the POLYAK arm (weights_arm in {ema, live, polyak}) and RECORDS the N-way
selection (per-arm scores + which won + margins) — the missing "picks the better candidate" consumer.
These tests exercise the pure selection logic (arm discovery / labelling / ranking / fail-open) by
STUBBING the heavy ``run`` (no inflate / GT decode), so they run at $0 and require no GPU.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
for _p in ("tools", "src", "experiments", "upstream"):
    _pp = str(_REPO / _p)
    if _pp not in sys.path:
        sys.path.insert(0, _pp)

_spec = importlib.util.spec_from_file_location(
    "_lbce_armsel", _REPO / "tools" / "levelset_byte_close_and_eval.py")
_L = importlib.util.module_from_spec(_spec)
sys.modules["_lbce_armsel"] = _L
_spec.loader.exec_module(_L)


def _touch_arms(d: Path, arms) -> None:
    for a in arms:
        (d / _L._ARM_NPZ[a]).write_text("x")


def _fake_report(arm: str, npz: str, *, s: float | None, bytes_: int = 100) -> dict:
    par = {"skipped": True} if s is None else {
        "skipped": False, "pairs_scored": 600,
        "d_seg_realized_on_inflated": 0.004, "d_pose_realized_on_inflated": 0.01,
        "implied_S_advisory": s, "pose_blind": False}
    return {
        "weights_arm": _L._arm_label_for_npz(npz), "npz_name": npz,
        "byte_close": {"archive_zip_bytes": bytes_, "rate_term": 0.06},
        "parity_on_inflated_frames": par,
    }


def _install_fake_run(monkeypatch, per_arm_s: dict[str, float | None]):
    """Patch the module ``run`` to return a synthetic report keyed by the npz it was asked to load."""
    npz_to_arm = {fn: arm for arm, fn in _L._ARM_NPZ.items()}

    def _fake_run(ckpt_dir, *, npz_name, **kw):
        arm = npz_to_arm[npz_name]
        return _fake_report(arm, npz_name, s=per_arm_s[arm])

    monkeypatch.setattr(_L, "run", _fake_run)


# --- arm discovery + labelling --------------------------------------------
def test_discover_available_arms_failopen(tmp_path):
    assert _L.discover_available_arms(tmp_path) == []           # nothing present => empty (fail-open)
    _touch_arms(tmp_path, ["ema"])
    assert _L.discover_available_arms(tmp_path) == ["ema"]      # older ema-only run unchanged
    _touch_arms(tmp_path, ["polyak", "live"])
    assert _L.discover_available_arms(tmp_path) == ["ema", "live", "polyak"]  # canonical order


def test_arm_label_for_npz():
    assert _L._arm_label_for_npz("levelset_witness_polyak_mlx.npz") == "polyak"
    assert _L._arm_label_for_npz("/run/levelset_witness_live_mlx.npz") == "live"
    assert _L._arm_label_for_npz(None) == "default(ema)"
    assert _L._arm_label_for_npz("something_else.npz") == "explicit"


# --- N-way selection records per-arm scores + winner + margins ------------
def test_select_records_three_way_selection_and_winner(tmp_path, monkeypatch):
    _touch_arms(tmp_path, ["ema", "live", "polyak"])
    _install_fake_run(monkeypatch, {"ema": 0.30, "live": 0.28, "polyak": 0.22})  # polyak best
    winner_report, arm_reports = _L.select_best_weights_arm(
        tmp_path, gt_cache=None, max_pairs=None, fold_pose_sidecar=False, pose_sidecar_path=None,
        keep_packet=False, packet_dir=None, skip_parity=False,
        so_overrides={}, lane_render_band=False, lane_band_cfg=None, lane_rd=True, lane_res=False,
        pose_carrier=False, pose_carrier_cfg=None, verify_bit_exact=False, bit_exact_pairs=2,
        bit_exact_strict=True, run_exact_eval=False, eval_device="cpu", uncompressed_dir=None,
        video_names_file=None, eval_timeout=10)
    sel = winner_report["arm_selection"]
    assert winner_report["weights_arm"] == "polyak"       # top-level reflects the WINNER
    assert sel["winner"] == "polyak" and sel["polyak_present"] is True and sel["polyak_scored"] is True
    assert sel["available_arms"] == ["ema", "live", "polyak"]
    assert sel["ranked_arms"] == ["polyak", "live", "ema"]  # ascending S
    assert set(sel["per_arm"]) == {"ema", "live", "polyak"}
    # margins are loser_S - winner_S (>= 0)
    assert sel["margin_vs_winner"]["ema"] == pytest.approx(0.08)
    assert sel["margin_vs_winner"]["live"] == pytest.approx(0.06)
    assert sel["winner_implied_S_advisory"] == pytest.approx(0.22)
    assert set(arm_reports) == {"ema", "live", "polyak"}


def test_select_failopen_ema_only(tmp_path, monkeypatch):
    """An older run with only the EMA npz => a 1-arm 'selection' (polyak absent, honestly recorded)."""
    _touch_arms(tmp_path, ["ema"])
    _install_fake_run(monkeypatch, {"ema": 0.31, "live": None, "polyak": None})
    winner_report, _ = _L.select_best_weights_arm(
        tmp_path, gt_cache=None, max_pairs=None, fold_pose_sidecar=False, pose_sidecar_path=None,
        keep_packet=False, packet_dir=None, skip_parity=False, so_overrides={},
        lane_render_band=False, lane_band_cfg=None, lane_rd=True, lane_res=False, pose_carrier=False,
        pose_carrier_cfg=None, verify_bit_exact=False, bit_exact_pairs=2, bit_exact_strict=True,
        run_exact_eval=False, eval_device="cpu", uncompressed_dir=None, video_names_file=None,
        eval_timeout=10)
    sel = winner_report["arm_selection"]
    assert sel["available_arms"] == ["ema"] and sel["winner"] == "ema"
    assert sel["polyak_present"] is False and sel["margin_vs_winner"] == {}


def test_select_refuses_skip_parity(tmp_path):
    """NO-FAKE: cannot pick 'the better candidate' without a MEASURED d_seg/d_pose per arm."""
    _touch_arms(tmp_path, ["ema", "polyak"])
    with pytest.raises(ValueError, match="skip-parity"):
        _L.select_best_weights_arm(tmp_path, skip_parity=True)


def test_select_raises_when_no_arm_present(tmp_path):
    with pytest.raises(FileNotFoundError, match="no weights-arm npz"):
        _L.select_best_weights_arm(tmp_path, skip_parity=False)


def test_single_run_report_records_weights_arm():
    """The single-arm report records which arm it byte-closed (additive field)."""
    assert _L._arm_label_for_npz("levelset_witness_ema_mlx.npz") == "ema"
    # the report's weights_arm derives from cfg["npz_name"] via _arm_label_for_npz (covered above);
    # here we lock the extract helper reads implied_S from the parity block.
    rep = _fake_report("polyak", "levelset_witness_polyak_mlx.npz", s=0.19)
    m = _L._extract_arm_metrics(rep)
    assert m["weights_arm"] == "polyak" and m["implied_S_advisory"] == pytest.approx(0.19)
    assert m["parity_skipped"] is False
