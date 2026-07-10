"""Tests for tac.witness_control.telemetry_binding (#404 P0 land-now read side).

Behavioral coverage (not constants): each analyzer's BINDING / INERT / STALLED / DIVERGING verdict
is exercised with synthetic rows shaped exactly like the trainer's emit sites (schema verified
against experiments/train_levelset_witness_realized_through_R_mlx.py 2026-07-10)."""
from __future__ import annotations

import json

from tac.witness_control import telemetry_binding as tb


# ── row factories (mirror the trainer's actual schemas) ───────────────────────────────────────────
def _loss_terms(ep: int, *, gnorm: float, chroma: float = 0.0, pose: float = 0.1,
                total: float = 1.0) -> dict:
    return {"stage": "loss_terms", "ep": ep, "accum_batch": 0,
            "terms": {"seg": 0.5, "pose": pose, "chroma_boundary": chroma},
            "total": total, "gnorm": gnorm}


def _verdict(ep: int, *, d_seg: float, d_pose: float, implied_s: float = 0.5) -> dict:
    return {"stage": "verdict", "epoch": ep, "d_seg": d_seg, "d_pose": d_pose,
            "implied_S": implied_s}


def _stab(grad_clip: float = 0.5) -> dict:
    return {"stage": "witness_stability_resolved", "grad_clip": grad_clip,
            "per_group_grad_clip": True}


def _jb(ep: int, smin: float = 0.01) -> dict:
    return {"stage": "jacobian_basin", "epoch": ep, "median_sigma_min": smin,
            "sigma_min_plateau_est": smin, "would_have_fired": False}


# ── parsing ───────────────────────────────────────────────────────────────────────────────────────
def test_parse_log_lines_skips_wrappers_and_prefixes() -> None:
    lines = [
        "[admission-guard] admission OK",
        'SAFE_RUN {"label": "x", "status": "timeout"}',          # no "stage" -> skipped
        json.dumps(_stab()),
        "not json at all",
        json.dumps(_verdict(10, d_seg=0.1, d_pose=0.01)),
    ]
    rows = tb.parse_log_lines(lines)
    assert [r["stage"] for r in rows] == ["witness_stability_resolved", "verdict"]


def test_load_run_rows_reads_nested_run_logs(tmp_path) -> None:
    (tmp_path / "run.log").write_text(json.dumps(_stab()) + "\n")
    sub = tmp_path / "dry_start"
    sub.mkdir()
    (sub / "run.log").write_text(json.dumps(_verdict(5, d_seg=0.2, d_pose=0.02)) + "\n")
    rows = tb.load_run_rows(tmp_path)
    assert {r["stage"] for r in rows} == {"witness_stability_resolved", "verdict"}


def test_load_run_rows_tail_bytes_bounds_read(tmp_path) -> None:
    lines = [json.dumps(_verdict(e, d_seg=0.1, d_pose=0.01)) for e in range(200)]
    (tmp_path / "run.log").write_text("\n".join(lines) + "\n")
    full = tb.load_run_rows(tmp_path)
    tail = tb.load_run_rows(tmp_path, tail_bytes=500)
    assert len(full) == 200
    assert 0 < len(tail) < 200


# ── (a) event decision table ──────────────────────────────────────────────────────────────────────
def test_event_decision_table_normalizes_fire_and_cap_rows() -> None:
    rows = [
        {"stage": "start_event_fired", "transition": "muon", "sensor": "powerlaw_meat",
         "epoch": 700, "fired_by": "event", "cap": 726, "sensor_data_epoch": 675,
         "sensor_lag_epochs": 25},
        {"stage": "cap_fired_before_event", "transition": "chroma", "sensor": "annulus_plateau",
         "epoch": 450, "cap": 450, "fired_by": "cap"},
        {"stage": "verdict", "epoch": 10, "d_seg": 0.1, "d_pose": 0.01},  # not an event row
    ]
    table = tb.event_decision_table(rows)
    assert len(table) == 2
    assert table[0]["event"] == "cap_fired_before_event"
    assert table[0]["fired_by"] == "cap"
    fired = table[1]
    assert fired["event"] == "start_event_fired:muon"
    assert fired["sensor"] == "powerlaw_meat"
    assert fired["sensor_lag_epochs"] == 25


# ── (b) amber ─────────────────────────────────────────────────────────────────────────────────────
def test_amber_inert_when_clip_never_binds() -> None:
    rows = [_stab(0.5)] + [_loss_terms(e, gnorm=0.1) for e in range(20)]
    out = tb.amber_binding(rows)
    assert out["verdict"] == "INERT_NEVER_BINDS"
    assert out["clip_binding_frac"] == 0.0
    assert out["per_group_rates_available"] is False


def test_amber_binding_when_clip_sometimes_binds() -> None:
    rows = [_stab(0.5)] + [_loss_terms(e, gnorm=(1.0 if e % 2 else 0.1)) for e in range(20)]
    out = tb.amber_binding(rows)
    assert out["verdict"] == "BINDING"
    assert 0.0 < out["clip_binding_frac"] < 0.9


def test_amber_saturated_when_always_clipping() -> None:
    rows = [_stab(0.5)] + [_loss_terms(e, gnorm=5.0) for e in range(20)]
    assert tb.amber_binding(rows)["verdict"] == "SATURATED_ALWAYS_CLIPS"


def test_amber_unknown_without_stability_row_or_rows() -> None:
    assert tb.amber_binding([])["verdict"] == "UNKNOWN"
    assert tb.amber_binding([_loss_terms(1, gnorm=0.2)])["verdict"] == "UNKNOWN"


def test_amber_counts_gnorm_hijack_alarms() -> None:
    rows = [_stab(0.5), {"stage": "confound_alarm", "alarm": "gnorm_hijack", "ep": 3}]
    rows += [_loss_terms(e, gnorm=0.1) for e in range(10)]
    assert tb.amber_binding(rows)["gnorm_hijack_alarms"] == 1


# ── (c) chroma ────────────────────────────────────────────────────────────────────────────────────
def test_chroma_pending_before_engage() -> None:
    rows = [_loss_terms(e, gnorm=0.1, chroma=0.0) for e in range(10)]
    assert tb.chroma_binding(rows)["verdict"] == "PENDING"


def test_chroma_inert_zero_after_engage_with_zero_term() -> None:
    rows = [{"stage": "seg_chroma_boundary_engage", "epoch": 100}]
    rows += [_loss_terms(e, gnorm=0.1, chroma=0.0) for e in range(100, 120)]
    out = tb.chroma_binding(rows)
    assert out["verdict"] == "INERT_ZERO"
    assert out["engaged_epoch"] == 100


def test_chroma_binding_and_dominating() -> None:
    engage = [{"stage": "seg_chroma_boundary_engage", "epoch": 10}]
    ok = engage + [_loss_terms(e, gnorm=0.1, chroma=0.05, total=1.0) for e in range(10, 30)]
    assert tb.chroma_binding(ok)["verdict"] == "BINDING"
    dom = engage + [_loss_terms(e, gnorm=0.1, chroma=0.6, total=1.0) for e in range(10, 30)]
    assert tb.chroma_binding(dom)["verdict"] == "DOMINATING"


def test_chroma_ignores_pre_engage_rows() -> None:
    rows = [_loss_terms(e, gnorm=0.1, chroma=0.9) for e in range(0, 10)]  # pre-engage domination
    rows += [{"stage": "seg_chroma_boundary_engage", "epoch": 10}]
    rows += [_loss_terms(e, gnorm=0.1, chroma=0.05) for e in range(10, 30)]
    assert tb.chroma_binding(rows)["verdict"] == "BINDING"


# ── (d) pose gate ─────────────────────────────────────────────────────────────────────────────────
def test_pose_gate_ok_when_sensor_tracks_verdicts() -> None:
    rows = [_jb(e) for e in range(25, 250, 25)] + [_verdict(e, d_seg=0.1, d_pose=0.01)
                                                   for e in range(25, 250, 25)]
    out = tb.pose_gate_health(rows)
    assert out["verdict"] == "OK"
    assert out["stalled"] is False


def test_pose_gate_stalled_when_sensor_stops_but_verdicts_continue() -> None:
    rows = [_jb(e) for e in (25, 50, 75)] + [_verdict(e, d_seg=0.1, d_pose=0.01)
                                             for e in range(25, 600, 25)]
    out = tb.pose_gate_health(rows)
    assert out["stalled"] is True
    assert out["verdict"] == "DETECTOR_STALLED"


def test_pose_gate_alarms_surface() -> None:
    rows = [_jb(25), _jb(50),
            {"stage": "confound_alarm", "alarm": "pose_finish_gate_canary_failed", "ep": 60}]
    out = tb.pose_gate_health(rows)
    assert out["verdict"] == "ALARMED"
    assert out["alarms"][0]["alarm"] == "pose_finish_gate_canary_failed"


def test_pose_gate_no_sensor_rows() -> None:
    assert tb.pose_gate_health([])["verdict"] == "NO_SENSOR_ROWS"


def test_pose_gate_grace_on_early_run_then_stall_after_grace() -> None:
    # 1 verdict + 0 sensor rows = too early to call (the dry-start false-positive fix) ...
    early = [_verdict(25, d_seg=0.1, d_pose=0.01)]
    assert tb.pose_gate_health(early)["verdict"] == "NO_SENSOR_ROWS"
    # ... but many verdicts with a never-emitting sensor IS the silent-crash class.
    late = [_verdict(e, d_seg=0.1, d_pose=0.01) for e in range(25, 300, 25)]
    assert tb.pose_gate_health(late)["verdict"] == "DETECTOR_STALLED"


def test_pose_gate_disabled_is_not_stalled() -> None:
    rows = [{"stage": "jacobian_basin_disabled", "epoch": 0}]
    rows += [_verdict(e, d_seg=0.1, d_pose=0.01) for e in range(25, 300, 25)]
    out = tb.pose_gate_health(rows)
    assert out["verdict"] == "DISABLED"
    assert out["stalled"] is None


# ── (e) EMA lag ───────────────────────────────────────────────────────────────────────────────────
def test_ema_lag_diverging_signature() -> None:
    # verdict d_pose RISES while live pose term FALLS (run-1 confound signature)
    rows = [_verdict(e, d_seg=0.1, d_pose=0.001 + e * 1e-4) for e in range(10, 90, 10)]
    rows += [_loss_terms(e, gnorm=0.1, pose=1.0 - e * 0.01) for e in range(10, 90, 10)]
    out = tb.ema_lag(rows)
    assert out["verdict"] == "EMA_LAG_DIVERGING"
    assert out["diverging"] is True


def test_ema_lag_consistent_when_both_fall() -> None:
    rows = [_verdict(e, d_seg=0.1, d_pose=0.01 - e * 1e-4) for e in range(10, 90, 10)]
    rows += [_loss_terms(e, gnorm=0.1, pose=1.0 - e * 0.01) for e in range(10, 90, 10)]
    assert tb.ema_lag(rows)["verdict"] == "CONSISTENT"


def test_ema_lag_unknown_on_short_history() -> None:
    assert tb.ema_lag([_verdict(1, d_seg=0.1, d_pose=0.01)])["verdict"] == "UNKNOWN"


# ── (f) terminal band / D27b ──────────────────────────────────────────────────────────────────────
def test_terminal_band_not_ready_before_muon() -> None:
    rows = [_verdict(e, d_seg=0.1, d_pose=0.01) for e in range(25, 300, 25)]
    out = tb.terminal_band_status(rows)
    assert out["in_basin"] is False
    assert out["d27b_ready"] is False


def test_terminal_band_ready_on_muon_plus_plateau() -> None:
    rows = [{"stage": "muon_finisher_switch", "epoch": 726}]
    rows += [_verdict(e, d_seg=0.005, d_pose=0.001) for e in range(726, 1000, 25)]  # flat
    out = tb.terminal_band_status(rows)
    assert out["in_basin"] is True
    assert out["d27b_ready"] is True
    assert out["terminal_band"] is False  # no tail stop / polyak yet


def test_terminal_band_full_on_tail_stop() -> None:
    rows = [{"stage": "muon_finisher_switch", "epoch": 726},
            {"stage": "tail_powerplay_stop", "epoch": 950, "cycle": 1,
             "net_marginal_s_per_ep": 1e-6, "reason": "marginal_below_floor"}]
    rows += [_verdict(e, d_seg=0.005, d_pose=0.001) for e in range(726, 1000, 25)]
    assert tb.terminal_band_status(rows)["terminal_band"] is True


def test_terminal_band_not_ready_while_still_descending() -> None:
    rows = [{"stage": "muon_finisher_switch", "epoch": 100}]
    rows += [_verdict(100 + i * 25, d_seg=0.01 * (0.9 ** i), d_pose=0.001) for i in range(12)]
    assert tb.terminal_band_status(rows)["in_basin"] is False


# ── (h) tail endpoints ────────────────────────────────────────────────────────────────────────────
def test_tail_cycle_endpoints_join_boundaries_to_verdicts() -> None:
    rows = [_verdict(e, d_seg=0.01 - e * 1e-6, d_pose=0.001) for e in range(25, 500, 25)]
    rows += [{"stage": "tail_cycle_begin", "epoch": 200, "cycle": 1, "tau": 0.3, "lr": 1e-4},
             {"stage": "tail_cycle_begin", "epoch": 400, "cycle": 2, "tau": 0.15, "lr": 5e-5}]
    eps = tb.tail_cycle_endpoints(rows)
    assert [e["segment_end"] for e in eps] == ["tail_cycle_begin", "tail_cycle_begin", "final"]
    assert eps[0]["boundary_epoch"] == 200
    assert eps[0]["endpoint_epoch"] == 200
    assert eps[1]["cycle_next"] == 2
    assert eps[2]["segment_end"] == "final"
    for e in eps:
        assert e["best_d_seg_in_segment"] <= e["d_seg"] + 1e-12


def test_tail_cycle_endpoints_empty_without_verdicts() -> None:
    assert tb.tail_cycle_endpoints([{"stage": "tail_cycle_begin", "epoch": 10, "cycle": 1}]) == []


# ── combined ──────────────────────────────────────────────────────────────────────────────────────
def test_audit_rows_sections_fail_open(monkeypatch) -> None:
    def boom(_rows):
        raise RuntimeError("synthetic")
    monkeypatch.setattr(tb, "amber_binding", boom)
    out = tb.audit_rows([_verdict(10, d_seg=0.1, d_pose=0.01)])
    assert "error" in out["amber"]
    assert isinstance(out["events"], list)  # other sections unaffected


def test_format_summary_one_line() -> None:
    out = tb.audit_rows([_stab(0.5)] + [_loss_terms(e, gnorm=0.1) for e in range(20)])
    line = tb.format_summary(out)
    assert "amber=INERT_NEVER_BINDS" in line
    assert "\n" not in line
    assert "NON-PROMOTABLE" in line


def test_audit_on_real_drystart_log_if_present() -> None:
    """Smoke against the REAL sealed dry-start log (read-only) when it exists on this machine."""
    from pathlib import Path
    run_dir = Path("experiments/results/__v752_drystart_final__")
    if not run_dir.exists():  # pragma: no cover - machine-dependent
        return
    rows = tb.load_run_rows(run_dir)
    assert rows, "dry-start run.log should parse to rows"
    audit = tb.audit_rows(rows)
    assert tb.format_summary(audit)
