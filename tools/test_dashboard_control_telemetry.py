"""Tests for ``tools/dashboard_control_telemetry.py`` (DASHBOARD PASS 2026-07-04).

Covers the pure control-system telemetry fns the dashboards render: stage-row parsing
(fired + legacy transitions, closed_loop, structured_init/lane_prior), the ep0
NUCLEATION GATE (incl. the REAL #205 FAIL row shape), the eikonal effective-weight
derivation (verbatim trainer-math replica: byte-identical base case, cosine-eased
step, fired-boundary tracking, bounded bump composition), per-stage slope decoupling
(the #205 surrogate↔verdict signature), the classification lane's DO-NOT-FORK parity
with ``witness_control_monitor.classify_trajectory``, run-role (config currency),
the lever status board, and the inline-HTML renderers.

Run: ``.venv/bin/python -m pytest tools/test_dashboard_control_telemetry.py``
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dashboard_control_telemetry as dct  # noqa: E402
import witness_control_monitor as wcm  # noqa: E402

# The REAL #205 ep0 structured_init row (verbatim shape from the live run.log) — the
# measured nucleation-FAIL anchor (part_frac["1"]=0.0, lane_px=0, mode=replace).
_205_STRUCTURED_INIT = json.loads(
    '{"stage": "structured_init", "roles": {"road": 0, "lane": 1, "sky": 2, "movable": 3,'
    ' "hood": 4}, "pretrain_direct_argmax_disagree_vs_part": 0.00035, "steps": 600,'
    ' "lr": 0.005, "sky_px": 97867, "hood_px": 49970, "include_lane": true, "lane_px": 0,'
    ' "lane_static_mask_px": 0, "lane_mean_iou": 0.0, "part_frac": {"0": 0.248, "1": 0.0,'
    ' "2": 0.4978, "3": 0.0, "4": 0.2542}}')
_205_LANE_PRIOR = {"stage": "lane_prior_phi1", "active": True, "mode": "replace"}

_205_FLAGS = {"lane-prior-phi1-mode": "replace", "eikonal-weight": "0.01",
              "curriculum": True, "tau-softplus-start-epoch": "300",
              "muon-start-epoch": "726", "mod-dim": "32", "island-dilate-px": "1"}
_FRESH_FLAGS = {"lane-prior-phi1-mode": "paint", "seed-islands": True,
                "eikonal-weight": "0.05", "eikonal-weight-end": "0.10",
                "curriculum": True, "curriculum-event-triggered": True,
                "closed-loop-control": True, "film-stiefel": True,
                "muon-warm-start-momentum": True, "muon-lr-final-frac": "0.1",
                "tau-softplus-start-epoch": "300",
                "stage-transition-rewarmup-epochs": "20", "mod-dim": "19",
                "bank-n-scales": "6", "muon-start-epoch": "726",
                "closed-loop-eikonal-max": "0.20"}


def _verdicts_205_like():
    """Synthetic verdict series with the #205 signature: CE descending (coupled), tau
    with d_seg RISING while ep_loss FALLS (decoupled = erosion)."""
    rows = []
    for i, ep in enumerate(range(25, 301, 25)):
        rows.append({"stage": "verdict", "epoch": ep, "seg_form": "ce",
                     "d_seg": 0.010 - 0.0004 * i, "ep_loss": 700.0 - 40.0 * i})
    for j, ep in enumerate(range(325, 476, 25)):
        rows.append({"stage": "verdict", "epoch": ep, "seg_form": "tau_softplus",
                     "d_seg": 0.00475 + 3e-4 * j, "ep_loss": 148.0 - 3.0 * j})
    return rows


# ── parse_stage_rows ─────────────────────────────────────────────────────────
def test_parse_stage_rows_merges_fired_and_legacy():
    lines = [
        json.dumps({"stage": "curriculum_transition", "epoch": 300,
                    "from_seg_form": "ce", "to_seg_form": "tau_softplus"}),
        json.dumps({"stage": "curriculum_transition_fired", "from": "tau_softplus",
                    "to": "l7_softplus", "epoch": 620, "trigger": "loss_plateau"}),
        "not json at all {",
        json.dumps({"stage": "closed_loop", "epoch": 350, "classification": "converging",
                    "d_seg_slope": -1e-6, "eikonal_bump": 0.0, "action": "none"}),
        json.dumps(_205_STRUCTURED_INIT),
        json.dumps(_205_LANE_PRIOR),
    ]
    out = dct.parse_stage_rows(lines)
    assert [t["trigger"] for t in out["transitions"]] == ["hardcoded", "loss_plateau"]
    assert out["transitions"][0]["to"] == "tau_softplus"
    assert out["transitions"][1] == {"epoch": 620, "from": "tau_softplus",
                                     "to": "l7_softplus", "trigger": "loss_plateau"}
    assert len(out["closed_loop"]) == 1
    assert out["structured_init"]["lane_px"] == 0
    assert out["lane_prior_phi1"]["mode"] == "replace"


def test_parse_stage_rows_dedup_fired_wins_over_legacy():
    lines = [
        json.dumps({"stage": "curriculum_transition", "epoch": 280,
                    "from_seg_form": "ce", "to_seg_form": "tau_softplus"}),
        json.dumps({"stage": "curriculum_transition_fired", "from": "ce",
                    "to": "tau_softplus", "epoch": 280, "trigger": "loss_plateau"}),
    ]
    out = dct.parse_stage_rows(lines)
    assert len(out["transitions"]) == 1
    assert out["transitions"][0]["trigger"] == "loss_plateau"


def test_parse_stage_rows_empty_and_garbage():
    out = dct.parse_stage_rows(["", "garbage", '{"stage": 5}'])
    assert out["transitions"] == [] and out["closed_loop"] == []
    assert out["structured_init"] is None and out["lane_prior_phi1"] is None


# ── nucleation gate ──────────────────────────────────────────────────────────
def test_nucleation_gate_real_205_row_is_FAIL():
    g = dct.nucleation_gate(_205_STRUCTURED_INIT, _205_LANE_PRIOR)
    assert g["state"] == "FAIL"
    assert g["lane_part_frac"] == 0.0
    assert g["lane_px"] == 0 and g["lane_cls"] == 1 and g["mode"] == "replace"


def test_nucleation_gate_pass():
    row = dict(_205_STRUCTURED_INIT)
    row["part_frac"] = {"0": 0.24, "1": 0.0064, "2": 0.49, "3": 0.0, "4": 0.25}
    row["lane_px"] = 1261
    g = dct.nucleation_gate(row, {"mode": "paint"})
    assert g["state"] == "PASS" and abs(g["lane_part_frac"] - 0.0064) < 1e-12
    assert g["mode"] == "paint"


def test_nucleation_gate_unmeasured():
    assert dct.nucleation_gate(None, None)["state"] == "UNMEASURED"
    row = {"stage": "structured_init", "roles": {"lane": 1}}  # no part_frac at all
    assert dct.nucleation_gate(row)["state"] == "UNMEASURED"


# ── eikonal schedule (trainer-math replica) ──────────────────────────────────
def test_scheduled_eikonal_base_cases_byte_identical():
    for ep in (0, 100, 299, 300, 999):
        # end unset -> base always (the trainer's byte-identity contract)
        assert dct.scheduled_eikonal_weight(ep, base=0.01, end=None, step_epoch=300,
                                            ease_epochs=20) == 0.01
        # end == base -> base
        assert dct.scheduled_eikonal_weight(ep, base=0.05, end=0.05, step_epoch=300,
                                            ease_epochs=20) == 0.05
        # curriculum off -> base
        assert dct.scheduled_eikonal_weight(ep, base=0.05, end=0.10, step_epoch=300,
                                            ease_epochs=20, curriculum=False) == 0.05
        # unfired sentinel (None step) -> base
        assert dct.scheduled_eikonal_weight(ep, base=0.05, end=0.10, step_epoch=None,
                                            ease_epochs=20) == 0.05


def test_scheduled_eikonal_step_and_cosine_ease():
    kw = dict(base=0.05, end=0.10, step_epoch=300, ease_epochs=20)
    assert dct.scheduled_eikonal_weight(299, **kw) == 0.05      # pre-step
    assert dct.scheduled_eikonal_weight(320, **kw) == 0.10      # past the ease window
    mid = dct.scheduled_eikonal_weight(310, **kw)               # cosine midpoint
    assert abs(mid - 0.075) < 1e-12
    # exact cosine value at 1/4 of the window
    q = dct.scheduled_eikonal_weight(305, **kw)
    expect = 0.05 + 0.05 * 0.5 * (1.0 - math.cos(math.pi * 0.25))
    assert abs(q - expect) < 1e-12
    # ease 0 -> immediate step
    assert dct.scheduled_eikonal_weight(300, base=0.05, end=0.10, step_epoch=300,
                                        ease_epochs=0) == 0.10


def test_effective_eikonal_series_flat_for_205():
    ser = dct.effective_eikonal_series(475, _205_FLAGS, [], [])
    assert ser[0] == (0, 0.01) and ser[-1] == (475, 0.01)
    assert all(w == 0.01 for _, w in ser)


def test_effective_eikonal_series_tracks_fired_boundary_and_bump():
    fired = [{"epoch": 250, "from": "ce", "to": "tau_softplus", "trigger": "loss_plateau"}]
    cl = [{"stage": "closed_loop", "epoch": 400, "classification": "diverging_erasing",
           "eikonal_bump": 0.05, "action": "eikonal_bump"}]
    ser = dict(dct.effective_eikonal_series(500, _FRESH_FLAGS, fired, cl))
    assert ser[0] == 0.05
    assert ser[249] == 0.05                    # pre-fired boundary (NOT the hardcoded 300)
    assert abs(ser[270] - 0.10) < 1e-12        # fired step 250 + ease 20 -> end
    assert abs(ser[399] - 0.10) < 1e-12        # pre-bump
    assert abs(ser[400] - 0.15) < 1e-12        # + bump 0.05 (cap 0.20 not binding)
    # bounded: min(sched+bump, max(cl_max, sched))
    cl_big = [{"epoch": 400, "eikonal_bump": 0.50, "action": "eikonal_bump"}]
    ser2 = dict(dct.effective_eikonal_series(500, _FRESH_FLAGS, fired, cl_big))
    assert abs(ser2[450] - 0.20) < 1e-12       # capped at --closed-loop-eikonal-max


def test_effective_eikonal_zero_bump_is_scheduled_exactly():
    fired = [{"epoch": 250, "to": "tau_softplus", "trigger": "loss_plateau"}]
    cl0 = [{"epoch": 300, "eikonal_bump": 0.0, "action": "none"}]
    a = dct.effective_eikonal_series(400, _FRESH_FLAGS, fired, cl0)
    b = dct.effective_eikonal_series(400, _FRESH_FLAGS, fired, [])
    assert a == b  # bump<=0 -> scheduled EXACTLY (byte-identity contract)


# ── per-stage slopes (the decoupling made legible) ──────────────────────────
def test_per_stage_slopes_flags_205_decoupling():
    slopes = dct.per_stage_slopes(_verdicts_205_like())
    by = {s["stage"]: s for s in slopes}
    assert list(by) == ["ce", "tau_softplus"]
    assert by["ce"]["d_seg_slope"] < 0 and not by["ce"]["decoupled"]
    tau = by["tau_softplus"]
    assert tau["d_seg_slope"] > 0 and tau["ep_loss_slope"] < 0 and tau["decoupled"]
    assert tau["ep_lo"] == 325 and tau["ep_hi"] == 475


def test_per_stage_slopes_skips_ep0_priming_row():
    rows = [{"epoch": 0, "d_seg": 0.746}] + _verdicts_205_like()  # ep0 has no seg_form
    slopes = dct.per_stage_slopes(rows)
    assert all(s["stage"] in ("ce", "tau_softplus") for s in slopes)


# ── classification lane (do-NOT-fork parity) ─────────────────────────────────
def test_classification_lane_replay_matches_canonical_monitor():
    rows = _verdicts_205_like()
    lane = dct.classification_lane(rows, [])
    assert lane and all(e["source"] == "monitor_replay" for e in lane)
    # PARITY: the lane's last entry must equal the canonical classifier on the full series
    cv = wcm.classify_trajectory(rows)
    assert lane[-1]["classification"] == cv.classification
    assert lane[-1]["epoch"] == cv.epoch_latest
    assert lane[-1]["d_seg_slope"] == cv.d_seg_slope_per_ep
    # the #205 signature must classify as sustained erosion at the end
    assert lane[-1]["classification"] == wcm.DIVERGING_ERASING


def test_classification_lane_prefers_live_closed_loop_rows():
    cl = [{"epoch": 350, "classification": "converging", "action": "none",
           "eikonal_bump": 0.0, "d_seg_slope": -1e-6},
          {"epoch": 375, "classification": "diverging_erasing", "action": "eikonal_bump",
           "eikonal_bump": 0.05, "d_seg_slope": 2e-5}]
    lane = dct.classification_lane(_verdicts_205_like(), cl)
    assert [e["source"] for e in lane] == ["closed_loop", "closed_loop"]
    assert lane[-1]["action"] == "eikonal_bump" and lane[-1]["eikonal_bump"] == 0.05


# ── run role (config currency) ───────────────────────────────────────────────
def test_run_role_205_fresh_none():
    assert dct.run_role(_205_FLAGS)["role"] == "205"
    assert "erosion" in dct.run_role(_205_FLAGS)["headline"]
    fresh = dct.run_role(_FRESH_FLAGS)
    assert fresh["role"] == "fresh" and "SEEDED" in fresh["headline"]
    assert dct.run_role({})["role"] == "none"
    assert dct.run_role({"eikonal-weight": "0.02"})["role"] == "unknown"


# ── lever status board ───────────────────────────────────────────────────────
def test_lever_status_rows_205_states():
    sr = dct.parse_stage_rows([json.dumps(_205_STRUCTURED_INIT), json.dumps(_205_LANE_PRIOR)])
    ctl = dct.build_control(sr, _205_FLAGS, _verdicts_205_like(), {"muon_start": 726})
    by = {r["lever"]: r for r in ctl["levers"]}
    # CORRECTED gate (FEED-04x): #205's verdicts sit BELOW the lane-frac bound (0.00475<0.0059)
    # which PROVES its CE birthed partial lane — #205's true failure was tau EROSION (carried by
    # the DECOUPLED/DIVERGING_ERASING panels), NOT zero-birth. So paint-seed reads BIRTH, not bad.
    assert by["paint-seed"]["state"] == "ok" and "BIRTH" in by["paint-seed"]["value"]
    assert by["eikonal-ramp"]["state"] == "off"
    assert by["event-trigger"]["state"] == "off"
    assert by["closed-loop"]["state"] == "off"
    assert by["muon warm-start"]["state"] == "off"


def test_lever_status_rows_fresh_prelaunch_awaiting():
    ctl = dct.build_control(dct.parse_stage_rows([]), _FRESH_FLAGS, [], {})
    by = {r["lever"]: r for r in ctl["levers"]}
    assert by["paint-seed"]["state"] == "pending" and by["paint-seed"]["value"] == "awaiting run"
    assert by["eikonal-ramp"]["state"] == "pending"       # armed, not yet stepped
    assert by["event-trigger"]["state"] == "pending"
    assert by["closed-loop"]["state"] == "pending"
    assert by["muon warm-start"]["state"] == "pending"


def test_lever_status_rows_fresh_live_fired():
    lines = [
        json.dumps({"stage": "structured_init", "roles": {"lane": 1},
                    "part_frac": {"1": 0.0064}, "lane_px": 1261}),
        json.dumps({"stage": "lane_prior_phi1", "mode": "paint"}),
        json.dumps({"stage": "curriculum_transition_fired", "from": "ce",
                    "to": "tau_softplus", "epoch": 250, "trigger": "loss_plateau"}),
        json.dumps({"stage": "closed_loop", "epoch": 350, "classification": "converging",
                    "eikonal_bump": 0.0, "bumps_used": 0, "action": "none"}),
    ]
    verdicts = [{"epoch": 750, "seg_form": "tau_softplus", "d_seg": 0.004, "ep_loss": 100.0},
                {"epoch": 775, "seg_form": "tau_softplus", "d_seg": 0.0039, "ep_loss": 99.0}]
    ctl = dct.build_control(dct.parse_stage_rows(lines), _FRESH_FLAGS, verdicts,
                            {"muon_start": 726})
    by = {r["lever"]: r for r in ctl["levers"]}
    # verdicts 0.004/0.0039 < LANE_FRAC_BOUND ⟹ BIRTH_CONFIRMED outranks the init part_frac
    assert by["paint-seed"]["state"] == "ok" and "BIRTH" in by["paint-seed"]["value"]
    assert "0.0039" in by["paint-seed"]["value"]
    assert by["eikonal-ramp"]["state"] == "ok"            # max_epoch 775 >= fired 250
    assert by["event-trigger"]["state"] == "ok" and "loss_plateau" in by["event-trigger"]["detail"]
    assert by["closed-loop"]["state"] == "ok"             # rows present, no action yet
    assert by["muon warm-start"]["value"] == "FIRED"      # 775 >= 726


def test_lever_run_value_booleans_and_absent():
    assert dct.lever_run_value({"film-stiefel": True}, ("film-stiefel",)) == "ON"
    assert dct.lever_run_value({}, ("film-stiefel",)) == "—"
    assert dct.lever_run_value({"mod-dim": "19"}, ("mod-dim",)) == "19"


# ── composition + HTML renderers ─────────────────────────────────────────────
def test_collect_control_never_raises_on_missing_log():
    ctl = dct.collect_control("/nonexistent/run.log", {}, [], None)
    assert ctl["nucleation"]["state"] == "UNMEASURED"
    assert ctl["classification_lane"] == [] and ctl["levers"]


def test_html_renderers_carry_key_tokens_and_advisory():
    sr = dct.parse_stage_rows([json.dumps(_205_STRUCTURED_INIT), json.dumps(_205_LANE_PRIOR)])
    ctl = dct.build_control(sr, _205_FLAGS, _verdicts_205_like(), {"muon_start": 726})
    h = dct.render_control_panel_html(ctl)
    # #205 fixture renders BIRTH (its verdicts prove partial lane) + DECOUPLED (the erosion story)
    assert "nucleation gate" in h and "BIRTH" in h and "DECOUPLED" in h
    assert "NON-PROMOTABLE" in h and "0.19110" in h
    hb = dct.render_lever_board_html(ctl["levers"])
    assert "lever status board" in hb and "paint-seed" in hb
    hc = dct.render_config_lever_table_html(_205_FLAGS, dct.run_role(_205_FLAGS))
    assert "primary lever table" in hc and "--mod-dim" in hc and "#205 erosion" in hc
    hr = dct.render_role_line_html(dct.run_role(_205_FLAGS))
    assert "DIAGNOSED erosion run" in hr
    assert dct.render_role_line_html({"role": "unknown"}) == ""
    # all-in-one wrapper is empty-safe
    assert dct.render_all_html(None, {}, None)


def test_html_escapes_untrusted_values():
    row = {"stage": "structured_init", "roles": {"lane": 1},
           "part_frac": {"1": 0.01}, "lane_px": 5}
    lp = {"stage": "lane_prior_phi1", "mode": "<script>alert(1)</script>"}
    ctl = dct.build_control(dct.parse_stage_rows([json.dumps(row), json.dumps(lp)]),
                            {}, [], None)
    h = dct.render_control_panel_html(ctl)
    assert "<script>" not in h and "&lt;script&gt;" in h


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"PASS {_name}")
    print("ALL PASS")


# ── CORRECTED nucleation-gate semantics (FEED-04x; the UI/UX fix 2026-07-04) ────────────
def test_nucleation_seeded_watch_fresh_init():
    """Fresh-run init: witness partition 0 (EXPECTED) but mechanism MEASURED present
    (paint-signal disagree ≈ band + island-seed support) => SEEDED_WATCH, NOT FAIL."""
    si = {"stage": "structured_init", "roles": {"lane": 1}, "part_frac": {"1": 0.0},
          "lane_px": 0, "pretrain_direct_argmax_disagree_vs_part": 0.00589}
    lp = {"stage": "lane_prior_phi1", "mode": "paint", "lane_band_px": 1261}
    isd = {"stage": "island_seed", "mean_support_frac": 0.02488}
    nuc = dct.nucleation_gate(si, lp, island_seed=isd, verdicts=[])
    assert nuc["state"] == "SEEDED_WATCH"
    assert nuc["pretrain_disagree"] == 0.00589 and nuc["seed_support"] == 0.02488


def test_nucleation_seeded_watch_on_seed_alone():
    """island_seed support alone (no paint signal) still arms the watch — mechanism present."""
    si = {"stage": "structured_init", "roles": {"lane": 1}, "part_frac": {"1": 0.0},
          "lane_px": 0, "pretrain_direct_argmax_disagree_vs_part": 0.0004}
    nuc = dct.nucleation_gate(si, {"mode": "replace", "lane_band_px": 1261},
                              island_seed={"mean_support_frac": 0.02}, verdicts=[])
    assert nuc["state"] == "SEEDED_WATCH"


def test_nucleation_true_205_signature_is_fail():
    """The TRUE #205 init signature: replace no-op (disagree 0.00035 — lane-less target fit
    near-perfectly), NO seed module, partition 0, no sub-bound verdict => FAIL."""
    si = {"stage": "structured_init", "roles": {"lane": 1}, "part_frac": {"1": 0.0},
          "lane_px": 0, "pretrain_direct_argmax_disagree_vs_part": 0.00035}
    nuc = dct.nucleation_gate(si, {"mode": "replace", "lane_band_px": 1261},
                              island_seed=None, verdicts=[{"epoch": 25, "d_seg": 0.0103}])
    assert nuc["state"] == "FAIL"


def test_nucleation_birth_confirmed_by_pixel_arithmetic():
    """Any verdict d_seg < LANE_FRAC_BOUND PROVES partial lane presence (a lane-less witness
    cannot score below the lane pixel fraction) — outranks every init state."""
    si = {"stage": "structured_init", "roles": {"lane": 1}, "part_frac": {"1": 0.0},
          "lane_px": 0, "pretrain_direct_argmax_disagree_vs_part": 0.00035}
    nuc = dct.nucleation_gate(si, {"mode": "replace"}, island_seed=None,
                              verdicts=[{"epoch": 300, "d_seg": 0.004752}])
    assert nuc["state"] == "BIRTH_CONFIRMED"
    assert nuc["birth_epoch"] == 300 and nuc["birth_d_seg"] == 0.004752


def test_nucleation_seeded_watch_renders_amber_not_fail():
    """The UI/UX fix itself: the fresh-run init state must render SEEDED · WATCH (amber),
    never the big red FAIL the pre-correction panel showed."""
    sr = dct.parse_stage_rows([
        json.dumps({"stage": "structured_init", "roles": {"lane": 1}, "part_frac": {"1": 0.0},
                    "lane_px": 0, "pretrain_direct_argmax_disagree_vs_part": 0.00589}),
        json.dumps({"stage": "lane_prior_phi1", "mode": "paint", "lane_band_px": 1261}),
        json.dumps({"stage": "island_seed", "mean_support_frac": 0.02488}),
    ])
    ctl = dct.build_control(sr, {}, [{"epoch": 0, "d_seg": 0.466, "ep_loss": 500.0,
                                      "seg_form": "ce"}], {})
    assert ctl["nucleation"]["state"] == "SEEDED_WATCH"
    h = dct.render_control_panel_html(ctl)
    assert "SEEDED · WATCH" in h and "FAIL" not in h
    assert "paint-signal IN target" in h and "birth watch armed" in h
