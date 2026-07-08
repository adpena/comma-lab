"""Tests for the schema-driven run introspection layer (#352, tools/witness_run_introspect).

Fixtures are SELF-CONTAINED: a minimal-but-faithful v6 ``launch.sh`` +
``constants_manifest.json`` + ``run.log`` + ``costate_shadow.jsonl`` are written into a
tmp dir per test (never a reference to the sacred live run dir). Coverage:

  * schedule classification EVENT / FIXED with live arm state (pending/cap)
  * LawRef constants manifest -> ranked provenance-ladder table
  * costate controller tail-parse (λ traces + axis EV + duty queue)
  * confound-immune liveness row + frozen/skip/accept alarms
  * mem_probe telemetry (cap + series + peak)
  * fired-event detection (setup-lever rows are NOT events)
  * planned τ / β / LR curves — FAITHFUL to the trainer's own formulas
  * graceful degradation over pre-v6 / missing / empty run dirs
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import witness_run_introspect as wri  # noqa: E402

# ── a minimal, faithful v6 launch.sh (the crucible_v6 run-1 shape) ──
V6_LAUNCH = """#!/usr/bin/env bash
set -euo pipefail
# tac-config-family: crucible_v6
cd /repo
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 \\
  .venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py \\
  --out-dir {OUT} \\
  --gt-cache x.npz \\
  --num-pairs 600 \\
  --epochs 3000 \\
  --anneal-epochs 3000 \\
  --eval-every 25 \\
  --curriculum \\
  --tau-softplus-start-epoch 300 \\
  --l7-start-epoch 3000 \\
  --muon-start-epoch 726 \\
  --softmax-temp-start 1.0 \\
  --softmax-temp-end 0.31 \\
  --tau-anneal-shape cosine_hold \\
  --tau-hold-frac 0.2 \\
  --hosc-beta 1.0 \\
  --hosc-beta-end 10.0 \\
  --hosc-beta-anneal linear \\
  --lr 1e-3 \\
  --lr-end 1e-4 \\
  --lr-anneal-epochs 1000 \\
  --lr-hold-frac 1.0 \\
  --warmup-epochs 1 \\
  --curriculum-event-triggered \\
  --curriculum-nucleus-guard
"""

V6_MANIFEST = {
    "schema": "constants_manifest.v1", "config_family": "crucible_v6",
    "generated_at": "20260708T095730Z",
    "constants": {
        "softmax_temp_end": {
            "value": 0.31, "equation_id": "tau_end_knee_launch_v1",
            "ladder_class": "measured_anchor", "fallback_used": False,
            "inputs": [{"name": "launch_tau", "kind": "anchor", "value": 0.31,
                        "source": "/repo/.omx/research/tau_knee.json",
                        "sha256": "9898d8d76eaa7ce1f55e6272514690fd4fe7f01e28e525dcfad9af72ee719399",
                        "provenance": "P-TAU2 knee probe launch_tau 0.31."}],
            "warnings": [],
        },
        "hosc_beta_end": {
            "value": 10.0, "equation_id": "hosc_beta_fireband_pin_v1",
            "ladder_class": "derived_at_config", "fallback_used": False,
            "inputs": [{"name": "beta_end", "kind": "literal", "value": 10.0,
                        "source": "literal", "sha256": None,
                        "provenance": "v6.3 β-pin 10.0."}],
            "warnings": [],
        },
    },
}

# a costate_shadow.jsonl row (the #247 SENSE schema, trimmed to the fields we read)
V6_COSTATE = {
    "actuation": "NONE", "classification": None, "epoch": None,
    "pointer": "pointer 0.19110 UNMOVED", "axis": "[macOS advisory] NON-PROMOTABLE",
    "ts": "2026-07-08T10:12:41Z", "state": {"n_verdicts": 3},
    "costates": [
        {"name": "lambda_d_seg", "value": 100.0, "band": [100.0, 100.0],
         "status": "ANALYTIC", "units": "S per unit d_seg", "method": "∂S/∂d_seg"},
        {"name": "lambda_d_pose", "value": None, "band": None,
         "status": "UNIDENTIFIABLE", "units": "S per unit d_pose", "method": "diverges"},
    ],
    "duty_to_measure": [
        {"lever": "A", "state": "never-fired"}, {"lever": "B", "state": "never-fired"},
        {"lever": "C", "state": "measured"},
    ],
    "duty_ranked": [{"candidate_lever": "AdamBeta2", "measurement_cost_epochs": 0}],
    "probe_queue": [{"costate": "lambda_d_pose"}],
    "producer_signals": [
        {"producer": "sensitivity_map.axis_weights",
         "signal": {"seg": 1.0, "pose": 2.71, "rate": 1.0, "mixed": 1.5}}],
    "recommendations": [],
}


def _verdict(ep, **kw):
    row = {"stage": "verdict", "epoch": ep, "d_seg": 0.5, "d_pose": 1.0,
           "blob_bytes": 1000, "implied_S": 0.3, "seg_form": "CE",
           "accepted_frac": 1.0, "accepted_batches": 8, "skipped_batches": 0,
           "weights_stepped": True, "frozen_epoch": False, "ep_loss": 0.1,
           "ts": "2026-07-08T10:00:00Z"}
    row.update(kw)
    return json.dumps(row)


def _write_v6(tmp: Path, *, manifest=True, costate=True, verdicts=True,
              mem=True, events=False, launch=True) -> Path:
    rd = tmp / "levelset_n600_crucible_v6_run1_20260708T095730Z"
    rd.mkdir(parents=True, exist_ok=True)
    if launch:
        (rd / "launch.sh").write_text(V6_LAUNCH.replace("{OUT}", str(rd)))
    if manifest:
        (rd / "constants_manifest.json").write_text(json.dumps(V6_MANIFEST))
    if costate:
        (rd / "costate_shadow.jsonl").write_text(json.dumps(V6_COSTATE) + "\n")
    lines = ['{"stage": "gt", "n_pairs": 600}',
             '{"stage": "lane_render_band"}',   # setup-lever row (NOT a fired event)
             '{"stage": "island_seed"}']
    if mem:
        lines.append('{"stage": "mem_probe", "phase": "after_cf_mx_cache_build", "rss_gib": 60.62, "mlx_active_gib": 46.31, "mlx_cache_gib": 0.86}')
        lines.append('{"stage": "mem_probe", "phase": "before_v0_verdict", "rss_gib": 60.9, "mlx_active_gib": 46.3}')
    if events:
        lines.append('{"stage": "muon_finisher_switch", "epoch": 726}')
        lines.append('{"stage": "curriculum_transition_fired", "epoch": 312}')
    if verdicts:
        lines.append(_verdict(0, weights_stepped=False, accepted_frac=None, ep_loss=None))
        lines.append(_verdict(25))
    (rd / "run.log").write_text("\n".join(lines) + "\n")
    return rd


# ─────────────────────────── schedule classification ───────────────────────────
def test_schedule_event_stage_classified(tmp_path):
    rd = _write_v6(tmp_path)
    d = wri.introspect_run(rd, log_paths=[str(rd / "run.log")])
    sch = d["schedule"]
    assert sch and sch["ok"] and sch["event_triggered"] is True
    tau = next(s for s in sch["stages"] if s["name"] == "tau")
    assert tau["klass"] == "event" and tau["mode"] == "event"
    assert tau["status"] == "pending" and tau["cap"] == 300
    assert tau["trigger"]  # a DSL-derived trigger description, never hand-typed


def test_schedule_fixed_stages_classified(tmp_path):
    rd = _write_v6(tmp_path)
    sch = wri.introspect_run(rd)["schedule"]
    ce = next(s for s in sch["stages"] if s["name"] == "CE")
    muon = next(s for s in sch["stages"] if s["name"] == "Muon")
    assert ce["klass"] == "fixed" and ce["start"] == 0
    assert muon["klass"] == "fixed" and muon["start"] == 726


def test_schedule_missing_launch_sh_reason(tmp_path):
    rd = _write_v6(tmp_path, launch=False, manifest=False, costate=False,
                   verdicts=False, mem=False)
    sch = wri.introspect_run(rd)["schedule"]
    # no launch.sh -> read-back reports ok False with a reason (fail-open, never raise)
    assert sch is None or sch.get("ok") is False


# ─────────────────────────── constants manifest ───────────────────────────
def test_constants_manifest_ranked_and_provenance(tmp_path):
    rd = _write_v6(tmp_path)
    c = wri.introspect_run(rd)["constants"]
    assert c and c["count"] == 2 and c["config_family"] == "crucible_v6"
    # sorted by ladder tier (derived_at_config tier 1 before measured_anchor tier 2)
    assert [r["name"] for r in c["rows"]] == ["hosc_beta_end", "softmax_temp_end"]
    ste = next(r for r in c["rows"] if r["name"] == "softmax_temp_end")
    assert ste["value"] == 0.31 and ste["equation_id"] == "tau_end_knee_launch_v1"
    assert ste["ladder_label"] == "measured anchor"
    assert ste["anchor_sha"] == "9898d8d76eaa"  # 12-char truncation
    assert ste["provenance"].startswith("P-TAU2")


def test_constants_absent_is_none(tmp_path):
    rd = _write_v6(tmp_path, manifest=False)
    assert wri.introspect_run(rd)["constants"] is None


# ─────────────────────────── controller / costate ───────────────────────────
def test_controller_tail_parse(tmp_path):
    rd = _write_v6(tmp_path)
    c = wri.introspect_run(rd)["controller"]
    assert c and c["ok"] and c["n_verdicts"] == 3
    assert len(c["costates"]) == 2
    lam = next(x for x in c["costates"] if x["name"] == "lambda_d_seg")
    assert lam["value"] == 100.0 and lam["status"] == "ANALYTIC"
    assert c["axis_ev"] == {"seg": 1.0, "pose": 2.71, "rate": 1.0, "mixed": 1.5}
    assert c["duty_owed"] == 3 and c["duty_never_fired"] == 2
    assert c["probe_queue"] == 1
    assert "0.19110" in c["pointer"]


def test_controller_reads_last_row_only(tmp_path):
    rd = _write_v6(tmp_path, costate=False)
    p = rd / "costate_shadow.jsonl"
    older = dict(V6_COSTATE); older["state"] = {"n_verdicts": 1}
    p.write_text(json.dumps(older) + "\n" + json.dumps(V6_COSTATE) + "\n")
    c = wri.read_controller(rd)
    assert c["n_verdicts"] == 3  # the LAST row wins (tail parse)


def test_controller_absent_is_none(tmp_path):
    rd = _write_v6(tmp_path, costate=False)
    assert wri.introspect_run(rd)["controller"] is None


# ─────────────────────────── liveness (confound-immune) ───────────────────────────
def test_liveness_row_no_false_frozen_alarm(tmp_path):
    rd = _write_v6(tmp_path)
    lv = wri.introspect_run(rd)["liveness"]
    # frozen_epoch=False is NOT frozen -> no alarm (the boolean-vs-epoch bug guard)
    assert lv["epoch"] == 25 and lv["frozen_epoch"] is False
    assert "frozen_epoch" not in lv["alarms"]
    assert lv["weights_stepped"] is True and lv["accepted_frac"] == 1.0


def test_liveness_alarms_fire(tmp_path):
    rd = _write_v6(tmp_path, verdicts=False)
    (rd / "run.log").write_text(
        '{"stage": "gt"}\n' +
        _verdict(50, frozen_epoch=True, ep_loss=0.0, accepted_frac=0.2) + "\n")
    lv = wri.read_liveness_row(rd)
    assert set(["frozen_epoch", "ep_loss_zero", "low_accepted_frac"]).issubset(set(lv["alarms"]))


def test_liveness_absent_until_first_verdict(tmp_path):
    rd = _write_v6(tmp_path, verdicts=False, mem=False, events=False)
    assert wri.read_liveness_row(rd) is None


# ─────────────────────────── mod_dim_dynamics ───────────────────────────
def _mdd_row(ep, eff, k90, seg="stageTau", tau=1.0):
    return json.dumps({
        "stage": "mod_dim_dynamics", "epoch": ep, "seg_form": seg, "tau": tau, "mod_dim": 32,
        "spectrum": {"effective_rank": eff, "k90": k90, "k99": k90 + 4,
                     "spectral_entropy_norm": 0.7},
        "per_dim": {"variance": [0.1] * 32, "film_consumption": [0.2] * 32,
                    "xi_max_r2": [0.05] * 32},
        "latent_xi_cca": {"canonical_corrs": [0.3], "mean": 0.2, "max": 0.3},
        "k90_truncate_bytes_estimate": 5000, "code_bytes_full": 8000,
        "axis": "[macOS-numpy advisory] NON-PROMOTABLE"})


def test_mod_dim_dynamics_reader(tmp_path):
    rd = _write_v6(tmp_path, verdicts=False, mem=False, events=False)
    (rd / "run.log").write_text(
        '{"stage": "gt"}\n' + _mdd_row(10, 12.0, 15) + "\n" + _mdd_row(20, 17.8, 20) + "\n")
    m = wri.introspect_run(rd)["mod_dim_dynamics"]
    assert m and m["count"] == 2
    assert m["latest"]["effective_rank"] == 17.8 and m["latest"]["k90"] == 20  # autopsy anchor
    assert m["effective_rank_series"] == [[0, 12.0], [1, 17.8]]  # rank grows with the anneal
    assert m["latest"]["k90_truncate_bytes_estimate"] == 5000


def test_mod_dim_dynamics_error_row_surfaced(tmp_path):
    rd = _write_v6(tmp_path, verdicts=False, mem=False, events=False)
    (rd / "run.log").write_text(
        '{"stage": "mod_dim_dynamics", "epoch": 5, "seg_form": "ce", "error": "ValueError: boom"}\n')
    m = wri.read_mod_dim_dynamics(rd)
    assert m["count"] == 1 and m["latest"]["error"] == "ValueError: boom"


def test_mod_dim_dynamics_absent_is_none(tmp_path):
    rd = _write_v6(tmp_path, verdicts=False, mem=False, events=False)
    (rd / "run.log").write_text('{"stage": "gt"}\n')
    assert wri.read_mod_dim_dynamics(rd) is None


# ─────────────────────────── mem_probe ───────────────────────────
def test_mem_probe_series_and_peak(tmp_path):
    rd = _write_v6(tmp_path)
    m = wri.introspect_run(rd)["mem"]
    assert m and m["count"] == 2 and m["peak_rss_gib"] == 60.9
    assert len(m["series"]) == 2 and m["latest"]["phase"] == "before_v0_verdict"


def test_mem_probe_series_capped(tmp_path):
    rd = _write_v6(tmp_path, mem=False, verdicts=False, events=False)
    rows = ['{"stage": "gt"}']
    for i in range(200):
        rows.append('{"stage": "mem_probe", "phase": "p%d", "rss_gib": %f, "mlx_active_gib": 40.0}'
                     % (i, 50.0 + i * 0.01))
    (rd / "run.log").write_text("\n".join(rows) + "\n")
    m = wri.read_mem_probes(rd)
    assert m["count"] == 200 and len(m["rows"]) == 64 and len(m["series"]) == 64


def test_mem_absent_is_none(tmp_path):
    rd = _write_v6(tmp_path, mem=False, verdicts=False, events=False)
    (rd / "run.log").write_text('{"stage": "gt"}\n')
    assert wri.read_mem_probes(rd) is None


# ─────────────────────────── fired events ───────────────────────────
def test_events_are_transitions_not_setup_levers(tmp_path):
    rd = _write_v6(tmp_path, events=True)
    ev = wri.introspect_run(rd)["events"]
    kinds = {e["stage"] for e in ev}
    assert "muon_finisher_switch" in kinds and "curriculum_transition_fired" in kinds
    # setup-lever config rows (lane_render_band / island_seed) are NOT fired events
    assert "lane_render_band" not in kinds and "island_seed" not in kinds


def test_events_absent_when_none_fired(tmp_path):
    rd = _write_v6(tmp_path, events=False)
    assert wri.introspect_run(rd)["events"] is None


# ─────────────────────────── planned curves (faithful) ───────────────────────────
def test_curves_present_and_endpoints(tmp_path):
    rd = _write_v6(tmp_path)
    cur = wri.introspect_run(rd)["curves"]
    assert cur and cur["muon_start"] == 726
    tau = cur["curves"]["tau"]
    assert tau["start"] == 1.0 and tau["end"] == 0.31 and tau["shape"] == "cosine_hold"
    assert tau["points"][0] == [1, 1.0]          # ep1 == start exactly
    assert abs(tau["points"][-1][1] - 0.31) < 1e-6  # ep3000 == end (held)


def test_tau_curve_faithful_to_trainer_formula(tmp_path):
    """The τ port must reproduce the trainer's ``_softmax_temp_for_epoch`` (cosine_hold)."""
    rd = _write_v6(tmp_path)
    a = wri.trainer_args_from_run(rd)
    # reference: the trainer's exact cosine_hold at a mid-window epoch (hold_frac 0.2)
    ae, start, end, hf = 3000, 1.0, 0.31, 0.2
    for ep in (1, 150, 601, 1500, 3000):
        prog = (ep - 1) / (ae - 1)
        if hf < 1.0 and prog >= hf:
            ref = end
        else:
            p = prog / hf if hf < 1.0 else prog
            ref = end + 0.5 * (start - end) * (1 + math.cos(math.pi * p))
        got = wri._tau_at(ep, a)
        assert abs(got - ref) < 1e-9, (ep, got, ref)


def test_tau_hold_and_muon_freeze(tmp_path):
    rd = _write_v6(tmp_path)
    cur = wri.introspect_run(rd)["curves"]
    tau = cur["curves"]["tau"]
    assert tau["hold_epoch"] == 601  # reaches floor at 20% of the 3000-ep window
    # past the Muon freeze the curve HOLDS the muon-start value (finisher freeze)
    a = wri.trainer_args_from_run(rd)
    frozen_val = wri._tau_at(726, a)
    tail = [v for (ep, v) in tau["points"] if ep > 726]
    assert tail and all(abs(v - frozen_val) < 1e-6 for v in tail)


def test_lr_curve_warmup_and_note(tmp_path):
    rd = _write_v6(tmp_path)
    lr = wri.introspect_run(rd)["curves"]["curves"]["lr"]
    assert lr["start"] == 0.001 and lr["end"] == 0.0001 and lr["warmup"] == 1
    assert "muon-lr" in lr["note"].lower()  # honest: base schedule ≠ muon params


def test_beta_curve_linear(tmp_path):
    rd = _write_v6(tmp_path)
    a = wri.trainer_args_from_run(rd)
    # linear β over the shared anneal denominator: midpoint ≈ average of endpoints
    mid = wri._beta_at(1 + (3000 - 1) // 2, a)
    assert abs(mid - 5.5) < 0.05


def test_curves_absent_without_launch(tmp_path):
    rd = _write_v6(tmp_path, launch=False)
    assert wri.planned_curves(rd) is None


# ─────────────────────────── top-level graceful degradation ───────────────────────────
def test_introspect_full_run_all_facets(tmp_path):
    rd = _write_v6(tmp_path, events=True)
    d = wri.introspect_run(rd, log_paths=[str(rd / "run.log")])
    assert d["ok"]
    for k in ("schedule", "constants", "controller", "liveness", "mem", "events", "curves"):
        assert d[k] is not None, k


def test_introspect_missing_dir_ok_false(tmp_path):
    d = wri.introspect_run(tmp_path / "nope")
    assert d["ok"] is False and d["schedule"] is None


def test_introspect_none_rundir():
    assert wri.introspect_run(None)["ok"] is False


def test_introspect_pre_v6_only_launch(tmp_path):
    # a pre-v6 dir: launch.sh only, no manifest/costate/mem -> schedule+curves present,
    # constants/controller/mem/events absent, and NO crash.
    rd = _write_v6(tmp_path, manifest=False, costate=False, mem=False,
                   events=False, verdicts=False)
    (rd / "run.log").write_text('{"stage": "gt"}\n')
    d = wri.introspect_run(rd)
    assert d["ok"] and d["curves"] is not None and d["schedule"]["ok"]
    assert d["constants"] is None and d["controller"] is None and d["mem"] is None


def test_bounded_tail_reads_end_of_large_log(tmp_path):
    p = tmp_path / "big.log"
    filler = "\n".join('{"stage": "noise", "i": %d}' % i for i in range(50_000))
    p.write_text(filler + "\n" + _verdict(999) + "\n")
    txt = wri._tail_text(p, max_bytes=8192)
    assert len(txt) <= 8192 and '"epoch": 999' in txt  # the tail, not the whole file
