#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""costate_digest — the #247 costate controller as a CORE SENSE ORGAN, agent-native.

Operator NON-NEGOTIABLE 2026-07-07 (verbatim): "We must ensure the costate controller
does not require human or manual activation or use and that it is agent native and a
core sense organ and actuator the agent always knows about and uses where and how
optimal and appropriate."

ONE command that renders the controller's current SENSE+DECIDE state as a compact
(~25-line) agent-readable digest. It is AUTO-SURFACED (no memory or manual step needed):
  * SessionStart hook (.claude/settings.json) — stdout injected as session context;
  * the .claude/skills/witness-status skill (canonical check-in) runs it too;
  * the design doc for going deeper is printed in the footer.

READ-ONLY + FAIL-OPEN: every section degrades to "unavailable" — this tool NEVER
crashes a session start and NEVER writes anything. Target wall-clock < 5s.

═══ THE ACTUATION BOUNDARY (binding; do not misread "agent-native" as autonomy) ═══
AUTONOMOUS scope (no GO needed): advisory recommendations, lever-queue (duty-to-
measure) ranking, event-curriculum CONDITION INPUTS, and surfacing this digest.
OPERATOR-GO scope (CONTAINMENT non-negotiable, unchanged): heavy/paid launches,
stopping a live run, and ANY config change to a live run. The shadow controller
package (tac.witness_control) structurally cannot actuate (source-scan-tested).
═══════════════════════════════════════════════════════════════════════════════════

Usage:
  .venv/bin/python tools/costate_digest.py            # human digest
  .venv/bin/python tools/costate_digest.py --json     # machine-readable
  .venv/bin/python tools/costate_digest.py --session-start  # hook mode (always rc 0)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))
# Canonical witness run-artifact CONTRACT — single source of truth for the run's
# emitted filenames (no hardcoding; see the module docstring).
from tac import witness_run_artifacts as _wra  # noqa: E402

_POINTER_JSON = _REPO / ".omx" / "state" / "canonical_frontier_pointer.json"
_DAG_GLOB = str(_REPO / ".omx" / "research" / "sub015_DAG_topaiml_reopen_and_pursuit_plan_*.md")
_DESIGN_DOC = ".omx/research/costate_controller_design_20260705.md"
_DISPATCH_DOC = ".omx/research/organ_regime_conditional_dispatch_436_20260711.md"
_REVIEW_COUNTER = _REPO / ".omx" / "state" / "review_counter.jsonl"
_DUTY_TOP_N = 6
_SHADOW_STALE_S = 2 * 3600.0
# section_ncde cache: {run_dir_str: (log_signature, line, data)} — keyed on run.log
# (mtime_ns, size) so a re-call within the same session reuses the fit instead of re-probing.
_NCDE_CACHE: dict[str, tuple[tuple, str | None, dict | None]] = {}
# section_verdict_trend cache: same run.log-signature keying as _NCDE_CACHE.
_VERDICT_TREND_CACHE: dict[str, tuple[tuple, str | None, dict | None]] = {}


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _last_jsonl_row(path: Path) -> dict | None:
    """Last parseable JSON row of a JSONL file (lenient; None on any failure)."""
    try:
        last = None
        with path.open("rb") as fh:
            for raw in fh:
                if raw.strip():
                    last = raw
        row = json.loads(last) if last else None
        return row if isinstance(row, dict) else None
    except Exception:
        return None


# ─────────────────────────── sections (each fail-open) ───────────────────────────
def section_pointer() -> tuple[str, dict]:
    """AMENDMENT 1 (means-as-ends firewall, operating manual §8.1): the END first."""
    try:
        d = json.loads(_POINTER_JSON.read_text())
        cpu = d.get("our_local_frontier_contest_cpu") or {}
        score = float(cpu["score"])
        since = str(cpu.get("measured_at_utc", ""))[:10] or "?"
        line = f"POINTER {score:.5f} [contest-CPU] UNMOVED since {since} — everything below is means."
        return line, {"score": score, "axis": "contest-CPU", "since": since}
    except Exception as exc:
        return (f"POINTER: unavailable ({type(exc).__name__}) — read .omx/state/canonical_frontier_pointer.json"), {
            "error": str(exc)
        }


def section_live_run() -> tuple[str, dict, Path | None]:
    """Live-run state via the canonical check-in tool (imported, not duplicated)."""
    try:
        import witness_checkin as wc

        procs = wc.find_trainer_procs()
        run_dir, proc, how = wc.pick_run_dir(procs, wc.RESULTS_DEFAULT)
        if run_dir is None or not run_dir.is_dir():
            return "live run: NONE found (no witness run dirs)", {"alive": False}, None
        status = wc.collect_status(run_dir, proc, wc.STALE_AFTER_S_DEFAULT)
        status["discovery"] = how
        return wc.human_line(status), status, run_dir
    except Exception as exc:
        return f"live run: unavailable ({type(exc).__name__}: {exc})", {"error": str(exc)}, None


def section_annulus(run_dir: Path | None) -> tuple[str | None, dict | None]:
    """Annulus/convergence headline from the run's annulus_live.jsonl (#333 SENSE)."""
    if run_dir is None:
        return None, None
    row = _last_jsonl_row(run_dir / "annulus_live.jsonl")
    if not row:
        return None, None
    try:
        ann = row.get("annulus") or {}
        lane = (ann.get("per_class_annulus_flip_frac") or {}).get("1")
        parts = [
            f"annulus: ep{row.get('epoch')} d_seg {ann.get('overall_d_seg'):.6f}",
            f"annulus mass share {100 * ann.get('annulus_flip_mass_share', 0):.1f}%",
        ]
        if lane is not None:
            parts.append(f"lane(cls1) flip {100 * float(lane):.1f}%")
        parts.append(f"[{row.get('seg_form')}, advisory]")
        return " | ".join(parts), {"epoch": row.get("epoch"), "annulus": ann}
    except Exception:
        return None, None


def section_shadow(run_dir: Path | None) -> tuple[list[str], dict | None]:
    """Latest shadow-observer DECIDE state (classification + pending recommendations)."""
    if run_dir is None:
        return [], None
    path = run_dir / _wra.COSTATE_JSONL
    row = _last_jsonl_row(path)
    if row is None and (run_dir / "run.log").is_file():
        # No sidecar yet but telemetry exists: compute ONE read-only in-memory report.
        try:
            from tac.witness_control import build_shadow_report, load_run_inputs

            row = build_shadow_report(load_run_inputs(run_dir)).to_row()
        except Exception:
            row = None
    if not row:
        return ["costate-shadow: no rows yet (observer will populate costate_shadow.jsonl)"], None
    lines: list[str] = []
    try:
        age = max(0.0, time.time() - path.stat().st_mtime) if path.exists() else None
        cls = (row.get("classification") or {}).get("classification", "?")
        head = f"costate-shadow: ep{row.get('epoch')} class={str(cls).upper()}"
        if age is not None:
            head += f" ({_fmt_age(age)} old)"
        recs = row.get("recommendations") or []
        for r in recs[:2]:
            pd = r.get("predicted_dS")
            pd_s = f"{pd:+.4f}" if isinstance(pd, (int, float)) else "?"
            # P2 (design_philosophies_eightfold): EVERY comparison carries its noise floor. The rec's
            # predicted_dS_band IS the composed noise floor. No band -> cited Δ lacks a floor -> INSTANCE
            # only (never above advisory). A band spanning 0 -> Δ within noise -> INSTANCE. Else state Δ/floor.
            # P9 (proxies are poison, use the thing itself): the band gates predicted_dS in the SAME units
            # (ΔS) as the Δ — never a proxy-unit floor (the R7 category-error lesson).
            band = r.get("predicted_dS_band")
            if not (
                isinstance(band, (list, tuple)) and len(band) == 2 and all(isinstance(x, (int, float)) for x in band)
            ):
                floor_tag = " [INSTANCE — no noise floor]"
            elif band[0] <= 0.0 <= band[1]:
                floor_tag = f" [INSTANCE — Δ within noise floor [{band[0]:+.4f},{band[1]:+.4f}]]"
            else:
                floor_tag = f" (floor [{band[0]:+.4f},{band[1]:+.4f}])"
            lines.append(f"  rec: {r.get('action')} ΔS {pd_s}/{r.get('horizon_epochs')}ep{floor_tag}")
        if not recs:
            lines.append("  rec: (none identifiable)")
        factor = row.get("factorized_adjoint")
        if isinstance(factor, dict):
            fac = factor.get("factorization") or {}
            exact = fac.get("exact") or {}
            derived = fac.get("derived") or {}
            learned = factor.get("learned_residual") or {}
            decision = factor.get("decision") or {}
            zero = exact.get("certified_zero_weight_camera_frac")
            ratio = derived.get("road_lane_gain_only_lambda_ratio_vs_other_median")
            zero_s = f"{100 * zero:.1f}%" if isinstance(zero, (int, float)) else "?"
            ratio_s = f"{ratio:.2f}x" if isinstance(ratio, (int, float)) else "?"
            lines.append(
                "  factorized-adjoint: " + str(factor.get("admission", "?"))
                + f" | EXACT head-rank={exact.get('head_rank', '?')} ker(A)-zero={zero_s}"
                + f" | DERIVED lambda_Road-Lane={ratio_s} other-major median"
                + f" | LEARNED={learned.get('n_parameters', '?')} shared scalars"
                + f" ({learned.get('amplitude_gate', 'not-fit')})")
            confidence = ((factor.get("recommendation_candidate") or {}).get("confidence")
                          or factor.get("validation_scope"))
            if confidence:
                lines.append("  factorized-confidence: " + str(confidence))
            if decision:
                pd = decision.get("predicted_dS")
                pd_s = f"{pd:+.4f}" if isinstance(pd, (int, float)) else "?"
                lines.append(
                    f"  factorized-DECIDE: DeltaS {pd_s}/{decision.get('horizon_epochs')}ep; "
                    + str(decision.get("why", "reason unavailable")))
        events = row.get("event_advisories") or []
        if events and isinstance(events[0], dict):
            e = events[0]
            ep = ((e.get("morse_smale") or {}).get("derived") or {})
            ncde = e.get("ncde_344") or {}
            lines.append(
                "  event-intelligence: "
                + ("WARNING ACTIVE" if e.get("warning_active") else "next-boundary watch")
                + f" | critical pair={ep.get('critical_lambda_pair', '?')}"
                + f" lambda-ratio={ep.get('critical_lambda_ratio_vs_other_major_pair_median', '?')}"
                + f" | #344 fire={ncde.get('fire', False)} reason={ncde.get('reason', 'unavailable')}")
        lines.insert(0, head)
        if age is not None and age > _SHADOW_STALE_S:
            lines.append(f"  refresh: .venv/bin/python tools/costate_shadow_report.py --run-dir {run_dir} --write")
    except Exception as exc:
        lines = [f"costate-shadow: unavailable ({type(exc).__name__}: {exc})"]
    return lines, row


def format_ncde_line(row: dict) -> str:
    """Pure formatter for one Linear-NCDE hit->solve advisory row (#344). Unit-testable
    without touching real state. Surfaces: asymptote estimate + ETA-to-threshold +
    BASIN/HANDOFF verdict + the instrument-invalid guard state. Advisory-only wording."""
    ep = row.get("epoch")
    r2 = row.get("fit_r2")
    stable = row.get("stable")
    reason = str(row.get("reason", ""))
    fire = bool(row.get("fire"))
    asym = row.get("predicted_asymptote")  # log_d_seg space
    eta = row.get("eta_handoff_epochs")
    rem = row.get("remaining_descent_frac")
    invalid = "instrument-invalid" in reason or stable is False or (isinstance(r2, (int, float)) and r2 < 0.5)
    ep_s = f"{ep:.0f}" if isinstance(ep, (int, float)) else "?"
    r2_s = f"{r2:.2f}" if isinstance(r2, (int, float)) else "?"
    if invalid:
        tag = f"INSTRUMENT-INVALID (stable={stable}, r2={r2_s}) — no verdict"
    elif fire and reason.startswith("BASIN"):
        tag = "BASIN-FIRE (terminal SOLVE #341 admissible)"
    elif fire:
        tag = "HANDOFF-FIRE (#315 plateau-slope predicted soon)"
    else:
        tag = "NO-FIRE (still descending)"
    parts = [f"ncde-trajectory: ep{ep_s} d_seg {tag}"]
    detail: list[str] = []
    if not invalid:
        if isinstance(asym, (int, float)):
            detail.append(f"plateau d_seg~{math.exp(asym):.5f}")
        if isinstance(rem, (int, float)):
            detail.append(f"rem {100 * rem:.1f}% of travelled")
        if isinstance(eta, (int, float)):
            detail.append(f"ETA-handoff {eta:.0f}ep")
    head = " — ".join([parts[0], ", ".join(detail)]) if detail else parts[0]
    return head + " [advisory NON-PROMOTABLE]"


def section_ncde(run_dir: Path | None) -> tuple[str | None, dict | None]:
    """Linear-NCDE trajectory advisory (#344): the score-relevant d_seg hit->solve verdict
    for the live run. Read-only + score-neutral -> defaults ON (CLAUDE.md "'Off' is a
    tracked queue": read-only telemetry is not gate-able). Reuses the probe entry point
    (tools/ncde_trajectory_probe.run_probe) -- the math is NOT duplicated here. Fail-open
    (any error -> omit the row); omitted when d_seg verdict telemetry is too short (<8 pts,
    no advisory yet). Cheap (~0.01s measured) + cached by run.log signature."""
    if run_dir is None:
        return None, None
    try:
        log = run_dir / "run.log"
        if not log.is_file():
            return None, None
        st = log.stat()
        sig = (st.st_mtime_ns, st.st_size)
        cached = _NCDE_CACHE.get(str(run_dir))
        if cached is not None and cached[0] == sig:
            return cached[1], cached[2]
        import ncde_trajectory_probe as ntp

        report = ntp.run_probe(run_dir, window=12, emit=False, do_backtest=False)
        adv = report.get("verdict_latest_advisory")
        if not adv:  # no d_seg advisory yet (short verdict telemetry) — omit, honestly
            _NCDE_CACHE[str(run_dir)] = (sig, None, None)
            return None, None
        line = format_ncde_line(adv)
        data = {"advisory": adv, "verdict_points": report.get("verdict_points"), "n_fires": report.get("n_fires")}
        _NCDE_CACHE[str(run_dir)] = (sig, line, data)
        return line, data
    except Exception:
        return None, None


def section_verdict_trend(run_dir: Path | None) -> tuple[str | None, dict | None]:
    """VERDICT-TREND / TRAIN-VERDICT-DECOUPLING alarm (operator-catch 2026-07-09): the
    advisory verdict d_seg RISING while the train seg-loss descends — the false-green the
    scalar classifier misses. Read-only + score-neutral -> defaults ON (CLAUDE.md "'Off' is
    a tracked queue"). Fail-open (any error -> omit); omitted when there is no material
    rising trend. Cheap + cached by run.log signature (like section_ncde)."""
    if run_dir is None:
        return None, None
    try:
        log = run_dir / "run.log"
        if not log.is_file():
            return None, None
        st = log.stat()
        sig = (st.st_mtime_ns, st.st_size)
        cached = _VERDICT_TREND_CACHE.get(str(run_dir))
        if cached is not None and cached[0] == sig:
            return cached[1], cached[2]
        from tac.witness_control import (
            format_verdict_trend_line,
            load_run_inputs,
            verdict_trend_alarm,
        )

        alarm = verdict_trend_alarm(load_run_inputs(run_dir).verdicts)
        if not alarm.fired():  # only surface when there is a real rising trend
            _VERDICT_TREND_CACHE[str(run_dir)] = (sig, None, None)
            return None, None
        line = format_verdict_trend_line(alarm)
        data = alarm.to_confound_alarm_row()
        _VERDICT_TREND_CACHE[str(run_dir)] = (sig, line, data)
        return line, data
    except Exception:
        return None, None


def section_pose_conditioning_gate(run_dir: Path | None) -> tuple[str | None, dict | None]:
    """owed-1 POSE-FINISH CONDITIONING gate (SYNTHESIS_v3_v752 §A.4): surface the LOUD pose-DISENGAGED
    alarm (shipped banked R1 because the rolling-slope σ_min plateau never engaged) if present, else the
    latest conditioning-gate observer row. Read-only + score-neutral -> defaults ON. Fail-open (any
    error -> omit); omitted when the run has no pose-gate telemetry."""
    if run_dir is None:
        return None, None
    try:
        from tac.witness_control.sigma_min_plateau import format_gate_line, scan_run_for_pose_gate

        row = scan_run_for_pose_gate(run_dir)
        if not row:
            return None, None
        return format_gate_line(row), row
    except Exception:
        return None, None


def section_telemetry_binding(run_dir: Path | None) -> tuple[str | None, dict | None]:
    """#404 P0 binding-vs-inert lever readback (tac.witness_control.telemetry_binding): amber
    grad-clip binding rate · chroma term share · pose-gate sensor liveness (the silent-crash class)
    · EMA-lag verdict-vs-live divergence · D27b terminal-band/solve-upon-basin trigger · TAIL
    endpoints. Read-only + score-neutral -> defaults ON (CLAUDE.md "'Off' is a tracked queue").
    Bounded tail read (cheap at SessionStart cadence); fail-open (any error -> omit)."""
    if run_dir is None:
        return None, None
    try:
        from tac.witness_control import telemetry_binding as _tb

        rows = _tb.load_run_rows(run_dir, tail_bytes=1_500_000)
        if not rows:
            return None, None
        audit = _tb.audit_rows(rows)
        return _tb.format_summary(audit), {
            k: audit.get(k) for k in ("amber", "chroma", "pose_gate", "ema_lag", "terminal_band")
        }
    except Exception:
        return None, None


def _duty_marker(r: dict) -> str:
    """Marker per ranked row: ~=un-built finding (missing wire) · ?=registered owed an estimate ·
    *=never-fired registered lever · ''=fired-but-unmeasured registered lever."""
    if not r.get("registered"):
        return "~"
    if r.get("est_delta_s") is None:
        return "?"
    return "*" if r.get("activation_state") == "never-fired" else ""


def _floor_marker(r: dict) -> str:
    """P8 floor marker (design_philosophies_eightfold): !FLOOR=target term AT its measured floor (buys ~0)
    · ^cap=est capped to the headroom-to-floor · ''=above floor / floor unmeasured. Score-neutral display."""
    st = r.get("floor_status")
    if st == "AT_FLOOR":
        return "!FLOOR"
    if st == "HEADROOM_CAPPED":
        return "^cap"
    return ""


def format_duty_to_measure_line(ranked: list[dict], top_n: int = _DUTY_TOP_N) -> str:
    """Pure formatter for the ranked duty-to-measure queue (unit-testable without touching real state).
    Renders each lever with its % of remaining descent and orphan marker; leads with the highest
    relative-value owed lever."""
    owed_n = sum(1 for r in ranked if r.get("in_duty_queue"))
    s_cur = ranked[0].get("s_current") if ranked else None
    tgt = ranked[0].get("s_target", 0.15) if ranked else 0.15

    def _cell(r: dict) -> str:
        pct = r.get("rel_sig_pct")
        tag = f" {pct:g}%" if pct is not None else " ?%"
        return f"{r['lever']}{_duty_marker(r)}{_floor_marker(r)}{tag}"

    top = ranked[:top_n]
    more = len(ranked) - len(top)
    anchor = f"pointer {s_cur:.5f}→{tgt:g}" if isinstance(s_cur, float) else "pointer unavailable"
    return (
        f"duty-to-measure ({owed_n} owed; ranked by % of remaining descent [P8 floor-aware], {anchor}; "
        f"*=never-fired ~=unbuilt ?=est-owed !FLOOR=at-floor ^cap=headroom-capped): "
        + ", ".join(_cell(r) for r in top)
        + (f" (+{more} more)" if more > 0 else "")
    )


def _live_term_current(annulus_data: dict | None) -> dict[str, float] | None:
    """Extract the live run's MEASURED current term value(s) for P8 floor-aware ranking. Today only
    ``d_seg`` is cleanly available (the annulus SENSE row's ``overall_d_seg``, an advisory MEASURED
    witness d_seg); d_pose/rate current-term stores are OWED (returned absent, honestly). Returns None
    when there is no live run (floors then surface as FLOOR_KNOWN_CURRENT_UNKNOWN — an owed measurement)."""
    if not annulus_data:
        return None
    try:
        d_seg = (annulus_data.get("annulus") or {}).get("overall_d_seg")
        if isinstance(d_seg, (int, float)) and d_seg == d_seg:  # not NaN
            return {"d_seg": float(d_seg)}
    except Exception:
        return None
    return None


def section_duty_to_measure(term_current: dict[str, float] | None = None) -> tuple[str, dict | None]:
    """Top-N owed DSL levers RANKED by relative significance — the fraction of the REMAINING descent
    to sub-0.15 each lever buys (est_delta_s / (pointer − 0.15)), **P8 floor-aware** (a lever whose
    target term is at its measured floor ranks ~0). This is the continual-learning fix for the recurring
    magnitude-dismissal bug: the CONTROLLER holds the value ranking, not the eyeball (CLAUDE.md "'Off' is
    a tracked queue" + "relative-not-absolute-significance-near-goal"). Markers: *=never-fired · ~=unbuilt
    · ?=owed-estimate · !FLOOR=at-floor · ^cap=headroom-capped.

    P1 (one-fact-one-store-one-key): the significance read is routed through
    ``duty_to_measure_ranked`` -> ``canonicalize_significance_keys`` (the ONLY significance-store reader);
    no raw-key read exists in the digest (audited 2026-07-09)."""
    try:
        from tac.witness_dsl.activation_ledger import duty_to_measure_ranked

        # reads the LIVE pointer + significance store (not hardcoded); term_current from live telemetry.
        ranked = duty_to_measure_ranked(term_current=term_current)
        line = format_duty_to_measure_line(ranked)
        s_cur = ranked[0].get("s_current") if ranked else None
        tgt = ranked[0].get("s_target", 0.15) if ranked else 0.15
        n_at_floor = sum(1 for r in ranked if r.get("floor_status") == "AT_FLOOR")
        n_capped = sum(1 for r in ranked if r.get("floor_status") == "HEADROOM_CAPPED")
        return line, {
            "ranked_top": ranked[:_DUTY_TOP_N],
            "owed_registered": sum(1 for r in ranked if r.get("in_duty_queue")),
            "term_current": term_current,
            "n_at_floor": n_at_floor,
            "n_headroom_capped": n_capped,
            "s_current": s_cur,
            "s_target": tgt,
        }
    except Exception as exc:
        return f"duty-to-measure: unavailable ({type(exc).__name__}: {exc})", None


def _pool_marker(r: dict) -> str:
    """DSL-leg marker per pool row: '' = held by a DSL Lever factory · ~ = NOT a single-flag lever
    (tool-side / vehicle-level / unbuilt, carries dsl_na_reason). Mirrors the duty-line vocabulary."""
    return "" if r.get("dsl_lever") else "~"


def format_curriculum_pool_line(summary: dict, top_n: int = _DUTY_TOP_N) -> str:
    """Pure formatter for the curriculum-candidate pool (unit-testable without touching real state).
    Leads with production-fireable duty, then a separate research-only SENSE channel. Research rows
    are visible but never presented as DECIDE/fireable. ~ = not-a-DSL lever (tool/vehicle/unbuilt)."""
    counts = summary.get("counts", {})
    owed = summary.get("owed", 0)
    total = summary.get("total", 0)
    top = summary.get("top_fireable", [])[:top_n]
    more = summary.get("owed", 0) - len(top)
    research = summary.get("research_signals", [])[:top_n]
    research_more = len(summary.get("research_signals", [])) - len(research)

    def _cell(r: dict) -> str:
        leg = r.get("dsl_lever") or "N/A"
        return f"{r['candidate']}{_pool_marker(r)}[{r.get('status', '?')[:5]}·{leg}]"

    line = (
        f"curriculum-pool ({total} tracked; {owed} owed a fire; "
        f"{counts.get('built-never-fired', 0)} built-never-fired · "
        f"{counts.get('needs-build', 0)} needs-build · "
        f"{counts.get('reformulation-queue', 0)} reformulation-queue · "
        f"{counts.get('armed', 0)} armed; ~=not-a-DSL-lever): "
        + ", ".join(_cell(r) for r in top)
        + (f" (+{more} more owed)" if more > 0 else "")
    )
    if research:
        line += " | research-only SENSE (non-fireable): " + ", ".join(
            f"{row['candidate']}[{row.get('status', '?')}]" for row in research
        )
        if research_more > 0:
            line += f" (+{research_more} more research signals)"
    return line


def section_curriculum_pool() -> tuple[str | None, dict | None]:
    """The CURRICULUM-CANDIDATE POOL as a tracked costate class (task #403; P0 orphan-class binding
    2026-07-10). A curriculum candidate in ANY form (stage / loss / init / preconditioning / data-order
    / averaging / solve-interleave / state-evolution) that is designed-or-built-but-never-fired is the
    SAME orphaned-signal class the lever ledger tracks — but is often a TOOL / stage / vehicle DOF, not a
    single-flag DSL lever, so it lives in the sibling ``curriculum_candidate_pool`` store. This SENSE row
    surfaces production-fireable rows beside a separate research-only SENSE channel, so NO curriculum
    candidate is orphaned and an unactivatable research finding never enters DECIDE. Read-only,
    score-neutral, fail-open (omit on any error)."""
    try:
        from tac.witness_dsl.curriculum_candidate_pool import pool_summary

        summary = pool_summary()
        if not summary.get("total"):
            return None, None
        line = format_curriculum_pool_line(summary)
        return line, {
            "total": summary["total"],
            "owed": summary["owed"],
            "counts": summary["counts"],
            "top_fireable": [
                {
                    "candidate": r.get("candidate"),
                    "status": r.get("status"),
                    "dsl_lever": r.get("dsl_lever"),
                    "dsl_na_reason": r.get("dsl_na_reason"),
                    "form_class": r.get("form_class"),
                    "owner": r.get("owner"),
                }
                for r in summary.get("top_fireable", [])
            ],
            "research_signals": [
                {
                    "candidate": r.get("candidate"),
                    "status": r.get("status"),
                    "research_only": r.get("research_only"),
                    "evidence_kind": r.get("evidence_kind"),
                    "authority_axis": r.get("authority_axis"),
                    "verdict_scope": r.get("verdict_scope"),
                    "activation_status": r.get("activation_status"),
                    "realized_speedup_factor": r.get("realized_speedup_factor"),
                    "derived_cost_reduction_fraction": r.get("derived_cost_reduction_fraction"),
                    "trusted_receipt_sha256": r.get("trusted_receipt_sha256"),
                    "blockers": r.get("blockers", []),
                }
                for r in summary.get("research_signals", [])
            ],
        }
    except Exception as exc:
        return f"curriculum-pool: unavailable ({type(exc).__name__}: {exc})", None


def section_deferral_ledger() -> tuple[str | None, dict | None]:
    """Operator-binding 2026-07-08 ("you deferred too much and now it's orphaned"): every
    deferral lives in .omx/state/deferral_ledger.md with a named trigger; this line makes
    the queue impossible to forget. Soft: absent file -> no line."""
    try:
        path = _REPO / ".omx" / "state" / "deferral_ledger.md"
        if not path.exists():
            return None, None
        rows = [ln for ln in path.read_text(errors="replace").splitlines() if ln.startswith("| D")]
        hot = [ln.split("|")[1].strip() for ln in rows if "SURFACED" in ln or "FIRING" in ln or "ARMED" in ln]
        line = (
            f"deferral-ledger: {len(rows)} open"
            + (f"; hot: {', '.join(hot)}" if hot else "")
            + " (.omx/state/deferral_ledger.md — every row has a named trigger)"
        )
        return line, {"open": len(rows), "hot": hot}
    except Exception as exc:
        return f"deferral-ledger: unavailable ({type(exc).__name__}: {exc})", None


def section_failure_ledger() -> tuple[str | None, dict | None]:
    """Sibling SENSE input (soft: the ledger may not exist yet). Glob-matched so the
    sibling's chosen filename is picked up without a code change here."""
    try:
        hits = sorted(glob.glob(str(_REPO / ".omx" / "state" / "*failure*ledger*.jsonl")))
        if not hits:
            return None, None
        path = Path(hits[-1])
        rows: list[dict] = []
        for ln in path.read_text(errors="replace").splitlines():
            if ln.strip():
                try:
                    row = json.loads(ln)
                    if isinstance(row, dict):
                        rows.append(row)
                except Exception:
                    continue
        if not rows:
            return None, None
        # per failure_id: unresolved = latest event is not a resolution; recurrence = seen 2+ times
        by_id: dict[str, list[dict]] = {}
        for r in rows:
            by_id.setdefault(str(r.get("failure_id") or "?"), []).append(r)
        unresolved = sorted(fid for fid, evs in by_id.items() if evs[-1].get("event") != "resolution")
        recurrent = sorted(
            fid for fid, evs in by_id.items() if sum(1 for e in evs if e.get("event") != "resolution") >= 2
        )
        line = (
            f"failure-ledger ({path.name}): {len(by_id)} class(es), "
            f"{len(unresolved)} unresolved, {len(recurrent)} recurrent"
        )
        if unresolved:
            line += f"; open: {', '.join(unresolved[:3])}"
        return line, {"path": str(path), "classes": len(by_id), "unresolved": unresolved, "recurrent": recurrent}
    except Exception:
        return None, None


def section_schedule(run_dir: Path | None) -> tuple[str | None, dict | None]:
    """Planned-vs-actual DSL schedule position (sibling module — referenced, not built)."""
    for mod_name in ("dsl_schedule_readback", "dashboard_schedule_readback"):
        try:
            mod = __import__(mod_name)
        except Exception:
            continue
        for fn_name in ("planned_vs_actual_summary", "schedule_position", "summary"):
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                try:
                    out = fn(run_dir) if run_dir is not None else fn()
                    return f"schedule: {str(out)[:110]}", {"module": mod_name}
                except Exception:
                    continue
    return None, None


def section_verdict_scope_advisories() -> tuple[str | None, dict | None]:
    """Verdict-scope FM advisories (requirement R, advisory layer): suspected
    OVER-SCOPED negative verdicts flagged by the drift-detector's on-device-FM leg
    (tools/triality_drift_detector.py). The Stop hook can only show text when it
    blocks, so non-blocking advisories persist here; this line surfaces them.
    Read-only; omitted when none exist (fail-open None)."""
    try:
        p = _REPO / ".omx" / "state" / "verdict_scope_advisories.jsonl"
        if not p.exists():
            return None, None
        rows: list[dict] = []
        for ln in p.read_text(errors="replace").splitlines():
            if not ln.strip():
                continue
            try:
                row = json.loads(ln)
                if isinstance(row, dict) and row.get("advisory"):
                    rows.append(row)
            except Exception:
                continue
        if not rows:
            return None, None
        # recency window: a 14-day count keeps the line live (the file is append-only,
        # so an all-time count would grow monotonically and go stale-misleading).
        import datetime as _dt

        cutoff = _dt.datetime.now(_dt.UTC) - _dt.timedelta(days=14)

        def _fresh(r: dict) -> bool:
            try:
                return _dt.datetime.fromisoformat(str(r.get("ts", ""))) >= cutoff
            except Exception:
                return True  # unparsable ts → keep (fail toward visibility)

        fresh = [r for r in rows if _fresh(r)]
        if not fresh:
            return None, None
        last = str(fresh[-1].get("advisory", ""))[:110]
        line = (
            f"verdict-scope advisories: {len(fresh)} doc-flag(s) in the last 14d "
            f"(latest: {last}) — re-check declared scope vs evidence (req R)"
        )
        return line, {
            "count_14d": len(fresh),
            "count_all": len(rows),
            "recent": [r.get("advisory") for r in fresh[-3:]],
        }
    except Exception:
        return None, None


def section_resume_spine() -> tuple[str, dict | None]:
    """AMENDMENT 2: the compact resume spine (pointers, not content dumps)."""
    try:
        dag_files = sorted(glob.glob(_DAG_GLOB))
        feed = None
        if dag_files:
            for ln in reversed(Path(dag_files[-1]).read_text(errors="replace").splitlines()):
                if ln.startswith("## FEED-"):
                    feed = ln[3:].strip()
                    break
        feed_txt = (feed[:100] + "…") if feed and len(feed) > 100 else (feed or "none found")
        return (f"resume spine: MEMORY.md ⭐CURRENT-STATE → newest DAG {feed_txt}", {"newest_feed": feed})
    except Exception as exc:
        return f"resume spine: unavailable ({type(exc).__name__})", None


def section_active_convening() -> tuple[str | None, dict | None]:
    """Auto-surface any ACTIVE council/crucible orchestration so NO session (and no operator)
    must remember it exists. Reads the newest orchestration ledger's phase checkboxes + last
    log line; fail-open (omit if absent). Operator binding 2026-07-07: 'you are once again
    requiring me to have an insane memory' — the system holds the thread, not the human."""
    try:
        ledgers = sorted(glob.glob(".omx/research/*crucible*/ORCHESTRATION_LEDGER.md")) + sorted(
            glob.glob(".omx/research/*/ORCHESTRATION_LEDGER.md")
        )
        if not ledgers:
            return None, None
        txt = Path(ledgers[-1]).read_text(errors="replace")
        done = txt.count("- [x] P")
        total = done + txt.count("- [ ] P")
        # current phase = first unchecked
        phase = next(
            (ln.strip()[6:60] for ln in txt.splitlines() if ln.strip().startswith("- [ ] P")), "ALL PHASES DONE"
        )
        last_log = next((ln.strip() for ln in reversed(txt.splitlines()) if ln.strip().startswith("- 2026")), "")
        line = (
            f"ACTIVE CONVENING ({Path(ledgers[-1]).parent.name}): phases {done}/{total} done; "
            f"NEXT: {phase} | last: {last_log[:120]} | ledger: {ledgers[-1]}"
        )
        return line, {"ledger": ledgers[-1], "phases_done": done, "phases_total": total, "next_phase": phase}
    except Exception as exc:
        return f"active convening: unavailable ({type(exc).__name__})", None


def section_corpus_recall(conv: dict | None) -> tuple[list[str], dict | None]:
    """#346 RECALL-PUSH: if a convening is ACTIVE, surface the top corpus hits for its
    next phase — push, not pull (the apparatus writes better than it reads; this section
    makes the corpus speak at session start without anyone asking). Fail-open, ≤5 lines,
    time-bounded ≤2s via corpus_query's budget (newest units win under truncation)."""
    if not conv or not conv.get("next_phase"):
        return [], None
    try:
        import corpus_query

        result = corpus_query.run_query(str(conv["next_phase"]), top=3, max_seconds=2.0)
        hits = result.get("hits") or []
        if not hits:
            return [], None
        lines = ["the corpus knows (recall for the active convening's next phase):"]
        for h in hits[:3]:
            first = (h["lines"][0][:80] + "…") if h.get("lines") else ""
            ref = h["ref"] if len(h["ref"]) <= 90 else "…" + h["ref"][-89:]
            lines.append(f"  [{h['store']}] {ref} — {first}")
        return lines[:5], {"hits": hits, "truncated": result.get("truncated", False)}
    except Exception as exc:
        return [f"corpus recall: unavailable ({type(exc).__name__})"], None


def section_graph_memory() -> tuple[str | None, dict | None]:
    """#411 RECONSTRUCT-ON-DEMAND: surface the graph-memory (the DAG-as-graph) so
    recall-by-reconstruction is a STRUCTURAL session-start affordance, not grep-by-
    volition (the read side of L83 / the #346 retrieval-first nexus done as a graph).
    Read-only + fast (counts the cached graph's JSONL lines; NEVER rebuilds at session
    start). Score-neutral, fail-open."""
    try:
        from tac.graph_memory import cache_paths

        npath, epath = cache_paths()
        if not (npath.is_file() and epath.is_file()):
            return (
                "graph-memory: not yet built — reconstruct-on-demand via "
                "`tools/graph_memory_recall.py --rebuild` (the DAG-as-graph, #411)"
            ), None
        n_nodes = sum(1 for _ in npath.open(encoding="utf-8"))
        n_edges = sum(1 for _ in epath.open(encoding="utf-8"))
        line = (
            f"graph-memory: {n_nodes} nodes / {n_edges} edges — RECONSTRUCT before "
            f'grepping: `tools/graph_memory_recall.py "<query>"` (#411 DAG-as-graph)'
        )
        return line, {"nodes": n_nodes, "edges": n_edges}
    except Exception as exc:
        return f"graph-memory: unavailable ({type(exc).__name__})", None


def section_costate_organ(run_dir: Path | None) -> tuple[list[str], dict | None]:
    """#426 λ-organ OBSERVATORY (Rudin/max-observability): which prototype regimes
    fired with what weights + uncertainty; the compounding organ-ledger summary
    (records/regimes/recommended architecture); the top PowerPlay acquisition rows.
    Read-only, numpy-only lenses (spread-gate discipline: no torch fits here),
    fail-open, bounded (<1s on verdict-cadence data)."""
    lines: list[str] = []
    data: dict = {}
    # (a) the compounding memory summary (triality ledger — cheap parse)
    try:
        from tac.witness_control.continual_costate import organ_summary

        s = organ_summary()
        data["memory"] = s
        if s["n_records"]:
            lines.append(
                f"λ-organ memory: {s['n_records']} trajectory record(s) / "
                f"{s['total_intervals']} intervals; {s['n_regimes']} regime(s) "
                f"[{', '.join(s['regimes'][:4])}{'…' if s['n_regimes'] > 4 else ''}]; "
                f"recommended arch {s['recommended_architecture']}"
            )
    except Exception as exc:
        lines.append(f"λ-organ memory: unavailable ({type(exc).__name__})")
    # (b) the live observatory + acquisition queue (needs a run dir with verdicts)
    if run_dir is None:
        return lines, data
    try:
        import numpy as np

        from tac.witness_control.control_alphabet import powerplay_acquisition
        from tac.witness_control.lambda_net import (
            RidgeSolveAdjoint,
            build_intervals,
            evaluate_lambda_field,
            fit_score_composition,
            lever_features,
            read_trajectory,
        )
        from tac.witness_control.prototype_router import PrototypeRouterLens

        traj = read_trajectory(run_dir)
        if traj.n_verdicts >= 3:
            comp = fit_score_composition(traj.verdicts)
            intervals = build_intervals(traj)
            phis = np.stack([lever_features(n) for n in traj.lever_names])
            lens = PrototypeRouterLens()
            lens.fit(intervals, phis)
            last = traj.verdicts[-1]
            x = np.concatenate(
                [
                    np.asarray(last["d_seg_by_class"], dtype=float),
                    [math.log(max(float(last.get("blob_bytes", 1.0)), 1.0))],
                ]
            )
            att = lens.attribute(x, float(last["epoch"]), comp.grad_s_wrt_state())
            data["observatory"] = {
                "explain": att.explain(),
                "fired": list(att.fired),
                "mixture_entropy": att.mixture_entropy,
            }
            lines.append("λ-organ observatory: " + att.explain())
            faith = lens.faithfulness_audit(x, intervals[-1].ctx, comp.grad_s_wrt_state(), lever_features("seg"))
            data["faithfulness"] = {k: faith[k] for k in ("faithful", "max_rel_gap", "rel_tol")}
            lines.append(
                f"λ-organ faithfulness: {'OK' if faith['faithful'] else 'DIVERGENT'} "
                f"(max_rel_gap {faith['max_rel_gap']:.3f} ≤ tol {faith['rel_tol']}; "
                "stated-vs-counterfactual, 2607.08046)"
            )
            m = RidgeSolveAdjoint()
            m.fit(intervals, phis)
            fld = evaluate_lambda_field(m, traj, comp, intervals, status="SPECULATIVE-UNTIL-BACKTESTED")
            ever = {n: fld.identified.get(n, False) for n in fld.per_lever}
            acq = powerplay_acquisition(fld.per_lever, ever_fired=ever)[:3]
            data["acquisition_top"] = [{"lever": r.lever, "acquisition": r.acquisition, "kind": r.kind} for r in acq]
            lines.append(
                "λ-organ acquisition (PowerPlay top-3): "
                + " · ".join(f"{r.lever}({r.kind.split('-')[0]},{r.acquisition:.2f})" for r in acq)
            )
            # (c) #436 regime-conditional SELF-DISPATCH — the per-state arbiter decision
            #     (advisory, past-only, NON-PROMOTABLE): current regime + which tool it
            #     dispatches (or PERSISTENCE), provenance-labeled to the measured verdict.
            from tac.witness_control.regime_dispatch import dispatch_for_trajectory

            dd = dispatch_for_trajectory(traj)
            data["dispatch"] = {
                "regime": dd.classification.regime,
                "tool": dd.tool,
                "deciding_signal": dd.classification.deciding_signal,
                "explain": dd.explain(),
                "artifact": _DISPATCH_DOC,
                "axis_tag": dd.axis_tag,
                "actuation": dd.actuation,
            }
            lines.append(
                f"organ-dispatch (#436, {dd.axis_tag}): regime={dd.classification.regime} "
                f"→ TOOL={dd.tool} [{dd.classification.deciding_signal}] "
                f"· actuation={dd.actuation} · {_DISPATCH_DOC}"
            )
    except Exception as exc:
        lines.append(f"λ-organ observatory: unavailable ({type(exc).__name__})")
    return lines, data


def section_review_counter() -> tuple[str | None, dict | None]:
    """Open review-counter state (sibling ledger; soft — omit entirely if absent)."""
    row = _last_jsonl_row(_REVIEW_COUNTER) if _REVIEW_COUNTER.exists() else None
    if not row:
        return None, None
    try:
        # sibling schema review_counter.v1: surface_id / round_n / findings_count / verdict
        line = (
            f"review-counter: {row.get('surface_id', '?')} round {row.get('round_n', '?')} "
            f"findings {row.get('findings_count', '?')} verdict {row.get('verdict', '?')}"
        )
        return line[:130], row
    except Exception:
        return None, None


# ─────────────────────────── assembly ───────────────────────────
def build_digest() -> tuple[list[str], dict]:
    t0 = time.time()
    lines: list[str] = []
    data: dict = {}

    ptr_line, data["pointer"] = section_pointer()
    lines.append(ptr_line)  # NEVER dropped (amendment 1)

    live_line, data["live_run"], run_dir = section_live_run()
    lines.append(live_line)  # NEVER dropped

    ann_line, data["annulus"] = section_annulus(run_dir)
    if ann_line:
        lines.append(ann_line)

    shadow_lines, data["shadow"] = section_shadow(run_dir)
    lines.extend(shadow_lines)

    ncde_line, data["ncde"] = section_ncde(run_dir)
    if ncde_line:
        lines.append(ncde_line)

    vtrend_line, data["verdict_trend"] = section_verdict_trend(run_dir)
    if vtrend_line:
        lines.append(vtrend_line)

    posegate_line, data["pose_conditioning_gate"] = section_pose_conditioning_gate(run_dir)
    if posegate_line:
        lines.append(posegate_line)

    tbind_line, data["telemetry_binding"] = section_telemetry_binding(run_dir)
    if tbind_line:
        lines.append(tbind_line)

    # P8 floor-aware: feed the live run's MEASURED current d_seg (annulus SENSE) so at-floor levers rank ~0.
    duty_line, data["duty_to_measure"] = section_duty_to_measure(_live_term_current(data.get("annulus")))
    lines.append(duty_line)

    # #403 P0: the curriculum-candidate pool (ANY form) as a tracked costate class, beside the lever queue.
    cpool_line, data["curriculum_pool"] = section_curriculum_pool()
    if cpool_line:
        lines.append(cpool_line)

    fl_line, data["failure_ledger"] = section_failure_ledger()
    lines.append(fl_line or "failure-ledger: none yet (sibling SENSE input pending)")

    dl_line, data["deferral_ledger"] = section_deferral_ledger()
    if dl_line:
        lines.append(dl_line)

    sched_line, data["schedule"] = section_schedule(run_dir)
    lines.append(sched_line or "schedule: planned-vs-actual read-back pending (sibling module)")

    vs_line, data["verdict_scope"] = section_verdict_scope_advisories()
    if vs_line:
        lines.append(vs_line)

    spine_line, data["resume_spine"] = section_resume_spine()
    lines.append(spine_line)

    conv_line, data["active_convening"] = section_active_convening()
    if conv_line:
        lines.append(conv_line)

    recall_lines, data["corpus_recall"] = section_corpus_recall(data["active_convening"])
    lines.extend(recall_lines)

    gm_line, data["graph_memory"] = section_graph_memory()
    if gm_line:
        lines.append(gm_line)

    organ_lines, data["costate_organ"] = section_costate_organ(run_dir)
    lines.extend(organ_lines)

    rc_line, data["review_counter"] = section_review_counter()
    if rc_line:
        lines.append(rc_line)

    lines.append(
        "BOUNDARY: autonomous = advisory recs · duty-to-measure ranking · curriculum "
        "condition inputs · this digest. Operator-GO = heavy/paid launches · run stops · "
        "live-config changes (CONTAINMENT)."
    )
    lines.append(f"deeper: {_DESIGN_DOC} (§2026-07-07 agent-native surfacing)")
    data["wall_clock_s"] = round(time.time() - t0, 3)
    return lines, data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument(
        "--session-start", action="store_true", help="hook mode: same digest, ALWAYS exits 0 (never blocks a session)"
    )
    args = ap.parse_args(argv)
    try:
        lines, data = build_digest()
        if args.json:
            print(json.dumps(data, indent=2, sort_keys=True, default=str))
        else:
            if args.session_start:
                print("[costate-digest] controller SENSE+DECIDE state (auto-surfaced; tools/costate_digest.py):")
            print("\n".join(lines))
        return 0
    except Exception as exc:  # fail-open: a broken digest must never crash a session
        print(f"[costate-digest] unavailable ({type(exc).__name__}: {exc})")
        return 0 if args.session_start else 1


if __name__ == "__main__":
    sys.exit(main())
