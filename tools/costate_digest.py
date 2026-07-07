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
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

_POINTER_JSON = _REPO / ".omx" / "state" / "canonical_frontier_pointer.json"
_DAG_GLOB = str(_REPO / ".omx" / "research" / "sub015_DAG_topaiml_reopen_and_pursuit_plan_*.md")
_DESIGN_DOC = ".omx/research/costate_controller_design_20260705.md"
_REVIEW_COUNTER = _REPO / ".omx" / "state" / "review_counter.jsonl"
_DUTY_TOP_N = 6
_SHADOW_STALE_S = 2 * 3600.0


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
        line = (f"POINTER {score:.5f} [contest-CPU] UNMOVED since {since} — "
                f"everything below is means.")
        return line, {"score": score, "axis": "contest-CPU", "since": since}
    except Exception as exc:
        return (f"POINTER: unavailable ({type(exc).__name__}) — read "
                f".omx/state/canonical_frontier_pointer.json"), {"error": str(exc)}


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
        parts = [f"annulus: ep{row.get('epoch')} d_seg {ann.get('overall_d_seg'):.6f}",
                 f"annulus mass share {100 * ann.get('annulus_flip_mass_share', 0):.1f}%"]
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
    path = run_dir / "costate_shadow.jsonl"
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
            lines.append(f"  rec: {r.get('action')} ΔS {pd_s}/{r.get('horizon_epochs')}ep")
        if not recs:
            lines.append("  rec: (none identifiable)")
        lines.insert(0, head)
        if age is not None and age > _SHADOW_STALE_S:
            lines.append(f"  refresh: .venv/bin/python tools/costate_shadow_report.py "
                         f"--run-dir {run_dir} --write")
    except Exception as exc:
        lines = [f"costate-shadow: unavailable ({type(exc).__name__}: {exc})"]
    return lines, row


def section_duty_to_measure() -> tuple[str, dict | None]:
    """Top-N never-fired/unmeasured DSL levers — the duty-to-measure queue the
    controller drains (CLAUDE.md "'Off' is a tracked queue")."""
    try:
        from tac.witness_dsl.activation_ledger import duty_to_measure, never_fired
        owed = duty_to_measure()
        nf = set(never_fired())
        top = list(owed)[:_DUTY_TOP_N]
        more = len(owed) - len(top)
        names = ", ".join(f"{n}{'*' if n in nf else ''}" for n in top)
        line = (f"duty-to-measure ({len(owed)} owed; *=never-fired): {names}"
                + (f" (+{more} more)" if more > 0 else ""))
        return line, {"owed": list(owed), "never_fired": sorted(nf)}
    except Exception as exc:
        return f"duty-to-measure: unavailable ({type(exc).__name__}: {exc})", None


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
        unresolved = sorted(fid for fid, evs in by_id.items()
                            if evs[-1].get("event") != "resolution")
        recurrent = sorted(fid for fid, evs in by_id.items()
                           if sum(1 for e in evs if e.get("event") != "resolution") >= 2)
        line = (f"failure-ledger ({path.name}): {len(by_id)} class(es), "
                f"{len(unresolved)} unresolved, {len(recurrent)} recurrent")
        if unresolved:
            line += f"; open: {', '.join(unresolved[:3])}"
        return line, {"path": str(path), "classes": len(by_id),
                      "unresolved": unresolved, "recurrent": recurrent}
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
        return (f"resume spine: MEMORY.md ⭐CURRENT-STATE → newest DAG {feed_txt}",
                {"newest_feed": feed})
    except Exception as exc:
        return f"resume spine: unavailable ({type(exc).__name__})", None


def section_review_counter() -> tuple[str | None, dict | None]:
    """Open review-counter state (sibling ledger; soft — omit entirely if absent)."""
    row = _last_jsonl_row(_REVIEW_COUNTER) if _REVIEW_COUNTER.exists() else None
    if not row:
        return None, None
    try:
        # sibling schema review_counter.v1: surface_id / round_n / findings_count / verdict
        line = (f"review-counter: {row.get('surface_id', '?')} round {row.get('round_n', '?')} "
                f"findings {row.get('findings_count', '?')} verdict {row.get('verdict', '?')}")
        return line[:130], row
    except Exception:
        return None, None


# ─────────────────────────── assembly ───────────────────────────
def build_digest() -> tuple[list[str], dict]:
    t0 = time.time()
    lines: list[str] = []
    data: dict = {}

    ptr_line, data["pointer"] = section_pointer()
    lines.append(ptr_line)                               # NEVER dropped (amendment 1)

    live_line, data["live_run"], run_dir = section_live_run()
    lines.append(live_line)                              # NEVER dropped

    ann_line, data["annulus"] = section_annulus(run_dir)
    if ann_line:
        lines.append(ann_line)

    shadow_lines, data["shadow"] = section_shadow(run_dir)
    lines.extend(shadow_lines)

    duty_line, data["duty_to_measure"] = section_duty_to_measure()
    lines.append(duty_line)

    fl_line, data["failure_ledger"] = section_failure_ledger()
    lines.append(fl_line or "failure-ledger: none yet (sibling SENSE input pending)")

    sched_line, data["schedule"] = section_schedule(run_dir)
    lines.append(sched_line or "schedule: planned-vs-actual read-back pending (sibling module)")

    spine_line, data["resume_spine"] = section_resume_spine()
    lines.append(spine_line)

    rc_line, data["review_counter"] = section_review_counter()
    if rc_line:
        lines.append(rc_line)

    lines.append(
        "BOUNDARY: autonomous = advisory recs · duty-to-measure ranking · curriculum "
        "condition inputs · this digest. Operator-GO = heavy/paid launches · run stops · "
        "live-config changes (CONTAINMENT).")
    lines.append(f"deeper: {_DESIGN_DOC} (§2026-07-07 agent-native surfacing)")
    data["wall_clock_s"] = round(time.time() - t0, 3)
    return lines, data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--session-start", action="store_true",
                    help="hook mode: same digest, ALWAYS exits 0 (never blocks a session)")
    args = ap.parse_args(argv)
    try:
        lines, data = build_digest()
        if args.json:
            print(json.dumps(data, indent=2, sort_keys=True, default=str))
        else:
            if args.session_start:
                print("[costate-digest] controller SENSE+DECIDE state "
                      "(auto-surfaced; tools/costate_digest.py):")
            print("\n".join(lines))
        return 0
    except Exception as exc:  # fail-open: a broken digest must never crash a session
        print(f"[costate-digest] unavailable ({type(exc).__name__}: {exc})")
        return 0 if args.session_start else 1


if __name__ == "__main__":
    sys.exit(main())
