#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit L1+ substrate lanes that have not received a paid dispatch.

Premortem #1 anchor (`.omx/research/12_month_frustration_premortem_and_
recommendations_20260516.md` Category E + Section 3 #1): without
substrate-retirement discipline, L1 SCAFFOLD lanes accumulate in the
lane registry without dispatch, polluting the cathedral autopilot
ranker surface and creating decision overload. The 12-month projection
is 200+ stale L1 substrates if the cadence continues unchanged.

This tool consumes `.omx/state/lane_registry.json` + the dispatch
posterior (`.omx/state/cost_band_posterior.jsonl` + the lane maturity
audit log) and emits the per-lane staleness verdict:

- ACTIVE_RECENT_DISPATCH: lane has a `successful_dispatch` outcome in
  the posterior within the staleness window.
- ACTIVE_RECENT_MARK: lane has a `lane_maturity.py mark` audit entry
  within the staleness window (proxy for "operator is actively
  iterating on the lane").
- STALE_PENDING_DECISION: in-scope L1+ substrate lane with no dispatch
  AND no audit-log activity within the staleness window AND no opt-out
  (research_only / substrate_engineering / archived state).
- OPT_OUT_RESEARCH_ONLY / OPT_OUT_SUBSTRATE_ENGINEERING /
  OPT_OUT_ARCHIVED: lane has explicitly opted out of dispatch-pending
  status.

Sister of Catalog #298 (`check_substrate_lane_l1_scaffold_not_stale_
dispatch`) which refuses STALE_PENDING_DECISION lanes at preflight time.
This tool is the *operator-facing* surface for the same data — produces
the monthly retirement candidate list per CLAUDE.md "Substrate
retirement discipline" non-negotiable.

Usage:
    .venv/bin/python tools/audit_stale_l1_substrates.py
    .venv/bin/python tools/audit_stale_l1_substrates.py --json
    .venv/bin/python tools/audit_stale_l1_substrates.py --staleness-days 60
    .venv/bin/python tools/audit_stale_l1_substrates.py --include-l2
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LANE_REGISTRY_PATH = REPO_ROOT / ".omx" / "state" / "lane_registry.json"
COST_BAND_POSTERIOR_PATH = REPO_ROOT / ".omx" / "state" / "cost_band_posterior.jsonl"
LANE_AUDIT_LOG_PATH = REPO_ROOT / ".omx" / "state" / "lane_maturity_audit.log"


# Substrate id-substring set mirroring Catalog #220 / #272 in-scope
# tokens. These are the lane id substrings that count as "substrate
# lanes" for the retirement audit. Per CLAUDE.md "Substrate scaffolds
# MUST be COMPLETE or RESEARCH-ONLY", non-substrate lanes (e.g.
# infrastructure-only, doc-only, gate-only) are out of scope.
_IN_SCOPE_ID_SUBSTRINGS: tuple[str, ...] = (
    "substrate_",
    "_substrate_",
    "_polytope_",
    "_sidecar_",
    "_overlay_",
    "yucr_",
    "d1_segnet",
    "d2_",
    "d4_",
    "lane_a1_",
    "_hnerv_",
    "_nerv_",
    "_lora_",
    "wavelet_residual",
    "siren_residual",
    "coord_mlp_residual",
    "nscs",
    "balle_",
    "cool_chic",
    "_c3_",
    "vq_vae",
    "self_compress",
)


_OPT_OUT_RESEARCH_TOKENS: tuple[str, ...] = (
    "research_only=true",
    "research_only:true",
    "research-only=true",
)


_OPT_OUT_SUBSTRATE_ENG_TOKENS: tuple[str, ...] = (
    "lane_class=substrate_engineering",
    "lane_class:substrate_engineering",
    "lane_class: substrate_engineering",
    "substrate_engineering_exception",
)


_OPT_OUT_ARCHIVED_TOKENS: tuple[str, ...] = (
    "archived=true",
    "lane_state=archived",
    "terminal_verdict",
)


def _is_in_scope(lane_id: str) -> bool:
    lane_id_lower = lane_id.lower()
    return any(tok in lane_id_lower for tok in _IN_SCOPE_ID_SUBSTRINGS)


def _collect_lane_text(lane: dict) -> str:
    parts: list[str] = []
    notes = lane.get("notes", "")
    if isinstance(notes, str):
        parts.append(notes)
    gates = lane.get("gates", {})
    if isinstance(gates, dict):
        for gate_obj in gates.values():
            if isinstance(gate_obj, dict):
                ev = gate_obj.get("evidence", "")
                if isinstance(ev, str):
                    parts.append(ev)
    return "\n".join(parts).lower()


def _opt_out_verdict(lane: dict, text: str) -> str | None:
    if lane.get("research_only") is True:
        return "OPT_OUT_RESEARCH_ONLY"
    if lane.get("lane_class") in ("substrate_engineering", "research_substrate"):
        return "OPT_OUT_SUBSTRATE_ENGINEERING"
    if lane.get("archived") is True:
        return "OPT_OUT_ARCHIVED"
    target_modes = lane.get("target_modes", [])
    if isinstance(target_modes, list):
        for tm in target_modes:
            if isinstance(tm, str) and tm.lower() in (
                "research_substrate",
                "research_only",
            ):
                return "OPT_OUT_RESEARCH_ONLY"
    for tok in _OPT_OUT_RESEARCH_TOKENS:
        if tok in text:
            return "OPT_OUT_RESEARCH_ONLY"
    for tok in _OPT_OUT_SUBSTRATE_ENG_TOKENS:
        if tok in text:
            return "OPT_OUT_SUBSTRATE_ENGINEERING"
    for tok in _OPT_OUT_ARCHIVED_TOKENS:
        if tok in text:
            return "OPT_OUT_ARCHIVED"
    return None


def _load_dispatch_anchors_by_lane(
    posterior_path: Path,
) -> dict[str, list[dict]]:
    """Map lane_id -> list of dispatch posterior rows."""
    by_lane: dict[str, list[dict]] = {}
    if not posterior_path.is_file():
        return by_lane
    for line in posterior_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Best-effort lane attribution: trainer + dispatch_label often
        # carry the substrate id substring.
        trainer = str(row.get("trainer", ""))
        label = str(row.get("dispatch_label", ""))
        for token, hits in by_lane.items():
            pass
        # Direct attribution attempt
        for candidate_key in ("lane_id", "lane"):
            v = row.get(candidate_key)
            if isinstance(v, str) and v.startswith("lane_"):
                by_lane.setdefault(v, []).append(row)
                break
        # Fallback: bucket under "*<trainer>*" so we can substring-match
        # lane ids against the trainer/label combo
        composite = f"{trainer}|{label}".lower()
        if composite:
            by_lane.setdefault("__fallback__", []).append(
                {**row, "_composite": composite}
            )
    return by_lane


def _load_lane_mark_timestamps(audit_log_path: Path) -> dict[str, str]:
    """Map lane_id -> most-recent ISO timestamp of any mark/add-lane event."""
    most_recent: dict[str, str] = {}
    if not audit_log_path.is_file():
        return most_recent
    for line in audit_log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        args = row.get("args", {})
        if not isinstance(args, dict):
            continue
        lane_id = args.get("lane_id") or args.get("id")
        if not isinstance(lane_id, str):
            continue
        ts = row.get("timestamp", "")
        if not isinstance(ts, str):
            continue
        existing = most_recent.get(lane_id, "")
        if ts > existing:
            most_recent[lane_id] = ts
    return most_recent


def _iso_age_days(iso_ts: str, now: datetime) -> float | None:
    if not iso_ts:
        return None
    try:
        # Strip Z if present; assume UTC
        cleaned = iso_ts.rstrip("Z").split("+")[0]
        dt = datetime.fromisoformat(cleaned).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    delta = now - dt
    return delta.total_seconds() / 86400.0


def _has_successful_dispatch(
    lane_id: str,
    dispatch_index: dict[str, list[dict]],
    staleness_days: int,
    now: datetime,
) -> bool:
    direct = dispatch_index.get(lane_id, [])
    for row in direct:
        outcome = str(row.get("outcome", "")).lower()
        if "successful_dispatch" not in outcome:
            continue
        ts = row.get("logged_at_utc", "")
        age = _iso_age_days(ts, now)
        if age is not None and age <= staleness_days:
            return True
    # Fallback substring scan against trainer / dispatch_label
    fallbacks = dispatch_index.get("__fallback__", [])
    lane_id_lower = lane_id.lower()
    for row in fallbacks:
        composite = row.get("_composite", "")
        if lane_id_lower in composite:
            outcome = str(row.get("outcome", "")).lower()
            if "successful_dispatch" not in outcome:
                continue
            ts = row.get("logged_at_utc", "")
            age = _iso_age_days(ts, now)
            if age is not None and age <= staleness_days:
                return True
    return False


def audit_stale_l1_substrates(
    *,
    repo_root: Path | None = None,
    staleness_days: int = 30,
    include_l2: bool = False,
    now: datetime | None = None,
) -> list[dict]:
    """Return a list of per-lane verdict rows.

    Each row has keys: lane_id, level, verdict, most_recent_mark_iso,
    most_recent_mark_age_days, has_recent_dispatch.
    """
    root = repo_root or REPO_ROOT
    registry_path = root / ".omx" / "state" / "lane_registry.json"
    cost_path = root / ".omx" / "state" / "cost_band_posterior.jsonl"
    audit_path = root / ".omx" / "state" / "lane_maturity_audit.log"

    if not registry_path.is_file():
        return []

    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []

    lanes = registry.get("lanes", [])
    if not isinstance(lanes, list):
        return []

    now = now or datetime.now(timezone.utc)
    dispatch_index = _load_dispatch_anchors_by_lane(cost_path)
    mark_timestamps = _load_lane_mark_timestamps(audit_path)

    rows: list[dict] = []
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        lane_id = str(lane.get("id", ""))
        if not lane_id or not _is_in_scope(lane_id):
            continue
        level = int(lane.get("level", 0) or 0)
        if level < 1:
            continue
        if level >= 2 and not include_l2:
            continue
        text = _collect_lane_text(lane)
        opt_out = _opt_out_verdict(lane, text)
        most_recent_mark = mark_timestamps.get(lane_id, "")
        mark_age = _iso_age_days(most_recent_mark, now)
        has_dispatch = _has_successful_dispatch(
            lane_id, dispatch_index, staleness_days, now
        )

        if has_dispatch:
            verdict = "ACTIVE_RECENT_DISPATCH"
        elif opt_out:
            verdict = opt_out
        elif mark_age is not None and mark_age <= staleness_days:
            verdict = "ACTIVE_RECENT_MARK"
        else:
            verdict = "STALE_PENDING_DECISION"

        rows.append({
            "lane_id": lane_id,
            "level": level,
            "verdict": verdict,
            "most_recent_mark_iso": most_recent_mark,
            "most_recent_mark_age_days": (
                round(mark_age, 1) if mark_age is not None else None
            ),
            "has_recent_dispatch": has_dispatch,
        })
    return rows


def _format_table(rows: list[dict]) -> str:
    out = []
    out.append(
        f"{'verdict':<35} {'lane_id':<60} {'L':<2} {'mark_age_days':>15}"
    )
    out.append("-" * 115)
    # Sort: STALE first, then opt-outs, then active.
    def sort_key(r):
        verdict = r["verdict"]
        priority = {
            "STALE_PENDING_DECISION": 0,
            "OPT_OUT_RESEARCH_ONLY": 1,
            "OPT_OUT_SUBSTRATE_ENGINEERING": 2,
            "OPT_OUT_ARCHIVED": 3,
            "ACTIVE_RECENT_MARK": 4,
            "ACTIVE_RECENT_DISPATCH": 5,
        }.get(verdict, 9)
        return (priority, r["lane_id"])
    for r in sorted(rows, key=sort_key):
        age_str = (
            f"{r['most_recent_mark_age_days']:.1f}"
            if r["most_recent_mark_age_days"] is not None
            else "n/a"
        )
        out.append(
            f"{r['verdict']:<35} {r['lane_id']:<60} {r['level']:<2} "
            f"{age_str:>15}"
        )
    return "\n".join(out)


def _format_summary(rows: list[dict]) -> str:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    total = len(rows)
    lines = [f"\nSummary across {total} in-scope substrate lanes:"]
    for verdict in (
        "STALE_PENDING_DECISION",
        "OPT_OUT_RESEARCH_ONLY",
        "OPT_OUT_SUBSTRATE_ENGINEERING",
        "OPT_OUT_ARCHIVED",
        "ACTIVE_RECENT_MARK",
        "ACTIVE_RECENT_DISPATCH",
    ):
        if verdict in counts:
            lines.append(f"  {verdict:<35} {counts[verdict]:>5}")
    stale = counts.get("STALE_PENDING_DECISION", 0)
    if stale:
        lines.append(
            f"\n{stale} STALE_PENDING_DECISION lane(s) require operator "
            "decision per CLAUDE.md \"Substrate retirement discipline\"."
        )
        lines.append(
            "Each must either (a) advance to L2+ via dispatch, "
            "(b) mark `research_only=true` with reactivation_criteria, "
            "OR (c) move to archived state with terminal verdict."
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true",
        help="Emit JSON rows instead of human-readable table",
    )
    parser.add_argument(
        "--staleness-days", type=int, default=30,
        help="Days since last dispatch / mark to count as stale (default 30)",
    )
    parser.add_argument(
        "--include-l2", action="store_true",
        help="Also audit L2+ lanes (default: L1 only)",
    )
    parser.add_argument(
        "--only-stale", action="store_true",
        help="Show only STALE_PENDING_DECISION rows",
    )
    args = parser.parse_args()

    rows = audit_stale_l1_substrates(
        staleness_days=args.staleness_days,
        include_l2=args.include_l2,
    )
    if args.only_stale:
        rows = [r for r in rows if r["verdict"] == "STALE_PENDING_DECISION"]

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    if not rows:
        print("No in-scope substrate lanes found.")
        return 0

    print(_format_table(rows))
    print(_format_summary(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
