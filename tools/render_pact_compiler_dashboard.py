#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Render the Vehicle-OS compiler dashboard — the single living index of every
vehicle, its evidence-assigned L0-L7 maturity, allowed claim, latest typed
artifact, authority_tier/metric_family, blocker, and next command.

Source: operator binding directive 2026-06-09 + ``docs/vehicle_operating_system.md``
"Dashboard discipline" (*"every turn begins with the dashboard; no stale-memory
decisions"*). This thin CLI delegates ALL logic to the reusable
``comma_lab.pact_compiler_dashboard`` module (AGENTS.md "tac stays clean;
comma-lab owns research state"); the dashboard is generated from the
MACHINE-READABLE sources ONLY:

  - vehicle_fidelity / objective_reachability / constants_provenance manifests
    under ``.omx/state/`` (the per-vehicle identity + reachability + constants
    provenance),
  - the canonical frontier pointer (scores are POINTER-ONLY, never hardcoded),
  - the latest typed verdict JSONs on the SSD tier (G1b verdict, ladder, R2
    candidate) — fail-soft to AUDIT_PENDING when the tier is detached,
  - the subagent progress log (live running daemons/agents).

Primary output: ``pact_compiler_dashboard.{json,md}`` at the repo root (the
canonical Vehicle-OS dashboard). The historical carrier-registry triage view
(``tac.optimization.composition_carrier_registry``) is appended as a
supplementary section so no triage signal is lost.

Usage:
  .venv/bin/python tools/render_pact_compiler_dashboard.py            # write {json,md} at repo root
  .venv/bin/python tools/render_pact_compiler_dashboard.py --print json
  .venv/bin/python tools/render_pact_compiler_dashboard.py --print md
  .venv/bin/python tools/render_pact_compiler_dashboard.py --no-triage  # OS table only
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from comma_lab.pact_compiler_dashboard import (  # noqa: E402
    build_dashboard_model,
    render_json,
    render_markdown,
    write_dashboard,
)


def _carrier_triage_section() -> str:
    """Supplementary carrier-registry triage (closest-to-exact-archive first).

    Best-effort: a missing/changed registry module yields an empty string so the
    canonical OS dashboard is never blocked by the legacy triage view.
    """
    try:
        from tac.optimization.composition_carrier_registry import (
            build_canonical_registry_20260609,
            rank_candidates,
        )
    except Exception:
        return ""
    try:
        reg = rank_candidates(build_canonical_registry_20260609())
    except Exception:
        return ""
    lines: list[str] = []
    lines.append("## Supplementary: carrier triage (closest-to-exact-archive first)")
    lines.append("")
    lines.append(
        "| vehicle | composition | readiness | score+axis | blocker | next route |"
    )
    lines.append("|---|---|---|---|---|---|")
    for row in reg:
        score = row.get("current_score")
        axis = row.get("current_score_axis")
        score_s = f"{score} {axis}" if score is not None else "—"
        blocker = str(row.get("known_blocker", "")).replace("|", "\\|")[:90]
        nxt = str(
            row.get("fastest_path_to_candidate_action_evaluation", "")
        ).replace("|", "\\|")[:90]
        lines.append(
            f"| {row['vehicle']} | {row['composition_kind']} | {row['readiness']} | "
            f"{score_s} | {blocker} | {nxt} |"
        )
    lines.append("")
    return "\n".join(lines)


def _live_heartbeats_section() -> str:
    """Supplementary: active training heartbeats (non-finished)."""
    rows: list[str] = []
    for hb in sorted(glob.glob(str(_REPO / ".omx/tmp/heartbeat_*.log")))[-8:]:
        try:
            last = Path(hb).read_text().strip().splitlines()[-1]
        except (OSError, IndexError):
            continue
        if "TRAIN_EXIT" in last:
            continue
        rows.append(f"- {Path(hb).name}: {last}")
    if not rows:
        return ""
    return "## Supplementary: active training heartbeats\n\n" + "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate the Vehicle-OS compiler dashboard.")
    p.add_argument("--repo-root", type=Path, default=_REPO)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir for pact_compiler_dashboard.{json,md} (default: repo root).",
    )
    p.add_argument("--print", dest="print_kind", choices=("json", "md"), default=None)
    p.add_argument(
        "--no-triage",
        action="store_true",
        help="Omit the supplementary carrier-triage + heartbeat sections.",
    )
    args = p.parse_args(argv)

    extra = ""
    if not args.no_triage:
        triage = _carrier_triage_section()
        hb = _live_heartbeats_section()
        extra = ("\n" + triage if triage else "") + ("\n" + hb if hb else "")

    if args.print_kind:
        model = build_dashboard_model(args.repo_root)
        if args.print_kind == "json":
            sys.stdout.write(render_json(model))
        else:
            sys.stdout.write(render_markdown(model) + extra)
        return 0

    json_path, md_path = write_dashboard(args.repo_root, out_dir=args.out_dir)
    if extra:
        # Append the supplementary sections to the written Markdown.
        with md_path.open("a", encoding="utf-8") as fh:
            fh.write(extra)
    sys.stdout.write(f"wrote {json_path}\nwrote {md_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
