#!/usr/bin/env python3
"""Render the Pact Compiler Dashboard — the single living index of every vehicle, its
latest authority-classified evidence, blocker, and next command (operator directive
2026-06-09: "a generated artifact, not a prose memo", so the war room never goes stale).

Joins the typed sources (NEVER hand-maintained):
  - tac.optimization.composition_carrier_registry  (the V1-V6 vehicle rows + blockers)
  - .omx/state/canonical_frontier_pointer.json      (contest-CPU / contest-CUDA frontier)
  - the live training heartbeat (if a run is active)

Re-run any time; it reflects the current registry + frontier. This is how every future
Claude/Codex turn starts from one coherent surface instead of re-discovering work.

Usage:
  .venv/bin/python tools/render_pact_compiler_dashboard.py            # print + write default
  .venv/bin/python tools/render_pact_compiler_dashboard.py --out <path>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from tac.optimization.composition_carrier_registry import (  # noqa: E402
    build_canonical_registry_20260609,
    rank_candidates,
)

# Vehicle -> (V-number, role) per the operator's 6-vehicle organization.
_VEHICLE_VNUM = {
    "hinerv": ("V1", "HiNeRV/HNeRV dense carrier"),
    "hinerv_codebook": ("V1b", "HiNeRV + codebook retrofit (gated)"),
    "snerv": ("V2", "SNeRV source-state carrier"),
    "pact_nerv_vq": ("V4", "PACT-NeRV-VQ composed-latent/codebook"),
    "pr110pp": ("V5", "PR110++ selector/menu frontier-direct"),
    "source_recode": ("V0", "fp11 source-recode (the CPU frontier anchor)"),
    # NB: canonical V6 designation RESERVED for the operator's incoming V6 design memo (2026-06-09);
    # this atlas/atom lane is labeled Vatlas until V6 is defined, to avoid pre-empting it.
    "atom": ("Vatlas", "evaluator-atlas atoms (inverse-steg + cooperative-receiver miners)"),
}


def _frontier(pointer: Path) -> dict[str, object]:
    try:
        d = json.loads(pointer.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, object] = {}
    for axis in ("contest_cpu", "contest_cuda"):
        node = d.get(f"our_local_frontier_{axis}")
        if isinstance(node, dict):
            out[axis] = {"score": node.get("score"), "archive_sha256": node.get("archive_sha256"),
                         "arch": node.get("extra", {}).get("architecture_class")}
    return out


def _live_runs() -> list[str]:
    rows = []
    for hb in sorted(glob.glob(str(_REPO / ".omx/tmp/heartbeat_*.log")))[-8:]:
        try:
            last = Path(hb).read_text().strip().splitlines()[-1]
        except (OSError, IndexError):
            continue
        if "TRAIN_EXIT" in last:
            continue  # finished
        rows.append(f"{Path(hb).name}: {last}")
    return rows


def render(pointer: Path) -> str:
    reg = rank_candidates(build_canonical_registry_20260609())
    fr = _frontier(pointer)
    live = _live_runs()
    lines: list[str] = []
    lines.append("# Pact Compiler Dashboard (GENERATED — re-run render_pact_compiler_dashboard.py)")
    lines.append("")
    lines.append("V3 is the compiler/judge; every vehicle below must produce a typed "
                 "CandidateActionEvaluation (archive_sha256 + d_seg/d_pose/bytes + authority_tier + "
                 "metric_family) to be admitted. ΔS<0 on a contest-axis exact_evaluate row is the only "
                 "thing that moves the score roadmap (see pact_evidence_constitution).")
    lines.append("")
    lines.append("## Frontier (pointer-only; never hardcoded)")
    for axis, node in fr.items():
        if isinstance(node, dict):
            lines.append(f"- **{axis}**: score={node.get('score')} sha={str(node.get('archive_sha256'))[:12]} "
                         f"arch={node.get('arch')}")
    if not fr:
        lines.append("- (frontier pointer unreadable)")
    lines.append("")
    lines.append("## Live heavy/light runs")
    if live:
        for r in live:
            lines.append(f"- {r}")
    else:
        lines.append("- (no active heartbeat)")
    lines.append("")
    lines.append("## Vehicles (triage-ranked: closest-to-exact-archive first)")
    lines.append("")
    lines.append("| V | vehicle | composition | readiness | score+axis | blocker | next-action route |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in reg:
        v = row["vehicle"]
        vnum, _role = _VEHICLE_VNUM.get(v, ("V?", ""))
        score = row.get("current_score")
        axis = row.get("current_score_axis")
        score_s = f"{score} {axis}" if score is not None else "—"
        blocker = str(row.get("known_blocker", ""))[:90]
        nxt = str(row.get("fastest_path_to_candidate_action_evaluation", ""))[:90]
        lines.append(
            f"| {vnum} | {v} | {row['composition_kind']} | {row['readiness']} | "
            f"{score_s} | {blocker} | {nxt} |"
        )
    lines.append("")
    lines.append("## The non-arbitrary law (every row obeys)")
    lines.append("- authority_tier (where) × metric_family (what) × eligibility (what it may change).")
    lines.append("- score roadmap moves ONLY on contest-axis exact_evaluate; promotion ONLY on paired CPU+CUDA.")
    lines.append("- never auto-kill; never branch from telemetry/PSNR; no upstream edits for authority.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pointer", type=Path, default=_REPO / ".omx/state/canonical_frontier_pointer.json")
    p.add_argument("--out", type=Path, default=_REPO / ".omx/research/pact_compiler_dashboard.md")
    args = p.parse_args(argv)
    md = render(args.pointer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(md)
    print(f"dashboard -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
