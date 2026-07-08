#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""convene — assemble a GROUNDING PACKET from the 20-store standing checklist (#346).

The T5 crucible's packet was hand-built once and the operator then had to catch missing
inventories one by one (clean-baseline memo, FEED-08l, LADDER, PowerPlay, group theory...).
This tool codifies the STANDING STORE CHECKLIST from
``.omx/research/t5_crucible/CONTEXT_COMPENDIUM_20260707.md`` so every future convening
starts complete BY CONSTRUCTION: for a topic string it runs the one-query corpus surface
(tools/corpus_query.py) once, buckets the hits into the 20 stores, adds the inline-derived
sections (run artifacts, DSL surface, durable-state freshness, git log, probe JSONs,
cross-agent channels, CLAUDE.md grep), and emits a packet .md carrying the standing
MUST-READ list + the generalized SEAT CONTRACT.

Usage:
  .venv/bin/python tools/convene.py "pose carrier byte close" \
      --out .omx/research/convenings/packet_pose_carrier_20260708.md
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

import corpus_query  # noqa: E402

_HITS_PER_SECTION = 6
_COMPENDIUM = ".omx/research/t5_crucible/CONTEXT_COMPENDIUM_20260707.md"

#: The standing MUST-READ list (grounding packet + compendium store 17; stable pointers).
MUST_READ: tuple[str, ...] = (
    "CLAUDE.md — every NON-NEGOTIABLE binds you (NO-FAKE supreme; THE GOAL sub-0.15; "
    "n600-or-not-evidence; MPS never a score; measurement-first).",
    "docs/operating_manual_craft_handoff.md — craft discipline (label MEASURED/DERIVED/"
    "INFERRED/ASSUMED out loud; attack your own conclusion; answer-first).",
    "~/.claude/projects/-Users-adpena-Projects-pact/memory/MEMORY.md — ⭐CURRENT-STATE "
    "first (the live thread).",
    ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md — newest FEED-* "
    "blocks (trajectory ground truth).",
    f"{_COMPENDIUM} — the 20-store checklist this packet instantiates.",
)

#: Generalized seat contract (codified from the T5 grounding packet §SEAT CONTRACT).
SEAT_CONTRACT = """\
- **ANTI-ANCHORING:** do NOT read sibling seats' position files before committing your own
  position. Commit blind; red-team + synthesis happen after all positions land.
- **Output sections (per position file):**
  1. `## Position` — concrete recommendation: exact configs/flags/values where possible;
     a design where a value needs measurement.
  2. `## Derivations + assumption tags` — every load-bearing claim tagged
     VERIFIED-VIA-SOURCE(path:line) / VERIFIED-VIA-ANCHOR(artifact) / INFERRED(basis) /
     ASSUMED(why unavoidable) per #363.
  3. `## Cargo-cult audit (my face)` — per inherited element: DERIVED-FROM-WITNESS-MATH |
     JUSTIFIED-KEPT(evidence) | DROP/REPLACE(replacement).
  4. `## RECESS measurement proposals` — each: what, exact command sketch, cost
     (wall-clock/mem), pre-registered predicted band + grounding, kill/proceed threshold.
  5. `## Interfaces` — what you need from / provide to the other seats.
- **Retrieval-first:** consult the durable stores (`tools/corpus_query.py "<topic>"`) before
  concluding and state a `STORES CONSULTED:` line in every decision-class doc.
- **Commit** via `tools/subagent_commit_serializer.py` with POST-EDIT working-tree sha256
  (`--expected-content-sha256 <file>=<sha>`) AND `--base-content-sha256 <file>=<pre-edit
  sha|new>` (the serializer-absorption fix); apparatus/position-only commits carry
  `[no-triality]`; `REVIEW_GATE_OVERRIDE=1` acceptable for .md only.
- **Execution envelope:** $0 read/analysis probes + CHEAP measurements inline (< ~10 min,
  < ~8 GiB, FOREGROUND only — no run_in_background, no nohup waiters; spawned long-runners
  die ~5 min). Anything bigger → a RECESS proposal. NO training launches. NO run stops.
  NO live-config changes. NO paid dispatch.
- **No reasoning-echo:** never instruct sub-processes to show/transcribe reasoning.
- **Progress:** checkpoint intermediate findings into your position file incrementally
  (crash-resumable); the FILE is the deliverable, the final message a ≤20-line summary."""


def _slugify(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:60] or "topic"


# ─────────────────── the 20-store standing checklist (codified) ───────────────────
# kind: "bucket" (filled from corpus hits), "inline" (generated), or "pointer" (static).
STANDING_CHECKLIST: tuple[dict, ...] = (
    {"num": 1, "name": "Canonical research index + post-index memos",
     "kind": "bucket", "bucket": "research_general",
     "pointers": [".omx/research/CANONICAL_RESEARCH_INDEX_20260629.md",
                  ".omx/research/*.md dated after the index"]},
    {"num": 2, "name": "Canonical equations registry", "kind": "bucket", "bucket": "equations",
     "pointers": [".omx/state/canonical_equations_registry.jsonl",
                  "tools/list_canonical_equations.py --json"]},
    {"num": 3, "name": "The DAG (FEED-* trajectory)", "kind": "bucket", "bucket": "dag",
     "pointers": [".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"]},
    {"num": 4, "name": "Council deliberation posterior", "kind": "bucket", "bucket": "council",
     "pointers": [".omx/state/council_deliberation_posterior.jsonl",
                  "tac.council_continual_learning query helpers"]},
    {"num": 5, "name": "Memory (MEMORY.md + topic files)", "kind": "bucket", "bucket": "memory",
     "pointers": ["~/.claude/projects/-Users-adpena-Projects-pact/memory/MEMORY.md "
                  "(⭐CURRENT-STATE first; papers-checked ledger L55)"]},
    {"num": 6, "name": "Lever inventories + activation ledger",
     "kind": "bucket", "bucket": "levers",
     "pointers": [".omx/research/fresh_run_master_lever_ledger_20260704.md",
                  ".omx/research/sweep_A_trainer_flag_lever_ledger_20260704.md",
                  "tac.witness_dsl.lever_registry.completeness() (unmapped flags)",
                  "tac.witness_dsl.activation_ledger (duty-to-measure; launch.sh is ground truth)"]},
    {"num": 7, "name": "Literature sweeps", "kind": "bucket", "bucket": "litsweep",
     "pointers": [".omx/research/litsweep_training_dynamics_control_20260705.md",
                  ".omx/research/litsweep_representation_taskspace_20260705.md"]},
    {"num": 8, "name": "Deep-math reviews", "kind": "bucket", "bucket": "deepmath",
     "pointers": [".omx/research/deepmath_lens_*_20260704.md (the #284 chapters)",
                  ".omx/research/group_theory_deepmath_review_20260707.md",
                  "docs/triality_dag_dsl_equations_deepmath.md"]},
    {"num": 9, "name": "Task backlog", "kind": "bucket", "bucket": "tasks",
     "pointers": ["live TaskList (SoT for #200+; the JSONL ledger is historical)",
                  ".omx/state/canonical_task_status.jsonl (historical rows)"]},
    {"num": 10, "name": "Orphan/deferral ledgers", "kind": "bucket", "bucket": "orphans",
     "pointers": [".omx/research/deferral_recovery_ledger_20260610T130200Z.md (#60)",
                  ".omx/research/sweep_C_task_research_orphan_lever_ledger_20260704.md"]},
    {"num": 11, "name": "Run artifacts (launch.sh + run.log = ground truth)", "kind": "inline",
     "pointers": ["experiments/results/levelset_n600_witness_*/launch.sh (NEVER the "
                  "activation ledger)"]},
    {"num": 12, "name": "openpilot mining", "kind": "bucket", "bucket": "openpilot",
     "pointers": [".omx/research/openpilot_cross_surface_audit_20260706.md (+ measured addendum)",
                  ".omx/research/openpilot_world_model_lane_alignment_plan_20260706.md"]},
    {"num": 13, "name": "Symposium design memos", "kind": "bucket", "bucket": "symposia",
     "pointers": ["council_symposium_clean_config_20260705.md (mod32cap authority)",
                  "council_t3_symposium_islands_treatment_arm_20260706.md"]},
    {"num": 14, "name": "DSL surface (what the program can express)", "kind": "inline",
     "pointers": ["src/tac/witness_dsl/ (curriculum_dsl, schedule, campaign, powerplay, "
                  "gauge, lever_registry, activation_ledger)"]},
    {"num": 15, "name": "CLAUDE.md-embedded measured facts", "kind": "inline",
     "pointers": ["CLAUDE.md §WITNESS-CAPSTONE lever ranking · SegNet class order · "
                  "L14-L32 · FORBIDDEN-patterns anchors"]},
    {"num": 16, "name": "Durable state files (freshness-audited)", "kind": "inline",
     "pointers": []},
    {"num": 17, "name": "docs/", "kind": "bucket", "bucket": "docs",
     "pointers": ["docs/operating_manual_craft_handoff.md",
                  "docs/meta_bug_class_catalog.md", "docs/vehicle_operating_system.md"]},
    {"num": 18, "name": "Git history (commit messages are a signal ledger)", "kind": "inline",
     "pointers": ["git log --oneline -200"]},
    {"num": 19, "name": "Probe-result data on disk (raw rows behind verdicts)", "kind": "inline",
     "pointers": ["experiments/results/*probe*/*.json (gitignored-durable; cite, don't re-run)"]},
    {"num": 20, "name": "Cross-agent channels", "kind": "inline",
     "pointers": [".omx/state/subagent_progress.jsonl", "codex inbox/outbox memos"]},
)

# research-store hits are sub-bucketed by basename pattern; first match wins.
_RESEARCH_SUBBUCKETS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("litsweep", re.compile(r"^litsweep")),
    ("deepmath", re.compile(r"deepmath|triality_dag_dsl")),
    ("openpilot", re.compile(r"openpilot")),
    ("orphans", re.compile(r"orphan|deferral")),
    ("levers", re.compile(r"lever_ledger|lever_registry|activation_ledger|duty_to_measure")),
    ("symposia", re.compile(r"^council_|symposium")),
)


def _bucket_for_hit(hit: dict) -> str:
    store = hit["store"]
    if store == "research":
        base = Path(hit["ref"].split(" :: ")[0]).name.lower()
        for bucket, pattern in _RESEARCH_SUBBUCKETS:
            if pattern.search(base):
                return bucket
        return "research_general"
    return {"equations": "equations", "dag": "dag", "council": "council",
            "memory": "memory", "tasks": "tasks", "docs": "docs"}.get(store, "research_general")


def _fmt_hits(hits: list[dict]) -> list[str]:
    lines: list[str] = []
    for hit in hits[:_HITS_PER_SECTION]:
        lines.append(f"- **{hit['ref']}** ({hit['date'] or '?'}, score {hit['score']})")
        for ml in hit["lines"][:2]:
            lines.append(f"  - `{ml[:180]}`")
    return lines


# ─────────────────────────── inline section generators (fail-open) ───────────────────────────


def _inline_run_artifacts(_terms: list[str]) -> list[str]:
    try:
        dirs = sorted((REPO_ROOT / "experiments" / "results").glob("levelset_n600_witness_*"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        return [f"- `{d.relative_to(REPO_ROOT)}/launch.sh`" for d in dirs[:6]] or \
            ["- (no witness run dirs found)"]
    except OSError as exc:
        return [f"- unavailable ({type(exc).__name__})"]


def _inline_dsl_surface(_terms: list[str]) -> list[str]:
    try:
        mods = sorted(p.name for p in (REPO_ROOT / "src" / "tac" / "witness_dsl").glob("*.py")
                      if p.name != "__init__.py")
        return [f"- modules: {', '.join(mods)}"] if mods else ["- (witness_dsl not found)"]
    except OSError as exc:
        return [f"- unavailable ({type(exc).__name__})"]


def _inline_claude_md(terms: list[str]) -> list[str]:
    try:
        need = 2 if len(terms) >= 3 else 1
        out: list[str] = []
        for i, line in enumerate((REPO_ROOT / "CLAUDE.md").read_text(errors="replace")
                                 .splitlines(), 1):
            lowered = line.lower()
            if sum(1 for t in terms if t in lowered) >= need:
                out.append(f"- `CLAUDE.md:{i}` {line.strip()[:170]}")
            if len(out) >= 5:
                break
        return out or ["- (no CLAUDE.md lines match the topic — read the pinned sections anyway)"]
    except OSError as exc:
        return [f"- unavailable ({type(exc).__name__})"]


def _inline_durable_state(_terms: list[str]) -> list[str]:
    files = [".omx/state/current_focus.md", ".omx/state/next_experiments.md",
             ".omx/research/findings.md", ".ralph/run_log.md", "reports/latest.md",
             ".omx/state/harness_failure_ledger.jsonl"]
    out: list[str] = []
    now = time.time()
    for rel in files:
        path = REPO_ROOT / rel
        if not path.exists():
            out.append(f"- `{rel}` — MISSING")
            continue
        age_d = (now - path.stat().st_mtime) / 86400
        tag = "STALE — do not cite as current state" if age_d > 14 else "live"
        out.append(f"- `{rel}` — updated {age_d:.0f}d ago ({tag})")
    return out


def _inline_git_log(terms: list[str]) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "log", "--oneline", "-300"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if proc.returncode != 0:
            return [f"- git log unavailable (rc={proc.returncode})"]
        hits = [ln for ln in proc.stdout.splitlines()
                if any(t in ln.lower() for t in terms)]
        return [f"- `{ln[:170]}`" for ln in hits[:8]] or \
            ["- (no matching commit messages in the last 300)"]
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"- git log unavailable ({type(exc).__name__})"]


def _inline_probe_data(terms: list[str]) -> list[str]:
    try:
        results = REPO_ROOT / "experiments" / "results"
        cands = list(results.glob("*probe*/*.json")) + list(results.glob("*probe*/**/*.json"))
        seen: dict[str, Path] = {str(p): p for p in cands}
        matched = [p for p in seen.values() if any(t in str(p).lower() for t in terms)]
        pick = matched or sorted(seen.values(), key=lambda p: p.stat().st_mtime, reverse=True)
        label = "topic-matched" if matched else "newest (no topic match)"
        return [f"- [{label}] `{p.relative_to(REPO_ROOT)}`" for p in pick[:6]] or \
            ["- (no probe JSONs found)"]
    except OSError as exc:
        return [f"- unavailable ({type(exc).__name__})"]


def _inline_cross_agent(_terms: list[str]) -> list[str]:
    out: list[str] = []
    path = REPO_ROOT / ".omx" / "state" / "subagent_progress.jsonl"
    try:
        rows = [json.loads(ln) for ln in path.read_text(errors="replace").splitlines()[-30:]
                if ln.strip()]
        tail = rows[-3:]
        for row in tail:
            out.append(f"- subagent `{row.get('subagent_id', '?')}` status "
                       f"{row.get('status', '?')} next: {str(row.get('next_action', ''))[:80]}")
    except (OSError, json.JSONDecodeError):
        out.append("- subagent_progress.jsonl unavailable")
    out.append("- codex channel: check .omx/research/*codex* memos for live threads")
    return out


_INLINE_GENERATORS = {
    11: _inline_run_artifacts,
    14: _inline_dsl_surface,
    15: _inline_claude_md,
    16: _inline_durable_state,
    18: _inline_git_log,
    19: _inline_probe_data,
    20: _inline_cross_agent,
}


# ─────────────────────────── packet assembly ───────────────────────────


def build_packet(topic: str, top: int = 100) -> str:
    terms = corpus_query.tokenize_query(topic)
    result = corpus_query.run_query(topic, top=top)
    buckets: dict[str, list[dict]] = {}
    for hit in result["hits"]:
        buckets.setdefault(_bucket_for_hit(hit), []).append(hit)

    now = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        f"# GROUNDING PACKET — {topic}",
        "",
        f"Auto-assembled by `tools/convene.py` on {now} from the 20-store STANDING",
        f"CHECKLIST ({_COMPENDIUM}).",
        "Per-store corpus hits below are RETRIEVAL, not synthesis — follow the pointers;",
        "a hit's absence is not evidence of absence (state `STORES CONSULTED:` anyway).",
        "",
        "## MUST-READ (in order)",
    ]
    lines += [f"{i}. {item}" for i, item in enumerate(MUST_READ, 1)]
    lines += ["", "## SEAT CONTRACT (binding)", "", SEAT_CONTRACT, "",
              "## THE 20 STORES (hits + pointers)", ""]

    for store in STANDING_CHECKLIST:
        lines.append(f"### Store {store['num']} — {store['name']}")
        for ptr in store["pointers"]:
            lines.append(f"- pointer: `{ptr}`")
        if store["kind"] == "bucket":
            hits = buckets.get(store["bucket"], [])
            if hits:
                lines += _fmt_hits(hits)
            else:
                lines.append("- (no corpus hits for this topic — consult pointers directly)")
        elif store["kind"] == "inline":
            lines += _INLINE_GENERATORS[store["num"]](terms)
        lines.append("")

    consulted = " ".join(f"{k}({v})" for k, v in result["stores_consulted"].items())
    lines.append(f"STORES CONSULTED: {consulted} + inline stores 11/14/15/16/18/19/20 "
                 f"(query terms: {', '.join(terms) or 'none'})")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("topic", help="topic string for the convening")
    ap.add_argument("--out", default=None,
                    help="output packet path (default: .omx/research/convenings/"
                         "packet_<slug>_<UTCstamp>.md)")
    ap.add_argument("--top", type=int, default=100,
                    help="corpus hits fetched before bucketing (default 100)")
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else (
        REPO_ROOT / ".omx" / "research" / "convenings" /
        f"packet_{_slugify(args.topic)}_"
        f"{_dt.datetime.now(tz=_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_packet(args.topic, top=args.top))
    print(f"packet written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
