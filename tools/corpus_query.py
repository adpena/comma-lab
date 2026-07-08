#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""corpus_query — ONE query surface over the durable signal stores (#346 retrieval-first layer).

Root cause (memory `apparatus_writes_better_than_it_reads_retrieval_first_nexus_20260707`):
the apparatus is WRITE-optimized but decision-time consumption is volitional — every
2026-07-07 operator catch already existed in a store but wasn't loaded at decision time.
This tool makes "ask the whole corpus" a single deterministic primitive so the
RETRIEVAL-FIRST clause (`tac.subagent_contract.RETRIEVAL_FIRST_CLAUSE`) is cheap to honor.

Stores searched (filter with --stores):
  research   .omx/research/**/*.md (memos, DAG excluded — it has its own store)
  equations  .omx/state/canonical_equations_registry.jsonl (equation_id + payload text)
  memory     MEMORY.md + memory topic files (~/.claude/projects/<repo-slug>/memory/*.md)
  dag        the canonical DAG's FEED-* blocks (sub015_DAG_*.md)
  council    .omx/state/council_deliberation_posterior.jsonl (topics / verdicts / dissent)
  tasks      .omx/state/canonical_task_status.jsonl (task subjects; skipped if absent)
  docs       docs/*.md

Deterministic by design: plain token scoring (term-hit density + distinct-term coverage
+ recency), zero external deps, no embeddings. Same corpus + same query → same ranking.

Usage:
  .venv/bin/python tools/corpus_query.py "dash comb registration"           # ranked hits
  .venv/bin/python tools/corpus_query.py "muon warm start" --stores dag,equations --top 5
  .venv/bin/python tools/corpus_query.py "pose carrier" --json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]

STORE_NAMES: tuple[str, ...] = (
    "research", "equations", "memory", "dag", "council", "tasks", "docs",
)

_EQUATIONS_JSONL = REPO_ROOT / ".omx" / "state" / "canonical_equations_registry.jsonl"
_COUNCIL_JSONL = REPO_ROOT / ".omx" / "state" / "council_deliberation_posterior.jsonl"
_TASKS_JSONL = REPO_ROOT / ".omx" / "state" / "canonical_task_status.jsonl"
_RESEARCH_DIR = REPO_ROOT / ".omx" / "research"
_DOCS_DIR = REPO_ROOT / "docs"
_DAG_GLOB = "sub015_DAG_topaiml_reopen_and_pursuit_plan_*.md"

# Per-file read cap: scoring on the first 256 KB of a giant file is an accepted,
# deterministic heuristic (the 790 KB catalog / 1.7 MB DAG are handled specially or capped).
_FILE_READ_CAP = 256_000
_ROW_TEXT_CAP = 20_000
_DAG_BLOCK_CAP = 8_000

_STOPWORDS = frozenset(
    "the a an of and or to in for on is at by with vs from as it its our we be are was".split()
)

_DATE_COMPACT_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})")
_DATE_DASHED_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _memory_dir() -> Path:
    """The auto-memory dir for THIS repo (path-slug convention), with a fallback."""
    slug = "-" + str(REPO_ROOT).strip("/").replace("/", "-")
    cand = Path.home() / ".claude" / "projects" / slug / "memory"
    if cand.is_dir():
        return cand
    return Path.home() / ".claude" / "projects" / "-Users-adpena-Projects-pact" / "memory"


def _derive_date(text: str, fallback_mtime: float | None) -> str | None:
    """Best-effort YYYY-MM-DD from a filename/id string, else the mtime date."""
    for match in _DATE_COMPACT_RE.finditer(text):
        y, m, d = match.groups()
        if 1 <= int(m) <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{m}-{d}"
    for match in _DATE_DASHED_RE.finditer(text):
        y, m, d = match.groups()
        if 1 <= int(m) <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{m}-{d}"
    if fallback_mtime:
        return _dt.datetime.fromtimestamp(fallback_mtime, tz=_dt.timezone.utc).strftime(
            "%Y-%m-%d"
        )
    return None


def _age_days(date_str: str | None) -> float:
    if not date_str:
        return 365.0
    try:
        then = _dt.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc)
        return max(0.0, (_dt.datetime.now(tz=_dt.timezone.utc) - then).total_seconds() / 86400)
    except ValueError:
        return 365.0


def tokenize_query(query: str) -> list[str]:
    """Lowercase alnum terms, len >= 2, stopwords dropped, order-stable + deduped."""
    raw = re.findall(r"[a-z0-9_]{2,}", query.lower())
    seen: list[str] = []
    for term in raw:
        if term not in _STOPWORDS and term not in seen:
            seen.append(term)
    return seen


# ─────────────────────────── unit generators (one per store) ───────────────────────────
# Each yields {"store", "ref", "text", "date"} — ref is a path or a row id.


def _iter_md_dir(store: str, root: Path, recursive: bool) -> Iterator[dict]:
    if not root.is_dir():
        return
    pattern = "**/*.md" if recursive else "*.md"
    files = [p for p in root.glob(pattern) if p.is_file()]
    if store == "research":
        files = [p for p in files if not p.name.startswith("sub015_DAG_")]
    # Newest-first so a time-budget cutoff keeps the freshest signal.
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        try:
            mtime = path.stat().st_mtime
            text = path.read_text(errors="replace")[:_FILE_READ_CAP]
        except OSError:
            continue
        yield {
            "store": store,
            "ref": _rel(path),
            "text": text,
            "date": _derive_date(path.name, mtime),
        }


def _iter_jsonl(store: str, path: Path, id_fields: tuple[str, ...],
                date_fields: tuple[str, ...]) -> Iterator[dict]:
    if not path.is_file():
        return
    try:
        raw_lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return
    mtime = path.stat().st_mtime
    for i, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        rid = next((str(row[f]) for f in id_fields if row.get(f)), f"{path.name}:row{i}")
        date_src = next((str(row[f]) for f in date_fields if row.get(f)), "")
        yield {
            "store": store,
            "ref": rid,
            "text": json.dumps(row, sort_keys=True)[:_ROW_TEXT_CAP],
            "date": _derive_date(date_src or rid, mtime),
        }


def _iter_dag_feeds() -> Iterator[dict]:
    dag_files = sorted(_RESEARCH_DIR.glob(_DAG_GLOB))
    for dag in dag_files:
        try:
            text = dag.read_text(errors="replace")
        except OSError:
            continue
        mtime = dag.stat().st_mtime
        blocks: list[tuple[str, list[str]]] = []
        current: tuple[str, list[str]] | None = None
        for line in text.splitlines():
            if line.startswith("## FEED-"):
                if current:
                    blocks.append(current)
                current = (line.strip(), [])
            elif current is not None:
                current[1].append(line)
        if current:
            blocks.append(current)
        rel = _rel(dag)
        for header, body in blocks:
            block_text = (header + "\n" + "\n".join(body))[:_DAG_BLOCK_CAP]
            yield {
                "store": "dag",
                "ref": f"{rel} :: {header[:80]}",
                "text": block_text,
                "date": _derive_date(header, mtime),
            }


def _iter_store(store: str) -> Iterator[dict]:
    if store == "research":
        yield from _iter_md_dir("research", _RESEARCH_DIR, recursive=True)
    elif store == "docs":
        yield from _iter_md_dir("docs", _DOCS_DIR, recursive=False)
    elif store == "memory":
        yield from _iter_md_dir("memory", _memory_dir(), recursive=False)
    elif store == "dag":
        yield from _iter_dag_feeds()
    elif store == "equations":
        yield from _iter_jsonl(
            "equations", _EQUATIONS_JSONL,
            id_fields=("equation_id",),
            date_fields=("written_at_utc", "registered_at_utc"),
        )
    elif store == "council":
        yield from _iter_jsonl(
            "council", _COUNCIL_JSONL,
            id_fields=("deliberation_id", "topic"),
            date_fields=("written_at_utc", "deliberated_at_utc"),
        )
    elif store == "tasks":
        yield from _iter_jsonl(
            "tasks", _TASKS_JSONL,
            id_fields=("task_id", "title"),
            date_fields=("event_timestamp_utc", "written_at_utc"),
        )


# ─────────────────────────── scoring ───────────────────────────


def _score_unit(text: str, terms: list[str], date: str | None) -> tuple[float, dict[str, int]]:
    lowered = text.lower()
    counts = {t: lowered.count(t) for t in terms}
    distinct = sum(1 for c in counts.values() if c > 0)
    if distinct == 0:
        return 0.0, counts
    if len(terms) >= 3 and distinct < 2:
        return 0.0, counts
    total = sum(min(c, 50) for c in counts.values())
    size_kb = max(1.0, len(text) / 1000.0)
    density = total / size_kb
    recency = 6.0 / (1.0 + _age_days(date) / 30.0)
    score = distinct * 10.0 + min(total, 100) * 0.3 + min(density, 10.0) * 1.5 + recency
    return score, counts


def _matching_lines(text: str, terms: list[str], max_lines: int = 3) -> list[str]:
    scored: list[tuple[int, int, int, str]] = []
    for idx, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        distinct = sum(1 for t in terms if t in lowered)
        if distinct == 0:
            continue
        hits = sum(lowered.count(t) for t in terms)
        scored.append((distinct, hits, -idx, stripped[:200]))
    scored.sort(reverse=True)
    return [s[3] for s in scored[:max_lines]]


def run_query(
    query: str,
    stores: list[str] | None = None,
    top: int = 15,
    max_seconds: float | None = None,
    max_lines: int = 3,
) -> dict:
    """Query the durable stores. Returns {"hits": [...], "stores_consulted": {...},
    "truncated": bool, "terms": [...]}.

    Each hit: {store, ref, date, score, lines}. Deterministic for a fixed corpus.
    """
    terms = tokenize_query(query)
    result: dict = {"terms": terms, "hits": [], "stores_consulted": {}, "truncated": False}
    if not terms:
        return result
    wanted = [s for s in (stores or list(STORE_NAMES)) if s in STORE_NAMES]
    t0 = time.time()
    # best hit per (store, ref) — append-only JSONL stores repeat ids; keep the best row.
    best: dict[tuple[str, str], dict] = {}
    for store in wanted:
        n_units = 0
        for unit in _iter_store(store):
            if max_seconds is not None and (time.time() - t0) > max_seconds:
                result["truncated"] = True
                break
            n_units += 1
            score, _counts = _score_unit(unit["text"], terms, unit["date"])
            if score <= 0.0:
                continue
            key = (store, unit["ref"])
            prev = best.get(key)
            if prev is None or score > prev["score"]:
                best[key] = {
                    "store": store,
                    "ref": unit["ref"],
                    "date": unit["date"],
                    "score": round(score, 2),
                    "lines": _matching_lines(unit["text"], terms, max_lines=max_lines),
                }
        result["stores_consulted"][store] = n_units
        if result["truncated"]:
            break
    hits = sorted(best.values(), key=lambda h: (-h["score"], h["store"], h["ref"]))
    result["hits"] = hits[:top]
    result["wall_clock_s"] = round(time.time() - t0, 3)
    return result


def format_human(result: dict) -> str:
    lines: list[str] = []
    if not result["terms"]:
        return "no usable query terms (all stopwords / too short)"
    for i, hit in enumerate(result["hits"], 1):
        date = hit["date"] or "?"
        lines.append(f"{i:2d}. [{hit['store']}] {hit['ref']}  ({date}, score {hit['score']})")
        for ml in hit["lines"]:
            lines.append(f"      | {ml}")
    if not result["hits"]:
        lines.append("(no hits)")
    consulted = " ".join(f"{k}({v})" for k, v in result["stores_consulted"].items())
    lines.append(f"STORES CONSULTED: {consulted}"
                 + ("  [TRUNCATED by time budget]" if result["truncated"] else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("query", help="query string (plain terms; scored deterministically)")
    ap.add_argument("--stores", default=None,
                    help=f"comma-separated store filter from: {','.join(STORE_NAMES)}")
    ap.add_argument("--top", type=int, default=15, help="max hits returned (default 15)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="optional wall-clock budget; newest units win under truncation")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    stores = None
    if args.stores:
        stores = [s.strip() for s in args.stores.split(",") if s.strip()]
        unknown = [s for s in stores if s not in STORE_NAMES]
        if unknown:
            print(f"unknown store(s): {', '.join(unknown)} — valid: {', '.join(STORE_NAMES)}",
                  file=sys.stderr)
            return 2

    result = run_query(args.query, stores=stores, top=args.top, max_seconds=args.max_seconds)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(format_human(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
