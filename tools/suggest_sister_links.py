#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Print advisory ``[[sister]]`` candidates for a note; never edits the corpus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

import corpus_query as cq  # noqa: E402

from tac.graph_memory import load_or_build, reconstruct  # noqa: E402
from tac.graph_memory.build import (  # noqa: E402
    _memory_dir,
    _memory_node_id_for_path,
    _parse_frontmatter,
    iter_wikilink_targets,
)
from tac.recall_evidence import RecallEvidence, fuse_recall  # noqa: E402


def _query_for_note(path: Path, text: str) -> str:
    name, _mtype, description = _parse_frontmatter(text)
    headings = " ".join(
        line.lstrip("# ") for line in text.splitlines() if line.startswith("#")
    )
    return " ".join(part for part in (name or path.stem, description, headings, text[:1600]) if part)


def _candidate_slug(row: RecallEvidence, memory_dir: Path) -> str | None:
    if row.ref.startswith("memory:"):
        return row.ref.removeprefix("memory:")
    raw_path = row.path.split("#L", 1)[0]
    path = Path(raw_path)
    if not path.is_absolute():
        path = _ROOT / path
    try:
        path.resolve().relative_to(memory_dir.resolve())
    except (OSError, ValueError):
        return None
    if not path.is_file() or path.suffix.lower() != ".md" or path.name == "MEMORY.md":
        return None
    if path.name.startswith("MEMORY_"):
        return None  # navigation/archive indexes are not semantic sister notes
    return _memory_node_id_for_path(path).removeprefix("memory:")


def rank_sister_candidates(
    note_path: Path,
    rows: list[RecallEvidence],
    *,
    top_k: int,
    memory_dir: Path | None = None,
) -> list[dict]:
    """Filter fused RecallEvidence rows into unique, non-self memory slugs."""
    mdir = memory_dir or _memory_dir()
    text = note_path.read_text(encoding="utf-8", errors="replace")
    name, _, _ = _parse_frontmatter(text)
    excluded = set(iter_wikilink_targets(text))
    excluded.update({name or note_path.stem, note_path.stem})
    out: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        slug = _candidate_slug(row, mdir)
        if not slug or slug in excluded or slug in seen:
            continue
        seen.add(slug)
        out.append({
            "wikilink": f"[[{slug}]]",
            "slug": slug,
            "rrf_score": row.rrf_score,
            "surfaces": list(row.contributing_surfaces),
            "source": row.path,
        })
        if len(out) >= top_k:
            break
    return out


def suggest_sister_links(note_path: Path, *, top_k: int = 10) -> list[dict]:
    """Run both #569 recall surfaces and return advisory sister candidates."""
    text = note_path.read_text(encoding="utf-8", errors="replace")
    query = _query_for_note(note_path, text)
    corpus_result = cq.run_query(query, top=max(50, top_k * 5))
    graph = load_or_build()
    reconstruction = reconstruct(graph, query, max_nodes=max(50, top_k * 5))
    rows = fuse_recall(
        corpus_result=corpus_result,
        reconstruction=reconstruction,
        graph=graph,
    )
    return rank_sister_candidates(note_path, rows, top_k=top_k)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("note", type=Path, help="new or existing markdown note")
    parser.add_argument("--top", type=int, default=10, help="number of candidates (default 10)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    if not args.note.is_file():
        parser.error(f"note does not exist: {args.note}")
    if args.top < 1:
        parser.error("--top must be positive")
    before = args.note.read_bytes()
    candidates = suggest_sister_links(args.note, top_k=args.top)
    if args.note.read_bytes() != before:
        raise RuntimeError("advisory suggester mutated its input note")
    if args.json:
        print(json.dumps({"note": str(args.note), "candidates": candidates}, indent=2))
    else:
        print(f"Advisory sister-link candidates for {args.note}:")
        for row in candidates:
            surfaces = ",".join(row["surfaces"]) or "single"
            print(f"- {row['wikilink']} — rrf={row['rrf_score']:.6f}; surfaces={surfaces}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
