#!/usr/bin/env python3
"""arxiv_scout.py — proactive arXiv discovery front-end for the PAPER_WARM_START engine.

WHY (operator GO 2026-07-14, from the openresearch-cli assessment): our paper intake is DEEP
(PAPER_WARM_START_FROM_DIVERGENCE traces each paper to its assumption-fork and carries it to
design+impl) but REACTIVE — it fires only when the operator drops a link. This closes the
discovery gap: the apparatus watches arXiv for OUR OWN measured cruxes and surfaces a ranked
queue, so the right papers arrive before anyone has to remember to look. Anti-forgetfulness
extended to the discovery layer.

WHAT IT IS NOT: not a summarizer, not a RAG assistant, not an experiment workbench — those are
dominated by the apparatus we already run. This is a ~200-line, stdlib-only (urllib + xml.etree)
ranked feed INTO the unchanged deep-read engine. It NEVER launches arms (CONTAINMENT): it emits
a queue; main/operator decide what to warm-start.

DEEP-READ CONTRACT (BINDING — memory paper_warm_start_is_deep_math_plus_oss_harvest_never_abstract):
a surfaced row is NOT "checked" until its PAPER_WARM_START delivers ALL of: (1) FULL MATH — whole
paper incl. appendices/proofs, load-bearing theorem(s) restated with hypotheses, key derivation
reproduced (not the abstract's summary); (2) AUTHOR-OSS HARVEST — find/read/assess the ACTUAL code
the authors + the cited lineage released (repo/license/reusable-function/transfers-to-our-vehicle?),
harvest patterns not links; (3) CRITIQUES + FORWARD citations; (4) HONEST FORK to the live task#/P0
with the $0 next-probe. Abstract-only intake dressed as a warm-start is the NO-FAKE surrogate trap at
the research layer and is FORBIDDEN. This queue FEEDS that contract; it does not replace it.

CRUX QUERIES: each row cites its MEASURED anchor (canonical equation / DAG feed / task#) so the
list is provenance-backed, not vibes. Refresh when the crux ledger moves (grep the anchors).

DEDUP: a fcntl-locked seen-ledger (.omx/state/arxiv_scout_seen.jsonl) + arXiv ids already cited
anywhere in memory/ or .omx/research/ (the papers-checked ledger + DAG mentions) are never
re-surfaced. No signal loss; no re-research (L55 discipline).

USAGE:
  .venv/bin/python tools/arxiv_scout.py                 # sweep all cruxes -> ranked queue
  .venv/bin/python tools/arxiv_scout.py --days 7        # recency window (default 14)
  .venv/bin/python tools/arxiv_scout.py --crux basis    # only cruxes whose key matches
  .venv/bin/python tools/arxiv_scout.py --mark-seen 2607.07470   # suppress an id manually
Output: ranked table to stdout + .omx/research/arxiv_scout_queue_<utcdate>.md (durable).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEEN = REPO / ".omx" / "state" / "arxiv_scout_seen.jsonl"
OUT_DIR = REPO / ".omx" / "research"
MEMORY_DIR = Path.home() / ".claude" / "projects" / "-Users-adpena-Projects-pact" / "memory"
API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"

# ── CRUX QUERIES — each provenance-backed by a MEASURED anchor (never vibes) ─────────────────
# key: short slug (for --crux filter) · query: arXiv full-text search · why: the measured crux
# + anchor it attacks. Weight = relative priority (higher = matters more per remaining descent).
CRUXES: list[dict] = [
    {"key": "basis_anisotropic", "w": 3.0,
         "query": '"anisotropic" AND ("implicit neural representation" OR "Fourier features" OR shearlet OR curvelet)',
         "why": "3.2x along-tangent frequency deficit (L25/L65, #277/#497/#502); waterfill-frequency Torralba-Weiss 2607.07470 queued"},
    {"key": "task_aware_coding", "w": 3.0,
         "query": '"coding for machines" OR "task-aware compression" OR ("rate distortion" AND segmentation)',
         "why": "contest = indirect-RD/CEO problem, floor ~0.118 (L74, #150-#155); the exact problem class"},
    {"key": "levelset_argmax", "w": 2.5,
         "query": '("level set" OR "signed distance") AND ("neural" AND (segmentation OR partition))',
         "why": "witness = viscous-HJ level-set flow of the argmax separatrix (L1/L13, #284/#318)"},
    {"key": "pontryagin_training", "w": 2.0,
         "query": '("Pontryagin" OR "optimal control") AND ("neural network training" OR hyperparameter OR curriculum)',
         "why": "costate organ lambda=dS/dx (#247/#426); n=1 organ ceiling measured (07-14 memo)"},
    {"key": "tropical_power_diagram", "w": 2.0,
         "query": '("tropical geometry" OR "power diagram" OR Laguerre) AND (neural OR compression OR quantization)',
         "why": "argmax = Laguerre power-diagram; store GENERATORS not boundaries (L75/L-v8, #284/#311)"},
    {"key": "inr_weight_compression", "w": 2.5,
         "query": '("implicit neural representation" OR INR) AND (compression OR quantization OR entropy)',
         "why": "counted bytes ARE weights; rate term 0.118 of S (#69/#157/#336/#496)"},
    {"key": "low_data_amortize", "w": 1.5,
         "query": '("test-time training" OR "single instance" OR overfitting) AND (compression OR "neural field")',
         "why": "per-video overfit-XOR-generalize; amortized init #211; organ n=1 starvation (L9)"},
    {"key": "temporal_flicker_phase", "w": 2.0,
         "query": '("temporal consistency" OR flicker OR "sub-pixel") AND (segmentation OR "video compression")',
         "why": "d_seg endgame = GT-side sub-pixel advection phase, flicker floor 0.005318 (L85/L86)"},
]

_ARXIV_ID = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")


def _known_ids() -> set[str]:
    """Every arXiv id already in the corpus (memory + research + seen-ledger) — never re-surface."""
    ids: set[str] = set()
    if SEEN.is_file():
        for ln in SEEN.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                ids.add(json.loads(ln)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    # grep is far faster than reading ~10k files in python
    for root in (str(MEMORY_DIR), str(REPO / ".omx" / "research")):
        r = subprocess.run(["grep", "-rhoE", r"\b[0-9]{4}\.[0-9]{4,5}\b", root],  # subprocess-no-check-OK: grep rc=1 means no-match (expected); an rc check would misread it as failure
                           capture_output=True, text=True)
        ids.update(m for m in r.stdout.split() if _ARXIV_ID.match(m))
    return ids


def _fetch(query: str, max_results: int) -> list[dict]:
    """One arXiv API call -> entries. Deterministic order (sortBy submittedDate desc)."""
    url = API + "?" + urllib.parse.urlencode({
        "search_query": f"all:({query})", "start": 0, "max_results": max_results,
        "sortBy": "submittedDate", "sortOrder": "descending"})
    with urllib.request.urlopen(url, timeout=30) as resp:
        tree = ET.fromstring(resp.read())
    out = []
    for e in tree.findall(f"{ATOM}entry"):
        raw = (e.findtext(f"{ATOM}id") or "").rsplit("/", 1)[-1]
        m = _ARXIV_ID.match(raw)
        out.append({
            "id": m.group(1) if m else raw,
            "title": " ".join((e.findtext(f"{ATOM}title") or "").split()),
            "abstract": " ".join((e.findtext(f"{ATOM}summary") or "").split()),
            "published": (e.findtext(f"{ATOM}published") or "")[:10],
        })
    return out


def _score(entry: dict, crux: dict) -> float:
    """Deterministic term-overlap score x crux weight. No embeddings, no network model."""
    terms = {t.strip('"()').lower() for t in re.split(r"\s+(?:AND|OR)\s+|\s+", crux["query"]) if len(t) > 3}
    text = (entry["title"] + " " + entry["abstract"]).lower()
    hits = sum(1 for t in terms if t and t in text)
    title_hits = sum(1 for t in terms if t and t in entry["title"].lower())
    return crux["w"] * (hits + 2.0 * title_hits)


def _mark_seen(ids_reasons: list[tuple[str, str]]) -> None:
    SEEN.parent.mkdir(parents=True, exist_ok=True)
    with open(SEEN, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            ts = datetime.now(UTC).isoformat()
            for pid, reason in ids_reasons:
                f.write(json.dumps({"id": pid, "reason": reason, "ts": ts}) + "\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Proactive arXiv discovery ranked against our measured cruxes.")
    ap.add_argument("--days", type=int, default=14, help="recency window (default 14)")
    ap.add_argument("--per-crux", type=int, default=25, help="API results per crux query")
    ap.add_argument("--top", type=int, default=20, help="queue size")
    ap.add_argument("--crux", help="only run cruxes whose key contains this substring")
    ap.add_argument("--mark-seen", nargs="*", help="suppress these arXiv ids and exit")
    args = ap.parse_args(argv)

    if args.mark_seen:
        _mark_seen([(pid, "operator/manual") for pid in args.mark_seen])
        print(f"marked {len(args.mark_seen)} id(s) seen")
        return 0

    known = _known_ids()
    cutoff = (datetime.now(UTC) - timedelta(days=args.days)).strftime("%Y-%m-%d")
    cruxes = [c for c in CRUXES if not args.crux or args.crux in c["key"]]
    ranked: dict[str, dict] = {}
    for c in cruxes:
        try:
            entries = _fetch(c["query"], args.per_crux)
        except Exception as exc:  # network blip: skip crux, keep sweep alive
            print(f"[warn] crux {c['key']}: fetch failed ({exc}) — skipped")
            continue
        for e in entries:
            if e["id"] in known or e["published"] < cutoff:
                continue
            s = _score(e, c)
            if s <= 0:
                continue
            cur = ranked.get(e["id"])
            if cur is None or s > cur["score"]:
                ranked[e["id"]] = {**e, "score": s, "crux": c["key"], "why": c["why"]}
        time.sleep(3)  # arXiv API politeness (their ToS asks >=3s between calls)

    queue = sorted(ranked.values(), key=lambda r: (-r["score"], r["id"]))[: args.top]
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    out = OUT_DIR / f"arxiv_scout_queue_{stamp}.md"
    lines = [f"# arXiv scout queue — {stamp} (window {args.days}d; {len(known)} ids deduped; "
             f"{len(cruxes)} cruxes)\n",
             "Each row routes into PAPER_WARM_START_FROM_DIVERGENCE (unchanged deep engine). "
             "NO auto-launch — main/operator decide.\n"]
    for r in queue:
        lines.append(f"- **{r['id']}** [{r['score']:.1f} · {r['crux']}] {r['title']} "
                     f"({r['published']})\n  - crux: {r['why']}\n"
                     f"  - https://arxiv.org/abs/{r['id']}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _mark_seen([(r["id"], f"surfaced:{r['crux']}") for r in queue])
    print("\n".join(lines))
    print(f"\n[queue written: {out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
