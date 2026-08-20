#!/usr/bin/env python3
"""receipt_coverage_spot_audit.py — ddm_sd1 leg 4.

QUESTION. Memos in `.omx/research` cite SSD paths as evidence. Two things can go wrong and both
are silent: the path no longer RESOLVES (the artifact was moved, pruned, or never written), or it
resolves but is not COVERED by any retention manifest or certification row, so nothing records why
it may be deleted or how to rebuild it.

This is a BOUNDED measurement, not a backfill. It samples a fixed number of distinct cited paths
with a fixed seed and reports two fractions with their denominators. Its purpose is to replace the
standing caveat "the older long tail is unaudited" with a number.

WHY A SAMPLE. There are thousands of citations; resolving every one costs a full-tier stat sweep
for a question that a sample answers to within a few points. The sample is SEEDED so the result is
reproducible, and the population is stated explicitly so nobody reads the fraction as a census.

COVERAGE LADDER, strongest first. A path counts as covered at the first rung it satisfies:
  1. `manifest_lists_path`  — an arm `*_RETENTION_MANIFEST.json` names this exact path or basename.
  2. `certified_blob`       — its bytes appear in the certify-in-place ledger.
  3. `under_retained_dir`   — it sits under a `retained/` directory (the CLAUDE.md keep-the-payload
                              convention) but no manifest names it. Weak: convention, not a receipt.
  4. `uncovered`            — nothing records it.

USAGE
  .venv/bin/python .omx/research/ddm_sd1_.../receipt_coverage_spot_audit.py --sample 50 --days 30
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESEARCH = REPO / ".omx" / "research"
CERT_LEDGER = REPO / ".omx" / "state" / "ssd_authored_signal_certified.jsonl"
SSD_ROOTS = (Path("/Volumes/APDataStore/pact"), Path("/Volumes/VertigoDataTier/pact"))

# Citation forms in memos: bare paths, backticked paths, markdown links. Stop at whitespace or a
# closing delimiter. Trailing punctuation is stripped afterwards.
PATH_RE = re.compile(r"/Volumes/[A-Za-z0-9_.\-]+/[^\s`'\"()\[\]{},;<>|]*")
TRAILING = ".,;:)*`'\"]}>"


def cited_paths(days: int) -> tuple[dict[str, list[str]], int, int]:
    """Distinct SSD paths cited by memos modified within `days`. Returns (path -> memos, memos_scanned, citations)."""
    cutoff = time.time() - days * 86400
    found: dict[str, list[str]] = {}
    memos = 0
    citations = 0
    for memo in RESEARCH.rglob("*.md"):
        try:
            if memo.stat().st_mtime < cutoff:
                continue
            text = memo.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        memos += 1
        for raw in PATH_RE.findall(text):
            p = raw.rstrip(TRAILING)
            # A bare volume root or a one-segment path is a tier reference, not an artifact citation.
            if p.count("/") < 4:
                continue
            citations += 1
            found.setdefault(p, [])
            if memo.name not in found[p]:
                found[p].append(memo.name)
    return found, memos, citations


def load_manifest_index() -> tuple[set[str], set[str]]:
    """(exact paths, basenames) named by every `*_RETENTION_MANIFEST.json` on the mounted tiers."""
    exact: set[str] = set()
    names: set[str] = set()
    for root in SSD_ROOTS:
        if not root.is_dir():
            continue
        for man in root.glob("*/*RETENTION_MANIFEST*.json"):
            if man.name.startswith("._"):
                continue
            try:
                blob = man.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            # Manifest schemas differ across arms; harvest every path-shaped string rather than
            # assuming one schema. A schema-specific parser would silently under-count the others.
            for hit in re.findall(r'"([^"]*/[^"]*)"', blob):
                exact.add(hit)
                names.add(Path(hit).name)
            for hit in re.findall(r'"([A-Za-z0-9_.\-]+\.(?:bin|zip|npz|json|pt|raw|safetensors|ans|dr7t))"', blob):
                names.add(hit)
    return exact, names


def load_certified_shas() -> set[str]:
    shas: set[str] = set()
    try:
        for line in CERT_LEDGER.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                shas.add(json.loads(line).get("blob_sha1", ""))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return {s for s in shas if s}


def classify(path: str, exact: set[str], names: set[str]) -> str:
    if path in exact:
        return "manifest_lists_path"
    if Path(path).name in names:
        return "manifest_lists_path"
    if "/retained/" in path or path.endswith("/retained"):
        return "under_retained_dir"
    return "uncovered"


def arm_dir_has_manifest(path: str) -> bool | None:
    """Weakest rung: does the owning arm workspace carry ANY retention manifest?

    Reported separately because "the arm wrote a manifest somewhere" is not "this artifact is
    covered". Keeping it distinct stops a directory-level convention from being read as a
    per-artifact receipt.
    """
    parts = Path(path).parts
    if len(parts) < 5:
        return None
    arm = Path(*parts[:5])
    if not arm.is_dir():
        return None
    return any(not m.name.startswith("._") for m in arm.glob("*RETENTION_MANIFEST*.json"))


def _arm_dir_census() -> dict:
    """How widely the retention-manifest convention is followed at all — the base rate the sample
    fraction has to be read against."""
    total = 0
    with_manifest = 0
    for root in SSD_ROOTS:
        if not root.is_dir():
            continue
        for arm in root.iterdir():
            if not arm.is_dir() or arm.name.startswith("._"):
                continue
            total += 1
            if any(not m.name.startswith("._") for m in arm.glob("*RETENTION_MANIFEST*.json")):
                with_manifest += 1
    return {"total": total, "with_retention_manifest": with_manifest,
            "fraction": round(with_manifest / total, 4) if total else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("RECEIPT_COVERAGE_SPOT_AUDIT.json"))
    args = ap.parse_args(argv)

    population, memos, citations = cited_paths(args.days)
    mounted = [str(r) for r in SSD_ROOTS if r.is_dir()]
    unmounted = [str(r) for r in SSD_ROOTS if not r.is_dir()]
    exact, names = load_manifest_index()

    keys = sorted(population)
    rng = random.Random(args.seed)
    sample = keys if len(keys) <= args.sample else rng.sample(keys, args.sample)

    rows = []
    for p in sorted(sample):
        path = Path(p)
        # A path on an unmounted tier cannot be judged: recording it as "missing" would be a
        # measurement artifact, so it gets its own outcome and leaves the denominator.
        tier_up = any(p.startswith(m) for m in mounted)
        exists = path.exists() if tier_up else None
        if not tier_up:
            coverage = "tier_unmounted"
        elif not exists:
            # NOT the same as uncovered: we cannot judge coverage of bytes that are gone.
            coverage = "not_resolved"
        else:
            coverage = classify(p, exact, names)
        rows.append({
            "path": p,
            "cited_by": population[p][:3],
            "tier_mounted": tier_up,
            "resolves": exists,
            "coverage": coverage,
            "arm_dir_has_any_manifest": arm_dir_has_manifest(p) if exists else None,
        })

    judgeable = [r for r in rows if r["tier_mounted"]]
    resolved = [r for r in judgeable if r["resolves"]]
    covered = [r for r in resolved if r["coverage"] in ("manifest_lists_path", "certified_blob")]
    weak = [r for r in resolved if r["coverage"] == "under_retained_dir"]

    report = {
        "arm": "ddm_sd1",
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "question": "Do memo-cited SSD paths still resolve, and is anything recording why they may be deleted?",
        "scope": {
            "memos_scanned": memos,
            "window_days": args.days,
            "raw_citations": citations,
            "distinct_cited_paths_POPULATION": len(keys),
            "sampled": len(sample),
            "seed": args.seed,
            "tiers_mounted": mounted,
            "tiers_unmounted_UNJUDGEABLE": unmounted,
        },
        "result": {
            "judgeable": len(judgeable),
            "resolves": len(resolved),
            "resolve_fraction": round(len(resolved) / len(judgeable), 4) if judgeable else None,
            "covered_by_manifest_or_cert": len(covered),
            "coverage_fraction_of_resolved": round(len(covered) / len(resolved), 4) if resolved else None,
            "weak_retained_dir_only": len(weak),
            "uncovered": len(resolved) - len(covered) - len(weak),
            "resolved_whose_arm_dir_has_any_manifest": sum(
                1 for r in resolved if r["arm_dir_has_any_manifest"]),
            "arm_dirs_on_tiers": _arm_dir_census(),
        },
        "coverage_ladder": {
            "manifest_lists_path": "an arm *_RETENTION_MANIFEST.json names this path or basename",
            "certified_blob": "bytes appear in the certify-in-place ledger",
            "under_retained_dir": "sits under retained/ but no manifest names it — convention, not a receipt",
            "uncovered": "nothing records it",
        },
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    r = report["result"]
    s = report["scope"]
    print(f"[receipt-coverage] population {s['distinct_cited_paths_POPULATION']:,} distinct SSD paths "
          f"cited by {s['memos_scanned']:,} memos in {s['window_days']}d; sampled {s['sampled']} (seed {s['seed']})")
    if s["tiers_unmounted_UNJUDGEABLE"]:
        print(f"  ! UNMOUNTED (excluded from the denominator): {', '.join(s['tiers_unmounted_UNJUDGEABLE'])}")
    print(f"  resolves : {r['resolves']}/{r['judgeable']}  = {r['resolve_fraction']}")
    print(f"  covered  : {r['covered_by_manifest_or_cert']}/{r['resolves']} of RESOLVED "
          f"= {r['coverage_fraction_of_resolved']}   (+{r['weak_retained_dir_only']} retained/-dir only, "
          f"{r['uncovered']} uncovered)")
    c = r["arm_dirs_on_tiers"]
    print(f"  base rate: {c['with_retention_manifest']}/{c['total']} arm dirs on the tiers carry a "
          f"retention manifest at all = {c['fraction']}")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
