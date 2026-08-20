#!/usr/bin/env python3
"""Resolve every backticked FILE:LINE citation in shipped packet documents.

rv17 R10-F1 class cure: a citation carried forward after its target moved is
the same genus as the stale recipe and the stale receipt chain -- so it gets
the same machine end. For each citation whose target file exists in the
PUBLISHED packet tree, this refuses (rc=1) when:

  - the cited name only resolves case-insensitively (two files on the
    contest's case-sensitive Linux -- the R10-F1 case trap), or
  - the cited line number exceeds the file's length.

Citations whose targets are NOT packet files (repo research memos, ledgers)
are counted as external and never failed -- they are provenance breadcrumbs
a contest reviewer is not expected to resolve. Citations named inside a
section whose header contains "Erratum" are covered: reported as notes, not
failures, so append-only history can be corrected without editing it.

Usage: python3 verify_citations.py [--tree DIR] [DOC ...]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_TREE = Path(
    "/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen6_rc2_composed"
)
DEFAULT_DOCS = ("BORROWED_SUBSTRATE_ACCOUNTING.md",)

_CITE_RE = re.compile(
    r"`([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|json|md|sh|c|txt|yaml)):(\d+)"
)


def _tree_files(tree: Path) -> dict[str, list[str]]:
    """Casefolded relative path -> list of true relative paths (recursive)."""
    out: dict[str, list[str]] = {}
    for path in tree.rglob("*"):
        if path.is_file() and not path.name.startswith("._"):
            rel = str(path.relative_to(tree))
            out.setdefault(rel.casefold(), []).append(rel)
    return out


def _erratum_text(doc_text: str) -> str:
    chunks, keep = [], False
    for line in doc_text.splitlines():
        if line.startswith("#"):
            keep = "erratum" in line.casefold()
        if keep:
            chunks.append(line)
    return "\n".join(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument(
        "docs", nargs="*",
        default=[str(Path(__file__).resolve().parent / d) for d in DEFAULT_DOCS],
    )
    args = parser.parse_args(argv)
    if not args.tree.is_dir():
        print(f"FAIL: packet tree not found: {args.tree}", file=sys.stderr)
        return 1
    tree_map = _tree_files(args.tree)

    failures: list[str] = []
    stats = {"packet_ok": 0, "external": 0, "erratum_covered": 0, "ambiguous": 0}
    for doc_arg in args.docs:
        doc = Path(doc_arg)
        if not doc.exists():
            print(f"FAIL: document not found: {doc}", file=sys.stderr)
            return 1
        text = doc.read_text()
        covered = _erratum_text(text)
        for match in _CITE_RE.finditer(text):
            cited, line_no = match.group(1), int(match.group(2))
            token = f"{cited}:{match.group(2)}"
            # A cited path may be tree-relative (runtime/x.py) or bare (inflate.py).
            candidates = tree_map.get(cited.casefold(), [])
            if not candidates:
                base_hits = [
                    rels for key, rels in tree_map.items()
                    if key.rsplit("/", 1)[-1] == Path(cited).name.casefold()
                ]
                if len(base_hits) == 1:
                    candidates = base_hits[0]
                elif len(base_hits) > 1:
                    # A bare name matching several tree files cannot be resolved
                    # to one target -- counted visibly, never silently external.
                    stats["ambiguous"] += 1
                    print(f"note: {doc.name} cites {token} -- ambiguous in tree, not checked")
                    continue
            if not candidates:
                stats["external"] += 1
                continue
            if token in covered:
                stats["erratum_covered"] += 1
                print(f"note: {doc.name} cites {token} -- erratum-covered, not failed")
                continue
            true_rel = candidates[0]
            if Path(true_rel).name != Path(cited).name:
                failures.append(
                    f"{doc.name}: `{token}` resolves only case-insensitively "
                    f"(tree has {true_rel}) -- distinct files on contest Linux"
                )
                continue
            n_lines = len((args.tree / true_rel).read_text(errors="replace").splitlines())
            if line_no > n_lines:
                failures.append(
                    f"{doc.name}: `{token}` cites line {line_no} but {true_rel} "
                    f"has {n_lines} lines"
                )
            else:
                stats["packet_ok"] += 1

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print(
            "Stale-citation class (rv17 R10-F1): fix the citation, or cover it\n"
            "in an Erratum section if the citing text is append-only history.",
            file=sys.stderr,
        )
        return 1
    print(
        f"PASS: packet citations resolve ({stats['packet_ok']} verified, "
        f"{stats['erratum_covered']} erratum-covered, {stats['ambiguous']} ambiguous, "
        f"{stats['external']} external)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
