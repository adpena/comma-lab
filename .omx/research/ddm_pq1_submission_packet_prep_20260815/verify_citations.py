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
a contest reviewer is not expected to resolve. A citation is erratum-covered
(reported as a note, not a failure, so append-only history can be corrected
without editing it) ONLY when an explicit `covered-citation:` declaration
line names its exact token inside a section whose header LEADS with the
word "Erratum" (rv17 R11-F1: the previous substring test let an unrelated
erratum mention -- or a header reading "This is NOT an erratum" -- launder a
genuine failure; a declaration can still be false, but it is a deliberate,
greppable, reviewable line rather than an accident of prose).

Usage: python3 verify_citations.py [--tree DIR] [DOC ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_TREE = Path(
    "/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen6_rc2_composed"
)
DEFAULT_PREP = Path(__file__).resolve().parent
DEFAULT_RECEIPTS = Path(
    "/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen6_receipts"
)
_DOC_SUFFIXES = (".md", ".txt")
_RECEIPT_RE = re.compile(r"^DOC_DIVERGENCE_RECEIPT(?:_R(\d+))?\.json$")


def _publish_sources(receipts_dir: Path) -> dict[str, str]:
    """Casefolded doc name -> declared publish_source from the LATEST receipt."""
    best: tuple[int, Path] | None = None
    if receipts_dir.is_dir():
        for path in receipts_dir.iterdir():
            match = _RECEIPT_RE.match(path.name)
            if match is None:
                continue
            rank = int(match.group(1)) if match.group(1) else 3
            if best is None or rank > best[0]:
                best = (rank, path)
    if best is None:
        return {}
    receipt = json.loads(best[1].read_text())
    return {
        name.casefold(): entry["publish_source"]
        for name, entry in receipt.get("diverged_files", {}).items()
        if entry.get("publish_source") in ("prep", "frozen")
    }


def _default_docs(prep: Path, frozen: Path, receipts_dir: Path) -> list[Path]:
    """DERIVE the checked doc set (rv17 R14-F1): every top-level .md/.txt in the
    prep/frozen UNION whose text carries a FILE:LINE citation. Two-copy names
    resolve to the copy the latest receipt's publish_source declares (identical
    or undeclared pairs default to frozen, the publish baseline); prep-only
    docs check the prep copy, frozen-only the frozen copy. Hand-naming the set
    was the round-14 finding -- the derived set immediately surfaced two ACTIVE
    stale citations (REVIEW_PASS6/9 manifest references) that three rounds of
    named sets had missed.

    Two hand-named boundaries remain, both MEASURED empty at rv17 R15 (not
    justified by the prep flatness invariant, which bounds an INTERSECTION
    while this universe is a UNION containing frozen subdirectory files):
    the 34 frozen subdirectory files and every non-.md/.txt top-level file
    carry ZERO _CITE_RE matches -- the subdirectories are runtime code, not
    prose. If either measurement ever changes, the derived-set print in
    main() makes the narrowing visible rather than silent."""
    sources = _publish_sources(receipts_dir)
    prep_names = {p.name.casefold(): p for p in prep.iterdir()
                  if p.is_file() and p.suffix in _DOC_SUFFIXES}
    frozen_names = {p.name.casefold(): p for p in frozen.iterdir()
                    if p.is_file() and p.suffix in _DOC_SUFFIXES
                    and not p.name.startswith("._")}
    docs: list[Path] = []
    for key in sorted(set(prep_names) | set(frozen_names)):
        if key in prep_names and key in frozen_names:
            chosen = prep_names[key] if sources.get(key) == "prep" else frozen_names[key]
        elif key in prep_names:
            chosen = prep_names[key]
        else:
            chosen = frozen_names[key]
        if _CITE_RE.search(chosen.read_text(errors="replace")):
            docs.append(chosen)
    return docs

# A citation is any filename-shaped backticked token followed by :N. The
# extension is NOT enumerated (rv17 R17-F1: the old (py|json|md|sh|c|txt|yaml)
# list was a hand-chosen membership test written in regex syntax, sitting
# upstream of every derived set -- and it leaked exactly as DEFAULT_DOCS did:
# `yml` unlisted made 4 live eval.yml tokens invisible, and a natural
# `MANIFEST.sha256:N` citation would have been silently unchecked). Matching
# the SHAPE and letting the three-way classification (in-tree checked /
# erratum-covered / external) absorb non-file lookalikes is the same
# classify-don't-prefilter move that cured the doc set; a false positive can
# only land in the non-failing external bucket unless a real tree file bears
# its name, in which case checking it is correct.
_CITE_RE = re.compile(
    r"`([A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9_]{1,12}):(\d+)"
)


def _tree_files(tree: Path) -> dict[str, list[str]]:
    """Casefolded relative path -> list of true relative paths (recursive)."""
    out: dict[str, list[str]] = {}
    for path in tree.rglob("*"):
        if path.is_file() and not path.name.startswith("._"):
            rel = str(path.relative_to(tree))
            out.setdefault(rel.casefold(), []).append(rel)
    return out


# Header must LEAD with "Erratum" after optional numbering ("### 10.6 Erratum -- ..."
# qualifies; "## This is NOT an erratum" does not).
_ERRATUM_HEADER_RE = re.compile(r"^#+\s*[0-9.§()\s]*erratum\b", re.IGNORECASE)
# A declaration may be indented at most 3 spaces: at 4+ it is an INDENTED code
# block in Markdown, i.e. illustration, and must not grant coverage (rv17
# R13-F1 sibling, same ≤3 rule as fences).
_DECLARE_RE = re.compile(r"^ {0,3}covered-citation:\s*`([^`]+)`")
# A fence OPENS on a run of >=3 backticks or tildes indented at most 3 spaces
# (CommonMark); it CLOSES only on a run of >= the opening length of the SAME
# character with nothing but whitespace after. rv17 R13-F1: the one-boolean
# lstrip/startswith tracker approximated this rule and was fail-OPEN two ways
# (a ~~~ line inside a backtick fence, and an indented ``` inside a fence,
# both "closed" the fence and resumed coverage-granting inside illustration).
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _declared_coverage(doc_text: str) -> set[str]:
    """Exact tokens declared covered on `covered-citation:` lines in Erratum sections.

    Code-fence aware (rv17 R12-F1 / R13-F1): text inside ``` / ~~~ fences is
    ILLUSTRATION, not declaration -- a fenced Erratum-header-plus-declaration
    example must not silently open real coverage, and a fenced `# comment`
    line must not silently CLOSE a live Erratum section. The fence state is a
    (char, count) tuple, not a boolean: only a same-character run of at least
    the opening length closes a block, and fence-looking lines of the OTHER
    character or at 4+ spaces of indent are content.
    """
    covered: set[str] = set()
    in_erratum = False
    fence: tuple[str, int] | None = None
    for line in doc_text.splitlines():
        fenced = _FENCE_RE.match(line)
        if fence is not None:
            if (
                fenced
                and fenced.group(1)[0] == fence[0]
                and len(fenced.group(1)) >= fence[1]
                and not fenced.group(2).strip()
            ):
                fence = None
            continue
        if fenced:
            fence = (fenced.group(1)[0], len(fenced.group(1)))
            continue
        if line.startswith("#"):
            in_erratum = bool(_ERRATUM_HEADER_RE.match(line))
            continue
        if in_erratum:
            declared = _DECLARE_RE.match(line)
            if declared:
                covered.add(declared.group(1))
    return covered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("docs", nargs="*")
    args = parser.parse_args(argv)
    if not args.tree.is_dir():
        print(f"FAIL: packet tree not found: {args.tree}", file=sys.stderr)
        return 1
    if args.docs:
        doc_paths = [Path(d) for d in args.docs]
    else:
        doc_paths = _default_docs(args.prep, args.tree, args.receipts)
        if not doc_paths:
            print("FAIL: derived doc set is EMPTY (vacuity guard)", file=sys.stderr)
            return 1
        print(f"derived doc set ({len(doc_paths)}): "
              + ", ".join(p.name for p in doc_paths))
    tree_map = _tree_files(args.tree)
    sources = _publish_sources(args.receipts)
    prep_top = {p.name.casefold(): p for p in args.prep.iterdir() if p.is_file()}

    failures: list[str] = []
    stats = {"packet_ok": 0, "external": 0, "erratum_covered": 0, "ambiguous": 0}
    for doc in doc_paths:
        if not doc.exists():
            print(f"FAIL: document not found: {doc}", file=sys.stderr)
            return 1
        text = doc.read_text()
        covered = _declared_coverage(text)
        for match in _CITE_RE.finditer(text):
            cited, line_no = match.group(1), int(match.group(2))
            token = f"{cited}:{match.group(2)}"
            # rv17 R18-note cure (#1172, post-seal): publish_source governs
            # RESOLUTION as well as selection. A two-copy name whose latest
            # receipt declares publish_source=prep is PUBLISHED from the prep
            # tree, so its citations resolve against the prep copy (existence,
            # exact case, line count). Frozen-resolution for such names was
            # measured strictly conservative (loud false failure, never a
            # silent pass; zero live exposure at R19) but would flip fail-open
            # the first time a prep-published copy became SHORTER than frozen.
            if "/" not in cited and sources.get(cited.casefold()) == "prep":
                prep_path = prep_top.get(cited.casefold())
                if prep_path is not None:
                    if token in covered:
                        stats["erratum_covered"] += 1
                        print(f"note: {doc.name} cites {token} -- erratum-covered, not failed")
                        continue
                    if prep_path.name != cited:
                        failures.append(
                            f"{doc.name}: `{token}` resolves only case-insensitively "
                            f"(published prep copy is {prep_path.name}) -- distinct "
                            f"files on contest Linux"
                        )
                        continue
                    n_lines = len(prep_path.read_text(errors="replace").splitlines())
                    if line_no > n_lines:
                        failures.append(
                            f"{doc.name}: `{token}` cites line {line_no} but the "
                            f"published prep copy of {prep_path.name} has {n_lines} lines"
                        )
                    else:
                        stats["packet_ok"] += 1
                    continue
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
            "Stale-citation class (rv17 R10-F1): fix the citation, or declare it\n"
            "on a `covered-citation:` line inside an Erratum section (R11-F1) if\n"
            "the citing text is append-only history.",
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
