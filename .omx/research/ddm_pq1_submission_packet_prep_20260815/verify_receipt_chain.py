#!/usr/bin/env python3
"""Refuse when a tracked packet document was edited without a successor receipt.

The gen6 receipt chain (DOC_DIVERGENCE_RECEIPT*.json, append-only, frozen SSD
custody) records the sha256 of every tracked packet document each time one is
deliberately changed. Eight rv17 findings shared one genus: hand-maintained
coupling between two artifacts that must agree -- a document was edited and the
record (or the document) did not follow. This script is the machine end of that
class (rv17 R8-F1 cure): it recomputes the live shas of every document the
LATEST receipt tracks and exits non-zero on any disagreement, naming the file
and the owed cure (append the next receipt; never edit an existing one).

It also refuses (rv17 R11-F2 cure) any pair whose two copies DIFFER but whose
receipt entry declares no `publish_source` ("prep" | "frozen") -- step 4A
publishes the declared copy, so an undeclared diverged pair means the publish
procedure would have to infer a source, which is the defect class itself.

Usage (defaults fit the gen6 packet; all overridable for controls):

    python3 verify_receipt_chain.py \
        [--receipts DIR] [--prep DIR] [--frozen DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# Shared derivation, not a duplicate (rv17 R16-F1): the citation-universe
# coverage check below consumes verify_citations._default_docs directly, so
# the two guards can never drift apart the way the hand-named repo_only_docs
# list drifted from the derived doc set (12 derived vs 11 tracked at round 16).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_citations as _vc

DEFAULT_RECEIPTS = Path(
    "/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen6_receipts"
)
DEFAULT_FROZEN = Path(
    "/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen6_rc2_composed"
)
DEFAULT_PREP = Path(__file__).resolve().parent

_RECEIPT_RE = re.compile(r"^DOC_DIVERGENCE_RECEIPT(?:_R(\d+))?\.json$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _latest_receipt(receipts_dir: Path) -> Path | None:
    best: tuple[int, Path] | None = None
    for path in receipts_dir.iterdir():
        match = _RECEIPT_RE.match(path.name)
        if match is None:
            continue
        # The unsuffixed receipt is the chain's oldest link (round 3).
        rank = int(match.group(1)) if match.group(1) else 3
        if best is None or rank > best[0]:
            best = (rank, path)
    return best[1] if best else None


def _derived_coverage(prep: Path, frozen: Path, tracked: set[str], failures: list[str]) -> int:
    """Refuse when a two-copy document is not tracked by the latest receipt.

    Coverage is DERIVED from the filesystem (rv17 R9-F1 cure): every top-level
    file present in BOTH trees must appear in the receipt's tracked set.
    Names are casefolded on both sides because both local filesystems resolve
    names case-insensitively -- a prep ARCHIVE_MANIFEST.json and a frozen
    archive_manifest.json are one coupled pair here and two distinct files on
    the contest's case-sensitive Linux, so a case mismatch is itself reported.
    AppleDouble ._* sidecars are excluded (environmental, #1122).

    Top-level-only is COMPLETE today because the prep tree has zero
    subdirectories (an invariant packet_census_guard enforces); if the prep
    tree ever gains a subdirectory this scope silently narrows -- revisit
    then (rv17 round-10 note).
    """
    prep_names = {p.name.casefold(): p.name for p in prep.iterdir() if p.is_file()}
    frozen_names = {
        p.name.casefold(): p.name
        for p in frozen.iterdir()
        if p.is_file() and not p.name.startswith("._")
    }
    pairs = sorted(set(prep_names) & set(frozen_names))
    tracked_folded = {name.casefold() for name in tracked}
    for key in pairs:
        if key not in tracked_folded:
            failures.append(
                f"{frozen_names[key]}: TWO-COPY DOCUMENT UNTRACKED by the latest receipt\n"
                f"  (prep {prep_names[key]} / frozen {frozen_names[key]} -- derived\n"
                f"  coverage: every prep-and-frozen pair must be receipt-tracked)"
            )
        elif prep_names[key] != frozen_names[key]:
            print(
                f"note: case mismatch prep={prep_names[key]} frozen={frozen_names[key]}"
                f" (one pair here; two files on case-sensitive filesystems)"
            )
    return len(pairs)


def _citation_universe_coverage(
    prep: Path, frozen: Path, receipts_dir: Path, tracked: set[str], failures: list[str]
) -> int:
    """Refuse when a citation-bearing doc is not tracked by the latest receipt.

    rv17 R16-F1: repo_only_docs was the LAST hand-named input set in either
    guard — at round 16 the citations checker derived 12 docs while the chain
    receipt tracked 11, leaving 8 citation-universe docs where a stale
    citation AND its covering `covered-citation:` declaration could land in
    one edit with no tracked sha moving and no receipt owed (the operative
    rule had been "track a doc once it earns an erratum", by hand, after the
    fact). Cure: the universe is DERIVED here from the SAME function the
    citations checker uses, so tracking coverage moves with the universe --
    a doc that gains its first citation immediately becomes owed a receipt.
    """
    universe = _vc._default_docs(prep, frozen, receipts_dir)
    tracked_folded = {name.casefold() for name in tracked}
    missing = 0
    for doc in universe:
        if doc.name.casefold() not in tracked_folded:
            missing += 1
            failures.append(
                f"{doc.name}: CITATION-UNIVERSE DOC UNTRACKED by the latest receipt "
                f"(rv17 R16-F1)\n"
                f"  (this doc carries FILE:LINE citations, so coverage declarations\n"
                f"  in it are load-bearing; every citation-universe doc must be\n"
                f"  receipt-tracked -- add it to repo_only_docs / frozen_only_docs\n"
                f"  in the next appended receipt)"
            )
    return len(universe) - missing


def _check(label: str, path: Path, expected: str, failures: list[str]) -> None:
    if not path.exists():
        failures.append(f"{label}: MISSING on disk ({path})")
        return
    actual = _sha256(path)
    if actual != expected:
        failures.append(
            f"{label}: sha mismatch\n"
            f"  latest receipt {expected}\n"
            f"  live file      {actual}\n"
            f"  ({path})"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    args = parser.parse_args(argv)

    if not args.receipts.is_dir():
        print(f"FAIL: receipts dir not found: {args.receipts}", file=sys.stderr)
        return 1
    latest = _latest_receipt(args.receipts)
    if latest is None:
        print(f"FAIL: no DOC_DIVERGENCE_RECEIPT*.json in {args.receipts}", file=sys.stderr)
        return 1
    receipt = json.loads(latest.read_text())

    failures: list[str] = []
    checked = 0
    for name, entry in receipt.get("diverged_files", {}).items():
        if "repo_final_sha256" in entry:
            _check(f"{name} (repo)", args.prep / name, entry["repo_final_sha256"], failures)
            checked += 1
        if "frozen_gen6_sha256" in entry:
            _check(f"{name} (frozen)", args.frozen / name, entry["frozen_gen6_sha256"], failures)
            checked += 1
        if (
            entry.get("repo_final_sha256")
            and entry.get("frozen_gen6_sha256")
            and entry["repo_final_sha256"] != entry["frozen_gen6_sha256"]
            and entry.get("publish_source") not in ("prep", "frozen")
        ):
            failures.append(
                f"{name}: DIVERGED PAIR WITHOUT DECLARED PUBLISH SOURCE (rv17 R11-F2)\n"
                f"  (copies differ; the latest receipt must carry publish_source:\n"
                f"  prep|frozen for this pair -- step 4A publishes the declared copy,\n"
                f"  never an inferred one)"
            )
    for name, entry in receipt.get("repo_only_docs", {}).items():
        if "repo_final_sha256" in entry:
            _check(f"{name} (repo)", args.prep / name, entry["repo_final_sha256"], failures)
            checked += 1
    # Frozen-only docs (rv17 R16-F1): citation-universe docs that exist only in
    # the frozen tree (e.g. README.md). The frozen custody is append-only, so a
    # tracked sha here doubles as a freeze-integrity check on that file.
    for name, entry in receipt.get("frozen_only_docs", {}).items():
        if "frozen_gen6_sha256" in entry:
            _check(f"{name} (frozen)", args.frozen / name, entry["frozen_gen6_sha256"], failures)
            checked += 1

    tracked_names = (
        set(receipt.get("diverged_files", {}))
        | set(receipt.get("repo_only_docs", {}))
        | set(receipt.get("frozen_only_docs", {}))
    )
    pair_count = _derived_coverage(args.prep, args.frozen, tracked_names, failures)
    universe_count = _citation_universe_coverage(
        args.prep, args.frozen, args.receipts, tracked_names, failures
    )

    if checked == 0:
        print(f"FAIL: latest receipt {latest.name} tracks zero documents", file=sys.stderr)
        return 1
    if failures:
        print(f"latest receipt: {latest.name}", file=sys.stderr)
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print(
            "A tracked document changed without a successor receipt (rv17 R8-F1\n"
            "class). Cure: append DOC_DIVERGENCE_RECEIPT_R<next>.json recording\n"
            "the new shas; never edit an existing receipt.",
            file=sys.stderr,
        )
        return 1
    print(
        f"PASS: {checked} tracked document shas match the latest receipt "
        f"({latest.name}); all {pair_count} derived two-copy pairs covered; "
        f"all {universe_count} citation-universe docs tracked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
