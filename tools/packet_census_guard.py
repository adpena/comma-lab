#!/usr/bin/env python3
"""Refuse a staged submission packet that holds files nobody declared.

WHY THIS EXISTS (round-10 F1, 2026-08-18)
-----------------------------------------
Between 13:16:25Z and 13:27:21Z the staged generation-3 packet held 33 undeclared
files: 15 CPython-3.13 ``.pyc``, 18 AppleDouble ``._*`` sidecars, 3 ``__pycache__``
directories. Cause traced to source: ``.omx/tmp/sa3/probe_identity.py`` inserts the
packet directory on ``sys.path`` and imports the receiver modules in place, so
CPython writes bytecode next to the sources. Every ``.pyc`` embeds ``co_filename``
= the packet's own ABSOLUTE LOCAL PATH, which must never ship in a submission
artifact.

The round-5 fix deleted that round's ``.pyc`` and the round-9 review recorded
"zero ``.pyc``" as evidence of cleanliness. Both were instance fixes. The class
went unguarded, and the quantity they trusted is now known to be UNSTABLE between
rounds — any sibling process can repopulate it in seconds.

TWO THINGS THIS GUARD DELIBERATELY DOES NOT DO
----------------------------------------------
1. It does not scan for ``.pyc``. Scanning for the file type that bit us last time
   is the instance fix wearing a gate's clothes; the next contaminant will have a
   different extension. The invariant is a CENSUS: the packet contains the declared
   runtime manifest, the declared non-runtime surfaces, and NOTHING ELSE.

2. It does not read the runtime-tree sha as a proxy for cleanliness. That sha is
   MANIFEST-DERIVED (``pre_submission_compliance_check._runtime_tree_sha_from_manifest``)
   over a file list that contains no contaminants, so it cannot move when undeclared
   files appear — measured during the incident: all 34 manifest files re-hashed
   identical WHILE the 33 contaminants were present. A green tree sha is not evidence
   of a clean directory, and this module exists partly to record that.

TWO SURFACES, NOT ONE (round-11 F6)
-----------------------------------
``--packet-dir`` censuses the STAGED SUBMISSION directory against a manifest.
``--prep-dir`` censuses the PACKET PREP directory, which has no manifest and
cannot have one -- it is a working document set that grows every round. Round-11
F6 found the identical CWD-resolved-write mechanism on that surface: a Stop-hook
wrote ``.omx/state/*.json`` markers 31 seconds after a fix commit, into a nested
``.omx/state/`` under the prep tree, where they sat staged in the git index for
six hours. The packet-dir census could not see them and did not.

The prep-dir invariant is STRUCTURAL rather than enumerated: the prep directory
is a FLAT set of documents. No subdirectories, no dot-entries. That property
needs no manifest, never rots as documents are added, and catches the whole
CWD-resolved-write class -- every instance of which creates a nested path.

THREE SURFACES, NOT TWO (round-12 F5)
-------------------------------------
Round 12 found the AppleDouble cure had covered the STAGED TREE -- the directory
this guard walks -- and stopped there. Its sibling ``gen4_receipts/``, the custody
store every packet document cites, held one ``._*`` sidecar per real receipt, and
``generations/`` held five more one level up. Neither surface was censused by
anything. A cure that covers exactly the directory the guard happens to walk is
an instance fix wearing a gate's clothes, which is the defect this module already
exists to refuse.

``--receipts-dir`` closes it with the same structural invariant as ``--prep-dir``:
a custody store is a FLAT set of files, no nested paths, no dot-entries at any
depth. It additionally censuses the receipts directory's IMMEDIATE PARENT for
dot-entries, non-recursively -- that is the exact "one level up" surface round 12
measured, and making the operator remember to pass it separately would rebuild the
hole.

WHY DELETION IS NOT ENOUGH, MEASURED (round-12 F5)
--------------------------------------------------
On the ExFAT custody tier macOS attaches ``com.apple.provenance`` to files written
by the managed process, and the OS then materializes a ``._`` sidecar to hold that
attribute. Measured three ways during the round-12 cure: ``shutil.copy2`` produced
sidecars; ``shutil.copyfile``, which copies content only, produced them too; and a
bare ``pathlib.Path.write_text("hello")`` produced ``.__plain_write_test.txt``. The
class is NOT a copy-function choice and cannot be written around. It regenerated
inside its own cure -- the arm found 14 sidecars in a directory the review had
counted 13 in, because the arm's own receipt had just been written there. So the
only durable answer is detection: strip (``xattr -rc``) and delete at freeze, and
let this census refuse if the next write puts them back.

This guard DETECTS and REFUSES. It never deletes. A guard that repairs what it
measures cannot be trusted to report what it found.

Exit codes: 0 census clean · 1 undeclared/stray files present · 2 usage/IO error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Non-runtime surfaces that legitimately live in a staged packet. These are the
# documentation/custody files the compliance checker treats as custody-excluded;
# they are declared here explicitly so "extra file" means exactly that.
#
# ROUND-11 F3(c): two of these names -- GENERATION_RECEIPT.json and
# RECEIVER_PARSEBACK.json -- are ALSO rows in the gen-3 runtime manifest. The
# allowlist therefore MASKS them: if either were dropped from the manifest, the
# census would still accept it silently, because the union covers it from the
# other side. The names stay (they are legitimately present, and a generation
# whose manifest omits them must still stage cleanly), but the overlap is now
# REPORTED and printed loudly instead of being invisible arithmetic. A guard
# that cannot say which of its two authorities is carrying a file is a guard
# whose denominator lies.
DECLARED_NON_RUNTIME: frozenset[str] = frozenset(
    {
        "README.md",
        "report.txt",
        "archive.zip",
        "archive_manifest.json",
        "GENERATION_RECEIPT.json",
        "RECEIVER_PARSEBACK.json",
        "BORROWED_SUBSTRATE_ACCOUNTING.md",
    }
)


def load_manifest_files(auth_eval_json: Path) -> list[str]:
    """Return the declared runtime file list from an auth-eval receipt.

    The runtime manifest is the authority for what executable content the packet
    is allowed to contain; ``archive_manifest.json`` does NOT carry it.
    """
    payload = json.loads(auth_eval_json.read_text())
    manifest = payload.get("provenance", {}).get("inflate_runtime_manifest")
    if not isinstance(manifest, dict):
        raise KeyError("provenance.inflate_runtime_manifest missing or not a dict")
    for key in ("files", "runtime_files", "entries", "file_list"):
        value = manifest.get(key)
        if isinstance(value, list) and value:
            out: list[str] = []
            for entry in value:
                if isinstance(entry, str):
                    out.append(entry)
                elif isinstance(entry, dict):
                    for field in ("path", "relative_path", "name", "file"):
                        if isinstance(entry.get(field), str):
                            out.append(entry[field])
                            break
            if out:
                return out
    raise KeyError(
        "inflate_runtime_manifest carries no recognizable file list "
        f"(keys present: {sorted(manifest)})"
    )


def census(packet_dir: Path, declared: set[str]) -> tuple[list[str], list[str]]:
    """Return (undeclared_files, missing_declared) as repo-relative POSIX paths."""
    on_disk: set[str] = set()
    for dirpath, _dirnames, filenames in os.walk(packet_dir):
        for name in filenames:
            rel = Path(dirpath, name).relative_to(packet_dir).as_posix()
            on_disk.add(rel)
    undeclared = sorted(on_disk - declared)
    missing = sorted(declared - on_disk)
    return undeclared, missing


def prep_census(prep_dir: Path) -> tuple[list[str], list[str]]:
    """Return (nested_paths, dot_entries) for a packet PREP directory.

    Structural invariant, no manifest: the prep directory is a FLAT set of
    documents. Anything at depth > 1, and any dot-entry at any depth, is a
    stray. Round-11 F6's ``.omx/state/*.json`` markers are both at once.
    """
    nested: list[str] = []
    dotted: list[str] = []
    for dirpath, dirnames, filenames in os.walk(prep_dir):
        for name in sorted(filenames):
            rel = Path(dirpath, name).relative_to(prep_dir).as_posix()
            if "/" in rel:
                nested.append(rel)
            if any(part.startswith(".") for part in Path(rel).parts):
                dotted.append(rel)
        dirnames.sort()
    return sorted(set(nested)), sorted(set(dotted))


def receipts_census(receipts_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (nested_paths, dot_entries, parent_dot_entries) for a custody store.

    Same structural invariant as ``prep_census`` -- a receipts directory is a FLAT
    set of custody files -- plus a non-recursive sweep of the IMMEDIATE PARENT for
    dot-entries. Round-12 F5 measured the AppleDouble class on both surfaces at
    once, so censusing one and not the other leaves the finding half-open.
    """
    # The flat/no-dots half is EXACTLY the prep invariant; reuse it rather than
    # re-walking, so the two surfaces cannot drift apart.
    nested, dotted = prep_census(receipts_dir)

    # The parent sweep looks at FILES only. An AppleDouble sidecar for a directory
    # is itself a file (``generations/._gen4_receipts``), so nothing is missed --
    # while a legitimate dot-DIRECTORY beside the custody store (``.git`` when a
    # receipts dir lives inside a repo) does not manufacture a false stray. A guard
    # that cries wolf on ``.git`` is a guard someone turns off.
    parent_dotted: list[str] = []
    parent = receipts_dir.parent
    if parent != receipts_dir:
        try:
            entries = sorted(parent.iterdir())
        except OSError:
            entries = []
        for entry in entries:
            if entry.name.startswith(".") and entry.is_file():
                parent_dotted.append(f"../{entry.name}")
    return nested, dotted, sorted(set(parent_dotted))


def _run_receipts_census(receipts_dir: Path) -> tuple[int, dict[str, object]]:
    if not receipts_dir.is_dir():
        print(f"REFUSE: receipts dir not found: {receipts_dir}", file=sys.stderr)
        return 2, {}
    nested, dotted, parent_dotted = receipts_census(receipts_dir)
    strays = sorted(set(nested) | set(dotted) | set(parent_dotted))
    # Denominator counts REAL custody files -- a sidecar is not a receipt, and
    # counting it as one would inflate the number a reader reconciles against.
    flat_count = sum(
        1 for p in receipts_dir.iterdir() if p.is_file() and not p.name.startswith(".")
    )
    verdict = "RECEIPTS_CLEAN" if not strays else "RECEIPTS_STRAYS_PRESENT"
    report: dict[str, object] = {
        "schema": "packet_census_guard.v1",
        "mode": "receipts",
        "receipts_dir": str(receipts_dir),
        "parent_dir": str(receipts_dir.parent),
        "flat_custody_file_count": flat_count,
        "nested_paths": nested,
        "dot_entries": dotted,
        "parent_dot_entries": parent_dotted,
        "stray_count": len(strays),
        "verdict": verdict,
    }
    # Denominator first, always: a census that prints only its hits is the vacuity
    # class. flat_custody_file_count is what a reader checks the receipt list against.
    print(
        f"receipts census {receipts_dir.name}: {flat_count} flat custody file(s) | "
        f"nested {len(nested)} | dot-entries {len(dotted)} | "
        f"parent dot-entries {len(parent_dotted)} | {verdict}"
    )
    for path in strays:
        print(f"  STRAY: {path}")
    return (1 if strays else 0), report


def _run_packet_census(args: argparse.Namespace) -> tuple[int, dict[str, object]]:
    if not args.packet_dir.is_dir():
        print(f"REFUSE: packet dir not found: {args.packet_dir}", file=sys.stderr)
        return 2, {}
    if args.auth_eval_json is None:
        print("REFUSE: --packet-dir requires --auth-eval-json", file=sys.stderr)
        return 2, {}
    try:
        runtime_files = load_manifest_files(args.auth_eval_json)
    except (OSError, ValueError, KeyError) as exc:
        print(f"REFUSE: cannot read runtime manifest: {exc}", file=sys.stderr)
        return 2, {}

    runtime_set = set(runtime_files)
    declared = runtime_set | set(DECLARED_NON_RUNTIME)
    # F3(c): the printed breakdown used to ADD the two list lengths while
    # `declared` was their UNION, so the header read "39 declared (34 + 7)".
    # 34 + 7 = 41. The two missing names were the overlap, and the arithmetic
    # was the only place it was visible -- as a wrong number nobody reconciled.
    overlap = sorted(runtime_set & set(DECLARED_NON_RUNTIME))
    undeclared, missing = census(args.packet_dir, declared)

    verdict = "CENSUS_CLEAN" if not undeclared else "UNDECLARED_FILES_PRESENT"
    report: dict[str, object] = {
        "schema": "packet_census_guard.v1",
        "mode": "packet",
        "packet_dir": str(args.packet_dir),
        "auth_eval_json": str(args.auth_eval_json),
        "declared_total_count": len(declared),
        "declared_runtime_count": len(runtime_files),
        "declared_non_runtime_count": len(DECLARED_NON_RUNTIME),
        "declared_overlap": overlap,
        "declared_overlap_count": len(overlap),
        "on_disk_count": len(undeclared) + len(declared) - len(missing),
        "undeclared_files": undeclared,
        "missing_declared_files": missing,
        "verdict": verdict,
    }

    # Denominators always printed: a census that reports only its hits is the
    # vacuity class this apparatus already paid for once. The arithmetic now
    # reconciles -- union = runtime + non-runtime - overlap.
    print(
        f"census: {len(declared)} declared "
        f"({len(runtime_files)} runtime + {len(DECLARED_NON_RUNTIME)} non-runtime "
        f"- {len(overlap)} in both) | "
        f"undeclared {len(undeclared)} | missing {len(missing)} | {verdict}"
    )
    for name in overlap:
        print(
            f"  DOUBLE-DECLARED: {name} (runtime manifest AND non-runtime allowlist "
            f"-- dropping it from the manifest would NOT be caught here)"
        )
    for path in undeclared:
        print(f"  UNDECLARED: {path}")
    for path in missing:
        print(f"  MISSING:    {path}")
    return (1 if undeclared else 0), report


def _run_prep_census(args: argparse.Namespace) -> tuple[int, dict[str, object]]:
    if not args.prep_dir.is_dir():
        print(f"REFUSE: prep dir not found: {args.prep_dir}", file=sys.stderr)
        return 2, {}
    nested, dotted = prep_census(args.prep_dir)
    strays = sorted(set(nested) | set(dotted))
    flat_count = sum(1 for p in args.prep_dir.iterdir() if p.is_file())
    verdict = "PREP_CLEAN" if not strays else "PREP_STRAYS_PRESENT"
    report: dict[str, object] = {
        "schema": "packet_census_guard.v1",
        "mode": "prep",
        "prep_dir": str(args.prep_dir),
        "flat_document_count": flat_count,
        "nested_paths": nested,
        "dot_entries": dotted,
        "stray_count": len(strays),
        "verdict": verdict,
    }
    print(
        f"prep census: {flat_count} flat document(s) | "
        f"nested {len(nested)} | dot-entries {len(dotted)} | {verdict}"
    )
    for path in strays:
        print(f"  STRAY: {path}")
    return (1 if strays else 0), report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--packet-dir", type=Path)
    parser.add_argument(
        "--auth-eval-json",
        type=Path,
        help="receipt carrying provenance.inflate_runtime_manifest (the authority)",
    )
    parser.add_argument(
        "--prep-dir",
        type=Path,
        help="packet PREP directory; censused structurally (flat, no dot-entries)",
    )
    parser.add_argument(
        "--receipts-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "custody/receipts directory; censused structurally (flat, no dot-entries) "
            "together with its immediate parent. Repeatable."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    if args.packet_dir is None and args.prep_dir is None and not args.receipts_dir:
        print(
            "REFUSE: pass --packet-dir and/or --prep-dir and/or --receipts-dir",
            file=sys.stderr,
        )
        return 2

    reports: list[dict[str, object]] = []
    worst = 0
    if args.packet_dir is not None:
        rc, report = _run_packet_census(args)
        worst = max(worst, rc)
        if report:
            reports.append(report)
        if rc == 2:
            return 2
    if args.prep_dir is not None:
        rc, report = _run_prep_census(args)
        worst = max(worst, rc)
        if report:
            reports.append(report)
        if rc == 2:
            return 2
    for receipts_dir in args.receipts_dir or []:
        rc, report = _run_receipts_census(receipts_dir)
        worst = max(worst, rc)
        if report:
            reports.append(report)
        if rc == 2:
            return 2

    if args.json_out:
        payload = reports[0] if len(reports) == 1 else {"reports": reports}
        args.json_out.write_text(json.dumps(payload, indent=2))
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
