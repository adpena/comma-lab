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

Exit codes: 0 census clean · 1 undeclared files present · 2 usage/IO error.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--packet-dir", required=True, type=Path)
    parser.add_argument(
        "--auth-eval-json",
        required=True,
        type=Path,
        help="receipt carrying provenance.inflate_runtime_manifest (the authority)",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    if not args.packet_dir.is_dir():
        print(f"REFUSE: packet dir not found: {args.packet_dir}", file=sys.stderr)
        return 2
    try:
        runtime_files = load_manifest_files(args.auth_eval_json)
    except (OSError, ValueError, KeyError) as exc:
        print(f"REFUSE: cannot read runtime manifest: {exc}", file=sys.stderr)
        return 2

    declared = set(runtime_files) | set(DECLARED_NON_RUNTIME)
    undeclared, missing = census(args.packet_dir, declared)

    verdict = "CENSUS_CLEAN" if not undeclared else "UNDECLARED_FILES_PRESENT"
    report = {
        "schema": "packet_census_guard.v1",
        "packet_dir": str(args.packet_dir),
        "auth_eval_json": str(args.auth_eval_json),
        "declared_runtime_count": len(runtime_files),
        "declared_non_runtime_count": len(DECLARED_NON_RUNTIME),
        "on_disk_count": len(undeclared) + len(declared) - len(missing),
        "undeclared_files": undeclared,
        "missing_declared_files": missing,
        "verdict": verdict,
    }
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2))

    # Denominators always printed: a census that reports only its hits is the
    # vacuity class this apparatus already paid for once.
    print(
        f"census: {len(declared)} declared "
        f"({len(runtime_files)} runtime + {len(DECLARED_NON_RUNTIME)} non-runtime) | "
        f"undeclared {len(undeclared)} | missing {len(missing)} | {verdict}"
    )
    for path in undeclared:
        print(f"  UNDECLARED: {path}")
    for path in missing:
        print(f"  MISSING:    {path}")
    return 1 if undeclared else 0


if __name__ == "__main__":
    raise SystemExit(main())
