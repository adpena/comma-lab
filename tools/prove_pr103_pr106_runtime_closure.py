#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove runtime closure for PR103 AC decoder bytes inside a PR106 archive.

This tool emits the missing section-length/ac-fallback metadata needed by a
PR103-aware PR106 inflate runtime.  It is not a score claim and does not load
the contest scorer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr103_pr106_runtime_closure import (  # noqa: E402
    build_runtime_closure_proof_record,
    derive_runtime_closure_from_pr106_source,
    extract_single_member_payload,
)

DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)
DEFAULT_CANDIDATE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr103_repack_pr106_standalone_20260507/archive.zip"
)
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "experiments/results/pr103_repack_pr106_standalone_20260507/manifest.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/results/pr103_repack_pr106_standalone_20260507/runtime_closure.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--candidate-archive", type=Path, default=DEFAULT_CANDIDATE_ARCHIVE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--brotli-quality", type=int, default=None)
    parser.add_argument(
        "--adaptive-lgwin",
        dest="adaptive_lgwin",
        action="store_true",
        default=None,
        help="Force adaptive lgwin on, overriding manifest/defaults.",
    )
    parser.add_argument(
        "--no-adaptive-lgwin",
        dest="adaptive_lgwin",
        action="store_false",
        help="Force adaptive lgwin off, overriding manifest/defaults.",
    )
    parser.add_argument(
        "--ac-auto-fallback",
        dest="ac_auto_fallback",
        action="store_true",
        default=None,
        help="Force PR103 AC auto-fallback on, overriding manifest/defaults.",
    )
    parser.add_argument(
        "--no-ac-auto-fallback",
        dest="ac_auto_fallback",
        action="store_false",
        help="Force PR103 AC auto-fallback off, overriding manifest/defaults.",
    )
    args = parser.parse_args()

    manifest: dict[str, object] | None = None
    if args.manifest and args.manifest.is_file():
        manifest = json.loads(args.manifest.read_text())

    brotli_quality = (
        args.brotli_quality
        if args.brotli_quality is not None
        else int((manifest or {}).get("brotli_quality", 11))
    )
    adaptive_lgwin = (
        args.adaptive_lgwin
        if args.adaptive_lgwin is not None
        else bool((manifest or {}).get("adaptive_lgwin", True))
    )
    ac_auto_fallback = (
        args.ac_auto_fallback
        if args.ac_auto_fallback is not None
        else bool((manifest or {}).get("ac_auto_fallback", True))
    )

    source = extract_single_member_payload(args.source_archive)
    candidate = extract_single_member_payload(args.candidate_archive)
    closure = derive_runtime_closure_from_pr106_source(
        source_payload=source.payload,
        candidate_payload=candidate.payload,
        brotli_quality=brotli_quality,
        adaptive_lgwin=adaptive_lgwin,
        ac_auto_fallback=ac_auto_fallback,
    )
    record = build_runtime_closure_proof_record(
        source_archive=source,
        candidate_archive=candidate,
        closure=closure,
        source_manifest=manifest,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    runtime = record["runtime_closure"]
    candidate_record = record["candidate_archive"]
    print(f"[pr103-pr106-closure] wrote {args.output}")
    print(
        "[pr103-pr106-closure] decoder="
        f"{candidate_record['decoder_section_bytes']}B "
        f"sha256={candidate_record['decoder_section_sha256'][:16]}..."
    )
    print(
        "[pr103-pr106-closure] section_lengths="
        f"{runtime['section_lengths']} ac_fallback_set={runtime['ac_fallback_set']}"
    )
    if "manifest_consistency" in record:
        print(
            "[pr103-pr106-closure] manifest_consistency="
            f"{record['manifest_consistency']['passed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
