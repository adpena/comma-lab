#!/usr/bin/env python3
"""Canonical CLI for the deterministic submission-packet compiler.

This is the **single canonical CLI entry point** for the
``tac.packet_compiler.deterministic_compiler`` module per CLAUDE.md
"Deterministic packet compiler" non-negotiable. New packet-compilation
surfaces MUST route through this CLI (or its library counterpart); enforced
by preflight Catalog #158
(``check_deterministic_compiler_canonical_use``).

Modes
=====

``identity``
    Re-emit the input packet byte-for-byte. The compiler refuses if any
    byte of the output archive differs from the input. Use this to prove
    byte-closure preserved on a round-trip + to produce golden vectors
    for future Rust/C/Zig/ASM ports.

``canonicalize``
    Normalise compliance-approved ZIP metadata only (deterministic
    timestamps, canonical permissions, canonical create_system, member
    ordering). Reports every changed byte. Refuses any change to
    score-affecting payload bytes.

``optimize``
    Emit a packet whose archive bytes are intentionally different from a
    baseline. The caller MUST supply ``--baseline-archive-sha256``,
    ``--baseline-archive-size-bytes``, ``--score-affecting-payload-changed``,
    and ``--runtime-consumption-proof``. The compiler fails closed if any
    of these are missing or if the runtime tree does not consume the new
    archive bytes.

Target profiles
===============

``contest_one_video_replay``    contest one-video overfit replay
``contest_generalized``         contest generalized (no fixed tables)
``production_generalized``      comma-ai/openpilot, portable
``production_edge_adaptive``    edge target, optional on-device learning

Per CLAUDE.md FORBIDDEN_PATTERNS:

* No /tmp paths in any persisted artifact. Default output dir is
  ``experiments/results/deterministic_packet_<utc>/``.
* No score claims. ``score_claim=false`` always.
* Never invent CLI flags: every flag emitted below was grepped against
  ``tac.packet_compiler.deterministic_compiler.compile_packet`` before
  being wired.

Usage
=====

::

    .venv/bin/python tools/build_deterministic_packet.py \\
        --input-packet experiments/results/A1_canonical/submission_dir \\
        --output-dir experiments/results/det_packet_<utc>/ \\
        --mode identity \\
        --target-profile contest_one_video_replay
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.packet_compiler.deterministic_compiler import (  # noqa: E402
    COMPILER_MODES,
    TARGET_PROFILES,
    DeterministicPacketCompilerError,
    DeterministicPacketResult,
    compile_packet,
)


def _utc_timestamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical CLI for the tac deterministic submission-packet "
            "compiler. Modes: identity / canonicalize / optimize. Profiles: "
            + ", ".join(TARGET_PROFILES) + "."
        ),
    )
    parser.add_argument(
        "--input-packet",
        type=Path,
        required=True,
        help=(
            "Input packet directory (containing archive.zip + inflate.sh + "
            "inflate.py + src/) OR a bare archive.zip whose parent dir "
            "contains the runtime tree."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output packet directory. Default: "
            "experiments/results/deterministic_packet_<UTC>/."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=COMPILER_MODES,
        default="identity",
        help="Compilation mode (identity / canonicalize / optimize).",
    )
    parser.add_argument(
        "--target-profile",
        choices=TARGET_PROFILES,
        default="contest_one_video_replay",
        help=(
            "Target profile. contest_* profiles are contest-dispatch "
            "candidates; production_* profiles are not."
        ),
    )
    parser.add_argument(
        "--baseline-archive-sha256",
        default=None,
        help=(
            "Baseline archive SHA-256 hex. Required for optimize mode; "
            "optional for identity (acts as a parity gate)."
        ),
    )
    parser.add_argument(
        "--baseline-archive-size-bytes",
        type=int,
        default=None,
        help="Baseline archive size (bytes). Required for optimize mode.",
    )
    parser.add_argument(
        "--score-affecting-payload-changed",
        action="store_true",
        help=(
            "REQUIRED for optimize mode. Acknowledges that the new "
            "archive.zip differs in score-relevant bytes from the baseline."
        ),
    )
    parser.add_argument(
        "--runtime-consumption-proof",
        action="store_true",
        help=(
            "REQUIRED for optimize mode. Asserts the caller has independently "
            "proved the new archive bytes are consumed by inflate.sh / "
            "inflate.py (e.g. via byte-mutation smoke or full-frame parity)."
        ),
    )
    parser.add_argument(
        "--allow-existing-output-dir",
        action="store_true",
        help="Replace contents of --output-dir if non-empty.",
    )
    parser.add_argument(
        "--print-result-json",
        action="store_true",
        help="Print structured result on stdout as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    out = args.output_dir
    if out is None:
        out = REPO_ROOT / "experiments" / "results" / (
            f"deterministic_packet_{_utc_timestamp()}"
        )
    elif not out.is_absolute():
        out = REPO_ROOT / out

    input_packet = args.input_packet
    if not input_packet.is_absolute():
        input_packet = REPO_ROOT / input_packet

    try:
        result: DeterministicPacketResult = compile_packet(
            input_packet=input_packet,
            output_dir=out,
            mode=args.mode,
            target_profile=args.target_profile,
            baseline_archive_sha256=args.baseline_archive_sha256,
            baseline_archive_size_bytes=args.baseline_archive_size_bytes,
            score_affecting_payload_changed=args.score_affecting_payload_changed,
            runtime_consumption_proof=args.runtime_consumption_proof,
            allow_existing_output_dir=args.allow_existing_output_dir,
        )
    except DeterministicPacketCompilerError as exc:
        print(f"[deterministic-packet-compiler] FAIL: {exc}", file=sys.stderr)
        return 2

    print(f"[deterministic-packet-compiler] mode={result.mode}")
    print(f"[deterministic-packet-compiler] target_profile={result.target_profile}")
    print(f"[deterministic-packet-compiler] output_dir={result.output_dir}")
    print(f"[deterministic-packet-compiler] archive_path={result.archive_path}")
    print(f"[deterministic-packet-compiler] archive_sha256={result.archive_sha256}")
    print(
        f"[deterministic-packet-compiler] archive_size_bytes="
        f"{result.archive_size_bytes}"
    )
    print(
        f"[deterministic-packet-compiler] runtime_tree_sha256="
        f"{result.runtime_tree_sha256}"
    )
    print(f"[deterministic-packet-compiler] score_claim={result.score_claim}")
    print(
        f"[deterministic-packet-compiler] promotion_eligible="
        f"{result.promotion_eligible}"
    )
    print(
        f"[deterministic-packet-compiler] ready_for_exact_eval_dispatch="
        f"{result.ready_for_exact_eval_dispatch}"
    )
    if result.blockers:
        print(
            f"[deterministic-packet-compiler] BLOCKERS "
            f"({len(result.blockers)}):"
        )
        for blocker in result.blockers:
            print(f"  - {blocker}")

    if args.print_result_json:
        print(
            json.dumps(
                {
                    "schema_version": result.schema_version,
                    "mode": result.mode,
                    "target_profile": result.target_profile,
                    "output_dir": result.output_dir,
                    "archive_path": result.archive_path,
                    "archive_sha256": result.archive_sha256,
                    "archive_size_bytes": result.archive_size_bytes,
                    "runtime_tree_sha256": result.runtime_tree_sha256,
                    "parser_section_manifest": result.parser_section_manifest,
                    "no_op_proof": result.no_op_proof,
                    "target_profile_policy": result.target_profile_policy,
                    "score_claim": result.score_claim,
                    "promotion_eligible": result.promotion_eligible,
                    "ready_for_exact_eval_dispatch": (
                        result.ready_for_exact_eval_dispatch
                    ),
                    "blockers": list(result.blockers),
                },
                indent=2,
                sort_keys=True,
            )
        )

    return 0 if not result.blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
