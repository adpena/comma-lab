#!/usr/bin/env python3
"""Phase 1 packet compiler CLI — dezeta-Phase-1 checkpoint -> byte-closed packet.

This is the thin CLI wrapper around ``tac.phase1_packet_compiler``. It exists
because the 2026-05-09 comprehensive adversarial review surfaced HIGH 1: the
Phase 1 trainer ``experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py``
produces a ``submission_dir/`` scaffold whose ``inflate.sh`` signature does
NOT match the contest contract (``bash inflate.sh <archive_dir>
<inflated_dir> <video_names_file>``). Without this compiler, a $80 Phase 1
GPU dispatch produces a CHECKPOINT, not byte-closed score evidence.

This tool turns a trained Phase 1 packet into a contest-compliant
``submission_dir/`` ready for ``experiments/contest_auth_eval.py`` exact
CUDA + CPU evaluation. It performs three modes of operation:

  identity        re-emit byte-for-byte (proves byte-closure preserved)
  canonicalize    re-emit normalising deterministic ZIP timestamps + sorted
                  member ordering only (refuses score-affecting byte changes)
  optimize        emit a new packet from a Phase 1 trained checkpoint
                  (score-affecting bytes change by definition; caller must
                  acknowledge and provide baseline SHA + size)

Per CLAUDE.md "Deterministic packet compiler" non-negotiable, all three modes
fail closed on hidden sidecars, scorer-at-inflate, external-state, network,
unsupported ZIP features, parser divergence, missing golden vectors, and
missing runtime tree custody. The output is a directory containing
``archive.zip``, ``inflate.sh``, ``inflate.py``, ``src/``,
``build_manifest.json``, and ``no_op_proof.json``.

Per CLAUDE.md FORBIDDEN_PATTERNS:
  * No /tmp paths in any persisted artifact (artifacts under
    ``experiments/results/<lane>_<timestamp>/`` or operator-supplied path).
  * No score claims emitted (``score_claim=false`` always; promotion +
    dispatch readiness require contest-CUDA + contest-CPU auth eval on the
    exact archive bytes).
  * Never invent CLI flags: every flag here was grepped against
    ``tac.phase1_packet_compiler.compile_phase1_packet`` before being wired.

Usage::

    .venv/bin/python tools/build_phase1_packet_compiler.py \\
        --input-packet experiments/results/A1_canonical/harvested_artifacts/finetuned_archive/submission_dir \\
        --output-dir experiments/results/phase1_packet_<timestamp>/ \\
        --mode identity \\
        --target-mode contest_one_video_replay
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

from tac.phase1_packet_compiler import (  # noqa: E402
    COMPILER_MODES,
    Phase1PacketCompilerError,
    Phase1PacketResult,
    TARGET_MODES,
    compile_phase1_packet,
)


def _utc_timestamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
            "experiments/results/phase1_packet_<UTC>/."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=COMPILER_MODES,
        default="identity",
        help="Compilation mode (identity / canonicalize / optimize).",
    )
    parser.add_argument(
        "--target-mode",
        choices=TARGET_MODES,
        default="contest_one_video_replay",
        help="Target packet profile.",
    )
    parser.add_argument(
        "--runtime-dep-closure",
        nargs="*",
        default=["torch", "brotli", "compressai"],
        help=(
            "Exhaustive runtime dep list recorded in build_manifest. Default: "
            "torch brotli compressai (Ballé hyperprior baseline)."
        ),
    )
    parser.add_argument(
        "--export-format",
        default="phase1_three_member_x_decoder_bin_balle_bin",
        help="HNeRV-parity export_format field (recorded in manifest).",
    )
    parser.add_argument(
        "--bolt-on-loc-budget",
        type=int,
        default=400,
        help="HNeRV-parity bolt_on_loc_budget field (recorded in manifest).",
    )
    parser.add_argument(
        "--allow-existing-output-dir",
        action="store_true",
        help="Replace contents of output_dir if non-empty.",
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
        "--baseline-archive-sha256",
        default=None,
        help="REQUIRED for optimize mode: baseline archive SHA-256 hex.",
    )
    parser.add_argument(
        "--baseline-archive-size-bytes",
        type=int,
        default=None,
        help="REQUIRED for optimize mode: baseline archive size (bytes).",
    )
    parser.add_argument(
        "--no-fail-on-score-affecting-change",
        dest="fail_on_score_affecting_change",
        action="store_false",
        help=(
            "Diagnostic-only: don't refuse canonicalize mode if it would "
            "alter payload bytes. Default: fail closed."
        ),
    )
    parser.set_defaults(fail_on_score_affecting_change=True)
    parser.add_argument(
        "--print-result-json",
        action="store_true",
        help="Print the structured result as JSON on stdout (for piping).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.output_dir is None:
        out = REPO_ROOT / "experiments" / "results" / f"phase1_packet_{_utc_timestamp()}"
    else:
        out = args.output_dir
        if not out.is_absolute():
            out = REPO_ROOT / out

    input_packet = args.input_packet
    if not input_packet.is_absolute():
        input_packet = REPO_ROOT / input_packet

    try:
        result: Phase1PacketResult = compile_phase1_packet(
            input_packet=input_packet,
            output_dir=out,
            mode=args.mode,
            target_mode=args.target_mode,
            runtime_dep_closure=args.runtime_dep_closure,
            export_format=args.export_format,
            bolt_on_loc_budget=args.bolt_on_loc_budget,
            allow_existing_output_dir=args.allow_existing_output_dir,
            score_affecting_payload_changed=args.score_affecting_payload_changed,
            baseline_archive_sha256=args.baseline_archive_sha256,
            baseline_archive_size_bytes=args.baseline_archive_size_bytes,
            fail_on_score_affecting_change=args.fail_on_score_affecting_change,
        )
    except Phase1PacketCompilerError as exc:
        print(f"[phase1-packet-compiler] FAIL: {exc}", file=sys.stderr)
        return 2

    print(f"[phase1-packet-compiler] mode={result.mode}")
    print(f"[phase1-packet-compiler] target_mode={result.target_mode}")
    print(f"[phase1-packet-compiler] output_dir={result.output_dir}")
    print(f"[phase1-packet-compiler] archive_path={result.archive_path}")
    print(f"[phase1-packet-compiler] archive_sha256={result.archive_sha256}")
    print(f"[phase1-packet-compiler] archive_size_bytes={result.archive_size_bytes}")
    print(f"[phase1-packet-compiler] runtime_tree_sha256={result.runtime_tree_sha256}")
    print(f"[phase1-packet-compiler] score_claim={result.score_claim}")
    print(f"[phase1-packet-compiler] promotion_eligible={result.promotion_eligible}")
    print(
        f"[phase1-packet-compiler] ready_for_exact_eval_dispatch="
        f"{result.ready_for_exact_eval_dispatch}"
    )
    if result.blockers:
        print(f"[phase1-packet-compiler] BLOCKERS ({len(result.blockers)}):")
        for blocker in result.blockers:
            print(f"  - {blocker}")
    if args.print_result_json:
        print(
            json.dumps(
                {
                    "schema_version": result.schema_version,
                    "mode": result.mode,
                    "target_mode": result.target_mode,
                    "output_dir": result.output_dir,
                    "archive_sha256": result.archive_sha256,
                    "archive_size_bytes": result.archive_size_bytes,
                    "runtime_tree_sha256": result.runtime_tree_sha256,
                    "score_claim": result.score_claim,
                    "promotion_eligible": result.promotion_eligible,
                    "ready_for_exact_eval_dispatch": result.ready_for_exact_eval_dispatch,
                    "blockers": list(result.blockers),
                },
                indent=2,
            )
        )

    return 0 if not result.blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
