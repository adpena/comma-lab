#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for the canonical archive grammar builder (Phase 3).

Sister of :mod:`tools.compression_pipeline_cli` (Phase 2) + the canonical
META-LIFT CLIs at ``tools/cross_substrate_master_gradient_cli.py`` +
``tools/pareto_polytope_unified_solver_cli.py`` +
``tools/uniward_invariant_enumerator_cli.py``.

Wraps :func:`tac.submission_packet.build_archive_grammar_from_compression_pipeline_result`
with operator-friendly flags + canonical exit codes per Phase 1 audit
specification memo §7:

Exit codes:
  0 CLEAN — archive grammar manifest emitted, all invariants satisfied
  1 MANIFEST-INVALID — HNeRV parity L3 invariants violated
  2 NO-OP-DETECTED-BYTES-NOT-CONSUMED — Catalog #266 violation (the
      research-substrate trap, 8th forbidden pattern)
  3 SECTION-OVERLAP — Catalog #146 fixed-offset discipline violation
  4 CLI error (missing required arg, invalid path, etc.)

Per the 12th canonicalization × standardization × ease-of-contest-
compliance trinity: ONE CLI invocation prepares the canonical archive
grammar manifest + emits the parser_section_manifest.json sidecar +
(optionally) runs the Catalog #272 byte-mutation smoke.

Per the 11th ORDER-MATTERS standing directive: this CLI is the SECOND
Phase 1 spec consumer (Phase 2 CLI was the first); future Phase 4-10
CLIs depend on this CLI's output shape.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tac.submission_packet.archive_grammar import (  # noqa: E402
    ArchiveGrammarError,
    ByteMutationSmokeVerdict,
    build_archive_grammar_from_compression_pipeline_result,
)
from tac.submission_packet.compression_pipeline import (  # noqa: E402
    CompressionPipelineError,
    HardwareSubstrateClass,
    build_compression_pipeline,
)


EXIT_OK = 0
EXIT_MANIFEST_INVALID = 1
EXIT_NO_OP_DETECTED = 2
EXIT_SECTION_OVERLAP = 3
EXIT_CLI_ERROR = 4


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="archive_grammar_cli",
        description=(
            "Canonical archive grammar builder (Phase 3, Layer 1). "
            "Wraps tac.submission_packet.build_archive_grammar_from_compression_pipeline_result "
            "with operator-friendly flags. Per HNeRV parity L3: monolithic single-file "
            "0.bin (or explicitly justified multi-file)."
        ),
    )
    parser.add_argument(
        "--lane-id",
        required=True,
        help="Lane registry id (e.g. 'lane_pr111_candidate_20260601').",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path("upstream/videos/0.mkv"),
        help="Path to the contest video (canonical: upstream/videos/0.mkv).",
    )
    parser.add_argument(
        "--substrate-trainer",
        type=Path,
        required=True,
        help="Path to experiments/train_substrate_<id>.py.",
    )
    parser.add_argument(
        "--recipe-path",
        type=Path,
        required=True,
        help="Path to .omx/operator_authorize_recipes/substrate_<id>_*.yaml.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        required=True,
        help="Path to the trainer-emitted archive.zip.",
    )
    parser.add_argument(
        "--monolithic-single-file",
        action="store_true",
        default=True,
        help="HNeRV parity L3 monolithic single-file 0.bin (default True).",
    )
    parser.add_argument(
        "--multi-file",
        action="store_true",
        default=False,
        help="Override to multi-file archive; requires --multi-file-justification.",
    )
    parser.add_argument(
        "--multi-file-justification",
        type=str,
        default=None,
        help="Substantive rationale (>=4 chars, non-placeholder per Catalog #287) "
             "when --multi-file is set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for sidecars; defaults to archive_path.parent.",
    )
    parser.add_argument(
        "--inflate-sh-path",
        type=Path,
        default=None,
        help="Path to inflate.sh (required when --verify-byte-mutation-smoke).",
    )
    parser.add_argument(
        "--verify-byte-mutation-smoke",
        action="store_true",
        default=False,
        help="Run the Catalog #272 byte-mutation smoke via the canonical helper "
             "at tools/verify_distinguishing_feature_byte_mutation.py.",
    )
    parser.add_argument(
        "--no-parser-section-manifest",
        action="store_true",
        default=False,
        help="Skip emitting the canonical parser_section_manifest.json sidecar.",
    )
    parser.add_argument(
        "--skip-protocol-verification",
        action="store_true",
        default=False,
        help="Bypass Catalog #270 umbrella verification at the compression "
             "pipeline layer (dry-run preparation only).",
    )
    parser.add_argument(
        "--hardware-substrate",
        default=HardwareSubstrateClass.AUTO.value,
        choices=[c.value for c in HardwareSubstrateClass],
        help="Hardware substrate class for the compression pipeline (default 'auto').",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit canonical JSON output (machine-readable; sorted keys).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress progress messages on stderr.",
    )
    return parser


def _render_human(manifest_dict: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("Phase 3 Layer 1 — Archive Grammar Manifest")
    lines.append("=" * 70)
    lines.append(f"  lane_id:          {manifest_dict['lane_id']}")
    lines.append(f"  substrate_id:     {manifest_dict['substrate_id']}")
    lines.append(f"  archive_path:     {manifest_dict['archive_path']}")
    lines.append(f"  archive_sha256:   {manifest_dict['archive_sha256'][:16]}...")
    lines.append(f"  archive_bytes:    {manifest_dict['archive_bytes']:,}")
    lines.append(f"  monolithic_0bin:  {manifest_dict['monolithic_single_file']}")
    if manifest_dict.get("multi_file_justification"):
        lines.append(f"  multi_file_just:  {manifest_dict['multi_file_justification']}")
    lines.append(f"  sections:         {len(manifest_dict['section_specs'])}")
    for i, spec in enumerate(manifest_dict["section_specs"][:8]):
        lines.append(
            f"    [{i}] {spec['section_name']!r:30s} "
            f"offset={spec['offset_in_archive']:>10,} "
            f"length={spec['length_in_archive']:>10,} "
            f"kind={spec['section_kind']} "
            f"ops_status={spec['operational_mechanism_status']}"
        )
    if len(manifest_dict["section_specs"]) > 8:
        lines.append(f"    ... ({len(manifest_dict['section_specs']) - 8} more)")
    lines.append(f"  byte_mutation:    {manifest_dict['byte_mutation_smoke_verdict']}")
    lines.append(f"  no_op_detector:   {manifest_dict['no_op_detector_passed']}")
    if manifest_dict.get("parser_section_manifest_path"):
        lines.append(f"  sidecar:          {manifest_dict['parser_section_manifest_path']}")
    lines.append(f"  evidence_grade:   {manifest_dict['evidence_grade']}")
    lines.append(f"  axis_tag:         {manifest_dict['axis_tag']}")
    lines.append(f"  score_claim:      {manifest_dict['score_claim']}")
    lines.append(f"  promotable:       {manifest_dict['promotable']}")
    lines.append(f"  canonical_eq:     {manifest_dict['canonical_equation_id']}")
    lines.append(f"  canonical_eq_st:  {manifest_dict['canonical_equation_status']}")
    lines.append(f"  elapsed_seconds:  {manifest_dict['elapsed_seconds']:.4f}")
    lines.append("")
    lines.append(
        "Per Catalog #341 + CLAUDE.md 'Apples-to-apples evidence discipline': "
        "this is observability-only. Score claims require paired-CUDA + Linux "
        "x86_64 CPU empirical anchor per Phase 6 paired_auth_eval."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Resolve monolithic flag with --multi-file override.
    monolithic_single_file = True
    multi_file_justification = None
    if args.multi_file:
        monolithic_single_file = False
        multi_file_justification = args.multi_file_justification
        if multi_file_justification is None:
            print(
                "ERROR: --multi-file requires --multi-file-justification per Catalog #287",
                file=sys.stderr,
            )
            return EXIT_CLI_ERROR

    if args.verify_byte_mutation_smoke and args.inflate_sh_path is None:
        print(
            "ERROR: --verify-byte-mutation-smoke requires --inflate-sh-path",
            file=sys.stderr,
        )
        return EXIT_CLI_ERROR

    # Resolve substrate trainer + recipe paths (relative resolves to repo root).
    repo_root = REPO_ROOT
    archive_abs = (
        args.archive_path
        if args.archive_path.is_absolute()
        else (repo_root / args.archive_path).resolve()
    )
    if not archive_abs.is_file():
        print(f"ERROR: archive {archive_abs} does not exist", file=sys.stderr)
        return EXIT_CLI_ERROR

    # Build Phase 2 compression pipeline result (PRE-RUN / preparation mode).
    output_dir = args.output_dir if args.output_dir is not None else archive_abs.parent
    try:
        pipeline_result = build_compression_pipeline(
            lane_id=args.lane_id,
            video_path=args.video_path,
            substrate_trainer=args.substrate_trainer,
            recipe_path=args.recipe_path,
            hardware_substrate=args.hardware_substrate,
            qat_enabled=True,
            output_dir=output_dir,
            skip_protocol_verification=args.skip_protocol_verification,
        )
    except CompressionPipelineError as exc:
        print(f"ERROR: Phase 2 compression pipeline failed: {exc}", file=sys.stderr)
        return EXIT_CLI_ERROR

    if not args.quiet:
        print(
            f"[archive_grammar_cli] Phase 2 compression pipeline prepared: "
            f"substrate_id={pipeline_result.substrate_id!r}, "
            f"hardware={pipeline_result.hardware_substrate}",
            file=sys.stderr,
        )

    # Build Phase 3 archive grammar manifest.
    try:
        manifest = build_archive_grammar_from_compression_pipeline_result(
            compression_pipeline_result=pipeline_result,
            archive_path=args.archive_path,
            monolithic_single_file=monolithic_single_file,
            multi_file_justification=multi_file_justification,
            output_dir=output_dir,
            inflate_sh_path=args.inflate_sh_path,
            verify_byte_mutation_smoke=args.verify_byte_mutation_smoke,
            emit_parser_section_manifest=not args.no_parser_section_manifest,
        )
    except ArchiveGrammarError as exc:
        msg = str(exc)
        # Section overlap is Catalog #146 violation; surface separately.
        if "section overlap" in msg.lower():
            print(f"ERROR (section overlap, Catalog #146): {exc}", file=sys.stderr)
            return EXIT_SECTION_OVERLAP
        print(f"ERROR (manifest invalid, HNeRV parity L3): {exc}", file=sys.stderr)
        return EXIT_MANIFEST_INVALID
    except ValueError as exc:
        msg = str(exc)
        if "section overlap" in msg.lower():
            print(f"ERROR (section overlap, Catalog #146): {exc}", file=sys.stderr)
            return EXIT_SECTION_OVERLAP
        print(f"ERROR (manifest invalid, HNeRV parity L3): {exc}", file=sys.stderr)
        return EXIT_MANIFEST_INVALID

    # Catalog #266 sister check: if byte-mutation smoke was requested AND the
    # verdict is FAILED_BYTES_NOT_CONSUMED, surface as a distinct exit code.
    if (
        manifest.byte_mutation_smoke_verdict
        == ByteMutationSmokeVerdict.FAILED_BYTES_NOT_CONSUMED.value
    ):
        if args.json:
            sys.stdout.write(
                json.dumps(manifest.as_dict(), sort_keys=True, indent=2) + "\n"
            )
        else:
            sys.stdout.write(_render_human(manifest.as_dict()))
        print(
            "ERROR (Catalog #266: archive bytes structurally consumed but no frame "
            "changes resulted; research-substrate trap, 8th forbidden pattern)",
            file=sys.stderr,
        )
        return EXIT_NO_OP_DETECTED

    if args.json:
        sys.stdout.write(
            json.dumps(manifest.as_dict(), sort_keys=True, indent=2) + "\n"
        )
    else:
        sys.stdout.write(_render_human(manifest.as_dict()))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
