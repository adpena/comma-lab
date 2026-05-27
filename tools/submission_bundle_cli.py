#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for the canonical submission bundle builder (Phase 4).

Sister of :mod:`tools.compression_pipeline_cli` (Phase 2) +
:mod:`tools.archive_grammar_cli` (Phase 3) + the canonical META-LIFT CLIs at
``tools/cross_substrate_master_gradient_cli.py`` +
``tools/pareto_polytope_unified_solver_cli.py`` +
``tools/uniward_invariant_enumerator_cli.py``.

Wraps :func:`tac.submission_packet.build_submission_bundle` with operator-
friendly flags + canonical exit codes per Phase 1 audit specification memo §7.

Exit codes:
  0 CLEAN — submission bundle emitted, all HNeRV parity L4 invariants satisfied
  1 ARCHIVE-GRAMMAR-INVALID — archive grammar manifest cannot be loaded / parsed
  2 INFLATE-PY-OVER-LOC-BUDGET — inflate.py LOC > budget AND no waiver
  3 DEPS-OVER-BUDGET — declared dependencies > deps budget AND no waiver
  4 PYTHONPATH-SELF-CONTAINMENT-FAILED — Catalog #295 violation
  5 CLI error (missing required arg, invalid path, etc.)

Per the 12th canonicalization × standardization × ease-of-contest-compliance
trinity: ONE CLI invocation prepares the canonical submission_dir bundle +
emits all 5 canonical components (inflate.sh + inflate.py + README.md +
report.txt + archive_manifest.json + archive.zip copy) per Catalog #146
contest-compliant runtime contract.

Per the 11th ORDER-MATTERS standing directive: this CLI is the THIRD Phase 1
spec consumer (Phase 2 CLI was first, Phase 3 CLI was second); future Phase
5-10 CLIs depend on this CLI's output shape.
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
    ArchiveGrammarManifest,
    ArchiveSectionSpec,
    ByteMutationSmokeVerdict,
    build_archive_grammar_from_compression_pipeline_result,
)
from tac.submission_packet.builder import (  # noqa: E402
    DEFAULT_INFLATE_DEPS_BUDGET,
    DEFAULT_INFLATE_PY_LOC_BUDGET,
    PythonpathSelfContainmentStatus,
    SelectInflateDeviceRouting,
    SubmissionBundleError,
    build_submission_bundle,
)
from tac.submission_packet.compression_pipeline import (  # noqa: E402
    CompressionPipelineError,
    HardwareSubstrateClass,
    build_compression_pipeline,
)


EXIT_OK = 0
EXIT_ARCHIVE_GRAMMAR_INVALID = 1
EXIT_INFLATE_PY_OVER_LOC_BUDGET = 2
EXIT_DEPS_OVER_BUDGET = 3
EXIT_PYTHONPATH_SELF_CONTAINMENT_FAILED = 4
EXIT_CLI_ERROR = 5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="submission_bundle_cli",
        description=(
            "Canonical submission bundle builder (Phase 4, Layer 2). "
            "Wraps tac.submission_packet.build_submission_bundle with operator-"
            "friendly flags. Per HNeRV parity L4: <=200 LOC inflate.py + <=2 ext "
            "deps + numpy-portable + CUDA-or-CPU agnostic + reviewable in 30s."
        ),
    )
    parser.add_argument(
        "--lane-id",
        required=True,
        help="Lane registry id (e.g. 'lane_pr111_candidate_20260601').",
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
        help="Path to the trainer-emitted archive.zip (Phase 3 archive_grammar input).",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path("upstream/videos/0.mkv"),
        help="Path to the contest video (canonical: upstream/videos/0.mkv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output submission_dir/ to bundle into "
        "(canonical: submissions/pr<N>_<lane>/).",
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
        "--inflate-py-loc-budget",
        type=int,
        default=DEFAULT_INFLATE_PY_LOC_BUDGET,
        help="HNeRV parity L4 LOC budget for inflate.py (default 200).",
    )
    parser.add_argument(
        "--inflate-deps-budget",
        type=int,
        default=DEFAULT_INFLATE_DEPS_BUDGET,
        help="HNeRV parity L4 deps budget for inflate.py (default 2).",
    )
    parser.add_argument(
        "--declared-deps",
        nargs="+",
        default=["numpy"],
        help="Declared external dependencies for inflate.py "
        "(canonical numpy-portable default: numpy; HNeRV-class: numpy torch).",
    )
    parser.add_argument(
        "--inflate-py-loc-waiver-rationale",
        type=str,
        default=None,
        help="Substantive rationale (>=4 chars, non-placeholder) when LOC > budget.",
    )
    parser.add_argument(
        "--inflate-deps-waiver-rationale",
        type=str,
        default=None,
        help="Substantive rationale (>=4 chars, non-placeholder) when deps > budget.",
    )
    parser.add_argument(
        "--vendor-pythonpath-self-containment",
        action="store_true",
        default=True,
        help="Catalog #295 PYTHONPATH self-containment (default True).",
    )
    parser.add_argument(
        "--no-vendor-pythonpath-self-containment",
        dest="vendor_pythonpath_self_containment",
        action="store_false",
        help="Disable Catalog #295 PYTHONPATH self-containment "
        "(advisory; default scaffold has no from-tac imports).",
    )
    parser.add_argument(
        "--select-inflate-device-routing",
        choices=[r.value for r in SelectInflateDeviceRouting],
        default=SelectInflateDeviceRouting.INLINE_WITH_WAIVER.value,
        help="Catalog #205 select_inflate_device routing (default inline-with-waiver).",
    )
    parser.add_argument(
        "--hardware-substrate",
        default=HardwareSubstrateClass.AUTO.value,
        choices=[c.value for c in HardwareSubstrateClass],
        help="Hardware substrate class for the compression pipeline (default 'auto').",
    )
    parser.add_argument(
        "--skip-protocol-verification",
        action="store_true",
        default=False,
        help="Bypass Catalog #270 umbrella verification at the compression "
        "pipeline layer (dry-run preparation only).",
    )
    parser.add_argument(
        "--python-invocation",
        type=str,
        default="python3",
        help="Canonical python invocation for inflate.sh (default 'python3').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview plan without writing bundle files (NOT YET IMPLEMENTED — "
        "Phase 5 sister-subagent landing).",
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


def _render_human(bundle_dict: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("Phase 4 Layer 2 — Submission Bundle Result")
    lines.append("=" * 70)
    lines.append(f"  lane_id:          {bundle_dict['lane_id']}")
    lines.append(f"  substrate_id:     {bundle_dict['substrate_id']}")
    lines.append(f"  submission_dir:   {bundle_dict['submission_dir']}")
    lines.append(f"  archive_sha256:   {bundle_dict['archive_sha256'][:16]}...")
    lines.append(f"  archive_bytes:    {bundle_dict['archive_bytes']:,}")
    lines.append(f"  inflate.sh:       {bundle_dict['inflate_sh_path']}")
    lines.append(f"  inflate.py:       {bundle_dict['inflate_py_path']}")
    lines.append(
        f"  inflate.py LOC:   {bundle_dict['inflate_py_loc']} / "
        f"{bundle_dict['inflate_py_loc_budget']}"
        f" ({'WITHIN' if bundle_dict['inflate_py_loc'] <= bundle_dict['inflate_py_loc_budget'] else 'OVER'} budget)"
    )
    if bundle_dict.get("inflate_py_loc_waiver_rationale"):
        lines.append(
            f"  inflate.py waiver:  {bundle_dict['inflate_py_loc_waiver_rationale']}"
        )
    lines.append(f"  README.md:        {bundle_dict['readme_md_path']}")
    lines.append(f"  report.txt:       {bundle_dict['report_txt_path']}")
    lines.append(f"  archive_manifest: {bundle_dict['archive_manifest_path']}")
    dep_man = bundle_dict["dependency_closure_manifest"]
    lines.append(
        f"  deps:             {dep_man['declared_dependencies']} "
        f"(budget={dep_man['dependency_budget']}; "
        f"within_budget={dep_man['within_budget']}; "
        f"numpy_portable={dep_man['numpy_portable']})"
    )
    lines.append(
        f"  device routing:   {bundle_dict['select_inflate_device_routing']}"
    )
    lines.append(
        f"  PYTHONPATH:       {bundle_dict['pythonpath_self_containment_status']}"
    )
    lines.append(f"  evidence_grade:   {bundle_dict['evidence_grade']}")
    lines.append(f"  axis_tag:         {bundle_dict['axis_tag']}")
    lines.append(f"  score_claim:      {bundle_dict['score_claim']}")
    lines.append(f"  promotable:       {bundle_dict['promotable']}")
    lines.append(f"  canonical_eq:     {bundle_dict['canonical_equation_id']}")
    lines.append(f"  canonical_eq_st:  {bundle_dict['canonical_equation_status']}")
    lines.append(f"  elapsed_seconds:  {bundle_dict['elapsed_seconds']:.4f}")
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

    if args.dry_run:
        print(
            "[submission_bundle_cli] --dry-run is NOT YET IMPLEMENTED (Phase 5 "
            "sister-subagent landing). Falling through to execute mode.",
            file=sys.stderr,
        )

    monolithic_single_file = True
    multi_file_justification: str | None = None
    if args.multi_file:
        monolithic_single_file = False
        multi_file_justification = args.multi_file_justification
        if multi_file_justification is None:
            print(
                "ERROR: --multi-file requires --multi-file-justification per Catalog #287",
                file=sys.stderr,
            )
            return EXIT_CLI_ERROR

    repo_root = REPO_ROOT
    archive_abs = (
        args.archive_path
        if args.archive_path.is_absolute()
        else (repo_root / args.archive_path).resolve()
    )
    if not archive_abs.is_file():
        print(f"ERROR: archive {archive_abs} does not exist", file=sys.stderr)
        return EXIT_CLI_ERROR

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else (repo_root / args.output_dir).resolve()
    )

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
            f"[submission_bundle_cli] Phase 2 compression pipeline prepared: "
            f"substrate_id={pipeline_result.substrate_id!r}",
            file=sys.stderr,
        )

    try:
        grammar_manifest = build_archive_grammar_from_compression_pipeline_result(
            compression_pipeline_result=pipeline_result,
            archive_path=args.archive_path,
            monolithic_single_file=monolithic_single_file,
            multi_file_justification=multi_file_justification,
            output_dir=output_dir,
            emit_parser_section_manifest=True,
        )
    except ArchiveGrammarError as exc:
        print(
            f"ERROR (archive grammar invalid, HNeRV parity L3): {exc}",
            file=sys.stderr,
        )
        return EXIT_ARCHIVE_GRAMMAR_INVALID
    except ValueError as exc:
        print(
            f"ERROR (archive grammar invalid, HNeRV parity L3): {exc}",
            file=sys.stderr,
        )
        return EXIT_ARCHIVE_GRAMMAR_INVALID

    if not args.quiet:
        print(
            f"[submission_bundle_cli] Phase 3 archive grammar manifest: "
            f"sha={grammar_manifest.archive_sha256[:12]}..., "
            f"sections={len(grammar_manifest.section_specs)}",
            file=sys.stderr,
        )

    declared_deps = tuple(sorted(set(args.declared_deps)))

    try:
        bundle_result = build_submission_bundle(
            compression_pipeline_result=pipeline_result,
            archive_grammar_manifest=grammar_manifest,
            output_dir=output_dir,
            declared_dependencies=declared_deps,
            inflate_py_loc_budget=args.inflate_py_loc_budget,
            inflate_deps_budget=args.inflate_deps_budget,
            inflate_py_loc_waiver_rationale=args.inflate_py_loc_waiver_rationale,
            inflate_deps_waiver_rationale=args.inflate_deps_waiver_rationale,
            vendor_pythonpath_self_containment=args.vendor_pythonpath_self_containment,
            select_inflate_device_routing=args.select_inflate_device_routing,
            python_invocation=args.python_invocation,
        )
    except SubmissionBundleError as exc:
        msg = str(exc)
        if "PYTHONPATH" in msg or "Catalog #295" in msg:
            print(
                f"ERROR (Catalog #295 PYTHONPATH self-containment): {exc}",
                file=sys.stderr,
            )
            return EXIT_PYTHONPATH_SELF_CONTAINMENT_FAILED
        if "HNeRV parity L4" in msg and "LOC" in msg:
            print(
                f"ERROR (HNeRV parity L4 LOC budget): {exc}", file=sys.stderr
            )
            return EXIT_INFLATE_PY_OVER_LOC_BUDGET
        print(f"ERROR (submission bundle invalid): {exc}", file=sys.stderr)
        return EXIT_CLI_ERROR
    except ValueError as exc:
        msg = str(exc)
        if "dependency_budget" in msg or "deps" in msg.lower():
            print(f"ERROR (deps over budget): {exc}", file=sys.stderr)
            return EXIT_DEPS_OVER_BUDGET
        print(f"ERROR (submission bundle invalid): {exc}", file=sys.stderr)
        return EXIT_CLI_ERROR

    if args.json:
        sys.stdout.write(
            json.dumps(bundle_result.as_dict(), sort_keys=True, indent=2) + "\n"
        )
    else:
        sys.stdout.write(_render_human(bundle_result.as_dict()))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
