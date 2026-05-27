#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Canonical CLI for Phase 2 Layer 0 (``tac.submission_packet.compression_pipeline``).

Per Phase 1 audit specification memo at
``.omx/research/canonical_submission_pipeline_specification_memo_20260526.md``
Layer 0 + sister META-LIFT CLIs (``tools/cross_substrate_master_gradient_analyzer_cli.py``,
``tools/uniward_invariant_enumerator_cli.py``, ``tools/pareto_polytope_unified_solver_cli.py``).

Per the 12th canonicalization × standardization × ease-of-contest-compliance
trinity: this CLI is the operator-facing surface for the canonical encoder
pipeline preparation phase. It does NOT invoke paid GPU dispatch (Phase 2
scope); Phase 6 ``tac.submission_packet.paired_auth_eval`` is where the
actual dispatch fires.

Exit codes (per Phase 1 spec memo §7):
    0  CLEAN — compression pipeline result emitted; canonical Provenance carried
    1  TRAINER_FAILURE — trainer + recipe pair invalid OR trainer absent
    2  QAT_FAILURE — Catalog #270 Tier-3 substrate-correctness blocker
    3  ARCHIVE_PACK_FAILURE — reserved for Phase 3 archive_grammar (out of Phase 2 scope)
    4  INFLATE_RUNTIME_FAILURE — reserved for Phase 4 builder (out of Phase 2 scope)
    5  CLI_ERROR — argparse / IO / unexpected exception

Per CLAUDE.md "Executing actions with care" + "Beauty, simplicity, and
developer experience": canonical helper invocation, frozen-dataclass
return, JSON-or-human-readable output via ``--json`` flag.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


# Per Catalog #295 PYTHONPATH self-containment: thin CLI that delegates to
# the canonical helper at tac.submission_packet.
def _resolve_repo_root_for_imports() -> Path:
    """Resolve the repo root so 'from tac.submission_packet import ...' works."""
    here = Path(__file__).resolve()
    # tools/<this>.py → repo_root is parent.parent
    return here.parent.parent


# Ensure src/ is on sys.path BEFORE importing tac.* (canonical entry pattern).
_REPO_ROOT = _resolve_repo_root_for_imports()
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# Canonical exit codes (per Phase 1 spec memo §7).
EXIT_CLEAN = 0
EXIT_TRAINER_FAILURE = 1
EXIT_QAT_FAILURE = 2
EXIT_ARCHIVE_PACK_FAILURE = 3
EXIT_INFLATE_RUNTIME_FAILURE = 4
EXIT_CLI_ERROR = 5


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compression_pipeline_cli",
        description=(
            "Canonical Phase 2 Layer 0 encoder pipeline orchestrator CLI. "
            "Routes a (trainer, recipe) pair through Catalog #270 umbrella "
            "protocol verification + canonical hardware classification + "
            "canonical Provenance routing. Emits a typed "
            "CompressionPipelineResult that downstream Phase 3-10 layers "
            "consume."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lane-id",
        required=True,
        help="Lane registry id per CLAUDE.md 'Lane maturity registry' non-negotiable.",
    )
    parser.add_argument(
        "--video-path",
        required=True,
        help="Path to the contest video (canonical: upstream/videos/0.mkv).",
    )
    parser.add_argument(
        "--substrate-trainer",
        required=True,
        help="Path to experiments/train_substrate_<id>.py.",
    )
    parser.add_argument(
        "--recipe-path",
        required=True,
        help="Path to .omx/operator_authorize_recipes/substrate_<id>_*.yaml.",
    )
    parser.add_argument(
        "--hardware-substrate",
        default="auto",
        choices=["auto", "local-mps", "local-cpu", "modal", "vastai", "lightning"],
        help="Hardware substrate routing class per Phase 1 spec memo.",
    )
    parser.add_argument(
        "--explicit-hardware-substrate",
        default=None,
        help="Optional canonical hardware token override per Catalog #190 (e.g. linux_x86_64_modal_t4).",
    )
    parser.add_argument(
        "--qat-enabled",
        action="store_true",
        default=True,
        help="Enable post-train QAT (default True per CLAUDE.md QAT pipeline).",
    )
    parser.add_argument(
        "--no-qat",
        action="store_false",
        dest="qat_enabled",
        help="Disable post-train QAT.",
    )
    parser.add_argument(
        "--mlx-first",
        action="store_true",
        default=None,
        help="Explicit MLX-first override per 8th standing directive (None=auto).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for emitted artifacts.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Override repo root (defaults to module-resolved REPO_ROOT).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Plan emission without execution (does NOT invoke paid dispatch). "
            "Per CLAUDE.md 'Executing actions with care' canonical pattern."
        ),
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help=(
            "5ep MLX-local smoke verification before full pipeline (Phase 2 "
            "scope marker; smoke execution lands at Phase 6/Phase 10)."
        ),
    )
    parser.add_argument(
        "--skip-protocol-verification",
        action="store_true",
        help=(
            "Bypass Catalog #270 umbrella verification (operator-routable for "
            "dry-run preparation only; default False)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output (default: human-readable summary).",
    )
    return parser


def _render_human_summary(result: dict) -> str:
    lines = [
        "=" * 72,
        "Phase 2 Layer 0 compression pipeline preparation",
        "=" * 72,
        f"  lane_id             : {result['lane_id']}",
        f"  substrate_id        : {result['substrate_id']}",
        f"  video_path          : {result['video_path']}",
        f"  hardware_substrate  : {result['hardware_substrate']}",
        f"  hardware_class      : {result['hardware_substrate_class']}",
        f"  mlx_first_encode    : {result['mlx_first_encode']}",
        f"  qat_enabled         : {result['qat_enabled']}",
        f"  protocol_overall_ok : {result['dispatch_optimization_protocol_overall_pass']}",
        f"  axis_tag            : {result['axis_tag']}",
        f"  score_claim         : {result['score_claim']}",
        f"  promotable          : {result['promotable']}",
        f"  evidence_grade      : {result['evidence_grade']}",
        f"  canonical_equation  : {result['canonical_equation_id']}",
        f"  equation_status     : {result['canonical_equation_status']}",
        f"  measurement_utc     : {result['measurement_utc']}",
    ]
    blockers = result.get("dispatch_optimization_protocol_blockers") or []
    if blockers:
        lines.append("  blockers:")
        for b in blockers[:10]:
            lines.append(f"    - {b}")
        if len(blockers) > 10:
            lines.append(f"    ... ({len(blockers) - 10} more)")
    if result.get("per_axis_predicted_band"):
        pb = result["per_axis_predicted_band"]
        lines.append(
            f"  predicted_band      : seg={pb['predicted_seg_distortion_band']} "
            f"pose={pb['predicted_pose_distortion_band']} "
            f"bytes={pb['predicted_archive_bytes_band']} "
            f"validation={pb['predicted_band_validation_status']}"
        )
    lines.append("=" * 72)
    lines.append(
        "Per Phase 1 spec memo: this is Phase 2 PREPARATION ONLY. Phase 6 "
        "(paired_auth_eval) fires the actual dispatch. Per CLAUDE.md "
        "'Submission auth eval BOTH CPU AND CUDA': promotion requires paired "
        "Linux x86_64 CPU + CUDA empirical anchors."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse exits via SystemExit; map to CLI_ERROR per Phase 1 spec.
        return int(exc.code) if exc.code is not None else EXIT_CLI_ERROR

    try:
        from tac.submission_packet import (
            CompressionPipelineError,
            build_compression_pipeline,
        )
    except ImportError as exc:
        print(
            f"[compression_pipeline_cli] FATAL: cannot import canonical helper: {exc}",
            file=sys.stderr,
        )
        return EXIT_CLI_ERROR

    try:
        substrate_trainer = Path(args.substrate_trainer)
        recipe_path = Path(args.recipe_path)
        output_dir = Path(args.output_dir)
        video_path = Path(args.video_path)
        repo_root = Path(args.repo_root) if args.repo_root else None

        result = build_compression_pipeline(
            lane_id=args.lane_id,
            video_path=video_path,
            substrate_trainer=substrate_trainer,
            recipe_path=recipe_path,
            hardware_substrate=args.hardware_substrate,
            qat_enabled=bool(args.qat_enabled),
            output_dir=output_dir,
            explicit_hardware_substrate=args.explicit_hardware_substrate,
            mlx_first=args.mlx_first,
            repo_root=repo_root,
            skip_protocol_verification=bool(args.skip_protocol_verification),
        )
    except CompressionPipelineError as exc:
        # Map common error patterns to canonical exit codes.
        msg = str(exc)
        if "trainer" in msg.lower() or "Catalog #240" in msg:
            print(
                f"[compression_pipeline_cli] TRAINER_FAILURE: {msg}", file=sys.stderr
            )
            return EXIT_TRAINER_FAILURE
        if "Catalog #270" in msg or "Tier-3" in msg or "Tier 3" in msg:
            print(f"[compression_pipeline_cli] QAT_FAILURE: {msg}", file=sys.stderr)
            return EXIT_QAT_FAILURE
        print(f"[compression_pipeline_cli] CLI_ERROR: {msg}", file=sys.stderr)
        return EXIT_CLI_ERROR
    except Exception as exc:  # pragma: no cover (defensive)
        print(
            f"[compression_pipeline_cli] CLI_ERROR (unexpected): {exc}\n"
            f"{traceback.format_exc()}",
            file=sys.stderr,
        )
        return EXIT_CLI_ERROR

    d = result.as_dict()
    if args.json:
        print(json.dumps(d, sort_keys=True))
    else:
        print(_render_human_summary(d))
    return EXIT_CLEAN


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
