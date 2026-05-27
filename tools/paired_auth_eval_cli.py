#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for Phase 7 Layer 5 paired auth-eval orchestration.

Wraps :func:`tac.submission_packet.plan_paired_auth_eval` for operator-
runbook use. Routes through the canonical Phase 7 helper that orchestrates
paired Modal CUDA + Linux x86_64 CPU auth-eval on the EXACT same archive
bytes (sha-locked invariant) per CLAUDE.md "Submission auth eval — BOTH
CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable.

Per the 10th apples-to-apples standing directive: macOS-CPU substrate
target is STRUCTURALLY non-promotable per Catalog #192 + CLAUDE.md non-
negotiable. The ``--cpu-target darwin_arm64_advisory`` flag is permitted
ONLY for dry-run planning; live dispatch refuses promotable=True on macOS.

Exit codes:
  0 -- PAIRED_PASS (both axes landed cleanly on 1:1 contest-compliant
       hardware; sha-locked invariant held; promotable=True)
  1 -- BLOCKED_PRE_DISPATCH (pre-dispatch validation failed; operator
       action required before any spawn fires)
  2 -- BLOCKED_HARVEST (dispatch fired but harvest failed; retry available)
  3 -- BLOCKED_AXIS_MISMATCH (sha-locked invariant violation per Catalog
       #127 custody discipline)
  4 -- BLOCKED_HARDWARE_NON_COMPLIANT (Catalog #192 forbidden axis)
  5 -- PAIRED_PARTIAL (one axis landed cleanly; other axis missing/failed;
       operator-routable: re-dispatch missing axis)
  6 -- CLI error (bad arguments, missing files, structural failure)

Usage (dry-run plan)::

    .venv/bin/python tools/paired_auth_eval_cli.py \\
        --from-submission-bundle path/to/submission_bundle_result.json \\
        --cost-band smoke \\
        --cuda-gpu T4 \\
        --cuda-platform modal \\
        --cpu-target linux_x86_64_modal \\
        --budget-usd 1.00 \\
        --dry-run

Usage (post-dispatch reconstruction)::

    .venv/bin/python tools/paired_auth_eval_cli.py \\
        --reconstruct-from-disk \\
        --from-submission-bundle path/to/bundle.json \\
        --cuda-auth-eval-json path/to/cuda_auth_eval.json \\
        --cpu-auth-eval-json path/to/cpu_auth_eval.json \\
        --cuda-call-id fc-... \\
        --cpu-call-id fc-... \\
        --cuda-gpu T4 \\
        --cpu-target linux_x86_64_modal \\
        --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Resolve repo paths for canonical imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from tac.submission_packet import (  # noqa: E402
    PairedAuthEvalError,
    PairedAuthEvalVerdict,
    PairedAuthEvalVerdictKind,
    SubmissionBundleResult,
    plan_paired_auth_eval,
    reconstruct_verdict_from_disk,
    submission_bundle_result_from_dict,
)

# Canonical CLI exit code constants.
EXIT_PAIRED_PASS = 0
EXIT_BLOCKED_PRE_DISPATCH = 1
EXIT_BLOCKED_HARVEST = 2
EXIT_BLOCKED_AXIS_MISMATCH = 3
EXIT_BLOCKED_HARDWARE_NON_COMPLIANT = 4
EXIT_PAIRED_PARTIAL = 5
EXIT_CLI_ERROR = 6


def _verdict_to_exit_code(verdict: PairedAuthEvalVerdict) -> int:
    """Map :class:`PairedAuthEvalVerdict` to canonical CLI exit code."""
    kind_to_exit = {
        PairedAuthEvalVerdictKind.PAIRED_PASS.value: EXIT_PAIRED_PASS,
        PairedAuthEvalVerdictKind.BLOCKED_PRE_DISPATCH.value: EXIT_BLOCKED_PRE_DISPATCH,
        PairedAuthEvalVerdictKind.BLOCKED_HARVEST.value: EXIT_BLOCKED_HARVEST,
        PairedAuthEvalVerdictKind.BLOCKED_AXIS_MISMATCH.value: EXIT_BLOCKED_AXIS_MISMATCH,
        PairedAuthEvalVerdictKind.BLOCKED_HARDWARE_NON_COMPLIANT.value: (
            EXIT_BLOCKED_HARDWARE_NON_COMPLIANT
        ),
        PairedAuthEvalVerdictKind.PAIRED_PARTIAL_CUDA_ONLY.value: EXIT_PAIRED_PARTIAL,
        PairedAuthEvalVerdictKind.PAIRED_PARTIAL_CPU_ONLY.value: EXIT_PAIRED_PARTIAL,
    }
    return kind_to_exit.get(verdict.verdict, EXIT_CLI_ERROR)


def _reconstruct_submission_bundle_from_dict(data: dict[str, Any]) -> SubmissionBundleResult:
    """Reconstruct a :class:`SubmissionBundleResult` from a JSON-loaded dict.

    Handles the nested ``dependency_closure_manifest`` reconstruction +
    tuple-vs-list normalization for frozen-dataclass round-trip.
    """
    return submission_bundle_result_from_dict(data)


def _load_submission_bundle(path_or_stdin: str) -> SubmissionBundleResult:
    """Load a SubmissionBundleResult from a JSON file or '-' for stdin."""
    if path_or_stdin == "-":
        raw = sys.stdin.read()
    else:
        p = Path(path_or_stdin)
        if not p.exists():
            raise FileNotFoundError(
                f"submission bundle path does not exist: {path_or_stdin}"
            )
        raw = p.read_text()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"submission bundle path is not parseable JSON: {exc}"
        ) from exc
    return _reconstruct_submission_bundle_from_dict(data)


def _render_human_readable_verdict(verdict: PairedAuthEvalVerdict) -> str:
    """Render a human-readable verdict summary on stderr."""
    lines: list[str] = []
    lines.append(f"[paired-auth-eval-cli] verdict={verdict.verdict}")
    lines.append(f"[paired-auth-eval-cli] lane_id={verdict.lane_id}")
    lines.append(f"[paired-auth-eval-cli] substrate_id={verdict.substrate_id}")
    lines.append(
        f"[paired-auth-eval-cli] archive_sha256={verdict.archive_sha256_paired[:12] or '(none)'} "
        f"bytes={verdict.archive_bytes}"
    )
    lines.append(
        f"[paired-auth-eval-cli] cuda: hardware={verdict.cuda_hardware_substrate} "
        f"score={verdict.cuda_score} call_id={verdict.cuda_call_id or '(none)'}"
    )
    lines.append(
        f"[paired-auth-eval-cli] cpu: hardware={verdict.cpu_hardware_substrate} "
        f"score={verdict.cpu_score} call_id={verdict.cpu_call_id or '(none)'}"
    )
    if verdict.cuda_cpu_gap is not None:
        lines.append(
            f"[paired-auth-eval-cli] cuda_cpu_gap={verdict.cuda_cpu_gap:+.6f}"
        )
    lines.append(
        f"[paired-auth-eval-cli] cost: estimated_total=${verdict.total_cost_usd:.2f} "
        f"budget=${verdict.budget_usd:.2f}"
    )
    lines.append(
        f"[paired-auth-eval-cli] score_claim={verdict.score_claim} "
        f"promotable={verdict.promotable} forbidden_macos={verdict.forbidden_macos_axis_detected}"
    )
    lines.append(
        f"[paired-auth-eval-cli] axis_tag={verdict.axis_tag} evidence_grade={verdict.evidence_grade}"
    )
    lines.append(
        f"[paired-auth-eval-cli] rationale: {verdict.verdict_rationale}"
    )
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 7 Layer 5 paired CUDA+CPU auth-eval orchestrator CLI. "
            "Routes through tac.submission_packet.plan_paired_auth_eval."
        ),
    )
    parser.add_argument(
        "--from-submission-bundle",
        required=True,
        help=(
            "Path to a SubmissionBundleResult JSON (Phase 4 output) "
            "or '-' for stdin."
        ),
    )
    parser.add_argument(
        "--cost-band",
        choices=("smoke", "full"),
        default="smoke",
        help="Per Catalog #270 cost-band envelope. Default: smoke ($1.00).",
    )
    parser.add_argument(
        "--cuda-gpu",
        choices=("T4", "L4", "A10G", "L40S", "A100", "4090", "H100"),
        default="T4",
        help="CUDA GPU class per Catalog #215. Default: T4 (smoke-safe).",
    )
    parser.add_argument(
        "--cuda-platform",
        choices=("modal", "vastai", "lightning"),
        default="modal",
        help="CUDA dispatch platform (always Linux x86_64 host). Default: modal.",
    )
    parser.add_argument(
        "--cpu-target",
        choices=(
            "linux_x86_64_modal",
            "linux_x86_64_vastai",
            "linux_x86_64_lightning",
            "linux_x86_64_gha",
            "darwin_arm64_advisory",
        ),
        default="linux_x86_64_modal",
        help=(
            "CPU axis target per Catalog #192. "
            "darwin_arm64_advisory permitted ONLY for dry-run; "
            "structurally non-promotable per CLAUDE.md."
        ),
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help=(
            "$USD envelope ceiling. Default: canonical cost-band default "
            "($1.00 for smoke, $5.00 for full)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Plan-only mode (default): no Modal spawn fires; cost estimate only. "
            "Required when --execute is NOT specified."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Execute mode: fires actual Modal dispatch. REQUIRES "
            "--operator-approved per CLAUDE.md 'Executing actions with care' "
            "non-negotiable. Phase 7 scope: routes to canonical "
            "tools/dispatch_modal_paired_auth_eval.py invocation; this CLI "
            "is the plan + reconstruction surface."
        ),
    )
    parser.add_argument(
        "--operator-approved",
        type=str,
        default=None,
        help=(
            "Operator approval handle (format: <handle>:<UTC>); REQUIRED "
            "when --execute is set."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for canonical verdict JSON + per-axis reports.",
    )
    parser.add_argument(
        "--reconstruct-from-disk",
        action="store_true",
        help=(
            "Post-dispatch reconstruction mode: read per-axis JSONs + emit "
            "canonical PairedAuthEvalVerdict. Requires --cuda-auth-eval-json + "
            "--cpu-auth-eval-json + --cuda-call-id + --cpu-call-id."
        ),
    )
    parser.add_argument(
        "--cuda-auth-eval-json",
        type=Path,
        default=None,
        help="Path to canonical CUDA contest_auth_eval.json (reconstruct mode).",
    )
    parser.add_argument(
        "--cpu-auth-eval-json",
        type=Path,
        default=None,
        help="Path to canonical CPU contest_auth_eval.json (reconstruct mode).",
    )
    parser.add_argument(
        "--cuda-call-id",
        type=str,
        default="",
        help="Modal call_id for CUDA axis (reconstruct mode).",
    )
    parser.add_argument(
        "--cpu-call-id",
        type=str,
        default="",
        help="Modal call_id for CPU axis (reconstruct mode).",
    )
    parser.add_argument(
        "--cuda-cost-usd",
        type=float,
        default=0.0,
        help="Actual CUDA-axis $USD cost (reconstruct mode).",
    )
    parser.add_argument(
        "--cpu-cost-usd",
        type=float,
        default=0.0,
        help="Actual CPU-axis $USD cost (reconstruct mode).",
    )
    parser.add_argument(
        "--cuda-elapsed-seconds",
        type=float,
        default=0.0,
        help="Actual CUDA-axis wall-clock seconds (reconstruct mode).",
    )
    parser.add_argument(
        "--cpu-elapsed-seconds",
        type=float,
        default=0.0,
        help="Actual CPU-axis wall-clock seconds (reconstruct mode).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit verdict as JSON on stdout (machine-readable mode).",
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    """Execute the CLI flow with parsed args; return exit code."""
    try:
        bundle = _load_submission_bundle(args.from_submission_bundle)
    except FileNotFoundError as exc:
        print(f"[paired-auth-eval-cli] FATAL: {exc}", file=sys.stderr)
        return EXIT_CLI_ERROR
    except (ValueError, KeyError) as exc:
        print(
            f"[paired-auth-eval-cli] FATAL: failed to load submission bundle: {exc}",
            file=sys.stderr,
        )
        return EXIT_CLI_ERROR

    # Reconstruction mode
    if args.reconstruct_from_disk:
        if (
            args.cuda_auth_eval_json is None
            or args.cpu_auth_eval_json is None
        ):
            print(
                "[paired-auth-eval-cli] FATAL: --reconstruct-from-disk requires "
                "--cuda-auth-eval-json AND --cpu-auth-eval-json",
                file=sys.stderr,
            )
            return EXIT_CLI_ERROR
        try:
            verdict = reconstruct_verdict_from_disk(
                submission_bundle_result=bundle,
                cuda_auth_eval_json_path=args.cuda_auth_eval_json,
                cpu_auth_eval_json_path=args.cpu_auth_eval_json,
                cuda_call_id=args.cuda_call_id,
                cpu_call_id=args.cpu_call_id,
                cost_band=args.cost_band,
                cuda_gpu=args.cuda_gpu,
                cuda_platform=args.cuda_platform,
                cpu_target=args.cpu_target,
                budget_usd=(
                    args.budget_usd
                    if args.budget_usd is not None
                    else 1.0 if args.cost_band == "smoke" else 5.0
                ),
                cuda_cost_usd=args.cuda_cost_usd,
                cpu_cost_usd=args.cpu_cost_usd,
                cuda_elapsed_seconds=args.cuda_elapsed_seconds,
                cpu_elapsed_seconds=args.cpu_elapsed_seconds,
            )
        except PairedAuthEvalError as exc:
            print(f"[paired-auth-eval-cli] FATAL: {exc}", file=sys.stderr)
            return EXIT_CLI_ERROR
    else:
        # Plan / execute mode
        # Default dry_run=True unless --execute set
        dry_run = args.dry_run or not args.execute
        if args.execute and not args.operator_approved:
            print(
                "[paired-auth-eval-cli] FATAL: --execute requires "
                "--operator-approved per CLAUDE.md 'Executing actions with "
                "care' non-negotiable",
                file=sys.stderr,
            )
            return EXIT_CLI_ERROR
        try:
            verdict = plan_paired_auth_eval(
                submission_bundle_result=bundle,
                cost_band=args.cost_band,
                cuda_gpu=args.cuda_gpu,
                cuda_platform=args.cuda_platform,
                cpu_target=args.cpu_target,
                budget_usd=args.budget_usd,
                dry_run=dry_run,
                operator_approved_handle=args.operator_approved,
                output_dir=args.output_dir,
            )
        except PairedAuthEvalError as exc:
            print(f"[paired-auth-eval-cli] FATAL: {exc}", file=sys.stderr)
            return EXIT_CLI_ERROR
        except ValueError as exc:
            print(f"[paired-auth-eval-cli] FATAL: {exc}", file=sys.stderr)
            return EXIT_CLI_ERROR

    # Emit verdict
    if args.json:
        print(json.dumps(verdict.as_dict(), sort_keys=True, indent=2))
    else:
        print(_render_human_readable_verdict(verdict), file=sys.stderr)
    return _verdict_to_exit_code(verdict)


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
