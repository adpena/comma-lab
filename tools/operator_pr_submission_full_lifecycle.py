#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Canonical single-command end-to-end PR-submission lifecycle CLI (Phase 9).

Layer 7 (the LAST layer) of the canonical-submission-pipeline 7-layer
architecture per the Phase 1 audit specification memo at
``.omx/research/canonical_submission_pipeline_specification_memo_20260526.md``
§3 Layer 7. Glue-layer orchestrator over the 6 prior layers + the Phase 8
STRICT gate (Catalog #370).

Per operator NON-NEGOTIABLE 2026-05-26 (9th standing directive + amendment):
*"Remember everything we had to do to clean up and properly bundle our
submission, let's make that canonical and automated moving forward"* +
*"Remember contest compliance and bundling full compression script and all
and everything"*. This single CLI collapses the prior ~3h x 4-subagent +
~5K-LOC + 6-phase manual PR-submission anti-pattern (2026-05-19 PR101 anchor)
to one command.

Canonical orchestration (each layer routes through its canonical helper):

  Layer 0  tac.submission_packet.build_compression_pipeline
           -> CompressionPipelineResult
  Layer 1  tac.submission_packet.build_archive_grammar_from_compression_pipeline_result
           -> ArchiveGrammarManifest
  Layer 2  tac.submission_packet.build_submission_bundle
           -> SubmissionBundleResult  (+ submission_bundle_result.json sidecar)
  Layer 3  tac.submission_packet.lint_submission_bundle
           -> LintVerdict             (+ lint_verdict.json sidecar)
  Layer 4  tac.submission_packet.enforce_contest_compliance
           -> ComplianceVerdict       (+ compliance_verdict.json sidecar)
  Layer 5  tac.submission_packet.plan_paired_auth_eval
           -> PairedAuthEvalVerdict   (+ paired_auth_eval_verdict.json sidecar)
  Layer 6  tac.preflight.check_no_pr_submission_without_canonical_compliance_verdict
           (Catalog #370 4-verdict-chain verification)

CLI signature::

    .venv/bin/python tools/operator_pr_submission_full_lifecycle.py \\
        --lane-id <lane> \\
        --substrate-trainer experiments/train_substrate_<id>.py \\
        --recipe-path .omx/operator_authorize_recipes/substrate_<id>_<platform>_dispatch.yaml \\
        --archive-path experiments/results/<lane>/archive.zip \\
        --video-path upstream/videos/0.mkv \\
        --target-repo commaai/comma_video_compression_challenge \\
        --predecessors @SajayR:56:HNeRV_substrate @AaronLeslie138:95:fec_curriculum \\
        --output-dir submissions/pr<N>_<lane>/ \\
        [--dry-run | --execute] [--json]

Exit-code taxonomy (per 9th-directive amendment Layer 7 binding contract):

  0  PACKET-CLEAN     reserved for a future no-human-action mode; this CLI's
                      clean terminal path remains exit 4 because `gh` is gated
  1  LINT-VIOLATIONS  Layer 3 ERROR-severity findings (forbidden token /
                      first-person-plural / emdash / inflate.py over-budget)
  2  COMPLIANCE-ERRORS Layer 4 structural / D3+D5 blockers
  3  MISSING-PAIRED-AXIS Layer 5 verdict not PAIRED_PASS (CPU/CUDA missing/failed)
  4  OPERATOR-GATED   all 7 layers PASS, packet clean, and operator-gated
                      action remains
                      (`gh pr create` + `gh release create` NEVER fired by CLI)
  5  CLI / usage error (missing arg, bad path, layer-0 / layer-1 failure)

Per CLAUDE.md "Executing actions with care" + "Public Disclosure Hygiene":
this CLI NEVER fires `gh pr create` / `gh release create`. At exit 4 it
EMITS the operator-routable commands; the operator runs them.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-
COMPLIANT HARDWARE" + the 8th MLX-first standing directive: ``--dry-run``
(default) runs Layers 0-4 + 6 at $0 and Layer 5 prescreen-only (MLX-local +
macOS-CPU advisory plan, NO paid dispatch). ``--execute`` runs the full
pipeline; the Layer 5 paired-CUDA GATED escalation requires the paired-env
discipline per Catalog #199 (``OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE``
+ ``OPERATOR_AUTHORIZE_SESSION_BUDGET_USD``) AND still stops at exit 4 before
any `gh` command.

Per the 12th canonicalization x standardization x ease-of-contest-compliance
trinity: this CLI IS the single-command default-path. Sister of
``tools/submission_bundle_cli.py`` (Phase 4) + ``tools/submission_linter_cli.py``
(Phase 5) + ``tools/submission_compliance_cli.py`` (Phase 6) +
``tools/paired_auth_eval_cli.py`` (Phase 7) which remain callable for
per-layer sister consumers; this CLI composes them end-to-end.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tac.submission_packet.archive_grammar import (  # noqa: E402
    ArchiveGrammarError,
    build_archive_grammar_from_compression_pipeline_result,
)
from tac.submission_packet.builder import (  # noqa: E402
    SubmissionBundleError,
    build_submission_bundle,
)
from tac.submission_packet.compliance import (  # noqa: E402
    SubmissionComplianceError,
    enforce_contest_compliance,
)
from tac.submission_packet.compression_pipeline import (  # noqa: E402
    CompressionPipelineError,
    HardwareSubstrateClass,
    build_compression_pipeline,
)
from tac.submission_packet.linter import (  # noqa: E402
    SubmissionLinterError,
    lint_submission_bundle,
)
from tac.submission_packet.paired_auth_eval import (  # noqa: E402
    PairedAuthEvalError,
    PairedAuthEvalVerdictKind,
    plan_paired_auth_eval,
)

# ---------------------------------------------------------------------------
# Exit-code taxonomy (binding per 9th-directive amendment Layer 7)
# ---------------------------------------------------------------------------
EXIT_PACKET_CLEAN = 0
EXIT_LINT_VIOLATIONS = 1
EXIT_COMPLIANCE_ERRORS = 2
EXIT_MISSING_PAIRED_AXIS = 3
EXIT_OPERATOR_GATED = 4
EXIT_CLI_ERROR = 5

# Canonical verdict sidecar filenames the Phase 8 gate (Catalog #370) searches
# for inside the submission_dir. Emitting to these exact names closes the
# 4-verdict chain that Layer 6 verification consumes.
_SIDECAR_BUNDLE = "submission_bundle_result.json"
_SIDECAR_LINT = "lint_verdict.json"
_SIDECAR_COMPLIANCE = "compliance_verdict.json"
_SIDECAR_PAIRED = "paired_auth_eval_verdict.json"

# Catalog #199 paired-env discipline tokens (the --execute paired-CUDA GATED
# escalation requires BOTH to be set).
_ENV_CONFIRMED = "OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE"
_ENV_BUDGET = "OPERATOR_AUTHORIZE_SESSION_BUDGET_USD"

# Forbidden public-PR surface tokens (CRITICAL attribution discipline per
# ~/.claude memory user_pr_attribution + feedback_forbidden_claude_attribution).
# The Layer 3 linter is the canonical enforcer; this CLI mirrors the contract
# for the predecessor-derived attribution markdown it generates so it never
# emits a forbidden token into a PR-facing surface.
_FORBIDDEN_PR_TOKENS: tuple[str, ...] = (
    "Claude",
    "Anthropic",
    "Co-Authored",
    "claude.com",
    "anthropic.com",
)
_FIRST_PERSON_PLURAL = re.compile(r"\b(we|our|us|we're|we've|we'll|we'd)\b", re.IGNORECASE)
_EMDASH = "—"

# Predecessor spec: @handle:PRnumber:slug  (e.g. @SajayR:56:HNeRV_substrate)
_PREDECESSOR_PATTERN = re.compile(r"^@([A-Za-z0-9_-]+):(\d+):(.+)$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve(p: Path, repo_root: Path) -> Path:
    return p if p.is_absolute() else (repo_root / p).resolve()


def _as_dict(obj: Any) -> dict[str, Any]:
    """Serialize a frozen result/verdict dataclass to a JSON-safe dict.

    Prefers the canonical ``as_dict()`` method when present; falls back to
    ``dataclasses.asdict`` for nested-dataclass safety.
    """
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    return dataclasses.asdict(obj)


def _write_sidecar(submission_dir: Path, filename: str, payload: dict[str, Any]) -> Path:
    """Write a canonical verdict sidecar JSON (sorted keys, byte-stable)."""
    submission_dir.mkdir(parents=True, exist_ok=True)
    target = submission_dir / filename
    target.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return target


def _parse_predecessors(raw: list[str] | None) -> tuple[list[dict[str, str]], list[str]]:
    """Parse ``@handle:PR:slug`` predecessor specs into structured rows.

    Returns ``(rows, errors)``. Malformed specs accumulate in ``errors``.
    """
    rows: list[dict[str, str]] = []
    errors: list[str] = []
    for spec in raw or []:
        m = _PREDECESSOR_PATTERN.match(spec.strip())
        if m is None:
            errors.append(
                f"malformed --predecessors spec {spec!r}; expected "
                f"'@handle:PRnumber:slug' (e.g. '@SajayR:56:HNeRV_substrate')"
            )
            continue
        rows.append({"handle": m.group(1), "pr_number": m.group(2), "slug": m.group(3)})
    return rows, errors


def _build_attribution_chain_markdown(
    predecessors: list[dict[str, str]], target_repo: str
) -> str:
    """Build sole-author attribution-chain markdown for the PR body placeholder.

    CRITICAL per user_pr_attribution memory: sole-author Alejandro Pena
    <adpena@gmail.com>. The chain cites predecessor PR authors by @-mention +
    PR# reference (PR 95/101/102/103 medal-class precedent) WITHOUT any
    first-person-plural pronoun, emdash, or Claude/Anthropic token (the Layer 3
    linter independently enforces this; this generator never emits one).
    """
    lines: list[str] = []
    lines.append("## Attribution")
    lines.append("")
    if predecessors:
        lines.append(
            "This submission builds on prior contest work. Predecessor authors:"
        )
        lines.append("")
        for row in predecessors:
            lines.append(
                f"- @{row['handle']} (PR #{row['pr_number']}): {row['slug']}"
            )
    else:
        lines.append(
            "Standalone submission; no predecessor attribution chain supplied."
        )
    return "\n".join(lines) + "\n"


def _scan_forbidden_pr_tokens(text: str) -> list[str]:
    """Return forbidden-token findings in PR-facing-bound generated text.

    Mirrors the Layer 3 linter contract so the CLI's own generated
    attribution markdown can never leak a forbidden token before the
    canonical linter runs.
    """
    findings: list[str] = []
    for token in _FORBIDDEN_PR_TOKENS:
        if token in text:
            findings.append(f"forbidden PR-surface token {token!r}")
    if _FIRST_PERSON_PLURAL.search(text):
        findings.append("first-person-plural pronoun in PR-facing surface")
    if _EMDASH in text:
        findings.append("emdash (U+2014) in PR-facing surface")
    return findings


def _execute_paired_env_active() -> tuple[bool, str | None]:
    """Return (active, rationale_or_error) for the Catalog #199 paired-env gate.

    The --execute paired-CUDA GATED escalation requires BOTH
    OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE AND a numeric
    OPERATOR_AUTHORIZE_SESSION_BUDGET_USD per the paired-env discipline.
    Bare CONFIRMED without a numeric BUDGET is REJECTED (returns error).
    """
    confirmed = os.environ.get(_ENV_CONFIRMED, "").strip()
    budget = os.environ.get(_ENV_BUDGET, "").strip()
    if not confirmed:
        return (False, None)
    if not budget:
        return (
            False,
            f"{_ENV_CONFIRMED} set without paired {_ENV_BUDGET} "
            f"(Catalog #199 paired-env discipline; bare intent rejected)",
        )
    try:
        budget_val = float(budget)
    except ValueError:
        return (
            False,
            f"{_ENV_BUDGET}={budget!r} is not a numeric USD value "
            f"(Catalog #199 paired-env discipline)",
        )
    if budget_val <= 0:
        return (False, f"{_ENV_BUDGET}={budget_val} must be > 0 USD")
    return (True, f"paired-env active; session budget ${budget_val:.2f} USD")


# ---------------------------------------------------------------------------
# Layer orchestration
# ---------------------------------------------------------------------------
def _run_layer_6_gate(submission_dir: Path, repo_root: Path) -> list[str]:
    """Run the Phase 8 Catalog #370 STRICT gate over the submission_dir.

    Returns the gate's violation list scoped to THIS submission_dir
    (warn-only; the CLI maps the result into its own exit-code taxonomy).
    Imported lazily so a preflight import failure surfaces as a CLI error
    rather than a module-load crash (fail-closed per Catalog #279 pattern).
    """
    from tac.preflight import (
        check_no_pr_submission_without_canonical_compliance_verdict,
    )

    all_violations = check_no_pr_submission_without_canonical_compliance_verdict(
        repo_root=repo_root, strict=False, verbose=False
    )
    # Scope to this submission_dir (the gate scans all submissions/*).
    try:
        sub_rel = submission_dir.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        sub_rel = submission_dir.name
    scoped = [v for v in all_violations if sub_rel in v or submission_dir.name in v]
    return scoped


def run_full_lifecycle(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    """Run the canonical end-to-end PR-submission lifecycle.

    Returns ``(exit_code, report_dict)``. The report_dict is the canonical
    machine-readable lifecycle verdict for ``--json`` output + cathedral
    consumer / autopilot ranker consumption.
    """
    repo_root = REPO_ROOT
    report: dict[str, Any] = {
        "schema_version": "operator_pr_submission_full_lifecycle_v1",
        "lane_id": args.lane_id,
        "target_repo": args.target_repo,
        "mode": "execute" if args.execute else "dry-run",
        "layers": {},
        "gh_commands_emitted": False,
        "gh_commands_fired": False,  # ALWAYS False per "Executing actions with care"
    }

    # ---- Predecessor parse + attribution chain self-lint ----
    predecessors, pred_errors = _parse_predecessors(args.predecessors)
    if pred_errors:
        report["layers"]["predecessor_parse"] = {"ok": False, "errors": pred_errors}
        return (EXIT_CLI_ERROR, report)
    attribution_md = _build_attribution_chain_markdown(predecessors, args.target_repo)
    forbidden = _scan_forbidden_pr_tokens(attribution_md)
    if forbidden:
        report["layers"]["attribution_self_lint"] = {
            "ok": False,
            "findings": forbidden,
        }
        return (EXIT_CLI_ERROR, report)
    report["layers"]["attribution_self_lint"] = {
        "ok": True,
        "predecessors": predecessors,
    }

    archive_abs = _resolve(args.archive_path, repo_root)
    if not archive_abs.is_file():
        report["layers"]["preflight"] = {
            "ok": False,
            "error": f"archive {archive_abs} does not exist",
        }
        return (EXIT_CLI_ERROR, report)

    output_dir = _resolve(args.output_dir, repo_root)

    # ---- Layer 0: compression pipeline ----
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
    except (CompressionPipelineError, ValueError) as exc:
        report["layers"]["layer_0_compression_pipeline"] = {
            "ok": False,
            "error": str(exc),
        }
        return (EXIT_CLI_ERROR, report)
    report["layers"]["layer_0_compression_pipeline"] = {
        "ok": True,
        "substrate_id": pipeline_result.substrate_id,
    }

    # ---- Layer 1: archive grammar ----
    try:
        grammar = build_archive_grammar_from_compression_pipeline_result(
            compression_pipeline_result=pipeline_result,
            archive_path=args.archive_path,
            monolithic_single_file=True,
            output_dir=output_dir,
            emit_parser_section_manifest=True,
        )
    except (ArchiveGrammarError, ValueError) as exc:
        report["layers"]["layer_1_archive_grammar"] = {"ok": False, "error": str(exc)}
        return (EXIT_CLI_ERROR, report)
    report["layers"]["layer_1_archive_grammar"] = {
        "ok": True,
        "archive_sha256": grammar.archive_sha256,
        "sections": len(grammar.section_specs),
    }

    # ---- Layer 2: builder ----
    try:
        bundle = build_submission_bundle(
            compression_pipeline_result=pipeline_result,
            archive_grammar_manifest=grammar,
            output_dir=output_dir,
            declared_dependencies=tuple(sorted(set(args.declared_deps))),
            inflate_py_loc_waiver_rationale=args.inflate_py_loc_waiver_rationale,
            attribution_chain_placeholder=attribution_md,
        )
    except (SubmissionBundleError, ValueError) as exc:
        report["layers"]["layer_2_builder"] = {"ok": False, "error": str(exc)}
        return (EXIT_CLI_ERROR, report)
    submission_dir = Path(bundle.submission_dir)
    if not submission_dir.is_absolute():
        submission_dir = _resolve(submission_dir, repo_root)
    bundle_dict = _as_dict(bundle)
    _write_sidecar(submission_dir, _SIDECAR_BUNDLE, bundle_dict)
    report["layers"]["layer_2_builder"] = {
        "ok": True,
        "submission_dir": str(submission_dir),
        "archive_sha256": bundle.archive_sha256,
        "inflate_py_loc": bundle.inflate_py_loc,
        "sidecar": _SIDECAR_BUNDLE,
    }

    # ---- Layer 3: linter (CRITICAL attribution discipline) ----
    pr_body_path = None
    for name in ("PR_BODY.md", "PR_BODY_CANONICAL.md", "PR_DESCRIPTION.md"):
        cand = submission_dir / name
        if cand.is_file():
            pr_body_path = cand
            break
    try:
        lint_verdict = lint_submission_bundle(
            bundle,
            target_repo=args.target_repo,
            pr_body_path=pr_body_path,
        )
    except (SubmissionLinterError, ValueError) as exc:
        report["layers"]["layer_3_linter"] = {"ok": False, "error": str(exc)}
        return (EXIT_CLI_ERROR, report)
    lint_dict = _as_dict(lint_verdict)
    _write_sidecar(submission_dir, _SIDECAR_LINT, lint_dict)
    report["layers"]["layer_3_linter"] = {
        "ok": bool(lint_verdict.overall_clean),
        "overall_clean": bool(lint_verdict.overall_clean),
        "error_count": int(lint_verdict.error_count),
        "warn_count": int(lint_verdict.warn_count),
        "sidecar": _SIDECAR_LINT,
    }
    if not lint_verdict.overall_clean:
        report["lifecycle_verdict"] = "LINT-VIOLATIONS"
        return (EXIT_LINT_VIOLATIONS, report)

    # ---- Layer 4: compliance ----
    try:
        compliance_verdict = enforce_contest_compliance(
            submission_bundle_result=bundle,
            contest_final_strict=True,
            expected_lane_id=args.lane_id,
            competitive_or_innovative_statement=args.competitive_or_innovative_statement,
            output_dir=output_dir,
        )
    except (SubmissionComplianceError, ValueError) as exc:
        report["layers"]["layer_4_compliance"] = {"ok": False, "error": str(exc)}
        return (EXIT_COMPLIANCE_ERRORS, report)
    compliance_dict = _as_dict(compliance_verdict)
    _write_sidecar(submission_dir, _SIDECAR_COMPLIANCE, compliance_dict)
    report["layers"]["layer_4_compliance"] = {
        "ok": bool(compliance_verdict.overall_clean),
        "overall_clean": bool(compliance_verdict.overall_clean),
        "sidecar": _SIDECAR_COMPLIANCE,
    }
    if not compliance_verdict.overall_clean:
        report["lifecycle_verdict"] = "COMPLIANCE-ERRORS"
        # Compliance D3+D5 blockers are operator-gated artifacts, not CLI faults.
        return (EXIT_COMPLIANCE_ERRORS, report)

    # ---- Layer 5: paired auth-eval (prescreen in dry-run; GATED escalation in execute) ----
    execute_paired = False
    paired_env_note = "dry-run prescreen-only (no paid dispatch)"
    if args.execute:
        active, rationale = _execute_paired_env_active()
        if not active and rationale is not None:
            # CONFIRMED set without valid BUDGET -> hard reject per Catalog #199.
            report["layers"]["layer_5_paired_auth_eval"] = {
                "ok": False,
                "error": rationale,
            }
            return (EXIT_CLI_ERROR, report)
        execute_paired = active
        paired_env_note = rationale or (
            "execute mode but paired-env not set; Layer 5 remains plan-only "
            "(Catalog #199 paired-env discipline gates paid-CUDA escalation)"
        )
    try:
        paired_verdict = plan_paired_auth_eval(
            submission_bundle_result=bundle,
            cost_band="smoke",
            cuda_gpu=args.cuda_gpu,
            cuda_platform=args.cuda_platform,
            cpu_target=args.cpu_target,
            dry_run=not execute_paired,
            operator_approved_handle=args.operator_approved_handle if execute_paired else None,
            output_dir=output_dir,
        )
    except (PairedAuthEvalError, ValueError) as exc:
        report["layers"]["layer_5_paired_auth_eval"] = {"ok": False, "error": str(exc)}
        return (EXIT_CLI_ERROR, report)
    paired_dict = _as_dict(paired_verdict)
    _write_sidecar(submission_dir, _SIDECAR_PAIRED, paired_dict)
    paired_pass = paired_verdict.verdict == PairedAuthEvalVerdictKind.PAIRED_PASS.value
    report["layers"]["layer_5_paired_auth_eval"] = {
        "ok": paired_pass,
        "verdict": paired_verdict.verdict,
        "verdict_rationale": paired_verdict.verdict_rationale,
        "paired_env": paired_env_note,
        "sidecar": _SIDECAR_PAIRED,
        "directs_to": "tools/dispatch_modal_paired_auth_eval.py (for paid execution)",
    }
    if not paired_pass:
        report["lifecycle_verdict"] = "MISSING-PAIRED-AXIS"
        return (EXIT_MISSING_PAIRED_AXIS, report)

    # ---- Layer 6: Phase 8 STRICT gate (Catalog #370) verification ----
    gate_violations = _run_layer_6_gate(submission_dir, repo_root)
    report["layers"]["layer_6_catalog_370_gate"] = {
        "ok": not gate_violations,
        "violations": gate_violations,
    }
    if gate_violations:
        # All 4 sidecars present + clean but gate still flags -> diagnostic.
        report["lifecycle_verdict"] = "CATALOG-370-GATE-VIOLATION"
        return (EXIT_CLI_ERROR, report)

    # ---- PACKET-CLEAN: emit operator-gated gh commands (NEVER fired) ----
    archive_rel = archive_abs.relative_to(repo_root) if archive_abs.is_relative_to(repo_root) else archive_abs
    gh_release_cmd = (
        f"gh release create pr_{args.lane_id} {archive_rel} "
        f"--repo adpena/comma_video_compression_challenge "
        f"--title 'submission archive {bundle.archive_sha256[:12]}'"
    )
    gh_pr_cmd = (
        f"gh pr create --repo {args.target_repo} "
        f"--title '<PR title>' --body-file {submission_dir}/PR_BODY.md"
    )
    report["gh_commands_emitted"] = True
    report["operator_gated_commands"] = {
        "step_1_host_archive": gh_release_cmd,
        "step_2_create_pr": gh_pr_cmd,
        "note": (
            "Operator-gated per CLAUDE.md 'Executing actions with care' + "
            "'Public Disclosure Hygiene'. This CLI NEVER fires gh commands. "
            "Sole-author Alejandro Pena <adpena@gmail.com>; ZERO Claude/Anthropic "
            "tokens in PR-facing surfaces per user_pr_attribution discipline."
        ),
    }
    report["lifecycle_verdict"] = "OPERATOR-GATED"
    return (EXIT_OPERATOR_GATED, report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="operator_pr_submission_full_lifecycle",
        description=(
            "Canonical single-command end-to-end PR-submission lifecycle "
            "(Phase 9, Layer 7). Orchestrates Layers 0-6 + the Phase 8 Catalog "
            "#370 gate. NEVER fires gh commands (operator-gated)."
        ),
    )
    parser.add_argument("--lane-id", required=True, help="Lane registry id.")
    parser.add_argument(
        "--substrate-trainer", type=Path, required=True,
        help="Path to experiments/train_substrate_<id>.py.",
    )
    parser.add_argument(
        "--recipe-path", type=Path, required=True,
        help="Path to .omx/operator_authorize_recipes/substrate_<id>_*.yaml.",
    )
    parser.add_argument(
        "--archive-path", type=Path, required=True,
        help="Path to the trainer-emitted archive.zip.",
    )
    parser.add_argument(
        "--video-path", type=Path, default=Path("upstream/videos/0.mkv"),
        help="Path to the contest video (canonical: upstream/videos/0.mkv).",
    )
    parser.add_argument(
        "--target-repo", default="commaai/comma_video_compression_challenge",
        help="Upstream PR target repo.",
    )
    parser.add_argument(
        "--predecessors", nargs="*", default=None,
        help="Predecessor attribution specs '@handle:PRnumber:slug' "
        "(e.g. @SajayR:56:HNeRV_substrate).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output submission_dir/ (canonical: submissions/pr<N>_<lane>/).",
    )
    parser.add_argument(
        "--hardware-substrate", default=HardwareSubstrateClass.AUTO.value,
        choices=[c.value for c in HardwareSubstrateClass],
        help="Compression-pipeline hardware substrate class (default 'auto').",
    )
    parser.add_argument(
        "--declared-deps", nargs="+", default=["numpy"],
        help="inflate.py declared external deps (numpy-portable default: numpy).",
    )
    parser.add_argument(
        "--inflate-py-loc-waiver-rationale", default=None,
        help="Substantive rationale (>=4 chars) when inflate.py LOC > budget.",
    )
    parser.add_argument(
        "--competitive-or-innovative-statement", default=None,
        help="PR101+ contest competitive-or-innovative statement for compliance.",
    )
    parser.add_argument(
        "--cuda-gpu", default="T4",
        help="Layer 5 paired CUDA GPU class (default T4).",
    )
    parser.add_argument(
        "--cuda-platform", default="modal", choices=["modal", "vastai", "lightning"],
        help="Layer 5 paired CUDA platform (Linux x86_64; default modal).",
    )
    parser.add_argument(
        "--cpu-target", default="linux_x86_64_modal",
        help="Layer 5 paired CPU target (1:1 contest-compliant Linux x86_64).",
    )
    parser.add_argument(
        "--operator-approved-handle", default=None,
        help="Operator handle '<handle>:<UTC>' for paid Layer 5 escalation "
        "(REQUIRED with --execute + paired-env per Catalog #199).",
    )
    parser.add_argument(
        "--skip-protocol-verification", action="store_true", default=False,
        help="Bypass Catalog #270 umbrella verification at Layer 0 (dry prep only).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Default. Layers 0-4 + 6 at $0; Layer 5 prescreen-only (no paid dispatch).",
    )
    mode.add_argument(
        "--execute", action="store_true", default=False,
        help="Full pipeline. Layer 5 paired-CUDA GATED escalation requires "
        "Catalog #199 paired-env. STILL stops at exit 4 before any gh command.",
    )
    parser.add_argument(
        "--json", action="store_true", default=False,
        help="Emit canonical machine-readable JSON (sorted keys).",
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False,
        help="Suppress progress messages on stderr.",
    )
    return parser


_EXIT_LABELS = {
    EXIT_PACKET_CLEAN: "PACKET-CLEAN",
    EXIT_LINT_VIOLATIONS: "LINT-VIOLATIONS",
    EXIT_COMPLIANCE_ERRORS: "COMPLIANCE-ERRORS",
    EXIT_MISSING_PAIRED_AXIS: "MISSING-PAIRED-AXIS",
    EXIT_OPERATOR_GATED: "OPERATOR-GATED",
    EXIT_CLI_ERROR: "CLI-ERROR",
}


def _render_human(exit_code: int, report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Phase 9 Layer 7 — Canonical PR-Submission Full Lifecycle")
    lines.append("=" * 72)
    lines.append(f"  lane_id:        {report['lane_id']}")
    lines.append(f"  target_repo:    {report['target_repo']}")
    lines.append(f"  mode:           {report['mode']}")
    lines.append(f"  verdict:        {_EXIT_LABELS.get(exit_code, '?')} (exit {exit_code})")
    lines.append("")
    lines.append("  Layer verdicts:")
    for key, state in report.get("layers", {}).items():
        ok = state.get("ok")
        mark = "PASS" if ok else "FAIL"
        detail = state.get("verdict") or state.get("error") or ""
        lines.append(f"    [{mark}] {key}  {detail}")
    if report.get("gh_commands_emitted"):
        lines.append("")
        lines.append("  OPERATOR-GATED next steps (NOT fired by this CLI):")
        og = report.get("operator_gated_commands", {})
        lines.append(f"    1. {og.get('step_1_host_archive', '')}")
        lines.append(f"    2. {og.get('step_2_create_pr', '')}")
    lines.append("")
    lines.append(
        "  Per CLAUDE.md 'Executing actions with care': gh commands are NEVER "
        "fired by this CLI. Sole-author Alejandro Pena <adpena@gmail.com>; "
        "ZERO Claude/Anthropic tokens in PR-facing surfaces."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    exit_code, report = run_full_lifecycle(args)

    if args.json:
        sys.stdout.write(json.dumps(report, sort_keys=True, indent=2, default=str) + "\n")
    elif not args.quiet:
        sys.stdout.write(_render_human(exit_code, report))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
