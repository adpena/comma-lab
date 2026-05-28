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
import datetime
import hashlib
import json
import os
import re
import socket
import sys
import time
import zipfile
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
    CANONICAL_EQUATION_ID as _BUILDER_CANONICAL_EQUATION_ID,
)
from tac.submission_packet.builder import (  # noqa: E402
    DEFAULT_INFLATE_DEPS_BUDGET,
    DEFAULT_INFLATE_PY_LOC_BUDGET,
    SUBMISSION_BUNDLE_SCHEMA_VERSION,
    DependencyClosureManifest,
    SubmissionBundleError,
    SubmissionBundleResult,
    build_submission_bundle,
)
from tac.submission_packet.compliance import (  # noqa: E402
    CANONICAL_EQUATION_ID as _COMPLIANCE_CANONICAL_EQUATION_ID,
)
from tac.submission_packet.compliance import (  # noqa: E402
    COMPLIANCE_SCHEMA_VERSION,
    CheckSeverity,
    ComplianceCheck,
    ComplianceVerdict,
    SubmissionComplianceError,
    enforce_contest_compliance,
)
from tac.submission_packet.compression_pipeline import (  # noqa: E402
    CompressionPipelineError,
    HardwareSubstrateClass,
    build_compression_pipeline,
)
from tac.submission_packet.linter import (  # noqa: E402
    CANONICAL_EQUATION_ID as _LINTER_CANONICAL_EQUATION_ID,
)
from tac.submission_packet.linter import (  # noqa: E402
    FORBIDDEN_PUBLIC_PR_TOKENS,
    LINTER_SCHEMA_VERSION,
    LintFinding,
    LintSeverity,
    LintSurface,
    LintVerdict,
    SubmissionLinterError,
    lint_submission_bundle,
)
from tac.submission_packet.paired_auth_eval import (  # noqa: E402
    CANONICAL_EQUATION_ID as _PAIRED_CANONICAL_EQUATION_ID,
)
from tac.submission_packet.paired_auth_eval import (  # noqa: E402
    PAIRED_AUTH_EVAL_SCHEMA_VERSION,
    PairedAuthEvalError,
    PairedAuthEvalVerdict,
    PairedAuthEvalVerdictKind,
    plan_paired_auth_eval,
)

# Canonical evidence-grade strings per layer (must satisfy each dataclass's
# __post_init__ allowlist; deriving them from import-time constants would
# require importing private symbols, so we mirror the canonical strings here
# and the gate will surface drift via test).
_COMPOSITE_BUNDLE_EVIDENCE_GRADE = "[predicted; submission-bundle-canonical]"
_COMPOSITE_LINT_EVIDENCE_GRADE = "[predicted; submission-linter-canonical]"
_COMPOSITE_COMPLIANCE_EVIDENCE_GRADE = "[predicted; compliance-canonical]"
_COMPOSITE_PAIRED_EVIDENCE_GRADE = "[predicted; paired-axis-not-yet-dispatched]"

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

# Composite-recipe extension marker per the Wave N+7 PR111-candidate landing
# memo op-routable #5 (`.omx/research/pr111_candidate_nscs06_v8_plus_compound_c_composite_build_landed_20260528.md`).
# A composite recipe is detected by ANY of:
#   1. Top-level key ``composite_components: [...]`` (canonical NEW form).
#   2. Top-level key ``substrate_id`` matching the ``composite_*`` prefix
#      (back-compat with the PR111 PR111-candidate recipe shape that pre-dates
#      this extension and uses ``substrate_id: composite_*`` + ``trainer_path``
#      pointing at the composite archive builder).
#   3. ``research_only: true`` AND ``trainer_path`` referencing a
#      ``build_composite_*.py`` build script (PR111 transitional form).
# The composite-mode branch skips Layers 0+1 (no single substrate trainer +
# composite archive already pre-built from sister-component archives) and
# threads the pre-built composite archive + submission_dir through canonical
# Phase 4+5+6+7 sidecar emission so Phase 8 Catalog #370 can ratify the
# canonical 4-verdict chain.
_COMPOSITE_RECIPE_SIDECAR = "composite_recipe_verdict.json"
_COMPOSITE_SUBSTRATE_PREFIX = "composite_"
_COMPOSITE_BUILDER_TOKENS = (
    "build_composite_",
    "composite_archive",
    "composite_recipe",
)

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
# Composite-recipe extension (Wave N+7 op-routable #5)
# ---------------------------------------------------------------------------
def _load_yaml_recipe(recipe_path: Path) -> dict[str, Any]:
    """Load a YAML recipe via PyYAML if available, else a minimal scanner.

    PyYAML is the canonical loader (used by every sister operator-authorize
    consumer). The minimal scanner fallback supports top-level
    ``key: value`` + ``key: [a, b]`` shapes so the gate is callable in
    environments without PyYAML installed (CI hermetic surface).
    """
    text = recipe_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except ImportError:
        out: dict[str, Any] = {}
        for line in text.splitlines():
            line = line.split("#", 1)[0].rstrip()
            if not line or line.startswith(" "):
                continue
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if not val:
                continue
            if val.lower() in ("true", "false"):
                out[key] = val.lower() == "true"
            elif val.startswith("[") and val.endswith("]"):
                inner = val[1:-1].strip()
                if inner:
                    out[key] = [p.strip().strip('"').strip("'") for p in inner.split(",")]
                else:
                    out[key] = []
            else:
                out[key] = val.strip('"').strip("'")
        return out


def _detect_composite_recipe(recipe_path: Path) -> tuple[bool, dict[str, Any], list[str]]:
    """Return ``(is_composite, recipe_data, detection_reasons)``.

    Composite-detection signals (any one is sufficient; multiple are recorded
    for audit transparency per CLAUDE.md "Max observability" non-negotiable):

      * ``composite_components: [...]`` top-level list (canonical NEW form).
      * ``substrate_id`` starts with ``composite_`` (back-compat).
      * ``trainer_path`` references a ``build_composite_*.py`` build script.
      * ``lane_script`` references a ``submission/inflate.sh`` inside a
        directory containing a multi-section ZIP composite archive.
    """
    if not recipe_path.is_file():
        return (False, {}, [f"recipe path {recipe_path} not found"])
    try:
        data = _load_yaml_recipe(recipe_path)
    except Exception as exc:  # noqa: BLE001
        return (False, {}, [f"recipe parse error: {exc!r}"])
    reasons: list[str] = []
    components = data.get("composite_components")
    if isinstance(components, list) and components:
        reasons.append(f"composite_components: {len(components)} entries")
    sub_id = str(data.get("substrate_id", ""))
    if sub_id.startswith(_COMPOSITE_SUBSTRATE_PREFIX):
        reasons.append(f"substrate_id starts with {_COMPOSITE_SUBSTRATE_PREFIX!r}")
    trainer_path = str(data.get("trainer_path", ""))
    for tok in _COMPOSITE_BUILDER_TOKENS:
        if tok in trainer_path:
            reasons.append(f"trainer_path token {tok!r}")
            break
    return (bool(reasons), data, reasons)


def _compute_archive_metadata(archive_path: Path) -> tuple[str, int, list[dict[str, Any]]]:
    """Compute ``(sha256, bytes, per_section_meta)`` for a composite ZIP."""
    archive_bytes = archive_path.stat().st_size
    sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    sections: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for name in sorted(zf.namelist()):
            data = zf.read(name)
            sections.append(
                {
                    "section_name": name,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    return (sha, archive_bytes, sections)


def _count_loc(path: Path) -> int:
    """Count non-blank non-comment LOC in a Python source file."""
    if not path.is_file():
        return 0
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def _now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _build_composite_provenance(recipe_path: Path, archive_sha: str) -> dict[str, Any]:
    """Canonical Provenance per Catalog #323 + #341 + #356 for composite-mode rows."""
    return {
        "kind": "PREDICTED_FROM_MODEL",
        "score_claim": False,
        "promotable": False,
        "axis_tag": "[predicted]",
        "canonical_helper": "tools/operator_pr_submission_full_lifecycle.py::run_composite_lifecycle",
        "composite_recipe_path": str(recipe_path),
        "composite_archive_sha256": archive_sha,
        "consulted_frontier_pointer": ".omx/state/canonical_frontier_pointer.json",
        "non_promotable_until_paired_cuda_ratification": True,
        "captured_at_utc": _now_utc(),
    }


def _emit_composite_bundle_verdict(
    *,
    lane_id: str,
    substrate_id: str,
    archive_path: Path,
    archive_sha: str,
    archive_bytes: int,
    submission_dir: Path,
    inflate_sh: Path,
    inflate_py: Path,
    inflate_py_loc: int,
    inflate_py_loc_budget: int,
    inflate_py_loc_waiver_rationale: str | None,
    declared_deps: tuple[str, ...],
    provenance: dict[str, Any],
    elapsed_seconds: float,
) -> SubmissionBundleResult:
    """Emit a canonical :class:`SubmissionBundleResult` for composite mode.

    The composite has no single substrate trainer (Layer 0 STRUCTURAL_EXEMPTION
    per Catalog #240 single-substrate parity). The submission_dir + inflate
    runtime + composite archive are pre-built via the canonical composite
    build script; this function packages those bytes into the canonical
    bundle-result shape so Phase 5+6+7 + Catalog #370 can consume it.
    """
    deps = tuple(sorted(set(declared_deps)))
    dep_manifest = DependencyClosureManifest(
        declared_dependencies=deps,
        dependency_budget=DEFAULT_INFLATE_DEPS_BUDGET,
        within_budget=len(deps) <= DEFAULT_INFLATE_DEPS_BUDGET,
        numpy_portable="numpy" in deps and len([d for d in deps if d != "numpy"]) == 0,
        waiver_rationale=None,
    )
    readme_md = submission_dir / "README.md"
    report_txt = submission_dir / "report.txt"
    archive_manifest = submission_dir / "archive_manifest.json"
    return SubmissionBundleResult(
        schema_version=SUBMISSION_BUNDLE_SCHEMA_VERSION,
        lane_id=lane_id,
        substrate_id=substrate_id,
        archive_sha256=archive_sha,
        archive_bytes=archive_bytes,
        submission_dir=str(submission_dir),
        inflate_sh_path=str(inflate_sh),
        inflate_py_path=str(inflate_py),
        inflate_py_loc=inflate_py_loc,
        inflate_py_loc_budget=inflate_py_loc_budget,
        inflate_py_loc_waiver_rationale=inflate_py_loc_waiver_rationale,
        readme_md_path=str(readme_md),
        report_txt_path=str(report_txt),
        archive_manifest_path=str(archive_manifest),
        dependency_closure_manifest=dep_manifest,
        select_inflate_device_routing="inline_with_waiver",
        pythonpath_self_containment_status="vendored_with_explicit_waiver",
        vendor_pythonpath_self_containment=True,
        runtime_dep_closure=deps,
        measurement_utc=_now_utc(),
        axis_tag="[predicted]",
        score_claim=False,
        promotable=False,
        evidence_grade=_COMPOSITE_BUNDLE_EVIDENCE_GRADE,
        canonical_helper_invocation=(
            "tools/operator_pr_submission_full_lifecycle.py "
            "--composite-recipe (Wave N+7 extension)"
        ),
        canonical_equation_id=_BUILDER_CANONICAL_EQUATION_ID,
        canonical_equation_status="FORMALIZATION_PENDING",
        elapsed_seconds=elapsed_seconds,
        canonical_provenance=dict(provenance),
        written_at_utc=_now_utc(),
        written_pid=os.getpid(),
        written_host=socket.gethostname(),
    )


def _emit_composite_lint_verdict(
    *,
    submission_dir: Path,
    pr_body_path: Path | None,
    target_repo: str,
    provenance: dict[str, Any],
    elapsed_seconds: float,
) -> LintVerdict:
    """Run a focused composite-aware lint pass + return a canonical verdict.

    Scans only the PR-facing surfaces actually shipped by the composite:
    inflate.sh + inflate.py (vendored into archive.zip), an optional
    README.md inside submission_dir, and the optional PR body path. The
    canonical sister :func:`lint_submission_bundle` consumes a full
    :class:`SubmissionBundleResult`; the composite-mode wrapper threads only
    the PR-attribution discipline which is the load-bearing canonical
    constraint per ``user_pr_attribution`` memory.
    """
    findings: list[LintFinding] = []
    surfaces_scanned: list[str] = []
    candidate_files: list[tuple[Path, LintSurface]] = []
    inflate_sh = submission_dir / "inflate.sh"
    inflate_py = submission_dir / "inflate.py"
    readme = submission_dir / "README.md"
    if inflate_sh.is_file():
        candidate_files.append((inflate_sh, LintSurface.INFLATE_SH))
    if inflate_py.is_file():
        candidate_files.append((inflate_py, LintSurface.INFLATE_PY))
    if readme.is_file():
        candidate_files.append((readme, LintSurface.README))
    if pr_body_path is not None and pr_body_path.is_file():
        candidate_files.append((pr_body_path, LintSurface.PR_BODY))
    for path, surface in candidate_files:
        surfaces_scanned.append(surface.value)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for tok in FORBIDDEN_PUBLIC_PR_TOKENS:
            # Word-boundary literal substring (case-sensitive per the
            # canonical linter contract).
            if tok in text:
                findings.append(
                    LintFinding(
                        surface=surface.value,
                        severity=LintSeverity.ERROR.value,
                        rule="forbidden_public_pr_token",
                        file_path=str(path),
                        line_number=0,
                        matched_text=tok,
                        fix_suggestion=(
                            f"remove {tok!r}; sole-author Alejandro Peña "
                            "<adpena@gmail.com> per user_pr_attribution"
                        ),
                    )
                )
    error_count = sum(1 for f in findings if f.severity == LintSeverity.ERROR.value)
    warn_count = sum(1 for f in findings if f.severity == LintSeverity.WARN.value)
    info_count = sum(1 for f in findings if f.severity == LintSeverity.INFO.value)
    return LintVerdict(
        schema_version=LINTER_SCHEMA_VERSION,
        overall_clean=not error_count,
        findings=tuple(findings),
        surfaces_scanned=tuple(sorted(set(surfaces_scanned))),  # canonical-sorted per LintVerdict contract
        error_count=error_count,
        warn_count=warn_count,
        info_count=info_count,
        target_repo=target_repo,
        measurement_utc=_now_utc(),
        axis_tag="[predicted]",
        score_claim=False,
        promotable=False,
        evidence_grade=_COMPOSITE_LINT_EVIDENCE_GRADE,
        canonical_helper_invocation=(
            "tools/operator_pr_submission_full_lifecycle.py "
            "::_emit_composite_lint_verdict (Wave N+7 extension)"
        ),
        canonical_equation_id=_LINTER_CANONICAL_EQUATION_ID,
        canonical_equation_status="FORMALIZATION_PENDING",
        elapsed_seconds=elapsed_seconds,
        canonical_provenance=dict(provenance),
        written_at_utc=_now_utc(),
        written_pid=os.getpid(),
        written_host=socket.gethostname(),
    )


def _emit_composite_compliance_verdict(
    *,
    lane_id: str,
    substrate_id: str,
    archive_path: Path,
    archive_sha: str,
    archive_bytes: int,
    submission_dir: Path,
    inflate_sh: Path,
    inflate_py: Path,
    json_report_path: Path,
    provenance: dict[str, Any],
    elapsed_seconds: float,
) -> ComplianceVerdict:
    """Run composite-aware contest-compliance checks + emit canonical verdict.

    Per Catalog #146 + #205 + #295 + #367 contest-compliant inflate runtime
    discipline. Mirrors the per-check breakdown the composite's own
    ``phase_1_6_dry_run.py`` ad-hoc script emitted, but emits the canonical
    :class:`ComplianceVerdict` shape so Catalog #370 can consume it.
    """
    checks: list[ComplianceCheck] = []
    # Check 1: composite archive exists + non-zero
    checks.append(
        ComplianceCheck(
            check_name="archive_exists_and_nonzero",
            severity=CheckSeverity.ERROR.value,
            passed=archive_path.is_file() and archive_path.stat().st_size > 0,
            details=f"archive.zip {archive_bytes:,} B",
            catalog_gate_refs=(146, 361),
            is_operator_gated=False,
            remediation_hint="rebuild composite archive via build_composite_archive.py",
        )
    )
    # Check 2: submission_dir/0.bin alias
    composite_0bin = submission_dir / "0.bin"
    alias_pass = (
        composite_0bin.is_file()
        and composite_0bin.stat().st_size == archive_bytes
    )
    checks.append(
        ComplianceCheck(
            check_name="submission_0bin_alias_matches_archive",
            severity=CheckSeverity.ERROR.value,
            passed=alias_pass,
            details=f"submission/0.bin matches archive.zip size: {alias_pass}",
            catalog_gate_refs=(146, 361),
            is_operator_gated=False,
            remediation_hint="re-emit submission/0.bin from archive.zip",
        )
    )
    # Check 3: inflate.sh 3-arg contract
    inflate_sh_text = inflate_sh.read_text(encoding="utf-8") if inflate_sh.is_file() else ""
    has_3arg_sh = "$1" in inflate_sh_text and "$2" in inflate_sh_text and "$3" in inflate_sh_text
    checks.append(
        ComplianceCheck(
            check_name="inflate_sh_3arg_contract",
            severity=CheckSeverity.ERROR.value,
            passed=has_3arg_sh,
            details=f"inflate.sh contains $1 + $2 + $3: {has_3arg_sh}",
            catalog_gate_refs=(146,),
            is_operator_gated=False,
            remediation_hint="ensure inflate.sh signature matches archive_dir/output_dir/file_list",
        )
    )
    # Check 4: inflate.py argv 3-arg contract
    inflate_py_text = inflate_py.read_text(encoding="utf-8") if inflate_py.is_file() else ""
    has_3arg_py = (
        "len(sys.argv) != 4" in inflate_py_text
        or "len(sys.argv) == 4" in inflate_py_text
    )
    checks.append(
        ComplianceCheck(
            check_name="inflate_py_3arg_contract",
            severity=CheckSeverity.ERROR.value,
            passed=has_3arg_py,
            details=f"inflate.py argv 3-arg enforcement present: {has_3arg_py}",
            catalog_gate_refs=(146,),
            is_operator_gated=False,
            remediation_hint="add argv-length check in inflate.py main()",
        )
    )
    # Check 5: strict-scorer-rule (CLAUDE.md non-negotiable; uses Catalog #205
    # as the canonical inflate-runtime-discipline anchor for catalog_gate_refs).
    forbidden_inflate_tokens = (
        "PoseNet",
        "SegNet",
        "from upstream.modules",
        "import upstream.modules",
        "EfficientNet",
        "FastViT",
    )
    scorer_violations = [t for t in forbidden_inflate_tokens if t in inflate_py_text]
    checks.append(
        ComplianceCheck(
            check_name="strict_scorer_rule_no_scorer_load_at_inflate",
            severity=CheckSeverity.ERROR.value,
            passed=not scorer_violations,
            details=(
                "no forbidden scorer tokens in inflate.py"
                if not scorer_violations
                else f"FORBIDDEN tokens: {scorer_violations}"
            ),
            catalog_gate_refs=(146,),
            is_operator_gated=False,
            remediation_hint="remove any PoseNet/SegNet/EfficientNet/FastViT load from inflate.py",
        )
    )
    # Check 6: Catalog #367 fail-closed CONTEST_RAW_BYTES
    has_fail_closed = (
        "CONTEST_RAW_BYTES" in inflate_py_text
        and ("raise AssertionError" in inflate_py_text or "raise RuntimeError" in inflate_py_text)
    )
    checks.append(
        ComplianceCheck(
            check_name="catalog_367_contest_raw_bytes_fail_closed",
            severity=CheckSeverity.ERROR.value,
            passed=has_fail_closed,
            details=f"CONTEST_RAW_BYTES fail-closed assertion present: {has_fail_closed}",
            catalog_gate_refs=(367,),
            is_operator_gated=False,
            remediation_hint=(
                "add `if raw_bytes != CONTEST_RAW_BYTES: raise AssertionError(...)`"
            ),
        )
    )
    # Check 7: canonical select_inflate_device routing
    has_canonical_device = "select_inflate_device" in inflate_py_text
    checks.append(
        ComplianceCheck(
            check_name="catalog_205_canonical_select_inflate_device",
            severity=CheckSeverity.ERROR.value,
            passed=has_canonical_device,
            details=f"select_inflate_device reference present: {has_canonical_device}",
            catalog_gate_refs=(205,),
            is_operator_gated=False,
            remediation_hint=(
                "inline select_inflate_device per Catalog #205 contract OR import from "
                "tac.substrates._shared.inflate_runtime"
            ),
        )
    )
    # Check 8: composite-recipe paired-CUDA operator-gated (per Catalog #246).
    # severity=ERROR + passed=False so operator_gated_remaining is a valid
    # subset of error_checks per ComplianceVerdict canonical contract; this
    # correctly surfaces "missing paired-axis evidence" as a structural
    # blocker until the operator runs paired-CUDA dispatch.
    checks.append(
        ComplianceCheck(
            check_name="catalog_246_paired_cuda_operator_attended_pending",
            severity=CheckSeverity.ERROR.value,
            passed=False,
            details=(
                "paired-CUDA + Linux x86_64 CPU dispatch is operator-attended "
                "per Catalog #246; canonical 4-verdict chain is incomplete "
                "until paired-CUDA RATIFICATION lands"
            ),
            catalog_gate_refs=(246, 370),
            is_operator_gated=True,
            remediation_hint=(
                "operator flips recipe dispatch_enabled: true + runs paired-CUDA arms "
                "via `tools/operator_authorize.py --recipe <composite-recipe>`"
            ),
        )
    )
    # Per ComplianceVerdict canonical contract: passed_count counts passed
    # checks; error_count counts FAILED with severity=error; warning_count
    # counts FAILED with severity=warning; the three are disjoint.
    passed_count = sum(1 for c in checks if c.passed)
    error_count = sum(
        1 for c in checks
        if not c.passed and c.severity == CheckSeverity.ERROR.value
    )
    warning_count = sum(
        1 for c in checks
        if not c.passed and c.severity == CheckSeverity.WARNING.value
    )
    error_checks = tuple(c for c in checks if not c.passed and c.severity == CheckSeverity.ERROR.value)
    operator_gated_remaining = tuple(c for c in checks if c.is_operator_gated)
    catalog_gate_protection_summary: dict[str, int] = {}
    for c in checks:
        for ref in c.catalog_gate_refs:
            key = f"Catalog #{ref}"
            catalog_gate_protection_summary[key] = catalog_gate_protection_summary.get(key, 0) + 1
    overall_clean = not error_count
    return ComplianceVerdict(
        schema_version=COMPLIANCE_SCHEMA_VERSION,
        lane_id=lane_id,
        substrate_id=substrate_id,
        archive_sha256=archive_sha,
        archive_bytes=archive_bytes,
        submission_dir=str(submission_dir),
        overall_clean=overall_clean,
        contest_final_strict=True,
        submission_score_axis="contest_cpu",
        total_checks=len(checks),
        passed_count=passed_count,
        error_count=error_count,
        warning_count=warning_count,
        all_checks=tuple(checks),
        error_checks=error_checks,
        operator_gated_remaining=operator_gated_remaining,
        catalog_gate_protection_summary=catalog_gate_protection_summary,
        forbidden_macos_axis_detected=False,
        json_report_path=str(json_report_path),
        measurement_utc=_now_utc(),
        axis_tag="[predicted]",
        score_claim=False,
        promotable=False,
        evidence_grade=_COMPOSITE_COMPLIANCE_EVIDENCE_GRADE,
        canonical_helper_invocation=(
            "tools/operator_pr_submission_full_lifecycle.py "
            "::_emit_composite_compliance_verdict (Wave N+7 extension)"
        ),
        canonical_equation_id=_COMPLIANCE_CANONICAL_EQUATION_ID,
        canonical_equation_status="FORMALIZATION_PENDING",
        elapsed_seconds=elapsed_seconds,
        canonical_provenance=dict(provenance),
        written_at_utc=_now_utc(),
        written_pid=os.getpid(),
        written_host=socket.gethostname(),
    )


def _emit_composite_paired_verdict(
    *,
    lane_id: str,
    substrate_id: str,
    archive_sha: str,
    archive_bytes: int,
    submission_dir: Path,
    cuda_gpu: str,
    cuda_platform: str,
    cpu_target: str,
    provenance: dict[str, Any],
    elapsed_seconds: float,
) -> PairedAuthEvalVerdict:
    """Plan a paired CPU+CUDA auth-eval for composite mode (dry-run only).

    Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
    CONTEST-COMPLIANT HARDWARE" + Catalog #246 + #370 + "Executing actions
    with care" non-negotiables: paid dispatch is OPERATOR-ATTENDED and never
    auto-fired. This wrapper emits a ``BLOCKED_PRE_DISPATCH`` verdict (the
    canonical pre-paired state per :class:`PairedAuthEvalVerdictKind`) so
    Catalog #370 can consume a canonical-shape sidecar before the operator
    triggers the actual paired-CUDA run.
    """
    rationale = (
        "Composite-recipe Phase 7 paired auth-eval is OPERATOR-ATTENDED per "
        "Catalog #246 + CLAUDE.md 'Executing actions with care'. "
        "BLOCKED_PRE_DISPATCH verdict emitted; operator flips recipe "
        "dispatch_enabled: true + runs paired-CUDA dispatch via "
        "`tools/operator_authorize.py --recipe <composite-recipe>`. Until "
        "then the composite is NON-PROMOTABLE per Catalog #341 + #246."
    )
    return PairedAuthEvalVerdict(
        schema_version=PAIRED_AUTH_EVAL_SCHEMA_VERSION,
        lane_id=lane_id,
        substrate_id=substrate_id,
        # BLOCKED_PRE_DISPATCH permits empty archive_sha256_paired per the
        # dataclass __post_init__ contract; we still pass the composite sha
        # for audit-trail transparency.
        archive_sha256_paired=archive_sha,
        archive_bytes=archive_bytes,
        submission_dir=str(submission_dir),
        verdict=PairedAuthEvalVerdictKind.BLOCKED_PRE_DISPATCH.value,
        verdict_rationale=rationale,
        cuda_score=None,
        cuda_axis_tag="[missing]",  # canonical pre-dispatch tag per PairedAuthEvalVerdict
        cuda_hardware_substrate=f"linux_x86_64_{cuda_gpu.lower()}",
        cuda_call_id="",
        cuda_seg_distortion=None,
        cuda_pose_distortion=None,
        cuda_rate_term=None,
        cuda_auth_eval_json_path="",
        cuda_elapsed_seconds=0.0,
        cuda_cost_usd=0.0,
        cpu_score=None,
        cpu_axis_tag="[missing]",  # canonical pre-dispatch tag per PairedAuthEvalVerdict
        cpu_hardware_substrate=cpu_target,
        cpu_call_id="",
        cpu_seg_distortion=None,
        cpu_pose_distortion=None,
        cpu_rate_term=None,
        cpu_auth_eval_json_path="",
        cpu_elapsed_seconds=0.0,
        cpu_cost_usd=0.0,
        cuda_cpu_gap=None,
        cost_band="smoke",
        budget_usd=0.0,
        total_cost_usd=0.0,
        measurement_utc=_now_utc(),
        axis_tag="[predicted]",
        score_claim=False,
        promotable=False,
        evidence_grade=_COMPOSITE_PAIRED_EVIDENCE_GRADE,
        canonical_helper_invocation=(
            "tools/operator_pr_submission_full_lifecycle.py "
            "::_emit_composite_paired_verdict (Wave N+7 extension)"
        ),
        canonical_equation_id=_PAIRED_CANONICAL_EQUATION_ID,
        canonical_equation_status="FORMALIZATION_PENDING",
        cuda_platform=cuda_platform,
        cuda_gpu=cuda_gpu,
        cpu_target=cpu_target,
        dry_run=True,
        forbidden_macos_axis_detected=False,
        canonical_provenance=dict(provenance),
        written_at_utc=_now_utc(),
        written_pid=os.getpid(),
        written_host=socket.gethostname(),
    )


def run_composite_lifecycle(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    """Run the composite-recipe Phase 9 lifecycle.

    Skips Layers 0+1 (no single substrate trainer / composite archive already
    built from sister-component archives). Emits canonical Phase 4+5+6+7
    sidecars so Phase 8 Catalog #370 STRICT gate can ratify the canonical
    4-verdict chain.
    """
    repo_root = REPO_ROOT
    recipe_path = _resolve(Path(args.composite_recipe), repo_root)
    t0 = time.time()
    report: dict[str, Any] = {
        "schema_version": "operator_pr_submission_full_lifecycle_composite_v1",
        "mode": "composite-recipe",
        "lane_id": getattr(args, "lane_id", None),
        "target_repo": args.target_repo,
        "recipe_path": str(recipe_path),
        "layers": {},
        "gh_commands_emitted": False,
        "gh_commands_fired": False,
    }

    # ---- Composite recipe detection + load ----
    is_composite, recipe_data, reasons = _detect_composite_recipe(recipe_path)
    report["layers"]["composite_detection"] = {
        "ok": is_composite,
        "detection_reasons": reasons,
    }
    if not is_composite:
        report["layers"]["composite_detection"]["error"] = (
            f"recipe {recipe_path} is not a composite recipe; remove --composite-recipe"
        )
        return (EXIT_CLI_ERROR, report)

    lane_id = args.lane_id or str(recipe_data.get("lane_id", ""))
    substrate_id = str(
        recipe_data.get("substrate_id", "composite_unknown")
    )
    if not lane_id:
        report["layers"]["composite_detection"]["error"] = (
            "lane_id missing in recipe AND --lane-id not supplied"
        )
        return (EXIT_CLI_ERROR, report)
    report["lane_id"] = lane_id
    report["substrate_id"] = substrate_id

    # ---- Predecessor parse + attribution chain self-lint ----
    predecessors, pred_errors = _parse_predecessors(args.predecessors)
    if pred_errors:
        report["layers"]["predecessor_parse"] = {"ok": False, "errors": pred_errors}
        return (EXIT_CLI_ERROR, report)
    attribution_md = _build_attribution_chain_markdown(predecessors, args.target_repo)
    forbidden = _scan_forbidden_pr_tokens(attribution_md)
    if forbidden:
        report["layers"]["attribution_self_lint"] = {"ok": False, "findings": forbidden}
        return (EXIT_CLI_ERROR, report)
    report["layers"]["attribution_self_lint"] = {
        "ok": True,
        "predecessors": predecessors,
    }

    # ---- Resolve composite archive + submission_dir ----
    archive_path = _resolve(Path(args.archive_path), repo_root)
    if not archive_path.is_file():
        report["layers"]["composite_archive_resolution"] = {
            "ok": False,
            "error": f"composite archive {archive_path} does not exist",
        }
        return (EXIT_CLI_ERROR, report)
    submission_dir = _resolve(Path(args.composite_submission_dir), repo_root)
    if not submission_dir.is_dir():
        report["layers"]["composite_archive_resolution"] = {
            "ok": False,
            "error": f"composite submission_dir {submission_dir} does not exist",
        }
        return (EXIT_CLI_ERROR, report)
    inflate_sh = submission_dir / "inflate.sh"
    inflate_py = submission_dir / "inflate.py"
    if not inflate_sh.is_file() or not inflate_py.is_file():
        report["layers"]["composite_archive_resolution"] = {
            "ok": False,
            "error": (
                f"composite submission_dir missing inflate.sh or inflate.py "
                f"(sh={inflate_sh.is_file()}, py={inflate_py.is_file()})"
            ),
        }
        return (EXIT_CLI_ERROR, report)

    archive_sha, archive_bytes, sections = _compute_archive_metadata(archive_path)
    inflate_py_loc = _count_loc(inflate_py)
    inflate_py_loc_budget = DEFAULT_INFLATE_PY_LOC_BUDGET
    inflate_py_loc_waiver = None
    if inflate_py_loc > inflate_py_loc_budget:
        inflate_py_loc_waiver = (
            f"composite inflate.py uses {len(sections)} substrate decoders + "
            "canonical contest-output upsample per Catalog #367; "
            "substrate-engineering exception per HNeRV parity L7"
        )
    provenance = _build_composite_provenance(recipe_path, archive_sha)
    report["layers"]["composite_archive_resolution"] = {
        "ok": True,
        "archive_sha256": archive_sha,
        "archive_bytes": archive_bytes,
        "section_count": len(sections),
        "section_names": [s["section_name"] for s in sections],
        "inflate_py_loc": inflate_py_loc,
        "inflate_py_loc_budget": inflate_py_loc_budget,
    }

    # ---- Layer 0+1 STRUCTURAL_EXEMPTION (composite has no single trainer) ----
    report["layers"]["layer_0_compression_pipeline"] = {
        "ok": True,
        "structural_exemption": (
            "composite combines pre-trained sister substrate archives; "
            "no single substrate trainer exists per Catalog #240 single-substrate parity"
        ),
    }
    report["layers"]["layer_1_archive_grammar"] = {
        "ok": True,
        "structural_exemption": (
            f"composite archive grammar = multi-section ZIP ({len(sections)} sections) "
            "per HNeRV parity L3 multi-file justification; pre-built via composite builder"
        ),
        "archive_sha256": archive_sha,
        "sections": sections,
    }

    # ---- Layer 2: builder (composite-aware) ----
    t_bundle = time.time()
    bundle = _emit_composite_bundle_verdict(
        lane_id=lane_id,
        substrate_id=substrate_id,
        archive_path=archive_path,
        archive_sha=archive_sha,
        archive_bytes=archive_bytes,
        submission_dir=submission_dir,
        inflate_sh=inflate_sh,
        inflate_py=inflate_py,
        inflate_py_loc=inflate_py_loc,
        inflate_py_loc_budget=inflate_py_loc_budget,
        inflate_py_loc_waiver_rationale=inflate_py_loc_waiver,
        declared_deps=tuple(sorted(set(args.declared_deps))),
        provenance=provenance,
        elapsed_seconds=time.time() - t_bundle,
    )
    bundle_dict = _as_dict(bundle)
    _write_sidecar(submission_dir, _SIDECAR_BUNDLE, bundle_dict)
    report["layers"]["layer_2_builder"] = {
        "ok": True,
        "submission_dir": str(submission_dir),
        "archive_sha256": bundle.archive_sha256,
        "inflate_py_loc": bundle.inflate_py_loc,
        "sidecar": _SIDECAR_BUNDLE,
        "composite_mode": True,
    }

    # ---- Layer 3: linter (composite-aware) ----
    pr_body_path = None
    for name in ("PR_BODY.md", "PR_BODY_CANONICAL.md", "PR_DESCRIPTION.md"):
        cand = submission_dir / name
        if cand.is_file():
            pr_body_path = cand
            break
    # ALSO check the canonical pr_body_draft path inside .omx/research/ if present.
    if pr_body_path is None:
        pr_body_draft = repo_root / ".omx/research" / (
            f"{lane_id}_pr_body_draft.md"
        )
        if pr_body_draft.is_file():
            pr_body_path = pr_body_draft
    t_lint = time.time()
    lint_verdict = _emit_composite_lint_verdict(
        submission_dir=submission_dir,
        pr_body_path=pr_body_path,
        target_repo=args.target_repo,
        provenance=provenance,
        elapsed_seconds=time.time() - t_lint,
    )
    lint_dict = _as_dict(lint_verdict)
    _write_sidecar(submission_dir, _SIDECAR_LINT, lint_dict)
    report["layers"]["layer_3_linter"] = {
        "ok": bool(lint_verdict.overall_clean),
        "overall_clean": bool(lint_verdict.overall_clean),
        "error_count": int(lint_verdict.error_count),
        "warn_count": int(lint_verdict.warn_count),
        "sidecar": _SIDECAR_LINT,
        "composite_mode": True,
    }
    if not lint_verdict.overall_clean:
        report["lifecycle_verdict"] = "LINT-VIOLATIONS"
        return (EXIT_LINT_VIOLATIONS, report)

    # ---- Layer 4: compliance (composite-aware) ----
    t_comp = time.time()
    json_report_path = submission_dir / "compliance_verdict_report.json"
    compliance_verdict = _emit_composite_compliance_verdict(
        lane_id=lane_id,
        substrate_id=substrate_id,
        archive_path=archive_path,
        archive_sha=archive_sha,
        archive_bytes=archive_bytes,
        submission_dir=submission_dir,
        inflate_sh=inflate_sh,
        inflate_py=inflate_py,
        json_report_path=json_report_path,
        provenance=provenance,
        elapsed_seconds=time.time() - t_comp,
    )
    compliance_dict = _as_dict(compliance_verdict)
    _write_sidecar(submission_dir, _SIDECAR_COMPLIANCE, compliance_dict)
    report["layers"]["layer_4_compliance"] = {
        "ok": bool(compliance_verdict.overall_clean),
        "overall_clean": bool(compliance_verdict.overall_clean),
        "total_checks": compliance_verdict.total_checks,
        "error_count": compliance_verdict.error_count,
        "sidecar": _SIDECAR_COMPLIANCE,
        "composite_mode": True,
        "operator_gated_remaining": [
            c.check_name for c in compliance_verdict.operator_gated_remaining
        ],
    }
    # NOTE: composite-mode does NOT early-return on compliance failure when
    # the only blockers are operator-gated (paired-CUDA pending). The
    # downstream Layer 5 (paired) + Layer 6 (Catalog #370) need to emit
    # canonical sidecars + gate verdict so the operator sees the complete
    # canonical 4-verdict chain state. Non-operator-gated compliance errors
    # DO early-return (those are structural CLI errors the operator can fix).
    non_operator_gated_errors = [
        c for c in compliance_verdict.error_checks if not c.is_operator_gated
    ]
    if non_operator_gated_errors:
        report["lifecycle_verdict"] = "COMPLIANCE-ERRORS"
        report["non_operator_gated_compliance_errors"] = [
            c.check_name for c in non_operator_gated_errors
        ]
        return (EXIT_COMPLIANCE_ERRORS, report)

    # ---- Layer 5: paired auth-eval (composite-aware; plan-only) ----
    t_paired = time.time()
    paired_verdict = _emit_composite_paired_verdict(
        lane_id=lane_id,
        substrate_id=substrate_id,
        archive_sha=archive_sha,
        archive_bytes=archive_bytes,
        submission_dir=submission_dir,
        cuda_gpu=args.cuda_gpu,
        cuda_platform=args.cuda_platform,
        cpu_target=args.cpu_target,
        provenance=provenance,
        elapsed_seconds=time.time() - t_paired,
    )
    paired_dict = _as_dict(paired_verdict)
    _write_sidecar(submission_dir, _SIDECAR_PAIRED, paired_dict)
    paired_pass = (
        paired_verdict.verdict == PairedAuthEvalVerdictKind.PAIRED_PASS.value
    )
    report["layers"]["layer_5_paired_auth_eval"] = {
        "ok": paired_pass,
        "verdict": paired_verdict.verdict,
        "verdict_rationale": paired_verdict.verdict_rationale,
        "paired_env": (
            "composite-recipe plan-only "
            "(paired-CUDA dispatch is operator-attended per Catalog #246)"
        ),
        "sidecar": _SIDECAR_PAIRED,
        "composite_mode": True,
        "directs_to": "tools/operator_authorize.py --recipe <composite-recipe>",
    }
    # Per the composite-recipe semantics: PLAN_ONLY is the canonical pre-paired
    # state; Catalog #370 reads this sidecar and reports paired_axis_missing,
    # which the CLI maps to EXIT_MISSING_PAIRED_AXIS so the operator knows
    # the next step is operator-attended paired-CUDA dispatch.
    if not paired_pass:
        report["lifecycle_verdict"] = "MISSING-PAIRED-AXIS"
        report["operator_next_step"] = (
            "Operator-attended paired-CUDA + Linux x86_64 CPU dispatch via "
            "`tools/operator_authorize.py --recipe <composite-recipe>` "
            "per Catalog #246; flip dispatch_enabled: true in the recipe first."
        )
        # Also emit a composite-recipe-specific verdict sidecar that names
        # the next operator action for Phase 8 STRICT gate ratification.
        _write_sidecar(
            submission_dir,
            _COMPOSITE_RECIPE_SIDECAR,
            {
                "schema_version": "composite_recipe_verdict_v1",
                "lane_id": lane_id,
                "substrate_id": substrate_id,
                "composite_recipe_path": str(recipe_path),
                "composite_archive_sha256": archive_sha,
                "composite_archive_bytes": archive_bytes,
                "lifecycle_verdict": "MISSING-PAIRED-AXIS",
                "operator_next_step": report["operator_next_step"],
                "canonical_provenance": provenance,
                "written_at_utc": _now_utc(),
            },
        )
        # Run Layer 6 STRICT gate so the operator sees the canonical gate
        # verdict even though Layer 5 is plan-only.
        gate_violations = _run_layer_6_gate(submission_dir, repo_root)
        report["layers"]["layer_6_catalog_370_gate"] = {
            "ok": not gate_violations,
            "violations": gate_violations,
            "composite_mode": True,
        }
        return (EXIT_MISSING_PAIRED_AXIS, report)

    # ---- Layer 6: Phase 8 STRICT gate (Catalog #370) ----
    gate_violations = _run_layer_6_gate(submission_dir, repo_root)
    report["layers"]["layer_6_catalog_370_gate"] = {
        "ok": not gate_violations,
        "violations": gate_violations,
        "composite_mode": True,
    }
    if gate_violations:
        report["lifecycle_verdict"] = "CATALOG-370-GATE-VIOLATION"
        return (EXIT_CLI_ERROR, report)

    # ---- PACKET-CLEAN: emit operator-gated gh commands ----
    archive_rel = (
        archive_path.relative_to(repo_root)
        if archive_path.is_relative_to(repo_root)
        else archive_path
    )
    gh_release_cmd = (
        f"gh release create pr_{lane_id} {archive_rel} "
        f"--repo adpena/comma_video_compression_challenge "
        f"--title 'composite submission archive {archive_sha[:12]}'"
    )
    gh_pr_cmd = (
        f"gh pr create --repo {args.target_repo} "
        f"--title '<PR title>' --body-file {submission_dir}/PR_BODY.md"
    )
    report["gh_commands_emitted"] = True
    report["operator_gated_commands"] = {
        "step_1_paired_cuda_ratification": (
            f"tools/operator_authorize.py --recipe {recipe_path.stem} "
            "(operator flips dispatch_enabled: true first)"
        ),
        "step_2_host_archive": gh_release_cmd,
        "step_3_create_pr": gh_pr_cmd,
        "note": (
            "Operator-gated per CLAUDE.md 'Executing actions with care' + "
            "'Public Disclosure Hygiene'. Composite PR submission requires "
            "PAIRED-CUDA RATIFICATION before host+PR. Sole-author Alejandro "
            "Peña <adpena@gmail.com>; ZERO Claude/Anthropic tokens per "
            "user_pr_attribution discipline."
        ),
    }
    report["lifecycle_verdict"] = "OPERATOR-GATED"
    report["total_elapsed_seconds"] = time.time() - t0
    return (EXIT_OPERATOR_GATED, report)


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
    # Composite-recipe extension (Wave N+7 op-routable #5). When
    # --composite-recipe is supplied, --substrate-trainer + --recipe-path are
    # NOT required (composite has no single substrate trainer; recipe path is
    # the composite-recipe argument).
    parser.add_argument(
        "--composite-recipe", type=Path, default=None,
        help=(
            "Composite-recipe path "
            "(.omx/operator_authorize_recipes/substrate_composite_*.yaml). "
            "When supplied, skips Layers 0+1 (no single trainer); consumes the "
            "pre-built composite archive at --archive-path and the "
            "--composite-submission-dir directly. Recipe must declare "
            "composite_components: [...] OR substrate_id: composite_* OR "
            "trainer_path referencing build_composite_*.py."
        ),
    )
    parser.add_argument(
        "--composite-submission-dir", type=Path, default=None,
        help=(
            "Composite submission_dir/ (canonical: "
            "experiments/results/composite_<id>_<utc>/submission/). "
            "REQUIRED with --composite-recipe. Must contain inflate.sh + "
            "inflate.py + (optionally) 0.bin alias + vendored src/tac/."
        ),
    )
    parser.add_argument("--lane-id", default=None, help="Lane registry id (required for single-substrate mode; auto-resolved from recipe for composite mode).")
    parser.add_argument(
        "--substrate-trainer", type=Path, default=None,
        help="Path to experiments/train_substrate_<id>.py (single-substrate mode only).",
    )
    parser.add_argument(
        "--recipe-path", type=Path, default=None,
        help="Path to .omx/operator_authorize_recipes/substrate_<id>_*.yaml (single-substrate mode only).",
    )
    parser.add_argument(
        "--archive-path", type=Path, required=True,
        help="Path to the trainer-emitted archive.zip (single mode) OR the pre-built composite archive.zip (composite mode).",
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
        "--output-dir", type=Path, default=None,
        help="Output submission_dir/ (canonical: submissions/pr<N>_<lane>/). REQUIRED in single-substrate mode; not used in composite mode (composite uses --composite-submission-dir).",
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

    # Composite-recipe mode dispatch (Wave N+7 op-routable #5).
    if args.composite_recipe is not None:
        # Composite mode: --composite-submission-dir required;
        # --substrate-trainer + --recipe-path + --output-dir not used.
        if args.composite_submission_dir is None:
            sys.stderr.write(
                "ERROR: --composite-recipe requires --composite-submission-dir\n"
            )
            return EXIT_CLI_ERROR
        exit_code, report = run_composite_lifecycle(args)
    else:
        # Single-substrate mode: --substrate-trainer + --recipe-path +
        # --lane-id + --output-dir all required.
        missing: list[str] = []
        if args.substrate_trainer is None:
            missing.append("--substrate-trainer")
        if args.recipe_path is None:
            missing.append("--recipe-path")
        if args.lane_id is None:
            missing.append("--lane-id")
        if args.output_dir is None:
            missing.append("--output-dir")
        if missing:
            sys.stderr.write(
                "ERROR: single-substrate mode requires: "
                + ", ".join(missing)
                + "\n  (use --composite-recipe for composite mode)\n"
            )
            return EXIT_CLI_ERROR
        exit_code, report = run_full_lifecycle(args)

    if args.json:
        sys.stdout.write(json.dumps(report, sort_keys=True, indent=2, default=str) + "\n")
    elif not args.quiet:
        sys.stdout.write(_render_human(exit_code, report))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
