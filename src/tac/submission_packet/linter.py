# SPDX-License-Identifier: MIT
"""Layer 3 — canonical pre-flight + lint enforcer for submission packets.

Wrap forbidden-token grep + first-person-plural grep + emdash audit +
inflate.py LOC budget + archive.zip sha/size validation + tone audit
+ Catalog #208 docs-no-local-absolute-paths into a single typed
:class:`LintVerdict` per Phase 1 audit specification memo at
``.omx/research/canonical_submission_pipeline_specification_memo_20260526.md``
Layer 3.

Per the 10th apples-to-apples standing directive + the 11th ORDER-MATTERS
standing directive: this layer is the canonical THIRD Phase 1 spec
consumer; depends on Layer 0
:class:`tac.submission_packet.compression_pipeline.CompressionPipelineResult`,
Layer 1
:class:`tac.submission_packet.archive_grammar.ArchiveGrammarManifest`,
and Layer 2
:class:`tac.submission_packet.builder.SubmissionBundleResult` shapes.

Per the 12th canonicalization × standardization × ease-of-contest-
compliance trinity: ONE canonical helper, ONE return shape, ONE lint
protocol. The bug class this layer extincts: ad-hoc per-submission
forbidden-token grep + tone audit + inflate.py LOC checks scattered
across multiple sister wrappers, each drifting on (i) public-PR hygiene
per CLAUDE.md + the standing directive at
``feedback_forbidden_claude_attribution_in_public_pr_surfaces.md``,
(ii) first-person-plural enforcement per the operator first-person-only
voice directive, (iii) emdash audit per the canonical typography
discipline, (iv) Catalog #208 docs-no-local-absolute-paths,
(v) HNeRV parity L4 ``inflate.py`` ≤200 LOC + ≤2 deps + numpy-portable.

Per Catalog #341 + CLAUDE.md "Apples-to-apples evidence discipline":
this layer is OBSERVABILITY-ONLY by construction. Every emitted
:class:`LintVerdict` carries ``score_claim=False`` + ``promotable=False``
+ ``axis_tag=[predicted]``. Promotion of a lint-clean bundle to a
contest score REQUIRES Phase 6 paired-CUDA + Linux x86_64 CPU empirical
anchor per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
CONTEST-COMPLIANT HARDWARE" non-negotiable.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
L4: the bundled inflate.py LOC budget IS the operator-facing
reviewability metric; this layer enforces it via the same physical-LOC
counter used by the Phase 4 builder so the two helpers stay in lock-step.

Per the standing directive at ``feedback_pr_95_full_deep_research_landed_20260519T192300Z.md``
+ ``feedback_pr_95_quantizr_study_citations_landed_20260519.md``:
medal-class tone is matter-of-fact axis-disclosure without flourishes;
no "Happy to discuss…" closings; no emoji; first-person operator voice
with @-mention attribution chain. This layer enforces the medal-class
tone discipline structurally.

Sister of:
  * :mod:`tac.submission_packet.compression_pipeline` (Phase 2 Layer 0)
  * :mod:`tac.submission_packet.archive_grammar` (Phase 3 Layer 1)
  * :mod:`tac.submission_packet.builder` (Phase 4 Layer 2)
  * ``scripts/pre_submission_compliance_check.py`` (sister Phase 6 Layer 4)
  * ``tac.cathedral_consumers.submission_linter_consumer`` (this layer's
    cathedral autopilot canonical observability sister)
"""
from __future__ import annotations

import datetime
import enum
import hashlib
import os
import re
import socket
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tac.submission_packet.builder import (
    DEFAULT_INFLATE_PY_LOC_BUDGET,
    SubmissionBundleResult,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Module-level constants — canonical schemas + canonical lint vocabulary
# ---------------------------------------------------------------------------

LINTER_SCHEMA_VERSION = "submission_linter_v1_20260526"
"""Pinned schema for :class:`LintVerdict` persistence rows."""

PHASE_5_LAYER_VERSION = "phase_5_submission_linter_canonical_landed_20260526"
"""Operator-readable Phase 5 landing marker per Phase 1 audit spec memo."""

CANONICAL_EQUATION_ID = (
    "submission_linter_canonical_helper_consolidation_savings_v1"
)
"""Canonical equation registered per Phase 1 audit spec memo §13.

FORMALIZATION_PENDING until Phase 10 first-PR-through-canonical-pipeline
regression lands the first empirical anchor of per-submission ad-hoc-
grep-helper divergence collapse (predicted: ad-hoc per-PR linter logic
consolidated to ONE canonical helper).
"""

# Per Catalog #341 routing markers (Tier A observability-only).
PREDICTED_AXIS_TAG = "[predicted]"
EVIDENCE_GRADE = "[predicted; submission-linter-canonical]"

# Per Catalog #287 placeholder rejection.
_PLACEHOLDER_RATIONALES: frozenset[str] = frozenset(
    {"<rationale>", "<reason>", "<rationale_here>", "<reason_here>", ""}
)

# Per CLAUDE.md "Public Disclosure Hygiene" + the standing directive at
# feedback_forbidden_claude_attribution_in_public_pr_surfaces.md.
# Sister of tac.submission_packet.builder._FORBIDDEN_PUBLIC_PR_TOKENS.
FORBIDDEN_PUBLIC_PR_TOKENS: tuple[str, ...] = (
    "Claude",
    "Anthropic",
    "Co-Authored",
    "claude.com",
    "anthropic.com",
)

# Per the operator first-person-only voice directive: PR body must be
# first-person Alejandro Peña ("I built…", "I stacked…") not first-person
# plural ("we built…", "our submission…"). Sister of PR 95/101/102/103
# medal-class precedent.
# Word-boundary regex so "we" matches isolated occurrence but NOT
# substrings like "weave", "swept", "Power", "however", "between".
FIRST_PERSON_PLURAL_PATTERNS: tuple[str, ...] = (
    r"\bwe\b",
    r"\bour\b",
    r"\bus\b",
    r"\bwe're\b",
    r"\bwe've\b",
    r"\bwe'll\b",
    r"\bwe'd\b",
    r"\bWe\b",
    r"\bOur\b",
    r"\bUs\b",
    r"\bWe're\b",
    r"\bWe've\b",
    r"\bWe'll\b",
    r"\bWe'd\b",
)

# U+2014 emdash forbidden per canonical typography discipline. PR 95
# medal-class precedent uses "; " or " - " (ASCII hyphen) instead.
EMDASH_CHARACTER = "—"

# Per CLAUDE.md "Public Disclosure Hygiene" + Catalog #208.
_LOCAL_ABSOLUTE_PATH_PATTERNS: tuple[str, ...] = (
    r"/Users/\w+/",
    r"/home/\w+/",
    r"/private/var/",
    r"C:\\Users\\\w+\\",
)

# Per the PR 95 medal-class study at
# feedback_pr_95_full_deep_research_landed_20260519T192300Z.md +
# feedback_pr_95_quantizr_study_citations_landed_20260519.md.
# Tone violations: marketing flourishes, sign-off bromides, AI-assisted
# tells, emoji, exclamation-mark hype.
TONE_VIOLATION_PATTERNS: tuple[tuple[str, str], ...] = (
    # (regex_pattern, canonical_rule_id)
    (r"(?i)happy to discuss", "tone_signoff_flourish"),
    (r"(?i)let me know if", "tone_signoff_flourish"),
    (r"(?i)feel free to", "tone_signoff_flourish"),
    (r"(?i)please don't hesitate", "tone_signoff_flourish"),
    (r"(?i)i'm excited to share", "tone_marketing_hype"),
    (r"(?i)thrilled to (?:announce|present|share)", "tone_marketing_hype"),
    (r"(?i)groundbreaking", "tone_marketing_hype"),
    (r"(?i)revolutionary", "tone_marketing_hype"),
    (r"(?i)cutting-edge", "tone_marketing_hype"),
    (r"(?i)state[ -]of[ -]the[ -]art", "tone_marketing_hype"),
    (r"(?i)AI-assisted", "tone_ai_tell"),
    (r"(?i)AI[- ]generated", "tone_ai_tell"),
    (r"(?i)submitted (?:with|via|by) Claude", "tone_ai_tell"),
    (r"(?i)implemented with Claude", "tone_ai_tell"),
    (r"(?i)built with Claude", "tone_ai_tell"),
    (r"!{2,}", "tone_excessive_punctuation"),
)

# Emoji patterns: skin-tone-modifier-friendly via broad Unicode classes.
# We use a conservative list of common-in-PR-body emoji ranges; the
# canonical rule is ZERO emoji on public-PR-bound surfaces per PR 95
# medal-class precedent (zero emoji across PR #56/95/100/101/102/103).
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"  # chess + medical + misc
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-A
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed alphanumerics
    "]+",
    flags=re.UNICODE,
)

# Per PR 95 medal-class precedent: attribution chain MUST cite predecessor
# PR authors via @-mention + PR# hyperlink. Sister of the canonical PR101
# GOLD body at 15 lines + PR102 @SajayR/@AaronLeslie138/@EthanYangTW
# chain.
_AT_MENTION_PATTERN = re.compile(r"@[A-Za-z][A-Za-z0-9_-]+")
_PR_REFERENCE_PATTERN = re.compile(r"(?:PR\s*)?#(\d{1,5})")

# Canonical contest axis tags per CLAUDE.md "Apples-to-apples evidence
# discipline". A PR body MUST disclose the score axis (CPU vs CUDA) +
# hardware substrate.
CANONICAL_AXIS_TAGS: frozenset[str] = frozenset(
    {
        "[contest-CPU]",
        "[contest-CUDA]",
        "[macOS-CPU advisory]",
        "[macOS-CPU advisory only]",
        "[MPS-PROXY]",
        "[advisory only]",
        "[macOS-MLX research-signal]",
        "[predicted]",
    }
)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class SubmissionLinterError(RuntimeError):
    """Submission linter orchestration error.

    Sister of :class:`tac.submission_packet.builder.SubmissionBundleError`,
    :class:`tac.submission_packet.archive_grammar.ArchiveGrammarError`,
    :class:`tac.submission_packet.compression_pipeline.CompressionPipelineError`.
    Raised by :func:`lint_submission_bundle` when canonical
    surfaces cannot be parsed or canonical contract violated.
    """


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LintSurface(enum.StrEnum):
    """Canonical lint surfaces per Phase 1 spec memo Layer 3."""

    PR_BODY = "pr_body"
    README = "readme"
    INFLATE_PY = "inflate_py"
    INFLATE_SH = "inflate_sh"
    ARCHIVE_ZIP = "archive_zip"
    COMPLIANCE = "compliance"
    ATTRIBUTION = "attribution"
    REPORT_TXT = "report_txt"
    ARCHIVE_MANIFEST = "archive_manifest"
    DOCS = "docs"


class LintSeverity(enum.StrEnum):
    """Canonical lint severity per Phase 1 spec memo Layer 3.

    ``error`` = blocks PR-submission readiness (zero ERROR required).
    ``warn`` = operator-routable but not blocking.
    ``info`` = observational; surfaces useful context.
    """

    ERROR = "error"
    WARN = "warn"
    INFO = "info"


# ---------------------------------------------------------------------------
# Frozen dataclasses — canonical contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LintFinding:
    """Per-finding lint result (frozen, canonical-provenance-compatible).

    Sister of canonical Provenance per Catalog #323. Every finding carries
    surface + severity + rule + matched text + fix suggestion so downstream
    consumers (cathedral autopilot ranker + operator-facing CLI + Phase 8
    Catalog #362 STRICT preflight gate) can route uniformly.
    """

    surface: str
    """One of :class:`LintSurface` values."""

    severity: str
    """One of :class:`LintSeverity` values."""

    rule: str
    """Canonical rule id (e.g. ``"forbidden_token_claude"`` /
    ``"first_person_plural_we"`` / ``"inflate_py_loc_over_budget"``)."""

    file_path: str
    """Path to the file containing the finding (canonical absolute or
    repo-relative)."""

    line_number: int | None
    """1-indexed line number; None when finding is whole-file scope."""

    matched_text: str | None
    """The matched substring (truncated to canonical 200-char cap when
    long)."""

    fix_suggestion: str
    """Operator-facing canonical fix recommendation (≥4 chars)."""

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or not self.surface:
            raise ValueError("surface must be a non-empty string")
        if self.surface not in {s.value for s in LintSurface}:
            raise ValueError(
                f"surface {self.surface!r} must be one of {[s.value for s in LintSurface]}"
            )
        if not isinstance(self.severity, str) or not self.severity:
            raise ValueError("severity must be a non-empty string")
        if self.severity not in {s.value for s in LintSeverity}:
            raise ValueError(
                f"severity {self.severity!r} must be one of {[s.value for s in LintSeverity]}"
            )
        if not isinstance(self.rule, str) or not self.rule.strip():
            raise ValueError("rule must be a non-empty string")
        if not isinstance(self.file_path, str) or not self.file_path:
            raise ValueError("file_path must be a non-empty string")
        if self.line_number is not None:
            if not isinstance(self.line_number, int) or isinstance(self.line_number, bool):
                raise ValueError("line_number must be int or None")
            if self.line_number < 1:
                raise ValueError("line_number must be >= 1 when not None")
        if self.matched_text is not None and not isinstance(self.matched_text, str):
            raise ValueError("matched_text must be str or None")
        if not isinstance(self.fix_suggestion, str) or len(self.fix_suggestion.strip()) < 4:
            raise ValueError("fix_suggestion must be a string with >=4 chars after strip")

    def as_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "severity": self.severity,
            "rule": self.rule,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "matched_text": self.matched_text,
            "fix_suggestion": self.fix_suggestion,
        }


@dataclass(frozen=True)
class LintVerdict:
    """Canonical Phase 5 Layer 3 lint verdict output.

    Sister of :class:`tac.submission_packet.builder.SubmissionBundleResult`
    + :class:`tac.submission_packet.archive_grammar.ArchiveGrammarManifest`
    at the lint-verdict sub-surface.

    ``overall_clean`` is True iff zero ERROR-severity findings. WARN and
    INFO findings do NOT block PR-submission readiness; they surface for
    operator review.
    """

    schema_version: str
    """Canonical schema version (current: :data:`LINTER_SCHEMA_VERSION`)."""

    overall_clean: bool
    """True iff zero ERROR-severity findings across all surfaces."""

    findings: tuple[LintFinding, ...]
    """Per-finding canonical results (frozen tuple)."""

    surfaces_scanned: tuple[str, ...]
    """Canonical sorted tuple of surface names actually scanned."""

    error_count: int
    """Number of ``error``-severity findings."""

    warn_count: int
    """Number of ``warn``-severity findings."""

    info_count: int
    """Number of ``info``-severity findings."""

    target_repo: str
    """Target upstream repo (canonical default
    ``"commaai/comma_video_compression_challenge"`` per PR-submission
    cascade)."""

    measurement_utc: str
    """ISO-8601 UTC timestamp of lint completion."""

    axis_tag: str
    """Always ``"[predicted]"`` per Catalog #341 + canonical Provenance."""

    score_claim: bool
    """Always ``False`` per CLAUDE.md "Apples-to-apples evidence discipline"."""

    promotable: bool
    """Always ``False`` per Catalog #341 + #192."""

    evidence_grade: str
    """Always ``"[predicted; submission-linter-canonical]"``."""

    canonical_helper_invocation: str
    """``"tac.submission_packet.lint_submission_bundle"`` per Catalog #190."""

    canonical_equation_id: str
    """:data:`CANONICAL_EQUATION_ID` (per Catalog #344)."""

    canonical_equation_status: str
    """``"FORMALIZATION_PENDING"`` until Phase 10 first empirical anchor."""

    elapsed_seconds: float
    """Lint-execution elapsed wall-clock."""

    canonical_provenance: Mapping[str, Any] = field(default_factory=dict)
    """Per Catalog #323 canonical Provenance umbrella."""

    written_at_utc: str = ""
    written_pid: int = 0
    written_host: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != LINTER_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {LINTER_SCHEMA_VERSION!r}; "
                f"got {self.schema_version!r}"
            )
        if not isinstance(self.overall_clean, bool):
            raise ValueError("overall_clean must be bool")
        if not isinstance(self.findings, tuple):
            raise ValueError("findings must be a tuple (frozen)")
        for f in self.findings:
            if not isinstance(f, LintFinding):
                raise ValueError("findings entries must be LintFinding instances")
        if not isinstance(self.surfaces_scanned, tuple):
            raise ValueError("surfaces_scanned must be a tuple (frozen)")
        sorted_scanned = tuple(sorted(self.surfaces_scanned))
        if sorted_scanned != self.surfaces_scanned:
            raise ValueError(
                "surfaces_scanned must be canonical-sorted tuple; "
                f"got {self.surfaces_scanned} canonical {sorted_scanned}"
            )
        for s in self.surfaces_scanned:
            if s not in {ls.value for ls in LintSurface}:
                raise ValueError(
                    f"surface {s!r} not in canonical {[ls.value for ls in LintSurface]}"
                )
        expected_error = sum(
            1 for f in self.findings if f.severity == LintSeverity.ERROR.value
        )
        expected_warn = sum(
            1 for f in self.findings if f.severity == LintSeverity.WARN.value
        )
        expected_info = sum(
            1 for f in self.findings if f.severity == LintSeverity.INFO.value
        )
        if self.error_count != expected_error:
            raise ValueError(
                f"error_count {self.error_count} inconsistent with findings "
                f"(expected {expected_error})"
            )
        if self.warn_count != expected_warn:
            raise ValueError(
                f"warn_count {self.warn_count} inconsistent with findings "
                f"(expected {expected_warn})"
            )
        if self.info_count != expected_info:
            raise ValueError(
                f"info_count {self.info_count} inconsistent with findings "
                f"(expected {expected_info})"
            )
        expected_clean = self.error_count == 0
        if self.overall_clean != expected_clean:
            raise ValueError(
                f"overall_clean {self.overall_clean} inconsistent with "
                f"error_count={self.error_count}"
            )
        if not isinstance(self.target_repo, str) or "/" not in self.target_repo:
            raise ValueError(
                f"target_repo must be a 'owner/repo' string; got {self.target_repo!r}"
            )
        if not self.measurement_utc:
            raise ValueError("measurement_utc must be non-empty")
        if self.axis_tag != PREDICTED_AXIS_TAG:
            raise ValueError(f"axis_tag must equal {PREDICTED_AXIS_TAG!r}")
        if self.score_claim is not False:
            raise ValueError("score_claim must be False per Catalog #341")
        if self.promotable is not False:
            raise ValueError("promotable must be False per Catalog #341")
        if not self.evidence_grade.startswith("[predicted;"):
            raise ValueError(
                "evidence_grade must start with '[predicted;' per Catalog #287/#323"
            )
        if self.canonical_equation_id != CANONICAL_EQUATION_ID:
            raise ValueError(
                f"canonical_equation_id must equal {CANONICAL_EQUATION_ID!r}; "
                f"got {self.canonical_equation_id!r}"
            )
        if self.canonical_equation_status not in {"FORMALIZATION_PENDING", "REGISTERED"}:
            raise ValueError(
                "canonical_equation_status must be 'FORMALIZATION_PENDING' or 'REGISTERED'"
            )
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        if not isinstance(self.canonical_provenance, Mapping):
            raise ValueError("canonical_provenance must be a Mapping per Catalog #323")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "overall_clean": bool(self.overall_clean),
            "findings": [f.as_dict() for f in self.findings],
            "surfaces_scanned": list(self.surfaces_scanned),
            "error_count": int(self.error_count),
            "warn_count": int(self.warn_count),
            "info_count": int(self.info_count),
            "target_repo": self.target_repo,
            "measurement_utc": self.measurement_utc,
            "axis_tag": self.axis_tag,
            "score_claim": bool(self.score_claim),
            "promotable": bool(self.promotable),
            "evidence_grade": self.evidence_grade,
            "canonical_helper_invocation": self.canonical_helper_invocation,
            "canonical_equation_id": self.canonical_equation_id,
            "canonical_equation_status": self.canonical_equation_status,
            "elapsed_seconds": float(self.elapsed_seconds),
            "canonical_provenance": dict(self.canonical_provenance),
            "written_at_utc": self.written_at_utc,
            "written_pid": int(self.written_pid),
            "written_host": self.written_host,
        }


# ---------------------------------------------------------------------------
# Core API helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Canonical UTC timestamp (ISO-8601 with tz)."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def _truncate(text: str, cap: int = 200) -> str:
    """Cap matched text to keep findings compact."""
    if len(text) <= cap:
        return text
    return text[: cap - 1] + "…"


def _line_at(body: str, char_offset: int) -> int:
    """1-indexed line number for a character offset."""
    return body.count("\n", 0, char_offset) + 1


def derive_linter_provenance(
    *,
    target_repo: str,
    archive_sha256: str | None,
    measurement_utc: str,
) -> dict[str, Any]:
    """Build the canonical Provenance dict for a lint verdict.

    Per Catalog #323 canonical Provenance umbrella: every persisted row
    carries (axis_tag + evidence_grade + score_claim + promotable +
    canonical_helper_invocation + captured_at_utc).
    """
    payload: dict[str, Any] = {
        "axis_tag": PREDICTED_AXIS_TAG,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "promotable": False,
        "canonical_helper_invocation": (
            "tac.submission_packet.lint_submission_bundle"
        ),
        "captured_at_utc": measurement_utc,
        "target_repo": target_repo,
        "canonical_equation_id": CANONICAL_EQUATION_ID,
        "canonical_equation_status": "FORMALIZATION_PENDING",
        "schema_version": LINTER_SCHEMA_VERSION,
    }
    if archive_sha256 is not None:
        payload["archive_sha256"] = archive_sha256
    return payload


# ---------------------------------------------------------------------------
# Per-surface lint helpers
# ---------------------------------------------------------------------------


def lint_pr_body(
    body_text: str,
    *,
    target_repo: str = "commaai/comma_video_compression_challenge",
    file_path: str = "PR_BODY.md",
) -> tuple[LintFinding, ...]:
    """Lint a PR body string for canonical surface invariants.

    Surfaces checked (per Phase 1 spec memo Layer 3):
      * Forbidden public-PR tokens per CLAUDE.md "Public Disclosure
        Hygiene" + the standing directive at
        ``feedback_forbidden_claude_attribution_in_public_pr_surfaces.md``.
      * First-person-plural ("we"/"our"/"us") per the operator
        first-person-only voice directive.
      * Emdash (U+2014) per canonical typography discipline.
      * Tone violations per PR 95 medal-class study.
      * Emoji per zero-emoji-on-public-PR convention.
      * Attribution chain shape (≥1 @-mention + ≥1 PR# reference) per
        PR 95/101/102/103 medal-class precedent.
      * Catalog #208 local-absolute-paths.
      * Canonical axis tag presence (info-only nudge when score is
        cited without an axis tag).
    """
    findings: list[LintFinding] = []

    for token in FORBIDDEN_PUBLIC_PR_TOKENS:
        for m in re.finditer(re.escape(token), body_text):
            findings.append(
                LintFinding(
                    surface=LintSurface.PR_BODY.value,
                    severity=LintSeverity.ERROR.value,
                    rule=f"forbidden_token_{token.lower().replace('.', '_').replace('-', '_')}",
                    file_path=file_path,
                    line_number=_line_at(body_text, m.start()),
                    matched_text=_truncate(m.group(0)),
                    fix_suggestion=(
                        f"Remove forbidden token {token!r}. "
                        "PR body must be sole-author Alejandro Peña voice per "
                        "feedback_forbidden_claude_attribution_in_public_pr_surfaces.md."
                    ),
                )
            )

    for pattern in FIRST_PERSON_PLURAL_PATTERNS:
        for m in re.finditer(pattern, body_text):
            findings.append(
                LintFinding(
                    surface=LintSurface.PR_BODY.value,
                    severity=LintSeverity.ERROR.value,
                    rule=f"first_person_plural_{m.group(0).lower().replace(chr(39), '')}",
                    file_path=file_path,
                    line_number=_line_at(body_text, m.start()),
                    matched_text=_truncate(m.group(0)),
                    fix_suggestion=(
                        "Replace first-person-plural with first-person-singular ('I built…', "
                        "'I stacked…') per operator first-person-only voice directive."
                    ),
                )
            )

    for m in re.finditer(re.escape(EMDASH_CHARACTER), body_text):
        findings.append(
            LintFinding(
                surface=LintSurface.PR_BODY.value,
                severity=LintSeverity.ERROR.value,
                rule="emdash_u2014",
                file_path=file_path,
                line_number=_line_at(body_text, m.start()),
                matched_text=_truncate(m.group(0)),
                fix_suggestion=(
                    "Replace U+2014 emdash with '; ' or ' - ' (ASCII hyphen) per "
                    "PR 95 medal-class typography convention."
                ),
            )
        )

    for pattern, rule_id in TONE_VIOLATION_PATTERNS:
        for m in re.finditer(pattern, body_text):
            findings.append(
                LintFinding(
                    surface=LintSurface.PR_BODY.value,
                    severity=LintSeverity.ERROR.value,
                    rule=rule_id,
                    file_path=file_path,
                    line_number=_line_at(body_text, m.start()),
                    matched_text=_truncate(m.group(0)),
                    fix_suggestion=(
                        "Drop tone flourish; matter-of-fact axis-disclosure per "
                        "PR 95 medal-class precedent (PR101 GOLD body is 15 lines)."
                    ),
                )
            )

    for m in _EMOJI_PATTERN.finditer(body_text):
        findings.append(
            LintFinding(
                surface=LintSurface.PR_BODY.value,
                severity=LintSeverity.ERROR.value,
                rule="emoji_forbidden_on_public_pr_surface",
                file_path=file_path,
                line_number=_line_at(body_text, m.start()),
                matched_text=_truncate(m.group(0)),
                fix_suggestion=(
                    "Remove emoji; zero-emoji convention per PR #56/95/100/101/102/103 "
                    "medal-class precedent."
                ),
            )
        )

    for pattern in _LOCAL_ABSOLUTE_PATH_PATTERNS:
        for m in re.finditer(pattern, body_text):
            findings.append(
                LintFinding(
                    surface=LintSurface.PR_BODY.value,
                    severity=LintSeverity.ERROR.value,
                    rule="catalog_208_local_absolute_path",
                    file_path=file_path,
                    line_number=_line_at(body_text, m.start()),
                    matched_text=_truncate(m.group(0)),
                    fix_suggestion=(
                        "Replace local absolute path with repo-relative path or "
                        "<placeholder> per Catalog #208 + CLAUDE.md Public Disclosure Hygiene."
                    ),
                )
            )

    at_mentions = _AT_MENTION_PATTERN.findall(body_text)
    pr_refs = _PR_REFERENCE_PATTERN.findall(body_text)
    if not at_mentions:
        findings.append(
            LintFinding(
                surface=LintSurface.ATTRIBUTION.value,
                severity=LintSeverity.WARN.value,
                rule="attribution_no_at_mention",
                file_path=file_path,
                line_number=None,
                matched_text=None,
                fix_suggestion=(
                    "Add at least one @-mention citing predecessor PR author per "
                    "PR 95/101/102/103 medal-class attribution chain precedent."
                ),
            )
        )
    if not pr_refs:
        findings.append(
            LintFinding(
                surface=LintSurface.ATTRIBUTION.value,
                severity=LintSeverity.WARN.value,
                rule="attribution_no_pr_reference",
                file_path=file_path,
                line_number=None,
                matched_text=None,
                fix_suggestion=(
                    "Add at least one PR# reference (e.g. '#101') for the predecessor "
                    "attribution chain per PR 95 medal-class precedent."
                ),
            )
        )

    score_like = re.search(r"\b0\.\d{3,5}\b", body_text)
    if score_like is not None:
        axis_present = any(tag in body_text for tag in CANONICAL_AXIS_TAGS)
        if not axis_present:
            findings.append(
                LintFinding(
                    surface=LintSurface.PR_BODY.value,
                    severity=LintSeverity.WARN.value,
                    rule="missing_axis_tag_on_score_citation",
                    file_path=file_path,
                    line_number=_line_at(body_text, score_like.start()),
                    matched_text=_truncate(score_like.group(0)),
                    fix_suggestion=(
                        "Append canonical axis tag (e.g. '[contest-CPU]' or '[contest-CUDA]') "
                        "per CLAUDE.md 'Apples-to-apples evidence discipline'."
                    ),
                )
            )

    return tuple(findings)


def lint_inflate_py(
    path: Path,
    *,
    loc_budget: int = DEFAULT_INFLATE_PY_LOC_BUDGET,
    waiver_rationale: str | None = None,
) -> tuple[LintFinding, ...]:
    """Lint a bundled inflate.py against HNeRV parity L4 invariants.

    Surfaces checked:
      * LOC ≤ ``loc_budget`` (canonical default 200 per HNeRV parity L4)
        OR substantive waiver rationale per Catalog #287.
      * Catalog #205 canonical ``select_inflate_device`` routing
        (canonical helper invocation OR inline-with-waiver pattern).
      * Catalog #295 PYTHONPATH self-containment (no bare ``from tac.*``
        without explicit ``# SUBMISSION_PYTHONPATH_SHIM_OK`` waiver or
        vendored sister package).
    """
    findings: list[LintFinding] = []
    if not path.exists():
        findings.append(
            LintFinding(
                surface=LintSurface.INFLATE_PY.value,
                severity=LintSeverity.ERROR.value,
                rule="inflate_py_missing",
                file_path=str(path),
                line_number=None,
                matched_text=None,
                fix_suggestion=(
                    "inflate.py is required by HNeRV parity L4; Phase 4 builder must "
                    "emit it via tac.submission_packet.build_submission_bundle."
                ),
            )
        )
        return tuple(findings)

    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        findings.append(
            LintFinding(
                surface=LintSurface.INFLATE_PY.value,
                severity=LintSeverity.ERROR.value,
                rule="inflate_py_decode_error",
                file_path=str(path),
                line_number=None,
                matched_text=_truncate(str(exc)),
                fix_suggestion=(
                    "inflate.py must be UTF-8 encoded; re-emit via Phase 4 builder."
                ),
            )
        )
        return tuple(findings)

    loc = len(source.splitlines()) if source else 0
    if loc > loc_budget:
        stripped = waiver_rationale.strip() if waiver_rationale else ""
        if stripped in _PLACEHOLDER_RATIONALES or len(stripped) < 4:
            findings.append(
                LintFinding(
                    surface=LintSurface.INFLATE_PY.value,
                    severity=LintSeverity.ERROR.value,
                    rule="inflate_py_loc_over_budget",
                    file_path=str(path),
                    line_number=None,
                    matched_text=f"loc={loc} budget={loc_budget}",
                    fix_suggestion=(
                        f"inflate.py is {loc} LOC > budget {loc_budget} per HNeRV "
                        "parity L4. Reduce LOC OR pass substantive waiver_rationale "
                        "(>=4 chars, non-placeholder per Catalog #287)."
                    ),
                )
            )
        else:
            findings.append(
                LintFinding(
                    surface=LintSurface.INFLATE_PY.value,
                    severity=LintSeverity.WARN.value,
                    rule="inflate_py_loc_over_budget_with_waiver",
                    file_path=str(path),
                    line_number=None,
                    matched_text=f"loc={loc} budget={loc_budget}",
                    fix_suggestion=(
                        f"inflate.py is {loc} LOC > budget {loc_budget}; waiver supplied. "
                        "Operator-routable to reduce LOC for reviewability."
                    ),
                )
            )

    has_canonical_helper = "select_inflate_device" in source
    has_inline_waiver = "INLINE_DEVICE_FORK_OK" in source
    if not has_canonical_helper and not has_inline_waiver:
        findings.append(
            LintFinding(
                surface=LintSurface.INFLATE_PY.value,
                severity=LintSeverity.WARN.value,
                rule="catalog_205_no_canonical_helper_routing",
                file_path=str(path),
                line_number=None,
                matched_text=None,
                fix_suggestion=(
                    "inflate.py should route through canonical select_inflate_device "
                    "helper OR carry '# INLINE_DEVICE_FORK_OK:<rationale>' waiver per "
                    "Catalog #205."
                ),
            )
        )

    for m in re.finditer(r"^\s*(?:from|import)\s+tac\.", source, flags=re.MULTILINE):
        # Per Catalog #295 the waiver discipline applies here too; we look
        # for the canonical SUBMISSION_PYTHONPATH_SHIM_OK token nearby OR
        # a vendored sister package alongside.
        line_no = _line_at(source, m.start())
        line_text = source.splitlines()[line_no - 1] if line_no <= len(source.splitlines()) else ""
        if "SUBMISSION_PYTHONPATH_SHIM_OK" in line_text:
            continue
        vendor_tac_init = path.parent / "src" / "tac" / "__init__.py"
        vendor_tac_alongside = path.parent / "tac" / "__init__.py"
        if vendor_tac_init.exists() or vendor_tac_alongside.exists():
            continue
        findings.append(
            LintFinding(
                surface=LintSurface.INFLATE_PY.value,
                severity=LintSeverity.WARN.value,
                rule="catalog_295_bare_tac_import_no_vendor",
                file_path=str(path),
                line_number=line_no,
                matched_text=_truncate(line_text.strip()),
                fix_suggestion=(
                    "Vendor sister tac.* package alongside inflate.py OR carry same-line "
                    "'# SUBMISSION_PYTHONPATH_SHIM_OK:<rationale>' waiver per Catalog #295."
                ),
            )
        )

    return tuple(findings)


def lint_archive_zip(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> tuple[LintFinding, ...]:
    """Lint an archive.zip against expected sha + size invariants.

    Per CLAUDE.md "Bit-level deconstruction and entropy discipline":
    the archive bytes ARE the contest-charged surface; sha + size
    mismatch invalidates every downstream consumer.
    """
    findings: list[LintFinding] = []
    if not path.exists():
        findings.append(
            LintFinding(
                surface=LintSurface.ARCHIVE_ZIP.value,
                severity=LintSeverity.ERROR.value,
                rule="archive_zip_missing",
                file_path=str(path),
                line_number=None,
                matched_text=None,
                fix_suggestion=(
                    "archive.zip required at the bundled submission_dir; emit via "
                    "tac.submission_packet.build_submission_bundle."
                ),
            )
        )
        return tuple(findings)

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    actual_sha = h.hexdigest()
    actual_size = path.stat().st_size

    if actual_sha != expected_sha256:
        findings.append(
            LintFinding(
                surface=LintSurface.ARCHIVE_ZIP.value,
                severity=LintSeverity.ERROR.value,
                rule="archive_sha256_mismatch",
                file_path=str(path),
                line_number=None,
                matched_text=f"actual={actual_sha[:12]}... expected={expected_sha256[:12]}...",
                fix_suggestion=(
                    "Recompute archive_sha256 OR re-emit submission bundle so "
                    "tac.submission_packet.SubmissionBundleResult.archive_sha256 matches."
                ),
            )
        )

    if actual_size != expected_size_bytes:
        findings.append(
            LintFinding(
                surface=LintSurface.ARCHIVE_ZIP.value,
                severity=LintSeverity.ERROR.value,
                rule="archive_size_mismatch",
                file_path=str(path),
                line_number=None,
                matched_text=f"actual={actual_size} expected={expected_size_bytes}",
                fix_suggestion=(
                    "Recompute archive_bytes OR re-emit submission bundle so "
                    "SubmissionBundleResult.archive_bytes matches."
                ),
            )
        )

    return tuple(findings)


def lint_compliance_placeholder(
    submission_dir: Path,
) -> tuple[LintFinding, ...]:
    """Placeholder lint sister to Phase 6 compliance enforcer.

    Per Phase 1 spec memo §3 the Phase 6 sister landing wraps
    ``scripts/pre_submission_compliance_check.py`` in a typed
    :class:`ComplianceVerdict`. This Layer 3 placeholder surfaces the
    sister discipline boundary so the operator-facing CLI can route
    consistently between lint + compliance + paired_auth_eval.

    Per the operator prompt: this layer is SISTER-DISJOINT from Phase 6.
    The placeholder emits an info-level finding when a compliance JSON
    sidecar exists; it does NOT invoke the compliance subprocess
    directly (that work belongs to the parallel Phase 6 spawn).
    """
    findings: list[LintFinding] = []
    compliance_dir = submission_dir.parent / "reports" / "pr_pre_submission"
    if compliance_dir.exists():
        json_reports = sorted(compliance_dir.glob("compliance_report_*.json"))
        if json_reports:
            findings.append(
                LintFinding(
                    surface=LintSurface.COMPLIANCE.value,
                    severity=LintSeverity.INFO.value,
                    rule="compliance_sidecar_present",
                    file_path=str(json_reports[-1]),
                    line_number=None,
                    matched_text=_truncate(json_reports[-1].name),
                    fix_suggestion=(
                        "Phase 6 compliance enforcer (sister-disjoint spawn) is the "
                        "canonical consumer of this artifact; lint is observational only."
                    ),
                )
            )
    return tuple(findings)


def lint_tone(
    body_text: str,
    *,
    file_path: str = "PR_BODY.md",
) -> tuple[LintFinding, ...]:
    """Lint a PR body string for medal-class tone per PR 95 precedent.

    This is a focused sub-helper that ONLY runs the tone audit (subset
    of :func:`lint_pr_body`). Useful when the operator wants a tone-only
    pass without re-running the full PR-body suite.

    Surfaces:
      * Marketing flourishes / sign-off bromides / AI-assisted tells.
      * Emoji on public-PR surface.
      * Excessive punctuation ("!!", "!!!").
    """
    findings: list[LintFinding] = []
    for pattern, rule_id in TONE_VIOLATION_PATTERNS:
        for m in re.finditer(pattern, body_text):
            findings.append(
                LintFinding(
                    surface=LintSurface.PR_BODY.value,
                    severity=LintSeverity.ERROR.value,
                    rule=rule_id,
                    file_path=file_path,
                    line_number=_line_at(body_text, m.start()),
                    matched_text=_truncate(m.group(0)),
                    fix_suggestion=(
                        "Drop tone flourish; matter-of-fact axis-disclosure per "
                        "PR 95 medal-class precedent (PR101 GOLD body is 15 lines)."
                    ),
                )
            )
    for m in _EMOJI_PATTERN.finditer(body_text):
        findings.append(
            LintFinding(
                surface=LintSurface.PR_BODY.value,
                severity=LintSeverity.ERROR.value,
                rule="emoji_forbidden_on_public_pr_surface",
                file_path=file_path,
                line_number=_line_at(body_text, m.start()),
                matched_text=_truncate(m.group(0)),
                fix_suggestion=(
                    "Remove emoji; zero-emoji convention per PR #56/95/100/101/102/103."
                ),
            )
        )
    return tuple(findings)


def lint_readme(
    path: Path,
) -> tuple[LintFinding, ...]:
    """Lint a submission_dir README.md against public-PR-bound surfaces.

    Sister of :func:`lint_pr_body` scoped to the README inside the
    submission_dir. Per the standing directive at
    ``feedback_forbidden_claude_attribution_in_public_pr_surfaces.md``,
    the README MUST NOT mention Claude/Anthropic.
    """
    findings: list[LintFinding] = []
    if not path.exists():
        findings.append(
            LintFinding(
                surface=LintSurface.README.value,
                severity=LintSeverity.ERROR.value,
                rule="readme_missing",
                file_path=str(path),
                line_number=None,
                matched_text=None,
                fix_suggestion=(
                    "README.md required at submission_dir; emit via Phase 4 builder."
                ),
            )
        )
        return tuple(findings)

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        findings.append(
            LintFinding(
                surface=LintSurface.README.value,
                severity=LintSeverity.ERROR.value,
                rule="readme_decode_error",
                file_path=str(path),
                line_number=None,
                matched_text=_truncate(str(exc)),
                fix_suggestion="README.md must be UTF-8 encoded.",
            )
        )
        return tuple(findings)

    for token in FORBIDDEN_PUBLIC_PR_TOKENS:
        for m in re.finditer(re.escape(token), text):
            findings.append(
                LintFinding(
                    surface=LintSurface.README.value,
                    severity=LintSeverity.ERROR.value,
                    rule=f"forbidden_token_{token.lower().replace('.', '_').replace('-', '_')}",
                    file_path=str(path),
                    line_number=_line_at(text, m.start()),
                    matched_text=_truncate(m.group(0)),
                    fix_suggestion=(
                        f"Remove forbidden token {token!r} from README per "
                        "feedback_forbidden_claude_attribution_in_public_pr_surfaces.md."
                    ),
                )
            )

    for pattern in _LOCAL_ABSOLUTE_PATH_PATTERNS:
        for m in re.finditer(pattern, text):
            findings.append(
                LintFinding(
                    surface=LintSurface.README.value,
                    severity=LintSeverity.ERROR.value,
                    rule="catalog_208_local_absolute_path",
                    file_path=str(path),
                    line_number=_line_at(text, m.start()),
                    matched_text=_truncate(m.group(0)),
                    fix_suggestion=(
                        "Replace local absolute path per Catalog #208 + CLAUDE.md "
                        "Public Disclosure Hygiene."
                    ),
                )
            )

    return tuple(findings)


# ---------------------------------------------------------------------------
# Main canonical entry point
# ---------------------------------------------------------------------------


def lint_submission_bundle(
    submission_bundle_result: SubmissionBundleResult,
    *,
    target_repo: str = "commaai/comma_video_compression_challenge",
    pr_body_path: Path | None = None,
    pr_body_text: str | None = None,
    inflate_py_loc_waiver_rationale: str | None = None,
) -> LintVerdict:
    """Canonical Phase 5 Layer 3 entry point — lint a SubmissionBundleResult.

    Runs the full canonical lint suite over a Phase 4 bundle per the
    11th ORDER-MATTERS directive lint surfaces ordering:

      1. PR body (when supplied)
      2. inflate.py (always; HNeRV parity L4)
      3. archive.zip (sha + size)
      4. compliance placeholder (sister-disjoint Phase 6 surface)
      5. README.md (when present)

    Args:
        submission_bundle_result: Phase 4 builder output (canonical
            data carrier; provides archive_sha256 + bytes + paths).
        target_repo: upstream PR target (canonical default
            ``"commaai/comma_video_compression_challenge"``).
        pr_body_path: optional path to a PR body markdown file; when
            supplied the body content is loaded from disk.
        pr_body_text: optional raw PR body text; supersedes
            ``pr_body_path`` when both supplied.
        inflate_py_loc_waiver_rationale: substantive waiver for
            inflate.py over-budget LOC (≥4 chars per Catalog #287).

    Returns:
        :class:`LintVerdict` with canonical Provenance per Catalog #323.

    Raises:
        SubmissionLinterError: when canonical surfaces cannot be parsed
            (missing builder paths, malformed bundle).
    """
    started = _utc_now_iso()
    started_perf = datetime.datetime.now(datetime.UTC)

    if not isinstance(submission_bundle_result, SubmissionBundleResult):
        raise SubmissionLinterError(
            "submission_bundle_result must be a "
            "tac.submission_packet.builder.SubmissionBundleResult"
        )
    if not isinstance(target_repo, str) or "/" not in target_repo:
        raise SubmissionLinterError(
            f"target_repo must be a 'owner/repo' string; got {target_repo!r}"
        )

    findings: list[LintFinding] = []
    surfaces_scanned: list[str] = []

    # Surface 1: PR body (per 11th ORDER directive — body is canonical first).
    resolved_pr_body: str | None = None
    pr_body_source_path: str = "PR_BODY.md"
    if pr_body_text is not None:
        resolved_pr_body = pr_body_text
    elif pr_body_path is not None:
        if not pr_body_path.exists():
            raise SubmissionLinterError(
                f"pr_body_path {pr_body_path} does not exist"
            )
        try:
            resolved_pr_body = pr_body_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SubmissionLinterError(
                f"pr_body_path {pr_body_path} is not UTF-8: {exc}"
            ) from exc
        pr_body_source_path = str(pr_body_path)
    if resolved_pr_body is not None:
        findings.extend(
            lint_pr_body(
                resolved_pr_body,
                target_repo=target_repo,
                file_path=pr_body_source_path,
            )
        )
        surfaces_scanned.append(LintSurface.PR_BODY.value)
        surfaces_scanned.append(LintSurface.ATTRIBUTION.value)

    # Surface 2: inflate.py (HNeRV parity L4 invariants).
    inflate_py_path = Path(submission_bundle_result.inflate_py_path)
    inflate_findings = lint_inflate_py(
        inflate_py_path,
        loc_budget=submission_bundle_result.inflate_py_loc_budget,
        waiver_rationale=(
            inflate_py_loc_waiver_rationale
            or submission_bundle_result.inflate_py_loc_waiver_rationale
        ),
    )
    findings.extend(inflate_findings)
    surfaces_scanned.append(LintSurface.INFLATE_PY.value)

    # Surface 3: archive.zip (sha + size).
    submission_dir = Path(submission_bundle_result.submission_dir)
    archive_zip_path = submission_dir / "archive.zip"
    archive_findings = lint_archive_zip(
        archive_zip_path,
        expected_sha256=submission_bundle_result.archive_sha256,
        expected_size_bytes=submission_bundle_result.archive_bytes,
    )
    findings.extend(archive_findings)
    surfaces_scanned.append(LintSurface.ARCHIVE_ZIP.value)

    # Surface 4: compliance placeholder (sister-disjoint Phase 6 surface).
    compliance_findings = lint_compliance_placeholder(submission_dir)
    findings.extend(compliance_findings)
    surfaces_scanned.append(LintSurface.COMPLIANCE.value)

    # Surface 5: README.md (when present at submission_dir).
    readme_path = Path(submission_bundle_result.readme_md_path)
    if readme_path.exists():
        readme_findings = lint_readme(readme_path)
        findings.extend(readme_findings)
        surfaces_scanned.append(LintSurface.README.value)

    findings_tuple = tuple(findings)
    error_count = sum(
        1 for f in findings_tuple if f.severity == LintSeverity.ERROR.value
    )
    warn_count = sum(
        1 for f in findings_tuple if f.severity == LintSeverity.WARN.value
    )
    info_count = sum(
        1 for f in findings_tuple if f.severity == LintSeverity.INFO.value
    )

    measurement_utc = _utc_now_iso()
    canonical_provenance = derive_linter_provenance(
        target_repo=target_repo,
        archive_sha256=submission_bundle_result.archive_sha256,
        measurement_utc=measurement_utc,
    )

    elapsed = (datetime.datetime.now(datetime.UTC) - started_perf).total_seconds()

    return LintVerdict(
        schema_version=LINTER_SCHEMA_VERSION,
        overall_clean=(error_count == 0),
        findings=findings_tuple,
        surfaces_scanned=tuple(sorted(set(surfaces_scanned))),
        error_count=int(error_count),
        warn_count=int(warn_count),
        info_count=int(info_count),
        target_repo=target_repo,
        measurement_utc=measurement_utc,
        axis_tag=PREDICTED_AXIS_TAG,
        score_claim=False,
        promotable=False,
        evidence_grade=EVIDENCE_GRADE,
        canonical_helper_invocation="tac.submission_packet.lint_submission_bundle",
        canonical_equation_id=CANONICAL_EQUATION_ID,
        canonical_equation_status="FORMALIZATION_PENDING",
        elapsed_seconds=float(elapsed),
        canonical_provenance=canonical_provenance,
        written_at_utc=measurement_utc,
        written_pid=os.getpid(),
        written_host=socket.gethostname(),
    )
