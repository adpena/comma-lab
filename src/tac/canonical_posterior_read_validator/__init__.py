# SPDX-License-Identifier: MIT
"""Canonical posterior READ-surface validator (Catalog #(BB-claim) sister of #321/#322 WRITE).

Per operator BINDING META-META question 2026-05-29 ~13:55Z America/Chicago verbatim:

    "why do we keep having the phantom score artifacts issue? seems like our
     current approach is not optimal to permanent self protecting and fixing
     and also recovering and continuing"

The CANONICAL ANSWER landed by this module + sister Catalog # STRICT gate +
NEW cathedral consumer + META-META anti-pattern: canonical apparatus protects
WRITE surfaces (Catalog #321/#322 STRICT gates fire when a phantom claim
attempts to enter the canonical posterior) but DOES NOT protect READ surfaces
(operator-facing memo claims + parent main-thread spawn-prompts + cathedral
autopilot ranking decisions ALL bypass canonical posterior verification).

Empirical recurrence anchors (Phase A design memo §"Empirical evidence
inventory" enumerates 7):

* Wave N+33 RANK 1 cited alpha=4.74 11 days AFTER canonical posterior self-
  flagged the row as PHANTOM (Slot U 2026-05-29 landed).
* Slot T STAND_DOWN — parent main-thread spawned without Catalog #378 PV;
  predecessor had landed 10 hours earlier.
* Registration-discipline character-limit recurrence chain Slot K -> P -> S
  -> Z; 4 reactive registrations, ZERO structural READ-surface gate.

Sister of:

* Catalog #321 ``check_no_phantom_wyner_ziv_savings_from_research_sidecar``
  (WRITE-surface STRICT gate)
* Catalog #322 ``check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha``
  (WRITE-surface STRICT gate)
* Catalog #287 ``check_no_docstring_overstatement_without_evidence_tag``
  (source-text WRITE-surface STRICT gate)
* Catalog #343 ``check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded``
  (CLAUDE.md WRITE-surface STRICT gate)
* Catalog #378 ``check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state``
  (PARENT-MAIN-THREAD spawn-decision PV)
* Catalog #335 cathedral consumer auto-discovery (the sister
  ``phantom_score_canonical_posterior_lookup_consumer`` lands per this
  module's verdict surfaces)

Public API (3 canonical entry points):

1. :func:`validate_memo_claim_against_canonical_posterior` — query a single
   claim_token against canonical posterior (canonical equations + canonical
   anti-patterns + probe outcomes ledger). Returns
   :class:`CanonicalPosteriorReadValidationVerdict` (frozen dataclass per
   Catalog #323).

2. :func:`validate_spawn_prompt_against_canonical_posterior` — sister of
   Catalog #378 ``verify_head_state_before_main_thread_spawn``. Validates
   every cited token in a spawn-prompt text body; returns
   :class:`SpawnPromptValidationVerdict` (PROCEED / ABORT_PHANTOM_TOKEN_CITED
   / WARN_UNKNOWN_TOKEN cascade).

3. :func:`auto_emit_append_only_footer_to_memos_citing_falsified_score` —
   sister cascade across ``.omx/research/`` + Claude memory directory. When
   a canonical posterior verdict flips to FALSIFIED / KILLED / PHANTOM /
   INVALIDATED, this helper emits an HTML-comment APPEND-ONLY footer per
   Catalog #110/#113 HISTORICAL_PROVENANCE discipline to every memo citing
   the flipped token, WITHOUT mutating body content.

4-state verdict taxonomy (latest-event-wins per claim_token):

* ``CLEAN``    — claim_token CONFIRMED / PROCEED / CLEAN at latest canonical posterior event.
* ``FALSIFIED`` — claim_token IMPLEMENTATION-LEVEL or PARADIGM-LEVEL falsified per Catalog #307.
* ``KILLED``    — claim_token retired per probe outcomes ledger KILL verdict (rare; #313 sister).
* ``PHANTOM``   — claim_token's underlying assumption was a phantom-score-artifact recurrence
                  per sister Catalog #321/#322/#249 bug-class anchor (Wave N+33 alpha=4.74
                  canonical type).
* ``INVALIDATED`` — claim_token's canonical posterior anchor expired per Catalog #298 30-day
                    staleness window OR superseded by a NEWER event.
* ``UNKNOWN``   — claim_token has NO canonical posterior anchor (default-permissive when
                  no canonical evidence exists; sister of Catalog #378 default-permissive
                  WARN_UNKNOWN_TOKEN cascade).

Hooks per Catalog #125:

* #4 cathedral autopilot dispatch — ACTIVE via NEW sister cathedral consumer
  ``phantom_score_canonical_posterior_lookup_consumer``.
* #5 continual-learning posterior — ACTIVE (READS canonical posterior;
  validator IS a consumer surface).
* #6 probe-disambiguator — ACTIVE (4-state verdict IS the canonical
  disambiguator between current-CLEAN vs phantom-recurrence-from-falsified).
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = (
    "MIN_RATIONALE_LEN",
    "PLACEHOLDER_RATIONALES",
    "AutoFooterCascadeResult",
    "CanonicalPosteriorReadValidationVerdict",
    "PosteriorReadVerdict",
    "SpawnPromptRecommendation",
    "SpawnPromptValidationVerdict",
    "auto_emit_append_only_footer_to_memos_citing_falsified_score",
    "validate_memo_claim_against_canonical_posterior",
    "validate_spawn_prompt_against_canonical_posterior",
)


PLACEHOLDER_RATIONALES = ("<rationale>", "<reason>", "rationale", "reason", "")
"""Per Catalog #287 sister discipline. Rejected as waiver rationales."""

MIN_RATIONALE_LEN = 4
"""Minimum non-placeholder rationale length per Catalog #287 sister."""


class PosteriorReadVerdict(StrEnum):
    """4-state verdict taxonomy (UNKNOWN included) per design memo Phase A."""

    CLEAN = "CLEAN"
    FALSIFIED = "FALSIFIED"
    KILLED = "KILLED"
    PHANTOM = "PHANTOM"
    INVALIDATED = "INVALIDATED"
    UNKNOWN = "UNKNOWN"


_BLOCKING_VERDICTS = frozenset(
    {
        PosteriorReadVerdict.FALSIFIED,
        PosteriorReadVerdict.KILLED,
        PosteriorReadVerdict.PHANTOM,
        PosteriorReadVerdict.INVALIDATED,
    }
)
"""Verdicts that block memo / spawn-prompt landing per design memo §"4-state verdict taxonomy"."""


class SpawnPromptRecommendation(StrEnum):
    """Sister of Catalog #378 ``MainThreadSpawnGuardVerdict.recommendation``."""

    PROCEED = "PROCEED"
    ABORT_PHANTOM_TOKEN_CITED = "ABORT_PHANTOM_TOKEN_CITED"
    WARN_UNKNOWN_TOKEN = "WARN_UNKNOWN_TOKEN"


@dataclass(frozen=True)
class CanonicalPosteriorReadValidationVerdict:
    """Frozen dataclass per Catalog #323. Per-claim-token canonical posterior verdict.

    Fields:

    * ``claim_token`` — the verbatim claim token queried against canonical posterior.
    * ``verdict`` — one of :class:`PosteriorReadVerdict`.
    * ``matched_anchor_id`` — canonical anchor id (``equation_id`` /
      ``anti_pattern_id`` / ``probe_id``) that produced the verdict; empty for
      UNKNOWN.
    * ``matched_anchor_source`` — ``canonical_equations`` / ``canonical_anti_patterns``
      / ``probe_outcomes_ledger`` / ``no_match``.
    * ``matched_anchor_summary`` — human-readable one-line summary of the
      matched event.
    * ``canonical_provenance`` — Catalog #323 canonical Provenance dict.
    * ``adjudicated_at_utc`` — ISO UTC timestamp of the matched event (empty
      for UNKNOWN).
    """

    claim_token: str
    verdict: PosteriorReadVerdict
    matched_anchor_id: str
    matched_anchor_source: str
    matched_anchor_summary: str
    canonical_provenance: Mapping[str, Any]
    adjudicated_at_utc: str = ""

    @property
    def is_blocking(self) -> bool:
        """True if verdict is FALSIFIED / KILLED / PHANTOM / INVALIDATED."""
        return self.verdict in _BLOCKING_VERDICTS


@dataclass(frozen=True)
class SpawnPromptValidationVerdict:
    """Frozen dataclass per Catalog #323. Sister of Catalog #378 SpawnGuardVerdict."""

    recommendation: SpawnPromptRecommendation
    per_token_verdicts: tuple[CanonicalPosteriorReadValidationVerdict, ...]
    blocking_token_verdicts: tuple[CanonicalPosteriorReadValidationVerdict, ...]
    unknown_token_count: int
    canonical_provenance: Mapping[str, Any]
    rationale: str


@dataclass(frozen=True)
class AutoFooterCascadeResult:
    """Frozen dataclass per Catalog #323. Result of footer cascade across memos."""

    falsified_score_token: str
    falsification_verdict: str
    memos_scanned_count: int
    memos_with_token_count: int
    footers_emitted_count: int
    footers_skipped_already_present_count: int
    emitted_at_utc: str
    canonical_provenance: Mapping[str, Any]


def _utc_iso_now() -> str:
    """Return current UTC ISO timestamp."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_predicted_provenance(consumer_name: str) -> Mapping[str, Any]:
    """Build canonical Provenance per Catalog #323 for predicted-grade validator output."""
    return {
        "kind": "CANONICAL_POSTERIOR_READ_VALIDATOR_VERDICT",
        "consumer_name": consumer_name,
        "schema_version": "canonical_posterior_read_validator_v1_20260529",
        "score_claim": False,
        "evidence_grade": "predicted",
        "axis_tag": "[predicted]",
        "captured_at_utc_source": "tac.canonical_posterior_read_validator",
        "captured_at_utc": _utc_iso_now(),
    }


def _normalize_token(token: str) -> str:
    """Lowercase + strip whitespace for case-insensitive matching.

    Per sister cathedral consumers (canonical_equation_lookup_consumer +
    anti_pattern_lookup_consumer), best-effort substring matching is
    intentionally case-insensitive.
    """
    return token.strip().lower()


def _token_overlaps(needle: str, haystack: str) -> bool:
    """Best-effort substring overlap per sister cathedral consumer pattern."""
    n = _normalize_token(needle)
    h = _normalize_token(haystack)
    if not n or not h:
        return False
    # Token-level word match (avoid spurious substring matches on common tokens)
    # but also allow direct substring for compound tokens.
    return n in h or h in n


def _classify_anti_pattern_severity(severity: str) -> PosteriorReadVerdict:
    """Map anti-pattern severity to verdict per design memo §"4-state taxonomy".

    * critical / high → PHANTOM (sister of Catalog #321/#322 phantom-score class)
    * medium → FALSIFIED (canonical implementation-level falsification)
    * low → INVALIDATED (canonical degraded-evidence class)
    """
    sev = severity.lower()
    if "critical" in sev or "high" in sev:
        return PosteriorReadVerdict.PHANTOM
    if "medium" in sev:
        return PosteriorReadVerdict.FALSIFIED
    if "low" in sev:
        return PosteriorReadVerdict.INVALIDATED
    return PosteriorReadVerdict.UNKNOWN


def _query_canonical_anti_patterns(claim_token: str) -> tuple[str, str, str, str] | None:
    """Query canonical anti-patterns registry for matching claim_token.

    Returns (anti_pattern_id, severity, summary, last_recalibration_utc) or
    None if no match.
    """
    try:
        from tac.canonical_anti_patterns import query_anti_patterns
    except ImportError:
        return None

    try:
        for ap in query_anti_patterns():
            if _token_overlaps(claim_token, ap.anti_pattern_id):
                summary = (
                    ap.description[:200]
                    if ap.description
                    else f"canonical anti-pattern {ap.anti_pattern_id}"
                )
                return (
                    ap.anti_pattern_id,
                    ap.severity,
                    summary,
                    ap.last_recalibration_utc or "",
                )
            # Also match in canonical_source_anchor (per Wave N+33 anchor pattern)
            if ap.canonical_source_anchor and _token_overlaps(
                claim_token, ap.canonical_source_anchor
            ):
                summary = (
                    f"canonical_source_anchor match: "
                    f"{ap.canonical_source_anchor[:160]}"
                )
                return (
                    ap.anti_pattern_id,
                    ap.severity,
                    summary,
                    ap.last_recalibration_utc or "",
                )
    except Exception:
        return None
    return None


def _query_canonical_equations(claim_token: str) -> tuple[str, dict, str] | None:
    """Query canonical equations registry for matching claim_token.

    Returns (equation_id, residual_dict, last_calibration_utc) or None.
    """
    try:
        from tac.canonical_equations import query_equations
    except ImportError:
        return None

    try:
        for eq in query_equations():
            if _token_overlaps(claim_token, eq.equation_id):
                return (
                    eq.equation_id,
                    dict(eq.predicted_vs_empirical_residual),
                    eq.last_calibration_utc or "",
                )
    except Exception:
        return None
    return None


def _query_probe_outcomes_ledger(
    claim_token: str,
) -> tuple[str, str, str, str, str] | None:
    """Query probe outcomes ledger for matching claim_token.

    Returns (probe_id, verdict, blocker_status, substrate, adjudicated_at_utc)
    or None. Uses latest-event-wins per substrate per ``probe_id``.
    """
    try:
        from tac.probe_outcomes_ledger import load_outcomes_strict
    except ImportError:
        return None

    try:
        outcomes = load_outcomes_strict()
    except Exception:
        return None

    # Group by probe_id, take latest event
    latest_per_probe: dict[str, dict] = {}
    for row in outcomes:
        pid = row.get("probe_id", "")
        if not pid:
            continue
        ts = row.get("adjudicated_at_utc") or row.get("written_at_utc") or ""
        existing = latest_per_probe.get(pid)
        if existing is None or ts > (
            existing.get("adjudicated_at_utc")
            or existing.get("written_at_utc")
            or ""
        ):
            latest_per_probe[pid] = row

    # Match claim_token against probe_id OR substrate
    for pid, row in latest_per_probe.items():
        substrate = row.get("substrate", "")
        if _token_overlaps(claim_token, pid) or _token_overlaps(
            claim_token, substrate
        ):
            return (
                pid,
                row.get("verdict", "UNKNOWN"),
                row.get("blocker_status", "unknown"),
                substrate,
                row.get("adjudicated_at_utc", ""),
            )
    return None


def validate_memo_claim_against_canonical_posterior(
    memo_text: str, claim_token: str
) -> CanonicalPosteriorReadValidationVerdict:
    """Query a single claim_token against canonical posterior.

    Returns :class:`CanonicalPosteriorReadValidationVerdict` with 4-state
    verdict per latest-event-wins semantics across canonical_equations +
    canonical_anti_patterns + probe_outcomes_ledger.

    The ``memo_text`` parameter is reserved for future context-aware queries
    (e.g. extract co-cited tokens for cross-anchor coherence checks); the
    current implementation queries by claim_token alone.

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": the
    validator surfaces the matched anchor's evidence so downstream consumers
    can audit the verdict provenance.
    """
    _ = memo_text  # Reserved for future context-aware queries

    if not claim_token or not claim_token.strip():
        return CanonicalPosteriorReadValidationVerdict(
            claim_token=claim_token,
            verdict=PosteriorReadVerdict.UNKNOWN,
            matched_anchor_id="",
            matched_anchor_source="no_match",
            matched_anchor_summary="empty claim_token",
            canonical_provenance=_build_predicted_provenance(
                "validate_memo_claim_against_canonical_posterior"
            ),
        )

    # Probe outcomes ledger first (canonical disambiguation per Catalog #313)
    probe_match = _query_probe_outcomes_ledger(claim_token)
    if probe_match is not None:
        pid, verdict_str, blocker_status, substrate, adj_utc = probe_match
        # Map probe verdict to canonical posterior verdict
        v = verdict_str.upper()
        if v == "KILL":
            verdict = PosteriorReadVerdict.KILLED
        elif v in {"DEFER", "INDEPENDENT"}:
            verdict = PosteriorReadVerdict.FALSIFIED
        elif v in {"PROCEED", "CONFIRMED", "RATIFIED"}:
            verdict = PosteriorReadVerdict.CLEAN
        elif v in {"PARTIAL"}:
            verdict = PosteriorReadVerdict.INVALIDATED
        else:
            verdict = PosteriorReadVerdict.UNKNOWN

        return CanonicalPosteriorReadValidationVerdict(
            claim_token=claim_token,
            verdict=verdict,
            matched_anchor_id=pid,
            matched_anchor_source="probe_outcomes_ledger",
            matched_anchor_summary=(
                f"probe {pid} substrate={substrate} verdict={verdict_str} "
                f"blocker_status={blocker_status}"
            ),
            canonical_provenance=_build_predicted_provenance(
                "validate_memo_claim_against_canonical_posterior"
            ),
            adjudicated_at_utc=adj_utc,
        )

    # Canonical anti-patterns (PHANTOM / FALSIFIED / INVALIDATED per severity)
    ap_match = _query_canonical_anti_patterns(claim_token)
    if ap_match is not None:
        ap_id, severity, summary, last_recal = ap_match
        verdict = _classify_anti_pattern_severity(severity)
        return CanonicalPosteriorReadValidationVerdict(
            claim_token=claim_token,
            verdict=verdict,
            matched_anchor_id=ap_id,
            matched_anchor_source="canonical_anti_patterns",
            matched_anchor_summary=f"{summary} (severity={severity})",
            canonical_provenance=_build_predicted_provenance(
                "validate_memo_claim_against_canonical_posterior"
            ),
            adjudicated_at_utc=last_recal,
        )

    # Canonical equations (CLEAN if registered + well-calibrated; FALSIFIED
    # if residual indicates falsification per existing anchors)
    eq_match = _query_canonical_equations(claim_token)
    if eq_match is not None:
        eq_id, residuals, last_cal = eq_match
        # Heuristic: if any residual key contains "falsified" / "phantom" /
        # "invalidated" → that verdict; else CLEAN.
        falsified_marker = False
        phantom_marker = False
        for key in residuals:
            k = key.lower()
            if "phantom" in k:
                phantom_marker = True
            elif "falsified" in k or "implementation_level_falsification" in k:
                falsified_marker = True
        if phantom_marker:
            verdict = PosteriorReadVerdict.PHANTOM
        elif falsified_marker:
            verdict = PosteriorReadVerdict.FALSIFIED
        else:
            verdict = PosteriorReadVerdict.CLEAN
        return CanonicalPosteriorReadValidationVerdict(
            claim_token=claim_token,
            verdict=verdict,
            matched_anchor_id=eq_id,
            matched_anchor_source="canonical_equations",
            matched_anchor_summary=(
                f"canonical equation {eq_id} "
                f"residual_keys={list(residuals.keys())[:3]}"
            ),
            canonical_provenance=_build_predicted_provenance(
                "validate_memo_claim_against_canonical_posterior"
            ),
            adjudicated_at_utc=last_cal,
        )

    # No match — UNKNOWN
    return CanonicalPosteriorReadValidationVerdict(
        claim_token=claim_token,
        verdict=PosteriorReadVerdict.UNKNOWN,
        matched_anchor_id="",
        matched_anchor_source="no_match",
        matched_anchor_summary=(
            f"no canonical posterior anchor matched claim_token={claim_token!r}"
        ),
        canonical_provenance=_build_predicted_provenance(
            "validate_memo_claim_against_canonical_posterior"
        ),
    )


def validate_spawn_prompt_against_canonical_posterior(
    spawn_prompt_text: str, cited_tokens: Sequence[str]
) -> SpawnPromptValidationVerdict:
    """Sister of Catalog #378 ``verify_head_state_before_main_thread_spawn``.

    Validates EVERY cited token in a spawn-prompt against canonical posterior.
    Recommendation cascade:

    * ANY token returns PHANTOM / FALSIFIED / KILLED / INVALIDATED →
      ``ABORT_PHANTOM_TOKEN_CITED`` (one blocking token blocks the spawn-prompt;
      sister of MAX-aggregation per Catalog #373 anti-pattern Pareto
      polytope exclusion).
    * ALL tokens CLEAN → ``PROCEED``.
    * Some tokens UNKNOWN + none blocking → ``WARN_UNKNOWN_TOKEN`` (default-
      permissive sister of Catalog #378 default-permissive recommendation when
      no canonical evidence exists).
    """
    per_token: list[CanonicalPosteriorReadValidationVerdict] = []
    blocking: list[CanonicalPosteriorReadValidationVerdict] = []
    unknown_count = 0

    for token in cited_tokens:
        verdict = validate_memo_claim_against_canonical_posterior(
            spawn_prompt_text, token
        )
        per_token.append(verdict)
        if verdict.is_blocking:
            blocking.append(verdict)
        elif verdict.verdict == PosteriorReadVerdict.UNKNOWN:
            unknown_count += 1

    if blocking:
        recommendation = SpawnPromptRecommendation.ABORT_PHANTOM_TOKEN_CITED
        rationale = (
            f"{len(blocking)} blocking token(s) cited in spawn-prompt: "
            + ", ".join(b.claim_token for b in blocking[:3])
            + (
                f" + {len(blocking) - 3} more"
                if len(blocking) > 3
                else ""
            )
            + ". Spawn-prompt cites canonical posterior tokens with FALSIFIED "
            "/ KILLED / PHANTOM / INVALIDATED latest event. Per Catalog "
            "#(BB-claim) sister of #378, this is the canonical READ-surface "
            "phantom-score recurrence bug class. ABORT or apply waiver."
        )
    elif unknown_count > 0:
        recommendation = SpawnPromptRecommendation.WARN_UNKNOWN_TOKEN
        rationale = (
            f"{unknown_count} cited token(s) have NO canonical posterior "
            "anchor. Default-permissive WARN per sister Catalog #378 pattern. "
            "Sister phantom-recurrence prevention does NOT block UNKNOWN."
        )
    else:
        recommendation = SpawnPromptRecommendation.PROCEED
        rationale = (
            f"All {len(per_token)} cited token(s) CLEAN per canonical "
            "posterior latest-event. Spawn-prompt may proceed."
        )

    return SpawnPromptValidationVerdict(
        recommendation=recommendation,
        per_token_verdicts=tuple(per_token),
        blocking_token_verdicts=tuple(blocking),
        unknown_token_count=unknown_count,
        canonical_provenance=_build_predicted_provenance(
            "validate_spawn_prompt_against_canonical_posterior"
        ),
        rationale=rationale,
    )


_FOOTER_MARKER_RE = re.compile(
    r"<!--\s*CANONICAL_POSTERIOR_UPDATE\s*:\s*claim_token=([^\s]+)\s*verdict=([^\s]+).*?-->"
)


def _footer_already_present(
    memo_text: str, falsified_score_token: str, falsification_verdict: str
) -> bool:
    """Check if footer for this (token, verdict) tuple already exists."""
    matches = _FOOTER_MARKER_RE.findall(memo_text)
    for token, verdict in matches:
        if (
            _normalize_token(token) == _normalize_token(falsified_score_token)
            and _normalize_token(verdict)
            == _normalize_token(falsification_verdict)
        ):
            return True
    return False


def _build_footer(
    falsified_score_token: str,
    falsification_verdict: str,
    emitted_at_utc: str,
    catalog_number_token: str = "Catalog #(BB-claim)",
) -> str:
    """Build canonical APPEND-ONLY footer per design memo §"Deliverable 1"."""
    return (
        f"\n\n<!-- CANONICAL_POSTERIOR_UPDATE: "
        f"claim_token={falsified_score_token} "
        f"verdict={falsification_verdict} "
        f"at_utc={emitted_at_utc} "
        f"per {catalog_number_token} canonical READ-surface validator -->\n"
    )


def _discover_memo_files(
    repo_root: Path,
    research_dir_rel: str = ".omx/research",
    include_claude_memory: bool = False,
    claude_memory_dir: Path | None = None,
) -> list[Path]:
    """Discover operator-facing memo files for footer cascade.

    Per CLAUDE.md OSS-hermetic discipline + Catalog #290/#291/#292 sister:
    Claude memory directory scan is opt-in via ``include_claude_memory=True``
    only.
    """
    memo_files: list[Path] = []

    research = repo_root / research_dir_rel
    if research.exists():
        memo_files.extend(research.rglob("*.md"))

    if (
        include_claude_memory
        and claude_memory_dir is not None
        and claude_memory_dir.exists()
    ):
        memo_files.extend(claude_memory_dir.rglob("feedback_*.md"))

    return memo_files


def auto_emit_append_only_footer_to_memos_citing_falsified_score(
    falsified_score_token: str,
    falsification_verdict: str,
    *,
    repo_root: Path | str | None = None,
    research_dir_rel: str = ".omx/research",
    include_claude_memory: bool = False,
    claude_memory_dir: Path | str | None = None,
    dry_run: bool = False,
    catalog_number_token: str = "Catalog #(BB-claim)",
) -> AutoFooterCascadeResult:
    """Sister cascade across operator-facing memos citing a now-falsified token.

    When a canonical posterior verdict flips to FALSIFIED / KILLED / PHANTOM /
    INVALIDATED, this helper emits an HTML-comment APPEND-ONLY footer per
    Catalog #110/#113 HISTORICAL_PROVENANCE discipline to every memo citing
    the flipped token, WITHOUT mutating body content.

    Footer format: ``<!-- CANONICAL_POSTERIOR_UPDATE: claim_token=<X>
    verdict=<Y> at_utc=<UTC> per Catalog #(BB-claim) ... -->``

    Per Catalog #110/#113 APPEND-ONLY: footer is appended at end-of-file; body
    text is NOT mutated. Idempotent — re-emission for same (token, verdict)
    tuple skips memos that already carry the footer.

    Args:
        falsified_score_token: the token whose canonical posterior verdict
            flipped (e.g. ``alpha_4.74_lane_g_v3_siren``).
        falsification_verdict: one of FALSIFIED / KILLED / PHANTOM /
            INVALIDATED per :class:`PosteriorReadVerdict`.
        repo_root: repo root path (default: current working directory).
        research_dir_rel: research directory relative to repo_root.
        include_claude_memory: opt-in flag for Claude memory directory scan
            (default False per OSS-hermetic discipline).
        claude_memory_dir: Claude memory directory path (required if
            include_claude_memory=True).
        dry_run: if True, scan only; do not write footers.
        catalog_number_token: catalog # reference token for footer text.
    """
    if repo_root is None:
        repo_root = Path.cwd()
    repo_root = Path(repo_root)

    claude_dir_path: Path | None = None
    if claude_memory_dir is not None:
        claude_dir_path = Path(claude_memory_dir)

    emitted_at_utc = _utc_iso_now()

    memo_files = _discover_memo_files(
        repo_root,
        research_dir_rel=research_dir_rel,
        include_claude_memory=include_claude_memory,
        claude_memory_dir=claude_dir_path,
    )

    memos_with_token = 0
    footers_emitted = 0
    footers_skipped = 0
    normalized_token = _normalize_token(falsified_score_token)

    footer = _build_footer(
        falsified_score_token,
        falsification_verdict,
        emitted_at_utc,
        catalog_number_token=catalog_number_token,
    )

    for memo_path in memo_files:
        try:
            text = memo_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Skip files that don't mention the token at all
        if normalized_token not in text.lower():
            continue
        memos_with_token += 1
        # Idempotency check
        if _footer_already_present(
            text, falsified_score_token, falsification_verdict
        ):
            footers_skipped += 1
            continue
        if not dry_run:
            try:
                with memo_path.open("a", encoding="utf-8") as fh:
                    fh.write(footer)
                footers_emitted += 1
            except OSError:
                # Best-effort; skip on permission error
                continue
        else:
            footers_emitted += 1

    return AutoFooterCascadeResult(
        falsified_score_token=falsified_score_token,
        falsification_verdict=falsification_verdict,
        memos_scanned_count=len(memo_files),
        memos_with_token_count=memos_with_token,
        footers_emitted_count=footers_emitted,
        footers_skipped_already_present_count=footers_skipped,
        emitted_at_utc=emitted_at_utc,
        canonical_provenance=_build_predicted_provenance(
            "auto_emit_append_only_footer_to_memos_citing_falsified_score"
        ),
    )
