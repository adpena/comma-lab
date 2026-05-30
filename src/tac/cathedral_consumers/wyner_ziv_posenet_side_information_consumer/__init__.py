# SPDX-License-Identifier: MIT
"""Cathedral consumer for the Wyner-Ziv PoseNet-conditioned side-information equation.

Per CLAUDE.md "Canonical equations + models registry" non-negotiable + operator
task #1496 Wave N+36 routing + Catalog #335 paradigm-shift (canonical contract
auto-discovery). Wires the orphan-signal closure for the canonical equation
``wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1``
so the cathedral autopilot ranker sees per-candidate Wyner-Ziv conditional-entropy
savings predictions whenever a candidate matches the equation's domain.

This consumer is **Tier A** (per Catalog #341 canonical-routing markers):
``predicted_delta_adjustment=0.0`` + ``promotable=False`` + ``axis_tag="[predicted]"``.
Per-equation predictions are NEVER promoted to score adjustments — they surface as
``[predicted]`` annotations that future paired-CUDA RATIFICATION dispatches can
compare against to refresh the equation's ``predicted_vs_empirical_residual``.
Candidates that mention PoseNet/Wyner-Ziv without archive-charged or fixed-input
side-information custody are deliberately marked as unmatched with a blocker;
free inflate-time scorer side information violates the strict scorer rule.

Hook assignments per Catalog #125:
  * #1 sensitivity-map — ACTIVE (PoseNet output is encoder-side sensitivity signal)
  * #2 Pareto constraint — ACTIVE (Wyner-Ziv R(D|Y) bound IS canonical Pareto constraint)
  * #3 bit-allocator — ACTIVE PRIMARY (canonical pose_conditional mode for per_pair allocator)
  * #4 cathedral autopilot dispatch — ACTIVE (annotate candidates)
  * #5 continual-learning posterior — ACTIVE (refresh equation calibration on new anchors)
  * #6 probe-disambiguator — ACTIVE (Wyner-Ziv conditional vs unconditional IS disambiguator)

Cross-references:
  * Canonical equation module: ``tac.canonical_equations.wyner_ziv_decoder_side_posenet_side_information``
  * Sister consumer: ``tac.cathedral_consumers.canonical_equation_lookup_consumer`` (the
    general canonical-equations cathedral consumer; THIS consumer is the specific
    paradigm-bound complement for Wyner-Ziv PoseNet side-info matches)
  * Catalog #335 (canonical cathedral consumer contract)
  * Catalog #341 (Tier A canonical-routing markers)
  * Catalog #323 (canonical Provenance umbrella)
  * Z8 M6 landing memo: ``.omx/research/z8_m6_wyner_ziv_top_level_coder_full_implementation_landed_20260530.md``
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tac.cathedral.consumer_contract import HookNumber

CONSUMER_NAME = "wyner_ziv_posenet_side_information_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.PARETO_CONSTRAINT,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)

# Equation id this consumer specializes on; lookup uses startswith match to
# tolerate version bumps (vN suffix per equation_id pattern).
_EQUATION_ID_PREFIX = "wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction"


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    When a new empirical anchor (paired-CUDA / paired-CPU / advisory) lands
    that matches the Wyner-Ziv decoder-side PoseNet side-information domain,
    the canonical refresh path is ``tac.canonical_equations.auto_recalibrate_from_continual_learning_posterior``
    per Catalog #371 sister discipline (triggers when >=3 in-domain anchors land).
    This consumer is structurally NO-OP at the per-anchor surface; the canonical
    refresh is operator-triggered via ``tools/recalibrate_equation.py`` because
    automatic refit requires explicit signed measurement Provenance per Catalog
    #287/#323.

    Per the canonical Tier A contract (Catalog #341): this consumer never
    mutates posterior state directly; it is observability-only.
    """
    _ = anchor


def _candidate_side_info_custody(candidate: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Return whether the candidate proves legal decoder-side-info custody."""
    blockers: list[str] = []
    delivery_mode = str(
        candidate.get("side_info_delivery_mode") or candidate.get("decoder_side_info_delivery_mode") or ""
    ).lower()
    if delivery_mode in {"archive_charged", "fixed_contest_input"}:
        return True, blockers
    if candidate.get("archive_bound_side_info") is True:
        return True, blockers
    if candidate.get("side_info_charged_bytes") is not None:
        try:
            if int(candidate["side_info_charged_bytes"]) >= 0:
                return True, blockers
        except (TypeError, ValueError):
            blockers.append("side_info_charged_bytes_not_integer")
    if delivery_mode == "scorer_runtime_free":
        blockers.append("non_compliant_inflate_time_scorer_side_information_forbidden")
    elif delivery_mode == "compress_time_advisory_only":
        blockers.append("compress_time_posenet_signal_is_not_decoder_side_information")
    else:
        blockers.append("posenet_side_info_archive_custody_or_fixed_input_proof_missing")
    return False, blockers


def _candidate_matches_wyner_ziv_posenet_domain(
    candidate: Mapping[str, Any],
) -> tuple[bool, str, list[str]]:
    """Best-effort match heuristic for candidates that admit Wyner-Ziv PoseNet side info.

    A candidate matches only when its substrate / lane / archive_family token
    references a known Wyner-Ziv/PoseNet conditional coding surface AND it proves
    legal side-information custody. Generic Wyner-Ziv prose without custody is
    deliberately fail-closed so it cannot become an acquisition prior.

    Returns ``(matches, rationale, blockers)``. The rationale always carries the
    canonical Tier A ``[predicted]`` axis tag per Catalog #341.
    """
    # Walk candidate dict values that are strings or numerics; build a
    # single search corpus so we can apply multi-keyword token matching.
    corpus_parts: list[str] = []
    for k, v in candidate.items():
        if isinstance(v, (str, int, float)):
            corpus_parts.append(f"{k}={v}".lower())
        elif isinstance(v, Mapping):
            for sk, sv in v.items():
                if isinstance(sv, (str, int, float)):
                    corpus_parts.append(f"{k}.{sk}={sv}".lower())
    corpus = " ".join(corpus_parts)
    if not corpus:
        return (False, "candidate carries no string/numeric fields [predicted]", [])

    # Canonical match tokens. Order them by specificity (most specific first).
    canonical_match_tokens = (
        "wyner_ziv",
        "wyner-ziv",
        "wynerziv",
        "posenet_side_info",
        "posenet_conditional",
        "decoder_side_posenet",
        "conditional_entropy_reduction",
        "z8_hierarchical_predictive_coding",
        "wyner_ziv_layer",
        "wyner_ziv_pipeline_stage",
        "cooperative_receiver",  # Atick-Redlich sister surface
    )
    for token in canonical_match_tokens:
        if token in corpus:
            custody_ok, blockers = _candidate_side_info_custody(candidate)
            if custody_ok:
                return (
                    True,
                    (
                        f"candidate matches Wyner-Ziv PoseNet side-info domain via "
                        f"token '{token}' with legal side-info custody [predicted]"
                    ),
                    [],
                )
            return (
                False,
                (
                    f"candidate has Wyner-Ziv/PoseNet token '{token}' but lacks "
                    "archive-charged or fixed-input side-info custody [predicted]"
                ),
                blockers,
            )
    return (False, "no canonical Wyner-Ziv PoseNet side-info token [predicted]", [])


def _query_equation_residual_summary() -> dict[str, Any]:
    """Return latest residual summary for the canonical equation; safe-default on failure."""
    try:
        from tac.canonical_equations import query_equations
    except ImportError:
        return {
            "equation_id_prefix": _EQUATION_ID_PREFIX,
            "registry_available": False,
            "residuals": {},
            "is_well_calibrated": False,
        }
    try:
        for eq in query_equations():
            if eq.equation_id.startswith(_EQUATION_ID_PREFIX):
                return {
                    "equation_id_prefix": _EQUATION_ID_PREFIX,
                    "equation_id": eq.equation_id,
                    "registry_available": True,
                    "anchor_count": len(eq.empirical_anchors),
                    "residuals": dict(eq.predicted_vs_empirical_residual),
                    "is_well_calibrated": eq.is_well_calibrated,
                    "last_calibration_utc": eq.last_calibration_utc,
                }
    except Exception:
        pass
    return {
        "equation_id_prefix": _EQUATION_ID_PREFIX,
        "registry_available": True,
        "anchor_count": 0,
        "residuals": {},
        "is_well_calibrated": False,
        "rationale": "equation not registered yet",
    }


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — annotate candidate with Wyner-Ziv PoseNet side-info prediction.

    For each candidate the cathedral autopilot ranker dispatches, this consumer:

    1. Checks whether the candidate's substrate / lane / archive_family token
       maps to the Wyner-Ziv PoseNet side-info domain (best-effort string match).
    2. If matched, queries the canonical equation registry for the latest
       per-axis residual summary.
    3. Returns a Tier A canonical-routing-markers contribution per Catalog #341:
       ``predicted_delta_adjustment=0.0`` + ``promotable=False`` + ``axis_tag=[predicted]``.

    The consumer is observability-only by canonical contract. Per CLAUDE.md
    "Apples-to-apples evidence discipline" + "Forbidden empirical-claim-
    without-evidence-tag" non-negotiables: per-equation predictions are
    NEVER promoted to score adjustments — they surface as ``[predicted]``
    annotations that downstream consumers (operator review / paired-CUDA
    RATIFICATION queue) can act on.
    """
    matches, match_rationale, custody_blockers = _candidate_matches_wyner_ziv_posenet_domain(candidate)
    if not matches:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": match_rationale,
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
            "wyner_ziv_posenet_side_info_match": False,
            "decoder_side_info_custody_proven": False,
            "blockers": custody_blockers,
        }

    residual_summary = _query_equation_residual_summary()
    well_calibrated = bool(residual_summary.get("is_well_calibrated", False))
    anchor_count = int(residual_summary.get("anchor_count", 0))
    residuals = residual_summary.get("residuals", {})
    equation_id = residual_summary.get("equation_id", "")

    rationale = (
        f"{match_rationale}; "
        f"canonical equation {equation_id or _EQUATION_ID_PREFIX} "
        f"(anchor_count={anchor_count}, "
        f"well_calibrated={well_calibrated}, "
        f"per-axis residuals={residuals}); "
        "Wyner-Ziv 1976 Theorem 1 predicts rate-axis savings "
        "R(D|Y)<<R(D) when this substrate adopts PoseNet-conditioned coding "
        "with archive-charged or fixed-input decoder side information; "
        "see tools/list_canonical_equations.py for canonical readback"
    )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "wyner_ziv_posenet_side_info_match": True,
        "decoder_side_info_custody_proven": True,
        "blockers": [],
        "matched_equation_id": equation_id,
        "anchor_count": anchor_count,
        "is_well_calibrated": well_calibrated,
        "per_axis_residuals": dict(residuals),
    }


__all__ = [
    "CONSUMER_HOOK_NUMBERS",
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "consume_candidate",
    "update_from_anchor",
]
