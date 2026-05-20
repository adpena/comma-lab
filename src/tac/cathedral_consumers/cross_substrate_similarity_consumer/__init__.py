# SPDX-License-Identifier: MIT
"""Cathedral consumer for cross-substrate sensitivity similarity matrix.

Per WAVE-3-CROSS-CANDIDATE-SENSITIVITY-COMPARISON-DIAGNOSTIC task spec
deliverable E + PACT-NERV-DESIGN-SYMPOSIUM Section 13 stack-of-stacks
Catalog #322 EMPIRICAL alpha validation. Sister of
:mod:`tac.cathedral_consumers.canonical_equation_lookup_consumer`
(Catalog #344) and the bit-allocator pattern at
:mod:`tac.cathedral_consumers.bit_allocator_per_pair_consumer`.

Loads the latest cross-substrate similarity matrix at
``.omx/state/cross_substrate_sensitivity_similarity_matrix_<UTC>.json``
and, for any candidate that mentions a known substrate label, emits a
Tier A observability-only annotation listing the substrate's
classification with every other known substrate (SUB_ADDITIVE /
SUPER_ADDITIVE / ANTAGONISTIC / INDETERMINATE) so the cathedral
autopilot ranker AND downstream operator review can see the empirical
stacking matrix without manual ledger lookups.

Per Catalog #341 + CLAUDE.md "Apples-to-apples evidence discipline":
this consumer NEVER mutates ``predicted_delta_adjustment`` (always 0.0);
``promotable=False`` always; ``axis_tag="[predicted]"`` always. The
similarity matrix's per-pair classification is observability-only
because cross-substrate sensitivity comparison via per-byte master
gradient is sister of Catalog #287 ``[predicted]``-grade signal —
promotion to score-contributing requires paired CUDA + CPU empirical
auth-eval per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".

Hook assignments per Catalog #125:
  * #4 cathedral autopilot dispatch — ACTIVE (annotate candidates with
    cross-substrate classification matrix entries)
  * #5 continual-learning posterior — ACTIVE (when a NEW master-gradient
    anchor lands, the similarity matrix should be refreshed; helper
    surface declared but recomputation is operator-triggered via
    ``tools/wave_3_cross_candidate_sensitivity_compute.py`` to honor
    Catalog #287/#323 measurement provenance)
  * #1, #2, #3, #6 — N/A (observability-only annotation; per-pair
    classification feeds future Pareto / bit-allocator consumers via
    the matrix JSON file but this consumer itself is annotation-only)

Sister wire-in:
  Catalog #322 ``check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha``
  — this consumer's similarity matrix is the canonical empirical alpha
  surface that Catalog #322 protects from phantom-provenance corruption.
  All matrix entries derive from canonical master-gradient sidecars
  registered via ``tac.master_gradient.append_anchor_locked`` per
  Catalog #131 sister discipline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "cross_substrate_similarity_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


# Canonical similarity matrix path glob. Catalog #335 / #341: the cathedral
# autopilot ranker invokes the consumer via the canonical Protocol; this
# consumer reads the most-recent matrix on every consume call to honor
# the live-updating posterior pattern per Catalog #344.
_MATRIX_GLOB = "cross_substrate_sensitivity_similarity_matrix_*.json"


def _state_dir() -> Path:
    """Locate .omx/state/ for matrix discovery.

    Walks up from this file to find the repo root + .omx/state/. Returns
    None equivalent (Path that won't exist) if not found.
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / ".omx" / "state"
        if candidate.is_dir():
            return candidate
    return Path(".omx/state")


def _load_latest_matrix() -> Mapping[str, Any] | None:
    """Load the most-recent cross-substrate similarity matrix.

    Returns None when no matrix is on disk (catastrophic-failure resilient
    per Catalog #245 + #248 sister fail-closed disciplines: a missing
    matrix does NOT crash the cathedral autopilot loop; the consumer
    emits a graceful annotation that the matrix is unavailable).
    """
    state = _state_dir()
    if not state.is_dir():
        return None
    matches = sorted(state.glob(_MATRIX_GLOB))
    if not matches:
        return None
    latest = matches[-1]
    try:
        with latest.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        # Corrupt or unreadable; fail closed via None per Catalog #138
        # sister strict-load discipline (we don't quarantine here; the
        # canonical helper that wrote the matrix owns that surface).
        return None


def _classifications_for_substrate(
    matrix: Mapping[str, Any],
    target_substrate: str,
) -> list[dict[str, Any]]:
    """Return all matrix entries where target_substrate appears.

    Each entry surfaces (sister_substrate, classification, jaccard,
    per-axis Pearson) so the operator-facing annotation conveys WHY a
    pair classifies as SUB_ADDITIVE / SUPER_ADDITIVE / ANTAGONISTIC /
    INDETERMINATE.
    """
    pairs = matrix.get("pairs", [])
    if not isinstance(pairs, list):
        return []
    result: list[dict[str, Any]] = []
    for row in pairs:
        if not isinstance(row, dict):
            continue
        a = row.get("substrate_a", "")
        b = row.get("substrate_b", "")
        if a == target_substrate:
            sister = b
        elif b == target_substrate:
            sister = a
        else:
            continue
        result.append(
            {
                "sister_substrate": sister,
                "classification": row.get("classification", "UNKNOWN"),
                "top_k_byte_overlap_jaccard": row.get(
                    "top_k_byte_overlap_jaccard_l1"
                ),
                "per_axis_pearson_seg": row.get("per_axis_pearson_seg"),
                "per_axis_pearson_pose": row.get("per_axis_pearson_pose"),
                "per_axis_pearson_rate": row.get("per_axis_pearson_rate"),
            }
        )
    return result


def _candidate_substrate_label(candidate: Mapping[str, Any]) -> str | None:
    """Heuristic: extract canonical substrate label from candidate dict.

    Walks string-valued fields looking for known substrate label tokens.
    Returns the first match. The cathedral autopilot main loop passes a
    dict candidate; we tolerate missing / non-dict fields.
    """
    candidate_text_parts: list[str] = []
    for k, v in candidate.items():
        if isinstance(v, (str, int, float)):
            candidate_text_parts.append(f"{k}={v}")
    candidate_text = " ".join(candidate_text_parts).lower()
    known_labels = (
        "fec6_cpu_scorer_advisory",
        "a1_finetuned",
        "pr101_gold",
        "fec6_frontier_macos_advisory",
        "fec6_frontier_cuda_t4",
        "pr106_format0d",
        "pr107_apogee",
        # Also match shorter forms
        "fec6_frontier",
        "fec6",
        "pr101",
        "pr106",
        "pr107",
        "a1",
    )
    for lab in known_labels:
        if lab in candidate_text:
            # Try the canonical full label first if a short form matched
            if lab == "fec6_frontier":
                return "fec6_frontier_cuda_t4"
            if lab == "fec6":
                return "fec6_frontier_cuda_t4"
            if lab == "pr101":
                return "pr101_gold"
            if lab == "pr106":
                return "pr106_format0d"
            if lab == "pr107":
                return "pr107_apogee"
            if lab == "a1":
                return "a1_finetuned"
            return lab
    return None


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Reference implementation: NO-OP. The canonical refresh path is
    operator-triggered via
    ``PYTHONPATH=src python /tmp/wave3_analysis/compute_similarity_matrix.py``
    (or the production-promoted ``tools/`` equivalent landed by a
    sister-subagent later) because automatic refit requires explicit
    signed measurement provenance per Catalog #287 + #323. A new
    master-gradient anchor (anchor argument) MAY trigger a manual
    matrix refresh; this consumer surfaces the latest matrix on every
    consume call so live updates propagate without per-consumer state.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — annotate candidate with substrate-pair classifications.

    Returns the canonical Tier A contribution per Catalog #341:
      * predicted_delta_adjustment=0.0 (NEVER mutates score)
      * promotable=False (NEVER promotes)
      * axis_tag="[predicted]" (advisory-only per Catalog #287)
      * rationale + matched_classifications (operator audit surface)
    """
    matrix = _load_latest_matrix()
    if matrix is None:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "cross-substrate similarity matrix unavailable in .omx/state/ "
                "(catalog #341 graceful-failure path) [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
            "matrix_available": False,
        }

    target_substrate = _candidate_substrate_label(candidate)
    if target_substrate is None:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "candidate did not match any known substrate label in cross-"
                "substrate similarity matrix; observability-only annotation "
                "[predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
            "matrix_available": True,
            "matrix_captured_at_utc": matrix.get("captured_at_utc", ""),
            "matched_substrate": None,
        }

    classifications = _classifications_for_substrate(matrix, target_substrate)
    if not classifications:
        rationale = (
            f"candidate matched substrate {target_substrate!r} but no sister "
            "pairs in the matrix; observability-only annotation [predicted]"
        )
    else:
        # Distribute classifications
        from collections import Counter
        cls_counts = Counter(c["classification"] for c in classifications)
        cls_summary = ", ".join(f"{k}={v}" for k, v in cls_counts.most_common())
        rationale = (
            f"candidate substrate {target_substrate!r} has {len(classifications)} "
            f"sister pairs in cross-substrate similarity matrix; classification "
            f"distribution: {cls_summary}; consult "
            "tools/list_canonical_equations.py / "
            ".omx/state/cross_substrate_sensitivity_similarity_matrix_*.json "
            "for per-pair details [predicted]"
        )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "matrix_available": True,
        "matrix_captured_at_utc": matrix.get("captured_at_utc", ""),
        "matched_substrate": target_substrate,
        "matched_classifications": classifications,
    }
