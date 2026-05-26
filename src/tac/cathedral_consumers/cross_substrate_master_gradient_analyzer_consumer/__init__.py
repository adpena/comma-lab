# SPDX-License-Identifier: MIT
"""Cathedral consumer for META-LIFT-1 cross-substrate master-gradient analyzer.

Per Catalog #335 :class:`tac.cathedral.consumer_contract.CathedralConsumerContract`
+ the 11th standing directive ORDER discipline (ONE canonical cross-substrate
analyzer FIRST; per-substrate consumption SECOND). Auto-discovered per Catalog
#336/#337 invocation gates by
:func:`tools.cathedral_autopilot_autonomous_loop.discover_and_register_consumers`.

Sister of:
  * :mod:`tac.cathedral_consumers.master_gradient_aggregate_consumer` (Catalog
    #354 exploit #1 — per-substrate aggregate gradient observability)
  * :mod:`tac.cathedral_consumers.master_gradient_per_pair_consumer` (per-pair
    gradient observability)
  * :mod:`tac.cathedral_consumers.cross_substrate_similarity_consumer`
    (SUB_ADDITIVE / SUPER_ADDITIVE / ANTAGONISTIC classification at a DIFFERENT
    axis than the META-LIFT-1 ranked-opportunity surface)

Where the sister Catalog #354 exploit consumers operate PER-SUBSTRATE, this
consumer is the META-lift cross-substrate sister: it loads the latest
canonical analysis from
``.omx/state/cross_substrate_master_gradient_analyses.jsonl`` (sister of
``.omx/state/master_gradient_anchors.jsonl`` at the cross-substrate META
sub-surface) and surfaces a per-candidate annotation listing the candidate's
substrate-archive rank within the canonical ranked-opportunity list.

Per Catalog #341 cathedral consumer routing markers: every return value
carries ``predicted_delta_adjustment=0.0`` + ``promotable=False`` +
``axis_tag="[predicted]"``. The cross-substrate ranking is OBSERVABILITY-ONLY
by construction. Promotion of a ranking entry to a contest score signal
REQUIRES paired-CUDA empirical anchor per CLAUDE.md "Submission auth eval —
BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable.

Hook numbers per Catalog #125 6-hook wire-in:
  * Hook #1 SENSITIVITY_MAP — ACTIVE (cross-substrate per-axis projections
    feed :mod:`tac.sensitivity_map` axis_weights downstream)
  * Hook #2 PARETO_CONSTRAINT — ACTIVE (canonical Cauchy-Schwarz aggregate
    upper bound IS the canonical Pareto feasibility boundary; Dim 1 Phase 4
    Dykstra alternating projections can consume the bound)
  * Hook #3 BIT_ALLOCATOR — ACTIVE (per-axis ranked opportunities feed the
    bit allocator priority cascade per Dim 6 Step 6.5)
  * Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH — ACTIVE (this consumer is the
    canonical cathedral entry point for the META-LIFT-1 analysis)
  * Hook #5 CONTINUAL_LEARNING_POSTERIOR — ACTIVE (new analysis rows
    written via :func:`tac.cross_substrate_master_gradient_analyzer.append_analysis_locked`
    are read fresh per candidate so the consumer always sees the latest
    posterior)
  * Hook #6 PROBE_DISAMBIGUATOR — ACTIVE (the per-axis ranking IS the
    canonical disambiguator between competing reactivation paths — a
    substrate with high per-axis leverage in the seg axis routes
    differently than the inverse per CLAUDE.md "SegNet vs PoseNet
    importance — operating-point dependent")

Mission contribution per Catalog #300: `frontier_breaking_enabler` —
the canonical cross-substrate ranking unblocks per-axis Pareto polytope
routing (Dim 1 Phase 4) AND per-axis bit-allocator priority (Dim 6 Step 6.5)
without per-substrate iteration. The immediate score-lowering value is via
re-routing dispatch budget from low-leverage substrate-axes to high-leverage
ones.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.cathedral.consumer_contract import HookNumber

CONSUMER_NAME = "cross_substrate_master_gradient_analyzer_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.PARETO_CONSTRAINT,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)

CONSUMES_MASTER_GRADIENT_ANCHORS = True
"""Per :mod:`tac.master_gradient` post-anchor consumer hook contract.

When a NEW master-gradient anchor is appended via
:func:`tac.master_gradient.append_anchor_locked`, the
``_fire_post_anchor_consumer_hooks`` discovery loop invokes this consumer's
:func:`update_from_anchor` so the cross-substrate analyzer can be re-run on
the freshly-updated per-substrate corpus.
"""


def _state_dir() -> Path:
    """Locate ``.omx/state/`` for canonical ledger discovery.

    Walks up from this file to find the repo root + ``.omx/state/``. Returns
    a non-existent Path-equivalent (``.omx/state``) if not found so the
    consumer fails GRACEFULLY (returns an empty annotation rather than
    crashing the cathedral autopilot loop).
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / ".omx" / "state"
        if candidate.is_dir():
            return candidate
    return Path(".omx/state")


def _load_latest_analysis() -> Mapping[str, Any] | None:
    """Load the most-recent cross-substrate analysis row from the ledger.

    Returns ``None`` if the ledger is missing (graceful degradation per
    Catalog #245 + #248 sister fail-closed disciplines: a missing analysis
    does NOT crash the cathedral autopilot loop).
    """
    ledger = _state_dir() / "cross_substrate_master_gradient_analyses.jsonl"
    if not ledger.exists():
        return None
    raw = ledger.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    # Most-recent row wins (latest written_at_utc).
    latest: Mapping[str, Any] | None = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # Per Catalog #138 sister: lenient-loader fallback for graceful
            # degradation; strict-load is available via
            # tac.cross_substrate_master_gradient_analyzer.load_analyses_strict.
            continue
        if not isinstance(row, dict):
            continue
        if latest is None or row.get("written_at_utc", "") > latest.get("written_at_utc", ""):
            latest = row
    return latest


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Triggered when a new master-gradient anchor lands via
    :func:`tac.master_gradient.append_anchor_locked`. This consumer is
    STATELESS (re-reads the latest cross-substrate analysis on every
    consume call) so the hook is a no-op by design.

    The canonical operator-routable flow is:

    1. Operator runs :mod:`tools.extract_master_gradient` to extract a
       new per-substrate anchor.
    2. ``append_anchor_locked`` writes the row + fires this hook.
    3. Operator runs :mod:`tools.cross_substrate_master_gradient_cli`
       (with ``--persist-to-ledger``) to rebuild the cross-substrate
       analysis incorporating the new anchor.
    4. Cathedral autopilot ranker invokes this consumer's
       :func:`consume_candidate` per candidate and reads the updated
       analysis fresh.

    Per Catalog #327 contest-axis custody: this hook does NOT promote
    diagnostic anchors. The authority filter is applied at analyzer time
    via :func:`tac.master_gradient.is_authoritative_axis_anchor`.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Annotates the candidate with its rank (if any) in the latest canonical
    cross-substrate ranked-opportunity list. The annotation is
    OBSERVABILITY-ONLY per Catalog #341 (``predicted_delta_adjustment=0.0``
    + ``promotable=False`` + ``axis_tag="[predicted]"``).

    Args:
        candidate: ranker candidate dict; expected to carry
          ``archive_sha256`` (or ``archive_sha`` / ``sha256``) for
          rank lookup.

    Returns:
        Canonical contribution dict with cross-substrate ranking
        annotation (Tier A observability-only).
    """
    analysis = _load_latest_analysis()
    if analysis is None:
        return {
            "consumer_name": CONSUMER_NAME,
            "predicted_delta_adjustment": 0.0,
            "promotable": False,
            "axis_tag": "[predicted]",
            "annotation": {
                "cross_substrate_analysis_status": "MISSING_ANALYSIS_LEDGER",
                "actionable_hint": (
                    "Run tools/cross_substrate_master_gradient_cli.py "
                    "--persist-to-ledger to populate the canonical analyses ledger."
                ),
            },
        }

    # Extract candidate sha (defensive — multiple ranker conventions).
    sha = (
        candidate.get("archive_sha256")
        or candidate.get("archive_sha")
        or candidate.get("sha256")
        or ""
    )

    ranked_opportunities = analysis.get("ranked_opportunities", [])
    cauchy_bound = analysis.get("cauchy_schwarz_aggregate_upper_bound", 0.0)
    n_substrates = len(analysis.get("substrate_rows", []))

    # Find this candidate's rank within the ranked opportunities.
    candidate_ranks: list[dict] = []
    if sha:
        for opp in ranked_opportunities:
            if opp.get("archive_sha256", "") == sha:
                candidate_ranks.append(
                    {
                        "rank": opp.get("rank"),
                        "axis": opp.get("axis"),
                        "per_byte_leverage": opp.get("per_byte_leverage"),
                        "is_authoritative": opp.get("is_authoritative"),
                    }
                )

    if candidate_ranks:
        cross_substrate_status = "RANKED"
    elif sha and ranked_opportunities:
        cross_substrate_status = "NOT_IN_TOP_N"
    else:
        cross_substrate_status = "UNRANKED"

    return {
        "consumer_name": CONSUMER_NAME,
        "predicted_delta_adjustment": 0.0,
        "promotable": False,
        "axis_tag": "[predicted]",
        "annotation": {
            "cross_substrate_analysis_id": analysis.get("analysis_id"),
            "cross_substrate_analysis_status": cross_substrate_status,
            "candidate_archive_sha256": sha,
            "candidate_ranks_per_axis": candidate_ranks,
            "cauchy_schwarz_aggregate_upper_bound": float(cauchy_bound),
            "n_substrates_analyzed": n_substrates,
            "canonical_equation_id": analysis.get("canonical_equation_id"),
            "canonical_equation_status": analysis.get("canonical_equation_status"),
            "evidence_grade": analysis.get(
                "evidence_grade", "[predicted; cross-substrate-aggregate-Taylor]"
            ),
            "promotion_routing": (
                "OBSERVABILITY-ONLY per Catalog #341 + CLAUDE.md "
                "'Apples-to-apples evidence discipline'. Promotion of "
                "ranking entry to contest score signal REQUIRES paired-CUDA "
                "empirical anchor per CLAUDE.md 'Submission auth eval'."
            ),
        },
    }


__all__ = [
    "CONSUMER_HOOK_NUMBERS",
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMES_MASTER_GRADIENT_ANCHORS",
    "consume_candidate",
    "update_from_anchor",
]
