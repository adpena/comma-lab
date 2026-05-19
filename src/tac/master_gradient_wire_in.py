# SPDX-License-Identifier: MIT
"""Canonical master-gradient wire-in helpers for analytical surfaces.

[verified-against: .omx/research/comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518.md §2.3 + §A1.1]
[verified-against: .omx/research/dynamic_per_candidate_composition_framework_all_canonical_apparatus_composed_20260518.md §3.1 6-hook wire-in declaration]
[verified-against: CLAUDE.md "Subagent coherence-by-default" non-negotiable + Catalog #125 6-hook wire-in]

This module closes the producer+consumer gap surfaced by the comprehensive
analytical-surfaces inventory (commit ``ac5921b2``): 33 surfaces (~47%) have
incomplete master-gradient wire-in. The canonical extractor
(``tools/extract_master_gradient.py``) emits ``MasterGradient`` anchors via
``tac.master_gradient.append_anchor_locked``; the canonical consumer surface
(``tac.master_gradient_consumers``) exposes
``load_per_pair_gradient_from_anchor`` /
``load_optimal_plan_for_archive``. Existing high-coverage consumers
(``bit_allocator_end_to_end`` / ``field_equation_planner`` /
``jacobian_fisher_importance_allocator`` / ``sensitivity_map.wyner_ziv_reweight``
/ ``cathedral_autopilot_autonomous_loop``) consume master-gradient directly.

The 4 zero-coverage surfaces inherited from the inventory's
"per-pair master gradient wire-in audit" table (§2.3):

1. ``tac.frontier_scan``      — annotate frontier anchors with the per-archive
                                  master-gradient anchor existence flag so
                                  drift-detection consumers can prioritize
                                  archives that already have gradient anchors
                                  for paired-comparison ranking.

2. ``tac.probe_outcomes_ledger`` — annotate adjudicated probe outcomes with
                                  ``master_gradient_anchor_archive_sha256`` via
                                  the canonical ``extra`` channel so future
                                  probe-disambiguator queries can correlate
                                  outcome verdicts with per-archive gradient
                                  signal.

3. ``tac.continual_learning``  — annotate posterior anchor rows with
                                  ``master_gradient_anchor_archive_sha256`` so
                                  cross-correlation between posterior anchors
                                  and master-gradient signal is queryable
                                  post-hoc (per inventory consumer table row 10).

4. ``tac.deploy.modal.call_id_ledger`` — annotate ``dispatched`` events with
                                  ``master_gradient_anchor_archive_sha256``
                                  via the canonical ``extra`` channel so a
                                  Modal harvested call_id can be cross-checked
                                  against the per-archive gradient anchor it
                                  was dispatched to refine (per inventory
                                  consumer table row 9).

Per CLAUDE.md "Catalog #327 master_gradient raw byte authority not landed":
this wire-in module emits SIGNAL-ANNOTATION ONLY. It does NOT expose raw
archive-byte / bit master-gradient APIs. Consumers wanting to mutate scored
archive bytes MUST route through the typed ``CandidateModificationSpec`` +
``grammar_aware_operator`` pattern in
``tac.master_gradient_operator_plan``.

Per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable:
this wire-in module returns DIAGNOSTIC METADATA only. It does NOT emit score
claims, does NOT promote anchors, does NOT change rank/kill verdicts. Every
consumer that consumes this module's output remains subject to its own
custody validator routing per Catalog #127.

6-hook wire-in declaration per Catalog #125:
- hook #1 sensitivity-map  = ACTIVE (annotation flows into sensitivity-map
                             cross-correlation via post-hoc query helpers).
- hook #2 Pareto constraint = N/A (signal-annotation only; no constraint).
- hook #3 bit-allocator     = ACTIVE (frontier_scan annotation lets bit-
                             allocator prioritize archives with gradient
                             anchors for paired-comparison ranking).
- hook #4 cathedral autopilot dispatch = ACTIVE (call_id_ledger annotation
                             lets the autopilot ranker cross-correlate
                             harvested call_id results with the gradient
                             anchor that motivated the dispatch).
- hook #5 continual-learning posterior = PRIMARY (THIS gate's posterior
                             annotation IS the canonical posterior-cross-
                             correlation surface).
- hook #6 probe-disambiguator = ACTIVE (probe_outcomes_ledger annotation
                             lets future probe-disambiguator queries route
                             through master-gradient-aware classification).
"""

from __future__ import annotations

import os
import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "MASTER_GRADIENT_WIRE_IN_SCHEMA_VERSION",
    "MasterGradientAnnotation",
    "annotate_frontier_anchors_with_master_gradient_existence",
    "annotate_posterior_row_with_master_gradient_anchor",
    "compute_master_gradient_wire_in_coverage",
    "query_anchors_with_master_gradient_coverage",
    "register_dispatched_call_id_with_master_gradient_anchor",
    "register_probe_outcome_with_master_gradient_anchor",
]

MASTER_GRADIENT_WIRE_IN_SCHEMA_VERSION = "master_gradient_wire_in_v1"

# Per CLAUDE.md "Beauty, simplicity, and developer experience" non-negotiable:
# small typed dataclass that consumer surfaces can attach without losing
# audit-trail integrity (every annotation carries its provenance source).
_DEFAULT_ANCHOR_DOMAIN_TOKEN = "master_gradient_anchor_archive_sha256"


@dataclass(frozen=True)
class MasterGradientAnnotation:
    """One per-archive master-gradient annotation for a downstream surface row.

    Frozen dataclass per CLAUDE.md "Beauty, simplicity, and developer
    experience" + Catalog #323 canonical-provenance contract pattern.

    Fields
    ------
    archive_sha256
        Hex sha256 of the scored archive the gradient was extracted on. This is
        the cross-correlation key between the wire-in annotation and the
        canonical master-gradient ledger at
        ``.omx/state/master_gradient_anchors.jsonl``.
    anchor_exists
        True iff at least one master-gradient anchor for ``archive_sha256``
        exists in the canonical ledger at annotation time.
    measurement_axis
        Lane tag from the latest anchor for ``archive_sha256`` (per CLAUDE.md
        "Apples-to-apples evidence discipline" — every score reference must
        carry an axis tag). ``None`` if anchor_exists is False.
    measurement_hardware
        Hardware substrate from the latest anchor. ``None`` if anchor_exists
        is False.
    measurement_utc
        ISO-UTC timestamp of the latest anchor. ``None`` if anchor_exists is
        False.
    is_authoritative_axis
        True iff the latest anchor's axis + hardware substrate combination is
        in the contest-authoritative set per Catalog #127 / CLAUDE.md
        "Submission auth eval" non-negotiable. macOS-CPU advisory rows + MPS
        rows resolve to False.
    annotation_utc
        ISO-UTC timestamp of this annotation (when the wire-in helper read the
        canonical ledger).
    """

    archive_sha256: str
    anchor_exists: bool
    measurement_axis: str | None = None
    measurement_hardware: str | None = None
    measurement_utc: str | None = None
    is_authoritative_axis: bool = False
    annotation_utc: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "schema": MASTER_GRADIENT_WIRE_IN_SCHEMA_VERSION,
            "archive_sha256": self.archive_sha256,
            "anchor_exists": self.anchor_exists,
            "measurement_axis": self.measurement_axis,
            "measurement_hardware": self.measurement_hardware,
            "measurement_utc": self.measurement_utc,
            "is_authoritative_axis": self.is_authoritative_axis,
            "annotation_utc": self.annotation_utc,
        }


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_archive_sha(archive_sha256: str) -> str:
    """Return canonical lowercase hex sha256 string."""
    if not isinstance(archive_sha256, str) or not archive_sha256.strip():
        raise ValueError("archive_sha256 must be a non-empty string")
    return archive_sha256.strip().lower()


def _resolve_latest_master_gradient_anchor(
    archive_sha256: str,
    *,
    ledger_path: Path | None = None,
) -> MasterGradientAnnotation:
    """Lookup the latest master-gradient anchor for ``archive_sha256``.

    Returns a ``MasterGradientAnnotation`` with ``anchor_exists=True`` and
    populated fields if an anchor is found; otherwise returns
    ``anchor_exists=False`` with ``measurement_*`` set to ``None``.

    Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
    fail-open behavior — missing ledger file / corrupt rows / lookup error
    all resolve to ``anchor_exists=False`` (annotation absence is honest;
    annotation overstatement is the bug class to prevent). Per Catalog #287
    docstring-overstatement trap.
    """
    normalized = _normalize_archive_sha(archive_sha256)
    annotation_utc = _now_iso()

    # Lazy import to avoid circular dependencies + keep module-load surface narrow.
    try:
        from tac.master_gradient import (
            is_authoritative_contest_axis_anchor,
            latest_anchor_for_archive,
        )
    except ImportError:
        return MasterGradientAnnotation(
            archive_sha256=normalized,
            anchor_exists=False,
            annotation_utc=annotation_utc,
        )

    # Per the canonical tac.master_gradient.latest_anchor_for_archive signature:
    #   latest_anchor_for_archive(archive_sha256: str, *, path: Path | None = None,
    #                              axis: str | None = None) -> dict | None
    # Anchors are returned as plain dicts (the canonical JSONL row schema),
    # not MasterGradient objects. We access fields via dict.get to honor the
    # canonical contract.
    try:
        anchor = latest_anchor_for_archive(normalized, path=ledger_path)
    except Exception:
        return MasterGradientAnnotation(
            archive_sha256=normalized,
            anchor_exists=False,
            annotation_utc=annotation_utc,
        )

    if anchor is None:
        return MasterGradientAnnotation(
            archive_sha256=normalized,
            anchor_exists=False,
            annotation_utc=annotation_utc,
        )

    try:
        is_authoritative = bool(is_authoritative_contest_axis_anchor(anchor))
    except Exception:
        is_authoritative = False

    # anchor is a dict (canonical JSONL row schema per
    # tac.master_gradient.append_anchor_locked).
    if isinstance(anchor, Mapping):
        measurement_axis = anchor.get("measurement_axis")
        measurement_hardware = anchor.get("measurement_hardware")
        measurement_utc = anchor.get("measurement_utc")
    else:
        # Defense-in-depth: support hypothetical object form via getattr fallback.
        measurement_axis = getattr(anchor, "measurement_axis", None)
        measurement_hardware = getattr(anchor, "measurement_hardware", None)
        measurement_utc = getattr(anchor, "measurement_utc", None)

    return MasterGradientAnnotation(
        archive_sha256=normalized,
        anchor_exists=True,
        measurement_axis=measurement_axis,
        measurement_hardware=measurement_hardware,
        measurement_utc=measurement_utc,
        is_authoritative_axis=is_authoritative,
        annotation_utc=annotation_utc,
    )


# --------------------------------------------------------------------------- #
# WIRE-IN #1: tac.frontier_scan annotation                                     #
# --------------------------------------------------------------------------- #


def annotate_frontier_anchors_with_master_gradient_existence(
    anchors: Sequence[Any],
    *,
    ledger_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Annotate each frontier ``Anchor`` with master-gradient existence flag.

    Consumer surface: ``tac.frontier_scan.build_frontier_scan_payload`` (or
    downstream operator-facing consumers like the autopilot ranker). The
    annotation lets bit-allocator and dispatch ranker prioritize archives
    that already have gradient anchors for paired-comparison ranking.

    The function does NOT mutate the canonical ``Anchor`` dataclass. It returns
    a list of dicts that combines the canonical anchor serialization with the
    master-gradient annotation. The original anchors remain immutable per
    CLAUDE.md "Operator gates must be wired and used" + HISTORICAL_PROVENANCE.

    Per CLAUDE.md "Apples-to-apples evidence discipline": the returned dicts
    carry the original anchor's axis + hardware + sha256 unchanged. The
    annotation ONLY adds the master-gradient existence flag and (when
    available) the latest anchor's axis + hardware + utc + authoritativeness.
    """
    annotated: list[dict[str, Any]] = []
    for anchor in anchors:
        archive_sha256 = getattr(anchor, "archive_sha256", None)
        anchor_payload: dict[str, Any] = {
            "score": getattr(anchor, "score", None),
            "axis": getattr(anchor, "axis", None),
            "archive_sha256": archive_sha256,
            "hardware_substrate": getattr(anchor, "hardware_substrate", None),
            "source_path": getattr(anchor, "source_path", None),
        }
        # Per the canonical Anchor.canonical_axis() contract:
        canonical_axis = getattr(anchor, "canonical_axis", None)
        if callable(canonical_axis):
            try:
                anchor_payload["canonical_axis"] = canonical_axis()
            except Exception:
                anchor_payload["canonical_axis"] = None
        # Per the canonical Anchor.is_qualifying() contract:
        is_qualifying = getattr(anchor, "is_qualifying", None)
        if callable(is_qualifying):
            try:
                anchor_payload["is_qualifying"] = bool(is_qualifying())
            except Exception:
                anchor_payload["is_qualifying"] = False

        if archive_sha256:
            annotation = _resolve_latest_master_gradient_anchor(
                archive_sha256, ledger_path=ledger_path
            )
            anchor_payload["master_gradient_annotation"] = annotation.as_dict()
        else:
            anchor_payload["master_gradient_annotation"] = MasterGradientAnnotation(
                archive_sha256="",
                anchor_exists=False,
                annotation_utc=_now_iso(),
            ).as_dict()

        annotated.append(anchor_payload)

    return annotated


# --------------------------------------------------------------------------- #
# WIRE-IN #2: tac.probe_outcomes_ledger annotation                             #
# --------------------------------------------------------------------------- #


def register_probe_outcome_with_master_gradient_anchor(
    *,
    probe_id: str,
    substrate: str,
    recipe_path: str | None,
    probe_kind: str,
    verdict: str,
    metric_name: str,
    metric_value: float,
    archive_sha256: str | None = None,
    master_gradient_ledger_path: Path | None = None,
    probe_outcomes_ledger_path: Path | None = None,
    probe_outcomes_lock_path: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Register a probe outcome with master-gradient anchor cross-correlation.

    Canonical wrapper around
    ``tac.probe_outcomes_ledger.register_probe_outcome`` that threads
    ``master_gradient_anchor_archive_sha256`` and
    ``master_gradient_anchor_present`` through the canonical ``extra`` channel
    when ``archive_sha256`` is provided.

    Per CLAUDE.md "Probe-outcomes ledger" canonical 4-layer pattern (Catalog
    #313): the extra fields are non-mutating (don't collide with reserved
    schema fields), append-only, and survive HISTORICAL_PROVENANCE per
    Catalog #110 / #113.

    Per CLAUDE.md "Apples-to-apples evidence discipline": the annotation does
    NOT promote a probe outcome or change verdict semantics. It ONLY adds
    cross-correlation metadata for post-hoc queries.

    Returns the appended probe-outcome ledger record (with the master-gradient
    annotation fields embedded via the canonical extra channel).
    """
    from tac.probe_outcomes_ledger import register_probe_outcome

    master_gradient_extra: dict[str, Any] = {}
    if archive_sha256:
        annotation = _resolve_latest_master_gradient_anchor(
            archive_sha256, ledger_path=master_gradient_ledger_path
        )
        master_gradient_extra = {
            "master_gradient_anchor_archive_sha256": annotation.archive_sha256,
            "master_gradient_anchor_present": annotation.anchor_exists,
            "master_gradient_anchor_axis": annotation.measurement_axis,
            "master_gradient_anchor_hardware": annotation.measurement_hardware,
            "master_gradient_anchor_utc": annotation.measurement_utc,
            "master_gradient_anchor_is_authoritative": annotation.is_authoritative_axis,
        }

    # Merge caller extras (caller wins on key collision per Python dict semantics,
    # but the reserved-field collision check inside register_probe_outcome
    # still applies — annotation keys are intentionally non-reserved).
    extra: dict[str, Any] = {**master_gradient_extra, **kwargs}

    return register_probe_outcome(
        probe_id=probe_id,
        substrate=substrate,
        recipe_path=recipe_path,
        probe_kind=probe_kind,
        verdict=verdict,
        metric_name=metric_name,
        metric_value=metric_value,
        path=probe_outcomes_ledger_path,
        lock_path=probe_outcomes_lock_path,
        **extra,
    )


# --------------------------------------------------------------------------- #
# WIRE-IN #3: tac.continual_learning posterior annotation                      #
# --------------------------------------------------------------------------- #


def annotate_posterior_row_with_master_gradient_anchor(
    posterior_row: Mapping[str, Any],
    *,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Return a copy of ``posterior_row`` annotated with master-gradient anchor.

    Consumer surface: callers of
    ``tac.continual_learning.posterior_update_locked`` that want to annotate
    the ``accepted_anchor_history`` row with master-gradient anchor cross-
    correlation metadata.

    The annotation reads ``archive_sha256`` from the posterior row (per
    canonical ``ContestResult.archive_sha256`` contract) and looks up the
    latest master-gradient anchor for that archive in the canonical ledger.

    Per CLAUDE.md "HISTORICAL_PROVENANCE non-negotiable" + Catalog #110 /
    #113: the input ``posterior_row`` is NOT mutated. A copy is returned with
    the annotation fields embedded under a single nested key
    ``master_gradient_annotation`` so the canonical posterior schema is not
    polluted with parallel toplevel fields.

    Per CLAUDE.md "Apples-to-apples evidence discipline": annotation does NOT
    change posterior score / axis / evidence_tag / hardware fields. It ONLY
    adds cross-correlation metadata for post-hoc queries.
    """
    if not isinstance(posterior_row, Mapping):
        raise TypeError(
            f"posterior_row must be a Mapping; got {type(posterior_row).__name__}"
        )

    row_copy: dict[str, Any] = dict(posterior_row)
    archive_sha256 = posterior_row.get("archive_sha256")
    if isinstance(archive_sha256, str) and archive_sha256.strip():
        annotation = _resolve_latest_master_gradient_anchor(
            archive_sha256, ledger_path=ledger_path
        )
        row_copy["master_gradient_annotation"] = annotation.as_dict()
    else:
        row_copy["master_gradient_annotation"] = MasterGradientAnnotation(
            archive_sha256="",
            anchor_exists=False,
            annotation_utc=_now_iso(),
        ).as_dict()
    return row_copy


# --------------------------------------------------------------------------- #
# WIRE-IN #4: tac.deploy.modal.call_id_ledger annotation                       #
# --------------------------------------------------------------------------- #


def register_dispatched_call_id_with_master_gradient_anchor(
    *,
    call_id: str,
    lane_id: str,
    label: str,
    archive_sha256: str | None = None,
    master_gradient_ledger_path: Path | None = None,
    call_id_ledger_path: Path | None = None,
    call_id_lock_path: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Register a Modal call_id with master-gradient anchor cross-correlation.

    Canonical wrapper around
    ``tac.deploy.modal.call_id_ledger.register_dispatched_call_id`` that
    threads ``master_gradient_anchor_archive_sha256`` and
    ``master_gradient_anchor_present`` through the canonical ``extra`` channel
    when ``archive_sha256`` is provided.

    Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" non-negotiable + Catalog
    #245 canonical 4-layer pattern: the extra fields are non-mutating (don't
    collide with reserved schema fields), append-only, and survive
    HISTORICAL_PROVENANCE per Catalog #110 / #113.

    Per CLAUDE.md "Apples-to-apples evidence discipline": the annotation does
    NOT promote a Modal dispatch or change score semantics. It ONLY adds
    cross-correlation metadata so a harvested call_id can be apples-to-apples
    cross-checked against the per-archive master-gradient anchor it was
    dispatched to refine.

    Returns the appended Modal call_id ledger record (with the master-gradient
    annotation fields embedded via the canonical extra channel).
    """
    from tac.deploy.modal.call_id_ledger import register_dispatched_call_id

    master_gradient_extra: dict[str, Any] = {}
    if archive_sha256:
        annotation = _resolve_latest_master_gradient_anchor(
            archive_sha256, ledger_path=master_gradient_ledger_path
        )
        master_gradient_extra = {
            "master_gradient_anchor_archive_sha256": annotation.archive_sha256,
            "master_gradient_anchor_present": annotation.anchor_exists,
            "master_gradient_anchor_axis": annotation.measurement_axis,
            "master_gradient_anchor_hardware": annotation.measurement_hardware,
            "master_gradient_anchor_utc": annotation.measurement_utc,
            "master_gradient_anchor_is_authoritative": annotation.is_authoritative_axis,
        }

    extra: dict[str, Any] = {**master_gradient_extra, **kwargs}

    return register_dispatched_call_id(
        call_id=call_id,
        lane_id=lane_id,
        label=label,
        path=call_id_ledger_path,
        lock_path=call_id_lock_path,
        **extra,
    )


# --------------------------------------------------------------------------- #
# Post-hoc query + coverage helpers                                            #
# --------------------------------------------------------------------------- #


def query_anchors_with_master_gradient_coverage(
    archive_sha256s: Sequence[str],
    *,
    ledger_path: Path | None = None,
) -> dict[str, MasterGradientAnnotation]:
    """Return per-archive master-gradient anchor existence map.

    Useful for batch consumers (autopilot ranker, dispatch planner, frontier
    scan output renderer) that need to know which archives have gradient
    anchors materialized vs which are still pending extraction.

    Per CLAUDE.md "Apples-to-apples evidence discipline": the returned
    annotations carry the master-gradient anchor's own axis + hardware
    metadata, NOT the queried archive's. Consumers must cross-reference
    independently.
    """
    out: dict[str, MasterGradientAnnotation] = {}
    for archive_sha256 in archive_sha256s:
        normalized = _normalize_archive_sha(archive_sha256)
        out[normalized] = _resolve_latest_master_gradient_anchor(
            normalized, ledger_path=ledger_path
        )
    return out


def compute_master_gradient_wire_in_coverage(
    archive_sha256s: Sequence[str],
    *,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Compute master-gradient wire-in coverage for a set of archives.

    Returns a structured payload with:
    - ``total_archives``
    - ``archives_with_anchor`` (count)
    - ``archives_without_anchor`` (count)
    - ``archives_with_authoritative_anchor`` (count of anchors whose axis is
      contest-authoritative per Catalog #127)
    - ``coverage_pct`` (percentage of archives with any anchor)
    - ``authoritative_coverage_pct`` (percentage with contest-axis anchor)
    - ``per_archive`` (per-archive annotation dict)

    Per the inventory memo's "A1.1 Materialized anchors" finding: only 1 of 8
    frontier archives has a per-pair fp64 gradient anchor (12.5% coverage at
    landing). This helper is the canonical surface for the autopilot ranker
    + frontier scan + operator briefing to consume per-archive wire-in
    coverage without re-reading the ledger.
    """
    per_archive = query_anchors_with_master_gradient_coverage(
        archive_sha256s, ledger_path=ledger_path
    )

    total = len(per_archive)
    with_anchor = sum(1 for a in per_archive.values() if a.anchor_exists)
    without_anchor = total - with_anchor
    with_authoritative = sum(
        1 for a in per_archive.values() if a.anchor_exists and a.is_authoritative_axis
    )

    coverage_pct = (with_anchor / total * 100.0) if total > 0 else 0.0
    authoritative_pct = (
        (with_authoritative / total * 100.0) if total > 0 else 0.0
    )

    return {
        "schema": MASTER_GRADIENT_WIRE_IN_SCHEMA_VERSION,
        "computed_at_utc": _now_iso(),
        "computed_pid": os.getpid(),
        "computed_host": socket.gethostname(),
        "total_archives": total,
        "archives_with_anchor": with_anchor,
        "archives_without_anchor": without_anchor,
        "archives_with_authoritative_anchor": with_authoritative,
        "coverage_pct": coverage_pct,
        "authoritative_coverage_pct": authoritative_pct,
        "per_archive": {sha: anno.as_dict() for sha, anno in per_archive.items()},
    }
