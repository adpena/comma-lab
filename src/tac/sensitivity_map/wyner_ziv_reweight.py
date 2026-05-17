# SPDX-License-Identifier: MIT
"""Wyner-Ziv side-info covariance → per-byte sensitivity-map reweighting.

This is the canonical wire-in for Catalog #125 hook #1 (sensitivity-map
contribution) from
:func:`tac.master_gradient_consumers.wyner_ziv_side_info_covariance`. The
producer classifies archive bytes by cross-pair gradient correlation into
three disjoint sets:

* ``candidate_shared_prior_byte_indices`` — bytes whose per-pair gradient
  patterns are HIGHLY correlated across the byte set (pair-INVARIANT signal).
  Per Wyner-Ziv 1976 these bytes are HOISTABLE to a shared side-information
  prior the decoder can reconstruct without paying archive rate. Byte-level
  perturbation here is structurally recoverable, so the per-byte sensitivity
  of the contest score to direct mutation is REDUCED.
* ``pair_specific_byte_indices`` — bytes whose per-pair vectors are
  uncorrelated; they carry pair-SPECIFIC signal that no side-info channel
  can reconstruct. Byte-level perturbation directly hits the score, so the
  per-byte sensitivity is INCREASED.
* ``mixed_byte_indices`` — bytes whose correlation is between the high and
  low thresholds; the side-info classification is indeterminate, so
  sensitivity stays at the baseline.

This module exposes the canonical per-byte reweighting API that the
:mod:`tac.master_gradient_consumers` docstring lines 75-76 documents as the
hook #1 surface:

    "consumers 4 (Wyner-Ziv covariance), 5 (per-pair difficulty) feed
    `tac.sensitivity_map.axis_level_reweight`."

The sister per-byte difficulty surface (consumer 5) will land in a follow-on
helper. This module is intentionally narrow to the Wyner-Ziv contribution so
the wire-in is per-consumer-traceable.

Apples-to-apples evidence discipline
====================================

Per CLAUDE.md "Apples-to-apples evidence discipline" the reweighted sensitivity
dict is a PLANNING-side EV multiplier; it is NOT a score claim. Downstream
consumers (autopilot ranker, bit allocator, Pareto solver) MUST propagate the
``operating_point_tag = "wyner_ziv_side_info_covariance_v1"`` provenance into
their own evidence strings; no axis label promotion is implied.

Cross-references
----------------
- :mod:`tac.master_gradient_consumers` — canonical producer of the
  classification dataclass (consumer 4 / WynerZivSideInfoClassification).
- :mod:`tac.sensitivity_map.axis_weights` — orthogonal per-AXIS EV multipliers
  (scalar per pose/seg/rate/mixed). The per-axis surface is unaffected by
  this per-byte surface.
- CLAUDE.md "Subagent coherence-by-default" (Mandatory wire-in for every
  landing) item 1 (Sensitivity-map contribution).
- Catalog #125 (`check_subagent_landing_has_solver_wire_in`).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import lazily inside helpers to avoid a hard ``np`` requirement at
    # module-import time for callers that only want to consume an already-
    # built classification (the producer side already requires numpy).
    from tac.master_gradient_consumers import WynerZivSideInfoClassification

__all__ = [
    "MIXED_SENSITIVITY_BASELINE",
    "PAIR_SPECIFIC_SENSITIVITY_UPWEIGHT",
    "SHARED_PRIOR_SENSITIVITY_DOWNWEIGHT",
    "WYNER_ZIV_REWEIGHT_OPERATING_POINT_TAG",
    "WynerZivAxisLevelReweightError",
    "axis_level_reweight",
    "update_sensitivity_map_from_master_gradient_anchor",
]

#: Bias factor applied to bytes classified as ``candidate_shared_prior_*``.
#: These bytes are HOISTABLE to a Wyner-Ziv side-information channel; the
#: decoder can reconstruct them without paying archive rate, so the per-byte
#: sensitivity to direct mutation is reduced by 10x relative to the baseline.
#: The 0.1 multiplier is the canonical "shared-prior eligibility downweight"
#: per the Wyner-Ziv 1976 R(D|Y) reduction (one order of magnitude reflects
#: the typical I(X;Y) compression-gain regime for dashcam-derived priors;
#: callers can override via ``axis_level_reweight(..., downweight=...)``).
SHARED_PRIOR_SENSITIVITY_DOWNWEIGHT: float = 0.1

#: Bias factor applied to bytes classified as ``pair_specific_*``. These
#: bytes carry pair-specific signal that no side-info channel can reconstruct;
#: direct mutation propagates straight to the contest score. The 2.0
#: multiplier doubles the per-byte sensitivity relative to the baseline,
#: biasing downstream bit-allocators / autopilot rankers to protect these
#: bytes harder.
PAIR_SPECIFIC_SENSITIVITY_UPWEIGHT: float = 2.0

#: Baseline sensitivity weight for bytes whose correlation falls between the
#: high and low thresholds (mixed) OR is out of any classified set. The
#: helper preserves this baseline so callers that pass a pre-existing
#: ``base_weights`` dict see only the SHARED / PAIR-SPECIFIC bias factors
#: applied at the classified indices.
MIXED_SENSITIVITY_BASELINE: float = 1.0

#: Canonical operating-point provenance tag downstream consumers must embed
#: in their evidence strings per CLAUDE.md "Apples-to-apples evidence
#: discipline" + Catalog #287. The tag is structural; do NOT bare-string it
#: from downstream sites — import this constant.
WYNER_ZIV_REWEIGHT_OPERATING_POINT_TAG: str = "wyner_ziv_side_info_covariance_v1"


class WynerZivAxisLevelReweightError(ValueError):
    """Raised when the per-byte reweighting inputs are malformed."""


def _validate_classification(classification: "WynerZivSideInfoClassification") -> None:
    """Validate the producer dataclass is well-formed before reweighting."""
    # Import lazily so this module is importable without numpy.
    from tac.master_gradient_consumers import WynerZivSideInfoClassification

    if not isinstance(classification, WynerZivSideInfoClassification):
        raise WynerZivAxisLevelReweightError(
            "classification must be a WynerZivSideInfoClassification; "
            f"got {type(classification).__name__}"
        )
    if classification.n_bytes < 0:
        raise WynerZivAxisLevelReweightError(
            f"classification.n_bytes must be non-negative; got {classification.n_bytes}"
        )
    for label, indices in (
        ("candidate_shared_prior_byte_indices", classification.candidate_shared_prior_byte_indices),
        ("pair_specific_byte_indices", classification.pair_specific_byte_indices),
        ("mixed_byte_indices", classification.mixed_byte_indices),
    ):
        for idx in indices:
            if not isinstance(idx, int) or isinstance(idx, bool):
                raise WynerZivAxisLevelReweightError(
                    f"{label} contains non-int entry {idx!r}"
                )
            if idx < 0:
                raise WynerZivAxisLevelReweightError(
                    f"{label} contains negative byte index {idx}"
                )
            if classification.n_bytes > 0 and idx >= classification.n_bytes:
                raise WynerZivAxisLevelReweightError(
                    f"{label} entry {idx} >= n_bytes={classification.n_bytes}"
                )
    # The producer guarantees disjoint sets; verify defensively because a
    # corrupt dataclass (e.g. hand-built fixture) is silently wrong and
    # would produce multiplied biases on overlap.
    shared_set = set(classification.candidate_shared_prior_byte_indices)
    specific_set = set(classification.pair_specific_byte_indices)
    mixed_set = set(classification.mixed_byte_indices)
    overlap_sp = shared_set & specific_set
    overlap_sm = shared_set & mixed_set
    overlap_pm = specific_set & mixed_set
    if overlap_sp or overlap_sm or overlap_pm:
        raise WynerZivAxisLevelReweightError(
            "classification byte-index sets must be disjoint; overlap detected: "
            f"shared∩specific={sorted(overlap_sp)[:5]}, "
            f"shared∩mixed={sorted(overlap_sm)[:5]}, "
            f"specific∩mixed={sorted(overlap_pm)[:5]}"
        )


def _validate_bias_factor(value: float, name: str) -> float:
    """Coerce-and-validate a bias factor; non-negative finite float."""
    try:
        fvalue = float(value)
    except (TypeError, ValueError) as exc:
        raise WynerZivAxisLevelReweightError(
            f"{name} must be coercible to float; got {value!r}"
        ) from exc
    if math.isnan(fvalue) or math.isinf(fvalue):
        raise WynerZivAxisLevelReweightError(
            f"{name} must be finite; got {fvalue!r}"
        )
    if fvalue < 0.0:
        raise WynerZivAxisLevelReweightError(
            f"{name} must be non-negative; got {fvalue!r}"
        )
    return fvalue


def axis_level_reweight(
    classification: "WynerZivSideInfoClassification",
    *,
    base_weights: Mapping[int, float] | None = None,
    downweight: float = SHARED_PRIOR_SENSITIVITY_DOWNWEIGHT,
    upweight: float = PAIR_SPECIFIC_SENSITIVITY_UPWEIGHT,
    baseline: float = MIXED_SENSITIVITY_BASELINE,
) -> dict[int, float]:
    """Apply the Wyner-Ziv classification to per-byte sensitivity weights.

    The returned mapping is keyed by integer byte index in ``[0, n_bytes)``.
    Each shared-prior byte is downweighted by ``downweight`` (default 0.1;
    10x lower sensitivity since the byte is recoverable via side-info), each
    pair-specific byte is upweighted by ``upweight`` (default 2.0; 2x higher
    sensitivity since direct mutation hits the score), and mixed / unclassified
    bytes keep ``baseline`` (default 1.0).

    Parameters
    ----------
    classification
        A
        :class:`tac.master_gradient_consumers.WynerZivSideInfoClassification`
        dataclass produced by
        :func:`tac.master_gradient_consumers.wyner_ziv_side_info_covariance`.
        The three byte-index sets must be disjoint (enforced by the producer
        and re-checked defensively here).
    base_weights
        Optional pre-existing per-byte weights. If supplied, the returned
        dict is the COPY of ``base_weights`` with bias factors MULTIPLIED at
        the classified indices (i.e. ``new[i] = base[i] * downweight`` for
        shared-prior bytes). Keys outside ``[0, n_bytes)`` are preserved
        as-is; missing keys in ``[0, n_bytes)`` default to ``baseline``.
        If ``None``, the helper builds a fresh dict for every byte in
        ``[0, n_bytes)`` initialised at ``baseline`` and applies the bias.
    downweight
        Per-byte multiplier for shared-prior bytes; default 0.1.
    upweight
        Per-byte multiplier for pair-specific bytes; default 2.0.
    baseline
        Default per-byte weight when ``base_weights`` is None or a key is
        missing; default 1.0.

    Returns
    -------
    ``dict[int, float]``
        Per-byte sensitivity weights with the Wyner-Ziv bias applied.
        Determinism: the same inputs always produce the same output dict
        (dict insertion order is byte-index ascending in the fresh-base
        case; the ``base_weights`` insertion order is preserved otherwise
        with classified indices updated in place).

    Raises
    ------
    WynerZivAxisLevelReweightError
        If the classification is malformed, a bias factor is non-finite or
        negative, or a ``base_weights`` value is non-finite.
    """
    _validate_classification(classification)
    downweight_f = _validate_bias_factor(downweight, "downweight")
    upweight_f = _validate_bias_factor(upweight, "upweight")
    baseline_f = _validate_bias_factor(baseline, "baseline")

    if base_weights is None:
        result: dict[int, float] = {
            i: baseline_f for i in range(int(classification.n_bytes))
        }
    else:
        # Coerce + validate base_weights values; copy into a fresh dict to
        # preserve caller's input.
        result = {}
        for key, value in base_weights.items():
            if not isinstance(key, int) or isinstance(key, bool):
                raise WynerZivAxisLevelReweightError(
                    f"base_weights key must be int; got {key!r}"
                )
            try:
                fvalue = float(value)
            except (TypeError, ValueError) as exc:
                raise WynerZivAxisLevelReweightError(
                    f"base_weights[{key}] must be coercible to float; got {value!r}"
                ) from exc
            if math.isnan(fvalue) or math.isinf(fvalue):
                raise WynerZivAxisLevelReweightError(
                    f"base_weights[{key}] must be finite; got {fvalue!r}"
                )
            if fvalue < 0.0:
                raise WynerZivAxisLevelReweightError(
                    f"base_weights[{key}] must be non-negative; got {fvalue!r}"
                )
            result[key] = fvalue
        # Backfill missing keys in [0, n_bytes) with baseline so the returned
        # dict covers the full byte range (downstream consumers expect a
        # dense per-byte map even when caller's base_weights was sparse).
        for i in range(int(classification.n_bytes)):
            if i not in result:
                result[i] = baseline_f

    # Apply bias factors at the classified indices. The producer guarantees
    # the three sets are disjoint, so per-index multiplication is unambiguous.
    for idx in classification.candidate_shared_prior_byte_indices:
        if idx in result:
            result[idx] = result[idx] * downweight_f
    for idx in classification.pair_specific_byte_indices:
        if idx in result:
            result[idx] = result[idx] * upweight_f
    # mixed bytes are intentionally NOT touched (baseline preserved).

    return result


def update_sensitivity_map_from_master_gradient_anchor(
    *,
    archive_sha256: str | None = None,
    sample_axis: int = 1,
    base_weights: Mapping[int, float] | None = None,
    downweight: float = SHARED_PRIOR_SENSITIVITY_DOWNWEIGHT,
    upweight: float = PAIR_SPECIFIC_SENSITIVITY_UPWEIGHT,
    baseline: float = MIXED_SENSITIVITY_BASELINE,
    write_sidecar: bool = False,
) -> dict[int, float]:
    """Single-call end-to-end Wyner-Ziv → per-byte sensitivity reweighting.

    Loads the canonical per-pair master gradient anchor from
    ``.omx/state/master_gradient_anchors.jsonl`` (optionally filtered by
    archive SHA), runs the Wyner-Ziv covariance classifier, and applies the
    bias factors via :func:`axis_level_reweight`. The full round trip from
    archive bytes to reweighted sensitivity dict is one call.

    Parameters
    ----------
    archive_sha256
        Optional archive SHA-256 filter. If ``None``, uses the most-recent
        per-pair anchor in the ledger regardless of archive.
    sample_axis
        Score axis to use for the cross-pair correlation (0=seg, 1=pose,
        2=rate; default 1=pose per the producer's default which is richer
        signal at the PR106 frontier operating point per CLAUDE.md "SegNet vs
        PoseNet importance — operating-point dependent").
    base_weights, downweight, upweight, baseline
        Forwarded to :func:`axis_level_reweight`.
    write_sidecar
        Forwarded to the underlying
        :func:`tac.master_gradient_consumers.wyner_ziv_side_info_covariance`
        producer.

    Returns
    -------
    ``dict[int, float]``
        Per-byte sensitivity weights with the Wyner-Ziv bias applied.

    Raises
    ------
    FileNotFoundError
        If no per-pair anchor exists for the requested archive.
    WynerZivAxisLevelReweightError
        If the producer returns a malformed classification (this is a
        defensive check; the producer itself enforces the contract).
    """
    # Import lazily so this module is importable without numpy/torch on
    # CLI-only consumers.
    from tac.master_gradient_consumers import (
        load_per_pair_gradient_from_anchor,
        wyner_ziv_side_info_covariance,
    )

    gradient, anchor = load_per_pair_gradient_from_anchor(
        archive_sha256=archive_sha256,
    )
    anchor_archive_sha256 = str(anchor.get("archive_sha256", archive_sha256 or ""))
    measurement_axis = str(anchor.get("measurement_axis", ""))
    measurement_hardware = str(anchor.get("measurement_hardware", ""))

    classification = wyner_ziv_side_info_covariance(
        gradient,
        archive_sha256=anchor_archive_sha256,
        measurement_axis=measurement_axis,
        measurement_hardware=measurement_hardware,
        sample_axis=sample_axis,
        write_sidecar=write_sidecar,
    )

    return axis_level_reweight(
        classification,
        base_weights=base_weights,
        downweight=downweight,
        upweight=upweight,
        baseline=baseline,
    )
