# SPDX-License-Identifier: MIT
"""Fail-closed admission for the shared PDW2/realization receiver.

This module does not claim to be the missing receiver.  It is the typed
boundary at which the three distinct objects must meet before the production
archive may call PDW2 ``receiver_consumed``:

* strict PDW2 bytes plus an explicit spatial quotient field;
* a scorer-free spatial/RGB pullback whose packet mutation changes output;
* an exact n600 archive and hard CPU-Torch through-R row.

The current receipts deliberately produce a measured, formulation-scoped
blocker.  Missing distortion authority is ``None`` and is never coerced to
zero.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final

SCHEMA: Final = "shared_receiver_admission.v1"
BLOCKER_ID: Final = "SHARED_RECEIVER_COUNTED_SPATIAL_HARD_ORACLE_INTERSECTION_EMPTY"
UNTESTED_OPTIMAL_FORM: Final = (
    "counted_curvelet_or_shearlet_spatial_generator_jointly_solved_with_"
    "active_set_hard_oracle_preimage_and_xi_pose_factor"
)
PAIR_COUNT: Final = 600
MAX_BYTES_PER_PAIR: Final = 477.8
MAX_ARCHIVE_BYTES: Final = 286_680
MAX_D_SEG: Final = 3.39e-4
SCORE_BYTES_NORMALIZER: Final = 37_545_489
RATE_PRICE_PER_BYTE: Final = 25.0 / SCORE_BYTES_NORMALIZER


class SharedReceiverAdmissionError(ValueError):
    """Raised when evidence is malformed or crosses an authority type."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SharedReceiverAdmissionError(f"{name} must be a mapping")
    return value


def _field(row: Mapping[str, Any], key: str, name: str) -> Any:
    if key not in row:
        raise SharedReceiverAdmissionError(f"{name}.{key} is required")
    return row[key]


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SharedReceiverAdmissionError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SharedReceiverAdmissionError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise SharedReceiverAdmissionError(f"{name} must be >= {minimum}")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SharedReceiverAdmissionError(f"{name} must be an integer >= {minimum}")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise SharedReceiverAdmissionError(f"{name} must be boolean")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise SharedReceiverAdmissionError(f"{name} must be a lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise SharedReceiverAdmissionError(f"{name} must be hexadecimal") from exc
    if value.lower() != value:
        raise SharedReceiverAdmissionError(f"{name} must be lowercase")
    return value


def _git_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise SharedReceiverAdmissionError(f"{name} must be a 40-character git SHA")
    try:
        int(value, 16)
    except ValueError as exc:
        raise SharedReceiverAdmissionError(f"{name} must be hexadecimal") from exc
    if value.lower() != value:
        raise SharedReceiverAdmissionError(f"{name} must be lowercase")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SharedReceiverAdmissionError(f"{name} must be a non-empty string")
    return value


def _share(value: Any, name: str) -> float:
    result = _finite(value, name, minimum=0.0)
    if result > 1.0:
        raise SharedReceiverAdmissionError(f"{name} must be <= 1.0")
    return result


def _validate_pdw2_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if _field(receipt, "schema", "pdw2") != "pdw2_spatial_receiver_blocker_receipt.v1":
        raise SharedReceiverAdmissionError("pdw2 receipt schema mismatch")
    authority = _mapping(_field(receipt, "authority", "pdw2"), "pdw2.authority")
    if _bool(_field(authority, "through_r_authority", "pdw2.authority"), "pdw2.authority.through_r_authority"):
        raise SharedReceiverAdmissionError("the blocker receipt cannot carry through-R authority")
    if (
        _field(authority, "d_seg", "pdw2.authority") is not None
        or _field(authority, "d_pose", "pdw2.authority") is not None
    ):
        raise SharedReceiverAdmissionError("target-only PDW2 distortions must be null")

    packet = _mapping(_field(receipt, "packet", "pdw2"), "pdw2.packet")
    quotient = _mapping(_field(receipt, "quotient_field", "pdw2"), "pdw2.quotient_field")
    n600 = _mapping(_field(receipt, "n600", "pdw2"), "pdw2.n600")
    through_r = _mapping(_field(receipt, "through_r_debt", "pdw2"), "pdw2.through_r_debt")
    if not _bool(
        _field(packet, "packet_to_partition_consumed", "pdw2.packet"),
        "pdw2.packet.packet_to_partition_consumed",
    ):
        raise SharedReceiverAdmissionError("PDW2 packet-consumption canary is required")
    if _integer(_field(n600, "pair_count", "pdw2.n600"), "pdw2.n600.pair_count", minimum=1) != PAIR_COUNT:
        raise SharedReceiverAdmissionError("PDW2 spatial receipt must cover n600")
    pixel_count = _integer(
        _field(n600, "pixel_count", "pdw2.n600"),
        "pdw2.n600.pixel_count",
        minimum=1,
    )
    raw_class_counts = _field(n600, "partition_label_counts_by_class_0_to_4", "pdw2.n600")
    if not isinstance(raw_class_counts, list) or len(raw_class_counts) != 5:
        raise SharedReceiverAdmissionError("pdw2.n600.partition_label_counts_by_class_0_to_4 must have five entries")
    class_counts = [
        _integer(
            value,
            f"pdw2.n600.partition_label_counts_by_class_0_to_4[{index}]",
        )
        for index, value in enumerate(raw_class_counts)
    ]
    if sum(class_counts) != pixel_count:
        raise SharedReceiverAdmissionError("PDW2 partition class counts must sum to pixel_count")
    if _bool(
        _field(through_r, "scorer_free_rgb_camera_pullback_present", "pdw2.through_r_debt"),
        "pdw2.through_r_debt.scorer_free_rgb_camera_pullback_present",
    ):
        raise SharedReceiverAdmissionError("blocker receipt unexpectedly claims an RGB pullback")
    return {
        "packet_raw_bytes": _integer(_field(packet, "raw_bytes", "pdw2.packet"), "pdw2.packet.raw_bytes", minimum=1),
        "packet_brotli_bytes": _integer(
            _field(packet, "brotli_bytes", "pdw2.packet"),
            "pdw2.packet.brotli_bytes",
            minimum=1,
        ),
        "packet_sha256": _sha256(_field(packet, "raw_sha256", "pdw2.packet"), "pdw2.packet.raw_sha256"),
        "quotient_file_bytes": _integer(
            _field(quotient, "file_bytes", "pdw2.quotient_field"),
            "pdw2.quotient_field.file_bytes",
            minimum=1,
        ),
        "quotient_file_sha256": _sha256(
            _field(quotient, "file_sha256", "pdw2.quotient_field"),
            "pdw2.quotient_field.file_sha256",
        ),
        "partition_labels_sha256": _sha256(
            _field(n600, "partition_labels_sha256", "pdw2.n600"),
            "pdw2.n600.partition_labels_sha256",
        ),
        "partition_pixel_count": pixel_count,
        "partition_label_counts_by_class_0_to_4": class_counts,
        "through_r_authority": False,
    }


def _validate_pdw1_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if _field(receipt, "schema", "pdw1") != "pdw1_fp32_realization_first_inbox_point.v1":
        raise SharedReceiverAdmissionError("pdw1 receipt schema mismatch")
    point = _mapping(
        _field(receipt, "phase_c_first_inbox_point", "pdw1"),
        "pdw1.phase_c_first_inbox_point",
    )
    payload = _mapping(_field(point, "payload", "pdw1.point"), "pdw1.point.payload")
    projection = _mapping(
        _field(point, "n600_extrapolation", "pdw1.point"),
        "pdw1.point.n600_extrapolation",
    )
    decomposition = _mapping(_field(point, "decomposition", "pdw1.point"), "pdw1.point.decomposition")
    n_pairs = _integer(_field(point, "n_pairs", "pdw1.point"), "pdw1.point.n_pairs", minimum=1)
    if n_pairs != 24:
        raise SharedReceiverAdmissionError("settled PDW1 realization receipt must be n24")
    d_a = _finite(_field(point, "d_A", "pdw1.point"), "pdw1.point.d_A", minimum=0.0)
    if d_a != 0.0:
        raise SharedReceiverAdmissionError("PDW1 encoding leg is expected to be exact")

    mismatch_pixels = _integer(
        _field(decomposition, "mismatch_total_px", "pdw1.point.decomposition"),
        "pdw1.point.decomposition.mismatch_total_px",
    )
    raw_confusion_rows = _field(decomposition, "confusion_rows_lstar_to_pred", "pdw1.point.decomposition")
    if not isinstance(raw_confusion_rows, list) or not raw_confusion_rows:
        raise SharedReceiverAdmissionError("pdw1 confusion decomposition must be non-empty")
    confusion_rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw_confusion_rows):
        row = _mapping(raw_row, f"pdw1.confusion[{index}]")
        confusion_rows.append(
            {
                "lstar": _nonempty_string(
                    _field(row, "lstar", f"pdw1.confusion[{index}]"),
                    f"pdw1.confusion[{index}].lstar",
                ),
                "pred": _nonempty_string(
                    _field(row, "pred", f"pdw1.confusion[{index}]"),
                    f"pdw1.confusion[{index}].pred",
                ),
                "px": _integer(
                    _field(row, "px", f"pdw1.confusion[{index}]"),
                    f"pdw1.confusion[{index}].px",
                    minimum=1,
                ),
                "share": _share(
                    _field(row, "share", f"pdw1.confusion[{index}]"),
                    f"pdw1.confusion[{index}].share",
                ),
            }
        )
    if sum(row["px"] for row in confusion_rows) != mismatch_pixels:
        raise SharedReceiverAdmissionError("pdw1 confusion decomposition must sum to mismatch_total_px")

    raw_boundary_rows = _mapping(
        _field(
            decomposition,
            "mismatch_within_chebyshev_r_of_lstar_boundary",
            "pdw1.point.decomposition",
        ),
        "pdw1.point.decomposition.mismatch_within_chebyshev_r_of_lstar_boundary",
    )
    boundary_proximity: dict[str, dict[str, Any]] = {}
    for radius in ("1", "2", "4"):
        row = _mapping(
            _field(raw_boundary_rows, radius, "pdw1.boundary_proximity"),
            f"pdw1.boundary_proximity.{radius}",
        )
        pixels = _integer(
            _field(row, "px", f"pdw1.boundary_proximity.{radius}"),
            f"pdw1.boundary_proximity.{radius}.px",
        )
        if pixels > mismatch_pixels:
            raise SharedReceiverAdmissionError("pdw1 boundary-proximity count cannot exceed total mismatches")
        boundary_proximity[radius] = {
            "px": pixels,
            "share": _share(
                _field(row, "share", f"pdw1.boundary_proximity.{radius}"),
                f"pdw1.boundary_proximity.{radius}.share",
            ),
        }

    payload_total = _integer(
        _field(payload, "total_bytes_n24", "pdw1.point.payload"),
        "pdw1.point.payload.total_bytes_n24",
        minimum=1,
    )
    payload_header = _integer(
        _field(payload, "header_bytes", "pdw1.point.payload"),
        "pdw1.point.payload.header_bytes",
    )
    payload_labels = _integer(
        _field(payload, "label_stream_bytes_total", "pdw1.point.payload"),
        "pdw1.point.payload.label_stream_bytes_total",
    )
    payload_fills = _integer(
        _field(payload, "fills_bytes_total", "pdw1.point.payload"),
        "pdw1.point.payload.fills_bytes_total",
    )
    payload_other = payload_total - payload_header - payload_labels - payload_fills
    if payload_other < 0:
        raise SharedReceiverAdmissionError("pdw1 payload byte components exceed total")

    projected_n600_bytes = _integer(
        _field(projection, "total_bytes", "pdw1.point.n600_extrapolation"),
        "pdw1.point.n600_extrapolation.total_bytes",
        minimum=1,
    )
    projected_bytes_per_pair = _finite(
        _field(projection, "bytes_per_pair", "pdw1.point.n600_extrapolation"),
        "pdw1.point.n600_extrapolation.bytes_per_pair",
        minimum=0.0,
    )
    if abs(projected_bytes_per_pair - projected_n600_bytes / PAIR_COUNT) > 0.01:
        raise SharedReceiverAdmissionError("pdw1 projected bytes-per-pair is inconsistent with its n600 total")
    return {
        "n_pairs": n_pairs,
        "d_a": d_a,
        "d_b": _finite(_field(point, "d_B", "pdw1.point"), "pdw1.point.d_B", minimum=0.0),
        "d_seg": _finite(
            _field(point, "d_seg_hard_oracle_vs_lstar", "pdw1.point"),
            "pdw1.point.d_seg_hard_oracle_vs_lstar",
            minimum=0.0,
        ),
        "mismatch_pixels": mismatch_pixels,
        "confusion_rows_lstar_to_pred": confusion_rows,
        "mismatch_within_chebyshev_r_of_lstar_boundary": boundary_proximity,
        "payload_n24_bytes": payload_total,
        "payload_n24_components": {
            "header_bytes": payload_header,
            "label_stream_bytes": payload_labels,
            "fills_bytes": payload_fills,
            "other_container_bytes": payload_other,
        },
        "payload_sha256": _sha256(
            _field(payload, "sha256", "pdw1.point.payload"),
            "pdw1.point.payload.sha256",
        ),
        "projected_n600_bytes": projected_n600_bytes,
        "projected_bytes_per_pair": projected_bytes_per_pair,
    }


def _validate_step2_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    if _field(summary, "schema", "step2") != "pdw1_step2_prior_measurement_summary.v1":
        raise SharedReceiverAdmissionError("step2 summary schema mismatch")
    n48 = _mapping(_field(summary, "n48", "step2"), "step2.n48")
    reasons = _mapping(_field(n48, "residual_reason_counts", "step2.n48"), "step2.n48.residual_reason_counts")
    marginal = _finite(
        _field(n48, "first_prefix_marginal_seg_score_per_projected_byte", "step2.n48"),
        "step2.n48.first_prefix_marginal_seg_score_per_projected_byte",
        minimum=0.0,
    )
    waterline = _finite(
        _field(n48, "rate_waterline_per_byte", "step2.n48"),
        "step2.n48.rate_waterline_per_byte",
        minimum=0.0,
    )
    n_pairs = _integer(_field(n48, "n_pairs", "step2.n48"), "step2.n48.n_pairs", minimum=1)
    if n_pairs != 48:
        raise SharedReceiverAdmissionError("settled STEP-2 summary must carry its n48 row")
    residual_pixels = _integer(
        _field(n48, "residual_pixels", "step2.n48"),
        "step2.n48.residual_pixels",
    )
    residual_reason_counts = {
        str(key): _integer(value, f"step2.n48.residual_reason_counts.{key}") for key, value in reasons.items()
    }
    if sum(residual_reason_counts.values()) != residual_pixels:
        raise SharedReceiverAdmissionError("step2 residual-reason counts must sum to residual_pixels")
    selected_ranked_cells = _integer(
        _field(n48, "selected_ranked_cells", "step2.n48"),
        "step2.n48.selected_ranked_cells",
    )
    if marginal <= waterline and selected_ranked_cells != 0:
        raise SharedReceiverAdmissionError("step2 below-waterline row cannot claim selected ranked cells")
    return {
        "source_commit": _git_sha(_field(summary, "source_commit", "step2"), "step2.source_commit"),
        "source_receipt_sha256": _sha256(
            _field(summary, "source_receipt_sha256", "step2"),
            "step2.source_receipt_sha256",
        ),
        "n_pairs": n_pairs,
        "residual_pixels": residual_pixels,
        "first_prefix_net_fixed_pixels": _integer(
            _field(n48, "first_prefix_net_fixed_pixels", "step2.n48"),
            "step2.n48.first_prefix_net_fixed_pixels",
        ),
        "first_prefix_projected_bytes": _integer(
            _field(n48, "first_prefix_projected_bytes", "step2.n48"),
            "step2.n48.first_prefix_projected_bytes",
        ),
        "marginal_seg_score_per_projected_byte": marginal,
        "rate_waterline_per_byte": waterline,
        "clears_rate_waterline": marginal > waterline,
        "selected_ranked_cells": selected_ranked_cells,
        "residual_reason_counts": residual_reason_counts,
    }


def _validate_dense_section(receipt: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if receipt is None:
        return None
    if _field(receipt, "schema", "dense_section") != "dense_quotient_field_zip_measurement.v1":
        raise SharedReceiverAdmissionError("dense spatial-section receipt schema mismatch")
    source_bytes = _integer(
        _field(receipt, "source_bytes", "dense_section"),
        "dense_section.source_bytes",
        minimum=1,
    )
    zip_bytes = _integer(
        _field(receipt, "zip_bytes", "dense_section"),
        "dense_section.zip_bytes",
        minimum=1,
    )
    member_uncompressed_bytes = _integer(
        _field(receipt, "member_uncompressed_bytes", "dense_section"),
        "dense_section.member_uncompressed_bytes",
        minimum=1,
    )
    member_compressed_bytes = _integer(
        _field(receipt, "member_compressed_bytes", "dense_section"),
        "dense_section.member_compressed_bytes",
        minimum=1,
    )
    if member_uncompressed_bytes != source_bytes:
        raise SharedReceiverAdmissionError("dense section member-uncompressed bytes must match source bytes")
    zip_overhead_bytes = zip_bytes - member_compressed_bytes
    if zip_overhead_bytes < 0:
        raise SharedReceiverAdmissionError("dense section compressed member bytes cannot exceed ZIP bytes")
    return {
        "source_bytes": source_bytes,
        "source_sha256": _sha256(
            _field(receipt, "source_sha256", "dense_section"),
            "dense_section.source_sha256",
        ),
        "member_uncompressed_bytes": member_uncompressed_bytes,
        "member_compressed_bytes": member_compressed_bytes,
        "zip_container_overhead_bytes": zip_overhead_bytes,
        "compression_ratio": member_compressed_bytes / member_uncompressed_bytes,
        "zip_bytes": zip_bytes,
        "zip_sha256": _sha256(
            _field(receipt, "zip_sha256", "dense_section"),
            "dense_section.zip_sha256",
        ),
        "compression": _nonempty_string(
            _field(receipt, "compression", "dense_section"),
            "dense_section.compression",
        ),
    }


def evaluate_shared_receiver_admission(
    *,
    pdw2_receipt: Mapping[str, Any],
    pdw1_receipt: Mapping[str, Any],
    step2_summary: Mapping[str, Any],
    dense_section_receipt: Mapping[str, Any] | None = None,
    exact_candidate: Mapping[str, Any] | None = None,
    exact_candidate_custody: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the strict shared-receiver intersection from typed evidence."""

    pdw2 = _validate_pdw2_receipt(_mapping(pdw2_receipt, "pdw2_receipt"))
    pdw1 = _validate_pdw1_receipt(_mapping(pdw1_receipt, "pdw1_receipt"))
    step2 = _validate_step2_summary(_mapping(step2_summary, "step2_summary"))
    dense = _validate_dense_section(dense_section_receipt)
    if exact_candidate is not None or exact_candidate_custody is not None:
        raise SharedReceiverAdmissionError(
            "self-authored candidate JSON is not admission authority; the canonical "
            "contest-CPU evaluator and trusted production-receiver parser are not wired"
        )

    if dense is not None:
        if dense["source_sha256"] != pdw2["quotient_file_sha256"]:
            raise SharedReceiverAdmissionError(
                "dense spatial-section source hash does not match the consumed n600 field"
            )
        if dense["source_bytes"] != pdw2["quotient_file_bytes"]:
            raise SharedReceiverAdmissionError(
                "dense spatial-section source bytes do not match the consumed n600 field"
            )
    # These keys preserve the machine-readable gate decomposition, but v1 has
    # no trusted parser that can derive them from an evaluator invocation.  No
    # caller-supplied boolean or self-authored JSON can turn them true.
    candidate_checks = dict.fromkeys(
        (
            "present",
            "n600",
            "archive_in_box",
            "bytes_per_pair_in_box",
            "d_seg_in_box",
            "exact_archive",
            "archive_parseback_identical",
            "production_receiver",
            "through_r_authority",
            "hard_cpu_torch_oracle",
            "packet_mutation_changes_decoded",
            "scorer_free_spatial_rgb_pullback",
            "pdw2_packet_hash_bound",
            "archive_content_hash_bound",
            "spatial_generator_payload_hash_bound",
            "hard_oracle_receipt_hash_bound",
        ),
        False,
    )
    success = False

    dense_row = None
    if dense is not None:
        dense_row = {
            **dense,
            "bytes_per_pair": dense["zip_bytes"] / PAIR_COUNT,
            "over_archive_gate_bytes": max(0, dense["zip_bytes"] - MAX_ARCHIVE_BYTES),
            "in_box": dense["zip_bytes"] <= MAX_ARCHIVE_BYTES,
            "evidence_label": "MEASURED_EXACT_ZIP_SECTION_BYTES_NOT_CONTEST_ARCHIVE",
        }

    blockers: list[dict[str, Any]] = []
    if not success:
        blockers.extend(
            [
                {
                    "term": "pdw2_packet_to_spatial_rgb",
                    "status": "COEFFICIENT_CONSUMED_ONLY_WITH_EXPLICIT_FIELD",
                    "packet_bytes": pdw2["packet_brotli_bytes"],
                    "through_r_authority": False,
                },
                {
                    "term": "pdw1_partition_carrier",
                    "status": "MEASURED_OUT_OF_BOX_FORMULATION",
                    "projected_n600_bytes": pdw1["projected_n600_bytes"],
                    "excess_bytes_vs_gate": pdw1["projected_n600_bytes"] - MAX_ARCHIVE_BYTES,
                    "bytes_per_pair": pdw1["projected_bytes_per_pair"],
                    "d_seg": pdw1["d_seg"],
                    "d_seg_excess": pdw1["d_seg"] - MAX_D_SEG,
                    "hard_oracle_mismatch_pixels_n24": pdw1["mismatch_pixels"],
                },
                {
                    "term": "ev_ranked_sparse_preimage_repair",
                    "status": "FIRST_MEASURED_PREFIX_BELOW_RATE_WATERLINE",
                    "n_pairs": step2["n_pairs"],
                    "residual_pixels": step2["residual_pixels"],
                    "first_prefix_net_fixed_pixels": step2["first_prefix_net_fixed_pixels"],
                    "first_prefix_projected_bytes": step2["first_prefix_projected_bytes"],
                    "marginal_seg_score_per_projected_byte": step2["marginal_seg_score_per_projected_byte"],
                    "rate_waterline_per_byte": step2["rate_waterline_per_byte"],
                    "selected_ranked_cells": step2["selected_ranked_cells"],
                    "residual_reason_counts": step2["residual_reason_counts"],
                },
            ]
        )
        if dense_row is not None:
            blockers.append(
                {
                    "term": "dense_float32_spatial_field_section",
                    "status": "MEASURED_SECTION_RATE_DOMINATES"
                    if not dense_row["in_box"]
                    else "SECTION_IN_BOX_BUT_RGB_PULLBACK_STILL_ABSENT",
                    **dense_row,
                }
            )

    return {
        "schema": SCHEMA,
        "gate": {
            "n_pairs": PAIR_COUNT,
            "max_bytes_per_pair": MAX_BYTES_PER_PAIR,
            "max_archive_bytes": MAX_ARCHIVE_BYTES,
            "max_d_seg": MAX_D_SEG,
            "rate_price_score_per_byte": RATE_PRICE_PER_BYTE,
        },
        "success": success,
        "verdict": "MEASURED_IN_BOX" if success else BLOCKER_ID,
        "verdict_scope": (
            "dense-float32 spatial field, PDW1 per-pair label/fill carrier, and "
            "measured EV-ranked sparse camera-value repair prefix; broader shared "
            "receiver family remains open"
        ),
        "candidate_checks": candidate_checks,
        "candidate": None,
        "candidate_custody": None,
        "pdw2": pdw2,
        "pdw1": pdw1,
        "step2": step2,
        "dense_spatial_section": dense_row,
        "blockers": blockers,
        "dominant_measured_term": (
            "dense_float32_spatial_field_section"
            if dense_row is not None and not dense_row["in_box"]
            else "pdw1_partition_carrier_and_realization"
        ),
        "untested_optimal_form": UNTESTED_OPTIMAL_FORM,
        "through_r_authority": bool(success),
        "score_claim": False,
        "promotion_eligible": False,
    }


__all__ = [
    "BLOCKER_ID",
    "MAX_ARCHIVE_BYTES",
    "MAX_BYTES_PER_PAIR",
    "MAX_D_SEG",
    "SCHEMA",
    "SharedReceiverAdmissionError",
    "evaluate_shared_receiver_admission",
]
