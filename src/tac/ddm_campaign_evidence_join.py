# SPDX-License-Identifier: MIT
"""Strict schema and accounting laws for the DDM EV1 evidence join.

This module is intentionally free of scorer and receiver imports.  The heavy
producer lives in ``tools/measure_ddm_ev1_campaign_evidence_joins.py``; every
read-only consumer validates the resulting receipt here before using it.

The rate-home law is an *exclusive accounting partition*.  It assigns each
integer byte index in one measured candidate delta to exactly one
``(dual edge, stratum, visibility, G4 class)`` cell.  It does not claim that a
ZIP byte is physically separable, and it does not manufacture an RD1 price.
The downstream ms2r solve remains the sole owner of prices.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from decimal import ROUND_FLOOR, Decimal
from typing import Any

SCHEMA = "ddm_ev1_campaign_evidence_join_receipt.v1"
V19_SCHEMA = "ddm_ev1_v19_receiver_closed_pair_join.v1"
RD1_SCHEMA = "ddm_ev1_rd1_byte_home_uint8_histograms.v1"
HISTOGRAM_SCHEMA = "ddm_ev1_uint8_abs_step_histogram.v1"
METRIC_ID = "exact_composite_R_rank4_margin_fisher_plus_pose6_quadratic"
PAIR_COUNT = 600
SEMANTIC_STRATA = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
STRATA = (*SEMANTIC_STRATA, "POSE6_GLOBAL")
VISIBILITIES = ("ker(A)-invisible", "seg-visible", "pose-visible")
G4_CLASSES = ("STATIC_IN_IMAGE", "STATIC_IN_XI_PROXY", "TRANSIENT")
HOME_SCOPES = ("per_frame", "shared_k_frames", "shared_clip")
RD1_DUAL_COUNT = 3
RD1_BUCKET_COUNT = RD1_DUAL_COUNT * len(STRATA) * len(VISIBILITIES) * len(G4_CLASSES)
_HIST_MAGIC = b"DDMEV1H1"


class CampaignEvidenceJoinError(ValueError):
    """The evidence receipt or its exclusive byte accounting is malformed."""


def canonical_bytes(value: Any) -> bytes:
    """Return the repository's deterministic compact JSON representation."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def bucket_key(row: Mapping[str, Any]) -> tuple[int, str, str, str]:
    """Return the unique typed RD1 cell key."""

    return (
        int(row["dual_index"]),
        str(row["stratum"]),
        str(row["scorer_visibility"]),
        str(row["g4_temporal_class"]),
    )


def allocate_exclusive_byte_homes(
    *,
    delta_counted_bytes: int,
    weights: Mapping[tuple[int, str, str, str], float | Decimal],
) -> dict[tuple[int, str, str, str], dict[str, Any]]:
    """Partition ``[0, delta_counted_bytes)`` by largest remainder.

    The weight is scorer-metric evidence supplied by the producer.  Receiver
    absolute-step mass may only break the G4 tie *inside* one measured scorer
    dimension; it is never used as an independent Euclidean score.
    """

    if isinstance(delta_counted_bytes, bool) or delta_counted_bytes <= 0:
        raise CampaignEvidenceJoinError("candidate delta bytes must be a positive integer")
    if not weights:
        raise CampaignEvidenceJoinError("exclusive byte homes require nonempty weights")
    ordered = sorted(weights)
    decimals: dict[tuple[int, str, str, str], Decimal] = {}
    for key in ordered:
        value = Decimal(str(weights[key]))
        if not value.is_finite() or value < 0:
            raise CampaignEvidenceJoinError(f"invalid byte-home weight for {key!r}")
        decimals[key] = value
    total = sum(decimals.values(), Decimal(0))
    if total <= 0:
        raise CampaignEvidenceJoinError("at least one scorer-metric byte-home weight must be positive")
    exact = {key: Decimal(delta_counted_bytes) * decimals[key] / total for key in ordered}
    allocation = {key: int(exact[key].to_integral_value(rounding=ROUND_FLOOR)) for key in ordered}
    residual = delta_counted_bytes - sum(allocation.values())
    remainder_order = sorted(
        ordered,
        key=lambda key: (-(exact[key] - allocation[key]), key),
    )
    for key in remainder_order[:residual]:
        allocation[key] += 1

    cursor = 0
    result: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for key in ordered:
        count = allocation[key]
        ranges = [[cursor, cursor + count]] if count else []
        result[key] = {
            "delta_counted_bytes_dimension": count,
            "byte_home_ranges": ranges,
            "byte_home_epistemic_status": ("DERIVED_FROM_MEASURED_SCORER_METRIC_BY_EXCLUSIVE_LARGEST_REMAINDER"),
        }
        cursor += count
    if cursor != delta_counted_bytes:
        raise CampaignEvidenceJoinError("exclusive byte homes do not close")
    return result


def derive_g4_reuse_profiles(
    *,
    recurrence_k_distribution: Mapping[str, Any],
    xi_track_lengths: Sequence[int],
) -> dict[str, dict[str, Any]]:
    """Derive the three G4 byte-home amortization profiles.

    G4's exact ``k`` table contains one row per exact recurrence count, with
    event mass equal to ``k * locus_count``.  One RD1 G4 bucket aggregates
    those loci, so its honest effective reach is total served events divided
    by the number of once-coded loci.  This preserves the measured table
    without pretending that every recurrent locus spans the whole clip.
    """

    exact = recurrence_k_distribution.get("exact_k")
    if not isinstance(exact, list) or not exact:
        raise CampaignEvidenceJoinError("G4 exact-k amortization table is absent")
    observed_k: set[int] = set()
    static_events = 0
    static_loci = 0
    transient_events = 0
    transient_loci = 0
    for row in exact:
        if not isinstance(row, Mapping):
            raise CampaignEvidenceJoinError("G4 exact-k row is malformed")
        k = _exact_int(row.get("k"), "G4 recurrence k")
        events = _exact_int(row.get("flip_event_mass"), "G4 flip event mass")
        loci = _exact_int(row.get("locus_count"), "G4 locus count")
        if k <= 0 or k > PAIR_COUNT or events < 0 or loci < 0 or events != k * loci:
            raise CampaignEvidenceJoinError("G4 exact-k row does not close")
        if k in observed_k:
            raise CampaignEvidenceJoinError("G4 exact-k table duplicates k")
        observed_k.add(k)
        if k == 1:
            transient_events += events
            transient_loci += loci
        else:
            static_events += events
            static_loci += loci
    if static_loci <= 0 or transient_loci <= 0:
        raise CampaignEvidenceJoinError("G4 recurrence table lacks static or transient support")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in xi_track_lengths):
        raise CampaignEvidenceJoinError("G4 xi track lengths must be exact integers >=2")
    xi_events = sum(xi_track_lengths)
    xi_tracks = len(xi_track_lengths)
    if xi_tracks <= 0:
        raise CampaignEvidenceJoinError("G4 xi amortization requires at least one measured track")
    transient_events -= xi_events
    transient_loci -= xi_events
    if transient_events <= 0 or transient_loci <= 0 or transient_events != transient_loci:
        raise CampaignEvidenceJoinError("G4 transient support does not close after xi partition")

    def profile(
        *,
        scope: str,
        numerator: int,
        denominator: int,
        source: str,
    ) -> dict[str, Any]:
        return {
            "scope": scope,
            "k": numerator / denominator,
            "k_numerator": numerator,
            "k_denominator": denominator,
            "k_source": source,
            "epistemic_status": (
                "DERIVED_FROM_MEASURED_G4_CLASS_LEVEL_TABLE_NOT_EDGE_SPECIFIC_K_REMEASUREMENT"
            ),
        }

    return {
        "STATIC_IN_IMAGE": profile(
            scope="shared_k_frames",
            numerator=static_events,
            denominator=static_loci,
            source="G4_EXACT_K_TABLE_EVENT_MASS_PER_RECURRENT_LOCUS",
        ),
        "STATIC_IN_XI_PROXY": profile(
            scope="shared_k_frames",
            numerator=xi_events,
            denominator=xi_tracks,
            source="G4_XI_PROXY_TRACK_EVENT_MASS_PER_TRACK",
        ),
        "TRANSIENT": profile(
            scope="per_frame",
            numerator=transient_events,
            denominator=transient_loci,
            source="G4_EXACT_K_TABLE_K_EQ_1_EXCLUDING_XI_PROXY_EVENTS",
        ),
    }


def pack_histogram_rows(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Pack the 162 uint8 absolute-step histograms in typed-key order."""

    ordered = sorted(rows, key=bucket_key)
    if len(ordered) != RD1_BUCKET_COUNT:
        raise CampaignEvidenceJoinError(f"histogram pack requires {RD1_BUCKET_COUNT} rows")
    payload = bytearray(_HIST_MAGIC)
    payload.extend(struct.pack("<H", len(ordered)))
    for row in ordered:
        hist = row.get("receiver_uint8_abs_step_histogram")
        if not isinstance(hist, list) or len(hist) != 256:
            raise CampaignEvidenceJoinError("histogram must have 256 bins")
        values: list[int] = []
        for value in hist:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CampaignEvidenceJoinError("histogram bins must be nonnegative integers")
            values.append(value)
        payload.extend(struct.pack("<256Q", *values))
    return bytes(payload)


def unpack_histogram_rows(
    payload: bytes,
    ordered_keys: Sequence[tuple[int, str, str, str]],
) -> dict[tuple[int, str, str, str], list[int]]:
    """Parse a packed histogram payload and refuse trailing or missing bytes."""

    if not payload.startswith(_HIST_MAGIC) or len(payload) < len(_HIST_MAGIC) + 2:
        raise CampaignEvidenceJoinError("histogram payload magic differs")
    count = struct.unpack_from("<H", payload, len(_HIST_MAGIC))[0]
    if count != len(ordered_keys):
        raise CampaignEvidenceJoinError("histogram payload row count differs")
    stride = 256 * 8
    expected = len(_HIST_MAGIC) + 2 + count * stride
    if len(payload) != expected:
        raise CampaignEvidenceJoinError("histogram payload byte length differs")
    cursor = len(_HIST_MAGIC) + 2
    result: dict[tuple[int, str, str, str], list[int]] = {}
    for key in ordered_keys:
        result[key] = list(struct.unpack_from("<256Q", payload, cursor))
        cursor += stride
    return result


def _exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CampaignEvidenceJoinError(f"{label} must be an exact JSON integer")
    return value


def _validate_histogram(row: Mapping[str, Any]) -> None:
    hist = row.get("receiver_uint8_abs_step_histogram")
    if not isinstance(hist, list) or len(hist) != 256:
        raise CampaignEvidenceJoinError("every RD1 row requires one 256-bin histogram")
    for value in hist:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CampaignEvidenceJoinError("histogram bins must be nonnegative integers")
    if hist[0] != 0:
        raise CampaignEvidenceJoinError("absolute-step bin zero must remain empty")
    changed = _exact_int(row.get("receiver_changed_channel_values"), "changed values")
    step_sum = _exact_int(row.get("receiver_uint8_abs_step_sum"), "absolute-step sum")
    if changed != sum(hist) or step_sum != sum(index * value for index, value in enumerate(hist)):
        raise CampaignEvidenceJoinError("histogram moments do not match their declared values")


def _validate_amortized_home(
    row: Mapping[str, Any],
    profiles: Mapping[str, Mapping[str, Any]],
) -> None:
    g4 = str(row.get("g4_temporal_class"))
    profile = profiles.get(g4)
    if not isinstance(profile, Mapping):
        raise CampaignEvidenceJoinError("RD1 home lacks its G4 reuse profile")
    scope = row.get("scope")
    if scope not in HOME_SCOPES or scope != profile.get("scope"):
        raise CampaignEvidenceJoinError("RD1 home scope differs from measured G4 reuse")
    numerator = _exact_int(row.get("k_numerator"), "RD1 k numerator")
    denominator = _exact_int(row.get("k_denominator"), "RD1 k denominator")
    if numerator <= 0 or denominator <= 0:
        raise CampaignEvidenceJoinError("RD1 amortization ratio must be positive")
    if numerator != profile.get("k_numerator") or denominator != profile.get("k_denominator"):
        raise CampaignEvidenceJoinError("RD1 amortization ratio differs from G4 custody")
    if (
        row.get("epistemic_status")
        != "DERIVED_FROM_MEASURED_G4_CLASS_LEVEL_TABLE_NOT_EDGE_SPECIFIC_K_REMEASUREMENT"
        or row.get("epistemic_status") != profile.get("epistemic_status")
    ):
        raise CampaignEvidenceJoinError("RD1 amortization epistemic scope differs")
    k = row.get("k")
    if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(float(k)):
        raise CampaignEvidenceJoinError("RD1 amortization k must be finite")
    if not math.isclose(float(k), numerator / denominator, rel_tol=0.0, abs_tol=1e-12):
        raise CampaignEvidenceJoinError("RD1 amortization k does not close")
    if (scope == "per_frame") != (numerator == denominator):
        raise CampaignEvidenceJoinError("RD1 per-frame scope and k disagree")
    if scope == "shared_clip" and numerator != PAIR_COUNT * denominator:
        raise CampaignEvidenceJoinError("RD1 shared-clip scope is not k=600")
    count = _exact_int(row.get("delta_counted_bytes_dimension"), "RD1 dimension bytes")
    amortized = row.get("amortized_bytes_per_frame")
    if (
        isinstance(amortized, bool)
        or not isinstance(amortized, (int, float))
        or not math.isfinite(float(amortized))
        or not math.isclose(
            float(amortized),
            count * denominator / numerator,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise CampaignEvidenceJoinError("RD1 amortized bytes per frame do not close")


def validate_campaign_evidence_join(receipt: Mapping[str, Any]) -> dict[str, int]:
    """Validate the complete producer receipt and return consumer-safe counts."""

    if receipt.get("schema") != SCHEMA:
        raise CampaignEvidenceJoinError("EV1 receipt schema differs")
    for key, expected in (
        ("pair_count", PAIR_COUNT),
        ("score_claim", False),
        ("pointer_moved", False),
        ("research_only", True),
        ("execution_allowed", False),
        ("promotion_eligible", False),
        ("main_landing_review_required", True),
    ):
        if receipt.get(key) != expected:
            raise CampaignEvidenceJoinError(f"EV1 authority firewall differs at {key}")
    metric = receipt.get("metric_custody")
    if not isinstance(metric, Mapping) or metric.get("metric_id") != METRIC_ID:
        raise CampaignEvidenceJoinError("EV1 scorer metric authority differs")
    if (
        metric.get("euclidean_naive_rows_admitted") is not False
        or metric.get("margin_fisher_gram_bound") is not True
        or metric.get("composite_R_hessian_bound") is not True
    ):
        raise CampaignEvidenceJoinError("EV1 scorer metric custody is incomplete")
    implementation = receipt.get("implementation_custody")
    if (
        not isinstance(implementation, Mapping)
        or implementation.get("git_head_at_measurement") != "UNCOMMITTED_DELEGATED_WORKTREE; LANDING_COMMIT_REQUIRED"
        or len(implementation.get("source_files") or []) != 2
    ):
        raise CampaignEvidenceJoinError("EV1 implementation custody is incomplete")
    endpoint_policy = receipt.get("rd1_endpoint_remeasurement")
    if (
        not isinstance(endpoint_policy, Mapping)
        or endpoint_policy.get("status") != "FRESH_CURRENT_RECEIVER_REPLAY_MEASURED"
        or endpoint_policy.get("prior_menu_score_batches_reused") is not False
        or endpoint_policy.get("receiver_scope")
        != (
            "EXACT_MEASUREMENT_HARNESS_PAYLOAD_DECODE_THROUGH_UINT8_R; MENU1_PAYLOAD_ENDPOINTS_ARE_NOT_CONTEST_ARCHIVES"
        )
    ):
        raise CampaignEvidenceJoinError("EV1 RD1 endpoint score/camera replay authority differs")

    v19 = receipt.get("v19_pair_join")
    if not isinstance(v19, Mapping) or v19.get("schema") != V19_SCHEMA:
        raise CampaignEvidenceJoinError("V19 pair-join schema differs")
    rows = v19.get("rows")
    if not isinstance(rows, list) or len(rows) != PAIR_COUNT:
        raise CampaignEvidenceJoinError("V19 pair join must contain 600 rows")
    pair_ids = [_exact_int(row.get("source_pair_id"), "source_pair_id") for row in rows]
    if sorted(pair_ids) != list(range(PAIR_COUNT)) or len(set(pair_ids)) != PAIR_COUNT:
        raise CampaignEvidenceJoinError("V19 pair ids are not exactly 0..599")
    if any(
        row.get("receiver_closed") is not True
        or row.get("score_claim") is not False
        or row.get("per_pair_byte_allocation") is not None
        for row in rows
    ):
        raise CampaignEvidenceJoinError("V19 rows violate receiver/rate authority")
    shared = v19.get("shared_rate_home")
    if not isinstance(shared, Mapping):
        raise CampaignEvidenceJoinError("V19 shared rate home is absent")
    delta = _exact_int(shared.get("delta_counted_bytes"), "V19 shared bytes")
    if (
        delta <= 0
        or shared.get("counted_exactly_once") is not True
        or shared.get("per_pair_allocation") is not None
        or shared.get("status") != "MEASURED_EXACT_GLOBAL_ARCHIVE_DELTA_HOME"
    ):
        raise CampaignEvidenceJoinError("V19 shared rate home is not exact and exclusive")

    rd1 = receipt.get("rd1_evidence")
    if not isinstance(rd1, Mapping) or rd1.get("schema") != RD1_SCHEMA:
        raise CampaignEvidenceJoinError("RD1 evidence schema differs")
    bucket_rows = rd1.get("bucket_rows")
    if not isinstance(bucket_rows, list) or len(bucket_rows) != RD1_BUCKET_COUNT:
        raise CampaignEvidenceJoinError("RD1 evidence must contain 162 bucket rows")
    expected_keys = {
        (dual, stratum, visibility, g4)
        for dual in range(1, RD1_DUAL_COUNT + 1)
        for stratum in STRATA
        for visibility in VISIBILITIES
        for g4 in G4_CLASSES
    }
    observed_keys = {bucket_key(row) for row in bucket_rows}
    if observed_keys != expected_keys or len(observed_keys) != len(bucket_rows):
        raise CampaignEvidenceJoinError("RD1 typed cube is incomplete or duplicated")
    if any(
        row.get("lambda_bytes_per_D_dimension") is not None
        or row.get("pricing_owner") != "ddm_ms2r_tolerance_capped_solve_r2"
        or row.get("score_claim") is not False
        for row in bucket_rows
    ):
        raise CampaignEvidenceJoinError("EV1 producer trespassed into ms2r pricing")
    amortization = rd1.get("amortization_custody")
    if not isinstance(amortization, Mapping):
        raise CampaignEvidenceJoinError("RD1 G4 amortization custody is absent")
    profiles = amortization.get("profiles")
    if (
        not isinstance(profiles, Mapping)
        or set(profiles) != set(G4_CLASSES)
        or amortization.get("single_owner_across_reach") is not True
        or amortization.get("shared_clip_bucket_count") != 0
        or amortization.get("shared_clip_absence_reason")
        != "NO_AGGREGATED_G4_BUCKET_IS_EXCLUSIVELY_K_EQ_600"
        or amortization.get("application_scope")
        != "G4_CLASS_LEVEL_REUSE_PRIOR_APPLIED_TO_TYPED_HOMES_NOT_EDGE_SPECIFIC_K_REMEASUREMENT"
    ):
        raise CampaignEvidenceJoinError("RD1 G4 amortization profiles are incomplete")
    interpreter = receipt.get("free_interpreter_custody")
    if (
        not isinstance(interpreter, Mapping)
        or interpreter.get("generic_decoder_and_transport_code_counted_bytes") != 0
        or interpreter.get("irreducible_video_statistic_bytes_counted") is not True
        or interpreter.get("xi_video_parameters_free") is not False
        or interpreter.get("independent_physical_bev_claim") is not False
    ):
        raise CampaignEvidenceJoinError("EV1 free-interpreter byte law is incomplete")
    for row in bucket_rows:
        _validate_histogram(row)
        _validate_amortized_home(row, profiles)

    edge_summaries = rd1.get("edge_summaries")
    if not isinstance(edge_summaries, list) or len(edge_summaries) != RD1_DUAL_COUNT:
        raise CampaignEvidenceJoinError("RD1 edge summaries are incomplete")
    by_edge = {
        dual: [row for row in bucket_rows if int(row["dual_index"]) == dual] for dual in range(1, RD1_DUAL_COUNT + 1)
    }
    for summary in edge_summaries:
        dual = _exact_int(summary.get("dual_index"), "RD1 dual index")
        expected_delta = _exact_int(summary.get("delta_counted_bytes"), "RD1 edge bytes")
        rows_for_edge = by_edge.get(dual)
        if rows_for_edge is None:
            raise CampaignEvidenceJoinError("RD1 edge index escaped 1..3")
        intervals: list[tuple[int, int]] = []
        assigned = 0
        for row in rows_for_edge:
            count = _exact_int(
                row.get("delta_counted_bytes_dimension"),
                "RD1 dimension bytes",
            )
            if count < 0:
                raise CampaignEvidenceJoinError("RD1 dimension byte home is negative")
            assigned += count
            ranges = row.get("byte_home_ranges")
            if not isinstance(ranges, list):
                raise CampaignEvidenceJoinError("RD1 byte-home ranges are absent")
            if (count == 0) != (ranges == []):
                raise CampaignEvidenceJoinError("RD1 zero/nonzero home range disagrees")
            for interval in ranges:
                if (
                    not isinstance(interval, list)
                    or len(interval) != 2
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in interval)
                    or interval[0] < 0
                    or interval[1] <= interval[0]
                ):
                    raise CampaignEvidenceJoinError("RD1 byte-home interval is invalid")
                if interval[1] - interval[0] != count:
                    raise CampaignEvidenceJoinError("RD1 byte-home interval size differs")
                intervals.append((interval[0], interval[1]))
        intervals.sort()
        if assigned != expected_delta or summary.get("assigned_counted_bytes") != assigned:
            raise CampaignEvidenceJoinError("RD1 edge bytes do not close")
        if intervals:
            cursor = 0
            for start, stop in intervals:
                if start != cursor:
                    raise CampaignEvidenceJoinError("RD1 byte homes overlap or leave a gap")
                cursor = stop
            if cursor != expected_delta:
                raise CampaignEvidenceJoinError("RD1 byte-home domain does not close")
        if (
            summary.get("counted_exactly_once") is not True
            or summary.get("accounting_home_not_physical_zip_separability") is not True
        ):
            raise CampaignEvidenceJoinError("RD1 exclusive accounting scope is unstated")

    codec = rd1.get("histogram_coder")
    if not isinstance(codec, Mapping) or (
        codec.get("codec") != "BROTLI_Q11" or codec.get("parse_back_identical") is not True
    ):
        raise CampaignEvidenceJoinError("RD1 real histogram coder proof is absent")
    shared = sum(row["scope"] != "per_frame" for row in bucket_rows)
    return {
        "exact_pair_rows": len(rows),
        "required_pair_rows": PAIR_COUNT,
        "missing_pair_rows": 0,
        "typed_dimension_rows": len(bucket_rows),
        "dimension_byte_homes": len(bucket_rows),
        "shared_across_frame_byte_homes": shared,
        "per_frame_byte_homes": len(bucket_rows) - shared,
        "shared_clip_byte_homes": sum(row["scope"] == "shared_clip" for row in bucket_rows),
        "receiver_uint8_histograms": len(bucket_rows),
    }


__all__ = [
    "G4_CLASSES",
    "HISTOGRAM_SCHEMA",
    "HOME_SCOPES",
    "METRIC_ID",
    "PAIR_COUNT",
    "RD1_BUCKET_COUNT",
    "RD1_DUAL_COUNT",
    "RD1_SCHEMA",
    "SCHEMA",
    "SEMANTIC_STRATA",
    "STRATA",
    "V19_SCHEMA",
    "VISIBILITIES",
    "CampaignEvidenceJoinError",
    "allocate_exclusive_byte_homes",
    "bucket_key",
    "canonical_bytes",
    "canonical_sha256",
    "derive_g4_reuse_profiles",
    "pack_histogram_rows",
    "unpack_histogram_rows",
    "validate_campaign_evidence_join",
]
