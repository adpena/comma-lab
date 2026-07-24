# SPDX-License-Identifier: MIT
"""Held-out exact conditional-codelength probe for DDM CONNECTION.

The probe consumes the same SHA-bound solved scorer planes and PF2 occupied
supports used by IS1/DM1.  It prices one deterministic held-out consecutive
transition for every eligible semantic bucket.  Program state is fitted only
from the other transitions in that bucket, and every selector, state byte,
framing byte, and residual coder container is charged.

This is a local frozen-SegNet semantic-record measurement.  It does not invoke
``evaluate.py``, emit an archive, run PoseNet, or move a frontier pointer.
"""

from __future__ import annotations

import json
import math
import os
import platform
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from tac.optimization.ddm_dm1_solved_value_pricing import (
    CODECS,
    SolvedValuePricingError,
    SolvedValueRecord,
    _class_id,
    _class_pair_from_bucket,
    _segnet_logits,
    _stratum_from_bucket,
    _winner_symbols,
    canonical_json_bytes,
    checked_json,
    decode_codec,
    encode_codec,
    price_raw,
    sha256_bytes,
    support_sha256,
    typed_home,
)
from tac.optimization.ddm_min_description_contract import (
    LayerHome,
    StreamType,
    TypedStreamTag,
)
from tac.optimization.resize_full_kernel import FullResizeKernel
from tac.optimization.solve_diff_operator_mining import (
    AXIS,
    POINTER,
    SolveDiffMiningConfigV1,
    _load_production_inputs,
    _open_production_inputs,
    realize_solve_camera,
    sha256_file,
)

SCHEMA = "ddm_dm3_connection_conditional_codelength.v1"
ROW_SCHEMA = "ddm_dm3_connection_conditional_codelength_row.v1"
CONFIG_SCHEMA = "ddm_dm3_connection_conditional_codelength_config.v1"
HISTORY_MAGIC = b"DM3HST1\0"
HISTORY_FAMILIES = ("identity", "xi_advected", "affine_tracked")
HOLDOUT_POLICY = "lower_median_consecutive_transition_per_bucket"
FIT_POLICY = "all_other_same_bucket_consecutive_transitions"
_HISTORY_IDS = {name: index for index, name in enumerate(HISTORY_FAMILIES)}
_ID_HISTORY = {index: name for name, index in _HISTORY_IDS.items()}
_HISTORY_HEADER = struct.Struct("<8sBBHI")
_XI_STATE = struct.Struct("<hh")
_AFFINE_STATE = struct.Struct("<6i")
_Q16 = 1 << 16
_IMAGE_H = 384
_IMAGE_W = 512
_PIXELS_PER_PAIR = _IMAGE_H * _IMAGE_W
_REPO = Path(__file__).resolve().parents[3]


class ConnectionCodelengthError(SolvedValuePricingError):
    """Fail-closed conditional-code, custody, or held-out-fit error."""


@dataclass(frozen=True)
class SupportPopulation:
    """One bucket's exact occupied supports and consecutive transition set."""

    bucket_id: str
    array_key: str
    supports: Mapping[int, np.ndarray]
    transitions: tuple[tuple[int, int], ...]
    holdout: tuple[int, int] | None


@dataclass(frozen=True)
class SolvedSupport:
    """One bounded solved semantic record plus its exact PF2 support."""

    record: SolvedValueRecord
    support: np.ndarray
    raw: bytes
    solved_summary: Mapping[str, Any]
    roundtrip_max_abs: float


@dataclass(frozen=True)
class ProgramFit:
    """All content-selected state fitted without the held-out transition."""

    family: str
    state: bytes | None
    training_transition_count: int
    training_correspondence_count: int
    status: str
    diagnostics: Mapping[str, Any]


def _validate_false_authority(config: Mapping[str, Any]) -> None:
    required = {
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_emitted": False,
        "pointer_moved": False,
        "pointer": POINTER,
        "evidence_axis": AXIS,
        "main_review_required": True,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ConnectionCodelengthError(f"config {key} must equal {expected!r}")
    if config.get("torch_threads") != 4:
        raise ConnectionCodelengthError("config torch_threads must equal 4")
    if tuple(config.get("coders", ())) != CODECS:
        raise ConnectionCodelengthError("config coders differ from the sealed DM1 stack")
    if tuple(config.get("history_families", ())) != HISTORY_FAMILIES:
        raise ConnectionCodelengthError(
            "config history_families differ from the preregistered generic programs"
        )
    if config.get("holdout_policy") != HOLDOUT_POLICY:
        raise ConnectionCodelengthError("config holdout policy drifted")
    if config.get("fit_policy") != FIT_POLICY:
        raise ConnectionCodelengthError("config fit policy drifted")


def _bucket_type(bucket_id: str) -> str:
    parts = bucket_id.split("__")
    if len(parts) != 3 or not parts[2]:
        raise ConnectionCodelengthError(f"bucket type cannot be parsed from {bucket_id!r}")
    return parts[2]


def _build_support_population(
    event_index: Any,
    bucket_arrays: Mapping[str, Any],
) -> tuple[SupportPopulation, ...]:
    populations: list[SupportPopulation] = []
    for bucket_id, raw_key in sorted(bucket_arrays.items()):
        if not isinstance(raw_key, str) or raw_key not in event_index:
            raise ConnectionCodelengthError(
                f"PF2 receipt has no exact array binding for {bucket_id!r}"
            )
        event_ids = np.asarray(event_index[raw_key], dtype=np.int64)
        if event_ids.ndim != 1 or len(event_ids) == 0:
            raise ConnectionCodelengthError(f"PF2 array {raw_key!r} must be nonempty 1D")
        if np.any(event_ids < 0) or np.any(event_ids >= 600 * _PIXELS_PER_PAIR):
            raise ConnectionCodelengthError(f"PF2 array {raw_key!r} escapes n600")
        if np.any(event_ids[1:] <= event_ids[:-1]):
            raise ConnectionCodelengthError(
                f"PF2 array {raw_key!r} must be strictly increasing"
            )
        pair_ids = event_ids // _PIXELS_PER_PAIR
        unique_pairs, starts = np.unique(pair_ids, return_index=True)
        stops = np.concatenate((starts[1:], np.asarray([len(event_ids)])))
        supports: dict[int, np.ndarray] = {}
        for pair_id, start, stop in zip(unique_pairs, starts, stops, strict=True):
            flat = np.asarray(
                event_ids[int(start) : int(stop)] % _PIXELS_PER_PAIR,
                dtype=np.uint32,
            )
            if np.any(flat[1:] <= flat[:-1]):
                raise ConnectionCodelengthError(
                    f"PF2 support is not canonical for bucket={bucket_id} pair={pair_id}"
                )
            supports[int(pair_id)] = flat
        ordered = tuple(sorted(supports))
        transitions = tuple(
            (left, right)
            for left, right in pairwise(ordered)
            if right == left + 1
        )
        holdout = transitions[(len(transitions) - 1) // 2] if transitions else None
        populations.append(
            SupportPopulation(
                bucket_id=str(bucket_id),
                array_key=raw_key,
                supports=supports,
                transitions=transitions,
                holdout=holdout,
            )
        )
    return tuple(populations)


def _coordinates(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(flat, dtype=np.int64)
    return values % _IMAGE_W, values // _IMAGE_W


def _quantile_correspondence_sufficient_statistics(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return deterministic ordinal-correspondence normal-equation statistics."""

    count = min(len(left), len(right))
    if count <= 0:
        raise ConnectionCodelengthError("support correspondence must be nonempty")
    positions = np.arange(count, dtype=np.int64)
    left_take = (positions * len(left)) // count
    right_take = (positions * len(right)) // count
    left_x, left_y = _coordinates(np.asarray(left)[left_take])
    right_x, right_y = _coordinates(np.asarray(right)[right_take])
    left_x = left_x.astype(np.int64)
    left_y = left_y.astype(np.int64)
    right_x = right_x.astype(np.int64)
    right_y = right_y.astype(np.int64)
    sum_x = int(np.sum(left_x, dtype=np.int64))
    sum_y = int(np.sum(left_y, dtype=np.int64))
    xtx = np.asarray(
        [
            [
                int(np.sum(left_x * left_x, dtype=np.int64)),
                int(np.sum(left_x * left_y, dtype=np.int64)),
                sum_x,
            ],
            [
                int(np.sum(left_x * left_y, dtype=np.int64)),
                int(np.sum(left_y * left_y, dtype=np.int64)),
                sum_y,
            ],
            [sum_x, sum_y, count],
        ],
        dtype=np.float64,
    )
    xty = np.asarray(
        [
            [
                int(np.sum(left_x * right_x, dtype=np.int64)),
                int(np.sum(left_x * right_y, dtype=np.int64)),
            ],
            [
                int(np.sum(left_y * right_x, dtype=np.int64)),
                int(np.sum(left_y * right_y, dtype=np.int64)),
            ],
            [
                int(np.sum(right_x, dtype=np.int64)),
                int(np.sum(right_y, dtype=np.int64)),
            ],
        ],
        dtype=np.float64,
    )
    return xtx, xty, count


def _round_half_away_from_zero(value: float) -> int:
    if not math.isfinite(value):
        raise ConnectionCodelengthError("program-state fit produced a nonfinite value")
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def fit_programs(
    population: SupportPopulation,
) -> tuple[ProgramFit, ...]:
    """Fit identity/xi/affine state while excluding exactly the priced row."""

    if population.holdout is None:
        raise ConnectionCodelengthError("program fit requires an eligible holdout")
    training = tuple(
        transition
        for transition in population.transitions
        if transition != population.holdout
    )
    identity = ProgramFit(
        family="identity",
        state=b"",
        training_transition_count=len(training),
        training_correspondence_count=0,
        status="FIT_FREE_GENERIC_PROGRAM",
        diagnostics={"heldout_excluded": True},
    )
    if not training:
        unavailable = tuple(
            ProgramFit(
                family=family,
                state=None,
                training_transition_count=0,
                training_correspondence_count=0,
                status="NULL_INSUFFICIENT_LEAVE_ONE_OUT_TRAINING",
                diagnostics={"heldout_excluded": True},
            )
            for family in HISTORY_FAMILIES[1:]
        )
        return (identity, *unavailable)

    shift_x = 0.0
    shift_y = 0.0
    xtx = np.zeros((3, 3), dtype=np.float64)
    xty = np.zeros((3, 2), dtype=np.float64)
    correspondence_count = 0
    for left_pair, right_pair in training:
        left = population.supports[left_pair]
        right = population.supports[right_pair]
        left_x, left_y = _coordinates(left)
        right_x, right_y = _coordinates(right)
        shift_x += float(np.mean(right_x) - np.mean(left_x))
        shift_y += float(np.mean(right_y) - np.mean(left_y))
        pair_xtx, pair_xty, pair_count = (
            _quantile_correspondence_sufficient_statistics(left, right)
        )
        xtx += pair_xtx
        xty += pair_xty
        correspondence_count += pair_count

    dx = _round_half_away_from_zero(shift_x / len(training))
    dy = _round_half_away_from_zero(shift_y / len(training))
    if not (-32_768 <= dx <= 32_767 and -32_768 <= dy <= 32_767):
        xi_fit = ProgramFit(
            family="xi_advected",
            state=None,
            training_transition_count=len(training),
            training_correspondence_count=correspondence_count,
            status="NULL_FITTED_STATE_OUT_OF_INT16_RANGE",
            diagnostics={"heldout_excluded": True, "dx": dx, "dy": dy},
        )
    else:
        xi_fit = ProgramFit(
            family="xi_advected",
            state=_XI_STATE.pack(dx, dy),
            training_transition_count=len(training),
            training_correspondence_count=correspondence_count,
            status="HELDOUT_FIT_AVAILABLE",
            diagnostics={
                "heldout_excluded": True,
                "translation_dx": dx,
                "translation_dy": dy,
                "state_format": "int16le_dx_dy",
            },
        )

    rank = int(np.linalg.matrix_rank(xtx))
    affine_state: bytes | None = None
    affine_status = "NULL_AFFINE_NORMAL_EQUATIONS_RANK_DEFICIENT"
    affine_diagnostics: dict[str, Any] = {
        "heldout_excluded": True,
        "normal_equation_rank": rank,
        "state_format": "q16le_a00_a01_tx_a10_a11_ty",
    }
    if rank == 3 and correspondence_count >= 3:
        coefficients = np.linalg.solve(xtx, xty)
        ordered = (
            coefficients[0, 0],
            coefficients[1, 0],
            coefficients[2, 0],
            coefficients[0, 1],
            coefficients[1, 1],
            coefficients[2, 1],
        )
        quantized = tuple(
            _round_half_away_from_zero(float(value) * _Q16)
            for value in ordered
        )
        affine_diagnostics["coefficients_float64"] = [float(value) for value in ordered]
        affine_diagnostics["coefficients_q16"] = list(quantized)
        if all(-(1 << 31) <= value < (1 << 31) for value in quantized):
            affine_state = _AFFINE_STATE.pack(*quantized)
            affine_status = "HELDOUT_FIT_AVAILABLE"
        else:
            affine_status = "NULL_FITTED_STATE_OUT_OF_INT32_RANGE"
    affine_fit = ProgramFit(
        family="affine_tracked",
        state=affine_state,
        training_transition_count=len(training),
        training_correspondence_count=correspondence_count,
        status=affine_status,
        diagnostics=affine_diagnostics,
    )
    return identity, xi_fit, affine_fit


def _round_q16_ties_positive(values: np.ndarray) -> np.ndarray:
    return np.floor_divide(values + (_Q16 // 2), _Q16)


def _transform_support(
    support: np.ndarray,
    family: str,
    state: bytes,
) -> np.ndarray:
    flat = np.asarray(support, dtype=np.uint32)
    if flat.ndim != 1 or len(flat) == 0:
        raise ConnectionCodelengthError("history input support must be nonempty 1D")
    x, y = _coordinates(flat)
    if family == "identity":
        if state:
            raise ConnectionCodelengthError("identity state must be empty")
        transformed_x, transformed_y = x, y
    elif family == "xi_advected":
        if len(state) != _XI_STATE.size:
            raise ConnectionCodelengthError("xi state has wrong byte length")
        dx, dy = _XI_STATE.unpack(state)
        transformed_x, transformed_y = x + dx, y + dy
    elif family == "affine_tracked":
        if len(state) != _AFFINE_STATE.size:
            raise ConnectionCodelengthError("affine state has wrong byte length")
        a00, a01, tx, a10, a11, ty = _AFFINE_STATE.unpack(state)
        transformed_x = _round_q16_ties_positive(a00 * x + a01 * y + tx)
        transformed_y = _round_q16_ties_positive(a10 * x + a11 * y + ty)
    else:
        raise ConnectionCodelengthError(f"unknown history family {family!r}")
    clipped_x = np.clip(transformed_x, 0, _IMAGE_W - 1)
    clipped_y = np.clip(transformed_y, 0, _IMAGE_H - 1)
    return np.asarray(clipped_y * _IMAGE_W + clipped_x, dtype=np.uint32)


def _transport_symbols(
    transformed_support: np.ndarray,
    winners: bytes,
    relations: bytes,
    *,
    target_support: np.ndarray | None,
) -> tuple[np.ndarray, bytes, bytes]:
    if len(transformed_support) != len(winners) or len(winners) != len(relations):
        raise ConnectionCodelengthError("history support/symbol cardinality mismatch")
    order = np.argsort(transformed_support, kind="stable")
    ordered_support = transformed_support[order]
    ordered_winners = np.frombuffer(winners, dtype=np.uint8)[order]
    ordered_relations = np.frombuffer(relations, dtype=np.uint8)[order]
    unique = np.concatenate(
        (np.asarray([True]), ordered_support[1:] != ordered_support[:-1])
    )
    predicted_support = ordered_support[unique]
    predicted_winners = ordered_winners[unique]
    predicted_relations = ordered_relations[unique]
    if target_support is None:
        return (
            np.asarray(predicted_support, dtype=np.uint32),
            predicted_winners.tobytes(),
            predicted_relations.tobytes(),
        )

    target = np.asarray(target_support, dtype=np.uint32)
    positions = np.searchsorted(predicted_support, target)
    upper = np.minimum(positions, len(predicted_support) - 1)
    lower = np.maximum(positions - 1, 0)
    lower_distance = np.abs(
        target.astype(np.int64) - predicted_support[lower].astype(np.int64)
    )
    upper_distance = np.abs(
        predicted_support[upper].astype(np.int64) - target.astype(np.int64)
    )
    selected = np.where(lower_distance <= upper_distance, lower, upper)
    return (
        target,
        predicted_winners[selected].tobytes(),
        predicted_relations[selected].tobytes(),
    )


def predict_record_raw(
    previous: SolvedValueRecord,
    previous_support: np.ndarray,
    target_support: np.ndarray,
    *,
    family: str,
    state: bytes,
) -> bytes:
    """Run one fixed generic history interpreter against prior decoded state."""

    if support_sha256(previous_support) != previous.support_sha256:
        raise ConnectionCodelengthError("previous external support SHA mismatch")
    transformed = _transform_support(previous_support, family, state)
    cell = previous.stratum == "cell"
    predicted_support, winners, relations = _transport_symbols(
        transformed,
        previous.winners,
        previous.margin_relations,
        target_support=target_support if cell else None,
    )
    predicted = SolvedValueRecord(
        pair_id=previous.pair_id + 1,
        bucket_id=previous.bucket_id,
        class_left=previous.class_left,
        class_right=previous.class_right,
        stream_type=previous.stream_type,
        layer_home=previous.layer_home,
        support_sha256=support_sha256(predicted_support),
        winners=winners,
        margin_relations=relations,
        flat_indices=(
            tuple(int(value) for value in predicted_support)
            if not cell
            else ()
        ),
    )
    raw = predicted.encode()
    parsed = SolvedValueRecord.decode(
        raw,
        external_cell_support=predicted_support if cell else None,
    )
    if parsed != predicted:
        raise ConnectionCodelengthError("history prediction record parseback failed")
    return raw


def _xor_against_prediction(target_raw: bytes, predicted_raw: bytes) -> bytes:
    return bytes(
        value ^ (predicted_raw[index] if index < len(predicted_raw) else 0)
        for index, value in enumerate(target_raw)
    )


def _unxor_against_prediction(residual: bytes, predicted_raw: bytes) -> bytes:
    return _xor_against_prediction(residual, predicted_raw)


def encode_history_packet(
    target_raw: bytes,
    previous: SolvedValueRecord,
    previous_support: np.ndarray,
    target_support: np.ndarray,
    *,
    family: str,
    state: bytes,
    codec: str,
) -> bytes:
    """Encode and independently parse one exact charged history packet."""

    if family not in _HISTORY_IDS:
        raise ConnectionCodelengthError(f"unknown history family {family!r}")
    predicted_raw = predict_record_raw(
        previous,
        previous_support,
        target_support,
        family=family,
        state=state,
    )
    residual = _xor_against_prediction(target_raw, predicted_raw)
    residual_container = encode_codec(residual, codec)
    packet = (
        _HISTORY_HEADER.pack(
            HISTORY_MAGIC,
            1,
            _HISTORY_IDS[family],
            len(state),
            len(predicted_raw),
        )
        + state
        + residual_container
    )
    decoded = decode_history_packet(
        packet,
        previous,
        previous_support,
        target_support,
    )
    if decoded["target_raw"] != target_raw or decoded["codec"] != codec:
        raise ConnectionCodelengthError("history packet exact parseback failed")
    return packet


def decode_history_packet(
    packet: bytes,
    previous: SolvedValueRecord,
    previous_support: np.ndarray,
    target_support: np.ndarray,
) -> Mapping[str, Any]:
    if len(packet) < _HISTORY_HEADER.size:
        raise ConnectionCodelengthError("history packet is truncated")
    magic, version, family_id, state_length, predicted_length = (
        _HISTORY_HEADER.unpack_from(packet)
    )
    if magic != HISTORY_MAGIC or version != 1 or family_id not in _ID_HISTORY:
        raise ConnectionCodelengthError("history packet magic/version/family mismatch")
    state_end = _HISTORY_HEADER.size + state_length
    if state_end >= len(packet):
        raise ConnectionCodelengthError("history packet state/residual is truncated")
    family = _ID_HISTORY[family_id]
    state = packet[_HISTORY_HEADER.size : state_end]
    residual_container = packet[state_end:]
    try:
        codec, residual = decode_codec(residual_container)
    except SolvedValuePricingError as error:
        raise ConnectionCodelengthError(
            "history residual container failed exact decode"
        ) from error
    predicted_raw = predict_record_raw(
        previous,
        previous_support,
        target_support,
        family=family,
        state=state,
    )
    if len(predicted_raw) != predicted_length:
        raise ConnectionCodelengthError("history predicted-length field mismatch")
    target_raw = _unxor_against_prediction(residual, predicted_raw)
    target = SolvedValueRecord.decode(
        target_raw,
        external_cell_support=(
            target_support if previous.stratum == "cell" else None
        ),
    )
    if (
        target.pair_id != previous.pair_id + 1
        or target.bucket_id != previous.bucket_id
        or target.class_left != previous.class_left
        or target.class_right != previous.class_right
        or target.stream_type is not previous.stream_type
        or target.layer_home is not previous.layer_home
    ):
        raise ConnectionCodelengthError(
            "history target differs from the required same-bucket consecutive row"
        )
    canonical = (
        _HISTORY_HEADER.pack(
            HISTORY_MAGIC,
            1,
            family_id,
            len(state),
            len(predicted_raw),
        )
        + state
        + encode_codec(residual, codec)
    )
    if canonical != packet:
        raise ConnectionCodelengthError("history packet is not canonical")
    return {
        "family": family,
        "state": state,
        "codec": codec,
        "target_raw": target_raw,
        "predicted_raw": predicted_raw,
        "residual_raw": residual,
    }


def price_history(
    target: SolvedSupport,
    previous: SolvedSupport,
    fits: Sequence[ProgramFit],
) -> tuple[dict[str, Any], str]:
    """Price every available family/coder and select with a sealed tie-break."""

    if target.record.pair_id != previous.record.pair_id + 1:
        raise ConnectionCodelengthError("history pricing requires consecutive pair ids")
    if target.record.bucket_id != previous.record.bucket_id:
        raise ConnectionCodelengthError("history pricing requires one semantic bucket")
    by_family: dict[str, Any] = {}
    for fit in fits:
        if fit.family not in HISTORY_FAMILIES:
            raise ConnectionCodelengthError("program fit introduced a new family")
        if fit.state is None:
            by_family[fit.family] = {
                "status": fit.status,
                "training_transition_count": fit.training_transition_count,
                "training_correspondence_count": fit.training_correspondence_count,
                "state_bytes": None,
                "prices": None,
                "winning_codec": None,
                "exact_counted_bytes": None,
                "diagnostics": dict(fit.diagnostics),
            }
            continue
        prices: dict[str, Any] = {}
        for codec in CODECS:
            packet = encode_history_packet(
                target.raw,
                previous.record,
                previous.support,
                target.support,
                family=fit.family,
                state=fit.state,
                codec=codec,
            )
            prices[codec] = {
                "packet_bytes": len(packet),
                "packet_sha256": sha256_bytes(packet),
                "history_program_bytes": _HISTORY_HEADER.size + len(fit.state),
                "program_selector_bytes": 1,
                "program_state_bytes": len(fit.state),
                "program_fixed_framing_bytes": _HISTORY_HEADER.size - 1,
                "residual_container_bytes": (
                    len(packet) - _HISTORY_HEADER.size - len(fit.state)
                ),
                "parseback_exact": True,
            }
        winner = min(
            CODECS,
            key=lambda codec: (
                prices[codec]["packet_bytes"],
                CODECS.index(codec),
            ),
        )
        by_family[fit.family] = {
            "status": fit.status,
            "training_transition_count": fit.training_transition_count,
            "training_correspondence_count": fit.training_correspondence_count,
            "state_bytes": len(fit.state),
            "state_sha256": sha256_bytes(fit.state),
            "prices": prices,
            "winning_codec": winner,
            "exact_counted_bytes": prices[winner]["packet_bytes"],
            "diagnostics": dict(fit.diagnostics),
        }
    available = [
        family
        for family in HISTORY_FAMILIES
        if by_family[family]["exact_counted_bytes"] is not None
    ]
    if not available:
        raise ConnectionCodelengthError("no history program is priceable")
    winning_family = min(
        available,
        key=lambda family: (
            by_family[family]["exact_counted_bytes"],
            HISTORY_FAMILIES.index(family),
        ),
    )
    return by_family, winning_family


def _contiguous_runs(values: Sequence[int], *, maximum: int = 12) -> tuple[tuple[int, ...], ...]:
    ordered = tuple(sorted({int(value) for value in values}))
    runs: list[list[int]] = []
    for value in ordered:
        if not runs or value != runs[-1][-1] + 1 or len(runs[-1]) >= maximum:
            runs.append([value])
        else:
            runs[-1].append(value)
    return tuple(tuple(run) for run in runs)


def _materialize_solved_holdouts(
    config: Mapping[str, Any],
    source_config: SolveDiffMiningConfigV1,
    populations: Sequence[SupportPopulation],
) -> tuple[Mapping[tuple[str, int], SolvedSupport], Mapping[str, str]]:
    required: dict[int, list[SupportPopulation]] = {}
    for population in populations:
        if population.holdout is None:
            continue
        for pair_id in population.holdout:
            required.setdefault(pair_id, []).append(population)
    kernel = FullResizeKernel.build()
    context = _open_production_inputs(source_config)

    import torch

    from tac.scorer import load_default_segnet

    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True)
    segnet = load_default_segnet(config["upstream_dir"], device="cpu")

    output: dict[tuple[str, int], SolvedSupport] = {}
    source_hashes: dict[str, str] = {}
    for run in _contiguous_runs(tuple(required)):
        chunk = _load_production_inputs(context, source_config, run, kernel)
        source_hashes.update(chunk.source_hashes)
        for local, pair_id in enumerate(chunk.pair_ids):
            solved_pair = chunk.solved_planes[local]
            camera = np.stack(
                [
                    realize_solve_camera(solved_pair[frame], kernel)
                    for frame in range(2)
                ],
                axis=0,
            )
            logits, realized_last = _segnet_logits(segnet, camera)
            rounded = np.clip(np.rint(realized_last), 0, 255).astype(np.uint8)
            if not np.array_equal(rounded, solved_pair[1]):
                raise ConnectionCodelengthError(
                    f"pair {pair_id} failed exact solved-plane camera roundtrip"
                )
            roundtrip_max_abs = float(
                np.max(np.abs(realized_last.astype(np.float64) - solved_pair[1]))
            )
            for population in required[pair_id]:
                support = population.supports[pair_id]
                left_name, right_name = _class_pair_from_bucket(population.bucket_id)
                class_left, class_right = _class_id(left_name), _class_id(right_name)
                winners, relations, summary = _winner_symbols(
                    logits,
                    support,
                    class_left,
                    class_right,
                )
                stratum = _stratum_from_bucket(population.bucket_id)
                record = SolvedValueRecord(
                    pair_id=pair_id,
                    bucket_id=population.bucket_id,
                    class_left=class_left,
                    class_right=class_right,
                    stream_type=(
                        StreamType.SKELETON
                        if stratum == "boundary"
                        else StreamType.FIBER
                    ),
                    layer_home=LayerHome.L4_SCORER_FEATURE,
                    support_sha256=support_sha256(support),
                    winners=winners,
                    margin_relations=relations,
                    flat_indices=(
                        tuple(int(value) for value in support)
                        if stratum == "boundary"
                        else ()
                    ),
                )
                raw = record.encode()
                parsed = SolvedValueRecord.decode(
                    raw,
                    external_cell_support=(
                        support if stratum == "cell" else None
                    ),
                )
                if parsed != record:
                    raise ConnectionCodelengthError(
                        "held-out semantic record exact parseback failed"
                    )
                output[(population.bucket_id, pair_id)] = SolvedSupport(
                    record=record,
                    support=support,
                    raw=raw,
                    solved_summary=summary,
                    roundtrip_max_abs=roundtrip_max_abs,
                )
            del camera, logits, realized_last, solved_pair
        del chunk
    expected = {
        (population.bucket_id, pair_id)
        for population in populations
        if population.holdout is not None
        for pair_id in population.holdout
    }
    if set(output) != expected:
        raise ConnectionCodelengthError("not all held-out support endpoints were solved")
    return output, dict(sorted(source_hashes.items()))


def _connection_tag(counted_bytes: int) -> Mapping[str, Any]:
    return TypedStreamTag(
        type=StreamType.CONNECTION,
        layer_home=LayerHome.L4_SCORER_FEATURE,
        evaluate_py_recursion_level_cited=(
            "L4 frozen-SegNet semantic state at pair p is predicted from the "
            "same-bucket L4 state at p-1; selector and fitted state are charged"
        ),
        counted_bytes=counted_bytes,
        free_receiver_code=True,
    ).to_dict()


def _residual_tag(counted_bytes: int) -> Mapping[str, Any]:
    return TypedStreamTag(
        type=StreamType.RESIDUAL,
        layer_home=LayerHome.L4_SCORER_FEATURE,
        evaluate_py_recursion_level_cited=(
            "exact XOR correction restores the later canonical L4 semantic record"
        ),
        counted_bytes=counted_bytes,
        free_receiver_code=True,
    ).to_dict()


def _next_measurement(delta_bytes: int) -> str:
    if delta_bytes > 0:
        return (
            "Run every held-out fold for this bucket and require the positive "
            "conditional-byte sign to survive before any receiver-family acquisition use."
        )
    return (
        "Keep CONNECTION open and test a separately preregistered generic piecewise "
        "history program on this bucket with identical leave-one-out charge accounting."
    )


def _group_decomposition(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["bucket_type"]),
            str(row["stratum"]),
            int(row["later_support"]["count"]),
        )
        groups.setdefault(key, []).append(row)
    exact = []
    for (bucket_type, stratum, support_size), selected in sorted(groups.items()):
        static = sum(int(row["B_static"]) for row in selected)
        program = sum(int(row["B_history_program"]) for row in selected)
        residual = sum(int(row["B_residual"]) for row in selected)
        exact.append(
            {
                "bucket_type": bucket_type,
                "stratum": stratum,
                "support_size": support_size,
                "row_count": len(selected),
                "B_static": static,
                "B_history_program": program,
                "B_residual": residual,
                "delta_B_connection": static - program - residual,
            }
        )
    return {
        "schema": "ddm_dm3_bucket_type_x_stratum_x_support_size.v1",
        "support_size_semantics": "exact later PF2 occupied-support count; no bins",
        "rows": exact,
    }


def _winning_family_summary(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    positive_total = sum(max(0, int(row["delta_B_connection"])) for row in rows)
    output: dict[str, Any] = {}
    for family in HISTORY_FAMILIES:
        selected = [row for row in rows if row["winning_history_family"] == family]
        positive = sum(max(0, int(row["delta_B_connection"])) for row in selected)
        output[family] = {
            "selected_bucket_count": len(selected),
            "selected_bucket_share": len(selected) / len(rows),
            "positive_bucket_count": sum(
                int(row["delta_B_connection"]) > 0 for row in selected
            ),
            "positive_savings_bytes": positive,
            "positive_savings_share": (
                positive / positive_total if positive_total else None
            ),
        }
    return output


def _implementation_custody() -> Mapping[str, Any]:
    paths = (
        Path(__file__).resolve(),
        _REPO / "tools/measure_ddm_dm3_connection_conditional_codelength.py",
        _REPO / "src/tac/optimization/ddm_dm1_solved_value_pricing.py",
        _REPO / "src/tac/lossless/range_coder.py",
        _REPO / "src/tac/optimization/ddm_min_description_contract.py",
        _REPO / "src/tac/optimization/resize_full_kernel.py",
        _REPO / "src/tac/optimization/solve_diff_operator_mining.py",
        _REPO / "src/tac/scorer.py",
    )
    result: dict[str, Any] = {}
    for path in paths:
        if not path.is_file():
            raise ConnectionCodelengthError(f"implementation file is absent: {path}")
        relative = str(path.relative_to(_REPO))
        result[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ConnectionCodelengthError(
                f"refusing to overwrite unequal artifact {path}"
            )
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _dm1_crosswalk(
    dm1_receipt: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    by_bucket = {str(row["bucket_id"]): row for row in rows}
    crosswalk = []
    for source in dm1_receipt.get("rows", ()):
        bucket_id = str(source["bucket_id"])
        measured = by_bucket.get(bucket_id)
        if measured is None:
            crosswalk.append(
                {
                    "dm1_row_index": int(source["row_index"]),
                    "dm1_pair_id": int(source["pair_id"]),
                    "bucket_id": bucket_id,
                    "connection_status": "NULL_NO_ELIGIBLE_BUCKET_MEASUREMENT",
                    "delta_B_connection": None,
                    "winning_history_family": None,
                }
            )
        else:
            crosswalk.append(
                {
                    "dm1_row_index": int(source["row_index"]),
                    "dm1_pair_id": int(source["pair_id"]),
                    "bucket_id": bucket_id,
                    "connection_status": "MEASURED_BUCKET_FAMILY_HELDOUT",
                    "measurement_scope": (
                        "same bucket family, deterministic held-out consecutive "
                        "transition; not the original nonconsecutive DM1 pair"
                    ),
                    "delta_B_connection": int(measured["delta_B_connection"]),
                    "winning_history_family": measured[
                        "winning_history_family"
                    ],
                    "dm3_row_index": int(measured["row_index"]),
                }
            )
    return {
        "source_dm1_row_count": len(crosswalk),
        "measured_bucket_family_cells": sum(
            row["connection_status"] == "MEASURED_BUCKET_FAMILY_HELDOUT"
            for row in crosswalk
        ),
        "rows": crosswalk,
    }


def materialize(config_path: str | Path, output_dir: str | Path) -> Mapping[str, Any]:
    """Run the bounded 36-bucket held-out semantic conditional-code probe."""

    config_path = Path(config_path)
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    if not isinstance(config, Mapping) or config.get("schema") != CONFIG_SCHEMA:
        raise ConnectionCodelengthError("DM3 config schema mismatch")
    _validate_false_authority(config)
    for path_key, sha_key in (
        ("source_config_path", "source_config_sha256"),
        ("pf2_event_index_path", "pf2_event_index_sha256"),
        ("pf2_index_receipt_path", "pf2_index_receipt_sha256"),
        ("segnet_weights_path", "segnet_weights_sha256"),
        ("upstream_modules_path", "upstream_modules_sha256"),
        ("dm1_receipt_path", "dm1_receipt_sha256"),
        ("gc1_council_path", "gc1_council_sha256"),
    ):
        if sha256_file(config[path_key]) != config[sha_key]:
            raise ConnectionCodelengthError(f"{path_key} SHA-256 mismatch")
    source_config = SolveDiffMiningConfigV1.model_validate_json(
        Path(config["source_config_path"]).read_bytes()
    )
    if (
        source_config.solved_planes_receipt_path
        != config["solved_planes_receipt_path"]
        or source_config.solved_planes_receipt_sha256
        != config["solved_planes_receipt_sha256"]
    ):
        raise ConnectionCodelengthError("source config-to-solved receipt binding drifted")
    if (
        sha256_file(config["solved_planes_receipt_path"])
        != config["solved_planes_receipt_sha256"]
    ):
        raise ConnectionCodelengthError("solved-plane receipt SHA-256 mismatch")
    index_receipt = checked_json(
        config["pf2_index_receipt_path"],
        config["pf2_index_receipt_sha256"],
    )
    if index_receipt.get("index_sha256") != config["pf2_event_index_sha256"]:
        raise ConnectionCodelengthError("PF2 receipt-to-index SHA binding mismatch")
    bucket_arrays = index_receipt.get("bucket_arrays")
    if not isinstance(bucket_arrays, Mapping):
        raise ConnectionCodelengthError("PF2 bucket-array mapping is missing")
    dm1_receipt = checked_json(
        config["dm1_receipt_path"],
        config["dm1_receipt_sha256"],
    )
    if dm1_receipt.get("schema") != "ddm_dm1_solved_value_pricing.v1":
        raise ConnectionCodelengthError("DM1 receipt schema drifted")

    with np.load(config["pf2_event_index_path"], allow_pickle=False) as event_index:
        populations = _build_support_population(event_index, bucket_arrays)
    eligible = tuple(
        population for population in populations if population.holdout is not None
    )
    ineligible = tuple(
        population for population in populations if population.holdout is None
    )
    if len(populations) != 37 or len(eligible) != 36:
        raise ConnectionCodelengthError(
            "preregistered PF2 population drifted from 37 represented / 36 eligible"
        )
    if sum(len(population.transitions) for population in eligible) != 8_602:
        raise ConnectionCodelengthError(
            "preregistered consecutive-transition population drifted from 8602"
        )

    solved, streamed_hashes = _materialize_solved_holdouts(
        config,
        source_config,
        eligible,
    )
    rows: list[Mapping[str, Any]] = []
    for row_index, population in enumerate(eligible):
        assert population.holdout is not None
        left_pair, right_pair = population.holdout
        previous = solved[(population.bucket_id, left_pair)]
        target = solved[(population.bucket_id, right_pair)]
        static_prices, static_codec = price_raw(target.raw)
        static_bytes = int(static_prices[static_codec]["container_bytes"])
        fits = fit_programs(population)
        history_prices, winning_family = price_history(target, previous, fits)
        winning = history_prices[winning_family]
        winning_codec = str(winning["winning_codec"])
        winning_price = winning["prices"][winning_codec]
        program_bytes = int(winning_price["history_program_bytes"])
        residual_bytes = int(winning_price["residual_container_bytes"])
        delta = static_bytes - program_bytes - residual_bytes
        stratum = target.record.stratum
        rows.append(
            {
                "schema": ROW_SCHEMA,
                "row_index": row_index,
                "bucket_id": population.bucket_id,
                "bucket_type": _bucket_type(population.bucket_id),
                "stratum": stratum,
                "heldout": {
                    "policy": HOLDOUT_POLICY,
                    "left_pair_id": left_pair,
                    "right_pair_id": right_pair,
                    "pair_gap": 1,
                    "eligible_transition_count": len(population.transitions),
                    "fit_transition_count": len(population.transitions) - 1,
                    "priced_transition_excluded_from_fit": True,
                },
                "previous_support": {
                    "count": len(previous.support),
                    "sha256_uint32le": support_sha256(previous.support),
                },
                "later_support": {
                    "count": len(target.support),
                    "sha256_uint32le": support_sha256(target.support),
                },
                "later_solved_record": {
                    "raw_bytes": len(target.raw),
                    "raw_sha256": sha256_bytes(target.raw),
                    "parseback_exact": True,
                    "solved_value": dict(target.solved_summary),
                    "solved_plane_roundtrip_max_abs": target.roundtrip_max_abs,
                },
                "static_typed_home": typed_home(stratum, static_bytes).to_dict(),
                "static_prices": static_prices,
                "static_winning_codec": static_codec,
                "B_static": static_bytes,
                "history_candidates": history_prices,
                "winning_history_family": winning_family,
                "winning_history_codec": winning_codec,
                "history_program_tag": _connection_tag(program_bytes),
                "history_residual_tag": _residual_tag(residual_bytes),
                "B_history_program": program_bytes,
                "B_residual": residual_bytes,
                "B_history_total": program_bytes + residual_bytes,
                "delta_B_connection": delta,
                "connection_evidence": (
                    "POSITIVE_HELDOUT_CONNECTION_EVIDENCE"
                    if delta > 0
                    else "INSTANCE_SCOPED_SURPRISAL_AT_THIS_FORMULATION"
                ),
                "next_measurement": _next_measurement(delta),
                "verdict_scope": (
                    "One deterministic leave-one-transition-out fold for this "
                    "bucket on exact DM1 semantic records. External PF2 support "
                    "remains context for FIBER rows. No receiver/archive/Pose/"
                    "contest-axis inference."
                ),
            }
        )

    aggregate_static = sum(int(row["B_static"]) for row in rows)
    aggregate_program = sum(int(row["B_history_program"]) for row in rows)
    aggregate_residual = sum(int(row["B_residual"]) for row in rows)
    aggregate_delta = aggregate_static - aggregate_program - aggregate_residual
    result = {
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "lane_id": config["lane_id"],
        "source_commit": config["source_commit"],
        "config_path": str(config_path),
        "config_sha256": sha256_bytes(config_raw),
        "custody": {
            "source_config_path": config["source_config_path"],
            "source_config_sha256": config["source_config_sha256"],
            "solved_planes_receipt_path": config["solved_planes_receipt_path"],
            "solved_planes_receipt_sha256": config[
                "solved_planes_receipt_sha256"
            ],
            "pf2_event_index_path": config["pf2_event_index_path"],
            "pf2_event_index_sha256": config["pf2_event_index_sha256"],
            "pf2_index_receipt_path": config["pf2_index_receipt_path"],
            "pf2_index_receipt_sha256": config["pf2_index_receipt_sha256"],
            "segnet_weights_path": config["segnet_weights_path"],
            "segnet_weights_sha256": config["segnet_weights_sha256"],
            "upstream_modules_path": config["upstream_modules_path"],
            "upstream_modules_sha256": config["upstream_modules_sha256"],
            "dm1_receipt_path": config["dm1_receipt_path"],
            "dm1_receipt_sha256": config["dm1_receipt_sha256"],
            "gc1_council_path": config["gc1_council_path"],
            "gc1_council_sha256": config["gc1_council_sha256"],
            "streamed_solved_chunk_hashes": streamed_hashes,
            "implementation": _implementation_custody(),
            "runtime": {
                "platform": platform.platform(),
                "python": sys.version,
                "numpy": np.__version__,
                "torch_threads": 4,
                "seed": 1234,
            },
        },
        "population": {
            "represented_bucket_count": len(populations),
            "eligible_bucket_count": len(eligible),
            "ineligible_bucket_count": len(ineligible),
            "eligible_consecutive_transition_count": sum(
                len(population.transitions) for population in eligible
            ),
            "heldout_row_count": len(rows),
            "holdout_policy": HOLDOUT_POLICY,
            "fit_policy": FIT_POLICY,
            "ineligible_buckets": [
                {
                    "bucket_id": population.bucket_id,
                    "represented_pair_count": len(population.supports),
                    "connection_bytes": None,
                    "next_measurement": (
                        "Acquire one exact same-bucket consecutive support transition; "
                        "do not impute CONNECTION from another bucket."
                    ),
                }
                for population in ineligible
            ],
        },
        "rows": rows,
        "aggregate": {
            "B_static": aggregate_static,
            "B_history_program": aggregate_program,
            "B_residual": aggregate_residual,
            "B_history_total": aggregate_program + aggregate_residual,
            "delta_B_connection": aggregate_delta,
            "positive_bucket_count": sum(
                int(row["delta_B_connection"]) > 0 for row in rows
            ),
            "nonpositive_bucket_count": sum(
                int(row["delta_B_connection"]) <= 0 for row in rows
            ),
            "verdict": (
                "POSITIVE_HELDOUT_CONNECTION_EVIDENCE"
                if aggregate_delta > 0
                else "AGGREGATE_INSTANCE_SCOPED_SURPRISAL_AT_THIS_FORMULATION"
            ),
            "winning_family_share": _winning_family_summary(rows),
        },
        "decomposition": _group_decomposition(rows),
        "dm1_connection_null_crosswalk": _dm1_crosswalk(dm1_receipt, rows),
        "directive_consumption": [
            {
                "directive": "DDM-CONNECTION-1 same-bucket consecutive supports",
                "disposition": "CONSUMED",
                "evidence": "37 represented PF2 buckets, 36 eligible, 8602 transitions",
            },
            {
                "directive": "identity / xi-advected / affine-tracked generic programs",
                "disposition": "CONSUMED_NO_NEW_TYPOLOGY",
                "evidence": list(HISTORY_FAMILIES),
            },
            {
                "directive": "leave-one-pair-out with all selected state charged",
                "disposition": "CONSUMED",
                "evidence": (
                    "one lower-median held-out transition per eligible bucket; "
                    "all other transitions fit state; 16B packet framing plus "
                    "0/4/24B state and full DM1 residual container charged"
                ),
            },
            {
                "directive": "no RG4 / no reachability vocabulary",
                "disposition": "PASS",
                "evidence": "only sealed PF2 supports, DM1 records, and five-type tags opened",
            },
            {
                "directive": "broadcast Fisher/waterfill residual guidance",
                "disposition": "CONSULTED_NOT_APPLICABLE_TO_BYTE_ONLY_PROBE",
                "evidence": (
                    "no scorer actuation, basis residual, score admission, or "
                    "waterfill decision is performed"
                ),
            },
        ],
        "context_only": {
            "dm1_joint_semantic_bytes": 1_569,
            "receiver_realization_bytes": None,
            "score_slack_arithmetic_permitted": False,
            "new_box_arithmetic_performed": False,
            "interpretation": (
                "Conditional semantic-record savings, if positive, are a first "
                "program-structure rung only and are not archive-byte savings."
            ),
        },
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_emitted": False,
        "pointer_moved": False,
        "pointer": POINTER,
        "evidence_axis": AXIS,
        "main_review_required": True,
        "verdict_scope": (
            "36 deterministic bucket-level held-out semantic conditional-code "
            "rows on the local frozen-SegNet axis. No all-fold estimate, RGB "
            "receiver, Pose verdict, archive, evaluate.py, contest axis, score, "
            "promotion, or frontier mutation."
        ),
    }
    payload = canonical_json_bytes(result)
    output_dir = Path(output_dir)
    receipt_path = (
        output_dir / "ddm_dm3_connection1_conditional_codelength_receipt.json"
    )
    _atomic_write(receipt_path, payload)
    manifest = {
        "schema": "ddm_dm3_connection_conditional_codelength_manifest.v1",
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256(payload).hexdigest(),
        "receipt_bytes": len(payload),
        "large_artifacts_created": False,
        "auto_cleanup": "not_applicable_streamed_planes_bounded_records_only",
        "score_claim": False,
        "pointer_moved": False,
        "main_review_required": True,
    }
    _atomic_write(output_dir / "manifest.json", canonical_json_bytes(manifest))
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "FIT_POLICY",
    "HISTORY_FAMILIES",
    "HOLDOUT_POLICY",
    "ConnectionCodelengthError",
    "ProgramFit",
    "SolvedSupport",
    "SupportPopulation",
    "decode_history_packet",
    "encode_history_packet",
    "fit_programs",
    "materialize",
    "predict_record_raw",
    "price_history",
]
