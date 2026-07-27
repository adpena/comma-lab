# SPDX-License-Identifier: MIT
"""Exact parsed-G105 semantic stage selection on the receiver-valid G110 wire.

This module measures one preserved G111 semantic state at the public receiver
boundary.  It deliberately does not score the bare G105 packet.  Each canonical
G105 Y1 wire family is wrapped in G110's rank-zero ``Y0 == Y1`` product, raced
under both counted outer-ZIP methods, parsed back, decoded through both the G105
and G110 receivers, realized through the exact factor-2 uint8 camera preimage,
and evaluated for all 600 pairs by an injected SegNet argmax callback.

Only the semantic lower bound is measured here:

``100 * d_seg + 25 * archive_bytes / 37_545_489``.

Pose is absent, so this is an optimistic pose-refit calculation rather than a
score, candidate, pointer, or production lower-bound verdict.  The selected receiver packet,
semantic-only counted archive, and canonical receipt are small durable stage
artifacts written atomically to a caller-owned non-temporary directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.witness_dsl.g105_public_wire_quantization_surface_v1 import (
    compile_g105_public_wire_quantization_surface_numpy,
)
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    SCORER_H,
    SCORER_W,
    V9RuntimeConfigV1,
    Y1WireCodecV1,
    render_scorer_y1,
)
from tac.witness_dsl.taskspace_g110_generic_two_layer_public_product_v1 import (
    G110OuterZipMethodV1,
    build_g110_counted_archive_variant,
    build_g110_public_archive,
    build_g110_rank_zero_semantic_floor_packet,
    parse_g110_counted_archive_variant,
    parse_g110_public_archive,
    parse_g110_two_layer_v1,
    render_g110_rank_zero_scorer_pair,
)
from tac.witness_dsl.v10_factor2_selected_preimage_v1 import (
    CAMERA_HW,
    realize_factor2_uint8_numpy,
)

SCHEMA: Final = "tac.g111_parsed_g105_stage_selector.v1"
BATCH_PROGRESS_SCHEMA: Final = "tac.g111_parsed_g105_stage_selector.batch_progress.v1"
CROSS_STAGE_PARETO_ROW_SCHEMA: Final = "tac.g111_parsed_g105_stage_selector.cross_stage_pareto_row.v1"
POPULATION_HASH_SCHEMA: Final = "sha256_canonical_batch_digest_chain.v1"
EXECUTION_SURFACE: Final = "engine_only_untrusted_injected_inputs"
PAIR_COUNT_N600: Final = 600
N_CLASSES: Final = 5
SCORER_CHANNELS: Final = 3
VERDICT_BATCH_SIZE: Final = 16
VERDICT_BATCH_SIZES: Final = (16,) * 37 + (8,)
ARCHIVE_RATE_DENOMINATOR: Final = 37_545_489
SEG_SCORE_WEIGHT: Final = 100.0
RATE_SCORE_WEIGHT: Final = 25.0
_STAGE_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_LOWER_SHA256 = frozenset("0123456789abcdef")
_EPHEMERAL_ROOTS = (
    Path("/tmp"),
    Path("/private/tmp"),
    Path("/var/tmp"),
)
_Y1_WIRE_CODECS = (
    Y1WireCodecV1.RAW_I16_LE,
    Y1WireCodecV1.DELTA_RICE_BEST_K,
)
_OUTER_ZIP_METHODS = (
    G110OuterZipMethodV1.STORE,
    G110OuterZipMethodV1.DEFLATE,
)

SegArgmaxBatchScorerV1 = Callable[[np.ndarray], np.ndarray]


class G111ParsedG105StageSelectorError(ValueError):
    """The exact semantic-only stage selector failed closed."""


def _sha256(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise G111ParsedG105StageSelectorError("stage receipt must be finite canonical ASCII JSON") from exc


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in _LOWER_SHA256 for character in value):
        raise G111ParsedG105StageSelectorError(f"{name} must be a lowercase SHA-256")
    return value


def _pose_reserve_record(
    *,
    measured_pose_reserve_bytes: int | None,
    measured_pose_reserve_receipt_sha256: str | None,
) -> dict[str, Any]:
    if measured_pose_reserve_bytes is None:
        if measured_pose_reserve_receipt_sha256 is not None:
            raise G111ParsedG105StageSelectorError("unmeasured zero pose reserve cannot carry a measured receipt")
        return {
            "status": "zero_unmeasured_semantic_only",
            "bytes": 0,
            "receipt_sha256": None,
            "included_in_semantic_selection": False,
        }
    if type(measured_pose_reserve_bytes) is not int or measured_pose_reserve_bytes <= 0:
        raise G111ParsedG105StageSelectorError("measured pose reserve bytes must be an exact positive integer")
    _require_sha256(
        measured_pose_reserve_receipt_sha256,
        name="measured pose reserve receipt",
    )
    return {
        "status": "measured_reserved_bytes",
        "bytes": measured_pose_reserve_bytes,
        "receipt_sha256": measured_pose_reserve_receipt_sha256,
        "included_in_semantic_selection": False,
    }


def _artifact_namespace(
    *,
    stage_tag: str,
    pointer_snapshot_identity_sha256: str,
) -> str:
    if type(stage_tag) is not str or _STAGE_TAG.fullmatch(stage_tag) is None or stage_tag in {".", ".."}:
        raise G111ParsedG105StageSelectorError("stage_tag must be safe ASCII filename text")
    _require_sha256(
        pointer_snapshot_identity_sha256,
        name="pointer snapshot identity",
    )
    return f"{stage_tag}.ptr_{pointer_snapshot_identity_sha256}"


def _require_exact_float32(
    values: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if (
        type(values) is not np.ndarray
        or values.dtype != np.dtype(np.float32)
        or not values.flags.c_contiguous
        or not values.size
        or not np.isfinite(values).all()
        or (shape is not None and values.shape != shape)
    ):
        expected = "" if shape is None else f" with shape {shape}"
        raise G111ParsedG105StageSelectorError(f"{name} must be an exact finite C-contiguous float32 ndarray{expected}")
    return values


def _expected_semantic_param_shapes(
    config: V9RuntimeConfigV1,
) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_dim
    layers = config.hidden_layer_count
    modulation = config.modulation_dim
    expected: dict[str, tuple[int, ...]] = {
        "in_proj.weight": (hidden, config.input_dim),
        "in_proj.bias": (hidden,),
        "film.weight": (2 * hidden * layers, modulation),
        "film.bias": (2 * hidden * layers,),
        "out_sdf.weight": (N_CLASSES, hidden),
        "out_sdf.bias": (N_CLASSES,),
        "out_tex.weight": (SCORER_CHANNELS, hidden),
        "out_tex.bias": (SCORER_CHANNELS,),
        "palette": (N_CLASSES, SCORER_CHANNELS),
    }
    for layer_index in range(layers):
        expected[f"hidden.{layer_index}.weight"] = (hidden, hidden)
        expected[f"hidden.{layer_index}.bias"] = (hidden,)
        if config.film_per_layer:
            expected[f"film_pl.{layer_index}.weight"] = (
                2 * hidden,
                modulation,
            )
            expected[f"film_pl.{layer_index}.bias"] = (2 * hidden,)
        if config.film_concat_code:
            expected[f"concat_pl.{layer_index}.weight"] = (
                hidden,
                modulation,
            )
            expected[f"concat_pl.{layer_index}.bias"] = (hidden,)
    return expected


def _validate_semantic_inputs(
    *,
    config: V9RuntimeConfigV1,
    semantic_params: dict[str, np.ndarray],
    odd_y1: np.ndarray,
) -> None:
    if type(config) is not V9RuntimeConfigV1:
        raise G111ParsedG105StageSelectorError("config must be an exact V9RuntimeConfigV1")
    if type(semantic_params) is not dict:
        raise G111ParsedG105StageSelectorError("semantic_params must be an exact dictionary")
    expected = _expected_semantic_param_shapes(config)
    if set(semantic_params) != set(expected):
        missing = sorted(set(expected) - set(semantic_params))
        extra = sorted(set(semantic_params) - set(expected))
        raise G111ParsedG105StageSelectorError(f"semantic parameter census differs: missing={missing}, extra={extra}")
    for name, shape in expected.items():
        _require_exact_float32(
            semantic_params[name],
            name=f"semantic_params[{name!r}]",
            shape=shape,
        )
    _require_exact_float32(
        odd_y1,
        name="odd_y1",
        shape=(PAIR_COUNT_N600, config.modulation_dim),
    )


def _validate_target_labels(target_labels: object) -> np.ndarray:
    expected_shape = (PAIR_COUNT_N600, SCORER_H, SCORER_W)
    if (
        type(target_labels) is not np.ndarray
        or target_labels.dtype != np.dtype(np.uint8)
        or not target_labels.flags.c_contiguous
        or target_labels.shape != expected_shape
        or np.any(target_labels >= N_CLASSES)
    ):
        raise G111ParsedG105StageSelectorError(
            "target_labels must be exact C-contiguous uint8[600,384,512] with class IDs in [0,4]"
        )
    return target_labels


def semantic_stage_action(*, d_seg: float, archive_bytes: int) -> float:
    """Return the exact same-object semantic term used for stage arbitration."""

    if type(d_seg) is not float or not math.isfinite(d_seg) or not 0.0 <= d_seg <= 1.0:
        raise G111ParsedG105StageSelectorError("d_seg must be an exact finite float in [0,1]")
    if type(archive_bytes) is not int or archive_bytes <= 0:
        raise G111ParsedG105StageSelectorError("archive_bytes must be an exact positive integer")
    return SEG_SCORE_WEIGHT * d_seg + RATE_SCORE_WEIGHT * archive_bytes / ARCHIVE_RATE_DENOMINATOR


def semantic_stage_conditional_observation(
    *,
    d_seg: float,
    archive_bytes: int,
    effective_frontier_target: float,
) -> dict[str, Any]:
    """Compute conditional math without attesting the injected inputs."""

    semantic_action = semantic_stage_action(
        d_seg=d_seg,
        archive_bytes=archive_bytes,
    )
    if (
        type(effective_frontier_target) is not float
        or not math.isfinite(effective_frontier_target)
        or effective_frontier_target <= 0.0
    ):
        raise G111ParsedG105StageSelectorError("effective_frontier_target must be an exact positive finite float")
    distortion_only = SEG_SCORE_WEIGHT * d_seg
    distortion_only_lower_bound_obstruction = distortion_only >= effective_frontier_target
    semantic_archive_exhaustion = (
        not distortion_only_lower_bound_obstruction and semantic_action >= effective_frontier_target
    )
    if distortion_only_lower_bound_obstruction:
        disposition = "SAME_OBJECT_DISTORTION_LOWER_BOUND_OBSTRUCTION"
    elif semantic_archive_exhaustion:
        disposition = "DEFER_POST_G105_POSE_REFIT"
    else:
        disposition = "POSE_REFIT_PATH_OPEN_IF_INPUT_CUSTODY_CLOSES"
    return {
        "distortion_only_value": distortion_only,
        "distortion_only_frontier_clear_conditional_observation": (distortion_only < effective_frontier_target),
        "semantic_action_value": semantic_action,
        "semantic_action_frontier_clear_conditional_observation": (semantic_action < effective_frontier_target),
        "optimistic_pose_refit_path_open_conditional_observation": (semantic_action < effective_frontier_target),
        "pose_contribution_assumed_for_gate": 0.0,
        "distortion_only_lower_bound_obstruction_conditional_observation": (distortion_only_lower_bound_obstruction),
        "semantic_archive_exhaustion_conditional_observation": (semantic_archive_exhaustion),
        "semantic_archive_exhaustion_conditional_scope": (
            "DEFER_POST_G105_POSE_REFIT" if semantic_archive_exhaustion else None
        ),
        "strict_lower_bound_condition_conditional_observation": (distortion_only_lower_bound_obstruction),
        "strict_lower_bound_condition_scope": (
            "this_parsed_G105_Y1_stage_pointer_identity_only" if distortion_only_lower_bound_obstruction else None
        ),
        "family_wide_claim": False,
        "conditional_disposition": disposition,
    }


@dataclass(frozen=True, slots=True)
class G111SemanticStageAlternativeV1:
    """One measured G105-codec/G110-outer-ZIP same-object alternative."""

    y1_wire_codec: Y1WireCodecV1
    outer_zip_method: G110OuterZipMethodV1
    d_seg: float
    disagreement_pixels: int
    semantic_packet: bytes = field(repr=False)
    product_packet: bytes = field(repr=False)
    archive: bytes = field(repr=False)
    g105_quantization_receipt_sha256: str
    scorer_y1_population_sha256: str
    camera_y1_population_sha256: str
    predicted_labels_sha256: str
    batch_progress_key_sha256: str
    batch_receipt_chain_sha256: str

    def __post_init__(self) -> None:
        if type(self.y1_wire_codec) is not Y1WireCodecV1:
            raise G111ParsedG105StageSelectorError("alternative Y1 wire codec is not typed")
        if type(self.outer_zip_method) is not G110OuterZipMethodV1:
            raise G111ParsedG105StageSelectorError("alternative outer ZIP method is not typed")
        semantic_stage_action(
            d_seg=self.d_seg,
            archive_bytes=len(self.archive),
        )
        total_pixels = PAIR_COUNT_N600 * SCORER_H * SCORER_W
        if (
            type(self.disagreement_pixels) is not int
            or not 0 <= self.disagreement_pixels <= total_pixels
            or self.d_seg != self.disagreement_pixels / total_pixels
        ):
            raise G111ParsedG105StageSelectorError("alternative d_seg does not exactly match its disagreement count")
        for name, payload in (
            ("semantic packet", self.semantic_packet),
            ("G110 product packet", self.product_packet),
            ("counted archive", self.archive),
        ):
            if type(payload) is not bytes or not payload:
                raise G111ParsedG105StageSelectorError(f"{name} must be exact nonempty bytes")
        _require_sha256(
            self.g105_quantization_receipt_sha256,
            name="G105 quantization receipt",
        )
        _require_sha256(
            self.scorer_y1_population_sha256,
            name="scorer Y1 population",
        )
        _require_sha256(
            self.camera_y1_population_sha256,
            name="camera Y1 population",
        )
        _require_sha256(
            self.predicted_labels_sha256,
            name="predicted labels",
        )
        _require_sha256(
            self.batch_progress_key_sha256,
            name="batch progress key",
        )
        _require_sha256(
            self.batch_receipt_chain_sha256,
            name="batch receipt chain",
        )

    @property
    def semantic_action(self) -> float:
        return semantic_stage_action(
            d_seg=self.d_seg,
            archive_bytes=len(self.archive),
        )

    def selection_key(self) -> tuple[float, float, int, int, int, str]:
        """Stable total order: same-object action first, then typed custody."""

        return (
            self.semantic_action,
            self.d_seg,
            len(self.archive),
            int(self.y1_wire_codec),
            int(self.outer_zip_method),
            _sha256(self.archive),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "y1_wire_codec": self.y1_wire_codec.name,
            "outer_zip_method": self.outer_zip_method.name,
            "d_seg": self.d_seg,
            "disagreement_pixels": self.disagreement_pixels,
            "semantic_action": self.semantic_action,
            "semantic_packet_bytes": len(self.semantic_packet),
            "semantic_packet_sha256": _sha256(self.semantic_packet),
            "g110_product_packet_bytes": len(self.product_packet),
            "g110_product_packet_sha256": _sha256(self.product_packet),
            "archive_bytes": len(self.archive),
            "archive_sha256": _sha256(self.archive),
            "g105_quantization_receipt_sha256": (self.g105_quantization_receipt_sha256),
            "scorer_y1_population_sha256": (self.scorer_y1_population_sha256),
            "camera_y1_population_sha256": (self.camera_y1_population_sha256),
            "predicted_labels_sha256": self.predicted_labels_sha256,
            "population_hash_schema": POPULATION_HASH_SCHEMA,
            "batch_progress_key_sha256": self.batch_progress_key_sha256,
            "batch_receipt_chain_sha256": self.batch_receipt_chain_sha256,
            "completed_batch_receipts": len(VERDICT_BATCH_SIZES),
            "resumable_from_atomic_batch_receipts": True,
            "g105_compile_encode_parse": True,
            "g105_parse_reencode_identical": True,
            "g110_rank_zero_receiver_valid": True,
            "g110_parse_back_identical": True,
            "g105_g110_double_decode_identical": True,
            "v10_uint8_realized": True,
            "all_n600_pairs_scored": True,
        }


def select_semantic_stage_alternative(
    alternatives: tuple[G111SemanticStageAlternativeV1, ...],
) -> G111SemanticStageAlternativeV1:
    """Choose the canonical minimum under the stable same-object action order."""

    if (
        type(alternatives) is not tuple
        or not alternatives
        or any(type(item) is not G111SemanticStageAlternativeV1 for item in alternatives)
    ):
        raise G111ParsedG105StageSelectorError("alternatives must be a nonempty exact tuple of typed rows")
    return min(alternatives, key=lambda item: item.selection_key())


def _pareto_ledger(
    alternatives: tuple[G111SemanticStageAlternativeV1, ...],
    *,
    pose_initializer_identity_sha256: str,
    pose_reserve: dict[str, Any],
) -> list[dict[str, Any]]:
    best_parsed_wire_d_seg = min(row.d_seg for row in alternatives)
    ledger: list[dict[str, Any]] = []
    for row in alternatives:
        wire_regret = row.d_seg - best_parsed_wire_d_seg
        dominated = any(
            other is not row
            and other.d_seg <= row.d_seg
            and len(other.archive) <= len(row.archive)
            and other.d_seg - best_parsed_wire_d_seg <= wire_regret
            and (
                other.d_seg < row.d_seg
                or len(other.archive) < len(row.archive)
                or other.d_seg - best_parsed_wire_d_seg < wire_regret
            )
            for other in alternatives
        )
        ledger.append(
            {
                "y1_wire_codec": row.y1_wire_codec.name,
                "outer_zip_method": row.outer_zip_method.name,
                "d_seg_wire": row.d_seg,
                "exact_archive_bytes": len(row.archive),
                "wire_regret": wire_regret,
                "wire_regret_scope": ("relative_to_best_parsed_G105_wire_family_same_source"),
                "pose_initializer_identity_sha256": (pose_initializer_identity_sha256),
                "pose_reserve": pose_reserve,
                "pareto_dominated": dominated,
            }
        )
    return ledger


def _batch_progress_key(identity: dict[str, Any]) -> str:
    return _sha256(_canonical_json(identity))


def _batch_row_body(
    *,
    identity: dict[str, Any],
    progress_key_sha256: str,
    batch_index: int,
    pair_start: int,
    pair_stop: int,
    target_batch_sha256: str,
    disagreement_pixels: int,
    scorer_y1_batch_sha256: str,
    camera_y1_batch_sha256: str,
    predicted_labels_batch_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": BATCH_PROGRESS_SCHEMA,
        "identity": identity,
        "progress_key_sha256": progress_key_sha256,
        "batch_index": batch_index,
        "pair_start": pair_start,
        "pair_stop": pair_stop,
        "pair_count": pair_stop - pair_start,
        "target_batch_sha256": target_batch_sha256,
        "disagreement_pixels": disagreement_pixels,
        "scorer_y1_batch_sha256": scorer_y1_batch_sha256,
        "camera_y1_batch_sha256": camera_y1_batch_sha256,
        "predicted_labels_batch_sha256": predicted_labels_batch_sha256,
    }


def _load_verified_batch_row(
    *,
    path: Path,
    identity: dict[str, Any],
    progress_key_sha256: str,
    batch_index: int,
    pair_start: int,
    pair_stop: int,
    target_batch_sha256: str,
) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise G111ParsedG105StageSelectorError(f"batch progress row is not a regular file: {path.name}")
    try:
        raw = path.read_bytes()
        row = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G111ParsedG105StageSelectorError(f"batch progress row cannot be decoded: {path.name}") from exc
    if type(row) is not dict or _canonical_json(row) != raw:
        raise G111ParsedG105StageSelectorError(f"batch progress row is not canonical: {path.name}")
    expected_keys = {
        "schema",
        "identity",
        "progress_key_sha256",
        "batch_index",
        "pair_start",
        "pair_stop",
        "pair_count",
        "target_batch_sha256",
        "disagreement_pixels",
        "scorer_y1_batch_sha256",
        "camera_y1_batch_sha256",
        "predicted_labels_batch_sha256",
        "row_body_sha256",
    }
    if set(row) != expected_keys:
        raise G111ParsedG105StageSelectorError(f"batch progress row has a noncanonical field census: {path.name}")
    row_body = {key: value for key, value in row.items() if key != "row_body_sha256"}
    if (
        row["schema"] != BATCH_PROGRESS_SCHEMA
        or type(row["identity"]) is not dict
        or _canonical_json(row["identity"]) != _canonical_json(identity)
        or row["progress_key_sha256"] != progress_key_sha256
        or type(row["batch_index"]) is not int
        or row["batch_index"] != batch_index
        or type(row["pair_start"]) is not int
        or row["pair_start"] != pair_start
        or type(row["pair_stop"]) is not int
        or row["pair_stop"] != pair_stop
        or type(row["pair_count"]) is not int
        or row["pair_count"] != pair_stop - pair_start
        or row["target_batch_sha256"] != target_batch_sha256
        or row["row_body_sha256"] != _sha256(_canonical_json(row_body))
    ):
        raise G111ParsedG105StageSelectorError(f"batch progress row identity or body hash differs: {path.name}")
    maximum_disagreements = (pair_stop - pair_start) * SCORER_H * SCORER_W
    if type(row["disagreement_pixels"]) is not int or not 0 <= row["disagreement_pixels"] <= maximum_disagreements:
        raise G111ParsedG105StageSelectorError(f"batch progress disagreement count is invalid: {path.name}")
    for field_name in (
        "row_body_sha256",
        "target_batch_sha256",
        "scorer_y1_batch_sha256",
        "camera_y1_batch_sha256",
        "predicted_labels_batch_sha256",
    ):
        _require_sha256(row[field_name], name=f"batch progress {field_name}")
    return row


def _batch_digest_chain(
    rows: list[dict[str, Any]],
    *,
    digest_field: str,
) -> str:
    return _sha256(
        _canonical_json(
            [
                {
                    "batch_index": row["batch_index"],
                    digest_field: row[digest_field],
                }
                for row in rows
            ]
        )
    )


def _score_one_parsed_surface(
    *,
    program: object,
    product_packet: bytes,
    target_labels: np.ndarray,
    scorer: SegArgmaxBatchScorerV1,
    progress_identity: dict[str, Any],
    progress_dir: Path,
) -> tuple[int, str, str, str, str, str]:
    disagreement_pixels = 0
    observed_batch_sizes: list[int] = []
    progress_key_sha256 = _batch_progress_key(progress_identity)
    progress_namespace = _durable_out_dir(progress_dir / progress_key_sha256)
    completed_rows: list[dict[str, Any]] = []

    pair_start = 0
    for batch_index, expected_batch_size in enumerate(VERDICT_BATCH_SIZES):
        pair_stop = pair_start + expected_batch_size
        target_batch = target_labels[pair_start:pair_stop]
        target_batch_sha256 = _sha256(memoryview(target_batch))
        row_path = progress_namespace / (f"batch_{batch_index:03d}_{pair_start:03d}_{pair_stop:03d}.json")
        completed_row = _load_verified_batch_row(
            path=row_path,
            identity=progress_identity,
            progress_key_sha256=progress_key_sha256,
            batch_index=batch_index,
            pair_start=pair_start,
            pair_stop=pair_stop,
            target_batch_sha256=target_batch_sha256,
        )
        if completed_row is not None:
            disagreement_pixels += completed_row["disagreement_pixels"]
            completed_rows.append(completed_row)
            observed_batch_sizes.append(expected_batch_size)
            pair_start = pair_stop
            continue

        scorer_digest = hashlib.sha256()
        camera_digest = hashlib.sha256()
        camera_batch: list[np.ndarray] = []
        for pair_id in range(pair_start, pair_stop):
            scorer_y1 = render_scorer_y1(program, pair_id)  # type: ignore[arg-type]
            if (
                type(scorer_y1) is not np.ndarray
                or scorer_y1.dtype != np.dtype(np.uint8)
                or scorer_y1.shape != (SCORER_H, SCORER_W, SCORER_CHANNELS)
                or not scorer_y1.flags.c_contiguous
            ):
                raise G111ParsedG105StageSelectorError("parsed G105 receiver emitted a malformed scorer Y1")
            rank_zero_pair = render_g110_rank_zero_scorer_pair(
                product_packet,
                pair_id,
            )
            if (
                type(rank_zero_pair) is not np.ndarray
                or rank_zero_pair.dtype != np.dtype(np.uint8)
                or rank_zero_pair.shape != (2, SCORER_H, SCORER_W, SCORER_CHANNELS)
                or not rank_zero_pair.flags.c_contiguous
                or not np.array_equal(rank_zero_pair[0], rank_zero_pair[1])
                or not np.array_equal(rank_zero_pair[1], scorer_y1)
            ):
                raise G111ParsedG105StageSelectorError("G105/G110 rank-zero double decode is not identical")
            camera_y1 = realize_factor2_uint8_numpy(scorer_y1)
            if (
                type(camera_y1) is not np.ndarray
                or camera_y1.dtype != np.dtype(np.uint8)
                or camera_y1.shape != (CAMERA_HW[0], CAMERA_HW[1], SCORER_CHANNELS)
                or not camera_y1.flags.c_contiguous
            ):
                raise G111ParsedG105StageSelectorError("V10 realization emitted a malformed camera Y1")
            scorer_digest.update(scorer_y1)
            camera_digest.update(camera_y1)
            camera_batch.append(camera_y1)

        exact_batch = np.ascontiguousarray(np.stack(camera_batch), dtype=np.uint8)
        predicted = scorer(exact_batch)
        expected_shape = (expected_batch_size, SCORER_H, SCORER_W)
        if (
            type(predicted) is not np.ndarray
            or predicted.dtype != np.dtype(np.uint8)
            or predicted.shape != expected_shape
            or not predicted.flags.c_contiguous
            or np.any(predicted >= N_CLASSES)
        ):
            raise G111ParsedG105StageSelectorError(
                f"Seg scorer callback must return exact C-contiguous uint8{expected_shape} argmax labels in [0,4]"
            )
        batch_disagreement_pixels = int(np.count_nonzero(predicted != target_batch))
        disagreement_pixels += batch_disagreement_pixels
        row_body = _batch_row_body(
            identity=progress_identity,
            progress_key_sha256=progress_key_sha256,
            batch_index=batch_index,
            pair_start=pair_start,
            pair_stop=pair_stop,
            target_batch_sha256=target_batch_sha256,
            disagreement_pixels=batch_disagreement_pixels,
            scorer_y1_batch_sha256=scorer_digest.hexdigest(),
            camera_y1_batch_sha256=camera_digest.hexdigest(),
            predicted_labels_batch_sha256=_sha256(memoryview(predicted)),
        )
        completed_row = {
            **row_body,
            "row_body_sha256": _sha256(_canonical_json(row_body)),
        }
        _atomic_write_idempotent(
            row_path,
            _canonical_json(completed_row),
        )
        completed_rows.append(completed_row)
        observed_batch_sizes.append(expected_batch_size)
        pair_start = pair_stop

    if (
        pair_start != PAIR_COUNT_N600
        or tuple(observed_batch_sizes) != VERDICT_BATCH_SIZES
        or len(completed_rows) != len(VERDICT_BATCH_SIZES)
    ):
        raise AssertionError("internal n600 batch geometry drifted")
    return (
        disagreement_pixels,
        _batch_digest_chain(
            completed_rows,
            digest_field="scorer_y1_batch_sha256",
        ),
        _batch_digest_chain(
            completed_rows,
            digest_field="camera_y1_batch_sha256",
        ),
        _batch_digest_chain(
            completed_rows,
            digest_field="predicted_labels_batch_sha256",
        ),
        progress_key_sha256,
        _batch_digest_chain(
            completed_rows,
            digest_field="row_body_sha256",
        ),
    )


def _durable_out_dir(out_dir: Path) -> Path:
    if not isinstance(out_dir, Path) or not out_dir.is_absolute():
        raise G111ParsedG105StageSelectorError("out_dir must be an absolute pathlib.Path")
    resolved = out_dir.resolve(strict=False)
    if any(resolved == root or root in resolved.parents for root in _EPHEMERAL_ROOTS):
        raise G111ParsedG105StageSelectorError("out_dir must not be under /tmp or another temporary root")
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.is_symlink() or not out_dir.is_dir():
        raise G111ParsedG105StageSelectorError("out_dir must be a real durable directory, not a symlink")
    return out_dir


def _atomic_write_idempotent(path: Path, payload: bytes) -> None:
    if type(payload) is not bytes or not payload:
        raise G111ParsedG105StageSelectorError("atomic artifact payload must be exact nonempty bytes")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise G111ParsedG105StageSelectorError(f"refusing non-regular output path: {path.name}")
        if path.read_bytes() == payload:
            return
        raise G111ParsedG105StageSelectorError(f"refusing to overwrite different stage artifact: {path.name}")

    temp_path: Path | None = None
    try:
        for suffix in range(100):
            candidate = path.with_name(f".{path.name}.tmp.{os.getpid()}.{suffix}")
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                continue
            temp_path = candidate
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temp_path, 0o644)
            try:
                os.link(temp_path, path)
            except FileExistsError as exc:
                if path.is_symlink() or path.read_bytes() != payload:
                    raise G111ParsedG105StageSelectorError(f"concurrent stage artifact differs: {path.name}") from exc
            os.unlink(temp_path)
            temp_path = None
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return
        raise G111ParsedG105StageSelectorError(f"could not reserve atomic temp path for {path.name}")
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


@dataclass(frozen=True, slots=True)
class G111CrossStageParetoRowV1:
    """Typed handoff row for the trainer-owned cross-stage Pareto ledger."""

    stage_tag: str
    source_checkpoint_identity_sha256: str
    semantic_source_state_sha256: str
    config_sha256: str
    target_labels_sha256: str
    seg_scorer_identity_sha256: str
    pointer_snapshot_identity_sha256: str
    pose_initializer_identity_sha256: str
    pose_reserve: dict[str, Any]
    effective_frontier_target: float
    conditional_observation: dict[str, Any]
    alternatives: tuple[G111SemanticStageAlternativeV1, ...]
    selected: G111SemanticStageAlternativeV1
    packet_filename: str
    archive_filename: str

    def __post_init__(self) -> None:
        if type(self.stage_tag) is not str or _STAGE_TAG.fullmatch(self.stage_tag) is None:
            raise G111ParsedG105StageSelectorError("cross-stage row has an invalid stage tag")
        for name, value in (
            (
                "source checkpoint identity",
                self.source_checkpoint_identity_sha256,
            ),
            ("semantic source state", self.semantic_source_state_sha256),
            ("config", self.config_sha256),
            ("target labels", self.target_labels_sha256),
            ("Seg scorer identity", self.seg_scorer_identity_sha256),
            (
                "pointer snapshot identity",
                self.pointer_snapshot_identity_sha256,
            ),
            (
                "pose initializer identity",
                self.pose_initializer_identity_sha256,
            ),
        ):
            _require_sha256(value, name=name)
        if (
            type(self.pose_reserve) is not dict
            or type(self.effective_frontier_target) is not float
            or not math.isfinite(self.effective_frontier_target)
            or self.effective_frontier_target <= 0.0
            or type(self.conditional_observation) is not dict
            or type(self.alternatives) is not tuple
            or len(self.alternatives) != 4
            or any(type(row) is not G111SemanticStageAlternativeV1 for row in self.alternatives)
            or type(self.selected) is not G111SemanticStageAlternativeV1
            or not any(self.selected is row for row in self.alternatives)
        ):
            raise G111ParsedG105StageSelectorError("cross-stage row has malformed typed alternatives")
        for name, value in (
            ("packet filename", self.packet_filename),
            ("archive filename", self.archive_filename),
        ):
            if type(value) is not str or not value or Path(value).name != value:
                raise G111ParsedG105StageSelectorError(f"cross-stage {name} is unsafe")

    def _identity_payload(self) -> dict[str, Any]:
        ordered_alternatives = sorted(
            self.alternatives,
            key=lambda row: row.selection_key(),
        )
        return {
            "schema": CROSS_STAGE_PARETO_ROW_SCHEMA,
            "stage_tag": self.stage_tag,
            "lineage_identity": {
                "source_checkpoint_identity_sha256": (self.source_checkpoint_identity_sha256),
                "semantic_source_state_sha256": (self.semantic_source_state_sha256),
                "config_sha256": self.config_sha256,
                "target_labels_sha256": self.target_labels_sha256,
                "seg_scorer_identity_sha256": (self.seg_scorer_identity_sha256),
                "pointer_snapshot_identity_sha256": (self.pointer_snapshot_identity_sha256),
                "pose_initializer_identity_sha256": (self.pose_initializer_identity_sha256),
            },
            "pose_reserve": self.pose_reserve,
            "dynamic_effective_frontier_target": (self.effective_frontier_target),
            "conditional_observation": self.conditional_observation,
            "frontier_snapshot_scope": {
                "conditional_path_is_snapshot_scoped": True,
                "production_current_status_confirmed_after_screen": False,
                "post_screen_dynamic_frontier_reverify_required": True,
            },
            "stage_pareto_rows": [row.to_dict() for row in ordered_alternatives],
            "selected_deploy_identity": {
                "y1_wire_codec": self.selected.y1_wire_codec.name,
                "outer_zip_method": self.selected.outer_zip_method.name,
                "packet_filename": self.packet_filename,
                "packet_bytes": len(self.selected.product_packet),
                "packet_sha256": _sha256(self.selected.product_packet),
                "archive_filename": self.archive_filename,
                "archive_bytes": len(self.selected.archive),
                "archive_sha256": _sha256(self.selected.archive),
            },
            "paired_resume_identity": {
                "batch_progress_key_sha256": (self.selected.batch_progress_key_sha256),
                "batch_receipt_chain_sha256": (self.selected.batch_receipt_chain_sha256),
                "completed_batch_receipts": len(VERDICT_BATCH_SIZES),
            },
            "cross_stage_retention_policy": ("retain_nondominated_and_semantically_second_best_stage_rows"),
            "cross_stage_file_owned_by_trainer": True,
            "selector_writes_cross_stage_file": False,
            "production_admission": False,
            "production_wrapper_required": True,
        }

    @property
    def row_identity_sha256(self) -> str:
        return _sha256(_canonical_json(self._identity_payload()))

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "row_identity_sha256": self.row_identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class G111ParsedG105StageSelectionV1:
    """Durable selected semantic-only stage artifacts and their receipt."""

    packet_path: Path
    archive_path: Path
    receipt_path: Path
    selected: G111SemanticStageAlternativeV1
    alternatives: tuple[G111SemanticStageAlternativeV1, ...]
    cross_stage_pareto_row: G111CrossStageParetoRowV1
    receipt: dict[str, Any]


def compile_select_parsed_g105_stage_v1(
    *,
    config: V9RuntimeConfigV1,
    semantic_params: dict[str, np.ndarray],
    odd_y1: np.ndarray,
    target_labels: np.ndarray,
    seg_argmax_batch_scorer: SegArgmaxBatchScorerV1,
    injected_inputs_are_test_only: bool,
    seg_scorer_identity_sha256: str,
    source_checkpoint_identity_sha256: str,
    pose_initializer_identity_sha256: str,
    measured_pose_reserve_bytes: int | None = None,
    measured_pose_reserve_receipt_sha256: str | None = None,
    effective_frontier_target: float,
    pointer_snapshot_identity_sha256: str,
    out_dir: Path,
    progress_dir: Path,
    stage_tag: str,
) -> G111ParsedG105StageSelectionV1:
    """Measure and persist one pointer-snapshot-scoped semantic stage floor.

    The pointer identity is part of both the progress key and artifact
    namespace.  The returned conditional observation is historical to that
    snapshot until the production wrapper re-verifies the dynamic frontier
    after this full-n600 screen.
    """

    _validate_semantic_inputs(
        config=config,
        semantic_params=semantic_params,
        odd_y1=odd_y1,
    )
    exact_target_labels = _validate_target_labels(target_labels)
    if injected_inputs_are_test_only is not True:
        raise G111ParsedG105StageSelectorError(
            "injected input engine is test-only; production requires the physical authority wrapper"
        )
    if not callable(seg_argmax_batch_scorer):
        raise G111ParsedG105StageSelectorError("seg_argmax_batch_scorer must be callable")
    _require_sha256(
        seg_scorer_identity_sha256,
        name="Seg scorer identity",
    )
    _require_sha256(
        source_checkpoint_identity_sha256,
        name="source checkpoint identity",
    )
    _require_sha256(
        pose_initializer_identity_sha256,
        name="pose initializer identity",
    )
    _require_sha256(
        pointer_snapshot_identity_sha256,
        name="pointer snapshot identity",
    )
    pose_reserve = _pose_reserve_record(
        measured_pose_reserve_bytes=measured_pose_reserve_bytes,
        measured_pose_reserve_receipt_sha256=(measured_pose_reserve_receipt_sha256),
    )
    if (
        type(effective_frontier_target) is not float
        or not math.isfinite(effective_frontier_target)
        or effective_frontier_target <= 0.0
    ):
        raise G111ParsedG105StageSelectorError("effective_frontier_target must be an exact positive finite float")
    artifact_namespace = _artifact_namespace(
        stage_tag=stage_tag,
        pointer_snapshot_identity_sha256=(pointer_snapshot_identity_sha256),
    )
    durable_out_dir = _durable_out_dir(out_dir)
    durable_progress_dir = _durable_out_dir(progress_dir)
    target_labels_sha256 = _sha256(memoryview(exact_target_labels))

    alternatives: list[G111SemanticStageAlternativeV1] = []
    semantic_programs: list[object] = []
    semantic_source_state_sha256: str | None = None
    for y1_wire_codec in _Y1_WIRE_CODECS:
        surface = compile_g105_public_wire_quantization_surface_numpy(
            config=config,
            params=semantic_params,
            y1_code=odd_y1,
            y1_wire_codec=y1_wire_codec,
        )
        if semantic_source_state_sha256 is None:
            semantic_source_state_sha256 = surface.receipt.source_state_sha256
        elif surface.receipt.source_state_sha256 != semantic_source_state_sha256:
            raise G111ParsedG105StageSelectorError("G105 wire families did not consume the same source state")
        for prior_program in semantic_programs:
            if (
                prior_program.config != surface.program.config  # type: ignore[attr-defined]
                or set(prior_program.params)  # type: ignore[attr-defined]
                != set(surface.program.params)
                or any(
                    not np.array_equal(
                        prior_program.params[name],  # type: ignore[attr-defined]
                        surface.program.params[name],
                    )
                    for name in surface.program.params
                )
                or not np.array_equal(
                    prior_program.y1_code,  # type: ignore[attr-defined]
                    surface.program.y1_code,
                )
            ):
                raise G111ParsedG105StageSelectorError("G105 RAW/Rice parsed semantic states differ")
        semantic_programs.append(surface.program)

        product_packet = build_g110_rank_zero_semantic_floor_packet(surface.packet)
        parsed_product = parse_g110_two_layer_v1(product_packet)
        if parsed_product.packet != product_packet:
            raise AssertionError("G110 rank-zero product parse-back drifted")
        if parsed_product.semantic_packet != surface.packet:
            raise G111ParsedG105StageSelectorError("G110 rank-zero product changed the selected G105 packet")
        archive_variants: dict[G110OuterZipMethodV1, bytes] = {}
        for outer_zip_method in _OUTER_ZIP_METHODS:
            archive = build_g110_counted_archive_variant(
                product_packet,
                outer_zip_method,
            )
            if (
                parse_g110_counted_archive_variant(
                    archive,
                    outer_zip_method,
                )
                != product_packet
            ):
                raise AssertionError("G110 archive parse-back changed packet")
            archive_variants[outer_zip_method] = archive
        progress_identity = {
            "schema": BATCH_PROGRESS_SCHEMA,
            "execution_surface": EXECUTION_SURFACE,
            "stage_tag": stage_tag,
            "y1_wire_codec": y1_wire_codec.name,
            "semantic_source_state_sha256": (surface.receipt.source_state_sha256),
            "source_checkpoint_identity_sha256": (source_checkpoint_identity_sha256),
            "semantic_packet_sha256": _sha256(surface.packet),
            "g110_product_packet_sha256": _sha256(product_packet),
            "archive_variants": [
                {
                    "outer_zip_method": method.name,
                    "archive_bytes": len(archive_variants[method]),
                    "archive_sha256": _sha256(archive_variants[method]),
                }
                for method in _OUTER_ZIP_METHODS
            ],
            "target_labels_sha256": target_labels_sha256,
            "seg_scorer_identity_sha256": seg_scorer_identity_sha256,
            "pointer_snapshot_identity_sha256": (pointer_snapshot_identity_sha256),
            "pose_initializer_identity_sha256": (pose_initializer_identity_sha256),
            "pair_count": PAIR_COUNT_N600,
            "batch_sizes": list(VERDICT_BATCH_SIZES),
        }
        (
            disagreement_pixels,
            scorer_population_sha256,
            camera_population_sha256,
            predicted_labels_sha256,
            batch_progress_key_sha256,
            batch_receipt_chain_sha256,
        ) = _score_one_parsed_surface(
            program=surface.program,
            product_packet=product_packet,
            target_labels=exact_target_labels,
            scorer=seg_argmax_batch_scorer,
            progress_identity=progress_identity,
            progress_dir=durable_progress_dir,
        )
        d_seg = disagreement_pixels / (PAIR_COUNT_N600 * SCORER_H * SCORER_W)
        for outer_zip_method in _OUTER_ZIP_METHODS:
            archive = archive_variants[outer_zip_method]
            alternatives.append(
                G111SemanticStageAlternativeV1(
                    y1_wire_codec=y1_wire_codec,
                    outer_zip_method=outer_zip_method,
                    d_seg=d_seg,
                    disagreement_pixels=disagreement_pixels,
                    semantic_packet=surface.packet,
                    product_packet=product_packet,
                    archive=archive,
                    g105_quantization_receipt_sha256=(surface.receipt.receipt_sha256),
                    scorer_y1_population_sha256=scorer_population_sha256,
                    camera_y1_population_sha256=camera_population_sha256,
                    predicted_labels_sha256=predicted_labels_sha256,
                    batch_progress_key_sha256=(batch_progress_key_sha256),
                    batch_receipt_chain_sha256=(batch_receipt_chain_sha256),
                )
            )

    exact_alternatives = tuple(alternatives)
    if len(exact_alternatives) != 4:
        raise AssertionError("internal G105/G110 archive matrix is incomplete")
    raw_rows = tuple(row for row in exact_alternatives if row.y1_wire_codec is Y1WireCodecV1.RAW_I16_LE)
    rice_rows = tuple(row for row in exact_alternatives if row.y1_wire_codec is Y1WireCodecV1.DELTA_RICE_BEST_K)
    if (
        len(raw_rows) != 2
        or len(rice_rows) != 2
        or raw_rows[0].d_seg != rice_rows[0].d_seg
        or raw_rows[0].disagreement_pixels != rice_rows[0].disagreement_pixels
        or raw_rows[0].scorer_y1_population_sha256 != rice_rows[0].scorer_y1_population_sha256
        or raw_rows[0].camera_y1_population_sha256 != rice_rows[0].camera_y1_population_sha256
        or raw_rows[0].predicted_labels_sha256 != rice_rows[0].predicted_labels_sha256
    ):
        raise G111ParsedG105StageSelectorError("same parsed G105 state changed across RAW/Rice full-n600 scoring")

    selected = select_semantic_stage_alternative(exact_alternatives)
    if (
        build_g110_public_archive(selected.product_packet) != selected.archive
        or parse_g110_public_archive(selected.archive) != selected.product_packet
    ):
        raise G111ParsedG105StageSelectorError("selected archive is not G110's canonical receiver archive")
    conditional_observation = semantic_stage_conditional_observation(
        d_seg=selected.d_seg,
        archive_bytes=len(selected.archive),
        effective_frontier_target=effective_frontier_target,
    )
    packet_name = f"{artifact_namespace}.g110_rank_zero_packet.bin"
    archive_name = f"{artifact_namespace}.semantic_only.archive.zip"
    receipt_name = f"{artifact_namespace}.parsed_g105_stage_selector.receipt.json"
    packet_path = durable_out_dir / packet_name
    archive_path = durable_out_dir / archive_name
    receipt_path = durable_out_dir / receipt_name
    config_sha256 = _sha256(_canonical_json(config.to_dict()))
    if semantic_source_state_sha256 is None:
        raise AssertionError("G105 source state identity was not established")
    cross_stage_pareto_row = G111CrossStageParetoRowV1(
        stage_tag=stage_tag,
        source_checkpoint_identity_sha256=(source_checkpoint_identity_sha256),
        semantic_source_state_sha256=semantic_source_state_sha256,
        config_sha256=config_sha256,
        target_labels_sha256=target_labels_sha256,
        seg_scorer_identity_sha256=seg_scorer_identity_sha256,
        pointer_snapshot_identity_sha256=(pointer_snapshot_identity_sha256),
        pose_initializer_identity_sha256=(pose_initializer_identity_sha256),
        pose_reserve=pose_reserve,
        effective_frontier_target=effective_frontier_target,
        conditional_observation=conditional_observation,
        alternatives=exact_alternatives,
        selected=selected,
        packet_filename=packet_name,
        archive_filename=archive_name,
    )

    receipt_body: dict[str, Any] = {
        "schema": SCHEMA,
        "execution_surface": EXECUTION_SURFACE,
        "engine_only": True,
        "injected_inputs_are_test_only": True,
        "production_authority_closed": False,
        "production_wrapper_required": True,
        "production_verdict_emitted": False,
        "production_admission": False,
        "stage_tag": stage_tag,
        "artifact_namespace": artifact_namespace,
        "full_n600": True,
        "pair_count": PAIR_COUNT_N600,
        "config_sha256": config_sha256,
        "source_checkpoint_identity_sha256": (source_checkpoint_identity_sha256),
        "semantic_source_state_sha256": semantic_source_state_sha256,
        "odd_y1_shape": list(odd_y1.shape),
        "target_labels_sha256": target_labels_sha256,
        "seg_scorer_identity_sha256": seg_scorer_identity_sha256,
        "pose_initializer_identity_sha256": (pose_initializer_identity_sha256),
        "pose_reserve": pose_reserve,
        "pointer_snapshot_identity_sha256": (pointer_snapshot_identity_sha256),
        "dynamic_effective_frontier_target": effective_frontier_target,
        "frontier_snapshot_scope": {
            "conditional_path_is_snapshot_scoped": True,
            "production_current_status_confirmed_after_screen": False,
            "post_screen_dynamic_frontier_reverify_required": True,
            "new_pointer_identity_requires_new_artifact_namespace": True,
        },
        "batch_geometry": {
            "maximum_batch_size": VERDICT_BATCH_SIZE,
            "batch_count_per_wire_family": len(VERDICT_BATCH_SIZES),
            "batch_sizes": list(VERDICT_BATCH_SIZES),
            "wire_family_count": len(_Y1_WIRE_CODECS),
            "total_scorer_callback_calls": (len(VERDICT_BATCH_SIZES) * len(_Y1_WIRE_CODECS)),
        },
        "batch_progress": {
            "schema": BATCH_PROGRESS_SCHEMA,
            "store_supplied_by_caller": True,
            "atomic_rows": True,
            "rows_preserved": True,
            "rows_per_wire_family": len(VERDICT_BATCH_SIZES),
            "wire_family_count": len(_Y1_WIRE_CODECS),
            "completed_rows": (len(VERDICT_BATCH_SIZES) * len(_Y1_WIRE_CODECS)),
            "reuse_requires_verified_identity_and_body_hash": True,
            "identity_binds_semantic_packet_and_archive_variants": True,
            "identity_binds_source_labels_pointer_and_scorer": True,
        },
        "semantic_formula": ("100*d_seg+25*archive_bytes/37_545_489"),
        "alternatives": [
            row.to_dict()
            for row in sorted(
                exact_alternatives,
                key=lambda item: (
                    int(item.y1_wire_codec),
                    int(item.outer_zip_method),
                ),
            )
        ],
        "pareto_ledger": _pareto_ledger(
            exact_alternatives,
            pose_initializer_identity_sha256=(pose_initializer_identity_sha256),
            pose_reserve=pose_reserve,
        ),
        "source_float_to_parsed_wire_regret": {
            "measured": False,
            "value": None,
            "reason": ("engine surface begins at parsed G105; no source-float Seg scorer row was measured"),
        },
        "selected": selected.to_dict(),
        "cross_stage_pareto_row": cross_stage_pareto_row.to_dict(),
        "selected_artifacts": {
            "packet_filename": packet_name,
            "packet_bytes": len(selected.product_packet),
            "packet_sha256": _sha256(selected.product_packet),
            "archive_filename": archive_name,
            "archive_bytes": len(selected.archive),
            "archive_sha256": _sha256(selected.archive),
            "receipt_filename": receipt_name,
        },
        "conditional_observation": conditional_observation,
        "conditional_observation_scope": ("untrusted_injected_inputs_same_object_semantic_only"),
        "production_lower_bound_verdict_emitted": False,
        "family_wide_claim": False,
        "pose_measured": False,
        "incomplete_semantic_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
        "scorer_weights_emitted": False,
        "ground_truth_emitted": False,
        "storage_class": "small_durable_stage_artifacts",
        "temporary_artifacts_retained": False,
    }
    receipt_body_sha256 = _sha256(_canonical_json(receipt_body))
    receipt = {
        **receipt_body,
        "receipt_body_sha256": receipt_body_sha256,
    }
    receipt_bytes = _canonical_json(receipt)

    _atomic_write_idempotent(packet_path, selected.product_packet)
    _atomic_write_idempotent(archive_path, selected.archive)
    _atomic_write_idempotent(receipt_path, receipt_bytes)
    return G111ParsedG105StageSelectionV1(
        packet_path=packet_path,
        archive_path=archive_path,
        receipt_path=receipt_path,
        selected=selected,
        alternatives=exact_alternatives,
        cross_stage_pareto_row=cross_stage_pareto_row,
        receipt=receipt,
    )


__all__ = [
    "ARCHIVE_RATE_DENOMINATOR",
    "BATCH_PROGRESS_SCHEMA",
    "CROSS_STAGE_PARETO_ROW_SCHEMA",
    "EXECUTION_SURFACE",
    "PAIR_COUNT_N600",
    "POPULATION_HASH_SCHEMA",
    "SCHEMA",
    "VERDICT_BATCH_SIZE",
    "VERDICT_BATCH_SIZES",
    "G111CrossStageParetoRowV1",
    "G111ParsedG105StageSelectionV1",
    "G111ParsedG105StageSelectorError",
    "G111SemanticStageAlternativeV1",
    "compile_select_parsed_g105_stage_v1",
    "select_semantic_stage_alternative",
    "semantic_stage_action",
    "semantic_stage_conditional_observation",
]
