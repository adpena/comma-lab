# SPDX-License-Identifier: MIT
"""Exact camera-lattice overlay that preserves predictor signal outside G.

The predictor already paid for a useful uint8 camera frame.  A task-space
correction must not discard that signal by repainting the whole image from a
five-colour palette.  For the frozen camera-to-scorer geometry, scorer supports
are disjoint.  We can therefore replace only the camera taps owned by semantic
cells whose label actually changed, while proving that every other camera byte
and scorer numerator remains identical to the predictor output.

This is a bounded structural primitive.  It does not invoke a scorer and makes
no score, candidate, originality, or promotion claim.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

import numpy as np

from tac.optimization.direct_description_carrier_compose import (
    REALIZATION_PAINT_ORDER,
    ROLE_CLASS_IDS,
    ReceiverRealizationProfileV1,
)
from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    Uint8LatticeError,
)
from tac.witness_dsl.generative_taskspace_correction import DecodedGenerativeCorrectionV1

CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
SCORER_HEIGHT: Final = 384
SCORER_WIDTH: Final = 512
CHANNELS: Final = 3
MAX_BOUNDED_PAIRS: Final = 4
RECEIPT_SCHEMA: Final = "tac.predictor_preserving_taskspace_overlay_receipt.v1"


class PredictorPreservingOverlayError(ValueError):
    """The base frame, ownership mask, realization, or exact proof failed."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PredictorPreservingOverlayError("overlay receipt is not finite canonical ASCII JSON") from exc


def _require_sha256(value: object, *, field: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PredictorPreservingOverlayError(f"{field} must be canonical lowercase SHA-256")


def _immutable_u8(value: np.ndarray) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=np.uint8).copy()
    copied.setflags(write=False)
    return copied


def _operator() -> DisjointResizeOperator:
    try:
        return DisjointResizeOperator.build(
            camera_h=CAMERA_HEIGHT,
            camera_w=CAMERA_WIDTH,
            scorer_h=SCORER_HEIGHT,
            scorer_w=SCORER_WIDTH,
        )
    except Uint8LatticeError as exc:
        raise PredictorPreservingOverlayError("frozen camera/scorer geometry is not disjoint") from exc


def _profile_sha256(profile: ReceiverRealizationProfileV1) -> str:
    return _sha256(
        _canonical_json(
            {
                "amplitude_u8": profile.amplitude_u8,
                "coverage_radius": profile.coverage_radius,
                "role_rgb_u8": [list(row) for row in profile.role_rgb_u8],
            }
        )
    )


def _paint_corrected_labels(
    labels: np.ndarray,
    profile: ReceiverRealizationProfileV1,
) -> np.ndarray:
    target = np.empty((*labels.shape, CHANNELS), dtype=np.uint8)
    for role in REALIZATION_PAINT_ORDER:
        target[labels == ROLE_CLASS_IDS[role]] = profile.colour_for(role)
    return np.ascontiguousarray(target)


@dataclass(frozen=True, slots=True)
class PredictorPreservingOverlayReceiptV1:
    schema: Literal["tac.predictor_preserving_taskspace_overlay_receipt.v1"]
    pair_count: int
    correction_packet_sha256: str
    realization_profile_sha256: str
    predictor_labels_sha256: str
    corrected_labels_sha256: str
    changed_label_mask_sha256: str
    base_camera_y1_sha256: str
    corrected_camera_y1_sha256: str
    base_scorer_numerators_sha256: str
    corrected_scorer_numerators_sha256: str
    changed_scorer_cells: int
    changed_scorer_values: int
    owned_camera_values: int
    actually_changed_camera_values: int
    scorer_denominator: int
    camera_values_total: int
    scorer_values_total: int
    exact_changed_target_numerators: Literal[True] = True
    unchanged_scorer_numerators_preserved: Literal[True] = True
    outside_owned_camera_values_preserved: Literal[True] = True
    predictor_rgb_preserved_outside_g: Literal[True] = True
    full_frame_palette_repaint: Literal[False] = False
    dense_predictor_rgb_serialized: Literal[False] = False
    scorer_invoked: Literal[False] = False
    score_claim: Literal[False] = False
    candidate_claim: Literal[False] = False
    originality_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise PredictorPreservingOverlayError("overlay receipt schema changed")
        if type(self.pair_count) is not int or not 1 <= self.pair_count <= MAX_BOUNDED_PAIRS:
            raise PredictorPreservingOverlayError("overlay receipt pair_count is outside bounded scope")
        for field in (
            "correction_packet_sha256",
            "realization_profile_sha256",
            "predictor_labels_sha256",
            "corrected_labels_sha256",
            "changed_label_mask_sha256",
            "base_camera_y1_sha256",
            "corrected_camera_y1_sha256",
            "base_scorer_numerators_sha256",
            "corrected_scorer_numerators_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        exact_ints = (
            "changed_scorer_cells",
            "changed_scorer_values",
            "owned_camera_values",
            "actually_changed_camera_values",
            "scorer_denominator",
            "camera_values_total",
            "scorer_values_total",
        )
        if any(type(getattr(self, field)) is not int for field in exact_ints):
            raise PredictorPreservingOverlayError("overlay receipt counts must be exact integers")
        if self.changed_scorer_cells < 1:
            raise PredictorPreservingOverlayError("overlay receipt must own at least one changed semantic cell")
        if self.changed_scorer_values != self.changed_scorer_cells * CHANNELS:
            raise PredictorPreservingOverlayError("overlay changed scorer value accounting differs from RGB cells")
        if not 1 <= self.actually_changed_camera_values <= self.owned_camera_values:
            raise PredictorPreservingOverlayError("overlay must cause at least one owned camera-byte change")
        if self.camera_values_total != self.pair_count * CAMERA_HEIGHT * CAMERA_WIDTH * CHANNELS:
            raise PredictorPreservingOverlayError("overlay camera cardinality changed")
        if self.scorer_values_total != self.pair_count * SCORER_HEIGHT * SCORER_WIDTH * CHANNELS:
            raise PredictorPreservingOverlayError("overlay scorer cardinality changed")
        truth = {
            "exact_changed_target_numerators": True,
            "unchanged_scorer_numerators_preserved": True,
            "outside_owned_camera_values_preserved": True,
            "predictor_rgb_preserved_outside_g": True,
            "full_frame_palette_repaint": False,
            "dense_predictor_rgb_serialized": False,
            "scorer_invoked": False,
            "score_claim": False,
            "candidate_claim": False,
            "originality_claim": False,
            "promotion_eligible": False,
            "research_only": True,
        }
        if any(getattr(self, field) is not expected for field, expected in truth.items()):
            raise PredictorPreservingOverlayError("overlay receipt truth labels became permissive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


@dataclass(frozen=True, slots=True)
class PredictorPreservingOverlayResultV1:
    camera_y1: np.ndarray
    changed_label_mask: np.ndarray
    owned_camera_mask: np.ndarray
    receipt: PredictorPreservingOverlayReceiptV1

    def __post_init__(self) -> None:
        camera = np.asarray(self.camera_y1)
        changed = np.asarray(self.changed_label_mask)
        owned = np.asarray(self.owned_camera_mask)
        if camera.dtype != np.uint8 or camera.shape != (
            self.receipt.pair_count,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            CHANNELS,
        ):
            raise PredictorPreservingOverlayError("overlay result camera ABI changed")
        if changed.dtype != np.bool_ or changed.shape != (
            self.receipt.pair_count,
            SCORER_HEIGHT,
            SCORER_WIDTH,
        ):
            raise PredictorPreservingOverlayError("overlay result scorer ownership ABI changed")
        if owned.dtype != np.bool_ or owned.shape != (
            self.receipt.pair_count,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
        ):
            raise PredictorPreservingOverlayError("overlay result camera ownership ABI changed")
        if _sha256(memoryview(camera).cast("B")) != self.receipt.corrected_camera_y1_sha256:
            raise PredictorPreservingOverlayError("overlay result camera differs from receipt")
        if _sha256(memoryview(changed).cast("B")) != self.receipt.changed_label_mask_sha256:
            raise PredictorPreservingOverlayError("overlay result changed mask differs from receipt")
        object.__setattr__(self, "camera_y1", _immutable_u8(camera))
        changed_copy = np.ascontiguousarray(changed, dtype=np.bool_).copy()
        changed_copy.setflags(write=False)
        object.__setattr__(self, "changed_label_mask", changed_copy)
        owned_copy = np.ascontiguousarray(owned, dtype=np.bool_).copy()
        owned_copy.setflags(write=False)
        object.__setattr__(self, "owned_camera_mask", owned_copy)


def overlay_g_on_predictor_camera_y1(
    predictor_camera_y1: np.ndarray,
    predictor_labels: np.ndarray,
    decoded_g: DecodedGenerativeCorrectionV1,
) -> PredictorPreservingOverlayResultV1:
    """Overlay only G-changed scorer supports onto exact predictor camera Y1."""

    if type(decoded_g) is not DecodedGenerativeCorrectionV1:
        raise PredictorPreservingOverlayError("overlay requires exact DecodedGenerativeCorrectionV1")
    if decoded_g.realization_profile is None:
        raise PredictorPreservingOverlayError("overlay G must carry a finite realization profile")
    base = np.asarray(predictor_camera_y1)
    labels = np.asarray(predictor_labels)
    pair_count = decoded_g.labels.shape[0]
    if not 1 <= pair_count <= MAX_BOUNDED_PAIRS:
        raise PredictorPreservingOverlayError("overlay is bounded to at most four pairs")
    if base.dtype != np.uint8 or base.shape != (
        pair_count,
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
        CHANNELS,
    ):
        raise PredictorPreservingOverlayError("predictor camera Y1 must be uint8 [pair,874,1164,3]")
    if labels.dtype != np.uint8 or labels.shape != (
        pair_count,
        SCORER_HEIGHT,
        SCORER_WIDTH,
    ):
        raise PredictorPreservingOverlayError("predictor labels must be uint8 [pair,384,512]")
    if int(labels.min()) < 0 or int(labels.max()) > 4:
        raise PredictorPreservingOverlayError("predictor labels escaped the five-class universe")
    changed = np.ascontiguousarray(decoded_g.labels != labels, dtype=np.bool_)
    changed_cells = int(np.count_nonzero(changed))
    if changed_cells < 1:
        raise PredictorPreservingOverlayError("G changed no semantic cells and has no overlay ownership")

    target_rgb = _paint_corrected_labels(decoded_g.labels, decoded_g.realization_profile)
    operator = _operator()
    corrected = np.ascontiguousarray(base.copy())
    owned = np.zeros((pair_count, CAMERA_HEIGHT, CAMERA_WIDTH), dtype=np.bool_)

    for pair_index in range(pair_count):
        scorer_rows, scorer_cols = np.nonzero(changed[pair_index])
        for scorer_row, scorer_col in zip(scorer_rows.tolist(), scorer_cols.tolist(), strict=True):
            row_support = operator.row_supports[scorer_row]
            col_support = operator.col_supports[scorer_col]
            for camera_row in row_support.indices:
                for camera_col in col_support.indices:
                    if owned[pair_index, camera_row, camera_col]:
                        raise PredictorPreservingOverlayError("changed scorer supports unexpectedly overlap")
                    owned[pair_index, camera_row, camera_col] = True
                    corrected[pair_index, camera_row, camera_col] = target_rgb[
                        pair_index,
                        scorer_row,
                        scorer_col,
                    ]

    base_numerators: list[np.ndarray] = []
    corrected_numerators: list[np.ndarray] = []
    denominator: int | None = None
    for pair_index in range(pair_count):
        try:
            base_num, base_den = operator.apply_numerators(base[pair_index])
            corrected_num, corrected_den = operator.apply_numerators(corrected[pair_index])
        except Uint8LatticeError as exc:
            raise PredictorPreservingOverlayError("camera-to-scorer exact numerator proof failed") from exc
        if base_den != corrected_den or (denominator is not None and denominator != base_den):
            raise PredictorPreservingOverlayError("scorer denominator changed across overlay pairs")
        denominator = base_den
        expected = target_rgb[pair_index].astype(np.int64) * base_den
        if not np.array_equal(corrected_num[changed[pair_index]], expected[changed[pair_index]]):
            raise PredictorPreservingOverlayError("G-owned scorer numerators do not realize target RGB exactly")
        if not np.array_equal(corrected_num[~changed[pair_index]], base_num[~changed[pair_index]]):
            raise PredictorPreservingOverlayError("overlay changed an unowned scorer numerator")
        if not np.array_equal(corrected[pair_index][~owned[pair_index]], base[pair_index][~owned[pair_index]]):
            raise PredictorPreservingOverlayError("overlay changed an unowned predictor camera value")
        base_numerators.append(np.ascontiguousarray(base_num))
        corrected_numerators.append(np.ascontiguousarray(corrected_num))

    actual_camera_changes = int(np.count_nonzero(corrected != base))
    if actual_camera_changes < 1:
        raise PredictorPreservingOverlayError("G ownership caused no realized camera-byte change")
    base_num_stack = np.stack(base_numerators, axis=0)
    corrected_num_stack = np.stack(corrected_numerators, axis=0)
    receipt = PredictorPreservingOverlayReceiptV1(
        schema=RECEIPT_SCHEMA,
        pair_count=pair_count,
        correction_packet_sha256=decoded_g.correction_packet_sha256,
        realization_profile_sha256=_profile_sha256(decoded_g.realization_profile),
        predictor_labels_sha256=_sha256(memoryview(np.ascontiguousarray(labels)).cast("B")),
        corrected_labels_sha256=_sha256(memoryview(decoded_g.labels).cast("B")),
        changed_label_mask_sha256=_sha256(memoryview(changed).cast("B")),
        base_camera_y1_sha256=_sha256(memoryview(np.ascontiguousarray(base)).cast("B")),
        corrected_camera_y1_sha256=_sha256(memoryview(corrected).cast("B")),
        base_scorer_numerators_sha256=_sha256(memoryview(base_num_stack).cast("B")),
        corrected_scorer_numerators_sha256=_sha256(memoryview(corrected_num_stack).cast("B")),
        changed_scorer_cells=changed_cells,
        changed_scorer_values=changed_cells * CHANNELS,
        owned_camera_values=int(np.count_nonzero(owned)) * CHANNELS,
        actually_changed_camera_values=actual_camera_changes,
        scorer_denominator=int(denominator),
        camera_values_total=pair_count * CAMERA_HEIGHT * CAMERA_WIDTH * CHANNELS,
        scorer_values_total=pair_count * SCORER_HEIGHT * SCORER_WIDTH * CHANNELS,
    )
    return PredictorPreservingOverlayResultV1(
        camera_y1=corrected,
        changed_label_mask=changed,
        owned_camera_mask=owned,
        receipt=receipt,
    )


def parse_predictor_preserving_overlay_receipt(payload: bytes) -> PredictorPreservingOverlayReceiptV1:
    """Strictly parse, type-check, and canonically re-emit an overlay receipt."""

    if type(payload) is not bytes or not payload:
        raise PredictorPreservingOverlayError("overlay receipt must be nonempty exact bytes")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PredictorPreservingOverlayError(f"overlay receipt repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=unique_pairs)
    except PredictorPreservingOverlayError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictorPreservingOverlayError("overlay receipt is not strict ASCII JSON") from exc
    if type(value) is not dict or set(value) != set(PredictorPreservingOverlayReceiptV1.__dataclass_fields__):
        raise PredictorPreservingOverlayError("overlay receipt fields differ from the closed schema")
    if _canonical_json(value) != payload:
        raise PredictorPreservingOverlayError("overlay receipt is not canonical JSON")
    try:
        receipt = PredictorPreservingOverlayReceiptV1(**value)
    except TypeError as exc:
        raise PredictorPreservingOverlayError("overlay receipt contains invalid typed fields") from exc
    if receipt.to_receipt_bytes() != payload:
        raise PredictorPreservingOverlayError("overlay receipt changed on typed parse-back")
    return receipt


__all__ = [
    "MAX_BOUNDED_PAIRS",
    "PredictorPreservingOverlayError",
    "PredictorPreservingOverlayReceiptV1",
    "PredictorPreservingOverlayResultV1",
    "overlay_g_on_predictor_camera_y1",
    "parse_predictor_preserving_overlay_receipt",
]
