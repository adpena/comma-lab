# SPDX-License-Identifier: MIT
"""Research-only reverse-causal grammar for a coupled ``(Y0, Y1)`` preimage.

The decoder order is part of the contract, not documentation:

``G labels -> exact realized uint8 Y1 -> verify Y1 content hash -> Y0 | exact Y1``.

The returned pair is nevertheless chronological ``(Y0, Y1)``.  This module
does not load a scorer, accept evaluator obligations, or claim that either
mechanical fibre improves PoseNet.  It is an executable L0 architecture slice
that closes the causality and packet-shape questions before any n600 evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from tac.optimization.direct_description_carrier_compose import (
    REALIZATION_PAINT_ORDER,
    ROLE_CLASS_IDS,
    ReceiverRealizationProfileV1,
)
from tac.witness_dsl.factorized_v9_predictor import SEMANTIC_HEIGHT, SEMANTIC_WIDTH
from tac.witness_dsl.generative_taskspace_correction import (
    DecodedGenerativeCorrectionV1,
    PredictorSemanticStateV1,
)

PACKET_SCHEMA: Final = "tac.coupled_preimage_program.v1"
PACKET_ENVELOPE_SCHEMA: Final = "tac.coupled_preimage_program.envelope.v1"
DECODER_CONTRACT_ID: Final = "tac.reverse_causal_g_to_y1_to_y0_given_exact_y1.v1"
LINEAGE_OWNERSHIP: Final = "DECLARED_ORIGINAL_OWN_UNVERIFIED"
LINEAGE_ORIGIN: Final = "DECLARED_LOCAL_ORIGINAL_PROGRAM_UNVERIFIED"
LINEAGE_CLASSIFICATION_AUTHORITY: Final = "PRODUCER_DECLARATION_UNVERIFIED_BY_TYPED_DERIVATION"
PAYLOAD_POLICY: Final = (
    "typed_numeric_fibres_with_producer_declared_payload_classification_pending_complete_derivation_lineage.v2"
)
DECODER_BINDING_SCOPE: Final = "direct_module_file_bytes_only_nontransitive.v1"
CONTROL_CARDINALITY_POLICY: Final = "exactly_one_mode_specific_control_row_per_source_pair.v1"
EXPRESSIVITY_SCOPE: Final = "one_integer_translation_and_global_or_per_role_rgb_delta_per_pair.v1"
BEHAVIOR_QUOTIENT: Final = "dy[-383,383]_dx[-511,511]_rgb_delta[-255,255]_coordinatewise_universal_aliases_rejected.v1"
TYPED_LINEAGE_BLOCKER: Final = "complete_typed_derivation_and_standalone_archive_lineage_for_every_control_byte_absent"
CAUSAL_MATERIALIZATION_ORDER: Final = (
    "G_SEMANTIC_LABELS",
    "EXACT_REALIZED_UINT8_Y1",
    "VERIFY_EXACT_Y1_CONTENT_SHA256",
    "MATERIALIZE_Y0_GIVEN_EXACT_Y1",
)
CHRONOLOGICAL_OUTPUT_ORDER: Final = ("Y0", "Y1")
_SHA256_HEX_LENGTH: Final = 64
_ABI_INT16_MIN: Final = -(1 << 15)
_ABI_INT16_MAX: Final = (1 << 15) - 1
_ABI_UINT32_MAX: Final = (1 << 32) - 1
_CANONICAL_SHIFT_Y_LIMIT: Final = SEMANTIC_HEIGHT - 1
_CANONICAL_SHIFT_X_LIMIT: Final = SEMANTIC_WIDTH - 1
_CANONICAL_RGB_DELTA_LIMIT: Final = 255
DECODER_DIRECT_SOURCE_SHA256: Final = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class CoupledPreimageProgramError(ValueError):
    """Malformed packet, forbidden lineage, or failed reverse-causal binding."""


class CoupledPreimageMode(StrEnum):
    """The complete v1 mode universe; strings outside it are not extensible data."""

    FRAME1_ANCHORED_Y0_FIBRE = "FRAME1_ANCHORED_Y0_FIBRE"
    JOINT_SHARED_SKELETON_TWO_FIBRE = "JOINT_SHARED_SKELETON_TWO_FIBRE"


class DeclaredLineageRole(StrEnum):
    """Every declared source or payload role in the closed reference packet."""

    PREDICTOR_SEMANTIC_SOURCE = "PREDICTOR_SEMANTIC_SOURCE"
    GENERATIVE_CORRECTION_SOURCE = "GENERATIVE_CORRECTION_SOURCE"
    EXACT_FRAME1_UINT8 = "EXACT_FRAME1_UINT8"
    FRAME0_ANCHORED_FIBRE_CONTROLS = "FRAME0_ANCHORED_FIBRE_CONTROLS"
    SHARED_SEMANTIC_SKELETON = "SHARED_SEMANTIC_SKELETON"
    FRAME0_SHARED_SKELETON_FIBRE_CONTROLS = "FRAME0_SHARED_SKELETON_FIBRE_CONTROLS"


def _canonical_json(value: Any) -> bytes:
    def require_string_keys(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise CoupledPreimageProgramError(f"canonical JSON key at {path} must be a string")
                require_string_keys(child, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                require_string_keys(child, f"{path}[{index}]")

    require_string_keys(value, "root")
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CoupledPreimageProgramError("value is not canonical ASCII JSON") from exc


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CoupledPreimageProgramError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_canonical_json(payload: bytes) -> Any:
    if not isinstance(payload, bytes):
        raise CoupledPreimageProgramError("packet must be exact bytes")
    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoupledPreimageProgramError("packet is not strict ASCII JSON") from exc
    if _canonical_json(value) != payload:
        raise CoupledPreimageProgramError("packet does not use the canonical JSON spelling")
    return value


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(_canonical_json(value))


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CoupledPreimageProgramError(f"{label} must be lowercase SHA-256 hex")
    return value


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CoupledPreimageProgramError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise CoupledPreimageProgramError(
            f"{label} keys differ from the closed schema: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise CoupledPreimageProgramError(f"{label} must be an exact integer")
    return value


def _abi_int16(value: Any, label: str) -> int:
    result = _exact_int(value, label)
    if not _ABI_INT16_MIN <= result <= _ABI_INT16_MAX:
        raise CoupledPreimageProgramError(f"{label} is not signed-int16 ABI representable")
    return result


def _canonical_shift_y(value: Any, label: str) -> int:
    result = _abi_int16(value, label)
    if not -_CANONICAL_SHIFT_Y_LIMIT <= result <= _CANONICAL_SHIFT_Y_LIMIT:
        raise CoupledPreimageProgramError(
            f"{label} is a noncanonical alias outside the behavior-distinct Y-shift quotient"
        )
    return result


def _canonical_shift_x(value: Any, label: str) -> int:
    result = _abi_int16(value, label)
    if not -_CANONICAL_SHIFT_X_LIMIT <= result <= _CANONICAL_SHIFT_X_LIMIT:
        raise CoupledPreimageProgramError(
            f"{label} is a noncanonical alias outside the behavior-distinct X-shift quotient"
        )
    return result


def _canonical_rgb_delta(value: Any, label: str) -> int:
    result = _abi_int16(value, label)
    if not -_CANONICAL_RGB_DELTA_LIMIT <= result <= _CANONICAL_RGB_DELTA_LIMIT:
        raise CoupledPreimageProgramError(
            f"{label} is a noncanonical alias outside the behavior-distinct uint8-delta quotient"
        )
    return result


def _rgb_i16(value: Any, label: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise CoupledPreimageProgramError(f"{label} must be one RGB triple")
    return tuple(_canonical_rgb_delta(channel, f"{label}[{index}]") for index, channel in enumerate(value))  # type: ignore[return-value]


def _array_content_sha256(value: np.ndarray, *, role: str) -> str:
    array = np.ascontiguousarray(value)
    header = _canonical_json(
        {
            "dtype": array.dtype.str,
            "role": role,
            "schema": "tac.ndarray_content.v1",
            "shape": list(array.shape),
        }
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\x00")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _realization_profile_sha256(profile: ReceiverRealizationProfileV1) -> str:
    return _canonical_sha256(
        {
            "amplitude_u8": profile.amplitude_u8,
            "coverage_radius": profile.coverage_radius,
            "role_order": list(REALIZATION_PAINT_ORDER),
            "role_rgb_u8": [list(row) for row in profile.role_rgb_u8],
            "schema": "tac.receiver_realization_profile_binding.v1",
        }
    )


_DECODER_CONTRACT_MANIFEST: Final = {
    "causal_materialization_order": list(CAUSAL_MATERIALIZATION_ORDER),
    "chronological_output_order": list(CHRONOLOGICAL_OUTPUT_ORDER),
    "contract_id": DECODER_CONTRACT_ID,
    "frame_shape": [SEMANTIC_HEIGHT, SEMANTIC_WIDTH, 3],
    "modes": [mode.value for mode in CoupledPreimageMode],
    "numeric_rule": "integer_transport_int32_add_clip_uint8.v1",
    "source_module": "tac.witness_dsl.coupled_preimage_program",
}
DECODER_CONTRACT_SHA256: Final = _canonical_sha256(_DECODER_CONTRACT_MANIFEST)


@dataclass(frozen=True, slots=True)
class Frame1AnchoredY0FibreControlV1:
    """Photometric/geometric Y0 actuator conditioned on exact Y1; no Pose6 claim."""

    source_pair_id: int
    shift_y_i16: int
    shift_x_i16: int
    rgb_delta_i16: tuple[int, int, int]

    def __post_init__(self) -> None:
        _exact_int(self.source_pair_id, "source_pair_id")
        _canonical_shift_y(self.shift_y_i16, "shift_y_i16")
        _canonical_shift_x(self.shift_x_i16, "shift_x_i16")
        if type(self.rgb_delta_i16) is not tuple:
            raise CoupledPreimageProgramError("rgb_delta_i16 must be an exact tuple")
        _rgb_i16(self.rgb_delta_i16, "rgb_delta_i16")

    def as_dict(self) -> dict[str, Any]:
        return {
            "rgb_delta_i16": list(self.rgb_delta_i16),
            "shift_x_i16": self.shift_x_i16,
            "shift_y_i16": self.shift_y_i16,
            "source_pair_id": self.source_pair_id,
        }


@dataclass(frozen=True, slots=True)
class JointSharedSkeletonTwoFibreControlV1:
    """Y0 per-role fibre over the same G skeleton that realizes exact Y1."""

    source_pair_id: int
    shift_y_i16: int
    shift_x_i16: int
    role_rgb_delta_i16: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        _exact_int(self.source_pair_id, "source_pair_id")
        _canonical_shift_y(self.shift_y_i16, "shift_y_i16")
        _canonical_shift_x(self.shift_x_i16, "shift_x_i16")
        if type(self.role_rgb_delta_i16) is not tuple or len(self.role_rgb_delta_i16) != len(REALIZATION_PAINT_ORDER):
            raise CoupledPreimageProgramError("role_rgb_delta_i16 must bind one exact RGB tuple per semantic role")
        for index, row in enumerate(self.role_rgb_delta_i16):
            if type(row) is not tuple:
                raise CoupledPreimageProgramError("each role_rgb_delta_i16 row must be an exact tuple")
            _rgb_i16(row, f"role_rgb_delta_i16[{index}]")

    def as_dict(self) -> dict[str, Any]:
        return {
            "role_rgb_delta_i16": [list(row) for row in self.role_rgb_delta_i16],
            "shift_x_i16": self.shift_x_i16,
            "shift_y_i16": self.shift_y_i16,
            "source_pair_id": self.source_pair_id,
        }


@dataclass(frozen=True, slots=True)
class CoupledPreimageSourceBindingV1:
    predictor_program_sha256: str
    predictor_renderer_sha256: str
    source_pair_ids: tuple[int, ...]
    predictor_labels_sha256: str
    correction_packet_sha256: str
    correction_labels_sha256: str
    realization_profile_sha256: str
    exact_y1_content_sha256: str
    expected_y0_content_sha256: str
    expected_chronological_content_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "predictor_program_sha256",
            "predictor_renderer_sha256",
            "predictor_labels_sha256",
            "correction_packet_sha256",
            "correction_labels_sha256",
            "realization_profile_sha256",
            "exact_y1_content_sha256",
            "expected_y0_content_sha256",
            "expected_chronological_content_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if type(self.source_pair_ids) is not tuple or not self.source_pair_ids:
            raise CoupledPreimageProgramError("source_pair_ids must be one nonempty exact tuple")
        if any(type(pair_id) is not int for pair_id in self.source_pair_ids):
            raise CoupledPreimageProgramError("source_pair_ids must contain exact integers")
        expected = tuple(range(self.source_pair_ids[0], self.source_pair_ids[0] + len(self.source_pair_ids)))
        if self.source_pair_ids != expected or not 0 <= expected[0] < expected[-1] + 1 <= 600:
            raise CoupledPreimageProgramError("source_pair_ids must be a contiguous contest-ABI subset of [0,600)")

    @property
    def pair_count(self) -> int:
        return len(self.source_pair_ids)

    @property
    def predictor_semantic_binding_sha256(self) -> str:
        return _canonical_sha256(
            {
                "predictor_labels_sha256": self.predictor_labels_sha256,
                "predictor_program_sha256": self.predictor_program_sha256,
                "predictor_renderer_sha256": self.predictor_renderer_sha256,
                "schema": "tac.predictor_semantic_source_binding.v1",
                "source_pair_ids": list(self.source_pair_ids),
            }
        )

    @property
    def correction_binding_sha256(self) -> str:
        return _canonical_sha256(
            {
                "correction_labels_sha256": self.correction_labels_sha256,
                "correction_packet_sha256": self.correction_packet_sha256,
                "realization_profile_sha256": self.realization_profile_sha256,
                "schema": "tac.decoded_generative_correction_binding.v1",
            }
        )

    @property
    def binding_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "correction_labels_sha256": self.correction_labels_sha256,
            "correction_packet_sha256": self.correction_packet_sha256,
            "exact_y1_content_sha256": self.exact_y1_content_sha256,
            "expected_y0_content_sha256": self.expected_y0_content_sha256,
            "expected_chronological_content_sha256": self.expected_chronological_content_sha256,
            "predictor_labels_sha256": self.predictor_labels_sha256,
            "predictor_program_sha256": self.predictor_program_sha256,
            "predictor_renderer_sha256": self.predictor_renderer_sha256,
            "realization_profile_sha256": self.realization_profile_sha256,
            "source_pair_ids": list(self.source_pair_ids),
        }


@dataclass(frozen=True, slots=True)
class DeclaredLineageItemV1:
    role: DeclaredLineageRole
    content_sha256: str
    ownership: Literal["DECLARED_ORIGINAL_OWN_UNVERIFIED"] = LINEAGE_OWNERSHIP
    origin: Literal["DECLARED_LOCAL_ORIGINAL_PROGRAM_UNVERIFIED"] = LINEAGE_ORIGIN
    classification_authority: Literal["PRODUCER_DECLARATION_UNVERIFIED_BY_TYPED_DERIVATION"] = (
        LINEAGE_CLASSIFICATION_AUTHORITY
    )
    candidate_lineage_proven: Literal[False] = False
    originality_proven: Literal[False] = False

    def __post_init__(self) -> None:
        if type(self.role) is not DeclaredLineageRole:
            raise CoupledPreimageProgramError("lineage role must use the closed DeclaredLineageRole enum")
        _require_sha256(self.content_sha256, "lineage content_sha256")
        if (
            self.ownership != LINEAGE_OWNERSHIP
            or self.origin != LINEAGE_ORIGIN
            or self.classification_authority != LINEAGE_CLASSIFICATION_AUTHORITY
            or self.candidate_lineage_proven is not False
            or self.originality_proven is not False
        ):
            raise CoupledPreimageProgramError("lineage items must remain producer-declared and unverified")

    def as_dict(self) -> dict[str, Any]:
        return {
            "content_sha256": self.content_sha256,
            "classification_authority": self.classification_authority,
            "candidate_lineage_proven": self.candidate_lineage_proven,
            "origin": self.origin,
            "originality_proven": self.originality_proven,
            "ownership": self.ownership,
            "role": self.role.value,
        }


def _controls_sha256(
    mode: CoupledPreimageMode,
    anchored: tuple[Frame1AnchoredY0FibreControlV1, ...],
    joint: tuple[JointSharedSkeletonTwoFibreControlV1, ...],
) -> str:
    rows = anchored if mode is CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE else joint
    return _canonical_sha256(
        {
            "mode": mode.value,
            "rows": [row.as_dict() for row in rows],
            "schema": "tac.coupled_preimage_controls.v1",
        }
    )


def _required_lineage_content(
    mode: CoupledPreimageMode,
    source_binding: CoupledPreimageSourceBindingV1,
    anchored: tuple[Frame1AnchoredY0FibreControlV1, ...],
    joint: tuple[JointSharedSkeletonTwoFibreControlV1, ...],
) -> dict[DeclaredLineageRole, str]:
    required = {
        DeclaredLineageRole.PREDICTOR_SEMANTIC_SOURCE: source_binding.predictor_semantic_binding_sha256,
        DeclaredLineageRole.GENERATIVE_CORRECTION_SOURCE: source_binding.correction_binding_sha256,
        DeclaredLineageRole.EXACT_FRAME1_UINT8: source_binding.exact_y1_content_sha256,
    }
    controls_sha256 = _controls_sha256(mode, anchored, joint)
    if mode is CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE:
        required[DeclaredLineageRole.FRAME0_ANCHORED_FIBRE_CONTROLS] = controls_sha256
    else:
        required[DeclaredLineageRole.SHARED_SEMANTIC_SKELETON] = source_binding.correction_labels_sha256
        required[DeclaredLineageRole.FRAME0_SHARED_SKELETON_FIBRE_CONTROLS] = controls_sha256
    return required


def _validate_control_coverage(
    mode: CoupledPreimageMode,
    source_pair_ids: tuple[int, ...],
    anchored: tuple[Frame1AnchoredY0FibreControlV1, ...],
    joint: tuple[JointSharedSkeletonTwoFibreControlV1, ...],
) -> None:
    if type(mode) is not CoupledPreimageMode:
        raise CoupledPreimageProgramError("mode must use the closed CoupledPreimageMode enum")
    if (
        type(source_pair_ids) is not tuple
        or not source_pair_ids
        or any(type(item) is not int for item in source_pair_ids)
    ):
        raise CoupledPreimageProgramError("control coverage requires exact nonempty source-pair IDs")
    if type(anchored) is not tuple or any(type(row) is not Frame1AnchoredY0FibreControlV1 for row in anchored):
        raise CoupledPreimageProgramError("anchored controls changed typed ABI")
    if type(joint) is not tuple or any(type(row) is not JointSharedSkeletonTwoFibreControlV1 for row in joint):
        raise CoupledPreimageProgramError("joint controls changed typed ABI")
    if mode is CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE:
        active_ids = tuple(row.source_pair_id for row in anchored)
        if joint:
            raise CoupledPreimageProgramError("anchored mode cannot carry joint-skeleton controls")
    else:
        active_ids = tuple(row.source_pair_id for row in joint)
        if anchored:
            raise CoupledPreimageProgramError("joint-skeleton mode cannot carry anchored controls")
    if active_ids != source_pair_ids:
        raise CoupledPreimageProgramError("active controls must bind every source pair exactly once and in order")


@dataclass(frozen=True, slots=True)
class CoupledPreimageProgramV1:
    mode: CoupledPreimageMode
    source_binding: CoupledPreimageSourceBindingV1
    anchored_controls: tuple[Frame1AnchoredY0FibreControlV1, ...]
    joint_controls: tuple[JointSharedSkeletonTwoFibreControlV1, ...]
    lineage: tuple[DeclaredLineageItemV1, ...]

    def __post_init__(self) -> None:
        if type(self.mode) is not CoupledPreimageMode:
            raise CoupledPreimageProgramError("mode must use the closed CoupledPreimageMode enum")
        if type(self.source_binding) is not CoupledPreimageSourceBindingV1:
            raise CoupledPreimageProgramError("source_binding changed typed ABI")
        typed_rows = (
            (self.anchored_controls, Frame1AnchoredY0FibreControlV1, "anchored_controls"),
            (self.joint_controls, JointSharedSkeletonTwoFibreControlV1, "joint_controls"),
            (self.lineage, DeclaredLineageItemV1, "lineage"),
        )
        for rows, row_type, label in typed_rows:
            if type(rows) is not tuple or any(type(row) is not row_type for row in rows):
                raise CoupledPreimageProgramError(f"{label} must contain only exact {row_type.__name__} rows")

        _validate_control_coverage(
            self.mode,
            self.source_binding.source_pair_ids,
            self.anchored_controls,
            self.joint_controls,
        )

        required = _required_lineage_content(
            self.mode,
            self.source_binding,
            self.anchored_controls,
            self.joint_controls,
        )
        actual: dict[DeclaredLineageRole, str] = {}
        for item in self.lineage:
            if item.role in actual:
                raise CoupledPreimageProgramError(f"duplicate lineage role: {item.role.value}")
            actual[item.role] = item.content_sha256
        if actual != required:
            missing = sorted(role.value for role in required.keys() - actual.keys())
            extra = sorted(role.value for role in actual.keys() - required.keys())
            mismatched = sorted(
                role.value for role in required.keys() & actual.keys() if required[role] != actual[role]
            )
            raise CoupledPreimageProgramError(
                f"lineage roles/content differ from the mode contract: missing={missing}, extra={extra}, "
                f"mismatched={mismatched}"
            )
        if tuple(item.role.value for item in self.lineage) != tuple(sorted(item.role.value for item in self.lineage)):
            raise CoupledPreimageProgramError("lineage items must use canonical role order")

    @property
    def research_only(self) -> Literal[True]:
        return True

    @property
    def candidate_score_claim(self) -> Literal[False]:
        return False

    @property
    def promotion_eligible(self) -> Literal[False]:
        return False

    @property
    def active_control_count(self) -> int:
        return len(self.anchored_controls) + len(self.joint_controls)

    def as_body(self) -> dict[str, Any]:
        return {
            "controls": {
                "frame1_anchored_y0_fibre": [row.as_dict() for row in self.anchored_controls],
                "joint_shared_skeleton_two_fibre": [row.as_dict() for row in self.joint_controls],
            },
            "decoder_binding": {
                "binding_scope": DECODER_BINDING_SCOPE,
                "contract_id": DECODER_CONTRACT_ID,
                "contract_sha256": DECODER_CONTRACT_SHA256,
                "direct_source_sha256": DECODER_DIRECT_SOURCE_SHA256,
                "source_module": "tac.witness_dsl.coupled_preimage_program",
                "standalone_runtime_custody": False,
            },
            "lineage": [item.as_dict() for item in self.lineage],
            "mode": self.mode.value,
            "policy": {
                "additional_compiler_score_threshold_caps_applied": False,
                "candidate_score_claim": False,
                "candidate_lineage_proven": False,
                "control_behavior_quotient": BEHAVIOR_QUOTIENT,
                "chronological_output_order": list(CHRONOLOGICAL_OUTPUT_ORDER),
                "control_cardinality_policy": CONTROL_CARDINALITY_POLICY,
                "decoder_source_binding_scope": DECODER_BINDING_SCOPE,
                "evaluator_evidence_claim": False,
                "expressivity_complete": False,
                "expressivity_scope": EXPRESSIVITY_SCOPE,
                "forbidden_serialized_payload_bytes_independently_verified": False,
                "materialization_order": list(CAUSAL_MATERIALIZATION_ORDER),
                "originality_proven": False,
                "payload_policy": PAYLOAD_POLICY,
                "payload_classification_authority": LINEAGE_CLASSIFICATION_AUTHORITY,
                "producer_declared_forbidden_serialized_payload_bytes": 0,
                "promotion_eligible": False,
                "research_only": True,
                "standalone_runtime_custody": False,
                "typed_derivation_archive_lineage_blocker": TYPED_LINEAGE_BLOCKER,
            },
            "resource_counts": {
                "active_control_rows": self.active_control_count,
                "behavior_map_known_noninjective_for_specific_inputs": True,
                "control_parameter_space_per_pair": (
                    str(767 * 1023 * 511**3)
                    if self.mode is CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE
                    else str(767 * 1023 * 511**15)
                ),
                "lineage_items": len(self.lineage),
                "source_pairs": self.source_binding.pair_count,
            },
            "schema": PACKET_SCHEMA,
            "source_binding": self.source_binding.as_dict(),
        }

    def to_packet(self) -> bytes:
        return _emit_body(self.as_body())


def _emit_body(body: Mapping[str, Any]) -> bytes:
    packet = _canonical_json(
        {
            "body": body,
            "body_sha256": _canonical_sha256(body),
            "schema": PACKET_ENVELOPE_SCHEMA,
        }
    )
    if len(packet) > _ABI_UINT32_MAX:
        raise CoupledPreimageProgramError("packet byte length is not uint32 ABI representable")
    return packet


def _parse_source_binding(value: Any) -> CoupledPreimageSourceBindingV1:
    row = _exact_keys(
        value,
        {
            "correction_labels_sha256",
            "correction_packet_sha256",
            "exact_y1_content_sha256",
            "expected_chronological_content_sha256",
            "expected_y0_content_sha256",
            "predictor_labels_sha256",
            "predictor_program_sha256",
            "predictor_renderer_sha256",
            "realization_profile_sha256",
            "source_pair_ids",
        },
        "source_binding",
    )
    pair_ids = row["source_pair_ids"]
    if not isinstance(pair_ids, list):
        raise CoupledPreimageProgramError("source_pair_ids must be a canonical JSON array")
    return CoupledPreimageSourceBindingV1(
        predictor_program_sha256=row["predictor_program_sha256"],
        predictor_renderer_sha256=row["predictor_renderer_sha256"],
        source_pair_ids=tuple(pair_ids),
        predictor_labels_sha256=row["predictor_labels_sha256"],
        correction_packet_sha256=row["correction_packet_sha256"],
        correction_labels_sha256=row["correction_labels_sha256"],
        realization_profile_sha256=row["realization_profile_sha256"],
        exact_y1_content_sha256=row["exact_y1_content_sha256"],
        expected_y0_content_sha256=row["expected_y0_content_sha256"],
        expected_chronological_content_sha256=row["expected_chronological_content_sha256"],
    )


def _parse_anchored_control(value: Any, index: int) -> Frame1AnchoredY0FibreControlV1:
    row = _exact_keys(
        value,
        {"rgb_delta_i16", "shift_x_i16", "shift_y_i16", "source_pair_id"},
        f"anchored_controls[{index}]",
    )
    return Frame1AnchoredY0FibreControlV1(
        source_pair_id=_exact_int(row["source_pair_id"], f"anchored_controls[{index}].source_pair_id"),
        shift_y_i16=_canonical_shift_y(row["shift_y_i16"], f"anchored_controls[{index}].shift_y_i16"),
        shift_x_i16=_canonical_shift_x(row["shift_x_i16"], f"anchored_controls[{index}].shift_x_i16"),
        rgb_delta_i16=_rgb_i16(row["rgb_delta_i16"], f"anchored_controls[{index}].rgb_delta_i16"),
    )


def _parse_joint_control(value: Any, index: int) -> JointSharedSkeletonTwoFibreControlV1:
    row = _exact_keys(
        value,
        {"role_rgb_delta_i16", "shift_x_i16", "shift_y_i16", "source_pair_id"},
        f"joint_controls[{index}]",
    )
    role_rows = row["role_rgb_delta_i16"]
    if not isinstance(role_rows, list) or len(role_rows) != len(REALIZATION_PAINT_ORDER):
        raise CoupledPreimageProgramError("joint role deltas must be one JSON RGB row per semantic role")
    return JointSharedSkeletonTwoFibreControlV1(
        source_pair_id=_exact_int(row["source_pair_id"], f"joint_controls[{index}].source_pair_id"),
        shift_y_i16=_canonical_shift_y(row["shift_y_i16"], f"joint_controls[{index}].shift_y_i16"),
        shift_x_i16=_canonical_shift_x(row["shift_x_i16"], f"joint_controls[{index}].shift_x_i16"),
        role_rgb_delta_i16=tuple(
            _rgb_i16(role_row, f"joint_controls[{index}].role_rgb_delta_i16[{role_index}]")
            for role_index, role_row in enumerate(role_rows)
        ),
    )


def _parse_lineage(value: Any, index: int) -> DeclaredLineageItemV1:
    row = _exact_keys(
        value,
        {
            "candidate_lineage_proven",
            "classification_authority",
            "content_sha256",
            "origin",
            "originality_proven",
            "ownership",
            "role",
        },
        f"lineage[{index}]",
    )
    try:
        role = DeclaredLineageRole(row["role"])
    except (TypeError, ValueError) as exc:
        raise CoupledPreimageProgramError(
            f"lineage[{index}] role is forbidden or outside the closed declared role universe"
        ) from exc
    return DeclaredLineageItemV1(
        role=role,
        content_sha256=row["content_sha256"],
        ownership=row["ownership"],
        origin=row["origin"],
        classification_authority=row["classification_authority"],
        candidate_lineage_proven=row["candidate_lineage_proven"],
        originality_proven=row["originality_proven"],
    )


def parse_coupled_preimage_program(packet: bytes) -> CoupledPreimageProgramV1:
    """Strictly parse, bind, and semantically validate one canonical packet."""

    if len(packet) > _ABI_UINT32_MAX:
        raise CoupledPreimageProgramError("packet byte length is not uint32 ABI representable")
    envelope = _exact_keys(
        _decode_canonical_json(packet),
        {"body", "body_sha256", "schema"},
        "packet envelope",
    )
    if envelope["schema"] != PACKET_ENVELOPE_SCHEMA:
        raise CoupledPreimageProgramError("packet envelope schema changed")
    _require_sha256(envelope["body_sha256"], "body_sha256")
    if _canonical_sha256(envelope["body"]) != envelope["body_sha256"]:
        raise CoupledPreimageProgramError("packet body SHA-256 mismatch")
    body = _exact_keys(
        envelope["body"],
        {
            "controls",
            "decoder_binding",
            "lineage",
            "mode",
            "policy",
            "resource_counts",
            "schema",
            "source_binding",
        },
        "packet body",
    )
    if body["schema"] != PACKET_SCHEMA:
        raise CoupledPreimageProgramError("packet body schema changed")
    try:
        mode = CoupledPreimageMode(body["mode"])
    except (TypeError, ValueError) as exc:
        raise CoupledPreimageProgramError("packet mode is outside the closed two-mode universe") from exc

    decoder_binding = _exact_keys(
        body["decoder_binding"],
        {
            "binding_scope",
            "contract_id",
            "contract_sha256",
            "direct_source_sha256",
            "source_module",
            "standalone_runtime_custody",
        },
        "decoder_binding",
    )
    expected_decoder_binding = {
        "binding_scope": DECODER_BINDING_SCOPE,
        "contract_id": DECODER_CONTRACT_ID,
        "contract_sha256": DECODER_CONTRACT_SHA256,
        "direct_source_sha256": DECODER_DIRECT_SOURCE_SHA256,
        "source_module": "tac.witness_dsl.coupled_preimage_program",
        "standalone_runtime_custody": False,
    }
    if dict(decoder_binding) != expected_decoder_binding:
        raise CoupledPreimageProgramError("decoder contract/source binding mismatch")

    policy = _exact_keys(
        body["policy"],
        {
            "additional_compiler_score_threshold_caps_applied",
            "candidate_score_claim",
            "candidate_lineage_proven",
            "chronological_output_order",
            "control_behavior_quotient",
            "control_cardinality_policy",
            "decoder_source_binding_scope",
            "evaluator_evidence_claim",
            "expressivity_complete",
            "expressivity_scope",
            "forbidden_serialized_payload_bytes_independently_verified",
            "materialization_order",
            "originality_proven",
            "payload_policy",
            "payload_classification_authority",
            "producer_declared_forbidden_serialized_payload_bytes",
            "promotion_eligible",
            "research_only",
            "standalone_runtime_custody",
            "typed_derivation_archive_lineage_blocker",
        },
        "policy",
    )
    expected_policy = {
        "additional_compiler_score_threshold_caps_applied": False,
        "candidate_score_claim": False,
        "candidate_lineage_proven": False,
        "control_behavior_quotient": BEHAVIOR_QUOTIENT,
        "chronological_output_order": list(CHRONOLOGICAL_OUTPUT_ORDER),
        "control_cardinality_policy": CONTROL_CARDINALITY_POLICY,
        "decoder_source_binding_scope": DECODER_BINDING_SCOPE,
        "evaluator_evidence_claim": False,
        "expressivity_complete": False,
        "expressivity_scope": EXPRESSIVITY_SCOPE,
        "forbidden_serialized_payload_bytes_independently_verified": False,
        "materialization_order": list(CAUSAL_MATERIALIZATION_ORDER),
        "originality_proven": False,
        "payload_policy": PAYLOAD_POLICY,
        "payload_classification_authority": LINEAGE_CLASSIFICATION_AUTHORITY,
        "producer_declared_forbidden_serialized_payload_bytes": 0,
        "promotion_eligible": False,
        "research_only": True,
        "standalone_runtime_custody": False,
        "typed_derivation_archive_lineage_blocker": TYPED_LINEAGE_BLOCKER,
    }
    if dict(policy) != expected_policy:
        raise CoupledPreimageProgramError("research-only, no-score, payload, or causality policy changed")

    controls = _exact_keys(
        body["controls"],
        {"frame1_anchored_y0_fibre", "joint_shared_skeleton_two_fibre"},
        "controls",
    )
    anchored_rows = controls["frame1_anchored_y0_fibre"]
    joint_rows = controls["joint_shared_skeleton_two_fibre"]
    if not isinstance(anchored_rows, list) or not isinstance(joint_rows, list):
        raise CoupledPreimageProgramError("control families must be canonical JSON arrays")
    anchored = tuple(_parse_anchored_control(row, index) for index, row in enumerate(anchored_rows))
    joint = tuple(_parse_joint_control(row, index) for index, row in enumerate(joint_rows))

    lineage_rows = body["lineage"]
    if not isinstance(lineage_rows, list):
        raise CoupledPreimageProgramError("lineage must be a canonical JSON array")
    lineage = tuple(_parse_lineage(row, index) for index, row in enumerate(lineage_rows))
    source_binding = _parse_source_binding(body["source_binding"])

    resource_counts = _exact_keys(
        body["resource_counts"],
        {
            "active_control_rows",
            "behavior_map_known_noninjective_for_specific_inputs",
            "control_parameter_space_per_pair",
            "lineage_items",
            "source_pairs",
        },
        "resource_counts",
    )
    expected_counts = {
        "active_control_rows": len(anchored) + len(joint),
        "behavior_map_known_noninjective_for_specific_inputs": True,
        "control_parameter_space_per_pair": (
            str(767 * 1023 * 511**3)
            if mode is CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE
            else str(767 * 1023 * 511**15)
        ),
        "lineage_items": len(lineage),
        "source_pairs": source_binding.pair_count,
    }
    for label in ("active_control_rows", "lineage_items", "source_pairs"):
        parsed_value = _exact_int(resource_counts[label], f"resource_counts.{label}")
        if not 0 <= parsed_value <= _ABI_UINT32_MAX:
            raise CoupledPreimageProgramError(f"resource_counts.{label} is not uint32 ABI representable")
    if dict(resource_counts) != expected_counts:
        raise CoupledPreimageProgramError("resource counts do not describe the exact packet")

    program = CoupledPreimageProgramV1(
        mode=mode,
        source_binding=source_binding,
        anchored_controls=anchored,
        joint_controls=joint,
        lineage=lineage,
    )
    if program.to_packet() != packet:
        raise CoupledPreimageProgramError("typed parse did not re-emit byte-identical canonical packet")
    return program


def _derive_source_binding(
    predictor_state: PredictorSemanticStateV1,
    decoded_g: DecodedGenerativeCorrectionV1,
    exact_y1: np.ndarray,
    expected_y0: np.ndarray,
    expected_chronological: np.ndarray,
) -> CoupledPreimageSourceBindingV1:
    if decoded_g.realization_profile is None:
        raise CoupledPreimageProgramError("G must carry a finite realization profile before Y1 can be exact")
    return CoupledPreimageSourceBindingV1(
        predictor_program_sha256=predictor_state.predictor_program_sha256,
        predictor_renderer_sha256=predictor_state.predictor_renderer_sha256,
        source_pair_ids=predictor_state.source_pair_ids,
        predictor_labels_sha256=predictor_state.labels_sha256,
        correction_packet_sha256=decoded_g.correction_packet_sha256,
        correction_labels_sha256=_array_content_sha256(decoded_g.labels, role="decoded_g_semantic_labels"),
        realization_profile_sha256=_realization_profile_sha256(decoded_g.realization_profile),
        exact_y1_content_sha256=_array_content_sha256(exact_y1, role="exact_realized_uint8_y1"),
        expected_y0_content_sha256=_array_content_sha256(expected_y0, role="expected_realized_uint8_y0"),
        expected_chronological_content_sha256=_array_content_sha256(
            expected_chronological,
            role="chronological_y0_y1",
        ),
    )


def _build_lineage(
    mode: CoupledPreimageMode,
    source_binding: CoupledPreimageSourceBindingV1,
    anchored: tuple[Frame1AnchoredY0FibreControlV1, ...],
    joint: tuple[JointSharedSkeletonTwoFibreControlV1, ...],
) -> tuple[DeclaredLineageItemV1, ...]:
    required = _required_lineage_content(mode, source_binding, anchored, joint)
    return tuple(
        DeclaredLineageItemV1(role=role, content_sha256=content_sha256)
        for role, content_sha256 in sorted(required.items(), key=lambda item: item[0].value)
    )


def _validate_live_source_binding(
    source: CoupledPreimageSourceBindingV1,
    predictor_state: PredictorSemanticStateV1,
    decoded_g: DecodedGenerativeCorrectionV1,
) -> None:
    if type(predictor_state) is not PredictorSemanticStateV1:
        raise CoupledPreimageProgramError("decoder requires exact PredictorSemanticStateV1")
    if type(decoded_g) is not DecodedGenerativeCorrectionV1:
        raise CoupledPreimageProgramError("decoder requires exact DecodedGenerativeCorrectionV1")
    if decoded_g.realization_profile is None:
        raise CoupledPreimageProgramError("decoded G has no finite realization profile")
    if decoded_g.labels.shape[0] != predictor_state.pair_count:
        raise CoupledPreimageProgramError("decoded G pair count differs from predictor state")
    live_values = {
        "predictor_program_sha256": predictor_state.predictor_program_sha256,
        "predictor_renderer_sha256": predictor_state.predictor_renderer_sha256,
        "source_pair_ids": predictor_state.source_pair_ids,
        "predictor_labels_sha256": predictor_state.labels_sha256,
        "correction_packet_sha256": decoded_g.correction_packet_sha256,
        "correction_labels_sha256": _array_content_sha256(decoded_g.labels, role="decoded_g_semantic_labels"),
        "realization_profile_sha256": _realization_profile_sha256(decoded_g.realization_profile),
    }
    mismatches = [field for field, live in live_values.items() if getattr(source, field) != live]
    if mismatches:
        raise CoupledPreimageProgramError(f"live P/G source binding mismatch: {sorted(mismatches)}")


def _translated(value: np.ndarray, *, shift_y: int, shift_x: int) -> np.ndarray:
    source_y = np.clip(np.arange(value.shape[0], dtype=np.int64) - shift_y, 0, value.shape[0] - 1)
    source_x = np.clip(np.arange(value.shape[1], dtype=np.int64) - shift_x, 0, value.shape[1] - 1)
    return np.ascontiguousarray(value[source_y[:, None], source_x[None, :]])


def _add_rgb_delta(frame: np.ndarray, delta: tuple[int, int, int]) -> np.ndarray:
    widened = frame.astype(np.int32) + np.asarray(delta, dtype=np.int32)[None, None, :]
    return np.ascontiguousarray(np.clip(widened, 0, 255).astype(np.uint8))


def _materialize_y0(
    program: CoupledPreimageProgramV1,
    exact_y1: np.ndarray,
    semantic_labels: np.ndarray,
) -> np.ndarray:
    """Run only after the caller has verified ``exact_y1_content_sha256``."""

    return _materialize_y0_from_controls(
        program.mode,
        program.source_binding.source_pair_ids,
        program.anchored_controls,
        program.joint_controls,
        exact_y1,
        semantic_labels,
    )


def _materialize_y0_from_controls(
    mode: CoupledPreimageMode,
    source_pair_ids: tuple[int, ...],
    anchored_controls: tuple[Frame1AnchoredY0FibreControlV1, ...],
    joint_controls: tuple[JointSharedSkeletonTwoFibreControlV1, ...],
    exact_y1: np.ndarray,
    semantic_labels: np.ndarray,
) -> np.ndarray:
    """Pure reference materializer over explicit controls and an already-exact Y1."""

    _validate_control_coverage(mode, source_pair_ids, anchored_controls, joint_controls)
    if exact_y1.shape != (len(source_pair_ids), SEMANTIC_HEIGHT, SEMANTIC_WIDTH, 3) or exact_y1.dtype != np.uint8:
        raise CoupledPreimageProgramError("exact Y1 changed materializer uint8 frame ABI")
    if semantic_labels.shape != (len(source_pair_ids), SEMANTIC_HEIGHT, SEMANTIC_WIDTH):
        raise CoupledPreimageProgramError("semantic skeleton changed materializer label ABI")
    output = np.empty_like(exact_y1)
    if mode is CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE:
        for index, control in enumerate(anchored_controls):
            base = _translated(exact_y1[index], shift_y=control.shift_y_i16, shift_x=control.shift_x_i16)
            output[index] = _add_rgb_delta(base, control.rgb_delta_i16)
        return np.ascontiguousarray(output)

    for index, control in enumerate(joint_controls):
        base = _translated(exact_y1[index], shift_y=control.shift_y_i16, shift_x=control.shift_x_i16)
        skeleton = _translated(
            semantic_labels[index],
            shift_y=control.shift_y_i16,
            shift_x=control.shift_x_i16,
        )
        widened = base.astype(np.int32)
        for role_index, role in enumerate(REALIZATION_PAINT_ORDER):
            widened[skeleton == ROLE_CLASS_IDS[role]] += np.asarray(
                control.role_rgb_delta_i16[role_index], dtype=np.int32
            )
        output[index] = np.clip(widened, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(output)


def _immutable_u8(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.uint8).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class DecodedCoupledPreimageV1:
    """Exact mechanical decode with chronological output and causal receipt."""

    y0: np.ndarray
    y1: np.ndarray
    chronological_frames: np.ndarray
    exact_y1_content_sha256: str
    chronological_content_sha256: str
    materialization_order: tuple[str, ...] = CAUSAL_MATERIALIZATION_ORDER
    chronological_output_order: tuple[str, str] = CHRONOLOGICAL_OUTPUT_ORDER
    research_only: Literal[True] = True
    candidate_score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        y0 = np.asarray(self.y0)
        y1 = np.asarray(self.y1)
        chronological = np.asarray(self.chronological_frames)
        expected_shape = (y0.shape[0], SEMANTIC_HEIGHT, SEMANTIC_WIDTH, 3)
        if y0.dtype != np.uint8 or y0.shape != expected_shape or y1.dtype != np.uint8 or y1.shape != expected_shape:
            raise CoupledPreimageProgramError("decoded Y0/Y1 changed exact uint8 frame ABI")
        if chronological.dtype != np.uint8 or chronological.shape != (y0.shape[0], 2, *expected_shape[1:]):
            raise CoupledPreimageProgramError("chronological output changed [pair,Y0/Y1,H,W,RGB] ABI")
        if not np.array_equal(chronological[:, 0], y0) or not np.array_equal(chronological[:, 1], y1):
            raise CoupledPreimageProgramError("chronological output is not exact (Y0,Y1)")
        _require_sha256(self.exact_y1_content_sha256, "exact_y1_content_sha256")
        _require_sha256(self.chronological_content_sha256, "chronological_content_sha256")
        if _array_content_sha256(y1, role="exact_realized_uint8_y1") != self.exact_y1_content_sha256:
            raise CoupledPreimageProgramError("decoded Y1 hash does not match exact realized uint8 bytes")
        if _array_content_sha256(chronological, role="chronological_y0_y1") != self.chronological_content_sha256:
            raise CoupledPreimageProgramError("chronological content hash mismatch")
        if (
            self.materialization_order != CAUSAL_MATERIALIZATION_ORDER
            or self.chronological_output_order != CHRONOLOGICAL_OUTPUT_ORDER
            or self.research_only is not True
            or self.candidate_score_claim is not False
        ):
            raise CoupledPreimageProgramError("decoded research-only causality policy changed")
        object.__setattr__(self, "y0", _immutable_u8(y0))
        object.__setattr__(self, "y1", _immutable_u8(y1))
        object.__setattr__(self, "chronological_frames", _immutable_u8(chronological))


def decode_coupled_preimage_program(
    packet: bytes,
    *,
    predictor_state: PredictorSemanticStateV1,
    decoded_g: DecodedGenerativeCorrectionV1,
) -> DecodedCoupledPreimageV1:
    """Decode in reverse-causal construction order and return chronological frames."""

    program = parse_coupled_preimage_program(packet)
    _validate_live_source_binding(program.source_binding, predictor_state, decoded_g)

    # The ordering below is the architecture's executable crux.  Nothing that
    # can materialize Y0 is called until the exact realized uint8 Y1 hash has
    # been computed from G and compared with the packet's conditioning hash.
    exact_y1 = np.ascontiguousarray(decoded_g.paint_rgb(), dtype=np.uint8)
    actual_y1_sha256 = _array_content_sha256(exact_y1, role="exact_realized_uint8_y1")
    if actual_y1_sha256 != program.source_binding.exact_y1_content_sha256:
        raise CoupledPreimageProgramError("exact realized uint8 Y1 conditioning hash mismatch before Y0")

    y0 = _materialize_y0(program, exact_y1, decoded_g.labels)
    chronological = np.ascontiguousarray(np.stack((y0, exact_y1), axis=1), dtype=np.uint8)
    actual_y0_sha256 = _array_content_sha256(y0, role="expected_realized_uint8_y0")
    actual_chronological_sha256 = _array_content_sha256(chronological, role="chronological_y0_y1")
    if actual_y0_sha256 != program.source_binding.expected_y0_content_sha256:
        raise CoupledPreimageProgramError("materialized Y0 differs from the packet-bound expected output")
    if actual_chronological_sha256 != program.source_binding.expected_chronological_content_sha256:
        raise CoupledPreimageProgramError("chronological output differs from the packet-bound expected output")
    return DecodedCoupledPreimageV1(
        y0=y0,
        y1=exact_y1,
        chronological_frames=chronological,
        exact_y1_content_sha256=actual_y1_sha256,
        chronological_content_sha256=actual_chronological_sha256,
    )


@dataclass(frozen=True, slots=True)
class CoupledPreimageCompileReceiptV1:
    packet_bytes: int
    packet_sha256: str
    source_binding_sha256: str
    decoder_direct_source_sha256: str
    exact_y1_content_sha256: str
    expected_y0_content_sha256: str
    chronological_content_sha256: str
    mode: CoupledPreimageMode
    source_pairs: int
    active_control_rows: int
    lineage_items: int
    control_cardinality_policy: Literal["exactly_one_mode_specific_control_row_per_source_pair.v1"] = (
        CONTROL_CARDINALITY_POLICY
    )
    expressivity_scope: Literal["one_integer_translation_and_global_or_per_role_rgb_delta_per_pair.v1"] = (
        EXPRESSIVITY_SCOPE
    )
    additional_compiler_score_threshold_caps_applied: Literal[False] = False
    expressivity_complete: Literal[False] = False
    producer_declared_serialized_scorer_gt_oracle_e_teacher_bytes: Literal[0] = 0
    forbidden_serialized_payload_bytes_independently_verified: Literal[False] = False
    candidate_lineage_proven: Literal[False] = False
    originality_proven: Literal[False] = False
    standalone_runtime_custody: Literal[False] = False
    research_only: Literal[True] = True
    candidate_score_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    n600_evidence_claim: Literal[False] = False


@dataclass(frozen=True, slots=True)
class CompiledCoupledPreimageProgramV1:
    packet: bytes
    program: CoupledPreimageProgramV1
    decoded: DecodedCoupledPreimageV1
    receipt: CoupledPreimageCompileReceiptV1


def compile_coupled_preimage_program(
    predictor_state: PredictorSemanticStateV1,
    decoded_g: DecodedGenerativeCorrectionV1,
    *,
    mode: CoupledPreimageMode,
    anchored_controls: tuple[Frame1AnchoredY0FibreControlV1, ...] = (),
    joint_controls: tuple[JointSharedSkeletonTwoFibreControlV1, ...] = (),
) -> CompiledCoupledPreimageProgramV1:
    """Compile and mechanically decode one fully bound research packet."""

    if type(predictor_state) is not PredictorSemanticStateV1:
        raise CoupledPreimageProgramError("compiler requires exact PredictorSemanticStateV1")
    if type(decoded_g) is not DecodedGenerativeCorrectionV1:
        raise CoupledPreimageProgramError("compiler requires exact DecodedGenerativeCorrectionV1")
    if type(mode) is not CoupledPreimageMode:
        raise CoupledPreimageProgramError("compiler mode must use CoupledPreimageMode")
    if type(anchored_controls) is not tuple or type(joint_controls) is not tuple:
        raise CoupledPreimageProgramError("control families must be exact tuples")
    if decoded_g.labels.shape[0] != predictor_state.pair_count:
        raise CoupledPreimageProgramError("decoded G pair count differs from predictor state")
    if decoded_g.realization_profile is None:
        raise CoupledPreimageProgramError("G must carry a realization profile for exact uint8 Y1")

    exact_y1 = np.ascontiguousarray(decoded_g.paint_rgb(), dtype=np.uint8)
    expected_y0 = _materialize_y0_from_controls(
        mode,
        predictor_state.source_pair_ids,
        anchored_controls,
        joint_controls,
        exact_y1,
        decoded_g.labels,
    )
    expected_chronological = np.ascontiguousarray(np.stack((expected_y0, exact_y1), axis=1), dtype=np.uint8)
    source_binding = _derive_source_binding(
        predictor_state,
        decoded_g,
        exact_y1,
        expected_y0,
        expected_chronological,
    )
    lineage = _build_lineage(mode, source_binding, anchored_controls, joint_controls)
    program = CoupledPreimageProgramV1(
        mode=mode,
        source_binding=source_binding,
        anchored_controls=anchored_controls,
        joint_controls=joint_controls,
        lineage=lineage,
    )
    packet = program.to_packet()
    parsed = parse_coupled_preimage_program(packet)
    decoded = decode_coupled_preimage_program(
        packet,
        predictor_state=predictor_state,
        decoded_g=decoded_g,
    )
    return CompiledCoupledPreimageProgramV1(
        packet=packet,
        program=parsed,
        decoded=decoded,
        receipt=CoupledPreimageCompileReceiptV1(
            packet_bytes=len(packet),
            packet_sha256=_sha256(packet),
            source_binding_sha256=source_binding.binding_sha256,
            decoder_direct_source_sha256=DECODER_DIRECT_SOURCE_SHA256,
            exact_y1_content_sha256=decoded.exact_y1_content_sha256,
            expected_y0_content_sha256=source_binding.expected_y0_content_sha256,
            chronological_content_sha256=decoded.chronological_content_sha256,
            mode=mode,
            source_pairs=source_binding.pair_count,
            active_control_rows=program.active_control_count,
            lineage_items=len(program.lineage),
        ),
    )


@dataclass(frozen=True, slots=True)
class RequiredReferenceRemovalVariantV1:
    """Parser-level required-reference removal; not a receiver-causal ablation."""

    removed_role: DeclaredLineageRole
    packet: bytes
    packet_sha256: str


def build_required_reference_removal_packet(packet: bytes, role: DeclaredLineageRole) -> bytes:
    """Remove one declared reference row and recompute the packet envelope."""

    if type(role) is not DeclaredLineageRole:
        raise CoupledPreimageProgramError("removal role must use DeclaredLineageRole")
    program = parse_coupled_preimage_program(packet)
    if role not in {item.role for item in program.lineage}:
        raise CoupledPreimageProgramError(f"lineage role is absent from this mode: {role.value}")
    envelope = _decode_canonical_json(packet)
    body = dict(envelope["body"])
    body["lineage"] = [item for item in body["lineage"] if item["role"] != role.value]
    counts = dict(body["resource_counts"])
    counts["lineage_items"] = len(body["lineage"])
    body["resource_counts"] = counts
    return _emit_body(body)


def required_reference_removal_variants(packet: bytes) -> tuple[RequiredReferenceRemovalVariantV1, ...]:
    """Return parser-level removal variants for every required declaration."""

    program = parse_coupled_preimage_program(packet)
    result: list[RequiredReferenceRemovalVariantV1] = []
    for item in program.lineage:
        removed = build_required_reference_removal_packet(packet, item.role)
        result.append(
            RequiredReferenceRemovalVariantV1(
                removed_role=item.role,
                packet=removed,
                packet_sha256=_sha256(removed),
            )
        )
    return tuple(result)
