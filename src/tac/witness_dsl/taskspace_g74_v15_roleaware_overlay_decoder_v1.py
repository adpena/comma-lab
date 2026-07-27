# SPDX-License-Identifier: MIT
"""V15-native role-aware shearlet overlay at scorer-support granularity.

G49's first analytic wire deliberately removed ``BoundaryShearletAtomV1.role``
and painted an RGB residual after a legacy scorer-grid render.  That is not the
same coordinate as V15: V15 consumes the role before paint, then applies its
counted realization profile and scorer-solved templates at camera resolution.

This module keeps the immutable V15 ``P`` archive and a counted role-aware
operand separate.  Decode constructs only an ephemeral ``P+A`` receiver by
adding the operand atoms to ``CarrierComposeReceiverV1.boundary_shearlets``.
Both the base and ephemeral receiver execute the donor's
``render_camera_pairs`` method.  The result copies all donor-mutated camera
taps for each frame/scorer-row/scorer-column/channel support containing any
changed tap.  Ownership is support/channel separable.  Copying the whole native
support is deliberately conservative: exact-integer numerator cancellation can
still change the frozen Torch float32 bilinear result through accumulation
roundoff.  This donor-reference policy therefore proves:

* selected Torch float32 scorer inputs are bit-equal to ephemeral ``P+A``;
* unselected Torch inputs and camera bytes equal immutable ``P``; and
* every camera byte outside the exact changed support/channel mask is preserved.

This is support-minimal donor-copy, not a minimum-byte local preimage solve.
No scorer model, pose proof, target plane, label table, dense residual, score,
candidate, public runtime, or n600 authority is part of this bounded primitive.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

import numpy as np

from tac.optimization.direct_description_carrier_compose import (
    _ROLE_TO_WIRE,
    RECEIVER_SCHEMA_V6,
    BoundaryShearletAtomV1,
    CarrierComposeReceiverV1,
    DirectDescriptionError,
    _decode_boundary_shearlet_atoms,
    _encode_boundary_shearlet_atoms,
    receive_carrier_compose_archive,
)
from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    Uint8LatticeError,
)
from tac.witness_dsl.c0b_semantic_quotient import exact_resize_round_u8
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    SelectedPreimageFrameSelectorV1,
)

OPERAND_MAGIC: Final = b"G74RA1\x00\x00"
OPERAND_VERSION: Final = 1
_OPERAND_HEADER: Final = struct.Struct(">8sBBI")
_SELECTOR_TO_WIRE: Final = {
    SelectedPreimageFrameSelectorV1.Y0: 0,
    SelectedPreimageFrameSelectorV1.Y1: 1,
    SelectedPreimageFrameSelectorV1.BOTH: 2,
}
_WIRE_TO_SELECTOR: Final = {value: key for key, value in _SELECTOR_TO_WIRE.items()}

RECEIPT_SCHEMA: Final = "tac.g74_v15_roleaware_overlay_decode_receipt.v1"
DECODER_CONTRACT_ID: Final = "tac.g74.v15_roleaware_native_support_overlay.v1"
DONOR_TAP_COPY_POLICY_ID: Final = "v15_native_donor_tap_copy_per_changed_support.v1"
CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
SCORER_HEIGHT: Final = 384
SCORER_WIDTH: Final = 512
CHANNELS: Final = 3


class V15RoleAwareOverlayError(ValueError):
    """Malformed operand, custody mismatch, or failed exact overlay proof."""


class V15OverlayFrameV1(StrEnum):
    """Array frame indices corresponding to chronological scorer members."""

    Y0 = "Y0"
    Y1 = "Y1"

    @property
    def index(self) -> int:
        return 0 if self is V15OverlayFrameV1.Y0 else 1


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return _sha256(memoryview(contiguous).cast("B"))


def _require_sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V15RoleAwareOverlayError(f"{field} must be canonical lowercase SHA-256")
    return value


def _require_exact_int(
    value: object,
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise V15RoleAwareOverlayError(f"{field} must be an exact integer in [{minimum},{maximum}]")
    return value


def _immutable_array(value: np.ndarray, *, dtype: np.dtype[object]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != dtype:
        raise V15RoleAwareOverlayError(f"array dtype must be exactly {dtype}")
    copied = np.ascontiguousarray(raw).copy()
    copied.setflags(write=False)
    return copied


def _atom_key(atom: BoundaryShearletAtomV1) -> tuple[int, int, int, int]:
    """Return the donor G2SH collision/canonical-order key."""

    return (
        atom.pair_index,
        _ROLE_TO_WIRE[atom.role],
        atom.center_y,
        atom.center_x,
    )


@dataclass(frozen=True, slots=True)
class RoleAwareBoundaryShearletOperandV1:
    """One counted frame selector plus the donor-exact role-aware G2SH bytes."""

    frame_selector: SelectedPreimageFrameSelectorV1
    atoms: tuple[BoundaryShearletAtomV1, ...]

    def __post_init__(self) -> None:
        if type(self.frame_selector) is not SelectedPreimageFrameSelectorV1:
            raise V15RoleAwareOverlayError("frame selector changed exact closed enum type")
        if type(self.atoms) is not tuple or not self.atoms:
            raise V15RoleAwareOverlayError("role-aware operand requires a nonempty exact atom tuple")
        if any(type(atom) is not BoundaryShearletAtomV1 for atom in self.atoms):
            raise V15RoleAwareOverlayError("role-aware operand atom changed exact donor type")
        try:
            encoded = _encode_boundary_shearlet_atoms(self.atoms)
        except DirectDescriptionError as exc:
            raise V15RoleAwareOverlayError("role-aware atoms conflict or are not in donor canonical order") from exc
        if _decode_boundary_shearlet_atoms(encoded) != self.atoms:
            raise V15RoleAwareOverlayError("role-aware atom donor round trip changed values")

    def to_bytes(self) -> bytes:
        try:
            body = _encode_boundary_shearlet_atoms(self.atoms)
        except DirectDescriptionError as exc:
            raise V15RoleAwareOverlayError("cannot encode role-aware donor atoms") from exc
        return (
            _OPERAND_HEADER.pack(
                OPERAND_MAGIC,
                OPERAND_VERSION,
                _SELECTOR_TO_WIRE[self.frame_selector],
                len(body),
            )
            + body
        )

    @property
    def sha256(self) -> str:
        return _sha256(self.to_bytes())


def parse_role_aware_boundary_shearlet_operand(
    payload: bytes,
    *,
    expected_sha256: str | None = None,
    maximum_operand_bytes: int | None = None,
) -> RoleAwareBoundaryShearletOperandV1:
    """Strictly parse and re-emit one counted role-aware operand."""

    if type(payload) is not bytes:
        raise V15RoleAwareOverlayError("role-aware operand must be exact bytes")
    if maximum_operand_bytes is not None:
        limit = _require_exact_int(
            maximum_operand_bytes,
            "maximum_operand_bytes",
            minimum=_OPERAND_HEADER.size + 1,
            maximum=(1 << 32) - 1,
        )
        if len(payload) > limit:
            raise V15RoleAwareOverlayError("role-aware operand exceeds caller byte ceiling")
    if len(payload) < _OPERAND_HEADER.size:
        raise V15RoleAwareOverlayError("role-aware operand is truncated")
    magic, version, selector_wire, body_bytes = _OPERAND_HEADER.unpack_from(payload)
    if magic != OPERAND_MAGIC or version != OPERAND_VERSION:
        raise V15RoleAwareOverlayError("role-aware operand magic/version mismatch")
    if selector_wire not in _WIRE_TO_SELECTOR:
        raise V15RoleAwareOverlayError("role-aware operand frame selector is unknown")
    if body_bytes < 1 or len(payload) != _OPERAND_HEADER.size + body_bytes:
        raise V15RoleAwareOverlayError("role-aware operand body length/EOF mismatch")
    if expected_sha256 is not None and _sha256(payload) != _require_sha256(
        expected_sha256,
        "expected_sha256",
    ):
        raise V15RoleAwareOverlayError("role-aware operand SHA-256 mismatch")
    try:
        atoms = _decode_boundary_shearlet_atoms(payload[_OPERAND_HEADER.size :])
    except DirectDescriptionError as exc:
        raise V15RoleAwareOverlayError("role-aware donor atom parse refused payload") from exc
    result = RoleAwareBoundaryShearletOperandV1(
        frame_selector=_WIRE_TO_SELECTOR[selector_wire],
        atoms=atoms,
    )
    if result.to_bytes() != payload:
        raise V15RoleAwareOverlayError("role-aware operand parse/re-encode changed bytes")
    return result


def resolve_source_pair_ids(
    local_pair_ids: tuple[int, ...],
    *,
    source_pair_start: int,
    pair_count: int,
) -> tuple[int, ...]:
    """Map receiver-local pair IDs to globally addressed donor atom IDs."""

    _require_exact_int(source_pair_start, "source_pair_start", minimum=0, maximum=599)
    _require_exact_int(pair_count, "pair_count", minimum=1, maximum=600)
    if type(local_pair_ids) is not tuple or not local_pair_ids:
        raise V15RoleAwareOverlayError("local pair IDs must be one nonempty exact tuple")
    if any(type(value) is not int for value in local_pair_ids):
        raise V15RoleAwareOverlayError("local pair IDs must contain exact integers")
    if local_pair_ids != tuple(sorted(set(local_pair_ids))):
        raise V15RoleAwareOverlayError("local pair IDs must be unique canonical order")
    if any(value < 0 or value >= pair_count for value in local_pair_ids):
        raise V15RoleAwareOverlayError("local pair ID escaped receiver window")
    source_ids = tuple(source_pair_start + value for value in local_pair_ids)
    if source_ids[-1] >= 600:
        raise V15RoleAwareOverlayError("resolved source pair ID escaped [0,600)")
    return source_ids


@runtime_checkable
class V15ScorerSupportReplacementPolicyV1(Protocol):
    """Seam for exact local 2x2/channel preimage policies."""

    policy_id: str

    def replace_changed_supports(
        self,
        *,
        base_frame: np.ndarray,
        native_mutated_frame: np.ndarray,
        changed_support_values: np.ndarray,
        operator: DisjointResizeOperator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return camera frame and owned camera-value mask."""


@dataclass(frozen=True, slots=True)
class DonorTapCopyPolicyV1:
    """Copy every native mutated tap in each changed support/channel."""

    policy_id: str = DONOR_TAP_COPY_POLICY_ID

    def replace_changed_supports(
        self,
        *,
        base_frame: np.ndarray,
        native_mutated_frame: np.ndarray,
        changed_support_values: np.ndarray,
        operator: DisjointResizeOperator,
    ) -> tuple[np.ndarray, np.ndarray]:
        base = np.asarray(base_frame)
        mutated = np.asarray(native_mutated_frame)
        changed = np.asarray(changed_support_values)
        expected_frame_shape = (operator.camera_h, operator.camera_w, CHANNELS)
        expected_scorer_shape = (operator.scorer_h, operator.scorer_w, CHANNELS)
        if (
            base.dtype != np.uint8
            or mutated.dtype != np.uint8
            or base.shape != expected_frame_shape
            or mutated.shape != expected_frame_shape
        ):
            raise V15RoleAwareOverlayError("support policy requires exact uint8 camera frames")
        if changed.dtype.kind != "b" or changed.shape != expected_scorer_shape:
            raise V15RoleAwareOverlayError("support policy requires one exact scorer-value mask")
        owned = _camera_owner_mask_for_support_values(operator, changed)
        output = base.copy()
        output[owned] = mutated[owned]
        return np.ascontiguousarray(output), np.ascontiguousarray(owned)


@dataclass(frozen=True, slots=True)
class V15RoleAwareOverlayReceiptV1:
    schema: str
    decoder_contract_id: str
    replacement_policy_id: str
    base_archive_bytes: int
    base_archive_sha256: str
    operand_bytes: int
    operand_sha256: str
    operand_parse_reencode_identical: bool
    operand_atom_count: int
    operand_roles: tuple[str, ...]
    frame_selector: str
    local_pair_ids: tuple[int, ...]
    source_pair_ids: tuple[int, ...]
    receiver_schema: str
    realization_profile_consumed: bool
    scorer_solved_template_count: int
    legacy_render_pairs_used: bool
    ephemeral_receiver_only: bool
    ephemeral_receiver_archive_claim: bool
    changed_support_cells_per_frame: tuple[int, ...]
    changed_support_values_per_frame: tuple[int, ...]
    changed_integer_numerator_values_per_frame: tuple[int, ...]
    fractional_changed_numerators_per_frame: tuple[int, ...]
    maximum_fractional_distance_to_u8_per_frame: tuple[float, ...]
    owned_camera_values_per_frame: tuple[int, ...]
    actually_changed_camera_values_per_frame: tuple[int, ...]
    preserved_unowned_camera_values: int
    unchanged_scorer_numerator_values: int
    exact_resize_denominator: int
    selected_frames_match_native_mutated_numerators: bool
    selected_frames_match_native_torch_bilinear: bool
    unselected_frames_byte_identical_to_base: bool
    torch_version: str
    cross_host_torch_parity_claim: bool
    decoder_source_sha256: str
    base_integer_numerators_sha256: str
    native_mutated_integer_numerators_sha256: str
    output_integer_numerators_sha256: str
    native_mutated_torch_bilinear_sha256: str
    output_torch_bilinear_sha256: str
    changed_support_values_sha256: str
    changed_integer_numerator_values_sha256: str
    owned_camera_values_sha256: str
    output_camera_sha256: str
    output_rounded_u8_diagnostic_sha256: str
    deterministic_double_decode: bool
    scorer_invoked: bool
    pose_invoked: bool
    pose_preservation_claim: bool
    score_claim: bool
    candidate_claim: bool
    public_runtime_claim: bool
    n600_evidence_claim: bool
    research_only: bool

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA or self.decoder_contract_id != DECODER_CONTRACT_ID:
            raise V15RoleAwareOverlayError("receipt schema/decoder contract mismatch")
        if self.replacement_policy_id != DONOR_TAP_COPY_POLICY_ID:
            raise V15RoleAwareOverlayError("receipt replacement policy is not the sealed G74 policy")
        _require_exact_int(
            self.base_archive_bytes,
            "receipt.base_archive_bytes",
            minimum=1,
            maximum=(1 << 32) - 1,
        )
        _require_exact_int(
            self.operand_bytes,
            "receipt.operand_bytes",
            minimum=_OPERAND_HEADER.size + 1,
            maximum=(1 << 32) - 1,
        )
        _require_exact_int(
            self.operand_atom_count,
            "receipt.operand_atom_count",
            minimum=1,
            maximum=(1 << 32) - 1,
        )
        for field_name in (
            "base_archive_sha256",
            "operand_sha256",
            "decoder_source_sha256",
            "base_integer_numerators_sha256",
            "native_mutated_integer_numerators_sha256",
            "output_integer_numerators_sha256",
            "native_mutated_torch_bilinear_sha256",
            "output_torch_bilinear_sha256",
            "changed_support_values_sha256",
            "changed_integer_numerator_values_sha256",
            "owned_camera_values_sha256",
            "output_camera_sha256",
            "output_rounded_u8_diagnostic_sha256",
        ):
            _require_sha256(getattr(self, field_name), f"receipt.{field_name}")
        if (
            type(self.local_pair_ids) is not tuple
            or not self.local_pair_ids
            or type(self.source_pair_ids) is not tuple
            or len(self.source_pair_ids) != len(self.local_pair_ids)
        ):
            raise V15RoleAwareOverlayError("receipt pair IDs changed exact tuple shape")
        if type(self.operand_roles) is not tuple or not self.operand_roles:
            raise V15RoleAwareOverlayError("receipt operand roles changed exact tuple shape")
        if type(self.torch_version) is not str or not self.torch_version:
            raise V15RoleAwareOverlayError("receipt Torch version must be one nonempty string")
        frame_values = len(self.local_pair_ids) * 2
        integer_tuple_fields = (
            "changed_support_cells_per_frame",
            "changed_support_values_per_frame",
            "changed_integer_numerator_values_per_frame",
            "fractional_changed_numerators_per_frame",
            "owned_camera_values_per_frame",
            "actually_changed_camera_values_per_frame",
        )
        for field_name in integer_tuple_fields:
            values = getattr(self, field_name)
            if (
                type(values) is not tuple
                or len(values) != frame_values
                or any(type(value) is not int or value < 0 for value in values)
            ):
                raise V15RoleAwareOverlayError(f"receipt.{field_name} changed exact count vector")
        if (
            type(self.maximum_fractional_distance_to_u8_per_frame) is not tuple
            or len(self.maximum_fractional_distance_to_u8_per_frame) != frame_values
            or any(
                type(value) is not float or not 0.0 <= value <= 0.5
                for value in self.maximum_fractional_distance_to_u8_per_frame
            )
        ):
            raise V15RoleAwareOverlayError("receipt fractional-distance vector is malformed")
        for field_name in (
            "preserved_unowned_camera_values",
            "unchanged_scorer_numerator_values",
            "exact_resize_denominator",
            "scorer_solved_template_count",
        ):
            if type(getattr(self, field_name)) is not int or getattr(self, field_name) < 0:
                raise V15RoleAwareOverlayError(f"receipt.{field_name} must be a nonnegative exact int")
        bool_fields = (
            "operand_parse_reencode_identical",
            "realization_profile_consumed",
            "legacy_render_pairs_used",
            "ephemeral_receiver_only",
            "ephemeral_receiver_archive_claim",
            "selected_frames_match_native_mutated_numerators",
            "selected_frames_match_native_torch_bilinear",
            "unselected_frames_byte_identical_to_base",
            "cross_host_torch_parity_claim",
            "deterministic_double_decode",
            "scorer_invoked",
            "pose_invoked",
            "pose_preservation_claim",
            "score_claim",
            "candidate_claim",
            "public_runtime_claim",
            "n600_evidence_claim",
            "research_only",
        )
        if any(type(getattr(self, field_name)) is not bool for field_name in bool_fields):
            raise V15RoleAwareOverlayError("receipt boolean field changed exact bool type")
        if not (
            self.operand_parse_reencode_identical
            and self.realization_profile_consumed
            and not self.legacy_render_pairs_used
            and self.ephemeral_receiver_only
            and not self.ephemeral_receiver_archive_claim
            and self.selected_frames_match_native_mutated_numerators
            and self.selected_frames_match_native_torch_bilinear
            and self.unselected_frames_byte_identical_to_base
            and not self.cross_host_torch_parity_claim
            and self.deterministic_double_decode
            and not self.scorer_invoked
            and not self.pose_invoked
            and not self.pose_preservation_claim
            and not self.score_claim
            and not self.candidate_claim
            and not self.public_runtime_claim
            and not self.n600_evidence_claim
            and self.research_only
        ):
            raise V15RoleAwareOverlayError("receipt changed bounded G74 truth flags")

    def to_bytes(self) -> bytes:
        payload = {
            field_name: (list(value) if isinstance(value := getattr(self, field_name), tuple) else value)
            for field_name in self.__dataclass_fields__
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")


@dataclass(frozen=True, slots=True)
class V15RoleAwareOverlayResultV1:
    camera_pairs: np.ndarray
    rounded_u8_diagnostic_planes: np.ndarray
    changed_support_values: np.ndarray
    changed_integer_numerator_values: np.ndarray
    owned_camera_values: np.ndarray
    receipt: V15RoleAwareOverlayReceiptV1

    def __post_init__(self) -> None:
        if type(self.receipt) is not V15RoleAwareOverlayReceiptV1:
            raise V15RoleAwareOverlayError("result receipt changed exact type")
        pair_count = len(self.receipt.local_pair_ids)
        camera_shape = (pair_count, 2, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
        scorer_shape = (pair_count, 2, SCORER_HEIGHT, SCORER_WIDTH, CHANNELS)
        for value, dtype, shape, field_name in (
            (self.camera_pairs, np.dtype(np.uint8), camera_shape, "camera_pairs"),
            (
                self.rounded_u8_diagnostic_planes,
                np.dtype(np.uint8),
                scorer_shape,
                "rounded_u8_diagnostic_planes",
            ),
            (self.changed_support_values, np.dtype(bool), scorer_shape, "changed_support_values"),
            (
                self.changed_integer_numerator_values,
                np.dtype(bool),
                scorer_shape,
                "changed_integer_numerator_values",
            ),
            (self.owned_camera_values, np.dtype(bool), camera_shape, "owned_camera_values"),
        ):
            raw = np.asarray(value)
            if raw.dtype != dtype or raw.shape != shape:
                raise V15RoleAwareOverlayError(f"result.{field_name} changed exact dtype/shape ABI")
        object.__setattr__(
            self,
            "camera_pairs",
            _immutable_array(self.camera_pairs, dtype=np.dtype(np.uint8)),
        )
        object.__setattr__(
            self,
            "rounded_u8_diagnostic_planes",
            _immutable_array(self.rounded_u8_diagnostic_planes, dtype=np.dtype(np.uint8)),
        )
        object.__setattr__(
            self,
            "changed_support_values",
            _immutable_array(self.changed_support_values, dtype=np.dtype(bool)),
        )
        object.__setattr__(
            self,
            "changed_integer_numerator_values",
            _immutable_array(self.changed_integer_numerator_values, dtype=np.dtype(bool)),
        )
        object.__setattr__(
            self,
            "owned_camera_values",
            _immutable_array(self.owned_camera_values, dtype=np.dtype(bool)),
        )
        if _array_sha256(self.camera_pairs) != self.receipt.output_camera_sha256:
            raise V15RoleAwareOverlayError("result camera bytes do not match receipt hash")
        if _array_sha256(self.rounded_u8_diagnostic_planes) != self.receipt.output_rounded_u8_diagnostic_sha256:
            raise V15RoleAwareOverlayError("result rounded diagnostic does not match receipt hash")
        if (
            _array_sha256(self.changed_support_values) != self.receipt.changed_support_values_sha256
            or _array_sha256(self.changed_integer_numerator_values)
            != self.receipt.changed_integer_numerator_values_sha256
            or _array_sha256(self.owned_camera_values) != self.receipt.owned_camera_values_sha256
        ):
            raise V15RoleAwareOverlayError("result ownership masks do not match receipt hashes")
        if self.receipt.decoder_source_sha256 != decoder_source_sha256():
            raise V15RoleAwareOverlayError("result decoder source bytes do not match receipt hash")
        try:
            operator = DisjointResizeOperator.build(
                camera_h=CAMERA_HEIGHT,
                camera_w=CAMERA_WIDTH,
                scorer_h=SCORER_HEIGHT,
                scorer_w=SCORER_WIDTH,
            )
            output_numerators, denominator = _integer_numerators_for_camera_pairs(
                operator,
                self.camera_pairs,
            )
        except Uint8LatticeError as exc:
            raise V15RoleAwareOverlayError("result exact numerator validation failed") from exc
        if (
            denominator != self.receipt.exact_resize_denominator
            or _array_sha256(output_numerators) != self.receipt.output_integer_numerators_sha256
        ):
            raise V15RoleAwareOverlayError("result exact numerators do not match receipt")
        output_torch = _torch_bilinear_for_camera_pairs(self.camera_pairs)
        if _array_sha256(output_torch) != self.receipt.output_torch_bilinear_sha256:
            raise V15RoleAwareOverlayError("result Torch scorer inputs do not match receipt hash")


_RECEIPT_TUPLE_FIELDS: Final = frozenset(
    {
        "operand_roles",
        "local_pair_ids",
        "source_pair_ids",
        "changed_support_cells_per_frame",
        "changed_support_values_per_frame",
        "changed_integer_numerator_values_per_frame",
        "fractional_changed_numerators_per_frame",
        "maximum_fractional_distance_to_u8_per_frame",
        "owned_camera_values_per_frame",
        "actually_changed_camera_values_per_frame",
    }
)


def parse_v15_role_aware_overlay_receipt(
    payload: bytes,
) -> V15RoleAwareOverlayReceiptV1:
    """Strict canonical JSON parse-back for the bounded proof receipt."""

    if type(payload) is not bytes:
        raise V15RoleAwareOverlayError("receipt payload must be exact bytes")
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V15RoleAwareOverlayError("receipt is not canonical JSON") from exc
    expected_fields = set(V15RoleAwareOverlayReceiptV1.__dataclass_fields__)
    if type(raw) is not dict or set(raw) != expected_fields:
        raise V15RoleAwareOverlayError("receipt field set changed exact schema")
    values = {
        field_name: tuple(value) if field_name in _RECEIPT_TUPLE_FIELDS else value for field_name, value in raw.items()
    }
    try:
        receipt = V15RoleAwareOverlayReceiptV1(**values)
    except TypeError as exc:
        raise V15RoleAwareOverlayError("receipt values changed typed schema") from exc
    if receipt.to_bytes() != payload:
        raise V15RoleAwareOverlayError("receipt parse/re-encode changed bytes")
    return receipt


def _selector_includes(
    selector: SelectedPreimageFrameSelectorV1,
    frame_index: int,
) -> bool:
    return (
        selector is SelectedPreimageFrameSelectorV1.BOTH
        or (selector is SelectedPreimageFrameSelectorV1.Y0 and frame_index == 0)
        or (selector is SelectedPreimageFrameSelectorV1.Y1 and frame_index == 1)
    )


def _fractional_numerator_stats(
    numerators: np.ndarray,
    denominator: int,
    changed: np.ndarray,
) -> tuple[int, float]:
    selected = np.asarray(numerators, dtype=np.int64)[changed]
    if selected.size == 0:
        return 0, 0.0
    remainder = np.mod(selected, denominator)
    fractional = remainder != 0
    distance = np.minimum(remainder, denominator - remainder).astype(np.float64) / denominator
    return int(np.count_nonzero(fractional)), float(np.max(distance))


def _uniform_support_indices(
    operator: DisjointResizeOperator,
) -> tuple[np.ndarray, np.ndarray]:
    row_sizes = {len(support.indices) for support in operator.row_supports}
    col_sizes = {len(support.indices) for support in operator.col_supports}
    if len(row_sizes) != 1 or len(col_sizes) != 1 or 0 in row_sizes or 0 in col_sizes:
        raise V15RoleAwareOverlayError("G74 requires nonempty uniform frozen resize supports")
    return (
        np.asarray([support.indices for support in operator.row_supports], dtype=np.intp),
        np.asarray([support.indices for support in operator.col_supports], dtype=np.intp),
    )


def _support_tap_difference_mask(
    operator: DisjointResizeOperator,
    base_frame: np.ndarray,
    native_mutated_frame: np.ndarray,
) -> np.ndarray:
    """Mark a scorer value if any camera tap in its channel support differs."""

    base = np.asarray(base_frame)
    mutated = np.asarray(native_mutated_frame)
    expected_shape = (operator.camera_h, operator.camera_w, CHANNELS)
    if (
        base.dtype != np.uint8
        or mutated.dtype != np.uint8
        or base.shape != expected_shape
        or mutated.shape != expected_shape
    ):
        raise V15RoleAwareOverlayError("support difference requires exact uint8 camera frames")
    row_indices, col_indices = _uniform_support_indices(operator)
    base_blocks = base[
        row_indices[:, None, :, None],
        col_indices[None, :, None, :],
        :,
    ]
    mutated_blocks = mutated[
        row_indices[:, None, :, None],
        col_indices[None, :, None, :],
        :,
    ]
    return np.ascontiguousarray(np.any(base_blocks != mutated_blocks, axis=(2, 3)))


def _camera_owner_mask_for_support_values(
    operator: DisjointResizeOperator,
    changed_support_values: np.ndarray,
) -> np.ndarray:
    """Expand a scorer support/channel mask to its exact disjoint camera taps."""

    changed = np.asarray(changed_support_values)
    expected_shape = (operator.scorer_h, operator.scorer_w, CHANNELS)
    if changed.dtype != np.dtype(bool) or changed.shape != expected_shape:
        raise V15RoleAwareOverlayError("changed support values changed exact boolean ABI")
    row_indices, col_indices = _uniform_support_indices(operator)
    owned = np.zeros((operator.camera_h, operator.camera_w, CHANNELS), dtype=bool)
    scorer_rows, scorer_cols, channels = np.nonzero(changed)
    for row_offset in range(row_indices.shape[1]):
        for col_offset in range(col_indices.shape[1]):
            camera_rows = row_indices[scorer_rows, row_offset]
            camera_cols = col_indices[scorer_cols, col_offset]
            owned[camera_rows, camera_cols, channels] = True
    return np.ascontiguousarray(owned)


def _torch_bilinear_float32(frame: np.ndarray) -> np.ndarray:
    """Execute the frozen evaluator's CPU Torch float32 bilinear resize."""

    try:
        import torch
        import torch.nn.functional as torch_functional
    except ImportError as exc:  # pragma: no cover - contest environment has Torch
        raise V15RoleAwareOverlayError("CPU Torch is required for scorer-input parity proof") from exc
    raw = np.asarray(frame)
    if raw.dtype != np.uint8 or raw.shape != (CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS):
        raise V15RoleAwareOverlayError("Torch scorer-input proof requires exact uint8 camera frame")
    tensor = torch.from_numpy(np.array(raw, copy=True, order="C")).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    with torch.inference_mode():
        resized = torch_functional.interpolate(
            tensor,
            size=(SCORER_HEIGHT, SCORER_WIDTH),
            mode="bilinear",
        )
    return np.ascontiguousarray(resized[0].permute(1, 2, 0).cpu().numpy())


def _torch_version() -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - contest environment has Torch
        raise V15RoleAwareOverlayError("CPU Torch is required for scorer-input parity proof") from exc
    version = str(torch.__version__)
    if not version:
        raise V15RoleAwareOverlayError("Torch runtime did not expose a version string")
    return version


def _integer_numerators_for_camera_pairs(
    operator: DisjointResizeOperator,
    camera_pairs: np.ndarray,
) -> tuple[np.ndarray, int]:
    numerators: list[np.ndarray] = []
    denominator: int | None = None
    for local_index in range(camera_pairs.shape[0]):
        per_pair: list[np.ndarray] = []
        for frame_index in range(2):
            current, current_denominator = operator.apply_numerators(camera_pairs[local_index, frame_index])
            if denominator is None:
                denominator = current_denominator
            elif denominator != current_denominator:
                raise V15RoleAwareOverlayError("exact resize denominator drifted across frames")
            per_pair.append(current)
        numerators.append(np.stack(per_pair))
    if denominator is None:
        raise V15RoleAwareOverlayError("cannot derive numerators for zero camera pairs")
    return np.ascontiguousarray(np.stack(numerators)), denominator


def _torch_bilinear_for_camera_pairs(camera_pairs: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.stack(
            [
                np.stack([_torch_bilinear_float32(camera_pairs[local_index, frame_index]) for frame_index in range(2)])
                for local_index in range(camera_pairs.shape[0])
            ]
        )
    )


@dataclass(frozen=True, slots=True, init=False)
class V15RoleAwareOverlayDecoderV1:
    """Immutable V15 P custody plus exact-R and support-replacement policy."""

    semantic_archive: bytes
    semantic_archive_sha256: str
    receiver: CarrierComposeReceiverV1
    operator: DisjointResizeOperator
    replacement_policy: V15ScorerSupportReplacementPolicyV1 = DonorTapCopyPolicyV1()
    _receiver_identity: int
    _operator_identity: int
    _policy_identity: int

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("V15RoleAwareOverlayDecoderV1 must be constructed through .open()")

    def _validate_construction_custody(self) -> None:
        if (
            type(self.semantic_archive) is not bytes
            or _sha256(self.semantic_archive) != self.semantic_archive_sha256
            or self.receiver.archive is not self.semantic_archive
            or id(self.receiver) != self._receiver_identity
            or id(self.operator) != self._operator_identity
            or id(self.replacement_policy) != self._policy_identity
        ):
            raise V15RoleAwareOverlayError("sealed V15 decoder construction custody drifted")
        _require_sha256(self.semantic_archive_sha256, "semantic_archive_sha256")
        if type(self.replacement_policy) is not DonorTapCopyPolicyV1:
            raise V15RoleAwareOverlayError("G74 v1 seals the exact donor-tap-copy policy")
        if self.replacement_policy.policy_id != DONOR_TAP_COPY_POLICY_ID:
            raise V15RoleAwareOverlayError("sealed G74 replacement policy ID drifted")
        if (
            self.operator.camera_h,
            self.operator.camera_w,
            self.operator.scorer_h,
            self.operator.scorer_w,
        ) != (CAMERA_HEIGHT, CAMERA_WIDTH, SCORER_HEIGHT, SCORER_WIDTH):
            raise V15RoleAwareOverlayError("sealed exact-R geometry drifted")
        if (
            self.receiver.realization_profile is None
            or self.receiver.scorer_solved_templates is None
            or (self.receiver.custody and self.receiver.custody.get("schema") != RECEIVER_SCHEMA_V6)
        ):
            raise V15RoleAwareOverlayError("sealed V15 receiver semantics drifted")

    @classmethod
    def open(
        cls,
        semantic_archive: bytes,
        *,
        expected_archive_bytes: int,
        expected_archive_sha256: str,
        verify_member_effects: bool = True,
        replacement_policy: V15ScorerSupportReplacementPolicyV1 | None = None,
    ) -> V15RoleAwareOverlayDecoderV1:
        if type(semantic_archive) is not bytes:
            raise V15RoleAwareOverlayError("semantic P must be exact archive bytes")
        expected_bytes = _require_exact_int(
            expected_archive_bytes,
            "expected_archive_bytes",
            minimum=1,
            maximum=(1 << 32) - 1,
        )
        expected_sha = _require_sha256(
            expected_archive_sha256,
            "expected_archive_sha256",
        )
        if len(semantic_archive) != expected_bytes or _sha256(semantic_archive) != expected_sha:
            raise V15RoleAwareOverlayError("semantic P exact bytes/SHA custody mismatch")
        resolved_policy = replacement_policy or DonorTapCopyPolicyV1()
        if type(resolved_policy) is not DonorTapCopyPolicyV1:
            raise V15RoleAwareOverlayError(
                "G74 v1 only admits the sealed donor-tap-copy policy; future solves need a new version"
            )
        try:
            receiver = receive_carrier_compose_archive(
                semantic_archive,
                verify_member_effects=verify_member_effects,
            )
        except (DirectDescriptionError, OSError, ValueError) as exc:
            raise V15RoleAwareOverlayError("strict V15 P reopen refused archive") from exc
        if (
            receiver.realization_profile is None
            or receiver.scorer_solved_templates is None
            or (receiver.custody and receiver.custody.get("schema") != RECEIVER_SCHEMA_V6)
        ):
            raise V15RoleAwareOverlayError("semantic P is not the V15 camera-realization/template receiver")
        if receiver.archive is not semantic_archive:
            raise V15RoleAwareOverlayError("V15 receiver did not retain exact semantic P bytes")
        try:
            operator = DisjointResizeOperator.build(
                camera_h=CAMERA_HEIGHT,
                camera_w=CAMERA_WIDTH,
                scorer_h=SCORER_HEIGHT,
                scorer_w=SCORER_WIDTH,
            )
        except Uint8LatticeError as exc:
            raise V15RoleAwareOverlayError("frozen exact-R geometry is unavailable") from exc
        instance = object.__new__(cls)
        object.__setattr__(instance, "semantic_archive", semantic_archive)
        object.__setattr__(instance, "semantic_archive_sha256", expected_sha)
        object.__setattr__(instance, "receiver", receiver)
        object.__setattr__(instance, "operator", operator)
        object.__setattr__(instance, "replacement_policy", resolved_policy)
        object.__setattr__(instance, "_receiver_identity", id(receiver))
        object.__setattr__(instance, "_operator_identity", id(operator))
        object.__setattr__(instance, "_policy_identity", id(resolved_policy))
        instance._validate_construction_custody()
        return instance

    def _validate_operand_and_pairs(
        self,
        operand: RoleAwareBoundaryShearletOperandV1,
        local_pair_ids: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[BoundaryShearletAtomV1, ...]]:
        self._validate_construction_custody()
        source_start = self.receiver.predictor.source_pair_start
        pair_count = self.receiver.z.n_pairs
        source_ids = resolve_source_pair_ids(
            local_pair_ids,
            source_pair_start=source_start,
            pair_count=pair_count,
        )
        source_stop = source_start + pair_count
        if any(atom.pair_index < source_start or atom.pair_index >= source_stop for atom in operand.atoms):
            raise V15RoleAwareOverlayError("role-aware atom escaped semantic P source window")
        existing_by_key = {_atom_key(atom) for atom in self.receiver.boundary_shearlets}
        new_keys = {_atom_key(atom) for atom in operand.atoms}
        if existing_by_key.intersection(new_keys):
            raise V15RoleAwareOverlayError("role-aware atom conflicts with immutable P G2SH address")
        combined = tuple(
            sorted(
                (*self.receiver.boundary_shearlets, *operand.atoms),
                key=_atom_key,
            )
        )
        try:
            _encode_boundary_shearlet_atoms(combined)
        except DirectDescriptionError as exc:
            raise V15RoleAwareOverlayError("combined immutable-P plus operand G2SH addresses conflict") from exc
        return source_ids, combined

    def _decode_once(
        self,
        operand_payload: bytes,
        *,
        expected_operand_sha256: str,
        maximum_operand_bytes: int,
        local_pair_ids: tuple[int, ...],
    ) -> V15RoleAwareOverlayResultV1:
        operand = parse_role_aware_boundary_shearlet_operand(
            operand_payload,
            expected_sha256=expected_operand_sha256,
            maximum_operand_bytes=maximum_operand_bytes,
        )
        source_ids, combined = self._validate_operand_and_pairs(
            operand,
            local_pair_ids,
        )
        # This object is execution state only.  Its ``archive`` still names P
        # and is deliberately never exposed as mutated archive custody.
        ephemeral = replace(self.receiver, boundary_shearlets=combined)
        try:
            base_camera = self.receiver.render_camera_pairs(local_pair_ids)
            native_mutated_camera = ephemeral.render_camera_pairs(local_pair_ids)
        except (DirectDescriptionError, ValueError) as exc:
            raise V15RoleAwareOverlayError("V15-native base/mutated camera realization failed") from exc
        if (
            base_camera.dtype != np.uint8
            or native_mutated_camera.dtype != np.uint8
            or base_camera.shape != (len(local_pair_ids), 2, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
            or native_mutated_camera.shape != base_camera.shape
        ):
            raise V15RoleAwareOverlayError("V15 camera realization changed exact ABI")

        output = base_camera.copy()
        owned = np.zeros(base_camera.shape, dtype=bool)
        changed_support = np.zeros(
            (len(local_pair_ids), 2, SCORER_HEIGHT, SCORER_WIDTH, CHANNELS),
            dtype=bool,
        )
        changed_integer_numerator = np.zeros(
            (len(local_pair_ids), 2, SCORER_HEIGHT, SCORER_WIDTH, CHANNELS),
            dtype=bool,
        )
        output_rounded_diagnostic = np.empty(
            (len(local_pair_ids), 2, SCORER_HEIGHT, SCORER_WIDTH, CHANNELS),
            dtype=np.uint8,
        )
        changed_support_cells_per_frame: list[int] = []
        changed_support_values_per_frame: list[int] = []
        changed_integer_values_per_frame: list[int] = []
        fractional_per_frame: list[int] = []
        maximum_fractional_per_frame: list[float] = []
        owned_values_per_frame: list[int] = []
        actually_changed_per_frame: list[int] = []
        unchanged_numerator_values = 0
        preserved_unowned_values = 0
        selected_native_exact = True
        selected_native_torch_exact = True
        unselected_base_exact = True
        common_denominator: int | None = None

        for local_index in range(len(local_pair_ids)):
            for frame_index in range(2):
                base_frame = base_camera[local_index, frame_index]
                mutated_frame = native_mutated_camera[local_index, frame_index]
                try:
                    base_num, denominator = self.operator.apply_numerators(base_frame)
                    mutated_num, mutated_denominator = self.operator.apply_numerators(mutated_frame)
                except Uint8LatticeError as exc:
                    raise V15RoleAwareOverlayError("exact-R numerator derivation refused V15 camera") from exc
                if denominator != mutated_denominator:
                    raise V15RoleAwareOverlayError("base/mutated exact-R denominator drifted")
                if common_denominator is None:
                    common_denominator = denominator
                elif common_denominator != denominator:
                    raise V15RoleAwareOverlayError("exact-R denominator drifted across frames")
                native_integer_changed = base_num != mutated_num
                native_support_changed = _support_tap_difference_mask(
                    self.operator,
                    base_frame,
                    mutated_frame,
                )
                selected = _selector_includes(operand.frame_selector, frame_index)
                selected_support_changed = (
                    native_support_changed if selected else np.zeros(native_support_changed.shape, dtype=bool)
                )
                selected_integer_changed = (
                    native_integer_changed if selected else np.zeros(native_integer_changed.shape, dtype=bool)
                )
                changed_support[local_index, frame_index] = selected_support_changed
                changed_integer_numerator[local_index, frame_index] = selected_integer_changed
                if selected:
                    replaced, frame_owned = self.replacement_policy.replace_changed_supports(
                        base_frame=base_frame,
                        native_mutated_frame=mutated_frame,
                        changed_support_values=selected_support_changed,
                        operator=self.operator,
                    )
                    expected_owned = _camera_owner_mask_for_support_values(
                        self.operator,
                        selected_support_changed,
                    )
                    if (
                        replaced.dtype != np.uint8
                        or replaced.shape != base_frame.shape
                        or frame_owned.dtype != np.dtype(bool)
                        or frame_owned.shape != base_frame.shape
                        or not np.array_equal(frame_owned, expected_owned)
                    ):
                        raise V15RoleAwareOverlayError(
                            "replacement policy ownership differs from exact changed supports"
                        )
                    if not np.array_equal(replaced[~expected_owned], base_frame[~expected_owned]):
                        raise V15RoleAwareOverlayError(
                            "replacement policy changed camera bytes outside exact ownership"
                        )
                    output[local_index, frame_index] = replaced
                    owned[local_index, frame_index] = frame_owned
                result_frame = output[local_index, frame_index]
                try:
                    result_num, result_denominator = self.operator.apply_numerators(result_frame)
                    result_scorer = exact_resize_round_u8(
                        self.operator,
                        result_frame,
                    )
                except (Uint8LatticeError, ValueError) as exc:
                    raise V15RoleAwareOverlayError("support overlay failed exact-R derivation") from exc
                if result_denominator != denominator:
                    raise V15RoleAwareOverlayError("support overlay denominator drifted")
                expected_num = mutated_num if selected else base_num
                if not np.array_equal(result_num, expected_num):
                    raise V15RoleAwareOverlayError("support overlay did not reproduce selected native numerators")
                output_rounded_diagnostic[local_index, frame_index] = result_scorer
                selected_native_exact &= bool(np.array_equal(result_num, expected_num))
                result_torch = _torch_bilinear_float32(result_frame)
                expected_torch = _torch_bilinear_float32(mutated_frame if selected else base_frame)
                if not np.array_equal(result_torch, expected_torch):
                    raise V15RoleAwareOverlayError(
                        "support overlay did not reproduce frozen Torch float32 scorer input"
                    )
                selected_native_torch_exact &= bool(np.array_equal(result_torch, expected_torch))
                if not selected:
                    unselected_base_exact &= bool(np.array_equal(result_frame, base_frame))
                unchanged_mask = ~selected_support_changed
                if not np.array_equal(result_num[unchanged_mask], base_num[unchanged_mask]):
                    raise V15RoleAwareOverlayError("support overlay changed an unowned scorer numerator")
                if not np.array_equal(
                    result_frame[~owned[local_index, frame_index]],
                    base_frame[~owned[local_index, frame_index]],
                ):
                    raise V15RoleAwareOverlayError("support overlay changed an unowned camera value")
                fractional_count, fractional_max = _fractional_numerator_stats(
                    mutated_num,
                    denominator,
                    selected_support_changed,
                )
                changed_support_cells_per_frame.append(int(np.count_nonzero(np.any(selected_support_changed, axis=2))))
                changed_support_values_per_frame.append(int(np.count_nonzero(selected_support_changed)))
                changed_integer_values_per_frame.append(int(np.count_nonzero(selected_integer_changed)))
                fractional_per_frame.append(fractional_count)
                maximum_fractional_per_frame.append(fractional_max)
                owned_values_per_frame.append(int(np.count_nonzero(owned[local_index, frame_index])))
                actually_changed_per_frame.append(int(np.count_nonzero(result_frame != base_frame)))
                unchanged_numerator_values += int(np.count_nonzero(unchanged_mask))
                preserved_unowned_values += int(np.count_nonzero(~owned[local_index, frame_index]))

        requested_operand_atoms = tuple(atom for atom in operand.atoms if atom.pair_index in set(source_ids))
        if requested_operand_atoms and not np.any(changed_support):
            raise V15RoleAwareOverlayError(
                "requested role-aware atoms had zero native support effect on selected frames"
            )
        if requested_operand_atoms and not np.any(output != base_camera):
            raise V15RoleAwareOverlayError("requested role-aware support overlay was camera-inert")
        if common_denominator is None:
            raise V15RoleAwareOverlayError("decode produced no exact resize denominator")

        base_numerators, base_denominator = _integer_numerators_for_camera_pairs(
            self.operator,
            base_camera,
        )
        native_mutated_numerators, native_denominator = _integer_numerators_for_camera_pairs(
            self.operator,
            native_mutated_camera,
        )
        output_numerators, output_denominator = _integer_numerators_for_camera_pairs(
            self.operator,
            output,
        )
        if (
            len(
                {
                    base_denominator,
                    native_denominator,
                    output_denominator,
                    common_denominator,
                }
            )
            != 1
        ):
            raise V15RoleAwareOverlayError("receipt numerator denominator binding drifted")
        native_mutated_torch = _torch_bilinear_for_camera_pairs(native_mutated_camera)
        output_torch = _torch_bilinear_for_camera_pairs(output)
        receipt = V15RoleAwareOverlayReceiptV1(
            schema=RECEIPT_SCHEMA,
            decoder_contract_id=DECODER_CONTRACT_ID,
            replacement_policy_id=self.replacement_policy.policy_id,
            base_archive_bytes=len(self.semantic_archive),
            base_archive_sha256=self.semantic_archive_sha256,
            operand_bytes=len(operand_payload),
            operand_sha256=_sha256(operand_payload),
            operand_parse_reencode_identical=operand.to_bytes() == operand_payload,
            operand_atom_count=len(operand.atoms),
            operand_roles=tuple(sorted({atom.role for atom in operand.atoms})),
            frame_selector=operand.frame_selector.value,
            local_pair_ids=local_pair_ids,
            source_pair_ids=source_ids,
            receiver_schema=RECEIVER_SCHEMA_V6,
            realization_profile_consumed=self.receiver.realization_profile is not None,
            scorer_solved_template_count=len(self.receiver.scorer_solved_templates.templates),
            legacy_render_pairs_used=False,
            ephemeral_receiver_only=True,
            ephemeral_receiver_archive_claim=False,
            changed_support_cells_per_frame=tuple(changed_support_cells_per_frame),
            changed_support_values_per_frame=tuple(changed_support_values_per_frame),
            changed_integer_numerator_values_per_frame=tuple(changed_integer_values_per_frame),
            fractional_changed_numerators_per_frame=tuple(fractional_per_frame),
            maximum_fractional_distance_to_u8_per_frame=tuple(maximum_fractional_per_frame),
            owned_camera_values_per_frame=tuple(owned_values_per_frame),
            actually_changed_camera_values_per_frame=tuple(actually_changed_per_frame),
            preserved_unowned_camera_values=preserved_unowned_values,
            unchanged_scorer_numerator_values=unchanged_numerator_values,
            exact_resize_denominator=common_denominator,
            selected_frames_match_native_mutated_numerators=selected_native_exact,
            selected_frames_match_native_torch_bilinear=selected_native_torch_exact,
            unselected_frames_byte_identical_to_base=unselected_base_exact,
            torch_version=_torch_version(),
            cross_host_torch_parity_claim=False,
            decoder_source_sha256=decoder_source_sha256(),
            base_integer_numerators_sha256=_array_sha256(base_numerators),
            native_mutated_integer_numerators_sha256=_array_sha256(native_mutated_numerators),
            output_integer_numerators_sha256=_array_sha256(output_numerators),
            native_mutated_torch_bilinear_sha256=_array_sha256(native_mutated_torch),
            output_torch_bilinear_sha256=_array_sha256(output_torch),
            changed_support_values_sha256=_array_sha256(changed_support),
            changed_integer_numerator_values_sha256=_array_sha256(changed_integer_numerator),
            owned_camera_values_sha256=_array_sha256(owned),
            output_camera_sha256=_array_sha256(output),
            output_rounded_u8_diagnostic_sha256=_array_sha256(output_rounded_diagnostic),
            deterministic_double_decode=True,
            scorer_invoked=False,
            pose_invoked=False,
            pose_preservation_claim=False,
            score_claim=False,
            candidate_claim=False,
            public_runtime_claim=False,
            n600_evidence_claim=False,
            research_only=True,
        )
        receipt = parse_v15_role_aware_overlay_receipt(receipt.to_bytes())
        return V15RoleAwareOverlayResultV1(
            camera_pairs=output,
            rounded_u8_diagnostic_planes=output_rounded_diagnostic,
            changed_support_values=changed_support,
            changed_integer_numerator_values=changed_integer_numerator,
            owned_camera_values=owned,
            receipt=receipt,
        )

    def decode(
        self,
        operand_payload: bytes,
        *,
        expected_operand_sha256: str,
        maximum_operand_bytes: int,
        local_pair_ids: tuple[int, ...],
    ) -> V15RoleAwareOverlayResultV1:
        """Decode twice and return only a byte-identical exact result."""

        first = self._decode_once(
            operand_payload,
            expected_operand_sha256=expected_operand_sha256,
            maximum_operand_bytes=maximum_operand_bytes,
            local_pair_ids=local_pair_ids,
        )
        second = self._decode_once(
            operand_payload,
            expected_operand_sha256=expected_operand_sha256,
            maximum_operand_bytes=maximum_operand_bytes,
            local_pair_ids=local_pair_ids,
        )
        if (
            first.receipt != second.receipt
            or not np.array_equal(first.camera_pairs, second.camera_pairs)
            or not np.array_equal(
                first.rounded_u8_diagnostic_planes,
                second.rounded_u8_diagnostic_planes,
            )
            or not np.array_equal(
                first.changed_support_values,
                second.changed_support_values,
            )
            or not np.array_equal(
                first.changed_integer_numerator_values,
                second.changed_integer_numerator_values,
            )
            or not np.array_equal(
                first.owned_camera_values,
                second.owned_camera_values,
            )
        ):
            raise V15RoleAwareOverlayError("role-aware V15 double decode drifted")
        return first


def audit_v15_legacy_coordinate_mismatch(
    decoder: V15RoleAwareOverlayDecoderV1,
    *,
    local_pair_id: int,
) -> dict[str, object]:
    """Bounded rounded-u8 diagnostic showing legacy/factor2 are not V15."""

    pair_id = _require_exact_int(
        local_pair_id,
        "local_pair_id",
        minimum=0,
        maximum=decoder.receiver.z.n_pairs - 1,
    )
    legacy = decoder.receiver.render_pairs((pair_id,))
    camera = decoder.receiver.render_camera_pairs((pair_id,))
    legacy_vs_native_values: list[int] = []
    legacy_vs_native_cells: list[int] = []
    legacy_vs_native_max_abs: list[int] = []
    factor2_vs_native_camera_values: list[int] = []
    factor2_zero_camera_values: list[int] = []
    native_zero_camera_values: list[int] = []
    for frame_index in range(2):
        native_rounded_diagnostic = exact_resize_round_u8(
            decoder.operator,
            camera[0, frame_index],
        )
        difference = legacy[0, frame_index] != native_rounded_diagnostic
        factor2 = decoder.operator.realize_factor2_uint8(native_rounded_diagnostic)
        legacy_vs_native_values.append(int(np.count_nonzero(difference)))
        legacy_vs_native_cells.append(int(np.count_nonzero(np.any(difference, axis=2))))
        legacy_vs_native_max_abs.append(
            int(np.max(np.abs(legacy[0, frame_index].astype(np.int16) - native_rounded_diagnostic.astype(np.int16))))
        )
        factor2_vs_native_camera_values.append(int(np.count_nonzero(factor2 != camera[0, frame_index])))
        factor2_zero_camera_values.append(int(np.count_nonzero(factor2 == 0)))
        native_zero_camera_values.append(int(np.count_nonzero(camera[0, frame_index] == 0)))
    return {
        "schema": "tac.g74_v15_legacy_coordinate_mismatch_audit.v1",
        "local_pair_id": pair_id,
        "source_pair_id": decoder.receiver.predictor.source_pair_start + pair_id,
        "legacy_vs_native_rounded_u8_changed_values_per_frame": legacy_vs_native_values,
        "legacy_vs_native_rounded_u8_changed_cells_per_frame": legacy_vs_native_cells,
        "legacy_vs_native_rounded_u8_max_abs_per_frame": legacy_vs_native_max_abs,
        "whole_factor2_vs_native_camera_changed_values_per_frame": (factor2_vs_native_camera_values),
        "whole_factor2_zero_camera_values_per_frame": factor2_zero_camera_values,
        "native_v15_zero_camera_values_per_frame": native_zero_camera_values,
        "diagnostic_only": True,
        "score_claim": False,
    }


def decoder_source_sha256() -> str:
    """Bind receipts to the exact generic receiver implementation bytes."""

    return _sha256(Path(__file__).read_bytes())


__all__ = [
    "CAMERA_HEIGHT",
    "CAMERA_WIDTH",
    "CHANNELS",
    "DECODER_CONTRACT_ID",
    "DONOR_TAP_COPY_POLICY_ID",
    "OPERAND_MAGIC",
    "OPERAND_VERSION",
    "RECEIPT_SCHEMA",
    "SCORER_HEIGHT",
    "SCORER_WIDTH",
    "DonorTapCopyPolicyV1",
    "RoleAwareBoundaryShearletOperandV1",
    "V15OverlayFrameV1",
    "V15RoleAwareOverlayDecoderV1",
    "V15RoleAwareOverlayError",
    "V15RoleAwareOverlayReceiptV1",
    "V15RoleAwareOverlayResultV1",
    "V15ScorerSupportReplacementPolicyV1",
    "audit_v15_legacy_coordinate_mismatch",
    "decoder_source_sha256",
    "parse_role_aware_boundary_shearlet_operand",
    "parse_v15_role_aware_overlay_receipt",
    "resolve_source_pair_ids",
]
