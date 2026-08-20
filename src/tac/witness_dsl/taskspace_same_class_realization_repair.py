# SPDX-License-Identifier: MIT
"""Counted same-class camera-realization repair subgrammar (G8R1).

Semantic ``G`` labels can already equal the encoder's target labels while the
realized uint8 camera witness still falls on the wrong side of the frozen
scorer after ``R``.  This module supplies the missing, distinct counted
coordinate: canonical horizontal scorer-cell runs whose one shared RGB value
is expanded by the receiver and applied to exact disjoint camera supports.

The receiver deliberately has no target-label or scorer input.  Exact target
labels are admitted only by the encoder and never enter the packet.  Decoding
preserves the current semantic labels bit-for-bit, delegates camera mutation to
``apply_disjoint_camera_cell_reconstruction_v1``, and proves preservation of
all unowned camera bytes and scorer numerators.  It does *not* run a scorer or
claim that through-R target-realization debt is closed.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Final, Literal

import numpy as np

from tac.witness_dsl.generative_taskspace_correction import GenerativeTaskspaceCorrectionError
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CHANNELS,
    MAX_BOUNDED_PAIRS,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    PredictorCameraPairSurfaceV1,
    PredictorPreservingA3Error,
    SparseConstantRGBCellV1,
    apply_disjoint_camera_cell_reconstruction_v1,
    derive_predictor_preserving_a3_source_binding,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    PACKET_HEADER as A3_PACKET_HEADER,
)
from tac.witness_dsl.predictor_preserving_taskspace_overlay import (
    PredictorPreservingOverlayError,
    overlay_g_on_predictor_camera_y1,
)
from tac.witness_dsl.taskspace_predictor_v2_consumer_seam import (
    TaskspacePredictorV2ConsumerSeamError,
    apply_generative_taskspace_correction_v2,
    parse_generative_taskspace_correction_v2,
)

PACKET_MAGIC: Final = b"TACG8R1\x00"
PACKET_VERSION: Final = 1
PACKET_FLAGS: Final = 0

# The header binds the complete source record plus the five load-bearing
# identities individually.  Pair IDs are additionally represented by the
# exact contiguous pair window and transitively by source_binding_sha256.
PACKET_HEADER: Final = struct.Struct(">8sBBHH32s32s32s32s32s32sIIII")

# pair_id, scorer_row, col_start, col_stop, semantic_class, semantic_role,
# shared R, shared G, shared B.  Columns use a half-open interval.
RUN_ROW: Final = struct.Struct(">HHHHBBBBB")
A3_EXPANDED_RGB_ROW: Final = struct.Struct(">HHHBBB")

# Concrete counted replacement for the monolithic G section.  It nests the
# existing semantic G packet and one G8R1 repair packet under an exact directory
# of two concatenated payloads; the real receiver must explicitly branch on
# this magic and then call the integration decoder below.
COMPOSITE_G_SECTION_MAGIC: Final = b"TACG8S1\x00"
COMPOSITE_G_SECTION_VERSION: Final = 1
COMPOSITE_G_SECTION_FLAGS: Final = 0
COMPOSITE_G_SECTION_HEADER: Final = struct.Struct(">8sBBHII32s32sII")
SEMANTIC_G_PACKET_MAGIC: Final = b"TACG1C\x00\x00"
PREDICTOR_PACKET_MAGIC: Final = b"LVLS1\x00"

SOURCE_SCHEMA: Final = "tac.taskspace_same_class_realization_repair_source.v1"
RECEIPT_SCHEMA: Final = "tac.taskspace_same_class_realization_repair_receipt.v1"
ADMISSION_SCHEMA: Final = "tac.taskspace_same_class_realization_repair_admission.v1"
TARGET_LINEAGE_POLICY: Final = "exact_target_labels_encoder_only_never_serialized.v1"
MAX_RUNS: Final = 0xFFFF
MAX_EXPANDED_CELLS: Final = MAX_BOUNDED_PAIRS * SCORER_HEIGHT * SCORER_WIDTH


class SameClassRealizationRepairError(ValueError):
    """The G8R1 packet, source binding, admission, or exact proof failed."""


class TaskspaceSectionRoleV1(StrEnum):
    """Closed counted-section roles used by the G8 source foreign keys."""

    P = "P"
    G = "G"


class SameClassSemanticRoleV1(StrEnum):
    """Closed five-class semantic-role universe carried redundantly per run."""

    ROAD = "Road"
    LANE = "Lane"
    UNDRIVABLE_BOUNDARY = "UndrivableBoundary"
    MOVABLE = "Movable"
    MY_CAR = "MyCar"


_ROLE_TO_CLASS: Final = {
    SameClassSemanticRoleV1.ROAD: 0,
    SameClassSemanticRoleV1.LANE: 1,
    SameClassSemanticRoleV1.UNDRIVABLE_BOUNDARY: 2,
    SameClassSemanticRoleV1.MOVABLE: 3,
    SameClassSemanticRoleV1.MY_CAR: 4,
}
_ROLE_TO_WIRE: Final = dict(_ROLE_TO_CLASS)
_WIRE_TO_ROLE: Final = {wire: role for role, wire in _ROLE_TO_WIRE.items()}


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return _sha256(memoryview(contiguous).cast("B"))


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
        raise SameClassRealizationRepairError(
            "same-class realization record is not finite canonical ASCII JSON"
        ) from exc


def _require_sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SameClassRealizationRepairError(f"{field} must be canonical lowercase SHA-256")
    return value


def _require_exact_int(value: object, field: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise SameClassRealizationRepairError(f"{field} must be an exact integer in [{minimum},{maximum}]")
    return value


def _validate_pair_ids(source_pair_ids: tuple[int, ...], *, field: str) -> tuple[int, ...]:
    if type(source_pair_ids) is not tuple or not 1 <= len(source_pair_ids) <= MAX_BOUNDED_PAIRS:
        raise SameClassRealizationRepairError(f"{field} must be a bounded nonempty exact tuple")
    if any(type(pair_id) is not int for pair_id in source_pair_ids):
        raise SameClassRealizationRepairError(f"{field} must contain exact integers")
    expected = tuple(range(source_pair_ids[0], source_pair_ids[0] + len(source_pair_ids)))
    if source_pair_ids != expected or not 0 <= expected[0] < expected[-1] + 1 <= 600:
        raise SameClassRealizationRepairError(f"{field} must be contiguous source-pair IDs inside [0,600)")
    return expected


def _immutable_array(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


def _require_ascii_token(value: object, field: str, *, maximum: int = 96) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        raise SameClassRealizationRepairError(f"{field} must be a nonempty bounded string")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise SameClassRealizationRepairError(f"{field} must be canonical ASCII") from exc
    if any(byte < 0x21 or byte > 0x7E for byte in encoded):
        raise SameClassRealizationRepairError(f"{field} must contain printable non-space ASCII")
    return value


def _require_packet_bytes(value: object, field: str, *, magic: bytes) -> bytes:
    if type(value) is not bytes or not value.startswith(magic):
        raise SameClassRealizationRepairError(f"{field} must be nonempty exact bytes with canonical magic")
    if len(value) > 0xFFFFFFFF:
        raise SameClassRealizationRepairError(f"{field} exceeds the version-1 byte ABI")
    return value


@dataclass(frozen=True, slots=True)
class ParsedSameClassCompositeGSectionV1:
    """Exact inner semantic-G and G8R1 payloads from one counted G section."""

    section_bytes: bytes
    semantic_g_packet: bytes
    repair_packet: bytes
    section_sha256: str
    semantic_g_packet_sha256: str
    repair_packet_sha256: str
    header_bytes: int

    def __post_init__(self) -> None:
        _require_packet_bytes(self.section_bytes, "composite G section", magic=COMPOSITE_G_SECTION_MAGIC)
        _require_packet_bytes(self.semantic_g_packet, "inner semantic G packet", magic=SEMANTIC_G_PACKET_MAGIC)
        _require_packet_bytes(self.repair_packet, "inner G8R1 repair packet", magic=PACKET_MAGIC)
        for field in ("section_sha256", "semantic_g_packet_sha256", "repair_packet_sha256"):
            _require_sha256(getattr(self, field), field)
        if self.section_sha256 != _sha256(self.section_bytes):
            raise SameClassRealizationRepairError("composite G section hash differs from exact bytes")
        if self.semantic_g_packet_sha256 != _sha256(self.semantic_g_packet):
            raise SameClassRealizationRepairError("inner semantic G hash differs from exact bytes")
        if self.repair_packet_sha256 != _sha256(self.repair_packet):
            raise SameClassRealizationRepairError("inner G8R1 hash differs from exact bytes")
        if type(self.header_bytes) is not int or self.header_bytes != COMPOSITE_G_SECTION_HEADER.size:
            raise SameClassRealizationRepairError("composite G section header ABI changed")
        if (
            compose_same_class_realization_repair_g_section(
                self.semantic_g_packet,
                self.repair_packet,
            )
            != self.section_bytes
        ):
            raise SameClassRealizationRepairError("composite G section differs from exact inner packets")


def compose_same_class_realization_repair_g_section(
    semantic_g_packet: bytes,
    repair_packet: bytes,
) -> bytes:
    """Build concrete counted G-section bytes for an allocator proposal."""

    semantic = _require_packet_bytes(
        semantic_g_packet,
        "inner semantic G packet",
        magic=SEMANTIC_G_PACKET_MAGIC,
    )
    repair = _require_packet_bytes(repair_packet, "inner G8R1 repair packet", magic=PACKET_MAGIC)
    body = semantic + repair
    section_bytes = COMPOSITE_G_SECTION_HEADER.size + len(body)
    if section_bytes > 0xFFFFFFFF:
        raise SameClassRealizationRepairError("composite G section exceeds the version-1 byte ABI")
    fixed = (
        COMPOSITE_G_SECTION_MAGIC,
        COMPOSITE_G_SECTION_VERSION,
        COMPOSITE_G_SECTION_FLAGS,
        0,
        len(semantic),
        len(repair),
        bytes.fromhex(_sha256(semantic)),
        bytes.fromhex(_sha256(repair)),
        section_bytes,
    )
    header_zero_crc = COMPOSITE_G_SECTION_HEADER.pack(*fixed, 0)
    crc = zlib.crc32(header_zero_crc + body) & 0xFFFFFFFF
    return COMPOSITE_G_SECTION_HEADER.pack(*fixed, crc) + body


def parse_same_class_realization_repair_g_section(
    section: bytes,
) -> ParsedSameClassCompositeGSectionV1:
    """Strict parse/re-encode closure for the concrete composite G section."""

    payload = _require_packet_bytes(section, "composite G section", magic=COMPOSITE_G_SECTION_MAGIC)
    if len(payload) < COMPOSITE_G_SECTION_HEADER.size:
        raise SameClassRealizationRepairError("composite G section is shorter than its exact header")
    try:
        (
            magic,
            version,
            flags,
            reserved,
            semantic_bytes,
            repair_bytes,
            semantic_sha_raw,
            repair_sha_raw,
            section_bytes,
            crc,
        ) = COMPOSITE_G_SECTION_HEADER.unpack_from(payload, 0)
    except struct.error as exc:  # pragma: no cover - guarded length
        raise SameClassRealizationRepairError("composite G section header is malformed") from exc
    if (
        magic != COMPOSITE_G_SECTION_MAGIC
        or version != COMPOSITE_G_SECTION_VERSION
        or flags != COMPOSITE_G_SECTION_FLAGS
        or reserved != 0
    ):
        raise SameClassRealizationRepairError("composite G section magic/version/flags are unsupported")
    if semantic_bytes < 1 or repair_bytes < 1:
        raise SameClassRealizationRepairError("composite G section requires both nonempty inner packets")
    if (
        section_bytes != len(payload)
        or section_bytes != COMPOSITE_G_SECTION_HEADER.size + semantic_bytes + repair_bytes
    ):
        raise SameClassRealizationRepairError("composite G section length accounting or trailing bytes changed")
    body = payload[COMPOSITE_G_SECTION_HEADER.size :]
    header_zero_crc = COMPOSITE_G_SECTION_HEADER.pack(
        magic,
        version,
        flags,
        reserved,
        semantic_bytes,
        repair_bytes,
        semantic_sha_raw,
        repair_sha_raw,
        section_bytes,
        0,
    )
    if zlib.crc32(header_zero_crc + body) & 0xFFFFFFFF != crc:
        raise SameClassRealizationRepairError("composite G section CRC differs from exact bytes")
    semantic = body[:semantic_bytes]
    repair = body[semantic_bytes:]
    _require_packet_bytes(semantic, "inner semantic G packet", magic=SEMANTIC_G_PACKET_MAGIC)
    _require_packet_bytes(repair, "inner G8R1 repair packet", magic=PACKET_MAGIC)
    if _sha256(semantic) != semantic_sha_raw.hex() or _sha256(repair) != repair_sha_raw.hex():
        raise SameClassRealizationRepairError("composite G inner payload hash differs from exact bytes")
    if compose_same_class_realization_repair_g_section(semantic, repair) != payload:
        raise SameClassRealizationRepairError("composite G section changed on strict parse/re-encode")
    return ParsedSameClassCompositeGSectionV1(
        section_bytes=payload,
        semantic_g_packet=semantic,
        repair_packet=repair,
        section_sha256=_sha256(payload),
        semantic_g_packet_sha256=_sha256(semantic),
        repair_packet_sha256=_sha256(repair),
        header_bytes=COMPOSITE_G_SECTION_HEADER.size,
    )


@dataclass(frozen=True, slots=True)
class CountedTaskspaceSectionIdentityV1:
    """Exact identity of one counted P or G section in the enclosing codec."""

    role: TaskspaceSectionRoleV1
    section_index: int
    codec_id: str
    payload_bytes: int
    payload_sha256: str

    def __post_init__(self) -> None:
        if type(self.role) is not TaskspaceSectionRoleV1:
            raise SameClassRealizationRepairError("section role escaped the closed P/G universe")
        _require_exact_int(self.section_index, "section_index", minimum=0, maximum=0xFFFF)
        _require_ascii_token(self.codec_id, "codec_id")
        _require_exact_int(self.payload_bytes, "payload_bytes", minimum=1, maximum=0xFFFFFFFF)
        _require_sha256(self.payload_sha256, "payload_sha256")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["role"] = self.role.value
        return value

    @property
    def identity_sha256(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "schema": "tac.counted_taskspace_section_identity.v1",
                    **self.as_dict(),
                }
            )
        )


@dataclass(frozen=True, slots=True)
class SameClassRealizationRepairSurfaceV1:
    """Live decoder inputs; dense camera and labels have no serialization API."""

    source_pair_ids: tuple[int, ...]
    corrected_y1_camera: np.ndarray
    g_semantic_labels: np.ndarray
    g_semantic_binding_sha256: str
    p_section_identity: CountedTaskspaceSectionIdentityV1
    g_section_identity: CountedTaskspaceSectionIdentityV1

    def __post_init__(self) -> None:
        pair_ids = _validate_pair_ids(self.source_pair_ids, field="source_pair_ids")
        camera = np.asarray(self.corrected_y1_camera)
        expected_camera = (len(pair_ids), CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
        if camera.dtype != np.uint8 or camera.shape != expected_camera:
            raise SameClassRealizationRepairError("corrected Y1 must be exact uint8 [pair,874,1164,3] camera bytes")
        labels = np.asarray(self.g_semantic_labels)
        expected_labels = (len(pair_ids), SCORER_HEIGHT, SCORER_WIDTH)
        if labels.dtype != np.uint8 or labels.shape != expected_labels:
            raise SameClassRealizationRepairError("G semantic labels must be exact uint8 [pair,384,512] bytes")
        if int(labels.min()) < 0 or int(labels.max()) > 4:
            raise SameClassRealizationRepairError("G semantic labels escaped the five-class universe")
        _require_sha256(self.g_semantic_binding_sha256, "g_semantic_binding_sha256")
        if (
            type(self.p_section_identity) is not CountedTaskspaceSectionIdentityV1
            or self.p_section_identity.role is not TaskspaceSectionRoleV1.P
        ):
            raise SameClassRealizationRepairError("P identity must be one exact counted P section")
        if (
            type(self.g_section_identity) is not CountedTaskspaceSectionIdentityV1
            or self.g_section_identity.role is not TaskspaceSectionRoleV1.G
        ):
            raise SameClassRealizationRepairError("G identity must be one exact counted G section")
        object.__setattr__(
            self,
            "corrected_y1_camera",
            _immutable_array(camera, dtype=np.dtype(np.uint8)),
        )
        object.__setattr__(
            self,
            "g_semantic_labels",
            _immutable_array(labels, dtype=np.dtype(np.uint8)),
        )

    @property
    def corrected_y1_camera_sha256(self) -> str:
        return _array_sha256(self.corrected_y1_camera)

    @property
    def g_labels_sha256(self) -> str:
        return _array_sha256(self.g_semantic_labels)


def build_same_class_realization_repair_surface_from_pga_sections(
    *,
    predictor_section_payload: bytes,
    semantic_g_section_payload: bytes,
    predictor_surface: PredictorCameraPairSurfaceV1,
    predictor_codec_id: str,
    semantic_g_codec_id: str,
) -> SameClassRealizationRepairSurfaceV1:
    """Strictly derive the G8 surface from typed P decode and parsed/applied G."""

    predictor = _require_packet_bytes(
        predictor_section_payload,
        "predictor section payload",
        magic=PREDICTOR_PACKET_MAGIC,
    )
    semantic_g = _require_packet_bytes(
        semantic_g_section_payload,
        "semantic G section payload",
        magic=SEMANTIC_G_PACKET_MAGIC,
    )
    if type(predictor_surface) is not PredictorCameraPairSurfaceV1:
        raise SameClassRealizationRepairError("PGA integration requires exact predictor camera surface")
    if predictor_surface.upstream_decode_receipt_sha256 is None:
        raise SameClassRealizationRepairError("PGA integration requires typed strict P decode receipt custody")
    state = predictor_surface.predictor_state
    if predictor != state.predictor_program:
        raise SameClassRealizationRepairError("predictor section bytes differ from typed strict P decode surface")
    try:
        parse_generative_taskspace_correction_v2(semantic_g, predictor_state=state)
        decoded_g = apply_generative_taskspace_correction_v2(semantic_g, predictor_state=state)
        overlay = overlay_g_on_predictor_camera_y1(
            predictor_surface.camera_p1,
            state.labels,
            decoded_g,
        )
        a3_source = derive_predictor_preserving_a3_source_binding(
            predictor_surface,
            decoded_g,
            overlay,
        )
    except (
        GenerativeTaskspaceCorrectionError,
        PredictorPreservingA3Error,
        PredictorPreservingOverlayError,
        TaskspacePredictorV2ConsumerSeamError,
    ) as exc:
        raise SameClassRealizationRepairError(
            "PGA integration failed strict semantic G parse/apply or typed P/G overlay custody"
        ) from exc
    semantic_binding = _sha256(
        _canonical_json(
            {
                "schema": "tac.same_class_realization_decoded_g_semantic_binding.v1",
                "predictor_semantic_binding_sha256": state.semantic_binding_sha256,
                "predictor_decode_receipt_sha256": predictor_surface.upstream_decode_receipt_sha256,
                "semantic_g_packet_sha256": _sha256(semantic_g),
                "g_labels_sha256": _array_sha256(decoded_g.labels),
                "overlay_receipt_sha256": a3_source.overlay_receipt_sha256,
                "corrected_y1_camera_sha256": a3_source.corrected_camera_y1_sha256,
                "source_pair_ids": list(state.source_pair_ids),
            }
        )
    )
    return SameClassRealizationRepairSurfaceV1(
        source_pair_ids=state.source_pair_ids,
        corrected_y1_camera=overlay.camera_y1,
        g_semantic_labels=decoded_g.labels,
        g_semantic_binding_sha256=semantic_binding,
        p_section_identity=CountedTaskspaceSectionIdentityV1(
            role=TaskspaceSectionRoleV1.P,
            section_index=0,
            codec_id=predictor_codec_id,
            payload_bytes=len(predictor),
            payload_sha256=_sha256(predictor),
        ),
        g_section_identity=CountedTaskspaceSectionIdentityV1(
            role=TaskspaceSectionRoleV1.G,
            section_index=1,
            codec_id=semantic_g_codec_id,
            payload_bytes=len(semantic_g),
            payload_sha256=_sha256(semantic_g),
        ),
    )


@dataclass(frozen=True, slots=True)
class SameClassRealizationRepairSourceBindingV1:
    """All live foreign keys embedded in one G8R1 packet."""

    schema: Literal["tac.taskspace_same_class_realization_repair_source.v1"]
    source_pair_ids: tuple[int, ...]
    corrected_y1_camera_sha256: str
    g_semantic_binding_sha256: str
    g_labels_sha256: str
    p_section_identity_sha256: str
    g_section_identity_sha256: str

    def __post_init__(self) -> None:
        if self.schema != SOURCE_SCHEMA:
            raise SameClassRealizationRepairError("G8R1 source schema changed")
        _validate_pair_ids(self.source_pair_ids, field="source_pair_ids")
        for field in (
            "corrected_y1_camera_sha256",
            "g_semantic_binding_sha256",
            "g_labels_sha256",
            "p_section_identity_sha256",
            "g_section_identity_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if self.p_section_identity_sha256 == self.g_section_identity_sha256:
            raise SameClassRealizationRepairError("P and G section identities must remain distinct")

    @property
    def pair_start(self) -> int:
        return self.source_pair_ids[0]

    @property
    def pair_count(self) -> int:
        return len(self.source_pair_ids)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_pair_ids"] = list(self.source_pair_ids)
        return value

    @property
    def binding_sha256(self) -> str:
        return _sha256(_canonical_json(self.as_dict()))


def derive_same_class_realization_repair_source_binding(
    surface: SameClassRealizationRepairSurfaceV1,
) -> SameClassRealizationRepairSourceBindingV1:
    """Derive exact packet foreign keys from the live corrected-Y1/G object."""

    if type(surface) is not SameClassRealizationRepairSurfaceV1:
        raise SameClassRealizationRepairError("G8R1 source derivation requires exact surface type")
    return SameClassRealizationRepairSourceBindingV1(
        schema=SOURCE_SCHEMA,
        source_pair_ids=surface.source_pair_ids,
        corrected_y1_camera_sha256=surface.corrected_y1_camera_sha256,
        g_semantic_binding_sha256=surface.g_semantic_binding_sha256,
        g_labels_sha256=surface.g_labels_sha256,
        p_section_identity_sha256=surface.p_section_identity.identity_sha256,
        g_section_identity_sha256=surface.g_section_identity.identity_sha256,
    )


@dataclass(frozen=True, slots=True)
class EncoderOnlyExactTargetLabelCustodyV1:
    """Exact compile-side target custody; none of this object is serialized."""

    source_artifact_sha256: str
    source_member_name: str
    source_member_sha256: str
    source_pair_ids: tuple[int, ...]
    target_labels_sha256: str
    target_labels: np.ndarray
    lineage_policy: Literal["exact_target_labels_encoder_only_never_serialized.v1"] = TARGET_LINEAGE_POLICY
    serialized_target_bytes: Literal[0] = 0

    def __post_init__(self) -> None:
        for field in ("source_artifact_sha256", "source_member_sha256", "target_labels_sha256"):
            _require_sha256(getattr(self, field), field)
        _require_ascii_token(self.source_member_name, "source_member_name", maximum=160)
        pair_ids = _validate_pair_ids(self.source_pair_ids, field="target source_pair_ids")
        if (
            self.lineage_policy != TARGET_LINEAGE_POLICY
            or type(self.serialized_target_bytes) is not int
            or self.serialized_target_bytes != 0
        ):
            raise SameClassRealizationRepairError("target labels must remain wholly encoder-only")
        labels = np.asarray(self.target_labels)
        if labels.dtype != np.uint8 or labels.shape != (
            len(pair_ids),
            SCORER_HEIGHT,
            SCORER_WIDTH,
        ):
            raise SameClassRealizationRepairError("encoder-only target labels must be uint8 [pair,384,512]")
        if int(labels.min()) < 0 or int(labels.max()) > 4:
            raise SameClassRealizationRepairError("target labels escaped the five-class universe")
        immutable = _immutable_array(labels, dtype=np.dtype(np.uint8))
        if _array_sha256(immutable) != self.target_labels_sha256:
            raise SameClassRealizationRepairError("target-label custody hash differs from exact bytes")
        object.__setattr__(self, "target_labels", immutable)

    @property
    def binding_sha256(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "schema": "tac.same_class_realization_target_custody.v1",
                    "source_artifact_sha256": self.source_artifact_sha256,
                    "source_member_name": self.source_member_name,
                    "source_member_sha256": self.source_member_sha256,
                    "source_pair_ids": list(self.source_pair_ids),
                    "target_labels_sha256": self.target_labels_sha256,
                    "target_labels_shape": list(self.target_labels.shape),
                    "lineage_policy": self.lineage_policy,
                    "serialized_target_bytes": self.serialized_target_bytes,
                }
            )
        )


@dataclass(frozen=True, slots=True, order=True)
class SameClassRealizationRepairRunV1:
    """One canonical horizontal scorer-cell run with one shared uint8 RGB."""

    source_pair_id: int
    scorer_row: int
    scorer_col_start: int
    scorer_col_stop: int
    semantic_class: int
    semantic_role: SameClassSemanticRoleV1
    rgb_u8: tuple[int, int, int]

    def __post_init__(self) -> None:
        _require_exact_int(self.source_pair_id, "source_pair_id", minimum=0, maximum=599)
        _require_exact_int(self.scorer_row, "scorer_row", minimum=0, maximum=SCORER_HEIGHT - 1)
        _require_exact_int(
            self.scorer_col_start,
            "scorer_col_start",
            minimum=0,
            maximum=SCORER_WIDTH - 1,
        )
        _require_exact_int(
            self.scorer_col_stop,
            "scorer_col_stop",
            minimum=1,
            maximum=SCORER_WIDTH,
        )
        if self.scorer_col_start >= self.scorer_col_stop:
            raise SameClassRealizationRepairError("repair run must be a nonempty half-open interval")
        _require_exact_int(self.semantic_class, "semantic_class", minimum=0, maximum=4)
        if type(self.semantic_role) is not SameClassSemanticRoleV1:
            raise SameClassRealizationRepairError("semantic role escaped the closed five-role universe")
        if _ROLE_TO_CLASS[self.semantic_role] != self.semantic_class:
            raise SameClassRealizationRepairError("semantic class and semantic role disagree")
        if type(self.rgb_u8) is not tuple or len(self.rgb_u8) != CHANNELS:
            raise SameClassRealizationRepairError("rgb_u8 must be one exact shared RGB tuple")
        for channel, value in enumerate(self.rgb_u8):
            _require_exact_int(value, f"rgb_u8[{channel}]", minimum=0, maximum=255)

    @property
    def address(self) -> tuple[int, int, int, int]:
        return (
            self.source_pair_id,
            self.scorer_row,
            self.scorer_col_start,
            self.scorer_col_stop,
        )

    @property
    def expanded_cell_count(self) -> int:
        return self.scorer_col_stop - self.scorer_col_start

    def merge_key(self) -> tuple[int, int, int, SameClassSemanticRoleV1, tuple[int, int, int]]:
        return (
            self.source_pair_id,
            self.scorer_row,
            self.semantic_class,
            self.semantic_role,
            self.rgb_u8,
        )


@dataclass(frozen=True, slots=True)
class SameClassRealizationRepairProgramV1:
    """Canonical nonempty G8R1 run program."""

    runs: tuple[SameClassRealizationRepairRunV1, ...]

    def __post_init__(self) -> None:
        if type(self.runs) is not tuple or not self.runs:
            raise SameClassRealizationRepairError("G8R1 program must contain nonempty exact run tuple")
        if len(self.runs) > MAX_RUNS:
            raise SameClassRealizationRepairError("G8R1 program exceeds the version-1 run limit")
        if any(type(run) is not SameClassRealizationRepairRunV1 for run in self.runs):
            raise SameClassRealizationRepairError("G8R1 run changed exact type")
        addresses = tuple(run.address for run in self.runs)
        if addresses != tuple(sorted(addresses)) or len(addresses) != len(set(addresses)):
            raise SameClassRealizationRepairError("repair runs must be unique canonical address order")
        previous: SameClassRealizationRepairRunV1 | None = None
        for run in self.runs:
            if previous is not None and (
                run.source_pair_id,
                run.scorer_row,
            ) == (
                previous.source_pair_id,
                previous.scorer_row,
            ):
                if run.scorer_col_start < previous.scorer_col_stop:
                    raise SameClassRealizationRepairError("repair runs overlap one scorer-cell support")
                if run.scorer_col_start == previous.scorer_col_stop and run.merge_key() == previous.merge_key():
                    raise SameClassRealizationRepairError(
                        "adjacent identical repair runs are noncanonical and must merge"
                    )
            previous = run
        if self.expanded_cell_count > MAX_EXPANDED_CELLS:
            raise SameClassRealizationRepairError("expanded G8R1 cells exceed bounded pair geometry")

    @property
    def run_count(self) -> int:
        return len(self.runs)

    @property
    def expanded_cell_count(self) -> int:
        return sum(run.expanded_cell_count for run in self.runs)

    @property
    def body_bytes(self) -> int:
        return self.run_count * RUN_ROW.size


def _encode_packet(
    program: SameClassRealizationRepairProgramV1,
    source: SameClassRealizationRepairSourceBindingV1,
) -> bytes:
    if type(program) is not SameClassRealizationRepairProgramV1:
        raise SameClassRealizationRepairError("G8R1 encode requires exact program type")
    if type(source) is not SameClassRealizationRepairSourceBindingV1:
        raise SameClassRealizationRepairError("G8R1 encode requires exact source binding")
    if any(run.source_pair_id not in source.source_pair_ids for run in program.runs):
        raise SameClassRealizationRepairError("repair run addresses a foreign source pair")
    body = bytearray()
    for run in program.runs:
        body.extend(
            RUN_ROW.pack(
                run.source_pair_id,
                run.scorer_row,
                run.scorer_col_start,
                run.scorer_col_stop,
                run.semantic_class,
                _ROLE_TO_WIRE[run.semantic_role],
                *run.rgb_u8,
            )
        )
    if len(body) != program.body_bytes:
        raise SameClassRealizationRepairError("G8R1 body byte accounting changed")
    packet_bytes = PACKET_HEADER.size + len(body)
    fixed = (
        PACKET_MAGIC,
        PACKET_VERSION,
        PACKET_FLAGS,
        source.pair_start,
        source.pair_count,
        bytes.fromhex(source.binding_sha256),
        bytes.fromhex(source.corrected_y1_camera_sha256),
        bytes.fromhex(source.g_semantic_binding_sha256),
        bytes.fromhex(source.g_labels_sha256),
        bytes.fromhex(source.p_section_identity_sha256),
        bytes.fromhex(source.g_section_identity_sha256),
        program.run_count,
        len(body),
        packet_bytes,
    )
    header_zero_crc = PACKET_HEADER.pack(*fixed, 0)
    crc = zlib.crc32(header_zero_crc + body) & 0xFFFFFFFF
    return PACKET_HEADER.pack(*fixed, crc) + bytes(body)


@dataclass(frozen=True, slots=True)
class ParsedSameClassRealizationRepairPacketV1:
    packet: bytes
    source_binding_sha256: str
    pair_start: int
    pair_count: int
    packet_bytes: int
    packet_header_bytes: int
    packet_body_bytes: int
    program: SameClassRealizationRepairProgramV1


def parse_same_class_realization_repair_packet(
    packet: bytes,
    *,
    expected_source_binding: SameClassRealizationRepairSourceBindingV1,
) -> ParsedSameClassRealizationRepairPacketV1:
    """Strictly parse CRC, all foreign keys, rows, and typed re-encoding."""

    if type(packet) is not bytes or len(packet) < PACKET_HEADER.size:
        raise SameClassRealizationRepairError("G8R1 packet is shorter than its exact header")
    if type(expected_source_binding) is not SameClassRealizationRepairSourceBindingV1:
        raise SameClassRealizationRepairError("G8R1 parse requires exact source binding")
    try:
        (
            magic,
            version,
            flags,
            pair_start,
            pair_count,
            source_binding_raw,
            corrected_y1_raw,
            g_semantic_binding_raw,
            g_labels_raw,
            p_identity_raw,
            g_identity_raw,
            run_count,
            body_bytes,
            packet_bytes,
            crc,
        ) = PACKET_HEADER.unpack_from(packet, 0)
    except struct.error as exc:  # pragma: no cover - guarded length
        raise SameClassRealizationRepairError("G8R1 packet header is malformed") from exc
    if magic != PACKET_MAGIC or version != PACKET_VERSION or flags != PACKET_FLAGS:
        raise SameClassRealizationRepairError("G8R1 packet magic/version/flags are unsupported")
    if pair_start != expected_source_binding.pair_start or pair_count != expected_source_binding.pair_count:
        raise SameClassRealizationRepairError("G8R1 pair window differs from exact source binding")
    expected_hashes = {
        "source binding": (source_binding_raw.hex(), expected_source_binding.binding_sha256),
        "corrected Y1": (
            corrected_y1_raw.hex(),
            expected_source_binding.corrected_y1_camera_sha256,
        ),
        "G semantic binding": (
            g_semantic_binding_raw.hex(),
            expected_source_binding.g_semantic_binding_sha256,
        ),
        "G labels": (g_labels_raw.hex(), expected_source_binding.g_labels_sha256),
        "P section identity": (
            p_identity_raw.hex(),
            expected_source_binding.p_section_identity_sha256,
        ),
        "G section identity": (
            g_identity_raw.hex(),
            expected_source_binding.g_section_identity_sha256,
        ),
    }
    for label, (actual, expected) in expected_hashes.items():
        if actual != expected:
            raise SameClassRealizationRepairError(f"G8R1 packet {label} differs from live source")
    if packet_bytes != len(packet) or body_bytes != len(packet) - PACKET_HEADER.size:
        raise SameClassRealizationRepairError("G8R1 packet length accounting or trailing bytes changed")
    if run_count < 1 or run_count > MAX_RUNS or body_bytes != run_count * RUN_ROW.size:
        raise SameClassRealizationRepairError("G8R1 run count/body length differs from its ABI")
    body = packet[PACKET_HEADER.size :]
    fixed = (
        magic,
        version,
        flags,
        pair_start,
        pair_count,
        source_binding_raw,
        corrected_y1_raw,
        g_semantic_binding_raw,
        g_labels_raw,
        p_identity_raw,
        g_identity_raw,
        run_count,
        body_bytes,
        packet_bytes,
    )
    if zlib.crc32(PACKET_HEADER.pack(*fixed, 0) + body) & 0xFFFFFFFF != crc:
        raise SameClassRealizationRepairError("G8R1 packet CRC differs from exact bytes")

    rows: list[SameClassRealizationRepairRunV1] = []
    offset = 0
    try:
        for _ in range(run_count):
            (
                pair_id,
                scorer_row,
                col_start,
                col_stop,
                semantic_class,
                role_wire,
                red,
                green,
                blue,
            ) = RUN_ROW.unpack_from(body, offset)
            offset += RUN_ROW.size
            try:
                semantic_role = _WIRE_TO_ROLE[role_wire]
            except KeyError as exc:
                raise SameClassRealizationRepairError("G8R1 run semantic role escaped the closed universe") from exc
            rows.append(
                SameClassRealizationRepairRunV1(
                    source_pair_id=pair_id,
                    scorer_row=scorer_row,
                    scorer_col_start=col_start,
                    scorer_col_stop=col_stop,
                    semantic_class=semantic_class,
                    semantic_role=semantic_role,
                    rgb_u8=(red, green, blue),
                )
            )
    except struct.error as exc:  # pragma: no cover - exact body length checked above
        raise SameClassRealizationRepairError("G8R1 run row is truncated") from exc
    if offset != len(body):
        raise SameClassRealizationRepairError("G8R1 packet body was not consumed exactly")
    program = SameClassRealizationRepairProgramV1(tuple(rows))
    if any(run.source_pair_id not in expected_source_binding.source_pair_ids for run in program.runs):
        raise SameClassRealizationRepairError("G8R1 run addresses a foreign source pair")
    if _encode_packet(program, expected_source_binding) != packet:
        raise SameClassRealizationRepairError("G8R1 packet changed on typed parse/re-encode")
    return ParsedSameClassRealizationRepairPacketV1(
        packet=packet,
        source_binding_sha256=expected_source_binding.binding_sha256,
        pair_start=pair_start,
        pair_count=pair_count,
        packet_bytes=packet_bytes,
        packet_header_bytes=PACKET_HEADER.size,
        packet_body_bytes=body_bytes,
        program=program,
    )


def _expanded_cells(
    program: SameClassRealizationRepairProgramV1,
) -> tuple[SparseConstantRGBCellV1, ...]:
    return tuple(
        SparseConstantRGBCellV1(
            source_pair_id=run.source_pair_id,
            scorer_row=run.scorer_row,
            scorer_col=scorer_col,
            rgb_u8=run.rgb_u8,
        )
        for run in program.runs
        for scorer_col in range(run.scorer_col_start, run.scorer_col_stop)
    )


def _validate_program_against_live_g(
    program: SameClassRealizationRepairProgramV1,
    surface: SameClassRealizationRepairSurfaceV1,
) -> None:
    """Reject a semantically foreign repair before any camera mutation."""

    pair_to_index = {pair_id: index for index, pair_id in enumerate(surface.source_pair_ids)}
    for run in program.runs:
        try:
            pair_index = pair_to_index[run.source_pair_id]
        except KeyError as exc:  # parse/source binding normally rejects this first
            raise SameClassRealizationRepairError("repair run addresses a foreign source pair") from exc
        live_classes = surface.g_semantic_labels[
            pair_index,
            run.scorer_row,
            run.scorer_col_start : run.scorer_col_stop,
        ]
        if not np.all(live_classes == run.semantic_class):
            raise SameClassRealizationRepairError(
                "G8R1 run semantic class differs from packet-bound live G semantic labels"
            )


def _expected_scorer_ownership_mask(
    program: SameClassRealizationRepairProgramV1,
    source_pair_ids: tuple[int, ...],
) -> np.ndarray:
    pair_to_index = {pair_id: index for index, pair_id in enumerate(source_pair_ids)}
    mask = np.zeros((len(source_pair_ids), SCORER_HEIGHT, SCORER_WIDTH), dtype=np.bool_)
    for run in program.runs:
        try:
            pair_index = pair_to_index[run.source_pair_id]
        except KeyError as exc:
            raise SameClassRealizationRepairError("repair run addresses a foreign source pair") from exc
        mask[
            pair_index,
            run.scorer_row,
            run.scorer_col_start : run.scorer_col_stop,
        ] = True
    return mask


@dataclass(frozen=True, slots=True)
class SameClassRealizationRepairReceiptV1:
    """Receiver proof and exact run-vs-expanded-A3 byte accounting."""

    schema: Literal["tac.taskspace_same_class_realization_repair_receipt.v1"]
    source_pair_ids: tuple[int, ...]
    source_binding_sha256: str
    corrected_y1_camera_sha256: str
    g_semantic_binding_sha256: str
    g_labels_sha256: str
    p_section_identity_sha256: str
    g_section_identity_sha256: str
    packet_sha256: str
    packet_bytes: int
    packet_header_bytes: int
    packet_body_bytes: int
    run_row_bytes: int
    run_count: int
    expanded_cell_count: int
    expanded_a3_packet_bytes: int
    expanded_a3_body_bytes: int
    run_packet_savings_bytes: int
    run_body_savings_bytes: int
    output_camera_y1_sha256: str
    output_g_labels_sha256: str
    owned_scorer_mask_sha256: str
    owned_camera_mask_sha256: str
    input_scorer_numerators_sha256: str
    output_scorer_numerators_sha256: str
    owned_scorer_cells: int
    owned_camera_values: int
    actually_changed_camera_values: int
    scorer_denominator: int
    run_body_compression_observed: bool
    coordinate_present: Literal[True] = True
    through_r_target_realization_debt_closed: Literal[False] = False
    semantic_labels_bit_identical: Literal[True] = True
    constant_rgb_target_numerators_exact: Literal[True] = True
    outside_owned_camera_bytes_identical: Literal[True] = True
    outside_owned_scorer_numerators_identical: Literal[True] = True
    packet_parse_reencode_exact: Literal[True] = True
    deterministic_double_decode: Literal[True] = True
    packet_absence_rejected: Literal[True] = True
    stale_integrity_mutation_rejected: Literal[True] = True
    counted_rgb_mutation_is_receiver_causal: Literal[True] = True
    counted_run_deletion_is_receiver_causal_or_empty_rejected: Literal[True] = True
    target_labels_encoder_only: Literal[True] = True
    dense_target_labels_serialized: Literal[False] = False
    dense_g_labels_serialized: Literal[False] = False
    dense_scorer_state_serialized: Literal[False] = False
    dense_margins_serialized: Literal[False] = False
    scorer_invoked: Literal[False] = False
    score_claim: Literal[False] = False
    candidate_claim: Literal[False] = False
    exact_eval_invoked: Literal[False] = False
    promotion_eligible: Literal[False] = False
    originality_claim: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise SameClassRealizationRepairError("G8R1 receipt schema changed")
        _validate_pair_ids(self.source_pair_ids, field="receipt source_pair_ids")
        for field in (
            "source_binding_sha256",
            "corrected_y1_camera_sha256",
            "g_semantic_binding_sha256",
            "g_labels_sha256",
            "p_section_identity_sha256",
            "g_section_identity_sha256",
            "packet_sha256",
            "output_camera_y1_sha256",
            "output_g_labels_sha256",
            "owned_scorer_mask_sha256",
            "owned_camera_mask_sha256",
            "input_scorer_numerators_sha256",
            "output_scorer_numerators_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        reconstructed = SameClassRealizationRepairSourceBindingV1(
            schema=SOURCE_SCHEMA,
            source_pair_ids=self.source_pair_ids,
            corrected_y1_camera_sha256=self.corrected_y1_camera_sha256,
            g_semantic_binding_sha256=self.g_semantic_binding_sha256,
            g_labels_sha256=self.g_labels_sha256,
            p_section_identity_sha256=self.p_section_identity_sha256,
            g_section_identity_sha256=self.g_section_identity_sha256,
        )
        if reconstructed.binding_sha256 != self.source_binding_sha256:
            raise SameClassRealizationRepairError("G8R1 receipt source binding does not recompose")
        exact_ints = (
            "packet_bytes",
            "packet_header_bytes",
            "packet_body_bytes",
            "run_row_bytes",
            "run_count",
            "expanded_cell_count",
            "expanded_a3_packet_bytes",
            "expanded_a3_body_bytes",
            "run_packet_savings_bytes",
            "run_body_savings_bytes",
            "owned_scorer_cells",
            "owned_camera_values",
            "actually_changed_camera_values",
            "scorer_denominator",
        )
        if any(type(getattr(self, field)) is not int for field in exact_ints):
            raise SameClassRealizationRepairError("G8R1 receipt byte/count fields must be exact integers")
        if self.packet_header_bytes != PACKET_HEADER.size or self.run_row_bytes != RUN_ROW.size:
            raise SameClassRealizationRepairError("G8R1 receipt packet/run ABI changed")
        if self.packet_bytes != self.packet_header_bytes + self.packet_body_bytes:
            raise SameClassRealizationRepairError("G8R1 receipt packet byte decomposition does not close")
        if self.run_count < 1 or self.packet_body_bytes != self.run_count * self.run_row_bytes:
            raise SameClassRealizationRepairError("G8R1 receipt run body accounting changed")
        if self.expanded_cell_count < self.run_count:
            raise SameClassRealizationRepairError("G8R1 expanded-cell count is smaller than run count")
        expected_a3_body = self.expanded_cell_count * A3_EXPANDED_RGB_ROW.size
        expected_a3_packet = A3_PACKET_HEADER.size + expected_a3_body
        if self.expanded_a3_body_bytes != expected_a3_body or self.expanded_a3_packet_bytes != expected_a3_packet:
            raise SameClassRealizationRepairError("expanded per-cell A3 byte accounting changed")
        if self.run_body_savings_bytes != self.expanded_a3_body_bytes - self.packet_body_bytes:
            raise SameClassRealizationRepairError("G8R1 run-body savings accounting changed")
        if self.run_packet_savings_bytes != self.expanded_a3_packet_bytes - self.packet_bytes:
            raise SameClassRealizationRepairError("G8R1 full-packet savings accounting changed")
        if self.run_body_compression_observed is not (self.run_body_savings_bytes > 0):
            raise SameClassRealizationRepairError("G8R1 compression observation differs from exact bytes")
        if self.owned_scorer_cells != self.expanded_cell_count:
            raise SameClassRealizationRepairError("G8R1 ownership differs from expanded run cells")
        if not 1 <= self.actually_changed_camera_values <= self.owned_camera_values:
            raise SameClassRealizationRepairError("G8R1 receipt must contain a realized non-no-op repair")
        if self.scorer_denominator <= 0:
            raise SameClassRealizationRepairError("G8R1 scorer denominator must be positive")
        if self.output_g_labels_sha256 != self.g_labels_sha256:
            raise SameClassRealizationRepairError("G8R1 receiver changed semantic labels")
        if self.output_camera_y1_sha256 == self.corrected_y1_camera_sha256:
            raise SameClassRealizationRepairError("G8R1 receipt claims a camera no-op")
        truth = {
            "coordinate_present": True,
            "through_r_target_realization_debt_closed": False,
            "semantic_labels_bit_identical": True,
            "constant_rgb_target_numerators_exact": True,
            "outside_owned_camera_bytes_identical": True,
            "outside_owned_scorer_numerators_identical": True,
            "packet_parse_reencode_exact": True,
            "deterministic_double_decode": True,
            "packet_absence_rejected": True,
            "stale_integrity_mutation_rejected": True,
            "counted_rgb_mutation_is_receiver_causal": True,
            "counted_run_deletion_is_receiver_causal_or_empty_rejected": True,
            "target_labels_encoder_only": True,
            "dense_target_labels_serialized": False,
            "dense_g_labels_serialized": False,
            "dense_scorer_state_serialized": False,
            "dense_margins_serialized": False,
            "scorer_invoked": False,
            "score_claim": False,
            "candidate_claim": False,
            "exact_eval_invoked": False,
            "promotion_eligible": False,
            "originality_claim": False,
            "research_only": True,
        }
        if any(getattr(self, field) is not expected for field, expected in truth.items()):
            raise SameClassRealizationRepairError("G8R1 receipt truth labels became permissive")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_pair_ids"] = list(self.source_pair_ids)
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


@dataclass(frozen=True, slots=True)
class DecodedSameClassRealizationRepairV1:
    """Exact corrected Y1 plus unchanged semantic labels and ownership masks."""

    camera_y1: np.ndarray
    semantic_labels: np.ndarray
    owned_scorer_mask: np.ndarray
    owned_camera_mask: np.ndarray
    receipt: SameClassRealizationRepairReceiptV1

    def __post_init__(self) -> None:
        pair_count = len(self.receipt.source_pair_ids)
        camera = np.asarray(self.camera_y1)
        labels = np.asarray(self.semantic_labels)
        scorer_mask = np.asarray(self.owned_scorer_mask)
        camera_mask = np.asarray(self.owned_camera_mask)
        if camera.dtype != np.uint8 or camera.shape != (
            pair_count,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            CHANNELS,
        ):
            raise SameClassRealizationRepairError("decoded G8R1 camera ABI changed")
        if labels.dtype != np.uint8 or labels.shape != (
            pair_count,
            SCORER_HEIGHT,
            SCORER_WIDTH,
        ):
            raise SameClassRealizationRepairError("decoded G8R1 labels ABI changed")
        if scorer_mask.dtype != np.bool_ or scorer_mask.shape != labels.shape:
            raise SameClassRealizationRepairError("decoded G8R1 scorer mask ABI changed")
        if camera_mask.dtype != np.bool_ or camera_mask.shape != camera.shape[:-1]:
            raise SameClassRealizationRepairError("decoded G8R1 camera mask ABI changed")
        expected_hashes = {
            "output_camera_y1_sha256": _array_sha256(camera),
            "output_g_labels_sha256": _array_sha256(labels),
            "owned_scorer_mask_sha256": _array_sha256(scorer_mask),
            "owned_camera_mask_sha256": _array_sha256(camera_mask),
        }
        for field, expected in expected_hashes.items():
            if getattr(self.receipt, field) != expected:
                raise SameClassRealizationRepairError(f"decoded G8R1 {field} differs from receipt")
        if self.receipt.owned_scorer_cells != int(np.count_nonzero(scorer_mask)):
            raise SameClassRealizationRepairError("decoded G8R1 scorer ownership count differs from mask")
        if self.receipt.owned_camera_values != int(np.count_nonzero(camera_mask)) * CHANNELS:
            raise SameClassRealizationRepairError("decoded G8R1 camera ownership count differs from mask")
        object.__setattr__(self, "camera_y1", _immutable_array(camera, dtype=np.dtype(np.uint8)))
        object.__setattr__(self, "semantic_labels", _immutable_array(labels, dtype=np.dtype(np.uint8)))
        object.__setattr__(
            self,
            "owned_scorer_mask",
            _immutable_array(scorer_mask, dtype=np.dtype(np.bool_)),
        )
        object.__setattr__(
            self,
            "owned_camera_mask",
            _immutable_array(camera_mask, dtype=np.dtype(np.bool_)),
        )


def _decode_parsed_packet(
    parsed: ParsedSameClassRealizationRepairPacketV1,
    source: SameClassRealizationRepairSourceBindingV1,
    surface: SameClassRealizationRepairSurfaceV1,
) -> DecodedSameClassRealizationRepairV1:
    _validate_program_against_live_g(parsed.program, surface)
    cells = _expanded_cells(parsed.program)
    try:
        reconstruction = apply_disjoint_camera_cell_reconstruction_v1(
            surface.corrected_y1_camera,
            source_pair_ids=source.source_pair_ids,
            constant_rgb_cells=cells,
        )
    except PredictorPreservingA3Error as exc:
        raise SameClassRealizationRepairError(f"G8R1 canonical disjoint reconstruction failed: {exc}") from exc
    if (
        reconstruction.constant_rgb_target_numerators_exact is not True
        or reconstruction.copy_source_target_numerators_exact is not False
    ):
        raise SameClassRealizationRepairError(
            "G8R1 canonical reconstruction did not return the exact constant-RGB proof"
        )
    labels = np.ascontiguousarray(surface.g_semantic_labels.copy())
    packet_bytes = parsed.packet_bytes
    expanded_a3_body_bytes = parsed.program.expanded_cell_count * A3_EXPANDED_RGB_ROW.size
    expanded_a3_packet_bytes = A3_PACKET_HEADER.size + expanded_a3_body_bytes
    receipt = SameClassRealizationRepairReceiptV1(
        schema=RECEIPT_SCHEMA,
        source_pair_ids=source.source_pair_ids,
        source_binding_sha256=source.binding_sha256,
        corrected_y1_camera_sha256=source.corrected_y1_camera_sha256,
        g_semantic_binding_sha256=source.g_semantic_binding_sha256,
        g_labels_sha256=source.g_labels_sha256,
        p_section_identity_sha256=source.p_section_identity_sha256,
        g_section_identity_sha256=source.g_section_identity_sha256,
        packet_sha256=_sha256(parsed.packet),
        packet_bytes=packet_bytes,
        packet_header_bytes=parsed.packet_header_bytes,
        packet_body_bytes=parsed.packet_body_bytes,
        run_row_bytes=RUN_ROW.size,
        run_count=parsed.program.run_count,
        expanded_cell_count=parsed.program.expanded_cell_count,
        expanded_a3_packet_bytes=expanded_a3_packet_bytes,
        expanded_a3_body_bytes=expanded_a3_body_bytes,
        run_packet_savings_bytes=expanded_a3_packet_bytes - packet_bytes,
        run_body_savings_bytes=expanded_a3_body_bytes - parsed.packet_body_bytes,
        output_camera_y1_sha256=_array_sha256(reconstruction.output_camera),
        output_g_labels_sha256=_array_sha256(labels),
        owned_scorer_mask_sha256=_array_sha256(reconstruction.owned_scorer_mask),
        owned_camera_mask_sha256=_array_sha256(reconstruction.owned_camera_mask),
        input_scorer_numerators_sha256=_array_sha256(reconstruction.base_scorer_numerators),
        output_scorer_numerators_sha256=_array_sha256(reconstruction.output_scorer_numerators),
        owned_scorer_cells=int(np.count_nonzero(reconstruction.owned_scorer_mask)),
        owned_camera_values=int(np.count_nonzero(reconstruction.owned_camera_mask)) * CHANNELS,
        actually_changed_camera_values=reconstruction.actually_changed_camera_values,
        scorer_denominator=reconstruction.scorer_denominator,
        run_body_compression_observed=(expanded_a3_body_bytes > parsed.packet_body_bytes),
    )
    return DecodedSameClassRealizationRepairV1(
        camera_y1=reconstruction.output_camera,
        semantic_labels=labels,
        owned_scorer_mask=reconstruction.owned_scorer_mask,
        owned_camera_mask=reconstruction.owned_camera_mask,
        receipt=receipt,
    )


def decode_same_class_realization_repair_packet(
    packet: bytes,
    *,
    surface: SameClassRealizationRepairSurfaceV1,
) -> DecodedSameClassRealizationRepairV1:
    """Decode twice against live corrected-Y1/G inputs; targets never enter."""

    if type(surface) is not SameClassRealizationRepairSurfaceV1:
        raise SameClassRealizationRepairError("G8R1 decode requires exact source surface")
    source = derive_same_class_realization_repair_source_binding(surface)
    parsed = parse_same_class_realization_repair_packet(
        packet,
        expected_source_binding=source,
    )
    first = _decode_parsed_packet(parsed, source, surface)
    second = _decode_parsed_packet(parsed, source, surface)
    if first.receipt != second.receipt:
        raise SameClassRealizationRepairError("G8R1 deterministic double decode changed receipt")
    for field in ("camera_y1", "semantic_labels", "owned_scorer_mask", "owned_camera_mask"):
        if not np.array_equal(getattr(first, field), getattr(second, field)):
            raise SameClassRealizationRepairError(f"G8R1 deterministic double decode changed {field}")
    if parse_same_class_realization_repair_receipt(first.receipt.to_receipt_bytes()) != first.receipt:
        raise SameClassRealizationRepairError("G8R1 receipt failed strict parse/re-emit closure")
    return first


def parse_same_class_realization_repair_receipt(
    payload: bytes,
) -> SameClassRealizationRepairReceiptV1:
    """Strict canonical JSON receipt parse with duplicate-key/type rejection."""

    if type(payload) is not bytes or not payload:
        raise SameClassRealizationRepairError("G8R1 receipt must be nonempty exact bytes")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SameClassRealizationRepairError(f"G8R1 receipt repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=unique_pairs)
    except SameClassRealizationRepairError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SameClassRealizationRepairError("G8R1 receipt is not strict ASCII JSON") from exc
    if type(value) is not dict or set(value) != set(SameClassRealizationRepairReceiptV1.__dataclass_fields__):
        raise SameClassRealizationRepairError("G8R1 receipt fields differ from the closed schema")
    if _canonical_json(value) != payload:
        raise SameClassRealizationRepairError("G8R1 receipt is not canonical JSON")
    if type(value["source_pair_ids"]) is not list or any(
        type(pair_id) is not int for pair_id in value["source_pair_ids"]
    ):
        raise SameClassRealizationRepairError("G8R1 receipt source_pair_ids must be a JSON list of exact integers")
    value["source_pair_ids"] = tuple(value["source_pair_ids"])
    try:
        receipt = SameClassRealizationRepairReceiptV1(**value)
    except TypeError as exc:
        raise SameClassRealizationRepairError("G8R1 receipt contains invalid typed fields") from exc
    if receipt.to_receipt_bytes() != payload:
        raise SameClassRealizationRepairError("G8R1 receipt changed on typed parse/re-emit")
    return receipt


@dataclass(frozen=True, slots=True)
class SameClassRealizationRepairAdmissionReceiptV1:
    """Compile-only proof that every run is semantic-label preserving."""

    schema: Literal["tac.taskspace_same_class_realization_repair_admission.v1"]
    source_pair_ids: tuple[int, ...]
    source_binding_sha256: str
    packet_sha256: str
    decode_receipt_sha256: str
    target_custody_binding_sha256: str
    target_labels_sha256: str
    run_count: int
    admitted_cell_count: int
    current_g_class_equals_target_class_for_every_cell: Literal[True] = True
    target_inputs_encoder_only: Literal[True] = True
    serialized_target_bytes: Literal[0] = 0
    scorer_invoked: Literal[False] = False
    score_claim: Literal[False] = False
    candidate_claim: Literal[False] = False
    exact_eval_invoked: Literal[False] = False
    promotion_eligible: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != ADMISSION_SCHEMA:
            raise SameClassRealizationRepairError("G8R1 admission schema changed")
        _validate_pair_ids(self.source_pair_ids, field="admission source_pair_ids")
        for field in (
            "source_binding_sha256",
            "packet_sha256",
            "decode_receipt_sha256",
            "target_custody_binding_sha256",
            "target_labels_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if type(self.run_count) is not int or type(self.admitted_cell_count) is not int:
            raise SameClassRealizationRepairError("G8R1 admission counts must be exact integers")
        if self.run_count < 1 or self.admitted_cell_count < self.run_count:
            raise SameClassRealizationRepairError("G8R1 admission counts are inconsistent")
        bool_truth = {
            "current_g_class_equals_target_class_for_every_cell": True,
            "target_inputs_encoder_only": True,
            "scorer_invoked": False,
            "score_claim": False,
            "candidate_claim": False,
            "exact_eval_invoked": False,
            "promotion_eligible": False,
            "research_only": True,
        }
        if (
            any(getattr(self, field) is not expected for field, expected in bool_truth.items())
            or type(self.serialized_target_bytes) is not int
            or self.serialized_target_bytes != 0
        ):
            raise SameClassRealizationRepairError("G8R1 admission truth labels became permissive")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_pair_ids"] = list(self.source_pair_ids)
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


@dataclass(frozen=True, slots=True)
class CompiledSameClassRealizationRepairV1:
    packet: bytes
    program: SameClassRealizationRepairProgramV1
    source_binding: SameClassRealizationRepairSourceBindingV1
    decoded: DecodedSameClassRealizationRepairV1
    admission_receipt: SameClassRealizationRepairAdmissionReceiptV1

    def __post_init__(self) -> None:
        if type(self.packet) is not bytes or not self.packet:
            raise SameClassRealizationRepairError("compiled G8R1 packet must be nonempty exact bytes")
        if type(self.program) is not SameClassRealizationRepairProgramV1:
            raise SameClassRealizationRepairError("compiled G8R1 program changed exact type")
        if type(self.source_binding) is not SameClassRealizationRepairSourceBindingV1:
            raise SameClassRealizationRepairError("compiled G8R1 source binding changed exact type")
        if type(self.decoded) is not DecodedSameClassRealizationRepairV1:
            raise SameClassRealizationRepairError("compiled G8R1 decoded result changed exact type")
        if type(self.admission_receipt) is not SameClassRealizationRepairAdmissionReceiptV1:
            raise SameClassRealizationRepairError("compiled G8R1 admission receipt changed exact type")
        if _encode_packet(self.program, self.source_binding) != self.packet:
            raise SameClassRealizationRepairError("compiled G8R1 program does not re-encode exact packet")
        if self.decoded.receipt.packet_sha256 != _sha256(self.packet) or self.decoded.receipt.packet_bytes != len(
            self.packet
        ):
            raise SameClassRealizationRepairError("compiled G8R1 packet differs from decode receipt")
        if self.decoded.receipt.source_binding_sha256 != self.source_binding.binding_sha256:
            raise SameClassRealizationRepairError("compiled G8R1 source differs from decode receipt")
        if (
            self.decoded.receipt.run_count != self.program.run_count
            or self.decoded.receipt.expanded_cell_count != self.program.expanded_cell_count
        ):
            raise SameClassRealizationRepairError("decoded G8R1 ownership counts differ from compiled program")
        expected_scorer_mask = _expected_scorer_ownership_mask(
            self.program,
            self.source_binding.source_pair_ids,
        )
        if not np.array_equal(self.decoded.owned_scorer_mask, expected_scorer_mask):
            raise SameClassRealizationRepairError("decoded G8R1 scorer ownership mask differs from packet program")
        if self.admission_receipt.packet_sha256 != self.decoded.receipt.packet_sha256:
            raise SameClassRealizationRepairError("G8R1 admission and decode packet identities diverged")
        if self.admission_receipt.decode_receipt_sha256 != self.decoded.receipt.receipt_sha256:
            raise SameClassRealizationRepairError("G8R1 admission did not bind exact decode receipt")
        if (
            self.admission_receipt.source_binding_sha256 != self.source_binding.binding_sha256
            or self.admission_receipt.source_pair_ids != self.source_binding.source_pair_ids
        ):
            raise SameClassRealizationRepairError("G8R1 admission differs from exact compiled source")
        if (
            self.admission_receipt.run_count != self.program.run_count
            or self.admission_receipt.admitted_cell_count != self.program.expanded_cell_count
        ):
            raise SameClassRealizationRepairError("G8R1 admission counts differ from compiled program")


def compile_same_class_realization_repair(
    program: SameClassRealizationRepairProgramV1,
    *,
    surface: SameClassRealizationRepairSurfaceV1,
    target_custody: EncoderOnlyExactTargetLabelCustodyV1,
) -> CompiledSameClassRealizationRepairV1:
    """Admit same-class rows, compile counted bytes, and decode with no scorer."""

    if type(program) is not SameClassRealizationRepairProgramV1:
        raise SameClassRealizationRepairError("G8R1 compile requires exact program type")
    if type(surface) is not SameClassRealizationRepairSurfaceV1:
        raise SameClassRealizationRepairError("G8R1 compile requires exact source surface")
    if type(target_custody) is not EncoderOnlyExactTargetLabelCustodyV1:
        raise SameClassRealizationRepairError("G8R1 compile requires exact encoder-only target custody")
    if target_custody.source_pair_ids != surface.source_pair_ids:
        raise SameClassRealizationRepairError("target custody pair IDs differ from live G source")
    pair_to_index = {pair_id: index for index, pair_id in enumerate(surface.source_pair_ids)}
    for run in program.runs:
        try:
            pair_index = pair_to_index[run.source_pair_id]
        except KeyError as exc:
            raise SameClassRealizationRepairError("repair run addresses a foreign source pair") from exc
        for scorer_col in range(run.scorer_col_start, run.scorer_col_stop):
            current_class = int(surface.g_semantic_labels[pair_index, run.scorer_row, scorer_col])
            target_class = int(target_custody.target_labels[pair_index, run.scorer_row, scorer_col])
            if current_class != run.semantic_class:
                raise SameClassRealizationRepairError("repair run semantic class differs from current G semantic label")
            if target_class != current_class:
                raise SameClassRealizationRepairError(
                    "same-class admission requires current G semantic label to equal target label"
                )

    source = derive_same_class_realization_repair_source_binding(surface)
    packet = _encode_packet(program, source)
    parsed = parse_same_class_realization_repair_packet(
        packet,
        expected_source_binding=source,
    )
    if parsed.program != program:
        raise SameClassRealizationRepairError("G8R1 compile parse-back changed exact program")
    decoded = decode_same_class_realization_repair_packet(packet, surface=surface)
    admission = SameClassRealizationRepairAdmissionReceiptV1(
        schema=ADMISSION_SCHEMA,
        source_pair_ids=source.source_pair_ids,
        source_binding_sha256=source.binding_sha256,
        packet_sha256=_sha256(packet),
        decode_receipt_sha256=decoded.receipt.receipt_sha256,
        target_custody_binding_sha256=target_custody.binding_sha256,
        target_labels_sha256=target_custody.target_labels_sha256,
        run_count=program.run_count,
        admitted_cell_count=program.expanded_cell_count,
    )
    return CompiledSameClassRealizationRepairV1(
        packet=packet,
        program=program,
        source_binding=source,
        decoded=decoded,
        admission_receipt=admission,
    )


@dataclass(frozen=True, slots=True)
class SameClassRealizationRepairPGAProposalV1:
    """Concrete replacement G-section bytes plus the exact post-G8 Y1."""

    surface: SameClassRealizationRepairSurfaceV1
    semantic_g_packet: bytes
    compiled_repair: CompiledSameClassRealizationRepairV1
    composite_g_section: bytes
    parsed_composite_g_section: ParsedSameClassCompositeGSectionV1
    monolithic_receiver_branch_required: Literal[True] = True
    existing_a3_rebind_to_post_g8_y1_required: Literal[True] = True
    scorer_invoked: Literal[False] = False
    exact_eval_invoked: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if type(self.surface) is not SameClassRealizationRepairSurfaceV1:
            raise SameClassRealizationRepairError("PGA proposal surface changed exact type")
        semantic = _require_packet_bytes(
            self.semantic_g_packet,
            "PGA proposal semantic G packet",
            magic=SEMANTIC_G_PACKET_MAGIC,
        )
        if type(self.compiled_repair) is not CompiledSameClassRealizationRepairV1:
            raise SameClassRealizationRepairError("PGA proposal repair changed exact compiled type")
        if type(self.parsed_composite_g_section) is not ParsedSameClassCompositeGSectionV1:
            raise SameClassRealizationRepairError("PGA proposal composite parse changed exact type")
        _require_packet_bytes(
            self.composite_g_section,
            "PGA proposal composite G section",
            magic=COMPOSITE_G_SECTION_MAGIC,
        )
        source = derive_same_class_realization_repair_source_binding(self.surface)
        if self.compiled_repair.source_binding != source:
            raise SameClassRealizationRepairError("PGA proposal repair differs from exact live P/G surface")
        if (
            _sha256(semantic) != self.surface.g_section_identity.payload_sha256
            or len(semantic) != self.surface.g_section_identity.payload_bytes
        ):
            raise SameClassRealizationRepairError("PGA proposal semantic G bytes differ from bound inner section")
        parsed = self.parsed_composite_g_section
        if (
            parsed.section_bytes != self.composite_g_section
            or parsed.semantic_g_packet != semantic
            or parsed.repair_packet != self.compiled_repair.packet
        ):
            raise SameClassRealizationRepairError("PGA proposal composite bytes differ from exact inner packets")
        truth = {
            "monolithic_receiver_branch_required": True,
            "existing_a3_rebind_to_post_g8_y1_required": True,
            "scorer_invoked": False,
            "exact_eval_invoked": False,
            "research_only": True,
        }
        if any(getattr(self, field) is not expected for field, expected in truth.items()):
            raise SameClassRealizationRepairError("PGA proposal truth labels became permissive")

    @property
    def replacement_g_section_payload(self) -> bytes:
        """Concrete bytes for ``TaskspaceSectionBundleV1.generative_correction_packet``."""

        return self.composite_g_section

    @property
    def replacement_g_section_sha256(self) -> str:
        return self.parsed_composite_g_section.section_sha256

    @property
    def replacement_g_section_bytes(self) -> int:
        return len(self.composite_g_section)

    @property
    def raw_g_section_delta_bytes(self) -> int:
        return len(self.composite_g_section) - len(self.semantic_g_packet)

    @property
    def post_g8_corrected_y1(self) -> np.ndarray:
        return self.compiled_repair.decoded.camera_y1


def compile_same_class_realization_repair_for_pga_sections(
    program: SameClassRealizationRepairProgramV1,
    *,
    predictor_section_payload: bytes,
    semantic_g_section_payload: bytes,
    predictor_surface: PredictorCameraPairSurfaceV1,
    predictor_codec_id: str,
    semantic_g_codec_id: str,
    target_custody: EncoderOnlyExactTargetLabelCustodyV1,
) -> SameClassRealizationRepairPGAProposalV1:
    """Compile G8 and emit concrete composite-G bytes for the whole allocator."""

    surface = build_same_class_realization_repair_surface_from_pga_sections(
        predictor_section_payload=predictor_section_payload,
        semantic_g_section_payload=semantic_g_section_payload,
        predictor_surface=predictor_surface,
        predictor_codec_id=predictor_codec_id,
        semantic_g_codec_id=semantic_g_codec_id,
    )
    compiled = compile_same_class_realization_repair(
        program,
        surface=surface,
        target_custody=target_custody,
    )
    composite = compose_same_class_realization_repair_g_section(
        semantic_g_section_payload,
        compiled.packet,
    )
    parsed = parse_same_class_realization_repair_g_section(composite)
    return SameClassRealizationRepairPGAProposalV1(
        surface=surface,
        semantic_g_packet=semantic_g_section_payload,
        compiled_repair=compiled,
        composite_g_section=composite,
        parsed_composite_g_section=parsed,
    )


@dataclass(frozen=True, slots=True)
class DecodedSameClassRealizationRepairPGAIntegrationV1:
    """Receiver-side composite-G transform result before conditional A."""

    parsed_composite_g_section: ParsedSameClassCompositeGSectionV1
    surface: SameClassRealizationRepairSurfaceV1
    decoded_repair: DecodedSameClassRealizationRepairV1

    def __post_init__(self) -> None:
        if type(self.parsed_composite_g_section) is not ParsedSameClassCompositeGSectionV1:
            raise SameClassRealizationRepairError("PGA integration composite parse changed exact type")
        if type(self.surface) is not SameClassRealizationRepairSurfaceV1:
            raise SameClassRealizationRepairError("PGA integration surface changed exact type")
        if type(self.decoded_repair) is not DecodedSameClassRealizationRepairV1:
            raise SameClassRealizationRepairError("PGA integration decode changed exact type")
        source = derive_same_class_realization_repair_source_binding(self.surface)
        if self.decoded_repair.receipt.source_binding_sha256 != source.binding_sha256:
            raise SameClassRealizationRepairError("PGA integration decode differs from exact live surface")
        if (
            self.surface.g_section_identity.payload_sha256 != self.parsed_composite_g_section.semantic_g_packet_sha256
            or self.surface.g_section_identity.payload_bytes != len(self.parsed_composite_g_section.semantic_g_packet)
        ):
            raise SameClassRealizationRepairError("PGA integration surface differs from counted inner semantic G")
        if self.decoded_repair.receipt.packet_sha256 != self.parsed_composite_g_section.repair_packet_sha256:
            raise SameClassRealizationRepairError("PGA integration decode differs from counted inner G8R1")

    @property
    def post_g8_corrected_y1(self) -> np.ndarray:
        return self.decoded_repair.camera_y1

    @property
    def unchanged_g_semantic_labels(self) -> np.ndarray:
        return self.decoded_repair.semantic_labels


def decode_same_class_realization_repair_from_pga_sections(
    composite_g_section: bytes,
    *,
    predictor_section_payload: bytes,
    predictor_surface: PredictorCameraPairSurfaceV1,
    predictor_codec_id: str,
    semantic_g_codec_id: str,
) -> DecodedSameClassRealizationRepairPGAIntegrationV1:
    """Apply the counted composite-G transform after semantic G/overlay decode."""

    parsed = parse_same_class_realization_repair_g_section(composite_g_section)
    surface = build_same_class_realization_repair_surface_from_pga_sections(
        predictor_section_payload=predictor_section_payload,
        semantic_g_section_payload=parsed.semantic_g_packet,
        predictor_surface=predictor_surface,
        predictor_codec_id=predictor_codec_id,
        semantic_g_codec_id=semantic_g_codec_id,
    )
    decoded = decode_same_class_realization_repair_packet(
        parsed.repair_packet,
        surface=surface,
    )
    return DecodedSameClassRealizationRepairPGAIntegrationV1(
        parsed_composite_g_section=parsed,
        surface=surface,
        decoded_repair=decoded,
    )


__all__ = [
    "A3_EXPANDED_RGB_ROW",
    "ADMISSION_SCHEMA",
    "COMPOSITE_G_SECTION_HEADER",
    "COMPOSITE_G_SECTION_MAGIC",
    "COMPOSITE_G_SECTION_VERSION",
    "PACKET_HEADER",
    "PACKET_MAGIC",
    "PACKET_VERSION",
    "PREDICTOR_PACKET_MAGIC",
    "RECEIPT_SCHEMA",
    "RUN_ROW",
    "SEMANTIC_G_PACKET_MAGIC",
    "SOURCE_SCHEMA",
    "CompiledSameClassRealizationRepairV1",
    "CountedTaskspaceSectionIdentityV1",
    "DecodedSameClassRealizationRepairPGAIntegrationV1",
    "DecodedSameClassRealizationRepairV1",
    "EncoderOnlyExactTargetLabelCustodyV1",
    "ParsedSameClassCompositeGSectionV1",
    "ParsedSameClassRealizationRepairPacketV1",
    "SameClassRealizationRepairAdmissionReceiptV1",
    "SameClassRealizationRepairError",
    "SameClassRealizationRepairPGAProposalV1",
    "SameClassRealizationRepairProgramV1",
    "SameClassRealizationRepairReceiptV1",
    "SameClassRealizationRepairRunV1",
    "SameClassRealizationRepairSourceBindingV1",
    "SameClassRealizationRepairSurfaceV1",
    "SameClassSemanticRoleV1",
    "TaskspaceSectionRoleV1",
    "build_same_class_realization_repair_surface_from_pga_sections",
    "compile_same_class_realization_repair",
    "compile_same_class_realization_repair_for_pga_sections",
    "compose_same_class_realization_repair_g_section",
    "decode_same_class_realization_repair_from_pga_sections",
    "decode_same_class_realization_repair_packet",
    "derive_same_class_realization_repair_source_binding",
    "parse_same_class_realization_repair_g_section",
    "parse_same_class_realization_repair_packet",
    "parse_same_class_realization_repair_receipt",
]
