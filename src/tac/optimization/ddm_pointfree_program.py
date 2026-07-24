# SPDX-License-Identifier: MIT
"""Point-free description programs over settled DDM receiver primitives.

PF1 is deliberately a *description* layer, not a synthesizer.  It mechanically
compiles three existing, strictly parseable description families:

* G1 EVENT/CENTROID/SHAPE movable worldsheets;
* V15 row-band scorer-template banks; and
* DV2 SDWL1 fact sentences.

The generic interpreter and the fixed recipe traces are rule-118 code.  Every
video-derived section, literal, shared-library entry, and program opcode is in
the returned counted program string.  No search over descriptions is performed:
the structural formulation is one bounded, deterministic anti-unification pass
over exact repetitions and deterministic framing.
"""

from __future__ import annotations

import hashlib
import lzma
import struct
import zlib
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any, Final

import brotli
import numpy as np

from tac.lie import _se3_numpy
from tac.optimization.ddm_continuous_paint_ceiling import apply_global_channel_statistics
from tac.optimization.ddm_dv2_sdwl1 import (
    FactInventory,
    SentenceLayout,
    SentenceOptions,
    TemporalMode,
    _expected_section_tags,
    _frame_sections,
    _options_from_schema,
    _parse_canonical_json,
    _parse_sections,
    _reassemble_numeric,
    _semantic_sha256,
    _temporal_decode,
    decode_sentence,
    decompress_outer_payload,
    serialize_sentence,
)
from tac.optimization.direct_description_carrier_compose import (
    REALIZATION_PAINT_ORDER,
    LaneDriftKnotV1,
    LanePeriodicProgramV1,
    RowBandScorerTemplateV1,
    ScorerSolvedTemplateBankV1,
    StructuredRoleLayerV1,
    _apply_lane_predictor_programs,
    decode_scorer_solved_template_bank,
    encode_scorer_solved_template_bank,
)
from tac.optimization.direct_description_g1_worldsheet import (
    CODEC_NAMES as G1_CODEC_NAMES,
)
from tac.optimization.direct_description_g1_worldsheet import (
    PRODUCTION_IDS as G1_PRODUCTION_IDS,
)
from tac.optimization.direct_description_g1_worldsheet import (
    _decode_envelope as _decode_g1_envelope,
)
from tac.optimization.direct_description_g1_worldsheet import (
    _decompress as _decompress_g1,
)
from tac.optimization.direct_description_g1_worldsheet import decode_g1_movable_worldsheet

PROGRAM_MAGIC: Final = b"PF1"
CODED_MAGIC: Final = b"RC1"
FLAT_BUNDLE_MAGIC: Final = b"FL1"
MAX_PROGRAM_BYTES: Final = 128 << 20
MAX_SECTIONS: Final = 64

_G1_ENVELOPE_HEADER: Final = b"G1S1\x03"
_G1_SECTION_HEADER: Final = struct.Struct("<BBII")
_V15_LIBRARY_ROW: Final = struct.Struct(">BBHHBBB")
_V15_ROLE_TO_WIRE: Final = {role: index for index, role in enumerate(REALIZATION_PAINT_ORDER)}
_V15_WIRE_TO_ROLE: Final = {value: key for key, value in _V15_ROLE_TO_WIRE.items()}
_V15_APPLICATION_TO_WIRE: Final = {"fill": 0, "inner_boundary": 1}
_V15_WIRE_TO_APPLICATION: Final = {value: key for key, value in _V15_APPLICATION_TO_WIRE.items()}
_LAYOUT_TO_WIRE: Final = {
    SentenceLayout.MONOLITHIC: 0,
    SentenceLayout.TYPED_SECTION: 1,
    SentenceLayout.STRATUM_SECTION: 2,
}
_WIRE_TO_LAYOUT: Final = {value: key for key, value in _LAYOUT_TO_WIRE.items()}
_TEMPORAL_TO_WIRE: Final = {TemporalMode.ABSOLUTE: 0, TemporalMode.CAUSAL_DELTA: 1}
_WIRE_TO_TEMPORAL: Final = {value: key for key, value in _TEMPORAL_TO_WIRE.items()}


class PointFreeProgramError(ValueError):
    """Raised when a PF1 program is invalid, noncanonical, or changes meaning."""


class Recipe(IntEnum):
    """Fixed generic interpreter recipes."""

    G1_WORLDSHEET = 1
    V15_TEMPLATE_BANK = 2
    DV2_SENTENCE = 3
    BUNDLE = 4


class Formulation(IntEnum):
    """Three distinct, preregisterable description formulations."""

    LITERAL = 1
    SHARED_LIBRARY = 2
    STRUCTURAL = 3


class BasisPrimitive(StrEnum):
    """Measured receiver primitives admitted into the PF1 basis."""

    COMPOSE = "compose"
    EVENT_SCAN = "event_scan"
    EVENT_FOLD = "event_fold"
    XI_ADVECT = "xi_advect"
    STRATUM_MASK = "stratum_mask"
    LANE_PROGRAM_EVAL = "lane_program_eval"
    TEMPLATE_APPLY = "template_apply"
    CHANNEL_AFFINE = "channel_affine"
    ARITHMETIC_DECODE = "arithmetic_decode"
    CANONICAL_REEMIT = "canonical_reemit"


@dataclass(frozen=True, slots=True)
class BasisSpec:
    primitive: BasisPrimitive
    rank_rule: str
    provenance: tuple[str, ...]


MEASURED_BASIS: Final = (
    BasisSpec(
        BasisPrimitive.COMPOSE,
        "rank_preserving_function_composition",
        ("src/tac/optimization/direct_description_carrier_compose.py",),
    ),
    BasisSpec(
        BasisPrimitive.EVENT_SCAN,
        "leading_pair_axis_scan",
        ("src/tac/optimization/direct_description_g1_worldsheet.py",),
    ),
    BasisSpec(
        BasisPrimitive.EVENT_FOLD,
        "leading_pair_axis_fold_to_masks",
        ("src/tac/optimization/direct_description_g1_worldsheet.py",),
    ),
    BasisSpec(
        BasisPrimitive.XI_ADVECT,
        "arbitrary_leading_axes_then_core_6_to_4x4",
        (
            "src/tac/lie/_se3_numpy.py",
            "src/tac/optimization/direct_description_carrier_compose.py",
        ),
    ),
    BasisSpec(
        BasisPrimitive.STRATUM_MASK,
        "arbitrary_leading_axes_then_spatial_hw",
        ("src/tac/optimization/direct_description_carrier_compose.py",),
    ),
    BasisSpec(
        BasisPrimitive.LANE_PROGRAM_EVAL,
        "pair_axis_map_with_sparse_knot_interpolation",
        ("src/tac/optimization/direct_description_carrier_compose.py",),
    ),
    BasisSpec(
        BasisPrimitive.TEMPLATE_APPLY,
        "arbitrary_leading_axes_then_spatial_hw_to_hw3",
        ("src/tac/optimization/direct_description_carrier_compose.py",),
    ),
    BasisSpec(
        BasisPrimitive.CHANNEL_AFFINE,
        "arbitrary_leading_axes_then_rgb3",
        (
            "src/tac/optimization/ddm_continuous_paint_ceiling.py",
            "tools/measure_ddm_pa1_posenet_amplitude_twin.py",
        ),
    ),
    BasisSpec(
        BasisPrimitive.ARITHMETIC_DECODE,
        "typed_numeric_sections_to_pair_record_tensor",
        ("src/tac/optimization/ddm_dv2_sdwl1.py",),
    ),
    BasisSpec(
        BasisPrimitive.CANONICAL_REEMIT,
        "semantic_value_to_original_description_bytes",
        (
            "src/tac/optimization/direct_description_g1_worldsheet.py",
            "src/tac/optimization/direct_description_carrier_compose.py",
            "src/tac/optimization/ddm_dv2_sdwl1.py",
        ),
    ),
)

RECIPE_TRACES: Final = {
    Recipe.G1_WORLDSHEET: (
        BasisPrimitive.EVENT_SCAN,
        BasisPrimitive.EVENT_FOLD,
        BasisPrimitive.CANONICAL_REEMIT,
    ),
    Recipe.V15_TEMPLATE_BANK: (
        BasisPrimitive.STRATUM_MASK,
        BasisPrimitive.TEMPLATE_APPLY,
        BasisPrimitive.CHANNEL_AFFINE,
        BasisPrimitive.CANONICAL_REEMIT,
    ),
    Recipe.DV2_SENTENCE: (
        BasisPrimitive.ARITHMETIC_DECODE,
        BasisPrimitive.EVENT_SCAN,
        BasisPrimitive.CANONICAL_REEMIT,
    ),
    Recipe.BUNDLE: (BasisPrimitive.COMPOSE, BasisPrimitive.CANONICAL_REEMIT),
}


@dataclass(frozen=True, slots=True)
class SharedLiteralLibrary:
    """A bounded exact-repeat library learned without combinatorial search."""

    entries: tuple[bytes, ...]
    references: tuple[int | None, ...]
    inline_literals: tuple[bytes, ...]

    @property
    def video_derived_library_bytes(self) -> int:
        return sum(len(value) for value in self.entries)


@dataclass(frozen=True, slots=True)
class CompiledProgram:
    """One canonical counted PF1 program string."""

    recipe: Recipe
    formulation: Formulation
    program: bytes
    source_bytes: int
    source_sha256: str
    video_derived_library_bytes: int
    operator_trace: tuple[BasisPrimitive, ...]


@dataclass(frozen=True, slots=True)
class CodedPayload:
    """One real-coded and self-describing byte string."""

    codec: str
    raw_bytes: int
    coded_payload_bytes: int
    framed_bytes: int
    framed_sha256: str
    payload: bytes


def _put_uleb(output: bytearray, value: int) -> None:
    if isinstance(value, bool) or value < 0:
        raise PointFreeProgramError("PF1 ULEB values must be nonnegative integers")
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return


def _get_uleb(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(data) or shift > 63:
            raise PointFreeProgramError("PF1 ULEB stream is truncated or overlong")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if _uleb(value) != data[offset - len(_uleb(value)) : offset]:
                raise PointFreeProgramError("PF1 ULEB is not minimally encoded")
            return value, offset
        shift += 7


def _uleb(value: int) -> bytes:
    output = bytearray()
    _put_uleb(output, value)
    return bytes(output)


def _take(data: bytes, offset: int, size: int, label: str) -> tuple[bytes, int]:
    if size < 0 or size > len(data) - offset:
        raise PointFreeProgramError(f"PF1 {label} is truncated")
    return data[offset : offset + size], offset + size


def _wrap_program(recipe: Recipe, formulation: Formulation, body: bytes) -> bytes:
    if len(body) > MAX_PROGRAM_BYTES:
        raise PointFreeProgramError("PF1 body exceeds the bounded program size")
    tag = ((int(recipe) & 0x0F) << 4) | (int(formulation) & 0x0F)
    return PROGRAM_MAGIC + bytes([tag]) + _uleb(len(body)) + body


def _unwrap_program(program: bytes) -> tuple[Recipe, Formulation, bytes]:
    if not isinstance(program, bytes) or len(program) < 5 or program[:3] != PROGRAM_MAGIC:
        raise PointFreeProgramError("PF1 program magic is invalid")
    try:
        recipe = Recipe(program[3] >> 4)
        formulation = Formulation(program[3] & 0x0F)
    except ValueError as exc:
        raise PointFreeProgramError("PF1 program recipe/formulation tag is unknown") from exc
    size, offset = _get_uleb(program, 4)
    body, offset = _take(program, offset, size, "body")
    if offset != len(program):
        raise PointFreeProgramError("PF1 program has trailing bytes")
    if _wrap_program(recipe, formulation, body) != program:
        raise PointFreeProgramError("PF1 program is not canonical on parse-back")
    return recipe, formulation, body


def learn_shared_literals(
    literals: Sequence[bytes],
    *,
    min_occurrences: int = 2,
    min_length: int = 1,
    max_entries: int = 256,
) -> SharedLiteralLibrary:
    """Learn exact repeated literals in one bounded deterministic pass.

    The candidates are equality classes, not substrings.  Complexity is linear
    in input bytes plus a bounded sort; no grammar search or alternative parse
    tree is explored.
    """

    values = tuple(bytes(value) for value in literals)
    if not 2 <= min_occurrences <= 255:
        raise PointFreeProgramError("shared-library min_occurrences must be in [2,255]")
    if not 1 <= min_length <= 1 << 20 or not 1 <= max_entries <= 256:
        raise PointFreeProgramError("shared-library bounds are invalid")
    counts = Counter(values)
    candidates = [
        value
        for value, count in counts.items()
        if count >= min_occurrences and len(value) >= min_length
    ]
    candidates.sort(
        key=lambda value: (
            -((counts[value] - 1) * len(value)),
            hashlib.sha256(value).digest(),
            value,
        )
    )
    entries = tuple(candidates[:max_entries])
    index = {value: position for position, value in enumerate(entries)}
    references = tuple(index.get(value) for value in values)
    inline = tuple(value for value, reference in zip(values, references, strict=True) if reference is None)
    return SharedLiteralLibrary(entries=entries, references=references, inline_literals=inline)


def compose_pointfree(*operations: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Return point-free left-to-right composition without named intermediates."""

    if not operations:
        raise PointFreeProgramError("point-free composition requires at least one operation")
    if any(not callable(operation) for operation in operations):
        raise PointFreeProgramError("point-free composition operands must be callable")

    def composed(value: Any) -> Any:
        for operation in operations:
            value = operation(value)
        return value

    return composed


def stratum_mask(labels: np.ndarray, class_id: int) -> np.ndarray:
    """Rank-polymorphic exact class mask over the final spatial axes."""

    value = np.asarray(labels)
    if value.ndim < 2 or not np.issubdtype(value.dtype, np.integer):
        raise PointFreeProgramError("stratum masks require an integer array with spatial axes")
    if isinstance(class_id, bool) or not 0 <= class_id <= 255:
        raise PointFreeProgramError("stratum class ID is outside uint8")
    return np.ascontiguousarray(value == class_id)


def xi_advect(xi: np.ndarray) -> np.ndarray:
    """Rank-polymorphic NumPy-authority ``se(3) -> SE(3)`` map."""

    value = np.asarray(xi)
    if value.ndim < 1 or value.shape[-1] != 6 or not np.issubdtype(value.dtype, np.floating):
        raise PointFreeProgramError("xi advection requires floating (...,6) twists")
    if not np.isfinite(value).all():
        raise PointFreeProgramError("xi advection requires finite twists")
    return np.ascontiguousarray(_se3_numpy.exp_se3(value))


def channel_affine(
    rgb_u8: np.ndarray,
    scale: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Rank-polymorphic PT1x/PA1 per-channel affine with exact uint8 rounding."""

    source = np.asarray(rgb_u8)
    scale_value = np.asarray(scale, dtype=np.float32)
    offset_value = np.asarray(offset, dtype=np.float32)
    if source.ndim < 1 or source.shape[-1] != 3 or source.dtype != np.uint8:
        raise PointFreeProgramError("channel affine requires uint8 (...,3) RGB")
    if scale_value.shape != (3,) or offset_value.shape != (3,):
        raise PointFreeProgramError("channel affine scale/offset must be RGB three-vectors")
    if not np.isfinite(scale_value).all() or not np.isfinite(offset_value).all():
        raise PointFreeProgramError("channel affine scale/offset must be finite")
    if source.ndim == 4:
        return apply_global_channel_statistics(source, scale_value, offset_value)
    matched = source.astype(np.float32) * scale_value + offset_value
    return np.ascontiguousarray(np.clip(np.rint(matched), 0, 255).astype(np.uint8))


def apply_template(mask: np.ndarray, patch_rgb_u8: np.ndarray) -> np.ndarray:
    """Rank-polymorphic tiling of one measured V15 RGB template over a mask."""

    active = np.asarray(mask)
    patch = np.asarray(patch_rgb_u8)
    if active.ndim < 2 or active.dtype != np.bool_:
        raise PointFreeProgramError("template application requires boolean (...,H,W) masks")
    if patch.ndim != 3 or patch.shape[-1] != 3 or patch.dtype != np.uint8:
        raise PointFreeProgramError("template application requires uint8 (h,w,3) patches")
    height, width = active.shape[-2:]
    rows = np.arange(height, dtype=np.intp) % patch.shape[0]
    columns = np.arange(width, dtype=np.intp) % patch.shape[1]
    field = patch[rows[:, None], columns[None, :]]
    return np.ascontiguousarray(np.where(active[..., None], field, 0).astype(np.uint8))


def evaluate_lane_programs(
    layers: Sequence[StructuredRoleLayerV1],
    programs: Sequence[LanePeriodicProgramV1],
    knots: Sequence[LaneDriftKnotV1],
    *,
    pose6_codes: np.ndarray,
    source_pair_start: int,
) -> tuple[StructuredRoleLayerV1, ...]:
    """Expose the settled Lane xi/scan evaluator as a PF1 basis operation."""

    return _apply_lane_predictor_programs(
        layers,
        programs,
        knots,
        pose6_codes=np.asarray(pose6_codes),
        source_pair_start=source_pair_start,
    )


def evaluate_g1_worldsheet(payload: bytes) -> np.ndarray:
    """Execute the measured G1 event scan/fold to its exact mask stack."""

    masks, _metadata = decode_g1_movable_worldsheet(payload)
    return masks


def _parse_g1_coded_sections(payload: bytes) -> tuple[bytes, ...]:
    _decode_g1_envelope(payload)
    if payload[:5] != _G1_ENVELOPE_HEADER:
        raise PointFreeProgramError("G1 source is not the measured three-production envelope")
    offset = 5
    sections: list[bytes] = []
    for _index in range(3):
        if len(payload) - offset < _G1_SECTION_HEADER.size:
            raise PointFreeProgramError("G1 source section is truncated")
        _production, _codec, _raw_size, coded_size = _G1_SECTION_HEADER.unpack_from(payload, offset)
        stop = offset + _G1_SECTION_HEADER.size + coded_size
        section, offset = _take(payload, offset, stop - offset, "G1 coded section")
        sections.append(section)
    if offset != len(payload):
        raise PointFreeProgramError("G1 source has trailing bytes")
    return tuple(sections)


def _g1_structural_body(payload: bytes) -> tuple[bytes, int]:
    sections = _parse_g1_coded_sections(payload)
    body = bytearray()
    for section in sections:
        _production, codec, _raw_size, coded_size = _G1_SECTION_HEADER.unpack_from(section)
        coded = section[_G1_SECTION_HEADER.size :]
        if len(coded) != coded_size:
            raise PointFreeProgramError("G1 coded section length differs")
        body.append(codec)
        _put_uleb(body, len(coded))
        body.extend(coded)
    return bytes(body), sum(len(section) - _G1_SECTION_HEADER.size for section in sections)


def compile_g1_worldsheet(payload: bytes, formulation: Formulation) -> CompiledProgram:
    """Mechanically compile one canonical G1 stream into PF1."""

    source = bytes(payload)
    _decode_g1_envelope(source)
    library_bytes = 0
    if formulation is Formulation.LITERAL:
        body = source
    elif formulation is Formulation.SHARED_LIBRARY:
        sections = _parse_g1_coded_sections(source)
        encoded = bytearray()
        _put_uleb(encoded, len(sections))
        for section in sections:
            _put_uleb(encoded, len(section))
            encoded.extend(section)
        body = bytes(encoded)
        library_bytes = sum(len(section) for section in sections)
    elif formulation is Formulation.STRUCTURAL:
        body, library_bytes = _g1_structural_body(source)
    else:  # pragma: no cover - IntEnum exhaustiveness
        raise PointFreeProgramError("unsupported G1 formulation")
    program = _wrap_program(Recipe.G1_WORLDSHEET, formulation, body)
    if execute_program(program) != source:
        raise PointFreeProgramError("G1 program failed byte-identical source replay")
    return CompiledProgram(
        recipe=Recipe.G1_WORLDSHEET,
        formulation=formulation,
        program=program,
        source_bytes=len(source),
        source_sha256=hashlib.sha256(source).hexdigest(),
        video_derived_library_bytes=library_bytes,
        operator_trace=RECIPE_TRACES[Recipe.G1_WORLDSHEET],
    )


def _execute_g1(formulation: Formulation, body: bytes) -> bytes:
    if formulation is Formulation.LITERAL:
        source = body
    elif formulation is Formulation.SHARED_LIBRARY:
        count, offset = _get_uleb(body, 0)
        if count != 3:
            raise PointFreeProgramError("G1 shared library must contain three productions")
        sections: list[bytes] = []
        for _index in range(count):
            size, offset = _get_uleb(body, offset)
            section, offset = _take(body, offset, size, "G1 shared section")
            sections.append(section)
        if offset != len(body):
            raise PointFreeProgramError("G1 shared-library body has trailing bytes")
        source = _G1_ENVELOPE_HEADER + b"".join(sections)
    elif formulation is Formulation.STRUCTURAL:
        offset = 0
        sections = []
        production_ids = (
            G1_PRODUCTION_IDS["EVENT"],
            G1_PRODUCTION_IDS["CENTROID"],
            G1_PRODUCTION_IDS["SHAPE"],
        )
        for production_id in production_ids:
            codec_raw, offset = _take(body, offset, 1, "G1 codec")
            codec_id = codec_raw[0]
            if codec_id not in G1_CODEC_NAMES:
                raise PointFreeProgramError("G1 structural program names an unknown codec")
            size, offset = _get_uleb(body, offset)
            coded, offset = _take(body, offset, size, "G1 structural coded payload")
            try:
                raw = _decompress_g1(G1_CODEC_NAMES[codec_id], coded)
            except (brotli.error, lzma.LZMAError, zlib.error) as exc:
                raise PointFreeProgramError("G1 structural coded payload is invalid") from exc
            sections.append(
                _G1_SECTION_HEADER.pack(
                    production_id,
                    codec_id,
                    len(raw),
                    len(coded),
                )
                + coded
            )
        if offset != len(body):
            raise PointFreeProgramError("G1 structural body has trailing bytes")
        source = _G1_ENVELOPE_HEADER + b"".join(sections)
    else:  # pragma: no cover
        raise PointFreeProgramError("unsupported G1 execution formulation")
    _decode_g1_envelope(source)
    return source


def _v15_library_body(bank: ScorerSolvedTemplateBankV1) -> tuple[bytes, int]:
    literals = tuple(row.rgb_u8 for row in bank.templates)
    learned = learn_shared_literals(literals)
    entries = list(learned.entries)
    for literal in learned.inline_literals:
        if literal not in entries:
            entries.append(literal)
    entries.sort()
    index = {value: position for position, value in enumerate(entries)}
    body = bytearray()
    _put_uleb(body, len(entries))
    for entry in entries:
        _put_uleb(body, len(entry))
        body.extend(entry)
    _put_uleb(body, len(bank.templates))
    for row in bank.templates:
        body.extend(
            _V15_LIBRARY_ROW.pack(
                _V15_ROLE_TO_WIRE[row.role],
                _V15_APPLICATION_TO_WIRE[row.application],
                row.scorer_row_start,
                row.scorer_row_stop,
                row.patch_height,
                row.patch_width,
                index[row.rgb_u8],
            )
        )
    return bytes(body), sum(len(entry) for entry in entries)


def _v15_structural_body(bank: ScorerSolvedTemplateBankV1) -> tuple[bytes, int]:
    grouped: dict[tuple[str, str, int, int, bytes], list[tuple[int, int]]] = {}
    for row in bank.templates:
        key = (row.role, row.application, row.patch_height, row.patch_width, row.rgb_u8)
        grouped.setdefault(key, []).append((row.scorer_row_start, row.scorer_row_stop))
    body = bytearray()
    _put_uleb(body, len(grouped))
    library_bytes = 0
    for (role, application, patch_height, patch_width, rgb), bands in grouped.items():
        body.extend(
            bytes(
                (
                    _V15_ROLE_TO_WIRE[role],
                    _V15_APPLICATION_TO_WIRE[application],
                    patch_height,
                    patch_width,
                )
            )
        )
        _put_uleb(body, len(rgb))
        body.extend(rgb)
        _put_uleb(body, len(bands))
        for start, stop in bands:
            body.extend(struct.pack(">HH", start, stop))
        library_bytes += len(rgb)
    return bytes(body), library_bytes


def compile_v15_template_bank(payload: bytes, formulation: Formulation) -> CompiledProgram:
    """Compile a canonical V15 row-band template bank."""

    source = bytes(payload)
    bank = decode_scorer_solved_template_bank(source)
    if bank is None:
        raise PointFreeProgramError("V15 PF1 compiler requires a nonempty template bank")
    library_bytes = 0
    if formulation is Formulation.LITERAL:
        body = source
    elif formulation is Formulation.SHARED_LIBRARY:
        body, library_bytes = _v15_library_body(bank)
    elif formulation is Formulation.STRUCTURAL:
        body, library_bytes = _v15_structural_body(bank)
    else:  # pragma: no cover
        raise PointFreeProgramError("unsupported V15 formulation")
    program = _wrap_program(Recipe.V15_TEMPLATE_BANK, formulation, body)
    if execute_program(program) != source:
        raise PointFreeProgramError("V15 program failed byte-identical source replay")
    return CompiledProgram(
        recipe=Recipe.V15_TEMPLATE_BANK,
        formulation=formulation,
        program=program,
        source_bytes=len(source),
        source_sha256=hashlib.sha256(source).hexdigest(),
        video_derived_library_bytes=library_bytes,
        operator_trace=RECIPE_TRACES[Recipe.V15_TEMPLATE_BANK],
    )


def _execute_v15_library(body: bytes) -> bytes:
    entry_count, offset = _get_uleb(body, 0)
    if not 1 <= entry_count <= 64:
        raise PointFreeProgramError("V15 shared library entry count is invalid")
    entries: list[bytes] = []
    for _index in range(entry_count):
        size, offset = _get_uleb(body, offset)
        entry, offset = _take(body, offset, size, "V15 library entry")
        entries.append(entry)
    count, offset = _get_uleb(body, offset)
    if not 1 <= count <= 64:
        raise PointFreeProgramError("V15 shared-library template count is invalid")
    rows: list[RowBandScorerTemplateV1] = []
    for _index in range(count):
        raw, offset = _take(body, offset, _V15_LIBRARY_ROW.size, "V15 library record")
        role, application, start, stop, height, width, reference = _V15_LIBRARY_ROW.unpack(raw)
        if role not in _V15_WIRE_TO_ROLE or application not in _V15_WIRE_TO_APPLICATION:
            raise PointFreeProgramError("V15 shared-library role/application is unknown")
        if reference >= len(entries):
            raise PointFreeProgramError("V15 shared-library reference is out of range")
        rows.append(
            RowBandScorerTemplateV1(
                _V15_WIRE_TO_ROLE[role],
                _V15_WIRE_TO_APPLICATION[application],
                start,
                stop,
                height,
                width,
                entries[reference],
            )
        )
    if offset != len(body):
        raise PointFreeProgramError("V15 shared-library body has trailing bytes")
    return encode_scorer_solved_template_bank(ScorerSolvedTemplateBankV1(tuple(rows)))


def _execute_v15_structural(body: bytes) -> bytes:
    group_count, offset = _get_uleb(body, 0)
    if not 1 <= group_count <= 64:
        raise PointFreeProgramError("V15 structural group count is invalid")
    rows: list[RowBandScorerTemplateV1] = []
    for _group in range(group_count):
        raw, offset = _take(body, offset, 4, "V15 structural group header")
        role, application, height, width = raw
        if role not in _V15_WIRE_TO_ROLE or application not in _V15_WIRE_TO_APPLICATION:
            raise PointFreeProgramError("V15 structural role/application is unknown")
        rgb_size, offset = _get_uleb(body, offset)
        rgb, offset = _take(body, offset, rgb_size, "V15 structural RGB literal")
        band_count, offset = _get_uleb(body, offset)
        if not 1 <= band_count <= 64:
            raise PointFreeProgramError("V15 structural band count is invalid")
        for _band in range(band_count):
            band, offset = _take(body, offset, 4, "V15 structural row band")
            start, stop = struct.unpack(">HH", band)
            rows.append(
                RowBandScorerTemplateV1(
                    _V15_WIRE_TO_ROLE[role],
                    _V15_WIRE_TO_APPLICATION[application],
                    start,
                    stop,
                    height,
                    width,
                    rgb,
                )
            )
    if offset != len(body):
        raise PointFreeProgramError("V15 structural body has trailing bytes")
    return encode_scorer_solved_template_bank(ScorerSolvedTemplateBankV1(tuple(rows)))


def _execute_v15(formulation: Formulation, body: bytes) -> bytes:
    if formulation is Formulation.LITERAL:
        source = body
    elif formulation is Formulation.SHARED_LIBRARY:
        source = _execute_v15_library(body)
    elif formulation is Formulation.STRUCTURAL:
        source = _execute_v15_structural(body)
    else:  # pragma: no cover
        raise PointFreeProgramError("unsupported V15 execution formulation")
    if decode_scorer_solved_template_bank(source) is None:
        raise PointFreeProgramError("V15 execution produced an empty bank")
    return source


def _encode_dv2_library(sections: Sequence[tuple[bytes, bytes]]) -> tuple[bytes, int]:
    body = bytearray()
    _put_uleb(body, len(sections))
    for tag, payload in sections:
        if len(tag) != 4:
            raise PointFreeProgramError("DV2 section tags must be four bytes")
        body.extend(tag)
        _put_uleb(body, len(payload))
        body.extend(payload)
    return bytes(body), sum(len(payload) for _tag, payload in sections)


def _counterfactual_bits(options: SentenceOptions) -> int:
    return (
        int(options.explicit_frame_indices)
        | (int(options.repeated_provenance) << 1)
        | (int(options.redundant_event_masks) << 2)
        | (int(options.split_topology_vocabulary) << 3)
    )


def _options_from_wire(layout: int, temporal: int, flags: int) -> SentenceOptions:
    if layout not in _WIRE_TO_LAYOUT or temporal not in _WIRE_TO_TEMPORAL or flags & ~0x0F:
        raise PointFreeProgramError("DV2 structural options are unknown")
    return SentenceOptions(
        layout=_WIRE_TO_LAYOUT[layout],
        temporal_mode=_WIRE_TO_TEMPORAL[temporal],
        explicit_frame_indices=bool(flags & 1),
        repeated_provenance=bool(flags & 2),
        redundant_event_masks=bool(flags & 4),
        split_topology_vocabulary=bool(flags & 8),
    )


def _encode_dv2_structural(
    inventory: FactInventory,
    options: SentenceOptions,
    sections: Sequence[tuple[bytes, bytes]],
) -> tuple[bytes, int]:
    expected = _expected_section_tags(options)
    by_tag = dict(sections)
    numeric_tags = expected[2 + int(options.repeated_provenance) :]
    # Counterfactual FIDX/EVNT are deterministic from the facts but remain in
    # the generic re-emitter.  Only the actual arithmetic fact sections cross
    # this structural wire.
    numeric_tags = [
        tag
        for tag in numeric_tags
        if tag not in {b"FIDX", b"EVNT"}
    ]
    body = bytearray()
    _put_uleb(body, inventory.pair_count)
    _put_uleb(body, inventory.source_height)
    _put_uleb(body, inventory.source_width)
    body.extend(
        (
            _LAYOUT_TO_WIRE[options.layout],
            _TEMPORAL_TO_WIRE[options.temporal_mode],
            _counterfactual_bits(options),
        )
    )
    _put_uleb(body, len(numeric_tags))
    library_bytes = 0
    for tag in numeric_tags:
        payload = by_tag[tag]
        _put_uleb(body, len(payload))
        body.extend(payload)
        library_bytes += len(payload)
    return bytes(body), library_bytes


def compile_dv2_sentence(outer_payload: bytes, formulation: Formulation) -> CompiledProgram:
    """Compile a complete outer-zlib DV2 sentence and reproduce it exactly."""

    source = bytes(outer_payload)
    inner = decompress_outer_payload(source)
    inventory = decode_sentence(inner)
    sections = _parse_sections(inner)
    schema = _parse_canonical_json(sections[1][1], "DV2 compiler schema")
    options = _options_from_schema(schema)
    library_bytes = 0
    if formulation is Formulation.LITERAL:
        body = source
    elif formulation is Formulation.SHARED_LIBRARY:
        body, library_bytes = _encode_dv2_library(sections)
    elif formulation is Formulation.STRUCTURAL:
        body, library_bytes = _encode_dv2_structural(inventory, options, sections)
    else:  # pragma: no cover
        raise PointFreeProgramError("unsupported DV2 formulation")
    program = _wrap_program(Recipe.DV2_SENTENCE, formulation, body)
    recipe, replay_formulation, replay_body = _unwrap_program(program)
    if (
        recipe is not Recipe.DV2_SENTENCE
        or _execute_dv2(replay_formulation, replay_body, validate_semantics=False) != source
    ):
        raise PointFreeProgramError("DV2 program failed byte-identical outer-payload replay")
    return CompiledProgram(
        recipe=Recipe.DV2_SENTENCE,
        formulation=formulation,
        program=program,
        source_bytes=len(source),
        source_sha256=hashlib.sha256(source).hexdigest(),
        video_derived_library_bytes=library_bytes,
        operator_trace=RECIPE_TRACES[Recipe.DV2_SENTENCE],
    )


def _execute_dv2_library(body: bytes) -> bytes:
    count, offset = _get_uleb(body, 0)
    if not 3 <= count <= MAX_SECTIONS:
        raise PointFreeProgramError("DV2 shared-library section count is invalid")
    sections: list[tuple[bytes, bytes]] = []
    seen: set[bytes] = set()
    for _index in range(count):
        tag, offset = _take(body, offset, 4, "DV2 section tag")
        size, offset = _get_uleb(body, offset)
        payload, offset = _take(body, offset, size, "DV2 section payload")
        if tag in seen:
            raise PointFreeProgramError("DV2 shared-library section tags are not unique")
        seen.add(tag)
        sections.append((tag, payload))
    if offset != len(body):
        raise PointFreeProgramError("DV2 shared-library body has trailing bytes")
    inner = _frame_sections(sections)
    decode_sentence(inner)
    return zlib.compress(inner, level=9)


def _execute_dv2_structural(body: bytes) -> bytes:
    pair_count, offset = _get_uleb(body, 0)
    source_height, offset = _get_uleb(body, offset)
    source_width, offset = _get_uleb(body, offset)
    options_raw, offset = _take(body, offset, 3, "DV2 structural options")
    options = _options_from_wire(*options_raw)
    count, offset = _get_uleb(body, offset)
    expected = _expected_section_tags(options)
    numeric_tags = [
        tag
        for tag in expected[2 + int(options.repeated_provenance) :]
        if tag not in {b"FIDX", b"EVNT"}
    ]
    if count != len(numeric_tags):
        raise PointFreeProgramError("DV2 structural numeric-section count differs from its recipe")
    payloads: dict[bytes, bytes] = {}
    for tag in numeric_tags:
        size, offset = _get_uleb(body, offset)
        payload, offset = _take(body, offset, size, f"DV2 {tag!r} payload")
        payloads[tag] = payload
    if offset != len(body):
        raise PointFreeProgramError("DV2 structural body has trailing bytes")
    encoded = _reassemble_numeric(payloads, options, pair_count)
    tensor = _temporal_decode(encoded, options.temporal_mode)
    inventory = FactInventory(
        tensor=tensor,
        source_height=source_height,
        source_width=source_width,
        semantic_sha256=_semantic_sha256(tensor),
    )
    inner = serialize_sentence(inventory, options)
    return zlib.compress(inner, level=9)


def _execute_dv2(
    formulation: Formulation,
    body: bytes,
    *,
    validate_semantics: bool = True,
) -> bytes:
    if formulation is Formulation.LITERAL:
        source = body
    elif formulation is Formulation.SHARED_LIBRARY:
        source = _execute_dv2_library(body)
    elif formulation is Formulation.STRUCTURAL:
        source = _execute_dv2_structural(body)
    else:  # pragma: no cover
        raise PointFreeProgramError("unsupported DV2 execution formulation")
    if validate_semantics:
        decode_sentence(decompress_outer_payload(source))
    return source


def compile_bundle(
    children: Sequence[CompiledProgram],
    formulation: Formulation,
    *,
    source_replays: Sequence[bytes] | None = None,
) -> CompiledProgram:
    """Compose already-compiled descriptions without changing child semantics.

    ``source_replays`` is an optional compile-time custody shortcut for callers
    that already hold the exact source bytes used to construct every child.  It
    is accepted only when each length and digest matches the child's validated
    compile receipt; public execution remains fully strict.
    """

    rows = tuple(children)
    if not rows or len(rows) > MAX_SECTIONS:
        raise PointFreeProgramError("PF1 bundles require between one and 64 children")
    if any(row.formulation is not formulation for row in rows):
        raise PointFreeProgramError("PF1 bundle children must share the declared formulation")
    body = bytearray()
    _put_uleb(body, len(rows))
    for row in rows:
        _put_uleb(body, len(row.program))
        body.extend(row.program)
    program = _wrap_program(Recipe.BUNDLE, formulation, bytes(body))
    if source_replays is None:
        executed = execute_program(program)
        if not isinstance(executed, tuple):
            raise PointFreeProgramError("PF1 bundle execution did not return a source tuple")
        sources = executed
    else:
        sources = tuple(bytes(value) for value in source_replays)
        if len(sources) != len(rows):
            raise PointFreeProgramError("PF1 bundle source-replay cardinality differs")
        if any(
            row.source_bytes != len(source)
            or row.source_sha256 != hashlib.sha256(source).hexdigest()
            for row, source in zip(rows, sources, strict=True)
        ):
            raise PointFreeProgramError("PF1 bundle source replay differs from a child compile receipt")
    flat = frame_flat_sources(sources)
    return CompiledProgram(
        recipe=Recipe.BUNDLE,
        formulation=formulation,
        program=program,
        source_bytes=len(flat),
        source_sha256=hashlib.sha256(flat).hexdigest(),
        video_derived_library_bytes=sum(row.video_derived_library_bytes for row in rows),
        operator_trace=RECIPE_TRACES[Recipe.BUNDLE],
    )


def _execute_bundle(body: bytes) -> tuple[bytes, ...]:
    count, offset = _get_uleb(body, 0)
    if not 1 <= count <= MAX_SECTIONS:
        raise PointFreeProgramError("PF1 bundle child count is invalid")
    sources: list[bytes] = []
    for _index in range(count):
        size, offset = _get_uleb(body, offset)
        child, offset = _take(body, offset, size, "bundle child")
        value = execute_program(child)
        if not isinstance(value, bytes):
            raise PointFreeProgramError("nested PF1 bundles are not admitted")
        sources.append(value)
    if offset != len(body):
        raise PointFreeProgramError("PF1 bundle has trailing bytes")
    return tuple(sources)


def execute_program(program: bytes) -> bytes | tuple[bytes, ...]:
    """Strictly execute a PF1 program back to its source description bytes."""

    recipe, formulation, body = _unwrap_program(program)
    if recipe is Recipe.G1_WORLDSHEET:
        return _execute_g1(formulation, body)
    if recipe is Recipe.V15_TEMPLATE_BANK:
        return _execute_v15(formulation, body)
    if recipe is Recipe.DV2_SENTENCE:
        return _execute_dv2(formulation, body)
    if recipe is Recipe.BUNDLE:
        return _execute_bundle(body)
    raise PointFreeProgramError("PF1 recipe is unsupported")  # pragma: no cover


def frame_flat_sources(sources: Sequence[bytes]) -> bytes:
    """Frame the same source tuple for a fair flat-coder control."""

    rows = tuple(bytes(value) for value in sources)
    if not rows or len(rows) > MAX_SECTIONS:
        raise PointFreeProgramError("flat source bundle cardinality is invalid")
    body = bytearray(FLAT_BUNDLE_MAGIC)
    _put_uleb(body, len(rows))
    for row in rows:
        _put_uleb(body, len(row))
        body.extend(row)
    return bytes(body)


def _real_codec_candidates(raw: bytes) -> dict[int, tuple[str, bytes]]:
    return {
        1: ("brotli_q11", brotli.compress(raw, quality=11)),
        2: (
            "lzma1_raw_1m",
            lzma.compress(
                raw,
                format=lzma.FORMAT_RAW,
                filters=[{"id": lzma.FILTER_LZMA1, "preset": 1, "dict_size": 1 << 20}],
            ),
        ),
        3: ("zlib9", zlib.compress(raw, level=9)),
    }


def code_real_payload(raw: bytes) -> CodedPayload:
    """Run the bounded real-coder set and return its exact counted frame."""

    source = bytes(raw)
    candidates = _real_codec_candidates(source)
    codec_id = min(candidates, key=lambda key: (len(candidates[key][1]), key))
    name, coded = candidates[codec_id]
    framed = CODED_MAGIC + bytes([codec_id]) + _uleb(len(source)) + coded
    if _decode_real_payload_body(codec_id, coded) != source:
        raise PointFreeProgramError("real-coded PF1 payload failed strict parse-back")
    return CodedPayload(
        codec=name,
        raw_bytes=len(source),
        coded_payload_bytes=len(coded),
        framed_bytes=len(framed),
        framed_sha256=hashlib.sha256(framed).hexdigest(),
        payload=framed,
    )


def decode_real_payload(payload: bytes) -> bytes:
    """Strictly decode a canonical PF1 real-coder frame."""

    if not isinstance(payload, bytes) or len(payload) < 5 or payload[:3] != CODED_MAGIC:
        raise PointFreeProgramError("real-coder frame magic is invalid")
    codec_id = payload[3]
    raw_size, offset = _get_uleb(payload, 4)
    coded = payload[offset:]
    raw = _decode_real_payload_body(codec_id, coded)
    if len(raw) != raw_size:
        raise PointFreeProgramError("real-coder raw-size custody failed")
    canonical = code_real_payload_unchecked(raw)
    if canonical != payload:
        raise PointFreeProgramError("real-coder frame is not canonical")
    return raw


def _decode_real_payload_body(codec_id: int, coded: bytes) -> bytes:
    """Decode one selected candidate without redundantly re-running the coder set."""

    try:
        if codec_id == 1:
            raw = brotli.decompress(coded)
        elif codec_id == 2:
            raw = lzma.decompress(
                coded,
                format=lzma.FORMAT_RAW,
                filters=[{"id": lzma.FILTER_LZMA1, "preset": 1, "dict_size": 1 << 20}],
            )
        elif codec_id == 3:
            decoder = zlib.decompressobj()
            raw = decoder.decompress(coded) + decoder.flush()
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise PointFreeProgramError("real-coded zlib stream is truncated or has trailers")
        else:
            raise PointFreeProgramError("real-coder frame codec is unknown")
    except (brotli.error, lzma.LZMAError, zlib.error) as exc:
        raise PointFreeProgramError("real-coder payload is invalid") from exc
    return raw


def code_real_payload_unchecked(raw: bytes) -> bytes:
    """Internal non-recursive canonical real-coder framing helper."""

    candidates = _real_codec_candidates(bytes(raw))
    codec_id = min(candidates, key=lambda key: (len(candidates[key][1]), key))
    coded = candidates[codec_id][1]
    return CODED_MAGIC + bytes([codec_id]) + _uleb(len(raw)) + coded


def rate_row(compiled: CompiledProgram, source: bytes) -> dict[str, Any]:
    """Measure program-vs-flat bytes on the identical source semantics."""

    source = bytes(source)
    source_sha256 = hashlib.sha256(source).hexdigest()
    if compiled.source_bytes != len(source) or compiled.source_sha256 != source_sha256:
        raise PointFreeProgramError("rate comparison differs from compile-time exact replay")
    program_coded = code_real_payload(compiled.program)
    flat_coded = code_real_payload(source)
    return {
        "recipe": compiled.recipe.name,
        "formulation": compiled.formulation.name,
        "source_bytes": len(source),
        "source_sha256": source_sha256,
        "program_raw_bytes": len(compiled.program),
        "program_codec": program_coded.codec,
        "program_counted_bytes": program_coded.framed_bytes,
        "program_counted_sha256": program_coded.framed_sha256,
        "flat_codec": flat_coded.codec,
        "flat_counted_bytes": flat_coded.framed_bytes,
        "flat_counted_sha256": flat_coded.framed_sha256,
        "delta_program_minus_flat_bytes": program_coded.framed_bytes - flat_coded.framed_bytes,
        "interpreter_bytes": "FREE_rule118_generic",
        "video_derived_library_bytes_inside_program": compiled.video_derived_library_bytes,
        "semantic_parseback_exact": True,
        "score_claim": False,
    }


def bundle_rate_row(compiled: CompiledProgram, sources: Sequence[bytes]) -> dict[str, Any]:
    """Measure a composed program against a same-content flat source bundle."""

    source_tuple = tuple(bytes(value) for value in sources)
    flat = frame_flat_sources(source_tuple)
    flat_sha256 = hashlib.sha256(flat).hexdigest()
    if compiled.source_bytes != len(flat) or compiled.source_sha256 != flat_sha256:
        raise PointFreeProgramError("bundle rate comparison differs from compile-time exact replay")
    program_coded = code_real_payload(compiled.program)
    flat_coded = code_real_payload(flat)
    return {
        "recipe": compiled.recipe.name,
        "formulation": compiled.formulation.name,
        "source_component_count": len(source_tuple),
        "flat_raw_bytes": len(flat),
        "flat_raw_sha256": flat_sha256,
        "program_raw_bytes": len(compiled.program),
        "program_codec": program_coded.codec,
        "program_counted_bytes": program_coded.framed_bytes,
        "program_counted_sha256": program_coded.framed_sha256,
        "flat_codec": flat_coded.codec,
        "flat_counted_bytes": flat_coded.framed_bytes,
        "flat_counted_sha256": flat_coded.framed_sha256,
        "delta_program_minus_flat_bytes": program_coded.framed_bytes - flat_coded.framed_bytes,
        "interpreter_bytes": "FREE_rule118_generic",
        "video_derived_library_bytes_inside_program": compiled.video_derived_library_bytes,
        "semantic_parseback_exact": True,
        "score_claim": False,
    }


def basis_manifest() -> tuple[dict[str, Any], ...]:
    """Return the fixed measured basis with no ungrounded vocabulary."""

    return tuple(
        {
            "primitive": row.primitive.value,
            "rank_rule": row.rank_rule,
            "provenance": list(row.provenance),
        }
        for row in MEASURED_BASIS
    )


__all__ = [
    "MEASURED_BASIS",
    "RECIPE_TRACES",
    "BasisPrimitive",
    "CodedPayload",
    "CompiledProgram",
    "Formulation",
    "PointFreeProgramError",
    "Recipe",
    "SharedLiteralLibrary",
    "apply_template",
    "basis_manifest",
    "bundle_rate_row",
    "channel_affine",
    "code_real_payload",
    "compile_bundle",
    "compile_dv2_sentence",
    "compile_g1_worldsheet",
    "compile_v15_template_bank",
    "compose_pointfree",
    "decode_real_payload",
    "evaluate_g1_worldsheet",
    "evaluate_lane_programs",
    "execute_program",
    "frame_flat_sources",
    "learn_shared_literals",
    "rate_row",
    "stratum_mask",
    "xi_advect",
]
