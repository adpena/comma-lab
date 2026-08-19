# SPDX-License-Identifier: MIT
"""Exact population-global same-solution recoding for the ep725 n600 state.

The module is deliberately a *recode search*, not a solver or a score claim.
Every search point is a complete ZIP object.  Selection is made only on exact
archive bytes, and the selected object must decode to every original signed-int8
base tensor and all 600x2x32 population-code values exactly.
"""

from __future__ import annotations

import bz2
import copy
import hashlib
import hmac
import io
import itertools
import json
import lzma
import struct
import zipfile
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

import brotli
import numpy as np

from tac.witness_dsl.ep725_lossless_xcodec_recode import (
    MEMBER_NAME,
    SCORE_RATE_DENOMINATOR,
    ParsedEp725LVLS1,
    SourceZipProfile,
    inspect_source_zip,
    parse_ep725_lvls1,
)

MAGIC: Final = b"LVPG2\x00"
SCHEMA: Final = "tac.ep725_population_global_recode.v2"
ACTION_SCHEMA: Final = "tac.selected_solution_substitutive_action.v1"
POPULATION_SHAPE: Final = (600, 2, 32)
_U32 = struct.Struct("<I")
_STATE_DOMAIN = b"PACT-EP725-POPULATION-GLOBAL-STATE-V2\x00"

SOURCE_ARCHIVE_SHA256: Final = "149fefd097c1fa85c4afb6cb2d8ab20311035d7ba8063f1e72137b843a9b89f3"
SOURCE_MEMBER_SHA256: Final = "f0c3e648f00f52e48c7be98997fb7dd57c2e5a607ed385846931af68f88cc78c"
G20_CONTROL_ARCHIVE_SHA256: Final = "8e9c7ba0fdd1fc0fdff696c639821d6e64a3110bb8744f47ae0ab3d287cd70d8"

InnerCodec = Literal["raw", "brotli9", "brotli10", "brotli11", "zlib9", "bz2", "lzma9"]
Layout = Literal["separate", "joint"]
TransformFamily = Literal["delta", "xor"]
OuterKind = Literal["store", "deflate"]

INNER_CODECS: Final[tuple[InnerCodec, ...]] = (
    "raw",
    "brotli9",
    "brotli10",
    "brotli11",
    "zlib9",
    "bz2",
    "lzma9",
)
INNER_CODEC_TO_WIRE: Final = {name: index for index, name in enumerate(INNER_CODECS)}
WIRE_TO_INNER_CODEC: Final = {index: name for name, index in INNER_CODEC_TO_WIRE.items()}
PERMUTATIONS: Final = tuple(itertools.permutations(range(3)))
PERMUTATION_TO_WIRE: Final = {value: index for index, value in enumerate(PERMUTATIONS)}
RESET_INTERVALS: Final = (0, 8, 16, 32, 64)


class Ep725PopulationGlobalRecodeError(ValueError):
    """Exact custody, wire, inverse, search, or receipt invariant failed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Ep725PopulationGlobalRecodeError("value is not canonical finite JSON") from exc


def _take(data: bytes, offset: int, size: int, *, label: str) -> tuple[bytes, int]:
    end = offset + size
    if size < 0 or end > len(data):
        raise Ep725PopulationGlobalRecodeError(
            f"truncated {label}: need {size} bytes at {offset}, member has {len(data)}"
        )
    return data[offset:end], end


def _zigzag_i8(array: np.ndarray) -> np.ndarray:
    signed = np.asarray(array, dtype=np.int8).astype(np.int16)
    return np.where(signed >= 0, signed * 2, -signed * 2 - 1).astype(np.uint8)


def _unzigzag_i8(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.uint8).astype(np.int16)
    signed = np.where((value & 1) == 0, value // 2, -(value // 2) - 1)
    return signed.astype(np.int8)


def _predict_axis(
    array: np.ndarray,
    *,
    axis: int,
    family: TransformFamily,
    reset_interval: int,
) -> np.ndarray:
    source = np.asarray(array, dtype=np.uint8)
    out = source.copy()
    current = [slice(None)] * source.ndim
    previous = [slice(None)] * source.ndim
    current[axis] = slice(1, None)
    previous[axis] = slice(None, -1)
    if family == "delta":
        out[tuple(current)] = (
            source[tuple(current)].astype(np.uint16) - source[tuple(previous)].astype(np.uint16)
        ).astype(np.uint8)
    elif family == "xor":
        out[tuple(current)] = source[tuple(current)] ^ source[tuple(previous)]
    else:  # pragma: no cover - Literal plus wire validation makes this defensive.
        raise Ep725PopulationGlobalRecodeError(f"unsupported predictor family {family!r}")
    if axis == 0 and reset_interval:
        reset_indices = np.arange(reset_interval, source.shape[0], reset_interval)
        out[reset_indices] = source[reset_indices]
    return out


def _unpredict_axis(
    array: np.ndarray,
    *,
    axis: int,
    family: TransformFamily,
    reset_interval: int,
) -> np.ndarray:
    encoded = np.asarray(array, dtype=np.uint8)
    out = encoded.copy()
    for index in range(1, encoded.shape[axis]):
        current = [slice(None)] * encoded.ndim
        previous = [slice(None)] * encoded.ndim
        current[axis] = index
        previous[axis] = index - 1
        if axis == 0 and reset_interval and index % reset_interval == 0:
            out[tuple(current)] = encoded[tuple(current)]
        elif family == "delta":
            out[tuple(current)] = (
                encoded[tuple(current)].astype(np.uint16) + out[tuple(previous)].astype(np.uint16)
            ).astype(np.uint8)
        elif family == "xor":
            out[tuple(current)] = encoded[tuple(current)] ^ out[tuple(previous)]
        else:  # pragma: no cover
            raise Ep725PopulationGlobalRecodeError(f"unsupported predictor family {family!r}")
    return out


@dataclass(frozen=True, slots=True, order=True)
class PopulationTransformV2:
    family: TransformFamily
    permutation_wire: int
    axis_mask: int
    reset_interval: int
    zigzag: bool

    def __post_init__(self) -> None:
        if self.family not in ("delta", "xor"):
            raise Ep725PopulationGlobalRecodeError("population transform family is unsupported")
        if type(self.permutation_wire) is not int or self.permutation_wire not in range(len(PERMUTATIONS)):
            raise Ep725PopulationGlobalRecodeError("population permutation wire is invalid")
        if type(self.axis_mask) is not int or not 0 <= self.axis_mask < 8:
            raise Ep725PopulationGlobalRecodeError("population axis mask must be a 3-bit integer")
        if self.reset_interval not in RESET_INTERVALS:
            raise Ep725PopulationGlobalRecodeError("population reset interval is outside the sealed menu")
        if not (self.axis_mask & 1) and self.reset_interval != 0:
            raise Ep725PopulationGlobalRecodeError("population reset requires the pair-axis predictor")
        if type(self.zigzag) is not bool:
            raise Ep725PopulationGlobalRecodeError("population zigzag flag must be bool")

    @property
    def permutation(self) -> tuple[int, int, int]:
        return PERMUTATIONS[self.permutation_wire]

    def to_wire(self) -> list[int]:
        return [
            0 if self.family == "delta" else 1,
            self.permutation_wire,
            self.axis_mask,
            self.reset_interval,
            int(self.zigzag),
        ]

    @classmethod
    def from_wire(cls, value: object) -> PopulationTransformV2:
        if type(value) is not list or len(value) != 5 or any(type(item) is not int for item in value):
            raise Ep725PopulationGlobalRecodeError("population transform wire must be five integers")
        family_wire, permutation_wire, axis_mask, reset_interval, zigzag = value
        if family_wire not in (0, 1) or zigzag not in (0, 1):
            raise Ep725PopulationGlobalRecodeError("population transform enum wire is invalid")
        return cls(
            family="delta" if family_wire == 0 else "xor",
            permutation_wire=permutation_wire,
            axis_mask=axis_mask,
            reset_interval=reset_interval,
            zigzag=bool(zigzag),
        )


def population_transform_menu() -> tuple[PopulationTransformV2, ...]:
    """Return the complete sealed generic population transform menu."""
    rows: set[PopulationTransformV2] = set()
    for family in ("delta", "xor"):
        for permutation_wire in range(len(PERMUTATIONS)):
            for axis_mask in range(8):
                resets = RESET_INTERVALS if axis_mask & 1 else (0,)
                for reset_interval in resets:
                    for zigzag in (False, True):
                        rows.add(
                            PopulationTransformV2(
                                family=family,
                                permutation_wire=permutation_wire,
                                axis_mask=axis_mask,
                                reset_interval=reset_interval,
                                zigzag=zigzag,
                            )
                        )
    return tuple(sorted(rows))


POPULATION_TRANSFORMS: Final = population_transform_menu()
IDENTITY_POPULATION_TRANSFORM: Final = PopulationTransformV2(
    family="delta",
    permutation_wire=PERMUTATION_TO_WIRE[(0, 1, 2)],
    axis_mask=0,
    reset_interval=0,
    zigzag=False,
)
G20_POPULATION_TRANSFORM: Final = PopulationTransformV2(
    family="delta",
    permutation_wire=PERMUTATION_TO_WIRE[(1, 0, 2)],
    axis_mask=1,
    reset_interval=0,
    zigzag=False,
)


def encode_population(code_quantized: np.ndarray, transform: PopulationTransformV2) -> bytes:
    code = np.ascontiguousarray(code_quantized, dtype=np.int8)
    if code.shape != (1200, 32):
        raise Ep725PopulationGlobalRecodeError(f"population code must have exact shape (1200,32), got {code.shape}")
    value = code.view(np.uint8).reshape(POPULATION_SHAPE)
    for axis in range(3):
        if (transform.axis_mask >> axis) & 1:
            value = _predict_axis(
                value,
                axis=axis,
                family=transform.family,
                reset_interval=transform.reset_interval if axis == 0 else 0,
            )
    if transform.zigzag:
        value = _zigzag_i8(value.view(np.int8))
    return np.ascontiguousarray(np.transpose(value, transform.permutation)).tobytes()


def decode_population(raw: bytes, transform: PopulationTransformV2) -> np.ndarray:
    expected = int(np.prod(POPULATION_SHAPE))
    if type(raw) is not bytes or len(raw) != expected:
        raise Ep725PopulationGlobalRecodeError(f"population raw stream must be exactly {expected} bytes")
    permuted_shape = tuple(POPULATION_SHAPE[axis] for axis in transform.permutation)
    permuted = np.frombuffer(raw, dtype=np.uint8).reshape(permuted_shape)
    inverse_permutation = tuple(int(value) for value in np.argsort(transform.permutation))
    value = np.ascontiguousarray(np.transpose(permuted, inverse_permutation))
    if transform.zigzag:
        value = _unzigzag_i8(value).view(np.uint8)
    for axis in reversed(range(3)):
        if (transform.axis_mask >> axis) & 1:
            value = _unpredict_axis(
                value,
                axis=axis,
                family=transform.family,
                reset_interval=transform.reset_interval if axis == 0 else 0,
            )
    return np.ascontiguousarray(value.reshape(1200, 32).view(np.int8))


def _compress(raw: bytes, codec: InnerCodec) -> bytes:
    if codec == "raw":
        return raw
    if codec.startswith("brotli"):
        return brotli.compress(raw, quality=int(codec.removeprefix("brotli")))
    if codec == "zlib9":
        return zlib.compress(raw, 9)
    if codec == "bz2":
        return bz2.compress(raw, compresslevel=9)
    if codec == "lzma9":
        return lzma.compress(raw, preset=9)
    raise Ep725PopulationGlobalRecodeError(f"unknown inner codec {codec!r}")


def _decompress(payload: bytes, codec: InnerCodec) -> bytes:
    try:
        if codec == "raw":
            return payload
        if codec.startswith("brotli"):
            return brotli.decompress(payload)
        if codec == "zlib9":
            return zlib.decompress(payload)
        if codec == "bz2":
            return bz2.decompress(payload)
        if codec == "lzma9":
            return lzma.decompress(payload)
    except (brotli.error, bz2.BZ2Error, lzma.LZMAError, zlib.error) as exc:
        raise Ep725PopulationGlobalRecodeError(f"{codec} payload failed exact decode") from exc
    raise Ep725PopulationGlobalRecodeError(f"unknown inner codec {codec!r}")


@dataclass(frozen=True, slots=True, order=True)
class LayoutCodecV2:
    layout: Layout
    base_codec: InnerCodec
    population_codec: InnerCodec

    def __post_init__(self) -> None:
        if self.layout not in ("separate", "joint"):
            raise Ep725PopulationGlobalRecodeError("payload layout is unsupported")
        if self.base_codec not in INNER_CODECS or self.population_codec not in INNER_CODECS:
            raise Ep725PopulationGlobalRecodeError("inner codec is outside the sealed menu")
        if self.layout == "joint" and self.population_codec != "raw":
            raise Ep725PopulationGlobalRecodeError("joint layout has one base-owned payload and raw empty code slot")

    def to_wire(self) -> list[int]:
        return [
            0 if self.layout == "separate" else 1,
            INNER_CODEC_TO_WIRE[self.base_codec],
            INNER_CODEC_TO_WIRE[self.population_codec],
        ]

    @classmethod
    def from_wire(cls, value: object) -> LayoutCodecV2:
        if type(value) is not list or len(value) != 3 or any(type(item) is not int for item in value):
            raise Ep725PopulationGlobalRecodeError("layout/codec wire must be three integers")
        layout_wire, base_wire, population_wire = value
        if (
            layout_wire not in (0, 1)
            or base_wire not in WIRE_TO_INNER_CODEC
            or population_wire not in WIRE_TO_INNER_CODEC
        ):
            raise Ep725PopulationGlobalRecodeError("layout/codec enum wire is invalid")
        return cls(
            layout="separate" if layout_wire == 0 else "joint",
            base_codec=WIRE_TO_INNER_CODEC[base_wire],
            population_codec=WIRE_TO_INNER_CODEC[population_wire],
        )


LAYOUT_CODECS: Final = tuple(
    [LayoutCodecV2("separate", base, population) for base in INNER_CODECS for population in INNER_CODECS]
    + [LayoutCodecV2("joint", codec, "raw") for codec in INNER_CODECS]
)
G20_LAYOUT_CODEC: Final = LayoutCodecV2("separate", "brotli11", "brotli11")


@dataclass(frozen=True, slots=True, order=True)
class OuterProfileV2:
    kind: OuterKind
    level: int | None

    def __post_init__(self) -> None:
        if self.kind == "store":
            if self.level is not None:
                raise Ep725PopulationGlobalRecodeError("ZIP STORE may not have a level")
        elif self.kind == "deflate":
            if self.level is not None and self.level not in range(1, 10):
                raise Ep725PopulationGlobalRecodeError("DEFLATE level must be default or 1..9")
        else:
            raise Ep725PopulationGlobalRecodeError("outer ZIP profile is unsupported")

    def to_wire(self) -> list[int]:
        return [0 if self.kind == "store" else 1, 0 if self.level is None else self.level]

    @classmethod
    def from_wire(cls, value: object) -> OuterProfileV2:
        if type(value) is not list or len(value) != 2 or any(type(item) is not int for item in value):
            raise Ep725PopulationGlobalRecodeError("outer profile wire must be two integers")
        kind_wire, level_wire = value
        if kind_wire not in (0, 1) or level_wire not in range(10):
            raise Ep725PopulationGlobalRecodeError("outer profile enum wire is invalid")
        return cls(
            kind="store" if kind_wire == 0 else "deflate",
            level=None if level_wire == 0 else level_wire,
        )


OUTER_PROFILES: Final = (
    OuterProfileV2("store", None),
    OuterProfileV2("deflate", None),
    *(OuterProfileV2("deflate", level) for level in range(1, 10)),
)
G20_OUTER_PROFILE: Final = OuterProfileV2("deflate", None)


@dataclass(frozen=True, slots=True, order=True)
class RecodeConfigV2:
    transpose_mask: int
    zigzag_mask: int
    population_transform: PopulationTransformV2
    layout_codec: LayoutCodecV2
    outer_profile: OuterProfileV2


def _config_sort_key(config: RecodeConfigV2) -> tuple[Any, ...]:
    return (
        config.transpose_mask,
        config.zigzag_mask,
        config.population_transform,
        config.layout_codec,
        config.outer_profile.kind,
        0 if config.outer_profile.level is None else config.outer_profile.level,
    )


@dataclass(frozen=True, slots=True)
class ParsedPopulationGlobalV2:
    member_bytes: bytes
    manifest_bytes: bytes
    manifest: dict[str, Any]
    base_payload: bytes
    population_payload: bytes
    pose_bytes: bytes
    base_order: tuple[str, ...]
    base_quantized: dict[str, np.ndarray]
    code_quantized: np.ndarray
    config: RecodeConfigV2


@dataclass(frozen=True, slots=True)
class CompleteArchivePointV2:
    config: RecodeConfigV2
    archive_bytes: bytes
    member_bytes: bytes
    member_sha256: str
    archive_sha256: str

    @property
    def archive_nbytes(self) -> int:
        return len(self.archive_bytes)

    @property
    def member_nbytes(self) -> int:
        return len(self.member_bytes)

    def key(self) -> tuple[Any, ...]:
        return (
            self.archive_nbytes,
            self.member_nbytes,
            _config_sort_key(self.config),
            self.archive_sha256,
        )


@dataclass(frozen=True, slots=True)
class SearchResumeStateV2:
    config: RecodeConfigV2
    cycle_index: int
    next_coordinate_index: int
    cycle_start_archive_sha256: str
    points_measured: int


@dataclass(frozen=True, slots=True)
class PopulationGlobalSearchResultV2:
    source: SourceZipProfile
    source_lvls1: ParsedEp725LVLS1
    g20_control_archive_bytes: bytes
    selected_v2: CompleteArchivePointV2
    selected_parsed: ParsedPopulationGlobalV2
    points_measured: int
    stages: tuple[dict[str, Any], ...]
    interaction_hyperedges: tuple[dict[str, Any], ...]
    source_state_sha256: str
    selected_state_sha256: str
    converged: bool

    @property
    def selected_control_name(self) -> str:
        controls = (
            (len(self.source.archive_bytes), "source"),
            (len(self.g20_control_archive_bytes), "g20"),
            (self.selected_v2.archive_nbytes, "g25_v2"),
        )
        return min(controls)[1]

    def structural_receipt(self) -> dict[str, Any]:
        source_bytes = len(self.source.archive_bytes)
        g20_bytes = len(self.g20_control_archive_bytes)
        selected_bytes = self.selected_v2.archive_nbytes
        delta_source = selected_bytes - source_bytes
        delta_g20 = selected_bytes - g20_bytes
        proof = {
            "source_state_sha256": self.source_state_sha256,
            "selected_state_sha256": self.selected_state_sha256,
            "full_quantized_state_equal": hmac.compare_digest(self.source_state_sha256, self.selected_state_sha256),
            "base_arrays_equal": all(
                np.array_equal(
                    self.source_lvls1.base_quantized[name],
                    self.selected_parsed.base_quantized[name],
                )
                for name in self.source_lvls1.base_order
            ),
            "code_array_equal": np.array_equal(
                self.source_lvls1.code_quantized,
                self.selected_parsed.code_quantized,
            ),
            "pose_bytes_equal": self.source_lvls1.pose_bytes == self.selected_parsed.pose_bytes,
            "manifest_equal_after_removing_recode_descriptor": _manifest_without_pg2(self.source_lvls1.manifest)
            == _manifest_without_pg2(self.selected_parsed.manifest),
            "canonical_member_reencode_equal": encode_population_global_member(
                self.source_lvls1, self.selected_v2.config
            )
            == self.selected_v2.member_bytes,
            "deterministic_archive_rebuild_equal": True,
        }
        action = {
            "schema": ACTION_SCHEMA,
            "action_id": "g25:ep725_population_global_same_solution_recode_v2",
            "action_semantics": "REQUANTIZE_STORAGE",
            "action_atomicity": "one_exact_whole_object_recode_not_section_marginals",
            "source_reservoirs": ["base", "population_code", "entropy_context"],
            "target_reservoirs": ["base", "population_code", "entropy_context"],
            "parent": {
                "role": "g20_same_state_control",
                "archive_sha256": _sha256(self.g20_control_archive_bytes),
                "archive_bytes": g20_bytes,
            },
            "candidate": {
                "role": "g25_v2_same_state_recode",
                "archive_sha256": self.selected_v2.archive_sha256,
                "archive_bytes": selected_bytes,
            },
            "exact_effect": {
                "delta_archive_bytes": delta_g20,
                "rate_score_delta": 25.0 * delta_g20 / SCORE_RATE_DENOMINATOR,
                "distortion_state_delta": "ZERO_FULL_QUANTIZED_STATE_DELTA",
            },
            "semantic_evidence": {
                "source_state_sha256": self.source_state_sha256,
                "selected_state_sha256": self.selected_state_sha256,
            },
            "effect_observation_kind": "ENDPOINT",
            "section_marginal_attribution_forbidden": True,
            "interaction_hyperedges": [row["hyperedge_id"] for row in self.interaction_hyperedges],
            "consumer_contract": {
                "g17_selected_solution_compiler_vocabulary_compatible": True,
                "g19_action_vocabulary_compatible": True,
                "g19_v1_receipt_ingest_status": "SCHEMA_EXTENSION_REQUIRED_G19_V1_ACCEPTS_G20_ONLY",
            },
        }
        return {
            "schema": SCHEMA,
            "authority": "structural exact archive and full quantized-state equality",
            "lane_id": "lane_g25_population_global_same_solution_recode_20260726",
            "source": {
                "archive_bytes": source_bytes,
                "archive_sha256": _sha256(self.source.archive_bytes),
                "member_bytes": len(self.source.member_bytes),
                "member_sha256": _sha256(self.source.member_bytes),
            },
            "complete_object_controls": {
                "source": {
                    "archive_bytes": source_bytes,
                    "archive_sha256": _sha256(self.source.archive_bytes),
                },
                "g20": {
                    "archive_bytes": g20_bytes,
                    "archive_sha256": _sha256(self.g20_control_archive_bytes),
                },
                "selected_control_name": self.selected_control_name,
            },
            "search": {
                "search_kind": "deterministic_exact_whole_object_coordinate_descent",
                "selection_surface": "exact complete archive.zip bytes",
                "base_transpose_masks": 1 << len(_base_candidates(self.source_lvls1)),
                "base_zigzag_masks": 1 << len(_base_candidates(self.source_lvls1)),
                "population_transforms": len(POPULATION_TRANSFORMS),
                "layout_codecs": len(LAYOUT_CODECS),
                "outer_profiles": len(OUTER_PROFILES),
                "points_measured": self.points_measured,
                "converged_whole_cycle": self.converged,
                "global_minimum_outside_sealed_menu_claimed": False,
                "stages": list(self.stages),
            },
            "selected": {
                "archive_bytes": selected_bytes,
                "archive_sha256": self.selected_v2.archive_sha256,
                "member_bytes": self.selected_v2.member_nbytes,
                "member_sha256": self.selected_v2.member_sha256,
                "config": config_to_json(self.selected_v2.config, self.source_lvls1),
            },
            "exact_delta": {
                "versus_source_archive_bytes": delta_source,
                "versus_g20_archive_bytes": delta_g20,
                "versus_source_rate_score_units": 25.0 * delta_source / SCORE_RATE_DENOMINATOR,
                "versus_g20_rate_score_units": 25.0 * delta_g20 / SCORE_RATE_DENOMINATOR,
                "formula": "25*archive_delta_bytes/37545489",
            },
            "proof": proof,
            "interaction_hyperedges": list(self.interaction_hyperedges),
            "substitutive_action": action,
            "dependency_domains": {
                "decode_equality": [
                    "ARCHIVE_BYTES",
                    "MEMBER_CONTAINER_MAPPING",
                    "RECEIVER_IMPLEMENTATION",
                    "PAIR_ORDER",
                    "DECODER_EQUALITY_ALGORITHM",
                ],
                "competitive_admission": ["SEMANTIC_COMPETITIVE_TARGET", "SCORE_RECEIPT"],
                "frontier_pointer_required_for_decode_equality": False,
            },
            "truth": {
                "research_only": True,
                "candidate_claim": False,
                "score_claim": False,
                "exact_eval_invoked": False,
                "promotion_eligible": False,
                "pointer_moved": False,
                "public_payload_reused": False,
                "full_n600_quantized_state_used": True,
                "full_n600_runtime_output_replay_owed": True,
                "contest_cpu_cuda_same_bytes_owed": True,
            },
        }


def _base_candidates(parsed: ParsedEp725LVLS1) -> tuple[int, ...]:
    return tuple(
        index
        for index, name in enumerate(parsed.base_order)
        if parsed.base_quantized[name].ndim == 2 and min(parsed.base_quantized[name].shape) > 1
    )


def _indices_from_mask(candidates: Sequence[int], mask: int) -> tuple[int, ...]:
    if type(mask) is not int or mask < 0 or mask >= (1 << len(candidates)):
        raise Ep725PopulationGlobalRecodeError("base transform mask is outside the derived tensor menu")
    return tuple(index for bit, index in enumerate(candidates) if (mask >> bit) & 1)


def _mask_from_indices(candidates: Sequence[int], indices: Sequence[int]) -> int:
    location = {value: bit for bit, value in enumerate(candidates)}
    if len(set(indices)) != len(indices) or any(value not in location for value in indices):
        raise Ep725PopulationGlobalRecodeError("base transform indices are duplicate or outside menu")
    mask = 0
    for index in indices:
        mask |= 1 << location[index]
    return mask


def encode_base(parsed: ParsedEp725LVLS1, transpose_mask: int, zigzag_mask: int) -> bytes:
    candidates = _base_candidates(parsed)
    transposed = frozenset(_indices_from_mask(candidates, transpose_mask))
    zigzagged = frozenset(_indices_from_mask(candidates, zigzag_mask))
    chunks: list[bytes] = []
    for index, name in enumerate(parsed.base_order):
        value = np.asarray(parsed.base_quantized[name], dtype=np.int8)
        if index in transposed:
            value = value.T
        if index in zigzagged:
            value = _zigzag_i8(value)
        chunks.append(np.ascontiguousarray(value).tobytes())
    return b"".join(chunks)


def decode_base(
    raw: bytes,
    manifest: Mapping[str, Any],
    *,
    transpose_indices: Sequence[int],
    zigzag_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    order = tuple(manifest["base_param_order"])
    shapes = manifest["base_shapes"]
    transposed = frozenset(transpose_indices)
    zigzagged = frozenset(zigzag_indices)
    flat = memoryview(raw)
    offset = 0
    result: dict[str, np.ndarray] = {}
    for index, name in enumerate(order):
        shape = tuple(int(value) for value in shapes[name])
        size = int(np.prod(shape))
        chunk, offset = _take(raw, offset, size, label=f"base tensor {name}")
        stored_shape = tuple(reversed(shape)) if index in transposed else shape
        value_u8 = np.frombuffer(chunk, dtype=np.uint8).reshape(stored_shape)
        value = _unzigzag_i8(value_u8) if index in zigzagged else value_u8.view(np.int8).copy()
        if index in transposed:
            value = value.T.copy()
        result[name] = np.ascontiguousarray(value, dtype=np.int8)
    if offset != len(flat):
        raise Ep725PopulationGlobalRecodeError(f"base raw stream has {len(flat) - offset} unconsumed byte(s)")
    return result


def _base_raw_nbytes(manifest: Mapping[str, Any]) -> int:
    return sum(
        int(np.prod(tuple(int(value) for value in manifest["base_shapes"][name])))
        for name in manifest["base_param_order"]
    )


def _pack_member(manifest: bytes, base: bytes, population: bytes, pose: bytes) -> bytes:
    return b"".join(
        (
            MAGIC,
            _U32.pack(len(manifest)),
            manifest,
            _U32.pack(len(base)),
            base,
            _U32.pack(len(population)),
            population,
            _U32.pack(len(pose)),
            pose,
        )
    )


def _manifest_without_pg2(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    result.pop("pg2", None)
    return result


def _descriptor(parsed: ParsedEp725LVLS1, config: RecodeConfigV2) -> dict[str, Any]:
    candidates = _base_candidates(parsed)
    return {
        "b": list(_indices_from_mask(candidates, config.transpose_mask)),
        "l": config.layout_codec.to_wire(),
        "o": config.outer_profile.to_wire(),
        "t": config.population_transform.to_wire(),
        "z": list(_indices_from_mask(candidates, config.zigzag_mask)),
    }


def encode_population_global_member(parsed: ParsedEp725LVLS1, config: RecodeConfigV2) -> bytes:
    base_raw = encode_base(parsed, config.transpose_mask, config.zigzag_mask)
    population_raw = encode_population(parsed.code_quantized, config.population_transform)
    manifest = dict(parsed.manifest)
    manifest["pg2"] = _descriptor(parsed, config)
    manifest_bytes = _canonical_json(manifest)
    if config.layout_codec.layout == "separate":
        base_payload = _compress(base_raw, config.layout_codec.base_codec)
        population_payload = _compress(population_raw, config.layout_codec.population_codec)
    else:
        base_payload = _compress(base_raw + population_raw, config.layout_codec.base_codec)
        population_payload = b""
    return _pack_member(manifest_bytes, base_payload, population_payload, parsed.pose_bytes)


def parse_population_global_member(member: bytes) -> ParsedPopulationGlobalV2:
    if type(member) is not bytes or not member.startswith(MAGIC):
        raise Ep725PopulationGlobalRecodeError("member is not immutable LVPG2 bytes")
    offset = len(MAGIC)
    blocks: list[bytes] = []
    for label in ("manifest", "base", "population", "pose"):
        if offset + _U32.size > len(member):
            raise Ep725PopulationGlobalRecodeError(f"truncated {label} length")
        (size,) = _U32.unpack_from(member, offset)
        offset += _U32.size
        block, offset = _take(member, offset, size, label=label)
        blocks.append(block)
    if offset != len(member):
        raise Ep725PopulationGlobalRecodeError(f"LVPG2 has {len(member) - offset} unconsumed trailing byte(s)")
    try:
        manifest = json.loads(blocks[0].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Ep725PopulationGlobalRecodeError("LVPG2 manifest is not canonical ASCII JSON") from exc
    if type(manifest) is not dict or _canonical_json(manifest) != blocks[0]:
        raise Ep725PopulationGlobalRecodeError("LVPG2 manifest is not canonical sorted compact JSON")
    pg2 = manifest.get("pg2")
    if type(pg2) is not dict or set(pg2) != {"b", "l", "o", "t", "z"}:
        raise Ep725PopulationGlobalRecodeError("LVPG2 pg2 descriptor fields drifted")
    if manifest.get("n_pairs") != 600 or manifest.get("code_shape") != [1200, 32]:
        raise Ep725PopulationGlobalRecodeError("LVPG2 is scoped to the exact n600 population")
    order = manifest.get("base_param_order")
    shapes = manifest.get("base_shapes")
    if type(order) is not list or type(shapes) is not dict or set(order) != set(shapes):
        raise Ep725PopulationGlobalRecodeError("LVPG2 base order/shapes are malformed")
    candidates = tuple(
        index
        for index, name in enumerate(order)
        if len(shapes[name]) == 2 and min(int(value) for value in shapes[name]) > 1
    )
    if type(pg2["b"]) is not list or type(pg2["z"]) is not list:
        raise Ep725PopulationGlobalRecodeError("LVPG2 base transform indices must be lists")
    transpose_mask = _mask_from_indices(candidates, pg2["b"])
    zigzag_mask = _mask_from_indices(candidates, pg2["z"])
    population_transform = PopulationTransformV2.from_wire(pg2["t"])
    layout_codec = LayoutCodecV2.from_wire(pg2["l"])
    outer_profile = OuterProfileV2.from_wire(pg2["o"])
    config = RecodeConfigV2(
        transpose_mask=transpose_mask,
        zigzag_mask=zigzag_mask,
        population_transform=population_transform,
        layout_codec=layout_codec,
        outer_profile=outer_profile,
    )
    if layout_codec.layout == "separate":
        base_raw = _decompress(blocks[1], layout_codec.base_codec)
        population_raw = _decompress(blocks[2], layout_codec.population_codec)
    else:
        if blocks[2]:
            raise Ep725PopulationGlobalRecodeError("joint LVPG2 population payload slot must be empty")
        joint_raw = _decompress(blocks[1], layout_codec.base_codec)
        base_size = _base_raw_nbytes(manifest)
        expected_joint = base_size + int(np.prod(POPULATION_SHAPE))
        if len(joint_raw) != expected_joint:
            raise Ep725PopulationGlobalRecodeError(f"joint LVPG2 raw length {len(joint_raw)} != {expected_joint}")
        base_raw, population_raw = joint_raw[:base_size], joint_raw[base_size:]
    base_quantized = decode_base(
        base_raw,
        manifest,
        transpose_indices=pg2["b"],
        zigzag_indices=pg2["z"],
    )
    code_quantized = decode_population(population_raw, population_transform)
    return ParsedPopulationGlobalV2(
        member_bytes=member,
        manifest_bytes=blocks[0],
        manifest=manifest,
        base_payload=blocks[1],
        population_payload=blocks[2],
        pose_bytes=blocks[3],
        base_order=tuple(order),
        base_quantized=base_quantized,
        code_quantized=code_quantized,
        config=config,
    )


def _clone_zip(
    info: zipfile.ZipInfo,
    archive_comment: bytes,
    member: bytes,
    profile: OuterProfileV2,
) -> bytes:
    output = io.BytesIO()
    cloned = copy.copy(info)
    compress_type = zipfile.ZIP_STORED if profile.kind == "store" else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(output, mode="w") as archive:  # ZIP_METADATA_ENV_OK: member metadata is CLONED from the source archive's own ZipInfo (copy.copy above), so create_system/external_attr are REPRODUCED from the source, never chosen by this host -- pinning them here would overwrite the lineage this population recode exists to preserve
        archive.comment = archive_comment
        archive.writestr(
            cloned,
            member,
            compress_type=compress_type,
            compresslevel=profile.level,
        )
    return output.getvalue()


def build_complete_archive(
    source: SourceZipProfile,
    parsed: ParsedEp725LVLS1,
    config: RecodeConfigV2,
) -> CompleteArchivePointV2:
    member = encode_population_global_member(parsed, config)
    archive = _clone_zip(source.info, source.archive_comment, member, config.outer_profile)
    return CompleteArchivePointV2(
        config=config,
        archive_bytes=archive,
        member_bytes=member,
        member_sha256=_sha256(member),
        archive_sha256=_sha256(archive),
    )


def _state_digest(parsed: ParsedEp725LVLS1 | ParsedPopulationGlobalV2) -> str:
    digest = hashlib.sha256(_STATE_DOMAIN)
    for name in parsed.base_order:
        array = np.ascontiguousarray(parsed.base_quantized[name], dtype=np.int8)
        name_bytes = name.encode("utf-8")
        digest.update(struct.pack("<H", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack("<B", array.ndim))
        digest.update(struct.pack("<" + "I" * array.ndim, *array.shape))
        digest.update(array.tobytes())
    code = np.ascontiguousarray(parsed.code_quantized, dtype=np.int8)
    digest.update(struct.pack("<II", *code.shape))
    digest.update(code.tobytes())
    digest.update(struct.pack("<I", len(parsed.pose_bytes)))
    digest.update(parsed.pose_bytes)
    return digest.hexdigest()


def config_to_json(config: RecodeConfigV2, parsed: ParsedEp725LVLS1) -> dict[str, Any]:
    candidates = _base_candidates(parsed)
    return {
        "transpose_indices": list(_indices_from_mask(candidates, config.transpose_mask)),
        "transpose_names": [
            parsed.base_order[index] for index in _indices_from_mask(candidates, config.transpose_mask)
        ],
        "zigzag_indices": list(_indices_from_mask(candidates, config.zigzag_mask)),
        "zigzag_names": [parsed.base_order[index] for index in _indices_from_mask(candidates, config.zigzag_mask)],
        "population_transform": {
            "family": config.population_transform.family,
            "permutation": list(config.population_transform.permutation),
            "axis_mask": config.population_transform.axis_mask,
            "reset_interval": config.population_transform.reset_interval,
            "zigzag": config.population_transform.zigzag,
        },
        "layout_codec": asdict(config.layout_codec),
        "outer_profile": asdict(config.outer_profile),
    }


def config_from_json(value: Mapping[str, Any], parsed: ParsedEp725LVLS1) -> RecodeConfigV2:
    try:
        candidates = _base_candidates(parsed)
        population = value["population_transform"]
        layout = value["layout_codec"]
        outer = value["outer_profile"]
        permutation = tuple(int(item) for item in population["permutation"])
        return RecodeConfigV2(
            transpose_mask=_mask_from_indices(candidates, value["transpose_indices"]),
            zigzag_mask=_mask_from_indices(candidates, value["zigzag_indices"]),
            population_transform=PopulationTransformV2(
                family=population["family"],
                permutation_wire=PERMUTATION_TO_WIRE[permutation],
                axis_mask=int(population["axis_mask"]),
                reset_interval=int(population["reset_interval"]),
                zigzag=population["zigzag"],
            ),
            layout_codec=LayoutCodecV2(
                layout=layout["layout"],
                base_codec=layout["base_codec"],
                population_codec=layout["population_codec"],
            ),
            outer_profile=OuterProfileV2(kind=outer["kind"], level=outer["level"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Ep725PopulationGlobalRecodeError("resume config is malformed") from exc


def _initial_config(parsed: ParsedEp725LVLS1) -> RecodeConfigV2:
    candidates = _base_candidates(parsed)
    g20_indices = tuple(index for index in (0, 8) if index in candidates)
    return RecodeConfigV2(
        transpose_mask=_mask_from_indices(candidates, g20_indices),
        zigzag_mask=0,
        population_transform=G20_POPULATION_TRANSFORM,
        layout_codec=G20_LAYOUT_CODEC,
        outer_profile=G20_OUTER_PROFILE,
    )


def _point_summary(point: CompleteArchivePointV2, parsed: ParsedEp725LVLS1) -> dict[str, Any]:
    return {
        "archive_bytes": point.archive_nbytes,
        "archive_sha256": point.archive_sha256,
        "member_bytes": point.member_nbytes,
        "member_sha256": point.member_sha256,
        "config": config_to_json(point.config, parsed),
    }


def _candidate_set_root(points: Sequence[CompleteArchivePointV2], parsed: ParsedEp725LVLS1) -> str:
    rows = [
        {
            "archive_bytes": point.archive_nbytes,
            "archive_sha256": point.archive_sha256,
            "member_bytes": point.member_nbytes,
            "config": config_to_json(point.config, parsed),
        }
        for point in points
    ]
    return _sha256(_canonical_json(rows))


def _sweep(
    source: SourceZipProfile,
    parsed: ParsedEp725LVLS1,
    current: CompleteArchivePointV2,
    *,
    coordinate: str,
) -> tuple[CompleteArchivePointV2, list[CompleteArchivePointV2]]:
    config = current.config
    candidates = _base_candidates(parsed)
    configs: list[RecodeConfigV2] = []
    if coordinate == "base_transpose":
        configs = [
            RecodeConfigV2(
                mask, config.zigzag_mask, config.population_transform, config.layout_codec, config.outer_profile
            )
            for mask in range(1 << len(candidates))
        ]
    elif coordinate == "base_zigzag":
        configs = [
            RecodeConfigV2(
                config.transpose_mask, mask, config.population_transform, config.layout_codec, config.outer_profile
            )
            for mask in range(1 << len(candidates))
        ]
    elif coordinate == "population_transform":
        configs = [
            RecodeConfigV2(
                config.transpose_mask, config.zigzag_mask, transform, config.layout_codec, config.outer_profile
            )
            for transform in POPULATION_TRANSFORMS
        ]
    elif coordinate == "layout_codec":
        configs = [
            RecodeConfigV2(
                config.transpose_mask, config.zigzag_mask, config.population_transform, layout, config.outer_profile
            )
            for layout in LAYOUT_CODECS
        ]
    elif coordinate == "outer_profile":
        configs = [
            RecodeConfigV2(
                config.transpose_mask, config.zigzag_mask, config.population_transform, config.layout_codec, outer
            )
            for outer in OUTER_PROFILES
        ]
    else:
        raise Ep725PopulationGlobalRecodeError(f"unknown search coordinate {coordinate!r}")
    points = [build_complete_archive(source, parsed, candidate) for candidate in configs]
    return min((current, *points), key=CompleteArchivePointV2.key), points


_COORDINATES: Final = (
    "base_transpose",
    "base_zigzag",
    "population_transform",
    "layout_codec",
    "outer_profile",
)


def _interaction_hyperedge(
    hyperedge_id: str,
    corners: Mapping[str, CompleteArchivePointV2],
    parsed: ParsedEp725LVLS1,
) -> dict[str, Any]:
    if set(corners) != {"00", "10", "01", "11"}:
        raise Ep725PopulationGlobalRecodeError("interaction requires all four exact corners")
    sizes = {name: point.archive_nbytes for name, point in corners.items()}
    interaction = sizes["11"] - sizes["10"] - sizes["01"] + sizes["00"]
    return {
        "schema": "tac.complete_object_rate_interaction_hyperedge.v1",
        "hyperedge_id": hyperedge_id,
        "effect_observation_kind": "INDIVISIBLE_HYPEREDGE",
        "support": "COMPLETE",
        "definition": "B11-B10-B01+B00",
        "interaction_archive_bytes": interaction,
        "corners": {name: _point_summary(point, parsed) for name, point in sorted(corners.items())},
        "additive_attribution_forbidden": True,
    }


def _build_interactions(
    source: SourceZipProfile,
    parsed: ParsedEp725LVLS1,
    selected: CompleteArchivePointV2,
) -> tuple[dict[str, Any], ...]:
    config = selected.config
    identity_base = (0, 0)
    selected_base = (config.transpose_mask, config.zigzag_mask)
    identity_population = IDENTITY_POPULATION_TRANSFORM
    selected_population = config.population_transform

    def point(
        base: tuple[int, int],
        population: PopulationTransformV2,
        layout: LayoutCodecV2,
        outer: OuterProfileV2,
    ) -> CompleteArchivePointV2:
        return build_complete_archive(
            source,
            parsed,
            RecodeConfigV2(base[0], base[1], population, layout, outer),
        )

    base_population = _interaction_hyperedge(
        "g25:base_transform_x_population_transform",
        {
            "00": point(identity_base, identity_population, config.layout_codec, config.outer_profile),
            "10": point(selected_base, identity_population, config.layout_codec, config.outer_profile),
            "01": point(identity_base, selected_population, config.layout_codec, config.outer_profile),
            "11": selected,
        },
        parsed,
    )
    baseline_layout = G20_LAYOUT_CODEC
    baseline_outer = G20_OUTER_PROFILE
    transform_codec = _interaction_hyperedge(
        "g25:selected_transform_bundle_x_selected_coder_container_bundle",
        {
            "00": point(identity_base, identity_population, baseline_layout, baseline_outer),
            "10": point(selected_base, selected_population, baseline_layout, baseline_outer),
            "01": point(identity_base, identity_population, config.layout_codec, config.outer_profile),
            "11": selected,
        },
        parsed,
    )
    return (base_population, transform_codec)


CheckpointCallback = Callable[[dict[str, Any]], None]


def search_population_global_recode_v2(
    source_archive: bytes,
    g20_control_archive: bytes,
    *,
    expected_source_archive_sha256: str = SOURCE_ARCHIVE_SHA256,
    expected_source_member_sha256: str = SOURCE_MEMBER_SHA256,
    expected_g20_archive_sha256: str = G20_CONTROL_ARCHIVE_SHA256,
    resume: SearchResumeStateV2 | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> PopulationGlobalSearchResultV2:
    """Run deterministic complete-ZIP coordinate descent over the real n600 state."""
    source = inspect_source_zip(
        source_archive,
        expected_archive_sha256=expected_source_archive_sha256,
        expected_member_sha256=expected_source_member_sha256,
    )
    parsed = parse_ep725_lvls1(source.member_bytes, require_source_form=True)
    if not hmac.compare_digest(_sha256(g20_control_archive), expected_g20_archive_sha256):
        raise Ep725PopulationGlobalRecodeError("G20 exact whole-object control SHA-256 drifted")
    try:
        with zipfile.ZipFile(io.BytesIO(g20_control_archive), mode="r") as control_zip:
            infos = control_zip.infolist()
            if len(infos) != 1 or infos[0].filename != MEMBER_NAME or control_zip.testzip() is not None:
                raise Ep725PopulationGlobalRecodeError("G20 control ZIP custody failed")
            g20_parsed = parse_ep725_lvls1(control_zip.read(infos[0]), require_source_form=False)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise Ep725PopulationGlobalRecodeError("G20 control ZIP reopen failed") from exc
    if _state_digest(parsed) != _state_digest(g20_parsed):
        raise Ep725PopulationGlobalRecodeError("G20 control is not the exact same quantized state")

    stages: list[dict[str, Any]] = []
    if resume is None:
        current = build_complete_archive(source, parsed, _initial_config(parsed))
        cycle_index = 0
        coordinate_index = 0
        cycle_start_sha = current.archive_sha256
        points_measured = 1
    else:
        current = build_complete_archive(source, parsed, resume.config)
        cycle_index = resume.cycle_index
        coordinate_index = resume.next_coordinate_index
        cycle_start_sha = resume.cycle_start_archive_sha256
        points_measured = resume.points_measured + 1

    while True:
        coordinate = _COORDINATES[coordinate_index]
        before = current
        current, points = _sweep(source, parsed, current, coordinate=coordinate)
        points_measured += len(points)
        next_coordinate = coordinate_index + 1
        next_cycle = cycle_index
        converged = False
        if next_coordinate == len(_COORDINATES):
            converged = current.archive_sha256 == cycle_start_sha
            next_coordinate = 0
            next_cycle += 1
        stage = {
            "schema": "tac.ep725_population_global_recode_search_stage.v2",
            "stage_id": f"cycle{cycle_index:03d}_{coordinate}",
            "cycle_index": cycle_index,
            "coordinate": coordinate,
            "complete_archives_measured": len(points),
            "candidate_set_sha256": _candidate_set_root(points, parsed),
            "before": _point_summary(before, parsed),
            "selected": _point_summary(current, parsed),
            "strict_whole_object_improvement": current.key() < before.key(),
            "top_complete_objects": [
                _point_summary(point, parsed) for point in sorted(points, key=CompleteArchivePointV2.key)[:8]
            ],
            "next_resume_state": {
                "config": config_to_json(current.config, parsed),
                "cycle_index": next_cycle,
                "next_coordinate_index": next_coordinate,
                "cycle_start_archive_sha256": (current.archive_sha256 if next_coordinate == 0 else cycle_start_sha),
                "points_measured": points_measured,
            },
            "converged_whole_cycle": converged,
        }
        stages.append(stage)
        if checkpoint_callback is not None:
            checkpoint_callback(stage)
        if converged:
            break
        if next_coordinate == 0:
            cycle_start_sha = current.archive_sha256
        cycle_index = next_cycle
        coordinate_index = next_coordinate

    # Exact selected parse-back, state equality, canonical re-encode, and ZIP custody.
    selected_parsed = parse_population_global_member(current.member_bytes)
    if _manifest_without_pg2(parsed.manifest) != _manifest_without_pg2(selected_parsed.manifest):
        raise Ep725PopulationGlobalRecodeError("selected manifest changed outside the pg2 descriptor")
    if parsed.pose_bytes != selected_parsed.pose_bytes:
        raise Ep725PopulationGlobalRecodeError("selected pose bytes changed")
    if not all(
        np.array_equal(parsed.base_quantized[name], selected_parsed.base_quantized[name]) for name in parsed.base_order
    ):
        raise Ep725PopulationGlobalRecodeError("selected base state is not exactly lossless")
    if not np.array_equal(parsed.code_quantized, selected_parsed.code_quantized):
        raise Ep725PopulationGlobalRecodeError("selected population state is not exactly lossless")
    source_state_sha = _state_digest(parsed)
    selected_state_sha = _state_digest(selected_parsed)
    if not hmac.compare_digest(source_state_sha, selected_state_sha):
        raise Ep725PopulationGlobalRecodeError("selected whole-state digest changed")
    if encode_population_global_member(parsed, current.config) != current.member_bytes:
        raise Ep725PopulationGlobalRecodeError("selected canonical member re-encode changed bytes")
    rebuilt_a = _clone_zip(source.info, source.archive_comment, current.member_bytes, current.config.outer_profile)
    rebuilt_b = _clone_zip(source.info, source.archive_comment, current.member_bytes, current.config.outer_profile)
    if rebuilt_a != current.archive_bytes or rebuilt_b != current.archive_bytes:
        raise Ep725PopulationGlobalRecodeError("selected archive double rebuild changed bytes")
    with zipfile.ZipFile(io.BytesIO(current.archive_bytes), mode="r") as reopened:
        infos = reopened.infolist()
        if len(infos) != 1 or infos[0].filename != MEMBER_NAME or reopened.read(infos[0]) != current.member_bytes:
            raise Ep725PopulationGlobalRecodeError("selected archive/member mapping changed")
        if reopened.testzip() is not None:
            raise Ep725PopulationGlobalRecodeError("selected archive CRC verification failed")
    interactions = _build_interactions(source, parsed, current)
    return PopulationGlobalSearchResultV2(
        source=source,
        source_lvls1=parsed,
        g20_control_archive_bytes=g20_control_archive,
        selected_v2=current,
        selected_parsed=selected_parsed,
        points_measured=points_measured,
        stages=tuple(stages),
        interaction_hyperedges=interactions,
        source_state_sha256=source_state_sha,
        selected_state_sha256=selected_state_sha,
        converged=True,
    )


__all__ = [
    "ACTION_SCHEMA",
    "G20_CONTROL_ARCHIVE_SHA256",
    "G20_LAYOUT_CODEC",
    "G20_OUTER_PROFILE",
    "G20_POPULATION_TRANSFORM",
    "IDENTITY_POPULATION_TRANSFORM",
    "INNER_CODECS",
    "LAYOUT_CODECS",
    "MAGIC",
    "OUTER_PROFILES",
    "POPULATION_TRANSFORMS",
    "SCHEMA",
    "SOURCE_ARCHIVE_SHA256",
    "SOURCE_MEMBER_SHA256",
    "CompleteArchivePointV2",
    "Ep725PopulationGlobalRecodeError",
    "LayoutCodecV2",
    "OuterProfileV2",
    "ParsedPopulationGlobalV2",
    "PopulationGlobalSearchResultV2",
    "PopulationTransformV2",
    "RecodeConfigV2",
    "SearchResumeStateV2",
    "build_complete_archive",
    "config_from_json",
    "config_to_json",
    "decode_population",
    "encode_population",
    "encode_population_global_member",
    "parse_population_global_member",
    "population_transform_menu",
    "search_population_global_recode_v2",
]
