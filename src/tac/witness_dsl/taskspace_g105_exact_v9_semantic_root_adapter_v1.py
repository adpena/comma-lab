# SPDX-License-Identifier: MIT
"""Exact V9 HOSC dual-head counted Y1 program and public NumPy receiver.

This module is an architecture-specific ``SemanticRootY1`` variant.  It does
not reinterpret the G103 ``ORIGINAL_COORDINR_FILM_MLP_V1`` model.  Its counted
packet contains:

* the exact public runtime configuration for the deterministic positional
  Fourier basis and ``tanh(beta * sin(omega * x))`` activation;
* power-of-two-quantized shared V9 ``in_proj`` / ``film`` / ``hidden`` /
  ``out_sdf`` / ``out_tex`` / ``palette`` tensors;
* optional, explicitly typed ``film_pl`` and ``concat_pl`` tensors; and
* only the 600 odd ``code[2*p+1]`` Y1 rows.

The even Y0 rows never enter this packet.  G94-V2 is their exclusive owner.
Phase advection is a training force which shapes these counted weights; it is
external encoder evidence, not a fictitious inflate-time operand.

The public plugin ABI is ``VARIANT_ID`` + ``parse_packet`` +
``render_scorer_y1``.  Source/G46/provenance evidence is verified by the
checkpoint intake and returned in a non-counted receipt.  It is never
serialized into candidate bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import struct
import zlib
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.witness_control.taskspace_fresh_teacher_materializer_v1 import (
    FreshTeacherMaterializationError,
    load_compile_ready_materialization_receipt,
)
from tac.witness_dsl.basis_control import (
    LEGACY_FOURIER_AB_CONTROL,
    normalize_basis_family,
)

VARIANT_ID: Final = "tac.semantic_root_y1.v9_hosc_dual_head_odd_y1.v1"
MAGIC: Final = b"SV9Y1V1\0"
VERSION: Final = 1
PAIR_COUNT_N600: Final = 600
SCORER_H: Final = 384
SCORER_W: Final = 512
SCORER_CHANNELS: Final = 3
N_CLASSES: Final = 5
MAX_PACKET_BYTES: Final = 2_000_000
MAX_TENSORS: Final = 16
MAX_TENSOR_RANK: Final = 4

UPSTREAM_SOURCE_CLOSURE_SCHEMA: Final = "tac.upstream_source_closure.v1"
UPSTREAM_SOURCE_CLOSURE_SHA256: Final = "e93f6c744fe0025ecc30d1f1cef00617a3f1397b68cadb856817766cfec63279"
G46_PATH_BOUND_RECEIPT_CLOSURE_SHA256: Final = (
    "9c588c725d66c6e840c157568fc5414c37f175348921c6353a64c6431a26cd99"
)
G46_TARGET_LABELS_SHA256: Final = (
    "6d2ca48ac07323c7fc3a5299023bc291363192e10130eb3bc63d446bb8e65b85"
)
G46_TARGET_LABELS_BYTES: Final = PAIR_COUNT_N600 * SCORER_H * SCORER_W
G46_SOURCE_PAIR_CHAIN_SHA256: Final = (
    "5b391fa4a5f651452fdf9a861af3f52abdc58017dcd8bfc0566ebcf86cab3559"
)
G46_MARGIN_AGGREGATE_SCHEMA: Final = "tac.taskspace_batch16_margin_base_scorer_aggregate.v1"
G46_PORTABLE_SOURCE_MEMBERS: Final = (
    (
        "evaluate.py",
        6005,
        "7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b",
    ),
    (
        "frame_utils.py",
        9345,
        "d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90",
    ),
    (
        "modules.py",
        8322,
        "065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa",
    ),
    (
        "public_test_video_names.txt",
        6,
        "7ff99d08c8351dd8167ec09213b758da5bbb705dedabe361ba881217374029a8",
    ),
)

_HEADER = struct.Struct(">8sBBHIII")
_SECTION = struct.Struct(">4sI")
_MODEL_HEADER = struct.Struct(">4sH")
_TENSOR_HEADER = struct.Struct(">BBbB4HI")
_Y1_HEADER = struct.Struct(">4sHHb3xI")
_SECTION_ORDER: Final = (b"CONF", b"MODL", b"Y1CD")
_MODEL_MAGIC: Final = b"V9M1"
_Y1_MAGIC: Final = b"Y1C1"
_FLAG_FILM_PL: Final = 1 << 0
_FLAG_CONCAT_PL: Final = 1 << 1
_FLAG_CHROMA: Final = 1 << 2
_KNOWN_FLAGS: Final = _FLAG_FILM_PL | _FLAG_CONCAT_PL | _FLAG_CHROMA


class ExactV9SemanticRootError(ValueError):
    """An exact V9 packet, checkpoint, or receiver invariant failed closed."""


class V9TensorDTypeV1(IntEnum):
    INT8 = 0
    INT16_LE = 1


def _sha256(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExactV9SemanticRootError(f"{name} must be lowercase SHA-256")
    return value


def _require_git_sha(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExactV9SemanticRootError("checkpoint git SHA must be a concrete lowercase 40-hex identity")
    return value


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
        raise ExactV9SemanticRootError("value is not finite canonical ASCII JSON") from exc


def _regular_file(path: Path, *, name: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    try:
        info = os.lstat(lexical)
    except FileNotFoundError as exc:
        raise ExactV9SemanticRootError(f"{name} does not exist: {lexical}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ExactV9SemanticRootError(f"{name} must be a non-symlink regular file: {lexical}")
    return lexical


def _scalar(value: np.ndarray, *, name: str) -> object:
    array = np.asarray(value)
    if array.size != 1:
        raise ExactV9SemanticRootError(f"{name} must be a scalar checkpoint field")
    return array.item()


@dataclass(frozen=True, slots=True)
class V9PolarFourierConfigV1:
    """The exact generic positional basis regenerated by the public receiver."""

    n_scales: int
    n_orient0: int
    f0: float
    base: float
    n_iso: int
    max_freq: float | None

    def __post_init__(self) -> None:
        if type(self.n_scales) is not int or not 1 <= self.n_scales <= 16:
            raise ExactV9SemanticRootError("n_scales must be an exact integer in [1,16]")
        if type(self.n_orient0) is not int or not 1 <= self.n_orient0 <= 64:
            raise ExactV9SemanticRootError("n_orient0 must be an exact integer in [1,64]")
        if type(self.n_iso) is not int or not 0 <= self.n_iso <= 64:
            raise ExactV9SemanticRootError("n_iso must be an exact integer in [0,64]")
        for name, value in (("f0", self.f0), ("base", self.base)):
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ExactV9SemanticRootError(f"{name} must be a finite positive float")
        if self.max_freq is not None and (
            type(self.max_freq) is not float
            or not math.isfinite(self.max_freq)
            or self.max_freq <= 0.0
        ):
            raise ExactV9SemanticRootError("max_freq must be None or a finite positive float")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "polar_directional_fourier.v1",
            "n_scales": self.n_scales,
            "n_orient0": self.n_orient0,
            "f0": self.f0,
            "base": self.base,
            "n_iso": self.n_iso,
            "max_freq": self.max_freq,
        }

    @classmethod
    def from_dict(cls, value: object) -> V9PolarFourierConfigV1:
        if type(value) is not dict or set(value) != {
            "kind",
            "n_scales",
            "n_orient0",
            "f0",
            "base",
            "n_iso",
            "max_freq",
        }:
            raise ExactV9SemanticRootError("basis config has a noncanonical key set")
        if value["kind"] != "polar_directional_fourier.v1":
            raise ExactV9SemanticRootError("only the exact polar-directional V9 basis is implemented")
        if (
            type(value["n_scales"]) is not int
            or type(value["n_orient0"]) is not int
            or type(value["n_iso"]) is not int
            or type(value["f0"]) not in {int, float}
            or type(value["base"]) not in {int, float}
            or (value["max_freq"] is not None and type(value["max_freq"]) not in {int, float})
        ):
            raise ExactV9SemanticRootError("basis config scalar types are not canonical")
        maximum = value["max_freq"]
        return cls(
            n_scales=int(value["n_scales"]),
            n_orient0=int(value["n_orient0"]),
            f0=float(value["f0"]),
            base=float(value["base"]),
            n_iso=int(value["n_iso"]),
            max_freq=None if maximum is None else float(maximum),
        )

    @property
    def input_dim(self) -> int:
        return 2 * _polar_fourier_b(self).shape[1]


@dataclass(frozen=True, slots=True)
class V9RuntimeConfigV1:
    """Every non-tensor operand consumed by the public V9 Y1 receiver."""

    input_dim: int
    hidden_dim: int
    hidden_layer_count: int
    modulation_dim: int
    softmax_temp: float
    hosc_beta: float
    hosc_omega: float
    chroma: bool
    film_per_layer: bool
    film_concat_code: bool
    basis: V9PolarFourierConfigV1

    def __post_init__(self) -> None:
        for name, value, lower, upper in (
            ("input_dim", self.input_dim, 2, 4096),
            ("hidden_dim", self.hidden_dim, 1, 1024),
            ("hidden_layer_count", self.hidden_layer_count, 1, 8),
            ("modulation_dim", self.modulation_dim, 1, 256),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise ExactV9SemanticRootError(f"{name} must be an exact integer in [{lower},{upper}]")
        for name, value in (
            ("softmax_temp", self.softmax_temp),
            ("hosc_beta", self.hosc_beta),
            ("hosc_omega", self.hosc_omega),
        ):
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ExactV9SemanticRootError(f"{name} must be a finite positive float")
        if type(self.chroma) is not bool:
            raise ExactV9SemanticRootError("chroma must be bool")
        if type(self.film_per_layer) is not bool or type(self.film_concat_code) is not bool:
            raise ExactV9SemanticRootError("optional FiLM route flags must be bool")
        if type(self.basis) is not V9PolarFourierConfigV1:
            raise ExactV9SemanticRootError("basis must be V9PolarFourierConfigV1")
        if self.input_dim != self.basis.input_dim:
            raise ExactV9SemanticRootError("input_dim disagrees with the exact positional basis width")

    @property
    def flags(self) -> int:
        return (
            (_FLAG_FILM_PL if self.film_per_layer else 0)
            | (_FLAG_CONCAT_PL if self.film_concat_code else 0)
            | (_FLAG_CHROMA if self.chroma else 0)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": VARIANT_ID,
            "activation": "hosc_tanh_beta_sin.v1",
            "pair_count": PAIR_COUNT_N600,
            "scorer_shape": [SCORER_H, SCORER_W, SCORER_CHANNELS],
            "n_classes": N_CLASSES,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_layer_count": self.hidden_layer_count,
            "modulation_dim": self.modulation_dim,
            "softmax_temp": self.softmax_temp,
            "hosc_beta": self.hosc_beta,
            "hosc_omega": self.hosc_omega,
            "chroma": self.chroma,
            "film_per_layer": self.film_per_layer,
            "film_concat_code": self.film_concat_code,
            "basis": self.basis.to_dict(),
            "y1_projection": "code[2*p+1]",
            "y0_owner": "G94-V2",
        }

    @classmethod
    def from_dict(cls, value: object) -> V9RuntimeConfigV1:
        if type(value) is not dict:
            raise ExactV9SemanticRootError("runtime config must be an object")
        expected = {
            "variant_id",
            "activation",
            "pair_count",
            "scorer_shape",
            "n_classes",
            "input_dim",
            "hidden_dim",
            "hidden_layer_count",
            "modulation_dim",
            "softmax_temp",
            "hosc_beta",
            "hosc_omega",
            "chroma",
            "film_per_layer",
            "film_concat_code",
            "basis",
            "y1_projection",
            "y0_owner",
        }
        if set(value) != expected:
            raise ExactV9SemanticRootError("runtime config has a noncanonical key set")
        if (
            value["variant_id"] != VARIANT_ID
            or value["activation"] != "hosc_tanh_beta_sin.v1"
            or value["pair_count"] != PAIR_COUNT_N600
            or value["scorer_shape"] != [SCORER_H, SCORER_W, SCORER_CHANNELS]
            or value["n_classes"] != N_CLASSES
            or value["y1_projection"] != "code[2*p+1]"
            or value["y0_owner"] != "G94-V2"
        ):
            raise ExactV9SemanticRootError("runtime config changes a sealed V9/Y1 contract")
        if (
            type(value["input_dim"]) is not int
            or type(value["hidden_dim"]) is not int
            or type(value["hidden_layer_count"]) is not int
            or type(value["modulation_dim"]) is not int
            or type(value["softmax_temp"]) not in {int, float}
            or type(value["hosc_beta"]) not in {int, float}
            or type(value["hosc_omega"]) not in {int, float}
            or type(value["chroma"]) is not bool
            or type(value["film_per_layer"]) is not bool
            or type(value["film_concat_code"]) is not bool
        ):
            raise ExactV9SemanticRootError("runtime config scalar types are not canonical")
        return cls(
            input_dim=int(value["input_dim"]),
            hidden_dim=int(value["hidden_dim"]),
            hidden_layer_count=int(value["hidden_layer_count"]),
            modulation_dim=int(value["modulation_dim"]),
            softmax_temp=float(value["softmax_temp"]),
            hosc_beta=float(value["hosc_beta"]),
            hosc_omega=float(value["hosc_omega"]),
            chroma=value["chroma"],
            film_per_layer=value["film_per_layer"],
            film_concat_code=value["film_concat_code"],
            basis=V9PolarFourierConfigV1.from_dict(value["basis"]),
        )


@dataclass(frozen=True, slots=True)
class V9CountedTensorV1:
    name: str
    dtype: V9TensorDTypeV1
    shape: tuple[int, ...]
    scale_exponent: int
    data: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or not self.name.isascii()
            or len(self.name.encode("ascii")) > 255
        ):
            raise ExactV9SemanticRootError("tensor name must be nonempty bounded ASCII")
        if type(self.dtype) is not V9TensorDTypeV1:
            raise ExactV9SemanticRootError("tensor dtype is not typed")
        if type(self.shape) is not tuple or not 1 <= len(self.shape) <= MAX_TENSOR_RANK:
            raise ExactV9SemanticRootError("tensor shape rank must be in [1,4]")
        if any(type(value) is not int or not 1 <= value <= 0xFFFF for value in self.shape):
            raise ExactV9SemanticRootError("tensor dimensions must be exact positive uint16 values")
        if type(self.scale_exponent) is not int or not -64 <= self.scale_exponent <= 63:
            raise ExactV9SemanticRootError("tensor scale_exponent must be an exact int8-range exponent")
        if type(self.data) is not bytes:
            raise ExactV9SemanticRootError("tensor data must be exact bytes")
        count = math.prod(self.shape)
        itemsize = 1 if self.dtype is V9TensorDTypeV1.INT8 else 2
        if len(self.data) != count * itemsize:
            raise ExactV9SemanticRootError("tensor data length disagrees with shape and dtype")

    @property
    def quantized(self) -> np.ndarray:
        dtype = np.dtype("i1") if self.dtype is V9TensorDTypeV1.INT8 else np.dtype("<i2")
        return np.frombuffer(self.data, dtype=dtype).reshape(self.shape).copy()

    @property
    def dequantized(self) -> np.ndarray:
        return np.ldexp(self.quantized.astype(np.float32), self.scale_exponent).astype(np.float32)


@dataclass(frozen=True, slots=True)
class ExactV9SemanticRootY1ProgramV1:
    config: V9RuntimeConfigV1
    tensors: tuple[V9CountedTensorV1, ...]
    y1_code_scale_exponent: int
    y1_code_q: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.config) is not V9RuntimeConfigV1:
            raise ExactV9SemanticRootError("program config is not V9RuntimeConfigV1")
        if type(self.tensors) is not tuple or not self.tensors:
            raise ExactV9SemanticRootError("program tensors must be a nonempty tuple")
        if len(self.tensors) > MAX_TENSORS or any(
            type(tensor) is not V9CountedTensorV1 for tensor in self.tensors
        ):
            raise ExactV9SemanticRootError("program tensor tuple is malformed")
        if type(self.y1_code_scale_exponent) is not int or not -64 <= self.y1_code_scale_exponent <= 63:
            raise ExactV9SemanticRootError("Y1 code scale exponent is invalid")
        code = np.asarray(self.y1_code_q)
        if code.dtype != np.dtype("<i2") or code.shape != (
            PAIR_COUNT_N600,
            self.config.modulation_dim,
        ):
            raise ExactV9SemanticRootError("Y1 code must be canonical little-endian int16[600,mod_dim]")
        if not code.flags.c_contiguous:
            raise ExactV9SemanticRootError("Y1 code must be C-contiguous")
        _validate_tensor_abi(self.config, self.tensors)

    @property
    def params(self) -> dict[str, np.ndarray]:
        """Expand the typed stacked wire tensors into repository V9 parameter keys."""

        stored = {tensor.name: tensor.dequantized for tensor in self.tensors}
        result = {
            "in_proj.weight": stored["in_proj.weight"],
            "in_proj.bias": stored["in_proj.bias"],
            "film.weight": stored["film.weight"],
            "film.bias": stored["film.bias"],
            "out_sdf.weight": stored["out_sdf.weight"],
            "out_sdf.bias": stored["out_sdf.bias"],
            "out_tex.weight": stored["out_tex.weight"],
            "out_tex.bias": stored["out_tex.bias"],
            "palette": stored["palette"],
        }
        for index in range(self.config.hidden_layer_count):
            result[f"hidden.{index}.weight"] = stored["hidden.weight"][index]
            result[f"hidden.{index}.bias"] = stored["hidden.bias"][index]
            if self.config.film_per_layer:
                result[f"film_pl.{index}.weight"] = stored["film_pl.weight"][index]
                result[f"film_pl.{index}.bias"] = stored["film_pl.bias"][index]
            if self.config.film_concat_code:
                result[f"concat_pl.{index}.weight"] = stored["concat_pl.weight"][index]
                result[f"concat_pl.{index}.bias"] = stored["concat_pl.bias"][index]
        return result

    @property
    def y1_code(self) -> np.ndarray:
        return np.ldexp(
            np.asarray(self.y1_code_q, dtype=np.float32),
            self.y1_code_scale_exponent,
        ).astype(np.float32)


@dataclass(frozen=True, slots=True)
class V9PhaseAdvectionTrainingEvidenceV1:
    """External encoder evidence; never serialized into the public packet."""

    weight: float
    start_epoch: int
    classes: tuple[int, ...]
    band: float
    gap_xi: str
    reference: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if type(self.weight) is not float or not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ExactV9SemanticRootError("phase-advection weight must be finite and positive")
        if type(self.start_epoch) is not int or self.start_epoch < 0:
            raise ExactV9SemanticRootError("phase-advection start epoch must be a nonnegative integer")
        if self.classes != (0, 1, 2):
            raise ExactV9SemanticRootError("phase-advection runtime lineage must bind ground classes (0,1,2)")
        if type(self.band) is not float or not math.isfinite(self.band) or self.band <= 0.0:
            raise ExactV9SemanticRootError("phase-advection band must be finite and positive")
        if self.gap_xi != "interp":
            raise ExactV9SemanticRootError("only the implemented phase-advection interp gap is admitted")
        if self.reference not in {"gt_advected", "gt_advected_with_own_tie_fallback"}:
            raise ExactV9SemanticRootError("phase-advection evidence names an unsupported reference")
        _require_sha256(self.evidence_sha256, name="phase-advection evidence")

    def identity_sha256(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "schema": "tac.v9_phase_advection_training_evidence.v1",
                    "weight": self.weight,
                    "start_epoch": self.start_epoch,
                    "classes": list(self.classes),
                    "band": self.band,
                    "gap_xi": self.gap_xi,
                    "reference": self.reference,
                    "evidence_sha256": self.evidence_sha256,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class V9G46Batch16TrainingTargetEvidenceV1:
    """External proof that the producer consumed the G46 batch-16 target fiber.

    G46 owns exact argmax labels but not the winner-margin field consumed by
    V9's saliency, tie-locus, phase-advection, horizon, birth, and costate
    paths. An old ``gt_n600.npz`` plus G46 labels is therefore insufficient:
    margins must come from the same batch-16 scorer forwards, and the live
    verdict/controller geometry must remain at 16. These identities are
    checkpoint intake evidence and never candidate bytes.
    """

    active_target_authority_sha256: str
    target_margins_sha256: str
    margin_aggregate_receipt_sha256: str
    consumer_binding_sha256: str
    evidence_sha256: str
    target_labels_sha256: str = G46_TARGET_LABELS_SHA256
    source_pair_chain_sha256: str = G46_SOURCE_PAIR_CHAIN_SHA256
    scorer_pair_batch_size: int = 16
    margins_from_same_batch16_forward: bool = True
    live_verdict_batch_size: int = 16
    margin_aggregate_schema: str = G46_MARGIN_AGGREGATE_SCHEMA

    def __post_init__(self) -> None:
        for name, value in (
            ("active target authority", self.active_target_authority_sha256),
            ("target margins", self.target_margins_sha256),
            ("margin aggregate receipt", self.margin_aggregate_receipt_sha256),
            ("consumer binding", self.consumer_binding_sha256),
            ("target evidence", self.evidence_sha256),
        ):
            _require_sha256(value, name=name)
        if self.target_labels_sha256 != G46_TARGET_LABELS_SHA256:
            raise ExactV9SemanticRootError("producer target labels are not the exact G46 n600 labels")
        if self.source_pair_chain_sha256 != G46_SOURCE_PAIR_CHAIN_SHA256:
            raise ExactV9SemanticRootError("producer source-pair chain is not the G46 n600 chain")
        if self.scorer_pair_batch_size != 16 or self.live_verdict_batch_size != 16:
            raise ExactV9SemanticRootError("producer target and live verdict batch geometry must both be 16")
        if self.margins_from_same_batch16_forward is not True:
            raise ExactV9SemanticRootError("target margins are not proven from the G46 batch-16 forwards")
        if self.margin_aggregate_schema != G46_MARGIN_AGGREGATE_SCHEMA:
            raise ExactV9SemanticRootError("target-margin aggregate schema is not the reviewed batch-16 ABI")

    def identity_sha256(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "schema": "tac.g105_g46_batch16_training_target_evidence.v1",
                    "active_target_authority_sha256": self.active_target_authority_sha256,
                    "target_labels_sha256": self.target_labels_sha256,
                    "target_margins_sha256": self.target_margins_sha256,
                    "source_pair_chain_sha256": self.source_pair_chain_sha256,
                    "scorer_pair_batch_size": self.scorer_pair_batch_size,
                    "margins_from_same_batch16_forward": self.margins_from_same_batch16_forward,
                    "live_verdict_batch_size": self.live_verdict_batch_size,
                    "margin_aggregate_schema": self.margin_aggregate_schema,
                    "margin_aggregate_receipt_sha256": self.margin_aggregate_receipt_sha256,
                    "consumer_binding_sha256": self.consumer_binding_sha256,
                    "evidence_sha256": self.evidence_sha256,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class V9CandidateWireAccountingV1:
    packet_bytes: int
    packet_sha256: str
    header_bytes: int
    section_directory_bytes: int
    config_bytes: int
    model_section_bytes: int
    model_tensor_data_bytes: int
    model_tensor_metadata_bytes: int
    y1_section_bytes: int
    y1_code_data_bytes: int
    y1_code_metadata_bytes: int
    counted_y1_rows: int
    excluded_y0_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_bytes": self.packet_bytes,
            "packet_sha256": self.packet_sha256,
            "header_bytes": self.header_bytes,
            "section_directory_bytes": self.section_directory_bytes,
            "config_bytes": self.config_bytes,
            "model_section_bytes": self.model_section_bytes,
            "model_tensor_data_bytes": self.model_tensor_data_bytes,
            "model_tensor_metadata_bytes": self.model_tensor_metadata_bytes,
            "y1_section_bytes": self.y1_section_bytes,
            "y1_code_data_bytes": self.y1_code_data_bytes,
            "y1_code_metadata_bytes": self.y1_code_metadata_bytes,
            "counted_y1_rows": self.counted_y1_rows,
            "excluded_y0_rows": self.excluded_y0_rows,
            "outer_zip_bytes_measured": False,
            "candidate_or_score_claim": False,
        }


@dataclass(frozen=True, slots=True)
class CompiledFreshV9SemanticRootV1:
    program: ExactV9SemanticRootY1ProgramV1
    packet: bytes = field(repr=False)
    wire_accounting: V9CandidateWireAccountingV1
    external_receipt: dict[str, Any]


def _polar_fourier_b(config: V9PolarFourierConfigV1) -> np.ndarray:
    columns: list[np.ndarray] = []
    for scale_index in range(config.n_scales):
        frequency = config.f0 * (config.base**scale_index)
        orientation_count = config.n_orient0 * (2 ** (scale_index // 2))
        for orientation_index in range(orientation_count):
            theta = np.pi * orientation_index / orientation_count
            columns.append(
                np.array(
                    [frequency * np.cos(theta), frequency * np.sin(theta)],
                    dtype=np.float32,
                )
            )
    for index in range(config.n_iso):
        theta = np.pi * index / max(config.n_iso, 1)
        frequency = config.f0 * 0.5
        columns.append(
            np.array(
                [frequency * np.cos(theta), frequency * np.sin(theta)],
                dtype=np.float32,
            )
        )
    if not columns:
        raise ExactV9SemanticRootError("positional basis cannot be empty")
    result = np.stack(columns, axis=1).astype(np.float32)
    if config.max_freq is not None:
        norms = np.sqrt((result.astype(np.float64) ** 2).sum(axis=0))
        keep = norms <= config.max_freq + 1e-6
        if not keep.any():
            keep = norms <= float(norms.min()) + 1e-6
        result = result[:, keep]
    return result


def build_runtime_features(config: V9RuntimeConfigV1) -> np.ndarray:
    """Regenerate the exact free positional feature grid for the public receiver."""

    ys = np.linspace(-1.0, 1.0, SCORER_H, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, SCORER_W, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).astype(np.float32)
    basis = _polar_fourier_b(config.basis)
    with np.errstate(all="ignore"):
        projection = (2.0 * np.pi) * (
            np.asarray(coords, np.float64) @ np.asarray(basis, np.float64)
        )
        features = np.concatenate(
            [np.sin(projection), np.cos(projection)],
            axis=-1,
        ).astype(np.float32)
    if features.shape != (SCORER_H * SCORER_W, config.input_dim):
        raise ExactV9SemanticRootError("regenerated positional features disagree with runtime input_dim")
    return features


def _hosc(values: np.ndarray, *, beta: float, omega: float) -> np.ndarray:
    """Op-for-op activation mirror, including repository float32 stage casts."""

    return np.tanh(beta * np.sin(omega * np.asarray(values, np.float64))).astype(np.float32)


def forward_float32(
    program: ExactV9SemanticRootY1ProgramV1,
    pair_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent deterministic NumPy mirror of the repository V9 dual-head forward."""

    if type(program) is not ExactV9SemanticRootY1ProgramV1:
        raise ExactV9SemanticRootError("forward requires ExactV9SemanticRootY1ProgramV1")
    if type(pair_id) is not int or not 0 <= pair_id < PAIR_COUNT_N600:
        raise ExactV9SemanticRootError("pair_id must be an exact integer in [0,599]")
    config = program.config
    params = {name: np.asarray(value, np.float64) for name, value in program.params.items()}
    features = np.asarray(build_runtime_features(config), np.float64)
    code = np.asarray(program.y1_code[pair_id], np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        hidden = _hosc(
            features @ params["in_proj.weight"].T + params["in_proj.bias"],
            beta=config.hosc_beta,
            omega=config.hosc_omega,
        )
        film = (
            code @ params["film.weight"].T + params["film.bias"]
        ).reshape(config.hidden_layer_count, 2, config.hidden_dim)
        for index in range(config.hidden_layer_count):
            scale = 1.0 + film[index, 0]
            shift = film[index, 1]
            if config.film_per_layer:
                per_layer = (
                    code @ params[f"film_pl.{index}.weight"].T
                    + params[f"film_pl.{index}.bias"]
                ).reshape(2, config.hidden_dim)
                scale = scale + per_layer[0]
                shift = shift + per_layer[1]
            preactivation = (
                hidden @ params[f"hidden.{index}.weight"].T
                + params[f"hidden.{index}.bias"]
            ) * scale + shift
            if config.film_concat_code:
                preactivation = (
                    preactivation
                    + code @ params[f"concat_pl.{index}.weight"].T
                    + params[f"concat_pl.{index}.bias"]
                )
            hidden = _hosc(
                preactivation,
                beta=config.hosc_beta,
                omega=config.hosc_omega,
            )
        phi = hidden @ params["out_sdf.weight"].T + params["out_sdf.bias"]
        texture = hidden @ params["out_tex.weight"].T + params["out_tex.bias"]
        logits = phi / config.softmax_temp
        logits = logits - logits.max(axis=-1, keepdims=True)
        soft = np.exp(logits)
        soft = soft / soft.sum(axis=-1, keepdims=True)
        base = soft @ params["palette"]
        rgb = (1.0 / (1.0 + np.exp(-(base + texture)))) * 255.0
        if not config.chroma:
            luma = (
                0.299 * rgb[:, 0:1]
                + 0.587 * rgb[:, 1:2]
                + 0.114 * rgb[:, 2:3]
            )
            rgb = np.concatenate([luma, luma, luma], axis=-1)
    return rgb.astype(np.float32), phi.astype(np.float32)


def render_scorer_y1(
    parsed: ExactV9SemanticRootY1ProgramV1,
    pair_id: int,
) -> np.ndarray:
    """Public plugin ABI: render scorer-grid Y1 as deterministic uint8 RGB."""

    rgb, _ = forward_float32(parsed, pair_id)
    return np.clip(np.rint(rgb), 0.0, 255.0).astype(np.uint8).reshape(
        SCORER_H,
        SCORER_W,
        SCORER_CHANNELS,
    )


def _expected_tensor_shapes(config: V9RuntimeConfigV1) -> tuple[tuple[str, tuple[int, ...]], ...]:
    hidden = config.hidden_dim
    layers = config.hidden_layer_count
    modulation = config.modulation_dim
    expected: list[tuple[str, tuple[int, ...]]] = [
        ("in_proj.weight", (hidden, config.input_dim)),
        ("in_proj.bias", (hidden,)),
        ("film.weight", (2 * hidden * layers, modulation)),
        ("film.bias", (2 * hidden * layers,)),
        ("hidden.weight", (layers, hidden, hidden)),
        ("hidden.bias", (layers, hidden)),
    ]
    if config.film_per_layer:
        expected.extend(
            [
                ("film_pl.weight", (layers, 2 * hidden, modulation)),
                ("film_pl.bias", (layers, 2 * hidden)),
            ]
        )
    if config.film_concat_code:
        expected.extend(
            [
                ("concat_pl.weight", (layers, hidden, modulation)),
                ("concat_pl.bias", (layers, hidden)),
            ]
        )
    expected.extend(
        [
            ("out_sdf.weight", (N_CLASSES, hidden)),
            ("out_sdf.bias", (N_CLASSES,)),
            ("out_tex.weight", (SCORER_CHANNELS, hidden)),
            ("out_tex.bias", (SCORER_CHANNELS,)),
            ("palette", (N_CLASSES, SCORER_CHANNELS)),
        ]
    )
    return tuple(expected)


def _validate_tensor_abi(
    config: V9RuntimeConfigV1,
    tensors: tuple[V9CountedTensorV1, ...],
) -> None:
    observed = tuple((tensor.name, tensor.shape) for tensor in tensors)
    expected = _expected_tensor_shapes(config)
    if observed != expected:
        raise ExactV9SemanticRootError("counted tensor order/shapes disagree with exact V9 ABI")
    for tensor in tensors:
        expected_dtype = (
            V9TensorDTypeV1.INT8
            if tensor.name.endswith(".weight")
            else V9TensorDTypeV1.INT16_LE
        )
        if tensor.dtype is not expected_dtype:
            raise ExactV9SemanticRootError(f"{tensor.name} dtype disagrees with exact V9 ABI")


def _quantize_pow2(
    name: str,
    values: np.ndarray,
    *,
    dtype: V9TensorDTypeV1,
) -> V9CountedTensorV1:
    array = np.asarray(values, dtype=np.float32)
    if not array.size or not np.isfinite(array).all():
        raise ExactV9SemanticRootError(f"{name} must be a finite nonempty tensor")
    limit = 127 if dtype is V9TensorDTypeV1.INT8 else 32767
    maximum = float(np.max(np.abs(array.astype(np.float64))))
    exponent = -32 if maximum == 0.0 else math.ceil(math.log2(maximum / limit))
    if not -64 <= exponent <= 63:
        raise ExactV9SemanticRootError(f"{name} cannot be represented by the power-of-two wire quantizer")
    scaled = np.rint(np.ldexp(array.astype(np.float64), -exponent))
    if np.any(scaled < -limit) or np.any(scaled > limit):
        raise ExactV9SemanticRootError(f"{name} overflowed the power-of-two wire quantizer")
    if dtype is V9TensorDTypeV1.INT8:
        quantized = np.ascontiguousarray(scaled.astype(np.int8))
    else:
        quantized = np.ascontiguousarray(scaled.astype("<i2"))
    return V9CountedTensorV1(
        name=name,
        dtype=dtype,
        shape=tuple(int(value) for value in quantized.shape),
        scale_exponent=exponent,
        data=quantized.tobytes(order="C"),
    )


def _stack_checkpoint_params(
    params: dict[str, np.ndarray],
    config: V9RuntimeConfigV1,
) -> dict[str, np.ndarray]:
    result = {
        "in_proj.weight": params["in_proj.weight"],
        "in_proj.bias": params["in_proj.bias"],
        "film.weight": params["film.weight"],
        "film.bias": params["film.bias"],
        "hidden.weight": np.stack(
            [params[f"hidden.{index}.weight"] for index in range(config.hidden_layer_count)]
        ),
        "hidden.bias": np.stack(
            [params[f"hidden.{index}.bias"] for index in range(config.hidden_layer_count)]
        ),
    }
    if config.film_per_layer:
        result["film_pl.weight"] = np.stack(
            [params[f"film_pl.{index}.weight"] for index in range(config.hidden_layer_count)]
        )
        result["film_pl.bias"] = np.stack(
            [params[f"film_pl.{index}.bias"] for index in range(config.hidden_layer_count)]
        )
    if config.film_concat_code:
        result["concat_pl.weight"] = np.stack(
            [params[f"concat_pl.{index}.weight"] for index in range(config.hidden_layer_count)]
        )
        result["concat_pl.bias"] = np.stack(
            [params[f"concat_pl.{index}.bias"] for index in range(config.hidden_layer_count)]
        )
    result.update(
        {
            "out_sdf.weight": params["out_sdf.weight"],
            "out_sdf.bias": params["out_sdf.bias"],
            "out_tex.weight": params["out_tex.weight"],
            "out_tex.bias": params["out_tex.bias"],
            "palette": params["palette"],
        }
    )
    return result


def compile_from_state(
    *,
    config: V9RuntimeConfigV1,
    params: dict[str, np.ndarray],
    interleaved_code: np.ndarray,
) -> ExactV9SemanticRootY1ProgramV1:
    """Compile an exact V9 state; only odd Y1 rows become counted state."""

    if type(config) is not V9RuntimeConfigV1:
        raise ExactV9SemanticRootError("compile config is not V9RuntimeConfigV1")
    if type(params) is not dict:
        raise ExactV9SemanticRootError("params must be a concrete dictionary")
    stacked = _stack_checkpoint_params(params, config)
    expected_names = tuple(name for name, _ in _expected_tensor_shapes(config))
    if tuple(stacked) != expected_names:
        raise ExactV9SemanticRootError("stacked parameter order disagrees with exact V9 ABI")
    tensors = tuple(
        _quantize_pow2(
            name,
            stacked[name],
            dtype=(
                V9TensorDTypeV1.INT8
                if name.endswith(".weight")
                else V9TensorDTypeV1.INT16_LE
            ),
        )
        for name in expected_names
    )
    code = np.asarray(interleaved_code, dtype=np.float32)
    if code.shape != (2 * PAIR_COUNT_N600, config.modulation_dim) or not np.isfinite(code).all():
        raise ExactV9SemanticRootError("interleaved code must be finite float[1200,modulation_dim]")
    y1 = np.ascontiguousarray(code[1::2])
    maximum = float(np.max(np.abs(y1.astype(np.float64))))
    exponent = -32 if maximum == 0.0 else math.ceil(math.log2(maximum / 32767.0))
    if not -64 <= exponent <= 63:
        raise ExactV9SemanticRootError("Y1 code cannot be represented by the power-of-two wire quantizer")
    quantized = np.rint(np.ldexp(y1.astype(np.float64), -exponent))
    if np.any(quantized < -32767) or np.any(quantized > 32767):
        raise ExactV9SemanticRootError("Y1 code overflowed the power-of-two wire quantizer")
    y1_q = np.ascontiguousarray(quantized.astype("<i2"))
    return ExactV9SemanticRootY1ProgramV1(
        config=config,
        tensors=tensors,
        y1_code_scale_exponent=exponent,
        y1_code_q=y1_q,
    )


def _encode_model(tensors: tuple[V9CountedTensorV1, ...]) -> bytes:
    rows = [_MODEL_HEADER.pack(_MODEL_MAGIC, len(tensors))]
    for tensor in tensors:
        name = tensor.name.encode("ascii")
        padded = tensor.shape + (0,) * (MAX_TENSOR_RANK - len(tensor.shape))
        rows.append(
            bytes([len(name)])
            + name
            + _TENSOR_HEADER.pack(
                int(tensor.dtype),
                len(tensor.shape),
                tensor.scale_exponent,
                0,
                *padded,
                len(tensor.data),
            )
            + tensor.data
        )
    return b"".join(rows)


def _decode_model(payload: bytes) -> tuple[V9CountedTensorV1, ...]:
    if len(payload) < _MODEL_HEADER.size:
        raise ExactV9SemanticRootError("model section is truncated")
    magic, count = _MODEL_HEADER.unpack_from(payload)
    if magic != _MODEL_MAGIC or not 1 <= count <= MAX_TENSORS:
        raise ExactV9SemanticRootError("model section header is invalid")
    cursor = _MODEL_HEADER.size
    result = []
    for _ in range(count):
        if cursor >= len(payload):
            raise ExactV9SemanticRootError("model tensor name length is truncated")
        name_length = payload[cursor]
        cursor += 1
        if not name_length or cursor + name_length + _TENSOR_HEADER.size > len(payload):
            raise ExactV9SemanticRootError("model tensor header is truncated")
        try:
            name = payload[cursor : cursor + name_length].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ExactV9SemanticRootError("model tensor name is not ASCII") from exc
        cursor += name_length
        row = _TENSOR_HEADER.unpack_from(payload, cursor)
        cursor += _TENSOR_HEADER.size
        dtype_raw, rank, exponent, reserved, *tail = row
        dimensions = tuple(int(value) for value in tail[:4])
        byte_length = int(tail[4])
        if (
            reserved != 0
            or not 1 <= rank <= MAX_TENSOR_RANK
            or any(dimensions[index] <= 0 for index in range(rank))
            or any(dimensions[index] != 0 for index in range(rank, MAX_TENSOR_RANK))
            or cursor + byte_length > len(payload)
        ):
            raise ExactV9SemanticRootError("model tensor shape/length metadata is invalid")
        try:
            dtype = V9TensorDTypeV1(dtype_raw)
        except ValueError as exc:
            raise ExactV9SemanticRootError("model tensor dtype is unknown") from exc
        result.append(
            V9CountedTensorV1(
                name=name,
                dtype=dtype,
                shape=dimensions[:rank],
                scale_exponent=int(exponent),
                data=payload[cursor : cursor + byte_length],
            )
        )
        cursor += byte_length
    if cursor != len(payload):
        raise ExactV9SemanticRootError("model section has hidden/trailing bytes")
    return tuple(result)


def _encode_y1(program: ExactV9SemanticRootY1ProgramV1) -> bytes:
    data = np.asarray(program.y1_code_q, dtype="<i2").tobytes(order="C")
    return _Y1_HEADER.pack(
        _Y1_MAGIC,
        PAIR_COUNT_N600,
        program.config.modulation_dim,
        program.y1_code_scale_exponent,
        len(data),
    ) + data


def _decode_y1(payload: bytes, *, modulation_dim: int) -> tuple[int, np.ndarray]:
    if len(payload) < _Y1_HEADER.size:
        raise ExactV9SemanticRootError("Y1 code section is truncated")
    magic, pair_count, inner_modulation_dim, exponent, byte_length = _Y1_HEADER.unpack_from(payload)
    if (
        magic != _Y1_MAGIC
        or pair_count != PAIR_COUNT_N600
        or inner_modulation_dim != modulation_dim
        or byte_length != PAIR_COUNT_N600 * modulation_dim * 2
        or len(payload) != _Y1_HEADER.size + byte_length
    ):
        raise ExactV9SemanticRootError("Y1 code header disagrees with the exact odd-row contract")
    code = np.frombuffer(payload[_Y1_HEADER.size :], dtype="<i2").reshape(
        PAIR_COUNT_N600,
        modulation_dim,
    )
    return int(exponent), np.ascontiguousarray(code)


def encode_packet(program: ExactV9SemanticRootY1ProgramV1) -> bytes:
    if type(program) is not ExactV9SemanticRootY1ProgramV1:
        raise ExactV9SemanticRootError("encode requires ExactV9SemanticRootY1ProgramV1")
    sections = (
        _canonical_json(program.config.to_dict()),
        _encode_model(program.tensors),
        _encode_y1(program),
    )
    body = b"".join(
        _SECTION.pack(tag, len(section)) + section
        for tag, section in zip(_SECTION_ORDER, sections, strict=True)
    )
    payload = _HEADER.pack(
        MAGIC,
        VERSION,
        program.config.flags,
        len(sections),
        len(body),
        zlib.crc32(body) & 0xFFFFFFFF,
        0,
    ) + body
    if len(payload) > MAX_PACKET_BYTES:
        raise ExactV9SemanticRootError("exact V9 semantic-root packet exceeds the bounded ABI")
    return payload


def _split_sections(payload: bytes) -> tuple[int, dict[bytes, bytes]]:
    if type(payload) is not bytes or len(payload) < _HEADER.size:
        raise ExactV9SemanticRootError("packet must be exact nontruncated bytes")
    magic, version, flags, count, body_length, checksum, reserved = _HEADER.unpack_from(payload)
    body = payload[_HEADER.size :]
    if (
        magic != MAGIC
        or version != VERSION
        or flags & ~_KNOWN_FLAGS
        or count != len(_SECTION_ORDER)
        or reserved != 0
        or body_length != len(body)
        or len(payload) > MAX_PACKET_BYTES
        or zlib.crc32(body) & 0xFFFFFFFF != checksum
    ):
        raise ExactV9SemanticRootError("packet header, length, flags, or CRC is invalid")
    cursor = 0
    sections: dict[bytes, bytes] = {}
    for expected_tag in _SECTION_ORDER:
        if cursor + _SECTION.size > len(body):
            raise ExactV9SemanticRootError("packet section directory is truncated")
        tag, length = _SECTION.unpack_from(body, cursor)
        cursor += _SECTION.size
        if tag != expected_tag or tag in sections or cursor + length > len(body):
            raise ExactV9SemanticRootError("packet sections are unknown, duplicate, reordered, or truncated")
        sections[tag] = body[cursor : cursor + length]
        cursor += length
    if cursor != len(body):
        raise ExactV9SemanticRootError("packet has hidden/trailing bytes")
    return int(flags), sections


def parse_packet(payload: bytes) -> ExactV9SemanticRootY1ProgramV1:
    """Public plugin ABI: strict parse with exact canonical re-encode identity."""

    flags, sections = _split_sections(payload)
    try:
        config_value = json.loads(sections[b"CONF"].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExactV9SemanticRootError("runtime config is not ASCII JSON") from exc
    if _canonical_json(config_value) != sections[b"CONF"]:
        raise ExactV9SemanticRootError("runtime config changed under canonical parse/re-emit")
    config = V9RuntimeConfigV1.from_dict(config_value)
    if config.flags != flags:
        raise ExactV9SemanticRootError("packet flags disagree with runtime config")
    tensors = _decode_model(sections[b"MODL"])
    exponent, y1_code = _decode_y1(
        sections[b"Y1CD"],
        modulation_dim=config.modulation_dim,
    )
    program = ExactV9SemanticRootY1ProgramV1(
        config=config,
        tensors=tensors,
        y1_code_scale_exponent=exponent,
        y1_code_q=y1_code,
    )
    if encode_packet(program) != payload:
        raise ExactV9SemanticRootError("packet changed under strict parse/re-emit")
    return program


def candidate_wire_accounting(payload: bytes) -> V9CandidateWireAccountingV1:
    _, sections = _split_sections(payload)
    tensors = _decode_model(sections[b"MODL"])
    tensor_data = sum(len(tensor.data) for tensor in tensors)
    y1_data = PAIR_COUNT_N600 * parse_packet(payload).config.modulation_dim * 2
    return V9CandidateWireAccountingV1(
        packet_bytes=len(payload),
        packet_sha256=_sha256(payload),
        header_bytes=_HEADER.size,
        section_directory_bytes=len(_SECTION_ORDER) * _SECTION.size,
        config_bytes=len(sections[b"CONF"]),
        model_section_bytes=len(sections[b"MODL"]),
        model_tensor_data_bytes=tensor_data,
        model_tensor_metadata_bytes=len(sections[b"MODL"]) - tensor_data,
        y1_section_bytes=len(sections[b"Y1CD"]),
        y1_code_data_bytes=y1_data,
        y1_code_metadata_bytes=len(sections[b"Y1CD"]) - y1_data,
        counted_y1_rows=PAIR_COUNT_N600,
        excluded_y0_rows=PAIR_COUNT_N600,
    )


def _portable_g46_receipt_identity(path: Path) -> dict[str, Any]:
    receipt_path = _regular_file(path, name="G46 encoder receipt")
    try:
        value = load_compile_ready_materialization_receipt(receipt_path)
    except (FreshTeacherMaterializationError, OSError, ValueError) as exc:
        raise ExactV9SemanticRootError("G46 encoder receipt is not canonically compile-ready") from exc
    if (
        type(value) is not dict
        or value.get("encoder_only") is not True
        or value.get("candidate_payload_allowed") is not False
        or value.get("pair_count") != PAIR_COUNT_N600
        or value.get("batch_size") != 16
        or value.get("full_public_population_proven") is not True
    ):
        raise ExactV9SemanticRootError("G46 encoder receipt does not prove full-n600 encoder-only custody")
    closure = value.get("upstream_closure")
    if (
        type(closure) is not dict
        or closure.get("closure_sha256") != G46_PATH_BOUND_RECEIPT_CLOSURE_SHA256
        or type(closure.get("members")) is not list
    ):
        raise ExactV9SemanticRootError("G46 receipt path-bound self-custody is missing")
    observed = tuple(
        (
            member.get("relative_path"),
            member.get("bytes"),
            member.get("sha256"),
        )
        for member in closure["members"]
        if type(member) is dict
    )
    if observed != G46_PORTABLE_SOURCE_MEMBERS:
        raise ExactV9SemanticRootError("G46 portable relative member identities disagree")
    target = value.get("target_labels")
    if (
        type(target) is not dict
        or target.get("bytes") != G46_TARGET_LABELS_BYTES
        or target.get("sha256") != G46_TARGET_LABELS_SHA256
        or target.get("shape") != [PAIR_COUNT_N600, SCORER_H, SCORER_W]
        or target.get("dtype") != "uint8"
        or target.get("chronological_pair_order") != list(range(PAIR_COUNT_N600))
    ):
        raise ExactV9SemanticRootError("G46 target-label identity/geometry differs")
    target_path_value = target.get("path")
    if type(target_path_value) is not str:
        raise ExactV9SemanticRootError("G46 target-label path is absent")
    target_path = _regular_file(Path(target_path_value), name="G46 target-label bank")
    target_bytes = target_path.read_bytes()
    if len(target_bytes) != G46_TARGET_LABELS_BYTES or _sha256(target_bytes) != G46_TARGET_LABELS_SHA256:
        raise ExactV9SemanticRootError("G46 target-label bank bytes differ")
    pair_rows = value.get("pair_checkpoints")
    if type(pair_rows) is not list or len(pair_rows) != PAIR_COUNT_N600:
        raise ExactV9SemanticRootError("G46 pair source custody is incomplete")
    pair_chain = hashlib.sha256()
    for pair_id, row in enumerate(pair_rows):
        if type(row) is not dict or row.get("pair_index") != pair_id:
            raise ExactV9SemanticRootError("G46 pair source custody is reordered or malformed")
        pair_chain.update(
            bytes.fromhex(_require_sha256(row.get("source_pair_rgb_sha256"), name="source pair"))
        )
    if pair_chain.hexdigest() != G46_SOURCE_PAIR_CHAIN_SHA256:
        raise ExactV9SemanticRootError("G46 source-pair digest chain differs")
    receipt_bytes = receipt_path.read_bytes()
    return {
        "receipt_path": os.path.abspath(os.fspath(receipt_path)),
        "receipt_bytes": len(receipt_bytes),
        "receipt_sha256": _sha256(receipt_bytes),
        "portable_closure_schema": UPSTREAM_SOURCE_CLOSURE_SCHEMA,
        "portable_closure_sha256": UPSTREAM_SOURCE_CLOSURE_SHA256,
        "portable_members": [
            {"relative_path": relative_path, "bytes": size, "sha256": digest}
            for relative_path, size, digest in G46_PORTABLE_SOURCE_MEMBERS
        ],
        "path_bound_receipt_closure_sha256": G46_PATH_BOUND_RECEIPT_CLOSURE_SHA256,
        "target_labels": {
            "path": os.path.abspath(os.fspath(target_path)),
            "bytes": len(target_bytes),
            "sha256": G46_TARGET_LABELS_SHA256,
            "shape": [PAIR_COUNT_N600, SCORER_H, SCORER_W],
            "dtype": "uint8",
        },
        "source_pair_chain_sha256": G46_SOURCE_PAIR_CHAIN_SHA256,
    }


def _checkpoint_runtime_state(
    checkpoint: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object], str, int]:
    checkpoint_path = _regular_file(checkpoint, name="fresh V9 checkpoint")
    checkpoint_bytes = checkpoint_path.read_bytes()
    try:
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise ExactV9SemanticRootError("fresh V9 checkpoint is not a strict NPZ") from exc
    scalars: dict[str, object] = {}
    for key, value in arrays.items():
        if not key.startswith("__"):
            continue
        scalars[key] = _scalar(value, name=key) if value.size == 1 else np.asarray(value).copy()
    params = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in arrays.items()
        if not key.startswith("__")
    }
    return params, scalars, _sha256(checkpoint_bytes), len(checkpoint_bytes)


def _require_exact_polar_basis(value: object) -> str:
    try:
        family = normalize_basis_family(value)
    except ValueError as exc:
        raise ExactV9SemanticRootError(
            "checkpoint basis is not the implemented exact polar Fourier ABI"
        ) from exc
    if family != LEGACY_FOURIER_AB_CONTROL:
        raise ExactV9SemanticRootError(
            "checkpoint basis is not the implemented exact polar Fourier ABI"
        )
    return family


def _checkpoint_config(
    params: dict[str, np.ndarray],
    scalars: dict[str, object],
) -> V9RuntimeConfigV1:
    if int(scalars.get("__cfg_fresh_init", 0)) != 1:
        raise ExactV9SemanticRootError("checkpoint is not marked as an applied fresh initialization")
    if scalars.get("__cfg_activation") != "hosc":
        raise ExactV9SemanticRootError("exact adapter admits only repository HOSC checkpoints")
    if scalars.get("__cfg_upstream_snapshot_schema") != UPSTREAM_SOURCE_CLOSURE_SCHEMA:
        raise ExactV9SemanticRootError("checkpoint portable upstream closure schema is missing or wrong")
    if scalars.get("__cfg_upstream_snapshot_sha256") != UPSTREAM_SOURCE_CLOSURE_SHA256:
        raise ExactV9SemanticRootError("checkpoint portable upstream closure SHA-256 is missing or wrong")
    _require_git_sha(scalars.get("__cfg_git_sha"))
    if int(scalars.get("__cfg_verdict_batch", 0)) != 16:
        raise ExactV9SemanticRootError(
            "checkpoint live verdict/controller geometry is not upstream-authority batch 16"
        )
    render_hw = np.asarray(scalars.get("__render_hw", ()), dtype=np.int64)
    if render_hw.shape != (2,) or tuple(int(value) for value in render_hw) != (SCORER_H, SCORER_W):
        raise ExactV9SemanticRootError("checkpoint render grid is not exact scorer-grid 384x512")
    if int(scalars.get("__cfg_self_orient", 0)) != 0:
        raise ExactV9SemanticRootError(
            "self-orient checkpoints require a separately reviewed decoder-owned fixed-point basis ABI"
        )
    _require_exact_polar_basis(scalars.get("__cfg_basis", LEGACY_FOURIER_AB_CONTROL))
    basis = V9PolarFourierConfigV1(
        n_scales=int(scalars.get("__bank_n_scales", 4)),
        n_orient0=int(scalars.get("__bank_n_orient0", 6)),
        f0=float(scalars.get("__bank_f0", 2.0)),
        base=float(scalars.get("__bank_base", 2.0)),
        n_iso=int(scalars.get("__bank_n_iso", 4)),
        max_freq=(
            None
            if float(scalars.get("__cfg_max_bank_freq", -1.0)) < 0.0
            else float(scalars["__cfg_max_bank_freq"])
        ),
    )
    required = {
        "code",
        "in_proj.weight",
        "in_proj.bias",
        "film.weight",
        "film.bias",
        "out_sdf.weight",
        "out_sdf.bias",
        "out_tex.weight",
        "out_tex.bias",
        "palette",
    }
    if not required.issubset(params):
        raise ExactV9SemanticRootError("checkpoint is missing one or more exact V9 dual-head tensors")
    hidden_dim = int(params["in_proj.weight"].shape[0])
    input_dim = int(params["in_proj.weight"].shape[1])
    modulation_dim = int(params["code"].shape[1]) if params["code"].ndim == 2 else -1
    hidden_indices = sorted(
        int(key.split(".")[1])
        for key in params
        if key.startswith("hidden.") and key.endswith(".weight")
    )
    if hidden_indices != list(range(len(hidden_indices))):
        raise ExactV9SemanticRootError("checkpoint hidden layers are not canonical contiguous indices")
    film_per_layer = any(key.startswith("film_pl.") for key in params)
    film_concat_code = any(key.startswith("concat_pl.") for key in params)
    config = V9RuntimeConfigV1(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        hidden_layer_count=len(hidden_indices),
        modulation_dim=modulation_dim,
        softmax_temp=float(scalars.get("__cfg_softmax_temp", 0.0)),
        hosc_beta=float(scalars.get("__cfg_hosc_beta", 0.0)),
        hosc_omega=float(scalars.get("__cfg_hosc_omega", 0.0)),
        chroma=bool(int(scalars.get("__cfg_chroma", 0))),
        film_per_layer=film_per_layer,
        film_concat_code=film_concat_code,
        basis=basis,
    )
    expected_expanded = {
        "code",
        "in_proj.weight",
        "in_proj.bias",
        "film.weight",
        "film.bias",
        "out_sdf.weight",
        "out_sdf.bias",
        "out_tex.weight",
        "out_tex.bias",
        "palette",
        *(
            f"hidden.{index}.{suffix}"
            for index in range(config.hidden_layer_count)
            for suffix in ("weight", "bias")
        ),
    }
    if config.film_per_layer:
        expected_expanded.update(
            f"film_pl.{index}.{suffix}"
            for index in range(config.hidden_layer_count)
            for suffix in ("weight", "bias")
        )
    if config.film_concat_code:
        expected_expanded.update(
            f"concat_pl.{index}.{suffix}"
            for index in range(config.hidden_layer_count)
            for suffix in ("weight", "bias")
        )
    if set(params) != expected_expanded:
        unexpected = sorted(set(params) - expected_expanded)
        missing = sorted(expected_expanded - set(params))
        raise ExactV9SemanticRootError(
            f"checkpoint learned tensor set is not exactly consumed; unexpected={unexpected}, missing={missing}"
        )
    return config


def _validate_checkpoint_target_binding(
    scalars: dict[str, object],
    evidence: V9G46Batch16TrainingTargetEvidenceV1,
) -> None:
    exact = {
        "__cfg_target_authority_sha256": evidence.active_target_authority_sha256,
        "__cfg_g46_target_labels_sha256": evidence.target_labels_sha256,
        "__cfg_g46_target_margins_sha256": evidence.target_margins_sha256,
        "__cfg_g46_source_pair_chain_sha256": evidence.source_pair_chain_sha256,
        "__cfg_g46_margin_aggregate_schema": evidence.margin_aggregate_schema,
        "__cfg_g46_margin_aggregate_sha256": evidence.margin_aggregate_receipt_sha256,
        "__cfg_g46_target_consumer_binding_sha256": evidence.consumer_binding_sha256,
        "__cfg_g46_target_evidence_sha256": evidence.evidence_sha256,
    }
    for key, expected in exact.items():
        if scalars.get(key) != expected:
            raise ExactV9SemanticRootError(f"checkpoint target binding differs at {key}")
    integer_exact = {
        "__cfg_g46_target_scorer_batch_size": evidence.scorer_pair_batch_size,
        "__cfg_g46_margin_same_forward": 1,
        "__cfg_verdict_batch": evidence.live_verdict_batch_size,
    }
    for key, expected in integer_exact.items():
        try:
            observed = int(scalars.get(key, -1))
        except (TypeError, ValueError) as exc:
            raise ExactV9SemanticRootError(f"checkpoint target binding is not integral at {key}") from exc
        if observed != expected:
            raise ExactV9SemanticRootError(f"checkpoint target binding differs at {key}")


def compile_fresh_checkpoint(
    *,
    checkpoint: Path,
    g46_encoder_receipt: Path,
    phase_advection: V9PhaseAdvectionTrainingEvidenceV1,
    training_target: V9G46Batch16TrainingTargetEvidenceV1,
) -> CompiledFreshV9SemanticRootV1:
    """Compile a fresh G46-bound checkpoint and keep all proof outside bytes."""

    if type(phase_advection) is not V9PhaseAdvectionTrainingEvidenceV1:
        raise ExactV9SemanticRootError("phase_advection must be typed external evidence")
    if type(training_target) is not V9G46Batch16TrainingTargetEvidenceV1:
        raise ExactV9SemanticRootError("training_target must be typed G46 batch-16 external evidence")
    g46 = _portable_g46_receipt_identity(g46_encoder_receipt)
    params, scalars, checkpoint_sha, checkpoint_bytes = _checkpoint_runtime_state(checkpoint)
    config = _checkpoint_config(params, scalars)
    _validate_checkpoint_target_binding(scalars, training_target)
    program = compile_from_state(
        config=config,
        params={key: value for key, value in params.items() if key != "code"},
        interleaved_code=params["code"],
    )
    packet = encode_packet(program)
    parsed = parse_packet(packet)
    if encode_packet(parsed) != packet:
        raise AssertionError("internal strict packet identity failed")
    accounting = candidate_wire_accounting(packet)
    external_receipt = {
        "schema": "tac.g105_fresh_v9_semantic_root_compile_receipt.v1",
        "variant_id": VARIANT_ID,
        "research_only": True,
        "candidate": False,
        "score_claim": False,
        "pointer_mutation_allowed": False,
        "checkpoint": {
            "path": os.path.abspath(os.fspath(checkpoint)),
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_sha,
            "git_sha": scalars["__cfg_git_sha"],
            "fresh_init": True,
            "portable_upstream_closure_schema": scalars["__cfg_upstream_snapshot_schema"],
            "portable_upstream_closure_sha256": scalars["__cfg_upstream_snapshot_sha256"],
        },
        "g46_encoder_evidence": g46,
        "phase_advection_training_evidence": {
            "identity_sha256": phase_advection.identity_sha256(),
            "source_evidence_sha256": phase_advection.evidence_sha256,
            "serialized_in_candidate": False,
        },
        "g46_batch16_training_target_evidence": {
            "identity_sha256": training_target.identity_sha256(),
            "active_target_authority_sha256": training_target.active_target_authority_sha256,
            "target_labels_sha256": training_target.target_labels_sha256,
            "target_margins_sha256": training_target.target_margins_sha256,
            "source_pair_chain_sha256": training_target.source_pair_chain_sha256,
            "scorer_pair_batch_size": training_target.scorer_pair_batch_size,
            "live_verdict_batch_size": training_target.live_verdict_batch_size,
            "margins_from_same_batch16_forward": True,
            "serialized_in_candidate": False,
        },
        "runtime": {
            "activation": "tanh(beta*sin(omega*x))",
            "dual_heads": ["out_sdf", "out_tex", "palette"],
            "film_per_layer": config.film_per_layer,
            "film_concat_code": config.film_concat_code,
            "basis": config.basis.to_dict(),
            "counted_y1_projection": "code[2*p+1]",
            "counted_y1_rows": PAIR_COUNT_N600,
            "serialized_y0_rows": 0,
            "exclusive_y0_owner": "G94-V2",
        },
        "wire": accounting.to_dict(),
    }
    return CompiledFreshV9SemanticRootV1(
        program=program,
        packet=packet,
        wire_accounting=accounting,
        external_receipt=external_receipt,
    )


__all__ = [
    "G46_PATH_BOUND_RECEIPT_CLOSURE_SHA256",
    "G46_PORTABLE_SOURCE_MEMBERS",
    "G46_SOURCE_PAIR_CHAIN_SHA256",
    "G46_TARGET_LABELS_SHA256",
    "MAGIC",
    "PAIR_COUNT_N600",
    "SCORER_H",
    "SCORER_W",
    "UPSTREAM_SOURCE_CLOSURE_SCHEMA",
    "UPSTREAM_SOURCE_CLOSURE_SHA256",
    "VARIANT_ID",
    "CompiledFreshV9SemanticRootV1",
    "ExactV9SemanticRootError",
    "ExactV9SemanticRootY1ProgramV1",
    "V9CandidateWireAccountingV1",
    "V9CountedTensorV1",
    "V9G46Batch16TrainingTargetEvidenceV1",
    "V9PhaseAdvectionTrainingEvidenceV1",
    "V9PolarFourierConfigV1",
    "V9RuntimeConfigV1",
    "V9TensorDTypeV1",
    "build_runtime_features",
    "candidate_wire_accounting",
    "compile_fresh_checkpoint",
    "compile_from_state",
    "encode_packet",
    "forward_float32",
    "parse_packet",
    "render_scorer_y1",
]
