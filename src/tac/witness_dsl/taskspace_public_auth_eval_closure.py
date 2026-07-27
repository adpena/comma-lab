# SPDX-License-Identifier: MIT
"""Typed public-decoder and authoritative-evaluation closure for task-space packets.

The module keeps three facts separate:

* the offline encoder may spend unbounded deterministic compute;
* the public VM may run arbitrary *generic* deterministic algorithms inside a
  bounded CPU/GPU envelope;
* every video-derived weight, latent, selector, threshold, exception, or other
  state is charged to the exact archive bytes.

It compiles the generic LVPG2 inverse into an ordinary public ``inflate.sh`` /
``inflate.py`` package, records recursive runtime and ABI identity, and exposes
strict receipts that a reviewed owner of the sealed ``C0BAuthEvalClosureV1``
can consume.  This module deliberately cannot mint contest authority itself.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Final, Self

from tac.contest_score import compute_contest_score
from tac.witness_dsl.ep725_lossless_xcodec_recode import parse_ep725_lvls1
from tac.witness_dsl.ep725_population_global_recode_v2 import (
    MEMBER_NAME,
    parse_population_global_member,
)
from tac.witness_dsl.taskspace_selected_solution_compiler import (
    G17ReopenedEvidencePacketV1,
    G17RuntimeDependencyEdgeV1,
    G17RuntimeDependencyFileV1,
    G17RuntimeDependencyMechanismV1,
    G17RuntimeFileScopeV1,
    G17TerminalCompilerScheduleV1,
)

RUNTIME_COMPILE_SCHEMA: Final = "tac.taskspace_public_runtime_compile_receipt.v1"
DEPENDENCY_DISCOVERY_SCHEMA: Final = "tac.taskspace_runtime_dependency_discovery_receipt.v1"
PUBLIC_EXECUTION_SCHEMA: Final = "tac.taskspace_public_evaluator_execution_receipt.v1"
PUBLIC_DECODE_EQUALITY_SCHEMA: Final = "tac.taskspace_public_decode_equality_receipt.v1"
STAGE_CHECKPOINT_SCHEMA: Final = "tac.taskspace_auth_closure_stage_checkpoint.v1"
READINESS_SCHEMA: Final = "tac.taskspace_auth_closure_execution_readiness.v1"
TRACE_CLOSURE_SCHEMA: Final = "tac.taskspace_public_trace_closure.v1"
SCORER_INPUT_BATCH_LEDGER_SCHEMA: Final = "tac.taskspace_scorer_input_batch_ledger.v1"
SCORER_OUTPUT_CELL_LEDGER_SCHEMA: Final = "tac.taskspace_scorer_output_cell_ledger.v1"
SCORER_OUTPUT_MIRROR_EQUIVALENCE_SCHEMA: Final = "tac.taskspace_scorer_output_mirror_equivalence.v1"
OFFICIAL_WORKFLOW_JOB_SCHEMA: Final = "tac.taskspace_official_workflow_job_receipt.v1"
PUBLIC_INVERSE_TRACE_SCHEMA: Final = "tac.taskspace_lvpg2_public_inverse_trace.v1"
ADAPTER_INGREDIENTS_SCHEMA: Final = "tac.taskspace_auth_eval_adapter_ingredients.v1"
ABI_CLOSURE_SCHEMA: Final = "tac.taskspace_interpreter_distribution_abi_closure.v1"
PLACEMENT_SCHEMA: Final = "tac.taskspace_asymmetric_compiler_vm_placement.v1"
GENERIC_SOURCE_AUDIT_SCHEMA: Final = "tac.taskspace_generic_runtime_source_audit.v1"

PUBLIC_EVALUATOR_PATH: Final = "upstream/evaluate.py"
PUBLIC_INFLATE_SH_PATH: Final = "inflate.sh"
PUBLIC_INFLATE_PY_PATH: Final = "inflate.py"
PUBLIC_LVLS1_RUNTIME_PATH: Final = "lvls1_runtime.py"
EXPECTED_N_PAIRS: Final = 600
EXPECTED_N_FRAMES: Final = 1200
OFFICIAL_EVALUATOR_BATCH_SIZE: Final = 16
EXPECTED_EVALUATOR_BATCH_COUNT: Final = (
    EXPECTED_N_PAIRS + OFFICIAL_EVALUATOR_BATCH_SIZE - 1
) // OFFICIAL_EVALUATOR_BATCH_SIZE
CAMERA_WIDTH: Final = 1164
CAMERA_HEIGHT: Final = 874
SCORER_INPUT_WIDTH: Final = 512
SCORER_INPUT_HEIGHT: Final = 384
EXPECTED_RAW_NBYTES: Final = CAMERA_WIDTH * CAMERA_HEIGHT * EXPECTED_N_FRAMES * 3
EXPECTED_UPSTREAM_SNAPSHOT_SHA256: Final = "d46d89155dbf0848e357858c8f62e12ef450a2914ef65814a4359ef6768d2d41"
EXPECTED_OFFICIAL_WORKFLOW_SHA256: Final = "8a6cd6300b51a44f36b49774bc0c6100dbb37ef8290d42bf8e584f1dceddce56"
OFFICIAL_TOTAL_WORKFLOW_SECONDS: Final = 1800.0
CONTEST_MEMORY_NBYTES: Final = 16 * 1024**3
CONTEST_CUDA_HOST_MEMORY_NBYTES: Final = 26 * 1024**3
CONTEST_CUDA_DEVICE_MEMORY_NBYTES: Final = 16 * 1024**3
SCORE_RATE_DENOMINATOR: Final = 37_545_489

_HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_.:-]{2,127}\Z")
_DIST_IMPORT_ALIASES: Final = {
    "PIL": "Pillow",
    "av": "av",
    "brotli": "Brotli",
    "einops": "einops",
    "numpy": "numpy",
    "nvidia": "nvidia-dali-cuda120",
    "safetensors": "safetensors",
    "scipy": "scipy",
    "segmentation_models_pytorch": "segmentation-models-pytorch",
    "timm": "timm",
    "torch": "torch",
    "torchvision": "torchvision",
    "tqdm": "tqdm",
}
_STDLIB_ROOTS: Final = frozenset(getattr(sys, "stdlib_module_names", ()))
_EVALUATOR_CPU_ABI_ROOTS: Final = frozenset(
    {
        "PIL",
        "av",
        "brotli",
        "einops",
        "numpy",
        "safetensors",
        "scipy",
        "segmentation_models_pytorch",
        "timm",
        "torch",
        "torchvision",
        "tqdm",
    }
)
_EVALUATOR_CUDA_ABI_ROOTS: Final = _EVALUATOR_CPU_ABI_ROOTS | {"nvidia"}
_FORBIDDEN_RUNTIME_IMPORT_ROOTS: Final = frozenset(
    {
        "requests",
        "secrets",
        "socket",
        "tac",
        "urllib3",
        "upstream",
    }
)
_FORBIDDEN_RUNTIME_NAME_FRAGMENTS: Final = (
    "ground_truth",
    "gt_n600",
    "oracle_state",
    "pose6_target",
    "posenet",
    "scorer_weight",
    "segnet",
    "target_frame",
    "teacher_state",
)
_FORBIDDEN_CALLS: Final = frozenset(
    {
        "compile",
        "ctypes.CDLL",
        "ctypes.PyDLL",
        "datetime.datetime.now",
        "datetime.datetime.utcnow",
        "eval",
        "exec",
        "importlib.import_module",
        "numpy.ctypeslib.load_library",
        "os.system",
        "os.urandom",
        "numpy.load",
        "pickle.load",
        "pickle.loads",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
        "subprocess.run",
        "time.time",
        "time.time_ns",
        "torch.hub.load",
        "torch.jit.load",
        "torch.load",
    }
)
_FORBIDDEN_CALL_LEAF_PREFIXES: Final = ("exec", "spawn")
_PUBLIC_RUNTIME_ENVIRONMENT_KEYS: Final = frozenset(
    {
        "INFLATE_FP32",
        "INFLATE_MAX_PAIRS",
        "INFLATE_WORKERS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PACT_AUTH_TRACE_DIR",
        "PACT_AUTH_TRACE_PATH",
        "PYTHONDONTWRITEBYTECODE",
        "VECLIB_MAXIMUM_THREADS",
    }
)
_MAX_STATIC_LITERAL_NBYTES: Final = 2048
_OFFICIAL_RUN_CONSTRUCTION_SEAL: Final = object()
_PUBLIC_EQUALITY_DERIVATION_SEAL: Final = object()
_RESOURCE_OBSERVATION_DERIVATION_SEAL: Final = object()
_REPRESENTATION_EVIDENCE_DERIVATION_SEAL: Final = object()
_STRICT_RECEIPT_REOPEN_SEAL: Final = object()


class PublicAuthClosureError(ValueError):
    """A placement, wire, runtime, ABI, receipt, or authority invariant failed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PublicAuthClosureError("value is not finite canonical ASCII JSON") from exc


def _parse_canonical_object(payload: bytes, *, exact_keys: frozenset[str]) -> dict[str, Any]:
    if type(payload) is not bytes or not payload:
        raise PublicAuthClosureError("receipt must be nonempty immutable bytes")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicAuthClosureError("receipt is not canonical ASCII JSON") from exc
    if type(value) is not dict or frozenset(value) != exact_keys or _canonical_json(value) != payload:
        raise PublicAuthClosureError("receipt fields or canonical parse/re-emit identity drifted")
    return value


def _require_ascii(value: object, *, label: str) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise PublicAuthClosureError(f"{label} must be nonempty ASCII")
    return value


def _require_identifier(value: object, *, label: str) -> str:
    result = _require_ascii(value, label=label)
    if _IDENTIFIER_RE.fullmatch(result) is None:
        raise PublicAuthClosureError(f"{label} is not a normalized identifier")
    return result


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _HEX64_RE.fullmatch(value) is None:
        raise PublicAuthClosureError(f"{label} must be a lowercase SHA-256")
    return value


def _require_finite_nonnegative(value: object, *, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < 0.0:
        raise PublicAuthClosureError(f"{label} must be finite and nonnegative")
    return float(value)


def _reported_score_interval_consistent(
    *,
    displayed_two_decimal: str,
    avg_segnet_dist_8dec: float,
    avg_posenet_dist_8dec: float,
    archive_nbytes: int,
    original_uncompressed_nbytes: int,
) -> bool:
    """Check that hidden full-precision score and displayed rounding can intersect."""

    if re.fullmatch(r"[0-9]+\.[0-9]{2}", displayed_two_decimal) is None:
        return False
    possible_low, possible_high = _reported_component_score_interval(
        avg_segnet_dist_8dec=avg_segnet_dist_8dec,
        avg_posenet_dist_8dec=avg_posenet_dist_8dec,
        archive_nbytes=archive_nbytes,
        original_uncompressed_nbytes=original_uncompressed_nbytes,
    )
    displayed = float(displayed_two_decimal)
    displayed_low = displayed - 0.0050000001
    displayed_high = displayed + 0.0050000001
    return possible_high >= displayed_low and possible_low <= displayed_high


def _reported_component_score_interval(
    *,
    avg_segnet_dist_8dec: float,
    avg_posenet_dist_8dec: float,
    archive_nbytes: int,
    original_uncompressed_nbytes: int,
) -> tuple[float, float]:
    """Return the score interval compatible with the evaluator's 8-decimal components."""

    half_component_unit = 0.5e-8
    seg_low = max(0.0, avg_segnet_dist_8dec - half_component_unit)
    seg_high = avg_segnet_dist_8dec + half_component_unit
    pose_low = max(0.0, avg_posenet_dist_8dec - half_component_unit)
    pose_high = avg_posenet_dist_8dec + half_component_unit
    possible_low = compute_contest_score(
        seg_low,
        pose_low,
        archive_nbytes,
        uncompressed_size=original_uncompressed_nbytes,
    )
    possible_high = compute_contest_score(
        seg_high,
        pose_high,
        archive_nbytes,
        uncompressed_size=original_uncompressed_nbytes,
    )
    return possible_low, possible_high


def _relative_path(value: object, *, label: str) -> str:
    result = _require_ascii(value, label=label)
    path = PurePosixPath(result)
    if path.is_absolute() or path.as_posix() != result or any(part in {"", ".", ".."} for part in path.parts):
        raise PublicAuthClosureError(f"{label} must be a normalized relative POSIX path")
    return result


def _atomic_write(path: Path, payload: bytes, *, executable: bool = False) -> None:
    """Create one durable immutable artifact without a replace race."""

    if path.exists():
        raise PublicAuthClosureError(f"refusing to overwrite retained artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise PublicAuthClosureError(f"stale partial artifact blocks atomic write: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o755 if executable else 0o644)
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        temporary.unlink()
        raise PublicAuthClosureError(f"refusing to overwrite retained artifact: {path}") from exc
    temporary.unlink()
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _sealed_frozen_dataclass(
    cls: type[Any],
    *,
    seal: object,
    allowed_seals: tuple[object, ...],
    values: dict[str, object],
) -> Any:
    if seal not in allowed_seals:
        raise PublicAuthClosureError(f"{cls.__name__} has no public constructor")
    descriptors = tuple(dataclass_fields(cls))
    expected = {item.name for item in descriptors if item.init}
    if set(values) != expected:
        raise PublicAuthClosureError(
            f"{cls.__name__} sealed fields drifted: expected {sorted(expected)}, got {sorted(values)}"
        )
    instance = object.__new__(cls)
    for descriptor in descriptors:
        if descriptor.init:
            object.__setattr__(instance, descriptor.name, values[descriptor.name])
        else:
            object.__setattr__(instance, descriptor.name, descriptor.default)
    instance.__post_init__()
    return instance


def _bytecode_contamination(root: Path) -> tuple[str, ...]:
    if not root.exists():
        return ()
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and (path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts)
        )
    )


def require_no_bytecode_contamination(*roots: Path) -> None:
    """Refuse stale or regenerated executable Python artifacts under owned roots."""

    contaminated = {root.as_posix(): _bytecode_contamination(root) for root in roots if _bytecode_contamination(root)}
    if contaminated:
        raise PublicAuthClosureError(f"generated Python bytecode contamination refused: {contaminated}")


def require_exact_public_runtime_tree(runtime_dir: Path, *, complete: bool) -> None:
    """Reject undeclared files, directories, symlinks, and partials beside public code."""

    expected = {PUBLIC_INFLATE_SH_PATH, PUBLIC_INFLATE_PY_PATH, PUBLIC_LVLS1_RUNTIME_PATH}
    if not runtime_dir.exists():
        if complete:
            raise PublicAuthClosureError("public runtime directory is missing")
        return
    if not runtime_dir.is_dir() or runtime_dir.is_symlink():
        raise PublicAuthClosureError("public runtime root must be a real directory")
    entries = tuple(runtime_dir.iterdir())
    names = {path.name for path in entries}
    if not names.issubset(expected) or (complete and names != expected):
        raise PublicAuthClosureError(
            f"public runtime directory contains undeclared or missing entries: {sorted(names)}"
        )
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise PublicAuthClosureError("public runtime entries must be exact non-symlink regular files")


class PlacementLocationV1(StrEnum):
    UNBOUNDED_OFFLINE_COMPILER = "UNBOUNDED_OFFLINE_COMPILER"
    BOUNDED_PUBLIC_VM = "BOUNDED_PUBLIC_VM"
    COUNTED_ARCHIVE_PAYLOAD = "COUNTED_ARCHIVE_PAYLOAD"


class ContentOriginV1(StrEnum):
    GENERIC_ALGORITHM = "GENERIC_ALGORITHM"
    VIDEO_DERIVED_STATE = "VIDEO_DERIVED_STATE"


class VideoDerivedPayloadClassV1(StrEnum):
    WEIGHT = "WEIGHT"
    LATENT = "LATENT"
    SELECTOR = "SELECTOR"
    THRESHOLD = "THRESHOLD"
    EXCEPTION = "EXCEPTION"
    CODEBOOK = "CODEBOOK"
    TRAJECTORY = "TRAJECTORY"
    MIXED_PAYLOAD_AND_CONTAINER = "MIXED_PAYLOAD_AND_CONTAINER"


class GenericVMFacilityV1(StrEnum):
    CODEC_INVERSE = "CODEC_INVERSE"
    DETERMINISTIC_GENERATIVE_REPAIR = "DETERMINISTIC_GENERATIVE_REPAIR"
    GENERIC_NETWORK_ARCHITECTURE = "GENERIC_NETWORK_ARCHITECTURE"
    OPTIMIZER = "OPTIMIZER"
    POSTFILTER = "POSTFILTER"
    RASTERIZER = "RASTERIZER"
    BOUNDED_DECODE_TIME_FITTING = "BOUNDED_DECODE_TIME_FITTING"


class ExecutionAxisV1(StrEnum):
    CPU = "CPU"
    CUDA = "CUDA"


class RepresentationClassV1(StrEnum):
    ANALYTIC_INVERSE = "ANALYTIC_INVERSE"
    BOUNDED_ENUMERATION = "BOUNDED_ENUMERATION"
    DETERMINISTIC_GENERATIVE_REPAIR = "DETERMINISTIC_GENERATIVE_REPAIR"
    IRREDUCIBLE_LEARNED_RESIDUE = "IRREDUCIBLE_LEARNED_RESIDUE"


class MeasurementAuthorityV1(StrEnum):
    CONTEST_CPU = "CONTEST_CPU"
    CONTEST_CUDA = "CONTEST_CUDA"


class AuthClosureStageV1(StrEnum):
    COMPILE_PUBLIC_RUNTIME = "COMPILE_PUBLIC_RUNTIME"
    DEPENDENCY_DISCOVERY = "DEPENDENCY_DISCOVERY"
    EXECUTION_PREFLIGHT = "EXECUTION_PREFLIGHT"
    PUBLIC_AUTH_EVAL_A = "PUBLIC_AUTH_EVAL_A"
    PUBLIC_AUTH_EVAL_B = "PUBLIC_AUTH_EVAL_B"
    PUBLIC_DECODE_EQUALITY = "PUBLIC_DECODE_EQUALITY"
    FINALIZE_AUTH_CLOSURE = "FINALIZE_AUTH_CLOSURE"


class ExternalReadKindV1(StrEnum):
    RUN_SCOPED_REGULAR_FILE = "RUN_SCOPED_REGULAR_FILE"
    SYSTEM_REGULAR_FILE = "SYSTEM_REGULAR_FILE"
    VIRTUAL_KERNEL_FILE = "VIRTUAL_KERNEL_FILE"


@dataclass(frozen=True, slots=True)
class PayloadPlacementItemV1:
    item_id: str
    origin: ContentOriginV1
    location: PlacementLocationV1
    content_sha256: str
    object_nbytes: int
    charged_archive_nbytes: int
    video_payload_class: VideoDerivedPayloadClassV1 | None = None
    vm_facility: GenericVMFacilityV1 | None = None
    archive_member_path: str | None = None
    source_population_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.item_id, label="placement item id")
        if type(self.origin) is not ContentOriginV1 or type(self.location) is not PlacementLocationV1:
            raise PublicAuthClosureError("placement origin/location must be typed")
        _require_sha256(self.content_sha256, label="placement content")
        if type(self.object_nbytes) is not int or self.object_nbytes < 1:
            raise PublicAuthClosureError("placement object_nbytes must be positive int")
        if type(self.charged_archive_nbytes) is not int or self.charged_archive_nbytes < 0:
            raise PublicAuthClosureError("placement charged bytes must be nonnegative int")
        if self.origin is ContentOriginV1.VIDEO_DERIVED_STATE:
            if self.location is not PlacementLocationV1.COUNTED_ARCHIVE_PAYLOAD:
                raise PublicAuthClosureError(
                    "video-derived weights/latents/selectors/thresholds/exceptions must be counted archive payload"
                )
            if type(self.video_payload_class) is not VideoDerivedPayloadClassV1:
                raise PublicAuthClosureError("video-derived placement requires typed payload class")
            if self.vm_facility is not None:
                raise PublicAuthClosureError("video-derived state cannot masquerade as a generic VM facility")
            if self.charged_archive_nbytes != self.object_nbytes:
                raise PublicAuthClosureError("counted video-derived object must charge every containing byte")
            if self.archive_member_path is None:
                raise PublicAuthClosureError("counted video-derived object requires archive member path")
            _relative_path(self.archive_member_path, label="archive member path")
        else:
            if self.video_payload_class is not None or self.archive_member_path is not None:
                raise PublicAuthClosureError("generic algorithm cannot claim a video-derived payload/member")
            if self.charged_archive_nbytes != 0:
                raise PublicAuthClosureError("generic algorithm code is free and may not invent charged payload bytes")
            if self.location is PlacementLocationV1.COUNTED_ARCHIVE_PAYLOAD:
                raise PublicAuthClosureError("generic algorithm belongs in compiler/public VM, not counted payload")
            if self.location is PlacementLocationV1.BOUNDED_PUBLIC_VM:
                if type(self.vm_facility) is not GenericVMFacilityV1:
                    raise PublicAuthClosureError("public generic code requires an allowed typed VM facility")
            elif self.vm_facility is not None:
                raise PublicAuthClosureError("offline compiler item may not claim public VM facility")
        if self.source_population_sha256 is not None:
            _require_sha256(self.source_population_sha256, label="source population")

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_member_path": self.archive_member_path,
            "charged_archive_nbytes": self.charged_archive_nbytes,
            "content_sha256": self.content_sha256,
            "item_id": self.item_id,
            "location": self.location.value,
            "object_nbytes": self.object_nbytes,
            "origin": self.origin.value,
            "source_population_sha256": self.source_population_sha256,
            "video_payload_class": (None if self.video_payload_class is None else self.video_payload_class.value),
            "vm_facility": None if self.vm_facility is None else self.vm_facility.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        expected = {
            "archive_member_path",
            "charged_archive_nbytes",
            "content_sha256",
            "item_id",
            "location",
            "object_nbytes",
            "origin",
            "source_population_sha256",
            "video_payload_class",
            "vm_facility",
        }
        if type(value) is not dict or set(value) != expected:
            raise PublicAuthClosureError("placement item fields drifted")
        return cls(
            item_id=value["item_id"],
            origin=ContentOriginV1(value["origin"]),
            location=PlacementLocationV1(value["location"]),
            content_sha256=value["content_sha256"],
            object_nbytes=value["object_nbytes"],
            charged_archive_nbytes=value["charged_archive_nbytes"],
            video_payload_class=(
                None
                if value["video_payload_class"] is None
                else VideoDerivedPayloadClassV1(value["video_payload_class"])
            ),
            vm_facility=(None if value["vm_facility"] is None else GenericVMFacilityV1(value["vm_facility"])),
            archive_member_path=value["archive_member_path"],
            source_population_sha256=value["source_population_sha256"],
        )


@dataclass(frozen=True, slots=True)
class DecodeResourceEnvelopeV1:
    max_total_workflow_seconds: float
    max_peak_memory_nbytes: int
    legal_axes: tuple[ExecutionAxisV1, ...]
    deterministic_output_required: bool = True
    network_forbidden: bool = True

    def __post_init__(self) -> None:
        _require_finite_nonnegative(self.max_total_workflow_seconds, label="total workflow wall limit")
        if self.max_total_workflow_seconds <= 0.0:
            raise PublicAuthClosureError("total workflow wall limit must be positive")
        if type(self.max_peak_memory_nbytes) is not int or self.max_peak_memory_nbytes < 1:
            raise PublicAuthClosureError("decode memory limit must be positive int")
        if (
            type(self.legal_axes) is not tuple
            or not self.legal_axes
            or any(type(axis) is not ExecutionAxisV1 for axis in self.legal_axes)
            or self.legal_axes != tuple(sorted(set(self.legal_axes), key=lambda item: item.value))
        ):
            raise PublicAuthClosureError("decode legal axes must be unique canonical typed tuple")
        if self.deterministic_output_required is not True or self.network_forbidden is not True:
            raise PublicAuthClosureError("public VM must be deterministic and network-free")

    @classmethod
    def contest_default(cls) -> Self:
        return cls(
            max_total_workflow_seconds=OFFICIAL_TOTAL_WORKFLOW_SECONDS,
            max_peak_memory_nbytes=CONTEST_CUDA_HOST_MEMORY_NBYTES,
            legal_axes=(ExecutionAxisV1.CPU, ExecutionAxisV1.CUDA),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "deterministic_output_required": self.deterministic_output_required,
            "legal_axes": [axis.value for axis in self.legal_axes],
            "max_peak_memory_nbytes": self.max_peak_memory_nbytes,
            "max_total_workflow_seconds": self.max_total_workflow_seconds,
            "network_forbidden": self.network_forbidden,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            "deterministic_output_required",
            "legal_axes",
            "max_peak_memory_nbytes",
            "max_total_workflow_seconds",
            "network_forbidden",
        }:
            raise PublicAuthClosureError("decode resource envelope fields drifted")
        return cls(
            max_total_workflow_seconds=value["max_total_workflow_seconds"],
            max_peak_memory_nbytes=value["max_peak_memory_nbytes"],
            legal_axes=tuple(ExecutionAxisV1(item) for item in value["legal_axes"]),
            deterministic_output_required=value["deterministic_output_required"],
            network_forbidden=value["network_forbidden"],
        )


@dataclass(frozen=True, slots=True)
class PayloadPlacementManifestV1:
    archive_sha256: str
    archive_nbytes: int
    items: tuple[PayloadPlacementItemV1, ...]
    decode_envelope: DecodeResourceEnvelopeV1
    schema: str = field(default=PLACEMENT_SCHEMA, init=False)

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {"archive_nbytes", "archive_sha256", "decode_envelope", "items", "schema"}
    )

    def __post_init__(self) -> None:
        _require_sha256(self.archive_sha256, label="placement archive")
        if type(self.archive_nbytes) is not int or self.archive_nbytes < 1:
            raise PublicAuthClosureError("placement archive_nbytes must be positive int")
        if (
            type(self.items) is not tuple
            or not self.items
            or any(type(item) is not PayloadPlacementItemV1 for item in self.items)
            or self.items != tuple(sorted(self.items, key=lambda item: item.item_id))
            or len({item.item_id for item in self.items}) != len(self.items)
        ):
            raise PublicAuthClosureError("placement items must be unique canonical typed tuple")
        if type(self.decode_envelope) is not DecodeResourceEnvelopeV1:
            raise PublicAuthClosureError("placement decode envelope must be typed")
        charged = sum(item.charged_archive_nbytes for item in self.items)
        if charged != self.archive_nbytes:
            raise PublicAuthClosureError(
                f"placement charges {charged} byte(s), exact archive has {self.archive_nbytes}"
            )
        locations = {item.location for item in self.items}
        if PlacementLocationV1.UNBOUNDED_OFFLINE_COMPILER not in locations:
            raise PublicAuthClosureError("placement ABI must name the unbounded offline compiler")
        if PlacementLocationV1.BOUNDED_PUBLIC_VM not in locations:
            raise PublicAuthClosureError("placement ABI must name the bounded public VM")
        if PlacementLocationV1.COUNTED_ARCHIVE_PAYLOAD not in locations:
            raise PublicAuthClosureError("placement ABI must name the exact counted payload")

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(
            {
                "archive_nbytes": self.archive_nbytes,
                "archive_sha256": self.archive_sha256,
                "decode_envelope": self.decode_envelope.to_dict(),
                "items": [item.to_dict() for item in self.items],
                "schema": self.schema,
            }
        )

    @classmethod
    def from_receipt_bytes(cls, payload: bytes) -> Self:
        value = _parse_canonical_object(payload, exact_keys=cls._KEYS)
        if value["schema"] != PLACEMENT_SCHEMA:
            raise PublicAuthClosureError("placement manifest schema drifted")
        return cls(
            archive_sha256=value["archive_sha256"],
            archive_nbytes=value["archive_nbytes"],
            items=tuple(PayloadPlacementItemV1.from_dict(item) for item in value["items"]),
            decode_envelope=DecodeResourceEnvelopeV1.from_dict(value["decode_envelope"]),
        )


# Backward-readable name for the ABI concept; this is the exact same runtime
# type, not a wrapper or a second source of placement truth.
AsymmetricCompilerVMABIV1 = PayloadPlacementManifestV1


@dataclass(frozen=True, slots=True, init=False)
class DecodeResourceObservationV1:
    decode_wall_seconds: float
    total_workflow_seconds: float
    peak_memory_nbytes: int
    execution_axis: ExecutionAxisV1
    output_raw_sha256: str
    output_raw_nbytes: int
    deterministic_double_run_equal: bool

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise PublicAuthClosureError("DecodeResourceObservationV1 has no public constructor")

    @classmethod
    def _construct(cls, *, seal: object, **values: object) -> Self:
        return _sealed_frozen_dataclass(
            cls,
            seal=seal,
            allowed_seals=(_RESOURCE_OBSERVATION_DERIVATION_SEAL,),
            values=values,
        )

    @classmethod
    def from_public_execution(cls, execution: PublicEvaluatorExecutionReceiptV1) -> Self:
        """Derive resource evidence only from a strict, completed public execution."""

        if type(execution) is not PublicEvaluatorExecutionReceiptV1:
            raise PublicAuthClosureError("decode observation requires a sealed public execution")
        if execution.research_only is not False or execution.exact_double_run_equal is not True:
            raise PublicAuthClosureError("decode observation parent lacks completed double-run authority")
        return cls._construct(
            seal=_RESOURCE_OBSERVATION_DERIVATION_SEAL,
            decode_wall_seconds=execution.decode_wall_seconds_max,
            total_workflow_seconds=execution.total_workflow_seconds_max,
            peak_memory_nbytes=execution.peak_process_tree_memory_nbytes_max,
            execution_axis=execution.execution_axis,
            output_raw_sha256=execution.raw_sha256,
            output_raw_nbytes=execution.raw_nbytes,
            deterministic_double_run_equal=execution.exact_double_run_equal,
        )

    def __post_init__(self) -> None:
        _require_finite_nonnegative(self.decode_wall_seconds, label="decode wall observation")
        _require_finite_nonnegative(self.total_workflow_seconds, label="total workflow observation")
        if self.decode_wall_seconds > self.total_workflow_seconds:
            raise PublicAuthClosureError("decode wall time cannot exceed total official workflow time")
        if type(self.peak_memory_nbytes) is not int or self.peak_memory_nbytes < 0:
            raise PublicAuthClosureError("decode peak memory must be nonnegative int")
        if type(self.execution_axis) is not ExecutionAxisV1:
            raise PublicAuthClosureError("decode execution axis must be typed")
        _require_sha256(self.output_raw_sha256, label="decode raw output")
        if type(self.output_raw_nbytes) is not int or self.output_raw_nbytes < 1:
            raise PublicAuthClosureError("decode raw output size must be positive int")
        if type(self.deterministic_double_run_equal) is not bool:
            raise PublicAuthClosureError("decode double-run equality flag must be bool")

    def admitted_by(self, envelope: DecodeResourceEnvelopeV1) -> bool:
        axis_memory_limit = {
            ExecutionAxisV1.CPU: CONTEST_MEMORY_NBYTES,
            ExecutionAxisV1.CUDA: CONTEST_CUDA_HOST_MEMORY_NBYTES,
        }[self.execution_axis]
        return (
            self.total_workflow_seconds <= envelope.max_total_workflow_seconds
            and self.peak_memory_nbytes <= min(envelope.max_peak_memory_nbytes, axis_memory_limit)
            and self.execution_axis in envelope.legal_axes
            and self.deterministic_double_run_equal
            and self.output_raw_nbytes == EXPECTED_RAW_NBYTES
        )

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(
            {
                "deterministic_double_run_equal": self.deterministic_double_run_equal,
                "execution_axis": self.execution_axis.value,
                "output_raw_nbytes": self.output_raw_nbytes,
                "output_raw_sha256": self.output_raw_sha256,
                "peak_memory_nbytes": self.peak_memory_nbytes,
                "decode_wall_seconds": self.decode_wall_seconds,
                "total_workflow_seconds": self.total_workflow_seconds,
            }
        )


@dataclass(frozen=True, slots=True, init=False)
class RepresentationEvidenceV1:
    representation: RepresentationClassV1
    authority: MeasurementAuthorityV1
    comparison_domain_sha256: str
    auth_closure_identity_sha256: str
    archive_sha256: str
    archive_nbytes: int
    avg_segnet_dist: float
    avg_posenet_dist: float
    reported_final_score_decimal: str
    observation: DecodeResourceObservationV1
    portable_axes: tuple[ExecutionAxisV1, ...]
    public_path_exact: bool
    interpreter_and_distributions_closed: bool
    sample_pairs: int
    measurement_receipt: G17ReopenedEvidencePacketV1 = field(repr=False)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise PublicAuthClosureError("RepresentationEvidenceV1 has no public constructor")

    @classmethod
    def _construct(cls, *, seal: object, **values: object) -> Self:
        return _sealed_frozen_dataclass(
            cls,
            seal=seal,
            allowed_seals=(_REPRESENTATION_EVIDENCE_DERIVATION_SEAL,),
            values=values,
        )

    @classmethod
    def from_public_execution(
        cls,
        *,
        representation: RepresentationClassV1,
        comparison_domain_sha256: str,
        execution: PublicEvaluatorExecutionReceiptV1,
    ) -> Self:
        """Derive one axis-specific representation row from retained public proof."""

        if type(execution) is not PublicEvaluatorExecutionReceiptV1:
            raise PublicAuthClosureError("representation evidence requires sealed public execution")
        packet = G17ReopenedEvidencePacketV1(
            exact_packet_bytes=execution.to_receipt_bytes(),
            strict_parser=PublicEvaluatorExecutionReceiptV1.from_receipt_bytes,
            expected_schema=PUBLIC_EXECUTION_SCHEMA,
        )
        packet.reopen()
        authority = {
            ExecutionAxisV1.CPU: MeasurementAuthorityV1.CONTEST_CPU,
            ExecutionAxisV1.CUDA: MeasurementAuthorityV1.CONTEST_CUDA,
        }[execution.execution_axis]
        return cls._construct(
            seal=_REPRESENTATION_EVIDENCE_DERIVATION_SEAL,
            representation=representation,
            authority=authority,
            comparison_domain_sha256=comparison_domain_sha256,
            auth_closure_identity_sha256=execution.identity_sha256,
            archive_sha256=execution.archive_sha256,
            archive_nbytes=execution.archive_nbytes,
            avg_segnet_dist=execution.avg_segnet_dist,
            avg_posenet_dist=execution.avg_posenet_dist,
            reported_final_score_decimal=execution.reported_final_score_decimal,
            observation=DecodeResourceObservationV1.from_public_execution(execution),
            portable_axes=(execution.execution_axis,),
            public_path_exact=execution.exact_runtime_file_observation_closed,
            interpreter_and_distributions_closed=True,
            sample_pairs=EXPECTED_N_PAIRS,
            measurement_receipt=packet,
        )

    def __post_init__(self) -> None:
        if type(self.representation) is not RepresentationClassV1:
            raise PublicAuthClosureError("representation evidence class must be typed")
        if type(self.authority) is not MeasurementAuthorityV1:
            raise PublicAuthClosureError("representation authority must be typed")
        for value, label in (
            (self.comparison_domain_sha256, "comparison domain"),
            (self.auth_closure_identity_sha256, "auth closure identity"),
            (self.archive_sha256, "evidence archive"),
        ):
            _require_sha256(value, label=label)
        if type(self.archive_nbytes) is not int or self.archive_nbytes < 1:
            raise PublicAuthClosureError("evidence archive_nbytes must be positive int")
        _require_finite_nonnegative(self.avg_segnet_dist, label="average SegNet distortion")
        _require_finite_nonnegative(self.avg_posenet_dist, label="average PoseNet distortion")
        if not _reported_score_interval_consistent(
            displayed_two_decimal=self.reported_final_score_decimal,
            avg_segnet_dist_8dec=self.avg_segnet_dist,
            avg_posenet_dist_8dec=self.avg_posenet_dist,
            archive_nbytes=self.archive_nbytes,
            original_uncompressed_nbytes=SCORE_RATE_DENOMINATOR,
        ):
            raise PublicAuthClosureError("representation evidence report text and components disagree")
        if type(self.observation) is not DecodeResourceObservationV1:
            raise PublicAuthClosureError("representation decode observation must be typed")
        if (
            type(self.portable_axes) is not tuple
            or not self.portable_axes
            or any(type(axis) is not ExecutionAxisV1 for axis in self.portable_axes)
            or self.portable_axes != tuple(sorted(set(self.portable_axes), key=lambda item: item.value))
        ):
            raise PublicAuthClosureError("representation portable axes must be canonical typed tuple")
        if self.public_path_exact is not True or self.interpreter_and_distributions_closed is not True:
            raise PublicAuthClosureError("exact evidence requires public-path and interpreter/distribution closure")
        if self.sample_pairs != EXPECTED_N_PAIRS:
            raise PublicAuthClosureError("representation evidence must cover all 600 non-overlapping pairs")
        if type(self.measurement_receipt) is not G17ReopenedEvidencePacketV1:
            raise PublicAuthClosureError("representation evidence requires strict reopened receipt")
        self.measurement_receipt.reopen()

    @property
    def report_component_recomputed_score(self) -> float:
        return compute_contest_score(
            self.avg_segnet_dist,
            self.avg_posenet_dist,
            self.archive_nbytes,
            uncompressed_size=SCORE_RATE_DENOMINATOR,
        )

    @property
    def report_component_score_interval(self) -> tuple[float, float]:
        return _reported_component_score_interval(
            avg_segnet_dist_8dec=self.avg_segnet_dist,
            avg_posenet_dist_8dec=self.avg_posenet_dist,
            archive_nbytes=self.archive_nbytes,
            original_uncompressed_nbytes=SCORE_RATE_DENOMINATOR,
        )

    def feasible(self, envelope: DecodeResourceEnvelopeV1) -> bool:
        return self.observation.admitted_by(envelope) and set(envelope.legal_axes).issubset(self.portable_axes)

    def dominates(self, other: RepresentationEvidenceV1, envelope: DecodeResourceEnvelopeV1) -> bool:
        if self.comparison_domain_sha256 != other.comparison_domain_sha256:
            return False
        if not self.feasible(envelope):
            return False
        if not other.feasible(envelope):
            return True
        self_score_low, self_score_high = self.report_component_score_interval
        other_score_low, _other_score_high = other.report_component_score_interval
        weak = (
            self_score_high <= other_score_low
            and self.archive_nbytes <= other.archive_nbytes
            and self.observation.decode_wall_seconds <= other.observation.decode_wall_seconds
            and self.observation.total_workflow_seconds <= other.observation.total_workflow_seconds
            and self.observation.peak_memory_nbytes <= other.observation.peak_memory_nbytes
            and set(self.portable_axes).issuperset(other.portable_axes)
        )
        strict = (
            self_score_high < other_score_low
            or self.archive_nbytes < other.archive_nbytes
            or self.observation.decode_wall_seconds < other.observation.decode_wall_seconds
            or self.observation.total_workflow_seconds < other.observation.total_workflow_seconds
            or self.observation.peak_memory_nbytes < other.observation.peak_memory_nbytes
            or set(self.portable_axes) != set(other.portable_axes)
        )
        return weak and strict


@dataclass(frozen=True, slots=True)
class TerminalTrainingAdmissionRequestV1:
    obligation_id: str
    current_exact_reference: RepresentationEvidenceV1
    nontrained_evidence: tuple[RepresentationEvidenceV1, ...]
    placement: PayloadPlacementManifestV1
    requested_representation: RepresentationClassV1
    terminal_schedule: G17TerminalCompilerScheduleV1

    def __post_init__(self) -> None:
        _require_identifier(self.obligation_id, label="training obligation id")
        if type(self.current_exact_reference) is not RepresentationEvidenceV1:
            raise PublicAuthClosureError("training reference must be exact typed evidence")
        if type(self.nontrained_evidence) is not tuple or any(
            type(item) is not RepresentationEvidenceV1 for item in self.nontrained_evidence
        ):
            raise PublicAuthClosureError("nontrained evidence must be exact typed tuple")
        if type(self.placement) is not PayloadPlacementManifestV1:
            raise PublicAuthClosureError("training request placement must be typed")
        if type(self.requested_representation) is not RepresentationClassV1:
            raise PublicAuthClosureError("requested training representation must be typed")
        if type(self.terminal_schedule) is not G17TerminalCompilerScheduleV1:
            raise PublicAuthClosureError("training request terminal schedule must be typed")


@dataclass(frozen=True, slots=True)
class TerminalTrainingAdmissionDecisionV1:
    admitted: bool
    reasons: tuple[str, ...]
    exact_rows_checked: int
    training_scope: RepresentationClassV1
    terminal_joint_descent_only: bool


def decide_terminal_training_admission(
    request: TerminalTrainingAdmissionRequestV1,
) -> TerminalTrainingAdmissionDecisionV1:
    """Admit training only after three exact public nontrained rows are dominated."""

    reasons: list[str] = []
    required = (
        RepresentationClassV1.ANALYTIC_INVERSE,
        RepresentationClassV1.BOUNDED_ENUMERATION,
        RepresentationClassV1.DETERMINISTIC_GENERATIVE_REPAIR,
    )
    observed = tuple(item.representation for item in request.nontrained_evidence)
    if observed != required:
        reasons.append("nontrained_evidence_missing_or_out_of_order")
    if request.requested_representation is not RepresentationClassV1.IRREDUCIBLE_LEARNED_RESIDUE:
        reasons.append("training_requested_for_reducible_representation")
    try:
        canonical_schedule = G17TerminalCompilerScheduleV1.canonical()
    except Exception as exc:  # pragma: no cover - defensive against upstream API drift.
        raise PublicAuthClosureError("canonical terminal schedule is unavailable") from exc
    if request.terminal_schedule != canonical_schedule:
        reasons.append("joint_descent_not_terminal_only")
    reference = request.current_exact_reference
    if reference.representation is not RepresentationClassV1.IRREDUCIBLE_LEARNED_RESIDUE:
        reasons.append("exact_reference_is_not_current_irreducible_vehicle")
    envelope = request.placement.decode_envelope
    if not reference.feasible(envelope):
        reasons.append("current_reference_fails_public_decode_resource_or_portability_envelope")
    for row in request.nontrained_evidence:
        if row.comparison_domain_sha256 != reference.comparison_domain_sha256:
            reasons.append(f"comparison_domain_mismatch:{row.representation.value}")
        elif not reference.dominates(row, envelope):
            reasons.append(f"nontrained_not_exactly_dominated:{row.representation.value}")
    return TerminalTrainingAdmissionDecisionV1(
        admitted=not reasons,
        reasons=tuple(reasons),
        exact_rows_checked=len(request.nontrained_evidence),
        training_scope=request.requested_representation,
        terminal_joint_descent_only=(request.terminal_schedule == canonical_schedule),
    )


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return None if prefix is None else f"{prefix}.{node.attr}"
    return None


@dataclass(frozen=True, slots=True)
class GenericSourceAuditReceiptV1:
    source_name: str
    source_sha256: str
    source_nbytes: int
    imported_roots: tuple[str, ...]
    call_targets: tuple[str, ...]
    environment_keys: tuple[str, ...]
    largest_literal_nbytes: int
    largest_literal_items: int
    lineage_attested_generic: bool
    forbidden_findings: tuple[str, ...]
    schema: str = field(default=GENERIC_SOURCE_AUDIT_SCHEMA, init=False)

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "call_targets",
            "environment_keys",
            "forbidden_findings",
            "imported_roots",
            "largest_literal_items",
            "largest_literal_nbytes",
            "lineage_attested_generic",
            "schema",
            "source_name",
            "source_nbytes",
            "source_sha256",
        }
    )

    def __post_init__(self) -> None:
        _relative_path(self.source_name, label="generic source name")
        _require_sha256(self.source_sha256, label="generic source")
        if type(self.source_nbytes) is not int or self.source_nbytes < 1:
            raise PublicAuthClosureError("generic source_nbytes must be positive int")
        for values, label in (
            (self.imported_roots, "generic source imports"),
            (self.call_targets, "generic source calls"),
            (self.environment_keys, "generic source environment keys"),
            (self.forbidden_findings, "generic source forbidden findings"),
        ):
            if (
                type(values) is not tuple
                or any(type(value) is not str or not value.isascii() for value in values)
                or values != tuple(sorted(set(values)))
            ):
                raise PublicAuthClosureError(f"{label} must be unique canonical ASCII tuple")
        if type(self.largest_literal_nbytes) is not int or self.largest_literal_nbytes < 0:
            raise PublicAuthClosureError("largest literal bytes must be nonnegative int")
        if type(self.largest_literal_items) is not int or self.largest_literal_items < 0:
            raise PublicAuthClosureError("largest literal items must be nonnegative int")
        if type(self.lineage_attested_generic) is not bool:
            raise PublicAuthClosureError("generic lineage attestation must be bool")

    @property
    def passed(self) -> bool:
        return self.lineage_attested_generic and not self.forbidden_findings

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(
            {
                "call_targets": list(self.call_targets),
                "environment_keys": list(self.environment_keys),
                "forbidden_findings": list(self.forbidden_findings),
                "imported_roots": list(self.imported_roots),
                "largest_literal_items": self.largest_literal_items,
                "largest_literal_nbytes": self.largest_literal_nbytes,
                "lineage_attested_generic": self.lineage_attested_generic,
                "schema": self.schema,
                "source_name": self.source_name,
                "source_nbytes": self.source_nbytes,
                "source_sha256": self.source_sha256,
            }
        )

    @classmethod
    def from_receipt_bytes(cls, payload: bytes) -> Self:
        value = _parse_canonical_object(payload, exact_keys=cls._KEYS)
        if value["schema"] != GENERIC_SOURCE_AUDIT_SCHEMA:
            raise PublicAuthClosureError("generic source audit schema drifted")
        return cls(
            source_name=value["source_name"],
            source_sha256=value["source_sha256"],
            source_nbytes=value["source_nbytes"],
            imported_roots=tuple(value["imported_roots"]),
            call_targets=tuple(value["call_targets"]),
            environment_keys=tuple(value["environment_keys"]),
            largest_literal_nbytes=value["largest_literal_nbytes"],
            largest_literal_items=value["largest_literal_items"],
            lineage_attested_generic=value["lineage_attested_generic"],
            forbidden_findings=tuple(value["forbidden_findings"]),
        )


def audit_generic_runtime_source(
    source_bytes: bytes,
    *,
    source_name: str,
    lineage_attested_generic: bool,
) -> GenericSourceAuditReceiptV1:
    """Audit executable AST for imports, loaders, and likely code-as-data literals.

    This is a fail-closed structural audit plus an explicit lineage attestation;
    it is not described as a proof that provenance can be inferred from syntax.
    """

    if type(source_bytes) is not bytes or not source_bytes:
        raise PublicAuthClosureError("generic runtime source must be nonempty immutable bytes")
    _relative_path(source_name, label="generic source name")
    try:
        source_text = source_bytes.decode("utf-8")
        tree = ast.parse(source_text, filename=source_name)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise PublicAuthClosureError(f"generic runtime source {source_name!r} does not parse") from exc
    imports: set[str] = set()
    calls: set[str] = set()
    environment_keys: set[str] = set()
    findings: set[str] = set()
    loop_environment_values: dict[str, frozenset[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name):
            continue
        if not isinstance(node.iter, (ast.Tuple, ast.List)):
            continue
        values = tuple(
            child.value for child in node.iter.elts if isinstance(child, ast.Constant) and isinstance(child.value, str)
        )
        if len(values) == len(node.iter.elts):
            loop_environment_values[node.target.id] = frozenset(values)

    def environment_call_keys(node: ast.Call) -> frozenset[str] | None:
        target = _dotted_name(node.func)
        if target not in {"os.getenv", "os.environ.get", "os.environ.setdefault"} or not node.args:
            return frozenset()
        key = node.args[0]
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            return frozenset({key.value})
        if isinstance(key, ast.Name) and key.id in loop_environment_values:
            return loop_environment_values[key.id]
        return None

    largest_literal_nbytes = 0
    largest_literal_items = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            target = _dotted_name(node.func)
            if target:
                calls.add(target)
                leaf = target.split(".")[-1]
                if (
                    target in _FORBIDDEN_CALLS
                    or leaf in {"exec", "eval", "compile", "Popen", "system"}
                    or any(leaf.startswith(prefix) for prefix in _FORBIDDEN_CALL_LEAF_PREFIXES)
                ):
                    findings.add(f"forbidden_dynamic_or_model_loader_call:{target}")
                keys = environment_call_keys(node)
                if keys is None:
                    findings.add(f"dynamic_environment_key:{target}")
                else:
                    environment_keys.update(keys)
        elif isinstance(node, ast.Name):
            lower = node.id.lower()
            for fragment in _FORBIDDEN_RUNTIME_NAME_FRAGMENTS:
                if fragment in lower:
                    findings.add(f"forbidden_scorer_target_teacher_name:{node.id}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            raw = node.value.encode("utf-8") if isinstance(node.value, str) else node.value
            largest_literal_nbytes = max(largest_literal_nbytes, len(raw))
            lower = raw.lower()
            for fragment in _FORBIDDEN_RUNTIME_NAME_FRAGMENTS:
                if fragment.encode("ascii") in lower:
                    findings.add(f"forbidden_scorer_target_teacher_literal:{fragment}")
            if len(raw) > _MAX_STATIC_LITERAL_NBYTES:
                findings.add(f"oversize_static_literal:{len(raw)}")
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            count = len(node.keys) if isinstance(node, ast.Dict) else len(node.elts)
            largest_literal_items = max(largest_literal_items, count)
            if count > 256 and all(
                isinstance(child, ast.Constant)
                for child in (node.keys if isinstance(node, ast.Dict) else node.elts)
                if child is not None
            ):
                findings.add(f"oversize_constant_sequence:{count}")
        elif isinstance(node, ast.Subscript) and _dotted_name(node.value) == "os.environ":
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                environment_keys.add(key.value)
            else:
                findings.add("dynamic_environment_subscript")
    for root in imports:
        if root in _FORBIDDEN_RUNTIME_IMPORT_ROOTS:
            findings.add(f"forbidden_runtime_import:{root}")
    for key in environment_keys - _PUBLIC_RUNTIME_ENVIRONMENT_KEYS:
        findings.add(f"unbound_environment_key:{key}")
    receipt = GenericSourceAuditReceiptV1(
        source_name=source_name,
        source_sha256=_sha256(source_bytes),
        source_nbytes=len(source_bytes),
        imported_roots=tuple(sorted(imports)),
        call_targets=tuple(sorted(calls)),
        environment_keys=tuple(sorted(environment_keys)),
        largest_literal_nbytes=largest_literal_nbytes,
        largest_literal_items=largest_literal_items,
        lineage_attested_generic=lineage_attested_generic,
        forbidden_findings=tuple(sorted(findings)),
    )
    if not receipt.passed:
        raise PublicAuthClosureError(
            f"generic runtime source audit refused {source_name}: {receipt.forbidden_findings}"
        )
    return receipt


@dataclass(frozen=True, slots=True)
class DistributionABIIdentityV1:
    import_root: str
    distribution_name: str
    version: str
    metadata_sha256: str
    record_sha256: str
    import_origin_kind: str
    import_origin_relpath: str
    import_origin_sha256: str | None
    installed_files_tree_sha256: str
    installed_files_nbytes: int
    native_linkage_sha256: str

    def __post_init__(self) -> None:
        _require_ascii(self.import_root, label="ABI import root")
        _require_ascii(self.distribution_name, label="ABI distribution name")
        _require_ascii(self.version, label="ABI distribution version")
        _require_sha256(self.metadata_sha256, label="ABI METADATA")
        _require_sha256(self.record_sha256, label="ABI RECORD")
        _require_ascii(self.import_origin_kind, label="ABI import origin kind")
        _relative_path(self.import_origin_relpath, label="ABI import origin relpath")
        if self.import_origin_sha256 is not None:
            _require_sha256(self.import_origin_sha256, label="ABI import origin")
        _require_sha256(self.installed_files_tree_sha256, label="ABI installed files tree")
        if type(self.installed_files_nbytes) is not int or self.installed_files_nbytes < 1:
            raise PublicAuthClosureError("ABI installed file bytes must be positive int")
        _require_sha256(self.native_linkage_sha256, label="ABI native linkage")

    def to_dict(self) -> dict[str, Any]:
        return {
            "distribution_name": self.distribution_name,
            "import_origin_kind": self.import_origin_kind,
            "import_origin_relpath": self.import_origin_relpath,
            "import_origin_sha256": self.import_origin_sha256,
            "import_root": self.import_root,
            "installed_files_nbytes": self.installed_files_nbytes,
            "installed_files_tree_sha256": self.installed_files_tree_sha256,
            "metadata_sha256": self.metadata_sha256,
            "native_linkage_sha256": self.native_linkage_sha256,
            "record_sha256": self.record_sha256,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        expected = {
            "distribution_name",
            "import_origin_kind",
            "import_origin_relpath",
            "import_origin_sha256",
            "import_root",
            "installed_files_nbytes",
            "installed_files_tree_sha256",
            "metadata_sha256",
            "native_linkage_sha256",
            "record_sha256",
            "version",
        }
        if type(value) is not dict or set(value) != expected:
            raise PublicAuthClosureError("distribution ABI identity fields drifted")
        return cls(**value)


def _metadata_payload(distribution: importlib.metadata.Distribution, name: str) -> bytes:
    value = distribution.read_text(name)
    return b"" if value is None else value.encode("utf-8")


def _top_level_import_origin(import_root: str) -> tuple[str, str, str | None]:
    from importlib.machinery import PathFinder

    spec = PathFinder.find_spec(import_root, sys.path)
    if spec is None:
        return "UNRESOLVED", f"unresolved/{import_root}", None
    if spec.origin and spec.origin not in {"built-in", "frozen"}:
        origin = Path(spec.origin).resolve()
        try:
            relative = origin.relative_to(Path(sys.prefix).resolve()).as_posix()
        except ValueError:
            relative = f"external/{origin.name}"
        return "FILE", relative, _sha256_file(origin) if origin.is_file() else None
    locations = tuple(sorted(os.fspath(Path(value).name) for value in (spec.submodule_search_locations or ())))
    label = "namespace/" + ("+".join(locations) if locations else import_root)
    return "NAMESPACE", label, None


def _installed_distribution_tree(
    distribution: importlib.metadata.Distribution,
) -> tuple[str, int, str]:
    distribution_name = distribution.metadata.get("Name") or "unknown"
    tree = hashlib.sha256(b"PACT-INSTALLED-DISTRIBUTION-TREE-V1\x00")
    linkage = hashlib.sha256(b"PACT-NATIVE-LINKAGE-V1\x00")
    total = 0
    native_suffixes = {".dylib", ".dll", ".pyd", ".so"}
    linker = shutil.which("otool") or shutil.which("ldd")
    for relative in sorted(distribution.files or (), key=lambda item: os.fspath(item)):
        path = Path(distribution.locate_file(relative))
        if not path.is_file():
            continue
        size = path.stat().st_size
        content_sha = _sha256_file(path)
        rel = os.fspath(relative).replace(os.sep, "/")
        tree.update(rel.encode("utf-8"))
        tree.update(b"\x00")
        tree.update(size.to_bytes(8, "big"))
        tree.update(bytes.fromhex(content_sha))
        total += size
        if path.suffix.lower() in native_suffixes or ".so." in path.name:
            linkage.update(rel.encode("utf-8"))
            linkage.update(bytes.fromhex(content_sha))
            if linker is not None:
                completed = subprocess.run(
                    (linker, "-L", os.fspath(path)) if Path(linker).name == "otool" else (linker, os.fspath(path)),
                    capture_output=True,
                    check=False,
                )
                linkage.update(completed.stdout)
                linkage.update(completed.stderr)
                linkage.update(str(completed.returncode).encode("ascii"))
    if total < 1:
        raise PublicAuthClosureError(f"installed distribution {distribution_name} has no byte-owned files")
    return tree.hexdigest(), total, linkage.hexdigest()


def _complete_prefix_tree_identity(prefixes: tuple[str, ...]) -> tuple[str, int]:
    """Hash every interpreter-prefix file/symlink for full ABI read custody."""

    digest = hashlib.sha256(b"PACT-COMPLETE-INTERPRETER-PREFIX-TREE-V1\x00")
    total = 0
    for root_value in prefixes:
        root = Path(root_value).resolve(strict=True)
        digest.update(root.as_posix().encode("ascii"))
        digest.update(b"\x00")
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                target = os.readlink(path).encode("utf-8")
                digest.update(b"L\x00" + relative.encode("utf-8") + b"\x00" + target + b"\x00")
                total += len(target)
            elif path.is_file():
                size = path.stat().st_size
                digest.update(b"F\x00" + relative.encode("utf-8") + b"\x00")
                digest.update(size.to_bytes(8, "big"))
                digest.update(bytes.fromhex(_sha256_file(path)))
                total += size
    if total < 1:
        raise PublicAuthClosureError("complete interpreter prefix tree has no retained bytes")
    return digest.hexdigest(), total


def _distribution_identity(import_root: str) -> DistributionABIIdentityV1 | None:
    mapping = importlib.metadata.packages_distributions()
    candidates = list(mapping.get(import_root, ()))
    alias = _DIST_IMPORT_ALIASES.get(import_root)
    if alias is not None:
        candidates.append(alias)
    for name in dict.fromkeys(candidates):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        metadata = _metadata_payload(distribution, "METADATA")
        record = _metadata_payload(distribution, "RECORD")
        if not record:
            file_names = sorted(os.fspath(value) for value in (distribution.files or ()))
            record = _canonical_json(file_names)
        kind, relpath, origin_sha = _top_level_import_origin(import_root)
        installed_tree, installed_nbytes, native_linkage = _installed_distribution_tree(distribution)
        return DistributionABIIdentityV1(
            import_root=import_root,
            distribution_name=distribution.metadata.get("Name") or name,
            version=distribution.version,
            metadata_sha256=_sha256(metadata),
            record_sha256=_sha256(record),
            import_origin_kind=kind,
            import_origin_relpath=relpath,
            import_origin_sha256=origin_sha,
            installed_files_tree_sha256=installed_tree,
            installed_files_nbytes=installed_nbytes,
            native_linkage_sha256=native_linkage,
        )
    return None


@dataclass(frozen=True, slots=True)
class InterpreterDistributionABIClosureV1:
    interpreter_implementation: str
    interpreter_version: str
    interpreter_cache_tag: str
    interpreter_executable_name: str
    interpreter_executable_realpath: str
    interpreter_executable_sha256: str
    interpreter_prefix_realpaths: tuple[str, ...]
    interpreter_prefix_tree_sha256: str | None
    interpreter_prefix_tree_nbytes: int
    soabi: str
    multiarch: str
    platform_system: str
    platform_machine: str
    distributions: tuple[DistributionABIIdentityV1, ...]
    unresolved_import_roots: tuple[str, ...]
    schema: str = field(default=ABI_CLOSURE_SCHEMA, init=False)

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "distributions",
            "interpreter_cache_tag",
            "interpreter_executable_name",
            "interpreter_executable_realpath",
            "interpreter_executable_sha256",
            "interpreter_prefix_realpaths",
            "interpreter_prefix_tree_nbytes",
            "interpreter_prefix_tree_sha256",
            "interpreter_implementation",
            "interpreter_version",
            "multiarch",
            "platform_machine",
            "platform_system",
            "schema",
            "soabi",
            "unresolved_import_roots",
        }
    )

    def __post_init__(self) -> None:
        for value, label in (
            (self.interpreter_implementation, "interpreter implementation"),
            (self.interpreter_version, "interpreter version"),
            (self.interpreter_cache_tag, "interpreter cache tag"),
            (self.interpreter_executable_name, "interpreter executable name"),
            (self.interpreter_executable_realpath, "interpreter executable realpath"),
            (self.soabi, "interpreter SOABI"),
            (self.multiarch, "interpreter MULTIARCH"),
            (self.platform_system, "platform system"),
            (self.platform_machine, "platform machine"),
        ):
            _require_ascii(value, label=label)
        _require_sha256(self.interpreter_executable_sha256, label="interpreter executable")
        if (
            type(self.interpreter_prefix_realpaths) is not tuple
            or not self.interpreter_prefix_realpaths
            or self.interpreter_prefix_realpaths != tuple(sorted(set(self.interpreter_prefix_realpaths)))
            or any(
                type(value) is not str or not value.isascii() or not Path(value).is_absolute()
                for value in self.interpreter_prefix_realpaths
            )
        ):
            raise PublicAuthClosureError("interpreter ABI prefixes must be canonical absolute roots")
        if self.interpreter_prefix_tree_sha256 is None:
            if self.interpreter_prefix_tree_nbytes != 0:
                raise PublicAuthClosureError("uncaptured interpreter prefix tree cannot claim bytes")
        else:
            _require_sha256(self.interpreter_prefix_tree_sha256, label="interpreter prefix tree")
            if type(self.interpreter_prefix_tree_nbytes) is not int or self.interpreter_prefix_tree_nbytes < 1:
                raise PublicAuthClosureError("captured interpreter prefix tree must retain positive bytes")
        if (
            type(self.distributions) is not tuple
            or any(type(item) is not DistributionABIIdentityV1 for item in self.distributions)
            or self.distributions != tuple(sorted(self.distributions, key=lambda item: item.import_root))
            or len({item.import_root for item in self.distributions}) != len(self.distributions)
        ):
            raise PublicAuthClosureError("ABI distributions must be unique canonical typed tuple")
        if type(self.unresolved_import_roots) is not tuple or self.unresolved_import_roots != tuple(
            sorted(set(self.unresolved_import_roots))
        ):
            raise PublicAuthClosureError("ABI unresolved roots must be unique canonical tuple")

    @property
    def closed(self) -> bool:
        return not self.unresolved_import_roots

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(
            {
                "distributions": [item.to_dict() for item in self.distributions],
                "interpreter_cache_tag": self.interpreter_cache_tag,
                "interpreter_executable_name": self.interpreter_executable_name,
                "interpreter_executable_realpath": self.interpreter_executable_realpath,
                "interpreter_executable_sha256": self.interpreter_executable_sha256,
                "interpreter_prefix_realpaths": list(self.interpreter_prefix_realpaths),
                "interpreter_prefix_tree_nbytes": self.interpreter_prefix_tree_nbytes,
                "interpreter_prefix_tree_sha256": self.interpreter_prefix_tree_sha256,
                "interpreter_implementation": self.interpreter_implementation,
                "interpreter_version": self.interpreter_version,
                "multiarch": self.multiarch,
                "platform_machine": self.platform_machine,
                "platform_system": self.platform_system,
                "schema": self.schema,
                "soabi": self.soabi,
                "unresolved_import_roots": list(self.unresolved_import_roots),
            }
        )

    @classmethod
    def from_receipt_bytes(cls, payload: bytes) -> Self:
        value = _parse_canonical_object(payload, exact_keys=cls._KEYS)
        if value["schema"] != ABI_CLOSURE_SCHEMA:
            raise PublicAuthClosureError("interpreter/distribution ABI closure schema drifted")
        return cls(
            interpreter_implementation=value["interpreter_implementation"],
            interpreter_version=value["interpreter_version"],
            interpreter_cache_tag=value["interpreter_cache_tag"],
            interpreter_executable_name=value["interpreter_executable_name"],
            interpreter_executable_realpath=value["interpreter_executable_realpath"],
            interpreter_executable_sha256=value["interpreter_executable_sha256"],
            interpreter_prefix_realpaths=tuple(value["interpreter_prefix_realpaths"]),
            interpreter_prefix_tree_sha256=value["interpreter_prefix_tree_sha256"],
            interpreter_prefix_tree_nbytes=value["interpreter_prefix_tree_nbytes"],
            soabi=value["soabi"],
            multiarch=value["multiarch"],
            platform_system=value["platform_system"],
            platform_machine=value["platform_machine"],
            distributions=tuple(DistributionABIIdentityV1.from_dict(row) for row in value["distributions"]),
            unresolved_import_roots=tuple(value["unresolved_import_roots"]),
        )


def capture_interpreter_distribution_abi(
    imported_roots: tuple[str, ...],
    *,
    capture_complete_prefix_tree: bool = False,
) -> InterpreterDistributionABIClosureV1:
    third_party = tuple(
        sorted(root for root in set(imported_roots) if root not in _STDLIB_ROOTS and root not in {"__future__"})
    )
    identities: list[DistributionABIIdentityV1] = []
    unresolved: list[str] = []
    for root in third_party:
        identity = _distribution_identity(root)
        if identity is None:
            unresolved.append(root)
        else:
            identities.append(identity)
    executable = Path(sys.executable).resolve()
    prefixes = tuple(sorted({Path(sys.prefix).resolve().as_posix(), Path(sys.base_prefix).resolve().as_posix()}))
    prefix_tree_sha256, prefix_tree_nbytes = (
        _complete_prefix_tree_identity(prefixes) if capture_complete_prefix_tree else (None, 0)
    )
    return InterpreterDistributionABIClosureV1(
        interpreter_implementation=sys.implementation.name,
        interpreter_version=platform.python_version(),
        interpreter_cache_tag=sys.implementation.cache_tag or "none",
        interpreter_executable_name=executable.name,
        interpreter_executable_realpath=executable.as_posix(),
        interpreter_executable_sha256=_sha256_file(executable),
        interpreter_prefix_realpaths=prefixes,
        interpreter_prefix_tree_sha256=prefix_tree_sha256,
        interpreter_prefix_tree_nbytes=prefix_tree_nbytes,
        soabi=str(sysconfig.get_config_var("SOABI") or "none"),
        multiarch=str(sysconfig.get_config_var("MULTIARCH") or "none"),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        distributions=tuple(sorted(identities, key=lambda item: item.import_root)),
        unresolved_import_roots=tuple(unresolved),
    )


def capture_axis_specific_evaluator_abi(
    execution_axis: ExecutionAxisV1,
) -> InterpreterDistributionABIClosureV1:
    """Capture the full known evaluator+decoder distribution/native surface for one axis."""

    if type(execution_axis) is not ExecutionAxisV1:
        raise PublicAuthClosureError("evaluator ABI capture axis must be typed")
    required = _EVALUATOR_CPU_ABI_ROOTS if execution_axis is ExecutionAxisV1.CPU else _EVALUATOR_CUDA_ABI_ROOTS
    closure = capture_interpreter_distribution_abi(
        tuple(sorted(required)),
        capture_complete_prefix_tree=True,
    )
    _require_axis_specific_evaluator_abi(closure, execution_axis=execution_axis)
    return closure


def _require_axis_specific_evaluator_abi(
    closure: InterpreterDistributionABIClosureV1,
    *,
    execution_axis: ExecutionAxisV1,
) -> None:
    if type(closure) is not InterpreterDistributionABIClosureV1 or not closure.closed:
        raise PublicAuthClosureError("axis-specific evaluator ABI is unresolved")
    required = _EVALUATOR_CPU_ABI_ROOTS if execution_axis is ExecutionAxisV1.CPU else _EVALUATOR_CUDA_ABI_ROOTS
    observed = {item.import_root for item in closure.distributions}
    if not required.issubset(observed):
        raise PublicAuthClosureError(f"axis-specific evaluator ABI lacks required roots: {sorted(required - observed)}")
    if closure.interpreter_prefix_tree_sha256 is None or closure.interpreter_prefix_tree_nbytes < 1:
        raise PublicAuthClosureError("axis-specific evaluator ABI lacks complete interpreter-prefix custody")
    if closure.platform_system != "Linux" or closure.platform_machine not in {"x86_64", "amd64"}:
        raise PublicAuthClosureError("axis-specific evaluator ABI requires Linux x86_64 custody")


@dataclass(frozen=True, slots=True)
class PublicRuntimeFileDigestV1:
    relative_path: str
    content_sha256: str
    nbytes: int
    executable: bool

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, label="public runtime file path")
        _require_sha256(self.content_sha256, label="public runtime file")
        if type(self.nbytes) is not int or self.nbytes < 1:
            raise PublicAuthClosureError("public runtime file size must be positive int")
        if type(self.executable) is not bool:
            raise PublicAuthClosureError("public runtime executable flag must be bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_sha256": self.content_sha256,
            "executable": self.executable,
            "nbytes": self.nbytes,
            "relative_path": self.relative_path,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            "content_sha256",
            "executable",
            "nbytes",
            "relative_path",
        }:
            raise PublicAuthClosureError("public runtime file digest fields drifted")
        return cls(
            relative_path=value["relative_path"],
            content_sha256=value["content_sha256"],
            nbytes=value["nbytes"],
            executable=value["executable"],
        )


@dataclass(frozen=True, slots=True)
class PublicRuntimeCompileReceiptV1:
    archive_sha256: str
    archive_nbytes: int
    member_sha256: str
    member_nbytes: int
    decoded_state_sha256: str
    materialized_lvls1_sha256: str
    runtime_tree_sha256: str
    runtime_files: tuple[PublicRuntimeFileDigestV1, ...]
    source_audit_receipt_sha256s: tuple[str, ...]
    placement_identity_sha256: str
    abi_identity_sha256: str
    inverse_process_argv_sha256: str
    parseback_full_state_equal: bool
    bytecode_contamination_paths: tuple[str, ...]
    research_only: bool
    public_n600_output_equality_owed: bool
    schema: str = field(default=RUNTIME_COMPILE_SCHEMA, init=False)

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "abi_identity_sha256",
            "archive_nbytes",
            "archive_sha256",
            "bytecode_contamination_paths",
            "decoded_state_sha256",
            "inverse_process_argv_sha256",
            "materialized_lvls1_sha256",
            "member_nbytes",
            "member_sha256",
            "parseback_full_state_equal",
            "placement_identity_sha256",
            "public_n600_output_equality_owed",
            "research_only",
            "runtime_files",
            "runtime_tree_sha256",
            "schema",
            "source_audit_receipt_sha256s",
        }
    )

    def __post_init__(self) -> None:
        for value, label in (
            (self.archive_sha256, "compiled archive"),
            (self.member_sha256, "compiled member"),
            (self.decoded_state_sha256, "compiled decoded state"),
            (self.materialized_lvls1_sha256, "materialized LVLS1"),
            (self.runtime_tree_sha256, "compiled runtime tree"),
            (self.placement_identity_sha256, "compiled placement"),
            (self.abi_identity_sha256, "compiled ABI"),
            (self.inverse_process_argv_sha256, "inverse process argv"),
        ):
            _require_sha256(value, label=label)
        for value, label in (
            (self.archive_nbytes, "compiled archive bytes"),
            (self.member_nbytes, "compiled member bytes"),
        ):
            if type(value) is not int or value < 1:
                raise PublicAuthClosureError(f"{label} must be positive int")
        if (
            type(self.runtime_files) is not tuple
            or not self.runtime_files
            or any(type(item) is not PublicRuntimeFileDigestV1 for item in self.runtime_files)
            or self.runtime_files != tuple(sorted(self.runtime_files, key=lambda item: item.relative_path))
            or len({item.relative_path for item in self.runtime_files}) != len(self.runtime_files)
        ):
            raise PublicAuthClosureError("compiled runtime files must be unique canonical typed tuple")
        if (
            type(self.source_audit_receipt_sha256s) is not tuple
            or not self.source_audit_receipt_sha256s
            or self.source_audit_receipt_sha256s != tuple(sorted(set(self.source_audit_receipt_sha256s)))
        ):
            raise PublicAuthClosureError("source audit identities must be unique canonical tuple")
        for value in self.source_audit_receipt_sha256s:
            _require_sha256(value, label="source audit receipt")
        if self.parseback_full_state_equal is not True:
            raise PublicAuthClosureError("compiled runtime requires exact full-state parseback equality")
        if self.bytecode_contamination_paths:
            raise PublicAuthClosureError("compiled runtime contains generated Python bytecode")
        if self.research_only is not True or self.public_n600_output_equality_owed is not True:
            raise PublicAuthClosureError("compile receipt must remain research-only with public output proof owed")

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(
            {
                "abi_identity_sha256": self.abi_identity_sha256,
                "archive_nbytes": self.archive_nbytes,
                "archive_sha256": self.archive_sha256,
                "bytecode_contamination_paths": list(self.bytecode_contamination_paths),
                "decoded_state_sha256": self.decoded_state_sha256,
                "inverse_process_argv_sha256": self.inverse_process_argv_sha256,
                "materialized_lvls1_sha256": self.materialized_lvls1_sha256,
                "member_nbytes": self.member_nbytes,
                "member_sha256": self.member_sha256,
                "parseback_full_state_equal": self.parseback_full_state_equal,
                "placement_identity_sha256": self.placement_identity_sha256,
                "public_n600_output_equality_owed": self.public_n600_output_equality_owed,
                "research_only": self.research_only,
                "runtime_files": [item.to_dict() for item in self.runtime_files],
                "runtime_tree_sha256": self.runtime_tree_sha256,
                "schema": self.schema,
                "source_audit_receipt_sha256s": list(self.source_audit_receipt_sha256s),
            }
        )

    @classmethod
    def from_receipt_bytes(cls, payload: bytes) -> Self:
        value = _parse_canonical_object(payload, exact_keys=cls._KEYS)
        if value["schema"] != RUNTIME_COMPILE_SCHEMA:
            raise PublicAuthClosureError("public runtime compile schema drifted")
        return cls(
            archive_sha256=value["archive_sha256"],
            archive_nbytes=value["archive_nbytes"],
            member_sha256=value["member_sha256"],
            member_nbytes=value["member_nbytes"],
            decoded_state_sha256=value["decoded_state_sha256"],
            materialized_lvls1_sha256=value["materialized_lvls1_sha256"],
            runtime_tree_sha256=value["runtime_tree_sha256"],
            runtime_files=tuple(PublicRuntimeFileDigestV1.from_dict(row) for row in value["runtime_files"]),
            source_audit_receipt_sha256s=tuple(value["source_audit_receipt_sha256s"]),
            placement_identity_sha256=value["placement_identity_sha256"],
            abi_identity_sha256=value["abi_identity_sha256"],
            inverse_process_argv_sha256=value["inverse_process_argv_sha256"],
            parseback_full_state_equal=value["parseback_full_state_equal"],
            bytecode_contamination_paths=tuple(value["bytecode_contamination_paths"]),
            research_only=value["research_only"],
            public_n600_output_equality_owed=value["public_n600_output_equality_owed"],
        )


@dataclass(frozen=True, slots=True)
class CompiledPublicRuntimeV1:
    compile_receipt: PublicRuntimeCompileReceiptV1
    placement: PayloadPlacementManifestV1
    abi_closure: InterpreterDistributionABIClosureV1
    source_audits: tuple[GenericSourceAuditReceiptV1, ...]


_INFLATE_SH_BYTES: Final = rb"""#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: inflate.sh ARCHIVE_DIR OUTPUT_DIR VIDEO_NAMES_FILE" >&2
  exit 2
fi
ARCHIVE_DIR="$1"
OUTPUT_DIR="$2"
VIDEO_NAMES_FILE="$3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXE="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUTPUT_DIR"
while IFS= read -r VIDEO_NAME || [ -n "$VIDEO_NAME" ]; do
  [ -z "$VIDEO_NAME" ] && continue
  case "$VIDEO_NAME" in
    /*|*\\*|../*|*/../*|*/..|./*|*/./*)
      echo "unsafe video name: $VIDEO_NAME" >&2
      exit 3
      ;;
  esac
  case "$VIDEO_NAME" in
    *.mkv) BASE="${VIDEO_NAME%.mkv}" ;;
    *) echo "video name must end in .mkv: $VIDEO_NAME" >&2; exit 3 ;;
  esac
  SOURCE="$ARCHIVE_DIR/${BASE}.bin"
  DESTINATION="$OUTPUT_DIR/${BASE}.raw"
  [ -f "$SOURCE" ] || { echo "missing counted member: $SOURCE" >&2; exit 4; }
  mkdir -p "$(dirname "$DESTINATION")"
  if [ -n "${PACT_AUTH_TRACE_DIR:-}" ]; then
    mkdir -p "$PACT_AUTH_TRACE_DIR"
    export PACT_AUTH_TRACE_PATH="$PACT_AUTH_TRACE_DIR/${BASE//\//_}.json"
  else
    unset PACT_AUTH_TRACE_PATH || true
  fi
  "$PYTHON_EXE" "$HERE/inflate.py" "$SOURCE" "$DESTINATION"
done < "$VIDEO_NAMES_FILE"
"""


def _decoded_state_sha256(parsed: Any) -> str:
    digest = hashlib.sha256(b"PACT-PUBLIC-LVPG2-DECODED-STATE-V1\x00")
    for name in parsed.base_order:
        value = parsed.base_quantized[name]
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(2, "big"))
        digest.update(encoded_name)
        digest.update(len(value.shape).to_bytes(1, "big"))
        for dimension in value.shape:
            digest.update(int(dimension).to_bytes(4, "big"))
        digest.update(value.tobytes(order="C"))
    code = parsed.code_quantized
    for dimension in code.shape:
        digest.update(int(dimension).to_bytes(4, "big"))
    digest.update(code.tobytes(order="C"))
    digest.update(len(parsed.pose_bytes).to_bytes(8, "big"))
    digest.update(parsed.pose_bytes)
    return digest.hexdigest()


def _runtime_tree_sha256(files: tuple[PublicRuntimeFileDigestV1, ...]) -> str:
    digest = hashlib.sha256(b"PACT-PUBLIC-RUNTIME-TREE-V1\x00")
    for item in files:
        digest.update(item.relative_path.encode("ascii"))
        digest.update(b"\x00")
        digest.update(item.nbytes.to_bytes(8, "big"))
        digest.update(bytes.fromhex(item.content_sha256))
        digest.update(bytes((int(item.executable),)))
    return digest.hexdigest()


def _write_runtime_file(path: Path, payload: bytes, *, executable: bool) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise PublicAuthClosureError(f"refusing to overwrite drifted public runtime file {path}")
        desired = 0o755 if executable else 0o644
        if stat.S_IMODE(path.stat().st_mode) != desired:
            raise PublicAuthClosureError(f"refusing to repair drifted public runtime mode {path}")
        return
    _atomic_write(path, payload, executable=executable)


def _inspect_exact_lvpg2_archive(archive_path: Path) -> tuple[bytes, bytes, Any]:
    archive_bytes = archive_path.read_bytes()
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            if len(infos) != 1 or infos[0].filename != MEMBER_NAME:
                raise PublicAuthClosureError("LVPG2 archive must contain exactly safe member 0.bin")
            info = infos[0]
            if (
                info.flag_bits != 0
                or info.extra
                or info.comment
                or archive.comment
                or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                or stat.S_ISLNK(info.external_attr >> 16)
            ):
                raise PublicAuthClosureError("LVPG2 archive metadata escaped the simple public ZIP profile")
            member = archive.read(info)
            if archive.testzip() is not None:
                raise PublicAuthClosureError("LVPG2 archive CRC verification failed")
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise PublicAuthClosureError("LVPG2 archive failed strict reopen") from exc
    parsed = parse_population_global_member(member)
    manifest = parsed.manifest
    if (
        manifest.get("n_pairs") != EXPECTED_N_PAIRS
        or manifest.get("code_shape", [None])[0] != EXPECTED_N_FRAMES
        or manifest.get("camera_w") != CAMERA_WIDTH
        or manifest.get("camera_h") != CAMERA_HEIGHT
    ):
        raise PublicAuthClosureError("LVPG2 archive is not the exact 600-pair 1200-frame public raw contract")
    return archive_bytes, member, parsed


def canonical_lvpg2_placement_manifest(
    *,
    archive_bytes: bytes,
    member_sha256: str,
    inverse_source_bytes: bytes,
    lvls1_runtime_source_bytes: bytes,
    offline_compiler_source_bytes: bytes,
    source_population_sha256: str | None = None,
) -> PayloadPlacementManifestV1:
    items = (
        PayloadPlacementItemV1(
            item_id="counted.lvpg2.packet",
            origin=ContentOriginV1.VIDEO_DERIVED_STATE,
            location=PlacementLocationV1.COUNTED_ARCHIVE_PAYLOAD,
            content_sha256=_sha256(archive_bytes),
            object_nbytes=len(archive_bytes),
            charged_archive_nbytes=len(archive_bytes),
            video_payload_class=VideoDerivedPayloadClassV1.MIXED_PAYLOAD_AND_CONTAINER,
            archive_member_path=MEMBER_NAME,
            source_population_sha256=source_population_sha256,
        ),
        PayloadPlacementItemV1(
            item_id="offline.unbounded.compiler",
            origin=ContentOriginV1.GENERIC_ALGORITHM,
            location=PlacementLocationV1.UNBOUNDED_OFFLINE_COMPILER,
            content_sha256=_sha256(offline_compiler_source_bytes),
            object_nbytes=len(offline_compiler_source_bytes),
            charged_archive_nbytes=0,
        ),
        PayloadPlacementItemV1(
            item_id="public.generic.lvls1.renderer",
            origin=ContentOriginV1.GENERIC_ALGORITHM,
            location=PlacementLocationV1.BOUNDED_PUBLIC_VM,
            content_sha256=_sha256(lvls1_runtime_source_bytes),
            object_nbytes=len(lvls1_runtime_source_bytes),
            charged_archive_nbytes=0,
            vm_facility=GenericVMFacilityV1.GENERIC_NETWORK_ARCHITECTURE,
        ),
        PayloadPlacementItemV1(
            item_id="public.generic.lvpg2.inverse",
            origin=ContentOriginV1.GENERIC_ALGORITHM,
            location=PlacementLocationV1.BOUNDED_PUBLIC_VM,
            content_sha256=_sha256(inverse_source_bytes),
            object_nbytes=len(inverse_source_bytes),
            charged_archive_nbytes=0,
            vm_facility=GenericVMFacilityV1.CODEC_INVERSE,
        ),
    )
    _require_sha256(member_sha256, label="placement LVPG2 member")
    return PayloadPlacementManifestV1(
        archive_sha256=_sha256(archive_bytes),
        archive_nbytes=len(archive_bytes),
        items=tuple(sorted(items, key=lambda item: item.item_id)),
        decode_envelope=DecodeResourceEnvelopeV1.contest_default(),
    )


def compile_lvpg2_public_runtime(
    *,
    archive_path: Path,
    lvls1_runtime_source_path: Path,
    runtime_dir: Path,
    lineage_attested_generic: bool,
    source_population_sha256: str | None = None,
) -> CompiledPublicRuntimeV1:
    """Compile and execute-conformance-test the real generic public inverse."""

    archive_path = archive_path.resolve(strict=True)
    lvls1_runtime_source_path = lvls1_runtime_source_path.resolve(strict=True)
    runtime_dir = runtime_dir.resolve(strict=False)
    archive_bytes, member, selected = _inspect_exact_lvpg2_archive(archive_path)
    inverse_source_path = Path(__file__).with_name("taskspace_lvpg2_public_inverse.py")
    inverse_source = inverse_source_path.read_bytes()
    renderer_source = lvls1_runtime_source_path.read_bytes()
    offline_source = Path(__file__).read_bytes()
    inverse_audit = audit_generic_runtime_source(
        inverse_source,
        source_name=PUBLIC_INFLATE_PY_PATH,
        lineage_attested_generic=lineage_attested_generic,
    )
    renderer_audit = audit_generic_runtime_source(
        renderer_source,
        source_name=PUBLIC_LVLS1_RUNTIME_PATH,
        lineage_attested_generic=lineage_attested_generic,
    )
    placement = canonical_lvpg2_placement_manifest(
        archive_bytes=archive_bytes,
        member_sha256=_sha256(member),
        inverse_source_bytes=inverse_source,
        lvls1_runtime_source_bytes=renderer_source,
        offline_compiler_source_bytes=offline_source,
        source_population_sha256=source_population_sha256,
    )
    imported_roots = tuple(sorted(set(inverse_audit.imported_roots) | set(renderer_audit.imported_roots)))
    abi = capture_interpreter_distribution_abi(imported_roots)
    if not abi.closed:
        raise PublicAuthClosureError(
            f"public decoder third-party distribution ABI is unresolved: {abi.unresolved_import_roots}"
        )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    require_exact_public_runtime_tree(runtime_dir, complete=False)
    _write_runtime_file(runtime_dir / PUBLIC_INFLATE_SH_PATH, _INFLATE_SH_BYTES, executable=True)
    _write_runtime_file(runtime_dir / PUBLIC_INFLATE_PY_PATH, inverse_source, executable=True)
    _write_runtime_file(runtime_dir / PUBLIC_LVLS1_RUNTIME_PATH, renderer_source, executable=False)
    require_exact_public_runtime_tree(runtime_dir, complete=True)
    require_no_bytecode_contamination(runtime_dir)
    require_exact_public_runtime_tree(runtime_dir, complete=True)
    with tempfile.TemporaryDirectory(prefix="g29_parseback_", dir=runtime_dir.parent) as scratch_name:
        scratch = Path(scratch_name)
        source_member = scratch / MEMBER_NAME
        materialized = scratch / "logical.lvls1"
        _atomic_write(source_member, member)
        command = (
            sys.executable,
            os.fspath(runtime_dir / PUBLIC_INFLATE_PY_PATH),
            "--materialize-lvls1",
            os.fspath(source_member),
            os.fspath(materialized),
        )
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        completed = subprocess.run(command, env=environment, capture_output=True, check=False)
        if completed.returncode != 0:
            raise PublicAuthClosureError(
                "emitted public inverse failed real LVPG2 parseback: "
                + completed.stderr.decode("utf-8", errors="replace")[-2000:]
            )
        materialized_bytes = materialized.read_bytes()
        reopened = parse_ep725_lvls1(materialized_bytes, require_source_form=True)
        selected_state = _decoded_state_sha256(selected)
        reopened_state = _decoded_state_sha256(reopened)
        if selected_state != reopened_state:
            raise PublicAuthClosureError("emitted public inverse changed the full quantized state")
        command_sha256 = _sha256(_canonical_json(list(command)))
    require_no_bytecode_contamination(runtime_dir)
    file_rows = tuple(
        sorted(
            (
                PublicRuntimeFileDigestV1(
                    relative_path=path.name,
                    content_sha256=_sha256_file(path),
                    nbytes=path.stat().st_size,
                    executable=bool(path.stat().st_mode & stat.S_IXUSR),
                )
                for path in (
                    runtime_dir / PUBLIC_INFLATE_SH_PATH,
                    runtime_dir / PUBLIC_INFLATE_PY_PATH,
                    runtime_dir / PUBLIC_LVLS1_RUNTIME_PATH,
                )
            ),
            key=lambda item: item.relative_path,
        )
    )
    audits = tuple(sorted((inverse_audit, renderer_audit), key=lambda item: item.source_name))
    receipt = PublicRuntimeCompileReceiptV1(
        archive_sha256=_sha256(archive_bytes),
        archive_nbytes=len(archive_bytes),
        member_sha256=_sha256(member),
        member_nbytes=len(member),
        decoded_state_sha256=selected_state,
        materialized_lvls1_sha256=_sha256(materialized_bytes),
        runtime_tree_sha256=_runtime_tree_sha256(file_rows),
        runtime_files=file_rows,
        source_audit_receipt_sha256s=tuple(sorted(item.identity_sha256 for item in audits)),
        placement_identity_sha256=placement.identity_sha256,
        abi_identity_sha256=abi.identity_sha256,
        inverse_process_argv_sha256=command_sha256,
        parseback_full_state_equal=True,
        bytecode_contamination_paths=(),
        research_only=True,
        public_n600_output_equality_owed=True,
    )
    return CompiledPublicRuntimeV1(
        compile_receipt=receipt,
        placement=placement,
        abi_closure=abi,
        source_audits=audits,
    )


@dataclass(frozen=True, slots=True)
class DependencyFileReceiptV1:
    relative_path: str
    content_sha256: str
    nbytes: int
    custody_owner: str
    scope: G17RuntimeFileScopeV1

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, label="dependency file path")
        _require_sha256(self.content_sha256, label="dependency file")
        if type(self.nbytes) is not int or self.nbytes < 1:
            raise PublicAuthClosureError("dependency file bytes must be positive int")
        _require_ascii(self.custody_owner, label="dependency file custody owner")
        if type(self.scope) is not G17RuntimeFileScopeV1:
            raise PublicAuthClosureError("dependency file scope must be typed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_sha256": self.content_sha256,
            "custody_owner": self.custody_owner,
            "nbytes": self.nbytes,
            "relative_path": self.relative_path,
            "scope": self.scope.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            "content_sha256",
            "custody_owner",
            "nbytes",
            "relative_path",
            "scope",
        }:
            raise PublicAuthClosureError("dependency file receipt fields drifted")
        return cls(
            relative_path=value["relative_path"],
            content_sha256=value["content_sha256"],
            nbytes=value["nbytes"],
            custody_owner=value["custody_owner"],
            scope=G17RuntimeFileScopeV1(value["scope"]),
        )


@dataclass(frozen=True, slots=True)
class DependencyEdgeReceiptV1:
    importer_path: str
    dependency_path: str
    mechanism: G17RuntimeDependencyMechanismV1

    def __post_init__(self) -> None:
        _relative_path(self.importer_path, label="dependency importer")
        _relative_path(self.dependency_path, label="dependency target")
        if self.importer_path == self.dependency_path:
            raise PublicAuthClosureError("dependency edge cannot be self-referential")
        if type(self.mechanism) is not G17RuntimeDependencyMechanismV1:
            raise PublicAuthClosureError("dependency edge mechanism must be typed")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.importer_path, self.dependency_path, self.mechanism.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dependency_path": self.dependency_path,
            "importer_path": self.importer_path,
            "mechanism": self.mechanism.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {"dependency_path", "importer_path", "mechanism"}:
            raise PublicAuthClosureError("dependency edge receipt fields drifted")
        return cls(
            importer_path=value["importer_path"],
            dependency_path=value["dependency_path"],
            mechanism=G17RuntimeDependencyMechanismV1(value["mechanism"]),
        )


@dataclass(frozen=True, slots=True)
class SystemToolIdentityV1:
    role: str
    executable_name: str
    executable_realpath: str
    executable_sha256: str
    version_output_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.role, label="system tool role")
        _require_ascii(self.executable_name, label="system tool executable name")
        _require_ascii(self.executable_realpath, label="system tool executable realpath")
        _require_sha256(self.executable_sha256, label="system tool executable")
        _require_sha256(self.version_output_sha256, label="system tool version output")

    def to_dict(self) -> dict[str, Any]:
        return {
            "executable_name": self.executable_name,
            "executable_realpath": self.executable_realpath,
            "executable_sha256": self.executable_sha256,
            "role": self.role,
            "version_output_sha256": self.version_output_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            "executable_name",
            "executable_realpath",
            "executable_sha256",
            "role",
            "version_output_sha256",
        }:
            raise PublicAuthClosureError("system tool identity fields drifted")
        return cls(**value)


def _system_tool_identity(role: str, command: str, version_args: tuple[str, ...]) -> SystemToolIdentityV1:
    resolved = shutil.which(command)
    if resolved is None:
        raise PublicAuthClosureError(f"required public-path system tool is unavailable: {command}")
    executable = Path(resolved).resolve(strict=True)
    completed = subprocess.run(
        (os.fspath(executable), *version_args),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        check=False,
    )
    output = completed.stdout + b"\x00" + completed.stderr + b"\x00" + str(completed.returncode).encode("ascii")
    return SystemToolIdentityV1(
        role=role,
        executable_name=executable.name,
        executable_realpath=executable.as_posix(),
        executable_sha256=_sha256_file(executable),
        version_output_sha256=_sha256(output),
    )


@dataclass(frozen=True, slots=True)
class RuntimeDependencyDiscoveryReceiptV1:
    upstream_snapshot_sha256: str
    compile_receipt_sha256: str
    decoder_abi_identity_sha256: str
    evaluator_abi_closure_owed: bool
    runtime_files: tuple[DependencyFileReceiptV1, ...]
    authority_input_files: tuple[DependencyFileReceiptV1, ...]
    environment_lock_files: tuple[DependencyFileReceiptV1, ...]
    static_authority_input_file_manifest_sha256: str
    dependency_edges: tuple[DependencyEdgeReceiptV1, ...]
    observed_runtime_paths: tuple[str, ...]
    system_tools: tuple[SystemToolIdentityV1, ...]
    executed_public_entrypoint_path: str
    discovery_method: str
    bytecode_contamination_paths: tuple[str, ...]
    actual_evaluate_sh_graph: bool
    c0b_graph_compatible: bool
    schema: str = field(default=DEPENDENCY_DISCOVERY_SCHEMA, init=False)

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "decoder_abi_identity_sha256",
            "evaluator_abi_closure_owed",
            "actual_evaluate_sh_graph",
            "authority_input_files",
            "static_authority_input_file_manifest_sha256",
            "bytecode_contamination_paths",
            "c0b_graph_compatible",
            "compile_receipt_sha256",
            "dependency_edges",
            "discovery_method",
            "executed_public_entrypoint_path",
            "environment_lock_files",
            "observed_runtime_paths",
            "runtime_files",
            "schema",
            "system_tools",
            "upstream_snapshot_sha256",
        }
    )

    def __post_init__(self) -> None:
        for value, label in (
            (self.upstream_snapshot_sha256, "discovery upstream snapshot"),
            (self.compile_receipt_sha256, "discovery compile receipt"),
            (self.decoder_abi_identity_sha256, "discovery decoder ABI identity"),
        ):
            _require_sha256(value, label=label)
        if (
            type(self.runtime_files) is not tuple
            or any(type(item) is not DependencyFileReceiptV1 for item in self.runtime_files)
            or self.runtime_files != tuple(sorted(self.runtime_files, key=lambda item: item.relative_path))
            or len({item.relative_path for item in self.runtime_files}) != len(self.runtime_files)
        ):
            raise PublicAuthClosureError("discovery runtime files must be unique canonical typed tuple")
        for values, label in (
            (self.authority_input_files, "discovery authority inputs"),
            (self.environment_lock_files, "discovery environment locks"),
        ):
            if (
                type(values) is not tuple
                or not values
                or any(type(item) is not DependencyFileReceiptV1 for item in values)
                or values != tuple(sorted(values, key=lambda item: item.relative_path))
                or len({item.relative_path for item in values}) != len(values)
            ):
                raise PublicAuthClosureError(f"{label} must be a unique canonical typed tuple")
        required_inputs = {
            "upstream/models/posenet.safetensors",
            "upstream/models/segnet.safetensors",
            "upstream/public_test_video_names.txt",
            "upstream/videos/0.mkv",
        }
        if {item.relative_path for item in self.authority_input_files} != required_inputs:
            raise PublicAuthClosureError("discovery authority input closure is incomplete")
        _require_sha256(
            self.static_authority_input_file_manifest_sha256,
            label="discovery static authority input file manifest",
        )
        expected_input_manifest = _sha256(_canonical_json([item.to_dict() for item in self.authority_input_files]))
        if self.static_authority_input_file_manifest_sha256 != expected_input_manifest:
            raise PublicAuthClosureError(
                "discovery static authority input manifest does not derive from exact input files"
            )
        required_locks = {
            "upstream/.python-version",
            "upstream/pyproject.toml",
            "upstream/uv.lock",
        }
        if {item.relative_path for item in self.environment_lock_files} != required_locks:
            raise PublicAuthClosureError("discovery environment lock closure is incomplete")
        if (
            type(self.dependency_edges) is not tuple
            or any(type(item) is not DependencyEdgeReceiptV1 for item in self.dependency_edges)
            or self.dependency_edges != tuple(sorted(self.dependency_edges, key=lambda item: item.key))
            or len({item.key for item in self.dependency_edges}) != len(self.dependency_edges)
        ):
            raise PublicAuthClosureError("discovery dependency edges must be unique canonical typed tuple")
        file_paths = {item.relative_path for item in self.runtime_files}
        if any(
            edge.importer_path not in file_paths or edge.dependency_path not in file_paths
            for edge in self.dependency_edges
        ):
            raise PublicAuthClosureError("discovery dependency edge escapes byte-owned runtime files")
        expected_paths = tuple(sorted(file_paths))
        if self.observed_runtime_paths != expected_paths:
            raise PublicAuthClosureError("discovery observed paths must exactly cover owned runtime files")
        if (
            type(self.system_tools) is not tuple
            or any(type(item) is not SystemToolIdentityV1 for item in self.system_tools)
            or self.system_tools != tuple(sorted(self.system_tools, key=lambda item: item.role))
            or len({item.role for item in self.system_tools}) != len(self.system_tools)
        ):
            raise PublicAuthClosureError("discovery system tools must be unique canonical typed tuple")
        if {item.role for item in self.system_tools} != {
            "archive.unzip",
            "environment.uv",
            "filesystem.dirname",
            "filesystem.mkdir",
            "filesystem.rm",
            "interpreter.python",
            "shell.bash",
        }:
            raise PublicAuthClosureError("discovery system executable/environment-tool closure is incomplete")
        if self.executed_public_entrypoint_path != "upstream/evaluate.sh":
            raise PublicAuthClosureError("official public root must be upstream/evaluate.sh")
        _require_ascii(self.discovery_method, label="dependency discovery method")
        if self.bytecode_contamination_paths:
            raise PublicAuthClosureError("dependency discovery contains generated upstream/runtime bytecode")
        if self.actual_evaluate_sh_graph is not True or self.c0b_graph_compatible is not True:
            raise PublicAuthClosureError("dependency discovery must use the amended real evaluate.sh graph")
        false_edge = (
            PUBLIC_EVALUATOR_PATH,
            PUBLIC_INFLATE_SH_PATH,
            G17RuntimeDependencyMechanismV1.PROCESS_EXEC.value,
        )
        if false_edge in {edge.key for edge in self.dependency_edges}:
            raise PublicAuthClosureError("fabricated evaluate.py -> inflate.sh edge is forbidden")
        if self.evaluator_abi_closure_owed is not True:
            raise PublicAuthClosureError("static discovery cannot claim the axis-specific evaluator ABI")

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(
            {
                "decoder_abi_identity_sha256": self.decoder_abi_identity_sha256,
                "evaluator_abi_closure_owed": self.evaluator_abi_closure_owed,
                "actual_evaluate_sh_graph": self.actual_evaluate_sh_graph,
                "authority_input_files": [item.to_dict() for item in self.authority_input_files],
                "static_authority_input_file_manifest_sha256": (self.static_authority_input_file_manifest_sha256),
                "bytecode_contamination_paths": list(self.bytecode_contamination_paths),
                "c0b_graph_compatible": self.c0b_graph_compatible,
                "compile_receipt_sha256": self.compile_receipt_sha256,
                "dependency_edges": [item.to_dict() for item in self.dependency_edges],
                "discovery_method": self.discovery_method,
                "executed_public_entrypoint_path": self.executed_public_entrypoint_path,
                "environment_lock_files": [item.to_dict() for item in self.environment_lock_files],
                "observed_runtime_paths": list(self.observed_runtime_paths),
                "runtime_files": [item.to_dict() for item in self.runtime_files],
                "schema": self.schema,
                "system_tools": [item.to_dict() for item in self.system_tools],
                "upstream_snapshot_sha256": self.upstream_snapshot_sha256,
            }
        )

    @classmethod
    def from_receipt_bytes(cls, payload: bytes) -> Self:
        value = _parse_canonical_object(payload, exact_keys=cls._KEYS)
        if value["schema"] != DEPENDENCY_DISCOVERY_SCHEMA:
            raise PublicAuthClosureError("dependency discovery schema drifted")
        return cls(
            upstream_snapshot_sha256=value["upstream_snapshot_sha256"],
            compile_receipt_sha256=value["compile_receipt_sha256"],
            decoder_abi_identity_sha256=value["decoder_abi_identity_sha256"],
            evaluator_abi_closure_owed=value["evaluator_abi_closure_owed"],
            runtime_files=tuple(DependencyFileReceiptV1.from_dict(row) for row in value["runtime_files"]),
            authority_input_files=tuple(
                DependencyFileReceiptV1.from_dict(row) for row in value["authority_input_files"]
            ),
            static_authority_input_file_manifest_sha256=value["static_authority_input_file_manifest_sha256"],
            environment_lock_files=tuple(
                DependencyFileReceiptV1.from_dict(row) for row in value["environment_lock_files"]
            ),
            dependency_edges=tuple(DependencyEdgeReceiptV1.from_dict(row) for row in value["dependency_edges"]),
            observed_runtime_paths=tuple(value["observed_runtime_paths"]),
            system_tools=tuple(SystemToolIdentityV1.from_dict(row) for row in value["system_tools"]),
            executed_public_entrypoint_path=value["executed_public_entrypoint_path"],
            discovery_method=value["discovery_method"],
            bytecode_contamination_paths=tuple(value["bytecode_contamination_paths"]),
            actual_evaluate_sh_graph=value["actual_evaluate_sh_graph"],
            c0b_graph_compatible=value["c0b_graph_compatible"],
        )


def _dependency_file(
    *,
    path: Path,
    relative_path: str,
    custody_owner: str,
    scope: G17RuntimeFileScopeV1,
) -> DependencyFileReceiptV1:
    return DependencyFileReceiptV1(
        relative_path=relative_path,
        content_sha256=_sha256_file(path),
        nbytes=path.stat().st_size,
        custody_owner=custody_owner,
        scope=scope,
    )


def discover_public_runtime_dependencies(
    *,
    repo_root: Path,
    runtime_dir: Path,
    compiled: CompiledPublicRuntimeV1,
) -> RuntimeDependencyDiscoveryReceiptV1:
    """Bind the actual evaluate.sh-rooted public graph and all local source bytes."""

    from tac.contest_compliance import compute_upstream_snapshot_sha256

    repo_root = repo_root.resolve(strict=True)
    runtime_dir = runtime_dir.resolve(strict=True)
    upstream_dir = repo_root / "upstream"
    require_exact_public_runtime_tree(runtime_dir, complete=True)
    require_no_bytecode_contamination(upstream_dir, runtime_dir)
    upstream_snapshot = compute_upstream_snapshot_sha256(
        repo_root,
        reject_executable_artifacts=True,
    )
    if upstream_snapshot != EXPECTED_UPSTREAM_SNAPSHOT_SHA256:
        raise PublicAuthClosureError(
            f"frozen upstream snapshot does not match expected d46 authority pin: {upstream_snapshot!r}"
        )
    locations = {
        "upstream/evaluate.sh": (
            upstream_dir / "evaluate.sh",
            "frozen_upstream_snapshot",
            G17RuntimeFileScopeV1.EVALUATOR_PUBLIC_ENTRYPOINT,
        ),
        PUBLIC_EVALUATOR_PATH: (
            upstream_dir / "evaluate.py",
            "frozen_upstream_snapshot",
            G17RuntimeFileScopeV1.EVALUATOR_RUNTIME_DEPENDENCY,
        ),
        "upstream/frame_utils.py": (
            upstream_dir / "frame_utils.py",
            "frozen_upstream_snapshot",
            G17RuntimeFileScopeV1.EVALUATOR_RUNTIME_DEPENDENCY,
        ),
        "upstream/modules.py": (
            upstream_dir / "modules.py",
            "frozen_upstream_snapshot",
            G17RuntimeFileScopeV1.EVALUATOR_RUNTIME_DEPENDENCY,
        ),
        PUBLIC_INFLATE_SH_PATH: (
            runtime_dir / PUBLIC_INFLATE_SH_PATH,
            "compiled_submission_runtime",
            G17RuntimeFileScopeV1.SUBMISSION_PUBLIC_ENTRYPOINT,
        ),
        PUBLIC_INFLATE_PY_PATH: (
            runtime_dir / PUBLIC_INFLATE_PY_PATH,
            "compiled_submission_runtime",
            G17RuntimeFileScopeV1.SUBMISSION_RUNTIME_DEPENDENCY,
        ),
        PUBLIC_LVLS1_RUNTIME_PATH: (
            runtime_dir / PUBLIC_LVLS1_RUNTIME_PATH,
            "compiled_submission_runtime",
            G17RuntimeFileScopeV1.SUBMISSION_RUNTIME_DEPENDENCY,
        ),
    }
    if any(not row[0].is_file() for row in locations.values()):
        missing = sorted(name for name, row in locations.items() if not row[0].is_file())
        raise PublicAuthClosureError(f"public dependency files are missing: {missing}")
    files = tuple(
        sorted(
            (
                _dependency_file(
                    path=path,
                    relative_path=name,
                    custody_owner=owner,
                    scope=scope,
                )
                for name, (path, owner, scope) in locations.items()
            ),
            key=lambda item: item.relative_path,
        )
    )
    authority_inputs = tuple(
        sorted(
            (
                _dependency_file(
                    path=repo_root / relative,
                    relative_path=relative,
                    custody_owner="frozen_upstream_authority_input",
                    scope=G17RuntimeFileScopeV1.EVALUATOR_RUNTIME_DEPENDENCY,
                )
                for relative in (
                    "upstream/models/posenet.safetensors",
                    "upstream/models/segnet.safetensors",
                    "upstream/public_test_video_names.txt",
                    "upstream/videos/0.mkv",
                )
            ),
            key=lambda item: item.relative_path,
        )
    )
    environment_locks = tuple(
        sorted(
            (
                _dependency_file(
                    path=repo_root / relative,
                    relative_path=relative,
                    custody_owner="frozen_upstream_environment_lock",
                    scope=G17RuntimeFileScopeV1.EVALUATOR_RUNTIME_DEPENDENCY,
                )
                for relative in (
                    "upstream/.python-version",
                    "upstream/pyproject.toml",
                    "upstream/uv.lock",
                )
            ),
            key=lambda item: item.relative_path,
        )
    )
    expected_compile = {item.relative_path: item for item in compiled.compile_receipt.runtime_files}
    for name in (PUBLIC_INFLATE_SH_PATH, PUBLIC_INFLATE_PY_PATH, PUBLIC_LVLS1_RUNTIME_PATH):
        discovered = next(item for item in files if item.relative_path == name)
        if discovered.content_sha256 != expected_compile[name].content_sha256:
            raise PublicAuthClosureError(f"compiled runtime file drifted before dependency discovery: {name}")
    edges = tuple(
        sorted(
            (
                DependencyEdgeReceiptV1("inflate.py", "lvls1_runtime.py", G17RuntimeDependencyMechanismV1.DYNAMIC_LOAD),
                DependencyEdgeReceiptV1("inflate.sh", "inflate.py", G17RuntimeDependencyMechanismV1.PROCESS_EXEC),
                DependencyEdgeReceiptV1(
                    "upstream/evaluate.py",
                    "upstream/frame_utils.py",
                    G17RuntimeDependencyMechanismV1.PYTHON_IMPORT,
                ),
                DependencyEdgeReceiptV1(
                    "upstream/evaluate.py",
                    "upstream/modules.py",
                    G17RuntimeDependencyMechanismV1.PYTHON_IMPORT,
                ),
                DependencyEdgeReceiptV1(
                    "upstream/evaluate.sh",
                    "inflate.sh",
                    G17RuntimeDependencyMechanismV1.PROCESS_EXEC,
                ),
                DependencyEdgeReceiptV1(
                    "upstream/evaluate.sh",
                    "upstream/evaluate.py",
                    G17RuntimeDependencyMechanismV1.PROCESS_EXEC,
                ),
                DependencyEdgeReceiptV1(
                    "upstream/modules.py",
                    "upstream/frame_utils.py",
                    G17RuntimeDependencyMechanismV1.PYTHON_IMPORT,
                ),
            ),
            key=lambda item: item.key,
        )
    )
    tools = tuple(
        sorted(
            (
                _system_tool_identity("shell.bash", "bash", ("--version",)),
                _system_tool_identity("archive.unzip", "unzip", ("-v",)),
                _system_tool_identity("environment.uv", "uv", ("--version",)),
                _system_tool_identity("filesystem.dirname", "dirname", ("--version",)),
                _system_tool_identity("filesystem.mkdir", "mkdir", ("--version",)),
                _system_tool_identity("filesystem.rm", "rm", ("--version",)),
                _system_tool_identity("interpreter.python", sys.executable, ("--version",)),
            ),
            key=lambda item: item.role,
        )
    )
    require_no_bytecode_contamination(upstream_dir, runtime_dir)
    return RuntimeDependencyDiscoveryReceiptV1(
        upstream_snapshot_sha256=upstream_snapshot,
        compile_receipt_sha256=compiled.compile_receipt.identity_sha256,
        decoder_abi_identity_sha256=compiled.abi_closure.identity_sha256,
        evaluator_abi_closure_owed=True,
        runtime_files=files,
        authority_input_files=authority_inputs,
        environment_lock_files=environment_locks,
        static_authority_input_file_manifest_sha256=_sha256(
            _canonical_json([item.to_dict() for item in authority_inputs])
        ),
        dependency_edges=edges,
        observed_runtime_paths=tuple(item.relative_path for item in files),
        system_tools=tools,
        executed_public_entrypoint_path="upstream/evaluate.sh",
        discovery_method="STRICT_STATIC_RECURSION_PLUS_PUBLIC_PROCESS_TRACE_REQUIRED_AT_EXECUTION",
        bytecode_contamination_paths=(),
        actual_evaluate_sh_graph=True,
        c0b_graph_compatible=True,
    )


@dataclass(frozen=True, slots=True)
class ScorerInputBatchEntryV1:
    """Ordered hashes of one real evaluator batch before and after preprocessing."""

    batch_index: int
    pair_start_index: int
    pair_count: int
    gt_decoded_uint8_sha256: str
    candidate_decoded_uint8_sha256: str
    gt_posenet_preprocessed_fp32_sha256: str
    candidate_posenet_preprocessed_fp32_sha256: str
    gt_segnet_preprocessed_fp32_sha256: str
    candidate_segnet_preprocessed_fp32_sha256: str

    def __post_init__(self) -> None:
        if type(self.batch_index) is not int or not 0 <= self.batch_index < EXPECTED_EVALUATOR_BATCH_COUNT:
            raise PublicAuthClosureError("scorer input batch index is out of range")
        expected_start = self.batch_index * OFFICIAL_EVALUATOR_BATCH_SIZE
        expected_count = min(OFFICIAL_EVALUATOR_BATCH_SIZE, EXPECTED_N_PAIRS - expected_start)
        if self.pair_start_index != expected_start or self.pair_count != expected_count:
            raise PublicAuthClosureError("scorer input batch does not match canonical n600 ordering")
        for value, label in (
            (self.gt_decoded_uint8_sha256, "GT decoded uint8 scorer batch"),
            (self.candidate_decoded_uint8_sha256, "candidate decoded uint8 scorer batch"),
            (self.gt_posenet_preprocessed_fp32_sha256, "GT PoseNet preprocessed fp32 batch"),
            (
                self.candidate_posenet_preprocessed_fp32_sha256,
                "candidate PoseNet preprocessed fp32 batch",
            ),
            (self.gt_segnet_preprocessed_fp32_sha256, "GT SegNet preprocessed fp32 batch"),
            (
                self.candidate_segnet_preprocessed_fp32_sha256,
                "candidate SegNet preprocessed fp32 batch",
            ),
        ):
            _require_sha256(value, label=label)

    @property
    def decoded_uint8_nbytes(self) -> int:
        return self.pair_count * 2 * CAMERA_HEIGHT * CAMERA_WIDTH * 3

    @property
    def posenet_preprocessed_fp32_nbytes(self) -> int:
        return self.pair_count * 12 * SCORER_INPUT_HEIGHT * SCORER_INPUT_WIDTH * 4

    @property
    def segnet_preprocessed_fp32_nbytes(self) -> int:
        return self.pair_count * 3 * SCORER_INPUT_HEIGHT * SCORER_INPUT_WIDTH * 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_index": self.batch_index,
            "candidate_decoded_uint8_sha256": self.candidate_decoded_uint8_sha256,
            "candidate_posenet_preprocessed_fp32_sha256": (self.candidate_posenet_preprocessed_fp32_sha256),
            "candidate_segnet_preprocessed_fp32_sha256": (self.candidate_segnet_preprocessed_fp32_sha256),
            "decoded_uint8_nbytes": self.decoded_uint8_nbytes,
            "gt_decoded_uint8_sha256": self.gt_decoded_uint8_sha256,
            "gt_posenet_preprocessed_fp32_sha256": self.gt_posenet_preprocessed_fp32_sha256,
            "gt_segnet_preprocessed_fp32_sha256": self.gt_segnet_preprocessed_fp32_sha256,
            "pair_count": self.pair_count,
            "pair_start_index": self.pair_start_index,
            "posenet_preprocessed_fp32_nbytes": self.posenet_preprocessed_fp32_nbytes,
            "segnet_preprocessed_fp32_nbytes": self.segnet_preprocessed_fp32_nbytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        expected = {
            "batch_index",
            "candidate_decoded_uint8_sha256",
            "candidate_posenet_preprocessed_fp32_sha256",
            "candidate_segnet_preprocessed_fp32_sha256",
            "decoded_uint8_nbytes",
            "gt_decoded_uint8_sha256",
            "gt_posenet_preprocessed_fp32_sha256",
            "gt_segnet_preprocessed_fp32_sha256",
            "pair_count",
            "pair_start_index",
            "posenet_preprocessed_fp32_nbytes",
            "segnet_preprocessed_fp32_nbytes",
        }
        if type(value) is not dict or set(value) != expected:
            raise PublicAuthClosureError("scorer input batch entry fields drifted")
        entry = cls(
            batch_index=value["batch_index"],
            pair_start_index=value["pair_start_index"],
            pair_count=value["pair_count"],
            gt_decoded_uint8_sha256=value["gt_decoded_uint8_sha256"],
            candidate_decoded_uint8_sha256=value["candidate_decoded_uint8_sha256"],
            gt_posenet_preprocessed_fp32_sha256=value["gt_posenet_preprocessed_fp32_sha256"],
            candidate_posenet_preprocessed_fp32_sha256=value["candidate_posenet_preprocessed_fp32_sha256"],
            gt_segnet_preprocessed_fp32_sha256=value["gt_segnet_preprocessed_fp32_sha256"],
            candidate_segnet_preprocessed_fp32_sha256=value["candidate_segnet_preprocessed_fp32_sha256"],
        )
        if (
            value["decoded_uint8_nbytes"] != entry.decoded_uint8_nbytes
            or value["posenet_preprocessed_fp32_nbytes"] != entry.posenet_preprocessed_fp32_nbytes
            or value["segnet_preprocessed_fp32_nbytes"] != entry.segnet_preprocessed_fp32_nbytes
        ):
            raise PublicAuthClosureError("scorer input batch byte geometry drifted")
        return entry


@dataclass(frozen=True, slots=True)
class ScorerInputBatchLedgerV1:
    """All 38 ordered n600 scorer input batches observed in one official run."""

    archive_sha256: str
    execution_axis: ExecutionAxisV1
    candidate_raw_sha256: str
    upstream_snapshot_sha256: str
    static_authority_input_file_manifest_sha256: str
    capture_trace_sha256: str
    entries: tuple[ScorerInputBatchEntryV1, ...]
    capture_method: str
    observation_mirror_equivalence_receipt_ascii: str = field(repr=False)
    schema: str = field(default=SCORER_INPUT_BATCH_LEDGER_SCHEMA, init=False)

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "archive_sha256",
            "batch_size",
            "candidate_raw_sha256",
            "capture_method",
            "capture_trace_sha256",
            "entries",
            "execution_axis",
            "n_batches",
            "n_pairs",
            "observation_mirror_equivalence_receipt_ascii",
            "schema",
            "static_authority_input_file_manifest_sha256",
            "upstream_snapshot_sha256",
        }
    )

    def __post_init__(self) -> None:
        if type(self.execution_axis) is not ExecutionAxisV1:
            raise PublicAuthClosureError("scorer input ledger axis must be typed")
        for value, label in (
            (self.archive_sha256, "scorer input ledger archive"),
            (self.candidate_raw_sha256, "scorer input ledger candidate raw"),
            (self.upstream_snapshot_sha256, "scorer input ledger upstream snapshot"),
            (
                self.static_authority_input_file_manifest_sha256,
                "scorer input ledger static authority input manifest",
            ),
            (self.capture_trace_sha256, "scorer input ledger capture trace"),
        ):
            _require_sha256(value, label=label)
        if self.upstream_snapshot_sha256 != EXPECTED_UPSTREAM_SNAPSHOT_SHA256:
            raise PublicAuthClosureError("scorer input ledger upstream snapshot drifted")
        if (
            type(self.entries) is not tuple
            or len(self.entries) != EXPECTED_EVALUATOR_BATCH_COUNT
            or any(type(item) is not ScorerInputBatchEntryV1 for item in self.entries)
            or tuple(item.batch_index for item in self.entries) != tuple(range(EXPECTED_EVALUATOR_BATCH_COUNT))
            or sum(item.pair_count for item in self.entries) != EXPECTED_N_PAIRS
        ):
            raise PublicAuthClosureError("scorer input ledger is not the complete ordered 38-batch n600 ledger")
        if self.capture_method != "INSTRUMENTED_OBSERVATION_MIRROR_DISTORTIONNET_PREPROCESS_CAPTURE_V1":
            raise PublicAuthClosureError("scorer input ledger capture method is not the reviewed mirror hook")
        _require_ascii(
            self.observation_mirror_equivalence_receipt_ascii,
            label="embedded scorer input observation mirror equivalence",
        )
        mirror = ScorerOutputMirrorEquivalenceReceiptV1.from_receipt_bytes(
            self.observation_mirror_equivalence_receipt_ascii.encode("ascii")
        )
        if (
            mirror.archive_sha256 != self.archive_sha256
            or mirror.execution_axis is not self.execution_axis
            or mirror.candidate_raw_sha256 != self.candidate_raw_sha256
            or mirror.upstream_snapshot_sha256 != self.upstream_snapshot_sha256
            or mirror.static_authority_input_file_manifest_sha256 != self.static_authority_input_file_manifest_sha256
            or mirror.scorer_input_batch_content_sha256 != self.input_content_sha256
            or mirror.mirror_process_trace_sha256 != self.capture_trace_sha256
        ):
            raise PublicAuthClosureError("scorer input ledger drifted from observation mirror proof")

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())

    @property
    def input_content_sha256(self) -> str:
        """Run-independent identity of the ordered input tensors themselves."""

        return _sha256(
            _canonical_json(
                {
                    "archive_sha256": self.archive_sha256,
                    "candidate_raw_sha256": self.candidate_raw_sha256,
                    "entries": [item.to_dict() for item in self.entries],
                    "execution_axis": self.execution_axis.value,
                    "static_authority_input_file_manifest_sha256": (self.static_authority_input_file_manifest_sha256),
                    "upstream_snapshot_sha256": self.upstream_snapshot_sha256,
                }
            )
        )

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(
            {
                "archive_sha256": self.archive_sha256,
                "batch_size": OFFICIAL_EVALUATOR_BATCH_SIZE,
                "candidate_raw_sha256": self.candidate_raw_sha256,
                "capture_method": self.capture_method,
                "capture_trace_sha256": self.capture_trace_sha256,
                "entries": [item.to_dict() for item in self.entries],
                "execution_axis": self.execution_axis.value,
                "n_batches": EXPECTED_EVALUATOR_BATCH_COUNT,
                "n_pairs": EXPECTED_N_PAIRS,
                "observation_mirror_equivalence_receipt_ascii": (self.observation_mirror_equivalence_receipt_ascii),
                "schema": self.schema,
                "static_authority_input_file_manifest_sha256": (self.static_authority_input_file_manifest_sha256),
                "upstream_snapshot_sha256": self.upstream_snapshot_sha256,
            }
        )

    @classmethod
    def from_receipt_bytes(cls, payload: bytes) -> Self:
        value = _parse_canonical_object(payload, exact_keys=cls._KEYS)
        if value["schema"] != SCORER_INPUT_BATCH_LEDGER_SCHEMA:
            raise PublicAuthClosureError("scorer input batch ledger schema drifted")
        if (
            value["batch_size"] != OFFICIAL_EVALUATOR_BATCH_SIZE
            or value["n_batches"] != EXPECTED_EVALUATOR_BATCH_COUNT
            or value["n_pairs"] != EXPECTED_N_PAIRS
        ):
            raise PublicAuthClosureError("scorer input batch ledger geometry drifted")
        return cls(
            archive_sha256=value["archive_sha256"],
            execution_axis=ExecutionAxisV1(value["execution_axis"]),
            candidate_raw_sha256=value["candidate_raw_sha256"],
            upstream_snapshot_sha256=value["upstream_snapshot_sha256"],
            static_authority_input_file_manifest_sha256=value["static_authority_input_file_manifest_sha256"],
            capture_trace_sha256=value["capture_trace_sha256"],
            entries=tuple(ScorerInputBatchEntryV1.from_dict(row) for row in value["entries"]),
            capture_method=value["capture_method"],
            observation_mirror_equivalence_receipt_ascii=value["observation_mirror_equivalence_receipt_ascii"],
        )


@dataclass(frozen=True, slots=True)
class ScorerOutputMirrorEquivalenceReceiptV1:
    """Evidence that an instrumented observation mirror did not become the authority run.

    The frozen, unmodified upstream run remains the only score authority.  The
    mirror may expose evaluator cells only when its exact raw inputs and report
    bytes agree with that run and a reviewed observation-only patch is retained.
    """

    run_label: str
    execution_axis: ExecutionAxisV1
    archive_sha256: str
    candidate_raw_sha256: str
    upstream_snapshot_sha256: str
    static_authority_input_file_manifest_sha256: str
    scorer_input_batch_content_sha256: str
    official_process_trace_sha256: str
    official_report_sha256: str
    mirror_process_trace_sha256: str
    mirror_source_sha256: str
    reviewed_observation_patch_sha256: str
    scorer_output_cell_rows_sha256: str
    exact_raw_inputs_equal: bool
    exact_preprocessed_inputs_equal: bool
    exact_report_bytes_equal: bool
    scorer_execution_unmodified_except_observation_serialization: bool
    instrumented_mirror_not_official_authority: bool
    capture_method: str
    schema: str = field(default=SCORER_OUTPUT_MIRROR_EQUIVALENCE_SCHEMA, init=False)

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "archive_sha256",
            "candidate_raw_sha256",
            "capture_method",
            "exact_preprocessed_inputs_equal",
            "exact_raw_inputs_equal",
            "exact_report_bytes_equal",
            "execution_axis",
            "instrumented_mirror_not_official_authority",
            "mirror_process_trace_sha256",
            "mirror_source_sha256",
            "official_process_trace_sha256",
            "official_report_sha256",
            "reviewed_observation_patch_sha256",
            "run_label",
            "schema",
            "scorer_execution_unmodified_except_observation_serialization",
            "scorer_input_batch_content_sha256",
            "scorer_output_cell_rows_sha256",
            "static_authority_input_file_manifest_sha256",
            "upstream_snapshot_sha256",
        }
    )

    def __post_init__(self) -> None:
        if self.run_label not in {"A", "B"}:
            raise PublicAuthClosureError("scorer mirror run label must be A or B")
        if type(self.execution_axis) is not ExecutionAxisV1:
            raise PublicAuthClosureError("scorer mirror axis must be typed")
        for value, label in (
            (self.archive_sha256, "scorer mirror archive"),
            (self.candidate_raw_sha256, "scorer mirror candidate raw"),
            (self.upstream_snapshot_sha256, "scorer mirror upstream"),
            (
                self.static_authority_input_file_manifest_sha256,
                "scorer mirror static input manifest",
            ),
            (self.scorer_input_batch_content_sha256, "scorer mirror input content"),
            (self.official_process_trace_sha256, "scorer mirror official process trace"),
            (self.official_report_sha256, "scorer mirror official report"),
            (self.mirror_process_trace_sha256, "scorer mirror process trace"),
            (self.mirror_source_sha256, "scorer mirror source"),
            (self.reviewed_observation_patch_sha256, "scorer mirror reviewed patch"),
            (self.scorer_output_cell_rows_sha256, "scorer mirror output cell rows"),
        ):
            _require_sha256(value, label=label)
        if self.upstream_snapshot_sha256 != EXPECTED_UPSTREAM_SNAPSHOT_SHA256:
            raise PublicAuthClosureError("scorer mirror upstream snapshot drifted")
        if (
            self.exact_raw_inputs_equal is not True
            or self.exact_preprocessed_inputs_equal is not True
            or self.exact_report_bytes_equal is not True
            or self.scorer_execution_unmodified_except_observation_serialization is not True
            or self.instrumented_mirror_not_official_authority is not True
        ):
            raise PublicAuthClosureError("scorer mirror lacks exact observation-only equivalence")
        if self.capture_method != "REVIEWED_INSTRUMENTED_OBSERVATION_MIRROR_V1":
            raise PublicAuthClosureError("scorer mirror capture method is not reviewed")

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(
            {
                "archive_sha256": self.archive_sha256,
                "candidate_raw_sha256": self.candidate_raw_sha256,
                "capture_method": self.capture_method,
                "exact_preprocessed_inputs_equal": self.exact_preprocessed_inputs_equal,
                "exact_raw_inputs_equal": self.exact_raw_inputs_equal,
                "exact_report_bytes_equal": self.exact_report_bytes_equal,
                "execution_axis": self.execution_axis.value,
                "instrumented_mirror_not_official_authority": (self.instrumented_mirror_not_official_authority),
                "mirror_process_trace_sha256": self.mirror_process_trace_sha256,
                "mirror_source_sha256": self.mirror_source_sha256,
                "official_process_trace_sha256": self.official_process_trace_sha256,
                "official_report_sha256": self.official_report_sha256,
                "reviewed_observation_patch_sha256": self.reviewed_observation_patch_sha256,
                "run_label": self.run_label,
                "schema": self.schema,
                "scorer_execution_unmodified_except_observation_serialization": (
                    self.scorer_execution_unmodified_except_observation_serialization
                ),
                "scorer_input_batch_content_sha256": self.scorer_input_batch_content_sha256,
                "scorer_output_cell_rows_sha256": self.scorer_output_cell_rows_sha256,
                "static_authority_input_file_manifest_sha256": (self.static_authority_input_file_manifest_sha256),
                "upstream_snapshot_sha256": self.upstream_snapshot_sha256,
            }
        )

    @classmethod
    def from_receipt_bytes(cls, payload: bytes) -> Self:
        value = _parse_canonical_object(payload, exact_keys=cls._KEYS)
        if value["schema"] != SCORER_OUTPUT_MIRROR_EQUIVALENCE_SCHEMA:
            raise PublicAuthClosureError("scorer output mirror equivalence schema drifted")
        return cls(
            run_label=value["run_label"],
            execution_axis=ExecutionAxisV1(value["execution_axis"]),
            archive_sha256=value["archive_sha256"],
            candidate_raw_sha256=value["candidate_raw_sha256"],
            upstream_snapshot_sha256=value["upstream_snapshot_sha256"],
            static_authority_input_file_manifest_sha256=value["static_authority_input_file_manifest_sha256"],
            scorer_input_batch_content_sha256=value["scorer_input_batch_content_sha256"],
            official_process_trace_sha256=value["official_process_trace_sha256"],
            official_report_sha256=value["official_report_sha256"],
            mirror_process_trace_sha256=value["mirror_process_trace_sha256"],
            mirror_source_sha256=value["mirror_source_sha256"],
            reviewed_observation_patch_sha256=value["reviewed_observation_patch_sha256"],
            scorer_output_cell_rows_sha256=value["scorer_output_cell_rows_sha256"],
            exact_raw_inputs_equal=value["exact_raw_inputs_equal"],
            exact_preprocessed_inputs_equal=value["exact_preprocessed_inputs_equal"],
            exact_report_bytes_equal=value["exact_report_bytes_equal"],
            scorer_execution_unmodified_except_observation_serialization=value[
                "scorer_execution_unmodified_except_observation_serialization"
            ],
            instrumented_mirror_not_official_authority=value["instrumented_mirror_not_official_authority"],
            capture_method=value["capture_method"],
        )


@dataclass(frozen=True, slots=True)
class ScorerOutputCellEntryV1:
    """Exact evaluator-owned target/candidate cells for one ordered pair."""

    pair_index: int
    target_seg_argmax_u8_sha256: str
    candidate_seg_argmax_u8_sha256: str
    seg_mismatch_pixels: int
    seg_dist_fp32_hex: str
    target_pose6_fp32_sha256: str
    candidate_pose6_fp32_sha256: str
    pose_mse_fp32_hex: str

    def __post_init__(self) -> None:
        if type(self.pair_index) is not int or not 0 <= self.pair_index < EXPECTED_N_PAIRS:
            raise PublicAuthClosureError("scorer output cell pair index is out of range")
        for value, label in (
            (self.target_seg_argmax_u8_sha256, "target SegNet argmax cell"),
            (self.candidate_seg_argmax_u8_sha256, "candidate SegNet argmax cell"),
            (self.target_pose6_fp32_sha256, "target PoseNet pose6 cell"),
            (self.candidate_pose6_fp32_sha256, "candidate PoseNet pose6 cell"),
        ):
            _require_sha256(value, label=label)
        seg_pixels = SCORER_INPUT_HEIGHT * SCORER_INPUT_WIDTH
        if type(self.seg_mismatch_pixels) is not int or not 0 <= self.seg_mismatch_pixels <= seg_pixels:
            raise PublicAuthClosureError("scorer output SegNet mismatch count is out of range")
        if type(self.seg_dist_fp32_hex) is not str or re.fullmatch(r"[0-9a-f]{8}", self.seg_dist_fp32_hex) is None:
            raise PublicAuthClosureError("scorer output SegNet distance must retain exact little-endian fp32 bytes")
        expected_seg_hex = struct.pack("<f", self.seg_mismatch_pixels / seg_pixels).hex()
        if self.seg_dist_fp32_hex != expected_seg_hex:
            raise PublicAuthClosureError("scorer output SegNet fp32 cell disagrees with exact mismatch count")
        if type(self.pose_mse_fp32_hex) is not str or re.fullmatch(r"[0-9a-f]{8}", self.pose_mse_fp32_hex) is None:
            raise PublicAuthClosureError("scorer output pose MSE must retain exact little-endian fp32 bytes")
        (pose_mse,) = struct.unpack("<f", bytes.fromhex(self.pose_mse_fp32_hex))
        _require_finite_nonnegative(pose_mse, label="scorer output pose MSE")

    @property
    def batch_index(self) -> int:
        return self.pair_index // OFFICIAL_EVALUATOR_BATCH_SIZE

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_index": self.batch_index,
            "candidate_pose6_fp32_nbytes": 6 * 4,
            "candidate_pose6_fp32_sha256": self.candidate_pose6_fp32_sha256,
            "candidate_seg_argmax_u8_nbytes": SCORER_INPUT_HEIGHT * SCORER_INPUT_WIDTH,
            "candidate_seg_argmax_u8_sha256": self.candidate_seg_argmax_u8_sha256,
            "pair_index": self.pair_index,
            "pose_mse_fp32_hex": self.pose_mse_fp32_hex,
            "seg_dist_fp32_hex": self.seg_dist_fp32_hex,
            "seg_mismatch_pixels": self.seg_mismatch_pixels,
            "target_pose6_fp32_nbytes": 6 * 4,
            "target_pose6_fp32_sha256": self.target_pose6_fp32_sha256,
            "target_seg_argmax_u8_nbytes": SCORER_INPUT_HEIGHT * SCORER_INPUT_WIDTH,
            "target_seg_argmax_u8_sha256": self.target_seg_argmax_u8_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        expected = {
            "batch_index",
            "candidate_pose6_fp32_nbytes",
            "candidate_pose6_fp32_sha256",
            "candidate_seg_argmax_u8_nbytes",
            "candidate_seg_argmax_u8_sha256",
            "pair_index",
            "pose_mse_fp32_hex",
            "seg_dist_fp32_hex",
            "seg_mismatch_pixels",
            "target_pose6_fp32_nbytes",
            "target_pose6_fp32_sha256",
            "target_seg_argmax_u8_nbytes",
            "target_seg_argmax_u8_sha256",
        }
        if type(value) is not dict or set(value) != expected:
            raise PublicAuthClosureError("scorer output cell entry fields drifted")
        entry = cls(
            pair_index=value["pair_index"],
            target_seg_argmax_u8_sha256=value["target_seg_argmax_u8_sha256"],
            candidate_seg_argmax_u8_sha256=value["candidate_seg_argmax_u8_sha256"],
            seg_mismatch_pixels=value["seg_mismatch_pixels"],
            seg_dist_fp32_hex=value["seg_dist_fp32_hex"],
            target_pose6_fp32_sha256=value["target_pose6_fp32_sha256"],
            candidate_pose6_fp32_sha256=value["candidate_pose6_fp32_sha256"],
            pose_mse_fp32_hex=value["pose_mse_fp32_hex"],
        )
        if (
            value["batch_index"] != entry.batch_index
            or value["candidate_pose6_fp32_nbytes"] != 24
            or value["target_pose6_fp32_nbytes"] != 24
            or value["candidate_seg_argmax_u8_nbytes"] != SCORER_INPUT_HEIGHT * SCORER_INPUT_WIDTH
            or value["target_seg_argmax_u8_nbytes"] != SCORER_INPUT_HEIGHT * SCORER_INPUT_WIDTH
        ):
            raise PublicAuthClosureError("scorer output cell byte geometry drifted")
        return entry


@dataclass(frozen=True, slots=True)
class ScorerOutputCellLedgerV1:
    archive_sha256: str
    execution_axis: ExecutionAxisV1
    candidate_raw_sha256: str
    upstream_snapshot_sha256: str
    static_authority_input_file_manifest_sha256: str
    scorer_input_batch_content_sha256: str
    capture_trace_sha256: str
    entries: tuple[ScorerOutputCellEntryV1, ...]
    capture_method: str
    evaluator_target_cells_evidence_only_not_payload: bool
    observation_mirror_equivalence_receipt_ascii: str = field(repr=False)
    schema: str = field(default=SCORER_OUTPUT_CELL_LEDGER_SCHEMA, init=False)

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "archive_sha256",
            "candidate_raw_sha256",
            "capture_method",
            "capture_trace_sha256",
            "entries",
            "evaluator_target_cells_evidence_only_not_payload",
            "execution_axis",
            "n_pairs",
            "observation_mirror_equivalence_receipt_ascii",
            "schema",
            "scorer_input_batch_content_sha256",
            "static_authority_input_file_manifest_sha256",
            "upstream_snapshot_sha256",
        }
    )

    def __post_init__(self) -> None:
        if type(self.execution_axis) is not ExecutionAxisV1:
            raise PublicAuthClosureError("scorer output cell ledger axis must be typed")
        for value, label in (
            (self.archive_sha256, "scorer output ledger archive"),
            (self.candidate_raw_sha256, "scorer output ledger candidate raw"),
            (self.upstream_snapshot_sha256, "scorer output ledger upstream"),
            (
                self.static_authority_input_file_manifest_sha256,
                "scorer output ledger static authority input manifest",
            ),
            (self.scorer_input_batch_content_sha256, "scorer output ledger input content"),
            (self.capture_trace_sha256, "scorer output ledger capture trace"),
        ):
            _require_sha256(value, label=label)
        if self.upstream_snapshot_sha256 != EXPECTED_UPSTREAM_SNAPSHOT_SHA256:
            raise PublicAuthClosureError("scorer output ledger upstream snapshot drifted")
        if (
            type(self.entries) is not tuple
            or len(self.entries) != EXPECTED_N_PAIRS
            or any(type(item) is not ScorerOutputCellEntryV1 for item in self.entries)
            or tuple(item.pair_index for item in self.entries) != tuple(range(EXPECTED_N_PAIRS))
        ):
            raise PublicAuthClosureError("scorer output ledger is not the complete ordered n600 cell ledger")
        if self.capture_method != "INSTRUMENTED_OBSERVATION_MIRROR_DISTORTIONNET_OUTPUT_CAPTURE_V1":
            raise PublicAuthClosureError("scorer output ledger capture method is not the reviewed mirror hook")
        if self.evaluator_target_cells_evidence_only_not_payload is not True:
            raise PublicAuthClosureError("evaluator target cells may exist only as authority evidence")
        _require_ascii(
            self.observation_mirror_equivalence_receipt_ascii,
            label="embedded scorer observation mirror equivalence",
        )
        mirror = ScorerOutputMirrorEquivalenceReceiptV1.from_receipt_bytes(
            self.observation_mirror_equivalence_receipt_ascii.encode("ascii")
        )
        if (
            mirror.archive_sha256 != self.archive_sha256
            or mirror.execution_axis is not self.execution_axis
            or mirror.candidate_raw_sha256 != self.candidate_raw_sha256
            or mirror.upstream_snapshot_sha256 != self.upstream_snapshot_sha256
            or mirror.static_authority_input_file_manifest_sha256 != self.static_authority_input_file_manifest_sha256
            or mirror.scorer_input_batch_content_sha256 != self.scorer_input_batch_content_sha256
            or mirror.mirror_process_trace_sha256 != self.capture_trace_sha256
            or mirror.scorer_output_cell_rows_sha256 != self.cell_rows_sha256
        ):
            raise PublicAuthClosureError("scorer output ledger drifted from observation mirror proof")

    @property
    def cell_rows_sha256(self) -> str:
        return _sha256(_canonical_json([item.to_dict() for item in self.entries]))

    @property
    def candidate_cell_content_sha256(self) -> str:
        """Pure functional-quotient identity of ordered candidate scorer cells."""

        return _sha256(
            _canonical_json(
                {
                    "candidate_cells": [
                        {
                            "candidate_pose6_fp32_sha256": item.candidate_pose6_fp32_sha256,
                            "candidate_seg_argmax_u8_sha256": item.candidate_seg_argmax_u8_sha256,
                            "pair_index": item.pair_index,
                        }
                        for item in self.entries
                    ],
                    "semantic_domain": "TAC_SEG_ARGMAX_U8_PLUS_POSE6_FP32_N600_V1",
                }
            )
        )

    @property
    def content_sha256(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "archive_sha256": self.archive_sha256,
                    "candidate_raw_sha256": self.candidate_raw_sha256,
                    "entries": [item.to_dict() for item in self.entries],
                    "execution_axis": self.execution_axis.value,
                    "scorer_input_batch_content_sha256": self.scorer_input_batch_content_sha256,
                    "static_authority_input_file_manifest_sha256": (self.static_authority_input_file_manifest_sha256),
                    "upstream_snapshot_sha256": self.upstream_snapshot_sha256,
                }
            )
        )

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(
            {
                "archive_sha256": self.archive_sha256,
                "candidate_raw_sha256": self.candidate_raw_sha256,
                "capture_method": self.capture_method,
                "capture_trace_sha256": self.capture_trace_sha256,
                "entries": [item.to_dict() for item in self.entries],
                "evaluator_target_cells_evidence_only_not_payload": (
                    self.evaluator_target_cells_evidence_only_not_payload
                ),
                "execution_axis": self.execution_axis.value,
                "n_pairs": EXPECTED_N_PAIRS,
                "observation_mirror_equivalence_receipt_ascii": (self.observation_mirror_equivalence_receipt_ascii),
                "schema": self.schema,
                "scorer_input_batch_content_sha256": self.scorer_input_batch_content_sha256,
                "static_authority_input_file_manifest_sha256": (self.static_authority_input_file_manifest_sha256),
                "upstream_snapshot_sha256": self.upstream_snapshot_sha256,
            }
        )

    @classmethod
    def from_receipt_bytes(cls, payload: bytes) -> Self:
        value = _parse_canonical_object(payload, exact_keys=cls._KEYS)
        if value["schema"] != SCORER_OUTPUT_CELL_LEDGER_SCHEMA or value["n_pairs"] != EXPECTED_N_PAIRS:
            raise PublicAuthClosureError("scorer output cell ledger schema/geometry drifted")
        return cls(
            archive_sha256=value["archive_sha256"],
            execution_axis=ExecutionAxisV1(value["execution_axis"]),
            candidate_raw_sha256=value["candidate_raw_sha256"],
            upstream_snapshot_sha256=value["upstream_snapshot_sha256"],
            static_authority_input_file_manifest_sha256=value["static_authority_input_file_manifest_sha256"],
            scorer_input_batch_content_sha256=value["scorer_input_batch_content_sha256"],
            capture_trace_sha256=value["capture_trace_sha256"],
            entries=tuple(ScorerOutputCellEntryV1.from_dict(row) for row in value["entries"]),
            capture_method=value["capture_method"],
            evaluator_target_cells_evidence_only_not_payload=value["evaluator_target_cells_evidence_only_not_payload"],
            observation_mirror_equivalence_receipt_ascii=value["observation_mirror_equivalence_receipt_ascii"],
        )


@dataclass(frozen=True, slots=True)
class PublicInverseTraceReceiptV1:
    """Strict retained trace emitted by the real public inverse process."""

    argv: tuple[str, ...]
    decoder_phase_scope: str
    dependency_paths: tuple[str, ...]
    decode_wall_seconds: float
    entropy_denied: bool
    external_reads_denied: bool
    filesystem_policy: str
    inverse_source_sha256: str
    logical_lvls1_sha256: str
    network_denied: bool
    output_raw_bytes: int
    output_raw_sha256: str
    pythondontwritebytecode: str
    runtime_sha256: str
    source_member_name: str
    source_member_sha256: str
    subprocess_denied: bool
    schema: str = field(default=PUBLIC_INVERSE_TRACE_SCHEMA, init=False)

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "argv",
            "decoder_phase_scope",
            "dependency_paths",
            "decode_wall_seconds",
            "entropy_denied",
            "external_reads_denied",
            "filesystem_policy",
            "inverse_source_sha256",
            "logical_lvls1_sha256",
            "network_denied",
            "output_raw_bytes",
            "output_raw_sha256",
            "pythondontwritebytecode",
            "runtime_sha256",
            "schema",
            "source_member_name",
            "source_member_sha256",
            "subprocess_denied",
        }
    )

    def __post_init__(self) -> None:
        if (
            type(self.argv) is not tuple
            or len(self.argv) != 3
            or any(type(value) is not str or not value.isascii() or not value for value in self.argv)
        ):
            raise PublicAuthClosureError("public inverse trace argv is not the exact 3-argument invocation")
        if self.decoder_phase_scope != "LVPG2_INVERSE_PLUS_GENERIC_LVLS1_RENDERER_ONLY":
            raise PublicAuthClosureError("public inverse trace phase scope drifted")
        if self.dependency_paths != ("inflate.py", "lvls1_runtime.py"):
            raise PublicAuthClosureError("public inverse trace dependency path set drifted")
        _require_finite_nonnegative(self.decode_wall_seconds, label="public inverse decode wall time")
        if self.decode_wall_seconds <= 0.0:
            raise PublicAuthClosureError("public inverse trace decode wall time must be positive")
        if (
            self.entropy_denied is not True
            or self.external_reads_denied is not True
            or self.network_denied is not True
            or self.subprocess_denied is not True
        ):
            raise PublicAuthClosureError("public inverse trace lacks decoder-only deny-policy closure")
        if self.filesystem_policy != "PYTHON_AUDIT_DENY_NETWORK_EXEC_ENTROPY_AND_EXTERNAL_FILE_READS_V2":
            raise PublicAuthClosureError("public inverse filesystem policy drifted")
        for value, label in (
            (self.inverse_source_sha256, "public inverse source"),
            (self.logical_lvls1_sha256, "public inverse logical LVLS1"),
            (self.output_raw_sha256, "public inverse raw output"),
            (self.runtime_sha256, "public inverse LVLS1 runtime"),
            (self.source_member_sha256, "public inverse source member"),
        ):
            _require_sha256(value, label=label)
        if self.output_raw_bytes != EXPECTED_RAW_NBYTES:
            raise PublicAuthClosureError("public inverse trace raw byte contract drifted")
        if self.pythondontwritebytecode != "1":
            raise PublicAuthClosureError("public inverse trace did not disable Python bytecode")
        if self.source_member_name != MEMBER_NAME:
            raise PublicAuthClosureError("public inverse trace source member is not exact 0.bin")

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(
            {
                "argv": list(self.argv),
                "decoder_phase_scope": self.decoder_phase_scope,
                "dependency_paths": list(self.dependency_paths),
                "decode_wall_seconds": self.decode_wall_seconds,
                "entropy_denied": self.entropy_denied,
                "external_reads_denied": self.external_reads_denied,
                "filesystem_policy": self.filesystem_policy,
                "inverse_source_sha256": self.inverse_source_sha256,
                "logical_lvls1_sha256": self.logical_lvls1_sha256,
                "network_denied": self.network_denied,
                "output_raw_bytes": self.output_raw_bytes,
                "output_raw_sha256": self.output_raw_sha256,
                "pythondontwritebytecode": self.pythondontwritebytecode,
                "runtime_sha256": self.runtime_sha256,
                "schema": self.schema,
                "source_member_name": self.source_member_name,
                "source_member_sha256": self.source_member_sha256,
                "subprocess_denied": self.subprocess_denied,
            }
        )

    @classmethod
    def from_receipt_bytes(cls, payload: bytes) -> Self:
        value = _parse_canonical_object(payload, exact_keys=cls._KEYS)
        if value["schema"] != PUBLIC_INVERSE_TRACE_SCHEMA:
            raise PublicAuthClosureError("public inverse trace schema drifted")
        return cls(
            argv=tuple(value["argv"]),
            decoder_phase_scope=value["decoder_phase_scope"],
            dependency_paths=tuple(value["dependency_paths"]),
            decode_wall_seconds=value["decode_wall_seconds"],
            entropy_denied=value["entropy_denied"],
            external_reads_denied=value["external_reads_denied"],
            filesystem_policy=value["filesystem_policy"],
            inverse_source_sha256=value["inverse_source_sha256"],
            logical_lvls1_sha256=value["logical_lvls1_sha256"],
            network_denied=value["network_denied"],
            output_raw_bytes=value["output_raw_bytes"],
            output_raw_sha256=value["output_raw_sha256"],
            pythondontwritebytecode=value["pythondontwritebytecode"],
            runtime_sha256=value["runtime_sha256"],
            source_member_name=value["source_member_name"],
            source_member_sha256=value["source_member_sha256"],
            subprocess_denied=value["subprocess_denied"],
        )


@dataclass(frozen=True, slots=True)
class OfficialWorkflowJobReceiptV1:
    """Whole GitHub ``test`` job timing/custody, distinct from evaluate.sh timing."""

    run_label: str
    execution_axis: ExecutionAxisV1
    runner_label: str
    workflow_relative_path: str
    workflow_sha256: str
    workflow_job_name: str
    workflow_run_receipt_id: str
    job_log_sha256: str
    upstream_snapshot_sha256: str
    archive_sha256: str
    report_sha256: str
    evaluate_step_process_trace_sha256: str
    job_wall_seconds: float
    timeout_minutes: int
    timing_source: str
    all_setup_and_evaluate_steps_included: bool
    job_outcome_success: bool
    external_governed_custody_verified: bool
    whole_job_graph_closure_owed: bool
    research_only: bool
    schema: str = field(default=OFFICIAL_WORKFLOW_JOB_SCHEMA, init=False)

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "all_setup_and_evaluate_steps_included",
            "archive_sha256",
            "evaluate_step_process_trace_sha256",
            "execution_axis",
            "external_governed_custody_verified",
            "job_log_sha256",
            "job_outcome_success",
            "job_wall_seconds",
            "research_only",
            "report_sha256",
            "run_label",
            "runner_label",
            "schema",
            "timeout_minutes",
            "timing_source",
            "upstream_snapshot_sha256",
            "workflow_job_name",
            "workflow_relative_path",
            "workflow_run_receipt_id",
            "workflow_sha256",
            "whole_job_graph_closure_owed",
        }
    )

    def __post_init__(self) -> None:
        if self.run_label not in {"A", "B"}:
            raise PublicAuthClosureError("official workflow run label must be A or B")
        if type(self.execution_axis) is not ExecutionAxisV1:
            raise PublicAuthClosureError("official workflow axis must be typed")
        expected_runner = {
            ExecutionAxisV1.CPU: "ubuntu-latest",
            ExecutionAxisV1.CUDA: "linux-nvidia-t4",
        }[self.execution_axis]
        if self.runner_label != expected_runner:
            raise PublicAuthClosureError("official workflow runner label disagrees with axis")
        if self.workflow_relative_path != "upstream/.github/workflows/eval.yml":
            raise PublicAuthClosureError("official workflow path drifted")
        if self.workflow_job_name != "test":
            raise PublicAuthClosureError("official workflow authority must cover the complete test job")
        _require_identifier(self.workflow_run_receipt_id, label="official workflow run receipt ID")
        for value, label in (
            (self.workflow_sha256, "official workflow source"),
            (self.job_log_sha256, "official workflow job log"),
            (self.upstream_snapshot_sha256, "official workflow upstream"),
            (self.archive_sha256, "official workflow archive"),
            (self.report_sha256, "official workflow report"),
            (self.evaluate_step_process_trace_sha256, "official workflow evaluate-step trace"),
        ):
            _require_sha256(value, label=label)
        if self.workflow_sha256 != EXPECTED_OFFICIAL_WORKFLOW_SHA256:
            raise PublicAuthClosureError("official workflow source is not the frozen eval.yml")
        if self.upstream_snapshot_sha256 != EXPECTED_UPSTREAM_SNAPSHOT_SHA256:
            raise PublicAuthClosureError("official workflow upstream snapshot drifted")
        _require_finite_nonnegative(self.job_wall_seconds, label="official whole-job wall time")
        if not 0.0 < self.job_wall_seconds <= OFFICIAL_TOTAL_WORKFLOW_SECONDS:
            raise PublicAuthClosureError("official whole test job exceeded the 30-minute timeout")
        if self.timeout_minutes != 30:
            raise PublicAuthClosureError("official workflow timeout-minutes drifted")
        if self.timing_source != "GITHUB_ACTIONS_TEST_JOB_STARTED_COMPLETED_AND_OUTCOME_V1":
            raise PublicAuthClosureError("official workflow timing source is not reviewed")
        if self.all_setup_and_evaluate_steps_included is not True or self.job_outcome_success is not True:
            raise PublicAuthClosureError("official workflow receipt omits setup/evaluate time or success")
        if (
            self.external_governed_custody_verified is not False
            or self.whole_job_graph_closure_owed is not True
            or self.research_only is not True
        ):
            raise PublicAuthClosureError(
                "caller-authored workflow observations cannot claim governed whole-job custody"
            )

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(
            {
                "all_setup_and_evaluate_steps_included": self.all_setup_and_evaluate_steps_included,
                "archive_sha256": self.archive_sha256,
                "evaluate_step_process_trace_sha256": self.evaluate_step_process_trace_sha256,
                "execution_axis": self.execution_axis.value,
                "external_governed_custody_verified": self.external_governed_custody_verified,
                "job_log_sha256": self.job_log_sha256,
                "job_outcome_success": self.job_outcome_success,
                "job_wall_seconds": self.job_wall_seconds,
                "research_only": self.research_only,
                "report_sha256": self.report_sha256,
                "run_label": self.run_label,
                "runner_label": self.runner_label,
                "schema": self.schema,
                "timeout_minutes": self.timeout_minutes,
                "timing_source": self.timing_source,
                "upstream_snapshot_sha256": self.upstream_snapshot_sha256,
                "workflow_job_name": self.workflow_job_name,
                "workflow_relative_path": self.workflow_relative_path,
                "workflow_run_receipt_id": self.workflow_run_receipt_id,
                "workflow_sha256": self.workflow_sha256,
                "whole_job_graph_closure_owed": self.whole_job_graph_closure_owed,
            }
        )

    @classmethod
    def from_receipt_bytes(cls, payload: bytes) -> Self:
        value = _parse_canonical_object(payload, exact_keys=cls._KEYS)
        if value["schema"] != OFFICIAL_WORKFLOW_JOB_SCHEMA:
            raise PublicAuthClosureError("official workflow job schema drifted")
        return cls(
            run_label=value["run_label"],
            execution_axis=ExecutionAxisV1(value["execution_axis"]),
            external_governed_custody_verified=value["external_governed_custody_verified"],
            runner_label=value["runner_label"],
            workflow_relative_path=value["workflow_relative_path"],
            workflow_sha256=value["workflow_sha256"],
            workflow_job_name=value["workflow_job_name"],
            workflow_run_receipt_id=value["workflow_run_receipt_id"],
            job_log_sha256=value["job_log_sha256"],
            upstream_snapshot_sha256=value["upstream_snapshot_sha256"],
            archive_sha256=value["archive_sha256"],
            report_sha256=value["report_sha256"],
            evaluate_step_process_trace_sha256=value["evaluate_step_process_trace_sha256"],
            job_wall_seconds=value["job_wall_seconds"],
            research_only=value["research_only"],
            timeout_minutes=value["timeout_minutes"],
            timing_source=value["timing_source"],
            all_setup_and_evaluate_steps_included=value["all_setup_and_evaluate_steps_included"],
            job_outcome_success=value["job_outcome_success"],
            whole_job_graph_closure_owed=value["whole_job_graph_closure_owed"],
        )


@dataclass(frozen=True, slots=True, init=False)
class OfficialEvaluationRunReceiptV1:
    run_label: str
    archive_sha256: str
    archive_nbytes: int
    upstream_snapshot_sha256: str
    raw_sha256: str
    raw_nbytes: int
    n_pairs: int
    n_frames: int
    static_authority_input_file_manifest_sha256: str
    scorer_input_batch_content_sha256: str
    scorer_input_batch_ledger_receipt_ascii: str = field(repr=False)
    scorer_output_cell_content_sha256: str
    scorer_candidate_cell_content_sha256: str
    scorer_output_cell_ledger_receipt_sha256: str
    scorer_output_cell_ledger_receipt_ascii: str = field(repr=False)
    avg_segnet_dist: float
    avg_posenet_dist: float
    original_uncompressed_nbytes: int
    report_component_recomputed_score: float
    reported_final_score_decimal: str
    decode_wall_seconds: float
    total_workflow_seconds: float
    peak_process_tree_memory_nbytes: int
    peak_device_memory_nbytes: int
    execution_axis: ExecutionAxisV1
    platform_system: str
    platform_machine: str
    hardware_profile: str
    report_sha256: str
    inverse_trace_sha256: str
    inverse_trace_receipt_ascii: str = field(repr=False)
    process_trace_sha256: str
    official_workflow_job_receipt_sha256: str
    official_workflow_job_receipt_ascii: str = field(repr=False)
    syscall_trace_sha256: str
    observed_read_set_sha256: str
    normalized_observed_read_set_sha256: str
    observed_exec_set_sha256: str
    output_set_sha256: str
    dependency_discovery_receipt_sha256: str
    compile_receipt_sha256: str
    abi_identity_sha256: str
    exact_argv_sha256: str
    exact_environment_sha256: str
    stdout_sha256: str
    stderr_sha256: str
    exit_code: int
    executed_public_entrypoint_path: str
    exact_output_paths: tuple[str, ...]
    output_paths_all_regular: bool
    output_paths_no_symlinks: bool
    bytecode_contamination_paths: tuple[str, ...]
    research_only: bool
    schema: str = field(default="tac.taskspace_official_evaluation_run.v1", init=False)

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "archive_nbytes",
            "archive_sha256",
            "abi_identity_sha256",
            "avg_posenet_dist",
            "avg_segnet_dist",
            "static_authority_input_file_manifest_sha256",
            "scorer_input_batch_content_sha256",
            "scorer_input_batch_ledger_receipt_ascii",
            "scorer_output_cell_content_sha256",
            "scorer_candidate_cell_content_sha256",
            "scorer_output_cell_ledger_receipt_sha256",
            "scorer_output_cell_ledger_receipt_ascii",
            "bytecode_contamination_paths",
            "report_component_recomputed_score",
            "decode_wall_seconds",
            "dependency_discovery_receipt_sha256",
            "compile_receipt_sha256",
            "exact_argv_sha256",
            "exact_environment_sha256",
            "exact_output_paths",
            "executed_public_entrypoint_path",
            "execution_axis",
            "inverse_trace_sha256",
            "inverse_trace_receipt_ascii",
            "n_frames",
            "n_pairs",
            "original_uncompressed_nbytes",
            "observed_exec_set_sha256",
            "observed_read_set_sha256",
            "normalized_observed_read_set_sha256",
            "output_paths_all_regular",
            "output_paths_no_symlinks",
            "output_set_sha256",
            "peak_process_tree_memory_nbytes",
            "peak_device_memory_nbytes",
            "platform_system",
            "platform_machine",
            "hardware_profile",
            "raw_nbytes",
            "raw_sha256",
            "report_sha256",
            "reported_final_score_decimal",
            "run_label",
            "schema",
            "stderr_sha256",
            "stdout_sha256",
            "syscall_trace_sha256",
            "process_trace_sha256",
            "official_workflow_job_receipt_sha256",
            "official_workflow_job_receipt_ascii",
            "exit_code",
            "total_workflow_seconds",
            "upstream_snapshot_sha256",
            "research_only",
        }
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise PublicAuthClosureError("OfficialEvaluationRunReceiptV1 has no public constructor")

    @classmethod
    def _construct(cls, *, seal: object, **values: object) -> Self:
        return _sealed_frozen_dataclass(
            cls,
            seal=seal,
            allowed_seals=(_OFFICIAL_RUN_CONSTRUCTION_SEAL, _STRICT_RECEIPT_REOPEN_SEAL),
            values=values,
        )

    def __post_init__(self) -> None:
        if self.run_label not in {"A", "B"}:
            raise PublicAuthClosureError("official run label must be A or B")
        for value, label in (
            (self.archive_sha256, "official run archive"),
            (self.upstream_snapshot_sha256, "official run upstream snapshot"),
            (self.raw_sha256, "official run raw"),
            (
                self.static_authority_input_file_manifest_sha256,
                "official run static authority input file manifest",
            ),
            (self.scorer_input_batch_content_sha256, "official run scorer input batch content"),
            (self.scorer_output_cell_content_sha256, "official run scorer output cell content"),
            (self.scorer_candidate_cell_content_sha256, "official run scorer candidate cells"),
            (self.scorer_output_cell_ledger_receipt_sha256, "official run scorer output ledger"),
            (self.report_sha256, "official run report"),
            (self.inverse_trace_sha256, "official run inverse trace"),
            (self.process_trace_sha256, "official run process trace"),
            (self.official_workflow_job_receipt_sha256, "official run workflow job receipt"),
            (self.syscall_trace_sha256, "official run syscall trace"),
            (self.observed_read_set_sha256, "official run observed read set"),
            (self.normalized_observed_read_set_sha256, "official run normalized observed read set"),
            (self.observed_exec_set_sha256, "official run observed exec set"),
            (self.output_set_sha256, "official run output set"),
            (self.dependency_discovery_receipt_sha256, "official run dependency discovery"),
            (self.compile_receipt_sha256, "official run compile receipt"),
            (self.abi_identity_sha256, "official run ABI identity"),
            (self.exact_argv_sha256, "official run argv"),
            (self.exact_environment_sha256, "official run environment"),
            (self.stdout_sha256, "official run stdout"),
            (self.stderr_sha256, "official run stderr"),
        ):
            _require_sha256(value, label=label)
        _require_ascii(
            self.scorer_input_batch_ledger_receipt_ascii,
            label="embedded scorer input batch ledger",
        )
        _require_ascii(
            self.scorer_output_cell_ledger_receipt_ascii,
            label="embedded scorer output cell ledger",
        )
        _require_ascii(
            self.official_workflow_job_receipt_ascii,
            label="embedded official workflow job receipt",
        )
        _require_ascii(self.inverse_trace_receipt_ascii, label="embedded public inverse trace")
        inverse_trace = PublicInverseTraceReceiptV1.from_receipt_bytes(self.inverse_trace_receipt_ascii.encode("ascii"))
        if (
            inverse_trace.identity_sha256 != self.inverse_trace_sha256
            or inverse_trace.output_raw_sha256 != self.raw_sha256
            or inverse_trace.output_raw_bytes != self.raw_nbytes
        ):
            raise PublicAuthClosureError("official run embedded public inverse trace drifted")
        scorer_input_ledger = ScorerInputBatchLedgerV1.from_receipt_bytes(
            self.scorer_input_batch_ledger_receipt_ascii.encode("ascii")
        )
        if (
            scorer_input_ledger.input_content_sha256 != self.scorer_input_batch_content_sha256
            or scorer_input_ledger.archive_sha256 != self.archive_sha256
            or scorer_input_ledger.execution_axis is not self.execution_axis
            or scorer_input_ledger.candidate_raw_sha256 != self.raw_sha256
            or scorer_input_ledger.upstream_snapshot_sha256 != self.upstream_snapshot_sha256
            or scorer_input_ledger.static_authority_input_file_manifest_sha256
            != self.static_authority_input_file_manifest_sha256
        ):
            raise PublicAuthClosureError("official run embedded scorer input ledger drifted")
        scorer_output_ledger = ScorerOutputCellLedgerV1.from_receipt_bytes(
            self.scorer_output_cell_ledger_receipt_ascii.encode("ascii")
        )
        if (
            scorer_output_ledger.identity_sha256 != self.scorer_output_cell_ledger_receipt_sha256
            or scorer_output_ledger.content_sha256 != self.scorer_output_cell_content_sha256
            or scorer_output_ledger.candidate_cell_content_sha256 != self.scorer_candidate_cell_content_sha256
            or scorer_output_ledger.archive_sha256 != self.archive_sha256
            or scorer_output_ledger.execution_axis is not self.execution_axis
            or scorer_output_ledger.candidate_raw_sha256 != self.raw_sha256
            or scorer_output_ledger.upstream_snapshot_sha256 != self.upstream_snapshot_sha256
            or scorer_output_ledger.static_authority_input_file_manifest_sha256
            != self.static_authority_input_file_manifest_sha256
            or scorer_output_ledger.scorer_input_batch_content_sha256 != self.scorer_input_batch_content_sha256
        ):
            raise PublicAuthClosureError("official run embedded scorer output ledger drifted")
        input_mirror = ScorerOutputMirrorEquivalenceReceiptV1.from_receipt_bytes(
            scorer_input_ledger.observation_mirror_equivalence_receipt_ascii.encode("ascii")
        )
        output_mirror = ScorerOutputMirrorEquivalenceReceiptV1.from_receipt_bytes(
            scorer_output_ledger.observation_mirror_equivalence_receipt_ascii.encode("ascii")
        )
        if input_mirror.to_receipt_bytes() != output_mirror.to_receipt_bytes():
            raise PublicAuthClosureError("scorer input/output ledgers lack one shared mirror proof")
        if (
            output_mirror.run_label != self.run_label
            or output_mirror.official_process_trace_sha256 != self.process_trace_sha256
            or output_mirror.official_report_sha256 != self.report_sha256
        ):
            raise PublicAuthClosureError("scorer observation mirror is not bound to this official run")
        workflow = OfficialWorkflowJobReceiptV1.from_receipt_bytes(
            self.official_workflow_job_receipt_ascii.encode("ascii")
        )
        if (
            workflow.identity_sha256 != self.official_workflow_job_receipt_sha256
            or workflow.run_label != self.run_label
            or workflow.execution_axis is not self.execution_axis
            or workflow.upstream_snapshot_sha256 != self.upstream_snapshot_sha256
            or workflow.archive_sha256 != self.archive_sha256
            or workflow.report_sha256 != self.report_sha256
            or workflow.evaluate_step_process_trace_sha256 != self.process_trace_sha256
            or workflow.job_wall_seconds != self.total_workflow_seconds
        ):
            raise PublicAuthClosureError("official run whole-workflow job receipt drifted")
        if type(self.archive_nbytes) is not int or self.archive_nbytes < 1:
            raise PublicAuthClosureError("official run archive bytes must be positive int")
        if self.upstream_snapshot_sha256 != EXPECTED_UPSTREAM_SNAPSHOT_SHA256:
            raise PublicAuthClosureError("official run upstream snapshot is not the pinned public evaluator")
        if self.raw_nbytes != EXPECTED_RAW_NBYTES:
            raise PublicAuthClosureError("official run raw size violates exact 1200-frame contract")
        if self.n_pairs != EXPECTED_N_PAIRS or self.n_frames != EXPECTED_N_FRAMES:
            raise PublicAuthClosureError("official run must cover exactly 600 pairs / 1200 frames")
        for value, label in (
            (self.avg_segnet_dist, "official average SegNet distortion"),
            (self.avg_posenet_dist, "official average PoseNet distortion"),
            (self.report_component_recomputed_score, "report-component recomputed score"),
            (self.decode_wall_seconds, "official decode wall time"),
            (self.total_workflow_seconds, "official total workflow time"),
        ):
            _require_finite_nonnegative(value, label=label)
        if self.decode_wall_seconds > self.total_workflow_seconds:
            raise PublicAuthClosureError("official decode time exceeds total workflow time")
        if self.total_workflow_seconds > OFFICIAL_TOTAL_WORKFLOW_SECONDS:
            raise PublicAuthClosureError("official evaluate.sh workflow exceeded total 30-minute envelope")
        if type(self.peak_process_tree_memory_nbytes) is not int or self.peak_process_tree_memory_nbytes < 1:
            raise PublicAuthClosureError("official run peak process-tree memory must be measured")
        if self.original_uncompressed_nbytes != SCORE_RATE_DENOMINATOR:
            raise PublicAuthClosureError("official run uncompressed byte denominator drifted")
        if type(self.execution_axis) is not ExecutionAxisV1:
            raise PublicAuthClosureError("official run execution axis must be typed")
        if self.platform_system != "Linux" or self.platform_machine not in {"x86_64", "amd64"}:
            raise PublicAuthClosureError("official run authority requires Linux x86_64 contest hardware")
        expected_profile = {
            ExecutionAxisV1.CPU: "CONTEST_CPU_4_VCPU_16_GIB_LINUX_X86_64",
            ExecutionAxisV1.CUDA: "CONTEST_CUDA_T4_16_GIB_VRAM_26_GIB_RAM_LINUX_X86_64",
        }[self.execution_axis]
        if self.hardware_profile != expected_profile:
            raise PublicAuthClosureError("official run hardware profile does not match its authority axis")
        if type(self.peak_device_memory_nbytes) is not int or self.peak_device_memory_nbytes < 0:
            raise PublicAuthClosureError("official run device memory observation must be nonnegative int")
        if self.execution_axis is ExecutionAxisV1.CPU and self.peak_device_memory_nbytes != 0:
            raise PublicAuthClosureError("contest-CPU run cannot report CUDA device memory")
        if (
            self.execution_axis is ExecutionAxisV1.CUDA
            and not 1 <= self.peak_device_memory_nbytes <= CONTEST_CUDA_DEVICE_MEMORY_NBYTES
        ):
            raise PublicAuthClosureError("contest-CUDA run lacks bounded T4 device-memory observation")
        if (
            type(self.reported_final_score_decimal) is not str
            or re.fullmatch(r"[0-9]+\.[0-9]{2}", self.reported_final_score_decimal) is None
        ):
            raise PublicAuthClosureError("official displayed score must retain the exact two-decimal report text")
        expected_score = compute_contest_score(
            self.avg_segnet_dist,
            self.avg_posenet_dist,
            self.archive_nbytes,
            uncompressed_size=self.original_uncompressed_nbytes,
        )
        if not math.isclose(
            self.report_component_recomputed_score,
            expected_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise PublicAuthClosureError(
                "official run report-component score does not rederive from reported components"
            )
        if not _reported_score_interval_consistent(
            displayed_two_decimal=self.reported_final_score_decimal,
            avg_segnet_dist_8dec=self.avg_segnet_dist,
            avg_posenet_dist_8dec=self.avg_posenet_dist,
            archive_nbytes=self.archive_nbytes,
            original_uncompressed_nbytes=self.original_uncompressed_nbytes,
        ):
            raise PublicAuthClosureError(
                "official displayed score rounding interval cannot intersect hidden component intervals"
            )
        if self.executed_public_entrypoint_path != "upstream/evaluate.sh":
            raise PublicAuthClosureError("official run did not execute upstream/evaluate.sh")
        if type(self.exit_code) is not int or self.exit_code != 0:
            raise PublicAuthClosureError("official evaluate.sh execution did not exit zero")
        expected_output_paths = ("archive/0.bin", "inflated/0.raw", "report.txt")
        if type(self.exact_output_paths) is not tuple or self.exact_output_paths != expected_output_paths:
            raise PublicAuthClosureError(
                "official run output set is not exactly archive/0.bin, inflated/0.raw, and report.txt"
            )
        if self.output_paths_all_regular is not True or self.output_paths_no_symlinks is not True:
            raise PublicAuthClosureError("official run lacks regular/no-symlink output proof")
        if self.bytecode_contamination_paths:
            raise PublicAuthClosureError("official run generated or consumed Python bytecode contamination")
        if self.research_only is not True:
            raise PublicAuthClosureError(
                "reopened official-run observations remain research-only until governed execution custody exists"
            )
        if self.total_workflow_seconds <= 0.0 or self.decode_wall_seconds <= 0.0:
            raise PublicAuthClosureError("official run wall observations must be positive")
        host_limit = (
            CONTEST_MEMORY_NBYTES if self.execution_axis is ExecutionAxisV1.CPU else CONTEST_CUDA_HOST_MEMORY_NBYTES
        )
        if self.peak_process_tree_memory_nbytes > host_limit:
            raise PublicAuthClosureError("official run exceeded the contest 16 GiB memory envelope")

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(
            {
                "archive_nbytes": self.archive_nbytes,
                "archive_sha256": self.archive_sha256,
                "abi_identity_sha256": self.abi_identity_sha256,
                "avg_posenet_dist": self.avg_posenet_dist,
                "avg_segnet_dist": self.avg_segnet_dist,
                "static_authority_input_file_manifest_sha256": (self.static_authority_input_file_manifest_sha256),
                "scorer_input_batch_content_sha256": self.scorer_input_batch_content_sha256,
                "scorer_input_batch_ledger_receipt_ascii": (self.scorer_input_batch_ledger_receipt_ascii),
                "scorer_output_cell_content_sha256": self.scorer_output_cell_content_sha256,
                "scorer_candidate_cell_content_sha256": self.scorer_candidate_cell_content_sha256,
                "scorer_output_cell_ledger_receipt_sha256": (self.scorer_output_cell_ledger_receipt_sha256),
                "scorer_output_cell_ledger_receipt_ascii": (self.scorer_output_cell_ledger_receipt_ascii),
                "bytecode_contamination_paths": list(self.bytecode_contamination_paths),
                "report_component_recomputed_score": self.report_component_recomputed_score,
                "decode_wall_seconds": self.decode_wall_seconds,
                "dependency_discovery_receipt_sha256": self.dependency_discovery_receipt_sha256,
                "compile_receipt_sha256": self.compile_receipt_sha256,
                "exact_argv_sha256": self.exact_argv_sha256,
                "exact_environment_sha256": self.exact_environment_sha256,
                "exact_output_paths": list(self.exact_output_paths),
                "executed_public_entrypoint_path": self.executed_public_entrypoint_path,
                "execution_axis": self.execution_axis.value,
                "inverse_trace_sha256": self.inverse_trace_sha256,
                "inverse_trace_receipt_ascii": self.inverse_trace_receipt_ascii,
                "n_frames": self.n_frames,
                "n_pairs": self.n_pairs,
                "original_uncompressed_nbytes": self.original_uncompressed_nbytes,
                "observed_exec_set_sha256": self.observed_exec_set_sha256,
                "observed_read_set_sha256": self.observed_read_set_sha256,
                "normalized_observed_read_set_sha256": self.normalized_observed_read_set_sha256,
                "output_paths_all_regular": self.output_paths_all_regular,
                "output_paths_no_symlinks": self.output_paths_no_symlinks,
                "output_set_sha256": self.output_set_sha256,
                "peak_process_tree_memory_nbytes": self.peak_process_tree_memory_nbytes,
                "peak_device_memory_nbytes": self.peak_device_memory_nbytes,
                "platform_system": self.platform_system,
                "platform_machine": self.platform_machine,
                "hardware_profile": self.hardware_profile,
                "raw_nbytes": self.raw_nbytes,
                "raw_sha256": self.raw_sha256,
                "report_sha256": self.report_sha256,
                "reported_final_score_decimal": self.reported_final_score_decimal,
                "run_label": self.run_label,
                "schema": self.schema,
                "stderr_sha256": self.stderr_sha256,
                "stdout_sha256": self.stdout_sha256,
                "syscall_trace_sha256": self.syscall_trace_sha256,
                "process_trace_sha256": self.process_trace_sha256,
                "official_workflow_job_receipt_sha256": (self.official_workflow_job_receipt_sha256),
                "official_workflow_job_receipt_ascii": self.official_workflow_job_receipt_ascii,
                "exit_code": self.exit_code,
                "total_workflow_seconds": self.total_workflow_seconds,
                "upstream_snapshot_sha256": self.upstream_snapshot_sha256,
                "research_only": self.research_only,
            }
        )

    @classmethod
    def from_receipt_bytes(cls, payload: bytes) -> Self:
        value = _parse_canonical_object(payload, exact_keys=cls._KEYS)
        if value["schema"] != "tac.taskspace_official_evaluation_run.v1":
            raise PublicAuthClosureError("official evaluation run schema drifted")
        return cls._construct(
            seal=_STRICT_RECEIPT_REOPEN_SEAL,
            run_label=value["run_label"],
            archive_sha256=value["archive_sha256"],
            archive_nbytes=value["archive_nbytes"],
            upstream_snapshot_sha256=value["upstream_snapshot_sha256"],
            raw_sha256=value["raw_sha256"],
            raw_nbytes=value["raw_nbytes"],
            n_pairs=value["n_pairs"],
            n_frames=value["n_frames"],
            static_authority_input_file_manifest_sha256=value["static_authority_input_file_manifest_sha256"],
            scorer_input_batch_content_sha256=value["scorer_input_batch_content_sha256"],
            scorer_input_batch_ledger_receipt_ascii=value["scorer_input_batch_ledger_receipt_ascii"],
            scorer_output_cell_content_sha256=value["scorer_output_cell_content_sha256"],
            scorer_candidate_cell_content_sha256=value["scorer_candidate_cell_content_sha256"],
            scorer_output_cell_ledger_receipt_sha256=value["scorer_output_cell_ledger_receipt_sha256"],
            scorer_output_cell_ledger_receipt_ascii=value["scorer_output_cell_ledger_receipt_ascii"],
            avg_segnet_dist=value["avg_segnet_dist"],
            avg_posenet_dist=value["avg_posenet_dist"],
            original_uncompressed_nbytes=value["original_uncompressed_nbytes"],
            report_component_recomputed_score=value["report_component_recomputed_score"],
            reported_final_score_decimal=value["reported_final_score_decimal"],
            decode_wall_seconds=value["decode_wall_seconds"],
            total_workflow_seconds=value["total_workflow_seconds"],
            peak_process_tree_memory_nbytes=value["peak_process_tree_memory_nbytes"],
            peak_device_memory_nbytes=value["peak_device_memory_nbytes"],
            execution_axis=ExecutionAxisV1(value["execution_axis"]),
            platform_system=value["platform_system"],
            platform_machine=value["platform_machine"],
            hardware_profile=value["hardware_profile"],
            report_sha256=value["report_sha256"],
            inverse_trace_sha256=value["inverse_trace_sha256"],
            inverse_trace_receipt_ascii=value["inverse_trace_receipt_ascii"],
            process_trace_sha256=value["process_trace_sha256"],
            official_workflow_job_receipt_sha256=value["official_workflow_job_receipt_sha256"],
            official_workflow_job_receipt_ascii=value["official_workflow_job_receipt_ascii"],
            syscall_trace_sha256=value["syscall_trace_sha256"],
            observed_read_set_sha256=value["observed_read_set_sha256"],
            normalized_observed_read_set_sha256=value["normalized_observed_read_set_sha256"],
            observed_exec_set_sha256=value["observed_exec_set_sha256"],
            output_set_sha256=value["output_set_sha256"],
            dependency_discovery_receipt_sha256=value["dependency_discovery_receipt_sha256"],
            compile_receipt_sha256=value["compile_receipt_sha256"],
            abi_identity_sha256=value["abi_identity_sha256"],
            exact_argv_sha256=value["exact_argv_sha256"],
            exact_environment_sha256=value["exact_environment_sha256"],
            stdout_sha256=value["stdout_sha256"],
            stderr_sha256=value["stderr_sha256"],
            exit_code=value["exit_code"],
            executed_public_entrypoint_path=value["executed_public_entrypoint_path"],
            exact_output_paths=tuple(value["exact_output_paths"]),
            output_paths_all_regular=value["output_paths_all_regular"],
            output_paths_no_symlinks=value["output_paths_no_symlinks"],
            bytecode_contamination_paths=tuple(value["bytecode_contamination_paths"]),
            research_only=value["research_only"],
        )


@dataclass(frozen=True, slots=True, init=False)
class PublicDecodeEqualityReceiptV1:
    run_a_receipt_sha256: str
    run_b_receipt_sha256: str
    archive_sha256: str
    archive_nbytes: int
    upstream_snapshot_sha256: str
    raw_sha256: str
    raw_nbytes: int
    static_authority_input_file_manifest_sha256: str
    scorer_input_batch_content_sha256: str
    scorer_output_cell_content_sha256: str
    scorer_candidate_cell_content_sha256: str
    run_a_scorer_output_cell_ledger_receipt_sha256: str
    run_b_scorer_output_cell_ledger_receipt_sha256: str
    avg_segnet_dist: float
    avg_posenet_dist: float
    report_component_recomputed_score: float
    reported_final_score_decimal: str
    execution_axis: ExecutionAxisV1
    total_workflow_seconds_max: float
    decode_wall_seconds_max: float
    peak_process_tree_memory_nbytes_max: int
    dependency_discovery_receipt_sha256: str
    compile_receipt_sha256: str
    abi_identity_sha256: str
    exact_argv_sha256: str
    exact_environment_sha256: str
    normalized_observed_read_set_sha256: str
    observed_exec_set_sha256: str
    output_set_sha256: str
    fresh_run_count: int
    exact_output_equal: bool
    exact_scorer_components_equal: bool
    exact_closure_equal: bool
    research_only: bool
    schema: str = field(default=PUBLIC_DECODE_EQUALITY_SCHEMA, init=False)

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "archive_sha256",
            "archive_nbytes",
            "static_authority_input_file_manifest_sha256",
            "scorer_input_batch_content_sha256",
            "scorer_output_cell_content_sha256",
            "scorer_candidate_cell_content_sha256",
            "run_a_scorer_output_cell_ledger_receipt_sha256",
            "run_b_scorer_output_cell_ledger_receipt_sha256",
            "upstream_snapshot_sha256",
            "abi_identity_sha256",
            "avg_posenet_dist",
            "avg_segnet_dist",
            "report_component_recomputed_score",
            "decode_wall_seconds_max",
            "dependency_discovery_receipt_sha256",
            "compile_receipt_sha256",
            "exact_argv_sha256",
            "exact_environment_sha256",
            "normalized_observed_read_set_sha256",
            "observed_exec_set_sha256",
            "output_set_sha256",
            "fresh_run_count",
            "reported_final_score_decimal",
            "exact_closure_equal",
            "exact_output_equal",
            "exact_scorer_components_equal",
            "execution_axis",
            "peak_process_tree_memory_nbytes_max",
            "raw_nbytes",
            "raw_sha256",
            "run_a_receipt_sha256",
            "run_b_receipt_sha256",
            "schema",
            "total_workflow_seconds_max",
            "research_only",
        }
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise PublicAuthClosureError("PublicDecodeEqualityReceiptV1 has no public constructor")

    @classmethod
    def _construct(cls, *, seal: object, **values: object) -> Self:
        return _sealed_frozen_dataclass(
            cls,
            seal=seal,
            allowed_seals=(_PUBLIC_EQUALITY_DERIVATION_SEAL, _STRICT_RECEIPT_REOPEN_SEAL),
            values=values,
        )

    def __post_init__(self) -> None:
        for value, label in (
            (self.run_a_receipt_sha256, "decode equality run A"),
            (self.run_b_receipt_sha256, "decode equality run B"),
            (self.archive_sha256, "decode equality archive"),
            (self.raw_sha256, "decode equality raw"),
            (
                self.static_authority_input_file_manifest_sha256,
                "decode equality static authority input file manifest",
            ),
            (self.scorer_input_batch_content_sha256, "decode equality scorer input batch content"),
            (self.scorer_output_cell_content_sha256, "decode equality scorer output cell content"),
            (self.scorer_candidate_cell_content_sha256, "decode equality scorer candidate cells"),
            (
                self.run_a_scorer_output_cell_ledger_receipt_sha256,
                "decode equality run A scorer output ledger",
            ),
            (
                self.run_b_scorer_output_cell_ledger_receipt_sha256,
                "decode equality run B scorer output ledger",
            ),
            (self.upstream_snapshot_sha256, "decode equality upstream snapshot"),
            (self.dependency_discovery_receipt_sha256, "decode equality discovery"),
            (self.compile_receipt_sha256, "decode equality compile receipt"),
            (self.abi_identity_sha256, "decode equality ABI"),
            (self.exact_argv_sha256, "decode equality argv"),
            (self.exact_environment_sha256, "decode equality environment"),
            (self.normalized_observed_read_set_sha256, "decode equality normalized read set"),
            (self.observed_exec_set_sha256, "decode equality exec set"),
            (self.output_set_sha256, "decode equality output set"),
        ):
            _require_sha256(value, label=label)
        if self.raw_nbytes != EXPECTED_RAW_NBYTES:
            raise PublicAuthClosureError("decode equality raw size violates exact public contract")
        if type(self.archive_nbytes) is not int or self.archive_nbytes < 1:
            raise PublicAuthClosureError("decode equality archive size must be positive int")
        if self.upstream_snapshot_sha256 != EXPECTED_UPSTREAM_SNAPSHOT_SHA256:
            raise PublicAuthClosureError("decode equality upstream snapshot drifted")
        for value, label in (
            (self.avg_segnet_dist, "decode equality SegNet"),
            (self.avg_posenet_dist, "decode equality PoseNet"),
            (self.report_component_recomputed_score, "decode equality report-component score"),
            (self.total_workflow_seconds_max, "decode equality total time"),
            (self.decode_wall_seconds_max, "decode equality decode time"),
        ):
            _require_finite_nonnegative(value, label=label)
        if self.total_workflow_seconds_max > OFFICIAL_TOTAL_WORKFLOW_SECONDS:
            raise PublicAuthClosureError("decode equality exceeds official total workflow envelope")
        if type(self.peak_process_tree_memory_nbytes_max) is not int or self.peak_process_tree_memory_nbytes_max < 1:
            raise PublicAuthClosureError("decode equality requires measured peak process-tree memory")
        if type(self.execution_axis) is not ExecutionAxisV1:
            raise PublicAuthClosureError("decode equality axis must be typed")
        if self.fresh_run_count != 2:
            raise PublicAuthClosureError("public decode equality requires exactly two fresh official runs")
        if (
            self.exact_output_equal is not True
            or self.exact_scorer_components_equal is not True
            or self.exact_closure_equal is not True
        ):
            raise PublicAuthClosureError("public authority requires double-run output/scorer/closure equality")
        if self.research_only is not True:
            raise PublicAuthClosureError("parsed A/B equality remains research-only without governed execution custody")
        if (
            type(self.reported_final_score_decimal) is not str
            or re.fullmatch(r"[0-9]+\.[0-9]{2}", self.reported_final_score_decimal) is None
            or not _reported_score_interval_consistent(
                displayed_two_decimal=self.reported_final_score_decimal,
                avg_segnet_dist_8dec=self.avg_segnet_dist,
                avg_posenet_dist_8dec=self.avg_posenet_dist,
                archive_nbytes=self.archive_nbytes,
                original_uncompressed_nbytes=SCORE_RATE_DENOMINATOR,
            )
        ):
            raise PublicAuthClosureError("decode equality displayed score is report-inconsistent")

    @classmethod
    def from_runs(cls, run_a: OfficialEvaluationRunReceiptV1, run_b: OfficialEvaluationRunReceiptV1) -> Self:
        if type(run_a) is not OfficialEvaluationRunReceiptV1 or type(run_b) is not OfficialEvaluationRunReceiptV1:
            raise PublicAuthClosureError("decode equality accepts only sealed official-run receipts")
        if run_a.run_label != "A" or run_b.run_label != "B":
            raise PublicAuthClosureError("decode equality requires labeled A then B official runs")
        inverse_a = PublicInverseTraceReceiptV1.from_receipt_bytes(run_a.inverse_trace_receipt_ascii.encode("ascii"))
        inverse_b = PublicInverseTraceReceiptV1.from_receipt_bytes(run_b.inverse_trace_receipt_ascii.encode("ascii"))
        shared = (
            run_a.archive_sha256 == run_b.archive_sha256
            and run_a.archive_nbytes == run_b.archive_nbytes
            and run_a.upstream_snapshot_sha256 == run_b.upstream_snapshot_sha256
            and run_a.raw_sha256 == run_b.raw_sha256
            and run_a.raw_nbytes == run_b.raw_nbytes
            and run_a.static_authority_input_file_manifest_sha256 == run_b.static_authority_input_file_manifest_sha256
            and run_a.scorer_input_batch_content_sha256 == run_b.scorer_input_batch_content_sha256
            and run_a.scorer_output_cell_content_sha256 == run_b.scorer_output_cell_content_sha256
            and run_a.scorer_candidate_cell_content_sha256 == run_b.scorer_candidate_cell_content_sha256
            and run_a.execution_axis is run_b.execution_axis
            and run_a.exact_output_paths == run_b.exact_output_paths
            and run_a.output_set_sha256 == run_b.output_set_sha256
            and run_a.platform_system == run_b.platform_system
            and run_a.platform_machine == run_b.platform_machine
            and run_a.hardware_profile == run_b.hardware_profile
        )
        scorer_equal = (
            run_a.avg_segnet_dist == run_b.avg_segnet_dist
            and run_a.avg_posenet_dist == run_b.avg_posenet_dist
            and run_a.report_component_recomputed_score == run_b.report_component_recomputed_score
            and run_a.reported_final_score_decimal == run_b.reported_final_score_decimal
            and run_a.original_uncompressed_nbytes == run_b.original_uncompressed_nbytes
        )
        closure_equal = (
            run_a.dependency_discovery_receipt_sha256 == run_b.dependency_discovery_receipt_sha256
            and run_a.compile_receipt_sha256 == run_b.compile_receipt_sha256
            and run_a.abi_identity_sha256 == run_b.abi_identity_sha256
            and run_a.exact_argv_sha256 == run_b.exact_argv_sha256
            and run_a.exact_environment_sha256 == run_b.exact_environment_sha256
            and run_a.normalized_observed_read_set_sha256 == run_b.normalized_observed_read_set_sha256
            and run_a.observed_exec_set_sha256 == run_b.observed_exec_set_sha256
            and run_a.executed_public_entrypoint_path == run_b.executed_public_entrypoint_path
            and inverse_a.inverse_source_sha256 == inverse_b.inverse_source_sha256
            and inverse_a.runtime_sha256 == inverse_b.runtime_sha256
            and inverse_a.source_member_sha256 == inverse_b.source_member_sha256
            and inverse_a.logical_lvls1_sha256 == inverse_b.logical_lvls1_sha256
            and inverse_a.output_raw_sha256 == inverse_b.output_raw_sha256
        )
        if not shared or not scorer_equal or not closure_equal:
            raise PublicAuthClosureError("official A/B runs are not exact output/scorer/closure equal")
        return cls._construct(
            seal=_PUBLIC_EQUALITY_DERIVATION_SEAL,
            run_a_receipt_sha256=run_a.identity_sha256,
            run_b_receipt_sha256=run_b.identity_sha256,
            archive_sha256=run_a.archive_sha256,
            archive_nbytes=run_a.archive_nbytes,
            upstream_snapshot_sha256=run_a.upstream_snapshot_sha256,
            raw_sha256=run_a.raw_sha256,
            raw_nbytes=run_a.raw_nbytes,
            static_authority_input_file_manifest_sha256=(run_a.static_authority_input_file_manifest_sha256),
            scorer_input_batch_content_sha256=run_a.scorer_input_batch_content_sha256,
            scorer_output_cell_content_sha256=run_a.scorer_output_cell_content_sha256,
            scorer_candidate_cell_content_sha256=run_a.scorer_candidate_cell_content_sha256,
            run_a_scorer_output_cell_ledger_receipt_sha256=(run_a.scorer_output_cell_ledger_receipt_sha256),
            run_b_scorer_output_cell_ledger_receipt_sha256=(run_b.scorer_output_cell_ledger_receipt_sha256),
            avg_segnet_dist=run_a.avg_segnet_dist,
            avg_posenet_dist=run_a.avg_posenet_dist,
            report_component_recomputed_score=run_a.report_component_recomputed_score,
            reported_final_score_decimal=run_a.reported_final_score_decimal,
            execution_axis=run_a.execution_axis,
            total_workflow_seconds_max=max(run_a.total_workflow_seconds, run_b.total_workflow_seconds),
            decode_wall_seconds_max=max(run_a.decode_wall_seconds, run_b.decode_wall_seconds),
            peak_process_tree_memory_nbytes_max=max(
                run_a.peak_process_tree_memory_nbytes,
                run_b.peak_process_tree_memory_nbytes,
            ),
            dependency_discovery_receipt_sha256=run_a.dependency_discovery_receipt_sha256,
            compile_receipt_sha256=run_a.compile_receipt_sha256,
            abi_identity_sha256=run_a.abi_identity_sha256,
            exact_argv_sha256=run_a.exact_argv_sha256,
            exact_environment_sha256=run_a.exact_environment_sha256,
            normalized_observed_read_set_sha256=(run_a.normalized_observed_read_set_sha256),
            observed_exec_set_sha256=run_a.observed_exec_set_sha256,
            output_set_sha256=run_a.output_set_sha256,
            fresh_run_count=2,
            exact_output_equal=True,
            exact_scorer_components_equal=True,
            exact_closure_equal=True,
            research_only=True,
        )

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(
            {
                "archive_sha256": self.archive_sha256,
                "archive_nbytes": self.archive_nbytes,
                "upstream_snapshot_sha256": self.upstream_snapshot_sha256,
                "abi_identity_sha256": self.abi_identity_sha256,
                "avg_posenet_dist": self.avg_posenet_dist,
                "avg_segnet_dist": self.avg_segnet_dist,
                "report_component_recomputed_score": self.report_component_recomputed_score,
                "decode_wall_seconds_max": self.decode_wall_seconds_max,
                "dependency_discovery_receipt_sha256": self.dependency_discovery_receipt_sha256,
                "compile_receipt_sha256": self.compile_receipt_sha256,
                "exact_argv_sha256": self.exact_argv_sha256,
                "exact_environment_sha256": self.exact_environment_sha256,
                "normalized_observed_read_set_sha256": (self.normalized_observed_read_set_sha256),
                "observed_exec_set_sha256": self.observed_exec_set_sha256,
                "output_set_sha256": self.output_set_sha256,
                "fresh_run_count": self.fresh_run_count,
                "reported_final_score_decimal": self.reported_final_score_decimal,
                "exact_closure_equal": self.exact_closure_equal,
                "exact_output_equal": self.exact_output_equal,
                "exact_scorer_components_equal": self.exact_scorer_components_equal,
                "execution_axis": self.execution_axis.value,
                "peak_process_tree_memory_nbytes_max": self.peak_process_tree_memory_nbytes_max,
                "raw_nbytes": self.raw_nbytes,
                "raw_sha256": self.raw_sha256,
                "static_authority_input_file_manifest_sha256": (self.static_authority_input_file_manifest_sha256),
                "scorer_input_batch_content_sha256": self.scorer_input_batch_content_sha256,
                "scorer_output_cell_content_sha256": self.scorer_output_cell_content_sha256,
                "scorer_candidate_cell_content_sha256": self.scorer_candidate_cell_content_sha256,
                "run_a_scorer_output_cell_ledger_receipt_sha256": (self.run_a_scorer_output_cell_ledger_receipt_sha256),
                "run_b_scorer_output_cell_ledger_receipt_sha256": (self.run_b_scorer_output_cell_ledger_receipt_sha256),
                "run_a_receipt_sha256": self.run_a_receipt_sha256,
                "run_b_receipt_sha256": self.run_b_receipt_sha256,
                "schema": self.schema,
                "total_workflow_seconds_max": self.total_workflow_seconds_max,
                "research_only": self.research_only,
            }
        )

    @classmethod
    def from_receipt_bytes(cls, payload: bytes) -> Self:
        value = _parse_canonical_object(payload, exact_keys=cls._KEYS)
        if value["schema"] != PUBLIC_DECODE_EQUALITY_SCHEMA:
            raise PublicAuthClosureError("public decode equality schema drifted")
        return cls._construct(
            seal=_STRICT_RECEIPT_REOPEN_SEAL,
            run_a_receipt_sha256=value["run_a_receipt_sha256"],
            run_b_receipt_sha256=value["run_b_receipt_sha256"],
            archive_sha256=value["archive_sha256"],
            archive_nbytes=value["archive_nbytes"],
            upstream_snapshot_sha256=value["upstream_snapshot_sha256"],
            raw_sha256=value["raw_sha256"],
            raw_nbytes=value["raw_nbytes"],
            static_authority_input_file_manifest_sha256=value["static_authority_input_file_manifest_sha256"],
            scorer_input_batch_content_sha256=value["scorer_input_batch_content_sha256"],
            scorer_output_cell_content_sha256=value["scorer_output_cell_content_sha256"],
            scorer_candidate_cell_content_sha256=value["scorer_candidate_cell_content_sha256"],
            run_a_scorer_output_cell_ledger_receipt_sha256=value["run_a_scorer_output_cell_ledger_receipt_sha256"],
            run_b_scorer_output_cell_ledger_receipt_sha256=value["run_b_scorer_output_cell_ledger_receipt_sha256"],
            avg_segnet_dist=value["avg_segnet_dist"],
            avg_posenet_dist=value["avg_posenet_dist"],
            report_component_recomputed_score=value["report_component_recomputed_score"],
            reported_final_score_decimal=value["reported_final_score_decimal"],
            execution_axis=ExecutionAxisV1(value["execution_axis"]),
            total_workflow_seconds_max=value["total_workflow_seconds_max"],
            decode_wall_seconds_max=value["decode_wall_seconds_max"],
            peak_process_tree_memory_nbytes_max=value["peak_process_tree_memory_nbytes_max"],
            dependency_discovery_receipt_sha256=value["dependency_discovery_receipt_sha256"],
            compile_receipt_sha256=value["compile_receipt_sha256"],
            abi_identity_sha256=value["abi_identity_sha256"],
            exact_argv_sha256=value["exact_argv_sha256"],
            exact_environment_sha256=value["exact_environment_sha256"],
            normalized_observed_read_set_sha256=value["normalized_observed_read_set_sha256"],
            observed_exec_set_sha256=value["observed_exec_set_sha256"],
            output_set_sha256=value["output_set_sha256"],
            fresh_run_count=value["fresh_run_count"],
            exact_output_equal=value["exact_output_equal"],
            exact_scorer_components_equal=value["exact_scorer_components_equal"],
            exact_closure_equal=value["exact_closure_equal"],
            research_only=value["research_only"],
        )


@dataclass(frozen=True, slots=True)
class AccountedExternalReadV1:
    absolute_path: str
    kind: ExternalReadKindV1
    custody_owner: str
    content_sha256: str | None
    nbytes: int

    def __post_init__(self) -> None:
        if (
            type(self.absolute_path) is not str
            or not self.absolute_path.isascii()
            or not Path(self.absolute_path).is_absolute()
        ):
            raise PublicAuthClosureError("accounted external read path must be absolute ASCII")
        if type(self.kind) is not ExternalReadKindV1:
            raise PublicAuthClosureError("accounted external read kind must be typed")
        _require_identifier(self.custody_owner, label="accounted external read custody owner")
        if self.kind is ExternalReadKindV1.VIRTUAL_KERNEL_FILE:
            if self.content_sha256 is not None or self.nbytes != 0:
                raise PublicAuthClosureError("virtual kernel read cannot claim stable content bytes")
            if not any(
                Path(root) == Path(self.absolute_path) or Path(root) in Path(self.absolute_path).parents
                for root in ("/dev", "/proc", "/sys")
            ):
                raise PublicAuthClosureError("virtual kernel read escaped explicit /dev,/proc,/sys roots")
        else:
            _require_sha256(self.content_sha256, label="accounted external regular file")
            if type(self.nbytes) is not int or self.nbytes < 1:
                raise PublicAuthClosureError("accounted external regular file must retain positive bytes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "absolute_path": self.absolute_path,
            "content_sha256": self.content_sha256,
            "custody_owner": self.custody_owner,
            "kind": self.kind.value,
            "nbytes": self.nbytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            "absolute_path",
            "content_sha256",
            "custody_owner",
            "kind",
            "nbytes",
        }:
            raise PublicAuthClosureError("accounted external read fields drifted")
        return cls(
            absolute_path=value["absolute_path"],
            kind=ExternalReadKindV1(value["kind"]),
            custody_owner=value["custody_owner"],
            content_sha256=value["content_sha256"],
            nbytes=value["nbytes"],
        )


@dataclass(frozen=True, slots=True)
class TracePathNormalizationV1:
    """Reviewed run-specific absolute root mapped to a stable semantic placeholder."""

    absolute_root: str
    placeholder: str

    def __post_init__(self) -> None:
        if (
            type(self.absolute_root) is not str
            or not self.absolute_root.isascii()
            or not Path(self.absolute_root).is_absolute()
        ):
            raise PublicAuthClosureError("trace normalization root must be absolute ASCII")
        if self.placeholder not in {"@DECODER_SCRATCH@", "@OFFICIAL_RUN_ROOT@"}:
            raise PublicAuthClosureError("trace normalization placeholder is not reviewed")

    def to_dict(self) -> dict[str, str]:
        return {"absolute_root": self.absolute_root, "placeholder": self.placeholder}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {"absolute_root", "placeholder"}:
            raise PublicAuthClosureError("trace normalization fields drifted")
        return cls(absolute_root=value["absolute_root"], placeholder=value["placeholder"])


@dataclass(frozen=True, slots=True)
class PublicTraceClosureReceiptV1:
    """Normalized whole-process trace/enforcement closure for one official run.

    A digest of opaque tracer output alone is not enough.  The normalized read
    and exec sets are retained, every path must lie under an explicit allowlist,
    and the trace backend must either observe or fail-closed deny escape.
    """

    backend: str
    raw_trace_sha256: str
    observed_read_paths: tuple[str, ...]
    observed_exec_paths: tuple[str, ...]
    decoder_observed_read_paths: tuple[str, ...]
    evaluator_observed_read_paths: tuple[str, ...]
    orchestrator_observed_read_paths: tuple[str, ...]
    phase_process_map_sha256: str
    accounted_external_reads: tuple[AccountedExternalReadV1, ...]
    path_normalizations: tuple[TracePathNormalizationV1, ...]
    allowed_read_roots: tuple[str, ...]
    allowed_exec_paths: tuple[str, ...]
    network_denied: bool
    outside_reads_denied_or_absent: bool
    process_tree_complete: bool
    phase_attribution_complete: bool
    decoder_authority_reads_denied_or_absent: bool
    schema: str = field(default=TRACE_CLOSURE_SCHEMA, init=False)

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "allowed_exec_paths",
            "allowed_read_roots",
            "accounted_external_reads",
            "backend",
            "decoder_authority_reads_denied_or_absent",
            "decoder_observed_read_paths",
            "evaluator_observed_read_paths",
            "network_denied",
            "observed_exec_paths",
            "observed_read_paths",
            "outside_reads_denied_or_absent",
            "path_normalizations",
            "orchestrator_observed_read_paths",
            "phase_attribution_complete",
            "phase_process_map_sha256",
            "process_tree_complete",
            "raw_trace_sha256",
            "schema",
        }
    )

    def __post_init__(self) -> None:
        if self.backend not in {
            "LINUX_STRACE_FILE_PROCESS_ALLOWLIST_V1",
            "MACOS_SANDBOX_FAIL_CLOSED_PLUS_PROCESS_TRACE_V1",
        }:
            raise PublicAuthClosureError("public trace closure backend is unsupported")
        _require_sha256(self.raw_trace_sha256, label="public trace bytes")
        _require_sha256(self.phase_process_map_sha256, label="public trace phase/process map")
        if (
            type(self.accounted_external_reads) is not tuple
            or any(type(item) is not AccountedExternalReadV1 for item in self.accounted_external_reads)
            or self.accounted_external_reads
            != tuple(sorted(self.accounted_external_reads, key=lambda item: item.absolute_path))
            or len({item.absolute_path for item in self.accounted_external_reads}) != len(self.accounted_external_reads)
        ):
            raise PublicAuthClosureError("accounted external reads must be a unique canonical typed tuple")
        if (
            type(self.path_normalizations) is not tuple
            or any(type(item) is not TracePathNormalizationV1 for item in self.path_normalizations)
            or self.path_normalizations != tuple(sorted(self.path_normalizations, key=lambda item: item.placeholder))
            or {item.placeholder for item in self.path_normalizations} != {"@DECODER_SCRATCH@", "@OFFICIAL_RUN_ROOT@"}
            or len({item.absolute_root for item in self.path_normalizations}) != 2
        ):
            raise PublicAuthClosureError("trace path normalizations must close exact run/scratch roots")
        for values, label in (
            (self.observed_read_paths, "observed read paths"),
            (self.observed_exec_paths, "observed exec paths"),
            (self.decoder_observed_read_paths, "decoder observed read paths"),
            (self.evaluator_observed_read_paths, "evaluator observed read paths"),
            (self.orchestrator_observed_read_paths, "orchestrator observed read paths"),
            (self.allowed_read_roots, "allowed read roots"),
            (self.allowed_exec_paths, "allowed exec paths"),
        ):
            if (
                type(values) is not tuple
                or not values
                or values != tuple(sorted(set(values)))
                or any(
                    type(value) is not str or not value.isascii() or not Path(value).is_absolute() for value in values
                )
            ):
                raise PublicAuthClosureError(f"{label} must be a unique canonical absolute ASCII tuple")
        attributed_reads = (
            set(self.decoder_observed_read_paths)
            | set(self.evaluator_observed_read_paths)
            | set(self.orchestrator_observed_read_paths)
        )
        if attributed_reads != set(self.observed_read_paths):
            raise PublicAuthClosureError("public trace phase attribution does not cover exact observed reads")
        for normalization in self.path_normalizations:
            root = Path(normalization.absolute_root)
            if not any(Path(value) == root or root in Path(value).parents for value in self.observed_read_paths):
                raise PublicAuthClosureError("trace normalization root did not own any observed read")
        read_roots = tuple(Path(value).resolve() for value in self.allowed_read_roots)
        if any(
            not any(path == root or root in path.parents for root in read_roots)
            for path in (Path(value).resolve() for value in self.observed_read_paths)
        ):
            raise PublicAuthClosureError("public trace observed a read outside the closed allowlist")
        if not set(self.observed_exec_paths).issubset(self.allowed_exec_paths):
            raise PublicAuthClosureError("public trace observed an unowned executable")
        if (
            self.network_denied is not True
            or self.outside_reads_denied_or_absent is not True
            or self.process_tree_complete is not True
            or self.phase_attribution_complete is not True
            or self.decoder_authority_reads_denied_or_absent is not True
        ):
            raise PublicAuthClosureError("public trace did not close network/read/process-tree/phase escape")

    @property
    def observed_read_set_sha256(self) -> str:
        return _sha256(_canonical_json(list(self.observed_read_paths)))

    @property
    def observed_exec_set_sha256(self) -> str:
        return _sha256(_canonical_json(list(self.observed_exec_paths)))

    @property
    def normalized_observed_read_paths(self) -> tuple[str, ...]:
        roots = tuple(
            sorted(
                ((Path(item.absolute_root), item.placeholder) for item in self.path_normalizations),
                key=lambda item: len(item[0].parts),
                reverse=True,
            )
        )
        normalized: list[str] = []
        for value in self.observed_read_paths:
            path = Path(value)
            rendered = value
            for root, placeholder in roots:
                if path == root or root in path.parents:
                    relative = path.relative_to(root).as_posix()
                    rendered = placeholder if relative == "." else f"{placeholder}/{relative}"
                    break
            normalized.append(rendered)
        return tuple(sorted(set(normalized)))

    @property
    def normalized_observed_read_set_sha256(self) -> str:
        return _sha256(_canonical_json(list(self.normalized_observed_read_paths)))

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(
            {
                "allowed_exec_paths": list(self.allowed_exec_paths),
                "allowed_read_roots": list(self.allowed_read_roots),
                "accounted_external_reads": [item.to_dict() for item in self.accounted_external_reads],
                "backend": self.backend,
                "decoder_authority_reads_denied_or_absent": (self.decoder_authority_reads_denied_or_absent),
                "decoder_observed_read_paths": list(self.decoder_observed_read_paths),
                "evaluator_observed_read_paths": list(self.evaluator_observed_read_paths),
                "network_denied": self.network_denied,
                "observed_exec_paths": list(self.observed_exec_paths),
                "observed_read_paths": list(self.observed_read_paths),
                "outside_reads_denied_or_absent": self.outside_reads_denied_or_absent,
                "path_normalizations": [item.to_dict() for item in self.path_normalizations],
                "orchestrator_observed_read_paths": list(self.orchestrator_observed_read_paths),
                "phase_attribution_complete": self.phase_attribution_complete,
                "phase_process_map_sha256": self.phase_process_map_sha256,
                "process_tree_complete": self.process_tree_complete,
                "raw_trace_sha256": self.raw_trace_sha256,
                "schema": self.schema,
            }
        )

    @classmethod
    def from_receipt_bytes(cls, payload: bytes) -> Self:
        value = _parse_canonical_object(payload, exact_keys=cls._KEYS)
        if value["schema"] != TRACE_CLOSURE_SCHEMA:
            raise PublicAuthClosureError("public trace closure schema drifted")
        return cls(
            backend=value["backend"],
            raw_trace_sha256=value["raw_trace_sha256"],
            observed_read_paths=tuple(value["observed_read_paths"]),
            observed_exec_paths=tuple(value["observed_exec_paths"]),
            decoder_observed_read_paths=tuple(value["decoder_observed_read_paths"]),
            evaluator_observed_read_paths=tuple(value["evaluator_observed_read_paths"]),
            orchestrator_observed_read_paths=tuple(value["orchestrator_observed_read_paths"]),
            phase_process_map_sha256=value["phase_process_map_sha256"],
            accounted_external_reads=tuple(
                AccountedExternalReadV1.from_dict(row) for row in value["accounted_external_reads"]
            ),
            path_normalizations=tuple(TracePathNormalizationV1.from_dict(row) for row in value["path_normalizations"]),
            allowed_read_roots=tuple(value["allowed_read_roots"]),
            allowed_exec_paths=tuple(value["allowed_exec_paths"]),
            network_denied=value["network_denied"],
            outside_reads_denied_or_absent=value["outside_reads_denied_or_absent"],
            process_tree_complete=value["process_tree_complete"],
            phase_attribution_complete=value["phase_attribution_complete"],
            decoder_authority_reads_denied_or_absent=value["decoder_authority_reads_denied_or_absent"],
        )


@dataclass(frozen=True, slots=True)
class AuthClosureExecutionReadinessReceiptV1:
    compile_receipt_sha256: str
    decoder_abi_identity_sha256: str
    runtime_tree_sha256: str
    dependency_discovery_receipt_sha256: str | None
    expected_upstream_snapshot_sha256: str
    trace_backend: str
    bytecode_contamination_paths: tuple[str, ...]
    preflight_blockers: tuple[str, ...]
    authority_owed: tuple[str, ...]
    ready_to_execute: bool
    auth_closure_proven: bool
    research_only: bool
    schema: str = field(default=READINESS_SCHEMA, init=False)

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "decoder_abi_identity_sha256",
            "auth_closure_proven",
            "authority_owed",
            "bytecode_contamination_paths",
            "compile_receipt_sha256",
            "dependency_discovery_receipt_sha256",
            "expected_upstream_snapshot_sha256",
            "preflight_blockers",
            "ready_to_execute",
            "research_only",
            "runtime_tree_sha256",
            "schema",
            "trace_backend",
        }
    )

    def __post_init__(self) -> None:
        for value, label in (
            (self.compile_receipt_sha256, "readiness compile receipt"),
            (self.decoder_abi_identity_sha256, "readiness decoder ABI"),
            (self.runtime_tree_sha256, "readiness runtime tree"),
            (self.expected_upstream_snapshot_sha256, "readiness upstream snapshot"),
        ):
            _require_sha256(value, label=label)
        if self.dependency_discovery_receipt_sha256 is not None:
            _require_sha256(
                self.dependency_discovery_receipt_sha256,
                label="readiness dependency discovery",
            )
        if self.expected_upstream_snapshot_sha256 != EXPECTED_UPSTREAM_SNAPSHOT_SHA256:
            raise PublicAuthClosureError("readiness receipt expected upstream pin drifted")
        _require_ascii(self.trace_backend, label="readiness trace backend")
        for values, label in (
            (self.bytecode_contamination_paths, "readiness bytecode paths"),
            (self.preflight_blockers, "readiness preflight blockers"),
            (self.authority_owed, "readiness authority owed"),
        ):
            if (
                type(values) is not tuple
                or values != tuple(sorted(set(values)))
                or any(type(value) is not str or not value.isascii() for value in values)
            ):
                raise PublicAuthClosureError(f"{label} must be a canonical ASCII tuple")
        if self.ready_to_execute != (not self.preflight_blockers):
            raise PublicAuthClosureError("readiness boolean does not derive from preflight blockers")
        required_owed = {
            "ACTUAL_38_BATCH_SCORER_INPUT_LEDGER",
            "ACTUAL_N600_SCORER_OUTPUT_CELL_LEDGER",
            "EXACT_PUBLIC_N600_OFFICIAL_RUN_A",
            "EXACT_PUBLIC_N600_OFFICIAL_RUN_B",
            "DOUBLE_RUN_RAW_SCORER_CLOSURE_EQUALITY",
            "EXTERNAL_GOVERNED_EXECUTION_EVIDENCE_BOUNDARY",
            "OFFICIAL_GITHUB_TEST_JOB_30_MINUTE_WALL_RECEIPT",
            "REVIEWED_SCORER_INPUT_OUTPUT_OBSERVATION_MIRROR_EQUIVALENCE",
            "SEALED_C0B_AUTH_EVAL_ADAPTER_REVIEW",
        }
        if not required_owed.issubset(self.authority_owed):
            raise PublicAuthClosureError("readiness receipt hid required authority debt")
        if self.auth_closure_proven is not False or self.research_only is not True:
            raise PublicAuthClosureError("pre-execution readiness must remain research-only and non-authority")

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(
            {
                "decoder_abi_identity_sha256": self.decoder_abi_identity_sha256,
                "auth_closure_proven": self.auth_closure_proven,
                "authority_owed": list(self.authority_owed),
                "bytecode_contamination_paths": list(self.bytecode_contamination_paths),
                "compile_receipt_sha256": self.compile_receipt_sha256,
                "dependency_discovery_receipt_sha256": self.dependency_discovery_receipt_sha256,
                "expected_upstream_snapshot_sha256": self.expected_upstream_snapshot_sha256,
                "preflight_blockers": list(self.preflight_blockers),
                "ready_to_execute": self.ready_to_execute,
                "research_only": self.research_only,
                "runtime_tree_sha256": self.runtime_tree_sha256,
                "schema": self.schema,
                "trace_backend": self.trace_backend,
            }
        )

    @classmethod
    def from_receipt_bytes(cls, payload: bytes) -> Self:
        value = _parse_canonical_object(payload, exact_keys=cls._KEYS)
        if value["schema"] != READINESS_SCHEMA:
            raise PublicAuthClosureError("auth-closure readiness schema drifted")
        return cls(
            compile_receipt_sha256=value["compile_receipt_sha256"],
            decoder_abi_identity_sha256=value["decoder_abi_identity_sha256"],
            runtime_tree_sha256=value["runtime_tree_sha256"],
            dependency_discovery_receipt_sha256=value["dependency_discovery_receipt_sha256"],
            expected_upstream_snapshot_sha256=value["expected_upstream_snapshot_sha256"],
            trace_backend=value["trace_backend"],
            bytecode_contamination_paths=tuple(value["bytecode_contamination_paths"]),
            preflight_blockers=tuple(value["preflight_blockers"]),
            authority_owed=tuple(value["authority_owed"]),
            ready_to_execute=value["ready_to_execute"],
            auth_closure_proven=value["auth_closure_proven"],
            research_only=value["research_only"],
        )


def assess_auth_eval_execution_readiness(
    *,
    repo_root: Path,
    runtime_dir: Path,
    compiled: CompiledPublicRuntimeV1,
    discovery: RuntimeDependencyDiscoveryReceiptV1 | None,
) -> AuthClosureExecutionReadinessReceiptV1:
    """Derive an execution-ready or exact-blocker receipt without claiming a run."""

    repo_root = repo_root.resolve(strict=True)
    runtime_dir = runtime_dir.resolve(strict=True)
    if type(compiled) is not CompiledPublicRuntimeV1:
        raise PublicAuthClosureError("readiness requires the exact compiled public runtime")
    require_exact_public_runtime_tree(runtime_dir, complete=True)
    if discovery is not None and type(discovery) is not RuntimeDependencyDiscoveryReceiptV1:
        raise PublicAuthClosureError("readiness discovery receipt is not typed")
    contamination = tuple(
        sorted(
            f"{root.name}/{relative}"
            for root in (repo_root / "upstream", runtime_dir)
            for relative in _bytecode_contamination(root)
        )
    )
    blockers: list[str] = []
    if contamination:
        blockers.append("EXECUTABLE_PYTHON_BYTECODE_CONTAMINATION")
    if discovery is None:
        blockers.append("STRICT_RUNTIME_DEPENDENCY_DISCOVERY_OWED")
    elif (
        discovery.compile_receipt_sha256 != compiled.compile_receipt.identity_sha256
        or discovery.decoder_abi_identity_sha256 != compiled.abi_closure.identity_sha256
    ):
        blockers.append("DISCOVERY_COMPILE_OR_ABI_IDENTITY_MISMATCH")
    blockers.append("AXIS_SPECIFIC_EVALUATOR_INTERPRETER_PACKAGE_NATIVE_ABI_CAPTURE_OWED")
    if platform.system() == "Darwin" and shutil.which("sandbox-exec"):
        trace_backend = "MACOS_SANDBOX_FAIL_CLOSED_PLUS_PROCESS_TRACE_V1"
        blockers.append("OFFICIAL_AUTHORITY_REQUIRES_LINUX_X86_64_CONTEST_HARDWARE")
    elif platform.system() == "Linux" and shutil.which("strace"):
        trace_backend = "LINUX_STRACE_FILE_PROCESS_ALLOWLIST_V1"
    else:
        trace_backend = "UNAVAILABLE"
        blockers.append("WHOLE_PROCESS_TRACE_OR_FAIL_CLOSED_SANDBOX_BACKEND_UNAVAILABLE")
    owed = tuple(
        sorted(
            {
                "ACTUAL_38_BATCH_SCORER_INPUT_LEDGER",
                "ACTUAL_N600_SCORER_OUTPUT_CELL_LEDGER",
                "DOUBLE_RUN_RAW_SCORER_CLOSURE_EQUALITY",
                "EXACT_PUBLIC_N600_OFFICIAL_RUN_A",
                "EXACT_PUBLIC_N600_OFFICIAL_RUN_B",
                "EXTERNAL_GOVERNED_EXECUTION_EVIDENCE_BOUNDARY",
                "OFFICIAL_GITHUB_TEST_JOB_30_MINUTE_WALL_RECEIPT",
                "REVIEWED_SCORER_INPUT_OUTPUT_OBSERVATION_MIRROR_EQUIVALENCE",
                "SEALED_C0B_AUTH_EVAL_ADAPTER_REVIEW",
            }
        )
    )
    return AuthClosureExecutionReadinessReceiptV1(
        compile_receipt_sha256=compiled.compile_receipt.identity_sha256,
        decoder_abi_identity_sha256=compiled.abi_closure.identity_sha256,
        runtime_tree_sha256=compiled.compile_receipt.runtime_tree_sha256,
        dependency_discovery_receipt_sha256=None if discovery is None else discovery.identity_sha256,
        expected_upstream_snapshot_sha256=EXPECTED_UPSTREAM_SNAPSHOT_SHA256,
        trace_backend=trace_backend,
        bytecode_contamination_paths=contamination,
        preflight_blockers=tuple(sorted(set(blockers))),
        authority_owed=owed,
        ready_to_execute=not blockers,
        auth_closure_proven=False,
        research_only=True,
    )


@dataclass(frozen=True, slots=True)
class AuthClosureCheckpointArtifactV1:
    artifact_kind: str
    relative_path: str
    content_sha256: str
    nbytes: int

    def __post_init__(self) -> None:
        _require_identifier(self.artifact_kind, label="checkpoint artifact kind")
        _relative_path(self.relative_path, label="checkpoint artifact path")
        _require_sha256(self.content_sha256, label="checkpoint artifact")
        if type(self.nbytes) is not int or self.nbytes < 1:
            raise PublicAuthClosureError("checkpoint artifact bytes must be positive int")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "content_sha256": self.content_sha256,
            "nbytes": self.nbytes,
            "relative_path": self.relative_path,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != {
            "artifact_kind",
            "content_sha256",
            "nbytes",
            "relative_path",
        }:
            raise PublicAuthClosureError("checkpoint artifact fields drifted")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class AuthClosureStageCheckpointV1:
    run_id: str
    stage: AuthClosureStageV1
    stage_ordinal: int
    previous_checkpoint_sha256: str | None
    artifacts: tuple[AuthClosureCheckpointArtifactV1, ...]
    blockers: tuple[str, ...]
    completed: bool
    research_only: bool
    cleanup_certification_sha256: str | None
    schema: str = field(default=STAGE_CHECKPOINT_SCHEMA, init=False)

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "artifacts",
            "blockers",
            "cleanup_certification_sha256",
            "completed",
            "previous_checkpoint_sha256",
            "research_only",
            "run_id",
            "schema",
            "stage",
            "stage_ordinal",
        }
    )

    def __post_init__(self) -> None:
        _require_identifier(self.run_id, label="auth-closure run ID")
        if type(self.stage) is not AuthClosureStageV1:
            raise PublicAuthClosureError("auth-closure checkpoint stage must be typed")
        if type(self.stage_ordinal) is not int or self.stage_ordinal < 0:
            raise PublicAuthClosureError("auth-closure stage ordinal must be nonnegative int")
        if self.previous_checkpoint_sha256 is not None:
            _require_sha256(self.previous_checkpoint_sha256, label="previous auth-closure checkpoint")
        if (
            type(self.artifacts) is not tuple
            or any(type(item) is not AuthClosureCheckpointArtifactV1 for item in self.artifacts)
            or self.artifacts
            != tuple(sorted(self.artifacts, key=lambda item: (item.artifact_kind, item.relative_path)))
            or len({(item.artifact_kind, item.relative_path) for item in self.artifacts}) != len(self.artifacts)
        ):
            raise PublicAuthClosureError("checkpoint artifacts must be a unique canonical typed tuple")
        if (
            type(self.blockers) is not tuple
            or self.blockers != tuple(sorted(set(self.blockers)))
            or any(type(item) is not str or not item.isascii() for item in self.blockers)
        ):
            raise PublicAuthClosureError("checkpoint blockers must be a canonical ASCII tuple")
        if type(self.completed) is not bool or type(self.research_only) is not bool:
            raise PublicAuthClosureError("checkpoint completion/research flags must be bool")
        if self.completed and self.blockers:
            raise PublicAuthClosureError("completed checkpoint cannot retain blockers")
        if self.cleanup_certification_sha256 is not None:
            _require_sha256(self.cleanup_certification_sha256, label="checkpoint cleanup certification")
        if self.stage is AuthClosureStageV1.FINALIZE_AUTH_CLOSURE and self.completed and self.research_only:
            raise PublicAuthClosureError("final authority closure cannot remain research-only")

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(
            {
                "artifacts": [item.to_dict() for item in self.artifacts],
                "blockers": list(self.blockers),
                "cleanup_certification_sha256": self.cleanup_certification_sha256,
                "completed": self.completed,
                "previous_checkpoint_sha256": self.previous_checkpoint_sha256,
                "research_only": self.research_only,
                "run_id": self.run_id,
                "schema": self.schema,
                "stage": self.stage.value,
                "stage_ordinal": self.stage_ordinal,
            }
        )

    @classmethod
    def from_receipt_bytes(cls, payload: bytes) -> Self:
        value = _parse_canonical_object(payload, exact_keys=cls._KEYS)
        if value["schema"] != STAGE_CHECKPOINT_SCHEMA:
            raise PublicAuthClosureError("auth-closure stage checkpoint schema drifted")
        return cls(
            run_id=value["run_id"],
            stage=AuthClosureStageV1(value["stage"]),
            stage_ordinal=value["stage_ordinal"],
            previous_checkpoint_sha256=value["previous_checkpoint_sha256"],
            artifacts=tuple(AuthClosureCheckpointArtifactV1.from_dict(row) for row in value["artifacts"]),
            blockers=tuple(value["blockers"]),
            completed=value["completed"],
            research_only=value["research_only"],
            cleanup_certification_sha256=value["cleanup_certification_sha256"],
        )

    def write_atomic(self, path: Path) -> None:
        _atomic_write(path, self.to_receipt_bytes())


@dataclass(frozen=True, slots=True, init=False)
class PublicEvaluatorExecutionReceiptV1:
    """Sealed two-run official evaluator execution receipt.

    It is derivable only from two sealed run receipts, their exact equality
    receipt, the compile/discovery identities, and whole-process trace closure.
    The type is evidence for a reviewed C0B adapter; it is not itself a
    constructor for C0B authority.
    """

    archive_sha256: str
    archive_nbytes: int
    upstream_snapshot_sha256: str
    compile_receipt_sha256: str
    dependency_discovery_receipt_sha256: str
    abi_identity_sha256: str
    equality_receipt_sha256: str
    run_a_receipt_sha256: str
    run_b_receipt_sha256: str
    trace_a_receipt_sha256: str
    trace_b_receipt_sha256: str
    abi_closure_receipt_ascii: str = field(repr=False)
    run_a_receipt_ascii: str = field(repr=False)
    run_b_receipt_ascii: str = field(repr=False)
    equality_receipt_ascii: str = field(repr=False)
    trace_a_receipt_ascii: str = field(repr=False)
    trace_b_receipt_ascii: str = field(repr=False)
    raw_sha256: str
    raw_nbytes: int
    static_authority_input_file_manifest_sha256: str
    scorer_input_batch_content_sha256: str
    scorer_output_cell_content_sha256: str
    scorer_candidate_cell_content_sha256: str
    run_a_scorer_output_cell_ledger_receipt_sha256: str
    run_b_scorer_output_cell_ledger_receipt_sha256: str
    execution_axis: ExecutionAxisV1
    avg_segnet_dist: float
    avg_posenet_dist: float
    report_component_recomputed_score: float
    reported_final_score_decimal: str
    exact_argv_sha256: str
    exact_environment_sha256: str
    normalized_observed_read_set_sha256: str
    observed_exec_set_sha256: str
    output_set_sha256: str
    observed_runtime_paths: tuple[str, ...]
    total_workflow_seconds_max: float
    decode_wall_seconds_max: float
    peak_process_tree_memory_nbytes_max: int
    peak_device_memory_nbytes_max: int
    executed_public_entrypoint_path: str
    exact_runtime_file_observation_closed: bool
    exact_double_run_equal: bool
    bytecode_contamination_paths: tuple[str, ...]
    research_only: bool
    schema: str = field(default=PUBLIC_EXECUTION_SCHEMA, init=False)

    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "abi_identity_sha256",
            "archive_nbytes",
            "archive_sha256",
            "abi_closure_receipt_ascii",
            "static_authority_input_file_manifest_sha256",
            "scorer_input_batch_content_sha256",
            "scorer_output_cell_content_sha256",
            "scorer_candidate_cell_content_sha256",
            "run_a_scorer_output_cell_ledger_receipt_sha256",
            "run_b_scorer_output_cell_ledger_receipt_sha256",
            "avg_posenet_dist",
            "avg_segnet_dist",
            "bytecode_contamination_paths",
            "compile_receipt_sha256",
            "dependency_discovery_receipt_sha256",
            "decode_wall_seconds_max",
            "equality_receipt_sha256",
            "equality_receipt_ascii",
            "exact_argv_sha256",
            "exact_double_run_equal",
            "exact_environment_sha256",
            "exact_runtime_file_observation_closed",
            "executed_public_entrypoint_path",
            "execution_axis",
            "observed_exec_set_sha256",
            "normalized_observed_read_set_sha256",
            "observed_runtime_paths",
            "output_set_sha256",
            "peak_process_tree_memory_nbytes_max",
            "peak_device_memory_nbytes_max",
            "raw_nbytes",
            "raw_sha256",
            "report_component_recomputed_score",
            "reported_final_score_decimal",
            "research_only",
            "run_a_receipt_sha256",
            "run_b_receipt_sha256",
            "run_a_receipt_ascii",
            "run_b_receipt_ascii",
            "schema",
            "total_workflow_seconds_max",
            "trace_a_receipt_sha256",
            "trace_b_receipt_sha256",
            "trace_a_receipt_ascii",
            "trace_b_receipt_ascii",
            "upstream_snapshot_sha256",
        }
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise PublicAuthClosureError("PublicEvaluatorExecutionReceiptV1 has no public constructor")

    @classmethod
    def _construct(cls, *, seal: object, **values: object) -> Self:
        return _sealed_frozen_dataclass(
            cls,
            seal=seal,
            allowed_seals=(_PUBLIC_EQUALITY_DERIVATION_SEAL, _STRICT_RECEIPT_REOPEN_SEAL),
            values=values,
        )

    def __post_init__(self) -> None:
        for value, label in (
            (self.archive_sha256, "public execution archive"),
            (self.upstream_snapshot_sha256, "public execution upstream"),
            (self.compile_receipt_sha256, "public execution compile"),
            (self.dependency_discovery_receipt_sha256, "public execution discovery"),
            (self.abi_identity_sha256, "public execution ABI"),
            (self.equality_receipt_sha256, "public execution equality"),
            (self.run_a_receipt_sha256, "public execution run A"),
            (self.run_b_receipt_sha256, "public execution run B"),
            (self.trace_a_receipt_sha256, "public execution trace A"),
            (self.trace_b_receipt_sha256, "public execution trace B"),
            (self.raw_sha256, "public execution raw"),
            (
                self.static_authority_input_file_manifest_sha256,
                "public execution static authority input file manifest",
            ),
            (self.scorer_input_batch_content_sha256, "public execution scorer input batch content"),
            (self.scorer_output_cell_content_sha256, "public execution scorer output cell content"),
            (self.scorer_candidate_cell_content_sha256, "public execution scorer candidate cells"),
            (
                self.run_a_scorer_output_cell_ledger_receipt_sha256,
                "public execution run A scorer output ledger",
            ),
            (
                self.run_b_scorer_output_cell_ledger_receipt_sha256,
                "public execution run B scorer output ledger",
            ),
            (self.exact_argv_sha256, "public execution argv"),
            (self.exact_environment_sha256, "public execution environment"),
            (self.normalized_observed_read_set_sha256, "public execution normalized read set"),
            (self.observed_exec_set_sha256, "public execution exec set"),
            (self.output_set_sha256, "public execution output set"),
        ):
            _require_sha256(value, label=label)
        for value, label in (
            (self.abi_closure_receipt_ascii, "embedded ABI closure"),
            (self.run_a_receipt_ascii, "embedded official run A"),
            (self.run_b_receipt_ascii, "embedded official run B"),
            (self.equality_receipt_ascii, "embedded public equality"),
            (self.trace_a_receipt_ascii, "embedded public trace A"),
            (self.trace_b_receipt_ascii, "embedded public trace B"),
        ):
            _require_ascii(value, label=label)
        abi = InterpreterDistributionABIClosureV1.from_receipt_bytes(self.abi_closure_receipt_ascii.encode("ascii"))
        _require_axis_specific_evaluator_abi(abi, execution_axis=self.execution_axis)
        run_a = OfficialEvaluationRunReceiptV1.from_receipt_bytes(self.run_a_receipt_ascii.encode("ascii"))
        run_b = OfficialEvaluationRunReceiptV1.from_receipt_bytes(self.run_b_receipt_ascii.encode("ascii"))
        equality = PublicDecodeEqualityReceiptV1.from_receipt_bytes(self.equality_receipt_ascii.encode("ascii"))
        trace_a = PublicTraceClosureReceiptV1.from_receipt_bytes(self.trace_a_receipt_ascii.encode("ascii"))
        trace_b = PublicTraceClosureReceiptV1.from_receipt_bytes(self.trace_b_receipt_ascii.encode("ascii"))
        if (
            abi.identity_sha256 != self.abi_identity_sha256
            or run_a.identity_sha256 != self.run_a_receipt_sha256
            or run_b.identity_sha256 != self.run_b_receipt_sha256
            or equality.identity_sha256 != self.equality_receipt_sha256
            or trace_a.identity_sha256 != self.trace_a_receipt_sha256
            or trace_b.identity_sha256 != self.trace_b_receipt_sha256
        ):
            raise PublicAuthClosureError("public execution embedded parent receipt identity drifted")
        if PublicDecodeEqualityReceiptV1.from_runs(run_a, run_b).to_receipt_bytes() != equality.to_receipt_bytes():
            raise PublicAuthClosureError("public execution embedded equality does not derive from embedded runs")
        equality_bound_values = (
            self.archive_sha256 == equality.archive_sha256,
            self.archive_nbytes == equality.archive_nbytes,
            self.upstream_snapshot_sha256 == equality.upstream_snapshot_sha256,
            self.compile_receipt_sha256 == equality.compile_receipt_sha256,
            self.dependency_discovery_receipt_sha256 == equality.dependency_discovery_receipt_sha256,
            self.abi_identity_sha256 == equality.abi_identity_sha256,
            self.raw_sha256 == equality.raw_sha256,
            self.raw_nbytes == equality.raw_nbytes,
            self.static_authority_input_file_manifest_sha256 == equality.static_authority_input_file_manifest_sha256,
            self.scorer_input_batch_content_sha256 == equality.scorer_input_batch_content_sha256,
            self.scorer_output_cell_content_sha256 == equality.scorer_output_cell_content_sha256,
            self.scorer_candidate_cell_content_sha256 == equality.scorer_candidate_cell_content_sha256,
            self.run_a_scorer_output_cell_ledger_receipt_sha256
            == equality.run_a_scorer_output_cell_ledger_receipt_sha256,
            self.run_b_scorer_output_cell_ledger_receipt_sha256
            == equality.run_b_scorer_output_cell_ledger_receipt_sha256,
            self.execution_axis is equality.execution_axis,
            self.avg_segnet_dist == equality.avg_segnet_dist,
            self.avg_posenet_dist == equality.avg_posenet_dist,
            self.report_component_recomputed_score == equality.report_component_recomputed_score,
            self.reported_final_score_decimal == equality.reported_final_score_decimal,
            self.exact_argv_sha256 == equality.exact_argv_sha256,
            self.exact_environment_sha256 == equality.exact_environment_sha256,
            self.normalized_observed_read_set_sha256 == equality.normalized_observed_read_set_sha256,
            self.observed_exec_set_sha256 == equality.observed_exec_set_sha256,
            self.output_set_sha256 == equality.output_set_sha256,
            self.total_workflow_seconds_max == equality.total_workflow_seconds_max,
            self.decode_wall_seconds_max == equality.decode_wall_seconds_max,
            self.peak_process_tree_memory_nbytes_max == equality.peak_process_tree_memory_nbytes_max,
            self.peak_device_memory_nbytes_max == max(run_a.peak_device_memory_nbytes, run_b.peak_device_memory_nbytes),
        )
        if not all(equality_bound_values):
            raise PublicAuthClosureError("public execution top-level fields drifted from embedded equality")
        if (
            trace_a.normalized_observed_read_set_sha256 != self.normalized_observed_read_set_sha256
            or trace_b.normalized_observed_read_set_sha256 != self.normalized_observed_read_set_sha256
            or trace_a.observed_exec_set_sha256 != self.observed_exec_set_sha256
            or trace_b.observed_exec_set_sha256 != self.observed_exec_set_sha256
        ):
            raise PublicAuthClosureError("public execution embedded trace observations drifted")
        if (
            type(self.observed_runtime_paths) is not tuple
            or not self.observed_runtime_paths
            or self.observed_runtime_paths != tuple(sorted(set(self.observed_runtime_paths)))
        ):
            raise PublicAuthClosureError("public execution observed runtime paths are not canonical")
        for value in self.observed_runtime_paths:
            _relative_path(value, label="public execution observed runtime path")
        if type(self.archive_nbytes) is not int or self.archive_nbytes < 1:
            raise PublicAuthClosureError("public execution archive bytes must be positive int")
        if self.upstream_snapshot_sha256 != EXPECTED_UPSTREAM_SNAPSHOT_SHA256:
            raise PublicAuthClosureError("public execution upstream snapshot drifted")
        if self.raw_nbytes != EXPECTED_RAW_NBYTES:
            raise PublicAuthClosureError("public execution raw contract drifted")
        if type(self.execution_axis) is not ExecutionAxisV1:
            raise PublicAuthClosureError("public execution axis must be typed")
        for value, label in (
            (self.avg_segnet_dist, "public execution SegNet"),
            (self.avg_posenet_dist, "public execution PoseNet"),
            (self.report_component_recomputed_score, "public execution report-component score"),
            (self.total_workflow_seconds_max, "public execution total workflow time"),
            (self.decode_wall_seconds_max, "public execution decode wall time"),
        ):
            _require_finite_nonnegative(value, label=label)
        if self.total_workflow_seconds_max > OFFICIAL_TOTAL_WORKFLOW_SECONDS:
            raise PublicAuthClosureError("public execution exceeded official 30-minute total workflow")
        if self.decode_wall_seconds_max > self.total_workflow_seconds_max:
            raise PublicAuthClosureError("public execution decode time exceeds total workflow time")
        host_limit = (
            CONTEST_MEMORY_NBYTES if self.execution_axis is ExecutionAxisV1.CPU else CONTEST_CUDA_HOST_MEMORY_NBYTES
        )
        if (
            type(self.peak_process_tree_memory_nbytes_max) is not int
            or not 1 <= self.peak_process_tree_memory_nbytes_max <= host_limit
        ):
            raise PublicAuthClosureError("public execution peak memory is outside the contest envelope")
        if type(self.peak_device_memory_nbytes_max) is not int or self.peak_device_memory_nbytes_max < 0:
            raise PublicAuthClosureError("public execution device-memory maximum must be nonnegative int")
        if self.execution_axis is ExecutionAxisV1.CPU and self.peak_device_memory_nbytes_max != 0:
            raise PublicAuthClosureError("contest-CPU execution cannot report CUDA device memory")
        if (
            self.execution_axis is ExecutionAxisV1.CUDA
            and not 1 <= self.peak_device_memory_nbytes_max <= CONTEST_CUDA_DEVICE_MEMORY_NBYTES
        ):
            raise PublicAuthClosureError("contest-CUDA execution lacks bounded T4 device-memory proof")
        if self.executed_public_entrypoint_path != "upstream/evaluate.sh":
            raise PublicAuthClosureError("public execution root is not upstream/evaluate.sh")
        if self.exact_runtime_file_observation_closed is not True or self.exact_double_run_equal is not True:
            raise PublicAuthClosureError("public execution lacks exact runtime observation or double-run equality")
        if self.bytecode_contamination_paths:
            raise PublicAuthClosureError("public execution contains Python bytecode contamination")
        if self.research_only is not True:
            raise PublicAuthClosureError("caller-authored/reopened execution packets cannot mint contest authority")

    @classmethod
    def from_verified_runs(
        cls,
        *,
        compiled: CompiledPublicRuntimeV1,
        discovery: RuntimeDependencyDiscoveryReceiptV1,
        run_a: OfficialEvaluationRunReceiptV1,
        run_b: OfficialEvaluationRunReceiptV1,
        equality: PublicDecodeEqualityReceiptV1,
        trace_a: PublicTraceClosureReceiptV1,
        trace_b: PublicTraceClosureReceiptV1,
        evaluator_abi_closure: InterpreterDistributionABIClosureV1,
        repo_root: Path,
        runtime_dir: Path,
    ) -> Self:
        if (
            type(compiled) is not CompiledPublicRuntimeV1
            or type(discovery) is not RuntimeDependencyDiscoveryReceiptV1
            or type(run_a) is not OfficialEvaluationRunReceiptV1
            or type(run_b) is not OfficialEvaluationRunReceiptV1
            or type(equality) is not PublicDecodeEqualityReceiptV1
            or type(trace_a) is not PublicTraceClosureReceiptV1
            or type(trace_b) is not PublicTraceClosureReceiptV1
            or type(evaluator_abi_closure) is not InterpreterDistributionABIClosureV1
        ):
            raise PublicAuthClosureError("public execution derivation requires exact typed parents")
        if PublicDecodeEqualityReceiptV1.from_runs(run_a, run_b).to_receipt_bytes() != equality.to_receipt_bytes():
            raise PublicAuthClosureError("public execution equality receipt does not derive from retained runs")
        if (
            equality.archive_sha256 != compiled.compile_receipt.archive_sha256
            or equality.archive_nbytes != compiled.compile_receipt.archive_nbytes
            or equality.compile_receipt_sha256 != compiled.compile_receipt.identity_sha256
            or equality.abi_identity_sha256 != evaluator_abi_closure.identity_sha256
            or equality.dependency_discovery_receipt_sha256 != discovery.identity_sha256
            or equality.static_authority_input_file_manifest_sha256
            != discovery.static_authority_input_file_manifest_sha256
            or discovery.compile_receipt_sha256 != compiled.compile_receipt.identity_sha256
            or discovery.decoder_abi_identity_sha256 != compiled.abi_closure.identity_sha256
            or not evaluator_abi_closure.closed
        ):
            raise PublicAuthClosureError("public execution parent archive/compile/discovery/ABI identity drifted")
        _require_axis_specific_evaluator_abi(
            evaluator_abi_closure,
            execution_axis=equality.execution_axis,
        )
        current_prefix_sha256, current_prefix_nbytes = _complete_prefix_tree_identity(
            evaluator_abi_closure.interpreter_prefix_realpaths
        )
        if (
            current_prefix_sha256 != evaluator_abi_closure.interpreter_prefix_tree_sha256
            or current_prefix_nbytes != evaluator_abi_closure.interpreter_prefix_tree_nbytes
        ):
            raise PublicAuthClosureError("evaluator interpreter prefix tree mutated after ABI capture")
        discovered_interpreter = next(item for item in discovery.system_tools if item.role == "interpreter.python")
        if (
            discovered_interpreter.executable_realpath != evaluator_abi_closure.interpreter_executable_realpath
            or discovered_interpreter.executable_sha256 != evaluator_abi_closure.interpreter_executable_sha256
        ):
            raise PublicAuthClosureError("discovered Python executable drifted from evaluator ABI")
        if (
            trace_a.observed_read_set_sha256 != run_a.observed_read_set_sha256
            or trace_b.observed_read_set_sha256 != run_b.observed_read_set_sha256
            or trace_a.observed_exec_set_sha256 != run_a.observed_exec_set_sha256
            or trace_b.observed_exec_set_sha256 != run_b.observed_exec_set_sha256
            or trace_a.raw_trace_sha256 != run_a.syscall_trace_sha256
            or trace_b.raw_trace_sha256 != run_b.syscall_trace_sha256
        ):
            raise PublicAuthClosureError("public execution trace receipts do not bind sealed run observations")
        runtime_digests = {item.relative_path: item.content_sha256 for item in compiled.compile_receipt.runtime_files}
        for run in (run_a, run_b):
            inverse_trace = PublicInverseTraceReceiptV1.from_receipt_bytes(
                run.inverse_trace_receipt_ascii.encode("ascii")
            )
            if (
                inverse_trace.inverse_source_sha256 != runtime_digests[PUBLIC_INFLATE_PY_PATH]
                or inverse_trace.runtime_sha256 != runtime_digests[PUBLIC_LVLS1_RUNTIME_PATH]
                or inverse_trace.source_member_sha256 != compiled.compile_receipt.member_sha256
                or inverse_trace.logical_lvls1_sha256 != compiled.compile_receipt.materialized_lvls1_sha256
                or inverse_trace.output_raw_sha256 != run.raw_sha256
            ):
                raise PublicAuthClosureError("public execution inverse trace drifted from compiled runtime")
        repo_root = repo_root.resolve(strict=True)
        runtime_dir = runtime_dir.resolve(strict=True)
        required_runtime_paths = {
            (repo_root / row.relative_path).resolve().as_posix()
            if row.relative_path.startswith("upstream/")
            else (runtime_dir / row.relative_path).resolve().as_posix()
            for row in (
                *discovery.runtime_files,
                *discovery.authority_input_files,
                *discovery.environment_lock_files,
            )
        }
        common_reads = set(trace_a.observed_read_paths) & set(trace_b.observed_read_paths)
        if not required_runtime_paths.issubset(common_reads):
            raise PublicAuthClosureError("public execution traces do not observe every recursive runtime file")
        authority_input_paths = {
            (repo_root / row.relative_path).resolve().as_posix() for row in discovery.authority_input_files
        }
        evaluator_runtime_paths = {
            (repo_root / row.relative_path).resolve().as_posix()
            for row in discovery.runtime_files
            if row.relative_path
            in {
                "upstream/evaluate.py",
                "upstream/frame_utils.py",
                "upstream/modules.py",
            }
        }
        decoder_runtime_paths = {
            (runtime_dir / row.relative_path).resolve().as_posix()
            for row in discovery.runtime_files
            if row.relative_path in {PUBLIC_INFLATE_PY_PATH, PUBLIC_LVLS1_RUNTIME_PATH}
        }
        environment_lock_paths = {
            (repo_root / row.relative_path).resolve().as_posix() for row in discovery.environment_lock_files
        }
        permitted_exec_paths = {item.executable_realpath for item in discovery.system_tools} | {
            evaluator_abi_closure.interpreter_executable_realpath
        }
        owned_exact_read_paths = required_runtime_paths | permitted_exec_paths
        abi_read_roots = tuple(Path(value).resolve() for value in evaluator_abi_closure.interpreter_prefix_realpaths)
        for trace, run in ((trace_a, run_a), (trace_b, run_b)):
            decoder_reads = set(trace.decoder_observed_read_paths)
            evaluator_reads = set(trace.evaluator_observed_read_paths)
            orchestrator_reads = set(trace.orchestrator_observed_read_paths)
            if authority_input_paths & decoder_reads:
                raise PublicAuthClosureError("decoder phase read GT/scorer authority inputs")
            if not authority_input_paths.issubset(evaluator_reads):
                raise PublicAuthClosureError("evaluator phase did not read every authority input")
            if not evaluator_runtime_paths.issubset(evaluator_reads):
                raise PublicAuthClosureError("evaluator phase attribution omitted evaluator runtime files")
            if not decoder_runtime_paths.issubset(decoder_reads):
                raise PublicAuthClosureError("decoder phase attribution omitted public decoder runtime files")
            if not environment_lock_paths.issubset(orchestrator_reads):
                raise PublicAuthClosureError("outer official workflow did not observe all environment locks")
            if set(trace.allowed_exec_paths) != permitted_exec_paths:
                raise PublicAuthClosureError("trace executable allowlist drifted from byte-owned system tools")
            if not set(trace.observed_exec_paths).issubset(permitted_exec_paths):
                raise PublicAuthClosureError("trace executed an unowned helper")
            accounted = {item.absolute_path: item for item in trace.accounted_external_reads}
            observed_outside_owned = {
                value
                for value in trace.observed_read_paths
                if value not in owned_exact_read_paths
                and not any(
                    Path(value).resolve() == root or root in Path(value).resolve().parents for root in abi_read_roots
                )
            }
            if set(accounted) != observed_outside_owned:
                raise PublicAuthClosureError("trace contains unowned or spurious externally-accounted reads")
            inverse_trace = PublicInverseTraceReceiptV1.from_receipt_bytes(
                run.inverse_trace_receipt_ascii.encode("ascii")
            )
            retained_run_digests = {
                compiled.compile_receipt.archive_sha256,
                compiled.compile_receipt.member_sha256,
                compiled.compile_receipt.materialized_lvls1_sha256,
                run.raw_sha256,
                run.report_sha256,
                run.inverse_trace_sha256,
                run.process_trace_sha256,
                run.syscall_trace_sha256,
                inverse_trace.inverse_source_sha256,
                inverse_trace.runtime_sha256,
            }
            for item in accounted.values():
                if item.kind is ExternalReadKindV1.VIRTUAL_KERNEL_FILE:
                    continue
                path = Path(item.absolute_path)
                if path.is_file() and not path.is_symlink():
                    if path.stat().st_size != item.nbytes or _sha256_file(path) != item.content_sha256:
                        raise PublicAuthClosureError("accounted external regular file bytes drifted")
                elif (
                    item.kind is not ExternalReadKindV1.RUN_SCOPED_REGULAR_FILE
                    or item.content_sha256 not in retained_run_digests
                ):
                    raise PublicAuthClosureError("accounted external regular read lacks retained byte custody")
            decoder_run_scoped_reads = {
                item.absolute_path
                for item in accounted.values()
                if item.kind is ExternalReadKindV1.RUN_SCOPED_REGULAR_FILE
                and item.content_sha256 in retained_run_digests
            }
            decoder_allowed_reads = (
                decoder_runtime_paths
                | decoder_run_scoped_reads
                | {
                    value
                    for value in decoder_reads
                    if any(
                        Path(value).resolve() == root or root in Path(value).resolve().parents
                        for root in abi_read_roots
                    )
                }
            )
            if not decoder_reads.issubset(decoder_allowed_reads):
                raise PublicAuthClosureError(
                    "decoder phase read outside exact runtime, counted/run-scoped bytes, or full ABI tree"
                )
        return cls._construct(
            seal=_PUBLIC_EQUALITY_DERIVATION_SEAL,
            archive_sha256=equality.archive_sha256,
            archive_nbytes=equality.archive_nbytes,
            upstream_snapshot_sha256=equality.upstream_snapshot_sha256,
            compile_receipt_sha256=equality.compile_receipt_sha256,
            dependency_discovery_receipt_sha256=equality.dependency_discovery_receipt_sha256,
            abi_identity_sha256=equality.abi_identity_sha256,
            equality_receipt_sha256=equality.identity_sha256,
            run_a_receipt_sha256=run_a.identity_sha256,
            run_b_receipt_sha256=run_b.identity_sha256,
            trace_a_receipt_sha256=trace_a.identity_sha256,
            trace_b_receipt_sha256=trace_b.identity_sha256,
            abi_closure_receipt_ascii=evaluator_abi_closure.to_receipt_bytes().decode("ascii"),
            run_a_receipt_ascii=run_a.to_receipt_bytes().decode("ascii"),
            run_b_receipt_ascii=run_b.to_receipt_bytes().decode("ascii"),
            equality_receipt_ascii=equality.to_receipt_bytes().decode("ascii"),
            trace_a_receipt_ascii=trace_a.to_receipt_bytes().decode("ascii"),
            trace_b_receipt_ascii=trace_b.to_receipt_bytes().decode("ascii"),
            raw_sha256=equality.raw_sha256,
            raw_nbytes=equality.raw_nbytes,
            static_authority_input_file_manifest_sha256=(equality.static_authority_input_file_manifest_sha256),
            scorer_input_batch_content_sha256=equality.scorer_input_batch_content_sha256,
            scorer_output_cell_content_sha256=equality.scorer_output_cell_content_sha256,
            scorer_candidate_cell_content_sha256=equality.scorer_candidate_cell_content_sha256,
            run_a_scorer_output_cell_ledger_receipt_sha256=(equality.run_a_scorer_output_cell_ledger_receipt_sha256),
            run_b_scorer_output_cell_ledger_receipt_sha256=(equality.run_b_scorer_output_cell_ledger_receipt_sha256),
            execution_axis=equality.execution_axis,
            avg_segnet_dist=equality.avg_segnet_dist,
            avg_posenet_dist=equality.avg_posenet_dist,
            report_component_recomputed_score=equality.report_component_recomputed_score,
            reported_final_score_decimal=equality.reported_final_score_decimal,
            exact_argv_sha256=equality.exact_argv_sha256,
            exact_environment_sha256=equality.exact_environment_sha256,
            normalized_observed_read_set_sha256=(equality.normalized_observed_read_set_sha256),
            observed_exec_set_sha256=equality.observed_exec_set_sha256,
            output_set_sha256=equality.output_set_sha256,
            observed_runtime_paths=tuple(sorted(row.relative_path for row in discovery.runtime_files)),
            total_workflow_seconds_max=equality.total_workflow_seconds_max,
            decode_wall_seconds_max=equality.decode_wall_seconds_max,
            peak_process_tree_memory_nbytes_max=equality.peak_process_tree_memory_nbytes_max,
            peak_device_memory_nbytes_max=max(
                run_a.peak_device_memory_nbytes,
                run_b.peak_device_memory_nbytes,
            ),
            executed_public_entrypoint_path="upstream/evaluate.sh",
            exact_runtime_file_observation_closed=True,
            exact_double_run_equal=True,
            bytecode_contamination_paths=(),
            research_only=True,
        )

    @property
    def identity_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(
            {
                "abi_identity_sha256": self.abi_identity_sha256,
                "abi_closure_receipt_ascii": self.abi_closure_receipt_ascii,
                "archive_nbytes": self.archive_nbytes,
                "archive_sha256": self.archive_sha256,
                "static_authority_input_file_manifest_sha256": (self.static_authority_input_file_manifest_sha256),
                "scorer_input_batch_content_sha256": self.scorer_input_batch_content_sha256,
                "scorer_output_cell_content_sha256": self.scorer_output_cell_content_sha256,
                "scorer_candidate_cell_content_sha256": self.scorer_candidate_cell_content_sha256,
                "run_a_scorer_output_cell_ledger_receipt_sha256": (self.run_a_scorer_output_cell_ledger_receipt_sha256),
                "run_b_scorer_output_cell_ledger_receipt_sha256": (self.run_b_scorer_output_cell_ledger_receipt_sha256),
                "avg_posenet_dist": self.avg_posenet_dist,
                "avg_segnet_dist": self.avg_segnet_dist,
                "bytecode_contamination_paths": list(self.bytecode_contamination_paths),
                "compile_receipt_sha256": self.compile_receipt_sha256,
                "dependency_discovery_receipt_sha256": self.dependency_discovery_receipt_sha256,
                "decode_wall_seconds_max": self.decode_wall_seconds_max,
                "equality_receipt_sha256": self.equality_receipt_sha256,
                "equality_receipt_ascii": self.equality_receipt_ascii,
                "exact_argv_sha256": self.exact_argv_sha256,
                "exact_double_run_equal": self.exact_double_run_equal,
                "exact_environment_sha256": self.exact_environment_sha256,
                "exact_runtime_file_observation_closed": self.exact_runtime_file_observation_closed,
                "executed_public_entrypoint_path": self.executed_public_entrypoint_path,
                "execution_axis": self.execution_axis.value,
                "observed_exec_set_sha256": self.observed_exec_set_sha256,
                "normalized_observed_read_set_sha256": (self.normalized_observed_read_set_sha256),
                "observed_runtime_paths": list(self.observed_runtime_paths),
                "output_set_sha256": self.output_set_sha256,
                "peak_process_tree_memory_nbytes_max": self.peak_process_tree_memory_nbytes_max,
                "peak_device_memory_nbytes_max": self.peak_device_memory_nbytes_max,
                "raw_nbytes": self.raw_nbytes,
                "raw_sha256": self.raw_sha256,
                "report_component_recomputed_score": self.report_component_recomputed_score,
                "reported_final_score_decimal": self.reported_final_score_decimal,
                "research_only": self.research_only,
                "run_a_receipt_sha256": self.run_a_receipt_sha256,
                "run_b_receipt_sha256": self.run_b_receipt_sha256,
                "run_a_receipt_ascii": self.run_a_receipt_ascii,
                "run_b_receipt_ascii": self.run_b_receipt_ascii,
                "schema": self.schema,
                "total_workflow_seconds_max": self.total_workflow_seconds_max,
                "trace_a_receipt_sha256": self.trace_a_receipt_sha256,
                "trace_b_receipt_sha256": self.trace_b_receipt_sha256,
                "trace_a_receipt_ascii": self.trace_a_receipt_ascii,
                "trace_b_receipt_ascii": self.trace_b_receipt_ascii,
                "upstream_snapshot_sha256": self.upstream_snapshot_sha256,
            }
        )

    @classmethod
    def from_receipt_bytes(cls, payload: bytes) -> Self:
        value = _parse_canonical_object(payload, exact_keys=cls._KEYS)
        if value["schema"] != PUBLIC_EXECUTION_SCHEMA:
            raise PublicAuthClosureError("public evaluator execution schema drifted")
        return cls._construct(
            seal=_STRICT_RECEIPT_REOPEN_SEAL,
            archive_sha256=value["archive_sha256"],
            archive_nbytes=value["archive_nbytes"],
            upstream_snapshot_sha256=value["upstream_snapshot_sha256"],
            compile_receipt_sha256=value["compile_receipt_sha256"],
            dependency_discovery_receipt_sha256=value["dependency_discovery_receipt_sha256"],
            abi_identity_sha256=value["abi_identity_sha256"],
            equality_receipt_sha256=value["equality_receipt_sha256"],
            run_a_receipt_sha256=value["run_a_receipt_sha256"],
            run_b_receipt_sha256=value["run_b_receipt_sha256"],
            trace_a_receipt_sha256=value["trace_a_receipt_sha256"],
            trace_b_receipt_sha256=value["trace_b_receipt_sha256"],
            abi_closure_receipt_ascii=value["abi_closure_receipt_ascii"],
            run_a_receipt_ascii=value["run_a_receipt_ascii"],
            run_b_receipt_ascii=value["run_b_receipt_ascii"],
            equality_receipt_ascii=value["equality_receipt_ascii"],
            trace_a_receipt_ascii=value["trace_a_receipt_ascii"],
            trace_b_receipt_ascii=value["trace_b_receipt_ascii"],
            raw_sha256=value["raw_sha256"],
            raw_nbytes=value["raw_nbytes"],
            static_authority_input_file_manifest_sha256=value["static_authority_input_file_manifest_sha256"],
            scorer_input_batch_content_sha256=value["scorer_input_batch_content_sha256"],
            scorer_output_cell_content_sha256=value["scorer_output_cell_content_sha256"],
            scorer_candidate_cell_content_sha256=value["scorer_candidate_cell_content_sha256"],
            run_a_scorer_output_cell_ledger_receipt_sha256=value["run_a_scorer_output_cell_ledger_receipt_sha256"],
            run_b_scorer_output_cell_ledger_receipt_sha256=value["run_b_scorer_output_cell_ledger_receipt_sha256"],
            execution_axis=ExecutionAxisV1(value["execution_axis"]),
            avg_segnet_dist=value["avg_segnet_dist"],
            avg_posenet_dist=value["avg_posenet_dist"],
            report_component_recomputed_score=value["report_component_recomputed_score"],
            reported_final_score_decimal=value["reported_final_score_decimal"],
            exact_argv_sha256=value["exact_argv_sha256"],
            exact_environment_sha256=value["exact_environment_sha256"],
            normalized_observed_read_set_sha256=value["normalized_observed_read_set_sha256"],
            observed_exec_set_sha256=value["observed_exec_set_sha256"],
            output_set_sha256=value["output_set_sha256"],
            observed_runtime_paths=tuple(value["observed_runtime_paths"]),
            total_workflow_seconds_max=value["total_workflow_seconds_max"],
            decode_wall_seconds_max=value["decode_wall_seconds_max"],
            peak_process_tree_memory_nbytes_max=value["peak_process_tree_memory_nbytes_max"],
            peak_device_memory_nbytes_max=value["peak_device_memory_nbytes_max"],
            executed_public_entrypoint_path=value["executed_public_entrypoint_path"],
            exact_runtime_file_observation_closed=value["exact_runtime_file_observation_closed"],
            exact_double_run_equal=value["exact_double_run_equal"],
            bytecode_contamination_paths=tuple(value["bytecode_contamination_paths"]),
            research_only=value["research_only"],
        )


@dataclass(frozen=True, slots=True)
class AuthEvalAdapterIngredientsV1:
    """Exact reviewed-adapter inputs; deliberately cannot access the C0B seal."""

    runtime_files: tuple[G17RuntimeDependencyFileV1, ...]
    dependency_edges: tuple[G17RuntimeDependencyEdgeV1, ...]
    observed_runtime_paths: tuple[str, ...]
    dependency_discovery_receipt: G17ReopenedEvidencePacketV1 = field(repr=False)
    public_evaluator_execution_receipt: G17ReopenedEvidencePacketV1 = field(repr=False)
    abi_closure_receipt: G17ReopenedEvidencePacketV1 = field(repr=False)
    executed_public_entrypoint_path: str
    evaluated_archive_sha256: str
    public_decode_receipt_sha256: str
    scorer_output_cell_content_sha256: str
    scorer_candidate_cell_content_sha256: str
    exact_argv_bytes: bytes = field(repr=False)
    exact_environment_bytes: bytes = field(repr=False)
    abi_identity_sha256: str
    schema: str = field(default=ADAPTER_INGREDIENTS_SCHEMA, init=False)

    def __post_init__(self) -> None:
        if (
            type(self.runtime_files) is not tuple
            or not self.runtime_files
            or any(type(item) is not G17RuntimeDependencyFileV1 for item in self.runtime_files)
            or self.runtime_files != tuple(sorted(self.runtime_files, key=lambda item: item.relative_path))
        ):
            raise PublicAuthClosureError("adapter runtime files must be a canonical exact typed tuple")
        if (
            type(self.dependency_edges) is not tuple
            or any(type(item) is not G17RuntimeDependencyEdgeV1 for item in self.dependency_edges)
            or self.dependency_edges
            != tuple(
                sorted(
                    self.dependency_edges,
                    key=lambda item: (item.importer_path, item.dependency_path, item.mechanism.value),
                )
            )
        ):
            raise PublicAuthClosureError("adapter dependency edges must be a canonical exact typed tuple")
        expected_paths = tuple(sorted(item.relative_path for item in self.runtime_files))
        if self.observed_runtime_paths != expected_paths:
            raise PublicAuthClosureError("adapter observed paths do not exactly cover runtime custody")
        for packet, schema, label in (
            (
                self.dependency_discovery_receipt,
                DEPENDENCY_DISCOVERY_SCHEMA,
                "adapter dependency discovery",
            ),
            (
                self.public_evaluator_execution_receipt,
                PUBLIC_EXECUTION_SCHEMA,
                "adapter public execution",
            ),
            (self.abi_closure_receipt, ABI_CLOSURE_SCHEMA, "adapter ABI closure"),
        ):
            if type(packet) is not G17ReopenedEvidencePacketV1 or packet.expected_schema != schema:
                raise PublicAuthClosureError(f"{label} packet is not strict-reopened")
            packet.reopen()
        if self.executed_public_entrypoint_path != "upstream/evaluate.sh":
            raise PublicAuthClosureError("adapter execution root is not upstream/evaluate.sh")
        for digest, label in (
            (self.evaluated_archive_sha256, "adapter evaluated archive"),
            (self.public_decode_receipt_sha256, "adapter public decode receipt"),
            (self.scorer_output_cell_content_sha256, "adapter scorer output proof content"),
            (self.scorer_candidate_cell_content_sha256, "adapter semantic candidate cells"),
            (self.abi_identity_sha256, "adapter ABI identity"),
        ):
            _require_sha256(digest, label=label)
        if type(self.exact_argv_bytes) is not bytes or not self.exact_argv_bytes:
            raise PublicAuthClosureError("adapter argv must be nonempty exact bytes")
        if type(self.exact_environment_bytes) is not bytes or not self.exact_environment_bytes:
            raise PublicAuthClosureError("adapter environment must be nonempty exact bytes")

    @property
    def identity_sha256(self) -> str:
        digest = hashlib.sha256(b"PACT-G29-AUTH-EVAL-ADAPTER-INGREDIENTS-V1\x00")
        for item in self.runtime_files:
            digest.update(bytes.fromhex(item.identity_sha256))
        for edge in self.dependency_edges:
            digest.update(edge.importer_path.encode("ascii") + b"\x00")
            digest.update(edge.dependency_path.encode("ascii") + b"\x00")
            digest.update(edge.mechanism.value.encode("ascii") + b"\x00")
        for value in (
            self.dependency_discovery_receipt.packet_sha256,
            self.public_evaluator_execution_receipt.packet_sha256,
            self.abi_closure_receipt.packet_sha256,
            self.evaluated_archive_sha256,
            self.public_decode_receipt_sha256,
            self.scorer_output_cell_content_sha256,
            self.scorer_candidate_cell_content_sha256,
            self.abi_identity_sha256,
            _sha256(self.exact_argv_bytes),
            _sha256(self.exact_environment_bytes),
        ):
            digest.update(bytes.fromhex(value))
        return digest.hexdigest()

    def to_c0b_kwargs(self) -> dict[str, object]:
        """Return only public C0B constructor arguments; no private seal is exposed."""

        return {
            "runtime_files": self.runtime_files,
            "dependency_edges": self.dependency_edges,
            "observed_runtime_paths": self.observed_runtime_paths,
            "dependency_discovery_receipt": self.dependency_discovery_receipt,
            "public_evaluator_execution_receipt": self.public_evaluator_execution_receipt,
            "executed_public_entrypoint_path": self.executed_public_entrypoint_path,
            "evaluated_archive_sha256": self.evaluated_archive_sha256,
            "public_decode_receipt_sha256": self.public_decode_receipt_sha256,
            "exact_argv_bytes": self.exact_argv_bytes,
            "exact_environment_bytes": self.exact_environment_bytes,
        }


def build_auth_eval_adapter_ingredients(
    *,
    repo_root: Path,
    runtime_dir: Path,
    compiled: CompiledPublicRuntimeV1,
    discovery: RuntimeDependencyDiscoveryReceiptV1,
    execution: PublicEvaluatorExecutionReceiptV1,
    equality: PublicDecodeEqualityReceiptV1,
    exact_argv_bytes: bytes,
    exact_environment_bytes: bytes,
) -> AuthEvalAdapterIngredientsV1:
    """Build exact C0B inputs after, and only after, public double-run closure."""

    if (
        type(compiled) is not CompiledPublicRuntimeV1
        or type(discovery) is not RuntimeDependencyDiscoveryReceiptV1
        or type(execution) is not PublicEvaluatorExecutionReceiptV1
        or type(equality) is not PublicDecodeEqualityReceiptV1
    ):
        raise PublicAuthClosureError("adapter ingredients require exact typed production parents")
    if (
        execution.archive_sha256 != compiled.compile_receipt.archive_sha256
        or execution.compile_receipt_sha256 != compiled.compile_receipt.identity_sha256
        or execution.dependency_discovery_receipt_sha256 != discovery.identity_sha256
        or discovery.decoder_abi_identity_sha256 != compiled.abi_closure.identity_sha256
        or execution.equality_receipt_sha256 != equality.identity_sha256
        or execution.exact_argv_sha256 != _sha256(exact_argv_bytes)
        or execution.exact_environment_sha256 != _sha256(exact_environment_bytes)
    ):
        raise PublicAuthClosureError("adapter ingredients drifted from execution parents")
    if execution.research_only is not False or equality.research_only is not False:
        raise PublicAuthClosureError("research-only parsed observations cannot be adapted into sealed C0B authority")
    repo_root = repo_root.resolve(strict=True)
    runtime_dir = runtime_dir.resolve(strict=True)
    runtime_files: list[G17RuntimeDependencyFileV1] = []
    for row in discovery.runtime_files:
        path = (
            repo_root / row.relative_path
            if row.relative_path.startswith("upstream/")
            else runtime_dir / row.relative_path
        ).resolve(strict=True)
        payload = path.read_bytes()
        if len(payload) != row.nbytes or _sha256(payload) != row.content_sha256:
            raise PublicAuthClosureError(f"adapter runtime dependency drifted: {row.relative_path}")
        runtime_files.append(
            G17RuntimeDependencyFileV1(
                relative_path=row.relative_path,
                exact_file_bytes=payload,
                custody_owner=row.custody_owner,
                scope=row.scope,
            )
        )
    edges = tuple(
        sorted(
            (
                G17RuntimeDependencyEdgeV1(
                    importer_path=row.importer_path,
                    dependency_path=row.dependency_path,
                    mechanism=row.mechanism,
                )
                for row in discovery.dependency_edges
            ),
            key=lambda item: (item.importer_path, item.dependency_path, item.mechanism.value),
        )
    )
    discovery_packet = G17ReopenedEvidencePacketV1(
        exact_packet_bytes=discovery.to_receipt_bytes(),
        strict_parser=RuntimeDependencyDiscoveryReceiptV1.from_receipt_bytes,
        expected_schema=DEPENDENCY_DISCOVERY_SCHEMA,
    )
    execution_packet = G17ReopenedEvidencePacketV1(
        exact_packet_bytes=execution.to_receipt_bytes(),
        strict_parser=PublicEvaluatorExecutionReceiptV1.from_receipt_bytes,
        expected_schema=PUBLIC_EXECUTION_SCHEMA,
    )
    abi_packet = G17ReopenedEvidencePacketV1(
        exact_packet_bytes=execution.abi_closure_receipt_ascii.encode("ascii"),
        strict_parser=InterpreterDistributionABIClosureV1.from_receipt_bytes,
        expected_schema=ABI_CLOSURE_SCHEMA,
    )
    return AuthEvalAdapterIngredientsV1(
        runtime_files=tuple(sorted(runtime_files, key=lambda item: item.relative_path)),
        dependency_edges=edges,
        observed_runtime_paths=execution.observed_runtime_paths,
        dependency_discovery_receipt=discovery_packet,
        public_evaluator_execution_receipt=execution_packet,
        abi_closure_receipt=abi_packet,
        executed_public_entrypoint_path="upstream/evaluate.sh",
        evaluated_archive_sha256=execution.archive_sha256,
        public_decode_receipt_sha256=equality.identity_sha256,
        scorer_output_cell_content_sha256=execution.scorer_output_cell_content_sha256,
        scorer_candidate_cell_content_sha256=execution.scorer_candidate_cell_content_sha256,
        exact_argv_bytes=exact_argv_bytes,
        exact_environment_bytes=exact_environment_bytes,
        abi_identity_sha256=execution.abi_identity_sha256,
    )
