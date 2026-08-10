#!/usr/bin/env python3
"""Resumable full-n600 LC2 int12 PoseNet re-solve with exact parse-back.

This runner is the local CPU implementation for ``ddm_ps135``.  It keeps the
LC2 semantic renderer, HPAC section, token stream, temporal sidecar, receiver,
and CX2/Brotli container frozen.  Only the deployed CPR1 carrier is mutable.

Each solve pass refreshes an exact finite-difference 6x12 PoseNet Jacobian on
all 600 pairs, uses the public ExperimentBook damped-GN primitives to propose
signed-int12 lattice points, screens the GN cube plus every singleton +/-1
coordinate through the deployed renderer and frozen CPU PoseNet, and then
charges the complete parsed LC2 archive.  The aggregate candidate and every
rate-trim variant are retained before selection.  The stop is eight complete
passes minimum plus three consecutive dry passes, followed by the distinct
exact JRD step-ladder finisher.  If that finisher fires, its byte-closed state
is promoted and the GN loop resumes to convergence before JRD is tried again.

Authority boundary: all scorer numbers emitted here are ``[macOS-CPU
advisory]``.  They are full-n600 and use the exact upstream CPU scorer code,
but they are not contest-CUDA rows and cannot move the canonical pointer.

borrowed_substrate_accounting:
  mechanism: PR133 exact PoseNet-guided signed-int12 coordinate re-solve
  proposal_math: PR135 ExperimentBook joint_pose_solve.py (public/granted)
  cpu_search_shape: PR130 search_pose_coeff_cpu.py (Fesal Fayed lineage)
  optional_warm_start: complete public PR133 CPR1 carrier, never codes alone
  base_and_container: our LC2 archive and CX2/Brotli/ANS receiver
  implementation_and_retention_hardening: Pact ddm_ps135
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LC2_STACK = REPO / "experiments" / "ddm_lc2_lossless_coder_stack.py"
LC2_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_lc2_20260810")
LC2_ARCHIVE = LC2_ROOT / "submission" / "archive.zip"
LC2_RAW = Path("/Volumes/APDataStore/pact/ddm_lc2_20260810/cold_decode/0.raw")
LC2_RUNTIME = LC2_ROOT / "submission"
LC2_INPUTS = LC2_ROOT / "retained" / "inputs"
LC2_SEARCH = LC2_ROOT / "stages" / "03_search.json"
PR133_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr133/archive.zip"
)
EXPERIMENT_BOOK = Path(
    "/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book"
)
EXPERIMENT_BOOK_SRC = EXPERIMENT_BOOK / "src"
JOINT_SOLVER = EXPERIMENT_BOOK_SRC / "cpr1_sub4" / "joint_pose_solve.py"
UPSTREAM = REPO / "upstream"
DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_ps135_20260810")
DEFAULT_BULK = Path("/Volumes/APDataStore/pact/ddm_ps135_20260810")
VERTIGO_ROOT = Path("/Volumes/VertigoDataTier/pact")
FLEET_SCORER_LOCK = VERTIGO_ROOT / ".locks" / "ddm_full_n600_scorer.lock"
BROTLI_CLI = Path("/opt/homebrew/bin/brotli")
BROTLI_PYTHON = REPO / ".venv" / "bin" / "python"
PQ1_SITE_PACKAGES = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pq1_runtime_20260809/venv/"
    "lib/python3.11/site-packages"
)
JRD_PRIOR_POLICY = REPO / "src" / "tac" / "witness_dsl" / "jrd_priors.py"
JRD_PRIOR_HARVEST_JSON = (
    REPO / ".omx" / "research" / "jrd_reusable_priors_harvest_20260713.json"
)
JRD_PRIOR_HARVEST_MEMO = (
    REPO / ".omx" / "research" / "jrd_reusable_priors_harvest_20260713.md"
)
JRD_PR110_RESPONSE_CURVES = (
    REPO
    / "experiments"
    / "results"
    / "jrd_pr110_pointer_completion_20260713T023300Z"
    / "section_precision_response_curves.json"
)
JRD_PR110_MEASUREMENT_RECEIPT = JRD_PR110_RESPONSE_CURVES.with_name(
    "measurement_receipt.json"
)
STAGE_C_IMPLEMENTATION_SPEC = (
    REPO / ".omx" / "research" / "ddm_ps135_stage_c_implementation_spec.md"
)
LC2_RUNTIME_DEPENDENCIES = LC2_RUNTIME / "runtime-dependencies.json"
SR1_RECEIVER_PROOF = Path(
    "/Volumes/VertigoDataTier/pact/ddm_sr1_20260809/SR1_RECEIVER_PROOF.json"
)
LEGACY_STAGE_C_MAP_STORE = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/mixed_precision"
)

LC2_ARCHIVE_BYTES = 187_226
LC2_ARCHIVE_SHA256 = (
    "f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45"
)
LC2_RAW_BYTES = 3_662_409_600
LC2_RAW_SHA256 = (
    "a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353"
)
LC2_CARRIER_SHA256 = (
    "a05d0985ca5a8d5110bd5bf5be39f238c6f89640b8a8bb888a3e1269bdf636e4"
)
PR133_ARCHIVE_BYTES = 190_212
PR133_ARCHIVE_SHA256 = (
    "051baf408f57fae3b343d6ee218ab963d070b3935ceb0b2f412c93a53cf3fab0"
)
PR133_CARRIER_SHA256 = (
    "080aaf3206e1afc1449c8deff14362bb0b910d937df15106bfb53befe6d5045e"
)
JOINT_SOLVER_SHA256 = (
    "5d7424f1e523105766ac1f45d7c9219899534394dae348cd7f29b3304fd4f286"
)
EXPERIMENT_BOOK_HEAD = "f229b26735dffc53fdf1ac9987ac7c303298d028"
LC2_STACK_SHA256 = (
    "2465b9bbac89480125d71c349df82e983ffb9da1f782153e9495ef0ccbd410f2"
)
LC2_RECEIVER_SHA256 = (
    "7dd29117a0cac30b32eb21bcc0e7ee6e1a45bf7f4af8f52ed5e94231945cc111"
)
LC2_MODELS_RAW_SHA256 = (
    "62dd72dfa0858a25ca32bdee1e536627a17883b6fc7efd7cd5b2de7b13b84517"
)
LC2_MODELS_RAW_WIRE_SHA256 = (
    "618ac80da2bfb82a52a94317877cfd79af71290f751e3d4f130a46258b29092a"
)
LC2_TOKENS_SHA256 = (
    "85d6c199ffb93ddab0fe1631448882a255e9fea1f6858bab5a04cea2310a7331"
)
LC2_TEMPORAL_SHA256 = (
    "f920f7be8108b83831971a8d07c9ef522eadb18abed095cf395bf3a6f871e796"
)
JRD_PRIOR_POLICY_SHA256 = (
    "dce176635a9c26efdd618e1ea0c9a52baf7428e265629b50ae7c2be8e49fce00"
)
JRD_PRIOR_HARVEST_JSON_SHA256 = (
    "21de7fb839789cca5e3998cbffa59b0ce7ce95ce3715af74b04793da75c7a02e"
)
JRD_PRIOR_HARVEST_MEMO_SHA256 = (
    "a4e59dd64d94ea42e9b91b6193d8308639c50969b4c469f2749388156da01d0a"
)
JRD_PR110_RESPONSE_CURVES_SHA256 = (
    "ac6ed0960c96a8bcd357da954e6e036824b672b4a4c262757c25e3c233b78cb4"
)
JRD_PR110_MEASUREMENT_RECEIPT_SHA256 = (
    "2cdb36ff2b842b72d284368de44a04883a50ab4db6d886ca7402631bcd5eabb8"
)
STAGE_C_IMPLEMENTATION_SPEC_SHA256 = (
    "617df4cd777c80adf9b3ce24d9949990ed702b940e7ea304b20b6d0cfb902dd1"
)
LC2_INFLATE_SD1M_SHA256 = (
    "e01325d65c42223d5e1ca8169f2bef0f62ae59bdcfeabf321e681fa2cd07d4e2"
)
LC2_RUNTIME_DEPENDENCIES_SHA256 = (
    "ff8b6749aea767d9ec4551e238670cf220d94f1b1c05b5c7ba031e093fa04af2"
)
SR1_RECEIVER_PROOF_SHA256 = (
    "7b956662c772a32b80637b11b0cd0162b66850d0cd35b09e3870ab0d3a3e6f58"
)

N = 600
D = 12
POSE_DIMS = 6
CAMERA_H = 874
CAMERA_W = 1164
SCORER_H = 384
SCORER_W = 512
CARRIER_H = 24
CARRIER_W = 32
ORIGINAL_BYTES = 37_545_489
AXIS = "[macOS-CPU advisory]"
MIN_PASSES = 8
DRY_PASSES = 3
DEFAULT_MAX_PASSES = 24
DEFAULT_BATCH_SIZE = 16
DEFAULT_THREADS = 8
TARGET_CHUNK_PAIRS = 112
SEARCH_CHUNK_PAIRS = 20
GN_DAMPING = 0.01
# The retained PR130/JRD-lineage CPU coordinate ladder is 1,2,4,8,16,32.
# Its largest exact step seeds this trust radius; exact lattice evaluation is
# still the acceptance oracle.
JRD_LINEAGE_STEPS = (1, 2, 4, 8, 16, 32)
MAX_CODE_STEP = float(max(JRD_LINEAGE_STEPS))
NEIGHBOUR_DIMS = 3
NEIGHBOUR_RADIUS = 1
JRD_GRID_SIZE = 1 + 2 * D * len(JRD_LINEAGE_STEPS)
GRID_MAX = 25 + (1 + (2 * NEIGHBOUR_RADIUS + 1) ** NEIGHBOUR_DIMS)
# A rate variant currently retains an archive, raw/recompressed carrier,
# coefficient/output/error arrays, and receipts.  One MiB is a conservative
# fail-closed allowance for that materialization; a pass/finisher may retain up
# to N+1 variants and gets additional headroom for search chunks and selected
# repeats.  These are dynamic admission floors, not projected artifact sizes.
RATE_CANDIDATE_REQUIRED_FREE_BYTES = 1_000_000
PASS_REQUIRED_FREE_BYTES = (
    (N + 1) * RATE_CANDIDATE_REQUIRED_FREE_BYTES + 150_000_000
)
FINISHER_REQUIRED_FREE_BYTES = PASS_REQUIRED_FREE_BYTES
TARGET_CACHE_DIRNAME = "target_cache_av_yuv420_v2"
TARGET_CACHE_SCHEMA = "ddm_ps135_av_target_manifest.v2"
TARGET_CHUNK_SCHEMA = "ddm_ps135_av_target_chunk.v2"
JRD_CHUNK_SCHEMA = "ddm_ps135_jrd_chunk.v2"
EXACT_DECODE_SUCCESS_SCHEMA = "ddm_ps135_exact_decode_success.v1"
EXACT_EVAL_SCHEMA = "ddm_ps135_exact_decode_eval.v2"
_INPUT_PINS_CACHE: dict[str, object] | None = None


class PoseResolveError(RuntimeError):
    """A custody, resume, retention, parse-back, or scorer invariant failed."""


@dataclasses.dataclass(frozen=True)
class FileRecord:
    path: str
    bytes: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class CarrierState:
    carrier: bytes
    basis_scales: np.ndarray
    basis_codes: np.ndarray
    coefficient_scales: np.ndarray
    codes: np.ndarray


@dataclasses.dataclass(frozen=True)
class LC2Source:
    """Retained byte-lineage source needed to mutate only the CPR1 carrier."""

    semantic: bytes
    carrier: bytes
    hpac_wire: bytes
    hpac_base: bytes
    tokens: bytes
    models_raw: bytes
    models_raw_wire: bytes
    temporal_packed: bytes


@dataclasses.dataclass(frozen=True)
class MasterFrameProvider:
    """Receiver-realized semantic masters plus their resumable identity.

    ``frames`` is always the generated odd/master half in evaluation order.
    ``slaves`` is present only for an interleaved retained decode and is used
    solely for the LC2 carrier-renderer parity proof.  Search chunks bind the
    canonical ``binding`` object, so a mixed-semantic bank cannot be silently
    replaced by the legacy q4 masters on resume.
    """

    frames: np.ndarray
    binding: dict[str, object]
    slaves: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.frames.shape != (N, CAMERA_H, CAMERA_W, 3):
            raise PoseResolveError("master provider has the wrong frame shape")
        if self.frames.dtype != np.uint8:
            raise PoseResolveError("master provider must expose uint8 frames")
        if self.slaves is not None and (
            self.slaves.shape != self.frames.shape or self.slaves.dtype != np.uint8
        ):
            raise PoseResolveError("master provider slave parity bank differs")
        if not isinstance(self.binding, dict) or not self.binding:
            raise PoseResolveError("master provider lacks a resumable binding")

    def masters(self, rows: np.ndarray) -> np.ndarray:
        indices = np.asarray(rows, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= N):
            raise PoseResolveError("master provider rows are out of range")
        return np.asarray(self.frames[indices])

    def expected_slaves(self, rows: np.ndarray) -> np.ndarray:
        if self.slaves is None:
            raise PoseResolveError("master provider has no retained slave parity bank")
        indices = np.asarray(rows, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= N):
            raise PoseResolveError("slave provider rows are out of range")
        return np.asarray(self.slaves[indices])


ArchiveBuilder = Callable[
    [bytes, LC2Source], tuple[bytes, bytes, dict[str, object]]
]


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def official_batch_sizes(batch_size: int) -> list[int]:
    if batch_size <= 0:
        raise PoseResolveError("official scorer batch size must be positive")
    return [min(batch_size, N - start) for start in range(0, N, batch_size)]


def file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return FileRecord(
        path=str(resolved),
        bytes=resolved.stat().st_size,
        sha256=sha256_file(resolved),
    ).as_dict()


def verify_file_record_binding(
    record: object,
    *,
    label: str,
) -> Path:
    """Fail closed unless a stored file record still names identical bytes."""

    if not isinstance(record, dict):
        raise PoseResolveError(f"{label} file-record binding is not an object")
    if set(record) != {"path", "bytes", "sha256"}:
        raise PoseResolveError(f"{label} file-record binding has the wrong fields")
    if (
        not isinstance(record.get("path"), str)
        or not isinstance(record.get("bytes"), int)
        or not isinstance(record.get("sha256"), str)
    ):
        raise PoseResolveError(f"{label} file-record binding has invalid field types")
    path = Path(record["path"])
    try:
        actual = file_record(path)
    except (FileNotFoundError, OSError) as error:
        raise PoseResolveError(f"{label} bound artifact is unavailable: {path}") from error
    if actual != record:
        raise PoseResolveError(f"{label} bound artifact changed: {path}")
    return path


def current_artifact_bindings(
    *,
    archive: Path,
    coefficients: Path,
    pose_outputs_path: Path,
    pair_errors: Path,
) -> dict[str, object]:
    """Bind every artifact needed to resume one immutable solver state."""

    bindings = {
        "schema": "ddm_ps135_current_artifacts.v1",
        "archive": file_record(archive),
        "coefficients": file_record(coefficients),
        "pose_outputs": file_record(pose_outputs_path),
        "pair_errors": file_record(pair_errors),
    }
    verify_current_artifact_bindings(bindings)
    return bindings


def verify_current_artifact_bindings(
    bindings: object,
) -> dict[str, Path]:
    """Verify and resolve all four immutable current-state artifacts."""

    if not isinstance(bindings, dict):
        raise PoseResolveError("current artifact bindings are not an object")
    if bindings.get("schema") != "ddm_ps135_current_artifacts.v1":
        raise PoseResolveError("current artifact bindings have an unsupported schema")
    expected = ("archive", "coefficients", "pose_outputs", "pair_errors")
    if set(bindings) != {"schema", *expected}:
        raise PoseResolveError("current artifact bindings have the wrong fields")
    return {
        key: verify_file_record_binding(bindings[key], label=f"current {key}")
        for key in expected
    }


def load_current_artifacts(
    bindings: object,
) -> tuple[bytes, np.ndarray, np.ndarray, np.ndarray, dict[str, Path]]:
    """Verify a state binding before returning its exact resume payloads."""

    paths = verify_current_artifact_bindings(bindings)
    try:
        codes = np.load(paths["coefficients"], allow_pickle=False)
        outputs = np.load(paths["pose_outputs"], allow_pickle=False)
        errors = np.load(paths["pair_errors"], allow_pickle=False)
    except (OSError, ValueError) as error:
        raise PoseResolveError("a bound current-state NumPy payload is unreadable") from error
    if codes.dtype != np.int16 or codes.shape != (N, D):
        raise PoseResolveError("bound current coefficients have the wrong dtype/shape")
    if outputs.dtype != np.float32 or outputs.shape != (N, POSE_DIMS):
        raise PoseResolveError("bound current pose outputs have the wrong dtype/shape")
    if errors.dtype != np.float64 or errors.shape != (N,):
        raise PoseResolveError("bound current pair errors have the wrong dtype/shape")
    return paths["archive"].read_bytes(), codes, outputs, errors, paths


def require_vertigo_free_space(
    destination: Path,
    *,
    required_free_bytes: int,
    stage: str,
) -> dict[str, object]:
    """Dynamically fail closed before a Vertigo materialization boundary."""

    if not isinstance(required_free_bytes, int) or required_free_bytes <= 0:
        raise PoseResolveError("required Vertigo free bytes must be a positive integer")
    resolved_root = VERTIGO_ROOT.resolve()
    resolved = destination.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise PoseResolveError(
            f"{stage} destination is outside the governed Vertigo store: {resolved}"
        )
    probe = resolved
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            raise PoseResolveError(f"no existing storage ancestor for {resolved}")
        probe = parent
    free_bytes = int(shutil.disk_usage(probe).free)
    receipt = {
        "schema": "ddm_ps135_dynamic_storage_preflight.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "stage": stage,
        "destination": str(resolved),
        "probe": str(probe.resolve()),
        "required_free_bytes": required_free_bytes,
        "observed_free_bytes": free_bytes,
        "passes": free_bytes >= required_free_bytes,
    }
    if not receipt["passes"]:
        raise PoseResolveError(
            f"{stage} needs {required_free_bytes} free Vertigo bytes, found {free_bytes}"
        )
    return receipt


def require_file(
    path: Path,
    *,
    label: str,
    size: int | None = None,
    digest: str | None = None,
) -> None:
    if not path.is_file():
        raise PoseResolveError(f"{label} is absent: {path}")
    if size is not None and path.stat().st_size != size:
        raise PoseResolveError(f"{label} byte count differs from its pin")
    if digest is not None and sha256_file(path) != digest:
        raise PoseResolveError(f"{label} SHA-256 differs from its pin")


def atomic_bytes(path: Path, payload: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if executable:
        temporary.chmod(0o755)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def atomic_numpy(path: Path, value: np.ndarray) -> dict[str, object]:
    value = np.asarray(value)
    if path.exists():
        existing = np.load(path, allow_pickle=False)
        if existing.dtype != value.dtype or not np.array_equal(existing, value):
            raise PoseResolveError(f"refusing divergent retained NumPy payload: {path}")
        return file_record(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_record(path)


def atomic_npz(path: Path, **values: np.ndarray) -> dict[str, object]:
    values = {key: np.asarray(value) for key, value in values.items()}
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(values) or any(
                existing[key].dtype != values[key].dtype
                or not np.array_equal(existing[key], values[key])
                for key in values
            ):
                raise PoseResolveError(f"refusing divergent retained NPZ payload: {path}")
        return file_record(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_record(path)


def persist_exact(path: Path, payload: bytes) -> dict[str, object]:
    expected = {"bytes": len(payload), "sha256": sha256_bytes(payload)}
    if path.exists():
        actual = file_record(path)
        if (
            actual["bytes"] != expected["bytes"]
            or actual["sha256"] != expected["sha256"]
        ):
            raise PoseResolveError(f"refusing divergent retained payload: {path}")
        return actual
    atomic_bytes(path, payload)
    return file_record(path)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PoseResolveError(f"JSON object required: {path}")
    return payload


def pinned_file_record(
    path: Path,
    *,
    label: str,
    digest: str,
) -> dict[str, object]:
    """Return a file record only after its durable source pin closes."""

    require_file(path, label=label, digest=digest)
    return file_record(path)


def score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return (
        100.0 * float(d_seg)
        + math.sqrt(10.0 * float(d_pose))
        + 25.0 * int(archive_bytes) / ORIGINAL_BYTES
    )


def pose_pair_errors(outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-pair PoseNet MSE with the scorer's float32 arithmetic surface."""

    difference = np.asarray(outputs, dtype=np.float32) - np.asarray(
        targets, dtype=np.float32
    )
    return np.mean(difference * difference, axis=1, dtype=np.float32).astype(
        np.float64
    )


def mean_pose_error(errors: np.ndarray) -> float:
    return float(np.asarray(errors, dtype=np.float64).sum() / N)


def signed_codes_from_delta_zigzag(encoded: np.ndarray) -> np.ndarray:
    encoded = np.asarray(encoded, dtype=np.int64)
    if encoded.shape != (N, D) or np.any(encoded < 0) or np.any(encoded > 4095):
        raise PoseResolveError("encoded coefficient lattice has invalid shape/range")
    delta = (encoded >> 1) ^ -(encoded & 1)
    unsigned = np.cumsum(delta, axis=0, dtype=np.int64) & 0xFFF
    return np.where(unsigned >= 0x800, unsigned - 0x1000, unsigned).astype(
        np.int16
    )


def delta_zigzag_from_signed_codes(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.int64)
    if codes.shape != (N, D) or np.any(codes < -2048) or np.any(codes > 2047):
        raise PoseResolveError("signed coefficient lattice has invalid shape/range")
    unsigned = codes & 0xFFF
    previous = np.zeros_like(unsigned)
    previous[1:] = unsigned[:-1]
    delta_unsigned = (unsigned - previous) & 0xFFF
    delta = np.where(delta_unsigned >= 0x800, delta_unsigned - 0x1000, delta_unsigned)
    return (((delta << 1) ^ (delta >> 63)) & 0xFFF).astype(np.int32)


def _prepend_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def import_runtime_modules():
    # The upstream Python 3.11 environment owns the working torch/av wheels.
    # LC2's receiver additionally needs constriction, retained in the PQ1
    # environment.  Append (never prepend) that site so it cannot shadow the
    # upstream scorer wheels.
    value = str(PQ1_SITE_PACKAGES)
    if value not in sys.path:
        sys.path.append(value)
    os.environ.setdefault("PR130_BROTLI_CLI", str(BROTLI_CLI))
    _prepend_path(LC2_RUNTIME)
    carrier_codec = importlib.import_module("carrier_codec")
    receiver = importlib.import_module("receiver")
    inflate = importlib.import_module("inflate")
    return carrier_codec, receiver, inflate


def brotli_compress(payload: bytes, *, quality: int) -> bytes:
    """Compress through the pinned Brotli 1.2 Python wheel out of process."""

    require_file(BROTLI_PYTHON, label="Brotli 1.2 Python")
    result = subprocess.run(
        [
            str(BROTLI_PYTHON),
            "-c",
            (
                "import brotli,sys;sys.stdout.buffer.write("
                f"brotli.compress(sys.stdin.buffer.read(),quality={quality}))"
            ),
        ],
        input=payload,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise PoseResolveError(f"Brotli q{quality} compression failed: {detail}")
    if not result.stdout:
        raise PoseResolveError(f"Brotli q{quality} produced an empty stream")
    return result.stdout


def split_pack(streams: tuple[bytes, bytes, bytes]) -> bytes:
    if any(not stream for stream in streams):
        raise PoseResolveError("LC2 split pack requires three non-empty streams")
    return struct.pack("<III", *(len(stream) for stream in streams)) + b"".join(
        streams
    )


def deterministic_stored_zip(member: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, member, compress_type=zipfile.ZIP_STORED)
    return output.getvalue()


def load_lc2_source() -> LC2Source:
    """Load the immutable retained LC2 source sections without rebuilding AI1."""

    paths = {
        "semantic": LC2_INPUTS / "semantic.raw",
        "carrier": LC2_INPUTS / "carrier.raw",
        "hpac_wire": LC2_INPUTS / "hpac_plus_temporal.raw",
        "tokens": LC2_INPUTS / "tokens.ans",
        "models_raw": LC2_INPUTS / "models_raw.bin",
        "models_raw_wire": LC2_INPUTS / "models_raw_with_temporal.bin",
        "temporal": LC2_INPUTS / "temporal_reversion.tm1p",
    }
    for label, path in paths.items():
        require_file(path, label=f"retained LC2 {label}")
    semantic = paths["semantic"].read_bytes()
    carrier = paths["carrier"].read_bytes()
    hpac_wire = paths["hpac_wire"].read_bytes()
    tokens = paths["tokens"].read_bytes()
    models_raw = paths["models_raw"].read_bytes()
    models_raw_wire = paths["models_raw_wire"].read_bytes()
    temporal = paths["temporal"].read_bytes()
    if sha256_bytes(carrier) != LC2_CARRIER_SHA256:
        raise PoseResolveError("retained LC2 carrier differs from its pin")
    if sha256_bytes(models_raw) != LC2_MODELS_RAW_SHA256:
        raise PoseResolveError("retained LC2 base model bytes differ from their pin")
    if sha256_bytes(models_raw_wire) != LC2_MODELS_RAW_WIRE_SHA256:
        raise PoseResolveError("retained LC2 temporal wire differs from its pin")
    if sha256_bytes(tokens) != LC2_TOKENS_SHA256:
        raise PoseResolveError("retained LC2 token bytes differ from their pin")
    if sha256_bytes(temporal) != LC2_TEMPORAL_SHA256:
        raise PoseResolveError("retained LC2 temporal payload differs from its pin")
    if len(models_raw) < 8:
        raise PoseResolveError("retained LC2 base model bytes are truncated")
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", models_raw)
    semantic_end = 8 + semantic_bytes
    carrier_end = semantic_end + carrier_bytes
    if models_raw[8:semantic_end] != semantic:
        raise PoseResolveError("retained LC2 semantic section is not source-identical")
    if models_raw[semantic_end:carrier_end] != carrier:
        raise PoseResolveError("retained LC2 carrier section is not source-identical")
    hpac_base = models_raw[carrier_end:]
    _, receiver, _ = import_runtime_modules()
    parsed_base, parsed_temporal = receiver.split_optional_temporal_reversion(
        models_raw_wire
    )
    if parsed_base != models_raw or parsed_temporal is None:
        raise PoseResolveError("retained LC2 temporal wire does not parse to its base")
    if parsed_temporal.packed != temporal:
        raise PoseResolveError("retained LC2 temporal payload is not parse-identical")
    if hpac_wire != hpac_base + models_raw_wire[len(models_raw) :]:
        raise PoseResolveError("retained LC2 HPAC wire is not HPAC+TM1 envelope")
    return LC2Source(
        semantic=semantic,
        carrier=carrier,
        hpac_wire=hpac_wire,
        hpac_base=hpac_base,
        tokens=tokens,
        models_raw=models_raw,
        models_raw_wire=models_raw_wire,
        temporal_packed=temporal,
    )


def import_joint_primitives():
    require_file(
        JOINT_SOLVER,
        label="public joint-pose primitives",
        digest=JOINT_SOLVER_SHA256,
    )
    _prepend_path(EXPERIMENT_BOOK_SRC)
    module = importlib.import_module("cpr1_sub4.joint_pose_solve")
    return (
        module.solve_damped_least_squares,
        module.quantize_int12_update,
        module.rank_neighbour_dimensions,
        module.nearby_int12_candidates,
    )


def decode_carrier(carrier: bytes) -> CarrierState:
    carrier_codec, _, _ = import_runtime_modules()
    basis_count = D * 3 * CARRIER_H * CARRIER_W
    basis_scales, basis_codes, coefficient_scales, encoded = (
        carrier_codec.decode_compact_carrier(
            carrier,
            basis_count=basis_count,
            frames=N,
            dimensions=D,
        )
    )
    codes = signed_codes_from_delta_zigzag(encoded)
    rebuilt = carrier_codec.encode_compact_carrier(
        basis_scales,
        basis_codes,
        coefficient_scales,
        delta_zigzag_from_signed_codes(codes),
    )
    if rebuilt != carrier:
        raise PoseResolveError("CPR1 parse/re-encode is not byte-identical")
    return CarrierState(
        carrier=carrier,
        basis_scales=np.asarray(basis_scales, dtype=np.float32),
        basis_codes=np.asarray(basis_codes, dtype=np.int8),
        coefficient_scales=np.asarray(coefficient_scales, dtype=np.float32),
        codes=np.asarray(codes, dtype=np.int16),
    )


def encode_carrier(template: CarrierState, codes: np.ndarray) -> bytes:
    carrier_codec, _, _ = import_runtime_modules()
    return carrier_codec.encode_compact_carrier(
        template.basis_scales,
        template.basis_codes,
        template.coefficient_scales,
        delta_zigzag_from_signed_codes(codes),
    )


def extract_pr133_carrier() -> bytes:
    require_file(
        PR133_ARCHIVE,
        label="public PR133 archive",
        size=PR133_ARCHIVE_BYTES,
        digest=PR133_ARCHIVE_SHA256,
    )
    _prepend_path(EXPERIMENT_BOOK_SRC)
    residual = importlib.import_module("cpr1_sub4.residual_archive")
    parts = residual.read_residual_archive(PR133_ARCHIVE)
    carrier = bytes(parts.carrier_blob)
    if sha256_bytes(carrier) != PR133_CARRIER_SHA256:
        raise PoseResolveError("PR133 archive did not expose the pinned complete CPR1")
    decode_carrier(carrier)
    return carrier


def selected_lc2_streams() -> tuple[bytes, bytes, dict[str, Any]]:
    search = load_json(LC2_SEARCH)
    winner = search.get("winner")
    if not isinstance(winner, dict):
        raise PoseResolveError("LC2 search receipt lacks a winner")
    semantic = winner.get("semantic")
    hpac = winner.get("hpac_wire")
    carrier = winner.get("carrier")
    zip_policy = winner.get("zip")
    if not all(isinstance(item, dict) for item in (semantic, hpac, carrier, zip_policy)):
        raise PoseResolveError("LC2 winner section metadata is malformed")
    if (
        semantic.get("brotli_quality") != 10
        or carrier.get("brotli_quality") != 9
        or hpac.get("brotli_quality") != 10
        or zip_policy.get("codec") != "stored"
    ):
        raise PoseResolveError("LC2 selected codec policy differs from the pin")
    semantic_path = Path(semantic["encoded"]["path"])
    hpac_path = Path(hpac["encoded"]["path"])
    return semantic_path.read_bytes(), hpac_path.read_bytes(), search


def parse_candidate_archive(
    archive: bytes,
    expected_carrier: bytes,
    source,
    *,
    expected_semantic: bytes | None = None,
) -> dict[str, object]:
    _, receiver, inflate = import_runtime_modules()
    with zipfile.ZipFile(io.BytesIO(archive)) as handle:
        entries = handle.infolist()
        if (
            len(entries) != 1
            or entries[0].filename != "p"
            or entries[0].compress_type != zipfile.ZIP_STORED
            or entries[0].flag_bits & 0x1
        ):
            raise PoseResolveError("candidate outer ZIP grammar differs from LC2")
        member = handle.read("p")
        if handle.testzip() is not None:
            raise PoseResolveError("candidate outer ZIP failed CRC")
    parts = receiver.split_payload(member)
    if parts.token_codec != "ans" or parts.model_codec != "split_brotli_cx2":
        raise PoseResolveError("candidate wire selectors differ from LC2")
    decoded = receiver.decode_models(parts.models, model_codec=parts.model_codec)
    models_raw, temporal = receiver.split_optional_temporal_reversion(decoded.raw)
    if temporal is None or temporal.packed != source.temporal_packed:
        raise PoseResolveError("candidate temporal sidecar changed")
    if parts.tokens != source.tokens:
        raise PoseResolveError("candidate token payload changed")
    if len(models_raw) < 8:
        raise PoseResolveError("candidate model bytes are truncated")
    semantic_bytes, carrier_bytes = np.frombuffer(
        models_raw[:8], dtype="<u4", count=2
    ).tolist()
    semantic_end = 8 + int(semantic_bytes)
    carrier_end = semantic_end + int(carrier_bytes)
    semantic = models_raw[8:semantic_end]
    carrier = models_raw[semantic_end:carrier_end]
    hpac = models_raw[carrier_end:]
    semantic_target = source.semantic if expected_semantic is None else expected_semantic
    if semantic != semantic_target or hpac != source.hpac_base:
        raise PoseResolveError("candidate changed its bound semantic/HPAC section")
    if carrier != expected_carrier:
        raise PoseResolveError("candidate parsed carrier differs from the proposal")
    semantic_model, basis, coefficients = inflate.unpack_semantic_pose(
        models_raw[:carrier_end]
    )
    allocation, _, semantic_format = inflate.semantic_allocation(
        semantic, semantic_model.state_dict()
    )
    parsed = decode_carrier(carrier)
    expected_coefficients = (
        parsed.codes.astype(np.float32)
        * parsed.coefficient_scales[None, :]
    )
    if not np.array_equal(coefficients.numpy(), expected_coefficients):
        raise PoseResolveError("candidate receiver coefficients differ after parse-back")
    return {
        "archive_bytes": len(archive),
        "archive_sha256": sha256_bytes(archive),
        "member_bytes": len(member),
        "member_sha256": sha256_bytes(member),
        "semantic_bytes": len(semantic),
        "semantic_sha256": sha256_bytes(semantic),
        "semantic_format": semantic_format,
        "semantic_allocation": {
            name: int(bits) for name, bits in allocation.items()
        },
        "carrier_bytes": len(carrier),
        "carrier_sha256": sha256_bytes(carrier),
        "tokens_sha256": sha256_bytes(parts.tokens),
        "temporal_packed_sha256": sha256_bytes(temporal.packed),
        "basis_shape": list(basis.shape),
        "coefficients_shape": list(coefficients.shape),
        "bound_semantic_surface": True,
    }


def build_candidate_archive(
    carrier: bytes,
    source: LC2Source,
    *,
    semantic_override: bytes | None = None,
) -> tuple[bytes, bytes, dict[str, object]]:
    _, receiver, _ = import_runtime_modules()
    semantic = source.semantic if semantic_override is None else semantic_override
    if semantic == source.semantic:
        # Preserve the exact shipped q4 streams instead of relying on a
        # compressor-version equivalence claim.
        semantic_stream, hpac_stream, _ = selected_lc2_streams()
        carrier_wire = carrier
    else:
        semantic_wire, carrier_wire, hpac_wire = receiver.encode_cx2_model_sections(
            semantic,
            carrier,
            source.hpac_wire,
        )
        if receiver.decode_cx2_model_sections(
            semantic_wire, carrier_wire, hpac_wire
        ) != (semantic, carrier, source.hpac_wire):
            raise PoseResolveError("mixed-semantic CX2 transform failed its inverse")
        semantic_stream = brotli_compress(semantic_wire, quality=10)
        hpac_stream = brotli_compress(hpac_wire, quality=10)
    carrier_stream = brotli_compress(carrier_wire, quality=9)
    model_pack = split_pack((semantic_stream, carrier_stream, hpac_stream))
    member = receiver.pack_payload(
        model_pack,
        source.tokens,
        token_codec="ans",
        model_codec="split_brotli_cx2",
    )
    archive = deterministic_stored_zip(member)
    receipt = parse_candidate_archive(
        archive,
        carrier,
        source,
        expected_semantic=semantic,
    )
    receipt["carrier_brotli_bytes"] = len(carrier_stream)
    receipt["carrier_brotli_sha256"] = sha256_bytes(carrier_stream)
    return archive, carrier_stream, receipt


def retain_candidate_bundle(
    directory: Path,
    *,
    carrier: bytes,
    carrier_stream: bytes,
    archive: bytes,
    codes: np.ndarray,
    parseback: dict[str, object],
    metadata: dict[str, object],
) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    records = {
        "carrier": persist_exact(directory / "carrier.cpr1", carrier),
        "carrier_brotli": persist_exact(directory / "carrier.q9.br", carrier_stream),
        "archive": persist_exact(directory / "archive.zip", archive),
        "coefficients": atomic_numpy(
            directory / "coefficients.int16.npy",
            np.asarray(codes, dtype=np.int16),
        ),
    }
    receipt = {
        "schema": "ddm_ps135_candidate_bundle.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[archive-byte exact + receiver parse-back; scorer fields as labelled]",
        "payloads_retained": True,
        "records": records,
        "parseback": parseback,
        **metadata,
    }
    atomic_json(directory / "receipt.json", receipt)
    return receipt


def input_pins() -> dict[str, object]:
    global _INPUT_PINS_CACHE
    if _INPUT_PINS_CACHE is None:
        _INPUT_PINS_CACHE = {
            "lc2_archive": file_record(LC2_ARCHIVE),
            "lc2_raw": file_record(LC2_RAW),
            "pr133_archive": file_record(PR133_ARCHIVE),
            "lc2_stack": file_record(LC2_STACK),
            "lc2_receiver": file_record(LC2_RUNTIME / "receiver.py"),
            "joint_solver": file_record(JOINT_SOLVER),
            "experiment_book_head": EXPERIMENT_BOOK_HEAD,
        }
    return _INPUT_PINS_CACHE


def verify_input_pins() -> None:
    pins = input_pins()
    expected = {
        "lc2_archive": (LC2_ARCHIVE_BYTES, LC2_ARCHIVE_SHA256),
        "lc2_raw": (LC2_RAW_BYTES, LC2_RAW_SHA256),
        "pr133_archive": (PR133_ARCHIVE_BYTES, PR133_ARCHIVE_SHA256),
        "lc2_stack": (LC2_STACK.stat().st_size, LC2_STACK_SHA256),
        "lc2_receiver": (
            (LC2_RUNTIME / "receiver.py").stat().st_size,
            LC2_RECEIVER_SHA256,
        ),
        "joint_solver": (JOINT_SOLVER.stat().st_size, JOINT_SOLVER_SHA256),
    }
    for label, (size, digest) in expected.items():
        row = pins[label]
        if row["bytes"] != size or row["sha256"] != digest:
            raise PoseResolveError(f"{label} differs from its custody pin")


def scorer_source_pins() -> dict[str, object]:
    files = (
        UPSTREAM / "evaluate.py",
        UPSTREAM / "frame_utils.py",
        UPSTREAM / "modules.py",
        UPSTREAM / "models" / "posenet.safetensors",
        UPSTREAM / "models" / "segnet.safetensors",
        UPSTREAM / "public_test_video_names.txt",
    )
    return {str(path.relative_to(REPO)): file_record(path) for path in files}


def exact_authority_source_pins() -> dict[str, object]:
    """Bind the free receiver/runtime and every untouched scorer input."""

    decode_files = {
        "inflate.py": LC2_RUNTIME / "inflate.py",
        "receiver.py": LC2_RUNTIME / "receiver.py",
        "carrier_codec.py": LC2_RUNTIME / "carrier_codec.py",
        "python": UPSTREAM / ".venv" / "bin" / "python",
        "brotli_cli": BROTLI_CLI,
    }
    return {
        "decode": {name: file_record(path) for name, path in decode_files.items()},
        "scorer": scorer_source_pins(),
    }


def parse_upstream_report(
    report_text: str,
    *,
    expected_archive_bytes: int,
) -> dict[str, object]:
    """Parse and assert the exact denominator/byte facts printed by evaluate.py."""

    samples = re.search(
        r"^=== Evaluation results over ([0-9]+) samples ===$", report_text, re.M
    )
    if samples is None or int(samples.group(1)) != N:
        raise PoseResolveError("upstream report does not prove the n600 denominator")

    def number(label: str) -> float:
        match = re.search(rf"^  {re.escape(label)}: ([0-9.]+)$", report_text, re.M)
        if match is None:
            raise PoseResolveError(f"upstream report lacks {label}")
        return float(match.group(1))

    def byte_count(label: str) -> int:
        match = re.search(
            rf"^  {re.escape(label)}: ([0-9,]+) bytes$", report_text, re.M
        )
        if match is None:
            raise PoseResolveError(f"upstream report lacks {label}")
        return int(match.group(1).replace(",", ""))

    archive_bytes = byte_count("Submission file size")
    original_bytes = byte_count("Original uncompressed size")
    if archive_bytes != expected_archive_bytes:
        raise PoseResolveError("upstream report archive byte count differs from the packet")
    if original_bytes != ORIGINAL_BYTES:
        raise PoseResolveError("upstream report original-byte denominator differs from its pin")
    return {
        "pair_count": int(samples.group(1)),
        "d_pose": number("Average PoseNet Distortion"),
        "d_seg": number("Average SegNet Distortion"),
        "archive_bytes": archive_bytes,
        "original_bytes": original_bytes,
    }


def validate_decode_success(
    proof: object,
    *,
    archive_sha: str,
    raw_path: Path,
    authority_sources: dict[str, object],
) -> dict[str, object]:
    if not isinstance(proof, dict):
        raise PoseResolveError("decode success proof is not an object")
    if (
        proof.get("schema") != EXACT_DECODE_SUCCESS_SCHEMA
        or proof.get("complete") is not True
        or proof.get("archive_sha256") != archive_sha
        or proof.get("authority_sources") != authority_sources
    ):
        raise PoseResolveError("decode success proof has stale authority bindings")
    for key in ("archive", "member", "decoded_raw", "decode_log"):
        verify_file_record_binding(proof.get(key), label=f"decode success {key}")
    token_checkpoints = proof.get("token_checkpoints")
    if not isinstance(token_checkpoints, list) or not token_checkpoints:
        raise PoseResolveError("decode success proof lacks retained token checkpoints")
    for index, record in enumerate(token_checkpoints):
        verify_file_record_binding(record, label=f"decode success token checkpoint {index}")
    decoded = proof["decoded_raw"]
    if (
        Path(decoded["path"]).resolve() != raw_path.resolve()
        or decoded["bytes"] != LC2_RAW_BYTES
        or proof["archive"]["sha256"] != archive_sha
    ):
        raise PoseResolveError("decode success proof names the wrong archive/raw payload")
    archive_path = Path(proof["archive"]["path"])
    member_path = Path(proof["member"]["path"])
    with zipfile.ZipFile(archive_path) as handle:
        if handle.read("p") != member_path.read_bytes():
            raise PoseResolveError("decode success member differs from its archive")
    return proof


def reusable_decode_success(
    attempts: Path,
    *,
    archive_sha: str,
    raw_path: Path,
    authority_sources: dict[str, object],
) -> tuple[dict[str, object], Path] | None:
    proofs = sorted(attempts.glob("attempt_*.decode_success.json"))
    if not proofs:
        return None
    path = proofs[-1]
    proof = validate_decode_success(
        load_json(path),
        archive_sha=archive_sha,
        raw_path=raw_path,
        authority_sources=authority_sources,
    )
    return proof, path


def validate_exact_evaluation_receipt(
    receipt: object,
    *,
    archive_sha: str,
    raw_path: Path,
    authority_sources: dict[str, object],
) -> dict[str, object]:
    """Fail closed on every retained byte/source behind a reused exact row."""

    if not isinstance(receipt, dict):
        raise PoseResolveError("exact-evaluation receipt is not an object")
    archive_record = receipt.get("archive")
    if (
        receipt.get("schema") != EXACT_EVAL_SCHEMA
        or receipt.get("complete") is not True
        or receipt.get("pair_count") != N
        or not isinstance(archive_record, dict)
        or archive_record.get("sha256") != archive_sha
        or receipt.get("authority_sources") != authority_sources
    ):
        raise PoseResolveError("exact-evaluation receipt has stale authority bindings")
    for key in (
        "archive",
        "member",
        "decoded_raw",
        "decode_log",
        "decode_proof",
        "evaluate_source",
        "evaluate_report",
        "evaluate_log",
    ):
        verify_file_record_binding(receipt.get(key), label=f"exact evaluation {key}")
    if Path(receipt["decoded_raw"]["path"]).resolve() != raw_path.resolve():
        raise PoseResolveError("exact-evaluation receipt names the wrong decoded raw")
    token_checkpoints = receipt.get("token_checkpoints")
    if not isinstance(token_checkpoints, list) or not token_checkpoints:
        raise PoseResolveError("exact-evaluation receipt lacks token checkpoints")
    for index, record in enumerate(token_checkpoints):
        verify_file_record_binding(record, label=f"exact evaluation token checkpoint {index}")
    proof_path = Path(receipt["decode_proof"]["path"])
    validate_decode_success(
        decode_proof := load_json(proof_path),
        archive_sha=archive_sha,
        raw_path=raw_path,
        authority_sources=authority_sources,
    )
    if any(
        receipt[key] != decode_proof[key]
        for key in ("archive", "member", "decoded_raw", "decode_log")
    ):
        raise PoseResolveError("exact-evaluation receipt differs from its decode proof")
    if receipt["token_checkpoints"] != decode_proof["token_checkpoints"]:
        raise PoseResolveError("exact-evaluation token checkpoints differ from decode proof")
    expected_evaluate_source = authority_sources["scorer"]["upstream/evaluate.py"]
    if receipt["evaluate_source"] != expected_evaluate_source:
        raise PoseResolveError("exact-evaluation receipt used a different evaluate.py")
    report_path = Path(receipt["evaluate_report"]["path"])
    parsed = parse_upstream_report(
        report_path.read_text(encoding="utf-8"),
        expected_archive_bytes=int(receipt["archive"]["bytes"]),
    )
    if (
        parsed["d_pose"] != receipt.get("d_pose_report_precision")
        or parsed["d_seg"] != receipt.get("d_seg_report_precision")
        or parsed["archive_bytes"] != receipt.get("archive_bytes")
    ):
        raise PoseResolveError("exact-evaluation receipt differs from its retained report")
    return receipt


def acquire_lock(path: Path, *, purpose: str = "ps135 writer"):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise PoseResolveError(f"another process owns the {purpose} lock: {path}") from error
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps({"pid": os.getpid(), "purpose": purpose, "utc": utc_now()})
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def process_scan_receipt() -> dict[str, object]:
    """Use canonical argv matching when possible, bounded lsof evidence otherwise."""

    tokens = (
        "pb1_receiver_realized_verdict",
        "train_levelset_witness",
        "train_witness_realized",
        "ddm_lv1_s2_nullspace",
        "contest_auth_eval",
        "evaluate.py",
        "ru1_endpoint_residual",
        "pb1_qdbs",
        "ddm_sd2_pr130_seg_decomposition_runner",
        "solve_f17_cbq_coefficients",
        "solve_f18_pr133_tail",
        "solve_f26_iterative_joint_carrier",
        "solve_joint_posenet_pairs",
        Path(__file__).name,
    )
    command = ["/bin/ps", "-axo", "command"]
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
    except OSError as error:
        completed = subprocess.CompletedProcess(
            command,
            126,
            stdout="",
            stderr=f"{type(error).__name__}: {error}",
        )
    if completed.returncode == 0:
        from tools.argv_role import process_table_entrypoint_holders

        holders = process_table_entrypoint_holders(
            completed.stdout,
            tokens,
            self_tokens=(Path(__file__).name,),
        )
        return {
            "surface": "canonical process table argv scan",
            "command": command,
            "returncode": 0,
            "holders": holders,
            "passes": not holders,
        }

    lsof_command = ["/usr/sbin/lsof", "-nP", "-c", "Python"]
    try:
        lsof = subprocess.run(
            lsof_command, text=True, capture_output=True, check=False
        )
    except OSError as error:
        lsof = subprocess.CompletedProcess(
            lsof_command,
            126,
            stdout="",
            stderr=f"{type(error).__name__}: {error}",
        )
    lines = lsof.stdout.splitlines()
    suspicious = sorted(
        {
            line
            for line in lines
            if any(token in line for token in tokens if token != Path(__file__).name)
        }
    )
    return {
        "surface": "bounded Python open-file census; argv process table sandbox-blocked",
        "canonical_command": command,
        "canonical_returncode": completed.returncode,
        "canonical_stderr": completed.stderr.strip(),
        "fallback_command": lsof_command,
        "fallback_returncode": lsof.returncode,
        "python_pid_count": len(
            {
                fields[1]
                for line in lines[1:]
                if len(fields := line.split()) >= 2 and fields[1].isdigit()
            }
        ),
        "suspicious_open_files": suspicious,
        "passes": lsof.returncode == 0 and not suspicious,
        "boundary": (
            "fallback is not a full argv census; admission also requires exclusive "
            "queue ownership plus an empty live dispatch-claim ledger"
        ),
    }


def active_dispatch_claims() -> dict[str, object]:
    command = [
        sys.executable,
        str(REPO / "tools" / "claim_lane_dispatch.py"),
        "summary",
        "--live-only",
        "--json",
        "--ttl-hours",
        "24",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PoseResolveError(f"dispatch-claim summary failed: {completed.stderr}")
    payload = json.loads(completed.stdout)
    return {
        "command": command,
        "active": payload.get("active", []),
        "active_count": payload.get("active_count"),
    }


def preflight(output: Path, bulk: Path, *, write: bool) -> dict[str, object]:
    verify_input_pins()
    if write:
        output.mkdir(parents=True, exist_ok=True)
        bulk.mkdir(parents=True, exist_ok=True)
    output_probe = output if output.exists() else output.parent
    bulk_probe = bulk if bulk.exists() else bulk.parent
    storage = {
        "candidate_store": {
            "path": str(output.resolve()),
            "free_bytes": shutil.disk_usage(output_probe).free,
            "required_free_bytes": 3_000_000_000,
        },
        "bulk_store": {
            "path": str(bulk.resolve()),
            "free_bytes": shutil.disk_usage(bulk_probe).free,
            "required_free_bytes": 8_000_000_000,
        },
    }
    storage_passes = all(
        item["free_bytes"] >= item["required_free_bytes"]
        for item in storage.values()
    )
    scan = process_scan_receipt()
    claims = active_dispatch_claims()
    result = {
        "schema": "ddm_ps135_preflight.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[scorer-free custody/storage/liveness preflight]",
        "score_claim": False,
        "inputs": input_pins(),
        "scorer_sources": scorer_source_pins(),
        "storage": storage,
        "storage_passes": storage_passes,
        "process_scan": scan,
        "dispatch_claims_before_ps135_claim": claims,
        "queue_ownership": {
            "arm": "ddm_ps135_pose_resolve",
            "owns_scorer": True,
            "source": ".omx/state/codex_arm_queue.jsonl latest live row",
        },
        "admission_ready_except_ps135_claim": bool(
            storage_passes and scan.get("passes") and claims.get("active_count") == 0
        ),
        "payload_routing": {
            "candidate_coefficients_and_archives": str(output.resolve()),
            "bulk_GT_and_scorer_outputs": str(bulk.resolve()),
            "reason": "Vertigo has limited headroom; charter-mandated candidate payloads stay there",
        },
    }
    if write:
        atomic_json(output / "preflight.json", result)
    return result


def configure_torch(threads: int):
    import torch

    if threads <= 0:
        raise PoseResolveError("torch thread count must be positive")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.manual_seed(20260810)
    np.random.seed(20260810)
    torch.use_deterministic_algorithms(True)
    return torch


def load_scorers(threads: int):
    torch = configure_torch(threads)
    _prepend_path(UPSTREAM)
    modules = importlib.import_module("modules")
    model = modules.DistortionNet().eval().to(torch.device("cpu"))
    model.load_state_dicts(
        modules.posenet_sd_path,
        modules.segnet_sd_path,
        torch.device("cpu"),
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return torch, model.posenet, model.segnet


def pad_batch(tensor, batch_size: int):
    if tensor.shape[0] <= 0 or tensor.shape[0] > batch_size:
        raise PoseResolveError("cannot pad invalid scorer batch")
    valid = int(tensor.shape[0])
    if valid < batch_size:
        tensor = __import__("torch").cat(
            (tensor, tensor[-1:].expand(batch_size - valid, *tensor.shape[1:])),
            dim=0,
        )
    return tensor, valid


def build_target_cache(
    bulk: Path,
    *,
    posenet,
    segnet,
    batch_size: int,
) -> dict[str, object]:
    torch = __import__("torch")
    _prepend_path(UPSTREAM)
    frame_utils = importlib.import_module("frame_utils")
    target_root = bulk / TARGET_CACHE_DIRNAME
    target_root.mkdir(parents=True, exist_ok=True)
    names = (UPSTREAM / "public_test_video_names.txt").read_text().splitlines()
    dataset = frame_utils.AVVideoDataset(
        names,
        data_dir=UPSTREAM / "videos",
        batch_size=batch_size,
        device=torch.device("cpu"),
        num_threads=2,
        seed=1234,
        prefetch_queue_depth=4,
    )
    dataset.prepare_data()
    target_batches: list[tuple[Any, Any]] = []
    chunk_start = 0
    chunk_index = 0
    chunks: list[dict[str, object]] = []
    observed_batch_sizes: list[int] = []
    started = time.time()

    def flush() -> None:
        nonlocal chunk_start, chunk_index
        if not target_batches:
            return
        pose = torch.cat([item[0] for item in target_batches], dim=0)
        seg = torch.cat([item[1] for item in target_batches], dim=0)
        chunk_end = chunk_start + int(pose.shape[0])
        path = target_root / f"chunk_{chunk_index:02d}_{chunk_start:04d}_{chunk_end:04d}.pt"
        payload = {
            "schema": TARGET_CHUNK_SCHEMA,
            "pair_start": chunk_start,
            "pair_end": chunk_end,
            "pose": pose,
            "seg": seg,
        }
        if path.exists():
            existing = torch.load(path, map_location="cpu", weights_only=False)
            if (
                existing.get("schema") != payload["schema"]
                or existing.get("pair_start") != chunk_start
                or existing.get("pair_end") != chunk_end
                or not torch.equal(existing.get("pose"), pose)
                or not torch.equal(existing.get("seg"), seg)
            ):
                raise PoseResolveError(f"existing target chunk differs: {path}")
        else:
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            torch.save(payload, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        chunks.append(
            {
                "pair_start": chunk_start,
                "pair_end": chunk_end,
                "payload": file_record(path),
            }
        )
        chunk_start = chunk_end
        chunk_index += 1
        target_batches.clear()

    with torch.inference_mode():
        for _, _, batch in dataset:
            batch = batch.to(torch.device("cpu"))
            observed_batch_sizes.append(int(batch.shape[0]))
            chw = batch.permute(0, 1, 4, 2, 3).float()
            pose = posenet(posenet.preprocess_input(chw))["pose"][:, :6]
            seg = segnet(segnet.preprocess_input(chw)).argmax(1)
            target_batches.append(
                (pose.float().cpu(), seg.to(torch.uint8).cpu())
            )
            if sum(int(item[0].shape[0]) for item in target_batches) >= TARGET_CHUNK_PAIRS:
                flush()
            print(
                json.dumps(
                    {
                        "stage": "targets",
                        "pairs": chunk_start
                        + sum(int(item[0].shape[0]) for item in target_batches),
                        "elapsed_seconds": time.time() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    flush()
    if chunk_start != N:
        raise PoseResolveError(f"AV target cache has {chunk_start} pairs, expected {N}")
    expected_batch_sizes = official_batch_sizes(batch_size)
    if observed_batch_sizes != expected_batch_sizes:
        raise PoseResolveError(
            "AV target cache did not use the untouched global scorer batch geometry"
        )
    manifest = {
        "schema": TARGET_CACHE_SCHEMA,
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": AXIS,
        "score_claim": False,
        "pair_count": N,
        "batch_size": batch_size,
        "observed_batch_sizes": observed_batch_sizes,
        "batch_geometry": f"{N // batch_size}x{batch_size}+{N % batch_size}",
        "final_partial_batch_padded": False,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "gt_decoder": "upstream.frame_utils.AVVideoDataset -> yuv420_to_rgb",
        "gt_decoder_required_function": "frame_utils.yuv420_to_rgb",
        "chunks": chunks,
        "scorer_sources": scorer_source_pins(),
        "elapsed_seconds": time.time() - started,
        "payloads_retained": True,
    }
    atomic_json(target_root / "manifest.json", manifest)
    return manifest


def load_target_cache(bulk: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    torch = __import__("torch")
    target_root = bulk / TARGET_CACHE_DIRNAME
    manifest = load_json(target_root / "manifest.json")
    if (
        manifest.get("schema") != TARGET_CACHE_SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("pair_count") != N
        or manifest.get("batch_size") != DEFAULT_BATCH_SIZE
        or manifest.get("observed_batch_sizes")
        != official_batch_sizes(DEFAULT_BATCH_SIZE)
        or manifest.get("final_partial_batch_padded") is not False
        or manifest.get("scorer_sources") != scorer_source_pins()
        or manifest.get("gt_decoder_required_function")
        != "frame_utils.yuv420_to_rgb"
    ):
        raise PoseResolveError("target manifest is not the pinned complete AV cache")
    pose_parts = []
    seg_parts = []
    cursor = 0
    for row in manifest["chunks"]:
        path = Path(row["payload"]["path"])
        if file_record(path) != row["payload"]:
            raise PoseResolveError(f"target chunk changed: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("schema") != TARGET_CHUNK_SCHEMA
            or payload.get("pair_start") != cursor
            or payload.get("pair_end") <= cursor
            or payload.get("pair_start") != row.get("pair_start")
            or payload.get("pair_end") != row.get("pair_end")
        ):
            raise PoseResolveError("target chunks are not contiguous")
        pose_parts.append(payload["pose"].float().numpy())
        seg_parts.append(payload["seg"].to(torch.uint8).numpy())
        cursor = int(payload["pair_end"])
    if cursor != N:
        raise PoseResolveError("target chunks do not close the n600 denominator")
    pose = np.concatenate(pose_parts).astype(np.float32, copy=False)
    seg = np.concatenate(seg_parts).astype(np.uint8, copy=False)
    if pose.shape != (N, POSE_DIMS) or seg.shape != (N, SCORER_H, SCORER_W):
        raise PoseResolveError("target cache tensor shape differs")
    return pose, seg, manifest


class ExactCarrierRenderer:
    def __init__(self, template: CarrierState, inflate_module, torch_module):
        self.torch = torch_module
        self.inflate = inflate_module
        basis = template.basis_codes.reshape(D, 3, CARRIER_H, CARRIER_W).astype(
            np.float32
        )
        basis *= template.basis_scales[:, None, None, None]
        basis_tensor = torch_module.from_numpy(basis)
        self.basis = inflate_module.normalized_basis(basis_tensor)
        self.scales = torch_module.from_numpy(
            template.coefficient_scales.astype(np.float32)
        )

    def render(self, codes: np.ndarray) -> np.ndarray:
        torch = self.torch
        import torch.nn.functional as functional

        values = np.asarray(codes, dtype=np.int16)
        if values.ndim != 2 or values.shape[1] != D:
            raise PoseResolveError("renderer needs [candidates,12] int12 codes")
        coefficient = torch.from_numpy(values.astype(np.float32)) * self.scales[None]
        with torch.inference_mode():
            carrier = torch.einsum("bk,kchw->bchw", coefficient, self.basis)
            carrier = carrier / math.sqrt(D)
            low = (127.5 + self.inflate.CARRIER_AMPLITUDE * carrier).clamp(
                0.0, 255.0
            ).round()
            high = functional.interpolate(
                low,
                size=(CAMERA_H, CAMERA_W),
                mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 255.0).round()
        return high.to(torch.uint8).permute(0, 2, 3, 1).numpy()


def raw_memmap() -> np.memmap:
    return np.memmap(
        LC2_RAW,
        mode="r",
        dtype=np.uint8,
        shape=(N * 2, CAMERA_H, CAMERA_W, 3),
    )


def lc2_master_provider(raw: np.memmap | None = None) -> MasterFrameProvider:
    interleaved = raw_memmap() if raw is None else raw
    if interleaved.shape != (N * 2, CAMERA_H, CAMERA_W, 3):
        raise PoseResolveError("LC2 interleaved decode has the wrong shape")
    return MasterFrameProvider(
        frames=interleaved[1::2],
        slaves=interleaved[0::2],
        binding={
            "schema": "ddm_ps135_master_provider.v1",
            "layout": "interleaved_slave_master_uint8",
            "pair_count": N,
            "payload": input_pins()["lc2_raw"],
            "semantic_sha256": sha256_bytes(load_lc2_source().semantic),
        },
    )


def pose_outputs(
    renderer: ExactCarrierRenderer,
    posenet,
    master_provider: MasterFrameProvider,
    codes: np.ndarray,
    rows: np.ndarray,
    batch_size: int,
    *,
    pad_partial: bool,
) -> np.ndarray:
    torch = __import__("torch")
    codes = np.asarray(codes, dtype=np.int16)
    rows = np.asarray(rows, dtype=np.int64)
    if codes.shape != (len(rows), D):
        raise PoseResolveError("pose candidate codes/rows differ")
    output = np.empty((len(rows), POSE_DIMS), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            chunk_codes = codes[start:end]
            chunk_rows = rows[start:end]
            slaves = renderer.render(chunk_codes)
            masters = master_provider.masters(chunk_rows)
            pairs = np.stack((slaves, masters), axis=1)
            tensor = torch.from_numpy(pairs).permute(0, 1, 4, 2, 3).float()
            if pad_partial:
                scorer_input, valid = pad_batch(tensor, batch_size)
            else:
                scorer_input, valid = tensor, int(tensor.shape[0])
            predicted = posenet(posenet.preprocess_input(scorer_input))["pose"][
                :valid, :6
            ]
            output[start:end] = predicted.float().cpu().numpy()
    return output


def scorer_baseline(
    bulk: Path,
    *,
    renderer: ExactCarrierRenderer,
    posenet,
    segnet,
    master_provider: MasterFrameProvider,
    codes: np.ndarray,
    pose_targets: np.ndarray,
    seg_targets: np.ndarray,
    batch_size: int,
    name: str,
    require_slave_parity: bool,
) -> dict[str, object]:
    torch = __import__("torch")
    root = bulk / "baseline_outputs" / name
    root.mkdir(parents=True, exist_ok=True)
    pose_parts = []
    seg_parts = []
    parity_mismatches = 0
    chunks = []
    started = time.time()
    with torch.inference_mode():
        for chunk_start in range(0, N, TARGET_CHUNK_PAIRS):
            chunk_end = min(chunk_start + TARGET_CHUNK_PAIRS, N)
            rows = np.arange(chunk_start, chunk_end, dtype=np.int64)
            outputs = pose_outputs(
                renderer,
                posenet,
                master_provider,
                codes[chunk_start:chunk_end],
                rows,
                batch_size,
                pad_partial=False,
            )
            rendered = []
            seg_pred = []
            for start in range(chunk_start, chunk_end, batch_size):
                end = min(start + batch_size, chunk_end)
                slave = renderer.render(codes[start:end])
                if require_slave_parity:
                    expected_slave = master_provider.expected_slaves(
                        np.arange(start, end, dtype=np.int64)
                    )
                    parity_mismatches += int(
                        np.count_nonzero(slave != expected_slave)
                    )
                rendered.append(slave)
                masters = master_provider.masters(
                    np.arange(start, end, dtype=np.int64)
                )
                master_tensor = torch.from_numpy(masters).permute(0, 3, 1, 2).float()
                pairs = torch.stack((master_tensor, master_tensor), dim=1)
                predicted = segnet(segnet.preprocess_input(pairs)).argmax(1)
                seg_pred.append(predicted.to(torch.uint8).cpu().numpy())
            rendered_payload = np.concatenate(rendered)
            seg_payload = np.concatenate(seg_pred)
            path = root / f"chunk_{chunk_start:04d}_{chunk_end:04d}.npz"
            record = atomic_npz(
                path,
                pair_start=np.asarray(chunk_start, dtype=np.int32),
                pair_end=np.asarray(chunk_end, dtype=np.int32),
                pose_outputs=outputs,
                seg_argmax=seg_payload,
                rendered_slaves=rendered_payload,
            )
            chunks.append(
                {"pair_start": chunk_start, "pair_end": chunk_end, "payload": record}
            )
            pose_parts.append(outputs)
            seg_parts.append(seg_payload)
            print(
                json.dumps(
                    {
                        "stage": f"baseline_{name}",
                        "pairs": chunk_end,
                        "elapsed_seconds": time.time() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    pose = np.concatenate(pose_parts)
    seg = np.concatenate(seg_parts)
    errors = pose_pair_errors(pose, pose_targets)
    d_pose = mean_pose_error(errors)
    d_seg = float(np.not_equal(seg, seg_targets).mean())
    receipt = {
        "schema": "ddm_ps135_baseline_scorer.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": AXIS,
        "score_claim": False,
        "pair_count": N,
        "batch_size": batch_size,
        "d_pose": d_pose,
        "d_seg": d_seg,
        "pose_outputs": atomic_numpy(root / "pose_outputs.float32.npy", pose),
        "pair_errors": atomic_numpy(root / "pair_errors.float64.npy", errors),
        "seg_argmax": atomic_numpy(root / "seg_argmax.uint8.npy", seg),
        "chunks": chunks,
        "renderer_vs_retained_raw_byte_mismatches": parity_mismatches,
        "renderer_parity_required": require_slave_parity,
        "renderer_parity_passes": not require_slave_parity or parity_mismatches == 0,
        "master_provider": master_provider.binding,
        "elapsed_seconds": time.time() - started,
        "payloads_retained": True,
    }
    if require_slave_parity and parity_mismatches:
        raise PoseResolveError(f"{name} carrier renderer differs from retained raw")
    atomic_json(root / "receipt.json", receipt)
    return receipt


def evaluate_start(
    output: Path,
    bulk: Path,
    *,
    name: str,
    carrier: bytes,
    source,
    posenet,
    segnet,
    master_provider: MasterFrameProvider,
    pose_targets: np.ndarray,
    seg_targets: np.ndarray,
    batch_size: int,
    expected_raw_parity: bool,
    archive_builder: ArchiveBuilder = build_candidate_archive,
) -> tuple[CarrierState, dict[str, object]]:
    state = decode_carrier(carrier)
    _, _, inflate = import_runtime_modules()
    renderer = ExactCarrierRenderer(state, inflate, __import__("torch"))
    archive, carrier_stream, parseback = archive_builder(carrier, source)
    scorer = scorer_baseline(
        bulk,
        renderer=renderer,
        posenet=posenet,
        segnet=segnet,
        master_provider=master_provider,
        codes=state.codes,
        pose_targets=pose_targets,
        seg_targets=seg_targets,
        batch_size=batch_size,
        name=name,
        require_slave_parity=expected_raw_parity,
    )
    metadata = {
        "stage": "start",
        "start_name": name,
        "axis": AXIS,
        "score_claim": False,
        "d_pose": scorer["d_pose"],
        "d_seg": scorer["d_seg"],
        "score": score(scorer["d_seg"], scorer["d_pose"], len(archive)),
        "public_warm_start": name == "public_pr133_complete_carrier",
    }
    bundle = retain_candidate_bundle(
        output / "leg_a" / "starts" / name,
        carrier=carrier,
        carrier_stream=carrier_stream,
        archive=archive,
        codes=state.codes,
        parseback=parseback,
        metadata=metadata,
    )
    bundle["scorer"] = scorer
    atomic_json(output / "leg_a" / "starts" / name / "receipt.json", bundle)
    return state, bundle


def candidate_grid_for_row(
    current: np.ndarray,
    current_output: np.ndarray,
    target: np.ndarray,
    renderer: ExactCarrierRenderer,
    posenet,
    master_provider: MasterFrameProvider,
    row: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    (
        solve_damped_least_squares,
        quantize_int12_update,
        rank_neighbour_dimensions,
        nearby_int12_candidates,
    ) = import_joint_primitives()

    fd = [np.asarray(current, dtype=np.int32)]
    for dimension in range(D):
        for delta in (-1, 1):
            candidate = np.asarray(current, dtype=np.int32).copy()
            candidate[dimension] = np.clip(
                candidate[dimension] + delta, -2048, 2047
            )
            fd.append(candidate)
    fd_codes = np.stack(fd).astype(np.int16)
    fd_outputs = pose_outputs(
        renderer,
        posenet,
        master_provider,
        fd_codes,
        np.full(len(fd_codes), row, dtype=np.int64),
        batch_size,
        pad_partial=True,
    )
    population_to_search_drift = float(
        np.max(np.abs(fd_outputs[0].astype(np.float64) - current_output))
    )
    jacobian = np.empty((POSE_DIMS, D), dtype=np.float64)
    for dimension in range(D):
        negative = fd_outputs[1 + 2 * dimension].astype(np.float64)
        positive = fd_outputs[2 + 2 * dimension].astype(np.float64)
        if current[dimension] <= -2048:
            jacobian[:, dimension] = positive - fd_outputs[0]
        elif current[dimension] >= 2047:
            jacobian[:, dimension] = fd_outputs[0] - negative
        else:
            jacobian[:, dimension] = 0.5 * (positive - negative)
    solve = solve_damped_least_squares(
        jacobian,
        target.astype(np.float64) - fd_outputs[0].astype(np.float64),
        damping=GN_DAMPING,
        max_code_step=MAX_CODE_STEP,
    )
    centre = quantize_int12_update(current.astype(np.int32), solve.update)
    active = rank_neighbour_dimensions(jacobian, solve.update, NEIGHBOUR_DIMS)
    gn = nearby_int12_candidates(
        current.astype(np.int32),
        centre,
        active_dimensions=active,
        radius=NEIGHBOUR_RADIUS,
    )
    combined: list[np.ndarray] = []
    seen: set[bytes] = set()
    for candidate in (*fd, *gn):
        value = np.asarray(candidate, dtype=np.int16)
        key = value.tobytes()
        if key not in seen:
            combined.append(value)
            seen.add(key)
    valid = len(combined)
    if valid > GRID_MAX:
        raise PoseResolveError("candidate grid exceeded its derived maximum")
    while len(combined) < GRID_MAX:
        combined.append(np.asarray(current, dtype=np.int16))
    grid_codes = np.stack(combined)
    grid_outputs = pose_outputs(
        renderer,
        posenet,
        master_provider,
        grid_codes,
        np.full(len(grid_codes), row, dtype=np.int64),
        batch_size,
        pad_partial=True,
    )
    errors = pose_pair_errors(
        grid_outputs[:valid], np.repeat(target[None], valid, axis=0)
    )
    l1 = np.abs(grid_codes[:valid].astype(np.int32) - current.astype(np.int32)).sum(1)
    order = sorted(
        range(valid),
        key=lambda index: (
            float(errors[index]),
            int(l1[index]),
            tuple(int(value) for value in grid_codes[index]),
        ),
    )
    best = order[0]
    current_error = float(pose_pair_errors(fd_outputs[:1], target[None])[0])
    if float(errors[best]) >= current_error - 1e-15:
        best = 0
    return {
        "fd_codes": fd_codes,
        "fd_outputs": fd_outputs,
        "jacobian": jacobian,
        "gn_update": solve.update,
        "gn_rank": np.asarray(solve.rank, dtype=np.int16),
        "gn_condition": np.asarray(solve.condition, dtype=np.float64),
        "gn_ridge_lambda": np.asarray(solve.ridge_lambda, dtype=np.float64),
        "gn_centre": centre.astype(np.int16),
        "active_dimensions": np.asarray(active, dtype=np.int8),
        "grid_codes": grid_codes,
        "grid_outputs": grid_outputs,
        "grid_valid": np.asarray(valid, dtype=np.int16),
        "grid_errors": np.pad(
            errors,
            (0, GRID_MAX - valid),
            constant_values=np.inf,
        ),
        "best_index": np.asarray(best, dtype=np.int16),
        "best_codes": grid_codes[best],
        "best_output": grid_outputs[best],
        "best_error": np.asarray(errors[best], dtype=np.float64),
        "current_error": np.asarray(current_error, dtype=np.float64),
        "population_to_search_max_abs_pose_drift": np.asarray(
            population_to_search_drift, dtype=np.float64
        ),
    }


def solve_pass_chunks(
    pass_root: Path,
    *,
    template: CarrierState,
    current_codes: np.ndarray,
    current_outputs: np.ndarray,
    pose_targets: np.ndarray,
    posenet,
    master_provider: MasterFrameProvider,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    _, _, inflate = import_runtime_modules()
    renderer = ExactCarrierRenderer(template, inflate, __import__("torch"))
    proposed_codes = np.asarray(current_codes, dtype=np.int16).copy()
    proposed_outputs = np.asarray(current_outputs, dtype=np.float32).copy()
    proposed_errors = np.empty(N, dtype=np.float64)
    chunks: list[dict[str, object]] = []
    chunk_root = pass_root / "search_chunks"
    started = time.time()
    template_sha = sha256_array(template.basis_codes) + ":" + sha256_array(
        template.coefficient_scales
    )
    master_binding_json = canonical_json(master_provider.binding)
    for chunk_start in range(0, N, SEARCH_CHUNK_PAIRS):
        chunk_end = min(chunk_start + SEARCH_CHUNK_PAIRS, N)
        path = chunk_root / f"chunk_{chunk_start:04d}_{chunk_end:04d}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as payload:
                if (
                    int(payload["pair_start"]) != chunk_start
                    or int(payload["pair_end"]) != chunk_end
                    or not np.array_equal(
                        payload["input_codes"], current_codes[chunk_start:chunk_end]
                    )
                    or str(payload["template_sha"].item()) != template_sha
                    or str(payload["target_sha"].item())
                    != sha256_array(pose_targets[chunk_start:chunk_end])
                    or str(payload["input_outputs_sha"].item())
                    != sha256_array(current_outputs[chunk_start:chunk_end])
                    or "master_binding_json" not in payload.files
                    or str(payload["master_binding_json"].item())
                    != master_binding_json
                ):
                    raise PoseResolveError(f"resume chunk input differs: {path}")
                proposed_codes[chunk_start:chunk_end] = payload["best_codes"]
                proposed_outputs[chunk_start:chunk_end] = payload["best_outputs"]
                proposed_errors[chunk_start:chunk_end] = payload["best_errors"]
            chunks.append(
                {
                    "pair_start": chunk_start,
                    "pair_end": chunk_end,
                    "payload": file_record(path),
                    "resumed": True,
                }
            )
            continue
        rows = []
        for row in range(chunk_start, chunk_end):
            rows.append(
                candidate_grid_for_row(
                    current_codes[row],
                    current_outputs[row],
                    pose_targets[row],
                    renderer,
                    posenet,
                    master_provider,
                    row,
                    batch_size,
                )
            )
        arrays: dict[str, np.ndarray] = {
            "pair_start": np.asarray(chunk_start, dtype=np.int32),
            "pair_end": np.asarray(chunk_end, dtype=np.int32),
            "input_codes": np.asarray(
                current_codes[chunk_start:chunk_end], dtype=np.int16
            ),
            "template_sha": np.asarray(template_sha),
            "target_sha": np.asarray(
                sha256_array(pose_targets[chunk_start:chunk_end])
            ),
            "input_outputs_sha": np.asarray(
                sha256_array(current_outputs[chunk_start:chunk_end])
            ),
            "master_binding_json": np.asarray(master_binding_json),
        }
        for key in rows[0]:
            arrays[key if key not in {"best_codes", "best_output", "best_error"} else {
                "best_codes": "best_codes",
                "best_output": "best_outputs",
                "best_error": "best_errors",
            }[key]] = np.stack([row[key] for row in rows])
        record = atomic_npz(path, **arrays)
        proposed_codes[chunk_start:chunk_end] = arrays["best_codes"]
        proposed_outputs[chunk_start:chunk_end] = arrays["best_outputs"]
        proposed_errors[chunk_start:chunk_end] = arrays["best_errors"]
        chunks.append(
            {
                "pair_start": chunk_start,
                "pair_end": chunk_end,
                "payload": record,
                "resumed": False,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "joint_pass",
                    "pass_root": str(pass_root),
                    "pairs": chunk_end,
                    "elapsed_seconds": time.time() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return proposed_codes, proposed_outputs, proposed_errors, chunks


def compile_jrd_reusable_prior_receipt() -> dict[str, object]:
    """Compile #453/#460 as dormant ordering data, never as an actuator."""

    source_records = {
        "policy": pinned_file_record(
            JRD_PRIOR_POLICY,
            label="#460 JRD reusable-prior policy",
            digest=JRD_PRIOR_POLICY_SHA256,
        ),
        "harvest_json": pinned_file_record(
            JRD_PRIOR_HARVEST_JSON,
            label="#460 JRD reusable-prior harvest JSON",
            digest=JRD_PRIOR_HARVEST_JSON_SHA256,
        ),
        "harvest_memo": pinned_file_record(
            JRD_PRIOR_HARVEST_MEMO,
            label="#460 JRD reusable-prior harvest memo",
            digest=JRD_PRIOR_HARVEST_MEMO_SHA256,
        ),
        "response_curves": pinned_file_record(
            JRD_PR110_RESPONSE_CURVES,
            label="#453 JRD response curves",
            digest=JRD_PR110_RESPONSE_CURVES_SHA256,
        ),
        "n600_receipt": pinned_file_record(
            JRD_PR110_MEASUREMENT_RECEIPT,
            label="#453 JRD n600 receipt",
            digest=JRD_PR110_MEASUREMENT_RECEIPT_SHA256,
        ),
    }
    harvest = load_json(JRD_PRIOR_HARVEST_JSON)
    if harvest.get("schema") != "jrd_reusable_priors_harvest.v1":
        raise PoseResolveError("#460 JRD harvest schema differs")
    authority = harvest.get("authority")
    if not isinstance(authority, dict) or any(
        authority.get(key) is not False
        for key in ("promotion_eligible", "score_claim", "pointer_moved")
    ):
        raise PoseResolveError("#460 JRD harvest authority boundary differs")
    harvest_sources = harvest.get("sources")
    expected_harvest_sources = {
        "response_curves": {
            "path": str(JRD_PR110_RESPONSE_CURVES.relative_to(REPO)),
            "sha256": JRD_PR110_RESPONSE_CURVES_SHA256,
        },
        "n600_receipt": {
            "path": str(JRD_PR110_MEASUREMENT_RECEIPT.relative_to(REPO)),
            "sha256": JRD_PR110_MEASUREMENT_RECEIPT_SHA256,
        },
    }
    if not isinstance(harvest_sources, dict) or any(
        harvest_sources.get(key) != value
        for key, value in expected_harvest_sources.items()
    ):
        raise PoseResolveError("#460 JRD harvest does not bind the #453 sources")

    module_name = f"_ddm_ps135_jrd_priors_{os.getpid()}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, JRD_PRIOR_POLICY)
        if spec is None or spec.loader is None:
            raise PoseResolveError("#460 JRD policy loader is unavailable")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        compiled = module.JrdReusablePriorPolicy().compile_warm_start(None)
    except PoseResolveError:
        raise
    except Exception as error:
        raise PoseResolveError("#460 JRD dormant policy compilation failed") from error
    finally:
        sys.modules.pop(module_name, None)
    if not isinstance(compiled, dict):
        raise PoseResolveError("#460 JRD policy did not compile a JSON object")
    required_dormant = {
        "state": "DORMANT_N1_SCREEN",
        "active": False,
        "screen_eval_pairs": 1,
        "activation_eval_pairs": N,
        "hypothesis_evidence_label": "ASSUMED_OPERATOR_WARM_START",
        "precision_actuation": "REFUSED_PENDING_N600_CONFIRMATION",
        "activation_receipt": None,
        "live_trainer_argv": [],
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    if any(compiled.get(key) != value for key, value in required_dormant.items()):
        raise PoseResolveError("#460 JRD policy no longer compiles dormant and fail-closed")
    return {
        "schema": "ddm_ps135_jrd_reusable_prior_receipt.v1",
        "complete": True,
        "source_bindings": source_records,
        "harvest_source_bindings_verified": True,
        "compiled_no_confirmation": compiled,
        "consumption": {
            "disposition": "ORDERING_DATA_ONLY_NOT_ACTUATED",
            "allowed_use": "initialize n600 measurement ordering",
            "precision_assignment_from_pr110": False,
            "signed_int12_finisher_relationship": (
                "distinct exact lattice traversal; the dormant PR110 int8 plane "
                "hypotheses do not select LC2 steps or precisions"
            ),
        },
        "score_claim": False,
        "promotion_eligible": False,
    }


def retain_jrd_reusable_prior_receipt(finisher_root: Path) -> dict[str, object]:
    """Retain one deterministic prior receipt and refuse divergent resume bytes."""

    receipt = compile_jrd_reusable_prior_receipt()
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return persist_exact(finisher_root / "jrd_reusable_prior_receipt.json", payload)


def jrd_protocol_binding(
    template: CarrierState,
    *,
    batch_size: int,
    jrd_prior_receipt: dict[str, object],
    master_provider: MasterFrameProvider,
) -> dict[str, object]:
    """All implementation/runtime facts that make one JRD chunk reusable."""

    verify_file_record_binding(
        jrd_prior_receipt,
        label="JRD reusable-prior receipt",
    )
    return {
        "schema": JRD_CHUNK_SCHEMA,
        "runner": file_record(Path(__file__)),
        "template_carrier_sha256": sha256_bytes(template.carrier),
        "basis_scales_sha256": sha256_array(template.basis_scales),
        "basis_codes_sha256": sha256_array(template.basis_codes),
        "coefficient_scales_sha256": sha256_array(template.coefficient_scales),
        "jrd_lineage_steps": list(JRD_LINEAGE_STEPS),
        "jrd_grid_size": JRD_GRID_SIZE,
        "batch_size": batch_size,
        "candidate_partial_batch_padded": True,
        "population_partial_batch_padded": False,
        "n": N,
        "dimensions": D,
        "pose_dimensions": POSE_DIMS,
        "authority_sources": exact_authority_source_pins(),
        "jrd_reusable_prior_receipt": jrd_prior_receipt,
        "master_provider": master_provider.binding,
    }


def jrd_chunk_binding_json(
    protocol: dict[str, object],
    *,
    chunk_start: int,
    chunk_end: int,
    input_codes: np.ndarray,
    input_outputs: np.ndarray,
    pose_targets: np.ndarray,
) -> str:
    return canonical_json(
        {
            **protocol,
            "pair_start": chunk_start,
            "pair_end": chunk_end,
            "input_codes_sha256": sha256_array(input_codes),
            "input_outputs_sha256": sha256_array(input_outputs),
            "pose_targets_sha256": sha256_array(pose_targets),
        }
    )


def solve_jrd_finisher_chunks(
    finisher_root: Path,
    *,
    template: CarrierState,
    current_codes: np.ndarray,
    current_outputs: np.ndarray,
    pose_targets: np.ndarray,
    posenet,
    master_provider: MasterFrameProvider,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Exact terminal JRD lineage ladder, distinct from the GN pass."""

    _, _, inflate = import_runtime_modules()
    renderer = ExactCarrierRenderer(template, inflate, __import__("torch"))
    proposed_codes = np.asarray(current_codes, dtype=np.int16).copy()
    proposed_outputs = np.asarray(current_outputs, dtype=np.float32).copy()
    proposed_errors = np.empty(N, dtype=np.float64)
    chunks: list[dict[str, object]] = []
    chunk_root = finisher_root / "search_chunks"
    started = time.time()
    prior_receipt = retain_jrd_reusable_prior_receipt(finisher_root)
    protocol = jrd_protocol_binding(
        template,
        batch_size=batch_size,
        jrd_prior_receipt=prior_receipt,
        master_provider=master_provider,
    )
    for chunk_start in range(0, N, SEARCH_CHUNK_PAIRS):
        chunk_end = min(chunk_start + SEARCH_CHUNK_PAIRS, N)
        path = chunk_root / f"chunk_{chunk_start:04d}_{chunk_end:04d}.npz"
        binding_json = jrd_chunk_binding_json(
            protocol,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            input_codes=current_codes[chunk_start:chunk_end],
            input_outputs=current_outputs[chunk_start:chunk_end],
            pose_targets=pose_targets[chunk_start:chunk_end],
        )
        if path.is_file():
            with np.load(path, allow_pickle=False) as payload:
                if (
                    "binding_json" not in payload.files
                    or str(payload["binding_json"].item()) != binding_json
                    or payload["best_codes"].dtype != np.int16
                    or payload["best_codes"].shape != (chunk_end - chunk_start, D)
                    or payload["best_outputs"].dtype != np.float32
                    or payload["best_outputs"].shape
                    != (chunk_end - chunk_start, POSE_DIMS)
                    or payload["best_errors"].dtype != np.float64
                    or payload["best_errors"].shape != (chunk_end - chunk_start,)
                ):
                    raise PoseResolveError(f"JRD resume chunk input differs: {path}")
                recomputed_errors = pose_pair_errors(
                    payload["best_outputs"], pose_targets[chunk_start:chunk_end]
                )
                if not np.array_equal(recomputed_errors, payload["best_errors"]):
                    raise PoseResolveError(f"JRD resume chunk error proof differs: {path}")
                proposed_codes[chunk_start:chunk_end] = payload["best_codes"]
                proposed_outputs[chunk_start:chunk_end] = payload["best_outputs"]
                proposed_errors[chunk_start:chunk_end] = payload["best_errors"]
            chunks.append(
                {
                    "pair_start": chunk_start,
                    "pair_end": chunk_end,
                    "payload": file_record(path),
                    "resumed": True,
                }
            )
            continue
        rows: list[dict[str, np.ndarray]] = []
        for row in range(chunk_start, chunk_end):
            candidates = [current_codes[row].astype(np.int32)]
            for dimension in range(D):
                for step in JRD_LINEAGE_STEPS:
                    for direction in (-1, 1):
                        candidate = current_codes[row].astype(np.int32).copy()
                        candidate[dimension] = np.clip(
                            candidate[dimension] + direction * step, -2048, 2047
                        )
                        candidates.append(candidate)
            unique: list[np.ndarray] = []
            seen: set[bytes] = set()
            for candidate in candidates:
                value = np.asarray(candidate, dtype=np.int16)
                key = value.tobytes()
                if key not in seen:
                    seen.add(key)
                    unique.append(value)
            valid = len(unique)
            while len(unique) < JRD_GRID_SIZE:
                unique.append(current_codes[row].astype(np.int16))
            codes = np.stack(unique)
            outputs = pose_outputs(
                renderer,
                posenet,
                master_provider,
                codes,
                np.full(JRD_GRID_SIZE, row, dtype=np.int64),
                batch_size,
                pad_partial=True,
            )
            errors = pose_pair_errors(
                outputs[:valid], np.repeat(pose_targets[row][None], valid, axis=0)
            )
            l1 = np.abs(
                codes[:valid].astype(np.int32)
                - current_codes[row].astype(np.int32)
            ).sum(1)
            best = min(
                range(valid),
                key=lambda index: (
                    float(errors[index]),
                    int(l1[index]),
                    tuple(int(value) for value in codes[index]),
                ),
            )
            if float(errors[best]) >= float(errors[0]) - 1e-15:
                best = 0
            rows.append(
                {
                    "grid_codes": codes,
                    "grid_outputs": outputs,
                    "grid_valid": np.asarray(valid, dtype=np.int16),
                    "grid_errors": np.pad(
                        errors,
                        (0, JRD_GRID_SIZE - valid),
                        constant_values=np.inf,
                    ),
                    "best_codes": codes[best],
                    "best_outputs": outputs[best],
                    "best_errors": np.asarray(errors[best], dtype=np.float64),
                }
            )
        arrays: dict[str, np.ndarray] = {
            "pair_start": np.asarray(chunk_start, dtype=np.int32),
            "pair_end": np.asarray(chunk_end, dtype=np.int32),
            "binding_json": np.asarray(binding_json),
        }
        for key in rows[0]:
            arrays[key] = np.stack([row[key] for row in rows])
        record = atomic_npz(path, **arrays)
        proposed_codes[chunk_start:chunk_end] = arrays["best_codes"]
        proposed_outputs[chunk_start:chunk_end] = arrays["best_outputs"]
        proposed_errors[chunk_start:chunk_end] = arrays["best_errors"]
        chunks.append(
            {
                "pair_start": chunk_start,
                "pair_end": chunk_end,
                "payload": record,
                "resumed": False,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "jrd_finisher",
                    "finisher_root": str(finisher_root),
                    "pairs": chunk_end,
                    "elapsed_seconds": time.time() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return proposed_codes, proposed_outputs, proposed_errors, chunks


def rate_aware_select(
    pass_root: Path,
    *,
    source,
    template: CarrierState,
    current_codes: np.ndarray,
    current_outputs: np.ndarray,
    current_errors: np.ndarray,
    current_archive: bytes,
    proposed_codes: np.ndarray,
    proposed_outputs: np.ndarray,
    proposed_errors: np.ndarray,
    d_seg: float,
    archive_builder: ArchiveBuilder = build_candidate_archive,
) -> dict[str, object]:
    moved = np.flatnonzero(np.any(proposed_codes != current_codes, axis=1))
    gains = current_errors[moved] - proposed_errors[moved]
    revert_order = moved[
        np.lexsort(
            (
                moved,
                gains,
            )
        )
    ]
    working_codes = proposed_codes.copy()
    working_outputs = proposed_outputs.copy()
    working_errors = proposed_errors.copy()
    candidates: list[dict[str, object]] = []
    variants_root = pass_root / "archive_variants"

    def materialize(index: int, kind: str, reverted_rows: int) -> None:
        storage_preflight = require_vertigo_free_space(
            variants_root / f"variant_{index:04d}_{kind}",
            required_free_bytes=RATE_CANDIDATE_REQUIRED_FREE_BYTES,
            stage=f"rate_candidate:{pass_root.name}:{index:04d}:{kind}",
        )
        carrier = encode_carrier(template, working_codes)
        archive, carrier_stream, parseback = archive_builder(carrier, source)
        d_pose = mean_pose_error(working_errors)
        candidate_score = score(d_seg, d_pose, len(archive))
        bundle = retain_candidate_bundle(
            variants_root / f"variant_{index:04d}_{kind}",
            carrier=carrier,
            carrier_stream=carrier_stream,
            archive=archive,
            codes=working_codes,
            parseback=parseback,
            metadata={
                "stage": "rate_aware_pass_selection",
                "variant": index,
                "kind": kind,
                "reverted_rows": reverted_rows,
                "remaining_changed_rows": int(
                    np.count_nonzero(np.any(working_codes != current_codes, axis=1))
                ),
                "d_seg": d_seg,
                "d_pose": d_pose,
                "score": candidate_score,
                "proposal_surface": "row-local padded-batch candidate screening",
                "rate_search_scope": (
                    "heuristic nested reversion ordered by proposal pose gain; "
                    "every materialized archive is exact and every variant receives "
                    "a global-population refresh before selection"
                ),
                "hard_byte_ceiling": LC2_ARCHIVE_BYTES,
                "byte_ceiling_passes": len(archive) <= LC2_ARCHIVE_BYTES,
                "axis": AXIS,
                "score_claim": False,
                "storage_preflight": storage_preflight,
            },
        )
        outputs_record = atomic_numpy(
            variants_root
            / f"variant_{index:04d}_{kind}"
            / "pose_outputs.float32.npy",
            working_outputs,
        )
        errors_record = atomic_numpy(
            variants_root
            / f"variant_{index:04d}_{kind}"
            / "pair_errors.float64.npy",
            working_errors,
        )
        candidates.append(
            {
                "variant": index,
                "kind": kind,
                "bundle": bundle,
                "pose_outputs": outputs_record,
                "pair_errors": errors_record,
                "d_pose": d_pose,
                "archive_bytes": len(archive),
                "archive_sha256": sha256_bytes(archive),
                "score": candidate_score,
                "eligible": len(archive) <= LC2_ARCHIVE_BYTES,
                "codes": working_codes.copy(),
                "outputs": working_outputs.copy(),
                "errors": working_errors.copy(),
                "archive": archive,
            }
        )

    materialize(0, "aggregate", 0)
    for index, row in enumerate(revert_order, 1):
        working_codes[row] = current_codes[row]
        working_outputs[row] = current_outputs[row]
        working_errors[row] = current_errors[row]
        materialize(index, "rate_trim", index)
    # The incumbent is an eligible exact control even if no proposed row moved.
    incumbent_d_pose = mean_pose_error(current_errors)
    incumbent_score = score(d_seg, incumbent_d_pose, len(current_archive))
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    eligible.append(
        {
            "variant": -1,
            "kind": "incumbent",
            "bundle": None,
            "pose_outputs": None,
            "pair_errors": None,
            "d_pose": incumbent_d_pose,
            "archive_bytes": len(current_archive),
            "archive_sha256": sha256_bytes(current_archive),
            "score": incumbent_score,
            "eligible": True,
            "codes": current_codes.copy(),
            "outputs": current_outputs.copy(),
            "errors": current_errors.copy(),
            "archive": current_archive,
        }
    )
    selected = min(
        eligible,
        key=lambda item: (
            float(item["score"]),
            int(item["archive_bytes"]),
            str(item["archive_sha256"]),
            int(item["variant"]),
        ),
    )
    public_candidates = [
        {key: value for key, value in candidate.items() if key not in {
            "codes", "outputs", "errors", "archive"
        }}
        for candidate in candidates
    ]
    return {
        "selected": selected,
        "candidate_rows": public_candidates,
        "_materialized_candidates": candidates,
        "moved_rows_proposed": len(moved),
        "rate_trim_denominator": len(revert_order),
    }


def exact_population_refresh(
    selection: dict[str, object],
    *,
    template: CarrierState,
    current_codes: np.ndarray,
    current_outputs: np.ndarray,
    current_errors: np.ndarray,
    current_archive: bytes,
    d_seg: float,
    pose_targets: np.ndarray,
    posenet,
    master_provider: MasterFrameProvider,
    batch_size: int,
) -> dict[str, object]:
    """Select only after every retained rate variant has official n600 geometry."""

    materialized = selection.pop("_materialized_candidates", None)
    if not isinstance(materialized, list) or not materialized:
        raise PoseResolveError("exact refresh lacks its retained materialized variants")
    aggregate = next(
        (
            candidate
            for candidate in materialized
            if candidate["variant"] == 0 and candidate["kind"] == "aggregate"
        ),
        None,
    )
    if aggregate is None:
        raise PoseResolveError("exact refresh lacks the aggregate proposal")
    aggregate_codes = np.asarray(aggregate["codes"], dtype=np.int16)
    aggregate_changed = np.any(aggregate_codes != current_codes, axis=1)
    exact_forward_count = 0
    if np.any(aggregate_changed):
        _, _, inflate = import_runtime_modules()
        renderer = ExactCarrierRenderer(template, inflate, __import__("torch"))
        aggregate_outputs = pose_outputs(
            renderer,
            posenet,
            master_provider,
            aggregate_codes,
            np.arange(N, dtype=np.int64),
            batch_size,
            pad_partial=False,
        )
        exact_forward_count = 1
    else:
        aggregate_outputs = np.asarray(current_outputs, dtype=np.float32).copy()

    exact_rows: list[dict[str, object]] = []
    for candidate in materialized:
        codes = np.asarray(candidate["codes"], dtype=np.int16)
        changed = np.any(codes != current_codes, axis=1)
        if np.any(codes[changed] != aggregate_codes[changed]):
            raise PoseResolveError("rate variants are not nested reversions of the aggregate")
        outputs = np.asarray(current_outputs, dtype=np.float32).copy()
        outputs[changed] = aggregate_outputs[changed]
        errors = pose_pair_errors(outputs, pose_targets)
        candidate_d_pose = mean_pose_error(errors)
        candidate_score = score(
            d_seg, candidate_d_pose, int(candidate["archive_bytes"])
        )
        candidate_root = Path(candidate["bundle"]["records"]["archive"]["path"]).parent
        outputs_record = atomic_numpy(
            candidate_root / "exact_population_pose_outputs.float32.npy", outputs
        )
        errors_record = atomic_numpy(
            candidate_root / "exact_population_pair_errors.float64.npy", errors
        )
        candidate.update(
            {
                "outputs": outputs,
                "errors": errors,
                "d_pose": candidate_d_pose,
                "score": candidate_score,
                "exact_population_refreshed": True,
                "exact_pose_outputs": outputs_record,
                "exact_pair_errors": errors_record,
            }
        )
        public_candidate = next(
            public
            for public in selection["candidate_rows"]
            if public["variant"] == candidate["variant"]
        )
        proposal_d_pose = public_candidate["d_pose"]
        public_candidate.update(
            {
                "exact_population_refreshed": True,
                "exact_d_pose": candidate_d_pose,
                "exact_score": candidate_score,
                "exact_pose_outputs": outputs_record,
                "exact_pair_errors": errors_record,
            }
        )
        row = {
            "schema": "ddm_ps135_exact_population_variant.v2",
            "complete": True,
            "written_at_utc": utc_now(),
            "axis": AXIS,
            "score_claim": False,
            "batch_geometry": f"{N // batch_size}x{batch_size}+{N % batch_size}",
            "final_partial_batch_padded": False,
            "candidate_variant": candidate["variant"],
            "candidate_kind": candidate["kind"],
            "candidate_archive_bytes": candidate["archive_bytes"],
            "candidate_archive_sha256": candidate["archive_sha256"],
            "proposal_d_pose": proposal_d_pose,
            "exact_d_pose": candidate_d_pose,
            "exact_score": candidate_score,
            "eligible": candidate["eligible"],
            "changed_rows": int(np.count_nonzero(changed)),
            "pose_outputs": outputs_record,
            "pair_errors": errors_record,
            "master_provider": master_provider.binding,
        }
        atomic_json(candidate_root / "exact_population_refresh.json", row)
        exact_rows.append(row)

    incumbent_d_pose = mean_pose_error(current_errors)
    incumbent_score = score(d_seg, incumbent_d_pose, len(current_archive))
    incumbent = {
        "variant": -1,
        "kind": "incumbent",
        "bundle": None,
        "pose_outputs": None,
        "pair_errors": None,
        "d_pose": incumbent_d_pose,
        "archive_bytes": len(current_archive),
        "archive_sha256": sha256_bytes(current_archive),
        "score": incumbent_score,
        "eligible": True,
        "codes": current_codes.copy(),
        "outputs": current_outputs.copy(),
        "errors": current_errors.copy(),
        "archive": current_archive,
        "exact_population_refreshed": True,
    }
    eligible = [candidate for candidate in materialized if candidate["eligible"]]
    best = min(
        [*eligible, incumbent],
        key=lambda item: (
            float(item["score"]),
            int(item["archive_bytes"]),
            str(item["archive_sha256"]),
            int(item["variant"]),
        ),
    )
    accepted = best["kind"] != "incumbent" and (
        float(best["score"]) < incumbent_score - 1e-15
    )
    selected = best if accepted else incumbent
    selection["selected"] = selected
    refresh = {
        "schema": "ddm_ps135_exact_population_refresh.v2",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": AXIS,
        "score_claim": False,
        "batch_geometry": f"{N // batch_size}x{batch_size}+{N % batch_size}",
        "final_partial_batch_padded": False,
        "global_pose_forward_count": exact_forward_count,
        "derivation": (
            "one official-geometry aggregate forward plus exact row substitution; "
            "PoseNet eval mode is sample-separable and every rate variant is a "
            "nested reversion to the already-global incumbent"
        ),
        "materialized_variant_count": len(materialized),
        "eligible_variant_count": len(eligible),
        "all_eligible_variants_refreshed": all(
            candidate.get("exact_population_refreshed") is True
            for candidate in eligible
        ),
        "master_provider": master_provider.binding,
        "selected_variant": selected["variant"],
        "selected_kind": selected["kind"],
        "selected_archive_bytes": selected["archive_bytes"],
        "selected_archive_sha256": selected["archive_sha256"],
        "selected_exact_d_pose": selected["d_pose"],
        "selected_exact_score": selected["score"],
        "incumbent_d_pose": incumbent_d_pose,
        "incumbent_score": incumbent_score,
        "accepted": accepted,
        "candidate_rows": exact_rows,
    }
    selection["exact_refresh"] = refresh
    return selection


def save_selected_pass(
    pass_root: Path,
    *,
    selection: dict[str, object],
    source,
    template: CarrierState,
    d_seg: float,
    pass_index: int,
    dry_streak_before: int,
    chunks: list[dict[str, object]],
    stage: str,
    candidate_neighbourhood: str,
    elapsed_seconds: float,
    previous_score: float,
    storage_preflight: dict[str, object],
    archive_builder: ArchiveBuilder = build_candidate_archive,
) -> dict[str, object]:
    selected = selection["selected"]
    codes = np.asarray(selected["codes"], dtype=np.int16)
    outputs = np.asarray(selected["outputs"], dtype=np.float32)
    errors = np.asarray(selected["errors"], dtype=np.float64)
    archive = bytes(selected["archive"])
    carrier = encode_carrier(template, codes)
    repeat, carrier_stream, parseback = archive_builder(carrier, source)
    if repeat != archive:
        raise PoseResolveError("selected archive repeat is not byte-identical")
    incumbent_selected = selected["kind"] == "incumbent"
    dry_streak = dry_streak_before + 1 if incumbent_selected else 0
    selected_score = score(d_seg, mean_pose_error(errors), len(archive))
    selected_bundle = retain_candidate_bundle(
        pass_root / "selected",
        carrier=carrier,
        carrier_stream=carrier_stream,
        archive=archive,
        codes=codes,
        parseback=parseback,
        metadata={
            "stage": stage,
            "pass": pass_index,
            "selected_variant": selected["variant"],
            "selected_kind": selected["kind"],
            "d_seg": d_seg,
            "d_pose": mean_pose_error(errors),
            "score": selected_score,
            "axis": AXIS,
            "score_claim": False,
            "hard_byte_ceiling": LC2_ARCHIVE_BYTES,
            "byte_ceiling_passes": len(archive) <= LC2_ARCHIVE_BYTES,
        },
    )
    repeat_record = persist_exact(pass_root / "selected" / "archive.repeat.zip", repeat)
    outputs_record = atomic_numpy(
        pass_root / "selected" / "pose_outputs.float32.npy", outputs
    )
    errors_record = atomic_numpy(
        pass_root / "selected" / "pair_errors.float64.npy", errors
    )
    exact_refresh = selection.get("exact_refresh")
    if not isinstance(exact_refresh, dict) or exact_refresh.get("complete") is not True:
        raise PoseResolveError("selected pass lacks a complete global exact refresh")
    atomic_json(pass_root / "selected" / "global_exact_refresh.json", exact_refresh)
    exact_refresh_record = file_record(
        pass_root / "selected" / "global_exact_refresh.json"
    )
    row = {
        "schema": "ddm_ps135_pass.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "pass": pass_index,
        "axis": AXIS,
        "score_claim": False,
        "stage": stage,
        "candidate_neighbourhood": candidate_neighbourhood,
        "candidate_evaluation_surface": "exact rendered uint8 pair -> CPU PoseNet first six",
        "pair_denominator": N,
        "search_chunks": chunks,
        "moved_rows_proposed": selection["moved_rows_proposed"],
        "rate_trim_denominator": selection["rate_trim_denominator"],
        "selected_variant": selected["variant"],
        "selected_kind": selected["kind"],
        "accepted_rows": int(
            np.count_nonzero(np.any(codes != np.load(
                pass_root / "input_coefficients.int16.npy", allow_pickle=False
            ), axis=1))
        ),
        "d_seg": d_seg,
        "d_pose": mean_pose_error(errors),
        "archive_bytes": len(archive),
        "archive_sha256": sha256_bytes(archive),
        "score": selected_score,
        "previous_score": previous_score,
        "delta_score": selected_score - previous_score,
        "elapsed_seconds": elapsed_seconds,
        "improvement_score_units_per_hour": (
            (previous_score - selected_score) * 3600.0 / max(elapsed_seconds, 1e-9)
        ),
        "storage_preflight": storage_preflight,
        "dry": incumbent_selected,
        "dry_streak": dry_streak,
        "selected_bundle": selected_bundle,
        "archive_repeat": repeat_record,
        "pose_outputs": outputs_record,
        "pair_errors": errors_record,
        "candidate_variants": selection["candidate_rows"],
        "exact_population_refresh": exact_refresh,
        "global_exact_refresh": exact_refresh_record,
        "payloads_retained": True,
    }
    atomic_json(pass_root / "receipt.json", row)
    return row


def write_leg_b_blocker(output: Path) -> dict[str, object]:
    pr135_archive = Path(
        "/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135/archive.zip"
    )
    runtime = Path(
        "/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135_src/cpr1_cbq_matched8/runtime/f26_inflate.py"
    )
    blocker = {
        "schema": "ddm_ps135_leg_b_blocker.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "leg": "B",
        "status": "BLOCKED_TYPED_CPU_PATH_AND_MASTER_CUSTODY",
        "verdict_scope": "FORMULATION_INSTANCE",
        "score_claim": False,
        "pr135_archive": file_record(pr135_archive),
        "runtime": file_record(runtime),
        "blocking_facts": [
            "shipped f26_inflate.py explicitly refuses non-CUDA execution",
            "no exact PR135 decoded raw or 600 master-frame bank is retained locally",
            "PR135 changed semantic FiLM state, so LC2/PR130 masters cannot substitute",
            "F26 pass 8 accepted zero rows; repeating identical +/-1 pose descent is duplicate",
        ],
        "tested_neighbourhood_already_closed": {
            "accepted_rows_by_pass": [412, 187, 72, 39, 15, 9, 2, 0],
            "scope": "independent per-row all-12 singleton +/-1 Pose-MSE moves",
        },
        "unblocked_by": (
            "retain exact PR135 master frames or prove CPU-equivalent full decode, then "
            "run broader receiver-realized GN/multistart/rate-aware search"
        ),
        "consumer_store": "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/",
    }
    path = output / "leg_b" / "blocker.json"
    atomic_json(path, blocker)
    return blocker


def stage_c_route_source_records() -> dict[str, object]:
    """Bind the committed LC2 SD1M route and its scorer-free receiver proof."""

    records = {
        "implementation_spec": pinned_file_record(
            STAGE_C_IMPLEMENTATION_SPEC,
            label="Stage-C implementation spec",
            digest=STAGE_C_IMPLEMENTATION_SPEC_SHA256,
        ),
        "lc2_inflate": pinned_file_record(
            LC2_RUNTIME / "inflate.py",
            label="LC2 SD1M inflate runtime",
            digest=LC2_INFLATE_SD1M_SHA256,
        ),
        "lc2_runtime_dependencies": pinned_file_record(
            LC2_RUNTIME_DEPENDENCIES,
            label="LC2 runtime dependency receipt",
            digest=LC2_RUNTIME_DEPENDENCIES_SHA256,
        ),
        "sr1_receiver_proof": pinned_file_record(
            SR1_RECEIVER_PROOF,
            label="SR1 SD1M receiver proof",
            digest=SR1_RECEIVER_PROOF_SHA256,
        ),
    }
    inflate_source = (LC2_RUNTIME / "inflate.py").read_text(encoding="utf-8")
    required_runtime_markers = (
        'SEMANTIC_MIXED_MAGIC = b"SD1M"',
        "def semantic_allocation(",
        "if semantic_blob.startswith(SEMANTIC_MIXED_MAGIC):",
    )
    if any(marker not in inflate_source for marker in required_runtime_markers):
        raise PoseResolveError("pinned LC2 runtime lacks its SD1M allocation path")
    dependencies = load_json(LC2_RUNTIME_DEPENDENCIES)
    modifications = dependencies.get("borrowed_substrate_accounting")
    if not isinstance(modifications, dict):
        raise PoseResolveError("LC2 dependency receipt lacks substrate accounting")
    cx2 = modifications.get("ddm_cx2_modified_files")
    sr1 = modifications.get("ddm_sr1_modified_files")
    if (
        not isinstance(cx2, dict)
        or "SD1M" not in str(cx2.get("inflate.py", ""))
        or not isinstance(sr1, dict)
        or "SD1M" not in str(sr1.get("inflate.py", ""))
    ):
        raise PoseResolveError("LC2 dependency receipt does not bind SD1M consumption")
    proof = load_json(SR1_RECEIVER_PROOF)
    boundaries = proof.get("boundaries")
    lineage = proof.get("reused_schema_lineage")
    if (
        proof.get("schema") != "ddm_sr1_semantic_alloc_schema_receipt.v1"
        or proof.get("complete") is not True
        or proof.get("score_claim") is not False
        or not isinstance(boundaries, dict)
        or boundaries.get("scorer_run") is not False
        or not isinstance(lineage, dict)
        or lineage.get("mechanism")
        != "counted SD1M v1 per-tensor q3/q4 allocation parser"
    ):
        raise PoseResolveError("SR1 receipt does not prove the scorer-free SD1M route")
    return records


def emit_sensitivity_and_stage_c_disposition(
    output: Path, state: dict[str, object]
) -> dict[str, object]:
    """Publish the real LC2 int12 map and route the separate SD1M consumer."""

    route_sources = stage_c_route_source_records()
    pass_index = int(state["passes_completed"])
    chunk_root = output / "leg_a" / "passes" / f"pass_{pass_index:02d}" / "search_chunks"
    chunk_paths = sorted(chunk_root.glob("chunk_*.npz"))
    if not chunk_paths:
        raise PoseResolveError("final GN pass lacks sensitivity chunks")
    jacobians = []
    updates = []
    active = []
    sources = []
    for path in chunk_paths:
        with np.load(path, allow_pickle=False) as payload:
            jacobians.append(payload["jacobian"].astype(np.float64))
            updates.append(payload["gn_update"].astype(np.float64))
            active.append(payload["active_dimensions"].astype(np.int8))
        sources.append(file_record(path))
    jacobian = np.concatenate(jacobians, axis=0)
    update = np.concatenate(updates, axis=0)
    active_dimensions = np.concatenate(active, axis=0)
    if jacobian.shape != (N, POSE_DIMS, D):
        raise PoseResolveError("consolidated int12 sensitivity map has wrong shape")
    norm = np.linalg.norm(jacobian, axis=1)
    weighted = np.abs(update) * norm
    LEGACY_STAGE_C_MAP_STORE.mkdir(parents=True, exist_ok=True)
    map_record = atomic_npz(
        LEGACY_STAGE_C_MAP_STORE / "lc2_int12_pose_sensitivity_map.npz",
        jacobian_6x12=jacobian,
        jacobian_norm=norm,
        gn_update=update,
        update_weighted_sensitivity=weighted,
        active_dimensions=active_dimensions,
        mean_norm=norm.mean(axis=0),
        max_norm=norm.max(axis=0),
        dimension_order_low_to_high=np.argsort(norm.mean(axis=0)).astype(np.int8),
    )
    stage_c_store = output / "stage_c"
    disposition = {
        "schema": "ddm_ps135_stage_c_disposition.v2",
        "complete": True,
        "written_at_utc": utc_now(),
        "score_claim": False,
        "status": "ROUTED_TO_SEPARATE_LC2_SD1M_DRIVER",
        "verdict_scope": "ROUTING_ONLY_NO_STAGE_C_SCORE",
        "route_sources": route_sources,
        "receiver_capability": {
            "vehicle": "LC2 fixed width-96 semantic renderer",
            "format": "counted SD1M v1 per-tensor allocation",
            "declared_bit_depth_range": [2, 8],
            "allocation_names": 16,
            "proof_scope": "scorer-free parse-back and raw-render receiver closure",
        },
        "measured_map": {
            "surface": "LC2 CPR1 signed-int12 coefficient -> PoseNet",
            "pair_count": N,
            "pass": pass_index,
            "payload": map_record,
            "source_chunks": sources,
        },
        "not_measured": (
            "this runner has not measured a cumulative LC2 SD1M rung composed with "
            "signed-int12 pose compensation; the retained SD1 n600 row omitted PoseNet"
        ),
        "routing_facts": [
            "the retained map differentiates 12 CPR1 coefficient codes and is a compensation input, not semantic-cell sensitivity",
            "LC2 inflate.py consumes counted SD1M v1 q3/q4 allocations through its fixed width-96 renderer",
            "the four retained SD1 rungs require cumulative receiver-realized composition and cannot be summed as marginals",
            "Stage C is sequential after Leg A and must use the shared fleet scorer lock",
        ],
        "static_w3_credit_imported": False,
        "wr1_or_869_byte_credit_imported": False,
        "disposition": "ROUTED-SEQUENTIAL-CONSUMER",
        "owner": "MAIN/#995 Stage-C separate-driver implementer",
        "consumer_store": str(stage_c_store.resolve()),
        "int12_map_compatibility_store": str(LEGACY_STAGE_C_MAP_STORE.resolve()),
        "fire_trigger": (
            "Leg A completes and scorer-free LC2 q4 identity plus all four cumulative "
            "SD1M archive parse-back/repeat proofs pass"
        ),
    }
    atomic_json(
        LEGACY_STAGE_C_MAP_STORE / "STAGE_C_DISPOSITION.json",
        disposition,
    )
    atomic_json(stage_c_store / "STAGE_C_DISPOSITION.json", disposition)
    return disposition


def exact_decode_and_evaluate(
    archive: bytes,
    *,
    bulk: Path,
    threads: int,
) -> dict[str, object]:
    """Retain the actual LC2 decode and score it through untouched evaluate.py."""

    archive_sha = sha256_bytes(archive)
    root = bulk / "exact_decode_eval" / archive_sha
    packet = root / "submission"
    inflated = packet / "inflated"
    raw_path = inflated / "0.raw"
    receipt_path = root / "receipt.json"
    authority_sources = exact_authority_source_pins()
    if receipt_path.is_file():
        return validate_exact_evaluation_receipt(
            load_json(receipt_path),
            archive_sha=archive_sha,
            raw_path=raw_path,
            authority_sources=authority_sources,
        )

    packet.mkdir(parents=True, exist_ok=True)
    inflated.mkdir(parents=True, exist_ok=True)
    persist_exact(packet / "archive.zip", archive)
    with zipfile.ZipFile(io.BytesIO(archive)) as handle:
        member = handle.read("p")
    persist_exact(packet / "p", member)
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt_index = 1 + len(list(attempts.glob("attempt_*.launch.json")))
    reusable: tuple[dict[str, object], Path] | None = None
    if raw_path.exists() and raw_path.stat().st_size == LC2_RAW_BYTES:
        reusable = reusable_decode_success(
            attempts,
            archive_sha=archive_sha,
            raw_path=raw_path,
            authority_sources=authority_sources,
        )
        if reusable is None:
            unverified = attempts / f"attempt_{attempt_index:04d}.unverified_full.raw"
            os.replace(raw_path, unverified)
            atomic_json(
                attempts / f"attempt_{attempt_index:04d}.unverified_full.json",
                {
                    "schema": "ddm_ps135_unverified_full_decode.v1",
                    "complete": True,
                    "written_at_utc": utc_now(),
                    "reason": (
                        "full byte count without an archive/runtime-bound decode-success proof"
                    ),
                    "payload": file_record(unverified),
                },
            )
    elif raw_path.exists():
        partial = attempts / f"attempt_{attempt_index:04d}.partial.raw"
        os.replace(raw_path, partial)
        atomic_json(
            attempts / f"attempt_{attempt_index:04d}.partial.json",
            {
                "schema": "ddm_ps135_partial_decode.v1",
                "complete": True,
                "written_at_utc": utc_now(),
                "reason": "prior decode did not reach the exact n600 byte count",
                "payload": file_record(partial),
            },
        )

    python = UPSTREAM / ".venv" / "bin" / "python"
    require_file(python, label="upstream pinned Python")
    token_root = root / "token_checkpoint"
    token_root.mkdir(parents=True, exist_ok=True)
    wrapper = (
        "import runpy,sys;"
        f"sys.path.insert(0,{str(LC2_RUNTIME)!r});"
        f"sys.path.append({str(PQ1_SITE_PACKAGES)!r});"
        f"sys.argv={[str(LC2_RUNTIME / 'inflate.py'), str(packet), '0', str(raw_path)]!r};"
        f"runpy.run_path({str(LC2_RUNTIME / 'inflate.py')!r},run_name='__main__')"
    )
    decode_command = [str(python), "-u", "-c", wrapper]
    decode_log = attempts / f"attempt_{attempt_index:04d}.decode.log"
    environment = os.environ.copy()
    environment.update(
        {
            "PR130_BROTLI_CLI": str(BROTLI_CLI),
            "PR130_INFLATE_DEVICE": "cpu",
            "PR130_TOKEN_CACHE": str(token_root / "tokens.npz"),
            "PR130_TOKEN_RECEIPT": str(token_root / "tokens.receipt.json"),
            "OMP_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "VECLIB_MAXIMUM_THREADS": str(threads),
            "PYTHONNOUSERSITE": "1",
        }
    )
    launch = {
        "schema": "ddm_ps135_exact_decode_launch.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "archive": file_record(packet / "archive.zip"),
        "member": file_record(packet / "p"),
        "command": decode_command,
        "decode_action": "reuse_verified_decode" if reusable else "run_decode",
        "authority_sources": authority_sources,
        "environment": {
            key: environment[key]
            for key in (
                "PR130_BROTLI_CLI",
                "PR130_INFLATE_DEVICE",
                "PR130_TOKEN_CACHE",
                "PR130_TOKEN_RECEIPT",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "PYTHONNOUSERSITE",
            )
        },
    }
    atomic_json(attempts / f"attempt_{attempt_index:04d}.launch.json", launch)
    decode_started = time.time()
    if reusable is None:
        with decode_log.open("ab", buffering=0) as log:
            completed = subprocess.run(
                decode_command,
                cwd=REPO,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            failure = {
                "schema": "ddm_ps135_exact_decode_failure.v1",
                "complete": True,
                "written_at_utc": utc_now(),
                "returncode": completed.returncode,
                "elapsed_seconds": time.time() - decode_started,
                "log": file_record(decode_log),
                "partial_raw": file_record(raw_path) if raw_path.exists() else None,
                "resumable_token_progress": [
                    file_record(path)
                    for path in sorted(token_root.glob("*"))
                    if path.is_file()
                ],
            }
            atomic_json(
                attempts / f"attempt_{attempt_index:04d}.failure.json", failure
            )
            raise PoseResolveError(
                f"exact retained decode failed with rc={completed.returncode}"
            )
        require_file(raw_path, label="exact decoded candidate raw", size=LC2_RAW_BYTES)
        raw_record = file_record(raw_path)
        token_checkpoint_records = [
            file_record(path)
            for path in sorted(token_root.glob("*"))
            if path.is_file()
        ]
        decode_success = {
            "schema": EXACT_DECODE_SUCCESS_SCHEMA,
            "complete": True,
            "written_at_utc": utc_now(),
            "archive_sha256": archive_sha,
            "archive": file_record(packet / "archive.zip"),
            "member": file_record(packet / "p"),
            "decoded_raw": raw_record,
            "decode_log": file_record(decode_log),
            "token_checkpoints": token_checkpoint_records,
            "decode_command": decode_command,
            "authority_sources": authority_sources,
            "elapsed_seconds": time.time() - decode_started,
        }
        decode_proof_path = (
            attempts / f"attempt_{attempt_index:04d}.decode_success.json"
        )
        atomic_json(decode_proof_path, decode_success)
        decode_log_record = decode_success["decode_log"]
        decode_elapsed = decode_success["elapsed_seconds"]
        decode_reused = False
    else:
        decode_success, decode_proof_path = reusable
        raw_record = decode_success["decoded_raw"]
        decode_log_record = decode_success["decode_log"]
        decode_elapsed = 0.0
        decode_reused = True
    decode_proof_record = file_record(decode_proof_path)

    report_path = root / "upstream_report.txt"
    eval_log = root / "upstream_evaluate.log"
    eval_command = [
        str(python),
        "-u",
        str(UPSTREAM / "evaluate.py"),
        "--batch-size",
        str(DEFAULT_BATCH_SIZE),
        "--num-threads",
        "2",
        "--prefetch-queue-depth",
        "4",
        "--submission-dir",
        str(packet),
        "--uncompressed-dir",
        str(UPSTREAM / "videos"),
        "--seed",
        "1234",
        "--device",
        "cpu",
        "--report",
        str(report_path),
        "--video-names-file",
        str(UPSTREAM / "public_test_video_names.txt"),
    ]
    eval_started = time.time()
    with eval_log.open("ab", buffering=0) as log:
        evaluated = subprocess.run(
            eval_command,
            cwd=UPSTREAM,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if evaluated.returncode != 0 or not report_path.is_file():
        raise PoseResolveError(
            f"untouched upstream evaluate.py failed with rc={evaluated.returncode}"
        )
    archive_bytes = int(file_record(packet / "archive.zip")["bytes"])
    parsed_report = parse_upstream_report(
        report_path.read_text(encoding="utf-8"),
        expected_archive_bytes=archive_bytes,
    )
    d_pose = float(parsed_report["d_pose"])
    d_seg = float(parsed_report["d_seg"])
    token_checkpoints = decode_success["token_checkpoints"]
    receipt = {
        "schema": EXACT_EVAL_SCHEMA,
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": AXIS,
        "score_claim": False,
        "pair_count": parsed_report["pair_count"],
        "authority_sources": authority_sources,
        "archive": file_record(packet / "archive.zip"),
        "member": file_record(packet / "p"),
        "decoded_raw": raw_record,
        "decode_log": decode_log_record,
        "decode_proof": decode_proof_record,
        "decode_reused": decode_reused,
        "decode_elapsed_seconds": decode_elapsed,
        "token_checkpoints": token_checkpoints,
        "evaluate_command": eval_command,
        "evaluate_source": file_record(UPSTREAM / "evaluate.py"),
        "evaluate_report": file_record(report_path),
        "evaluate_log": file_record(eval_log),
        "evaluate_elapsed_seconds": time.time() - eval_started,
        "d_pose_report_precision": d_pose,
        "d_seg_report_precision": d_seg,
        "archive_bytes": archive_bytes,
        "score_recomputed_from_report_components": score(
            d_seg, d_pose, archive_bytes
        ),
        "boundary": (
            "untouched upstream evaluate.py formats component values to 8 decimals; "
            "the search cache retains higher-precision advisory components separately"
        ),
        "payloads_retained": True,
    }
    atomic_json(receipt_path, receipt)
    return validate_exact_evaluation_receipt(
        receipt,
        archive_sha=archive_sha,
        raw_path=raw_path,
        authority_sources=authority_sources,
    )


def run_solver(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.resolve()
    bulk = args.bulk.resolve()
    pre = preflight(output, bulk, write=True)
    claims = active_dispatch_claims()
    active = claims.get("active", [])
    allowed = [
        row
        for row in active
        if row.get("lane_id") == "lane_ddm_ps135_pose_resolve_20260810"
        and row.get("instance_job_id") == "ddm_ps135_lc2_joint_pose_n600"
    ]
    if len(active) != 1 or len(allowed) != 1:
        raise PoseResolveError(
            "run requires exactly the live ps135 lane claim and no other live claim"
        )
    if not pre["storage_passes"] or not pre["process_scan"]["passes"]:
        raise PoseResolveError("preflight storage/process gate did not pass")

    fleet_lock = acquire_lock(
        FLEET_SCORER_LOCK,
        purpose="fleetwide full-n600 scorer",
    )
    try:
        lock = acquire_lock(output / ".single_writer.lock", purpose="ps135 writer")
    except BaseException:
        fleet_lock.close()
        raise
    try:
        torch, posenet, segnet = load_scorers(args.threads)
        source = load_lc2_source()
        base_carrier = source.carrier
        if sha256_bytes(base_carrier) != LC2_CARRIER_SHA256:
            raise PoseResolveError("LC2 source carrier differs from its pin")
        base_archive, _, _ = build_candidate_archive(base_carrier, source)
        if base_archive != LC2_ARCHIVE.read_bytes():
            raise PoseResolveError("LC2 candidate builder does not reproduce the base archive")

        target_manifest_path = bulk / TARGET_CACHE_DIRNAME / "manifest.json"
        if not target_manifest_path.is_file():
            build_target_cache(
                bulk,
                posenet=posenet,
                segnet=segnet,
                batch_size=args.batch_size,
            )
        pose_targets, seg_targets, target_manifest = load_target_cache(bulk)
        master_provider = lc2_master_provider()

        state_path = output / "leg_a" / "state.json"
        if state_path.is_file():
            state = load_json(state_path)
            if not args.resume:
                raise PoseResolveError("existing solve state requires --resume")
        else:
            own_state, own = evaluate_start(
                output,
                bulk,
                name="lc2_native",
                carrier=base_carrier,
                source=source,
                posenet=posenet,
                segnet=segnet,
                master_provider=master_provider,
                pose_targets=pose_targets,
                seg_targets=seg_targets,
                batch_size=args.batch_size,
                expected_raw_parity=True,
            )
            # Imported PR133 is a complete carrier (basis, scales, and codes),
            # never the invalid operation of copying integer codes alone.
            pr133_carrier = extract_pr133_carrier()
            pr133_state = decode_carrier(pr133_carrier)
            _, _, inflate = import_runtime_modules()
            pr133_renderer = ExactCarrierRenderer(pr133_state, inflate, torch)
            pr133_archive, pr133_stream, pr133_parseback = build_candidate_archive(
                pr133_carrier, source
            )
            pr133_outputs = pose_outputs(
                pr133_renderer,
                posenet,
                master_provider,
                pr133_state.codes,
                np.arange(N),
                args.batch_size,
                pad_partial=False,
            )
            pr133_errors = pose_pair_errors(pr133_outputs, pose_targets)
            pr133_dpose = mean_pose_error(pr133_errors)
            pr133_dseg = float(own["d_seg"])
            pr133_bundle = retain_candidate_bundle(
                output / "leg_a" / "starts" / "public_pr133_complete_carrier",
                carrier=pr133_carrier,
                carrier_stream=pr133_stream,
                archive=pr133_archive,
                codes=pr133_state.codes,
                parseback=pr133_parseback,
                metadata={
                    "stage": "start",
                    "start_name": "public_pr133_complete_carrier",
                    "axis": AXIS,
                    "score_claim": False,
                    "d_pose": pr133_dpose,
                    "d_seg": pr133_dseg,
                    "score": score(pr133_dseg, pr133_dpose, len(pr133_archive)),
                    "public_warm_start": True,
                    "invalid_codes_only_copy": False,
                },
            )
            pr133_outputs_record = atomic_numpy(
                output
                / "leg_a"
                / "starts"
                / "public_pr133_complete_carrier"
                / "pose_outputs.float32.npy",
                pr133_outputs,
            )
            pr133_errors_record = atomic_numpy(
                output
                / "leg_a"
                / "starts"
                / "public_pr133_complete_carrier"
                / "pair_errors.float64.npy",
                pr133_errors,
            )
            starts = [
                {
                    "name": "lc2_native",
                    "state": own_state,
                    "receipt": own,
                    "archive": base_archive,
                    "outputs": np.load(own["scorer"]["pose_outputs"]["path"]),
                    "errors": np.load(own["scorer"]["pair_errors"]["path"]),
                    "d_seg": float(own["d_seg"]),
                    "score": float(own["score"]),
                },
                {
                    "name": "public_pr133_complete_carrier",
                    "state": pr133_state,
                    "receipt": pr133_bundle,
                    "archive": pr133_archive,
                    "outputs": pr133_outputs,
                    "errors": pr133_errors,
                    "d_seg": pr133_dseg,
                    "score": score(pr133_dseg, pr133_dpose, len(pr133_archive)),
                    "pose_outputs": pr133_outputs_record,
                    "pair_errors": pr133_errors_record,
                },
            ]
            eligible_starts = [
                item
                for item in starts
                if len(item["archive"]) <= LC2_ARCHIVE_BYTES
            ]
            selected_start = min(
                eligible_starts,
                key=lambda item: (item["score"], len(item["archive"]), item["name"]),
            )
            start_race = {
                "schema": "ddm_ps135_start_race.v1",
                "complete": True,
                "written_at_utc": utc_now(),
                "axis": AXIS,
                "score_claim": False,
                "rows": [
                    {
                        "name": item["name"],
                        "d_pose": mean_pose_error(item["errors"]),
                        "d_seg": item["d_seg"],
                        "archive_bytes": len(item["archive"]),
                        "archive_sha256": sha256_bytes(item["archive"]),
                        "score": item["score"],
                        "eligible": len(item["archive"]) <= LC2_ARCHIVE_BYTES,
                    }
                    for item in starts
                ],
                "lowest_score_control": selected_start["name"],
                "required_native_leg_start": "lc2_native",
                "reason": (
                    "the charter requires an independent mechanism-transfer convergence "
                    "history on LC2's own coefficients; the public PR133 carrier remains "
                    "an attributed composition control and cannot replace that leg"
                ),
            }
            atomic_json(output / "leg_a" / "starts" / "race.json", start_race)
            selected_start = starts[0]
            selected_pose_record = (
                selected_start["receipt"]
                .get("scorer", {})
                .get("pose_outputs", selected_start.get("pose_outputs"))
            )
            selected_error_record = (
                selected_start["receipt"]
                .get("scorer", {})
                .get("pair_errors", selected_start.get("pair_errors"))
            )
            if not isinstance(selected_pose_record, dict) or not isinstance(
                selected_error_record, dict
            ):
                raise PoseResolveError("selected start lacks bound scorer payloads")
            state = {
                "schema": "ddm_ps135_state.v2",
                "complete": False,
                "started_at_utc": utc_now(),
                "axis": AXIS,
                "score_claim": False,
                "config": {
                    "batch_size": args.batch_size,
                    "threads": args.threads,
                    "min_passes": args.min_passes,
                    "dry_passes": args.dry_passes,
                    "damping": GN_DAMPING,
                    "max_code_step": MAX_CODE_STEP,
                    "jrd_lineage_steps": list(JRD_LINEAGE_STEPS),
                    "neighbour_dimensions": NEIGHBOUR_DIMS,
                    "neighbour_radius": NEIGHBOUR_RADIUS,
                },
                "inputs": input_pins(),
                "target_manifest": file_record(target_manifest_path),
                "selected_start": selected_start["name"],
                "start_race": file_record(output / "leg_a" / "starts" / "race.json"),
                "operational_max_passes_seen": args.max_passes,
                "d_seg": selected_start["d_seg"],
                "passes_completed": 0,
                "dry_streak": 0,
                "history": [],
                "current": current_artifact_bindings(
                    archive=Path(
                        selected_start["receipt"]["records"]["archive"]["path"]
                    ),
                    coefficients=Path(
                        selected_start["receipt"]["records"]["coefficients"]["path"]
                    ),
                    pose_outputs_path=Path(selected_pose_record["path"]),
                    pair_errors=Path(selected_error_record["path"]),
                ),
            }
            atomic_json(state_path, state)

        if state.get("schema") != "ddm_ps135_state.v2":
            raise PoseResolveError(
                "resume state predates immutable current-artifact bindings"
            )
        if state.get("config") != {
            "batch_size": args.batch_size,
            "threads": args.threads,
            "min_passes": args.min_passes,
            "dry_passes": args.dry_passes,
            "damping": GN_DAMPING,
            "max_code_step": MAX_CODE_STEP,
            "jrd_lineage_steps": list(JRD_LINEAGE_STEPS),
            "neighbour_dimensions": NEIGHBOUR_DIMS,
            "neighbour_radius": NEIGHBOUR_RADIUS,
        }:
            raise PoseResolveError("resume configuration differs")
        if state.get("target_manifest") != file_record(target_manifest_path):
            raise PoseResolveError("resume target-cache authority binding differs")
        state["operational_max_passes_seen"] = max(
            int(state.get("operational_max_passes_seen", 0)), args.max_passes
        )

        (
            current_archive,
            current_codes,
            current_outputs,
            current_errors,
            current_paths,
        ) = load_current_artifacts(state["current"])
        current_carrier = parse_candidate_archive(
            current_archive,
            encode_carrier(decode_carrier(
                current_paths["archive"].parent.joinpath("carrier.cpr1").read_bytes()
            ), current_codes),
            source,
        )
        del current_carrier
        template_carrier = (
            extract_pr133_carrier()
            if state["selected_start"] == "public_pr133_complete_carrier"
            else source.carrier
        )
        template = decode_carrier(template_carrier)
        # After pass 1, the template codes naturally differ; only basis/scales
        # define the encoder/render template.
        if not state["history"] and not np.array_equal(
            template.codes, current_codes
        ):
            raise PoseResolveError("selected start coefficients differ from state")

        while (
            int(state["passes_completed"]) < args.min_passes
            or int(state["dry_streak"]) < args.dry_passes
        ):
            if int(state["passes_completed"]) >= args.max_passes:
                state["continuation_required"] = {
                    "reason": "operational pass tranche reached before convergence",
                    "passes_completed": state["passes_completed"],
                    "dry_streak": state["dry_streak"],
                    "resume_with_max_passes_greater_than": args.max_passes,
                }
                atomic_json(state_path, state)
                raise PoseResolveError(
                    "operational pass tranche reached; resume with a larger --max-passes"
                )
            pass_index = int(state["passes_completed"]) + 1
            pass_started = time.time()
            previous_score = score(
                float(state["d_seg"]),
                mean_pose_error(current_errors),
                len(current_archive),
            )
            pass_root = output / "leg_a" / "passes" / f"pass_{pass_index:02d}"
            pass_storage_preflight = require_vertigo_free_space(
                pass_root,
                required_free_bytes=PASS_REQUIRED_FREE_BYTES,
                stage=f"joint_pose_pass:{pass_index:02d}",
            )
            pass_root.mkdir(parents=True, exist_ok=True)
            atomic_json(
                pass_root / "storage_preflight.json", pass_storage_preflight
            )
            atomic_numpy(pass_root / "input_coefficients.int16.npy", current_codes)
            atomic_numpy(pass_root / "input_pose_outputs.float32.npy", current_outputs)
            atomic_numpy(pass_root / "input_pair_errors.float64.npy", current_errors)
            persist_exact(pass_root / "input_archive.zip", current_archive)
            proposed_codes, proposed_outputs, proposed_errors, chunks = solve_pass_chunks(
                pass_root,
                template=template,
                current_codes=current_codes,
                current_outputs=current_outputs,
                pose_targets=pose_targets,
                posenet=posenet,
                master_provider=master_provider,
                batch_size=args.batch_size,
            )
            selection = rate_aware_select(
                pass_root,
                source=source,
                template=template,
                current_codes=current_codes,
                current_outputs=current_outputs,
                current_errors=current_errors,
                current_archive=current_archive,
                proposed_codes=proposed_codes,
                proposed_outputs=proposed_outputs,
                proposed_errors=proposed_errors,
                d_seg=float(state["d_seg"]),
            )
            selection = exact_population_refresh(
                selection,
                template=template,
                current_codes=current_codes,
                current_outputs=current_outputs,
                current_errors=current_errors,
                current_archive=current_archive,
                d_seg=float(state["d_seg"]),
                pose_targets=pose_targets,
                posenet=posenet,
                master_provider=master_provider,
                batch_size=args.batch_size,
            )
            row = save_selected_pass(
                pass_root,
                selection=selection,
                source=source,
                template=template,
                d_seg=float(state["d_seg"]),
                pass_index=pass_index,
                dry_streak_before=int(state["dry_streak"]),
                chunks=chunks,
                stage="joint_pose_pass",
                candidate_neighbourhood=(
                    "all singleton +/-1 plus public damped-GN center rank-3 "
                    "radius-1 cube"
                ),
                elapsed_seconds=time.time() - pass_started,
                previous_score=previous_score,
                storage_preflight=pass_storage_preflight,
            )
            selected_root = pass_root / "selected"
            current_codes = np.load(
                selected_root / "coefficients.int16.npy", allow_pickle=False
            ).astype(np.int16)
            current_outputs = np.load(
                selected_root / "pose_outputs.float32.npy", allow_pickle=False
            ).astype(np.float32)
            current_errors = np.load(
                selected_root / "pair_errors.float64.npy", allow_pickle=False
            ).astype(np.float64)
            current_archive = (selected_root / "archive.zip").read_bytes()
            state["passes_completed"] = pass_index
            state["dry_streak"] = row["dry_streak"]
            state["history"].append(
                {
                    key: row[key]
                    for key in (
                        "pass",
                        "stage",
                        "accepted_rows",
                        "d_seg",
                        "d_pose",
                        "archive_bytes",
                        "archive_sha256",
                        "score",
                        "dry",
                        "dry_streak",
                        "elapsed_seconds",
                        "improvement_score_units_per_hour",
                    )
                }
            )
            state["current"] = current_artifact_bindings(
                archive=selected_root / "archive.zip",
                coefficients=selected_root / "coefficients.int16.npy",
                pose_outputs_path=selected_root / "pose_outputs.float32.npy",
                pair_errors=selected_root / "pair_errors.float64.npy",
            )
            atomic_json(state_path, state)
            print(json.dumps(row, sort_keys=True), flush=True)

        finisher_round = len(state.get("jrd_history", [])) + 1
        finisher_root = (
            output / "leg_a" / "jrd_finisher" / f"round_{finisher_round:02d}"
        )
        finisher_storage_preflight = require_vertigo_free_space(
            finisher_root,
            required_free_bytes=FINISHER_REQUIRED_FREE_BYTES,
            stage=f"jrd_finisher:{finisher_round:02d}",
        )
        atomic_json(
            finisher_root / "storage_preflight.json", finisher_storage_preflight
        )
        finisher_started = time.time()
        previous_score = score(
            float(state["d_seg"]),
            mean_pose_error(current_errors),
            len(current_archive),
        )
        atomic_numpy(finisher_root / "input_coefficients.int16.npy", current_codes)
        atomic_numpy(finisher_root / "input_pose_outputs.float32.npy", current_outputs)
        atomic_numpy(finisher_root / "input_pair_errors.float64.npy", current_errors)
        persist_exact(finisher_root / "input_archive.zip", current_archive)
        proposed_codes, proposed_outputs, proposed_errors, finisher_chunks = (
            solve_jrd_finisher_chunks(
                finisher_root,
                template=template,
                current_codes=current_codes,
                current_outputs=current_outputs,
                pose_targets=pose_targets,
                posenet=posenet,
                master_provider=master_provider,
                batch_size=args.batch_size,
            )
        )
        finisher_selection = rate_aware_select(
            finisher_root,
            source=source,
            template=template,
            current_codes=current_codes,
            current_outputs=current_outputs,
            current_errors=current_errors,
            current_archive=current_archive,
            proposed_codes=proposed_codes,
            proposed_outputs=proposed_outputs,
            proposed_errors=proposed_errors,
            d_seg=float(state["d_seg"]),
        )
        finisher_selection = exact_population_refresh(
            finisher_selection,
            template=template,
            current_codes=current_codes,
            current_outputs=current_outputs,
            current_errors=current_errors,
            current_archive=current_archive,
            d_seg=float(state["d_seg"]),
            pose_targets=pose_targets,
            posenet=posenet,
            master_provider=master_provider,
            batch_size=args.batch_size,
        )
        finisher = save_selected_pass(
            finisher_root,
            selection=finisher_selection,
            source=source,
            template=template,
            d_seg=float(state["d_seg"]),
            pass_index=int(state["passes_completed"]),
            dry_streak_before=int(state["dry_streak"]),
            chunks=finisher_chunks,
            stage="jrd_terminal_finisher",
            candidate_neighbourhood=(
                "exact per-dimension +/-{1,2,4,8,16,32} JRD lineage ladder"
            ),
            elapsed_seconds=time.time() - finisher_started,
            previous_score=previous_score,
            storage_preflight=finisher_storage_preflight,
        )
        finisher["verdict"] = "SUBSUMED" if finisher["dry"] else "FIRED"
        jrd_prior_record = file_record(
            finisher_root / "jrd_reusable_prior_receipt.json"
        )
        verify_file_record_binding(
            jrd_prior_record,
            label="JRD reusable-prior receipt",
        )
        finisher["jrd_reusable_prior_receipt"] = jrd_prior_record
        atomic_json(finisher_root / "receipt.json", finisher)
        state.setdefault("jrd_history", []).append(
            {
                "round": finisher_round,
                "verdict": finisher["verdict"],
                "accepted_rows": finisher["accepted_rows"],
                "d_pose": finisher["d_pose"],
                "archive_bytes": finisher["archive_bytes"],
                "score": finisher["score"],
                "elapsed_seconds": finisher["elapsed_seconds"],
                "jrd_reusable_prior_receipt": finisher[
                    "jrd_reusable_prior_receipt"
                ],
                "receipt": file_record(finisher_root / "receipt.json"),
            }
        )
        if not finisher["dry"]:
            selected_root = finisher_root / "selected"
            state["dry_streak"] = 0
            state["current"] = current_artifact_bindings(
                archive=selected_root / "archive.zip",
                coefficients=selected_root / "coefficients.int16.npy",
                pose_outputs_path=selected_root / "pose_outputs.float32.npy",
                pair_errors=selected_root / "pair_errors.float64.npy",
            )
            state["continuation_required"] = {
                "reason": "terminal JRD ladder fired after GN convergence",
                "jrd_round": finisher_round,
                "accepted_rows": finisher["accepted_rows"],
                "fire_trigger": "resume GN passes to three dry, then rerun JRD finisher",
            }
            atomic_json(state_path, state)
            raise PoseResolveError(
                "JRD terminal finisher fired; checkpoint promoted, resume GN convergence"
            )
        state.pop("continuation_required", None)
        atomic_json(state_path, state)

        final_root = output / "leg_a" / "final"
        final_root.mkdir(parents=True, exist_ok=True)
        final_archive = current_archive
        final_carrier = encode_carrier(template, current_codes)
        repeat, final_stream, final_parseback = build_candidate_archive(
            final_carrier, source
        )
        if repeat != final_archive or len(final_archive) > LC2_ARCHIVE_BYTES:
            raise PoseResolveError("final archive repeat/byte ceiling failed")
        final_bundle = retain_candidate_bundle(
            final_root,
            carrier=final_carrier,
            carrier_stream=final_stream,
            archive=final_archive,
            codes=current_codes,
            parseback=final_parseback,
            metadata={
                "stage": "final",
                "axis": AXIS,
                "score_claim": False,
                "d_seg": float(state["d_seg"]),
                "d_pose": mean_pose_error(current_errors),
                "score": score(
                    float(state["d_seg"]),
                    mean_pose_error(current_errors),
                    len(final_archive),
                ),
                "hard_byte_ceiling": LC2_ARCHIVE_BYTES,
                "byte_ceiling_passes": True,
            },
        )
        final_outputs = atomic_numpy(
            final_root / "pose_outputs.float32.npy", current_outputs
        )
        final_errors = atomic_numpy(
            final_root / "pair_errors.float64.npy", current_errors
        )
        final_repeat = persist_exact(final_root / "archive.repeat.zip", repeat)
        stage_c = emit_sensitivity_and_stage_c_disposition(output, state)
        exact_eval = exact_decode_and_evaluate(
            final_archive,
            bulk=bulk,
            threads=args.threads,
        )
        blocker = write_leg_b_blocker(output)
        result = {
            "schema": "ddm_ps135_result.v1",
            "complete": True,
            "written_at_utc": utc_now(),
            "axis": AXIS,
            "score_claim": False,
            "pair_count": N,
            "selected_start": state["selected_start"],
            "passes_completed": state["passes_completed"],
            "dry_streak": state["dry_streak"],
            "history": state["history"],
            "final": {
                "d_seg": exact_eval["d_seg_report_precision"],
                "d_pose": exact_eval["d_pose_report_precision"],
                "archive_bytes": len(final_archive),
                "archive_sha256": sha256_bytes(final_archive),
                "score": score(
                    exact_eval["d_seg_report_precision"],
                    exact_eval["d_pose_report_precision"],
                    len(final_archive),
                ),
                "search_d_seg": float(state["d_seg"]),
                "search_d_pose": mean_pose_error(current_errors),
                "search_score": score(
                    float(state["d_seg"]),
                    mean_pose_error(current_errors),
                    len(final_archive),
                ),
                "bundle": final_bundle,
                "pose_outputs": final_outputs,
                "pair_errors": final_errors,
                "archive_repeat": final_repeat,
                "exact_decode_eval": exact_eval,
            },
            "jrd_finisher": {
                "dry": finisher["dry"],
                "accepted_rows": finisher["accepted_rows"],
                "verdict": finisher["verdict"],
                "receipt": file_record(finisher_root / "receipt.json"),
            },
            "leg_b": blocker,
            "stage_c": stage_c,
            "target_manifest": target_manifest,
            "payloads_retained": True,
            "modal_dispatched": False,
            "next_authority": "MAIN fires one exact Modal contest-CUDA row",
        }
        atomic_json(output / "RESULT.json", result)
        state["complete"] = True
        state["finished_at_utc"] = utc_now()
        state["result"] = file_record(output / "RESULT.json")
        atomic_json(state_path, state)
        return result
    finally:
        try:
            lock.close()
        finally:
            fleet_lock.close()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight", help="scorer-free custody/storage/liveness receipt")
    pre.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    pre.add_argument("--bulk", type=Path, default=DEFAULT_BULK)
    run = sub.add_parser("run", help="full-n600 retained LC2 solve")
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--bulk", type=Path, default=DEFAULT_BULK)
    run.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    run.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    run.add_argument("--min-passes", type=int, default=MIN_PASSES)
    run.add_argument("--dry-passes", type=int, default=DRY_PASSES)
    run.add_argument("--max-passes", type=int, default=DEFAULT_MAX_PASSES)
    run.add_argument("--resume", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "preflight":
        result = preflight(args.output.resolve(), args.bulk.resolve(), write=True)
    else:
        if (
            args.batch_size != DEFAULT_BATCH_SIZE
            or args.threads <= 0
            or args.min_passes < MIN_PASSES
            or args.dry_passes < DRY_PASSES
            or args.max_passes < args.min_passes
        ):
            raise PoseResolveError("run configuration violates the sealed protocol")
        result = run_solver(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
