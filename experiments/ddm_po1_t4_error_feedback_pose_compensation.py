#!/usr/bin/env python3
"""Receiver-closed PO1 damped PoseNet coefficient compensation.

The solve consumes retained T4 GT/decoded/repeat first-six vectors and the
exact decoded CP135 raw stream.  A local CPU PoseNet Jacobian supplies only a
preconditioner direction; the residual and prediction origin are the retained
T4 vectors.  One damped signed-int12 step is taken per identity-selector pair,
with no local-score admission.  All Jacobians, code states, predictions,
archives, runtimes, and checkpoints are retained on the SSD tier.

The ``adjudicate`` subcommand compares a later T4 round against the first one,
enforces the SegNet/F1/F2/F3 gates, and computes the same-object Pose/rate
score delta.  It does not run an evaluator or mutate the canonical pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np

try:
    from experiments.ddm_cp135_rate_compose import (
        deterministic_zip,
        pack_cap1_metadata,
        pack_split_models,
        unpack_cap1_metadata,
        unpack_split_models,
    )
except ModuleNotFoundError:
    from ddm_cp135_rate_compose import (  # type: ignore[no-redef]
        deterministic_zip,
        pack_cap1_metadata,
        pack_split_models,
        unpack_cap1_metadata,
        unpack_split_models,
    )

REPO: Final = Path(__file__).resolve().parents[1]
UPSTREAM: Final = REPO / "upstream"
DEFAULT_ARCHIVE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip"
)
DEFAULT_RUNTIME: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime"
)
DEFAULT_ROUND1: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_po1_20260813/round1_cp135"
)
DEFAULT_OUTPUT: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_po1_20260813/solve/attempt1"
)
BROTLI: Final = Path("/opt/homebrew/bin/brotli")
EXPERIMENT_BOOK_SRC: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book/src"
)
CP135_BYTES: Final = 186_252
CP135_SHA256: Final = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
N: Final = 600
D: Final = 12
POSE_DIMS: Final = 6
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
SEG_H: Final = 384
SEG_W: Final = 512
RATE_DENOMINATOR: Final = 37_545_489
CAP1_PREFIX: Final = b"CAP1\x01\x00\x00\x00"
SPARSE_SELECTOR_PREFIX: Final = b"F0E1\x01"
CAP_FIELDS: Final = ("predictor", "scales", "lengths", "ks", "basis", "rice")
STORED_CAP_FIELDS: Final = ("scales", "predictor", "lengths", "ks", "basis", "rice")
STORAGE_RESERVE_BYTES: Final = 4 * 1024**3


class PO1SolveError(RuntimeError):
    """A T4 binding, local-J, lattice, archive, or resume invariant failed."""


@dataclass(frozen=True)
class CarrierState:
    canonical: bytes
    basis_scales: np.ndarray
    basis_codes: np.ndarray
    coefficient_scales: np.ndarray
    codes: np.ndarray
    selector: bytes
    selector_choices: np.ndarray


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".partial.{os.getpid()}")
    with staging.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical_json_bytes(value))


def checkpoint_once(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value)
    if path.is_file():
        if path.read_bytes() != payload:
            raise PO1SolveError(f"resume checkpoint differs: {path}")
        return
    atomic_bytes(path, payload)


def atomic_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    atomic_bytes(path, stream.getvalue())
    return file_record(path)


def require_file(path: Path, *, size: int | None = None, digest: str | None = None) -> None:
    if not path.is_file():
        raise PO1SolveError(f"missing required file: {path}")
    if size is not None and path.stat().st_size != size:
        raise PO1SolveError(f"file has unexpected byte count: {path}")
    if digest is not None and sha256_file(path) != digest:
        raise PO1SolveError(f"file has unexpected SHA-256: {path}")


def storage_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    # Raw is consumed in place.  Jacobians, state, candidate/runtime copies,
    # and receipts are small; a 4 GiB reserve is deliberately conservative.
    result = {
        "schema": "ddm_po1_local_storage_preflight.v1",
        "tier": str(output),
        "free_bytes": usage.free,
        "reserve_bytes": STORAGE_RESERVE_BYTES,
        "passed": usage.free >= STORAGE_RESERVE_BYTES,
        "cleanup_policy": "block rather than delete; every candidate and solve payload retained",
    }
    atomic_json(output / "STORAGE_PREFLIGHT.json", result)
    if not result["passed"]:
        raise PO1SolveError("local SSD storage preflight failed")
    return result


def _runtime_modules(runtime_root: Path) -> tuple[ModuleType, ModuleType, ModuleType, ModuleType]:
    os.environ["CP135_BROTLI_CLI"] = str(BROTLI)
    value = str(runtime_root.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)
    residual = importlib.import_module("runtime.residual_archive")
    repack = importlib.import_module("runtime.carrier_repack")
    coefficient = importlib.import_module("runtime.entropy.coefficient_ar1_codec")
    selector = importlib.import_module("runtime.frame0_selector")
    return residual, repack, coefficient, selector


def _load_renderer(runtime_root: Path) -> ModuleType:
    root = (runtime_root / "cpr1").resolve()
    path = root / "inflate.py"
    name = "_ddm_po1_cp135_renderer"
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise PO1SolveError("a different PO1 renderer is already loaded")
        return existing
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise PO1SolveError(f"cannot load CP135 renderer: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _carrier_codec() -> ModuleType:
    return importlib.import_module("tac.pr130_runtime.fx1_runtime_tree.carrier_codec")


def _cap1_encoder() -> ModuleType:
    value = str(EXPERIMENT_BOOK_SRC)
    if value not in sys.path:
        sys.path.insert(0, value)
    return importlib.import_module("cpr1_sub4.entropy.coefficient_ar1_codec")


def signed_codes_from_delta_zigzag(encoded: np.ndarray) -> np.ndarray:
    value = np.asarray(encoded, dtype=np.int64)
    if value.shape != (N, D) or np.any(value < 0) or np.any(value > 4095):
        raise PO1SolveError("encoded coefficient lattice has invalid shape/range")
    delta = (value >> 1) ^ -(value & 1)
    unsigned = np.cumsum(delta, axis=0, dtype=np.int64) & 0xFFF
    return np.where(unsigned >= 0x800, unsigned - 0x1000, unsigned).astype(np.int16)


def delta_zigzag_from_signed_codes(codes: np.ndarray) -> np.ndarray:
    value = np.asarray(codes, dtype=np.int64)
    if value.shape != (N, D) or np.any(value < -2048) or np.any(value > 2047):
        raise PO1SolveError("signed coefficient lattice has invalid shape/range")
    unsigned = value & 0xFFF
    previous = np.zeros_like(unsigned)
    previous[1:] = unsigned[:-1]
    delta_unsigned = (unsigned - previous) & 0xFFF
    delta = np.where(delta_unsigned >= 0x800, delta_unsigned - 0x1000, delta_unsigned)
    return (((delta << 1) ^ (delta >> 63)) & 0xFFF).astype(np.int32)


def load_carrier(archive: Path, runtime_root: Path) -> tuple[Any, CarrierState]:
    residual, repack, _coefficient, selector_module = _runtime_modules(runtime_root)
    parts = residual.read_residual_archive(archive)
    cap1, selector = repack.split_frame0_selector_carrier(parts.carrier_blob)
    if selector is None:
        raise PO1SolveError("CP135 must expose the F0E1 selector")
    canonical = repack.materialize_cpr1(cap1, _load_renderer(runtime_root))
    codec = _carrier_codec()
    basis_count = D * 3 * 24 * 32
    basis_scales, basis_codes, coefficient_scales, encoded = codec.decode_compact_carrier(
        canonical,
        basis_count=basis_count,
        frames=N,
        dimensions=D,
    )
    codes = signed_codes_from_delta_zigzag(encoded)
    rebuilt = codec.encode_compact_carrier(
        basis_scales,
        basis_codes,
        coefficient_scales,
        delta_zigzag_from_signed_codes(codes),
    )
    if rebuilt != canonical:
        raise PO1SolveError("CP135 CPR1 decode/re-encode is not byte-identical")
    _modes, choices = selector_module.decode_selector(selector)
    return parts, CarrierState(
        canonical=canonical,
        basis_scales=np.asarray(basis_scales, dtype=np.float32),
        basis_codes=np.asarray(basis_codes, dtype=np.int8),
        coefficient_scales=np.asarray(coefficient_scales, dtype=np.float32),
        codes=codes,
        selector=selector,
        selector_choices=np.asarray(choices, dtype=np.uint8),
    )


def encode_canonical_carrier(state: CarrierState, codes: np.ndarray) -> bytes:
    return _carrier_codec().encode_compact_carrier(
        state.basis_scales,
        state.basis_codes,
        state.coefficient_scales,
        delta_zigzag_from_signed_codes(codes),
    )


def solve_damped_least_squares(
    jacobian: np.ndarray,
    residual: np.ndarray,
    *,
    damping: float,
    max_code_step: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(jacobian, dtype=np.float64)
    error = np.asarray(residual, dtype=np.float64)
    if matrix.shape != (POSE_DIMS, D) or error.shape != (POSE_DIMS,):
        raise PO1SolveError("damped solve expects one 6x12 Jacobian and 6-vector residual")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(error)):
        raise PO1SolveError("damped solve inputs must be finite")
    if damping < 0.0 or max_code_step <= 0.0:
        raise PO1SolveError("damping/trust radius is invalid")
    singular = np.linalg.svd(matrix, compute_uv=False)
    largest = float(singular[0]) if singular.size else 0.0
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * largest
    rank = int(np.count_nonzero(singular > tolerance))
    smallest = float(singular[rank - 1]) if rank else 0.0
    condition = float("inf") if not rank or smallest == 0.0 else largest / smallest
    ridge = float((damping * largest) ** 2)
    if ridge:
        normal = matrix.T @ matrix + ridge * np.eye(D)
        update = np.linalg.solve(normal, matrix.T @ error)
    else:
        update = np.linalg.lstsq(matrix, error, rcond=None)[0]
    update = np.clip(update, -max_code_step, max_code_step)
    return update, {"rank": rank, "condition": condition, "ridge_lambda": ridge}


def quantize_int12_update(current: np.ndarray, update: np.ndarray) -> np.ndarray:
    codes = np.asarray(current, dtype=np.int64)
    step = np.asarray(update, dtype=np.float64)
    if codes.shape != (D,) or step.shape != (D,) or not np.all(np.isfinite(step)):
        raise PO1SolveError("int12 update has invalid shape or values")
    return np.clip(codes + np.rint(step).astype(np.int64), -2048, 2047).astype(np.int16)


def _round_ste(value: Any) -> Any:
    return value + (value.round() - value).detach()


class LocalJacobian:
    def __init__(self, runtime_root: Path, state: CarrierState) -> None:
        import torch

        renderer = _load_renderer(runtime_root)
        raw_basis = torch.from_numpy(
            state.basis_codes.reshape(D, 3, 24, 32).astype(np.float32)
            * state.basis_scales[:, None, None, None]
        )
        self.renderer = renderer
        self.basis = renderer.normalized_basis(raw_basis)
        self.scales = torch.from_numpy(state.coefficient_scales.astype(np.float32))
        self.device = torch.device("cpu")
        from safetensors.torch import load_file

        sys.path.insert(0, str(UPSTREAM))
        try:
            from modules import PoseNet, posenet_sd_path
        finally:
            sys.path.pop(0)
        self.posenet = PoseNet().eval().to(self.device)
        self.posenet.load_state_dict(load_file(posenet_sd_path, device="cpu"))
        for parameter in self.posenet.parameters():
            parameter.requires_grad_(False)
        # Upstream rgb_to_yuv6 is @torch.no_grad() (frame_utils.py:50) and severs the
        # codes->pose autograd graph inside preprocess_input. Patch both module
        # references with the canonical differentiable twin (the PR95 data.py:80-81
        # lesson). Forward values are unchanged; only gradient flow is restored.
        from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally

        patch_upstream_yuv6_globally()

    def pose_and_jacobian(
        self,
        current: np.ndarray,
        master: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch
        import torch.nn.functional as functional

        codes = torch.as_tensor(
            np.asarray(current, dtype=np.float32), device=self.device
        ).clone().requires_grad_(True)
        coefficient = codes[None] * self.scales[None]
        carrier = torch.einsum("bk,kchw->bchw", coefficient, self.basis) / math.sqrt(D)
        low = _round_ste(
            (127.5 + self.renderer.CARRIER_AMPLITUDE * carrier).clamp(0.0, 255.0)
        )
        high = functional.interpolate(
            low,
            size=(CAMERA_H, CAMERA_W),
            mode="bicubic",
            align_corners=False,
        )
        slave = _round_ste(high.clamp(0.0, 255.0))[0]
        master_tensor = torch.from_numpy(np.ascontiguousarray(master)).permute(2, 0, 1).float()
        pair = torch.stack((slave, master_tensor), dim=0).unsqueeze(0)
        output = self.posenet(self.posenet.preprocess_input(pair))["pose"][0, :POSE_DIMS]
        rows = []
        for dimension in range(POSE_DIMS):
            gradient = torch.autograd.grad(
                output[dimension],
                codes,
                retain_graph=dimension < POSE_DIMS - 1,
            )[0]
            rows.append(gradient.detach().numpy().astype(np.float64, copy=False))
        return (
            output.detach().numpy().astype(np.float64, copy=False),
            np.stack(rows),
        )


def _cap1_stored_body(cap1: bytes) -> bytes:
    if not cap1.startswith(CAP1_PREFIX):
        raise PO1SolveError("candidate CAP1 prefix differs")
    raw = cap1[len(CAP1_PREFIX) :]
    if len(raw) < 6:
        raise PO1SolveError("candidate CAP1 bit counts are truncated")
    basis_bits = int.from_bytes(raw[:3], "little")
    rice_bits = int.from_bytes(raw[3:6], "little")
    sizes = {
        "predictor": 36,
        "scales": 96,
        "lengths": 32,
        "ks": 12,
        "basis": (basis_bits + 7) // 8,
        "rice": (rice_bits + 7) // 8,
    }
    offset = 6
    fields: dict[str, bytes] = {}
    for name in CAP_FIELDS:
        end = offset + sizes[name]
        fields[name] = raw[offset:end]
        offset = end
    if offset != len(raw):
        raise PO1SolveError("candidate CAP1 field accounting differs")
    return raw[:6] + b"".join(fields[name] for name in STORED_CAP_FIELDS)


def _copy_runtime(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("archive.zip", "__pycache__", "*.pyc", ".DS_Store"),
    )


def _adapt_runtime_for_variable_canonical_cap1(runtime_root: Path) -> dict[str, Any]:
    path = runtime_root / "runtime/residual_archive.py"
    original = path.read_text()
    old = """    carrier = sections[2]\n    if len(carrier) == PACKED_CAP1_SECTION_BYTES:\n        carrier = _restore_packed_cap1_metadata(carrier)\n    elif len(carrier) != CANONICAL_CAP1_SECTION_BYTES:\n        return None\n    return b\"F24S\" + sections[0] + sections[1] + carrier, outer[model_end:]\n"""
    new = """    carrier = sections[2]\n    # PO1 candidates carry the canonical stored CAP1 body because a changed\n    # Rice stream does not retain CP135's fixed packed-section byte count.\n    # Geometry is derived from the in-band CAP1 bit counts; the trailing F0E1\n    # selector remains mandatory and is validated by _decode_models.\n    try:\n        cap1_bytes = _cap1_body_bytes(carrier)\n    except ResidualArchiveError:\n        return None\n    if cap1_bytes >= len(carrier):\n        return None\n    return b\"F24S\" + sections[0] + sections[1] + carrier, outer[model_end:]\n"""
    if old not in original:
        if new in original:
            return {"already_adapted": True, "file": file_record(path)}
        raise PO1SolveError("candidate runtime CAP1 adaptation anchor differs")
    atomic_bytes(path, original.replace(old, new).encode())
    return {
        "already_adapted": False,
        "file": file_record(path),
        "generic_algorithm_only": True,
        "video_derived_bytes_embedded_in_code": False,
    }


def build_candidate(
    *,
    base_archive: Path,
    source_runtime: Path,
    output_runtime: Path,
    parts: Any,
    state: CarrierState,
    codes: np.ndarray,
) -> dict[str, Any]:
    _residual, repack, coefficient, _selector = _runtime_modules(source_runtime)
    canonical = encode_canonical_carrier(state, codes)
    cap1, _cap1_report = _cap1_encoder().encode_cap1(
        canonical,
        frames=N,
        dimensions=D,
    )
    restored = coefficient.decode_cap1(cap1, frames=N, dimensions=D)
    if restored != canonical:
        raise PO1SolveError("candidate CAP1 does not restore the changed CPR1")
    repack.pack_frame0_selector_carrier(cap1, state.selector)
    base_hpac, base_semantic, base_packed_carrier = unpack_split_models(
        parts.compressed_models,
        brotli_binary=str(BROTLI),
    )
    base_physical = unpack_cap1_metadata(base_packed_carrier)
    base_cap1, base_selector = repack.split_frame0_selector_carrier(parts.carrier_blob)
    expected_base_physical = _cap1_stored_body(base_cap1) + base_selector[len(SPARSE_SELECTOR_PREFIX) :]
    if base_physical != expected_base_physical:
        raise PO1SolveError("CP135 physical carrier parse-back differs before mutation")
    physical = _cap1_stored_body(cap1) + state.selector[len(SPARSE_SELECTOR_PREFIX) :]

    packed = False
    runtime_adaptation: dict[str, Any] | None = None
    try:
        packed_physical, _packed_report = pack_cap1_metadata(physical)
        selected_physical = packed_physical
        packed = True
    except RuntimeError:
        selected_physical = physical
    model_payload, model_report = pack_split_models(
        base_hpac,
        base_semantic,
        selected_physical,
        qualities=(10, 11, 11),
        brotli_binary=str(BROTLI),
    )
    member = model_payload + parts.residual_payload[4:] + parts.token_stream
    archive_payload = deterministic_zip(member)

    _copy_runtime(source_runtime, output_runtime)
    if not packed and len(physical) != 22_223:
        runtime_adaptation = _adapt_runtime_for_variable_canonical_cap1(output_runtime)
    archive_path = output_runtime / "archive.zip"
    atomic_bytes(archive_path, archive_payload)

    # cp135's inflate.py pins archive.zip to the promoted artifact (_verify_input
    # raises "does not match the promoted F26 artifact" — the Round-2 T4 refusal:
    # the local parse-back below reads the residual archive directly and never
    # exercised this pin). Repoint the pin to the candidate archive. A generic
    # self-verification constant repoint; no video-derived data enters code.
    inflate_path = output_runtime / "inflate.py"
    inflate_source = inflate_path.read_text()
    candidate_record = file_record(archive_path)
    old_sha_line = f'ARCHIVE_SHA256 = "{CP135_SHA256}"'
    old_bytes_line = f"ARCHIVE_BYTES = {CP135_BYTES:_}"
    new_sha_line = f'ARCHIVE_SHA256 = "{candidate_record["sha256"]}"'
    new_bytes_line = f"ARCHIVE_BYTES = {int(candidate_record['bytes']):_}"
    if old_sha_line not in inflate_source or old_bytes_line not in inflate_source:
        raise PO1SolveError("candidate inflate.py archive pin anchor differs")
    atomic_bytes(
        inflate_path,
        inflate_source.replace(old_sha_line, new_sha_line)
        .replace(old_bytes_line, new_bytes_line)
        .encode(),
    )
    if new_sha_line not in inflate_path.read_text():
        raise PO1SolveError("candidate inflate.py archive pin rewrite failed")

    # Parse-back runs in a clean interpreter so source/candidate runtime modules
    # cannot alias through Python's module cache.
    command = [
        str(REPO / ".venv/bin/python"),
        "-c",
        (
            "import os,sys;from pathlib import Path;"
            f"os.environ['CP135_BROTLI_CLI']={str(BROTLI)!r};"
            f"sys.path.insert(0,{str(output_runtime)!r});"
            "from runtime.residual_archive import read_residual_archive;"
            f"p=read_residual_archive(Path({str(archive_path)!r}));"
            "assert p.token_codec=='rc64';print(len(p.carrier_blob))"
        ),
    ]
    completed = __import__("subprocess").run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise PO1SolveError(f"candidate adapted receiver parse-back failed: {completed.stderr}")
    if packed and np.array_equal(codes, state.codes) and archive_payload != base_archive.read_bytes():
        raise PO1SolveError("identity coefficient rebuild did not reproduce CP135 bytes")
    return {
        "schema": "ddm_po1_candidate_build.v1",
        "archive": file_record(archive_path),
        "runtime_root": str(output_runtime.resolve()),
        "runtime_adaptation": runtime_adaptation,
        "cap1_metadata_packed": packed,
        "canonical_carrier_bytes": len(canonical),
        "canonical_carrier_sha256": sha256_bytes(canonical),
        "cap1_bytes": len(cap1),
        "cap1_sha256": sha256_bytes(cap1),
        "physical_carrier_bytes": len(physical),
        "selected_physical_carrier_bytes": len(selected_physical),
        "model_report": model_report,
        "residual_identity": True,
        "token_stream_identity": True,
        "selector_identity": True,
        "receiver_parseback_stdout": completed.stdout.strip(),
        "score_claim": False,
    }


def _round_paths(round_root: Path) -> tuple[Path, dict[str, Path]]:
    final = round_root / "FINAL_RESULT.json"
    require_file(final)
    result = json.loads(final.read_text())
    if result.get("execution_status") != "COMPLETE":
        raise PO1SolveError(f"T4 round is not complete: {final}")
    vector_root = round_root / "retained/pose_vectors"
    paths = {
        "gt": vector_root / "gt_first6_n600.npy",  # GT_LINEAGE_OK: round-local custody -- this is the T4 round's OWN emitted GT, read from that round's retained/pose_vectors/ next to the candidate vectors it is differenced against, so GT and candidate share one decode by construction; adjudicated correct (ddm_sp2)
        "first": vector_root / "candidate_first_first6_n600.npy",
        "repeat": vector_root / "candidate_repeat_first6_n600.npy",
        "error": vector_root / "pair_error_rms_n600.npy",
        "noise": vector_root / "pair_repeat_noise_rms_n600.npy",
    }
    records = {
        "gt": result["pose_scorers"]["gt"]["first6_vectors"],
        "first": result["pose_scorers"]["candidate_first"]["first6_vectors"],
        "repeat": result["pose_scorers"]["candidate_repeat"]["first6_vectors"],
        "error": result["pose_feedback"]["pair_error_rms"],
        "noise": result["pose_feedback"]["pair_repeat_noise_rms"],
    }
    for name, path in paths.items():
        record = records[name]
        require_file(path, size=int(record["bytes"]), digest=str(record["sha256"]))
    return final, paths


def solve(args: argparse.Namespace) -> int:
    archive = args.archive.resolve()
    runtime = args.runtime.resolve()
    round_root = args.round1.resolve()
    output = args.output.resolve()
    require_file(archive, size=CP135_BYTES, digest=CP135_SHA256)
    require_file(BROTLI)
    storage = storage_preflight(output)
    final_path, vector_paths = _round_paths(round_root)
    round_result = json.loads(final_path.read_text())
    if round_result.get("status") == "F1_INSTRUMENT_FLOOR":
        result = {
            "schema": "ddm_po1_solve_result.v1",
            "status": "CLOSED_F1_INSTRUMENT_FLOOR",
            "round1": file_record(final_path),
            "candidate_built": False,
            "reason": "same-job repeat noise is comparable to decoded-vs-GT error for most pairs",
        }
        atomic_json(output / "SOLVE_RESULT.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    gt = np.load(vector_paths["gt"], allow_pickle=False).astype(np.float64)
    first = np.load(vector_paths["first"], allow_pickle=False).astype(np.float64)
    repeat = np.load(vector_paths["repeat"], allow_pickle=False).astype(np.float64)
    pair_error = np.load(vector_paths["error"], allow_pickle=False).astype(np.float64)
    pair_noise = np.load(vector_paths["noise"], allow_pickle=False).astype(np.float64)
    if any(value.shape != (N, POSE_DIMS) for value in (gt, first, repeat)):
        raise PO1SolveError("round-1 T4 vector geometry differs")
    raw_path = round_root / "retained/raw/candidate/0.raw"
    raw_record = round_result["receiver"]["raw"]
    require_file(
        raw_path,
        size=int(raw_record["bytes"]),
        digest=str(raw_record["sha256"]),
    )
    if raw_path.stat().st_size != N * 2 * CAMERA_H * CAMERA_W * 3:
        raise PO1SolveError("round-1 retained raw has invalid n600 geometry")
    raw = np.memmap(raw_path, mode="r", dtype=np.uint8, shape=(N * 2, CAMERA_H, CAMERA_W, 3))
    parts, state = load_carrier(archive, runtime)

    state_path = output / "SOLVE_STATE.json"
    codes_path = output / "retained/coefficients.int16.npy"
    jacobian_path = output / "retained/jacobians.float64.npy"
    local_outputs_path = output / "retained/local_pose_outputs.float64.npy"
    predicted_path = output / "retained/predicted_t4_outputs.float64.npy"
    if args.resume:
        require_file(state_path)
        progress = json.loads(state_path.read_text())
        if progress.get("resume_from") != args.resume_from:
            raise PO1SolveError("resume token differs from retained solve state")
        cursor = int(progress["cursor"])
        codes = np.load(codes_path, allow_pickle=False)
        jacobians = np.load(jacobian_path, allow_pickle=False)
        local_outputs = np.load(local_outputs_path, allow_pickle=False)
        predicted = np.load(predicted_path, allow_pickle=False)
    else:
        if state_path.exists():
            raise PO1SolveError("solve state exists; pass --resume")
        cursor = 0
        codes = state.codes.copy()
        jacobians = np.full((N, POSE_DIMS, D), np.nan, dtype=np.float64)
        local_outputs = np.full((N, POSE_DIMS), np.nan, dtype=np.float64)
        predicted = first.copy()
    if codes.shape != (N, D) or jacobians.shape != (N, POSE_DIMS, D):
        raise PO1SolveError("retained solve arrays have invalid geometry")

    local = LocalJacobian(runtime, state)
    stop = N if args.max_pairs is None else min(N, cursor + args.max_pairs)
    pair_receipt_root = output / "retained/pair_receipts"
    ledger_path = output / "PAIR_LEDGER.jsonl"
    for pair in range(cursor, stop):
        current = state.codes[pair]
        local_output, jacobian = local.pose_and_jacobian(current, np.asarray(raw[2 * pair + 1]))
        local_outputs[pair] = local_output
        jacobians[pair] = jacobian
        skip_reason = None
        if state.selector_choices[pair] != 0:
            skip_reason = "nonidentity F0E1 selector; local-J renderer excludes pixel-mode derivative"
            selected = current.copy()
            diagnostics = {"rank": 0, "condition": None, "ridge_lambda": None}
        elif pair_noise[pair] >= 0.5 * pair_error[pair]:
            skip_reason = "pair repeat noise reaches pre-registered half-error instrument floor"
            selected = current.copy()
            diagnostics = {"rank": 0, "condition": None, "ridge_lambda": None}
        else:
            update, diagnostics = solve_damped_least_squares(
                jacobian,
                gt[pair] - first[pair],
                damping=args.damping,
                max_code_step=args.max_code_step,
            )
            selected = quantize_int12_update(current, update)
        codes[pair] = selected
        delta = selected.astype(np.float64) - current.astype(np.float64)
        predicted[pair] = first[pair] + jacobian @ delta
        # Per-pair acceptance gate: keep the quantized step only where the
        # linearized prediction strictly improves this pair's T4 pose error.
        # Ungated, int16 quantization turned a ~0-gain continuous step into
        # +55.6% predicted d_pose (250/600 pairs regressed, 10 pairs carried
        # 91.4% of the damage) — the js5 quantum-floor law on the pose lattice.
        quantized_gate_rejected = False
        if skip_reason is None:
            base_sq = float(np.mean(np.square(first[pair] - gt[pair])))
            pred_sq = float(np.mean(np.square(predicted[pair] - gt[pair])))
            if pred_sq >= base_sq:
                quantized_gate_rejected = True
                selected = current.copy()
                codes[pair] = selected
                predicted[pair] = first[pair]
        row = {
            "schema": "ddm_po1_pair_solve.v1",
            "pair": pair,
            "selector_choice": int(state.selector_choices[pair]),
            "skip_reason": skip_reason,
            "quantized_gate_rejected": quantized_gate_rejected,
            "t4_error_rms": float(pair_error[pair]),
            "t4_repeat_noise_rms": float(pair_noise[pair]),
            "local_vs_t4_output_rms": float(np.sqrt(np.mean(np.square(local_output - first[pair])))),
            "changed_coefficients": int(np.count_nonzero(selected != current)),
            "max_abs_code_step": int(np.max(np.abs(selected.astype(np.int32) - current.astype(np.int32)))),
            "jacobian_rank": diagnostics["rank"],
            "jacobian_condition": (
                diagnostics["condition"]
                if diagnostics["condition"] is None
                or np.isfinite(diagnostics["condition"])
                else None
            ),
            "ridge_lambda": diagnostics["ridge_lambda"],
        }
        checkpoint_once(pair_receipt_root / f"pair_{pair:04d}.json", row)
        atomic_npy(codes_path, codes)
        atomic_npy(jacobian_path, jacobians)
        atomic_npy(local_outputs_path, local_outputs)
        atomic_npy(predicted_path, predicted)
        atomic_json(
            state_path,
            {
                "schema": "ddm_po1_solve_state.v1",
                "resume_from": args.resume_from,
                "cursor": pair + 1,
                "pairs": N,
                "damping": args.damping,
                "max_code_step": args.max_code_step,
                "base_archive": file_record(archive),
                "round1": file_record(final_path),
                "complete": pair + 1 == N,
            },
        )

    if stop < N:
        result = {
            "schema": "ddm_po1_solve_result.v1",
            "status": "PARTIAL_RESUMABLE",
            "cursor": stop,
            "pairs": N,
            "resume_from": args.resume_from,
            "candidate_built": False,
        }
        atomic_json(output / "SOLVE_RESULT.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    changed = int(np.count_nonzero(codes != state.codes))
    pair_receipts = [pair_receipt_root / f"pair_{pair:04d}.json" for pair in range(N)]
    if any(not path.is_file() for path in pair_receipts):
        raise PO1SolveError("complete solve is missing an immutable per-pair receipt")
    atomic_bytes(
        ledger_path,
        b"".join(canonical_json_bytes(json.loads(path.read_text())) for path in pair_receipts),
    )
    predicted_base_dpose = float(np.mean(np.square(first - gt)))
    predicted_candidate_dpose = float(np.mean(np.square(predicted - gt)))
    candidate_runtime = output / "candidate_runtime"
    build = build_candidate(
        base_archive=archive,
        source_runtime=runtime,
        output_runtime=candidate_runtime,
        parts=parts,
        state=state,
        codes=codes,
    )
    candidate_archive_path = str(Path(build["archive"]["path"]))
    candidate_runtime_path = str(candidate_runtime.resolve())
    round2_output = "/Volumes/VertigoDataTier/pact/ddm_po1_20260813/dispatch/round2_candidate"
    round2_command = (
        ".venv/bin/modal run --detach experiments/ddm_po1_modal_t4_pose_feedback.py::main "
        f"--archive {candidate_archive_path} --runtime {candidate_runtime_path} "
        f"--expected-archive-bytes {build['archive']['bytes']} "
        f"--expected-archive-sha256 {build['archive']['sha256']} "
        f"--output-dir {round2_output} --run-id ddm_po1_round2_candidate_20260813 "
        "--resume-from ddm_po1_round2_candidate_20260813 --round-ordinal 2 "
        "--lane-id ddm_po1_t4_pose_feedback_round2 "
        "--instance-job-id modal:ddm_po1_round2_candidate_20260813 "
        "--claim-agent main:ddm_po1 --detach --provider-detach-ack"
    )
    result = {
        "schema": "ddm_po1_solve_result.v1",
        "status": "CANDIDATE_READY_FOR_T4_ROUND2",
        "axis": "[local macOS-CPU Jacobian preconditioner; T4 residual origin] NON-AUTHORITY",
        "resume_from": args.resume_from,
        "base_archive": file_record(archive),
        "round1": file_record(final_path),
        "storage": storage,
        "damping": args.damping,
        "max_code_step": args.max_code_step,
        "changed_coefficient_denominator": N * D,
        "changed_coefficients": changed,
        "nonidentity_selector_pairs_skipped": int(np.count_nonzero(state.selector_choices)),
        "predicted_base_d_pose_t4_origin": predicted_base_dpose,
        "predicted_candidate_d_pose_t4_origin": predicted_candidate_dpose,
        "predicted_gain": predicted_base_dpose - predicted_candidate_dpose,
        "candidate": build,
        "round2_fire_order": {
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "MAIN",
            "consumer_store": round2_output,
            "fire_trigger": (
                "round1 status FEEDBACK_USABLE, this solve complete, sole T4 component lane clear, "
                "and candidate archive bytes/SHA reverified"
            ),
            "dispatch_command": round2_command,
            "recover_command": (
                ".venv/bin/python experiments/ddm_po1_modal_t4_pose_feedback.py recover "
                f"--output-dir {round2_output}"
            ),
        },
        "retained": {
            "coefficients": file_record(codes_path),
            "jacobians": file_record(jacobian_path),
            "local_pose_outputs": file_record(local_outputs_path),
            "predicted_t4_outputs": file_record(predicted_path),
            "pair_ledger": file_record(ledger_path),
            "solve_state": file_record(state_path),
        },
        "score_claim": False,
        "promotion_eligible": False,
    }
    atomic_json(output / "SOLVE_RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def adjudicate(args: argparse.Namespace) -> int:
    round1_path = args.round1.resolve() / "FINAL_RESULT.json"
    round2_path = args.round2.resolve() / "FINAL_RESULT.json"
    solve_path = args.solve_result.resolve()
    for path in (round1_path, round2_path, solve_path):
        require_file(path)
    round1 = json.loads(round1_path.read_text())
    round2 = json.loads(round2_path.read_text())
    solve_result = json.loads(solve_path.read_text())
    r1_pose = float(round1["pose_feedback"]["d_pose_decoded_first"])
    r2_pose = float(round2["pose_feedback"]["d_pose_decoded_first"])
    predicted = float(solve_result["predicted_candidate_d_pose_t4_origin"])
    predicted_gain = r1_pose - predicted
    realized_gain = r1_pose - r2_pose
    realization = realized_gain / predicted_gain if predicted_gain > 0.0 else None

    r1_field = args.round1.resolve() / "retained/fields/candidate_argmax_n600.npy"
    r2_field = args.round2.resolve() / "retained/fields/candidate_argmax_n600.npy"
    r1_field_record = round1["seg_scorers"]["candidate"]["argmax"]
    r2_field_record = round2["seg_scorers"]["candidate"]["argmax"]
    require_file(
        r1_field,
        size=int(r1_field_record["bytes"]),
        digest=str(r1_field_record["sha256"]),
    )
    require_file(
        r2_field,
        size=int(r2_field_record["bytes"]),
        digest=str(r2_field_record["sha256"]),
    )
    r1_field_value = np.load(r1_field, mmap_mode="r", allow_pickle=False)
    r2_field_value = np.load(r2_field, mmap_mode="r", allow_pickle=False)
    if r1_field_value.shape != (N, SEG_H, SEG_W) or r2_field_value.shape != r1_field_value.shape:
        raise PO1SolveError("round SegNet fields have invalid geometry")
    field_changes = int(np.count_nonzero(r1_field_value != r2_field_value))
    base_bytes = int(round1["candidate_archive"]["bytes"])
    candidate_bytes = int(round2["candidate_archive"]["bytes"])
    r1_seg = float(round1["seg_feedback"]["d_seg"])
    r2_seg = float(round2["seg_feedback"]["d_seg"])
    base_component = 100.0 * r1_seg + math.sqrt(10.0 * r1_pose) + 25.0 * base_bytes / RATE_DENOMINATOR
    candidate_component = 100.0 * r2_seg + math.sqrt(10.0 * r2_pose) + 25.0 * candidate_bytes / RATE_DENOMINATOR
    joint_delta = candidate_component - base_component

    if field_changes:
        status = "CLOSED_F3_SEGNET_MOVED"
        disposition = "do not ship"
        third_round_allowed = False
    elif r2_pose >= r1_pose:
        status = "CLOSED_POSE_NOT_LOWER"
        disposition = "do not ship"
        third_round_allowed = False
    elif joint_delta >= 0.0:
        status = "CLOSED_JOINT_DELTA_NOT_NEGATIVE"
        disposition = "do not ship"
        third_round_allowed = False
    elif realization is None:
        status = "CLOSED_NONPOSITIVE_PREDICTED_GAIN"
        disposition = "do not ship"
        third_round_allowed = False
    elif realization < 0.2:
        status = "CLOSED_F2_BELOW_20_PERCENT_PREDICTION"
        disposition = (
            "retain one higher-damping local retry, but do not fire it because the independent "
            "third-round gate requires at least 50 percent realization"
        )
        third_round_allowed = False
    else:
        status = "ADMITTED_COMPONENT_IMPROVEMENT"
        disposition = "queue same-bytes exact evaluator replay under MAIN custody"
        third_round_allowed = realization >= 0.5

    candidate_build = solve_result.get("candidate", {})
    candidate_archive_record = candidate_build.get("archive", {})
    candidate_runtime = candidate_build.get("runtime_root")
    if status == "ADMITTED_COMPONENT_IMPROVEMENT":
        exact_output = "/Volumes/VertigoDataTier/pact/ddm_po1_20260813/exact_eval"
        follow_on = {
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "MAIN",
            "consumer_store": exact_output,
            "fire_trigger": (
                "component admission holds, sole exact lane clear, candidate bytes/SHA and runtime "
                "tree reverified"
            ),
            "command": (
                ".venv/bin/python tools/dispatch_modal_paired_auth_eval.py "
                f"--archive {candidate_archive_record.get('path')} "
                f"--expected-archive-sha256 {candidate_archive_record.get('sha256')} "
                f"--submission-dir {candidate_runtime} --inflate-sh inflate.sh "
                "--label ddm_po1_admitted_candidate --run-id ddm_po1_admitted_exact_20260813 "
                "--pair-group-id ddm_po1_admitted_exact_20260813 "
                "--lane-id-base lane_ddm_po1_admitted_exact_20260813 "
                f"--output-root {exact_output} --gpu T4 --claim-agent MAIN "
                "--claim-notes 'PO1 same-object T4 Pose feedback admitted; exact paired replay' "
                "--expected-runtime-tree-sha256 auto --execute"
            ),
        }
    elif status == "CLOSED_F2_BELOW_20_PERCENT_PREDICTION":
        retry_output = "/Volumes/VertigoDataTier/pact/ddm_po1_20260813/solve/attempt2_damped_retry"
        follow_on = {
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "ddm_po1 local solver",
            "consumer_store": retry_output,
            "fire_trigger": "this F2 adjudication, once only; retain result and close without T4 fire",
            "command": (
                ".venv/bin/python experiments/ddm_po1_t4_error_feedback_pose_compensation.py solve "
                f"--round1 {args.round1.resolve()} --output {retry_output} "
                "--resume-from ddm_po1_attempt2_damped_retry_20260813 "
                "--damping 0.04 --max-code-step 16"
            ),
            "t4_fire_after_retry": False,
            "reason": "the independent third-round gate requires at least 50 percent realization",
        }
    else:
        follow_on = {
            "disposition": "FOLDED",
            "owner": "ddm_po1",
            "consumer_store": None,
            "fire_trigger": None,
            "reason": status,
        }

    result = {
        "schema": "ddm_po1_round2_adjudication.v1",
        "status": status,
        "disposition": disposition,
        "round1": file_record(round1_path),
        "round2": file_record(round2_path),
        "solve_result": file_record(solve_path),
        "base": {"d_pose": r1_pose, "d_seg": r1_seg, "archive_bytes": base_bytes},
        "candidate": {"d_pose": r2_pose, "d_seg": r2_seg, "archive_bytes": candidate_bytes},
        "predicted_candidate_d_pose": predicted,
        "predicted_gain": predicted_gain,
        "realized_gain": realized_gain,
        "prediction_realization_fraction": realization,
        "seg_field_changed_pixels": field_changes,
        "seg_field_denominator": N * SEG_H * SEG_W,
        "joint_score_component_base": base_component,
        "joint_score_component_candidate": candidate_component,
        "joint_delta_s": joint_delta,
        "pose_lower": r2_pose < r1_pose,
        "seg_unchanged": field_changes == 0,
        "joint_delta_negative": joint_delta < 0.0,
        "third_round_allowed": third_round_allowed,
        "follow_on": follow_on,
        "score_claim": False,
        "pointer_moved": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    solve_parser = sub.add_parser("solve")
    solve_parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    solve_parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    solve_parser.add_argument("--round1", type=Path, default=DEFAULT_ROUND1)
    solve_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    solve_parser.add_argument("--resume-from", required=True)
    solve_parser.add_argument("--resume", action="store_true")
    solve_parser.add_argument("--damping", type=float, default=0.01)
    solve_parser.add_argument("--max-code-step", type=float, default=32.0)
    solve_parser.add_argument("--max-pairs", type=int)
    solve_parser.set_defaults(func=solve)

    adjudicate_parser = sub.add_parser("adjudicate")
    adjudicate_parser.add_argument("--round1", type=Path, required=True)
    adjudicate_parser.add_argument("--round2", type=Path, required=True)
    adjudicate_parser.add_argument("--solve-result", type=Path, required=True)
    adjudicate_parser.add_argument("--output", type=Path, required=True)
    adjudicate_parser.set_defaults(func=adjudicate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if getattr(args, "max_pairs", None) is not None and args.max_pairs <= 0:
        raise PO1SolveError("--max-pairs must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
