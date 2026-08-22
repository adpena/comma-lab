#!/usr/bin/env python3
"""Build the measured DX1 CABAC coefficient win into the live FX5 receiver.

This is a lossless, byte-only fold.  It does not run a scorer or dispatch
remote work.  The builder pins the retained FX5 archive and DX1 artifacts,
re-encodes the retained 600x12 symbol array, requires byte equality with the
measured cap=8 winner, composes reserved bit 0x10 with RR5's 0x08, and emits a
runtime whose receiver restores the original Rice carrier byte-for-byte.

Every materialized payload is written under ``--out-dir`` before a byte claim
is admitted.  A wrong archive delta is retained and reported as a refusal.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

# Receiver parse-back imports the staged runtime.  Never let that verification
# contaminate the candidate tree with host-specific bytecode (#1122 genus).
sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac import dx2_cabac_coefficients as dx2

BASE_RUNTIME = Path("/Volumes/APDataStore/pact/ddm_fx5/candidate_runtime_fx5")
BASE_ARCHIVE_SHA256 = (
    "4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841"
)
BASE_ARCHIVE_BYTES = 180_386
BASE_RAW_SHA256_MACOS = (
    "7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7"
)
BASE_RAW_SHA256_CONTEST_CUDA = (
    "6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883"
)
BASE_RAW_BYTES = 3_662_409_600
BASE_T4_RECEIPT = Path(
    "/Volumes/APDataStore/pact/ddm_fx5/t4_row_r1/MODAL_REMOTE_RESULT.json"
)
BASE_T4_RECEIPT_SHA256 = (
    "bc78d8493c67a4b4174a9143707e8c67df2ed90daccfcf1c72595fb86a709f30"
)

DX1_RETAINED = Path("/Volumes/VertigoDataTier/pact/ddm_dx1/retained")
DX1_RACE = DX1_RETAINED / "DX1_RECODE_RACE.json"
DX1_SYMBOLS = DX1_RETAINED / "dx1_coded_symbols_U.int32.npy"
DX1_PAYLOAD = (
    DX1_RETAINED / "dx1_payload_adaptive-ctx_Rice_CABAC_prefix_cap8.bin"
)
DX1_RACE_FILE_SHA256 = (
    "95a8462d23dbd36c5d2d03a66cc3f94aa00f73dada16d18e15059c3cc7a69061"
)
DX1_SYMBOLS_FILE_SHA256 = (
    "8fc44020c3d5cb8ebe7d4adfabe7d1b0e05ad321f85bed03cb7086f04f201d95"
)
DX1_SYMBOLS_CONTENT_SHA256 = (
    "0bfe31cf9586104f4308329fec8f76f748c56441ac5bd85b824dfcca3434db50"
)
DX1_PAYLOAD_SHA256 = (
    "b93131a52674abb4ada677e1b6cf08eebc6afb94381136d23d010e70a287e210"
)
DX1_PAYLOAD_BYTES = 9_811

BASE_FILE_PINS = {
    "inflate.py": "693f43fc433139ed9ce3c8b0ac695487c04f551a5128083bb819421678fe06ef",
    "runtime/residual_archive.py": (
        "e62489099c6d6d236bbb946ccd5fc9f55e75696dd74c0a1e0ebeece093bede5e"
    ),
    "runtime/rr5_arith_basis.py": (
        "c44758dfa6b530b0e3185241c05971f75b14db7f4d2a329763f1a7bc0332c0bb"
    ),
}

UNCOMPRESSED_BYTES = 37_545_489
EXPECTED_ARCHIVE_BYTES = BASE_ARCHIVE_BYTES - 18
EXPECTED_DELTA_S = -18 * 25.0 / UNCOMPRESSED_BYTES


class Dx2BuildError(RuntimeError):
    """A custody pin or lossless-fold control refused the build."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_once(text: str, anchor: str, replacement: str, label: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise Dx2BuildError(
            f"receiver patch anchor {label!r} appears {count} times, expected exactly 1"
        )
    return text.replace(anchor, replacement, 1)


def _load_rr5_tool():
    path = REPO / "tools/ddm_rr5_rider_apply.py"
    spec = importlib.util.spec_from_file_location("ddm_rr5_rider_apply_dx2", path)
    if spec is None or spec.loader is None:
        raise Dx2BuildError(f"cannot import retained container builder {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pin(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise Dx2BuildError(f"{label} sha256 {actual} != pinned {expected}")
    return actual


def _source_custody() -> tuple[dict[str, Any], np.ndarray, bytes, dict[str, Any]]:
    pins = {
        "race_receipt_file_sha256": _pin(
            DX1_RACE, DX1_RACE_FILE_SHA256, "DX1 race receipt"
        ),
        "symbols_npy_file_sha256": _pin(
            DX1_SYMBOLS, DX1_SYMBOLS_FILE_SHA256, "DX1 symbols npy"
        ),
        "winner_payload_file_sha256": _pin(
            DX1_PAYLOAD, DX1_PAYLOAD_SHA256, "DX1 winner payload"
        ),
    }
    race = json.loads(DX1_RACE.read_text())
    best = race.get("best_recode", {})
    required = {
        "coder": "adaptive-ctx Rice (CABAC prefix, cap=8)",
        "payload_bytes": DX1_PAYLOAD_BYTES,
        "total_bytes": DX1_PAYLOAD_BYTES,
        "delta_bytes_vs_shipped": -18,
        "decode_identity": True,
        "payload_sha256": DX1_PAYLOAD_SHA256,
    }
    for key, value in required.items():
        if best.get(key) != value:
            raise Dx2BuildError(
                f"DX1 race best_recode.{key}={best.get(key)!r}, expected {value!r}"
            )
    symbols = np.load(DX1_SYMBOLS, allow_pickle=False)
    if symbols.shape != (dx2.N_FRAMES, dx2.CARRIER_DIM) or symbols.dtype != np.int32:
        raise Dx2BuildError("DX1 retained symbol array is not int32[600,12]")
    content_sha = sha256_bytes(np.ascontiguousarray(symbols).tobytes())
    if content_sha != DX1_SYMBOLS_CONTENT_SHA256:
        raise Dx2BuildError(
            f"DX1 symbol content sha256 {content_sha} != pinned {DX1_SYMBOLS_CONTENT_SHA256}"
        )
    pins["symbols_array_content_sha256"] = content_sha
    payload = DX1_PAYLOAD.read_bytes()
    if len(payload) != DX1_PAYLOAD_BYTES:
        raise Dx2BuildError("DX1 winner payload is not 9,811 bytes")
    return race, symbols, payload, pins


def _patch_runtime(base: Path, destination: Path, archive: bytes) -> dict[str, Any]:
    if destination.exists():
        raise Dx2BuildError(f"runtime destination already exists: {destination}")
    for relative, expected in BASE_FILE_PINS.items():
        _pin(base / relative, expected, f"FX5 {relative}")
    shutil.copytree(
        base,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*"),
        copy_function=shutil.copyfile,
    )

    coder_source = REPO / "src/tac/dx2_cabac_coefficients.py"
    coder_target = destination / "runtime/dx2_cabac_coefficients.py"
    # copyfile deliberately omits macOS xattrs: copying them to APDataStore
    # materializes non-portable AppleDouble sidecars inside the runtime tree.
    shutil.copyfile(coder_source, coder_target)

    receiver = destination / "runtime/residual_archive.py"
    before = receiver.read_text()
    constants_anchor = """# DDM_RR5_ARITH_BASIS_V1: the rider's reserved bit joins the known set.
RR5_RESERVED_ARITH_BASIS = 0x08
SZ1_RESERVED_KNOWN_BITS = 0x0F"""
    constants_patch = """# DDM_RR5_ARITH_BASIS_V1: the basis rider's reserved bit.
RR5_RESERVED_ARITH_BASIS = 0x08
# DDM_DX2_CABAC_COEFFICIENTS_V1: disjoint coefficient-stream rider.
DX2_RESERVED_CABAC_COEFFICIENTS = 0x10
SZ1_RESERVED_KNOWN_BITS = 0x1F"""
    hook_anchor = """    if reserved & RR5_RESERVED_ARITH_BASIS:
        from .rr5_arith_basis import restore_carrier_body

        carrier_body = restore_carrier_body(carrier_body)
    # DDM_RX1_HP4_CARRIER_INVERSE_V1"""
    hook_patch = """    if reserved & RR5_RESERVED_ARITH_BASIS:
        from .rr5_arith_basis import restore_carrier_body as restore_rr5_carrier_body

        carrier_body = restore_rr5_carrier_body(carrier_body)
    # DDM_DX2_CABAC_COEFFICIENTS_V1: the coefficient rider is disjoint from
    # RR5's basis rider. Restore it after RR5 and before packed-CAP1 framing
    # reads the changed residual bit count. Integer-only, device-free decoder.
    if reserved & DX2_RESERVED_CABAC_COEFFICIENTS:
        from .dx2_cabac_coefficients import restore_carrier_body as restore_dx2_carrier_body

        carrier_body = restore_dx2_carrier_body(carrier_body)
    # DDM_RX1_HP4_CARRIER_INVERSE_V1"""
    after = replace_once(before, constants_anchor, constants_patch, "reserved mask")
    after = replace_once(after, hook_anchor, hook_patch, "CABAC restore hook")
    receiver.write_text(after)

    archive_path = destination / "archive.zip"
    archive_path.write_bytes(archive)
    archive_sha = sha256_bytes(archive)
    inflate = destination / "inflate.py"
    inflate_before = inflate.read_text()
    old_sha_line = next(
        line
        for line in inflate_before.splitlines()
        if line.startswith("ARCHIVE_SHA256 = ")
    )
    old_bytes_line = next(
        line for line in inflate_before.splitlines() if line.startswith("ARCHIVE_BYTES = ")
    )
    inflate_after = replace_once(
        inflate_before,
        old_sha_line,
        f'ARCHIVE_SHA256 = "{archive_sha}"',
        "inflate archive sha",
    )
    inflate_after = replace_once(
        inflate_after,
        old_bytes_line,
        f"ARCHIVE_BYTES = {len(archive):_}",
        "inflate archive bytes",
    )
    inflate.write_text(inflate_after)

    # The archive pin is derived from the file just written, then re-read.
    if sha256_file(archive_path) != archive_sha or archive_path.stat().st_size != len(
        archive
    ):
        raise Dx2BuildError("runtime archive pin derivation disagrees with disk")
    return {
        "base_receiver_sha256": sha256_bytes(before.encode()),
        "candidate_receiver_sha256": sha256_file(receiver),
        "coder_module_sha256": sha256_file(coder_target),
        "coder_module_bytes": coder_target.stat().st_size,
        "inflate_sha256": sha256_file(inflate),
        "archive_sha256": archive_sha,
        "archive_bytes": len(archive),
    }


def _zip_member(archive: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as zipped:
        if zipped.namelist() != ["p"]:
            raise Dx2BuildError("candidate archive must contain exactly member p")
        payload = zipped.read("p")
    return {"name": "p", "bytes": len(payload), "sha256": sha256_bytes(payload)}


def _runtime_files(runtime: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(p for p in runtime.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(runtime)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _remove_appledouble(root: Path) -> list[str]:
    """Remove non-semantic macOS sidecars created by the SSD filesystem bridge."""

    removed = []
    for path in sorted(root.rglob("._*")):
        if path.is_file() or path.is_symlink():
            removed.append(str(path.relative_to(root)))
            path.unlink()
    return removed


def build(out_dir: Path) -> dict[str, Any]:
    started = time.time()
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise Dx2BuildError(f"output directory already exists: {out_dir}")
    retained = out_dir / "retained"
    runtime = out_dir / "candidate_runtime_dx2"
    retained.mkdir(parents=True)

    base_archive = BASE_RUNTIME / "archive.zip"
    _pin(base_archive, BASE_ARCHIVE_SHA256, "FX5 archive")
    if base_archive.stat().st_size != BASE_ARCHIVE_BYTES:
        raise Dx2BuildError("FX5 archive is not 180,386 bytes")
    race, retained_symbols, retained_payload, source_pins = _source_custody()

    tool = _load_rr5_tool()
    container = tool.parse_container(
        base_archive, BASE_RUNTIME, expect_sha256=BASE_ARCHIVE_SHA256
    )
    identity = tool.identity_control(container)
    if not identity.get("byte_identical"):
        raise Dx2BuildError("container identity control cannot reproduce FX5 byte-for-byte")
    incumbent = identity["container"]

    applied = dx2.apply_cabac_to_carrier_body(container.carrier_body)
    symbols = np.asarray(applied["symbols"], dtype=np.int32)
    if not np.array_equal(symbols, retained_symbols):
        raise Dx2BuildError("FX5 carrier symbols differ from the retained DX1 object")
    if sha256_bytes(symbols.tobytes()) != DX1_SYMBOLS_CONTENT_SHA256:
        raise Dx2BuildError("FX5 carrier symbol content hash differs from DX1")
    produced_payload = bytes(applied["cabac_payload"])
    # This is the charter's optimal-form gate: re-encode, do not merely splice.
    if produced_payload != retained_payload:
        raise Dx2BuildError("re-encoded cap=8 payload differs from the measured DX1 winner")

    candidate_body = bytes(applied["body"])
    if len(container.carrier_body) - len(candidate_body) != 18:
        raise Dx2BuildError("CABAC carrier-body delta is not exactly -18 bytes")
    if dx2.restore_carrier_body(candidate_body) != container.carrier_body:
        raise Dx2BuildError("decoder does not restore the exact FX5 carrier body")

    candidate_bytes = tool._build_archive(
        container,
        candidate_body,
        ck2=bool(incumbent["ck2_carrier"]),
        quality=int(incumbent["quality"]),
        lgwin=int(incumbent["lgwin"]),
        reserved_extra=dx2.DX2_RESERVED_CABAC_COEFFICIENTS,
    )
    repeat_bytes = tool._build_archive(
        container,
        candidate_body,
        ck2=bool(incumbent["ck2_carrier"]),
        quality=int(incumbent["quality"]),
        lgwin=int(incumbent["lgwin"]),
        reserved_extra=dx2.DX2_RESERVED_CABAC_COEFFICIENTS,
    )
    if candidate_bytes is None or repeat_bytes is None:
        raise Dx2BuildError("incumbent container settings refused the DX2 body")

    # Persist every materialized payload before admitting any archive delta.
    candidate_archive = retained / "candidate_dx2_cabac.zip"
    repeat_archive = retained / "candidate_dx2_cabac.repeat.zip"
    generated_payload = retained / "dx2_payload_adaptive_ctx_rice_cap8.bin"
    corrupt_payload_path = retained / "negative_control_corrupt_cabac.bin"
    candidate_archive.write_bytes(candidate_bytes)
    repeat_archive.write_bytes(repeat_bytes)
    generated_payload.write_bytes(produced_payload)
    corrupt_payload = bytearray(produced_payload)
    corrupt_payload[len(corrupt_payload) // 2] ^= 0x01
    corrupt_payload_path.write_bytes(bytes(corrupt_payload))
    try:
        dx2.decode_cabac_checked(bytes(corrupt_payload), applied["ks"])
    except dx2.CabacCoefficientError:
        corrupt_control_fired = True
    else:
        corrupt_control_fired = False
    if not corrupt_control_fired:
        raise Dx2BuildError("corrupted CABAC negative control did not fire")

    archive_delta = len(candidate_bytes) - BASE_ARCHIVE_BYTES
    if candidate_bytes != repeat_bytes:
        raise Dx2BuildError("deterministic archive repeat differs")
    if archive_delta != -18 or len(candidate_bytes) != EXPECTED_ARCHIVE_BYTES:
        raise Dx2BuildError(
            f"archive delta is {archive_delta:+d} B, required exactly -18 B"
        )

    runtime_patch = _patch_runtime(BASE_RUNTIME, runtime, candidate_bytes)
    receiver_identity = tool.receiver_decode_identity(
        base_archive, candidate_archive, BASE_RUNTIME, runtime
    )
    if not receiver_identity.get("identical"):
        raise Dx2BuildError(
            f"receiver decode identity failed: {receiver_identity.get('mismatched_fields')}"
        )

    # A second import of the candidate receiver must reproduce the same parsed proof.
    receiver_repeat = tool.receiver_decode_identity(
        base_archive, repeat_archive, BASE_RUNTIME, runtime
    )
    if not receiver_repeat.get("identical"):
        raise Dx2BuildError("repeat receiver parse-back is not identical")
    removed_appledouble = _remove_appledouble(runtime)
    bytecode_residue = [
        path
        for path in runtime.rglob("*")
        if "__pycache__" in path.parts or path.suffix == ".pyc"
    ]
    if bytecode_residue:
        raise Dx2BuildError(
            "candidate runtime contains host bytecode after verification: "
            + ", ".join(str(path.relative_to(runtime)) for path in bytecode_residue[:5])
        )
    appledouble_residue = list(runtime.rglob("._*"))
    if appledouble_residue:
        raise Dx2BuildError(
            "candidate runtime contains AppleDouble residue after cleanup: "
            + ", ".join(
                str(path.relative_to(runtime)) for path in appledouble_residue[:5]
            )
        )

    _pin(BASE_T4_RECEIPT, BASE_T4_RECEIPT_SHA256, "FX5 contest-CUDA T4 receipt")
    t4 = json.loads(BASE_T4_RECEIPT.read_text())
    required_t4 = {
        "expected_archive_sha256": BASE_ARCHIVE_SHA256,
        "expected_archive_size_bytes": BASE_ARCHIVE_BYTES,
        "score_axis": "contest_cuda",
        "gpu_t4_match": True,
        "n_samples": 600,
        "passed": True,
        "score_claim": True,
    }
    for key, value in required_t4.items():
        if t4.get(key) != value:
            raise Dx2BuildError(
                f"FX5 T4 receipt {key}={t4.get(key)!r}, expected {value!r}"
            )
    if t4.get("validation_errors") != []:
        raise Dx2BuildError("FX5 T4 receipt carries validation errors")
    base_score = float(t4["score_recomputed_from_components"])
    projected_score = base_score + EXPECTED_DELTA_S
    runtime_files = _runtime_files(runtime)
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    reproduction = {
        "argv": [
            str(REPO / ".venv/bin/python"),
            str(REPO / "experiments/ddm_dx2_cabac_receiver_fold.py"),
            "--out-dir",
            str(out_dir),
        ],
        "cwd": str(REPO),
        "git_head_at_build": git_head,
        "source_sha256": {
            "builder": sha256_file(
                REPO / "experiments/ddm_dx2_cabac_receiver_fold.py"
            ),
            "coder_and_receiver": sha256_file(
                REPO / "src/tac/dx2_cabac_coefficients.py"
            ),
            "container_tool": sha256_file(REPO / "tools/ddm_rr5_rider_apply.py"),
            "base_t4_receipt": sha256_file(BASE_T4_RECEIPT),
        },
    }
    result: dict[str, Any] = {
        "schema": "ddm_dx2_cabac_receiver_fold.v1",
        "axis": "[macOS-CPU scorer-free exact byte and receiver parse-back]",
        "score_claim": False,
        "promotion_eligible": False,
        "base": {
            "archive": str(base_archive),
            "archive_bytes": BASE_ARCHIVE_BYTES,
            "archive_sha256": BASE_ARCHIVE_SHA256,
            "contest_cuda_t4_score": base_score,
            "contest_cuda_t4_receipt": str(BASE_T4_RECEIPT),
            "d_seg": float(t4["avg_segnet_dist"]),
            "d_pose": float(t4["avg_posenet_dist"]),
        },
        "source_custody": {
            **source_pins,
            "race_receipt": str(DX1_RACE),
            "symbols_npy": str(DX1_SYMBOLS),
            "winner_payload": str(DX1_PAYLOAD),
            "race_best_recode": race["best_recode"],
            "note": (
                "the charter's 0bfe31... pin is the raw int32 array-content hash; "
                "the .npy container file hash is 8fc440...; both are verified"
            ),
        },
        "fold": {
            "reserved_before": int(container.reserved),
            "reserved_after": int(
                container.reserved | dx2.DX2_RESERVED_CABAC_COEFFICIENTS
            ),
            "rr5_basis_flag_preserved": bool(container.reserved & 0x08),
            "dx2_coefficients_flag_set": True,
            "symbols_shape": list(symbols.shape),
            "symbols_content_sha256": sha256_bytes(symbols.tobytes()),
            "rice_payload_bytes": len(bytes(applied["rice_payload"])),
            "rice_payload_bits": int(applied["rice_bits"]),
            "cabac_payload_bytes": len(produced_payload),
            "cabac_payload_sha256": sha256_bytes(produced_payload),
            "carrier_body_bytes_before": len(container.carrier_body),
            "carrier_body_bytes_after": len(candidate_body),
            "carrier_body_delta_bytes": len(candidate_body) - len(container.carrier_body),
            "device_path": "integer-only Python + NumPy byte arrays; no torch/CUDA/MPS branch",
        },
        "candidate": {
            "archive": str(candidate_archive),
            "archive_bytes": len(candidate_bytes),
            "archive_sha256": sha256_bytes(candidate_bytes),
            "archive_member": _zip_member(candidate_archive),
            "archive_delta_bytes": archive_delta,
            "delta_S_rate": EXPECTED_DELTA_S,
            "projected_contest_cuda_score_if_identity_holds": projected_score,
            "runtime": str(runtime),
            "runtime_patch": runtime_patch,
        },
        "controls": {
            "C1_fx5_container_identity": identity,
            "C2_reencoded_payload_equals_dx1_winner": True,
            "C3_cabac_roundtrip_symbols_exact": True,
            "C4_cabac_restore_equals_fx5_carrier_body": True,
            "C5_archive_delta_exactly_minus_18": True,
            "C6_deterministic_archive_repeat": {
                "identical": candidate_bytes == repeat_bytes,
                "repeat_path": str(repeat_archive),
                "repeat_sha256": sha256_bytes(repeat_bytes),
                "repeat_bytes": len(repeat_bytes),
            },
            "C7_real_receiver_parseback_identity": receiver_identity,
            "C8_repeat_real_receiver_parseback_identity": receiver_repeat,
            "C9_corrupted_cabac_payload_refused": {
                "fired": corrupt_control_fired,
                "payload": str(corrupt_payload_path),
                "bytes": corrupt_payload_path.stat().st_size,
                "sha256": sha256_file(corrupt_payload_path),
            },
        },
        "raw_identity_gate": {
            "status": "NOT_RUN_HEAVY_LOCAL_SLOT_OCCUPIED",
            "required_macos_raw_sha256": BASE_RAW_SHA256_MACOS,
            "required_contest_cuda_raw_sha256": BASE_RAW_SHA256_CONTEST_CUDA,
            "required_raw_bytes": BASE_RAW_BYTES,
            "reason": (
                "the charter forbids a heavy local launch while MAIN's governed JO4 "
                "local_cpu solve owns the slot; receiver parse-back is complete but is "
                "not mislabeled as the required fresh-process 0.raw proof"
            ),
        },
        "storage": {
            "out_dir": str(out_dir),
            "free_bytes_after": shutil.disk_usage(out_dir).free,
            "all_materialized_payloads_retained": True,
            "nonsemantic_appledouble_sidecars_removed": removed_appledouble,
        },
        "reproduction": reproduction,
        "runtime_files": runtime_files,
        "elapsed_seconds": time.time() - started,
    }
    result_path = out_dir / "RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    retention_items = []
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        retention_items.append(
            {
                "relative_path": str(path.relative_to(out_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema": "ddm_dx2_retention_manifest.v1",
        "policy": "every materialized payload is retained with bytes and sha256",
        "items": retention_items,
        "external_source_payloads": {
            "dx1_retained": str(DX1_RETAINED),
            "fx5_runtime": str(BASE_RUNTIME),
        },
    }
    (out_dir / "RETENTION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return result


def _write_refusal(out_dir: Path, error: Exception) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        payloads.append(
            {
                "relative_path": str(path.relative_to(out_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    receipt = {
        "schema": "ddm_dx2_build_refusal.v1",
        "verdict": "REFUSED",
        "error_type": type(error).__name__,
        "error": str(error),
        "retained_payloads": payloads,
    }
    (out_dir / "REFUSAL.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build(args.out_dir)
    except Exception as error:
        _write_refusal(args.out_dir, error)
        print(f"DX2 REFUSED: {error}", file=sys.stderr)
        return 3
    print(
        "DX2 BYTE-CLOSED "
        f"{result['candidate']['archive_bytes']} B "
        f"sha={result['candidate']['archive_sha256']} "
        f"delta={result['candidate']['archive_delta_bytes']:+d} B"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
