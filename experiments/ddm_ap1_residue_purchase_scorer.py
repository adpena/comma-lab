#!/usr/bin/env python3
"""DDM-AP1: receiver-closed DX2 residue purchase scorer.

This arm changes exactly one counted DX2 parameter group at a time along the
group's existing quantization axis, retains every encoded payload and archive,
and scores the resulting full n600 raw field against the pinned DALI GT tables.

The command is deliberately split into resumable stages:

``materialize``
    Build the exact control and twelve candidate archives (three levels for
    semantic, carrier, HPAC, and the fixed residual table).  Each candidate is
    copied into its own receiver tree and parse-backed through that tree.

``score``
    Consume the retained raw emitted by ``tools/fire_local_advisory.py`` for
    one candidate.  Frozen CPU SegNet/PoseNet outputs are retained per chunk,
    then reduced against the pinned n600 DALI GT arrays.

``aggregate``
    Join all thirteen complete scorer receipts into the purchase table.  It
    refuses partial results and recomputes every score component from integer
    denominators rather than trusting evaluate.py's rounded final score.

No command edits the shipped archive or receiver in place.  ``run-queue``
launches local advisories only by invoking ``tools/fire_local_advisory.py`` as
required by the common arm contract; it never invokes the evaluator directly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
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
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import brotli
import numpy as np

# The pinned upstream mirror is a read-only authority input.  Importing its
# scorer modules must never materialize ``__pycache__`` beside those inputs.
sys.dont_write_bytecode = True

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SOURCE_ARCHIVE: Final = Path(
    "/Volumes/APDataStore/pact/ddm_dx2/r7/retained/candidate_dx2_cabac.zip"
)
SOURCE_RUNTIME: Final = Path(
    "/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2"
)
SOURCE_ARCHIVE_BYTES: Final = 180_368
SOURCE_ARCHIVE_SHA256: Final = (
    "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"
)
SOURCE_TOKEN_BYTES: Final = 113_777
SOURCE_TOKEN_SHA256: Final = (
    "e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5"
)
RECEIPT_ROOT: Final = (
    REPO / ".omx/tmp/arm_receipts_local/ddm_ap1_residue_purchase_scorer"
)
GT_SEG: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/"
    "gt_argmax_n600.npy"
)
GT_SEG_SHA256: Final = (
    "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248"
)
GT_POSE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/"
    "direct_v6/full_n600_eval/retained/pose_vectors/gt_first6_dali_n600.npy"
)
GT_POSE_SHA256: Final = (
    "8d5cfa83df55b89493ba43b1e5386d792c836c32791666192499a089068e7eff"
)
UPSTREAM: Final = Path("/Volumes/APDataStore/pact/upstream_eval_mirror_20260815")
ADVISORY_UPSTREAM: Final = (
    RECEIPT_ROOT / "retained/upstream_eval_mirror_20260815_clean"
)
ADVISORY_TOKEN_CACHE: Final = RECEIPT_ROOT / "retained/f26_token_cache"
UPSTREAM_SNAPSHOT_SHA256: Final = (
    "fa7c4bf51d47a6140ec0f95275ebf86b0e6c3c1dc00caff03a417ee989645799"
)
VIDEO_NAMES: Final = ADVISORY_UPSTREAM / "public_test_video_names.txt"
N: Final = 600
SEG_H: Final = 384
SEG_W: Final = 512
SEG_PIXELS: Final = N * SEG_H * SEG_W
POSE_VALUES: Final = N * 6
RAW_BYTES: Final = 3_662_409_600
CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
CLASS_DENOMINATORS: Final = (27_407_372, 690_754, 58_413_067, 1_460_386, 29_993_221)
ORIGINAL_BYTES: Final = 37_545_489
S_PER_BYTE: Final = 25.0 / ORIGINAL_BYTES
DEMAND_BYTES: Final = 42_382
AXIS: Final = "[macOS-CPU advisory; DALI-GT pinned n600]"
LOCAL_OPT_IN: Final = "local_disk_explicit_opt_in_because_both_ssd_tiers_are_full"
MIN_FREE_BYTES: Final = 96 * (1 << 30)


class AP1Error(RuntimeError):
    """A source, custody, receiver, scorer, or reduction invariant failed."""


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    group: str
    level: int
    quantum: str


SPECS: Final = (
    CandidateSpec("control", "control", 0, "shipped_exact"),
    CandidateSpec("semantic_l1", "semantic", 1, "q4_to_q3_keep_existing_q3"),
    CandidateSpec("semantic_l2", "semantic", 2, "q4_to_q2_keep_existing_q3"),
    CandidateSpec("semantic_l3", "semantic", 3, "all_quantized_tensors_q2"),
    CandidateSpec(
        "carrier_l1_fixed_coder", "carrier", 1, "basis_and_coefficient_lattice_step_2"
    ),
    CandidateSpec(
        "carrier_l2_fixed_coder", "carrier", 2, "basis_and_coefficient_lattice_step_4"
    ),
    CandidateSpec(
        "carrier_l3_fixed_coder", "carrier", 3, "basis_and_coefficient_lattice_step_8"
    ),
    CandidateSpec("hpac_l1", "hpac", 1, "each_IHS1_row_depth_minus_1"),
    CandidateSpec("hpac_l2", "hpac", 2, "each_IHS1_row_depth_minus_2"),
    CandidateSpec("hpac_l3", "hpac", 3, "each_IHS1_row_depth_minus_3"),
    CandidateSpec("residual_l1", "residual", 1, "signed6_code_lattice_step_2"),
    CandidateSpec("residual_l2", "residual", 2, "signed6_code_lattice_step_4"),
    CandidateSpec("residual_l3", "residual", 3, "signed6_code_lattice_step_8"),
)
SPEC_BY_ID: Final = {spec.candidate_id: spec for spec in SPECS}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise AP1Error(f"required file is absent: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def tree_fact(root: Path) -> dict[str, Any]:
    records = []
    total = 0
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if path.name.startswith("._") or "__pycache__" in relative.parts:
            continue
        fact = file_fact(path)
        record = {
            "relative_path": relative.as_posix(),
            "bytes": fact["bytes"],
            "sha256": fact["sha256"],
        }
        records.append(record)
        total += int(fact["bytes"])
        digest.update(
            (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    return {
        "path": str(root.resolve()),
        "file_count": len(records),
        "bytes": total,
        "tree_sha256": digest.hexdigest(),
        "hash_law": "sha256 over sorted compact-json lines of relative_path, bytes, sha256",
    }


def validate_fact(expected: Mapping[str, Any]) -> Path:
    path = Path(str(expected.get("path", ""))).resolve()
    actual = file_fact(path)
    if actual["bytes"] != expected.get("bytes") or actual["sha256"] != expected.get("sha256"):
        raise AP1Error(f"retained artifact drifted: {path}")
    return path


def atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_fact(path)


def atomic_json(path: Path, value: object) -> dict[str, Any]:
    return atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
    )


def atomic_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    array = np.asarray(value)
    return file_fact(path) | {"dtype": array.dtype.str, "shape": list(array.shape)}


def verify_array_fact(expected: Mapping[str, Any]) -> np.ndarray:
    path = validate_fact(expected)
    value = np.load(path, allow_pickle=False)
    if value.dtype.str != expected.get("dtype") or list(value.shape) != expected.get("shape"):
        raise AP1Error(f"retained array geometry drifted: {path}")
    return value


def verify_source_pins() -> None:
    if SOURCE_ARCHIVE.stat().st_size != SOURCE_ARCHIVE_BYTES:
        raise AP1Error("DX2 authority archive byte count differs")
    if sha256_file(SOURCE_ARCHIVE) != SOURCE_ARCHIVE_SHA256:
        raise AP1Error("DX2 authority archive sha256 differs")
    for path, digest in ((GT_SEG, GT_SEG_SHA256), (GT_POSE, GT_POSE_SHA256)):
        if sha256_file(path) != digest:
            raise AP1Error(f"pinned DALI GT drifted: {path}")
    if not ADVISORY_UPSTREAM.is_dir():
        raise AP1Error(f"clean retained upstream snapshot is missing: {ADVISORY_UPSTREAM}")
    from tac.contest_compliance import compute_upstream_snapshot_sha256

    upstream_digest = compute_upstream_snapshot_sha256(
        ADVISORY_UPSTREAM,
        upstream_subdir=".",
        reject_executable_artifacts=True,
    )
    if upstream_digest != UPSTREAM_SNAPSHOT_SHA256:
        raise AP1Error(
            f"clean retained upstream snapshot drifted: {upstream_digest}"
        )


def storage_preflight() -> dict[str, Any]:
    usage = shutil.disk_usage(REPO)
    if usage.free < MIN_FREE_BYTES:
        raise AP1Error(
            f"local retained-payload tier has only {usage.free} free bytes; "
            f"AP1 requires at least {MIN_FREE_BYTES}"
        )
    rows = []
    for path in (REPO, Path("/Volumes/VertigoDataTier/pact"), Path("/Volumes/APDataStore/pact")):
        item = shutil.disk_usage(path)
        rows.append(
            {
                "path": str(path),
                "total_bytes": item.total,
                "used_bytes": item.used,
                "free_bytes": item.free,
            }
        )
    result = {
        "schema": "ddm_ap1_storage_preflight.v1",
        "created_utc": utc_now(),
        "selected_root": str(RECEIPT_ROOT),
        "selection": LOCAL_OPT_IN,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "tiers": rows,
        "auto_cleanup": "none; every materialized raw/archive/scorer field is required evidence",
        "certify_or_block": True,
    }
    atomic_json(RECEIPT_ROOT / "STORAGE_PREFLIGHT.json", result)
    return result


def _load_runtime() -> SimpleNamespace:
    runtime = SOURCE_RUNTIME.resolve()
    # Insert the runtime root first and cpr1 second so cpr1's renderer module
    # wins the top-level ``inflate`` import while ``runtime.*`` remains visible.
    for extra in (str(runtime), str(runtime / "cpr1")):
        if extra in sys.path:
            sys.path.remove(extra)
        sys.path.insert(0, extra)
    stale_prefixes = (
        "runtime",
        "inflate",
        "carrier_codec",
        "ddm_mp2_semantic_receiver",
        "integer_model_io",
        "hpac_integer",
    )
    for name in list(sys.modules):
        if name in stale_prefixes or name.startswith("runtime."):
            del sys.modules[name]
    residual_archive = importlib.import_module("runtime.residual_archive")
    if Path(residual_archive.__file__).resolve().parent.parent != runtime:
        raise AP1Error("imported residual_archive does not belong to the DX2 receiver")
    return SimpleNamespace(
        residual_archive=residual_archive,
        carrier_repack=importlib.import_module("runtime.carrier_repack"),
        coefficient_codec=importlib.import_module("runtime.entropy.coefficient_ar1_codec"),
        coefficient_predictor=importlib.import_module("runtime.entropy.coefficient_predictor"),
        frame0_selector=importlib.import_module("runtime.frame0_selector"),
        rr5=importlib.import_module("runtime.rr5_arith_basis"),
        dx2=importlib.import_module("runtime.dx2_cabac_coefficients"),
        carrier_codec=importlib.import_module("carrier_codec"),
        semantic_receiver=importlib.import_module("ddm_mp2_semantic_receiver"),
        integer_model_io=importlib.import_module("integer_model_io"),
        renderer=importlib.import_module("inflate"),
    )


def _single_member(archive_path: Path) -> tuple[bytes, zipfile.ZipInfo]:
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["p"]:
            raise AP1Error("DX2 archive is not a canonical one-member ZIP")
        info = archive.getinfo("p")
        return archive.read("p"), info


def _emit_zip(member: bytes, info: zipfile.ZipInfo) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        entry = zipfile.ZipInfo("p", date_time=info.date_time)
        entry.compress_type = info.compress_type
        entry.external_attr = info.external_attr
        entry.create_system = info.create_system
        archive.writestr(entry, member)
    return buffer.getvalue()


def _ck2_interleave(body: bytes) -> bytes:
    span = len(body) & ~1
    values = np.frombuffer(body[:span], dtype=np.uint8)
    return values[0::2].tobytes() + values[1::2].tobytes() + body[span:]


def split_outer(modules: SimpleNamespace) -> dict[str, Any]:
    outer, zip_info = _single_member(SOURCE_ARCHIVE)
    header = modules.residual_archive.RX1_MODEL_HEADER
    fields = header.unpack_from(outer)
    magic, version, codec, table_mode, reserved, hpac_bytes, semantic_bytes, carrier_bytes = fields
    offset = header.size
    hpac_stream = outer[offset : offset + hpac_bytes]
    offset += hpac_bytes
    semantic_stream = outer[offset : offset + semantic_bytes]
    offset += semantic_bytes
    carrier_stream = outer[offset : offset + carrier_bytes]
    offset += carrier_bytes
    return {
        "outer": outer,
        "zip_info": zip_info,
        "header": header,
        "magic": magic,
        "version": version,
        "codec": codec,
        "table_mode": table_mode,
        "reserved": reserved,
        "hpac_stream": hpac_stream,
        "semantic_stream": semantic_stream,
        "carrier_stream": carrier_stream,
        "section_tail": outer[offset:],
    }


def build_archive(
    outer: Mapping[str, Any],
    *,
    hpac_stream: bytes,
    semantic_stream: bytes,
    carrier_stream: bytes,
    section_tail: bytes,
) -> bytes:
    member = b"".join(
        (
            outer["header"].pack(
                outer["magic"],
                outer["version"],
                outer["codec"],
                outer["table_mode"],
                outer["reserved"],
                len(hpac_stream),
                len(semantic_stream),
                len(carrier_stream),
            ),
            hpac_stream,
            semantic_stream,
            carrier_stream,
            section_tail,
        )
    )
    return _emit_zip(member, outer["zip_info"])


def signed_lattice(values: np.ndarray, step: int, bits: int) -> np.ndarray:
    if step <= 0 or step & (step - 1):
        raise AP1Error("lattice step must be a positive power of two")
    source = np.asarray(values, dtype=np.int64)
    low = -(1 << (bits - 1))
    high = (1 << (bits - 1)) - 1
    if source.size == 0 or int(source.min()) < low or int(source.max()) > high:
        raise AP1Error(f"source values leave signed-{bits} domain")
    magnitude = ((np.abs(source) + step // 2) // step) * step
    result = np.sign(source) * magnitude
    result = np.clip(result, math.ceil(low / step) * step, math.floor(high / step) * step)
    return result.astype(values.dtype, copy=False)


def pack_depth_nibbles(depths: np.ndarray) -> bytes:
    values = np.asarray(depths, dtype=np.uint8).reshape(-1)
    if values.size == 0 or np.any(values > 15):
        raise AP1Error("IHS1 row depth leaves nibble domain")
    packed = np.zeros((values.size + 1) // 2, dtype=np.uint8)
    packed |= values[0::2]
    packed[: values[1::2].size] |= values[1::2] << 4
    return packed.tobytes()


def unpack_depth_nibbles(raw: bytes, count: int) -> np.ndarray:
    packed = np.frombuffer(raw[: (count + 1) // 2], dtype=np.uint8)
    values = np.empty(packed.size * 2, dtype=np.uint8)
    values[0::2] = packed & 0xF
    values[1::2] = packed >> 4
    return values[:count].copy()


def pack_signed_rows(rows: Sequence[np.ndarray], depths: np.ndarray) -> bytes:
    if len(rows) != len(depths):
        raise AP1Error("IHS1 row/depth census differs")
    bits: list[int] = []
    for row, raw_depth in zip(rows, depths, strict=True):
        depth = int(raw_depth)
        values = np.asarray(row, dtype=np.int64).reshape(-1)
        if depth == 0:
            if np.any(values):
                raise AP1Error("nonzero IHS1 row assigned zero bits")
            continue
        low, high = -(1 << (depth - 1)), (1 << (depth - 1)) - 1
        if values.size and (int(values.min()) < low or int(values.max()) > high):
            raise AP1Error("IHS1 row value leaves selected signed depth")
        unsigned = values & ((1 << depth) - 1)
        for value in unsigned.tolist():
            bits.extend((int(value) >> shift) & 1 for shift in range(depth))
    if not bits:
        return b""
    return np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="little").tobytes()


def _hpac_surface(parts: Any, modules: SimpleNamespace) -> dict[str, Any]:
    import torch

    model = modules.renderer.load_hpac(parts.hpac_blob, torch.device("cpu"))
    compressed_modules = [
        module
        for module in model.modules()
        if isinstance(module, modules.integer_model_io.COMPRESSIBLE_TYPES)
    ]
    rows = [
        row.detach().cpu().numpy().astype(np.int16, copy=False).reshape(-1)
        for module in compressed_modules
        for row in modules.integer_model_io._weight_rows(module, module.weight)
    ]
    count = len(rows)
    depth_bytes = (count + 1) // 2
    depths = unpack_depth_nibbles(parts.hpac_blob[4:], count)
    weight_bits = sum(int(depth) * row.size for depth, row in zip(depths, rows, strict=True))
    weight_bytes = (weight_bits + 7) // 8
    tail = parts.hpac_blob[4 + depth_bytes + weight_bytes :]
    rebuilt = b"IHS1" + pack_depth_nibbles(depths) + pack_signed_rows(rows, depths) + tail
    if rebuilt != parts.hpac_blob:
        raise AP1Error("IHS1 identity encoder does not reproduce the source bytes")
    return {"model": model, "rows": rows, "depths": depths, "tail": tail}


def coarsen_hpac(surface: Mapping[str, Any], level: int) -> tuple[bytes, dict[str, Any]]:
    source_depths = np.asarray(surface["depths"], dtype=np.uint8)
    target_depths = np.maximum(source_depths.astype(np.int16) - level, 0).astype(np.uint8)
    target_rows: list[np.ndarray] = []
    changed_values = 0
    for row, depth in zip(surface["rows"], target_depths, strict=True):
        values = np.asarray(row, dtype=np.int64)
        if int(depth) == 0:
            changed_values += int(np.count_nonzero(values))
            target = np.zeros_like(values, dtype=np.int16)
        else:
            low, high = -(1 << (int(depth) - 1)), (1 << (int(depth) - 1)) - 1
            target = np.clip(values, low, high).astype(np.int16)
            changed_values += int(np.count_nonzero(target != values))
        target_rows.append(target)
    payload = (
        b"IHS1"
        + pack_depth_nibbles(target_depths)
        + pack_signed_rows(target_rows, target_depths)
        + bytes(surface["tail"])
    )
    return payload, {
        "source_depth_histogram": {
            str(int(value)): int(np.count_nonzero(source_depths == value))
            for value in np.unique(source_depths)
        },
        "target_depth_histogram": {
            str(int(value)): int(np.count_nonzero(target_depths == value))
            for value in np.unique(target_depths)
        },
        "changed_values": changed_values,
        "rows": len(target_rows),
    }


def _semantic_surface(parts: Any, modules: SimpleNamespace) -> dict[str, Any]:
    from experiments import ddm_sm3_semantic_representation as sm3

    template = modules.renderer.SemanticTokenRenderer(96).state_dict()
    state = modules.semantic_receiver.unpack_variant_semantic_or_none(parts.semantic_blob, template)
    if state is None or parts.semantic_blob[:6] != b"SM3R\x01\x06":
        raise AP1Error("DX2 semantic object is not the expected SM3R mode-6 payload")
    names = [name for name, value in template.items() if value.ndim >= 2]
    depth_bytes = (len(names) + 1) // 2
    depths = unpack_depth_nibbles(parts.semantic_blob[10 : 10 + depth_bytes], len(names))
    keep_percent = int(parts.semantic_blob[6])
    control, expected, _ = sm3.pack_prune_mixed_candidate(
        state,
        keep_percent,
        OrderedDict(zip(names, depths.tolist(), strict=True)),
    )
    if control != parts.semantic_blob:
        raise AP1Error("SM3R identity encoder does not reproduce the source bytes")
    decoded_control = modules.semantic_receiver.unpack_variant_semantic_or_none(control, template)
    for name in state:
        if not np.array_equal(state[name].cpu().numpy(), decoded_control[name].cpu().numpy()):
            raise AP1Error(f"SM3R identity decode differs for {name}")
    return {
        "template": template,
        "state": state,
        "names": names,
        "depths": depths,
        "keep_percent": keep_percent,
    }


def coarsen_semantic(
    surface: Mapping[str, Any], level: int
) -> tuple[bytes, Mapping[str, Any], dict[str, Any]]:
    from experiments import ddm_sm3_semantic_representation as sm3

    source = np.asarray(surface["depths"], dtype=np.uint8)
    if level == 1:
        target = np.where(source == 4, 3, source)
    elif level == 2:
        target = np.where(source == 4, 2, source)
    elif level == 3:
        target = np.full_like(source, 2)
    else:
        raise AP1Error("semantic level must be 1, 2, or 3")
    allocation = OrderedDict(zip(surface["names"], target.tolist(), strict=True))
    payload, expected, metadata = sm3.pack_prune_mixed_candidate(
        surface["state"], int(surface["keep_percent"]), allocation
    )
    return payload, expected, {
        "source_depths": dict(zip(surface["names"], source.tolist(), strict=True)),
        "target_depths": dict(allocation),
        "packer": metadata,
    }


def _carrier_surface(
    parts: Any, modules: SimpleNamespace, outer: Mapping[str, Any]
) -> dict[str, Any]:
    carrier, selector = modules.carrier_repack.split_frame0_selector_carrier(parts.carrier_blob)
    if selector is None:
        raise AP1Error("DX2 carrier has no frame-0 selector")
    if not carrier.startswith(modules.residual_archive.CAP1_PREFIX):
        raise AP1Error("DX2 carrier is not the expected CAP1 object")
    if int(outer["reserved"]) & int(modules.residual_archive.CK2_RESERVED_CARRIER_PLANE2):
        raise AP1Error("DX2 carrier unexpectedly uses the CK2 carrier transform")
    canonical = modules.carrier_repack.materialize_cpr1(carrier, modules.renderer)
    basis_scales, basis_codes, coefficient_scales, encoded = (
        modules.carrier_codec.decode_compact_carrier(
            canonical,
            basis_count=12 * 3 * 24 * 32,
            frames=N,
            dimensions=12,
        )
    )
    delta = (encoded.astype(np.int64) >> 1) ^ -(encoded.astype(np.int64) & 1)
    unsigned = np.cumsum(delta, axis=0) & 0xFFF
    codes = np.where(unsigned >= 0x800, unsigned - 0x1000, unsigned).astype(np.int32)
    return {
        "canonical": canonical,
        "basis_scales": basis_scales,
        "basis_codes": basis_codes,
        "coefficient_scales": coefficient_scales,
        "codes": codes,
        "selector": selector,
        "source_cap1": carrier,
        "source_predictor_metadata": carrier[14:50],
    }


def delta_zigzag(codes: np.ndarray) -> np.ndarray:
    values = np.asarray(codes, dtype=np.int64)
    if values.shape != (N, 12) or np.any(values < -2048) or np.any(values > 2047):
        raise AP1Error("carrier coefficient lattice differs")
    unsigned = values & 0xFFF
    previous = np.zeros_like(unsigned)
    previous[1:] = unsigned[:-1]
    delta_unsigned = (unsigned - previous) & 0xFFF
    delta = np.where(delta_unsigned >= 0x800, delta_unsigned - 0x1000, delta_unsigned)
    return (((delta << 1) ^ (delta >> 63)) & 0xFFF).astype(np.int32)


def encode_cap1_with_source_predictor(
    cpr1: bytes,
    codes: np.ndarray,
    source_predictor_metadata: bytes,
    modules: SimpleNamespace,
) -> bytes:
    """Encode CAP1 while freezing the shipped predictor model byte-for-byte.

    The predictor metadata is part of the fixed coder, not the carrier lattice
    parameter group under purchase.  Freezing it makes the level-zero encoder
    reproduce the shipped CAP1 exactly and prevents a predictor refit from
    contaminating the measured byte credit.
    """

    if len(cpr1) < 152 or cpr1[:4] != b"CPR1":
        raise AP1Error("canonical CPR1 framing differs")
    if len(source_predictor_metadata) != 36:
        raise AP1Error("source CAP1 predictor metadata differs")
    values = np.asarray(codes, dtype=np.int32)
    if values.shape != (N, 12):
        raise AP1Error("carrier coefficient lattice differs")
    model = modules.coefficient_predictor.unpack_ar1_bias_metadata(
        source_predictor_metadata, 12
    )
    residuals = np.empty_like(values)
    residuals[0] = values[0]
    for frame in range(1, N):
        prediction = modules.coefficient_predictor.signed_mod(
            modules.coefficient_predictor.round_q8(
                values[frame - 1], model.factors_q8
            )
            + model.biases
        )
        residuals[frame] = modules.coefficient_predictor.signed_mod(
            values[frame] - prediction
        )
    zigzag = (
        (residuals.astype(np.int64) << 1)
        ^ (residuals.astype(np.int64) >> 63)
    ) & 0xFFF
    ks, rice_payload, residual_bits = modules.carrier_repack._rice_encode(zigzag, 1)
    magic, basis_bits, _ = struct.unpack_from("<4sII", cpr1)
    if magic != b"CPR1" or basis_bits <= 0:
        raise AP1Error("canonical CPR1 header differs")
    basis_bytes = (basis_bits + 7) // 8
    fixed_end = 12 + 8 * 12 + 32 + 12
    cap1 = (
        b"CAP1"
        + bytes((1, 0, 0, 0))
        + int(basis_bits).to_bytes(3, "little")
        + int(residual_bits).to_bytes(3, "little")
        + bytes(source_predictor_metadata)
        + cpr1[12 : 12 + 8 * 12]
        + cpr1[12 + 8 * 12 : 12 + 8 * 12 + 32]
        + ks.reshape(-1).tobytes()
        + cpr1[fixed_end : fixed_end + basis_bytes]
        + rice_payload
    )
    if modules.coefficient_codec.decode_cap1(cap1, frames=N, dimensions=12) != cpr1:
        raise AP1Error("fixed-predictor CAP1 round-trip differs")
    return cap1


def encode_carrier(
    surface: Mapping[str, Any],
    basis_codes: np.ndarray,
    codes: np.ndarray,
    modules: SimpleNamespace,
) -> tuple[bytes, bytes]:
    from experiments import ddm_jo2_receiver_close as jo2
    from tac.pr130_lift.pose.lifted.carrier_codec import encode_compact_carrier

    cpr1 = encode_compact_carrier(
        surface["basis_scales"],
        basis_codes,
        surface["coefficient_scales"],
        delta_zigzag(codes),
    )
    cap1 = encode_cap1_with_source_predictor(
        cpr1,
        np.asarray(codes, dtype=np.int32),
        bytes(surface["source_predictor_metadata"]),
        modules,
    )
    stripped = cap1[len(modules.residual_archive.CAP1_PREFIX) :]
    body_bytes = modules.residual_archive._cap1_body_bytes(stripped)
    bit_counts, predictor = stripped[:6], stripped[6:42]
    scales, lengths = stripped[42:138], stripped[138:170]
    ks, rest = stripped[170:182], stripped[182:body_bytes]
    canonical_stored = (
        bit_counts
        + scales
        + predictor
        + lengths
        + ks
        + rest
        + bytes(surface["selector"])[5:]
    )
    packed = jo2._pack_cap1_metadata(canonical_stored, modules)
    # The receiver restores RR5 first and DX2 second, so the exact encoder
    # inverse applies DX2 first and RR5 second.
    dx2 = modules.dx2.apply_cabac_to_carrier_body(packed)
    rr5 = modules.rr5.apply_rider_to_carrier_body(dx2["body"])
    if modules.rr5.restore_carrier_body(rr5["body"]) != dx2["body"]:
        raise AP1Error("RR5 inverse closure differs")
    if modules.dx2.restore_carrier_body(dx2["body"]) != packed:
        raise AP1Error("DX2 inverse closure differs")
    return cpr1, bytes(rr5["body"])


def coarsen_carrier(
    surface: Mapping[str, Any], level: int, modules: SimpleNamespace
) -> tuple[bytes, bytes, dict[str, Any]]:
    step = 1 << level
    basis = signed_lattice(np.asarray(surface["basis_codes"]), step, 5)
    codes = signed_lattice(np.asarray(surface["codes"]), step, 12)
    cpr1, physical = encode_carrier(surface, basis, codes, modules)
    return cpr1, physical, {
        "step": step,
        "basis_values": int(basis.size),
        "basis_values_changed": int(np.count_nonzero(basis != surface["basis_codes"])),
        "coefficient_values": int(codes.size),
        "coefficient_values_changed": int(np.count_nonzero(codes != surface["codes"])),
    }


def pack_signed_fixed(values: np.ndarray, bits: int) -> bytes:
    source = np.asarray(values, dtype=np.int64).reshape(-1)
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if source.size == 0 or int(source.min()) < low or int(source.max()) > high:
        raise AP1Error("fixed signed values leave selected domain")
    unsigned = source & ((1 << bits) - 1)
    bit_values = np.empty(source.size * bits, dtype=np.uint8)
    for shift in range(bits):
        bit_values[shift::bits] = ((unsigned >> shift) & 1).astype(np.uint8)
    return np.packbits(bit_values, bitorder="little").tobytes()


def coarsen_residual(parts: Any, level: int) -> tuple[bytes, dict[str, Any]]:
    step = 1 << level
    codes = signed_lattice(parts.table.codes, step, 6).astype(np.int8)
    payload = (
        b"RCF1"
        + np.asarray(parts.table.scale, dtype="<f2").tobytes()
        + pack_signed_fixed(codes, 6)
    )
    return payload, {
        "step": step,
        "codes": int(codes.size),
        "codes_changed": int(np.count_nonzero(codes != parts.table.codes)),
        "scale_fp16_unchanged": True,
    }


def _physical_streams_for_candidate(
    spec: CandidateSpec,
    *,
    modules: SimpleNamespace,
    parts: Any,
    outer: Mapping[str, Any],
    semantic_surface: Mapping[str, Any],
    carrier_surface: Mapping[str, Any],
    hpac_surface: Mapping[str, Any],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    streams = {
        "hpac": bytes(outer["hpac_stream"]),
        "semantic": bytes(outer["semantic_stream"]),
        "carrier": bytes(outer["carrier_stream"]),
        "tail": bytes(outer["section_tail"]),
    }
    detail: dict[str, Any] = {"group": spec.group, "level": spec.level, "quantum": spec.quantum}
    if spec.group == "control":
        detail["target_payload"] = parts.semantic_blob
    elif spec.group == "semantic":
        payload, expected, metadata = coarsen_semantic(semantic_surface, spec.level)
        transformed = _ck2_interleave(payload)
        stream = brotli.compress(transformed, quality=11, lgwin=24)
        if brotli.decompress(stream) != transformed:
            raise AP1Error("semantic Brotli inverse differs")
        streams["semantic"] = stream
        detail.update({"target_payload": payload, "expected_semantic_state": expected, **metadata})
    elif spec.group == "carrier":
        canonical, physical, metadata = coarsen_carrier(carrier_surface, spec.level, modules)
        stream = brotli.compress(physical, quality=9, lgwin=16)
        if brotli.decompress(stream) != physical:
            raise AP1Error("carrier Brotli inverse differs")
        streams["carrier"] = stream
        detail.update({"target_payload": physical, "expected_cpr1": canonical, **metadata})
    elif spec.group == "hpac":
        payload, metadata = coarsen_hpac(hpac_surface, spec.level)
        stream = brotli.compress(payload, quality=10, lgwin=24)
        if brotli.decompress(stream) != payload:
            raise AP1Error("HPAC Brotli inverse differs")
        streams["hpac"] = stream
        detail.update({"target_payload": payload, **metadata})
    elif spec.group == "residual":
        payload, metadata = coarsen_residual(parts, spec.level)
        streams["tail"] = payload[4:] + parts.token_stream
        detail.update({"target_payload": payload, **metadata})
    else:
        raise AP1Error(f"unknown candidate group: {spec.group}")
    return streams, detail


def _copy_runtime(destination: Path, archive_record: Mapping[str, Any]) -> dict[str, Any]:
    if destination.exists():
        raise AP1Error(
            f"partial candidate runtime is retained and will not be overwritten: {destination}"
        )

    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name.startswith("._") or name in {"__pycache__", ".DS_Store"}
        }

    shutil.copytree(SOURCE_RUNTIME, destination, ignore=ignore)
    shutil.copy2(validate_fact(archive_record), destination / "archive.zip")
    inflate_path = destination / "inflate.py"
    source = inflate_path.read_text(encoding="utf-8")
    sha_matches = re.findall(r'^ARCHIVE_SHA256 = "([0-9a-f]{64})"$', source, flags=re.MULTILINE)
    byte_matches = re.findall(r"^ARCHIVE_BYTES = ([0-9_]+)$", source, flags=re.MULTILINE)
    if sha_matches != [SOURCE_ARCHIVE_SHA256] or byte_matches != [f"{SOURCE_ARCHIVE_BYTES:_}"]:
        raise AP1Error("receiver archive pins differ before candidate patch")
    source = source.replace(SOURCE_ARCHIVE_SHA256, str(archive_record["sha256"]), 1)
    source = source.replace(
        f"ARCHIVE_BYTES = {SOURCE_ARCHIVE_BYTES:_}",
        f"ARCHIVE_BYTES = {int(archive_record['bytes']):_}",
        1,
    )
    atomic_bytes(inflate_path, source.encode())
    return {
        "runtime_dir": str(destination.resolve()),
        "source_runtime_tree": tree_fact(SOURCE_RUNTIME),
        "candidate_runtime_tree": tree_fact(destination),
        "inflate_py": file_fact(inflate_path),
        "inflate_sh": file_fact(destination / "inflate.sh"),
        "archive": file_fact(destination / "archive.zip"),
    }


def _verify_candidate(
    spec: CandidateSpec,
    archive_path: Path,
    *,
    modules: SimpleNamespace,
    source_parts: Any,
    detail: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = modules.residual_archive.read_residual_archive(archive_path)
    fields = {
        "semantic": (candidate.semantic_blob, source_parts.semantic_blob),
        "carrier": (candidate.carrier_blob, source_parts.carrier_blob),
        "hpac": (candidate.hpac_blob, source_parts.hpac_blob),
        "residual": (candidate.residual_payload, source_parts.residual_payload),
        "token": (candidate.token_stream, source_parts.token_stream),
    }
    non_target = []
    for name, (actual, expected) in fields.items():
        if name not in {spec.group, "token"} and actual != expected:
            raise AP1Error(f"{spec.candidate_id} changed non-target {name} bytes")
        if name != spec.group:
            non_target.append({"field": name, "byte_identical": actual == expected, "sha256": sha256_bytes(actual)})
    if candidate.token_stream != source_parts.token_stream:
        raise AP1Error(f"{spec.candidate_id} changed the fixed token stream")
    target_verified: dict[str, Any]
    if spec.group == "control":
        if archive_path.read_bytes() != SOURCE_ARCHIVE.read_bytes():
            raise AP1Error("control archive is not byte-identical to DX2")
        target_verified = {"archive_byte_identical": True}
    elif spec.group == "semantic":
        template = modules.renderer.SemanticTokenRenderer(96).state_dict()
        decoded = modules.semantic_receiver.unpack_variant_semantic_or_none(candidate.semantic_blob, template)
        expected = detail["expected_semantic_state"]
        for name in expected:
            if not np.array_equal(expected[name].cpu().numpy(), decoded[name].cpu().numpy()):
                raise AP1Error(f"semantic parse-back differs for {name}")
        target_verified = {"decoded_tensor_count": len(decoded)}
    elif spec.group == "carrier":
        carrier, _ = modules.carrier_repack.split_frame0_selector_carrier(candidate.carrier_blob)
        decoded = modules.carrier_repack.materialize_cpr1(carrier, modules.renderer)
        if decoded != detail["expected_cpr1"]:
            raise AP1Error("carrier parse-back differs from encoded CPR1")
        target_verified = {"cpr1_sha256": sha256_bytes(decoded), "cpr1_bytes": len(decoded)}
    elif spec.group == "hpac":
        import torch

        model = modules.renderer.load_hpac(candidate.hpac_blob, torch.device("cpu"))
        target_verified = {
            "model_parameter_count": sum(parameter.numel() for parameter in model.parameters())
        }
    elif spec.group == "residual":
        if candidate.residual_payload != detail["target_payload"]:
            raise AP1Error("fixed residual parse-back differs")
        target_verified = {"codes": candidate.table.codes.tolist(), "scale": candidate.table.scale}
    else:
        raise AP1Error("unsupported candidate target verification")
    return {
        "receiver_parse_back": True,
        "token_stream": {
            "bytes": len(candidate.token_stream),
            "sha256": sha256_bytes(candidate.token_stream),
            "matches_pin": len(candidate.token_stream) == SOURCE_TOKEN_BYTES
            and sha256_bytes(candidate.token_stream) == SOURCE_TOKEN_SHA256,
        },
        "non_target_fields": non_target,
        "target": target_verified,
    }


def materialize_one(
    spec: CandidateSpec,
    *,
    modules: SimpleNamespace,
    parts: Any,
    outer: Mapping[str, Any],
    semantic_surface: Mapping[str, Any],
    carrier_surface: Mapping[str, Any],
    hpac_surface: Mapping[str, Any],
) -> dict[str, Any]:
    root = RECEIPT_ROOT / "retained/candidates" / spec.candidate_id
    result_path = root / "MATERIALIZE_RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for key in ("archive", "archive_repeat", "target_payload"):
            validate_fact(result[key])
        if (
            result["archive"]["bytes"] != result["archive_repeat"]["bytes"]
            or result["archive"]["sha256"] != result["archive_repeat"]["sha256"]
        ):
            raise AP1Error(f"retained repeat archive drifted for {spec.candidate_id}")
        if "source_runtime_tree" not in result["runtime"]:
            result["runtime"]["source_runtime_tree"] = tree_fact(SOURCE_RUNTIME)
            result["runtime"]["candidate_runtime_tree"] = tree_fact(root / "runtime")
            atomic_json(result_path, result)
        return result

    streams, detail = _physical_streams_for_candidate(
        spec,
        modules=modules,
        parts=parts,
        outer=outer,
        semantic_surface=semantic_surface,
        carrier_surface=carrier_surface,
        hpac_surface=hpac_surface,
    )
    archive = build_archive(
        outer,
        hpac_stream=streams["hpac"],
        semantic_stream=streams["semantic"],
        carrier_stream=streams["carrier"],
        section_tail=streams["tail"],
    )
    repeat = build_archive(
        outer,
        hpac_stream=streams["hpac"],
        semantic_stream=streams["semantic"],
        carrier_stream=streams["carrier"],
        section_tail=streams["tail"],
    )
    if archive != repeat:
        raise AP1Error(f"archive repeat differs for {spec.candidate_id}")
    archive_record = atomic_bytes(root / "archive.zip", archive)
    repeat_record = atomic_bytes(root / "archive.repeat.zip", repeat)
    target_record = atomic_bytes(root / "payloads/target_payload.bin", detail["target_payload"])
    physical_records = {
        name: atomic_bytes(root / f"payloads/{name}.bin", payload)
        for name, payload in streams.items()
    }
    verification = _verify_candidate(
        spec,
        Path(archive_record["path"]),
        modules=modules,
        source_parts=parts,
        detail=detail,
    )
    runtime = _copy_runtime(root / "runtime", archive_record)
    serializable_detail = {key: value for key, value in detail.items() if key not in {"target_payload", "expected_semantic_state", "expected_cpr1"}}
    result = {
        "schema": "ddm_ap1_materialized_candidate.v1",
        "created_utc": utc_now(),
        "candidate_id": spec.candidate_id,
        "group": spec.group,
        "level": spec.level,
        "quantum": spec.quantum,
        "source_archive": file_fact(SOURCE_ARCHIVE),
        "archive": archive_record,
        "archive_repeat": repeat_record,
        "archive_delta_bytes": len(archive) - SOURCE_ARCHIVE_BYTES,
        "byte_credit": SOURCE_ARCHIVE_BYTES - len(archive),
        "target_payload": target_record,
        "physical_streams": physical_records,
        "physical_bytes_held": {
            "semantic": len(streams["semantic"]),
            "carrier": len(streams["carrier"]),
            "hpac": len(streams["hpac"]),
            "residual_and_token_tail": len(streams["tail"]),
            "fixed_residual": len(parts.residual_payload) - 4,
            "token_stream": len(parts.token_stream),
        },
        "detail": serializable_detail,
        "verification": verification,
        "runtime": runtime,
        "storage": LOCAL_OPT_IN,
        "retention": "FULL_BYTES",
        "score_claim": False,
    }
    atomic_json(result_path, result)
    return result


def materialize_all() -> dict[str, Any]:
    verify_source_pins()
    RECEIPT_ROOT.mkdir(parents=True, exist_ok=True)
    storage = storage_preflight()
    modules = _load_runtime()
    parts = modules.residual_archive.read_residual_archive(SOURCE_ARCHIVE)
    if len(parts.token_stream) != SOURCE_TOKEN_BYTES or sha256_bytes(parts.token_stream) != SOURCE_TOKEN_SHA256:
        raise AP1Error("DX2 token stream pin differs")
    outer = split_outer(modules)
    source_census = {
        "semantic_renderer": len(outer["semantic_stream"]),
        "carrier": len(outer["carrier_stream"]),
        "hpac_probability_model": len(outer["hpac_stream"]),
        "fixed_residual_table": len(parts.residual_payload) - 4,
    }
    source_census["zip_and_rx1_structural_framing"] = (
        SOURCE_ARCHIVE_BYTES - SOURCE_TOKEN_BYTES - sum(source_census.values())
    )
    source_census["total_residue"] = sum(source_census.values())
    if source_census != {
        "semantic_renderer": 30_856,
        "carrier": 22_010,
        "hpac_probability_model": 13_515,
        "fixed_residual_table": 96,
        "zip_and_rx1_structural_framing": 114,
        "total_residue": 66_591,
    }:
        raise AP1Error(f"AR1B source census drifted: {source_census}")
    semantic_surface = _semantic_surface(parts, modules)
    carrier_surface = _carrier_surface(parts, modules, outer)
    hpac_surface = _hpac_surface(parts, modules)
    carrier_cpr1, carrier_physical = encode_carrier(
        carrier_surface,
        np.asarray(carrier_surface["basis_codes"]),
        np.asarray(carrier_surface["codes"]),
        modules,
    )
    identity_controls = {
        "semantic_stream": (
            brotli.compress(_ck2_interleave(parts.semantic_blob), quality=11, lgwin=24)
            == outer["semantic_stream"]
        ),
        "carrier_cpr1": carrier_cpr1 == carrier_surface["canonical"],
        "carrier_cap1_fixed_predictor": (
            encode_cap1_with_source_predictor(
                carrier_surface["canonical"],
                np.asarray(carrier_surface["codes"], dtype=np.int32),
                bytes(carrier_surface["source_predictor_metadata"]),
                modules,
            )
            == carrier_surface["source_cap1"]
        ),
        "carrier_stream": (
            brotli.compress(carrier_physical, quality=9, lgwin=16)
            == outer["carrier_stream"]
        ),
        "hpac_stream": (
            brotli.compress(parts.hpac_blob, quality=10, lgwin=24)
            == outer["hpac_stream"]
        ),
        "fixed_residual_and_token_tail": (
            parts.residual_payload[4:] + parts.token_stream == outer["section_tail"]
        ),
    }
    if not all(identity_controls.values()):
        raise AP1Error(f"one or more incumbent encoders failed identity: {identity_controls}")
    results = []
    for spec in SPECS:
        print(f"materialize {spec.candidate_id}", flush=True)
        results.append(
            materialize_one(
                spec,
                modules=modules,
                parts=parts,
                outer=outer,
                semantic_surface=semantic_surface,
                carrier_surface=carrier_surface,
                hpac_surface=hpac_surface,
            )
        )
    result = {
        "schema": "ddm_ap1_materialize_all.v1",
        "created_utc": utc_now(),
        "status": "COMPLETE",
        "storage_preflight": storage,
        "source_census": source_census | {"unexplained_remainder": 0},
        "identity_controls": identity_controls,
        "candidate_count": len(results),
        "candidates": [
            {
                "candidate_id": item["candidate_id"],
                "group": item["group"],
                "level": item["level"],
                "archive": item["archive"],
                "byte_credit": item["byte_credit"],
                "runtime_dir": item["runtime"]["runtime_dir"],
            }
            for item in results
        ],
        "next_stage": {
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "ddm_ap1_residue_purchase_scorer",
            "consumer_store": str(RECEIPT_ROOT / "advisory"),
            "fire_trigger": "materialize_all status COMPLETE and exclusive n600 lane remains held",
            "command_template": (
                ".venv/bin/python tools/fire_local_advisory.py --runtime-dir <runtime_dir> "
                "--attempt-dir <receipt_root>/advisory/<candidate_id> --label ddm_ap1_<candidate_id>"
            ),
        },
    }
    atomic_json(RECEIPT_ROOT / "MATERIALIZE_ALL.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _chunk_statistics(
    gt_seg: np.ndarray,
    candidate_seg: np.ndarray,
    gt_pose: np.ndarray,
    candidate_pose: np.ndarray,
) -> dict[str, Any]:
    if gt_seg.shape != candidate_seg.shape or gt_seg.ndim != 3:
        raise AP1Error("SegNet argmax chunk geometry differs")
    if gt_pose.shape != candidate_pose.shape or gt_pose.shape != (gt_seg.shape[0], 6):
        raise AP1Error("PoseNet chunk geometry differs")
    if np.any(gt_seg >= len(CLASS_NAMES)) or np.any(candidate_seg >= len(CLASS_NAMES)):
        raise AP1Error("SegNet argmax contains an invalid class")
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (gt_seg.reshape(-1), candidate_seg.reshape(-1)), 1)
    delta = candidate_pose.astype(np.float64) - gt_pose.astype(np.float64)
    return {
        "pairs": int(gt_seg.shape[0]),
        "pixels": int(gt_seg.size),
        "flips": int(gt_seg.size - np.trace(confusion)),
        "confusion_gt_rows_candidate_columns": confusion.tolist(),
        "pose_squared_error_sum": float(np.square(delta).sum(dtype=np.float64)),
    }


def _score_chunk(
    *,
    chunk_index: int,
    start: int,
    stop: int,
    batch_candidate: Any,
    distortion_net: Any,
    gt_seg: np.ndarray,
    gt_pose: np.ndarray,
    out_dir: Path,
) -> dict[str, Any]:
    receipt_path = out_dir / "chunks" / f"{start:04d}_{stop - 1:04d}.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        candidate_seg = verify_array_fact(receipt["candidate_argmax"])
        candidate_pose = verify_array_fact(receipt["candidate_pose6"])
        statistics = _chunk_statistics(
            np.asarray(gt_seg[start:stop]), candidate_seg, np.asarray(gt_pose[start:stop]), candidate_pose
        )
        if statistics != receipt.get("statistics"):
            raise AP1Error(f"retained score chunk drifted: {receipt_path}")
        return receipt

    import torch

    with torch.inference_mode():
        pose_outputs, seg_logits = distortion_net(batch_candidate.to("cpu"))
        candidate_seg = seg_logits.argmax(dim=1).to(dtype=torch.uint8).cpu().numpy()
        candidate_pose = (
            pose_outputs["pose"][..., :6]
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
            .reshape(stop - start, 6)
        )
    seg_record = atomic_npy(
        out_dir / "chunks" / f"{start:04d}_{stop - 1:04d}.candidate_argmax.uint8.npy",
        candidate_seg.astype(np.uint8, copy=False),
    )
    pose_record = atomic_npy(
        out_dir / "chunks" / f"{start:04d}_{stop - 1:04d}.candidate_pose6.float32.npy",
        candidate_pose.astype(np.float32, copy=False),
    )
    statistics = _chunk_statistics(
        np.asarray(gt_seg[start:stop]), candidate_seg, np.asarray(gt_pose[start:stop]), candidate_pose
    )
    receipt = {
        "schema": "ddm_ap1_dali_score_chunk.v1",
        "chunk_index": chunk_index,
        "pair_start": start,
        "pair_stop_exclusive": stop,
        "candidate_argmax": seg_record,
        "candidate_pose6": pose_record,
        "statistics": statistics,
        "checkpoint_complete": True,
        "retention": "FULL_SCORER_OUTPUT_BYTES",
    }
    atomic_json(receipt_path, receipt)
    return receipt


def _aggregate_chunks(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    pairs = pixels = flips = 0
    pose_sse = 0.0
    for receipt in receipts:
        stats = receipt["statistics"]
        pairs += int(stats["pairs"])
        pixels += int(stats["pixels"])
        flips += int(stats["flips"])
        pose_sse += float(stats["pose_squared_error_sum"])
        confusion += np.asarray(stats["confusion_gt_rows_candidate_columns"], dtype=np.int64)
    if pairs != N or pixels != SEG_PIXELS or int(confusion.sum()) != SEG_PIXELS:
        raise AP1Error("n600 scorer coverage differs")
    if flips != SEG_PIXELS - int(np.trace(confusion)):
        raise AP1Error("segmentation flip reduction differs")
    class_denominators = tuple(int(value) for value in confusion.sum(axis=1))
    if class_denominators != CLASS_DENOMINATORS:
        raise AP1Error(
            f"DALI GT class census differs: {class_denominators} != {CLASS_DENOMINATORS}"
        )
    rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        denominator = int(confusion[class_id].sum())
        class_flips = denominator - int(confusion[class_id, class_id])
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt_pixels": denominator,
                "flips_from_gt_class": class_flips,
                "conditional_d_seg": class_flips / denominator,
                "contribution_to_total_d_seg": class_flips / SEG_PIXELS,
            }
        )
    return {
        "pairs": pairs,
        "seg_denominator": SEG_PIXELS,
        "pose_denominator": POSE_VALUES,
        "flips": flips,
        "d_seg": flips / SEG_PIXELS,
        "pose_squared_error_sum": pose_sse,
        "d_pose": pose_sse / POSE_VALUES,
        "confusion_gt_rows_candidate_columns": confusion.tolist(),
        "per_class": rows,
        "lane_class_1": rows[1],
    }


def score_candidate(candidate_id: str, raw_path: Path, batch_pairs: int, torch_threads: int) -> dict[str, Any]:
    if candidate_id not in SPEC_BY_ID:
        raise AP1Error(f"unknown candidate id: {candidate_id}")
    if not 1 <= batch_pairs <= 120:
        raise AP1Error("--batch-pairs must be in [1, 120]")
    verify_source_pins()
    candidate_root = RECEIPT_ROOT / "retained/candidates" / candidate_id
    materialized_path = candidate_root / "MATERIALIZE_RESULT.json"
    materialized = json.loads(materialized_path.read_text(encoding="utf-8"))
    validate_fact(materialized["archive"])
    raw_path = raw_path.resolve()
    if raw_path.stat().st_size != RAW_BYTES:
        raise AP1Error(f"retained candidate raw has {raw_path.stat().st_size} bytes, expected {RAW_BYTES}")
    out_dir = RECEIPT_ROOT / "scorer" / candidate_id
    result_path = out_dir / "RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        validate_fact(result["candidate_raw"])
        verify_array_fact(result["candidate_argmax_n600"])
        verify_array_fact(result["candidate_pose6_n600"])
        receipts = []
        for record in result["chunk_receipts"]:
            receipt_path = validate_fact(record)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            verify_array_fact(receipt["candidate_argmax"])
            verify_array_fact(receipt["candidate_pose6"])
            receipts.append(receipt)
        if _aggregate_chunks(receipts) != result["summary"]:
            raise AP1Error(f"retained scorer reduction drifted for {candidate_id}")
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    if str(ADVISORY_UPSTREAM) not in sys.path:
        sys.path.insert(0, str(ADVISORY_UPSTREAM))
    import torch
    from frame_utils import TensorVideoDataset
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    torch.manual_seed(12_341)
    torch.use_deterministic_algorithms(True)
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
    distortion_net = DistortionNet().eval().to("cpu")
    distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    names = [line.strip() for line in VIDEO_NAMES.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != 1:
        raise AP1Error("AP1 expects exactly one public video")
    submission_dir = raw_path.parent.parent
    expected_raw = submission_dir / "inflated/0.raw"
    if raw_path != expected_raw:
        raise AP1Error(f"raw must be retained at advisory submission path {expected_raw}")
    dataset = TensorVideoDataset(
        names,
        data_dir=submission_dir / "inflated",
        batch_size=batch_pairs,
        device=torch.device("cpu"),
        num_threads=2,
        seed=1234,
    )
    dataset.prepare_data()
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)
    gt_seg = np.load(GT_SEG, allow_pickle=False, mmap_mode="r")
    gt_pose = np.load(GT_POSE, allow_pickle=False, mmap_mode="r")
    if gt_seg.shape != (N, SEG_H, SEG_W) or gt_seg.dtype != np.uint8:
        raise AP1Error("pinned DALI segmentation GT geometry differs")
    if gt_pose.shape != (N, 6) or gt_pose.dtype != np.float32:
        raise AP1Error("pinned DALI pose GT geometry differs")
    receipts = []
    start = 0
    for chunk_index, (_, _, batch_candidate) in enumerate(loader):
        stop = start + int(batch_candidate.shape[0])
        receipts.append(
            _score_chunk(
                chunk_index=chunk_index,
                start=start,
                stop=stop,
                batch_candidate=batch_candidate,
                distortion_net=distortion_net,
                gt_seg=gt_seg,
                gt_pose=gt_pose,
                out_dir=out_dir,
            )
        )
        print(f"score {candidate_id}: {stop}/{N}", flush=True)
        start = stop
    if start != N:
        raise AP1Error("candidate scorer did not cover n600")
    summary = _aggregate_chunks(receipts)
    full_seg = np.concatenate(
        [verify_array_fact(receipt["candidate_argmax"]) for receipt in receipts], axis=0
    )
    full_pose = np.concatenate(
        [verify_array_fact(receipt["candidate_pose6"]) for receipt in receipts], axis=0
    )
    full_seg_record = atomic_npy(out_dir / "candidate_argmax_n600.uint8.npy", full_seg)
    full_pose_record = atomic_npy(out_dir / "candidate_pose6_n600.float32.npy", full_pose)
    result = {
        "schema": "ddm_ap1_dali_score.v1",
        "created_utc": utc_now(),
        "status": "COMPLETE",
        "candidate_id": candidate_id,
        "group": SPEC_BY_ID[candidate_id].group,
        "level": SPEC_BY_ID[candidate_id].level,
        "axis": AXIS,
        "promotable": False,
        "score_claim": False,
        "gt_lineage": {
            "seg_argmax": file_fact(GT_SEG),
            "pose_first6": file_fact(GT_POSE),
            "description": "contest-CUDA DALI GT tables; candidate scored by frozen CPU-torch models",
        },
        "candidate_raw": file_fact(raw_path),
        "candidate_archive": materialized["archive"],
        "candidate_argmax_n600": full_seg_record,
        "candidate_pose6_n600": full_pose_record,
        "segnet_weights": file_fact(Path(segnet_sd_path)),
        "posenet_weights": file_fact(Path(posenet_sd_path)),
        "batch_pairs": batch_pairs,
        "torch_threads": torch.get_num_threads(),
        "chunk_receipts": [
            file_fact(out_dir / "chunks" / f"{receipt['pair_start']:04d}_{receipt['pair_stop_exclusive'] - 1:04d}.json")
            for receipt in receipts
        ],
        "summary": summary,
        "retention": "raw + every chunk argmax/pose + concatenated n600 argmax/pose",
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _class_delta(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for base, item in zip(control["per_class"], candidate["per_class"], strict=True):
        if base["class_id"] != item["class_id"] or base["gt_pixels"] != item["gt_pixels"]:
            raise AP1Error("per-class scorer census differs across candidates")
        rows.append(
            {
                "class_id": item["class_id"],
                "class_name": item["class_name"],
                "denominator": item["gt_pixels"],
                "control_flips": base["flips_from_gt_class"],
                "candidate_flips": item["flips_from_gt_class"],
                "delta_flips": item["flips_from_gt_class"] - base["flips_from_gt_class"],
                "delta_conditional_d_seg": item["conditional_d_seg"] - base["conditional_d_seg"],
                "delta_contribution_to_total_d_seg": (
                    item["contribution_to_total_d_seg"] - base["contribution_to_total_d_seg"]
                ),
            }
        )
    return rows


def aggregate_all() -> dict[str, Any]:
    rows = []
    materialized_by_id: dict[str, Any] = {}
    scored_by_id: dict[str, Any] = {}
    for spec in SPECS:
        materialized_path = RECEIPT_ROOT / "retained/candidates" / spec.candidate_id / "MATERIALIZE_RESULT.json"
        score_path = RECEIPT_ROOT / "scorer" / spec.candidate_id / "RESULT.json"
        if not materialized_path.is_file() or not score_path.is_file():
            raise AP1Error(f"candidate is incomplete: {spec.candidate_id}")
        materialized_by_id[spec.candidate_id] = json.loads(materialized_path.read_text(encoding="utf-8"))
        scored_by_id[spec.candidate_id] = json.loads(score_path.read_text(encoding="utf-8"))
    control = scored_by_id["control"]["summary"]
    control_archive = materialized_by_id["control"]["archive"]["bytes"]
    control_score = (
        100.0 * control["d_seg"]
        + math.sqrt(10.0 * control["d_pose"])
        + 25.0 * control_archive / ORIGINAL_BYTES
    )
    for spec in SPECS:
        materialized = materialized_by_id[spec.candidate_id]
        score = scored_by_id[spec.candidate_id]["summary"]
        archive_bytes = int(materialized["archive"]["bytes"])
        byte_credit = control_archive - archive_bytes
        delta_seg = score["d_seg"] - control["d_seg"]
        delta_pose = score["d_pose"] - control["d_pose"]
        delta_distortion = 100.0 * delta_seg + (
            math.sqrt(10.0 * score["d_pose"]) - math.sqrt(10.0 * control["d_pose"])
        )
        byte_credit_s = byte_credit * S_PER_BYTE
        final_score = (
            100.0 * score["d_seg"]
            + math.sqrt(10.0 * score["d_pose"])
            + 25.0 * archive_bytes / ORIGINAL_BYTES
        )
        net_delta_s = final_score - control_score
        if abs(net_delta_s - (delta_distortion - byte_credit_s)) > 1e-12:
            raise AP1Error("purchase score arithmetic does not close")
        rows.append(
            {
                "candidate_id": spec.candidate_id,
                "group": spec.group,
                "level": spec.level,
                "quantum": spec.quantum,
                "archive_bytes": archive_bytes,
                "bytes_held": (
                    0
                    if spec.group == "control"
                    else materialized["physical_bytes_held"].get(spec.group, 96)
                ),
                "byte_credit": byte_credit,
                "d_seg": score["d_seg"],
                "seg_flips": score["flips"],
                "d_pose": score["d_pose"],
                "delta_d_seg": delta_seg,
                "delta_d_pose": delta_pose,
                "delta_s_distortion": delta_distortion,
                "byte_credit_s": byte_credit_s,
                "net_delta_s": net_delta_s,
                "score": final_score,
                "score_per_byte_credited": (
                    delta_distortion / byte_credit if byte_credit > 0 else None
                ),
                "receiver_required": spec.group != "control",
                "distortion_load_bearing": spec.group != "control" and delta_distortion > 0.0,
                "net_negative": net_delta_s < 0.0,
                "per_class_delta": _class_delta(control, score),
                "lane_delta": _class_delta(control, score)[1],
                "denominators": {"seg_pixels": SEG_PIXELS, "pose_values": POSE_VALUES},
            }
        )
    purchase_rows = [row for row in rows if row["group"] != "control"]
    ranking = sorted(
        purchase_rows,
        key=lambda row: (
            math.inf if row["score_per_byte_credited"] is None else row["score_per_byte_credited"],
            row["candidate_id"],
        ),
    )
    best_by_group = []
    for group in ("semantic", "carrier", "hpac", "residual"):
        group_rows = [row for row in purchase_rows if row["group"] == group]
        best_by_group.append(min(group_rows, key=lambda row: (row["net_delta_s"], row["level"])))
    net_negative_best = [row for row in best_by_group if row["net_negative"]]
    independent_credit = sum(int(row["byte_credit"]) for row in net_negative_best)
    result = {
        "schema": "ddm_ap1_residue_purchase_table.v1",
        "created_utc": utc_now(),
        "status": "COMPLETE",
        "axis": AXIS,
        "promotable": False,
        "score_claim": False,
        "control": {
            "archive_bytes": control_archive,
            "archive_sha256": materialized_by_id["control"]["archive"]["sha256"],
            "d_seg": control["d_seg"],
            "seg_flips": control["flips"],
            "d_pose": control["d_pose"],
            "score": control_score,
        },
        "exchange": {
            "s_per_byte": S_PER_BYTE,
            "demand_bytes": DEMAND_BYTES,
            "source_residue_bytes": 66_591,
        },
        "denominators": {
            "pairs": N,
            "seg_pixels": SEG_PIXELS,
            "pose_values": POSE_VALUES,
            "class_gt_pixels": dict(zip(CLASS_NAMES, CLASS_DENOMINATORS, strict=True)),
        },
        "gt_lineage": scored_by_id["control"]["gt_lineage"],
        "rows": rows,
        "exchange_rate_ranking": [row["candidate_id"] for row in ranking],
        "independent_best_by_disjoint_group": [row["candidate_id"] for row in best_by_group],
        "waterfill": {
            "scope": "independent single-group n600 rows; no unmeasured cross-group score composition",
            "net_negative_set": [row["candidate_id"] for row in net_negative_best],
            "independent_byte_credit_sum": independent_credit,
            "share_of_42382_byte_demand": independent_credit / DEMAND_BYTES,
            "composition_score_claim": False,
        },
        "prior_law": {
            "prediction": "at least one >5000 B group has a net-negative coarsening",
            "falsifier": "every cheapest net-negative group returns <1000 B or every row is net-positive",
            "all_purchase_rows_net_positive": all(
                not row["net_negative"] for row in purchase_rows
            ),
            "falsified": all(not row["net_negative"] for row in purchase_rows),
            "observed_groups_over_5000_net_negative": sorted(
                {
                    row["group"]
                    for row in purchase_rows
                    if row["net_negative"] and row["byte_credit"] > 5_000
                }
            ),
        },
        "boundaries": {
            "shipping_candidate_built": False,
            "contest_cpu_or_cuda_authority": False,
            "coder_and_token_stream_fixed": True,
            "one_parameter_group_changed_per_row": True,
            "distortion_interpolation_used": False,
        },
    }
    atomic_json(RECEIPT_ROOT / "PURCHASE_TABLE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def seed_token_cache(candidate_id: str, checkpoint_path: Path) -> dict[str, Any]:
    """Publish a retained CPU token checkpoint through the receiver cache API."""

    if candidate_id not in SPEC_BY_ID:
        raise AP1Error(f"unknown candidate id: {candidate_id}")
    verify_source_pins()
    candidate_root = RECEIPT_ROOT / "retained/candidates" / candidate_id
    materialized = json.loads(
        (candidate_root / "MATERIALIZE_RESULT.json").read_text(encoding="utf-8")
    )
    archive = validate_fact(materialized["archive"])
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    token_path = validate_fact(checkpoint["tokens"])
    if checkpoint.get("complete") is not True:
        raise AP1Error("token checkpoint is not stage-complete")
    if checkpoint["binding"].get("archive_sha256") != materialized["archive"]["sha256"]:
        raise AP1Error("token checkpoint is not bound to the candidate archive")

    runtime = candidate_root / "runtime"
    if str(runtime) in sys.path:
        sys.path.remove(str(runtime))
    sys.path.insert(0, str(runtime))
    for name in list(sys.modules):
        if name == "runtime" or name.startswith("runtime."):
            del sys.modules[name]
    f26 = importlib.import_module("runtime.f26_inflate")
    wc1 = importlib.import_module("runtime.ddm_wc1_advisory_runtime")
    parts = f26.read_residual_archive(archive)
    fingerprint = checkpoint["binding"]["token_decoder_fingerprint"]
    binding = f26._token_cache_binding(
        parts=parts,
        pair_count=N,
        fingerprint=fingerprint,
    )
    entry, cache_report = wc1.publish_token_cache(
        ADVISORY_TOKEN_CACHE,
        binding,
        source=token_path,
        expected_bytes=N * SEG_H * SEG_W,
        created={
            "token_decoder": checkpoint["token_decoder"],
            "archive_sha256": checkpoint["binding"]["archive_sha256"],
            "runtime_source_tree_sha256": checkpoint["binding"]["source_tree_sha256"],
            "source_checkpoint": file_fact(checkpoint_path),
        },
    )
    result = {
        "schema": "ddm_ap1_token_cache_seed.v1",
        "created_utc": utc_now(),
        "status": "COMPLETE",
        "candidate_id": candidate_id,
        "source_checkpoint": file_fact(checkpoint_path),
        "source_tokens": file_fact(token_path),
        "cache_entry": str(entry),
        "cache_report": cache_report,
        "binding": binding,
    }
    atomic_json(ADVISORY_TOKEN_CACHE / f"SEED_FROM_{candidate_id}.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _queue_checkpoint(
    *,
    active: str | None,
    completed: Sequence[str],
    folded: Sequence[Mapping[str, Any]],
    stage: str,
) -> dict[str, Any]:
    pending = [spec.candidate_id for spec in SPECS if spec.candidate_id not in completed]
    state = {
        "schema": "ddm_ap1_scorer_queue.v1",
        "updated_utc": utc_now(),
        "stage": stage,
        "active_candidate": active,
        "completed_candidates": list(completed),
        "pending_candidates": pending,
        "folded_attempts": list(folded),
        "full_n600_concurrency": 1,
        "owner": "ddm_ap1_residue_purchase_scorer",
        "consumer_store": str(RECEIPT_ROOT / "scorer"),
        "fire_trigger": "prior candidate advisory and DALI post-score both complete",
    }
    atomic_json(RECEIPT_ROOT / "QUEUE_STATE.json", state)
    return state


def _select_advisory_attempt(candidate_id: str) -> tuple[str, list[dict[str, Any]]]:
    """Return the resumable/successful attempt or mint the next fresh suffix.

    A terminal nonzero child is evidence, not a directory to overwrite.  Keep
    it folded and advance from ``candidate`` to ``candidate_r2`` (and so on).
    An absent or live terminal receipt means the existing attempt is resumed.
    """

    folded = []
    for attempt_index in range(1, 100):
        attempt_name = (
            candidate_id if attempt_index == 1 else f"{candidate_id}_r{attempt_index}"
        )
        attempt = RECEIPT_ROOT / "advisory" / attempt_name
        if not attempt.exists():
            return attempt_name, folded
        status_path = attempt / "safe_run_status.json"
        result_path = attempt / "contest_auth_eval.json"
        if not status_path.is_file():
            return attempt_name, folded
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("exit") is None:
            return attempt_name, folded
        if int(status.get("exit", -1)) == 0 and result_path.is_file():
            return attempt_name, folded
        folded.append(
            {
                "candidate_id": candidate_id,
                "attempt": attempt_name,
                "disposition": "FOLDED",
                "reason": f"canonical advisory child exited {status.get('exit')}",
                "receipt": str(status_path),
            }
        )
    raise AP1Error(f"too many folded advisory attempts for {candidate_id}")


def _census_folded_advisory_attempts() -> list[dict[str, Any]]:
    folded = []
    for spec in SPECS:
        for attempt_index in range(1, 100):
            attempt_name = (
                spec.candidate_id
                if attempt_index == 1
                else f"{spec.candidate_id}_r{attempt_index}"
            )
            attempt = RECEIPT_ROOT / "advisory" / attempt_name
            if not attempt.exists():
                break
            status_path = attempt / "safe_run_status.json"
            if not status_path.is_file():
                continue
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("exit") is None:
                continue
            if int(status.get("exit", -1)) == 0 and (attempt / "contest_auth_eval.json").is_file():
                continue
            folded.append(
                {
                    "candidate_id": spec.candidate_id,
                    "attempt": attempt_name,
                    "disposition": "FOLDED",
                    "reason": f"canonical advisory child exited {status.get('exit')}",
                    "receipt": str(status_path),
                }
            )
    return folded


def run_queue() -> dict[str, Any]:
    """Sequentially fire canonical advisories and consume each retained raw.

    This orchestration never invokes contest_auth_eval directly: every advisory
    child is minted by tools/fire_local_advisory.py.  A candidate does not
    release the exclusive lane until both that child and AP1's pinned-DALI
    post-score have completed.
    """

    materialize_path = RECEIPT_ROOT / "MATERIALIZE_ALL.json"
    if not materialize_path.is_file():
        raise AP1Error("materialize stage is incomplete")
    materialized = json.loads(materialize_path.read_text(encoding="utf-8"))
    if materialized.get("status") != "COMPLETE":
        raise AP1Error("materialize receipt is not complete")
    runtime_by_id = {
        row["candidate_id"]: Path(row["runtime_dir"])
        for row in materialized["candidates"]
    }
    completed: list[str] = []
    folded = _census_folded_advisory_attempts()
    folded_keys = {(row["candidate_id"], row["attempt"]) for row in folded}
    for spec in SPECS:
        score_result = RECEIPT_ROOT / "scorer" / spec.candidate_id / "RESULT.json"
        if score_result.is_file():
            completed.append(spec.candidate_id)
            continue
        attempt_name, prior_folds = _select_advisory_attempt(spec.candidate_id)
        for row in prior_folds:
            key = (row["candidate_id"], row["attempt"])
            if key not in folded_keys:
                folded.append(row)
                folded_keys.add(key)
        attempt = RECEIPT_ROOT / "advisory" / attempt_name
        status_path = attempt / "safe_run_status.json"
        contest_result = attempt / "contest_auth_eval.json"
        if not attempt.exists():
            command = [
                str(REPO / ".venv/bin/python"),
                "tools/fire_local_advisory.py",
                "--runtime-dir",
                str(runtime_by_id[spec.candidate_id]),
                "--attempt-dir",
                str(attempt),
                "--label",
                f"ddm_ap1_{attempt_name}",
                "--upstream-dir",
                str(ADVISORY_UPSTREAM),
                "--env",
                f"F26_ADVISORY_DECODE_CACHE_ROOT={ADVISORY_TOKEN_CACHE}",
                "--projected-gib",
                "8",
            ]
            launched = subprocess.run(command, cwd=REPO, check=False)
            if launched.returncode:
                raise AP1Error(
                    f"canonical local advisory launcher failed for {spec.candidate_id}: "
                    f"rc={launched.returncode}"
                )
        _queue_checkpoint(
            active=spec.candidate_id,
            completed=completed,
            folded=folded,
            stage="WAITING_FOR_CANONICAL_ADVISORY",
        )
        last_update = 0.0
        status: dict[str, Any] | None = None
        while status is None or status.get("exit") is None:
            now = time.monotonic()
            if now - last_update >= 60.0:
                print(f"queue {spec.candidate_id}: advisory still running", flush=True)
                last_update = now
            if status_path.is_file():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("exit") is not None:
                    break
            time.sleep(10.0)
        if status is None:
            raise AP1Error(f"advisory status vanished for {spec.candidate_id}")
        if int(status.get("exit", -1)) != 0 or not contest_result.is_file():
            folded.append(
                {
                    "candidate_id": spec.candidate_id,
                    "attempt": attempt_name,
                    "disposition": "FOLDED",
                    "reason": f"canonical advisory child exited {status.get('exit')}",
                    "receipt": str(status_path),
                }
            )
            _queue_checkpoint(
                active=None,
                completed=completed,
                folded=folded,
                stage="BLOCKED_ON_FOLDED_ADVISORY",
            )
            raise AP1Error(f"canonical advisory folded for {spec.candidate_id}")
        raw = attempt / "work/inflated/0.raw"
        _queue_checkpoint(
            active=spec.candidate_id,
            completed=completed,
            folded=folded,
            stage="DALI_POST_SCORE",
        )
        score_candidate(spec.candidate_id, raw, batch_pairs=16, torch_threads=4)
        completed.append(spec.candidate_id)
        _queue_checkpoint(
            active=None,
            completed=completed,
            folded=folded,
            stage="CANDIDATE_COMPLETE",
        )
    aggregate_all()
    return _queue_checkpoint(
        active=None,
        completed=completed,
        folded=folded,
        stage="COMPLETE",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("materialize")
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--candidate-id", choices=tuple(SPEC_BY_ID), required=True)
    score_parser.add_argument("--raw", type=Path, required=True)
    score_parser.add_argument("--batch-pairs", type=int, default=16)
    score_parser.add_argument("--torch-threads", type=int, default=4)
    subparsers.add_parser("aggregate")
    cache_parser = subparsers.add_parser("seed-token-cache")
    cache_parser.add_argument("--candidate-id", choices=tuple(SPEC_BY_ID), required=True)
    cache_parser.add_argument("--checkpoint", type=Path, required=True)
    subparsers.add_parser("run-queue")
    args = parser.parse_args()
    if args.command == "materialize":
        materialize_all()
    elif args.command == "score":
        score_candidate(args.candidate_id, args.raw, args.batch_pairs, args.torch_threads)
    elif args.command == "aggregate":
        aggregate_all()
    elif args.command == "seed-token-cache":
        seed_token_cache(args.candidate_id, args.checkpoint)
    elif args.command == "run-queue":
        run_queue()
    else:
        raise AP1Error(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
