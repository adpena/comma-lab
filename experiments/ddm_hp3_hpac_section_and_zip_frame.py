#!/usr/bin/env python3
"""DDM HP3: retained n600 HPAC representation race and ZIP-frame closure.

This scorer-free instrument changes only PR130's entropy-model representation.
Every candidate still Range-encodes all 117,964,800 real semantic labels, so a
successful exact receiver decode leaves the rendered video invariant.  Heavy
work checkpoints at 24-frame boundaries and every serialized payload is kept.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import lzma
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import constriction
import numpy as np
import torch

from tac.pr130_runtime.ddm_hp3_runtime.hp3_codec import (
    FRAME_COUNT,
    FRAME_DIM,
    TOKEN_CHUNK_FRAMES,
    factor_frame_embedding,
    pack_monolithic_checkpoint,
    pack_token_chunks,
    restore_ihs1,
    unpack_monolithic_checkpoint,
    unpack_token_chunks,
)

AXIS = "[macOS-CPU advisory, scorer-free]"
SCORE_CLAIM = False
SCHEMA = "ddm_hp3_hpac_section_and_zip_frame.v1"
H = 384
W = 512
K = 5
TOKENS_PER_FRAME = H * W
TOKEN_COUNT = FRAME_COUNT * TOKENS_PER_FRAME
RATE_DENOMINATOR = 37_545_489
BASE_SCORE = 0.172141297491896447
BASE_BYTES = 191_052
BASE_SHA256 = "0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"
RAW_TOKEN_SHA256 = "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece"
RANGE_BYTES = 116_980
RANGE_SHA256 = "948379872ff81a4e5d948ec301c143be00ebd0033544c8abdfb4af0f4c4a15eb"
HPAC_RAW_BYTES = 20_179
HPAC_RAW_SHA256 = "b07fff73fac41c5fec2d8acbfd7c43c518852696f18d95cf7465fc6ed7510b58"
CHECKPOINT_SHA256 = "0f4775920aeb2fb419555cc4d68703dd90b88be9d24c82466a99fddc1b1f1aa7"
CACHE_SHA256 = "382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195"
INTAKE_HEAD = "e34f31bc4969042c0051ac81aa3c56884419a231"
INTAKE_CODEC_SHA256 = "70632168250cbecc40b9d6de5da5b167adeb56031368311ff936404a1ceba7e0"
DT1_MANIFEST_SHA256 = "23089d6f627e1da56a3f947900727e94ee4a99d1a2ce30fd582aeeac3130caea"

ROOT = Path(__file__).resolve().parents[1]
INTAKE = Path("/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo")
BASE_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip")
CACHE = Path("/Volumes/VertigoDataTier/pact/ddm_pr130_encode_tokens_metal_20260809/caches/gt_cache_600_official_ada.pt")  # GT_LINEAGE_OK: symlink target bytes are registry-classified DALI_NVDEC sha256 382d7dfe38b37c0c
CHECKPOINT = INTAKE / "artifacts/checkpoints/hpac_selfcompress_l1_fastbits_e60.pt"
CANONICAL_HPAC_XZ = INTAKE / "artifacts/hpac/hpac_selfcompress_l1_fastbits_e60.bin.xz"
CANONICAL_RANGE = INTAKE / "artifacts/hpac/hpac_selfcompress_l1_fastbits_e60.tokens.bin"
DT1_MANIFEST = Path("/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/chunk_manifest.json")
FX1_RUNTIME = ROOT / "src/tac/pr130_runtime/fx1_runtime_tree"
HP3_RUNTIME = ROOT / "src/tac/pr130_runtime/ddm_hp3_runtime"
DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_hp3_20260810")
REPO_RECEIPT = ROOT / ".omx/research/ddm_hp3_20260810/FINAL_RECEIPT.json"

LZMA_FILTERS = [
    {
        "id": lzma.FILTER_LZMA2,
        "dict_size": 1 << 16,
        "lc": 0,
        "lp": 1,
        "pb": 0,
        "mode": lzma.MODE_NORMAL,
        "nice_len": 273,
        "mf": lzma.MF_BT4,
        "depth": 0,
    }
]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite retained payload with new bytes: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def replace_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def retain_payload(path: Path, payload: bytes) -> dict[str, Any]:
    atomic_bytes(path, payload)
    return {"path": str(path), "bytes": len(payload), "sha256": sha256_bytes(payload)}


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def storage_preflight(output: Path, required_free_bytes: int) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    if usage.free < required_free_bytes:
        raise RuntimeError(f"HP3 requires {required_free_bytes} free bytes, found {usage.free} at {output}")
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "required_free_bytes": required_free_bytes,
    }


def import_from(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_sources() -> tuple[Any, Any]:
    intake_code = str(INTAKE / "code")
    runtime = str(FX1_RUNTIME)
    for path in (intake_code, runtime):
        if path not in sys.path:
            sys.path.insert(0, path)
    packer = import_from(INTAKE / "code/pack_hpac_self_compress.py", "ddm_hp3_packer")
    inflate = import_from(FX1_RUNTIME / "inflate.py", "ddm_hp3_inflate")
    return packer, inflate


def pin_inputs() -> dict[str, Any]:
    pins = {
        "base_archive": (BASE_ARCHIVE, BASE_BYTES, BASE_SHA256),
        "checkpoint": (CHECKPOINT, None, CHECKPOINT_SHA256),
        "cache": (CACHE, None, CACHE_SHA256),
        "canonical_range": (CANONICAL_RANGE, RANGE_BYTES, RANGE_SHA256),
        "intake_codec": (INTAKE / "code/codec_hpac_integer.py", None, INTAKE_CODEC_SHA256),
        "dt1_manifest": (DT1_MANIFEST, None, DT1_MANIFEST_SHA256),
    }
    result: dict[str, Any] = {}
    for name, (path, expected_bytes, expected_sha) in pins.items():
        if not path.is_file():
            raise RuntimeError(f"required HP3 input is absent: {path}")
        record = file_record(path)
        if expected_bytes is not None and record["bytes"] != expected_bytes:
            raise RuntimeError(f"{name} byte pin failed")
        if record["sha256"] != expected_sha:
            raise RuntimeError(f"{name} SHA-256 pin failed")
        result[name] = record
    intake_status = subprocess.check_output(["git", "-C", str(INTAKE), "status", "--short"], text=True).strip()
    if intake_status:
        raise RuntimeError("immutable PR130 intake has working-tree changes")
    result["intake_head"] = subprocess.check_output(["git", "-C", str(INTAKE), "rev-parse", "HEAD"], text=True).strip()
    if result["intake_head"] != INTAKE_HEAD:
        raise RuntimeError("immutable PR130 intake HEAD pin failed")
    return result


def write_zip(path: Path, payload: bytes, *, compression: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=compression,
        compresslevel=9 if compression == zipfile.ZIP_DEFLATED else None,
        allowZip64=False,
    ) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = compression
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, payload)
    produced = temporary.read_bytes()
    temporary.unlink()
    atomic_bytes(path, produced)


def zip_breakdown(blob: bytes) -> dict[str, int]:
    if blob[:4] != b"PK\x03\x04":
        raise ValueError("ZIP local header signature is absent")
    local_name = struct.unpack_from("<H", blob, 26)[0]
    local_extra = struct.unpack_from("<H", blob, 28)[0]
    compressed = struct.unpack_from("<I", blob, 18)[0]
    local_header = 30
    member_offset = local_header + local_name + local_extra
    eocd = blob.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError("ZIP EOCD is absent")
    central_offset = struct.unpack_from("<I", blob, eocd + 16)[0]
    central_size = struct.unpack_from("<I", blob, eocd + 12)[0]
    if blob[central_offset : central_offset + 4] != b"PK\x01\x02":
        raise ValueError("ZIP central-directory signature is absent")
    central_name = struct.unpack_from("<H", blob, central_offset + 28)[0]
    central_extra = struct.unpack_from("<H", blob, central_offset + 30)[0]
    central_comment = struct.unpack_from("<H", blob, central_offset + 32)[0]
    comment = struct.unpack_from("<H", blob, eocd + 20)[0]
    if central_offset != member_offset + compressed or central_size != (
        46 + central_name + central_extra + central_comment
    ):
        raise ValueError("ZIP record offsets do not reconcile")
    breakdown = {
        "local_header": local_header,
        "local_filename": local_name,
        "local_extra": local_extra,
        "member_data": compressed,
        "central_header": 46,
        "central_filename": central_name,
        "central_extra": central_extra,
        "central_comment": central_comment,
        "eocd": 22,
        "zip_comment": comment,
    }
    if sum(breakdown.values()) != len(blob):
        raise ValueError("ZIP byte breakdown does not sum to the archive stat")
    return breakdown


def measure_container(output: Path, base_payload: bytes) -> dict[str, Any]:
    root = output / "retained/container"
    control = root / "stored_control.zip"
    repeat = root / "stored_control.repeat.zip"
    deflated = root / "deflated_q9.zip"
    write_zip(control, base_payload, compression=zipfile.ZIP_STORED)
    write_zip(repeat, base_payload, compression=zipfile.ZIP_STORED)
    write_zip(deflated, base_payload, compression=zipfile.ZIP_DEFLATED)
    if control.read_bytes() != repeat.read_bytes():
        raise RuntimeError("stored ZIP repeat is not byte-identical")
    extracted = root / "extracted/p"
    extracted.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(deflated) as archive:
        if archive.testzip() is not None or archive.namelist() != ["p"]:
            raise RuntimeError("deflated container candidate did not parse")
        retain_payload(extracted, archive.read("p"))
    if extracted.read_bytes() != base_payload:
        raise RuntimeError("deflated container changed member bytes")
    base_blob = BASE_ARCHIVE.read_bytes()
    base_breakdown = zip_breakdown(base_blob)
    actual_overhead = len(base_blob) - len(base_payload)
    lower_bound = 30 + 1 + 46 + 1 + 22
    if actual_overhead != lower_bound:
        raise RuntimeError("base ZIP is not at the one-character-member structural floor")
    return {
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "base": file_record(BASE_ARCHIVE),
        "base_breakdown": base_breakdown,
        "actual_overhead_bytes": actual_overhead,
        "standards_lower_bound_bytes": lower_bound,
        "lower_bound_derivation": "30 local + 1 name + 46 central + 1 name + 22 EOCD",
        "stored_control": file_record(control),
        "stored_repeat": file_record(repeat),
        "stored_repeat_identical": True,
        "deflated_q9": file_record(deflated),
        "deflated_extracted": file_record(extracted),
        "deflated_member_exact": True,
    }


def model_from_checkpoint(packer: Any) -> Any:
    class Args:
        channels = 64
        patch = 64
        delta = 2
        frame_dim = 8
        weight_bound = 127
        activation_bound = 127
        weight_exponent_min = -6

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = packer.model_from_args(Args, True)
    model.load_state_dict(checkpoint["state_dict"])
    from hpac_self_compress import set_deployed_bit_depths

    set_deployed_bit_depths(model, True)
    return model.eval()


def recompute_depths(model: Any, packer: Any) -> None:
    with torch.no_grad():
        for module in packer.compressible_modules(model):
            rows = packer.module_weight_rows(module, module.codes()[0])
            depths: list[int] = []
            for row in rows:
                low = int(row.min(initial=0))
                high = int(row.max(initial=0))
                selected = 0
                if low != 0 or high != 0:
                    for bits in range(1, 9):
                        bound = getattr(module, "weight_bound", 127)
                        if low >= max(-(1 << (bits - 1)), -bound) and high <= min((1 << (bits - 1)) - 1, bound):
                            selected = bits
                            break
                if selected == 0 and (low != 0 or high != 0):
                    raise RuntimeError("candidate row is outside the 8-bit HPAC grammar")
                depths.append(selected)
            module.bit_depth.copy_(torch.tensor(depths, dtype=module.bit_depth.dtype))


@dataclass(frozen=True)
class CandidateModel:
    name: str
    raw: bytes
    exact_to_control: bool
    changed_values: int
    mechanism: str
    token_chunk_frames: int | None
    code_source: str


def build_candidate_models(output: Path, packer: Any) -> tuple[list[CandidateModel], dict[str, Any]]:
    control_model = model_from_checkpoint(packer)
    control = packer.serialize_self_compressed(control_model)
    if len(control) != HPAC_RAW_BYTES or sha256_bytes(control) != HPAC_RAW_SHA256:
        raise RuntimeError("canonical HPAC reserialization control failed")
    metadata_bytes = math.ceil(sum(module.weight.shape[0] for module in packer.compressible_modules(control_model)) / 2)
    weight_bits = 0
    for module in packer.compressible_modules(control_model):
        depths = packer.deployed_depths(module)
        rows = packer.module_weight_rows(module, module.codes()[0])
        weight_bits += sum(int(bits) * len(row) for bits, row in zip(depths, rows, strict=True))
    fixed_offset = 4 + metadata_bytes + math.ceil(weight_bits / 8)
    fixed_fields: list[dict[str, Any]] = []
    module_by_name = dict(control_model.named_modules())
    cursor = fixed_offset
    for name, parameter in control_model.named_parameters():
        module_name, field = name.rsplit(".", 1)
        module = module_by_name[module_name]
        if field == "bit_depth" or (field == "weight" and isinstance(module, packer.COMPRESSIBLE_TYPES)):
            continue
        byte_count = parameter.numel() * (2 if field == "bias" else 1)
        fixed_fields.append({"name": name, "offset": cursor, "bytes": byte_count})
        cursor += byte_count
    if cursor != len(control):
        raise RuntimeError("HPAC named fixed-field decomposition did not reconcile")
    frame_field = next(field for field in fixed_fields if field["name"] == "frame_embed.weight")
    frame_offset = int(frame_field["offset"])
    if frame_field["bytes"] != FRAME_COUNT * FRAME_DIM:
        raise RuntimeError("HPAC frame embedding geometry changed")
    factored = factor_frame_embedding(control, frame_offset)
    if restore_ihs1(factored) != control:
        raise RuntimeError("HP31 structural factorization did not restore canonical IHS1")

    requant_model = model_from_checkpoint(packer)
    with torch.no_grad():
        original = requant_model.frame_embed.weight.round().clamp(-127, 127)
        requantized = torch.div(original, 2, rounding_mode="trunc") * 2
        requant_model.frame_embed.weight.copy_(requantized)
    requant = packer.serialize_self_compressed(requant_model)
    requant_changed = int(torch.count_nonzero(original != requantized).item())

    prune_model = model_from_checkpoint(packer)
    pruned = 0
    with torch.no_grad():
        for module in packer.compressible_modules(prune_model):
            original_weight = module.weight.round().clamp(-127, 127)
            deployed_codes = module.codes()[0]
            mask = original_weight.abs() <= 1
            pruned += int(torch.count_nonzero(deployed_codes.abs() == 1).item())
            module.weight.copy_(torch.where(mask, torch.zeros_like(original_weight), original_weight))
    recompute_depths(prune_model, packer)
    prune = packer.serialize_self_compressed(prune_model)

    candidates = [
        CandidateModel("control_ihs1", control, True, 0, "canonical IHS1 control", None, "control_ihs1"),
        CandidateModel(
            "chunked_control_ihs1",
            control,
            True,
            0,
            "HPT1 24-frame resumable token framing control",
            24,
            "control_ihs1",
        ),
        CandidateModel(
            "chunked120_control_ihs1",
            control,
            True,
            0,
            "HPT1 120-frame resumable token framing control",
            120,
            "control_ihs1",
        ),
        CandidateModel(
            "checkpoint300_control_ihs1",
            control,
            True,
            0,
            "monolithic Range plus one counted 300-frame seek checkpoint",
            300,
            "control_ihs1",
        ),
        CandidateModel(
            "factor_frame_delta",
            factored,
            True,
            0,
            "exact temporal residual chart over frame_embed with HPT1/24",
            24,
            "control_ihs1",
        ),
        CandidateModel(
            "factor_frame_delta_hpt120",
            factored,
            True,
            0,
            "exact temporal residual chart over frame_embed with HPT1/120",
            120,
            "control_ihs1",
        ),
        CandidateModel(
            "factor_frame_delta_hpm300",
            factored,
            True,
            0,
            "exact temporal residual chart with monolithic Range and one 300-frame checkpoint",
            300,
            "control_ihs1",
        ),
        CandidateModel(
            "requant_frame_embed_step2",
            requant,
            False,
            requant_changed,
            "int8 frame_embed nearest-even with half ties toward zero and HPT1/24",
            24,
            "requant_frame_embed_step2",
        ),
        CandidateModel(
            "requant_frame_embed_step2_hpt120",
            requant,
            False,
            requant_changed,
            "int8 frame_embed nearest-even with half ties toward zero and HPT1/120",
            120,
            "requant_frame_embed_step2",
        ),
        CandidateModel(
            "requant_frame_embed_step2_hpm300",
            requant,
            False,
            requant_changed,
            "int8 frame_embed step2 with monolithic Range and one 300-frame checkpoint",
            300,
            "requant_frame_embed_step2",
        ),
        CandidateModel(
            "prune_weight_abs1",
            prune,
            False,
            pruned,
            "set deployed compressible weights with abs(w)<=1 to zero and HPT1/24",
            24,
            "prune_weight_abs1",
        ),
        CandidateModel(
            "prune_weight_abs1_hpt120",
            prune,
            False,
            pruned,
            "set deployed compressible weights with abs(w)<=1 to zero and HPT1/120",
            120,
            "prune_weight_abs1",
        ),
    ]
    records: dict[str, Any] = {}
    for candidate in candidates:
        path = output / "retained/candidates" / candidate.name / "hpac.raw"
        records[candidate.name] = {
            "payload": retain_payload(path, candidate.raw),
            "mechanism": candidate.mechanism,
            "exact_to_control": candidate.exact_to_control,
            "changed_values": candidate.changed_values,
            "token_chunk_frames": candidate.token_chunk_frames,
            "code_source": candidate.code_source,
            "restored_ihs1": {
                "bytes": len(restore_ihs1(candidate.raw)),
                "sha256": sha256_bytes(restore_ihs1(candidate.raw)),
            },
        }
    decomposition = {
        "raw_hpac_bytes": len(control),
        "magic_bytes": 4,
        "depth_metadata_bytes": metadata_bytes,
        "packed_weight_bits": weight_bits,
        "packed_weight_bytes": math.ceil(weight_bits / 8),
        "fixed_field_bytes": len(control) - fixed_offset,
        "fixed_fields": fixed_fields,
        "frame_embed_offset": frame_offset,
        "frame_embed_bytes": FRAME_COUNT * FRAME_DIM,
        "candidate_records": records,
    }
    return candidates, decomposition


def save_npy_atomic(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = np.load(path, mmap_mode="r", allow_pickle=False)
        if existing.dtype != array.dtype or existing.shape != array.shape or not np.array_equal(existing, array):
            raise RuntimeError(f"retained NumPy payload differs from resumed value: {path}")
        return file_record(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return file_record(path)


def validate_code_manifest(manifest: dict[str, Any], *, expected_hpac_sha256: str) -> dict[str, Any]:
    if manifest.get("complete") is not True:
        raise RuntimeError("HP3 code manifest is not complete")
    if manifest.get("hpac_sha256") != expected_hpac_sha256:
        raise RuntimeError("HP3 code manifest belongs to different HPAC bytes")
    expected_start = 0
    token_total = 0
    for row in manifest.get("chunks", []):
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        if start != expected_start or end <= start:
            raise RuntimeError("HP3 code chunks are not contiguous")
        for field in ("symbols", "codes"):
            record = row[field]
            path = Path(record["path"])
            if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"HP3 retained {field} chunk failed its custody pin")
        token_total += (end - start) * TOKENS_PER_FRAME
        expected_start = end
    if expected_start != FRAME_COUNT or token_total != TOKEN_COUNT:
        raise RuntimeError("HP3 code manifest denominator is not n600")
    return manifest


def baseline_dt1_frames() -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    manifest = json.loads(DT1_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise RuntimeError("DT1 manifest is incomplete")
    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    verified: list[dict[str, Any]] = []
    for row in manifest["chunks"]:
        symbols_path = Path(row["symbols_path"])
        codes_path = Path(row["codes_path"])
        if sha256_file(symbols_path) != row["symbols_sha256"] or sha256_file(codes_path) != row["codes_sha256"]:
            raise RuntimeError("DT1 retained chunk hash failed")
        symbols = np.load(symbols_path, mmap_mode="r", allow_pickle=False)
        codes = np.load(codes_path, mmap_mode="r", allow_pickle=False)
        for frame in range(int(row["start_frame"]), int(row["end_frame"])):
            local = frame - int(row["start_frame"])
            start = local * TOKENS_PER_FRAME
            end = start + TOKENS_PER_FRAME
            frames[frame] = (symbols[start:end], codes[start:end])
        verified.append({"symbols": file_record(symbols_path), "codes": file_record(codes_path)})
    if len(frames) != FRAME_COUNT:
        raise RuntimeError("DT1 retained chunks do not cover n600")
    return frames, {"manifest": file_record(DT1_MANIFEST), "chunks": verified}


def rechunk_dt1(output: Path) -> dict[str, Any]:
    root = output / "retained/codes/control_ihs1"
    manifest_path = root / "chunk_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is True:
            return validate_code_manifest(manifest, expected_hpac_sha256=HPAC_RAW_SHA256)
    frames, source = baseline_dt1_frames()
    rows: list[dict[str, Any]] = []
    for start in range(0, FRAME_COUNT, TOKEN_CHUNK_FRAMES):
        end = min(start + TOKEN_CHUNK_FRAMES, FRAME_COUNT)
        symbols = np.concatenate([frames[frame][0] for frame in range(start, end)])
        codes = np.concatenate([frames[frame][1] for frame in range(start, end)])
        symbols_path = root / f"symbols_{start:03d}_{end:03d}.npy"
        codes_path = root / f"codes_{start:03d}_{end:03d}.npy"
        rows.append(
            {
                "start_frame": start,
                "end_frame": end,
                "symbols": save_npy_atomic(symbols_path, np.asarray(symbols)),
                "codes": save_npy_atomic(codes_path, np.asarray(codes)),
                "source": "DT1 retained canonical n600 lattice, rechunked without value changes",
            }
        )
    manifest = {
        "schema": "ddm_hp3_code_chunks.v1",
        "complete": True,
        "candidate": "control_ihs1",
        "hpac_sha256": HPAC_RAW_SHA256,
        "frames": FRAME_COUNT,
        "tokens": TOKEN_COUNT,
        "chunks": rows,
        "source": source,
    }
    atomic_json(manifest_path, manifest)
    return validate_code_manifest(manifest, expected_hpac_sha256=HPAC_RAW_SHA256)


@torch.no_grad()
def materialize_candidate_codes(
    output: Path,
    candidate: CandidateModel,
    inflate: Any,
    raw_tokens: torch.Tensor,
) -> dict[str, Any]:
    root = output / "retained/codes" / candidate.name
    manifest_path = root / "chunk_manifest.json"
    state_path = root / "run_state.json"
    prior_rows: list[dict[str, Any]] = []
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("hpac_sha256") != sha256_bytes(candidate.raw):
            raise RuntimeError(f"resume HPAC identity mismatch for {candidate.name}")
        prior_rows = list(prior.get("chunks", []))
        if prior.get("complete") is True:
            return validate_code_manifest(prior, expected_hpac_sha256=sha256_bytes(candidate.raw))
    hpac = inflate.load_hpac(restore_ihs1(candidate.raw), torch.device("cpu"))
    masks = inflate.group_masks(torch.device("cpu"))
    sparse = inflate.SparseIntegerHPAC(hpac, H, W)
    rows: list[dict[str, Any]] = []
    for start in range(0, FRAME_COUNT, TOKEN_CHUNK_FRAMES):
        end = min(start + TOKEN_CHUNK_FRAMES, FRAME_COUNT)
        existing = next((row for row in prior_rows if row["start_frame"] == start), None)
        if existing is not None:
            symbols_path = Path(existing["symbols"]["path"])
            codes_path = Path(existing["codes"]["path"])
            if (
                sha256_file(symbols_path) == existing["symbols"]["sha256"]
                and sha256_file(codes_path) == existing["codes"]["sha256"]
            ):
                rows.append(existing)
                continue
        started = time.perf_counter()
        symbol_parts: list[np.ndarray] = []
        code_parts: list[np.ndarray] = []
        for frame in range(start, end):
            previous = torch.zeros((1, H, W), dtype=torch.long) if frame == 0 else raw_tokens[frame - 1 : frame].long()
            current = torch.zeros_like(previous)
            context = hpac.prepare_frame_context(torch.tensor([frame]), previous)
            for group, mask in enumerate(masks):
                selected = sparse.selected_logits(current, context, group)
                codes = selected.mul(8).round().clamp(-32768, 32767).to(torch.int16)
                symbols = raw_tokens[frame][mask].numpy().astype(np.uint8)
                code_parts.append(codes.numpy())
                symbol_parts.append(symbols)
                current[0, mask] = torch.from_numpy(symbols.astype(np.int64))
        symbols_array = np.concatenate(symbol_parts)
        codes_array = np.concatenate(code_parts)
        if symbols_array.shape != ((end - start) * TOKENS_PER_FRAME,) or codes_array.shape != (
            (end - start) * TOKENS_PER_FRAME,
            K,
        ):
            raise RuntimeError("candidate code materialization geometry changed")
        symbols_path = root / f"symbols_{start:03d}_{end:03d}.npy"
        codes_path = root / f"codes_{start:03d}_{end:03d}.npy"
        row = {
            "start_frame": start,
            "end_frame": end,
            "symbols": save_npy_atomic(symbols_path, symbols_array),
            "codes": save_npy_atomic(codes_path, codes_array),
            "materialize_wall_s": time.perf_counter() - started,
        }
        rows.append(row)
        replace_json(
            state_path,
            {
                "schema": "ddm_hp3_materialize_resume.v1",
                "candidate": candidate.name,
                "hpac_sha256": sha256_bytes(candidate.raw),
                "completed_frames": end,
                "complete": end == FRAME_COUNT,
            },
        )
        replace_json(
            manifest_path,
            {
                "schema": "ddm_hp3_code_chunks.v1",
                "complete": end == FRAME_COUNT,
                "candidate": candidate.name,
                "hpac_sha256": sha256_bytes(candidate.raw),
                "frames": end,
                "tokens": sum((item["end_frame"] - item["start_frame"]) * TOKENS_PER_FRAME for item in rows),
                "chunks": rows,
            },
        )
    return validate_code_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        expected_hpac_sha256=sha256_bytes(candidate.raw),
    )


def probability_tables(codes: np.ndarray) -> np.ndarray:
    logits = codes.astype(np.float64) / 8
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def encode_range_chunks(
    output: Path,
    candidate: str,
    manifest: dict[str, Any],
    *,
    chunk_frames: int,
) -> dict[str, Any]:
    if chunk_frames % TOKEN_CHUNK_FRAMES or not 1 <= chunk_frames <= 120:
        raise ValueError("HP3 Range chunk size must be a <=120-frame multiple of the 24-frame source stages")
    root = output / "retained/candidates" / candidate
    chunks: list[bytes] = []
    rows: list[dict[str, Any]] = []
    family = constriction.stream.model.Categorical(perfect=False)
    ideal_bits = 0.0
    source_rows = manifest["chunks"]
    rows_per_chunk = chunk_frames // TOKEN_CHUNK_FRAMES
    groups = [source_rows[start : start + rows_per_chunk] for start in range(0, len(source_rows), rows_per_chunk)]
    for index, group in enumerate(groups):
        chunk_start = int(group[0]["start_frame"])
        chunk_end = int(group[-1]["end_frame"])
        if chunk_end - chunk_start > chunk_frames:
            raise RuntimeError("HP3 grouped Range stage exceeds its declared frame count")
        chunk_path = root / f"range_{index:02d}.bin"
        if chunk_path.exists():
            chunk = chunk_path.read_bytes()
        else:
            encoder = constriction.stream.queue.RangeEncoder()
            for source_row in group:
                symbols = np.load(source_row["symbols"]["path"], mmap_mode="r", allow_pickle=False)
                codes = np.load(source_row["codes"]["path"], mmap_mode="r", allow_pickle=False)
                for start in range(0, len(symbols), 65_536):
                    end = min(start + 65_536, len(symbols))
                    tables = probability_tables(codes[start:end])
                    target = np.asarray(symbols[start:end], dtype=np.int32)
                    encoder.encode(target, family, tables)
            chunk = encoder.get_compressed().tobytes()
            retain_payload(chunk_path, chunk)
        decoder = constriction.stream.queue.RangeDecoder(np.frombuffer(chunk, dtype=np.uint32))
        exact = True
        for source_row in group:
            symbols = np.load(source_row["symbols"]["path"], mmap_mode="r", allow_pickle=False)
            codes = np.load(source_row["codes"]["path"], mmap_mode="r", allow_pickle=False)
            for start in range(0, len(symbols), 65_536):
                end = min(start + 65_536, len(symbols))
                tables = probability_tables(codes[start:end])
                target = np.asarray(symbols[start:end], dtype=np.int32)
                ideal_bits += float(-np.log2(tables[np.arange(len(target)), target].astype(np.float64)).sum())
                decoded = decoder.decode(family, tables)
                exact = exact and np.array_equal(decoded.astype(np.uint8), symbols[start:end])
        if not exact:
            raise RuntimeError(f"cached-table Range decode failed for {candidate} chunk {index}")
        chunks.append(chunk)
        rows.append(
            {
                "index": index,
                "start_frame": chunk_start,
                "end_frame": chunk_end,
                "payload": file_record(chunk_path),
                "exact_symbols": True,
            }
        )
    envelope = pack_token_chunks(tuple(chunks), chunk_frames=chunk_frames)
    token_path = root / "tokens.hpt1"
    token_record = retain_payload(token_path, envelope)
    parsed = unpack_token_chunks(envelope)
    if parsed.chunks != tuple(chunks):
        raise RuntimeError("HPT1 parse-back changed retained Range chunks")
    return {
        "payload": token_record,
        "range_chunks": rows,
        "chunk_count": len(chunks),
        "chunk_frames": chunk_frames,
        "header_and_length_bytes": len(envelope) - sum(len(chunk) for chunk in chunks),
        "all_cached_table_decodes_exact": True,
        "ideal_bits_if_materialized_here": ideal_bits,
    }


def encode_monolithic_checkpoint(
    output: Path,
    candidate: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Retain one Range stream plus a counted, receiver-seekable midpoint state."""

    root = output / "retained/candidates" / candidate
    token_path = root / "tokens.hpm1"
    checkpoint_manifest_path = root / "range_checkpoint_manifest.json"
    snapshot_frames = [120, 240, 300, 420, 540, 600]
    family = constriction.stream.model.Categorical(perfect=False)
    if token_path.exists():
        token_payload = token_path.read_bytes()
        envelope = unpack_monolithic_checkpoint(token_payload)
        checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
        if checkpoint_manifest.get("complete") is not True:
            raise RuntimeError(f"incomplete HPM1 checkpoint manifest for {candidate}")
        range_payload = envelope.range_payload
    else:
        encoder = constriction.stream.queue.RangeEncoder()
        checkpoints: list[dict[str, Any]] = []
        midpoint: tuple[int, tuple[int, int]] | None = None
        encoded_symbols = 0
        snapshot_index = 0
        for row in manifest["chunks"]:
            symbols = np.load(row["symbols"]["path"], mmap_mode="r", allow_pickle=False)
            codes = np.load(row["codes"]["path"], mmap_mode="r", allow_pickle=False)
            start = 0
            while start < len(symbols):
                snapshot_symbols = snapshot_frames[snapshot_index] * TOKENS_PER_FRAME
                until_snapshot = snapshot_symbols - encoded_symbols
                end = min(start + 65_536, start + until_snapshot, len(symbols))
                encoder.encode(
                    np.asarray(symbols[start:end], dtype=np.int32),
                    family,
                    probability_tables(codes[start:end]),
                )
                encoded_symbols += end - start
                start = end
                if encoded_symbols == snapshot_symbols:
                    completed_frames = snapshot_frames[snapshot_index]
                    position, state = encoder.pos()
                    snapshot = encoder.get_compressed().tobytes()
                    snapshot_record = retain_payload(
                        root / f"range_snapshot_{completed_frames:03d}.bin",
                        snapshot,
                    )
                    checkpoints.append(
                        {
                            "completed_frames": completed_frames,
                            "position": position,
                            "state": list(state),
                            "snapshot": snapshot_record,
                        }
                    )
                    if completed_frames == 300:
                        midpoint = (position, state)
                    snapshot_index += 1
        if encoded_symbols != TOKEN_COUNT or snapshot_index != len(snapshot_frames):
            raise RuntimeError("HPM1 encoder did not consume n600 at every declared checkpoint")
        if midpoint is None:
            raise RuntimeError("HPM1 encoder did not reach its 300-frame checkpoint")
        range_payload = encoder.get_compressed().tobytes()
        range_record = retain_payload(root / "range_final.bin", range_payload)
        if checkpoints[-1]["snapshot"]["sha256"] != sha256_bytes(range_payload):
            raise RuntimeError("HPM1 final preserved snapshot differs from final Range bytes")
        if manifest["hpac_sha256"] == HPAC_RAW_SHA256 and (
            len(range_payload) != RANGE_BYTES or sha256_bytes(range_payload) != RANGE_SHA256
        ):
            raise RuntimeError("HPM1 exact-control Range stream differs from canonical bytes")
        token_payload = pack_monolithic_checkpoint(range_payload, position=midpoint[0], state=midpoint[1])
        retain_payload(token_path, token_payload)
        checkpoint_manifest = {
            "schema": "ddm_hp3_monolithic_range_checkpoints.v1",
            "complete": True,
            "candidate": candidate,
            "hpac_sha256": manifest["hpac_sha256"],
            "checkpoint_frames": 300,
            "checkpoints": checkpoints,
            "range_payload": range_record,
        }
        atomic_json(checkpoint_manifest_path, checkpoint_manifest)
        envelope = unpack_monolithic_checkpoint(token_payload)

    decoder = constriction.stream.queue.RangeDecoder(np.frombuffer(range_payload, dtype=np.uint32))
    ideal_bits = 0.0
    all_exact = True
    for row in manifest["chunks"]:
        symbols = np.load(row["symbols"]["path"], mmap_mode="r", allow_pickle=False)
        codes = np.load(row["codes"]["path"], mmap_mode="r", allow_pickle=False)
        for start in range(0, len(symbols), 65_536):
            end = min(start + 65_536, len(symbols))
            tables = probability_tables(codes[start:end])
            target = np.asarray(symbols[start:end], dtype=np.int32)
            ideal_bits += float(-np.log2(tables[np.arange(len(target)), target].astype(np.float64)).sum())
            decoded = decoder.decode(family, tables)
            all_exact = all_exact and np.array_equal(decoded.astype(np.uint8), target)
    if not all_exact:
        raise RuntimeError(f"HPM1 full Range decode failed for {candidate}")

    seek_decoder = constriction.stream.queue.RangeDecoder(np.frombuffer(range_payload, dtype=np.uint32))
    seek_decoder.seek(envelope.position, envelope.state)
    seek_exact = True
    for row in manifest["chunks"]:
        row_start_frame = int(row["start_frame"])
        row_end_frame = int(row["end_frame"])
        if row_end_frame <= 300:
            continue
        symbols = np.load(row["symbols"]["path"], mmap_mode="r", allow_pickle=False)
        codes = np.load(row["codes"]["path"], mmap_mode="r", allow_pickle=False)
        first_symbol = max(0, 300 - row_start_frame) * TOKENS_PER_FRAME
        for start in range(first_symbol, len(symbols), 65_536):
            end = min(start + 65_536, len(symbols))
            decoded = seek_decoder.decode(family, probability_tables(codes[start:end]))
            seek_exact = seek_exact and np.array_equal(decoded.astype(np.uint8), symbols[start:end])
    if not seek_exact:
        raise RuntimeError(f"HPM1 300-frame seek decode failed for {candidate}")
    return {
        "payload": file_record(token_path),
        "range_payload": {
            "bytes": len(range_payload),
            "sha256": sha256_bytes(range_payload),
        },
        "checkpoint_manifest": file_record(checkpoint_manifest_path),
        "checkpoint_frames": 300,
        "header_bytes": len(token_payload) - len(range_payload),
        "full_decode_exact": True,
        "seek_decode_exact": True,
        "ideal_bits": ideal_bits,
    }


def split_base() -> tuple[bytes, bytes, bytes, dict[str, int]]:
    with zipfile.ZipFile(BASE_ARCHIVE) as archive:
        if archive.namelist() != ["p"]:
            raise RuntimeError("base archive grammar changed")
        payload = archive.read("p")
    models_bytes = struct.unpack_from("<I", payload)[0]
    models_xz = payload[4 : 4 + models_bytes]
    tokens = payload[4 + models_bytes :]
    models = lzma.decompress(models_xz)
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", models)
    semantic_pose_end = 8 + semantic_bytes + carrier_bytes
    return (
        payload,
        models[:semantic_pose_end],
        models[semantic_pose_end:],
        {
            "member_bytes": len(payload),
            "models_xz_bytes": len(models_xz),
            "models_raw_bytes": len(models),
            "semantic_raw_bytes": semantic_bytes,
            "carrier_raw_bytes": carrier_bytes,
            "hpac_raw_bytes": len(models) - semantic_pose_end,
            "range_bytes": len(tokens),
        },
    )


def build_archive_candidate(
    output: Path,
    candidate: CandidateModel,
    semantic_pose: bytes,
    token_payload: bytes,
) -> dict[str, Any]:
    root = output / "retained/candidates" / candidate.name
    models_raw = semantic_pose + candidate.raw
    models_xz = lzma.compress(models_raw, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS)
    member = struct.pack("<I", len(models_xz)) + models_xz + token_payload
    models_raw_record = retain_payload(root / "models.raw", models_raw)
    models_xz_record = retain_payload(root / "models.xz", models_xz)
    member_record = retain_payload(root / "p", member)
    archive_path = root / "archive.zip"
    repeat_path = root / "archive.repeat.zip"
    write_zip(archive_path, member, compression=zipfile.ZIP_STORED)
    write_zip(repeat_path, member, compression=zipfile.ZIP_STORED)
    if archive_path.read_bytes() != repeat_path.read_bytes():
        raise RuntimeError(f"candidate ZIP is nondeterministic: {candidate.name}")
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["p"] or archive.read("p") != member:
            raise RuntimeError(f"candidate ZIP parse-back failed: {candidate.name}")
    archive_record = file_record(archive_path)
    return {
        "candidate": candidate.name,
        "mechanism": candidate.mechanism,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "model_exact_to_control": candidate.exact_to_control,
        "changed_values": candidate.changed_values,
        "hpac_raw": file_record(root / "hpac.raw"),
        "models_raw": models_raw_record,
        "models_xz": models_xz_record,
        "token_payload": file_record(
            root
            / (
                "tokens.range"
                if candidate.token_chunk_frames is None
                else "tokens.hpm1"
                if candidate.token_chunk_frames == 300
                else "tokens.hpt1"
            )
        ),
        "member": member_record,
        "archive": archive_record,
        "archive_repeat": file_record(repeat_path),
        "repeat_identical": True,
        "zip_breakdown": zip_breakdown(archive_path.read_bytes()),
        "delta_bytes_vs_exact_base": archive_record["bytes"] - BASE_BYTES,
        "derived_rate_score_delta": 25 * (archive_record["bytes"] - BASE_BYTES) / RATE_DENOMINATOR,
    }


def stage_runtime(output: Path, winner: dict[str, Any]) -> dict[str, Any]:
    stage = (
        output
        / "retained/winner_submissions"
        / (
            f"{winner['candidate']}_{winner['archive']['sha256'][:16]}_"
            f"{sha256_file(HP3_RUNTIME / 'inflate_hp3.py')[:12]}"
        )
    )
    stage.mkdir(parents=True, exist_ok=True)
    borrowed = [
        "receiver.py",
        "hpac_integer.py",
        "hpac_integer_sparse.py",
        "integer_model_io.py",
        "carrier_codec.py",
    ]
    records: list[dict[str, Any]] = []
    for name in borrowed:
        source = FX1_RUNTIME / name
        destination = stage / name
        atomic_bytes(destination, source.read_bytes())
        records.append({"source": str(source), "staged": file_record(destination)})
    atomic_bytes(stage / "inflate_base.py", (FX1_RUNTIME / "inflate.py").read_bytes())
    records.append({"source": str(FX1_RUNTIME / "inflate.py"), "staged": file_record(stage / "inflate_base.py")})
    for name in ("inflate_hp3.py", "inflate.sh", "hp3_codec.py"):
        source = HP3_RUNTIME / name
        destination = stage / name
        atomic_bytes(destination, source.read_bytes())
        records.append({"source": str(source), "staged": file_record(destination)})
    archive_source = Path(winner["archive"]["path"])
    atomic_bytes(stage / "archive.zip", archive_source.read_bytes())
    archive_dir = stage / "archive"
    archive_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_source) as archive:
        retain_payload(archive_dir / "p", archive.read("p"))
    return {"path": str(stage), "files": records, "archive": file_record(stage / "archive.zip")}


def run_inflate(output: Path, staged: dict[str, Any]) -> dict[str, Any]:
    stage = Path(staged["path"])
    destination = stage / "inflated/0.raw"
    command = [
        "bash",
        str(stage / "inflate.sh"),
        str(stage / "archive"),
        str(stage / "inflated"),
        str(ROOT / "upstream/public_test_video_names.txt"),
    ]
    env = os.environ.copy()
    env["DDM_HP3_DEVICE"] = "cpu"
    env["PATH"] = f"{ROOT / '.venv/bin'}:{env.get('PATH', '')}"
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    log = (
        json.dumps({"argv": command, "returncode": completed.returncode}, sort_keys=True)
        + "\nSTDOUT\n"
        + completed.stdout
        + "\nSTDERR\n"
        + completed.stderr
    ).encode()
    log_record = retain_payload(stage / "inflate.log", log)
    if completed.returncode != 0 or not destination.is_file():
        raise RuntimeError(f"HP3 real inflate failed with rc={completed.returncode}")
    token_path = destination.with_name(destination.name + ".hp3_state") / "tokens.u8"
    if sha256_file(token_path) != RAW_TOKEN_SHA256 or token_path.stat().st_size != TOKEN_COUNT:
        raise RuntimeError("HP3 real receiver did not reconstruct all canonical n600 tokens")
    render_state = json.loads(
        (destination.with_name(destination.name + ".hp3_state") / "render_state.json").read_text(encoding="utf-8")
    )
    if render_state.get("complete") is not True:
        raise RuntimeError("HP3 real inflate render checkpoint is incomplete")
    state_dir = destination.with_name(destination.name + ".hp3_state")
    checkpoint_records = [file_record(path) for path in sorted(state_dir.glob("*_stage_*.json"))]
    if not checkpoint_records:
        raise RuntimeError("HP3 receiver did not preserve per-stage checkpoint receipts")
    return {
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "argv": command,
        "returncode": completed.returncode,
        "wall_s": time.perf_counter() - started,
        "log": log_record,
        "decoded_tokens": file_record(token_path),
        "decoded_tokens_exact": True,
        "inflated_raw": file_record(destination),
        "render_state": render_state,
        "checkpoint_files": checkpoint_records,
    }


def initialize_state(args: argparse.Namespace, pins: dict[str, Any], preflight: dict[str, int]) -> None:
    expected = args.output / "run_state.json"
    if args.resume_from.resolve() != expected.resolve():
        raise RuntimeError(f"--resume-from must name the HP3 state path exactly: {expected}")
    if args.resume_from.exists():
        state = json.loads(args.resume_from.read_text(encoding="utf-8"))
        if state.get("schema") != SCHEMA or state.get("base_sha256") != BASE_SHA256:
            raise RuntimeError("HP3 resume state provenance mismatch")
        return
    atomic_json(
        args.resume_from,
        {
            "schema": SCHEMA,
            "complete": False,
            "base_sha256": BASE_SHA256,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "argv": sys.argv,
            "pins": pins,
            "storage_preflight": preflight,
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    preflight = storage_preflight(args.output, args.required_free_bytes)
    pins = pin_inputs()
    initialize_state(args, pins, preflight)
    packer, inflate = configure_sources()
    base_payload, semantic_pose, hpac_raw, base_anatomy = split_base()
    if sha256_bytes(hpac_raw) != HPAC_RAW_SHA256:
        raise RuntimeError("HPAC bytes extracted from the exact base changed")
    container = measure_container(args.output, base_payload)
    candidates, decomposition = build_candidate_models(args.output, packer)
    models_by_name = {candidate.name: candidate for candidate in candidates}
    control_manifest = rechunk_dt1(args.output)
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)["seg"][:FRAME_COUNT].to(torch.uint8)
    if tuple(cache.shape) != (FRAME_COUNT, H, W):
        raise RuntimeError("canonical cache geometry changed")
    candidate_rows: list[dict[str, Any]] = []
    token_results: dict[str, dict[str, Any]] = {}
    archive_results: dict[str, dict[str, Any]] = {}
    canonical_range = CANONICAL_RANGE.read_bytes()
    retain_payload(args.output / "retained/candidates/control_ihs1/tokens.range", canonical_range)
    for candidate in candidates:
        if candidate.token_chunk_frames is None:
            token_payload = canonical_range
            token_results[candidate.name] = {
                "payload": file_record(args.output / "retained/candidates/control_ihs1/tokens.range"),
                "canonical_monolithic_range": True,
            }
        else:
            manifest = (
                control_manifest
                if candidate.exact_to_control
                else materialize_candidate_codes(args.output, models_by_name[candidate.code_source], inflate, cache)
            )
            if candidate.token_chunk_frames == 300:
                token_results[candidate.name] = encode_monolithic_checkpoint(args.output, candidate.name, manifest)
            else:
                token_results[candidate.name] = encode_range_chunks(
                    args.output,
                    candidate.name,
                    manifest,
                    chunk_frames=candidate.token_chunk_frames,
                )
            token_results[candidate.name]["code_manifest"] = manifest
            token_payload = Path(token_results[candidate.name]["payload"]["path"]).read_bytes()
        archive = build_archive_candidate(args.output, candidate, semantic_pose, token_payload)
        archive_results[candidate.name] = archive
        candidate_rows.append(archive)
    control_archive = archive_results["control_ihs1"]["archive"]
    if control_archive["bytes"] != BASE_BYTES or control_archive["sha256"] != BASE_SHA256:
        raise RuntimeError("HP3 exact archive control failed to reproduce CPR1 bytes")
    survivors = [
        row for row in candidate_rows if row["candidate"] != "control_ihs1" and row["delta_bytes_vs_exact_base"] < 0
    ]
    winner = min(survivors, key=lambda row: row["archive"]["bytes"]) if survivors else None
    staged = inflate_result = None
    if winner is not None and not args.skip_inflate:
        staged = stage_runtime(args.output, winner)
        inflate_result = run_inflate(args.output, staged)
    result = {
        "schema": SCHEMA,
        "complete": True,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "authority_boundary": "no scorer slot; no upstream/evaluate.py row fired",
        "base": {
            "archive": file_record(BASE_ARCHIVE),
            "score": BASE_SCORE,
            "score_axis": "[contest-CUDA, DALI GT, n600] borrowed CPR1 base",
            "anatomy": base_anatomy,
        },
        "storage_preflight": preflight,
        "pins": pins,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "constriction": getattr(constriction, "__version__", "0.5.0"),
        },
        "hpac_decomposition": decomposition,
        "container": container,
        "token_results": token_results,
        "candidates": candidate_rows,
        "survivor_count": len(survivors),
        "winner": winner,
        "winner_staged_runtime": staged,
        "winner_inflate": inflate_result,
        "pointer_moved": False,
        "exact_eval_fired": False,
    }
    replace_json(args.output / "FINAL_RECEIPT.json", result)
    replace_json(REPO_RECEIPT, result)
    state = json.loads(args.resume_from.read_text(encoding="utf-8"))
    state.update(
        {
            "complete": True,
            "final_receipt": file_record(args.output / "FINAL_RECEIPT.json"),
            "winner": winner["candidate"] if winner is not None else None,
        }
    )
    replace_json(args.resume_from, state)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--required-free-bytes", type=int, default=15 << 30)
    parser.add_argument("--skip-inflate", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "winner": None if result["winner"] is None else result["winner"]["candidate"],
                "survivor_count": result["survivor_count"],
                "exact_eval_fired": result["exact_eval_fired"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
