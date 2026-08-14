#!/usr/bin/env python3
"""RX1: retained, scorer-free tq1c probability-object transfer onto MC36.

All expensive work is split into frame- or coder-state-checkpointed stages.
Every materialized model, probability lattice, entropy stream, decoded token
field, and candidate archive is persisted below ``--output`` before its size is
reported. This runner never invokes a scorer. Its optional final CPU stage
renders and persists the complete raw output for an exact identity receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import lzma
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Literal

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

cp = importlib.import_module("experiments.ddm_cp135_rate_compose")


DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_rx1_rate_attack_20260814")
DEFAULT_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/micro35_candidate/archive.zip")
DEFAULT_RUNTIME = Path(
    "/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/micro35_candidate/adapted_runtime"
)
DEFAULT_SOURCE_MANIFEST = Path(
    "/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532/compile_workspace/retained/"
    "candidates/qs1_combined_unique_pairs/primary/chunk_manifest.json"
)
DEFAULT_EXPECTED_SPATIAL = Path(
    "/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532/compile_workspace/retained/"
    "candidates/qs1_combined_unique_pairs/primary/receiver_state/decoded_spatial_tokens.shipped.bin"
)
DEFAULT_TQ1C_XZ = Path("/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/artifacts/tq1c/hpac.bin.xz")
DEFAULT_TQ1C_CHECKPOINT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/tq1c/hpac_selfcompress_e60.pt"
)
DEFAULT_EXPERIMENT_BOOK = Path("/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book")

EXPECTED_ARCHIVE_BYTES = 186_269
EXPECTED_ARCHIVE_SHA256 = "f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de"
EXPECTED_TQ1C_XZ_BYTES = 14_116
EXPECTED_TQ1C_XZ_SHA256 = "6c44216e8f79bd7d04e998b898d5bf0dc16bae6e3763f8bc19ce4ec8ebdabb40"
EXPECTED_TQ1C_CHECKPOINT_BYTES = 177_614
EXPECTED_TQ1C_CHECKPOINT_SHA256 = "2a907f06cc5d278e1df12eac6cd575fb3dcb32477446f0da842bb92a14d05ddc"
EXPECTED_EVENT_SHA256 = "f4149ab66096e9de8771d5cf9be1058c543177acc0041fed6c361b73e0820be8"
EXPECTED_SPATIAL_SHA256 = "9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52"
EXPECTED_EVENTS = 600 * 384 * 512
EVENTS_PER_FRAME = 384 * 512
RATE_DENOMINATOR = 37_545_489
MC36_SCORE = 0.1619344578804448
F26P_LIFTED_RUNTIME = Path("/Volumes/VertigoDataTier/pact/ddm_f26p_runtime_cpu_lift_20260814/lifted_submission_cpu")
EXPECTED_CPU_RAW_BYTES = 3_662_409_600
EXPECTED_CPU_RAW_SHA256 = "e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9"
AXIS = "[macOS-CPU advisory, scorer-free lossless composition]"
SCORE_CLAIM = False
VARIANTS = ("tq1c_table_on", "tq1c_table_off")
Variant = Literal["tq1c_table_on", "tq1c_table_off"]

HP4_HEADER = struct.Struct("<4sBBBBHHHHH")
RX1_HEADER = struct.Struct("<4sBBBBHHH")
RX1_MAGIC = b"RX1M"
RX1_VERSION = 1
RX1_CODEC_XZ = 1
RX1_CODEC_BROTLI = 2
RX1_TABLE_ON = 0
RX1_TABLE_OFF = 1
RC64_STATE_HEADER = cp.RC64_STATE_HEADER


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def retention_inventory(output: Path) -> dict[str, Any]:
    retained = output / "retained"
    files = [file_record(path) for path in sorted(retained.rglob("*")) if path.is_file()]
    return {
        "schema": "ddm_rx1_retention_inventory.v1",
        "root": str(retained.resolve()),
        "file_count": len(files),
        "total_bytes": sum(record["bytes"] for record in files),
        "files": files,
    }


def atomic_bytes(path: Path, value: bytes, *, executable: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    if executable:
        temporary.chmod(0o755)
    os.replace(temporary, path)
    return file_record(path)


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    return atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return atomic_bytes(path, buffer.getvalue())


def deterministic_zip(member: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, member)
    return output.getvalue()


def read_stored_member(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != "p" or infos[0].compress_type != zipfile.ZIP_STORED:
            raise RuntimeError("archive must contain exactly one stored member p")
        value = archive.read(infos[0])
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
    return value


def require_file(path: Path, *, size: int | None = None, digest: str | None = None) -> None:
    if not path.is_file():
        raise RuntimeError(f"required file is absent: {path}")
    if size is not None and path.stat().st_size != size:
        raise RuntimeError(f"required file size differs: {path}")
    if digest is not None and sha256_file(path) != digest:
        raise RuntimeError(f"required file SHA-256 differs: {path}")


def require_prepared(args: argparse.Namespace) -> dict[str, Any]:
    path = args.output / "PREPARE_RESULT.json"
    if not path.is_file():
        raise RuntimeError("prepare stage is incomplete")
    value = json.loads(path.read_text())
    if value.get("complete") is not True:
        raise RuntimeError("prepare receipt is incomplete")
    return value


def _source(args: argparse.Namespace) -> cp.SourceSymbols:
    return cp.SourceSymbols(args.source_manifest)


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    require_file(args.archive, size=EXPECTED_ARCHIVE_BYTES, digest=EXPECTED_ARCHIVE_SHA256)
    require_file(args.tq1c_xz, size=EXPECTED_TQ1C_XZ_BYTES, digest=EXPECTED_TQ1C_XZ_SHA256)
    require_file(
        args.tq1c_checkpoint,
        size=EXPECTED_TQ1C_CHECKPOINT_BYTES,
        digest=EXPECTED_TQ1C_CHECKPOINT_SHA256,
    )
    require_file(args.expected_spatial, size=EXPECTED_EVENTS, digest=EXPECTED_SPATIAL_SHA256)
    if not args.runtime.is_dir() or not args.experiment_book.is_dir():
        raise RuntimeError("runtime or ExperimentBook input is absent")
    source = _source(args)
    source_digest = source.digest()
    if source_digest != EXPECTED_EVENT_SHA256:
        raise RuntimeError("MC36 event-order source digest differs")
    args.output.mkdir(parents=True, exist_ok=True)
    stats = os.statvfs(args.output)
    free_bytes = stats.f_bavail * stats.f_frsize
    required_bytes = args.required_free_gib * (1 << 30)
    if free_bytes < required_bytes:
        raise RuntimeError(f"RX1 needs {required_bytes} free bytes but only {free_bytes} are available")
    result = {
        "schema": "ddm_rx1_preflight.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "archive": file_record(args.archive),
        "tq1c_xz": file_record(args.tq1c_xz),
        "tq1c_checkpoint": file_record(args.tq1c_checkpoint),
        "source_manifest": file_record(args.source_manifest),
        "source_event_order_sha256": source_digest,
        "expected_spatial": file_record(args.expected_spatial),
        "output": str(args.output.resolve()),
        "filesystem_free_bytes": free_bytes,
        "required_free_bytes": required_bytes,
        "storage_preflight_pass": True,
        "all_long_stages_checkpointed": True,
        "complete": True,
    }
    atomic_json(args.output / "PREFLIGHT.json", result)
    return result


def _brotli(value: bytes, quality: int, binary: str) -> bytes:
    completed = subprocess.run([binary, "-q", str(quality), "-c"], input=value, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"Brotli q{quality} failed: {completed.stderr.decode(errors='replace')}")
    return completed.stdout


def _brotli_restore(value: bytes, binary: str) -> bytes:
    completed = subprocess.run([binary, "-d", "-c"], input=value, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError("Brotli parse-back failed")
    return completed.stdout


def _hp4_fields(archive_path: Path) -> tuple[tuple[bytes, ...], bytes]:
    member = read_stored_member(archive_path)
    if len(member) < HP4_HEADER.size:
        raise RuntimeError("MC36 HP4 member is truncated")
    magic, version, predictor, codec, flags, *lengths = HP4_HEADER.unpack_from(member)
    if (magic, version, predictor, codec, flags) != (b"HP4M", 1, 0, 2, 0):
        raise RuntimeError("MC36 HP4 header differs")
    model_end = HP4_HEADER.size + sum(lengths)
    if model_end + 96 >= len(member):
        raise RuntimeError("MC36 HP4 field accounting differs")
    fields = []
    offset = HP4_HEADER.size
    for length in lengths:
        fields.append(member[offset : offset + length])
        offset += length
    return tuple(fields), member[model_end:]


def _patch_runtime(source: str) -> str:
    marker = "# DDM_RX1_IHS1_CONTAINER_V1"
    if marker in source:
        old = """    semantic_body = _decompress_brotli(semantic_stream)
    carrier_body = _decompress_brotli(carrier_stream)
    if len(semantic_body) != WANS_BODY_BYTES:
"""
        new = """    semantic_body = _decompress_brotli(semantic_stream)
    carrier_body = _decompress_brotli(carrier_stream)
    # DDM_RX1_HP4_CARRIER_INVERSE_V1
    if len(carrier_body) == PACKED_CAP1_SECTION_BYTES:
        carrier_body = _restore_packed_cap1_metadata(carrier_body)
    elif len(carrier_body) > PACKED_CAP1_SECTION_BYTES and carrier_body[PACKED_CAP1_SECTION_BYTES:].startswith(COMPENSATION_MAGIC):
        carrier_body = _restore_packed_cap1_metadata(carrier_body[:PACKED_CAP1_SECTION_BYTES]) + carrier_body[PACKED_CAP1_SECTION_BYTES:]
    elif len(carrier_body) != CANONICAL_CAP1_SECTION_BYTES:
        raise ResidualArchiveError("RX1 carrier representation differs")
    if len(semantic_body) != WANS_BODY_BYTES:
"""
        if "# DDM_RX1_HP4_CARRIER_INVERSE_V1" not in source:
            if source.count(old) != 1:
                raise RuntimeError("RX1 runtime carrier migration point differs")
            source = source.replace(old, new)
        return source
    constants_old = 'HP4_MAGIC = b"HP4M"\n'
    constants_new = constants_old + (
        f"{marker}\n"
        'RX1_MAGIC = b"RX1M"\n'
        'RX1_MODEL_HEADER = struct.Struct("<4sBBBBHHH")\n'
        "RX1_CODEC_XZ = 1\n"
        "RX1_CODEC_BROTLI = 2\n"
    )
    if source.count(constants_old) != 1:
        raise RuntimeError("RX1 runtime constant insertion point differs")
    source = source.replace(constants_old, constants_new)
    helper_marker = "def _decode_hp4_models(outer: bytes) -> tuple[bytes, bytes] | None:\n"
    helper = '''def _decode_rx1_models(outer: bytes):
    """Restore RX1's counted IHS1 probability object and frozen MC36 renderer."""
    if len(outer) < RX1_MODEL_HEADER.size or not outer.startswith(RX1_MAGIC):
        return None
    magic, version, codec, table_mode, reserved, hpac_bytes, semantic_bytes, carrier_bytes = RX1_MODEL_HEADER.unpack_from(outer)
    model_end = RX1_MODEL_HEADER.size + hpac_bytes + semantic_bytes + carrier_bytes
    if magic != RX1_MAGIC or version != 1 or codec not in (RX1_CODEC_XZ, RX1_CODEC_BROTLI):
        raise ResidualArchiveError("unsupported RX1 model")
    if table_mode not in (0, 1) or reserved != 0 or min(hpac_bytes, semantic_bytes, carrier_bytes) <= 0:
        raise ResidualArchiveError("invalid RX1 model metadata")
    if model_end + 96 >= len(outer):
        raise ResidualArchiveError("truncated RX1 model")
    offset = RX1_MODEL_HEADER.size
    hpac_stream = outer[offset : offset + hpac_bytes]
    offset += hpac_bytes
    semantic_stream = outer[offset : offset + semantic_bytes]
    offset += semantic_bytes
    carrier_stream = outer[offset : offset + carrier_bytes]
    try:
        hpac = lzma.decompress(hpac_stream, format=lzma.FORMAT_XZ) if codec == RX1_CODEC_XZ else _decompress_brotli(hpac_stream)
    except lzma.LZMAError as error:
        raise ResidualArchiveError("invalid RX1 IHS1 stream") from error
    if not hpac.startswith(b"IHS1"):
        raise ResidualArchiveError("RX1 HPAC is not canonical IHS1")
    semantic_body = _decompress_brotli(semantic_stream)
    carrier_body = _decompress_brotli(carrier_stream)
    # DDM_RX1_HP4_CARRIER_INVERSE_V1
    if len(carrier_body) == PACKED_CAP1_SECTION_BYTES:
        carrier_body = _restore_packed_cap1_metadata(carrier_body)
    elif len(carrier_body) > PACKED_CAP1_SECTION_BYTES and carrier_body[PACKED_CAP1_SECTION_BYTES:].startswith(COMPENSATION_MAGIC):
        carrier_body = _restore_packed_cap1_metadata(carrier_body[:PACKED_CAP1_SECTION_BYTES]) + carrier_body[PACKED_CAP1_SECTION_BYTES:]
    elif len(carrier_body) != CANONICAL_CAP1_SECTION_BYTES:
        raise ResidualArchiveError("RX1 carrier representation differs")
    if len(semantic_body) != WANS_BODY_BYTES:
        raise ResidualArchiveError("RX1 semantic section length differs")
    try:
        semantic = decode_f12_wans_body(semantic_body, WANS_STREAM_ORDER)
        decode_wans1(semantic)
    except RendererWeightCodecError as error:
        raise ResidualArchiveError("invalid RX1 renderer weights") from error
    cap1_bytes = _cap1_body_bytes(carrier_body)
    if cap1_bytes >= len(carrier_body):
        raise ResidualArchiveError("truncated RX1 carrier")
    cap1 = _restore_cap1(carrier_body[:cap1_bytes])
    selector_tail = SPARSE_SELECTOR_PREFIX + carrier_body[cap1_bytes:]
    try:
        selector, compensation = split_selector_compensation(selector_tail)
        decode_cap1(cap1, frames=600, dimensions=12)
        decode_selector(selector)
        carrier = pack_frame0_selector_carrier(cap1, selector)
    except (CoefficientAr1CodecError, ValueError) as error:
        raise ResidualArchiveError("invalid RX1 carrier") from error
    return semantic, carrier, hpac, compensation, outer[model_end:], outer[:model_end]


'''
    if source.count(helper_marker) != 1:
        raise RuntimeError("RX1 runtime helper insertion point differs")
    source = source.replace(helper_marker, helper + helper_marker)
    old = """    split = _decode_hp4_models(outer)
    if split is None:
        split = _decode_split_models(outer)
    if split is not None:
        models, section = split
        compressed = outer[: len(outer) - len(section)]
    else:
        try:
            decoder = lzma.LZMADecompressor(
                format=lzma.FORMAT_RAW,
                filters=LZMA_FILTERS,
            )
            models = decoder.decompress(outer)
        except lzma.LZMAError as error:
            raise ResidualArchiveError("invalid F24S model section") from error
        section = decoder.unused_data
        if not decoder.eof or not section:
            raise ResidualArchiveError("truncated F24S model section")
        compressed = outer[: len(outer) - len(section)]
    semantic, carrier, hpac, compensation = _decode_models(models)
"""
    new = """    rx1 = _decode_rx1_models(outer)
    if rx1 is not None:
        semantic, carrier, hpac, compensation, section, compressed = rx1
    else:
        split = _decode_hp4_models(outer)
        if split is None:
            split = _decode_split_models(outer)
        if split is not None:
            models, section = split
            compressed = outer[: len(outer) - len(section)]
        else:
            try:
                decoder = lzma.LZMADecompressor(
                    format=lzma.FORMAT_RAW,
                    filters=LZMA_FILTERS,
                )
                models = decoder.decompress(outer)
            except lzma.LZMAError as error:
                raise ResidualArchiveError("invalid F24S model section") from error
            section = decoder.unused_data
            if not decoder.eof or not section:
                raise ResidualArchiveError("truncated F24S model section")
            compressed = outer[: len(outer) - len(section)]
        semantic, carrier, hpac, compensation = _decode_models(models)
"""
    if source.count(old) != 1:
        raise RuntimeError("RX1 runtime read path insertion point differs")
    return source.replace(old, new)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    preflight(args)
    model_root = args.output / "retained" / "models" / "tq1c"
    tq1c_xz = args.tq1c_xz.read_bytes()
    atomic_bytes(model_root / "hpac.ihs1.xz", tq1c_xz)
    ihs1 = lzma.decompress(tq1c_xz, format=lzma.FORMAT_XZ)
    if not ihs1.startswith(b"IHS1"):
        raise RuntimeError("tq1c payload does not restore canonical IHS1")
    ihs1_path = model_root / "hpac.ihs1.raw"
    atomic_bytes(ihs1_path, ihs1)
    representations = [
        {
            "name": "xz_custodied",
            "codec": "xz",
            "codec_id": RX1_CODEC_XZ,
            "payload": file_record(model_root / "hpac.ihs1.xz"),
            "parseback_sha256": sha256_bytes(lzma.decompress(tq1c_xz, format=lzma.FORMAT_XZ)),
        }
    ]
    for quality in range(12):
        payload = _brotli(ihs1, quality, args.brotli)
        path = model_root / f"hpac.ihs1.br.q{quality}"
        atomic_bytes(path, payload)
        restored = _brotli_restore(payload, args.brotli)
        if restored != ihs1:
            raise RuntimeError(f"Brotli q{quality} IHS1 parse-back differs")
        representations.append(
            {
                "name": f"brotli_q{quality}",
                "codec": "brotli",
                "codec_id": RX1_CODEC_BROTLI,
                "payload": file_record(path),
                "parseback_sha256": sha256_bytes(restored),
            }
        )

    runtime_module = cp.load_runtime(args.runtime)
    parts = runtime_module.read_residual_archive(args.archive)
    base_root = args.output / "retained" / "base_parse"
    base_records = {
        "hpac": atomic_bytes(base_root / "hpac.bin", parts.hpac_blob),
        "semantic": atomic_bytes(base_root / "semantic.bin", parts.semantic_blob),
        "carrier": atomic_bytes(base_root / "carrier.bin", parts.carrier_blob),
        "residual": atomic_bytes(base_root / "residual.rcf1", parts.residual_payload),
        "tokens": atomic_bytes(base_root / "tokens.rc64", parts.token_stream),
    }
    fields, section = _hp4_fields(args.archive)
    if len(fields) != 5:
        raise RuntimeError("MC36 HP4 field count differs")
    semantic_stream, carrier_stream = fields[3], fields[4]
    physical_root = args.output / "retained" / "models" / "mc36_frozen"
    semantic_stream_path = physical_root / "semantic.br"
    carrier_stream_path = physical_root / "carrier.br"
    atomic_bytes(semantic_stream_path, semantic_stream)
    atomic_bytes(carrier_stream_path, carrier_stream)
    semantic_body = _brotli_restore(semantic_stream, args.brotli)
    carrier_body = _brotli_restore(carrier_stream, args.brotli)
    semantic_body_path = physical_root / "semantic.physical.raw"
    carrier_body_path = physical_root / "carrier.physical.raw"
    atomic_bytes(semantic_body_path, semantic_body)
    atomic_bytes(carrier_body_path, carrier_body)
    section_path = base_root / "residual_and_tokens.compact"
    atomic_bytes(section_path, section)

    adapted = args.output / "adapted_runtime"
    if not adapted.exists():
        shutil.copytree(args.runtime, adapted)
    residual_archive = adapted / "runtime" / "residual_archive.py"
    patched = _patch_runtime(residual_archive.read_text())
    atomic_bytes(residual_archive, patched.encode())
    if "# DDM_RX1_IHS1_CONTAINER_V1" not in residual_archive.read_text():
        raise RuntimeError("RX1 adapted runtime patch is absent")

    winner = min(representations, key=lambda row: row["payload"]["bytes"])
    smoke_root = args.output / "retained" / "receiver_smoke"
    smoke_hpac = Path(winner["payload"]["path"]).read_bytes()
    smoke_model = pack_rx1_model(
        smoke_hpac,
        semantic_stream,
        carrier_stream,
        codec_id=int(winner["codec_id"]),
        table_mode=RX1_TABLE_ON,
    )
    smoke_member = smoke_model + section
    smoke_archive = deterministic_zip(smoke_member)
    smoke_records = {
        "model": atomic_bytes(smoke_root / "models.rx1m", smoke_model),
        "member": atomic_bytes(smoke_root / "p", smoke_member),
        "archive": atomic_bytes(smoke_root / "archive.zip", smoke_archive),
    }
    smoke_parseback = _receiver_parseback_subprocess(adapted, smoke_root / "archive.zip", brotli_binary=args.brotli)
    expected_smoke = {
        "hpac_sha256": sha256_bytes(ihs1),
        "semantic_sha256": sha256_bytes(parts.semantic_blob),
        "carrier_sha256": sha256_bytes(parts.carrier_blob),
        "residual_sha256": sha256_bytes(parts.residual_payload),
        "token_sha256": sha256_bytes(parts.token_stream),
    }
    if any(smoke_parseback[name] != digest for name, digest in expected_smoke.items()):
        raise RuntimeError("RX1 receiver smoke parse-back differs from retained inputs")
    smoke_receipt = {
        "schema": "ddm_rx1_receiver_smoke.v1",
        "archive": smoke_records["archive"],
        "model": smoke_records["model"],
        "member": smoke_records["member"],
        "parseback": smoke_parseback,
        "expected": expected_smoke,
        "complete": True,
    }
    atomic_json(smoke_root / "RECEIVER_SMOKE.json", smoke_receipt)
    result = {
        "schema": "ddm_rx1_prepare.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "base": file_record(args.archive),
        "tq1c_ihs1": file_record(ihs1_path),
        "representations": representations,
        "representation_winner": winner,
        "base_parse": base_records,
        "semantic_stream": file_record(semantic_stream_path),
        "carrier_stream": file_record(carrier_stream_path),
        "semantic_physical": file_record(semantic_body_path),
        "carrier_physical": file_record(carrier_body_path),
        "compact_section": file_record(section_path),
        "adapted_runtime_residual_archive": file_record(residual_archive),
        "receiver_smoke": smoke_receipt,
        "all_materialized_payloads_retained": True,
        "complete": True,
    }
    atomic_json(args.output / "PREPARE_RESULT.json", result)
    return result


def spatial_frame(events: np.ndarray, group_positions: list[np.ndarray]) -> np.ndarray:
    flat = np.empty(EVENTS_PER_FRAME, dtype=np.uint8)
    offset = 0
    for positions in group_positions:
        end = offset + len(positions)
        flat[positions] = events[offset:end]
        offset = end
    if offset != EVENTS_PER_FRAME:
        raise RuntimeError("group positions do not consume one token frame")
    return flat.reshape(384, 512)


def _frame_record(path: Path, frame: int, variant: Variant) -> dict[str, Any]:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.dtype != np.int16 or value.shape != (EVENTS_PER_FRAME, 5):
        raise RuntimeError(f"invalid retained RX1 probability frame: {path}")
    return {"frame": frame, "variant": variant, "codes": file_record(path), "complete": True}


def export_probabilities(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    require_prepared(args)
    variant: Variant = args.variant
    runtime_module = cp.load_runtime(args.runtime)
    archive_module = importlib.import_module("runtime.residual_archive")
    inference_module = importlib.import_module("runtime.hpac_inference")
    parts = runtime_module.read_residual_archive(args.archive)
    renderer = runtime_module._load_renderer(args.runtime / "cpr1")
    ihs1_path = args.output / "retained" / "models" / "tq1c" / "hpac.ihs1.raw"
    ihs1 = ihs1_path.read_bytes()
    model = renderer.load_hpac(ihs1, torch.device("cpu"))
    masks = renderer.group_masks(torch.device("cpu"))
    sparse = archive_module._sparse_class(args.runtime / "cpr1")(model, renderer.EVAL_H, renderer.EVAL_W)
    inference_module.optimize_sparse_evaluator(sparse)
    group_positions = [np.flatnonzero(mask.detach().cpu().numpy().reshape(-1)) for mask in masks]
    source = _source(args)
    output = args.output / "retained" / "probabilities" / variant
    output.mkdir(parents=True, exist_ok=True)
    binding = {
        "schema": "ddm_rx1_probability_source_binding.v1",
        "variant": variant,
        "archive": file_record(args.archive),
        "tq1c_ihs1": file_record(ihs1_path),
        "source_manifest": file_record(args.source_manifest),
        "source_event_order_sha256": source.digest(),
        "table_mode": "mc36_table" if variant == "tq1c_table_on" else "neutral_table",
    }
    binding_path = output / "SOURCE_BINDING.json"
    if binding_path.is_file() and json.loads(binding_path.read_text()) != binding:
        raise RuntimeError("RX1 probability output is bound to different inputs")
    atomic_json(binding_path, binding)
    started = time.time()
    with torch.inference_mode():
        for frame in range(args.start_frame, args.end_frame):
            path = output / f"codes_{frame:04d}.npy"
            receipt = output / f"codes_{frame:04d}.json"
            if path.is_file() and receipt.is_file():
                if json.loads(receipt.read_text()) != _frame_record(path, frame, variant):
                    raise RuntimeError(f"RX1 probability checkpoint changed: {receipt}")
                print(json.dumps({"variant": variant, "frame": frame + 1, "status": "reused"}), flush=True)
                continue
            events = source.frame(frame)
            previous_events = np.zeros(EVENTS_PER_FRAME, dtype=np.uint8) if frame == 0 else source.frame(frame - 1)
            previous_np = (
                np.zeros((384, 512), dtype=np.uint8) if frame == 0 else spatial_frame(previous_events, group_positions)
            )
            previous = torch.from_numpy(previous_np.astype(np.int64, copy=False))[None]
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(torch.tensor([frame]), previous)
            boundary = (
                np.full(EVENTS_PER_FRAME, 4, dtype=np.uint8)
                if frame == 0
                else archive_module._boundary_buckets(previous_np).reshape(-1)
            )
            frame_codes = np.empty((EVENTS_PER_FRAME, 5), dtype=np.int16)
            symbol_offset = 0
            for group, (_mask, positions) in enumerate(zip(masks, group_positions, strict=True)):
                selected = sparse.selected_logits(current, context, group)
                base_logits = selected.cpu().numpy()
                if variant == "tq1c_table_on":
                    predicted = base_logits.argmax(axis=1).astype(np.int64)
                    feature = boundary[positions].astype(np.int64) * 5 + predicted
                    corrected = base_logits + parts.table.values[feature]
                else:
                    corrected = base_logits
                codes = np.clip(
                    np.rint(np.asarray(corrected, dtype=np.float32) * renderer.HPAC_LOGIT_PRECISION),
                    -32768,
                    32767,
                ).astype(np.int16)
                end = symbol_offset + len(positions)
                symbols = events[symbol_offset:end]
                current.reshape(-1)[torch.from_numpy(positions)] = torch.from_numpy(
                    symbols.astype(np.int64, copy=False)
                )
                frame_codes[symbol_offset:end] = codes
                symbol_offset = end
            if symbol_offset != EVENTS_PER_FRAME:
                raise RuntimeError("RX1 probability export did not consume one frame")
            if not np.array_equal(current[0].numpy(), spatial_frame(events, group_positions)):
                raise RuntimeError("RX1 teacher-forced reconstruction differs")
            atomic_npy(path, frame_codes)
            record = _frame_record(path, frame, variant)
            atomic_json(receipt, record)
            print(
                json.dumps(
                    {
                        "variant": variant,
                        "frame": frame + 1,
                        "codes_sha256": record["codes"]["sha256"],
                        "elapsed_s": round(time.time() - started, 3),
                    }
                ),
                flush=True,
            )
    completed = []
    for frame in range(600):
        path = output / f"codes_{frame:04d}.npy"
        receipt = output / f"codes_{frame:04d}.json"
        if path.is_file() and receipt.is_file():
            completed.append(_frame_record(path, frame, variant))
    identity = {
        "schema": "ddm_rx1_probability_identity.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "variant": variant,
        "source_binding": file_record(binding_path),
        "completed_frames": len(completed),
        "complete_n600": len(completed) == 600,
        "frames": completed,
    }
    identity_path = output / "PROBABILITY_IDENTITY.json"
    atomic_json(identity_path, identity)
    result = {**identity, "probability_identity": file_record(identity_path), "wall_s": time.time() - started}
    atomic_json(output / "EXPORT_RESULT.json", result)
    return result


def _load_codes(output: Path, variant: Variant, frame: int) -> np.ndarray:
    path = output / "retained" / "probabilities" / variant / f"codes_{frame:04d}.npy"
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.dtype != np.int16 or value.shape != (EVENTS_PER_FRAME, 5):
        raise RuntimeError(f"invalid RX1 probability codes: {path}")
    return np.asarray(value)


def encode_rc64(args: argparse.Namespace) -> dict[str, Any]:
    require_prepared(args)
    variant: Variant = args.variant
    export_path = args.output / "retained" / "probabilities" / variant / "EXPORT_RESULT.json"
    export = json.loads(export_path.read_text())
    if not export.get("complete_n600"):
        raise RuntimeError(f"RX1 probability export is incomplete for {variant}")
    identity = export["probability_identity"]
    identity_path = Path(identity["path"])
    if file_record(identity_path) != identity:
        raise RuntimeError("RX1 probability identity failed custody")
    source = _source(args)
    library_path = cp._compile_checkpointable_rc64(args)
    sys.path.insert(0, str(args.experiment_book / "src"))
    try:
        from cpr1_sub4.entropy.rc64 import NativeDecoder, NativeEncoder
    finally:
        sys.path.pop(0)
    retained = args.output / "retained" / "coders" / variant
    checkpoint_root = retained / "rc64_checkpoints"
    progress_path = checkpoint_root / "LATEST.json"
    start_frame = 0
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if progress.get("probability_identity") != identity:
            raise RuntimeError("RX1 RC64 checkpoint is bound to different probabilities")
        state_path = Path(progress["state"]["path"])
        if file_record(state_path) != progress["state"]:
            raise RuntimeError("RX1 RC64 checkpoint failed custody")
        encoder = cp._rc64_resume(NativeEncoder, library_path, state_path.read_bytes())
        start_frame = int(progress["next_frame"])
    else:
        encoder = NativeEncoder(library_path)
    started = time.time()
    for frame in range(start_frame, 600):
        probabilities = cp.probability_from_codes(_load_codes(args.output, variant, frame), 8)
        encoder.encode(source.frame(frame).astype(np.int32), probabilities)
        if (frame + 1) % 24 == 0 or frame == 599:
            state = cp._rc64_snapshot(encoder)
            state_path = checkpoint_root / f"through_frame_{frame:04d}.rc64.state"
            atomic_bytes(state_path, state)
            receipt = {
                "schema": "ddm_rx1_rc64_checkpoint.v1",
                "variant": variant,
                "through_frame": frame,
                "next_frame": frame + 1,
                "state": file_record(state_path),
                "probability_identity": identity,
            }
            atomic_json(state_path.with_suffix(".json"), receipt)
            atomic_json(progress_path, receipt)
            print(
                json.dumps(
                    {"variant": variant, "rc64_encoded_frames": frame + 1, "elapsed_s": round(time.time() - started, 3)}
                ),
                flush=True,
            )
    payload = encoder.finish()
    token_path = retained / "tokens.rc64"
    atomic_bytes(token_path, payload)
    encode_wall_s = time.time() - started

    decoder = NativeDecoder(library_path, token_path.read_bytes())
    event_path = retained / "decoded_symbols.rc64.bin"
    spatial_path = retained / "decoded_spatial_tokens.rc64.bin"
    event_tmp = event_path.with_name(f".{event_path.name}.{os.getpid()}.tmp")
    spatial_tmp = spatial_path.with_name(f".{spatial_path.name}.{os.getpid()}.tmp")
    event_digest = hashlib.sha256()
    spatial_digest = hashlib.sha256()
    group_positions = cp._group_positions(args.runtime)
    decode_started = time.time()
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_tmp.open("wb") as event_stream, spatial_tmp.open("wb") as spatial_stream:
        for frame in range(600):
            probabilities = cp.probability_from_codes(_load_codes(args.output, variant, frame), 8)
            decoded = decoder.decode(probabilities).astype(np.uint8)
            expected = source.frame(frame)
            if not np.array_equal(decoded, expected):
                raise RuntimeError(f"RX1 RC64 symbol mismatch at frame {frame}")
            event_raw = decoded.tobytes()
            spatial_raw = spatial_frame(decoded, group_positions).tobytes()
            event_stream.write(event_raw)
            spatial_stream.write(spatial_raw)
            event_digest.update(event_raw)
            spatial_digest.update(spatial_raw)
        event_stream.flush()
        os.fsync(event_stream.fileno())
        spatial_stream.flush()
        os.fsync(spatial_stream.fileno())
    os.replace(event_tmp, event_path)
    os.replace(spatial_tmp, spatial_path)
    if event_digest.hexdigest() != EXPECTED_EVENT_SHA256 or spatial_digest.hexdigest() != EXPECTED_SPATIAL_SHA256:
        raise RuntimeError("RX1 RC64 decoded-token digest differs")
    result = {
        "schema": "ddm_rx1_rc64_result.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "variant": variant,
        "token_payload": file_record(token_path),
        "decoded_symbols": file_record(event_path),
        "decoded_spatial_tokens": file_record(spatial_path),
        "event_order_identity": True,
        "spatial_token_identity": True,
        "zero_token_distortion": True,
        "probability_identity": identity,
        "resumable_from_disk": True,
        "checkpoint_count": len(list(checkpoint_root.glob("through_frame_*.rc64.state"))),
        "decoder_bit_position": decoder.bit_position,
        "encode_wall_s": encode_wall_s,
        "decode_wall_s": time.time() - decode_started,
    }
    atomic_json(retained / "RC64_RESULT.json", result)
    return result


def _neutral_residual(compact: bytes) -> bytes:
    if len(compact) != 96:
        raise RuntimeError("MC36 compact residual length differs")
    return compact[:2] + bytes(len(compact) - 2)


def pack_rx1_model(
    hpac_stream: bytes,
    semantic_stream: bytes,
    carrier_stream: bytes,
    *,
    codec_id: int,
    table_mode: int,
) -> bytes:
    if codec_id not in (RX1_CODEC_XZ, RX1_CODEC_BROTLI) or table_mode not in (RX1_TABLE_ON, RX1_TABLE_OFF):
        raise ValueError("invalid RX1 model mode")
    if min(len(hpac_stream), len(semantic_stream), len(carrier_stream)) <= 0:
        raise ValueError("RX1 streams must be non-empty")
    if max(len(hpac_stream), len(semantic_stream), len(carrier_stream)) >= 1 << 16:
        raise ValueError("RX1 stream exceeds u16 length")
    return (
        RX1_HEADER.pack(
            RX1_MAGIC,
            RX1_VERSION,
            codec_id,
            table_mode,
            0,
            len(hpac_stream),
            len(semantic_stream),
            len(carrier_stream),
        )
        + hpac_stream
        + semantic_stream
        + carrier_stream
    )


def unpack_rx1_model(value: bytes, *, brotli_binary: str) -> dict[str, Any]:
    if len(value) < RX1_HEADER.size:
        raise ValueError("truncated RX1 model")
    magic, version, codec_id, table_mode, reserved, a, b, c = RX1_HEADER.unpack_from(value)
    if magic != RX1_MAGIC or version != RX1_VERSION or reserved != 0:
        raise ValueError("invalid RX1 model header")
    if codec_id not in (RX1_CODEC_XZ, RX1_CODEC_BROTLI) or table_mode not in (RX1_TABLE_ON, RX1_TABLE_OFF):
        raise ValueError("invalid RX1 model mode")
    if min(a, b, c) <= 0 or RX1_HEADER.size + a + b + c != len(value):
        raise ValueError("invalid RX1 model field accounting")
    offset = RX1_HEADER.size
    hpac_stream = value[offset : offset + a]
    semantic_stream = value[offset + a : offset + a + b]
    carrier_stream = value[offset + a + b :]
    hpac = (
        lzma.decompress(hpac_stream, format=lzma.FORMAT_XZ)
        if codec_id == RX1_CODEC_XZ
        else _brotli_restore(hpac_stream, brotli_binary)
    )
    return {
        "codec_id": codec_id,
        "table_mode": table_mode,
        "hpac": hpac,
        "semantic_stream": semantic_stream,
        "carrier_stream": carrier_stream,
    }


def _load_adapted_runtime(root: Path):
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module("runtime.f26_inflate")
    finally:
        sys.path.pop(0)


def _receiver_parseback_subprocess(runtime_root: Path, archive: Path, *, brotli_binary: str) -> dict[str, Any]:
    code = """
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from runtime.f26_inflate import read_residual_archive
parts = read_residual_archive(Path(sys.argv[2]))
def digest(value): return hashlib.sha256(value).hexdigest()
print(json.dumps({
    "hpac_sha256": digest(parts.hpac_blob),
    "semantic_sha256": digest(parts.semantic_blob),
    "carrier_sha256": digest(parts.carrier_blob),
    "residual_sha256": digest(parts.residual_payload),
    "token_sha256": digest(parts.token_stream),
    "table_nonzero_codes": int((parts.table.codes != 0).sum()),
}))
"""
    environment = dict(os.environ)
    environment["CP135_BROTLI_CLI"] = brotli_binary
    completed = subprocess.run(
        [sys.executable, "-c", code, str(runtime_root), str(archive)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode:
        raise RuntimeError(f"RX1 shipped receiver parse-back failed: {completed.stderr}")
    return json.loads(completed.stdout)


def build_candidates(args: argparse.Namespace) -> dict[str, Any]:
    prepared = require_prepared(args)
    variant: Variant = args.variant
    rc64_path = args.output / "retained" / "coders" / variant / "RC64_RESULT.json"
    rc64 = json.loads(rc64_path.read_text())
    if not rc64.get("event_order_identity") or not rc64.get("spatial_token_identity"):
        raise RuntimeError("RX1 RC64 identity gate is not closed")
    token_path = Path(rc64["token_payload"]["path"])
    if file_record(token_path) != rc64["token_payload"]:
        raise RuntimeError("RX1 RC64 payload failed custody")
    token = token_path.read_bytes()
    semantic_stream = Path(prepared["semantic_stream"]["path"]).read_bytes()
    carrier_stream = Path(prepared["carrier_stream"]["path"]).read_bytes()
    base_section = Path(prepared["compact_section"]["path"]).read_bytes()
    base_residual = base_section[:96]
    residual = base_residual if variant == "tq1c_table_on" else _neutral_residual(base_residual)
    table_mode = RX1_TABLE_ON if variant == "tq1c_table_on" else RX1_TABLE_OFF
    ihs1 = Path(prepared["tq1c_ihs1"]["path"]).read_bytes()
    adapted = _load_adapted_runtime(args.output / "adapted_runtime")
    rows = []
    for representation in prepared["representations"]:
        hpac_path = Path(representation["payload"]["path"])
        if file_record(hpac_path) != representation["payload"]:
            raise RuntimeError("RX1 model representation failed custody")
        model = pack_rx1_model(
            hpac_path.read_bytes(),
            semantic_stream,
            carrier_stream,
            codec_id=int(representation["codec_id"]),
            table_mode=table_mode,
        )
        unpacked = unpack_rx1_model(model, brotli_binary=args.brotli)
        if unpacked["hpac"] != ihs1:
            raise RuntimeError("RX1 model parse-back differs")
        member = model + residual + token
        archive = deterministic_zip(member)
        repeat = deterministic_zip(member)
        root = args.output / "retained" / "candidates" / variant / representation["name"]
        records = {
            "model": atomic_bytes(root / "models.rx1m", model),
            "residual": atomic_bytes(root / "residual.compact.bin", residual),
            "token": atomic_bytes(root / "tokens.rc64", token),
            "member": atomic_bytes(root / "p", member),
            "archive": atomic_bytes(root / "archive.zip", archive),
            "repeat_archive": atomic_bytes(root / "archive.repeat.zip", repeat),
        }
        if archive != repeat or read_stored_member(root / "archive.zip") != member:
            raise RuntimeError("RX1 deterministic archive parse-back differs")
        parsed = adapted.read_residual_archive(root / "archive.zip")
        if parsed.hpac_blob != ihs1 or parsed.token_stream != token:
            raise RuntimeError("RX1 shipped receiver HPAC/token parse-back differs")
        if variant == "tq1c_table_off" and np.any(parsed.table.codes):
            raise RuntimeError("RX1 neutral table parsed with non-zero codes")
        if variant == "tq1c_table_on" and parsed.residual_payload[4:] != residual:
            raise RuntimeError("RX1 MC36 table parse-back differs")
        archive_bytes = records["archive"]["bytes"]
        row = {
            "variant": variant,
            "representation": representation["name"],
            "archive": records["archive"],
            "repeat_archive": records["repeat_archive"],
            "repeat_byte_identical": archive == repeat,
            "member": records["member"],
            "model": records["model"],
            "residual": records["residual"],
            "token": records["token"],
            "receiver_hpac_identity": True,
            "receiver_token_payload_identity": True,
            "decoded_event_order_identity": True,
            "decoded_spatial_token_identity": True,
            "delta_distortion": 0.0,
            "archive_delta_vs_mc36": archive_bytes - EXPECTED_ARCHIVE_BYTES,
            "projected_score_if_mc36_distortion_held": (
                MC36_SCORE + 25.0 * (archive_bytes - EXPECTED_ARCHIVE_BYTES) / RATE_DENOMINATOR
            ),
            "axis": AXIS,
            "score_claim": SCORE_CLAIM,
        }
        atomic_json(root / "RESULT.json", row)
        rows.append(row)
    winner = min(rows, key=lambda row: row["archive"]["bytes"])
    result = {
        "schema": "ddm_rx1_build.v1",
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "variant": variant,
        "base": file_record(args.archive),
        "candidates": rows,
        "winner": winner,
        "candidate_count": len(rows),
        "all_payloads_retained": True,
        "whole_container_recount": True,
        "receiver_parseback": True,
        "zero_token_distortion": True,
    }
    atomic_json(args.output / f"BUILD_RESULT_{variant}.json", result)
    return result


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    prepared = require_prepared(args)
    builds = []
    for variant in VARIANTS:
        path = args.output / f"BUILD_RESULT_{variant}.json"
        if not path.is_file():
            raise RuntimeError(f"RX1 build is incomplete for {variant}")
        builds.append(json.loads(path.read_text()))
    candidates = [row for build in builds for row in build["candidates"]]
    winner = min(candidates, key=lambda row: row["archive"]["bytes"])
    cpu_receipt_path = args.output / "retained" / "cpu_decode" / "best_rx1" / "receipts" / "CPU_DECODE_RESULT.json"
    if not cpu_receipt_path.is_file():
        raise RuntimeError("RX1 finalize requires the lifted CPU raw-output receipt")
    cpu_receipt = json.loads(cpu_receipt_path.read_text())
    if not cpu_receipt.get("raw_identity_vs_mc36_cpu") or not cpu_receipt.get("decoded_token_identity"):
        raise RuntimeError("RX1 lifted CPU identity gate is incomplete")
    admitted = winner["archive"]["bytes"] < EXPECTED_ARCHIVE_BYTES
    tq1c_original_tokens = Path("/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/artifacts/tq1c/tokens.bin")
    original_token_record = file_record(tq1c_original_tokens) if tq1c_original_tokens.is_file() else None
    inventory = retention_inventory(args.output)
    inventory_record = atomic_json(args.output / "RETENTION_INVENTORY.json", inventory)
    result = {
        "schema": "ddm_rx1_final.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": SCORE_CLAIM,
        "base_mc36": {
            "archive": file_record(args.archive),
            "exact_score": MC36_SCORE,
            "authority": "[contest-CUDA] n600 T4",
        },
        "tq1c_historical_token_payload": original_token_record,
        "model_representation_race": prepared["representations"],
        "builds": builds,
        "winner": winner,
        "winner_admitted_vs_mc36": admitted,
        "measured_archive_delta_bytes": winner["archive_delta_vs_mc36"],
        "measured_delta_distortion": 0.0,
        "projected_score_not_authority": winner["projected_score_if_mc36_distortion_held"],
        "all_payloads_retained": True,
        "retention_inventory": {
            **inventory_record,
            "file_count": inventory["file_count"],
            "total_bytes": inventory["total_bytes"],
        },
        "receiver_parseback": True,
        "decoded_token_identity": True,
        "local_rgb_raw_decode": cpu_receipt,
        "main_t4_fire_order": {
            "sealed": True,
            "owner": "MAIN",
            "consumer_store": "main_hot_state.md plus canonical_frontier_pointer.json",
            "disposition": "FIRE" if admitted else "DO_NOT_FIRE",
            "trigger": "a receiver-valid archive is strictly smaller than MC36 and passes lifted CPU raw identity",
            "reason": (
                "RX1 winner is strictly smaller than MC36"
                if admitted
                else "all RX1 candidates are larger than MC36, so exact evaluation cannot improve the frontier"
            ),
            "command_template": (
                "MAIN-owned governed T4 upstream/evaluate.py on the exact retained winner archive" if admitted else None
            ),
        },
    }
    atomic_json(args.output / "FINAL_RESULT.json", result)
    return result


def cpu_decode(args: argparse.Namespace) -> dict[str, Any]:
    """Run the smallest RX1 archive through the retained four-thread CPU lift."""

    builds = []
    for variant in VARIANTS:
        path = args.output / f"BUILD_RESULT_{variant}.json"
        if not path.is_file():
            raise RuntimeError(f"RX1 build is incomplete for {variant}")
        builds.append(json.loads(path.read_text()))
    winner = min((build["winner"] for build in builds), key=lambda row: row["archive"]["bytes"])
    candidate = Path(winner["archive"]["path"])
    if file_record(candidate) != winner["archive"]:
        raise RuntimeError("RX1 CPU-decode candidate failed custody")
    if not F26P_LIFTED_RUNTIME.is_dir():
        raise RuntimeError("custodied F26P lifted CPU runtime is absent")

    root = args.output / "retained" / "cpu_decode" / "best_rx1"
    runtime_root = root / "lifted_submission_cpu"
    input_root = root / "input"
    output_root = root / "output"
    receipt_root = root / "receipts"
    log_root = root / "logs"
    file_list = root / "file_list.txt"
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    required = EXPECTED_CPU_RAW_BYTES + 300_000_000
    if usage.free < required:
        raise RuntimeError(f"RX1 CPU decode needs {required} free bytes but only {usage.free} remain")
    if not runtime_root.exists():
        shutil.copytree(F26P_LIFTED_RUNTIME, runtime_root)

    atomic_bytes(runtime_root / "archive.zip", candidate.read_bytes())
    receiver_path = runtime_root / "runtime" / "residual_archive.py"
    receiver_source = _patch_runtime(receiver_path.read_text())
    atomic_bytes(receiver_path, receiver_source.encode())
    entrypoint = runtime_root / "inflate.py"
    entry_source = entrypoint.read_text()
    entry_source, sha_count = re.subn(
        r'^ARCHIVE_SHA256 = "[0-9a-f]{64}"$',
        f'ARCHIVE_SHA256 = "{winner["archive"]["sha256"]}"',
        entry_source,
        count=1,
        flags=re.MULTILINE,
    )
    entry_source, byte_count = re.subn(
        r"^ARCHIVE_BYTES = [0-9_]+$",
        f"ARCHIVE_BYTES = {winner['archive']['bytes']:_}",
        entry_source,
        count=1,
        flags=re.MULTILINE,
    )
    if (sha_count, byte_count) != (1, 1):
        raise RuntimeError("RX1 CPU entrypoint archive pins could not be updated exactly")
    atomic_bytes(entrypoint, entry_source.encode(), executable=True)
    member = read_stored_member(candidate)
    atomic_bytes(input_root / "p", member)
    atomic_bytes(file_list, b"0.mkv\n")

    raw_path = output_root / "0.raw"
    receipt_path = receipt_root / "CPU_DECODE_RESULT.json"
    if receipt_path.is_file() and raw_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("complete") and file_record(raw_path) == receipt["raw_output"]:
            return receipt
        raise RuntimeError("existing RX1 CPU decode differs from its receipt")
    if raw_path.exists():
        raise RuntimeError("RX1 CPU raw output exists without a completion receipt")

    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / "decode.log"
    command = [str(runtime_root / "inflate.sh"), str(input_root), str(output_root), str(file_list)]
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "VECLIB_MAXIMUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
            "PATH": os.pathsep.join([str(Path(sys.executable).parent), environment.get("PATH", "")]),
        }
    )
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "candidate": winner["archive"]}) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        failure = {
            "schema": "ddm_rx1_cpu_decode_failure.v1",
            "complete": False,
            "return_code": return_code,
            "candidate": winner["archive"],
            "log": file_record(log_path),
        }
        atomic_json(receipt_root / "CPU_DECODE_FAILURE.json", failure)
        raise RuntimeError(f"RX1 lifted CPU decoder exited {return_code}; payloads retained")
    require_file(raw_path, size=EXPECTED_CPU_RAW_BYTES, digest=EXPECTED_CPU_RAW_SHA256)
    report = None
    for line in reversed(log_path.read_text().splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("schema") == "ddm_f26p_inflate_report.v1":
            report = value
            break
    if report is None or report.get("raw_sha256") != EXPECTED_CPU_RAW_SHA256:
        raise RuntimeError("RX1 CPU decode log lacks the exact raw-output receipt")
    result = {
        "schema": "ddm_rx1_cpu_decode.v1",
        "complete": True,
        "axis": "[macOS-CPU advisory, four-thread lifted F26]",
        "score_claim": False,
        "candidate": winner["archive"],
        "adapted_runtime_receiver": file_record(receiver_path),
        "adapted_runtime_entrypoint": file_record(entrypoint),
        "input_member": file_record(input_root / "p"),
        "raw_output": file_record(raw_path),
        "expected_mc36_cpu_raw_sha256": EXPECTED_CPU_RAW_SHA256,
        "raw_identity_vs_mc36_cpu": True,
        "decoded_token_sha256": report["token_decoder"]["decoded_token_sha256"],
        "decoded_token_identity": report["token_decoder"]["decoded_token_sha256"] == EXPECTED_SPATIAL_SHA256,
        "checkpoint_resume": report["checkpoint_resume"],
        "checkpoint_dir": report["checkpoint_dir"],
        "wall_seconds": time.time() - started,
        "inflate_report": report,
        "log": file_record(log_path),
    }
    if not result["decoded_token_identity"]:
        raise RuntimeError("RX1 lifted CPU decoder token identity differs")
    atomic_json(receipt_path, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("stage", choices=("preflight", "prepare", "export", "encode", "build", "cpu-decode", "finalize"))
    value.add_argument("--variant", choices=VARIANTS, default="tq1c_table_on")
    value.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    value.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    value.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    value.add_argument("--expected-spatial", type=Path, default=DEFAULT_EXPECTED_SPATIAL)
    value.add_argument("--tq1c-xz", type=Path, default=DEFAULT_TQ1C_XZ)
    value.add_argument("--tq1c-checkpoint", type=Path, default=DEFAULT_TQ1C_CHECKPOINT)
    value.add_argument("--experiment-book", type=Path, default=DEFAULT_EXPERIMENT_BOOK)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--start-frame", type=int, default=0)
    value.add_argument("--end-frame", type=int, default=600)
    value.add_argument("--torch-threads", type=int, default=4)
    value.add_argument("--required-free-gib", type=int, default=4)
    value.add_argument("--brotli", default=shutil.which("brotli") or "brotli")
    return value


def main() -> None:
    args = parser().parse_args()
    if not 0 <= args.start_frame < args.end_frame <= 600:
        raise SystemExit("invalid frame interval")
    if args.stage == "export":
        import torch

        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)
        result = export_probabilities(args)
    elif args.stage == "preflight":
        result = preflight(args)
    elif args.stage == "prepare":
        result = prepare(args)
    elif args.stage == "encode":
        result = encode_rc64(args)
    elif args.stage == "build":
        result = build_candidates(args)
    elif args.stage == "cpu-decode":
        result = cpu_decode(args)
    else:
        result = finalize(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
