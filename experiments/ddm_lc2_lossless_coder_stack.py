#!/usr/bin/env python3
"""Compose and certify LC2's lossless coder stack on the AI1 candidate.

The input is the receiver-closed AI1 ANS + temporal-reversion archive.  LC2
changes only the lossless model-section representation: first to three
independent Brotli streams, then through CX2's reversible byte-coordinate
family selected on complete deterministic ZIP bytes.  The token payload,
temporal sidecar, and reconstructed PR130 model bytes remain exact.

Every section candidate and every complete-ZIP candidate is retained.  The
literal receiver decode is a separate resumable command so a completed build
and its 54 archived reference-form candidates survive interruption of the long
n600 run.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import hashlib
import importlib
import importlib.metadata
import io
import itertools
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import brotli
import torch

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "src" / "tac" / "pr130_runtime" / "dv1_cpu_runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))
receiver = importlib.import_module("receiver")
inflate = importlib.import_module("inflate")

AI1_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_ai1_20260809/temporal_v3")
AI1_ARCHIVE = AI1_ROOT / "retained" / "temporal_reversion" / "archive.zip"
AI1_BUILD_RECEIPT = AI1_ROOT / "temporal_reversion_build_receipt.json"
AI1_DETERMINISM_RECEIPT = AI1_ROOT / "temporal_reversion_determinism_receipt.json"
AI1_RAW = AI1_ROOT / "decode_temporal_reversion" / "a" / "0.raw"
DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_lc2_20260810")
DEFAULT_EXACT_EVAL_STORE = Path("/Volumes/APDataStore/pact/ddm_lc2_20260810/exact_eval")
DEFAULT_COLD_OUTPUT = Path("/Volumes/APDataStore/pact/ddm_lc2_20260810/cold_decode")
PINNED_PYTHON = Path("/Volumes/VertigoDataTier/pact/ddm_pq1_runtime_20260809/venv/bin/python")
VIDEO_NAMES = REPO / "upstream" / "public_test_video_names.txt"

EXPECTED_AI1_BYTES = 188_636
EXPECTED_AI1_SHA256 = "0f5a797fda844ee63f6057fdb7203f6578b135b4e12deafa98d6ddc3260a5c84"
EXPECTED_AI1_BUILD_RECEIPT_SHA256 = "984ba74fb54dd7bf20fe100bb8c50a8ae4d591bb4fac27db59bd690f63cb36be"
EXPECTED_AI1_DETERMINISM_RECEIPT_SHA256 = "ca700af7e1f76627e83bf9299417de032f8c795dc905326fa78d775d9001e23b"
EXPECTED_MODELS_RAW_SHA256 = "62dd72dfa0858a25ca32bdee1e536627a17883b6fc7efd7cd5b2de7b13b84517"
EXPECTED_MODELS_RAW_WIRE_SHA256 = "618ac80da2bfb82a52a94317877cfd79af71290f751e3d4f130a46258b29092a"
EXPECTED_TOKEN_BYTES = 114_528
EXPECTED_TOKEN_SHA256 = "85d6c199ffb93ddab0fe1631448882a255e9fea1f6858bab5a04cea2310a7331"
EXPECTED_TEMPORAL_PACKED_BYTES = 39
EXPECTED_TEMPORAL_PACKED_SHA256 = "f920f7be8108b83831971a8d07c9ef522eadb18abed095cf395bf3a6f871e796"
EXPECTED_DECODED_TOKEN_SHA256 = "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece"
EXPECTED_RAW_BYTES = 3_662_409_600
EXPECTED_RAW_SHA256 = "a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353"
EXPECTED_VIDEO_NAMES_SHA256 = "7ff99d08c8351dd8167ec09213b758da5bbb705dedabe361ba881217374029a8"
EXPECTED_FRAMES = 600
EXPECTED_TOKENS = 117_964_800
EXPECTED_ORIGINAL_BYTES = 37_545_489
PR130_CONTEST_CUDA_SCORE = 0.172141297491896447
PR130_ARCHIVE_BYTES = 191_052
RUNTIME_FILES = (
    "carrier_codec.py",
    "hpac_integer.py",
    "hpac_integer_sparse.py",
    "inflate.py",
    "inflate.sh",
    "integer_model_io.py",
    "receiver.py",
    "runtime-dependencies.json",
)
QUALITIES = (9, 10, 11)
ZIP_POLICIES = (("stored", None), ("deflate", 6))


@dataclasses.dataclass(frozen=True)
class SourceObject:
    semantic: bytes
    carrier: bytes
    hpac_wire: bytes
    tokens: bytes
    models_raw: bytes
    models_raw_wire: bytes
    temporal_packed: bytes


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def provenance() -> dict[str, Any]:
    return {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
        ).strip(),
        "producer": file_record(Path(__file__)),
        "python": sys.version,
        "brotli_python": importlib.metadata.version("brotli"),
        "numpy": importlib.metadata.version("numpy"),
        "torch": torch.__version__,
        "seed": None,
        "determinism": "byte transforms, compressors, and ZIP writer are deterministic",
    }


def atomic_bytes(path: Path, payload: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if executable:
        temporary.chmod(0o755)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )


def persist_exact(path: Path, payload: bytes) -> dict[str, Any]:
    """Persist one materialized payload without overwriting divergent evidence."""

    expected = {"bytes": len(payload), "sha256": sha256_bytes(payload)}
    if path.exists():
        actual = file_record(path)
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise RuntimeError(f"refusing to overwrite divergent retained payload: {path}")
        return actual
    atomic_bytes(path, payload)
    return file_record(path)


def require_file(
    path: Path,
    *,
    size: int | None = None,
    digest: str | None = None,
    label: str,
) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} is absent: {path}")
    if size is not None and path.stat().st_size != size:
        raise RuntimeError(f"{label} byte count differs from its pin")
    if digest is not None and sha256_file(path) != digest:
        raise RuntimeError(f"{label} SHA-256 differs from its pin")


def read_stored_member(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) != 1 or entries[0].filename != "p":
            raise RuntimeError("archive.zip must contain exactly one member named p")
        info = entries[0]
        if (
            info.is_dir()
            or info.compress_type != zipfile.ZIP_STORED
            or info.file_size != info.compress_size
            or info.flag_bits & 0x1
        ):
            raise RuntimeError("archive member p is not an unencrypted stored file")
        payload = archive.read(info)
        if archive.testzip() is not None:
            raise RuntimeError("archive.zip failed CRC validation")
        return payload


def read_archive_member_bytes(archive_bytes: bytes) -> bytes:
    """Read the sole payload member from deterministic ZIP bytes."""

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        entries = archive.infolist()
        if len(entries) != 1 or entries[0].filename != "p" or entries[0].is_dir():
            raise RuntimeError("archive bytes must contain exactly one file member named p")
        if entries[0].flag_bits & 0x1:
            raise RuntimeError("archive member p must not be encrypted")
        payload = archive.read(entries[0])
        if archive.testzip() is not None:
            raise RuntimeError("archive bytes failed CRC validation")
        return payload


def split_pack(streams: tuple[bytes, bytes, bytes]) -> bytes:
    if any(not stream for stream in streams):
        raise ValueError("split pack requires three non-empty streams")
    return struct.pack("<III", *(len(stream) for stream in streams)) + b"".join(streams)


def deterministic_zip(
    member: bytes,
    *,
    codec: str = "stored",
    level: int | None = None,
) -> bytes:
    compression = {
        "stored": zipfile.ZIP_STORED,
        "deflate": zipfile.ZIP_DEFLATED,
    }.get(codec)
    if compression is None:
        raise ValueError(f"unknown ZIP codec: {codec}")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = compression
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(
            info,
            member,
            compress_type=compression,
            compresslevel=level,
        )
    return output.getvalue()


def build_archive(
    streams: tuple[bytes, bytes, bytes],
    tokens: bytes,
    *,
    model_codec: str,
    zip_codec: str = "stored",
    zip_level: int | None = None,
) -> bytes:
    member = receiver.pack_payload(
        split_pack(streams),
        tokens,
        token_codec="ans",
        model_codec=model_codec,
    )
    return deterministic_zip(member, codec=zip_codec, level=zip_level)


def parse_source_archive(path: Path) -> SourceObject:
    member = read_stored_member(path)
    parts = receiver.split_payload(member)
    if parts.token_codec != "ans" or parts.model_codec != "legacy_lzma":
        raise RuntimeError("AI1 source is not the expected legacy-model/ANS wire")
    decoded = receiver.decode_models(parts.models, model_codec=parts.model_codec)
    models_raw, temporal = receiver.split_optional_temporal_reversion(decoded.raw)
    if temporal is None:
        raise RuntimeError("AI1 source lacks the counted temporal-reversion sidecar")
    if len(models_raw) < 8:
        raise RuntimeError("AI1 model bytes are truncated before raw section lengths")
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", models_raw)
    semantic_end = 8 + semantic_bytes
    carrier_end = semantic_end + carrier_bytes
    if carrier_end >= len(models_raw):
        raise RuntimeError("AI1 model bytes lack a complete HPAC section")
    source = SourceObject(
        semantic=models_raw[8:semantic_end],
        carrier=models_raw[semantic_end:carrier_end],
        hpac_wire=decoded.raw[carrier_end:],
        tokens=parts.tokens,
        models_raw=models_raw,
        models_raw_wire=decoded.raw,
        temporal_packed=temporal.packed,
    )
    if sha256_bytes(source.models_raw) != EXPECTED_MODELS_RAW_SHA256:
        raise RuntimeError("AI1 reconstructed PR130 model bytes differ from the pin")
    if sha256_bytes(source.models_raw_wire) != EXPECTED_MODELS_RAW_WIRE_SHA256:
        raise RuntimeError("AI1 temporal model wire differs from the pin")
    if len(source.tokens) != EXPECTED_TOKEN_BYTES or sha256_bytes(source.tokens) != EXPECTED_TOKEN_SHA256:
        raise RuntimeError("AI1 temporal ANS payload differs from the pin")
    if (
        len(source.temporal_packed) != EXPECTED_TEMPORAL_PACKED_BYTES
        or sha256_bytes(source.temporal_packed) != EXPECTED_TEMPORAL_PACKED_SHA256
    ):
        raise RuntimeError("AI1 temporal sidecar differs from the pin")
    return source


def validate_archive(archive_bytes: bytes, source: SourceObject) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        if [entry.filename for entry in archive.infolist()] != ["p"]:
            raise RuntimeError("candidate archive has the wrong member set")
        member = archive.read("p")
        if archive.testzip() is not None:
            raise RuntimeError("candidate archive failed ZIP CRC validation")
    parts = receiver.split_payload(member)
    if parts.token_codec != "ans" or parts.model_codec != "split_brotli_cx2":
        raise RuntimeError("candidate selectors do not name ANS + CX2 split Brotli")
    decoded = receiver.decode_models(parts.models, model_codec=parts.model_codec)
    models_raw, temporal = receiver.split_optional_temporal_reversion(decoded.raw)
    if decoded.raw != source.models_raw_wire:
        raise RuntimeError("candidate failed exact temporal model-wire parse-back")
    if models_raw != source.models_raw or temporal is None:
        raise RuntimeError("candidate failed exact base-model/temporal split")
    if temporal.packed != source.temporal_packed or parts.tokens != source.tokens:
        raise RuntimeError("candidate changed temporal or token payload bytes")

    semantic_bytes, carrier_bytes = struct.unpack_from("<II", models_raw)
    semantic_pose_end = 8 + semantic_bytes + carrier_bytes
    semantic_model, basis, coefficients = inflate.unpack_semantic_pose(models_raw[:semantic_pose_end])
    hpac_model = inflate.load_hpac(
        models_raw[semantic_pose_end:],
        torch.device("cpu"),
    )
    return {
        "all_sections_consumed": True,
        "member_bytes": len(member),
        "member_sha256": sha256_bytes(member),
        "models_raw_wire_bytes": len(decoded.raw),
        "models_raw_wire_sha256": sha256_bytes(decoded.raw),
        "base_models_raw_bytes": len(models_raw),
        "base_models_raw_sha256": sha256_bytes(models_raw),
        "semantic_bytes": semantic_bytes,
        "carrier_bytes": carrier_bytes,
        "hpac_bytes": len(models_raw) - semantic_pose_end,
        "hpac_wire_bytes": len(source.hpac_wire),
        "temporal_packed_bytes": len(temporal.packed),
        "temporal_packed_sha256": sha256_bytes(temporal.packed),
        "tokens_bytes": len(parts.tokens),
        "tokens_sha256": sha256_bytes(parts.tokens),
        "semantic_tensor_count": len(semantic_model.state_dict()),
        "carrier_basis_shape": list(basis.shape),
        "carrier_coefficients_shape": list(coefficients.shape),
        "hpac_tensor_count": len(hpac_model.state_dict()),
    }


def stage_inputs(output: Path, source: SourceObject) -> dict[str, Any]:
    require_file(
        AI1_BUILD_RECEIPT,
        digest=EXPECTED_AI1_BUILD_RECEIPT_SHA256,
        label="AI1 build receipt",
    )
    require_file(
        AI1_DETERMINISM_RECEIPT,
        digest=EXPECTED_AI1_DETERMINISM_RECEIPT_SHA256,
        label="AI1 determinism receipt",
    )
    retained = output / "retained" / "inputs"
    member = read_stored_member(AI1_ARCHIVE)
    artifacts = {
        "source_archive": persist_exact(retained / "ai1_source_archive.zip", AI1_ARCHIVE.read_bytes()),
        "source_member": persist_exact(retained / "ai1_source_payload.p", member),
        "semantic": persist_exact(retained / "semantic.raw", source.semantic),
        "carrier": persist_exact(retained / "carrier.raw", source.carrier),
        "hpac_wire": persist_exact(retained / "hpac_plus_temporal.raw", source.hpac_wire),
        "tokens": persist_exact(retained / "tokens.ans", source.tokens),
        "models_raw": persist_exact(retained / "models_raw.bin", source.models_raw),
        "models_raw_wire": persist_exact(retained / "models_raw_with_temporal.bin", source.models_raw_wire),
        "temporal_packed": persist_exact(retained / "temporal_reversion.tm1p", source.temporal_packed),
    }
    result = {
        "schema": "ddm_lc2_inputs.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[archive-byte exact; scorer-free]",
        "score_claim": False,
        "source": file_record(AI1_ARCHIVE),
        "source_build_receipt": file_record(AI1_BUILD_RECEIPT),
        "source_determinism_receipt": file_record(AI1_DETERMINISM_RECEIPT),
        "artifacts": artifacts,
    }
    atomic_json(output / "stages" / "01_inputs.json", result)
    return result


def section_candidates(
    output: Path,
    section_name: str,
    transformed: bytes,
    transform: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = output / "retained" / "section_candidates" / section_name
    transform_record = persist_exact(
        root / f"{transform}.raw",
        transformed,
    )
    for quality in QUALITIES:
        encoded = brotli.compress(transformed, quality=quality)
        encoded_record = persist_exact(
            root / f"{transform}_q{quality}.br",
            encoded,
        )
        rows.append(
            {
                "section": section_name,
                "transform": transform,
                "brotli_quality": quality,
                "transformed": transform_record,
                "encoded": encoded_record,
                "encoded_bytes_value": encoded,
            }
        )
    return rows


def public_section_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "encoded_bytes_value"}


def stage_search(output: Path, source: SourceObject) -> dict[str, Any]:
    complete_path = output / "stages" / "03_search.json"
    if complete_path.is_file():
        existing = json.loads(complete_path.read_text())
        winner = Path(existing["winner"]["archive"]["path"])
        if (
            existing.get("schema") != "ddm_lc2_complete_zip_search.v1"
            or existing.get("complete") is not True
            or file_record(winner) != existing["winner"]["archive"]
        ):
            raise RuntimeError("completed LC2 search checkpoint changed")
        return existing

    transformed_sections = receiver.encode_cx2_model_sections(
        source.semantic,
        source.carrier,
        source.hpac_wire,
    )
    if receiver.decode_cx2_model_sections(*transformed_sections) != (
        source.semantic,
        source.carrier,
        source.hpac_wire,
    ):
        raise RuntimeError("CX2 reference transform failed its exact receiver inverse")
    semantic_rows = section_candidates(
        output,
        "semantic",
        transformed_sections[0],
        "signed_zigzag_block4096_lane2",
    )
    carrier_rows = section_candidates(
        output,
        "carrier",
        transformed_sections[1],
        "identity",
    )
    hpac_rows = section_candidates(
        output,
        "hpac_wire",
        transformed_sections[2],
        "xor80",
    )
    section_manifest = {
        "schema": "ddm_lc2_section_candidates.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "semantic": [public_section_row(row) for row in semantic_rows],
        "carrier": [public_section_row(row) for row in carrier_rows],
        "hpac_wire": [public_section_row(row) for row in hpac_rows],
    }
    atomic_json(output / "stages" / "02_section_candidates.json", section_manifest)

    candidate_root = output / "retained" / "search_candidates"
    candidates: list[dict[str, Any]] = []
    candidate_index = 0
    for semantic, carrier, hpac in itertools.product(
        semantic_rows,
        carrier_rows,
        hpac_rows,
    ):
        streams = (
            semantic["encoded_bytes_value"],
            carrier["encoded_bytes_value"],
            hpac["encoded_bytes_value"],
        )
        for zip_codec, zip_level in ZIP_POLICIES:
            archive = build_archive(
                streams,
                source.tokens,
                model_codec="split_brotli_cx2",
                zip_codec=zip_codec,
                zip_level=zip_level,
            )
            name = f"candidate_{candidate_index:04d}_{zip_codec}.zip"
            archive_record = persist_exact(candidate_root / name, archive)
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "archive": archive_record,
                    "semantic": public_section_row(semantic),
                    "carrier": public_section_row(carrier),
                    "hpac_wire": public_section_row(hpac),
                    "zip": {"codec": zip_codec, "level": zip_level},
                }
            )
            candidate_index += 1

    candidates.sort(
        key=lambda row: (
            row["archive"]["bytes"],
            row["archive"]["sha256"],
            row["candidate_index"],
        )
    )
    winner = candidates[0]
    minimum_bytes = winner["archive"]["bytes"]
    ties = [row for row in candidates if row["archive"]["bytes"] == minimum_bytes]
    tie_root = output / "retained" / "tie_set"
    tie_rows: list[dict[str, Any]] = []
    for tie_index, row in enumerate(ties):
        archive = Path(row["archive"]["path"]).read_bytes()
        tie_archive = persist_exact(tie_root / f"tie_{tie_index:03d}.zip", archive)
        repeat = build_archive(
            (
                Path(row["semantic"]["encoded"]["path"]).read_bytes(),
                Path(row["carrier"]["encoded"]["path"]).read_bytes(),
                Path(row["hpac_wire"]["encoded"]["path"]).read_bytes(),
            ),
            source.tokens,
            model_codec="split_brotli_cx2",
            zip_codec=row["zip"]["codec"],
            zip_level=row["zip"]["level"],
        )
        repeat_record = persist_exact(
            tie_root / f"tie_{tie_index:03d}.repeat.zip",
            repeat,
        )
        if tie_archive["sha256"] != repeat_record["sha256"]:
            raise RuntimeError("tie-set archive repeat is not byte-identical")
        tie_rows.append({**row, "tie_archive": tie_archive, "repeat": repeat_record})

    result = {
        "schema": "ddm_lc2_complete_zip_search.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[archive-byte exact; scorer-free]",
        "score_claim": False,
        "candidate_denominator": len(candidates),
        "semantic_variant_denominator": len(semantic_rows),
        "carrier_variant_denominator": len(carrier_rows),
        "hpac_variant_denominator": len(hpac_rows),
        "zip_policy_denominator": len(ZIP_POLICIES),
        "selection_surface": (
            "complete deterministic archive.zip bytes over the CX2 reference "
            "transform's Brotli q9/q10/q11 and stored/deflate policies"
        ),
        "selection_mode": ("minimum archive bytes; archive SHA-256 then enumeration index tiebreak"),
        "winner": winner,
        "minimum_byte_tie_denominator": len(ties),
        "minimum_unique_archive_denominator": len({row["archive"]["sha256"] for row in ties}),
        "tie_set": tie_rows,
        "all_candidates": candidates,
    }
    atomic_json(output / "search" / "complete_zip_search.json", result)
    atomic_json(complete_path, result)
    return result


def stage_build(
    output: Path,
    source: SourceObject,
    search: dict[str, Any],
) -> dict[str, Any]:
    standard_streams = tuple(
        brotli.compress(section, quality=11) for section in (source.semantic, source.carrier, source.hpac_wire)
    )
    standard = build_archive(
        standard_streams,
        source.tokens,
        model_codec="split_brotli",
    )
    # Parse the standard comparator with its own selector before retaining it.
    with zipfile.ZipFile(io.BytesIO(standard)) as archive:
        standard_member = archive.read("p")
    standard_parts = receiver.split_payload(standard_member)
    standard_decoded = receiver.decode_models(
        standard_parts.models,
        model_codec=standard_parts.model_codec,
    )
    if (
        standard_parts.token_codec != "ans"
        or standard_parts.model_codec != "split_brotli"
        or standard_decoded.raw != source.models_raw_wire
        or standard_parts.tokens != source.tokens
    ):
        raise RuntimeError("standard split-Brotli comparator failed exact parse-back")

    winner_archive = Path(search["winner"]["archive"]["path"]).read_bytes()
    parseback = validate_archive(winner_archive, source)
    retained = output / "retained" / "per_step"
    source_record = persist_exact(retained / "00_ai1_source.zip", AI1_ARCHIVE.read_bytes())
    standard_record = persist_exact(retained / "01_split_brotli.zip", standard)
    standard_repeat = persist_exact(
        retained / "01_split_brotli.repeat.zip",
        build_archive(
            standard_streams,
            source.tokens,
            model_codec="split_brotli",
        ),
    )
    final_record = persist_exact(retained / "02_xcodec_split_brotli.zip", winner_archive)
    winner = search["winner"]
    final_repeat_bytes = build_archive(
        (
            Path(winner["semantic"]["encoded"]["path"]).read_bytes(),
            Path(winner["carrier"]["encoded"]["path"]).read_bytes(),
            Path(winner["hpac_wire"]["encoded"]["path"]).read_bytes(),
        ),
        source.tokens,
        model_codec="split_brotli_cx2",
        zip_codec=winner["zip"]["codec"],
        zip_level=winner["zip"]["level"],
    )
    final_repeat = persist_exact(
        retained / "02_xcodec_split_brotli.repeat.zip",
        final_repeat_bytes,
    )
    if standard_record["sha256"] != standard_repeat["sha256"]:
        raise RuntimeError("standard split archive is not deterministic")
    if final_record["sha256"] != final_repeat["sha256"]:
        raise RuntimeError("selected xcodec archive is not deterministic")

    submission = output / "submission"
    archive_dir = submission / "archive"
    inflated = submission / "inflated"
    archive_dir.mkdir(parents=True, exist_ok=True)
    inflated.mkdir(parents=True, exist_ok=True)
    if any(inflated.iterdir()):
        raise RuntimeError("refusing build over existing evaluator-facing raw output")
    evaluator_archive = persist_exact(submission / "archive.zip", winner_archive)
    evaluator_member = persist_exact(archive_dir / "p", read_archive_member_bytes(winner_archive))
    runtime_records: dict[str, Any] = {}
    for name in RUNTIME_FILES:
        source_path = RUNTIME / name
        payload = source_path.read_bytes()
        destination = submission / name
        if destination.exists():
            actual = file_record(destination)
            if actual["sha256"] != sha256_bytes(payload):
                raise RuntimeError(f"shipping runtime changed: {destination}")
        else:
            atomic_bytes(destination, payload, executable=name == "inflate.sh")
        runtime_records[name] = {
            **file_record(destination),
            "source": str(source_path.relative_to(REPO)),
            "source_sha256": sha256_bytes(payload),
        }

    final_bytes = final_record["bytes"]
    result = {
        "schema": "ddm_lc2_build.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[archive-byte exact + macOS-CPU parse-back; scorer-free]",
        "score_claim": False,
        "falsifier": {
            "threshold_bytes": EXPECTED_AI1_BYTES,
            "passes": final_bytes < EXPECTED_AI1_BYTES,
            "rule": "stop without receiver decode if final archive is not below AI1",
        },
        "stages": [
            {"name": "AI1 ANS + temporal source", "archive": source_record},
            {
                "name": "AI1 + split Brotli",
                "archive": standard_record,
                "delta_bytes_from_previous": standard_record["bytes"] - source_record["bytes"],
            },
            {
                "name": "AI1 + CX2 reversible xcodec split Brotli",
                "archive": final_record,
                "delta_bytes_from_previous": final_record["bytes"] - standard_record["bytes"],
            },
        ],
        "interaction": {
            "charter_arithmetic_prediction_bytes": 187_733,
            "measured_final_minus_charter_prediction_bytes": final_bytes - 187_733,
            "split_delta_vs_ai1_bytes": standard_record["bytes"] - source_record["bytes"],
            "xcodec_delta_after_split_bytes": final_record["bytes"] - standard_record["bytes"],
            "total_delta_vs_ai1_bytes": final_bytes - source_record["bytes"],
            "total_delta_vs_pr130_bytes": final_bytes - PR130_ARCHIVE_BYTES,
        },
        "archive": final_record,
        "archive_repeat": final_repeat,
        "repeat_byte_identical": True,
        "standard_repeat": standard_repeat,
        "parseback": parseback,
        "search_receipt": file_record(output / "search" / "complete_zip_search.json"),
        "submission": {
            "path": str(submission.resolve()),
            "archive": evaluator_archive,
            "member": evaluator_member,
            "runtime_files": runtime_records,
        },
        "derived_score": {
            "status": "DERIVED_NOT_EXACT_RECEIPT",
            "formula": ("PR130 contest-CUDA score - 25*(191052-final_bytes)/37545489"),
            "value": PR130_CONTEST_CUDA_SCORE - 25 * (PR130_ARCHIVE_BYTES - final_bytes) / EXPECTED_ORIGINAL_BYTES,
            "display_rounding_caveat": ("PR130 bar distortion components were published to 8 decimals"),
        },
        "provenance": provenance(),
    }
    atomic_json(output / "LC2_BUILD_RESULT.json", result)
    atomic_json(output / "stages" / "04_build.json", result)
    return result


def build_pipeline(output: Path, minimum_free_bytes: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free
    if free < minimum_free_bytes:
        raise RuntimeError(f"storage preflight refused: free={free}, required={minimum_free_bytes}")
    require_file(
        AI1_ARCHIVE,
        size=EXPECTED_AI1_BYTES,
        digest=EXPECTED_AI1_SHA256,
        label="AI1 temporal archive",
    )
    source = parse_source_archive(AI1_ARCHIVE)
    inputs = stage_inputs(output, source)
    search = stage_search(output, source)
    build = stage_build(output, source, search)
    result = {
        "schema": "ddm_lc2_pipeline.v1",
        "complete": build["falsifier"]["passes"],
        "build_complete": True,
        "written_at_utc": utc_now(),
        "axis": "[archive-byte exact + macOS-CPU parse-back; scorer-free]",
        "score_claim": False,
        "storage_preflight": {
            "path": str(output.resolve()),
            "free_bytes_at_start": free,
            "required_free_bytes": minimum_free_bytes,
        },
        "inputs": inputs,
        "search": {
            "receipt": file_record(output / "search" / "complete_zip_search.json"),
            "candidate_denominator": search["candidate_denominator"],
            "minimum_byte_tie_denominator": search["minimum_byte_tie_denominator"],
        },
        "build": build,
        "pending_terminal_gates": (
            [
                "literal receiver reconstructs n600 raw within 1800 seconds",
                "raw SHA-256 equals the PR130/AI1 reference byte-for-byte",
                "MAIN fires one exact n600 confirming evaluator row",
            ]
            if build["falsifier"]["passes"]
            else []
        ),
    }
    atomic_json(output / "LC2_RESULT.json", result)
    return result


def acquire_run_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"LC2 decode already owns run lock: {path}") from error
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "utc": utc_now()}, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def preserve_certified_partial_raw(decode_root: Path, staging_raw: Path) -> dict[str, Any] | None:
    """Cold-store a timeout raw only after its durable receipt proves identity."""

    if not staging_raw.exists():
        return None
    timeout_path = decode_root / "decode_timeout_receipt.json"
    if not timeout_path.is_file():
        raise RuntimeError("uncertified retained staging raw blocks relaunch; preserve and adjudicate it")
    timeout = json.loads(timeout_path.read_text())
    expected = timeout.get("partial_raw")
    actual = file_record(staging_raw)
    if (
        timeout.get("schema") != "ddm_lc2_decode_timeout.v1"
        or timeout.get("payloads_retained") is not True
        or expected != actual
    ):
        raise RuntimeError("staging raw differs from its timeout certification")
    destination = decode_root / "attempts" / f"timeout_partial_{actual['sha256'][:16]}.raw"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError("certified timeout destination already exists while staging remains")
    os.replace(staging_raw, destination)
    moved = file_record(destination)
    if moved["bytes"] != actual["bytes"] or moved["sha256"] != actual["sha256"]:
        raise RuntimeError("cold-stored timeout raw changed during the atomic move")
    receipt = {
        "schema": "ddm_lc2_timeout_partial_move.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU literal receiver; scorer-free]",
        "score_claim": False,
        "source_timeout_receipt": file_record(timeout_path),
        "original": actual,
        "cold_stored": moved,
        "reason": (
            "the receiver cannot resume a partial render; preserve its certified bytes "
            "before restarting from the complete token checkpoint"
        ),
        "payloads_retained": True,
    }
    atomic_json(decode_root / "timeout_partial_move_receipt.json", receipt)
    return receipt


def validate_token_receipt(path: Path, member: bytes, source: SourceObject) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if (
        result.get("schema") != "ddm_cx2_token_checkpoint.v1"
        or result.get("complete") is not True
        or result.get("frames") != EXPECTED_FRAMES
        or result.get("tokens") != EXPECTED_TOKENS
        or result.get("token_codec") != "ans"
        or result.get("finish_token_decode_returned") is not True
        or result.get("ans_final_state_empty") is not True
        or result.get("decoded_token_sha256") != EXPECTED_DECODED_TOKEN_SHA256
        or result.get("archive_member_sha256") != sha256_bytes(member)
        or result.get("models_raw_sha256") != EXPECTED_MODELS_RAW_WIRE_SHA256
        or result.get("token_payload_sha256") != sha256_bytes(source.tokens)
    ):
        raise RuntimeError("LC2 token checkpoint is not the complete n600 ANS proof")
    cache = Path(result["cache"]["path"])
    if file_record(cache) != result["cache"]:
        raise RuntimeError("LC2 token checkpoint payload changed")
    return result


def decode_pipeline(
    output: Path,
    python: Path,
    brotli_cli: Path,
    timeout_seconds: int,
    minimum_free_bytes: int,
) -> dict[str, Any]:
    build = json.loads((output / "LC2_BUILD_RESULT.json").read_text())
    if build.get("complete") is not True or build.get("falsifier", {}).get("passes") is not True:
        raise RuntimeError("LC2 build is not a size-winning completed candidate")
    submission = Path(build["submission"]["path"])
    archive = submission / "archive.zip"
    member_path = submission / "archive" / "p"
    archive_record = file_record(archive)
    retained_record = build["archive"]
    if archive_record["bytes"] != retained_record["bytes"] or archive_record["sha256"] != retained_record["sha256"]:
        raise RuntimeError("LC2 evaluator archive differs from retained winner")
    member = member_path.read_bytes()
    if member != read_archive_member_bytes(archive.read_bytes()):
        raise RuntimeError("LC2 expanded member differs from archive.zip")
    source = parse_source_archive(AI1_ARCHIVE)
    validate_archive(archive.read_bytes(), source)
    require_file(
        AI1_RAW,
        size=EXPECTED_RAW_BYTES,
        digest=EXPECTED_RAW_SHA256,
        label="AI1 raw reference",
    )
    require_file(
        VIDEO_NAMES,
        digest=EXPECTED_VIDEO_NAMES_SHA256,
        label="public video names",
    )
    if not python.is_file():
        raise RuntimeError(f"pinned runtime Python is absent: {python}")
    if not brotli_cli.is_file():
        raise RuntimeError(f"pinned Brotli CLI is absent: {brotli_cli}")

    decode_root = output / "retained" / "decode"
    complete_path = decode_root / "decode_receipt.json"
    final_raw = decode_root / "0.raw"
    token_cache = decode_root / "checkpoint" / "tokens.npz"
    token_progress = decode_root / "checkpoint" / "tokens.progress.npz"
    token_receipt = decode_root / "checkpoint" / "tokens_receipt.json"
    log_path = decode_root / "inflate.log"
    if complete_path.is_file():
        result = json.loads(complete_path.read_text())
        validate_token_receipt(token_receipt, member, source)
        if result.get("complete") is not True or file_record(final_raw) != result.get("raw"):
            raise RuntimeError("completed LC2 decode receipt changed")
        ready_path = output / "READY_EXACT_EVAL.json"
        ready = json.loads(ready_path.read_text())
        ready["dispatch"]["consumer_store"] = str(DEFAULT_EXACT_EVAL_STORE)
        ready["written_at_utc"] = utc_now()
        atomic_json(ready_path, ready)
        terminal = json.loads((output / "LC2_RESULT.json").read_text())
        terminal["ready"] = ready
        terminal["written_at_utc"] = utc_now()
        atomic_json(output / "LC2_RESULT.json", terminal)
        return terminal

    free = shutil.disk_usage(output).free
    if free < minimum_free_bytes:
        raise RuntimeError(f"decode storage preflight refused: free={free}, required={minimum_free_bytes}")
    run_lock = acquire_run_lock(decode_root / ".run.lock")
    staging = decode_root / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    staging_raw = staging / "0.raw"
    preserve_certified_partial_raw(decode_root, staging_raw)
    command = [
        str(submission / "inflate.sh"),
        str(submission / "archive"),
        str(staging),
        str(VIDEO_NAMES),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON": str(python),
            "PR130_BROTLI_CLI": str(brotli_cli),
            "PR130_INFLATE_DEVICE": "cpu",
            "PR130_TOKEN_CACHE": str(token_cache),
            "PR130_TOKEN_RECEIPT": str(token_receipt),
            "PR130_RUNTIME_DEPS_DIR": str(decode_root / "runtime-deps"),
            "PYTHONHASHSEED": "0",
        }
    )
    state = {
        "schema": "ddm_lc2_decode_state.v1",
        "complete": False,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU literal receiver; scorer-free]",
        "score_claim": False,
        "command": command,
        "environment": {
            key: environment[key]
            for key in (
                "PYTHON",
                "PR130_BROTLI_CLI",
                "PR130_INFLATE_DEVICE",
                "PR130_TOKEN_CACHE",
                "PR130_TOKEN_RECEIPT",
                "PR130_RUNTIME_DEPS_DIR",
                "PYTHONHASHSEED",
            )
        },
        "cpu_thread_policy": "inherit the pinned runtime defaults",
        "archive": file_record(archive),
        "storage_preflight": {
            "free_bytes_at_launch": free,
            "required_free_bytes": minimum_free_bytes,
        },
        "resumable_from_disk": True,
        "stage_checkpoints": [str(token_progress), str(token_cache), str(token_receipt)],
    }
    atomic_json(decode_root / "decode_state.json", state)
    started = time.perf_counter()
    with log_path.open("ab") as log:
        log.write(f"\nDDM_LC2_DECODE_START utc={utc_now()}\n".encode())
        log.flush()
        os.fsync(log.fileno())
        process = subprocess.Popen(
            command,
            cwd=submission,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            wall_seconds = time.perf_counter() - started
            timeout_receipt = {
                **state,
                "schema": "ddm_lc2_decode_timeout.v1",
                "complete": False,
                "finished_at_utc": utc_now(),
                "wall_seconds": wall_seconds,
                "timeout_seconds": timeout_seconds,
                "partial_raw": file_record(staging_raw) if staging_raw.is_file() else None,
                "token_cache": file_record(token_cache) if token_cache.is_file() else None,
                "token_progress": (file_record(token_progress) if token_progress.is_file() else None),
                "payloads_retained": True,
            }
            atomic_json(decode_root / "decode_timeout_receipt.json", timeout_receipt)
            raise RuntimeError("LC2 literal receiver exceeded its timeout") from error
        log.flush()
        os.fsync(log.fileno())
    wall_seconds = time.perf_counter() - started
    if returncode != 0:
        failure = {
            **state,
            "schema": "ddm_lc2_decode_failure.v1",
            "complete": False,
            "finished_at_utc": utc_now(),
            "returncode": returncode,
            "wall_seconds": wall_seconds,
            "partial_raw": file_record(staging_raw) if staging_raw.is_file() else None,
            "log": file_record(log_path),
            "payloads_retained": True,
        }
        atomic_json(decode_root / "decode_failure_receipt.json", failure)
        raise RuntimeError(f"LC2 literal receiver failed with rc={returncode}")
    if not staging_raw.is_file() or staging_raw.stat().st_size != EXPECTED_RAW_BYTES:
        raise RuntimeError("LC2 literal receiver produced the wrong raw byte count")
    token_result = validate_token_receipt(token_receipt, member, source)
    raw_record = file_record(staging_raw)
    raw_identity = raw_record["sha256"] == EXPECTED_RAW_SHA256
    if final_raw.exists():
        raise RuntimeError("refusing to overwrite an existing retained LC2 raw")
    os.replace(staging_raw, final_raw)
    raw_record = file_record(final_raw)
    result = {
        "schema": "ddm_lc2_literal_decode.v1",
        "complete": raw_identity and wall_seconds <= 1_800,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU literal receiver; scorer-free]",
        "score_claim": False,
        "command": command,
        "wall_seconds": wall_seconds,
        "within_1800_second_inflate_limit": wall_seconds <= 1_800,
        "archive": file_record(archive),
        "archive_member": file_record(member_path),
        "raw": raw_record,
        "reference_raw": file_record(AI1_RAW),
        "raw_byte_identical_to_pr130": raw_identity,
        "token_checkpoint": token_result,
        "token_progress_checkpoint": file_record(token_progress),
        "log": file_record(log_path),
        "runtime_files": {name: file_record(submission / name) for name in RUNTIME_FILES},
        "resumable_from_disk": True,
        "provenance": provenance(),
    }
    atomic_json(complete_path, result)
    state["complete"] = result["complete"]
    state["completed_receipt"] = file_record(complete_path)
    atomic_json(decode_root / "decode_state.json", state)
    run_lock.close()
    if not raw_identity:
        raise RuntimeError("LC2 raw differs by at least one byte from PR130; score licence void")
    if wall_seconds > 1_800:
        raise RuntimeError("LC2 raw is exact but the literal receiver exceeded 1800 seconds")

    evaluator_raw = submission / "inflated" / "0.raw"
    if evaluator_raw.exists():
        if file_record(evaluator_raw)["sha256"] != raw_record["sha256"]:
            raise RuntimeError("existing evaluator raw differs from certified LC2 raw")
    else:
        os.link(final_raw, evaluator_raw)
    ready = {
        "schema": "ddm_lc2_exact_eval_ready.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[READY receipt; score prediction DERIVED; score_claim=false]",
        "score_claim": False,
        "ready_for_exact_eval_dispatch": True,
        "archive": file_record(archive),
        "raw": file_record(evaluator_raw),
        "raw_byte_identical_to_pr130": True,
        "decode_wall_seconds": wall_seconds,
        "within_1800_second_inflate_limit": True,
        "predicted_score_status": "DERIVED_NOT_EXACT_RECEIPT",
        "predicted_score": build["derived_score"]["value"],
        "prediction_formula": build["derived_score"]["formula"],
        "dispatch": {
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "MAIN exact contest-row owner",
            "consumer_store": str(DEFAULT_EXACT_EVAL_STORE),
            "fire_trigger": (
                "MAIN claims the sole exact-eval lane and the locked upstream "
                "environment passes parity; archive bytes must remain unchanged"
            ),
        },
    }
    atomic_json(output / "READY_EXACT_EVAL.json", ready)
    terminal = json.loads((output / "LC2_RESULT.json").read_text())
    terminal["complete"] = True
    terminal["decode"] = result
    terminal["ready"] = ready
    terminal["pending_terminal_gates"] = ["MAIN fires one exact n600 confirming row"]
    terminal["written_at_utc"] = utc_now()
    atomic_json(output / "LC2_RESULT.json", terminal)
    return terminal


def cold_decode_pipeline(
    output: Path,
    cold_output: Path,
    python: Path,
    brotli_cli: Path,
    timeout_seconds: int,
    minimum_free_bytes: int,
) -> dict[str, Any]:
    """Run and certify a fresh-cache contest-shape receiver on APDataStore."""

    build = json.loads((output / "LC2_BUILD_RESULT.json").read_text())
    if build.get("complete") is not True or build.get("falsifier", {}).get("passes") is not True:
        raise RuntimeError("LC2 build is not a size-winning completed candidate")
    submission = Path(build["submission"]["path"])
    archive = submission / "archive.zip"
    member_path = submission / "archive" / "p"
    member = member_path.read_bytes()
    source = parse_source_archive(AI1_ARCHIVE)
    validate_archive(archive.read_bytes(), source)
    if member != read_archive_member_bytes(archive.read_bytes()):
        raise RuntimeError("LC2 expanded member differs from archive.zip")
    require_file(
        AI1_RAW,
        size=EXPECTED_RAW_BYTES,
        digest=EXPECTED_RAW_SHA256,
        label="AI1 raw reference",
    )
    require_file(
        VIDEO_NAMES,
        digest=EXPECTED_VIDEO_NAMES_SHA256,
        label="public video names",
    )
    if not python.is_file() or not brotli_cli.is_file():
        raise RuntimeError("pinned cold-decode Python or Brotli CLI is absent")

    cold_output.mkdir(parents=True, exist_ok=True)
    complete_path = cold_output / "cold_decode_receipt.json"
    final_raw = cold_output / "0.raw"
    token_cache = cold_output / "checkpoint" / "tokens.npz"
    token_progress = cold_output / "checkpoint" / "tokens.progress.npz"
    token_receipt = cold_output / "checkpoint" / "tokens_receipt.json"
    log_path = cold_output / "inflate.log"

    def attach_to_terminal(result: dict[str, Any]) -> None:
        ready_path = output / "READY_EXACT_EVAL.json"
        ready = json.loads(ready_path.read_text())
        ready["decode_wall_seconds"] = result["wall_seconds"]
        ready["decode_mode"] = "fresh-cache literal receiver"
        ready["cold_decode_receipt"] = file_record(complete_path)
        ready["cold_raw"] = result["raw"]
        ready["written_at_utc"] = utc_now()
        atomic_json(ready_path, ready)
        terminal_path = output / "LC2_RESULT.json"
        terminal = json.loads(terminal_path.read_text())
        terminal["cold_decode"] = result
        terminal["ready"] = ready
        terminal["written_at_utc"] = utc_now()
        atomic_json(terminal_path, terminal)

    if complete_path.is_file():
        result = json.loads(complete_path.read_text())
        validate_token_receipt(token_receipt, member, source)
        if result.get("complete") is not True or file_record(final_raw) != result.get("raw"):
            raise RuntimeError("completed LC2 cold-decode receipt changed")
        attach_to_terminal(result)
        return result

    free = shutil.disk_usage(cold_output).free
    if free < minimum_free_bytes:
        raise RuntimeError(f"cold-decode storage preflight refused: free={free}, required={minimum_free_bytes}")
    run_lock = acquire_run_lock(cold_output / ".run.lock")
    staging = cold_output / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    staging_raw = staging / "0.raw"
    if staging_raw.exists():
        raise RuntimeError("retained cold-decode staging raw blocks relaunch")
    if token_cache.exists() or token_progress.exists() or token_receipt.exists():
        raise RuntimeError("cold decode requires an empty token-checkpoint surface")
    command = [
        str(submission / "inflate.sh"),
        str(submission / "archive"),
        str(staging),
        str(VIDEO_NAMES),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON": str(python),
            "PR130_BROTLI_CLI": str(brotli_cli),
            "PR130_INFLATE_DEVICE": "cpu",
            "PR130_TOKEN_CACHE": str(token_cache),
            "PR130_TOKEN_RECEIPT": str(token_receipt),
            "PR130_RUNTIME_DEPS_DIR": str(cold_output / "runtime-deps"),
            "PYTHONHASHSEED": "0",
        }
    )
    state = {
        "schema": "ddm_lc2_cold_decode_state.v1",
        "complete": False,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU fresh-cache literal receiver; scorer-free]",
        "score_claim": False,
        "command": command,
        "environment": {
            key: environment[key]
            for key in (
                "PYTHON",
                "PR130_BROTLI_CLI",
                "PR130_INFLATE_DEVICE",
                "PR130_TOKEN_CACHE",
                "PR130_TOKEN_RECEIPT",
                "PR130_RUNTIME_DEPS_DIR",
                "PYTHONHASHSEED",
            )
        },
        "cpu_thread_policy": "inherit the pinned runtime defaults",
        "archive": file_record(archive),
        "storage_preflight": {
            "free_bytes_at_launch": free,
            "required_free_bytes": minimum_free_bytes,
        },
        "fresh_token_cache_at_launch": True,
        "stage_checkpoints": [str(token_progress), str(token_cache), str(token_receipt)],
    }
    atomic_json(cold_output / "cold_decode_state.json", state)
    started = time.perf_counter()
    with log_path.open("ab") as log:
        log.write(f"\nDDM_LC2_COLD_DECODE_START utc={utc_now()}\n".encode())
        log.flush()
        os.fsync(log.fileno())
        process = subprocess.Popen(
            command,
            cwd=submission,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            timeout = {
                **state,
                "schema": "ddm_lc2_cold_decode_timeout.v1",
                "complete": False,
                "finished_at_utc": utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "timeout_seconds": timeout_seconds,
                "partial_raw": file_record(staging_raw) if staging_raw.is_file() else None,
                "token_cache": file_record(token_cache) if token_cache.is_file() else None,
                "token_progress": file_record(token_progress) if token_progress.is_file() else None,
                "log": file_record(log_path),
                "payloads_retained": True,
            }
            atomic_json(cold_output / "cold_decode_timeout_receipt.json", timeout)
            raise RuntimeError("LC2 fresh-cache literal receiver exceeded its timeout") from error
        log.flush()
        os.fsync(log.fileno())
    wall_seconds = time.perf_counter() - started
    if returncode != 0:
        failure = {
            **state,
            "schema": "ddm_lc2_cold_decode_failure.v1",
            "complete": False,
            "finished_at_utc": utc_now(),
            "returncode": returncode,
            "wall_seconds": wall_seconds,
            "partial_raw": file_record(staging_raw) if staging_raw.is_file() else None,
            "token_cache": file_record(token_cache) if token_cache.is_file() else None,
            "token_progress": file_record(token_progress) if token_progress.is_file() else None,
            "log": file_record(log_path),
            "payloads_retained": True,
        }
        atomic_json(cold_output / "cold_decode_failure_receipt.json", failure)
        raise RuntimeError(f"LC2 fresh-cache literal receiver failed with rc={returncode}")
    if not staging_raw.is_file() or staging_raw.stat().st_size != EXPECTED_RAW_BYTES:
        raise RuntimeError("LC2 fresh-cache receiver produced the wrong raw byte count")
    token_result = validate_token_receipt(token_receipt, member, source)
    raw_record = file_record(staging_raw)
    raw_identity = raw_record["sha256"] == EXPECTED_RAW_SHA256
    if final_raw.exists():
        raise RuntimeError("refusing to overwrite an existing retained cold-decode raw")
    os.replace(staging_raw, final_raw)
    raw_record = file_record(final_raw)
    result = {
        "schema": "ddm_lc2_fresh_cache_literal_decode.v1",
        "complete": raw_identity and wall_seconds <= 1_800,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU fresh-cache literal receiver; scorer-free]",
        "score_claim": False,
        "command": command,
        "cpu_thread_policy": "inherit the pinned runtime defaults",
        "fresh_token_cache_at_launch": True,
        "wall_seconds": wall_seconds,
        "within_1800_second_inflate_limit": wall_seconds <= 1_800,
        "archive": file_record(archive),
        "archive_member": file_record(member_path),
        "raw": raw_record,
        "reference_raw": file_record(AI1_RAW),
        "raw_byte_identical_to_pr130": raw_identity,
        "token_checkpoint": token_result,
        "token_progress_checkpoint": file_record(token_progress),
        "log": file_record(log_path),
        "runtime_files": {name: file_record(submission / name) for name in RUNTIME_FILES},
        "provenance": provenance(),
    }
    atomic_json(complete_path, result)
    state["complete"] = result["complete"]
    state["completed_receipt"] = file_record(complete_path)
    atomic_json(cold_output / "cold_decode_state.json", state)
    run_lock.close()
    if not raw_identity:
        raise RuntimeError("LC2 fresh-cache raw differs from PR130; score licence void")
    if wall_seconds > 1_800:
        raise RuntimeError("LC2 fresh-cache raw is exact but exceeded 1800 seconds")
    attach_to_terminal(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "decode", "cold-decode"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cold-output", type=Path, default=DEFAULT_COLD_OUTPUT)
    parser.add_argument("--python", type=Path, default=PINNED_PYTHON)
    parser.add_argument("--brotli-cli", type=Path, default=Path("/opt/homebrew/bin/brotli"))
    parser.add_argument("--timeout-seconds", type=int, default=1_800)
    parser.add_argument("--minimum-free-bytes", type=int, default=8 << 30)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.minimum_free_bytes <= 0:
        parser.error("timeout and minimum-free-bytes must be positive")
    return args


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if args.command == "build":
        result = build_pipeline(output, args.minimum_free_bytes)
    elif args.command == "decode":
        result = decode_pipeline(
            output,
            args.python.expanduser().absolute(),
            args.brotli_cli.expanduser().absolute(),
            args.timeout_seconds,
            args.minimum_free_bytes,
        )
    else:
        result = cold_decode_pipeline(
            output,
            args.cold_output.resolve(),
            args.python.expanduser().absolute(),
            args.brotli_cli.expanduser().absolute(),
            args.timeout_seconds,
            args.minimum_free_bytes,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
