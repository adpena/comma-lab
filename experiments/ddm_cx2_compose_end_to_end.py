#!/usr/bin/env python3
"""Build the receiver-closed PR130 CX2 composition from custodied bytes.

The builder composes three independently measured changes on one real n600
object: SD1's counted mixed semantic allocation, independently compressed
model streams, and DT1's retained ANS token payload.  It also races the
G25-style reversible byte coordinates on the *complete final ZIP* rather than
selecting from section estimates.

This program does not materialize conditional tables or run a scorer.  DT1
owns the single-flight n600 table job and supplies an already decoded-exact ANS
payload.  Every CX2 build stage is atomic and checkpointed under
``--resume-from``; interrupted composition never invalidates a completed DT1
stage or a prior CX2 stage.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import io
import itertools
import json
import lzma
import os
import shutil
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "src" / "tac" / "pr130_runtime" / "dv1_cpu_runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))
receiver = importlib.import_module("receiver")
inflate = importlib.import_module("inflate")

BASE_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/"
    "reproduction/archive.zip"
)
SD1_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_sd1_semantic_20260809/"
    "cpu_screen/archives/selected_mixed_n600.zip"
)
DT1_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_dt1_20260809")
DEFAULT_OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_cx2_20260809")

EXPECTED_BASE_BYTES = 191_052
EXPECTED_BASE_SHA256 = (
    "0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"
)
EXPECTED_SD1_BYTES = 190_204
EXPECTED_SD1_SHA256 = (
    "010a8a5273ae87595191ffc03447fa36e61978ae9f827c2def46dea7075dfa67"
)
EXPECTED_SD1_SEMANTIC_SHA256 = (
    "39002165c78ab707c15586110678671cd832101a970de5bd0f3b96824a2aa2cc"
)
EXPECTED_RANGE_BYTES = 116_980
EXPECTED_RANGE_SHA256 = (
    "948379872ff81a4e5d948ec301c143be00ebd0033544c8abdfb4af0f4c4a15eb"
)
EXPECTED_MODELS_RAW_SHA256 = (
    "62dd72dfa0858a25ca32bdee1e536627a17883b6fc7efd7cd5b2de7b13b84517"
)
EXPECTED_ANS_BYTES = 114_860
EXPECTED_DT1_RECEIPT_SHA256 = (
    "5c15f38ab68df68c09a5859d17d19e4247f90e76457282edccbc8a34d060916c"
)
EXPECTED_FRAMES = 600
EXPECTED_TOKENS = 117_964_800
EXPECTED_ORIGINAL_BYTES = 37_545_489
EXPECTED_BROTLI_VERSION = "1.2.0"
EXPECTED_CONSTRICTION_VERSION = "0.5.0"
EXPECTED_SELECTED_SPEC = {
    "semantic": {
        "transform": "signed_zigzag_block4096_lane2",
        "brotli_quality": 10,
    },
    "carrier": {"transform": "identity", "brotli_quality": 9},
    "hpac": {"transform": "xor80", "brotli_quality": 10},
    "zip": {"codec": "stored", "level": None},
}
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


@dataclasses.dataclass(frozen=True)
class Sections:
    semantic: bytes
    carrier: bytes
    hpac: bytes
    tokens: bytes


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


def load_complete(path: Path, schema: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if payload.get("schema") != schema or payload.get("complete") is not True:
        raise ValueError(f"invalid completed stage checkpoint: {path}")
    return payload


def read_member(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) != 1 or entries[0].filename != "p":
            raise ValueError(f"{path} must contain exactly one member named p")
        info = entries[0]
        if (
            info.is_dir()
            or info.compress_type != zipfile.ZIP_STORED
            or info.file_size != info.compress_size
            or info.flag_bits & 0x1
        ):
            raise ValueError(f"{path} member p is not an unencrypted stored file")
        payload = archive.read(info)
        if archive.testzip() is not None:
            raise ValueError(f"{path} failed ZIP CRC validation")
        return payload


def legacy_sections(path: Path) -> Sections:
    payload = read_member(path)
    if len(payload) < 4:
        raise ValueError(f"{path} is truncated before model length")
    model_bytes = struct.unpack_from("<I", payload)[0]
    if model_bytes & ~receiver.MODEL_LENGTH_MASK:
        raise ValueError(f"{path} is not the expected legacy-XZ wire form")
    model_end = 4 + model_bytes
    if model_end >= len(payload):
        raise ValueError(f"{path} has no token tail")
    raw = lzma.decompress(payload[4:model_end])
    if len(raw) < 8:
        raise ValueError(f"{path} model bundle is truncated")
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", raw)
    semantic_end = 8 + semantic_bytes
    carrier_end = semantic_end + carrier_bytes
    if carrier_end >= len(raw):
        raise ValueError(f"{path} has no HPAC section")
    return Sections(
        semantic=raw[8:semantic_end],
        carrier=raw[semantic_end:carrier_end],
        hpac=raw[carrier_end:],
        tokens=payload[model_end:],
    )


def lane2_blocks(data: bytes, block_bytes: int) -> bytes:
    if block_bytes <= 0:
        raise ValueError("block_bytes must be positive")
    output = bytearray(len(data))
    for start in range(0, len(data), block_bytes):
        block = data[start:start + block_bytes]
        midpoint = (len(block) + 1) // 2
        output[start:start + midpoint] = block[0::2]
        output[start + midpoint:start + len(block)] = block[1::2]
    return bytes(output)


def transform_bytes(data: bytes, name: str) -> bytes:
    values = np.frombuffer(data, dtype=np.uint8)
    if name == "identity":
        transformed = values
    elif name == "reverse":
        transformed = values[::-1]
    elif name == "xor80":
        transformed = values ^ 0x80
    elif name == "signed_zigzag":
        signed = values.view(np.int8).astype(np.int16)
        transformed = ((signed << 1) ^ (signed >> 7)).astype(np.uint8)
    elif name in ("delta", "signed_zigzag_delta"):
        source = values
        if name == "signed_zigzag_delta":
            signed = values.view(np.int8).astype(np.int16)
            source = ((signed << 1) ^ (signed >> 7)).astype(np.uint8)
        transformed = np.empty_like(source)
        transformed[:1] = source[:1]
        transformed[1:] = (
            source[1:].astype(np.int16) - source[:-1].astype(np.int16)
        ).astype(np.uint8)
    elif name == "nibble_swap":
        transformed = ((values >> 4) | ((values & 0xF) << 4)).astype(np.uint8)
    elif name == "bit_reverse":
        transformed = np.packbits(
            np.unpackbits(values[:, None], axis=1)[:, ::-1],
            axis=1,
        ).reshape(-1)
    elif name == "global_lane2":
        return values[0::2].tobytes() + values[1::2].tobytes()
    elif name == "block4096_lane2":
        return lane2_blocks(data, 4096)
    elif name == "signed_zigzag_block4096_lane2":
        signed = values.view(np.int8).astype(np.int16)
        zigzag = ((signed << 1) ^ (signed >> 7)).astype(np.uint8)
        return lane2_blocks(zigzag.tobytes(order="C"), 4096)
    else:
        raise ValueError(f"unknown reversible transform {name!r}")
    return transformed.tobytes(order="C")


def split_pack(streams: tuple[bytes, bytes, bytes]) -> bytes:
    if any(not stream for stream in streams):
        raise ValueError("split pack requires three non-empty streams")
    return struct.pack("<III", *(len(stream) for stream in streams)) + b"".join(streams)


def deterministic_zip(
    payload: bytes,
    *,
    codec: str,
    level: int | None,
) -> bytes:
    if codec == "stored":
        compression = zipfile.ZIP_STORED
    elif codec == "deflate":
        compression = zipfile.ZIP_DEFLATED
    else:
        raise ValueError(f"unknown ZIP codec {codec!r}")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = compression
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(
            info,
            payload,
            compress_type=compression,
            compresslevel=level,
        )
    return output.getvalue()


def payload_for(
    streams: tuple[bytes, bytes, bytes],
    tokens: bytes,
    *,
    token_codec: str,
    model_codec: str,
) -> bytes:
    return receiver.pack_payload(
        split_pack(streams),
        tokens,
        token_codec=token_codec,
        model_codec=model_codec,
    )


def candidate_archive(
    streams: tuple[bytes, bytes, bytes],
    tokens: bytes,
    *,
    token_codec: str,
    model_codec: str,
    zip_codec: str,
    zip_level: int | None,
) -> bytes:
    return deterministic_zip(
        payload_for(
            streams,
            tokens,
            token_codec=token_codec,
            model_codec=model_codec,
        ),
        codec=zip_codec,
        level=zip_level,
    )


def input_stage(args: argparse.Namespace) -> tuple[Sections, Sections, bytes, dict[str, Any]]:
    if args.base_archive.stat().st_size != EXPECTED_BASE_BYTES:
        raise ValueError("base archive byte count differs from the charter pin")
    if sha256_file(args.base_archive) != EXPECTED_BASE_SHA256:
        raise ValueError("base archive SHA-256 differs from the charter pin")
    if args.sd1_archive.stat().st_size != EXPECTED_SD1_BYTES:
        raise ValueError("SD1 archive byte count differs from the charter pin")
    if sha256_file(args.sd1_archive) != EXPECTED_SD1_SHA256:
        raise ValueError("SD1 archive SHA-256 differs from the charter pin")
    base = legacy_sections(args.base_archive)
    sd1 = legacy_sections(args.sd1_archive)
    if sha256_bytes(sd1.semantic) != EXPECTED_SD1_SEMANTIC_SHA256:
        raise ValueError("SD1 semantic blob differs from the selected n600 object")
    if not sd1.semantic.startswith(inflate.SEMANTIC_MIXED_MAGIC):
        raise ValueError("selected semantic blob lacks the counted SD1M header")
    if (sd1.carrier, sd1.hpac, sd1.tokens) != (
        base.carrier,
        base.hpac,
        base.tokens,
    ):
        raise ValueError("SD1 changed carrier, HPAC, or Range token bytes")
    if len(base.tokens) != EXPECTED_RANGE_BYTES:
        raise ValueError("base Range token byte count differs")
    if sha256_bytes(base.tokens) != EXPECTED_RANGE_SHA256:
        raise ValueError("base Range token SHA-256 differs")

    if sha256_file(args.dt1_receipt) != EXPECTED_DT1_RECEIPT_SHA256:
        raise ValueError("DT1 terminal receipt SHA-256 differs from the CX2 pin")
    dt1_receipt = json.loads(args.dt1_receipt.read_text())
    if dt1_receipt.get("schema") != "ddm_dt1_retained_n600.v1":
        raise ValueError("DT1 receipt schema is not the retained n600 contract")
    if dt1_receipt.get("complete") is not True:
        raise ValueError("DT1 retained n600 receipt is not complete")
    provenance = dt1_receipt.get("provenance", {})
    if (
        provenance.get("archive_bytes") != EXPECTED_BASE_BYTES
        or provenance.get("archive_sha256") != EXPECTED_BASE_SHA256
        or provenance.get("recorded_range_bytes") != EXPECTED_RANGE_BYTES
        or provenance.get("recorded_range_sha256") != EXPECTED_RANGE_SHA256
        or provenance.get("recorded_range_equals_archive") is not True
        or provenance.get("models_raw_sha256") != EXPECTED_MODELS_RAW_SHA256
    ):
        raise ValueError("DT1 receipt is not bound to the pinned PR130 object")
    streams = dt1_receipt.get("streams", {})
    range_stream = streams.get("range", {})
    ans_stream = streams.get("ans", {})
    if (
        range_stream.get("byte_identical_to_shipped") is not True
        or range_stream.get("bytes") != EXPECTED_RANGE_BYTES
        or range_stream.get("sha256") != EXPECTED_RANGE_SHA256
    ):
        raise ValueError("DT1 did not pass the byte-identical Range control")
    if (
        ans_stream.get("bytes") != EXPECTED_ANS_BYTES
        or Path(ans_stream.get("path", "")).resolve()
        != args.ans_payload.resolve()
    ):
        raise ValueError("DT1 ANS declaration is not bound to the selected payload")
    range_decode = dt1_receipt.get("range_decode", {})
    ans_decode = dt1_receipt.get("ans_decode", {})
    if ans_decode.get("exact_target_equality") is not True:
        raise ValueError("DT1 did not decode the retained n600 ANS payload exactly")
    if ans_decode.get("all_tokens_reconstructed") is not True:
        raise ValueError("DT1 did not reconstruct all n600 ANS tokens")
    for label, decoded in (("Range", range_decode), ("ANS", ans_decode)):
        if (
            decoded.get("frames") != EXPECTED_FRAMES
            or decoded.get("tokens") != EXPECTED_TOKENS
            or decoded.get("exact_target_equality") is not True
            or decoded.get("all_tokens_reconstructed") is not True
        ):
            raise ValueError(f"DT1 {label} denominator or equality proof differs")
    if range_decode.get("decoded_sha256") != ans_decode.get("decoded_sha256"):
        raise ValueError("DT1 Range and ANS decoded target hashes differ")
    ans = args.ans_payload.read_bytes()
    if len(ans) != EXPECTED_ANS_BYTES:
        raise ValueError("retained ANS payload differs from measured length")
    if sha256_bytes(ans) != ans_stream.get("sha256"):
        raise ValueError("retained ANS payload differs from the DT1 receipt")

    versions = {
        "brotli": importlib.metadata.version("Brotli"),
        "constriction_builder": importlib.metadata.version("constriction"),
        "constriction_dt1": dt1_receipt["provenance"]["host"]["constriction"],
    }
    if versions["brotli"] != EXPECTED_BROTLI_VERSION:
        raise ValueError("builder Brotli version differs from the pinned runtime")
    if versions["constriction_dt1"] != EXPECTED_CONSTRICTION_VERSION:
        raise ValueError("DT1 ANS payload was not built with constriction 0.5.0")
    receipt = {
        "schema": "ddm_cx2_inputs.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[archive-byte exact; DT1 token proof macOS-CPU advisory]",
        "score_claim": False,
        "base": {
            "path": str(args.base_archive),
            "bytes": args.base_archive.stat().st_size,
            "sha256": sha256_file(args.base_archive),
        },
        "sd1": {
            "path": str(args.sd1_archive),
            "bytes": args.sd1_archive.stat().st_size,
            "sha256": sha256_file(args.sd1_archive),
            "semantic_bytes": len(sd1.semantic),
            "semantic_sha256": sha256_bytes(sd1.semantic),
        },
        "dt1": {
            "receipt": str(args.dt1_receipt),
            "receipt_sha256": sha256_file(args.dt1_receipt),
            "ans_path": str(args.ans_payload),
            "ans_bytes": len(ans),
            "ans_sha256": sha256_bytes(ans),
        },
        "versions": versions,
    }
    stage_path = args.resume_from / "stages" / "01_inputs.json"
    existing = load_complete(stage_path, "ddm_cx2_inputs.v1")
    if existing is not None:
        for key in ("axis", "score_claim", "base", "sd1", "dt1", "versions"):
            if existing.get(key) != receipt.get(key):
                raise ValueError(f"input checkpoint changed at field {key}")
        return base, sd1, ans, existing
    atomic_json(stage_path, receipt)
    return base, sd1, ans, receipt


def compress_variants(data: bytes, transforms: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for transform in transforms:
        transformed = transform_bytes(data, transform)
        for quality in (9, 10, 11):
            encoded = brotli.compress(transformed, quality=quality)
            rows.append(
                {
                    "transform": transform,
                    "brotli_quality": quality,
                    "bytes": len(encoded),
                    "sha256": sha256_bytes(encoded),
                    "encoded": encoded,
                }
            )
    return rows


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "encoded"}


def search_stage(
    args: argparse.Namespace,
    sd1: Sections,
    ans: bytes,
) -> tuple[bytes, dict[str, Any], tuple[bytes, bytes, bytes]]:
    stage_path = args.resume_from / "stages" / "02_search.json"
    existing = load_complete(stage_path, "ddm_cx2_complete_zip_search.v1")
    if existing is not None:
        artifacts = existing.get("checkpoint_artifacts", {})
        payloads: dict[str, bytes] = {}
        for name in ("archive", "semantic", "carrier", "hpac"):
            record = artifacts.get(name, {})
            path = Path(record.get("path", ""))
            if not path.is_file():
                raise ValueError(f"search checkpoint lacks {name} artifact")
            payload = path.read_bytes()
            if (
                len(payload) != record.get("bytes")
                or sha256_bytes(payload) != record.get("sha256")
            ):
                raise ValueError(f"search checkpoint {name} artifact changed")
            payloads[name] = payload
        if (
            payloads["archive"]
            != candidate_archive(
                (payloads["semantic"], payloads["carrier"], payloads["hpac"]),
                ans,
                token_codec="ans",
                model_codec="split_brotli_cx2",
                zip_codec="stored",
                zip_level=None,
            )
        ):
            raise ValueError("search checkpoint does not rebuild from selected streams")
        return (
            payloads["archive"],
            existing,
            (payloads["semantic"], payloads["carrier"], payloads["hpac"]),
        )

    transforms = (
        "identity",
        "reverse",
        "xor80",
        "signed_zigzag",
        "delta",
        "signed_zigzag_delta",
        "nibble_swap",
        "bit_reverse",
        "global_lane2",
        "block4096_lane2",
        "signed_zigzag_block4096_lane2",
    )
    semantic_rows = compress_variants(sd1.semantic, transforms)
    carrier_rows = compress_variants(sd1.carrier, ("identity",))
    hpac_rows = compress_variants(sd1.hpac, transforms)
    zip_policies = (("stored", None), ("deflate", 6))
    candidates: list[dict[str, Any]] = []
    selected_archive: bytes | None = None
    selected_streams: tuple[bytes, bytes, bytes] | None = None
    for semantic, carrier, hpac in itertools.product(
        semantic_rows,
        carrier_rows,
        hpac_rows,
    ):
        streams = semantic["encoded"], carrier["encoded"], hpac["encoded"]
        for zip_codec, zip_level in zip_policies:
            archive = candidate_archive(
                streams,
                ans,
                token_codec="ans",
                model_codec="split_brotli_cx2",
                zip_codec=zip_codec,
                zip_level=zip_level,
            )
            row = {
                "archive_bytes": len(archive),
                "archive_sha256": sha256_bytes(archive),
                "semantic": public_row(semantic),
                "carrier": public_row(carrier),
                "hpac": public_row(hpac),
                "zip": {"codec": zip_codec, "level": zip_level},
            }
            candidates.append(row)
    candidates.sort(
        key=lambda row: (
            row["archive_bytes"],
            row["archive_sha256"],
        )
    )
    winner = candidates[0]
    minimum_rows = [
        row for row in candidates
        if row["archive_bytes"] == winner["archive_bytes"]
    ]
    minimum_archives = {row["archive_sha256"] for row in minimum_rows}
    selected_equivalent_rows = [
        row for row in minimum_rows
        if row["archive_sha256"] == winner["archive_sha256"]
    ]
    observed_spec = {
        "semantic": {
            "transform": winner["semantic"]["transform"],
            "brotli_quality": winner["semantic"]["brotli_quality"],
        },
        "carrier": {
            "transform": winner["carrier"]["transform"],
            "brotli_quality": winner["carrier"]["brotli_quality"],
        },
        "hpac": {
            "transform": winner["hpac"]["transform"],
            "brotli_quality": winner["hpac"]["brotli_quality"],
        },
        "zip": winner["zip"],
    }
    if observed_spec != EXPECTED_SELECTED_SPEC:
        raise RuntimeError(
            "complete-ZIP search selected a transform not implemented by the "
            f"CX2 receiver: {observed_spec}"
        )
    semantic = next(
        row for row in semantic_rows
        if row["transform"] == observed_spec["semantic"]["transform"]
        and row["brotli_quality"] == observed_spec["semantic"]["brotli_quality"]
    )
    carrier = next(
        row for row in carrier_rows
        if row["transform"] == observed_spec["carrier"]["transform"]
        and row["brotli_quality"] == observed_spec["carrier"]["brotli_quality"]
    )
    hpac = next(
        row for row in hpac_rows
        if row["transform"] == observed_spec["hpac"]["transform"]
        and row["brotli_quality"] == observed_spec["hpac"]["brotli_quality"]
    )
    selected_streams = semantic["encoded"], carrier["encoded"], hpac["encoded"]
    selected_archive = candidate_archive(
        selected_streams,
        ans,
        token_codec="ans",
        model_codec="split_brotli_cx2",
        zip_codec="stored",
        zip_level=None,
    )
    if sha256_bytes(selected_archive) != winner["archive_sha256"]:
        raise RuntimeError("selected complete ZIP did not rebuild byte-identically")

    result = {
        "schema": "ddm_cx2_complete_zip_search.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[archive-byte exact, scorer-free]",
        "score_claim": False,
        "candidate_denominator": len(candidates),
        "semantic_variant_denominator": len(semantic_rows),
        "carrier_variant_denominator": len(carrier_rows),
        "hpac_variant_denominator": len(hpac_rows),
        "zip_policy_denominator": len(zip_policies),
        "selection_surface": "final deterministic archive.zip bytes",
        "selection_tiebreaker": (
            "minimum archive bytes, then lexicographic archive SHA-256; "
            "byte-identical parameter rows retain enumeration order"
        ),
        "minimum_byte_tie_denominator": len(minimum_rows),
        "minimum_unique_archive_denominator": len(minimum_archives),
        "selected_archive_equivalent_parameter_denominator": len(
            selected_equivalent_rows
        ),
        "selected_parameterization_is_unique": len(selected_equivalent_rows) == 1,
        "selected_archive_is_unique_minimum": len(minimum_archives) == 1,
        "selected": winner,
        "top_100": candidates[:100],
        "section_variants": {
            "semantic": [public_row(row) for row in semantic_rows],
            "carrier": [public_row(row) for row in carrier_rows],
            "hpac": [public_row(row) for row in hpac_rows],
        },
    }
    checkpoint_payloads = {
        "archive": selected_archive,
        "semantic": selected_streams[0],
        "carrier": selected_streams[1],
        "hpac": selected_streams[2],
    }
    checkpoint_artifacts: dict[str, dict[str, Any]] = {}
    for name, payload in checkpoint_payloads.items():
        path = args.resume_from / "stages" / f"02_{name}.bin"
        atomic_bytes(path, payload)
        checkpoint_artifacts[name] = {
            "path": str(path),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
    result["checkpoint_artifacts"] = checkpoint_artifacts
    atomic_json(args.output / "search" / "complete_zip_search.json", result)
    atomic_json(stage_path, result)
    return selected_archive, result, selected_streams


def build_stage(
    args: argparse.Namespace,
    base: Sections,
    sd1: Sections,
    ans: bytes,
    final_archive: bytes,
    selected_streams: tuple[bytes, bytes, bytes],
) -> dict[str, Any]:
    stage_path = args.resume_from / "stages" / "03_build.json"
    existing = load_complete(stage_path, "ddm_cx2_build.v1")
    if existing is not None:
        archive_record = existing.get("archive", {})
        if (
            archive_record.get("bytes") != len(final_archive)
            or archive_record.get("sha256") != sha256_bytes(final_archive)
        ):
            raise ValueError("build checkpoint names a different selected archive")
        existing_artifacts = existing.get("artifacts", {})
        if set(existing_artifacts) != {
            "mixed_split_brotli_range.zip",
            "mixed_xcodec_range.zip",
            "archive.zip",
        }:
            raise ValueError("build checkpoint artifact denominator differs")
        for name, record in existing_artifacts.items():
            path = Path(record.get("path", ""))
            if (
                not path.is_file()
                or path.stat().st_size != record.get("bytes")
                or sha256_file(path) != record.get("sha256")
            ):
                raise ValueError(f"build checkpoint artifact changed: {name}")
        return existing

    identity_streams = tuple(
        brotli.compress(section, quality=11)
        for section in (sd1.semantic, sd1.carrier, sd1.hpac)
    )
    mixed_split_range = candidate_archive(
        identity_streams,
        base.tokens,
        token_codec="range",
        model_codec="split_brotli",
        zip_codec="stored",
        zip_level=None,
    )
    mixed_xcodec_range = candidate_archive(
        selected_streams,
        base.tokens,
        token_codec="range",
        model_codec="split_brotli_cx2",
        zip_codec="stored",
        zip_level=None,
    )
    artifacts = {
        "mixed_split_brotli_range.zip": mixed_split_range,
        "mixed_xcodec_range.zip": mixed_xcodec_range,
        "archive.zip": final_archive,
    }
    artifact_records: dict[str, dict[str, Any]] = {}
    for name, payload in artifacts.items():
        destination = (
            args.output / "composed" / name
            if name == "archive.zip"
            else args.output / "comparators" / name
        )
        atomic_bytes(destination, payload)
        artifact_records[name] = {
            "path": str(destination),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }

    final_path = args.output / "composed" / "archive.zip"
    rows = [
        {
            "stage": "PR130 CPR1 base; q4 + joint XZ + Range",
            "archive_bytes": EXPECTED_BASE_BYTES,
            "delta_bytes_from_previous": None,
        },
        {
            "stage": "SD1 mixed semantic + joint XZ + Range",
            "archive_bytes": EXPECTED_SD1_BYTES,
            "delta_bytes_from_previous": EXPECTED_SD1_BYTES - EXPECTED_BASE_BYTES,
        },
        {
            "stage": "SD1 mixed + split Brotli + Range",
            "archive_bytes": len(mixed_split_range),
            "delta_bytes_from_previous": len(mixed_split_range) - EXPECTED_SD1_BYTES,
        },
        {
            "stage": "SD1 mixed + CX2 xcodec split Brotli + Range",
            "archive_bytes": len(mixed_xcodec_range),
            "delta_bytes_from_previous": len(mixed_xcodec_range) - len(mixed_split_range),
        },
        {
            "stage": "SD1 mixed + CX2 xcodec split Brotli + ANS",
            "archive_bytes": len(final_archive),
            "delta_bytes_from_previous": len(final_archive) - len(mixed_xcodec_range),
        },
    ]
    additive_prediction = EXPECTED_BASE_BYTES - 848 - 903
    interactions = {
        "sd1_by_split_brotli_bytes": len(mixed_split_range) - additive_prediction,
        "cx2_xcodec_on_mixed_split_bytes": len(mixed_xcodec_range) - len(mixed_split_range),
        "ans_on_final_model_pack_bytes": len(final_archive) - len(mixed_xcodec_range),
    }
    result = {
        "schema": "ddm_cx2_build.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[archive-byte exact, scorer-free]",
        "score_claim": False,
        "archive": {
            "path": str(final_path),
            "bytes": len(final_archive),
            "sha256": sha256_bytes(final_archive),
            "rate_term": 25 * len(final_archive) / EXPECTED_ORIGINAL_BYTES,
        },
        "artifacts": artifact_records,
        "stages": rows,
        "interactions": interactions,
        "streams": {
            "semantic_brotli_bytes": len(selected_streams[0]),
            "carrier_brotli_bytes": len(selected_streams[1]),
            "hpac_brotli_bytes": len(selected_streams[2]),
            "split_header_bytes": receiver.SPLIT_HEADER.size,
            "outer_header_bytes": receiver.OUTER_HEADER.size,
            "ans_bytes": len(ans),
            "zip_overhead_bytes": len(final_archive) - len(read_member(final_path)),
        },
    }
    atomic_json(args.output / "composed" / "build_receipt.json", result)
    atomic_json(stage_path, result)
    return result


def tensor_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(struct.pack("<I", array.ndim))
        digest.update(struct.pack(f"<{array.ndim}Q", *array.shape))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def parseback_stage(
    args: argparse.Namespace,
    sd1: Sections,
    ans: bytes,
    build: dict[str, Any],
) -> dict[str, Any]:
    archive_path = Path(build["archive"]["path"])
    runtime_sha256 = {
        "inflate.py": sha256_file(RUNTIME / "inflate.py"),
        "receiver.py": sha256_file(RUNTIME / "receiver.py"),
    }
    stage_path = args.resume_from / "stages" / "04_parseback.json"
    existing = load_complete(stage_path, "ddm_cx2_parseback.v1")
    if existing is not None:
        if (
            existing.get("archive", {}).get("sha256")
            != build["archive"]["sha256"]
            or existing.get("runtime_sha256") != runtime_sha256
        ):
            raise ValueError("parse-back checkpoint runtime or archive changed")
        return existing
    if sha256_file(archive_path) != build["archive"]["sha256"]:
        raise ValueError("composed archive changed before parse-back")
    payload = read_member(archive_path)
    parts = receiver.split_payload(payload)
    if parts.token_codec != "ans" or parts.model_codec != "split_brotli_cx2":
        raise ValueError("composed outer selectors do not name the CX2 wire form")
    decoded = receiver.decode_models(parts.models, model_codec=parts.model_codec)
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", decoded.raw)
    semantic_end = 8 + semantic_bytes
    carrier_end = semantic_end + carrier_bytes
    semantic = decoded.raw[8:semantic_end]
    carrier = decoded.raw[semantic_end:carrier_end]
    hpac = decoded.raw[carrier_end:]
    if (semantic, carrier, hpac, parts.tokens) != (
        sd1.semantic,
        sd1.carrier,
        sd1.hpac,
        ans,
    ):
        raise ValueError("composed archive failed exact four-section parse-back")

    template = inflate.SemanticTokenRenderer(96).state_dict()
    allocation, _, format_name = inflate.semantic_allocation(semantic, template)
    expected_q3 = {
        "frame_embed.weight",
        "blocks.1.film.weight",
        "blocks.2.film.weight",
        "blocks.3.film.weight",
    }
    observed_q3 = {name for name, bits in allocation.items() if bits == 3}
    if observed_q3 != expected_q3 or any(
        bits not in (3, 4) for bits in allocation.values()
    ):
        raise ValueError("runtime did not consume the selected SD1 allocation")
    runtime_state = inflate.unpack_semantic(semantic, template)
    semantic_model, basis, coefficients = inflate.unpack_semantic_pose(
        decoded.raw[:carrier_end]
    )
    hpac_model = inflate.load_hpac(decoded.raw[carrier_end:], torch.device("cpu"))
    if list(runtime_state) != list(semantic_model.state_dict()):
        raise ValueError("semantic loader key order differs after full model load")
    for name, value in runtime_state.items():
        if not torch.equal(value, semantic_model.state_dict()[name]):
            raise ValueError(f"full semantic loader changed tensor {name}")
    result = {
        "schema": "ddm_cx2_parseback.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU parse-back; scorer-free]",
        "score_claim": False,
        "archive": build["archive"],
        "runtime_sha256": runtime_sha256,
        "outer": {
            "model_codec": parts.model_codec,
            "token_codec": parts.token_codec,
            "member_bytes": len(payload),
            "member_sha256": sha256_bytes(payload),
            "models_raw_bytes": len(decoded.raw),
            "models_raw_sha256": sha256_bytes(decoded.raw),
            "token_payload_sha256": sha256_bytes(parts.tokens),
        },
        "sections": {
            "semantic": {
                "bytes": len(semantic),
                "sha256": sha256_bytes(semantic),
                "format": format_name,
                "allocation": allocation,
                "state_sha256": tensor_sha256(runtime_state),
                "tensor_count": len(runtime_state),
            },
            "carrier": {
                "bytes": len(carrier),
                "sha256": sha256_bytes(carrier),
                "basis_shape": list(basis.shape),
                "coefficient_shape": list(coefficients.shape),
            },
            "hpac": {
                "bytes": len(hpac),
                "sha256": sha256_bytes(hpac),
                "state_tensor_count": len(hpac_model.state_dict()),
            },
            "tokens": {
                "bytes": len(parts.tokens),
                "sha256": sha256_bytes(parts.tokens),
                "dt1_decode_receipt": str(args.dt1_receipt),
                "dt1_decode_receipt_sha256": sha256_file(args.dt1_receipt),
            },
        },
        "all_sections_consumed": True,
        "legacy_q4_identity_covered_by": (
            "src/tac/tests/test_ddm_cx2_compose_end_to_end.py"
        ),
    }
    atomic_json(args.output / "composed" / "parseback_receipt.json", result)
    atomic_json(stage_path, result)
    return result


def submission_stage(
    args: argparse.Namespace,
    build: dict[str, Any],
) -> dict[str, Any]:
    submission = args.output / "submission"
    archive_dir = submission / "archive"
    inflated_dir = submission / "inflated"
    stage_path = args.resume_from / "stages" / "05_submission.json"
    existing = load_complete(stage_path, "ddm_cx2_submission.v1")
    if existing is not None:
        evaluator_archive = existing.get("evaluator_archive", {})
        member = existing.get("archive_member", {})
        for record, label in (
            (evaluator_archive, "evaluator archive"),
            (member, "archive member"),
        ):
            path = Path(record.get("path", ""))
            if (
                not path.is_file()
                or path.stat().st_size != record.get("bytes")
                or sha256_file(path) != record.get("sha256")
            ):
                raise ValueError(f"submission checkpoint {label} changed")
        if evaluator_archive.get("sha256") != build["archive"]["sha256"]:
            raise ValueError("submission checkpoint names a different archive")
        existing_runtime = existing.get("runtime_files", {})
        if set(existing_runtime) != set(RUNTIME_FILES):
            raise ValueError("submission checkpoint runtime denominator differs")
        for name, record in existing_runtime.items():
            source = RUNTIME / name
            destination = Path(record.get("destination", ""))
            expected = record.get("sha256")
            if (
                name not in RUNTIME_FILES
                or not source.is_file()
                or not destination.is_file()
                or sha256_file(source) != expected
                or sha256_file(destination) != expected
            ):
                raise ValueError(f"submission checkpoint runtime changed: {name}")
        return existing

    archive_dir.mkdir(parents=True, exist_ok=True)
    if inflated_dir.exists() and any(inflated_dir.iterdir()):
        raise RuntimeError(
            "refusing to prepare a new submission over existing inflated outputs"
        )
    inflated_dir.mkdir(parents=True, exist_ok=True)
    archive_path = Path(build["archive"]["path"])
    evaluator_archive_path = submission / "archive.zip"
    member_path = archive_dir / "p"
    atomic_bytes(evaluator_archive_path, archive_path.read_bytes())
    atomic_bytes(member_path, read_member(archive_path))
    if sha256_file(evaluator_archive_path) != build["archive"]["sha256"]:
        raise RuntimeError("evaluator-facing archive copy differs from composed archive")
    if member_path.read_bytes() != read_member(evaluator_archive_path):
        raise RuntimeError("expanded archive member differs from evaluator-facing ZIP")
    copied: dict[str, Any] = {}
    for name in RUNTIME_FILES:
        source = RUNTIME / name
        payload = source.read_bytes()
        destination = submission / name
        atomic_bytes(destination, payload, executable=name == "inflate.sh")
        copied[name] = {
            "source": str(source),
            "destination": str(destination),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
    result = {
        "schema": "ddm_cx2_submission.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[archive-byte exact; runtime copied by explicit allowlist]",
        "score_claim": False,
        "submission_dir": str(submission),
        "source_archive": build["archive"],
        "evaluator_archive": {
            "path": str(evaluator_archive_path),
            "bytes": evaluator_archive_path.stat().st_size,
            "sha256": sha256_file(evaluator_archive_path),
        },
        "archive_member": {
            "path": str(member_path),
            "bytes": member_path.stat().st_size,
            "sha256": sha256_file(member_path),
        },
        "runtime_files": copied,
        "excluded_untracked_runtime_files": True,
        "inflated_output_policy": (
            "created empty; a literal receiver run must populate it and bind "
            "the raw to this archive before scoring"
        ),
    }
    atomic_json(submission / "submission_manifest.json", result)
    atomic_json(stage_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-archive", type=Path, default=BASE_ARCHIVE)
    parser.add_argument("--sd1-archive", type=Path, default=SD1_ARCHIVE)
    parser.add_argument(
        "--ans-payload",
        type=Path,
        default=DT1_ROOT / "retained" / "ans_n600.bin",
    )
    parser.add_argument(
        "--dt1-receipt",
        type=Path,
        default=DT1_ROOT / "retained" / "retained_n600_result.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--minimum-free-bytes", type=int, default=2 << 30)
    args = parser.parse_args()
    if args.minimum_free_bytes <= 0:
        parser.error("--minimum-free-bytes must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.resume_from.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(args.output)
    if usage.free < args.minimum_free_bytes:
        raise RuntimeError(
            f"storage preflight refused: free={usage.free}, "
            f"required={args.minimum_free_bytes}"
        )
    base, sd1, ans, inputs = input_stage(args)
    archive, search, selected_streams = search_stage(args, sd1, ans)
    build = build_stage(args, base, sd1, ans, archive, selected_streams)
    parseback = parseback_stage(args, sd1, ans, build)
    submission = submission_stage(args, build)
    result = {
        "schema": "ddm_cx2_build_pipeline.v1",
        "complete": False,
        "build_complete": True,
        "written_at_utc": utc_now(),
        "axis": "[archive-byte exact + macOS-CPU receiver parse-back; score_claim=false]",
        "score_claim": False,
        "pending_terminal_gates": [
            "literal current inflate.sh reconstructs the exact n600 token stream",
            "inflated raw is size/hash-bound atomically to this archive and runtime",
            "immutable upstream evaluate.py measures n600 SegNet and PoseNet",
        ],
        "storage_preflight": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "required_free_bytes": args.minimum_free_bytes,
        },
        "inputs": inputs,
        "search": {
            "path": str(args.output / "search" / "complete_zip_search.json"),
            "sha256": sha256_file(
                args.output / "search" / "complete_zip_search.json"
            ),
            "candidate_denominator": search["candidate_denominator"],
        },
        "build": build,
        "parseback": parseback,
        "submission": submission,
    }
    atomic_json(args.output / "CX2_PIPELINE_RECEIPT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
