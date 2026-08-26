#!/usr/bin/env python3
"""Exact RB1 archive and receiver bridge for a born-small field plus WD3 renderer.

The counted archive keeps the retained BS3 body byte-for-byte and appends one
Brotli-q11 WD3 renderer stream.  Decode validates both identities, reconstructs
the BS3 token field through the retained HG1 packet grammar, and dispatches the
renderer through the same WD3 branch installed into the shipped GB1 runtime.
"""

from __future__ import annotations

import hashlib
import io
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
from torch.nn import functional as F

EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from experiments import ddm_hg1_heterogeneous_analytic_generator_gate as hg1
from experiments import ddm_wd3_student_receiver as wd3_receiver

MAGIC = b"RB1A"
VERSION = 1
HEADER = struct.Struct("<4sB3xII32s32s")


class RB1ReceiverError(RuntimeError):
    """The counted body, renderer, or receiver parse-back failed closed."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RB1ReceiverError(f"shipped GB1 patch point differs: {label}")
    return source.replace(old, new)


def patch_gb1_runtime_tree(source_tree: Path, destination: Path) -> dict[str, Any]:
    """Add the WD3Q semantic branch to the exact shipped GB1 runtime."""

    source_tree = source_tree.resolve(strict=True)
    if destination.exists():
        raise RB1ReceiverError(f"runtime patch destination already exists: {destination}")
    shutil.copytree(source_tree, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    residual_path = destination / "runtime/residual_archive.py"
    residual = residual_path.read_text(encoding="utf-8")
    residual = _replace_once(
        residual,
        'tagged_semantic = semantic_body.startswith((b"SD1M", b"SM3R"))',
        'tagged_semantic = semantic_body.startswith((b"SD1M", b"SM3R", b"WD3Q"))',
        "residual WD3Q semantic dispatch",
    )
    residual_path.write_text(residual, encoding="utf-8")

    f26_path = destination / "runtime/f26_inflate.py"
    f26 = f26_path.read_text(encoding="utf-8")
    f26 = _replace_once(
        f26,
        '''    if not parts.semantic_blob.startswith((WANS1_MAGIC, b"SD1M", b"SM3R")):
        raise InflationError("F26 requires WANS1, SD1M, or SM3R semantic weights")
''',
        '''    if not parts.semantic_blob.startswith((WANS1_MAGIC, b"SD1M", b"SM3R", b"WD3Q")):
        raise InflationError("F26 requires WANS1, SD1M, SM3R, or WD3Q semantic weights")
''',
        "F26 WD3Q semantic guard",
    )
    old_load = '''    semantic = renderer.SemanticTokenRenderer(96)
    tagged_state = renderer.unpack_variant_semantic_or_none(
        parts.semantic_blob,
        semantic.state_dict(),
    )
    if tagged_state is None:
        records = decode_wans1(parts.semantic_blob)
        tagged_state = {
            record.schema.name: torch.from_numpy(
                np.ascontiguousarray(record.values, dtype=np.float32)
            )
            for record in records
        }
    semantic.load_state_dict(tagged_state, strict=True)
'''
    new_load = '''    if parts.semantic_blob.startswith(b"WD3Q"):
        receiver_path = renderer_dir / "wd3_receiver.py"
        receiver_spec = importlib.util.spec_from_file_location("_f26_wd3_receiver", receiver_path)
        if receiver_spec is None or receiver_spec.loader is None:
            raise InflationError("could not load the counted WD3 receiver")
        receiver = importlib.util.module_from_spec(receiver_spec)
        sys.modules[receiver_spec.name] = receiver
        receiver_spec.loader.exec_module(receiver)
        semantic = receiver.unpack_student(parts.semantic_blob)
    else:
        semantic = renderer.SemanticTokenRenderer(96)
        tagged_state = renderer.unpack_variant_semantic_or_none(
            parts.semantic_blob,
            semantic.state_dict(),
        )
        if tagged_state is None:
            records = decode_wans1(parts.semantic_blob)
            tagged_state = {
                record.schema.name: torch.from_numpy(
                    np.ascontiguousarray(record.values, dtype=np.float32)
                )
                for record in records
            }
        semantic.load_state_dict(tagged_state, strict=True)
'''
    f26_path.write_text(
        _replace_once(f26, old_load, new_load, "F26 WD3Q model construction"),
        encoding="utf-8",
    )
    shutil.copy2(Path(wd3_receiver.wd2.__file__).resolve(), destination / "cpr1/wd2_receiver.py")
    shutil.copy2(Path(wd3_receiver.__file__).resolve(), destination / "cpr1/wd3_receiver.py")
    files = [
        {
            "path": path.relative_to(destination).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_record(path)["sha256"],
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    return {
        "schema": "ddm_rb1_gb1_runtime_patch.v1",
        "source_tree": str(source_tree),
        "destination": str(destination.resolve()),
        "student_magic": wd3_receiver.MAGIC.decode("ascii"),
        "inactive_wans_sd1m_sm3r_branches_retained": True,
        "files": files,
    }


def _zip_member(payload: bytes) -> bytes:
    destination = io.BytesIO()
    info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(destination, mode="w") as archive:
        archive.writestr(info, payload)
    return destination.getvalue()


def pack_archive_bytes(body_archive: bytes, semantic_stream: bytes) -> bytes:
    header = HEADER.pack(
        MAGIC,
        VERSION,
        len(body_archive),
        len(semantic_stream),
        hashlib.sha256(body_archive).digest(),
        hashlib.sha256(semantic_stream).digest(),
    )
    return _zip_member(header + body_archive + semantic_stream)


def write_archive(path: Path, body_archive: bytes, semantic_stream: bytes) -> dict[str, Any]:
    payload = pack_archive_bytes(body_archive, semantic_stream)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_record(path)


def parse_archive_bytes(payload: bytes) -> tuple[bytes, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if archive.namelist() != ["p"]:
                raise RB1ReceiverError("RB1 archive must contain exactly member p")
            member = archive.read("p")
    except zipfile.BadZipFile as error:
        raise RB1ReceiverError("RB1 ZIP framing is invalid") from error
    if len(member) < HEADER.size:
        raise RB1ReceiverError("RB1 member is truncated")
    magic, version, body_n, semantic_n, body_sha, semantic_sha = HEADER.unpack_from(member)
    if magic != MAGIC or version != VERSION:
        raise RB1ReceiverError("RB1 archive magic/version differs")
    if len(member) != HEADER.size + body_n + semantic_n:
        raise RB1ReceiverError("RB1 counted sections do not close")
    body = member[HEADER.size : HEADER.size + body_n]
    semantic = member[HEADER.size + body_n :]
    if hashlib.sha256(body).digest() != body_sha or hashlib.sha256(semantic).digest() != semantic_sha:
        raise RB1ReceiverError("RB1 counted section identity differs")
    return body, semantic


def parse_archive(path: Path) -> tuple[bytes, bytes]:
    return parse_archive_bytes(path.read_bytes())


def parse_body_packet(body_archive: bytes) -> tuple[dict[str, bytes], bytes]:
    with zipfile.ZipFile(io.BytesIO(body_archive)) as archive:
        if archive.namelist() != ["p"]:
            raise RB1ReceiverError("nested BS3 body must contain exactly member p")
        member = archive.read("p")
    if len(member) < hg1.COMPLETE_HEADER.size:
        raise RB1ReceiverError("nested BS3 body is truncated")
    magic, version, semantic_n, carrier_n, residual_n, packet_n = hg1.COMPLETE_HEADER.unpack_from(member)
    if magic != hg1.COMPLETE_MAGIC or version != hg1.COMPLETE_VERSION:
        raise RB1ReceiverError("nested BS3 body grammar differs")
    expected = hg1.COMPLETE_HEADER.size + semantic_n + carrier_n + residual_n + packet_n
    if len(member) != expected:
        raise RB1ReceiverError("nested BS3 sections do not close")
    cursor = hg1.COMPLETE_HEADER.size
    sections: dict[str, bytes] = {}
    for name, size in (
        ("semantic_renderer", semantic_n),
        ("pose_carrier", carrier_n),
        ("compact_residual", residual_n),
    ):
        sections[name] = member[cursor : cursor + size]
        cursor += size
    packet = member[cursor:]
    if len(packet) != packet_n:
        raise RB1ReceiverError("nested BS3 packet length differs")
    hg1.parse_packet(packet)
    return sections, packet


def unpack_renderer(semantic_stream: bytes) -> wd3_receiver.StudentSemanticRenderer:
    try:
        packet = brotli.decompress(semantic_stream)
        allocation = wd3_receiver.packet_allocation(packet)
        model = wd3_receiver.unpack_student(packet)
    except (brotli.error, wd3_receiver.WD3ReceiverError) as error:
        raise RB1ReceiverError("RB1 WD3 renderer stream is invalid") from error
    if wd3_receiver.pack_student(model, allocation) != packet:
        raise RB1ReceiverError("RB1 WD3 renderer parse-back is not byte-idempotent")
    return model.eval()


def decode_body_tokens(body_archive: bytes, destination: Path) -> dict[str, Any]:
    _, packet = parse_body_packet(body_archive)
    result = hg1.decode_packet_to_file(packet, destination)
    if result.get("corrections") != 0:
        raise RB1ReceiverError("RB1 born-small packet unexpectedly requires corrections")
    return result


@torch.no_grad()
def render_camera_uint8(
    model: wd3_receiver.StudentSemanticRenderer,
    tokens: torch.Tensor,
    pair_indices: torch.Tensor,
) -> torch.Tensor:
    master = F.interpolate(
        model(tokens.long(), pair_indices),
        size=(wd3_receiver.CAMERA_H, wd3_receiver.CAMERA_W),
        mode="bilinear",
        align_corners=False,
    )
    return master.clamp(0.0, 255.0).round().to(torch.uint8)


def token_memmap(path: Path) -> np.memmap:
    return np.memmap(
        path,
        mode="r",
        dtype=np.uint8,
        shape=(wd3_receiver.N, wd3_receiver.EVAL_H, wd3_receiver.EVAL_W),
    )


__all__ = [
    "HEADER",
    "MAGIC",
    "VERSION",
    "RB1ReceiverError",
    "decode_body_tokens",
    "file_record",
    "pack_archive_bytes",
    "parse_archive",
    "parse_archive_bytes",
    "parse_body_packet",
    "patch_gb1_runtime_tree",
    "render_camera_uint8",
    "sha256_bytes",
    "token_memmap",
    "unpack_renderer",
    "write_archive",
]
