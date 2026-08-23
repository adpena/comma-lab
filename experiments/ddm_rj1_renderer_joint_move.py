#!/usr/bin/env python3
"""Build scorer-free DX2 renderer re-representation rungs for RJ1.

This module deliberately stops before the two mechanism gates that require
scorer ownership: exact-object pose compensation and carrier re-solve.  Every
serialized object is retained, receiver-parsed, and marked non-candidate until
those gates are complete.  It must never be used to infer SegNet, PoseNet, or
score behavior from representation agreement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import struct
import sys
import zipfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _root in (REPO, SRC):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_wd2_student_receiver as receiver
from experiments import ddm_wd4_warm_lineage_width as wd4

DX2_RUNTIME = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2")
DX2_ARCHIVE = DX2_RUNTIME / "archive.zip"
DX2_ARCHIVE_BYTES = 180_368
DX2_ARCHIVE_SHA256 = "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"
DX2_SEMANTIC_SHA256 = "17e0fd0b197ac147afe98397ef38f02f7915b69372d03c042e6be6fa0f992e50"
DX2_RESERVED = 0x1A
OUTPUT_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_rj1_renderer_joint_move/precompile_r1")
SUPERSEDED_LOCAL_ROOT = REPO / ".omx/tmp/arm_receipts_local/ddm_rj1_renderer_joint_move/precompile_r1"
RX1_HEADER = struct.Struct("<4sBBBBHHH")
RATE_EXCHANGE_S_PER_BYTE = 6.658590e-7
FIXED_DISTORTION_DEMAND_BYTES = 42_382
ZERO_DISTORTION_DEMAND_BYTES = 150
AXIS = "[macOS-CPU scorer-free exact byte/container + receiver parse-back]"
SCORER_STATUS = "NOT_RUN_CHARTER_FORBIDS_SCORER_LANE"


class RJ1Error(RuntimeError):
    """Raised when RJ1 custody or fail-closed mechanism gates are violated."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RJ1Error(f"required retained file is absent: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_bytes(path: Path, value: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != value:
            raise RJ1Error(f"refusing to overwrite differing retained payload: {path}")
        return file_record(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_record(path)


def atomic_json(path: Path, value: object) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return atomic_bytes(path, payload)


def atomic_torch(path: Path, value: object) -> dict[str, Any]:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return atomic_bytes(path, buffer.getvalue())


def verify_retained_file_records(value: object) -> None:
    """Fail closed if any file record in a completed receipt changed on disk."""

    if isinstance(value, Mapping):
        if {"path", "bytes", "sha256"}.issubset(value):
            path = Path(str(value["path"]))
            observed = file_record(path)
            expected = {key: value[key] for key in ("path", "bytes", "sha256")}
            if observed != expected:
                raise RJ1Error(f"retained payload custody changed: {path}")
        for nested in value.values():
            verify_retained_file_records(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            verify_retained_file_records(nested)


def write_or_verify_custody_inventory(root: Path, minimum_free_bytes: int) -> dict[str, Any]:
    """Inventory the retained tree without making the manifest self-referential."""

    destination = root / "CUSTODY_INVENTORY.json"
    records = []
    tree_digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == destination or path.name.startswith(f".{destination.name}."):
            continue
        record = file_record(path)
        relative = path.relative_to(root).as_posix()
        records.append({"relative_path": relative, **record})
        tree_digest.update(relative.encode())
        tree_digest.update(b"\0")
        tree_digest.update(str(record["bytes"]).encode())
        tree_digest.update(b"\0")
        tree_digest.update(str(record["sha256"]).encode())
        tree_digest.update(b"\n")
    manifest = {
        "schema": "ddm_rj1_retained_tree.v1",
        "root": str(root.resolve()),
        "files": records,
        "file_count": len(records),
        "payload_bytes": sum(int(record["bytes"]) for record in records),
        "tree_sha256": tree_digest.hexdigest(),
        "command": ".venv/bin/python experiments/ddm_rj1_renderer_joint_move.py",
        "config": {
            "minimum_free_bytes": minimum_free_bytes,
            "output": str(root.resolve()),
        },
    }
    if destination.is_file():
        observed = json.loads(destination.read_text(encoding="utf-8"))
        if observed != manifest:
            raise RJ1Error("retained custody inventory differs from the tree")
        return file_record(destination)
    return atomic_json(destination, manifest)


def deterministic_zip(member: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, member)
    return output.getvalue()


def storage_preflight(output: Path, required_free_bytes: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    resolved = output.resolve()
    expected = OUTPUT_ROOT.resolve()
    if resolved != expected:
        raise RJ1Error(f"output must be the first-priority SSD root: {expected}")
    free = shutil.disk_usage(output).free
    if free < required_free_bytes:
        raise RJ1Error(f"need {required_free_bytes} free bytes, observed {free}")
    return {
        "status": "PASS",
        "root": str(resolved),
        "required_free_bytes": required_free_bytes,
        "observed_free_bytes": free,
        "routing_reason": (
            "Vertigo is the first-priority SSD tier and retained enough free bytes for "
            "this bounded build despite displaying 100% capacity"
        ),
        "cleanup_policy": "certify-or-block; no materialized payload is auto-deleted",
    }


def _clear_runtime_modules() -> dict[str, Any]:
    prior = {
        name: module
        for name, module in sys.modules.items()
        if name == "runtime" or name.startswith("runtime.") or name == "_f26_renderer"
    }
    for name in prior:
        sys.modules.pop(name, None)
    return prior


def require_dx2() -> dict[str, Any]:
    archive = file_record(DX2_ARCHIVE)
    if (archive["bytes"], archive["sha256"]) != (DX2_ARCHIVE_BYTES, DX2_ARCHIVE_SHA256):
        raise RJ1Error("DX2 archive custody changed")
    return archive


def load_dx2_state() -> OrderedDict[str, torch.Tensor]:
    """Decode the exact 38-tensor semantic state consumed by the DX2 receiver."""

    prior = _clear_runtime_modules()
    sys.path.insert(0, str(DX2_RUNTIME.resolve()))
    try:
        f26 = importlib.import_module("runtime.f26_inflate")
        parts = f26.read_residual_archive(DX2_ARCHIVE)
        if sha256_bytes(parts.semantic_blob) != DX2_SEMANTIC_SHA256:
            raise RJ1Error("DX2 decoded semantic state changed")
        renderer = f26._load_renderer(DX2_RUNTIME / "cpr1")
        template = renderer.SemanticTokenRenderer(96).state_dict()
        decoded = renderer.unpack_variant_semantic_or_none(parts.semantic_blob, template)
        if decoded is None or tuple(decoded) != tuple(template) or len(decoded) != 38:
            raise RJ1Error("DX2 semantic state is not the required 38-tensor renderer")
        return OrderedDict((name, value.detach().cpu().float().contiguous()) for name, value in decoded.items())
    finally:
        sys.path.pop(0)
        for name in list(sys.modules):
            if name == "runtime" or name.startswith("runtime.") or name == "_f26_renderer":
                sys.modules.pop(name, None)
        sys.modules.update(prior)


def source_container() -> dict[str, Any]:
    archive = DX2_ARCHIVE.read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        if bundle.namelist() != ["p"]:
            raise RJ1Error("DX2 ZIP members changed")
        member = bundle.read("p")
    magic, version, codec, table_mode, reserved, hpac_bytes, semantic_bytes, carrier_bytes = RX1_HEADER.unpack_from(
        member
    )
    if (magic, version, codec, table_mode, reserved) != (b"RX1M", 1, 2, 0, DX2_RESERVED):
        raise RJ1Error("DX2 RX1 header changed")
    offset = RX1_HEADER.size
    hpac = member[offset : offset + hpac_bytes]
    offset += hpac_bytes
    semantic = member[offset : offset + semantic_bytes]
    offset += semantic_bytes
    carrier = member[offset : offset + carrier_bytes]
    offset += carrier_bytes
    tail = member[offset:]
    if min(map(len, (hpac, semantic, carrier, tail))) <= 0:
        raise RJ1Error("DX2 container has an empty required section")
    if deterministic_zip(member) != archive:
        raise RJ1Error("DX2 identity rebuild is not byte-identical")
    return {
        "archive": archive,
        "member": member,
        "magic": magic,
        "version": version,
        "codec": codec,
        "table_mode": table_mode,
        "reserved": reserved,
        "hpac": hpac,
        "semantic": semantic,
        "carrier": carrier,
        "tail": tail,
    }


def ck2_interleave(body: bytes) -> bytes:
    span = len(body) & ~1
    values = np.frombuffer(body[:span], dtype=np.uint8)
    return values[0::2].tobytes() + values[1::2].tobytes() + body[span:]


def ck2_uninterleave(body: bytes) -> bytes:
    span = len(body) & ~1
    half = span // 2
    restored = np.empty(span, dtype=np.uint8)
    values = np.frombuffer(body[:span], dtype=np.uint8)
    restored[0::2] = values[:half]
    restored[1::2] = values[half:]
    return restored.tobytes() + body[span:]


def build_archive(container: Mapping[str, Any], packet: bytes) -> tuple[bytes, bytes, bytes]:
    semantic_stream = brotli.compress(ck2_interleave(packet), mode=brotli.MODE_GENERIC, quality=11)
    if ck2_uninterleave(brotli.decompress(semantic_stream)) != packet:
        raise RJ1Error("semantic CK2+Brotli round-trip differs")
    if max(len(container["hpac"]), len(semantic_stream), len(container["carrier"])) > 0xFFFF:
        raise RJ1Error("RX1 uint16 section ceiling exceeded")
    model = (
        RX1_HEADER.pack(
            container["magic"],
            container["version"],
            container["codec"],
            container["table_mode"],
            container["reserved"],
            len(container["hpac"]),
            len(semantic_stream),
            len(container["carrier"]),
        )
        + container["hpac"]
        + semantic_stream
        + container["carrier"]
    )
    member = model + container["tail"]
    return semantic_stream, member, deterministic_zip(member)


def _copy_common_state(source: Mapping[str, torch.Tensor], target: OrderedDict[str, torch.Tensor]) -> None:
    for name in target:
        if name in source and target[name].shape == source[name].shape:
            target[name] = source[name].clone()


def _canonical_svd(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU SVD with component signs fixed by the largest-magnitude U entry."""

    u, singular, vh = torch.linalg.svd(matrix.double(), full_matrices=False)
    for component in range(u.shape[1]):
        column = u[:, component]
        pivot = int(torch.argmax(column.abs()))
        if column[pivot] < 0:
            u[:, component].neg_()
            vh[component].neg_()
    return u.float(), singular.float(), vh.float()


def build_factorized_state(source: Mapping[str, torch.Tensor], *, rank: int) -> OrderedDict[str, torch.Tensor]:
    spec = receiver.StudentSpec("pointwise_svd_w96_r32", "factorized", 96, 4, rank)
    target = OrderedDict(receiver.StudentSemanticRenderer(spec).state_dict())
    _copy_common_state(source, target)
    for block in range(4):
        prefix = f"blocks.{block}"
        matrix = source[f"{prefix}.pw.weight"].reshape(96, 96)
        u, singular, vh = _canonical_svd(matrix)
        root = singular[:rank].sqrt()
        target[f"{prefix}.down.weight"] = (root[:, None] * vh[:rank]).reshape(rank, 96, 1, 1)
        target[f"{prefix}.down.bias"].zero_()
        target[f"{prefix}.up.weight"] = (u[:, :rank] * root[None, :]).reshape(96, rank, 1, 1)
        target[f"{prefix}.up.bias"] = source[f"{prefix}.pw.bias"].clone()
    return target


def build_flattened_state(source: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Amortize four per-block FiLM maps into one counted trunk FiLM map.

    The arithmetic mean is only a deterministic initialization.  It is not an
    equivalence or quality claim and is therefore withheld from scoring until
    exact-object joint optimization and compensation are complete.
    """

    spec = receiver.StudentSpec("film_amortized_flat_w96", "flattened", 96, 4)
    target = OrderedDict(receiver.StudentSemanticRenderer(spec).state_dict())
    _copy_common_state(source, target)
    target["flat_film.weight"] = torch.stack([source[f"blocks.{block}.film.weight"] for block in range(4)]).mean(dim=0)
    target["flat_film.bias"] = torch.stack([source[f"blocks.{block}.film.bias"] for block in range(4)]).mean(dim=0)
    return target


def representation_rungs(
    source: Mapping[str, torch.Tensor],
) -> list[tuple[receiver.StudentSpec, OrderedDict[str, torch.Tensor], str]]:
    salience = wd4.salience_order(source)
    dense_spec = receiver.StudentSpec("nested_group_dense_w72", "dense", 72, 4)
    dense_state = wd4.slice_dense_state(source, salience[:72])
    factor_spec = receiver.StudentSpec("pointwise_svd_w96_r32", "factorized", 96, 4, 32)
    flat_spec = receiver.StudentSpec("film_amortized_flat_w96", "flattened", 96, 4)
    rows = [
        (
            dense_spec,
            dense_state,
            "nested 8-channel GroupNorm-group subspace; changes renderer width/topology",
        ),
        (
            factor_spec,
            build_factorized_state(source, rank=32),
            "rank-32 SVD factors replace every full 96x96 pointwise operator",
        ),
        (
            flat_spec,
            build_flattened_state(source),
            "one counted trunk FiLM map replaces four block-local FiLM maps",
        ),
    ]
    if len({spec.form for spec, _, _ in rows}) != len(rows):
        raise RJ1Error("every rung must use a distinct representation form")
    if any(spec.depth != 4 for spec, _, _ in rows):
        raise RJ1Error("RJ1 may not buy bytes by dropping renderer depth")
    return rows


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RJ1Error(f"runtime patch point differs: {label}")
    return source.replace(old, new)


def patch_runtime(source: Path, destination: Path, archive_path: Path) -> dict[str, Any]:
    """Add the existing counted WD2S receiver branch to an exact DX2 runtime copy."""

    resumed = destination.exists()
    if not resumed:
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    receiver_path = destination / "cpr1/wd2_receiver.py"
    source_receiver = Path(receiver.__file__).resolve()
    if receiver_path.is_file():
        if receiver_path.read_bytes() != source_receiver.read_bytes():
            raise RJ1Error("retained runtime WD2 receiver differs")
    else:
        shutil.copy2(source_receiver, receiver_path)

    residual_path = destination / "runtime/residual_archive.py"
    residual = residual_path.read_text(encoding="utf-8")
    old_residual = 'tagged_semantic = semantic_body.startswith((b"SD1M", b"SM3R"))'
    new_residual = 'tagged_semantic = semantic_body.startswith((b"SD1M", b"SM3R", b"WD2S"))'
    if residual.count(new_residual) == 1 and old_residual not in residual:
        pass
    elif residual.count(old_residual) == 1 and new_residual not in residual:
        residual_path.write_text(
            _replace_once(residual, old_residual, new_residual, "RX1 semantic dispatch"),
            encoding="utf-8",
        )
    else:
        raise RJ1Error("retained runtime RX1 semantic dispatch differs")

    f26_path = destination / "runtime/f26_inflate.py"
    f26 = f26_path.read_text(encoding="utf-8")
    old_guard = (
        'if not parts.semantic_blob.startswith((WANS1_MAGIC, b"SD1M", b"SM3R")):\n'
        '        raise InflationError("F26 requires WANS1, SD1M, or SM3R semantic weights")'
    )
    new_guard = (
        'if not parts.semantic_blob.startswith((WANS1_MAGIC, b"SD1M", b"SM3R", b"WD2S")):\n'
        '        raise InflationError("F26 requires WANS1, SD1M, SM3R, or WD2S semantic weights")'
    )
    old_loader = """    semantic = renderer.SemanticTokenRenderer(96)
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
"""
    new_loader = """    if parts.semantic_blob.startswith(b"WD2S"):
        receiver_path = renderer_dir / "wd2_receiver.py"
        receiver_spec = importlib.util.spec_from_file_location("_f26_rj1_receiver", receiver_path)
        if receiver_spec is None or receiver_spec.loader is None:
            raise InflationError("could not load the counted RJ1 renderer receiver")
        rj1_receiver = importlib.util.module_from_spec(receiver_spec)
        sys.modules[receiver_spec.name] = rj1_receiver
        receiver_spec.loader.exec_module(rj1_receiver)
        semantic = rj1_receiver.unpack_student(parts.semantic_blob)
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
"""
    f26_changed = False
    if f26.count(new_guard) == 1 and old_guard not in f26:
        pass
    elif f26.count(old_guard) == 1 and new_guard not in f26:
        f26 = _replace_once(f26, old_guard, new_guard, "F26 semantic guard")
        f26_changed = True
    else:
        raise RJ1Error("retained runtime F26 semantic guard differs")
    if f26.count(new_loader) == 1 and old_loader not in f26:
        pass
    elif f26.count(old_loader) == 1 and new_loader not in f26:
        f26 = _replace_once(f26, old_loader, new_loader, "F26 renderer construction")
        f26_changed = True
    else:
        raise RJ1Error("retained runtime F26 renderer construction differs")
    if f26_changed:
        f26_path.write_text(f26, encoding="utf-8")

    destination_archive = destination / "archive.zip"
    if destination_archive.is_file():
        if destination_archive.read_bytes() != archive_path.read_bytes():
            raise RJ1Error("retained runtime archive differs")
    else:
        shutil.copy2(archive_path, destination_archive)
    public_path = destination / "inflate.py"
    original_public = public_path.read_text(encoding="utf-8")
    digest = sha256_file(destination_archive)
    size = destination_archive.stat().st_size
    public, sha_count = re.subn(
        r'^ARCHIVE_SHA256 = "[0-9a-f]{64}"$',
        f'ARCHIVE_SHA256 = "{digest}"',
        original_public,
        count=1,
        flags=re.MULTILINE,
    )
    public, size_count = re.subn(
        r"^ARCHIVE_BYTES = [0-9_]+$",
        f"ARCHIVE_BYTES = {size:_}",
        public,
        count=1,
        flags=re.MULTILINE,
    )
    if (sha_count, size_count) != (1, 1):
        raise RJ1Error("public archive binding points differ")
    if public != original_public:
        public_path.write_text(public, encoding="utf-8")
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "resumed_from_retained": resumed,
        "archive": file_record(destination_archive),
        "additive_magic": "WD2S",
        "runtime_f26": file_record(f26_path),
        "runtime_parser": file_record(residual_path),
        "receiver": file_record(receiver_path),
        "public_entrypoint": file_record(public_path),
    }


def parse_with_runtime(runtime: Path, archive: Path, expected_packet: bytes) -> dict[str, Any]:
    prior = _clear_runtime_modules()
    sys.path.insert(0, str(runtime.resolve()))
    try:
        module = importlib.import_module("runtime.residual_archive")
        parts = module.read_residual_archive(archive)
        if parts.semantic_blob != expected_packet:
            raise RJ1Error("patched receiver semantic parse-back differs")
    finally:
        sys.path.pop(0)
        for name in list(sys.modules):
            if name == "runtime" or name.startswith("runtime."):
                sys.modules.pop(name, None)
        sys.modules.update(prior)
    model = receiver.unpack_student(expected_packet)
    if receiver.pack_student(model) != expected_packet:
        raise RJ1Error("WD2S packet parse/repack differs")
    return {
        "semantic_packet_sha256": sha256_bytes(expected_packet),
        "semantic_packet_bytes": len(expected_packet),
        "parsed_spec": model.spec.as_dict(),
        "strict_state_load": True,
        "packet_repack_byte_identical": True,
    }


def retain_rung(
    output: Path,
    container: Mapping[str, Any],
    spec: receiver.StudentSpec,
    state: OrderedDict[str, torch.Tensor],
    representation: str,
) -> dict[str, Any]:
    root = output / "rungs" / spec.candidate_id
    float_initialization = atomic_torch(root / "renderer_initialization.pt", state)
    model = receiver.StudentSemanticRenderer(spec)
    model.load_state_dict(state, strict=True)
    packet = receiver.pack_student(model)
    semantic_packet = atomic_bytes(root / "semantic.wd2s", packet)
    parsed = receiver.unpack_student(packet)
    if (
        parsed.spec.form,
        parsed.spec.width,
        parsed.spec.depth,
        parsed.spec.rank,
    ) != (spec.form, spec.width, spec.depth, spec.rank) or receiver.pack_student(parsed) != packet:
        raise RJ1Error(f"packet closure failed: {spec.candidate_id}")
    semantic_stream, member, archive = build_archive(container, packet)
    repeat = deterministic_zip(member)
    if repeat != archive:
        raise RJ1Error(f"archive repeat differs: {spec.candidate_id}")

    payloads = {
        "float_initialization": float_initialization,
        "semantic_packet": semantic_packet,
        "semantic_stream": atomic_bytes(root / "semantic.ck2.brotli", semantic_stream),
        "member": atomic_bytes(root / "p", member),
        "archive": atomic_bytes(root / "archive.zip", archive),
        "archive_repeat": atomic_bytes(root / "archive.repeat.zip", repeat),
    }
    runtime = root / "runtime_sealed"
    runtime_receipt = patch_runtime(DX2_RUNTIME, runtime, root / "archive.zip")
    parseback = parse_with_runtime(runtime, runtime / "archive.zip", packet)
    saved = DX2_ARCHIVE_BYTES - len(archive)
    disposition = "BYTE-NEGATIVE-WITHHELD" if saved <= 0 else "PRECOMPENSATION-RUNG-WITHHELD"
    return {
        "candidate_id": spec.candidate_id,
        "disposition": disposition,
        "axis": AXIS,
        "representation": representation,
        "why_not_depth_coarsening": (
            "all learned tensors retain the WD2S int4/fp16 deployment law; the changed "
            "object is topology/operator placement, not a lower quantizer bit depth"
        ),
        "spec": spec.as_dict(),
        "source_dx2": {
            "archive_bytes": DX2_ARCHIVE_BYTES,
            "archive_sha256": DX2_ARCHIVE_SHA256,
        },
        "payloads": payloads,
        "runtime": runtime_receipt,
        "superseded_partial_runtime": (
            file_record(root / "runtime/runtime/residual_archive.py")
            if (root / "runtime/runtime/residual_archive.py").is_file()
            else None
        ),
        "parseback": parseback,
        "bytes_bought": saved,
        "rate_credit_s": saved * RATE_EXCHANGE_S_PER_BYTE,
        "fixed_distortion_demand": {
            "demand_bytes": FIXED_DISTORTION_DEMAND_BYTES,
            "covered_bytes": saved,
            "covered_fraction": saved / FIXED_DISTORTION_DEMAND_BYTES,
            "remaining_bytes": FIXED_DISTORTION_DEMAND_BYTES - saved,
        },
        "zero_distortion_demand": {
            "demand_bytes": ZERO_DISTORTION_DEMAND_BYTES,
            "covered_bytes": saved,
            "surplus_bytes": saved - ZERO_DISTORTION_DEMAND_BYTES,
        },
        "realized_d_seg": None,
        "realized_d_pose": None,
        "collateral_B": None,
        "collateral_H": None,
        "collateral_W": None,
        "joint_delta_s": None,
        "scorer_status": SCORER_STATUS,
        "compensation": {
            "status": "NOT_SOLVED",
            "required_binding": ("final moved renderer packet SHA + exact rendered odd-frame field + DX2 archive SHA"),
        },
        "carrier_resolve": {
            "status": "NOT_SOLVED",
            "must_follow": "final renderer optimization and exact-object compensation solve",
        },
        "candidate_admissible": False,
        "fire_gate": (
            "REFUSE until compensation.status=SOLVED_EXACT_OBJECT, carrier_resolve.status="
            "PARSEBACK_EXACT, primary/repeat archives match, and MAIN owns the scorer lane"
        ),
    }


def run(output: Path, minimum_free_bytes: int) -> dict[str, Any]:
    if output.resolve() != OUTPUT_ROOT.resolve():
        raise RJ1Error(f"output must be the first-priority SSD root: {OUTPUT_ROOT.resolve()}")
    completed = output / "RESULT.json"
    if completed.is_file():
        require_dx2()
        result = json.loads(completed.read_text(encoding="utf-8"))
        verify_retained_file_records(result)
        retained_minimum = int(result["storage_preflight"]["required_free_bytes"])
        write_or_verify_custody_inventory(output, retained_minimum)
        return result
    partial_canonical_attempt = output.exists() and any(output.iterdir()) and not (completed).is_file()
    superseded_local_result = SUPERSEDED_LOCAL_ROOT / "RESULT.json"
    known_local_incident = superseded_local_result.is_file()
    preflight = storage_preflight(output, minimum_free_bytes)
    source_archive = require_dx2()
    source = source_container()
    state = load_dx2_state()
    source_state = atomic_torch(output / "source/dx2_semantic_state_f32.pt", state)
    source_manifest = atomic_json(
        output / "source/SOURCE_CUSTODY.json",
        {
            "archive": source_archive,
            "semantic_state": source_state,
            "semantic_state_tensor_count": len(state),
            "semantic_state_decoded_sha256": DX2_SEMANTIC_SHA256,
            "source_runtime": str(DX2_RUNTIME.resolve()),
            "source_sections": {
                "hpac_stream_bytes": len(source["hpac"]),
                "semantic_stream_bytes": len(source["semantic"]),
                "carrier_stream_bytes": len(source["carrier"]),
                "tail_bytes": len(source["tail"]),
            },
        },
    )

    rows = [
        retain_rung(output, source, spec, rung_state, representation)
        for spec, rung_state, representation in representation_rungs(state)
    ]
    if any(row["candidate_admissible"] for row in rows):
        raise RJ1Error("a pre-compensation rung was incorrectly marked admissible")
    result = {
        "schema": "ddm_rj1_renderer_joint_move_precompile.v1",
        "status": "MECHANISM-INCOMPLETE-WITHHELD",
        "verdict_scope": "NO-VERDICT:DX2_PRECOMPENSATION_REPRESENTATION_RUNGS",
        "axis": AXIS,
        "score_claim": False,
        "pointer_moved": False,
        "storage_preflight": preflight,
        "source_custody": source_manifest,
        "exchange_rate_s_per_byte": RATE_EXCHANGE_S_PER_BYTE,
        "exchange_rate_source": ".omx/research/ddm_tx1_toolbox_crosswalk_20260819.md §0",
        "rungs": rows,
        "scorer_lane_touched": False,
        "heavy_launch_performed": False,
        "prior_incomplete_attempt_recovery": {
            "observed": partial_canonical_attempt or known_local_incident,
            "cause": (
                "the first run compared the non-serialized candidate_id and aborted after "
                "materializing the dense WD2S packet"
                if partial_canonical_attempt or known_local_incident
                else None
            ),
            "payload_rule_incident": partial_canonical_attempt or known_local_incident,
            "recovery": (
                "all three deterministic initializations and WD2S packets were rebuilt from "
                "the pinned DX2 semantic state and retained before closure validation on the "
                "first-priority SSD tier"
                if partial_canonical_attempt or known_local_incident
                else None
            ),
            "superseded_local_receipt": (file_record(superseded_local_result) if known_local_incident else None),
        },
        "next_gate": (
            "jointly optimize each moved renderer on DX2, solve compensation in-compile "
            "against that final object, re-solve/re-encode the carrier, then hand retained "
            "primary+repeat archives to MAIN for n600 B/H/W and Seg/Pose measurement"
        ),
    }
    atomic_json(output / "RESULT.json", result)
    write_or_verify_custody_inventory(output, minimum_free_bytes)
    return result


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    argument_parser.add_argument("--minimum-free-bytes", type=int, default=1 << 30)
    return argument_parser


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = run(args.output, args.minimum_free_bytes)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
