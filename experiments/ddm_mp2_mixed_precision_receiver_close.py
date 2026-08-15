#!/usr/bin/env python3
"""Receiver-close the retained MZ2 semantic candidates on the HV1 base.

This stage is scorer-free.  It rebuilds the complete archive for every MZ2
mixed-precision/FiLM-row candidate, stages one candidate-bound runtime tree per
archive, and proves the copied receiver reconstructs the packer's intended
semantic state exactly.  Every materialized archive and decoded payload is
retained below the APDataStore output root.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import zipfile
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

rx1 = importlib.import_module("experiments.ddm_rx1_rate_representation_attack")
mz2 = importlib.import_module("experiments.ddm_mz2_frozen_section_representation_attack")

BASE_GENERATION = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pq1_submission_packet/generations/"
    "hv1_ep0634_s1p25_c1p0_brotli_q10"
)
MZ2_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_mz2_frozen_section_representation")
MZ2_SCORE_RESULT = MZ2_ROOT / "SCORE_GATE_RESULT.json"
MZ2_RETENTION_INVENTORY = MZ2_ROOT / "RETENTION_INVENTORY.json"
RECEIVER_SOURCE = REPO / "experiments/ddm_mp2_semantic_receiver.py"
DEFAULT_OUTPUT = Path("/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815")

BASE_ARCHIVE_BYTES = 182_759
BASE_ARCHIVE_SHA256 = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
BASE_OUTER_INFLATE_SHA256 = "fdd2d19a4c7bb51918ebadcf56d85856ad1f7bcca1cf272de8e2f268610e60f1"
BASE_INNER_INFLATE_SHA256 = "808b4ffd7eb3cea34fdf5b55dd1919a546697feca1a54c2391c08d6c34f3bc2d"
MZ2_RETENTION_INVENTORY_SHA256 = "156112d0a0b8caeec0f0a6eaedd3bc1d24e2d389b199dad2495324ebd6c2dbcc"
BASE_SEMANTIC_STREAM_BYTES = 34_763
RATE_DENOMINATOR = 37_545_489
AXIS = "[macOS-CPU advisory; scorer-free receiver-close preflight]"
SEED = 20260815


class MP2BuildError(RuntimeError):
    """An MP2 provenance, archive, or receiver invariant failed closed."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise MP2BuildError(f"required file is absent: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_file(
    path: Path,
    *,
    size: int | None = None,
    digest: str | None = None,
) -> dict[str, Any]:
    record = file_record(path)
    if size is not None and record["bytes"] != size:
        raise MP2BuildError(f"file size differs: {path}")
    if digest is not None and record["sha256"] != digest:
        raise MP2BuildError(f"file SHA-256 differs: {path}")
    return record


def atomic_bytes(path: Path, value: bytes, *, executable: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        if executable:
            temporary.chmod(0o755)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_record(path)


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    return atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return atomic_bytes(path, buffer.getvalue())


def copy_runtime_file(source: str, destination: str) -> str:
    """Copy runtime bytes and executable mode without ExFAT xattr metadata."""

    shutil.copyfile(source, destination)
    os.chmod(destination, os.stat(source).st_mode & 0o777)
    return destination


def read_stored_member(archive_path: Path) -> bytes:
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != "p" or infos[0].compress_type != zipfile.ZIP_STORED:
            raise MP2BuildError("archive must contain exactly one stored member p")
        member = archive.read(infos[0])
        if archive.testzip() is not None:
            raise MP2BuildError("archive CRC validation failed")
    return member


def split_member(member: bytes) -> dict[str, bytes | int]:
    if len(member) < rx1.RX1_HEADER.size:
        raise MP2BuildError("RX1M member is truncated")
    magic, version, codec_id, table_mode, reserved, hpac_size, semantic_size, carrier_size = (
        rx1.RX1_HEADER.unpack_from(member)
    )
    if (magic, version, codec_id, table_mode, reserved) != (
        rx1.RX1_MAGIC,
        rx1.RX1_VERSION,
        rx1.RX1_CODEC_BROTLI,
        rx1.RX1_TABLE_ON,
        0,
    ):
        raise MP2BuildError("HV1 RX1M header differs")
    offset = rx1.RX1_HEADER.size
    model_size = offset + hpac_size + semantic_size + carrier_size
    if model_size >= len(member):
        raise MP2BuildError("RX1M model/tail boundary differs")
    hpac = member[offset : offset + hpac_size]
    semantic = member[offset + hpac_size : offset + hpac_size + semantic_size]
    carrier = member[offset + hpac_size + semantic_size : model_size]
    return {
        "hpac": hpac,
        "semantic": semantic,
        "carrier": carrier,
        "model": member[:model_size],
        "tail": member[model_size:],
        "hpac_size": hpac_size,
        "semantic_size": semantic_size,
        "carrier_size": carrier_size,
    }


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise MP2BuildError(f"runtime patch point differs: {label}")
    return source.replace(old, new, 1)


def patch_inner_runtime(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    import_line = "from ddm_mp2_semantic_receiver import unpack_variant_semantic_or_none\n"
    dispatch_line = "    tagged_state = unpack_variant_semantic_or_none(\n"
    if import_line in source or dispatch_line in source:
        if source.count(import_line) != 1 or source.count(dispatch_line) != 1:
            raise MP2BuildError("partially applied MP2 runtime patch")
        return
    source = _replace_once(
        source,
        "from integer_model_io import deserialize_integer_model\n",
        "from integer_model_io import deserialize_integer_model\n"
        + import_line,
        "MP2 receiver import",
    )
    old = '''    try:
        semantic_width = SEMANTIC_WIDTH_BY_PAYLOAD_BYTES[semantic_bytes]
    except KeyError as error:
        raise ValueError(
            f"unsupported semantic payload size: {semantic_bytes} bytes"
        ) from error
    semantic = SemanticTokenRenderer(semantic_width)
    semantic.load_state_dict(unpack_semantic(semantic_blob, semantic.state_dict()))
'''
    new = '''    tagged_state = None
    if semantic_blob.startswith((b"SD1M", b"SM3R")):
        semantic_width = SEMANTIC_WIDTH
    else:
        try:
            semantic_width = SEMANTIC_WIDTH_BY_PAYLOAD_BYTES[semantic_bytes]
        except KeyError as error:
            raise ValueError(
                f"unsupported semantic payload size: {semantic_bytes} bytes"
            ) from error
    semantic = SemanticTokenRenderer(semantic_width)
    tagged_state = unpack_variant_semantic_or_none(
        semantic_blob,
        semantic.state_dict(),
    )
    if tagged_state is None:
        tagged_state = unpack_semantic(semantic_blob, semantic.state_dict())
    semantic.load_state_dict(tagged_state, strict=True)
'''
    source = _replace_once(source, old, new, "semantic tagged dispatch")
    path.write_text(source, encoding="utf-8")


def bind_outer_runtime(path: Path, archive: Path) -> None:
    source = path.read_text(encoding="utf-8")
    source, sha_count = re.subn(
        r'^ARCHIVE_SHA256 = "[0-9a-f]{64}"$',
        f'ARCHIVE_SHA256 = "{sha256_file(archive)}"',
        source,
        count=1,
        flags=re.MULTILINE,
    )
    source, size_count = re.subn(
        r"^ARCHIVE_BYTES = [0-9_]+$",
        f"ARCHIVE_BYTES = {archive.stat().st_size:_}",
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if (sha_count, size_count) != (1, 1):
        raise MP2BuildError("outer candidate archive binding points differ")
    path.write_text(source, encoding="utf-8")


def load_runtime_module(path: Path, name: str):
    runtime_dir = str(path.parent)
    if runtime_dir not in sys.path:
        sys.path.insert(0, runtime_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise MP2BuildError(f"could not load candidate runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def expected_state(
    parser: str,
    semantic_raw: bytes,
    template: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    if parser == "sd1m":
        restored, _, format_name = mz2.sd1.unpack_semantic_state(semantic_raw, template)
        if format_name != "sd1_mixed_v1":
            raise MP2BuildError("MZ2 mixed candidate no longer parses as SD1M v1")
        return OrderedDict(restored)
    if parser == "sm3r":
        return OrderedDict(mz2.sm3.unpack_prune_candidate(semantic_raw, template))
    if parser == "legacy":
        return OrderedDict(template)
    raise MP2BuildError(f"unknown candidate parser: {parser}")


def retain_receiver_decode(
    generation: Path,
    *,
    candidate_id: str,
    parser: str,
    semantic_raw: bytes,
    carrier_raw: bytes,
    template: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    runtime = load_runtime_module(
        generation / "cpr1/inflate.py",
        f"ddm_mp2_runtime_{candidate_id}_{os.getpid()}",
    )
    semantic_pose = (
        len(semantic_raw).to_bytes(4, "little")
        + len(carrier_raw).to_bytes(4, "little")
        + semantic_raw
        + carrier_raw
    )
    semantic, basis, coefficients = runtime.unpack_semantic_pose(semantic_pose)
    actual = OrderedDict(semantic.state_dict())
    expected = expected_state(parser, semantic_raw, template)
    if tuple(actual) != tuple(expected):
        raise MP2BuildError(f"receiver schema differs for {candidate_id}")
    for name in expected:
        if not torch.equal(actual[name], expected[name]):
            delta = float(torch.max(torch.abs(actual[name] - expected[name])).item())
            raise MP2BuildError(
                f"receiver state differs for {candidate_id}:{name}; max_abs={delta}"
            )

    decoded_root = generation / "retained/receiver_decode"
    state_records = []
    for index, (name, value) in enumerate(actual.items()):
        safe_name = name.replace(".", "_")
        record = atomic_npy(
            decoded_root / "semantic_state" / f"{index:02d}_{safe_name}.npy",
            value.detach().cpu().numpy(),
        )
        state_records.append({"name": name, **record})
    basis_record = atomic_npy(decoded_root / "carrier_basis.npy", basis.detach().cpu().numpy())
    coefficient_record = atomic_npy(
        decoded_root / "carrier_coefficients.npy",
        coefficients.detach().cpu().numpy(),
    )
    result = {
        "schema": "ddm_mp2_receiver_parseback.v1",
        "candidate_id": candidate_id,
        "parser": parser,
        "semantic_tensor_denominator": len(actual),
        "semantic_state_exact_to_packer": True,
        "carrier_shape": list(basis.shape),
        "coefficient_shape": list(coefficients.shape),
        "state_payloads": state_records,
        "carrier_basis": basis_record,
        "carrier_coefficients": coefficient_record,
        "complete": True,
    }
    atomic_json(generation / "RECEIVER_PARSEBACK.json", result)
    return result


def candidate_rows(score_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    mixed = dict(score_result["candidate"])
    structured = [dict(row) for row in score_result["structured_sparsity_candidates"]]
    rows = [mixed, *sorted(structured, key=lambda row: -int(row["keep_percent"]))]
    identifiers = [str(row["candidate_id"]) for row in rows]
    if identifiers != [
        "score_gated_selected_mixed_q3q4",
        "score_gated_film_row_prune_keep87",
        "score_gated_film_row_prune_keep75",
        "score_gated_film_row_prune_keep62",
        "score_gated_film_row_prune_keep50",
        "score_gated_film_row_prune_keep37",
        "score_gated_film_row_prune_keep25",
    ]:
        raise MP2BuildError("MZ2 score-gated candidate set differs")
    return rows


def verify_generation_receipt(destination: Path, receipt: Mapping[str, Any]) -> None:
    if receipt.get("complete") is not True or receipt.get("receiver_closed") is not True:
        raise MP2BuildError(f"generation receipt is incomplete: {destination}")
    def verify_record(record: Mapping[str, Any]) -> None:
        path = Path(str(record["path"]))
        if destination not in path.parents and path != destination:
            raise MP2BuildError(f"generation receipt path escapes candidate root: {path}")
        require_file(path, size=int(record["bytes"]), digest=str(record["sha256"]))

    records = [receipt["archive"]]
    records.extend(receipt["candidate_bound_runtime"].values())
    records.extend(receipt["retained_payloads"].values())
    parseback_record = receipt["receiver_parseback"]["receipt"]
    records.append(parseback_record)
    for record in records:
        if isinstance(record, Mapping) and {"path", "bytes", "sha256"}.issubset(record):
            verify_record(record)
    parseback = json.loads(Path(str(parseback_record["path"])).read_text(encoding="utf-8"))
    if parseback.get("complete") is not True or parseback.get("semantic_state_exact_to_packer") is not True:
        raise MP2BuildError(f"receiver parse-back receipt is incomplete: {destination}")
    decoded_records = [
        *parseback["state_payloads"],
        parseback["carrier_basis"],
        parseback["carrier_coefficients"],
    ]
    for record in decoded_records:
        verify_record(record)


def preflight(output: Path) -> dict[str, Any]:
    if not str(output.resolve()).startswith("/Volumes/APDataStore/pact/"):
        raise MP2BuildError("MP2 retained output must be on APDataStore")
    output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free
    required = 64 << 30
    if free < required:
        raise MP2BuildError(f"MP2 requires {required} free bytes; observed {free}")
    base_archive = require_file(
        BASE_GENERATION / "archive.zip",
        size=BASE_ARCHIVE_BYTES,
        digest=BASE_ARCHIVE_SHA256,
    )
    inputs = {
        "base_archive": base_archive,
        "base_outer_inflate": require_file(
            BASE_GENERATION / "inflate.py",
            digest=BASE_OUTER_INFLATE_SHA256,
        ),
        "base_inner_inflate": require_file(
            BASE_GENERATION / "cpr1/inflate.py",
            digest=BASE_INNER_INFLATE_SHA256,
        ),
        "mz2_score_result": require_file(MZ2_SCORE_RESULT),
        "mz2_retention_inventory": require_file(
            MZ2_RETENTION_INVENTORY,
            digest=MZ2_RETENTION_INVENTORY_SHA256,
        ),
        "receiver_source": require_file(RECEIVER_SOURCE),
    }
    score_result = json.loads(MZ2_SCORE_RESULT.read_text(encoding="utf-8"))
    rows = candidate_rows(score_result)
    source_payloads = []
    for row in rows:
        for label in ("archive", "model", "semantic_brotli_q11", "semantic_raw"):
            declared = row["payloads"][label]
            source_payloads.append(
                {
                    "candidate_id": row["candidate_id"],
                    "label": label,
                    **require_file(
                        Path(declared["path"]),
                        size=int(declared["bytes"]),
                        digest=str(declared["sha256"]),
                    ),
                }
            )
    result = {
        "schema": "ddm_mp2_preflight.v1",
        "axis": AXIS,
        "score_claim": False,
        "seed": SEED,
        "storage": {
            "output": str(output.resolve()),
            "free_bytes": free,
            "required_bytes": required,
        },
        "inputs": inputs,
        "candidate_denominator": len(rows),
        "candidate_source_payloads": source_payloads,
        "resumable_from_disk": True,
        "stage_checkpoints": ["PREFLIGHT.json", "BUILD_RESULT.json", "FINAL_RESULT.json"],
        "complete": True,
    }
    atomic_json(output / "PREFLIGHT.json", result)
    return result


def persist_decomposition(
    generation: Path,
    *,
    member: bytes,
    model: bytes,
    hpac_stream: bytes,
    semantic_stream: bytes,
    semantic_raw: bytes,
    carrier_stream: bytes,
    carrier_raw: bytes,
    tail: bytes,
    archive_repeat: bytes,
) -> dict[str, Any]:
    root = generation / "retained"
    return {
        "member": atomic_bytes(root / "p", member),
        "model": atomic_bytes(root / "models.rx1m", model),
        "hpac_stream": atomic_bytes(root / "hpac.br", hpac_stream),
        "semantic_stream": atomic_bytes(root / "semantic.br", semantic_stream),
        "semantic_raw": atomic_bytes(root / "semantic.raw.bin", semantic_raw),
        "carrier_stream": atomic_bytes(root / "carrier.br", carrier_stream),
        "carrier_raw": atomic_bytes(root / "carrier.raw.bin", carrier_raw),
        "tail": atomic_bytes(root / "member_tail.bin", tail),
        "archive_repeat": atomic_bytes(root / "archive.repeat.zip", archive_repeat),
    }


def build_generation(
    output: Path,
    *,
    candidate_id: str,
    parser: str,
    semantic_stream: bytes,
    semantic_raw: bytes,
    expected_delta_bytes: int,
    base_parts: Mapping[str, bytes | int],
    template: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    generations = output / "generations"
    destination = generations / candidate_id
    receipt_path = destination / "GENERATION_RECEIPT.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        verify_generation_receipt(destination, receipt)
        return receipt

    generations.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        BASE_GENERATION,
        destination,
        symlinks=True,
        dirs_exist_ok=True,
        copy_function=copy_runtime_file,
        ignore=shutil.ignore_patterns("__pycache__", "._*"),
    )

    hpac_stream = bytes(base_parts["hpac"])
    carrier_stream = bytes(base_parts["carrier"])
    tail = bytes(base_parts["tail"])
    model = rx1.pack_rx1_model(
        hpac_stream,
        semantic_stream,
        carrier_stream,
        codec_id=rx1.RX1_CODEC_BROTLI,
        table_mode=rx1.RX1_TABLE_ON,
    )
    member = model + tail
    archive = rx1.deterministic_zip(member)
    archive_repeat = rx1.deterministic_zip(member)
    if archive != archive_repeat:
        raise MP2BuildError(f"candidate archive is nondeterministic: {candidate_id}")
    expected_bytes = BASE_ARCHIVE_BYTES + expected_delta_bytes
    if len(archive) != expected_bytes:
        raise MP2BuildError(
            f"candidate archive delta differs for {candidate_id}: {len(archive)} != {expected_bytes}"
        )

    archive_record = atomic_bytes(destination / "archive.zip", archive)
    receiver_copy = destination / "cpr1/ddm_mp2_semantic_receiver.py"
    shutil.copy2(RECEIVER_SOURCE, receiver_copy)
    patch_inner_runtime(destination / "cpr1/inflate.py")
    bind_outer_runtime(destination / "inflate.py", destination / "archive.zip")
    if sha256_file(destination / "archive.zip") != archive_record["sha256"]:
        raise MP2BuildError("candidate archive changed during runtime binding")

    carrier_raw = brotli.decompress(carrier_stream)
    retained = persist_decomposition(
        destination,
        member=member,
        model=model,
        hpac_stream=hpac_stream,
        semantic_stream=semantic_stream,
        semantic_raw=semantic_raw,
        carrier_stream=carrier_stream,
        carrier_raw=carrier_raw,
        tail=tail,
        archive_repeat=archive_repeat,
    )
    parseback = retain_receiver_decode(
        destination,
        candidate_id=candidate_id,
        parser=parser,
        semantic_raw=semantic_raw,
        carrier_raw=carrier_raw,
        template=template,
    )
    receipt = {
        "schema": "ddm_mp2_generation_receipt.v1",
        "candidate_id": candidate_id,
        "parser": parser,
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "archive": archive_record,
        "archive_delta_bytes_vs_hv1": len(archive) - BASE_ARCHIVE_BYTES,
        "projected_rate_only_delta_score": (
            (len(archive) - BASE_ARCHIVE_BYTES) * 25 / RATE_DENOMINATOR
        ),
        "candidate_bound_runtime": {
            "outer_inflate": file_record(destination / "inflate.py"),
            "inner_inflate": file_record(destination / "cpr1/inflate.py"),
            "semantic_receiver": file_record(receiver_copy),
            "archive_sha_pin_updated": True,
            "archive_bytes_pin_updated": True,
        },
        "retained_payloads": retained,
        "receiver_parseback": {
            "receipt": file_record(destination / "RECEIVER_PARSEBACK.json"),
            "semantic_state_exact_to_packer": parseback["semantic_state_exact_to_packer"],
            "semantic_tensor_denominator": parseback["semantic_tensor_denominator"],
        },
        "archive_repeat_byte_identical": True,
        "receiver_closed": True,
        "n600_status": "PENDING",
        "complete": True,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def build_control(
    output: Path,
    *,
    base_parts: Mapping[str, bytes | int],
    template: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    semantic_stream = bytes(base_parts["semantic"])
    semantic_raw = brotli.decompress(semantic_stream)
    return build_generation(
        output,
        candidate_id="hv1_base_control",
        parser="legacy",
        semantic_stream=semantic_stream,
        semantic_raw=semantic_raw,
        expected_delta_bytes=0,
        base_parts=base_parts,
        template=template,
    )


def build_all(output: Path) -> dict[str, Any]:
    preflight_receipt = output / "PREFLIGHT.json"
    if not preflight_receipt.is_file():
        raise MP2BuildError("preflight receipt is required before build")
    base_member = read_stored_member(BASE_GENERATION / "archive.zip")
    base_parts = split_member(base_member)
    if int(base_parts["semantic_size"]) != BASE_SEMANTIC_STREAM_BYTES:
        raise MP2BuildError("HV1 semantic stream size differs")
    records, _, _ = mz2._load_records()
    template = OrderedDict(
        (
            record.schema.name,
            torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32)),
        )
        for record in records
    )
    results = [
        build_control(
            output,
            base_parts=base_parts,
            template=template,
        )
    ]
    score_result = json.loads(MZ2_SCORE_RESULT.read_text(encoding="utf-8"))
    for row in candidate_rows(score_result):
        stream_declared = row["payloads"]["semantic_brotli_q11"]
        raw_declared = row["payloads"]["semantic_raw"]
        stream_path = Path(stream_declared["path"])
        raw_path = Path(raw_declared["path"])
        require_file(
            stream_path,
            size=int(stream_declared["bytes"]),
            digest=str(stream_declared["sha256"]),
        )
        require_file(
            raw_path,
            size=int(raw_declared["bytes"]),
            digest=str(raw_declared["sha256"]),
        )
        semantic_stream = stream_path.read_bytes()
        semantic_raw = raw_path.read_bytes()
        if brotli.decompress(semantic_stream) != semantic_raw:
            raise MP2BuildError(f"MZ2 semantic stream parse-back differs: {row['candidate_id']}")
        results.append(
            build_generation(
                output,
                candidate_id=str(row["candidate_id"]),
                parser=str(row["parser"]),
                semantic_stream=semantic_stream,
                semantic_raw=semantic_raw,
                expected_delta_bytes=int(row["delta_archive_bytes_vs_current"]),
                base_parts=base_parts,
                template=template,
            )
        )
    result = {
        "schema": "ddm_mp2_build_result.v1",
        "axis": AXIS,
        "score_claim": False,
        "generation_denominator": len(results),
        "candidate_denominator": len(results) - 1,
        "generations": results,
        "all_archives_retained": True,
        "all_receiver_decodes_retained": True,
        "all_receivers_closed": all(row["receiver_closed"] for row in results),
        "complete": True,
    }
    atomic_json(output / "BUILD_RESULT.json", result)
    return result


def finalize(output: Path) -> dict[str, Any]:
    build_path = output / "BUILD_RESULT.json"
    if not build_path.is_file():
        raise MP2BuildError("build receipt is required before finalize")
    build = json.loads(build_path.read_text(encoding="utf-8"))
    if build.get("complete") is not True or build.get("all_receivers_closed") is not True:
        raise MP2BuildError("receiver-close build is incomplete")
    eval_order = [row["candidate_id"] for row in build["generations"]]
    eval_queue = {
        "schema": "ddm_mp2_advisory_eval_queue.v1",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "ddm_mp2",
        "consumer_store": str((output / "ADVISORY_N600_RESULTS.json").resolve()),
        "fire_trigger": (
            "No other full-n600 scorer process is active; clean AppleDouble files from every "
            "candidate tree and the pinned mirror, then launch the listed generations serially "
            "through experiments/contest_auth_eval.py with retained work directories."
        ),
        "order": eval_order,
        "current_fire": False,
    }
    atomic_json(output / "EVAL_QUEUE.json", eval_queue)
    final = {
        "schema": "ddm_mp2_receiver_close_final.v1",
        "axis": AXIS,
        "score_claim": False,
        "base_archive": file_record(BASE_GENERATION / "archive.zip"),
        "mz2_retention_inventory": file_record(MZ2_RETENTION_INVENTORY),
        "build_result": file_record(build_path),
        "eval_queue": file_record(output / "EVAL_QUEUE.json"),
        "candidate_rows": [
            {
                "candidate_id": row["candidate_id"],
                "archive": row["archive"],
                "archive_delta_bytes_vs_hv1": row["archive_delta_bytes_vs_hv1"],
                "receiver_closed": row["receiver_closed"],
                "n600_status": row["n600_status"],
            }
            for row in build["generations"]
        ],
        "complete": True,
    }
    atomic_json(output / "FINAL_RESULT.json", final)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("preflight", "build", "finalize", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.resolve() != args.resume_from.resolve():
        raise MP2BuildError("--resume-from must equal --output")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    stages = (
        ("preflight", preflight),
        ("build", build_all),
        ("finalize", finalize),
    )
    for name, function in stages:
        if args.stage not in {name, "all"}:
            continue
        result = function(args.output)
        print(json.dumps({"stage": name, "complete": result["complete"]}, sort_keys=True))


if __name__ == "__main__":
    main()
