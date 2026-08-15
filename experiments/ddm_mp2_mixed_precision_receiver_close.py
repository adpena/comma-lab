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
DIFFERENTIAL_CANDIDATE_ID = "score_gated_film_row_prune_keep75_minus_keep87"

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


def patch_residual_runtime(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    marker = "    tagged_semantic = semantic_body.startswith((b\"SD1M\", b\"SM3R\"))\n"
    if marker in source:
        if source.count(marker) != 1:
            raise MP2BuildError("partially applied MP2 residual-runtime patch")
        return
    old = '''    if len(semantic_body) != WANS_BODY_BYTES:
        raise ResidualArchiveError("RX1 semantic section length differs")
    try:
        semantic = decode_f12_wans_body(semantic_body, WANS_STREAM_ORDER)
        decode_wans1(semantic)
    except RendererWeightCodecError as error:
        raise ResidualArchiveError("invalid RX1 renderer weights") from error
'''
    new = '''    tagged_semantic = semantic_body.startswith((b"SD1M", b"SM3R"))
    if tagged_semantic:
        semantic = semantic_body
    else:
        if len(semantic_body) != WANS_BODY_BYTES:
            raise ResidualArchiveError("RX1 semantic section length differs")
        try:
            semantic = decode_f12_wans_body(semantic_body, WANS_STREAM_ORDER)
            decode_wans1(semantic)
        except RendererWeightCodecError as error:
            raise ResidualArchiveError("invalid RX1 renderer weights") from error
'''
    path.write_text(
        _replace_once(source, old, new, "RX1 tagged semantic decode"),
        encoding="utf-8",
    )


def patch_f26_runtime(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    marker = "    tagged_state = renderer.unpack_variant_semantic_or_none(\n"
    if marker in source:
        if source.count(marker) != 1:
            raise MP2BuildError("partially applied MP2 F26-runtime patch")
        return
    source = _replace_once(
        source,
        '''    if not parts.semantic_blob.startswith(WANS1_MAGIC):
        raise InflationError("F26 requires WANS1 semantic weights")
''',
        '''    if not parts.semantic_blob.startswith((WANS1_MAGIC, b"SD1M", b"SM3R")):
        raise InflationError("F26 requires WANS1, SD1M, or SM3R semantic weights")
''',
        "F26 semantic guard",
    )
    old = '''    semantic = renderer.SemanticTokenRenderer(96)
    records = decode_wans1(parts.semantic_blob)
    state = {
        record.schema.name: torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32))
        for record in records
    }
    semantic.load_state_dict(state, strict=True)
'''
    new = '''    semantic = renderer.SemanticTokenRenderer(96)
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
    path.write_text(
        _replace_once(source, old, new, "F26 semantic construction"),
        encoding="utf-8",
    )


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


def load_runtime_package_module(generation: Path, candidate_id: str, module_name: str):
    safe_id = re.sub(r"[^A-Za-z0-9_]", "_", candidate_id)
    package_name = f"_ddm_mp2_{safe_id}_runtime_{os.getpid()}"
    package_path = generation / "runtime"
    package_spec = importlib.util.spec_from_file_location(
        package_name,
        package_path / "__init__.py",
        submodule_search_locations=[str(package_path)],
    )
    if package_spec is None or package_spec.loader is None:
        raise MP2BuildError(f"could not load candidate runtime package: {generation}")
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[package_name] = package
    package_spec.loader.exec_module(package)
    return importlib.import_module(f"{package_name}.{module_name}")


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
    template: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    residual_runtime = load_runtime_package_module(
        generation,
        candidate_id,
        "residual_archive",
    )
    runtime = load_runtime_module(
        generation / "cpr1/inflate.py",
        f"ddm_mp2_runtime_{candidate_id}_{os.getpid()}",
    )
    parts = residual_runtime.read_residual_archive(generation / "archive.zip")
    if parser == "legacy":
        records = residual_runtime.decode_wans1(parts.semantic_blob)
        actual = OrderedDict(
            (
                record.schema.name,
                torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32)),
            )
            for record in records
        )
        expected = OrderedDict(template)
    else:
        decoded = runtime.unpack_variant_semantic_or_none(
            parts.semantic_blob,
            runtime.SemanticTokenRenderer(96).state_dict(),
        )
        if decoded is None:
            raise MP2BuildError(f"tagged candidate reached the legacy receiver: {candidate_id}")
        actual = OrderedDict(decoded)
        expected = expected_state(parser, parts.semantic_blob, template)
    if tuple(actual) != tuple(expected):
        raise MP2BuildError(f"receiver schema differs for {candidate_id}")
    for name in expected:
        if not torch.equal(actual[name], expected[name]):
            delta = float(torch.max(torch.abs(actual[name] - expected[name])).item())
            raise MP2BuildError(
                f"receiver state differs for {candidate_id}:{name}; max_abs={delta}"
            )

    decoded_root = generation / "retained/receiver_decode"
    outer_payloads = {
        "semantic_blob": atomic_bytes(decoded_root / "outer_semantic.bin", parts.semantic_blob),
        "carrier_blob": atomic_bytes(decoded_root / "outer_carrier.bin", parts.carrier_blob),
        "hpac_blob": atomic_bytes(decoded_root / "outer_hpac.bin", parts.hpac_blob),
        "token_stream": atomic_bytes(decoded_root / "outer_tokens.rc64", parts.token_stream),
        "residual_payload": atomic_bytes(
            decoded_root / "outer_residual.bin",
            parts.residual_payload,
        ),
        "compressed_models": atomic_bytes(
            decoded_root / "outer_compressed_models.bin",
            parts.compressed_models,
        ),
    }
    if parts.compensation_blob is not None:
        outer_payloads["compensation_blob"] = atomic_bytes(
            decoded_root / "outer_compensation.bin",
            parts.compensation_blob,
        )
    table_codes = atomic_npy(decoded_root / "residual_table_codes.npy", parts.table.codes)
    table_values = atomic_npy(decoded_root / "residual_table_values.npy", parts.table.values)
    state_records = []
    for index, (name, value) in enumerate(actual.items()):
        safe_name = name.replace(".", "_")
        record = atomic_npy(
            decoded_root / "semantic_state" / f"{index:02d}_{safe_name}.npy",
            value.detach().cpu().numpy(),
        )
        state_records.append({"name": name, **record})
    result = {
        "schema": "ddm_mp2_receiver_parseback.v1",
        "candidate_id": candidate_id,
        "parser": parser,
        "semantic_tensor_denominator": len(actual),
        "semantic_state_exact_to_packer": True,
        "exact_archive_outer_parse": True,
        "outer_payloads": outer_payloads,
        "residual_table_codes": table_codes,
        "residual_table_values": table_values,
        "state_payloads": state_records,
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


def pack_differential_prune_candidate(
    state: Mapping[str, torch.Tensor],
) -> tuple[bytes, OrderedDict[str, torch.Tensor], dict[str, Any]]:
    """Prune only the rows removed by keep75 after keep87 already retained them.

    The SM3R receiver permits any row mask with the count declared by its
    keep-percent header.  For the current 192-row FiLM tensors, the
    ``keep87 ∖ keep75`` marginal contains 23 rows, so retaining every other
    row produces 169 rows, the count declared exactly by a keep88 header.  The
    exact selected indices are returned for retained per-row attribution.
    """

    header_keep_percent = 88
    qnames = mz2.sd1.quantized_names(state)
    prune_names = set(mz2.sm3.PRUNE_NAMES)
    tensor_mask = mz2.sm3.mask_for_names(qnames, prune_names)
    payload = bytearray(
        mz2.sm3.representation_header(
            mz2.sm3.MODE_ROW_PRUNE,
            header_keep_percent,
        )
    )
    payload.extend(int(tensor_mask).to_bytes(2, byteorder="little", signed=False))
    expected: OrderedDict[str, torch.Tensor] = OrderedDict()
    tensor_rows: dict[str, Any] = {}

    for name, value in state.items():
        if value.ndim < 2:
            stored = value.detach().cpu().to(torch.float16)
            payload.extend(stored.numpy().astype("<f2", copy=False).tobytes())
            expected[name] = stored.float()
            continue
        if name not in prune_names:
            encoded, restored = mz2.sm3.standard_q4_payload(name, value)
            payload.extend(encoded)
            expected[name] = restored
            continue

        rows = int(value.shape[0])
        keep87 = max(1, round(rows * 87 / 100.0))
        keep75 = max(1, round(rows * 75 / 100.0))
        flat = value.detach().cpu().float().reshape(rows, -1)
        norms = flat.square().sum(dim=1)
        order = sorted(range(rows), key=lambda index: (-float(norms[index]), index))
        reference_keep87 = frozenset(order[:keep87])
        reference_keep75 = frozenset(order[:keep75])
        marginal = reference_keep87 - reference_keep75
        retained = frozenset(range(rows)) - marginal
        if not reference_keep75.issubset(reference_keep87):
            raise MP2BuildError(f"non-nested prune reference for {name}")
        header_keep = max(1, round(rows * header_keep_percent / 100.0))
        if len(retained) != header_keep:
            raise MP2BuildError(
                f"differential row count cannot use the keep88 receiver header for {name}"
            )

        selected = np.zeros(rows, dtype=np.uint8)
        selected[list(retained)] = 1
        payload.extend(np.packbits(selected, bitorder="little").tobytes())
        compact = flat[torch.from_numpy(selected.astype(bool))]
        encoded, restored_compact = mz2.sm3.standard_q4_payload(
            "pruned.rows",
            compact,
        )
        payload.extend(encoded)
        restored = torch.zeros_like(flat)
        restored[torch.from_numpy(selected.astype(bool))] = restored_compact
        expected[name] = restored.reshape(value.shape)
        tensor_rows[name] = {
            "row_denominator": rows,
            "reference_keep87_rows": sorted(reference_keep87),
            "reference_keep75_rows": sorted(reference_keep75),
            "marginal_pruned_rows": sorted(marginal),
            "differential_retained_rows": sorted(retained),
            "marginal_pruned_count": len(marginal),
            "differential_retained_count": len(retained),
            "header_declared_retained_count": header_keep,
        }

    metadata = {
        "schema": "ddm_mp2_film_row_differential_selection.v1",
        "candidate_id": DIFFERENTIAL_CANDIDATE_ID,
        "seed": SEED,
        "relationship": "prune only rows selected by keep87 and omitted by keep75",
        "header_keep_percent": header_keep_percent,
        "selected_tensor_names": sorted(prune_names),
        "selection_mask": tensor_mask,
        "tensor_rows": tensor_rows,
        "complete": True,
    }
    return bytes(payload), expected, metadata


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
    decoded_records = [*parseback["state_payloads"], *parseback["outer_payloads"].values()]
    decoded_records.extend(
        (parseback["residual_table_codes"], parseback["residual_table_values"])
    )
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
    expected_delta_bytes: int | None,
    base_parts: Mapping[str, bytes | int],
    template: Mapping[str, torch.Tensor],
    selection_metadata: Mapping[str, Any] | None = None,
    additional_retained_payloads: Mapping[str, Mapping[str, Any]] | None = None,
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
    expected_bytes = (
        None if expected_delta_bytes is None else BASE_ARCHIVE_BYTES + expected_delta_bytes
    )
    if expected_bytes is not None and len(archive) != expected_bytes:
        raise MP2BuildError(
            f"candidate archive delta differs for {candidate_id}: {len(archive)} != {expected_bytes}"
        )

    archive_record = atomic_bytes(destination / "archive.zip", archive)
    receiver_copy = destination / "cpr1/ddm_mp2_semantic_receiver.py"
    shutil.copy2(RECEIVER_SOURCE, receiver_copy)
    patch_inner_runtime(destination / "cpr1/inflate.py")
    patch_residual_runtime(destination / "runtime/residual_archive.py")
    patch_f26_runtime(destination / "runtime/f26_inflate.py")
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
    for label, record in (additional_retained_payloads or {}).items():
        retained_path = Path(str(record["path"]))
        if destination not in retained_path.parents:
            raise MP2BuildError(f"additional retained payload escapes candidate root: {label}")
        require_file(
            retained_path,
            size=int(record["bytes"]),
            digest=str(record["sha256"]),
        )
        retained[label] = dict(record)
    if selection_metadata is not None:
        selection_path = destination / "retained/selection_map.json"
        if "selection_map" in retained:
            observed_selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if observed_selection != dict(selection_metadata):
                raise MP2BuildError("pre-retained differential selection map differs")
        else:
            retained["selection_map"] = atomic_json(
                selection_path,
                selection_metadata,
            )
    parseback = retain_receiver_decode(
        destination,
        candidate_id=candidate_id,
        parser=parser,
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
            "residual_archive": file_record(destination / "runtime/residual_archive.py"),
            "f26_inflate": file_record(destination / "runtime/f26_inflate.py"),
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
        "selection_attribution": (
            {
                "selection_map": retained["selection_map"],
                "schema": selection_metadata["schema"],
            }
            if selection_metadata is not None
            else None
        ),
        "archive_repeat_byte_identical": True,
        "receiver_closed": True,
        "n600_status": "PENDING",
        "complete": True,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def build_differential(output: Path) -> dict[str, Any]:
    """Build the Relay-7 keep75-minus-keep87 differential generation."""

    preflight_path = output / "PREFLIGHT.json"
    build_path = output / "BUILD_RESULT.json"
    if not preflight_path.is_file() or not build_path.is_file():
        raise MP2BuildError("original MP2 preflight and build receipts are required")
    original_build = json.loads(build_path.read_text(encoding="utf-8"))
    if (
        original_build.get("complete") is not True
        or original_build.get("all_receivers_closed") is not True
    ):
        raise MP2BuildError("original MP2 receiver-close build is incomplete")

    base_member = read_stored_member(BASE_GENERATION / "archive.zip")
    base_parts = split_member(base_member)
    records, _, _ = mz2._load_records()
    template = OrderedDict(
        (
            record.schema.name,
            torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32)),
        )
        for record in records
    )
    semantic_raw, expected, selection_map = pack_differential_prune_candidate(template)
    destination = output / "generations" / DIFFERENTIAL_CANDIDATE_ID
    prebuilt_raw = atomic_bytes(
        destination / "retained/differential_source_semantic.raw.bin",
        semantic_raw,
    )
    independent = OrderedDict(mz2.sm3.unpack_prune_candidate(semantic_raw, template))
    if tuple(independent) != tuple(expected) or any(
        not torch.equal(independent[name], expected[name]) for name in expected
    ):
        raise MP2BuildError("differential SM3R independent parse-back differs")
    semantic_stream = brotli.compress(semantic_raw, quality=11)
    prebuilt_stream = atomic_bytes(
        destination / "retained/differential_source_semantic.br",
        semantic_stream,
    )
    if brotli.decompress(semantic_stream) != semantic_raw:
        raise MP2BuildError("differential semantic Brotli parse-back differs")

    score_result = json.loads(MZ2_SCORE_RESULT.read_text(encoding="utf-8"))
    reference_records: dict[str, Any] = {}
    by_keep = {
        int(row["keep_percent"]): row
        for row in score_result["structured_sparsity_candidates"]
    }
    for keep_percent in (75, 87):
        declared = by_keep[keep_percent]["payloads"]["semantic_raw"]
        reference_path = Path(str(declared["path"]))
        reference_records[f"keep{keep_percent}"] = require_file(
            reference_path,
            size=int(declared["bytes"]),
            digest=str(declared["sha256"]),
        )
        repacked, _, _ = mz2.sm3.pack_prune_candidate(template, keep_percent)
        if repacked != reference_path.read_bytes():
            raise MP2BuildError(
                f"differential selection reference keep{keep_percent} differs from custody"
            )
    selection_map["reference_candidate_raw_payloads"] = reference_records
    prebuilt_selection = atomic_json(
        destination / "retained/selection_map.json",
        selection_map,
    )

    receipt = build_generation(
        output,
        candidate_id=DIFFERENTIAL_CANDIDATE_ID,
        parser="sm3r",
        semantic_stream=semantic_stream,
        semantic_raw=semantic_raw,
        expected_delta_bytes=None,
        base_parts=base_parts,
        template=template,
        selection_metadata=selection_map,
        additional_retained_payloads={
            "differential_source_semantic_raw": prebuilt_raw,
            "differential_source_semantic_stream": prebuilt_stream,
            "selection_map": prebuilt_selection,
        },
    )
    fire_order = {
        "schema": "ddm_mp2_differential_advisory_fire_order.v1",
        "seed": SEED,
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN",
        "consumer_store": str(
            (
                output
                / "advisory_n600_cpu"
                / DIFFERENTIAL_CANDIDATE_ID
                / "attempt_0000"
                / "contest_auth_eval.json"
            ).resolve()
        ),
        "candidate": {
            "candidate_id": DIFFERENTIAL_CANDIDATE_ID,
            "archive": receipt["archive"],
            "runtime": receipt["candidate_bound_runtime"],
            "selection_map": receipt["selection_attribution"]["selection_map"],
        },
        "fire_trigger": (
            "MAIN confirms the WD3 teacher-cache scorer lane has released and no full-n600 "
            "scorer is active; reverify the exact archive and candidate-bound runtime hashes, "
            "delete every ._* file from the generation, attempt directory, and pinned mirror "
            "with /usr/bin/find, then launch one retained macOS-CPU advisory n600 through "
            "experiments/contest_auth_eval.py. Admit only if exact recomputed delta S versus "
            "the HV1 same-axis control is below -3.5e-6."
        ),
        "current_fire": False,
    }
    fire_record = atomic_json(output / "DIFFERENTIAL_N600_FIRE_ORDER.json", fire_order)
    result = {
        "schema": "ddm_mp2_differential_receiver_close.v1",
        "axis": AXIS,
        "score_claim": False,
        "seed": SEED,
        "builder_source": file_record(Path(__file__)),
        "receiver_source": file_record(RECEIVER_SOURCE),
        "candidate": {
            "candidate_id": DIFFERENTIAL_CANDIDATE_ID,
            "archive": receipt["archive"],
            "archive_delta_bytes_vs_hv1": receipt["archive_delta_bytes_vs_hv1"],
            "projected_rate_only_delta_score": receipt[
                "projected_rate_only_delta_score"
            ],
            "receiver_closed": receipt["receiver_closed"],
            "semantic_state_exact_to_packer": receipt["receiver_parseback"][
                "semantic_state_exact_to_packer"
            ],
            "semantic_tensor_denominator": receipt["receiver_parseback"][
                "semantic_tensor_denominator"
            ],
            "selection_map": receipt["selection_attribution"]["selection_map"],
            "n600_status": "QUEUED",
        },
        "fire_order": fire_record,
        "complete": True,
    }
    atomic_json(output / "DIFFERENTIAL_RESULT.json", result)
    return result


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
        choices=("preflight", "build", "finalize", "differential", "all"),
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
    if args.stage == "differential":
        result = build_differential(args.output)
        print(json.dumps({"stage": "differential", "complete": result["complete"]}, sort_keys=True))
        return
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
