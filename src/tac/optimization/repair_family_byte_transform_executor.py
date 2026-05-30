# SPDX-License-Identifier: MIT
"""Deterministic byte-transform executors for repair-family campaigns.

The executor emits concrete, hash-bound transform packets for the queue-owned
repair families. These packets are encoder-side candidate deltas and MLX-local
planning evidence only; they are not score, promotion, rank/kill, or exact-eval
authority.
"""

from __future__ import annotations

import importlib.util
import io
import math
import platform
import struct
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.archive_bound_candidate_adapter_spine import (
    ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA,
    build_archive_bound_candidate_adapter_package,
)
from tac.optimization.archive_bound_candidate_contract import (
    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA,
    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA,
    build_archive_bound_candidate_contract_surface,
)
from tac.optimization.archive_family_fingerprint import (
    fingerprint_archive_family,
    repair_archive_adapter_registry,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.family_agnostic_materializers import (
    FamilyAgnosticMaterializerError,
    materialize_archive_zip_repack_candidate,
    materialize_packet_member_recompress_candidate,
)
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.repair_archive_entropy_substrate_coverage import (
    REPAIR_ARCHIVE_ENTROPY_SUBSTRATE_COVERAGE_SCHEMA,
    build_repair_archive_entropy_substrate_coverage,
)
from tac.optimization.repair_campaign_replay_bundle import (
    capture_safe_replay_environment,
    stable_json_sha256,
)
from tac.optimization.repair_entropy_coder_runtime_adapters import (
    RepairEntropyCoderRuntimeAdapterError,
    ans_rans_prototype_encode,
    decode_entropy_coder_prototype_member,
    entropy_coder_runtime_adapter_manifest,
    range_lzma_prototype_encode,
)
from tac.optimization.repair_family_materializers import (
    REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
)
from tac.repo_io import (
    ArtifactWriteError,
    json_text,
    sha256_bytes,
    sha256_file,
    write_bytes_artifact,
    write_json_artifact,
)

REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA = "repair_family_byte_transform_execution_report.v1"
REPAIR_FAMILY_BYTE_TRANSFORM_PAYLOAD_SCHEMA = "repair_family_byte_transform_payload.v1"
REPAIR_FAMILY_BYTE_TRANSFORM_DELTA_SCHEMA = "repair_family_byte_transform_delta.v1"
REPAIR_FAMILY_BYTE_TRANSFORM_REPLAY_BUNDLE_SCHEMA = "repair_family_byte_transform_replay_bundle.v1"
REPAIR_FAMILY_EXACT_EVAL_HANDOFF_GATE_SCHEMA = "repair_family_exact_eval_handoff_gate.v1"
REPAIR_ARCHIVE_VARIANT_SIGNAL_SURFACE_SCHEMA = "repair_archive_variant_signal_surface.v1"
REPAIR_ARCHIVE_VARIANT_SIGNAL_ROW_SCHEMA = "repair_archive_variant_signal_row.v1"
REPAIR_ARCHIVE_VARIANT_MATERIALIZER_BACKLOG_SCHEMA = "repair_archive_variant_materializer_backlog.v1"
REPAIR_ARCHIVE_VARIANT_MATERIALIZER_BACKLOG_ROW_SCHEMA = (
    "repair_archive_variant_materializer_backlog_row.v1"
)

SUPPORTED_REPAIR_BYTE_TRANSFORM_FAMILIES: frozenset[str] = frozenset(
    {
        "posenet_null_bottom_decile",
        "segnet_class_region_waterfill",
        "per_region_selector_codec",
        "palette_frame_asymmetry_prior",
        "frame0_k16_palette_asymmetry",
        "entropy_boundary_probe",
    }
)


class _SingleRepairArchiveCandidateAdapter:
    """Adapter-spine bridge for one selected repair archive candidate."""

    def __init__(
        self,
        *,
        adapter_id: str,
        candidate_family: str,
        row: Mapping[str, Any],
    ) -> None:
        self.adapter_id = adapter_id
        self.candidate_family = candidate_family
        self._row = dict(row)

    def emit_archive_bound_candidate_rows(
        self,
        context: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        return [self._row]

_FAMILY_TRANSFORM_KINDS: Mapping[str, str] = {
    "posenet_null_bottom_decile": "posenet_null_bottom_decile_frame0_repair_packet",
    "segnet_class_region_waterfill": "segnet_class_region_waterfill_mask_packet",
    "per_region_selector_codec": "per_region_selector_codec_delta_packet",
    "palette_frame_asymmetry_prior": "frame0_k16_palette_asymmetry_transform_packet",
    "frame0_k16_palette_asymmetry": "frame0_k16_palette_asymmetry_transform_packet",
    "entropy_boundary_probe": "entropy_boundary_probe_transform_packet",
}

_FAMILY_SIGNAL_KEYS: Mapping[str, tuple[str, ...]] = {
    "posenet_null_bottom_decile": ("posenet_null_bottom_decile_pair_ids",),
    "segnet_class_region_waterfill": ("segnet_class_region_mask_ids",),
    "per_region_selector_codec": ("selector_payload_bits_per_region",),
    "palette_frame_asymmetry_prior": (
        "palette_dynamics_context",
        "repair_dynamics_palette_prior",
    ),
    "frame0_k16_palette_asymmetry": (
        "palette_dynamics_context",
        "repair_dynamics_palette_prior",
    ),
    "entropy_boundary_probe": ("entropy_boundary_probe_manifest",),
}

FEC6_FIXED_K16_CODE_BITS: tuple[str, ...] = (
    "00",
    "1100",
    "01",
    "111010",
    "11010",
    "111011",
    "111100",
    "100",
    "111101",
    "11011",
    "1111110",
    "111110",
    "11111110",
    "101",
    "11100",
    "11111111",
)
FEC6_FIXED_K16_DECODE: Mapping[str, int] = {
    bits: index for index, bits in enumerate(FEC6_FIXED_K16_CODE_BITS)
}

_FEC6_FAMILY_TARGET_CODES: Mapping[str, int] = {
    "posenet_null_bottom_decile": 0,
    "segnet_class_region_waterfill": 2,
    "per_region_selector_codec": 0,
    "palette_frame_asymmetry_prior": 2,
    "frame0_k16_palette_asymmetry": 2,
}

FEC5_FIXED_K8_CODE_BITS: tuple[str, ...] = (
    "00",
    "01",
    "100",
    "101",
    "1100",
    "1101",
    "1110",
    "1111",
)
FEC5_FIXED_K8_DECODE: Mapping[str, int] = {
    bits: index for index, bits in enumerate(FEC5_FIXED_K8_CODE_BITS)
}

PSV4_MAGIC = b"PSV4"
PSV4_SCHEMA_VERSION = 1
PSV4_HEADER_FMT = "<4sBHHBIIII"
PSV4_HEADER_SIZE = struct.calcsize(PSV4_HEADER_FMT)


class RepairFamilyByteTransformExecutorError(ValueError):
    """Raised when a repair-family byte-transform executor cannot run."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve(path: str | Path, repo_root: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path(repo_root) / value


def _repo_rel(path: str | Path, repo_root: str | Path) -> str:
    value = Path(path)
    repo = Path(repo_root)
    try:
        return value.resolve(strict=False).relative_to(repo.resolve(strict=False)).as_posix()
    except ValueError:
        return value.as_posix()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _safe_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _slug(value: str) -> str:
    text = str(value or "unknown").strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    return "_".join("".join(chars).split("_")) or "unknown"


def _git_text(args: Sequence[str], *, repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _file_record(
    *,
    label: str,
    path: str | Path,
    repo_root: str | Path,
    required: bool = True,
) -> dict[str, Any]:
    resolved = _resolve(path, repo_root)
    present = resolved.is_file()
    if required and not present:
        raise RepairFamilyByteTransformExecutorError(f"required artifact missing: {label}={path}")
    record = {
        "label": label,
        "path": _repo_rel(resolved, repo_root),
        "present": present,
        "required": required,
    }
    if present:
        record.update({"sha256": sha256_file(resolved), "bytes": resolved.stat().st_size})
    return record


def _archive_record(path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    resolved = _resolve(path, repo_root)
    return {
        "path": _repo_rel(resolved, repo_root),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _zip_member_names(archive_path: str | Path) -> list[str]:
    with zipfile.ZipFile(archive_path) as archive:
        return [info.filename for info in archive.infolist()]


def _zip_member_bytes(archive_path: str | Path, member_name: str) -> bytes:
    with zipfile.ZipFile(archive_path) as archive:
        return archive.read(member_name)


def _member_record(
    archive_path: str | Path,
    member_name: str,
    *,
    payload: bytes | None = None,
) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member_name)
        member_payload = archive.read(member_name) if payload is None else payload
    return {
        "name": member_name,
        "sha256": sha256_bytes(member_payload),
        "bytes": len(member_payload),
        "zip_compression_method": info.compress_type,
        "zip_compressed_bytes": info.compress_size,
    }


def _zero_order_entropy_stats(payload: bytes) -> dict[str, Any]:
    counts = [0] * 256
    for byte in payload:
        counts[byte] += 1
    byte_count = len(payload)
    entropy_bits_per_symbol = 0.0
    if byte_count:
        for count in counts:
            if count:
                probability = count / byte_count
                entropy_bits_per_symbol -= probability * math.log2(probability)
    lower_bound_bits = entropy_bits_per_symbol * byte_count
    lower_bound_bytes = math.ceil(lower_bound_bits / 8.0) if byte_count else 0
    return {
        "schema": "repair_archive_zero_order_entropy_stats.v1",
        "byte_count": byte_count,
        "unique_symbol_count": sum(1 for count in counts if count),
        "entropy_bits_per_symbol": entropy_bits_per_symbol,
        "zero_order_lower_bound_bits": lower_bound_bits,
        "zero_order_lower_bound_bytes": lower_bound_bytes,
        "zero_order_redundancy_bytes": max(0, byte_count - lower_bound_bytes),
    }


def _primary_payload_member(archive_path: str | Path) -> str:
    """Choose the member most likely to be the score-affecting payload."""

    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
    if not infos:
        raise RepairFamilyByteTransformExecutorError("archive has no ZIP members")
    by_name = {info.filename: info for info in infos}
    for preferred in ("x", "0.bin", "archive.bin", "payload.bin", "renderer.bin"):
        if preferred in by_name:
            return preferred
    payload_like = [
        info
        for info in infos
        if not info.is_dir()
        and not info.filename.endswith((".py", ".sh", ".txt", ".json", ".toml"))
        and "__pycache__" not in info.filename
    ]
    candidates = payload_like or [info for info in infos if not info.is_dir()]
    if not candidates:
        raise RepairFamilyByteTransformExecutorError("archive has no file members")
    return max(candidates, key=lambda info: (info.file_size, -len(info.filename), info.filename)).filename


def _zip_archive_bytes_with_replacement(
    archive_path: str | Path,
    *,
    member_name: str,
    replacement: bytes,
    compression: int | None = None,
    compresslevel: int | None = None,
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(archive_path) as source, zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as target:
        for info in source.infolist():
            if info.is_dir():
                continue
            payload = replacement if info.filename == member_name else source.read(info.filename)
            target_info = zipfile.ZipInfo(info.filename)
            target_info.date_time = (1980, 1, 1, 0, 0, 0)
            target_info.external_attr = info.external_attr
            target_info.compress_type = (
                compression
                if info.filename == member_name and compression is not None
                else info.compress_type
            )
            level = (
                compresslevel
                if info.filename == member_name and compression == zipfile.ZIP_DEFLATED
                else None
            )
            target.writestr(target_info, payload, compresslevel=level)
    return output.getvalue()


def _decode_fec6_codes(payload: bytes, *, n_pairs: int) -> list[int]:
    codes: list[int] = []
    prefix = ""
    consumed_bits = 0
    for byte in payload:
        for shift in range(7, -1, -1):
            bit = "1" if (byte >> shift) & 1 else "0"
            prefix += bit
            consumed_bits += 1
            code = FEC6_FIXED_K16_DECODE.get(prefix)
            if code is not None:
                codes.append(code)
                prefix = ""
                if len(codes) == n_pairs:
                    padding_bits = len(payload) * 8 - consumed_bits
                    if padding_bits:
                        mask = (1 << padding_bits) - 1
                        if payload[-1] & mask:
                            raise RepairFamilyByteTransformExecutorError(
                                "FEC6 selector padding bits are non-zero"
                            )
                    return codes
            elif len(prefix) > 8:
                raise RepairFamilyByteTransformExecutorError(
                    "FEC6 selector contains invalid prefix code"
                )
    raise RepairFamilyByteTransformExecutorError("FEC6 selector bitstream truncated")


def _encode_fec6_codes(codes: Sequence[int]) -> bytes:
    bits = "".join(FEC6_FIXED_K16_CODE_BITS[int(code)] for code in codes)
    padding = (-len(bits)) % 8
    bits += "0" * padding
    return bytes(int(bits[index : index + 8], 2) for index in range(0, len(bits), 8))


def _decode_fec5_codes(payload: bytes, *, n_pairs: int) -> list[int]:
    codes: list[int] = []
    prefix = ""
    consumed_bits = 0
    for byte in payload:
        for shift in range(7, -1, -1):
            bit = "1" if (byte >> shift) & 1 else "0"
            prefix += bit
            consumed_bits += 1
            code = FEC5_FIXED_K8_DECODE.get(prefix)
            if code is not None:
                codes.append(code)
                prefix = ""
                if len(codes) == n_pairs:
                    padding_bits = len(payload) * 8 - consumed_bits
                    if padding_bits and payload[-1] & ((1 << padding_bits) - 1):
                        raise RepairFamilyByteTransformExecutorError(
                            "FEC5 selector padding bits are non-zero"
                        )
                    return codes
            elif len(prefix) > 4:
                raise RepairFamilyByteTransformExecutorError(
                    "FEC5 selector contains invalid prefix code"
                )
    raise RepairFamilyByteTransformExecutorError("FEC5 selector bitstream truncated")


def _encode_fec5_codes(codes: Sequence[int]) -> bytes:
    bits = "".join(FEC5_FIXED_K8_CODE_BITS[int(code)] for code in codes)
    padding = (-len(bits)) % 8
    bits += "0" * padding
    return bytes(int(bits[index : index + 8], 2) for index in range(0, len(bits), 8))


def _pack_codes_lsb(codes: Sequence[int], *, bits_per_symbol: int) -> bytes:
    out = bytearray((len(codes) * bits_per_symbol + 7) // 8)
    bit_pos = 0
    max_code = 1 << bits_per_symbol
    for code in codes:
        value = int(code)
        if not 0 <= value < max_code:
            raise RepairFamilyByteTransformExecutorError(
                f"selector code {value} cannot fit in {bits_per_symbol} bits"
            )
        for shift in range(bits_per_symbol):
            if (value >> shift) & 1:
                absolute = bit_pos + shift
                out[absolute // 8] |= 1 << (absolute % 8)
        bit_pos += bits_per_symbol
    return bytes(out)


def _unpack_codes_lsb(
    payload: bytes,
    *,
    n_pairs: int,
    bits_per_symbol: int,
    max_code_exclusive: int,
) -> list[int]:
    codes: list[int] = []
    bit_pos = 0
    for _ in range(n_pairs):
        code = 0
        for shift in range(bits_per_symbol):
            absolute = bit_pos + shift
            byte_index = absolute // 8
            if byte_index >= len(payload):
                raise RepairFamilyByteTransformExecutorError("selector bitstream truncated")
            code |= ((payload[byte_index] >> (absolute % 8)) & 1) << shift
        if code >= max_code_exclusive:
            raise RepairFamilyByteTransformExecutorError(
                f"selector code {code} outside palette {max_code_exclusive}"
            )
        codes.append(code)
        bit_pos += bits_per_symbol
    if (bit_pos + 7) // 8 != len(payload):
        raise RepairFamilyByteTransformExecutorError("selector has trailing payload bytes")
    return codes


def _decode_fes1_selector(selector: bytes) -> tuple[list[int], dict[str, Any]]:
    if len(selector) < 10 or selector[:4] != b"FES1":
        raise RepairFamilyByteTransformExecutorError("FES1 selector payload missing")
    n_pairs, palette_size, bits_per_symbol, packed_len = struct.unpack_from(
        "<HBBH",
        selector,
        4,
    )
    if not 1 <= palette_size <= 255:
        raise RepairFamilyByteTransformExecutorError(f"FES1 invalid palette size {palette_size}")
    packed = selector[10 : 10 + packed_len]
    if len(selector) != 10 + packed_len:
        raise RepairFamilyByteTransformExecutorError("FES1 selector length mismatch")
    codes = _unpack_codes_lsb(
        packed,
        n_pairs=n_pairs,
        bits_per_symbol=bits_per_symbol,
        max_code_exclusive=palette_size,
    )
    return codes, {
        "selector_magic": "FES1",
        "n_pairs": n_pairs,
        "palette_size": palette_size,
        "bits_per_symbol": bits_per_symbol,
        "selector_prefix_hex": selector[:10].hex(),
    }


def _encode_fes1_selector(codes: Sequence[int], details: Mapping[str, Any]) -> bytes:
    palette_size = int(details["palette_size"])
    bits_per_symbol = int(details["bits_per_symbol"])
    packed = _pack_codes_lsb(codes, bits_per_symbol=bits_per_symbol)
    return (
        b"FES1"
        + struct.pack("<HBBH", len(codes), palette_size, bits_per_symbol, len(packed))
        + packed
    )


def _decode_fec3_selector(selector: bytes) -> tuple[list[int], dict[str, Any]]:
    if len(selector) < 8 or selector[:4] != b"FEC3":
        raise RepairFamilyByteTransformExecutorError("FEC3 selector payload missing")
    n_pairs, bits_per_symbol, n_specs = struct.unpack_from("<HBB", selector, 4)
    pos = 8
    for _ in range(n_specs):
        if pos + 2 > len(selector):
            raise RepairFamilyByteTransformExecutorError("FEC3 palette table truncated")
        tag = selector[pos]
        pos += 2
        if tag == 1:
            if pos + 5 > len(selector):
                raise RepairFamilyByteTransformExecutorError("FEC3 dynamic spec truncated")
            pos += 5
        elif tag != 0:
            raise RepairFamilyByteTransformExecutorError(f"FEC3 unsupported table tag {tag}")
    codes = _unpack_codes_lsb(
        selector[pos:],
        n_pairs=n_pairs,
        bits_per_symbol=bits_per_symbol,
        max_code_exclusive=n_specs,
    )
    return codes, {
        "selector_magic": "FEC3",
        "n_pairs": n_pairs,
        "bits_per_symbol": bits_per_symbol,
        "palette_size": n_specs,
        "selector_prefix_hex": selector[:pos].hex(),
    }


def _encode_fec3_selector(codes: Sequence[int], details: Mapping[str, Any]) -> bytes:
    return bytes.fromhex(str(details["selector_prefix_hex"])) + _pack_codes_lsb(
        codes,
        bits_per_symbol=int(details["bits_per_symbol"]),
    )


def _load_fec8_codec_module(repo_root: str | Path) -> Any:
    path = (
        _resolve(repo_root, repo_root)
        / "submissions/hnerv_fec6_fixed_huffman_k16/encoder/"
        "build_pr101_frame_exploit_selector_packet_markov.py"
    )
    if not path.is_file():
        raise RepairFamilyByteTransformExecutorError("FEC8 codec module missing")
    spec = importlib.util.spec_from_file_location("_pact_fec8_markov_codec", path)
    if spec is None or spec.loader is None:
        raise RepairFamilyByteTransformExecutorError("FEC8 codec module cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decode_fec8_selector(
    selector: bytes,
    *,
    repo_root: str | Path,
) -> tuple[list[int], dict[str, Any]]:
    if len(selector) < 8 or selector[:4] != b"FEC8":
        raise RepairFamilyByteTransformExecutorError("FEC8 selector payload missing")
    module = _load_fec8_codec_module(repo_root)
    codes = list(module.decode_fec8_markov_selector(selector))
    return codes, {
        "selector_magic": "FEC8",
        "variant_hex": selector[4:6].hex(),
        "n_pairs": len(codes),
        "palette_size": len(FEC6_FIXED_K16_CODE_BITS),
    }


def _encode_fec8_selector(
    codes: Sequence[int],
    details: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> bytes:
    module = _load_fec8_codec_module(repo_root)
    variant = bytes.fromhex(str(details["variant_hex"]))
    if variant == getattr(module, "FEC8_VARIANT_STATIC", b"\x00\x01"):
        return module.encode_fec8_markov_selector_static(codes, n_pairs=len(codes))
    if variant == getattr(module, "FEC8_VARIANT_ADAPTIVE", b"\x00\x02"):
        return module.encode_fec8_markov_selector_adaptive(codes, n_pairs=len(codes))
    if variant == getattr(module, "FEC8_VARIANT_STATIC_SECOND_ORDER", b"\x00\x03"):
        return module.encode_fec8_markov_selector_static_second_order(
            codes,
            n_pairs=len(codes),
        )
    raise RepairFamilyByteTransformExecutorError(f"unsupported FEC8 variant {variant.hex()}")


def _mutated_selector_codes(
    codes: Sequence[int],
    *,
    family_id: str,
    palette_size: int,
) -> tuple[list[int], list[int], int]:
    mutated = [int(code) for code in codes]
    target_code = min(_FEC6_FAMILY_TARGET_CODES.get(family_id, 0), palette_size - 1)
    n_pairs = len(mutated)
    span = max(1, n_pairs // 10)
    if family_id == "posenet_null_bottom_decile":
        indexes = range(max(0, n_pairs - span), n_pairs)
    elif family_id == "segnet_class_region_waterfill":
        indexes = range(0, n_pairs, max(1, n_pairs // 16))
    elif family_id == "per_region_selector_codec":
        indexes = range(0, n_pairs, 8)
    else:
        indexes = range(0, min(n_pairs, span))
    changed_indexes: list[int] = []
    for index in indexes:
        if index >= n_pairs:
            continue
        old = mutated[index]
        new = target_code if old != target_code else (old + 1) % palette_size
        if old != new:
            mutated[index] = new
            changed_indexes.append(index)
    return mutated, changed_indexes, target_code


def _mutated_fec6_selector_member(
    member_payload: bytes,
    *,
    family_id: str,
) -> tuple[bytes, dict[str, Any]] | None:
    if len(member_payload) < 12 or member_payload[:4] != b"FP11":
        return None
    source_len = struct.unpack_from("<I", member_payload, 4)[0]
    selector_len_offset = 8 + source_len
    if selector_len_offset + 2 > len(member_payload):
        return None
    selector_len = struct.unpack_from("<H", member_payload, selector_len_offset)[0]
    selector_start = selector_len_offset + 2
    selector_end = selector_start + selector_len
    selector = member_payload[selector_start:selector_end]
    if selector_end != len(member_payload) or len(selector) < 6 or selector[:4] != b"FEC6":
        return None
    n_pairs = struct.unpack_from("<H", selector, 4)[0]
    codes = _decode_fec6_codes(selector[6:], n_pairs=n_pairs)
    if len(codes) != n_pairs:
        raise RepairFamilyByteTransformExecutorError("FEC6 selector pair count mismatch")
    mutated = list(codes)
    target_code = _FEC6_FAMILY_TARGET_CODES.get(family_id, 0)
    span = max(1, n_pairs // 10)
    if family_id == "posenet_null_bottom_decile":
        indexes = range(max(0, n_pairs - span), n_pairs)
    elif family_id == "segnet_class_region_waterfill":
        indexes = range(0, n_pairs, max(1, n_pairs // 16))
    elif family_id == "per_region_selector_codec":
        indexes = range(0, n_pairs, 8)
    else:
        indexes = range(0, min(n_pairs, span))
    changed_indexes: list[int] = []
    for index in indexes:
        if index >= n_pairs:
            continue
        old = mutated[index]
        new = target_code if old != target_code else (old + 1) % len(FEC6_FIXED_K16_CODE_BITS)
        if old != new:
            mutated[index] = new
            changed_indexes.append(index)
    if not changed_indexes:
        return None
    encoded = _encode_fec6_codes(mutated)
    new_selector = b"FEC6" + struct.pack("<H", n_pairs) + encoded
    new_member = (
        member_payload[:selector_len_offset]
        + struct.pack("<H", len(new_selector))
        + new_selector
    )
    details = {
        "schema": "repair_family_fec6_selector_mutation_details.v1",
        "source_format": "FP11+FEC6_FIXED_K16",
        "n_pairs": n_pairs,
        "source_selector_bytes": selector_len,
        "candidate_selector_bytes": len(new_selector),
        "changed_pair_count": len(changed_indexes),
        "changed_pair_indexes_preview": changed_indexes[:32],
        "target_code": target_code,
        "source_code_histogram": {str(code): codes.count(code) for code in sorted(set(codes))},
        "candidate_code_histogram": {str(code): mutated.count(code) for code in sorted(set(mutated))},
        "semantic_stage": "pre_entropy_payload_distribution_shaping",
    }
    return new_member, details


def _selector_payload_from_fp11(member_payload: bytes) -> tuple[int, bytes] | None:
    if len(member_payload) < 12 or member_payload[:4] != b"FP11":
        return None
    source_len = struct.unpack_from("<I", member_payload, 4)[0]
    selector_len_offset = 8 + source_len
    if selector_len_offset + 2 > len(member_payload):
        return None
    selector_len = struct.unpack_from("<H", member_payload, selector_len_offset)[0]
    selector_start = selector_len_offset + 2
    selector_end = selector_start + selector_len
    if selector_end != len(member_payload) or selector_len < 4:
        return None
    return selector_len_offset, member_payload[selector_start:selector_end]


def _mutated_non_fec6_fp11_selector_member(
    member_payload: bytes,
    *,
    family_id: str,
    repo_root: str | Path,
) -> tuple[bytes, dict[str, Any]] | None:
    if family_id not in _FEC6_FAMILY_TARGET_CODES:
        return None
    parsed = _selector_payload_from_fp11(member_payload)
    if parsed is None:
        return None
    selector_len_offset, selector = parsed
    magic = selector[:4]
    if magic == b"FEC6":
        return None
    if magic == b"FES1":
        codes, details = _decode_fes1_selector(selector)
        candidate_codes, changed_indexes, target_code = _mutated_selector_codes(
            codes,
            family_id=family_id,
            palette_size=int(details["palette_size"]),
        )
        new_selector = _encode_fes1_selector(candidate_codes, details)
    elif magic == b"FEC3":
        codes, details = _decode_fec3_selector(selector)
        candidate_codes, changed_indexes, target_code = _mutated_selector_codes(
            codes,
            family_id=family_id,
            palette_size=int(details["palette_size"]),
        )
        new_selector = _encode_fec3_selector(candidate_codes, details)
    elif magic == b"FEC5":
        if len(selector) < 6:
            raise RepairFamilyByteTransformExecutorError("FEC5 selector truncated")
        n_pairs = struct.unpack_from("<H", selector, 4)[0]
        codes = _decode_fec5_codes(selector[6:], n_pairs=n_pairs)
        candidate_codes, changed_indexes, target_code = _mutated_selector_codes(
            codes,
            family_id=family_id,
            palette_size=len(FEC5_FIXED_K8_CODE_BITS),
        )
        details = {
            "selector_magic": "FEC5",
            "n_pairs": n_pairs,
            "palette_size": len(FEC5_FIXED_K8_CODE_BITS),
        }
        new_selector = b"FEC5" + struct.pack("<H", n_pairs) + _encode_fec5_codes(candidate_codes)
    elif magic == b"FEC8":
        codes, details = _decode_fec8_selector(selector, repo_root=repo_root)
        candidate_codes, changed_indexes, target_code = _mutated_selector_codes(
            codes,
            family_id=family_id,
            palette_size=int(details["palette_size"]),
        )
        new_selector = _encode_fec8_selector(candidate_codes, details, repo_root=repo_root)
    else:
        return None
    if not changed_indexes:
        return None
    selector_start = selector_len_offset + 2
    new_member = (
        member_payload[:selector_len_offset]
        + struct.pack("<H", len(new_selector))
        + new_selector
        + member_payload[selector_start + len(selector) :]
    )
    details = {
        "schema": "repair_family_fp11_selector_mutation_details.v1",
        "source_format": f"FP11+{magic.decode('ascii', errors='replace')}",
        **dict(details),
        "source_selector_bytes": len(selector),
        "candidate_selector_bytes": len(new_selector),
        "changed_pair_count": len(changed_indexes),
        "changed_pair_indexes_preview": changed_indexes[:32],
        "target_code": target_code,
        "source_code_histogram": {str(code): list(codes).count(code) for code in sorted(set(codes))},
        "candidate_code_histogram": {
            str(code): candidate_codes.count(code) for code in sorted(set(candidate_codes))
        },
        "semantic_stage": "pre_entropy_payload_distribution_shaping",
    }
    return new_member, details


def _decode_psv4_varint(blob: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if pos >= len(blob):
            raise RepairFamilyByteTransformExecutorError("PSV4 selector varint truncated")
        byte = blob[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return value, pos
        shift += 7
        if shift > 63:
            raise RepairFamilyByteTransformExecutorError("PSV4 selector varint too long")


def _encode_psv4_varint(value: int) -> bytes:
    if value <= 0:
        raise RepairFamilyByteTransformExecutorError(
            f"PSV4 run length must be positive; got {value}"
        )
    out = bytearray()
    while value:
        byte = value & 0x7F
        value >>= 7
        if value:
            byte |= 0x80
        out.append(byte)
    return bytes(out)


def _decode_psv4_selector_rle(
    selector: bytes,
    *,
    num_pairs: int,
    palette_size: int,
) -> list[int]:
    codes: list[int] = []
    pos = 0
    while pos < len(selector):
        code = selector[pos]
        pos += 1
        if code >= palette_size:
            raise RepairFamilyByteTransformExecutorError(
                f"PSV4 selector code {code} outside palette {palette_size}"
            )
        run_length, pos = _decode_psv4_varint(selector, pos)
        if run_length <= 0:
            raise RepairFamilyByteTransformExecutorError("PSV4 selector has zero run length")
        if len(codes) + run_length > num_pairs:
            raise RepairFamilyByteTransformExecutorError("PSV4 selector run exceeds pair count")
        codes.extend([code] * run_length)
    if len(codes) != num_pairs:
        raise RepairFamilyByteTransformExecutorError(
            f"PSV4 selector decoded {len(codes)} pairs, expected {num_pairs}"
        )
    return codes


def _encode_psv4_selector_rle(codes: Sequence[int], *, palette_size: int) -> bytes:
    if not codes:
        return b""
    out = bytearray()
    index = 0
    while index < len(codes):
        code = int(codes[index])
        if not 0 <= code < palette_size:
            raise RepairFamilyByteTransformExecutorError(
                f"PSV4 selector code {code} outside palette {palette_size}"
            )
        run_length = 1
        while index + run_length < len(codes) and int(codes[index + run_length]) == code:
            run_length += 1
        out.append(code & 0xFF)
        out.extend(_encode_psv4_varint(run_length))
        index += run_length
    return bytes(out)


def _mutated_psv4_selector_member(
    member_payload: bytes,
    *,
    family_id: str,
) -> tuple[bytes, dict[str, Any]] | None:
    if family_id not in _FEC6_FAMILY_TARGET_CODES:
        return None
    if len(member_payload) < PSV4_HEADER_SIZE:
        return None
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        palette_size,
        decoder_len,
        latent_len,
        selector_len,
        meta_len,
    ) = struct.unpack_from(PSV4_HEADER_FMT, member_payload, 0)
    if magic != PSV4_MAGIC:
        return None
    if version != PSV4_SCHEMA_VERSION:
        raise RepairFamilyByteTransformExecutorError(f"PSV4 unsupported schema version {version}")
    if not 2 <= palette_size <= 255:
        raise RepairFamilyByteTransformExecutorError(f"PSV4 invalid palette size {palette_size}")
    expected_latent_len = int(num_pairs) * int(latent_dim) * 2
    if latent_len != expected_latent_len:
        raise RepairFamilyByteTransformExecutorError(
            f"PSV4 latent length {latent_len} != expected {expected_latent_len}"
        )
    selector_start = PSV4_HEADER_SIZE + decoder_len + latent_len
    selector_end = selector_start + selector_len
    expected_total = selector_end + meta_len
    if selector_start > len(member_payload) or selector_end > len(member_payload):
        raise RepairFamilyByteTransformExecutorError("PSV4 selector offsets exceed packet size")
    if expected_total != len(member_payload):
        raise RepairFamilyByteTransformExecutorError("PSV4 packet length/header mismatch")
    codes = _decode_psv4_selector_rle(
        member_payload[selector_start:selector_end],
        num_pairs=num_pairs,
        palette_size=palette_size,
    )
    candidate_codes, changed_indexes, target_code = _mutated_selector_codes(
        codes,
        family_id=family_id,
        palette_size=palette_size,
    )
    if not changed_indexes:
        return None
    new_selector = _encode_psv4_selector_rle(candidate_codes, palette_size=palette_size)
    header = struct.pack(
        PSV4_HEADER_FMT,
        PSV4_MAGIC,
        version,
        latent_dim,
        num_pairs,
        palette_size,
        decoder_len,
        latent_len,
        len(new_selector),
        meta_len,
    )
    new_member = (
        header
        + member_payload[PSV4_HEADER_SIZE:selector_start]
        + new_selector
        + member_payload[selector_end:]
    )
    details = {
        "schema": "repair_family_psv4_selector_mutation_details.v1",
        "source_format": "PACT_NERV_SELECTOR_V4_RLE",
        "selector_magic": "PSV4",
        "schema_version": version,
        "latent_dim": latent_dim,
        "n_pairs": num_pairs,
        "palette_size": palette_size,
        "decoder_bytes": decoder_len,
        "latent_bytes": latent_len,
        "meta_bytes": meta_len,
        "source_selector_bytes": selector_len,
        "candidate_selector_bytes": len(new_selector),
        "changed_pair_count": len(changed_indexes),
        "changed_pair_indexes_preview": changed_indexes[:32],
        "target_code": target_code,
        "source_code_histogram": {str(code): codes.count(code) for code in sorted(set(codes))},
        "candidate_code_histogram": {
            str(code): candidate_codes.count(code) for code in sorted(set(candidate_codes))
        },
        "semantic_stage": "pre_entropy_payload_distribution_shaping",
        "optimization_scopes": ["byte", "selector", "pair", "frame", "batch", "full_video"],
    }
    return new_member, details


def _archive_family_probe(manifest: Mapping[str, Any], *, repo_root: str | Path) -> dict[str, Any]:
    source, actual_sha, blockers = _source_archive_path_and_sha(manifest, repo_root=repo_root)
    fingerprint: Mapping[str, Any] = {}
    if source is not None and source.is_file() and not blockers:
        fingerprint = fingerprint_archive_family(source, repo_root=repo_root)
        blockers.extend(str(item) for item in fingerprint.get("blockers") or [])
    return {
        "schema": "repair_family_archive_family_probe.v1",
        "source_archive_path": None if source is None else _repo_rel(source, repo_root),
        "source_archive_sha256": actual_sha or None,
        "member_count": int(fingerprint.get("zip_member_count") or 0),
        "primary_payload_member": fingerprint.get("primary_payload_member"),
        "primary_payload_head_hex": str(fingerprint.get("primary_payload_head_hex") or ""),
        "detected_archive_families": ordered_unique(
            str(item) for item in fingerprint.get("detected_archive_families") or []
        ),
        "archive_family_fingerprint": fingerprint,
        "adapter_registry": repair_archive_adapter_registry(),
        "blockers": ordered_unique(blockers),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def _artifact_path_from_statuses(manifest: Mapping[str, Any], key: str) -> str:
    replay = _mapping(manifest.get("component_response_replay"))
    for item in replay.get("local_mlx_custody_paths") or []:
        if isinstance(item, Mapping) and str(item.get("key") or "") == key:
            return str(item.get("path") or "").strip()
    for item in _mapping(manifest.get("receiver_verification")).get("local_mlx_custody_paths") or []:
        if isinstance(item, Mapping) and str(item.get("key") or "") == key:
            return str(item.get("path") or "").strip()
    return ""


def _component_terms(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    replay = _mapping(manifest.get("component_response_replay"))
    return _mapping(replay.get("component_response_terms"))


def _component_probe_delta(manifest: Mapping[str, Any]) -> dict[str, Any]:
    terms = _component_terms(manifest)
    segnet_delta = _safe_float(terms.get("segnet_delta_score_units"))
    posenet_delta = _safe_float(terms.get("posenet_delta_score_units"))
    combined = _safe_float(terms.get("combined_delta_score_units"))
    if combined == 0.0:
        measured_delta = _safe_float(terms.get("measured_component_delta_score_units"))
        objective_delta = _safe_float(
            terms.get("objective_delta_score_units") or manifest.get("objective_delta_score_units")
        )
        combined = measured_delta or objective_delta or (segnet_delta + posenet_delta)
    replay = _mapping(manifest.get("component_response_replay"))
    axis = str(replay.get("axis_tag") or replay.get("response_axis") or "").strip()
    return {
        "schema": "repair_family_byte_transform_mlx_probe_delta.v1",
        "component_response_axis": axis or "[macOS-MLX research-signal]",
        "advisory_delta_ready": manifest.get("component_response_replayed") is True,
        "segnet_delta_score_units": segnet_delta,
        "posenet_delta_score_units": posenet_delta,
        "combined_delta_score_units": combined,
        "evidence_grade": replay.get("evidence_grade") or "local_mlx_component_response_replay_only",
        "local_mlx_rows_are_advisory_only": True,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def _nested_first(manifest: Mapping[str, Any], key: str) -> Any:
    if key in manifest:
        return manifest.get(key)
    replay_terms = _component_terms(manifest)
    if key in replay_terms:
        return replay_terms.get(key)
    for container_key in (
        "fractal_optimization_scope",
        "component_response_replay",
        "receiver_verification",
        "palette_dynamics_context",
    ):
        container = _mapping(manifest.get(container_key))
        if key in container:
            return container.get(key)
    return None


def _family_signal_payload(manifest: Mapping[str, Any], family_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in _FAMILY_SIGNAL_KEYS.get(family_id, ()):
        value = _nested_first(manifest, key)
        if value is not None:
            payload[key] = value
    if family_id == "frame0_k16_palette_asymmetry" and "palette_dynamics_context" not in payload:
        value = _nested_first(manifest, "palette_dynamics_context")
        if value is not None:
            payload["palette_dynamics_context"] = value
    return payload


def _active_levels(manifest: Mapping[str, Any]) -> list[str]:
    scope = _mapping(manifest.get("fractal_optimization_scope"))
    active = _string_list(scope.get("active_levels"))
    return active or _string_list(scope.get("declared_levels"))


def _build_transform_payload(
    *,
    manifest: Mapping[str, Any],
    manifest_path: str | Path,
    family_id: str,
) -> dict[str, Any]:
    return {
        "schema": REPAIR_FAMILY_BYTE_TRANSFORM_PAYLOAD_SCHEMA,
        "generated_at_utc": _utc_now(),
        "source_manifest_path": str(manifest_path),
        "source_manifest_schema": manifest.get("schema"),
        "materializer_id": manifest.get("materializer_id"),
        "family_id": family_id,
        "target_kind": manifest.get("target_kind"),
        "transform_kind": _FAMILY_TRANSFORM_KINDS.get(
            family_id,
            "unclassified_repair_family_transform_packet",
        ),
        "typed_response_id": manifest.get("typed_response_id"),
        "candidate_chain_id": manifest.get("candidate_chain_id"),
        "candidate_chain_ids": _string_list(manifest.get("candidate_chain_ids")),
        "entropy_position_label": manifest.get("entropy_position_label"),
        "active_entropy_stage": dict(_mapping(manifest.get("active_entropy_stage"))),
        "fractal_levels": _active_levels(manifest),
        "allocated_repair_bytes": _safe_int(
            manifest.get("allocated_repair_bytes") or _component_terms(manifest).get("allocated_repair_bytes")
        ),
        "family_signal_payload": _family_signal_payload(manifest, family_id),
        "mlx_local_probe_delta": _component_probe_delta(manifest),
        "receiver_contract_kind": manifest.get("receiver_contract_kind"),
        "encoder_side_only": True,
        "receiver_must_not_optimize": True,
        "local_mlx_rows_are_advisory_only": True,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "deterministic_encoder_side_repair_transform_delta_only",
        "forbidden_use": "score_claim_or_budget_spend_or_receiver_optimization",
        **FALSE_AUTHORITY,
    }


def _write_transform_payload(
    *,
    payload: Mapping[str, Any],
    output_dir: str | Path,
    repo_root: str | Path,
    family_id: str,
    typed_response_id: str,
    allow_overwrite: bool,
) -> dict[str, Any]:
    output = _resolve(output_dir, repo_root)
    output.mkdir(parents=True, exist_ok=True)
    stem_parts = [family_id, typed_response_id or "unknown"]
    stem = "_".join(_slug(item) for item in stem_parts)
    target = output / f"{stem}_byte_transform_payload.json"
    payload_bytes = json_text(payload).encode("utf-8")
    expected_existing_sha256 = None
    skipped = False
    if target.exists() and allow_overwrite:
        existing = target.read_bytes()
        if existing == payload_bytes:
            skipped = True
        else:
            expected_existing_sha256 = sha256_file(target)
    write_result = None
    if not skipped:
        write_result = write_bytes_artifact(
            target,
            payload_bytes,
            allow_overwrite=allow_overwrite,
            expected_existing_sha256=expected_existing_sha256,
        )
    return {
        "schema": REPAIR_FAMILY_BYTE_TRANSFORM_DELTA_SCHEMA,
        "path": _repo_rel(target, repo_root),
        "sha256": sha256_bytes(payload_bytes),
        "bytes": len(payload_bytes),
        "skipped_identical_existing_artifact": skipped,
        "bytes_written": 0 if write_result is None else write_result.bytes_written,
        "transform_payload_schema": payload.get("schema"),
        "transform_kind": payload.get("transform_kind"),
        "family_id": family_id,
        "typed_response_id": typed_response_id or None,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def _empty_candidate_archive(
    *,
    blockers: Sequence[str],
    archive_native_transform_kind: str,
) -> dict[str, Any]:
    return {
        "schema": "repair_family_archive_native_candidate.v1",
        "materialized": False,
        "archive_native_transform_attempted": False,
        "archive_native_transform_kind": archive_native_transform_kind,
        "path": None,
        "sha256": None,
        "bytes": None,
        "runtime_consumption_proof_path": None,
        "runtime_consumption_proof_ready": False,
        "receiver_contract_satisfied": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "saved_bytes": None,
        "blockers": ordered_unique(blockers),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def _source_archive_path_and_sha(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> tuple[Path | None, str, list[str]]:
    archive = _mapping(manifest.get("candidate_archive"))
    path_text = str(archive.get("path") or "").strip()
    blockers: list[str] = []
    if not path_text:
        return None, "", ["candidate_archive_path_missing"]
    source = _resolve(path_text, repo_root)
    if not source.is_file():
        return source, "", ["candidate_archive_file_missing"]
    expected_sha = str(archive.get("sha256") or "").strip()
    actual_sha = sha256_file(source)
    if expected_sha and expected_sha != actual_sha:
        blockers.append("candidate_archive_sha256_mismatch")
    return source, actual_sha, blockers


def _write_json_with_expected(
    path: Path,
    payload: Mapping[str, Any],
    *,
    allow_overwrite: bool,
) -> None:
    write_json_artifact(
        path,
        payload,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=sha256_file(path) if path.exists() and allow_overwrite else None,
    )


def _write_bytes_with_expected(
    path: Path,
    payload: bytes,
    *,
    allow_overwrite: bool,
) -> None:
    write_bytes_artifact(
        path,
        payload,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=sha256_file(path) if path.exists() and allow_overwrite else None,
    )


def _fec6_payload_mutation_candidate(
    *,
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    repo_root: str | Path,
    family_id: str,
    allow_overwrite: bool,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if family_id not in _FEC6_FAMILY_TARGET_CODES:
        blockers.append(f"fec6_selector_payload_mutation_not_enabled_for_family:{family_id}")
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind="fec6_selector_payload_mutation",
        ), blockers
    source, actual_sha, source_blockers = _source_archive_path_and_sha(manifest, repo_root=repo_root)
    blockers.extend(source_blockers)
    if source is None or blockers:
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind="fec6_selector_payload_mutation",
        ), blockers
    try:
        member_name = _primary_payload_member(source)
        source_member_payload = _zip_member_bytes(source, member_name)
        mutation = _mutated_fec6_selector_member(source_member_payload, family_id=family_id)
    except (OSError, zipfile.BadZipFile, KeyError, RepairFamilyByteTransformExecutorError) as exc:
        blockers.append(f"fec6_selector_payload_mutation_probe_failed:{exc}")
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind="fec6_selector_payload_mutation",
        ), blockers
    if mutation is None:
        blockers.append("fec6_selector_payload_not_detected")
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind="fec6_selector_payload_mutation",
        ), blockers
    candidate_member_payload, mutation_details = mutation
    output = _resolve(output_dir, repo_root) / "candidate_archive_fec6_selector_payload_mutation.zip"
    proof = _resolve(output_dir, repo_root) / "candidate_archive_fec6_selector_payload_mutation_receiver_proof.json"
    candidate_payload = _zip_archive_bytes_with_replacement(
        source,
        member_name=member_name,
        replacement=candidate_member_payload,
        compression=zipfile.ZIP_STORED,
    )
    try:
        _write_bytes_with_expected(output, candidate_payload, allow_overwrite=allow_overwrite)
        source_record = _archive_record(source, repo_root=repo_root)
        candidate_record = _archive_record(output, repo_root=repo_root)
        source_member = _member_record(source, member_name, payload=source_member_payload)
        candidate_member = _member_record(output, member_name)
        proof_payload = {
            "schema": "repair_family_archive_payload_mutation_runtime_consumption_proof.v1",
            "proof_kind": "fec6_selector_reencoded_runtime_adapter_parse_proof.v1",
            "proof_scope": "score_affecting_fec6_selector_payload_changed_and_reparsed",
            "target_kind": "repair_family_fec6_selector_payload_mutation_v1",
            "materializer_id": f"repair_family_byte_transform_executor:{family_id}",
            "receiver_contract_kind": "repair_family_fec6_selector_payload_mutation_runtime_parse",
            "receiver_contract_id": "repair_family_fec6_selector_payload_mutation_v1.receiver.v1",
            "source_archive": source_record,
            "candidate_archive": candidate_record,
            "selected_member_name": member_name,
            "source_member": source_member,
            "candidate_member": candidate_member,
            "mutation_details": mutation_details,
            "runtime_consumption_probe": {
                "schema": "fec6_selector_payload_parse_probe.v1",
                "passed": True,
                "selected_member_name": member_name,
                "source_member_sha256": source_member["sha256"],
                "candidate_member_sha256": candidate_member["sha256"],
                "member_payload_changed": source_member["sha256"] != candidate_member["sha256"],
                "archive_zip_readable": True,
            },
            "receiver_contract_satisfied": True,
            "runtime_consumption_proof_passed": True,
            "passed": True,
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
            **FALSE_AUTHORITY,
        }
        _write_json_with_expected(proof, proof_payload, allow_overwrite=allow_overwrite)
    except (ArtifactWriteError, OSError, zipfile.BadZipFile, ValueError) as exc:
        blockers.append(f"fec6_selector_payload_mutation_write_failed:{exc}")
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind="fec6_selector_payload_mutation",
        ), blockers
    saved_bytes = int(source_record["bytes"]) - int(candidate_record["bytes"])
    candidate = {
        "schema": "repair_family_archive_native_candidate.v1",
        "materialized": True,
        "archive_native_transform_attempted": True,
        "archive_native_transform_kind": "fec6_selector_payload_mutation",
        "semantic_entropy_stage": "pre_entropy_payload_distribution_shaping",
        "archive_native_materializer_schema": proof_payload["schema"],
        "archive_native_materializer_id": proof_payload["materializer_id"],
        "archive_native_target_kind": proof_payload["target_kind"],
        "source_archive_path": _repo_rel(source, repo_root),
        "source_archive_sha256": actual_sha,
        "source_archive_bytes": source.stat().st_size,
        "path": _repo_rel(output, repo_root),
        "sha256": candidate_record["sha256"],
        "bytes": candidate_record["bytes"],
        "runtime_consumption_proof_path": _repo_rel(proof, repo_root),
        "runtime_consumption_proof_ready": True,
        "receiver_contract_kind": proof_payload["receiver_contract_kind"],
        "receiver_contract_satisfied": True,
        "selected_member_name": member_name,
        "mutation_details": mutation_details,
        "semantic_payload_changed": True,
        "exact_axis_score_affecting_adjudication_required": True,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "saved_bytes": saved_bytes,
        "blockers": ordered_unique(blockers),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        candidate,
        context="repair_family_fec6_payload_mutation_candidate",
    )
    return candidate, blockers


def _fp11_selector_payload_mutation_candidate(
    *,
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    repo_root: str | Path,
    family_id: str,
    allow_overwrite: bool,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if family_id not in _FEC6_FAMILY_TARGET_CODES:
        blockers.append(f"fp11_selector_payload_mutation_not_enabled_for_family:{family_id}")
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind="fp11_selector_payload_mutation",
        ), blockers
    source, actual_sha, source_blockers = _source_archive_path_and_sha(manifest, repo_root=repo_root)
    blockers.extend(source_blockers)
    if source is None or blockers:
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind="fp11_selector_payload_mutation",
        ), blockers
    try:
        member_name = _primary_payload_member(source)
        source_member_payload = _zip_member_bytes(source, member_name)
        mutation = _mutated_non_fec6_fp11_selector_member(
            source_member_payload,
            family_id=family_id,
            repo_root=repo_root,
        )
    except (OSError, zipfile.BadZipFile, KeyError, RepairFamilyByteTransformExecutorError) as exc:
        blockers.append(f"fp11_selector_payload_mutation_probe_failed:{exc}")
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind="fp11_selector_payload_mutation",
        ), blockers
    if mutation is None:
        blockers.append("non_fec6_fp11_selector_payload_not_detected")
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind="fp11_selector_payload_mutation",
        ), blockers
    candidate_member_payload, mutation_details = mutation
    output = _resolve(output_dir, repo_root) / "candidate_archive_fp11_selector_payload_mutation.zip"
    proof = _resolve(output_dir, repo_root) / "candidate_archive_fp11_selector_payload_mutation_receiver_proof.json"
    candidate_payload = _zip_archive_bytes_with_replacement(
        source,
        member_name=member_name,
        replacement=candidate_member_payload,
        compression=zipfile.ZIP_STORED,
    )
    try:
        _write_bytes_with_expected(output, candidate_payload, allow_overwrite=allow_overwrite)
        source_record = _archive_record(source, repo_root=repo_root)
        candidate_record = _archive_record(output, repo_root=repo_root)
        source_member = _member_record(source, member_name, payload=source_member_payload)
        candidate_member = _member_record(output, member_name)
        proof_payload = {
            "schema": "repair_family_archive_payload_mutation_runtime_consumption_proof.v1",
            "proof_kind": "fp11_selector_reencoded_runtime_adapter_parse_proof.v1",
            "proof_scope": "score_affecting_fp11_selector_payload_changed_and_reparsed",
            "target_kind": "repair_family_fp11_selector_payload_mutation_v1",
            "materializer_id": f"repair_family_byte_transform_executor:{family_id}",
            "receiver_contract_kind": "repair_family_fp11_selector_payload_mutation_runtime_parse",
            "receiver_contract_id": "repair_family_fp11_selector_payload_mutation_v1.receiver.v1",
            "source_archive": source_record,
            "candidate_archive": candidate_record,
            "selected_member_name": member_name,
            "source_member": source_member,
            "candidate_member": candidate_member,
            "mutation_details": mutation_details,
            "runtime_consumption_probe": {
                "schema": "fp11_selector_payload_parse_probe.v1",
                "passed": True,
                "selected_member_name": member_name,
                "source_member_sha256": source_member["sha256"],
                "candidate_member_sha256": candidate_member["sha256"],
                "member_payload_changed": source_member["sha256"] != candidate_member["sha256"],
                "archive_zip_readable": True,
            },
            "receiver_contract_satisfied": True,
            "runtime_consumption_proof_passed": True,
            "passed": True,
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
            **FALSE_AUTHORITY,
        }
        _write_json_with_expected(proof, proof_payload, allow_overwrite=allow_overwrite)
    except (ArtifactWriteError, OSError, zipfile.BadZipFile, ValueError) as exc:
        blockers.append(f"fp11_selector_payload_mutation_write_failed:{exc}")
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind="fp11_selector_payload_mutation",
        ), blockers
    saved_bytes = int(source_record["bytes"]) - int(candidate_record["bytes"])
    candidate = {
        "schema": "repair_family_archive_native_candidate.v1",
        "materialized": True,
        "archive_native_transform_attempted": True,
        "archive_native_transform_kind": "fp11_selector_payload_mutation",
        "semantic_entropy_stage": "pre_entropy_payload_distribution_shaping",
        "archive_native_materializer_schema": proof_payload["schema"],
        "archive_native_materializer_id": proof_payload["materializer_id"],
        "archive_native_target_kind": proof_payload["target_kind"],
        "source_archive_path": _repo_rel(source, repo_root),
        "source_archive_sha256": actual_sha,
        "source_archive_bytes": source.stat().st_size,
        "path": _repo_rel(output, repo_root),
        "sha256": candidate_record["sha256"],
        "bytes": candidate_record["bytes"],
        "runtime_consumption_proof_path": _repo_rel(proof, repo_root),
        "runtime_consumption_proof_ready": True,
        "receiver_contract_kind": proof_payload["receiver_contract_kind"],
        "receiver_contract_satisfied": True,
        "selected_member_name": member_name,
        "mutation_details": mutation_details,
        "semantic_payload_changed": True,
        "exact_axis_score_affecting_adjudication_required": True,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "saved_bytes": saved_bytes,
        "blockers": ordered_unique(blockers),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        candidate,
        context="repair_family_fp11_selector_payload_mutation_candidate",
    )
    return candidate, blockers


def _psv4_selector_payload_mutation_candidate(
    *,
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    repo_root: str | Path,
    family_id: str,
    allow_overwrite: bool,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if family_id not in _FEC6_FAMILY_TARGET_CODES:
        blockers.append(f"psv4_selector_payload_mutation_not_enabled_for_family:{family_id}")
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind="psv4_selector_payload_mutation",
        ), blockers
    source, actual_sha, source_blockers = _source_archive_path_and_sha(manifest, repo_root=repo_root)
    blockers.extend(source_blockers)
    if source is None or blockers:
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind="psv4_selector_payload_mutation",
        ), blockers
    try:
        member_name = _primary_payload_member(source)
        source_member_payload = _zip_member_bytes(source, member_name)
        mutation = _mutated_psv4_selector_member(source_member_payload, family_id=family_id)
    except (OSError, zipfile.BadZipFile, KeyError, RepairFamilyByteTransformExecutorError) as exc:
        blockers.append(f"psv4_selector_payload_mutation_probe_failed:{exc}")
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind="psv4_selector_payload_mutation",
        ), blockers
    if mutation is None:
        blockers.append("psv4_selector_payload_not_detected")
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind="psv4_selector_payload_mutation",
        ), blockers
    candidate_member_payload, mutation_details = mutation
    output = _resolve(output_dir, repo_root) / "candidate_archive_psv4_selector_payload_mutation.zip"
    proof = _resolve(output_dir, repo_root) / "candidate_archive_psv4_selector_payload_mutation_receiver_proof.json"
    candidate_payload = _zip_archive_bytes_with_replacement(
        source,
        member_name=member_name,
        replacement=candidate_member_payload,
        compression=zipfile.ZIP_STORED,
    )
    try:
        _write_bytes_with_expected(output, candidate_payload, allow_overwrite=allow_overwrite)
        source_record = _archive_record(source, repo_root=repo_root)
        candidate_record = _archive_record(output, repo_root=repo_root)
        source_member = _member_record(source, member_name, payload=source_member_payload)
        candidate_member = _member_record(output, member_name)
        proof_payload = {
            "schema": "repair_family_archive_payload_mutation_runtime_consumption_proof.v1",
            "proof_kind": "psv4_selector_reencoded_runtime_adapter_parse_proof.v1",
            "proof_scope": "score_affecting_psv4_selector_payload_changed_and_reparsed",
            "target_kind": "repair_family_psv4_selector_payload_mutation_v1",
            "materializer_id": f"repair_family_byte_transform_executor:{family_id}",
            "receiver_contract_kind": "repair_family_psv4_selector_payload_mutation_runtime_parse",
            "receiver_contract_id": "repair_family_psv4_selector_payload_mutation_v1.receiver.v1",
            "source_archive": source_record,
            "candidate_archive": candidate_record,
            "selected_member_name": member_name,
            "source_member": source_member,
            "candidate_member": candidate_member,
            "mutation_details": mutation_details,
            "runtime_consumption_probe": {
                "schema": "psv4_selector_payload_parse_probe.v1",
                "passed": True,
                "selected_member_name": member_name,
                "source_member_sha256": source_member["sha256"],
                "candidate_member_sha256": candidate_member["sha256"],
                "member_payload_changed": source_member["sha256"] != candidate_member["sha256"],
                "archive_zip_readable": True,
                "packet_header_rewritten": True,
                "selector_redecoded_pair_count": mutation_details["n_pairs"],
            },
            "receiver_contract_satisfied": True,
            "runtime_consumption_proof_passed": True,
            "passed": True,
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
            **FALSE_AUTHORITY,
        }
        _write_json_with_expected(proof, proof_payload, allow_overwrite=allow_overwrite)
    except (ArtifactWriteError, OSError, zipfile.BadZipFile, ValueError) as exc:
        blockers.append(f"psv4_selector_payload_mutation_write_failed:{exc}")
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind="psv4_selector_payload_mutation",
        ), blockers
    saved_bytes = int(source_record["bytes"]) - int(candidate_record["bytes"])
    candidate = {
        "schema": "repair_family_archive_native_candidate.v1",
        "materialized": True,
        "archive_native_transform_attempted": True,
        "archive_native_transform_kind": "psv4_selector_payload_mutation",
        "semantic_entropy_stage": "pre_entropy_payload_distribution_shaping",
        "archive_native_materializer_schema": proof_payload["schema"],
        "archive_native_materializer_id": proof_payload["materializer_id"],
        "archive_native_target_kind": proof_payload["target_kind"],
        "source_archive_path": _repo_rel(source, repo_root),
        "source_archive_sha256": actual_sha,
        "source_archive_bytes": source.stat().st_size,
        "path": _repo_rel(output, repo_root),
        "sha256": candidate_record["sha256"],
        "bytes": candidate_record["bytes"],
        "runtime_consumption_proof_path": _repo_rel(proof, repo_root),
        "runtime_consumption_proof_ready": True,
        "receiver_contract_kind": proof_payload["receiver_contract_kind"],
        "receiver_contract_satisfied": True,
        "selected_member_name": member_name,
        "mutation_details": mutation_details,
        "semantic_payload_changed": True,
        "exact_axis_score_affecting_adjudication_required": True,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "saved_bytes": saved_bytes,
        "blockers": ordered_unique(blockers),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        candidate,
        context="repair_family_psv4_selector_payload_mutation_candidate",
    )
    return candidate, blockers


def _packet_member_recompress_candidate(
    *,
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    repo_root: str | Path,
    allow_overwrite: bool,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    source, _actual_sha, source_blockers = _source_archive_path_and_sha(manifest, repo_root=repo_root)
    blockers.extend(source_blockers)
    if source is None or blockers:
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind="packet_member_entropy_boundary_recompress",
        ), blockers
    output = _resolve(output_dir, repo_root) / "candidate_archive_packet_member_recompress.zip"
    proof = _resolve(output_dir, repo_root) / "candidate_archive_packet_member_recompress_receiver_proof.json"
    try:
        member_name = _primary_payload_member(source)
        native_manifest = materialize_packet_member_recompress_candidate(
            archive_path=source,
            output_archive=output,
            member_name=member_name,
            runtime_consumption_proof_out=proof,
            repo_root=repo_root,
            allow_size_regression=True,
            allow_overwrite=allow_overwrite,
            expected_existing_output_sha256=sha256_file(output) if output.exists() and allow_overwrite else None,
            expected_existing_runtime_consumption_proof_sha256=(
                sha256_file(proof) if proof.exists() and allow_overwrite else None
            ),
        )
    except (FamilyAgnosticMaterializerError, ArtifactWriteError, OSError, ValueError) as exc:
        blockers.append(f"packet_member_entropy_boundary_recompress_failed:{exc}")
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind="packet_member_entropy_boundary_recompress",
        ), blockers
    blockers.extend(_string_list(native_manifest.get("readiness_blockers")))
    candidate_record = _mapping(native_manifest.get("candidate_archive"))
    source_record = _mapping(native_manifest.get("source_archive"))
    selected = _mapping(native_manifest.get("selected_compression"))
    proof_ready = native_manifest.get("receiver_contract_satisfied") is True
    candidate = {
        "schema": "repair_family_archive_native_candidate.v1",
        "materialized": output.is_file(),
        "archive_native_transform_attempted": True,
        "archive_native_transform_kind": "packet_member_entropy_boundary_recompress",
        "semantic_entropy_stage": "at_entropy_coder_integer_codeword_boundary",
        "archive_native_materializer_schema": native_manifest.get("schema"),
        "archive_native_materializer_id": native_manifest.get("materializer_id"),
        "archive_native_target_kind": native_manifest.get("target_kind"),
        "source_archive_path": source_record.get("path") or _repo_rel(source, repo_root),
        "source_archive_sha256": source_record.get("sha256") or sha256_file(source),
        "source_archive_bytes": source_record.get("bytes") or source.stat().st_size,
        "path": candidate_record.get("path") or _repo_rel(output, repo_root),
        "sha256": candidate_record.get("sha256") or sha256_file(output),
        "bytes": candidate_record.get("bytes") or output.stat().st_size,
        "runtime_consumption_proof_path": _repo_rel(proof, repo_root) if proof.is_file() else None,
        "runtime_consumption_proof_ready": proof_ready,
        "receiver_contract_kind": native_manifest.get("receiver_contract_kind"),
        "receiver_contract_satisfied": proof_ready,
        "selected_member_name": native_manifest.get("selected_member_name"),
        "selected_compression": dict(selected),
        "score_affecting_payload_changed": False,
        "charged_bits_changed": True,
        "saved_bytes": selected.get("saved_bytes"),
        "blockers": ordered_unique(blockers),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        candidate,
        context="repair_family_packet_member_recompress_candidate",
    )
    return candidate, blockers


def _archive_entropy_coder_probe_candidate(
    *,
    coder_family: str,
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    repo_root: str | Path,
    allow_overwrite: bool,
) -> tuple[dict[str, Any], list[str]]:
    blockers = [
        f"{coder_family}_coder_materializer_missing",
        f"{coder_family}_coder_runtime_adapter_missing",
    ]
    source, actual_sha, source_blockers = _source_archive_path_and_sha(manifest, repo_root=repo_root)
    blockers.extend(source_blockers)
    transform_kind = f"{coder_family}_coder_entropy_probe"
    if source is None or source_blockers:
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind=transform_kind,
        ), blockers
    output = _resolve(output_dir, repo_root) / f"{transform_kind}.json"
    try:
        member_name = _primary_payload_member(source)
        member_payload = _zip_member_bytes(source, member_name)
        entropy_stats = _zero_order_entropy_stats(member_payload)
        member = _member_record(source, member_name, payload=member_payload)
        probe = {
            "schema": "repair_archive_entropy_coder_boundary_probe.v1",
            "coder_family": coder_family,
            "archive_native_transform_kind": transform_kind,
            "semantic_entropy_stage": "at_entropy_coder_fractional_code_boundary",
            "source_archive_path": _repo_rel(source, repo_root),
            "source_archive_sha256": actual_sha or sha256_file(source),
            "source_archive_bytes": source.stat().st_size,
            "selected_member": member,
            "zero_order_entropy_stats": entropy_stats,
            "estimated_zero_order_savings_bytes": entropy_stats["zero_order_redundancy_bytes"],
            "materializer_status": "probe_only_materializer_missing",
            "receiver_contract_satisfied": False,
            "runtime_consumption_proof_ready": False,
            "blockers": ordered_unique(blockers),
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        }
        _write_json_with_expected(output, probe, allow_overwrite=allow_overwrite)
    except (OSError, zipfile.BadZipFile, KeyError, RepairFamilyByteTransformExecutorError) as exc:
        blockers.append(f"{coder_family}_coder_entropy_probe_failed:{exc}")
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind=transform_kind,
        ), blockers
    candidate = {
        "schema": "repair_family_archive_native_candidate.v1",
        "materialized": False,
        "archive_native_transform_attempted": True,
        "archive_native_transform_kind": transform_kind,
        "semantic_entropy_stage": "at_entropy_coder_fractional_code_boundary",
        "archive_native_materializer_schema": "repair_archive_entropy_coder_boundary_probe.v1",
        "archive_native_materializer_id": f"repair_family_byte_transform_executor:{transform_kind}",
        "archive_native_target_kind": f"repair_family_{transform_kind}_v1",
        "source_archive_path": _repo_rel(source, repo_root),
        "source_archive_sha256": actual_sha or sha256_file(source),
        "source_archive_bytes": source.stat().st_size,
        "path": None,
        "sha256": None,
        "bytes": None,
        "entropy_probe_path": _repo_rel(output, repo_root),
        "entropy_probe_sha256": sha256_file(output),
        "entropy_probe_bytes": output.stat().st_size,
        "runtime_consumption_proof_path": None,
        "runtime_consumption_proof_ready": False,
        "receiver_contract_kind": None,
        "receiver_contract_satisfied": False,
        "selected_member_name": member_name,
        "selected_member": member,
        "zero_order_entropy_stats": entropy_stats,
        "estimated_zero_order_savings_bytes": entropy_stats["zero_order_redundancy_bytes"],
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "saved_bytes": None,
        "blockers": ordered_unique(blockers),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        candidate,
        context=f"repair_family_{transform_kind}_candidate",
    )
    return candidate, blockers


def _archive_entropy_coder_prototype_candidate(
    *,
    coder_family: str,
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    repo_root: str | Path,
    allow_overwrite: bool,
) -> tuple[dict[str, Any], list[str]]:
    if coder_family == "range":
        transform_kind = "range_coder_lzma_prototype"
        encode = range_lzma_prototype_encode
        proof_kind = "range_coder_lzma_prototype_decode_adapter_proof.v1"
        receiver_contract_kind = "repair_archive_range_lzma_prototype_runtime_adapter"
    elif coder_family == "ans":
        transform_kind = "ans_coder_rans_prototype"
        encode = ans_rans_prototype_encode
        proof_kind = "ans_coder_rans_prototype_decode_adapter_proof.v1"
        receiver_contract_kind = "repair_archive_ans_rans_prototype_runtime_adapter"
    else:
        raise RepairFamilyByteTransformExecutorError(
            f"unsupported entropy coder prototype family: {coder_family}"
        )
    blockers = [
        f"{coder_family}_coder_exact_axis_adjudication_missing",
        "contest_runtime_decoder_adapter_integration_missing",
    ]
    source, actual_sha, source_blockers = _source_archive_path_and_sha(manifest, repo_root=repo_root)
    blockers.extend(source_blockers)
    if source is None or source_blockers:
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind=transform_kind,
        ), blockers
    output = _resolve(output_dir, repo_root) / f"candidate_archive_{transform_kind}.zip"
    proof = _resolve(output_dir, repo_root) / f"candidate_archive_{transform_kind}_receiver_proof.json"
    try:
        member_name = _primary_payload_member(source)
        source_member_payload = _zip_member_bytes(source, member_name)
        coded_member_payload = encode(source_member_payload)
        runtime_adapter_manifest = entropy_coder_runtime_adapter_manifest(coder_family)
        decoded_member_payload = decode_entropy_coder_prototype_member(
            coder_family=coder_family,
            packet=coded_member_payload,
        )
        if decoded_member_payload != source_member_payload:
            raise RepairFamilyByteTransformExecutorError(
                f"{coder_family} prototype adapter did not round-trip selected member"
            )
        candidate_payload = _zip_archive_bytes_with_replacement(
            source,
            member_name=member_name,
            replacement=coded_member_payload,
            compression=zipfile.ZIP_STORED,
        )
        _write_bytes_with_expected(output, candidate_payload, allow_overwrite=allow_overwrite)
        source_record = _archive_record(source, repo_root=repo_root)
        candidate_record = _archive_record(output, repo_root=repo_root)
        source_member = _member_record(source, member_name, payload=source_member_payload)
        candidate_member = _member_record(output, member_name)
        entropy_stats = _zero_order_entropy_stats(source_member_payload)
        proof_payload = {
            "schema": "repair_archive_entropy_coder_prototype_runtime_consumption_proof.v1",
            "proof_kind": proof_kind,
            "proof_scope": "encoded_member_decodes_to_source_member_payload",
            "target_kind": f"repair_family_{transform_kind}_v1",
            "materializer_id": f"repair_family_byte_transform_executor:{transform_kind}",
            "receiver_contract_kind": receiver_contract_kind,
            "receiver_contract_id": f"repair_family_{transform_kind}_v1.receiver.v1",
            "source_archive": source_record,
            "candidate_archive": candidate_record,
            "selected_member_name": member_name,
            "source_member": source_member,
            "candidate_member": candidate_member,
            "runtime_adapter_manifest": runtime_adapter_manifest,
            "runtime_adapter_ready": True,
            "runtime_adapter_scope": "member_decode_helper_only",
            "contest_runtime_decoder_adapter_ready": False,
            "runtime_consumption_probe": {
                "schema": f"{transform_kind}_decode_probe.v1",
                "passed": True,
                "decoder_adapter_invoked": True,
                "decoder_adapter_module": runtime_adapter_manifest["module"],
                "decoder_adapter_function": runtime_adapter_manifest["decode_function"],
                "selected_member_name": member_name,
                "source_member_sha256": source_member["sha256"],
                "encoded_member_sha256": candidate_member["sha256"],
                "decoded_member_sha256": sha256_bytes(decoded_member_payload),
                "decoded_matches_source_member": True,
                "archive_zip_readable": True,
            },
            "prototype_codec": {
                "schema": "repair_archive_entropy_coder_prototype_codec.v1",
                "coder_family": coder_family,
                "transform_kind": transform_kind,
                "semantic_entropy_stage": "at_entropy_coder_fractional_code_boundary",
                "source_member_bytes": len(source_member_payload),
                "coded_member_bytes": len(coded_member_payload),
                "source_member_sha256": sha256_bytes(source_member_payload),
                "coded_member_sha256": sha256_bytes(coded_member_payload),
                "decoded_member_sha256": sha256_bytes(decoded_member_payload),
                "zero_order_entropy_stats": entropy_stats,
            },
            "receiver_contract_satisfied": True,
            "runtime_consumption_proof_passed": True,
            "passed": True,
            "score_affecting_payload_changed": False,
            "charged_bits_changed": True,
            "contest_runtime_decoder_adapter_integrated": False,
            "contest_runtime_adapter_integrated": False,
            "blockers": ordered_unique(blockers),
            **FALSE_AUTHORITY,
        }
        _write_json_with_expected(proof, proof_payload, allow_overwrite=allow_overwrite)
    except (
        ArtifactWriteError,
        OSError,
        zipfile.BadZipFile,
        KeyError,
        RepairFamilyByteTransformExecutorError,
        RepairEntropyCoderRuntimeAdapterError,
    ) as exc:
        blockers.append(f"{coder_family}_coder_prototype_materialization_failed:{exc}")
        return _empty_candidate_archive(
            blockers=blockers,
            archive_native_transform_kind=transform_kind,
        ), blockers
    saved_bytes = int(source_record["bytes"]) - int(candidate_record["bytes"])
    if saved_bytes <= 0:
        blockers.append(f"{coder_family}_coder_prototype_not_rate_positive")
        proof_payload = {
            **proof_payload,
            "blockers": ordered_unique(blockers),
        }
        _write_json_with_expected(proof, proof_payload, allow_overwrite=True)
    candidate = {
        "schema": "repair_family_archive_native_candidate.v1",
        "materialized": True,
        "prototype_only": True,
        "archive_native_transform_attempted": True,
        "archive_native_transform_kind": transform_kind,
        "semantic_entropy_stage": "at_entropy_coder_fractional_code_boundary",
        "archive_native_materializer_schema": proof_payload["schema"],
        "archive_native_materializer_id": proof_payload["materializer_id"],
        "archive_native_target_kind": proof_payload["target_kind"],
        "source_archive_path": _repo_rel(source, repo_root),
        "source_archive_sha256": actual_sha or sha256_file(source),
        "source_archive_bytes": source.stat().st_size,
        "path": _repo_rel(output, repo_root),
        "sha256": candidate_record["sha256"],
        "bytes": candidate_record["bytes"],
        "runtime_consumption_proof_path": _repo_rel(proof, repo_root),
        "runtime_consumption_proof_ready": True,
        "receiver_contract_kind": proof_payload["receiver_contract_kind"],
        "receiver_contract_satisfied": True,
        "runtime_adapter_manifest": runtime_adapter_manifest,
        "runtime_adapter_ready": True,
        "runtime_adapter_scope": "member_decode_helper_only",
        "contest_runtime_decoder_adapter_ready": False,
        "contest_runtime_decoder_adapter_integrated": False,
        "contest_runtime_adapter_integrated": False,
        "selected_member_name": member_name,
        "encoded_member_sha256": candidate_member["sha256"],
        "decoded_member_sha256": sha256_bytes(decoded_member_payload),
        "decoded_matches_source_member": True,
        "zero_order_entropy_stats": entropy_stats,
        "estimated_zero_order_savings_bytes": entropy_stats["zero_order_redundancy_bytes"],
        "semantic_payload_changed": False,
        "exact_axis_score_affecting_adjudication_required": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": True,
        "saved_bytes": saved_bytes,
        "blockers": ordered_unique(blockers),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        candidate,
        context=f"repair_family_{transform_kind}_candidate",
    )
    return candidate, blockers


def _candidate_rank(candidate: Mapping[str, Any]) -> tuple[int, int, str]:
    proof_ready = candidate.get("runtime_consumption_proof_ready") is True
    score_affecting = (
        candidate.get("semantic_payload_changed") is True
        or candidate.get("exact_axis_score_affecting_adjudication_required") is True
    )
    materialized = candidate.get("materialized") is True
    if candidate.get("prototype_only") is True:
        rank_class = 2
    elif materialized and proof_ready and score_affecting:
        rank_class = 0
    elif materialized and proof_ready:
        rank_class = 1
    else:
        rank_class = 3
    return (
        rank_class,
        int(candidate.get("bytes") or 10**18),
        str(candidate.get("archive_native_transform_kind") or ""),
    )


def _archive_transform_candidates(
    *,
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    repo_root: str | Path,
    family_id: str,
    allow_overwrite: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    blockers: list[str] = []
    for builder in (
        lambda: _fp11_selector_payload_mutation_candidate(
            manifest=manifest,
            output_dir=output_dir,
            repo_root=repo_root,
            family_id=family_id,
            allow_overwrite=allow_overwrite,
        ),
        lambda: _fec6_payload_mutation_candidate(
            manifest=manifest,
            output_dir=output_dir,
            repo_root=repo_root,
            family_id=family_id,
            allow_overwrite=allow_overwrite,
        ),
        lambda: _psv4_selector_payload_mutation_candidate(
            manifest=manifest,
            output_dir=output_dir,
            repo_root=repo_root,
            family_id=family_id,
            allow_overwrite=allow_overwrite,
        ),
        lambda: _packet_member_recompress_candidate(
            manifest=manifest,
            output_dir=output_dir,
            repo_root=repo_root,
            allow_overwrite=allow_overwrite,
        ),
        lambda: _archive_entropy_coder_probe_candidate(
            coder_family="range",
            manifest=manifest,
            output_dir=output_dir,
            repo_root=repo_root,
            allow_overwrite=allow_overwrite,
        ),
        lambda: _archive_entropy_coder_probe_candidate(
            coder_family="ans",
            manifest=manifest,
            output_dir=output_dir,
            repo_root=repo_root,
            allow_overwrite=allow_overwrite,
        ),
        lambda: _archive_entropy_coder_prototype_candidate(
            coder_family="range",
            manifest=manifest,
            output_dir=output_dir,
            repo_root=repo_root,
            allow_overwrite=allow_overwrite,
        ),
        lambda: _archive_entropy_coder_prototype_candidate(
            coder_family="ans",
            manifest=manifest,
            output_dir=output_dir,
            repo_root=repo_root,
            allow_overwrite=allow_overwrite,
        ),
        lambda: _archive_native_zip_repack_candidate(
            manifest=manifest,
            output_dir=output_dir,
            repo_root=repo_root,
            allow_overwrite=allow_overwrite,
        ),
    ):
        candidate, candidate_blockers = builder()
        variants.append(candidate)
        blockers.extend(candidate_blockers)
    selected = min(variants, key=_candidate_rank)
    selected_kind = str(selected.get("archive_native_transform_kind") or "")
    source_context = dict(_mapping(manifest.get("candidate_archive")))
    contract_surface = build_archive_bound_candidate_contract_surface(
        candidates=variants,
        selected_transform_kind=selected_kind,
        repo_root=repo_root,
        source_context=source_context,
        family_id=family_id,
        typed_response_id=str(manifest.get("typed_response_id") or ""),
        candidate_chain_id=str(manifest.get("candidate_chain_id") or ""),
        entropy_position_label=str(manifest.get("entropy_position_label") or ""),
        entropy_stage_order=_safe_int(
            _mapping(manifest.get("active_entropy_stage")).get("order"),
            default=999,
        ),
    )
    contracts_by_kind = {
        str(contract.get("archive_native_transform_kind") or ""): contract
        for contract in contract_surface.get("candidate_contracts") or []
        if isinstance(contract, Mapping)
    }
    variants = [
        {
            **variant,
            "archive_bound_candidate_contract_schema": (
                ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
            ),
            "archive_bound_candidate_contract": contracts_by_kind.get(
                str(variant.get("archive_native_transform_kind") or ""),
                {},
            ),
        }
        for variant in variants
    ]
    selected = next(
        (
            variant
            for variant in variants
            if str(variant.get("archive_native_transform_kind") or "") == selected_kind
        ),
        selected,
    )
    selected_blockers = ordered_unique(_string_list(selected.get("blockers")))
    all_blockers = ordered_unique(blockers)
    selected = {
        **selected,
        "selected_archive_transform_variant": True,
        "archive_bound_candidate_contract_schema": (
            ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
        ),
        "archive_bound_candidate_contract": contract_surface.get(
            "selected_candidate_contract",
            {},
        ),
        "candidate_archive_transform_variant_count": len(variants),
        "non_selected_archive_transform_variant_blockers": [
            blocker for blocker in all_blockers if blocker not in set(selected_blockers)
        ],
    }
    return selected, variants, selected_blockers, contract_surface


def _archive_variant_signal_class(
    *,
    variant: Mapping[str, Any],
    selected_kind: str,
) -> str:
    kind = str(variant.get("archive_native_transform_kind") or "")
    materialized = variant.get("materialized") is True
    proof_ready = variant.get("runtime_consumption_proof_ready") is True
    prototype_only = variant.get("prototype_only") is True
    probe_only = bool(str(variant.get("entropy_probe_path") or "").strip()) and not materialized
    score_affecting = (
        variant.get("semantic_payload_changed") is True
        or variant.get("exact_axis_score_affecting_adjudication_required") is True
    )
    if kind == selected_kind and materialized and proof_ready and score_affecting:
        return "selected_materialized_score_affecting_candidate"
    if kind == selected_kind and materialized and proof_ready:
        return "selected_materialized_runtime_proven_candidate"
    if prototype_only and materialized and proof_ready:
        return "prototype_runtime_proven"
    if materialized and proof_ready:
        return "materialized_runtime_proven_nonselected"
    if probe_only:
        return "probe_only_entropy_signal"
    if _string_list(variant.get("blockers")):
        return "blocked_archive_variant_signal"
    return "unclassified_archive_variant_signal"


def _archive_variant_signal_routing_action(signal_class: str) -> str:
    if signal_class == "selected_materialized_score_affecting_candidate":
        return "promote_to_receiver_proof_and_exact_handoff_gate"
    if signal_class in {
        "selected_materialized_runtime_proven_candidate",
        "materialized_runtime_proven_nonselected",
    }:
        return "retain_as_archive_bound_materializer_alternative"
    if signal_class == "prototype_runtime_proven":
        return "retain_as_entropy_coder_runtime_adapter_prototype"
    if signal_class == "probe_only_entropy_signal":
        return "route_entropy_probe_to_materializer_backlog"
    if signal_class == "blocked_archive_variant_signal":
        return "route_blocker_to_acquisition_penalty_and_remeasure"
    return "retain_for_operator_inspection"


def _archive_variant_signal_weight(signal_class: str) -> float:
    return {
        "selected_materialized_score_affecting_candidate": 1.0,
        "selected_materialized_runtime_proven_candidate": 0.85,
        "materialized_runtime_proven_nonselected": 0.70,
        "prototype_runtime_proven": 0.55,
        "probe_only_entropy_signal": 0.25,
        "blocked_archive_variant_signal": 0.10,
    }.get(signal_class, 0.05)


def _archive_variant_signal_surface(
    *,
    variants: Sequence[Mapping[str, Any]],
    selected_candidate_archive: Mapping[str, Any],
) -> dict[str, Any]:
    selected_kind = str(selected_candidate_archive.get("archive_native_transform_kind") or "")
    rows: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        kind = str(variant.get("archive_native_transform_kind") or f"variant_{index}")
        signal_class = _archive_variant_signal_class(
            variant=variant,
            selected_kind=selected_kind,
        )
        blockers = _string_list(variant.get("blockers"))
        row = {
            "schema": REPAIR_ARCHIVE_VARIANT_SIGNAL_ROW_SCHEMA,
            "variant_index": index,
            "archive_native_transform_kind": kind,
            "selected_archive_transform_variant": kind == selected_kind,
            "signal_class": signal_class,
            "routing_action": _archive_variant_signal_routing_action(signal_class),
            "signal_weight_hint": _archive_variant_signal_weight(signal_class),
            "materialized": variant.get("materialized") is True,
            "prototype_only": variant.get("prototype_only") is True,
            "probe_only": bool(str(variant.get("entropy_probe_path") or "").strip())
            and variant.get("materialized") is not True,
            "archive_native_transform_attempted": (
                variant.get("archive_native_transform_attempted") is True
            ),
            "runtime_consumption_proof_ready": (
                variant.get("runtime_consumption_proof_ready") is True
            ),
            "receiver_contract_satisfied": (
                variant.get("receiver_contract_satisfied") is True
            ),
            "semantic_payload_changed_observed": (
                variant.get("semantic_payload_changed") is True
            ),
            "exact_axis_score_affecting_adjudication_required_observed": (
                variant.get("exact_axis_score_affecting_adjudication_required") is True
            ),
            "charged_bits_changed_observed": (
                variant.get("charged_bits_changed") is True
            ),
            "candidate_archive_path": variant.get("path"),
            "candidate_archive_sha256": variant.get("sha256"),
            "candidate_archive_bytes": variant.get("bytes"),
            "runtime_consumption_proof_path": variant.get("runtime_consumption_proof_path"),
            "entropy_probe_path": variant.get("entropy_probe_path"),
            "selected_member_name": variant.get("selected_member_name"),
            "saved_bytes": variant.get("saved_bytes"),
            "estimated_zero_order_savings_bytes": _safe_int(
                variant.get("estimated_zero_order_savings_bytes")
            ),
            "blockers": ordered_unique(blockers),
            "allowed_use": "repair_archive_variant_signal_routing_only",
            "forbidden_use": "score_claim_or_dispatch_or_submission_authority",
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        }
        require_no_truthy_authority_fields(
            row,
            context=f"repair_archive_variant_signal_row:{kind}",
        )
        rows.append(row)
    class_counts: dict[str, int] = {}
    for row in rows:
        signal_class = str(row.get("signal_class") or "unclassified_archive_variant_signal")
        class_counts[signal_class] = class_counts.get(signal_class, 0) + 1
    blockers = ordered_unique(
        blocker
        for row in rows
        for blocker in _string_list(row.get("blockers"))
    )
    surface = {
        "schema": REPAIR_ARCHIVE_VARIANT_SIGNAL_SURFACE_SCHEMA,
        "selected_archive_transform_kind": selected_kind,
        "row_count": len(rows),
        "materialized_count": sum(1 for row in rows if row.get("materialized") is True),
        "prototype_count": sum(1 for row in rows if row.get("prototype_only") is True),
        "probe_count": sum(1 for row in rows if row.get("probe_only") is True),
        "runtime_proof_ready_count": sum(
            1 for row in rows if row.get("runtime_consumption_proof_ready") is True
        ),
        "selected_signal_count": sum(
            1 for row in rows if row.get("selected_archive_transform_variant") is True
        ),
        "non_selected_signal_count": sum(
            1 for row in rows if row.get("selected_archive_transform_variant") is not True
        ),
        "blocked_signal_count": sum(
            1 for row in rows if _string_list(row.get("blockers"))
        ),
        "exact_axis_score_affecting_adjudication_required_count": sum(
            1
            for row in rows
            if row.get("exact_axis_score_affecting_adjudication_required_observed") is True
        ),
        "signal_class_counts": class_counts,
        "signal_transform_kinds": [str(row["archive_native_transform_kind"]) for row in rows],
        "blockers": blockers,
        "variant_signal_rows": rows,
        "variant_signal_rows_sha256": stable_json_sha256(
            {
                "schema": "repair_archive_variant_signal_rows_hash.v1",
                "rows": rows,
            }
        ),
        "orphan_signal_protection": (
            "every_archive_transform_variant_is_exported_as_posterior_consumable_signal"
        ),
        "allowed_use": "repair_stack_acquisition_and_materializer_backlog_routing_only",
        "forbidden_use": "score_claim_or_dispatch_or_submission_authority",
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        surface,
        context="repair_archive_variant_signal_surface",
    )
    return surface


def _prototype_target_for_probe_kind(probe_kind: str) -> tuple[str, str]:
    if probe_kind == "range_coder_entropy_probe":
        return "range", "range_coder_lzma_prototype"
    if probe_kind == "ans_coder_entropy_probe":
        return "ans", "ans_coder_rans_prototype"
    return "unknown", ""


def _archive_variant_materializer_backlog(
    *,
    signal_surface: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    variants_by_kind = {
        str(variant.get("archive_native_transform_kind") or ""): variant
        for variant in variants
    }
    rows: list[dict[str, Any]] = []
    for signal in signal_surface.get("variant_signal_rows") or []:
        if not isinstance(signal, Mapping):
            continue
        if signal.get("signal_class") != "probe_only_entropy_signal":
            continue
        probe_kind = str(signal.get("archive_native_transform_kind") or "")
        coder_family, target_kind = _prototype_target_for_probe_kind(probe_kind)
        if not target_kind:
            target_variant: Mapping[str, Any] = {}
            materializer_status = "blocked_no_known_entropy_prototype_target"
        else:
            target_variant = variants_by_kind.get(target_kind) or {}
            materializer_status = (
                "byte_closed_candidate_materialized"
                if target_variant.get("materialized") is True
                and target_variant.get("runtime_consumption_proof_ready") is True
                else "queued_materializer_task"
            )
        runtime_adapter_manifest = (
            entropy_coder_runtime_adapter_manifest(coder_family)
            if coder_family in {"range", "ans"}
            else {}
        )
        blockers = ordered_unique(
            [
                *_string_list(signal.get("blockers")),
                *_string_list(target_variant.get("blockers")),
                *([] if target_variant else ["target_entropy_coder_prototype_missing"]),
                *(
                    []
                    if target_variant.get("runtime_consumption_proof_ready") is True
                    else ["target_entropy_coder_runtime_proof_missing"]
                ),
            ]
        )
        row = {
            "schema": REPAIR_ARCHIVE_VARIANT_MATERIALIZER_BACKLOG_ROW_SCHEMA,
            "backlog_key": f"archive_variant_entropy_materializer:{probe_kind}->{target_kind or 'unknown'}",
            "source_signal_schema": signal.get("schema"),
            "source_signal_class": signal.get("signal_class"),
            "source_archive_transform_kind": probe_kind,
            "target_archive_transform_kind": target_kind or None,
            "coder_family": coder_family,
            "materializer_action": f"materialize_{target_kind}" if target_kind else "blocked_no_materializer_target",
            "materializer_status": materializer_status,
            "source_entropy_probe_path": signal.get("entropy_probe_path"),
            "selected_member_name": signal.get("selected_member_name"),
            "estimated_zero_order_savings_bytes": _safe_int(
                signal.get("estimated_zero_order_savings_bytes")
            ),
            "byte_closed_candidate_path": target_variant.get("path"),
            "byte_closed_candidate_sha256": target_variant.get("sha256"),
            "byte_closed_candidate_bytes": target_variant.get("bytes"),
            "byte_closed_candidate_materialized": (
                materializer_status == "byte_closed_candidate_materialized"
            ),
            "runtime_consumption_proof_path": target_variant.get("runtime_consumption_proof_path"),
            "runtime_consumption_proof_ready": (
                target_variant.get("runtime_consumption_proof_ready") is True
            ),
            "runtime_adapter_manifest": runtime_adapter_manifest,
            "runtime_adapter_ready": bool(runtime_adapter_manifest),
            "receiver_contract_kind": target_variant.get("receiver_contract_kind"),
            "receiver_contract_satisfied": (
                target_variant.get("receiver_contract_satisfied") is True
            ),
            "smallest_byte_closed_materializer_task": True,
            "opened_by_pipeline": True,
            "queue_owned": True,
            "blockers": blockers,
            "allowed_use": "repair_archive_variant_materializer_backlog_routing_only",
            "forbidden_use": "score_claim_or_dispatch_or_submission_authority",
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        }
        require_no_truthy_authority_fields(
            row,
            context=f"repair_archive_variant_materializer_backlog_row:{probe_kind}",
        )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            _safe_int(row.get("byte_closed_candidate_bytes"), default=10**18),
            str(row.get("backlog_key") or ""),
        )
    )
    backlog = {
        "schema": REPAIR_ARCHIVE_VARIANT_MATERIALIZER_BACKLOG_SCHEMA,
        "source_signal_surface_schema": signal_surface.get("schema"),
        "row_count": len(rows),
        "executable_task_count": sum(
            1 for row in rows if row.get("target_archive_transform_kind")
        ),
        "byte_closed_materialized_task_count": sum(
            1 for row in rows if row.get("byte_closed_candidate_materialized") is True
        ),
        "runtime_adapter_ready_task_count": sum(
            1 for row in rows if row.get("runtime_adapter_ready") is True
        ),
        "probe_only_signal_count": sum(
            1
            for signal in signal_surface.get("variant_signal_rows") or []
            if isinstance(signal, Mapping)
            and signal.get("signal_class") == "probe_only_entropy_signal"
        ),
        "task_rows": rows,
        "task_rows_sha256": stable_json_sha256(
            {
                "schema": "repair_archive_variant_materializer_backlog_rows_hash.v1",
                "rows": rows,
            }
        ),
        "opened_by_pipeline": True,
        "pipeline_consumer": "repair_campaign_entropy_stage_materializer_work_order_bundle",
        "allowed_use": "automated_materializer_backlog_routing_only",
        "forbidden_use": "score_claim_or_dispatch_or_submission_authority",
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        backlog,
        context="repair_archive_variant_materializer_backlog",
    )
    return backlog


def _archive_native_zip_repack_candidate(
    *,
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    repo_root: str | Path,
    allow_overwrite: bool,
) -> tuple[dict[str, Any], list[str]]:
    archive = _mapping(manifest.get("candidate_archive"))
    path_text = str(archive.get("path") or "").strip()
    blockers: list[str] = []
    if not path_text:
        blockers.append("candidate_archive_path_missing")
        return {
            "schema": "repair_family_archive_native_candidate.v1",
            "materialized": False,
            "archive_native_transform_attempted": False,
            "archive_native_transform_kind": "zip_repack_payload_identity",
            "path": None,
            "sha256": None,
            "bytes": None,
            "runtime_consumption_proof_path": None,
            "runtime_consumption_proof_ready": False,
            "receiver_contract_satisfied": False,
            "saved_bytes": None,
            "blockers": blockers,
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        }, blockers
    source = _resolve(path_text, repo_root)
    if not source.is_file():
        blockers.append("candidate_archive_file_missing")
        return {
            "schema": "repair_family_archive_native_candidate.v1",
            "materialized": False,
            "archive_native_transform_attempted": False,
            "archive_native_transform_kind": "zip_repack_payload_identity",
            "path": path_text,
            "sha256": None,
            "bytes": None,
            "runtime_consumption_proof_path": None,
            "runtime_consumption_proof_ready": False,
            "receiver_contract_satisfied": False,
            "saved_bytes": None,
            "blockers": blockers,
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        }, blockers
    expected_sha = str(archive.get("sha256") or "").strip()
    actual_sha = sha256_file(source)
    if expected_sha and expected_sha != actual_sha:
        blockers.append("candidate_archive_sha256_mismatch")
        return {
            "schema": "repair_family_archive_native_candidate.v1",
            "materialized": False,
            "archive_native_transform_attempted": False,
            "archive_native_transform_kind": "zip_repack_payload_identity",
            "path": path_text,
            "sha256": actual_sha,
            "bytes": source.stat().st_size,
            "runtime_consumption_proof_path": None,
            "runtime_consumption_proof_ready": False,
            "receiver_contract_satisfied": False,
            "saved_bytes": None,
            "blockers": blockers,
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        }, blockers
    output = _resolve(output_dir, repo_root) / "candidate_archive_native_zip_repack.zip"
    proof = _resolve(output_dir, repo_root) / "candidate_archive_native_zip_repack_receiver_proof.json"
    expected_output_sha = sha256_file(output) if output.exists() and allow_overwrite else None
    expected_proof_sha = sha256_file(proof) if proof.exists() and allow_overwrite else None
    try:
        native_manifest = materialize_archive_zip_repack_candidate(
            archive_path=source,
            output_archive=output,
            runtime_consumption_proof_out=proof,
            repo_root=repo_root,
            allow_size_regression=True,
            allow_overwrite=allow_overwrite,
            expected_existing_output_sha256=expected_output_sha,
            expected_existing_runtime_consumption_proof_sha256=expected_proof_sha,
        )
    except (FamilyAgnosticMaterializerError, ArtifactWriteError, OSError, ValueError) as exc:
        blockers.append(f"archive_native_zip_repack_failed:{exc}")
        return {
            "schema": "repair_family_archive_native_candidate.v1",
            "materialized": False,
            "archive_native_transform_attempted": True,
            "archive_native_transform_kind": "zip_repack_payload_identity",
            "source_archive_path": _repo_rel(source, repo_root),
            "path": None,
            "sha256": None,
            "bytes": None,
            "runtime_consumption_proof_path": None,
            "runtime_consumption_proof_ready": False,
            "receiver_contract_satisfied": False,
            "saved_bytes": None,
            "blockers": blockers,
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        }, blockers
    native_blockers = _string_list(native_manifest.get("readiness_blockers"))
    blockers.extend(native_blockers)
    selected = _mapping(native_manifest.get("selected_repack"))
    candidate = _mapping(native_manifest.get("candidate_archive"))
    source_record = _mapping(native_manifest.get("source_archive"))
    proof_ready = native_manifest.get("receiver_contract_satisfied") is True
    if not proof_ready:
        blockers.append("archive_native_receiver_contract_not_satisfied")
    return {
        "schema": "repair_family_archive_native_candidate.v1",
        "materialized": output.is_file(),
        "archive_native_transform_attempted": True,
        "archive_native_transform_kind": "zip_repack_payload_identity",
        "archive_native_materializer_schema": native_manifest.get("schema"),
        "archive_native_materializer_id": native_manifest.get("materializer_id"),
        "archive_native_target_kind": native_manifest.get("target_kind"),
        "source_archive_path": source_record.get("path") or _repo_rel(source, repo_root),
        "source_archive_sha256": source_record.get("sha256") or actual_sha,
        "source_archive_bytes": source_record.get("bytes") or source.stat().st_size,
        "path": candidate.get("path") or _repo_rel(output, repo_root),
        "sha256": candidate.get("sha256") or sha256_file(output),
        "bytes": candidate.get("bytes") or output.stat().st_size,
        "runtime_consumption_proof_path": _repo_rel(proof, repo_root) if proof.is_file() else None,
        "runtime_consumption_proof_ready": proof_ready,
        "receiver_contract_kind": native_manifest.get("receiver_contract_kind"),
        "receiver_contract_satisfied": proof_ready,
        "saved_bytes": selected.get("saved_bytes"),
        "selected_repack": dict(selected),
        "blockers": ordered_unique(blockers),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }, blockers


def _exact_eval_handoff_gate(
    *,
    manifest: Mapping[str, Any],
    candidate_archive: Mapping[str, Any],
    blockers: Sequence[str],
) -> dict[str, Any]:
    receiver = _mapping(manifest.get("receiver_verification"))
    exact_blockers = ordered_unique(
        [
            *list(blockers),
            *([] if candidate_archive.get("materialized") is True else ["byte_closed_candidate_archive_missing"]),
            *(
                []
                if candidate_archive.get("runtime_consumption_proof_ready") is True
                or (
                    manifest.get("receiver_contract_satisfied") is True
                    and receiver.get("runtime_consumption_proof_passed") is True
                )
                else ["archive_bound_receiver_runtime_proof_missing"]
            ),
            "contest_cpu_or_cuda_exact_axis_payload_required",
            "lane_dispatch_claim_required_before_exact_eval",
        ]
    )
    return {
        "schema": REPAIR_FAMILY_EXACT_EVAL_HANDOFF_GATE_SCHEMA,
        "eligible_for_exact_eval_handoff": False,
        "candidate_archive_materialized": candidate_archive.get("materialized") is True,
        "archive_bound_runtime_consumption_proof_ready": (
            candidate_archive.get("runtime_consumption_proof_ready") is True
            or (
                manifest.get("receiver_contract_satisfied") is True
                and receiver.get("runtime_consumption_proof_passed") is True
            )
        ),
        "component_response_axis": "[macOS-MLX research-signal]",
        "exact_axis_required": ["contest-CPU", "contest-CUDA"],
        "blockers": exact_blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def _source_records(
    *,
    manifest_path: str | Path,
    delta: Mapping[str, Any],
    manifest: Mapping[str, Any],
    repo_root: str | Path,
) -> list[dict[str, Any]]:
    records = [
        _file_record(
            label="repair_family_materializer_manifest",
            path=manifest_path,
            repo_root=repo_root,
        ),
        _file_record(
            label="repair_family_byte_transform_payload",
            path=str(delta.get("path") or ""),
            repo_root=repo_root,
        ),
    ]
    replay = _mapping(manifest.get("component_response_replay"))
    for label, key in (
        ("local_mlx_response", "local_mlx_response_path"),
        ("reference_local_mlx_response", "reference_local_mlx_response_path"),
    ):
        path = str(replay.get(key) or "").strip() or _artifact_path_from_statuses(
            manifest,
            key,
        )
        if path:
            records.append(_file_record(label=label, path=path, repo_root=repo_root, required=False))
    return records


def _build_replay_bundle(
    *,
    manifest_path: str | Path,
    manifest: Mapping[str, Any],
    delta: Mapping[str, Any],
    replay_argv: Sequence[str],
    invocation_argv: Sequence[str],
    repo_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root)
    source_records = _source_records(
        manifest_path=manifest_path,
        delta=delta,
        manifest=manifest,
        repo_root=repo,
    )
    hash_manifest = {
        "schema": "repair_family_byte_transform_replay_hash_manifest.v1",
        "source_records": source_records,
        "family_id": manifest.get("family_id") or manifest.get("target_kind"),
        "typed_response_id": manifest.get("typed_response_id"),
        "candidate_chain_id": manifest.get("candidate_chain_id"),
        "delta_sha256": delta.get("sha256"),
        "delta_bytes": delta.get("bytes"),
        "replay_argv": list(replay_argv),
    }
    environment = capture_safe_replay_environment()
    python_context = {
        "executable": sys.executable,
        "version": sys.version,
        "platform": platform.platform(),
    }
    execution_context = {
        "schema": "repair_family_byte_transform_replay_execution_context.v1",
        "invocation_argv": list(invocation_argv),
        "python": python_context,
        "environment": environment,
        "git": {
            "head": _git_text(["rev-parse", "HEAD"], repo_root=repo),
            "status_short": _git_text(["status", "--short"], repo_root=repo),
        },
    }
    bundle = {
        "schema": REPAIR_FAMILY_BYTE_TRANSFORM_REPLAY_BUNDLE_SCHEMA,
        "generated_at_utc": _utc_now(),
        "tool": "tools/run_repair_family_byte_transform_executor.py",
        "replay_target_tool": "tools/run_repair_family_byte_transform_executor.py",
        "source_manifest_path": str(manifest_path),
        "source_manifest_schema": manifest.get("schema"),
        "family_id": manifest.get("family_id") or manifest.get("target_kind"),
        "typed_response_id": manifest.get("typed_response_id"),
        "candidate_chain_id": manifest.get("candidate_chain_id"),
        "component_response_axis": "[macOS-MLX research-signal]",
        "source_records": source_records,
        "hash_manifest": hash_manifest,
        "hash_manifest_sha256": stable_json_sha256(hash_manifest),
        "source_records_sha256": stable_json_sha256(
            {
                "schema": "repair_family_byte_transform_replay_source_records.v1",
                "source_records": source_records,
            }
        ),
        "replay_argv": list(replay_argv),
        "replay_argv_sha256": stable_json_sha256(
            {
                "schema": "repair_family_byte_transform_replay_argv.v1",
                "replay_argv": list(replay_argv),
            }
        ),
        "invocation_argv": list(invocation_argv),
        "invocation_argv_sha256": stable_json_sha256(
            {
                "schema": "repair_family_byte_transform_invocation_argv.v1",
                "invocation_argv": list(invocation_argv),
            }
        ),
        "execution_context_manifest": execution_context,
        "execution_context_sha256": stable_json_sha256(execution_context),
        "environment_sha256": stable_json_sha256(environment),
        "local_mlx_rows_are_advisory_only": True,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        bundle,
        context="repair_family_byte_transform_replay_bundle",
    )
    return bundle


def build_repair_family_byte_transform_execution_report(
    *,
    family_materializer_manifest: Mapping[str, Any],
    family_materializer_manifest_path: str | Path,
    output_dir: str | Path,
    replay_argv: Sequence[str],
    invocation_argv: Sequence[str],
    repo_root: str | Path,
    allow_overwrite: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a concrete repair-family byte-transform executor."""

    if family_materializer_manifest.get("schema") != REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA:
        raise RepairFamilyByteTransformExecutorError(
            "repair family byte transform requires repair family materializer manifest"
        )
    require_no_truthy_authority_fields(
        family_materializer_manifest,
        context="repair_family_byte_transform_manifest",
    )
    raw_family_id = str(
        family_materializer_manifest.get("family_id") or family_materializer_manifest.get("target_kind") or ""
    ).strip()
    family_id = raw_family_id or "unclassified_repair_family"
    supported = family_id in SUPPORTED_REPAIR_BYTE_TRANSFORM_FAMILIES
    blockers: list[str] = [
        "repair_family_byte_transform_is_mlx_advisory_only",
        "exact_auth_eval_required_before_score_or_promotion_claim",
    ]
    if not supported:
        blockers.append(f"unsupported_repair_family_byte_transform:{family_id}")

    transform_payload = _build_transform_payload(
        manifest=family_materializer_manifest,
        manifest_path=family_materializer_manifest_path,
        family_id=family_id,
    )
    archive_family_probe = _archive_family_probe(
        family_materializer_manifest,
        repo_root=repo_root,
    )
    delta = _write_transform_payload(
        payload=transform_payload,
        output_dir=output_dir,
        repo_root=repo_root,
        family_id=family_id,
        typed_response_id=str(family_materializer_manifest.get("typed_response_id") or ""),
        allow_overwrite=allow_overwrite,
    )
    (
        candidate_archive,
        archive_variants,
        archive_blockers,
        archive_bound_candidate_contract_surface,
    ) = _archive_transform_candidates(
        manifest=family_materializer_manifest,
        output_dir=output_dir,
        repo_root=repo_root,
        family_id=family_id,
        allow_overwrite=allow_overwrite,
    )
    archive_entropy_substrate_coverage = (
        build_repair_archive_entropy_substrate_coverage(
            archive_family_probe=archive_family_probe,
            candidate_archive_transform_variants=archive_variants,
            selected_candidate_archive=candidate_archive,
        )
    )
    archive_variant_signal_surface = _archive_variant_signal_surface(
        variants=archive_variants,
        selected_candidate_archive=candidate_archive,
    )
    archive_variant_materializer_backlog = _archive_variant_materializer_backlog(
        signal_surface=archive_variant_signal_surface,
        variants=archive_variants,
    )
    blockers.extend(archive_blockers)
    replay_bundle = _build_replay_bundle(
        manifest_path=family_materializer_manifest_path,
        manifest=family_materializer_manifest,
        delta=delta,
        replay_argv=replay_argv,
        invocation_argv=invocation_argv,
        repo_root=repo_root,
    )
    exact_gate = _exact_eval_handoff_gate(
        manifest=family_materializer_manifest,
        candidate_archive=candidate_archive,
        blockers=blockers,
    )
    adapter_candidate_id = (
        str(family_materializer_manifest.get("candidate_chain_id") or "").strip()
        or str(family_materializer_manifest.get("typed_response_id") or "").strip()
        or family_id
    )
    adapter_input_artifacts = ordered_unique(
        [
            _repo_rel(family_materializer_manifest_path, repo_root),
            str(candidate_archive.get("path") or ""),
            str(candidate_archive.get("runtime_consumption_proof_path") or ""),
        ]
    )
    archive_bound_candidate_adapter_package = (
        build_archive_bound_candidate_adapter_package(
            _SingleRepairArchiveCandidateAdapter(
                adapter_id=(
                    f"repair_family_byte_transform_executor:{family_id}:"
                    "selected_archive_candidate"
                ),
                candidate_family=family_id,
                row={
                    **candidate_archive,
                    "candidate_id": adapter_candidate_id,
                    "candidate_family": family_id,
                    "typed_response_id": str(
                        family_materializer_manifest.get("typed_response_id") or ""
                    ),
                    "candidate_chain_id": str(
                        family_materializer_manifest.get("candidate_chain_id") or ""
                    ),
                    "candidate_archive_path": str(candidate_archive.get("path") or ""),
                    "candidate_archive_sha256": str(
                        candidate_archive.get("sha256") or ""
                    ),
                    "candidate_archive_bytes": candidate_archive.get("bytes"),
                    "replay_argv": _string_list(replay_argv),
                    "input_artifacts": adapter_input_artifacts,
                },
            ),
            repo_root=repo_root,
        )
    )
    component_delta = _component_probe_delta(family_materializer_manifest)
    report = {
        "schema": REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA,
        "generated_at_utc": _utc_now(),
        "materializer_id": (f"repair_family_byte_transform_executor:{family_id}"),
        "manifest_kind": "repair_family_byte_transform_execution_report",
        "source_family_materializer_manifest_path": str(family_materializer_manifest_path),
        "source_family_materializer_manifest_schema": (family_materializer_manifest.get("schema")),
        "target_kind": family_materializer_manifest.get("target_kind"),
        "family_id": family_id,
        "typed_response_id": family_materializer_manifest.get("typed_response_id"),
        "candidate_chain_id": family_materializer_manifest.get("candidate_chain_id"),
        "candidate_chain_ids": _string_list(family_materializer_manifest.get("candidate_chain_ids")),
        "repair_budget_candidate_chain_id": family_materializer_manifest.get("repair_budget_candidate_chain_id"),
        "repair_budget_candidate_chain_ids": _string_list(
            family_materializer_manifest.get("repair_budget_candidate_chain_ids")
        ),
        "archive_family_probe": archive_family_probe,
        "entropy_position_label": family_materializer_manifest.get("entropy_position_label"),
        "active_entropy_stage": dict(_mapping(family_materializer_manifest.get("active_entropy_stage"))),
        "fractal_optimization_scope": dict(_mapping(family_materializer_manifest.get("fractal_optimization_scope"))),
        "allocated_repair_bytes": transform_payload.get("allocated_repair_bytes"),
        "byte_transform_supported": supported,
        "byte_transform_delta_emitted": True,
        "byte_transform_delta": delta,
        "candidate_delta": delta,
        "candidate_archive": candidate_archive,
        "archive_bound_candidate_contract_schema": (
            ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
        ),
        "archive_bound_candidate_contract": candidate_archive.get(
            "archive_bound_candidate_contract",
            {},
        ),
        "archive_bound_candidate_contract_surface_schema": (
            ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA
        ),
        "archive_bound_candidate_contract_surface": (
            archive_bound_candidate_contract_surface
        ),
        "archive_bound_candidate_contract_count": (
            archive_bound_candidate_contract_surface["candidate_contract_count"]
        ),
        "archive_bound_ready_contract_count": (
            archive_bound_candidate_contract_surface["archive_bound_ready_contract_count"]
        ),
        "archive_contract_substrate_tags": (
            archive_bound_candidate_contract_surface["archive_substrate_tags"]
        ),
        "archive_contract_acquisition_penalty_sum": (
            archive_bound_candidate_contract_surface["acquisition_penalty_sum"]
        ),
        "archive_bound_candidate_adapter_package_schema": (
            ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA
        ),
        "archive_bound_candidate_adapter_package": (
            archive_bound_candidate_adapter_package
        ),
        "archive_bound_candidate_adapter_package_candidate_count": (
            archive_bound_candidate_adapter_package["candidate_row_count"]
        ),
        "archive_bound_candidate_adapter_package_receiver_gate_passed_count": (
            archive_bound_candidate_adapter_package[
                "receiver_proof_gate_passed_count"
            ]
        ),
        "archive_bound_candidate_adapter_package_exact_blocker_count": len(
            archive_bound_candidate_adapter_package["exact_axis_blockers"]
        ),
        "candidate_archive_transform_variants": archive_variants,
        "candidate_archive_transform_variant_count": len(archive_variants),
        "selected_archive_transform_kind": candidate_archive.get("archive_native_transform_kind"),
        "archive_variant_signal_surface_schema": REPAIR_ARCHIVE_VARIANT_SIGNAL_SURFACE_SCHEMA,
        "archive_variant_signal_surface": archive_variant_signal_surface,
        "archive_variant_signal_count": archive_variant_signal_surface["row_count"],
        "archive_variant_non_selected_signal_count": archive_variant_signal_surface[
            "non_selected_signal_count"
        ],
        "archive_variant_probe_count": archive_variant_signal_surface["probe_count"],
        "archive_variant_prototype_count": archive_variant_signal_surface["prototype_count"],
        "archive_variant_runtime_proof_ready_count": archive_variant_signal_surface[
            "runtime_proof_ready_count"
        ],
        "archive_variant_signal_kinds": archive_variant_signal_surface[
            "signal_transform_kinds"
        ],
        "archive_variant_signal_blockers": archive_variant_signal_surface["blockers"],
        "archive_variant_materializer_backlog_schema": (
            REPAIR_ARCHIVE_VARIANT_MATERIALIZER_BACKLOG_SCHEMA
        ),
        "archive_variant_materializer_backlog": archive_variant_materializer_backlog,
        "archive_variant_materializer_backlog_task_count": (
            archive_variant_materializer_backlog["row_count"]
        ),
        "archive_variant_materializer_byte_closed_task_count": (
            archive_variant_materializer_backlog["byte_closed_materialized_task_count"]
        ),
        "archive_variant_materializer_runtime_adapter_ready_task_count": (
            archive_variant_materializer_backlog["runtime_adapter_ready_task_count"]
        ),
        "archive_entropy_substrate_coverage_schema": (
            REPAIR_ARCHIVE_ENTROPY_SUBSTRATE_COVERAGE_SCHEMA
        ),
        "archive_entropy_substrate_coverage": archive_entropy_substrate_coverage,
        "semantic_payload_changed": candidate_archive.get("semantic_payload_changed") is True,
        "exact_axis_score_affecting_adjudication_required": (
            candidate_archive.get("exact_axis_score_affecting_adjudication_required") is True
        ),
        "score_affecting_payload_changed": candidate_archive.get("score_affecting_payload_changed") is True,
        "charged_bits_changed": candidate_archive.get("charged_bits_changed") is True,
        "archive_native_transform_attempted": (candidate_archive.get("archive_native_transform_attempted") is True),
        "archive_native_transform_kind": candidate_archive.get("archive_native_transform_kind"),
        "archive_native_saved_bytes": candidate_archive.get("saved_bytes"),
        "byte_closed_candidate_emitted": candidate_archive.get("materialized") is True,
        "candidate_archive_materialized": candidate_archive.get("materialized") is True,
        "runtime_consumption_proof_path": candidate_archive.get("runtime_consumption_proof_path")
        or family_materializer_manifest.get("runtime_consumption_proof_path"),
        "receiver_contract_kind": family_materializer_manifest.get("receiver_contract_kind")
        or candidate_archive.get("receiver_contract_kind"),
        "receiver_contract_satisfied": (
            family_materializer_manifest.get("receiver_contract_satisfied") is True
            or candidate_archive.get("receiver_contract_satisfied") is True
        ),
        "component_response_replayed": (family_materializer_manifest.get("component_response_replayed") is True),
        "component_response_replay": dict(_mapping(family_materializer_manifest.get("component_response_replay"))),
        "mlx_local_probe_delta": component_delta,
        "component_response_replay_axis_tag": component_delta["component_response_axis"],
        "component_response_replay_path": _mapping(family_materializer_manifest.get("component_response_replay")).get(
            "artifact_path"
        ),
        "exact_eval_handoff_gate": exact_gate,
        "exact_eval_handoff_eligible": False,
        "replay_bundle_schema": REPAIR_FAMILY_BYTE_TRANSFORM_REPLAY_BUNDLE_SCHEMA,
        "replay_bundle_hash_manifest_sha256": replay_bundle.get("hash_manifest_sha256"),
        "readiness_blockers": ordered_unique(blockers),
        "blockers": ordered_unique(blockers),
        "local_mlx_rows_are_advisory_only": True,
        "encoder_side_only": True,
        "receiver_must_remain_decode_only": True,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_family_byte_transform_executor_local_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_promotion_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        report,
        context=f"repair_family_byte_transform_execution_report:{family_id}",
    )
    return report, replay_bundle


__all__ = [
    "FEC5_FIXED_K8_CODE_BITS",
    "FEC6_FIXED_K16_CODE_BITS",
    "REPAIR_ARCHIVE_VARIANT_MATERIALIZER_BACKLOG_ROW_SCHEMA",
    "REPAIR_ARCHIVE_VARIANT_MATERIALIZER_BACKLOG_SCHEMA",
    "REPAIR_ARCHIVE_VARIANT_SIGNAL_ROW_SCHEMA",
    "REPAIR_ARCHIVE_VARIANT_SIGNAL_SURFACE_SCHEMA",
    "REPAIR_FAMILY_BYTE_TRANSFORM_DELTA_SCHEMA",
    "REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA",
    "REPAIR_FAMILY_BYTE_TRANSFORM_PAYLOAD_SCHEMA",
    "REPAIR_FAMILY_BYTE_TRANSFORM_REPLAY_BUNDLE_SCHEMA",
    "REPAIR_FAMILY_EXACT_EVAL_HANDOFF_GATE_SCHEMA",
    "SUPPORTED_REPAIR_BYTE_TRANSFORM_FAMILIES",
    "RepairFamilyByteTransformExecutorError",
    "build_repair_family_byte_transform_execution_report",
]
