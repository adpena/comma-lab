#!/usr/bin/env python3
"""Build A2 sensitivity-weighted PR101 lossy-coarsening packet ladders.

This consumes ``tools/sensitivity_weighted_lossy_coarsening.py`` manifests and
materializes selected-K schedules into byte-closed local PR101 runtime packets.
It never claims a score, never claims a lane, never dispatches remote work, and
keeps all variants non-promotable until exact auth eval plus operator review.

The packet-local runtime patch is intentionally emitted only inside the output
packet directory. The source PR101 runtime remains untouched.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.pr101_split_brotli_codec import (  # noqa: E402
    CONV4_STORAGE_PERMS,
    DECODER_BLOB_LEN,
    DECODER_BYTE_MAPS,
    DECODER_STORAGE_ORDER,
    DECODER_STREAM_ENDS,
    FIXED_STATE_SCHEMA,
    LATENT_BLOB_LEN,
    N_QUANT,
    _encode_mapped_u8,
    _quantize_tensor,
    decode_decoder_compact,
    pack_brotli_stream,
)
from tac.repo_io import json_text, read_json, repo_relative, sha256_file, write_json  # noqa: E402

TOOL_NAME = "tools/build_a2_sensitivity_weighted_pr101_packet.py"
SCHEMA_VERSION = "a2_sensitivity_weighted_pr101_packet_ladder.v1"
VARIANT_SCHEMA_VERSION = "a2_sensitivity_weighted_pr101_packet_variant.v1"
A2_SOURCE_SCHEMA = "phase_a2_sensitivity_weighted_lossy_coarsening.v1"
A2_SOURCE_TOOL = "tools/sensitivity_weighted_lossy_coarsening.py"
MAGIC = b"A2K1"
DEFAULT_SOURCE_ARCHIVE = Path(
    "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
)
DEFAULT_SOURCE_RUNTIME_DIR = Path(
    "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex"
    "/source/submissions/hnerv_ft_microcodec"
)
DEFAULT_STATE_DICT = Path(
    "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt"
)
DEFAULT_OUTPUT_ROOT = Path("experiments/results/track1_phase_a2_sensitivity_pr101_packet_ladder")
EXCLUDED_DIR_NAMES = frozenset({"__pycache__", ".git", ".mypy_cache", ".pytest_cache"})
EXCLUDED_FILE_NAMES = frozenset({".DS_Store"})
EXCLUDED_SUFFIXES = (".pyc", ".pyo")
PACKET_CUSTODY_FILENAMES = frozenset(
    {
        "archive.zip",
        "archive_manifest.json",
        "candidate_manifest.json",
        "contest_auth_eval.json",
        "pre_submission_compliance.json",
        "pre_submission_compliance.nonfinal.json",
        "report.txt",
    }
)
BASE_DISPATCH_BLOCKERS = [
    "no_exact_cuda_auth_eval",
    "no_contest_cpu_auth_eval",
    "no_active_level2_lane_dispatch_claim",
    "operator_score_claim_review_not_done",
    "packet_local_inflate_parity_not_run",
]
CLEARED_BY_PACKET_LADDER = frozenset(
    {
        "no_byte_closed_runtime_packet_built",
        "packet_local_inflate_parity_not_run",
    }
)
FALSE_AUTHORITY_FIELDS = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
UPSTREAM_FALSE_AUTHORITY_FIELDS = (*FALSE_AUTHORITY_FIELDS, "dispatch_attempted")


class PacketClosureBlocked(ValueError):
    """Raised when a selected-K schedule cannot be made byte-closed."""

    def __init__(self, reason: str, missing_wire_contracts: list[str]):
        super().__init__(reason)
        self.missing_wire_contracts = missing_wire_contracts


def _utc_ts() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _format_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(value: str | None) -> dt.datetime:
    if not value:
        return dt.datetime.now(dt.UTC).replace(microsecond=0)
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC).replace(microsecond=0)


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_rel(path: Path) -> str:
    return repo_relative(_repo_path(path), REPO_ROOT)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(json_text(payload).encode("utf-8")).hexdigest()


def _mode_string(mode: int) -> str:
    return f"{stat.S_IMODE(mode):04o}"


def _copy_mode(source: Path) -> int:
    return 0o755 if source.stat().st_mode & 0o111 else 0o644


def _should_exclude(path: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return True
    return path.name in EXCLUDED_FILE_NAMES or path.suffix in EXCLUDED_SUFFIXES


def _is_packet_custody_file(path: Path) -> bool:
    return path.name in PACKET_CUSTODY_FILENAMES or path.name.startswith(
        "pre_submission_compliance."
    )


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _assert_output_dir_isolated(
    *,
    output_dir: Path,
    a2_manifest_path: Path,
    state_dict_path: Path,
    source_archive: Path,
    source_runtime_dir: Path,
) -> None:
    output_resolved = output_dir.resolve(strict=False)
    runtime_resolved = source_runtime_dir.resolve(strict=False)
    if _is_relative_to(output_resolved, runtime_resolved) or _is_relative_to(
        runtime_resolved, output_resolved
    ):
        raise PacketClosureBlocked(
            (
                "output directory must not overlap the source runtime tree; "
                f"output={output_dir} source_runtime_dir={source_runtime_dir}"
            ),
            ["output_directory_overlaps_source_runtime_tree"],
        )

    for label, input_path in (
        ("a2_manifest", a2_manifest_path),
        ("state_dict", state_dict_path),
        ("source_archive", source_archive),
    ):
        input_resolved = input_path.resolve(strict=False)
        if output_resolved == input_resolved or _is_relative_to(input_resolved, output_resolved):
            raise PacketClosureBlocked(
                f"output directory would contain input {label}: {input_path}",
                [f"output_directory_contains_{label}"],
            )


def _iter_runtime_files(runtime_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in runtime_dir.rglob("*"):
        rel = path.relative_to(runtime_dir)
        if _should_exclude(rel):
            continue
        if path.is_symlink():
            raise PacketClosureBlocked(
                f"runtime packet refuses symlink: {path}",
                ["runtime_tree_contains_symlink"],
            )
        if path.is_file():
            files.append(rel)
    return sorted(files, key=lambda item: item.as_posix())


def _file_record(path: Path, *, relpath: str | None = None) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "mode": _mode_string(path.stat().st_mode),
        "relpath": relpath or path.name,
        "sha256": sha256_file(path),
    }


def _runtime_tree_sha256(runtime_files: list[dict[str, Any]]) -> str:
    basis = [
        {
            "bytes": row["bytes"],
            "mode": row["mode"],
            "relpath": row["relpath"],
            "sha256": row["sha256"],
        }
        for row in runtime_files
    ]
    return _canonical_json_sha256(basis)


def _bash_n(path: Path) -> dict[str, Any]:
    proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
    return {
        "command": f"bash -n {path}",
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip(),
        "stdout": proc.stdout.strip(),
    }


def _prepare_empty_dir(path: Path, *, force: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise PacketClosureBlocked(
                f"output path exists and is not a directory: {path}",
                ["output_path_not_directory"],
            )
        if any(path.iterdir()):
            if not force:
                raise PacketClosureBlocked(
                    f"output directory is not empty; pass --force to replace: {path}",
                    ["output_directory_not_empty_without_force"],
                )
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_runtime_tree(source_runtime_dir: Path, packet_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rel in _iter_runtime_files(source_runtime_dir):
        source = source_runtime_dir / rel
        target = packet_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        os.chmod(target, _copy_mode(source))
        records.append(_file_record(target, relpath=rel.as_posix()))
    return records


def _read_single_member_zip(path: Path) -> tuple[dict[str, Any], str, bytes]:
    archive_bytes = path.read_bytes()
    archive_record: dict[str, Any] = {
        "path": _repo_rel(path),
        "bytes": len(archive_bytes),
        "sha256": _sha256_bytes(archive_bytes),
        "members": [],
    }
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise PacketClosureBlocked(
                f"duplicate ZIP members are not valid custody input: {path}",
                ["source_archive_duplicate_zip_members"],
            )
        if len(infos) != 1:
            raise PacketClosureBlocked(
                f"expected one PR101 archive member, found {len(infos)} in {path}",
                ["source_archive_not_single_member_pr101_packet"],
            )
        info = infos[0]
        _validate_zip_member_name(info.filename)
        local = _local_header_record(path, info)
        if local["name"] != info.filename:
            raise PacketClosureBlocked(
                (
                    "ZIP local/central header member-name mismatch: "
                    f"central={info.filename!r} local={local['name']!r}"
                ),
                ["zip_local_header_name_mismatch"],
            )
        _validate_zip_member_name(str(local["name"]))
        member = zf.read(info)
        archive_record["members"].append(
            {
                "bytes": info.file_size,
                "compress_size": info.compress_size,
                "compress_type": info.compress_type,
                "crc": f"{info.CRC:08x}",
                "date_time": list(info.date_time),
                "external_attr": info.external_attr,
                "flag_bits": info.flag_bits,
                "local_header": local,
                "name": info.filename,
                "sha256": _sha256_bytes(member),
            }
        )
    return archive_record, infos[0].filename, member


def _write_single_member_zip(path: Path, member_name: str, payload: bytes) -> dict[str, Any]:
    _validate_zip_member_name(member_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(filename=member_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = (0o644 & 0xFFFF) << 16
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(info, payload)
    archive_record, _member_name, _payload = _read_single_member_zip(path)
    return archive_record


def _decode_zip_name(raw: bytes, flag_bits: int) -> str | None:
    encoding = "utf-8" if flag_bits & 0x800 else "cp437"
    try:
        return raw.decode(encoding, errors="strict")
    except UnicodeDecodeError:
        return None


def _local_header_record(path: Path, info: zipfile.ZipInfo) -> dict[str, Any]:
    with path.open("rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise PacketClosureBlocked(
                f"bad ZIP local header for member {info.filename!r} in {path}",
                ["zip_local_header_unreadable"],
            )
        flag_bits = struct.unpack_from("<H", header, 6)[0]
        method = struct.unpack_from("<H", header, 8)[0]
        name_len, extra_len = struct.unpack_from("<HH", header, 26)
        raw_name = handle.read(name_len)
        handle.read(extra_len)
    return {
        "name": _decode_zip_name(raw_name, flag_bits),
        "flag_bits": flag_bits,
        "compress_type": method,
        "extra_len": extra_len,
    }


def _validate_zip_member_name(name: str) -> None:
    path = Path(name)
    if (
        not name
        or "\\" in name
        or "\x00" in name
        or any(ord(ch) < 32 for ch in name)
        or re.match(r"^[A-Za-z]:", name)
        or name.startswith("/")
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(part.startswith(".") or part == "__MACOSX" for part in path.parts)
    ):
        raise PacketClosureBlocked(
            f"unsafe ZIP member name for A2 packet custody: {name!r}",
            ["unsafe_zip_member_name"],
        )


def _split_pr101_member(payload: bytes) -> tuple[bytes, bytes, bytes]:
    minimum = DECODER_BLOB_LEN + LATENT_BLOB_LEN
    if len(payload) < minimum:
        raise PacketClosureBlocked(
            f"PR101 member payload length {len(payload)} < fixed decoder+latent minimum {minimum}",
            ["source_archive_member_too_short_for_pr101_fixed_layout"],
        )
    decoder_blob = payload[:DECODER_BLOB_LEN]
    latent_blob = payload[DECODER_BLOB_LEN : DECODER_BLOB_LEN + LATENT_BLOB_LEN]
    sidecar_blob = payload[DECODER_BLOB_LEN + LATENT_BLOB_LEN :]
    return decoder_blob, latent_blob, sidecar_blob


def _validate_selected_ks(values: Any, *, expected_count: int) -> list[int]:
    if not isinstance(values, list):
        raise PacketClosureBlocked(
            "selected_Ks must be a JSON list",
            ["selected_k_schedule_missing_or_not_list"],
        )
    if len(values) != expected_count:
        raise PacketClosureBlocked(
            f"selected_Ks length {len(values)} != expected tensor count {expected_count}",
            ["selected_k_schedule_tensor_count_mismatch"],
        )
    out: list[int] = []
    for idx, value in enumerate(values):
        if not isinstance(value, int) or isinstance(value, bool):
            raise PacketClosureBlocked(
                f"selected_Ks[{idx}] is not an integer",
                ["selected_k_schedule_contains_non_integer"],
            )
        if value < 1 or value > 255:
            raise PacketClosureBlocked(
                f"selected_Ks[{idx}]={value} outside uint8 K range [1, 255]",
                ["selected_k_schedule_outside_uint8_range"],
            )
        out.append(int(value))
    return out


def _variant_id(row: dict[str, Any], index: int) -> str:
    target = row.get("rms_target", row.get("budget", index))
    text = str(target).replace(".", "p").replace("-", "m")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    return f"weighted_k_{index:02d}_rms_{text or index}"


def _extract_weighted_schedules(a2_manifest: dict[str, Any], *, variant_limit: int | None) -> list[dict[str, Any]]:
    allocations = a2_manifest.get("weighted_k_allocations")
    if not isinstance(allocations, list) or not allocations:
        raise PacketClosureBlocked(
            "A2 manifest has no weighted_k_allocations rows",
            ["a2_manifest_missing_weighted_k_allocations"],
        )
    schedules: list[dict[str, Any]] = []
    expected_count = len(FIXED_STATE_SCHEMA)
    for index, row in enumerate(allocations):
        if not isinstance(row, dict):
            raise PacketClosureBlocked(
                f"weighted_k_allocations[{index}] is not an object",
                ["a2_manifest_weighted_k_row_not_object"],
            )
        selected_ks = _validate_selected_ks(row.get("selected_Ks"), expected_count=expected_count)
        schedules.append(
            {
                "source": "weighted_k_allocations",
                "source_index": index,
                "variant_id": _variant_id(row, index),
                "rms_target": row.get("rms_target"),
                "reported_total_bytes": row.get("total_bytes"),
                "reported_rel_err": row.get("rel_err"),
                "joint_encoder_extras": row.get("joint_encoder_extras")
                if isinstance(row.get("joint_encoder_extras"), dict)
                else None,
                "selected_Ks": selected_ks,
            }
        )
    if variant_limit is not None and variant_limit > 0:
        schedules = schedules[:variant_limit]
    return schedules


def _list_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _active_packet_blockers(*groups: list[str], clear: set[str] | frozenset[str] = frozenset()) -> list[str]:
    return sorted({item for group in groups for item in group if item not in clear})


def _validate_a2_manifest_authority(a2_manifest: dict[str, Any], *, manifest_path: Path) -> dict[str, Any]:
    blockers: list[str] = []
    schema = a2_manifest.get("schema")
    tool = a2_manifest.get("tool")
    if schema != A2_SOURCE_SCHEMA:
        blockers.append("a2_manifest_wrong_schema")
    if tool != A2_SOURCE_TOOL:
        blockers.append("a2_manifest_wrong_tool")
    for field in UPSTREAM_FALSE_AUTHORITY_FIELDS:
        if a2_manifest.get(field) is not False:
            blockers.append(f"a2_manifest_{field}_not_false")
    if blockers:
        raise PacketClosureBlocked(
            f"A2 manifest failed authority validation: {', '.join(sorted(blockers))}",
            blockers,
        )

    manifest_blockers = _list_strings(a2_manifest.get("dispatch_blockers"))
    sensitivity_artifact = (
        a2_manifest.get("sensitivity_artifact")
        if isinstance(a2_manifest.get("sensitivity_artifact"), dict)
        else {}
    )
    metadata_blockers = _list_strings(sensitivity_artifact.get("metadata_blockers"))
    propagated_blockers = sorted(set(manifest_blockers + metadata_blockers))
    return {
        "path": _repo_rel(manifest_path),
        "sha256": sha256_file(manifest_path),
        "schema": schema,
        "tool": tool,
        "status": a2_manifest.get("status"),
        "authority_fields": {
            field: a2_manifest.get(field) for field in UPSTREAM_FALSE_AUTHORITY_FIELDS
        },
        "dispatch_blockers": manifest_blockers,
        "sensitivity_artifact": {
            "path": sensitivity_artifact.get("path"),
            "status": sensitivity_artifact.get("status"),
            "allow_diagnostic_sensitivity": sensitivity_artifact.get(
                "allow_diagnostic_sensitivity"
            ),
            "metadata_blockers": metadata_blockers,
            "metadata": sensitivity_artifact.get("metadata")
            if isinstance(sensitivity_artifact.get("metadata"), dict)
            else {},
        },
        "propagated_blockers": propagated_blockers,
    }


def _tensor_payload(q_i8: np.ndarray, scale: float, idx: int) -> bytes:
    _name, shape = FIXED_STATE_SCHEMA[idx]
    q = q_i8.reshape(shape)
    if len(shape) == 4 and idx in CONV4_STORAGE_PERMS:
        flat = np.transpose(q, CONV4_STORAGE_PERMS[idx]).copy().reshape(-1)
    else:
        flat = q.reshape(-1)
    byte_map = DECODER_BYTE_MAPS.get(idx, "zig")
    mapped = _encode_mapped_u8(flat.astype(np.int8), byte_map)
    return mapped.tobytes() + np.array([scale], dtype=np.float16).tobytes()


def _encode_rounded_decoder_blob(
    *,
    state_dict_path: Path,
    selected_ks: list[int],
    brotli_quality: int,
) -> tuple[bytes, dict[str, Any]]:
    if not state_dict_path.is_file():
        raise PacketClosureBlocked(
            f"state_dict not found: {state_dict_path}",
            ["state_dict_missing_for_k_schedule_materialization"],
        )
    state = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise PacketClosureBlocked(
            f"{state_dict_path}: expected tensor state_dict",
            ["state_dict_not_tensor_mapping"],
        )
    if len(selected_ks) != len(FIXED_STATE_SCHEMA):
        raise PacketClosureBlocked(
            f"selected_Ks length {len(selected_ks)} != schema count {len(FIXED_STATE_SCHEMA)}",
            ["selected_k_schedule_tensor_count_mismatch"],
        )

    payloads_by_index: dict[int, bytes] = {}
    tensor_rows: list[dict[str, Any]] = []
    abs_orig_total = 0.0
    abs_err_total = 0.0
    for idx, ((name, shape), k) in enumerate(zip(FIXED_STATE_SCHEMA, selected_ks, strict=True)):
        if name not in state or not torch.is_tensor(state[name]):
            raise PacketClosureBlocked(
                f"state_dict missing tensor {name!r}",
                ["state_dict_missing_schema_tensor"],
            )
        qt = _quantize_tensor(name, state[name], n_quant=N_QUANT)
        raw = qt.q_i8.astype(np.float64)
        rounded = np.round(raw / k) * k
        clipped = rounded.clip(-N_QUANT, N_QUANT).astype(np.int8)
        abs_err = float(np.abs(clipped.astype(np.float64) - raw).sum())
        abs_orig = float(np.abs(raw).sum())
        abs_err_total += abs_err
        abs_orig_total += abs_orig
        payloads_by_index[idx] = _tensor_payload(clipped, float(qt.scale), idx)
        tensor_rows.append(
            {
                "idx": idx,
                "name": name,
                "shape": list(shape),
                "K": k,
                "scale": float(qt.scale),
                "numel": int(clipped.size),
                "abs_err_sum": abs_err,
                "abs_orig_sum": abs_orig,
                "q_sha256": _sha256_bytes(clipped.tobytes()),
            }
        )

    streams: list[bytes] = []
    start = 0
    for end in DECODER_STREAM_ENDS:
        window_raw = b"".join(payloads_by_index[idx] for idx in DECODER_STORAGE_ORDER[start:end])
        streams.append(pack_brotli_stream(window_raw, quality=brotli_quality))
        start = end
    decoder_blob = b"".join(streams)
    rel_err = abs_err_total / abs_orig_total if abs_orig_total > 1e-9 else 0.0
    return decoder_blob, {
        "brotli_quality": brotli_quality,
        "decoder_blob_bytes": len(decoder_blob),
        "decoder_blob_sha256": _sha256_bytes(decoder_blob),
        "rel_err_l1_quantized_proxy": rel_err,
        "abs_err_sum": abs_err_total,
        "abs_orig_sum": abs_orig_total,
        "tensor_rows": tensor_rows,
    }


def _verify_state_dict_matches_source_decoder(
    *,
    state_dict_path: Path,
    source_decoder_blob: bytes,
    brotli_quality: int,
) -> dict[str, Any]:
    reference_blob, reference_encoding = _encode_rounded_decoder_blob(
        state_dict_path=state_dict_path,
        selected_ks=[1] * len(FIXED_STATE_SCHEMA),
        brotli_quality=brotli_quality,
    )
    source_sha = _sha256_bytes(source_decoder_blob)
    reference_sha = reference_encoding["decoder_blob_sha256"]
    record = {
        "attempted": True,
        "passed": reference_blob == source_decoder_blob,
        "state_dict": _repo_rel(state_dict_path),
        "source_decoder_blob_bytes": len(source_decoder_blob),
        "reference_decoder_blob_bytes": len(reference_blob),
        "source_decoder_blob_sha256": source_sha,
        "reference_decoder_blob_sha256": reference_sha,
        "brotli_quality": brotli_quality,
        "selected_Ks": "all_ones_reference",
    }
    if not record["passed"]:
        raise PacketClosureBlocked(
            (
                "state_dict all-ones materialization does not reproduce the source PR101 "
                f"decoder blob: source={source_sha} reference={reference_sha}"
            ),
            ["state_dict_does_not_reproduce_source_decoder_blob"],
        )
    return record


def _decode_decoder_blob_for_closure(decoder_blob: bytes) -> dict[str, torch.Tensor]:
    return decode_decoder_compact(decoder_blob)


def _build_a2_inner_member(decoder_blob: bytes, latent_blob: bytes, sidecar_blob: bytes) -> bytes:
    if len(decoder_blob) >= (1 << 32):
        raise PacketClosureBlocked(
            f"decoder blob too large for A2 u32 length prefix: {len(decoder_blob)}",
            ["a2_decoder_blob_exceeds_u32_length_prefix"],
        )
    return MAGIC + len(decoder_blob).to_bytes(4, "little") + decoder_blob + latent_blob + sidecar_blob


_PARSE_ARCHIVE_RE = re.compile(
    r"def parse_archive\(archive_bytes\):\n"
    r"(?P<body>.*?)"
    r"    return decode_decoder_compact\(decoder_blob\), latents, meta\n",
    re.DOTALL,
)


def _patch_packet_codec(packet_dir: Path) -> dict[str, Any]:
    codec_path = packet_dir / "src" / "codec.py"
    if not codec_path.is_file():
        raise PacketClosureBlocked(
            f"packet runtime missing src/codec.py: {codec_path}",
            ["runtime_codec_py_missing"],
        )
    text = codec_path.read_text(encoding="utf-8")
    replacement = '''A2_LOSSY_COARSENING_MAGIC = b"A2K1"
A2_LOSSY_COARSENING_HEADER_LEN = 8


def parse_archive(archive_bytes):
    if archive_bytes.startswith(A2_LOSSY_COARSENING_MAGIC):
        if len(archive_bytes) < A2_LOSSY_COARSENING_HEADER_LEN:
            raise ValueError("bad A2 lossy-coarsening archive header")
        decoder_len = int.from_bytes(
            archive_bytes[4:A2_LOSSY_COARSENING_HEADER_LEN],
            "little",
        )
        decoder_start = A2_LOSSY_COARSENING_HEADER_LEN
        decoder_end = decoder_start + decoder_len
        latent_end = decoder_end + LATENT_BLOB_LEN
        if decoder_len <= 0 or latent_end > len(archive_bytes):
            raise ValueError("bad A2 lossy-coarsening archive lengths")
        decoder_blob = archive_bytes[decoder_start:decoder_end]
        latent_blob = archive_bytes[decoder_end:latent_end]
        sidecar_blob = archive_bytes[latent_end:]
    else:
        decoder_blob = archive_bytes[:DECODER_BLOB_LEN]
        latent_blob = archive_bytes[DECODER_BLOB_LEN:DECODER_BLOB_LEN + LATENT_BLOB_LEN]
        sidecar_blob = archive_bytes[DECODER_BLOB_LEN + LATENT_BLOB_LEN:]
    if not decoder_blob or not latent_blob:
        raise ValueError("bad compact archive")
    meta = {
        "n_pairs": N_PAIRS,
        "latent_dim": LATENT_DIM,
        "base_channels": BASE_CHANNELS,
        "eval_size": list(EVAL_SIZE),
    }
    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)
    return decode_decoder_compact(decoder_blob), latents, meta
'''
    patched, count = _PARSE_ARCHIVE_RE.subn(replacement, text, count=1)
    if count != 1:
        raise PacketClosureBlocked(
            "could not patch PR101 runtime parse_archive anchor for A2 variable decoder length",
            ["runtime_codec_parse_archive_patch_anchor_missing"],
        )
    codec_path.write_text(patched, encoding="utf-8")
    return {
        "codec_path": repo_relative(codec_path, REPO_ROOT),
        "codec_sha256": sha256_file(codec_path),
        "codec_parse_archive_supports_a2_length_prefix": True,
        "magic": MAGIC.decode("ascii"),
        "header_bytes": 8,
        "source_runtime_mutated": False,
    }


def _noop_detection(
    *,
    selected_ks: list[int],
    source_decoder_sha256: str,
    candidate_decoder_sha256: str,
    source_member_sha256: str,
    candidate_member_sha256: str,
    source_archive_sha256: str,
    candidate_archive_sha256: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    schedule_all_ones = all(k == 1 for k in selected_ks)
    decoder_matches_source = source_decoder_sha256 == candidate_decoder_sha256
    member_matches_source = source_member_sha256 == candidate_member_sha256
    archive_matches_source = source_archive_sha256 == candidate_archive_sha256
    if schedule_all_ones:
        reasons.append("schedule_all_ones_no_semantic_coarsening")
    if decoder_matches_source:
        reasons.append("candidate_decoder_matches_source")
    if member_matches_source:
        reasons.append("candidate_member_matches_source")
    if archive_matches_source:
        reasons.append("candidate_archive_matches_source")
    return {
        "is_noop": bool(reasons),
        "reasons": reasons,
        "schedule_all_ones": schedule_all_ones,
        "decoder_blob_matches_source": decoder_matches_source,
        "member_matches_source": member_matches_source,
        "archive_matches_source": archive_matches_source,
    }


def _write_packet_report(
    *,
    packet_dir: Path,
    variant_id: str,
    candidate_archive_record: dict[str, Any],
    source_archive_record: dict[str, Any],
    dispatch_blockers: list[str],
    runtime_tree_sha256: str | None = None,
) -> dict[str, Any]:
    report_path = packet_dir / "report.txt"
    lines = [
        "A2 sensitivity-weighted PR101 packet",
        "",
        f"variant_id: {variant_id}",
        f"archive_bytes: {candidate_archive_record['bytes']}",
        f"archive_sha256: {candidate_archive_record['sha256']}",
        f"source_archive_sha256: {source_archive_record['sha256']}",
        "score_claim: false",
        "promotion_eligible: false",
        "rank_or_kill_eligible: false",
        "ready_for_exact_eval_dispatch: false",
        "evidence_grade: empirical",
    ]
    if runtime_tree_sha256:
        lines.append(f"runtime_tree_sha256: {runtime_tree_sha256}")
    lines.extend(["", "dispatch_blockers:"])
    lines.extend(f"- {blocker}" for blocker in dispatch_blockers)
    lines.extend(
        [
            "",
            "This packet is a byte-closed local custody artifact only. It has "
            "not run exact CUDA auth eval, contest-CPU auth eval, or operator "
            "score-claim review.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return _file_record(report_path, relpath="report.txt")


def _packet_local_parse_smoke(packet_dir: Path, candidate_archive_path: Path) -> dict[str, Any]:
    script = r'''
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

packet_dir = Path(sys.argv[1])
archive_path = Path(sys.argv[2])
src_dir = packet_dir / "src"
sys.path.insert(0, str(src_dir))

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

load_module("model", src_dir / "model.py")
codec = load_module("codec", src_dir / "codec.py")
with zipfile.ZipFile(archive_path) as zf:
    infos = [info for info in zf.infolist() if not info.is_dir()]
    if len(infos) != 1:
        raise RuntimeError(f"expected one candidate member, found {len(infos)}")
    payload = zf.read(infos[0])
decoder_sd, latents, meta = codec.parse_archive(payload)
print(json.dumps({
    "passed": True,
    "member_name": infos[0].filename,
    "a2_magic": payload.startswith(b"A2K1"),
    "decoder_tensor_count": len(decoder_sd),
    "latents_shape": list(latents.shape) if hasattr(latents, "shape") else None,
    "latents_dtype": str(getattr(latents, "dtype", "")),
    "meta": {
        "n_pairs": int(meta["n_pairs"]),
        "latent_dim": int(meta["latent_dim"]),
        "base_channels": int(meta["base_channels"]),
        "eval_size": list(meta["eval_size"]),
    },
}, sort_keys=True))
'''
    proc = subprocess.run(
        [sys.executable, "-c", script, str(packet_dir), str(candidate_archive_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise PacketClosureBlocked(
            f"packet-local runtime parse smoke failed: {proc.stderr.strip() or proc.stdout.strip()}",
            ["packet_local_runtime_parse_smoke_failed"],
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise PacketClosureBlocked(
            f"packet-local runtime parse smoke emitted non-JSON output: {proc.stdout!r}",
            ["packet_local_runtime_parse_smoke_non_json"],
        ) from exc
    payload["command"] = "python -c <packet_local_parse_smoke> packet_dir archive.zip"
    return payload


def _proxy_vs_materialized(
    schedule: dict[str, Any],
    candidate_archive_record: dict[str, Any],
    decoder_blob: bytes,
) -> dict[str, Any]:
    extras = schedule.get("joint_encoder_extras")
    if not isinstance(extras, dict):
        extras = {}
    reported_total = schedule.get("reported_total_bytes")
    actual_archive = int(candidate_archive_record["bytes"])
    actual_member = int(candidate_archive_record["members"][0]["bytes"])
    return {
        "authoritative_bytes_field": "candidate_archive.bytes",
        "reported_total_bytes": reported_total,
        "reported_rel_err": schedule.get("reported_rel_err"),
        "reported_payload_brotli_bytes": extras.get("payload_brotli_bytes"),
        "reported_side_info_bytes": extras.get("side_info_bytes"),
        "reported_archive_overhead_bytes": extras.get("archive_overhead_bytes"),
        "actual_archive_bytes": actual_archive,
        "actual_member_bytes": actual_member,
        "actual_decoder_blob_bytes": len(decoder_blob),
        "delta_reported_total_vs_actual_archive_bytes": (
            int(reported_total) - actual_archive if isinstance(reported_total, int) else None
        ),
        "delta_reported_payload_vs_actual_decoder_bytes": (
            int(extras["payload_brotli_bytes"]) - len(decoder_blob)
            if isinstance(extras.get("payload_brotli_bytes"), int)
            else None
        ),
        "note": (
            "A2 selector bytes are analytical/proxy planning fields. The built "
            "packet archive bytes are authoritative for any later exact eval."
        ),
    }


def _build_variant(
    *,
    schedule: dict[str, Any],
    upstream_a2_manifest: dict[str, Any],
    state_dict_path: Path,
    source_runtime_dir: Path,
    source_archive_record: dict[str, Any],
    source_member_name: str,
    source_decoder_blob: bytes,
    source_latent_blob: bytes,
    source_sidecar_blob: bytes,
    variant_dir: Path,
    recorded_at_utc: dt.datetime,
    brotli_quality: int,
) -> dict[str, Any]:
    packet_dir = variant_dir / "packet"
    packet_dir.mkdir(parents=True)
    runtime_files = _copy_runtime_tree(source_runtime_dir, packet_dir)
    runtime_patch = _patch_packet_codec(packet_dir)

    selected_ks = list(schedule["selected_Ks"])
    decoder_blob, encoding = _encode_rounded_decoder_blob(
        state_dict_path=state_dict_path,
        selected_ks=selected_ks,
        brotli_quality=brotli_quality,
    )
    decoded = _decode_decoder_blob_for_closure(decoder_blob)
    if len(decoded) != len(FIXED_STATE_SCHEMA):
        raise PacketClosureBlocked(
            f"decoder roundtrip returned {len(decoded)} tensors, expected {len(FIXED_STATE_SCHEMA)}",
            ["a2_decoder_blob_roundtrip_tensor_count_mismatch"],
        )

    inner_member = _build_a2_inner_member(decoder_blob, source_latent_blob, source_sidecar_blob)
    candidate_archive_path = packet_dir / "archive.zip"
    candidate_archive_record = _write_single_member_zip(
        candidate_archive_path,
        source_member_name,
        inner_member,
    )
    parse_smoke = _packet_local_parse_smoke(packet_dir, candidate_archive_path)

    source_member_sha = source_archive_record["members"][0]["sha256"]
    candidate_member_sha = candidate_archive_record["members"][0]["sha256"]
    noop = _noop_detection(
        selected_ks=selected_ks,
        source_decoder_sha256=_sha256_bytes(source_decoder_blob),
        candidate_decoder_sha256=encoding["decoder_blob_sha256"],
        source_member_sha256=source_member_sha,
        candidate_member_sha256=candidate_member_sha,
        source_archive_sha256=source_archive_record["sha256"],
        candidate_archive_sha256=candidate_archive_record["sha256"],
    )
    semantic_payload_changed = (
        encoding["decoder_blob_sha256"] != _sha256_bytes(source_decoder_blob)
        and not noop["schedule_all_ones"]
    )
    dispatch_blockers = _active_packet_blockers(
        BASE_DISPATCH_BLOCKERS,
        upstream_a2_manifest["propagated_blockers"],
        noop["reasons"],
        clear=CLEARED_BY_PACKET_LADDER,
    )
    report_record = _write_packet_report(
        packet_dir=packet_dir,
        variant_id=schedule["variant_id"],
        candidate_archive_record=candidate_archive_record,
        source_archive_record=source_archive_record,
        dispatch_blockers=dispatch_blockers,
    )
    runtime_files_after_patch = [
        _file_record(path, relpath=path.relative_to(packet_dir).as_posix())
        for path in sorted(
            packet_dir.rglob("*"), key=lambda item: item.relative_to(packet_dir).as_posix()
        )
        if path.is_file()
        and not _is_packet_custody_file(path.relative_to(packet_dir))
        and not _should_exclude(path.relative_to(packet_dir))
    ]
    runtime_tree_sha256 = _runtime_tree_sha256(runtime_files_after_patch)
    report_record = _write_packet_report(
        packet_dir=packet_dir,
        variant_id=schedule["variant_id"],
        candidate_archive_record=candidate_archive_record,
        source_archive_record=source_archive_record,
        dispatch_blockers=dispatch_blockers,
        runtime_tree_sha256=runtime_tree_sha256,
    )
    proxy_vs_materialized = _proxy_vs_materialized(schedule, candidate_archive_record, decoder_blob)
    variant = {
        "schema": VARIANT_SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "variant_id": schedule["variant_id"],
        "created_utc": _format_utc(recorded_at_utc),
        **FALSE_AUTHORITY_FIELDS,
        "dispatch_attempted": False,
        "remote_dispatch_allowed": False,
        "remote_gpu_run": False,
        "evidence_grade": "empirical",
        "evidence_semantics": "byte_closed_archive_runtime_packet_no_score_no_dispatch",
        "target_modes": ["contest_exact_eval"],
        "deployment_target": "t4_contest_runtime_pending_exact_eval",
        "dispatch_blockers": dispatch_blockers,
        "packet_closure": {
            "byte_closed_packet_built": True,
            "runtime_consumes_changed_archive_bytes": bool(
                parse_smoke.get("passed") and candidate_member_sha != source_member_sha
            ),
            "runtime_source_mutated": False,
            "wire_contract": "A2K1 + decoder_len:u32le + PR101 decoder_blob + PR101 latent_blob + sidecar",
            "missing_wire_contracts": [],
            "cleared_blockers": sorted(CLEARED_BY_PACKET_LADDER),
        },
        "upstream_a2_manifest": upstream_a2_manifest,
        "schedule": {
            **schedule,
            "selected_Ks_sha256": _canonical_json_sha256(selected_ks),
            "selected_Ks_count": len(selected_ks),
        },
        "source_archive": source_archive_record,
        "candidate_archive": candidate_archive_record,
        "candidate_archive_relpath": _repo_rel(candidate_archive_path),
        "archive_member_manifest": {
            "member_name": source_member_name,
            "layout_magic": MAGIC.decode("ascii"),
            "header_bytes": 8,
            "decoder_len_field": len(decoder_blob),
            "decoder_len_field_matches_decoder_blob": True,
            "decoder_blob_offset": 8,
            "latent_blob_offset": 8 + len(decoder_blob),
            "latent_blob_bytes": len(source_latent_blob),
            "sidecar_blob_offset": 8 + len(decoder_blob) + len(source_latent_blob),
            "sidecar_blob_bytes": len(source_sidecar_blob),
            "source_latent_sha256": _sha256_bytes(source_latent_blob),
            "candidate_latent_sha256": _sha256_bytes(source_latent_blob),
            "source_sidecar_sha256": _sha256_bytes(source_sidecar_blob),
            "candidate_sidecar_sha256": _sha256_bytes(source_sidecar_blob),
        },
        "decoder_materialization": {
            **encoding,
            "source_decoder_blob_bytes": len(source_decoder_blob),
            "source_decoder_blob_sha256": _sha256_bytes(source_decoder_blob),
            "candidate_decoder_blob_sha256": encoding["decoder_blob_sha256"],
            "candidate_decoder_blob_bytes": len(decoder_blob),
            "decode_roundtrip": {
                "attempted": True,
                "passed": True,
                "decoded_tensor_count": len(decoded),
            },
        },
        "runtime_packet": {
            "packet_dir": _repo_rel(packet_dir),
            "runtime_patch": runtime_patch,
            "runtime_custody": {
                "copied_file_count": len(runtime_files),
                "runtime_tree_sha256": runtime_tree_sha256,
                "runtime_files": runtime_files_after_patch,
                "excluded_packet_custody_filenames": sorted(PACKET_CUSTODY_FILENAMES),
                "excluded_dir_names": sorted(EXCLUDED_DIR_NAMES),
                "excluded_file_names": sorted(EXCLUDED_FILE_NAMES),
                "excluded_suffixes": list(EXCLUDED_SUFFIXES),
            },
            "runtime_checks": {
                "inflate_sh_bash_n": _bash_n(packet_dir / "inflate.sh"),
                "packet_local_parse_smoke": parse_smoke,
            },
            "report": report_record,
        },
        "proxy_vs_materialized": proxy_vs_materialized,
        "noop_detection": noop,
        "charged_bits_changed": candidate_member_sha != source_member_sha,
        "score_affecting_payload_changed": semantic_payload_changed,
        "semantic_payload_changed": semantic_payload_changed,
    }
    variant["manifest_sha256_excluding_self"] = _canonical_json_sha256(variant)
    write_json(variant_dir / "candidate_manifest.json", variant)
    return variant


def build_packet_ladder(
    *,
    a2_manifest_path: Path,
    state_dict_path: Path,
    source_archive: Path,
    source_runtime_dir: Path,
    output_dir: Path,
    recorded_at_utc: dt.datetime,
    brotli_quality: int = 11,
    variant_limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    a2_manifest_path = _repo_path(a2_manifest_path)
    state_dict_path = _repo_path(state_dict_path)
    source_archive = _repo_path(source_archive)
    source_runtime_dir = _repo_path(source_runtime_dir)
    output_dir = _repo_path(output_dir)

    if not a2_manifest_path.is_file():
        raise PacketClosureBlocked(
            f"A2 manifest not found: {a2_manifest_path}",
            ["a2_manifest_missing"],
        )
    if not source_archive.is_file():
        raise PacketClosureBlocked(
            f"source archive not found: {source_archive}",
            ["source_archive_missing"],
        )
    if not source_runtime_dir.is_dir():
        raise PacketClosureBlocked(
            f"source runtime directory not found: {source_runtime_dir}",
            ["source_runtime_dir_missing"],
        )
    _assert_output_dir_isolated(
        output_dir=output_dir,
        a2_manifest_path=a2_manifest_path,
        state_dict_path=state_dict_path,
        source_archive=source_archive,
        source_runtime_dir=source_runtime_dir,
    )
    _prepare_empty_dir(output_dir, force=force)

    a2_manifest = read_json(a2_manifest_path)
    if not isinstance(a2_manifest, dict):
        raise PacketClosureBlocked(
            f"A2 manifest is not a JSON object: {a2_manifest_path}",
            ["a2_manifest_not_object"],
        )
    upstream_a2_manifest = _validate_a2_manifest_authority(
        a2_manifest,
        manifest_path=a2_manifest_path,
    )
    schedules = _extract_weighted_schedules(a2_manifest, variant_limit=variant_limit)
    source_archive_record, source_member_name, source_member_payload = _read_single_member_zip(source_archive)
    source_decoder_blob, source_latent_blob, source_sidecar_blob = _split_pr101_member(source_member_payload)
    state_source_closure = _verify_state_dict_matches_source_decoder(
        state_dict_path=state_dict_path,
        source_decoder_blob=source_decoder_blob,
        brotli_quality=brotli_quality,
    )

    variants: list[dict[str, Any]] = []
    variants_dir = output_dir / "variants"
    for schedule in schedules:
        variants.append(
            _build_variant(
                schedule=schedule,
                upstream_a2_manifest=upstream_a2_manifest,
                state_dict_path=state_dict_path,
                source_runtime_dir=source_runtime_dir,
                source_archive_record=source_archive_record,
                source_member_name=source_member_name,
                source_decoder_blob=source_decoder_blob,
                source_latent_blob=source_latent_blob,
                source_sidecar_blob=source_sidecar_blob,
                variant_dir=variants_dir / schedule["variant_id"],
                recorded_at_utc=recorded_at_utc,
                brotli_quality=brotli_quality,
            )
        )

    manifest = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "created_utc": _format_utc(recorded_at_utc),
        "status": "completed_byte_closed_packet_ladder",
        "phase": "A2",
        "decision": "A2",
        "lane_id": "track1_phase_a2_sensitivity_quant_packet_ladder",
        **FALSE_AUTHORITY_FIELDS,
        "dispatch_attempted": False,
        "remote_dispatch_allowed": False,
        "remote_gpu_run": False,
        "evidence_grade": "empirical",
        "evidence_semantics": "byte_closed_packet_ladder_no_score_no_dispatch",
        "target_modes": ["contest_exact_eval"],
        "deployment_target": "t4_contest_runtime_pending_exact_eval",
        "dispatch_blockers": _active_packet_blockers(
            BASE_DISPATCH_BLOCKERS,
            upstream_a2_manifest["propagated_blockers"],
            clear=CLEARED_BY_PACKET_LADDER,
        ),
        "upstream_a2_manifest": upstream_a2_manifest,
        "inputs": {
            "a2_manifest": _repo_rel(a2_manifest_path),
            "a2_manifest_sha256": sha256_file(a2_manifest_path),
            "state_dict": _repo_rel(state_dict_path),
            "state_dict_sha256": sha256_file(state_dict_path),
            "source_archive": _repo_rel(source_archive),
            "source_archive_sha256": sha256_file(source_archive),
            "source_runtime_dir": _repo_rel(source_runtime_dir),
        },
        "output_dir": _repo_rel(output_dir),
        "packet_closure": {
            "byte_closed_packet_ladder_built": True,
            "variant_count": len(variants),
            "missing_wire_contracts": [],
            "runtime_source_mutated": False,
            "state_dict_reproduces_source_decoder": state_source_closure,
            "cleared_blockers": sorted(CLEARED_BY_PACKET_LADDER),
        },
        "variants": variants,
        "next_required_actions": [
            "run local source-vs-candidate inflate parity on selected variants",
            "run pre-submission compliance on a reviewed packet surface",
            "claim Level-2 lane before any exact eval dispatch",
            "run exact CUDA and contest-CPU auth eval before any score or promotion claim",
        ],
    }
    manifest["manifest_sha256_excluding_self"] = _canonical_json_sha256(manifest)
    write_json(output_dir / "a2_packet_ladder_manifest.json", manifest)
    return manifest


def _blocked_manifest(
    *,
    reason: str,
    missing_wire_contracts: list[str],
    args: argparse.Namespace,
    recorded_at_utc: dt.datetime,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "created_utc": _format_utc(recorded_at_utc),
        "status": "blocked_fail_closed",
        "phase": "A2",
        "decision": "A2",
        "lane_id": "track1_phase_a2_sensitivity_quant_packet_ladder",
        **FALSE_AUTHORITY_FIELDS,
        "dispatch_attempted": False,
        "remote_dispatch_allowed": False,
        "remote_gpu_run": False,
        "evidence_grade": "blocked",
        "evidence_semantics": "no_score_no_dispatch",
        "reason": reason,
        "inputs": {
            "a2_manifest": repo_relative(args.a2_manifest, REPO_ROOT),
            "state_dict": repo_relative(args.state_dict, REPO_ROOT),
            "source_archive": repo_relative(args.source_archive, REPO_ROOT),
            "source_runtime_dir": repo_relative(args.source_runtime_dir, REPO_ROOT),
        },
        "packet_closure": {
            "byte_closed_packet_ladder_built": False,
            "missing_wire_contracts": sorted(set(missing_wire_contracts)),
            "runtime_source_mutated": False,
        },
        "dispatch_blockers": sorted(
            {*BASE_DISPATCH_BLOCKERS, "packet_closure_blocked", *missing_wire_contracts}
        ),
        "charged_bits_changed": False,
        "score_affecting_payload_changed": False,
        "semantic_payload_changed": False,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a2-manifest", type=Path, required=True)
    parser.add_argument("--state-dict", type=Path, default=DEFAULT_STATE_DICT)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--source-runtime-dir", type=Path, default=DEFAULT_SOURCE_RUNTIME_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument("--variant-limit", type=int, default=0)
    parser.add_argument("--now-utc")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-if-blocked", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    recorded_at_utc = _parse_utc(args.now_utc)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_ROOT / _utc_ts()
    json_out = args.json_out
    if json_out is None:
        json_out = output_dir / "a2_packet_ladder_manifest.json"

    try:
        manifest = build_packet_ladder(
            a2_manifest_path=args.a2_manifest,
            state_dict_path=args.state_dict,
            source_archive=args.source_archive,
            source_runtime_dir=args.source_runtime_dir,
            output_dir=output_dir,
            recorded_at_utc=recorded_at_utc,
            brotli_quality=args.brotli_quality,
            variant_limit=args.variant_limit if args.variant_limit > 0 else None,
            force=args.force,
        )
        if _repo_path(json_out) != _repo_path(output_dir) / "a2_packet_ladder_manifest.json":
            write_json(_repo_path(json_out), manifest)
        print(f"manifest: {repo_relative(_repo_path(json_out), REPO_ROOT)}")
        print(f"variants: {len(manifest['variants'])}")
        return 0
    except PacketClosureBlocked as exc:
        manifest = _blocked_manifest(
            reason=str(exc),
            missing_wire_contracts=exc.missing_wire_contracts,
            args=args,
            recorded_at_utc=recorded_at_utc,
        )
        write_json(_repo_path(json_out), manifest)
        print(f"blocked manifest: {repo_relative(_repo_path(json_out), REPO_ROOT)}")
        print(f"reason: {exc}")
        return 1 if args.fail_if_blocked else 2
    except Exception as exc:
        manifest = _blocked_manifest(
            reason=str(exc),
            missing_wire_contracts=["unexpected_packet_builder_exception"],
            args=args,
            recorded_at_utc=recorded_at_utc,
        )
        write_json(_repo_path(json_out), manifest)
        print(f"blocked manifest: {repo_relative(_repo_path(json_out), REPO_ROOT)}")
        print(f"reason: {exc}")
        return 1 if args.fail_if_blocked else 2


if __name__ == "__main__":
    raise SystemExit(main())
