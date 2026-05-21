#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a single-member HFV3 embedded sidecar candidate from dense HFV1.

HFV2 proved the dense HFV1 foveation table is pair-sparse. For the current
PR101/FEC6 hardpair candidate, all non-default sparse pairs share the same
five-float row. HFV3 encodes that exact repeated-row structure and embeds the
payload at the end of member ``x`` so the archive stays single-member and avoids
the second ZIP member overhead.

This writes research artifacts only and does not claim a score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import struct
import sys
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.deploy.modal.paired_dispatch import paired_auth_eval_dispatch_command_template

CONTEST_DENOM_BYTES = 37_545_489
RATE_MULTIPLIER = 25.0
CAMERA_H = 874
CAMERA_W = 1164
HFV1_HEADER = struct.Struct("<4sIII")
HFV1_ROW = struct.Struct("<fffff")
HFV3_HEADER = struct.Struct("<4sIIIfffffHfffff")
HFV3_PAIR_INDEX = struct.Struct("<H")
DEFAULT_DENSE_ARCHIVE = Path(
    "experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/"
    "archive_seed_top16_component_hardpairs/archive.zip"
)
DEFAULT_SOURCE_SUBMISSION_DIR = Path(
    "experiments/results/hfv2_sparse_sidecar_candidate_20260521T070412Z/"
    "submission_dir_hfv2"
)
DEFAULT_BASELINE_ARCHIVE = Path(
    "experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/"
    "submission_dir/archive.zip"
)


@dataclass(frozen=True)
class Hfv3EmbeddedManifest:
    schema: str
    generated_at_utc: str
    source_dense_archive: str
    source_dense_archive_bytes: int
    source_dense_archive_sha256: str
    baseline_archive: str
    baseline_archive_bytes: int
    baseline_archive_sha256: str
    output_archive: str
    output_archive_bytes: int
    output_archive_sha256: str
    output_submission_dir: str
    output_submission_archive: str
    output_inflate_py_sha256: str
    output_archive_manifest: str
    output_archive_manifest_sha256: str
    source_hfv1_payload_bytes: int
    source_hfv1_payload_sha256: str
    output_hfv3_payload_bytes: int
    output_hfv3_payload_sha256: str
    original_x_member_bytes: int
    original_x_member_sha256: str
    embedded_x_member_bytes: int
    embedded_x_member_sha256: str
    default_row: tuple[float, float, float, float, float]
    repeated_active_row: tuple[float, float, float, float, float]
    sparse_pair_count: int
    sparse_pairs: list[int]
    row_parity_exact: bool
    bytes_saved_vs_dense_archive: int
    bytes_saved_vs_hfv2_archive: int
    bytes_delta_vs_baseline_archive: int
    rate_delta_vs_dense_archive: float
    rate_delta_vs_hfv2_archive: float
    rate_delta_vs_baseline_archive: float
    target_modes: list[str]
    dispatch_blockers: list[str]
    paired_auth_eval_required: bool
    paired_auth_eval_plan_ready: bool
    paired_modal_auth_eval_plan_command_template_not_run: list[str]
    paired_modal_auth_eval_execute_command_template_after_claim_surface_clear: list[str]
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _rate_delta(byte_delta: int) -> float:
    return RATE_MULTIPLIER * float(byte_delta) / float(CONTEST_DENOM_BYTES)


def _read_dense_hfv1(raw: bytes) -> tuple[int, int, int, list[tuple[float, float, float, float, float]]]:
    if len(raw) < HFV1_HEADER.size:
        raise ValueError("HFV1 payload truncated before header")
    magic, n_frames, frame_height, frame_width = HFV1_HEADER.unpack_from(raw)
    if magic != b"HFV1":
        raise ValueError(f"HFV1 magic mismatch: {magic!r}")
    expected = HFV1_HEADER.size + int(n_frames) * HFV1_ROW.size
    if len(raw) != expected:
        raise ValueError(f"HFV1 payload size mismatch: got {len(raw)}, expected {expected}")
    rows = [
        HFV1_ROW.unpack_from(raw, HFV1_HEADER.size + index * HFV1_ROW.size)
        for index in range(int(n_frames))
    ]
    return int(n_frames), int(frame_height), int(frame_width), rows


def _repeated_sparse_pairs(
    rows: list[tuple[float, float, float, float, float]]
) -> tuple[
    tuple[float, float, float, float, float],
    tuple[float, float, float, float, float],
    list[int],
]:
    if len(rows) % 2:
        raise ValueError("HFV3 repeated-pair conversion requires an even frame count")
    default_row = Counter(rows).most_common(1)[0][0]
    active_rows: list[tuple[float, float, float, float, float]] = []
    active_pairs: list[int] = []
    for pair_index in range(len(rows) // 2):
        row0 = rows[2 * pair_index]
        row1 = rows[2 * pair_index + 1]
        if row0 == default_row and row1 == default_row:
            continue
        if row0 != row1:
            raise ValueError(f"pair {pair_index} rows differ; not HFV3 repeated-pair encodable")
        active_pairs.append(pair_index)
        active_rows.append(row0)
    unique_active = sorted(set(active_rows))
    if len(unique_active) != 1:
        raise ValueError(f"HFV3 repeated-row requires one active row, got {len(unique_active)}")
    return default_row, unique_active[0], active_pairs


def _pack_hfv3(
    *,
    n_frames: int,
    frame_height: int,
    frame_width: int,
    default_row: tuple[float, float, float, float, float],
    active_row: tuple[float, float, float, float, float],
    active_pairs: list[int],
) -> bytes:
    if len(active_pairs) > 65535:
        raise ValueError("too many HFV3 active pairs")
    out = bytearray()
    out.extend(
        HFV3_HEADER.pack(
            b"HFV3",
            n_frames,
            frame_height,
            frame_width,
            *default_row,
            len(active_pairs),
            *active_row,
        )
    )
    for pair_index in active_pairs:
        if not 0 <= pair_index <= 65535:
            raise ValueError(f"HFV3 pair index out of uint16 range: {pair_index}")
        out.extend(HFV3_PAIR_INDEX.pack(pair_index))
    return bytes(out)


def _decode_hfv3_rows(raw: bytes) -> list[tuple[float, float, float, float, float]]:
    if len(raw) < HFV3_HEADER.size:
        raise ValueError("HFV3 payload truncated before header")
    unpacked = HFV3_HEADER.unpack_from(raw)
    magic, n_frames, frame_height, frame_width = unpacked[:4]
    if magic != b"HFV3":
        raise ValueError(f"HFV3 magic mismatch: {magic!r}")
    if int(frame_height) != CAMERA_H or int(frame_width) != CAMERA_W:
        raise ValueError("HFV3 image size mismatch")
    default_row = tuple(float(value) for value in unpacked[4:9])
    pair_count = int(unpacked[9])
    active_row = tuple(float(value) for value in unpacked[10:15])
    expected = HFV3_HEADER.size + pair_count * HFV3_PAIR_INDEX.size
    if len(raw) != expected:
        raise ValueError(f"HFV3 payload size mismatch: got {len(raw)}, expected {expected}")
    rows = [default_row for _ in range(int(n_frames))]
    offset = HFV3_HEADER.size
    for _ in range(pair_count):
        (pair_index,) = HFV3_PAIR_INDEX.unpack_from(raw, offset)
        offset += HFV3_PAIR_INDEX.size
        rows[2 * int(pair_index)] = active_row
        rows[2 * int(pair_index) + 1] = active_row
    return rows


def _patch_inflate_py(source: str) -> str:
    if "HFV3_MAGIC" in source:
        return source
    source = source.replace(
        'HFV2_MAGIC = b"HFV2"\n',
        'HFV2_MAGIC = b"HFV2"\nHFV3_MAGIC = b"HFV3"\n',
    )
    source = source.replace(
        'HFV2_PAIR_ROW_STRUCT = struct.Struct("<Hfffff")\n',
        'HFV2_PAIR_ROW_STRUCT = struct.Struct("<Hfffff")\n'
        'HFV3_HEADER_STRUCT = struct.Struct("<4sIIIfffffHfffff")\n'
        'HFV3_PAIR_INDEX_STRUCT = struct.Struct("<H")\n',
    )
    source = source.replace(
        "def parse_pr101_frame_selector_archive(\n"
        "    bin_bytes: bytes,\n"
        ") -> tuple[bytes, str, list[int], tuple[tuple[str, tuple[int, ...], int], ...]]:\n",
        "def embedded_foveation_payload_from_wrapped_x(bin_bytes: bytes) -> bytes | None:\n"
        "    if len(bin_bytes) < 10:\n"
        "        raise ValueError(\"PR101 frame-selector wrapper truncated before header\")\n"
        "    magic = bin_bytes[:4]\n"
        "    if magic != OUTER_MAGIC:\n"
        "        raise ValueError(f\"PR101 frame-selector magic mismatch: {magic!r}\")\n"
        "    pos = 4\n"
        "    (source_len,) = struct.unpack_from(\"<I\", bin_bytes, pos)\n"
        "    pos += 4 + int(source_len)\n"
        "    if pos > len(bin_bytes):\n"
        "        raise ValueError(\"PR101 source payload truncated\")\n"
        "    if pos + 2 > len(bin_bytes):\n"
        "        raise ValueError(\"PR101 frame-selector wrapper truncated before selector length\")\n"
        "    (selector_len,) = struct.unpack_from(\"<H\", bin_bytes, pos)\n"
        "    pos += 2 + int(selector_len)\n"
        "    if pos > len(bin_bytes):\n"
        "        raise ValueError(\"FES1 selector payload truncated\")\n"
        "    if pos == len(bin_bytes):\n"
        "        return None\n"
        "    trailer = bin_bytes[pos:]\n"
        "    if trailer.startswith(HFV3_MAGIC):\n"
        "        return trailer\n"
        "    raise ValueError(f\"PR101 frame-selector trailing bytes: pos={pos} total={len(bin_bytes)}\")\n\n\n"
        "def parse_pr101_frame_selector_archive(\n"
        "    bin_bytes: bytes,\n"
        ") -> tuple[bytes, str, list[int], tuple[tuple[str, tuple[int, ...], int], ...]]:\n",
    )
    source = source.replace(
        "    if pos != len(bin_bytes):\n"
        "        raise ValueError(f\"PR101 frame-selector trailing bytes: pos={pos} total={len(bin_bytes)}\")\n",
        "    if pos != len(bin_bytes):\n"
        "        trailer = embedded_foveation_payload_from_wrapped_x(bin_bytes)\n"
        "        if trailer is None:\n"
        "            raise ValueError(f\"PR101 frame-selector trailing bytes: pos={pos} total={len(bin_bytes)}\")\n",
    )
    marker = "\n\ndef load_foveation_sidecar(src_bin: Path) -> dict[str, object] | None:\n"
    insert = r'''

def load_hfv3_repeat_params_from_bytes(raw: bytes) -> dict[str, object]:
    if len(raw) < HFV3_HEADER_STRUCT.size:
        raise ValueError("HFV3 foveation params truncated before header")
    unpacked = HFV3_HEADER_STRUCT.unpack(raw[: HFV3_HEADER_STRUCT.size])
    magic, n_frames, frame_height, frame_width = unpacked[:4]
    if magic != HFV3_MAGIC:
        raise ValueError(f"HFV3 foveation params magic mismatch: {magic!r}")
    if int(frame_height) != CAMERA_H or int(frame_width) != CAMERA_W:
        raise ValueError(
            "HFV3 image size does not match PR101 output: "
            f"{(frame_height, frame_width)} vs {(CAMERA_H, CAMERA_W)}"
        )
    default_row = tuple(float(value) for value in unpacked[4:9])
    pair_count = int(unpacked[9])
    active_row = tuple(float(value) for value in unpacked[10:15])
    expected = HFV3_HEADER_STRUCT.size + pair_count * HFV3_PAIR_INDEX_STRUCT.size
    if len(raw) != expected:
        raise ValueError(f"HFV3 foveation params size mismatch: got {len(raw)}, expected {expected}")
    row_by_frame: dict[int, tuple[float, float, float, float, float]] = {}
    for index in range(pair_count):
        (pair_index,) = HFV3_PAIR_INDEX_STRUCT.unpack_from(
            raw,
            HFV3_HEADER_STRUCT.size + index * HFV3_PAIR_INDEX_STRUCT.size,
        )
        row_by_frame[2 * int(pair_index)] = active_row
        row_by_frame[2 * int(pair_index) + 1] = active_row
    return {
        "source_format": "HFV3_repeated_pair_sparse",
        "n_frames": int(n_frames),
        "default_row": default_row,
        "row_by_frame": row_by_frame,
        "pair_count": pair_count,
    }
'''
    source = source.replace(marker, insert + marker)
    source = source.replace(
        "    lfv1 = src_bin.with_name(\"lapose_foveation_tuples.lfv1\")\n"
        "    if lfv1.is_file():\n"
        "        return load_lfv1_sparse_params(lfv1)\n"
        "    return None\n",
        "    lfv1 = src_bin.with_name(\"lapose_foveation_tuples.lfv1\")\n"
        "    if lfv1.is_file():\n"
        "        return load_lfv1_sparse_params(lfv1)\n"
        "    trailer = embedded_foveation_payload_from_wrapped_x(src_bin.read_bytes())\n"
        "    if trailer is None:\n"
        "        return None\n"
        "    if trailer.startswith(HFV3_MAGIC):\n"
        "        return load_hfv3_repeat_params_from_bytes(trailer)\n"
        "    raise ValueError(f\"unsupported embedded foveation magic: {trailer[:4]!r}\")\n",
    )
    source = source.replace(
        '    if params.get("source_format") in {"LFV1_sparse", "HFV2_pair_sparse"}:\n',
        '    if params.get("source_format") in {"LFV1_sparse", "HFV2_pair_sparse", "HFV3_repeated_pair_sparse"}:\n',
    )
    if "HFV3_repeated_pair_sparse" not in source:
        raise ValueError("failed to patch inflate.py with HFV3 decoder")
    return source


def _copy_runtime(source_dir: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(
        source_dir,
        output_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    inflate_py = output_dir / "inflate.py"
    inflate_py.write_text(_patch_inflate_py(inflate_py.read_text(encoding="utf-8")), encoding="utf-8")
    return output_dir


def _write_stored_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)


def _zip_member_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            payload = archive.read(info.filename)
            records.append(
                {
                    "name": info.filename,
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "crc": int(info.CRC),
                    "compression_method": "stored"
                    if info.compress_type == zipfile.ZIP_STORED
                    else str(info.compress_type),
                    "sha256": _sha256_bytes(payload),
                }
            )
    return records


def _write_archive_manifest(
    *,
    submission_dir: Path,
    archive_path: Path,
    dense_archive: Path,
    baseline_archive: Path,
    hfv1_raw: bytes,
    hfv3_raw: bytes,
    active_pairs: list[int],
    row_parity_exact: bool,
    paired_plan_cmd: list[str],
    paired_execute_cmd: list[str],
) -> Path:
    manifest_path = submission_dir / "archive_manifest.json"
    payload = {
        "schema": "hfv3_embedded_repeated_pair_sidecar_candidate_manifest_v1",
        "generated_at_utc": _utc_iso(),
        "archive_path": _repo_rel(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256_file(archive_path),
        "members": _zip_member_records(archive_path),
        "member_shape": "single_member_x_with_pr101_source_payload_fec6_selector_and_embedded_hfv3_repeated_pair_sidecar",
        "source_dense_archive": _repo_rel(dense_archive),
        "source_dense_archive_bytes": dense_archive.stat().st_size,
        "source_dense_archive_sha256": _sha256_file(dense_archive),
        "baseline_archive": _repo_rel(baseline_archive),
        "baseline_archive_bytes": baseline_archive.stat().st_size,
        "baseline_archive_sha256": _sha256_file(baseline_archive),
        "source_hfv1_payload_bytes": len(hfv1_raw),
        "source_hfv1_payload_sha256": _sha256_bytes(hfv1_raw),
        "output_hfv3_payload_bytes": len(hfv3_raw),
        "output_hfv3_payload_sha256": _sha256_bytes(hfv3_raw),
        "sparse_pair_count": len(active_pairs),
        "sparse_pairs": active_pairs,
        "row_parity_exact": row_parity_exact,
        "target_modes": ["contest_exact_eval"],
        "dispatch_blockers": [
            "exact_contest_cpu_eval_missing",
            "exact_contest_cuda_eval_missing",
            "lane_dispatch_claim_required_before_execute",
        ],
        "paired_auth_eval_required": True,
        "paired_auth_eval_plan_ready": row_parity_exact,
        "paired_modal_auth_eval_plan_command_template_not_run": paired_plan_cmd,
        "paired_modal_auth_eval_execute_command_template_after_claim_surface_clear": paired_execute_cmd,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_submission": False,
        "ready_for_exact_eval_dispatch": False,
        "submission_blocker": (
            "Research candidate only. Requires exact contest eval plus fresh strict "
            "pre-submission compliance before any promotion or submission claim."
        ),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def build_candidate(
    *,
    dense_archive: Path,
    source_submission_dir: Path,
    baseline_archive: Path,
    hfv2_archive: Path,
    output_dir: Path,
) -> Hfv3EmbeddedManifest:
    with zipfile.ZipFile(dense_archive) as archive:
        hfv1_raw = archive.read("foveation_params.bin")
        x_payload = archive.read("x")
    n_frames, frame_height, frame_width, rows = _read_dense_hfv1(hfv1_raw)
    if frame_height != CAMERA_H or frame_width != CAMERA_W:
        raise ValueError("HFV1 image size mismatch")
    default_row, active_row, active_pairs = _repeated_sparse_pairs(rows)
    hfv3_raw = _pack_hfv3(
        n_frames=n_frames,
        frame_height=frame_height,
        frame_width=frame_width,
        default_row=default_row,
        active_row=active_row,
        active_pairs=active_pairs,
    )
    row_parity_exact = _decode_hfv3_rows(hfv3_raw) == rows
    if not row_parity_exact:
        raise ValueError("HFV3 row parity check failed")
    embedded_x = x_payload + hfv3_raw
    archive_path = output_dir / "archive.zip"
    _write_stored_zip(archive_path, [("x", embedded_x)])
    submission_dir = _copy_runtime(source_submission_dir, output_dir / "submission_dir_hfv3_embedded")
    submission_archive = submission_dir / "archive.zip"
    shutil.copy2(archive_path, submission_archive)

    lane_id_base = "hfv3_embedded_pr101_hfv1_sidecar_exact_eval"
    run_id = f"hfv3_embedded_pr101_{_sha256_file(submission_archive)[:12]}"
    claim_notes = (
        "HFV3 embedded repeated-row sidecar candidate; score_claim=false until paired "
        "contest CPU/CUDA exact eval harvest."
    )
    paired_plan_cmd = paired_auth_eval_dispatch_command_template(
        archive_path=_repo_rel(submission_archive),
        submission_dir=_repo_rel(submission_dir),
        lane_id_base=lane_id_base,
        archive_sha256=_sha256_file(submission_archive),
        execute=False,
        label=lane_id_base,
        run_id=run_id,
        claim_agent="codex:hfv3_embedded_sidecar",
        claim_notes=claim_notes,
    )
    paired_execute_cmd = paired_auth_eval_dispatch_command_template(
        archive_path=_repo_rel(submission_archive),
        submission_dir=_repo_rel(submission_dir),
        lane_id_base=lane_id_base,
        archive_sha256=_sha256_file(submission_archive),
        execute=True,
        label=lane_id_base,
        run_id=run_id,
        claim_agent="codex:hfv3_embedded_sidecar",
        claim_notes=claim_notes,
    )
    generated_manifest = _write_archive_manifest(
        submission_dir=submission_dir,
        archive_path=submission_archive,
        dense_archive=dense_archive,
        baseline_archive=baseline_archive,
        hfv1_raw=hfv1_raw,
        hfv3_raw=hfv3_raw,
        active_pairs=active_pairs,
        row_parity_exact=row_parity_exact,
        paired_plan_cmd=paired_plan_cmd,
        paired_execute_cmd=paired_execute_cmd,
    )
    return Hfv3EmbeddedManifest(
        schema="hfv3_embedded_repeated_pair_sidecar_candidate_v1",
        generated_at_utc=_utc_iso(),
        source_dense_archive=_repo_rel(dense_archive),
        source_dense_archive_bytes=dense_archive.stat().st_size,
        source_dense_archive_sha256=_sha256_file(dense_archive),
        baseline_archive=_repo_rel(baseline_archive),
        baseline_archive_bytes=baseline_archive.stat().st_size,
        baseline_archive_sha256=_sha256_file(baseline_archive),
        output_archive=_repo_rel(archive_path),
        output_archive_bytes=archive_path.stat().st_size,
        output_archive_sha256=_sha256_file(archive_path),
        output_submission_dir=_repo_rel(submission_dir),
        output_submission_archive=_repo_rel(submission_archive),
        output_inflate_py_sha256=_sha256_file(submission_dir / "inflate.py"),
        output_archive_manifest=_repo_rel(generated_manifest),
        output_archive_manifest_sha256=_sha256_file(generated_manifest),
        source_hfv1_payload_bytes=len(hfv1_raw),
        source_hfv1_payload_sha256=_sha256_bytes(hfv1_raw),
        output_hfv3_payload_bytes=len(hfv3_raw),
        output_hfv3_payload_sha256=_sha256_bytes(hfv3_raw),
        original_x_member_bytes=len(x_payload),
        original_x_member_sha256=_sha256_bytes(x_payload),
        embedded_x_member_bytes=len(embedded_x),
        embedded_x_member_sha256=_sha256_bytes(embedded_x),
        default_row=default_row,
        repeated_active_row=active_row,
        sparse_pair_count=len(active_pairs),
        sparse_pairs=active_pairs,
        row_parity_exact=row_parity_exact,
        bytes_saved_vs_dense_archive=dense_archive.stat().st_size - archive_path.stat().st_size,
        bytes_saved_vs_hfv2_archive=hfv2_archive.stat().st_size - archive_path.stat().st_size,
        bytes_delta_vs_baseline_archive=archive_path.stat().st_size - baseline_archive.stat().st_size,
        rate_delta_vs_dense_archive=_rate_delta(archive_path.stat().st_size - dense_archive.stat().st_size),
        rate_delta_vs_hfv2_archive=_rate_delta(archive_path.stat().st_size - hfv2_archive.stat().st_size),
        rate_delta_vs_baseline_archive=_rate_delta(archive_path.stat().st_size - baseline_archive.stat().st_size),
        target_modes=["contest_exact_eval"],
        dispatch_blockers=[
            "exact_contest_cpu_eval_missing",
            "exact_contest_cuda_eval_missing",
            "lane_dispatch_claim_required_before_execute",
        ],
        paired_auth_eval_required=True,
        paired_auth_eval_plan_ready=row_parity_exact,
        paired_modal_auth_eval_plan_command_template_not_run=paired_plan_cmd,
        paired_modal_auth_eval_execute_command_template_after_claim_surface_clear=paired_execute_cmd,
    )


def render_markdown(manifest: Hfv3EmbeddedManifest) -> str:
    return "\n".join(
        [
            "# HFV3 Embedded Sidecar Candidate",
            "",
            f"- Generated UTC: {manifest.generated_at_utc}",
            f"- Source dense archive: `{manifest.source_dense_archive}`",
            f"- Output archive: `{manifest.output_archive}`",
            f"- Output submission dir: `{manifest.output_submission_dir}`",
            f"- Output submission archive: `{manifest.output_submission_archive}`",
            f"- Dense archive bytes: {manifest.source_dense_archive_bytes}",
            f"- Embedded archive bytes: {manifest.output_archive_bytes}",
            f"- Bytes saved vs dense archive: {manifest.bytes_saved_vs_dense_archive}",
            f"- Bytes saved vs HFV2 two-member archive: {manifest.bytes_saved_vs_hfv2_archive}",
            f"- Bytes delta vs FEC6 baseline archive: {manifest.bytes_delta_vs_baseline_archive}",
            f"- Rate delta vs dense archive: {manifest.rate_delta_vs_dense_archive:.12g}",
            f"- Rate delta vs HFV2 two-member archive: {manifest.rate_delta_vs_hfv2_archive:.12g}",
            f"- Rate delta vs FEC6 baseline: {manifest.rate_delta_vs_baseline_archive:.12g}",
            f"- Dense HFV1 payload bytes: {manifest.source_hfv1_payload_bytes}",
            f"- Embedded HFV3 payload bytes: {manifest.output_hfv3_payload_bytes}",
            f"- Sparse pair count: {manifest.sparse_pair_count}",
            f"- Row parity exact: {str(manifest.row_parity_exact).lower()}",
            f"- Paired auth eval plan ready: {str(manifest.paired_auth_eval_plan_ready).lower()}",
            "",
            "Plan command:",
            "",
            "```bash",
            shlex.join(manifest.paired_modal_auth_eval_plan_command_template_not_run),
            "```",
            "",
            "Execute command after claim surface clears:",
            "",
            "```bash",
            shlex.join(
                manifest.paired_modal_auth_eval_execute_command_template_after_claim_surface_clear
            ),
            "```",
            "",
            "- Score claim: false",
            "- Promotion eligible: false",
            "- Ready for exact eval dispatch: false",
            "",
        ]
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-archive", type=Path, default=DEFAULT_DENSE_ARCHIVE)
    parser.add_argument("--source-submission-dir", type=Path, default=DEFAULT_SOURCE_SUBMISSION_DIR)
    parser.add_argument("--baseline-archive", type=Path, default=DEFAULT_BASELINE_ARCHIVE)
    parser.add_argument(
        "--hfv2-archive",
        type=Path,
        default=Path("experiments/results/hfv2_sparse_sidecar_candidate_20260521T070412Z/archive.zip"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results") / f"hfv3_embedded_sidecar_candidate_{_utc_stamp()}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_candidate(
        dense_archive=args.dense_archive,
        source_submission_dir=args.source_submission_dir,
        baseline_archive=args.baseline_archive,
        hfv2_archive=args.hfv2_archive,
        output_dir=args.output_dir,
    )
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    (args.output_dir / "hfv3_embedded_manifest.json").write_text(payload, encoding="utf-8")
    (args.output_dir / "hfv3_embedded_manifest.md").write_text(
        render_markdown(manifest),
        encoding="utf-8",
    )
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
