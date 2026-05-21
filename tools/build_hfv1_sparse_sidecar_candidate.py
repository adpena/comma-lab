#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an exact sparse HFV2 sidecar candidate from a dense HFV1 archive.

HFV1 stores one five-float foveation row per frame. The current PR101 HFV1
candidate archives carry 1,200 rows even when only a small set of paired frames
differs from the identity row. This tool builds a deterministic candidate with
an exact pair-sparse ``foveation_params.hfv2`` sidecar and a generated runtime
copy that decodes HFV2 without changing decoder math.

It writes artifacts only under ``experiments/results`` and does not run exact
eval or claim a score.
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
HFV2_HEADER = struct.Struct("<4sIIIfffffH")
HFV2_PAIR_ROW = struct.Struct("<Hfffff")
DEFAULT_DENSE_ARCHIVE = Path(
    "experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/"
    "archive_seed_top16_component_hardpairs/archive.zip"
)
DEFAULT_SUBMISSION_DIR = Path(
    "experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/"
    "submission_dir"
)
DEFAULT_BASELINE_ARCHIVE = Path(
    "experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/"
    "submission_dir/archive.zip"
)


@dataclass(frozen=True)
class Hfv2SparseManifest:
    schema: str
    generated_at_utc: str
    source_dense_archive: str
    source_dense_archive_bytes: int
    source_dense_archive_sha256: str
    source_hfv1_payload_bytes: int
    source_hfv1_payload_sha256: str
    output_archive: str
    output_archive_bytes: int
    output_archive_sha256: str
    output_hfv2_payload_bytes: int
    output_hfv2_payload_sha256: str
    output_submission_dir: str
    output_submission_archive: str
    output_inflate_py_sha256: str
    output_archive_manifest: str
    output_archive_manifest_sha256: str
    baseline_archive: str
    baseline_archive_bytes: int
    baseline_archive_sha256: str
    x_member_sha256: str
    x_member_bytes: int
    default_row: tuple[float, float, float, float, float]
    sparse_pair_count: int
    sparse_pairs: list[int]
    row_parity_exact: bool
    bytes_saved_vs_dense_archive: int
    bytes_delta_vs_baseline_archive: int
    rate_delta_vs_dense_archive: float
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


def _sparse_pairs_from_rows(
    rows: list[tuple[float, float, float, float, float]]
) -> tuple[tuple[float, float, float, float, float], list[tuple[int, tuple[float, float, float, float, float]]]]:
    if len(rows) % 2:
        raise ValueError("HFV2 pair-sparse conversion requires an even frame count")
    default_row = Counter(rows).most_common(1)[0][0]
    pairs: list[tuple[int, tuple[float, float, float, float, float]]] = []
    for pair_index in range(len(rows) // 2):
        row0 = rows[2 * pair_index]
        row1 = rows[2 * pair_index + 1]
        if row0 == default_row and row1 == default_row:
            continue
        if row0 != row1:
            raise ValueError(
                f"HFV2 pair-sparse conversion cannot encode pair {pair_index}: rows differ"
            )
        pairs.append((pair_index, row0))
    return default_row, pairs


def _pack_hfv2(
    *,
    n_frames: int,
    frame_height: int,
    frame_width: int,
    default_row: tuple[float, float, float, float, float],
    pairs: list[tuple[int, tuple[float, float, float, float, float]]],
) -> bytes:
    if len(pairs) > 65535:
        raise ValueError("too many HFV2 sparse pairs")
    out = bytearray()
    out.extend(HFV2_HEADER.pack(b"HFV2", n_frames, frame_height, frame_width, *default_row, len(pairs)))
    for pair_index, row in pairs:
        if not 0 <= pair_index <= 65535:
            raise ValueError(f"HFV2 pair index out of uint16 range: {pair_index}")
        out.extend(HFV2_PAIR_ROW.pack(pair_index, *row))
    return bytes(out)


def _decode_hfv2_rows(raw: bytes) -> list[tuple[float, float, float, float, float]]:
    if len(raw) < HFV2_HEADER.size:
        raise ValueError("HFV2 payload truncated before header")
    magic, n_frames, frame_height, frame_width, *tail = HFV2_HEADER.unpack_from(raw)
    if magic != b"HFV2":
        raise ValueError(f"HFV2 magic mismatch: {magic!r}")
    if int(frame_height) != CAMERA_H or int(frame_width) != CAMERA_W:
        raise ValueError("HFV2 image size mismatch")
    default_row = tuple(float(v) for v in tail[:5])
    pair_count = int(tail[5])
    expected = HFV2_HEADER.size + pair_count * HFV2_PAIR_ROW.size
    if len(raw) != expected:
        raise ValueError(f"HFV2 payload size mismatch: got {len(raw)}, expected {expected}")
    rows = [default_row for _ in range(int(n_frames))]
    offset = HFV2_HEADER.size
    for _ in range(pair_count):
        pair_index, *values = HFV2_PAIR_ROW.unpack_from(raw, offset)
        offset += HFV2_PAIR_ROW.size
        row = tuple(float(v) for v in values)
        rows[2 * int(pair_index)] = row
        rows[2 * int(pair_index) + 1] = row
    return rows


def _patch_inflate_py(source: str) -> str:
    if "HFV2_MAGIC" in source:
        return source
    source = source.replace(
        'HFV1_MAGIC = b"HFV1"\n',
        'HFV1_MAGIC = b"HFV1"\nHFV2_MAGIC = b"HFV2"\n',
    )
    source = source.replace(
        "HFV1_ROW_STRUCT = struct.Struct(\"<fffff\")\n",
        "HFV1_ROW_STRUCT = struct.Struct(\"<fffff\")\n"
        "HFV2_HEADER_STRUCT = struct.Struct(\"<4sIIIfffffH\")\n"
        "HFV2_PAIR_ROW_STRUCT = struct.Struct(\"<Hfffff\")\n",
    )
    marker = "\n\ndef load_foveation_sidecar(src_bin: Path) -> dict[str, object] | None:\n"
    insert = r'''

def load_hfv2_sparse_params(params_path: Path) -> dict[str, object]:
    if not params_path.is_file():
        raise ValueError(f"HFV2 foveation params not found: {params_path}")
    raw = params_path.read_bytes()
    if len(raw) < HFV2_HEADER_STRUCT.size:
        raise ValueError("HFV2 foveation params truncated before header")
    unpacked = HFV2_HEADER_STRUCT.unpack(raw[: HFV2_HEADER_STRUCT.size])
    magic, n_frames, frame_height, frame_width = unpacked[:4]
    if magic != HFV2_MAGIC:
        raise ValueError(f"HFV2 foveation params magic mismatch: {magic!r}")
    if int(frame_height) != CAMERA_H or int(frame_width) != CAMERA_W:
        raise ValueError(
            "HFV2 image size does not match PR101 output: "
            f"{(frame_height, frame_width)} vs {(CAMERA_H, CAMERA_W)}"
        )
    default_row = tuple(float(value) for value in unpacked[4:9])
    pair_count = int(unpacked[9])
    expected = HFV2_HEADER_STRUCT.size + pair_count * HFV2_PAIR_ROW_STRUCT.size
    if len(raw) != expected:
        raise ValueError(f"HFV2 foveation params size mismatch: got {len(raw)}, expected {expected}")
    row_by_frame: dict[int, tuple[float, float, float, float, float]] = {}
    for index in range(pair_count):
        pair_index, *values = HFV2_PAIR_ROW_STRUCT.unpack_from(
            raw,
            HFV2_HEADER_STRUCT.size + index * HFV2_PAIR_ROW_STRUCT.size,
        )
        row = tuple(float(value) for value in values)
        row_by_frame[2 * int(pair_index)] = row
        row_by_frame[2 * int(pair_index) + 1] = row
    return {
        "source_format": "HFV2_pair_sparse",
        "n_frames": int(n_frames),
        "default_row": default_row,
        "row_by_frame": row_by_frame,
        "pair_count": pair_count,
    }
'''
    source = source.replace(marker, insert + marker)
    source = source.replace(
        "def load_foveation_sidecar(src_bin: Path) -> dict[str, object] | None:\n"
        "    hfv1 = src_bin.with_name(\"foveation_params.bin\")\n",
        "def load_foveation_sidecar(src_bin: Path) -> dict[str, object] | None:\n"
        "    hfv2 = src_bin.with_name(\"foveation_params.hfv2\")\n"
        "    if hfv2.is_file():\n"
        "        return load_hfv2_sparse_params(hfv2)\n"
        "    hfv1 = src_bin.with_name(\"foveation_params.bin\")\n",
    )
    source = source.replace(
        '    if params.get("source_format") == "LFV1_sparse":\n',
        '    if params.get("source_format") in {"LFV1_sparse", "HFV2_pair_sparse"}:\n',
    )
    if "load_hfv2_sparse_params" not in source:
        raise ValueError("failed to patch inflate.py with HFV2 decoder")
    return source


def _copy_runtime_with_hfv2(source_dir: Path, output_dir: Path) -> Path:
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


def _zip_member_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            payload = archive.read(info.filename)
            method = "stored" if info.compress_type == zipfile.ZIP_STORED else str(info.compress_type)
            records.append(
                {
                    "name": info.filename,
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "crc": int(info.CRC),
                    "compression_method": method,
                    "sha256": _sha256_bytes(payload),
                }
            )
    return records


def _write_stored_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)


def _write_generated_archive_manifest(
    *,
    submission_dir: Path,
    archive_path: Path,
    dense_archive: Path,
    baseline_archive: Path,
    source_submission_dir: Path,
    hfv1_raw: bytes,
    hfv2_raw: bytes,
    sparse_pairs: list[tuple[int, tuple[float, float, float, float, float]]],
    row_parity_exact: bool,
    paired_plan_cmd: list[str],
    paired_execute_cmd: list[str],
) -> Path:
    manifest_path = submission_dir / "archive_manifest.json"
    prior_schema = None
    prior_manifest = source_submission_dir / "archive_manifest.json"
    if prior_manifest.is_file():
        try:
            prior_schema = json.loads(prior_manifest.read_text(encoding="utf-8")).get("schema")
        except json.JSONDecodeError:
            prior_schema = "unreadable_source_manifest"
    payload = {
        "schema": "hfv2_pair_sparse_sidecar_candidate_manifest_v1",
        "generated_at_utc": _utc_iso(),
        "archive_path": _repo_rel(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256_file(archive_path),
        "members": _zip_member_records(archive_path),
        "member_shape": "two_member_pr101_source_payload_plus_hfv2_exact_pair_sparse_foveation_sidecar",
        "source_dense_archive": _repo_rel(dense_archive),
        "source_dense_archive_bytes": dense_archive.stat().st_size,
        "source_dense_archive_sha256": _sha256_file(dense_archive),
        "baseline_archive": _repo_rel(baseline_archive),
        "baseline_archive_bytes": baseline_archive.stat().st_size,
        "baseline_archive_sha256": _sha256_file(baseline_archive),
        "source_hfv1_payload_bytes": len(hfv1_raw),
        "source_hfv1_payload_sha256": _sha256_bytes(hfv1_raw),
        "output_hfv2_payload_bytes": len(hfv2_raw),
        "output_hfv2_payload_sha256": _sha256_bytes(hfv2_raw),
        "sparse_pair_count": len(sparse_pairs),
        "sparse_pairs": [pair_index for pair_index, _row in sparse_pairs],
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
        "source_submission_dir": _repo_rel(source_submission_dir),
        "source_submission_manifest_schema": prior_schema,
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
    output_dir: Path,
) -> Hfv2SparseManifest:
    with zipfile.ZipFile(dense_archive) as archive:
        hfv1_raw = archive.read("foveation_params.bin")
        x_payload = archive.read("x")
    n_frames, frame_height, frame_width, rows = _read_dense_hfv1(hfv1_raw)
    if frame_height != CAMERA_H or frame_width != CAMERA_W:
        raise ValueError("HFV1 image size mismatch")
    default_row, sparse_pairs = _sparse_pairs_from_rows(rows)
    hfv2_raw = _pack_hfv2(
        n_frames=n_frames,
        frame_height=frame_height,
        frame_width=frame_width,
        default_row=default_row,
        pairs=sparse_pairs,
    )
    row_parity_exact = _decode_hfv2_rows(hfv2_raw) == rows
    if not row_parity_exact:
        raise ValueError("HFV2 row parity check failed")

    archive_path = output_dir / "archive.zip"
    _write_stored_zip(
        archive_path,
        [
            ("foveation_params.hfv2", hfv2_raw),
            ("x", x_payload),
        ],
    )
    submission_dir = _copy_runtime_with_hfv2(
        source_submission_dir,
        output_dir / "submission_dir_hfv2",
    )
    submission_archive = submission_dir / "archive.zip"
    shutil.copy2(archive_path, submission_archive)
    lane_id_base = "hfv2_pair_sparse_pr101_hfv1_sidecar_exact_eval"
    paired_plan_cmd = paired_auth_eval_dispatch_command_template(
        archive_path=_repo_rel(submission_archive),
        submission_dir=_repo_rel(submission_dir),
        lane_id_base=lane_id_base,
        archive_sha256=_sha256_file(submission_archive),
        execute=False,
        label=lane_id_base,
        run_id=f"hfv2_pair_sparse_pr101_{_sha256_file(submission_archive)[:12]}",
        claim_agent="codex:hfv2_sparse_sidecar",
        claim_notes=(
            "HFV2 pair-sparse sidecar candidate; score_claim=false until paired "
            "contest CPU/CUDA exact eval harvest."
        ),
    )
    paired_execute_cmd = paired_auth_eval_dispatch_command_template(
        archive_path=_repo_rel(submission_archive),
        submission_dir=_repo_rel(submission_dir),
        lane_id_base=lane_id_base,
        archive_sha256=_sha256_file(submission_archive),
        execute=True,
        label=lane_id_base,
        run_id=f"hfv2_pair_sparse_pr101_{_sha256_file(submission_archive)[:12]}",
        claim_agent="codex:hfv2_sparse_sidecar",
        claim_notes=(
            "HFV2 pair-sparse sidecar candidate; score_claim=false until paired "
            "contest CPU/CUDA exact eval harvest."
        ),
    )
    generated_archive_manifest = _write_generated_archive_manifest(
        submission_dir=submission_dir,
        archive_path=submission_archive,
        dense_archive=dense_archive,
        baseline_archive=baseline_archive,
        source_submission_dir=source_submission_dir,
        hfv1_raw=hfv1_raw,
        hfv2_raw=hfv2_raw,
        sparse_pairs=sparse_pairs,
        row_parity_exact=row_parity_exact,
        paired_plan_cmd=paired_plan_cmd,
        paired_execute_cmd=paired_execute_cmd,
    )
    manifest = Hfv2SparseManifest(
        schema="hfv1_to_hfv2_sparse_sidecar_candidate_v1",
        generated_at_utc=_utc_iso(),
        source_dense_archive=_repo_rel(dense_archive),
        source_dense_archive_bytes=dense_archive.stat().st_size,
        source_dense_archive_sha256=_sha256_file(dense_archive),
        source_hfv1_payload_bytes=len(hfv1_raw),
        source_hfv1_payload_sha256=_sha256_bytes(hfv1_raw),
        output_archive=_repo_rel(archive_path),
        output_archive_bytes=archive_path.stat().st_size,
        output_archive_sha256=_sha256_file(archive_path),
        output_hfv2_payload_bytes=len(hfv2_raw),
        output_hfv2_payload_sha256=_sha256_bytes(hfv2_raw),
        output_submission_dir=_repo_rel(submission_dir),
        output_submission_archive=_repo_rel(submission_archive),
        output_inflate_py_sha256=_sha256_file(submission_dir / "inflate.py"),
        output_archive_manifest=_repo_rel(generated_archive_manifest),
        output_archive_manifest_sha256=_sha256_file(generated_archive_manifest),
        baseline_archive=_repo_rel(baseline_archive),
        baseline_archive_bytes=baseline_archive.stat().st_size,
        baseline_archive_sha256=_sha256_file(baseline_archive),
        x_member_sha256=_sha256_bytes(x_payload),
        x_member_bytes=len(x_payload),
        default_row=default_row,
        sparse_pair_count=len(sparse_pairs),
        sparse_pairs=[pair_index for pair_index, _row in sparse_pairs],
        row_parity_exact=row_parity_exact,
        bytes_saved_vs_dense_archive=dense_archive.stat().st_size - archive_path.stat().st_size,
        bytes_delta_vs_baseline_archive=archive_path.stat().st_size - baseline_archive.stat().st_size,
        rate_delta_vs_dense_archive=_rate_delta(archive_path.stat().st_size - dense_archive.stat().st_size),
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
    return manifest


def render_markdown(manifest: Hfv2SparseManifest) -> str:
    return "\n".join(
        [
            "# HFV2 Sparse Sidecar Candidate",
            "",
            f"- Generated UTC: {manifest.generated_at_utc}",
            f"- Source dense archive: `{manifest.source_dense_archive}`",
            f"- Output archive: `{manifest.output_archive}`",
            f"- Output submission dir: `{manifest.output_submission_dir}`",
            f"- Output submission archive: `{manifest.output_submission_archive}`",
            f"- Dense archive bytes: {manifest.source_dense_archive_bytes}",
            f"- Sparse archive bytes: {manifest.output_archive_bytes}",
            f"- Bytes saved vs dense archive: {manifest.bytes_saved_vs_dense_archive}",
            f"- Bytes delta vs FEC6 baseline archive: {manifest.bytes_delta_vs_baseline_archive}",
            f"- Rate delta vs dense archive: {manifest.rate_delta_vs_dense_archive:.12g}",
            f"- Rate delta vs FEC6 baseline: {manifest.rate_delta_vs_baseline_archive:.12g}",
            f"- Dense HFV1 payload bytes: {manifest.source_hfv1_payload_bytes}",
            f"- Sparse HFV2 payload bytes: {manifest.output_hfv2_payload_bytes}",
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
    parser.add_argument("--source-submission-dir", type=Path, default=DEFAULT_SUBMISSION_DIR)
    parser.add_argument("--baseline-archive", type=Path, default=DEFAULT_BASELINE_ARCHIVE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results") / f"hfv1_sparse_sidecar_candidate_{_utc_stamp()}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_candidate(
        dense_archive=args.dense_archive,
        source_submission_dir=args.source_submission_dir,
        baseline_archive=args.baseline_archive,
        output_dir=args.output_dir,
    )
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    (args.output_dir / "hfv2_sparse_manifest.json").write_text(payload, encoding="utf-8")
    (args.output_dir / "hfv2_sparse_manifest.md").write_text(
        render_markdown(manifest),
        encoding="utf-8",
    )
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
