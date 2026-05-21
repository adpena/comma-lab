#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an HFV7 Exp-Golomb-coded embedded foveation sidecar candidate.

HFV6 stores active-pair deltas as raw uint8 bytes. HFV7 keeps the same
profile-coded active row, but encodes the archive-contained delta sequence with
unsigned Exp-Golomb order 3. For the current PR101/FEC6 HFV active-pair list,
that cuts the sidecar from 16 bytes to 12 bytes without hardcoding the pair
list in runtime code.

This writes research artifacts only and does not claim a score.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tac.deploy.modal.paired_dispatch import paired_auth_eval_dispatch_command_template
from tools.build_hfv4_embedded_sidecar_candidate import (
    CAMERA_H,
    CAMERA_W,
    DEFAULT_BASELINE_ARCHIVE,
    DEFAULT_DENSE_ARCHIVE,
    _deltas_from_pairs,
    _patch_inflate_py as _patch_hfv4_inflate_py,
    _rate_delta,
    _read_dense_hfv1,
    _repeated_sparse_pairs,
    _repo_rel,
    _sha256_bytes,
    _sha256_file,
    _utc_iso,
    _utc_stamp,
    _write_stored_zip,
    _zip_member_records,
)
from tools.build_hfv5_profile_sidecar_candidate import _profile_row

HFV7_PROFILE_ID = 1
HFV7_EXP_GOLOMB_ORDER = 3
DEFAULT_SOURCE_SUBMISSION_DIR = Path(
    "experiments/results/hfv3_embedded_sidecar_candidate_20260521T072238Z/"
    "submission_dir_hfv3_embedded"
)
DEFAULT_HFV6_ARCHIVE = Path(
    "experiments/results/hfv6_implicit_delta_sidecar_candidate_20260521T192355Z/archive.zip"
)


@dataclass(frozen=True)
class Hfv7ExpGolombManifest:
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
    output_hfv7_payload_bytes: int
    output_hfv7_payload_bits: int
    output_hfv7_payload_sha256: str
    original_x_member_bytes: int
    original_x_member_sha256: str
    embedded_x_member_bytes: int
    embedded_x_member_sha256: str
    exp_golomb_order: int
    profile_id: int
    dense_active_row: tuple[float, float, float, float, float]
    profile_active_row: tuple[float, float, float, float, float]
    profile_row_abs_delta: tuple[float, float, float, float, float]
    sparse_pair_count: int
    sparse_pairs: list[int]
    sparse_pair_deltas: list[int]
    bytes_saved_vs_dense_archive: int
    bytes_saved_vs_hfv6_archive: int
    bytes_delta_vs_baseline_archive: int
    rate_delta_vs_dense_archive: float
    rate_delta_vs_hfv6_archive: float
    rate_delta_vs_baseline_archive: float
    target_modes: list[str]
    dispatch_blockers: list[str]
    paired_auth_eval_required: bool
    paired_auth_eval_plan_ready: bool
    paired_modal_auth_eval_plan_command_template_not_run: list[str]
    paired_modal_auth_eval_execute_command_template_after_claim_surface_clear: list[str]
    compliance_note: str
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _append_bits(bits: list[int], value: int, width: int) -> None:
    for shift in range(width - 1, -1, -1):
        bits.append((int(value) >> shift) & 1)


def _pack_bits(bits: list[int]) -> bytes:
    out = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        if bit:
            out[index // 8] |= 1 << (7 - (index % 8))
    return bytes(out)


def _pack_hfv7(active_pairs: list[int]) -> tuple[bytes, int]:
    deltas = _deltas_from_pairs(active_pairs)
    if not deltas:
        raise ValueError("HFV7 requires at least one active pair")
    bits: list[int] = []
    suffix_mask = (1 << HFV7_EXP_GOLOMB_ORDER) - 1
    for delta in deltas:
        if delta <= 0:
            raise ValueError(f"HFV7 Exp-Golomb deltas must be positive: {deltas}")
        value = int(delta) - 1
        prefix_value = (value >> HFV7_EXP_GOLOMB_ORDER) + 1
        prefix_width = prefix_value.bit_length()
        bits.extend([0] * (prefix_width - 1))
        _append_bits(bits, prefix_value, prefix_width)
        _append_bits(bits, value & suffix_mask, HFV7_EXP_GOLOMB_ORDER)
    return _pack_bits(bits), len(bits)


def _patch_inflate_py(source: str) -> str:
    source = _patch_hfv4_inflate_py(source)
    if "HFV7_exp_golomb_sparse" in source:
        return source
    source = source.replace(
        "    raise ValueError(f\"PR101 frame-selector trailing bytes: pos={pos} total={len(bin_bytes)}\")\n",
        "    return trailer\n",
    )
    marker = "\n\ndef load_foveation_sidecar(src_bin: Path) -> dict[str, object] | None:\n"
    insert = r'''

def hfv7_profile_row() -> tuple[float, float, float, float, float]:
    camera_diag = (float(CAMERA_W * CAMERA_W + CAMERA_H * CAMERA_H)) ** 0.5
    return (
        5.5e-4,
        camera_diag * 0.78,
        1.4,
        (float(CAMERA_W) - 1.0) / 2.0,
        float(CAMERA_H) * 0.45,
    )


def hfv7_bit(raw: bytes, bit_pos: int) -> int:
    return (raw[bit_pos // 8] >> (7 - (bit_pos % 8))) & 1


def decode_hfv7_exp_golomb_order3(raw: bytes) -> list[int]:
    values: list[int] = []
    bit_pos = 0
    max_bits = len(raw) * 8
    while bit_pos < max_bits:
        if not any(hfv7_bit(raw, probe) for probe in range(bit_pos, max_bits)):
            break
        leading_zeroes = 0
        while bit_pos < max_bits and hfv7_bit(raw, bit_pos) == 0:
            leading_zeroes += 1
            bit_pos += 1
        if bit_pos >= max_bits:
            raise ValueError("HFV7 Exp-Golomb payload ended mid-prefix")
        prefix_value = 0
        for _ in range(leading_zeroes + 1):
            prefix_value = (prefix_value << 1) | hfv7_bit(raw, bit_pos)
            bit_pos += 1
        suffix = 0
        for _ in range(3):
            if bit_pos >= max_bits:
                raise ValueError("HFV7 Exp-Golomb payload ended mid-suffix")
            suffix = (suffix << 1) | hfv7_bit(raw, bit_pos)
            bit_pos += 1
        values.append((((prefix_value - 1) << 3) | suffix) + 1)
    if not values:
        raise ValueError("HFV7 Exp-Golomb payload decoded no deltas")
    return values


def load_hfv7_exp_golomb_params_from_bytes(raw: bytes) -> dict[str, object]:
    active_row = hfv7_profile_row()
    row_by_frame: dict[int, tuple[float, float, float, float, float]] = {}
    current_pair = 0
    for index, delta in enumerate(decode_hfv7_exp_golomb_order3(raw)):
        current_pair = int(delta) if index == 0 else current_pair + int(delta)
        row_by_frame[2 * current_pair] = active_row
        row_by_frame[2 * current_pair + 1] = active_row
    return {
        "source_format": "HFV7_exp_golomb_sparse",
        "n_frames": 1200,
        "default_row": (0.0, 0.0, 0.0, 0.0, 0.0),
        "row_by_frame": row_by_frame,
        "pair_count": len(row_by_frame) // 2,
        "exp_golomb_order": 3,
    }
'''
    source = source.replace(marker, insert + marker)
    source = source.replace(
        "    raise ValueError(f\"unsupported embedded foveation magic: {trailer[:4]!r}\")\n",
        "    if len(trailer) == 12:\n"
        "        return load_hfv7_exp_golomb_params_from_bytes(trailer)\n"
        "    raise ValueError(f\"unsupported embedded foveation magic: {trailer[:4]!r}\")\n",
    )
    source = source.replace(
        '"HFV4_repeated_pair_delta_sparse"}',
        '"HFV4_repeated_pair_delta_sparse", "HFV7_exp_golomb_sparse"}',
    )
    if "HFV7_exp_golomb_sparse" not in source:
        raise ValueError("failed to patch inflate.py with HFV7 decoder")
    return source


def _copy_runtime(source_dir: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(
        source_dir,
        output_dir,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "archive.zip",
            "archive_manifest.json",
            "README.md",
            "report.txt",
            "pre_submission_compliance.*",
        ),
    )
    inflate_py = output_dir / "inflate.py"
    inflate_py.write_text(_patch_inflate_py(inflate_py.read_text(encoding="utf-8")), encoding="utf-8")
    return output_dir


def _write_research_submission_metadata(
    submission_dir: Path,
    manifest: Hfv7ExpGolombManifest,
) -> None:
    readme = "\n".join(
        [
            "# HFV7 Exp-Golomb Sidecar Candidate",
            "",
            "Research-only byte-closed candidate. This packet has full in-process",
            "frame parity against the dense HFV1 source path, but it has not run",
            "paired contest CPU/CUDA exact eval and does not claim a score.",
            "",
            f"- Archive: `archive.zip`, SHA-256 `{manifest.output_archive_sha256}`, {manifest.output_archive_bytes} bytes.",
            f"- Payload: {manifest.output_hfv7_payload_bytes} bytes / {manifest.output_hfv7_payload_bits} bits.",
            f"- Bytes over FEC6/PR110 baseline: {manifest.bytes_delta_vs_baseline_archive}.",
            f"- Rate delta vs FEC6/PR110 baseline: {manifest.rate_delta_vs_baseline_archive:.15g}.",
            f"- Sparse pair count: {manifest.sparse_pair_count}.",
            "- Score claim: false.",
            "- Promotion eligible: false.",
            "- Ready for exact eval dispatch: false.",
            "",
            "Compliance status:",
            "",
            manifest.compliance_note,
            "",
            "Required before promotion:",
            "",
            "- Shell-level `inflate.sh` output parity or paired exact eval harvest.",
            "- Paired contest CPU/CUDA exact eval on the same archive/runtime packet.",
            "- Explicit compliance acceptance for the implicit trailer and runtime profile row.",
            "",
        ]
    )
    report = "\n".join(
        [
            "HFV7 Exp-Golomb Sidecar Candidate - research-only report",
            f"generated_at_utc: {manifest.generated_at_utc}",
            f"archive_sha256: {manifest.output_archive_sha256}",
            f"archive_size_bytes: {manifest.output_archive_bytes}",
            f"payload_bytes: {manifest.output_hfv7_payload_bytes}",
            f"payload_bits: {manifest.output_hfv7_payload_bits}",
            f"bytes_delta_vs_fec6_pr110_baseline: {manifest.bytes_delta_vs_baseline_archive}",
            f"rate_delta_vs_fec6_pr110_baseline: {manifest.rate_delta_vs_baseline_archive:.15g}",
            "score_claim: false",
            "contest_cpu_exact_eval: not_run",
            "contest_cuda_exact_eval: not_run",
            "parity_status: in_process_frame_parity_required_or_external_artifact",
            f"compliance_note: {manifest.compliance_note}",
            "",
        ]
    )
    (submission_dir / "README.md").write_text(readme, encoding="utf-8")
    (submission_dir / "report.txt").write_text(report, encoding="utf-8")


def _manifest_common(
    *,
    archive_path: Path,
    dense_archive: Path,
    baseline_archive: Path,
    hfv1_raw: bytes,
    hfv7_raw: bytes,
    hfv7_bits: int,
    active_pairs: list[int],
    profile_active_row: tuple[float, float, float, float, float],
    dense_active_row: tuple[float, float, float, float, float],
) -> dict[str, Any]:
    return {
        "archive_path": _repo_rel(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256_file(archive_path),
        "members": _zip_member_records(archive_path),
        "source_dense_archive": _repo_rel(dense_archive),
        "source_dense_archive_bytes": dense_archive.stat().st_size,
        "source_dense_archive_sha256": _sha256_file(dense_archive),
        "baseline_archive": _repo_rel(baseline_archive),
        "baseline_archive_bytes": baseline_archive.stat().st_size,
        "baseline_archive_sha256": _sha256_file(baseline_archive),
        "source_hfv1_payload_bytes": len(hfv1_raw),
        "source_hfv1_payload_sha256": _sha256_bytes(hfv1_raw),
        "output_hfv7_payload_bytes": len(hfv7_raw),
        "output_hfv7_payload_bits": hfv7_bits,
        "output_hfv7_payload_sha256": _sha256_bytes(hfv7_raw),
        "exp_golomb_order": HFV7_EXP_GOLOMB_ORDER,
        "profile_id": HFV7_PROFILE_ID,
        "dense_active_row": dense_active_row,
        "profile_active_row": profile_active_row,
        "profile_row_abs_delta": tuple(
            abs(float(a) - float(b))
            for a, b in zip(dense_active_row, profile_active_row, strict=True)
        ),
        "sparse_pair_count": len(active_pairs),
        "sparse_pairs": active_pairs,
        "sparse_pair_deltas": _deltas_from_pairs(active_pairs),
        "target_modes": ["contest_exact_eval"],
        "dispatch_blockers": [
            "exact_contest_cpu_eval_missing",
            "exact_contest_cuda_eval_missing",
            "full_parity_proof_required_before_execute",
            "lane_dispatch_claim_required_before_execute",
            "exp_golomb_profile_compliance_review_required_before_submission",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_submission": False,
        "ready_for_exact_eval_dispatch": False,
        "compliance_note": (
            "HFV7 stores an Exp-Golomb-coded active-pair delta sequence after the "
            "selector payload. The active-row profile and implicit trailer "
            "interpretation live in inflate.py. Treat as research-only until "
            "contest-compliance review accepts the implicit-trailer profile-code "
            "interpretation."
        ),
    }


def _write_archive_manifest(
    *,
    submission_dir: Path,
    archive_path: Path,
    dense_archive: Path,
    baseline_archive: Path,
    hfv1_raw: bytes,
    hfv7_raw: bytes,
    hfv7_bits: int,
    active_pairs: list[int],
    profile_active_row: tuple[float, float, float, float, float],
    dense_active_row: tuple[float, float, float, float, float],
    paired_plan_cmd: list[str],
    paired_execute_cmd: list[str],
) -> Path:
    manifest_path = submission_dir / "archive_manifest.json"
    payload = {
        "schema": "hfv7_exp_golomb_sidecar_candidate_manifest_v1",
        "generated_at_utc": _utc_iso(),
        "member_shape": "single_member_x_with_pr101_source_payload_fec6_selector_and_embedded_hfv7_exp_golomb_sidecar",
        **_manifest_common(
            archive_path=archive_path,
            dense_archive=dense_archive,
            baseline_archive=baseline_archive,
            hfv1_raw=hfv1_raw,
            hfv7_raw=hfv7_raw,
            hfv7_bits=hfv7_bits,
            active_pairs=active_pairs,
            profile_active_row=profile_active_row,
            dense_active_row=dense_active_row,
        ),
        "paired_auth_eval_required": True,
        "paired_auth_eval_plan_ready": False,
        "paired_modal_auth_eval_plan_command_template_not_run": paired_plan_cmd,
        "paired_modal_auth_eval_execute_command_template_after_claim_surface_clear": paired_execute_cmd,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def build_candidate(
    *,
    dense_archive: Path,
    source_submission_dir: Path,
    baseline_archive: Path,
    hfv6_archive: Path,
    output_dir: Path,
) -> Hfv7ExpGolombManifest:
    with zipfile.ZipFile(dense_archive) as archive:
        hfv1_raw = archive.read("foveation_params.bin")
        x_payload = archive.read("x")
    n_frames, frame_height, frame_width, rows = _read_dense_hfv1(hfv1_raw)
    if n_frames != 1200 or frame_height != CAMERA_H or frame_width != CAMERA_W:
        raise ValueError("HFV7 Exp-Golomb form expects PR101/FEC6 camera geometry")
    _default_row, dense_active_row, active_pairs = _repeated_sparse_pairs(rows)
    profile_active_row = _profile_row(HFV7_PROFILE_ID)
    profile_row_abs_delta = tuple(
        abs(float(a) - float(b)) for a, b in zip(dense_active_row, profile_active_row, strict=True)
    )
    hfv7_raw, hfv7_bits = _pack_hfv7(active_pairs)
    embedded_x = x_payload + hfv7_raw
    archive_path = output_dir / "archive.zip"
    _write_stored_zip(archive_path, [("x", embedded_x)])
    submission_dir = _copy_runtime(source_submission_dir, output_dir / "submission_dir_hfv7_exp_golomb")
    submission_archive = submission_dir / "archive.zip"
    shutil.copy2(archive_path, submission_archive)

    lane_id_base = "hfv7_exp_golomb_pr101_hfv1_sidecar_exact_eval"
    run_id = f"hfv7_exp_golomb_pr101_{_sha256_file(submission_archive)[:12]}"
    claim_notes = (
        "HFV7 Exp-Golomb sidecar candidate; score_claim=false until full parity "
        "and paired contest CPU/CUDA exact eval harvest."
    )
    paired_plan_cmd = paired_auth_eval_dispatch_command_template(
        archive_path=_repo_rel(submission_archive),
        submission_dir=_repo_rel(submission_dir),
        lane_id_base=lane_id_base,
        archive_sha256=_sha256_file(submission_archive),
        execute=False,
        label=lane_id_base,
        run_id=run_id,
        claim_agent="codex:hfv7_exp_golomb_sidecar",
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
        claim_agent="codex:hfv7_exp_golomb_sidecar",
        claim_notes=claim_notes,
    )
    generated_manifest = _write_archive_manifest(
        submission_dir=submission_dir,
        archive_path=submission_archive,
        dense_archive=dense_archive,
        baseline_archive=baseline_archive,
        hfv1_raw=hfv1_raw,
        hfv7_raw=hfv7_raw,
        hfv7_bits=hfv7_bits,
        active_pairs=active_pairs,
        profile_active_row=profile_active_row,
        dense_active_row=dense_active_row,
        paired_plan_cmd=paired_plan_cmd,
        paired_execute_cmd=paired_execute_cmd,
    )
    manifest = Hfv7ExpGolombManifest(
        schema="hfv7_exp_golomb_sidecar_candidate_v1",
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
        output_hfv7_payload_bytes=len(hfv7_raw),
        output_hfv7_payload_bits=hfv7_bits,
        output_hfv7_payload_sha256=_sha256_bytes(hfv7_raw),
        original_x_member_bytes=len(x_payload),
        original_x_member_sha256=_sha256_bytes(x_payload),
        embedded_x_member_bytes=len(embedded_x),
        embedded_x_member_sha256=_sha256_bytes(embedded_x),
        exp_golomb_order=HFV7_EXP_GOLOMB_ORDER,
        profile_id=HFV7_PROFILE_ID,
        dense_active_row=dense_active_row,
        profile_active_row=profile_active_row,
        profile_row_abs_delta=profile_row_abs_delta,
        sparse_pair_count=len(active_pairs),
        sparse_pairs=active_pairs,
        sparse_pair_deltas=_deltas_from_pairs(active_pairs),
        bytes_saved_vs_dense_archive=dense_archive.stat().st_size - archive_path.stat().st_size,
        bytes_saved_vs_hfv6_archive=hfv6_archive.stat().st_size - archive_path.stat().st_size,
        bytes_delta_vs_baseline_archive=archive_path.stat().st_size - baseline_archive.stat().st_size,
        rate_delta_vs_dense_archive=_rate_delta(archive_path.stat().st_size - dense_archive.stat().st_size),
        rate_delta_vs_hfv6_archive=_rate_delta(archive_path.stat().st_size - hfv6_archive.stat().st_size),
        rate_delta_vs_baseline_archive=_rate_delta(archive_path.stat().st_size - baseline_archive.stat().st_size),
        target_modes=["contest_exact_eval"],
        dispatch_blockers=[
            "exact_contest_cpu_eval_missing",
            "exact_contest_cuda_eval_missing",
            "full_parity_proof_required_before_execute",
            "lane_dispatch_claim_required_before_execute",
            "exp_golomb_profile_compliance_review_required_before_submission",
        ],
        paired_auth_eval_required=True,
        paired_auth_eval_plan_ready=False,
        paired_modal_auth_eval_plan_command_template_not_run=paired_plan_cmd,
        paired_modal_auth_eval_execute_command_template_after_claim_surface_clear=paired_execute_cmd,
        compliance_note=(
            "HFV7 stores an Exp-Golomb-coded active-pair delta sequence after the "
            "selector payload. The active-row profile and implicit trailer "
            "interpretation live in inflate.py. Treat as research-only until "
            "contest-compliance review accepts the implicit-trailer profile-code "
            "interpretation."
        ),
    )
    _write_research_submission_metadata(submission_dir, manifest)
    return manifest


def render_markdown(manifest: Hfv7ExpGolombManifest) -> str:
    return "\n".join(
        [
            "# HFV7 Exp-Golomb Sidecar Candidate",
            "",
            f"- Generated UTC: {manifest.generated_at_utc}",
            f"- Source dense archive: `{manifest.source_dense_archive}`",
            f"- Output archive: `{manifest.output_archive}`",
            f"- Output submission dir: `{manifest.output_submission_dir}`",
            f"- Output submission archive: `{manifest.output_submission_archive}`",
            f"- Dense archive bytes: {manifest.source_dense_archive_bytes}",
            f"- Embedded archive bytes: {manifest.output_archive_bytes}",
            f"- Bytes saved vs dense archive: {manifest.bytes_saved_vs_dense_archive}",
            f"- Bytes saved vs HFV6 archive: {manifest.bytes_saved_vs_hfv6_archive}",
            f"- Bytes delta vs FEC6 baseline archive: {manifest.bytes_delta_vs_baseline_archive}",
            f"- Rate delta vs FEC6 baseline: {manifest.rate_delta_vs_baseline_archive:.12g}",
            f"- Dense HFV1 payload bytes: {manifest.source_hfv1_payload_bytes}",
            f"- Embedded HFV7 payload bytes: {manifest.output_hfv7_payload_bytes}",
            f"- Embedded HFV7 payload bits: {manifest.output_hfv7_payload_bits}",
            f"- Sparse pair count: {manifest.sparse_pair_count}",
            f"- Compliance note: {manifest.compliance_note}",
            "",
            "Plan command:",
            "",
            "```bash",
            shlex.join(manifest.paired_modal_auth_eval_plan_command_template_not_run),
            "```",
            "",
            "Execute command after full parity proof and claim surface clears:",
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
    parser.add_argument("--hfv6-archive", type=Path, default=DEFAULT_HFV6_ARCHIVE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results") / f"hfv7_exp_golomb_sidecar_candidate_{_utc_stamp()}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_candidate(
        dense_archive=args.dense_archive,
        source_submission_dir=args.source_submission_dir,
        baseline_archive=args.baseline_archive,
        hfv6_archive=args.hfv6_archive,
        output_dir=args.output_dir,
    )
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    (args.output_dir / "hfv7_exp_golomb_manifest.json").write_text(payload, encoding="utf-8")
    (args.output_dir / "hfv7_exp_golomb_manifest.md").write_text(
        render_markdown(manifest),
        encoding="utf-8",
    )
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
