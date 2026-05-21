#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an HFV6 implicit-delta embedded foveation sidecar candidate.

HFV5 stores a profile id plus active-pair delta-varints. HFV6 keeps the
profile-coded active row but stores only the active-pair deltas as raw uint8
trailing bytes. The runtime interprets any non-empty non-HFV3/HFV4/HFV5 trailer
after the PR101/FEC6 selector payload as this implicit HFV6 sidecar.

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

HFV6_PROFILE_ID = 1
DEFAULT_SOURCE_SUBMISSION_DIR = Path(
    "experiments/results/hfv3_embedded_sidecar_candidate_20260521T072238Z/"
    "submission_dir_hfv3_embedded"
)
DEFAULT_HFV5_ARCHIVE = Path(
    "experiments/results/hfv5_profile_sidecar_candidate_20260521T191742Z/archive.zip"
)


@dataclass(frozen=True)
class Hfv6ImplicitDeltaManifest:
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
    output_hfv6_payload_bytes: int
    output_hfv6_payload_sha256: str
    original_x_member_bytes: int
    original_x_member_sha256: str
    embedded_x_member_bytes: int
    embedded_x_member_sha256: str
    profile_id: int
    dense_active_row: tuple[float, float, float, float, float]
    profile_active_row: tuple[float, float, float, float, float]
    profile_row_abs_delta: tuple[float, float, float, float, float]
    sparse_pair_count: int
    sparse_pairs: list[int]
    sparse_pair_deltas: list[int]
    bytes_saved_vs_dense_archive: int
    bytes_saved_vs_hfv5_archive: int
    bytes_delta_vs_baseline_archive: int
    rate_delta_vs_dense_archive: float
    rate_delta_vs_hfv5_archive: float
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


def _pack_hfv6(active_pairs: list[int]) -> bytes:
    deltas = _deltas_from_pairs(active_pairs)
    if not deltas:
        raise ValueError("HFV6 requires at least one active pair")
    if any(delta < 0 or delta > 255 for delta in deltas):
        raise ValueError(f"HFV6 raw uint8 deltas out of range: {deltas}")
    return bytes(deltas)


def _patch_inflate_py(source: str) -> str:
    source = _patch_hfv4_inflate_py(source)
    if "HFV6_implicit_delta_sparse" in source:
        return source
    source = source.replace(
        "    raise ValueError(f\"PR101 frame-selector trailing bytes: pos={pos} total={len(bin_bytes)}\")\n",
        "    return trailer\n",
    )
    marker = "\n\ndef load_foveation_sidecar(src_bin: Path) -> dict[str, object] | None:\n"
    insert = r'''

def hfv6_profile_row() -> tuple[float, float, float, float, float]:
    camera_diag = (float(CAMERA_W * CAMERA_W + CAMERA_H * CAMERA_H)) ** 0.5
    return (
        5.5e-4,
        camera_diag * 0.78,
        1.4,
        (float(CAMERA_W) - 1.0) / 2.0,
        float(CAMERA_H) * 0.45,
    )


def load_hfv6_implicit_delta_params_from_bytes(raw: bytes) -> dict[str, object]:
    if not raw:
        raise ValueError("HFV6 implicit delta payload is empty")
    active_row = hfv6_profile_row()
    row_by_frame: dict[int, tuple[float, float, float, float, float]] = {}
    current_pair = 0
    for index, delta in enumerate(raw):
        current_pair = int(delta) if index == 0 else current_pair + int(delta)
        row_by_frame[2 * current_pair] = active_row
        row_by_frame[2 * current_pair + 1] = active_row
    return {
        "source_format": "HFV6_implicit_delta_sparse",
        "n_frames": 1200,
        "default_row": (0.0, 0.0, 0.0, 0.0, 0.0),
        "row_by_frame": row_by_frame,
        "pair_count": len(raw),
    }
'''
    source = source.replace(marker, insert + marker)
    source = source.replace(
        "    raise ValueError(f\"unsupported embedded foveation magic: {trailer[:4]!r}\")\n",
        "    return load_hfv6_implicit_delta_params_from_bytes(trailer)\n",
    )
    source = source.replace(
        '"HFV4_repeated_pair_delta_sparse"}',
        '"HFV4_repeated_pair_delta_sparse", "HFV6_implicit_delta_sparse"}',
    )
    if "HFV6_implicit_delta_sparse" not in source:
        raise ValueError("failed to patch inflate.py with HFV6 decoder")
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


def _write_archive_manifest(
    *,
    submission_dir: Path,
    archive_path: Path,
    dense_archive: Path,
    baseline_archive: Path,
    hfv1_raw: bytes,
    hfv6_raw: bytes,
    active_pairs: list[int],
    profile_active_row: tuple[float, float, float, float, float],
    dense_active_row: tuple[float, float, float, float, float],
    paired_plan_cmd: list[str],
    paired_execute_cmd: list[str],
) -> Path:
    manifest_path = submission_dir / "archive_manifest.json"
    profile_row_abs_delta = tuple(
        abs(float(a) - float(b)) for a, b in zip(dense_active_row, profile_active_row, strict=True)
    )
    payload = {
        "schema": "hfv6_implicit_delta_sidecar_candidate_manifest_v1",
        "generated_at_utc": _utc_iso(),
        "archive_path": _repo_rel(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256_file(archive_path),
        "members": _zip_member_records(archive_path),
        "member_shape": "single_member_x_with_pr101_source_payload_fec6_selector_and_embedded_hfv6_implicit_delta_sidecar",
        "source_dense_archive": _repo_rel(dense_archive),
        "source_dense_archive_bytes": dense_archive.stat().st_size,
        "source_dense_archive_sha256": _sha256_file(dense_archive),
        "baseline_archive": _repo_rel(baseline_archive),
        "baseline_archive_bytes": baseline_archive.stat().st_size,
        "baseline_archive_sha256": _sha256_file(baseline_archive),
        "source_hfv1_payload_bytes": len(hfv1_raw),
        "source_hfv1_payload_sha256": _sha256_bytes(hfv1_raw),
        "output_hfv6_payload_bytes": len(hfv6_raw),
        "output_hfv6_payload_sha256": _sha256_bytes(hfv6_raw),
        "profile_id": HFV6_PROFILE_ID,
        "dense_active_row": dense_active_row,
        "profile_active_row": profile_active_row,
        "profile_row_abs_delta": profile_row_abs_delta,
        "sparse_pair_count": len(active_pairs),
        "sparse_pairs": active_pairs,
        "sparse_pair_deltas": _deltas_from_pairs(active_pairs),
        "target_modes": ["contest_exact_eval"],
        "dispatch_blockers": [
            "exact_contest_cpu_eval_missing",
            "exact_contest_cuda_eval_missing",
            "full_parity_proof_required_before_execute",
            "lane_dispatch_claim_required_before_execute",
            "implicit_trailer_profile_compliance_review_required_before_submission",
        ],
        "paired_auth_eval_required": True,
        "paired_auth_eval_plan_ready": False,
        "paired_modal_auth_eval_plan_command_template_not_run": paired_plan_cmd,
        "paired_modal_auth_eval_execute_command_template_after_claim_surface_clear": paired_execute_cmd,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_submission": False,
        "ready_for_exact_eval_dispatch": False,
        "compliance_note": (
            "HFV6 stores only raw uint8 active-pair deltas after the selector payload. "
            "The active-row profile and trailer interpretation live in inflate.py. "
            "Treat as research-only until contest-compliance review accepts the "
            "implicit-trailer profile-code interpretation."
        ),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def build_candidate(
    *,
    dense_archive: Path,
    source_submission_dir: Path,
    baseline_archive: Path,
    hfv5_archive: Path,
    output_dir: Path,
) -> Hfv6ImplicitDeltaManifest:
    with zipfile.ZipFile(dense_archive) as archive:
        hfv1_raw = archive.read("foveation_params.bin")
        x_payload = archive.read("x")
    n_frames, frame_height, frame_width, rows = _read_dense_hfv1(hfv1_raw)
    if n_frames != 1200 or frame_height != CAMERA_H or frame_width != CAMERA_W:
        raise ValueError("HFV6 implicit form expects the PR101/FEC6 1200-frame camera geometry")
    _default_row, dense_active_row, active_pairs = _repeated_sparse_pairs(rows)
    profile_active_row = _profile_row(HFV6_PROFILE_ID)
    profile_row_abs_delta = tuple(
        abs(float(a) - float(b)) for a, b in zip(dense_active_row, profile_active_row, strict=True)
    )
    hfv6_raw = _pack_hfv6(active_pairs)
    embedded_x = x_payload + hfv6_raw
    archive_path = output_dir / "archive.zip"
    _write_stored_zip(archive_path, [("x", embedded_x)])
    submission_dir = _copy_runtime(source_submission_dir, output_dir / "submission_dir_hfv6_implicit_delta")
    submission_archive = submission_dir / "archive.zip"
    shutil.copy2(archive_path, submission_archive)

    lane_id_base = "hfv6_implicit_delta_pr101_hfv1_sidecar_exact_eval"
    run_id = f"hfv6_implicit_delta_pr101_{_sha256_file(submission_archive)[:12]}"
    claim_notes = (
        "HFV6 implicit-delta sidecar candidate; score_claim=false until full parity "
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
        claim_agent="codex:hfv6_implicit_delta_sidecar",
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
        claim_agent="codex:hfv6_implicit_delta_sidecar",
        claim_notes=claim_notes,
    )
    generated_manifest = _write_archive_manifest(
        submission_dir=submission_dir,
        archive_path=submission_archive,
        dense_archive=dense_archive,
        baseline_archive=baseline_archive,
        hfv1_raw=hfv1_raw,
        hfv6_raw=hfv6_raw,
        active_pairs=active_pairs,
        profile_active_row=profile_active_row,
        dense_active_row=dense_active_row,
        paired_plan_cmd=paired_plan_cmd,
        paired_execute_cmd=paired_execute_cmd,
    )
    return Hfv6ImplicitDeltaManifest(
        schema="hfv6_implicit_delta_sidecar_candidate_v1",
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
        output_hfv6_payload_bytes=len(hfv6_raw),
        output_hfv6_payload_sha256=_sha256_bytes(hfv6_raw),
        original_x_member_bytes=len(x_payload),
        original_x_member_sha256=_sha256_bytes(x_payload),
        embedded_x_member_bytes=len(embedded_x),
        embedded_x_member_sha256=_sha256_bytes(embedded_x),
        profile_id=HFV6_PROFILE_ID,
        dense_active_row=dense_active_row,
        profile_active_row=profile_active_row,
        profile_row_abs_delta=profile_row_abs_delta,
        sparse_pair_count=len(active_pairs),
        sparse_pairs=active_pairs,
        sparse_pair_deltas=_deltas_from_pairs(active_pairs),
        bytes_saved_vs_dense_archive=dense_archive.stat().st_size - archive_path.stat().st_size,
        bytes_saved_vs_hfv5_archive=hfv5_archive.stat().st_size - archive_path.stat().st_size,
        bytes_delta_vs_baseline_archive=archive_path.stat().st_size - baseline_archive.stat().st_size,
        rate_delta_vs_dense_archive=_rate_delta(archive_path.stat().st_size - dense_archive.stat().st_size),
        rate_delta_vs_hfv5_archive=_rate_delta(archive_path.stat().st_size - hfv5_archive.stat().st_size),
        rate_delta_vs_baseline_archive=_rate_delta(archive_path.stat().st_size - baseline_archive.stat().st_size),
        target_modes=["contest_exact_eval"],
        dispatch_blockers=[
            "exact_contest_cpu_eval_missing",
            "exact_contest_cuda_eval_missing",
            "full_parity_proof_required_before_execute",
            "lane_dispatch_claim_required_before_execute",
            "implicit_trailer_profile_compliance_review_required_before_submission",
        ],
        paired_auth_eval_required=True,
        paired_auth_eval_plan_ready=False,
        paired_modal_auth_eval_plan_command_template_not_run=paired_plan_cmd,
        paired_modal_auth_eval_execute_command_template_after_claim_surface_clear=paired_execute_cmd,
        compliance_note=(
            "HFV6 stores only raw uint8 active-pair deltas after the selector payload. "
            "The active-row profile and trailer interpretation live in inflate.py. "
            "Treat as research-only until contest-compliance review accepts the "
            "implicit-trailer profile-code interpretation."
        ),
    )


def render_markdown(manifest: Hfv6ImplicitDeltaManifest) -> str:
    return "\n".join(
        [
            "# HFV6 Implicit Delta Sidecar Candidate",
            "",
            f"- Generated UTC: {manifest.generated_at_utc}",
            f"- Source dense archive: `{manifest.source_dense_archive}`",
            f"- Output archive: `{manifest.output_archive}`",
            f"- Output submission dir: `{manifest.output_submission_dir}`",
            f"- Output submission archive: `{manifest.output_submission_archive}`",
            f"- Dense archive bytes: {manifest.source_dense_archive_bytes}",
            f"- Embedded archive bytes: {manifest.output_archive_bytes}",
            f"- Bytes saved vs dense archive: {manifest.bytes_saved_vs_dense_archive}",
            f"- Bytes saved vs HFV5 archive: {manifest.bytes_saved_vs_hfv5_archive}",
            f"- Bytes delta vs FEC6 baseline archive: {manifest.bytes_delta_vs_baseline_archive}",
            f"- Rate delta vs FEC6 baseline: {manifest.rate_delta_vs_baseline_archive:.12g}",
            f"- Dense HFV1 payload bytes: {manifest.source_hfv1_payload_bytes}",
            f"- Embedded HFV6 payload bytes: {manifest.output_hfv6_payload_bytes}",
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
    parser.add_argument("--hfv5-archive", type=Path, default=DEFAULT_HFV5_ARCHIVE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results") / f"hfv6_implicit_delta_sidecar_candidate_{_utc_stamp()}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_candidate(
        dense_archive=args.dense_archive,
        source_submission_dir=args.source_submission_dir,
        baseline_archive=args.baseline_archive,
        hfv5_archive=args.hfv5_archive,
        output_dir=args.output_dir,
    )
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    (args.output_dir / "hfv6_implicit_delta_manifest.json").write_text(payload, encoding="utf-8")
    (args.output_dir / "hfv6_implicit_delta_manifest.md").write_text(
        render_markdown(manifest),
        encoding="utf-8",
    )
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
