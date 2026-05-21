#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an HFV8 explicit-row embedded foveation sidecar candidate.

HFV7 is the best byte result for the active-pair delta sequence, but it keeps
the active foveation row as a runtime profile. HFV8 pays for that row inside
``archive.zip`` as five float32 values, followed by the same HFV7 Exp-Golomb
order-3 delta stream. This is rate-worse than HFV7, but removes the
profile-coded active-row side-information blocker.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import struct
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
from tools.build_hfv7_exp_golomb_sidecar_candidate import (
    DEFAULT_SOURCE_SUBMISSION_DIR,
    HFV7_EXP_GOLOMB_ORDER,
    _pack_hfv7,
    _patch_inflate_py as _patch_hfv7_inflate_py,
)

HFV8_ROW_STRUCT = struct.Struct("<fffff")
HFV8_EXPLICIT_ROW_PAYLOAD_BYTES = HFV8_ROW_STRUCT.size + 12
DEFAULT_HFV7_ARCHIVE = Path(
    "experiments/results/hfv7_exp_golomb_sidecar_candidate_20260521T193740Z/archive.zip"
)


@dataclass(frozen=True)
class Hfv8ExplicitRowManifest:
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
    output_hfv8_payload_bytes: int
    output_hfv8_payload_sha256: str
    explicit_row_bytes: int
    exp_golomb_delta_bytes: int
    exp_golomb_delta_bits: int
    original_x_member_bytes: int
    original_x_member_sha256: str
    embedded_x_member_bytes: int
    embedded_x_member_sha256: str
    dense_active_row: tuple[float, float, float, float, float]
    explicit_row_unpacked: tuple[float, float, float, float, float]
    explicit_row_abs_delta: tuple[float, float, float, float, float]
    sparse_pair_count: int
    sparse_pairs: list[int]
    sparse_pair_deltas: list[int]
    bytes_saved_vs_dense_archive: int
    bytes_delta_vs_hfv7_archive: int
    bytes_delta_vs_baseline_archive: int
    rate_delta_vs_dense_archive: float
    rate_delta_vs_hfv7_archive: float
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


def _pack_hfv8(
    active_row: tuple[float, float, float, float, float],
    active_pairs: list[int],
) -> tuple[bytes, int]:
    delta_payload, delta_bits = _pack_hfv7(active_pairs)
    if len(delta_payload) != 12:
        raise ValueError(f"HFV8 expects the current HFV7 delta payload to be 12 bytes, got {len(delta_payload)}")
    return HFV8_ROW_STRUCT.pack(*[float(v) for v in active_row]) + delta_payload, delta_bits


def _patch_inflate_py(source: str) -> str:
    source = _patch_hfv7_inflate_py(source)
    if "HFV8_explicit_row_sparse" in source:
        return source
    marker = "\n\ndef load_foveation_sidecar(src_bin: Path) -> dict[str, object] | None:\n"
    insert = r'''

def load_hfv8_explicit_row_params_from_bytes(raw: bytes) -> dict[str, object]:
    if len(raw) != 32:
        raise ValueError(f"HFV8 explicit-row payload must be 32 bytes, got {len(raw)}")
    active_row = HFV8_ROW_STRUCT.unpack(raw[: HFV8_ROW_STRUCT.size])
    deltas = decode_hfv7_exp_golomb_order3(raw[HFV8_ROW_STRUCT.size :])
    row_by_frame: dict[int, tuple[float, float, float, float, float]] = {}
    current_pair = 0
    for index, delta in enumerate(deltas):
        current_pair = int(delta) if index == 0 else current_pair + int(delta)
        row_by_frame[2 * current_pair] = active_row
        row_by_frame[2 * current_pair + 1] = active_row
    return {
        "source_format": "HFV8_explicit_row_sparse",
        "n_frames": 1200,
        "default_row": (0.0, 0.0, 0.0, 0.0, 0.0),
        "row_by_frame": row_by_frame,
        "pair_count": len(row_by_frame) // 2,
        "explicit_row_bytes": HFV8_ROW_STRUCT.size,
        "exp_golomb_order": 3,
    }
'''
    source = source.replace(marker, insert + marker)
    if "HFV8_ROW_STRUCT = " not in source:
        source = source.replace(
            'HFV4_HEADER_STRUCT = struct.Struct("<4sBfffff")\n',
            'HFV4_HEADER_STRUCT = struct.Struct("<4sBfffff")\n'
            'HFV8_ROW_STRUCT = struct.Struct("<fffff")\n',
        )
    source = source.replace(
        "    if len(trailer) == 12:\n"
        "        return load_hfv7_exp_golomb_params_from_bytes(trailer)\n",
        "    if len(trailer) == 32:\n"
        "        return load_hfv8_explicit_row_params_from_bytes(trailer)\n"
        "    if len(trailer) == 12:\n"
        "        return load_hfv7_exp_golomb_params_from_bytes(trailer)\n",
    )
    source = source.replace(
        '"HFV4_repeated_pair_delta_sparse", "HFV7_exp_golomb_sparse"}',
        '"HFV4_repeated_pair_delta_sparse", "HFV7_exp_golomb_sparse", "HFV8_explicit_row_sparse"}',
    )
    if "HFV8_explicit_row_sparse" not in source:
        raise ValueError("failed to patch inflate.py with HFV8 explicit-row decoder")
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


def _manifest_common(
    *,
    archive_path: Path,
    dense_archive: Path,
    baseline_archive: Path,
    hfv1_raw: bytes,
    hfv8_raw: bytes,
    hfv8_delta_bits: int,
    active_pairs: list[int],
    dense_active_row: tuple[float, float, float, float, float],
) -> dict[str, Any]:
    unpacked = HFV8_ROW_STRUCT.unpack(hfv8_raw[: HFV8_ROW_STRUCT.size])
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
        "output_hfv8_payload_bytes": len(hfv8_raw),
        "output_hfv8_payload_sha256": _sha256_bytes(hfv8_raw),
        "explicit_row_bytes": HFV8_ROW_STRUCT.size,
        "exp_golomb_delta_bytes": len(hfv8_raw) - HFV8_ROW_STRUCT.size,
        "exp_golomb_delta_bits": hfv8_delta_bits,
        "exp_golomb_order": HFV7_EXP_GOLOMB_ORDER,
        "dense_active_row": dense_active_row,
        "explicit_row_unpacked": unpacked,
        "explicit_row_abs_delta": tuple(
            abs(float(a) - float(b)) for a, b in zip(dense_active_row, unpacked, strict=True)
        ),
        "sparse_pair_count": len(active_pairs),
        "sparse_pairs": active_pairs,
        "sparse_pair_deltas": _deltas_from_pairs(active_pairs),
        "target_modes": ["contest_exact_eval"],
        "dispatch_blockers": [
            "exact_contest_cpu_eval_missing",
            "exact_contest_cuda_eval_missing",
            "shell_parity_proof_required_before_execute",
            "lane_dispatch_claim_required_before_execute",
            "implicit_trailer_format_compliance_review_required_before_submission",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_submission": False,
        "ready_for_exact_eval_dispatch": False,
        "compliance_note": (
            "HFV8 stores the active foveation row as five float32 values followed "
            "by the Exp-Golomb-coded active-pair delta sequence. This removes the "
            "HFV7 runtime-profile active-row side-information blocker, but still "
            "requires review of the implicit 32-byte trailer grammar."
        ),
    }


def _write_archive_manifest(
    *,
    submission_dir: Path,
    archive_path: Path,
    dense_archive: Path,
    baseline_archive: Path,
    hfv1_raw: bytes,
    hfv8_raw: bytes,
    hfv8_delta_bits: int,
    active_pairs: list[int],
    dense_active_row: tuple[float, float, float, float, float],
    paired_plan_cmd: list[str],
    paired_execute_cmd: list[str],
) -> Path:
    manifest_path = submission_dir / "archive_manifest.json"
    payload = {
        "schema": "hfv8_explicit_row_sidecar_candidate_manifest_v1",
        "generated_at_utc": _utc_iso(),
        "member_shape": "single_member_x_with_pr101_source_payload_fec6_selector_and_embedded_hfv8_explicit_row_sidecar",
        **_manifest_common(
            archive_path=archive_path,
            dense_archive=dense_archive,
            baseline_archive=baseline_archive,
            hfv1_raw=hfv1_raw,
            hfv8_raw=hfv8_raw,
            hfv8_delta_bits=hfv8_delta_bits,
            active_pairs=active_pairs,
            dense_active_row=dense_active_row,
        ),
        "paired_auth_eval_required": True,
        "paired_auth_eval_plan_ready": False,
        "paired_modal_auth_eval_plan_command_template_not_run": paired_plan_cmd,
        "paired_modal_auth_eval_execute_command_template_after_claim_surface_clear": paired_execute_cmd,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _write_research_submission_metadata(
    submission_dir: Path,
    manifest: Hfv8ExplicitRowManifest,
) -> None:
    readme = "\n".join(
        [
            "# HFV8 Explicit-Row Sidecar Candidate",
            "",
            "Research-only byte-closed candidate. HFV8 is the compliance-safe",
            "counterpart to HFV7: it charges the active foveation row inside",
            "`archive.zip` as float32 values, then appends the 12-byte HFV7",
            "Exp-Golomb delta stream.",
            "",
            f"- Archive: `archive.zip`, SHA-256 `{manifest.output_archive_sha256}`, {manifest.output_archive_bytes} bytes.",
            f"- Payload: {manifest.output_hfv8_payload_bytes} bytes = {manifest.explicit_row_bytes} row bytes + {manifest.exp_golomb_delta_bytes} delta bytes.",
            f"- Bytes over FEC6/PR110 baseline: {manifest.bytes_delta_vs_baseline_archive}.",
            f"- Rate delta vs FEC6/PR110 baseline: {manifest.rate_delta_vs_baseline_archive:.15g}.",
            "- Score claim: false.",
            "- Promotion eligible: false.",
            "- Ready for exact eval dispatch: false.",
            "",
            "Compliance status:",
            "",
            manifest.compliance_note,
            "",
        ]
    )
    report = "\n".join(
        [
            "HFV8 Explicit-Row Sidecar Candidate - research-only report",
            f"generated_at_utc: {manifest.generated_at_utc}",
            f"archive_sha256: {manifest.output_archive_sha256}",
            f"archive_size_bytes: {manifest.output_archive_bytes}",
            f"payload_bytes: {manifest.output_hfv8_payload_bytes}",
            f"explicit_row_bytes: {manifest.explicit_row_bytes}",
            f"exp_golomb_delta_bytes: {manifest.exp_golomb_delta_bytes}",
            f"bytes_delta_vs_fec6_pr110_baseline: {manifest.bytes_delta_vs_baseline_archive}",
            f"rate_delta_vs_fec6_pr110_baseline: {manifest.rate_delta_vs_baseline_archive:.15g}",
            "score_claim: false",
            "contest_cpu_exact_eval: not_run",
            "contest_cuda_exact_eval: not_run",
            f"compliance_note: {manifest.compliance_note}",
            "",
        ]
    )
    (submission_dir / "README.md").write_text(readme, encoding="utf-8")
    (submission_dir / "report.txt").write_text(report, encoding="utf-8")


def build_candidate(
    *,
    dense_archive: Path,
    source_submission_dir: Path,
    baseline_archive: Path,
    hfv7_archive: Path,
    output_dir: Path,
) -> Hfv8ExplicitRowManifest:
    with zipfile.ZipFile(dense_archive) as archive:
        hfv1_raw = archive.read("foveation_params.bin")
        x_payload = archive.read("x")
    n_frames, frame_height, frame_width, rows = _read_dense_hfv1(hfv1_raw)
    if n_frames != 1200 or frame_height != CAMERA_H or frame_width != CAMERA_W:
        raise ValueError("HFV8 explicit-row form expects PR101/FEC6 camera geometry")
    _default_row, dense_active_row, active_pairs = _repeated_sparse_pairs(rows)
    hfv8_raw, hfv8_delta_bits = _pack_hfv8(dense_active_row, active_pairs)
    if len(hfv8_raw) != HFV8_EXPLICIT_ROW_PAYLOAD_BYTES:
        raise ValueError(f"unexpected HFV8 payload length: {len(hfv8_raw)}")
    embedded_x = x_payload + hfv8_raw
    archive_path = output_dir / "archive.zip"
    _write_stored_zip(archive_path, [("x", embedded_x)])
    submission_dir = _copy_runtime(source_submission_dir, output_dir / "submission_dir_hfv8_explicit_row")
    submission_archive = submission_dir / "archive.zip"
    shutil.copy2(archive_path, submission_archive)

    lane_id_base = "hfv8_explicit_row_pr101_hfv1_sidecar_exact_eval"
    run_id = f"hfv8_explicit_row_pr101_{_sha256_file(submission_archive)[:12]}"
    claim_notes = (
        "HFV8 explicit-row sidecar candidate; score_claim=false until shell parity "
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
        claim_agent="codex:hfv8_explicit_row_sidecar",
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
        claim_agent="codex:hfv8_explicit_row_sidecar",
        claim_notes=claim_notes,
    )
    generated_manifest = _write_archive_manifest(
        submission_dir=submission_dir,
        archive_path=submission_archive,
        dense_archive=dense_archive,
        baseline_archive=baseline_archive,
        hfv1_raw=hfv1_raw,
        hfv8_raw=hfv8_raw,
        hfv8_delta_bits=hfv8_delta_bits,
        active_pairs=active_pairs,
        dense_active_row=dense_active_row,
        paired_plan_cmd=paired_plan_cmd,
        paired_execute_cmd=paired_execute_cmd,
    )
    unpacked = HFV8_ROW_STRUCT.unpack(hfv8_raw[: HFV8_ROW_STRUCT.size])
    manifest = Hfv8ExplicitRowManifest(
        schema="hfv8_explicit_row_sidecar_candidate_v1",
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
        output_hfv8_payload_bytes=len(hfv8_raw),
        output_hfv8_payload_sha256=_sha256_bytes(hfv8_raw),
        explicit_row_bytes=HFV8_ROW_STRUCT.size,
        exp_golomb_delta_bytes=len(hfv8_raw) - HFV8_ROW_STRUCT.size,
        exp_golomb_delta_bits=hfv8_delta_bits,
        original_x_member_bytes=len(x_payload),
        original_x_member_sha256=_sha256_bytes(x_payload),
        embedded_x_member_bytes=len(embedded_x),
        embedded_x_member_sha256=_sha256_bytes(embedded_x),
        dense_active_row=dense_active_row,
        explicit_row_unpacked=unpacked,
        explicit_row_abs_delta=tuple(
            abs(float(a) - float(b)) for a, b in zip(dense_active_row, unpacked, strict=True)
        ),
        sparse_pair_count=len(active_pairs),
        sparse_pairs=active_pairs,
        sparse_pair_deltas=_deltas_from_pairs(active_pairs),
        bytes_saved_vs_dense_archive=dense_archive.stat().st_size - archive_path.stat().st_size,
        bytes_delta_vs_hfv7_archive=archive_path.stat().st_size - hfv7_archive.stat().st_size,
        bytes_delta_vs_baseline_archive=archive_path.stat().st_size - baseline_archive.stat().st_size,
        rate_delta_vs_dense_archive=_rate_delta(archive_path.stat().st_size - dense_archive.stat().st_size),
        rate_delta_vs_hfv7_archive=_rate_delta(archive_path.stat().st_size - hfv7_archive.stat().st_size),
        rate_delta_vs_baseline_archive=_rate_delta(archive_path.stat().st_size - baseline_archive.stat().st_size),
        target_modes=["contest_exact_eval"],
        dispatch_blockers=[
            "exact_contest_cpu_eval_missing",
            "exact_contest_cuda_eval_missing",
            "shell_parity_proof_required_before_execute",
            "lane_dispatch_claim_required_before_execute",
            "implicit_trailer_format_compliance_review_required_before_submission",
        ],
        paired_auth_eval_required=True,
        paired_auth_eval_plan_ready=False,
        paired_modal_auth_eval_plan_command_template_not_run=paired_plan_cmd,
        paired_modal_auth_eval_execute_command_template_after_claim_surface_clear=paired_execute_cmd,
        compliance_note=(
            "HFV8 stores the active foveation row as five float32 values followed "
            "by the Exp-Golomb-coded active-pair delta sequence. This removes the "
            "HFV7 runtime-profile active-row side-information blocker, but still "
            "requires review of the implicit 32-byte trailer grammar."
        ),
    )
    _write_research_submission_metadata(submission_dir, manifest)
    return manifest


def render_markdown(manifest: Hfv8ExplicitRowManifest) -> str:
    return "\n".join(
        [
            "# HFV8 Explicit-Row Sidecar Candidate",
            "",
            f"- Generated UTC: {manifest.generated_at_utc}",
            f"- Source dense archive: `{manifest.source_dense_archive}`",
            f"- Output archive: `{manifest.output_archive}`",
            f"- Output submission dir: `{manifest.output_submission_dir}`",
            f"- Output submission archive: `{manifest.output_submission_archive}`",
            f"- Dense archive bytes: {manifest.source_dense_archive_bytes}",
            f"- Embedded archive bytes: {manifest.output_archive_bytes}",
            f"- Payload bytes: {manifest.output_hfv8_payload_bytes}",
            f"- Explicit row bytes: {manifest.explicit_row_bytes}",
            f"- Exp-Golomb delta bytes: {manifest.exp_golomb_delta_bytes}",
            f"- Bytes saved vs dense archive: {manifest.bytes_saved_vs_dense_archive}",
            f"- Bytes delta vs HFV7 archive: {manifest.bytes_delta_vs_hfv7_archive}",
            f"- Bytes delta vs FEC6 baseline archive: {manifest.bytes_delta_vs_baseline_archive}",
            f"- Rate delta vs FEC6 baseline: {manifest.rate_delta_vs_baseline_archive:.12g}",
            f"- Sparse pair count: {manifest.sparse_pair_count}",
            f"- Compliance note: {manifest.compliance_note}",
            "",
            "Plan command:",
            "",
            "```bash",
            shlex.join(manifest.paired_modal_auth_eval_plan_command_template_not_run),
            "```",
            "",
            "Execute command after shell parity proof and claim surface clears:",
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
    parser.add_argument("--hfv7-archive", type=Path, default=DEFAULT_HFV7_ARCHIVE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results") / f"hfv8_explicit_row_sidecar_candidate_{_utc_stamp()}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_candidate(
        dense_archive=args.dense_archive,
        source_submission_dir=args.source_submission_dir,
        baseline_archive=args.baseline_archive,
        hfv7_archive=args.hfv7_archive,
        output_dir=args.output_dir,
    )
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    (args.output_dir / "hfv8_explicit_row_manifest.json").write_text(payload, encoding="utf-8")
    (args.output_dir / "hfv8_explicit_row_manifest.md").write_text(
        render_markdown(manifest),
        encoding="utf-8",
    )
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
