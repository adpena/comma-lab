# SPDX-License-Identifier: MIT
"""Archive-bound candidate bridge for HPRC packets."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.archive_byte_profile import CONTEST_ORIGINAL_BYTES, contest_rate_term
from tac.optimization.archive_bound_candidate_runtime_bridge import (
    emit_archive_bound_candidate_runtime_package,
)
from tac.repo_io import sha256_file
from tac.substrates._shared.pact_nerv_full_main import (
    build_archive_zip,
    write_contest_runtime,
)
from tac.substrates.hprc.archive import (
    HprcPacketConfig,
    HprcSectionKind,
    pack_hprc_packet,
    parse_hprc_packet,
)
from tac.substrates.hprc.inflate import (
    CAMERA_H,
    CAMERA_W,
    CONTEST_RAW_BYTES,
    HPRC_METADATA_ONLY_SECTIONS,
    HPRC_PIXEL_DRIVING_SECTIONS,
    hprc_preview_digest,
)
from tac.substrates.hprc.learned_receiver import (
    COMPACT_RECEIVER_MODE,
    is_compact_receiver_packet,
    mutate_compact_receiver_section,
)
from tac.substrates.hprc.resolution_contract import hprc_resolution_contract

HPRC_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA = (
    "hprc_archive_bound_adapter_package.v1"
)
HPRC_RECEIVER_PROOF_SCHEMA = "hprc_generated_receiver_proof.v1"
HPRC_ARCHIVE_BOUND_ADAPTER_ID = "hprc_archive_export"
HPRC_ARCHIVE_CANDIDATE_FAMILY = "hprc_hierarchical_predictive_receiver_codec"
HPRC_ARCHIVE_TRANSFORM_KIND = (
    "hprc_predictive_coding_compact_receiver_latent_stream_plus_scorer_residual_sidecar"
)
HPRC_RUNTIME_MODULE_FILES: tuple[str, ...] = (
    "archive.py",
    "inflate.py",
    "learned_receiver.py",
)
HPRC_SUB019_ZERO_DISTORTION_BYTE_CEILING = int(0.19 * CONTEST_ORIGINAL_BYTES / 25)
HPRC_RECEIVER_PROOF_SCRATCH_BYTES = CONTEST_RAW_BYTES + (1 << 30)

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
}


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_output_dir(
    output_dir: str | Path,
    *,
    repo_root: str | Path | None,
) -> tuple[Path, Path]:
    root = Path(repo_root) if repo_root is not None else _repo_root_from_here()
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return root, out_dir


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return (
            path.resolve(strict=False)
            .relative_to(repo_root.resolve(strict=False))
            .as_posix()
        )
    except ValueError:
        return path.as_posix()


def _assert_receiver_proof_scratch_space(output_dir: Path) -> int:
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < HPRC_RECEIVER_PROOF_SCRATCH_BYTES:
        raise RuntimeError(
            "insufficient scratch space for HPRC receiver proof: "
            f"free={free_bytes}, required={HPRC_RECEIVER_PROOF_SCRATCH_BYTES}, "
            f"path={output_dir}"
        )
    return int(free_bytes)


def _expected_receiver_output_bytes_for_packet(packet: Any) -> int:
    return int(packet.config.frames) * CAMERA_H * CAMERA_W * 3


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    )


def build_minimal_hprc_v0_packet(
    *,
    config: HprcPacketConfig | None = None,
    decoder_family_id: int = 95,
    include_residual_stub: bool = True,
) -> bytes:
    """Build a tiny deterministic HPRC V0 receiver-scaffold packet."""

    cfg = config or HprcPacketConfig(decoder_family_id=decoder_family_id)
    manifest = {
        "schema": "hprc_minimal_v0_packet_manifest.v1",
        "candidate_kind": "receiver_scaffold_not_trained_renderer",
        "base_receiver_family": "pr95_hnerv_control_or_rnerv_pact_future",
        "z8_role": "teacher_residual_sidecar_future",
        "score_claim": False,
        "promotion_eligible": False,
    }
    sections: dict[HprcSectionKind, bytes] = {
        HprcSectionKind.DECODER_QW: b"hprc-v0-decoder-qw-placeholder",
        HprcSectionKind.LATENTS_RC: bytes((idx * 17 + 5) % 256 for idx in range(600)),
        HprcSectionKind.RDO_PLAN: _json_bytes(
            {
                "schema": "hprc_rdo_plan.v1",
                "authority": "none_yet",
                "requires_full_video_p18_p19": True,
                "byte_ceiling_sub019": HPRC_SUB019_ZERO_DISTORTION_BYTE_CEILING,
            }
        ),
        HprcSectionKind.MANIFEST_JSON: _json_bytes(manifest),
    }
    if include_residual_stub:
        sections[HprcSectionKind.RESIDUAL_RC] = b""
    return pack_hprc_packet(sections, config=cfg)


def _mutate_payload(payload: bytes, *, salt: int) -> bytes:
    if not payload:
        return bytes([salt & 0xFF])
    data = bytearray(payload)
    data[0] ^= (0xA5 ^ salt) & 0xFF
    return bytes(data)


def build_hprc_section_mutation_proof(
    archive_bytes: bytes | bytearray | memoryview,
) -> dict[str, Any]:
    """Prove which sections are pixel-driving under valid packet mutations."""

    packet = parse_hprc_packet(archive_bytes)
    compact_receiver_packet = is_compact_receiver_packet(packet)
    base_preview = hprc_preview_digest(archive_bytes)
    section_payloads = packet.section_map()
    per_section: list[dict[str, Any]] = []
    blockers: list[str] = []
    for index, section in enumerate(packet.sections):
        mutated = dict(section_payloads)
        semantic_mutation = mutate_compact_receiver_section(
            packet,
            section.kind,
            salt=index + 1,
        )
        mutated[section.kind] = (
            semantic_mutation
            if semantic_mutation is not None
            else _mutate_payload(section.payload, salt=index + 1)
        )
        mutated_packet = pack_hprc_packet(mutated, config=packet.config)
        mutated_preview = hprc_preview_digest(mutated_packet)
        preview_changed = mutated_preview != base_preview
        pixel_driving_expected = section.kind in HPRC_PIXEL_DRIVING_SECTIONS
        metadata_only_expected = section.kind in HPRC_METADATA_ONLY_SECTIONS
        if pixel_driving_expected and not preview_changed:
            blockers.append(f"{section.name}_semantic_mutation_did_not_change_preview")
        if metadata_only_expected and preview_changed:
            blockers.append(f"{section.name}_metadata_mutation_changed_preview")
        per_section.append(
            {
                "section": section.name,
                "id": int(section.kind),
                "section_sha256": section.sha256,
                "valid_packet_mutation": True,
                "preview_digest_before": base_preview,
                "preview_digest_after": mutated_preview,
                "receiver_preview_changed": preview_changed,
                "pixel_driving_expected": pixel_driving_expected,
                "metadata_only_expected": metadata_only_expected,
                "proof_scope": (
                    "valid_semantic_packet_mutation_preview"
                    if compact_receiver_packet
                    else (
                        "valid_packet_mutation_preview_for_v0_scaffold; "
                        "full_receiver_replay is run for the unmutated "
                        "archive-bound candidate"
                    )
                ),
                "compact_receiver_mode": compact_receiver_packet,
            }
        )
    return {
        "schema": "hprc_section_semantic_mutation_proof.v1",
        "archive_sha256": _sha256_bytes(bytes(archive_bytes)),
        "receiver_mode": (
            COMPACT_RECEIVER_MODE if compact_receiver_packet else "hprc_v0_digest_scaffold"
        ),
        "base_preview_digest": base_preview,
        "pixel_driving_sections": [
            section.name
            for section in packet.sections
            if section.kind in HPRC_PIXEL_DRIVING_SECTIONS
        ],
        "metadata_only_sections": [
            section.name
            for section in packet.sections
            if section.kind in HPRC_METADATA_ONLY_SECTIONS
        ],
        "per_section": per_section,
        "blockers": blockers,
        "section_mutation_preview_ready": not blockers,
        **FALSE_AUTHORITY,
    }


def _runtime_payload_consumption_manifest(
    section_proof: Mapping[str, Any],
    *,
    packet: Any | None = None,
) -> dict[str, Any]:
    section_pixel_proofs: dict[str, dict[str, Any]] = {}
    pixel_consumed: list[str] = []
    for raw in section_proof.get("per_section", []):
        if not isinstance(raw, Mapping):
            continue
        section = str(raw.get("section") or "").strip()
        if not section:
            continue
        if raw.get("pixel_driving_expected") is not True:
            continue
        proven = raw.get("receiver_preview_changed") is True
        section_pixel_proofs[section] = {
            "schema": "hprc_section_pixel_consumption_proof.v1",
            "section": section,
            "pixel_consumption_proven": proven,
            "proof_kind": "valid_semantic_packet_mutation_preview",
            "preview_digest_before": raw.get("preview_digest_before"),
            "preview_digest_after": raw.get("preview_digest_after"),
            "full_receiver_replay_required_for_candidate": True,
        }
        if proven:
            pixel_consumed.append(section)
    compact_receiver = str(section_proof.get("receiver_mode") or "") == COMPACT_RECEIVER_MODE
    if compact_receiver:
        next_tasks = [
            "attach_z8_scorer_weighted_residual_sidecar",
            "prove_mamba_dreamer_wyner_ziv_sections_drive_receiver_pixels",
            "run_exact_cpu_cuda_auth_eval_for_compact_receiver_candidate",
        ]
        status_note = "compact learned receiver sections are semantic-pixel-consuming"
    else:
        next_tasks = [
            "replace_hprc_v0_receiver_scaffold_with_trained_renderer_export",
            "attach_z8_scorer_weighted_residual_sidecar",
            "prove_mamba_dreamer_wyner_ziv_sections_drive_receiver_pixels",
        ]
        status_note = "digest scaffold must be replaced before score promotion"
    decoder_grid: dict[str, Any] = {}
    if packet is not None:
        decoder_grid = {
            "decoder_grid_height": int(packet.config.height),
            "decoder_grid_width": int(packet.config.width),
            "contest_output_height": CAMERA_H,
            "contest_output_width": CAMERA_W,
            "resolution_authority_note": (
                "HPRC compact receiver decodes on its charged learned grid, "
                "then deterministic inflate emits contest raw camera resolution; "
                "exact SegNet/PoseNet authority still requires contest CPU/CUDA replay."
            ),
        }
    return {
        "section_pixel_consumption_proofs": section_pixel_proofs,
        "pixel_consumed_archive_sections": pixel_consumed,
        "full_stack_pixel_consumption_claim": compact_receiver and bool(pixel_consumed),
        "predictive_stack_pixel_consumption": {
            "schema": "hprc_predictive_stack_pixel_consumption_backlog.v1",
            "receiver_mode": section_proof.get("receiver_mode"),
            "status_note": status_note,
            "next_required_tasks": next_tasks,
            **decoder_grid,
        },
    }


def _write_hprc_runtime(
    *,
    submission_dir: Path,
    repo_root: Path,
    bin_bytes: bytes,
) -> None:
    write_contest_runtime(
        submission_dir,
        substrate_pkg_name="hprc",
        repo_root=repo_root,
        runtime_module_files=HPRC_RUNTIME_MODULE_FILES,
        inflate_import_line="from tac.substrates.hprc.inflate import inflate_one_video",
    )
    (submission_dir / "0.bin").write_bytes(bin_bytes)


def export_hprc_archive_bytes(
    archive_bytes: bytes,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    emit_archive_bound_candidate_package: bool = True,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
) -> tuple[Path, str, int]:
    """Export already-built HPRC bytes as a contest-shaped candidate."""

    root, out_dir = _resolve_output_dir(output_dir, repo_root=repo_root)
    preflight_free_bytes = _assert_receiver_proof_scratch_space(out_dir)
    bin_bytes = bytes(archive_bytes)
    packet = parse_hprc_packet(bin_bytes)
    compact_receiver_packet = is_compact_receiver_packet(packet)
    (out_dir / "0.bin").write_bytes(bin_bytes)
    (out_dir / "hprc_packet_manifest.json").write_text(
        json.dumps(packet.manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    section_proof = build_hprc_section_mutation_proof(bin_bytes)
    section_proof_path = out_dir / "hprc_section_semantic_mutation_proof.json"
    section_proof_path.write_text(
        json.dumps(section_proof, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    submission_dir = out_dir / "submission"
    _write_hprc_runtime(submission_dir=submission_dir, repo_root=root, bin_bytes=bin_bytes)
    archive_zip_path = out_dir / "archive.zip"
    build_archive_zip(archive_zip_path, bin_bytes=bin_bytes, submission_dir=submission_dir)
    archive_sha256 = sha256_file(archive_zip_path)
    archive_bytes_count = archive_zip_path.stat().st_size
    byte_ledger = {
        "schema": "hprc_archive_byte_ledger.v1",
        "archive_zip_bytes": int(archive_bytes_count),
        "archive_zip_sha256": archive_sha256,
        "hprc_0bin_bytes": len(bin_bytes),
        "contest_rate_term": contest_rate_term(int(archive_bytes_count)),
        "sub019_zero_distortion_byte_ceiling": HPRC_SUB019_ZERO_DISTORTION_BYTE_CEILING,
        "below_sub019_zero_distortion_byte_ceiling": (
            int(archive_bytes_count) <= HPRC_SUB019_ZERO_DISTORTION_BYTE_CEILING
        ),
        "runtime_and_zip_counted": True,
        "receiver_proof_raw_output_retained_by_default": False,
        "receiver_proof_scratch_required_bytes": HPRC_RECEIVER_PROOF_SCRATCH_BYTES,
        "receiver_proof_scratch_free_bytes_at_preflight": preflight_free_bytes,
        "score_claim": False,
        "promotion_eligible": False,
    }
    byte_ledger_path = out_dir / "hprc_archive_byte_ledger.json"
    byte_ledger_path.write_text(
        json.dumps(byte_ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if emit_archive_bound_candidate_package:
        input_artifacts = [
            _repo_relative(archive_zip_path, root),
            _repo_relative(out_dir / "0.bin", root),
            _repo_relative(out_dir / "hprc_packet_manifest.json", root),
            _repo_relative(section_proof_path, root),
            _repo_relative(byte_ledger_path, root),
        ]
        emit_archive_bound_candidate_runtime_package(
            adapter_id=HPRC_ARCHIVE_BOUND_ADAPTER_ID,
            candidate_family=HPRC_ARCHIVE_CANDIDATE_FAMILY,
            candidate_id_prefix="hprc",
            transform_kind=HPRC_ARCHIVE_TRANSFORM_KIND,
            archive_zip_path=archive_zip_path,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes_count,
            submission_dir=submission_dir,
            output_dir=out_dir,
            repo_root=root,
            receiver_contract_kind="hprc_generated_inflate_sh_decode_only_receiver",
            proof_schema=HPRC_RECEIVER_PROOF_SCHEMA,
            proof_filename="hprc_receiver_proof.json",
            candidate_label="hprc",
            expected_receiver_output_name="0.raw",
            expected_receiver_output_bytes=_expected_receiver_output_bytes_for_packet(packet),
            retain_receiver_output=retain_receiver_proof_output,
            runtime_adapter_manifest_extra={
                "schema": "hprc_runtime_adapter_manifest.v1",
                "packet_manifest_path": _repo_relative(
                    out_dir / "hprc_packet_manifest.json", root
                ),
                "section_mutation_proof_path": _repo_relative(section_proof_path, root),
                "archive_byte_ledger_path": _repo_relative(byte_ledger_path, root),
                "section_mutation_preview_ready": bool(
                    section_proof["section_mutation_preview_ready"]
                ),
                **_runtime_payload_consumption_manifest(
                    section_proof,
                    packet=packet,
                ),
                "byte_ledger": byte_ledger,
                "trained_renderer_export_ready": compact_receiver_packet,
                "z8_residual_sidecar_ready": False,
                "resolution_contract": {
                    **hprc_resolution_contract(),
                    "decoder_grid": {
                        "height": int(packet.config.height),
                        "width": int(packet.config.width),
                        "exact_axis_required_for_resolution_authority": True,
                    },
                },
            },
            candidate_row_schema="hprc_archive_bound_candidate_row.v1",
            wrapper_schema=HPRC_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
            input_artifacts=input_artifacts,
            extra_blockers=[
                *(
                    [
                        "hprc_compact_receiver_missing_z8_scorer_weighted_residual_sidecar",
                        "hprc_compact_receiver_exact_distortion_replay_not_executed",
                    ]
                    if compact_receiver_packet
                    else ["hprc_v0_receiver_scaffold_not_trained_renderer"]
                ),
                "contest_cpu_cuda_exact_eval_not_executed",
            ],
            mlx_triage_argv=mlx_triage_argv,
        )
    return archive_zip_path, archive_sha256, archive_bytes_count


__all__ = [
    "HPRC_ARCHIVE_BOUND_ADAPTER_ID",
    "HPRC_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA",
    "HPRC_ARCHIVE_CANDIDATE_FAMILY",
    "HPRC_ARCHIVE_TRANSFORM_KIND",
    "HPRC_RECEIVER_PROOF_SCHEMA",
    "HPRC_RECEIVER_PROOF_SCRATCH_BYTES",
    "HPRC_RUNTIME_MODULE_FILES",
    "HPRC_SUB019_ZERO_DISTORTION_BYTE_CEILING",
    "build_hprc_section_mutation_proof",
    "build_minimal_hprc_v0_packet",
    "export_hprc_archive_bytes",
]
