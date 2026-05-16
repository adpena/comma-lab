# SPDX-License-Identifier: MIT
"""NSCS01 byte-closed packet build and consumption proof helpers.

This module is intentionally scorer-free.  It builds an untrained but
byte-closed NSCS01 split-renderer packet skeleton and proves that the
score-affecting payload sections are consumed by the same inflate-time
reconstruction path used by the runtime.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from tac.substrates.nscs01_nullspace_split_renderer.architecture import (
    NullspaceSplitConfig,
    NullspaceSplitRenderer,
)
from tac.substrates.nscs01_nullspace_split_renderer.archive import (
    NSP1_HEADER_FMT,
    NSP1_HEADER_SIZE,
    NullspaceSplitArchive,
    pack_archive,
    parse_archive,
)
from tac.substrates.nscs01_nullspace_split_renderer.inflate import (
    _build_renderer_from_archive_bytes,
)

LANE_ID = "lane_nscs01_nullspace_split_renderer_20260515"
PACKET_SCHEMA = "nscs01_split_renderer_packet_build_v1"
CONSUMPTION_PROOF_SCHEMA = "nscs01_runtime_consumption_proof_v1"
SCORE_AFFECTING_SECTIONS = ("HEAD0_BLOB", "HEAD1_BLOB", "LATENT_BLOB")


def sha256_bytes(payload: bytes) -> str:
    """Return the hex SHA-256 for bytes."""

    return hashlib.sha256(payload).hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to("cpu", dtype=torch.float32).contiguous()
    return sha256_bytes(cpu.numpy().tobytes(order="C"))


def build_skeleton_archive_bytes(
    *,
    cfg: NullspaceSplitConfig,
    seed: int,
    extra_meta: dict[str, Any] | None = None,
) -> bytes:
    """Build deterministic NSP1 payload bytes from an initialized renderer.

    This is a packet skeleton, not a trained substrate.  The metadata makes
    that fail-closed status explicit so downstream tools cannot promote it as
    score evidence.
    """

    torch.manual_seed(seed)
    renderer = NullspaceSplitRenderer(cfg)
    meta = {
        "lane_id": LANE_ID,
        "packet_schema": PACKET_SCHEMA,
        "training_status": "untrained_skeleton",
        "frame0_role": "pose_heavy_segnet_nullspace",
        "frame1_role": "seg_heavy_pose_aware",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "runtime_consumption_proof_required": True,
    }
    if extra_meta:
        meta.update(extra_meta)
    return pack_archive(
        head0_state_dict=renderer.frame_0_head.state_dict(),
        head1_state_dict=renderer.frame_1_head.state_dict(),
        latents=renderer.latents,
        head0_bits=cfg.head0_bits,
        head1_bits=cfg.head1_bits,
        latent_bits=cfg.latent_bits,
        head0_base_channels=cfg.head0_base_channels,
        head1_base_channels=cfg.head1_base_channels,
        extra_meta=meta,
    )


def section_offsets_from_archive_bytes(archive_bytes: bytes) -> dict[str, dict[str, Any]]:
    """Return compressed section offsets/lengths from the fixed NSP1 header."""

    if len(archive_bytes) < NSP1_HEADER_SIZE:
        raise ValueError("archive too small for NSP1 header")
    (
        _magic,
        _version,
        _num_pairs,
        _latent_dim,
        _head0_bits,
        _head1_bits,
        _latent_bits,
        _head0_base_channels,
        _head1_base_channels,
        head0_len,
        head1_len,
        latent_len,
        meta_len,
    ) = struct.unpack_from(NSP1_HEADER_FMT, archive_bytes, 0)
    head0_off = NSP1_HEADER_SIZE
    head1_off = head0_off + head0_len
    latent_off = head1_off + head1_len
    meta_off = latent_off + latent_len
    sections = {
        "HEAD0_BLOB": (head0_off, head0_len, "frame0_pose_heavy_head_weights"),
        "HEAD1_BLOB": (head1_off, head1_len, "frame1_seg_heavy_head_weights"),
        "LATENT_BLOB": (latent_off, latent_len, "shared_per_pair_latents"),
        "META_BLOB": (meta_off, meta_len, "runtime_manifest_and_provenance"),
    }
    out: dict[str, dict[str, Any]] = {}
    for name, (offset, length, role) in sections.items():
        if offset < NSP1_HEADER_SIZE or length < 1 or offset + length > len(archive_bytes):
            raise ValueError(f"invalid section bounds for {name}: offset={offset} length={length}")
        payload = archive_bytes[offset : offset + length]
        out[name] = {
            "offset": offset,
            "length": length,
            "sha256": sha256_bytes(payload),
            "role": role,
            "score_affecting": name in SCORE_AFFECTING_SECTIONS,
        }
    if meta_off + meta_len != len(archive_bytes):
        raise ValueError(
            f"section lengths do not consume archive exactly: {meta_off + meta_len} != {len(archive_bytes)}"
        )
    return out


def render_digest_from_archive_bytes(
    archive_bytes: bytes,
    *,
    pair_indices: tuple[int, ...] = (0,),
    device: str = "cpu",
) -> dict[str, Any]:
    """Render selected pairs through the inflate reconstruction path and hash them."""

    renderer, _latents = _build_renderer_from_archive_bytes(archive_bytes, device)
    idx = torch.tensor(pair_indices, dtype=torch.long, device=device)
    with torch.no_grad():
        frame0, frame1 = renderer.reconstruct_pair(idx)
    return {
        "pair_indices": list(pair_indices),
        "frame0_sha256": _tensor_digest(frame0),
        "frame1_sha256": _tensor_digest(frame1),
        "combined_sha256": sha256_bytes(
            (frame0.detach().to("cpu", dtype=torch.float32).contiguous().numpy().tobytes(order="C"))
            + (frame1.detach().to("cpu", dtype=torch.float32).contiguous().numpy().tobytes(order="C"))
        ),
        "frame0_mean": float(frame0.detach().mean().cpu().item()),
        "frame1_mean": float(frame1.detach().mean().cpu().item()),
    }


def _flip_archive_byte(
    archive_bytes: bytes,
    *,
    section_offset: int,
    section_length: int,
    candidate_positions: tuple[int, ...],
) -> tuple[bytes, int]:
    for pos in candidate_positions:
        if 0 <= pos < section_length:
            target = section_offset + pos
            mutated = bytearray(archive_bytes)
            mutated[target] ^= 0x55
            return bytes(mutated), pos
    raise ValueError("no candidate byte position inside section")


def prove_runtime_consumes_score_affecting_sections(
    archive_bytes: bytes,
    *,
    pair_indices: tuple[int, ...] = (0,),
    device: str = "cpu",
) -> dict[str, Any]:
    """Prove HEAD0/HEAD1/LATENT bytes are consumed by inflate reconstruction.

    A section is consumed if flipping one byte either changes the rendered
    output digest or causes the runtime parser/decompressor/load path to reject
    the mutated packet.  Rejection is valid consumption evidence, not score
    evidence.
    """

    parsed: NullspaceSplitArchive = parse_archive(archive_bytes)
    sections = section_offsets_from_archive_bytes(archive_bytes)
    baseline = render_digest_from_archive_bytes(
        archive_bytes, pair_indices=pair_indices, device=device
    )
    proof_sections: list[dict[str, Any]] = []
    for section_name in SCORE_AFFECTING_SECTIONS:
        section = sections[section_name]
        length = int(section["length"])
        candidates = (8, 16, 32, length // 2, max(0, length - 8))
        consumed = False
        mutation_result: dict[str, Any] = {}
        for _attempt, _pos in enumerate(candidates):
            try:
                mutated, flipped_pos = _flip_archive_byte(
                    archive_bytes,
                    section_offset=int(section["offset"]),
                    section_length=length,
                    candidate_positions=(_pos,),
                )
            except ValueError:
                continue
            try:
                mutated_digest = render_digest_from_archive_bytes(
                    mutated, pair_indices=pair_indices, device=device
                )
            except Exception as exc:
                consumed = True
                mutation_result = {
                    "mechanism": "runtime_decode_rejected_mutation",
                    "flip_position": flipped_pos,
                    "exception_type": type(exc).__name__,
                    "exception_text": str(exc)[:240],
                }
                break
            changed = mutated_digest["combined_sha256"] != baseline["combined_sha256"]
            if changed:
                consumed = True
                mutation_result = {
                    "mechanism": "render_digest_changed",
                    "flip_position": flipped_pos,
                    "mutated_combined_sha256": mutated_digest["combined_sha256"],
                }
                break
        proof_sections.append(
            {
                "section": section_name,
                "consumed_by_runtime": consumed,
                "offset": section["offset"],
                "length": section["length"],
                "sha256": section["sha256"],
                "role": section["role"],
                "mutation_result": mutation_result,
            }
        )

    return {
        "schema": CONSUMPTION_PROOF_SCHEMA,
        "lane_id": LANE_ID,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "runtime_hook": (
            "tac.substrates.nscs01_nullspace_split_renderer.inflate."
            "_build_renderer_from_archive_bytes"
        ),
        "parser_hook": "parse_archive + deserialize_head_state_dicts + deserialize_latents",
        "archive_sha256": sha256_bytes(archive_bytes),
        "archive_bytes": len(archive_bytes),
        "num_pairs": parsed.num_pairs,
        "latent_dim": parsed.latent_dim,
        "baseline_render_digest": baseline,
        "sections": proof_sections,
        "all_score_affecting_sections_consumed": all(
            bool(item["consumed_by_runtime"]) for item in proof_sections
        ),
        "non_score_sections": [
            sections["META_BLOB"],
        ],
    }


def write_deterministic_archive_zip(archive_zip_path: Path, *, bin_bytes: bytes) -> None:
    """Write deterministic single-member archive.zip containing ``0.bin``."""

    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("0.bin", date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(archive_zip_path, "w") as zf:
        zf.writestr(info, bin_bytes)


def build_manifest(
    *,
    cfg: NullspaceSplitConfig,
    seed: int,
    archive_bytes: bytes,
    archive_zip_path: Path,
    runtime_dir: Path | None,
    consumption_proof: dict[str, Any],
) -> dict[str, Any]:
    """Build the packet manifest.  This never marks the packet score-ready."""

    archive_zip_bytes = archive_zip_path.read_bytes()
    return {
        "schema": PACKET_SCHEMA,
        "lane_id": LANE_ID,
        "packet_kind": "frame0_pose_heavy_frame1_seg_heavy_split_renderer",
        "byte_closed_runtime_packet": runtime_dir is not None,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "[byte-closed-runtime-packet-no-score]",
        "archive_member": "0.bin",
        "archive_bytes": len(archive_bytes),
        "archive_sha256": sha256_bytes(archive_bytes),
        "archive_zip_path": str(archive_zip_path),
        "archive_zip_bytes": len(archive_zip_bytes),
        "archive_zip_sha256": sha256_bytes(archive_zip_bytes),
        "runtime_dir": None if runtime_dir is None else str(runtime_dir),
        "config": asdict(cfg),
        "seed": seed,
        "section_manifest": section_offsets_from_archive_bytes(archive_bytes),
        "runtime_consumption_proof": consumption_proof,
        "dispatch_blockers": [
            "untrained_skeleton_no_score_claim",
            "no_contest_cuda_anchor",
            "no_contest_cpu_anchor",
            "head0_arch_disambiguator_not_measured",
            "paired_axis_exact_eval_required_before_ranking",
        ],
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write sorted JSON manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "CONSUMPTION_PROOF_SCHEMA",
    "LANE_ID",
    "PACKET_SCHEMA",
    "SCORE_AFFECTING_SECTIONS",
    "build_manifest",
    "build_skeleton_archive_bytes",
    "prove_runtime_consumes_score_affecting_sections",
    "render_digest_from_archive_bytes",
    "section_offsets_from_archive_bytes",
    "sha256_bytes",
    "write_deterministic_archive_zip",
    "write_manifest",
]
