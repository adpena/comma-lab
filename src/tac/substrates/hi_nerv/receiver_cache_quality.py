# SPDX-License-Identifier: MIT
"""HiNeRV archive-backed receiver-cache quality probes.

This module renders scorer-input cache tensors directly from HIV1 archive
bytes through the same PyTorch receiver used by ``inflate.py``.  It is a local
MLX/CPU research-signal helper only: the cache and quality gate never create a
contest score claim, but they do prevent flat-renderer artifacts from entering
section-value, waterfill, or exact-dispatch queues unnoticed.
"""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from tac.analysis.mlx_cache_quality_gate import write_mlx_cache_quality_gate
from tac.local_acceleration.mlx_preprocess import (
    write_scorer_input_cache_from_pair_batches,
)
from tac.repo_io import sha256_file, write_json
from tac.substrates._shared.inflate_runtime import CAMERA_HW, rgb_pair_to_uint8_frames
from tac.substrates.hi_nerv.inflate import build_model_from_archive
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

HI_NERV_RECEIVER_CACHE_QUALITY_REPORT_SCHEMA = (
    "hi_nerv_receiver_cache_quality_report.v1"
)
HI_NERV_DIRECT_RECEIVER_CACHE_AUDIT_SCHEMA = (
    "hi_nerv_direct_receiver_render_cache_identity_audit.v1"
)
HI_NERV_DIRECT_RECEIVER_CACHE_REPORT_SCHEMA = (
    "hi_nerv_direct_receiver_cache_report.v1"
)


def write_hi_nerv_receiver_cache_quality_report(
    *,
    archive_zip_path: str | Path,
    output_dir: str | Path,
    reference_cache_dir: str | Path | None = None,
    max_pairs: int = 1,
    batch_pairs: int = 1,
    sample_pairs: int | None = None,
    min_segnet_std: float = 1.0,
    min_segnet_dynamic_range: float = 16.0,
    max_segnet_mae_vs_reference_for_fit_gate: float = 64.0,
) -> dict[str, Any]:
    """Render a small HiNeRV receiver cache and optionally run a quality gate.

    ``archive_zip_path`` must contain a root ``0.bin`` HIV1 payload.  The
    receiver render path is the exact ``build_model_from_archive`` +
    ``rgb_pair_to_uint8_frames`` lowering used by the packaged inflate runtime,
    but the result is still false-authority local evidence.
    """

    archive_path = Path(archive_zip_path).expanduser().resolve(strict=False)
    out = Path(output_dir).expanduser().resolve(strict=False)
    if max_pairs < 1:
        raise ValueError(f"max_pairs must be >= 1, got {max_pairs}")
    if batch_pairs < 1:
        raise ValueError(f"batch_pairs must be >= 1, got {batch_pairs}")
    if not archive_path.is_file():
        raise FileNotFoundError(f"HiNeRV archive.zip missing: {archive_path}")
    out.mkdir(parents=True, exist_ok=True)

    member_name, payload = _read_hiv1_payload_from_archive_zip(archive_path)
    archive_sha256 = sha256_file(archive_path)
    cache_dir = out / "candidate_cache"
    direct_report, manifest = write_hi_nerv_direct_receiver_cache_from_payload(
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        member_name=member_name,
        archive_payload=payload,
        output_cache_dir=cache_dir,
        max_pairs=int(max_pairs),
        batch_pairs=int(batch_pairs),
    )

    quality_gate: dict[str, Any] | None = None
    quality_gate_path: Path | None = None
    if reference_cache_dir is not None:
        quality_gate_path = out / "cache_quality_gate.json"
        quality_gate = write_mlx_cache_quality_gate(
            output_json=quality_gate_path,
            candidate_cache_dir=cache_dir,
            reference_cache_dir=reference_cache_dir,
            sample_pairs=int(sample_pairs or max_pairs),
            min_segnet_std=float(min_segnet_std),
            min_segnet_dynamic_range=float(min_segnet_dynamic_range),
            max_segnet_mae_vs_reference_for_fit_gate=float(
                max_segnet_mae_vs_reference_for_fit_gate
            ),
        )

    blockers = ["hi_nerv_receiver_cache_quality_is_false_authority"]
    if quality_gate is None:
        blockers.append("hi_nerv_receiver_cache_quality_reference_gate_not_run")
    else:
        blockers.extend(str(v) for v in quality_gate.get("blockers") or [])

    report = {
        "schema": HI_NERV_RECEIVER_CACHE_QUALITY_REPORT_SCHEMA,
        "archive_path": archive_path.as_posix(),
        "archive_sha256": archive_sha256,
        "archive_bytes": int(archive_path.stat().st_size),
        "zip_member": member_name,
        "output_dir": out.as_posix(),
        "candidate_cache_dir": cache_dir.as_posix(),
        "candidate_cache_manifest_path": (cache_dir / "manifest.json").as_posix(),
        "candidate_cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
        "direct_receiver_cache_report": direct_report,
        "cache_manifest_summary": {
            "pair_count": int(manifest["pair_count"]),
            "raw_sha256": manifest.get("raw_sha256"),
            "array_sha256": dict(manifest.get("array_sha256") or {}),
            "source_kind": manifest.get("source_kind"),
        },
        "quality_gate_path": (
            quality_gate_path.as_posix() if quality_gate_path is not None else None
        ),
        "quality_gate": quality_gate,
        "quality_gate_passed": (
            bool(quality_gate.get("fit_gate_passed")) if quality_gate else False
        ),
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }
    report_path = out / "hi_nerv_receiver_cache_quality_report.json"
    report["report_path"] = report_path.as_posix()
    write_json(report_path, report)
    return report


def write_hi_nerv_direct_receiver_cache_from_payload(
    *,
    archive_path: str | Path,
    archive_sha256: str,
    member_name: str,
    archive_payload: bytes,
    output_cache_dir: str | Path,
    max_pairs: int,
    batch_pairs: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write a direct receiver scorer-input cache from raw HIV1 payload bytes."""

    import torch

    if max_pairs < 1:
        raise ValueError(f"max_pairs must be >= 1, got {max_pairs}")
    if batch_pairs < 1:
        raise ValueError(f"batch_pairs must be >= 1, got {batch_pairs}")
    source_archive = Path(archive_path).expanduser().resolve(strict=False)
    cache_dir = Path(output_cache_dir).expanduser().resolve(strict=False)
    arc, cfg, model = build_model_from_archive(archive_payload, device="cpu")
    raw_pair_count = int(cfg.num_pairs)
    pair_count = min(raw_pair_count, int(max_pairs))
    if pair_count < 1:
        raise ValueError("HiNeRV direct receiver cache has no complete pairs")
    selected_pair_indices = list(range(pair_count))
    h, w = CAMERA_HW
    scorer_pair_indices = np.array(
        [[2 * idx, 2 * idx + 1] for idx in selected_pair_indices],
        dtype=np.int64,
    )

    def pair_batches() -> Iterable[np.ndarray]:
        with torch.no_grad():
            for start in range(0, pair_count, int(batch_pairs)):
                chunk_indices = selected_pair_indices[start : start + int(batch_pairs)]
                rendered: list[np.ndarray] = []
                for pair_index in chunk_indices:
                    idx = torch.tensor([pair_index], device="cpu", dtype=torch.long)
                    rgb_0, rgb_1 = model(idx)
                    rendered.append(
                        rgb_pair_to_uint8_frames(
                            rgb_0,
                            rgb_1,
                            input_range="unit",
                        ).reshape(1, 2, h, w, 3)
                    )
                yield np.concatenate(rendered, axis=0)

    manifest = write_scorer_input_cache_from_pair_batches(
        pair_batches(),
        cache_dir,
        pair_count=pair_count,
        pair_indices=scorer_pair_indices,
        frame_shape_hwc=(h, w, 3),
        source=source_archive.as_posix(),
        source_kind="hi_nerv_direct_receiver_render",
        archive_sha256=str(archive_sha256),
        inflated_outputs_aggregate_sha256=None,
        batch_pairs=int(batch_pairs),
        compute_raw_sha256=True,
    )
    manifest["inflated_outputs_aggregate_sha256"] = manifest.get("raw_sha256")

    audit_path = cache_dir / "hi_nerv_direct_receiver_render_cache_identity_audit.json"
    audit = {
        "schema_version": HI_NERV_DIRECT_RECEIVER_CACHE_AUDIT_SCHEMA,
        "verdict": "PASS_HI_NERV_DIRECT_RECEIVER_RENDER_CACHE_IDENTITY",
        "passed": True,
        "created_by": "tac.substrates.hi_nerv.receiver_cache_quality",
        "allowed_use": (
            "certify_hi_nerv_direct_mlx_cache_rebuildability_and_render_quality"
        ),
        "forbidden_use": "score_claim_or_promotion_or_rank_or_exact_dispatch",
        "cache": {
            "archive_sha256": manifest.get("archive_sha256"),
            "inflated_outputs_aggregate_sha256": manifest.get(
                "inflated_outputs_aggregate_sha256"
            ),
            "raw_sha256": manifest.get("raw_sha256"),
            "pair_count": manifest.get("pair_count"),
            "hash_domain": manifest.get("hash_domain"),
            "array_sha256": manifest.get("array_sha256"),
        },
        "source": {
            "archive_path": source_archive.as_posix(),
            "archive_sha256": str(archive_sha256),
            "zip_member": str(member_name),
            "archive_magic": "HIV1",
            "schema_version": int(arc.schema_version),
            "config": {
                "num_pairs": int(cfg.num_pairs),
                "latent_dim_coarse": int(cfg.latent_dim_coarse),
                "latent_dim_mid": int(cfg.latent_dim_mid),
                "latent_dim_fine": int(cfg.latent_dim_fine),
                "embed_dim": int(cfg.embed_dim),
                "initial_grid_h": int(cfg.initial_grid_h),
                "initial_grid_w": int(cfg.initial_grid_w),
                "decoder_channels": [int(c) for c in cfg.decoder_channels],
                "sin_frequency": float(cfg.sin_frequency),
                "num_upsample_blocks": int(cfg.num_upsample_blocks),
                "mid_injection_block_index": int(cfg.mid_injection_block_index),
                "fine_injection_block_index": int(cfg.fine_injection_block_index),
                "output_height": int(cfg.output_height),
                "output_width": int(cfg.output_width),
            },
        },
        "direct_render": {
            "raw_pair_count": raw_pair_count,
            "selected_pair_count": int(pair_count),
            "selected_pair_ranges": [[0, int(pair_count) - 1]],
            "pair_index_scope": "prefix_from_zero",
            "frame_shape_hwc": [h, w, 3],
            "batch_pairs": int(batch_pairs),
            "max_pairs": int(max_pairs),
            "raw_file_written": False,
            "rebuilds_from_archive_bytes": True,
            "lowering": "rgb_pair_to_uint8_frames_input_range_unit_bicubic",
        },
        "receiver_proof_required_for_promotion": True,
        **FALSE_AUTHORITY,
    }
    write_json(audit_path, audit)
    manifest["hi_nerv_direct_receiver_render_cache_identity_audit"] = {
        "schema_version": audit["schema_version"],
        "path": audit_path.as_posix(),
        "sha256": sha256_file(audit_path),
        "verdict": audit["verdict"],
        "passed": True,
        "archive_path": source_archive.as_posix(),
        "archive_sha256": str(archive_sha256),
        **FALSE_AUTHORITY,
    }
    manifest["eligible_for_hi_nerv_direct_rebuild_cleanup"] = True
    write_json(cache_dir / "manifest.json", manifest)

    report = {
        "schema": HI_NERV_DIRECT_RECEIVER_CACHE_REPORT_SCHEMA,
        "source_family": "hi_nerv",
        "archive_path": source_archive.as_posix(),
        "archive_sha256": str(archive_sha256),
        "zip_member": str(member_name),
        "archive_magic": "HIV1",
        "schema_version": int(arc.schema_version),
        "raw_pair_count": raw_pair_count,
        "cached_pair_count": int(manifest["pair_count"]),
        "selected_pair_count": int(pair_count),
        "selected_pair_ranges": [[0, int(pair_count) - 1]],
        "pair_index_scope": "prefix_from_zero",
        "frame_shape_hwc": [h, w, 3],
        "direct_render_raw_bytes": int(manifest["pair_count"]) * 2 * h * w * 3,
        "direct_render_raw_pair_count": int(manifest["pair_count"]),
        "direct_render_raw_sha256": manifest.get("raw_sha256"),
        "direct_render_raw_sha256_scope": manifest.get("raw_sha256_scope"),
        "raw_file_written": False,
        "receiver_proof_required_for_promotion": True,
        "identity_audit_path": audit_path.as_posix(),
        "identity_audit_sha256": manifest[
            "hi_nerv_direct_receiver_render_cache_identity_audit"
        ]["sha256"],
        "candidate_cache_identity_mode": (
            "hi_nerv_direct_receiver_render_cache_identity_audited_false_authority"
        ),
        **FALSE_AUTHORITY,
    }
    return report, manifest


def _read_hiv1_payload_from_archive_zip(archive_zip_path: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(archive_zip_path, "r") as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
        member_name = "0.bin" if "0.bin" in names else None
        if member_name is None:
            raise ValueError(
                "HiNeRV archive.zip missing root 0.bin member; found "
                f"{names[:10]}"
            )
        payload = zf.read(member_name)
    if not payload.startswith(b"HIV1"):
        digest = hashlib.sha256(payload).hexdigest()
        raise ValueError(
            f"archive member {member_name} is not HIV1 payload; sha256={digest}"
        )
    return member_name, payload


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


__all__ = [
    "HI_NERV_DIRECT_RECEIVER_CACHE_AUDIT_SCHEMA",
    "HI_NERV_DIRECT_RECEIVER_CACHE_REPORT_SCHEMA",
    "HI_NERV_RECEIVER_CACHE_QUALITY_REPORT_SCHEMA",
    "write_hi_nerv_direct_receiver_cache_from_payload",
    "write_hi_nerv_receiver_cache_quality_report",
]
