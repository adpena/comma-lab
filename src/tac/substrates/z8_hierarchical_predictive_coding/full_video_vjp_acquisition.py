# SPDX-License-Identifier: MIT
"""Full-video VJP acquisition lane for Z8 joint P18/P19 water-fill.

The contest optimizer may use mini-batch/window gradients as cheap ranking
probes, but budget-spending Z8 coefficient attacks must be ratified by a
full-video, archive-pinned joint P18/P19 surface. This module is the queue-owned
spine for that workflow: deterministic pair shards, target-mode policy, and a
surface bundle contract consumed by the coefficient materializer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.optimization.target_modes import (
    CONTEST_VIDEO_OVERFIT_MODE,
    CORPUS_GENERALIZATION_MODE,
    HYBRID_CONTEST_PLUS_CORPUS_MODE,
    normalize_target_optimization_mode,
    target_mode_declares_overfit_allowed,
    target_mode_requires_corpus_manifest,
)
from tac.repo_io import write_json
from tac.substrates.z8_hierarchical_predictive_coding.archive import parse_archive

Z8_FULL_VIDEO_VJP_ACQUISITION_PLAN_SCHEMA = "z8_full_video_vjp_acquisition_plan.v1"
Z8_FULL_VIDEO_VJP_SURFACE_BUNDLE_SCHEMA = "z8_full_video_vjp_surface_bundle.v1"
FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION = "full_video_exact_accumulation"
SINGLE_UPDATE_AFTER_FULL_REDUCTION = "single_update_after_all_pair_shards_reduce"


@dataclass(frozen=True)
class Z8FullVideoVjpAcquisitionConfig:
    """Config for deterministic full-video VJP shard planning."""

    target_mode: str = CONTEST_VIDEO_OVERFIT_MODE
    pair_chunk_size: int = 64
    parallel_workers: int | None = None
    corpus_manifest_path: str | None = None
    allow_minibatch_probe_between_full_passes: bool = True
    allow_partial_production_probe_surface: bool = False

    def __post_init__(self) -> None:
        normalize_target_optimization_mode(self.target_mode)
        if self.pair_chunk_size <= 0:
            raise ValueError("pair_chunk_size must be positive")
        if self.parallel_workers is not None and self.parallel_workers <= 0:
            raise ValueError("parallel_workers must be positive when provided")
        if target_mode_requires_corpus_manifest(self.target_mode) and not self.corpus_manifest_path:
            raise ValueError("corpus_manifest_path is required for production/hybrid target modes")

    @property
    def normalized_target_mode(self) -> str:
        return normalize_target_optimization_mode(self.target_mode)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pair_shards(num_pairs: int, chunk_size: int) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, int(num_pairs), int(chunk_size))):
        end = min(start + int(chunk_size), int(num_pairs))
        shards.append(
            {
                "schema": "z8_full_video_vjp_pair_shard.v1",
                "shard_index": int(shard_index),
                "pair_start": int(start),
                "pair_end": int(end),
                "pair_count": int(end - start),
                "execution_hint": "mlx_full_video_resident_pair_chunk_vjp",
            }
        )
    return shards


def build_z8_full_video_vjp_acquisition_plan(
    archive_bytes: bytes,
    *,
    config: Z8FullVideoVjpAcquisitionConfig | None = None,
) -> dict[str, Any]:
    """Return a deterministic work plan for full-video local VJP acquisition."""

    cfg = config or Z8FullVideoVjpAcquisitionConfig()
    arc = parse_archive(archive_bytes)
    archive_sha = _sha256_bytes(archive_bytes)
    target_mode = cfg.normalized_target_mode
    shards = _pair_shards(arc.num_pairs, cfg.pair_chunk_size)
    parallel_workers = cfg.parallel_workers or min(
        len(shards), max(1, (arc.num_pairs + cfg.pair_chunk_size - 1) // cfg.pair_chunk_size)
    )
    return {
        "schema": Z8_FULL_VIDEO_VJP_ACQUISITION_PLAN_SCHEMA,
        "local_axis": "[macOS-MLX research-signal]",
        "archive_sha256": archive_sha,
        "archive_num_pairs": int(arc.num_pairs),
        "target_mode": target_mode,
        "declared_overfit_allowed": target_mode_declares_overfit_allowed(target_mode),
        "corpus_manifest_required": target_mode_requires_corpus_manifest(target_mode),
        "corpus_manifest_path": cfg.corpus_manifest_path,
        "full_video_vjp_is_first_class_acquisition_lane": True,
        "full_video_residency_required": True,
        "surface_linearization_archive_sha_required": True,
        "surface_relinearization_required_after_accepted_mutation": True,
        "pair_chunk_updates_forbidden": True,
        "gradient_reduction_semantics": FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION,
        "optimizer_update_semantics": SINGLE_UPDATE_AFTER_FULL_REDUCTION,
        "minibatch_window_gradients_policy": (
            "ranking_probe_only_between_full_video_passes"
            if cfg.allow_minibatch_probe_between_full_passes
            else "disabled"
        ),
        "minibatch_window_gradients_budget_spend_authority": False,
        "contest_mode_budget_spend_requires_full_video_archive_pinned_surface": (
            target_mode == CONTEST_VIDEO_OVERFIT_MODE
        ),
        "production_mode_requires_declared_corpus_manifest": target_mode
        in {
            CORPUS_GENERALIZATION_MODE,
            HYBRID_CONTEST_PLUS_CORPUS_MODE,
        },
        "pair_chunk_size": int(cfg.pair_chunk_size),
        "parallel_workers": int(parallel_workers),
        "shard_count": len(shards),
        "pair_shards": shards,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _as_array(row: Mapping[str, Any], key: str, *, dtype: Any) -> np.ndarray:
    if key not in row:
        raise ValueError(f"surface shard missing {key}")
    arr = np.asarray(row[key], dtype=dtype)
    if arr.ndim != 5:
        raise ValueError(f"{key} shard must have shape (pairs, frames, H, W, C); got {arr.shape}")
    return arr


def assemble_z8_full_video_vjp_surface_bundle(
    archive_bytes: bytes,
    *,
    shard_surfaces: Sequence[Mapping[str, Any]],
    config: Z8FullVideoVjpAcquisitionConfig | None = None,
) -> dict[str, Any]:
    """Assemble pair-sharded VJP outputs into one materializer-ready surface."""

    cfg = config or Z8FullVideoVjpAcquisitionConfig()
    plan = build_z8_full_video_vjp_acquisition_plan(archive_bytes, config=cfg)
    archive_sha = str(plan["archive_sha256"])
    expected_start = 0
    joint_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    shard_reports: list[dict[str, Any]] = []
    for shard_index, raw in enumerate(sorted(shard_surfaces, key=lambda row: int(row.get("pair_start", -1)))):
        if (
            raw.get("optimizer_update_applied")
            or raw.get("budget_spend_authority")
            or raw.get("optimizer_update_authority")
            or raw.get("gradient_reduction_authority")
        ):
            raise ValueError(
                "full-video VJP shards cannot carry optimizer update authority; "
                "assemble the complete archive-pinned pair grid before updating"
            )
        pair_start = int(raw.get("pair_start", -1))
        pair_end = int(raw.get("pair_end", -1))
        if pair_start != expected_start or pair_end <= pair_start:
            raise ValueError(
                "full-video VJP shards must be contiguous and ordered; "
                f"expected_start={expected_start} got=({pair_start},{pair_end})"
            )
        pinned = str(raw.get("linearization_archive_sha") or "")
        if pinned != archive_sha:
            raise ValueError(
                "full-video VJP shard linearization archive mismatch: "
                f"expected={archive_sha} got={pinned or '<missing>'}"
            )
        joint = _as_array(raw, "joint_weight", dtype=np.float64)
        mask = _as_array(raw, "rate_attack_deadzone_mask", dtype=bool)
        if joint.shape[0] != pair_end - pair_start or mask.shape != joint.shape:
            raise ValueError("surface shard pair span and tensor shape disagree")
        joint_chunks.append(joint)
        mask_chunks.append(mask)
        expected_start = pair_end
        shard_reports.append(
            {
                "schema": "z8_full_video_vjp_surface_shard_report.v1",
                "shard_index": int(raw.get("shard_index", shard_index)),
                "pair_start": pair_start,
                "pair_end": pair_end,
                "pair_count": int(pair_end - pair_start),
                "linearization_archive_sha": pinned,
                "optimizer_update_applied": False,
            }
        )

    archive_num_pairs = int(plan["archive_num_pairs"])
    full_coverage = expected_start == archive_num_pairs
    if not full_coverage and not cfg.allow_partial_production_probe_surface:
        raise ValueError(
            "full-video VJP surface does not cover archive pair grid: "
            f"covered={expected_start} required={archive_num_pairs}"
        )
    joint_full = np.concatenate(joint_chunks, axis=0) if joint_chunks else np.zeros((0, 2, 1, 1, 1))
    mask_full = np.concatenate(mask_chunks, axis=0) if mask_chunks else np.zeros_like(joint_full, dtype=bool)
    return {
        "schema": Z8_FULL_VIDEO_VJP_SURFACE_BUNDLE_SCHEMA,
        "target_mode": plan["target_mode"],
        "local_axis": plan["local_axis"],
        "archive_sha256": archive_sha,
        "linearization_archive_sha": archive_sha,
        "evidence_scope": "full_video" if full_coverage else "proposal_only",
        "full_video_pair_count": archive_num_pairs,
        "covered_pair_count": int(expected_start),
        "pair_coverage_fraction": float(expected_start / archive_num_pairs) if archive_num_pairs else 1.0,
        "full_video_surface_coverage": bool(full_coverage),
        "full_video_vjp_is_first_class_acquisition_lane": True,
        "full_video_reduction_complete": bool(full_coverage),
        "gradient_reduction_semantics": (
            FULL_VIDEO_EXACT_ACCUMULATION_REDUCTION if full_coverage else "proposal_or_sampled"
        ),
        "gradient_reduction_authority": bool(full_coverage),
        "minibatch_window_gradients_budget_spend_authority": False,
        "budget_spend_authority": bool(full_coverage),
        "optimizer_update_authority": bool(full_coverage),
        "optimizer_update_semantics": (
            SINGLE_UPDATE_AFTER_FULL_REDUCTION if full_coverage else "no_update_partial_surface_probe_only"
        ),
        "surface_relinearization_required_after_accepted_mutation": True,
        "joint_weight": joint_full,
        "rate_attack_deadzone_mask": mask_full,
        "shard_count": len(shard_reports),
        "shard_reports": shard_reports,
        "acquisition_plan": plan,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def write_z8_full_video_vjp_surface_bundle(
    bundle: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write a materializer-ready NPZ surface plus a compact manifest."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    surface_path = out_dir / "z8_full_video_vjp_surface_bundle.npz"
    np.savez_compressed(
        surface_path,
        joint_weight=np.asarray(bundle["joint_weight"], dtype=np.float32),
        rate_attack_deadzone_mask=np.asarray(bundle["rate_attack_deadzone_mask"], dtype=bool),
        linearization_archive_sha=np.asarray(str(bundle["linearization_archive_sha"])),
        evidence_scope=np.asarray(str(bundle["evidence_scope"])),
        target_mode=np.asarray(str(bundle["target_mode"])),
        gradient_reduction_semantics=np.asarray(str(bundle["gradient_reduction_semantics"])),
        gradient_reduction_authority=np.asarray(bool(bundle["gradient_reduction_authority"])),
        optimizer_update_authority=np.asarray(bool(bundle["optimizer_update_authority"])),
        optimizer_update_semantics=np.asarray(str(bundle["optimizer_update_semantics"])),
        full_video_reduction_complete=np.asarray(bool(bundle["full_video_reduction_complete"])),
        budget_spend_authority=np.asarray(bool(bundle["budget_spend_authority"])),
    )
    manifest = {
        "schema": "z8_full_video_vjp_surface_bundle_manifest.v1",
        "surface_bundle_schema": bundle["schema"],
        "surface_path": surface_path.as_posix(),
        "surface_sha256": _sha256_bytes(surface_path.read_bytes()),
        "archive_sha256": bundle["archive_sha256"],
        "linearization_archive_sha": bundle["linearization_archive_sha"],
        "target_mode": bundle["target_mode"],
        "evidence_scope": bundle["evidence_scope"],
        "full_video_surface_coverage": bundle["full_video_surface_coverage"],
        "covered_pair_count": bundle["covered_pair_count"],
        "full_video_pair_count": bundle["full_video_pair_count"],
        "gradient_reduction_semantics": bundle["gradient_reduction_semantics"],
        "gradient_reduction_authority": bundle["gradient_reduction_authority"],
        "budget_spend_authority": bundle["budget_spend_authority"],
        "optimizer_update_authority": bundle["optimizer_update_authority"],
        "optimizer_update_semantics": bundle["optimizer_update_semantics"],
        "shard_count": bundle["shard_count"],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    manifest_path = out_dir / "z8_full_video_vjp_surface_bundle_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = manifest_path.as_posix()
    write_json(manifest_path, manifest)
    return manifest


def write_z8_full_video_vjp_acquisition_plan(
    archive_bytes: bytes,
    output_dir: str | Path,
    *,
    config: Z8FullVideoVjpAcquisitionConfig | None = None,
) -> dict[str, Any]:
    """Write the deterministic full-video VJP shard plan for queue execution."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = build_z8_full_video_vjp_acquisition_plan(archive_bytes, config=config)
    plan_path = out_dir / "z8_full_video_vjp_acquisition_plan.json"
    write_json(plan_path, plan)
    plan["plan_path"] = plan_path.as_posix()
    write_json(plan_path, plan)
    return plan


def load_z8_full_video_vjp_surface_shard_file(path: str | Path) -> dict[str, Any]:
    """Load one archive-pinned full-video VJP shard from NPZ or JSON."""

    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p)
        required = {
            "joint_weight",
            "rate_attack_deadzone_mask",
            "pair_start",
            "pair_end",
            "linearization_archive_sha",
        }
        missing = sorted(key for key in required if key not in data)
        if missing:
            raise ValueError(f"{p} missing required shard keys: {missing}")
        return {
            "shard_index": int(np.asarray(data.get("shard_index", 0)).reshape(-1)[0]),
            "pair_start": int(np.asarray(data["pair_start"]).reshape(-1)[0]),
            "pair_end": int(np.asarray(data["pair_end"]).reshape(-1)[0]),
            "linearization_archive_sha": str(np.asarray(data["linearization_archive_sha"]).reshape(-1)[0]),
            "joint_weight": np.asarray(data["joint_weight"], dtype=np.float64),
            "rate_attack_deadzone_mask": np.asarray(
                data["rate_attack_deadzone_mask"],
                dtype=bool,
            ),
            "optimizer_update_applied": bool(np.asarray(data.get("optimizer_update_applied", False)).reshape(-1)[0]),
            "optimizer_update_authority": bool(
                np.asarray(data.get("optimizer_update_authority", False)).reshape(-1)[0]
            ),
            "gradient_reduction_authority": bool(
                np.asarray(data.get("gradient_reduction_authority", False)).reshape(-1)[0]
            ),
            "budget_spend_authority": bool(np.asarray(data.get("budget_spend_authority", False)).reshape(-1)[0]),
        }
    payload = json.loads(p.read_text(encoding="utf-8"))
    return {
        "shard_index": int(payload.get("shard_index", 0)),
        "pair_start": int(payload["pair_start"]),
        "pair_end": int(payload["pair_end"]),
        "linearization_archive_sha": str(payload["linearization_archive_sha"]),
        "joint_weight": np.asarray(payload["joint_weight"], dtype=np.float64),
        "rate_attack_deadzone_mask": np.asarray(
            payload["rate_attack_deadzone_mask"],
            dtype=bool,
        ),
        "optimizer_update_applied": bool(payload.get("optimizer_update_applied", False)),
        "optimizer_update_authority": bool(payload.get("optimizer_update_authority", False)),
        "gradient_reduction_authority": bool(payload.get("gradient_reduction_authority", False)),
        "budget_spend_authority": bool(payload.get("budget_spend_authority", False)),
    }


def build_z8_full_video_vjp_acquisition_contract() -> dict[str, Any]:
    """Return the stable contract embedded in Z8 driver metadata."""

    return {
        "schema": "z8_full_video_vjp_acquisition_contract.v1",
        "plan_schema": Z8_FULL_VIDEO_VJP_ACQUISITION_PLAN_SCHEMA,
        "surface_bundle_schema": Z8_FULL_VIDEO_VJP_SURFACE_BUNDLE_SCHEMA,
        "contest_mode": CONTEST_VIDEO_OVERFIT_MODE,
        "production_modes": [CORPUS_GENERALIZATION_MODE, HYBRID_CONTEST_PLUS_CORPUS_MODE],
        "contest_budget_spend_requires": [
            "full_video_pair_grid_coverage",
            "linearization_archive_sha_equals_current_archive_sha",
            "single_optimizer_update_after_full_shard_reduction",
            "relinearize_after_each_accepted_archive_mutation",
            "receiver_proof_plus_exact_cpu_cuda_before_score_authority",
        ],
        "production_budget_spend_requires": [
            "declared_corpus_manifest",
            "same_surface_schema",
            "explicit_generalization_target_mode",
        ],
        "minibatch_window_gradients_role": "ranking_probe_only_between_full_video_passes",
        "mlx_execution_model": "keep_full_video_resident_accumulate_pair_chunk_vjp_in_parallel",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


__all__ = [
    "Z8_FULL_VIDEO_VJP_ACQUISITION_PLAN_SCHEMA",
    "Z8_FULL_VIDEO_VJP_SURFACE_BUNDLE_SCHEMA",
    "Z8FullVideoVjpAcquisitionConfig",
    "assemble_z8_full_video_vjp_surface_bundle",
    "build_z8_full_video_vjp_acquisition_contract",
    "build_z8_full_video_vjp_acquisition_plan",
    "load_z8_full_video_vjp_surface_shard_file",
    "write_z8_full_video_vjp_acquisition_plan",
    "write_z8_full_video_vjp_surface_bundle",
]
