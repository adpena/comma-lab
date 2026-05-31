# SPDX-License-Identifier: MIT
"""Reusable Z8 P18/P19 freeze-vs-KKT allocator comparison primitives.

This module is intentionally TAC code, not a one-off experiment script. It owns
the deterministic allocator arms, archive mutation, matched operating-point
rows, and macOS advisory replay helpers used by Z8 codec experiments. Score
authority remains false: local DistortionNet replay is characterization signal,
not contest CPU/CUDA promotion.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from tac.optimization.joint_p18_p19_waterfill import (
    solve_joint_p18_p19_implicit_kkt_dykstra_allocator,
)
from tac.substrates.z8_hierarchical_predictive_coding.archive import (
    parse_z8hpc1_archive_bytes,
)
from tac.substrates.z8_hierarchical_predictive_coding.joint_p18_p19_deadzone_rate_attack import (
    _resize_pixel_map_to_grid,
    apply_deadzone_to_pair_details,
    build_joint_surface_for_pair,
    flatten_detail_coefficients,
    joint_deadzone_mask_for_pair,
    joint_keep_priority_for_pair,
    pack_pair_pyramids_to_wavelet_blob,
    parse_pair_blobs_from_wavelet_blob,
    posenet_pixel_jacobian_norm,
    push_pixel_saliency_to_detail_coeffs,
    segnet_boundary_pixel_saliency,
    splice_wavelet_blob_into_archive,
)

Z8_P18_P19_ALLOCATOR_COMPARISON_SCHEMA = "z8_p18_p19_freeze_vs_implicit_kkt_comparison.v2"
Z8_P18_P19_MATCHED_OPERATING_POINT_SCHEMA = "z8_p18_p19_allocator_matched_operating_point.v2"
Z8_P18_P19_ALLOCATOR_VERDICT_SCHEMA = "z8_p18_p19_allocator_comparison_verdict.v1"
LEGACY_SINGLE_NORM_P19_CHARACTERIZATION_SURFACE = "legacy_single_norm_pose_jacobian_characterization_v1"
TRUE_P19_MATERIALIZER_SURFACE = "per_axis_posenet_jacobian_mahalanobis_v1"
FREEZE_ALLOCATOR_ARM = "freeze"
IMPLICIT_KKT_ALLOCATOR_ARM = "implicit_kkt"
CANONICAL_N_BYTES = 37_545_489

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotable": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "is_contest_cpu_claim": False,
}


def freeze_deadzone_mask(keep_priority_flat: np.ndarray, *, keep_fraction: float) -> np.ndarray:
    """Freeze arm: keep the top ``keep_fraction`` atoms by keep priority."""

    return joint_deadzone_mask_for_pair(
        np.asarray(keep_priority_flat, dtype=np.float64),
        keep_fraction=float(keep_fraction),
    )


def implicit_kkt_deadzone_mask(
    keep_priority_flat: np.ndarray,
    *,
    budget: float,
    atom_capacity: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Implicit-KKT/Dykstra arm over the same monotone keep-priority ranking."""

    keep_priority = np.asarray(keep_priority_flat, dtype=np.float64)
    if keep_priority.ndim != 1:
        raise ValueError("keep_priority_flat must be one-dimensional")
    coarsening_priority = 1.0 / (1.0 + np.maximum(keep_priority, 0.0))
    allocator = solve_joint_p18_p19_implicit_kkt_dykstra_allocator(
        coarsening_priority=coarsening_priority,
        safe_rate_spend_mask=np.ones_like(coarsening_priority, dtype=bool),
        budget=float(budget),
        atom_capacity=float(atom_capacity),
    )
    deadzone = np.asarray(allocator["allocation"], dtype=np.float64) > 0.0
    return deadzone, allocator


def allocator_deadzone_mask(
    keep_priority_flat: np.ndarray,
    *,
    arm: str,
    knob: float,
    atom_capacity: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``(deadzone_mask, report)`` for one allocator arm."""

    keep_priority = np.asarray(keep_priority_flat, dtype=np.float64)
    if arm == FREEZE_ALLOCATOR_ARM:
        mask = freeze_deadzone_mask(keep_priority, keep_fraction=float(knob))
        return mask, {
            "schema": "z8_p18_p19_allocator_arm_report.v1",
            "arm": FREEZE_ALLOCATOR_ARM,
            "knob_name": "keep_fraction",
            "knob": float(knob),
            "allocation_rule": "threshold_keep_top_fraction_by_joint_keep_priority",
            "solver_blockers": [],
        }
    if arm == IMPLICIT_KKT_ALLOCATOR_ARM:
        budget = float(knob) * float(keep_priority.size)
        mask, allocator = implicit_kkt_deadzone_mask(
            keep_priority,
            budget=budget,
            atom_capacity=float(atom_capacity),
        )
        return mask, {
            "schema": "z8_p18_p19_allocator_arm_report.v1",
            "arm": IMPLICIT_KKT_ALLOCATOR_ARM,
            "knob_name": "budget_fraction",
            "knob": float(knob),
            "budget": float(budget),
            "atom_capacity": float(atom_capacity),
            "allocation_rule": "implicit_kkt_dykstra_box_budget_projection",
            "allocator": {
                key: value
                for key, value in allocator.items()
                if key
                not in {
                    "allocation",
                    "active_mask",
                    "dykstra_projection",
                    "implicit_jacobian_diagonal",
                    "implicit_budget_jacobian",
                }
            },
            "solver_blockers": list(allocator.get("solver_blockers") or []),
        }
    raise ValueError(f"unknown allocator arm {arm!r}")


def apply_allocator_arm_to_archive(
    archive_bytes: bytes,
    *,
    arm: str,
    knob: float,
    per_pair_joint: Sequence[Mapping[str, Any]],
    atom_capacity: float = 1.0,
) -> dict[str, Any]:
    """Apply one allocator arm to every Z8 wavelet-detail frame and repack."""

    sections = parse_z8hpc1_archive_bytes(archive_bytes)
    wavelet_start, wavelet_len = sections["wavelet_blob"]
    pyramids = parse_pair_blobs_from_wavelet_blob(archive_bytes[wavelet_start : wavelet_start + wavelet_len])
    if len(per_pair_joint) != len(pyramids):
        raise ValueError(
            f"per_pair_joint length must match archive pair count: {len(per_pair_joint)} vs {len(pyramids)}"
        )

    n_total = 0
    n_zeroed = 0
    frame_reports: list[dict[str, Any]] = []
    new_pyramids: list[dict[str, Any]] = []
    for pair_idx, pyramid in enumerate(pyramids):
        new_pyramid = dict(pyramid)
        for frame_key in ("frame_0", "frame_1"):
            details = pyramid[f"{frame_key}_details"]
            seg_term, pose_term, pose_null_mask = per_pair_joint[pair_idx][frame_key]
            coeff_mag = flatten_detail_coefficients(details)
            keep_priority = joint_keep_priority_for_pair(
                coeff_mag,
                np.asarray(seg_term, dtype=np.float64),
                np.asarray(pose_term, dtype=np.float64),
                np.asarray(pose_null_mask, dtype=bool),
            )
            mask, allocator_report = allocator_deadzone_mask(
                keep_priority,
                arm=arm,
                knob=float(knob),
                atom_capacity=float(atom_capacity),
            )
            new_details, pair_total, already_zero = apply_deadzone_to_pair_details(details, mask)
            new_pyramid[f"{frame_key}_details"] = new_details
            n_total += int(pair_total)
            n_zeroed += int(mask.sum())
            frame_reports.append(
                {
                    "schema": "z8_p18_p19_allocator_frame_application.v1",
                    "pair_index": int(pair_idx),
                    "frame_key": frame_key,
                    "coefficients_total": int(pair_total),
                    "coefficients_zeroed": int(mask.sum()),
                    "coefficients_already_zero": int(already_zero),
                    "deadzone_fraction": float(mask.sum() / max(pair_total, 1)),
                    "allocator_report": allocator_report,
                }
            )
        new_pyramids.append(new_pyramid)

    new_blob = pack_pair_pyramids_to_wavelet_blob(new_pyramids)
    new_archive = splice_wavelet_blob_into_archive(archive_bytes, new_blob)
    return {
        "schema": "z8_p18_p19_allocator_archive_application.v1",
        "arm": str(arm),
        "knob": float(knob),
        "archive_bytes": new_archive,
        "coefficients_total": int(n_total),
        "coefficients_zeroed": int(n_zeroed),
        "deadzone_fraction": float(n_zeroed / max(n_total, 1)),
        "frame_reports": frame_reports,
        **FALSE_AUTHORITY,
    }


def _row_match_key(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[float, float]:
    if "zip_bytes" in reference and "zip_bytes" in candidate:
        return (
            abs(float(candidate["zip_bytes"]) - float(reference["zip_bytes"])),
            abs(float(candidate["deadzone_fraction"]) - float(reference["deadzone_fraction"])),
        )
    if "rate" in reference and "rate" in candidate:
        return (
            abs(float(candidate["rate"]) - float(reference["rate"])),
            abs(float(candidate["deadzone_fraction"]) - float(reference["deadzone_fraction"])),
        )
    return (
        abs(float(candidate["deadzone_fraction"]) - float(reference["deadzone_fraction"])),
        0.0,
    )


def match_allocator_rows_on_deadzone(
    freeze_rows: Sequence[Mapping[str, Any]],
    implicit_kkt_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Match rows by charged rate/bytes first, dead-zone fraction second.

    The historical tool matched by dead-zone fraction. The contest binding axis
    is charged archive rate, so ZIP bytes/rate are now primary when present;
    dead-zone fraction remains a diagnostic tie-breaker and report field.
    """

    if not freeze_rows:
        return []
    if not implicit_kkt_rows:
        raise ValueError("implicit_kkt_rows must not be empty when freeze_rows are provided")
    matched: list[dict[str, Any]] = []
    for freeze in freeze_rows:
        best = min(implicit_kkt_rows, key=lambda row: _row_match_key(freeze, row))
        freeze_blockers = list(freeze.get("solver_blockers") or [])
        kkt_blockers = list(best.get("solver_blockers") or [])
        zip_gap = (
            abs(int(best["zip_bytes"]) - int(freeze["zip_bytes"]))
            if "zip_bytes" in freeze and "zip_bytes" in best
            else None
        )
        matched.append(
            {
                "schema": Z8_P18_P19_MATCHED_OPERATING_POINT_SCHEMA,
                "match_metric": "zip_bytes_then_deadzone_fraction"
                if zip_gap is not None
                else (
                    "rate_then_deadzone_fraction"
                    if "rate" in freeze and "rate" in best
                    else "deadzone_fraction"
                ),
                "freeze_knob": float(freeze["knob"]),
                "implicit_kkt_knob": float(best["knob"]),
                "freeze_deadzone_fraction": float(freeze["deadzone_fraction"]),
                "implicit_kkt_deadzone_fraction": float(best["deadzone_fraction"]),
                "deadzone_fraction_gap": abs(
                    float(best["deadzone_fraction"]) - float(freeze["deadzone_fraction"])
                ),
                "zip_bytes_gap": zip_gap,
                "rate_gap": abs(float(best["rate"]) - float(freeze["rate"])),
                "freeze_rate": float(freeze["rate"]),
                "implicit_kkt_rate": float(best["rate"]),
                "freeze_d_seg": float(freeze["d_seg"]),
                "implicit_kkt_d_seg": float(best["d_seg"]),
                "freeze_d_pose": float(freeze["d_pose"]),
                "implicit_kkt_d_pose": float(best["d_pose"]),
                "freeze_S": float(freeze["contest_score"]),
                "implicit_kkt_S": float(best["contest_score"]),
                "implicit_kkt_minus_freeze_S": float(best["contest_score"]) - float(freeze["contest_score"]),
                "freeze_solver_blockers": freeze_blockers,
                "implicit_kkt_solver_blockers": kkt_blockers,
            }
        )
    return matched


def classify_allocator_comparison(
    matched_operating_points: Sequence[Mapping[str, Any]],
    *,
    noise_band: float,
) -> dict[str, Any]:
    """Return an honest advisory verdict from matched operating points."""

    if noise_band < 0.0:
        raise ValueError("noise_band must be >= 0")
    if not matched_operating_points:
        return {
            "schema": Z8_P18_P19_ALLOCATOR_VERDICT_SCHEMA,
            "verdict_catalog_307": "characterization_not_score_claim",
            "winner": None,
            "blocker": "no_matched_operating_points",
            "n_matched_points": 0,
            **FALSE_AUTHORITY,
        }
    matched_blockers = sorted(
        {
            str(blocker)
            for row in matched_operating_points
            for blocker in row.get("implicit_kkt_solver_blockers", [])
        }
    )
    if matched_blockers:
        return {
            "schema": Z8_P18_P19_ALLOCATOR_VERDICT_SCHEMA,
            "verdict_catalog_307": "characterization_not_score_claim",
            "noise_band": float(noise_band),
            "winner": None,
            "blocker": "implicit_kkt_solver_blockers_present",
            "solver_blockers": matched_blockers,
            "n_matched_points": len(matched_operating_points),
            **FALSE_AUTHORITY,
        }
    headline = min(matched_operating_points, key=lambda row: float(row["deadzone_fraction_gap"]))
    deltas = [float(row["implicit_kkt_minus_freeze_S"]) for row in matched_operating_points]
    headline_delta = float(headline["implicit_kkt_minus_freeze_S"])
    n_kkt_wins = int(sum(delta < -float(noise_band) for delta in deltas))
    n_freeze_wins = int(sum(delta > float(noise_band) for delta in deltas))
    n_ties = len(deltas) - n_kkt_wins - n_freeze_wins
    if abs(headline_delta) <= float(noise_band):
        winner = "TIE_WITHIN_NOISE"
    elif headline_delta < 0.0:
        winner = "IMPLICIT_KKT"
    else:
        winner = "FREEZE"
    return {
        "schema": Z8_P18_P19_ALLOCATOR_VERDICT_SCHEMA,
        "verdict_catalog_307": "characterization_not_score_claim",
        "noise_band": float(noise_band),
        "headline_match": dict(headline),
        "headline_implicit_kkt_minus_freeze_S": headline_delta,
        "winner": winner,
        "mean_implicit_kkt_minus_freeze_S_over_all_matches": float(np.mean(deltas)),
        "n_implicit_kkt_wins": n_kkt_wins,
        "n_freeze_wins": n_freeze_wins,
        "n_ties_within_noise": n_ties,
        "n_matched_points": len(deltas),
        **FALSE_AUTHORITY,
    }


def build_allocator_comparison_report(
    *,
    baseline: Mapping[str, Any],
    freeze_rows: Sequence[Mapping[str, Any]],
    implicit_kkt_rows: Sequence[Mapping[str, Any]],
    matched_operating_points: Sequence[Mapping[str, Any]],
    verdict: Mapping[str, Any],
    video_path: str,
    num_pairs: int,
    eval_h: int,
    eval_w: int,
    pose_null_fraction: float,
    seg_tau: float,
    frontier_cpu_anchor: float,
    wall_clock_seconds: float,
) -> dict[str, Any]:
    """Build the canonical non-promotional comparison result payload."""

    return {
        "schema": Z8_P18_P19_ALLOCATOR_COMPARISON_SCHEMA,
        "characterization_not_score_claim": True,
        "binding_axis": "rate",
        "surface_kind": LEGACY_SINGLE_NORM_P19_CHARACTERIZATION_SURFACE,
        "materializer_authority_surface_kind": TRUE_P19_MATERIALIZER_SURFACE,
        "budget_spend_authority": False,
        "budget_spend_blockers": [
            "allocator_comparison_is_characterization_not_materializer_surface",
            "legacy_single_norm_p19_surface_not_true_six_axis_mahalanobis",
        ],
        "comparison": (
            "freeze-allocate vs implicit-KKT-Dykstra on identical RD-energy-aware "
            "joint P18/P19 keep priority; matched on charged rate/bytes first; "
            "byte-closed Z8 advisory replay required for both arms"
        ),
        "joint_keep_priority": "keep_priority_i = |coeff_i| * (1 + seg_protect_i) * (1 + pose_protect_i)",
        "video_path": str(video_path),
        "num_pairs": int(num_pairs),
        "eval_h": int(eval_h),
        "eval_w": int(eval_w),
        "pose_null_fraction": float(pose_null_fraction),
        "seg_boundary_tau": float(seg_tau),
        "baseline": dict(baseline),
        "arm_a_freeze": [dict(row) for row in freeze_rows],
        "arm_b_implicit_kkt": [dict(row) for row in implicit_kkt_rows],
        "matched_operating_points": [dict(row) for row in matched_operating_points],
        "verdict": dict(verdict),
        "frontier_cpu_anchor": float(frontier_cpu_anchor),
        "axis_tag": "[macOS-CPU advisory]",
        "evidence_grade": "macos_cpu_advisory",
        "wall_clock_seconds": round(float(wall_clock_seconds), 1),
        **FALSE_AUTHORITY,
    }


def canonical_z8_config(*, num_pairs: int, eval_h: int, eval_w: int) -> Any:
    """Return the lightweight config shape consumed by the Z8 binding builder."""

    return SimpleNamespace(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        num_pairs=int(num_pairs),
        deterministic_state_dim=16,
        ego_motion_dim=6,
        eval_size=(int(eval_h), int(eval_w)),
    )


def build_z8_baseline_archive_from_video(
    *,
    video_path: str,
    num_pairs: int,
    eval_h: int,
    eval_w: int,
) -> tuple[Any, np.ndarray, np.ndarray, bytes]:
    """Build the canonical Z8HPC1 baseline archive from real video pairs."""

    from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
        build_canonical_quadruple_binding_from_z8_config,
        build_z8hpc1_archive_bytes_from_canonical_quadruple,
        load_real_video_pair_targets_numpy,
    )

    cfg = canonical_z8_config(num_pairs=num_pairs, eval_h=eval_h, eval_w=eval_w)
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    f0, f1 = load_real_video_pair_targets_numpy(
        video_path,
        num_pairs=int(num_pairs),
        output_height=int(eval_h),
        output_width=int(eval_w),
    )
    archive_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(binding, f0, f1)
    return binding, np.asarray(f0), np.asarray(f1), archive_bytes


def byte_close_z8_archive_zip(
    archive_bytes: bytes,
    out_dir: Path,
    *,
    repo_root: str | Path,
) -> int:
    """Export a Z8 archive package and return charged ZIP bytes."""

    from tac.substrates.z8_hierarchical_predictive_coding.archive_candidate import (
        export_z8hpc1_archive_bytes,
    )

    pkg_dir = Path(out_dir) / "byte_closed_archive"
    archive_zip_path, _sha, _bin = export_z8hpc1_archive_bytes(
        archive_bytes,
        pkg_dir,
        repo_root=repo_root,
        emit_archive_bound_candidate_package=False,
        emit_byte_mutation_proof=False,
        retain_receiver_proof_output=False,
    )
    return int(Path(archive_zip_path).stat().st_size)


def reconstruct_z8_archive_pairs_unit(archive_bytes: bytes) -> np.ndarray:
    """Reconstruct archive pairs as unit-range ``(pairs,2,H,W,3)`` arrays."""

    from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
        reconstruct_pair_rgb_from_pyramid,
    )
    from tac.substrates.z8_hierarchical_predictive_coding.runtime_payload_bridge import (
        projected_pair_pyramids_from_archive_bytes,
    )

    binding, pair_pyramids, _stats = projected_pair_pyramids_from_archive_bytes(archive_bytes)
    recon: list[Any] = []
    for pyramid in pair_pyramids:
        r0, r1 = reconstruct_pair_rgb_from_pyramid(binding, pyramid)
        r0_hwc = np.transpose(r0[0], (1, 2, 0))
        r1_hwc = np.transpose(r1[0], (1, 2, 0))
        recon.append(np.stack([r0_hwc, r1_hwc], axis=0))
    if not recon:
        raise ValueError("Z8 archive reconstructs zero pairs")
    return np.stack(recon, axis=0).astype(np.float32)


def decode_gt_pairs_from_video(video_path: str, num_pairs: int) -> np.ndarray:
    """Decode contest-grid GT pairs from a video path."""

    from tac.data import decode_video

    frames = decode_video(video_path, target_h=384, target_w=512, max_frames=2 * int(num_pairs))
    if len(frames) < 2 * int(num_pairs):
        raise RuntimeError(f"decoded {len(frames)} frames; need {2 * int(num_pairs)}")
    gt = np.stack([frame.numpy() for frame in frames[: 2 * int(num_pairs)]], axis=0)
    return gt.reshape(int(num_pairs), 2, 384, 512, 3).astype(np.float32)


def resize_recon_to_scorer_grid(recon_unit: np.ndarray, *, num_pairs: int) -> np.ndarray:
    """Resize reconstructed Z8 frames to DistortionNet scorer grid."""

    import torch
    import torch.nn.functional as F

    flat = recon_unit.reshape(int(num_pairs) * 2, recon_unit.shape[2], recon_unit.shape[3], 3)
    tensor = torch.from_numpy(np.transpose(flat, (0, 3, 1, 2)).copy())
    up = F.interpolate(tensor, size=(384, 512), mode="bicubic", align_corners=False)
    up = up.clamp(0.0, 1.0) * 255.0
    up_np = up.numpy().astype(np.float32)
    return np.transpose(up_np, (0, 2, 3, 1)).reshape(int(num_pairs), 2, 384, 512, 3)


def load_real_distortion_net(*, upstream_dir: str | Path, device: str = "cpu") -> Any:
    """Load upstream DistortionNet on the requested local device."""

    import sys

    upstream = Path(upstream_dir)
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from modules import DistortionNet  # type: ignore[import-not-found]

    distortion_net = DistortionNet().eval()
    distortion_net.load_state_dicts(
        str(upstream / "models" / "posenet.safetensors"),
        str(upstream / "models" / "segnet.safetensors"),
        device,
    )
    return distortion_net


def measure_distortion_pairs(
    distortion_net: Any,
    gt_pairs: np.ndarray,
    recon_pairs: np.ndarray,
    *,
    batch: int = 32,
) -> dict[str, Any]:
    """Measure local DistortionNet component means for reconstructed pairs."""

    import torch

    n_pairs = int(gt_pairs.shape[0])
    d_seg_all: list[float] = []
    d_pose_all: list[float] = []
    gt_t = torch.from_numpy(gt_pairs)
    rec_t = torch.from_numpy(recon_pairs)
    for start in range(0, n_pairs, int(batch)):
        end = min(start + int(batch), n_pairs)
        with torch.inference_mode():
            d_pose, d_seg = distortion_net.compute_distortion(gt_t[start:end], rec_t[start:end])
        d_pose_all.extend(float(value) for value in d_pose.tolist())
        d_seg_all.extend(float(value) for value in d_seg.tolist())
    return {
        "mean_d_seg": float(np.mean(d_seg_all)),
        "mean_d_pose": float(np.mean(d_pose_all)),
        "max_d_seg": float(np.max(d_seg_all)),
        "max_d_pose": float(np.max(d_pose_all)),
        "n_pairs": n_pairs,
    }


def measure_z8_archive_advisory(
    archive_bytes: bytes,
    *,
    num_pairs: int,
    gt_pairs: np.ndarray,
    distortion_net: Any,
    out_dir: Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Byte-close and measure a Z8 archive with local DistortionNet advisory replay."""

    zip_size = byte_close_z8_archive_zip(archive_bytes, Path(out_dir), repo_root=repo_root)
    recon_unit = reconstruct_z8_archive_pairs_unit(archive_bytes)
    recon_scorer = resize_recon_to_scorer_grid(recon_unit, num_pairs=int(num_pairs))
    metrics = measure_distortion_pairs(distortion_net, gt_pairs, recon_scorer)
    rate = float(zip_size) / float(CANONICAL_N_BYTES)
    seg_term = 100.0 * float(metrics["mean_d_seg"])
    pose_term = math.sqrt(10.0 * float(metrics["mean_d_pose"]))
    rate_term = 25.0 * rate
    return {
        "zip_bytes": int(zip_size),
        "rate": rate,
        "d_seg": float(metrics["mean_d_seg"]),
        "d_pose": float(metrics["mean_d_pose"]),
        "seg_term": seg_term,
        "pose_term": pose_term,
        "rate_term": rate_term,
        "contest_score": seg_term + pose_term + rate_term,
        "max_d_pose": float(metrics["max_d_pose"]),
        "axis_tag": "[macOS-CPU advisory]",
        **FALSE_AUTHORITY,
    }


def build_per_pair_joint_surfaces(
    binding: Any,
    f0: np.ndarray,
    f1: np.ndarray,
    pose_scorer: Any,
    seg_scorer: Any,
    *,
    num_pairs: int,
    eval_h: int,
    eval_w: int,
    pose_null_fraction: float,
    seg_tau: float,
) -> list[dict[str, Any]]:
    """Build legacy single-norm per-frame detail saliencies for characterization.

    This is not the true six-axis P19 materializer authority surface. It is kept
    for apples-to-apples allocator comparison against the older freeze lane, and
    every result emitted by this module remains false-authority for budget spend.
    """

    import torch

    out: list[dict[str, Any]] = []
    for pair_idx in range(int(num_pairs)):
        gt = torch.from_numpy(np.stack([f0[pair_idx], f1[pair_idx]], axis=0)[None]).float()
        seg_384 = segnet_boundary_pixel_saliency(pose_scorer, seg_scorer, gt, tau=float(seg_tau))
        seg_native = _resize_pixel_map_to_grid(seg_384, int(eval_h), int(eval_w))
        pose2 = posenet_pixel_jacobian_norm(pose_scorer, seg_scorer, gt)
        seg_detail_coeffs = push_pixel_saliency_to_detail_coeffs(binding, seg_native, num_levels=3)
        per_frame: dict[str, Any] = {}
        for frame_index, frame_key in enumerate(("frame_0", "frame_1")):
            frame_seg_detail_coeffs = segnet_detail_saliency_for_scored_frame(
                seg_detail_coeffs,
                frame_key=frame_key,
            )
            pose_detail_coeffs = push_pixel_saliency_to_detail_coeffs(binding, pose2[frame_index], num_levels=3)
            surface = build_joint_surface_for_pair(
                seg_detail_saliency=frame_seg_detail_coeffs,
                pose_detail_saliency=pose_detail_coeffs,
                d_pose=0.5,
                pose_null_fraction=float(pose_null_fraction),
            )
            per_frame[frame_key] = (
                np.asarray(surface["segnet_term"], dtype=np.float64),
                np.asarray(surface["pose_term"], dtype=np.float64),
                np.asarray(surface["pose_null_mask"], dtype=bool),
            )
        out.append(per_frame)
    return out


def segnet_detail_saliency_for_scored_frame(
    seg_detail_saliency: Sequence[Mapping[str, np.ndarray]],
    *,
    frame_key: str,
) -> list[dict[str, np.ndarray]]:
    """Return SegNet detail saliency only for the scored last frame.

    Contest SegNet observes pair frame 1. Frame 0 still matters through PoseNet,
    but applying the same P18 saliency to frame 0 over-protects coefficients
    that the SegNet term cannot see.
    """

    if frame_key == "frame_1":
        return [
            {key: np.asarray(value, dtype=np.float64) for key, value in level.items()}
            for level in seg_detail_saliency
        ]
    if frame_key != "frame_0":
        raise ValueError(f"unknown frame_key {frame_key!r}")
    return [
        {key: np.zeros_like(np.asarray(value, dtype=np.float64)) for key, value in level.items()}
        for level in seg_detail_saliency
    ]


def run_allocator_comparison(
    *,
    video_path: str,
    num_pairs: int,
    eval_h: int,
    eval_w: int,
    freeze_keep_fractions: Sequence[float],
    implicit_kkt_budget_fractions: Sequence[float],
    pose_null_fraction: float,
    seg_tau: float,
    noise_band: float,
    out_dir: Path,
    repo_root: str | Path,
    upstream_dir: str | Path,
    frontier_cpu_anchor: float,
    emit: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the local advisory allocator comparison end to end."""

    started = time.time()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log = emit or (lambda _message: None)

    log(f"[z8-cmp] build baseline: {num_pairs} pairs @ ({eval_h},{eval_w})")
    binding, f0, f1, baseline_archive = build_z8_baseline_archive_from_video(
        video_path=video_path,
        num_pairs=int(num_pairs),
        eval_h=int(eval_h),
        eval_w=int(eval_w),
    )
    gt_pairs = decode_gt_pairs_from_video(video_path, int(num_pairs))
    distortion_net = load_real_distortion_net(upstream_dir=upstream_dir, device="cpu")
    baseline = measure_z8_archive_advisory(
        baseline_archive,
        num_pairs=int(num_pairs),
        gt_pairs=gt_pairs,
        distortion_net=distortion_net,
        out_dir=out_dir / "baseline",
        repo_root=repo_root,
    )
    log(
        f"[z8-cmp] BASELINE: rate={baseline['rate']:.4f} d_seg={baseline['d_seg']:.6f} "
        f"d_pose={baseline['d_pose']:.4f} S={baseline['contest_score']:.4f}"
    )

    from tac.scorer import load_differentiable_scorers

    pose_scorer, seg_scorer = load_differentiable_scorers(str(upstream_dir), device="cpu")
    per_pair_joint = build_per_pair_joint_surfaces(
        binding,
        f0,
        f1,
        pose_scorer,
        seg_scorer,
        num_pairs=int(num_pairs),
        eval_h=int(eval_h),
        eval_w=int(eval_w),
        pose_null_fraction=float(pose_null_fraction),
        seg_tau=float(seg_tau),
    )

    freeze_rows: list[dict[str, Any]] = []
    for keep_fraction in freeze_keep_fractions:
        row = measure_allocator_arm_advisory(
            baseline_archive,
            arm=FREEZE_ALLOCATOR_ARM,
            knob=float(keep_fraction),
            per_pair_joint=per_pair_joint,
            num_pairs=int(num_pairs),
            gt_pairs=gt_pairs,
            distortion_net=distortion_net,
            out_dir=out_dir / f"freeze_keep{float(keep_fraction):.3f}",
            repo_root=repo_root,
        )
        freeze_rows.append(row)
        log(
            f"[z8-cmp] FREEZE keep={float(keep_fraction):.3f} deadzone={row['deadzone_fraction']:.4f} "
            f"rate={row['rate']:.4f} d_seg={row['d_seg']:.6f} d_pose={row['d_pose']:.4f} S={row['contest_score']:.4f}"
        )

    implicit_kkt_rows: list[dict[str, Any]] = []
    for budget_fraction in implicit_kkt_budget_fractions:
        row = measure_allocator_arm_advisory(
            baseline_archive,
            arm=IMPLICIT_KKT_ALLOCATOR_ARM,
            knob=float(budget_fraction),
            per_pair_joint=per_pair_joint,
            num_pairs=int(num_pairs),
            gt_pairs=gt_pairs,
            distortion_net=distortion_net,
            out_dir=out_dir / f"kkt_budget{float(budget_fraction):.3f}",
            repo_root=repo_root,
        )
        implicit_kkt_rows.append(row)
        log(
            f"[z8-cmp] KKT budget_frac={float(budget_fraction):.3f} deadzone={row['deadzone_fraction']:.4f} "
            f"rate={row['rate']:.4f} d_seg={row['d_seg']:.6f} d_pose={row['d_pose']:.4f} S={row['contest_score']:.4f}"
        )

    matched = match_allocator_rows_on_deadzone(freeze_rows, implicit_kkt_rows)
    verdict = classify_allocator_comparison(matched, noise_band=float(noise_band))
    return build_allocator_comparison_report(
        baseline=baseline,
        freeze_rows=freeze_rows,
        implicit_kkt_rows=implicit_kkt_rows,
        matched_operating_points=matched,
        verdict=verdict,
        video_path=video_path,
        num_pairs=int(num_pairs),
        eval_h=int(eval_h),
        eval_w=int(eval_w),
        pose_null_fraction=float(pose_null_fraction),
        seg_tau=float(seg_tau),
        frontier_cpu_anchor=float(frontier_cpu_anchor),
        wall_clock_seconds=time.time() - started,
    )


def measure_allocator_arm_advisory(
    archive_bytes: bytes,
    *,
    arm: str,
    knob: float,
    per_pair_joint: Sequence[Mapping[str, Any]],
    num_pairs: int,
    gt_pairs: np.ndarray,
    distortion_net: Any,
    out_dir: Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Apply an allocator arm, byte-close it, and attach local advisory metrics."""

    application = apply_allocator_arm_to_archive(
        archive_bytes,
        arm=arm,
        knob=float(knob),
        per_pair_joint=per_pair_joint,
    )
    solver_blockers = sorted(
        {
            str(blocker)
            for frame in application["frame_reports"]
            for blocker in frame.get("allocator_report", {}).get("solver_blockers", [])
        }
    )
    measured = measure_z8_archive_advisory(
        application["archive_bytes"],
        num_pairs=int(num_pairs),
        gt_pairs=gt_pairs,
        distortion_net=distortion_net,
        out_dir=Path(out_dir),
        repo_root=repo_root,
    )
    return {
        "schema": "z8_p18_p19_allocator_arm_measurement.v1",
        "arm": str(arm),
        "knob": float(knob),
        "coeffs_total": int(application["coefficients_total"]),
        "coeffs_zeroed": int(application["coefficients_zeroed"]),
        "deadzone_fraction": float(application["deadzone_fraction"]),
        "solver_blockers": solver_blockers,
        **measured,
    }


def write_allocator_comparison_result(result: Mapping[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Write result JSON and return a compact manifest."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema": "z8_p18_p19_allocator_comparison_result_manifest.v1",
        "result_path": result_path.as_posix(),
        "result_schema": result.get("schema"),
        "axis_tag": result.get("axis_tag"),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


__all__ = [
    "FREEZE_ALLOCATOR_ARM",
    "IMPLICIT_KKT_ALLOCATOR_ARM",
    "LEGACY_SINGLE_NORM_P19_CHARACTERIZATION_SURFACE",
    "TRUE_P19_MATERIALIZER_SURFACE",
    "Z8_P18_P19_ALLOCATOR_COMPARISON_SCHEMA",
    "Z8_P18_P19_ALLOCATOR_VERDICT_SCHEMA",
    "Z8_P18_P19_MATCHED_OPERATING_POINT_SCHEMA",
    "allocator_deadzone_mask",
    "apply_allocator_arm_to_archive",
    "build_allocator_comparison_report",
    "build_per_pair_joint_surfaces",
    "build_z8_baseline_archive_from_video",
    "byte_close_z8_archive_zip",
    "classify_allocator_comparison",
    "decode_gt_pairs_from_video",
    "freeze_deadzone_mask",
    "implicit_kkt_deadzone_mask",
    "load_real_distortion_net",
    "match_allocator_rows_on_deadzone",
    "measure_allocator_arm_advisory",
    "measure_distortion_pairs",
    "measure_z8_archive_advisory",
    "reconstruct_z8_archive_pairs_unit",
    "resize_recon_to_scorer_grid",
    "run_allocator_comparison",
    "segnet_detail_saliency_for_scored_frame",
    "write_allocator_comparison_result",
]
