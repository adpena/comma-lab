# SPDX-License-Identifier: MIT
"""Z8 joint P18/P19 dead-zone RATE ATTACK driver — empirical actuator increment.

Z8's binding axis is RATE (~97% of the 104.94 [macOS-CPU advisory] contest
score). The 99.5%-of-bytes ``wavelet_blob`` stores raw float32 Mallat detail
coefficients of the real frames. This driver:

1. builds the byte-closed Z8HPC1 baseline archive (real frames -> Mallat
   decompose -> brotli q=11) and measures its real contest score;
2. computes, per pair, the REAL joint P18/P19 weight over the detail
   coefficients from real differentiable SegNet boundary saliency (P18) and
   real differentiable PoseNet pixel-Jacobian (P19) pushed into the
   coefficient domain via the analysis DWT (adjoint of the orthonormal
   Daubechies synthesis);
3. at a target keep-fraction, ACTUALLY zeros the dead-zone coefficients
   (low joint weight AND pose-null), re-packs the wavelet blob, byte-closes
   the archive, and RE-MEASURES the real contest score (rate + d_seg + d_pose);
4. runs the SAME rate target with a SegNet-only (magnitude) dead-zone and
   compares: does the P19 pose-null protection actually preserve pose
   distortion at matched rate? (the whole point of the joint surface).

Per the joint-surface fresh-surface discipline the full-video replay (the
re-measured byte-closed archive) is the authority, NOT the proposal weights.

ALL outputs are ``[macOS-CPU advisory]`` / ``[macOS-MLX research-signal]``,
non-promotable, with canonical Provenance, per Catalog #192/#341/#127/#323.
Local macOS-CPU is NOT 1:1 contest hardware -> explicitly NOT a [contest-CPU]
claim. $0 local, NO cloud, NO paid GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "upstream"
DEFAULT_VIDEO = str(UPSTREAM / "videos" / "0.mkv")
CANONICAL_N_BYTES = 37_545_489
FRONTIER_CPU_ANCHOR = 0.192  # HISTORICAL_SCORE_LITERAL_OK:gap_banner_only_read_from_pointer_at_runtime


def _canonical_cfg(*, num_pairs: int, eval_h: int, eval_w: int) -> Any:
    return SimpleNamespace(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        num_pairs=num_pairs,
        deterministic_state_dim=16,
        ego_motion_dim=6,
        eval_size=(eval_h, eval_w),
    )


# ---------------------------------------------------------------------------
# Archive build + byte-closure (reused contract from WAVE-1G score tool)
# ---------------------------------------------------------------------------


def _build_baseline(*, video_path: str, num_pairs: int, eval_h: int, eval_w: int):
    import numpy as np

    from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
        build_canonical_quadruple_binding_from_z8_config,
        build_z8hpc1_archive_bytes_from_canonical_quadruple,
        load_real_video_pair_targets_numpy,
    )

    cfg = _canonical_cfg(num_pairs=num_pairs, eval_h=eval_h, eval_w=eval_w)
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    f0, f1 = load_real_video_pair_targets_numpy(
        video_path, num_pairs=num_pairs, output_height=eval_h, output_width=eval_w
    )
    archive_bytes = build_z8hpc1_archive_bytes_from_canonical_quadruple(binding, f0, f1)
    return binding, np.asarray(f0), np.asarray(f1), archive_bytes


def _byte_close_zip(archive_bytes: bytes, out_dir: Path) -> int:
    from tac.substrates.z8_hierarchical_predictive_coding.archive_candidate import (
        export_z8hpc1_archive_bytes,
    )

    pkg_dir = out_dir / "byte_closed_archive"
    archive_zip_path, _sha, _bin = export_z8hpc1_archive_bytes(
        archive_bytes,
        pkg_dir,
        repo_root=REPO_ROOT,
        emit_archive_bound_candidate_package=False,
        emit_byte_mutation_proof=False,
        retain_receiver_proof_output=False,
    )
    return int(Path(archive_zip_path).stat().st_size)


# ---------------------------------------------------------------------------
# Real DistortionNet measurement of a byte-closed archive
# ---------------------------------------------------------------------------


def _reconstruct_archive_pairs(archive_bytes: bytes):
    import numpy as np

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
    return np.stack(recon, axis=0).astype(np.float32)


def _decode_gt_pairs(video_path: str, num_pairs: int):
    import numpy as np

    from tac.data import decode_video

    frames = decode_video(video_path, target_h=384, target_w=512, max_frames=2 * num_pairs)
    if len(frames) < 2 * num_pairs:
        raise RuntimeError(f"decoded {len(frames)} frames; need {2 * num_pairs}")
    gt = np.stack([f.numpy() for f in frames[: 2 * num_pairs]], axis=0)
    return gt.reshape(num_pairs, 2, 384, 512, 3).astype(np.float32)


def _resize_recon_to_scorer_grid(recon_unit, *, num_pairs: int):
    import numpy as np
    import torch
    import torch.nn.functional as F

    flat = recon_unit.reshape(num_pairs * 2, recon_unit.shape[2], recon_unit.shape[3], 3)
    t = torch.from_numpy(np.transpose(flat, (0, 3, 1, 2)).copy())
    up = F.interpolate(t, size=(384, 512), mode="bicubic", align_corners=False)
    up = up.clamp(0.0, 1.0) * 255.0
    up_np = up.numpy().astype(np.float32)
    return np.transpose(up_np, (0, 2, 3, 1)).reshape(num_pairs, 2, 384, 512, 3)


def _real_distortion_net():
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from modules import DistortionNet  # type: ignore[import-not-found]

    dn = DistortionNet().eval()
    dn.load_state_dicts(
        str(UPSTREAM / "models" / "posenet.safetensors"),
        str(UPSTREAM / "models" / "segnet.safetensors"),
        "cpu",
    )
    return dn


def _measure(dn, gt_pairs, recon_pairs, *, batch: int = 32) -> dict:
    import numpy as np
    import torch

    n = gt_pairs.shape[0]
    d_seg_all: list[float] = []
    d_pose_all: list[float] = []
    gt_t = torch.from_numpy(gt_pairs)
    rec_t = torch.from_numpy(recon_pairs)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        with torch.inference_mode():
            d_pose, d_seg = dn.compute_distortion(gt_t[s:e], rec_t[s:e])
        d_pose_all.extend(float(x) for x in d_pose.tolist())
        d_seg_all.extend(float(x) for x in d_seg.tolist())
    return {
        "mean_d_seg": float(np.mean(d_seg_all)),
        "mean_d_pose": float(np.mean(d_pose_all)),
        "max_d_seg": float(np.max(d_seg_all)),
        "max_d_pose": float(np.max(d_pose_all)),
        "n_pairs": int(n),
    }


def _measure_archive(archive_bytes: bytes, *, num_pairs: int, gt_pairs, dn, out_dir: Path) -> dict:
    zip_size = _byte_close_zip(archive_bytes, out_dir)
    recon_unit = _reconstruct_archive_pairs(archive_bytes)
    recon_scorer = _resize_recon_to_scorer_grid(recon_unit, num_pairs=num_pairs)
    metrics = _measure(dn, gt_pairs, recon_scorer)
    rate = zip_size / CANONICAL_N_BYTES
    seg_term = 100.0 * metrics["mean_d_seg"]
    pose_term = math.sqrt(10.0 * metrics["mean_d_pose"])
    rate_term = 25.0 * rate
    score = seg_term + pose_term + rate_term
    return {
        "zip_bytes": zip_size,
        "rate": rate,
        "d_seg": metrics["mean_d_seg"],
        "d_pose": metrics["mean_d_pose"],
        "seg_term": seg_term,
        "pose_term": pose_term,
        "rate_term": rate_term,
        "contest_score": score,
        "max_d_pose": metrics["max_d_pose"],
    }


# ---------------------------------------------------------------------------
# Joint P18/P19 + SegNet-only dead-zone attacks over the wavelet blob
# ---------------------------------------------------------------------------


def _per_pair_joint_masks(
    binding,
    f0,
    f1,
    pose_scorer,
    seg_scorer,
    *,
    num_pairs: int,
    eval_h: int,
    eval_w: int,
    pose_null_fraction: float,
    seg_tau: float,
):
    """Return per-pair per-frame (seg_term, pose_term, pose_null_mask) over the
    detail coefficients. The keep priority (which multiplies in the actual coeff
    magnitudes) is computed at attack time from these saliencies."""

    import numpy as np
    import torch

    from tac.substrates.z8_hierarchical_predictive_coding.joint_p18_p19_deadzone_rate_attack import (
        _resize_pixel_map_to_grid,
        build_joint_surface_for_pair,
        posenet_pixel_jacobian_norm,
        push_pixel_saliency_to_detail_coeffs,
        segnet_boundary_pixel_saliency,
    )

    out: list[dict[str, Any]] = []
    for pair_idx in range(num_pairs):
        gt = torch.from_numpy(
            np.stack([f0[pair_idx], f1[pair_idx]], axis=0)[None]
        ).float()  # (1, 2, H, W, C)
        seg_384 = segnet_boundary_pixel_saliency(pose_scorer, seg_scorer, gt, tau=seg_tau)
        seg_native = _resize_pixel_map_to_grid(seg_384, eval_h, eval_w)
        pose2 = posenet_pixel_jacobian_norm(pose_scorer, seg_scorer, gt)  # (2, H, W)
        seg_dc = push_pixel_saliency_to_detail_coeffs(binding, seg_native, num_levels=3)
        per_frame: dict[str, Any] = {}
        for fi, frame_key in enumerate(("frame_0", "frame_1")):
            pose_dc = push_pixel_saliency_to_detail_coeffs(binding, pose2[fi], num_levels=3)
            surface = build_joint_surface_for_pair(
                seg_detail_saliency=seg_dc,
                pose_detail_saliency=pose_dc,
                d_pose=0.5,
                pose_null_fraction=pose_null_fraction,
            )
            per_frame[frame_key] = (
                np.asarray(surface["segnet_term"], dtype=np.float64),
                np.asarray(surface["pose_term"], dtype=np.float64),
                np.asarray(surface["pose_null_mask"], dtype=bool),
            )
        out.append(per_frame)
    return out


def _apply_attack_to_blob(
    archive_bytes: bytes,
    *,
    keep_fraction: float,
    mode: str,
    per_pair_joint: list[dict[str, Any]] | None,
):
    """Apply a dead-zone attack to every pair's frame_0/frame_1 details and
    return (new_archive_bytes, n_total, n_zeroed). ``mode`` in
    {"joint", "segnet_only", "magnitude"}: joint = mag*seg*pose protection;
    segnet_only = mag*seg (no P19); magnitude = pure RD baseline."""

    import numpy as np

    from tac.substrates.z8_hierarchical_predictive_coding.archive import (
        parse_z8hpc1_archive_bytes,
    )
    from tac.substrates.z8_hierarchical_predictive_coding.joint_p18_p19_deadzone_rate_attack import (
        _flatten_coeffs,
        apply_deadzone_to_pair_details,
        joint_deadzone_mask_for_pair,
        joint_keep_priority_for_pair,
        magnitude_deadzone_mask_for_pair,
        pack_pair_pyramids_to_wavelet_blob,
        parse_pair_blobs_from_wavelet_blob,
        splice_wavelet_blob_into_archive,
    )

    secs = parse_z8hpc1_archive_bytes(archive_bytes)
    ws, wl = secs["wavelet_blob"]
    pyramids = parse_pair_blobs_from_wavelet_blob(archive_bytes[ws : ws + wl])
    n_total = 0
    n_zeroed = 0
    new_pyramids: list[dict[str, Any]] = []
    for pair_idx, pyramid in enumerate(pyramids):
        new_pyramid = dict(pyramid)
        for frame_key in ("frame_0", "frame_1"):
            details = pyramid[f"{frame_key}_details"]
            if mode == "joint":
                seg_term, pose_term, pn = per_pair_joint[pair_idx][frame_key]
                coeff_mag = _flatten_coeffs(details)
                keep_priority = joint_keep_priority_for_pair(
                    coeff_mag, seg_term, pose_term, pn
                )
                mask = joint_deadzone_mask_for_pair(keep_priority, keep_fraction=keep_fraction)
            elif mode == "segnet_only":
                # SegNet-only baseline = magnitude * seg-protection, NO P19 pose
                # protection. The joint-vs-segnet_only delta isolates P19.
                seg_term, _pose_term, _pn = per_pair_joint[pair_idx][frame_key]
                coeff_mag = _flatten_coeffs(details)
                keep_priority = joint_keep_priority_for_pair(
                    coeff_mag,
                    seg_term,
                    np.zeros_like(seg_term),  # no pose term
                    np.ones_like(seg_term, dtype=bool),  # all pose-null (no protect)
                    pose_protect_gain=0.0,
                )
                mask = joint_deadzone_mask_for_pair(keep_priority, keep_fraction=keep_fraction)
            elif mode == "magnitude":
                # Pure RD-optimal magnitude baseline (no scorer at all).
                mask = magnitude_deadzone_mask_for_pair(details, keep_fraction=keep_fraction)
            else:
                raise ValueError(f"unknown mode {mode!r}")
            new_details, pair_total, _already = apply_deadzone_to_pair_details(details, mask)
            new_pyramid[f"{frame_key}_details"] = new_details
            n_total += pair_total
            n_zeroed += int(mask.sum())
        new_pyramids.append(new_pyramid)
    new_blob = pack_pair_pyramids_to_wavelet_blob(new_pyramids)
    new_archive = splice_wavelet_blob_into_archive(archive_bytes, new_blob)
    return new_archive, n_total, n_zeroed


def _frontier_cpu_anchor() -> float:
    try:
        ptr = REPO_ROOT / ".omx" / "state" / "canonical_frontier_pointer.json"
        if ptr.is_file():
            data = json.loads(ptr.read_text())
            cpu = data.get("our_local_frontier_contest_cpu") or {}
            v = cpu.get("score") if isinstance(cpu, dict) else None
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
    except Exception:  # pragma: no cover
        pass
    return FRONTIER_CPU_ANCHOR


def _provenance_for(result_path: Path) -> dict:
    from tac.provenance import (
        build_provenance_for_macos_cpu_advisory,
        provenance_to_dict,
    )

    sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    try:
        source_path = str(result_path.relative_to(REPO_ROOT))
    except ValueError:
        source_path = str(result_path)
    prov = build_provenance_for_macos_cpu_advisory(archive_sha256=sha, source_path=source_path)
    return provenance_to_dict(prov)


def run_rate_attack(
    *,
    video_path: str,
    num_pairs: int,
    eval_h: int,
    eval_w: int,
    keep_fractions: list[float],
    pose_null_fraction: float,
    seg_tau: float,
    out_dir: Path,
) -> dict:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[z8-attack] build baseline: {num_pairs} pairs @ ({eval_h},{eval_w})", flush=True)
    binding, f0, f1, baseline_archive = _build_baseline(
        video_path=video_path, num_pairs=num_pairs, eval_h=eval_h, eval_w=eval_w
    )
    gt_pairs = _decode_gt_pairs(video_path, num_pairs)
    print("[z8-attack] loading DistortionNet (real SegNet+PoseNet)", flush=True)
    dn = _real_distortion_net()

    baseline = _measure_archive(
        baseline_archive, num_pairs=num_pairs, gt_pairs=gt_pairs, dn=dn, out_dir=out_dir / "baseline"
    )
    print(
        f"[z8-attack] BASELINE: rate={baseline['rate']:.4f} d_seg={baseline['d_seg']:.6f} "
        f"d_pose={baseline['d_pose']:.4f} score={baseline['contest_score']:.4f} "
        f"[seg={baseline['seg_term']:.3f} pose={baseline['pose_term']:.3f} rate={baseline['rate_term']:.3f}]",
        flush=True,
    )

    print("[z8-attack] computing per-pair joint P18/P19 surfaces (real scorers)", flush=True)
    from tac.scorer import load_differentiable_scorers

    pose_scorer, seg_scorer = load_differentiable_scorers(str(UPSTREAM), device="cpu")
    per_pair_joint = _per_pair_joint_masks(
        binding,
        f0,
        f1,
        pose_scorer,
        seg_scorer,
        num_pairs=num_pairs,
        eval_h=eval_h,
        eval_w=eval_w,
        pose_null_fraction=pose_null_fraction,
        seg_tau=seg_tau,
    )

    sweep: list[dict] = []
    for keep in keep_fractions:
        joint_archive, n_total, n_zeroed_j = _apply_attack_to_blob(
            baseline_archive, keep_fraction=keep, mode="joint", per_pair_joint=per_pair_joint
        )
        joint_m = _measure_archive(
            joint_archive, num_pairs=num_pairs, gt_pairs=gt_pairs, dn=dn,
            out_dir=out_dir / f"joint_keep{keep:.2f}",
        )
        seg_archive, _t, n_zeroed_s = _apply_attack_to_blob(
            baseline_archive, keep_fraction=keep, mode="segnet_only", per_pair_joint=per_pair_joint
        )
        seg_m = _measure_archive(
            seg_archive, num_pairs=num_pairs, gt_pairs=gt_pairs, dn=dn,
            out_dir=out_dir / f"segonly_keep{keep:.2f}",
        )
        mag_archive, _t2, n_zeroed_m = _apply_attack_to_blob(
            baseline_archive, keep_fraction=keep, mode="magnitude", per_pair_joint=None
        )
        mag_m = _measure_archive(
            mag_archive, num_pairs=num_pairs, gt_pairs=gt_pairs, dn=dn,
            out_dir=out_dir / f"magnitude_keep{keep:.2f}",
        )
        row = {
            "keep_fraction": keep,
            "coeffs_total": n_total,
            "joint": {**joint_m, "coeffs_zeroed": n_zeroed_j, "deadzone_fraction": n_zeroed_j / max(n_total, 1)},
            "segnet_only": {**seg_m, "coeffs_zeroed": n_zeroed_s, "deadzone_fraction": n_zeroed_s / max(n_total, 1)},
            "magnitude": {**mag_m, "coeffs_zeroed": n_zeroed_m, "deadzone_fraction": n_zeroed_m / max(n_total, 1)},
            "pose_preserved_by_joint_vs_segnet_only": float(seg_m["d_pose"] - joint_m["d_pose"]),
            "joint_score_minus_segnet_only_score": float(joint_m["contest_score"] - seg_m["contest_score"]),
            "joint_score_minus_magnitude_score": float(joint_m["contest_score"] - mag_m["contest_score"]),
        }
        sweep.append(row)
        print(
            f"[z8-attack] keep={keep:.2f}  "
            f"JOINT: rate={joint_m['rate']:.4f} d_seg={joint_m['d_seg']:.6f} d_pose={joint_m['d_pose']:.4f} score={joint_m['contest_score']:.4f}  ||  "
            f"SEGONLY(P18): rate={seg_m['rate']:.4f} d_seg={seg_m['d_seg']:.6f} d_pose={seg_m['d_pose']:.4f} score={seg_m['contest_score']:.4f}  ||  "
            f"MAGNITUDE: rate={mag_m['rate']:.4f} d_seg={mag_m['d_seg']:.6f} d_pose={mag_m['d_pose']:.4f} score={mag_m['contest_score']:.4f}  ||  "
            f"pose_preserved(joint vs segonly)={row['pose_preserved_by_joint_vs_segnet_only']:+.4f}",
            flush=True,
        )

    frontier = _frontier_cpu_anchor()
    result = {
        "schema": "z8_joint_p18_p19_deadzone_rate_attack_v1",
        "characterization_not_score_claim": True,  # Catalog #307
        "binding_axis": "rate",
        "joint_weight_formula": "w_i = 100*|dL_seg/dx_i| + 5/sqrt(10*d_pose)*||J_pose,i||_{Sigma^-1}",
        "attack": (
            "real SegNet boundary saliency (P18) + real PoseNet pixel-Jacobian (P19) "
            "pushed to coeff domain via analysis DWT (adjoint of orthonormal Daubechies); "
            "low-joint-weight pose-null detail coeffs zeroed; wavelet blob repacked; "
            "byte-closed archive re-measured on real DistortionNet"
        ),
        "video_path": video_path,
        "num_pairs": num_pairs,
        "eval_h": eval_h,
        "eval_w": eval_w,
        "pose_null_fraction": pose_null_fraction,
        "seg_boundary_tau": seg_tau,
        "baseline": baseline,
        "sweep": sweep,
        "frontier_cpu_anchor": frontier,
        "axis_tag": "[macOS-CPU advisory]",
        "evidence_grade": "macos_cpu_advisory",
        "score_claim": False,
        "promotable": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "is_contest_cpu_claim": False,
        "wall_clock_seconds": round(time.time() - t0, 1),
    }
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["provenance"] = _provenance_for(result_path)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[z8-attack] -> {result_path}  ({result['wall_clock_seconds']}s)", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Z8 joint P18/P19 dead-zone RATE ATTACK on the Mallat wavelet detail "
            "bands. macOS-CPU advisory, non-promotable, $0 local."
        )
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--num-pairs", type=int, default=600)
    parser.add_argument("--eval-h", type=int, default=96)
    parser.add_argument("--eval-w", type=int, default=128)
    parser.add_argument(
        "--keep-fractions",
        default="0.5,0.25,0.1,0.05",
        help="comma-separated keep fractions (top-weight coeffs kept; rest dead-zoned)",
    )
    parser.add_argument("--pose-null-fraction", type=float, default=0.05)
    parser.add_argument("--seg-tau", type=float, default=1.0)
    parser.add_argument(
        "--out-dir",
        default=str(
            REPO_ROOT / "experiments" / "results" / "z8_joint_p18_p19_deadzone_rate_attack"
        ),
    )
    args = parser.parse_args(argv)
    keep_fractions = [float(x) for x in str(args.keep_fractions).split(",") if x.strip()]
    run_rate_attack(
        video_path=args.video,
        num_pairs=int(args.num_pairs),
        eval_h=int(args.eval_h),
        eval_w=int(args.eval_w),
        keep_fractions=keep_fractions,
        pose_null_fraction=float(args.pose_null_fraction),
        seg_tau=float(args.seg_tau),
        out_dir=Path(args.out_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
