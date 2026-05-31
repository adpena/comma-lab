# SPDX-License-Identifier: MIT
"""WAVE-1F — Z8 wavelet+WZ ARCHIVE-path render-faithfulness re-measure.

The operator diagnosed a coefficient-domain ``[0, 1]`` clamp in the Wyner-Ziv
top-LL projection that saturated the projection and erased WZ state
dependence. Codex landed the primary fix (removed the codec-domain clamp from
``runtime_payload_bridge._state_to_top_ll_delta`` / the top-LL projection; final
pixel clipping stays at the RGB/raw write boundary) and proved WZ state
*DEPENDENCE* (mutate WZ payload -> frame-1 pixels move). Codex did NOT answer
the open empirical question: is the post-fix render *FAITHFUL* (reconstructs in
the GT range with a real DistortionNet ``d_seg`` meaningfully below chance)?

This tool measures the **contest-archive path** — NOT the MLX HNeRV-decoder
``_full_main`` argmax path the WAVE-1E gumbel-vs-argmax audit already found
collapsed. It exercises exactly the bytes inflate consumes:

    load real upstream/videos/0.mkv pairs
      -> build_z8hpc1_archive_bytes_from_canonical_quadruple   (M5 Mallat
         decompose + M6 Wyner-Ziv encode + M4 deterministic step)
      -> projected_pair_pyramids_from_archive_bytes            (decode WZ top
         states + project into frame-1 top-LL, post-clamp-fix codec domain)
      -> reconstruct_pair_rgb_from_pyramid                     (inverse wavelet
         recompose + FINAL-PIXEL clip @ canonical_quadruple_binding:1492)

Then measures, with the SAME canonical helpers as the gumbel audit:

  * recon_mean / recon_std vs GT (GT mean ~= 21.5 uint8, std ~= 21.1):
    FAITHFUL (in GT range) or STILL SATURATED?
  * real DistortionNet ``d_seg`` (per-pixel SegNet argmax-flip rate;
    chance ~= 0.505): meaningfully below chance or still ~= chance?
  * WZ state-dependence magnitude: mutate the WZ payload via the canonical
    ``build_wyner_ziv_payload_mutation_receiver_proof`` and report
    ``frame_1_max_abs_delta`` (does WZ state drive output, and by how much).

ALL outputs are ``[macOS-MLX research-signal]`` / ``[macOS-CPU advisory]``,
non-promotable, with canonical Provenance. $0 local, MLX-first, NO cloud.
Per Catalog #307 the verdict stays IMPLEMENTATION-LEVEL not paradigm-kill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "upstream"
DEFAULT_VIDEO = str(UPSTREAM / "videos" / "0.mkv")


# ---------------------------------------------------------------------------
# Canonical archive-path reconstruction (the bytes inflate consumes)
# ---------------------------------------------------------------------------


def _canonical_cfg(*, num_pairs: int, eval_h: int, eval_w: int) -> Any:
    """Canonical Z8HPC1 inflate config (mirrors inflate.py + the canonical
    archive-consumption test ``_canonical_cfg``)."""

    return SimpleNamespace(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        num_pairs=num_pairs,
        deterministic_state_dim=16,  # M9 canonical default
        ego_motion_dim=6,
        eval_size=(eval_h, eval_w),
    )


def _build_archive_from_real_frames(
    *, video_path: str, num_pairs: int, eval_h: int, eval_w: int
) -> bytes:
    """Build a Z8HPC1 archive from real upstream frames (codec-domain bytes)."""

    from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
        build_canonical_quadruple_binding_from_z8_config,
        build_z8hpc1_archive_bytes_from_canonical_quadruple,
        load_real_video_pair_targets_numpy,
    )

    cfg = _canonical_cfg(num_pairs=num_pairs, eval_h=eval_h, eval_w=eval_w)
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)
    f0, f1 = load_real_video_pair_targets_numpy(
        video_path,
        num_pairs=num_pairs,
        output_height=eval_h,
        output_width=eval_w,
    )
    return build_z8hpc1_archive_bytes_from_canonical_quadruple(binding, f0, f1)


def _reconstruct_archive_pairs(archive_bytes: bytes):
    """Reconstruct every trained pair through the WZ-projected receiver path.

    Returns ``recon`` as ``(num_pairs, 2, H, W, 3)`` numpy float32 in [0, 1],
    exactly the pixels inflate writes (before bicubic upsample + uint8 cast).
    """

    import numpy as np

    from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
        reconstruct_pair_rgb_from_pyramid,
    )
    from tac.substrates.z8_hierarchical_predictive_coding.runtime_payload_bridge import (
        projected_pair_pyramids_from_archive_bytes,
    )

    binding, pair_pyramids, stats = projected_pair_pyramids_from_archive_bytes(
        archive_bytes
    )
    recon: list[Any] = []
    for pyramid in pair_pyramids:
        # NCHW (1,3,H,W) in [0,1] -> HWC
        r0, r1 = reconstruct_pair_rgb_from_pyramid(binding, pyramid)
        r0_hwc = np.transpose(r0[0], (1, 2, 0))
        r1_hwc = np.transpose(r1[0], (1, 2, 0))
        recon.append(np.stack([r0_hwc, r1_hwc], axis=0))
    return np.stack(recon, axis=0).astype(np.float32), stats


# ---------------------------------------------------------------------------
# Canonical scorer + faithfulness helpers (reused verbatim from WAVE-1E)
# ---------------------------------------------------------------------------


def _decode_gt_pairs(video_path: str, num_pairs: int):
    import numpy as np

    from tac.data import decode_video

    frames = decode_video(
        video_path, target_h=384, target_w=512, max_frames=2 * num_pairs
    )
    if len(frames) < 2 * num_pairs:
        raise RuntimeError(f"decoded {len(frames)} frames; need {2 * num_pairs}")
    gt = np.stack([f.numpy() for f in frames[: 2 * num_pairs]], axis=0)
    return gt.reshape(num_pairs, 2, 384, 512, 3).astype(np.float32)


def _resize_recon_to_scorer_grid(recon_unit, *, num_pairs: int):
    """Bicubic-resize archive-resolution recon (float [0,1]) to 384x512 uint8.

    Mirrors the inflate-side upsample-to-contest-resolution path, then scales
    to uint8 so the recon and GT live in the same [0, 255] space the
    DistortionNet ``compute_distortion`` consumes (GT comes from decode_video
    in uint8-equivalent float).
    """

    import numpy as np
    import torch
    import torch.nn.functional as F

    # recon_unit: (P, 2, H, W, 3) float [0,1] -> (P*2, 3, H, W)
    flat = recon_unit.reshape(num_pairs * 2, recon_unit.shape[2], recon_unit.shape[3], 3)
    t = torch.from_numpy(np.transpose(flat, (0, 3, 1, 2)).copy())
    up = F.interpolate(t, size=(384, 512), mode="bicubic", align_corners=False)
    up = up.clamp(0.0, 1.0) * 255.0
    up_np = up.numpy().astype(np.float32)
    # back to (P, 2, 384, 512, 3)
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


def _measure_real(dn, gt_pairs, recon_pairs, *, batch: int = 32) -> dict:
    import numpy as np
    import torch

    P = gt_pairs.shape[0]
    d_seg_all: list[float] = []
    d_pose_all: list[float] = []
    gt_t = torch.from_numpy(gt_pairs)
    rec_t = torch.from_numpy(recon_pairs)
    for s in range(0, P, batch):
        e = min(s + batch, P)
        with torch.inference_mode():
            d_pose, d_seg = dn.compute_distortion(gt_t[s:e], rec_t[s:e])
        d_pose_all.extend([float(x) for x in d_pose.tolist()])
        d_seg_all.extend([float(x) for x in d_seg.tolist()])
    d_seg_arr = np.asarray(d_seg_all, dtype=np.float64)
    d_pose_arr = np.asarray(d_pose_all, dtype=np.float64)
    return {
        "mean_d_seg": float(d_seg_arr.mean()),
        "mean_d_pose": float(d_pose_arr.mean()),
        "max_d_seg": float(d_seg_arr.max()),
        "min_d_seg": float(d_seg_arr.min()),
        "n_pairs": int(P),
    }


def _render_faithfulness(recon_pairs, gt_pairs) -> dict:
    """COLLAPSED if std < 10% GT std OR mean > 3x / < 1/3 GT mean."""
    import numpy as np

    gt_mean = float(np.mean(gt_pairs))
    gt_std = float(np.std(gt_pairs))
    recon_mean = float(np.mean(recon_pairs))
    recon_std = float(np.std(recon_pairs))
    collapsed_const = recon_std < 0.10 * gt_std
    collapsed_sat = (recon_mean > 3.0 * gt_mean) or (recon_mean < gt_mean / 3.0)
    collapsed = bool(collapsed_const or collapsed_sat)
    return {
        "gt_mean": gt_mean,
        "gt_std": gt_std,
        "recon_mean": recon_mean,
        "recon_std": recon_std,
        "collapsed_near_constant": bool(collapsed_const),
        "collapsed_saturated": bool(collapsed_sat),
        "verdict": "COLLAPSED" if collapsed else "FAITHFUL",
    }


def _wz_state_dependence(archive_bytes: bytes) -> dict:
    """Measure WZ state-dependence magnitude via the canonical mutation proof."""

    from tac.substrates.z8_hierarchical_predictive_coding.runtime_payload_bridge import (
        build_wyner_ziv_payload_mutation_receiver_proof,
    )

    proof = build_wyner_ziv_payload_mutation_receiver_proof(archive_bytes)
    return {
        "frame_0_max_abs_delta": float(proof.get("frame_0_max_abs_delta", 0.0)),
        "frame_1_max_abs_delta": float(proof.get("frame_1_max_abs_delta", 0.0)),
        "wz_pixel_consumption_proven": bool(
            proof.get("wyner_ziv_top_state_pixel_consumption_proven")
        ),
    }


# ---------------------------------------------------------------------------
# Provenance + main
# ---------------------------------------------------------------------------


def _provenance_for(result_path: Path) -> dict:
    from tac.provenance import (
        build_provenance_for_macos_mlx_research_signal,
        provenance_to_dict,
    )

    sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    prov = build_provenance_for_macos_mlx_research_signal(
        artifact_sha256=sha,
        source_path=str(result_path.relative_to(REPO_ROOT)),
    )
    return provenance_to_dict(prov)


def run_remeasure(
    *, video_path: str, num_pairs: int, eval_h: int, eval_w: int, out_dir: Path
) -> dict:
    import numpy as np

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[z8-faithful] build archive: {num_pairs} pairs @ ({eval_h},{eval_w}) "
        f"from {video_path}",
        flush=True,
    )
    archive_bytes = _build_archive_from_real_frames(
        video_path=video_path,
        num_pairs=num_pairs,
        eval_h=eval_h,
        eval_w=eval_w,
    )
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    print(
        f"[z8-faithful] archive {len(archive_bytes)}B sha {archive_sha[:12]}",
        flush=True,
    )

    recon_unit, proj_stats = _reconstruct_archive_pairs(archive_bytes)
    print(
        f"[z8-faithful] reconstructed {recon_unit.shape[0]} pairs; "
        f"WZ-projected-changed={proj_stats.get('projected_pair_changed_count')}",
        flush=True,
    )

    gt_pairs = _decode_gt_pairs(video_path, num_pairs)
    recon_scorer = _resize_recon_to_scorer_grid(recon_unit, num_pairs=num_pairs)

    # Faithfulness in the scorer-grid uint8 space (apples-to-apples with GT).
    faith = _render_faithfulness(recon_scorer, gt_pairs)
    print(
        f"[z8-faithful] recon_mean={faith['recon_mean']:.2f} "
        f"recon_std={faith['recon_std']:.2f} vs GT "
        f"mean={faith['gt_mean']:.2f} std={faith['gt_std']:.2f} "
        f"=> {faith['verdict']}",
        flush=True,
    )

    print("[z8-faithful] loading DistortionNet (real SegNet+PoseNet)", flush=True)
    dn = _real_distortion_net()
    metrics = _measure_real(dn, gt_pairs, recon_scorer)
    # chance d_seg baseline = identity vs a random/uniform recon (same as WAVE-1E)
    rng = np.random.default_rng(0)
    chance_recon = rng.uniform(0, 255, size=recon_scorer.shape).astype(np.float32)
    chance = _measure_real(dn, gt_pairs, chance_recon)["mean_d_seg"]
    identity = _measure_real(dn, gt_pairs, gt_pairs)
    print(
        f"[z8-faithful] real d_seg={metrics['mean_d_seg']:.6f} "
        f"d_pose={metrics['mean_d_pose']:.2f} | chance(random)={chance:.6f} "
        f"identity(GT,GT)={identity['mean_d_seg']:.6f}",
        flush=True,
    )

    wz = _wz_state_dependence(archive_bytes)
    print(
        f"[z8-faithful] WZ state-dependence frame1_max_abs_delta="
        f"{wz['frame_1_max_abs_delta']:.6g} frame0={wz['frame_0_max_abs_delta']:.6g}",
        flush=True,
    )

    # Verdict: FAITHFUL requires (a) recon in GT range AND (b) d_seg
    # meaningfully below chance (>= 0.02 absolute margin below the random
    # chance baseline) AND (c) WZ state actually drives output.
    d_seg = metrics["mean_d_seg"]
    margin_below_chance = float(chance - d_seg)
    recon_in_range = faith["verdict"] == "FAITHFUL"
    seg_below_chance = margin_below_chance >= 0.02
    wz_drives = wz["frame_1_max_abs_delta"] > 0.0
    faithful = bool(recon_in_range and seg_below_chance)
    if faithful:
        verdict = "FAITHFUL_CONTEST_VALID_UNLOCK"
    elif wz_drives and not recon_in_range:
        verdict = "STILL_COLLAPSED_RECON_SATURATED"
    elif recon_in_range and not seg_below_chance:
        verdict = "STILL_COLLAPSED_DSEG_AT_CHANCE"
    else:
        verdict = "STILL_COLLAPSED"

    result = {
        "schema": "z8_top_ll_clamp_fix_render_faithfulness_remeasure_v1",
        "render_path": (
            "build_z8hpc1_archive_bytes_from_canonical_quadruple"
            "->projected_pair_pyramids_from_archive_bytes"
            "->reconstruct_pair_rgb_from_pyramid (WAVELET+WZ ARCHIVE PATH)"
        ),
        "video_path": video_path,
        "num_pairs": num_pairs,
        "eval_h": eval_h,
        "eval_w": eval_w,
        "archive_bytes": len(archive_bytes),
        "archive_sha256": archive_sha,
        "wz_projected_pair_changed_count": int(
            proj_stats.get("projected_pair_changed_count") or 0
        ),
        "render_faithfulness": faith,
        "distortion_net": metrics,
        "chance_d_seg_random_recon": chance,
        "identity_d_seg_gt_vs_gt": identity["mean_d_seg"],
        "margin_below_chance": margin_below_chance,
        "wz_state_dependence": wz,
        "faithful": faithful,
        "verdict": verdict,
        "axis_tag": "[macOS-MLX research-signal]",
        "evidence_grade": "macos_mlx_research_signal",
        "score_claim": False,
        "promotable": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
    }

    result_path = out_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["provenance"] = _provenance_for(result_path)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[z8-faithful] VERDICT={verdict} -> {result_path}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Z8 wavelet+WZ ARCHIVE-path render-faithfulness re-measure after "
            "the top-LL clamp fix (WAVE-1F)."
        )
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--num-pairs", type=int, default=64)
    parser.add_argument("--eval-h", type=int, default=96)
    parser.add_argument("--eval-w", type=int, default=128)
    parser.add_argument(
        "--out-dir",
        default=str(
            REPO_ROOT
            / "experiments"
            / "results"
            / "z8_top_ll_clamp_fix_render_faithfulness_remeasure"
        ),
    )
    args = parser.parse_args(argv)
    run_remeasure(
        video_path=args.video,
        num_pairs=int(args.num_pairs),
        eval_h=int(args.eval_h),
        eval_w=int(args.eval_w),
        out_dir=Path(args.out_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
