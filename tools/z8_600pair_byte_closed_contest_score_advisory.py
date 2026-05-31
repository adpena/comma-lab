# SPDX-License-Identifier: MIT
"""WAVE-1G — Z8 600-pair byte-closed contest-score advisory baseline.

Sister Wave-1F (commit ``b1b6d2270``) confirmed the Z8 wavelet+WZ ARCHIVE-path
render is FAITHFUL post-clamp-fix at 64-pair scale (recon 23.82/19.32 vs GT
24.01/19.57; real d_seg 0.0129 = 39x below chance; WZ drives frame-1). That was
a RENDER-level faithfulness probe, NOT a byte-closed contest score: it omitted
(a) the full 600-pair scale and (b) the rate term ``25 * archive.zip / N``.

This tool answers the OPEN question: what is the REAL byte-closed CONTEST score
of the Z8 wavelet+WZ codec at full 600-pair scale with the actual archive rate
included? It is a CHARACTERIZATION (Catalog #307), NOT a score claim — the
faithful render is real; the question is whether the byte-closed contest score
is competitive or far from the canonical frontier 0.192 [contest-CPU].

The canonical contest score (verified ``upstream/evaluate.py:92``):

    score = 100 * segnet_dist + sqrt(10 * posenet_dist) + 25 * rate
    rate  = archive.zip_size / uncompressed_size   (N = 37,545,489 bytes)

Pipeline (the EXACT bytes the contest inflate consumes):

    load real upstream/videos/0.mkv pairs (num_pairs at eval res)
      -> build_z8hpc1_archive_bytes_from_canonical_quadruple  (Mallat decompose
         + Wyner-Ziv encode + deterministic step; wavelet_blob brotli q=11)
      -> export_z8hpc1_archive_bytes                          (canonical byte-
         closed archive.zip: DEFLATE(0.bin) + inflate runtime + src/)
      -> projected_pair_pyramids_from_archive_bytes           (WZ projection)
      -> reconstruct_pair_rgb_from_pyramid                    (inverse wavelet
         + final-pixel clip)

then measures real DistortionNet d_seg / d_pose on all pairs + computes the
canonical contest score with the rate term from the BYTE-CLOSED archive.zip.

ALL outputs are ``[macOS-CPU advisory]`` / ``[macOS-MLX research-signal]``,
non-promotable, with canonical Provenance, per Catalog #192/#341/#127/#323.
Local macOS-CPU is NOT 1:1 contest hardware -> this is explicitly NOT a
[contest-CPU] claim. $0 local, NO cloud. The paired CUDA+CPU ratification
(operator-funded <=$20) is op-routable, NOT fired.
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

# Canonical contest denominator N = sum of upstream/videos/* file sizes.
# Verified == 37,545,489 (CLAUDE.md "ARCHIVE MEASUREMENT" + upstream/evaluate.py
# ``uncompressed_size``). Computed live so the gate is self-checking.
CANONICAL_N_BYTES = 37_545_489

# Canonical frontier [contest-CPU] (pointer-only per CLAUDE.md "Frontier scores
# are pointer-only"). Read from the canonical pointer at runtime; the literal
# here is the documented anchor for the gap computation banner only.
FRONTIER_CPU_ANCHOR = 0.192  # HISTORICAL_SCORE_LITERAL_OK:gap_banner_only_read_from_pointer_at_runtime


# ---------------------------------------------------------------------------
# Canonical archive build + byte-closure
# ---------------------------------------------------------------------------


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


def _build_archive_bytes(*, video_path: str, num_pairs: int, eval_h: int, eval_w: int) -> bytes:
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
    return build_z8hpc1_archive_bytes_from_canonical_quadruple(binding, f0, f1)


def _byte_close_archive_zip(archive_bytes: bytes, out_dir: Path) -> tuple[int, str]:
    """Export the canonical contest-shaped archive.zip and return its on-disk
    size + sha256 (the EXACT byte-closure ``upstream/evaluate.py`` measures).
    """

    from tac.substrates.z8_hierarchical_predictive_coding.archive_candidate import (
        export_z8hpc1_archive_bytes,
    )

    pkg_dir = out_dir / "byte_closed_archive"
    archive_zip_path, _archive_sha, _bin_len = export_z8hpc1_archive_bytes(
        archive_bytes,
        pkg_dir,
        repo_root=REPO_ROOT,
        emit_archive_bound_candidate_package=False,
        emit_byte_mutation_proof=False,
        retain_receiver_proof_output=False,
    )
    zip_path = Path(archive_zip_path)
    zip_size = int(zip_path.stat().st_size)
    zip_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    return zip_size, zip_sha


def _reconstruct_archive_pairs(archive_bytes: bytes):
    import numpy as np

    from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
        reconstruct_pair_rgb_from_pyramid,
    )
    from tac.substrates.z8_hierarchical_predictive_coding.runtime_payload_bridge import (
        projected_pair_pyramids_from_archive_bytes,
    )

    binding, pair_pyramids, stats = projected_pair_pyramids_from_archive_bytes(archive_bytes)
    recon: list[Any] = []
    for pyramid in pair_pyramids:
        r0, r1 = reconstruct_pair_rgb_from_pyramid(binding, pyramid)
        r0_hwc = np.transpose(r0[0], (1, 2, 0))
        r1_hwc = np.transpose(r1[0], (1, 2, 0))
        recon.append(np.stack([r0_hwc, r1_hwc], axis=0))
    return np.stack(recon, axis=0).astype(np.float32), stats


# ---------------------------------------------------------------------------
# Canonical scorer (reused verbatim from WAVE-1F)
# ---------------------------------------------------------------------------


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


def _measure_real(dn, gt_pairs, recon_pairs, *, batch: int = 32) -> dict:
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
        d_pose_all.extend([float(x) for x in d_pose.tolist()])
        d_seg_all.extend([float(x) for x in d_seg.tolist()])
    d_seg_arr = np.asarray(d_seg_all, dtype=np.float64)
    d_pose_arr = np.asarray(d_pose_all, dtype=np.float64)
    return {
        "mean_d_seg": float(d_seg_arr.mean()),
        "mean_d_pose": float(d_pose_arr.mean()),
        "max_d_seg": float(d_seg_arr.max()),
        "min_d_seg": float(d_seg_arr.min()),
        "max_d_pose": float(d_pose_arr.max()),
        "min_d_pose": float(d_pose_arr.min()),
        "n_pairs": int(n),
    }


# ---------------------------------------------------------------------------
# Contest score
# ---------------------------------------------------------------------------


def compute_contest_score(*, d_seg: float, d_pose: float, archive_zip_size: int, n_bytes: int) -> dict:
    """Canonical contest score per upstream/evaluate.py:92.

    ``score = 100 * d_seg + sqrt(10 * d_pose) + 25 * rate``;
    ``rate = archive_zip_size / n_bytes``.
    """

    rate = float(archive_zip_size) / float(n_bytes)
    seg_term = 100.0 * float(d_seg)
    pose_term = math.sqrt(10.0 * float(d_pose))
    rate_term = 25.0 * rate
    score = seg_term + pose_term + rate_term
    # Dominant-gap axis = the largest of the three terms.
    terms = {"seg": seg_term, "pose": pose_term, "rate": rate_term}
    dominant = max(terms, key=terms.get)
    return {
        "rate": rate,
        "seg_term_100_d_seg": seg_term,
        "pose_term_sqrt_10_d_pose": pose_term,
        "rate_term_25_rate": rate_term,
        "contest_score": score,
        "dominant_term": dominant,
        "term_breakdown": terms,
    }


def _frontier_cpu_anchor() -> float:
    """Read the canonical frontier [contest-CPU] from the pointer (pointer-only
    per CLAUDE.md "Frontier scores are pointer-only"). Falls back to the
    documented anchor if the pointer is unavailable.
    """

    try:
        ptr = REPO_ROOT / ".omx" / "state" / "canonical_frontier_pointer.json"
        if ptr.is_file():
            data = json.loads(ptr.read_text())
            cpu = data.get("our_local_frontier_contest_cpu") or {}
            v = cpu.get("score") if isinstance(cpu, dict) else None
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
    except Exception:  # pragma: no cover - pointer shape drift tolerant
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
        # out-of-repo (e.g. pytest tmp_path) -> use absolute path string.
        source_path = str(result_path)
    prov = build_provenance_for_macos_cpu_advisory(
        archive_sha256=sha,
        source_path=source_path,
    )
    return provenance_to_dict(prov)


def run_contest_score(
    *, video_path: str, num_pairs: int, eval_h: int, eval_w: int, out_dir: Path
) -> dict:
    import numpy as np

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(
        f"[z8-contest] build archive: {num_pairs} pairs @ ({eval_h},{eval_w})", flush=True
    )
    archive_bytes = _build_archive_bytes(
        video_path=video_path, num_pairs=num_pairs, eval_h=eval_h, eval_w=eval_w
    )
    bin_sha = hashlib.sha256(archive_bytes).hexdigest()
    print(
        f"[z8-contest] 0.bin {len(archive_bytes):,}B sha {bin_sha[:12]} "
        f"({round(time.time() - t0, 1)}s)",
        flush=True,
    )

    # BYTE-CLOSURE: the contest measures archive.zip, not raw 0.bin bytes.
    zip_size, zip_sha = _byte_close_archive_zip(archive_bytes, out_dir)
    rate = zip_size / CANONICAL_N_BYTES
    print(
        f"[z8-contest] BYTE-CLOSED archive.zip {zip_size:,}B sha {zip_sha[:12]} "
        f"=> rate {rate:.6f} (= zip/{CANONICAL_N_BYTES:,})",
        flush=True,
    )

    recon_unit, proj_stats = _reconstruct_archive_pairs(archive_bytes)
    print(
        f"[z8-contest] reconstructed {recon_unit.shape[0]} pairs "
        f"(WZ-projected-changed={proj_stats.get('projected_pair_changed_count')})",
        flush=True,
    )

    gt_pairs = _decode_gt_pairs(video_path, num_pairs)
    recon_scorer = _resize_recon_to_scorer_grid(recon_unit, num_pairs=num_pairs)
    gt_mean, gt_std = float(np.mean(gt_pairs)), float(np.std(gt_pairs))
    recon_mean, recon_std = float(np.mean(recon_scorer)), float(np.std(recon_scorer))
    print(
        f"[z8-contest] recon_mean={recon_mean:.2f} recon_std={recon_std:.2f} "
        f"vs GT mean={gt_mean:.2f} std={gt_std:.2f}",
        flush=True,
    )

    print("[z8-contest] loading DistortionNet (real SegNet+PoseNet)", flush=True)
    dn = _real_distortion_net()
    metrics = _measure_real(dn, gt_pairs, recon_scorer)
    print(
        f"[z8-contest] real d_seg={metrics['mean_d_seg']:.6f} "
        f"d_pose={metrics['mean_d_pose']:.4f} over {metrics['n_pairs']} pairs",
        flush=True,
    )

    score = compute_contest_score(
        d_seg=metrics["mean_d_seg"],
        d_pose=metrics["mean_d_pose"],
        archive_zip_size=zip_size,
        n_bytes=CANONICAL_N_BYTES,
    )
    frontier = _frontier_cpu_anchor()
    gap = float(score["contest_score"] - frontier)
    gap_ratio = float(score["contest_score"] / frontier) if frontier > 0 else float("inf")
    print(
        f"[z8-contest] CONTEST SCORE = {score['contest_score']:.4f}  "
        f"[seg={score['seg_term_100_d_seg']:.3f} "
        f"pose={score['pose_term_sqrt_10_d_pose']:.3f} "
        f"rate={score['rate_term_25_rate']:.3f}] "
        f"dominant={score['dominant_term']}",
        flush=True,
    )
    print(
        f"[z8-contest] vs frontier {frontier:.4f} [contest-CPU]: "
        f"gap +{gap:.4f} (= {gap_ratio:.1f}x frontier)",
        flush=True,
    )

    result = {
        "schema": "z8_600pair_byte_closed_contest_score_advisory_v1",
        "characterization_not_score_claim": True,  # Catalog #307
        "render_path": (
            "build_z8hpc1_archive_bytes_from_canonical_quadruple"
            "->export_z8hpc1_archive_bytes(byte-closed archive.zip)"
            "->projected_pair_pyramids_from_archive_bytes"
            "->reconstruct_pair_rgb_from_pyramid (WAVELET+WZ ARCHIVE PATH)"
        ),
        "video_path": video_path,
        "num_pairs": num_pairs,
        "eval_h": eval_h,
        "eval_w": eval_w,
        "bin_bytes": len(archive_bytes),
        "bin_sha256": bin_sha,
        "byte_closed_archive_zip_bytes": zip_size,
        "byte_closed_archive_zip_sha256": zip_sha,
        "canonical_n_bytes": CANONICAL_N_BYTES,
        "wz_projected_pair_changed_count": int(
            proj_stats.get("projected_pair_changed_count") or 0
        ),
        "render_faithfulness": {
            "gt_mean": gt_mean,
            "gt_std": gt_std,
            "recon_mean": recon_mean,
            "recon_std": recon_std,
        },
        "distortion_net": metrics,
        "contest_score_breakdown": score,
        "frontier_cpu_anchor": frontier,
        "gap_to_frontier_abs": gap,
        "gap_to_frontier_ratio": gap_ratio,
        "dominant_gap_axis": score["dominant_term"],
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
    print(f"[z8-contest] -> {result_path}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Z8 600-pair byte-closed contest-score advisory baseline (WAVE-1G). "
            "macOS-CPU advisory, non-promotable, $0 local."
        )
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--num-pairs", type=int, default=600)
    parser.add_argument("--eval-h", type=int, default=96)
    parser.add_argument("--eval-w", type=int, default=128)
    parser.add_argument(
        "--out-dir",
        default=str(
            REPO_ROOT / "experiments" / "results" / "z8_600pair_byte_closed_contest_score_advisory"
        ),
    )
    args = parser.parse_args(argv)
    run_contest_score(
        video_path=args.video,
        num_pairs=int(args.num_pairs),
        eval_h=int(args.eval_h),
        eval_w=int(args.eval_w),
        out_dir=Path(args.out_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
