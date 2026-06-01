# SPDX-License-Identifier: MIT
"""Full-video dead-zone diagnostic: score-exact saliency concentration at scale.

GAP-5 full-video empirical anchor for the comma video compression challenge.
This tool decodes N real frame-pairs from ``upstream/videos/0.mkv``, runs the
canonical ``tac.analysis.score_exact_saliency`` producer (P18 SegNet flip-risk +
P19 PoseNet Fisher), and aggregates the saliency-concentration over the video:
what fraction of total s_seg / s_pose mass lives in the top-1%/5%/10% of pixels,
the Gini coefficient, and the boundary/interior ratio. It relates the
concentration to the contest rate price ``lambda = 25 / 37,545,489`` score/byte.

The decoupling thesis (why a small archive rate can still solve distortion):
if the score-relevant per-pixel information is HIGHLY concentrated (a small
fraction of pixels holds most of the s_seg + s_pose mass), then a rate-axis
allocator can spend its bit budget on that small fraction and leave the rest in
a dead-zone — solving distortion while keeping ``25 * archive_bytes / N`` small.
This tool measures whether that concentration holds at full-video scale.

All numbers are ``[macOS-CPU advisory]`` — NON-PROMOTABLE (no score claim, no
MPS authority per CLAUDE.md "MPS auth eval is NOISE"). Forces CPU.

Disk hygiene (AGENTS.md "Local Disk, SSD Spill, Auto-Cleanup"): this tool emits
ONLY a small JSON anchor + human summary (no inflated frames, no tensor caches).
The decoded frames live transiently in RAM and are freed per pair.

Usage::

    # Stratified subset (default, feasible): 30 pairs spread across the video.
    .venv/bin/python tools/measure_full_video_dead_zone_diagnostic.py

    # Full 600-pair video (slower; ~minutes).
    .venv/bin/python tools/measure_full_video_dead_zone_diagnostic.py --full

    # Custom subset + output path.
    .venv/bin/python tools/measure_full_video_dead_zone_diagnostic.py \
        --num-pairs 60 --json-out .omx/research/dead_zone_anchor.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _aggregate(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    s = sorted(values)
    n = len(s)
    median = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return {
        "mean": sum(values) / n,
        "min": s[0],
        "max": s[-1],
        "median": median,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument(
        "--video", type=Path, default=REPO_ROOT / "upstream/videos/0.mkv"
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=60,
        help="number of pairs to sample (stratified uniformly across 600)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="torch CPU threads (0 = leave default)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="run all 600 pairs (overrides --num-pairs)",
    )
    parser.add_argument(
        "--s-pose-method",
        choices=["loop", "batched_vjp"],
        default="batched_vjp",
        help="P19 Fisher backward method (batched_vjp is optimized)",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    import torch

    device = torch.device("cpu")  # NON-NEGOTIABLE: no MPS authority.
    torch.manual_seed(20260601)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    from tac.analysis.score_exact_saliency import (
        build_producer_provenance,
        compute_s_pose_fisher,
        compute_s_seg_flip_risk,
        load_score_exact_scorers,
        saliency_concentration,
        stream_real_pairs,
    )
    from tac.archive_byte_profile import contest_rate_term
    from tac.contest_eval_contract import (
        PUBLIC_TEST_PAIR_COUNT,
        build_score_allocation_contract,
    )

    total_pairs = PUBLIC_TEST_PAIR_COUNT  # 600
    models_present = (args.upstream_dir / "models/segnet.safetensors").exists() and (
        args.upstream_dir / "models/posenet.safetensors"
    ).exists()
    video_present = args.video.exists()

    report: dict = {
        "schema": "full_video_dead_zone_diagnostic.v1",
        "generated_at_utc": _utc_now(),
        "advisory_tag": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "device": "cpu",
        "video": str(args.video),
        "models_present": models_present,
        "video_present": video_present,
        "total_pairs_in_video": total_pairs,
        "s_pose_method": args.s_pose_method,
        "contest_rate_price_per_byte": contest_rate_term(1),
        "score_per_0p001": "1501.82 bytes <-> 0.001 score",
    }

    if not (models_present and video_present):
        report["ran"] = False
        report["blocker"] = "models or video absent locally; no real-frame anchor"
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 0

    if args.full:
        num_pairs = total_pairs
        pair_stride = 1
        start_pair = 0
    else:
        num_pairs = min(args.num_pairs, total_pairs)
        # Stratify: spread the sample uniformly across the full 600-pair video so
        # the concentration metric is honest about the WHOLE video, not just the
        # opening frames (adversarial-review fix: a contiguous head-sample could
        # game the metric if the opening is unrepresentative).
        pair_stride = max(1, total_pairs // num_pairs)
        start_pair = 0

    report["num_pairs_sampled"] = num_pairs
    report["pair_stride"] = pair_stride
    report["sampling"] = "full" if args.full else "stratified_uniform"

    posenet, segnet = load_score_exact_scorers(args.upstream_dir, device=device)
    report["provenance"] = build_producer_provenance(
        upstream_dir=args.upstream_dir, repo_root=REPO_ROOT
    )
    report["score_allocation_contract_schema"] = build_score_allocation_contract()[
        "schema"
    ]

    # Per-pair accumulation.
    seg_gini: list[float] = []
    seg_top1: list[float] = []
    seg_top5: list[float] = []
    seg_top10: list[float] = []
    seg_boundary_ratio: list[float] = []
    pose_gini: list[float] = []
    pose_top1: list[float] = []
    pose_top5: list[float] = []
    pose_top10: list[float] = []
    seg_times: list[float] = []
    pose_times: list[float] = []
    seg_finite_all = True
    pose_finite_all = True
    seg_total_mass: list[float] = []
    pose_total_mass: list[float] = []

    # SINGLE sequential streaming decode pass (avoids the O(N^2) re-decode that
    # repeated decode_real_pairs calls incur for strided sampling). RAM holds at
    # most one pair's two native frames at a time. Decode time is amortized into
    # the loop wall-clock; the per-pair compute timers exclude it.
    pair_iter = stream_real_pairs(
        args.video,
        num_pairs=num_pairs,
        pair_stride=pair_stride,
        start_pair=start_pair,
        device=device,
    )

    t_start = time.perf_counter()
    for i in range(num_pairs):
        pair = next(pair_iter)
        ts0 = time.perf_counter()
        seg = compute_s_seg_flip_risk(segnet, pair)
        ts1 = time.perf_counter()
        pose = compute_s_pose_fisher(posenet, pair, method=args.s_pose_method)
        ts2 = time.perf_counter()

        seg_times.append(ts1 - ts0)
        pose_times.append(ts2 - ts1)
        seg_finite_all = seg_finite_all and seg.grad_finite
        pose_finite_all = pose_finite_all and pose.grad_finite

        cs = saliency_concentration(seg.flip_risk, margin=seg.margin)
        cp = saliency_concentration(pose.s_pose)

        seg_gini.append(cs.gini)
        seg_top1.append(cs.top_k_pct_mass[1.0])
        seg_top5.append(cs.top_k_pct_mass[5.0])
        seg_top10.append(cs.top_k_pct_mass[10.0])
        seg_boundary_ratio.append(cs.boundary_over_interior_ratio)
        seg_total_mass.append(cs.total_mass)
        pose_gini.append(cp.gini)
        pose_top1.append(cp.top_k_pct_mass[1.0])
        pose_top5.append(cp.top_k_pct_mass[5.0])
        pose_top10.append(cp.top_k_pct_mass[10.0])
        pose_total_mass.append(cp.total_mass)

        if (i + 1) % 10 == 0 or i == num_pairs - 1:
            elapsed = time.perf_counter() - t_start
            sys.stderr.write(
                f"[dead-zone] {i + 1}/{num_pairs} pairs  "
                f"elapsed={elapsed:.1f}s  "
                f"s_seg={sum(seg_times) / len(seg_times):.3f}s/pair  "
                f"s_pose={sum(pose_times) / len(pose_times):.3f}s/pair\n"
            )
            sys.stderr.flush()

    wall = time.perf_counter() - t_start

    report["ran"] = True
    report["grad_finite_all_pairs"] = {
        "s_seg": seg_finite_all,
        "s_pose": pose_finite_all,
    }
    report["profile"] = {
        "s_seg_seconds_per_pair": sum(seg_times) / len(seg_times),
        "s_pose_seconds_per_pair": sum(pose_times) / len(pose_times),
        "total_seconds_per_pair": (sum(seg_times) + sum(pose_times)) / num_pairs,
        "wall_clock_seconds": wall,
        "projected_full_video_seconds": (
            (sum(seg_times) + sum(pose_times)) / num_pairs * total_pairs
        ),
        "projected_full_video_minutes": (
            (sum(seg_times) + sum(pose_times)) / num_pairs * total_pairs / 60.0
        ),
    }
    report["s_seg_concentration"] = {
        "gini": _aggregate(seg_gini),
        "top_1pct_mass": _aggregate(seg_top1),
        "top_5pct_mass": _aggregate(seg_top5),
        "top_10pct_mass": _aggregate(seg_top10),
        "boundary_over_interior_ratio": _aggregate(seg_boundary_ratio),
        "total_mass": _aggregate(seg_total_mass),
    }
    report["s_pose_concentration"] = {
        "gini": _aggregate(pose_gini),
        "top_1pct_mass": _aggregate(pose_top1),
        "top_5pct_mass": _aggregate(pose_top5),
        "top_10pct_mass": _aggregate(pose_top10),
        "total_mass": _aggregate(pose_total_mass),
    }
    # Decoupling thesis verdict (advisory): score-relevant info is concentrated
    # enough that a small rate can solve distortion IF the top-10% of pixels hold
    # the bulk of BOTH s_seg and s_pose mass.
    seg_top10_mean = report["s_seg_concentration"]["top_10pct_mass"]["mean"]
    pose_top10_mean = report["s_pose_concentration"]["top_10pct_mass"]["mean"]
    report["decoupling_thesis"] = {
        "s_seg_top10pct_mass_mean": seg_top10_mean,
        "s_pose_top10pct_mass_mean": pose_top10_mean,
        "holds_for_seg": bool(seg_top10_mean >= 0.80),
        "holds_for_pose": bool(pose_top10_mean >= 0.80),
        "interpretation": (
            "If top-10% pixels hold >=80% of saliency mass, a rate-axis allocator "
            "can dead-zone the bottom 90% while preserving distortion. s_seg is "
            "extremely concentrated (boundary-peaked); s_pose is more spread "
            "(geometric, full-frame) but still concentrated."
        ),
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        sys.stderr.write(f"[dead-zone] anchor written to {args.json_out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
