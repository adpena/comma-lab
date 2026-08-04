#!/usr/bin/env python
"""q31 -- solve Road/Lane target corrections inside Q3 from the start.

This is the named successor to se2's project-after negative.  It uses the same
qo1 decoded frames, the same fixed n32 pair set, and the same Road/Lane target
denominator as se2, but optimizes only through frame-1 yuv6-null directions.

Axis: [macOS-CPU advisory / CPU Torch SegNet+PoseNet bounded n32].
score_claim=false; no archive build; no n600 scorer slot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO / "experiments", REPO / "upstream"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ddm_sq1_eta_seg_realization import (  # noqa: E402
    CAM_H,
    CAM_W,
    N_PAIRS_TOTAL,
    SEG_H,
    SEG_W,
    Scorer,
    decode_gt_frames,
    seq_len,
)
from ddm_sq1_pose_null_constrained_paint import (  # noqa: E402
    pose_null_projector,
    project_null,
    snap_band_to_blocks,
    yuv6_shift,
)
from ddm_sq1_stage_decomposition_and_solved_paint import (  # noqa: E402
    confusion,
    realize_scorer_paint_to_camera,
    resize_to_scorer,
)


PAIR_SET = [
    31,
    43,
    62,
    82,
    94,
    118,
    147,
    165,
    167,
    182,
    185,
    200,
    237,
    241,
    247,
    259,
    272,
    286,
    288,
    292,
    296,
    306,
    327,
    382,
    390,
    419,
    473,
    488,
    525,
    555,
    560,
    581,
]

ROAD = 0
LANE = 1
RATE_DENOMINATOR_BYTES = 37_545_489
BASELINE_S = 0.7539807296911207
BASELINE_BYTES = 357_836
BASELINE_D_SEG = 0.00431179
BASELINE_D_POSE = 0.00071459
QO1_ARCHIVE_SHA256 = "d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a"
ED1_BREAK_EVEN_SURVIVAL = 0.6964303814
SE2_R0_DELTA32_TARGET_SURVIVAL = 0.263238
SE2_R0_DELTA32_GLOBAL_NET_REDUCTION = 1563
SE2_PROJECT_AFTER_Q3_TARGET_SURVIVAL = 0.017007
SE2_PROJECT_AFTER_Q3_GLOBAL_NET_REDUCTION = 60
SQ1_UNCONSTRAINED_ETA_25 = 0.7895095948827292
SQ2_UNCONSTRAINED_ETA_50 = 0.8620042643923241


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception as exc:  # pragma: no cover - provenance fallback only
        return f"UNKNOWN:{exc}"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=1, default=jsonable)
        fh.write("\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, default=jsonable, sort_keys=True))
        fh.write("\n")


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_raw_pairs(path: Path) -> np.memmap:
    frame_bytes = CAM_H * CAM_W * 3
    expected_frames = N_PAIRS_TOTAL * seq_len
    if path.stat().st_size != frame_bytes * expected_frames:
        raise RuntimeError(f"raw size drift for {path}")
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(expected_frames, CAM_H, CAM_W, 3))


def road_lane_target(gt: np.ndarray, current: np.ndarray) -> np.ndarray:
    return ((gt == ROAD) & (current == LANE)) | ((gt == LANE) & (current == ROAD))


def target_counts(gt_cache: np.ndarray, cur_cache: np.ndarray) -> np.ndarray:
    counts = np.zeros((N_PAIRS_TOTAL,), dtype=np.int64)
    for pair in range(N_PAIRS_TOTAL):
        counts[pair] = int(road_lane_target(np.asarray(gt_cache[pair]), np.asarray(cur_cache[pair])).sum())
    return counts


def scorer_tensor_from_hwc(paint_hwc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(paint_hwc)).permute(2, 0, 1)[None].float()


def score_camera_pair(
    sc: Scorer,
    pose_gt: Any,
    lgt: np.ndarray,
    flips0: np.ndarray,
    target_mask: np.ndarray,
    dec_f0: np.ndarray,
    cam_f1: np.ndarray,
) -> dict[str, Any]:
    pair = np.stack([dec_f0, cam_f1])
    lam = sc.seg_argmax(pair)
    after = lam != lgt
    target = target_mask.astype(bool)
    eligible = target & flips0
    corrected = eligible & (lam == lgt)
    return {
        "flips_after": int(after.sum()),
        "global_net_flip_reduction": int(flips0.sum() - after.sum()),
        "fixed_global": int((flips0 & ~after).sum()),
        "introduced_global": int((~flips0 & after).sum()),
        "target_cells": int(target.sum()),
        "eligible_target_cells": int(eligible.sum()),
        "corrected_target_cells": int(corrected.sum()),
        "target_survival": float(corrected.sum() / max(1, eligible.sum())),
        "target_still_wrong_cells": int((eligible & (lam != lgt)).sum()),
        "collateral_new_wrong_non_target": int(((~target) & (~flips0) & after).sum()),
        "C_after": confusion(lgt, lam).tolist(),
        "d_pose_after": sc.d_pose(pose_gt, sc.pose_out(pair)),
    }


def project_param_(raw: torch.Tensor, mask: torch.Tensor, projector: torch.Tensor) -> None:
    with torch.no_grad():
        raw.copy_(project_null(raw, projector) * mask)


def solve_q3_constrained(
    segnet: torch.nn.Module,
    dec_f1: np.ndarray,
    gt_f1: np.ndarray,
    target_mask: np.ndarray,
    lgt: np.ndarray,
    projector: torch.Tensor,
    *,
    steps: int,
    lr: float,
    eval_every: int,
    convergence_patience_evals: int,
    convergence_min_improvement: int,
    starts: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    if steps < 50:
        raise RuntimeError("q31 requires steps >= 50")
    base = resize_to_scorer(dec_f1)
    truth = resize_to_scorer(gt_f1)
    snapped_mask_np = snap_band_to_blocks(target_mask.astype(bool))
    mask = torch.from_numpy(snapped_mask_np)[None, None].float()
    target = torch.from_numpy(lgt.astype(np.int64))[None]
    best: tuple[int, int, str, np.ndarray] | None = None
    start_diagnostics: list[dict[str, Any]] = []

    with torch.enable_grad():
        for start_name in starts:
            if start_name == "dec":
                init = torch.zeros_like(base)
            elif start_name == "truth":
                init = project_null((truth - base) * mask, projector)
            else:
                raise RuntimeError(f"unknown start {start_name!r}")

            raw = init.clone().detach().requires_grad_(True)
            project_param_(raw, mask, projector)
            opt = torch.optim.Adam([raw], lr=lr)
            start_best_bad: int | None = None
            start_best_step: int | None = None
            evals_since_best = 0
            stop_reason = "iteration_cap_before_plateau"
            curve = []

            for step in range(steps + 1):
                delta = project_null(raw, projector) * mask
                preclip = base + delta
                cur = torch.clamp(preclip, 0.0, 255.0)
                if step % eval_every == 0 or step == steps:
                    clipped_channel_values = int(((preclip < 0.0) | (preclip > 255.0)).sum().item())
                    q = torch.round(cur).detach()
                    with torch.no_grad():
                        lam = segnet(q).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
                    flips_after = int((lam != lgt).sum())
                    target_surv_num = int(((target_mask.astype(bool)) & (lam == lgt)).sum())
                    target_surv_den = int(target_mask.sum())
                    changed = int(
                        (
                            q[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                            != torch.round(base)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        ).any(axis=2).sum()
                    )
                    curve.append(
                        {
                            "step": int(step),
                            "proxy_global_flips_after": flips_after,
                            "proxy_target_survival": float(target_surv_num / max(1, target_surv_den)),
                            "changed_scorer_pixels": changed,
                            "clipped_channel_values_pre_uint8": clipped_channel_values,
                        }
                    )
                    improved = (
                        start_best_bad is None
                        or start_best_bad - flips_after >= max(1, int(convergence_min_improvement))
                    )
                    if improved:
                        start_best_bad = flips_after
                        start_best_step = int(step)
                        evals_since_best = 0
                    else:
                        evals_since_best += 1
                    if best is None or flips_after < best[0]:
                        paint = q[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        best = (flips_after, int(step), start_name, paint)
                    if (
                        convergence_patience_evals > 0
                        and evals_since_best >= convergence_patience_evals
                        and step < steps
                    ):
                        stop_reason = "plateau_no_global_flip_improvement"
                        break
                if step == steps:
                    if start_best_step == steps:
                        stop_reason = "iteration_cap_best_at_cap"
                    else:
                        stop_reason = "iteration_cap_before_plateau"
                    break

                loss = torch.nn.functional.cross_entropy(segnet(cur), target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if raw.grad is not None:
                    raw.grad.copy_(project_null(raw.grad, projector) * mask)
                opt.step()
                project_param_(raw, mask, projector)

            start_diagnostics.append(
                {
                    "start": start_name,
                    "stop_reason": stop_reason,
                    "steps_run": int(curve[-1]["step"]) if curve else 0,
                    "best_step": start_best_step,
                    "best_proxy_global_flips_after": start_best_bad,
                    "curve": curve,
                }
            )

    if best is None:
        raise RuntimeError("q31 solve produced no evaluated iterate")
    selected = next(d for d in start_diagnostics if d["start"] == best[2])
    diagnostics = {
        "selected_start": best[2],
        "selected_best_step": int(best[1]),
        "selected_proxy_global_flips_after": int(best[0]),
        "selected_stop_reason": selected["stop_reason"],
        "selected_steps_run": selected["steps_run"],
        "starts": start_diagnostics,
        "snapped_target_cells": int(snapped_mask_np.sum()),
        "snap_tax_vs_target_cells": float(snapped_mask_np.sum() / max(1, target_mask.sum())),
    }
    return best[3], diagnostics


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_rows": 0}
    baseline_flips = int(sum(r["flips_before"] for r in rows))
    flips_after = int(sum(r["q3_constrained"]["flips_after"] for r in rows))
    target_cells = int(sum(r["target_cells"] for r in rows))
    corrected = int(sum(r["q3_constrained"]["corrected_target_cells"] for r in rows))
    net = baseline_flips - flips_after
    before_pose = np.array([r["d_pose_before"] for r in rows], dtype=np.float64)
    after_pose = np.array([r["q3_constrained"]["d_pose_after"] for r in rows], dtype=np.float64)
    stop_counts = Counter(r["solve"]["selected_stop_reason"] for r in rows)
    survival = corrected / max(1, target_cells)
    return {
        "n_rows": len(rows),
        "baseline_flips_subset": baseline_flips,
        "q3_constrained_flips_subset": flips_after,
        "q3_constrained_global_net_flip_reduction": net,
        "target_cells": target_cells,
        "corrected_target_cells": corrected,
        "q3_constrained_target_survival": survival,
        "target_survival_vs_ed1_break_even": survival / ED1_BREAK_EVEN_SURVIVAL,
        "clears_ed1_break_even": bool(survival >= ED1_BREAK_EVEN_SURVIVAL),
        "vs_se2": {
            "r0_delta32_target_survival": SE2_R0_DELTA32_TARGET_SURVIVAL,
            "r0_delta32_global_net_reduction": SE2_R0_DELTA32_GLOBAL_NET_REDUCTION,
            "project_after_q3_target_survival": SE2_PROJECT_AFTER_Q3_TARGET_SURVIVAL,
            "project_after_q3_global_net_reduction": SE2_PROJECT_AFTER_Q3_GLOBAL_NET_REDUCTION,
            "q31_over_se2_r0_delta32_survival": survival / SE2_R0_DELTA32_TARGET_SURVIVAL,
            "q31_over_se2_project_after_q3_survival": survival / SE2_PROJECT_AFTER_Q3_TARGET_SURVIVAL,
            "q31_over_se2_r0_delta32_global_net": (
                net / SE2_R0_DELTA32_GLOBAL_NET_REDUCTION
                if SE2_R0_DELTA32_GLOBAL_NET_REDUCTION
                else None
            ),
        },
        "vs_sq1_unconstrained_comparator": {
            "scope_note": "sq1/sq2 eta used the sq1 band denominator, not se2 Road/Lane targets",
            "sq1_eta_25": SQ1_UNCONSTRAINED_ETA_25,
            "sq2_eta_50": SQ2_UNCONSTRAINED_ETA_50,
            "q31_target_survival_over_sq1_eta_25": survival / SQ1_UNCONSTRAINED_ETA_25,
            "q31_target_survival_over_sq2_eta_50": survival / SQ2_UNCONSTRAINED_ETA_50,
        },
        "d_pose_before_mean": float(before_pose.mean()),
        "d_pose_after_mean": float(after_pose.mean()),
        "d_pose_delta_mean": float((after_pose - before_pose).mean()),
        "d_pose_ratio_vs_before": float(after_pose.mean() / before_pose.mean()) if before_pose.mean() else None,
        "d_pose_ratio_max_pair": float(max(r["q3_constrained"]["d_pose_ratio"] for r in rows)),
        "stop_reason_census": dict(sorted(stop_counts.items())),
        "cap_bound_rows": int(
            sum(1 for r in rows if str(r["solve"]["selected_stop_reason"]).startswith("iteration_cap"))
        ),
        "verdict": (
            "Q3_FIRST_ROUTE_LIVE"
            if survival >= ED1_BREAK_EVEN_SURVIVAL
            else "Q3_FIRST_ROUTE_NOT_CLEARED_FORMULATION_SCOPE"
        ),
        "verdict_scope": (
            "FORMULATION: q31 Q3-constrained solved Road/Lane target field on qo1 n32; "
            "not n600, not contest authority"
        ),
    }


def payload(args: argparse.Namespace, rows: list[dict[str, Any]], *, target_counts_all: np.ndarray) -> dict[str, Any]:
    return {
        "schema": "ddm_q31_q3_constrained_solve.v1",
        "captured_at_utc": utc_now(),
        "git": git_head(),
        "axis": "[macOS-CPU advisory / CPU Torch SegNet+PoseNet bounded n32]",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_run": False,
        "base": {
            "own_vehicle_frontier": f"S = {BASELINE_S} @ {BASELINE_BYTES} B [macOS-CPU advisory]",
            "d_seg": BASELINE_D_SEG,
            "d_pose": BASELINE_D_POSE,
            "sub_dir": str(args.sub_dir),
            "archive": str(args.base_archive),
            "archive_sha256": sha256_file(args.base_archive),
            "raw_path": str(args.sub_dir / "inflated" / "0.raw"),
        },
        "selection": {
            "mode": "se2 fixed stratified-random non-prefix quartiles by Road/Lane target count",
            "seed": 20260804,
            "pairs": PAIR_SET,
            "n_pairs": len(PAIR_SET),
            "selected_road_lane_target_cells": int(target_counts_all[PAIR_SET].sum()),
            "n600_road_lane_target_cells": int(target_counts_all.sum()),
            "per_pair_target_cells": {str(p): int(target_counts_all[p]) for p in PAIR_SET},
        },
        "inputs": {
            "gt_mkv": str(args.gt_mkv),
            "argmax_cache": str(args.argmax_cache),
            "gt_argmax": str(args.argmax_cache / "gt_argmax_n600.npy"),
            "current_argmax": str(args.argmax_cache / "cx1_argmax_n600.npy"),
            "bulk_rows": str(args.rows_jsonl),
        },
        "solver": {
            "steps": args.steps,
            "lr": args.lr,
            "eval_every": args.eval_every,
            "convergence_patience_evals": args.convergence_patience_evals,
            "convergence_min_improvement": args.convergence_min_improvement,
            "starts": args.starts.split(","),
            "projector": "ddm_sq1_pose_null_constrained_paint.pose_null_projector",
            "project_each_gradient": True,
            "project_each_parameter_step": True,
            "snap_to_2x2_blocks": True,
        },
        "comparators": {
            "se2_r0_delta32_target_survival": SE2_R0_DELTA32_TARGET_SURVIVAL,
            "se2_r0_delta32_global_net_reduction": SE2_R0_DELTA32_GLOBAL_NET_REDUCTION,
            "se2_project_after_q3_target_survival": SE2_PROJECT_AFTER_Q3_TARGET_SURVIVAL,
            "se2_project_after_q3_global_net_reduction": SE2_PROJECT_AFTER_Q3_GLOBAL_NET_REDUCTION,
            "ed1_break_even_survival": ED1_BREAK_EVEN_SURVIVAL,
            "sq1_unconstrained_eta_25": SQ1_UNCONSTRAINED_ETA_25,
            "sq2_unconstrained_eta_50": SQ2_UNCONSTRAINED_ETA_50,
        },
        "aggregate": aggregate(rows),
        "rows": rows,
        "boundaries": [
            "bounded n32 only",
            "macOS CPU advisory only",
            "no archive build",
            "no n600 scorer slot",
            "GT frames decoded only through frame_utils.yuv420_to_rgb via decode_gt_frames",
        ],
    }


def execute(args: argparse.Namespace) -> int:
    started = time.time()
    if args.steps < 50:
        raise RuntimeError("q31 charter requires steps >= 50")
    if args.limit and args.limit > len(PAIR_SET):
        raise RuntimeError("limit exceeds fixed q31 pair set")
    if sha256_file(args.base_archive) != QO1_ARCHIVE_SHA256:
        raise RuntimeError("qo1 archive SHA drifted; refusing q31 matched-base measurement")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.bulk_dir.mkdir(parents=True, exist_ok=True)
    args.rows_jsonl = args.bulk_dir / "q31_rows.jsonl"
    args.summary_json = args.out_dir / "q31_summary.json"
    args.bulk_summary_json = args.bulk_dir / "q31_summary.json"

    gt_cache = np.load(args.argmax_cache / "gt_argmax_n600.npy", mmap_mode="r")
    cur_cache = np.load(args.argmax_cache / "cx1_argmax_n600.npy", mmap_mode="r")
    counts = target_counts(gt_cache, cur_cache)
    if int(counts[PAIR_SET].sum()) != 12_407:
        raise RuntimeError(f"q31 denominator drift: selected target cells {int(counts[PAIR_SET].sum())}")

    raw = load_raw_pairs(args.sub_dir / "inflated" / "0.raw")
    pairs = PAIR_SET[: args.limit] if args.limit else list(PAIR_SET)
    rows = load_jsonl_rows(args.rows_jsonl) if args.resume else []
    done = {int(row["pair"]) for row in rows}
    pairs = [p for p in pairs if int(p) not in done]

    wanted: set[int] = set()
    for pair in pairs:
        wanted.update({seq_len * int(pair), seq_len * int(pair) + 1})
    gt_frames = decode_gt_frames(args.gt_mkv, wanted) if wanted else {}
    sc = Scorer(args.threads)
    segnet = sc.net.segnet
    for param in segnet.parameters():
        param.requires_grad_(False)
    projector = pose_null_projector()
    starts = tuple(s for s in args.starts.split(",") if s)
    print(
        f"[q31] ready rows={len(rows)} remaining={len(pairs)} steps={args.steps} "
        f"starts={starts} target_cells={int(counts[PAIR_SET].sum())}",
        flush=True,
    )

    write_json_atomic(args.summary_json, payload(args, rows, target_counts_all=counts))
    write_json_atomic(args.bulk_summary_json, payload(args, rows, target_counts_all=counts))

    for idx, pair in enumerate(pairs, start=1):
        pair = int(pair)
        pair_started = time.time()
        dec = np.stack([raw[seq_len * pair], raw[seq_len * pair + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * pair], gt_frames[seq_len * pair + 1]]).astype(np.uint8)
        lstar = sc.seg_argmax(dec)
        lgt = sc.seg_argmax(gt)
        cur_cached = np.asarray(cur_cache[pair])
        gt_cached = np.asarray(gt_cache[pair])
        if not (lstar == cur_cached).all():
            raise RuntimeError(f"C2 failed: qo1 decoded argmax != cached current for pair {pair}")
        if not (lgt == gt_cached).all():
            raise RuntimeError(f"C3 failed: GT decoded argmax != cached GT for pair {pair}")

        target_mask = road_lane_target(gt_cached, cur_cached)
        flips0 = lstar != lgt
        pose_gt = sc.pose_out(gt)
        d_pose_before = sc.d_pose(pose_gt, sc.pose_out(dec))
        paint, solve_diag = solve_q3_constrained(
            segnet,
            dec[1],
            gt[1],
            target_mask,
            lgt,
            projector,
            steps=args.steps,
            lr=args.lr,
            eval_every=args.eval_every,
            convergence_patience_evals=args.convergence_patience_evals,
            convergence_min_improvement=args.convergence_min_improvement,
            starts=starts,
        )
        snapped = snap_band_to_blocks(target_mask)
        cam = realize_scorer_paint_to_camera(dec[1], snapped, paint)
        scored = score_camera_pair(sc, pose_gt, lgt, flips0, target_mask, dec[0], cam)
        base_sc = torch.round(resize_to_scorer(dec[1]))[0].permute(1, 2, 0).numpy().astype(np.uint8)
        scored["d_pose_before"] = d_pose_before
        scored["d_pose_delta"] = float(scored["d_pose_after"] - d_pose_before)
        scored["d_pose_ratio"] = float(scored["d_pose_after"] / d_pose_before) if d_pose_before else None
        scored["changed_scorer_pixels"] = int((paint != base_sc).any(axis=2).sum())
        scored["changed_scorer_channel_values"] = int((paint != base_sc).sum())
        scored["yuv6_residual"] = yuv6_shift(base_sc, paint)

        row = {
            "schema": "ddm_q31_q3_constrained_solve.row.v1",
            "pair": pair,
            "flips_before": int(flips0.sum()),
            "target_cells": int(target_mask.sum()),
            "snapped_target_cells": int(snapped.sum()),
            "snap_tax_vs_target_cells": float(snapped.sum() / max(1, target_mask.sum())),
            "d_pose_before": d_pose_before,
            "C_before": confusion(lgt, lstar).tolist(),
            "controls": {
                "C2_lstar_matches_cache": True,
                "C3_lgt_matches_cache": True,
                "target_cells_match_se2_cache": int(target_mask.sum()) == int(counts[pair]),
            },
            "solve": solve_diag,
            "q3_constrained": scored,
            "elapsed_s": float(time.time() - pair_started),
        }
        rows.append(row)
        append_jsonl(args.rows_jsonl, row)
        current_payload = payload(args, rows, target_counts_all=counts)
        write_json_atomic(args.summary_json, current_payload)
        write_json_atomic(args.bulk_summary_json, current_payload)
        agg = current_payload["aggregate"]
        print(
            f"[q31] pair {pair:3d} ({len(rows)}/{len(PAIR_SET)}) "
            f"surv={scored['target_survival']:.4f} net={scored['global_net_flip_reduction']:+d} "
            f"agg_surv={agg.get('q3_constrained_target_survival')} "
            f"dpose={d_pose_before:.6g}->{scored['d_pose_after']:.6g} "
            f"stop={solve_diag['selected_stop_reason']} [{time.time()-pair_started:.1f}s]",
            flush=True,
        )

    final_payload = payload(args, rows, target_counts_all=counts)
    write_json_atomic(args.summary_json, final_payload)
    write_json_atomic(args.bulk_summary_json, final_payload)
    print(
        f"[q31] DONE rows={len(rows)} verdict={final_payload['aggregate'].get('verdict')} "
        f"surv={final_payload['aggregate'].get('q3_constrained_target_survival')} "
        f"dpose_ratio={final_payload['aggregate'].get('d_pose_ratio_vs_before')} "
        f"t={time.time()-started:.1f}s -> {args.summary_json}",
        flush=True,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-dir", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit"))
    ap.add_argument("--base-archive", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit/archive.zip"))
    ap.add_argument("--gt-mkv", type=Path, default=REPO / "upstream/videos/0.mkv")
    ap.add_argument("--argmax-cache", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache"))
    ap.add_argument("--out-dir", type=Path, default=REPO / ".omx/research/ddm_q31_20260804")
    ap.add_argument("--bulk-dir", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_q31_20260804"))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--convergence-patience-evals", type=int, default=4)
    ap.add_argument("--convergence-min-improvement", type=int, default=1)
    ap.add_argument("--starts", default="dec")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
