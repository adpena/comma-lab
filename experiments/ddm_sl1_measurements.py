#!/usr/bin/env python
"""SL1 bounded measurements.

This tool drains two debts from the SL1 charter without creating a candidate or
running a full n600 evaluator:

* ``lc1-curve`` regenerates PE3 per-record local-net attribution from the
  byte-closed PE3 section and frozen CPU SegNet argmax surfaces.
* ``f2-tail`` reruns only DQ1's high-d_pose tail pairs with the p3v2 free
  frame_0 solver at a 10x iteration budget, then recomposes the n120 pose term.

Axis for both modes: macOS CPU frozen-scorer advisory, non-promotable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "4")

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "src", REPO / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ddm_p3v2_optimal_form_pose_resolve import (  # noqa: E402
    _f0work_to_u8,
    d_pose_u8,
    load_pair,
    load_posenet,
    load_targets,
    s1d_free_solve,
    warp_base_work,
)
from ddm_rz1_pe3_head_solve import (  # noqa: E402
    effective_component_ownership,
    extract_pe3_section,
    parse_pe3_components,
)
from ddm_sq1_eta_seg_realization import (  # noqa: E402
    CAM_H,
    CAM_W,
    CLASS_NAMES,
    N_PAIRS_TOTAL,
    SEG_H,
    SEG_W,
    Scorer,
    decode_gt_frames,
    seq_len,
)

AXIS_SEG = "[macOS-CPU frozen-SegNet advisory] NON-PROMOTABLE"
AXIS_POSE = "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE"
SSD_OUT = Path("/Volumes/VertigoDataTier/pact/ddm_sl1_20260805")
P3V2_UPSTREAM = Path("/Volumes/VertigoDataTier/pact/molab_witness_machine_upstream_20260709")
DQ1_PARTIAL = Path(
    "/Volumes/VertigoDataTier/pact/ddm_dq1_20260805/"
    "dq1_p3v2_free_upper_bound_n120.partial.jsonl"
)
LC1_JSON = Path("/Volumes/VertigoDataTier/pact/ddm_lc1_20260805/lc1_label_ceiling_n32.json")


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def add_counts(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    for key in (
        "effective_pixels",
        "fixed",
        "introduced",
        "wrong_to_wrong",
        "unchanged_wrong",
        "changed_label_pixels",
        "lane_to_road_introduced",
        "record_bytes",
    ):
        bucket[key] = int(bucket.get(key, 0)) + int(row.get(key, 0))
    bucket["net_fixed"] = int(bucket.get("net_fixed", 0)) + int(row.get("net_fixed", 0))


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, dict[str, Any]] = defaultdict(lambda: {"record_count": 0})
    by_lane_bucket: dict[str, dict[str, Any]] = defaultdict(lambda: {"record_count": 0})
    for row in records:
        by_mode[row["mode_name"]]["record_count"] += 1
        add_counts(by_mode[row["mode_name"]], row)
        lane_bucket = "has_lane_to_road_introduced" if row["lane_to_road_introduced"] else "rest"
        by_lane_bucket[lane_bucket]["record_count"] += 1
        add_counts(by_lane_bucket[lane_bucket], row)

    ordered = sorted(records, key=lambda r: (-int(r["net_fixed"]), int(r["pair"]), int(r["component_ordinal"])))
    curve = []
    cum_net = 0
    cum_fixed = 0
    cum_intro = 0
    best = {"prefix_records": 0, "net_fixed": 0, "fixed": 0, "introduced": 0}
    for index, row in enumerate(ordered, start=1):
        cum_net += int(row["net_fixed"])
        cum_fixed += int(row["fixed"])
        cum_intro += int(row["introduced"])
        curve.append(
            {
                "rank": index,
                "pair": int(row["pair"]),
                "component_ordinal": int(row["component_ordinal"]),
                "mode_name": row["mode_name"],
                "net_fixed": int(row["net_fixed"]),
                "fixed": int(row["fixed"]),
                "introduced": int(row["introduced"]),
                "effective_pixels": int(row["effective_pixels"]),
                "cumulative_net_fixed": int(cum_net),
                "cumulative_fixed": int(cum_fixed),
                "cumulative_introduced": int(cum_intro),
            }
        )
        if cum_net > best["net_fixed"]:
            best = {
                "prefix_records": index,
                "net_fixed": int(cum_net),
                "fixed": int(cum_fixed),
                "introduced": int(cum_intro),
            }

    positive = [row for row in records if int(row["net_fixed"]) > 0]
    positive_totals = {"record_count": len(positive)}
    for row in positive:
        add_counts(positive_totals, row)

    return {
        "record_count": len(records),
        "positive_record_count": len(positive),
        "negative_record_count": sum(1 for row in records if int(row["net_fixed"]) < 0),
        "zero_record_count": sum(1 for row in records if int(row["net_fixed"]) == 0),
        "net_fixed_all_records": int(sum(int(row["net_fixed"]) for row in records)),
        "fixed_all_records": int(sum(int(row["fixed"]) for row in records)),
        "introduced_all_records": int(sum(int(row["introduced"]) for row in records)),
        "lane_to_road_introduced_all_records": int(
            sum(int(row["lane_to_road_introduced"]) for row in records)
        ),
        "mode_totals": dict(sorted(by_mode.items())),
        "lane_to_road_split": dict(sorted(by_lane_bucket.items())),
        "optimal_static_positive_subset": positive_totals,
        "best_sorted_prefix": best,
        "trust_gate_headroom": {
            "positive_subset_net_fixed": int(positive_totals.get("net_fixed", 0)),
            "best_prefix_net_fixed": int(best["net_fixed"]),
            "all_record_net_fixed": int(sum(int(row["net_fixed"]) for row in records)),
            "interpretation": (
                "Static trust is only positive for records with net_fixed > 0; "
                "the sorted-prefix curve is an oracle ordering upper bound, not a receiver."
            ),
        },
        "sorted_cumulative_curve": curve,
    }


def run_lc1_curve(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    lc1 = json.loads(args.lc1_json.read_text())
    inputs = lc1["inputs"]
    base_raw = Path(inputs["base_raw_path"])
    gt_mkv = Path(inputs["gt_mkv"])
    pe3_archive = Path(inputs["pe3_archive"])
    pe3_blob, pe3_field = extract_pe3_section(pe3_archive)
    components, pe3_meta = parse_pe3_components(pe3_blob)
    if len(components) != N_PAIRS_TOTAL:
        raise RuntimeError(f"PE3 pair count {len(components)} != {N_PAIRS_TOTAL}")

    raw = np.memmap(
        base_raw,
        dtype=np.uint8,
        mode="r",
        shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3),
    )
    scorer = Scorer(args.threads)
    records: list[dict[str, Any]] = []
    total_slots = SEG_H * SEG_W
    global_record = 0
    pair_chunk = int(args.pair_chunk)
    for lo in range(0, N_PAIRS_TOTAL, pair_chunk):
        hi = min(N_PAIRS_TOTAL, lo + pair_chunk)
        wanted = {seq_len * pair + t for pair in range(lo, hi) for t in (0, 1)}
        gt_frames = decode_gt_frames(gt_mkv, wanted)
        for pair in range(lo, hi):
            dec = np.stack([raw[seq_len * pair], raw[seq_len * pair + 1]]).astype(np.uint8)
            gt = np.stack([gt_frames[seq_len * pair], gt_frames[seq_len * pair + 1]])
            lstar = scorer.seg_argmax(dec).reshape(-1)
            lgt = scorer.seg_argmax(gt).reshape(-1)
            owner, owner_class, _slots = effective_component_ownership(components[pair], total_slots)
            for comp_index, component in enumerate(components[pair]):
                final_indices = np.flatnonzero(owner == int(comp_index))
                global_record += 1
                if final_indices.size:
                    target = owner_class[final_indices].astype(np.int16)
                    base = lstar[final_indices].astype(np.int16)
                    gt_labels = lgt[final_indices].astype(np.int16)
                    before_wrong = base != gt_labels
                    after_wrong = target != gt_labels
                    changed = target != base
                    fixed = int((before_wrong & ~after_wrong).sum())
                    introduced = int((~before_wrong & after_wrong).sum())
                    wrong_to_wrong = int((before_wrong & after_wrong & changed).sum())
                    unchanged_wrong = int((before_wrong & after_wrong & ~changed).sum())
                    lane_to_road = int(((~before_wrong) & after_wrong & (target == 1) & (gt_labels == 0)).sum())
                    target_counts = {
                        CLASS_NAMES[int(cls)]: int((target == int(cls)).sum())
                        for cls in np.unique(target)
                    }
                else:
                    fixed = introduced = wrong_to_wrong = unchanged_wrong = lane_to_road = 0
                    target_counts = {}
                records.append(
                    {
                        "record_id": int(global_record),
                        "pair": int(pair),
                        "component_ordinal": int(comp_index),
                        "mode_name": component.mode_name,
                        "record_bytes": int(component.record_bytes),
                        "effective_pixels": int(final_indices.size),
                        "fixed": fixed,
                        "introduced": introduced,
                        "wrong_to_wrong": wrong_to_wrong,
                        "unchanged_wrong": unchanged_wrong,
                        "changed_label_pixels": int(fixed + introduced + wrong_to_wrong),
                        "lane_to_road_introduced": lane_to_road,
                        "net_fixed": int(fixed - introduced),
                        "target_class_counts": target_counts,
                    }
                )
        print(f"[sl1 lc1] pairs {lo:03d}-{hi:03d} records={len(records)}", flush=True)

    summary = summarize_records(records)
    pe3_band_px = int(pe3_field["raster"].sum()) if "raster" in pe3_field else int(
        summary["mode_totals"]["depth_conditioned_curve"]["effective_pixels"]
        + summary["mode_totals"]["generator_pair_bisector"]["effective_pixels"]
    )
    payload = {
        "schema": "ddm_sl1_lc1_per_record_curve.v1",
        "utc": utc_now(),
        "axis": AXIS_SEG,
        "score_claim": False,
        "promotion_eligible": False,
        "method": (
            "Parse byte-closed PE3 component records; assign final effective ownership; "
            "score each record's local target-label substitution against frozen SegNet "
            "argmax for the PE4 qo1 base and canonical GT decode."
        ),
        "inputs": {
            "lc1_json": str(args.lc1_json),
            "lc1_json_sha256": sha256_path(args.lc1_json),
            "base_raw": str(base_raw),
            "base_raw_sha256": sha256_path(base_raw),
            "gt_mkv": str(gt_mkv),
            "gt_mkv_sha256": sha256_path(gt_mkv),
            "pe3_archive": str(pe3_archive),
            "pe3_archive_sha256": sha256_path(pe3_archive),
            "pe3_section_sha256": sha256(pe3_blob).hexdigest(),
        },
        "denominators": {
            "records": int(len(records)),
            "pairs": N_PAIRS_TOTAL,
            "scorer_pixels_per_pair": SEG_H * SEG_W,
            "n600_scorer_pixels": N_PAIRS_TOTAL * SEG_H * SEG_W,
            "pe3_band_px_n600": pe3_band_px,
        },
        "pe3_parse": pe3_meta,
        "summary": summary,
        "records": records,
        "elapsed_s": round(time.time() - t0, 3),
    }
    write_json(args.out, payload)
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def plateau_from_trace(trace: list[list[float]], final_iter: int) -> dict[str, Any]:
    if not trace:
        return {"plateau_ratio_final10pct": None, "plateau_pass": False}
    threshold = 0.9 * float(final_iter)
    prior = [float(d) for it, d in trace if float(it) < threshold]
    window = [(int(it), float(d)) for it, d in trace if float(it) >= threshold]
    if not window:
        return {"plateau_ratio_final10pct": None, "plateau_pass": False, "reason": "no_final_window_points"}
    start_best = min(prior) if prior else float(window[0][1])
    final_best = min(float(d) for _it, d in window)
    ratio = final_best / start_best if start_best > 0 else math.inf
    return {
        "final10pct_start_iter": int(math.ceil(threshold)),
        "final10pct_points": window,
        "best_before_final10pct": float(start_best),
        "best_in_final10pct": float(final_best),
        "plateau_ratio_final10pct": float(ratio),
        "plateau_pass": bool(ratio > 0.995),
        "criterion": "best_in_final_10_percent / best_before_final_10_percent > 0.995",
    }


def run_f2_tail(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.set_num_threads(args.threads)
    t0 = time.time()
    rows = read_jsonl(args.dq1_partial)
    tail = [row for row in rows if float(row["d_pose_free_u8"]) > args.tail_threshold]
    if args.limit:
        tail = tail[: int(args.limit)]
    p3v2_upstream_str = str(P3V2_UPSTREAM)
    sys.path[:] = [entry for entry in sys.path if entry != p3v2_upstream_str]
    sys.path.insert(0, p3v2_upstream_str)
    cached_modules = sys.modules.get("modules")
    cached_modules_path = Path(getattr(cached_modules, "__file__", "")) if cached_modules else None
    if cached_modules_path and cached_modules_path.resolve() != (P3V2_UPSTREAM / "modules.py").resolve():
        del sys.modules["modules"]
    targets = load_targets(N_PAIRS_TOTAL)
    posenet, _modules = load_posenet()
    out_jsonl = args.out_jsonl
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done: dict[int, dict[str, Any]] = {}
    if args.resume and out_jsonl.exists():
        for row in read_jsonl(out_jsonl):
            done[int(row["pair"])] = row
    handle = out_jsonl.open("a")
    try:
        for old in tail:
            pair_id = int(old["pair"])
            if pair_id in done:
                continue
            pair = load_pair(pair_id)
            f1 = pair[1]
            target = targets[pair_id]
            base_work, s_t, dpose_warp = warp_base_work(
                posenet,
                f1,
                target,
                int(args.work_h),
                int(args.work_w),
            )
            result = s1d_free_solve(
                posenet,
                base_work,
                f1,
                target,
                iters=int(args.iters),
                lr=float(args.lr),
                wh=int(args.work_h),
                ww=int(args.work_w),
                tol=float(args.legacy_tol),
            )
            final_u8 = _f0work_to_u8(result["f0_work"])
            verified = d_pose_u8(posenet, final_u8, f1, target)
            trace = [[int(it), float(dp)] for it, dp in result["traj"]]
            plateau = plateau_from_trace(trace, int(result["iters_used"]))
            row = {
                "schema": "ddm_sl1_f2_tail_pair.v1",
                "pair": pair_id,
                "selection_ordinal": int(old["selection_ordinal"]),
                "selection_mode": old.get("selection_mode"),
                "selection_seed": old.get("selection_seed"),
                "old_d_pose_free_u8_160": float(old["d_pose_free_u8"]),
                "new_d_pose_free_u8": float(result["d_pose_free_u8"]),
                "new_d_pose_free_u8_verified": float(verified),
                "d_pose_warp_base": float(dpose_warp),
                "warp_s_t": float(s_t),
                "iters_budget": int(args.iters),
                "iters_used": int(result["iters_used"]),
                "lr": float(args.lr),
                "work_res": [int(args.work_h), int(args.work_w)],
                "free_traj": trace,
                "plateau": plateau,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            done[pair_id] = row
            print(
                f"[sl1 f2] pair {pair_id:3d}: old={row['old_d_pose_free_u8_160']:.6g} "
                f"new={row['new_d_pose_free_u8']:.6g} iters={row['iters_used']} "
                f"plateau={plateau['plateau_pass']}",
                flush=True,
            )
    finally:
        handle.close()

    corrected = []
    for row in rows:
        pair_id = int(row["pair"])
        if pair_id in done:
            corrected.append(float(done[pair_id]["new_d_pose_free_u8_verified"]))
        else:
            corrected.append(float(row["d_pose_free_u8"]))
    corrected_mean = float(np.mean(corrected))
    old_mean = float(np.mean([float(row["d_pose_free_u8"]) for row in rows]))
    payload = {
        "schema": "ddm_sl1_f2_tail_confirmation.v1",
        "utc": utc_now(),
        "axis": AXIS_POSE,
        "score_claim": False,
        "promotion_eligible": False,
        "source_dq1_partial": str(args.dq1_partial),
        "source_dq1_partial_sha256": sha256_path(args.dq1_partial),
        "tail_threshold": float(args.tail_threshold),
        "tail_pairs_requested": [int(row["pair"]) for row in tail],
        "tail_pairs_done": sorted(done),
        "tail_done_count": len(done),
        "tail_required_count": len([row for row in rows if float(row["d_pose_free_u8"]) > args.tail_threshold]),
        "iters_budget": int(args.iters),
        "old_n120_mean_d_pose": old_mean,
        "old_n120_pose_term": math.sqrt(10.0 * old_mean),
        "corrected_n120_mean_d_pose": corrected_mean,
        "corrected_n120_pose_term": math.sqrt(10.0 * corrected_mean),
        "old_tail_mean_d_pose": float(np.mean([float(row["d_pose_free_u8"]) for row in tail])) if tail else None,
        "new_tail_mean_d_pose": float(np.mean([float(done[int(row["pair"])]["new_d_pose_free_u8_verified"]) for row in tail if int(row["pair"]) in done])) if done else None,
        "tail_rows": [done[int(row["pair"])] for row in tail if int(row["pair"]) in done],
        "verdict": {
            "dq1_wall_budget_conditional": bool(any(not done[int(row["pair"])]["plateau"]["plateau_pass"] for row in tail if int(row["pair"]) in done)),
            "refuted_at_convergence": bool(
                len(done) == len([row for row in rows if float(row["d_pose_free_u8"]) > args.tail_threshold])
                and math.sqrt(10.0 * corrected_mean) <= 0.05
                and all(done[int(row["pair"])]["plateau"]["plateau_pass"] for row in tail if int(row["pair"]) in done)
            ),
            "scope": "FORMULATION; tail-only 10x rerun plus DQ1 non-tail rows",
        },
        "elapsed_s": round(time.time() - t0, 3),
    }
    write_json(args.out, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    lc1 = sub.add_parser("lc1-curve")
    lc1.add_argument("--lc1-json", type=Path, default=LC1_JSON)
    lc1.add_argument("--out", type=Path, default=SSD_OUT / "sl1_lc1_per_record_curve.json")
    lc1.add_argument("--threads", type=int, default=4)
    lc1.add_argument("--pair-chunk", type=int, default=24)
    lc1.set_defaults(func=run_lc1_curve)

    f2 = sub.add_parser("f2-tail")
    f2.add_argument("--dq1-partial", type=Path, default=DQ1_PARTIAL)
    f2.add_argument("--out", type=Path, default=SSD_OUT / "sl1_f2_tail_confirmation.json")
    f2.add_argument("--out-jsonl", type=Path, default=SSD_OUT / "sl1_f2_tail_confirmation.partial.jsonl")
    f2.add_argument("--tail-threshold", type=float, default=1.0e-3)
    f2.add_argument("--iters", type=int, default=1600)
    f2.add_argument("--lr", type=float, default=3.0)
    f2.add_argument("--work-h", type=int, default=192)
    f2.add_argument("--work-w", type=int, default=256)
    f2.add_argument("--legacy-tol", type=float, default=1.0e-12)
    f2.add_argument("--threads", type=int, default=4)
    f2.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    f2.add_argument("--limit", type=int, default=0)
    f2.set_defaults(func=run_f2_tail)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = args.func(args)
    print(f"[sl1] wrote {args.out} sha256={sha256_path(args.out)}", flush=True)
    if args.mode == "f2-tail":
        print(
            f"[sl1] corrected n120 pose term={payload['corrected_n120_pose_term']:.9f} "
            f"done={payload['tail_done_count']}/{payload['tail_required_count']}",
            flush=True,
        )
    if args.mode == "lc1-curve":
        summary = payload["summary"]
        print(
            f"[sl1] lc1 records={summary['record_count']} "
            f"positive={summary['positive_record_count']} "
            f"best_prefix_net={summary['best_sorted_prefix']['net_fixed']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
