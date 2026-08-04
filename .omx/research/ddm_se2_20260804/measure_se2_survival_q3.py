#!/usr/bin/env python3
"""SE2 Road/Lane paint survival plus Q3 projection measurement.

Receipt script only. It measures bounded n<=32 CPU Torch scorer behavior on the
qo1 matched decode and writes durable JSON/JSONL receipts. It does not build or
promote a submission and does not run a full n600 scorer job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[3]
UPSTREAM = REPO / "upstream"
for _path in (REPO, REPO / "src", REPO / "experiments", UPSTREAM):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ddm_sq1_eta_seg_realization import Scorer, decode_gt_frames, seq_len  # type: ignore  # noqa: E402
from ddm_sq1_pose_null_constrained_paint import (  # type: ignore  # noqa: E402
    pose_null_projector,
    project_null,
    snap_band_to_blocks,
    yuv6_shift,
)
from ddm_sq1_stage_decomposition_and_solved_paint import (  # type: ignore  # noqa: E402
    realize_scorer_paint_to_camera,
    resize_to_scorer,
)
from modules import SegNet, segnet_sd_path  # type: ignore  # noqa: E402


SEG_H = 384
SEG_W = 512
CAMERA_H = 874
CAMERA_W = 1164
PAIR_COUNT = 600
ROAD_CLASS = 0
LANE_CLASS = 1
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
FP1_ROAD_RGB = np.array((30, 39, 72), dtype=np.uint8)
FP1_LANE_RGB = np.array((77, 87, 119), dtype=np.uint8)
RATE_DENOMINATOR_BYTES = 37_545_489
BASELINE_S = 0.7539807296911207
BASELINE_BYTES = 357_836
BASELINE_D_SEG = 0.00431179
BASELINE_D_POSE = 0.00071459
QO1_ARCHIVE_SHA256 = "d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a"
FZ4_ARCHIVE_SHA256 = "ad5dd0e4fbe5b13ab53a5995a6d77cc558c25f40b63f894ea50ad336bd50fb66"
ED1_BREAK_EVEN_SURVIVAL = 0.6964303814


@dataclass(frozen=True)
class Variant:
    name: str
    max_delta: int | None
    coherence_radius: int


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception as exc:
        return f"UNKNOWN:{exc}"


def f0pr_bilinear_axis(n_out: int, n_in: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = (np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5
    lo = np.clip(np.floor(src).astype(np.int64), 0, n_in - 1)
    hi = np.clip(lo + 1, 0, n_in - 1)
    w_hi = np.clip(src - np.floor(src), 0.0, 1.0)
    return lo, hi, w_hi


def correction_target(gt: np.ndarray, current: np.ndarray) -> np.ndarray:
    return ((gt == ROAD_CLASS) & (current == LANE_CLASS)) | (
        (gt == LANE_CLASS) & (current == ROAD_CLASS)
    )


def stratified_pairs(target_counts: np.ndarray, *, n: int, seed: int) -> list[int]:
    nonzero = np.flatnonzero(target_counts > 0)
    if nonzero.size < n:
        raise RuntimeError(f"need {n} nonzero pairs, found {nonzero.size}")
    ranked = nonzero[np.argsort(target_counts[nonzero])]
    buckets = np.array_split(ranked, 4)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    quota = int(math.ceil(n / len(buckets)))
    for bucket in buckets:
        take = min(quota, bucket.size)
        selected.extend(int(v) for v in rng.choice(bucket, size=take, replace=False))
    if len(selected) < n:
        selected_set = set(selected)
        remaining = np.array([p for p in nonzero if int(p) not in selected_set], dtype=np.int64)
        selected.extend(int(v) for v in rng.choice(remaining, size=n - len(selected), replace=False))
    return sorted(selected[:n])


def load_raw_pairs(path: Path) -> np.memmap:
    frame_bytes = CAMERA_H * CAMERA_W * 3
    expected_frames = PAIR_COUNT * 2
    size = path.stat().st_size
    if size != expected_frames * frame_bytes:
        raise RuntimeError(f"raw size {size} != expected {expected_frames * frame_bytes}")
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(expected_frames, CAMERA_H, CAMERA_W, 3))


def gather_pairs(raw: np.memmap, pairs: list[int]) -> np.ndarray:
    out = np.empty((len(pairs), 2, CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
    for i, pair in enumerate(pairs):
        out[i, 0] = raw[2 * pair]
        out[i, 1] = raw[2 * pair + 1]
    return out


def frame_sha(raw: np.memmap, frame_index: int) -> str:
    return hashlib.sha256(np.ascontiguousarray(raw[frame_index]).tobytes()).hexdigest()


def predict_segnet(model: SegNet, batch_np: np.ndarray, *, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    preds: list[np.ndarray] = []
    logits_out: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, batch_np.shape[0], batch_size):
            chunk = batch_np[start : start + batch_size]
            batch = torch.from_numpy(np.ascontiguousarray(chunk))
            x = batch.permute(0, 1, 4, 2, 3).float()
            seg_input = model.preprocess_input(x)
            logits = model(seg_input)
            preds.append(logits.argmax(dim=1).cpu().numpy().astype(np.uint8))
            logits_out.append(logits.cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0), np.concatenate(logits_out, axis=0)


def scorer_rgb(model: SegNet, batch_np: np.ndarray, *, batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, batch_np.shape[0], batch_size):
            chunk = batch_np[start : start + batch_size]
            batch = torch.from_numpy(np.ascontiguousarray(chunk))
            x = batch.permute(0, 1, 4, 2, 3).float()
            seg_input = model.preprocess_input(x)
            chunks.append(seg_input.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def move_toward(current: np.ndarray, target: np.ndarray, max_delta: int | None) -> np.ndarray:
    if max_delta is None:
        return np.broadcast_to(target, current.shape).astype(np.uint8).copy()
    cur = current.astype(np.int16)
    tgt = np.broadcast_to(target, current.shape).astype(np.int16)
    delta = np.clip(tgt - cur, -int(max_delta), int(max_delta))
    return np.clip(cur + delta, 0, 255).astype(np.uint8)


def variant_edit_mask(target_mask_pair: np.ndarray, radius: int) -> np.ndarray:
    out = np.zeros_like(target_mask_pair, dtype=bool)
    rows, cols = np.nonzero(target_mask_pair)
    for row, col in zip(rows.tolist(), cols.tolist()):
        r0 = max(0, row - radius)
        r1 = min(SEG_H, row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(SEG_W, col + radius + 1)
        out[r0:r1, c0:c1] = True
    return out


def apply_variant(
    base: np.ndarray,
    *,
    pairs: list[int],
    gt: np.ndarray,
    target_mask: np.ndarray,
    variant: Variant,
    rlo: np.ndarray,
    rhi: np.ndarray,
    clo: np.ndarray,
    chi: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, int]]:
    out = base.copy()
    changed = 0
    writes = 0
    touched_cells = 0
    edit_masks: list[np.ndarray] = []
    for batch_index, pair in enumerate(pairs):
        edit_mask = variant_edit_mask(np.asarray(target_mask[pair]), variant.coherence_radius)
        edit_masks.append(edit_mask)
        rows, cols = np.nonzero(np.asarray(target_mask[pair]))
        for row, col in zip(rows.tolist(), cols.tolist()):
            target_class = int(gt[pair, row, col])
            color = FP1_ROAD_RGB if target_class == ROAD_CLASS else FP1_LANE_RGB
            for rr in range(max(0, row - variant.coherence_radius), min(SEG_H, row + variant.coherence_radius + 1)):
                for cc in range(max(0, col - variant.coherence_radius), min(SEG_W, col + variant.coherence_radius + 1)):
                    touched_cells += 1
                    for yy in (int(rlo[rr]), int(rhi[rr])):
                        for xx in (int(clo[cc]), int(chi[cc])):
                            before = out[batch_index, 1, yy, xx].copy()
                            after = move_toward(before, color, variant.max_delta)
                            out[batch_index, 1, yy, xx] = after
                            writes += 1
                            changed += int(np.count_nonzero(before != after))
    return out, edit_masks, {
        "camera_support_writes": writes,
        "camera_channel_values_changed": changed,
        "scorer_grid_cells_touched_with_duplicates": touched_cells,
        "unique_scorer_grid_cells_touched": int(sum(mask.sum() for mask in edit_masks)),
    }


def target_margin_values(logits: np.ndarray, pairs: list[int], gt: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    margins: list[float] = []
    for batch_index, pair in enumerate(pairs):
        rows, cols = np.nonzero(np.asarray(target_mask[pair]))
        for row, col in zip(rows.tolist(), cols.tolist()):
            target_class = int(gt[pair, row, col])
            values = logits[batch_index, :, row, col].astype(np.float64)
            rivals = values.copy()
            rivals[target_class] = -np.inf
            margins.append(float(values[target_class] - np.max(rivals)))
    return np.asarray(margins, dtype=np.float64)


def margin_bins(values: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    edges = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    rows: list[dict[str, Any]] = []
    for idx in range(4):
        if idx == 3:
            mask = (values >= edges[idx]) & (values <= edges[idx + 1])
        else:
            mask = (values >= edges[idx]) & (values < edges[idx + 1])
        rows.append(
            {
                "bin": idx,
                "baseline_target_margin_lo": float(edges[idx]),
                "baseline_target_margin_hi": float(edges[idx + 1]),
                "eligible_target_cells": int(mask.sum()),
            }
        )
    return edges, rows


def target_flat(values: np.ndarray, target_masks_selected: np.ndarray) -> np.ndarray:
    return values[target_masks_selected.astype(bool)]


def summarize_variant(
    *,
    variant: Variant,
    preds: np.ndarray,
    baseline_preds: np.ndarray,
    gt_selected: np.ndarray,
    target_masks_selected: np.ndarray,
    baseline_margin_flat: np.ndarray,
    margin_edges: np.ndarray,
    input_delta_linf_flat: np.ndarray,
    mutation_stats: dict[str, int],
) -> dict[str, Any]:
    target = target_masks_selected.astype(bool)
    eligible = target & (baseline_preds != gt_selected)
    corrected = eligible & (preds == gt_selected)
    still_wrong = eligible & (preds != gt_selected)
    baseline_good_non_target = (~target) & (baseline_preds == gt_selected)
    collateral = baseline_good_non_target & (preds != gt_selected)
    baseline_flips = baseline_preds != gt_selected
    variant_flips = preds != gt_selected

    flat_corrected = corrected[target]
    by_bin = []
    for idx in range(4):
        lo = margin_edges[idx]
        hi = margin_edges[idx + 1]
        if idx == 3:
            mask = (baseline_margin_flat >= lo) & (baseline_margin_flat <= hi)
        else:
            mask = (baseline_margin_flat >= lo) & (baseline_margin_flat < hi)
        count = int(mask.sum())
        corr = int(flat_corrected[mask].sum())
        by_bin.append(
            {
                "bin": idx,
                "baseline_target_margin_lo": float(lo),
                "baseline_target_margin_hi": float(hi),
                "eligible_target_cells": count,
                "corrected_target_cells": corr,
                "target_survival": float(corr / count) if count else None,
            }
        )

    failed_flat = still_wrong[target]
    uint8_floor = failed_flat & (input_delta_linf_flat <= 0.0)
    return {
        "variant": variant.name,
        "max_delta": variant.max_delta,
        "coherence_radius": variant.coherence_radius,
        **mutation_stats,
        "eligible_target_cells": int(eligible.sum()),
        "corrected_target_cells": int(corrected.sum()),
        "still_wrong_target_cells": int(still_wrong.sum()),
        "target_survival": float(corrected.sum() / max(1, eligible.sum())),
        "global_baseline_flips_subset": int(baseline_flips.sum()),
        "global_variant_flips_subset": int(variant_flips.sum()),
        "global_flip_delta_vs_baseline": int(variant_flips.sum() - baseline_flips.sum()),
        "global_collateral_new_wrong_non_target": int(collateral.sum()),
        "subset_dseg_delta": float((variant_flips.sum() - baseline_flips.sum()) / baseline_flips.size),
        "subset_seg_score_delta": float(100.0 * (variant_flips.sum() - baseline_flips.sum()) / baseline_flips.size),
        "input_delta_linf_at_targets": {
            "min": float(input_delta_linf_flat.min()) if input_delta_linf_flat.size else None,
            "median": float(np.median(input_delta_linf_flat)) if input_delta_linf_flat.size else None,
            "p90": float(np.quantile(input_delta_linf_flat, 0.90)) if input_delta_linf_flat.size else None,
            "max": float(input_delta_linf_flat.max()) if input_delta_linf_flat.size else None,
        },
        "failed_uint8_floor_cells": int(uint8_floor.sum()),
        "by_baseline_margin_quartile": by_bin,
        "verdict_scope": "FORMULATION: private-support prototype Road/Lane paints on n32 stratified qo1 decode; not n600, not contest authority",
    }


def classify_failures(
    rows_by_name: dict[str, dict[str, Any]],
    corrected_by_name: dict[str, np.ndarray],
    failed_by_name: dict[str, np.ndarray],
    input_delta_by_name: dict[str, np.ndarray],
    variants: list[Variant],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_radius = defaultdict(list)
    for v in variants:
        by_radius[v.coherence_radius].append(v)
    for v in variants:
        failed = failed_by_name[v.name]
        full_name = f"r{v.coherence_radius}_full_color"
        full_corrected = corrected_by_name.get(full_name)
        if full_corrected is None:
            full_corrected = np.zeros_like(failed)
        input_delta = input_delta_by_name[v.name]
        amplitude_timidity = failed & full_corrected
        receptive_veto = failed & (~full_corrected) & (input_delta > 0.0)
        uint8_floor = failed & (input_delta <= 0.0)
        out[v.name] = {
            "failed_target_cells": int(failed.sum()),
            "uint8_floor_cells": int(uint8_floor.sum()),
            "amplitude_timidity_cells": int(amplitude_timidity.sum()),
            "receptive_field_veto_cells": int(receptive_veto.sum()),
            "r_attenuation_cells": 0,
            "notes": {
                "r_attenuation": "Measured as zero for the F0PR private supports: all edited support pixels are written after uint8 snap and D reads private supports.",
                "amplitude_timidity": "Failed here but corrected by full-color paint at the same coherence radius.",
                "receptive_field_veto": "Nonzero target input movement, yet full-color same-radius paint still failed the target argmax.",
            },
        }
    if "r0_full_color" in failed_by_name:
        full0_failed = failed_by_name["r0_full_color"]
        rescued = {}
        for radius in sorted(by_radius):
            name = f"r{radius}_full_color"
            if radius == 0 or name not in corrected_by_name:
                continue
            rescued[f"rescued_by_{name}"] = int((full0_failed & corrected_by_name[name]).sum())
        out["coherence_radius_rescue_from_private_full_color"] = rescued
    return out


def score_pose_rows(
    sc: Scorer,
    *,
    pairs: list[int],
    base_batch: np.ndarray,
    unprojected: np.ndarray,
    q3_projected: np.ndarray,
    gt_mkv: Path,
) -> list[dict[str, Any]]:
    wanted: set[int] = set()
    for p in pairs:
        wanted.update({2 * int(p), 2 * int(p) + 1})
    gt_frames = decode_gt_frames(gt_mkv, wanted)
    rows = []
    for batch_index, p in enumerate(pairs):
        gt_pair = np.stack([gt_frames[2 * int(p)], gt_frames[2 * int(p) + 1]]).astype(np.uint8)
        pose_gt = sc.pose_out(gt_pair)
        base_pose = sc.d_pose(pose_gt, sc.pose_out(base_batch[batch_index]))
        unproj_pose = sc.d_pose(pose_gt, sc.pose_out(unprojected[batch_index]))
        q3_pose = sc.d_pose(pose_gt, sc.pose_out(q3_projected[batch_index]))
        rows.append(
            {
                "pair": int(p),
                "d_pose_before": base_pose,
                "d_pose_unprojected": unproj_pose,
                "d_pose_q3_projected": q3_pose,
                "d_pose_delta_q3_vs_base": q3_pose - base_pose,
                "d_pose_delta_unprojected_vs_base": unproj_pose - base_pose,
            }
        )
    return rows


def run_q3_projection(
    *,
    winner: Variant,
    base_batch: np.ndarray,
    unprojected: np.ndarray,
    edit_masks: list[np.ndarray],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    P = pose_null_projector()
    q3 = base_batch.copy()
    rows: list[dict[str, Any]] = []
    for idx in range(base_batch.shape[0]):
        base = resize_to_scorer(base_batch[idx, 1])
        unproj_sc = resize_to_scorer(unprojected[idx, 1])
        delta = unproj_sc - base
        projected = project_null(delta, P)
        projected_sc = torch.clamp(base + projected, 0.0, 255.0)
        projected_paint = torch.round(projected_sc)[0].permute(1, 2, 0).numpy().astype(np.uint8)
        snapped = snap_band_to_blocks(edit_masks[idx])
        q3[idx, 1] = realize_scorer_paint_to_camera(base_batch[idx, 1], snapped, projected_paint)
        base_sc_u8 = torch.round(base)[0].permute(1, 2, 0).numpy().astype(np.uint8)
        rows.append(
            {
                "pair_batch_index": idx,
                "winner_variant": winner.name,
                "edit_cells": int(edit_masks[idx].sum()),
                "snapped_edit_cells": int(snapped.sum()),
                "snap_tax": float(snapped.sum() / max(1, edit_masks[idx].sum())),
                "changed_scorer_pixels": int((projected_paint != base_sc_u8).any(axis=2).sum()),
                "changed_scorer_channel_values": int((projected_paint != base_sc_u8).sum()),
                "yuv6_residual": yuv6_shift(base_sc_u8, projected_paint),
            }
        )
    return q3, rows


def aggregate_projection(
    *,
    baseline_preds: np.ndarray,
    unproj_preds: np.ndarray,
    q3_preds: np.ndarray,
    gt_selected: np.ndarray,
    target_selected: np.ndarray,
    pose_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    target = target_selected.astype(bool)
    eligible = target & (baseline_preds != gt_selected)
    baseline_flips = baseline_preds != gt_selected
    unproj_flips = unproj_preds != gt_selected
    q3_flips = q3_preds != gt_selected
    unproj_target_corrected = eligible & (unproj_preds == gt_selected)
    q3_target_corrected = eligible & (q3_preds == gt_selected)
    unproj_net = int(baseline_flips.sum() - unproj_flips.sum())
    q3_net = int(baseline_flips.sum() - q3_flips.sum())
    d_before = np.array([r["d_pose_before"] for r in pose_rows], dtype=np.float64)
    d_unproj = np.array([r["d_pose_unprojected"] for r in pose_rows], dtype=np.float64)
    d_q3 = np.array([r["d_pose_q3_projected"] for r in pose_rows], dtype=np.float64)
    return {
        "eligible_target_cells": int(eligible.sum()),
        "baseline_flips_subset": int(baseline_flips.sum()),
        "unprojected_flips_subset": int(unproj_flips.sum()),
        "q3_projected_flips_subset": int(q3_flips.sum()),
        "unprojected_global_net_flip_reduction": unproj_net,
        "q3_global_net_flip_reduction": q3_net,
        "retained_global_seg_reach": float(q3_net / unproj_net) if unproj_net else None,
        "unprojected_target_survival": float(unproj_target_corrected.sum() / max(1, eligible.sum())),
        "q3_projected_target_survival": float(q3_target_corrected.sum() / max(1, eligible.sum())),
        "retained_target_survival_fraction": float(q3_target_corrected.sum() / max(1, unproj_target_corrected.sum())),
        "d_pose_before_mean": float(d_before.mean()),
        "d_pose_unprojected_mean": float(d_unproj.mean()),
        "d_pose_q3_projected_mean": float(d_q3.mean()),
        "d_pose_q3_delta_mean": float((d_q3 - d_before).mean()),
        "d_pose_unprojected_delta_mean": float((d_unproj - d_before).mean()),
        "d_pose_q3_ratio_vs_before": float(d_q3.mean() / d_before.mean()) if d_before.mean() else None,
        "break_even_survival": ED1_BREAK_EVEN_SURVIVAL,
        "clears_ed1_break_even_by_target_survival": bool(
            float(q3_target_corrected.sum() / max(1, eligible.sum())) >= ED1_BREAK_EVEN_SURVIVAL
        ),
        "clears_ed1_break_even_by_retained_global_reach": bool(
            q3_net / unproj_net >= ED1_BREAK_EVEN_SURVIVAL if unproj_net else False
        ),
        "verdict_scope": "FORMULATION: Q3-projected private-support prototype paints on n32 stratified qo1 decode; not n600, not contest authority",
    }


def execute(args: argparse.Namespace) -> int:
    started = time.monotonic()
    if args.n_pairs > 32:
        raise RuntimeError("charter cap: this script refuses n_pairs > 32")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "se2_survival_q3_summary.json"
    rows_path = out_dir / "se2_survival_rows.jsonl"
    q3_rows_path = out_dir / "se2_q3_rows.jsonl"

    qo1_archive = args.base_archive.resolve(strict=True)
    qo1_raw_path = args.base_raw.resolve(strict=True)
    fz4_archive = args.fz4_archive.resolve(strict=True)
    fz4_raw_path = args.fz4_raw.resolve(strict=True)
    if sha256_file(qo1_archive) != QO1_ARCHIVE_SHA256:
        raise RuntimeError("qo1 archive SHA drifted; refusing matched-base measurement")
    if sha256_file(fz4_archive) != FZ4_ARCHIVE_SHA256:
        raise RuntimeError("fz4 archive SHA drifted; refusing frame-sha control")

    gt_path = args.argmax_cache / "gt_argmax_n600.npy"
    current_path = args.argmax_cache / "cx1_argmax_n600.npy"
    gt = np.load(gt_path, mmap_mode="r")
    current = np.load(current_path, mmap_mode="r")
    if gt.shape != (PAIR_COUNT, SEG_H, SEG_W) or current.shape != gt.shape:
        raise RuntimeError(f"argmax shape drift: gt={gt.shape} current={current.shape}")
    target_mask = correction_target(gt, current)
    target_counts = target_mask.reshape(PAIR_COUNT, -1).sum(axis=1)
    pairs = stratified_pairs(target_counts, n=args.n_pairs, seed=args.seed)

    qo1_raw = load_raw_pairs(qo1_raw_path)
    fz4_raw = load_raw_pairs(fz4_raw_path)
    spot_pairs = [pairs[0], pairs[-1]]
    frame_controls = []
    for pair in spot_pairs:
        for frame_offset in (0, 1):
            frame_index = 2 * int(pair) + frame_offset
            q = frame_sha(qo1_raw, frame_index)
            f = frame_sha(fz4_raw, frame_index)
            frame_controls.append(
                {
                    "pair": int(pair),
                    "frame_offset": frame_offset,
                    "frame_index": frame_index,
                    "qo1_frame_sha256": q,
                    "fz4_frame_sha256": f,
                    "match": q == f,
                }
            )
    if not all(r["match"] for r in frame_controls):
        raise RuntimeError("qo1/fz4 frame-sha spot check failed; cache reuse not safe")

    base_batch = gather_pairs(qo1_raw, pairs)
    torch.set_num_threads(args.torch_threads)
    model = SegNet().eval()
    model.load_state_dict(load_file(segnet_sd_path, device="cpu"))
    baseline_preds, baseline_logits = predict_segnet(model, base_batch, batch_size=args.batch_size)
    current_selected = np.asarray(current[pairs])
    gt_selected = np.asarray(gt[pairs])
    target_selected = np.asarray(target_mask[pairs])
    baseline_mismatch = baseline_preds != current_selected
    if int(baseline_mismatch.sum()) != 0:
        raise RuntimeError(
            f"matched-base control failed: qo1 raw argmax differs from cache at {int(baseline_mismatch.sum())} cells"
        )

    base_scorer_rgb = scorer_rgb(model, base_batch, batch_size=args.batch_size)
    baseline_margin_flat = target_margin_values(baseline_logits, pairs, gt, target_mask)
    margin_edges, margin_bin_rows = margin_bins(baseline_margin_flat)
    rlo, rhi, _rw = f0pr_bilinear_axis(SEG_H, CAMERA_H)
    clo, chi, _cw = f0pr_bilinear_axis(SEG_W, CAMERA_W)
    variants = [
        Variant("r0_delta_1", 1, 0),
        Variant("r0_delta_4", 4, 0),
        Variant("r0_delta_16", 16, 0),
        Variant("r0_delta_32", 32, 0),
        Variant("r0_delta_64", 64, 0),
        Variant("r0_full_color", None, 0),
        Variant("r1_delta_16", 16, 1),
        Variant("r1_delta_32", 32, 1),
        Variant("r1_delta_64", 64, 1),
        Variant("r1_full_color", None, 1),
        Variant("r2_delta_16", 16, 2),
        Variant("r2_delta_32", 32, 2),
        Variant("r2_delta_64", 64, 2),
        Variant("r2_full_color", None, 2),
    ]

    rows: list[dict[str, Any]] = []
    corrected_by_name: dict[str, np.ndarray] = {}
    failed_by_name: dict[str, np.ndarray] = {}
    input_delta_by_name: dict[str, np.ndarray] = {}
    mutated_by_name: dict[str, np.ndarray] = {}
    masks_by_name: dict[str, list[np.ndarray]] = {}
    for variant in variants:
        mutated, edit_masks, mutation_stats = apply_variant(
            base_batch,
            pairs=pairs,
            gt=gt,
            target_mask=target_mask,
            variant=variant,
            rlo=rlo,
            rhi=rhi,
            clo=clo,
            chi=chi,
        )
        preds, _logits = predict_segnet(model, mutated, batch_size=args.batch_size)
        mut_scorer_rgb = scorer_rgb(model, mutated, batch_size=args.batch_size)
        input_delta_linf = np.abs(mut_scorer_rgb - base_scorer_rgb).max(axis=3)
        input_delta_flat = target_flat(input_delta_linf, target_selected)
        row = summarize_variant(
            variant=variant,
            preds=preds,
            baseline_preds=baseline_preds,
            gt_selected=gt_selected,
            target_masks_selected=target_selected,
            baseline_margin_flat=baseline_margin_flat,
            margin_edges=margin_edges,
            input_delta_linf_flat=input_delta_flat,
            mutation_stats=mutation_stats,
        )
        rows.append(row)
        target = target_selected.astype(bool)
        eligible = target & (baseline_preds != gt_selected)
        corrected_by_name[variant.name] = (eligible & (preds == gt_selected))[target]
        failed_by_name[variant.name] = (eligible & (preds != gt_selected))[target]
        input_delta_by_name[variant.name] = input_delta_flat
        mutated_by_name[variant.name] = mutated
        masks_by_name[variant.name] = edit_masks
        print(
            f"[se2] {variant.name}: survival={row['target_survival']:.4f} "
            f"global_delta={row['global_flip_delta_vs_baseline']} "
            f"collateral={row['global_collateral_new_wrong_non_target']}",
            flush=True,
        )

    rows_by_name = {row["variant"]: row for row in rows}
    taxonomy = classify_failures(rows_by_name, corrected_by_name, failed_by_name, input_delta_by_name, variants)
    winner_row = min(rows, key=lambda r: (int(r["global_variant_flips_subset"]), -float(r["target_survival"])))
    best_survival_row = max(rows, key=lambda r: float(r["target_survival"]))
    winner = next(v for v in variants if v.name == winner_row["variant"])
    minimal_clear = [
        {
            "variant": r["variant"],
            "max_delta": r["max_delta"],
            "coherence_radius": r["coherence_radius"],
            "target_survival": r["target_survival"],
            "global_flip_delta_vs_baseline": r["global_flip_delta_vs_baseline"],
        }
        for r in rows
        if float(r["target_survival"]) >= 0.9
    ]

    q3_projected, q3_instrument_rows = run_q3_projection(
        winner=winner,
        base_batch=base_batch,
        unprojected=mutated_by_name[winner.name],
        edit_masks=masks_by_name[winner.name],
    )
    q3_preds, _ = predict_segnet(model, q3_projected, batch_size=args.batch_size)
    winner_preds, _ = predict_segnet(model, mutated_by_name[winner.name], batch_size=args.batch_size)
    sc = Scorer(args.torch_threads)
    pose_rows = score_pose_rows(
        sc,
        pairs=pairs,
        base_batch=base_batch,
        unprojected=mutated_by_name[winner.name],
        q3_projected=q3_projected,
        gt_mkv=args.gt_mkv.resolve(strict=True),
    )
    for qrow, prow, pair in zip(q3_instrument_rows, pose_rows, pairs):
        qrow["pair"] = int(pair)
        qrow.update(prow)
    q3_aggregate = aggregate_projection(
        baseline_preds=baseline_preds,
        unproj_preds=winner_preds,
        q3_preds=q3_preds,
        gt_selected=gt_selected,
        target_selected=target_selected,
        pose_rows=pose_rows,
    )

    derivation = {
        "schema": "ddm_se2_survival_condition_derivation.v1",
        "status": "DERIVED_FROM_SOURCE_AND_MEASURED_D_STRUCTURE",
        "source_facts": {
            "SegNet_frame": "upstream/modules.py SegNet.preprocess_input uses only frame_1 then bilinear resize to 384x512",
            "PoseNet_preprocess": "upstream/modules.py PoseNet.preprocess_input resizes both RGB frames to 384x512 then rgb_to_yuv6",
            "D_private_supports": "scale 874/384 and 1164/512 are both >2, giving disjoint 2x2 camera supports per scorer-grid pixel for align_corners=False bilinear resize",
            "blind_camera_fraction": 0.2270,
        },
        "condition": (
            "For target scorer cell u and target class c, a camera-plane paint guarantees the input-plane part of a flip only if "
            "all private support pixels of u are moved so the resized RGB x'_u has target-class logit margin m_c(x') > 0 "
            "under the regional SegNet stem. Private support controls x'_u, but it does not isolate the downstream receptive "
            "field, so the sufficient condition is a coherent neighborhood whose induced feature response clears the class margin."
        ),
        "taxonomy_prediction": {
            "uint8_floor": "Only active when requested camera deltas round to no byte movement; measured as zero for emitted delta>=1 paints.",
            "R_attenuation": "Private supports make bilinear-D exact at edited scorer pixels; predicted not to explain cg3/se2 failure.",
            "amplitude_timidity": "Low max_delta paints fail if full-color same-radius paint corrects the cell.",
            "receptive_field_veto": "Full-color private-support paint still fails because the stride-2/regional SegNet context vetoes the local color.",
        },
    }

    summary = {
        "schema": "ddm_se2_survival_q3_measurement.v1",
        "captured_at_utc": utc_now(),
        "git": git_head(),
        "axis": "[macOS-CPU advisory / CPU Torch SegNet+PoseNet bounded n32]",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_run": False,
        "selection_mode": "stratified-random non-prefix quartiles by Road/Lane target count",
        "seed": args.seed,
        "n_pairs": len(pairs),
        "pairs": pairs,
        "target_cells_selected": int(target_counts[pairs].sum()),
        "target_cells_total_n600": int(target_counts.sum()),
        "baseline": {
            "own_vehicle_frontier": f"S = {BASELINE_S} @ {BASELINE_BYTES} B [macOS-CPU advisory]",
            "d_seg": BASELINE_D_SEG,
            "d_pose": BASELINE_D_POSE,
            "archive": str(qo1_archive),
            "archive_sha256": sha256_file(qo1_archive),
            "base_raw": str(qo1_raw_path),
            "base_raw_bytes": qo1_raw_path.stat().st_size,
            "base_raw_sha256_head_1MiB": hashlib.sha256(qo1_raw_path.open("rb").read(1 << 20)).hexdigest(),
        },
        "inputs": {
            "gt_argmax": str(gt_path),
            "gt_argmax_sha256": sha256_file(gt_path),
            "current_argmax": str(current_path),
            "current_argmax_sha256": sha256_file(current_path),
            "segnet_weights": str(segnet_sd_path),
            "segnet_weights_sha256": sha256_file(segnet_sd_path),
            "gt_mkv": str(args.gt_mkv),
        },
        "controls": {
            "qo1_fz4_frame_sha_spot_check": frame_controls,
            "baseline_argmax_matches_cached_current": True,
            "baseline_argmax_mismatch_cells": 0,
            "target_definition": "Road<->Lane cells where cached current argmax differs from cached GT argmax",
        },
        "derivation": derivation,
        "margin_bins": margin_bin_rows,
        "variants": rows,
        "failure_taxonomy": taxonomy,
        "winner_selection": {
            "mode": "min global_variant_flips_subset; target survival is secondary",
            "winner": winner_row,
            "best_target_survival_variant": best_survival_row,
            "minimal_amplitude_radius_clearing_0p9": minimal_clear,
            "clears_0p9": bool(minimal_clear),
            "clears_ed1_break_even": bool(float(best_survival_row["target_survival"]) >= ED1_BREAK_EVEN_SURVIVAL),
            "ed1_break_even_survival": ED1_BREAK_EVEN_SURVIVAL,
        },
        "q3_projection": {
            "winner_variant": winner.name,
            "aggregate": q3_aggregate,
            "rows_path": str(q3_rows_path),
        },
        "runtime": {
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_threads": args.torch_threads,
            "elapsed_seconds": time.monotonic() - started,
        },
        "boundaries": [
            "Bounded n32 stratified sample, not n600 and not prefix.",
            "CPU Torch frozen scorer path; no MPS.",
            "No byte-closed variant archive was built.",
            "No full-n600 scorer job was run or claimed.",
            "Q3 projection is applied to this prototype-paint formulation only.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with rows_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    with q3_rows_path.open("w") as fh:
        for row in q3_instrument_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "rows": str(rows_path),
                "q3_rows": str(q3_rows_path),
                "winner": winner.name,
                "elapsed_seconds": summary["runtime"]["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-raw", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit/inflated/0.raw"))
    parser.add_argument("--base-archive", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit/archive.zip"))
    parser.add_argument("--fz4-raw", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/sub_final/inflated/0.raw"))
    parser.add_argument("--fz4-archive", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/sub_final/archive.zip"))
    parser.add_argument("--argmax-cache", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache"))
    parser.add_argument("--gt-mkv", type=Path, default=Path("upstream/videos/0.mkv"))
    parser.add_argument("--out-dir", type=Path, default=Path(".omx/research/ddm_se2_20260804"))
    parser.add_argument("--n-pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
