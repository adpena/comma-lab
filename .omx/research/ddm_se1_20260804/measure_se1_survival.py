#!/usr/bin/env python3
"""SE1 bounded Road/Lane paint survival probe.

This is receipt code, not a production receiver. It measures bounded SegNet
survival on the fz4/fz1 sub_final decode without running a full n600 scorer
job. The base raw is memory-mapped in place and never copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
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
for path in (REPO, REPO / "src", UPSTREAM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
BASELINE_S = 0.7541459
BASELINE_BYTES = 358_084
BASELINE_D_SEG = 0.00431179
BASELINE_D_POSE = 0.0007145917
PR130_BAR = 0.1721417
ARCHIVE_SHA256 = "ad5dd0e4fbe5b13ab53a5995a6d77cc558c25f40b63f894ea50ad336bd50fb66"


@dataclass(frozen=True)
class Variant:
    name: str
    max_delta: int | None
    coherence_radius: int


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


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
        remaining = np.array([p for p in nonzero if int(p) not in set(selected)], dtype=np.int64)
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


def predict_segnet(model: SegNet, batch_np: np.ndarray, *, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    preds: list[np.ndarray] = []
    target_logits: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, batch_np.shape[0], batch_size):
            chunk = batch_np[start : start + batch_size]
            batch = torch.from_numpy(np.ascontiguousarray(chunk))
            x = batch.permute(0, 1, 4, 2, 3).float()
            seg_input = model.preprocess_input(x)
            logits = model(seg_input)
            preds.append(logits.argmax(dim=1).cpu().numpy().astype(np.uint8))
            target_logits.append(logits.cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0), np.concatenate(target_logits, axis=0)


def move_toward(current: np.ndarray, target: np.ndarray, max_delta: int | None) -> np.ndarray:
    if max_delta is None:
        return np.broadcast_to(target, current.shape).astype(np.uint8).copy()
    cur = current.astype(np.int16)
    tgt = np.broadcast_to(target, current.shape).astype(np.int16)
    delta = np.clip(tgt - cur, -int(max_delta), int(max_delta))
    return np.clip(cur + delta, 0, 255).astype(np.uint8)


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
) -> tuple[np.ndarray, dict[str, int]]:
    out = base.copy()
    changed = 0
    writes = 0
    touched_cells = 0
    for batch_index, pair in enumerate(pairs):
        rows, cols = np.nonzero(target_mask[pair])
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
    return out, {
        "camera_support_writes": writes,
        "camera_channel_values_changed": changed,
        "scorer_grid_cells_touched_with_duplicates": touched_cells,
    }


def target_margin_quartiles(logits: np.ndarray, pairs: list[int], gt: np.ndarray, target_mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    margins: list[float] = []
    for batch_index, pair in enumerate(pairs):
        rows, cols = np.nonzero(target_mask[pair])
        for row, col in zip(rows.tolist(), cols.tolist()):
            target_class = int(gt[pair, row, col])
            values = logits[batch_index, :, row, col].astype(np.float64)
            rival_values = values.copy()
            rival_values[target_class] = -np.inf
            margins.append(float(values[target_class] - np.max(rival_values)))
    arr = np.asarray(margins, dtype=np.float64)
    qs = np.quantile(arr, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins: list[dict[str, Any]] = []
    for idx in range(4):
        lo, hi = float(qs[idx]), float(qs[idx + 1])
        bins.append({"bin": idx, "lo": lo, "hi": hi, "count": int(((arr >= lo) & (arr <= hi)).sum())})
    return arr, bins


def summarize_variant(
    *,
    name: str,
    preds: np.ndarray,
    baseline_preds: np.ndarray,
    gt_selected: np.ndarray,
    target_masks_selected: np.ndarray,
    baseline_margin_values: np.ndarray,
    margin_edges: np.ndarray,
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
    rows: dict[str, Any] = {
        "variant": name,
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
        "verdict_scope": "SegNet-only bounded n32 stratified selected fz4 sub_final; not pose, not n600, not contest authority",
    }
    flat_corrected = corrected[target]
    by_bin = []
    offset = 0
    for idx in range(4):
        lo, hi = margin_edges[idx], margin_edges[idx + 1]
        if idx == 3:
            mask = (baseline_margin_values >= lo) & (baseline_margin_values <= hi)
        else:
            mask = (baseline_margin_values >= lo) & (baseline_margin_values < hi)
        count = int(mask.sum())
        corr = int(flat_corrected[offset : offset + baseline_margin_values.size][mask].sum())
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
    rows["by_baseline_margin_quartile"] = by_bin
    return rows


def execute(args: argparse.Namespace) -> int:
    started = time.monotonic()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "se1_survival_summary.json"
    rows_path = out_dir / "se1_survival_rows.jsonl"

    raw_path = args.base_raw.resolve(strict=True)
    gt_path = args.argmax_cache / "gt_argmax_n600.npy"
    current_path = args.argmax_cache / "cx1_argmax_n600.npy"
    archive_path = args.base_archive.resolve(strict=True)
    if sha256_file(archive_path) != ARCHIVE_SHA256:
        raise RuntimeError("base archive SHA drifted; refusing matched-base survival measurement")

    gt = np.load(gt_path, mmap_mode="r")
    current = np.load(current_path, mmap_mode="r")
    if gt.shape != (PAIR_COUNT, SEG_H, SEG_W) or current.shape != gt.shape:
        raise RuntimeError(f"argmax shape drift: gt={gt.shape} current={current.shape}")
    target_mask = correction_target(gt, current)
    target_counts = target_mask.reshape(PAIR_COUNT, -1).sum(axis=1)
    pairs = stratified_pairs(target_counts, n=args.n_pairs, seed=args.seed)
    raw = load_raw_pairs(raw_path)
    base_batch = gather_pairs(raw, pairs)

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
            f"matched-base control failed: SegNet(base raw) differs from cx1_argmax at {int(baseline_mismatch.sum())} cells"
        )

    baseline_margin_values, margin_bins = target_margin_quartiles(baseline_logits, pairs, gt, target_mask)
    margin_edges = np.quantile(baseline_margin_values, [0.0, 0.25, 0.5, 0.75, 1.0])
    rlo, rhi, _rw = f0pr_bilinear_axis(SEG_H, CAMERA_H)
    clo, chi, _cw = f0pr_bilinear_axis(SEG_W, CAMERA_W)
    variants = [
        Variant("private_delta_1", 1, 0),
        Variant("private_delta_2", 2, 0),
        Variant("private_delta_4", 4, 0),
        Variant("private_delta_8", 8, 0),
        Variant("private_delta_16", 16, 0),
        Variant("private_delta_32", 32, 0),
        Variant("private_delta_64", 64, 0),
        Variant("private_delta_128", 128, 0),
        Variant("ed1_private_full_color", None, 0),
        Variant("region_r1_full_color", None, 1),
        Variant("region_r2_full_color", None, 2),
    ]

    rows: list[dict[str, Any]] = []
    for variant in variants:
        mutated, mutation_stats = apply_variant(
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
        row = summarize_variant(
            name=variant.name,
            preds=preds,
            baseline_preds=baseline_preds,
            gt_selected=gt_selected,
            target_masks_selected=target_selected,
            baseline_margin_values=baseline_margin_values,
            margin_edges=margin_edges,
            mutation_stats=mutation_stats,
        )
        row["max_delta"] = variant.max_delta
        row["coherence_radius"] = variant.coherence_radius
        rows.append(row)

    by_name = {row["variant"]: row for row in rows}
    full_surv = by_name["ed1_private_full_color"]["target_survival"]
    r1_surv = by_name["region_r1_full_color"]["target_survival"]
    r2_surv = by_name["region_r2_full_color"]["target_survival"]
    taxonomy = {
        "private_full_color_survival": full_surv,
        "region_r1_full_color_survival": r1_surv,
        "region_r2_full_color_survival": r2_surv,
        "killer_assignment": (
            "region_coupling_dominant"
            if r1_surv > full_surv * 1.25
            else "amplitude_or_color_instrument_dominant"
            if full_surv < 0.25
            else "private_support_partially_effective"
        ),
        "uint8_amplitude_floor": "delta_1 changes support bytes; failure at delta_1 is not a no-emission claim",
        "r_attenuation_vs_segnet_region_coupling": (
            "The private support makes the target scorer input move, but the SegNet prediction can still be vetoed by "
            "the stride-2 stem/receptive-field context; region_r1/r2 separates that from pure amplitude."
        ),
    }
    target_cells = int(target_counts[pairs].sum())
    summary = {
        "schema": "ddm_se1_survival_measurement.v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "axis": "[macOS-CPU advisory / CPU Torch SegNet-only]",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_run": False,
        "selection_mode": "stratified-random non-prefix quartiles by Road/Lane target count",
        "seed": args.seed,
        "n_pairs": len(pairs),
        "pairs": pairs,
        "target_cells_selected": target_cells,
        "target_cells_total_n600": int(target_counts.sum()),
        "baseline": {
            "own_vehicle_frontier": f"S = {BASELINE_S} @ {BASELINE_BYTES} B [macOS-CPU advisory]",
            "d_seg": BASELINE_D_SEG,
            "d_pose": BASELINE_D_POSE,
            "archive": str(archive_path),
            "archive_sha256": sha256_file(archive_path),
            "base_raw": str(raw_path),
            "base_raw_bytes": raw_path.stat().st_size,
            "base_raw_sha256_head_1MiB": hashlib.sha256(raw_path.open("rb").read(1 << 20)).hexdigest(),
        },
        "inputs": {
            "gt_argmax": str(gt_path),
            "gt_argmax_sha256": sha256_file(gt_path),
            "current_argmax": str(current_path),
            "current_argmax_sha256": sha256_file(current_path),
            "segnet_weights": str(segnet_sd_path),
            "segnet_weights_sha256": sha256_file(segnet_sd_path),
        },
        "controls": {
            "baseline_argmax_matches_cached_current": True,
            "baseline_argmax_mismatch_cells": 0,
            "target_definition": "Road<->Lane cells where cached current argmax differs from cached GT argmax",
            "segnet_reads_frame": "frame_1 only per upstream/modules.py SegNet.preprocess_input",
        },
        "margin_bins": margin_bins,
        "variants": rows,
        "taxonomy": taxonomy,
        "runtime": {
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_threads": args.torch_threads,
            "elapsed_seconds": time.monotonic() - started,
        },
        "boundaries": [
            "SegNet-only: PoseNet was not run, so this is not a contest score or net-S verdict.",
            "n32 stratified random, not n600 and not prefix.",
            "Region-coherent paints use square scorer-grid neighborhoods; they are a coherence instrument, not a shipped receiver.",
            "No variant archive was built; bytes are priced separately in the description leg.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with rows_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"summary": str(summary_path), "rows": str(rows_path), "elapsed_seconds": summary["runtime"]["elapsed_seconds"]}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-raw", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/sub_final/inflated/0.raw"))
    parser.add_argument("--base-archive", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/sub_final/archive.zip"))
    parser.add_argument("--argmax-cache", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache"))
    parser.add_argument("--out-dir", type=Path, default=Path(".omx/research/ddm_se1_20260804"))
    parser.add_argument("--n-pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
