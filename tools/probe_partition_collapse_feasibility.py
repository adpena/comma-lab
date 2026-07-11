#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""v8 rate-half feasibility probe: does the SegNet argmax partition collapse?

Measures, on the REAL cached SegNet argmax labels (numpy load — NO scorer
forward, NO model inference), per class and per technique:

1. GLOBAL Laguerre/power-diagram K sweep (greedy error-driven + tau-relaxed
   CE weight refinement)  -> the "one power diagram for everything" curve.
2. PER-CLASS geometry-matched carriers: Road/Undriv small-K power diagram on
   the inpainted base; Lane polynomial curve + width band + dash occupancy
   models (per-run vs periodic-in-image vs periodic-in-ego-distance); Movable
   moment ellipses; MyCar static temporal-majority mask.
3. "Store the boundaries" comparators: crack-edge contour byte floor
   (~1.25 b/px, #307) and zlib label-map bytes.
4. UNION HYBRID: compose the per-class carriers, measure the honest residual.

Output: JSON + printed tables. ADVISORY geometric feasibility only
([macOS-MLX advisory]-class): fidelity is measured against the cached argmax,
not through R + the frozen SegNet, and no byte-closed archive is produced.

Usage:
  .venv/bin/python tools/probe_partition_collapse_feasibility.py \
      --cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
      --sweep-stride 50 --hybrid-stride 25 --k-max 1024 \
      --out experiments/results/partition_collapse_probe_<utc>/probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.boundary_math.partition_collapse import (  # noqa: E402
    CLASS_NAMES,
    LANE,
    MYCAR,
    NUM_CLASSES,
    UNDRIV,
    boundary_band_mask,
    contour_stats,
    dash_occupancy_models,
    evaluate_partition,
    fit_lane_curves,
    fit_movable_ellipses,
    fit_power_diagram_greedy,
    generator_bytes,
    lane_curve_bytes,
    movable_bytes,
    power_assign,
    refine_weights_ce,
    render_lane_band,
    static_mask_bytes,
    static_mycar_mask,
    union_hybrid_reconstruction,
    zlib_label_bytes,
)

N600_FRAMES = 600  # amortization horizon for static carriers


def _load_labels(cache: Path, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Load the cached argmax label stack (memory-light: cast to uint8, free int64)."""
    with np.load(cache) as z:
        lstars = z["lstars"]
        labels = lstars.astype(np.uint8)
        del lstars
    frames = np.arange(0, labels.shape[0], stride)
    return labels, frames


def _self_detect_classes(labels: np.ndarray) -> dict:
    """Verify the canonical class order by spatial/static signature (never re-derive)."""
    n, h, _w = labels.shape
    out: dict = {}
    areas = np.stack([(labels == c).mean(axis=(1, 2)) for c in range(NUM_CLASSES)])
    centroids = []
    for c in range(NUM_CLASSES):
        m = labels == c
        rows = np.argwhere(m.any(axis=2))
        centroids.append(float(rows[:, 1].mean()) if len(rows) else float("nan"))
    m4 = labels == MYCAR
    inter = (m4[:-1] & m4[1:]).sum()
    union = (m4[:-1] | m4[1:]).sum()
    out["mycar_temporal_iou"] = float(inter / union) if union else 0.0
    out["mycar_centroid_row"] = centroids[MYCAR]
    out["undriv_centroid_row"] = centroids[UNDRIV]
    out["lane_area_frac"] = float(areas[LANE].mean())
    out["areas"] = {CLASS_NAMES[c]: float(areas[c].mean()) for c in range(NUM_CLASSES)}
    out["signature_ok"] = bool(
        out["mycar_temporal_iou"] > 0.9
        and out["mycar_centroid_row"] > 0.6 * h
        and out["undriv_centroid_row"] < 0.5 * h
        and out["lane_area_frac"] < 0.03
    )
    return out


def _mean_over(rows: list[dict], key_path: tuple) -> float:
    vals = []
    for r in rows:
        v = r
        for k in key_path:
            v = v[k]
        vals.append(v)
    return float(np.mean(vals))


def run_global_sweep(
    labels: np.ndarray,
    frames: np.ndarray,
    *,
    k_max: int,
    checkpoints: tuple[int, ...],
    refine_at: tuple[int, ...],
) -> dict:
    per_frame: list[dict] = []
    for fi in frames:
        gt = labels[fi].astype(np.int32)
        t0 = time.time()
        _, curve = fit_power_diagram_greedy(
            gt, k_max, checkpoints=checkpoints, keep_snapshots=True
        )
        band = boundary_band_mask(gt)
        for row in curve:
            snap = row.pop("snapshot", None)
            if snap is not None and (
                refine_at == "all"
                or any(abs(row["k"] - r) <= max(2, r // 8) for r in refine_at)
            ):
                refined = refine_weights_ce(
                    snap, gt, steps=150, n_sample=12000, tau=25.0,
                    optimize_positions=True, seed=0,
                )
                _, pred_r = power_assign(refined, gt.shape)
                m = evaluate_partition(pred_r, gt, band)
                row["refined_bulk_agreement"] = m["bulk_agreement"]
                row["refined_band_agreement"] = m["band_agreement"]
                row["refined_per_class"] = m["per_class"]
            row["gen_bytes"] = generator_bytes(row["k"])
        per_frame.append(
            {"frame": int(fi), "curve": curve, "fit_seconds": round(time.time() - t0, 1)}
        )
        print(
            f"  frame {fi}: K-sweep done in {per_frame[-1]['fit_seconds']}s; "
            f"final bulk={curve[-1]['bulk_agreement']:.4f} band={curve[-1]['band_agreement']:.4f}",
            flush=True,
        )
    # aggregate by checkpoint index (frames share the checkpoint ladder)
    n_ck = min(len(pf["curve"]) for pf in per_frame)
    agg = []
    for i in range(n_ck):
        rows = [pf["curve"][i] for pf in per_frame]
        entry = {
            "k_mean": float(np.mean([r["k"] for r in rows])),
            "gen_bytes_mean": float(np.mean([r["gen_bytes"] for r in rows])),
            "bulk_agreement": _mean_over(rows, ("bulk_agreement",)),
            "band_agreement": _mean_over(rows, ("band_agreement",)),
            "disagreement": _mean_over(rows, ("disagreement",)),
            "per_class_recall": {
                CLASS_NAMES[c]: float(
                    np.mean(
                        [
                            r["per_class"][CLASS_NAMES[c]].get("recall", np.nan)
                            for r in rows
                        ]
                    )
                )
                for c in range(NUM_CLASSES)
            },
            "k_per_class_mean": np.mean(
                [r["k_per_class"] for r in rows], axis=0
            ).tolist(),
        }
        if all("refined_bulk_agreement" in r for r in rows):
            entry["refined_bulk_agreement"] = _mean_over(rows, ("refined_bulk_agreement",))
            entry["refined_band_agreement"] = _mean_over(rows, ("refined_band_agreement",))
        agg.append(entry)
    return {"per_frame": per_frame, "aggregate": agg}


def run_perclass_and_hybrid(
    labels: np.ndarray, frames: np.ndarray, *, ru_k: int
) -> dict:
    stack = labels[frames].astype(np.int32)
    mycar = static_mycar_mask(stack)
    mycar_bytes = static_mask_bytes(mycar)
    hybrid_rows: list[dict] = []
    lane_rows: list[dict] = []
    dash_rows: list[dict] = []
    contour_rows: list[dict] = []
    for j, fi in enumerate(frames):
        gt = stack[j]
        st = contour_stats(gt)
        st["zlib_bytes"] = zlib_label_bytes(gt)
        st["frame"] = int(fi)
        contour_rows.append(st)

        curves, lane_uncov = fit_lane_curves(gt)
        band_mask = render_lane_band(curves, gt.shape)
        lane_gt = gt == LANE
        n_lane = int(lane_gt.sum())
        lane_rows.append(
            {
                "frame": int(fi),
                "n_curves": len(curves),
                "lane_px": n_lane,
                "uncovered_px": lane_uncov,
                "recall": float((band_mask & lane_gt).sum() / n_lane) if n_lane else 1.0,
                "precision": float((band_mask & lane_gt).sum() / max(1, band_mask.sum())),
                "bytes_solid": lane_curve_bytes(curves, model="solid"),
                "bytes_runs": lane_curve_bytes(curves, model="runs"),
                "bytes_periodic": lane_curve_bytes(curves, model="periodic"),
            }
        )
        for cv in curves:
            dm = dash_occupancy_models(cv)
            if not dm.get("skipped", True) and not dm.get("solid", False):
                dm["frame"] = int(fi)
                dash_rows.append(dm)

        out = union_hybrid_reconstruction(gt, ru_k=ru_k, mycar_mask=mycar)
        ellipses, _ = fit_movable_ellipses(gt)
        bb = out["bytes_breakdown"]
        bb["mycar_static_amortized_600"] = mycar_bytes / N600_FRAMES
        bb["movable_ellipses"] = movable_bytes(len(ellipses))
        total = (
            bb["road_undriv_generators"]
            + bb["lane_curves_runs"]
            + bb["movable_ellipses"]
            + bb["mycar_static_amortized_600"]
        )
        hybrid_rows.append(
            {
                "frame": int(fi),
                "metrics": {k: v for k, v in out["metrics"].items()},
                "ru_k": out["ru_k"],
                "n_lane_curves": out["n_lane_curves"],
                "n_movable_islands": out["n_movable_islands"],
                "bytes_breakdown": bb,
                "bytes_total_per_frame": total,
            }
        )
        print(
            f"  frame {fi}: hybrid bulk={out['metrics']['bulk_agreement']:.4f} "
            f"band={out['metrics']['band_agreement']:.4f} bytes/frame={total:.0f}",
            flush=True,
        )
    # mycar per-frame fidelity of the static mask
    mycar_metrics = []
    for j in range(len(frames)):
        m = stack[j] == MYCAR
        inter = int((m & mycar).sum())
        union = int((m | mycar).sum())
        mycar_metrics.append(inter / union if union else 1.0)
    return {
        "mycar": {
            "static_mask_bytes": mycar_bytes,
            "amortized_bytes_per_frame_600": mycar_bytes / N600_FRAMES,
            "iou_mean": float(np.mean(mycar_metrics)),
            "iou_min": float(np.min(mycar_metrics)),
        },
        "lane": lane_rows,
        "dashes": dash_rows,
        "contours": contour_rows,
        "hybrid": hybrid_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache",
        default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        help="cached gt argmax npz (lstars member)",
    )
    ap.add_argument("--sweep-stride", type=int, default=50)
    ap.add_argument("--hybrid-stride", type=int, default=25)
    ap.add_argument("--k-max", type=int, default=1024)
    ap.add_argument("--ru-k", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        Path(args.out)
        if args.out
        else REPO / "experiments" / "results" / f"partition_collapse_probe_{utc}" / "probe.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache = Path(args.cache)
    print(f"loading {cache} ...", flush=True)
    labels, _ = _load_labels(cache, args.sweep_stride)
    n = labels.shape[0]
    sweep_frames = np.arange(0, n, args.sweep_stride)
    hybrid_frames = np.arange(0, n, args.hybrid_stride)
    print(
        f"labels {labels.shape}; sweep frames {len(sweep_frames)}; "
        f"hybrid frames {len(hybrid_frames)}",
        flush=True,
    )

    sig = _self_detect_classes(labels[hybrid_frames])
    print(f"class-signature self-check: {sig}", flush=True)
    if not sig["signature_ok"]:
        print("WARNING: class signatures do not match the canonical order!", flush=True)

    checkpoints = tuple(
        k for k in (8, 16, 32, 64, 128, 256, 512, 1024) if k <= args.k_max
    )
    print("=== global Laguerre K sweep ===", flush=True)
    sweep = run_global_sweep(
        labels,
        sweep_frames,
        k_max=args.k_max,
        checkpoints=checkpoints,
        refine_at="all",
    )
    print("=== per-class carriers + union hybrid ===", flush=True)
    perclass = run_perclass_and_hybrid(labels, hybrid_frames, ru_k=args.ru_k)
    ru_sweep = []
    for ru_k in (8, 16, 64):
        if ru_k == args.ru_k:
            continue
        rows = []
        for fi in hybrid_frames[:: max(1, len(hybrid_frames) // 8)]:
            gt = labels[fi].astype(np.int32)
            out = union_hybrid_reconstruction(gt, ru_k=ru_k)
            rows.append(
                {
                    "frame": int(fi),
                    "bulk": out["metrics"]["bulk_agreement"],
                    "band": out["metrics"]["band_agreement"],
                    "ru_k": out["ru_k"],
                }
            )
        ru_sweep.append(
            {
                "ru_k": ru_k,
                "bulk_mean": float(np.mean([r["bulk"] for r in rows])),
                "band_mean": float(np.mean([r["band"] for r in rows])),
                "rows": rows,
            }
        )
        print(
            f"  ru_k={ru_k}: hybrid bulk={ru_sweep[-1]['bulk_mean']:.4f} "
            f"band={ru_sweep[-1]['band_mean']:.4f}",
            flush=True,
        )

    result = {
        "generated_at": utc,
        "cache": str(cache),
        "n_frames_total": int(n),
        "sweep_frames": sweep_frames.tolist(),
        "hybrid_frames": hybrid_frames.tolist(),
        "class_signature": sig,
        "global_sweep": {
            "aggregate": sweep["aggregate"],
            "per_frame": [
                {"frame": pf["frame"], "fit_seconds": pf["fit_seconds"], "curve": pf["curve"]}
                for pf in sweep["per_frame"]
            ],
        },
        "perclass": perclass,
        "ru_k_sweep": ru_sweep,
        "authority": "[macOS-MLX advisory] geometric feasibility vs cached argmax; "
        "NOT byte-closed, NOT through R+SegNet",
    }
    out_path.write_text(json.dumps(result, indent=1, default=float))
    print(f"\nwrote {out_path}", flush=True)

    print("\n=== SUMMARY: global Laguerre curve (mean over frames) ===")
    print("K      bytes/f  bulk     band     Road    Lane    Undriv  Movable MyCar")
    for e in sweep["aggregate"]:
        pc = e["per_class_recall"]
        ref = (
            f"  (refined: bulk {e['refined_bulk_agreement']:.4f} band {e['refined_band_agreement']:.4f})"
            if "refined_bulk_agreement" in e
            else ""
        )
        print(
            f"{e['k_mean']:6.0f} {e['gen_bytes_mean']:7.0f}  {e['bulk_agreement']:.4f}  "
            f"{e['band_agreement']:.4f}  "
            + " ".join(f"{pc[CLASS_NAMES[c]]:.3f}" for c in range(NUM_CLASSES))
            + ref
        )
    h = perclass["hybrid"]
    print("\n=== SUMMARY: union hybrid (mean over frames) ===")
    print(
        f"bulk={np.mean([r['metrics']['bulk_agreement'] for r in h]):.4f} "
        f"band={np.mean([r['metrics']['band_agreement'] for r in h]):.4f} "
        f"bytes/frame={np.mean([r['bytes_total_per_frame'] for r in h]):.0f}"
    )
    ct = perclass["contours"]
    print(
        f"contour floor bytes/frame={np.mean([r['bytes_floor'] for r in ct]):.0f} "
        f"zlib bytes/frame={np.mean([r['zlib_bytes'] for r in ct]):.0f} "
        f"edge px/frame={np.mean([r['edge_px'] for r in ct]):.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
