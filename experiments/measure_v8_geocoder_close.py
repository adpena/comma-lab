# SPDX-License-Identifier: MIT
"""measure_v8_geocoder_close — the #394 UNIT A through-R measurement driver.

Composes the v8 Road/Lane texture generator (:mod:`tac.through_r.roadlane_texture_generator`) +
Movable sparse-site carrier (:mod:`tac.boundary_math.movable_site_coder`) and MEASURES the
composed generator through the real R + frozen CPU-torch SegNet vs the flat-paint floor (0.0416)
and the trained witness (0.0048), per class. Emits typed :class:`tac.verdicts.MeasurementRow` rows
+ a JSON artifact.

n600 discipline: ``--n 600`` is the authority; a subset is LABELLED non-authority (provisional
prior). Heavy n600 legs go through the governor (``tools/safe_run.py``); a REFUSE is information.

Usage (governed n600 authority leg):
    TAC_GOVERNED_ADMISSION=1 .venv/bin/python tools/safe_run.py \
        --rationale "#394 v8 geocoder close: composed texture generator through-R n600" -- \
        .venv/bin/python experiments/measure_v8_geocoder_close.py --n 600 \
        --out experiments/results/v8_geocoder_close_n600

Usage (ungoverned n96 provisional prior, labelled non-authority):
    .venv/bin/python experiments/measure_v8_geocoder_close.py --n 96 --subset-reason \
        "provisional prior; owed-16 A/B owns machine; n600 queued" \
        --out experiments/results/v8_geocoder_close_n96
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# Reference floors (per-video, MEASURED, cited — NOT re-derived here).
FLAT_PAINT_FLOOR = 0.0416       # best flat palette per class through R (segnet_texture_perception)
WITNESS_DSEG = 0.0048           # trained mod32cap witness EMA-BEST (levelset_n600_witness_mod32cap)
FLOOR_PROVENANCE = (
    "flat-paint 0.0416 + witness 0.0048: .omx/research/segnet_texture_perception_20260710.md "
    "(MEASURED, cited)"
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _scene_mean_palette(gt_cache: str, lstars: np.ndarray, *, n: int, n_classes: int = 5) -> np.ndarray:
    """Per-class GLOBAL mean RGB (n_classes,3) from gt_f1, R-consistent bilinear DOWN to seg-res.

    Streamed frame-by-frame (mmap) so n600 does not hold 1.8 GB of camera frames. Each gt_f1[i]
    (camera 874x1164) is bilinear-downsampled to seg-res (SEG_H,SEG_W) — the R DOWN leg the scorer
    applies — then per-class pixel sums accumulate over L*. The result is the REAL scene colour per
    class (what survives SegNet's context; the flat-paint-floor colours).
    """
    import torch

    from tac.through_r.resolution_chain import SEG_H, SEG_W

    z = np.load(gt_cache, mmap_mode="r")
    if "gt_f1" not in z.files:
        raise ValueError(f"{gt_cache} has no gt_f1; cannot fit scene colours")
    gt_f1 = z["gt_f1"]
    acc = np.zeros((n_classes, 3), dtype=np.float64)
    cnt = np.zeros((n_classes,), dtype=np.int64)
    with torch.inference_mode():
        for i in range(n):
            cam = np.asarray(gt_f1[i]).astype(np.float32)  # (874,1164,3)
            x = torch.from_numpy(cam).permute(2, 0, 1)[None]  # (1,3,H,W)
            rg = (
                torch.nn.functional.interpolate(
                    x, size=(SEG_H, SEG_W), mode="bilinear", align_corners=False
                )[0]
                .permute(1, 2, 0)
                .numpy()
            )  # (SEG_H,SEG_W,3) float
            lab = lstars[i]
            for c in range(n_classes):
                m = lab == c
                k = int(m.sum())
                if k:
                    acc[c] += rg[m].sum(axis=0)
                    cnt[c] += k
    out = np.zeros((n_classes, 3), dtype=np.float64)
    nz = cnt > 0
    out[nz] = acc[nz] / cnt[nz, None]
    return out


def _gt_cache_for(n: int) -> str:
    exact = f"experiments/results/mlx_fleet_gt_cache/gt_n{n}.npz"
    if Path(exact).exists():
        return exact
    # fall back to the n600 cache truncated (harness/run_arm take the first N).
    return "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=600, help="pairs (600 = authority)")
    ap.add_argument("--subset-reason", type=str, default="", help="required if --n != 600")
    ap.add_argument("--out", type=str, required=True, help="output dir")
    ap.add_argument("--verdict-batch", type=int, default=32)
    ap.add_argument(
        "--grating-orientation", type=float, default=0.0,
        help="stripe normal (deg); 0 -> vertical stripes (varies along x)",
    )
    ap.add_argument(
        "--review-status", type=str, default="unreviewed_recovery_written",
        help="reviewed|unreviewed_recovery_written|provisional",
    )
    ap.add_argument(
        "--rerun-price-list", action="store_true",
        help="re-run the 568-tile through-R price-list sweep (SLOW; already landed in "
        "stem_perception + segnet_texture_perception memo — default OFF, cite it).",
    )
    ns = ap.parse_args(argv)

    n = int(ns.n)
    if n != 600 and not ns.subset_reason.strip():
        ap.error("--n != 600 requires --subset-reason (n600 discipline)")

    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)

    from tac.boundary_math.movable_site_coder import (
        MOVABLE_CLASS,
        byte_account_sites,
        extract_movable_sites,
        render_sites_to_mask,
        track_sites,
    )
    from tac.through_r.harness import load_gt_lstars
    from tac.through_r.palette_realization import load_frozen_segnet
    from tac.through_r.roadlane_texture_generator import (
        default_roadlane_grating_specs,
        plan_from_palette,
        run_composed_generator_arm,
    )
    gt_cache = _gt_cache_for(n)
    lstars = load_gt_lstars(gt_cache, n=n)
    print(f"[geocoder] loaded L* {lstars.shape} from {gt_cache}", flush=True)

    segnet = load_frozen_segnet("cpu-torch")
    print("[geocoder] frozen SegNet loaded (cpu-torch)", flush=True)

    # The price list (period-4 grating winner + flat-basin winners) is ALREADY LANDED + MEASURED
    # (stem_perception + segnet_texture_perception_20260710). Default: cite it via the landed
    # default gratings + decision-geometry basin colours (fast). --rerun-price-list re-sweeps the
    # 568-tile through-R search (slow; only for a fresh price-list re-derivation).
    price = None
    if ns.rerun_price_list:
        from tac.through_r.stem_perception import per_class_price_list

        price = per_class_price_list(segnet, through_R=True)
        print("[geocoder] price list per-class cheapest (re-swept):", flush=True)
        for nm, pt in price.per_class_cheapest.items():
            tag = (
                f"{pt.spec.family:8s} bits={pt.bits} margin={pt.margin:+.3f}"
                if pt is not None else "(no winner)"
            )
            print(f"    {nm:12s} {tag}", flush=True)

    # SCENE-MEAN palette (the HONEST baseline colours — what actually survives SegNet's context, per
    # the 0.0416 flat-paint floor; decision-geometry context-free colours do NOT survive composition).
    # Streamed from gt_f1 (camera) -> R-consistent bilinear DOWN to seg-res -> per-class global mean.
    scene_palette = _scene_mean_palette(gt_cache, lstars, n=n)
    names5 = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
    print("[geocoder] scene-mean palette (per-class RGB):", flush=True)
    for c in range(5):
        print(f"    {names5[c]:12s} {tuple(round(float(v), 1) for v in scene_palette[c])}", flush=True)

    grat = default_roadlane_grating_specs(orientation=float(ns.grating_orientation))
    # baseline: ALL classes scene-mean flat (reproduce the flat-paint floor in-harness).
    flat_plan = plan_from_palette(scene_palette)
    # texture generator: scene-mean basins + price-list grating for Road/Lane.
    plan = plan_from_palette(scene_palette, grating_specs=grat)
    print(
        f"[geocoder] texture generator rate = {plan.total_texture_bits} bits "
        f"= {plan.total_texture_bytes:.2f} bytes (whole video)",
        flush=True,
    )
    for c, f in sorted(plan.fills.items()):
        print(f"    class {c} {f.class_name:12s} {f.spec.family:8s} bits={f.bits} src={f.source}", flush=True)

    subset = ns.subset_reason.strip() or None

    # Matched in-harness FLAT control (all classes scene-mean flat) — reproduces the flat-paint
    # floor on the SAME frames/SegNet (philosophy P: matched control).
    flat_res = run_composed_generator_arm(
        lstars, plan=flat_plan, segnet=segnet, verdict_batch=int(ns.verdict_batch),
        allow_subset_reason=subset, arm_name="scene_flat_control",
    )
    print(f"[geocoder] FLAT control agg d_seg = {flat_res.agg_dseg:.6f}", flush=True)
    for nm, v in flat_res.per_class_dseg.items():
        print(f"    flat {nm:12s} {v:.6f}", flush=True)

    res = run_composed_generator_arm(
        lstars,
        plan=plan,
        segnet=segnet,
        verdict_batch=int(ns.verdict_batch),
        allow_subset_reason=subset,
    )
    print(f"[geocoder] composed TEXTURE generator agg d_seg = {res.agg_dseg:.6f}", flush=True)
    print("[geocoder] per-class d_seg:", flush=True)
    for nm, v in res.per_class_dseg.items():
        print(f"    tex  {nm:12s} {v:.6f}", flush=True)
    print(
        f"[geocoder] TEXTURE vs matched FLAT control: {res.agg_dseg - flat_res.agg_dseg:+.5f} "
        f"({'texture BEATS flat' if res.agg_dseg < flat_res.agg_dseg else 'texture WORSE than flat'})",
        flush=True,
    )
    print(
        f"[geocoder] vs flat-paint floor {FLAT_PAINT_FLOOR:.4f}: "
        f"{'BEATS' if res.agg_dseg < FLAT_PAINT_FLOOR else 'WORSE'} "
        f"({FLAT_PAINT_FLOOR - res.agg_dseg:+.5f}); vs witness {WITNESS_DSEG:.4f}: "
        f"gap {res.agg_dseg - WITNESS_DSEG:+.5f}",
        flush=True,
    )

    # Movable sparse-site carrier: extract + track + byte-account + coverage tell.
    per_frame_sites = [extract_movable_sites(lstars[i]) for i in range(n)]
    tracked = track_sites(per_frame_sites)
    site_bytes = byte_account_sites(tracked, per_frame_sites)
    # geometry-coverage tell: IoU of site-rendered Movable vs GT Movable, per frame (mean).
    ious = []
    for i in range(n):
        gt_m = lstars[i] == MOVABLE_CLASS
        if not gt_m.any():
            continue
        rm = render_sites_to_mask(per_frame_sites[i])
        inter = int((rm & gt_m).sum())
        union = int((rm | gt_m).sum())
        ious.append(inter / union if union else 1.0)
    mean_iou = float(np.mean(ious)) if ious else float("nan")
    print(
        f"[geocoder] Movable sites: {site_bytes.n_sites_total} sites, K={site_bytes.K}, "
        f"tracked+presence={site_bytes.tracked_bytes + site_bytes.presence_bytes} B "
        f"(raw {site_bytes.raw_perframe_bytes} B), box-IoU vs GT-Movable {mean_iou:.3f}",
        flush=True,
    )

    # Typed measurement rows (aggregate + per-class), with the P2 floor stated.
    git_sha = _git_sha()
    rows = res_to_rows(res, git_sha=git_sha, review_status=ns.review_status, n=n, subset=subset)
    payload = {
        "task": "#394 UNIT A v8 geocoder close",
        "axis_tag": "[through-R]",
        "note": "MEANS; pointer contest-CPU 0.19110 UNMOVED. NON-PROMOTABLE.",
        "n_pairs": n,
        "is_n600": n == 600,
        "subset_reason": subset,
        "gt_cache": gt_cache,
        "git_sha": git_sha,
        "composed_generator_agg_dseg": res.agg_dseg,
        "composed_generator_per_class_dseg": res.per_class_dseg,
        "flat_control_agg_dseg": flat_res.agg_dseg,
        "flat_control_per_class_dseg": flat_res.per_class_dseg,
        "texture_minus_flat_control": res.agg_dseg - flat_res.agg_dseg,
        "flat_paint_floor": FLAT_PAINT_FLOOR,
        "witness_dseg": WITNESS_DSEG,
        "beats_flat_paint_floor": bool(res.agg_dseg < FLAT_PAINT_FLOOR),
        "gap_to_witness": res.agg_dseg - WITNESS_DSEG,
        "texture_rate_bits": plan.total_texture_bits,
        "texture_rate_bytes": plan.total_texture_bytes,
        "fill_plan": {
            str(c): {"class": f.class_name, "family": f.spec.family, "bits": f.bits, "source": f.source}
            for c, f in plan.fills.items()
        },
        "price_list_cheapest": (
            {
                nm: (
                    {"family": pt.spec.family, "bits": pt.bits, "margin": pt.margin}
                    if pt is not None else None
                )
                for nm, pt in price.per_class_cheapest.items()
            }
            if price is not None
            else "cited-not-reswept (stem_perception + segnet_texture_perception_20260710)"
        ),
        "movable_sites": {
            "n_sites_total": site_bytes.n_sites_total,
            "K": site_bytes.K,
            "tracked_plus_presence_bytes": site_bytes.tracked_bytes + site_bytes.presence_bytes,
            "raw_perframe_bytes": site_bytes.raw_perframe_bytes,
            "box_iou_vs_gt_movable": mean_iou,
        },
        "flip_decomposition": {
            "third_class_share": res.decomposition.third_class_share,
            "boundary_share": res.decomposition.boundary_share,
            "per_class_third_share": res.decomposition.per_class_third_share,
        },
        "inputs_sha256": res.inputs_sha256,
        "measurement_rows": [r.to_json_dict() for r in rows],
        "floor_provenance": FLOOR_PROVENANCE,
    }
    out_json = out / f"v8_geocoder_close_n{n}.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[geocoder] wrote {out_json}", flush=True)
    return 0


def res_to_rows(res, *, git_sha, review_status, n, subset):
    """Build aggregate + per-class MeasurementRow list with the P2 floor stated."""
    from tac.verdicts import AxisTag, MeasurementRow, Provenance, ReviewStatus

    prov = Provenance(
        git_sha=git_sha,
        tool="experiments.measure_v8_geocoder_close",
        config_ref="composed_texture_generator",
        inputs_sha256=res.inputs_sha256,
    )
    rs = ReviewStatus.coerce(review_status)
    reason = subset if n != 600 else None
    rows = [
        MeasurementRow(
            value=res.agg_dseg, units="fraction", axis_tag=AxisTag.THROUGH_R, provenance=prov,
            n_samples=n, review_status=rs, n_samples_reason=reason, quantity="d_seg",
            noise_floor=FLAT_PAINT_FLOOR, floor_provenance=FLOOR_PROVENANCE,
        )
    ]
    for nm, v in res.per_class_dseg.items():
        rows.append(
            MeasurementRow(
                value=v, units="fraction", axis_tag=AxisTag.THROUGH_R, provenance=prov,
                n_samples=n, review_status=rs, n_samples_reason=reason, quantity=f"d_seg::{nm}",
            )
        )
    return rows


if __name__ == "__main__":
    sys.exit(main())
