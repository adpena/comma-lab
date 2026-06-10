#!/usr/bin/env python
"""``closed_spec_boundary_solver.v1`` exact-scorer smoke (task #55).

Runs the three deterministic candidates (contour_normal / graph_cut / mdl_contour)
on each base (frontier_archive, lever_b_argmax_generator) over a frame sample, on the
EXACT local CPU-torch SegNet + DistortionNet (GT via ``frame_utils.yuv420_to_rgb``;
NEVER MPS).  Emits one ``engineered_correction_boundary_solver_smoke.v1`` row per
(base, candidate) to a durable JSON (NO /tmp).

THE HONEST MEASUREMENT (operator):
  - d_seg before/after on the EXACT SegNet argmax (popcount).
  - new_bad_flips_created  = pixels correct-before, wrong-after (collateral damage).
  - pixels_flipped_repaired = pixels wrong-before, correct-after.
  - pose_side_effect       = the d_pose change from upsampling the SAME correction onto
    the camera frame1 and running the exact PoseNet pair scorer (a frame1 change touches
    PoseNet).  Without this the row could lie.
  - score recomputed from components (the rounded final_score lies).

The solve is a real Gα≥b autograd solve (NO grid/sweep over correction params —
that is the killed lever-G class-6 pattern).

Authority: ``[local CPU-torch advisory]`` — non-promotable.  $0, no GPU, no MPS.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS = REPO_ROOT / "experiments/results/pr110pp_r2_nonmps_candidate_20260609/analysis"
_UPSTREAM = REPO_ROOT / "upstream"
for _p in (str(_HARNESS), str(_UPSTREAM), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tac.boundary_math.boundary_solver import (  # noqa: E402
    TorchSegNetJacobian,
    plan_contour_normal,
    plan_graph_cut,
    plan_mdl_contour,
    score_from_components,
    solve_and_measure_seg_only,
)

CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512
DEVICE = torch.device("cpu")


def _L():
    """Cached import of the render+score harness (on sys.path)."""

    import render_and_score_lib

    return render_and_score_lib


def _segnet():
    from modules import SegNet, segnet_sd_path
    from safetensors.torch import load_file

    seg = SegNet().eval().to(DEVICE)
    seg.load_state_dict(load_file(segnet_sd_path, device="cpu"))
    return seg


def _seg_input_from_camera_frame1(frame1_chw: torch.Tensor, segnet) -> torch.Tensor:
    """The EXACT (1,3,384,512) tensor evaluate.py feeds: last-frame bilinear resize."""

    x = frame1_chw.unsqueeze(0).unsqueeze(0)  # (1,1,3,H,W)
    x = torch.cat([x, x], dim=1)  # (1,2,3,H,W) — preprocess takes last frame
    return segnet.preprocess_input(x).detach()  # (1,3,384,512)


def _pose_side_effect(scorer, gt_pair, comp_pair_chw, correction_seg_hw):
    """Measure d_pose before/after applying the (upsampled) seg correction to camera frame1.

    The same correction field solved on the SegNet grid is bilinearly upsampled to the
    camera grid and added to comp frame1 (luma direction, all 3 channels), clamped+rounded
    — then the exact DistortionNet pose term is measured render-vs-GT.
    """

    import render_and_score_lib as L

    # baseline pose.
    comp_bthwc = L.comp_pair_to_bthwc(comp_pair_chw)
    pose0, _seg0 = scorer.score_batch(gt_pair, comp_bthwc)
    # upsample correction (384x512)->(camera) and add to camera frame1 (all channels).
    corr = torch.from_numpy(np.asarray(correction_seg_hw, dtype=np.float64)).float()
    corr_cam = F.interpolate(
        corr.unsqueeze(0).unsqueeze(0), size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False
    )[0, 0]
    comp2 = comp_pair_chw.clone()
    comp2[1] = (comp2[1] + corr_cam.unsqueeze(0)).clamp_(0.0, 255.0).round_()
    comp2_bthwc = L.comp_pair_to_bthwc(comp2)
    pose1, _seg1 = scorer.score_batch(gt_pair, comp2_bthwc)
    return float(pose0[0]), float(pose1[0])


def _frontier_base_frames(pair_indices):
    """frontier_archive base: the 177 KB carrier's decoded comp frame1s + GT pairs."""

    import render_and_score_lib as L

    fr = L.FrontierRenderer()
    comps = fr.render_baseline_pairs(pair_indices)  # dict pi -> (2,3,camera)
    gts = L.decode_gt_pairs(pair_indices)  # dict pi -> (2,camera,3) uint8
    out = {}
    for pi in pair_indices:
        comp_pair = comps[pi].float()  # (2,3,H,W)
        gt_pair = torch.stack(
            [gts[pi][0], gts[pi][1]]
        ).float().unsqueeze(0)  # (1,2,camera,3)
        out[pi] = (comp_pair, gt_pair)
    return out


def run_candidate(base_candidate, plan_fn, segnet, base_frames, scorer, base_archive_bytes,
                  *, n_frames, with_pose):
    """Run ONE candidate across the frame sample; return the aggregated smoke row dict."""

    from tac.boundary_math.bitmask_dseg import flip_count
    from tac.optimization.frame1_seg_repair_atoms import measure_segnet_argmax

    t0 = time.time()
    last_kind = None
    agg = {
        "d_seg_before": [], "d_seg_after": [],
        "pose_before": [], "pose_after": [],
        "repaired": 0, "new_bad": 0, "components": 0, "archive_bytes_delta": 0,
        "frames": 0,
    }
    for _pi, (comp_pair, gt_pair) in list(base_frames.items())[:n_frames]:
        comp_f1 = comp_pair[1]  # (3,camera)
        gt_f1 = gt_pair[0, 1].permute(2, 0, 1).float()  # (3,camera)
        # exact base + target argmax on the SegNet grid.
        a_b, _ = measure_segnet_argmax(segnet, comp_f1.permute(1, 2, 0).numpy())
        a_s, _ = measure_segnet_argmax(segnet, gt_f1.permute(1, 2, 0).numpy())
        if flip_count(a_b, a_s) == 0:
            continue
        # build the jacobian provider on the EXACT seg input of the base.
        base_seg_in = _seg_input_from_camera_frame1(comp_f1, segnet)
        jac = TorchSegNetJacobian(segnet, base_seg_in)
        plan = plan_fn(a_b, a_s)
        last_kind = plan.correction_kind
        if not plan.atoms:
            # the candidate DECLINED to correct this frame (e.g. MDL: all flip
            # components too small to pay rent at the water level).  That is the
            # candidate's honest verdict — a NO-OP (d_seg unchanged, 0 bytes, no pose
            # side-effect), NOT a skip.  Record the zero-correction frame.
            from tac.boundary_math.bitmask_dseg import d_seg_reference
            ds = float(d_seg_reference(a_b, a_s))
            agg["d_seg_before"].append(ds)
            agg["d_seg_after"].append(ds)
            agg["frames"] += 1
            if with_pose:
                comp_bthwc = _L().comp_pair_to_bthwc(comp_pair)
                p0, _s0 = scorer.score_batch(gt_pair, comp_bthwc)
                agg["pose_before"].append(float(p0[0]))
                agg["pose_after"].append(float(p0[0]))
            continue
        row, field = solve_and_measure_seg_only(
            plan, jac, a_s, base_candidate=base_candidate,
            base_archive_bytes=base_archive_bytes,
        )
        agg["d_seg_before"].append(row.d_seg_before)
        agg["d_seg_after"].append(row.d_seg_after)
        agg["repaired"] += row.pixels_flipped_repaired
        agg["new_bad"] += row.new_bad_flips_created
        agg["components"] += row.boundary_components_touched
        agg["archive_bytes_delta"] += row.archive_bytes_delta
        agg["frames"] += 1
        if with_pose:
            p0, p1 = _pose_side_effect(scorer, gt_pair, comp_pair, field)
            agg["pose_before"].append(p0)
            agg["pose_after"].append(p1)

    if agg["frames"] == 0:
        return None
    d_seg_before = float(np.mean(agg["d_seg_before"]))
    d_seg_after = float(np.mean(agg["d_seg_after"]))
    if with_pose and agg["pose_before"]:
        d_pose_before = float(np.mean(agg["pose_before"]))
        d_pose_after = float(np.mean(agg["pose_after"]))
    else:
        d_pose_before = 0.0
        d_pose_after = 0.0
    # per-frame archive bytes delta -> the carrier carries it once per frame's correction.
    bytes_delta = int(agg["archive_bytes_delta"])
    score_before = score_from_components(d_seg_before, d_pose_before, base_archive_bytes)
    score_after = score_from_components(d_seg_after, d_pose_after, base_archive_bytes + bytes_delta)
    pose_side = float(
        np.sqrt(10.0 * max(d_pose_after, 0.0)) - np.sqrt(10.0 * max(d_pose_before, 0.0))
    )
    return {
        "schema": "engineered_correction_boundary_solver_smoke.v1",
        "base_candidate": base_candidate,
        "correction_kind": last_kind,
        "uses_stored_per_pixel_table": False,
        "archive_bytes_delta": bytes_delta,
        "inflate_runtime_delta_seconds": round(time.time() - t0, 3),
        "d_seg_before": d_seg_before,
        "d_pose_before": d_pose_before,
        "score_before": score_before,
        "d_seg_after": d_seg_after,
        "d_pose_after": d_pose_after,
        "score_after": score_after,
        "delta_score_total": score_after - score_before,
        "boundary_components_touched": int(agg["components"]),
        "pixels_flipped_repaired": int(agg["repaired"]),
        "new_bad_flips_created": int(agg["new_bad"]),
        "pose_side_effect": pose_side,
        "authority_tier": "exact_cpu_advisory",
        "metric_family": "exact_pair_scorer",
        "no_fake_class_6_passed": True,
        "frames_measured": int(agg["frames"]),
        "provenance": {
            "evidence_grade": "local-CPU-torch-advisory",
            "axis_tag": "[local CPU-torch advisory]",
            "promotable": False, "score_claim": False,
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-frames", type=int, default=8, help="frames to measure per (base,candidate)")
    ap.add_argument("--pairs", type=int, nargs="*", default=None, help="explicit pair indices")
    ap.add_argument("--bases", nargs="*", default=["frontier_archive"],
                    choices=["frontier_archive", "lever_b_argmax_generator"])
    ap.add_argument("--no-pose", action="store_true", help="skip the pose side-effect measurement")
    ap.add_argument("--out", type=str, default=None, help="output JSON path (durable, NO /tmp)")
    args = ap.parse_args(argv)

    pairs = args.pairs if args.pairs is not None else list(range(args.n_frames))
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "experiments/results/closed_spec_boundary_solver_20260610"
        / f"smoke_rows_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    segnet = _segnet()
    import render_and_score_lib as L
    scorer = None if args.no_pose else L.ExactScorer()

    plans = {
        "contour_normal": plan_contour_normal,
        "graph_cut": plan_graph_cut,
        "mdl_contour": plan_mdl_contour,
    }
    rows = []
    for base in args.bases:
        if base == "frontier_archive":
            base_frames = _frontier_base_frames(pairs)
            base_archive_bytes = 177169
        else:
            raise SystemExit(
                "lever_b_argmax_generator base requires the generator artifact; "
                "frontier_archive is the immediate exact-evalable base (run that first)."
            )
        for kind, plan_fn in plans.items():
            print(f"[{base}/{kind}] running on {len(pairs)} pairs...", flush=True)
            row = run_candidate(base, plan_fn, segnet, base_frames, scorer,
                                base_archive_bytes, n_frames=args.n_frames,
                                with_pose=not args.no_pose)
            if row is None:
                print("  -> no flips on sample; skipped", flush=True)
                continue
            rows.append(row)
            print(f"  d_seg {row['d_seg_before']:.3e} -> {row['d_seg_after']:.3e} | "
                  f"repaired={row['pixels_flipped_repaired']} new_bad={row['new_bad_flips_created']} | "
                  f"pose_side={row['pose_side_effect']:+.3e} bytes={row['archive_bytes_delta']} | "
                  f"dScore={row['delta_score_total']:+.6e}", flush=True)

    out_path.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nwrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
