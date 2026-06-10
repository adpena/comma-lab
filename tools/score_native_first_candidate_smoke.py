#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""SCORE-NATIVE first candidate smoke — solver-on-lever_b + legal-frame bridge + exact S row.

The campaign's first build (tasks #55-reactivation + #56). Executes, on the exact CPU-torch
scorer (GT via ``frame_utils.yuv420_to_rgb`` ONLY; NEVER MPS):

  STEP 1a (DECISIVE, cheap): characterize the lever-B generator's argmax residual structure
    (A_g != L*) — contiguity histogram. Is it contiguous patches (repairable) or salt-and-pepper
    (the frontier-base finding, unrepairable)?
  STEP 1b: run the closed-spec boundary solver on the legal-frame's actual SegNet argmax (the
    real Gα≥b solve), emitting the engineered_correction_boundary_solver_smoke.v1 row.
  STEP 2: the legal-frame bridge — synthesize a scorer-free palette frame whose SegNet argmax
    lands in the cell { argmax SegNet(y1) == A_g }; measure the actual recovered d_seg.
  STEP 3: byte-close + measure the exact advisory S row {d_seg, d_pose, bytes, S}.

Authority: ``[local CPU-torch advisory]`` — non-promotable, NO score claim. $0, no GPU, no MPS.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS = REPO_ROOT / "experiments/results/pr110pp_r2_nonmps_candidate_20260609/analysis"
_UPSTREAM = REPO_ROOT / "upstream"
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(_HARNESS), str(_UPSTREAM)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from tac.boundary_math.bitmask_dseg import d_seg_reference  # noqa: E402
from tac.boundary_math.boundary_solver import (  # noqa: E402
    TorchSegNetJacobian,
    plan_contour_normal,
    plan_graph_cut,
    plan_mdl_contour,
    score_from_components,
    solve_and_measure_seg_only,
)
from tac.boundary_math.legal_frame_bridge import (  # noqa: E402
    fit_palette_gt_region_mean,
    measure_label_map_coded_bytes,
    rasterize_palette_frame,
)
from tac.boundary_math.lever_b_generator import (  # noqa: E402
    aggregate_residual_stats,
    build_coords,
    generator_argmax,
    load_generator_npz,
    residual_component_stats,
)

CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512
DEVICE = torch.device("cpu")
_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
_CONTEST_TOTAL_BYTES = 37_545_489


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path, field: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{field}={path!r} is a /tmp-class path; use the SSD tier per CLAUDE.md.")


def _segnet():
    from modules import SegNet, segnet_sd_path
    from safetensors.torch import load_file

    seg = SegNet().eval().to(DEVICE)
    seg.load_state_dict(load_file(segnet_sd_path, device="cpu"))
    return seg


def _seg_argmax_of_frame(segnet, frame_hwc_unit255: np.ndarray) -> np.ndarray:
    """Exact SegNet argmax of a single camera-res frame1 (the EXACT eval path)."""
    from tac.optimization.frame1_seg_repair_atoms import measure_segnet_argmax

    am, _ = measure_segnet_argmax(segnet, np.asarray(frame_hwc_unit255))
    return am.astype(np.int64)


def _seg_input_from_camera_frame1(frame1_chw: torch.Tensor, segnet) -> torch.Tensor:
    x = frame1_chw.unsqueeze(0).unsqueeze(0)
    x = torch.cat([x, x], dim=1)
    return segnet.preprocess_input(x).detach()


def run(
    ckpt_path: Path,
    targets_dir: Path,
    out_dir: Path,
    n_pairs: int,
    solver_candidate: str,
    with_pose: bool,
) -> dict[str, Any]:
    _refuse_tmp(out_dir, "out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    import render_and_score_lib as L

    t0 = time.time()
    params, cfg = load_generator_npz(ckpt_path)
    meta = json.loads((targets_dir / "targets_meta.json").read_text())
    n_built = int(meta["num_pairs_built"])
    H, W = meta["seg_input_hw"]
    coords = build_coords(H, W)
    gt_argmax = np.memmap(targets_dir / "gt_segnet_argmax.u8", dtype=np.uint8, mode="r",
                          shape=(n_built, H, W))

    pairs = list(range(min(n_pairs, cfg.num_pairs)))
    segnet = _segnet()
    scorer = L.ExactScorer() if with_pose else None
    gt_pairs = L.decode_gt_pairs(pairs)  # pi -> (2,camera,3) uint8

    # ---- STEP 1a: generator residual structure (A_g vs L*) ----
    gen_argmax: dict[int, np.ndarray] = {}
    residual_stats = []
    gen_d_seg_vs_L = []
    for pi in pairs:
        ag = generator_argmax(params, cfg, coords, pi, H, W).astype(np.int64)
        gen_argmax[pi] = ag
        ls = np.asarray(gt_argmax[pi]).astype(np.int64)
        residual_stats.append(residual_component_stats(ag, ls))
        gen_d_seg_vs_L.append(float(d_seg_reference(ag, ls)))
    step1a = aggregate_residual_stats(residual_stats)
    step1a["mean_generator_d_seg_vs_L"] = float(np.mean(gen_d_seg_vs_L))

    # ---- STEP 2: legal-frame bridge — fit palette from GT regions, rasterize ----
    gt_frames_hwc = np.stack([np.asarray(gt_pairs[pi][1]) for pi in pairs])  # (P,camera,3) frame1
    gt_seg_argmax_stack = np.stack([np.asarray(gt_argmax[pi]) for pi in pairs])  # (P,SEG_H,SEG_W)
    palette = fit_palette_gt_region_mean(gt_frames_hwc, gt_seg_argmax_stack, n_classes=cfg.n_classes)

    plan_fns = {"contour_normal": plan_contour_normal, "graph_cut": plan_graph_cut,
                "mdl_contour": plan_mdl_contour}
    plan_fn = plan_fns[solver_candidate]

    bridge_rows = []
    solver_rows = []
    for pi in pairs:
        ag = gen_argmax[pi]  # the generator's target argmax (what we paint)
        ls = np.asarray(gt_argmax[pi]).astype(np.int64)  # the TRUE target (what d_seg scores against)
        # rasterize the scorer-free palette frame from the GENERATOR argmax.
        frame_cam = rasterize_palette_frame(ag, palette, camera_h=CAMERA_H, camera_w=CAMERA_W)
        # measure the ACTUAL SegNet argmax of the synthesized frame (the cell-landing test).
        a_palette = _seg_argmax_of_frame(segnet, frame_cam)
        d_seg_palette = float(d_seg_reference(a_palette, ls))

        # ---- STEP 1b: boundary solver on the synthesized frame (real Gα≥b solve) ----
        frame_chw = torch.from_numpy(frame_cam.transpose(2, 0, 1)).float()
        base_seg_in = _seg_input_from_camera_frame1(frame_chw, segnet)
        jac = TorchSegNetJacobian(segnet, base_seg_in)
        plan = plan_fn(a_palette, ls)
        if plan.atoms:
            row, field = solve_and_measure_seg_only(
                plan, jac, ls, base_candidate="lever_b_palette_bridge", base_archive_bytes=0)
            d_seg_corrected = row.d_seg_after
            repaired = row.pixels_flipped_repaired
            new_bad = row.new_bad_flips_created
            bytes_delta = row.archive_bytes_delta
            corr_field = field
        else:
            d_seg_corrected = d_seg_palette
            repaired = new_bad = bytes_delta = 0
            corr_field = None
        solver_rows.append({
            "pi": pi, "d_seg_palette": d_seg_palette, "d_seg_corrected": d_seg_corrected,
            "repaired": repaired, "new_bad": new_bad, "bytes_delta": bytes_delta,
        })

        # pose: frame0 = GT frame0 (real motion, SegNet-invisible); frame1 = corrected palette.
        if with_pose:
            frame_corr = rasterize_palette_frame(
                ag, palette, camera_h=CAMERA_H, camera_w=CAMERA_W, correction_hw_seg=corr_field)
            gt0 = np.asarray(gt_pairs[pi][0])  # (camera,3) uint8 GT frame0
            comp_chw = torch.stack([
                torch.from_numpy(gt0.transpose(2, 0, 1)).float(),
                torch.from_numpy(frame_corr.transpose(2, 0, 1)).float(),
            ])  # (2,3,camera)
            gt_pair_t = torch.stack([gt_pairs[pi][0], gt_pairs[pi][1]]).float().unsqueeze(0)
            pose_d, seg_d = scorer.score_batch(gt_pair_t, L.comp_pair_to_bthwc(comp_chw))
            # DIAGNOSTIC: d_pose with GT frame1 (perfect appearance) — the pose FLOOR.
            # Isolates the palette's pose damage from the intrinsic pose cost of using
            # GT frame0 + (any) frame1. If this is ~0, the palette is the pose problem.
            gt_comp = torch.stack([
                torch.from_numpy(gt0.transpose(2, 0, 1)).float(),
                gt_pairs[pi][1].permute(2, 0, 1).float(),
            ])
            pose_gt, _ = scorer.score_batch(gt_pair_t, L.comp_pair_to_bthwc(gt_comp))
            bridge_rows.append({
                "pi": pi, "d_seg_palette": d_seg_palette, "d_seg_corrected": d_seg_corrected,
                "pose_dist": float(pose_d[0]), "seg_dist_exact_scorer": float(seg_d[0]),
                "pose_dist_gt_frame1_floor": float(pose_gt[0]),
            })
        else:
            bridge_rows.append({
                "pi": pi, "d_seg_palette": d_seg_palette, "d_seg_corrected": d_seg_corrected})

    # ---- STEP 3: byte-close + the advisory S row ----
    # generator blob (seg carrier) — reuse the smoke's quantized blob size from train_result if present.
    gen_blob_bytes = _generator_blob_bytes(params)
    label_map_floor = int(measure_label_map_coded_bytes(np.stack([gen_argmax[pi] for pi in pairs]).astype(np.uint8)))
    palette_bytes = len(palette.to_bytes())
    pose_carrier_bytes = int(meta.get("pose_trajectory", {}).get("best_pose_carrier_bytes", 6650))
    total_solver_bytes = int(sum(r["bytes_delta"] for r in solver_rows))
    # The carrier = generator blob + palette + pose section + solver corrections.
    archive_bytes = gen_blob_bytes + palette_bytes + pose_carrier_bytes + total_solver_bytes

    mean_d_seg_palette = float(np.mean([r["d_seg_palette"] for r in solver_rows]))
    mean_d_seg_corrected = float(np.mean([r["d_seg_corrected"] for r in solver_rows]))
    if with_pose:
        mean_d_pose = float(np.mean([r["pose_dist"] for r in bridge_rows]))
        mean_seg_exact = float(np.mean([r["seg_dist_exact_scorer"] for r in bridge_rows]))
    else:
        mean_d_pose = 0.0
        mean_seg_exact = mean_d_seg_corrected

    # THE LAW score recomputed from components (advisory; rate uses the FULL archive bytes).
    S_palette = score_from_components(mean_d_seg_palette, mean_d_pose, archive_bytes)
    S_corrected = score_from_components(mean_d_seg_corrected, mean_d_pose, archive_bytes)

    result = {
        "subagent": "score_native_first_candidate_55_56",
        "utc": _utc(),
        "evidence_grade": "[local CPU-torch advisory]",
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "n_pairs_measured": len(pairs),
        "solver_candidate": solver_candidate,
        "wall_s": round(time.time() - t0, 1),
        "step1a_generator_residual": step1a,
        "step1b_solver_aggregate": {
            "total_repaired": int(sum(r["repaired"] for r in solver_rows)),
            "total_new_bad": int(sum(r["new_bad"] for r in solver_rows)),
            "total_bytes_delta": total_solver_bytes,
            "mean_d_seg_palette": mean_d_seg_palette,
            "mean_d_seg_corrected": mean_d_seg_corrected,
            "net_d_seg_change": mean_d_seg_corrected - mean_d_seg_palette,
        },
        "step3_byte_close": {
            "generator_blob_bytes": gen_blob_bytes,
            "label_map_raw_brotli_floor_bytes": label_map_floor,
            "palette_bytes": palette_bytes,
            "pose_carrier_bytes": pose_carrier_bytes,
            "solver_correction_bytes": total_solver_bytes,
            "archive_bytes_total": archive_bytes,
            "rate_term": _CONTEST_TOTAL_BYTES and 25.0 * archive_bytes / _CONTEST_TOTAL_BYTES,
        },
        "advisory_S": {
            "S_palette_only": S_palette,
            "S_solver_corrected": S_corrected,
            "mean_d_seg": mean_d_seg_corrected,
            "mean_d_pose": mean_d_pose,
            "mean_seg_dist_exact_scorer": mean_seg_exact,
            "archive_bytes": archive_bytes,
        },
        "config": cfg.to_dict(),
        "solver_rows": solver_rows,
        "bridge_rows": bridge_rows,
        "provenance": {
            "axis_tag": "[local CPU-torch advisory]", "promotable": False,
            "score_claim": False, "hardware_substrate": "local_macos_cpu",
            "gt_decode": "frame_utils.yuv420_to_rgb", "scorer": "exact upstream modules.py CPU",
        },
    }
    (out_dir / "score_native_first_candidate.json").write_text(json.dumps(result, indent=2))
    return result


def _generator_blob_bytes(params: dict[str, np.ndarray]) -> int:
    """int8 per-tensor + brotli-q11 quantized generator blob (the seg carrier)."""
    import brotli

    base_chunks = []
    mod_chunk = b""
    for name, a in params.items():
        a = np.asarray(a).astype(np.float32)
        s = float(np.abs(a).max()) + 1e-8
        q = np.clip(np.round(a / s * 127.0), -127, 127).astype(np.int8)
        if name == "mod":
            mod_chunk = q.tobytes()
        else:
            base_chunks.append(q.tobytes())
    base = len(brotli.compress(b"".join(base_chunks), quality=11))
    mod = len(brotli.compress(mod_chunk, quality=11)) if mod_chunk else 0
    return base + mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    base = "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610"
    ap.add_argument("--ckpt", type=Path, default=Path(base) / "generator_ckpt" / "generator_n600.npz")
    ap.add_argument("--targets-dir", type=Path, default=Path(base) / "targets_n600")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "experiments/results/score_native_candidate_20260610")
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--solver-candidate", default="mdl_contour",
                    choices=["contour_normal", "graph_cut", "mdl_contour"])
    ap.add_argument("--no-pose", action="store_true")
    args = ap.parse_args(argv)

    result = run(
        ckpt_path=args.ckpt, targets_dir=args.targets_dir, out_dir=args.out_dir,
        n_pairs=args.n_pairs, solver_candidate=args.solver_candidate, with_pose=not args.no_pose)
    print("\n=== SCORE-NATIVE FIRST CANDIDATE ===")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("solver_rows", "bridge_rows")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
