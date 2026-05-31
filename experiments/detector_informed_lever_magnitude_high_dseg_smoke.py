# SPDX-License-Identifier: MIT
"""MAGNITUDE TEST: does the detector-informed allocation lever scale into
contest-relevant territory on a HIGH-baseline-d_seg render surface?

$0 macOS-MLX/CPU research-signal. NON-PROMOTABLE per Catalog #192/#341/#127/#323.
NO cloud. All numbers tagged ``[macOS-CPU advisory]`` — never a contest score.

CONTEXT (the chain):
  - Sister #1585 (commit 59ba009f0) VINDICATED detector-informed allocation at optimal
    form (direct-payload surface + REAL detector cost-map), but the magnitude was only
    ~1e-5 d_seg because the degraded reconstruction was a SLOW-STRIDE temporal
    prediction with a tiny baseline d_seg (0.0176) → almost no boundary-flip headroom.
  - CODEX correction (Finding 2): "SegNet interiors are free" is STALE — SegNet
    responds across the FULL 384x512 argmax grid, not just boundary flips. So the
    detector weight should be the FULL MEASURED SegNet response (input-gradient
    saliency, dense ~77%), NOT just the sparse boundary band (~4.58%).

THE MAGNITUDE TEST:
  - HIGH-baseline-d_seg surface: a genuinely degraded render (spatial downsample /
    blockify — a palette/LUT-render artifact) that COLLAPSES the spatial structure
    SegNet's argmax depends on → baseline d_seg 1-2 orders above the 0.0176 proxy
    (e.g. downsample x16 ≈ 0.035, x32 ≈ 0.34; blockify 16px ≈ 0.05, 32px ≈ 0.51).
  - DETECTOR weight = FULL-GRID SegNet response ``|∂ L_seg / ∂ pixel|`` (one backward
    pass per frame; dense across boundary + class-interior + region per codex).
  - DIRECT-PAYLOAD surface: the byte-closed UWD1 sparse-delta sidechannel
    (``pack_sparse_delta`` → ``unpack`` → ``apply_delta_to_frame``).
  - Apples-to-apples: the degraded render is CORRECTED by a sparse δ = GT − degraded,
    packed THREE ways at MATCHED target_bytes:
        (1) uniform           — flat cost-map (rank by |δ| only)
        (2) texture_only      — S-UNIWARD only (sister's prior-form)
        (3) detector_informed — S-UNIWARD × FULL-GRID SegNet response (codex form)
    Then the REAL SegNet d_seg of each corrected reconstruction is measured at the
    SAME bytes.

FALSIFIABLE CLAIM (Catalog #307): on a high-baseline-d_seg surface, detector-informed
allocation Pareto-dominates uniform with CONTEST-RELEVANT magnitude (|Δd_seg| >> 1e-5,
ideally 1e-3..1e-2) at matched bytes. If the magnitude STAYS ~1e-5 even on a
high-baseline surface → the lever is real but contest-negligible (honest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
UPSTREAM = REPO_ROOT / "upstream"

from tac.scorer import load_default_scorers
from tac.substrates.uniward_per_pixel_distortion.full_grid_segnet_response_cost_map import (
    RESPONSE_ROLE_CORRECTION,
    compose_full_grid_response_cost_map,
    full_grid_response_allocation_diff_proof,
)
from tac.uniward_delta import (
    apply_delta_to_frame,
    compute_uniward_cost_map,
    pack_sparse_delta,
    unpack_sparse_delta,
)


def _decode_frames(video_path: Path, n_frames: int, stride: int) -> np.ndarray:
    """Decode the first ``n_frames`` (every ``stride``-th) real frames as (N,H,W,3) uint8."""
    import av

    out: list[np.ndarray] = []
    container = av.open(str(video_path))
    try:
        for idx, frame in enumerate(container.decode(video=0)):
            if idx % stride == 0:
                out.append(frame.to_ndarray(format="rgb24"))
                if len(out) >= n_frames:
                    break
    finally:
        container.close()
    return np.stack(out, axis=0)


def _segnet_logits(segnet, frame_hwc_uint8: np.ndarray, device) -> torch.Tensor:
    """REAL SegNet logits on a single RGB frame. Input -> (1, T=1, C, H, W)."""
    x = torch.from_numpy(frame_hwc_uint8.astype(np.float32)).permute(2, 0, 1)  # (3,H,W)
    x = x.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3, H, W)
    with torch.inference_mode():
        seg_in = segnet.preprocess_input(x)  # (1, 3, 384, 512)
        logits = segnet(seg_in)  # (1, 5, 384, 512)
    return logits


def _d_seg(segnet, gt_uint8: np.ndarray, recon_float: torch.Tensor, device) -> float:
    """REAL contest d_seg: argmax-flip rate between GT and recon SegNet outputs."""
    recon_uint8 = recon_float.clamp(0, 255).round().to(torch.uint8).cpu().numpy()
    lg_gt = _segnet_logits(segnet, gt_uint8, device)
    lg_re = _segnet_logits(segnet, recon_uint8, device)
    with torch.inference_mode():
        flip = (lg_gt.argmax(dim=1) != lg_re.argmax(dim=1)).float().mean()
    return float(flip.item())


def _full_grid_segnet_response(
    segnet, frame_hwc_float: torch.Tensor, device, H: int, W: int
) -> np.ndarray:
    """FULL-GRID SegNet response = |∂ L_seg / ∂ pixel| (summed over RGB) at native H,W.

    ONE backward pass per frame. ``L_seg`` is the self-consistent cross-entropy of the
    SegNet logits against their own argmax pseudo-labels (the contest seg distortion is
    an argmax flip rate, so this gradient is the per-pixel argmax sensitivity). The
    response is computed at the 384x512 scorer resolution then bilinear-resized to the
    native frame resolution so it ranks the native-resolution δ entries.
    """
    x = frame_hwc_float.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,3,H,W)
    x = x.detach().clone().requires_grad_(True)
    seg_in = segnet.preprocess_input(x)  # (1,3,384,512) — differentiable wrt x
    logits = segnet(seg_in)  # (1,5,384,512)
    pseudo = logits.argmax(dim=1).detach()  # (1,384,512)
    loss = F.cross_entropy(logits, pseudo)
    loss.backward()
    grad = x.grad.detach().abs()[0, 0]  # (3,H,W)
    sal_native = grad.sum(dim=0)  # (H,W) — already at native res (preprocess resizes internally)
    if sal_native.shape != (H, W):
        sal_native = F.interpolate(
            sal_native[None, None], size=(H, W), mode="bilinear", align_corners=False
        )[0, 0]
    return sal_native.cpu().numpy().astype(np.float32)


def _degrade(gt_uint8: np.ndarray, surface: str) -> np.ndarray:
    """Produce a HIGH-baseline-d_seg degraded render of a single GT frame.

    Surfaces collapse the spatial structure SegNet's argmax depends on (a palette/
    LUT-render artifact): ``downsample_x{N}`` low-passes via downsample-upsample;
    ``blockify_{N}px`` averages NxN blocks (nearest upsample). Both are REAL,
    deterministic, and produce a baseline d_seg far above the 0.0176 temporal proxy.
    """
    H, W = gt_uint8.shape[:2]
    g = torch.from_numpy(gt_uint8.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    if surface.startswith("downsample_x"):
        ds = int(surface.split("x")[1])
        small = F.interpolate(g, scale_factor=1.0 / ds, mode="bilinear", align_corners=False)
        up = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
    elif surface.startswith("blockify_") and surface.endswith("px"):
        blk = int(surface[len("blockify_") : -len("px")])
        pooled = F.avg_pool2d(g, blk)
        up = F.interpolate(pooled, size=(H, W), mode="nearest")
    else:
        raise ValueError(f"unknown surface {surface!r}")
    return up[0].permute(1, 2, 0).clamp(0, 255).round().numpy().astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--l-inf", type=float, default=16.0)
    ap.add_argument(
        "--surfaces",
        nargs="+",
        default=["downsample_x16", "blockify_16px", "downsample_x32"],
    )
    ap.add_argument(
        "--target-bytes",
        type=int,
        nargs="+",
        default=[800, 1600, 3200, 6400, 12800],
    )
    ap.add_argument("--device", default="mps")
    ap.add_argument(
        "--out",
        default="experiments/results/detector_informed_lever_magnitude_high_dseg_20260531/smoke_output.json",
    )
    args = ap.parse_args()

    device = torch.device(args.device if torch.backends.mps.is_available() else "cpu")
    print(f"[mag] device={device} (NON-PROMOTABLE [macOS-CPU advisory])", flush=True)

    frames = _decode_frames(UPSTREAM / "videos" / "0.mkv", args.n_frames, args.stride)
    H, W = frames.shape[1], frames.shape[2]
    n = frames.shape[0]
    print(f"[mag] decoded {n} real frames {H}x{W} from 0.mkv", flush=True)
    segnet = load_default_scorers(str(UPSTREAM), device=device)[1]
    print("[mag] loaded REAL SegNet (segnet.safetensors)", flush=True)

    surface_results: dict[str, dict] = {}
    for surface in args.surfaces:
        print(f"\n[mag] ===== SURFACE: {surface} =====", flush=True)
        degraded = np.stack([_degrade(frames[i], surface) for i in range(n)], axis=0)  # (n,H,W,3)

        gt_f = torch.from_numpy(frames.astype(np.float32))      # (n,H,W,3)
        deg_f = torch.from_numpy(degraded.astype(np.float32))   # (n,H,W,3)
        delta = (gt_f - deg_f).permute(0, 3, 1, 2).contiguous() # (n,3,H,W)

        # REAL S-UNIWARD texture on the DEGRADED (rendered) frames.
        deg_bchw = deg_f.permute(0, 3, 1, 2).contiguous().to(device)
        texture_cost = compute_uniward_cost_map(deg_bchw).cpu()  # (n,H,W)

        # FULL-GRID SegNet response per degraded frame (codex Finding 2).
        response = np.zeros((n, H, W), dtype=np.float32)
        for p in range(n):
            response[p] = _full_grid_segnet_response(segnet, deg_f[p], device, H, W)
        resp_t = torch.from_numpy(response)
        nz_frac = float((response > 0).mean())
        print(
            f"[mag] full-grid SegNet response: nonzero_frac={nz_frac:.4f} "
            f"mean={response.mean():.3e} (dense full-grid per codex Finding 2)",
            flush=True,
        )

        uniform_cost = torch.ones((n, H, W), dtype=torch.float32)
        detector = compose_full_grid_response_cost_map(
            texture_cost, resp_t, role=RESPONSE_ROLE_CORRECTION
        )
        detector_cost = torch.from_numpy(detector.cost_bhw)

        diff_vs_uniform = full_grid_response_allocation_diff_proof(
            detector.cost_bhw, uniform_cost.numpy(), n_kept=4000
        )
        diff_vs_texture = full_grid_response_allocation_diff_proof(
            detector.cost_bhw, texture_cost.numpy(), n_kept=4000
        )
        print(
            f"[mag] alloc-diff: vs_uniform symdiff="
            f"{diff_vs_uniform['kept_set_symmetric_difference']} "
            f"vs_texture symdiff={diff_vs_texture['kept_set_symmetric_difference']}",
            flush=True,
        )

        methods = {
            "uniform": uniform_cost,
            "texture_only": texture_cost,
            "detector_informed": detector_cost,
        }

        baseline_vals = [_d_seg(segnet, frames[p], deg_f[p], device) for p in range(n)]
        baseline_mean = float(np.mean(baseline_vals))
        print(f"[mag] baseline d_seg (uncorrected degraded) = {baseline_mean:.6f}", flush=True)

        results: dict[str, list[dict]] = {m: [] for m in methods}
        for tb in args.target_bytes:
            for mname, cmap in methods.items():
                blob = pack_sparse_delta(delta, cmap, l_inf_budget=args.l_inf, target_bytes=tb)
                spec = unpack_sparse_delta(blob)
                dseg_vals = []
                for p in range(n):
                    corrected = apply_delta_to_frame(deg_f[p], spec, p)
                    dseg_vals.append(_d_seg(segnet, frames[p], corrected, device))
                mean_dseg = float(np.mean(dseg_vals))
                results[mname].append(
                    {
                        "target_bytes": int(tb),
                        "actual_bytes": len(blob),
                        "n_kept": int(spec.n_kept),
                        "mean_d_seg": mean_dseg,
                        "d_seg_reduction_vs_baseline": float(baseline_mean - mean_dseg),
                    }
                )
                print(
                    f"[mag] tb={tb:6d} {mname:18s} bytes={len(blob):6d} "
                    f"n_kept={spec.n_kept:7d} d_seg={mean_dseg:.6f} "
                    f"(Δ vs base {baseline_mean - mean_dseg:+.6f})",
                    flush=True,
                )

        verdict_rows = []
        detector_wins = 0
        detector_beats_uniform = 0
        max_abs_margin_vs_uniform = 0.0
        for i, tb in enumerate(args.target_bytes):
            d_det = results["detector_informed"][i]["mean_d_seg"]
            d_tex = results["texture_only"][i]["mean_d_seg"]
            d_uni = results["uniform"][i]["mean_d_seg"]
            wins_all = d_det <= d_tex and d_det <= d_uni
            beats_uni = d_det < d_uni
            margin_vs_uni = d_uni - d_det  # >0 ⇒ detector lower (better)
            detector_wins += int(wins_all)
            detector_beats_uniform += int(beats_uni)
            max_abs_margin_vs_uniform = max(max_abs_margin_vs_uniform, abs(margin_vs_uni))
            verdict_rows.append(
                {
                    "target_bytes": int(tb),
                    "d_seg_detector": d_det,
                    "d_seg_texture_only": d_tex,
                    "d_seg_uniform": d_uni,
                    "margin_detector_vs_uniform": float(margin_vs_uni),
                    "detector_lowest": bool(wins_all),
                    "detector_beats_uniform": bool(beats_uni),
                }
            )

        # Magnitude verdict: is the detector lever contest-relevant on this surface?
        CONTEST_RELEVANT_THRESHOLD = 1e-3
        if max_abs_margin_vs_uniform >= CONTEST_RELEVANT_THRESHOLD and detector_wins >= 1:
            mag_verdict = "CONTEST_RELEVANT"
        elif detector_beats_uniform >= 1:
            mag_verdict = "REAL_BUT_NEGLIGIBLE"
        else:
            mag_verdict = "NO_LEVER_ON_THIS_SURFACE"

        surface_results[surface] = {
            "baseline_d_seg": baseline_mean,
            "full_grid_response_nonzero_fraction": nz_frac,
            "detector_cost_map_stats": detector.as_dict(),
            "allocation_diff_vs_uniform": diff_vs_uniform,
            "allocation_diff_vs_texture_only": diff_vs_texture,
            "results_per_method": results,
            "verdict_per_budget": verdict_rows,
            "n_budgets": len(args.target_bytes),
            "detector_wins_all_count": detector_wins,
            "detector_beats_uniform_count": detector_beats_uniform,
            "max_abs_margin_detector_vs_uniform": float(max_abs_margin_vs_uniform),
            "magnitude_verdict": mag_verdict,
        }
        print(
            f"[mag] SURFACE {surface}: baseline={baseline_mean:.4f} "
            f"max_margin_vs_uniform={max_abs_margin_vs_uniform:.6f} "
            f"verdict={mag_verdict}",
            flush=True,
        )

    # Global magnitude verdict across surfaces.
    best_margin = max(s["max_abs_margin_detector_vs_uniform"] for s in surface_results.values())
    any_contest_relevant = any(
        s["magnitude_verdict"] == "CONTEST_RELEVANT" for s in surface_results.values()
    )
    any_real_lever = any(
        s["detector_beats_uniform_count"] >= 1 for s in surface_results.values()
    )
    if any_contest_relevant:
        global_verdict = "CONTEST_RELEVANT"
    elif any_real_lever:
        global_verdict = "REAL_BUT_NEGLIGIBLE"
    else:
        global_verdict = "NO_LEVER"

    inputs_sha = hashlib.sha256(
        json.dumps(
            {
                "n_frames": n,
                "stride": args.stride,
                "l_inf": args.l_inf,
                "surfaces": args.surfaces,
                "target_bytes": args.target_bytes,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    out = {
        "schema": "detector_informed_lever_magnitude_high_dseg_smoke_v1",
        "tool": "experiments/detector_informed_lever_magnitude_high_dseg_smoke.py",
        "captured_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": str(device),
        "axis_tag": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "macOS-CPU-advisory",
        "inputs_sha256": inputs_sha,
        "n_frames": n,
        "frame_H": int(H),
        "frame_W": int(W),
        "l_inf_budget": float(args.l_inf),
        "real_segnet": "upstream/models/segnet.safetensors",
        "real_frames": "upstream/videos/0.mkv",
        "detector_weight": "full_grid_segnet_input_gradient_saliency_codex_finding_2",
        "surfaces": args.surfaces,
        "surface_results": surface_results,
        "best_margin_detector_vs_uniform_across_surfaces": float(best_margin),
        "global_magnitude_verdict": global_verdict,
        "provenance": {
            "kind": "predicted",
            "model_id": "detector_informed_lever_magnitude_high_dseg_v1",
            "measurement_axis": "[macOS-CPU advisory]",
            "hardware_substrate": "macos_arm64_mps",
            "score_claim": False,
            "promotion_eligible": False,
            "inputs_sha256": inputs_sha,
        },
    }
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\n[mag] GLOBAL VERDICT: {global_verdict}", flush=True)
    print(f"[mag] best margin detector vs uniform across surfaces: {best_margin:.6f}", flush=True)
    print(f"[mag] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
