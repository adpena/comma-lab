# SPDX-License-Identifier: MIT
"""OPTIMAL-FORM UNIWARD best-chance empirical smoke (Catalog #307 falsifiable).

$0 macOS-MLX/CPU research-signal. NON-PROMOTABLE per Catalog #192/#341/#127/#323.
NO cloud. All numbers tagged ``[macOS-CPU advisory]`` — never a contest score.

THE TEST (detector-informed cost-map × direct-payload surface):
  - DIRECT-PAYLOAD surface: the byte-closed UWD1 sparse-delta sidechannel
    (``tac.uniward_delta.pack_sparse_delta`` → ``unpack`` → ``apply_delta_to_frame``).
  - DETECTOR-informed cost-map: S-UNIWARD texture × SegNet boundary band
    (``segnet_boundary_band_weights`` from the REAL SegNet logits on REAL frames).
  - Apples-to-apples: a real degraded reconstruction (temporal prediction:
    frame N predicted by frame N-1 — exactly the NSCS06/HNeRV inter-frame premise)
    is CORRECTED by a sparse δ = GT − pred packed THREE ways at MATCHED target_bytes:
        (1) uniform           — flat cost-map (rank by |δ| only)
        (2) texture_only       — S-UNIWARD only (the prior-negative form)
        (3) detector_informed  — S-UNIWARD × SegNet boundary band (OPTIMAL FORM)
    Then the REAL SegNet d_seg (per-pixel argmax-flip RATE vs GT, the contest seg
    distortion) of each corrected reconstruction is measured at the SAME bytes.

FALSIFIABLE CLAIM: at optimal form, detector_informed achieves LOWER d_seg than
texture_only AND uniform at matched bytes → the prior negatives were FALSE
(cargo-culted). If it does NOT → genuine exhaustion (honest paradigm-level DEFER).
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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
UPSTREAM = REPO_ROOT / "upstream"

from tac.multi_granularity_sensitivity import segnet_boundary_band_weights
from tac.scorer import load_default_scorers
from tac.substrates.uniward_per_pixel_distortion.detector_informed_direct_payload_cost_map import (
    SIDECAR_ROLE_CORRECTION,
    allocation_diff_proof,
    compose_detector_informed_cost_map,
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=6)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--l-inf", type=float, default=8.0)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument(
        "--target-bytes",
        type=int,
        nargs="+",
        default=[400, 800, 1600, 3200],
    )
    ap.add_argument("--device", default="mps")
    ap.add_argument(
        "--out",
        default="experiments/results/uniward_optimal_form_detector_informed_smoke_20260531/smoke_output.json",
    )
    args = ap.parse_args()

    device = torch.device(args.device if torch.backends.mps.is_available() else "cpu")
    print(f"[smoke] device={device} (NON-PROMOTABLE [macOS-CPU advisory])", flush=True)

    # ── Real frames + real SegNet ──────────────────────────────────────────
    n_needed = args.n_pairs + 1
    frames = _decode_frames(UPSTREAM / "videos" / "0.mkv", n_needed, args.stride)
    H, W = frames.shape[1], frames.shape[2]
    print(f"[smoke] decoded {len(frames)} real frames {H}x{W} from 0.mkv", flush=True)
    segnet = load_default_scorers(str(UPSTREAM), device=device)[1]
    print("[smoke] loaded REAL SegNet (segnet.safetensors)", flush=True)

    # ── Build the real degraded reconstruction + residual δ ────────────────
    # pred(frame_n) = frame_{n-1}  (temporal prediction; the NSCS06/HNeRV premise)
    gt = frames[1:n_needed]          # (n_pairs, H, W, 3) GT
    pred = frames[0 : n_needed - 1]  # (n_pairs, H, W, 3) degraded prediction
    n_pairs = gt.shape[0]

    gt_f = torch.from_numpy(gt.astype(np.float32))            # (P,H,W,3)
    pred_f = torch.from_numpy(pred.astype(np.float32))        # (P,H,W,3)
    delta = (gt_f - pred_f).permute(0, 3, 1, 2).contiguous()  # (P,3,H,W) in [0,255] units

    # ── REAL cost-maps on the predicted (rendered) frames ──────────────────
    pred_bchw = pred_f.permute(0, 3, 1, 2).contiguous().to(device)  # (P,3,H,W)
    texture_cost = compute_uniward_cost_map(pred_bchw).cpu()        # (P,H,W) S-UNIWARD

    # REAL SegNet boundary band per predicted frame (resized to native H,W).
    boundary_native = np.zeros((n_pairs, H, W), dtype=np.float32)
    for p in range(n_pairs):
        lg = _segnet_logits(segnet, pred[p], device)  # (1,5,384,512)
        bw, _stats = segnet_boundary_band_weights(lg, tau=args.tau)  # (384,512)
        bw_t = bw.unsqueeze(0).unsqueeze(0)  # (1,1,384,512)
        bw_native = torch.nn.functional.interpolate(
            bw_t, size=(H, W), mode="bilinear", align_corners=False
        )[0, 0]
        boundary_native[p] = bw_native.cpu().numpy()
    boundary_t = torch.from_numpy(boundary_native)
    print(
        f"[smoke] real SegNet boundary band: mean={boundary_native.mean():.4f} "
        f"band_frac={(boundary_native > np.exp(-1)).mean():.4f}",
        flush=True,
    )

    # ── The three cost-maps ────────────────────────────────────────────────
    uniform_cost = torch.ones((n_pairs, H, W), dtype=torch.float32)
    detector = compose_detector_informed_cost_map(
        texture_cost, boundary_t, role=SIDECAR_ROLE_CORRECTION, tau=args.tau
    )
    detector_cost = torch.from_numpy(detector.cost_bhw)

    # Allocation-diff proof (no-op guard, Catalog #105/#139/#220) at a fixed n_kept.
    diff_vs_uniform = allocation_diff_proof(
        detector.cost_bhw, uniform_cost.numpy(), n_kept=2000
    )
    diff_vs_texture = allocation_diff_proof(
        detector.cost_bhw, texture_cost.numpy(), n_kept=2000
    )
    print(
        f"[smoke] alloc-diff proof: vs_uniform symdiff="
        f"{diff_vs_uniform['kept_set_symmetric_difference']} "
        f"vs_texture symdiff={diff_vs_texture['kept_set_symmetric_difference']}",
        flush=True,
    )

    methods = {
        "uniform": uniform_cost,
        "texture_only": texture_cost,
        "detector_informed": detector_cost,
    }

    # ── Baseline d_seg (uncorrected prediction) ───────────────────────────
    baseline_dseg = []
    for p in range(n_pairs):
        baseline_dseg.append(_d_seg(segnet, gt[p], pred_f[p], device))
    baseline_mean = float(np.mean(baseline_dseg))
    print(f"[smoke] baseline d_seg (uncorrected pred) = {baseline_mean:.6f}", flush=True)

    # ── Sweep: pack each method at each budget, measure REAL d_seg ──────────
    results: dict[str, list[dict]] = {m: [] for m in methods}
    for tb in args.target_bytes:
        for mname, cmap in methods.items():
            blob = pack_sparse_delta(
                delta, cmap, l_inf_budget=args.l_inf, target_bytes=tb
            )
            spec = unpack_sparse_delta(blob)
            dseg_vals = []
            for p in range(n_pairs):
                # spec tensors are CPU-built; apply on CPU, _d_seg moves to device.
                corrected = apply_delta_to_frame(pred_f[p], spec, p)
                dseg_vals.append(_d_seg(segnet, gt[p], corrected, device))
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
                f"[smoke] tb={tb:5d} {mname:18s} bytes={len(blob):5d} "
                f"n_kept={spec.n_kept:6d} d_seg={mean_dseg:.6f} "
                f"(Δ vs base {baseline_mean - mean_dseg:+.6f})",
                flush=True,
            )

    # ── Verdict: at each budget, is detector_informed the lowest d_seg? ────
    verdict_rows = []
    detector_wins = 0
    detector_beats_texture = 0
    for i, tb in enumerate(args.target_bytes):
        d_det = results["detector_informed"][i]["mean_d_seg"]
        d_tex = results["texture_only"][i]["mean_d_seg"]
        d_uni = results["uniform"][i]["mean_d_seg"]
        wins_all = d_det <= d_tex and d_det <= d_uni
        beats_tex = d_det < d_tex
        detector_wins += int(wins_all)
        detector_beats_texture += int(beats_tex)
        verdict_rows.append(
            {
                "target_bytes": int(tb),
                "d_seg_detector": d_det,
                "d_seg_texture_only": d_tex,
                "d_seg_uniform": d_uni,
                "detector_lowest": bool(wins_all),
                "detector_beats_texture_only": bool(beats_tex),
            }
        )

    n_budgets = len(args.target_bytes)
    if detector_wins == n_budgets:
        verdict = "FALSE_NEGATIVE_CONVERTED_TO_LIVE"
    elif detector_beats_texture >= 1 or detector_wins >= 1:
        verdict = "MIXED_PARTIAL_DETECTOR_HELPS_AT_SOME_BUDGETS"
    else:
        verdict = "GENUINE_EXHAUSTION_DETECTOR_DOES_NOT_HELP"

    inputs_sha = hashlib.sha256(
        json.dumps(
            {
                "n_pairs": n_pairs,
                "stride": args.stride,
                "l_inf": args.l_inf,
                "tau": args.tau,
                "target_bytes": args.target_bytes,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    out = {
        "schema": "uniward_optimal_form_detector_informed_smoke_v1",
        "tool": "experiments/uniward_optimal_form_detector_informed_smoke.py",
        "captured_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": str(device),
        "axis_tag": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "macOS-CPU-advisory",
        "inputs_sha256": inputs_sha,
        "n_pairs": n_pairs,
        "frame_H": int(H),
        "frame_W": int(W),
        "l_inf_budget": float(args.l_inf),
        "tau": float(args.tau),
        "real_segnet": "upstream/models/segnet.safetensors",
        "real_frames": "upstream/videos/0.mkv",
        "detector_cost_map_confirmed_segnet_informed": True,
        "detector_cost_map_stats": detector.as_dict(),
        "allocation_diff_vs_uniform": diff_vs_uniform,
        "allocation_diff_vs_texture_only": diff_vs_texture,
        "baseline_d_seg_uncorrected": baseline_mean,
        "results_per_method": results,
        "verdict_per_budget": verdict_rows,
        "n_budgets": n_budgets,
        "detector_wins_all_three": detector_wins,
        "detector_beats_texture_only_count": detector_beats_texture,
        "verdict": verdict,
        "provenance": {
            "kind": "predicted",
            "model_id": "uniward_optimal_form_detector_informed_v1",
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
    print(f"\n[smoke] VERDICT: {verdict}", flush=True)
    print(f"[smoke] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
