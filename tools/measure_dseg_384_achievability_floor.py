#!/usr/bin/env python3
"""dseg_384_achievability_floor.v1 — the ABSOLUTE-FLOOR vs CAPACITY-LIMITED verdict.

THE QUESTION (settles whether sub-0.15 is reachable on the RGB-decoder rung AT ALL):
the residual ``d_seg`` of our small-basis decoder concentrates at low frozen-GT-margin
pixels (sister tool ``measure_dseg_reducibility_gt_margin.py`` → IRREDUCIBLE for OUR
decoder). The open question that verdict could NOT answer: is the d_seg cap ABSOLUTE
(no decoder, however perfect, beats it through the eval round-trip) or just OUR
CAPACITY (a perfect / higher-resolution decoder would reach much lower)?

THE DECISIVE MEASUREMENT — the d_seg of the BEST-POSSIBLE 384-output reconstruction.
A perfect 384-output decoder's output IS the GT frame, bilinear-downsampled to the
decoder's native 384×512. We then push that through the EXACT eval round-trip the
inflate path uses (384 → bicubic↑ camera-res(874×1164) → uint8 round → SegNet
preprocess (x[:,-1], bilinear↓ to 384×512)) and score its SegNet argmax against the
GT-camera-res SegNet argmax (the d_seg=0 reference; GT-vs-GT = 0 by construction).

This isolates each pipeline stage's IRREDUCIBLE contribution — the d_seg floor that
ANY 384-output RGB decoder incurs from the resolution bottleneck + uint8 round-trip,
independent of decoder fidelity:

  (1) FLOOR-384 (headline): GT_camres(uint8) → float → bilinear↓ (384,512) →
      bicubic↑ (874,1164) → clamp/round → uint8 → SegNet. The achievability floor
      for ANY 384-output decoder. The uint8 round MATCHES driver.kit_aware_exact_eval.
  (2) FLOOR-384-float (no uint8): same but SKIP the uint8 round (keep float into
      SegNet). FLOOR-384 − FLOOR-384-float ISOLATES the uint8-round contribution
      on top of the resolution bottleneck; FLOOR-384-float IS the resolution-bottleneck
      contribution alone.
  (3) FLOOR-UINT8only: GT_camres → camera-res uint8 re-round only (NO 384 bottleneck)
      → SegNet. Isolates the uint8-quantization contribution at camera res alone.
  (4) FLOOR-CAMRES (self-consistency): GT_camres straight to SegNet preprocess. MUST
      be 0 (GT-vs-GT) — a pipeline-faithfulness gate. STOP if non-zero.

VERDICT:
  * FLOOR-384 d_seg ≳ 0.0007 (≈ the 177KB frontier's d_seg, ≳ the T_1 d_seg budget):
    the 384 bottleneck ALONE floors d_seg near/above T_1 → NO 384-output decoder can
    reach the T_3 d_seg budget (0.00032). sub-0.15 via d_seg needs a ≥camera-res
    decoder OR a non-RGB paradigm. → ABSOLUTE-FLOOR.
  * FLOOR-384 d_seg ≈ 0 (≪ 0.0003): the d_seg cap is decoder CAPACITY/fidelity, not
    the pipeline → d_seg IS reducible by a better/bigger/higher-res decoder. →
    CAPACITY-LIMITED (reopens the higher-res decoder path).

[contest-CPU advisory] / [macOS-CPU advisory] — NON-PROMOTABLE, mechanism only
(``promotable=false``, ``score_claim=false``). NEVER MPS (a live MPS training run
owns the GPU). CPU-only, $0. GT decode ONLY via ``frame_utils.yuv420_to_rgb`` (PyAV
rgb24 is FORBIDDEN, ~100x phantom pose). ALL score math via ``tac.contest_score``.

Run:
  .venv/bin/python tools/measure_dseg_384_achievability_floor.py \\
    --n-pairs 48 \\
    --out .omx/research/dseg_384_achievability_floor_20260623.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO / "src"), str(_REPO / "upstream"), str(_REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_EVAL_H, _EVAL_W = 384, 512  # decoder native output / SegNet model input
_CAM_H, _CAM_W = 874, 1164  # camera resolution (the inflate raw-frame size)
_SEG_H, _SEG_W = 384, 512  # SegNet model input size (segnet_model_input_size = (512, 384))


def _utc() -> str:
    return subprocess.run(  # subprocess-no-check-OK: timestamp capture; date(1) failure yields empty string in a receipt, never a decision
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-pairs", type=int, default=48)
    ap.add_argument("--batch-pairs", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    # live operating point (held fixed so the floor is expressed in S-units)
    ap.add_argument("--live-d-pose", type=float, default=0.0003658535717598473)
    ap.add_argument("--live-archive-bytes", type=int, default=82457)
    # frontier / budget reference d_seg values for the verdict band
    ap.add_argument(
        "--frontier-d-seg",
        type=float,
        default=0.0007,
        help="the 177KB frontier's approximate d_seg (the ABSOLUTE-FLOOR trip line)",
    )
    ap.add_argument(
        "--t3-d-seg-budget",
        type=float,
        default=0.00032,
        help="the T_3 (sub-0.15) d_seg budget; FLOOR-384 << this => CAPACITY-LIMITED",
    )
    args = ap.parse_args(argv)

    if str(args.device) != "cpu":
        raise SystemExit("device must be cpu (NEVER MPS — a live MPS training run owns the GPU)")

    import av
    import numpy as np
    import torch

    from frame_utils import yuv420_to_rgb

    from tac.contest_score import compute_contest_score
    from tac.score_aware_loop.targets import load_frozen_distortion_net

    torch.manual_seed(0)
    np.random.seed(0)

    N = int(args.n_pairs)
    dev = torch.device(args.device)

    # --- frozen scorer (canonical loader; CPU authority) ---
    net = load_frozen_distortion_net(upstream_dir=str(_REPO / "upstream"), device=args.device)
    segnet = net.segnet
    segnet.eval()

    # The SegNet preprocess (upstream/modules.py:107-109): take x[:,-1] (last frame),
    # bilinear-resize to (segnet_model_input_size[1], [0]) = (384, 512). We mirror
    # DistortionNet.preprocess_input's (b t h w c -> b t c h w).float() rearrange, then
    # SegNet.preprocess_input. We feed FULL pairs (b,2,Hc,Wc,3) so x[:,-1] picks frame1
    # exactly as the authority eval does (SegNet scores the LAST frame of each pair).
    def _seg_logits_from_pair_chw_camres(pair_chw: torch.Tensor) -> torch.Tensor:
        """pair_chw: (b,2,3,Hc,Wc) float in [0,255]. Returns (b,5,384,512) logits."""
        seg_in = segnet.preprocess_input(pair_chw)  # x[:,-1] + bilinear -> (b,3,384,512)
        return segnet(seg_in)

    # margin histogram edges (logit units) — identical to the sister tool for parity.
    hist_edges = np.array(
        [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 1e9],
        dtype=np.float64,
    )
    n_bins = len(hist_edges) - 1

    # per-floor accumulators
    FLOORS = ("floor_384", "floor_384_float", "floor_uint8only", "floor_camres")
    per_pair_d_seg: dict[str, list[float]] = {f: [] for f in FLOORS}
    total_flips: dict[str, int] = {f: 0 for f in FLOORS}
    total_pixels = 0
    # per-row flip count (SegNet-grid row 0.._SEG_H-1) for the horizon-band profile
    row_flip_count: dict[str, Any] = {f: np.zeros(_SEG_H, dtype=np.int64) for f in FLOORS}
    row_pixel_count = np.zeros(_SEG_H, dtype=np.int64)
    # flip-margin pools (GT margin at the flipped pixels — to see WHERE the floor flips sit)
    flip_margin_pool: dict[str, list[Any]] = {f: [] for f in FLOORS}

    # --- iterate pairs: GT decode (yuv420_to_rgb), build the 4 floor inputs, score each ---
    container = av.open(str(_REPO / "upstream" / "videos" / "0.mkv"))
    frames_iter = container.decode(container.streams.video[0])

    def _next_pair() -> torch.Tensor | None:
        f0 = None
        for frame in frames_iter:
            rgb = yuv420_to_rgb(frame)  # (Hc,Wc,3) uint8 camera res
            if f0 is None:
                f0 = rgb
                continue
            return torch.stack([f0, rgb])  # (2,Hc,Wc,3) uint8
        return None

    pair_idx = 0
    while pair_idx < N:
        batch_gt = []
        for _ in range(min(int(args.batch_pairs), N - pair_idx)):
            pair = _next_pair()
            if pair is None:
                break
            batch_gt.append(pair)
        if not batch_gt:
            break
        b = len(batch_gt)
        gt = torch.stack(batch_gt).to(dev)  # (b,2,Hc,Wc,3) uint8 camera res

        with torch.inference_mode():
            # GT pair as (b,2,3,Hc,Wc) float — the canonical SegNet input layout.
            gt_chw = gt.permute(0, 1, 4, 2, 3).float()  # (b,2,3,Hc,Wc)
            gt_logits = _seg_logits_from_pair_chw_camres(gt_chw)  # reference (b,5,384,512)
            gt_arg = gt_logits.argmax(dim=1)  # (b,384,512)
            top2 = torch.topk(gt_logits, k=2, dim=1).values
            gt_margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)  # (b,384,512)
            gt_margin_np = gt_margin.cpu().numpy()

            # ---- build the BEST-POSSIBLE 384 reconstruction of frame1 (the scored frame) ----
            # A perfect 384-output decoder's output is the GT frame bilinear-downsampled
            # to 384×512. We round-trip the WHOLE pair (both frames) through the SAME path
            # so x[:,-1] / preprocess behaves identically to the authority eval; SegNet
            # only reads frame1 anyway. We operate per-frame on the (b*2,3,Hc,Wc) stack.
            flat_cam = gt_chw.reshape(b * 2, 3, _CAM_H, _CAM_W)  # float [0,255]

            # (1) 384 bottleneck: bilinear↓ to native 384×512 (the perfect decoder output)
            down384 = torch.nn.functional.interpolate(
                flat_cam, size=(_EVAL_H, _EVAL_W), mode="bilinear", align_corners=False
            )
            # bicubic↑ back to camera res — the EXACT inflate up-path (driver line 3766)
            up_cam = torch.nn.functional.interpolate(
                down384, size=(_CAM_H, _CAM_W), mode="bicubic", align_corners=False
            )
            # FLOOR-384-float: NO uint8 round (resolution-bottleneck contribution alone)
            floor384_float_chw = up_cam.clamp(0, 255).reshape(b, 2, 3, _CAM_H, _CAM_W)
            # FLOOR-384 (headline): uint8 round — EXACT driver.kit_aware_exact_eval cast
            floor384_uint8_chw = (
                up_cam.clamp(0, 255).round().to(torch.uint8).float().reshape(b, 2, 3, _CAM_H, _CAM_W)
            )
            # FLOOR-UINT8only: camera-res uint8 re-round of GT (NO 384 bottleneck). GT is
            # already uint8, so this is the identity at camera res — a self-consistency
            # check that an already-quantized confident GT frame does not self-flip.
            floor_uint8only_chw = (
                gt_chw.clamp(0, 255).round().to(torch.uint8).float()
            )
            # FLOOR-CAMRES (self-consistency): GT straight to SegNet (must be 0)
            floor_camres_chw = gt_chw

            floor_inputs = {
                "floor_384": floor384_uint8_chw,
                "floor_384_float": floor384_float_chw,
                "floor_uint8only": floor_uint8only_chw,
                "floor_camres": floor_camres_chw,
            }

            for fname, fchw in floor_inputs.items():
                flogits = _seg_logits_from_pair_chw_camres(fchw)
                farg = flogits.argmax(dim=1)  # (b,384,512)
                flip = (gt_arg != farg)  # (b,384,512) bool
                flip_np = flip.cpu().numpy()
                for k in range(b):
                    per_pair_d_seg[fname].append(float(flip_np[k].mean()))
                total_flips[fname] += int(flip_np.sum())
                row_flip_count[fname] += flip_np.sum(axis=(0, 2))
                fm = gt_margin_np[flip_np]
                if fm.size:
                    flip_margin_pool[fname].append(fm.astype(np.float32))

            total_pixels += int(gt_arg.numel())
            row_pixel_count += int(b) * _SEG_W  # b images × W columns per row

        pair_idx += b

    container.close()

    # --- aggregate per floor + S-units ---
    def _q(a: np.ndarray, qs: list[float]) -> dict[str, float | None]:
        if a.size == 0:
            return {f"p{int(q * 100)}": None for q in qs}
        vals = np.quantile(a, qs)
        return {f"p{int(q * 100)}": float(v) for q, v in zip(qs, vals)}

    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    S_base = compute_contest_score(0.0, args.live_d_pose, args.live_archive_bytes)
    floors: dict[str, Any] = {}
    for fname in FLOORS:
        d = per_pair_d_seg[fname]
        d_seg = float(np.mean(d)) if d else float("nan")
        # S-unit floor: hold pose + rate at live values, put THIS d_seg in the seg term
        S_at_floor = compute_contest_score(d_seg, args.live_d_pose, args.live_archive_bytes)
        s_units = float(S_at_floor - S_base)  # == 100 * d_seg (seg term is linear)
        fm = (
            np.concatenate(flip_margin_pool[fname])
            if flip_margin_pool[fname]
            else np.array([], dtype=np.float32)
        )
        # per-row flip-rate profile (full + horizon-band summary rows 96..320)
        rfr = [
            (int(row_flip_count[fname][r]) / int(row_pixel_count[r])) if row_pixel_count[r] else 0.0
            for r in range(_SEG_H)
        ]
        top_rows = sorted(range(_SEG_H), key=lambda r: rfr[r], reverse=True)[:15]
        floors[fname] = {
            "d_seg": d_seg,
            "S_units_floor": s_units,
            "total_flips": total_flips[fname],
            "flip_margin_median": (float(np.median(fm)) if fm.size else None),
            "flip_margin_quantiles": _q(fm, qs),
            "per_row_flip_rate": rfr,
            "top_flip_rows": [{"row": r, "flip_rate": rfr[r]} for r in top_rows],
        }

    # --- self-consistency gate: FLOOR-CAMRES must be ~0 (GT-vs-GT) ---
    camres_d_seg = floors["floor_camres"]["d_seg"]
    camres_ok = (not np.isnan(camres_d_seg)) and camres_d_seg <= 1e-6

    floor384 = floors["floor_384"]["d_seg"]
    floor384_float = floors["floor_384_float"]["d_seg"]
    floor_uint8 = floors["floor_uint8only"]["d_seg"]
    # decomposition: resolution-bottleneck (float) + uint8-on-top
    uint8_on_top_of_bottleneck = float(floor384 - floor384_float)

    # --- VERDICT ---
    verdict = "INCONCLUSIVE"
    verdict_detail = ""
    if not camres_ok:
        verdict = "BLOCKER_PIPELINE_MISMATCH"
        verdict_detail = (
            f"FLOOR-CAMRES (GT-vs-GT) d_seg={camres_d_seg:.2e} is not ~0 — the SegNet "
            "preprocess pipeline is not deterministic / faithful. Do NOT issue a floor "
            "verdict."
        )
    elif floor384 >= args.frontier_d_seg:
        verdict = "ABSOLUTE-FLOOR"
        verdict_detail = (
            f"FLOOR-384 d_seg={floor384:.2e} (S-units {100 * floor384:.4f}) >= the "
            f"frontier/T_1 trip line {args.frontier_d_seg:.2e}. The 384 resolution "
            f"bottleneck + uint8 round-trip ALONE floor d_seg at/above T_1 — NO 384-output "
            f"RGB decoder, however perfect, can reach the T_3 d_seg budget "
            f"({args.t3_d_seg_budget:.2e}). sub-0.15 via the d_seg axis requires a "
            f"≥camera-res decoder OR a non-RGB paradigm. Resolution-bottleneck contributes "
            f"{floor384_float:.2e}; uint8-on-top contributes {uint8_on_top_of_bottleneck:.2e}."
        )
    elif floor384 <= args.t3_d_seg_budget:
        verdict = "CAPACITY-LIMITED"
        verdict_detail = (
            f"FLOOR-384 d_seg={floor384:.2e} (S-units {100 * floor384:.4f}) <= the T_3 "
            f"d_seg budget {args.t3_d_seg_budget:.2e}. A PERFECT 384-output decoder passes "
            f"the round-trip nearly cleanly — the residual d_seg is decoder CAPACITY/fidelity, "
            f"NOT the pipeline. d_seg IS reducible by a better/bigger/higher-res decoder; the "
            f"label-noise concentration is about WHERE our residual sits, not an absolute "
            f"floor. (reopens the higher-res / higher-capacity decoder path.)"
        )
    else:
        verdict = "PARTIAL-FLOOR"
        verdict_detail = (
            f"FLOOR-384 d_seg={floor384:.2e} (S-units {100 * floor384:.4f}) sits BETWEEN the "
            f"T_3 budget {args.t3_d_seg_budget:.2e} and the frontier/T_1 line "
            f"{args.frontier_d_seg:.2e}. A perfect 384 decoder is floored ABOVE the T_3 d_seg "
            f"budget but BELOW T_1: the 384 bottleneck blocks T_3-via-d_seg but a perfect 384 "
            f"decoder would still beat OUR current d_seg. sub-0.15 via d_seg needs higher "
            f"resolution; incremental d_seg below our current value is decoder-capacity-reachable."
        )

    artifact: dict[str, Any] = {
        "schema": "dseg_384_achievability_floor.v1",
        "utc": _utc(),
        "authority_tier": "exact_cpu_advisory",
        "metric_family": "exact_segnet_per_pixel",
        "promotable": False,
        "score_claim": False,
        "mechanism_update_eligible": True,
        "score_roadmap_update_eligible": False,
        "device": args.device,
        "n_pairs_scored": len(per_pair_d_seg["floor_384"]),
        "total_pixels": total_pixels,
        "live_operating_point": {
            "d_pose": args.live_d_pose,
            "archive_bytes": args.live_archive_bytes,
            "S_base_seg0": S_base,
        },
        "verdict_band": {
            "frontier_d_seg_trip_line": args.frontier_d_seg,
            "t3_d_seg_budget": args.t3_d_seg_budget,
        },
        "self_consistency_floor_camres": {
            "d_seg": camres_d_seg,
            "passed_is_zero": camres_ok,
        },
        "floors": floors,
        "decomposition": {
            "resolution_bottleneck_d_seg": floor384_float,
            "uint8_on_top_of_bottleneck_d_seg": uint8_on_top_of_bottleneck,
            "uint8_only_camres_d_seg": floor_uint8,
            "floor_384_total_d_seg": floor384,
            "note": (
                "FLOOR-384 = resolution_bottleneck (float) + uint8_on_top. "
                "uint8_only_camres isolates the uint8 step with NO 384 bottleneck "
                "(≈0 for already-uint8 GT = self-consistency)."
            ),
        },
        "margin_hist_edges": hist_edges.tolist(),
        "verdict": verdict,
        "verdict_detail": verdict_detail,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    print(f"=== d_seg 384 achievability floor (N={len(per_pair_d_seg['floor_384'])} pairs) ===")
    print(f"  FLOOR-CAMRES (GT-vs-GT, must be ~0): {camres_d_seg:.3e}  ok={camres_ok}")
    print(f"  FLOOR-384       d_seg={floor384:.4e}  S-units={100 * floor384:.4f}")
    print(f"  FLOOR-384-float d_seg={floor384_float:.4e}  S-units={100 * floor384_float:.4f}  (resolution bottleneck)")
    print(f"  uint8-on-top    d_seg={uint8_on_top_of_bottleneck:.4e}  S-units={100 * uint8_on_top_of_bottleneck:.4f}")
    print(f"  FLOOR-UINT8only d_seg={floor_uint8:.4e}  (camera-res uint8 only)")
    print(f"  VERDICT: {verdict}")
    print(f"  {verdict_detail}")
    print(f"  wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
