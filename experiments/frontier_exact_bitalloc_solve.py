#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Frontier exact precision-allocation SOLVE harness — the deterministic reverse-water-fill.

Operator correction (2026-06-19): the int5 "confirmed structural" cap was un-nuanced — it
held our own codec fixed and used a UNIFORM low-nbits grid + per-tensor LSQ. We possess the
frozen SegNet+PoseNet, the frozen 600 pairs, and the closed-form eval → the optimal
per-tensor precision allocation is a DETERMINISTIC OPTIMIZATION we SOLVE, not probe. We OWN
the codec — it is a design variable. Design memo:
``.omx/research/frontier_rate_exact_bitalloc_solve_design_20260618.md``.

THE SOLVE (all REAL — NO-FAKE), no fragile finetune (deterministic PTQ + closed-form alloc):
  1. decode the REAL frontier decoder (``tac.frontier_decoder_ptq``; byte-identical identity
     round-trip proven).
  2. compute the EXACT per-tensor score-sensitivity (one backward each, no training):
     * d_seg: ``‖∂(Σ SegNet top1−top2 margin)/∂W‖`` (REUSE
       ``tac.margin_saliency_map.compute_decoder_tensor_margin_saliency``);
     * d_pose: ``‖∂d_pose/∂W‖`` through the frozen PoseNet on the eval-roundtripped output
       (``tac.frontier_exact_bitalloc.compute_decoder_tensor_pose_saliency``);
     combine by the master gradient ``s = w_seg·g_seg + w_pose·g_pose``.
  3. reverse-water-fill / KKT: ``b_t* = log₂(λ·ln2·c_t/numel_t)``, ``c_t = s·absmax·√numel``
     — NON-UNIFORM per-tensor bits, more where the scorer is sensitive, fewer where blind.
     Sweep λ (pinned to a target mean-bits grid) to trace the RD curve.
  4. per-tensor requant at each tensor's OWN nbits (``intn_qdq``) → byte-close through the
     REAL frontier codec (``reencode_frontier_archive`` — the qdq grid is per-tensor int8 +
     fp16 scale, fewer distinct symbols at lower nbits = the rate win; NO per-channel scales,
     NO decoder change) → RE-DECODE the shipped bytes → CPU-AUTHORITY exact eval on the
     re-decoded weights (NEVER MPS).

AUTHORITY: every score is ``[contest-CPU advisory]`` NON-PROMOTABLE; pointer UNMOVED
0.19110. If a byte-closed S beats the pointer → FLAG for paired CPU/CUDA exact eval (the
pointer-mover) + borrowed_substrate_accounting (frontier is PR101/106-derived) — do NOT
self-promote. $0 (local MPS gradient for the saliency backward; CPU authority eval).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F

_EVAL_H, _EVAL_W = 384, 512
_RATE_DENOM = 37_545_489
_FRONTIER_S = 0.19109982419209975
_INT8_LOCAL_BASELINE_S = 0.1965
_GENERIC_INT5_S = 0.5593
_SUB015 = 0.15


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return 100.0 * d_seg + (10.0 * d_pose + 1e-12) ** 0.5 + 25.0 * archive_bytes / _RATE_DENOM


def _eval_roundtrip(decoded_pair_b23hw: torch.Tensor) -> torch.Tensor:
    """The contest eval-roundtrip (bicubic↑→bilinear↓→uint8 STE) → (B,2,H,W,3) in [0,255].

    1:1 with the finetune harness / ``common.py``. The decoder output is (B,2,3,384,512)
    in [0,255]."""
    B = decoded_pair_b23hw.shape[0]
    flat = decoded_pair_b23hw.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    decoded_bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
    dc = decoded_bhwc.clamp(0, 255)
    return dc + (dc.round() - dc).detach()  # uint8 STE


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mean-bits-grid",
        default="4.0,4.5,5.0,5.5,6.0,6.5",
        help="comma list of target param-weighted MEAN allocated bits (the λ-sweep)",
    )
    ap.add_argument("--b-min", type=int, default=2)
    ap.add_argument("--b-max", type=int, default=8)
    ap.add_argument(
        "--sens-frames",
        type=int,
        default=16,
        help="frames to accumulate the per-tensor sensitivity over (deterministic spread)",
    )
    ap.add_argument(
        "--w-pose",
        type=float,
        default=271.0,
        help="∂S/∂d_pose master weight at the operating point (5/sqrt(10*d_pose))",
    )
    ap.add_argument("--w-seg", type=float, default=100.0)
    ap.add_argument(
        "--sens-device",
        default="cpu",
        help="device for the sensitivity backward (cpu authority, or mps gradient)",
    )
    ap.add_argument(
        "--eval-pairs", type=int, default=600, help="pairs for the CPU authority eval"
    )
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="experiments/results/frontier_exact_bitalloc_solve")
    ap.add_argument(
        "--targets-cache", default="experiments/results/capstone_gt_targets_cache"
    )
    ap.add_argument("--out-json", default=None)
    ap.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="skip mean-bits points already in the curve JSONL (resumable)",
    )
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args(argv)

    torch.set_num_threads(int(args.threads))
    torch.manual_seed(int(args.seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_jsonl = out_dir / "rd_curve.jsonl"
    best_dir = out_dir / "best"
    best_dir.mkdir(exist_ok=True)

    from tac.frontier_decoder_ptq import (
        FRONTIER_ARCHIVE,
        build_frontier_decoder,
        decode_frontier_member,
        reencode_frontier_archive,
    )
    from tac.frontier_exact_bitalloc import (
        combine_sensitivities,
        compute_decoder_tensor_margin_saliency,
        compute_decoder_tensor_pose_saliency,
        lam_for_target_mean_bits,
        per_stage_bits_summary,
        requant_state_dict_per_tensor_nbits,
        waterfill_bit_allocation,
    )
    from tac.torch_vehicle.driver import import_vendored_bundle
    from tac.torch_vehicle.scorer_context import RealScorerContext

    import_vendored_bundle()
    from tac.torch_vehicle.vendored_imports import import_vendored

    video_path = import_vendored("data").get_default_video_path()

    # ── decode the frontier ──────────────────────────────────────────────────────
    t0 = time.time()
    member = decode_frontier_member(FRONTIER_ARCHIVE)
    n_pairs_total = int(member.latents.shape[0])
    print(
        f"[frontier] archive={FRONTIER_ARCHIVE.name} member_bytes={member.member_bytes} "
        f"n_pairs={n_pairs_total} ({time.time() - t0:.1f}s)",
        flush=True,
    )

    # ── CPU AUTHORITY scorer context (full eval; NEVER MPS for eval) ──────────────
    eval_pairs = min(int(args.eval_pairs), n_pairs_total)
    t_ctx = time.time()
    ctx = RealScorerContext(
        video_path,
        device="cpu",
        train_device=args.sens_device,
        split_by_head=False,
        max_pairs=eval_pairs,
        targets_cache=args.targets_cache,
    )
    print(f"[scorer] ctx built ({time.time() - t_ctx:.1f}s)", flush=True)

    # ── Step 1: EXACT per-tensor score-sensitivity (one backward each) ───────────
    sens_dev = torch.device(args.sens_device)
    sens_decoder = build_frontier_decoder(member.state_dict).to(sens_dev).eval()
    latents = member.latents.to(sens_dev)
    n_fr = min(int(args.sens_frames), n_pairs_total)
    frame_idx = [round(i * (n_pairs_total - 1) / max(1, n_fr - 1)) for i in range(n_fr)]

    # d_seg producer: needs render_frame_chw(decoder, latent_row, head) -> (3,H,W) camera-res
    def _render_frame_chw(decoder, latent_row, score_head):
        pair = decoder(latent_row)  # (1, 2, 3, 384, 512) in [0,255]
        head_i = 0 if score_head == "rgb_0" else 1
        return pair[0, head_i]  # (3, 384, 512) WITH grad

    seg_net = ctx.train_distortion_net.segnet

    class _SegWrap:
        def preprocess_input(self, x):
            # x: (1, 2, 3, H, W) -> the SegNet path preprocess (last frame, resize)
            _posenet_in, segnet_in = ctx.train_distortion_net.preprocess_input(
                x.permute(0, 1, 3, 4, 2)  # -> (1,2,H,W,3) the scorer's BHWC contract
            )
            return segnet_in

        def __call__(self, seg_in):
            return seg_net(seg_in)

    t_seg = time.time()
    seg_sal = compute_decoder_tensor_margin_saliency(
        sens_decoder,
        latents,
        _SegWrap(),
        render_frame_chw=_render_frame_chw,
        frame_indices=frame_idx,
        score_head="rgb_0",
    )
    print(f"[sens] d_seg saliency ({len(frame_idx)} frames, {time.time() - t_seg:.1f}s)", flush=True)

    # d_pose producer: render the pair WITH eval-roundtrip, run frozen PoseNet
    def _render_pair_bchw(decoder, latent_rows):
        pair = decoder(latent_rows)  # (B,2,3,384,512)
        return _eval_roundtrip(pair)  # (B,2,H,W,3) in [0,255] WITH grad

    def _pose_forward(decoded_bhwc):
        _seg_out, pose6 = ctx.seg_pose_forward(decoded_bhwc)
        return pose6  # (B,6)

    t_pose = time.time()
    pose_sal = compute_decoder_tensor_pose_saliency(
        sens_decoder,
        latents,
        _pose_forward,
        ctx.pose_targets,
        render_pair_bchw=_render_pair_bchw,
        frame_indices=frame_idx,
    )
    print(f"[sens] d_pose saliency ({len(frame_idx)} frames, {time.time() - t_pose:.1f}s)", flush=True)

    sens = combine_sensitivities(
        sens_decoder, seg_sal, pose_sal, w_seg=float(args.w_seg), w_pose=float(args.w_pose)
    )
    # log the per-weight impact ranking (the allocation prior)
    ranked = sorted(sens.per_tensor.items(), key=lambda kv: -kv[1])
    print("[sens] combined per-tensor score-sensitivity (s = w_seg*g_seg + w_pose*g_pose):", flush=True)
    for name, s in ranked:
        pw = (sens.absmax[name] * (sens.numel[name] ** 0.5) * s) / sens.numel[name]
        print(
            f"    {name:22s} s={s:.4e} g_seg={sens.g_seg[name]:.3e} "
            f"g_pose={sens.g_pose[name]:.3e} absmax={sens.absmax[name]:.4f} "
            f"numel={sens.numel[name]:7d} per_w_impact={pw:.3e}",
            flush=True,
        )

    sens_blob = {
        "per_tensor": sens.per_tensor,
        "g_seg": sens.g_seg,
        "g_pose": sens.g_pose,
        "absmax": sens.absmax,
        "numel": sens.numel,
        "w_seg": sens.w_seg,
        "w_pose": sens.w_pose,
        "n_frames": len(frame_idx),
    }
    (out_dir / "sensitivity.json").write_text(json.dumps(sens_blob, indent=2))

    # ── the byte-closed CPU AUTHORITY eval for one allocation ────────────────────
    def _eval_alloc(nbits: dict[str, int], tag: str) -> dict:
        shrunk_sd = requant_state_dict_per_tensor_nbits(member.state_dict, nbits)
        scratch = out_dir / f"_cand_{tag}.zip"
        bytes_meas = reencode_frontier_archive(member, shrunk_sd, scratch)
        # NO-FAKE eval-on-shipped-bytes: RE-DECODE the byte-closed archive.
        shipped = decode_frontier_member(scratch)
        eval_dec = build_frontier_decoder(shipped.state_dict).to("cpu").eval()
        eval_latents = shipped.latents[:eval_pairs]
        res = ctx.exact_eval(eval_dec, eval_latents, bytes_meas)
        d_seg = float(res["seg_distortion"])
        d_pose = float(res["pose_distortion"])
        s = _score(d_seg, d_pose, bytes_meas)
        try:
            scratch.unlink()
        except OSError:
            pass
        return {
            "d_seg": d_seg,
            "d_pose": d_pose,
            "archive_bytes": bytes_meas,
            "rate_term": 25.0 * bytes_meas / _RATE_DENOM,
            "score": s,
        }

    # already-evaluated mean-bits points (resume).
    done_targets: set[float] = set()
    if args.resume and curve_jsonl.is_file():
        for line in curve_jsonl.read_text().splitlines():
            try:
                done_targets.add(round(float(json.loads(line)["target_mean_bits"]), 3))
            except (json.JSONDecodeError, KeyError):
                pass

    targets = [float(x) for x in args.mean_bits_grid.split(",") if x.strip()]
    best = {"score": float("inf")}
    rows: list[dict] = []
    for tgt in targets:
        if round(tgt, 3) in done_targets:
            print(f"[resume] skip target_mean_bits={tgt} (already in curve)", flush=True)
            continue
        lam = lam_for_target_mean_bits(sens, tgt, b_min=int(args.b_min), b_max=int(args.b_max))
        alloc = waterfill_bit_allocation(sens, lam, b_min=int(args.b_min), b_max=int(args.b_max))
        total_numel = sum(sens.numel.values())
        actual_mean = alloc.total_weight_bits / total_numel
        t_e = time.time()
        ev = _eval_alloc(alloc.nbits, f"mb{tgt}")
        ev["wall_s"] = round(time.time() - t_e, 1)
        row = {
            "target_mean_bits": tgt,
            "actual_mean_bits": round(actual_mean, 4),
            "lam": lam,
            "nbits": alloc.nbits,
            "per_stage_nbits": per_stage_bits_summary(alloc.nbits),
            **ev,
        }
        rows.append(row)
        with curve_jsonl.open("a") as f:
            f.write(json.dumps(row) + "\n")
        improved = ev["score"] < best["score"]
        print(
            f"  >>> mean_bits={tgt} (actual {actual_mean:.3f}) "
            f"[contest-CPU advisory] S={ev['score']:.4f} d_seg={ev['d_seg']:.6f} "
            f"d_pose={ev['d_pose']:.6f} bytes={ev['archive_bytes']} rate={ev['rate_term']:.4f} "
            f"({'NEW BEST' if improved else 'no improve'}) ({ev['wall_s']}s)",
            flush=True,
        )
        if improved:
            best = dict(row)
            shrunk_sd = requant_state_dict_per_tensor_nbits(member.state_dict, alloc.nbits)
            reencode_frontier_archive(member, shrunk_sd, best_dir / "best_archive.zip")
            (best_dir / "best_meta.json").write_text(json.dumps(best, indent=2))

    # if everything was resumed, recover the best from the curve.
    if best["score"] == float("inf") and curve_jsonl.is_file():
        all_rows = [json.loads(line) for line in curve_jsonl.read_text().splitlines() if line.strip()]
        if all_rows:
            best = min(all_rows, key=lambda r: r["score"])

    # ── verdict ──────────────────────────────────────────────────────────────────
    bs = best["score"] if best["score"] < float("inf") else None
    beats_generic_int5 = bs is not None and bs < _GENERIC_INT5_S - 1e-9
    beats_int8 = bs is not None and bs < _INT8_LOCAL_BASELINE_S - 1e-9
    beats_pointer = bs is not None and bs < _FRONTIER_S - 1e-9
    sub015 = bs is not None and bs < _SUB015 - 1e-9
    if bs is None:
        headline = "NO EVAL — empty mean-bits grid or all resumed-with-no-curve."
    elif beats_pointer:
        headline = (
            f"EXACT-ALLOC BEATS THE POINTER LOCALLY: best byte-closed S={bs:.4f} < "
            f"{_FRONTIER_S:.4f} [contest-CPU advisory]. FLAG for paired CPU/CUDA exact eval "
            f"(the pointer-mover) + borrowed_substrate_accounting — do NOT self-promote, "
            f"pointer UNMOVED. {'AND reaches sub-0.15.' if sub015 else ''}"
        )
    elif beats_int8:
        headline = (
            f"EXACT-ALLOC RECOVERS below the int8 baseline: best byte-closed S={bs:.4f} < "
            f"{_INT8_LOCAL_BASELINE_S:.4f} but NOT < pointer 0.191. The non-uniform "
            f"allocation works; closing to the pointer needs the codec mixed-precision "
            f"pack or a finetune polish. [contest-CPU advisory]"
        )
    elif beats_generic_int5:
        gap_closed = (_GENERIC_INT5_S - bs) / (_GENERIC_INT5_S - _FRONTIER_S)
        headline = (
            f"EXACT-ALLOC BEATS the generic int5 ({bs:.4f} < {_GENERIC_INT5_S}) but caps "
            f"above the int8 baseline: closed {100 * gap_closed:.1f}% of the int5→pointer "
            f"gap by allocating bits exactly. Binding remainder: see the per-stage "
            f"d_seg-axis split. [contest-CPU advisory]"
        )
    else:
        headline = (
            f"EXACT-ALLOC caps at S={bs:.4f} >= generic int5 {_GENERIC_INT5_S}: the exact "
            f"per-tensor allocation did NOT beat uniform int5 at this RD point. "
            f"[contest-CPU advisory]"
        )

    report = {
        "probe": "frontier_exact_bitalloc_solve",
        "authority": "[contest-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotable": False,
        "frontier_pointer_moved": False,
        "frontier_archive": str(FRONTIER_ARCHIVE),
        "frontier_archive_sha256": "b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e",
        "references": {
            "frontier_pointer_S": _FRONTIER_S,
            "int8_local_baseline_S": _INT8_LOCAL_BASELINE_S,
            "generic_int5_S": _GENERIC_INT5_S,
            "sub_015_target": _SUB015,
        },
        "config": vars(args),
        "sensitivity_summary": {
            "top5_per_weight_impact": [
                {
                    "tensor": name,
                    "s": sens.per_tensor[name],
                    "g_seg": sens.g_seg[name],
                    "g_pose": sens.g_pose[name],
                }
                for name, _ in ranked[:5]
            ],
        },
        "best": best if bs is not None else None,
        "beats_generic_int5": beats_generic_int5,
        "beats_int8_baseline": beats_int8,
        "beats_pointer_local": beats_pointer,
        "reaches_sub_015": sub015,
        "verdict_headline": headline,
        "built_at_utc": _now(),
        "note": (
            "DETERMINISTIC SOLVE (no finetune): exact frozen-scorer per-tensor sensitivity "
            "(d_seg margin grad + d_pose grad, master-gradient combined) → reverse-water-fill "
            "KKT bit allocation (NON-uniform per-tensor nbits) → per-tensor requant → "
            "byte-closed REAL frontier codec → RE-DECODED CPU-authority exact eval (NEVER "
            "MPS). Local CPU ≈ contest-CPU with a +0.0054 offset → local-beats-pointer needs "
            "the paired contest row. NON-PROMOTABLE; pointer UNMOVED 0.19110. SUBMISSION "
            "needs borrowed_substrate_accounting (frontier is PR101/106-derived)."
        ),
    }
    print("\n=== FRONTIER EXACT BIT-ALLOC SOLVE VERDICT ===")
    print(f"  {headline}")
    out = Path(args.out_json) if args.out_json else (out_dir / "frontier_exact_bitalloc_solve.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[wrote] {out}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
