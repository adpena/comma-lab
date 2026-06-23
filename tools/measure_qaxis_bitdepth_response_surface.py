#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Q-AXIS response surface: how d_seg / d_pose / archive-bytes / S vary with the
decoder-weight QAT bit-depth (int8 → int3) on the REAL 0.19110 frontier decoder.

THE QUESTION (CLAUDE.md "THE GOAL — SUB-0.15"): a sister measurement proved the
frontier's int8 decoder weights are already at the order-0 Shannon floor (pure
entropy recode is exhausted — `pr110_payload_entropy_recode` only saved 1326 B).
The ONLY rate lever left on the borrowed frontier is LOWER-BIT weights. This tool
builds the REAL d_seg-vs-bits cost curve so the math-optimal solver knows whether
any bit-depth reaches sub-0.15 / sub-0.19.

WHAT IT DOES (all REAL — NO-FAKE, $0, CPU-only, NEVER MPS):
  1. Decode the REAL frontier decoder state_dict (`tac.frontier_decoder_ptq`,
     byte-identical int8 identity round-trip proven: int8-requant → 177169 B,
     0.0 weight err).
  2. For each bit-depth Q in {8,7,6,5,4,3} and each PTQ variant:
       * ``absmax`` — per-tensor symmetric int-N (scale = abs_max / qmax). The
         codec's native grid generalized to Q bits (`intn_qdq`).
       * ``mse_calib`` — the canonical low-bit OUTLIER-HANDLING fix the int5-cap
         abs-max test omitted: per-tensor symmetric int-N with the MSE-OPTIMAL
         clip step (`mse_optimal_step`, the codec-byte-closeable form of the fix;
         per-CHANNEL scales are NOT byte-closeable through the per-tensor-int8
         codec grammar — measured in the int5 retest, blows 118k→197k).
     re-encode through the REAL frontier split-brotli codec → MEASURED archive
     bytes (the rate term).
  3. NO-FAKE eval-on-shipped-bytes: RE-DECODE the byte-closed archive (the
     codec-int8-of-int-N codes that actually SHIP, NOT the ~off in-memory int-N)
     and exact-eval THOSE weights on the FULL 600-pair CPU authority via
     `RealScorerContext.exact_eval`. Latents/sidecar stay verbatim (decoder-only
     change), so the latent payload bytes are constant across Q.
  4. Build the table Q → (variant, d_seg, d_pose, bytes, bits/param, S),
     recompute S from components, and write the JSON response surface.

AUTHORITY: every score is ``[contest-CPU advisory]`` NON-PROMOTABLE. PTQ is the
$0 lower bound; QAT-FINETUNE (which trains the decoder to be robust at the coarse
grid) is the real version and does better — flagged, not run here (needs the
training loop, a separate campaign; the int5 best-shot already ran it and the
d_seg wall held). The frontier pointer stays pointer-only at 0.19110. MPS is
NEVER used (CLAUDE.md "MPS auth eval is NOISE"); CPU is authority. GT decode via
the cached `RealScorerContext` (yuv420_to_rgb, never PyAV).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

_FRONTIER_S = 0.19109982419209975
_SUB015 = 0.15
_SUB019 = 0.19


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mse_optimal_step_any_nbits(
    tensor: torch.Tensor,
    nbits: int,
    *,
    n_grid: int = 64,
    lo_ratio: float = 0.30,
    hi_ratio: float = 1.0,
) -> float:
    """Per-tensor symmetric int-N MSE-optimal clip step for ANY nbits >= 2.

    Mirrors ``tac.frontier_int5_qat.mse_optimal_step`` (the canonical outlier-handling
    calibration) but computes ``qmax = 2**(nbits-1) - 1`` directly so it also covers
    int3 (the int5-helper ``_qmax_for_nbits`` rejects nbits < 4). Searches the clip-ratio
    grid on the REAL tensor and returns the step minimizing the real round-trip MSE.
    """
    qmax = 2 ** (int(nbits) - 1) - 1
    t = tensor.detach().float()
    abs_max = t.abs().max()
    if float(abs_max) == 0.0:
        return 1.0
    best_step = float(abs_max / qmax)
    best_mse = float("inf")
    for i in range(n_grid + 1):
        r = lo_ratio + (hi_ratio - lo_ratio) * (i / n_grid)
        step = r * float(abs_max) / qmax
        if step <= 0.0:
            continue
        q = (t / step).round().clamp(-qmax, qmax) * step
        mse = float((q - t).pow(2).mean())
        if mse < best_mse:
            best_mse = mse
            best_step = step
    return best_step


def _calibrated_requant_state_dict(
    dec_sd: dict[str, torch.Tensor],
    nbits: int,
    *,
    weight_min_dim: int = 2,
) -> dict[str, torch.Tensor]:
    """Per-tensor symmetric int-N with the MSE-OPTIMAL clip step (outlier handling).

    The codec-byte-closeable canonical low-bit fix: instead of abs-max scale (one
    outlier sets the grid, starving the bulk), the per-tensor step minimizes the
    real round-trip MSE (clip the heavy tail, finer resolution for the bulk). Still
    one scale per tensor → byte-close compatible with the per-tensor-int8 codec.
    """
    qmax = 2 ** (int(nbits) - 1) - 1
    out: dict[str, torch.Tensor] = {}
    for k, v in dec_sd.items():
        if not torch.is_tensor(v) or v.dim() < weight_min_dim:
            out[k] = v.clone()
            continue
        step = _mse_optimal_step_any_nbits(v, nbits)
        out[k] = (v / step).round().clamp_(-qmax, qmax) * step
    return out


def main(argv: list[str] | None = None) -> int:
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--bits",
        default="8,7,6,5,4,3",
        help="comma-separated bit-depths to sweep (default 8,7,6,5,4,3)",
    )
    ap.add_argument(
        "--variants",
        default="absmax,mse_calib",
        help="comma-separated PTQ variants: absmax,mse_calib (default both)",
    )
    ap.add_argument(
        "--eval-pairs",
        type=int,
        default=600,
        help="pairs for the CPU-authority exact eval (600 = full contest)",
    )
    ap.add_argument(
        "--targets-cache",
        default="experiments/results/capstone_gt_targets_cache",
        help="GT-targets cache dir (the cached CPU-authority targets)",
    )
    ap.add_argument(
        "--out-dir",
        default=".omx/research",
        help="where to write the JSON response surface",
    )
    ap.add_argument(
        "--scratch-dir",
        default=".omx/tmp/qaxis_scratch",
        help="scratch dir for byte-closed candidate archives (auto-cleaned)",
    )
    args = ap.parse_args(argv)

    from tac.contest_score import (
        UNCOMPRESSED_SIZE_BYTES,
        compute_contest_score,
        pose_term,
        rate_term,
        seg_term,
    )
    from tac.frontier_decoder_ptq import (
        FRONTIER_ARCHIVE,
        build_frontier_decoder,
        decode_frontier_member,
        reencode_frontier_archive,
    )
    from tac.post_hoc_weight_shrink import requantize_decoder_state_dict
    from tac.torch_vehicle.driver import import_vendored_bundle
    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.torch_vehicle.vendored_imports import import_vendored

    bits = [int(b) for b in args.bits.split(",") if b.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    import_vendored_bundle()
    video_path = import_vendored("data").get_default_video_path()

    # ── decode the frontier decoder (the borrowed substrate) ──────────────────────
    member = decode_frontier_member(FRONTIER_ARCHIVE)
    dec_sd = member.state_dict
    total_params = sum(int(v.numel()) for v in dec_sd.values() if torch.is_tensor(v))
    weight_params = sum(
        int(v.numel()) for v in dec_sd.values() if torch.is_tensor(v) and v.dim() >= 2
    )
    print(
        f"[qaxis] frontier decoded: total_params={total_params} "
        f"weight_params(>=2D,quantized)={weight_params}",
        flush=True,
    )

    # ── CPU AUTHORITY scorer context (full eval_pairs; NEVER MPS) ─────────────────
    eval_pairs = int(args.eval_pairs)
    t_ctx = time.time()
    ctx = RealScorerContext(
        video_path,
        device="cpu",
        train_device="cpu",
        split_by_head=False,
        max_pairs=eval_pairs,
        targets_cache=args.targets_cache,
    )
    print(f"[qaxis] scorer ctx built ({time.time() - t_ctx:.1f}s, eval_pairs={eval_pairs})", flush=True)

    scratch_dir = Path(args.scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    def _measure(variant: str, q: int) -> dict:
        """Quantize → byte-close → RE-DECODE (eval-on-shipped-bytes) → exact-eval."""
        mode = f"int{q}"
        if variant == "absmax":
            shrunk_sd = requantize_decoder_state_dict(dec_sd, mode)
        elif variant == "mse_calib":
            shrunk_sd = _calibrated_requant_state_dict(dec_sd, q)
        else:
            raise ValueError(f"unknown variant {variant!r}")

        # in-memory weight perturbation vs the fp-decoded frontier (diagnostic only)
        inmem_err = max(
            float((shrunk_sd[k] - dec_sd[k]).abs().max())
            for k in dec_sd
            if torch.is_tensor(dec_sd[k]) and dec_sd[k].dim() >= 2
        )

        scratch = scratch_dir / f"_cand_{variant}_int{q}.zip"
        bytes_meas = reencode_frontier_archive(member, shrunk_sd, scratch)

        # NO-FAKE eval-on-shipped-bytes: re-decode the byte-closed archive (the
        # codec-int8-of-int-N codes that actually ship).
        shipped = decode_frontier_member(scratch)
        eval_dec = build_frontier_decoder(shipped.state_dict).to("cpu").eval()
        eval_latents = shipped.latents[:eval_pairs]
        # the re-decoded (shipped) weights vs the in-memory shrunk weights — the
        # codec-int8 storage error. At Q=8 this is exactly 0; at Q<8 it is the
        # int8-of-int-N rounding the int5-cap memo flagged.
        shipped_vs_inmem_err = max(
            float((shipped.state_dict[k] - shrunk_sd[k]).abs().max())
            for k in shrunk_sd
            if torch.is_tensor(shrunk_sd[k]) and shrunk_sd[k].dim() >= 2
        )

        t0 = time.time()
        res = ctx.exact_eval(eval_dec, eval_latents, bytes_meas)
        dt = time.time() - t0
        d_seg = float(res["seg_distortion"])
        d_pose = float(res["pose_distortion"])
        # ALL score math via tac.contest_score (the canonical, byte-identical-to-
        # upstream/evaluate.py:92 surface; never reinvent the 25/N coefficient).
        s = compute_contest_score(d_seg, d_pose, bytes_meas)
        bits_per_param = (q * weight_params) / total_params  # approx (biases fp32)
        row = {
            "variant": variant,
            "bits": q,
            "d_seg": d_seg,
            "d_pose": d_pose,
            "archive_bytes": bytes_meas,
            "rate_term": rate_term(bytes_meas),
            "seg_term": seg_term(d_seg),
            "pose_term": pose_term(d_pose),
            "bits_per_param_approx": bits_per_param,
            "score": s,
            "inmem_weight_err_vs_fp": inmem_err,
            "shipped_vs_inmem_weight_err": shipped_vs_inmem_err,
            "eval_seconds": dt,
            "crosses_sub015": s < _SUB015,
            "crosses_sub019": s < _SUB019,
        }
        print(
            f"[qaxis] {variant:9s} int{q}: d_seg={d_seg:.6f} d_pose={d_pose:.6f} "
            f"bytes={bytes_meas} S={s:.5f} (eval {dt:.1f}s, ship_err={shipped_vs_inmem_err:.2e})",
            flush=True,
        )
        try:
            scratch.unlink()
        except OSError:
            pass
        return row

    rows: list[dict] = []
    for variant in variants:
        for q in bits:
            rows.append(_measure(variant, q))

    # ── analysis: S-minimizing bit-depth, crossings, d_seg sensitivity ────────────
    best = min(rows, key=lambda r: r["score"])
    sub015 = [r for r in rows if r["crosses_sub015"]]
    sub019 = [r for r in rows if r["crosses_sub019"]]

    # d_seg(Q) sensitivity per variant: how fast d_seg degrades as bits drop
    sensitivity: dict[str, list] = {}
    for variant in variants:
        vr = sorted(
            [r for r in rows if r["variant"] == variant], key=lambda r: -r["bits"]
        )
        chain = []
        for i in range(1, len(vr)):
            hi, lo = vr[i - 1], vr[i]
            d_d_seg = lo["d_seg"] - hi["d_seg"]
            d_bytes = lo["archive_bytes"] - hi["archive_bytes"]
            chain.append(
                {
                    "from_bits": hi["bits"],
                    "to_bits": lo["bits"],
                    "delta_d_seg": d_d_seg,
                    "delta_bytes": d_bytes,
                    "delta_S": lo["score"] - hi["score"],
                    "d_seg_per_bit_dropped": d_d_seg,  # per 1-bit step
                    "bytes_saved": -d_bytes,
                }
            )
        sensitivity[variant] = chain

    out = {
        "schema": "qaxis_bitdepth_response_surface.v1",
        "utc": _now(),
        "axis_tag": "[contest-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "frontier_pointer_S": _FRONTIER_S,
        "frontier_archive_sha256": "b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e",
        "frontier_archive_bytes": 177169,
        "method": "post-hoc PTQ (no training); per-tensor symmetric int-N; byte-closed "
        "through the REAL frontier split-brotli codec; eval-on-shipped-bytes (re-decode); "
        "full 600-pair CPU-authority exact eval",
        "eval_pairs": eval_pairs,
        "total_params": total_params,
        "weight_params_quantized": weight_params,
        "bits_swept": bits,
        "variants": variants,
        "rows": rows,
        "s_minimizing": best,
        "sub015_crossings": sub015,
        "sub019_crossings": sub019,
        "d_seg_sensitivity_per_step": sensitivity,
        "ptq_vs_qat_caveat": (
            "These are PTQ (post-hoc, no training) numbers — the $0 LOWER BOUND on what "
            "bit-shrink costs. QAT-FINETUNE (train the decoder to be robust at the coarse "
            "grid) is the REAL version and does better on d_pose (the int5 best-shot "
            "recovered d_pose -89%) but the d_seg wall held (-9.5% only): the d_seg-critical "
            "early/low-res stages need finer-than-int-N per-CHANNEL resolution that no "
            "per-tensor scale provides AND per-channel is NOT byte-closeable through the "
            "per-tensor-int8 codec grammar (measured: 118k->197k). The QAT-finetune column "
            "requires the training run (a separate campaign)."
        ),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"qaxis_bitdepth_response_surface_{_now().replace(':', '').replace('-', '')}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[qaxis] wrote {out_path}", flush=True)
    print(
        f"[qaxis] S-min: {best['variant']} int{best['bits']} S={best['score']:.5f} | "
        f"sub015={len(sub015)} sub019={len(sub019)} | frontier={_FRONTIER_S:.5f}",
        flush=True,
    )

    # cleanup scratch dir if empty
    try:
        if not any(scratch_dir.iterdir()):
            scratch_dir.rmdir()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
