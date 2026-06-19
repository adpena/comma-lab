#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""$0 calibration-PTQ probe — does the BEST-SHOT per-tensor LSQ/outlier-clip step
recover the int5 d_seg collapse vs the original per-tensor abs-max, BEFORE any finetune?

The recursive-adversarial-review finding: the int5 Path-B cap was measured with a
per-tensor symmetric ABS-MAX quantizer (no per-channel, no LSQ, no outlier handling).
The frontier codec stores ONE per-tensor int8 scale, so per-channel scales are NOT
byte-closeable (MEASURED: they blow the archive 118,589 → 197,353 B). The byte-close-
COMPATIBLE form of the canonical low-bit fix is the per-tensor LSQ learned step +
outlier clip — calibrated to the MSE-optimal clip (~0.5× abs-max on the heavy-tailed
rate carriers). This probe measures the CALIBRATION-ONLY (zero-training) PTQ score for
abs-max vs LSQ-calibrated int5, byte-closed + RE-DECODED + eval'd on the REAL CPU
scorer (NEVER MPS). If calibration alone moves d_seg, the finetune has strong signal;
if not, the cap is closer to structural.

AUTHORITY: ``[contest-CPU advisory]`` NON-PROMOTABLE. Pointer UNMOVED 0.19110. $0.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

_RATE_DENOM = 37_545_489
_FRONTIER_S = 0.19109982419209975
_INT8_LOCAL_BASELINE_S = 0.1965


def _score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return 100.0 * d_seg + (10.0 * d_pose + 1e-12) ** 0.5 + 25.0 * archive_bytes / _RATE_DENOM


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--low-nbits", type=int, default=5)
    ap.add_argument("--eval-pairs", type=int, default=600)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--eval-redecode",
        action="store_true",
        default=True,
        help="eval the RE-DECODED byte-closed archive (eval-on-shipped-bytes, NO-FAKE)",
    )
    ap.add_argument(
        "--out-json",
        default="experiments/results/frontier_int5_qat_finetune/calibration_ptq_probe.json",
    )
    args = ap.parse_args(argv)

    torch.set_num_threads(int(args.threads))

    from tac.frontier_decoder_ptq import (
        FRONTIER_ARCHIVE,
        build_frontier_decoder,
        decode_frontier_member,
        reencode_frontier_archive,
    )
    from tac.frontier_int5_qat import (
        FrontierLSQQuantizer,
        FrontierQATConfig,
        hard_quantize_state_dict_lsq,
        hard_quantize_state_dict_to_nbits,
        per_tensor_nbits_for_decoder,
    )
    from tac.torch_vehicle.driver import import_vendored_bundle
    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.torch_vehicle.vendored_imports import import_vendored

    import_vendored_bundle()
    video_path = import_vendored("data").get_default_video_path()

    out_dir = Path(args.out_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    member = decode_frontier_member(FRONTIER_ARCHIVE)
    n_pairs = int(member.latents.shape[0])
    eval_pairs = min(int(args.eval_pairs), n_pairs)

    dec = build_frontier_decoder(member.state_dict)
    nb = per_tensor_nbits_for_decoder(dec, FrontierQATConfig(mode="uniform", low_nbits=int(args.low_nbits)))

    ctx = RealScorerContext(
        video_path,
        device="cpu",
        train_device="cpu",
        split_by_head=False,
        max_pairs=eval_pairs,
        targets_cache="experiments/results/capstone_gt_targets_cache",
    )

    def _eval_variant(label: str, shrunk_sd: dict) -> dict:
        scratch = out_dir / f"_calib_{label}.zip"
        bytes_meas = reencode_frontier_archive(member, shrunk_sd, scratch)
        if args.eval_redecode:
            # eval-on-shipped-bytes: re-decode the byte-closed archive (the weights that
            # SHIP, codec-int8-of-int5 — NOT the in-memory int5, which is ~5.6e-3 off).
            shipped = decode_frontier_member(scratch)
            eval_dec = build_frontier_decoder(shipped.state_dict).to("cpu").eval()
            eval_lat = shipped.latents[:eval_pairs]
        else:
            eval_dec = build_frontier_decoder(shrunk_sd).to("cpu").eval()
            eval_lat = member.latents[:eval_pairs]
        res = ctx.exact_eval(eval_dec, eval_lat, bytes_meas)
        d_seg = float(res["seg_distortion"])
        d_pose = float(res["pose_distortion"])
        s = _score(d_seg, d_pose, bytes_meas)
        try:
            scratch.unlink()
        except OSError:
            pass
        return {
            "label": label,
            "d_seg": d_seg,
            "d_pose": d_pose,
            "archive_bytes": bytes_meas,
            "rate_term": 25.0 * bytes_meas / _RATE_DENOM,
            "score": s,
        }

    # variant A: original per-tensor abs-max int5 (the cap's quantizer).
    sd_absmax = hard_quantize_state_dict_to_nbits(dec, nb)
    # variant B: LSQ-calibrated int5 (MSE-optimal clip, zero training).
    q = FrontierLSQQuantizer.from_decoder(dec, nb)
    sd_lsq = hard_quantize_state_dict_lsq(dec, q)

    t0 = time.time()
    print(f"[probe] eval_pairs={eval_pairs} low=int{args.low_nbits} redecode={args.eval_redecode}", flush=True)
    row_a = _eval_variant("absmax", sd_absmax)
    print(f"  absmax  S={row_a['score']:.4f} d_seg={row_a['d_seg']:.6f} d_pose={row_a['d_pose']:.6f} bytes={row_a['archive_bytes']} ({time.time()-t0:.0f}s)", flush=True)
    t1 = time.time()
    row_b = _eval_variant("lsq_calib", sd_lsq)
    print(f"  lsq     S={row_b['score']:.4f} d_seg={row_b['d_seg']:.6f} d_pose={row_b['d_pose']:.6f} bytes={row_b['archive_bytes']} ({time.time()-t1:.0f}s)", flush=True)

    d_seg_delta = row_b["d_seg"] - row_a["d_seg"]
    s_delta = row_b["score"] - row_a["score"]
    if s_delta < -1e-4:
        verdict = (
            f"LSQ-CALIBRATION ALONE MOVES THE SCORE: ΔS={s_delta:+.4f} (d_seg {d_seg_delta:+.6f}) "
            f"vs abs-max — the per-tensor outlier-clip lever has real signal; finetune should "
            f"recover more. [contest-CPU advisory]"
        )
    elif d_seg_delta < -1e-4:
        verdict = (
            f"LSQ-CALIBRATION cuts d_seg ({d_seg_delta:+.6f}) but bytes/pose offset it (ΔS={s_delta:+.4f}); "
            f"signal present, finetune needed to convert it. [contest-CPU advisory]"
        )
    else:
        verdict = (
            f"LSQ-CALIBRATION ≈ abs-max at the byte-closed grid (ΔS={s_delta:+.4f}, d_seg {d_seg_delta:+.6f}): "
            f"the calibration gain is washed out by the codec int8 re-quant; the finetune (training the "
            f"decoder robust to the grid) is the only remaining lever. [contest-CPU advisory]"
        )

    report = {
        "probe": "int5_lsq_calibration_vs_absmax",
        "authority": "[contest-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotable": False,
        "frontier_pointer_moved": False,
        "frontier_pointer_S": _FRONTIER_S,
        "int8_local_baseline_S": _INT8_LOCAL_BASELINE_S,
        "eval_pairs": eval_pairs,
        "low_nbits": int(args.low_nbits),
        "eval_redecode": bool(args.eval_redecode),
        "absmax": row_a,
        "lsq_calib": row_b,
        "d_seg_delta": d_seg_delta,
        "score_delta": s_delta,
        "verdict": verdict,
        "built_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    Path(args.out_json).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n=== CALIBRATION-PTQ PROBE VERDICT ===\n  {verdict}\n[wrote] {args.out_json}\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
