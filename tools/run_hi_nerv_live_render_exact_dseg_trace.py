#!/usr/bin/env python3
"""b1_clean_ep1000_authority_trace.v1 — the DECISIVE never-measured surface (operator P0).

THE QUESTION (after clean-PR95 ep1000 archive eval gave d_seg=0.5048 / d_pose=168.8, like
EVERY config): is the MLX MODEL producing evaluator-bad frames (Category A: carrier/training/
topology), or does the export/inflate BRIDGE corrupt good frames (Category B)? Arm A's 21.7 dB
was PSNR-vs-source, NOT d_seg -- so it never answered this. This trace measures the ONE missing
surface: the LIVE MLX render scored through the EXACT upstream `DistortionNet` d_seg/d_pose.

METHOD (apples-to-apples with the archive surface; only the model source differs):
  1. decode source -> camera-res .raw (GT).
  2. load the checkpoint into the MLX renderer; render the first N pairs LIVE.
  3. lower each via the SAME `write_rgb_pair_to_raw` the inflate path uses (input_range="unit")
     -> live .raw (camera res, identical layout to the archive-inflate .raw).
  4. score live .raw vs source .raw with the EXACT `DistortionNet` (sanity-ladder `score_pairs`).
  5. compare live d_seg/d_pose to the archive eval (0.5048 / 168.8).

VERDICT:
  live d_seg ~ 0.50  => CATEGORY A: the MLX model itself renders evaluator-bad frames
                        (carrier/training/topology/PR95-fidelity failure; fix the model, not bytes).
  live d_seg GOOD    => CATEGORY B: the export/inflate/archive bridge corrupts good frames
                        (fix the bridge; the fp16~int8 + roundtrip atol 5e-2 argue against this).

[macOS-CPU advisory] / exact_pair_scorer -> mechanism_update_eligible only (never score roadmap).

Run detached (scorer load + N-pair render+score is a few min):
  nohup .venv/bin/python tools/run_hi_nerv_live_render_exact_dseg_trace.py \\
    --checkpoint-meta <run>/checkpoints/epoch000999_*.meta.json \\
    --archive-d-seg 0.504824 --archive-d-pose 168.83 \\
    --out <run>/b1_clean_ep1000_authority_trace.v1.json --n-pairs 48 </dev/null >/dev/null 2>&1 & disown
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO / "src"), str(_REPO / "upstream"), str(_REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _utc() -> str:
    import subprocess

    return subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True).stdout.strip()  # subprocess-no-check-OK: timestamp capture; date(1) failure yields empty string in a receipt, never a decision


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-meta", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--n-pairs", type=int, default=48, help="first-N pairs to render+score (sample)")
    ap.add_argument("--archive-d-seg", type=float, default=None, help="the archive-surface d_seg for comparison")
    ap.add_argument("--archive-d-pose", type=float, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    import numpy as np
    import torch

    import hi_nerv_renderer_sanity_ladder as ladder
    from watch_and_harvest_b1_checkpoint import (
        _build_pilot_model_and_config,
        _checkpoint_ref_from_meta,
        _load_checkpoint_meta,
        _load_npsd_state_into_model,
    )
    from tac.substrates._shared.inflate_runtime import write_rgb_pair_to_raw

    work = (args.work_dir or args.out.parent / "live_render_trace_work").resolve()
    if "/tmp" in str(work):
        raise SystemExit("work-dir must not be under /tmp")
    work.mkdir(parents=True, exist_ok=True)
    H, W, C = ladder._frame_dims()
    N = int(args.n_pairs)

    # 1. GT camera-res .raw (first 2N frames).
    src_raw = work / "source_sample.raw"
    ladder.decode_source_to_raw(_REPO / "upstream" / "videos" / "0.mkv", src_raw, max_frames=2 * N)

    # 2. load the checkpoint into the MLX renderer.
    meta = _load_checkpoint_meta(args.checkpoint_meta)
    if meta is None:
        raise SystemExit(f"could not load checkpoint meta {args.checkpoint_meta}")
    ref = _checkpoint_ref_from_meta(args.checkpoint_meta, meta)
    model, _cfg, _pc = _build_pilot_model_and_config()
    load_info = _load_npsd_state_into_model(model, ref.ema_state_path)

    # 3. render first N pairs LIVE -> live .raw via the SAME inflate writer.
    live_raw = work / "live_render_sample.raw"
    with torch.no_grad(), live_raw.open("wb") as fh:
        import mlx.core as mx

        for i in range(N):
            rgb0, rgb1 = model.reconstruct_pair(mx.array([i]))
            t0 = torch.from_numpy(np.asarray(rgb0, dtype=np.float32))  # (1,3,H,W) [0,1]
            t1 = torch.from_numpy(np.asarray(rgb1, dtype=np.float32))
            write_rgb_pair_to_raw(fh, t0, t1, input_range="unit")

    # 4. score live vs GT with the EXACT DistortionNet. score_pairs returns
    # {"device", "n_pairs_total", "pairs": [per-pair rows]}.
    res = ladder.score_pairs(live_raw, src_raw, list(range(N)), device=args.device)
    rows = res["pairs"]
    d_segs = [float(r["d_seg"]) for r in rows]
    d_poses = [float(r["d_pose"]) for r in rows]
    collapsed = [bool(r.get("comp_collapsed_to_one_class")) for r in rows]
    live_d_seg = sum(d_segs) / len(d_segs) if d_segs else None
    live_d_pose = sum(d_poses) / len(d_poses) if d_poses else None

    # 5. verdict.
    verdict = "INCONCLUSIVE"
    detail = ""
    if live_d_seg is not None:
        if live_d_seg > 0.40:
            verdict = "CATEGORY_A_MODEL_RENDERS_EVALUATOR_BAD_FRAMES"
            detail = (
                f"LIVE MLX render d_seg={live_d_seg:.4f} (~archive 0.50): the model ITSELF produces "
                "evaluator-bad frames. NOT a bridge bug. Fix = carrier/training/topology/PR95-fidelity "
                "(21dB PSNR is too blurry for SegNet argmax). Do NOT chase the export/inflate path."
            )
        elif live_d_seg < 0.20:
            verdict = "CATEGORY_B_BRIDGE_CORRUPTS_GOOD_FRAMES"
            detail = (
                f"LIVE MLX render d_seg={live_d_seg:.4f} is GOOD but the archive eval was ~0.50: the "
                "export/inflate/archive bridge corrupts good frames. Fix the bridge (MLX->numpy/layout/"
                "range/order), NOT the model."
            )
        else:
            verdict = "CATEGORY_A_PARTIAL_LIVE_ALSO_DEGRADED"
            detail = f"LIVE d_seg={live_d_seg:.4f} intermediate; model is partly the issue."

    artifact: dict[str, Any] = {
        "schema": "b1_clean_ep1000_authority_trace.v1",
        "utc": _utc(),
        "authority_tier": "exact_cpu_advisory",
        "metric_family": "exact_pair_scorer",
        "score_roadmap_update_eligible": False,
        "mechanism_update_eligible": True,
        "promotable": False,
        "checkpoint_meta": str(args.checkpoint_meta),
        "checkpoint_global_epoch": ref.global_epoch,
        "ema_state_path": str(ref.ema_state_path),
        "load_info": load_info if isinstance(load_info, dict) else str(load_info),
        "n_pairs_scored": len(rows),
        "live_render_surface": {
            "d_seg": live_d_seg,
            "d_pose": live_d_pose,
            "n_collapsed_to_one_class": sum(collapsed),
            "per_pair": rows,
        },
        "archive_surface_reference": {
            "d_seg": args.archive_d_seg,
            "d_pose": args.archive_d_pose,
            "note": "from the ep1000 B2 archive->inflate->evaluate.py exact eval",
        },
        "controlled_variable": "model source: LIVE MLX fp32 render vs archive->inflate numpy; same .raw writer + same DistortionNet",
        "verdict": verdict,
        "verdict_detail": detail,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"=== live-render exact-d_seg trace (N={len(rows)} pairs) ===")
    print(f"  LIVE   d_seg={live_d_seg} d_pose={live_d_pose} collapsed={sum(collapsed)}/{len(rows)}")
    print(f"  ARCHIVE d_seg={args.archive_d_seg} d_pose={args.archive_d_pose}")
    print(f"  VERDICT: {verdict}")
    print(f"  {detail}")
    print(f"  wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
