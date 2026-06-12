#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""One-shot REDUCED-N exact d_seg comparator: lever (argmax-flip) EMA shadow vs
control (CE) EMA shadow, both byte-closed through the SAME codec + parse-back,
evaluated on the FIRST ``--n-pairs`` GT pairs on the CPU authority.

WHY (the contention salvage): the async 600-pair authority eval crawls under
heavy CPU contention (10+ sister smoke procs). This runs ONE focused eval per
arm on a reduced pair count so we get a REAL, MEASURED, matched d_seg read
NOW instead of waiting hours for the saturated 600-pair async cadence. The
d_seg / d_pose are the genuine DistortionNet argmax-flip / pose-MSE on real GT
(``yuv420_to_rgb``); only the pair COUNT is reduced.

Authority: ``[contest-CPU advisory, reduced-n]`` NON-PROMOTABLE. Reduced-n d_seg
is a DIRECTIONAL read of the same quantity the 600-pair eval measures (the mean
argmax-flip rate over the evaluated pairs), not the contest 600-sample number.
Frontier UNMOVED. This gates, never IS, a paired contest-CPU+CUDA exact eval.

Both arms are byte-closed identically (same build_archive / parse_archive), so
the d_seg gap is the LOSS-FUNCTION effect (soft_cosine vs CE) from the SAME init.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

_LEVER_DEFAULT = Path("experiments/results/lever_ab_argmaxflip_20260612T123136Z")
_CONTROL = Path("experiments/results/torch_vehicle_full_mps_basin_bc20_n600")
_STATE = "torch_vehicle_checkpoint_state.pt"
_MANIFEST = "torch_vehicle_checkpoint_manifest.json"


def _load_ema(ckpt_dir: Path) -> tuple[dict, torch.Tensor, dict]:
    """Load the EMA shadow (decoder state_dict + latents) + manifest from a run
    dir's checkpoint. The EMA shadow is the INFERENCE/export artifact (the EMA
    non-negotiable) — not the live weights."""
    blob = torch.load(ckpt_dir / _STATE, map_location="cpu", weights_only=False)
    man = json.loads((ckpt_dir / _MANIFEST).read_text())
    ema_sd = {k: v.cpu() for k, v in blob["ema_decoder"].items()}
    ema_latents = blob["ema_latents"].cpu()
    return ema_sd, ema_latents, man


def _eval_arm(
    label: str,
    ckpt_dir: Path,
    *,
    n_pairs: int,
    base_channels: int,
    latent_dim: int,
    video_path: str,
    vendored,
    distortion_net,
    score_mod,
) -> dict:
    ema_sd, ema_latents, man = _load_ema(ckpt_dir)
    # Byte-close through the SAME codec, then parse back (the contest-visible
    # int8-dequantized artifact) — identical pipeline for both arms.
    archive = vendored.build_archive(
        ema_sd,
        ema_latents,
        meta_dict={
            "n_pairs": int(ema_latents.shape[0]),
            "latent_dim": latent_dim,
            "base_channels": base_channels,
            "eval_size": [384, 512],
        },
    )
    archive_bytes = len(archive)
    eval_sd, eval_latents, _meta = vendored.parse_archive(archive)
    model = vendored.HNeRVDecoder(
        latent_dim=latent_dim, base_channels=base_channels, eval_size=(384, 512)
    )
    model.load_state_dict(dict(eval_sd))
    model.eval()
    # Reduced-N: evaluate only the first n_pairs (evaluate_decoder iterates
    # latents.shape[0], so truncating the latents truncates the pair count).
    z = eval_latents[:n_pairs]
    dist = score_mod.evaluate_decoder(
        model, z, distortion_net, video_path, batch_pairs=8, device="cpu"
    )
    return {
        "label": label,
        "ckpt_stage": int(man["stage_index"]),
        "ckpt_epoch_in_stage": int(man["epoch_in_stage"]),
        # stage 0 => global epoch == epoch_in_stage (both arms forked in stage 1).
        "global_epoch": int(man["epoch_in_stage"]),
        "n_pairs_evaluated": int(min(n_pairs, eval_latents.shape[0])),
        "d_seg": float(dist["seg_distortion"]),
        "d_pose": float(dist["pose_distortion"]),
        "archive_bytes": archive_bytes,
        "best_score_so_far": float(man["best_score"]),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lever", type=Path, default=_LEVER_DEFAULT)
    p.add_argument("--control", type=Path, default=_CONTROL)
    p.add_argument("--n-pairs", type=int, default=48,
                   help="reduced pair count for the fast directional read")
    p.add_argument("--base-channels", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=28)
    p.add_argument("--video-path", type=Path, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(argv)

    from tac.score_aware_loop.targets import load_frozen_distortion_net
    from tac.torch_vehicle.driver import import_vendored_bundle
    from tac.torch_vehicle.vendored_imports import import_vendored

    vendored = import_vendored_bundle()
    score_mod = import_vendored("score")
    video_path = args.video_path
    if video_path is None:
        video_path = import_vendored("data").get_default_video_path()
    distortion_net = load_frozen_distortion_net(device="cpu")

    common = {
        "n_pairs": args.n_pairs,
        "base_channels": args.base_channels,
        "latent_dim": args.latent_dim,
        "video_path": str(video_path),
        "vendored": vendored,
        "distortion_net": distortion_net,
        "score_mod": score_mod,
    }
    print(f"[oneshot] reduced-n exact eval, n_pairs={args.n_pairs} "
          f"[contest-CPU advisory, reduced-n] NON-PROMOTABLE", flush=True)
    ctrl = _eval_arm("CONTROL_CE", args.control, **common)
    print(f"[oneshot] CONTROL_CE ep={ctrl['global_epoch']} "
          f"d_seg={ctrl['d_seg']:.6f} d_pose={ctrl['d_pose']:.6f}", flush=True)
    lev = _eval_arm("LEVER_soft_cosine", args.lever, **common)
    print(f"[oneshot] LEVER ep={lev['global_epoch']} "
          f"d_seg={lev['d_seg']:.6f} d_pose={lev['d_pose']:.6f}", flush=True)

    d_seg_delta = lev["d_seg"] - ctrl["d_seg"]
    d_pose_delta = lev["d_pose"] - ctrl["d_pose"]
    verdict = (
        "LEVER<CE (argmax-flip surrogate LOWERS d_seg)" if d_seg_delta < 0
        else "tie" if abs(d_seg_delta) < 1e-9
        else "CE<LEVER (vendored CE LOWERS d_seg)"
    )
    result = {
        "n_pairs": args.n_pairs,
        "control": ctrl,
        "lever": lev,
        "d_seg_delta_lever_minus_control": d_seg_delta,
        "d_pose_delta_lever_minus_control": d_pose_delta,
        "verdict": verdict,
        "authority": "[contest-CPU advisory, reduced-n] NON-PROMOTABLE",
        "note": (
            "both arms byte-closed identically; d_seg gap is the loss-function "
            "effect (soft_cosine vs CE) from the SAME forkpoint init. Reduced-n "
            "is a directional read, NOT the contest 600-sample number."
        ),
    }
    print("\n" + json.dumps(result, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"\n[oneshot] wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
