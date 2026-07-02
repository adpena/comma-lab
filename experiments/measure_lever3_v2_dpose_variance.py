# SPDX-License-Identifier: MIT
"""Lever-3 v2 vs v1 d_pose-VARIANCE A/B — the point of v2 (REAL frozen scorer, CPU).

The fork-arm trajectory exposed d_pose variance (best 0.00043 but spikes to
0.0046, a 10x swing). v2 (residual, sin-bounded, later-stage FiLM) should show
LOWER d_pose variance than v1 (stem, direct ``gamma*x+beta``) at comparable mean.

This is an APPLES-TO-APPLES A/B: same REAL forkpoint decoder (base_ch=20, 600
pairs) + same latents + same GT pose targets (real frozen PoseNet) + same
optimizer / lr / steps / seed; the ONLY thing that differs is the FiLM structure
(v1 stem-direct vs v2 residual-sin-later). For each variant on each pair-SLICE we
run a short pose-conditioned descent (optimize the FiLM weights + that slice's
latents to lower the REAL PoseNet d_pose) and record the per-step d_pose
trajectory. The headline metric is the post-warmup d_pose STD (the volatility the
fork-arm exposed) and the d_pose MEAN.

Authority: torch-CPU TRUSTED (CLAUDE.md "local CPU + MLX GPU good"). The d_pose
here is ``[contest-CPU advisory]`` NON-PROMOTABLE (an in-loop research signal on
a fixed forkpoint, not a byte-closed ``upstream/evaluate.py`` row).

Runs SYNCHRONOUSLY (small slices, foreground). No GPU, no MPS, no background.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tac.torch_vehicle.driver import import_vendored_bundle
from tac.torch_vehicle.pose_film import PoseFiLMHNeRVWrapper  # v1
from tac.torch_vehicle.pose_film_v2 import PoseFiLMHNeRVWrapperV2  # v2
from tac.torch_vehicle.vendored_imports import import_vendored

_FORKPOINT = Path(
    "experiments/results/forkpoints/basin_bc20_20260612T121523Z/best"
)
_TARGETS_CACHE = Path(
    "experiments/results/torch_vehicle_mps_descent_ab/gt_targets_cache/gt_targets_n48.pt"
)


def _load_forkpoint(device: str):
    v = import_vendored_bundle()
    dec_sd = torch.load(_FORKPOINT / "best_ema_decoder.pt", map_location=device, weights_only=False)
    if "stem.weight" not in dec_sd and isinstance(dec_sd, dict) and "state_dict" in dec_sd:
        dec_sd = dec_sd["state_dict"]
    latents = torch.load(_FORKPOINT / "best_ema_latents.pt", map_location=device, weights_only=False)
    base_channels = dec_sd["stem.weight"].shape[0] // (6 * 8)
    latent_dim = dec_sd["stem.weight"].shape[1]
    dec = v.HNeRVDecoder(latent_dim=latent_dim, base_channels=base_channels).to(device)
    dec.load_state_dict(dec_sd)
    dec.eval()
    return v, dec, latents, base_channels, latent_dim


def _real_posenet(device: str):
    """The frozen upstream PoseNet via DistortionNet (CPU-TRUSTED authority)."""
    from tac.score_aware_loop.targets import load_frozen_distortion_net

    return load_frozen_distortion_net(device=device)


def _d_pose(net, render_b2chw: torch.Tensor, pose_gt: torch.Tensor) -> torch.Tensor:
    """REAL PoseNet d_pose = MSE(posenet(render), GT pose). ``render_b2chw`` is the
    wrapper output ``(B, 2, 3, 384, 512)`` = (b,t,c,h,w) float [0,255]; the upstream
    DistortionNet.preprocess_input expects (b,t,h,w,c), so permute to BHWC."""
    render_bthwc = render_b2chw.permute(0, 1, 3, 4, 2).contiguous()  # (B,2,384,512,3)
    posenet_in, _seg = net.preprocess_input(render_bthwc)
    pose_out = net.posenet(posenet_in)
    pose_pred = pose_out["pose"][:, :6]
    return torch.nn.functional.mse_loss(pose_pred, pose_gt)


def _build_wrapper(variant: str, dec, n_pairs: int, seed: int, *, film_scale: float):
    """Build the FiLM wrapper for the variant with a SEEDED non-identity FiLM
    perturbation (so both arms start away from the trivial identity and the descent
    has something to do). Same seed across variants for apples-to-apples."""
    if variant == "v1":
        w = PoseFiLMHNeRVWrapper(dec, n_pairs=n_pairs, film_hidden=8)
    elif variant == "v2":
        w = PoseFiLMHNeRVWrapperV2(dec, n_pairs=n_pairs, cond_dim=32, film_scale=film_scale)
    else:
        raise ValueError(variant)
    return w


def _run_arm(
    *,
    variant: str,
    v,
    dec,
    latents: torch.Tensor,
    pose_gt: torch.Tensor,
    net,
    slice_idx: tuple[int, int],
    steps: int,
    lr: float,
    seed: int,
    film_scale: float,
    device: str,
) -> dict:
    """One arm: build the FiLM wrapper, descend d_pose for ``steps`` (optimizing
    FiLM weights + the slice latents), record the per-step d_pose trajectory."""
    i, j = slice_idx
    n = j - i
    torch.manual_seed(seed)
    w = _build_wrapper(variant, dec, n_pairs=n, seed=seed, film_scale=film_scale).to(device)
    w.set_stored_pose(pose_gt.to(device))
    # FREEZE the vendored decoder (the forkpoint is fixed); train FiLM + latents only.
    for p in w.decoder.parameters():
        p.requires_grad_(False)
    z = latents[i:j].clone().to(device).requires_grad_(True)
    idx = torch.arange(n, device=device)
    film_params = list(w.pose_film.parameters())
    opt = torch.optim.Adam([{"params": film_params, "lr": lr}, {"params": [z], "lr": lr}])

    traj: list[float] = []
    for _step in range(steps):
        opt.zero_grad()
        render = w(z, idx)  # (n, 2, 3, 384, 512)
        loss = _d_pose(net, render, pose_gt.to(device))
        loss.backward()
        opt.step()
        traj.append(float(loss.detach().item()))
    return {"variant": variant, "slice": list(slice_idx), "traj": traj}


def _summarize(traj: list[float], warmup: int) -> dict:
    post = traj[warmup:] if len(traj) > warmup else traj
    return {
        "mean": statistics.fmean(post),
        "std": statistics.pstdev(post) if len(post) > 1 else 0.0,
        "min": min(post),
        "max": max(post),
        "swing_ratio": (max(post) / min(post)) if min(post) > 0 else float("inf"),
        "n_post_warmup": len(post),
        "first": traj[0],
        "last": traj[-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--film-scale", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--slices", default="0:8,8:16", help="comma list of i:j pair slices")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if str(args.device).lower().startswith("mps"):
        raise SystemExit("MPS is NEVER the authority (CLAUDE.md). Use --device cpu.")

    t0 = time.time()
    v, dec, latents, base_ch, latent_dim = _load_forkpoint(args.device)
    blob = torch.load(_TARGETS_CACHE, map_location="cpu", weights_only=False)
    pose_all = blob["pose"].float()

    slices = [tuple(int(x) for x in s.split(":")) for s in args.slices.split(",")]
    results: list[dict] = []
    for (i, j) in slices:
        pose_gt = pose_all[i:j]
        for variant in ("v1", "v2"):
            r = _run_arm(
                variant=variant, v=v, dec=dec, latents=latents, pose_gt=pose_gt,
                net=_real_posenet(args.device), slice_idx=(i, j),
                steps=args.steps, lr=args.lr, seed=args.seed,
                film_scale=args.film_scale, device=args.device,
            )
            r["summary"] = _summarize(r["traj"], args.warmup)
            results.append(r)
            s = r["summary"]
            print(
                f"[{variant}] slice {i}:{j}  d_pose mean={s['mean']:.6f} "
                f"std={s['std']:.6f} swing={s['swing_ratio']:.2f}x "
                f"min={s['min']:.6f} max={s['max']:.6f}"
            )

    # Aggregate v2-vs-v1 std reduction per slice.
    print("\n=== d_pose VARIANCE (std) reduction: v2 vs v1 ===")
    by = {(r["variant"], tuple(r["slice"])): r["summary"] for r in results}
    agg = {"slices": []}
    for (i, j) in slices:
        s1 = by[("v1", (i, j))]
        s2 = by[("v2", (i, j))]
        std_red = (s1["std"] - s2["std"]) / s1["std"] if s1["std"] > 0 else 0.0
        mean_red = (s1["mean"] - s2["mean"]) / s1["mean"] if s1["mean"] > 0 else 0.0
        row = {
            "slice": [i, j],
            "v1_mean": s1["mean"], "v1_std": s1["std"], "v1_swing": s1["swing_ratio"],
            "v2_mean": s2["mean"], "v2_std": s2["std"], "v2_swing": s2["swing_ratio"],
            "std_reduction_frac": std_red, "mean_reduction_frac": mean_red,
            "v2_std_lower": s2["std"] < s1["std"],
        }
        agg["slices"].append(row)
        print(
            f"slice {i}:{j}: v1_std={s1['std']:.6f} v2_std={s2['std']:.6f} "
            f"-> std reduction {std_red*100:+.1f}%  (mean {mean_red*100:+.1f}%); "
            f"v2 lower std: {row['v2_std_lower']}"
        )

    n_better = sum(1 for r in agg["slices"] if r["v2_std_lower"])
    agg["v2_lower_std_count"] = n_better
    agg["n_slices"] = len(slices)
    agg["verdict"] = "V2_REDUCES_VARIANCE" if n_better == len(slices) else "MIXED_OR_NO"
    agg["elapsed_sec"] = time.time() - t0
    agg["axis_tag"] = "[contest-CPU advisory]"
    agg["promotable"] = False
    agg["base_channels"] = base_ch
    agg["config"] = vars(args)
    agg["per_arm"] = results
    print(f"\nVERDICT: {agg['verdict']} ({n_better}/{len(slices)} slices v2<v1 std)  "
          f"[{agg['elapsed_sec']:.1f}s] [contest-CPU advisory] NON-PROMOTABLE")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(agg, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
