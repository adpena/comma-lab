# SPDX-License-Identifier: MIT
"""Lever-3 v2 real-scorer measurement: (1) d_seg DECOUPLING + (2) d_pose VARIANCE.

$0 real-frozen-scorer, torch-CPU (TRUSTED authority; NO MPS). All numbers
``[contest-CPU advisory]`` NON-PROMOTABLE until byte-closed exact eval.

MEASUREMENT 1 (the headline — d_seg DECOUPLING):
  On the real base_ch=20 forkpoint decoder + real frozen SegNet, vary the stored
  pose and recompute the CONTEST d_seg (per-pixel argmax-flip rate of f1's SegNet
  mask vs GT). For v2 (rgb_0-head residual FiLM) d_seg must be INVARIANT to the
  pose (f1 is FiLM-clean). For v1 (stem FiLM on the shared trunk) d_seg CHANGES
  with the pose (the coupling we are eliminating).

MEASUREMENT 2 (the stability win — d_pose VARIANCE):
  Short pose-conditioned optimization of the FiLM weights (small slice, ~12-20
  steps); track the d_pose trajectory. v2 (residual, head-local) vs v1 (direct
  stem). Report mean + std across steps. v2 should show LOWER variance at
  comparable/better mean (the residual containment + low-leverage win).

Apples-to-apples: SAME decoder, SAME pairs, SAME stored pose; ONLY the FiLM
structure differs.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from tac.torch_vehicle.driver import import_vendored_bundle
from tac.torch_vehicle.pose_film import PoseFiLMHNeRVWrapper  # v1 (stem FiLM)
from tac.torch_vehicle.pose_film_v2 import PoseFiLMHNeRVWrapperV2  # v2 (rgb_0 head)
from tac.torch_vehicle.scorer_context import RealScorerContext

_FORKPOINT = (
    "experiments/results/forkpoints/basin_bc20_20260612T121523Z/"
    "torch_vehicle_checkpoint_state.pt"
)
_VIDEO = "upstream/videos/0.mkv"
_EVAL = (384, 512)


def _roundtrip_bhwc(decoded_pair: torch.Tensor) -> torch.Tensor:
    """Replicate the vendored common.py eval roundtrip EXACTLY: decoded
    (B,2,3,384,512) -> bicubic up to (874,1164) -> bilinear down to (384,512) ->
    permute to BHWC -> clamp/round STE. Returns (B,2,384,512,3)."""
    B = decoded_pair.shape[0]
    flat = decoded_pair.reshape(B * 2, 3, _EVAL[0], _EVAL[1])
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
    clamped = bhwc.clamp(0, 255)
    return clamped + (clamped.round() - clamped).detach()


def _seg_pose(ctx: RealScorerContext, decoded_pair: torch.Tensor):
    """Real frozen scorer on roundtripped decoder output -> (seg_logits, pose6)."""
    bhwc = _roundtrip_bhwc(decoded_pair)
    return ctx.seg_pose_forward(bhwc)


def _contest_d_seg(seg_logits: torch.Tensor, seg_targets_hard: torch.Tensor) -> float:
    """Contest d_seg = per-pixel argmax-flip rate of the SegNet mask vs GT."""
    pred = seg_logits.argmax(dim=1)
    return (pred != seg_targets_hard).float().mean().item()


def _pose_mse(pose6: torch.Tensor, pose_targets: torch.Tensor) -> float:
    return F.mse_loss(pose6, pose_targets).item()


def _load_decoder(v, device: str):
    st = torch.load(_FORKPOINT, map_location="cpu", weights_only=False)
    dec = v.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=_EVAL).to(device)
    dec.load_state_dict(st["ema_decoder"])  # the inference shadow (BEST tracker)
    dec.eval()
    latents = st["ema_latents"].to(device)
    return dec, latents


def measure_decoupling(ctx, v, latents, n: int, device: str) -> dict:
    """MEASUREMENT 1: d_seg invariance to pose (v2) vs coupling (v1)."""
    idx = torch.arange(n, device=device)
    z = latents[idx]
    gt_seg = ctx.seg_targets_hard[:n].to(device)
    # Two distinct stored poses (the real GT pose and a scrambled pose).
    pose_a = ctx.pose_targets[:n].detach().clone().to(device)
    g = torch.Generator().manual_seed(20260613)
    pose_b = (pose_a + torch.randn(pose_a.shape, generator=g).to(device) * 3.0)

    out = {}
    for name, Wrapper in (("v1_stem", PoseFiLMHNeRVWrapper), ("v2_rgb0", PoseFiLMHNeRVWrapperV2)):
        # Fresh wrapper on the SAME decoder; TRAIN the FiLM a little so it is a
        # non-identity (otherwise both are trivially invariant at init).
        fresh = v.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=_EVAL).to(device)
        fresh.load_state_dict(_load_decoder(v, device)[0].state_dict())
        fresh.eval()
        w = Wrapper(fresh, n_pairs=n, film_hidden=8).to(device)
        gg = torch.Generator().manual_seed(7)
        film_mods = (
            list(w.pose_film.parameters()) if name == "v1_stem"
            else list(w.pose_mlp.parameters()) + list(w.film_resid.parameters())
        )
        with torch.no_grad():
            for p in film_mods:
                p.add_(torch.randn(p.shape, generator=gg).to(device) * 0.3)
        # d_seg under pose_a
        w.set_stored_pose(pose_a)
        with torch.no_grad():
            seg_a, _ = _seg_pose(ctx, w(z, idx))
        dseg_a = _contest_d_seg(seg_a, gt_seg)
        # d_seg under pose_b (scrambled)
        w.set_stored_pose(pose_b)
        with torch.no_grad():
            seg_b, _ = _seg_pose(ctx, w(z, idx))
        dseg_b = _contest_d_seg(seg_b, gt_seg)
        out[name] = {
            "d_seg_pose_a": dseg_a,
            "d_seg_pose_b": dseg_b,
            "d_seg_delta_abs": abs(dseg_a - dseg_b),
        }
    return out


def measure_dpose_variance(ctx, v, latents, n: int, steps: int, device: str, lr: float) -> dict:
    """MEASUREMENT 2: short pose-conditioned FiLM optimization; d_pose trajectory
    mean+std for v1 (stem) vs v2 (rgb_0 residual). Same decoder/pairs/pose."""
    idx = torch.arange(n, device=device)
    z = latents[idx]
    pose = ctx.pose_targets[:n].detach().clone().to(device)
    pose_tgt = ctx.pose_targets[:n].to(device)

    out = {}
    for name, Wrapper in (("v1_stem", PoseFiLMHNeRVWrapper), ("v2_rgb0", PoseFiLMHNeRVWrapperV2)):
        fresh = v.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=_EVAL).to(device)
        fresh.load_state_dict(_load_decoder(v, device)[0].state_dict())
        fresh.eval()
        w = Wrapper(fresh, n_pairs=n, film_hidden=8).to(device)
        w.set_stored_pose(pose)
        # Start from the NATURAL identity-init FiLM (how the real from-0 run
        # begins): NO random perturbation. The optimizer drives the FiLM up from
        # identity — this is the apples-to-apples training scenario for the
        # stability comparison (a random-perturbed init would inject an arm-
        # specific transient that confounds the variance signal).
        film_params = (
            list(w.pose_film.parameters()) if name == "v1_stem"
            else list(w.pose_mlp.parameters()) + list(w.film_resid.parameters())
        )
        for p in film_params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(film_params, lr=lr)
        traj = []
        for _ in range(steps):
            opt.zero_grad()
            decoded = w(z, idx)
            _seg, pose6 = _seg_pose(ctx, decoded)
            # d_pose train objective: the score-domain sqrt(10*mse) (contest pose term).
            mse = F.mse_loss(pose6, pose_tgt)
            loss = torch.sqrt(10.0 * mse + 1e-12)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(film_params, 1.0)
            opt.step()
            traj.append(mse.item())  # record the raw d_pose (pose MSE)
        t = torch.tensor(traj)
        out[name] = {
            "d_pose_traj": traj,
            "d_pose_mean": float(t.mean()),
            "d_pose_std": float(t.std(unbiased=False)),
            "d_pose_final": traj[-1],
            "d_pose_min": float(t.min()),
            "d_pose_max": float(t.max()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="pairs (small slice)")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--slices", type=int, default=2, help="distinct pair slices for variance")
    ap.add_argument(
        "--single-slice-offset",
        type=int,
        default=-1,
        help="if >=0, run ONLY the d_pose-variance slice at this pair offset "
        "(synchronous fast call; skips decoupling). Use to run each slice in its "
        "own foreground call.",
    )
    ap.add_argument(
        "--no-decoupling", action="store_true", help="skip MEASUREMENT 1"
    )
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    device = "cpu"  # TRUSTED authority; NO MPS for the exact metric.
    torch.manual_seed(0)
    v = import_vendored_bundle()
    # RealScorerContext with a cap so the GT-targets precompute returns in-call.
    if args.single_slice_offset >= 0:
        cap = max(args.single_slice_offset + args.n, args.n)
    else:
        cap = max(args.n * args.slices, args.n)
    ctx = RealScorerContext(
        _VIDEO,
        device=device,
        max_pairs=cap,
        targets_cache=".omx/tmp/lever3_v2_targets",
    )
    dec, latents = _load_decoder(v, device)

    result: dict = {
        "authority": "contest-CPU advisory (NON-PROMOTABLE; torch-CPU TRUSTED, no MPS)",
        "forkpoint": _FORKPOINT,
        "n_pairs_total": ctx.n_pairs,
        "n": args.n,
        "decoupling": {},
        "dpose_variance_slices": [],
    }

    # Single-slice fast path: run ONLY one variance slice (own foreground call).
    if args.single_slice_offset >= 0:
        off = args.single_slice_offset
        sl_latents = latents[off : off + args.n]

        class _SliceCtx:
            def __init__(self, base, off, n):
                self._base = base
                self.pose_targets = base.pose_targets[off : off + n]
                self.seg_targets_hard = base.seg_targets_hard[off : off + n]
                self.n_pairs = n

            def seg_pose_forward(self, bhwc):
                return self._base.seg_pose_forward(bhwc)

        sctx = _SliceCtx(ctx, off, args.n)
        slice_result = measure_dpose_variance(
            sctx, v, sl_latents, args.n, args.steps, device, args.lr
        )
        slice_result["slice_index"] = "single"
        slice_result["pair_offset"] = off
        result["dpose_variance_slices"].append(slice_result)
        text = json.dumps(result, indent=2)
        print(text)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text)
        return

    # MEASUREMENT 1: decoupling on the first slice.
    if not args.no_decoupling:
        result["decoupling"] = measure_decoupling(ctx, v, latents, args.n, device)

    # MEASUREMENT 2: d_pose variance on >=2 distinct slices (apples-to-apples).
    for s in range(args.slices):
        off = s * args.n
        # Build a per-slice context view by re-using the same ctx targets offset.
        sl_latents = latents[off : off + args.n]

        # Re-point the ctx targets to this slice by a thin shim (pose/seg targets
        # are indexed [:n] inside the measure fn, so slice the latents+targets).
        class _SliceCtx:
            def __init__(self, base, off, n):
                self._base = base
                self.pose_targets = base.pose_targets[off : off + n]
                self.seg_targets_hard = base.seg_targets_hard[off : off + n]
                self.n_pairs = n

            def seg_pose_forward(self, bhwc):
                return self._base.seg_pose_forward(bhwc)

        sctx = _SliceCtx(ctx, off, args.n)
        # latents passed in must be 0-indexed for the slice; idx=arange(n).
        slice_result = measure_dpose_variance(
            sctx, v, sl_latents, args.n, args.steps, device, args.lr
        )
        slice_result["slice_index"] = s
        slice_result["pair_offset"] = off
        result["dpose_variance_slices"].append(slice_result)

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
