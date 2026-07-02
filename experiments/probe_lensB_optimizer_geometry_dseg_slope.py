#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""LENS B optimizer-geometry $0 MPS A/B: does a different optimizer / LR-schedule
STEEPEN the d_seg-loss training-signal descent of the bc20 HNeRV stage-1 basin?

Measurement-first, $0 local MPS, NON-PROMOTABLE [macOS-MLX research-signal]:
we measure the MPS SEG-LOSS descent slope (the training signal, NOT an exact
score — the 13-min CPU exact eval is too costly per arm). Every arm starts from
the IDENTICAL fresh init (same decoder weights + latents + RNG), runs the same
number of epochs on the same n-pairs, full-MPS (no split-by-head, matching the
decisive basin's --no-split-by-head). The ONLY thing that varies is the
optimizer/LR config. We log per-epoch seg-loss + fit a power-law slope.

Arms (mandate item 2):
  A0  vendored stage-1 AdamW (adamw_lr=1e-3, cosine over `cos_epochs`)  [baseline]
  A1  AdamW adamw_lr=1e-3, CONSTANT lr (no cosine decay)  -> isolates LR-collapse
  A2  AdamW adamw_lr=3e-3 (3x peak), cosine                -> higher LR
  M1  Muon-EARLY muon_lr=2e-4 (vendored stage-8 lr), cosine -> Muon-from-stage-1
  M2  Muon-EARLY muon_lr=1e-3, cosine
  M3  Muon-EARLY muon_lr=0.03, cosine                      -> the CapstoneTrainer A/B lr
We deliberately keep epochs SHORT (default 250) on a SMALL n (default 48) for a
fast slope comparison; MPS is shared with a sister loss-probe.

Authority: [macOS-MLX research-signal] / NON-PROMOTABLE. seg-loss is a training
proxy; the pointer 0.19110 is UNMOVED. No paid, no PR.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path


def _power_slope(eps, vals, ep_min):
    import numpy as np

    eps = np.asarray(eps, float)
    vals = np.asarray(vals, float)
    m = (eps >= ep_min) & (vals > 0)
    if m.sum() < 3:
        return None, None
    b, la = np.polyfit(np.log(eps[m]), np.log(vals[m]), 1)
    return float(math.exp(la)), float(b)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--cos-epochs", type=int, default=250,
                    help="cosine horizon for the schedulers (use --epochs for a "
                         "full-decay arm; a large value ~= near-constant within window)")
    ap.add_argument("--base-channels", type=int, default=20)
    ap.add_argument("--latent-dim", type=int, default=28)
    ap.add_argument("--train-device", default="mps")
    ap.add_argument("--out", type=Path,
                    default=Path("experiments/results/lensB_optimizer_geometry_probe"))
    ap.add_argument("--arms", default="A0,A1,A2,M1,M2,M3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.optim as optim

    from tac.torch_vehicle.curriculum import build_curriculum
    from tac.torch_vehicle.driver import (
        TorchVehicleConfig,
        TorchVehicleDriver,
        import_vendored_bundle,
        _StageRuntime,
    )
    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.torch_vehicle.vendored_imports import import_vendored

    video_path = import_vendored("data").get_default_video_path()
    cfg = TorchVehicleConfig(
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        out_dir=args.out,
        device="cpu",
        train_device=args.train_device,
        split_by_head=False,          # full-MPS, matches the decisive basin
        seed=args.seed,
    )
    scorer = RealScorerContext(
        video_path,
        device="cpu",
        train_device=args.train_device,
        split_by_head=False,
        max_pairs=args.n_pairs,
        targets_cache=Path("experiments/results/capstone_gt_targets_cache"),
    )
    drv = TorchVehicleDriver(cfg, scorer=scorer, vendored=import_vendored_bundle())

    spec1 = build_curriculum()[0]                         # vendored stage1 spec
    spec1 = type(spec1)(**{**spec1.__dict__, "epochs": args.cos_epochs})

    # Fresh init (identical for every arm) — mirrors driver.run() stage-0 init exactly:
    #   decoder = _new_decoder(); latents = Parameter((randn(n_pairs, latent_dim)*0.1))
    dev = drv.train_device
    torch.manual_seed(args.seed)
    decoder0 = drv._new_decoder(dev)
    init_lat = (torch.randn(drv.n_pairs, cfg.latent_dim) * 0.1).to(dev)
    init_dec_sd = deepcopy(decoder0.state_dict())

    v = drv.v

    def make_runtime(arm):
        torch.manual_seed(args.seed)
        dec = drv._new_decoder(dev)
        dec.load_state_dict(init_dec_sd)
        lat = torch.nn.Parameter(init_lat.detach().clone().to(dev))
        muon_params, adamw_params = v.partition_params_for_muon(dec)

        def cosine(opt, base_floor=1e-3):
            def lam(ep):
                return max(0.5 * (1 + math.cos(math.pi * ep / args.cos_epochs)), base_floor)
            return optim.lr_scheduler.LambdaLR(opt, lam)

        def constant(opt):
            return optim.lr_scheduler.LambdaLR(opt, lambda ep: 1.0)

        muon_opt = None
        muon_sched = None
        if arm == "A0":   # vendored AdamW 1e-3 cosine
            groups = [{"params": adamw_params + muon_params, "lr": 1e-3},
                      {"params": [lat], "lr": 1e-3 * spec1.latent_lr_mult}]
            aopt = optim.AdamW(groups, weight_decay=0.0); asched = cosine(aopt)
        elif arm == "A1":  # AdamW 1e-3 CONSTANT (isolate LR collapse)
            groups = [{"params": adamw_params + muon_params, "lr": 1e-3},
                      {"params": [lat], "lr": 1e-3 * spec1.latent_lr_mult}]
            aopt = optim.AdamW(groups, weight_decay=0.0); asched = constant(aopt)
        elif arm == "A2":  # AdamW 3e-3 cosine (higher peak)
            groups = [{"params": adamw_params + muon_params, "lr": 3e-3},
                      {"params": [lat], "lr": 3e-3 * spec1.latent_lr_mult}]
            aopt = optim.AdamW(groups, weight_decay=0.0); asched = cosine(aopt)
        elif arm in ("M1", "M2", "M3"):  # Muon-EARLY on the 2D+ hidden weights
            mlr = {"M1": 2e-4, "M2": 1e-3, "M3": 0.03}[arm]
            muon_opt = v.Muon(muon_params, lr=mlr, momentum=0.95, nesterov=True,
                              ns_steps=5, weight_decay=spec1.muon_weight_decay)
            muon_sched = cosine(muon_opt)
            groups = [{"params": adamw_params, "lr": spec1.adamw_lr},  # 1e-3 stem/rgb/bias
                      {"params": [lat], "lr": spec1.adamw_lr * spec1.latent_lr_mult}]
            aopt = optim.AdamW(groups, weight_decay=0.0); asched = cosine(aopt)
        else:
            raise SystemExit(f"unknown arm {arm}")

        ema_dec = deepcopy(dec)
        return _StageRuntime(
            decoder=dec, latents=lat, ema_decoder=ema_dec,
            ema_latents=lat.detach().clone(),
            adamw_opt=aopt, muon_opt=muon_opt,
            adamw_sched=asched, muon_sched=muon_sched,
            muon_params=muon_params,
            adamw_clip_params=adamw_params + (muon_params if muon_opt is None else []),
        )

    results = {}
    for arm in args.arms.split(","):
        arm = arm.strip()
        t0 = time.time()
        rt = make_runtime(arm)
        drv._global_epoch = 0
        eps, segs = [], []
        for e in range(1, args.epochs + 1):
            drv._global_epoch = e
            loss, pose_mse, gn_a, gn_m = drv._train_one_epoch(rt, spec1, epoch_in_stage=e - 1)
            rt.adamw_sched.step()
            if rt.muon_sched is not None:
                rt.muon_sched.step()
            # seg-loss training signal = total loss / seg_weight (stage1 has cat_lambda=0,
            # full-MPS path: loss == seg_weight*seg_l, so seg_l = loss/seg_weight).
            seg_l = loss / spec1.seg_weight
            eps.append(e); segs.append(seg_l)
            if e % 25 == 0 or e == 1:
                print(f"[{arm}] ep={e:4d} seg_l={seg_l:.6f} loss={loss:.4f} "
                      f"gn_a={gn_a} gn_m={gn_m}", flush=True)
        a50, b50 = _power_slope(eps, segs, max(25, args.epochs // 5))
        wall = time.time() - t0
        results[arm] = {
            "final_seg_l": segs[-1],
            "min_seg_l": min(segs),
            "first_window_seg_l": segs[max(0, args.epochs // 5 - 1)],
            "power_a": a50, "power_b": b50,
            "wall_s": wall, "epochs": args.epochs,
            "seg_l_trace": list(zip(eps, segs)),
        }
        print(f"==> [{arm}] final_seg_l={segs[-1]:.6f} min={min(segs):.6f} "
              f"slope b={b50} (a={a50}) wall={wall:.1f}s", flush=True)

    # strip the verbose trace from the console summary but keep it on disk
    out_json = args.out / f"optimizer_geometry_probe_n{args.n_pairs}_ep{args.epochs}.json"
    out_json.write_text(json.dumps({
        "authority": "[macOS-MLX research-signal] NON-PROMOTABLE; seg-loss training "
                     "proxy, NOT an exact score; pointer 0.19110 UNMOVED",
        "config": vars(args) | {"out": str(args.out)},
        "results": results,
    }, default=str, indent=2))
    print("\n=== SUMMARY (lower final_seg_l + more-negative slope b = steeper descent) ===")
    print(f"{'arm':4s} {'final_seg_l':>12s} {'min_seg_l':>12s} {'slope_b':>9s} {'wall_s':>8s}")
    for arm, r in results.items():
        print(f"{arm:4s} {r['final_seg_l']:12.6f} {r['min_seg_l']:12.6f} "
              f"{(r['power_b'] or float('nan')):9.4f} {r['wall_s']:8.1f}")
    print(f"\nartifact: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
