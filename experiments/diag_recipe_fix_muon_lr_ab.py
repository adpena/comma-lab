# SPDX-License-Identifier: MIT
"""FAST targeted A/B: does the BUG-A fix (muon_lr 2e-4 -> 0.03 under muon_throughout)
change the d_seg descent through the PR95 curriculum's seg-surrogate stages?

This isolates the EXACT recipe lever the full campaign confounds, at a throughput
that fits an in-window run: ONE bundle/bridge load, a SMALL pair count, eval ONLY
at stage boundaries (the campaign's per-eval cost over 48 pairs is the wall). Same
ARCHITECTURE for both arms (base_ch=20, tie_depth=2, stored_latent); the ONLY
difference is the StageSpec muon_lr the curriculum uses, which is exactly what the
fix routes around.

Arm BUGGY  : configure_stage uses spec.muon_lr=2e-4 (the pre-fix curriculum value)
Arm FIXED  : configure_stage uses cfg.muon_lr=0.03 (the post-fix muon_throughout value)

We drive stages 1 (CE) -> 2 (tau_softplus) -> 3 (smooth_disagreement) so the
differentiator (the buggy curriculum plateaued ~0.0097 and smooth RAISED d_seg)
is exercised. d_seg measured on the EXACT torch-CPU SegNet (live + EMA, both
reported to confirm no shadow-lag). [macOS-CPU advisory], NON-PROMOTABLE, $0.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "upstream"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def build_once(n_pairs: int, base_ch: int, seed: int):
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    cache = torch.load(
        ROOT / "experiments/results/capstone_gt_targets_cache/gt_targets_n48.pt",
        weights_only=False,
    )
    seg_t = cache["seg"][:n_pairs].contiguous()
    pose_t = cache["pose"][:n_pairs].contiguous()
    from tac.score_aware_loop.targets import load_frozen_distortion_net

    net = load_frozen_distortion_net(device="cpu")
    pose_store = pose_t.float().cpu().numpy()
    return seg_t, pose_t, pose_store, net, CapstoneVqNervBundle, CapstoneVqNervConfig, TorchScorerBridge


def run_arm(arm: str, *, n_pairs: int, base_ch: int, seed: int, stage_epochs: list[int]):
    """Run stages 1->2->3 for one arm. arm in {'buggy','fixed'}."""
    from tac.capstone_vq_nerv.capstone_trainer import CapstoneTrainConfig, CapstoneTrainer
    from tac.mlx_pr95_port.curriculum import build_pr95_8stage_curriculum

    (seg_t, pose_t, pose_store, net, Bundle, BundleCfg, Bridge) = build_once(
        n_pairs, base_ch, seed
    )
    bundle = Bundle(
        BundleCfg(
            num_pairs=n_pairs, base_channels=base_ch, codebook_size=256,
            carrier="stored_latent", seed=seed, tie_depth=2,
        )
    )
    bridge = Bridge(
        net, seg_t, pose_t, seg_loss_form="ce_seg_loss",
        seg_weight=100.0, pose_weight=1.0, eval_roundtrip=True,
    )
    # BOTH arms get config muon_lr=0.03; the BUGGY arm forces the curriculum to use
    # the StageSpec value by selecting the pr95_adamw_then_muon... no: the bug is that
    # muon_throughout previously used spec.muon_lr. We reproduce the two regimes by
    # passing the schedule + (for buggy) temporarily monkeypatching back to spec value.
    cfg = CapstoneTrainConfig(
        epochs=10, batch_size=min(8, n_pairs), seed=seed,
        muon_lr=0.03, grad_clip=50.0, grad_clip_muon=50.0,
        eval_every=10, ema_decay=0.997, use_ema_for_eval=True,
        cosine_lr_schedule=True,
    )
    trainer = CapstoneTrainer(bundle, bridge, pose_store, cfg)

    stages = build_pr95_8stage_curriculum(total_epochs=sum(stage_epochs) * 8)
    # Override the first 3 stages' epochs to our small budget.
    from dataclasses import replace
    stages = [replace(stages[i], epochs=stage_epochs[i]) for i in range(3)]

    traj = []
    d_seg0 = trainer.exact_d_seg(use_ema=False)
    traj.append({"stage": "init", "epoch": 0, "d_seg_live": d_seg0,
                 "d_pose_live": trainer.mean_d_pose(use_ema=False)})
    print(f"[{arm}] init d_seg(live)={d_seg0:.5f}", flush=True)

    for i, spec in enumerate(stages):
        trainer.configure_stage(spec, optimizer_schedule="muon_throughout")
        if arm == "buggy":
            # Force the PRE-FIX behavior: the curriculum used spec.muon_lr (2e-4) +
            # spec.grad_clip_muon (1.0). opt_config is a FROZEN dataclass, so rebuild
            # it with the spec values (dataclasses.replace) — exactly what the pre-fix
            # configure_stage produced under muon_throughout.
            from dataclasses import replace as _dc_replace

            trainer.opt_config = _dc_replace(
                trainer.opt_config,
                muon_lr=spec.muon_lr,            # 2e-4 (the bug)
                grad_clip_muon=spec.grad_clip_muon,  # 1.0 (the bug)
                grad_clip=spec.grad_clip,        # 1.0 (the bug)
            )
        # fixed arm: opt_config already uses cfg.muon_lr=0.03 + grad_clip_muon=50 (post-fix)

        t0 = time.time()
        summary = trainer.run_stage_epochs(spec)
        d_seg_live = trainer.exact_d_seg(use_ema=False)
        d_seg_ema = trainer.exact_d_seg(use_ema=True)
        d_pose_live = trainer.mean_d_pose(use_ema=False)
        row = {
            "stage": spec.name, "stage_idx": i, "epochs": spec.epochs,
            "muon_lr": trainer.opt_config.muon_lr,
            "grad_clip_muon": trainer.opt_config.grad_clip_muon,
            "d_seg_live": d_seg_live, "d_seg_ema": d_seg_ema,
            "d_pose_live": d_pose_live,
            "d_seg_best_in_stage": summary.get("d_seg_best"),
            "wall_s": round(time.time() - t0, 1),
        }
        traj.append(row)
        print(f"[{arm}] {spec.name} muon_lr={row['muon_lr']:.1e} "
              f"clip={row['grad_clip_muon']:.0f} -> d_seg(live)={d_seg_live:.5f} "
              f"d_seg(ema)={d_seg_ema:.5f} ({row['wall_s']}s)", flush=True)
    return traj


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--base-ch", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage-epochs", type=int, nargs=3, default=[15, 25, 12])
    ap.add_argument("--out", default="experiments/results/diag_recipe_fix_muon_lr_ab")
    ap.add_argument("--arm", choices=("buggy", "fixed", "both"), default="both")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    arms = ["buggy", "fixed"] if args.arm == "both" else [args.arm]
    for arm in arms:
        results[arm] = run_arm(
            arm, n_pairs=args.n_pairs, base_ch=args.base_ch, seed=args.seed,
            stage_epochs=args.stage_epochs,
        )
        (out / f"{arm}_trajectory.json").write_text(json.dumps(results[arm], indent=1))

    if "buggy" in results and "fixed" in results:
        bf = results["fixed"][-1]["d_seg_live"]
        bb = results["buggy"][-1]["d_seg_live"]
        verdict = {
            "axis": "[macOS-CPU advisory]", "score_claim": False, "promotable": False,
            "n_pairs": args.n_pairs, "base_ch": args.base_ch,
            "stage_epochs": args.stage_epochs,
            "buggy_final_d_seg_live": bb, "fixed_final_d_seg_live": bf,
            "fixed_beats_buggy": bf < bb,
            "ratio_buggy_over_fixed": (bb / bf) if bf > 0 else None,
        }
        (out / "verdict.json").write_text(json.dumps(verdict, indent=1))
        print("\n=== VERDICT ===", flush=True)
        print(json.dumps(verdict, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
