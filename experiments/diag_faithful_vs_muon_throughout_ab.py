# SPDX-License-Identifier: MIT
"""BOUNDED $0 A/B: FAITHFUL PR95 (AdamW stages 1-7 + Muon stage 8) vs the
muon_throughout-fixed deviation — the recipe-choice head-to-head the n600
config decision needs.

The prior `diag_recipe_fix_muon_lr_ab.py` proved muon_throughout-fixed DESCENDS
(muon_lr 0.03 vs the buggy 2e-4). It did NOT compare faithful-vs-divergence.
Per CLAUDE.md "Substrate MUST be at OPTIMAL FORM" + the directive's
"DEFAULT to the faithful proven recipe unless a bounded $0 A/B justifies the
divergence", this harness runs the two recipes head-to-head at the SAME small
architecture (base_ch=20, tie_depth=2, stored_latent) so the ONLY difference is
the optimizer schedule.

Arm FAITHFUL : optimizer_schedule="pr95_adamw_then_muon" (AdamW stages 1-7, Muon
               stage 8 only) — the recipe that PROVABLY reached the 5.6e-4 basin
               at base_ch=36 (the existence proof the frontier rests on).
Arm MUON_TP  : optimizer_schedule="muon_throughout" + cfg.muon_lr=0.03 / clip=50
               — the #77 deviation, BUG-A-fixed.

We drive stages 1 (CE) -> 2 (tau_softplus) so the d_seg WORKHORSE stages run.
At this small scale we do NOT expect to reach the basin (that's the n600 paid
run); we ask the bounded question: does the faithful AdamW recipe DESCEND d_seg
comparably to (or better than) the muon_throughout deviation at matched arch +
matched epoch budget? If faithful descends comparably, faithful is the safe
default (it ALSO has the basin existence-proof). If muon_throughout decisively
beats faithful here, the divergence is justified.

d_seg measured on the EXACT torch-CPU SegNet (live + EMA). [macOS-CPU advisory],
NON-PROMOTABLE, $0.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "upstream"), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def run_arm(schedule: str, *, n_pairs: int, base_ch: int, seed: int,
            stage_epochs: list[int]):
    from dataclasses import replace

    from tac.capstone_vq_nerv.capstone_trainer import (
        CapstoneTrainConfig,
        CapstoneTrainer,
    )
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )
    from tac.mlx_pr95_port.curriculum import build_pr95_8stage_curriculum
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge
    from tac.score_aware_loop.targets import load_frozen_distortion_net

    cache = torch.load(
        ROOT / "experiments/results/capstone_gt_targets_cache/gt_targets_n48.pt",
        weights_only=False,
    )
    seg_t = cache["seg"][:n_pairs].contiguous()
    pose_t = cache["pose"][:n_pairs].contiguous()
    net = load_frozen_distortion_net(device="cpu")
    pose_store = pose_t.float().cpu().numpy()

    bundle = CapstoneVqNervBundle(
        CapstoneVqNervConfig(
            num_pairs=n_pairs, base_channels=base_ch, codebook_size=256,
            carrier="stored_latent", seed=seed, tie_depth=2,
        )
    )
    bridge = TorchScorerBridge(
        net, seg_t, pose_t, seg_loss_form="ce_seg_loss",
        seg_weight=100.0, pose_weight=1.0, eval_roundtrip=True,
    )
    # Faithful EMA decay = PR95's 0.999; muon_tp keeps the session's 0.997.
    ema_decay = 0.999 if schedule == "pr95_adamw_then_muon" else 0.997
    cfg = CapstoneTrainConfig(
        epochs=10, batch_size=min(8, n_pairs), seed=seed,
        muon_lr=0.03, grad_clip=50.0, grad_clip_muon=50.0,
        eval_every=10, ema_decay=ema_decay, use_ema_for_eval=True,
        cosine_lr_schedule=True,
    )
    trainer = CapstoneTrainer(bundle, bridge, pose_store, cfg)

    stages = build_pr95_8stage_curriculum(total_epochs=sum(stage_epochs) * 8)
    stages = [replace(stages[i], epochs=stage_epochs[i]) for i in range(len(stage_epochs))]

    traj = []
    d_seg0 = trainer.exact_d_seg(use_ema=False)
    traj.append({"stage": "init", "epoch": 0, "d_seg_live": d_seg0,
                 "d_pose_live": trainer.mean_d_pose(use_ema=False)})
    print(f"[{schedule}] init d_seg(live)={d_seg0:.5f}", flush=True)

    for i, spec in enumerate(stages):
        trainer.configure_stage(spec, optimizer_schedule=schedule)
        t0 = time.time()
        summary = trainer.run_stage_epochs(spec)
        d_seg_live = trainer.exact_d_seg(use_ema=False)
        d_seg_ema = trainer.exact_d_seg(use_ema=True)
        d_pose_live = trainer.mean_d_pose(use_ema=False)
        row = {
            "stage": spec.name, "stage_idx": i, "epochs": spec.epochs,
            "use_muon": trainer.opt_config.use_muon,
            "muon_lr": trainer.opt_config.muon_lr,
            "adamw_lr": trainer.opt_config.adamw_lr,
            "grad_clip": trainer.opt_config.grad_clip,
            "d_seg_live": d_seg_live, "d_seg_ema": d_seg_ema,
            "d_pose_live": d_pose_live,
            "d_seg_best_in_stage": summary.get("d_seg_best"),
            "wall_s": round(time.time() - t0, 1),
        }
        traj.append(row)
        print(f"[{schedule}] {spec.name} use_muon={row['use_muon']} "
              f"adamw_lr={row['adamw_lr']:.1e} muon_lr={row['muon_lr']:.1e} "
              f"-> d_seg(live)={d_seg_live:.5f} d_seg(ema)={d_seg_ema:.5f} "
              f"({row['wall_s']}s)", flush=True)
    return traj


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--base-ch", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage-epochs", type=int, nargs="+", default=[12, 12])
    ap.add_argument("--out", default="experiments/results/diag_faithful_vs_muon_ab")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for schedule in ["pr95_adamw_then_muon", "muon_throughout"]:
        results[schedule] = run_arm(
            schedule, n_pairs=args.n_pairs, base_ch=args.base_ch, seed=args.seed,
            stage_epochs=args.stage_epochs,
        )
        (out / f"{schedule}_trajectory.json").write_text(
            json.dumps(results[schedule], indent=1)
        )

    f_final = results["pr95_adamw_then_muon"][-1]["d_seg_live"]
    m_final = results["muon_throughout"][-1]["d_seg_live"]
    verdict = {
        "axis": "[macOS-CPU advisory]", "score_claim": False, "promotable": False,
        "n_pairs": args.n_pairs, "base_ch": args.base_ch,
        "stage_epochs": args.stage_epochs,
        "faithful_final_d_seg_live": f_final,
        "muon_throughout_final_d_seg_live": m_final,
        "faithful_descended": f_final < results["pr95_adamw_then_muon"][0]["d_seg_live"] - 1e-3,
        "muon_throughout_beats_faithful": m_final < f_final,
        "ratio_faithful_over_muon": (f_final / m_final) if m_final > 0 else None,
        "note": (
            "If faithful descends comparably (ratio within ~2x), faithful is the "
            "safe default (it ALSO holds the 5.6e-4 basin existence-proof at "
            "base_ch=36). Only a DECISIVE muon_throughout win (ratio >> 1) would "
            "justify diverging from the proven recipe for the $100 n600 spend."
        ),
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print("\n=== RECIPE-CHOICE VERDICT ===", flush=True)
    print(json.dumps(verdict, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
