"""DIAGNOSTIC: is the curriculum d_seg-frozen-at-0.505 an EMA-shadow-lag artifact?

Hypothesis (math): StageSpec.ema_decay=0.999 -> EMA time constant ~1000 steps.
At 8 pairs / batch_size 8 = 1 step/epoch (or 48/8=6), a short stage's shadow is
~init weights, so exact_d_seg (which uses the shadow) reads a frozen near-init
value EVEN THOUGH the live weights solved seg (CE -> ~0).

Decisive test: run stage-1 for N epochs, then compare
    exact_d_seg(use_ema=False)  [LIVE weights]
    exact_d_seg(use_ema=True)   [EMA shadow]
If LIVE << SHADOW, the freeze is an EMA-lag MEASUREMENT+EXPORT bug, not capacity.

$0 macOS-CPU. NOT a score claim (advisory). Real bridge, real video targets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_capstone_campaign import _load_or_build_targets  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    cache = Path("experiments/results/capstone_gt_targets_cache")
    net, seg_t, pose_t, n = _load_or_build_targets(8, cache, "cpu")

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

    bundle = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=n, base_channels=20, carrier="vq_index", seed=0)
    )
    bridge = TorchScorerBridge(
        net, seg_t, pose_t, seg_loss_form="ce_seg_loss",
        seg_weight=100.0, pose_weight=1.0, eval_roundtrip=True,
    )
    cfg = CapstoneTrainConfig(epochs=1, eval_every=5, seed=0)
    trainer = CapstoneTrainer(bundle, bridge, pose_t.float().cpu().numpy(), cfg)

    print(f"[init] n={n} d_seg(live)={trainer.exact_d_seg(use_ema=False):.6f} "
          f"d_seg(ema)={trainer.exact_d_seg(use_ema=True):.6f}", flush=True)

    # Configure curriculum stage 1 (CE) and run epochs by hand so we can probe
    # both the LIVE and EMA d_seg at the same trained state.
    stages = build_pr95_8stage_curriculum(total_epochs=400)
    stage1 = stages[0]
    print(f"[stage1] {stage1.name} epochs(canonical-scaled)={stage1.epochs} "
          f"ema_decay={getattr(stage1, 'ema_decay', cfg.ema_decay)} "
          f"seg_w={stage1.seg_weight} pose_w={stage1.pose_weight}", flush=True)
    trainer.configure_stage(stage1, optimizer_schedule="pr95_adamw_then_muon")

    steps_per_epoch = max(1, (n + cfg.batch_size - 1) // cfg.batch_size)
    n_probe_epochs = 25
    for ep in range(n_probe_epochs):
        trainer._current_epoch = ep
        lr_scale = trainer._lr_scale_for_epoch(ep)
        perm = np.random.permutation(n)
        seg_losses = []
        for start in range(0, n, cfg.batch_size):
            idx_np = perm[start:start + cfg.batch_size]
            row = trainer.step(idx_np, lr_scale=lr_scale)
            seg_losses.append(row["seg"])
            live_batch_dseg = row["d_seg_batch"]
        if (ep + 1) % 5 == 0:
            live = trainer.exact_d_seg(use_ema=False)
            ema = trainer.exact_d_seg(use_ema=True)
            n_updates = trainer._ema._num_updates if hasattr(
                trainer._ema, "_num_updates"
            ) else "n/a"
            decay = trainer._ema.decay
            shadow_init_wt = decay ** ((ep + 1) * steps_per_epoch)
            print(
                f"[ep {ep+1:3d}] seg_loss={np.mean(seg_losses):.5f} "
                f"live_batch_dseg={live_batch_dseg:.5f} || "
                f"d_seg(LIVE)={live:.6f}  d_seg(EMA)={ema:.6f}  "
                f"GAP={ema-live:+.6f}  ema_updates={n_updates} "
                f"decay={decay} shadow~init_wt={shadow_init_wt:.3f}",
                flush=True,
            )

    live = trainer.exact_d_seg(use_ema=False)
    ema = trainer.exact_d_seg(use_ema=True)
    print("\n=== VERDICT ===", flush=True)
    print(f"LIVE d_seg = {live:.6f}", flush=True)
    print(f"EMA  d_seg = {ema:.6f}", flush=True)
    if ema - live > 0.05 and live < ema - 0.05:
        print("CONFIRMED: EMA-shadow LAG freezes exact_d_seg near init while the "
              "LIVE weights descend. The 0.505 freeze is a MEASUREMENT+EXPORT bug "
              "(slow decay 0.999 on a short run), NOT a seg-capacity wall.",
              flush=True)
    else:
        print("NOT the EMA-lag bug (live and ema d_seg agree). Investigate elsewhere.",
              flush=True)


if __name__ == "__main__":
    main()
