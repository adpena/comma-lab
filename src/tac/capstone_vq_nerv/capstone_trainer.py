# SPDX-License-Identifier: MIT
"""The capstone score-aware trainer (Task #78) — VQ-NeRV + FiLM-pose, #82 loop.

This wraps the verified #82 score-aware mechanism (MLX decoder render -> torch
frozen scorer -> pixel cotangent -> ``mx.vjp`` -> Muon/AdamW) for the ORIGINAL
VQ-NeRV + FiLM-pose bundle. The two additions over the #82 ``MlxScoreAwareTrainer``:

1. **FiLM-pose forward**: the bundle's forward takes ``(indices, pose)``; the
   trainer threads the STORED GT pose (the same scalars the archive stores) into
   both the render and the ``mx.vjp`` traced forward, so the gradient flows
   through the FiLM MLP. The pose half of the score-aware objective re-anchors
   the FiLM to render pose-correct frames.

2. **VQ EMA + commitment**: after each gradient step, the VQ codebook is
   EMA-updated from the most-recent forward (van den Oord §3.2). The commitment
   loss is added to the score-aware Lagrangian (it pulls ``z_e`` toward the
   nearest code so the straight-through gradient is well-conditioned).

The decisive observables are BOTH halves of the joint objective on the EXACT
scorer: ``exact_d_seg`` (the SegNet argmax-disagreement, must DESCEND toward
~5.6e-4) AND ``mean_d_pose`` (the PoseNet MSE, must stay LOW / HOLD because the
FiLM injects the stored pose). A working capstone moves both at the target bytes.

NO MPS; the scorer is frozen torch-CPU (the exact authority); GT only via
``frame_utils.yuv420_to_rgb`` (provided through the bridge's targets).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from tac.local_acceleration.pr95_hnerv_mlx import (
    Pr95MlxOptimizerConfig,
    Pr95MlxOptimizerState,
    apply_pr95_mlx_optimizer_step,
    build_parameter_group_lr_policy_fingerprint,
    pr95_mlx_parameter_shape_records,
)

if TYPE_CHECKING:
    from tac.capstone_vq_nerv.vq_nerv_bundle import CapstoneVqNervBundle
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

try:  # pragma: no cover - import guard
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
except Exception:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    tree_flatten = tree_unflatten = None  # type: ignore[assignment]


def _require_mlx() -> None:
    if mx is None:  # pragma: no cover
        raise RuntimeError("tac.capstone_vq_nerv.capstone_trainer requires mlx.")


@dataclass
class CapstoneTrainConfig:
    """Config for the capstone joint-descent loop (PR95-faithful, C1-C9 fixed)."""

    epochs: int = 60
    batch_size: int = 8
    seg_loss_form: str = "ce_seg_loss"
    seg_weight: float = 100.0
    pose_weight: float = 1.0
    commitment_weight: float = 0.25
    eval_every: int = 10
    seed: int = 0
    # C7: Muon-throughout from epoch 0.
    use_muon: bool = True
    adamw_lr: float = 3.0e-5
    muon_lr: float = 2.0e-4
    latent_lr_mult: float = 10.0
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    grad_clip_muon: float | None = 1.0
    cast_muon_float32_to_bfloat16: bool = True
    ema_decay: float = 0.999
    use_ema_for_eval: bool = False  # eval LIVE weights (the 0.999-lag landmine).
    telemetry: list[dict[str, Any]] = field(default_factory=list)


class CapstoneTrainer:
    """Joint score-aware trainer for the VQ-NeRV + FiLM-pose capstone bundle.

    Args:
        bundle: a :class:`CapstoneVqNervBundle`.
        bridge: a :class:`TorchScorerBridge` (frozen scorer + GT seg/pose targets).
        pose_store: ``(num_pairs, 6)`` float32 stored GT pose (the FiLM carrier —
            this is the SAME tensor the archive stores; the FiLM reads it, the
            export bytes it).
        config: the loop config.
    """

    def __init__(
        self,
        bundle: CapstoneVqNervBundle,
        bridge: TorchScorerBridge,
        pose_store: np.ndarray,
        config: CapstoneTrainConfig,
    ) -> None:
        _require_mlx()
        self.bundle = bundle
        self.bridge = bridge
        self.cfg = config
        self.n_pairs = int(bundle.latents.shape[0])
        self.pose_store = np.asarray(pose_store, dtype=np.float32)
        if self.pose_store.shape[0] != self.n_pairs:
            raise ValueError(
                f"pose_store has {self.pose_store.shape[0]} rows but bundle has "
                f"{self.n_pairs} pairs"
            )
        # Set the FiLM standardization from the stored pose.
        bundle.set_pose_stats(self.pose_store.mean(0), self.pose_store.std(0))
        self.opt_config = Pr95MlxOptimizerConfig(
            use_muon=config.use_muon,
            adamw_lr=config.adamw_lr,
            latent_lr_mult=config.latent_lr_mult,
            muon_lr=config.muon_lr,
            muon_momentum=config.muon_momentum,
            muon_nesterov=config.muon_nesterov,
            muon_ns_steps=config.muon_ns_steps,
            muon_weight_decay=config.muon_weight_decay,
            grad_clip=config.grad_clip,
            grad_clip_muon=config.grad_clip_muon,
            cast_muon_float32_to_bfloat16=config.cast_muon_float32_to_bfloat16,
        )
        self.opt_state = Pr95MlxOptimizerState()
        self._fingerprint = build_parameter_group_lr_policy_fingerprint(
            pr95_mlx_parameter_shape_records(bundle.trainable_parameters())
        )

    def _pose_mx(self, idx_np: np.ndarray) -> Any:
        return mx.array(self.pose_store[idx_np])

    def _render(self, indices: Any, pose: Any) -> Any:
        return self.bundle(indices, pose=pose)

    def _vjp_grads(
        self, indices: Any, pose: Any, pixel_cotangent: Any, commit_cotangent: float
    ) -> dict[str, Any]:
        """Backprop pixel cotangent + commitment gradient through the bundle.

        The traced forward renders the pair (which sets ``last_commitment_loss``)
        and returns ``(render, commit)``; ``mx.vjp`` consumes both with the pixel
        cotangent on the render and ``commit_cotangent`` (= commitment_weight) on
        the scalar commitment loss, so the gradient updates the carrier for BOTH
        the score-aware pixel objective AND the VQ commitment.
        """
        flat = tree_flatten(self.bundle.trainable_parameters())
        names = [k for k, _ in flat]
        primals = [v for _, v in flat]

        def forward(*param_arrays: Any) -> list[Any]:
            self.bundle.update(
                tree_unflatten(list(zip(names, param_arrays, strict=True)))
            )
            render = self.bundle(indices, pose=pose)
            commit = self.bundle.last_commitment_loss
            return [render, commit]

        cotangents = [pixel_cotangent, mx.array(float(commit_cotangent))]
        _, vjps = mx.vjp(forward, list(primals), cotangents)
        return tree_unflatten(list(zip(names, vjps, strict=True)))

    def step(self, idx_np: np.ndarray) -> dict[str, Any]:
        """One joint training step. Returns telemetry."""
        _require_mlx()
        indices = mx.array(idx_np.astype(np.int32))
        pose = self._pose_mx(idx_np)
        idx_t = torch.from_numpy(idx_np.astype(np.int64))
        render = self._render(indices, pose)
        mx.eval(render, self.bundle.last_commitment_loss)
        result = self.bridge.loss_and_pixel_grad(render, idx_t)
        grads = self._vjp_grads(
            indices, pose, result.pixel_cotangent, self.cfg.commitment_weight
        )
        summary = apply_pr95_mlx_optimizer_step(
            self.bundle,
            grads,
            self.opt_state,
            self.opt_config,
            parameter_group_fingerprint=self._fingerprint,
        )
        # VQ EMA codebook update from the most-recent forward (van den Oord §3.2).
        self.bundle.ema_update_from_last()
        mx.eval(self.bundle.parameters())
        return {
            "loss": result.loss_value,
            "seg": result.seg_loss_value,
            "pose": result.pose_loss_value,
            "commit": float(self.bundle.last_commitment_loss),
            "d_seg_batch": result.d_seg,
            "grad_clip_would_clip": summary.get(
                "gradient_clip_would_clip_count", 0
            ),
        }

    def exact_d_seg(self) -> float:
        """Mean EXACT live-render d_seg over all pairs (pose-FiLM render)."""
        _require_mlx()
        total = 0.0
        n = 0
        for start in range(0, self.n_pairs, self.cfg.batch_size):
            idx_np = np.arange(
                start, min(start + self.cfg.batch_size, self.n_pairs)
            )
            indices = mx.array(idx_np.astype(np.int32))
            render = self._render(indices, self._pose_mx(idx_np))
            mx.eval(render)
            d = self.bridge.exact_d_seg(
                render, torch.from_numpy(idx_np.astype(np.int64))
            )
            total += d * len(idx_np)
            n += len(idx_np)
        return total / max(n, 1)

    def mean_d_pose(self) -> float:
        """Mean EXACT PoseNet MSE over all pairs (the pose half of the score).

        Re-measures the PoseNet output on the LIVE pose-FiLM render vs the GT
        pose targets the bridge holds — the same per-pair pose MSE the evaluator
        charges (``d_pose``). This is the decisive POSE observable.
        """
        _require_mlx()
        bridge = self.bridge
        if bridge.pose_targets is None:
            return 0.0
        total = 0.0
        n = 0
        for start in range(0, self.n_pairs, self.cfg.batch_size):
            idx_np = np.arange(
                start, min(start + self.cfg.batch_size, self.n_pairs)
            )
            indices = mx.array(idx_np.astype(np.int32))
            render = self._render(indices, self._pose_mx(idx_np))
            mx.eval(render)
            idx_t = torch.from_numpy(idx_np.astype(np.int64))
            d = _exact_d_pose(bridge, render, idx_t)
            total += d * len(idx_np)
            n += len(idx_np)
        return total / max(n, 1)

    def train(self) -> dict[str, Any]:
        """Run the joint loop. Returns the joint-descent summary + trajectory."""
        _require_mlx()
        cfg = self.cfg
        np.random.seed(cfg.seed)
        mx.random.seed(cfg.seed)

        d_seg_initial = self.exact_d_seg()
        d_pose_initial = self.mean_d_pose()
        trajectory: list[dict[str, Any]] = []
        best_d_seg = float("inf")
        clip_would_steps = 0
        total_steps = 0

        for epoch in range(cfg.epochs):
            perm = np.random.permutation(self.n_pairs)
            ep_seg = ep_loss = ep_commit = 0.0
            nb = 0
            for start in range(0, self.n_pairs, cfg.batch_size):
                idx_np = perm[start : start + cfg.batch_size]
                row = self.step(idx_np)
                ep_seg += row["seg"]
                ep_loss += row["loss"]
                ep_commit += row["commit"]
                if row["grad_clip_would_clip"] > 0:
                    clip_would_steps += 1
                total_steps += 1
                nb += 1

            if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
                d_seg = self.exact_d_seg()
                d_pose = self.mean_d_pose()
                best_d_seg = min(best_d_seg, d_seg)
                trow = {
                    "epoch": epoch + 1,
                    "exact_d_seg": d_seg,
                    "mean_d_pose": d_pose,
                    "loss_mean": ep_loss / max(nb, 1),
                    "seg_loss_mean": ep_seg / max(nb, 1),
                    "commit_mean": ep_commit / max(nb, 1),
                    "clip_would_fraction": clip_would_steps / max(total_steps, 1),
                }
                trajectory.append(trow)
                cfg.telemetry.append(trow)

        return {
            "d_seg_initial": d_seg_initial,
            "d_pose_initial": d_pose_initial,
            "d_seg_final": trajectory[-1]["exact_d_seg"] if trajectory else None,
            "d_pose_final": trajectory[-1]["mean_d_pose"] if trajectory else None,
            "d_seg_best": best_d_seg,
            "seg_descended": (
                trajectory[-1]["exact_d_seg"] < d_seg_initial - 1e-4
                if trajectory
                else False
            ),
            "pose_held": (
                trajectory[-1]["mean_d_pose"] <= d_pose_initial + 1e-3
                if trajectory
                else False
            ),
            "trajectory": trajectory,
        }


@torch.no_grad()
def _exact_d_pose(bridge: Any, render_n2chw: Any, idx: torch.Tensor) -> float:
    """Re-measure PoseNet MSE on the live render vs GT pose (the d_pose term)."""
    import torch.nn.functional as F

    np_render = np.asarray(render_n2chw, dtype=np.float32)
    leaf = torch.tensor(np_render, dtype=torch.float32)
    b = leaf.shape[0]
    flat = leaf.reshape(b * 2, 3, leaf.shape[-2], leaf.shape[-1])
    if (int(flat.shape[-2]), int(flat.shape[-1])) != bridge.scorer_hw:
        flat = F.interpolate(
            flat, size=bridge.scorer_hw, mode="bilinear", align_corners=False
        )
    flat = flat.clamp(0.0, 255.0)
    bchw = flat.reshape(b, 2, 3, bridge.scorer_hw[0], bridge.scorer_hw[1])
    bhwc = bchw.permute(0, 1, 3, 4, 2).contiguous()
    posenet_in, _ = bridge.dnet.preprocess_input(bhwc)
    pose_out = bridge.dnet.posenet(posenet_in)
    pose_pred = pose_out["pose"][:, :6]
    target = bridge.pose_targets[idx]
    return float(((pose_pred - target) ** 2).mean().item())


__all__ = [
    "CapstoneTrainConfig",
    "CapstoneTrainer",
]
