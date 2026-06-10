# SPDX-License-Identifier: MIT
"""Stabilized pose-FiLM score-aware trainer + per-size recipe (task #84 D2/D3).

This is the #74-instability fix made a reusable module. #74 found the small
distilled student destabilized at 60kb (d_pose 0.0024 -> 1.44) on a *fixed-LR /
fixed-schedule* loop — the larger decoder diverged out-of-basin. The fix
(per #77 + the audit C7 finding) is:

1. **Muon-throughout** — the NS-Muon update magnitude is independent of the raw
   gradient norm (all singular values driven to ~1), so the scale-instability that
   blew up the fixed-LR AdamW loop is structurally removed. The pose-FiLM MLP
   weights are 2D -> Muon-eligible -> scale-stable.
2. **Per-size LR / grad-clip / EMA tuning** — :class:`StabilizedRecipe` returns a
   size-appropriate config so the larger bases stay in-basin (smaller LR + a
   real EMA decay at the size where #74 blew up).
3. **The LIVE render is the observable** — d_seg/d_pose are measured on the live
   MLX render (``use_ema_for_eval=False``), NOT the lagging EMA-0.999 shadow (the
   #82 landmine), so the true descent (and any blowup) is visible immediately.

The trainer drives the FULL ``100*seg + 1*pose`` objective through the verified
torch-frozen-scorer <-> ``mx.vjp`` bridge over the :class:`StoredPoseBundleMLX`
(pose-FiLM decoder + stored pose). The decisive observables:

- :meth:`PoseFilmTrainer.exact_d_pose` — the EXACT frozen-PoseNet d_pose on the
  LIVE render (the limiting term);
- the ``monotone`` flag in :meth:`PoseFilmTrainer.train` — True iff d_pose never
  blew up past the #74 instability threshold (the stability proof).

Authority: ``[macOS-MLX research-signal]`` / ``[local CPU-torch advisory]``;
non-promotable per Catalog #192.
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
    from tac.mlx_pr95_port.pose_film import StoredPoseBundleMLX
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

try:  # pragma: no cover - import guard
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
except Exception:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    tree_flatten = tree_unflatten = None  # type: ignore[assignment]


def _require_mlx() -> None:
    if mx is None:  # pragma: no cover
        raise RuntimeError("tac.mlx_pr95_port.pose_film_trainer requires mlx.core.")


# The #74 instability threshold: d_pose blew up 0.0024 -> 1.44 (600x) at 60kb on
# the unstable fixed-LR loop. A run is "monotone-stable" iff d_pose never exceeds
# this multiple of its running minimum after warmup (the blowup signature).
INSTABILITY_BLOWUP_FACTOR = 50.0


@dataclass(frozen=True)
class StabilizedRecipe:
    """Per-size LR / grad-clip / EMA recipe — the #74-instability fix.

    The larger the carrier (base_channels), the more #74-prone it is, so the
    recipe scales the LR DOWN and the EMA decay UP at the size where the fixed-LR
    loop blew up. Muon is ON at every size (the scale-stable core). The numbers
    are conservative-in-basin, not tuned-to-a-leaderboard.
    """

    muon_lr: float
    adamw_lr: float
    latent_lr_mult: float
    grad_clip: float
    ema_decay: float
    use_muon: bool = True

    @staticmethod
    def for_base_channels(base_channels: int) -> StabilizedRecipe:
        """Return the size-appropriate stabilized recipe.

        The break point is the #74 60kb blowup. ``base_channels`` is the carrier
        size knob (PR95 frontier is 36); smaller bases are more stable, larger
        bases need the smaller LR + stronger EMA to stay in-basin.
        """
        bc = int(base_channels)
        if bc <= 16:
            # smallest, most stable (the #74 40kb in-basin point) — can afford the
            # PR95-default LR.
            return StabilizedRecipe(
                muon_lr=2.0e-4, adamw_lr=3.0e-5, latent_lr_mult=10.0,
                grad_clip=1.0, ema_decay=0.99,
            )
        if bc <= 28:
            # the #74 60kb regime that blew up on fixed-LR — pull the LR down + a
            # real EMA so the bigger carrier stays in-basin.
            return StabilizedRecipe(
                muon_lr=1.0e-4, adamw_lr=2.0e-5, latent_lr_mult=8.0,
                grad_clip=1.0, ema_decay=0.997,
            )
        # frontier-class carrier (PR95 base 36) — the most conservative LR.
        return StabilizedRecipe(
            muon_lr=7.0e-5, adamw_lr=1.5e-5, latent_lr_mult=6.0,
            grad_clip=1.0, ema_decay=0.997,
        )


@dataclass
class PoseFilmTrainerConfig:
    """Config for the stabilized pose-FiLM score-aware loop."""

    epochs: int = 80
    batch_size: int = 8
    seg_loss_form: str = "ce_seg_loss"
    seg_weight: float = 100.0
    pose_weight: float = 1.0
    eval_roundtrip: bool = True
    eval_every: int = 10
    seed: int = 0
    base_channels: int = 28
    # Stability recipe (overrides the PR95-defaults below when set). When None the
    # recipe is auto-selected from base_channels via StabilizedRecipe.
    recipe: StabilizedRecipe | None = None
    # PR95-default fallbacks (only used when recipe is None AND auto-select off).
    use_muon: bool = True
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    cast_muon_float32_to_bfloat16: bool = True
    use_ema_for_eval: bool = False
    scorer_hw: tuple[int, int] = (384, 512)
    telemetry: list[dict[str, Any]] = field(default_factory=list)


class _MlxEMA:
    """Param-wise EMA shadow over an MLX module's flat parameter tree."""

    def __init__(self, module: Any, decay: float) -> None:
        _require_mlx()
        self.decay = float(decay)
        self.shadow = {k: mx.array(v) for k, v in tree_flatten(module.parameters())}

    def update(self, module: Any) -> None:
        d = self.decay
        for k, v in tree_flatten(module.parameters()):
            if k in self.shadow:
                self.shadow[k] = self.shadow[k] * d + v * (1.0 - d)
            else:
                self.shadow[k] = mx.array(v)
        mx.eval(list(self.shadow.values()))


class PoseFilmTrainer:
    """Stabilized pose-FiLM score-aware loop (Muon-throughout, per-size recipe).

    Args:
        bundle: a :class:`StoredPoseBundleMLX` (pose-FiLM decoder + latents + stored
            pose). ``bundle(indices)`` renders the pairs conditioning each frame head
            on the STORED pose.
        bridge: a :class:`TorchScorerBridge` (frozen torch scorer + GT targets); its
            ``pose_targets`` MUST be set so the pose term is live.
        config: the loop config (the recipe is auto-selected from base_channels).
    """

    def __init__(
        self,
        bundle: StoredPoseBundleMLX,
        bridge: TorchScorerBridge,
        config: PoseFilmTrainerConfig,
    ) -> None:
        _require_mlx()
        self.bundle = bundle
        self.bridge = bridge
        self.cfg = config
        self.n_pairs = int(bundle.latents.shape[0])
        recipe = config.recipe or StabilizedRecipe.for_base_channels(
            config.base_channels
        )
        self.recipe = recipe
        self.opt_config = Pr95MlxOptimizerConfig(
            use_muon=recipe.use_muon and config.use_muon,
            adamw_lr=recipe.adamw_lr,
            latent_lr_mult=recipe.latent_lr_mult,
            muon_lr=recipe.muon_lr,
            muon_momentum=config.muon_momentum,
            muon_nesterov=config.muon_nesterov,
            muon_ns_steps=config.muon_ns_steps,
            grad_clip=recipe.grad_clip,
            grad_clip_muon=recipe.grad_clip,
            cast_muon_float32_to_bfloat16=config.cast_muon_float32_to_bfloat16,
        )
        self.opt_state = Pr95MlxOptimizerState()
        self._fingerprint = build_parameter_group_lr_policy_fingerprint(
            pr95_mlx_parameter_shape_records(bundle.parameters())
        )

    def _vjp_grads(self, indices: Any, pixel_cotangent: Any) -> Any:
        flat = tree_flatten(self.bundle.trainable_parameters())
        names = [k for k, _ in flat]
        primals = [v for _, v in flat]

        def forward(*param_arrays: Any) -> Any:
            self.bundle.update(
                tree_unflatten(list(zip(names, param_arrays, strict=True)))
            )
            return self.bundle(indices)

        _, vjps = mx.vjp(forward, list(primals), [pixel_cotangent])
        return tree_unflatten(list(zip(names, vjps, strict=True)))

    def step(self, idx_np: np.ndarray) -> dict[str, Any]:
        """One stabilized training step. Returns telemetry."""
        _require_mlx()
        indices = mx.array(idx_np.astype(np.int32))
        idx_t = torch.from_numpy(idx_np.astype(np.int64))
        render = self.bundle(indices)
        mx.eval(render)
        result = self.bridge.loss_and_pixel_grad(render, idx_t)
        grads = self._vjp_grads(indices, result.pixel_cotangent)
        summary = apply_pr95_mlx_optimizer_step(
            self.bundle,
            grads,
            self.opt_state,
            self.opt_config,
            parameter_group_fingerprint=self._fingerprint,
        )
        mx.eval(self.bundle.parameters())
        return {
            "loss": result.loss_value,
            "seg": result.seg_loss_value,
            "pose": result.pose_loss_value,
            "d_seg_batch": result.d_seg,
            "grad_clip_would_clip": summary.get("gradient_clip_would_clip_count", 0),
        }

    def _eval_metric(self, which: str) -> float:
        """Mean EXACT live-render d_seg or d_pose over all pairs."""
        _require_mlx()
        total = 0.0
        n = 0
        for start in range(0, self.n_pairs, self.cfg.batch_size):
            idx_np = np.arange(
                start, min(start + self.cfg.batch_size, self.n_pairs)
            )
            indices = mx.array(idx_np.astype(np.int32))
            render = self.bundle(indices)
            mx.eval(render)
            idx_t = torch.from_numpy(idx_np.astype(np.int64))
            v = (
                self.bridge.exact_d_seg(render, idx_t)
                if which == "d_seg"
                else self.bridge.exact_d_pose(render, idx_t)
            )
            total += v * len(idx_np)
            n += len(idx_np)
        return total / max(n, 1)

    def exact_d_seg(self) -> float:
        """Mean EXACT live-render d_seg over all pairs."""
        return self._eval_metric("d_seg")

    def exact_d_pose(self) -> float:
        """Mean EXACT live-render d_pose over all pairs (the limiting term)."""
        return self._eval_metric("d_pose")

    def train(self) -> dict[str, Any]:
        """Run the stabilized loop. Returns descent + monotonicity summary."""
        _require_mlx()
        cfg = self.cfg
        np.random.seed(cfg.seed)
        mx.random.seed(cfg.seed)
        self._ema = _MlxEMA(self.bundle, self.recipe.ema_decay)

        d_pose_initial = self.exact_d_pose()
        d_seg_initial = self.exact_d_seg()
        trajectory: list[dict[str, Any]] = []
        best_d_pose = float("inf")
        running_min_d_pose = d_pose_initial
        monotone = True
        blowup_epoch = None

        for epoch in range(cfg.epochs):
            perm = np.random.permutation(self.n_pairs)
            for start in range(0, self.n_pairs, cfg.batch_size):
                idx_np = perm[start : start + cfg.batch_size]
                self.step(idx_np)
                self._ema.update(self.bundle)

            if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
                d_pose = self.exact_d_pose()
                d_seg = self.exact_d_seg()
                best_d_pose = min(best_d_pose, d_pose)
                # Blowup detection: after the first eval, d_pose must not exceed
                # INSTABILITY_BLOWUP_FACTOR * the running min (the #74 signature).
                if (
                    trajectory
                    and d_pose > INSTABILITY_BLOWUP_FACTOR * max(running_min_d_pose, 1e-8)
                ):
                    monotone = False
                    if blowup_epoch is None:
                        blowup_epoch = epoch + 1
                running_min_d_pose = min(running_min_d_pose, d_pose)
                trow = {
                    "epoch": epoch + 1,
                    "exact_d_pose": d_pose,
                    "exact_d_seg": d_seg,
                }
                trajectory.append(trow)
                cfg.telemetry.append(trow)

        return {
            "d_pose_initial": d_pose_initial,
            "d_pose_final": trajectory[-1]["exact_d_pose"] if trajectory else None,
            "d_pose_best": best_d_pose,
            "d_seg_initial": d_seg_initial,
            "d_seg_final": trajectory[-1]["exact_d_seg"] if trajectory else None,
            "pose_descended": (
                trajectory[-1]["exact_d_pose"] < d_pose_initial - 1e-6
                if trajectory
                else False
            ),
            # The #74-instability-fix proof: True iff d_pose never blew up.
            "monotone": monotone,
            "blowup_epoch": blowup_epoch,
            "recipe": {
                "muon_lr": self.recipe.muon_lr,
                "adamw_lr": self.recipe.adamw_lr,
                "grad_clip": self.recipe.grad_clip,
                "ema_decay": self.recipe.ema_decay,
                "use_muon": self.recipe.use_muon,
            },
            "trajectory": trajectory,
        }


__all__ = [
    "INSTABILITY_BLOWUP_FACTOR",
    "PoseFilmTrainer",
    "PoseFilmTrainerConfig",
    "StabilizedRecipe",
]
