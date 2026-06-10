# SPDX-License-Identifier: MIT
"""The 1:1 MLX score-aware trainer — PR95-faithful, parity-gated, C1-C9-fixed.

This is the clean MLX port of PR95's ``stages/common.py`` training loop. It wires
the THREE verified MLX kernels (the bit-exact ``HNeRVDecoderMLX``, the NS-Muon
``zeropower_via_newtonschulz5_mlx``, the AdamW/Muon ``apply_pr95_mlx_optimizer_step``)
through the torch-frozen-scorer <-> ``mx.vjp`` bridge so the MLX decoder's LIVE
render d_seg descends against the real contest scorer — without the broken
shared harness's learnable-head surrogate or its C1-C9 config defects.

The C1-C9 fixes (per ``.omx/research/full_stack_audit_*.md``), made the DEFAULT:
  - C1: M-arch ON  — the decoder is ``HNeRVDecoderMLX`` (bilinear-skip + refine).
  - C4/C6: scorer-weighted, NOT recon-1.0 — the ONLY objective is ``100*seg + 1*pose``
           through the live scorer; there is no recon-MSE term at all.
  - C7: Muon-throughout — ``use_muon=True`` from epoch 0 (NOT AdamW stages 1-7 /
           Muon-stage-8-only), per ``.omx/research/tilde_optimizers_*.md`` (#77):
           the inert loop's grad pathology was AdamW + 100%-clip in stages 1-7.
  - EMA: decay is a config knob (default 0.999 = PR95) with a short-run guard so
         the LIVE render is what descends, not just the lagging EMA shadow
         (the 0.999-lag landmine — for short runs use a smaller decay or evaluate
         the live weights).

The decisive observable is :meth:`MlxScoreAwareTrainer.exact_d_seg` — the EXACT
SegNet argmax-disagreement on the LIVE MLX render. A working loop drives it
below the ~0.50 mean-field wall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from tac.local_acceleration.pr95_hnerv_mlx import (
    HNeRVSyntheticTrainingBundleMLX,
    Pr95MlxOptimizerConfig,
    Pr95MlxOptimizerState,
    apply_pr95_mlx_optimizer_step,
    build_parameter_group_lr_policy_fingerprint,
    pr95_mlx_parameter_shape_records,
)

if TYPE_CHECKING:  # the bridge is passed in by the caller (runtime-injected).
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

try:  # pragma: no cover - import guard
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
except Exception:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    tree_flatten = tree_unflatten = None  # type: ignore[assignment]


def _require_mlx() -> None:
    if mx is None:  # pragma: no cover
        raise RuntimeError("tac.mlx_pr95_port.mlx_trainer requires mlx.core.")


@dataclass
class MlxScoreAwareConfig:
    """Config for the 1:1 MLX score-aware loop (PR95-faithful defaults, C1-C9 fixed)."""

    epochs: int = 60
    batch_size: int = 8
    seg_loss_form: str = "ce_seg_loss"
    seg_weight: float = 100.0
    pose_weight: float = 1.0
    pose_enabled: bool = True
    eval_roundtrip: bool = True
    eval_every: int = 10
    seed: int = 0
    # C7 fix: Muon-throughout (NOT stage-8-only). use_muon=True from epoch 0.
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
    # EMA: PR95 default 0.999. For short runs the shadow lags; eval the LIVE
    # weights (use_ema_for_eval=False) to see the true descent, per the
    # 0.999-lag landmine in the audit.
    ema_decay: float = 0.999
    use_ema_for_eval: bool = False
    scorer_hw: tuple[int, int] = (384, 512)
    telemetry: list[dict[str, Any]] = field(default_factory=list)


class _MlxEMA:
    """Param-wise EMA shadow over an MLX module's flat parameter tree."""

    def __init__(self, module: Any, decay: float) -> None:
        _require_mlx()
        self.decay = float(decay)
        self.shadow = {
            k: mx.array(v) for k, v in tree_flatten(module.parameters())
        }

    def update(self, module: Any) -> None:
        d = self.decay
        for k, v in tree_flatten(module.parameters()):
            if k in self.shadow:
                self.shadow[k] = self.shadow[k] * d + v * (1.0 - d)
            else:
                self.shadow[k] = mx.array(v)
        mx.eval(list(self.shadow.values()))

    def snapshot_live(self, module: Any) -> dict[str, Any]:
        return {k: mx.array(v) for k, v in tree_flatten(module.parameters())}

    def apply_to(self, module: Any) -> dict[str, Any]:
        orig = self.snapshot_live(module)
        module.update(tree_unflatten(list(self.shadow.items())))
        return orig

    @staticmethod
    def restore(module: Any, orig: dict[str, Any]) -> None:
        module.update(tree_unflatten(list(orig.items())))


class MlxScoreAwareTrainer:
    """1:1 MLX port of the PR95 score-aware loop over the bit-exact decoder.

    Args:
        bundle: a :class:`HNeRVSyntheticTrainingBundleMLX` (decoder + per-pair
            latents) — the verified bit-exact MLX decoder.
        bridge: a :class:`TorchScorerBridge` (frozen torch scorer + GT targets).
        config: the loop config (PR95-faithful, C1-C9 fixed defaults).
    """

    def __init__(
        self,
        bundle: HNeRVSyntheticTrainingBundleMLX,
        bridge: TorchScorerBridge,
        config: MlxScoreAwareConfig,
    ) -> None:
        _require_mlx()
        self.bundle = bundle
        self.bridge = bridge
        self.cfg = config
        self.n_pairs = int(bundle.latents.shape[0])
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
            pr95_mlx_parameter_shape_records(bundle.parameters())
        )

    def _render(self, indices: Any) -> Any:
        """Render the pair batch (N2CHW) from the bundle for ``indices``."""
        return self.bundle(indices)

    def _vjp_grads(self, indices: Any, pixel_cotangent: Any) -> dict[str, Any]:
        """Backprop the pixel cotangent through the bundle via ``mx.vjp``.

        ``mx.vjp`` consumes a FLAT list of primal arrays + cotangents, so we
        flatten the bundle parameter tree to ``(name, array)`` pairs, run the
        vjp over the array list (re-installing them into the bundle inside the
        traced forward), and re-key the resulting gradient list back to the
        parameter names so :func:`apply_pr95_mlx_optimizer_step` can consume it.

        Returns the gradient tree (nested dict, unflattened) for the bundle.
        """
        flat = tree_flatten(self.bundle.trainable_parameters())
        names = [k for k, _ in flat]
        primals = [v for _, v in flat]

        def forward(*param_arrays: Any) -> Any:
            self.bundle.update(tree_unflatten(list(zip(names, param_arrays, strict=True))))
            return self.bundle(indices)

        _, vjps = mx.vjp(forward, list(primals), [pixel_cotangent])
        grad_pairs = list(zip(names, vjps, strict=True))
        return tree_unflatten(grad_pairs)

    def step(self, idx_np: np.ndarray) -> dict[str, Any]:
        """One training step on the given pair indices. Returns telemetry."""
        _require_mlx()
        indices = mx.array(idx_np.astype(np.int32))
        idx_t = torch.from_numpy(idx_np.astype(np.int64))
        # Forward render (MLX) -> torch scorer loss + pixel cotangent.
        render = self._render(indices)
        mx.eval(render)
        result = self.bridge.loss_and_pixel_grad(render, idx_t)
        # Backprop the cotangent through the bundle (MLX vjp).
        grads = self._vjp_grads(indices, result.pixel_cotangent)
        # PR95 optimizer step (Muon hidden conv weights + AdamW stem/rgb/latents).
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
            "grad_clip_applied": summary.get(
                "gradient_clip_actual_application_count", 0
            ),
        }

    def exact_d_seg(self, *, use_ema: bool | None = None) -> float:
        """Mean EXACT live-render d_seg over all pairs (the decisive observable)."""
        _require_mlx()
        ema = getattr(self, "_ema", None)
        do_ema = self.cfg.use_ema_for_eval if use_ema is None else use_ema
        orig = None
        if do_ema and ema is not None:
            orig = ema.apply_to(self.bundle)
        try:
            total = 0.0
            n = 0
            for start in range(0, self.n_pairs, self.cfg.batch_size):
                idx_np = np.arange(start, min(start + self.cfg.batch_size, self.n_pairs))
                indices = mx.array(idx_np.astype(np.int32))
                render = self._render(indices)
                mx.eval(render)
                d = self.bridge.exact_d_seg(render, torch.from_numpy(idx_np.astype(np.int64)))
                total += d * len(idx_np)
                n += len(idx_np)
            return total / max(n, 1)
        finally:
            if orig is not None:
                _MlxEMA.restore(self.bundle, orig)

    def train(self) -> dict[str, Any]:
        """Run the loop. Returns the descent summary + trajectory."""
        _require_mlx()
        cfg = self.cfg
        np.random.seed(cfg.seed)
        mx.random.seed(cfg.seed)
        self._ema = _MlxEMA(self.bundle, cfg.ema_decay)

        d_seg_initial = self.exact_d_seg(use_ema=False)
        trajectory: list[dict[str, Any]] = []
        best_d_seg = float("inf")
        clip_would_steps = 0
        total_steps = 0

        for epoch in range(cfg.epochs):
            perm = np.random.permutation(self.n_pairs)
            ep_seg = 0.0
            ep_loss = 0.0
            nb = 0
            for start in range(0, self.n_pairs, cfg.batch_size):
                idx_np = perm[start : start + cfg.batch_size]
                row = self.step(idx_np)
                self._ema.update(self.bundle)
                ep_seg += row["seg"]
                ep_loss += row["loss"]
                if row["grad_clip_would_clip"] > 0:
                    clip_would_steps += 1
                total_steps += 1
                nb += 1

            if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
                d_seg = self.exact_d_seg()
                best_d_seg = min(best_d_seg, d_seg)
                trow = {
                    "epoch": epoch + 1,
                    "exact_d_seg": d_seg,
                    "loss_mean": ep_loss / max(nb, 1),
                    "seg_loss_mean": ep_seg / max(nb, 1),
                    "clip_would_fraction": clip_would_steps / max(total_steps, 1),
                }
                trajectory.append(trow)
                cfg.telemetry.append(trow)

        return {
            "d_seg_initial": d_seg_initial,
            "d_seg_final": trajectory[-1]["exact_d_seg"] if trajectory else None,
            "d_seg_best": best_d_seg,
            "descended": (
                trajectory[-1]["exact_d_seg"] < d_seg_initial - 1e-4
                if trajectory
                else False
            ),
            "clip_would_fraction_final": (
                trajectory[-1]["clip_would_fraction"] if trajectory else None
            ),
            "trajectory": trajectory,
        }


__all__ = [
    "MlxScoreAwareConfig",
    "MlxScoreAwareTrainer",
]
