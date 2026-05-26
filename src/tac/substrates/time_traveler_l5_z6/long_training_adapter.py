# SPDX-License-Identifier: MIT
"""Z6 long-training adapter — canonical proof-of-pattern for L2 cascade.

Per Path 3 canonical-substrate-development-cascade doctrine + L2-INFRA-BUILD
charter: this module is the canonical D=Z6 adapter satisfying
:class:`tac.training.long_training_canonical.SubstrateLongTrainingAdapter`
Protocol. The L2 trainer entry-point uses this adapter via the canonical
:func:`tac.training.long_training_canonical.run_long_training` helper,
making the substrate-specific trainer ~30 LOC instead of the L1 promotion's
~600 LOC.

Cascade reference per doctrine §"L2 LONG-TRAINING INFRASTRUCTURE":

- L1 promotion (``experiments/train_substrate_z6_predictive_coding_mlx.py``,
  ~600 LOC): direct MLX training loop + hand-rolled EMA + hand-rolled
  checkpoint emission + hand-rolled posterior anchor + hand-rolled
  archive build.
- L2 long-training (canonical pattern; THIS module + thin entry-point):
  substrate-specific Z6 adapter implements the Protocol + canonical
  ``run_long_training`` handles EMA / checkpoint / Provenance / posterior
  anchor / observability / OOM-safety / multi-arm dispatch.

The adapter's substrate-specific surface is:
- :class:`Z6LongTrainingAdapter`: wraps the Z6 MLX renderer + provides the
  Protocol methods (sample_batch / loss_fn / optimizer_step /
  export_state_dict / export_archive / score_aware_components).
- The Z6 trainer reuses canonical primitives in
  ``tac.training.long_training_canonical`` for everything else.

Sister of:
- ``tac.substrates.time_traveler_l5_z6.architecture`` (Z6 PredictiveCoding
  config + reference torch model for sister/CUDA dispatch)
- ``tac.substrates.time_traveler_l5_z6.mlx_renderer`` (MLX-native renderer
  used in L1 promotion at commit 8833b9db5)
- ``tac.substrates.time_traveler_l5_z6.mlx_export_bridge`` (PyTorch state
  dict + Z6PCWM1 archive export bridge per Catalog #1251)

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #127/#192/
#317/#341: every L2 long-training output is non-promotable by construction;
canonical Provenance + posterior anchor markers are auto-stamped by the
canonical helper.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class Z6LongTrainingAdapter:
    """Canonical Z6 substrate adapter for tac.training.long_training_canonical.

    Wraps the MLX-native Z6 renderer + per-pair video targets + canonical
    MSE proxy loss into the Protocol contract. The adapter is the
    substrate-specific axis; everything substrate-agnostic (EMA /
    checkpoint / Provenance / posterior anchor) is the canonical helper.

    Args:
        config: Z6PredictiveCodingConfig instance (per
            tac.substrates.time_traveler_l5_z6.architecture).
        target_rgb_0: MLX array (num_pairs, H, W, 3) float32 in [0, 1].
        target_rgb_1: MLX array (num_pairs, H, W, 3) float32 in [0, 1].
        lambda_residual: Rao-Ballard residual L2 Lagrangian weight.

    Raises:
        ImportError: MLX is not installed (Z6 is MLX-only by design).
    """

    substrate_id: str = "time_traveler_l5_z6"

    def __init__(
        self,
        config: Any,
        target_rgb_0: Any,
        target_rgb_1: Any,
        lambda_residual: float = 1.0,
    ):
        try:
            import mlx.core as mx
            import mlx.nn as mlx_nn
            import mlx.optimizers as mlx_optim
        except ImportError as exc:
            raise ImportError(
                "Z6 long-training adapter requires MLX (Apple Silicon only). "
                "Install via `pip install mlx`. The PyTorch sister trainer at "
                "experiments/train_substrate_time_traveler_l5_z6.py covers the "
                "CUDA / paid-dispatch path."
            ) from exc

        from tac.substrates.time_traveler_l5_z6.mlx_renderer import (
            Z6PredictiveCodingMLXRenderer,
        )

        self._mx = mx
        self._mlx_nn = mlx_nn
        self._mlx_optim = mlx_optim
        self.config = config
        self.target_rgb_0 = target_rgb_0
        self.target_rgb_1 = target_rgb_1
        self.lambda_residual = float(lambda_residual)
        self.model = Z6PredictiveCodingMLXRenderer(config)
        # Cache the canonical optimizer; canonical helper invokes
        # optimizer_step() per step (we lazy-create on first step to
        # honor the canonical helper's learning_rate parameter).
        self._optimizer: Any = None
        self._optimizer_lr: float | None = None
        self._loss_grad: Any = None

    def sample_batch(self, batch_size: int, seed: int) -> Any:
        """Sample a batch of pair_indices for one training step.

        For Z6, the "batch" is the index array; targets are accessed via
        the pre-loaded ``target_rgb_0`` / ``target_rgb_1`` MLX buffers.
        Per the canonical contract, sample_batch returns whatever loss_fn
        consumes (here: an MLX int32 indices array).
        """
        mx = self._mx
        num_pairs = self.config.num_pairs
        # Deterministic sampling per seed for reproducibility per Catalog #229.
        import numpy as np

        rng = np.random.RandomState(seed)
        size = min(batch_size, num_pairs)
        sampled_np = rng.choice(num_pairs, size=size, replace=False)
        return mx.array(sampled_np.astype("int32"))

    def loss_fn(
        self,
        model: Any,
        batch: Any,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Diagnostic-only loss_fn (Style A fallback; not used by Style B path).

        The canonical helper prefers ``train_step`` (Style B) when present
        because Z6 uses MLX ``value_and_grad`` which requires combined
        value+grad+update. This method exists as a diagnostic fallback for
        sister tooling that wants pure loss without the gradient/update.
        """
        mx = self._mx
        rgb_0, rgb_1, _z = model.reconstruct_pair(batch)
        target_0_batch = self.target_rgb_0[batch]
        target_1_batch = self.target_rgb_1[batch]
        mse_0 = mx.mean((rgb_0 - target_0_batch) ** 2)
        mse_1 = mx.mean((rgb_1 - target_1_batch) ** 2)
        recon_loss = mse_0 + mse_1
        residual_l2 = mx.mean(model.residuals ** 2)
        recon_w = float(loss_weights.get("recon", 1.0))
        resid_w = float(loss_weights.get("residual", self.lambda_residual))
        total = recon_w * recon_loss + resid_w * residual_l2
        mx.eval(total)
        return {
            "total": float(total.item()),
            "recon": float(recon_loss.item()),
            "residual": float(residual_l2.item()),
        }

    def optimizer_step(self, model: Any, loss: Any, learning_rate: float) -> None:
        """Style A optimizer_step (raises; Z6 uses Style B train_step path).

        Per CLAUDE.md "Comment-only contracts are FORBIDDEN" + the canonical
        Style A/B Protocol contract: Z6 adapters MUST use train_step. This
        method exists for Protocol conformance; canonical helper detects
        train_step + bypasses this method.
        """
        raise NotImplementedError(
            "Z6 adapter uses Style B (train_step combined value+grad+update). "
            "The canonical helper prefers train_step when present; this "
            "optimizer_step is a Protocol-conformance stub only. "
            "Call train_step(batch, learning_rate, loss_weights) instead."
        )

    def train_step(
        self,
        batch: Any,
        learning_rate: float,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Style B combined value+grad+update for MLX value_and_grad pattern.

        This is the canonical MLX training step:
        1. Build ``loss_fn_inner(model)`` closure capturing batch + targets.
        2. Use ``mlx.nn.value_and_grad`` to compute (loss, grads).
        3. ``optimizer.update(model, grads)``.
        4. ``mx.eval(model.parameters(), optimizer.state)``.

        Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog
        #341 non-promotable markers: the canonical helper auto-stamps
        ``score_claim=False`` on the resulting TrainingArtifact.
        """
        mx = self._mx
        mlx_nn = self._mlx_nn
        mlx_optim = self._mlx_optim

        # Lazy-create or recreate optimizer if learning_rate changed.
        if self._optimizer is None or self._optimizer_lr != learning_rate:
            self._optimizer = mlx_optim.AdamW(learning_rate=learning_rate)
            self._optimizer_lr = learning_rate

        target_0_batch = self.target_rgb_0[batch]
        target_1_batch = self.target_rgb_1[batch]
        recon_w = float(loss_weights.get("recon", 1.0))
        resid_w = float(loss_weights.get("residual", self.lambda_residual))

        def _loss_fn_inner(model):
            rgb_0, rgb_1, _z = model.reconstruct_pair(batch)
            mse_0 = mx.mean((rgb_0 - target_0_batch) ** 2)
            mse_1 = mx.mean((rgb_1 - target_1_batch) ** 2)
            recon_loss = mse_0 + mse_1
            residual_l2 = mx.mean(model.residuals ** 2)
            total = recon_w * recon_loss + resid_w * residual_l2
            return total

        loss_and_grad_fn = mlx_nn.value_and_grad(self.model, _loss_fn_inner)
        loss_value, grads = loss_and_grad_fn(self.model)
        self._optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self._optimizer.state)
        return {
            "total": float(loss_value.item()),
        }

    def export_state_dict(self, model: Any, path: Path) -> None:
        """Export Z6 MLX state to .pt via canonical Catalog #1251 bridge."""
        from tac.substrates.time_traveler_l5_z6.mlx_export_bridge import (
            build_z6_pytorch_pt_from_mlx_renderer,
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        build_z6_pytorch_pt_from_mlx_renderer(model, path, overwrite=True)

    def export_archive(
        self,
        model: Any,
        output_dir: Path,
    ) -> tuple[Path, str, int] | None:
        """Export Z6PCWM1 archive via canonical mlx_export_bridge helper."""
        from tac.substrates.time_traveler_l5_z6.mlx_export_bridge import (
            build_z6pcwm1_archive_from_mlx_renderer,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        archive_path = output_dir / "0.bin"
        manifest = build_z6pcwm1_archive_from_mlx_renderer(
            model, archive_path, overwrite=True,
            lambda_residual_entropy=self.lambda_residual,
        )
        if not archive_path.is_file():
            return None
        archive_bytes = archive_path.read_bytes()
        import hashlib

        sha = hashlib.sha256(archive_bytes).hexdigest()
        return archive_path, sha, len(archive_bytes)

    def score_aware_components(
        self,
        model: Any,
        batch: Any,
    ) -> Mapping[str, float] | None:
        """Z6 L2 currently DEFERS per-axis SegNet/PoseNet decomposition.

        Per per-substrate symposium PROCEED_WITH_REVISIONS verdict + Yousfi
        dissent (`.omx/research/path_3_d_z6_per_substrate_symposium_l1_promotion_20260526.md`):
        true contest-grade score-aware Lagrangian routes through SegNet/PoseNet
        in PyTorch sister L2 promotion path (per Catalog #164 + #226). The
        L2 MLX trainer is reconstruction-proxy only; per-axis decomposition
        is the responsibility of the L3+ sister cascade.

        Returns:
            None (signaling caller that per-axis is N/A at L2 MLX).
        """
        return None


__all__ = ["Z6LongTrainingAdapter"]
