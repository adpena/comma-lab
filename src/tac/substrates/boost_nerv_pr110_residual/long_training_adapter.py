# SPDX-License-Identifier: MIT
"""E=BoostNeRV-against-PR110 long-training adapter — L1-PROMOTION-CASCADE shell.

Per Path 3 canonical-substrate-development-cascade doctrine (commit
`fb270e9b6`) + L2-INFRA-BUILD canonical helper (commit `f5e4784ef`) +
Tier1-T3-OP7-OP8 cascade amendment (commit `b96418424`) + L1-PROMOTION-
CASCADE-B-C-E-G-J charter 2026-05-26: this adapter is the canonical
Protocol-conformant skeleton mirroring D=Z6's reference template
(``src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py``)
so the substrate's L2 trainer entry-point becomes ~136 LOC matching
``experiments/train_substrate_z6_predictive_coding_mlx_l2.py``.

L0 SCAFFOLD POSTURE (Catalog #240 recipe-vs-trainer-state consistency)
===================================================================
Per ``__init__.py`` L0 SCAFFOLD posture: the substrate's
``ResidualHeadMLX`` exists at ``architecture.py`` (forward returns
residual prediction) but is NOT yet wrapped as an ``mlx.nn.Module``
with registered parameters per the ``mlx.nn.value_and_grad`` contract.
Additionally per Phase 3 design memo §"Stage 0 cache PR110 base
reconstructions": E's training pipeline requires a one-time PR110
inflate to cache base reconstructions; this Stage 0 dependency is NOT
yet wired (no ``cache_pr110_base.py`` helper exists in the substrate
package).

Per Catalog #229 PV empirical finding (charter premise verification):
the adapter `__init__` fails-fast with explicit L1-follow-up guidance
rather than silently constructing a partial model that would crash
inside the canonical helper's ``PolyakEMAShadow``.

L1 FOLLOW-UP CONTRACT (drop-in replacement)
==========================================
The L1 follow-up subagent removes the ``__init__`` ``NotImplementedError``
+ implements:

1. Wrap ``ResidualHeadMLX`` as an ``mlx.nn.Module`` subclass with
   trainable ``z_proj`` / ``conv1`` / ``conv2`` parameters per
   ``architecture.py`` lines 88-144 (canonical MLX module pattern).
2. Implement Stage 0 PR110 base reconstruction caching: subprocess
   inflate of PR110's archive at
   ``submissions/pr110_<sha>/inflate.sh`` -> cached frames as MLX arrays.
3. Implement ``train_step`` Style B per Z6 reference template:
   ``mlx.nn.value_and_grad`` over a closure that computes:
   - Stage 2 (warm-up): L2(residual_pred, residual_target) where
     ``residual_target = GT - PR110_base_reconstruction``.
   - Stage 3 (score-aware fine-tune): canonical Catalog #164
     ``score_pair_components`` on composed frames =
     ``compose_pr110_base_plus_residual(PR110_base, residual_pred,
     boosting_gain_clamp)``.
4. Implement ``export_state_dict`` via canonical
   ``boosting_curriculum.BoostingCurriculum`` checkpoint.
5. Implement ``export_archive`` via canonical
   ``archive.compose_archive`` (BPR1 sidecar + PR110 base inline).
6. Implement ``inflate.py`` (currently MISSING per Catalog #229 PV).

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" +
Catalog #290 canonical-vs-unique decision per layer:

- ADOPT canonical Protocol contract (substrate-agnostic surface; LOC reduction)
- ADOPT canonical PolyakEMAShadow + TelemetrySink + CheckpointWriter via helper
- ADOPT canonical Provenance + posterior emission via helper
- UNIQUE E substrate-specific axes (L1 follow-up):
  * Frozen PR110 base learner with residual-only training (boosting paradigm)
  * Per-pair latent z_pr110 extracted from PR110 archive (NOT a fresh latent)
  * Tanh-bounded + boosting_gain_clamp residual on composed RGB
  * Iterative boosting rounds (1 at L0; sweep 1/2/3 at L1)
  * int8 residual blob quantization + brotli compression (rate budget ~8 KB)

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #127/
#192/#317/#341: every L2 long-training output is non-promotable by
construction; canonical Provenance + posterior anchor markers are
auto-stamped by the canonical helper.

[verified-against: src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py canonical reference template]
[verified-against: src/tac/training/long_training_canonical.py SubstrateLongTrainingAdapter Protocol]
[verified-against: src/tac/substrates/boost_nerv_pr110_residual/__init__.py L0 SCAFFOLD posture]
[verified-against: src/tac/substrates/boost_nerv_pr110_residual/architecture.py ResidualHeadMLX]
[verified-against: .omx/research/path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md]
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.framework_agnostic import require_mlx_core


class BoostNervPr110ResidualLongTrainingAdapter:
    """E=BoostNeRV canonical adapter shell — L1-PROMOTION-CASCADE structural skeleton.

    Args:
        config: ``BoostNervPr110ResidualConfig`` instance (per
            ``architecture.py``).
        target_rgb_0: MLX array (num_pairs, H, W, 3) float32 in [0, 1].
        target_rgb_1: MLX array (num_pairs, H, W, 3) float32 in [0, 1].
        pr110_base_path: Path to PR110 base archive (Stage 0 dependency;
            L1 follow-up wires the subprocess inflate to cache base
            reconstructions).
        residual_loss_weight: weight on per-pair L2 residual loss
            (warm-up stage). Default 1.0.

    Raises:
        ImportError: MLX is not installed (E is MLX-first per design memo).
        NotImplementedError: substrate is L0 SCAFFOLD; trainable
            ``mlx.nn.Module`` wrapper of ``ResidualHeadMLX`` + Stage 0
            PR110 base reconstruction caching land at L1 follow-up per
            CLAUDE.md "Substrate scaffolds MUST be COMPLETE or
            RESEARCH-ONLY" + Catalog #240.
    """

    substrate_id: str = "boost_nerv_pr110_residual"

    def __init__(
        self,
        config: Any,
        target_rgb_0: Any,
        target_rgb_1: Any,
        pr110_base_path: Path | None = None,
        residual_loss_weight: float = 1.0,
    ):
        require_mlx_core()

        self.config = config
        self.target_rgb_0 = target_rgb_0
        self.target_rgb_1 = target_rgb_1
        self.pr110_base_path = pr110_base_path
        self.residual_loss_weight = float(residual_loss_weight)
        self.model: Any = None

        raise NotImplementedError(
            "E=BoostNeRV-PR110-residual adapter __init__ is L0 SCAFFOLD "
            "structural shell per L1-PROMOTION-CASCADE charter 2026-05-26. "
            "The substrate's ResidualHeadMLX (architecture.py) exists for "
            "forward but is NOT yet wrapped as an mlx.nn.Module with "
            "registered z_proj/conv1/conv2 parameters per "
            "mlx.nn.value_and_grad contract. Additionally Stage 0 PR110 base "
            "reconstruction caching is NOT yet wired (no cache_pr110_base.py "
            "in substrate package). Sister observation: no inflate.py exists "
            "yet either (Catalog #146 contract violation pending). L1 "
            "follow-up subagent: (1) wraps ResidualHeadMLX as mlx.nn.Module; "
            "(2) wires Stage 0 PR110 base caching subprocess inflate; "
            "(3) lands inflate.py per Catalog #146; (4) implements Style B "
            "train_step per Z6 reference template at "
            "src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py. "
            "See module docstring 'L1 FOLLOW-UP CONTRACT' section for the "
            "drop-in replacement specification. Per Catalog #240 + CLAUDE.md "
            "'Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY' "
            "non-negotiable: failing-fast HERE is the correct L0 SCAFFOLD "
            "posture."
        )

    def sample_batch(self, batch_size: int, seed: int) -> Any:
        """Sample a batch of pair_indices for one training step."""
        import numpy as np

        mx = require_mlx_core()

        # L1 follow-up reads num_pairs from config; here hard-coded to
        # avoid premature import; the L1 trainer passes a real config.
        num_pairs = int(getattr(self.config, "num_pairs", 600))
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
        """Style A diagnostic-only stub (E adapter uses Style B train_step)."""
        raise NotImplementedError(
            "E=BoostNeRV-PR110 adapter uses Style B (train_step). See module "
            "docstring 'L1 FOLLOW-UP CONTRACT' #3."
        )

    def optimizer_step(self, model: Any, loss: Any, learning_rate: float) -> None:
        """Style A stub (E adapter uses Style B train_step)."""
        raise NotImplementedError(
            "E=BoostNeRV-PR110 adapter uses Style B (train_step). See module docstring."
        )

    def train_step(
        self,
        batch: Any,
        learning_rate: float,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Style B combined value+grad+update (L1 follow-up contract #3)."""
        raise NotImplementedError(
            "E=BoostNeRV-PR110 train_step pending L1 follow-up; see module "
            "docstring 'L1 FOLLOW-UP CONTRACT' #3 for the drop-in "
            "mlx.nn.value_and_grad over warm-up L2(residual_pred, "
            "residual_target=GT-PR110_base) + score-aware "
            "compose_pr110_base_plus_residual + canonical Catalog #164 "
            "score_pair_components composition."
        )

    def export_state_dict(self, model: Any, path: Path) -> None:
        """Export E trained state via canonical BoostingCurriculum checkpoint."""
        raise NotImplementedError(
            "E=BoostNeRV-PR110 export_state_dict pending L1 follow-up; "
            "canonical sister at tac.substrates.boost_nerv_pr110_residual"
            ".boosting_curriculum.BoostingCurriculum will handle checkpoint."
        )

    def export_archive(
        self,
        model: Any,
        output_dir: Path,
    ) -> tuple[Path, str, int] | None:
        """Export BPR1 archive via canonical compose_archive (archive.py)."""
        raise NotImplementedError(
            "E=BoostNeRV-PR110 export_archive pending L1 follow-up; canonical "
            "compose_archive at tac.substrates.boost_nerv_pr110_residual.archive "
            "composes BPR1 header + residual_blob (brotli-quality9 int8) + "
            "PR110 base archive bytes inline."
        )

    def score_aware_components(
        self,
        model: Any,
        batch: Any,
    ) -> Mapping[str, float] | None:
        """E L2 currently DEFERS per-axis SegNet/PoseNet decomposition.

        Per Z6 reference template precedent + Catalog #164 + #226: true
        contest-grade score-aware Lagrangian routes through SegNet/PoseNet
        in PyTorch sister L2 promotion path; the L2 MLX trainer is
        reconstruction-proxy + composed-RGB-proxy only.

        Returns:
            None (signaling caller that per-axis is N/A at L2 MLX).
        """
        return None


__all__ = ["BoostNervPr110ResidualLongTrainingAdapter"]
