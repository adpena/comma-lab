# SPDX-License-Identifier: MIT
"""G=NIRVANA-cascading-NeRV long-training adapter — L1-PROMOTION-CASCADE shell.

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
Per ``mlx_renderer.py`` module docstring line 2:
*"MLX hierarchical residual decoder cascade SCAFFOLD (config + helpers;
actual renderer class lands Phase 2)"*

The substrate ships ``NirvanaCascadingNervConfig`` + factory helpers +
parameter-count estimators, but the actual hierarchical-residual MLX
renderer class with ``__call__`` forward path is deferred to Phase 2
per Catalog #325 per-substrate symposium contract. The substrate's
``_full_main`` raises ``NotImplementedError`` per Catalog #240; this
adapter's ``__init__`` honors the same L0 SCAFFOLD posture.

Per Catalog #229 PV empirical finding (charter premise verification):
the renderer class is not yet implemented; the canonical L2 helper's
``PolyakEMAShadow`` ``state_dict()`` / ``parameters()`` contract cannot
be satisfied until the L1+Phase-2 follow-up subagent lands the
hierarchical-residual MLX renderer module.

L1 FOLLOW-UP CONTRACT (drop-in replacement)
==========================================
The L1+Phase-2 follow-up subagent removes the ``__init__``
``NotImplementedError`` + implements:

1. Land ``NirvanaCascadingNervRendererMLX(nn.Module)`` class in
   ``mlx_renderer.py`` per Phase 2 design memo's hierarchical-residual
   decoder cascade specification (coarse-scale base + N residual
   refinement levels with cascading composition).
2. Implement ``train_step`` Style B per Z6 reference template:
   ``mlx.nn.value_and_grad`` over a closure that computes per-level
   cascading residual loss summed across all decoder levels.
3. Implement ``export_state_dict`` via PyTorch sister bridge.
4. Implement ``export_archive`` via canonical
   ``archive.write_nirvana1_archive`` (NIRVANA1 grammar already
   landed; only state_dict serialization piece is pending).

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" +
Catalog #290 canonical-vs-unique decision per layer:

- ADOPT canonical Protocol contract (substrate-agnostic surface; LOC reduction)
- ADOPT canonical PolyakEMAShadow + TelemetrySink + CheckpointWriter via helper
- ADOPT canonical Provenance + posterior emission via helper
- UNIQUE G substrate-specific axes (L1+Phase-2 follow-up):
  * Hierarchical residual decoder cascade (Daubechies multi-scale wavelet
    discipline per CLAUDE.md "Council conduct" + Catalog #277)
  * Coarse-scale base + N residual refinement levels
  * NIRVANA1 byte-deterministic grammar (Catalog #146 contract; already landed)
  * PyTorch inflate runtime per Catalog #205 (already landed)

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #127/
#192/#317/#341: every L2 long-training output is non-promotable by
construction; canonical Provenance + posterior anchor markers are
auto-stamped by the canonical helper.

[verified-against: src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py canonical reference template]
[verified-against: src/tac/training/long_training_canonical.py SubstrateLongTrainingAdapter Protocol]
[verified-against: src/tac/substrates/nirvana_cascading_nerv/mlx_renderer.py L0 SCAFFOLD posture]
[verified-against: .omx/research/path_3_g_nirvana_cascading_nerv_substrate_design_20260526.md]
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class NirvanaCascadingNervLongTrainingAdapter:
    """G=NIRVANA canonical adapter shell — L1-PROMOTION-CASCADE structural skeleton.

    Args:
        config: ``NirvanaCascadingNervConfig`` instance (per
            ``mlx_renderer.py``).
        target_rgb_0: MLX array (num_pairs, H, W, 3) float32 in [0, 1].
        target_rgb_1: MLX array (num_pairs, H, W, 3) float32 in [0, 1].
        residual_loss_weight: weight on per-level residual loss
            (cascading composition). Default 1.0.

    Raises:
        ImportError: MLX is not installed (G is MLX-first by design).
        NotImplementedError: substrate is L0 SCAFFOLD; the
            ``NirvanaCascadingNervRendererMLX`` class lands at Phase 2
            per Catalog #325 per-substrate symposium contract.
    """

    substrate_id: str = "nirvana_cascading_nerv"

    def __init__(
        self,
        config: Any,
        target_rgb_0: Any,
        target_rgb_1: Any,
        residual_loss_weight: float = 1.0,
    ):
        try:
            import mlx.core as mx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "G=NIRVANA-cascading-NeRV long-training adapter requires MLX "
                "(Apple Silicon only). Install via `pip install mlx`. The "
                "numpy reference at tac.substrates.nirvana_cascading_nerv."
                "numpy_reference covers CPU-only inference."
            ) from exc

        self.config = config
        self.target_rgb_0 = target_rgb_0
        self.target_rgb_1 = target_rgb_1
        self.residual_loss_weight = float(residual_loss_weight)
        self.model: Any = None

        raise NotImplementedError(
            "G=NIRVANA adapter __init__ is L0 SCAFFOLD structural shell per "
            "L1-PROMOTION-CASCADE charter 2026-05-26. The substrate's "
            "mlx_renderer.py module docstring explicitly declares 'SCAFFOLD "
            "(config + helpers; actual renderer class lands Phase 2)'. The "
            "NirvanaCascadingNervRendererMLX class with __call__ forward "
            "path is NOT yet implemented; the canonical L2 helper's "
            "PolyakEMAShadow state_dict()/parameters() contract cannot be "
            "satisfied. L1+Phase-2 follow-up subagent: (1) lands "
            "NirvanaCascadingNervRendererMLX(nn.Module) class per Phase 2 "
            "design memo's hierarchical-residual decoder cascade spec; "
            "(2) implements Style B train_step per Z6 reference template at "
            "src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py. "
            "See module docstring 'L1 FOLLOW-UP CONTRACT' section for the "
            "drop-in replacement specification. Per Catalog #240 + #325 + "
            "CLAUDE.md 'Substrate scaffolds MUST be COMPLETE or "
            "RESEARCH-ONLY' non-negotiable: failing-fast HERE is the correct "
            "L0 SCAFFOLD posture."
        )

    def sample_batch(self, batch_size: int, seed: int) -> Any:
        """Sample a batch of pair_indices for one training step."""
        import numpy as np

        import mlx.core as mx

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
        """Style A diagnostic-only stub (G adapter uses Style B train_step)."""
        raise NotImplementedError(
            "G=NIRVANA adapter uses Style B (train_step). See module docstring "
            "'L1 FOLLOW-UP CONTRACT' #2."
        )

    def optimizer_step(self, model: Any, loss: Any, learning_rate: float) -> None:
        """Style A stub (G adapter uses Style B train_step)."""
        raise NotImplementedError(
            "G=NIRVANA adapter uses Style B (train_step). See module docstring."
        )

    def train_step(
        self,
        batch: Any,
        learning_rate: float,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Style B combined value+grad+update (L1+Phase-2 follow-up contract #2)."""
        raise NotImplementedError(
            "G=NIRVANA train_step pending L1+Phase-2 follow-up; see module "
            "docstring 'L1 FOLLOW-UP CONTRACT' #2 for the drop-in "
            "mlx.nn.value_and_grad over per-level cascading residual loss "
            "summed across all decoder levels."
        )

    def export_state_dict(self, model: Any, path: Path) -> None:
        """Export G trained state via PyTorch sister bridge."""
        raise NotImplementedError(
            "G=NIRVANA export_state_dict pending L1+Phase-2 follow-up; "
            "PyTorch sister bridge target is tac.substrates.nirvana_cascading"
            "_nerv.inflate (already landed PyTorch inflate path)."
        )

    def export_archive(
        self,
        model: Any,
        output_dir: Path,
    ) -> tuple[Path, str, int] | None:
        """Export NIRVANA1 archive via canonical archive grammar (already landed)."""
        raise NotImplementedError(
            "G=NIRVANA export_archive pending L1+Phase-2 follow-up; the "
            "NIRVANA1 byte-deterministic grammar at tac.substrates.nirvana_"
            "cascading_nerv.archive is already landed; only the state_dict "
            "serialization piece is pending."
        )

    def score_aware_components(
        self,
        model: Any,
        batch: Any,
    ) -> Mapping[str, float] | None:
        """G L2 currently DEFERS per-axis SegNet/PoseNet decomposition.

        Per Z6 reference template precedent + Catalog #164 + #226: true
        contest-grade score-aware Lagrangian routes through SegNet/PoseNet
        in PyTorch sister L2 promotion path; the L2 MLX trainer is
        reconstruction-proxy only.

        Returns:
            None (signaling caller that per-axis is N/A at L2 MLX).
        """
        return None


__all__ = ["NirvanaCascadingNervLongTrainingAdapter"]
