# SPDX-License-Identifier: MIT
"""J=MDL-IBPS long-training adapter — L1-PROMOTION-CASCADE structural shell.

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
Per ``__init__.py`` ``RESEARCH_ONLY = True`` + ``IMPLEMENTATION_STATUS``
explicit L0 SCAFFOLD posture: the substrate's ``MDLIBPSJRendererMLX``
exists at ``mlx_renderer.py`` (forward ``render_pair`` works for
inference) but the standalone primitives (FilmProjMLX / CoordMLPBaseMLX
/ MINECriticMLX) are NOT yet wrapped as a trainable ``mlx.nn.Module``
with registered parameters per the ``mlx.nn.value_and_grad`` contract.

Per Catalog #229 PV empirical finding (charter premise verification):
``self.model`` cannot satisfy the canonical L2 helper's
``PolyakEMAShadow`` ``state_dict()`` / ``parameters()`` contract until
the L1 follow-up subagent lands the trainable wrapper. Per CLAUDE.md
"Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable:
the adapter's ``__init__`` fails-fast with explicit L1-follow-up
guidance rather than silently constructing a partial model that would
crash inside the canonical helper.

L1 FOLLOW-UP CONTRACT (drop-in replacement)
==========================================
The L1 follow-up subagent removes the ``__init__`` ``NotImplementedError``
+ implements:

1. Wrap FilmProjMLX + CoordMLPBaseMLX + MINECriticMLX as ONE
   ``mlx.nn.Module`` subclass with trainable ``self.weights = ...``
   parameters (canonical MLX module pattern per Z6 reference at
   ``src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py:447``
   ``Z6PredictiveCodingMLXRenderer(nn.Module)``).
2. Implement ``train_step`` Style B per Z6 reference template:
   ``mlx.nn.value_and_grad`` over a closure that computes
   ``L_score`` (canonical Catalog #164 ``score_pair_components``) +
   ``beta * L_IB`` (``MINE_lower_bound_mlx``) +
   ``lambda_sparse * L_sparse`` (``sparse_laplacian_l1``).
3. Implement ``export_state_dict`` via PyTorch sister bridge
   (Catalog #1251 pattern; see ``inflate.py`` ``CoordMLPBaseTorch`` /
   ``FilmProjTorch`` for the target shape).
4. Implement ``export_archive`` via canonical ``pack_archive`` at
   ``archive.py`` (header + film_proj_blob + coord_mlp_blob +
   categorical_indices_blob + (optional) mine_critic_blob).

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" +
Catalog #290 canonical-vs-unique decision per layer:

- ADOPT canonical Protocol contract (substrate-agnostic surface; ~78% LOC reduction)
- ADOPT canonical PolyakEMAShadow + TelemetrySink + CheckpointWriter via helper
- ADOPT canonical Provenance + posterior emission via helper
- UNIQUE J substrate-specific axes (L1 follow-up):
  * Discrete categorical posterior K=16 x G=12 = 48 bits/pair (CC-J-2 unwind)
  * FiLM-modulated procedural coord-MLP (CC-J-5 unwind)
  * MINE critic for I(z; frames) lower bound (CC-J-4 unwind; Belghazi 2018)
  * Sparse-Laplacian regularizer on FiLM matrices (MacKay sparse-coding)
  * Gumbel-Softmax reparametrization for discrete-posterior training

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #127/
#192/#317/#341: every L2 long-training output is non-promotable by
construction; canonical Provenance + posterior anchor markers are
auto-stamped by the canonical helper.

[verified-against: src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py canonical reference template]
[verified-against: src/tac/training/long_training_canonical.py SubstrateLongTrainingAdapter Protocol]
[verified-against: src/tac/substrates/mdl_ibps_j_discrete_categorical_mine_hybrid/__init__.py RESEARCH_ONLY=True L0 SCAFFOLD posture]
[verified-against: .omx/research/path_3_canonical_substrate_development_cascade_doctrine_20260526.md]
[verified-against: .omx/research/path_3_j_mdl_ibps_substrate_design_20260526.md]
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class MdlIbpsJLongTrainingAdapter:
    """J=MDL-IBPS canonical adapter shell — L1-PROMOTION-CASCADE structural skeleton.

    Args:
        config: J substrate config (substrate-specific; L1 follow-up adds
            ``MdlIbpsJConfig`` dataclass exposing CATEGORICAL_K / CATEGORICAL_G
            / HIDDEN_DIM module-level constants per CC-J-2/5 unwind).
        target_rgb_0: MLX array (num_pairs, H, W, 3) float32 in [0, 1].
        target_rgb_1: MLX array (num_pairs, H, W, 3) float32 in [0, 1].
        beta_ib: IB Lagrangian weight (Higgins-memorial; default 1e-3 per
            DEFAULT_BETA_SWEEP center).
        lambda_sparse: sparse-Laplacian L1 weight on FiLM matrices
            (MacKay sparse-coding; default 1e-4 per DEFAULT_LAMBDA_SPARSE).

    Raises:
        ImportError: MLX is not installed (J is MLX-first by design).
        NotImplementedError: substrate is L0 SCAFFOLD; trainable
            ``mlx.nn.Module`` wrapper of MDLIBPSJRendererMLX primitives
            lands at L1 follow-up per CLAUDE.md "Substrate scaffolds
            MUST be COMPLETE or RESEARCH-ONLY" + Catalog #240.
    """

    substrate_id: str = "path_3_j_mdl_ibps"

    def __init__(
        self,
        config: Any,
        target_rgb_0: Any,
        target_rgb_1: Any,
        beta_ib: float = 1e-3,
        lambda_sparse: float = 1e-4,
    ):
        try:
            import mlx.core as mx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "J=MDL-IBPS long-training adapter requires MLX (Apple Silicon "
                "only). Install via `pip install mlx`. The numpy reference at "
                "tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid."
                "numpy_reference covers CPU-only inference."
            ) from exc

        # Stash inputs for L1 follow-up.
        self.config = config
        self.target_rgb_0 = target_rgb_0
        self.target_rgb_1 = target_rgb_1
        self.beta_ib = float(beta_ib)
        self.lambda_sparse = float(lambda_sparse)
        self.model: Any = None

        raise NotImplementedError(
            "J=MDL-IBPS adapter __init__ is L0 SCAFFOLD structural shell per "
            "L1-PROMOTION-CASCADE charter 2026-05-26. The substrate's "
            "MDLIBPSJRendererMLX (mlx_renderer.py) exists for inference but the "
            "standalone primitives (FilmProjMLX/CoordMLPBaseMLX/MINECriticMLX) "
            "are NOT yet wrapped as a trainable mlx.nn.Module with registered "
            "parameters per mlx.nn.value_and_grad contract. L1 follow-up "
            "subagent removes this guard + implements the trainable wrapper + "
            "Style B train_step per Z6 reference template at "
            "src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py. "
            "See module docstring 'L1 FOLLOW-UP CONTRACT' section for the "
            "drop-in replacement specification. Per Catalog #240 + CLAUDE.md "
            "'Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY' "
            "non-negotiable: failing-fast HERE is the correct L0 SCAFFOLD "
            "posture; the canonical L2 helper's PolyakEMAShadow would "
            "otherwise raise the same TypeError downstream with a less "
            "actionable error message."
        )

    def sample_batch(self, batch_size: int, seed: int) -> Any:
        """Sample a batch of pair_indices for one training step."""
        import numpy as np

        import mlx.core as mx

        from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid import NUM_PAIRS

        rng = np.random.RandomState(seed)
        size = min(batch_size, NUM_PAIRS)
        sampled_np = rng.choice(NUM_PAIRS, size=size, replace=False)
        return mx.array(sampled_np.astype("int32"))

    def loss_fn(
        self,
        model: Any,
        batch: Any,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Style A diagnostic-only stub (J adapter uses Style B train_step)."""
        raise NotImplementedError(
            "J=MDL-IBPS adapter uses Style B (train_step combined value+grad+update). "
            "See module docstring 'L1 FOLLOW-UP CONTRACT' #2."
        )

    def optimizer_step(self, model: Any, loss: Any, learning_rate: float) -> None:
        """Style A stub (J adapter uses Style B train_step)."""
        raise NotImplementedError(
            "J=MDL-IBPS adapter uses Style B (train_step). See module docstring."
        )

    def train_step(
        self,
        batch: Any,
        learning_rate: float,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Style B combined value+grad+update (L1 follow-up contract #2)."""
        raise NotImplementedError(
            "J=MDL-IBPS train_step pending L1 follow-up; see module docstring "
            "'L1 FOLLOW-UP CONTRACT' #2 for the drop-in mlx.nn.value_and_grad "
            "+ L_score + beta * L_IB + lambda_sparse * L_sparse composition."
        )

    def export_state_dict(self, model: Any, path: Path) -> None:
        """Export J trained state to .pt for PyTorch sister inflate (Catalog #1251)."""
        raise NotImplementedError(
            "J=MDL-IBPS export_state_dict pending L1 follow-up; see "
            "tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid.inflate "
            "(CoordMLPBaseTorch + FilmProjTorch target shape)."
        )

    def export_archive(
        self,
        model: Any,
        output_dir: Path,
    ) -> tuple[Path, str, int] | None:
        """Export MDLIBPSJArchive via canonical pack_archive (archive.py)."""
        raise NotImplementedError(
            "J=MDL-IBPS export_archive pending L1 follow-up; canonical "
            "pack_archive at tac.substrates.mdl_ibps_j_discrete_categorical"
            "_mine_hybrid.archive composes: header + film_proj_blob + "
            "coord_mlp_blob + categorical_indices_blob + (optional) mine_critic_blob."
        )

    def score_aware_components(
        self,
        model: Any,
        batch: Any,
    ) -> Mapping[str, float] | None:
        """J L2 currently DEFERS per-axis SegNet/PoseNet decomposition.

        Per Z6 reference template precedent + Catalog #164 + #226: true
        contest-grade score-aware Lagrangian routes through SegNet/PoseNet
        in PyTorch sister L2 promotion path; the L2 MLX trainer is
        reconstruction-proxy + IB-proxy + sparse-proxy only.

        Returns:
            None (signaling caller that per-axis is N/A at L2 MLX).
        """
        return None


__all__ = ["MdlIbpsJLongTrainingAdapter"]
