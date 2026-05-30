# SPDX-License-Identifier: MIT
"""B'=Z7-Mamba-2-v2 long-training adapter — L1-PROMOTION-CASCADE shell.

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
Per ``__init__.py`` ``IMPLEMENTATION_STATUS`` =
``"L0_scaffold_skeleton_only_design_complete_implementation_pending_per_phase_3_L0_SCAFFOLD_landing_memo"``
+ ``RESEARCH_ONLY = True``: B' is design-only L0 SCAFFOLD. Per
``architecture.py`` empirical verification:
- ``Z7Mamba2V2Substrate.__init__`` raises ``NotImplementedError``
- ``Mamba2TemporalDecoder.__init__`` raises ``NotImplementedError``
- ``Mamba2V2Cell.__init__`` raises ``NotImplementedError``

Per ``PLANNED_PUBLIC_API``: ``Mamba2V2Cell + Mamba2TemporalDecoder +
Z7MCM3Archive + pack_archive + parse_archive + replay_latent_sequence
+ inflate_one_video`` are ALL pending L1 EMPIRICAL build per Phase 3
L0 SCAFFOLD design memo.

Per Catalog #229 PV empirical finding (charter premise verification):
NONE of the substrate's class constructors function; the canonical L2
helper's ``PolyakEMAShadow`` ``state_dict()`` / ``parameters()``
contract cannot be satisfied until the L1 EMPIRICAL build subagent
lands ALL of the ``PLANNED_PUBLIC_API`` classes.

L1 FOLLOW-UP CONTRACT (drop-in replacement)
==========================================
The L1 EMPIRICAL build follow-up subagent removes the ``__init__``
``NotImplementedError`` + implements (per Phase 3 L0 SCAFFOLD design
memo §7.1 + §7.2):

1. Land ``Mamba2V2Cell(nn.Module)`` per design memo §7.1 (MLX-native +
   SSD-scan integration; selective state-space primitives).
2. Land ``Mamba2TemporalDecoder(nn.Module)`` per design memo §7.2
   (temporal Conv1D pre-stage + Mamba2V2Cell sequence).
3. Land ``Z7Mamba2V2Substrate(nn.Module)`` composing the above.
4. Land ``Z7MCM3Archive`` byte-deterministic grammar per Catalog #146.
5. Implement ``train_step`` Style B per Z6 reference template:
   ``mlx.nn.value_and_grad`` over reconstruction loss + predictive-
   coding residual capacity term per ``CANONICAL_EQUATION_IDS`` =
   ``("predictive_coding_residual_capacity_v1_proposed_per_audit_e757bb74c_op_routable_3",)``.
6. Implement ``export_state_dict`` via PyTorch sister bridge
   (Catalog #1251 pattern).
7. Implement ``export_archive`` via canonical ``pack_archive``.
8. Implement ``inflate_one_video`` per Catalog #146 runtime contract.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" +
Catalog #290 canonical-vs-unique decision per layer:

- ADOPT canonical Protocol contract (substrate-agnostic surface; LOC reduction)
- ADOPT canonical PolyakEMAShadow + TelemetrySink + CheckpointWriter via helper
- ADOPT canonical Provenance + posterior emission via helper
- UNIQUE B' substrate-specific axes (L1 follow-up):
  * Mamba-2 selective state-space primitives (CC-A through CC-J unwound
    per Phase 1 cargo-cult audit; HARD-EARNED-vs-CARGO-CULTED 16-layer
    canonical-vs-unique decision table from Phase 3)
  * Fresh substrate design (Path c per Phase 2 decision; NOT extension
    of time_traveler_l5_z7_mamba2 v1 sister)
  * Z7MCM3 archive grammar (substrate-class-shift from v1's Z7MCM2)

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #127/
#192/#317/#341: every L2 long-training output is non-promotable by
construction; canonical Provenance + posterior anchor markers are
auto-stamped by the canonical helper.

[verified-against: src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py canonical reference template]
[verified-against: src/tac/training/long_training_canonical.py SubstrateLongTrainingAdapter Protocol]
[verified-against: src/tac/substrates/z7_mamba2_v2_fresh_substrate/__init__.py RESEARCH_ONLY=True L0 SCAFFOLD posture]
[verified-against: src/tac/substrates/z7_mamba2_v2_fresh_substrate/architecture.py all classes raise NotImplementedError]
[verified-against: .omx/research/path_3_b_z7_mamba_2_substrate_design_20260526.md]
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.framework_agnostic import require_mlx_core


class Z7Mamba2V2LongTrainingAdapter:
    """B'=Z7-Mamba-2-v2 canonical adapter shell — L1-PROMOTION-CASCADE skeleton.

    Args:
        config: ``Z7Mamba2V2Config`` instance (per ``architecture.py``).
        target_rgb_0: MLX array (num_pairs, H, W, 3) float32 in [0, 1].
        target_rgb_1: MLX array (num_pairs, H, W, 3) float32 in [0, 1].

    Raises:
        ImportError: MLX is not installed (B' is MLX-first by design).
        NotImplementedError: substrate is design-only L0 SCAFFOLD;
            ``Mamba2V2Cell`` + ``Mamba2TemporalDecoder`` +
            ``Z7Mamba2V2Substrate`` ALL raise NotImplementedError. L1
            EMPIRICAL build follow-up lands the full substrate per
            Phase 3 L0 SCAFFOLD design memo + Catalog #325 per-substrate
            symposium contract.
    """

    substrate_id: str = "z7_mamba2_v2_fresh_substrate"

    def __init__(
        self,
        config: Any,
        target_rgb_0: Any,
        target_rgb_1: Any,
    ):
        require_mlx_core()

        self.config = config
        self.target_rgb_0 = target_rgb_0
        self.target_rgb_1 = target_rgb_1
        self.model: Any = None

        raise NotImplementedError(
            "B'=Z7-Mamba-2-v2 adapter __init__ is L0 SCAFFOLD design-only "
            "structural shell per L1-PROMOTION-CASCADE charter 2026-05-26. "
            "The substrate is DESIGN-ONLY: Z7Mamba2V2Substrate.__init__, "
            "Mamba2TemporalDecoder.__init__, AND Mamba2V2Cell.__init__ ALL "
            "raise NotImplementedError per architecture.py. The PLANNED_PUBLIC"
            "_API (Mamba2V2Cell + Mamba2TemporalDecoder + Z7MCM3Archive + "
            "pack_archive + parse_archive + replay_latent_sequence + "
            "inflate_one_video) is ENTIRELY pending L1 EMPIRICAL build per "
            "Phase 3 L0 SCAFFOLD design memo §7.1 + §7.2 + Catalog #325 "
            "per-substrate symposium contract. L1 EMPIRICAL build subagent "
            "removes this guard + implements ALL 8 planned classes per "
            "Phase 3 design memo. See module docstring 'L1 FOLLOW-UP CONTRACT' "
            "section for the drop-in replacement specification. Per Catalog "
            "#240 + CLAUDE.md 'Substrate scaffolds MUST be COMPLETE or "
            "RESEARCH-ONLY' non-negotiable: failing-fast HERE is the correct "
            "L0 SCAFFOLD posture; the canonical L2 helper's PolyakEMAShadow "
            "would otherwise raise less actionable errors downstream."
        )

    def sample_batch(self, batch_size: int, seed: int) -> Any:
        """Sample a batch of pair_indices for one training step."""
        import numpy as np

        mx = require_mlx_core()

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
        """Style A diagnostic-only stub (B' adapter uses Style B train_step)."""
        raise NotImplementedError(
            "B'=Z7-Mamba-2-v2 adapter uses Style B (train_step). See module "
            "docstring 'L1 FOLLOW-UP CONTRACT' #5."
        )

    def optimizer_step(self, model: Any, loss: Any, learning_rate: float) -> None:
        """Style A stub (B' adapter uses Style B train_step)."""
        raise NotImplementedError(
            "B'=Z7-Mamba-2-v2 adapter uses Style B (train_step). See module docstring."
        )

    def train_step(
        self,
        batch: Any,
        learning_rate: float,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Style B combined value+grad+update (L1 EMPIRICAL build contract #5)."""
        raise NotImplementedError(
            "B'=Z7-Mamba-2-v2 train_step pending L1 EMPIRICAL build per "
            "Phase 3 L0 SCAFFOLD design memo §7.1 + §7.2; see module "
            "docstring 'L1 FOLLOW-UP CONTRACT' #5 for the drop-in "
            "mlx.nn.value_and_grad over reconstruction loss + predictive-"
            "coding residual capacity term per CANONICAL_EQUATION_IDS = "
            "('predictive_coding_residual_capacity_v1_proposed_per_audit_"
            "e757bb74c_op_routable_3',)."
        )

    def export_state_dict(self, model: Any, path: Path) -> None:
        """Export B' trained state via PyTorch sister bridge (Catalog #1251)."""
        raise NotImplementedError(
            "B'=Z7-Mamba-2-v2 export_state_dict pending L1 EMPIRICAL build "
            "follow-up; PyTorch sister bridge target = Phase 3 design memo "
            "§7.3."
        )

    def export_archive(
        self,
        model: Any,
        output_dir: Path,
    ) -> tuple[Path, str, int] | None:
        """Export Z7MCM3 archive via canonical pack_archive (PLANNED_PUBLIC_API)."""
        raise NotImplementedError(
            "B'=Z7-Mamba-2-v2 export_archive pending L1 EMPIRICAL build "
            "follow-up; PLANNED_PUBLIC_API.pack_archive + Z7MCM3Archive "
            "land at L1 per Phase 3 design memo §7.4."
        )

    def score_aware_components(
        self,
        model: Any,
        batch: Any,
    ) -> Mapping[str, float] | None:
        """B' L2 currently DEFERS per-axis SegNet/PoseNet decomposition.

        Returns:
            None (signaling caller that per-axis is N/A at L2 MLX).
        """
        return None


__all__ = ["Z7Mamba2V2LongTrainingAdapter"]
