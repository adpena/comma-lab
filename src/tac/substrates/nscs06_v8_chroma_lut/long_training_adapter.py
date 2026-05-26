# SPDX-License-Identifier: MIT
"""C'=NSCS06-v8-chroma-LUT long-training adapter — L1-PROMOTION-CASCADE shell.

Per Path 3 canonical-substrate-development-cascade doctrine (commit
`fb270e9b6`) + L2-INFRA-BUILD canonical helper (commit `f5e4784ef`) +
Tier1-T3-OP7-OP8 cascade amendment (commit `b96418424`) + L1-PROMOTION-
CASCADE-B-C-E-G-J charter 2026-05-26: this adapter is the canonical
Protocol-conformant skeleton mirroring D=Z6's reference template
(``src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py``).

PARADIGM ROUTING DECISION (Catalog #290 canonical-vs-unique per layer)
====================================================================
NSCS06 v8 chroma_lut is FUNDAMENTALLY NOT gradient-trainable: the
substrate is a deterministic per-SegNet-class chroma lookup-table
codec. The "training" loop at
``tac.substrates.nscs06_v8_chroma_lut.mlx_iteration.iterate_chroma_lut_policies_via_mlx``
is an iterative POLICY refinement (cargo-cult-unwind aggregation
policies enumerated at ``enumerate_cargo_cult_unwind_arms``), not a
gradient-descent training loop.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog
#290: the L2 helper's gradient-train assumption (Style A loss_fn +
optimizer_step OR Style B train_step via mlx.nn.value_and_grad) is
PRINCIPLED MISMATCH for C'. Forking the canonical L2 helper into a
canonical L2-iteration helper (sister abstraction over iterative-policy
refinement) is the correct L1-PROMOTION path for C' — and is OUT OF
SCOPE for the L1-PROMOTION-CASCADE-B-C-E-G-J charter (which assumes
gradient-trainable substrates per the D=Z6 reference template).

L1 FOLLOW-UP CONTRACT (paradigm-routed)
======================================
The L1+sister-canonical-helper follow-up:

1. EITHER: land a sister canonical helper
   ``tac.training.long_iteration_canonical`` mirroring
   ``tac.training.long_training_canonical`` for iterative-policy
   substrates (C' + sister deterministic-codec substrates); the
   canonical Protocol surface would be ``iterate_policies`` instead
   of ``train_step``.
2. OR: implement C' as a degenerate adapter where ``train_step``
   wraps ``iterate_chroma_lut_policies_via_mlx`` as a single
   iteration step (sub-optimal but Protocol-conformant for the
   parallel L2 dispatch surface; sacrifices per-epoch granularity).
3. OR: route C' through the existing
   ``tac.substrates.nscs06_v8_chroma_lut.distinguishing_feature_smoke``
   sister that runs the deterministic-LUT cascade end-to-end (NOT a
   long-training surface; the L2 helper would be misused).

THIS adapter shell raises ``NotImplementedError`` with explicit
paradigm-routing guidance per Catalog #240 + CLAUDE.md "Substrate
scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable.

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #127/
#192/#317/#341: every L2 long-training output is non-promotable by
construction (paradigm-routing also).

[verified-against: src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py canonical reference template]
[verified-against: src/tac/training/long_training_canonical.py SubstrateLongTrainingAdapter Protocol]
[verified-against: src/tac/substrates/nscs06_v8_chroma_lut/mlx_iteration.py iterate_chroma_lut_policies_via_mlx]
[verified-against: src/tac/substrates/nscs06_v8_chroma_lut/architecture.py Nscs06V8ChromaLutConfig deterministic LUT]
[verified-against: .omx/research/path_3_c_nscs06_v8_chroma_lut_substrate_design_20260526.md]
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class Nscs06V8ChromaLutLongTrainingAdapter:
    """C'=NSCS06-v8-chroma-LUT canonical adapter shell — paradigm-routed shell.

    Args:
        config: ``Nscs06V8ChromaLutConfig`` instance (per ``architecture.py``).
        target_rgb_0: MLX array (num_pairs, H, W, 3) float32 in [0, 1].
        target_rgb_1: MLX array (num_pairs, H, W, 3) float32 in [0, 1].

    Raises:
        NotImplementedError: substrate is deterministic-LUT-codec paradigm;
            canonical L2 gradient-training helper is PRINCIPLED MISMATCH per
            Catalog #290. L1 follow-up routes to sister canonical iteration
            helper OR adapts the deterministic-LUT cascade.
    """

    substrate_id: str = "nscs06_v8_chroma_lut"

    def __init__(
        self,
        config: Any,
        target_rgb_0: Any,
        target_rgb_1: Any,
    ):
        try:
            import mlx.core as mx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "C'=NSCS06-v8-chroma-LUT long-training adapter requires MLX "
                "(Apple Silicon only) for the iterative policy refinement "
                "sister at mlx_iteration.py. Install via `pip install mlx`."
            ) from exc

        self.config = config
        self.target_rgb_0 = target_rgb_0
        self.target_rgb_1 = target_rgb_1
        self.model: Any = None

        raise NotImplementedError(
            "C'=NSCS06-v8-chroma-LUT adapter __init__ is L0 SCAFFOLD "
            "paradigm-routed shell per L1-PROMOTION-CASCADE charter "
            "2026-05-26. NSCS06 v8 chroma_lut is FUNDAMENTALLY a "
            "deterministic per-SegNet-class chroma lookup-table codec; the "
            "'training' loop is iterative policy refinement at "
            "tac.substrates.nscs06_v8_chroma_lut.mlx_iteration."
            "iterate_chroma_lut_policies_via_mlx, NOT gradient descent. The "
            "canonical L2 helper's gradient-train assumption (Style A "
            "loss_fn + optimizer_step OR Style B train_step via "
            "mlx.nn.value_and_grad) is PRINCIPLED MISMATCH per Catalog #290. "
            "L1 follow-up routes to: (1) sister canonical "
            "tac.training.long_iteration_canonical helper for iterative-"
            "policy substrates (recommended; preserves L0 SCAFFOLD posture); "
            "OR (2) degenerate train_step wrapper around "
            "iterate_chroma_lut_policies_via_mlx (sub-optimal); "
            "OR (3) route through distinguishing_feature_smoke sister "
            "(deterministic-LUT cascade end-to-end; NOT a long-training "
            "surface). Per Catalog #240 + #290 + CLAUDE.md 'UNIQUE-AND-"
            "COMPLETE-PER-METHOD operating mode': failing-fast HERE is the "
            "correct posture; the canonical L2 helper is the wrong layer "
            "for this substrate's paradigm."
        )

    def sample_batch(self, batch_size: int, seed: int) -> Any:
        """Sample a batch of pair_indices for one iteration step."""
        import numpy as np

        import mlx.core as mx

        rng = np.random.RandomState(seed)
        size = min(batch_size, 600)
        sampled_np = rng.choice(600, size=size, replace=False)
        return mx.array(sampled_np.astype("int32"))

    def loss_fn(
        self,
        model: Any,
        batch: Any,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Paradigm-mismatch stub."""
        raise NotImplementedError(
            "C'=NSCS06-v8-chroma-LUT is deterministic-LUT-codec paradigm; "
            "see module docstring for paradigm-routing options."
        )

    def optimizer_step(self, model: Any, loss: Any, learning_rate: float) -> None:
        """Paradigm-mismatch stub."""
        raise NotImplementedError(
            "C'=NSCS06-v8-chroma-LUT is deterministic-LUT-codec paradigm; "
            "see module docstring for paradigm-routing options."
        )

    def train_step(
        self,
        batch: Any,
        learning_rate: float,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Paradigm-mismatch stub."""
        raise NotImplementedError(
            "C'=NSCS06-v8-chroma-LUT train_step is paradigm-mismatch; "
            "iterative policy refinement at mlx_iteration."
            "iterate_chroma_lut_policies_via_mlx is the substrate's "
            "canonical 'training' loop. See module docstring for "
            "L1 paradigm-routing options."
        )

    def export_state_dict(self, model: Any, path: Path) -> None:
        """Export deterministic LUT state (no neural state_dict)."""
        raise NotImplementedError(
            "C'=NSCS06-v8-chroma-LUT export_state_dict pending L1 follow-up; "
            "deterministic chroma_lut at archive.py is the canonical "
            "state-equivalent (no neural state_dict)."
        )

    def export_archive(
        self,
        model: Any,
        output_dir: Path,
    ) -> tuple[Path, str, int] | None:
        """Export NSCS06 v8 chroma_lut archive."""
        raise NotImplementedError(
            "C'=NSCS06-v8-chroma-LUT export_archive pending L1 follow-up; "
            "canonical archive sister at tac.substrates.nscs06_v8_chroma_lut."
            "archive."
        )

    def score_aware_components(
        self,
        model: Any,
        batch: Any,
    ) -> Mapping[str, float] | None:
        """C' L2 currently DEFERS per-axis decomposition.

        Returns:
            None (signaling caller that per-axis is N/A at L2 paradigm-routed).
        """
        return None


__all__ = ["Nscs06V8ChromaLutLongTrainingAdapter"]
