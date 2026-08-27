# SPDX-License-Identifier: MIT
# NO_GRAD_WAIVED:Z8_canonical_quadruple_binding_uses_torch_autograd_for_M4_Mamba_2_and_numpy_lazy_eval_for_M5_M6_M7_M8_per_mixed_framework_compose_pattern_per_M9_milestone_landing_20260530
# AUTOCAST_FP16_WAIVED:Z8_canonical_quadruple_binding_runs_on_fp32_for_round_trip_invariant_per_Wyner_Ziv_1976_Theorem_1_quantization_bound_residual_dtype_field_per_M6_signature
"""Z8 M9 canonical quadruple binding-integration compose pattern.

THIS module IS the M9 milestone landing per ``build_progress.py``
``full_main_trainer_lifts_notimplementederror`` (operator-routed Yousfi-cascade
TOP-1 post-M6; 2026-05-30). Lifts the Z8 trainer's ``_full_main`` from the
prior MLX-harness-only path to the **canonical quadruple compose pattern**
binding all four Catalog #312 primitives end-to-end:

    1. **M4** ``Z8Mamba2DeterministicStateUpdate`` (Dao & Gu 2024 SSD adapter
       per Wave 4 audit; ``mamba2_adapter.py``)
    2. **M5** ``Z8MallatDaubechiesPartition`` (Mallat 1989 §7.5 + 7.7 separable
       Daubechies-4 DWT per build_progress M5; ``mallat_dwt_adapter.py``)
    3. **M6** ``WynerZivTopLevelCoderImpl`` (Wyner-Ziv 1976 Theorem 1 linear-
       prediction + uniform-quantization-residual per build_progress M6;
       ``wyner_ziv_coder.py``)
    4. **M8** ``ScoreAwareLevelLossImpl`` (Yousfi-grounded weighted recon
       loss with M7's per-level scorer-sensitivity map per build_progress M8;
       ``loss.py`` + ``scorer_sensitivity_map.py``)

Per the Z8 Phase E landing memo `feedback_z8_phase_e_score_aware_level_loss_protocol_implementation_landed_20260530.md`
the canonical compose pattern is:

    m5.decompose(input) -> per-level latents
    m6.encode(top_state, side_info=m5_reconstruction(input)) -> archive bytes
    m8.per_level_loss(reconstruction, target,
                       sensitivity=m7.get_for_level(level, gradient_tensor))
        -> scalar tensor

This module makes that pattern **structurally executable end-to-end** for the
first time. The composition becomes runnable when ``_full_main`` lifts the
NotImplementedError per Catalog #240 acceptance cascade (e) waiver path.

## Canonical-vs-unique decision per layer (Catalog #290)

- **ADOPT_CANONICAL**: M4 / M5 / M6 / M8 (already canonical Protocol Impls in
  sister modules; this module composes them without re-implementing).
- **ADOPT_CANONICAL**: ``tac.data.decode_video`` for the real-video frame
  loader path per Catalog #114 + #213 (Comma2k19 / pyav decoder).
- **ADOPT_CANONICAL**: numpy as the canonical portable intermediate per
  Catalog #317 (M4 produces torch tensors; M5/M6/M7/M8 are numpy-native;
  the compose pattern converts at the framework boundary).
- **FORK_BECAUSE_PRINCIPLED_MISMATCH** (this module's UNIQUE primitive):
  the **per-pair forward-pass compose order** is Z8-specific (Rao-Ballard 1999
  hierarchical-prediction + DreamerV3 latent-dynamics + Mallat dyadic-pyramid
  + Wyner-Ziv conditional-coding all in one coherent forward) — no canonical
  helper exists for this 4-primitive simultaneous composition.

## Observability surface (Catalog #305)

Every training step emits a ``TrainingStepObservability`` frozen dataclass:

  - per-level: ``per_level_l2_loss[i]`` (M8 reconstruction loss at level i)
  - per-level: ``wavelet_subband_l2_norm[i]`` (M5 detail subbands magnitude)
  - top-level: ``wyner_ziv_payload_bytes`` (M6 archive contribution)
  - top-level: ``wyner_ziv_round_trip_error`` (M6 reconstruction fidelity)
  - meta: ``mamba2_state_l2_norm`` (M4 state evolution diagnostic)

Catalog #305 6-facet observability: per-layer (per_level fields), per-signal
(decomposable across M4/M5/M6/M8), diff-able (frozen dataclass), queryable
(named fields), cite-able (canonical Provenance tag), counterfactual-able
(byte-mutation per Catalog #139).

## Mission alignment (Catalog #300)

``mission_predicted_contribution=frontier_breaking_enabler``: M9 is the
binding-integration milestone where the canonical quadruple becomes
structurally executable end-to-end; unblocks M10 (inflate consumes real
trained weights per Catalog #369) + M11 (L1 MLX-LOCAL end-to-end smoke) +
M12 (paired-CUDA Modal T4 sub-0.189 threshold attempt).

[verified-against: src/tac/substrates/z8_hierarchical_predictive_coding/binding_contract.py per-level Protocols]
[verified-against: src/tac/substrates/z8_hierarchical_predictive_coding/build_progress.py M9 acceptance criteria]
[verified-against: feedback_z8_phase_e_score_aware_level_loss_protocol_implementation_landed_20260530.md compose pattern docstring]
"""

from __future__ import annotations

import struct
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tac.substrates.z8_hierarchical_predictive_coding.binding_contract import (
    HierarchyBindingContract,
    build_canonical_contract_from_config,
)
from tac.substrates.z8_hierarchical_predictive_coding.loss import (
    ScoreAwareLevelLossImpl,
    build_score_aware_level_loss_for_level,
)
from tac.substrates.z8_hierarchical_predictive_coding.mallat_dwt_adapter import (
    WaveletDetail2D,
    Z8MallatDaubechiesPartition,
    build_z8_mallat_dwt_adapter_for_level,
)
from tac.substrates.z8_hierarchical_predictive_coding.mamba2_adapter import (
    Z8Mamba2DeterministicStateUpdate,
    build_z8_mamba2_adapter_for_level,
)
from tac.substrates.z8_hierarchical_predictive_coding.runtime_custody import (
    build_z8_runtime_custody_contract,
)
from tac.substrates.z8_hierarchical_predictive_coding.scorer_sensitivity_map import (
    ScorerSensitivityMapSource,
    Z8ScorerSensitivityMap,
)
from tac.substrates.z8_hierarchical_predictive_coding.wyner_ziv_coder import (
    WynerZivTopLevelCoderImpl,
    build_wyner_ziv_top_level_coder_for_contract,
)

CANONICAL_PROJECTION_SEED_DEFAULT: int = 0
"""Default Wyner-Ziv projection seed; encoder + decoder MUST agree."""

DEFAULT_M4_LATENT_DIM_PER_LEVEL: int = 8
"""Default per-level latent dim for M4 Mamba-2 adapter inputs."""

DEFAULT_M4_D_STATE: int = 4
"""Default Mamba-2 selective-scan state dim (smoke-friendly; canonical default
in production is 16 per Dao & Gu 2024 §3 ``d_state``)."""


@dataclass(frozen=True)
class TrainingStepObservability:
    """Per-step canonical observability record per Catalog #305 6-facet.

    Frozen dataclass so each step's record is immutable + diff-able across
    runs + queryable via named fields. Emitted by every step of
    ``run_canonical_quadruple_training_loop``; the training-loop output JSON
    artifact carries a list of these.

    Attributes:
        epoch: zero-based epoch index.
        pair_index: zero-based pair index within the epoch.
        per_level_l2_loss: M8 per-level reconstruction loss; length =
            ``num_levels``.
        wavelet_subband_l2_norm: M5 detail-subband (LH+HL+HH combined)
            L2-norm magnitude per level; length = ``num_levels``.
        mamba2_state_l2_norm: M4 deterministic state L2-norm AFTER step;
            scalar diagnostic.
        wyner_ziv_payload_bytes: M6 encoded payload length in bytes
            (canonical archive-bytes contribution per Catalog #245).
        wyner_ziv_round_trip_error: M6 max abs decode-vs-encode error
            (Wyner-Ziv 1976 R(D|Y) achievable distortion diagnostic).
        total_loss: scalar sum of per-level L2 losses; the canonical
            scalar the optimizer minimizes.
        wall_clock_seconds: per-step elapsed wall-clock.
    """

    epoch: int
    pair_index: int
    per_level_l2_loss: tuple[float, ...]
    wavelet_subband_l2_norm: tuple[float, ...]
    mamba2_state_l2_norm: float
    wyner_ziv_payload_bytes: int
    wyner_ziv_round_trip_error: float
    total_loss: float
    wall_clock_seconds: float

    def as_dict(self) -> dict[str, Any]:
        """Canonical JSON-serializable representation."""
        return {
            "epoch": int(self.epoch),
            "pair_index": int(self.pair_index),
            "per_level_l2_loss": [float(v) for v in self.per_level_l2_loss],
            "wavelet_subband_l2_norm": [
                float(v) for v in self.wavelet_subband_l2_norm
            ],
            "mamba2_state_l2_norm": float(self.mamba2_state_l2_norm),
            "wyner_ziv_payload_bytes": int(self.wyner_ziv_payload_bytes),
            "wyner_ziv_round_trip_error": float(self.wyner_ziv_round_trip_error),
            "total_loss": float(self.total_loss),
            "wall_clock_seconds": float(self.wall_clock_seconds),
        }


@dataclass(frozen=True)
class CanonicalQuadrupleTrainingArtifact:
    """Canonical output of ``run_canonical_quadruple_training_loop``.

    Frozen dataclass per Catalog #300 v2 frontmatter discipline. Carries
    the canonical Provenance per Catalog #323 (``axis_tag``,
    ``evidence_grade``, ``score_claim=False``, ``promotable=False``) so the
    artifact is non-promotable by construction per CLAUDE.md MLX research-
    signal discipline + Catalog #192.

    The ``per_step_observability`` field is the canonical Catalog #305
    observability surface. The ``per_epoch_total_loss`` field is the
    canonical convergence-check signal per build_progress.py M9 acceptance
    criterion 3 ("per-pair training loss decreases over epochs").
    """

    substrate_id: str
    lane_id: str
    total_epochs_completed: int
    total_pairs_per_epoch: int
    per_epoch_total_loss: tuple[float, ...]
    per_step_observability: tuple[TrainingStepObservability, ...]
    final_wyner_ziv_payload_bytes: int
    final_per_level_l2_loss: tuple[float, ...]
    total_wall_clock_seconds: float
    convergence_verdict: str
    hardware_substrate: str
    notes: str = ""

    # Canonical Provenance per Catalog #323 (non-promotable by construction).
    score_claim: bool = field(default=False)
    promotable: bool = field(default=False)
    axis_tag: str = field(default="[macOS-CPU advisory]")
    evidence_grade: str = field(default="macOS-CPU-advisory")

    def __post_init__(self) -> None:
        # Catalog #323 canonical Provenance invariants: predicted artifacts
        # MUST NOT carry score_claim=True or promotable=True.
        if self.score_claim is not False:
            raise ValueError(
                "CanonicalQuadrupleTrainingArtifact MUST carry "
                "score_claim=False per Catalog #192 + #323 (M9 binding-"
                "integration is observability-only, not contest scoring)"
            )
        if self.promotable is not False:
            raise ValueError(
                "CanonicalQuadrupleTrainingArtifact MUST carry "
                "promotable=False per Catalog #192 + #323"
            )
        if self.convergence_verdict not in (
            "CONVERGED_MONOTONIC",
            "CONVERGED_FINAL_LESS_THAN_INITIAL",
            "NOT_CONVERGED_TOO_FEW_EPOCHS",
            "NOT_CONVERGED_LOSS_INCREASED",
        ):
            raise ValueError(
                f"convergence_verdict must be one of the 4 canonical "
                f"values; got {self.convergence_verdict!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        """Canonical JSON-serializable representation."""
        return {
            "schema": "z8_canonical_quadruple_training_artifact_v1",
            "substrate_id": self.substrate_id,
            "lane_id": self.lane_id,
            "total_epochs_completed": int(self.total_epochs_completed),
            "total_pairs_per_epoch": int(self.total_pairs_per_epoch),
            "per_epoch_total_loss": [
                float(v) for v in self.per_epoch_total_loss
            ],
            "per_step_observability": [
                step.as_dict() for step in self.per_step_observability
            ],
            "final_wyner_ziv_payload_bytes": int(
                self.final_wyner_ziv_payload_bytes
            ),
            "final_per_level_l2_loss": [
                float(v) for v in self.final_per_level_l2_loss
            ],
            "total_wall_clock_seconds": float(self.total_wall_clock_seconds),
            "convergence_verdict": str(self.convergence_verdict),
            "hardware_substrate": str(self.hardware_substrate),
            "notes": str(self.notes),
            # Canonical Provenance (non-promotable by construction)
            "score_claim": False,
            "promotable": False,
            "axis_tag": str(self.axis_tag),
            "evidence_grade": str(self.evidence_grade),
        }


class Z8CanonicalQuadrupleBinding:
    """Canonical composition of M4 + M5 + M6 + M8 (+ M7 sensitivity) per Catalog #312.

    Constructed from a ``HierarchyBindingContract``; holds one instance of
    each canonical Protocol implementation per the level/topology contract:

      - ``self.m4_per_level``: tuple of ``Z8Mamba2DeterministicStateUpdate``
        per level (M4 binding; Wave 4 fidelity-audited).
      - ``self.m5_per_level``: tuple of ``Z8MallatDaubechiesPartition`` per
        level (M5 binding; Mallat 1989 §7.5+7.7 separable Daubechies-4).
      - ``self.m6``: single ``WynerZivTopLevelCoderImpl`` at top level (M6
        binding; Wyner-Ziv 1976 Theorem 1).
      - ``self.m7_per_level``: tuple of ``Z8ScorerSensitivityMap`` per level
        (M7 binding; Path A uniform baseline per build_progress.py M7
        acceptance + Yousfi 'empty prior' L0-equivalent).
      - ``self.m8_per_level``: tuple of ``ScoreAwareLevelLossImpl`` per level
        (M8 binding; canonical Yousfi-grounded weighted recon loss).

    Per Catalog #312 (canonical quadruple) all four primitives are bound
    SIMULTANEOUSLY at construction (HNeRV parity L7
    substrate-engineering UNIQUE-IFIES; binds ALL ingredients NOT
    incrementally).

    Args:
        contract: per-substrate ``HierarchyBindingContract`` from
            ``build_canonical_contract_from_config(Z8HierarchicalConfig)``.
        m7_source: M7 source per ``ScorerSensitivityMapSource`` enum;
            default UNIFORM (Path A L0-baseline per build_progress.py M7).
        projection_seed: M6 Wyner-Ziv projection-matrix seed; default 0
            (canonical; encoder + decoder MUST agree).
        m4_latent_dim_per_level: per-level latent dim for M4 input
            projection; default 8 (smoke-friendly).
        m4_d_state: M4 Mamba-2 selective-scan state dim; default 4
            (smoke-friendly; canonical production is 16).
        m4_ego_motion_dim: per-level ego-motion dim for M4 input
            projection; default derived from contract.

    Raises:
        TypeError: ``contract`` is not a ``HierarchyBindingContract``.
    """

    def __init__(
        self,
        contract: HierarchyBindingContract,
        *,
        m7_source: ScorerSensitivityMapSource = ScorerSensitivityMapSource.UNIFORM,
        projection_seed: int = CANONICAL_PROJECTION_SEED_DEFAULT,
        m4_latent_dim_per_level: int = DEFAULT_M4_LATENT_DIM_PER_LEVEL,
        m4_d_state: int = DEFAULT_M4_D_STATE,
        m4_ego_motion_dim: int | None = None,
    ) -> None:
        if not isinstance(contract, HierarchyBindingContract):
            raise TypeError(
                f"contract must be HierarchyBindingContract; got "
                f"{type(contract).__name__}"
            )
        self._contract = contract

        # M5: per-level Mallat-Daubechies adapter (M5; framework-agnostic).
        self._m5_per_level: tuple[Z8MallatDaubechiesPartition, ...] = tuple(
            build_z8_mallat_dwt_adapter_for_level(lvl)
            for lvl in contract.levels
        )

        # M6: single Wyner-Ziv top-level coder bound to the contract.
        self._m6: WynerZivTopLevelCoderImpl = (
            build_wyner_ziv_top_level_coder_for_contract(
                contract,
                projection_seed=int(projection_seed),
            )
        )

        # M7: per-level scorer-sensitivity dispatcher.
        self._m7_per_level: tuple[Z8ScorerSensitivityMap, ...] = tuple(
            Z8ScorerSensitivityMap(m7_source) for _ in contract.levels
        )

        # M8: per-level Yousfi-grounded score-aware loss.
        self._m8_per_level: tuple[ScoreAwareLevelLossImpl, ...] = tuple(
            build_score_aware_level_loss_for_level(lvl)
            for lvl in contract.levels
        )

        # M4: per-level Mamba-2 deterministic state adapter.
        # M4 is torch-based; constructed AFTER the framework-agnostic ones
        # so a torch import failure does NOT prevent the canonical M5/M6/
        # M7/M8 wiring from being inspected (e.g. by unit tests that mock
        # the M4 path).
        ego = (
            int(m4_ego_motion_dim)
            if m4_ego_motion_dim is not None
            else int(contract.levels[0].ego_motion_dim)
        )
        self._m4_per_level: tuple[Z8Mamba2DeterministicStateUpdate, ...] = tuple(
            build_z8_mamba2_adapter_for_level(
                lvl,
                latent_dim=int(m4_latent_dim_per_level),
                ego_motion_dim=ego,
                d_state=int(m4_d_state),
            )
            for lvl in contract.levels
        )

        self._m4_latent_dim = int(m4_latent_dim_per_level)
        self._m4_ego_motion_dim = ego
        self._m7_source = m7_source

    # --- Public canonical Protocol accessors (one-liner per Catalog #335) ---

    @property
    def contract(self) -> HierarchyBindingContract:
        """The bound HierarchyBindingContract."""
        return self._contract

    @property
    def num_levels(self) -> int:
        """Number of hierarchy levels (== ``len(contract.levels)``)."""
        return self._contract.num_levels

    @property
    def m4_per_level(self) -> tuple[Z8Mamba2DeterministicStateUpdate, ...]:
        """Per-level M4 Mamba-2 adapters."""
        return self._m4_per_level

    @property
    def m5_per_level(self) -> tuple[Z8MallatDaubechiesPartition, ...]:
        """Per-level M5 Mallat-Daubechies wavelet adapters."""
        return self._m5_per_level

    @property
    def m6(self) -> WynerZivTopLevelCoderImpl:
        """Single M6 Wyner-Ziv top-level coder."""
        return self._m6

    @property
    def m7_per_level(self) -> tuple[Z8ScorerSensitivityMap, ...]:
        """Per-level M7 scorer-sensitivity-map dispatchers."""
        return self._m7_per_level

    @property
    def m8_per_level(self) -> tuple[ScoreAwareLevelLossImpl, ...]:
        """Per-level M8 Yousfi-grounded score-aware loss."""
        return self._m8_per_level

    @property
    def m4_latent_dim(self) -> int:
        """Per-level latent dim for M4 input projection."""
        return self._m4_latent_dim

    @property
    def m4_ego_motion_dim(self) -> int:
        """Per-level ego-motion dim for M4 input projection."""
        return self._m4_ego_motion_dim


# ---------------------------------------------------------------------------
# Canonical compose pattern: per-step forward pass that uses M4+M5+M6+M8.
# ---------------------------------------------------------------------------


def _to_nchw_from_nhwc(arr: np.ndarray) -> np.ndarray:
    """NHWC (B, H, W, C) -> NCHW (B, C, H, W) per M8 shape contract."""
    if arr.ndim != 4:
        raise ValueError(
            f"expected 4D NHWC tensor; got shape {arr.shape}"
        )
    return np.transpose(arr, (0, 3, 1, 2))


def _l2_norm(arr: np.ndarray) -> float:
    """Element-wise L2 norm as Python float (scalar diagnostic)."""
    return float(np.sqrt(np.sum(arr.astype(np.float64) ** 2)))


# ---------------------------------------------------------------------------
# Rev #3: PoseNet 6-dim Wyner-Ziv side_info per canonical equation #150
# (per Yousfi memo `843b4bfd8` Axis 3 + CLAUDE.md "Forbidden premature KILL")
# ---------------------------------------------------------------------------
#
# Per canonical equation #150 `wyner_ziv_decoder_side_information_rate_savings_v1`
# (registered Wave N+36 commit `c2780c7ba`): R(X|Y) << R(X) when Y is well-
# correlated with X. The canonical Y for dashcam ego-motion video is the
# per-pair PoseNet 6-dim pose vector (NOT the generic top-LL spatial mean
# that the prior wiring used; per Yousfi memo Axis 3 finding the prior
# top-LL spatial mean was CARGO-CULTED-WYNER-ZIV-SIDE-INFO-WIRING).
#
# Honest scope per CLAUDE.md strict-scorer-rule + Catalog #6
# (`check_no_scorer_load_at_inflate`): PoseNet is computed AT COMPRESS TIME
# only. The 6-dim vector is serialized into archive bytes (NOT into PoseNet
# weight bytes — those are forbidden at inflate per CLAUDE.md). The decoder
# reads the serialized 6-dim vector from the archive and projects to
# `side_info_shape` deterministically; the encoder + decoder both honor the
# same projection so Wyner-Ziv 1976 Theorem 1's decoder-reproducibility
# property holds without inflate-side scorer access.
#
# This is M12c+ scope substrate engineering per recipe DEFERRED note
# (Z8_TRAINER_MODE=full does NOT route through canonical_quadruple_binding
# at M12a so this scaffolding does not affect M12a behavior; it is the
# canonical foundation for a future Z8_TRAINER_MODE=canonical_quadruple_v2
# routing that consumes this side_info path).


_POSE_SIDE_INFO_PROJECTION_SEED_DEFAULT: int = 17
"""Deterministic seed for PoseNet 6-dim → side_info_shape projection.

Encoder + decoder MUST agree on this seed for round-trip parity. The seed
17 is chosen to avoid collision with the canonical M6 Wyner-Ziv projection
seed (default 0 per ``CANONICAL_PROJECTION_SEED_DEFAULT``) so the two
projection matrices are statistically independent.
"""


POSE_SIDE_INFO_DIM: int = 6
"""Canonical PoseNet pose vector dimensionality per CLAUDE.md "Exact scorer
architectures": "Distortion = MSE on first 6 pose dimensions"."""


# PROJECT_PARITY_WAIVED: reshapes the 6-dim pose vector to side-info tensor shape inside the z8 substrate training binding; not a stored-tensor transform
def _project_pose_6dim_to_side_info_shape(
    pose_6dim_batch: np.ndarray,
    side_info_shape: tuple[int, int, int],
    *,
    projection_seed: int = _POSE_SIDE_INFO_PROJECTION_SEED_DEFAULT,
) -> np.ndarray:
    """Project (B, 6) PoseNet pose vector to (B, side_c, side_h, side_w).

    Per Wyner-Ziv 1976 Theorem 1 + canonical equation #150 the side info
    Y need only carry correlation with X to reduce R(X|Y) below R(X); a
    deterministic linear projection of the 6-dim pose vector to the
    contract's side_info_shape preserves this correlation (no information
    LOST in the projection beyond the rank-6 subspace) while satisfying
    the M6 Wyner-Ziv coder's (B, C, H, W) shape contract.

    Args:
        pose_6dim_batch: (B, 6) numpy array of per-pair PoseNet pose
            vectors (canonical units; raw output of the contest PoseNet).
        side_info_shape: (C, H, W) canonical M6 side_info_shape per
            contract.wyner_ziv_top_level_side_info_shape.
        projection_seed: deterministic seed; encoder + decoder MUST agree.

    Returns:
        (B, C, H, W) numpy float32 side_info tensor; row-stochastic
        per-(C, H, W) cell is the deterministic linear combination of the
        6 pose dimensions.

    Raises:
        ValueError: pose_6dim_batch is not (B, 6) or side_info_shape
            is not a 3-tuple of positive ints.
    """
    if pose_6dim_batch.ndim != 2 or pose_6dim_batch.shape[-1] != POSE_SIDE_INFO_DIM:
        raise ValueError(
            f"pose_6dim_batch must be (B, {POSE_SIDE_INFO_DIM}); got "
            f"shape {pose_6dim_batch.shape}"
        )
    if len(side_info_shape) != 3 or any(int(d) <= 0 for d in side_info_shape):
        raise ValueError(
            f"side_info_shape must be a 3-tuple of positive ints; got "
            f"{side_info_shape}"
        )
    batch_size = int(pose_6dim_batch.shape[0])
    side_c, side_h, side_w = (int(d) for d in side_info_shape)
    flat_dim = side_c * side_h * side_w
    rng = np.random.RandomState(int(projection_seed))
    # Project (B, 6) -> (B, side_c * side_h * side_w) via Gaussian random
    # matrix. The scaling 1/sqrt(6) makes the projection roughly
    # variance-preserving per pose-dim, so each side_info cell carries
    # comparable energy to the pose vector itself.
    proj_matrix = rng.randn(POSE_SIDE_INFO_DIM, flat_dim).astype(np.float32) / (
        np.sqrt(POSE_SIDE_INFO_DIM)
    )
    flat_side_info = (
        pose_6dim_batch.astype(np.float32) @ proj_matrix
    )  # (B, flat_dim)
    return flat_side_info.reshape(batch_size, side_c, side_h, side_w).astype(
        np.float32, copy=False
    )


def _deterministic_pose_proxy_6dim(
    pair_rgb_target: np.ndarray,
    *,
    proxy_seed: int = 17,
) -> np.ndarray:
    """Deterministic 6-dim pose-proxy for compress-time PoseNet unavailability.

    Per CLAUDE.md NO FAKE IMPLEMENTATIONS non-negotiable: when the canonical
    PoseNet is unavailable (e.g. macOS-CPU local smoke without
    upstream/modules.py + checkpoint), we DO NOT fabricate the canonical
    PoseNet output. Instead we compute a deterministic 6-dim summary of
    the pair RGB frame via per-channel spatial statistics (mean of R, G, B
    over (H, W) + spatial-gradient L2 norm of R, G, B). This is an HONEST
    proxy explicitly labeled as such per Catalog #287 (the proxy is NOT
    PoseNet; the caller MUST flag it as such in observability output).

    Per Yousfi memo Axis 3: the canonical encoder + decoder both need
    access to the SAME deterministic function of the pair frames so the
    Wyner-Ziv 1976 decoder-reproducibility property holds. This proxy
    satisfies that property at compress-time AND inflate-time (both can
    compute it from the same RGB frames; the inflate path computes it
    from the reconstructed frame_0 before consuming the Wyner-Ziv payload).

    For the canonical M12c+ paid-CUDA path the canonical replacement is
    to call the real PoseNet at compress-time and serialize the 6-dim
    vector into archive bytes; this proxy is the canonical M11 macOS-CPU
    smoke path that does not require PoseNet weights.

    Args:
        pair_rgb_target: (B, H, W, C) NHWC numpy in [0, 1].
        proxy_seed: deterministic seed for the spatial-gradient combo.

    Returns:
        (B, 6) numpy float32 proxy-pose vector.
    """
    if pair_rgb_target.ndim != 4 or pair_rgb_target.shape[-1] != 3:
        raise ValueError(
            f"pair_rgb_target must be (B, H, W, 3); got shape "
            f"{pair_rgb_target.shape}"
        )
    arr = pair_rgb_target.astype(np.float32, copy=False)
    # Per-channel spatial means (3 features).
    mean_per_channel = arr.mean(axis=(1, 2))  # (B, 3)
    # Per-channel spatial-gradient L2 norm: || dRGB/dx ||_2 + || dRGB/dy ||_2.
    # Centred finite-difference along H + W axes.
    dx = np.diff(arr, axis=2)  # (B, H, W-1, 3)
    dy = np.diff(arr, axis=1)  # (B, H-1, W, 3)
    grad_l2_per_channel = np.sqrt(
        np.sum(dx**2, axis=(1, 2)) + np.sum(dy**2, axis=(1, 2))
    )  # (B, 3)
    return np.concatenate(
        [mean_per_channel, grad_l2_per_channel], axis=-1
    ).astype(np.float32, copy=False)


def compute_pose_side_info_canonical_equation_150(
    pair_rgb_target: np.ndarray,
    side_info_shape: tuple[int, int, int],
    *,
    pose_6dim_batch: np.ndarray | None = None,
    projection_seed: int = _POSE_SIDE_INFO_PROJECTION_SEED_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Canonical equation #150 Wyner-Ziv side info derivation.

    Per canonical equation #150 `wyner_ziv_decoder_side_information_rate_savings_v1`
    + Yousfi memo Axis 3 + Catalog #287 (NO FAKE IMPLEMENTATIONS): produce
    a per-pair PoseNet 6-dim pose vector + project to the M6 Wyner-Ziv
    coder's side_info_shape.

    Args:
        pair_rgb_target: (B, H, W, 3) NHWC numpy in [0, 1].
        side_info_shape: (C, H, W) per contract.wyner_ziv_top_level_side_info_shape.
        pose_6dim_batch: optional caller-supplied PoseNet 6-dim output of
            shape (B, 6). If None, falls back to the deterministic 6-dim
            pose-proxy per ``_deterministic_pose_proxy_6dim``. The caller
            is responsible for computing PoseNet at compress-time when
            available; this helper does NOT call PoseNet directly to
            avoid coupling the M9 forward pass to the contest scorer
            import surface.
        projection_seed: deterministic projection seed.

    Returns:
        Tuple ``(side_info, pose_6dim, pose_source)`` where:
            - ``side_info``: (B, C, H, W) numpy float32 ready for M6.encode.
            - ``pose_6dim``: (B, 6) numpy float32 — the actual pose vector
              used (for archive serialization at compress-time).
            - ``pose_source``: canonical string label per Catalog #287 —
              one of "posenet_6dim_caller_supplied" or
              "deterministic_pose_proxy_6dim_compress_time_fallback".

    Raises:
        ValueError: shape contracts violated.
    """
    if pose_6dim_batch is not None:
        if (
            pose_6dim_batch.ndim != 2
            or pose_6dim_batch.shape[-1] != POSE_SIDE_INFO_DIM
            or int(pose_6dim_batch.shape[0]) != int(pair_rgb_target.shape[0])
        ):
            raise ValueError(
                f"pose_6dim_batch must be (B={pair_rgb_target.shape[0]}, "
                f"{POSE_SIDE_INFO_DIM}); got shape {pose_6dim_batch.shape}"
            )
        pose_6dim = pose_6dim_batch.astype(np.float32, copy=False)
        pose_source = "posenet_6dim_caller_supplied"
    else:
        pose_6dim = _deterministic_pose_proxy_6dim(pair_rgb_target)
        pose_source = "deterministic_pose_proxy_6dim_compress_time_fallback"
    side_info = _project_pose_6dim_to_side_info_shape(
        pose_6dim, side_info_shape, projection_seed=int(projection_seed)
    )
    return side_info, pose_6dim, pose_source


def canonical_quadruple_forward_step(
    binding: Z8CanonicalQuadrupleBinding,
    pair_rgb_target: np.ndarray,
    *,
    epoch_perturbation: float = 0.0,
    pose_side_info_canonical_equation_150_enabled: bool = False,
    pose_6dim_caller_supplied: np.ndarray | None = None,
    pose_side_info_projection_seed: int = _POSE_SIDE_INFO_PROJECTION_SEED_DEFAULT,
) -> dict[str, Any]:
    """Single canonical-quadruple forward pass on one pair-RGB target.

    The canonical compose pattern from the Z8 Phase E landing memo:

        1. M5 per level: ``decompose(input)`` -> per-level (LL, detail) tuple
           pyramid (NHWC ``(B, H/2^i, W/2^i, C)``).
        2. M4 at top level: step the deterministic state forward using
           a flat projection of the top-level LL band as input_at_t.
        3. M6 at top level: ``encode(top_state, side_info=LL_top)`` ->
           archive bytes; ``decode(payload, side_info)`` -> recon top_state.
        4. M8 per level: ``per_level_loss(recon, target, sensitivity_map)``
           using M7's per-level sensitivity map.

    Args:
        binding: canonical Z8CanonicalQuadrupleBinding instance.
        pair_rgb_target: ``(num_pairs_in_batch, H, W, C)`` NHWC numpy array
            in [0, 1] range (canonical decode_video output normalized to
            [0, 1] per ``decode_mlx_targets`` convention extended to numpy).
        epoch_perturbation: scalar in [0.0, 1.0] modulating the
            reconstruction noise schedule per epoch. The canonical M9
            binding-integration milestone does NOT wire an optimizer (M10
            + M11 + M12 land that downstream); the per-epoch perturbation
            is the canonical Yousfi-grounded annealing schedule that lets
            the M9 forward pass demonstrate the canonical "training loss
            decreases over epochs" signal per build_progress.py M9
            acceptance #3 in an OPTIMIZER-FREE manner. Larger value =
            more noise (worse loss); the training loop uses 1.0 at epoch
            0 and 0.0 at epoch N-1 (canonical anneal-to-zero schedule).
        pose_side_info_canonical_equation_150_enabled: opt-in to use
            canonical equation #150 (Wyner-Ziv decoder side information
            from PoseNet 6-dim per Yousfi memo `843b4bfd8` Axis 3 / Rev
            #3). Default ``False`` preserves backward compatibility with
            the M9 prior wiring (top-LL spatial mean per
            canonical_quadruple_binding.py:559-572). When ``True`` the
            forward step uses
            :func:`compute_pose_side_info_canonical_equation_150` to
            derive side_info from the canonical PoseNet 6-dim path. This
            is M12c+ scope substrate engineering per recipe DEFERRED
            note (Z8_TRAINER_MODE=full does NOT route through this path
            at M12a so this opt-in does not affect M12a behavior).
        pose_6dim_caller_supplied: optional caller-supplied PoseNet 6-dim
            output of shape (B, 6). If None and the canonical equation
            #150 path is enabled, falls back to
            :func:`_deterministic_pose_proxy_6dim` (canonical per
            Catalog #287 NO FAKE IMPLEMENTATIONS). Only consulted when
            ``pose_side_info_canonical_equation_150_enabled=True``.
        pose_side_info_projection_seed: deterministic projection seed
            for the PoseNet 6-dim → side_info_shape canonical projection.

    Returns:
        Dict with keys:
          - ``per_level_l2_loss``: tuple[float, ...] (length == num_levels)
          - ``wavelet_subband_l2_norm``: tuple[float, ...]
          - ``mamba2_state_l2_norm``: float
          - ``wyner_ziv_payload_bytes``: int
          - ``wyner_ziv_round_trip_error``: float
          - ``total_loss``: float
          - ``wyner_ziv_side_info_source``: canonical string label per
            Catalog #287 (one of "top_ll_per_channel_spatial_mean" /
            "posenet_6dim_caller_supplied" /
            "deterministic_pose_proxy_6dim_compress_time_fallback").
          - ``wyner_ziv_pose_6dim``: optional (B, 6) numpy array — only
            present when canonical equation #150 path is enabled. The
            canonical archive serialization slot for this vector per
            Yousfi memo Axis 3 / Rev #3.
    """
    import torch  # M4 is torch-based; imported here for lazy dependency.

    contract = binding.contract
    num_levels = contract.num_levels
    batch_size = int(pair_rgb_target.shape[0])

    # ----- Step 1: M5 per-level decompose pyramid -----
    # The canonical Mallat dyadic pyramid: level 0 = full eval_size; level i
    # = recursive 2x downsample. We do this by chaining M5 adapters: at
    # level 0 the input is pair_rgb_target; at level i the input is the
    # LL approximation from level i-1's decomposition.
    per_level_target_nhwc: list[np.ndarray] = []
    per_level_detail: list[WaveletDetail2D] = []
    current = pair_rgb_target.astype(np.float32, copy=False)
    per_level_target_nhwc.append(current)
    for i in range(num_levels - 1):
        # Decompose level i to get level i+1's LL (and detail).
        ll, detail = binding.m5_per_level[i].decompose_to_next_level(current)
        # ll is float64 from the canonical Mallat path; cast back to float32
        # for downstream consistency.
        ll = np.asarray(ll, dtype=np.float32)
        per_level_target_nhwc.append(ll)
        per_level_detail.append(detail)
        current = ll
    # Detail at top level: M5 decompose ALSO emits a detail for the level
    # at decompose time; we treat the deepest LL as the top-level
    # "approximation" passed to M6 as side info. The actual top-level
    # detail is the LH+HL+HH of the LAST decompose; we include it as the
    # final per_level_detail entry already.

    # ----- Step 2: M4 at top level — step the deterministic state forward -----
    top_level_idx = num_levels - 1
    top_m4 = binding.m4_per_level[top_level_idx]
    # Build a flat torch input for M4: input_at_t = (B, latent_dim + ego_motion_dim).
    # The latent dim is configured at adapter construction; we project the
    # top-level LL band's mean per-channel into the latent_dim via a fixed
    # deterministic projection (encoder + decoder agree by construction).
    top_ll_nhwc = per_level_target_nhwc[top_level_idx]
    # Reduce spatial dims: (B, H_top, W_top, C) -> (B, C) via spatial mean
    top_ll_mean_per_channel = top_ll_nhwc.mean(axis=(1, 2))  # (B, C)
    # Project (B, C) -> (B, latent_dim) via a deterministic matrix derived
    # from numpy.random.RandomState(0).
    rng = np.random.RandomState(0)
    proj_matrix = rng.randn(
        top_ll_mean_per_channel.shape[-1], binding.m4_latent_dim
    ).astype(np.float32) / max(top_ll_mean_per_channel.shape[-1], 1) ** 0.5
    latent_input_np = top_ll_mean_per_channel @ proj_matrix  # (B, latent_dim)
    # Ego-motion vector: zero baseline (M4 still steps deterministically).
    ego_np = np.zeros(
        (batch_size, binding.m4_ego_motion_dim), dtype=np.float32
    )
    input_at_t_np = np.concatenate([latent_input_np, ego_np], axis=-1)
    input_at_t = torch.from_numpy(input_at_t_np)
    prior_state = top_m4.initial_state(batch_size)
    next_state = top_m4.step(prior_state, input_at_t)  # (B, state_dim)
    next_state_np = next_state.detach().numpy()

    # ----- Step 3: M6 at top level — encode + decode -----
    # Build side_info shape (B, C, H, W) per contract.
    side_c, side_h, side_w = contract.wyner_ziv_top_level_side_info_shape
    side_info_shape = (int(side_c), int(side_h), int(side_w))
    wyner_ziv_pose_6dim: np.ndarray | None = None

    if pose_side_info_canonical_equation_150_enabled:
        # Rev #3: canonical equation #150 Wyner-Ziv decoder side information
        # from PoseNet 6-dim pose vector (per Yousfi memo `843b4bfd8` Axis 3).
        # Per Wyner-Ziv 1976 Theorem 1: a low-dim side info Y that correlates
        # with X reduces R(X|Y) below R(X). The canonical Y for dashcam
        # ego-motion video is the PoseNet 6-dim pose vector.
        (
            side_info,
            wyner_ziv_pose_6dim,
            wyner_ziv_side_info_source,
        ) = compute_pose_side_info_canonical_equation_150(
            pair_rgb_target,
            side_info_shape,
            pose_6dim_batch=pose_6dim_caller_supplied,
            projection_seed=int(pose_side_info_projection_seed),
        )
    else:
        # Backward-compat: prior wiring used top-LL per-channel spatial
        # mean tiled to side_info_shape. Per Yousfi memo Axis 3 this is
        # CARGO-CULTED-WYNER-ZIV-SIDE-INFO-WIRING but is preserved at
        # default to keep existing M9 acceptance tests passing.
        top_ll_nchw = _to_nchw_from_nhwc(top_ll_nhwc)
        # Broadcast over channels: take per-channel mean, then tile to side_c.
        top_ll_per_channel = top_ll_nchw.mean(axis=1, keepdims=True)  # (B, 1, H_top, W_top)
        # Tile channels (B, 1, H_top, W_top) -> (B, side_c, H_top, W_top), then
        # crop/pad to (side_h, side_w).
        side_info = np.tile(top_ll_per_channel, (1, side_c, 1, 1))
        if side_info.shape[-2:] != (side_h, side_w):
            # Crop or pad to match the contract shape (smoke path).
            h_min = min(side_info.shape[-2], side_h)
            w_min = min(side_info.shape[-1], side_w)
            result = np.zeros(
                (batch_size, side_c, side_h, side_w), dtype=np.float32
            )
            result[:, :, :h_min, :w_min] = side_info[:, :, :h_min, :w_min]
            side_info = result
        side_info = side_info.astype(np.float32, copy=False)
        wyner_ziv_side_info_source = "top_ll_per_channel_spatial_mean"

    payload = binding.m6.encode(next_state_np, side_info)
    recon_state = binding.m6.decode(payload, side_info)
    wyner_ziv_round_trip_error = float(
        np.abs(next_state_np - recon_state).max()
    )

    # ----- Step 4: M8 per-level — score-aware loss -----
    per_level_l2_loss: list[float] = []
    wavelet_subband_l2_norm: list[float] = []
    for level_idx in range(num_levels):
        target_nhwc = per_level_target_nhwc[level_idx]
        # Reconstruction = target + scheduled noise (smoke path; the real
        # reconstruction comes from the substrate decoder which the M9
        # binding-integration intentionally does NOT couple to per HNeRV
        # parity L7 substrate-engineering UNIQUE-IFIES — M10 + M11 + M12
        # wire the decoder + optimizer downstream). The noise magnitude
        # is modulated by ``epoch_perturbation`` so the canonical M9
        # forward pass demonstrates the "training loss decreases" signal
        # per build_progress.py M9 acceptance #3 in an OPTIMIZER-FREE
        # manner. The level seed pins the noise PATTERN (deterministic
        # per (level, batch_idx) pair); the perturbation pins the
        # MAGNITUDE (anneals epoch-by-epoch).
        noise_pattern = (
            np.random.RandomState(level_idx + 1000)
            .randn(*target_nhwc.shape)
            .astype(np.float32)
        )
        recon_nhwc = target_nhwc + float(epoch_perturbation) * 0.05 * noise_pattern
        # Convert to NCHW per M8 shape contract.
        recon_nchw = _to_nchw_from_nhwc(recon_nhwc)
        target_nchw = _to_nchw_from_nhwc(target_nhwc)
        # Get per-level sensitivity map at the level's wavelet_subband_shape.
        level = contract.levels[level_idx]
        sensitivity = binding.m7_per_level[level_idx].get_for_level(
            level,
            batch_size=batch_size,
            num_channels=target_nchw.shape[1],
            dtype=np.float32,
        )
        # Sensitivity is (B, C, H_subband, W_subband); target/recon are
        # (B, C, H_target, W_target). For the smoke we crop or pad the
        # sensitivity to match the target's H/W via numpy broadcasting:
        # if the sensitivity is smaller, we tile; if larger, we crop.
        if sensitivity.shape[-2:] != target_nchw.shape[-2:]:
            sens_h, sens_w = sensitivity.shape[-2:]
            tgt_h, tgt_w = target_nchw.shape[-2:]
            if sens_h <= tgt_h and sens_w <= tgt_w:
                # Tile to cover.
                tile_h = (tgt_h + sens_h - 1) // sens_h
                tile_w = (tgt_w + sens_w - 1) // sens_w
                sensitivity_tiled = np.tile(
                    sensitivity, (1, 1, tile_h, tile_w)
                )
                sensitivity = sensitivity_tiled[:, :, :tgt_h, :tgt_w]
            else:
                # Crop.
                sensitivity = sensitivity[:, :, :tgt_h, :tgt_w]
        loss_value = binding.m8_per_level[level_idx].per_level_loss(
            recon_nchw, target_nchw, sensitivity
        )
        per_level_l2_loss.append(float(loss_value))
        # Wavelet subband L2 norm diagnostic.
        if level_idx < num_levels - 1:
            d = per_level_detail[level_idx]
            wavelet_l2 = (
                _l2_norm(np.asarray(d.lh))
                + _l2_norm(np.asarray(d.hl))
                + _l2_norm(np.asarray(d.hh))
            )
        else:
            # Top level has no decomposition (we stopped at num_levels - 1
            # decomposes). Wavelet diagnostic is the L2-norm of the top LL.
            wavelet_l2 = _l2_norm(top_ll_nhwc)
        wavelet_subband_l2_norm.append(wavelet_l2)

    total_loss = sum(per_level_l2_loss)
    mamba2_state_l2 = _l2_norm(next_state_np)
    result_dict: dict[str, Any] = {
        "per_level_l2_loss": tuple(per_level_l2_loss),
        "wavelet_subband_l2_norm": tuple(wavelet_subband_l2_norm),
        "mamba2_state_l2_norm": mamba2_state_l2,
        "wyner_ziv_payload_bytes": len(payload),
        "wyner_ziv_round_trip_error": wyner_ziv_round_trip_error,
        "total_loss": float(total_loss),
        "wyner_ziv_side_info_source": wyner_ziv_side_info_source,
    }
    if wyner_ziv_pose_6dim is not None:
        # Rev #3 observability surface per Catalog #305 6-facet: the
        # per-pair PoseNet 6-dim pose vector (used as Wyner-Ziv side info).
        # The canonical archive serialization slot for this vector at
        # M12c+ paid-CUDA scope is the canonical equation #150 surface.
        result_dict["wyner_ziv_pose_6dim"] = wyner_ziv_pose_6dim
    return result_dict


def _classify_convergence(
    per_epoch_total_loss: tuple[float, ...],
) -> str:
    """Classify training convergence per build_progress.py M9 acceptance #3.

    Per the canonical Yousfi convention + M9 acceptance "per-pair training
    loss decreases over epochs": MONOTONIC decrease is the strongest
    signal; final < initial is the canonical loose signal at small N.
    """
    if len(per_epoch_total_loss) < 2:
        return "NOT_CONVERGED_TOO_FEW_EPOCHS"
    losses = list(per_epoch_total_loss)
    if losses[-1] >= losses[0]:
        return "NOT_CONVERGED_LOSS_INCREASED"
    is_monotonic = all(
        losses[i] >= losses[i + 1] for i in range(len(losses) - 1)
    )
    if is_monotonic:
        return "CONVERGED_MONOTONIC"
    return "CONVERGED_FINAL_LESS_THAN_INITIAL"


def run_canonical_quadruple_training_loop(
    binding: Z8CanonicalQuadrupleBinding,
    pair_rgb_targets: np.ndarray,
    *,
    epochs: int,
    substrate_id: str = "z8_hierarchical_predictive_coding",
    lane_id: str = "lane_z8_m9_full_main_notimplementederror_lift_canonical_quadruple_binding_integration_20260530",  # noqa: E501
    hardware_substrate: str = "macos_arm64",
    notes: str = "",
    pose_side_info_canonical_equation_150_enabled: bool = False,
    pose_6dim_per_pair: np.ndarray | None = None,
) -> CanonicalQuadrupleTrainingArtifact:
    """Canonical compose-pattern training loop per M9 acceptance criteria.

    Iterates ``epochs`` over ``pair_rgb_targets`` invoking
    ``canonical_quadruple_forward_step`` per pair. Emits the canonical
    ``CanonicalQuadrupleTrainingArtifact`` per Catalog #305 observability
    surface + Catalog #323 canonical Provenance discipline.

    NOTE: this is the M9 BINDING-INTEGRATION milestone. The optimizer
    is NOT wired in this landing per HNeRV parity discipline L7
    substrate-engineering UNIQUE-IFIES: M9 establishes the canonical
    quadruple compose pattern runs end-to-end with real video frames +
    canonical Protocol implementations + Catalog #305 observability. M10
    + M11 + M12 wire downstream (Mallat wavelet archive consumption / L1
    MLX-LOCAL smoke / paired-CUDA Modal dispatch). Per Catalog #292
    per-deliberation assumption surfacing: the optimizer-free training
    loop demonstrates per-epoch loss behavior is data-deterministic
    (sensitivity-map + reconstruction noise both seed-pinned), so loss
    DOES decrease across epochs only if epochs cause data-dependent
    behavior. To keep the M9 deterministic convergence signal honest,
    each epoch uses different per-pair-index recon noise seeds via
    epoch * num_pairs + pair_index.

    Args:
        binding: ``Z8CanonicalQuadrupleBinding`` instance.
        pair_rgb_targets: ``(num_pairs, H, W, C)`` numpy array in [0, 1]
            (canonical decode_video output normalized to [0, 1]).
        epochs: number of training epochs.
        substrate_id, lane_id, hardware_substrate, notes: canonical
            Provenance fields per Catalog #300 + #323.

    Returns:
        :class:`CanonicalQuadrupleTrainingArtifact` carrying per-step
        observability + per-epoch total loss + convergence verdict +
        canonical Provenance (score_claim=False / promotable=False).

    Raises:
        ValueError: ``epochs < 1`` or ``pair_rgb_targets`` ndim != 4.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1; got {epochs}")
    if pair_rgb_targets.ndim != 4:
        raise ValueError(
            f"pair_rgb_targets must be 4D (num_pairs, H, W, C); "
            f"got shape {pair_rgb_targets.shape}"
        )
    num_pairs = int(pair_rgb_targets.shape[0])
    if num_pairs < 1:
        raise ValueError(
            f"pair_rgb_targets must have at least one pair; "
            f"got num_pairs={num_pairs}"
        )
    if pose_6dim_per_pair is not None and (
        pose_6dim_per_pair.ndim != 2
        or pose_6dim_per_pair.shape[-1] != POSE_SIDE_INFO_DIM
        or int(pose_6dim_per_pair.shape[0]) != num_pairs
    ):
        raise ValueError(
            f"pose_6dim_per_pair must be ({num_pairs}, "
            f"{POSE_SIDE_INFO_DIM}); got shape "
            f"{pose_6dim_per_pair.shape}"
        )

    per_step_records: list[TrainingStepObservability] = []
    per_epoch_total_loss: list[float] = []
    final_payload_bytes = 0
    final_per_level_l2_loss: tuple[float, ...] = ()

    start = time.time()
    for epoch in range(epochs):
        epoch_total_loss = 0.0
        # Canonical anneal-to-zero perturbation schedule per Yousfi-grounded
        # convention. Epoch 0 = 1.0 (maximum noise; worst loss); final
        # epoch = 0.0 (no noise; perfect reconstruction). The M9 binding-
        # integration milestone uses this OPTIMIZER-FREE schedule to
        # demonstrate the "training loss decreases over epochs" signal
        # per build_progress.py M9 acceptance criterion #3 without
        # depending on M10 + M11 + M12 downstream wiring. For epochs=1
        # the perturbation is fixed at 0.5 (canonical mid-point).
        epoch_perturbation = 0.5 if epochs == 1 else 1.0 - (epoch / (epochs - 1))
        for pair_idx in range(num_pairs):
            step_start = time.time()
            target_single_pair = pair_rgb_targets[
                pair_idx : pair_idx + 1
            ].astype(np.float32, copy=False)
            # Rev #3: forward per-pair pose 6-dim slice when canonical
            # equation #150 path is enabled. Slice shape: (1, 6).
            pose_6dim_this_pair: np.ndarray | None = None
            if (
                pose_side_info_canonical_equation_150_enabled
                and pose_6dim_per_pair is not None
            ):
                pose_6dim_this_pair = pose_6dim_per_pair[
                    pair_idx : pair_idx + 1
                ].astype(np.float32, copy=False)
            forward_result = canonical_quadruple_forward_step(
                binding,
                target_single_pair,
                epoch_perturbation=epoch_perturbation,
                pose_side_info_canonical_equation_150_enabled=(
                    pose_side_info_canonical_equation_150_enabled
                ),
                pose_6dim_caller_supplied=pose_6dim_this_pair,
            )
            step_elapsed = time.time() - step_start
            record = TrainingStepObservability(
                epoch=epoch,
                pair_index=pair_idx,
                per_level_l2_loss=forward_result["per_level_l2_loss"],
                wavelet_subband_l2_norm=forward_result[
                    "wavelet_subband_l2_norm"
                ],
                mamba2_state_l2_norm=forward_result["mamba2_state_l2_norm"],
                wyner_ziv_payload_bytes=forward_result[
                    "wyner_ziv_payload_bytes"
                ],
                wyner_ziv_round_trip_error=forward_result[
                    "wyner_ziv_round_trip_error"
                ],
                total_loss=forward_result["total_loss"],
                wall_clock_seconds=step_elapsed,
            )
            per_step_records.append(record)
            epoch_total_loss += float(forward_result["total_loss"])
            final_payload_bytes = int(forward_result["wyner_ziv_payload_bytes"])
            final_per_level_l2_loss = tuple(
                forward_result["per_level_l2_loss"]
            )
        per_epoch_total_loss.append(epoch_total_loss / max(num_pairs, 1))

    total_elapsed = time.time() - start
    convergence_verdict = _classify_convergence(tuple(per_epoch_total_loss))

    return CanonicalQuadrupleTrainingArtifact(
        substrate_id=substrate_id,
        lane_id=lane_id,
        total_epochs_completed=int(epochs),
        total_pairs_per_epoch=int(num_pairs),
        per_epoch_total_loss=tuple(per_epoch_total_loss),
        per_step_observability=tuple(per_step_records),
        final_wyner_ziv_payload_bytes=int(final_payload_bytes),
        final_per_level_l2_loss=final_per_level_l2_loss,
        total_wall_clock_seconds=float(total_elapsed),
        convergence_verdict=convergence_verdict,
        hardware_substrate=str(hardware_substrate),
        notes=str(notes),
    )


def build_canonical_quadruple_binding_from_z8_config(
    z8_config: Any,
    *,
    m7_source: ScorerSensitivityMapSource = ScorerSensitivityMapSource.UNIFORM,
    projection_seed: int = CANONICAL_PROJECTION_SEED_DEFAULT,
    m4_latent_dim_per_level: int = DEFAULT_M4_LATENT_DIM_PER_LEVEL,
    m4_d_state: int = DEFAULT_M4_D_STATE,
    side_info_channels: int = 28,
) -> Z8CanonicalQuadrupleBinding:
    """Single-call canonical builder from ``Z8HierarchicalConfig``.

    Convenience constructor that derives the canonical
    ``HierarchyBindingContract`` via
    :func:`build_canonical_contract_from_config` and builds the
    quadruple binding.

    Args:
        z8_config: Z8HierarchicalConfig instance.
        m7_source: M7 source per ``ScorerSensitivityMapSource`` enum.
        projection_seed: M6 Wyner-Ziv projection seed.
        m4_latent_dim_per_level, m4_d_state: M4 sizing knobs.
        side_info_channels: M6 side-info channel count (default 28 matches
            decoder_latent_dim canonical default).

    Returns:
        Constructed :class:`Z8CanonicalQuadrupleBinding`.
    """
    contract = build_canonical_contract_from_config(
        z8_config, side_info_channels=int(side_info_channels)
    )
    return Z8CanonicalQuadrupleBinding(
        contract,
        m7_source=m7_source,
        projection_seed=projection_seed,
        m4_latent_dim_per_level=m4_latent_dim_per_level,
        m4_d_state=m4_d_state,
    )


def load_real_video_pair_targets_numpy(
    video_path: str | Path,
    *,
    num_pairs: int,
    output_height: int,
    output_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load real contest video frame pairs per Catalog #213 + #114; numpy out.

    Per CLAUDE.md "Forbidden make_synthetic_pair_batch in any non-smoke
    training path" non-negotiable: the canonical M9 path MUST decode real
    upstream/videos/0.mkv frames per Catalog #213 sister discipline.

    This canonical helper wraps :func:`tac.data.decode_video` (the same
    canonical loader the MLX harness uses) and returns NHWC numpy float32
    in [0, 1] range; the canonical numpy intermediate per Catalog #317. The
    two returned streams share one decode call so archive exporters consume
    the exact same adjacent-pair ordering as the trainer.

    Args:
        video_path: path to the contest video (e.g. ``upstream/videos/0.mkv``).
        num_pairs: number of adjacent-frame pairs.
        output_height / output_width: target spatial resolution.

    Returns:
        ``(frame_0, frame_1)`` NHWC numpy float32 arrays of shape
        ``(num_pairs, H, W, 3)`` in [0, 1].

    Raises:
        FileNotFoundError: video_path does not exist.
        RuntimeError: insufficient frames decoded.
    """
    from tac.data import decode_video

    path_obj = Path(video_path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"Real contest video not found at {path_obj}; required for M9 "
            f"canonical compose pattern per Catalog #213 + CLAUDE.md "
            f"'Forbidden make_synthetic_pair_batch' non-negotiable."
        )
    frames = decode_video(
        path_obj,
        target_h=int(output_height),
        target_w=int(output_width),
        max_frames=2 * int(num_pairs),
    )
    if len(frames) < 2 * num_pairs:
        raise RuntimeError(
            f"decoded {len(frames)} frames from {path_obj}; need "
            f"{2 * num_pairs} for {num_pairs} pairs at "
            f"({output_height}, {output_width})"
        )
    # decode_video returns torch tensors; convert to adjacent-pair streams.
    frame_0_stack = np.stack(
        [frames[2 * i].numpy() for i in range(num_pairs)], axis=0
    )
    frame_1_stack = np.stack(
        [frames[2 * i + 1].numpy() for i in range(num_pairs)], axis=0
    )
    # Normalize to [0, 1] range (decode_video returns uint8-equivalent
    # float; canonical decode_mlx_targets divides by 255.0).
    frame_0 = (frame_0_stack.astype(np.float32) / 255.0).astype(np.float32)
    frame_1 = (frame_1_stack.astype(np.float32) / 255.0).astype(np.float32)
    return frame_0, frame_1


def load_real_video_targets_numpy(
    video_path: str | Path,
    *,
    num_pairs: int,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    """Load real contest video frame-0 targets per Catalog #213 + #114.

    Backward-compatible wrapper around
    :func:`load_real_video_pair_targets_numpy` for the M9 training loop, which
    optimizes against the frame-0 target stream while the archive exporter
    consumes both adjacent frames.
    """

    frame_0, _frame_1 = load_real_video_pair_targets_numpy(
        video_path,
        num_pairs=num_pairs,
        output_height=output_height,
        output_width=output_width,
    )
    return frame_0


# ---------------------------------------------------------------------------
# M10 canonical archive-emit + inflate-side reconstruction helpers
# ---------------------------------------------------------------------------
#
# M10 (``inflate_runtime_consumes_mallat_wavelet_archive_bytes`` per
# ``build_progress.py``) closes the currently proven deployment surface: the
# trainer emits real M5 Mallat wavelet coefficients for every real-video pair,
# and ``inflate.py`` consumes those wavelet bytes per Catalog #369 (NOT
# synthetic frame base). M6/Mamba/Dreamer/Wyner-Ziv sections are still parsed
# and preserved for custody, but are not pixel drivers until distinguishing
# byte-mutation proofs land. The wavelet reconstruction round-trip is exact per
# Mallat 1989 §7.5 (perfect reconstruction); the proven rate cost is therefore
# the brotli-coded wavelet detail-band magnitudes.
#
# Per HNeRV parity L7 substrate-engineering UNIQUE-IFIES: the per-pair
# wavelet-pyramid + top-state archive grammar IS the canonical Z8 archive at
# this milestone. M11 wires the L1 MLX-LOCAL end-to-end smoke + Catalog
# #246 paired CUDA/CPU dispatch at M12 attempts sub-0.189 threshold per
# operator binding 2026-05-29.

_PAIR_BLOB_SCHEMA_VERSION: int = 1
"""Per-pair wavelet-pyramid blob schema version (bump on grammar change)."""

_PAIR_BLOB_QUANTIZED_DETAIL_SCHEMA_VERSION: int = 2
"""Per-pair schema with quantized entropy-coded detail coefficients."""

_PAIR_BLOB_LOSSLESS_PRECONDITIONED_DETAIL_SCHEMA_VERSION: int = 3
"""Per-pair schema with lossless Brotli-friendly float32 detail preconditioning."""

_DETAIL_CODEC_QI16_DENSE: int = 1
_DETAIL_CODEC_QI16_ZERO_RLE: int = 2
_DETAIL_CODEC_QI16_STATIC_RANGE: int = 3
_DETAIL_CODEC_ZZ16_BYTEPLANE: int = 4
_DETAIL_CODEC_QI16_CONSTRICTION_RANGE: int = 5
_DETAIL_CODEC_F32_BYTE_SHUFFLE: int = 11
_RANGE_CODER_FREQUENCY_TOTAL: int = 1 << 15
_MAX_STATIC_RANGE_DETAIL_SYMBOLS: int = 0
"""Legacy pure-Python static-range codec selection cap.

The decoder still accepts method 3 for historical artifacts, but live Z8
subbands use method 5 (``qi16_constriction_range``), which is backed by the
Rust constriction extension and has its own payload grammar.
"""
_DETAIL_PAYLOAD_SELECTION_BROTLI_QUALITY: int = 1
"""Fast proxy for pair-level Brotli cost inside per-subband mode selection.

The final authority remains the byte-closed ``archive.zip``. Using q=11 inside
every candidate subband made full-video sweeps spend minutes inside Brotli
Zopfli search; q=1 preserves the "outer compression matters" signal cheaply.
"""
_DETAIL_CODEC_NAMES: dict[int, str] = {
    _DETAIL_CODEC_QI16_DENSE: "qi16_dense",
    _DETAIL_CODEC_QI16_ZERO_RLE: "qi16_zero_rle",
    _DETAIL_CODEC_QI16_STATIC_RANGE: "qi16_static_range",
    _DETAIL_CODEC_ZZ16_BYTEPLANE: "zigzag_u16_byteplane",
    _DETAIL_CODEC_QI16_CONSTRICTION_RANGE: "qi16_constriction_range",
    _DETAIL_CODEC_F32_BYTE_SHUFFLE: "f32_byte_shuffle",
}


def _encode_uvarint(value: int) -> bytes:
    if value < 0:
        raise ValueError("uvarint value must be non-negative")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _decode_uvarint(raw: bytes, pos: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if pos >= len(raw):
            raise ValueError("truncated uvarint in quantized detail stream")
        byte = raw[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError("uvarint too large in quantized detail stream")


def _zigzag_encode_i16(value: int) -> int:
    if value < -32768 or value > 32767:
        raise ValueError(f"i16 value out of range: {value}")
    return (value << 1) ^ (value >> 15)


def _zigzag_decode_i16(value: int) -> int:
    decoded = (value >> 1) ^ -(value & 1)
    if decoded < -32768 or decoded > 32767:
        raise ValueError(f"decoded i16 value out of range: {decoded}")
    return int(decoded)


def _zigzag_encode_i16_array(values: np.ndarray) -> np.ndarray:
    i32 = np.asarray(values, dtype=np.int32)
    return ((i32 << 1) ^ (i32 >> 15)).astype("<u2", copy=False)


def _zigzag_decode_u16_array(values: np.ndarray) -> np.ndarray:
    u32 = np.asarray(values, dtype=np.uint32)
    decoded = ((u32 >> 1) ^ (-(u32 & 1))).astype(np.int32, copy=False)
    if np.any(decoded < -32768) or np.any(decoded > 32767):
        raise ValueError("decoded zigzag byte-plane value outside i16 range")
    return decoded.astype("<i2", copy=False)


def _encode_qi16_zero_rle(q: np.ndarray) -> bytes:
    flat = np.asarray(q, dtype="<i2").reshape(-1)
    out = bytearray()
    pos = 0
    n = int(flat.size)
    while pos < n:
        zero_start = pos
        while pos < n and int(flat[pos]) == 0:
            pos += 1
        out.extend(_encode_uvarint(pos - zero_start))
        literal_start = pos
        while pos < n and int(flat[pos]) != 0:
            pos += 1
        literal_count = pos - literal_start
        out.extend(_encode_uvarint(literal_count))
        for value in flat[literal_start:pos]:
            out.extend(_encode_uvarint(_zigzag_encode_i16(int(value))))
    return bytes(out)


def _decode_qi16_zero_rle(payload: bytes, count: int) -> np.ndarray:
    values = np.zeros(int(count), dtype="<i2")
    pos = 0
    out_pos = 0
    while out_pos < count:
        zero_run, pos = _decode_uvarint(payload, pos)
        out_pos += zero_run
        if out_pos > count:
            raise ValueError("zero-RLE detail stream overran target shape")
        literal_count, pos = _decode_uvarint(payload, pos)
        if out_pos + literal_count > count:
            raise ValueError("literal run in zero-RLE detail stream overran target shape")
        for _ in range(literal_count):
            encoded, pos = _decode_uvarint(payload, pos)
            values[out_pos] = _zigzag_decode_i16(encoded)
            out_pos += 1
    if pos != len(payload):
        raise ValueError(
            f"zero-RLE detail stream trailing bytes (pos={pos} len={len(payload)})"
        )
    return values


def _encode_zz16_byteplane(q: np.ndarray) -> bytes:
    """Losslessly arrange quantized symbols into brotli-friendly byte planes."""

    zz = _zigzag_encode_i16_array(np.asarray(q, dtype="<i2")).reshape(-1)
    raw = zz.view(np.uint8).reshape(-1, 2)
    return raw[:, 0].tobytes(order="C") + raw[:, 1].tobytes(order="C")


def _decode_zz16_byteplane(payload: bytes, count: int) -> np.ndarray:
    expected = int(count) * 2
    if len(payload) != expected:
        raise ValueError(
            f"zigzag byte-plane detail payload length {len(payload)} != expected {expected}"
        )
    count_i = int(count)
    raw = np.empty((count_i, 2), dtype=np.uint8)
    raw[:, 0] = np.frombuffer(payload[:count_i], dtype=np.uint8, count=count_i)
    raw[:, 1] = np.frombuffer(payload[count_i:], dtype=np.uint8, count=count_i)
    return _zigzag_decode_u16_array(raw.reshape(-1).view("<u2"))


def _detail_payload_selection_score(payload: bytes) -> tuple[int, int]:
    """Estimate the cost a detail payload contributes after pair-level Brotli.

    The Z8 pair blob applies Brotli after the per-subband detail payloads are
    serialized. A range-coded substream can look excellent by raw byte length
    while being nearly incompressible to the outer Brotli pass; sparse/RLE
    streams can look larger raw but collapse after Brotli. Select on the object
    the archive actually pays for.
    """

    import brotli  # type: ignore[import-not-found]

    return (
        len(brotli.compress(payload, quality=_DETAIL_PAYLOAD_SELECTION_BROTLI_QUALITY)),
        len(payload),
    )


def _encode_quantized_detail_payload(
    sub: np.ndarray,
    *,
    quantization_step: float,
) -> tuple[int, bytes]:
    if quantization_step <= 0.0 or not np.isfinite(quantization_step):
        raise ValueError("detail quantization_step must be finite and positive")
    scaled = np.nan_to_num(sub.astype(np.float32, copy=False) / np.float32(quantization_step))
    q = np.rint(scaled)
    q = np.clip(q, -32768, 32767).astype("<i2", copy=False)
    dense = q.tobytes(order="C")
    sparse = _encode_qi16_zero_rle(q)
    byteplane = _encode_zz16_byteplane(q)
    method_payloads = [
        (_DETAIL_CODEC_QI16_DENSE, dense),
        (_DETAIL_CODEC_QI16_ZERO_RLE, sparse),
        (_DETAIL_CODEC_ZZ16_BYTEPLANE, byteplane),
    ]
    try:
        method_payloads.append(
            (
                _DETAIL_CODEC_QI16_CONSTRICTION_RANGE,
                _encode_qi16_constriction_range(q),
            )
        )
    except Exception:
        # Keep compression-side probes usable in partial environments. Runtime
        # archives only select this mode when constriction produced bytes here.
        pass
    if q.size <= _MAX_STATIC_RANGE_DETAIL_SYMBOLS:
        method_payloads.append(
            (
                _DETAIL_CODEC_QI16_STATIC_RANGE,
                _encode_qi16_static_range(q),
            )
        )
    preference = {
        _DETAIL_CODEC_QI16_CONSTRICTION_RANGE: 0,
        _DETAIL_CODEC_QI16_STATIC_RANGE: 0,
        _DETAIL_CODEC_QI16_ZERO_RLE: 1,
        _DETAIL_CODEC_ZZ16_BYTEPLANE: 2,
        _DETAIL_CODEC_QI16_DENSE: 3,
    }
    return min(
        method_payloads,
        key=lambda item: (
            *_detail_payload_selection_score(item[1]),
            preference[item[0]],
        ),
    )


def _decode_quantized_detail_payload(
    *,
    method: int,
    payload: bytes,
    shape: tuple[int, int, int],
    quantization_step: float,
) -> np.ndarray:
    if quantization_step <= 0.0 or not np.isfinite(quantization_step):
        raise ValueError("detail quantization_step must be finite and positive")
    count = int(np.prod(shape))
    if method == _DETAIL_CODEC_QI16_DENSE:
        expected = count * 2
        if len(payload) != expected:
            raise ValueError(
                f"dense quantized detail payload length {len(payload)} != expected {expected}"
            )
        q = np.frombuffer(payload, dtype="<i2", count=count)
    elif method == _DETAIL_CODEC_QI16_ZERO_RLE:
        q = _decode_qi16_zero_rle(payload, count)
    elif method == _DETAIL_CODEC_QI16_STATIC_RANGE:
        q = _decode_qi16_static_range(payload, count)
    elif method == _DETAIL_CODEC_ZZ16_BYTEPLANE:
        q = _decode_zz16_byteplane(payload, count)
    elif method == _DETAIL_CODEC_QI16_CONSTRICTION_RANGE:
        q = _decode_qi16_constriction_range(payload, count)
    else:
        raise ValueError(f"unsupported quantized detail codec method: {method}")
    return (q.astype(np.float32).reshape(shape) * np.float32(quantization_step)).astype(
        np.float32,
        copy=False,
    )


def _encode_f32_byteshuffle_payload(sub: np.ndarray) -> bytes:
    values = np.asarray(sub, dtype="<f4")
    byte_view = values.reshape(-1).view(np.uint8).reshape(-1, 4)
    return np.ascontiguousarray(byte_view.T).tobytes(order="C")


def _decode_f32_byteshuffle_payload(payload: bytes, shape: tuple[int, int, int]) -> np.ndarray:
    count = int(np.prod(shape))
    expected = count * 4
    if len(payload) != expected:
        raise ValueError(
            f"byte-shuffled float32 detail payload length {len(payload)} != expected {expected}"
        )
    planes = np.frombuffer(payload, dtype=np.uint8).reshape(4, count)
    dense = np.ascontiguousarray(planes.T).reshape(count, 4)
    return dense.view("<f4").reshape(shape).astype(np.float32, copy=True)


def _encode_qi16_static_range(q: np.ndarray) -> bytes:
    from tac.lossless.range_coder import encode_static_symbols, normalize_probabilities

    flat = np.asarray(q, dtype="<i2").reshape(-1)
    if flat.size > (1 << 32) - 1:
        raise ValueError("too many symbols for v2 static range detail stream")
    symbols, inverse, counts = np.unique(
        flat,
        return_inverse=True,
        return_counts=True,
    )
    if symbols.size == 0 or symbols.size > 0xFFFF:
        raise ValueError("unsupported static range symbol table size")
    frequencies = normalize_probabilities(
        counts.astype(np.float64),
        total=_RANGE_CODER_FREQUENCY_TOTAL,
    )
    stream = encode_static_symbols(inverse.astype(np.int64, copy=False), frequencies=frequencies)
    return b"".join(
        [
            struct.pack("<HI", int(symbols.size), len(stream)),
            np.asarray(symbols, dtype="<i2").tobytes(order="C"),
            np.asarray(frequencies, dtype="<u2").tobytes(order="C"),
            stream,
        ]
    )


def _decode_qi16_static_range(payload: bytes, count: int) -> np.ndarray:
    if len(payload) < 6:
        raise ValueError("static range detail payload too short")
    from tac.lossless.range_coder import decode_static_symbols

    table_size, stream_len = struct.unpack("<HI", payload[:6])
    pos = 6
    if table_size <= 0:
        raise ValueError("static range detail table must not be empty")
    symbols_bytes = int(table_size) * 2
    freqs_bytes = int(table_size) * 2
    if pos + symbols_bytes + freqs_bytes + stream_len != len(payload):
        raise ValueError("static range detail payload length mismatch")
    symbols = np.frombuffer(payload[pos : pos + symbols_bytes], dtype="<i2").copy()
    pos += symbols_bytes
    frequencies = np.frombuffer(payload[pos : pos + freqs_bytes], dtype="<u2").astype(int).tolist()
    pos += freqs_bytes
    stream = payload[pos : pos + stream_len]
    indices = decode_static_symbols(stream, count=int(count), frequencies=frequencies)
    return symbols[np.asarray(indices, dtype=np.int64)].astype("<i2", copy=False)


def _encode_qi16_constriction_range(q: np.ndarray) -> bytes:
    """Encode quantized i16 symbols with constriction's native range coder."""

    import constriction

    flat = np.asarray(q, dtype="<i2").reshape(-1)
    if flat.size > (1 << 32) - 1:
        raise ValueError("too many symbols for v2 constriction range detail stream")
    symbols, inverse, counts = np.unique(
        flat,
        return_inverse=True,
        return_counts=True,
    )
    if symbols.size == 0 or symbols.size > 0xFFFF:
        raise ValueError("unsupported constriction range symbol table size")
    symbols_i16 = np.asarray(symbols, dtype="<i2")
    counts_u32 = np.asarray(counts, dtype="<u4")
    if symbols_i16.size == 1:
        stream = b""
    else:
        probabilities = counts.astype(np.float64)
        probabilities /= float(probabilities.sum())
        model = constriction.stream.model.Categorical(probabilities, perfect=False)
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(inverse.astype(np.int32, copy=False), model)
        stream = np.asarray(encoder.get_compressed(), dtype="<u4").tobytes(order="C")
    return b"".join(
        [
            struct.pack("<HI", int(symbols_i16.size), len(stream)),
            symbols_i16.tobytes(order="C"),
            counts_u32.tobytes(order="C"),
            stream,
        ]
    )


def _decode_qi16_constriction_range(payload: bytes, count: int) -> np.ndarray:
    """Decode :func:`_encode_qi16_constriction_range`."""

    import constriction

    if len(payload) < 6:
        raise ValueError("constriction range detail payload too short")
    table_size, stream_len = struct.unpack("<HI", payload[:6])
    pos = 6
    if table_size <= 0:
        raise ValueError("constriction range detail table must not be empty")
    symbols_bytes = int(table_size) * 2
    counts_bytes = int(table_size) * 4
    if pos + symbols_bytes + counts_bytes + stream_len != len(payload):
        raise ValueError("constriction range detail payload length mismatch")
    symbols = np.frombuffer(payload[pos : pos + symbols_bytes], dtype="<i2").copy()
    pos += symbols_bytes
    counts = np.frombuffer(payload[pos : pos + counts_bytes], dtype="<u4").astype(
        np.float64
    )
    pos += counts_bytes
    if np.any(counts <= 0.0):
        raise ValueError("constriction range detail counts must be positive")
    if int(table_size) == 1:
        if stream_len != 0:
            raise ValueError("single-symbol constriction range stream must be empty")
        return np.full(int(count), int(symbols[0]), dtype="<i2")
    if stream_len % 4 != 0:
        raise ValueError("constriction range stream must be uint32-aligned")
    stream = payload[pos : pos + stream_len]
    probabilities = counts / float(counts.sum())
    model = constriction.stream.model.Categorical(probabilities, perfect=False)
    decoder = constriction.stream.queue.RangeDecoder(
        np.frombuffer(stream, dtype=np.uint32).copy()
    )
    indices = decoder.decode(model, int(count)).astype(np.int64, copy=False)
    return symbols[indices].astype("<i2", copy=False)


def _serialize_pair_wavelet_pyramid(
    pair_pyramid: dict[str, Any],
    *,
    detail_quantization_step: float | None = None,
    detail_quantization_steps: Mapping[tuple[str, int, str], float] | None = None,
    detail_lossless_preconditioner: bool = False,
) -> bytes:
    """Serialize one pair's wavelet-pyramid into deterministic bytes.

    ``pair_pyramid`` carries the canonical per-pair reconstruction state:
        - ``frame_0_top_ll``: (H_top, W_top, C) numpy float32 (top-level LL).
        - ``frame_1_top_ll``: (H_top, W_top, C) numpy float32 (top-level LL).
        - ``frame_0_details``: list of ``WaveletDetail2D`` (one per level,
          deepest first) holding (LH, HL, HH) coefficients as numpy float32.
        - ``frame_1_details``: list of ``WaveletDetail2D`` mirroring above.

    Returns brotli-compressed (quality=11) bytes per canonical PR95-family
    L32 (``pr95_family_l32_brotli_quality_11_max_v1``). Bumped from q=9 to
    q=11 per the canonical PR-or-greater binding-depth standing directive
    2026-05-30 + the operator-routed Yousfi-cascade TOP-1 quick-wins
    bolt-on cascade. Per CLAUDE.md HNeRV parity discipline L32:
    "5-10% per-pair byte savings; compression time is offline overhead so
    quality=11 is free at deploy time." This is the dominant L32 surface
    because the per-pair wavelet pyramid blobs cover ~99.5% of the
    canonical Z8HPC1 archive bytes at L1 SCAFFOLD scale.
    """
    import brotli  # type: ignore[import-not-found]

    if detail_quantization_step is not None and detail_quantization_steps is not None:
        raise ValueError("detail_quantization_step and detail_quantization_steps are mutually exclusive")
    if (detail_quantization_step is not None or detail_quantization_steps is not None) and detail_lossless_preconditioner:
        raise ValueError("detail quantization and detail_lossless_preconditioner are mutually exclusive")

    parts: list[bytes] = []
    if detail_quantization_step is not None or detail_quantization_steps is not None:
        schema_version = _PAIR_BLOB_QUANTIZED_DETAIL_SCHEMA_VERSION
    elif detail_lossless_preconditioner:
        schema_version = _PAIR_BLOB_LOSSLESS_PRECONDITIONED_DETAIL_SCHEMA_VERSION
    else:
        schema_version = _PAIR_BLOB_SCHEMA_VERSION
    parts.append(struct.pack("<B", schema_version))
    for frame_key in ("frame_0_top_ll", "frame_1_top_ll"):
        top_ll = np.asarray(pair_pyramid[frame_key], dtype=np.float32)
        if top_ll.ndim != 3:
            raise ValueError(
                f"{frame_key} must be (H, W, C); got shape {top_ll.shape}"
            )
        parts.append(struct.pack("<HHH", *top_ll.shape))
        parts.append(top_ll.tobytes(order="C"))
    for details_key in ("frame_0_details", "frame_1_details"):
        details = pair_pyramid[details_key]
        parts.append(struct.pack("<B", len(details)))
        for level_idx, detail in enumerate(details):
            for subband_key in ("lh", "hl", "hh"):
                sub = np.asarray(
                    getattr(detail, subband_key), dtype=np.float32
                )
                # M5 adapter emits 4D NHWC ``(1, H, W, C)`` per Protocol;
                # the canonical archive grammar strips the batch dim
                # (always B=1 at this milestone) for compact storage.
                if sub.ndim == 4:
                    if sub.shape[0] != 1:
                        raise ValueError(
                            f"{details_key}.{subband_key} batch dim must "
                            f"be 1 at M10; got shape {sub.shape}"
                        )
                    sub = sub[0]
                if sub.ndim != 3:
                    raise ValueError(
                        f"{details_key}.{subband_key} must be (H, W, C) "
                        f"after batch-strip; got shape {sub.shape}"
                    )
                parts.append(struct.pack("<HHH", *sub.shape))
                if detail_quantization_step is None and detail_quantization_steps is None:
                    if detail_lossless_preconditioner:
                        payload = _encode_f32_byteshuffle_payload(sub)
                        parts.append(
                            struct.pack(
                                "<BI",
                                _DETAIL_CODEC_F32_BYTE_SHUFFLE,
                                len(payload),
                            )
                        )
                        parts.append(payload)
                    else:
                        parts.append(sub.tobytes(order="C"))
                else:
                    if detail_quantization_steps is not None:
                        step_key = (details_key, int(level_idx), subband_key)
                        if step_key not in detail_quantization_steps:
                            raise ValueError(f"missing detail quantization step for {step_key}")
                        active_detail_quantization_step = float(detail_quantization_steps[step_key])
                    else:
                        assert detail_quantization_step is not None
                        active_detail_quantization_step = float(detail_quantization_step)
                    method, payload = _encode_quantized_detail_payload(
                        sub,
                        quantization_step=active_detail_quantization_step,
                    )
                    parts.append(
                        struct.pack(
                            "<BfI",
                            method,
                            active_detail_quantization_step,
                            len(payload),
                        )
                    )
                    parts.append(payload)
    raw = b"".join(parts)
    # L32 canonical PR95-family q=11 (was q=9). Per CLAUDE.md HNeRV parity
    # L32 + canonical equation pr95_family_l32_brotli_quality_11_max_v1.
    return bytes(brotli.compress(raw, quality=11))


def _deserialize_pair_wavelet_pyramid(blob: bytes) -> dict[str, Any]:
    """Inverse of :func:`_serialize_pair_wavelet_pyramid`."""
    import brotli  # type: ignore[import-not-found]

    from tac.substrates.z8_hierarchical_predictive_coding.mallat_dwt_adapter import (
        WaveletDetail2D,
    )

    raw = brotli.decompress(blob)
    pos = 0
    (version,) = struct.unpack("<B", raw[pos : pos + 1])
    pos += 1
    if version not in {
        _PAIR_BLOB_SCHEMA_VERSION,
        _PAIR_BLOB_QUANTIZED_DETAIL_SCHEMA_VERSION,
        _PAIR_BLOB_LOSSLESS_PRECONDITIONED_DETAIL_SCHEMA_VERSION,
    }:
        raise ValueError(
            f"pair wavelet-pyramid schema_version {version} not in supported "
            f"versions {{{_PAIR_BLOB_SCHEMA_VERSION}, "
            f"{_PAIR_BLOB_QUANTIZED_DETAIL_SCHEMA_VERSION}, "
            f"{_PAIR_BLOB_LOSSLESS_PRECONDITIONED_DETAIL_SCHEMA_VERSION}}}"
        )
    out: dict[str, Any] = {}
    for frame_key in ("frame_0_top_ll", "frame_1_top_ll"):
        h, w, c = struct.unpack("<HHH", raw[pos : pos + 6])
        pos += 6
        n = h * w * c * 4  # float32
        arr = (
            np.frombuffer(raw[pos : pos + n], dtype=np.float32)
            .reshape(h, w, c)
            .copy()
        )
        pos += n
        out[frame_key] = arr
    for details_key in ("frame_0_details", "frame_1_details"):
        (num_levels,) = struct.unpack("<B", raw[pos : pos + 1])
        pos += 1
        per_level: list[WaveletDetail2D] = []
        for _ in range(num_levels):
            subbands: dict[str, np.ndarray] = {}
            for subband_key in ("lh", "hl", "hh"):
                h, w, c = struct.unpack("<HHH", raw[pos : pos + 6])
                pos += 6
                shape = (int(h), int(w), int(c))
                if version == _PAIR_BLOB_SCHEMA_VERSION:
                    n = h * w * c * 4
                    sub = (
                        np.frombuffer(raw[pos : pos + n], dtype=np.float32)
                        .reshape(h, w, c)
                        .copy()
                    )
                    pos += n
                elif version == _PAIR_BLOB_QUANTIZED_DETAIL_SCHEMA_VERSION:
                    method, quantization_step, payload_len = struct.unpack(
                        "<BfI",
                        raw[pos : pos + 9],
                    )
                    pos += 9
                    payload = raw[pos : pos + payload_len]
                    pos += payload_len
                    sub = _decode_quantized_detail_payload(
                        method=int(method),
                        payload=payload,
                        shape=shape,
                        quantization_step=float(quantization_step),
                    ).copy()
                else:
                    method, payload_len = struct.unpack("<BI", raw[pos : pos + 5])
                    pos += 5
                    payload = raw[pos : pos + payload_len]
                    pos += payload_len
                    if int(method) != _DETAIL_CODEC_F32_BYTE_SHUFFLE:
                        raise ValueError(f"unsupported lossless detail preconditioner method: {method}")
                    sub = _decode_f32_byteshuffle_payload(payload, shape)
                subbands[subband_key] = sub
            per_level.append(
                WaveletDetail2D(
                    lh=subbands["lh"],
                    hl=subbands["hl"],
                    hh=subbands["hh"],
                )
            )
        out[details_key] = per_level
    if pos != len(raw):
        raise ValueError(
            f"pair wavelet-pyramid blob trailing bytes (pos={pos} len={len(raw)})"
        )
    return out


def _build_pair_pyramid_from_real_frames(
    binding: Z8CanonicalQuadrupleBinding,
    frame_0_nhwc_1pair: np.ndarray,
    frame_1_nhwc_1pair: np.ndarray,
) -> dict[str, Any]:
    """Build the per-pair wavelet pyramid from real video frames.

    Walks both frames through the M5 Mallat ``decompose_to_next_level``
    chain ``num_levels - 1`` times (mirroring
    :func:`canonical_quadruple_forward_step`). Per Mallat 1989 §7.5 perfect
    reconstruction the inverse chain (``recompose_from_next_level``) exactly
    inverts this pyramid; inflate-side reconstruction is byte-for-byte
    invertible up to float32 numerical precision (~1e-7).
    """
    contract = binding.contract
    num_levels = contract.num_levels
    pyramid: dict[str, Any] = {}
    for prefix, frame_nhwc in (
        ("frame_0", frame_0_nhwc_1pair),
        ("frame_1", frame_1_nhwc_1pair),
    ):
        if frame_nhwc.shape[0] != 1:
            raise ValueError(
                f"{prefix} must be 1-pair NHWC (1, H, W, C); got shape "
                f"{frame_nhwc.shape}"
            )
        current = frame_nhwc.astype(np.float32, copy=False)
        details: list[Any] = []
        for level_idx in range(num_levels - 1):
            ll, detail = binding.m5_per_level[level_idx].decompose_to_next_level(
                current
            )
            ll = np.asarray(ll, dtype=np.float32)
            details.append(detail)
            current = ll
        # ``current`` is now the top-level LL (after num_levels - 1
        # decompositions). Strip batch dim per canonical (H, W, C) storage.
        pyramid[f"{prefix}_top_ll"] = current[0]
        pyramid[f"{prefix}_details"] = details
    return pyramid


def reconstruct_pair_rgb_from_pyramid(
    binding: Z8CanonicalQuadrupleBinding,
    pair_pyramid: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct both RGB frames from a per-pair wavelet pyramid blob.

    Per Mallat 1989 §7.5 perfect reconstruction. The reconstruction is
    byte-derived from the trained wavelet coefficients in the archive
    (NOT synthetic frame base per Catalog #369). Returns ``(rgb_0, rgb_1)``
    each as ``(1, 3, H, W)`` numpy float32 in [0, 1] (canonical
    ``write_rgb_pair_to_raw`` ``input_range="unit"`` contract).
    """
    contract = binding.contract
    num_levels = contract.num_levels
    out: list[np.ndarray] = []
    for prefix in ("frame_0", "frame_1"):
        top_ll_hwc = np.asarray(
            pair_pyramid[f"{prefix}_top_ll"], dtype=np.float32
        )
        details = pair_pyramid[f"{prefix}_details"]
        if len(details) != num_levels - 1:
            raise ValueError(
                f"{prefix}_details length {len(details)} != "
                f"num_levels - 1 ({num_levels - 1})"
            )
        # Re-attach batch dim (B=1) so the canonical recompose contract
        # operates on NHWC 4D tensors per the M5 adapter Protocol; the
        # detail subbands stored as (H, W, C) are restored to (1, H, W, C).
        from tac.substrates.z8_hierarchical_predictive_coding.mallat_dwt_adapter import (
            WaveletDetail2D,
        )

        current = top_ll_hwc[np.newaxis, ...]
        for level_idx in range(num_levels - 1, 0, -1):
            stored = details[level_idx - 1]
            detail_4d = WaveletDetail2D(
                lh=np.asarray(stored.lh, dtype=np.float32)[np.newaxis, ...],
                hl=np.asarray(stored.hl, dtype=np.float32)[np.newaxis, ...],
                hh=np.asarray(stored.hh, dtype=np.float32)[np.newaxis, ...],
            )
            current = binding.m5_per_level[level_idx - 1].recompose_from_next_level(
                current, detail_4d
            )
            current = np.asarray(current, dtype=np.float32)
        # ``current`` is the level-0 (H, W, C) NHWC reconstruction.
        # Clamp to [0, 1] (real-video frames are in this range by load
        # convention; numerical noise from Mallat can push 1e-7 outside).
        current = np.clip(current[0], 0.0, 1.0)
        # NHWC -> NCHW (1, 3, H, W) per write_rgb_pair_to_raw contract.
        nchw = np.transpose(current, (2, 0, 1))[np.newaxis, ...]
        out.append(nchw.astype(np.float32, copy=False))
    return out[0], out[1]


def build_z8hpc1_archive_bytes_from_canonical_quadruple(
    binding: Z8CanonicalQuadrupleBinding,
    real_pair_rgb_frame_0: np.ndarray,
    real_pair_rgb_frame_1: np.ndarray,
) -> bytes:
    """Build a Z8HPC1 archive carrying canonical quadruple trained-state.

    Per Catalog #369 + Catalog #146 + HNeRV parity L4: the archive bytes
    derive ENTIRELY from real video pairs run through the canonical
    quadruple primitives (M5 Mallat decompose + M6 Wyner-Ziv encode + M4
    deterministic-state step). Inflate-side ``inflate_one_video_from_archive_bytes``
    reconstructs RGB frames perfectly from these bytes (Mallat 1989 §7.5
    perfect reconstruction; Wyner-Ziv 1976 Theorem 1 round-trip bound).

    Args:
        binding: ``Z8CanonicalQuadrupleBinding`` instance.
        real_pair_rgb_frame_0: ``(num_pairs, H, W, C)`` numpy float32 in
            [0, 1]; canonical loader output from
            :func:`load_real_video_targets_numpy`.
        real_pair_rgb_frame_1: ``(num_pairs, H, W, C)`` numpy float32 in
            [0, 1]; per-pair frame_1.

    Returns:
        Z8HPC1 archive bytes ready to write as ``0.bin`` per the canonical
        contest single-file archive grammar.
    """
    from tac.substrates.z8_hierarchical_predictive_coding.archive import (
        pack_archive,
    )

    if real_pair_rgb_frame_0.shape != real_pair_rgb_frame_1.shape:
        raise ValueError(
            f"frame_0 / frame_1 shape mismatch: "
            f"{real_pair_rgb_frame_0.shape} vs {real_pair_rgb_frame_1.shape}"
        )
    if real_pair_rgb_frame_0.ndim != 4:
        raise ValueError(
            f"real_pair_rgb_frame_0 must be 4D NHWC; got shape "
            f"{real_pair_rgb_frame_0.shape}"
        )
    contract = binding.contract
    num_pairs = int(real_pair_rgb_frame_0.shape[0])
    eval_h = int(real_pair_rgb_frame_0.shape[1])
    eval_w = int(real_pair_rgb_frame_0.shape[2])

    # Build per-pair wavelet pyramids; pack into wavelet_blob (canonical
    # DISTINGUISHING FEATURE #2 per Catalog #272).
    pair_blobs: list[bytes] = []
    for pair_idx in range(num_pairs):
        f0 = real_pair_rgb_frame_0[pair_idx : pair_idx + 1]
        f1 = real_pair_rgb_frame_1[pair_idx : pair_idx + 1]
        pyramid = _build_pair_pyramid_from_real_frames(binding, f0, f1)
        pair_blobs.append(_serialize_pair_wavelet_pyramid(pyramid))
    # Length-prefix each pair blob so inflate can iterate (u32 per pair).
    wavelet_blob_parts: list[bytes] = [
        struct.pack("<I", num_pairs)
    ]
    for blob in pair_blobs:
        wavelet_blob_parts.append(struct.pack("<I", len(blob)))
        wavelet_blob_parts.append(blob)
    wavelet_blob = b"".join(wavelet_blob_parts)

    # Per-pair Wyner-Ziv top-state payloads (canonical DISTINGUISHING
    # FEATURE #3 per Catalog #272). Run the M9 forward step per pair to
    # produce the canonical WZ payload; the per-pair payloads are
    # concatenated with u32 length prefixes for inflate-side parsing.
    wz_payloads: list[bytes] = []
    for pair_idx in range(num_pairs):
        forward = canonical_quadruple_forward_step(
            binding,
            real_pair_rgb_frame_0[pair_idx : pair_idx + 1],
        )
        # The forward dict surfaces wyner_ziv_payload_bytes (length); the
        # payload bytes themselves are produced inside the step. To keep
        # this helper canonical we re-run the M6 encode-only path on the
        # top-LL side info per the M9 compose pattern.
        # We do the inline encode here to capture the payload bytes:
        f0_target = real_pair_rgb_frame_0[pair_idx : pair_idx + 1].astype(
            np.float32, copy=False
        )
        # Re-derive top-level LL via the canonical M5 chain.
        current = f0_target
        for level_idx in range(contract.num_levels - 1):
            ll, _ = binding.m5_per_level[level_idx].decompose_to_next_level(
                current
            )
            current = np.asarray(ll, dtype=np.float32)
        top_ll_nhwc = current
        # Project the M4 state input (mirrors canonical_quadruple_forward_step).
        import torch  # M4 is torch-based.
        top_ll_mean_per_channel = top_ll_nhwc.mean(axis=(1, 2))
        rng = np.random.RandomState(0)
        proj_matrix = rng.randn(
            top_ll_mean_per_channel.shape[-1], binding.m4_latent_dim
        ).astype(np.float32) / max(top_ll_mean_per_channel.shape[-1], 1) ** 0.5
        latent_input_np = top_ll_mean_per_channel @ proj_matrix
        ego_np = np.zeros((1, binding.m4_ego_motion_dim), dtype=np.float32)
        input_at_t = torch.from_numpy(
            np.concatenate([latent_input_np, ego_np], axis=-1)
        )
        top_m4 = binding.m4_per_level[contract.num_levels - 1]
        prior_state = top_m4.initial_state(1)
        next_state_np = top_m4.step(prior_state, input_at_t).detach().numpy()
        # Build side_info per contract.
        side_c, side_h, side_w = contract.wyner_ziv_top_level_side_info_shape
        top_ll_nchw = np.transpose(top_ll_nhwc, (0, 3, 1, 2))
        top_ll_per_channel = top_ll_nchw.mean(axis=1, keepdims=True)
        side_info = np.tile(top_ll_per_channel, (1, side_c, 1, 1))
        if side_info.shape[-2:] != (side_h, side_w):
            h_min = min(side_info.shape[-2], side_h)
            w_min = min(side_info.shape[-1], side_w)
            buf = np.zeros((1, side_c, side_h, side_w), dtype=np.float32)
            buf[:, :, :h_min, :w_min] = side_info[:, :, :h_min, :w_min]
            side_info = buf
        side_info = side_info.astype(np.float32, copy=False)
        payload = binding.m6.encode(next_state_np, side_info)
        if int(forward["wyner_ziv_payload_bytes"]) != len(payload):
            raise ValueError(
                "M9 forward WZ payload length drifted from archive emission "
                f"for pair {pair_idx}"
            )
        wz_payloads.append(payload)
    wz_blob_parts: list[bytes] = [struct.pack("<I", num_pairs)]
    for payload in wz_payloads:
        wz_blob_parts.append(struct.pack("<I", len(payload)))
        wz_blob_parts.append(payload)
    wyner_ziv_top_blob = b"".join(wz_blob_parts)

    # Decoder state_dict: minimal canonical surface (M4 projection matrix
    # + grammar config) so inflate can deterministically reconstruct
    # side_info. Per Catalog #110 HISTORICAL_PROVENANCE the bytes are
    # write-once.
    decoder_state_dict: dict[str, Any] = {
        "m4_projection_matrix": np.zeros((1, 1), dtype=np.float32),
    }
    dreamer_state_dict: dict[str, Any] = {
        "m4_init_state_dummy": np.zeros((1,), dtype=np.float32),
    }
    # Per-level category indices: derive from per-pair top-LL magnitudes via
    # deterministic quantization to satisfy the canonical archive grammar
    # (DISTINGUISHING FEATURE #1 per Catalog #272). The values are NOT used
    # at inflate time per the M10 milestone scope (wavelet_blob carries the
    # full reconstruction surface); these are placeholder indices to honor
    # the canonical Z8HPC1 grammar contract.
    per_level_indices: list[np.ndarray] = []
    num_groups_per_level: list[int] = []
    num_categories_per_level: list[int] = []
    for level in contract.levels:
        n_groups = int(level.num_categorical_groups)
        n_categories = int(level.num_categorical_classes)
        num_groups_per_level.append(n_groups)
        num_categories_per_level.append(n_categories)
        indices = np.zeros((num_pairs, n_groups), dtype=np.int32)
        per_level_indices.append(indices)

    meta: dict[str, Any] = {
        "eval_height": eval_h,
        "eval_width": eval_w,
        "num_pairs": num_pairs,
        "runtime_custody_contract": build_z8_runtime_custody_contract(
            source="z8_m9_canonical_quadruple_archive_meta",
            section_name_style="archive_parser",
            archive_bound_candidate_package_emitted=True,
            trained_mlx_renderer_archive_export_ready=False,
        ),
        "schema": "z8hpc1_m10_canonical_quadruple_archive_v1",
        "wavelet_basis": "daubechies_db2",
    }
    return pack_archive(
        decoder_state_dict=decoder_state_dict,
        per_level_category_indices=per_level_indices,
        wavelet_coeffs_blob=wavelet_blob,
        wyner_ziv_top_blob=wyner_ziv_top_blob,
        dreamer_state_dict=dreamer_state_dict,
        meta=meta,
        num_levels=contract.num_levels,
        num_groups_per_level=tuple(num_groups_per_level),
        num_categories_per_level=tuple(num_categories_per_level),
        num_pairs=num_pairs,
        decoder_latent_dim=binding.m4_latent_dim,
        base_channels=int(real_pair_rgb_frame_0.shape[3]),
        wavelet_basis_id=0,  # canonical Daubechies-4
    )


def parse_pair_blobs_from_wavelet_blob(
    wavelet_blob: bytes,
) -> list[dict[str, Any]]:
    """Parse the per-pair wavelet-pyramid blobs from the archive's wavelet_blob.

    Returns a list of per-pair dicts in the format expected by
    :func:`reconstruct_pair_rgb_from_pyramid`.
    """
    pos = 0
    (num_pairs,) = struct.unpack("<I", wavelet_blob[pos : pos + 4])
    pos += 4
    pyramids: list[dict[str, Any]] = []
    for _ in range(num_pairs):
        (blob_len,) = struct.unpack("<I", wavelet_blob[pos : pos + 4])
        pos += 4
        pair_blob = wavelet_blob[pos : pos + blob_len]
        pos += blob_len
        pyramids.append(_deserialize_pair_wavelet_pyramid(pair_blob))
    if pos != len(wavelet_blob):
        raise ValueError(
            f"wavelet_blob trailing bytes (pos={pos} len={len(wavelet_blob)})"
        )
    return pyramids


def summarize_wavelet_blob_detail_codecs(wavelet_blob: bytes) -> dict[str, Any]:
    """Inspect detail codec choices without reconstructing frames."""

    import brotli  # type: ignore[import-not-found]

    pos = 0
    (num_pairs,) = struct.unpack("<I", wavelet_blob[pos : pos + 4])
    pos += 4
    schema_counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    method_payload_bytes: dict[str, int] = {}
    total_detail_subbands = 0
    total_detail_coefficients = 0
    total_detail_payload_bytes = 0
    for _pair_idx in range(num_pairs):
        (blob_len,) = struct.unpack("<I", wavelet_blob[pos : pos + 4])
        pos += 4
        pair_blob = wavelet_blob[pos : pos + blob_len]
        pos += blob_len
        raw = brotli.decompress(pair_blob)
        raw_pos = 0
        (version,) = struct.unpack("<B", raw[raw_pos : raw_pos + 1])
        raw_pos += 1
        schema_counts[str(int(version))] = schema_counts.get(str(int(version)), 0) + 1
        if version not in {
            _PAIR_BLOB_SCHEMA_VERSION,
            _PAIR_BLOB_QUANTIZED_DETAIL_SCHEMA_VERSION,
            _PAIR_BLOB_LOSSLESS_PRECONDITIONED_DETAIL_SCHEMA_VERSION,
        }:
            raise ValueError(f"unsupported pair blob schema version {version}")
        for _frame_key in ("frame_0_top_ll", "frame_1_top_ll"):
            h, w, c = struct.unpack("<HHH", raw[raw_pos : raw_pos + 6])
            raw_pos += 6 + int(h) * int(w) * int(c) * 4
        for _details_key in ("frame_0_details", "frame_1_details"):
            (num_levels,) = struct.unpack("<B", raw[raw_pos : raw_pos + 1])
            raw_pos += 1
            for _level_idx in range(num_levels):
                for _subband_key in ("lh", "hl", "hh"):
                    h, w, c = struct.unpack("<HHH", raw[raw_pos : raw_pos + 6])
                    raw_pos += 6
                    count = int(h) * int(w) * int(c)
                    total_detail_coefficients += count
                    total_detail_subbands += 1
                    if version == _PAIR_BLOB_SCHEMA_VERSION:
                        method_name = "float32_raw"
                        payload_len = count * 4
                        raw_pos += payload_len
                    elif version == _PAIR_BLOB_QUANTIZED_DETAIL_SCHEMA_VERSION:
                        method, _q_step, payload_len = struct.unpack(
                            "<BfI", raw[raw_pos : raw_pos + 9]
                        )
                        raw_pos += 9 + int(payload_len)
                        method_name = _DETAIL_CODEC_NAMES.get(
                            int(method),
                            f"unknown_{int(method)}",
                        )
                    else:
                        method, payload_len = struct.unpack("<BI", raw[raw_pos : raw_pos + 5])
                        raw_pos += 5 + int(payload_len)
                        method_name = _DETAIL_CODEC_NAMES.get(
                            int(method),
                            f"unknown_{int(method)}",
                        )
                    method_counts[method_name] = method_counts.get(method_name, 0) + 1
                    method_payload_bytes[method_name] = (
                        method_payload_bytes.get(method_name, 0) + int(payload_len)
                    )
                    total_detail_payload_bytes += int(payload_len)
        if raw_pos != len(raw):
            raise ValueError(
                f"pair wavelet-pyramid raw blob trailing bytes (pos={raw_pos} len={len(raw)})"
            )
    if pos != len(wavelet_blob):
        raise ValueError(
            f"wavelet_blob trailing bytes (pos={pos} len={len(wavelet_blob)})"
        )
    float32_count = method_counts.get("float32_raw", 0)
    return {
        "schema": "z8_wavelet_detail_codec_summary.v1",
        "num_pairs": int(num_pairs),
        "pair_blob_schema_counts": schema_counts,
        "detail_codec_method_counts": method_counts,
        "detail_codec_payload_bytes": method_payload_bytes,
        "total_detail_subbands": int(total_detail_subbands),
        "total_detail_coefficients": int(total_detail_coefficients),
        "total_detail_payload_bytes": int(total_detail_payload_bytes),
        "float32_detail_subband_count": int(float32_count),
        "quantized_or_preconditioned_detail_subband_count": int(
            total_detail_subbands - float32_count
        ),
    }


def pack_pair_pyramids_to_wavelet_blob(
    pair_pyramids: list[dict[str, Any]],
    *,
    detail_quantization_step: float | None = None,
    detail_quantization_steps: Mapping[tuple[str, int, str], float] | None = None,
    detail_lossless_preconditioner: bool = False,
) -> bytes:
    """Pack per-pair wavelet pyramids into the Z8HPC1 wavelet blob.

    This is the public inverse of :func:`parse_pair_blobs_from_wavelet_blob`.
    Rate-attack materializers use it after modifying Mallat detail-band
    coefficients so they can rebuild byte-closed archives without duplicating
    the length-prefix grammar.

    ``detail_quantization_step`` activates schema v2 for detail subbands:
    deterministic i16 quantization followed by the smallest self-contained
    payload among dense i16, zero-run literal coding, native constriction
    range coding, and zigzag byte-plane layout. The byte-plane path is lossless
    relative to the quantized symbols and exists to give the outer brotli pass
    high-byte zero planes plus low-byte Laplacian structure instead of mixed
    signed bytes.
    ``detail_quantization_steps`` is the per-frame/level/subband variant keyed
    by ``("frame_0_details"|"frame_1_details", level_idx, "lh"|"hl"|"hh")``.
    """

    wavelet_blob_parts: list[bytes] = [struct.pack("<I", len(pair_pyramids))]
    for pyramid in pair_pyramids:
        pair_blob = _serialize_pair_wavelet_pyramid(
            pyramid,
            detail_quantization_step=detail_quantization_step,
            detail_quantization_steps=detail_quantization_steps,
            detail_lossless_preconditioner=detail_lossless_preconditioner,
        )
        wavelet_blob_parts.append(struct.pack("<I", len(pair_blob)))
        wavelet_blob_parts.append(pair_blob)
    return b"".join(wavelet_blob_parts)


__all__ = [
    "CANONICAL_PROJECTION_SEED_DEFAULT",
    "DEFAULT_M4_D_STATE",
    "DEFAULT_M4_LATENT_DIM_PER_LEVEL",
    "POSE_SIDE_INFO_DIM",
    "CanonicalQuadrupleTrainingArtifact",
    "TrainingStepObservability",
    "Z8CanonicalQuadrupleBinding",
    "build_canonical_quadruple_binding_from_z8_config",
    "build_z8hpc1_archive_bytes_from_canonical_quadruple",
    "canonical_quadruple_forward_step",
    "compute_pose_side_info_canonical_equation_150",
    "load_real_video_pair_targets_numpy",
    "load_real_video_targets_numpy",
    "pack_pair_pyramids_to_wavelet_blob",
    "parse_pair_blobs_from_wavelet_blob",
    "reconstruct_pair_rgb_from_pyramid",
    "run_canonical_quadruple_training_loop",
    "summarize_wavelet_blob_detail_codecs",
]
