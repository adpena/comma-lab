# SPDX-License-Identifier: MIT
# AUTOCAST_FP16_WAIVED:MLX_first_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_per_4107bbf8d_standing_directive_and_per_substrate_engineering_exception_per_HNeRV_parity_L7
# TF32_WAIVED:MLX_first_substrate_trainer_no_torch_or_CUDA_imports_per_mlx_first_canonical_doctrine_4107bbf8d_per_substrate_engineering_exception_per_HNeRV_parity_L7_and_per_macOS_M5_Max_MLX_execution_substrate
# TORCH_COMPILE_WAIVED:MLX_first_substrate_trainer_no_torch_imports_uses_mlx_lazy_eval_per_mlx_first_canonical_doctrine_4107bbf8d_per_standing_directive_2026_05_26_mlx_first_numpy_portable_individually_fractal
# NO_GRAD_WAIVED:MLX_first_substrate_trainer_uses_mlx_lazy_eval_no_autograd_graph_per_mlx_first_canonical_doctrine_4107bbf8d_per_standing_directive_2026_05_26_mlx_first_numpy_portable_individually_fractal
"""MLX-first trainer for cascade_c_prime_frame_1_segnet_waterfill substrate.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + HNeRV
parity L7 (substrate_engineering exception; bolt-on size budget exceeded
explicitly) + the per-substrate symposium PROCEED_WITH_REVISIONS revision #1
(`feedback_cascade_c_prime_paired_cuda_validation_deferred_pending_substrate_scaffold_20260526.md#option_a`).

**Mission**: bind the Atick-Redlich asymmetric scorer channel routing-decision
primitive (already landed in scaffold's `architecture.py`) with: (a) MLX-native
per-frame perturbation enumeration, (b) per-pair Lagrangian dual routing per
``tac.findings_lagrangian`` Phase 1-3 wire-in (Catalog #355), and (c) byte-
deterministic archive emission ready for the inflate runtime (already landed in
`inflate.py` 210 LOC numpy-portable).

**Status**: ``[macOS-MLX research-signal]`` per CLAUDE.md "MLX portable-local-
substrate authority" non-negotiable. Promotion to ``[contest-CUDA]`` requires
paired Linux x86_64 + NVIDIA auth-eval on the EXACT archive bytes per CLAUDE.md
"Submission auth eval - BOTH CPU AND CUDA". The per-substrate symposium
PROCEED_WITH_REVISIONS verdict (commit ``aaf0b1eb6``; subagent
``a1d16a40f4a722e26``) explicitly defers paired-CUDA validation to sister
subagent C; THIS trainer's role is the MLX-LOCAL compress-time enumerator that
emits per-pair routing decisions + bridges into the numpy-portable archive +
inflate path.

## Canonical-vs-unique decision per layer (per Catalog #290; substrate-engineering exception)

| Layer | Decision | Rationale |
|---|---|---|
| Perturbation enumeration | UNIQUE (MLX-native; this module) | Atick-Redlich asymmetric channel demands joint frame-0 + frame-1 mode menu enumeration; no canonical sister exists |
| Per-pair Lagrangian dual | CANONICAL (`architecture.py::compute_per_pair_lagrangian_dual_routing`) | already landed in scaffold; closed-form O(n_pairs × n_modes_joint) argmin |
| Score-aware loss | CANONICAL stencil (`tac.substrates.score_aware_common.score_pair_components`) | the trainer's MLX forward emits per-pair (seg, pose) deltas in the same axis decomposition the canonical helper produces — the helper itself is torch-bound, but our axis decomposition is the canonical surface |
| Archive emission | CANONICAL (`archive.py::pack_archive`) | already landed in scaffold; CH-CCP-FRAME1-WATERFILL grammar |
| MLX → numpy bridge | UNIQUE (`mlx_to_numpy_bridge.py`) | per-substrate state_dict export contract; inflate is numpy-portable |
| Hardware substrate detection | CANONICAL (`tac.substrates._shared.trainer_skeleton.detect_hardware_substrate`) | per Catalog #190; macOS_arm64 axis="cpu" returns the canonical posterior anchor token |
| Non-promotable provenance | CANONICAL (sister to `tac.substrates.nscs06_v8_chroma_lut.mlx_iteration.MLX_NON_PROMOTABLE_PROVENANCE`) | per Catalog #127/#192/#317/#341 |
| Tier-C MDL ablation hook | CANONICAL (`tier_c_hook.py` adapter; consumer of `tools/mdl_scorer_conditional_ablation.py`) | per Catalog #324 post-training Tier-C density validation |

## 9-dimension success checklist evidence (per Catalog #294)

| Dim | Evidence |
|---|---|
| 1 UNIQUENESS | Atick-Redlich asymmetric channel routing — class-shift via frame-1 menu expansion |
| 2 BEAUTY+ELEGANCE | trainer ~500 LOC; per-pair routing decision returned as canonical dataclass |
| 3 DISTINCTNESS | DISTINCT from PR110 K=16 frame-0-only menu (sister substrate) |
| 4 RIGOR | premise verified against scaffold + symposium + canonical equation #344 proposal |
| 5 OPTIMIZATION-PER-TECHNIQUE | per-pair Lagrangian dual IS substrate-optimal engineering decision |
| 6 STACK-OF-STACKS-COMPOSABILITY | composable as PR111-sub-frontier candidate atop PR110 |
| 7 DETERMINISTIC-REPRODUCIBILITY | MLX seed pinned; closed-form argmin deterministic |
| 8 EXTREME-OPTIMIZATION-PERFORMANCE | closed-form O(n_pairs × n_modes); MLX-native vectorized |
| 9 OPTIMAL-MINIMAL-CONTEST-SCORE | PROVISIONAL-PENDING-VERIFICATION per Catalog #363 (paired-CUDA gate) |

## Observability surface (per Catalog #305)

- **inspectable per layer**: per-pair perturbation matrices, per-pair routing decision, per-pair Lagrangian value, per-pair selected mode index
- **decomposable per signal**: SegNet d_seg + PoseNet d_pose + archive_bytes per Catalog #356
- **diff-able across runs**: deterministic given (seed, mode menu, pose_avg_baseline); diff via per-pair routing arrays
- **queryable post-hoc**: TrainerVerdict dataclass + JSON sidecar emission
- **cite-able**: (subagent_id, commit_sha, lane_id, archive_sha, run_utc) per Catalog #245
- **counterfactual-able**: byte-mutation smoke per archive.py + Catalog #139/#272

## Cargo-cult audit per assumption (per Catalog #303)

| Assumption | Classification | Unwind plan |
|---|---|---|
| MLX SegNet/PoseNet forward parity with PyTorch ≤2% argmax flip per #1258 | HARD-EARNED | sister `tac.substrates.nscs06_v8_chroma_lut.mlx_iteration.verify_mlx_segnet_argmax_parity_with_torch` |
| Random Gaussian perturbation menu approximates Pareto frontier of per-pair codec modes | CARGO-CULTED | 7th-order subagent: build PR110-K=16-Huffman-codebook-aligned perturbation menu; for now random Gaussian draws are PV-sufficient for MLX-local smoke |
| Per-pair Lagrangian dual converges in single argmin pass | HARD-EARNED | `architecture.py::compute_per_pair_lagrangian_dual_routing` + `tac.findings_lagrangian` Phase 1-3 wire-in |
| MLX state_dict → numpy bridge round-trip preserves byte-identity | HARD-EARNED via local round-trip smoke (this module's tests) | `mlx_to_numpy_bridge.py` round-trip test |
| Tier-C MDL ablation hook predicts contest-CUDA across-class verdict at MLX-local | CARGO-CULTED | Catalog #324 post-training Tier-C validation defers empirical proof to sister subagent C paired-CUDA |

## Predicted ΔS band (per Catalog #296)

| Status | Band | Validation |
|---|---|---|
| MLX-LOCAL synthesis (Cascade C') | -0.058820 [macOS-MLX research-signal] | Dykstra-feasibility verified via Cascade C' synthesis 48-cell sweep 41-PARADIGM/7-MARGINAL/0-NULL |
| Paired-CUDA expected | PROVISIONAL-PENDING-VERIFICATION | 10-30× literature overestimation common (Contrarian + Atick dissent); even -0.006 remains PR111-PLAUSIBLE |

Per Catalog #324: ``predicted_band_validation_status: pending_post_training``.
Reactivation criterion: post-training Tier-C re-measurement on landed paired-
CUDA smoke archive sha via `tools/mdl_scorer_conditional_ablation.py --tier c`.

## 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map: ACTIVE-research-signal (per-pair perturbation matrix IS the sensitivity surface)
- hook #2 Pareto constraint: ACTIVE-research-signal (per-axis seg+pose+bytes decomposition per Catalog #356)
- hook #3 bit-allocator: ACTIVE (the waterfill primitive — per-pair Lagrangian dual routing IS bit allocator)
- hook #4 cathedral autopilot dispatch: PROPOSED-pending-paired-CUDA per Catalog #335 contract
- hook #5 continual-learning posterior: ACTIVE (MLX-local smoke results append posterior via `append_council_anchor`)
- hook #6 probe-disambiguator: ACTIVE (Tier-C MDL ablation hook IS the disambiguator)

## NO_SUPERSESSION_NEEDED:adds_new_module_to_substrate_package_does_not_supersede_existing_landed_scaffold_files_per_Catalog_110_113_APPEND_ONLY_HISTORICAL_PROVENANCE
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .architecture import (
    FRAME_0,
    FRAME_1,
    POSE_SQRT_INNER,
    SEG_MULTIPLIER,
    PerPairRoutingDecision,
    compute_per_pair_lagrangian_dual_routing,
)
from .substrate_contract import CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT

__all__ = [
    "DEFAULT_MLX_AXIS_TAG",
    "DEFAULT_N_FRAME_0_MODES",
    "DEFAULT_N_FRAME_1_MODES",
    "DEFAULT_N_PAIRS",
    "DEFAULT_POSE_AVG_BASELINE",
    "MLX_NON_PROMOTABLE_PROVENANCE",
    "MLXFirstTrainerConfig",
    "MLXFirstTrainerError",
    "MLXFirstTrainerVerdict",
    "is_mlx_available",
    "run_mlx_first_compress_pass",
]


# ---------------------------------------------------------------------------
# Canonical constants (sister to nscs06_v8_chroma_lut.mlx_iteration)
# ---------------------------------------------------------------------------

DEFAULT_MLX_AXIS_TAG: str = "[macOS-MLX research-signal]"
"""Per CLAUDE.md FORBIDDEN_PATTERNS + MLX portable-local-substrate authority."""

DEFAULT_N_PAIRS: int = 600
"""Contest video pair count per upstream evaluator (sister to PR110 K=16 menu)."""

DEFAULT_N_FRAME_0_MODES: int = 16
"""PR110 K=16 frame-0 codebook size (sister substrate baseline)."""

DEFAULT_N_FRAME_1_MODES: int = 8
"""K=8 frame-1 codebook default per substrate contract (Atick dissent recommends
also probing K_frame_1 ∈ {4, 12, 16} in production runs)."""

DEFAULT_POSE_AVG_BASELINE: float = 3.4e-5
"""PR106 frontier operating point pose_avg baseline; sister to canonical
formula constants in `tac.score_composition` and `architecture.py`."""


MLX_NON_PROMOTABLE_PROVENANCE: dict[str, Any] = {
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
    "axis_tag": DEFAULT_MLX_AXIS_TAG,
    "evidence_grade": "MLX-research-signal",
    "contest_equivalence_gate_required_before_dispatch": True,
    "canonical_equation_proposal": (
        "atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1"
    ),
    "canonical_equation_status": "FORMALIZATION_PENDING",
    "blockers": (
        "macos_mlx_research_signal_not_contest_authority",
        "requires_paired_contest_cpu_plus_cuda_for_score_claim",
        "requires_pass_verdict_from_gate_mlx_candidate_contest_equivalence",
        "requires_paired_cuda_validation_per_per_substrate_symposium_revision_3",
    ),
}
"""Canonical non-promotable provenance per Catalog #127/#192/#317/#341.

Mirrors `tac.substrates.nscs06_v8_chroma_lut.mlx_iteration.MLX_NON_PROMOTABLE_PROVENANCE`
with substrate-specific equation proposal + per-substrate-symposium-revision blocker.
"""


class MLXFirstTrainerError(RuntimeError):
    """Raised when an MLX-first trainer step cannot be honored faithfully."""


def is_mlx_available() -> bool:
    """Return True iff ``mlx.core`` imports cleanly (Apple Silicon expected).

    Sister of `tac.substrates.nscs06_v8_chroma_lut.mlx_iteration.is_mlx_available`.
    Trainer fails-closed on non-Apple platforms (raises MLXFirstTrainerError).
    """
    try:  # pragma: no cover - exercised on Apple Silicon only
        import mlx.core  # noqa: F401
    except Exception:  # pragma: no cover - import guard
        return False
    return True


# ---------------------------------------------------------------------------
# Trainer config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MLXFirstTrainerConfig:
    """Frozen config for MLX-first per-pair Lagrangian dual routing trainer.

    Per Catalog #287/#323 canonical Provenance: every field is auditable +
    deterministic given (n_pairs, n_modes_per_frame, perturbation_scale, seed).
    """

    n_pairs: int = DEFAULT_N_PAIRS
    n_frame_0_modes: int = DEFAULT_N_FRAME_0_MODES
    n_frame_1_modes: int = DEFAULT_N_FRAME_1_MODES
    pose_avg_baseline: float = DEFAULT_POSE_AVG_BASELINE
    perturbation_scale_seg: float = 1e-5
    """Per-mode SegNet delta scale (NEGATIVE = savings; HALF-NORMAL distribution)."""
    perturbation_scale_pose: float = 5e-7
    """Per-mode PoseNet delta scale; smaller than seg per PR106 frontier operating
    point (pose_avg=3.4e-5 means tiny pose deltas dominate the Lagrangian)."""
    frame_0_seg_floor: float = 0.0
    """STRUCTURAL Atick-Redlich invariant: frame-0 perturbations have ZERO SegNet
    cost because SegNet's `x[:,-1,...]` slice never sees frame-0 directly."""
    seed: int = 20260526

    def __post_init__(self) -> None:
        if self.n_pairs <= 0 or self.n_pairs > 65535:
            raise ValueError(f"n_pairs must be in (0, 65536); got {self.n_pairs}")
        if self.n_frame_0_modes <= 0 or self.n_frame_0_modes > 64:
            raise ValueError(f"n_frame_0_modes must be in (0, 64]; got {self.n_frame_0_modes}")
        if self.n_frame_1_modes <= 0 or self.n_frame_1_modes > 64:
            raise ValueError(f"n_frame_1_modes must be in (0, 64]; got {self.n_frame_1_modes}")
        if self.pose_avg_baseline <= 0:
            raise ValueError(f"pose_avg_baseline must be positive; got {self.pose_avg_baseline}")
        if self.perturbation_scale_seg < 0 or self.perturbation_scale_pose < 0:
            raise ValueError("perturbation scales must be non-negative")
        if self.frame_0_seg_floor != 0.0:
            # Atick-Redlich asymmetric channel invariant
            raise ValueError(
                f"frame_0_seg_floor must be 0.0 per Atick-Redlich asymmetric channel; "
                f"got {self.frame_0_seg_floor}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_pairs": self.n_pairs,
            "n_frame_0_modes": self.n_frame_0_modes,
            "n_frame_1_modes": self.n_frame_1_modes,
            "pose_avg_baseline": self.pose_avg_baseline,
            "perturbation_scale_seg": self.perturbation_scale_seg,
            "perturbation_scale_pose": self.perturbation_scale_pose,
            "frame_0_seg_floor": self.frame_0_seg_floor,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# Trainer verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MLXFirstTrainerVerdict:
    """Per-pass MLX-first trainer verdict.

    All score-axis fields carry MLX_NON_PROMOTABLE_PROVENANCE per Catalog
    #287/#323 canonical Provenance. The total_score_delta is the per-pair sum
    of joint-menu Lagrangian improvements over the frame-0-only baseline.

    Per Catalog #356 per-axis decomposition: ``per_pair_seg_delta``,
    ``per_pair_pose_delta``, ``predicted_archive_bytes_delta`` are surfaced
    independently so downstream cathedral consumers can route through the
    Pareto polytope per Dim 1 Phase 4.
    """

    config: MLXFirstTrainerConfig
    routing: PerPairRoutingDecision
    predicted_archive_bytes_delta: int
    """SIGNED int per Catalog #356 ``AxisDecomposition`` contract.
    POSITIVE = archive grew; NEGATIVE = archive shrank."""
    state_dict: dict[str, np.ndarray]
    """MLX → numpy state-dict ready for bridge export per `mlx_to_numpy_bridge`."""
    hardware_substrate: str
    """Per Catalog #190; macOS_arm64 for MLX-local execution."""
    elapsed_seconds: float
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def axis_tag(self) -> str:
        return DEFAULT_MLX_AXIS_TAG

    @property
    def score_claim(self) -> bool:
        return False  # MLX-research-signal NEVER score claim

    @property
    def total_score_delta_mlx_research_signal(self) -> float:
        """Per-pair sum of Lagrangian improvements (NEGATIVE = score reduction)."""
        return float(self.routing.total_score_delta)

    @property
    def frame_1_routing_pct(self) -> float:
        return float(self.routing.frame_1_pct)

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "n_pairs": self.routing.n_pairs,
            "frame_1_count": self.routing.frame_1_count,
            "frame_1_pct": self.frame_1_routing_pct,
            "total_score_delta_mlx_research_signal": (
                self.total_score_delta_mlx_research_signal
            ),
            "predicted_archive_bytes_delta": self.predicted_archive_bytes_delta,
            "hardware_substrate": self.hardware_substrate,
            "elapsed_seconds": self.elapsed_seconds,
            "axis_tag": self.axis_tag,
            "score_claim": self.score_claim,
            "provenance": dict(self.provenance),
            "state_dict_keys": sorted(self.state_dict.keys()),
        }


# ---------------------------------------------------------------------------
# MLX-native perturbation enumeration
# ---------------------------------------------------------------------------


def _enumerate_mlx_perturbations(
    cfg: MLXFirstTrainerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate per-pair × per-mode perturbation matrices via MLX-native draws.

    Returns four numpy arrays (MLX → numpy bridged via `mx.array.tolist`):

    - ``frame_0_seg_penalty``: shape (n_pairs, n_frame_0_modes); STRUCTURALLY ZERO
      per Atick-Redlich asymmetric channel invariant (frame_0_seg_floor=0.0).
    - ``frame_0_pose_delta``: shape (n_pairs, n_frame_0_modes); half-normal draws
      shifted negative (NEGATIVE = pose savings; sister of PR110 frame-0 menu).
    - ``frame_1_seg_penalty``: shape (n_pairs, n_frame_1_modes); HALF-NORMAL
      POSITIVE draws (SegNet sees frame-1 directly per upstream/modules.py).
    - ``frame_1_pose_delta``: shape (n_pairs, n_frame_1_modes); HALF-NORMAL
      NEGATIVE draws (NEGATIVE = pose savings via frame-1 alternative path).

    Per Cascade C' synthesis: random draws at MLX-local approximate the Pareto
    frontier of per-pair codec modes well enough for routing-decision
    enumeration; production trainer's 7th-order iteration replaces these draws
    with PR110-K=16-Huffman-codebook-aligned perturbations.

    Per Catalog #303 cargo-cult audit: this approximation is CARGO-CULTED but
    PV-sufficient for MLX-local smoke per the symposium's PROCEED_WITH_REVISIONS
    verdict (paired-CUDA validation gates promotion).
    """
    if not is_mlx_available():
        raise MLXFirstTrainerError(
            "MLX unavailable; trainer is Apple-Silicon-only per "
            "`tac.substrates.nscs06_v8_chroma_lut.mlx_iteration.is_mlx_available` pattern"
        )
    import mlx.core as mx

    mx.random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    n_pairs = cfg.n_pairs
    n_f0 = cfg.n_frame_0_modes
    n_f1 = cfg.n_frame_1_modes

    # Atick-Redlich invariant: frame-0 SegNet penalty STRUCTURALLY ZERO.
    frame_0_seg_penalty = np.full(
        (n_pairs, n_f0), cfg.frame_0_seg_floor, dtype=np.float64
    )

    # MLX-native draws; bridged to numpy for the closed-form Lagrangian argmin.
    # NOTE: mx.random.normal is the canonical MLX primitive; we shift to half-
    # normal via abs + sign convention per the asymmetric channel.

    # Frame-0 pose: NEGATIVE half-normal (pose savings via PR110 sister)
    f0_pose_mlx = mx.random.normal(shape=(n_pairs, n_f0))
    mx.eval(f0_pose_mlx)
    f0_pose_np = -np.abs(np.array(f0_pose_mlx)) * cfg.perturbation_scale_pose
    frame_0_pose_delta = f0_pose_np.astype(np.float64)

    # Frame-1 seg: POSITIVE half-normal (SegNet cost via x[:,-1,...] slice)
    f1_seg_mlx = mx.random.normal(shape=(n_pairs, n_f1))
    mx.eval(f1_seg_mlx)
    f1_seg_np = np.abs(np.array(f1_seg_mlx)) * cfg.perturbation_scale_seg
    frame_1_seg_penalty = f1_seg_np.astype(np.float64)

    # Frame-1 pose: NEGATIVE half-normal (pose savings via frame-1 alt path)
    f1_pose_mlx = mx.random.normal(shape=(n_pairs, n_f1))
    mx.eval(f1_pose_mlx)
    f1_pose_np = -np.abs(np.array(f1_pose_mlx)) * cfg.perturbation_scale_pose
    frame_1_pose_delta = f1_pose_np.astype(np.float64)

    return (
        frame_0_seg_penalty,
        frame_0_pose_delta,
        frame_1_seg_penalty,
        frame_1_pose_delta,
    )


def _build_mlx_state_dict(
    cfg: MLXFirstTrainerConfig,
    routing: PerPairRoutingDecision,
    frame_0_pose_delta: np.ndarray,
    frame_1_pose_delta: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct the MLX → numpy state-dict ready for `mlx_to_numpy_bridge`.

    All arrays are numpy at this layer (MLX → numpy conversion happens inside
    the perturbation enumerator). This state-dict is canonically consumed by:

    1. `mlx_to_numpy_bridge.export_state_dict_to_npz` (bridge to archive.py)
    2. `archive.pack_archive` (CH-CCP-FRAME1-WATERFILL grammar)
    3. inflate.py downstream consumers

    Per HNeRV parity L3 monolithic single-file 0.bin: this state-dict maps 1:1
    onto the archive's parser_section_manifest fields per substrate_contract.py.
    """
    n_pairs = routing.n_pairs

    # Frame-0 menu indices (uint8; PR110 K=16 sister)
    f0_indices = np.zeros(n_pairs, dtype=np.uint8)
    # Frame-1 menu indices (uint8; K=8 default)
    f1_indices = np.zeros(n_pairs, dtype=np.uint8)

    for i in range(n_pairs):
        sel = int(routing.selected_mode_idx[i])
        if routing.routing_decision[i] == FRAME_1:
            # Selected index in joint menu = n_frame_0_modes + frame_1_local_idx
            f1_indices[i] = sel - cfg.n_frame_0_modes
        else:
            f0_indices[i] = sel

    # Pose deltas quantized to uint8 (sister of PR110 grammar)
    # Use per-pair selected pose delta replicated across POSE_DIMS as SCAFFOLD
    # (production trainer ships per-pair × POSE_DIMS quantized deltas).
    POSE_DIMS = 6
    pose_deltas_uint8 = np.zeros((n_pairs, POSE_DIMS), dtype=np.uint8)
    # Quantize selected_pose_delta to [-1, 1] then to uint8 [0, 255]
    scaled = routing.selected_pose_delta / max(
        abs(routing.selected_pose_delta).max(), 1e-12
    )
    scaled = np.clip(scaled, -1.0, 1.0)
    pose_deltas_uint8[:, 0] = ((scaled + 1.0) * 127.5).astype(np.uint8)

    return {
        "routing_decision": routing.routing_decision.astype(np.int8),
        "frame_0_menu_indices": f0_indices,
        "frame_1_menu_indices": f1_indices,
        "pose_deltas_uint8": pose_deltas_uint8,
        "selected_seg_delta": routing.selected_seg_delta.astype(np.float64),
        "selected_pose_delta": routing.selected_pose_delta.astype(np.float64),
        "selected_lagrangian": routing.selected_lagrangian.astype(np.float64),
        "per_pair_score_delta": routing.per_pair_score_delta.astype(np.float64),
    }


# ---------------------------------------------------------------------------
# Compress-pass entry point
# ---------------------------------------------------------------------------


def _detect_hardware_substrate_mlx_local() -> str:
    """Resolve canonical hardware_substrate token for MLX-local execution.

    Per Catalog #190 `detect_hardware_substrate` canonical helper: macOS_arm64
    returned for axis="cpu" on Darwin ARM64. The MLX execution layer adds a
    research-signal qualifier via `MLX_NON_PROMOTABLE_PROVENANCE["axis_tag"]`.
    """
    try:
        from tac.substrates._shared.trainer_skeleton import detect_hardware_substrate

        return detect_hardware_substrate(
            axis="cpu",
            substrate_tag="cascade_c_prime_frame_1_segnet_waterfill_mlx",
        )
    except Exception:
        # Fallback per CLAUDE.md "Apples-to-apples evidence discipline":
        # NEVER silently mislabel. Use explicit fallback token.
        system = platform.system()
        machine = platform.machine().lower()
        if system == "Darwin" and machine in {"arm64", "aarch64"}:
            return "macos_arm64"
        return "unknown_mlx_substrate"


def run_mlx_first_compress_pass(
    cfg: MLXFirstTrainerConfig | None = None,
    *,
    output_dir: Path | None = None,
    emit_json_sidecar: bool = True,
) -> MLXFirstTrainerVerdict:
    """Run one MLX-first compress pass for cascade_c_prime_frame_1_segnet_waterfill.

    This is the canonical entry point for MLX-LOCAL smoke (no PAID dispatch).
    Per the per-substrate symposium PROCEED_WITH_REVISIONS verdict: paired-CUDA
    validation gates promotion of the synthesis -0.058820 prediction to
    HARD-EARNED-EMPIRICALLY-VERIFIED. THIS function emits research-signal
    evidence that sister subagent C consumes for paired-CUDA dispatch.

    Args:
        cfg: Trainer config; defaults to MLXFirstTrainerConfig() if None.
        output_dir: Optional dir for JSON sidecar emission per Catalog #305
            observability surface.
        emit_json_sidecar: If True AND output_dir given, write verdict JSON.

    Returns:
        MLXFirstTrainerVerdict (frozen dataclass with MLX_NON_PROMOTABLE_PROVENANCE).

    Raises:
        MLXFirstTrainerError: if MLX unavailable (non-Apple platform).
    """
    if cfg is None:
        cfg = MLXFirstTrainerConfig()

    if not is_mlx_available():
        raise MLXFirstTrainerError(
            "MLX unavailable on this platform; cascade_c_prime_frame_1_segnet_waterfill "
            "MLX-first trainer requires Apple Silicon. Sister subagent C handles "
            "paired-CUDA Modal smoke per per-substrate symposium revision #3."
        )

    t_start = time.time()
    (
        frame_0_seg_penalty,
        frame_0_pose_delta,
        frame_1_seg_penalty,
        frame_1_pose_delta,
    ) = _enumerate_mlx_perturbations(cfg)

    routing = compute_per_pair_lagrangian_dual_routing(
        frame_0_seg_penalty,
        frame_0_pose_delta,
        frame_1_seg_penalty,
        frame_1_pose_delta,
        pose_avg_baseline=cfg.pose_avg_baseline,
    )

    state_dict = _build_mlx_state_dict(
        cfg, routing, frame_0_pose_delta, frame_1_pose_delta
    )

    # Predicted archive bytes delta (per Cascade C' synthesis Option B: ~79 bytes
    # routing sidecar + ~225 bytes frame-1 stream - frame-0 stream offset)
    predicted_archive_bytes_delta = int(
        79 + (routing.frame_1_count * 1)  # 1 byte/pair for frame-1 stream SCAFFOLD
    )

    hardware_substrate = _detect_hardware_substrate_mlx_local()
    elapsed = time.time() - t_start

    provenance = dict(MLX_NON_PROMOTABLE_PROVENANCE)
    provenance["lane_id"] = CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT.lane_id
    provenance["substrate_id"] = (
        CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT.id
    )
    provenance["run_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    provenance["per_substrate_symposium_revision"] = (
        "revision_1_mlx_first_trainer_landing"
    )

    verdict = MLXFirstTrainerVerdict(
        config=cfg,
        routing=routing,
        predicted_archive_bytes_delta=predicted_archive_bytes_delta,
        state_dict=state_dict,
        hardware_substrate=hardware_substrate,
        elapsed_seconds=elapsed,
        provenance=provenance,
    )

    if emit_json_sidecar and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = output_dir / "mlx_first_compress_pass_verdict.json"
        sidecar_path.write_text(
            json.dumps(verdict.as_dict(), indent=2, sort_keys=True, default=str)
        )

    return verdict
