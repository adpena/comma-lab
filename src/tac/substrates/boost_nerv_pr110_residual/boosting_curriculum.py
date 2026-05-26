# SPDX-License-Identifier: MIT
"""MLX iterative boosting curriculum for boost_nerv_pr110_residual (L0 SCAFFOLD).

Per binding 2026-05-26 reframing: design the curriculum FOR BoostNeRV (not
bolt-on training schedule). This module declares the canonical curriculum
stages as typed dataclasses so future Phase 2 council deliberation has a
machine-readable contract; the actual MLX trainer (NotImplementedError at
L0) lands at L1+ after Phase 2 council symposium per Catalog #325.

Design memo: .omx/research/path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md

Stages:
    0. PR110 base extraction (subprocess inflate; one-time; cache to .omx/state/)
    1. Per-pair residual target computation (Stage 1 diagnostic per design memo)
    2. MLX residual learner warm-up (L2 loss, ~10 epochs, lr=1e-3)
    3. MLX score-aware fine-tune (Lagrangian, eval_roundtrip, EMA 0.997, ~50 epochs)
    4. Archive build + Catalog #1265 contest-equivalence gate (MANDATORY before paid dispatch)
    5. [L1+ optional] Round 2 residual on top of round 1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class CurriculumStage:
    """Single boosting-curriculum stage declaration.

    Stages are immutable + machine-readable so Phase 2 council symposium
    + cathedral autopilot ranker consumers can audit the curriculum at
    rest (no introspection of trainer source required).
    """

    stage_id: int
    name: str
    purpose: str
    mlx_implementable: bool
    """True for MLX stages; False for subprocess-PR110-inflate or other
    Python-host steps (Stage 0 + Stage 1 are not MLX-trained)."""

    convergence_target: str
    """Operator-readable convergence criterion. NOT a score claim per
    Catalog #127 — these are MLX-research-signal verdicts only."""

    estimated_wallclock_seconds: float
    """L0 estimate; Phase 2 council symposium replaces with empirical anchor."""


@dataclass(frozen=True)
class BoostingCurriculum:
    """Canonical 5-stage curriculum (L0 SCAFFOLD declaration).

    The curriculum is the substrate's CORE INTELLECTUAL CONTENT per the
    binding 2026-05-26 reframing: "design the curriculum FOR BoostNeRV".
    Future L1+ optimization is the substrate-optimal sweep across the 8
    CARGO-CULTED assumptions in the design memo cargo-cult audit.
    """

    stages: tuple[CurriculumStage, ...] = field(
        default_factory=lambda: (
            CurriculumStage(
                stage_id=0,
                name="pr110_base_extraction",
                purpose="Subprocess-invoke PR110 inflate.sh; cache per-pair RGB reconstructions to .omx/state/",
                mlx_implementable=False,
                convergence_target="600 pair-RGB frames cached; sha256 prefix matches PR110_BASE_SHA256_PREFIX",
                estimated_wallclock_seconds=30.0,
            ),
            CurriculumStage(
                stage_id=1,
                name="residual_target_computation",
                purpose="Compute residual_target = GT - PR110_base; emit diagnostic per design memo Stage 1",
                mlx_implementable=False,
                convergence_target=(
                    "residual_magnitude_p99_rgb_range >= 0.05 (PROCEED to Stage 2); "
                    "< 0.01 (DEFER per CLAUDE.md 'Forbidden premature KILL'); "
                    "otherwise marginal (Phase 2 council review per Catalog #325)"
                ),
                estimated_wallclock_seconds=10.0,
            ),
            CurriculumStage(
                stage_id=2,
                name="mlx_residual_learner_warmup",
                purpose=(
                    "MLX Adam(lr=1e-3) on L2(predicted_residual, residual_target); "
                    "~10 epochs warm-up; mlx.optimizers.Adam canonical"
                ),
                mlx_implementable=True,
                convergence_target="training_loss_reduction_from_initial_random >= 0.50 within 10 epochs",
                estimated_wallclock_seconds=120.0,  # ~2 min on M-series MLX
            ),
            CurriculumStage(
                stage_id=3,
                name="mlx_score_aware_finetune",
                purpose=(
                    "MLX score-aware Lagrangian = alpha*Δrate_bytes/37545489 + "
                    "beta*d_seg + gamma*sqrt(d_pose) on composed (PR110_base + clamped_residual); "
                    "eval_roundtrip + EMA 0.997 + canonical scorer helper per Catalog #164"
                ),
                mlx_implementable=True,
                convergence_target=(
                    "MLX scorer proxy shows >= -0.001 contest-units improvement vs PR110-alone baseline; "
                    "[macOS-MLX research-signal] per Catalog #1265 + #341 Tier A"
                ),
                estimated_wallclock_seconds=900.0,  # ~15 min on M-series MLX for 50 ep
            ),
            CurriculumStage(
                stage_id=4,
                name="archive_build_plus_catalog_1265_gate",
                purpose=(
                    "Extract EMA shadow weights → int8 quantize → brotli-q9 → BPR1 sidecar; "
                    "compose with PR110 archive; export PyTorch state_dict via Catalog #1251 bridge; "
                    "MANDATORY: invoke tools/gate_mlx_candidate_contest_equivalence.py per Catalog #1265 "
                    "BEFORE any paid CUDA dispatch is authorized"
                ),
                mlx_implementable=False,
                convergence_target=(
                    "Catalog #1265 gate verdict=PASS at threshold 0.001 contest-units "
                    "(MLX↔PyTorch decoder parity); FAIL = do NOT dispatch; audit per #1251 + #1257 + #1258"
                ),
                estimated_wallclock_seconds=180.0,
            ),
            CurriculumStage(
                stage_id=5,
                name="optional_round_2_boosting",
                purpose=(
                    "[L1+ DEFERRED] Re-run Stages 1-4 with residual_target_round_2 = "
                    "GT - (PR110_base + round_1_EMA_shadow_residual). Sidecar carries "
                    "both round-1 + round-2 residual blobs. Rate cost ~doubles (~16 KB); "
                    "empirical question whether diminishing-returns residual signal nets to negative ΔS."
                ),
                mlx_implementable=True,
                convergence_target=(
                    "[L1+] Same as Stage 3 + Stage 4 but applied to round-2 residual; "
                    "DEFERRED at L0 per Catalog #240 _full_main raises NotImplementedError"
                ),
                estimated_wallclock_seconds=1200.0,
            ),
        )
    )

    def stage_by_id(self, stage_id: int) -> CurriculumStage:
        """Lookup helper for downstream consumers."""
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(f"unknown stage_id: {stage_id}")

    def total_wallclock_seconds_estimate(self, include_optional: bool = False) -> float:
        """L0 wallclock-budget estimate (operator-routable input for cost-band)."""
        if include_optional:
            return sum(s.estimated_wallclock_seconds for s in self.stages)
        return sum(
            s.estimated_wallclock_seconds for s in self.stages if s.stage_id != 5
        )


# Canonical curriculum singleton — consumers import this rather than
# instantiating BoostingCurriculum() (canonical-instance pattern per
# Catalog #335 cathedral consumer contract).
CANONICAL_CURRICULUM = BoostingCurriculum()


# Sentinel: the trainer's _full_main raises NotImplementedError pointing
# here per Catalog #240 L0 SCAFFOLD posture. Phase 2 council symposium
# per Catalog #325 replaces this stub with the real implementation.
def _full_main_stub_raises() -> None:
    """L0 SCAFFOLD posture per Catalog #240.

    Refuses paid GPU dispatch until Phase 2 council symposium per Catalog
    #325 lands the real MLX trainer + score-aware Lagrangian + canonical
    auth-eval helper invocation per Catalog #226.
    """
    raise NotImplementedError(
        "boost_nerv_pr110_residual is L0 SCAFFOLD per Catalog #240. "
        "Full path council-gated until Phase 2 council symposium per "
        "Catalog #325 returns PROCEED or PROCEED_WITH_REVISIONS verdict + "
        "cargo-cult audit empirically validates the 8 CARGO-CULTED choices "
        "per design memo .omx/research/path_3_e_boost_nerv_against_pr110_"
        "substrate_design_20260526.md. Reactivation criteria pinned in "
        "lane registry notes per CLAUDE.md 'Forbidden premature KILL'."
    )
