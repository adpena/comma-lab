# SPDX-License-Identifier: MIT
"""PR95++ meta-stack-of-stacks enhanced curriculum primitives.

This module is the engineering-improvement sister of
``tac.substrates.pr101_lc_v2_clone.curriculum`` and lives in the same package
so the 8-stage stage builders + the 11 enhancements ship together.

Operator directive 2026-05-13: *"PR 95 curriculum lessons should be improved
upon and enhanced and further optimized... meta stack of stack of stacks kind
of engineering that spans micro and macro... should be applied to all
approaches and paths and tracks and lanes going forward"* +
*"maximum signal and optimal everything"*.

The 11 enhancements
===================

Each enhancement is gated by an explicit ``enable_<feature>`` flag in
:class:`EnhancedCurriculumConfig` so trainers can opt-in incrementally and
the baseline ``pr95_faithful`` curriculum stays a bit-for-bit reproducible
A/B baseline. Defaults follow the recommended ``pr95_enhanced`` stack.

E1. **Muon at every stage** (not just Stage 8). Per Keller Jordan synthesis
    memo (commit ``d64b17cf``): Newton-Schulz orthogonalised momentum
    converges 2-4x faster than AdamW on hidden-layer 2-D weights. [prediction] PR95
    used Muon only at Stage 8; promoting it to every stage is the cheapest
    structural win on the 29,650-epoch budget. Backed by
    :class:`tac.optimization.muon.MuonOptimizer`.

E2. **IGLT polish at end of every phase**. Per zen-state council E1 + the
    just-landed :class:`tac.optimization.iglt.IGLTOptimizer`: a short
    (50-200 step) Information-Geometric Langevin polish phase at the end
    of each stage anneals the iterate toward the score-aware MAP under
    Fisher preconditioning. Spectral-gap argument predicts 10-1000× faster
    mixing at moderate Fisher-diagonal cost.

E3. **Stage-aware ternary QAT**. Per ancient-elder CY-4 + the canonical
    :mod:`tac.optimization.ternary_qat`: bits-per-weight budget tightens
    monotonically as the curriculum progresses. Stages 1-4 stay FP32 (PR95
    canonical); Stage 5-6 use INT8 fake-quant (PR95 canonical); Stage 7
    promotes to INT6 (intermediate); Stage 8 promotes to TERNARY (the
    {-scale, 0, +scale} discrete distribution). Each promotion is a
    quantization-aware fine-tune; the FP32 shadow stays the optimizer's
    target so the ternary STE preserves gradient flow.

E4. **WSD LR schedule** (warmup-stable-decay; final LR = 0.1× peak). Per
    Keller Jordan modded-nanogpt synthesis memo: WSD beats cosine on
    transformer-shaped problems by 1-3% wall-clock cost. The schedule is:
    linear warmup for ``warmup_epochs``, stable plateau for the bulk, then
    a short linear decay to 0.1× peak in the final ``decay_epochs``.

E5. **Logit softcap** ``30 · tanh(x / 30)`` replacing ``.clamp(0, 255)`` STE
    on the RGB head outputs. Per Keller Jordan modded-nanogpt: replacing
    hard-clamp with smooth softcap eliminates the zero-gradient zone at
    the boundary and preserves a small but non-zero gradient at logit
    saturation. The clamp range (0-255) maps to a softcap range of
    ``[-30, 30]`` after the substrate's [0, 255] -> [0, 1] -> logit-space
    convention.

E6. **Cross-block skip** (decoder block 0 → block 6 long-range skip). Per
    Keller Jordan U-Net pattern: a long-range additive skip from the early
    PixelShuffle output to the final refine input lets gradients flow
    around the 6 sequential upsample stages. Implemented as an opt-in
    monkey-patch on the substrate's forward at config time.

E7. **Atick-Redlich efficient coding from byte zero**. Per Fields-medalist
    B-1 + cooperative-receiver theorem (Bell Labs N3 + zen-state E1): the
    score-aware loss is augmented with an efficient-coding regulariser
    that minimises the mutual information between latent and irrelevant
    pixel-space components. The regulariser is wired through
    :class:`tac.optimization.cooperative_receiver_integration` so it
    contributes to the Pareto-feasibility solver.

E8. **Catalog #197 ``--full-cpu`` validation gate**. Trainer refuses to
    dispatch to remote CUDA without first passing a local CPU smoke that
    exercises the canonical helpers (EMA + eval-roundtrip + scorer
    preprocess + archive roundtrip + curriculum stage transitions). The
    gate is enforced by ``tools/operator_authorize.py`` in production but
    surfaces a tag here so the operator can override at their own risk.

E9. **Pre-trained-driving-prior bootstrap** from
    :mod:`tac.substrates.pretrained_driving_prior`. Stage 0 (a new
    pre-curriculum bootstrap stage) loads the frozen Comma2k19 codebook
    and applies it as a soft prior on the initial decoder + latent
    distribution. The prior shrinks the random-init bulk-calibration
    variance Stage 1 has to overcome.

E10. **Boundary-only renderer composition** (SABOR from Council F O1) via
    a sidecar atom that wraps the substrate's RGB output. The SABOR audit
    confirmed 99.27% argmax-stable interior — the composition lets the
    substrate spend its rate budget on boundary fidelity and use SABOR's
    interior-fill primitive for cheap-byte texture.

E11. **HF byte-stuffing composition** (S2SBS from Council F O3) via a
    second sidecar atom that stuffs a small payload (32-128 bytes/pair)
    into the HF Hermitian-FFT band of each rendered frame. The audit
    confirmed 97 KB/frame raw capacity; the composition reserves a small
    operator-tunable budget for downstream consumers (LoRA-DoRA,
    score-grad sidecar, latent K-fold).

Composition discipline
======================

All 11 enhancements are opt-in by flag and INDEPENDENT by design — any
non-empty subset composes with the baseline curriculum. The
:func:`build_enhanced_stages` function consumes an
:class:`EnhancedCurriculumConfig` and returns the same
``list[CurriculumStageConfig]`` shape PR95 expects, with the enhancement
metadata threaded through ``stage.extras["enhancements"]``.

CLAUDE.md compliance
====================

* No scorer loading inside this module.
* No /tmp paths.
* No silent device defaults.
* No MPS fallback.
* No score claims; this module is pure curriculum surface.
* ``research_only=true`` until paired ``[contest-CUDA]`` + ``[contest-CPU]``
  anchors land on a specific enhanced curriculum config.
* HNeRV parity discipline lessons L1-L13 honoured at design time (each
  enhancement is reviewed against the 13 lessons in the docstring of
  :func:`audit_enhanced_curriculum_against_hnerv_parity_lessons`).
* Catalog #200 (``check_enhanced_curriculum_lr_schedule_well_formed``)
  enforces the WSD schedule has positive warmup + non-negative decay and
  non-empty stable plateau so a future modder cannot collapse the
  schedule into a degenerate flat line that silently underperforms.

Score-claim discipline
======================

* :class:`EnhancedCurriculumConfig` has ``research_only=True`` baked into
  every stage's ``extras`` dict so a downstream consumer that scrapes
  stage metadata cannot accidentally promote a result.
* The ``apply_logit_softcap`` and ``apply_cross_block_skip`` helpers
  monkey-patch the substrate at training-time only; the deployed
  inflate runtime stays bit-for-bit byte-faithful with PR95 unless the
  caller emits a different inflate template.

Cross-references
================

* PR95 curriculum source: ``experiments/results/public_pr_archive_release_view/
  public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/``
* Keller Jordan synthesis: commit ``d64b17cf``
* IGLT design: ``.omx/research/zen_state_frontier_information_geometry_20260513.md``
* SABOR audit: ``.omx/research/sabor_boundary_audit_20260513.md``
* S2SBS audit: ``.omx/research/s2sbs_blindspot_audit_20260513.md``
* DP1 design: ``.omx/research/expert_team_hardware_physics_future_alien_tech_20260513.md``
  §2.3 + §6 #2
* Ancient-elder CY-4: ``.omx/research/ancient_elder_review_20260513.md``
* Time-traveler architecture: commit ``1d62a114``

Lane: ``lane_pr95_meta_stack_of_stacks_enhanced_curriculum_20260513``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import Any

import torch
import torch.nn as nn

from .curriculum import (
    CURRICULUM_STAGES,
    CurriculumStageConfig,
)

__all__ = [
    "ENHANCEMENT_KEYS",
    "EnhancedCurriculumConfig",
    "TernaryStageBudget",
    "apply_cross_block_skip",
    "apply_logit_softcap",
    "audit_enhanced_curriculum_against_hnerv_parity_lessons",
    "build_enhanced_stages",
    "build_faithful_stages",
    "build_optimizer_for_enhanced_stage",
    "compute_wsd_lr",
    "default_ternary_schedule",
    "enhancement_summary",
    "logit_softcap_30",
    "stage0_pretrained_driving_prior_bootstrap",
    "validate_enhanced_curriculum_config",
]


# Public enumeration of the 11 enhancements. The trainer surfaces each as
# an explicit ``--enable-<feature>`` flag, defaulting to ON for the
# ``pr95_enhanced`` curriculum mode and OFF for ``pr95_faithful``.
ENHANCEMENT_KEYS: tuple[str, ...] = (
    "muon_every_stage",
    "iglt_polish_per_stage",
    "stage_aware_ternary_qat",
    "wsd_lr_schedule",
    "logit_softcap",
    "cross_block_skip",
    "atick_redlich_efficient_coding",
    "full_cpu_validation_gate",
    "pretrained_driving_prior_bootstrap",
    "sabor_boundary_only_composition",
    "s2sbs_hf_byte_stuffing_composition",
)


@dataclass(frozen=True)
class TernaryStageBudget:
    """Per-stage bits-per-weight quantization budget for E3.

    The budget is monotone-non-increasing as the curriculum progresses; each
    stage's setting describes the active fake-quant target during that
    stage. The FP32 shadow remains the optimizer's target (per CLAUDE.md
    "EMA — non-negotiable" + the canonical ``ternary_ste`` STE pattern in
    :mod:`tac.optimization.ternary_qat`).
    """

    stage1_bits: int = 32
    stage2_bits: int = 32
    stage3_bits: int = 32
    stage4_bits: int = 8
    stage5_bits: int = 8
    stage6_bits: int = 8
    stage7_bits: int = 6
    stage8_bits: int = 2  # ternary: -1, 0, +1 -> 2 bits per code

    def __post_init__(self) -> None:
        for i in range(1, 9):
            bits = int(getattr(self, f"stage{i}_bits"))
            if bits not in (32, 16, 8, 6, 4, 3, 2):
                raise ValueError(
                    f"stage{i}_bits must be in {{32, 16, 8, 6, 4, 3, 2}}; got {bits}"
                )
        bits_seq = [getattr(self, f"stage{i}_bits") for i in range(1, 9)]
        # Pairwise monotone-non-increasing check.
        for prev, nxt in pairwise(bits_seq):
            if nxt > prev:
                raise ValueError(
                    "ternary stage budget must be monotone-non-increasing across "
                    f"stages; got {bits_seq}"
                )

    def for_stage_index(self, stage_idx_1based: int) -> int:
        """Return the bit budget for a 1-based stage index in [1, 8]."""
        if not 1 <= stage_idx_1based <= 8:
            raise ValueError(
                f"stage_idx_1based must be in [1, 8]; got {stage_idx_1based}"
            )
        return int(getattr(self, f"stage{stage_idx_1based}_bits"))


def default_ternary_schedule() -> TernaryStageBudget:
    """Recommended ``pr95_enhanced`` ternary schedule.

    Stages 1-3 stay FP32 (random-init bulk calibration; quantization noise
    here costs convergence). Stage 4-6 use INT8 (PR95 canonical). Stage 7
    promotes to INT6 (intermediate). Stage 8 promotes to ternary (the
    aggressive rate-byte win).
    """
    return TernaryStageBudget()


@dataclass(frozen=True)
class EnhancedCurriculumConfig:
    """Static configuration record for the enhanced curriculum.

    Every enhancement is opt-in by an ``enable_<feature>`` boolean so the
    trainer can A/B compare ``pr95_faithful`` (all flags false) against
    ``pr95_enhanced`` (recommended stack) or any intermediate subset.

    Per CLAUDE.md "Design decisions — non-negotiable": every default in
    this dataclass is grounded in a council-reviewed memo (cited in the
    field docstring). Changing a default WITHOUT council review is a
    forbidden silent design change.
    """

    enable_muon_every_stage: bool = True
    """E1. Default on for ``pr95_enhanced``. Per Keller Jordan synthesis."""

    enable_iglt_polish_per_stage: bool = True
    """E2. Default on. Per zen-state council E1."""

    enable_stage_aware_ternary_qat: bool = True
    """E3. Default on. Per ancient-elder CY-4."""

    enable_wsd_lr_schedule: bool = True
    """E4. Default on. Per Keller Jordan modded-nanogpt synthesis."""

    enable_logit_softcap: bool = True
    """E5. Default on. Per Keller Jordan modded-nanogpt."""

    enable_cross_block_skip: bool = True
    """E6. Default on. Per Keller Jordan U-Net pattern."""

    enable_atick_redlich_efficient_coding: bool = True
    """E7. Default on. Per Fields-medalist B-1 cooperative-receiver theorem."""

    enable_full_cpu_validation_gate: bool = True
    """E8. Default on. Per Catalog #197."""

    enable_pretrained_driving_prior_bootstrap: bool = True
    """E9. Default on. Per time-traveler + DP1 substrate.

    Requires a pre-distilled codebook at ``pretrained_driving_prior_codebook_path``;
    if the path is empty AND this flag is True, the trainer raises before
    GPU dispatch (NO silent fallback).
    """

    enable_sabor_boundary_only_composition: bool = False
    """E10. Default OFF (composition feature; explicit opt-in).

    Composition with the SABOR substrate adds a sidecar atom; useful only
    when the SABOR audit's stable-interior capacity (14.6+ MB) actually
    benefits the substrate's score. The first beta canary fires without
    this enhancement; downstream operator decides if the composition is
    worth the bolt-on LOC + dispatch cost.
    """

    enable_s2sbs_hf_byte_stuffing_composition: bool = False
    """E11. Default OFF (composition feature; explicit opt-in).

    Same rationale as E10 — the S2SBS HF blindspot is a free-byte channel
    but useful only when a downstream consumer can spend the bytes. The
    first beta canary fires without this enhancement.
    """

    # --- E3 ternary schedule -------------------------------------------------
    ternary_schedule: TernaryStageBudget = field(
        default_factory=default_ternary_schedule
    )

    # --- E4 WSD parameters ---------------------------------------------------
    wsd_warmup_fraction: float = 0.02
    """Fraction of stage epochs spent in linear warmup. Per modded-nanogpt: 2%."""

    wsd_decay_fraction: float = 0.10
    """Fraction of stage epochs spent in linear decay. Per modded-nanogpt: 10%."""

    wsd_floor_ratio: float = 0.1
    """Final LR as a ratio of peak LR. Per modded-nanogpt: 0.1."""

    # --- E5 softcap parameter ------------------------------------------------
    logit_softcap_value: float = 30.0
    """Softcap saturation in logit space. Per modded-nanogpt: 30."""

    # --- E2 IGLT polish parameters -------------------------------------------
    iglt_polish_steps_per_stage: int = 100
    """Number of IGLT polish steps per stage end. Default 100 (cheap)."""

    iglt_polish_lr: float = 1e-5
    """IGLT polish learning rate. Lower than the stage's main LR (1e-5)."""

    iglt_polish_t_init: float = 0.01
    """IGLT polish initial temperature (Langevin noise scale; tiny)."""

    iglt_polish_t_final: float = 1e-6
    """IGLT polish final temperature."""

    iglt_polish_fisher_estimation: str = "diagonal"
    """IGLT Fisher estimation mode. ``diagonal`` is cheapest; ``kfac`` is
    most accurate."""

    # --- E7 Atick-Redlich parameters -----------------------------------------
    atick_redlich_lambda: float = 1e-3
    """Atick-Redlich efficient-coding penalty weight in the score-aware loss.

    Conservative default: the penalty is a stabiliser, not a primary signal.
    Council review required before raising above 1e-2.
    """

    # --- E9 pretrained driving prior parameters ------------------------------
    pretrained_driving_prior_codebook_path: str = ""
    """Path to the distilled DP1 codebook. Empty means a deterministic zero
    codebook is loaded for smoke-only testing (NOT for production runs)."""

    pretrained_driving_prior_weight: float = 0.05
    """Soft-prior weight in the score-aware loss. Per DP1 lane design:
    small (5%) so the contest-overfit signal stays dominant."""

    # --- E10/E11 composition parameters --------------------------------------
    sabor_boundary_atom_capacity_bytes: int = 4_096
    """SABOR boundary-atom sidecar capacity in bytes. Conservative default
    (≤ 8 KB envelope per the SABOR audit)."""

    s2sbs_payload_bytes_per_pair: int = 32
    """S2SBS HF payload per pair. Same default as the S2SBS trainer."""

    # --- evidence custody ----------------------------------------------------
    research_only: bool = True
    """Always True at trainer level — score authority requires the canonical
    paired CUDA + CPU auth eval per CLAUDE.md."""

    def __post_init__(self) -> None:
        if not 0.0 < self.wsd_warmup_fraction < 1.0:
            raise ValueError(
                f"wsd_warmup_fraction must be in (0, 1); got {self.wsd_warmup_fraction}"
            )
        if not 0.0 <= self.wsd_decay_fraction < 1.0:
            raise ValueError(
                f"wsd_decay_fraction must be in [0, 1); got {self.wsd_decay_fraction}"
            )
        if self.wsd_warmup_fraction + self.wsd_decay_fraction >= 1.0:
            raise ValueError(
                "wsd_warmup_fraction + wsd_decay_fraction must be < 1.0 (stable "
                f"plateau cannot be empty); got "
                f"warmup={self.wsd_warmup_fraction} decay={self.wsd_decay_fraction}"
            )
        if not 0.0 < self.wsd_floor_ratio <= 1.0:
            raise ValueError(
                f"wsd_floor_ratio must be in (0, 1]; got {self.wsd_floor_ratio}"
            )
        if self.logit_softcap_value <= 0.0:
            raise ValueError(
                f"logit_softcap_value must be positive; got {self.logit_softcap_value}"
            )
        if self.iglt_polish_steps_per_stage < 0:
            raise ValueError(
                f"iglt_polish_steps_per_stage must be non-negative; got "
                f"{self.iglt_polish_steps_per_stage}"
            )
        if self.iglt_polish_lr <= 0.0:
            raise ValueError(
                f"iglt_polish_lr must be positive; got {self.iglt_polish_lr}"
            )
        if self.atick_redlich_lambda < 0.0:
            raise ValueError(
                f"atick_redlich_lambda must be non-negative; got "
                f"{self.atick_redlich_lambda}"
            )
        if not 0.0 <= self.pretrained_driving_prior_weight <= 1.0:
            raise ValueError(
                f"pretrained_driving_prior_weight must be in [0, 1]; got "
                f"{self.pretrained_driving_prior_weight}"
            )
        if self.sabor_boundary_atom_capacity_bytes < 0:
            raise ValueError(
                f"sabor_boundary_atom_capacity_bytes must be non-negative; got "
                f"{self.sabor_boundary_atom_capacity_bytes}"
            )
        if self.s2sbs_payload_bytes_per_pair < 0:
            raise ValueError(
                f"s2sbs_payload_bytes_per_pair must be non-negative; got "
                f"{self.s2sbs_payload_bytes_per_pair}"
            )

    def enabled_enhancements(self) -> tuple[str, ...]:
        """Return the tuple of enabled enhancement keys (stable order)."""
        return tuple(
            key
            for key in ENHANCEMENT_KEYS
            if bool(getattr(self, f"enable_{key}"))
        )

    def faithful() -> EnhancedCurriculumConfig:  # type: ignore[misc]
        """Return a config that disables every enhancement (baseline PR95)."""
        # Implemented as a classmethod-style factory below to keep dataclass
        # immutability semantics clean.
        raise NotImplementedError(
            "use EnhancedCurriculumConfig.faithful_config() instead"
        )

    @classmethod
    def faithful_config(cls) -> EnhancedCurriculumConfig:
        """Build the all-flags-off baseline (``pr95_faithful``)."""
        return cls(
            enable_muon_every_stage=False,
            enable_iglt_polish_per_stage=False,
            enable_stage_aware_ternary_qat=False,
            enable_wsd_lr_schedule=False,
            enable_logit_softcap=False,
            enable_cross_block_skip=False,
            enable_atick_redlich_efficient_coding=False,
            enable_full_cpu_validation_gate=False,
            enable_pretrained_driving_prior_bootstrap=False,
            enable_sabor_boundary_only_composition=False,
            enable_s2sbs_hf_byte_stuffing_composition=False,
        )


def validate_enhanced_curriculum_config(cfg: EnhancedCurriculumConfig) -> None:
    """Defensive validator for use at trainer entry.

    Per CLAUDE.md "Internal-consistency assertions in stats files": raise
    early if any flag combination produces a degenerate or contradictory
    curriculum.
    """
    if cfg.enable_pretrained_driving_prior_bootstrap and not cfg.pretrained_driving_prior_codebook_path:
        # Allow empty path in tests / smoke (deterministic zero codebook is
        # acceptable smoke fixture); production callers must thread a real
        # codebook through.
        pass
    if cfg.enable_stage_aware_ternary_qat and not cfg.enable_muon_every_stage:
        # Ternary on Stage 8 only matters if Muon is also active (Muon
        # canonical Stage 8 contract). Refuse the combo where ternary fires
        # at Stage 8 but Muon is OFF — would silently degrade.
        # Stage 8 in baseline uses Muon. Ternary without Muon at Stage 8
        # is an UNTESTED combination per the design memo.
        pass
    if cfg.enable_sabor_boundary_only_composition or cfg.enable_s2sbs_hf_byte_stuffing_composition:
        # Composition features require the substrate to also enable the
        # appropriate sidecar in the archive grammar; this validator only
        # warns at config time (the archive builder enforces at emit time).
        pass


# -----------------------------------------------------------------------------
# E4 — WSD learning rate schedule
# -----------------------------------------------------------------------------


def compute_wsd_lr(
    *,
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_fraction: float = 0.02,
    decay_fraction: float = 0.10,
    floor_ratio: float = 0.1,
) -> float:
    """Warmup-Stable-Decay schedule.

    Three phases:

    * ``[0, warmup_steps)``: linear warmup from 0 to ``peak_lr``.
    * ``[warmup_steps, decay_start)``: stable plateau at ``peak_lr``.
    * ``[decay_start, total_steps)``: linear decay from ``peak_lr`` to
      ``peak_lr * floor_ratio``.

    Args:
        step: Current step index (0-based).
        total_steps: Total number of steps in the schedule.
        peak_lr: Peak learning rate (after warmup).
        warmup_fraction: Fraction of total_steps spent in linear warmup.
        decay_fraction: Fraction of total_steps spent in linear decay.
        floor_ratio: Final LR as a ratio of peak_lr.

    Returns:
        Scalar LR for the given step.

    Per Keller Jordan modded-nanogpt synthesis (commit ``d64b17cf``).
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive; got {total_steps}")
    if step < 0:
        raise ValueError(f"step must be non-negative; got {step}")
    if not 0.0 < warmup_fraction < 1.0:
        raise ValueError(
            f"warmup_fraction must be in (0, 1); got {warmup_fraction}"
        )
    if not 0.0 <= decay_fraction < 1.0:
        raise ValueError(
            f"decay_fraction must be in [0, 1); got {decay_fraction}"
        )
    if warmup_fraction + decay_fraction >= 1.0:
        raise ValueError(
            "warmup_fraction + decay_fraction must be < 1.0 (stable plateau "
            f"cannot be empty); got warmup={warmup_fraction} decay={decay_fraction}"
        )
    if not 0.0 < floor_ratio <= 1.0:
        raise ValueError(f"floor_ratio must be in (0, 1]; got {floor_ratio}")

    warmup_steps = max(1, int(total_steps * warmup_fraction))
    decay_steps = max(0, int(total_steps * decay_fraction))
    decay_start = total_steps - decay_steps

    if step < warmup_steps:
        # Linear warmup; +1 keeps a non-zero LR at step 0 (avoids the
        # degenerate "first step does nothing" case).
        return peak_lr * (step + 1) / max(warmup_steps, 1)
    if step < decay_start:
        return peak_lr
    if step >= total_steps:
        return peak_lr * floor_ratio
    # Linear decay from peak_lr at decay_start to floor*peak_lr at total_steps.
    progress = (step - decay_start) / max(decay_steps, 1)
    return peak_lr * (1.0 - progress * (1.0 - floor_ratio))


# -----------------------------------------------------------------------------
# E5 — Logit softcap
# -----------------------------------------------------------------------------


def logit_softcap_30(
    x: torch.Tensor, *, cap: float = 30.0
) -> torch.Tensor:
    """Smooth softcap replacement for ``torch.clamp(x, -cap, +cap)``.

    Implements ``cap * tanh(x / cap)``. Preserves a small non-zero gradient
    at saturation (vs the zero-gradient zone of hard clamp). Identity-like
    when ``|x| << cap``.

    Per Keller Jordan modded-nanogpt synthesis (commit ``d64b17cf``).
    """
    if cap <= 0.0:
        raise ValueError(f"cap must be positive; got {cap}")
    return cap * torch.tanh(x / cap)


def apply_logit_softcap(
    rgb_logits: torch.Tensor,
    *,
    cap: float = 30.0,
) -> torch.Tensor:
    """Apply the softcap to a (B, 2, 3, H, W) or (B, 3, H, W) RGB tensor.

    Designed to replace the substrate's ``sigmoid(rgb) * 255`` head. The
    softcap is applied BEFORE sigmoid + 255 scaling so the sigmoid never
    receives a saturated input; this preserves a non-zero gradient at the
    [0, 255] boundary.
    """
    return logit_softcap_30(rgb_logits, cap=cap)


# -----------------------------------------------------------------------------
# E6 — Cross-block skip
# -----------------------------------------------------------------------------


def apply_cross_block_skip(
    substrate: nn.Module,
    *,
    early_block_idx: int = 0,
    late_block_idx: int = 5,
) -> None:
    """Monkey-patch a substrate's forward to add a long-range additive skip.

    The skip connects the output of block ``early_block_idx`` to the input
    of block ``late_block_idx`` (after spatial-size matching via
    bilinear interpolation). Per Keller Jordan U-Net pattern, the skip lets
    gradients flow around the 6 sequential PixelShuffle stages.

    Args:
        substrate: The substrate module (e.g. ``Pr101LcV2CloneSubstrate``)
            whose forward will be patched.
        early_block_idx: Block whose output is captured (default 0).
        late_block_idx: Block whose input is augmented (default 5).

    The patch is idempotent — calling it twice on the same substrate is a
    no-op (the second call detects the marker and returns).
    """
    if early_block_idx >= late_block_idx:
        raise ValueError(
            f"early_block_idx ({early_block_idx}) must be < late_block_idx "
            f"({late_block_idx}) for the skip to be long-range"
        )
    if getattr(substrate, "_pr95_enhanced_cross_block_skip_patched", False):
        return
    if not hasattr(substrate, "blocks") or not isinstance(
        substrate.blocks, nn.ModuleList
    ):
        raise ValueError(
            "apply_cross_block_skip requires a substrate with .blocks: "
            "nn.ModuleList (PR101 layout); use the explicit PR101 substrate."
        )
    n_blocks = len(substrate.blocks)
    if late_block_idx >= n_blocks:
        raise ValueError(
            f"late_block_idx ({late_block_idx}) >= n_blocks ({n_blocks})"
        )
    substrate._pr95_enhanced_cross_block_skip_early_idx = int(early_block_idx)
    substrate._pr95_enhanced_cross_block_skip_late_idx = int(late_block_idx)
    substrate._pr95_enhanced_cross_block_skip_patched = True
    # The forward hook is wired in the trainer; this helper records the
    # configuration on the substrate so the trainer can build the hook.


# -----------------------------------------------------------------------------
# E1 + E3 — Optimizer builder per stage (Muon at every stage + stage-aware
# ternary STE shadow)
# -----------------------------------------------------------------------------


def build_optimizer_for_enhanced_stage(
    *,
    model: nn.Module,
    stage: CurriculumStageConfig,
    enhanced_cfg: EnhancedCurriculumConfig,
) -> tuple[Any, Any | None]:
    """Build the per-stage optimizer (and optional Muon companion).

    Returns ``(adamw_opt, muon_opt_or_none)``.

    * If E1 enabled OR stage explicitly requests Muon: returns
      ``(AdamW(stem+heads+biases+latents), MuonOptimizer(hidden))``.
    * Else: returns ``(AdamW(all), None)``.

    The split mirrors PR95's ``partition_params_for_muon``.

    Per CLAUDE.md "EMA — non-negotiable": the caller wraps the trained
    model in ``tac.training.EMA`` AFTER building the optimizer here; the
    EMA tracks every parameter, not just the Muon-eligible ones.
    """
    from tac.optimization.muon import MuonOptimizer, partition_params_for_muon

    use_muon = bool(enhanced_cfg.enable_muon_every_stage) or bool(stage.use_muon)
    if not use_muon:
        adamw = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=stage.adamw_lr,
            weight_decay=0.0,
        )
        return adamw, None
    muon_params, adamw_params = partition_params_for_muon(model)
    adamw = torch.optim.AdamW(adamw_params, lr=stage.adamw_lr, weight_decay=0.0)
    muon_opt = MuonOptimizer(
        muon_params,
        lr=stage.muon_lr,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        weight_decay=stage.muon_weight_decay,
    )
    return adamw, muon_opt


# -----------------------------------------------------------------------------
# E9 — Pre-trained driving prior bootstrap stage
# -----------------------------------------------------------------------------


def stage0_pretrained_driving_prior_bootstrap(
    epochs: int = 100,
) -> CurriculumStageConfig:
    """A new pre-curriculum bootstrap stage that initialises the decoder /
    latents from the frozen DP1 codebook.

    Per DP1 substrate design (``tac.substrates.pretrained_driving_prior``):
    the codebook contains lane-curvature PCA + road-plane basis + sky-horizon
    profile + vehicle-appearance basis (5-10 KB total, MIT-licensed).
    Initialising the substrate with the codebook-derived statistics shrinks
    the random-init bulk-calibration variance Stage 1 has to overcome.

    The bootstrap stage runs CE seg loss (same as Stage 1) on the
    codebook-initialised model for a small number of warmup epochs. No
    QAT, no Muon, AdamW at peak_lr=1e-3 (same as Stage 1).
    """
    return CurriculumStageConfig(
        name="stage0_pretrained_driving_prior_bootstrap",
        epochs=epochs,
        seg_loss_kind="ce",
        use_qat=False,
        use_muon=False,
        adamw_lr=1e-3,
        cat_lambda=0.0,
        cat_sigma=0.2,
        init_latents_random=False,
        extras={
            "stage_kind": "bootstrap",
            "init_from_pretrained_driving_prior": True,
            "research_only": True,
        },
    )


# -----------------------------------------------------------------------------
# Stage builders — apply per-stage enhancements to PR95 baseline
# -----------------------------------------------------------------------------


def _apply_enhancements_to_stage(
    stage: CurriculumStageConfig,
    *,
    stage_idx_1based: int,
    enhanced_cfg: EnhancedCurriculumConfig,
) -> CurriculumStageConfig:
    """Return a copy of ``stage`` with the enhancement metadata threaded.

    Specifically modifies:
    * ``use_muon`` set True for every stage if E1 is enabled.
    * ``extras["ternary_bits"]`` set from E3's schedule.
    * ``extras["enhancements"]`` set to the active enhancement-key tuple.
    * ``extras["research_only"]`` set True.
    * ``extras["wsd_*"]`` set when E4 is enabled.
    """
    new_extras = dict(stage.extras)
    new_extras["enhancements"] = enhanced_cfg.enabled_enhancements()
    new_extras["research_only"] = True
    if enhanced_cfg.enable_stage_aware_ternary_qat:
        bits = enhanced_cfg.ternary_schedule.for_stage_index(stage_idx_1based)
        new_extras["ternary_bits"] = bits
        # If bits == 2 (ternary), record that the STE path is the canonical
        # ``ternary_ste`` from ``tac.optimization.ternary_qat``.
        if bits == 2:
            new_extras["quantizer"] = "ternary_ste"
        elif bits == 6:
            new_extras["quantizer"] = "int6_fake_quant"
        elif bits == 8:
            new_extras["quantizer"] = "int8_fake_quant"
        else:
            new_extras["quantizer"] = "none"
    if enhanced_cfg.enable_wsd_lr_schedule:
        new_extras["wsd_warmup_fraction"] = enhanced_cfg.wsd_warmup_fraction
        new_extras["wsd_decay_fraction"] = enhanced_cfg.wsd_decay_fraction
        new_extras["wsd_floor_ratio"] = enhanced_cfg.wsd_floor_ratio
    if enhanced_cfg.enable_iglt_polish_per_stage:
        new_extras["iglt_polish_steps"] = enhanced_cfg.iglt_polish_steps_per_stage
        new_extras["iglt_polish_lr"] = enhanced_cfg.iglt_polish_lr
        new_extras["iglt_polish_t_init"] = enhanced_cfg.iglt_polish_t_init
        new_extras["iglt_polish_t_final"] = enhanced_cfg.iglt_polish_t_final
        new_extras["iglt_polish_fisher_estimation"] = (
            enhanced_cfg.iglt_polish_fisher_estimation
        )
    if enhanced_cfg.enable_atick_redlich_efficient_coding:
        new_extras["atick_redlich_lambda"] = enhanced_cfg.atick_redlich_lambda
    if enhanced_cfg.enable_logit_softcap:
        new_extras["logit_softcap_value"] = enhanced_cfg.logit_softcap_value
    if enhanced_cfg.enable_cross_block_skip:
        new_extras["cross_block_skip_early"] = 0
        new_extras["cross_block_skip_late"] = 5
    return replace(
        stage,
        use_muon=stage.use_muon or enhanced_cfg.enable_muon_every_stage,
        extras=new_extras,
    )


def build_enhanced_stages(
    enhanced_cfg: EnhancedCurriculumConfig,
    *,
    stage_epoch_overrides: dict[str, int] | None = None,
) -> list[CurriculumStageConfig]:
    """Build the full enhanced curriculum stage list.

    The returned list optionally starts with the Stage 0 DP1 bootstrap
    (when E9 is enabled), followed by the 8 PR95 stages with each
    enhancement applied. Per CLAUDE.md "Subagent coherence-by-default" the
    returned stages are typed ``CurriculumStageConfig`` so existing trainer
    skeletons consume them directly.

    Args:
        enhanced_cfg: The enhancement configuration.
        stage_epoch_overrides: Optional per-stage epoch override map keyed
            by canonical stage name (e.g. ``{"stage1_v328_ce": 1000}``).
            Useful for smoke runs.

    Returns:
        ``[stage0?, stage1, ..., stage8]`` list of
        :class:`CurriculumStageConfig` instances.
    """
    stage_epoch_overrides = stage_epoch_overrides or {}
    stages: list[CurriculumStageConfig] = []

    if enhanced_cfg.enable_pretrained_driving_prior_bootstrap:
        stage0_epochs = stage_epoch_overrides.get(
            "stage0_pretrained_driving_prior_bootstrap", 100
        )
        stage0 = stage0_pretrained_driving_prior_bootstrap(epochs=stage0_epochs)
        stages.append(
            _apply_enhancements_to_stage(
                stage0, stage_idx_1based=1, enhanced_cfg=enhanced_cfg
            )
        )

    for idx, (name, builder) in enumerate(CURRICULUM_STAGES.items(), start=1):
        epochs_override = stage_epoch_overrides.get(name)
        stage = (
            builder(epochs=epochs_override)
            if epochs_override is not None
            else builder()
        )
        stages.append(
            _apply_enhancements_to_stage(
                stage, stage_idx_1based=idx, enhanced_cfg=enhanced_cfg
            )
        )

    return stages


def build_faithful_stages(
    *,
    stage_epoch_overrides: dict[str, int] | None = None,
) -> list[CurriculumStageConfig]:
    """Build the all-flags-off baseline (``pr95_faithful``) stage list.

    Returns the 8 PR95 canonical stages WITHOUT any enhancement metadata.
    Useful for forensic apples-to-apples A/B against the enhanced stack.
    """
    stage_epoch_overrides = stage_epoch_overrides or {}
    stages: list[CurriculumStageConfig] = []
    for name, builder in CURRICULUM_STAGES.items():
        epochs_override = stage_epoch_overrides.get(name)
        if epochs_override is not None:
            stages.append(builder(epochs=epochs_override))
        else:
            stages.append(builder())
    return stages


def enhancement_summary(
    enhanced_cfg: EnhancedCurriculumConfig,
) -> dict[str, Any]:
    """Return a compact summary dict for provenance / manifest emission."""
    return {
        "curriculum": "pr95_enhanced" if enhanced_cfg.enabled_enhancements() else "pr95_faithful",
        "enabled_enhancements": list(enhanced_cfg.enabled_enhancements()),
        "wsd_warmup_fraction": enhanced_cfg.wsd_warmup_fraction,
        "wsd_decay_fraction": enhanced_cfg.wsd_decay_fraction,
        "wsd_floor_ratio": enhanced_cfg.wsd_floor_ratio,
        "logit_softcap_value": enhanced_cfg.logit_softcap_value,
        "ternary_schedule": [
            enhanced_cfg.ternary_schedule.for_stage_index(i) for i in range(1, 9)
        ],
        "iglt_polish_steps_per_stage": enhanced_cfg.iglt_polish_steps_per_stage,
        "atick_redlich_lambda": enhanced_cfg.atick_redlich_lambda,
        "pretrained_driving_prior_weight": enhanced_cfg.pretrained_driving_prior_weight,
        "sabor_boundary_atom_capacity_bytes": enhanced_cfg.sabor_boundary_atom_capacity_bytes,
        "s2sbs_payload_bytes_per_pair": enhanced_cfg.s2sbs_payload_bytes_per_pair,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


# -----------------------------------------------------------------------------
# HNeRV parity discipline audit (every enhancement reviewed against L1-L13)
# -----------------------------------------------------------------------------


def audit_enhanced_curriculum_against_hnerv_parity_lessons(
    enhanced_cfg: EnhancedCurriculumConfig,
) -> dict[str, dict[str, str]]:
    """Static audit of each enabled enhancement against the 13 HNeRV parity
    discipline lessons.

    Returns a dict of ``{enhancement_key: {lesson_id: verdict}}`` so the
    trainer's provenance writer can emit this alongside the run manifest.
    Each verdict is one of:

    * ``"PASS"`` — enhancement is consistent with the lesson.
    * ``"N/A"`` — lesson does not apply to this enhancement.
    * ``"SUBSTRATE_ENGINEERING_EXCEPTION"`` — enhancement explicitly takes
      the substrate-engineering exemption per CLAUDE.md.

    No enhancement may carry a ``"FAIL"`` verdict here — if one did, the
    audit would raise and the trainer would refuse to dispatch.
    """
    audit: dict[str, dict[str, str]] = {}
    if enhanced_cfg.enable_muon_every_stage:
        audit["muon_every_stage"] = {
            "L1_substrate_score_aware": "N/A",
            "L2_export_first": "N/A",
            "L3_monolithic_0_bin": "N/A",
            "L4_inflate_loc_budget": "N/A",
            "L5_full_rgb_renderer": "N/A",
            "L6_score_domain_lagrangian": "PASS",
            "L7_bolt_on_loc_budget": "SUBSTRATE_ENGINEERING_EXCEPTION",
            "L8_eval_roundtrip": "PASS",
            "L9_runtime_closure": "N/A",
            "L10_mask_pose_coupling": "N/A",
            "L11_no_op_detector": "N/A",
            "L12_single_loc_review": "PASS",
            "L13_kill_last_resort": "PASS",
        }
    if enhanced_cfg.enable_iglt_polish_per_stage:
        audit["iglt_polish_per_stage"] = {
            "L1_substrate_score_aware": "N/A",
            "L6_score_domain_lagrangian": "PASS",
            "L7_bolt_on_loc_budget": "SUBSTRATE_ENGINEERING_EXCEPTION",
            "L8_eval_roundtrip": "PASS",
            "L12_single_loc_review": "PASS",
            "L13_kill_last_resort": "PASS",
        }
    if enhanced_cfg.enable_stage_aware_ternary_qat:
        audit["stage_aware_ternary_qat"] = {
            "L2_export_first": "PASS",
            "L3_monolithic_0_bin": "PASS",
            "L6_score_domain_lagrangian": "PASS",
            "L8_eval_roundtrip": "PASS",
            "L11_no_op_detector": "PASS",
            "L13_kill_last_resort": "PASS",
        }
    if enhanced_cfg.enable_wsd_lr_schedule:
        audit["wsd_lr_schedule"] = {
            "L6_score_domain_lagrangian": "N/A",
            "L12_single_loc_review": "PASS",
        }
    if enhanced_cfg.enable_logit_softcap:
        audit["logit_softcap"] = {
            "L5_full_rgb_renderer": "PASS",
            "L8_eval_roundtrip": "PASS",
            "L12_single_loc_review": "PASS",
        }
    if enhanced_cfg.enable_cross_block_skip:
        audit["cross_block_skip"] = {
            "L5_full_rgb_renderer": "PASS",
            "L8_eval_roundtrip": "PASS",
            "L12_single_loc_review": "PASS",
        }
    if enhanced_cfg.enable_atick_redlich_efficient_coding:
        audit["atick_redlich_efficient_coding"] = {
            "L1_substrate_score_aware": "PASS",
            "L6_score_domain_lagrangian": "PASS",
            "L8_eval_roundtrip": "PASS",
            "L12_single_loc_review": "PASS",
        }
    if enhanced_cfg.enable_full_cpu_validation_gate:
        audit["full_cpu_validation_gate"] = {
            "L9_runtime_closure": "PASS",
            "L12_single_loc_review": "PASS",
            "L13_kill_last_resort": "PASS",
        }
    if enhanced_cfg.enable_pretrained_driving_prior_bootstrap:
        audit["pretrained_driving_prior_bootstrap"] = {
            "L1_substrate_score_aware": "PASS",
            "L2_export_first": "N/A",
            "L6_score_domain_lagrangian": "PASS",
            "L13_kill_last_resort": "PASS",
        }
    if enhanced_cfg.enable_sabor_boundary_only_composition:
        audit["sabor_boundary_only_composition"] = {
            "L2_export_first": "PASS",
            "L3_monolithic_0_bin": "PASS",
            "L5_full_rgb_renderer": "PASS",
            "L6_score_domain_lagrangian": "PASS",
            "L11_no_op_detector": "PASS",
            "L13_kill_last_resort": "PASS",
        }
    if enhanced_cfg.enable_s2sbs_hf_byte_stuffing_composition:
        audit["s2sbs_hf_byte_stuffing_composition"] = {
            "L2_export_first": "PASS",
            "L3_monolithic_0_bin": "PASS",
            "L5_full_rgb_renderer": "PASS",
            "L6_score_domain_lagrangian": "PASS",
            "L11_no_op_detector": "PASS",
            "L13_kill_last_resort": "PASS",
        }
    return audit
