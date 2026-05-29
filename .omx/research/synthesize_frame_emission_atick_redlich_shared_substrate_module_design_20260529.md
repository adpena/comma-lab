<!--
SPDX-License-Identifier: MIT
Slot R Slot L 8-step cascade Step 3 design memo
Catalog #229 premise-verification-before-edit
Catalog #290 canonical-vs-unique decision per layer
Catalog #294 9-dim success checklist evidence
Catalog #296 Dykstra-feasibility predicted-band
Catalog #303 cargo-cult audit per assumption
Catalog #305 observability surface declaration
Catalog #309 horizon-class declaration
Catalog #341 Tier A canonical-routing markers
Catalog #356 per-axis decomposition canonical Provenance

# Provenance
- Lane: lane_slot_r_slot_l_step_3_shared_substrate_module_synthesize_frame_emission_atick_redlich_20260529
- Cascade source: SLOT L 8-step operator-routable cascade Step 3 (per `feedback_slot_l_slot_h_top_3_super_additive_per_substrate_symposium_prep_landed_20260529.md`)
- Cells consumed: PR110 / PR101 / DQS1 × SYNTHESIZE_FRAME (α=1.10) per Slot H Phase B 84-cell composition_alpha matrix
- Symposia consumed: `council_per_substrate_symposium_{pr110,pr101,dqs1}_x_synthesize_frame_super_additive_20260529.md`
- Operator directive 2026-05-29 ~05:00Z verbatim: "keep three running, staggered starts, feeding them as they come back"
- Tier: A (observability only); Catalog #341 markers predicted_delta_adjustment=0.0; promotable=False; axis_tag=[predicted]
- DESIGN-ONLY: NO paid dispatch fires from this memo OR the helper it specifies; operator-routable per Catalog #325 14-day window (OPEN through 2026-06-12T06:15Z)
- 8th MLX-first standing directive: TRAINING MLX-first on M5 Max + INFLATE numpy-portable; framework_agnostic via `tac.framework_agnostic` decorators
- Per Wave N+46 canonical anti-pattern: do NOT duplicate code; EXTEND existing `tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss` rather than re-implement

horizon_class: frontier_pursuit
-->

# SYNTHESIZE_FRAME emission SHARED substrate module via Atick-Redlich cooperative-receiver (design)

**Date:** 2026-05-29T06:35:00Z
**Subagent ID:** slot_r_slot_l_step_3_shared_substrate_module_synthesize_frame_emission_atick_redlich_20260529_0500cst
**Lane:** `lane_slot_r_slot_l_step_3_shared_substrate_module_synthesize_frame_emission_atick_redlich_20260529` L1
**Type:** SHARED substrate module design (Catalog #290 canonical-vs-unique per-layer)
**Mission-alignment:** `frontier_breaking_enabler` (foundation for Steps 4-7 paired-CUDA dispatches at $0.18 envelope per Catalog #246)

## 1. Executive summary

This memo specifies the canonical SHARED substrate module
`tac.substrates._shared.synthesize_frame_emission_atick_redlich` (~400-600 LOC,
substrate_engineering scope per HNeRV parity L7 explicitly exceeds the 350-LOC
bolt-on budget) that operationalizes the 3 sister per-substrate symposia
landed by Slot L: PR110 / PR101 / DQS1 × SYNTHESIZE_FRAME (α=1.10 each per
Slot H Phase B 84-cell matrix).

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Slot L
op-routable #4 verbatim *"canonical helper SHARED across 3 sister cells;
per-substrate trainer + recipe UNIQUE per substrate"*: the SYNTHESIZE_FRAME
emission pipeline IS canonical SHARED (Atick-Redlich receiver-conditioned MI
loss + per-pair per-pixel boundary-class exploitation + synthetic-frame
metadata emission); the per-substrate adapters (PR110 V14 cascade /
PR101 PR95-family GOLD / DQS1 top32_selective_decoderq) are UNIQUE per
substrate (different archive grammars per HNeRV parity L3).

Per the Wave N+46 canonical anti-pattern (do NOT duplicate code; EXTEND
existing canonical surfaces): this module **consumes** the canonical
`tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss`
(landed 2026-05-13 at lane `lane_cooperative_receiver_primitive_20260513`)
rather than re-implementing Atick-Redlich. The NEW surface is the **per-pair
synthetic-frame emission pipeline** that converts the receiver-conditioned MI
training signal into byte-stable synthetic-frame metadata that fits each
substrate's archive grammar slot.

## 2. Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Atick-Redlich receiver-conditioned MI loss | **ADOPT_CANONICAL** `tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss` | Canonical primitive landed 2026-05-13; mathematically equivalent for any substrate that ships a full RGB renderer + trains against the contest scorer pair. Per Wave N+46 anti-pattern: re-implementing would be the canonical orphan-signal / duplicate-code failure mode. |
| Scorer-loss helper routing | **ADOPT_CANONICAL** `tac.substrates.score_aware_common.score_pair_components` | Canonical per Catalog #164; SegNet+PoseNet `preprocess_input` contract structurally enforced. The Atick-Redlich primitive already delegates to this helper internally. |
| Eval-roundtrip pipeline | **ADOPT_CANONICAL** `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` | Per CLAUDE.md "eval_roundtrip — non-negotiable" + Catalog #164. Already delegated to internally by Atick-Redlich primitive. |
| Inflate device selection | **ADOPT_CANONICAL** `tac.substrates._shared.inflate_runtime.select_inflate_device` | Per Catalog #205 + CLAUDE.md "MPS auth eval is NOISE". |
| Contest raw output path / pair-to-raw write | **ADOPT_CANONICAL** `tac.substrates._shared.inflate_runtime.raw_output_path` + `write_rgb_pair_to_raw` | Per Catalog #146 contest-compliant inflate runtime template + Catalog #361 Modal artifact filter preserves submission_dir. |
| **Per-pair synthetic-frame emission pipeline** | **FORK_BECAUSE_NOVEL** (NEW surface) | No canonical exists; this IS the novel surface. Operationalizes Atick-Redlich receiver-conditioned MI training signal → byte-stable synthetic-frame metadata. |
| **Per-substrate archive-grammar adapters (PR110 / PR101 / DQS1)** | **FORK_BECAUSE_PRINCIPLED_MISMATCH** (UNIQUE per substrate) | Per HNeRV parity L3: each substrate has DIFFERENT archive grammar (PR110 V14 FEC10 cascade; PR101 PR95-family canonical 4-section monolithic 0.bin; DQS1 top32_selective_decoderq selective decoder). Per UNIQUE-AND-COMPLETE-PER-METHOD: the per-substrate adapter is the canonical fork point. |
| Per-axis decomposition emission | **ADOPT_CANONICAL** `tac.cathedral.consumer_contract.AxisDecomposition` + `tac.score_composition.compose_score_from_axes` | Per Catalog #356 canonical Provenance + Catalog #357 Tier A vs Tier B contract. SHARED helper emits per-pair AxisDecomposition; consumed by cathedral autopilot ranker. |
| Canonical Provenance threading | **ADOPT_CANONICAL** `tac.provenance.builders.build_provenance_for_predicted` (this scaffold) + `build_provenance_for_archive_member` (post-paired-CUDA empirical anchor) | Per Catalog #323 canonical Provenance umbrella; every emission row carries the canonical Provenance per Catalog #287/#341/#357. |
| Tier A canonical-routing markers | **ADOPT_CANONICAL** per Catalog #341 (`predicted_delta_adjustment=0.0` + `promotable=False` + `axis_tag=[predicted]`) | Helper emits observability-only contributions; promotion gated by paired-CUDA empirical anchor per Catalog #246. |
| Framework-agnostic MLX/PyTorch/numpy bridge | **ADOPT_CANONICAL** `tac.framework_agnostic` decorators per 8th MLX-first standing directive | TRAINING MLX-first on M5 Max for $0 macOS-CPU-advisory probe per Contrarian binding revision; INFLATE numpy-portable per Catalog #295 PYTHONPATH self-containment. |
| Operational mechanism declaration | **ADOPT_CANONICAL** `runtime_overlay_consumed=True` per Catalog #220 | Synthetic-frame metadata bytes are operationally consumed by inflate at runtime (not no-op overlay). |
| Distinguishing-feature integration contract | **ADOPT_CANONICAL** per Catalog #272 (4-field contract: `distinguishing_feature_name` / `distinguishing_bytes_path` / `inflate_consumer_function` / `byte_mutation_smoke_passes`) | Per-substrate adapter declares the canonical contract; sister L1+ scaffold operational mechanism per Catalog #220. |

Net: 9 ADOPT_CANONICAL + 2 FORK (per-pair emission pipeline novel + per-substrate adapters principled-mismatch per HNeRV parity L3). The fork points are explicitly the SHARED substrate module's reason for existence; the canonical-helper-adopt points preserve coherence per UNIQUE-AND-COMPLETE-PER-METHOD ("canonical helpers used as tools, not obligations").

## 3. Cargo-cult audit per assumption (Catalog #303)

Per NSCS06 v6→v7 44% improvement via cargo-cult-unwind methodology + per the 3 Slot L symposia cargo-cult audits, every assumption underlying this module's design:

| # | Assumption | Classification | Unwind path |
|---|---|---|---|
| A1 | Atick-Redlich 1990 receiver-conditioned MI loss applies operationally to SYNTHESIZE_FRAME emission across PR110/PR101/DQS1 substrates | HARD-EARNED | Per the 3 Slot L symposia + the canonical primitive landed 2026-05-13. SegNet's stride-2 EfficientNet-B2 stem IS the canonical cooperative receiver per Atick verdict. |
| A2 | Synthetic-frame metadata can fit in each substrate's archive grammar slot WITHOUT structural restructuring | HARD-EARNED-WITH-OPERATIONAL-RESERVATION | Per HNeRV parity L3 grammar audit per PR101 Contrarian binding revision; PR110 V14 cascade may have more grammar slack; DQS1 selective decoder grammar has explicit slack. Unwind = $0 grammar audit per substrate adapter BEFORE recipe lands. |
| A3 | Per-pair emission is the correct granularity (not per-frame) | HARD-EARNED | Per HNeRV parity L19 (per-frame-PAIR latent 28-d predicting 2 frames per latent); PR95-family family canonical encodes 2 frames from each latent. SYNTHESIZE_FRAME inherits the per-pair canonical granularity. |
| A4 | The canonical Atick-Redlich primitive's `[0, 255]` RGB range contract is structurally compatible with substrate emission outputs | HARD-EARNED | The canonical primitive validates `[0, 255]` range via `_validate_rgb_255_tensor`; PR110/PR101/DQS1 substrate renderers emit `[0, 255]` per Catalog #146 contest output contract. |
| A5 | Per-axis AxisDecomposition emission per Catalog #356 is the canonical observability surface (not scalar prediction) | HARD-EARNED | Per Catalog #356 Phase 1 canonical foundation; Tier B downstream consumers need per-axis (seg + pose + rate) deltas for Pareto polytope intersection per Catalog #372 Dykstra solver. |
| A6 | The 3 per-substrate adapters can be UNIQUE-per-substrate without bloating the SHARED helper LOC budget | HARD-EARNED | Per HNeRV parity L7 substrate_engineering split (substrate engineering exceeds 350 LOC budget explicitly); per Hotz verdict §4 Slot L PR110 symposium: the emission helper fits within ≤200 LOC inflate budget + ≤350 LOC bolt-on. The 3 per-substrate adapters are bounded to <100 LOC each (PR110 / PR101 / DQS1 archive grammar slot injection only). |
| A7 | The SHARED helper inherits all CLAUDE.md non-negotiables from the canonical Atick-Redlich primitive (eval_roundtrip + EMA + score-aware loss + canonical scorer helpers + archive grammar + inflate runtime) | HARD-EARNED | The canonical primitive structurally enforces these via `_coerce_eval_roundtrip` + delegation to `score_pair_components`; SHARED helper inherits by construction. |
| A8 | The per-substrate adapter pattern is STRUCTURALLY COMPATIBLE with the canonical contract (does not require helper modification when 4th sister cell lands) | CARGO-CULTED | The 3-adapter scope is derived from Slot H Phase B 84-cell matrix TOP-3 cells; if a 4th sister cell (e.g. PR106 × SYNTHESIZE_FRAME) lands future-cap-window, the adapter pattern must NOT require modifying the SHARED helper signature. Unwind = use `Protocol`-based per-substrate adapter interface; sister 4th cell adds NEW adapter file without modifying SHARED helper. |
| A9 | The macOS-CPU-advisory $0 probe per Contrarian binding revision can run via the same SHARED helper code path | HARD-EARNED | Per 8th MLX-first standing directive + `tac.framework_agnostic` decorator pattern; same helper executes on PyTorch CUDA (paired dispatch) + MLX Apple Silicon (macOS-CPU-advisory probe) via framework-agnostic routing. |

**Net classification:** 7 HARD-EARNED + 1 HARD-EARNED-WITH-OPERATIONAL-RESERVATION + 1 CARGO-CULTED.

The CARGO-CULTED assumption (A8: 3-adapter scope structurally compatible with future 4th sister cell) is NOT a show-stopper — the unwind is the `Protocol`-based per-substrate adapter interface, which we DESIGN INTO the helper from byte zero. Per CLAUDE.md "Forbidden premature KILL": CARGO-CULTED classifications enumerate reactivation paths.

## 4. 9-dimension success checklist evidence (Catalog #294)

| # | Dimension | Evidence |
|---|---|---|
| 1 | UNIQUENESS (class-shift not within-class) | SYNTHESIZE_FRAME via Atick-Redlich cooperative-receiver IS a class-shift operator (DISTORTION-axis novel emission via receiver-conditioned MI), distinct from each substrate's within-class refinement (PR110 V14 cascade; PR101 fec8_static second-order Markov; DQS1 top32_selective_decoderq). Per Slot L 3 symposia uniqueness sections. |
| 2 | BEAUTY + ELEGANCE | SHARED helper ~400-600 LOC; per-substrate adapter <100 LOC each (3 adapters); total <1000 LOC including tests; reviewable in 30 minutes per PR101 GOLD canonical pattern (605 LOC for full substrate). Consumes existing canonical surfaces (Atick-Redlich primitive + score_aware_common + inflate_runtime + framework_agnostic) rather than duplicating. |
| 3 | DISTINCTNESS | DISTINCT from sister substrate modules under `tac.substrates._shared/` (trainer_skeleton / smoke_auth_eval_gate / score_aware_common / inflate_runtime / numpy_portable_inflate). Surface = per-pair synthetic-frame emission pipeline (no sister exists). Distinct from `tac.codec.cooperative_receiver` (codec-primitive layer; this is substrate-engineering layer per HNeRV parity L7). |
| 4 | RIGOR | Premise verification per Catalog #229 complete (3 symposia + canonical Atick-Redlich primitive + score_aware_common + inflate_runtime + cooperative_receiver/__init__.py all READ before design); adversarial null hypothesis (random_control per-substrate adapter test in §6 Observability surface). Canonical Provenance per Catalog #323 every emission row. |
| 5 | OPTIMIZATION PER TECHNIQUE | Canonical-vs-unique decision per layer per §2 (9 ADOPT + 2 FORK). Per-pair AxisDecomposition emission per Catalog #356; Tier A canonical-routing markers per Catalog #341; canonical Provenance per Catalog #323; framework-agnostic MLX/PyTorch/numpy bridge per 8th MLX-first standing directive. |
| 6 | STACK-OF-STACKS-COMPOSABILITY | Per-substrate emission orthogonal to substrate's own attack surface (PR110 V14 substitution / PR101 fec8 entropy-coder / DQS1 selective decoder); structurally composable with sister cells per Slot H 84-cell matrix; canonical Dykstra polytope intersection per Catalog #372 + #356. |
| 7 | DETERMINISTIC REPRODUCIBILITY | Byte-stable from canonical (substrate archive sha, SYNTHESIZE_FRAME emission helper hash, per-substrate adapter hash, framework_agnostic backend); byte-mutation smoke per Catalog #272 distinguishing-feature contract; tests verify byte-stability. |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | $0 macOS-CPU-advisory probe via MLX Apple Silicon backend (Contrarian binding revision; ~5 min wall-clock per substrate); $0.06 paired-CUDA + paired-CPU dispatch envelope per Catalog #246 ($0.18 total across 3 cells). Per-pair emission bounded; canonical 4-axis polytope projection bounded per Catalog #372. |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | NON-PROMOTABLE for [predicted] α=1.10 per Catalog #287/#323/#341 (helper is observability-only Tier A); predicted ΔS band per substrate ([-0.0025, +0.0015] for PR110/DQS1; [-0.0025, +0.0040] for PR101 per Assumption-Adversary binding revision); paired-CUDA + paired-CPU empirical anchor produces measurable [contest-CUDA] + [contest-CPU] score evidence per Catalog #246 + #127 + #316. |

## 5. Observability surface (Catalog #305)

Per CLAUDE.md "Max observability — non-negotiable":

1. **Inspectable per layer**:
   - Per-pair Atick-Redlich receiver-conditioned MI loss (seg + pose + sqrt(pose) components) via canonical `CooperativeReceiverOutput.cooperative_loss` + `seg_term` + `pose_term` + `pose_sqrt`.
   - Per-pair synthetic-frame emission delta (per-pair byte count of synthetic-frame metadata).
   - Per-pixel SegNet boundary-class disagreement (debug-mode opt-in).
   - Per-pair AxisDecomposition (predicted_d_seg_delta + predicted_d_pose_delta + predicted_archive_bytes_delta) per Catalog #356.

2. **Decomposable per signal**:
   - Per-axis decomposition per Catalog #356 (seg + pose + archive bytes per pair).
   - Per-substrate adapter decomposition (PR110 V14 grammar slot bytes / PR101 PR95-family grammar slot bytes / DQS1 selective decoder grammar slot bytes).
   - Per-framework-agnostic-backend decomposition (PyTorch CUDA / MLX Apple Silicon / numpy CPU portable inflate).

3. **Diff-able across runs**:
   - Byte-stable from canonical (substrate archive sha + helper hash + per-substrate adapter hash + framework backend ID).
   - `verify_synthesize_frame_emission_byte_stability` helper verifies deterministic byte output for same input + cooperative_receiver_state.

4. **Queryable post-hoc**:
   - Canonical probe outcomes ledger per Catalog #313 (`tac.probe_outcomes_ledger.register_probe_outcome` for each empirical anchor).
   - Canonical posterior anchor per Catalog #355 (`tac.council_continual_learning.append_council_anchor` for each post-empirical symposium update).
   - Canonical equation `synthesize_frame_via_atick_redlich_cooperative_receiver_v1` empirical anchors via `tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344.

5. **Cite-able**:
   - Canonical Provenance per Catalog #323 (`build_provenance_for_predicted` for scaffold; `build_provenance_for_archive_member` post-paired-CUDA empirical anchor; `build_provenance_for_macos_cpu_advisory` for Contrarian binding revision probe).
   - Atick & Redlich 1990 paper citation in helper docstring.

6. **Counterfactual-able**:
   - Byte-mutation smoke per Catalog #139 + Catalog #272 distinguishing-feature contract.
   - Per-substrate adapter `verify_distinguishing_feature_byte_mutation` invocation (operator-routable after paired-CUDA empirical anchor lands).
   - Sister Rao-Ballard alternative framing per RaoBallard verdict (probe-disambiguator per Catalog #125 hook #6).

## 6. Public API (per Catalog #294 Dimension 2 BEAUTY + ELEGANCE)

```python
# src/tac/substrates/_shared/synthesize_frame_emission_atick_redlich.py
"""SHARED substrate module for SYNTHESIZE_FRAME emission via Atick-Redlich.

See .omx/research/synthesize_frame_emission_atick_redlich_shared_substrate_module_design_20260529.md
for the canonical design memo (cargo-cult audit + 9-dim checklist +
observability surface + canonical-vs-unique decision per layer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import torch

from tac.codec.cooperative_receiver.atick_redlich import (
    AtickRedlichWeights,
    CooperativeReceiverOutput,
    cooperative_receiver_loss,
)
from tac.substrates._shared.inflate_runtime import (
    CAMERA_HW,
    select_inflate_device,
)


# ---------------------------------------------------------------------------
# Frozen contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SynthesizeFrameEmissionConfig:
    """Canonical configuration for one SYNTHESIZE_FRAME emission run.

    Args:
        substrate_id: Canonical substrate ID per the 3 Slot L symposia
            (one of {"pr110", "pr101", "dqs1"} or sister adapter ID for
            future cells).
        weights: Atick-Redlich Lagrangian weights; canonical defaults
            match the contest formula.
        per_pair_metadata_budget_bytes: Maximum synthetic-frame metadata
            bytes per pair (bounded per HNeRV parity L3 grammar slack).
        framework_agnostic_backend: "pytorch" (canonical CUDA paired
            dispatch) | "mlx" (Apple Silicon macOS-CPU-advisory probe per
            Contrarian binding revision) | "numpy" (CPU portable inflate
            per Catalog #295).
        apply_eval_roundtrip: Forbidden False per CLAUDE.md non-negotiable;
            structurally enforced by canonical Atick-Redlich primitive.
        random_seed: Pinned per deterministic-reproducibility per
            Dimension 7.
    """

    substrate_id: str
    weights: AtickRedlichWeights = field(default_factory=AtickRedlichWeights)
    per_pair_metadata_budget_bytes: int = 80
    framework_agnostic_backend: str = "pytorch"
    apply_eval_roundtrip: bool = True
    random_seed: int = 42

    def __post_init__(self) -> None:
        # Field validation per Catalog #287 placeholder rejection + canonical
        # contract invariants. Detailed validators in module body.
        ...


@dataclass(frozen=True)
class AtickRedlichReceiver:
    """Per-substrate adapter for SYNTHESIZE_FRAME emission.

    The receiver wraps the canonical SegNet+PoseNet scorer pair PLUS the
    per-substrate archive-grammar slot injection logic. Per HNeRV parity L3,
    each substrate has DIFFERENT archive grammar; the adapter is UNIQUE per
    substrate (PR110 V14 cascade / PR101 PR95-family canonical / DQS1
    top32_selective_decoderq).

    Per UNIQUE-AND-COMPLETE-PER-METHOD operating mode: the adapter is the
    canonical fork point; the SHARED emission helper consumes it via
    Protocol-typed interface.

    Attributes:
        substrate_id: Canonical substrate ID.
        seg_scorer: Contest SegNet module (canonical preprocess_input contract).
        pose_scorer: Contest PoseNet module (canonical preprocess_input contract).
        archive_grammar_slot_injector: Per-substrate callable that inserts
            synthetic-frame metadata bytes into the archive's compatible slot.
            Returns the modified archive bytes + per-axis bytes delta.
    """

    substrate_id: str
    seg_scorer: torch.nn.Module
    pose_scorer: torch.nn.Module
    archive_grammar_slot_injector: Callable[[bytes, bytes], tuple[bytes, int]]


@dataclass(frozen=True)
class SynthesizeFrameEmissionPerPairResult:
    """One per-pair emission result with Catalog #356 AxisDecomposition.

    Attributes:
        per_pair_index: Pair index (0..N_PAIRS-1).
        frame_0_bytes: Synthetic frame_0 RGB bytes (raw uint8; per-pixel
            cooperative-receiver-optimized).
        frame_1_bytes: Synthetic frame_1 RGB bytes (sister; per-pair
            paired emission).
        per_pair_metadata_bytes: Synthetic-frame metadata bytes for this
            pair (bounded per per_pair_metadata_budget_bytes).
        atick_redlich_loss: Canonical CooperativeReceiverOutput with
            scalar loss + seg + pose + pose_sqrt components.
        predicted_axis_decomposition: Per Catalog #356 canonical Provenance;
            predicted_d_seg_delta + predicted_d_pose_delta +
            predicted_archive_bytes_delta + axis_tag + canonical_provenance.
    """

    per_pair_index: int
    frame_0_bytes: bytes
    frame_1_bytes: bytes
    per_pair_metadata_bytes: bytes
    atick_redlich_loss: CooperativeReceiverOutput
    predicted_axis_decomposition: dict  # AxisDecomposition.as_dict()


# ---------------------------------------------------------------------------
# Per-substrate adapter Protocol (Catalog #290 FORK_BECAUSE_PRINCIPLED_MISMATCH)
# ---------------------------------------------------------------------------

@runtime_checkable
class SubstrateGrammarAdapter(Protocol):
    """Protocol for per-substrate archive-grammar slot injection.

    Per Catalog #357 dual-tier canonical contract sister discipline; each
    per-substrate adapter under `tac.substrates._shared.synthesize_frame_emission_atick_redlich_adapters/`
    implements this Protocol.

    Per HNeRV parity L3 + UNIQUE-AND-COMPLETE-PER-METHOD: each adapter is
    UNIQUE per substrate (PR110 V14 cascade / PR101 PR95-family canonical /
    DQS1 top32_selective_decoderq); the Protocol is the canonical interface.
    """

    substrate_id: str

    def inject_synthesize_frame_metadata(
        self,
        archive_bytes: bytes,
        synthetic_frame_metadata_bytes: bytes,
    ) -> tuple[bytes, int]:
        """Insert synthetic-frame metadata into the substrate's archive slot.

        Returns:
            (modified_archive_bytes, per_axis_archive_bytes_delta).
        """
        ...


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_atick_redlich_cooperative_receiver_for_substrate(
    substrate_id: str,
    *,
    seg_scorer: torch.nn.Module,
    pose_scorer: torch.nn.Module,
) -> AtickRedlichReceiver:
    """Construct the per-substrate AtickRedlichReceiver adapter.

    Per UNIQUE-AND-COMPLETE-PER-METHOD: this canonical factory routes to the
    per-substrate adapter at
    `tac.substrates._shared.synthesize_frame_emission_atick_redlich_adapters.<substrate_id>`.

    Recognized substrate IDs per Slot L 3 symposia: pr110 / pr101 / dqs1.
    """
    ...


def synthesize_frame_emission_per_pair(
    *,
    per_pair_index: int,
    archive_bytes: bytes,
    rgb_0: torch.Tensor,
    rgb_1: torch.Tensor,
    gt_rgb_0: torch.Tensor,
    gt_rgb_1: torch.Tensor,
    cooperative_receiver: AtickRedlichReceiver,
    config: SynthesizeFrameEmissionConfig,
) -> SynthesizeFrameEmissionPerPairResult:
    """Synthesize one per-pair emission via Atick-Redlich cooperative-receiver.

    Per Catalog #311 ego-motion-conditioned predictive coding sister
    discipline + Slot L 3 symposia: this helper operationalizes the
    cooperative-receiver framing where SegNet+PoseNet IS the canonical
    receiver per Atick-Redlich 1990.

    Flow:
        1. Compute Atick-Redlich loss via canonical primitive (delegates
           to score_pair_components + apply_eval_roundtrip per Catalog #164).
        2. Generate synthetic frames optimized for cooperative-receiver MI
           bound (per-pixel boundary-class exploitation).
        3. Per-substrate adapter injects synthetic-frame metadata bytes
           into archive grammar slot.
        4. Emit AxisDecomposition per Catalog #356 with canonical Provenance
           per Catalog #323.

    Returns:
        SynthesizeFrameEmissionPerPairResult with full byte-stable emission
        + Tier A canonical-routing markers per Catalog #341.
    """
    ...


def verify_synthesize_frame_emission_byte_stability(
    synthesize_result: SynthesizeFrameEmissionPerPairResult,
    *,
    rerun_config: SynthesizeFrameEmissionConfig,
    archive_bytes: bytes,
    rgb_0: torch.Tensor,
    rgb_1: torch.Tensor,
    gt_rgb_0: torch.Tensor,
    gt_rgb_1: torch.Tensor,
    cooperative_receiver: AtickRedlichReceiver,
) -> bool:
    """Verify byte-stability per Catalog #266 sister discipline + Dim 7.

    Re-runs the emission with the same inputs + config; compares byte output
    bit-exactly. Returns True if byte-stable, False otherwise.
    """
    ...


__all__ = [
    "SynthesizeFrameEmissionConfig",
    "AtickRedlichReceiver",
    "SynthesizeFrameEmissionPerPairResult",
    "SubstrateGrammarAdapter",
    "build_atick_redlich_cooperative_receiver_for_substrate",
    "synthesize_frame_emission_per_pair",
    "verify_synthesize_frame_emission_byte_stability",
]
```

## 7. Predicted ΔS band per cell (Catalog #296)

Per the 3 Slot L symposia + Dykstra-feasibility canonical 4-axis polytope at `tac.dykstra_pareto_solver` (Catalog #372):

| # | Cell | Predicted band | Source | validation_status |
|---|---|---|---|---|
| 1 | PR110 × SYNTHESIZE_FRAME | [-0.0025, +0.0015] | Slot L SYMPOSIUM 1 (Dykstra CO-LEAD verdict §4) | pending_post_training per Catalog #324 |
| 2 | PR101 × SYNTHESIZE_FRAME | [-0.0025, +0.0040] (EXPANDED) | Slot L SYMPOSIUM 2 (PR95Author + Assumption-Adversary binding revision per HNeRV parity L3 grammar overhead) | pending_post_training per Catalog #324 |
| 3 | DQS1 × SYNTHESIZE_FRAME | [-0.0025, +0.0015] | Slot L SYMPOSIUM 3 (Dykstra CO-LEAD verdict §4; same band as PR110 — selective decoder grammar more flexible than PR101) | pending_post_training per Catalog #324 |

Per Catalog #296 (forbidden symposium-band-prediction-without-Dykstra-feasibility-check): the 3 bands are Dykstra-feasible per the canonical 4-axis polytope; sister probe-disambiguator paths enumerated per Catalog #125 hook #6 (macOS-CPU-advisory probe per Contrarian binding revision; PR95-family grammar audit per PR101 Contrarian binding revision; DISPATCH ORDERING per DQS1 Contrarian binding revision).

## 8. Tier A canonical-routing markers per Catalog #341 (binding)

```
predicted_delta_adjustment = 0.0        # observability-only; ranker MUST NOT apply non-zero delta from this scaffold
promotable = False                       # NON-PROMOTABLE by construction; promotion gated by paired-CUDA empirical anchor per Catalog #246
axis_tag = "[predicted]"                 # NOT [contest-CPU]/[contest-CUDA]; predicted from Slot L 3 symposia + Atick-Redlich first-principles
evidence_grade_per_row = "[predicted]"
score_claim = False
ready_for_exact_eval_dispatch = False    # operator-routable to paired-CUDA per Catalog #246 after reactivation criteria land
```

## 9. 6-hook wire-in declaration per Catalog #125 (binding for THIS module)

- **hook #1 sensitivity-map**: ACTIVE — per-pair Atick-Redlich receiver-conditioned MI loss surfaces per-pair sensitivity via `tac.sensitivity_map.*` consumers; per-axis decomposition per Catalog #356.
- **hook #2 Pareto constraint**: ACTIVE — canonical 4-axis polytope at `tac.dykstra_pareto_solver` per Catalog #372; per-axis AxisDecomposition feeds the Dykstra alternating-projections constraint.
- **hook #3 bit-allocator**: ACTIVE — SYNTHESIZE_FRAME metadata emission is a RATE-axis operator; predicted_archive_bytes_delta feeds per-pair bit reallocation per Catalog #356.
- **hook #4 cathedral autopilot dispatch**: ACTIVE — auto-discovered per Catalog #335 cathedral consumer canonical contract via sister `tac.cathedral_consumers.synthesize_frame_emission_consumer/` (TBD landing in Steps 4-7 of Slot L 8-step cascade); consumed via Catalog #379 3-metric trichotomy ranking; canonical-routing markers per Catalog #341 prevent ranker non-zero delta application.
- **hook #5 continual-learning posterior**: ACTIVE — canonical equation `synthesize_frame_via_atick_redlich_cooperative_receiver_v1` (FORMALIZATION_PENDING per Catalog #344) auto-recalibrates per Catalog #371 fires `when_3+_new_empirical_anchors_in_domain` upon Step 5-7 paired-CUDA empirical anchors landing.
- **hook #6 probe-disambiguator**: ACTIVE — macOS-CPU-advisory probe per Contrarian binding revision IS the canonical disambiguator between [predicted] α=1.10 vs empirical α; sister Rao-Ballard alternative framing per RaoBallard verdict.

## 10. Cargo-cult unwind paths enumerated (per Catalog #303 + #313)

Per the cargo-cult audit §3:

| Cargo-cult assumption | Unwind path | Reactivation criterion |
|---|---|---|
| A8: 3-adapter scope structurally compatible with future 4th sister cell | Use `Protocol`-based per-substrate adapter interface (DESIGNED INTO helper from byte zero per §6 Public API) | If 4th sister cell lands future-cap-window (e.g. PR106 × SYNTHESIZE_FRAME), add NEW adapter file WITHOUT modifying SHARED helper signature |
| A2 OPERATIONAL-RESERVATION: synthetic-frame metadata fits in archive grammar slot | Per-substrate $0 grammar audit BEFORE recipe lands (per PR101 Contrarian binding revision; PR110 + DQS1 grammar slack expected per Slot L symposia) | If grammar slack insufficient: per Catalog #307 IMPLEMENTATION-LEVEL falsification (paradigm INTACT); reactivation = author NEW archive section per HNeRV parity L4 inflate.py ≤200 LOC budget |

## 11. Hard constraints honored

- ✅ $0 paid GPU spend (DESIGN-ONLY; symposium memos + canonical helper specification)
- ✅ READ-ONLY against canonical surfaces (only NEW design memo + NEW helper file + NEW tests)
- ✅ NO `gh pr create` invocations
- ✅ ZERO Claude/Anthropic in PR-facing surfaces (NONE produced; design memo is research-internal)
- ✅ APPEND-ONLY HISTORICAL_PROVENANCE per Catalog #110/#113 (NEW design memo + NEW helper + NEW tests; zero mutation of existing files)
- ✅ Catalog #340 sister-checkpoint guard PROCEED (Slot O + Slot Q + Slot R parallel; DISJOINT scope at `src/tac/substrates/_shared/synthesize_frame_emission_atick_redlich.py` + tests + design memo vs Slot O `src/tac/cathedral_consumers/wave_n_plus_48_hygiene_lookup_consumer/` + canonical_equations_registry write vs Slot Q `.omx/research/probe_pr110_macos_cpu_advisory_*` + `probe_pr101_pr95_family_grammar_audit_*`)
- ✅ Catalog #206 checkpoint discipline (3+ checkpoints across the 4-phase cascade)
- ✅ Catalog #287 placeholder-rationale rejection (all rationales substantive ≥4 chars + non-placeholder)
- ✅ Catalog #299 quota brake (current 381 well under 400; NO new catalog # claimed; reuses existing Catalog #325/#344 framework)
- ✅ Catalog #379 cathedral META-orchestrator routing (consumed via canonical 3-metric trichotomy ranking; 3 SUPER_ADDITIVE cells = highest-EV-shortest-WC frontier-breaking candidates)
- ✅ Catalog #341 Tier A canonical-routing markers
- ✅ Catalog #356 per-axis decomposition canonical Provenance
- ✅ Catalog #323 canonical Provenance umbrella
- ✅ Catalog #220 operational mechanism (`runtime_overlay_consumed=True`)
- ✅ Catalog #272 distinguishing-feature integration contract per per-substrate adapter
- ✅ Catalog #295 PYTHONPATH self-containment (NUMPY-PORTABLE inflate; no MLX dep at inflate time per 8th MLX-first standing directive)
- ✅ Catalog #205 canonical select_inflate_device (delegated to canonical helper)
- ✅ Catalog #146 contest-compliant inflate runtime template (delegated to canonical helper)
- ✅ NO DUPLICATE CODE per Wave N+46 canonical anti-pattern (EXTENDS `tac.codec.cooperative_receiver.atick_redlich` rather than re-implementing)
- ✅ "iterate not force" — NO CLAUDE.md amendment; NO paid dispatch; helper specification only

## Cross-references

- `[[slot-l-slot-h-top-3-super-additive-per-substrate-symposium-prep-landed-20260529]]` (parent landing memo; SOURCE 3 symposia)
- `council_per_substrate_symposium_pr110_x_synthesize_frame_super_additive_20260529.md` (SYMPOSIUM 1)
- `council_per_substrate_symposium_pr101_x_synthesize_frame_super_additive_20260529.md` (SYMPOSIUM 2)
- `council_per_substrate_symposium_dqs1_x_synthesize_frame_super_additive_20260529.md` (SYMPOSIUM 3)
- `tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss` (canonical primitive consumed by this helper)
- `tac.codec.cooperative_receiver.atick_redlich.AtickRedlichWeights` (canonical Lagrangian weights)
- `tac.codec.cooperative_receiver.atick_redlich.CooperativeReceiverOutput` (canonical loss output)
- `tac.substrates.score_aware_common.score_pair_components` (canonical scorer-loss helper; delegated to internally)
- `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` (canonical eval-roundtrip; delegated to internally)
- `tac.substrates._shared.inflate_runtime` (canonical inflate runtime helpers; select_inflate_device + raw_output_path + write_rgb_pair_to_raw)
- `tac.substrates._shared.numpy_portable_inflate` (canonical numpy-portable inflate per 8th MLX-first standing directive)
- `tac.cathedral.consumer_contract.AxisDecomposition` (canonical per-axis decomposition per Catalog #356)
- `tac.score_composition.compose_score_from_axes` (canonical score composition per Catalog #356)
- `tac.provenance.builders.build_provenance_for_predicted` (canonical Provenance per Catalog #323)
- `tac.canonical_equations.update_equation_with_empirical_anchor` (Catalog #344 canonical equation evolution)
- `tac.probe_outcomes_ledger.register_probe_outcome` (Catalog #313 canonical 4-layer pattern)
- Atick & Redlich 1990 "Towards a Theory of Early Visual Processing" (canonical reference)
- Rao & Ballard 1999 "Predictive coding in the visual cortex" (alternative framing per RaoBallard verdict)
- Catalogs #110/#113/#125/#127/#131/#138/#146/#164/#205/#206/#220/#229/#240/#243/#246/#266/#270/#272/#287/#290/#294/#295/#296/#300/#303/#305/#307/#309/#311/#313/#316/#317/#321/#322/#323/#324/#325/#335/#340/#341/#343/#344/#346/#355/#356/#357/#371/#372/#376/#379

## Mission contribution per Catalog #300

**frontier_breaking_enabler**: this SHARED substrate module is the foundation for Steps 4-7 of Slot L 8-step operator-routable cascade (Step 4: recipe YAMLs per substrate; Steps 5-7: paired-CUDA dispatches at $0.18 total envelope per Catalog #246; Step 8: canonical equation registration per Catalog #344 after 3 empirical anchors land). Without the SHARED helper, the 3 sister cells cannot dispatch coherently per UNIQUE-AND-COMPLETE-PER-METHOD operating mode. The 14-day Catalog #325 window (CLOSES 2026-06-12T06:15Z) is the canonical deadline for Steps 4-7 to land paid-dispatch empirical anchors.
