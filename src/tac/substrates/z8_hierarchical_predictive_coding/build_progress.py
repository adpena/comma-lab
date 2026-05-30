# SPDX-License-Identifier: MIT
"""Z8 hierarchical predictive coding Phase 2 build progress tracking surface.

THIS file IS the tracking surface for the Z8 multi-week coherent build per
operator 2026-05-29 binding *"ensure we have a way of tracking this to
completion"*. The canonical pattern lives IN SOURCE — not in TaskList
(1531+ entries, mostly stale per operator standing directive), not in
``.omx/research/`` memos (which become orphan signal per *"all memos must
be implemented"* standing directive). The operator reads current build
state by opening this file and looking at ``Z8_PHASE_2_BUILD_MILESTONES``.

The behavior IS the tracking: the milestone field updates BECAUSE the code
shipped. Each milestone's ``status``, ``landed_commit_sha``, and
``landed_at_utc`` fields update in the SAME commit as the work it tracks.
Tests verify milestone status matches implementation reality (a milestone
that says ``landed`` but whose ``acceptance_criteria`` tests fail is a
phantom-progress bug — sister of the phantom-score bug class per Catalog
#321 / #322 / #323).

This pattern generalizes: future coherent multi-week substrate builds
(DreamerV3-pure / TT5L / NIRVANA / SC++ / etc.) inherit by adding their
own ``<substrate>/build_progress.py`` with a similarly-shaped milestone
tuple. No new Catalog # gate, no new cathedral consumer, no new TaskList
category. If 3+ substrates eventually use the pattern, a cross-substrate
aggregator may be considered THEN — premature design is itself the
apparatus-accretion failure mode.

Cross-references:

- Sister memory ``z8-phase-2-build-tracking-in-source-not-tasklist-not-
  memos-20260529`` (the apparatus-adjustment memory this file
  operationalizes).
- Sister memory ``z8-hierarchical-predictive-coding-binding-first-active-
  build-target-yousfi-grounded-20260529`` (the build target this tracks).
- Sister module ``binding_contract.py`` (the per-level Protocols +
  dataclasses milestone M1 lands; this file's M1 references it).
- Operator visibility: open this file. The ``Z8_PHASE_2_BUILD_MILESTONES``
  tuple at the bottom IS the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BuildMilestoneStatus(str, Enum):
    """Per-milestone lifecycle state.

    The canonical transitions are pending -> in_progress -> landed.
    superseded is for milestones replaced by a sister formulation (rare;
    requires APPEND-ONLY HISTORICAL_PROVENANCE per Catalog #110/#113 if
    the milestone landed first).
    """

    PENDING = "pending"
    """Work not yet started; predecessor milestones may still be open."""

    IN_PROGRESS = "in_progress"
    """Work currently underway; commit not yet landed."""

    LANDED = "landed"
    """Code shipped + tests pass + milestone updated in same commit."""

    SUPERSEDED = "superseded"
    """Replaced by a sister milestone; preserved per HISTORICAL_PROVENANCE."""


@dataclass(frozen=True)
class BuildMilestone:
    """Single Phase-2 build milestone for the Z8 substrate.

    Frozen so the milestone tuple is immutable at module level; updates
    happen by replacing the tuple entry (full dataclass replacement, not
    field mutation). This makes every change reviewable as a code diff.

    Attributes:
        milestone_id: Short canonical slug (e.g.,
            ``binding_contract_landed``). Used as the stable identifier
            across the tuple; must be unique within
            ``Z8_PHASE_2_BUILD_MILESTONES``.
        description: One-sentence operator-readable summary.
        acceptance_criteria: Tuple of conditions that must hold for
            ``status == landed``. Each entry is a string; ideally each
            also names a test function (e.g., ``test_binding_contract_
            level_dimension_contract_invariants``) that asserts the
            condition. Tests verify consistency.
        status: Current lifecycle state (see ``BuildMilestoneStatus``).
        landed_commit_sha: Canonical-serializer commit that landed the
            milestone (None until ``status == landed``; may also be None
            briefly for the commit that lands the milestone itself,
            because the sha is self-referential at landing time —
            backfilled by the next commit).
        landed_at_utc: ISO timestamp when status flipped to ``landed``.
            None until landed.
        predecessor_milestone_ids: Tuple of milestone_ids that must be
            ``landed`` before this milestone can transition to
            ``in_progress``. Enforced by ``validate_milestone_tuple``.
        notes: Free-form one-to-two-line context. If a milestone's notes
            balloon past 2 lines, the work is probably incoherent and
            should be split.
    """

    milestone_id: str
    description: str
    acceptance_criteria: tuple[str, ...]
    status: BuildMilestoneStatus
    landed_commit_sha: str | None = None
    landed_at_utc: str | None = None
    predecessor_milestone_ids: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.milestone_id:
            raise ValueError("milestone_id must be non-empty")
        if not self.description:
            raise ValueError(
                f"milestone {self.milestone_id} description must be non-empty"
            )
        if len(self.acceptance_criteria) < 1:
            raise ValueError(
                f"milestone {self.milestone_id} requires "
                f">= 1 acceptance criterion"
            )
        # Landed + landed_at_utc consistency. landed_commit_sha may be None
        # for the commit that lands the milestone itself (self-referential
        # backfill in next commit).
        if self.status == BuildMilestoneStatus.LANDED:
            if self.landed_at_utc is None:
                raise ValueError(
                    f"milestone {self.milestone_id} status=landed "
                    f"requires landed_at_utc to be set"
                )
        if self.status != BuildMilestoneStatus.LANDED:
            if self.landed_commit_sha is not None:
                raise ValueError(
                    f"milestone {self.milestone_id} has "
                    f"landed_commit_sha={self.landed_commit_sha} but "
                    f"status={self.status.value}; sha implies landed"
                )
            if self.landed_at_utc is not None:
                raise ValueError(
                    f"milestone {self.milestone_id} has "
                    f"landed_at_utc={self.landed_at_utc} but "
                    f"status={self.status.value}; timestamp implies landed"
                )


def validate_milestone_tuple(
    milestones: tuple[BuildMilestone, ...],
) -> None:
    """Cross-milestone consistency checks.

    Raises ValueError on:
        - Duplicate milestone_id.
        - predecessor_milestone_ids referencing non-existent ids.
        - status=in_progress or landed when predecessors are not landed.

    Tests call this at import time to catch tracking-drift bugs at
    collection time.
    """
    ids_seen: set[str] = set()
    by_id: dict[str, BuildMilestone] = {}
    for milestone in milestones:
        if milestone.milestone_id in ids_seen:
            raise ValueError(
                f"duplicate milestone_id: {milestone.milestone_id}"
            )
        ids_seen.add(milestone.milestone_id)
        by_id[milestone.milestone_id] = milestone

    for milestone in milestones:
        for pred_id in milestone.predecessor_milestone_ids:
            if pred_id not in by_id:
                raise ValueError(
                    f"milestone {milestone.milestone_id} references "
                    f"unknown predecessor {pred_id}"
                )
        if milestone.status in (
            BuildMilestoneStatus.IN_PROGRESS,
            BuildMilestoneStatus.LANDED,
        ):
            for pred_id in milestone.predecessor_milestone_ids:
                pred = by_id[pred_id]
                if pred.status != BuildMilestoneStatus.LANDED:
                    raise ValueError(
                        f"milestone {milestone.milestone_id} "
                        f"status={milestone.status.value} but predecessor "
                        f"{pred_id} status={pred.status.value}; "
                        f"predecessors must be landed first"
                    )


def get_landed_milestones(
    milestones: tuple[BuildMilestone, ...],
) -> tuple[BuildMilestone, ...]:
    """Filter to landed milestones (chronological by tuple order)."""
    return tuple(
        m for m in milestones if m.status == BuildMilestoneStatus.LANDED
    )


def get_pending_milestones(
    milestones: tuple[BuildMilestone, ...],
) -> tuple[BuildMilestone, ...]:
    """Filter to milestones still pending (not landed, not in_progress)."""
    return tuple(
        m for m in milestones if m.status == BuildMilestoneStatus.PENDING
    )


def get_in_progress_milestones(
    milestones: tuple[BuildMilestone, ...],
) -> tuple[BuildMilestone, ...]:
    """Filter to milestones currently in progress."""
    return tuple(
        m for m in milestones if m.status == BuildMilestoneStatus.IN_PROGRESS
    )


def get_next_actionable_milestones(
    milestones: tuple[BuildMilestone, ...],
) -> tuple[BuildMilestone, ...]:
    """Return pending milestones whose predecessors are all landed.

    These are the milestones that can transition to ``in_progress`` next.
    Operator routes by picking among these.
    """
    by_id = {m.milestone_id: m for m in milestones}
    actionable: list[BuildMilestone] = []
    for milestone in milestones:
        if milestone.status != BuildMilestoneStatus.PENDING:
            continue
        if all(
            by_id[pred_id].status == BuildMilestoneStatus.LANDED
            for pred_id in milestone.predecessor_milestone_ids
        ):
            actionable.append(milestone)
    return tuple(actionable)


def render_progress_summary(
    milestones: tuple[BuildMilestone, ...] | None = None,
) -> str:
    """Render a markdown table summary for occasional briefing.

    The tuple itself is the source of truth; this helper just produces a
    human-readable view. Defaults to Z8_PHASE_2_BUILD_MILESTONES when
    called without args.
    """
    if milestones is None:
        milestones = Z8_PHASE_2_BUILD_MILESTONES
    lines = [
        "| Status | ID | Description | Landed at | Commit |",
        "|---|---|---|---|---|",
    ]
    for m in milestones:
        commit = m.landed_commit_sha[:12] if m.landed_commit_sha else "—"
        landed = m.landed_at_utc or "—"
        lines.append(
            f"| {m.status.value} | `{m.milestone_id}` | "
            f"{m.description} | {landed} | `{commit}` |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The canonical Z8 Phase-2 build milestone tuple.
#
# Operator visibility: this IS the report. Open this file, scroll to here,
# read the tuple. Status updates land in the SAME commit as the work they
# describe. The order is build-sequence-canonical (predecessors before
# dependents); the tuple is NOT sorted by status.
#
# Twelve milestones span:
#   M1-M3: foundation (binding contract + tracking surface + canonical
#          consistency with existing config); ALL landed in the initial
#          binding-contract commit.
#   M4-M8: per-piece Phase-2 implementations (Mamba-2 SSD / Mallat full
#          DWT / Wyner-Ziv full / scorer-sensitivity map / score-aware
#          loss); each pending; each gets its own commit batch.
#   M9-M11: integration (full_main trainer / inflate / L1 smoke); land
#          after M4-M8 satisfy contract.
#   M12: terminal validation (paired-CUDA dispatch crosses sub-0.189
#        threshold); the operator's submission-eligibility gate.
# ---------------------------------------------------------------------------


Z8_PHASE_2_BUILD_MILESTONES: tuple[BuildMilestone, ...] = (
    BuildMilestone(
        milestone_id="binding_contract_landed",
        description=(
            "Per-level binding contract defined as Python Protocols + "
            "frozen dataclasses in binding_contract.py."
        ),
        acceptance_criteria=(
            "binding_contract module imports without error",
            "LevelDimensionContract __post_init__ rejects invalid shapes",
            "HierarchyBindingContract enforces contiguous level_index",
            "DeterministicStateUpdate Protocol declares state_dim + initial_state + step",
            "WaveletPartition Protocol declares decompose + recompose",
            "WynerZivTopLevelCoder Protocol declares encode + decode",
            "ScoreAwareLevelLoss Protocol declares per_level_loss",
            "build_canonical_contract_from_config produces valid contract "
            "from Z8HierarchicalConfig defaults",
        ),
        status=BuildMilestoneStatus.LANDED,
        landed_commit_sha=None,  # self-referential; backfilled next commit
        landed_at_utc="2026-05-29T00:00:00Z",
        notes=(
            "M1 + M2 + M3 land together in the initial Phase-2 binding "
            "commit. The contract is the source of truth M4-M8 build against."
        ),
    ),
    BuildMilestone(
        milestone_id="build_progress_surface_landed",
        description=(
            "In-source build tracking surface (this file) lands with "
            "BuildMilestone dataclass + Z8_PHASE_2_BUILD_MILESTONES tuple."
        ),
        acceptance_criteria=(
            "build_progress module imports without error",
            "validate_milestone_tuple passes on Z8_PHASE_2_BUILD_MILESTONES",
            "BuildMilestone __post_init__ rejects landed status without "
            "landed_at_utc",
            "BuildMilestone __post_init__ rejects sha or timestamp on "
            "non-landed status",
            "predecessor consistency enforced (in_progress/landed "
            "requires predecessors landed)",
            "render_progress_summary produces non-empty markdown table",
        ),
        status=BuildMilestoneStatus.LANDED,
        landed_commit_sha=None,
        landed_at_utc="2026-05-29T00:00:00Z",
        notes=(
            "Tracking surface lives in source per "
            "'z8-phase-2-build-tracking-in-source-not-tasklist-not-memos' "
            "memory; not memos, not TaskList."
        ),
    ),
    BuildMilestone(
        milestone_id="existing_config_satisfies_per_level_contract",
        description=(
            "The existing Z8HierarchicalConfig at L0 SCAFFOLD satisfies "
            "the per-level shape contract via build_canonical_contract_"
            "from_config; tests verify."
        ),
        acceptance_criteria=(
            "build_canonical_contract_from_config(Z8HierarchicalConfig()) "
            "succeeds without ValueError",
            "produced contract has num_levels == 3 (canonical default)",
            "per-level categorical_index_bytes_per_pair matches sister "
            "Z8HierarchicalConfig.total_latent_packing_bytes_per_pair sum",
            "wavelet_subband_shapes halve correctly from eval_size",
        ),
        status=BuildMilestoneStatus.LANDED,
        landed_commit_sha=None,
        landed_at_utc="2026-05-29T00:00:00Z",
        predecessor_milestone_ids=("binding_contract_landed",),
        notes=(
            "Verifies the contract is consistent with existing L0 SCAFFOLD "
            "config; the foundation for M4-M8 to build against."
        ),
    ),
    BuildMilestone(
        milestone_id="mamba_2_adapter_binds_canonical_primitive_to_protocol",
        description=(
            "Z8Mamba2DeterministicStateUpdate adapter wraps the canonical "
            "tac.optimization.mamba2_predictor.Mamba2Predictor primitive "
            "to satisfy DeterministicStateUpdate Protocol. Adapter pivot "
            "per operator binding-first methodology (2026-05-29): reuse "
            "the existing Wave-4-audited Mamba-2 primitive + add canonical "
            "step_externalized_state surface to it; do NOT reimplement "
            "the selective state-space recurrence inside Z8."
        ),
        acceptance_criteria=(
            "Z8Mamba2DeterministicStateUpdate satisfies "
            "DeterministicStateUpdate Protocol (runtime_checkable "
            "isinstance returns True)",
            "state_dim property matches contract.deterministic_state_dim",
            "Mamba2Predictor.step_externalized_state landed as canonical "
            "sister surface (iteration of underlying primitive per "
            "operator 'iterate and optimize underlying pieces as well' "
            "directive)",
            "Backend honest classification: adapter pins "
            "REFERENCE_TORCH_BACKEND (Mamba-1 S6 per Wave 4 fidelity "
            "audit landed 2026-05-29); true Mamba-2 SSD upgrade deferred "
            "to sister milestone when upstream mamba_ssm.utils."
            "inference_params single-step surface is wired",
            "Mathematical grounding: structured state shape (B, d_inner, "
            "d_state) per Dao & Gu 2024 §3; adapter reshapes to flat "
            "(B, state_dim) for Protocol contract; reshape is byte-stable "
            "and gradient-preserving",
        ),
        status=BuildMilestoneStatus.LANDED,
        landed_at_utc="2026-05-29T00:00:00Z",
        predecessor_milestone_ids=(
            "binding_contract_landed",
            "existing_config_satisfies_per_level_contract",
        ),
        notes=(
            "Adapter pivot per operator 2026-05-29 'binding-first + iterate "
            "underlying pieces'. Reference cell honestly = Mamba-1 S6 per "
            "Wave 4 fidelity audit; upstream mamba_ssm.Mamba2 IS true SSD "
            "but lacks public single-step inference-params surface. Sub-"
            "milestone (true-SSD upgrade) queued post-MLX-pyhsm closure."
        ),
    ),
    BuildMilestone(
        milestone_id="mallat_full_dwt_replaces_sum_pool_proxy",
        description=(
            "Z8MallatDaubechiesPartition adapter wraps the canonical "
            "tac.symposium_impls.daubechies_wavelet_codec 1D primitive "
            "as a 2D separable Daubechies-4 transform per Mallat 1989 "
            "§7.7; satisfies WaveletPartition Protocol. Detail subbands "
            "emitted as structured WaveletDetail2D(lh, hl, hh) frozen "
            "dataclass per Mallat §7.5 exact-reconstruction (collapsing "
            "to one tensor would either lose info or violate shape "
            "contract). Sum-pool proxy in mlx_renderer.py REMAINS for "
            "L0 SCAFFOLD; the adapter is the Phase 2 binding the Z8 "
            "hierarchy uses going forward."
        ),
        acceptance_criteria=(
            "Z8MallatDaubechiesPartition satisfies WaveletPartition "
            "Protocol via @runtime_checkable isinstance() check",
            "decompose + recompose round-trip to within Daubechies-4 "
            "reconstruction tolerance (atol 1e-6; empirically atol "
            "~1e-12 at fp64 per Mallat §7.5 perfect-reconstruction)",
            "per-level approximation subband shape (B, H/2, W/2, C) "
            "matches L0 sum-pool's contract",
            "WaveletDetail2D(lh, hl, hh) carries three NON-ZERO high-"
            "pass subbands (vs sum-pool proxy's approximation-only)",
            "framework-agnostic: accepts MLX / PyTorch / numpy via "
            "np.asarray intermediate per Catalog #317 portability",
        ),
        status=BuildMilestoneStatus.LANDED,
        landed_at_utc="2026-05-29T00:00:00Z",
        predecessor_milestone_ids=("binding_contract_landed",),
        notes=(
            "Adapter binds canonical tac.symposium_impls."
            "daubechies_wavelet_codec primitive per binding-first "
            "methodology (operator 2026-05-29 'iterate underlying "
            "pieces as well'). 2D separable construction is canonical "
            "Mallat 1989 §7.7 — composition of the canonical 1D "
            "primitive, not a fresh design. Per Catalog #290 the only "
            "FORK is the WaveletDetail2D dataclass to preserve all "
            "three high-pass subbands needed for exact reconstruction. "
            "L0 sum-pool _mallat_sum_pool_2x_nhwc remains in "
            "mlx_renderer.py for backward compat; Phase 3 NaN-burn-in "
            "code can opt into either via the Protocol-typed slot."
        ),
    ),
    BuildMilestone(
        milestone_id="wyner_ziv_full_top_level_coder_landed",
        description=(
            "Full Wyner-Ziv (1976) coding at top level (level_index == "
            "num_levels - 1) replaces sketch; satisfies "
            "WynerZivTopLevelCoder Protocol."
        ),
        acceptance_criteria=(
            "WynerZivTopLevelCoder implementation satisfies encode + "
            "decode round-trip within rate-distortion bound",
            "encoded payload bytes <= bit_budget_estimate target",
            "decode-side reconstruction uses side_info shape matching "
            "contract.wyner_ziv_top_level_side_info_shape",
            "compression ratio improvement over unconditional coding "
            "verified empirically on representative top-level state"
        ),
        status=BuildMilestoneStatus.PENDING,
        predecessor_milestone_ids=(
            "binding_contract_landed",
            "mallat_full_dwt_replaces_sum_pool_proxy",
        ),
        notes=(
            "Wyner-Ziv top-level coder consumes wavelet-reconstructed "
            "side info; depends on Mallat full DWT."
        ),
    ),
    BuildMilestone(
        milestone_id="empirical_scorer_sensitivity_map_v1_landed",
        description=(
            "Canonical M7 scorer-sensitivity-map helper at "
            "tac.substrates.z8_hierarchical_predictive_coding."
            "scorer_sensitivity_map; produces (B, C, H, W) per-level "
            "sensitivity tensor that M8 ScoreAwareLevelLoss consumes "
            "per binding_contract.py:451-472. Honest 4-path production: "
            "Path A (uniform_sensitivity_map_for_level) LANDED satisfies "
            "the M8 L2-reduction invariant; Path B2 "
            "(empirical_sensitivity_map_from_master_gradient) LANDED "
            "2026-05-30 (Phase D Yousfi-ordered wire-in) consuming Phase A "
            "canonical extract_M_pixel + broadcast_sensitivity_map_to_channels "
            "at commit 8a95c9cc5; Path B (Slot GGG empirical anchor) + Path C "
            "(Yousfi UNIWARD-analog finite-difference) remain NotImplementedError "
            "stubs with reactivation criteria pinned per CLAUDE.md 'Forbidden "
            "premature KILL'. M7 unblocks M8 design via Path A baseline + "
            "Path B2 operational consumption of master-gradient empirical "
            "scorer-axis sensitivity (the canonical Yousfi-grounded source "
            "the M8 Protocol docstring names verbatim)."
        ),
        acceptance_criteria=(
            "Z8ScorerSensitivityMap dispatcher importable; "
            "build_z8_scorer_sensitivity_map_for_level convenience builder "
            "importable; ScorerSensitivityMapSource enum has 4 members "
            "(UNIFORM, EMPIRICAL_SLOT_GGG, FINITE_DIFFERENCE_UNIWARD, "
            "EMPIRICAL_FROM_MASTER_GRADIENT); uniform_sensitivity_map_for_level "
            "returns (B, C, H, W) all-ones at the level's wavelet_subband_shape; "
            "reduces loss to standard L2 per M8 Protocol invariant when "
            "broadcast into ScoreAwareLevelLoss.per_level_loss; "
            "empirical_sensitivity_map_from_master_gradient consumes a "
            "ContestGradientTensor or InflatedGradientTensor via Phase A "
            "extract_M_pixel + broadcast_sensitivity_map_to_channels and "
            "raises ResolutionMismatchAwaitingPerLevelProjectionError when "
            "gradient (H, W) does not match level.wavelet_subband_shape "
            "(Phase C unblocks resolution projection); Path B + Path C "
            "raise EmpiricalSensitivityMapNotYetLandedError with actionable "
            "reactivation criteria in the message; honest data-domain "
            "finding documented (master_gradient ledger is per-archive-byte "
            "not per-pixel — Path B2 reads ContestGradientTensor / "
            "InflatedGradientTensor not the byte ledger)",
        ),
        status=BuildMilestoneStatus.LANDED,
        landed_at_utc="2026-05-30T00:00:00Z",
        predecessor_milestone_ids=("binding_contract_landed",),
        notes=(
            "Path A + Path B2 LANDED at 2026-05-30; Path B / Path C honest "
            "stubs. Reactivation paths pinned in helper docstrings: Slot GGG "
            "per-pixel sister probe OR canonical Wyner-Ziv decoder-side "
            "PoseNet side-info (Path B); paid-GPU finite-difference via "
            "real PoseNet+SegNet (Path C); Phase C per-level resolution "
            "projection (Path B2 mismatched-resolution case). M8 design "
            "unblocked via Path A baseline; M8 OPERATIONAL via Path B2 "
            "consuming empirical master-gradient scorer-axis sensitivity. "
            "If Z8 trains with uniform sensitivity the loss reduces to "
            "per-level L2, which is the honest L0-equivalent baseline "
            "Yousfi would accept as 'empty prior'. Per Catalog #287: every "
            "claim is paired with adjacent source/citation evidence in "
            "helper docstrings."
        ),
    ),
    BuildMilestone(
        milestone_id="score_aware_level_loss_uniward_analog_landed",
        description=(
            "ScoreAwareLevelLoss Protocol implementation using the "
            "empirical sensitivity map at each Z8 hierarchy level."
        ),
        acceptance_criteria=(
            "ScoreAwareLevelLoss implementation satisfies "
            "per_level_loss(reconstruction, target, sensitivity_map)",
            "uniform sensitivity map (all-ones) reduces to standard L2 "
            "reconstruction loss",
            "non-uniform sensitivity map reweights per-pixel contribution "
            "proportionally",
            "per-level instances at each Z8 hierarchy level consume the "
            "sensitivity map at that level's resolution (downsampled "
            "where needed)",
        ),
        status=BuildMilestoneStatus.PENDING,
        predecessor_milestone_ids=(
            "binding_contract_landed",
            "empirical_scorer_sensitivity_map_v1_landed",
        ),
        notes=(
            "The Yousfi-grounding made concrete: cost function at each "
            "level uses empirical scorer-sensitivity weights, not "
            "generic L2."
        ),
    ),
    BuildMilestone(
        milestone_id="full_main_trainer_lifts_notimplementederror",
        description=(
            "Z8 _full_main lifts from NotImplementedError; trainer "
            "consumes binding contract + Mamba-2 SSD + Mallat full + "
            "Wyner-Ziv + score-aware loss in one coherent forward pass."
        ),
        acceptance_criteria=(
            "Z8 trainer's _full_main runs without raising "
            "NotImplementedError on real upstream/videos/0.mkv per "
            "Catalog #213",
            "produces archive bytes that fit declared "
            "total_categorical_index_bytes_per_pair * NUM_PAIRS budget",
            "per-pair training loss decreases over epochs (canonical "
            "convergence check)",
            "trainer routes through canonical scorer-preprocess helpers "
            "per Catalog #164",
            "no synthetic-noise smoke; real video frames only per Slot "
            "EEE META finding",
        ),
        status=BuildMilestoneStatus.PENDING,
        predecessor_milestone_ids=(
            "mamba_2_adapter_binds_canonical_primitive_to_protocol",
            "mallat_full_dwt_replaces_sum_pool_proxy",
            "wyner_ziv_full_top_level_coder_landed",
            "score_aware_level_loss_uniward_analog_landed",
        ),
        notes=(
            "The binding integration step. All four Phase-2 pieces must "
            "land before this can proceed."
        ),
    ),
    BuildMilestone(
        milestone_id="inflate_runtime_consumes_real_trained_weights",
        description=(
            "inflate.py consumes real trained weights from M9 archive; "
            "no synthetic frame generation per Catalog #369."
        ),
        acceptance_criteria=(
            "inflate.py loads weights from archive (no NotImplementedError)",
            "produced raw bytes match contest contract 3,662,409,600 "
            "(1164*874*1200*3) per Catalog #367",
            "inflate runtime LOC <= 200 per HNeRV parity L4",
            "PYTHONPATH self-contained per Catalog #295",
            "select_inflate_device canonical helper per Catalog #205",
        ),
        status=BuildMilestoneStatus.PENDING,
        predecessor_milestone_ids=("full_main_trainer_lifts_notimplementederror",),
        notes=(
            "Cascade C' WAVE-4 through WAVE-7 lesson: synthetic frame "
            "base = 8000x worse score. Real trained weights only."
        ),
    ),
    BuildMilestone(
        milestone_id="l1_macos_cpu_smoke_landed",
        description=(
            "L1 macOS-CPU advisory smoke on Z8 archive passes (sanity "
            "check that the full stack inflates without error on real "
            "video before paid CUDA dispatch)."
        ),
        acceptance_criteria=(
            "Z8 archive sha256 stable across two builds (deterministic)",
            "inflate.py produces canonical 3,662,409,600 bytes on macOS-"
            "CPU advisory smoke",
            "upstream/evaluate.py --device cpu produces finite score "
            "(NOT NaN, NOT inf, NOT > 100)",
            "result tagged [macOS-CPU advisory] per Catalog #192; NOT "
            "promotable to contest score claim",
            "lane registry L1 (impl_complete) + memory entry per Catalog "
            "#298 substrate retirement discipline",
        ),
        status=BuildMilestoneStatus.PENDING,
        predecessor_milestone_ids=(
            "inflate_runtime_consumes_real_trained_weights",
        ),
        notes=(
            "MLX-first per CLAUDE.md MLX portable-local-substrate "
            "authority; the cheap pre-paid-GPU smoke gate."
        ),
    ),
    BuildMilestone(
        milestone_id="paired_cuda_dispatch_crosses_sub_0_189_threshold",
        description=(
            "Paired CPU+CUDA Modal dispatch on Z8 archive crosses the "
            "operator's sub-0.189 submission threshold per Catalog #246 "
            "+ #343 canonical frontier pointer."
        ),
        acceptance_criteria=(
            "paired CPU [contest-CPU Linux x86_64] score < 0.189",
            "paired CUDA [contest-CUDA T4] score reported alongside per "
            "Catalog #246 + CLAUDE.md 'Submission auth eval — BOTH CPU "
            "AND CUDA' non-negotiable",
            "canonical frontier pointer .omx/state/canonical_frontier_"
            "pointer.json auto-updated per Catalog #343",
            "archive sha256 + size + runtime tree custody complete per "
            "Catalog #245 modal_call_id_ledger",
            "result registers PROCEED outcome in tac.probe_outcomes_"
            "ledger per Catalog #313",
            "operator-routable submission cascade unblocked (Phase 9 CLI "
            "exit 4)",
        ),
        status=BuildMilestoneStatus.PENDING,
        predecessor_milestone_ids=("l1_macos_cpu_smoke_landed",),
        notes=(
            "Terminal milestone. Crossing sub-0.189 is the operator's "
            "submission-eligibility gate per 2026-05-29 binding."
        ),
    ),
)


# Validate at import time so phantom-progress bugs surface immediately
# rather than at first audit.
validate_milestone_tuple(Z8_PHASE_2_BUILD_MILESTONES)


__all__ = [
    "BuildMilestoneStatus",
    "BuildMilestone",
    "validate_milestone_tuple",
    "get_landed_milestones",
    "get_pending_milestones",
    "get_in_progress_milestones",
    "get_next_actionable_milestones",
    "render_progress_summary",
    "Z8_PHASE_2_BUILD_MILESTONES",
]
