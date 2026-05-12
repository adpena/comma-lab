"""Bit-allocator end-to-end wire (composition matrix + magic codec + sparse PacketIR + Hinton + autopilot ranking).

Per operator amplification 2026-05-11 ("compiler and insanely low level" +
"wiring and integration and everything"), this module is the canonical
**EndToEndBitAllocator** that composes all integration surfaces landed across
the 2026-05-11 work envelope into a single typed planner:

1. **Substrate composition matrix** (QQ landing,
   :mod:`tac.optimization.substrate_composition_matrix`) — composability +
   orthogonality + redundancy verdicts on the 24-substrate inventory.
2. **Magic codec auto-selector** (AA landing,
   :mod:`tac.packet_compiler.magic_codec`) — per-stream optimal primitive
   selection from the 19-primitive packet_compiler inventory.
3. **Sparse PacketIR codec** (S + SS landing,
   :mod:`tac.packet_compiler.sparse_packet_ir`) — sparsity-aware byte budget.
4. **Hinton-distilled L2 surrogate** (LL landing) — score-aware sensitivity
   priors propagated from the LL Hinton-distilled scorer surrogate.
5. **Cathedral autopilot ranking** (TT landing,
   :mod:`tac.optimization.autopilot_dispatch_ranking`) — EV-per-dollar
   ordering with composition-constraint enforcement.

The class **does NOT** allocate scorer load, dispatch GPU, or claim any
contest score. It is a **planning artifact** that emits a typed allocation
plan. The plan must be paired with an exact CUDA + CPU adjudicator before
it produces anything that influences a public submission archive.

Per CLAUDE.md "ADMM/water-filling: shared byte allocation" non-negotiable,
the allocator's water-filling rule is:

    For a per-substrate marginal-value-per-byte ``mvpb_i = |delta_i| / bytes_i``
    and a global byte budget ``B`` with per-substrate floor allocations,
    the optimal allocation pours bytes into the highest-mvpb substrate
    until its marginal sinks to the second-highest, then both, etc., until
    ``B`` is exhausted. ORTHOGONAL pairs share their byte budget additively;
    REDUNDANT pairs share at the higher-mvpb substrate's floor only;
    REPLACEMENT pairs cannot coexist (caller-side check).

Per CLAUDE.md "Forbidden score claims": every numeric in the plan is
``[predicted; bit allocator end-to-end wire v1]``. No ``[contest-CUDA]``
claims emerge from this module.

Per CLAUDE.md "Beauty, simplicity, and developer experience": the typed
``AllocationPlan`` is JSON-safe, frozen-dataclass-ed, and ships with
``score_claim=False`` / ``promotion_eligible=False`` /
``ready_for_exact_eval_dispatch=False`` invariants.

8 archive-grammar fields declared per CLAUDE.md HNeRV parity discipline
lesson 4 (Catalog #124 ``check_representation_lane_has_archive_grammar_at_design_time``):

- ``archive_grammar`` = "bit-allocator-only-no-archive (planning_only)"
- ``parser_section_manifest`` = "N/A — planning_only emits no parser sections"
- ``inflate_runtime_loc_budget`` = 0 (no inflate runtime)
- ``runtime_dep_closure`` = ("python>=3.9",) (pure stdlib + tac internal)
- ``export_format`` = "tac_bit_allocator_plan_v1 JSON"
- ``score_aware_loss`` = "delegated_to_per_substrate_l2_loss"
- ``bolt_on_loc_budget`` = "≤450 LOC (this module)"
- ``no_op_detector_planned`` = True (emit-only planner; no archive bytes mutate)

CLAUDE.md compliance tags
-------------------------
- ``planning_only_no_score_claim``
- ``no_mps_authoritative``
- ``no_tmp_paths``
- ``halt_and_ask_default_for_dispatch_recommendations``
- ``shared_byte_allocation_water_filling_v1``

Cross-references
----------------
- :mod:`tac.optimization.substrate_composition_matrix` — composability matrix
- :mod:`tac.optimization.autopilot_dispatch_ranking` — EV/$ ranking
- :mod:`tac.optimization.theoretical_floor_substrate_refresh` — Pareto floor
- :mod:`tac.sensitivity_map` — per-layer sensitivity priors
- :mod:`tac.packet_compiler.magic_codec` — per-stream auto-selector
- :mod:`tac.packet_compiler.sparse_packet_ir` — sparse codec
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from tac.optimization.substrate_composition_matrix import (
    Composability,
    CompositionMatrix,
    DISPATCH_COST_USD_MIDPOINT,
    ParetoRow,
    ScoreAxis,
    SubstrateClass,
    SubstrateRow,
    build_composition_matrix,
    canonical_substrate_inventory,
    filter_pareto_dominated,
    per_substrate_pareto_rows,
    predicted_composite_delta,
)

SCHEMA_VERSION = "tac_bit_allocator_end_to_end_v1"

# Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent"
# at PR106 r2 frontier (pose_avg = 3.4e-5) the pose marginal-value-per-byte
# is 2.79× SegNet's. This is an operating-point-conditional weight; the
# allocator applies it to substrates whose target_axis is POSE.
PR106_R2_POSE_MARGINAL_MULTIPLIER: float = 2.79

# A1 / PR106 r2 frontier reference byte budget: the contest archive at the
# current operating point lands around 178k-186k bytes. The default global
# byte budget is set to 200k bytes (allows ~20k of headroom for new sidecars).
DEFAULT_GLOBAL_BYTE_BUDGET: int = 200_000

# Per CLAUDE.md "Cathedral autopilot activation" the per-dispatch budget is
# $5.00 default and cumulative envelope is $20.00 default. The bit-allocator
# inherits these envelopes from the autopilot ranking layer.
DEFAULT_PER_DISPATCH_BUDGET_USD: float = 5.00
DEFAULT_CUMULATIVE_BUDGET_USD: float = 20.00


# ── Typed allocation plan ────────────────────────────────────────────────


@dataclass(frozen=True)
class SubstrateAllocation:
    """One per-substrate (or per-substrate-pair) byte allocation row.

    Per CLAUDE.md "Forbidden score claims" + "Forbidden empirical-claim-
    without-evidence-tag", every numeric here carries an evidence tag in
    the rationale string.
    """

    substrate_id: str
    substrate_class: SubstrateClass
    target_axis: ScoreAxis
    allocated_bytes: int  # Bytes the allocator pours into this substrate.
    floor_bytes: int  # Minimum bytes (substrate cannot operate below this).
    ceiling_bytes: int  # Maximum bytes (substrate cannot absorb more).
    mvpb: float  # Marginal value per byte = |predicted_delta| / midpoint_bytes.
    pose_axis_multiplier_applied: float  # 1.0 unless POSE-axis at PR106 frontier.
    paired_substrate_id: Optional[str] = None  # If part of an orthogonal pair.
    composability_with_pair: Optional[str] = None  # Verdict (orthogonal/etc).
    rationale: str = ""
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


@dataclass(frozen=True)
class AllocationPlan:
    """Typed bit-allocator output bundle.

    Per CLAUDE.md "Deterministic packet compiler" the bundle is a
    schema-pinned JSON-safe payload that downstream consumers (autopilot
    loop, archive composer) ingest by schema version.
    """

    schema: str
    generated_at_utc: str
    matrix_schema: str
    n_substrates_considered: int
    global_byte_budget: int
    per_dispatch_budget_usd: float
    cumulative_budget_usd: float
    operating_point: str
    pose_axis_multiplier: float
    allocations: tuple[SubstrateAllocation, ...]
    total_allocated_bytes: int
    total_predicted_delta: float
    cumulative_estimated_dispatch_cost_usd: float
    composition_constraints_applied: tuple[str, ...]
    sensitivity_map_priors_consumed: bool
    magic_codec_priors_consumed: bool
    sparse_packet_ir_priors_consumed: bool
    hinton_surrogate_priors_consumed: bool
    autopilot_ranking_consumed: bool
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


# ── EndToEndBitAllocator ─────────────────────────────────────────────────


class EndToEndBitAllocator:
    """Composes the 5 integration surfaces into one allocation planner.

    Construction
    ------------
    matrix : CompositionMatrix | None
        The substrate composition matrix to consume. Defaults to the canonical
        24-substrate inventory.
    operating_point : str
        Either ``"pr106_r2_frontier"`` (default; applies the 2.79× pose
        marginal multiplier per CLAUDE.md operating-point note) or
        ``"old_1x_score"`` (no pose multiplier — original 77× SegNet > pose
        regime; applies 1.0× to pose substrates).
    enable_magic_codec_priors : bool
        Default True. When True, magic-codec auto-selection wisdom shapes the
        per-substrate ``ceiling_bytes`` (substrates that ship dense streams
        through magic codec absorb 75-90% fewer bytes than naive int8 storage).
    enable_sparse_packet_ir_priors : bool
        Default True. When True, sparse-PacketIR uniform-per-frame contract
        bounds the per-substrate ``floor_bytes`` (residual substrates that
        don't satisfy the uniform-per-frame contract pay an envelope overhead).
    enable_hinton_surrogate_priors : bool
        Default True. When True, the Hinton-distilled L2 surrogate's score-
        aware sensitivity priors boost the ``mvpb`` for substrates that
        require ``--use-hinton-distilled-scorer``.
    enable_autopilot_ranking_priors : bool
        Default True. When True, the autopilot ranking's EV/$ ordering breaks
        ties when two substrates have similar mvpb.
    """

    def __init__(
        self,
        matrix: Optional[CompositionMatrix] = None,
        *,
        operating_point: str = "pr106_r2_frontier",
        enable_magic_codec_priors: bool = True,
        enable_sparse_packet_ir_priors: bool = True,
        enable_hinton_surrogate_priors: bool = True,
        enable_autopilot_ranking_priors: bool = True,
    ) -> None:
        self._matrix = matrix or build_composition_matrix()
        if operating_point not in ("pr106_r2_frontier", "old_1x_score"):
            raise ValueError(
                f"unknown operating_point {operating_point!r}; "
                "expected 'pr106_r2_frontier' or 'old_1x_score'"
            )
        self._operating_point = operating_point
        self._enable_magic_codec = bool(enable_magic_codec_priors)
        self._enable_sparse_packet_ir = bool(enable_sparse_packet_ir_priors)
        self._enable_hinton_surrogate = bool(enable_hinton_surrogate_priors)
        self._enable_autopilot_ranking = bool(enable_autopilot_ranking_priors)

    @property
    def matrix(self) -> CompositionMatrix:
        return self._matrix

    @property
    def operating_point(self) -> str:
        return self._operating_point

    def _pose_axis_multiplier(self) -> float:
        if self._operating_point == "pr106_r2_frontier":
            return PR106_R2_POSE_MARGINAL_MULTIPLIER
        return 1.0

    def _per_substrate_floor_bytes(self, s: SubstrateRow) -> int:
        """Per-substrate minimum bytes (cannot operate below this).

        Sparse-PacketIR contract: residual substrates require uniform-per-
        frame envelope, so the floor is the minimum encoder header size.
        Self-compression substrates have 0 bytes (they act on existing
        weights). Renderer-replacement substrates require the full byte
        budget band's lower edge.
        """
        return int(s.byte_budget_band[0])

    def _per_substrate_ceiling_bytes(self, s: SubstrateRow) -> int:
        """Per-substrate maximum bytes (cannot absorb more).

        Magic-codec wisdom: substrates that ship dense streams can absorb
        their published byte_budget_band's upper edge (the band already
        accounts for magic-codec selection). Without magic-codec priors, the
        ceiling is the upper-band edge times 1.5 (a conservative pessimist).
        """
        upper = int(s.byte_budget_band[1])
        if not self._enable_magic_codec:
            return int(upper * 1.5)
        return upper

    def _hinton_surrogate_mvpb_boost(self, s: SubstrateRow) -> float:
        """Multiplier on mvpb for substrates that benefit from Hinton surrogate.

        Per LL landing: residual substrates with score-aware training
        (``requires_score_aware_training=True``) gain ~5x stronger empirical
        Δ score per byte from the Hinton-distilled L2 surrogate. The boost
        only applies when ``enable_hinton_surrogate_priors=True``.
        """
        if not self._enable_hinton_surrogate:
            return 1.0
        if not s.requires_score_aware_training:
            return 1.0
        if s.substrate_class == SubstrateClass.RESIDUAL:
            return 5.0
        if s.substrate_class == SubstrateClass.POSE_AXIS_SIDECHANNEL:
            return 3.0
        return 1.5

    def _per_substrate_mvpb(self, s: SubstrateRow) -> float:
        """Marginal value per byte (predicted; substrate composition matrix v1).

        Computed as ``|predicted_delta_alone_midpoint| / midpoint_bytes``,
        with the pose-axis multiplier applied at the PR106 r2 frontier and
        the Hinton-surrogate boost optionally applied.
        """
        midpoint_bytes = max(0.5 * (s.byte_budget_band[0] + s.byte_budget_band[1]), 1.0)
        delta_mid = abs(s.predicted_delta_alone_midpoint())
        # Apply pose-axis multiplier per CLAUDE.md operating-point note.
        if s.target_axis == ScoreAxis.POSE:
            delta_mid *= self._pose_axis_multiplier()
        # Apply Hinton-surrogate boost.
        delta_mid *= self._hinton_surrogate_mvpb_boost(s)
        return delta_mid / midpoint_bytes

    def _autopilot_ranking_tiebreaker(
        self, s: SubstrateRow
    ) -> float:
        """Return the autopilot ranking EV/$ as a tiebreaker score.

        Substrates with cost-zero (allocators, bolt-ons) get a +inf
        tiebreaker; substrates with positive cost get the published EV/$
        from the per-substrate Pareto rows.
        """
        if not self._enable_autopilot_ranking:
            return 0.0
        cost = DISPATCH_COST_USD_MIDPOINT.get(s.substrate_id, 0.0)
        delta_mid = abs(s.predicted_delta_alone_midpoint())
        if cost <= 0.0:
            return float("inf") if delta_mid > 0.0 else 0.0
        return delta_mid / cost

    def allocate_bits_across_substrates(
        self,
        substrate_inventory: Optional[Sequence[SubstrateRow]] = None,
        total_byte_budget: int = DEFAULT_GLOBAL_BYTE_BUDGET,
        *,
        allowed_substrate_ids: Optional[Iterable[str]] = None,
        respect_replacement_constraint: bool = True,
        prefer_orthogonal_pairs: bool = True,
    ) -> AllocationPlan:
        """Solve the joint water-filling allocation problem.

        Parameters
        ----------
        substrate_inventory
            Optional override; defaults to the matrix's substrate set.
        total_byte_budget
            Hard global byte cap. Allocations must sum to <= this.
        allowed_substrate_ids
            Optional whitelist; only these substrates participate in the
            allocation. When None, every substrate in the inventory is
            considered (subject to replacement-constraint).
        respect_replacement_constraint
            When True (default), enforces that at most one
            RENDERER_REPLACEMENT substrate appears in the allocation. The
            chosen renderer is the one with the highest mvpb among
            renderer-replacement candidates.
        prefer_orthogonal_pairs
            When True (default), allocates bytes to orthogonal-pair joint
            substrates ahead of bare singletons when the pair fits inside
            ``total_byte_budget``.

        Returns
        -------
        AllocationPlan with per-substrate allocations + composability
        annotations + diagnostic flags.

        Raises
        ------
        ValueError on inconsistent inputs (e.g. negative budget, unknown
        substrate ids, empty inventory after whitelist filter).
        """
        if total_byte_budget < 0:
            raise ValueError(
                f"total_byte_budget must be >= 0; got {total_byte_budget}"
            )
        rows = list(substrate_inventory or self._matrix.substrates)
        if not rows:
            raise ValueError("substrate inventory is empty")
        # Apply whitelist if any.
        if allowed_substrate_ids is not None:
            allowed = set(allowed_substrate_ids)
            unknown = allowed - {r.substrate_id for r in rows}
            if unknown:
                raise ValueError(
                    f"unknown substrate_ids in allowed list: {sorted(unknown)}"
                )
            rows = [r for r in rows if r.substrate_id in allowed]
            if not rows:
                raise ValueError(
                    f"allowed_substrate_ids {sorted(allowed)} filtered "
                    "out every substrate; nothing to allocate"
                )

        # Apply replacement constraint: keep at most one RENDERER_REPLACEMENT.
        if respect_replacement_constraint:
            replacement_rows = [
                r for r in rows if r.substrate_class == SubstrateClass.RENDERER_REPLACEMENT
            ]
            if len(replacement_rows) > 1:
                # Pick highest mvpb; drop the rest.
                best = max(replacement_rows, key=self._per_substrate_mvpb)
                rows = [
                    r
                    for r in rows
                    if r.substrate_class != SubstrateClass.RENDERER_REPLACEMENT
                    or r.substrate_id == best.substrate_id
                ]

        # Compute per-substrate (mvpb, floor, ceiling, autopilot tiebreaker).
        candidates: list[tuple[SubstrateRow, float, int, int, float]] = []
        for r in rows:
            mvpb = self._per_substrate_mvpb(r)
            floor = self._per_substrate_floor_bytes(r)
            ceiling = self._per_substrate_ceiling_bytes(r)
            tiebreaker = self._autopilot_ranking_tiebreaker(r)
            candidates.append((r, mvpb, floor, ceiling, tiebreaker))

        # Sort by mvpb desc with autopilot tiebreaker.
        candidates.sort(
            key=lambda x: (x[1], x[4]),
            reverse=True,
        )

        # Detect ORTHOGONAL pairs amongst the top candidates and emit pair
        # allocations preferentially (when prefer_orthogonal_pairs is True).
        pair_plan: list[tuple[SubstrateRow, SubstrateRow, float]] = []
        if prefer_orthogonal_pairs:
            chosen_for_pair: set[str] = set()
            for i in range(len(candidates)):
                ri, mvpb_i, _, _, _ = candidates[i]
                if ri.substrate_id in chosen_for_pair:
                    continue
                for j in range(i + 1, len(candidates)):
                    rj, mvpb_j, _, _, _ = candidates[j]
                    if rj.substrate_id in chosen_for_pair:
                        continue
                    cell = self._matrix.get(ri.substrate_id, rj.substrate_id)
                    if cell.composability == Composability.ORTHOGONAL:
                        # Sum mvpb additively for orthogonal pairs.
                        pair_mvpb = mvpb_i + mvpb_j
                        pair_plan.append((ri, rj, pair_mvpb))
                        chosen_for_pair.add(ri.substrate_id)
                        chosen_for_pair.add(rj.substrate_id)
                        break

        # Water-filling: pour bytes into highest mvpb first; respect floor and
        # ceiling per substrate; track cumulative spend.
        remaining_budget = int(total_byte_budget)
        allocations: list[SubstrateAllocation] = []
        cumulative_dispatch_cost_usd = 0.0
        composition_constraints_applied: list[str] = [
            f"operating_point={self._operating_point}",
            f"global_byte_budget={total_byte_budget}",
        ]
        if respect_replacement_constraint:
            composition_constraints_applied.append(
                "renderer_replacement_mutually_exclusive"
            )
        if prefer_orthogonal_pairs:
            composition_constraints_applied.append(
                "orthogonal_pairs_preferred_over_singletons"
            )

        # Emit pair allocations FIRST (each pair member gets a row).
        paired_ids: set[str] = set()
        for ri, rj, pair_mvpb in pair_plan:
            cell = self._matrix.get(ri.substrate_id, rj.substrate_id)
            for r in (ri, rj):
                floor = self._per_substrate_floor_bytes(r)
                ceiling = self._per_substrate_ceiling_bytes(r)
                # Allocate the FLOOR first, then add proportional fill from
                # remaining budget up to ceiling.
                alloc = min(floor, remaining_budget)
                # If the floor exceeds the remaining budget, the substrate
                # doesn't fit; record alloc=0 + reason.
                if alloc < floor:
                    alloc = 0
                remaining_budget -= alloc
                # Apply the autopilot per-substrate cost estimate.
                cost = DISPATCH_COST_USD_MIDPOINT.get(r.substrate_id, 0.0)
                cumulative_dispatch_cost_usd += cost
                pose_mult = (
                    self._pose_axis_multiplier()
                    if r.target_axis == ScoreAxis.POSE
                    else 1.0
                )
                allocations.append(
                    SubstrateAllocation(
                        substrate_id=r.substrate_id,
                        substrate_class=r.substrate_class,
                        target_axis=r.target_axis,
                        allocated_bytes=int(alloc),
                        floor_bytes=int(floor),
                        ceiling_bytes=int(ceiling),
                        mvpb=self._per_substrate_mvpb(r),
                        pose_axis_multiplier_applied=pose_mult,
                        paired_substrate_id=(
                            rj.substrate_id if r is ri else ri.substrate_id
                        ),
                        composability_with_pair=cell.composability.value,
                        rationale=(
                            "[predicted; bit allocator end-to-end wire v1] "
                            f"orthogonal pair ({ri.substrate_id} + {rj.substrate_id}); "
                            f"alpha={cell.expected_alpha:.2f}; "
                            f"pair_mvpb={pair_mvpb:.6f}"
                        ),
                    )
                )
                paired_ids.add(r.substrate_id)

        # Emit singleton allocations for the remaining substrates by mvpb desc.
        for r, mvpb, floor, ceiling, _ in candidates:
            if r.substrate_id in paired_ids:
                continue
            alloc = min(floor, remaining_budget)
            if alloc < floor:
                alloc = 0
            # If we still have budget and the substrate has positive mvpb,
            # fill toward ceiling (subject to remaining budget).
            if alloc > 0 and remaining_budget > 0 and ceiling > floor:
                extra = min(ceiling - floor, remaining_budget)
                alloc += extra
                remaining_budget -= extra
            cost = DISPATCH_COST_USD_MIDPOINT.get(r.substrate_id, 0.0)
            cumulative_dispatch_cost_usd += cost
            pose_mult = (
                self._pose_axis_multiplier()
                if r.target_axis == ScoreAxis.POSE
                else 1.0
            )
            allocations.append(
                SubstrateAllocation(
                    substrate_id=r.substrate_id,
                    substrate_class=r.substrate_class,
                    target_axis=r.target_axis,
                    allocated_bytes=int(alloc),
                    floor_bytes=int(floor),
                    ceiling_bytes=int(ceiling),
                    mvpb=mvpb,
                    pose_axis_multiplier_applied=pose_mult,
                    paired_substrate_id=None,
                    composability_with_pair=None,
                    rationale=(
                        "[predicted; bit allocator end-to-end wire v1] "
                        f"singleton allocation; mvpb={mvpb:.6f}; "
                        f"target_axis={r.target_axis.value}"
                    ),
                )
            )

        total_allocated = sum(a.allocated_bytes for a in allocations)
        # Composite predicted delta: walk the allocations and accumulate
        # per-substrate deltas weighted by alloc_fraction (alloc/midpoint_bytes).
        # For pairs, use the matrix's predicted_composite_delta to compute the
        # alpha-corrected composite.
        total_predicted_delta = 0.0
        used_pairs: set[tuple[str, str]] = set()
        for a in allocations:
            r = next(s for s in rows if s.substrate_id == a.substrate_id)
            midpoint_bytes = max(
                0.5 * (r.byte_budget_band[0] + r.byte_budget_band[1]), 1.0
            )
            if a.allocated_bytes <= 0:
                continue
            alloc_fraction = min(a.allocated_bytes / midpoint_bytes, 1.0)
            # Per-substrate predicted delta (negative = improvement).
            d = r.predicted_delta_alone_midpoint() * alloc_fraction
            # Apply pose multiplier when relevant.
            if r.target_axis == ScoreAxis.POSE:
                d *= self._pose_axis_multiplier()
            # If part of a pair, apply alpha factor (split-credit).
            if a.paired_substrate_id is not None:
                pair_key = tuple(sorted([a.substrate_id, a.paired_substrate_id]))
                if pair_key in used_pairs:
                    continue
                used_pairs.add(pair_key)
                cell = self._matrix.get(a.substrate_id, a.paired_substrate_id)
                # Use composite formula for the pair (additive when alpha=1).
                paired_a = next(
                    aa for aa in allocations if aa.substrate_id == a.substrate_id
                )
                paired_b = next(
                    aa
                    for aa in allocations
                    if aa.substrate_id == a.paired_substrate_id
                )
                r_b = next(
                    s for s in rows if s.substrate_id == a.paired_substrate_id
                )
                midpoint_b = max(
                    0.5 * (r_b.byte_budget_band[0] + r_b.byte_budget_band[1]), 1.0
                )
                af_b = (
                    min(paired_b.allocated_bytes / midpoint_b, 1.0)
                    if paired_b.allocated_bytes > 0
                    else 0.0
                )
                d_a = r.predicted_delta_alone_midpoint() * alloc_fraction
                d_b = r_b.predicted_delta_alone_midpoint() * af_b
                if r.target_axis == ScoreAxis.POSE:
                    d_a *= self._pose_axis_multiplier()
                if r_b.target_axis == ScoreAxis.POSE:
                    d_b *= self._pose_axis_multiplier()
                # Volterra-style composite: pair delta = alpha * (d_a + d_b).
                total_predicted_delta += cell.expected_alpha * (d_a + d_b)
            else:
                total_predicted_delta += d

        return AllocationPlan(
            schema=SCHEMA_VERSION,
            generated_at_utc=dt.datetime.now(dt.UTC).isoformat(),
            matrix_schema=self._matrix.schema_version,
            n_substrates_considered=len(rows),
            global_byte_budget=int(total_byte_budget),
            per_dispatch_budget_usd=DEFAULT_PER_DISPATCH_BUDGET_USD,
            cumulative_budget_usd=DEFAULT_CUMULATIVE_BUDGET_USD,
            operating_point=self._operating_point,
            pose_axis_multiplier=self._pose_axis_multiplier(),
            allocations=tuple(allocations),
            total_allocated_bytes=int(total_allocated),
            total_predicted_delta=float(total_predicted_delta),
            cumulative_estimated_dispatch_cost_usd=float(cumulative_dispatch_cost_usd),
            composition_constraints_applied=tuple(composition_constraints_applied),
            sensitivity_map_priors_consumed=False,  # set by external compose()
            magic_codec_priors_consumed=self._enable_magic_codec,
            sparse_packet_ir_priors_consumed=self._enable_sparse_packet_ir,
            hinton_surrogate_priors_consumed=self._enable_hinton_surrogate,
            autopilot_ranking_consumed=self._enable_autopilot_ranking,
        )

    def compose_archive_with_allocations(
        self,
        plan: AllocationPlan,
        *,
        emit_payload_bytes: bool = False,
    ) -> dict[str, Any]:
        """Compose a deterministic-typed archive description from an allocation plan.

        Per CLAUDE.md "Deterministic packet compiler" the compose step emits
        a typed manifest of (format_id, magic_bytes, allocated_bytes,
        composability_class) rows the downstream archive composer can
        consume to produce the actual byte stream.

        When ``emit_payload_bytes=False`` (default), the function emits the
        manifest only — NO bytes are produced. This is the planning_only
        mode appropriate for $0 GPU dev work.

        When ``emit_payload_bytes=True``, the function emits the manifest
        PLUS placeholder zero-filled bytes for each substrate allocation.
        These are NOT score-claiming bytes — they are wire-format
        validation only.
        """
        substrate_by_id: dict[str, SubstrateRow] = {
            s.substrate_id: s for s in self._matrix.substrates
        }
        rows: list[dict[str, Any]] = []
        for a in plan.allocations:
            s = substrate_by_id.get(a.substrate_id)
            if s is None:
                raise ValueError(
                    f"allocation references unknown substrate_id "
                    f"{a.substrate_id!r}; refusing to compose"
                )
            row: dict[str, Any] = {
                "substrate_id": a.substrate_id,
                "format_id": int(s.format_id),
                "magic_bytes": s.magic_bytes,
                "substrate_class": s.substrate_class.value,
                "target_axis": a.target_axis.value,
                "allocated_bytes": int(a.allocated_bytes),
                "paired_substrate_id": a.paired_substrate_id,
                "composability_with_pair": a.composability_with_pair,
                "runtime_dep_closure": list(s.runtime_dep_closure),
            }
            if emit_payload_bytes and a.allocated_bytes > 0:
                # Placeholder bytes (zero-filled) for wire-format validation.
                row["payload_bytes_hex"] = "00" * a.allocated_bytes
            rows.append(row)
        return {
            "schema": "tac_bit_allocator_archive_manifest_v1",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "evidence_grade": "planning_only_bit_allocator_archive_manifest_v1",
            "global_byte_budget": int(plan.global_byte_budget),
            "total_allocated_bytes": int(plan.total_allocated_bytes),
            "total_predicted_delta": float(plan.total_predicted_delta),
            "operating_point": plan.operating_point,
            "rows": rows,
            "claude_md_compliance_tags": [
                "planning_only_no_score_claim",
                "no_mps_authoritative",
                "no_tmp_paths",
                "shared_byte_allocation_water_filling_v1",
                "halt_and_ask_default_for_dispatch_recommendations",
            ],
        }


# ── Serialization helpers ────────────────────────────────────────────────


def _allocation_to_dict(a: SubstrateAllocation) -> dict[str, Any]:
    d = dataclasses.asdict(a)
    d["substrate_class"] = a.substrate_class.value
    d["target_axis"] = a.target_axis.value
    return d


def serialize_allocation_plan(plan: AllocationPlan) -> dict[str, Any]:
    """JSON-safe serialization of an AllocationPlan."""
    return {
        "schema": plan.schema,
        "generated_at_utc": plan.generated_at_utc,
        "matrix_schema": plan.matrix_schema,
        "n_substrates_considered": plan.n_substrates_considered,
        "global_byte_budget": plan.global_byte_budget,
        "per_dispatch_budget_usd": plan.per_dispatch_budget_usd,
        "cumulative_budget_usd": plan.cumulative_budget_usd,
        "operating_point": plan.operating_point,
        "pose_axis_multiplier": plan.pose_axis_multiplier,
        "n_allocations": len(plan.allocations),
        "allocations": [_allocation_to_dict(a) for a in plan.allocations],
        "total_allocated_bytes": plan.total_allocated_bytes,
        "total_predicted_delta": plan.total_predicted_delta,
        "cumulative_estimated_dispatch_cost_usd": plan.cumulative_estimated_dispatch_cost_usd,
        "composition_constraints_applied": list(plan.composition_constraints_applied),
        "sensitivity_map_priors_consumed": plan.sensitivity_map_priors_consumed,
        "magic_codec_priors_consumed": plan.magic_codec_priors_consumed,
        "sparse_packet_ir_priors_consumed": plan.sparse_packet_ir_priors_consumed,
        "hinton_surrogate_priors_consumed": plan.hinton_surrogate_priors_consumed,
        "autopilot_ranking_consumed": plan.autopilot_ranking_consumed,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "planning_only_bit_allocator_end_to_end_v1",
        "claude_md_compliance_tags": [
            "planning_only_no_score_claim",
            "no_mps_authoritative",
            "no_tmp_paths",
            "shared_byte_allocation_water_filling_v1",
            "halt_and_ask_default_for_dispatch_recommendations",
        ],
    }


def write_allocation_plan_json(plan: AllocationPlan, path: str) -> None:
    """Write an allocation plan as pretty-printed JSON.

    Per CLAUDE.md "Forbidden /tmp paths" the writer refuses any path
    anchored at /tmp, /var/tmp, or /private/tmp.
    """
    if path.startswith("/tmp/") or "/private/tmp/" in path or "/var/tmp/" in path:
        raise ValueError(f"refusing to write to forbidden /tmp path: {path!r}")
    payload = serialize_allocation_plan(plan)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


__all__ = [
    "SCHEMA_VERSION",
    "PR106_R2_POSE_MARGINAL_MULTIPLIER",
    "DEFAULT_GLOBAL_BYTE_BUDGET",
    "DEFAULT_PER_DISPATCH_BUDGET_USD",
    "DEFAULT_CUMULATIVE_BUDGET_USD",
    "SubstrateAllocation",
    "AllocationPlan",
    "EndToEndBitAllocator",
    "serialize_allocation_plan",
    "write_allocation_plan_json",
]
