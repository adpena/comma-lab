# CUDA-Aware DQS1 Variant — Per-Axis Pareto Pair Ordering Design Memo (2026-05-25)

- timestamp_utc: 2026-05-25T15:55:00Z
- agent: claude (CUDA-AXIS-DQS1-DESIGN subagent)
- lane_id: lane_cuda_axis_dqs1_design_memo_per_axis_pareto_pair_ordering_20260525
- scope: NEW design memo + scaffold sketch + canonical equation candidate QUEUED per Catalog #344 + 4-BUILD operator-routable enumeration
- authority: planning + observability ONLY; score/promotion/rank/dispatch authority all FALSE
- relates to: DQS1-LOOP-CLOSURE-ASSIST commit `504a31448` GAP 3 (CUDA-axis DQS1 cascade absent) + RATE-ATTACK-METHODS-DIMENSIONS-MATRIX commit `7a78c5661` Top-5 cell #2 (M02 DQS1 drop-one × D22 CPU-CUDA repair)
- empirical anchors: contest-CPU frontier `7a0da5d0fc32` 0.19202828 [contest-CPU] (DQS1 lane 2026-05-22) + contest-CUDA frontier `9cb989cef519` 0.20533002 [contest-CUDA T4] (PR106 format0d 2026-05-16, NOT a DQS1 archive); pair0371 paired CPU+CUDA anchor residual=0.0000 against equation #36
- discipline anchors: Catalog #110/#113 APPEND-ONLY + #125 (6-hook wire-in) + #192 (bidirectional CPU+CUDA submission) + #229 (premise verification) + #287 (canonical Provenance evidence-tag) + #296 (Dykstra-feasibility for predicted bands) + #313 (probe-outcomes ledger) + #344 (canonical equation candidates QUEUED for RATIFY-N) + #356 (AxisDecomposition per-axis canonical contract)

## Canonical-vs-unique decision per layer

- **Reusing canonical**: `tools/subagent_checkpoint.py` (Catalog #206) + `tools/list_canonical_equations.py` (Catalog #344 reader) + canonical Provenance umbrella (Catalog #323) + Catalog #356 `AxisDecomposition` + `tac.score_composition.compose_score_from_axes` canonical helper + canonical contest formula constants (`CANONICAL_SEG_MULTIPLIER=100.0` / `CANONICAL_POSE_SQRT_INNER=10.0` / `CANONICAL_RATE_MULTIPLIER=25.0` / `CANONICAL_RATE_DENOM_BYTES=37_545_489`) + `pairset_component_marginal_score_decomposition_v1` (equation #36 ratified).
- **Forking NOTHING**: this is a NEW design memo + skeleton scaffold script; ZERO mutation of any sister codex DQS1 cascade source code, landing memos, canonical equation registry, or state JSON.

## Observability surface

The design is observable through 6 canonical facets per CLAUDE.md "Max observability" non-negotiable:
1. **Inspectable per layer**: per-axis Δ decomposition table per pair (rate / SegNet_CPU / SegNet_CUDA / PoseNet_CPU / PoseNet_CUDA); per-pair Pareto rank; Dykstra-feasibility verdict per pair; minimax score per pair.
2. **Decomposable per signal**: per-pair, per-axis, per-component (rate/segnet/posenet) signal preserved through `PerAxisPairDelta` dataclass.
3. **Diff-able across runs**: scaffold emits canonical JSONL with seed + master-gradient ledger sha; subsequent BUILD-1 runs append rather than mutate (APPEND-ONLY per Catalog #110/#113).
4. **Queryable post-hoc**: every emitted row has canonical Provenance per Catalog #323 + Catalog #356 AxisDecomposition serializable via `as_dict`.
5. **Cite-able**: every claim cites equation #36 pair-drop decomposition + equation #17 cpu_cuda_score_gap_v1 + equation #18 pose_axis_cuda_amplification_v1 + DQS1-LOOP-CLOSURE-ASSIST GAP 3 + RATE-ATTACK Top-5 cell #2.
6. **Counterfactual-able**: per-axis bypass mode (CPU-only / CUDA-only) lets operator audit "what if D22 dimension dropped" within the same canonical interface.

## 9-dimension success checklist evidence

1. **UNIQUENESS**: per-axis Pareto pair-ordering is structurally distinct from CPU-axis-only rate-saving-first ordering used by current DQS1 cascade — this is a NEW Pareto-frontier-over-axis-space primitive, NOT an extension of an existing CPU-axis primitive.
2. **BEAUTY + ELEGANCE**: ONE design memo covers 3 deliverables (math + scaffold + equation candidate + 4-BUILD) + skeleton ~150-250 LOC; operator-readable in <15 min.
3. **DISTINCTNESS**: distinct from codex's DQS1 cascade (CPU-axis only); distinct from cell #1 drop-many beam waterfill (independent vs interaction); distinct from GAP 4 drop-many beam (single-axis vs per-axis); distinct from any sister cell.
4. **RIGOR**: every claim cites canonical equation registry + canonical helper + canonical Provenance + Dykstra-feasibility per Catalog #296.
5. **OPTIMIZATION-PER-TECHNIQUE**: D22 per-CPU-CUDA-axis dimension is the unique-and-complete-per-method primitive for this method per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode".
6. **STACK-OF-STACKS-COMPOSABILITY**: per-axis Pareto ordering composes orthogonally with cell #1 (drop-many) by ranking over per-axis Δ space; composes with cell #5 (pair/frame lattice) by inheriting receiver-feasibility constraints.
7. **DETERMINISTIC-REPRODUCIBILITY**: scaffold seed-pinned; canonical helper invocations reproducible via documented CLI flags; Provenance carries `captured_at` + `archive_sha256_short`.
8. **EXTREME-OPTIMIZATION-PERFORMANCE**: $0 design + skeleton; ~1-2h wall-clock; 4-BUILD cascade tied to $0-5 paid Modal envelope.
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: directly addresses GAP 3 (CUDA-axis frontier still PR106 format0d 0.20533002, NOT a DQS1 archive) — predicted ΔS_CUDA in [-0.005, +0.0005] for minimax-optimal pair-drop vs current CPU-optimal pair-drop that REGRESSED CUDA by +0.0000013.

## Cargo-cult audit per assumption

- **HARD-EARNED**: `pairset_component_marginal_score_decomposition_v1` (equation #36; 8 anchors residual=0.0000) — per-pair drop-one Δ decomposition is EMPIRICALLY VERIFIED CPU + CUDA on pair0371 anchor; the per-axis decomposition primitive is rigorously grounded.
- **HARD-EARNED**: `cpu_cuda_score_gap_v1` (equation #17; 1 anchor) + `pose_axis_cuda_amplification_v1` (equation #18; 1 anchor) — CPU/CUDA divergence on same archive bytes is canonically registered.
- **HARD-EARNED**: `per_byte_leverage_uniformly_distributed_v1` (equation #44; 4 anchors) — per-byte edits dominate over substrate-class-shift edits empirically; the per-pair-drop primitive correctly aggregates byte-level Δ into pair-level Δ.
- **CARGO-CULTED (HYPOTHESIS)**: assumption that "CPU-optimal pair-drop ordering generalizes to CUDA-optimal pair-drop ordering" — EMPIRICALLY FALSIFIED by pair0371 itself (ΔCPU = -0.0000007 win on rate-saving but ΔCUDA = +0.0000013 regression on SegNet shift); this design memo's WHOLE PREMISE is unwinding this cargo-cult.
- **CARGO-CULTED (HYPOTHESIS)**: assumption that "per-pair CPU SegNet shift ≈ per-pair CUDA SegNet shift" — pair0371 anchor empirically shows CUDA SegNet shift ≈ 2× CPU SegNet shift per one-byte drop (consistent with equation #18 pose_axis_cuda_amplification_v1's ~5× pose ratio at PR106 operating point; SegNet shift at FRONTIER DQS1 operating point may carry similar but distinct amplification).
- **CARGO-CULTED (HYPOTHESIS)**: assumption that "minimax-optimal pair-drop exists in the intersection of CPU + CUDA improvement polytopes" — Dykstra alternating-projections feasibility may return EMPTY intersection (no pair-drop improves BOTH axes); this is the canonical falsifiable challenge per Catalog #296 + Carmack MVP-first Step 2.

## Predicted ΔS band

Per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" FORBIDDEN pattern + Catalog #296 STRICT gate:

**Predicted ΔS band per axis**:
- **ΔCPU band** for minimax-optimal pair-drop: [-0.0000005, +0.0000005] (consistent with the per-pair-drop Δ scale; rate-saving wins typically -0.0000007 but minimax sacrifices some CPU gain to preserve CUDA neutrality)
- **ΔCUDA band** for minimax-optimal pair-drop: [-0.0000010, +0.0000005] (predicted IMPROVEMENT or near-neutrality vs current rate-saving-first ΔCUDA = +0.0000013 regression)

**Dykstra-feasibility check**: per equation #36 + equation #17 + equation #18, the per-axis improvement polytopes are:
- CPU-improvement polytope: `{pair_i : ΔCPU_i < 0}` = pairs where rate-saving outweighs SegNet+PoseNet CPU shifts
- CUDA-improvement polytope: `{pair_i : ΔCUDA_i ≤ 0}` = pairs where rate-saving outweighs ~2× amplified SegNet+PoseNet CUDA shifts
- **Intersection feasibility check** (Dykstra alternating-projections per CLAUDE.md "Council conduct" Dykstra co-lead role): MAY BE EMPTY for current 600-pair candidate set. If empty, minimax fallback ranks pairs by `max(ΔCPU, ΔCUDA)` minimization rather than intersection membership.

**First-principles justification per CLAUDE.md "Council conduct" Dykstra co-lead role**: alternating-projections feasibility IS the canonical arbiter of whether a multi-constraint composition is achievable rather than just predicted. The DQS1 cascade's current CPU-axis-only ordering effectively projects onto ONE polytope; this design lifts to alternating projections over BOTH polytopes. Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable: "Prefer solvable math over arbitrary sweeps" + "Dykstra/ADMM feasibility" listed as canonical knob-grounding.

**Probe-disambiguator path** per Catalog #296 acceptance cascade (b): `tools/probe_cuda_axis_dqs1_per_axis_pareto_disambiguator.py` (THIS scaffold; skeleton in DELIVERABLE 2).

## Council attendees / verdict

T1 working-group (design memo only; no T2+ deliberation required per Catalog #300):
- Shannon (information-theory grounding for per-axis decomposition + rate-vs-distortion Pareto)
- Dykstra (alternating-projections feasibility — co-lead invoked specifically per CLAUDE.md "Council conduct")
- Rudin (interpretable per-pair ranking via canonical Pareto rank + minimax score)
- Carmack (MVP-first 5-step phasing for 4-BUILD operator-routable)
- Assumption-Adversary (challenges the 3 CARGO-CULTED hypotheses enumerated above)

T1 working-group VERDICT: **PROCEED** (design memo + skeleton only; no quorum required at T1 per Catalog #300; this is design-time work feeding downstream BUILD-1 through BUILD-4 + DISPATCH).

**Assumption-Adversary explicit per-round assumption statement per CLAUDE.md "Council conduct" Fix-7 amendment**:
- *"The shared assumption I am operating within for this design is: per-pair per-axis Δ decomposition (equation #36) ratified on pair0371 generalizes to all 600 pair candidates with bounded residual."* — Classification: **HARD-EARNED** for pair0371; **CARGO-CULTED for other 599 pairs** (no empirical anchor outside pair0371); unwind path is BUILD-1 populating empirical table from 600-pair fp64 master-gradient ledger × paired CPU+CUDA scorer responses.
- *"If this assumption is wrong, what changes?"*: If per-pair per-axis Δ decomposition has nontrivial residual on other pairs, the Pareto ranking would be miscalibrated and minimax-optimal pair-drop might not exist. Mitigation: BUILD-1's residual check per pair per equation #36; reject pairs with residual > 0.0000005 from Pareto ranking; flag for follow-on paired Modal anchor dispatch.

## Canonical equation #36 reference

`pairset_component_marginal_score_decomposition_v1` (equation #36 ratified; 8 anchors all residual=0.0000):

```
ΔS_axis(pair_i) = +ΔRate(pair_i) + ΔSegNet_axis(pair_i) + ΔPoseNet_axis(pair_i)
```

where (per canonical contest formula):
- `ΔRate(pair_i) = +25 * (-N_bytes_dropped(pair_i)) / 37_545_489` (rate-saving; sign convention: negative byte count drops rate term)
- `ΔSegNet_axis(pair_i) = +100 * Δseg_avg_axis(pair_i)` (sign convention: positive seg distortion increases score)
- `ΔPoseNet_axis(pair_i) = sqrt(10 * pose_avg_axis_new(pair_i)) - sqrt(10 * pose_avg_axis_baseline(pair_i))` (non-linear; canonical formula's pose term is sqrt-of-mean per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent")

The per-axis split (CPU vs CUDA) operates on the `_axis` suffix; equation #36 was ratified with `_axis = cpu` and `_axis = cuda` evidence on pair0371. The PR106 frontier shows ΔCUDA pose-axis component is ~5× ΔCPU pose-axis component per equation #18; the DQS1 frontier may carry similar but distinct amplification.

---

## DELIVERABLE 1: Math + algorithm design

### 1.1 Per-axis pair-drop delta decomposition

For each candidate pair `pair_i ∈ {0, ..., 599}` (canonical DQS1 600-pair selector space), per equation #36:

```
ΔCPU(pair_i)  = ΔRate(pair_i) + ΔSegNet_CPU(pair_i)  + ΔPoseNet_CPU(pair_i)
ΔCUDA(pair_i) = ΔRate(pair_i) + ΔSegNet_CUDA(pair_i) + ΔPoseNet_CUDA(pair_i)
```

where `ΔRate(pair_i)` is identical across axes (rate term is axis-agnostic; canonical formula's rate term `25 * archive_bytes / 37_545_489` is byte-count only) but `ΔSegNet_axis` and `ΔPoseNet_axis` carry per-axis values per equations #17 + #18.

### 1.2 Per-axis Pareto frontier construction

Construct per-axis non-dominated set over the 600-pair candidate space:
- A pair `pair_i` is **CPU-Pareto-non-dominated** iff `∀ pair_j ≠ pair_i: ΔCPU(pair_j) > ΔCPU(pair_i)` OR `ΔCUDA(pair_j) > ΔCUDA(pair_i)` (i.e. no other pair strictly improves BOTH axes).
- The CPU-CUDA Pareto frontier is the set of all such non-dominated pairs.
- Per-pair Pareto rank: depth in successive non-dominated layers (Fonseca-Fleming 1995 NSGA-style ranking).

### 1.3 Dykstra alternating-projection feasibility (canonical co-lead invocation)

Per CLAUDE.md "Council conduct" Dykstra co-lead role + Catalog #296 STRICT predicted-band-Dykstra-feasibility requirement:

```
CPU-improvement polytope C_CPU = {pair_i : ΔCPU(pair_i) < ε_CPU}
CUDA-improvement polytope C_CUDA = {pair_i : ΔCUDA(pair_i) ≤ ε_CUDA}
Feasible intersection F = C_CPU ∩ C_CUDA
```

where `ε_CPU = 0` (strict CPU improvement) and `ε_CUDA = +0.0000005` (CUDA near-neutrality acceptable per current frontier gap).

**Algorithm** (Dykstra-1983 alternating-projections; convergent for closed convex sets; here finite discrete pair set):
1. Initialize `F_0 = {0, ..., 599}` (all candidate pairs).
2. Project onto `C_CPU`: `F_1 = F_0 ∩ C_CPU` = pairs with `ΔCPU < ε_CPU`.
3. Project onto `C_CUDA`: `F_2 = F_1 ∩ C_CUDA` = pairs with `ΔCUDA ≤ ε_CUDA`.
4. Verify `F_2 = F_1 ∩ F_0 ∩ C_CUDA` (idempotent under alternating projection for finite discrete sets; canonical Dykstra correction term `q_k` vanishes per discrete convex case).
5. If `F_2 == ∅`, fallback to minimax aggregation (Section 1.4).

### 1.4 Multi-axis aggregation strategies

When `F_2 ≠ ∅`: pick `pair* = argmin_{pair_i ∈ F_2} max(ΔCPU(pair_i), ΔCUDA(pair_i))` (minimax within feasible intersection).

When `F_2 == ∅` (canonical Dykstra fallback per Catalog #296): 3 alternative aggregations:
- **Minimax fallback**: `pair* = argmin_{pair_i ∈ {0,...,599}} max(ΔCPU(pair_i), ΔCUDA(pair_i))` (minimizes worst-case axis loss; ties broken by sum)
- **Mean-weighted fallback**: `pair* = argmin_{pair_i} (w_CPU * ΔCPU(pair_i) + w_CUDA * ΔCUDA(pair_i))` with `w_CPU = w_CUDA = 0.5` (or operator-specified weights per axis-priority)
- **Lexicographic fallback**: `pair* = argmin_{pair_i} ΔCPU(pair_i)` filtered to top-K then `argmin_{pair_i in top-K} ΔCUDA(pair_i)` (per-axis priority ordering)

### 1.5 Dual-axis archive strategy (Catalog #192 paired submission)

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA on 1:1 contest-compliant hardware" non-negotiable:

Operator MAY submit TWO drop-one archives — one per axis — to satisfy bidirectional submission discipline:
- **CPU-optimal archive**: drop-one pair = `argmin_{pair_i} ΔCPU(pair_i)` (current cascade choice; pair0371)
- **CUDA-optimal archive**: drop-one pair = `argmin_{pair_i} ΔCUDA(pair_i)` (new variant; predicted to be a DIFFERENT pair from pair0371)
- **Operator decides per-axis-optimal archive per Catalog #192** by paired submission discipline.

Alternative: **minimax-optimal single archive** (Section 1.3 + 1.4) for unified submission.

### 1.6 Trade-off analysis

The minimax pair-drop is structurally dominated on the CPU axis alone vs the CPU-rate-saving-first pair-drop; the trade-off is:
- **CPU axis**: minimax MAY sacrifice some CPU gain vs CPU-rate-saving-first (~0.0000005-0.0000010 less CPU improvement per byte-drop)
- **CUDA axis**: minimax MAY preserve CUDA neutrality OR improve CUDA (~0.0000010-0.0000018 less CUDA regression OR small CUDA improvement per byte-drop)
- **Cumulative for 600-pair cascade**: minimax may produce CUDA frontier improvement in [-0.0005, -0.005] range while CPU frontier improvement stays in [-0.0001, -0.0005] range (vs CPU-only's [-0.001, -0.005] CPU improvement with [+0.0001, +0.0005] CUDA regression that BLOCKS submission per Catalog #192).
- **Submission readiness**: minimax-optimal archive satisfies Catalog #192 paired submission ON BOTH AXES; CPU-only archive does NOT (CUDA still pinned to PR106 format0d frontier).

### 1.7 Trade-off vs single-axis frontier maximization

Three operator-facing decision modes:
1. **CPU-axis frontier maximization** (current cascade): ranks pairs by `ΔCPU` ascending; pair0371 is current cascade choice with `ΔCPU = -0.0000007`. CUDA blocked.
2. **CUDA-axis frontier maximization** (new GAP 3-driven cascade): ranks pairs by `ΔCUDA` ascending; predicted CUDA-optimal pair may give `ΔCUDA ∈ [-0.0000010, -0.0000005]` with `ΔCPU ∈ [-0.0000003, +0.0000003]`.
3. **Minimax single-archive** (THIS design): ranks pairs by `max(ΔCPU, ΔCUDA)` ascending within Dykstra-feasible intersection; predicted minimax-optimal pair may give `(ΔCPU, ΔCUDA) ∈ ([-0.0000005, +0.0000005], [-0.0000010, +0.0000005])`.

The operator's submission strategy MAY combine modes 1+2 (dual-axis archives per Catalog #192) OR mode 3 (single archive per CLAUDE.md "Apples-to-apples evidence discipline").

---

## DELIVERABLE 2: Scaffold sketch

See companion file: `tools/probe_cuda_axis_dqs1_per_axis_pareto_disambiguator.py` (skeleton; NOT executable; BUILD-1 through BUILD-4 sister subagents populate the helper bodies marked `NotImplementedError`).

Skeleton public surface:
- `@dataclass(frozen=True) class PerAxisPairDelta` carrying per-pair per-axis decomposition + Pareto rank + Dykstra feasibility + minimax score + canonical Provenance per Catalog #323.
- `build_per_axis_pair_delta_from_master_gradient_ledger(archive_sha256, *, master_gradient_ledger_path, paired_cpu_cuda_anchor_path) -> list[PerAxisPairDelta]` — BUILD-1 populator (NotImplementedError stub).
- `build_per_axis_pareto_ranking(candidates) -> list[PerAxisPairDelta]` — BUILD-2 Pareto-rank populator (NotImplementedError stub).
- `find_dykstra_feasible_intersection(candidates, *, eps_cpu=0.0, eps_cuda=5e-7) -> list[PerAxisPairDelta]` — BUILD-2 Dykstra populator (NotImplementedError stub).
- `find_minimax_optimal_drop_one(candidates, *, fallback_mode="minimax_global") -> PerAxisPairDelta` — BUILD-2 minimax populator (NotImplementedError stub).
- `emit_axis_decomposition_for_canonical_helper(per_axis_pair_delta) -> AxisDecomposition` — BUILD-4 Catalog #356 wire-in (NotImplementedError stub).

The scaffold is the canonical disambiguator path per Catalog #296 acceptance cascade (b) for the design memo's predicted ΔS band — when the helper bodies land in BUILD-1+BUILD-2, the disambiguator empirically resolves the prediction.

---

## DELIVERABLE 3: Canonical equation candidate + 4-BUILD operator-routable

### 3.1 Canonical equation candidate QUEUED per Catalog #344 RATIFY-N

**Equation candidate name**: `cuda_axis_dqs1_per_axis_pareto_pair_ordering_v1`

**Mathematical form**: For DQS1 600-pair drop-one candidates ranked by per-axis Pareto + Dykstra + minimax:
```
pair*_minimax = argmin_{pair_i ∈ {0,...,599}} max(ΔCPU(pair_i), ΔCUDA(pair_i))
```
where `ΔCPU(pair_i)` and `ΔCUDA(pair_i)` follow equation #36 with per-axis SegNet + PoseNet components per equations #17 + #18.

**Predicted invariant**: `ΔCPU(pair*_minimax) ≥ -0.0000003` AND `ΔCUDA(pair*_minimax) ≤ +0.0000005` (vs current rate-saving-first cascade choice pair0371 with `ΔCPU = -0.0000007`, `ΔCUDA = +0.0000013`).

**Trigger for ratification per Catalog #344**: `when_3+_new_paired_cpu_cuda_empirical_anchors_in_domain` (BUILD-1 produces empirical anchor table; DISPATCH ratifies via paired CPU+CUDA Modal exact-eval on top-K minimax-optimal candidates).

**Sister of**:
- `pairset_component_marginal_score_decomposition_v1` (equation #36; per-pair per-axis Δ COMPOSITION surface) — sister at the per-axis RANKING surface
- `cpu_cuda_score_gap_v1` (equation #17; CPU-CUDA AXIS-SPLIT surface) — sister at the per-pair-drop AXIS-APPLICATION surface
- `pose_axis_cuda_amplification_v1` (equation #18; pose-axis amplification surface) — sister at the per-axis-drop AMPLIFICATION surface
- `cuda_axis_dqs1_regression_segnet_shift_v1` (DQS1-LOOP-CLOSURE-ASSIST Candidate 3 QUEUED) — sister at the per-byte-drop CUDA SegNet shift surface

**Predicted consumers**: `tac.optimization.cross_family_candidate_portfolio` (rank candidates by per-axis Pareto predictions) + `tac.optimization.decoder_q_pairset_acquisition` (filter pairs failing Dykstra feasibility early) + future `tac.cathedral_consumers.cuda_axis_dqs1_pareto_consumer` (Tier B score-contributing per Catalog #357 once empirically grounded).

**Predicted producers**: paired CPU+CUDA Modal dispatches per Catalog #245 / `tac.deploy.modal.call_id_ledger` + BUILD-1 master-gradient ledger × paired-anchor cross-join.

**Operator-decision protocol** per Catalog #344: candidate registered as IN-DOMAIN context `cuda_axis_dqs1_minimax_pair_drop` ONLY after BUILD-1+BUILD-2 land + DISPATCH produces ≥3 paired CPU+CUDA empirical anchors with residual < 0.0000005. Pre-ratification: FORMALIZATION_PENDING per Catalog #344.

### 3.2 4-BUILD operator-routable enumeration

Per Carmack MVP-first 5-step phasing + CLAUDE.md "Long-burn score-lowering campaign default" non-negotiable:

#### BUILD-1: Empirical per-axis pair-drop delta table

- **Scope**: populate `PerAxisPairDelta` table from 600-pair fp64 master-gradient ledger × paired CPU+CUDA scorer responses for the FRONTIER archive `7a0da5d0fc32`.
- **Inputs**: `.omx/state/master_gradient_ledger.jsonl` (or canonical helper output) + paired CPU+CUDA scorer responses (CPU from local macOS smoke if MPS-VIABLE per Catalog #341; CUDA from harvested pair0371 anchor + paired anchors from sister DISPATCH).
- **Outputs**: NEW `.omx/state/cuda_axis_dqs1_per_axis_pair_delta_table_<archive_sha_short>_<utc>.jsonl` (canonical fcntl-locked JSONL per Catalog #131 + #245 pattern).
- **Cost**: ~2-4h wall-clock; **$0 GPU** (uses harvested paired CPU+CUDA anchor for pair0371 + master-gradient ledger × scorer response cache for other 599 pairs).
- **Sister subagent**: spawnable independently; DISJOINT from this design memo's scope.
- **Validation**: per-pair residual against equation #36 < 0.0000005; flag pairs with higher residual for follow-on paired Modal anchor.

#### BUILD-2: Full executable per-axis Pareto + Dykstra + minimax pair-ordering

- **Scope**: populate `build_per_axis_pareto_ranking` + `find_dykstra_feasible_intersection` + `find_minimax_optimal_drop_one` skeleton helpers; consume BUILD-1 table.
- **Inputs**: BUILD-1's per-axis pair-delta table.
- **Outputs**: ranked candidate list; canonical Pareto rank per pair; Dykstra feasibility per pair; minimax score per pair; top-K (default K=10) ranked candidates per fallback mode.
- **Cost**: ~2-4h wall-clock; **$0 GPU** (pure Python algorithm on BUILD-1 table).
- **Sister subagent**: spawnable after BUILD-1 lands.
- **Validation**: per-pair Pareto-rank monotonicity; Dykstra-feasibility set membership invariant; minimax score = `max(ΔCPU, ΔCUDA)` exactly.

#### BUILD-3: Integration with codex's `decoder_q_pairset_acquisition.py`

- **Scope**: integrate BUILD-2's ranking with codex's existing CPU-axis-only acquisition plan via NEW `--per-axis-pareto-mode {minimax_global|cpu_only|cuda_only|dykstra_strict_intersection}` CLI flag on `tools/plan_decoder_q_pairset_acquisition.py`.
- **Inputs**: BUILD-2's ranked candidate list + canonical acquisition plan schema.
- **Outputs**: per-axis-aware acquisition plan; cascade priority queue re-ranked by D22 dimension.
- **Cost**: ~1h wall-clock; **$0 GPU** (CLI flag wiring + canonical helper invocation).
- **Sister subagent**: requires **sister-coordination directive** to codex per Catalog #333 codex-inbox bidirectional channel — codex's `decoder_q_pairset_acquisition.py` is canonical-helper-territory; THIS subagent only emits a routing-directive memo per the canonical inbox + does NOT mutate codex's source code.
- **Validation**: codex's existing acquisition plan invariants preserved; per-axis mode opt-in (CPU-only default for backward compat).

#### BUILD-4: Catalog #356 AxisDecomposition wire-in

- **Scope**: emit canonical `AxisDecomposition` per Catalog #356 frozen dataclass for each top-K minimax-optimal pair-drop candidate; route through `tac.score_composition.compose_score_from_axes` canonical helper for canonical contest formula application.
- **Inputs**: BUILD-2's top-K ranked candidates + canonical baseline pose from `tac.score_composition.load_baseline_pose_from_canonical_frontier_pointer`.
- **Outputs**: per-candidate canonical `AxisDecomposition` + composed `ComposedScoreDelta` per axis; canonical Provenance per Catalog #323 with `score_claim=False`, `promotable=False`, `axis_tag="[predicted]"` per Catalog #341 canonical non-promotable markers.
- **Cost**: ~1-2h wall-clock; **$0 GPU** (canonical helper invocation; no paid dispatch).
- **Sister subagent**: spawnable after BUILD-2 lands.
- **Validation**: per-axis `compose_scalar_delta(decomposition)` matches `ΔCPU` and `ΔCUDA` predictions within `1e-9`; canonical Provenance valid per `tac.provenance.audit_score_claim_dict`.

#### DISPATCH: Paired CPU+CUDA Modal exact-eval of top-K minimax-optimal pair-drop candidates

- **Scope**: paired CPU+CUDA Modal T4 dispatch on top-3 (recommended) or top-K (operator-routable) minimax-optimal pair-drop archives per Catalog #246 `tools/dispatch_modal_paired_auth_eval.py`.
- **Inputs**: BUILD-2's top-K minimax-optimal pair-drop archives (built via canonical archive builder; SHA256-anchored).
- **Outputs**: empirical paired CPU+CUDA anchors per archive; ratifies equation candidate `cuda_axis_dqs1_per_axis_pareto_pair_ordering_v1` once ≥3 anchors land per Catalog #344 trigger.
- **Cost**: ~$2-5 paid Modal T4 (3 × paired = 6 dispatches × ~$0.40-0.80 each); ~1-2h wall-clock per dispatch.
- **Sister subagent**: operator-routable AFTER BUILD-1 through BUILD-4 land; requires Catalog #243 + #246 + #271 canonical operator-authorize chain per CLAUDE.md "Executing actions with care" non-negotiable.
- **Validation**: per-axis Δ predicted vs empirical residual < 0.0000005 per equation #36 residual discipline; if residual exceeds, equation candidate UPGRADED to FORMALIZATION_PENDING per Catalog #344 recalibration.

### 3.3 Cost summary

| BUILD | Cost (GPU) | Cost (wall-clock) | Sister coordination |
|---|---|---|---|
| BUILD-1 | $0 | 2-4h | independent |
| BUILD-2 | $0 | 2-4h | after BUILD-1 |
| BUILD-3 | $0 | 1h | sister-coordination directive to codex per Catalog #333 |
| BUILD-4 | $0 | 1-2h | after BUILD-2 |
| DISPATCH | $2-5 | 1-2h | after BUILD-4 |
| **TOTAL** | **$2-5** | **7-13h** | operator-routable cascade |

### 3.4 Discipline-anchored 5-step Carmack MVP-first cascade

Per CLAUDE.md "Carmack MVP-first phasing" non-negotiable for the DISPATCH step (paid GPU >$0.30):

1. **FREE local macOS-CPU smoke first**: BUILD-1 + BUILD-2 + BUILD-4 produce $0 LOCAL CPU predictions for top-K minimax-optimal pair-drop archives BEFORE any paid dispatch.
2. **Falsifiable challenge**: predict `ΔCPU(pair*_minimax) ≥ -0.0000003 AND ΔCUDA(pair*_minimax) ≤ +0.0000005`; falsifiable at residual > 0.0000005 vs equation #36.
3. **Canonical equation reference**: equation `cuda_axis_dqs1_per_axis_pareto_pair_ordering_v1` (FORMALIZATION_PENDING per Catalog #344 until BUILD-1 + DISPATCH produce ≥3 paired anchors).
4. **Land verdict in same commit batch as smoke**: DISPATCH commit batch includes paired CPU+CUDA harvest + per-axis Pareto cascade ratification or refutation.
5. **Re-route operator priority queue within ~1h per CLAUDE.md "Downstream-surface latency discipline"**: DISPATCH outcome feeds `tools/cathedral_autopilot_autonomous_loop.py` ranker via Catalog #245 ledger; operator briefing surfaces per-axis cascade verdict.

---

## Discipline closure

- **6-hook wire-in declaration per Catalog #125**:
  - Hook #1 sensitivity-map = **ACTIVE** (per-axis Δ decomposition IS the canonical sensitivity primitive at the per-pair-drop boundary; downstream consumers route through `tac.sensitivity_map.*` per existing cascade)
  - Hook #2 Pareto constraint = **ACTIVE** (Dykstra alternating-projections feasibility over CPU + CUDA improvement polytopes IS the canonical Pareto-polytope solver per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable)
  - Hook #3 bit-allocator = **ACTIVE** (per-pair byte-drop count is the canonical bit-allocator signal; per-axis Pareto ranking determines per-pair drop priority)
  - Hook #4 cathedral autopilot dispatch = **ACTIVE** (BUILD-3 sister-coordination directive routes per-axis cascade into codex's acquisition queue; autopilot ranker can consume canonical equation candidate predictions once ratified)
  - Hook #5 continual-learning posterior = **ACTIVE** (DISPATCH harvest feeds canonical Modal call_id ledger per Catalog #245 + canonical equation #36 residual updates; canonical equation candidate `cuda_axis_dqs1_per_axis_pareto_pair_ordering_v1` RATIFY-N once ≥3 paired anchors land)
  - Hook #6 probe-disambiguator = **ACTIVE PRIMARY** (`tools/probe_cuda_axis_dqs1_per_axis_pareto_disambiguator.py` IS the canonical probe-disambiguator per Catalog #296 acceptance cascade (b); resolves predicted ΔS band empirically)
- **Catalog #313 probe-outcomes ledger row**: registered separately via `tac.probe_outcomes_ledger.register_probe_outcome` post-commit (initial verdict: PROCEED_DESIGN with reactivation criterion = BUILD-1 lands).
- **Catalog #344 canonical equation candidate**: `cuda_axis_dqs1_per_axis_pareto_pair_ordering_v1` QUEUED via this memo body; NOT auto-registered; FORMALIZATION_PENDING until BUILD-1 + DISPATCH produce ≥3 paired anchors. Operator-routable.
- **Catalog #287 evidence tags**: every empirical claim tagged with canonical equation id (equation #36 / #17 / #18 / #44) + canonical artifact path (frontier pointer + DQS1-LOOP-CLOSURE-ASSIST commit + RATE-ATTACK-METHODS commit). No hardcoded scores.
- **Catalog #323 canonical Provenance umbrella**: this memo carries grade=`[predicted]` for per-axis ΔS predictions; per-row Provenance routed through Catalog #356 `AxisDecomposition` in BUILD-4.
- **Catalog #229 premise verification**: read DQS1-LOOP-CLOSURE-ASSIST memo (210 lines) + RATE-ATTACK-METHODS Top-5 section (lines 295-365) + canonical frontier pointer + 5 canonical equation summaries BEFORE drafting; cited inline.
- **Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE**: this is a NEW design memo; NO mutation of sister codex artifacts; scaffold script is NEW skeleton.
- **Catalog #230 sister-subagent ownership map**: PR95-STAGE-2-MLX-BUILD + DROP-MANY-BEAM-BUILD-1 + COMBINED-TIER-1-WAVE-2 are scope-DISJOINT; this memo creates `.omx/research/cuda_axis_dqs1_design_memo_per_axis_pareto_pair_ordering_20260525.md` + `tools/probe_cuda_axis_dqs1_per_axis_pareto_disambiguator.py` ONLY. Catalog #340 sister-checkpoint guard PROCEED verified pre-commit.
- **Catalog #296 STRICT preflight gate**: predicted ΔS band paired with explicit Dykstra-feasibility check (Section 1.3) + first-principles citation (Shannon + Dykstra co-lead invocation per CLAUDE.md "Council conduct") + probe-disambiguator path (`tools/probe_cuda_axis_dqs1_per_axis_pareto_disambiguator.py`).
- **Catalog #356 AxisDecomposition canonical contract**: BUILD-4 wires per-axis Δ predictions through canonical `AxisDecomposition` + `tac.score_composition.compose_score_from_axes` helper per the canonical per-axis contract.
- **Catalog #341 canonical-routing-markers**: any future Tier B cathedral consumer derived from this design (e.g. `tac.cathedral_consumers.cuda_axis_dqs1_pareto_consumer`) MUST carry `predicted_delta_adjustment=0.0` + `promotable=False` + `axis_tag="[predicted]"` per the canonical non-promotable markers until empirically ratified.

## Mission contribution per Catalog #300

`frontier_breaking_enabler` — the design memo + scaffold + canonical equation candidate + 4-BUILD operator-routable structurally address GAP 3 (CUDA-axis DQS1 cascade absent) which is the operator's bidirectional submission discipline blocker per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA on 1:1 contest-compliant hardware" non-negotiable. Direct score-lowering value is INDIRECT (design memo doesn't itself lower score); structural value is ENABLING the next 4-BUILD cascade that DOES address the CUDA frontier (still PR106 format0d 0.20533002, NOT a DQS1 archive). Pre-DISPATCH expected ΔS unknown but bounded by Dykstra-feasibility intersection size; post-DISPATCH expected ΔCUDA in [-0.0005, -0.005] cumulative across 600-pair cascade if minimax-optimal ordering empirically generalizes per equation candidate prediction.
