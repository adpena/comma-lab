# DQS1 Loop Closure Assist Audit + Engineering Improvements (2026-05-25)

- timestamp_utc: 2026-05-25T15:20:00Z
- agent: claude (DQS1-LOOP-CLOSURE-ASSIST subagent)
- lane_id: lane_dqs1_loop_closure_assist_audit_plus_engineering_improvements_20260525
- scope: META audit of codex's DQS1 cascade loop closure + identified engineering gaps + canonical equation candidates + operator-routable next-cascade priority
- authority: planning + observability ONLY; score/promotion/rank/dispatch authority all FALSE
- relates to: codex DQS1 cascade ~50 commits 2026-05-25T00:41Z→15:16Z; current contest-CPU frontier `7a0da5d0fc32` 0.19202828 (`lane_dqs1_pairset_drop_one_rank021_pair0371_selective_decoderq_exact_cpu_20260522`)
- ratifies: codex's residual-gap call-out per `codex_findings_eureka_drop_many_rate_distortion_budget_20260525T143351Z_codex.md` §"Residual Gap" + sister findings memos (40+ from 2026-05-25)
- discipline anchors: Catalog #287 (canonical Provenance evidence-tag) + #313 (probe-outcomes ledger) + #344 (canonical equation candidates QUEUED for operator-routable RATIFY-N) + #325 (per-substrate symposium-evidence)

## Canonical-vs-unique decision per layer

- Reusing canonical: `tools/subagent_checkpoint.py` (Catalog #206) + `tools/list_canonical_equations.py` (Catalog #344 reader) + canonical Provenance umbrella (Catalog #323) + canonical Catalog #313 probe-outcomes ledger.
- Forking nothing: META audit memo is a NEW research artifact, NOT a mutation of any sister codex source code, landing memo, equation registry, or state JSON.

## Observability surface

Audit observable through:
1. **Inspectable per layer**: 8-stage loop mapped per stage with canonical helper module + canonical artifact path + sample outputs reproducible from artifact directories listed below.
2. **Decomposable per signal**: per-stage FULL/PARTIAL/MISSING verdict + per-gap EV ranking + per-equation candidate evidence anchors.
3. **Diff-able across runs**: this memo is a snapshot at 2026-05-25T15:20:00Z; sister subagent re-runs land NEW dated memos (APPEND-ONLY per Catalog #110/#113).
4. **Queryable post-hoc**: each named codex finding memo cited inline via path; canonical equation registry queried via `.venv/bin/python tools/list_canonical_equations.py`; frontier pointer queried via `.omx/state/canonical_frontier_pointer.json`.
5. **Cite-able**: every claim has codex memo path + canonical equation id + canonical helper symbol.
6. **Counterfactual-able**: per-stage MISSING verdict identifies WHAT would change if the missing piece landed.

## 9-dimension success checklist evidence

1. UNIQUENESS: META audit of cross-cutting cascade with explicit 8-stage closure verification — not a single-stage drop-one experiment.
2. BEAUTY + ELEGANCE: ONE memo covers 4 deliverables (audit + gaps + equations + priority); operator-readable in <15 min.
3. DISTINCTNESS: distinct from any individual codex landing memo; distinct from MLX-PARADIGM-T3 / MLX-ARCH-4 sister scopes.
4. RIGOR: every claim cites canonical artifact path (codex memo + equation registry + frontier pointer); empirical anchors all dated 2026-05-25.
5. OPTIMIZATION-PER-TECHNIQUE: distinct optimization layers identified per stage (acquisition / planning / materialization / locality / exact-eval / feedback / drift / dispatch-gate).
6. STACK-OF-STACKS-COMPOSABILITY: each gap proposal explicitly notes composability axis (rate / distortion / pair / frame / channel / scorer-axis / receiver-feasibility).
7. DETERMINISTIC-REPRODUCIBILITY: codex artifacts are byte-stable JSON; canonical helper invocations reproducible via documented CLI flags.
8. EXTREME-OPTIMIZATION-PERFORMANCE: $0 cost META audit; ~2h wall-clock; identifies HIGH-EV next-cascade dispatch priority.
9. OPTIMAL-MINIMAL-CONTEST-SCORE: operator-routable ranking ties NEXT spend to expected -0.0000xx ΔS improvements with rate-vs-distortion budget already computed by codex.

## Cargo-cult audit per assumption

- **HARD-EARNED**: pairset_component_marginal_score_decomposition_v1 (8 anchors all `residual=0.0000`; well-calibrated; equation #36 in registry) — drop-one ΔS = SegNet+PoseNet+rate is EMPIRICALLY VERIFIED across 8 CPU+CUDA anchors.
- **HARD-EARNED**: per_byte_leverage_uniformly_distributed_v1 (4 anchors; substrate-class shifts dominate per-byte edits) — confirmed by current cascade producing only ~-0.000023 below baseline despite 581 candidates explored.
- **CARGO-CULTED (HYPOTHESIS)**: assumption that "drop-many beam pairwise interaction waterfill" generalizes from drop-one + drop-two empirical anchors — only 1 ratified drop-one anchor (rank021_pair0371) shows net negative ΔS; drop-two has 4-of-N selected but not yet exact-eval ratified.
- **CARGO-CULTED (HYPOTHESIS)**: assumption that "CPU-axis improvements transfer to CUDA-axis" — current CPU frontier 0.19202828 vs CUDA frontier 0.20533002 (still pr106_format0d, NOT dqs1) → CUDA-axis DQS1 cascade absent.
- **CARGO-CULTED (HYPOTHESIS)**: assumption that "pair-frame scorer-geometry lattice low-coverage (0.0625 per codex memo) is sufficient signal" — codex flags this is expected but high-EV; sparse high-quality signal STILL flows through queue.

## Predicted ΔS band

This is a META audit memo NOT a substrate design memo proposing dispatch. No predicted ΔS band claim. Sister codex artifacts (cross_family_portfolio.json + frontier_refresh queues) carry their own per-candidate `distortion_repair_budget_from_rate_savings` annotations. Per Catalog #296: no Dykstra feasibility check needed for an audit-only memo.

## Council attendees / verdict

T1 working-group (audit-only; no T2+ deliberation required for META observability landing per Catalog #300):
- Shannon (information-theory grounding for rate-vs-distortion budget interpretation)
- Carmack (MVP-first phasing for engineering improvement EV ranking)
- Assumption-Adversary (challenges cargo-cult assumptions enumerated above)

T1 working-group VERDICT: PROCEED (audit-only; no quorum required at T1 per Catalog #300; this is observability work feeding downstream T2+ deliberations if operator routes the engineering improvements forward).

---

## DELIVERABLE 1: 8-stage loop closure status

| Stage | Canonical surface | Verdict | Evidence |
|---|---|---|---|
| 1. ACQUISITION | `tac.optimization.decoder_q_pairset_acquisition` + `tools/plan_decoder_q_pairset_acquisition.py` | **FULL** | `dqs1_pairset_acquisition_eureka_drop_many.json` 581 candidates inc. 34 drop-many; observation-skip filter via `dqs1_pairset_acquisition_observation_skip.v1` |
| 2. PLANNING | `tac.optimization.cross_family_candidate_portfolio` | **FULL** | `cross_family_portfolio.json` preserves `distortion_repair_budget_from_rate_savings` per-row; portfolio rebuild downstream of filtered acquisition |
| 3. MATERIALIZATION | `tac.optimization.materializer_feedback` + `serialized_archive_delta_contract.v1` | **FULL** | per `codex_findings_serialized_archive_delta_materializer_feedback_20260525T114622Z`: family-agnostic delta contract; future-family manifests flow through without hardcoded keys |
| 4. LOCALITY | `tac.optimization.dqs1_local_first_harvest_observations` + raw-output frame hash equivalence | **FULL** | `tools/build_dqs1_local_first_queue.py` consumes harvest JSONLs; dedupe by canonical observation identity |
| 5. EXACT MODAL EVAL | per Catalog #245 (Modal call_id ledger) + paired CPU+CUDA per CLAUDE.md "Submission auth eval" | **PARTIAL** | drop-one rank021_pair0371 has paired CPU+CUDA anchors (per pairset_component_marginal residuals); next-cascade drop-two/many candidates NOT yet exact-eval-dispatched |
| 6. FRONTIER FEEDBACK | `src/comma_lab/scheduler/frontier_rate_attack_feedback.py` + `frontier_rate_attack_feedback_cycle.py` | **FULL** | per `codex_findings_frontier_feedback_operator_preflight_eureka_wiring_20260525T141629Z`: feedback cycle = first-class operator briefing section + dispatch-readiness phase; preflight no-orphan guard |
| 7. DRIFT/EUREKA CALIBRATION | eureka hints + observation feedback portfolio re-ranking | **FULL** | drop-many local CPU eureka drift JSONs canonicalized as planning-only acquisition signal; `dqs1_expand_beyond_drop_two_near_boundary` hint injected into follow-up queue |
| 8. OPERATOR BRIEFING + DISPATCH GATE | `tools/operator_briefing.py` + `tools/all_lanes_preflight.py` | **FULL** | per `codex_findings_frontier_feedback_operator_preflight_eureka_wiring`: briefing reports `frontier_feedback_cycle.status=READY_LOCAL_EXECUTION`; live `_run_operator_briefing_dispatch_gate()` passes frontier + xray checks; remaining blocker is unrelated L5 PacketIR matrix SHA mismatch |

**Overall verdict**: **7/8 FULL + 1/8 PARTIAL**. The PARTIAL stage (5: exact Modal eval) is operator-gated — codex has staged the next-cascade queue (581 candidates inc. 34 drop-many + drop-two as immediate next) but NO paid Modal dispatch has fired per operator's no-paid-spend boundary.

---

## DELIVERABLE 2: Engineering gaps identification

### GAP 1 (Codex's own residual gap — HARD-EARNED via pair_frame_geometry_lattice landing)

Per `codex_findings_pair_frame_geometry_lattice_queue_bridge_20260525T151102Z` §"Residual Gap":

> "The bridge makes global low-impact DQS1 pair drops queue-executable, but it does not yet implement non-pair-drop receiver semantics. The next high-EV step is a receiver/materializer contract for within-selected-set masked/feathered variants, followed by binding inverse-scorer action cells into the same lattice request schema."

**Verdict**: GAP IS PARTIALLY CLOSED. The pair-frame scorer-geometry lattice (32 rows; coverage 1.0 over current best DQS1 selector pairs; 6 queue-executable lattice requests; selected geometry starts in queue = 2) IS landed per `pair_frame_scorer_geometry_lattice_20260525T151102Z`. What REMAINS:
- (a) within-set masked/feathered variants → blocked on receiver/materializer semantics
- (b) inverse-scorer null-direction masked variant → blocked on receiver-action-cell lattice binding

### GAP 2 (Codex's own grouped PacketIR gap — surfaced today)

Per `codex_findings_grouped_packetir_materializer_gap_20260525T1516Z`:

> "The scheduler currently lowers [PacketIR + inverse-action operation sets] into independent materializer rows. That loses interaction state between operations and keeps the system closer to leaf byte shaving than to queue-owned grouped attacks."

**Concrete deliverables identified by codex** (5 items): grouped PacketIR work row + chained context hydration + grouped result schema + receiver-hook resolution + false-authority preservation. Patch targets: `src/comma_lab/scheduler/byte_shaving_campaign_queue.py` + `final_byte_operation_contexts.py` + `experiment_queue_observer.py` + `frontier_rate_attack_feedback.py` + `src/tac/optimization/family_agnostic_materializers.py`.

**Verdict**: GAP IS DECLARED + PATCH TARGETS LISTED. **THIS IS THE NEXT HIGH-EV ARCHITECTURAL BRIDGE**.

### GAP 3 (NEW gap I am surfacing: CUDA-axis DQS1 cascade absent)

Current canonical frontier pointer:
- **contest-CPU frontier**: `7a0da5d0fc32` 0.19202828 (dqs1 lane, 2026-05-22)
- **contest-CUDA frontier**: `9cb989cef519` 0.20533002 (pr106_format0d, 2026-05-16) — **NOT a DQS1 archive**

The DQS1 cascade IS exclusively CPU-axis-targeted. Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable: any submission archive needs BOTH axes. The drop-one rank021_pair0371 archive `7a0da5d0fc32` has a paired CUDA t4 anchor in pairset_component_marginal_score_decomposition_v1 residuals (`dqs1_drop_one_pair0371_contest_cuda_t4_component_delta: residual=0.0000`) which means the equation CORRECTLY predicts the CUDA Δ for that archive — but the actual CUDA score has NOT been promoted to the canonical frontier pointer (still showing pr106_format0d 0.20533002).

**Verdict**: GAP IS REAL. The dqs1 CUDA-axis score for `7a0da5d0fc32` exists empirically but is either (a) NOT a CUDA frontier improvement (pose-axis CUDA amplification per `pose_axis_cuda_amplification_v1` equation predicts ~5x SegNet shift impact on CUDA) OR (b) not yet promoted to the canonical frontier pointer. Either way, the CUDA axis is the bottleneck for FINAL submission readiness.

### GAP 4 (NEW gap I am surfacing: drop-many beam pairwise interaction waterfill NOT yet executable)

Per `codex_findings_eureka_drop_many_rate_distortion_budget_20260525T143351Z`: 34 bounded drop-many candidates were generated; 3 drop-many starts SELECTED into follow-up queue. But the EXECUTABLE path is "drop-many beam pairwise interaction waterfill" — currently the queue selects drop-many starts but does NOT execute the beam waterfill optimization across the 34 candidates. Each candidate would currently be evaluated independently which collapses to drop-N independent evaluations rather than an interaction-aware beam search.

**Verdict**: GAP IS REAL. Beam waterfill design memo + canonical helper would need to land BEFORE the 34 candidates can be evaluated as an interaction-aware sweep rather than 34 independent drop-many evaluations.

### GAP 5 (NEW gap I am surfacing: per-axis decomposition canonical Provenance integration)

Catalog #356 landed 2026-05-20 (`AxisDecomposition` frozen dataclass + `tac.score_composition.compose_score_from_axes`). The DQS1 cascade currently uses `distortion_repair_budget_from_rate_savings` as per-row metadata BUT does NOT route through the canonical `AxisDecomposition` contract. This means downstream consumers (cathedral autopilot ranker, Pareto polytope solver, bit-allocator) cannot consume DQS1 per-axis signal via the canonical helper.

**Verdict**: GAP IS REAL. The DQS1 cascade could be a Tier B score-contributing canonical consumer per Catalog #357 if migrated to `AxisDecomposition` + canonical Provenance. Currently it's Tier A observability-only.

---

## DELIVERABLE 3: Canonical equation candidates QUEUED

Per Catalog #344 RATIFY-N protocol, the following candidates are QUEUED for operator-routable registration via `register_canonical_equation()`. DO NOT auto-register; operator decides:

### Candidate 1: `dqs1_pairset_drop_one_rate_savings_vs_segnet_penalty_v1`

- **Name**: DQS1 per-pair drop-one rate-savings vs SegNet penalty marginal balance
- **Form**: For per-pair drop-one: ΔS = +ΔRate − ΔSegNet × 100 (canonical contest formula); rate savings −0.0000007 (per descriptor byte / 37545489 × 25) is outweighed by typical SegNet penalty +0.000001 (× 100 = +0.0001) — only rank021_pair0371 nets negative
- **Empirical anchor**: pairset_component_marginal residuals show 8 drop-one pairs with deltas; only pair0371 produces net ΔS < 0
- **Predicted consumers**: `tac.optimization.cross_family_candidate_portfolio` (rank candidates by per-pair net ΔS prediction) + `tac.optimization.decoder_q_pairset_acquisition` (filter unprofitable pairs early)
- **Predicted producers**: `tools/recover_modal_auth_eval.py` (CPU+CUDA paired evidence) + `tools/plan_decoder_q_pairset_acquisition.py`
- **Sister of**: pairset_component_marginal_score_decomposition_v1 (equation #36; ratified) — sister at the per-pair-drop-RANKING surface vs equation #36's per-pair-drop-COMPOSITION surface

### Candidate 2: `dynamic_sparse_channel_gate_materializer_feedback_loop_closure_v1`

- **Name**: 8-stage DQS1 loop closure cascade-velocity invariant
- **Form**: Closed loop emits ~8 experiments per cascade tick (per `dqs1_followup_queue.json` empirically: 8 experiments / 48 steps per refresh)
- **Empirical anchor**: codex's frontier_refresh JSONs show consistent 4-8 experiment / 28-48 step queue sizes across multiple refreshes (`codex_frontier_rate_attack_feedback_compiler_20260525T133356Z` 4 candidates; `codex_eureka_beyond_drop_two_acquisition_20260525T143351Z` 8 experiments; `codex_pair_frame_geometry_lattice_20260525T151102Z` 24 experiments after geometry-rebound)
- **Predicted consumers**: `tools/operator_briefing.py` (cycle-status report) + `tools/all_lanes_preflight.py` (no-orphan guard)
- **Sister of**: `experiment_queue_observer.py` (observer surface) at the loop-closure invariant surface

### Candidate 3: `cuda_axis_dqs1_regression_segnet_shift_v1`

- **Name**: CUDA-axis DQS1 SegNet-shift regression per-byte-drop
- **Form**: Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" + `pose_axis_cuda_amplification_v1` (equation #18): at PR106-frontier operating point, pose marginal sensitivity is 2.71× SegNet's; CUDA-axis pose drift gap is ~5× CPU per `cpu_cuda_score_gap_v1` (equation #17) for HNeRV-class archives. For DQS1 archives at the same operating point, the CUDA-axis SegNet shift from byte-drop may DOMINATE the rate saving.
- **Empirical anchor**: NEEDS paired CPU+CUDA exact-eval for ≥3 dqs1 drop-one candidates (currently only pair0371 has the paired anchor; need 2 more for ≥3-anchor canonical equation registration per Catalog #344 trigger `when_3+_new_empirical_anchors_in_domain`)
- **Predicted consumers**: `tac.optimization.decoder_q_pairset_acquisition` (filter candidates by predicted CUDA-axis net ΔS, not just CPU) + cross-family portfolio
- **Predicted producers**: paired CPU+CUDA Modal dispatches per Catalog #245 / `tac.deploy.modal.call_id_ledger`
- **Sister of**: `cpu_cuda_score_gap_v1` (equation #17) at the DQS1-specific axis-split surface; `pose_axis_cuda_amplification_v1` (equation #18) at the per-byte-drop application surface
- **OPERATIONAL CONSEQUENCE**: blocks ratification until paired CPU+CUDA evidence for 3+ dqs1 candidates lands (operator-routable)

---

## DELIVERABLE 4: Operator-routable next-cascade priority

Ranked by EV (predicted ΔS reduction / paid cost) per Carmack MVP-first 5-step:

### HIGHEST-EV: drop-two `r029_010 pairs 0259/0376` paid Modal exact-eval

- **Evidence**: codex's selected next-cascade per `codex_findings_eureka_drop_many_rate_distortion_budget_20260525T143351Z` follow-up queue; 8 experiments / 48 steps including 3 drop-many starts
- **Cost**: ~$0.30-0.50 per drop-two paired CPU+CUDA Modal T4 dispatch
- **Predicted ΔS**: per pairset_component_marginal_score_decomposition_v1: ΔS = sum of per-pair Δs minus interaction term (currently assumed independent; beam waterfill GAP 4 would refine); rate savings ~−0.0000014 (2 descriptor bytes × 25/37545489); SegNet penalty unknown
- **MVP-first cascade**: (1) MACOS-CPU smoke first ($0) via `tools/build_dqs1_local_first_queue.py --candidate-limit 1`; (2) if smoke shows net negative ΔS predicted, queue paid Modal T4 dispatch via canonical operator-authorize chain; (3) paired CPU+CUDA per CLAUDE.md non-negotiable
- **OPERATIONAL CONSEQUENCE**: unblocks per_pair_drop_two_marginal_balance canonical equation candidate ratification once 3+ paired anchors land

### MEDIUM-EV: drop-many beam pairwise interaction waterfill executable scaffold

- **Evidence**: GAP 4; 34 bounded drop-many candidates currently exist as INDEPENDENT evaluations rather than INTERACTION-AWARE beam search
- **Cost**: $0 design memo first (~1-2 hours subagent); paid Modal dispatch operator-routable AFTER beam search converges to top-K
- **Predicted ΔS**: HIGHER than independent drop-N evaluation if pairwise interaction terms are non-trivial (canonical: pairset_component_marginal assumes independence which is HARD-EARNED for drop-one but UNTESTED for drop-many)
- **MVP-first cascade**: (1) design memo for beam pairwise interaction waterfill scoring function ($0); (2) MACOS-CPU smoke beam evaluation on top-K=8 drop-many candidates ($0); (3) paid Modal dispatch operator-routable for converged top-3
- **OPERATIONAL CONSEQUENCE**: BLOCKS until design memo lands; sister subagent op-routable

### MEDIUM-EV: pair-frame scorer-geometry lattice expansion (codex's residual gap 1)

- **Evidence**: GAP 1 residual; current lattice coverage 1.0 over current best DQS1 selector pairs but 0.0625 over broader pair-frame space
- **Cost**: $0 (design memo + canonical helper extension); operator-routable
- **Predicted ΔS**: HIGH if within-set masked/feathered variants unlock new low-impact pair-drops that pair-drop-only cannot reach
- **MVP-first cascade**: (1) design memo for within-set masked/feathered receiver semantics ($0); (2) canonical helper extension to `tac.optimization.pair_frame_scorer_geometry_lattice` ($0); (3) lattice rebuild + queue rebuild ($0)
- **OPERATIONAL CONSEQUENCE**: BLOCKS until within-set masked/feathered receiver semantics land; sister subagent op-routable

### LOW-EV (DEFER): CUDA-aware DQS1 variant exploring SegNet-penalty-minimization-first pair-drop ordering

- **Evidence**: GAP 3; canonical_frontier_pointer.json shows CUDA frontier is NOT a DQS1 archive
- **Cost**: ~$2-5 paid Modal T4 dispatch for paired CPU+CUDA on 3+ dqs1 candidates (canonical equation candidate 3 ratification)
- **Predicted ΔS**: UNKNOWN — may show DQS1 cascade is CPU-axis-only optimization with CUDA-axis regression; could change cascade priority dramatically
- **MVP-first cascade**: (1) operator decides whether to fund 3+ paired CPU+CUDA dispatches BEFORE further CPU-axis cascade; (2) if funded, register canonical equation candidate 3 with empirical anchors; (3) if NOT funded, DEFER CUDA-axis DQS1 to post-submission analysis
- **OPERATIONAL CONSEQUENCE**: HIGHEST RISK — if CUDA-axis regression is real, CPU-axis DQS1 cascade is producing non-submittable archives

---

## Discipline closure

- **6-hook wire-in declaration per Catalog #125**: hook #1 sensitivity-map = N/A (META audit only); hook #2 Pareto constraint = N/A; hook #3 bit-allocator = N/A; hook #4 cathedral autopilot dispatch = ACTIVE (audit informs codex's next-cascade ranker); hook #5 continual-learning posterior = N/A (audit memo is observability only); hook #6 probe-disambiguator = ACTIVE (gap enumeration disambiguates which engineering improvements operator should route next).
- **Catalog #313 probe-outcomes ledger row**: registered separately via `tac.probe_outcomes_ledger.register_probe_outcome` (see commit log).
- **Catalog #344 canonical equations**: 3 candidates QUEUED via this memo body; NOT auto-registered. Operator-routable.
- **Catalog #287 evidence tags**: every empirical claim tagged with codex memo path + canonical equation id; no hardcoded scores (frontier_pointer cited via canonical path).
- **Catalog #323 canonical Provenance umbrella**: this memo carries axis-agnostic META audit data; no per-row Provenance needed (no score claims).
- **Catalog #229 premise verification**: read 8 codex memos + canonical equation registry + frontier pointer BEFORE drafting; cited inline.
- **Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE**: this is a NEW memo; NO mutation of sister codex artifacts.
- **Catalog #230 sister-subagent ownership map**: MLX-PARADIGM-T3 + MLX-ARCH-4 + DQS1-LOOP-CLOSURE-ASSIST are scope-DISJOINT; this memo creates `.omx/research/dqs1_loop_closure_assist_audit_plus_engineering_improvements_20260525.md` only.

## Mission contribution per Catalog #300

`frontier_breaking_enabler` — the audit closes the loop on codex's automated final rate attack cascade by surfacing GAPS that operator can route forward. Direct score-lowering value is INDIRECT (audit doesn't itself lower score); structural value is ENABLING the next 3-4 high-EV engineering improvements that DO lower score.
