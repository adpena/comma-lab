# Codex routing directive: CHEAP-PROBE WAVE — pose-axis OP-1+OP-2+OP-6+OP-7+OP-10
# Date: 2026-05-18
# Authority: T3 grand council symposium `feedback_pose_axis_non_hnerv_t3_council` verdict PROCEED_WITH_REVISIONS
#   Memo: .omx/research/grand_council_t3_pose_axis_non_hnerv_paths_to_frontier_breaking_symposium_20260518.md
#   Commit: bfce23a5d
#   Council quorum: 19-of-20 (T3 threshold ≥12-of-20)
#   Sextet pact: 6-of-6 unanimous on cheap-probe family
# Operator: approved 2026-05-18 ("Approved")
# Per CLAUDE.md "Frontier target — NON-NEGOTIABLE, HIGHEST EMPHASIS" + Catalog #325 per-substrate symposium PROCEED
# Per CLAUDE.md "Mission alignment — non-negotiable" Consequence 4 (frontier-breaking dominates rigor budget)

## CANONICAL POINTERS (read FIRST)

1. `/Users/adpena/Projects/pact/CLAUDE.md` (FULL; especially "Frontier target" + "Meta-Lagrangian/Pareto solver" + "SegNet vs PoseNet importance — operating-point dependent" + Catalog #270 + #318 + #319 + #322 + #325)
2. `/Users/adpena/Projects/pact/AGENTS.md`
3. `.omx/research/grand_council_t3_pose_axis_non_hnerv_paths_to_frontier_breaking_symposium_20260518.md` (THE AUTHORITY MEMO; read OP-1+OP-2+OP-6+OP-7+OP-10 sections verbatim)
4. `.omx/state/council_deliberation_posterior.jsonl` (council anchor for traceability)
5. `tools/extract_master_gradient.py` (Codex's just-landed extractor; OP-7 consumer)
6. `src/tac/master_gradient.py` + `src/tac/master_gradient_consumers.py` (canonical helpers OP-7 + OP-2 build on)
7. `src/tac/codec/wyner_ziv_layer.py` (OP-1 substrate)
8. `src/tac/optimization/substrate_composition_matrix.py` (OP-10 autopilot consumer)
9. `tools/cathedral_autopilot_autonomous_loop.py` (OP-10 wire-in surface)
10. `reports/latest.md` (current frontier: 0.19205 [contest-CPU] / 0.20533 [contest-CUDA] per Catalog #316)
11. `.omx/state/master_gradient_anchors.jsonl` (existing per-pair fp64 anchors; PR101_lc_v2 `f174192aeadf...` is the canonical anchor)

## STRATEGIC FRAMING

Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" 2026-05-04 update, at PR106 frontier (pose_avg ~3.4e-5) **pose marginal is 2.71× SegNet's**. The 5 cheap probes below TARGET POSE AXIS specifically. Predicted aggregate frontier displacement under realistic α-adjusted composition per Catalog #322 sub-additive/saturating bands: **0.19205 → [0.182, 0.189] [contest-CPU prediction]**.

Total cost cap: **$1 GPU** (largely $0-CPU; OP-1 + OP-6 may invoke Modal CPU dispatch). Each probe is INDEPENDENTLY testable before composition smoke.

## EXECUTION ORDER (council-binding)

### OP-7 FIRST: Direct master-gradient pose-byte hoist
Predicted ΔS: -0.002 / Cost: $0 CPU / Codex's just-landed extractor IS the canonical consumer

#### What Codex builds
- `tools/hoist_pose_bytes_from_master_gradient.py` (~250 LOC) — reads `.omx/state/master_gradient_anchors.jsonl` for `PR101_lc_v2` archive `f174192aeadf...`; selects top-K pose-axis-dominant bytes per `score_axis_dominance` field; emits canonical `CandidateModificationSpec` per Catalog #318 typed-operator API; writes manifest at `experiments/results/pose_byte_hoist_op7_20260518/manifest.json`
- `src/tac/master_gradient_consumers.py` extension: new `select_pose_axis_dominant_bytes(archive_sha256, *, top_k=128, axis_dominance_threshold=0.7) -> tuple[CandidateModificationSpec, ...]`
- Tests: `src/tac/tests/test_select_pose_axis_dominant_bytes.py` covering threshold semantics + top-k bounds + axis-dominance correctness + integration with `CandidateModificationSpec`

#### Discipline
- Catalog #318 typed-operator API: NO raw byte modification arrays; route through `CandidateModificationSpec` + `grammar_aware_operator`
- Catalog #313 probe-outcome registration via `tac.probe_outcomes_ledger.register_probe_outcome` after first empirical result lands
- Catalog #131 fcntl-locked writes to `.omx/state/`
- Catalog #229 premise verification BEFORE editing: confirm `master_gradient_anchors.jsonl` has `score_axis_dominance` field for the PR101_lc_v2 anchor (if not, OP-7 has a DEPENDENCY on Codex's extractor adding this field; surface as inbox question via the just-landed `tools/codex_to_claude_inbox.py ask` channel once it's built)

#### Smoke (local CPU; $0)
- Hoist top-K=128 pose-axis-dominant bytes → emit operator-spec → run `tools/verify_distinguishing_feature_byte_mutation.py` per Catalog #272 to confirm bytes affect downstream output
- Predicted band: ΔS ∈ [-0.003, -0.001] [contest-CPU prediction]; `predicted_band_validation_status: pending_post_training` per Catalog #324

### OP-2: Master-gradient pose-byte classification extension
Predicted ΔS: -0.002 / Cost: $0 CPU / Extends OP-7 via per-pair classification granularity

#### What Codex builds
- `src/tac/master_gradient_consumers.py` extension: `classify_pose_bytes_by_pair_variance(archive_sha256) -> tuple[VennCell, ...]` — extends existing `classify_bytes_by_pair_variance` (Catalog #319) with explicit pose-axis cell filtering
- Tests: 6+ dedicated tests in existing `test_master_gradient_consumers.py` covering pose-axis-cell filter + bypass + composition with Catalog #319 v2 cascade
- Wire-in to `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_venn_classification_v2` Cascade 3 (passthrough) — when Wyner-Ziv DeliverabilityProof exists AND pose-axis cell is HIGH_PAIR_INVARIANT, apply 1.05× reward (smaller than HIGH_PAIR_INVARIANT general 1.20× per Catalog #319 v2)

#### Discipline
- Catalog #319 Q1-Q5 backward compat preserved (consumer chain unchanged)
- Catalog #322 phantom-provenance composition_alpha gate: pose-axis classification IS valid contest-archive-member provenance (NOT research-sidecar phantom)
- Catalog #323 canonical provenance contract: every emitted classification row carries `tac.provenance.Provenance` per `requires_canonical_provenance` decorator

#### Smoke (local CPU; $0)
- Run classifier on PR101_lc_v2 anchor → confirm pose-axis-dominant cells distinct from seg-axis-dominant + rate-axis-dominant
- Confirm autopilot Cascade 3 reward fires when Wyner-Ziv proof + pose-cell both present

### OP-10: Autopilot Cascade 2 extension to per-pair pose-axis reward
Predicted ΔS: -0.001 / Cost: $0 / Cathedral autopilot rerank consumes OP-7+OP-2 outputs

#### What Codex builds
- `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_venn_classification_v2` extension: when `optimal_per_pair_treatment_plan` exists AND treatment plan's per-pair `pose_axis_weight > 0.5`, apply additive -0.0005 reward (small but cumulative across 600 pairs = -0.30 in aggregate-weighted terms, but heavily α-discounted per Catalog #322 to -0.001 realistic)
- Tests: 8+ dedicated tests in `test_cathedral_autopilot_z1_revision.py` covering Cascade 2 per-pair pose-axis-weight extension + α-saturation behavior + composition with OP-7 OP-2

#### Discipline
- Cascade 2 PASSTHROUGH semantics preserved (no replacement of Cascade 1 Lagrangian-dual primary)
- Per CLAUDE.md "Council conduct" + Catalog #319 Q3 architectural correction: planner is canonical answer; rewards are advisory bounded [1.0, 2.0]

### OP-1: Wyner-Ziv pose-residual Tier-1 hoist (deterministic optical-flow side-info)
Predicted ΔS: -0.003 / Cost: $0-$0.30 / Builds on existing `src/tac/codec/wyner_ziv_layer.py`

#### What Codex builds
- `src/tac/codec/wyner_ziv_pose_residual_hoist.py` (~400 LOC) — implements Wyner-Ziv 1976 side-information-at-decoder theorem applied to pose-axis residuals:
  - Encoder: compute optical-flow side-info via canonical `tac.optical_flow.raft_flow_estimator` (deterministic; no neural inference at inflate time per CLAUDE.md "Strict scorer rule")
  - Decoder: leverage side-info to reduce per-pair pose-residual entropy
  - Tier-1 hoist (per Catalog #319 DeliverabilityTier.TIER_1_ZERO_COST): the side-info bytes are derived deterministically at inflate-time from the archive's existing pose-residual bytes; ZERO additional archive cost
- `tac.wyner_ziv_deliverability.build_deliverability_proof_from_wyner_ziv_classification` extension to accept pose-residual provenance type
- Tests: 12+ dedicated tests covering encode/decode roundtrip + zero-archive-cost invariant + canonical helper integration

#### Discipline
- Catalog #319 Tier 1 strict-scorer-rule: deliverability_proof must validate `bytes_added = 0` (Tier 1 = zero-cost; canonical operator-readable per HNeRV parity L4)
- Catalog #213 Comma2k19 canonical helper N/A (Wyner-Ziv operates on contest video only; no external data dependency)
- Catalog #318 raw-byte-API ban: ALL byte modifications routed through typed `CandidateModificationSpec`
- Catalog #229 premise verification: confirm `tac.optical_flow.raft_flow_estimator` exists in repo (if not, surface inbox question via just-landed channel)

#### Smoke (local CPU; $0-$0.30 if Modal needed)
- Encode/decode on PR101_lc_v2 pose-residual bytes; confirm zero archive cost; measure decoded MSE vs baseline
- Predicted band: ΔS ∈ [-0.004, -0.002] [contest-CPU prediction]; `predicted_band_validation_status: pending_post_training` per Catalog #324

### OP-6: Pose-FOE sparse encoding LFV1 (Telescope + LAPose 2-stage)
Predicted ΔS: -0.005 / Cost: $0-$1 / Cross-references TT5L V2 redesign + Time-traveler L5 telescopic foveation

#### What Codex builds
- `src/tac/codec/pose_foe_sparse_lfv1.py` (~600 LOC) — Latent-Foveal-Variable v1:
  - Stage 1 (Telescope): per-frame focus-of-expansion (FOE) detection via canonical optical-flow analysis (deterministic; no neural inference at inflate time)
  - Stage 2 (LAPose): Linear Approximate Pose encoding using FOE prior as sparse basis
  - Sparse encoding: only encode pose-residuals OUTSIDE the FOE cone (typically <20% of frame area; sparsity exploit per Daubechies wavelet hierarchical compression discipline)
- Tests: 15+ dedicated tests covering FOE detection determinism + LAPose encoding accuracy + sparsity bound + canonical helper integration
- Sister design memo: cross-reference `.omx/research/tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_memo_20260518.md` (already landed; uses similar Telescope + LAPose primitives)

#### Discipline
- Catalog #311 ego-motion-conditioned predictive coding requirement: LFV1 ego-motion conditioning via FOE prior IS canonical per Gibson 1950 + Atick-Redlich 1990 cooperative-receiver
- Catalog #310 F-asymptote class-shift discipline: LFV1 is BOLT-ON to existing substrates (NOT primary class-shift); declare `lane_class=bolt_on_codec` NOT `lane_class=substrate_class_shift`
- Catalog #229 premise verification: confirm `tac.optical_flow` package exists (sister of OP-1 dependency check)

#### Smoke (local CPU; $0)
- FOE detection on first 10 frame pairs of contest video; confirm cone-area sparsity < 25%
- LAPose encoding accuracy on FOE-outside pose-residuals; confirm < 5% reconstruction error
- Predicted band: ΔS ∈ [-0.007, -0.003] [contest-CPU prediction]; `predicted_band_validation_status: pending_post_training` per Catalog #324

## COMPOSITION ORDER (after all 5 land independently)

Per Catalog #322 v2 cascade composition_alpha:
1. **OP-7 + OP-2 = SUB-ADDITIVE** (α ≈ 0.7; both consume same master-gradient anchor; partial overlap on pose-byte selection)
2. **OP-7 + OP-2 + OP-10 = ADDITIVE** (α ≈ 1.0; OP-10 is autopilot rerank consumer of OP-7+OP-2 outputs; no overlap)
3. **OP-7+OP-2+OP-10 + OP-1 = ADDITIVE** (α ≈ 1.0; OP-1 Wyner-Ziv operates on pose-residual ENTROPY surface; OP-7+OP-2 operate on pose-byte SELECTION surface; orthogonal)
4. **OP-7+OP-2+OP-10+OP-1 + OP-6 = SUB-ADDITIVE** (α ≈ 0.6; OP-6 LFV1 partially overlaps OP-1 Wyner-Ziv on optical-flow primitive; FOE detection is sister of side-info derivation)

Aggregate predicted ΔS under realistic α-adjusted composition: **-0.013** (vs naive sum -0.013; α-discounted to -0.010 conservative)

Frontier displacement: 0.19205 → 0.182 [contest-CPU prediction conservative]; upper band 0.189 [contest-CPU prediction realistic].

## DISCIPLINE (Codex execution per OP)

All standard discipline applies per OP:
- Catalog #229 premise verification BEFORE each OP's first edit
- Catalog #117/#157/#174 commit serializer with POST-EDIT sha for EVERY file edit
- Catalog #186 catalog # claim transactional (if new STRICT gate needed)
- Catalog #206 checkpoint discipline every ~10 tool uses
- Catalog #131 fcntl-locked writes to `.omx/state/`
- Catalog #314 absorption avoidance: Codex owns these files; sister-subagents (Claude-spawned design memos) own ONLY `.omx/research/*.md`
- Catalog #313 register probe outcome AFTER first empirical anchor lands per OP
- Catalog #324 set `predicted_band_validation_status: pending_post_training` on every cost-band-posterior write for these OPs
- Catalog #325 per-substrate symposium PROCEED satisfied via THIS T3 council verdict
- Catalog #287 evidence-tag discipline on every ΔS / percentage claim

## INTEGRATION WITH PERSISTENT /goal LOOP (v2.4)

Codex picks up this directive on next /goal LOOP iteration via POINTERS glob `.omx/research/codex_routing_directive_*.md`. Execute in order:

1. PRE-FLIGHT step 1: read this directive + read pose-axis council memo for cited OP details
2. SELECT step 3: queue OP-7 first (cheapest dep + already-landed extractor)
3. CLAIM step 4: create canonical_task_status row per OP with `task_id=cheap_probe_wave_op_<N>_pose_axis_council` + `owner=codex`
4. EXECUTE step 5: build per OP; if ambiguity surfaces (e.g., extractor field missing), USE inbox channel `tools/codex_to_claude_inbox.py ask` (LANDED earlier via routing directive `745fc2e19`) instead of guessing/blocking
5. REVIEW step 6: codex:adversarial-review per OP landing
6. PERSIST step 7: per OP: update canonical_task_status to 'completed' + emit relay via inbox channel if novel observation surfaces (e.g., "OP-7 ΔS empirical -0.0018 outside band [-0.003, -0.001] — band needs refinement")
7. After ALL 5 OPs land: run composition smoke + emit `feedback_cheap_probe_wave_landed_20260518.md`

## EXIT CRITERIA (Codex done when ALL true)

- [ ] OP-7 hoist tool + extension + tests landed; smoke confirms top-K byte-mutation produces downstream change
- [ ] OP-2 classifier + autopilot Cascade 3 wire-in landed; tests pass
- [ ] OP-10 Cascade 2 extension landed; tests pass
- [ ] OP-1 Wyner-Ziv pose-residual hoist landed; encode/decode roundtrip works; zero-archive-cost invariant verified
- [ ] OP-6 LFV1 Telescope + LAPose landed; FOE detection + LAPose encoding tests pass
- [ ] Composition smoke (all 5 stacked on PR101_lc_v2 anchor) emits aggregate ΔS prediction with `predicted_band_validation_status: pending_post_training`
- [ ] 5 probe-outcome rows in `.omx/state/probe_outcomes.jsonl` per Catalog #313
- [ ] 5 canonical_task_status rows updated 'completed' with commit_shas + test_status
- [ ] Council deliberation posterior anchor referenced (council_id from `.omx/state/council_deliberation_posterior.jsonl` matching the pose-axis council)
- [ ] Final landing memo `feedback_cheap_probe_wave_landed_20260518.md` per Catalog #229+#287 evidence-tag discipline

## EMPIRICAL DISPATCH (operator-gated)

After ALL 5 probes land + composition smoke confirms predicted band:
- **Stage 1**: $0.30 Modal CPU paired auth eval per Catalog #225 dual-axis discipline (PR101_lc_v2 + composed cheap-probe stack) — confirms ΔS direction
- **Stage 2 (operator-approved)**: $5-15 Modal A100 100ep canary if Stage 1 confirms predicted band — produces first contest-grade empirical anchor for the cheap-probe family
- **Stage 3 (operator-approved post Stage 2)**: PR submission packet if Stage 2 lands sub-0.19205 [contest-CPU]

## SISTER SUBAGENT COORDINATION

In-flight at directive-write time (2-cap):
- Codex session `019de465` continuing ITEM_3 master-gradient extractor (DEPENDENCY for OP-7+OP-2)
- `a278dc871d4ce1461` TROPICAL d_seg solver design memo — owns `.omx/research/tropical_d_seg_solver_design_memo_20260518.md`
- Pose-axis T3 council `ae3c4b603a3931d74` COMPLETED (slot freed; this directive consumes the freed slot's spirit but is main-thread work — no new subagent spawned for THIS directive write)

DISJOINT scope: Codex's existing per-pair master-gradient extractor work IS what OP-7+OP-2 build on (the work is COMPLEMENTARY, not collision); Tropical d_seg subagent owns `.omx/research/*.md` only.

— Main-Claude 2026-05-18 (council-authorized routing per T3 PROCEED_WITH_REVISIONS verdict)
