---
schema: pact_design_memo_v1
memo_id: rate_attack_vector_2_g1_cpu_axis_optimization_design_memo_20260518
review_date: "2026-05-18"
lane_id: lane_rate_attack_g1_cpu_axis_specific_20260518
parent_master_memo: rate_attack_43_vectors_meta_paradigm_deep_research_20260518
meta_paradigm_anchor: structural_information_not_shipped_meta_paradigm_unification_20260518
vector_id: G1
vector_name: "CPU-axis-specific optimization (exploit empirical PR102 +0.033 CPU-CUDA gap)"
horizon_class: frontier_breaking
predicted_band_validation_status: pending_post_training
score_claim: false
promotion_eligible: false
research_only: true
write_scope: ".omx/research only"
canonical_frontier_anchor:
  contest_cpu: "0.19205 [contest-CPU]"
  contest_cuda: "0.20533 [contest-CUDA T4]"
predicted_delta_band_contest_cpu: "[-0.010, -0.003]"
council_tier_assignment: T2_inner_skunkworks_sextet_pact
target_modes:
  - contest_exact_eval
  - contest_generalized
deployment_target: leaderboard_cpu_priority
hardware_substrate: linux_x86_64_cpu_for_axis_optimal_archive
---

# TOP-2 Design Memo — Vector G1: CPU-Axis-Specific Optimization

**Master memo**: `.omx/research/rate_attack_43_vectors_meta_paradigm_deep_research_20260518.md`
**META-paradigm**: SINS — `.omx/research/structural_information_not_shipped_meta_paradigm_unification_20260518.md`
**Lane**: `lane_rate_attack_g1_cpu_axis_specific_20260518` L0

## 0. Executive Summary

**HARD-EARNED-VERIFIED EMPIRICAL ANCHOR** per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA":
- PR102 (third prize): CUDA 0.22839, CPU 0.19538, **Δ = +0.033 (CUDA worse)**
- PR107 (ours): CUDA 0.22936, CPU 0.19664, **Δ = +0.033 (consistent)**
- PR101 (gold): CPU 0.193 (no CUDA posted)

**The leaderboard ranks by CPU axis, not CUDA.** Every contest archive optimized for the CUDA axis at the cost of CPU is shipping bytes that worsen leaderboard ranking by ~0.033 vs the CPU-optimal alternative.

**G1 exploit**: re-rank existing PR101+102+103+106+107 archives by CPU axis ONLY; for each archive family, select the per-axis-optimal variant. This is essentially FREE (no GPU spend; pure ranker re-computation on existing dual-eval data).

**Hotz binding directive**: PROCEED IMMEDIATELY within this session.

**Predicted ΔS**: [-0.010, -0.003] [contest-CPU].

## 1. Canonical-vs-unique Decision Per Layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| `tools/scan_best_anchor_per_axis.py` | ADOPT_CANONICAL | Catalog #316 frontier signal-loss canonical helper |
| `tac.frontier_scan.Anchor` | ADOPT_CANONICAL | Existing canonical dataclass |
| Per-axis ranker | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | Current ranker selects best-of-BOTH-axes; G1 needs CPU-axis-only ranker |
| Submission archive selection | ADOPT_CANONICAL (post-G1 selection) | Once G1 picks the CPU-optimal archive, submission proceeds through canonical custody chain |
| Cathedral autopilot routing | ADOPT_CANONICAL + small extension | Add CPU-axis-priority field to candidate metadata |
| Catalog #316 frontier ledger | ADOPT_CANONICAL | The G1 verdict updates `reports/latest.md` FRONTIER section |
| CPU-CUDA paired anchor preservation | ADOPT_CANONICAL | CUDA score remains as transparency artifact per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable; G1 PRIORITIZES CPU but does NOT discard CUDA |

**1 fork; 6 canonical adoptions.** G1 is the THINNEST substrate engineering on the TOP-5 list — ~50 LOC total.

## 2. 9-Dimension Success Checklist Evidence (per Catalog #294)

### Dim 1: UNIQUENESS
G1 is class-shift: it operates on the CPU-axis-priority dimension. No prior contest archive selection optimizes for CPU-axis only.

### Dim 2: BEAUTY + ELEGANCE
~50 LOC extension to `tools/scan_best_anchor_per_axis.py`. 30-second-reviewable. Single function `select_cpu_optimal_archive(archive_set)` returns the CPU-best per-family.

### Dim 3: DISTINCTNESS
G1 vs G2-G7: G1 is the AGGREGATE CPU-axis exploit; G2-G7 are SPECIFIC sub-mechanisms (AVX-512 / MKL / fp80 / cache-line / per-byte / inflate-device). G1 covers them all empirically.

### Dim 4: RIGOR
- Premise verification: PR102 + PR107 dual-eval data is HARD-EARNED-VERIFIED per CLAUDE.md
- Empirical anchor: existing PR101+102+103+106+107 dual-eval data (already collected)
- No new probe needed (the empirical anchor IS the dual-eval data)

### Dim 5: OPTIMIZATION PER TECHNIQUE
G1's substrate-optimal engineering: per-axis ranker; ZERO GPU spend; pure CPU re-computation.

### Dim 6: STACK-OF-STACKS-COMPOSABILITY
G1 is SUB with all other rate-attack vectors (composition_alpha 0.5-1.5 sub-additive) — applying G1 AFTER F1+B1+H1 re-ranks the post-vector archives by CPU axis.

### Dim 7: DETERMINISTIC REPRODUCIBILITY
Re-ranking is deterministic; same dual-eval data → same selection.

### Dim 8: EXTREME OPTIMIZATION + PERFORMANCE
~50 LOC; <1 second runtime; $0 cost.

### Dim 9: OPTIMAL MINIMAL CONTEST SCORE
G1 predicted: 0.182-0.189 [contest-CPU] = 1.6-5.2% improvement over current 0.19205.

## 3. Observability Surface (per Catalog #305)

1. **Inspectable per layer**: per-archive (CPU score, CUDA score, axis-gap) tuple
2. **Decomposable per signal**: per-axis per-component (seg, pose, rate) decomposition for each archive
3. **Diff-able across runs**: archive_sha → CPU rank vs CUDA rank
4. **Queryable post-hoc**: `select_cpu_optimal_archive('pr106_format0d_family')` → archive_sha with min CPU score
5. **Cite-able**: each selection cites the dual-eval JSON path + Catalog #316 frontier ledger row
6. **Counterfactual-able**: counterfactual = "what would leaderboard rank be if we submitted CUDA-optimal vs CPU-optimal?" — answer derivable from existing data

## 4. Cargo-Cult Audit Per Assumption (per Catalog #303)

| Assumption | HARD-EARNED vs CARGO-CULTED | Unwind path |
|---|---|---|
| PR102 +0.033 CPU-CUDA gap | **HARD-EARNED-VERIFIED** (CLAUDE.md anchor) | N/A |
| PR107 +0.033 consistent | **HARD-EARNED-VERIFIED** (CLAUDE.md anchor) | N/A |
| The gap generalizes to OUR archives | **HARD-EARNED IF empirical** (we have PR101+102+103+106+107 dual-eval data) | Re-eval any archive missing CPU axis on Linux x86_64 |
| Leaderboard ranks by CPU not CUDA | **HARD-EARNED-VERIFIED** (CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" section explicitly: PR102 third prize was awarded on CPU 0.19538) | N/A |
| Pose component is dominant gap source | **HARD-EARNED** (CLAUDE.md PR102 analysis) | N/A |
| Re-ranking captures 10-30% of the gap | **CARGO-CULTED** (untested assumption; could be higher if rank order flips entirely) | EMPIRICAL: run G1 ranker; observe magnitude |
| All archives in a family have similar CPU-CUDA pattern | **CARGO-CULTED** (per-archive variation possible) | Per-archive dual-eval |
| Submitting CPU-optimal archive on a contest where CUDA is the smoke axis is contest-compliant | **HARD-EARNED** per CLAUDE.md (CPU eval IS the official leaderboard ranking) | N/A |
| CUDA score remains acceptable (not requiring re-evaluation) | **HARD-EARNED** (CPU is the ranking axis; CUDA is transparency) | N/A |

**5 HARD-EARNED + 2 HARD-EARNED IF + 2 CARGO-CULTED.** Both cargo-cults resolve via the G1 ranker run itself.

## 5. Dykstra-Feasibility Intersection (per Catalog #296)

### Constraint set:
- **(R) Rate**: unchanged (G1 selects an existing archive; no byte change)
- **(S) Segmentation**: per-axis preserved; CPU-axis selected ARCHIVE has CPU-axis-min SegNet
- **(P) Pose**: per-axis preserved; CPU-axis selected ARCHIVE has CPU-axis-min PoseNet
- **(L) Inflate LOC**: unchanged (uses existing archive's inflate.py)
- **(D) Determinism**: per-axis is by definition device-specific; consistent within each axis

### First-principles Dykstra-feasibility check:
Intersection is non-empty because each existing archive ALREADY satisfies the constraints; G1 only SELECTS among them.

### Citation chain:
- Empirical anchor: PR102 + PR107 dual-eval data from CLAUDE.md
- Boyd convex optimization: argmin selection from finite set is well-defined
- IEEE 754 numerical reproducibility: per-axis determinism inherits from canonical inflate.sh discipline

## 6. Predicted Band Per Catalog #324

### Derivation:

Lower bound (-0.003): conservative estimate; assumes only 10% of PR102's +0.033 gap is captured (0.033 × 0.1 = 0.0033)
Upper bound (-0.010): assumes 30% captured (0.033 × 0.3 = 0.0099)

### Calculation rationale:
- Current best CPU 0.19205 (fec6_fixed_huffman_k16_clean)
- Sister format0d_latent_score_table family currently best-CUDA (0.20533) BUT possibly has CPU score better than 0.19205 — not yet measured systematically
- G1's first action: scan ALL PR101+102+103+106+107 archives for CPU score; identify min-CPU-score-per-family; verify if it differs from current best

### Catalog #324 post-training Tier-C validation:
Predicted band [-0.010, -0.003] validated when:
1. G1 ranker re-computation completes within session
2. Identified CPU-optimal archive's CPU score is OUTSIDE current 0.19205 by at least -0.003
3. Paired Linux x86_64 [contest-CPU] re-eval confirms the score (per CLAUDE.md non-negotiable)

### Reactivation:
- (a) If G1 finds no improvement: re-classify as "all archives already CPU-optimal at this operating point"
- (b) If G1 finds improvement > -0.010: investigate as substrate-class-shift opportunity

## 7. 6-Hook Wire-In Declaration (per Catalog #125)

### Hook 1: Sensitivity-map contribution
**ACTIVE**. G1 emits per-archive per-axis (CPU, CUDA) score-sensitivity row to `tac.sensitivity_map`.

### Hook 2: Pareto constraint
**ACTIVE**. G1 selects archives on the CPU-axis-Pareto-front; updates `tac.pareto_*` solver.

### Hook 3: Bit-allocator hook
**N/A** (G1 doesn't change bit allocation; only ranker selection).

### Hook 4: Cathedral autopilot dispatch hook
**ACTIVE**. G1's CPU-optimal archive selection feeds `tools/cathedral_autopilot_autonomous_loop.py::rank_candidates`.

### Hook 5: Continual-learning posterior update
**ACTIVE**. G1 verdict (which archive is CPU-optimal) registers per Catalog #245 + Catalog #316 frontier ledger.

### Hook 6: Probe-disambiguator
**ACTIVE**. The empirical-vs-prediction disambiguator IS the existing dual-eval data.

## 8. Routing Directive Sketch For Codex Execution

Full routing directive: `.omx/research/codex_routing_directive_rate_attack_vector_2_g1_cpu_axis_optimization_20260518.md`.

### Phase 1 (LOCAL re-ranking; $0):
1. Land `tools/cpu_axis_optimal_archive_selector.py`:
   - Scan `.omx/state/continual_learning_posterior.jsonl` + `.omx/state/modal_call_id_ledger.jsonl` for ALL archives with paired CPU+CUDA anchors
   - Group by family (pr101_*, pr102_*, pr103_*, pr106_*, pr107_*)
   - Compute argmin per family on CPU axis
   - Compare to current frontier per Catalog #316
   - Emit verdict to `reports/latest.md` FRONTIER section
2. Identify CPU-optimal archive across all families
3. If CPU-optimal differs from current 0.19205 by > -0.003 → Phase 2

### Phase 2 (Linux x86_64 paired re-eval; $0):
1. Re-eval CPU-optimal archive on Linux x86_64 per CLAUDE.md non-negotiable
2. Confirm score within predicted band
3. If validated → submit PR

## 9. Cross-References

- Master memo: `.omx/research/rate_attack_43_vectors_meta_paradigm_deep_research_20260518.md`
- META-paradigm: `.omx/research/structural_information_not_shipped_meta_paradigm_unification_20260518.md`
- Routing directive: `.omx/research/codex_routing_directive_rate_attack_vector_2_g1_cpu_axis_optimization_20260518.md`
- Sister canonical helper: `tac.frontier_scan` + `tools/scan_best_anchor_per_axis.py` (Catalog #316)
- CLAUDE.md non-negotiables: "Submission auth eval — BOTH CPU AND CUDA" / "SegNet vs PoseNet importance — operating-point dependent" / "MPS auth eval is NOISE"
- Catalog gates: #127 / #192 / #205 / #316 / #324 / #287

## 10. Closeout

G1 is the **CHEAPEST rate-attack vector** of the 43. ZERO GPU spend. Pure local ranker re-computation on existing dual-eval data. Hotz binding directive: PROCEED IMMEDIATELY.

**Predicted band [-0.010, -0.003] [contest-CPU]. Even capturing the LOWER bound (-0.003) at $0 cost is highest-EV per-dollar of the entire TOP-5.**

**Next action**: Phase 1 LOCAL re-ranking via Codex `019de465` per routing directive. Can land THIS SESSION.
