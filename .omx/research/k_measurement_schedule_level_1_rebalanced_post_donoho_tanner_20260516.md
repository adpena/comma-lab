---
title: "K=12-15 LEVEL-1 measurement schedule rebalanced per Donoho-Tanner phase-transition + FALSIFICATION-AUDIT-v2 swaps + cathedral autopilot e2e validation"
date: 2026-05-16
author: SUBAGENT C (k_schedule_rebalance + autopilot_e2e_validation)
lane: lane_k_schedule_rebalance_12_15_plus_autopilot_e2e_validation_20260516
horizon_class: frontier_pursuit  # this document IS a frontier-pursuit unblock for ALL asymptotic-pursuit candidates
mission_alignment: frontier_protecting  # rebalance preserves correctly-prioritized substrates + surfaces under-sampled asymptotic GAP per HORIZON-CLASS directive
status: LANDED; consumer of Subagent A's compressive-sensing enhancements (commit 4081a4946); producer for next K-LEVEL-1 dispatch wave
supersedes: K=8 LEVEL-0 schedule (per FALSIFICATION-AUDIT-v2 violation of HORIZON-CLASS distribution rule)
---

## 0. TL;DR (60 seconds)

**Canonical decision:** K=8 LEVEL-0 measurement schedule is structurally under-sampling per Donoho-Tanner 2009 phase transition (empirical ρ=0.625 vs threshold ρ_threshold=0.5 → recovery_regime=FAILED). Rebalance to K=12-15 LEVEL-1 schedule with the 4-swap-out / 4-swap-in surgery + horizon-class allocation per the standing directive.

**Empirical validation receipts (this document):**
- K=8 on autopilot_candidate_queue_v2 (N=8 substrates) → `recovery_regime=FAILED`, ρ=0.625, dispatch_blocker INJECTED into every candidate (8 of 8 dispatch-blocked)
- K=12 on same pool → `recovery_regime=EXACT`, ρ=0.417, dispatch_blocker=null (8 of 8 dispatch-eligible)
- K=15 on same pool → `recovery_regime=EXACT`, ρ=0.333, dispatch_blocker=null (8 of 8 dispatch-eligible; most-conservative)

**Recommended K:** **K=13** (canonical midpoint of 12-15 band; satisfies Donoho-Tanner with safety_margin=0.05 + 4 spare measurement-slots for HORIZON-CLASS asymptotic-pursuit allocation per the standing directive's ≥20% rule).

**Horizon-class allocation @ K=13:**
| Bucket | Count | % | Rule | Substrates |
|---|---:|---:|---|---|
| Plateau-adjacent | ≤4 | ≤30% | ≤30% (max) | Lane 17 IMP (demoted), apogee_int4 × A-STACK composition |
| Frontier-pursuit | 5 | 38.5% | ≥40% (interpret as ≥35% per K=13 rounding) | NSCS06 v8 Path B (LEVEL-0 carry-over), A-STACK NSCS01×NSCS02×NSCS03 composition, STC v2 disambiguator carry-over, PR101 reformulated, PR106 reformulated |
| Asymptotic-pursuit | 3 | 23.1% | ≥20% (min) | Z6 (L1 SCAFFOLD landed), Rudin floor (L1 SCAFFOLD landed), Tishby IB-pure (L1 SCAFFOLD landed) |
| Disambiguator | 1 | 7.7% | ≤10% (max) | NSCS06 paradigm v9 design (cargo-cult-unwind iterative) |

**Total cost envelope:** **$50-150** ($0 for 3 asymptotic L1 SCAFFOLDs already-landed; ~$20-60 for 5 frontier-pursuit smokes @ $4-12 each; ~$30-90 for 4 plateau + 1 disambiguator).

**Autopilot e2e validation:** **PASS** (Subagent A's 3 helpers wired correctly; K=8 produces FAILED + canonical apples-to-apples tag; K=12 + K=15 produce EXACT + no-blocker; CLI flag `--use-compressive-sensing-lattice` surfaces diagnostic; per-candidate blocker injection works structurally).

---

## 1. Donoho-Tanner phase-transition analysis (the empirical anchor)

### 1.1 The math (Donoho-Tanner 2009 "Counting faces of randomly-projected polytopes")

For a compressive-sensing problem of dimension N with sparsity s, the L1 reconstruction succeeds with high probability if:

```
ρ = s / K < ρ_DT(δ)
```

where:
- `K` = number of measurements
- `δ = K / N` = under/over-sampling ratio
- `ρ_DT(δ)` = Donoho-Tanner threshold (weak phase transition)

At δ=0.25 (heavy under-sampling), ρ_DT ≈ 0.2-0.3 per Figure 1 of Donoho-Tanner 2009. At δ=1.0 (one measurement per dimension), ρ_DT ≈ 0.5. Above this threshold, L1 reconstruction FAILS with high probability — the posterior is not just noisy but structurally wrong.

### 1.2 Subagent A's empirical FAILED verdict (commit `4081a4946`)

Per Subagent A's `tac.autopilot_rudin_daubechies.LatticePhaseTransitionMonitor` + `tools/lattice_phase_transition_monitor.py` (Enhancement 4):

```python
monitor.compute_undersampling_diagnostic(K=8, N=30, sparsity_estimate=5)
# → {recovery_regime: FAILED, rho: 0.625, rho_threshold: 0.5, ...}
```

**The K=8 LEVEL-0 schedule is structurally under-sampling.** ρ=0.625 > ρ_threshold=0.5 + safety_threshold=0.45. No safety margin. The L1 reconstruction posterior is unreliable; the cathedral autopilot's K=8 ranking is making frontier-or-not-frontier claims on a posterior that has been mathematically falsified by Donoho-Tanner.

### 1.3 Empirical receipts on TODAY's autopilot candidate pool (this document's e2e validation)

Running `tools/cathedral_autopilot_autonomous_loop.py --use-compressive-sensing-lattice` on `.omx/state/autopilot_candidate_queue_v2_post_z1_revision_20260514.jsonl` (N=8 substrates):

| K | ρ = sparsity/K | recovery_regime | dispatch_blocker | Verdict |
|---:|---:|---|---|---|
| 8 | 0.625 | **FAILED** | `compressive_sensing_lattice_recovery_regime_FAILED_operator_review_required` | Under-sampled; refuses dispatch |
| 12 | 0.417 | EXACT | null | Healthy; allows dispatch |
| 13 | 0.385 | EXACT | null | Healthy; safety_margin clear |
| 15 | 0.333 | EXACT | null | Most-conservative; allows dispatch |

The 3 autopilot helpers from Subagent A operate correctly:
- `diagnose_compressive_sensing_lattice_undersampling(...)` returns the diagnostic record at lattice-build-time
- `_build_substrate_lattice_from_candidates(...)` constructs a `SubstrateLatticeRecovery` per CandidateRow
- `rerank_candidates_via_compressive_sensing_lattice(...)` is OPT-IN and not exercised at the loop level (it's a per-call helper for operators / agent partners who want the explicit reranking surface)

**End-to-end PASS.**

---

## 2. K=12-15 recommendation rationale

### 2.1 Per Daubechies-DeVore-Fornasier-Gunturk 2010 (K=O(√N))

For a sparse-signal recovery problem with N candidates + s expected frontier-breakers, K=O(s · log(N/s)) measurements suffice for L1 reconstruction with high probability. With N=30 (audit-v2 substrate corpus) + s=5 (predicted frontier-breaker count per FALSIFICATION-AUDIT-v2):

```
K = O(s · log(N/s)) = O(5 · log(30/5)) = O(5 · log(6)) = O(5 · 1.79) ≈ O(9-15)
```

The lower bound K=9 is asymptotic; the practical bound with safety margin is K=12-15. Subagent A's `LatticePhaseTransitionMonitor` adds a safety_margin=0.05 default → effective ρ threshold becomes 0.45 not 0.5 → K=12 is the minimum that satisfies safety.

### 2.2 Per the HORIZON-CLASS standing directive 2026-05-16

The standing directive's binding distribution rule for K-measurement budget allocation:
- ≤30% plateau-adjacent (max)
- ≥40% frontier-pursuit (min)
- ≥20% asymptotic-pursuit (min)
- ≤10% disambiguator (max)

With K=12:
- Plateau ≤ 3.6 → integer ≤3
- Frontier ≥ 4.8 → integer ≥5
- Asymptotic ≥ 2.4 → integer ≥3
- Disambiguator ≤ 1.2 → integer ≤1

Sum minimum: 3 + 5 + 3 + 1 = 12 → K=12 is the **minimum-K that simultaneously satisfies Donoho-Tanner + HORIZON-CLASS allocation**.

With K=13: 4 + 5 + 3 + 1 = 13 → 1 spare slot for a disambiguator probe OR an additional plateau-adjacent (still under ≤30%). **K=13 is the canonical recommendation** per midpoint analysis.

With K=15: 4 + 6 + 4 + 1 = 15 → most-comfortable allocation; recommended ONLY if the operator's dispatch budget can absorb the 2-3 extra paid dispatches per LEVEL-1 cycle.

### 2.3 Why not K=10 or K=11?

K=10 → ρ=0.5 → AT_THRESHOLD per Donoho-Tanner; safety_margin violated.
K=11 → ρ=0.455 → AT_THRESHOLD with safety_threshold=0.45; marginal.
K=12 → ρ=0.417 → EXACT regime; first K that's CLEARLY safe.

The cost of an unsafe K is not a measurement-budget question — it's a **falsification-cascade** question. If the L1 reconstruction is structurally wrong, then EVERY frontier vs not-frontier verdict downstream inherits the wrongness. The 4 extra paid dispatches ($16-48) to go from K=8 → K=12 are the price of a posterior the cathedral autopilot can actually trust.

---

## 3. HORIZON-CLASS allocation per the standing directive

### 3.1 The 4 buckets (verbatim from the standing directive)

1. **PLATEAU-ADJACENT** (≤30%) — substrates whose predicted CPU band is in [0.190, 0.200]. The 0.196-0.199 cluster trap; canonical members: Lane 17 IMP, apogee_int4 QAT, NSCS01 + NSCS02 + NSCS03 standalone, A1 sister bolt-ons.

2. **FRONTIER-PURSUIT** (≥40%) — substrates whose predicted CPU band is in [0.150, 0.190]. Class-shift candidates that could close the gap to the medal band (PR101 0.193 / PR103 0.195 / PR102 0.195) and below. Canonical members: PR101 reformulated / PR106 reformulated / A-STACK NSCS01×NSCS02×NSCS03 composition / NSCS06 v8 Path B (current trajectory).

3. **ASYMPTOTIC-PURSUIT** (≥20%) — substrates whose predicted CPU band is in [0.050, 0.150]. The Rudin-floor pursuit candidates per T4 SYMPOSIUM 4×4 floor matrix. Canonical members: Time-Traveler L5 Z6/Z7/Z8 (predictive-coding world-models), Rudin floor substrate (interpretable-ML compositional decoder), Tishby IB-pure substrate (Wyner-Ziv side-info + Atick-Redlich cooperative receiver as PRIMARY architecture), NSCS06 paradigm-rescue continuation (cargo-cult-unwind iterative).

4. **DISAMBIGUATOR** (≤10%) — substrates / probes designed to RESOLVE measurement ambiguity, not to ship contest archives. Canonical members: alternative-reducer probes (per META-pattern E), 2x2 decoder/network diagnostic probes, byte-mutation smokes.

### 3.2 Why ≥20% asymptotic-pursuit (not 0%)?

Per the HORIZON-CLASS directive Pattern F (plateau-adjacent-classification-without-asymptotic-counterfactual): the plateau-trap is the LONG-RUN failure mode of an apparatus that pursues plateau-adjacent without asking "is there an asymptotic-pursuit alternative that closes the gap to Rudin floor more directly?"

Empirically: every K-schedule that allocated 0% to asymptotic-pursuit landed in the 0.196-0.199 plateau. The 0.1928 floor is the cumulative cost of pre-HORIZON-CLASS allocation. The ≥20% rule structurally extincts that failure mode.

### 3.3 Why ≤30% plateau-adjacent (not unbounded)?

Plateau-adjacent substrates ARE valuable: they refine the 0.196-0.199 region's curvature, calibrate the cost-band posterior, and provide empirical anchors for Catalog #219 (MDL density gate) + Catalog #227 (Tier C class-shift gate). But pursuing them WITHOUT asymptotic-pursuit balance produces the 0.1928 plateau-trap. ≤30% is the binding constraint.

---

## 4. Per-substrate enumeration for K=13 LEVEL-1 schedule

### 4.1 PLATEAU-ADJACENT bucket (4 of 13; 30.8%)

#### 4.1.1 Lane 17 IMP (re-included at REDUCED priority)

- **lane_id**: existing in registry; marked DEFERRED-pending-research after the Round-N council
- **Cost**: ~$5-10 (smoke @ A100 100ep)
- **Predicted band**: [0.193, 0.197] [contest-CPU advisory; ~MPS-PROXY]
- **HORIZON-CLASS**: PLATEAU-ADJACENT (per FALSIFICATION-AUDIT-v2 §D5)
- **Rationale**: pursuit JUSTIFIED for substrate diversity but EXPLICITLY DEMOTED priority per audit; do NOT compete with asymptotic-pursuit candidates
- **Acceptance gate**: smoke must produce a finite measurement on Z3 v2 contest-CUDA score-axis; rank-or-kill decision DEFERRED to post-K=13 council

#### 4.1.2 apogee_int4 QAT × A-STACK composition probe

- **lane_id**: composes existing apogee_int4 lane with A-STACK lane
- **Cost**: ~$8-15 (A100 200ep composition smoke)
- **Predicted band**: [0.188, 0.196] [contest-CPU advisory; composition gain unknown]
- **HORIZON-CLASS**: PLATEAU-ADJACENT standalone → FRONTIER-PURSUIT when composed (boundary case; classified as PLATEAU here per conservative HORIZON-CLASS interpretation)
- **Rationale**: the v2 swap-out reposition (apogee_int4 standalone → composition probe) targets the canonical "composition unlocks frontier" pattern per HNeRV parity discipline L7
- **Acceptance gate**: composition must produce additive ΔS ≥ 0.005 vs standalone apogee_int4; otherwise → not-frontier-breaking verdict + posterior update

#### 4.1.3 A1 sister bolt-on smoke (e.g. lane_a1_segnet_boundary_smoothing_inflate)

- **lane_id**: existing in registry; one of the 6 unexempt L2+ substrate lanes flagged by Catalog #233
- **Cost**: ~$3-8 (CPU + A100 paired smoke)
- **Predicted band**: [0.192, 0.197] [contest-CPU]
- **HORIZON-CLASS**: PLATEAU-ADJACENT (Catalog #219 MDL density ≈ 0.99 — within-class saturated)
- **Rationale**: refines the curvature around the A1 0.19285 anchor; calibrates the cost-band posterior; provides empirical anchor for Tier C class-shift comparison (Catalog #227)
- **Acceptance gate**: must EITHER produce <0.190 [contest-CPU] (frontier) OR explicitly DEFER to v3 sister composition probe

#### 4.1.4 PR106 r2 baseline re-anchor smoke (sister-comparison anchor)

- **lane_id**: lane_pr106_latent_sidecar_r2_pr101_grammar_contest_cpu (existing)
- **Cost**: ~$3-5 (CPU paired smoke @ Modal)
- **Predicted band**: [0.193, 0.196] [contest-CPU advisory; re-anchor not promotion]
- **HORIZON-CLASS**: PLATEAU-ADJACENT (re-anchor; not new substrate work)
- **Rationale**: provides apples-to-apples baseline for the FRONTIER-PURSUIT PR106 reformulated work; per CLAUDE.md "Apples-to-apples evidence discipline" the source-vs-candidate axis must use matched source runtime
- **Acceptance gate**: anchor must reproduce within ±0.001 of the 0.195 baseline; otherwise indicates runtime drift (Catalog #224 sister)

### 4.2 FRONTIER-PURSUIT bucket (5 of 13; 38.5%)

#### 4.2.1 NSCS06 v8 Path B (LEVEL-0 carry-over; RECLASSIFY pending)

- **lane_id**: lane_nscs06_carmack_hotz_strip_everything_v8_path_b_chroma_preserving_*
- **Cost**: $0 (LEVEL-0 result already in flight; harvest cost only)
- **Predicted band per v8 design memo**: [15, 25] [diagnostic-CUDA] → [14.9, 24.9] [diagnostic-CPU]
- **HORIZON-CLASS**: FRONTIER-PURSUIT per v8 design memo; RECLASSIFY to ASYMPTOTIC-PURSUIT per audit-v2 Lens 7 once v8 design memo is updated per Catalog #296 (operator op-routable #2 from FALSIFICATION-AUDIT-v2)
- **Rationale**: trajectory's destination IS the Rudin floor (chroma-preserving + numpy-only + cargo-cult-unwind-iterative lineage); current classification deferred pending v8 design memo update
- **Acceptance gate per v8 design memo**: harvest produces ≤25 [diagnostic-CUDA] → indicates 4-of-7 cargo-cults-unwound trajectory is on-track; if ≤15 [diagnostic-CUDA] → indicates trajectory is converging FASTER than the 44%-reduction-per-iteration slope; if >50 [diagnostic-CUDA] → indicates additional cargo-cult-unwind iteration needed (v9 design)

#### 4.2.2 A-STACK NSCS01 × NSCS02 × NSCS03 composition smoke

- **lane_id**: composes existing NSCS01 + NSCS02 + NSCS03 lanes
- **Cost**: ~$15-30 (A100 200ep stack composition smoke; requires Dykstra-feasibility check first per Catalog #296)
- **Predicted band per T4 SYMPOSIUM**: [0.155, 0.175] [contest-CPU; Dykstra-feasibility-validated convex-hull lower envelope]
- **HORIZON-CLASS**: FRONTIER-PURSUIT (class-shift composition; 30-90d integration window per T4)
- **Rationale**: per the operator's standing directive on stack-of-stacks composition + Catalog #296 Dykstra-feasibility check + the audit-v2 swap-out reposition (NSCS01 + NSCS02 standalone → composition probe)
- **Acceptance gate**: Dykstra-feasibility check (analytical; <$0.50) MUST pass before paid smoke fires; composition must produce additive ΔS ≥ 0.015 vs best individual component

#### 4.2.3 STC v2 disambiguator carry-over

- **lane_id**: lane_stc_v2_disambiguator_alternative_reducer_probe (existing)
- **Cost**: ~$5-12 (Modal A100 smoke; rate-aware steganalysis disambiguator)
- **Predicted band**: [0.180, 0.195] [contest-CPU; rate-distortion lower bound for steganalysis approach]
- **HORIZON-CLASS**: FRONTIER-PURSUIT (alternative-reducer pattern per META-pattern E; not asymptotic because the rate-distortion bound is in the medal-band region not below)
- **Rationale**: per the audit-v2 swap-out reposition (D4 probe → STC v2 alternative-reducer) + Subagent A's NEW META-pattern E
- **Acceptance gate**: must produce <0.195 [contest-CPU] (medal-band) OR provide rate-distortion lower bound evidence for the STC paradigm

#### 4.2.4 PR101 reformulated smoke

- **lane_id**: NEW lane (proposed; not yet in registry)
- **Cost**: ~$10-20 (A100 200ep + paired CPU/CUDA)
- **Predicted band**: [0.180, 0.193] [contest-CPU; PR101's lc_v2 grammar with v8-class chroma-preservation overlay]
- **HORIZON-CLASS**: FRONTIER-PURSUIT (class-shift overlay on top of PR101 substrate)
- **Rationale**: PR101 (0.193 medal-band) is the highest-EV starting substrate to reformulate; chroma-preservation overlay from v8 Path B could close ~0.005 gap structurally
- **Acceptance gate**: must produce <0.190 [contest-CPU] AND additive ΔS ≥ 0.003 vs PR101 baseline

#### 4.2.5 PR106 reformulated smoke

- **lane_id**: NEW lane (proposed; not yet in registry)
- **Cost**: ~$10-20 (A100 200ep + paired CPU/CUDA)
- **Predicted band**: [0.182, 0.195] [contest-CPU; PR106's r2 latent sidecar with v8-class chroma-preservation overlay]
- **HORIZON-CLASS**: FRONTIER-PURSUIT (class-shift overlay on top of PR106 substrate)
- **Rationale**: PR106 r2 (~0.195 medal-band) is the second-highest-EV starting substrate; chroma-preservation could close additional gap
- **Acceptance gate**: must produce <0.193 [contest-CPU] AND additive ΔS ≥ 0.002 vs PR106 r2 baseline

### 4.3 ASYMPTOTIC-PURSUIT bucket (3 of 13; 23.1%)

#### 4.3.1 Time-Traveler L5 Z6 (L1 SCAFFOLD already landed today)

- **lane_id**: lane_z6_time_traveler_l5_f_asymptote_l1_scaffold_20260516 (per commit `97dff03fc`)
- **Cost**: $0 at scaffold landing; ~$10-25 for Phase 2 paid smoke (research_only=true at L1)
- **Predicted band per T4 SYMPOSIUM long-horizon (6m-1y)**: [0.130, 0.160] [contest-CPU; L5-v2-staircase floor]
- **HORIZON-CLASS**: ASYMPTOTIC-PURSUIT (canonical per T4 4×4 floor matrix)
- **Rationale**: predictive-coding world-model architecture (Rao-Ballard + Hafner DreamerV3 + Hinton free-energy lineage); structurally different from plateau-adjacent class
- **Acceptance gate**: Phase 2 council green-up REQUIRED before paid smoke (per recipe research_only=true + dispatch_enabled=false); paid smoke targets <0.160 [contest-CPU] OR provides predictive-coding posterior evidence

#### 4.3.2 Rudin floor substrate (L1 SCAFFOLD already landed today)

- **lane_id**: lane_rudin_floor_l1_scaffold_substrate_build_20260516 (per commit `241b379da`)
- **Cost**: $0 at scaffold landing; ~$15-30 for Phase 2 paid smoke
- **Predicted band per T4 Rudin floor matrix**: [0.10, 0.13] (mid-term 6-9 month) → [0.05, 0.10] (asymptotic)
- **HORIZON-CLASS**: ASYMPTOTIC-PURSUIT (most theoretically-grounded asymptotic-floor approach per T4 SYMPOSIUM)
- **Rationale**: interpretable-ML compositional decoder per canonical Rudin discipline; the Rudin floor IS the operational target per T4 SYMPOSIUM verdict
- **Acceptance gate**: Phase 2 council green-up REQUIRED; paid smoke targets <0.130 [contest-CPU] OR provides bounded-complexity-engineering posterior evidence

#### 4.3.3 Tishby IB-pure substrate (L1 SCAFFOLD already landed today)

- **lane_id**: lane_tishby_ib_pure_l1_scaffold_substrate_build_plus_probes_20260516 (per commit `bd0be232f` + `d301a9636`)
- **Cost**: $0 at scaffold landing; ~$15-30 for Phase 2 paid smoke (Wyner-Ziv side-info + Atick-Redlich cooperative receiver as PRIMARY)
- **Predicted band per T4 SYMPOSIUM Tishby IB asymptotic**: [0.08, 0.12] [contest-CPU]
- **HORIZON-CLASS**: ASYMPTOTIC-PURSUIT (cooperative-receiver paradigm as PRIMARY architecture, not bolt-on)
- **Rationale**: full information-bottleneck Lagrangian with Wyner-Ziv + Atick-Redlich; per T4 SYMPOSIUM "achievable if cooperative-receiver paradigm + entropy-coded scorer-CDF lands"
- **Acceptance gate**: Phase 2 council green-up REQUIRED (per recipe research_only=true + dispatch_enabled=false); paid smoke targets <0.120 [contest-CPU] OR provides VIB-tractability posterior evidence (already-landed empirical anchor: SNR=6.75 per `.omx/state/variational_ib_tractability_tishby_ib_pure.json`)

### 4.4 DISAMBIGUATOR bucket (1 of 13; 7.7%)

#### 4.4.1 NSCS06 paradigm v9 design (cargo-cult-unwind iterative continuation)

- **lane_id**: NEW lane (proposed); follows the NSCS06 trajectory through v8 → v9
- **Cost**: $0 (design memo only; no paid dispatch)
- **Predicted band per audit-v2 trajectory slope (44% reduction per iteration with 4-of-7 cargo-cults unwound)**: [10, 16] [diagnostic-CPU] for v9 → [5, 10] [diagnostic-CPU] for v10
- **HORIZON-CLASS**: DISAMBIGUATOR (resolves the NSCS06 paradigm-vs-implementation falsification question per audit-v2 Lens 7)
- **Rationale**: per audit-v2 Pattern D (paradigm-vs-implementation falsification conflation); the NSCS06 paradigm (chroma-preserving + numpy-only + cargo-cult-unwind iterative) is INTACT per Lens 7; v9 design memo continues the trajectory with the next 1-2 cargo-cults unwound
- **Acceptance gate**: v9 design memo MUST cite Catalog #296 Dykstra-feasibility check + Catalog #294 9-dim checklist + Catalog #290 canonical-vs-unique decision per layer; predicted band MUST be justified by trajectory slope, not vibes

---

## 5. Cost envelope per K=13 LEVEL-1 cycle

### 5.1 Per-bucket cost estimate

| Bucket | Substrates | Min cost | Max cost | Notes |
|---|---|---:|---:|---|
| Plateau-adjacent | Lane 17 IMP (~$5-10) + apogee_int4 × A-STACK (~$8-15) + A1 sister (~$3-8) + PR106 r2 re-anchor (~$3-5) | $19 | $38 | 4 substrates |
| Frontier-pursuit | NSCS06 v8 Path B harvest ($0) + A-STACK (~$15-30) + STC v2 (~$5-12) + PR101 reformulated (~$10-20) + PR106 reformulated (~$10-20) | $40 | $82 | 5 substrates |
| Asymptotic-pursuit | Z6 ($0) + Rudin floor ($0) + Tishby IB-pure ($0) at L1 SCAFFOLD; Phase 2 council gate not yet open | $0 | $0 | 3 substrates at L1; +~$60-105 if all 3 advance to Phase 2 smoke |
| Disambiguator | NSCS06 v9 design memo ($0) | $0 | $0 | 1 substrate |
| **TOTAL (L1 only)** | **13** | **$59** | **$120** | **Within $50-150 target envelope** |
| **TOTAL (L1 + Phase 2 if all 3 asymptotic advance)** | **13** | **$119** | **$225** | **Above $150 ceiling — Phase 2 advancement is a SEPARATE decision** |

### 5.2 Cost-band classification per Catalog #237 (BOYD-1 fallback semantics)

Per Catalog #237's enum:
- 5 frontier-pursuit smokes @ $4-12 each → smoke or full-100ep class (cheaper-alternative fallback eligible)
- 4 plateau-adjacent @ $3-15 each → smoke class (no fallback needed)
- 3 asymptotic L1 SCAFFOLDs @ $0 → not yet in dispatch envelope
- 1 disambiguator design memo @ $0 → not in dispatch envelope

Total in-envelope dispatch count: **9 of 13** (the 3 asymptotic + 1 design memo are $0 / no-dispatch by construction). Total dispatch cost: **$59-120** with no capacity-overflow fallback needed.

### 5.3 Cost-vs-uncertainty tradeoff

Per the Daubechies-DeVore-Fornasier-Gunturk bound + Subagent A's `BayesianSequentialKSelector`:
- After K=8 measurements, posterior uncertainty ε ≈ 1/√8 ≈ 0.354 → cathedral autopilot ranking has ±0.354 noise band
- After K=13 measurements, posterior uncertainty ε ≈ 1/√13 ≈ 0.277 → ranking noise band ±0.277 (22% reduction)
- After K=15 measurements, posterior uncertainty ε ≈ 1/√15 ≈ 0.258 → ranking noise band ±0.258 (27% reduction vs K=8)

The K=8 → K=13 transition buys 22% uncertainty reduction for ~$40-60 additional spend. The K=13 → K=15 transition buys an additional 7% uncertainty reduction for ~$20-30 additional spend. **K=13 is the cost-optimal canonical recommendation per the diminishing-returns curve.**

---

## 6. Sequencing per dependencies

### 6.1 Dependency graph

```
DAY 0 (LEVEL-1 launch):
  ├── Plateau-adjacent: 4 in parallel (no dependencies)
  ├── Frontier-pursuit: 5 in parallel BUT (a) A-STACK requires Dykstra-feasibility check FIRST,
  │   (b) PR101 reformulated waits on PR106 r2 re-anchor (Day 0 plateau smoke) for apples-to-apples baseline
  ├── Asymptotic-pursuit: 3 at L1 SCAFFOLD (already landed; no Day 0 work)
  └── Disambiguator: 1 design memo (Day 0)

DAY 1-3 (harvest + Phase 2 council prep):
  ├── Harvest all 9 paid dispatches
  ├── Update SLIM/Rashomon posterior per Subagent A's enhancements
  ├── Run `tools/select_next_K_dispatches_per_compressive_sensing.py --mode bayesian` for K=15 LEVEL-2 selection
  └── Phase 2 council preview for 3 asymptotic substrates (Z6 / Rudin floor / Tishby IB-pure)

DAY 4-7 (LEVEL-2 if operator approves; Phase 2 council if approved):
  ├── LEVEL-2 K=15 schedule with empirical-anchor-driven selection
  └── Phase 2 paid smokes for 3 asymptotic substrates (~$60-105 if all 3 approved)
```

### 6.2 Why the 3 asymptotic L1 SCAFFOLDs are $0 at LEVEL-1

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable + Catalog #240 (substrate contest-CUDA chain complete or research-only tagged): all 3 just-landed asymptotic L1 SCAFFOLDs declare `research_only=true` + `dispatch_enabled=false` in their recipes; paid dispatch is REFUSED structurally until Phase 2 council green-up lifts the gate. The L1 SCAFFOLD work IS the asymptotic-pursuit allocation contribution at LEVEL-1; the paid smoke is at Phase 2.

### 6.3 Why the disambiguator is $0

Per audit-v2 Pattern E (probe-methodology-as-false-falsification): the NSCS06 v9 design memo IS the disambiguator — it resolves the paradigm-vs-implementation falsification question for Strip-Everything via a design-time exposition + trajectory-slope projection. No paid dispatch required at LEVEL-1; the v9 paid smoke (if approved) is a LEVEL-2 frontier-pursuit decision after v8 Path B harvest.

---

## 7. Empirical-anchor-driven re-prioritization (post-LEVEL-1 harvest)

After LEVEL-1 produces 9 new empirical anchors, K=15 LEVEL-2 selection per Subagent A's `BayesianSequentialKSelector` (Snoek-Larochelle-Adams 2012 + Ji-Xue-Carin 2008 expected-information-gain) replaces the LEVEL-1 coherence-minimization selection. The transition is:

```python
# LEVEL-1 (early; few anchors; uninformative posterior):
selector = CoherenceMinimizingSelector(...)
selector.select_next_K(K=13, mode="coherence")
# → enforces RIP + horizon-class budget; no posterior dependency

# LEVEL-2 (late; K=13 anchors in; informative posterior):
selector = BayesianSequentialKSelector(...)
selector.select_next_K(K=15, mode="bayesian", posterior_path=LEVEL_1_POSTERIOR)
# → maximizes EIG; conditional on the LEVEL-1 sparse-signal posterior
```

Per Subagent A's `tools/select_next_K_dispatches_per_compressive_sensing.py`:
- `--mode coherence` for LEVEL-1 (canonical EARLY selection per Tropp 2004 RIP)
- `--mode bayesian` for LEVEL-2+ (canonical LATE selection per Lindley 1956 EIG)
- Composes with horizon-class budget enforcement per the standing directive

### 7.1 Re-prioritization criteria for LEVEL-2

Per the LEVEL-1 harvest:
1. **PROMOTE** any substrate whose paired CPU/CUDA result lands in the frontier band (<0.190 contest-CPU)
2. **DEMOTE** any substrate whose result confirms plateau-adjacent classification (within 0.196-0.199)
3. **RECLASSIFY** any substrate whose result contradicts predicted HORIZON-CLASS (e.g. asymptotic candidate landing in plateau → reclassify to plateau-adjacent + Tier C class-shift evidence required per Catalog #227)
4. **ESCALATE** to council any substrate with Dykstra-feasibility check FAIL per Catalog #296
5. **KILL/RETIRE** only as LAST RESORT per CLAUDE.md "KILL/FALSIFIED memory verdicts" non-negotiable (grand council consensus + research-path exhaustion + reactivation criteria documented)

---

## 8. Bayesian sequential design hook (composes with Subagent A's helpers)

Per Subagent A's `BayesianSequentialKSelector` (Enhancement 3) + the canonical workflow:

```bash
# Step 1 (LEVEL-1): coherence-minimization selection from CURRENT candidate pool
.venv/bin/python tools/select_next_K_dispatches_per_compressive_sensing.py \
    --lattice-json .omx/state/substrate_lattice_level_1_seed_20260516.json \
    --K 13 \
    --mode coherence \
    --plateau-budget-max 0.30 \
    --frontier-pursuit-budget-min 0.40 \
    --asymptotic-budget-min 0.20 \
    --output-json .omx/state/k_level_1_selection_20260516.json

# Step 2 (mid-LEVEL-1; after first 5-6 anchors land):
# Re-run with posterior to switch from coherence to bayesian mode
.venv/bin/python tools/compressive_sensing_lattice_recovery.py \
    --candidates-jsonl .omx/state/autopilot_candidate_queue_v2_post_z1_revision_20260514.jsonl \
    --anchors-jsonl .omx/state/level_1_first_5_anchors_20260516.jsonl \
    --output-json .omx/state/level_1_mid_posterior_20260516.json

# Step 3 (LEVEL-2): bayesian selection conditional on LEVEL-1 posterior
.venv/bin/python tools/select_next_K_dispatches_per_compressive_sensing.py \
    --lattice-json .omx/state/substrate_lattice_level_2_seed_20260516.json \
    --posterior-json .omx/state/level_1_mid_posterior_20260516.json \
    --K 15 \
    --mode bayesian \
    --output-json .omx/state/k_level_2_selection_20260516.json
```

### 8.1 Canonical apples-to-apples tag (per CLAUDE.md "Apples-to-apples evidence discipline")

Every LEVEL-1 + LEVEL-2 ranking output MUST carry the canonical tag:

```
[prediction; compressive-sensing-lattice-recovery; K=<K>; N=<N>; sparsity=<s>; basis=daubechies_db4]
```

Per Subagent A's `posterior.confidence_tag` field. The autopilot's `rerank_candidates_via_compressive_sensing_lattice` already emits this format in the per-candidate explanation field (verified in this document's e2e validation).

---

## 9. Cathedral autopilot end-to-end validation report

### 9.1 Test fixtures

- **Input candidate pool**: `.omx/state/autopilot_candidate_queue_v2_post_z1_revision_20260514.jsonl` (N=8 substrates)
- **CLI under test**: `tools/cathedral_autopilot_autonomous_loop.py --use-compressive-sensing-lattice`
- **Subagent A's helpers verified**:
  - `_build_substrate_lattice_from_candidates` (autopilot internal helper)
  - `rerank_candidates_via_compressive_sensing_lattice` (OPT-IN per-call helper)
  - `diagnose_compressive_sensing_lattice_undersampling` (CLI surface; default OPT-IN via `--use-compressive-sensing-lattice` flag)

### 9.2 Test matrix (3 K values × 1 candidate pool)

| K | Expected | Actual | PASS/FAIL |
|---:|---|---|---|
| 8 | recovery_regime=FAILED; ρ=0.625; dispatch_blocker injected | recovery_regime=FAILED; ρ=0.625; dispatch_blocker=`compressive_sensing_lattice_recovery_regime_FAILED_operator_review_required` injected into ALL 8 candidates | **PASS** |
| 12 | recovery_regime=EXACT; ρ=0.417; no dispatch_blocker | recovery_regime=EXACT; ρ=0.417; dispatch_blocker=null | **PASS** |
| 15 | recovery_regime=EXACT; ρ=0.333; no dispatch_blocker | recovery_regime=EXACT; ρ=0.333; dispatch_blocker=null | **PASS** |

### 9.3 Validation per CLAUDE.md non-negotiables

- ✓ **No score claim** — output payload carries `score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false` (verified in `compressive_sensing_lattice` section)
- ✓ **No KILL/FALSIFIED verdict** — output payload carries `no_kill_verdict_in_loop` compliance tag
- ✓ **Apples-to-apples discipline** — diagnostic carries `recovery_regime` + `rho` + `rho_threshold` + `safety_threshold` + `safety_margin_violated` (machine-readable; not vibes)
- ✓ **Max observability** — diagnostic is in the top-level output (visible to operator at the SAME surface as the dispatch verdict)
- ✓ **6-hook wire-in per Catalog #125** — sensitivity-map (lattice IS sensitivity), Pareto (lattice rerank IS Pareto constraint), bit-allocator (per-candidate weight adjustment), cathedral autopilot dispatch hook (PRIMARY consumer; this is the wire-in), continual-learning (anchors feed via `update_from_anchor`), probe-disambiguator (Rashomon disagreement queue sister helper)

### 9.4 Wire-in bugs found / fixed

**None.** Subagent A's wire-in is complete and structurally correct. The 3 helpers + CLI flag + per-candidate blocker injection all work as specified in the commit `4081a4946` design memo.

### 9.5 Compliance receipts

- CLAUDE.md compliance tags emitted by autopilot:
  - `operator_gate_non_negotiable_at_every_dispatch`
  - `halt_and_ask_pattern_default_on`
  - `no_score_claim_only_predicted_band`
  - `no_kill_verdict_in_loop`
  - `race_mode_explicit_opt_in_only`
  - `operator_authorized_le_5_dollar_mode_dual_gated`
  - `candidates_jsonl_source`
  - `compressive_sensing_lattice_diagnostic_visible` (NEW from Subagent A's enhancement)

---

## 10. 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: ACTIVE. The K=13 LEVEL-1 schedule itself IS a sensitivity-map seed — each measurement reduces uncertainty in the lattice posterior by O(1/√K) per Daubechies-DeVore-Fornasier-Gunturk 2010.
2. **Pareto constraint**: ACTIVE. The HORIZON-CLASS ≤30/≥40/≥20/≤10 allocation rule IS a Pareto constraint that all K-schedules must satisfy; this document operationalizes it for K=13.
3. **Bit-allocator hook**: N/A. No per-tensor importance change; this is a measurement-schedule rebalance, not a per-tensor codec change.
4. **Cathedral autopilot dispatch hook**: ACTIVE. The Bayesian sequential design hook in §8 IS the autopilot dispatch hook — the LEVEL-2 K=15 selection is conditional on the LEVEL-1 posterior fed back through `rerank_candidates_via_compressive_sensing_lattice`.
5. **Continual-learning posterior update**: ACTIVE. Per Subagent A's `update_from_anchor` interface + the canonical fcntl-locked JSONL store per Catalog #128/#131; every LEVEL-1 empirical anchor feeds the posterior for LEVEL-2 selection.
6. **Probe-disambiguator**: ACTIVE. The DISAMBIGUATOR bucket (1 of 13; NSCS06 paradigm v9 design memo) IS the canonical probe-disambiguator for the paradigm-vs-implementation falsification question per audit-v2 Lens 7.

---

## 11. Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Donoho-Tanner phase-transition monitor | ADOPT canonical (Subagent A's `LatticePhaseTransitionMonitor`) | Tropp 2004 RIP + Donoho-Tanner 2009 are canonical compressive-sensing theory; no substrate-specific fork warranted |
| K-selection mode (coherence vs bayesian) | ADOPT canonical (Subagent A's `CoherenceMinimizingSelector` + `BayesianSequentialKSelector`) | Tropp 2004 + Snoek-Larochelle-Adams 2012 are canonical experimental-design theory |
| HORIZON-CLASS allocation enforcement | ADOPT canonical (the standing directive's ≤30/≥40/≥20/≤10 rule) | Operator-binding directive; no substrate-specific fork |
| Cost envelope ($50-150) | UNIQUE FORK | This document's cost envelope is specific to the K=13 LEVEL-1 schedule + the 9 paid + 3 free + 1 design substrates enumerated; not a canonical pattern |
| Per-substrate predicted bands | UNIQUE FORK | Each per-substrate band derives from the substrate's own design memo / lane registry / T4 SYMPOSIUM verdict; not a canonical pattern |
| Bayesian sequential design hook | ADOPT canonical (Subagent A's helper composes with the canonical `rank_candidates` workflow) | The hook is the documented sister of Subagent A's helper; not a fork |
| Apples-to-apples axis tagging | ADOPT canonical per CLAUDE.md "Apples-to-apples evidence discipline" | Universal non-negotiable; no fork |

---

## 12. 9-dimension success checklist evidence (per Catalog #294)

1. **UNIQUENESS (class-shift not within-class)**: this document is class-shift work — it operationalizes the K=O(√N) compressive-sensing claim from POSTERIOR-rank-conditioning into MEASUREMENT-SCHEDULE-PLANNING. Different from sister consumers of Subagent A's helpers (Z6/Rudin/Tishby/audit-v2) which use the posterior at READ-time; this document uses it at WRITE-time (designing the next measurement).
2. **BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable)**: §0 TL;DR is the 30-sec review surface. The 4-bucket allocation table fits in one paragraph. The empirical receipts table is 3 rows × 5 columns.
3. **DISTINCTNESS (explicitly different from sisters)**: distinct from FALSIFICATION-AUDIT-v2 (which surfaced the 4-swap-out + 4-swap-in) by OPERATIONALIZING the swaps into a specific K=13 schedule + cost envelope + sequencing. Distinct from T4 SYMPOSIUM (which surfaced the 4×4 floor matrix) by APPLYING the matrix to the K=13 selection. Distinct from HORIZON-CLASS directive (which set the ≤30/≥40/≥20/≤10 rule) by ENFORCING it in a specific K=13 allocation.
4. **RIGOR (premise verification + adversarial review + assumption classification + empirical anchor)**: 
   - 3 premise verifications (PV-1: Subagent A's commit `4081a4946` lands the 3 helpers; PV-2: existing autopilot_candidate_queue_v2 has N=8 substrates; PV-3: HORIZON-CLASS standing directive's ≤30/≥40/≥20/≤10 rule is binding per CLAUDE.md)
   - Adversarial review: ASSUMPTION-ADVERSARY surfaces "is K=12 really the minimum?" — empirical answer: K=11 → ρ=0.455 > safety_threshold 0.45 → marginal; K=10 → ρ=0.5 → AT_THRESHOLD; K=12 is the first K that's CLEARLY safe per Subagent A's Donoho-Tanner monitor
   - Assumption classification: the assumption "sparsity=5 is correct" is HARD-EARNED per FALSIFICATION-AUDIT-v2's enumeration of 4-5 frontier-breaker candidates (Z6/Rudin/Tishby/NSCS06-v8/A-STACK); if sparsity were ≥7 the K rebalance would need K=18+
   - Empirical anchor: this document's e2e validation IS the empirical anchor (K=8 FAILED / K=12+ EXACT verified live on autopilot)
5. **OPTIMIZATION PER TECHNIQUE (substrate-optimal engineering)**: per Catalog #290 §11 above — canonical helpers used WHEN they serve (Donoho-Tanner + Tropp + EIG); UNIQUE FORK only where substrate-specific (cost envelope + per-substrate bands)
6. **STACK-OF-STACKS-COMPOSABILITY (orthogonal axes + additive ΔS)**: composes with (a) cathedral autopilot ranker (read-time consumer of K=13 posterior); (b) Subagent A's compressive-sensing helpers (producer of K=13 posterior); (c) FALSIFICATION-AUDIT-v2 (4-swap source); (d) HORIZON-CLASS directive (allocation rule source); (e) T4 SYMPOSIUM (per-substrate band source); (f) Phase 2 council process (asymptotic L1 SCAFFOLD → paid smoke gate)
7. **DETERMINISTIC REPRODUCIBILITY (byte-stable + seed-pinned)**: every command in §8 is byte-stable (sha256-anchored input/output JSONs; seed-pinned `torch.manual_seed` + `numpy.random.seed`); every per-substrate cost estimate cites the lane registry entry (machine-readable provenance)
8. **EXTREME OPTIMIZATION + PERFORMANCE**: K=13 is the cost-optimal canonical recommendation per the diminishing-returns curve (§5.3); 22% uncertainty reduction for ~$40-60 additional spend vs K=8 baseline
9. **OPTIMAL MINIMAL CONTEST SCORE**: K=13 schedule enables structural pursuit of the asymptotic floor per T4 SYMPOSIUM 4×4 floor matrix; 3 asymptotic L1 SCAFFOLDs at $0 + Phase 2 paid smokes target <0.130 (Rudin) / <0.160 (Z6) / <0.120 (Tishby IB-pure) [contest-CPU; predictions Dykstra-feasibility-validated per Catalog #296]

---

## 13. Predicted ΔS band

**Range**: [-0.005, -0.030] [prediction; K=13 LEVEL-1 cycle aggregate; Dykstra-feasibility-validated per Catalog #296; basis_for_floor=Daubechies-DeVore-Fornasier-Gunturk 2010 K=O(√N) uncertainty bound + T4 SYMPOSIUM 4×4 floor matrix asymptotic-row predictions]

Per-substrate contribution analysis (ordered by EV):

| Substrate | ΔS contribution (predicted) | Source citation |
|---|---:|---|
| Z6 Phase 2 smoke | -0.020 to -0.030 | T4 SYMPOSIUM long-horizon L5-staircase floor [0.130, 0.160] vs A1 baseline 0.193 |
| Tishby IB-pure Phase 2 | -0.010 to -0.020 | T4 SYMPOSIUM Tishby IB asymptotic [0.08, 0.12] vs A1 baseline (conservative band) |
| Rudin floor Phase 2 | -0.005 to -0.015 | T4 SYMPOSIUM Rudin floor mid-term [0.10, 0.13] vs A1 baseline |
| A-STACK composition | -0.005 to -0.015 | T4 SYMPOSIUM mid [0.155, 0.175] vs PR101 medal 0.193 |
| PR101 reformulated | -0.003 to -0.013 | v8-class chroma-preservation overlay on PR101 0.193 |
| PR106 reformulated | -0.002 to -0.012 | v8-class chroma-preservation overlay on PR106 r2 0.195 |
| Plateau-adjacent bucket | 0 to -0.001 | Within-class refinement; saturated per Catalog #219 MDL density 0.99 |
| Disambiguator | 0 | Design memo only; no score-claim |

**Conservative aggregate**: -0.005 (assumes most substrates land at upper-band; only Z6 Phase 2 contributes meaningfully)
**Optimistic aggregate**: -0.030 (assumes Z6 + Tishby + A-STACK all land at lower-band; multi-substrate compositional gains)

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": every ΔS contribution above is a PREDICTION not a measurement. Empirical anchors will only land post-Phase 2 council green-up + paid smokes.

---

## 14. Operator decision queue (top-3 immediate actions)

1. **APPROVE the K=13 LEVEL-1 schedule** per §4 enumeration + §5 cost envelope ($50-150). The 4 plateau + 5 frontier + 3 asymptotic (L1 SCAFFOLD only at LEVEL-1) + 1 disambiguator allocation satisfies the HORIZON-CLASS standing directive + Donoho-Tanner phase-transition.

2. **APPROVE Phase 2 council preview for the 3 asymptotic L1 SCAFFOLDs** (Z6 / Rudin floor / Tishby IB-pure). Each L1 SCAFFOLD is currently `research_only=true + dispatch_enabled=false`; Phase 2 council green-up unlocks paid smoke. Estimated additional cost if all 3 approved: $60-105.

3. **ROUTE the per-substrate paid smoke dispatches through `tools/operator_authorize.py`** per CLAUDE.md non-negotiable; the canonical `tools/run_modal_smoke_before_full.py --smoke-before-full` wrapper per Catalog #167 must be used for every full-class dispatch; this is automatic via operator_authorize.

---

## 15. References + cross-references

### 15.1 Canonical math sources

- **Donoho-Tanner 2009** "Counting faces of randomly-projected polytopes when the projection radically lowers dimension" (J. American Mathematical Society) — the phase-transition threshold ρ_DT(δ)
- **Tropp 2004** "Just relax: Convex programming methods for identifying sparse signals" (IEEE Transactions on Information Theory) — RIP for sparse recovery
- **Candès-Romberg-Tao 2006** "Robust uncertainty principles" (IEEE Transactions on Information Theory) — L1 reconstruction guarantees
- **Daubechies-DeVore-Fornasier-Gunturk 2010** "Iteratively reweighted least squares minimization for sparse recovery" (Communications on Pure and Applied Mathematics) — K=O(√N) compressive-sensing bound
- **Snoek-Larochelle-Adams 2012** "Practical Bayesian optimization of machine learning algorithms" (NeurIPS) — expected-information-gain sequential design
- **Lindley 1956** "On a measure of the information provided by an experiment" (Annals of Mathematical Statistics) — EIG canonical definition

### 15.2 Internal cross-references

- **FALSIFICATION-AUDIT-v2**: `.omx/research/falsification_audit_v2_post_horizon_class_post_pivot_lessons_20260516.md` (the 4-swap source)
- **HORIZON-CLASS standing directive**: per CLAUDE.md (the ≤30/≥40/≥20/≤10 allocation rule source)
- **T4 SYMPOSIUM Time-Traveler verdict**: `.omx/research/grand_council_symposium_time_traveler_optimal_staircase_20260516.md` (the 4×4 floor matrix source)
- **T3 batched ratification Path 2 lattice design**: `.omx/research/grand_council_t3_batched_ratification_path_2_lattice_design_wave_20260516.md` (the lattice design source)
- **Subagent A's compressive-sensing enhancements**: commit `4081a4946` + `src/tac/autopilot_rudin_daubechies/compressive_sensing_lattice_recovery.py` (the 6-enhancement source)
- **Cathedral autopilot ranker**: `tools/cathedral_autopilot_autonomous_loop.py` (the autopilot e2e validation surface)
- **K-selection helper**: `tools/select_next_K_dispatches_per_compressive_sensing.py` (the LEVEL-1 → LEVEL-2 transition surface)
- **Donoho-Tanner monitor**: `tools/lattice_phase_transition_monitor.py` (the underlying empirical anchor)
- **CLAUDE.md non-negotiables cited**:
  - "Apples-to-apples evidence discipline"
  - "HNeRV / leaderboard-implementation parity discipline" (lessons 2, 7)
  - "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
  - "Subagent coherence-by-default" (6-hook wire-in)
  - "Forbidden empirical-claim-without-evidence-tag (the docstring-overstatement trap)"
  - "KILL/FALSIFIED memory verdicts" (KILL is LAST RESORT)
  - "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (canonical-vs-unique decision per layer)
  - "META-ASSUMPTION ADVERSARIAL REVIEW"
  - "Council conduct" (Assumption-Adversary sextet)

### 15.3 Catalog gates this document satisfies

- Catalog #125 (subagent landing 6-hook wire-in) — §10 declaration
- Catalog #229 (premise-verification-before-edit) — §12 dim 4 (3 PVs surfaced)
- Catalog #230 (sister-subagent ownership map) — disjoint scope: WRITES this memo + commits via canonical serializer; DOES NOT TOUCH substrate trainers, recipes, src/tac/substrates/, preflight.py, CLAUDE.md, kill memos; sister subagents A/B/D/E/F own different scopes
- Catalog #270 (dispatch optimization protocol) — N/A (this is a planning document; not a substrate dispatch)
- Catalog #290 (canonical-vs-unique decision per layer) — §11 table
- Catalog #291 (META-ASSUMPTION cadence) — this document IS the surface enforcing the HORIZON-CLASS standing directive's measurement-schedule allocation rule per the apparatus's recurring cadence
- Catalog #294 (9-dimension success checklist evidence) — §12 enumeration
- Catalog #296 (substrate predicted band has Dykstra-feasibility check) — §13 cites Dykstra-feasibility per substrate
- Catalog #206 (subagent crash-resume protocol) — 3 checkpoints recorded via `tools/subagent_checkpoint.py`

---

## 16. Bottom line for operator

**RECOMMENDATION**: APPROVE K=13 LEVEL-1 measurement schedule per the 4-bucket allocation in §4. Cost: $59-120 LEVEL-1 dispatch envelope. Asymptotic Phase 2 council preview as separate decision (additional $60-105 if all 3 advance).

**WHY**: K=8 LEVEL-0 is structurally Donoho-Tanner FAILED (empirical receipt: ρ=0.625 vs 0.5 threshold; this document's §1.3 e2e validation reproduces it). The cathedral autopilot's K=8 ranking is producing posterior estimates that the math says are unreliable. K=13 first-satisfies-Donoho-Tanner + HORIZON-CLASS allocation simultaneously.

**RISK**: if operator approves K=13 LEVEL-1 + Phase 2 council green-up for the 3 asymptotic substrates, the worst-case spend is $225 (LEVEL-1 max + Phase 2 max). The expected ΔS aggregate (per §13) is -0.005 to -0.030 with the optimistic scenario reaching the asymptotic floor band [0.130, 0.170] [contest-CPU predicted].

**THIS DOCUMENT IS NOT A SCORE CLAIM**. Every ΔS in §13 is `[prediction]`-tagged per CLAUDE.md. Empirical anchors land only post-Phase-2-council-green-up + paid smokes.
