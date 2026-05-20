# Strategy Dependency + Wire-in + Hook-Coverage Graph — 2026-05-20T12:00:00Z

> **Deliverable C of T3 grand council symposium 2026-05-20**
> Cite-chain: `council_t3_grand_strategy_review_20260520T120000Z` + `strategy_staircase_synthesis_20260520T120000Z`

## Graph 1 — Substrate-class topology with frontier anchors

```mermaid
graph LR
  subgraph "FRONTIER PLATEAU [0.192-0.199 CPU]"
    PR101["PR101 fec6 frame-exploit selector<br/>k=16 fixed-Huffman<br/>0.192051 [contest-CPU]<br/>archive: 6bae0201<br/>FRONTIER LOCAL"]
    PR101GOLD["PR101 GOLD<br/>0.193 [contest-CPU]<br/>upstream"]
    PR102["PR102 (bronze)<br/>0.19538 [contest-CPU]<br/>0.22839 [contest-CUDA]"]
    PR103["PR103 (silver)<br/>~0.195 [contest-CPU]"]
    PR106["PR106 format0d latent score-table<br/>0.205330 [contest-CUDA T4]<br/>archive: 9cb989cef519<br/>FRONTIER CUDA"]
  end

  subgraph "PARADIGM-BRIDGE CANDIDATES [predicted [0.18-0.40]]"
    DreamerV3["DreamerV3 RSSM categorical posterior B2<br/>C6 paradigm-bridge<br/>$5-15 smoke after OPTIMAL FORM<br/>STEP 4 rank 1"]
    NSCS06v8["NSCS06 v8 hybrid_class_shift_path_C<br/>neural residual decoder<br/>predicted medal-class [0,15]<br/>STEP 4 rank 2"]
    Z7Mamba2["Z7-Mamba-2 substrate<br/>design memo landed<br/>STEP 4 rank 3"]
    Z6v2["Z6-v2 Wave 2<br/>FiLM ego-motion conditioning<br/>driver fix landed (commits 02d7fc3f + 611495f26)"]
    V1FaissV8["V1 Faiss V8 learned-compression<br/>scaffold landed codex 20260520<br/>research_only at current stage"]
  end

  subgraph "INFORMATION-THEORETIC FLOOR"
    R_D_bound["R(D) bound<br/>ε ≈ 6.7e-4 SegNet arch ceiling<br/>per dual_cpu_cuda_auth_eval_mandatory_20260508"]
  end

  PR101 -->|"-0.000794 advantage"| PR101GOLD
  PR101 -->|"-0.00333 advantage"| PR102
  PR101 -->|"-0.0028 advantage"| PR103
  DreamerV3 -.predicted band.-> R_D_bound
  NSCS06v8 -.predicted band.-> R_D_bound
  Z7Mamba2 -.predicted band.-> R_D_bound
  Z6v2 -.PROCEED_WITH_REVISIONS.-> PR101
  V1FaissV8 -.research_only.-> PR101
```

## Graph 2 — 6-hook wire-in coverage per Catalog #125

```mermaid
graph TD
  subgraph "Hook 1: Sensitivity-map contribution"
    H1_consumers["8 master-gradient exploit consumers<br/>(Catalog #354)<br/>per-pair difficulty atlas<br/>top-K byte sensitivity<br/>per-segnet-class chroma"]
    H1_helper["tac.sensitivity_map.*<br/>canonical surface"]
    H1_consumers --> H1_helper
  end

  subgraph "Hook 2: Pareto constraint"
    H2_dykstra["Dykstra feasibility intersection<br/>(Catalog #296)<br/>R+seg+pose+archive-size polytope"]
    H2_lagrangian["tac.pareto_* + findings_lagrangian"]
    H2_dykstra --> H2_lagrangian
  end

  subgraph "Hook 3: Bit-allocator"
    H3_allocator["per-byte sensitivity bit-allocator<br/>Catalog #354 exploit #3 + #4"]
    H3_helper["tac.bit_allocator (proposed)"]
    H3_allocator -.proposed.-> H3_helper
  end

  subgraph "Hook 4: Cathedral autopilot dispatch"
    H4_consumers["24+ cathedral_consumers/* packages<br/>auto-discovered via Catalog #335"]
    H4_main["cathedral_autopilot_autonomous_loop.py::main"]
    H4_helper["invoke_cathedral_consumers_on_candidates"]
    H4_consumers --> H4_helper
    H4_helper --> H4_main
  end

  subgraph "Hook 5: Continual-learning posterior"
    H5_posterior["tac.continual_learning.posterior_update_locked<br/>+ council_continual_learning.append_council_anchor<br/>+ tac.probe_outcomes_ledger.register_probe_outcome<br/>+ tac.canonical_equations registry"]
  end

  subgraph "Hook 6: Probe-disambiguator"
    H6_probes["tools/probe_*_disambiguator.py<br/>+ canonical_equations consumers"]
    H6_resolver["canonical disambiguator surface"]
    H6_probes --> H6_resolver
  end

  H1_helper -.binds.-> H4_consumers
  H2_lagrangian -.binds.-> H4_consumers
  H3_helper -.binds.-> H4_consumers
  H5_posterior -.binds.-> H4_consumers
  H6_resolver -.binds.-> H4_consumers
```

## Graph 3 — Canonical-helper canonical-contract topology (paradigm-shift via Catalog #335)

```mermaid
graph LR
  subgraph "Canonical contracts (auto-discovered)"
    SymContract["CathedralConsumerContract<br/>(Catalog #335)<br/>5 fields + 2 callables"]
    SymposiumContract["symposium_impls 5-token contract<br/>(Catalog #265)"]
    EquationsContract["CanonicalEquation contract<br/>(Catalog #344)"]
  end

  subgraph "Auto-discovery loop"
    DiscoverConsumers["discover_and_register_consumers<br/>(Catalog #336)"]
    InvokeConsumers["invoke_cathedral_consumers_on_candidates<br/>(Catalog #336)"]
    RerankGradient["rerank_candidates_via_master_gradient<br/>(Catalog #337)"]
  end

  subgraph "Per-element consumers (24+)"
    UncertaintyRanker["risk-adjusted ranker (MG-1)"]
    BayesianPosterior["Bayesian conjugate-prior posterior (MG-2)"]
    MultiGranularity["multi-granularity master-gradient (MG-3)"]
    DifficultyAtlas["per-pair difficulty atlas (MG-4)"]
    StreamingPrediction["streaming master-gradient + Kalman + GP (MG-5)"]
    MGExploits["8 master-gradient exploit consumers (MG-7)"]
    EquationLookup["canonical_equation_lookup (Catalog #344)"]
    MPSPrescreen["MPS-VIABLE prescreen routing (Catalog #341)"]
  end

  SymContract --> UncertaintyRanker
  SymContract --> BayesianPosterior
  SymContract --> MultiGranularity
  SymContract --> DifficultyAtlas
  SymContract --> StreamingPrediction
  SymContract --> MGExploits
  SymContract --> EquationLookup
  SymContract --> MPSPrescreen

  UncertaintyRanker --> DiscoverConsumers
  BayesianPosterior --> DiscoverConsumers
  MultiGranularity --> DiscoverConsumers
  DifficultyAtlas --> DiscoverConsumers
  StreamingPrediction --> DiscoverConsumers
  MGExploits --> DiscoverConsumers
  EquationLookup --> DiscoverConsumers
  MPSPrescreen --> DiscoverConsumers

  DiscoverConsumers --> InvokeConsumers
  InvokeConsumers --> RerankGradient
```

## Graph 4 — Council deliberation cite-chain topology (last 30d sampled)

```mermaid
graph TD
  THIS["council_t3_grand_strategy_review<br/>20260520T120000Z (THIS)"]

  subgraph "PR #110 lifecycle cite-chain"
    PR110edit["council_t3_pr_110_editorial_positioning_symposium_20260520T050557Z"]
    PR110yousfi["council_t3_pr_110_hnerv_fec6_yousfi_collaborator_impression_plus_hair_splitting_verification_20260520"]
    PR110mg16["council_t3_mg16_voice_tone_style_review_symposium_20260520"]
    PR110final["feedback_t3_council_pr_body_final_recursive_review_landed_20260519"]
    PR110upstream["feedback_t3_grand_council_upstream_contest_compliance_symposium_landed_20260519"]
  end

  subgraph "Paradigm-bridge cite-chain"
    DreamerV3sym["council_t3_dreamerv3_rssm_paradigm_bridge_per_substrate_symposium_20260519"]
    NSCS06v8sym["council_t3_cargo_cult_resurrection_nscs06_v8_variant_c_20260519"]
    C6IBPSv2sym["council_t3_cargo_cult_resurrection_c6_ibps_v2_20260519"]
    NSCS06v8early["council_per_substrate_symposium_nscs06_v8_path_b_20260517"]
    NSCS06v8mid["council_per_substrate_symposium_nscs06_v8_path_b_variant_c_reactivation_20260518"]
  end

  subgraph "Strategy + apparatus cite-chain"
    PathForward["council_t3_path_forward_recalibration_20260519"]
    TierBacklog["council_t3_tier_45_backlog_prioritization_20260519"]
  end

  THIS --> PR110edit
  THIS --> PR110yousfi
  THIS --> PR110mg16
  THIS --> PR110final
  THIS --> PR110upstream
  THIS --> DreamerV3sym
  THIS --> NSCS06v8sym
  THIS --> C6IBPSv2sym
  THIS --> PathForward
  THIS --> TierBacklog
  NSCS06v8sym --> NSCS06v8mid
  NSCS06v8mid --> NSCS06v8early
```

## Graph 5 — Orphan-signal extinction surface (cathedral autopilot)

```mermaid
graph LR
  subgraph "Before paradigm-shift (PRE-Catalog-335)"
    Orphan1["12 NEW tac.* namespaces<br/>2026-05-19 window<br/>ZERO consumed by autopilot"]
  end

  subgraph "After paradigm-shift (POST-Catalog-335)"
    PostShift1["NEW tac.cathedral_consumers/*<br/>package land"]
    PostShift2["Catalog #335 contract validated<br/>at design-time STRICT preflight"]
    PostShift3["Auto-discovery loop ingests<br/>at runtime via Catalog #336"]
    PostShift4["Catalog #337 master-gradient rerank<br/>actually fires"]
    PostShift5["Canonical routing markers per<br/>Catalog #341"]
    PostShift1 --> PostShift2
    PostShift2 --> PostShift3
    PostShift3 --> PostShift4
    PostShift4 --> PostShift5
  end

  Orphan1 -.bug class extinct.-> PostShift1
```

## Missing edges (gaps identified by symposium)

### MISSING EDGE 1: per-substrate symposium → OPTIMAL FORM iteration

**Problem:** Catalog #325 enforces per-substrate symposium EXISTS within 14 days; Catalog #315 enforces OPTIMAL FORM iteration; but no STRICT gate enforces symposium → iteration → dispatch chain (PROCEED_WITH_REVISIONS verdicts can dispatch).

**Op-routable:** Per Decision 3, refuse paid dispatch on substrates with PROCEED_WITH_REVISIONS verdict at runtime (operator-authorize.py check). Implementation = add to `tools/operator_authorize.py::_dispatch_modal` adjacent to Catalog #313 probe-outcomes check.

### MISSING EDGE 2: per-tier cadence audit → council deliberation proposal gate

**Problem:** `tools/audit_council_tier_cadence.py` exists; it produces OVER_CADENCE alerts; but no STRICT gate enforces consulting it before proposing a new T3 deliberation.

**Op-routable:** Per Decision 2, nominate `tools/audit_council_tier_cadence.py` as pre-deliberation discipline in CLAUDE.md "Council conduct" subsection (does NOT require new gate; behavior-change only).

### MISSING EDGE 3: provenance compliance backfill loop

**Problem:** Catalog #323 currently at 202 violations (down from 543 baseline); 136 MISSING_PROVENANCE + 66 INVALID_PROVENANCE_SHAPE; sweep deprioritized.

**Op-routable:** Per Decision 7, one small subagent session to (a) reclassify state-artifact rows as DERIVED_OUTPUT + emit waivers, (b) fix INVALID_PROVENANCE_SHAPE via one-pass schema fix.

### MISSING EDGE 4: bit-allocator canonical helper (Hook 3)

**Problem:** Catalog #125 Hook 3 (bit-allocator) is referenced in `tac.bit_allocator (proposed)` in canonical helper inventory but not yet a top-level canonical surface; existing per-byte sensitivity sourced from Catalog #354 exploit #3 + #4 doesn't have a dedicated top-level canonical helper.

**Op-routable:** Per Decision 10 (canonical-helper-sister-extension over new-tool), extend existing `tac.master_gradient_consumers` with `bit_allocator_from_per_byte_sensitivity` sister method rather than create new top-level surface.

### MISSING EDGE 5: unified Lagrangian phase 2 wire-in

**Problem:** Findings Lagrangian phase 1-a tests landed; phase 2 (full unified `S_total`) per Decision 5 is the long-term centerpiece but no concrete wire-in cadence.

**Op-routable:** Per Decision 5, nominate 1 subagent per week for 4-12 sessions to mature phase 2; per Decision 6 (consolidation-over-addition), the phase 2 work should SUBSUME existing per-track Lagrangians, not parallel them.
