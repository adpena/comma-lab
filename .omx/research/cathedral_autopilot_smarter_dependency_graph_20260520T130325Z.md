# Cathedral Autopilot Smarter Dependency Graph — 2026-05-20T13:03:25Z

> **Deliverable B (dependency graph) of SLOT CATHEDRAL-SMARTER-DESIGN-MEMO**
> **Lane**: `lane_cathedral_autopilot_smarter_design_blueprint_20260520`
> **Cite-chain**: master memo `cathedral_autopilot_smarter_design_blueprint_20260520T130325Z.md` + cost envelope sister

---

## Graph 1 — 6-dimension critical-path topology

```mermaid
graph LR
  subgraph "EXISTING (FULLY_WIRED ~35%)"
    Adjusters["10 adjust_predicted_delta_for_*<br/>in-main-line adjusters<br/>(score-mutating)"]
    Posterior["tac.continual_learning<br/>118 anchors"]
    Ledger["tac.probe_outcomes_ledger<br/>59 outcomes"]
    CallId["tac.deploy.modal.call_id_ledger<br/>393 rows"]
    Provenance["tac.provenance<br/>Catalog #323"]
    FrontierPtr["tac.canonical_frontier_pointer<br/>Catalog #343"]
    Equations["tac.canonical_equations<br/>11 registered"]
  end

  subgraph "EXISTING (FACADE ~65%; observability-only)"
    Consumers["44 cathedral_consumers/*<br/>STUB by Catalog #341<br/>predicted_delta=0.0"]
    Lagrangian["tac.findings_lagrangian<br/>+ findings_lagrangian_pp<br/>SCAFFOLD_ONLY at consumer"]
    UnifiedAction["tac.unified_action<br/>ZERO production callsites"]
    MasterGrad["tac.master_gradient<br/>10/10 non-authoritative"]
  end

  subgraph "DIMENSION 1 — Mathematical grounding (META-LAGRANGIAN-WIRE-1 + Phase 2-N)"
    D1P1["Phase 1: invocation point<br/>(WIRE-1 in flight)"]
    D1P2["Phase 2: per-adjuster ablation<br/>10 adjusters → solver terms"]
    D1P3["Phase 3: TRACK A+B ensemble"]
    D1P4["Phase 4: Dykstra feasibility<br/>+ Pareto constraints"]
    D1P5["Phase 5: sensitivity regularization"]
    D1P6["Phase 6+: deprecate hand-derived"]
  end

  subgraph "DIMENSION 2 — Feedback-loop frequency"
    D2S1["Step 2.1: auto-subscriber<br/>to call_id_ledger"]
    D2S2["Step 2.2: per-iteration<br/>auto_recalibrate"]
    D2S3["Step 2.3: consumer update_from_anchor<br/>pre-consumption"]
    D2S4["Step 2.4: CLI lookback flag"]
  end

  subgraph "DIMENSION 3 — Problem-space grounding (per-axis)"
    D3S1["Step 3.1: Protocol extension<br/>predicted_axis_decomposition"]
    D3S2["Step 3.2: tac.score_composition<br/>compose_score_from_axes"]
    D3S3["Step 3.3: ranker per-axis composition"]
    D3S4["Step 3.4: convert 3-5 high-EV consumers"]
    D3S5["Step 3.5: Catalog #341 sister-extension"]
  end

  subgraph "DIMENSION 4 — Domain grounding (comma2k19 / ego-motion)"
    D4S1["Step 4.1: tac.domain_priors<br/>namespace (sister-extend)"]
    D4S2["Step 4.2: 3 canonical equations<br/>per_frame / ego_motion / per_class"]
    D4S3["Step 4.3: domain_prior_consumer<br/>cathedral package"]
    D4S4["Step 4.4: substrate citation audit"]
  end

  subgraph "DIMENSION 5 — Continual-learning closure"
    D5S1["Step 5.1: propose_sister_candidates<br/>Protocol extension"]
    D5S2["Step 5.2: tac.substrate_kinship_graph"]
    D5S3["Step 5.3: auto-discovery extension"]
    D5S4["Step 5.4: Catalog #313 wire-in"]
    D5S5["Step 5.5: Catalog #344 wire-in"]
    D5S6["Step 5.6: --closed-loop-strict flag"]
  end

  subgraph "DIMENSION 6 — Dual-tier architecture"
    D6S1["Step 6.1: ConsumerTier enum"]
    D6S2["Step 6.2: consumer_tier field"]
    D6S3["Step 6.3: Catalog #341 tier-aware"]
    D6S4["Step 6.4: README dual-tier docs"]
    D6S5["Step 6.5: 3-5 consumers promote Tier B"]
  end

  %% EXISTING surfaces consumed by dimensions
  Adjusters --> D1P2
  Posterior --> D1P2
  Equations --> D1P2
  UnifiedAction --> D1P1
  Lagrangian --> D1P1
  CallId --> D2S1
  Posterior --> D2S1
  Equations --> D2S2
  Consumers --> D2S3
  Consumers --> D3S1
  Consumers --> D3S4
  Equations --> D3S2
  Adjusters --> D3S3
  MasterGrad --> D4S1
  Equations --> D4S2
  Posterior --> D5S2
  Ledger --> D5S4
  Equations --> D5S5
  Consumers --> D6S1

  %% Intra-dimension chain
  D1P1 --> D1P2
  D1P2 --> D1P3
  D1P3 --> D1P5
  D3S1 --> D3S2
  D3S2 --> D3S3
  D3S3 --> D3S4
  D3S4 --> D3S5
  D4S1 --> D4S2
  D4S2 --> D4S3
  D4S3 --> D4S4
  D5S1 --> D5S2
  D5S2 --> D5S3
  D5S3 --> D5S4
  D5S4 --> D5S5
  D6S1 --> D6S2
  D6S2 --> D6S3
  D6S3 --> D6S4
  D6S4 --> D6S5

  %% Inter-dimension dependencies (critical path)
  D1P3 --> D1P4
  D3S3 --> D1P4
  D4S3 --> D1P5
  D2S1 --> D5S1
  D3S1 --> D6S1
  D1P4 --> D1P6
  D1P5 --> D1P6

  %% Frontier impact (terminal nodes feed back to frontier)
  D1P6 -.frontier impact.-> FrontierPtr
  D3S5 -.frontier impact.-> FrontierPtr
  D4S4 -.frontier impact.-> FrontierPtr
  D5S6 -.frontier impact.-> FrontierPtr
```

## Graph 2 — 6-hook wire-in coverage per dimension

```mermaid
graph TD
  subgraph "Hook 1: Sensitivity-map"
    H1["tac.sensitivity_map.*"]
    D1P5_H1["Dim 1 P5: regularization R(θ)"]
    D3_H1["Dim 3: per-axis IS sensitivity"]
    D4_H1["Dim 4: domain priors ARE sensitivity"]
    D5_H1["Dim 5: kinship graph = substrate-level"]
    H1 --> D1P5_H1
    H1 --> D3_H1
    H1 --> D4_H1
    H1 --> D5_H1
  end

  subgraph "Hook 2: Pareto constraint"
    H2["Dykstra alternating projection<br/>tac.pareto_* + findings_lagrangian"]
    D1P4_H2["Dim 1 P4: feasibility constraints"]
    D3_H2["Dim 3: per-axis Pareto polytope"]
    H2 --> D1P4_H2
    H2 --> D3_H2
  end

  subgraph "Hook 3: Bit-allocator"
    H3["tac.bit_allocator (proposed)<br/>OR sister-extension"]
    D1_H3["Dim 1: solver-derived allocation"]
    D3_H3["Dim 3: per-axis archive bytes"]
    D4_H3["Dim 4: per-frame difficulty allocation"]
    H3 --> D1_H3
    H3 --> D3_H3
    H3 --> D4_H3
  end

  subgraph "Hook 4: Cathedral autopilot dispatch"
    H4["invoke_cathedral_consumers_on_candidates<br/>+ rerank_candidates_via_master_gradient"]
    AllDims["ALL 6 DIMENSIONS<br/>ACTIVE PRIMARY"]
    H4 --> AllDims
  end

  subgraph "Hook 5: Continual-learning posterior"
    H5["tac.continual_learning.posterior_update_locked<br/>+ tac.canonical_equations.update_*"]
    D1_H5["Dim 1: posterior_update_from_anchors"]
    D2_H5["Dim 2: auto-subscriber PRIMARY"]
    D3_H5["Dim 3: per-axis posteriors"]
    D4_H5["Dim 4: domain equation refresh"]
    D5_H5["Dim 5: closure PRIMARY"]
    H5 --> D1_H5
    H5 --> D2_H5
    H5 --> D3_H5
    H5 --> D4_H5
    H5 --> D5_H5
  end

  subgraph "Hook 6: Probe-disambiguator"
    H6["tools/probe_*_disambiguator.py"]
    D1_H6["Dim 1: solver-vs-adjuster divergence"]
    D3_H6["Dim 3: per-axis residual attribution"]
    D4_H6["Dim 4: contest-vs-comma2k19 priors"]
    D5_H6["Dim 5: per-falsification disambiguator"]
    D6_H6["Dim 6: tier-A-vs-tier-B paired"]
    H6 --> D1_H6
    H6 --> D3_H6
    H6 --> D4_H6
    H6 --> D5_H6
    H6 --> D6_H6
  end
```

## Graph 3 — Critical-path ordering with parallel paths

```mermaid
gantt
    title Cathedral Autopilot Smarter Blueprint (12-18 weeks)
    dateFormat YYYY-MM-DD
    axisFormat %Y-%m

    section Dim 1 (Mathematical)
    Phase 1 (META-LAGRANGIAN-WIRE-1; in flight)      :active, d1p1, 2026-05-20, 14d
    Phase 2 (per-adjuster ablation)                  :d1p2, after d1p1, 35d
    Phase 3 (TRACK A+B ensemble)                     :d1p3, after d1p2, 14d
    Phase 4 (Dykstra + Pareto)                       :d1p4, after d3s3, 21d
    Phase 5 (sensitivity regularization)             :d1p5, after d4s3, 14d
    Phase 6+ (deprecate adjusters)                   :d1p6, after d1p5, 14d

    section Dim 2 (Feedback loop)
    Step 2.1 (auto-subscriber)                       :d2s1, 2026-05-22, 14d
    Step 2.2 (per-iteration recalibrate)             :d2s2, after d2s1, 7d
    Step 2.3 (consumer update_from_anchor)           :d2s3, after d2s2, 10d
    Step 2.4 (CLI lookback)                          :d2s4, after d2s3, 7d

    section Dim 3 (Per-axis)
    Step 3.1 (Protocol extension)                    :d3s1, 2026-05-27, 7d
    Step 3.2 (compose_score_from_axes)               :d3s2, after d3s1, 7d
    Step 3.3 (ranker composition)                    :d3s3, after d3s2, 14d
    Step 3.4 (convert 3-5 consumers)                 :d3s4, after d3s3, 21d
    Step 3.5 (Catalog #341 extension)                :d3s5, after d3s4, 7d

    section Dim 4 (Domain priors)
    Step 4.1 (tac.domain_priors namespace)           :d4s1, 2026-05-27, 21d
    Step 4.2 (3 canonical equations)                 :d4s2, after d4s1, 14d
    Step 4.3 (domain_prior_consumer)                 :d4s3, after d4s2, 10d
    Step 4.4 (substrate citation audit)              :d4s4, after d4s3, 7d

    section Dim 5 (Closed-loop)
    Step 5.1 (propose_sister_candidates)             :d5s1, after d2s3, 14d
    Step 5.2 (substrate_kinship_graph)               :d5s2, after d5s1, 21d
    Step 5.3 (auto-discovery extension)              :d5s3, after d5s2, 10d
    Step 5.4-5.5 (Catalog wire-ins)                  :d5s4, after d5s3, 10d
    Step 5.6 (CLI strict flag)                       :d5s6, after d5s4, 7d

    section Dim 6 (Dual-tier)
    Step 6.1 (ConsumerTier enum)                     :d6s1, after d3s1, 7d
    Step 6.2-6.3 (Protocol + gate)                   :d6s2, after d6s1, 10d
    Step 6.4 (README)                                :d6s4, after d6s2, 7d
    Step 6.5 (3-5 Tier B promotions)                 :d6s5, after d6s4, 21d
```

## Graph 4 — Dimension × HORIZON-CLASS × Mission-Contribution matrix

```mermaid
graph TB
  subgraph "HORIZON-CLASS: asymptotic_pursuit (frontier_breaking)"
    AP1["Dim 3 (per-axis decomposition)<br/>frontier_breaking enabler"]
    AP2["Dim 4 (domain priors)<br/>frontier_breaking enabler"]
  end

  subgraph "HORIZON-CLASS: frontier_protecting (transition)"
    FP1["Dim 1 (mathematical grounding)<br/>frontier_protecting → frontier_breaking transition"]
    FP5["Dim 5 (closed-loop)<br/>frontier_protecting → frontier_breaking transition"]
  end

  subgraph "HORIZON-CLASS: plateau_adjacent (apparatus_maintenance)"
    PA2["Dim 2 (feedback loop)<br/>apparatus_maintenance"]
    PA6["Dim 6 (dual-tier)<br/>apparatus_maintenance"]
  end

  subgraph "Frontier impact targets"
    Target1["7 asymptotic-pursuit candidates<br/>from T3 Decision 4<br/>(DreamerV3 / NSCS06 v8 / Z7-Mamba-2 / Z6-v2 / V1 Faiss V8 / Q4-Q5 / rate-attack)"]
    Target2["10 in-main-line adjusters<br/>subsumed by unified solver"]
    Target3["44 cathedral consumers<br/>Tier B promotion candidates"]
  end

  AP1 --> Target1
  AP2 --> Target1
  FP1 --> Target2
  FP5 --> Target1
  PA2 --> Target2
  PA6 --> Target3
```

## Graph 5 — Existing → Smarter cathedral autopilot surface diff

```mermaid
graph LR
  subgraph "EXISTING cathedral autopilot main()"
    M1["10 adjust_predicted_delta_for_*<br/>(hand-derived)"]
    M2["invoke_cathedral_consumers_on_candidates<br/>(44 STUBS return 0.0)"]
    M3["rerank_candidates_via_master_gradient<br/>(non-authoritative anchors)"]
    M4["--load-continual-posterior<br/>(at-startup load only)"]
  end

  subgraph "SMARTER cathedral autopilot main()"
    SM1["solver.evaluate_with_admm<br/>(canonical solver per Dim 1 P1+P2)"]
    SM2["invoke_cathedral_consumers_on_candidates<br/>(Tier A + Tier B per Dim 6)"]
    SM3["rerank_candidates_via_master_gradient<br/>(authoritative anchors per master-gradient TODO)"]
    SM4["auto-subscriber + per-iteration recalibrate<br/>(canonical per Dim 2)"]
    SM5["per-axis compose_score_from_axes<br/>(canonical per Dim 3)"]
    SM6["domain_prior_consumer integration<br/>(canonical per Dim 4)"]
    SM7["sister-candidate kinship propagation<br/>(canonical per Dim 5)"]
  end

  M1 -.subsumed.-> SM1
  M2 -.extended.-> SM2
  M3 -.fed by master-gradient on contest hw.-> SM3
  M4 -.replaced.-> SM4
  M2 -.composed.-> SM5
  M2 -.composed.-> SM6
  M2 -.composed.-> SM7
```

## Graph 6 — Operator-attention allocation per Dimension (recommended)

```mermaid
pie title Operator-attention allocation (recommended; 30-day window)
    "Dim 1 Phase 2 (per-adjuster ablation)" : 25
    "Dim 3 (per-axis decomposition)" : 20
    "Dim 4 (domain priors)" : 20
    "Dim 2 (feedback loop)" : 10
    "Dim 5 (closed-loop)" : 10
    "Dim 6 (dual-tier)" : 10
    "Dim 1 Phase 1 oversight (META-LAGRANGIAN-WIRE-1)" : 5
```

## Missing edges / unresolved dependencies

| Missing | Description | Resolution path |
|---|---|---|
| `findings_lagrangian_consumer` cathedral consumer | Referenced in T3 dependency graph (Graph 2 Hook 4) but `src/tac/cathedral_consumers/findings_lagrangian_consumer/` does NOT exist | Owned by META-LAGRANGIAN-WIRE-1 likely; verify in landing memo |
| `tac.bit_allocator` top-level helper | Referenced in T3 graph as "proposed" | Per Decision 10: sister-extend `tac.master_gradient_consumers.bit_allocator_from_per_byte_sensitivity` |
| Authoritative master-gradient anchor | All 10 anchors non-authoritative `[macOS-CPU advisory]` | Operator-routable: 1 paid Modal A100 / T4 dispatch ($2-5) to land contest-CUDA anchor (independent of blueprint) |
| Lagrangian `OptimalPerPairTreatmentPlan` sidecar | 0 sidecars exist; Cascade 1 falls through | Operator-routable: produce 1 sidecar from authoritative master-gradient anchor (independent of blueprint) |
| `tac.substrate_kinship_graph` | New canonical helper (Dim 5 Step 5.2); does not exist | This blueprint proposes it |
| `tac.score_composition.compose_score_from_axes` | New canonical helper (Dim 3 Step 3.2); does not exist | This blueprint proposes it |
| `tac.domain_priors` namespace | New canonical namespace (Dim 4 Step 4.1); does not exist | This blueprint proposes it; sister-extends existing surfaces |
