---
title: "Campaign C7 — DARTS-SuperNet Architecture Search (Substrate Family Discovery)"
date: 2026-05-14
status: campaign_ledger; operator-routable
lane_id: lane_c7_darts_supernet_architecture_search_campaign_20260514
score_claim: false
evidence_axes: [literature-prediction, mathematical-derivation, council-deliberation]
score_claim_valid: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
target_modes: [contest_exact_eval, contest_generalized]
deployment_target: t4_contest_runtime
campaign_tier: medium_to_long_term
expected_horizon_weeks: 6-12
expected_dispatch_cost_usd_band: [100, 300]
expected_delta_S_band: [-0.005, -0.030]  # discovery-driven; high variance
predicted_post_campaign_score_band: [0.12, 0.19]  # uncertain; depends on discovery
---

# Campaign C7 — DARTS-SuperNet Architecture Search

## 0. One-line summary

DARTS-SuperNet (Differentiable Architecture Search) over substrate family parameterized search space. Identifies the OPTIMAL substrate family from a discrete-continuous architecture search. Predicted score band **[0.12, 0.19]** with high variance (discovery-driven). Cost $100-300. Multi-arch search; produces 5-20 candidate architectures empirically ranked.

## 1. lane_id + dispatch-claim plan

- `lane_id`: `lane_c7_darts_supernet_architecture_search_campaign_20260514` (pre-registered L0, phase 3)
- Existing sister lane: `lane_darts_supernet_time_traveler_architecture_search_20260513` (predecessor L0 design memo with Time-Traveler framing)
- This campaign IMPLEMENTS the DARTS-SuperNet sweep with operator-funded compute budget.

## 2. Source evidence + score-lowering hypothesis

**Source evidence:**
- `feedback_solver_stack_wire_in_sweep_landed_20260513.md` (DARTS-SuperNet wired into autopilot candidate queue)
- Liu et al. 2019 DARTS: Differentiable Architecture Search (ICLR 2019)
- Pham et al. 2018 ENAS: Efficient Neural Architecture Search
- Ren et al. 2021 NAS-Bench-Suite-Zero: Accelerating Research on Zero Cost Proxies
- `feedback_grand_council_first_principles_original_score_lowering_landed_20260513.md` (Council F first-principles original)
- Time-Traveler memo (DARTS over the L5 architecture parameterized space)

**Hypothesis (literature-prediction):**
1. DARTS-SuperNet searches over a continuous-relaxed discrete architecture space (mixing weights `α` over operations; final architecture selected by argmax).
2. The substrate-family search space includes: NeRV variants (FF/Block/DS/Hi/TC), coordinate-MLP variants (SIREN, FINER, WIRE, BACON), HNeRV-family LC v2 forks, hybrid renderer+residual, score-aware vs score-blind trainers.
3. Predicted discovery: at least 1 architecture variant beats every existing substrate by ≥0.005 ΔS at fixed budget.
4. High variance: discovery-driven; ΔS band [-0.005, -0.030] reflects the empirical NAS literature outcome (typically 1-2% improvement over hand-designed baselines).
5. The discovered architecture becomes a NEW SUBSTRATE candidate for C2 / C3 maturation.

**Why this is NOT a research-only memo:**
- The DARTS-SuperNet framework is well-established (5+ years of literature); implementation is engineering, not research.
- The substrate-family search space is already enumerated in `tac.composition.registry.canonical_primitive_inventory()` (per `feedback_solver_stack_wire_in_sweep_landed_20260513.md`).
- Build effort: 2-3 weeks for SuperNet wrapper + 4-9 weeks for empirical sweeps + 1-2 weeks for final architecture export.

## 3. Timing-smoke command (≤$1, ≤60 min)

```bash
# Stage 0A — macOS-CPU advisory smoke (search space verification)
.venv/bin/python experiments/train_substrate_c7_darts_supernet.py \
    --epochs 5 --batch-size 1 --quick-smoke \
    --search-space-size 4 \
    --output-dir experiments/results/c7_smoke_macos_$(date -u +%Y%m%dT%H%M%SZ) \
    --advisory-cpu-explicitly-waived

# Stage 0B — Modal A100 timing smoke (per-architecture per-search-iteration cost)
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe .omx/operator_authorize_recipes/substrate_c7_darts_supernet_smoke_modal_a100_dispatch.yaml \
    --smoke-epochs 100 --smoke-batch-size 4 \
    --search-space-size 8 \
    --max-cost-usd 1.00
```

**Timing-smoke kill criterion:** if Stage 0B does not produce ranked candidate architectures within 60 min OR per-arch wall-clock > 30 min, escalate (search space too large; reduce `--search-space-size`).

## 4. Full-run command (resumable + harvest)

```bash
# Stage 1 — DARTS-SuperNet inner-loop sweep (3-5 weeks, $50-150 Modal A100)
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_c7_darts_supernet_inner_loop_modal_a100_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 300.0 \
    --search-space-size 16 \
    --inner-loop-iterations 50

# Stage 2 — Outer-loop architecture selection (1-2 weeks, $20-50 Modal A100)
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_c7_darts_supernet_outer_loop_modal_a100_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 300.0 \
    --resume-from-checkpoint experiments/results/c7_inner_loop/best_supernet.pt

# Stage 3 — Top-K architecture full training (3-4 weeks, $30-100 Modal A100)
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_c7_darts_top_k_full_modal_a100_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 300.0 \
    --top-k 5 \
    --resume-from-checkpoint experiments/results/c7_outer_loop/best_arch_*.pt
```

**Harvest path:** `experiments/results/c7_*/harvested_artifacts/` per Catalog #204. Architecture-export to `experiments/results/c7_discovered_substrates/`.

## 5. Live provider rate/cost model

| Provider | GPU | $/hr | Per-stage wall-clock | Per-stage $ |
|---|---|---|---|---|
| Modal | A100-80GB | $5.40 | 10-30h (inner loop) | $54-162 |
| Modal | A100-40GB | $4.10 | 5-12h (outer loop) | $20-49 |
| Modal | A100-80GB | $5.40 | 6-18h (top-K full) | $32-97 |
| Vast.ai | RTX 4090 | $0.25 | 40-100h | $10-25 |

**Total campaign cost band: $100-300** (3 stages; high variance due to search-space-size choice).

**Cost gate:** consult `tac.cost_band_calibration` per Catalog #175/#177. Refuse if estimated > 1.5× band.

## 6. Byte-closed archive/export/inflate plan

**Architecture export contract (per discovered substrate):**

Each top-K discovered architecture gets its own L1+ archive grammar declaration:

```
DiscoveredArchitectureArchive-C7-Arch${K} (target: variant-dependent, 80-150 KB)
├── HEADER (~2 KB): magic + lengths + grammar version + architecture descriptor
├── DECODER (~60-110 KB; varies per discovered architecture)
│   └── [architecture-dependent layout; export contract emitted by trainer]
├── LATENTS (~10-30 KB; varies)
├── SIDE INFO (~10-20 KB if applicable)
├── ARITHMETIC CODING STATE (~5-10 KB)
└── SECTION OFFSETS (~2-3 KB)
```

**Inflate.py LOC budget: ≤250 LOC per discovered architecture** (HNeRV parity lesson 4; discovered architectures may push limit; explicit waiver allowed with rationale).

**Per-arch export contract** must be DECLARED BEFORE training each top-K architecture per Catalog #146.

**Score-aware loss + differentiable scorers:** every discovered architecture trained with `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` per Catalog #187.

## 7. Stop/continue thresholds

| Stage | Continue if | Stop if (DEFERRED-pending-research) |
|---|---|---|
| Stage 0 (smoke) | seconds/epoch < 60s per arch; search space converging | timeout OR no architecture under 200 KB |
| Stage 1 (inner loop) | SuperNet trains to ≤0.20 macOS-CPU; α-mixture distinguishable | SuperNet diverges OR α-mixture remains uniform after 50 iter |
| Stage 2 (outer loop) | top-K architectures ranked; ≥3 architectures show distinct ΔS | only 1 architecture survives → de-facto hand-designed baseline |
| Stage 3 (top-K full) | top-1 architecture macOS-CPU [0.12, 0.19] | top-1 > 0.20 → campaign falsified |
| Exact eval | [contest-CPU] AND [contest-CUDA] in [0.12, 0.19] → L2 promotion of top-1 | both >0.20 → DEFERRED |

**Falsification criteria:**
- If Stage 3 top-1 [contest-CPU] > 0.20 → DARTS-SuperNet hypothesis falsified for the contest search space.
- Reactivation: search space expansion OR DARTS variant (PC-DARTS, P-DARTS, NSGA-Net, GDAS) substitution.

## 8. Dependencies + sequence gating

```
C4f (cathedral autopilot active) RECOMMENDED FIRST  [for orchestration]
   ↓
C5 OR C6 lands  [provides empirical anchor baseline; informs search space]
   ↓
**THIS CAMPAIGN (C7): $100-300, 6-12 weeks, 3 stages, 5-20 candidate architectures**
   ↓
C2 / C3 substrate iteration: TRIGGERS on successful C7 discovery
```

C7 has WEAK dependencies on C5/C6; can be dispatched in parallel if operator funding permits.

## 9. 6-hook wire-in (Catalog #125 NON-NEGOTIABLE)

1. **Sensitivity-map (`tac.sensitivity_map.*`)** — ENGAGED. DARTS architecture α-mixture IS a sensitivity profile over architectural operations; register `sensitivity_map.darts_alpha_v1` per discovered top-K.
2. **Pareto constraint (`tac.pareto_*`)** — ENGAGED. Each discovered architecture adds (params, archive_bytes, inflate_loc, runtime_sec, score_components) point to the Pareto cone; register `tac.pareto.darts_discovered_v${K}`.
3. **Bit-allocator hook (`tac.composition.registry`)** — ENGAGED. Discovered architectures with content-adaptive embeddings register their per-component bit-allocator; the DARTS framework itself provides architecture-level bit-allocator selection.
4. **Cathedral autopilot dispatch hook** — ENGAGED. Add to autopilot queue with cost-band $100-300, predicted_dS [-0.005, -0.030] (high-variance), EIG VERY-HIGH (closes the architecture-space search question).
5. **Continual-learning posterior update (`tac.continual_learning.append_anchor`)** — ENGAGED. Top-K discovered architectures = K empirical anchors per Stage 3. The largest single architecture-discovery batch in the campaign portfolio.
6. **Probe-disambiguator** — ENGAGED. Two defensible interpretations: (a) "DARTS-SuperNet discovers fundamentally new architecture" vs (b) "DARTS rediscovers hand-designed NeRV/HNeRV family at search-space limit". Probe `tools/probe_c7_novel_vs_rediscovery.py` (compares top-K discovered architectures against hand-designed baselines via architectural similarity metrics).

## 10. Cross-references

- `feedback_solver_stack_wire_in_sweep_landed_20260513.md` (DARTS-SuperNet wired into autopilot)
- `feedback_grand_council_first_principles_original_score_lowering_landed_20260513.md` (Council F floor 0.10±0.03)
- `feedback_adjusted_floor_v3_alien_tech_routing_landed_20260513.md` (alien-tech routing for discovery)
- Liu et al. 2019 DARTS DOI 10.48550/arXiv.1806.09055
- Pham et al. 2018 ENAS DOI 10.48550/arXiv.1802.03268
- Ren et al. 2021 NAS-Bench-Suite-Zero arXiv:2104.01177
- CLAUDE.md "Experiment design — non-negotiable" (no janky smoke tests; representative resolution; enough steps for signal)

## 11. Operator-routable decision

**STATUS:** READY-TO-AUTHORIZE; weak dependencies allow parallel dispatch.

**Decision matrix entry:** NOW=PARTIAL (Stage 0 smoke YES if budget permits); SOON=YES (post-C5/C6 landing); LATER=YES; DEFER=if operator funding tier < Tier 2 ($300+).

**Recommended timing:** authorize after at least one substrate campaign (C5 or C6) lands to inform the search space. Stage 0 smoke ($1) can be authorized immediately to validate the framework.

**OPERATOR routing question:** is the operator interested in DISCOVERING new architectures or just OPTIMIZING existing ones? If discovery, C7 authorize; if optimization, defer C7 in favor of C2 iteration.

Tag: `[literature-prediction]` + `[discovery-driven]` + `[high-variance]`. NO score claim at landing. NO archive bytes built. $0 GPU spent at landing.
