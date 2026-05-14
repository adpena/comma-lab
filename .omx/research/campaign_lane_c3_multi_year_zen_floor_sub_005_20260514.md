---
title: "Campaign C3 — Multi-Year Zen-Floor Sub-0.05 Pursuit"
date: 2026-05-14
status: campaign_ledger; multi-year-charter
lane_id: lane_c3_multi_year_zen_floor_sub_005_campaign_20260514
score_claim: false
evidence_axes: [time-traveler-prediction, mathematical-derivation, council-deliberation, production-deployment]
score_claim_valid: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
target_modes: [contest_exact_eval, production_generalized, production_edge_adaptive]
deployment_target: comma_ai_production  # primary; also t4_contest_runtime
campaign_tier: multi_year
expected_horizon_weeks: 52-156  # 1-3 years
expected_dispatch_cost_usd_band: [500, 2000]  # multi-year
expected_delta_S_band: [-0.020, -0.050]  # cumulative on top of C2 (post-staircase asymptote)
predicted_post_campaign_score_band: [0.02, 0.05]  # Time-Traveler deep-future band
council_tally: time_traveler_only_unanimously_endorses  # 20% confidence per council
operator_decision_required: true
operator_decision_horizon: multi_year_strategic
---

# Campaign C3 — Multi-Year Zen-Floor Sub-0.05 Pursuit

## 0. One-line summary

Multi-year charter for the 0.02-0.05 zen-floor band per Time-Traveler. Beyond the L5 staircase asymptote. Predicted final post-campaign score band **[0.02, 0.05]** (Time-Traveler's deep-future prediction; council ~20% confidence). Cost $500-2000 over 1-3 years. Strategic priorities beyond contest score: production deployment alignment (comma.ai openpilot integration), public release (HuggingFace), papers + benchmarks. **OPERATOR STRATEGIC DECISION REQUIRED.**

## 1. lane_id + dispatch-claim plan

- `lane_id`: `lane_c3_multi_year_zen_floor_sub_005_campaign_20260514` (pre-registered L0, phase 4)
- Dispatch claim: append row per substrate-iteration cycle (3-6 generations of L5 evolution); per cross-substrate composition matrix dispatch; per public release point.
- Multi-year claim lifecycle: claims pruned per Catalog #154 every 7 days; campaign-level summary appended to `.omx/state/c3_multi_year_summary_ledger.md` quarterly.

## 2. Source evidence + score-lowering hypothesis

**Source evidence:**
- `zen_floor_field_medal_grade_council_20260514.md` Section 7 (multi-year + $500+ tier = 0.02-0.03 absolute zen-floor, 20% confidence)
- Time-Traveler unanimous endorsement (their timeline's empirical anchor; our timeline's prediction needing validation)
- Shannon 1959 vector R(D) absolute theoretical floor (per ancient-elder §16 — 2-3 days of mathematics replaces empirical 0.10±0.03 floor)
- Council F multi-year horizon mapping (Hotz "engineering-effort-funding-integral" canonical zen-floor predictor)
- `feedback_adjusted_floor_v3_alien_tech_routing_landed_20260513.md` (floor v3 multi-year horizon row: 0.02-0.03 at $500+ multi-year)

**Hypothesis (multi-year first-principles):**
1. **Sustained investment hits Shannon-1959 vector R(D) lower bound asymptote.** Per ancient-elder §16, Shannon 1959 defined R(D) for vector-valued distortion (exactly our 3-axis contest). Currently nobody in the lab has cited Shannon 1959 for this purpose. 2-3 days of theoretical work + sustained empirical iteration could replace Council F's empirical 0.10±0.03 floor with a DERIVED bound — single highest-leverage theoretical work in next quarter.
2. **Substrate iteration cycles (3-6 generations of L5 evolution) reach diminishing returns at 0.04-0.05.** Each generation tightens the cooperative-receiver bound, the prediction-coding hierarchy, and the foveation matching. Empirical Time-Traveler prediction: 6 generations of iteration over 18-36 months.
3. **Cross-substrate composition matrix mature** — all pairwise substrate compositions (Z6 × A1 × D4 × Wyner-Ziv full × E4 MDL-IBPS × DARTS output) systematically empirically tested; the convex hull of the achievable Pareto cone is FULLY EXPLORED.
4. **Adversarial scorer-robustness research** — counter the implementation-noise floor (CUDA-CPU drift, FP4 quantization noise, scorer-version drift). Below 0.05, implementation noise dominates; this campaign explicitly invests in tightening the noise budget.
5. **Production deployment alignment** — comma.ai openpilot integration; the contest exists to support production deployment. Below 0.05, the contest score is a proxy for production-deployment value; aligning explicitly with production is required for sustained motivation + funding.
6. **HuggingFace public release + community contributions** — open-source the substrate; community iteration extends to research surfaces we cannot afford solo.
7. **Publish papers + benchmark** — Shannon 1959 application, Time-Traveler substrate architecture, cooperative-receiver theorem operationalized in modern neural compression.

## 3. Timing-smoke command (continuous; per-iteration)

Per CLAUDE.md "Long-burn" non-negotiable: continuous campaign means each substrate-iteration cycle has its own timing-smoke. Use the canonical pattern:

```bash
# Per iteration N (continuous loop over multi-year campaign)
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe .omx/operator_authorize_recipes/substrate_c3_multi_year_iterN_modal_a100_dispatch.yaml \
    --smoke-epochs 100 --smoke-batch-size 4 \
    --max-cost-usd 0.50 \
    --advisory-cpu-explicitly-waived

# Quarterly cross-substrate composition matrix re-run
.venv/bin/python tools/build_cross_substrate_composition_matrix.py \
    --substrates all_active --max-cells 32 --cost-band-usd 50 \
    --output-dir experiments/results/c3_composition_matrix_Q${QUARTER}_$(date -u +%Y%m%dT%H%M%SZ)
```

**Per-iteration kill criterion:** if Stage 0B smoke shows no ≥0.002 cumulative improvement vs prior iteration after 3 consecutive iterations → saturation; escalate to grand council for strategic re-routing.

## 4. Full-run command (resumable + harvest; multi-stage)

### Track A: Score-side campaign (~$300-700 over 18-36 months)

```bash
# 3-6 generations of L5 substrate evolution, $30-100 each
for GEN in 1 2 3 4 5 6; do
  .venv/bin/python tools/operator_authorize.py \
      --recipe substrate_c3_multi_year_l5_gen${GEN}_modal_a100_dispatch \
      --operator-authorize-confirmed-via-session-directive \
      --operator-authorize-session-budget-usd 700.0 \
      --resume-from-checkpoint experiments/results/c3_gen$((GEN-1))/best_ckpt.pt
done
```

### Track B: Production deployment alignment campaign (~$200-500 over 12-24 months)

```bash
# Production-edge substrate maturation (target: comma.ai openpilot integration)
.venv/bin/python experiments/train_substrate_c3_production_edge.py \
    --target-modes openpilot_edge,production_edge_adaptive \
    --deployment-target comma_ai_production \
    --output-dir experiments/results/c3_production_edge_$(date -u +%Y%m%dT%H%M%SZ)

# Production-generalized regression test suite (per CLAUDE.md "Contest vs production target modes")
.venv/bin/python tools/run_production_generalized_regression_suite.py \
    --substrates c3 --videos comma2k19 --output-dir experiments/results/c3_production_regression_$(date)
```

### Track C: Public release + papers (~$50-200 over 24-36 months)

```bash
# HuggingFace public release prep
.venv/bin/python tools/prep_huggingface_release.py \
    --substrate c3_mature_l5 --version v0.1.0 \
    --output-dir .omx/oss_export/c3_l5_substrate_release_v0.1.0/

# Papers: Shannon 1959 application + Time-Traveler architecture + cooperative-receiver
# (sister memos in docs/paper/ when ready)
```

**Harvest path:** `experiments/results/c3_gen*/harvested_artifacts/` per Catalog #204; quarterly summary to `.omx/state/c3_multi_year_summary_ledger.md`.

## 5. Live provider rate/cost model

| Track | Dispatch cadence | $/quarter | Total / year |
|---|---|---|---|
| Track A: Score-side | 1 generation per 3-6 months | $50-100 | $200-400 |
| Track B: Production-edge | continuous regression suite | $30-60 | $120-240 |
| Track C: Public release | quarterly releases | $20-40 | $80-160 |
| **Total per year** | | | **$400-800** |
| **Total 1-3 year campaign** | | | **$500-2000** |

**Cost gate:** every dispatch routes through `tac.cost_band_calibration` per Catalog #175/#177. Refuse if estimated > 1.5× band.

**Operator funding decision:** multi-year campaign explicitly requires sustained operator funding commitment. Per CLAUDE.md "Long-burn" non-negotiable, historical $24/$300 caps are superseded by explicit operator strategic decisions.

## 6. Byte-closed archive/export/inflate plan

**Multi-generation L5 archive evolution:**

- **Gen 1 (post-C2 mature L5, ~90 KB):** unchanged from C2 baseline; this generation's role is to validate C2's claim survives long-term.
- **Gen 2-3 (~70-80 KB):** Shannon 1959 vector R(D) lower bound application; tropical compression on SegNet argmax; Mallat scattering on rate term.
- **Gen 4-5 (~55-70 KB):** alien-tech compositions (E4 MDL-IBPS + DARTS-discovered substrate + Wyner-Ziv full pair).
- **Gen 6+ (~40-60 KB):** target asymptotic 0.02-0.05 zen-floor; archive grammar permits production-edge variant with deterministic fallback per CLAUDE.md "Contest vs production target modes" non-negotiable.

**Inflate.py LOC budget:** ≤250 LOC across all generations (matured substrate allows the cap; explicit waiver for matured composition).

**Production-generalized variant:** must preserve cross-video behavior, portability, maintainability per CLAUDE.md "Deterministic packet compiler" non-negotiable; deterministic native builds; explicit `target_modes=["production_generalized"]` declaration in lane registry.

**Export-first contract:** each generation declared in `.omx/research/campaign_c3_gen${N}_export_contract_20*.md` BEFORE training.

## 7. Stop/continue thresholds

| Year | Continue if | Stop if (DEFERRED-pending-research per CLAUDE.md KILL-LAST-RESORT) |
|---|---|---|
| Year 1 | C2 lands successfully; gen 1-2 show ≥-0.003 cumulative each | gen 2 shows no improvement OR Shannon 1959 derivation falsifies the [0.02, 0.05] band |
| Year 2 | gen 3-4 show ≥-0.005 cumulative; production-edge regression suite passes | gen 4 score >0.10 OR production-edge regression suite fails repeatedly |
| Year 3 | gen 5-6 show ≥0.005 cumulative; HuggingFace release published | gen 6 saturation at >0.07 OR no production deployment uptake |
| Multi-year exit | [contest-CPU] AND [contest-CUDA] in [0.02, 0.05] band → council ratify; paper accepted; production deployment confirmed | else: DEFERRED-pending-research; preserve all artifacts |

**Falsification criteria (campaign-level):**
- If gen 3 final [contest-CPU] > 0.10 → Time-Traveler deep-future prediction PARTIALLY FALSIFIED. The L5-staircase asymptote at 0.07-0.10 may be the true floor (Council F engineering 0.10±0.03 confirmed).
- If gen 6 final saturates >0.07 → multi-year path falsified; reactivation criteria = paradigm shift (new scorer family, new contest formulation, alien-tech break-through from DARTS-SuperNet).

**Reactivation criteria if KILL is genuinely warranted:**
- Every plausible architectural / training / codec angle attempted empirically (≥10 configurations over 6+ generations)
- Grand council UNANIMOUS CONSENSUS over 3 quarterly reviews
- Production-deployment value-axis demonstrates the engineering-effort-funding-integral has saturated
- Reactivation criteria documented in `feedback_c3_multi_year_DEFERRED_pending_<reason>_20*.md`

## 8. Dependencies + sequence gating

```
C1 (Z6 world model + foveation) lands successfully in [0.06, 0.10]
   ↓
C2 (Z7 mature predictive-receiver L5) lands successfully in [0.035, 0.07]
   ↓
**THIS CAMPAIGN (C3): 1-3 years, $500-2000, 3-6 substrate generations + production + public**
```

C3 cannot proceed before C2 lands successfully. If C2 lands >0.10, C3 deferred to alien-tech composition campaign (C5+C6+C7).

## 9. 6-hook wire-in (Catalog #125 NON-NEGOTIABLE)

1. **Sensitivity-map (`tac.sensitivity_map.*`)** — ENGAGED. Multi-year substrate iteration registers `sensitivity_map.c3_multi_year_unified_v${GEN}` per generation; each gen extends the prior with new sensitivity priors discovered in that iteration's empirical anchor.
2. **Pareto constraint (`tac.pareto_*`)** — ENGAGED. Each generation's archive size + per-pair byte budget + scorer-component bands register as Pareto constraints. Shannon-1959 vector R(D) lower bound from ancient-elder §16 derived + registered as theoretical floor (single highest-leverage theoretical work per the ancient-elder memo).
3. **Bit-allocator hook (`tac.composition.registry`)** — ENGAGED. Per-generation bit allocation across Stage 1 (world model) / Stage 2 (prediction error) / cross-substrate composition cells; register `bit_allocator.c3_multi_year_unified_v${GEN}`.
4. **Cathedral autopilot dispatch hook** — ENGAGED. Each generation's iteration adds rows to `.omx/state/autopilot_candidate_queue_solver_stack_wire_in_20260513.jsonl`. The cross-substrate composition matrix (Quarterly Q1-Qmax) adds 16-32 rows per quarter.
5. **Continual-learning posterior update (`tac.continual_learning.append_anchor`)** — ENGAGED. Every iteration + every composition cell + every regression test produces empirical anchors. This is the largest empirical-anchor producer in the lab's history; per Catalog #128 `posterior_update_locked` is essential under multi-process concurrency.
6. **Probe-disambiguator** — ENGAGED. At each generation, ≥2 defensible interpretations exist for the next gen's design space. Multi-year substrate-discovery probe orchestration loop:
   - `tools/probe_c3_generation_disambiguator.py` (master orchestrator; runs all per-gen probes)
   - `tools/probe_c3_shannon_1959_application.py` (vector R(D) derivation probe)
   - `tools/probe_c3_alien_tech_composition_probe.py` (DARTS + E4 + Wyner-Ziv composition selection)

## 10. Cross-references

- `.omx/research/zen_floor_field_medal_grade_council_20260514.md` (Section 7 multi-year band [0.02, 0.03])
- `.omx/research/time_traveler_architecture_reverse_engineered_20260513.md`
- `feedback_ancient_elder_polymath_landed_20260513.md` §16 Shannon 1959 vector R(D)
- `feedback_adjusted_floor_v3_alien_tech_routing_landed_20260513.md`
- `feedback_expert_team_signal_processing_alien_tech_landed_20260513.md` (alien-tech primitives)
- `feedback_expert_team_aerospace_stealth_analytic_alien_tech_landed_20260513.md`
- CLAUDE.md "Long-burn score-lowering campaign default" non-negotiable
- CLAUDE.md "Contest vs production target modes" non-negotiable (production_generalized / openpilot_edge required)
- CLAUDE.md "Public Disclosure Hygiene" (HuggingFace release prep must sanitize per Catalog #208)
- CLAUDE.md "Beauty, simplicity, and developer experience" (public release prep is the test of beauty/simplicity)

## 11. Operator-routable decision

**STATUS:** OPERATOR STRATEGIC DECISION REQUIRED (multi-year commitment $500-2000 + 1-3 years).

**Council confidence:** ~20% per zen-floor council Section 7 (multi-year + $500+ row).

**Recommended timing:** authorize after C2 lands successfully in [0.035, 0.07] band AND operator strategic alignment with comma.ai production roadmap is confirmed.

**Decision matrix entry:** NOW=NO; SOON=NO; LATER=PENDING-OPERATOR-STRATEGIC-DECISION; DEFER=if C2 lands >0.10.

**OPERATOR routing question:** is the operator committed to multi-year sustained funding for sub-0.05 zen-floor pursuit AND production deployment alignment? If yes, C3 authorize; if not, sunset at C2.

Tag: `[council-deliberation]` + `[operator-strategic-decision-required]` + `[multi-year-charter]`. NO score claim. NO archive bytes built. $0 GPU spent at landing.
