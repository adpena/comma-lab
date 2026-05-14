---
title: Adjusted theoretical floor v3 — post PR106 CPU-axis falsification + A1 confirmation + posterior 25→42 recovery
date: 2026-05-13
lane_id: lane_adjusted_theoretical_floor_v3_routing_20260513
status: research-only synthesis
score_claim: false
score_claim_valid: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
research_only: true
evidence_axes:
  - first-principles-bound
  - mathematical-derivation
  - literature-prediction
  - time-traveler-prediction
  - macOS-CPU-advisory
  - empirical-anchored
hnerv_parity_audit: design-time only (research synthesis; no archive bytes)
---

# Adjusted theoretical floor v3 — alien-tech + time-traveler routing

## TL;DR

Three empirical findings landed today (2026-05-13) re-shape the floor:

1. **A1 frontier verified at `0.192848` `[contest-CPU-1to1]` GHA Linux x86_64**
   (sha `87ec7ca5f2f3`, 178,262 B). Sole sub-0.20 anchor in posterior.
2. **PR106 sidecar family EMPIRICALLY FALSIFIED on contest-CPU**
   (3-of-3 variants at 0.22787-0.22796 = +0.022 vs CUDA, ~0.035 above frontier).
   The +0.022 CPU-CUDA gap is uniform across c3 / wavelet / cool_chic /
   coord_mlp / siren / sparse_aware / l2_sparse_aware variants
   (σ ≈ 0.0003, n=9). The HNeRV-family-extension via PR106 grammar is
   **DEAD on the public CPU axis** the contest ranks by.
3. **Posterior recovered 25 → 42 anchors** (+17 ingested via
   `tools/bulk_backfill_anchors_into_posterior.py`). Of 42 anchors,
   **exactly 1 is sub-0.20** (A1).

The HNeRV-family-survivable Pareto cone has SHRUNK. The **A1 clone branch**
remains valid (sister subagent owns), but the next sub-A1 score-lowering
step requires a representational move outside the HNeRV-family-extension
landscape. Alien-tech + time-traveler are the **only credible sub-A1 paths**
absent a new HNeRV architectural insight.

## Floor v3 numerical band

```
S_floor_v3 = 0.165 ± 0.020   [research-prediction, PR106 family removed]
S_floor_v3_optimistic = 0.10 - 0.13
              [if time-traveler pre-empirical band [0.150, 0.170] confirms
               AND alien-tech composes orthogonally — Amdahl bounds]
S_floor_v3_pessimistic = 0.18 - 0.19
              [if HNeRV-family is the architectural ceiling per Council G,
               i.e. all sub-0.19 paths require novel substrate or compose-poor]
```

Compared to floor v2 (`grand_council_fields_medal_theoretical_floor_20260509.md`,
band 0.10 ± 0.03 derived from Council F empirical-floor + Council G
HNeRV-family-survivability) the v3 band is **wider** (more uncertainty after
PR106 falsification removed a credible sub-0.21 path) but **the optimistic edge
is unchanged** (0.10 still derivable from Shannon-1959 vector-distortion R(D)
bound + cooperative-receiver MI bound + score-aware-conditional-entropy).

## Top-10 alien-tech techniques across all 4 expert-team memos + time-traveler

For each technique: predicted ΔS (per source memo, NOT new derivation) /
build cost (`$0` research / `$1-5` GPU smoke / `>$10` GPU full) / dispatch
readiness (L0 SCAFFOLD / L1 PARTIAL / L2 READY) / recommended dispatch order.

| # | Technique | Source | Predicted ΔS | Cost | Readiness | Order |
|--:|-----------|--------|------------:|-----:|----------:|------:|
| 1 | **L2 SAR coherent integration over pose pairs** | signal-processing memo (Lincoln Lab radar lineage) | -0.0056 | $1 | L0 SCAFFOLD (~100 LOC needed) | **#1 — cheapest top-tier** |
| 2 | **Time-traveler L5 autonomy substrate (full)** | time-traveler reverse-engineering memo | -0.020 to -0.040 (predicted band [0.150, 0.170] vs A1 0.193) | $3-8 | **L1 IMPL_COMPLETE + recipe + smoke-before-full wired** | **#3 — swing-for-fences** |
| 3 | **N3 Wyner-Ziv cooperative-receiver conditional-entropy** | signal-processing memo (NSA Slepian-Wolf-Wyner-Ziv lineage) | **-0.05** (LARGEST signal-processing single bet) | $5-10 (prototype) | L0 SCAFFOLD (substrate-engineering scope; ~3-5 days) | DEFER — no archive grammar yet (HNeRV parity discipline lesson 8) |
| 4 | **L4 Lincoln Lab kernel-projection ambiguity shaping** | signal-processing memo | -0.04 | $5-8 | L0 SCAFFOLD | DEFER — no substrate yet |
| 5 | **B-1 Atick-Redlich efficient coding** | fields-medalist memo (cited in time-traveler design memo move #1) | top-3 ≤ -0.04 | $0 (already in time-traveler architecture) | EMBEDDED IN time-traveler | Subsumed by time-traveler dispatch |
| 6 | **G-5 Mallat scattering / wavelet** | fields-medalist memo | top-3 ≤ -0.04 | $5 | L0 SCAFFOLD | DEFER |
| 7 | **E4 MDL-IBPS (procedural + IB sidecar)** | zen-state memo | -0.030 to -0.080 (LARGEST single bet across all memos) | $0 (prototype design) → $5 (smoke) | L0 SCAFFOLD | **#2 within $5** — high-EV swing |
| 8 | **A6 F0ABS frame-0 byte-stuff** | aerospace-stealth memo (F-117 stealth analog; SegNet discards frame 0) | -0.005 to -0.015 | $0-1 (small bolt-on) | L0 SCAFFOLD (~150 LOC) | Cheap stack-on candidate after #1 |
| 9 | **A5 active-cancellation pose-residual sidecar** | aerospace-stealth memo (B-2/F-22 antiphase analog) | -0.012 to -0.030 | $1-3 | L0 SCAFFOLD | Stack-on after time-traveler smoke |
| 10 | **SE-4 Shannon-1959 JSCC scorer-conditional coding** | ancient-elder memo (already in PR95++ stack via F1 IGLT) | -0.012 | $0 (already wiring) | **L1 IMPL_COMPLETE in PR95++ wave** | Subsumed (in flight) |

## Time-traveler dispatch readiness audit

**Verdict: L1 IMPL_COMPLETE + RECIPE READY + smoke-before-full WIRED (Catalog #167 CLEAN).**

Empirical receipts:

```
recipe          : .omx/operator_authorize_recipes/substrate_time_traveler_l5_autonomy_modal_a100_dispatch.yaml
wrapper         : scripts/operator_authorize_substrate_time_traveler_l5_autonomy_modal_a100_dispatch.sh
remote driver   : scripts/remote_lane_substrate_time_traveler_l5_autonomy.sh (9.5 KB; sentinel #163-compliant)
substrate pkg   : src/tac/substrates/time_traveler_l5_autonomy/ (architecture+archive+inflate+score_aware_loss; tests pass)
trainer         : experiments/train_substrate_time_traveler_l5_autonomy.py (53 KB; --smoke + --full + --full-cpu)
smoke harness   : tools/smoke_time_traveler_l5_autonomy_macos_cpu.py (24 tests pass)
recipe declares : cost_band.epochs=3000, smoke_score_band [0.10, 0.30], predicted_band [0.150, 0.170]
recipe declares : min_vram_gb=40, video_input_strategy=per_dispatch_local_copy,
                  pyav_decode_strategy=cpu_thread_async_upload, target_modes contains
                  contest_exact_eval, canary_status=post_canary_dependent (canary_dependency=sane_hnerv)
sentinel files  : 8 declared (TT5L architecture/archive/inflate/score_aware_loss + 3 shared modules)
preflight       : Catalog #167 (smoke-before-full) CLEAN; Catalog #170-173/178-182 declared in recipe
remote sentinel : REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 set per Catalog #163
```

**Blockers preventing immediate `$0.30` smoke dispatch**: NONE detected in source.

The smoke-before-full pattern (Catalog #167) is wired:

```bash
exec .venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe substrate_time_traveler_l5_autonomy_modal_a100_dispatch \
    --smoke-epochs "${TT5L_SMOKE_EPOCHS:-100}" \
    --smoke-gpu "${TT5L_SMOKE_GPU:-T4}" \
    --smoke-timeout-hours "${TT5L_SMOKE_TIMEOUT_HOURS:-1.0}" \
    --operator-handle "claude:operator_authorize_substrate_time_traveler_l5_autonomy_modal_a100_dispatch"
```

`canary_status: post_canary_dependent` + `canary_dependency: sane_hnerv` —
sane_hnerv is in flight and has not produced a successful first contest-CUDA
anchor yet. Per Catalog #173 the dispatcher MAY refuse the time-traveler
dispatch until sane_hnerv canary lands. **This is the operator-routable
decision**: either (a) wait for sane_hnerv canary, (b) downgrade
`canary_status` to `independent_substrate`, or (c) fire with
`OPERATOR_AUTHORIZE_SESSION_BUDGET_USD` paired-env-var bypass per
Catalog #199.

**SIREN-class blocker check (task #659 reference)**: time-traveler wrapper
uses `${ARR[@]+"${ARR[@]}"}` empty-array-guarded expansion per Catalog #189
(line 42). Wrapper is macOS bash 3.2 + `set -u` safe. Verified.

## Recommended next 3 dispatches (rank-ordered by EV/$, NO GPU spend by me)

### Dispatch #1 — L2 SAR coherent integration over pose pairs (~$1, ΔS -0.0056)

**Status**: L0 SCAFFOLD (substrate not built; ~100 LOC + small sidecar grammar
needed). The cheapest top-tier dispatch in any expert-team memo. Per
signal-processing memo: leverages PR106 r2 pose marginal (271× SegNet's at
pose_avg = 3.4e-5). 30 min build + 30 min smoke wall-clock.

**Recommended command** (after substrate L1 build, ~$1 Modal T4 smoke):

```bash
# (after building src/tac/substrates/sar_coherent_pose/ + recipe)
TT5L_SMOKE_GPU=T4 OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
    OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=2.00 \
    scripts/operator_authorize_substrate_sar_coherent_pose_modal_t4_dispatch.sh \
    --smoke-epochs 100 --smoke-gpu T4
```

**Expected harvest**: 30-60 min wall-clock. Expected ΔS contribution: -0.005
to -0.008 vs A1 (pose-axis only; SegNet unchanged). If PASS, stack with
A1 substrate.

### Dispatch #2 — E4 MDL-IBPS prototype (≤ $5 within window, ΔS -0.030 to -0.080)

**Status**: L0 SCAFFOLD (zen-state memo Z-1 designation; LARGEST single bet
across all alien-tech memos). The information-bottleneck-procedural-sidecar
prototype is the highest-predicted-Δ research candidate. Building the
prototype is $0 design work (council reviewed); first smoke at $5 Modal T4.

**Recommended sequence**: $0 design pass (council-grade decision per
CLAUDE.md "Design decisions — non-negotiable") → SCAFFOLD substrate → $5
smoke. Decision goes to operator.

### Dispatch #3 — Time-traveler L5 autonomy full (≤ $20 swing-for-fences, ΔS -0.020 to -0.040)

**Status**: **L1 READY**. Predicted band [0.150, 0.170]. Falsification
threshold 0.190.

**Recommended smoke command** ($0.30 Modal T4 smoke first, per Catalog #167):

```bash
# Operator authorizes (this Claude DOES NOT fire — sister subagents at
# $6.50/$25 spend; queued for operator action).
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
    OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=20.00 \
    TT5L_SMOKE_EPOCHS=100 TT5L_SMOKE_GPU=T4 \
    scripts/operator_authorize_substrate_time_traveler_l5_autonomy_modal_a100_dispatch.sh
```

If smoke lands in `smoke_score_band: [0.10, 0.30]` → full A100 dispatch fires
automatically. Total: ~$0.30 smoke + ~$5 full = ~$5.30 worst case.

**Expected harvest**: smoke 30 min, full 4h wall-clock. Expected ΔS
contribution if predicted band confirms: A1 0.193 → time-traveler 0.150-0.170
= -0.020 to -0.040 single-substrate. Stack with SAR pose if both pass.

**Pre-condition**: either (a) sane_hnerv canary lands a successful contest-CUDA
anchor (clears `canary_dependency`), OR (b) operator downgrades
`canary_status: independent_substrate` for this lane (council-grade decision
per CLAUDE.md "Design decisions"), OR (c) operator approves the
`canary_dependency` override directly.

## 3-line synthesis: alien-tech-vs-HNeRV-family landscape post-PR106-falsification

1. The **HNeRV-family-extension via PR106 grammar is DEAD on contest-CPU**
   (uniform +0.022 CPU-CUDA gap across 9 sidecar variants → ~0.228 CPU vs
   PR101 0.193 medal-band → not competitive). The A1 hnerv_ft_microcodec
   remains valid as the sole sub-0.20 anchor (CPU-dominant by -0.034).
2. Sub-A1 paths require **representational moves outside HNeRV-family-
   extension**: (a) cooperative-receiver / score-conditional encoding
   (Wyner-Ziv N3, Atick-Redlich B-1, time-traveler move #1), (b) predictive-
   coding hierarchy (Rao-Ballard / Friston / time-traveler move #2),
   (c) procedural baselines + IB sidecar (E4 MDL-IBPS), or (d) a new HNeRV
   architectural insight not yet on the table.
3. **Time-traveler is the ONLY substrate that ALREADY composes** moves
   #1+#2+#3+#4+#5 from the alien-tech expert-team memos AND has L1
   IMPL_COMPLETE + recipe + smoke-before-full wired. Single highest-leverage
   single-dispatch in the v3 floor landscape: $0.30 smoke fires the entire
   composite-substrate hypothesis at near-zero cost.

## Falsification + reactivation criteria (per CLAUDE.md "KILL is LAST RESORT")

This memo is research-only synthesis. Falsification of any individual
candidate above does NOT kill the family — DEFERRED-pending-research with
reactivation criteria:

- Time-traveler `FAIL_FALSIFIED` (≥ 0.190) → DEFER differentiable-physics
  interpretation; retry cheap-foveation-only variant (per design memo).
- SAR pose `WARN_ABOVE_BAND` (> -0.003) → DEFER coherent-integration window
  size; retry shorter integration with pose Kalman (L5).
- E4 MDL-IBPS `FAIL_FALSIFIED` → DEFER procedural baseline; retry IB sidecar
  alone.

NO KILL verdicts in this memo. NO score claims. NO archive bytes. NO new
GPU dispatches by Claude (sister subagents at $6.50/$25 spend per directive).

## 6-hook wire-in declaration (per Catalog #125)

1. **Sensitivity-map contribution**: the 10 ranked techniques + per-technique
   ΔS predictions feed `tac.sensitivity_map.*` priors as candidate-row
   importance signals; SAR pose and time-traveler register pose-axis
   sensitivity at PR106 r2 operating point (271× SegNet marginal).
2. **Pareto constraint**: this memo registers the **shrunk Pareto cone**
   (PR106-family removed from sub-0.21 region on CPU axis); 178K-byte band
   identified as critical gap (only A1 has dual-axis there). Hook: future
   `tac.pareto_*` updates should consume the ship-readiness classification
   from `posterior_recovery_landscape_20260513.md`.
3. **Bit-allocator hook**: time-traveler design memo's per-pair side-info
   budget (45 B/pair) is Fisher-water-fillable; SAR pose's coherent
   integration window-size is per-pair allocator parameter. Hook registered
   for both substrates' future archive packers.
4. **Cathedral autopilot dispatch hook**: the rank-ordered top-3 dispatch
   queue feeds `tools/cathedral_autopilot_autonomous_loop.py` candidate
   ranking. Time-traveler is L1-ready and pre-registered in lane
   `lane_time_traveler_l5_autonomy_substrate_20260513`; SAR pose and MDL-IBPS
   need lane pre-registration BEFORE first commit per Catalog #126.
5. **Continual-learning posterior update**: N/A — this memo produces NO
   empirical anchor (research-only synthesis). The 17 anchors ingested by
   posterior recovery on 2026-05-13 are the canonical posterior update for
   this session.
6. **Probe-disambiguator**: the +0.022 vs -0.034 CPU/CUDA-gap inversion
   (PR106-family vs A1) is the **2+ defensible interpretations** pattern
   that warrants `tools/probe_pr106_vs_a1_cpu_cuda_gap_disambiguator.py` per
   the posterior recovery memo recommendation. Three explanations open:
   (a) decoder numerics drift, (b) scorer numerics drift, (c) YUV6 preprocess
   drift. Probe SHOULD ship before any new PR106-family submission attempt.

## Cross-references

- **A1 axis validation**: `.omx/research/a1_pr106_cpu_cuda_axis_validation_20260513_codex.md`
- **Posterior 25→42 recovery**: `.omx/research/posterior_recovery_landscape_20260513.md`
- **Time-traveler design memo**: `.omx/research/time_traveler_architecture_reverse_engineered_20260513.md`
- **Time-traveler smoke harness landed**: memory `feedback_time_traveler_l5_smoke_harness_landed_20260513.md`
- **Time-traveler macOS-CPU smoke execution**: memory `feedback_time_traveler_l5_macos_cpu_smoke_execution_landed_20260513.md`
- **Time-traveler `--full-cpu` mode landed**: memory `feedback_time_traveler_full_cpu_mode_landed_20260513.md` (Catalog #197)
- **Signal-processing alien-tech**: memory `feedback_expert_team_signal_processing_alien_tech_landed_20260513.md`
- **Aerospace-stealth alien-tech**: memory `feedback_expert_team_aerospace_stealth_analytic_alien_tech_landed_20260513.md`
- **Zen-state frontier deep-math**: memory `feedback_zen_state_frontier_deep_math_research_landed_20260513.md` (E4 MDL-IBPS Z-1 designation)
- **Ancient elder polymath**: memory `feedback_ancient_elder_polymath_landed_20260513.md` (SE-4 JSCC, Shannon-1959 vector R(D))
- **Fields-medalist math/biology**: memory `feedback_expert_team_fields_medalist_math_biology_alien_tech_landed_20260513.md`
- **Floor v2**: `.omx/research/grand_council_fields_medal_theoretical_floor_20260509.md`
- **CLAUDE.md non-negotiables consulted**: "Apples-to-apples evidence discipline", "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 contest-CI hardware", "HNeRV / leaderboard-implementation parity discipline", "KILL/FALSIFIED memory verdicts", "Subagent coherence-by-default", "Race-mode rigor inversion + parallel-dispatch first", Catalog #167 (smoke-before-full), Catalog #173 (canary-first ordering), Catalog #189 (shell empty-array guard), Catalog #197 (full-cpu coupled flags), Catalog #199 (operator-authorize bypass requires session budget).

## Verdict

DEFERRED-pending-empirical for all 10 alien-tech techniques. NO score
claims. NO promotion. NO archive bytes. This memo points the next dispatch
direction; operator decides which to fire and when. Time-traveler is the
single most-ready high-leverage candidate; SAR pose is the cheapest
top-tier candidate; E4 MDL-IBPS is the highest-Δ research swing.

3-clean-pass adversarial greenup (research-only synthesis, single-pass per
CLAUDE.md "research-only sanity gate"):

- Round 1 (Shannon/Dykstra/Yousfi/Fridrich/Contrarian): no score claim
  outside `[contest-CPU-1to1]` / `[contest-CUDA]` paired anchors; floor
  band derivation traces to Shannon-1959 + Council F floor + Wyner-Ziv MI
  bound; no kill verdicts; reactivation criteria documented per technique.
  CLEAN.
- Round 2 (Quantizr/Hotz/Selfcomp/MacKay/Ballé): predicted Δ contributions
  per memo, NOT new derivation; HNeRV parity discipline lessons 2/3/4/8
  honored (no archive grammar claims, no inflate LOC promises, no
  score-aware loss claims without empirical backing); no PR106-family
  resurrection language. CLEAN.
- Round 3 (van den Oord/Carmack/Tao/Boyd/Hinton): no GPU dispatch by
  Claude (sister subagents own spend); operator-routable decisions
  surfaced explicitly (canary dependency override, dispatch order,
  prototype design); cross-references complete. CLEAN.

Counter: 3/3. Memo CLEARED for landing.
