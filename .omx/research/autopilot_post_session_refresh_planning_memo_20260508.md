# Autopilot Post-Session Refresh Planning Memo — 2026-05-08

Subagent: AUTOPILOT-REFRESH (claude)
Operator state: PR106 frontier (d_seg=6.7e-4, d_pose=3.4e-5, B=178,873 → score 0.20454)
Target: 0.155 (gap +0.04954)

## Refresh provenance

- Pre-refresh ledger: 32 rows (last refresh 2026-05-07 by codex-built `tools/cathedral_autopilot.py`)
- Post-refresh ledger: **41 rows** (`reports/cathedral_autopilot_evidence.jsonl`)
- 9 net-new rows added during 2026-05-07/08 sessions: ADMM step 6 byte-closed; ADMM no-dead-K; cross-paradigm 137,469 B (substrate-corrected); cross-paradigm 137,531 B (phantom kept as forensic record); 3× phase4 orchestrator rows; lossy_coarsening exact-CUDA 0.351719 [contest-CUDA A-negative]; multipass_imp 131,967 B
- Plan output: `reports/autopilot_plan_post_session_refresh_20260508.json`
- Meta-Lagrangian bridge: `reports/cathedral_meta_lagrangian_ranking_post_session_refresh_20260508.json`
- Note: UNIWARD-weighted Lagrangian frontier + Filler STC ternary mask deltas mentioned in mandate are NOT yet in the JSONL. They appear to be in flight from sister subagents (IMPL-UNIWARD-STC-V2 not landed by my snapshot). Re-run autopilot when those rows land.

## Top-3 score-delta recommendations [predicted-band, NOT contest-CUDA]

| Rank | Technique | Predicted Δscore | Bytes | Cost | Grade |
|---:|---|---:|---:|---:|---|
| 1 | self_compress_neural_codec (Selfcomp/Quantizr arch) | +0.05918 | 90,000 | $20.0 / 18h | [predicted] |
| 2 | brotli_optuna_default (q=11/lgwin=16/lgblock=19) | +0.00049 | 178,144 | $0.0 / 0.5h | [contest-CUDA, already landed] |
| — | (no #3 above the +0.001 min-score-delta filter) | | | | |

Score-delta ranking is THIN. The autopilot's score-claim filter (per CLAUDE.md `forbidden_CPU_MPS_derived_dispatch_readiness_flag`) correctly excludes proxy_row=True candidates from the active recommendation set. This is the source of asymmetry between "rank-by-bytes-only" lists below and the formal score recommendation.

## Top-5 byte-anchor frontier [CPU-prep faithful]

These are the lowest-byte deployable anchors in the ledger. None are score-claims; all need exact CUDA auth eval before any promotion.

| Rank | Technique | Bytes | Δ vs 178,873 | Notes |
|---:|---|---:|---:|---|
| 1 | omega_opt_linear_stack_post_hoc_composition | **41,303** | -137,570 | post-hoc composition; dispatch_blockers include `no_runtime_decoder_packet_built` + `score_impact_unknown_requires_full_retrain_plus_cuda` |
| 2 | arch_shrink_x0.4_quantizr_class | 83,571 | -95,302 | DEFERRED-pending-research per CLAUDE.md; previously dispatched to Lightning T4 |
| 3 | sparsity_alpha_0.7_imp_retrain | 94,671 | -84,202 | requires retrain; planning_signal |
| 4 | lossy_int4_quantization_awq | 100,714 | -78,159 | audit-criterion-5 test, `[CPU-prep faithful]` |
| 5 | lossy_int4_quantization (naive PTQ) | 100,799 | -78,074 | DEFERRED-pending-research; rel_err 1.55% basin-parity-FAIL at apogee_int4 |

Honorable mention (rows 6-10): lossy_int4_quantization_gptq (100,844), lossy_int4_quantization_qat (115,641), lossy_int4_per_channel_scales (115,958), joint_admm_lagrangian_allocation (130,032), multipass_imp_post_hoc_composition (131,967), cross_paradigm_admm_continuous_k_then_op1_finalizer_substrate_corrected (137,469).

## Top-5 Lagrangian-ranked (meta_lagrangian_bridge)

| Rank | Technique | L | Sanity | Predicted band |
|---:|---|---:|---|---|
| 1 | self_compress_neural_codec | 0.1462 | FAIL | [0.0712, 0.2212] |
| 2 | brotli_optuna_default | 0.2049 | FAIL | [0.1299, 0.2799] |

Bridge produced only 2 candidates (the two that survive the score-claim-eligible filter). 0 candidates are eligible_for_dispatch. This is correct CLAUDE.md behavior: every CPU-derived signal stays `ready_for_exact_eval_dispatch=False`. The sanity FAIL flag means the predicted score band is too wide for confident dispatch — the `[0.0712, 0.2212]` lower-bound CI for self_compress reflects how much architecture-class change inflates uncertainty.

## Strategic dispatch ordering (next operator-authorized GPU spend)

Per CLAUDE.md `forbidden_premature_class_level_falsification` and operator exclusion of Q-FAITHFUL retrain, recommended order is:

**Tier A — within-compute-budget bolt-ons (no retrain):**
1. **brotli_optuna_default verify** — already [contest-CUDA] tagged; if archive bytes have drifted since last verification, re-anchor on current PR106. Predicted Δ +0.00049, $0.
2. **lossy_coarsening reactivation criteria** — exact-CUDA ledger row at 0.351719 retired this measured config. Reactivation needs scorer-aware loss + byte-closed runtime packet (per row `reactivation_criteria`). DO NOT re-dispatch the same config.
3. **cross_paradigm_admm_continuous_k_then_op1_finalizer_substrate_corrected (137,469 B)** — the substrate-corrected variant supersedes the 137,531 phantom; needs WIRE-DECODER subagent's wire-format parity check before a dispatch.

**Tier B — architecture-touching, retrain-required (each is multi-day GPU):**
4. **arch_shrink_x0.4** — Lightning T4 dispatch already in flight (per MEMORY.md `project_arch_shrink_x0_4_lightning_DISPATCHED_20260508`); harvest before queueing further architecture work. Estimated harvest window 12-18h, ~$9.90 sunk.
5. **lossy_int4_quantization_qat** — only QAT row in ledger ([MPS-research-signal] grade, 115,641 B, rel_err 1.67% > 5% deployable threshold blocks dispatch). Reactivation requires LSQ + per-channel scales + outlier handling per `forbidden_premature_kill_without_research_exhaustion`.
6. **self_compress_neural_codec** — highest predicted Δ +0.05918 but predicted-band [0.0712, 0.2212]; the wide CI alone says treat as STUDY before any dispatch. Engages Selfcomp council seat directly.

**Tier C — hold for evidence:**
- omega_opt_linear_stack at 41,303 B — too good to be deployable without retrain and runtime packet; fascinating planning signal but `score_impact_unknown_requires_full_retrain_plus_cuda` blocker is the right gate.

## Risk assessment per candidate

| Candidate | Risk class | Specific risks |
|---|---|---|
| brotli_optuna_default | LOW | param drift only; already [contest-CUDA] |
| lossy_coarsening reactivation | MEDIUM | already retired one config at 0.352; scorer-aware loss is non-trivial integration |
| cross_paradigm 137,469 B | MEDIUM | wire-format parity hasn't been independently verified; substrate-correction is recent |
| arch_shrink_x0.4 | MEDIUM-HIGH | Lightning T4 dispatch in flight; outcome will reframe ranking |
| lossy_int4 QAT | HIGH | rel_err 1.67% currently FAILs deployable threshold; basin-parity FAILED at int4 (CLAUDE.md anti-pattern) |
| self_compress_neural_codec | HIGH | architecture-class change; predicted-band CI too wide; 18h+$20 cost; Selfcomp's own 0.38 archive used arch_shrink + block-FP, not from-scratch self-compress |
| omega_opt_linear_stack | UNKNOWN | post-hoc composition with no retrain; bytes too low to be real on PR106 substrate |

## Cost estimates ($)

| Tier | Min $ | Max $ | Wall-clock |
|---|---:|---:|---|
| A (verify + cross-paradigm CUDA replay) | $0.50 | $5 | 0.5-2h |
| B (arch_shrink harvest in flight) | $9.90 sunk | $9.90 | 12-18h |
| B (lossy_int4 QAT proper retrain) | $15 | $30 | 12-24h |
| B (self_compress_neural_codec) | $20 | $40 | 18-36h |

## CLAUDE.md compliance footer

- All score predictions tagged `[predicted-band, NOT contest-CUDA]`
- All byte anchors tagged `[CPU-prep faithful]`
- No `ready_for_exact_eval_dispatch=True` set anywhere; all bridge candidates remain non-dispatchable per `forbidden_CPU_MPS_derived_dispatch_readiness_flag`
- No KILL verdicts emitted; lossy_coarsening retired only the measured config (`falsification_scope: measured_config_only`)
- No new dispatches recommended without operator approval
- Q-FAITHFUL retrain explicitly excluded from recommendations per operator
- Memo location `.omx/research/` (private) per Strategic Secrecy Rule

## Cross-references

- `tools/cathedral_autopilot.py` (codex-built recommender)
- `tools/cathedral_autopilot_meta_lagrangian_bridge.py` (B1 bridge)
- `reports/cathedral_autopilot_evidence.jsonl` (41 rows)
- `reports/autopilot_plan_post_session_refresh_20260508.json`
- `reports/cathedral_meta_lagrangian_ranking_post_session_refresh_20260508.json`
- MEMORY.md: `project_arch_shrink_x0_4_lightning_DISPATCHED_20260508`, `feedback_grand_council_reactivation_dispatch_order_20260507`, `feedback_adversarial_audit_4_falsifications_DEFERRED_not_killed_20260507`

## Notes on UNIWARD-STC-V2 / Filler STC mandate items

The mandate referenced ADMM step 6 byte-closed (FIX-ALL-FINDINGS), cross-paradigm 137,469 B corrected (FIX-CODEX-FINDINGS, **landed**), 153,513 B deployable (WIRE-DECODER), UNIWARD-weighted Lagrangian frontier (5 rms_targets), and Filler STC ternary mask deltas. Of these:

- ADMM step 6 byte-closed: row at 130,032 / 131,967 / 153,639 (Path-B-step6 series) IS in ledger.
- Cross-paradigm 137,469 B: IS in ledger (substrate-corrected, supersedes 137,531 phantom).
- Cross-paradigm 153,513 B: NOT in ledger by my snapshot — possibly WIRE-DECODER subagent landing in parallel.
- UNIWARD-weighted Lagrangian frontier (5 rms_targets): NOT in ledger.
- Filler STC ternary mask deltas: NOT in ledger.

Re-run autopilot when IMPL-UNIWARD-STC-V2 + WIRE-DECODER land their rows.
