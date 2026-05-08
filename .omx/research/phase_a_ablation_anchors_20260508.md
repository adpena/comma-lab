---
title: Phase A Ablation Anchors — Track 1 Cross-Reference Table
date: 2026-05-08
author: Subagent SYNTHESIZE (claude-opus-4-7-1m)
status: synthesis — empirical anchors from CPU-prep work and dispatch attempts; GPU [contest-CUDA] anchors all pending
related_tasks: ["#307 PARADIGM-δεζ Phase A", "#308 PHASE 4 INTEGRATION"]
council_memo_ref: ".omx/research/grand_council_extreme_rigor_track_1_20260508.md"
score_claim: false
ready_for_exact_eval_dispatch: false
---

# Phase A Ablation Anchors — Track 1 Cross-Reference Table

## Purpose

Single-table snapshot of the seven Phase A decision anchors (A0–A6) per the
council's verdict in `.omx/research/grand_council_extreme_rigor_track_1_20260508.md`.
Each row records the strongest currently-landed empirical anchor (best byte
result OR best dispatch outcome), the rigor classification, and the blocker
preventing escalation to a [contest-CUDA] anchor.

No row in this table is a score claim. Every numeric is `[byte-anchor]`,
`[CPU-prep]`, `[predicted]`, or `[infrastructure-blocked]`.

## 1. Cross-ablation summary table

| Phase | Decision | Best Byte Anchor | Rel-Err | Dispatch Status | Evidence Grade | Blocker / Notes |
|---|---|---:|---:|---|---|---|
| **A0** | MDL closed-form floor (PR101 weight stream) | total_lower_bound 151,700 B; realistic 158,700 B | 0% (analytical bound) | local CPU; no dispatch needed | `byte_proxy_only_deterministic_closed_form` | `synthetic_pr101_proxy_no_path` weights — joint-floor 148–162 KB validated by closed form, but per-tensor breakdown not measured against real PR101 state-dict |
| **A1** | Score-gradient supervision (PR101 fine-tune) | — | — | `failed_external_blockers` | `[infrastructure_blocked]` | Lightning Studio: nvidia-smi missing (no GPU attached); Vast.ai: account out of credit. Tooling complete (training script + dispatcher + remote lane script + smoke passed); GPU access is the gate. |
| **A2** | Sensitivity-aware per-tensor quantization | uniform 157,611 B; weighted 159,544 B (+1,933) | 3.36% (weighted) / 3.55% (uniform) | local CPU only | `CPU-local allocator proxy` | Sensitivity map is `[stub-design-mode]` (`is_stub: true`); allocator weights derived from beta-Fisher proxy on a stub map — must replace with certified PR101 component-sensitivity map before ranking eligible. |
| **A3-alt** | Mallat wavelet importance (db4 level=2) | 176,624 B at η=0.5 (-1,520 vs 178,144 baseline) | 0.37% | local CPU | `[byte-anchor; sensitivity_proxy=mallat_wavelet]` | Wavelet-coefficient importance proxy; saves 1.5 KB at <0.5% rel_err. byte-proxy only; `cuda_eval_worth_testing=false` per row tags. Sister tool: `tools/pr101_sensitivity_aware_mallat_wavelet.py`. |
| **A4** | ChARM 50K toy substrate (Ballé channel-AR prior) | encoded 7,031 B / 46,995 model params (-2.5KB vs uniform); rate term finite, roundtrip exact | smoke_passed | local CPU smoke | `byte-anchor toy substrate; smoke only` | Toy substrate (50K params, NOT PR101's 228,958 elements). Roundtrip-exact + finite-rate is the gate; substrate-scale validation requires GPU dispatch on real PR101 weights. Per Quantizr+Carmack: Decision 1 LOW priority vs Decisions 2/3. |
| **A4-alt** | Cross-paradigm ADMM × Op1 finalizer (real PR101) | 137,531 B (orch_smoke) / 153,513 B (per-archive zip) | 4.15% | byte-closed candidate; dispatch_blockers={byte_proxy_only, no_real_archive_substrate_for_cuda_replay, missing_exact_cuda_auth_eval} | `[CPU-build]` | Best CPU-built archive in the Phase-A family. Distinct evidence than A4 (real substrate, not toy). `cuda_eval_worth_testing: true`. SHA: `7bbba307...28489489`. |
| **A5** | Frame-conditional bit budget | archive Δ -1,278 B at η=2.0 (latent Δ -1,503 B, sidechannel +225 B) | byte proxy only | local CPU | `[CPU-prep faithful frame-conditional byte anchor]` | Tooling landed in `tools/pr101_frame_conditional_bit_anchor.py` + `tac.codec.frame_conditional_bit_budget`; per-pair score marginal and inflate schema update are blockers before dispatch. |
| **A6** | Block-FP × hyperprior (Selfcomp/Ballé hybrid) | 214,035 B | 0% on int8 proxy stream | local CPU proxy; not dispatch-worthy | `[byte-roundtrip proxy; measured-config negative]` | Current max-abs-scale conditional Gaussian range-coder loses to PR101 brotli by +35,891 B. Not a score claim and not a family kill; real Selfcomp per-channel block-FP, learned ChARM, tensor-aware PMFs, and byte-map-preserving arithmetic rewrites remain open. |

### A2 packet-ladder (A2 byte-closure in inflate-replayable form)

| Variant | Bytes | Reference | Notes |
|---|---:|---|---|
| A2 packet ladder reference | 162,164 (decoder blob) | `track1_phase_a2_packet_ladder_codex_hardened_20260508T161558Z` | Reproduces source decoder blob byte-for-byte; `state_dict_reproduces_source_decoder.passed: true`. Inflate-parity not yet run; clearance gate B B2 (Catalog #115) tracks the blocker. |

## 2. Per-anchor evidence pointers

| Phase | Manifest path | Evidence file |
|---|---|---|
| A0 | `experiments/results/track1_phase_a0_mdl_20260508T154125Z/build_manifest.json` | `A0_result.json` (228,958 elements / 28 tensors / charm_2020 hyperprior config / 175,916 iid floor / 148K–162K joint floor) |
| A1 | `experiments/results/track1_phase_a1_score_gradient_20260508T184355Z/build_manifest.json` | `dispatch_status: failed_external_blockers`; tool outputs landed in `tools/dispatch_phase_a1_score_gradient_pr101.py`, `experiments/train_score_gradient_pr101_finetune.py`, `tools/build_pr101_finetuned_archive.py`, `scripts/remote_track1_phase_a1_score_gradient_pr101.sh` |
| A2 | `experiments/results/track1_phase_a2_sensitivity_quant_20260508T154125Z/build_manifest.json` | `A2_result.json` (uniform 157,611 / weighted 159,544 / 28 tensor breakdowns / sensitivity map stub); packet ladder at `track1_phase_a2_packet_ladder_codex_hardened_20260508T161558Z/a2_packet_ladder_manifest.json` |
| A3-alt | `experiments/results/pr101_sensitivity_aware_mallat_wavelet_codex_20260508Tlocal/build_manifest.json` | rows[] with η ∈ {0.0, 0.5, 1.0} sweep at average_budget=0.02 |
| A4 | `experiments/results/track1_a4_charm_50k_toy_20260508T183937Z/smoke/smoke_log.json` | rate_finite=true, roundtrip_exact=true, encoded 7,031 B / 46,995 params; toy substrate, NOT PR101 |
| A4-alt | `experiments/results/cross_paradigm_admm_x_op1_finalizer_20260508T062603Z/build_manifest.json` | archive 153,513 B sha 7bbba307; predicted_band [0.18, 0.22] grade=`predicted`; orch_smoke variant 137,531 B sha ea3b23ed |
| A5 | `experiments/results/pr101_frame_conditional_bit_codex_20260508Tlocal/build_manifest.json` | best eta=2.0; archive_delta_bytes=-1,278; score_claim=false; dispatch_blocker=`awaiting_per_frame_score_marginal` |
| A6 | `experiments/results/pr101_a6_blockfp_hyperprior_codex_20260508Tlocal/build_manifest.json` | best B=64,uint8 is 214,035 B (+35,891 B vs PR101 brotli); current proxy config measured-negative |

## 3. Council decision verdicts (recap)

From `.omx/research/grand_council_extreme_rigor_track_1_20260508.md`:

- **Decision 1 (A0+A4):** Bayesian-MDL anchored ≤ 165 KB on PR101; A4 ChARM toy must show ≥ 5 KB savings before promoting Decision 1 into Phase C.
  Status: **A0 GREEN** (158,700 B realistic ≤ 165 KB target); **A4 toy substrate roundtrip-exact** but real-PR101 smoke not yet run.
- **Decision 2 (A1):** UNANIMOUS HIGHEST PRIORITY. Predicted gain -10% to -30% on seg, -10% to -25% on pose. Score impact -0.005 to -0.015.
  Status: **infrastructure-blocked** — operator has not attached GPU to Lightning Studio nor refilled Vast.ai.
- **Decision 3 (A2):** HIGHEST EV-PER-$. Predicted byte savings 8–15 KB.
  Status: **A2 weighted not yet beating uniform** (-1,933 B regression at this rms target). Sensitivity map is a stub; certified-map dispatch is the open gate.
- **Decision 4 (pose-deriver):** LOW priority; superseded by existing `lane_pd_v2`.
- **Decision 5 (A5 frame budget):** ENDORSE with runtime-cost gate; local byte proxy landed at net -1,278 B, but score-marginal and runtime-schema gates remain open.

## 4. Promotion gate readiness (Phase C trigger)

Per council G5/G8/G9/G10, Phase C dispatch needs ≥ 4 of 5 GREEN:

| Gate | Description | Current |
|---|---|---|
| G5 | Decision 1 co-design anchor (Phase A4) | YELLOW — toy roundtrip exact, real-PR101 not run |
| G8 | Phase A0 (MDL) ≤ 165 KB | **GREEN** — 158,700 B realistic |
| G9 | Phase A1 ≥ 10% on seg or pose | RED — infrastructure-blocked, no measurement |
| G10 | Phase A4 ≥ 5 KB on co-design ablation | YELLOW — toy substrate only |

Current GREEN count: **1 of 4** (A0 only). Phase C dispatch BLOCKED. The
critical path is Decision 2 (A1) — it is unanimously highest-EV and the only
Phase A whose blocker is purely infrastructure (operator GPU access).

## 5. Cross-references

- `.omx/research/grand_council_extreme_rigor_track_1_20260508.md` — council deliberation, gates, dispatch order
- `.omx/research/track_1_co_designed_substrate_design_20260508_claude.md` — Track 1 substrate scope
- `.omx/research/phase4_optimal_stack_design_20260508_claude.md` — Phase 4 stack composition design
- `experiments/results/phase_a_dispatch_rollup_20260508T154209Z.json` — operator-side rollup of Phase A0+A2 launches
- Cathedral autopilot recommender (`tools/cathedral_autopilot.py`) — feedback-loop consumer of these anchors

## 6. Reactivation criteria

Per CLAUDE.md "KILL is LAST RESORT": no row in this table is killed. Each
deferred row records the precise empirical evidence that would unblock it:

- **A1**: operator attaches T4 to Lightning Studio OR refills Vast.ai → dispatch fires, harvested archive + adjudicated.json land.
- **A2**: certified PR101 component-sensitivity map (non-stub) replaces the diagnostic; weighted-vs-uniform delta becomes byte-positive at the target rms.
- **A3-alt**: byte-anchor savings hold at score level via [contest-CUDA] eval on the wavelet-importance archive.
- **A4 (real)**: ChARM trained on real PR101 INT8 stream achieves ≥ 5 KB savings vs brotli-q11 baseline at <2% rel_err.
- **A4-alt**: real-archive CUDA dispatch on the cross_paradigm_admm_x_op1_finalizer artifact (153,513 B sha 7bbba307) confirms predicted band [0.18, 0.22] holds at score level.
- **A5**: per-pair score-marginal evidence lands, the frame-budget side-info path is wired into inflate, and inflate.sh runs ≤ 30 min on T4.
- **A6**: A1+A2 succeed; block-FP × hyperprior re-prioritized.
