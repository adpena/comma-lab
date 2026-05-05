---
name: 2026-04-30 session state checkpoint — Phase 1/1.5/2/3 status + active dispatches + open subagents
description: Comprehensive snapshot of where the PACT project stands as of late 2026-04-30. Captures: 6 Vast.ai/Modal experiments dispatching overnight, 5+ BG dev subagents in flight, 88 STRICT preflight checks, every Phase 1/1.5/2 lane's actual code state vs Council E reprioritization, what still needs implementation, theoretical floor analysis vs current standings.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## TL;DR — where we are right now

- **Confirmed [contest-CUDA] anchor**: Lane G v3 = 1.05 (the ONLY trustworthy score we have)
- **Theoretical floor estimates** (council range): senior-eng 0.245 / Council #271 next-48h forecast 0.93-0.97 / contest target sub-0.30 / Quantizr leader 0.33
- **9 SegMapTrainer-using lanes** were silently invalidated for weeks by `.round()` zero-grad bug; fixed today via Council A + Council C bf16+scorer-chunk
- **Today's empirical wins**: Lane Ω-W-V2 = 40.98% byte savings [empirical] on real Lane G v3 renderer.bin; Lane Joint-ADMM converges + budget feasibility on 4-stream non-convex
- **3-clean-pass adversarial gate**: Round 8 + 9 + 10 all CLEAN → gate complete → user authorized $3 SegMap re-train wave (HM-S + FR-Ω)
- **Live overnight portfolio**: 6 GPU experiments + 5 local-dev subagents running

## Phase 1 status (8 original lanes)

| # | Lane | Code state | Dispatch state | Verdict |
|---|---|---|---|---|
| 1 | SC++/q_faithful | Council C bf16+scorer-chunk fix landed | **Vast.ai 4090 firing now** ($3.12, 12h) | RE-DISPATCHED — overnight |
| 2 | Pose-delta | LANDED commit 2d913687 + ef8592d9 + PD-V2 152ba503 | Local code, no GPU dispatch needed | DONE — bolt-on ready |
| 3 | STC clean-source | Script + watchdog | **Vast.ai 4090 firing now** ($1.00, ~1h) | RE-DISPATCHED |
| 4 | Ω-W water-filling | V1 16/16 tests + V2 13/13 tests + 40.98% [empirical] | **Vast.ai 4090 firing now** ($0.50, ~30min stack archive auth eval via #272) | DERIVATION-grade dispatch firing |
| 5 | LCT | LANDED 8 tests bolt-on (commits bcf11f20, 90398b7c, 8bd467a9) | Local, no dispatch needed | DONE — bolt-on ready |
| 6 | GP rerun | Council #271 KILLED (Runge phenomenon polynomial fit) | KILLED at 89.67 [contest-CPU advisory] | DEAD |
| 7 | PSD | Script + inline watchdog (commit a63c062b) | NEVER DISPATCHED to GPU | DEFERRED per Council F |
| 8 | Multi-pass inflate | MVP control flow + plateau check landed (0e43d299) | GPU inner-step deferred | NEEDS Phase 2 GPU integration |

**9-lane SegMapTrainer invalidation** (Council A + Round 6): SC++, SA-v2, SO, WC-S, PA, HM-S, FR-Ω, FC, q_faithful's SegMap variant. Council F filtered: HM-S APPROVE (firing) + FR-Ω APPROVE (gated on HM-S) + WC-S DEFER + PA + FC KILL. **Council never ruled on SC++/SA-v2/SO/q_faithful** — I dispatched SC++ + SA-v2 + SO tonight on remaining-budget logic.

## Phase 1.5 status (Council E's stacking architecture)

| Lane | Code | Tests | Empirical/Local | Real-archive integration |
|---|---|---|---|---|
| Lane PD-V2 (arithmetic-coded pose deltas) | LANDED 152ba503 | 19 pass | 18.6% byte savings on idle-dominant trajectory [empirical] | #277 subagent landing bolt-on into Lane G v3 archive |
| Lane Ω-W-V2 (water-fill+arithmetic) | LANDED 22a2bcd2 | 13 pass + interior tests | **40.98% on REAL Lane G v3 renderer.bin** [empirical] | #272 subagent landing OWV2 inflate handler + stack auth eval ($0.50 [contest-CUDA] firing) |
| Lane Joint-ADMM (Boyd coordinator) | LANDED 152ba503/0e43d299 | 9 + 12 + 7 (4-stream + interior-optimum) pass | KKT residual 0.02 + budget-feasible convergence [empirical] | Round 11 fix #276 (Nesterov bias + adaptive rho_init) in flight |
| Lane LCT (10-byte payload) | LANDED bcf11f20 + 90398b7c + 8bd467a9 | 8 pass | Local | #277 subagent integrating bolt-on |
| Lane 8 (multi-pass inflate MVP) | LANDED 0e43d299 | 4 pass | Local control flow + plateau check | GPU inner-step deferred (Phase 2 work) |
| Lane J-NWC (neural weight codec) | LANDED 12b43507 end-to-end | 19 pass + bidirectional magic | Predicted band [0.95, 1.30] [prediction]; favorable for renderers ≥150K params | NEEDS shared-corpus codec to amortize on 88K Lane A |

## Phase 2 status (15 lanes per project_phases_2_3_4_design...)

**ACCELERATE list (Council E reprioritization)**:

| Lane | Status | Code | Tests |
|---|---|---|---|
| 10 — Joint-ADMM real-codec wrap (water_filling_codec_v2) | scaffold landed | a9ab3dc1 | 11 pass |
| 12 — NeRV mask codec | scaffold landed | 6ae70682 | 9 pass |
| 17 — IMP 10-cycle orchestrator | scaffold landed | 6351684b | 7 pass |
| 19 — SegNet logit-margin loss | scaffold landed | 142b5777 | 8 pass |
| 20 — Ballé hyperprior renderer | scaffold landed | ccbe6591 | 8 pass |

**PAUSED list (Council E reprioritization)**:
- Lane 9 — full STC rebuild
- Lane 11 — wavelet
- Lane 13 — DARTS-S full sweep
- Lane 21 — decoder rewrite
- Lane 16 — MDL

**STILL NEEDS IMPLEMENTATION in Phase 2** (next-step per lane):
- Lane 10 ADMM: Round 11 Nesterov fix (#276 in flight) → then real-archive dispatch
- Lane 12 NeRV: CUDA training pass on 1200-frame mask sequence + byte measurement vs AV1 421KB baseline. ~$1-2 / 2-4h Vast 4090 OR Modal L4
- Lane 17 IMP: CUDA full 10-cycle on Lane G v3 anchor. ~$10-20 / 12-24h. Best on Modal H100 (3x faster ≈ $40 effective vs $54 list)
- Lane 19 logit-margin: A/B vs standard CE on Lane G v3 anchor. ~$2-4 / 4-8h Vast 4090
- Lane 20 Ballé hyperprior: train ScalePriorMLP on Lane G v3 qint stream — local CPU initially fine (700-param MLP), then CUDA optional

## Phase 3 status (per project_phases_2_3_4_design...)

| Lane | Code | Status |
|---|---|---|
| Multi-pass compress optimization | Lane 8 MVP scaffold | NEEDS GPU inner-step (depends on Phase 2 sensitivity-map for byte allocation) |
| Bit-level archive optimization | not started | sketch only |
| MDL/Bayesian (MacKay) | not started | sketch only |
| Full IMP 10-cycle | Lane 17 scaffold | needs CUDA full run |
| RAFT/radial pose | not started | sketch only |
| SegNet logit-margin boundary | Lane 19 scaffold | needs A/B vs CE |
| Ballé hyperprior | Lane 20 scaffold | needs train + amortization measurement |
| Decoder systems rewrite | not started | PAUSED per Council E |

**Phase 3 strategic adds from today**:
- SegNet/PoseNet sensitivity-map module (`src/tac/sensitivity_map.py`) — #275 subagent landing local module + GPU dispatch design. Direct EV -0.005 to -0.020; INDIRECT EV unblocks Lane 19 + Lane 9 + Ω-W-V2 V3 + Lane 20 = -0.050 to -0.150 across Phase 2.
- RL/PufferLib bandit pilot (Thompson sampling, $5 cap). PILOT-WITH-CAP per Council #271. Graduate to PPO only if bandit beats hand-tuning by ≥0.005 score at same archive bytes.

## Live overnight dispatch portfolio (as of 2026-04-30 ~02:15 CDT)

| Lane | Platform | Cost | Time | Predicted band | Tag |
|---|---|---|---|---|---|
| HM-S (8-DOF homography SegMap) | Vast.ai 4090 (35885106, 54% util) | $1.75 | 6h | [0.32, 0.45] | [prediction] |
| Ω-W-V2 stack auth eval | Vast.ai 4090 (via #272) | $0.50 | ~30min | ~0.97 | [derivation] |
| SC++ retry (Council C bf16+scorer-chunk fixes) | Vast.ai 4090 | $3.12 | 12h | [0.30, 0.55] | [prediction] |
| STC clean-source retry | Vast.ai 4090 | $1.00 | ~1h | [0.45, 1.20] | [prediction] |
| SA-v2 retry (Modal A10G post-OOM-fix) | Modal A10G | $11.00 | 10h | [0.40, 0.65] | [prediction] |
| SO Hessian (Modal A10G post-OOM-fix) | Modal A10G | $8.80 | 8h | [0.30, 0.55] | [prediction] |

**Total committed**: ~$26.17 (Vast $6.37 of $30 cap = 21%; Modal $19.80 of $70 fresh = 28%). Headroom: $73.83.

**+ user has $500 in reserve** to deploy where the morning data shows wins.

## 5+ BG dev subagents currently running

| # | Subagent | Output |
|---|---|---|
| 272 | OWV2 inflate handler + Ω-W-V2 stack dispatch | Already firing the Vast.ai $0.50 |
| 273 | FR-Ω band reconciliation (script [0.25, 0.32] vs Council F [0.27, 0.45]) | local doc fix |
| 275 | Sensitivity-map module + GPU dispatch design | local module + tests + remote_lane script |
| 276 | Round 11 Joint-ADMM Nesterov + adaptive rho_init fixes | local + 2 regression tests |
| 277 | Lane PD-V2 + LCT bolt-on integration into Lane G v3 archive | deterministic local stack |
| 278 | Comprehensive Phase 1-4 battleplan (re-spawn after rate-limit) | unified theoretical-floor plan |

## STRICT preflight checks (88 total, +8 today)

- 81 silent-default audit
- 82 callsite contracts (Lane GP fit_pose_gp.py:33 missing kwarg pattern)
- 83 no-MPS-decision (STC FALSIFICATION-from-MPS pattern)
- 85 training-script metric-key consistency (DARTS-S NaN-display bug)
- 86 no-bare-round-in-eval-roundtrip (DARTS-S freeze ROOT CAUSE)
- 87 SegMap-class lane OOM-guard (bf16 + scorer-chunk required)
- 88 EMA-NON-NEGOTIABLE (training paths must wire EMA correctly)
- 89 encode-then-discard antipattern (UNIWARD NO-OP class) — warn-only

## Bug class permanent extinctions today (9 total)

1. `.round()` zero-grad in eval-roundtrip → Check 86
2. OOM 21GB FastViT attention → Check 87 + bf16+scorer-chunk
3. Missing EMA in 8 training scripts → Check 88 + Council D wire-ins
4. Concurrent commit-message swap → file-lock serializer
5. Concurrent commit staging-race → temp-index serializer V2 (b860710c)
6. UNIWARD-NO-OP encode-then-discard → Check 89 warn-only
7. KL-distill silent default drop → Q4A + TrainConfig field
8. Modal `.spawn()` 24h cache loss → tools/harvest_modal_calls.py + CLAUDE.md non-negotiable
9. Bash `run_in_background` SIGURG-144 on long-running dispatch → CLAUDE.md Pattern A non-negotiable + memory feedback_bash_run_in_background_kills_vastai_dispatch_20260430.md

## Theoretical floor analysis (multi-source synthesis)

| Source | Estimate | Notes |
|---|---|---|
| Shannon R(D) bound (Council #271 strict) | 0.28 | hard floor |
| Senior-engineer revised | 0.245 | with stacking + STC + wavelet |
| Council #271 next-48h forecast | 0.93-0.97 | with Phase A — Ω-W-V2 stack + sensitivity-map foundation |
| Quantizr leader | 0.33 | hand-tuned, no RL |
| Selfcomp #2 | 0.38 | block-FP + grayscale-LUT |
| Council brutal forecast 8% prob | sub-0.30 | 8% chance with current tools + budget |
| Lane G v3 [contest-CUDA] | 1.05 | the only trustworthy datapoint we have |

**Realistic 48h portfolio target**: 1.05 → ~0.93-0.97 (Council #271 estimate) IF:
- Ω-W-V2 stack lands at ~0.97 (DERIVATION grade) — high confidence
- HM-S lands within band [0.32, 0.45] (33% prior hit-rate)
- Sensitivity-map lands and unblocks Phase 2 lanes
- Joint-ADMM Nesterov fix unblocks Lane 10 V2 real-archive dispatch

## Cross-refs (memory files to consult)

- project_phase1_dispatch_verdict_20260429.md — original Phase 1 verdicts
- project_phases_2_3_4_design_implementation_math_provenance_20260429.md — Phase 2-4 spec
- project_codec_stacking_composition_canonical_orders_20260429.md — per-lane EV
- project_6month_strategic_plan_20260429.md — long-term roadmap
- project_lane_g_v3_landed_1_05_20260428.md — the 1.05 anchor
- feedback_round6_defects_lane_mm_correction_segmap_invalidation_extended_20260429.md — 9-lane invalidation
- feedback_modal_spawn_result_cache_pattern_20260429.md — Modal harvest discipline
- feedback_bash_run_in_background_kills_vastai_dispatch_20260430.md — Pattern A non-negotiable
- feedback_subagent_serializer_temp_index_landed_20260430.md — staging-race extinction
- All `.omx/research/council_*.md` reports — A/B/C/D/E/F + Round 6/7/8/9/10 + chain audit + #271 strategic + #278 (forthcoming)
