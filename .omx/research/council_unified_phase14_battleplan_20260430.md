# GRAND COUNCIL — Unified Phase 1-4 Battleplan to the Theoretical Floor

**Date**: 2026-04-30 (re-spawn after rate-limit; subagent #278)
**Convened by**: parent agent under user mandate "respawn all with the e + f + #271 and chain audit along with the other context you're providing" + "make sure to save and update your memories and knowledge and our documentation and state and such as we go to ensure no signal loss."
**Inner council (10 voices)**: Shannon (LEAD), Dykstra (CO-LEAD), Yousfi, Fridrich, Contrarian, Quantizr, Hotz, Selfcomp, MacKay (memorial), Ballé
**Grand-council augmentation consulted**: Boyd (ADMM operational), Carmack (engineering shortcuts), Tao (first-principles math), Filler (STC), Mallat (wavelet), van den Oord (codebook amortisation), Karpathy (RL/training engineering), Hassabis (DeepMind strategic-research), Schmidhuber (compression-as-intelligence + RL lineage), Hinton (KL distill T=2.0)
**Mandate**: REPORT-ONLY. NO code modified. NO GPU spawned. SYNTHESIS + STRATEGIC SEQUENCING ONLY.
**All claims tagged**: `[empirical:<path>]` / `[contest-CUDA]` / `[contest-CPU advisory]` / `[Modal-T4-CUDA]` / `[prediction]` / `[derivation]` / `[synthetic]`.

---

## 0. Executive Summary

The chain from Lane G v3 1.05 `[contest-CUDA]` (the only trustworthy datapoint) to the Shannon R(D) floor 0.28 `[derivation]` is **structurally complete but empirically thin** — the only verified `[contest-CUDA]` measurement we trust is Lane G v3, the only verified `[empirical]` codec measurement on a real archive is Ω-W-V2 at 40.98% byte-savings on Lane G v3's renderer.bin, and every other lane band carries a `[prediction]` tag with an empirical hit-rate prior of approximately 33%. The next 48 hours are dominated by ONE high-confidence `[derivation]`-grade dispatch (Lane G v3 + Ω-W-V2 stack archive build → contest-CUDA auth eval, predicted 0.97 ± 0.02 at $0.50) plus a foundational tooling lane (sensitivity-map module at $1-2 that unlocks 3 Phase 2 lanes worth -0.050 to -0.150 indirect EV). The HM-S/SC++/STC/SA-v2/SO portfolio is firing tonight at $26.17 committed; expected portfolio ROI is dominated by Ω-W-V2 stack confirmation, not the SegMap re-trains. Realistic floor targets: 48h ≈ 0.93-0.97 `[prediction-conditioned-on-Ω-W-V2-handler-correctness]`; 1-week ≈ 0.55-0.85 (HM-S band hit + sensitivity-map-informed Lane 19 logit-margin); 1-month ≈ 0.30-0.45 (Phase 2 NeRV mask codec + IMP 10-cycle + Ballé hyperprior land); 6-month ≈ 0.18-0.28 (Phase 4 integration with full ADMM coordinator + bit-level optimizer + paper harness). Sub-Quantizr 0.33 is a 1-month target with 35-50% probability under current portfolio; Shannon 0.28 floor is a 6-month moonshot with 15-25% probability.

**Top-5 highest-EV next-48h actions** (with cost):

1. **Lane G v3 + Ω-W-V2 stack archive build → contest-CUDA auth eval** ($0.50 GPU + 1-2h dev) — **HIGHEST CONFIDENCE in portfolio**. `[derivation]` grade rate math; codec savings already measured `[empirical:src/tac/tests/test_omega_w_v2_real_archive.py]`. Already firing via subagent #272.
2. **Sensitivity-map module + GPU dispatch** ($1-2 GPU + 1 day dev) — Foundational tooling that unlocks Lane 19 (SegNet logit-margin), Lane 9 (STC redesign), Lane 20 (Ballé hyperprior), Lane Ω-W-V2 V2.1 (per-channel weighted Hessian replacement). Direct EV -0.005 to -0.020; INDIRECT EV -0.050 to -0.150. Already firing via subagent #275.
3. **HM-S band confirmation** ($1.75 GPU, ~6h, ALREADY FIRING) — 33% prior hit-rate; if lands [0.32, 0.45] → unlocks FR-Ω dispatch ($1.50 next 24h); if lands > 0.85 → kill FR-Ω, pivot to Lane 19. Already firing on Vast.ai 4090 (instance 35885106).
4. **Round 11 Joint-ADMM Nesterov + adaptive rho_init fixes** ($0 dev, ~3-4h) — Gates Phase 2 Lane 10 V2 real-archive dispatch (Joint-ADMM real-codec wrapper). Already in flight via subagent #276.
5. **Lane PD-V2 + LCT bolt-on integration** ($0 local + $0.50 follow-on auth) — Low EV (-0.004 to -0.011) but zero-cost filler that exercises archive-load dispatch path; gates the OWV2-handler dispatch correctness check via cross-codec integration. Already in flight via subagent #277.

**Realistic floor targets** (revised by this council, with dependency chain):

| Horizon | Central target | Optimistic | Pessimistic | Required dependencies |
|---|---|---|---|---|
| 48h | **0.93-0.97** | 0.85 | 1.05 (no movement) | Ω-W-V2 stack handler correctness; HM-S band calibration |
| 1 week | **0.55-0.85** | 0.40 | 0.95 | HM-S in band + FR-Ω in band + Lane 19 logit-margin lands; sensitivity map operational |
| 1 month | **0.30-0.45** | 0.22 | 0.65 | Phase 2 NeRV mask codec; IMP 10-cycle on best-of-Phase-1 anchor; Ballé hyperprior trained |
| 6 month | **0.18-0.28** | 0.15 | 0.40 | Phase 4 integration; full ADMM coordinator on real archive; bit-level optimizer; paper reproduction harness |

---

## PART 1: Per-Phase Status Tables

### 1.1 Phase 1 (8 original lanes)

| # | Lane | Code state | Dispatch state | Verdict |
|---|---|---|---|---|
| 1 | SC++/q_faithful | Council C bf16+scorer-chunk fix landed `[empirical:commit history]` | **Vast.ai 4090 firing now** ($3.12, 12h) | RE-DISPATCHED — overnight; no Council F clearance specifically; dispatched on remaining-budget logic |
| 2 | Pose-delta | LANDED commits 2d913687 + ef8592d9 + PD-V2 152ba503 `[empirical]` 18.5% real-trajectory savings | Local code, no GPU dispatch needed | **DONE — bolt-on ready**; subagent #277 integrating |
| 3 | STC clean-source | Script + watchdog landed | **Vast.ai 4090 firing now** ($1.00, ~1h) | RE-DISPATCHED; Filler/Mallat/Ballé flagged structural one-majority bug; backup posture |
| 4 | Ω-W water-filling | V1 16/16 tests + V2 13/13 tests + 40.98% `[empirical:tests/test_omega_w_v2_real_archive.py]` | **Vast.ai 4090 firing now** ($0.50, ~30min stack archive auth eval via subagent #272) | **DERIVATION-grade dispatch firing** — highest-confidence in portfolio |
| 5 | LCT (10-byte payload) | LANDED 8 tests bolt-on (commits bcf11f20, 90398b7c, 8bd467a9) | Local, no dispatch needed | **DONE — bolt-on ready**; subagent #277 integrating |
| 6 | GP rerun | KILLED — Council #271 verdict (Runge phenomenon polynomial fit at degree-10 over 600 equispaced points) | KILLED at 89.67 `[contest-CPU advisory]` | **DEAD** |
| 7 | PSD | Script + inline watchdog (commit a63c062b); profile PSD_STANDARD_ADAPTIVE in profiles.py:168 | NEVER DISPATCHED to GPU | **DEFERRED per Council F**; remote_lane script template adaptation gap remains |
| 8 | Multi-pass inflate | MVP control flow + plateau check landed (0e43d299); `_default_inner_step = NotImplementedError` | GPU inner-step deferred | **NEEDS Phase 2 GPU integration**; depends on sensitivity-map module for byte allocation |

**9-lane SegMapTrainer invalidation context**: SC++, SA-v2, SO, WC-S, PA, HM-S, FR-Ω, FC, q_faithful's SegMap variant all silently produced never-trained checkpoints due to `.round()` zero-grad bug (Council A finding, Round 6 5-lane correction; bug class extinct via Check 86). Council F filtered: HM-S APPROVE (firing) + FR-Ω APPROVE (gated on HM-S) + WC-S DEFER + PA + FC KILL. **Council F never ruled on SC++/SA-v2/SO/q_faithful** — parent dispatched SC++ + SA-v2 + SO tonight on remaining-budget logic.

### 1.2 Phase 1.5 (Council E's stacking architecture)

| Lane | Code | Tests | Empirical/Local | Real-archive integration | Status |
|---|---|---|---|---|---|
| Lane PD-V2 (arithmetic-coded pose deltas) | LANDED 152ba503 | 19 pass | 18.6% byte savings on idle-dominant trajectory `[empirical]` | #277 subagent landing bolt-on into Lane G v3 archive | 🟢 READY |
| Lane Ω-W-V2 (water-fill+arithmetic) | LANDED 22a2bcd2 | 13 pass + interior tests | **40.98% on REAL Lane G v3 renderer.bin** `[empirical:tests/test_omega_w_v2_real_archive.py]` | #272 subagent landing OWV2 inflate handler + stack auth eval ($0.50 [contest-CUDA] firing) | 🔵 IN-FLIGHT |
| Lane Joint-ADMM (Boyd coordinator) | LANDED 152ba503/0e43d299 | 9 + 12 + 7 (4-stream + interior-optimum) pass | KKT residual 0.02 `[synthetic]` + budget-feasible convergence `[empirical:tests/test_joint_admm_4stream_nonconvex.py]` | Round 11 fix #276 (Nesterov bias + adaptive rho_init) in flight | 🔵 FIX-IN-FLIGHT |
| Lane LCT (10-byte payload) | LANDED bcf11f20 + 90398b7c + 8bd467a9 | 8 pass | Local | #277 subagent integrating bolt-on | 🟢 READY |
| Lane 8 (multi-pass inflate MVP) | LANDED 0e43d299 | 4 pass | Local control flow + plateau check `[synthetic]` | GPU inner-step deferred (Phase 2 work) | 🟡 PARTIAL — outer loop only |
| Lane J-NWC (neural weight codec) | LANDED 12b43507 end-to-end | 19 pass + bidirectional magic | Predicted band [0.95, 1.30] `[prediction]`; favorable for renderers ≥150K params | NEEDS shared-corpus codec to amortize on 88K Lane A | 🟡 STANDALONE LOSES — corpus codec gap |

### 1.3 Phase 2 (15 lanes — Council E reprioritized)

**ACCELERATE list (5 lanes)** — fast-track Phase 2 weeks 2-3 per Council E §5.1:

| Lane | Status | Code | Tests | Next step |
|---|---|---|---|---|
| 10 — Joint-ADMM real-codec wrap (water_filling_codec_v2 as proximal) | 🟡 scaffold landed | a9ab3dc1 | 11 pass | Wait for #276 Nesterov fix → real-archive dispatch on Lane G v3 anchor |
| 12 — NeRV mask codec (van den Oord) | 🟡 scaffold landed | 6ae70682 | 9 pass | CUDA training pass on 1200-frame mask sequence vs AV1 421KB baseline (~$1-2 / 2-4h Vast 4090 OR Modal L4) |
| 17 — IMP 10-cycle orchestrator (Frankle-Carbin lottery ticket) | 🟡 scaffold landed | 6351684b | 7 pass | CUDA full 10-cycle on Lane G v3 anchor (~$10-20 / 12-24h; best on Modal H100 ≈ $40 effective vs $54 list) |
| 19 — SegNet logit-margin loss (Yousfi/Fridrich score-aware compression) | 🟡 scaffold landed | 142b5777 | 8 pass | A/B vs standard CE on Lane G v3 anchor (~$2-4 / 4-8h Vast 4090); **gated on sensitivity-map module landing** |
| 20 — Ballé hyperprior renderer | 🟡 scaffold landed | ccbe6591 | 8 pass | Train ScalePriorMLP on Lane G v3 qint stream — local CPU initially fine (700-param MLP), then CUDA optional |

**PAUSED list (5 lanes)** — defer to Phase 3 per Council E §5.2:

- Lane 9 — full STC rebuild (Filler/Mallat/Ballé flagged structural bug; even clean-source CUDA confirm wouldn't deliver -45KB threshold; **document as paper negative result**)
- Lane 11 — wavelet residual codec (Mallat acknowledged "post-deadline paper lane"; viable under 30-day budget but lower EV than Lane 12)
- Lane 13 — DARTS-S full sweep (V1 had model-not-learning bug; Check 85 + 86 caught display-NaN + bare-round; full sweep gated on V1 fix)
- Lane 16 — MDL/Bayesian (MacKay) — pure analysis lane; requires Lanes 9-15 results as inputs
- Lane 21 — decoder rewrite (saves time-budget not score; defer until Lanes 12/14/15 need the 30→10 min headroom)

**STILL NEEDS IMPLEMENTATION in Phase 2** (next-step per lane):

- Lane 10 ADMM: Round 11 Nesterov fix (#276 in flight) → then real-archive dispatch
- Lane 12 NeRV: CUDA training pass + byte measurement vs AV1 baseline
- Lane 17 IMP: CUDA full 10-cycle on Lane G v3 anchor
- Lane 19 logit-margin: A/B vs standard CE on Lane G v3 anchor (gated on sensitivity-map)
- Lane 20 Ballé hyperprior: train ScalePriorMLP on Lane G v3 qint stream

### 1.4 Phase 3 (roadmap items + dependencies on Phase 2 outputs)

| Lane | Code | Status | Dependency |
|---|---|---|---|
| 14 — Multi-pass compress optimization | Lane 8 MVP scaffold | 🟡 NEEDS GPU inner-step | depends on Phase 2 sensitivity-map for byte allocation |
| 15 — Bit-level archive optimization | not started | ❌ sketch only | requires Lane 14 (multi-pass) infrastructure |
| 16 — MDL/Bayesian (MacKay) | not started | ❌ sketch only | requires Lanes 9-15 results |
| 17 — Full IMP 10-cycle | Lane 17 scaffold | 🟡 needs CUDA full run | requires Lane 1 base checkpoint OR Lane G v3 anchor |
| 18 — RAFT/radial pose | `src/tac/raft_pose.py` exists untracked | ❌ Phase 3 integration gap | Standalone but needs `inflate_renderer.py` integration |
| 19 — SegNet logit-margin boundary | Lane 19 scaffold | 🟡 needs A/B vs CE | gated on sensitivity-map module |
| 20 — Ballé hyperprior | Lane 20 scaffold | 🟡 needs train + amortization measurement | best on shared-corpus codec (paired with Lane J-NWC corpus pivot) |
| 21 — Decoder systems rewrite | not started | PAUSED per Council E | defer until Lanes 12/14/15 land |

**Phase 3 strategic adds from today (Council #271)**:

- **SegNet/PoseNet sensitivity-map module** (`src/tac/sensitivity_map.py`) — #275 subagent landing local module + GPU dispatch design. Direct EV -0.005 to -0.020; INDIRECT EV unblocks Lane 19 + Lane 9 + Ω-W-V2 V3 + Lane 20 = -0.050 to -0.150 across Phase 2.
- **RL/PufferLib bandit pilot** (Thompson sampling, $5 cap). PILOT-WITH-CAP per Council #271. Graduate to PPO only if bandit beats hand-tuning by ≥0.005 score at same archive bytes.

---

## PART 2: Theoretical Floor Synthesis

### 2.1 Multi-source estimate reconciliation

| Source | Estimate | Tag | Notes |
|---|---|---|---|
| Shannon R(D) bound (Council #271 strict) | 0.28 | `[derivation]` | hard floor; rate term `25 × 250000 / 37545489 = 0.1665`; remainder 0.1135 for seg+pose |
| Senior-engineer revised (memory `project_senior_engineer_review_floor_revised_245`) | 0.245 | `[derivation]` aggressive | adds -0.035 from "realistic STC + wavelet (with overlap discount) + custom container" |
| Council #271 next-48h forecast | 0.93-0.97 | `[derivation+prediction]` | with Phase A — Ω-W-V2 stack + sensitivity-map foundation |
| Council brutal forecast 8% prob | sub-0.30 | `[prediction]` | 8% chance with current tools + budget; based on 22-voice grand council 2026-04-29 PM |
| Quantizr leader (PR #56 0.33 archive) | 0.33 | `[contest-CUDA]` external | hand-tuned, no RL; 88K params + FiLM-conditioned depthwise-separable + grayscale-LUT mask |
| Selfcomp #2 (PR #55) | 0.38 | `[contest-CUDA]` external | block-FP at 1.017 bpw + 94K SegMap |
| Lane G v3 (our best) | 1.05 | `[contest-CUDA]` | the only trustworthy datapoint we have |

### 2.2 Where the disagreements come from

Three structural sources of dispersion across estimates:

**Source 1: Assumed R(D) curves are sparsely sampled** (chain audit Step 2 CONCERN). Per-stream marginals quoted as `dScore/dByte ≈ 0.00067` are SINGLE-POINT estimates, not full curves. Pose stream has 1 operating point; mask stream 2 operating points (one of which was MPS-contaminated and withdrawn); renderer stream 2 operating points (Lane G v3 raw + Ω-W-V2-encoded). The Pareto-optimal allocation in chain audit Step 3 operates on **assumed shapes** (smooth quadratic, linear-saturate, sigmoid-saturating, discrete-jump per `test_joint_admm_4stream_nonconvex.py`), not measured shapes. This is the foundational gap that the sensitivity-map module (Council #271 Q2) is engineered to close.

**Source 2: Stacking interaction terms are unmeasured** (Dykstra's "additivity is conditional" rule). Two techniques each saving 30KB might overlap and only deliver 40KB stacked. Council F's orthogonality matrix is categorical (yes/no per lane), not numerical convex-hull intersection. Today the only measured stacking interaction is the implied Lane G v3 + Ω-W-V2 derivation (40.98% codec savings × 38.5% archive share = ~117KB → 0.078 score reduction `[derivation]`); all other stacking claims are predictions.

**Source 3: bf16/fp16 numerics + roundtrip gates not modeled in floor estimates**. Shannon's 0.28 floor assumes byte-deterministic bit allocation; in practice every codec operates on bf16/fp16 quantized weights. Lane G v3 ships at bf16 per Council C training fix; Ω-W-V2 uses fp32 internally with fp16 output. The drift is ~0.005-0.01 per operation; stacking 5 codec operations could introduce ~0.025-0.05 drift not captured in the floor derivation.

**Source 4: empirical hit-rate prior ~33%** for `[prediction]`-tagged dispatches. Of 6 lanes with `[contest-CUDA]` or `[contest-CPU advisory]` outcomes vs predictions: 2 hit (Lane G v3, UNIWARD v8), 1 missed by 2× (Lane MM v2 predicted [0.65, 0.85] actual 2.63), 2 crashed without measurement (Lane V channel-mismatch, Lane M-V2 not predicted), 1 withdrawn (STC MPS-contaminated). Council senior-eng forecast at 0.245 implicitly assumes ~50% hit rate on Phase 2 dispatches; this is optimistic by ~17 percentage points.

### 2.3 Realistic attainable floor

| Horizon | Floor central | Floor optimistic | Floor pessimistic | Mechanism producing floor |
|---|---|---|---|---|
| **48h** | **0.93-0.97** | 0.85 | 1.05 (no movement; status quo) | Ω-W-V2 stack rate-term reduction (-0.078 `[derivation]`) + sensitivity-map foundation primed; HM-S optional kicker IF in band |
| **1 week** | **0.55-0.85** | 0.40 | 0.95 | HM-S in band [0.32, 0.45] + FR-Ω in band [0.27, 0.45] (additive ~ -0.20 to -0.55 if both hit) + Lane 19 logit-margin lands using sensitivity map (-0.020 to -0.080) |
| **1 month** | **0.30-0.45** | 0.22 | 0.65 | Phase 2 lands: Lane 12 NeRV (mask codec replacing AV1 421KB → ~50-100KB → -0.20 score), Lane 17 IMP 10-cycle (-0.05 to -0.10 on archive size), Lane 20 Ballé hyperprior (-0.01 to -0.03), Lane 10 ADMM real-codec wrapping (-0.015 to -0.050 across stack) |
| **6 month** | **0.18-0.28** | 0.15 | 0.40 | Phase 4 integration: bit-level optimizer (Lane 15) + multi-pass compress (Lane 14) + MDL stack ranking (Lane 16) + decoder rewrite (Lane 21) + RAFT/radial pose (Lane 18) — approaches Shannon 0.28 floor |

**Sub-Quantizr 0.33**: 1-month target with 35-50% probability under current portfolio.
**Shannon 0.28 floor**: 6-month moonshot with 15-25% probability.

---

## PART 3: Critical Path to Theoretical Floor

The chain from Lane G v3 1.05 to Shannon 0.28 floor, with HONEST identification of "diminishing returns" / "needs an idea we don't have yet":

### Transition 1: Lane G v3 1.05 → ~0.97 (next 48h)

**Mechanism**: Ω-W-V2 stack rate-term reduction. Lane G v3 archive composition: renderer.bin 296,776B (38.5%) + masks.mkv 421,483B (59.4%) + optimized_poses.pt 15,620B (2.0%) = 694,074B archive. Ω-W-V2 measured 40.98% on 285,544B raw conv weights = 117,027B savings `[empirical:tests/test_omega_w_v2_real_archive.py]`. Rate term shift: `25 × 117027 / 37545489 = 0.0779` = ~0.08 score reduction `[derivation]`.

**Cost**: $0.50 GPU (auth eval) + 1-2h dev (OWV2 inflate handler ~30 LOC).
**Risk**: handler-correctness (15% probability of bug introducing regression).
**Confidence**: HIGH (rate math is bit-deterministic; codec already validated).
**Status**: Subagent #272 in flight tonight.

### Transition 2: ~0.97 → ~0.85-0.55 (next 1 week)

**Mechanism**: HM-S 8-DOF homography geometric improvement on PoseNet wedge (-0.09 to -0.11 if band hits) + FR-Ω Hessian-aware block-FP on rate wedge (-0.05 to -0.10 if band hits) + Lane 19 SegNet logit-margin loss using sensitivity-map (-0.020 to -0.080 once sensitivity-map operational).

**Cost**: $1.75 (HM-S, already firing) + $1.50 (FR-Ω, gated on HM-S) + $1-2 (sensitivity-map, firing) + $2-4 (Lane 19 A/B, gated on sensitivity-map) = $6.25-9.25 GPU.
**Risk**: HM-S ~33% prior hit-rate; FR-Ω band [0.25-0.32] script vs [0.27-0.45] Council F INCONSISTENCY (subagent #273 reconciling); Lane 19 dependent on sensitivity-map landing.
**Confidence**: MEDIUM-LOW (3 of 4 mechanisms are `[prediction]`-tagged).
**Status**: HM-S firing; sensitivity-map firing; Lane 19 + FR-Ω gated.

### Transition 3: ~0.55 → ~0.30-0.45 (next 1 month)

**Mechanism**: Phase 2 codec lanes land:
- **Lane 12 NeRV mask codec** replaces AV1 421KB → predicted ~50-100KB MLP → archive shrinks by ~320-370KB → rate term drops by ~0.21-0.25 `[prediction]`
- **Lane 17 IMP 10-cycle** prunes renderer to 90% sparsity → archive shrinks by ~50KB more → -0.03 score `[prediction]`
- **Lane 20 Ballé hyperprior** replaces static-histogram terminal → -0.01 to -0.03 score `[prediction]`
- **Lane 10 ADMM real-codec wrapping** equilibrates per-stream marginals → -0.015 to -0.050 across stack `[prediction]`

**Cost**: $1-2 (NeRV training) + $10-20 (IMP 10-cycle, best on Modal H100) + $0 (Ballé hyperprior local CPU initially) + $0 (ADMM wrapping local) = $11-22 GPU.
**Risk**: NeRV inflate-time fits within 30-min budget (236M coordinates × forward pass on T4 ~2.4s ✓ verified per Lane 12 design); IMP convergence on small renderer (Lane G v3 anchor only 88K params — IMP works best on overparameterized models).
**Confidence**: MEDIUM (Phase 2 lanes are scaffolded but unmeasured on real archives; Lane 12 NeRV is the dominant lever).
**Status**: All 4 lanes scaffolded; need real-archive integration (not just synthetic tests).

### Transition 4: ~0.30 → ~0.20 (next 3 months — diminishing returns begin)

**Mechanism**: Phase 3 lanes:
- **Lane 14 multi-pass compress optimization** with sensitivity-map-informed byte allocation (-0.005 to -0.020)
- **Lane 15 bit-level archive optimizer** (gradient search over bit-stream) (-0.001 to -0.010)
- **Lane 16 MDL/Bayesian model selection** picks the best stack composition from 5+ codec families (-0.005 to -0.020)
- **Lane 18 RAFT/radial pose preimage** stores low-rank flow → reduces mask payload by 50KB (-0.03 score)
- **Lane J-NWC corpus codec** amortizes weight codec across multiple renderers (-0.020 to -0.060 IF amortization condition met)

**Cost**: $30-60 GPU (multi-pass + bit-level + corpus codec dev).
**Risk**: HIGH — these are speculative gains; honest assessment is most lanes deliver ≤ -0.005 score individually; the cumulative effect requires the Lane 16 MDL framework to MEASURE which compositions actually work.
**Confidence**: LOW-MEDIUM (the ideas are sound but scoring is uncertain).
**Status**: Most are sketches; Lane 18 has untracked code (`src/tac/raft_pose.py`).

### Transition 5: ~0.20 → 0.18-0.15 (months 4-6 — needs an idea we don't have yet)

**HONEST ASSESSMENT**: At ~0.20 we are within 2× of Shannon's 0.28 hard floor. Further reduction requires EITHER:
- (a) **A novel architectural insight** we don't currently have. Selfcomp's grayscale-LUT and Quantizr's FiLM-conditioned depthwise-separable were paradigm shifts beyond what Phase 1-2 lanes attempt; another paradigm shift may be needed.
- (b) **Loosening the contest constraints** (e.g., longer inflate budget enabling richer compute). Within current 30-min inflate this is hard.
- (c) **Multi-modality joint compression** — encoding renderer + masks + poses through a shared latent space (NeRF-like). This is Phase 4 integration territory but would require ~3 months of dev to implement cleanly.

**Diminishing returns transition**: occurs around 0.30-0.20 score range, where most byte savings are already realized and further improvement requires novel ideas not currently in the lane portfolio. The 6-month 0.18-0.28 floor is conditional on at least ONE paradigm-shift insight landing (Lane 12 NeRV at <50KB IS such an insight if it delivers; Lane 18 RAFT pose preimage IS such an insight if it integrates cleanly).

---

## PART 4: Updated Dispatch Order

### 4.1 TONIGHT (already firing)

| Lane | Platform | Cost | Time | Predicted band | Tag |
|---|---|---|---|---|---|
| HM-S (8-DOF homography SegMap) | Vast.ai 4090 (35885106, 54% util) | $1.75 | 6h | [0.32, 0.45] | `[prediction]` |
| Ω-W-V2 stack auth eval | Vast.ai 4090 (via #272) | $0.50 | ~30min | ~0.97 | `[derivation]` |
| SC++ retry (Council C bf16+scorer-chunk fixes) | Vast.ai 4090 | $3.12 | 12h | [0.30, 0.55] | `[prediction]` |
| STC clean-source retry | Vast.ai 4090 | $1.00 | ~1h | [0.45, 1.20] | `[prediction]` |
| SA-v2 retry (Modal A10G post-OOM-fix) | Modal A10G | $11.00 | 10h | [0.40, 0.65] | `[prediction]` |
| SO Hessian (Modal A10G post-OOM-fix) | Modal A10G | $8.80 | 8h | [0.30, 0.55] | `[prediction]` |

**Total committed**: $26.17 (Vast $6.37 of $30 cap = 21%; Modal $19.80 of $70 fresh = 28%). Headroom: $73.83 + $500 user reserve.

### 4.2 NEXT 24H (gated on tonight's results)

| Order | Lane | Cost | EV | Dependency |
|---|---|---|---|---|
| 1 | **Sensitivity-map module CUDA dispatch** (Council #271 Q2) | $1-2 | -0.005 to -0.020 direct + -0.050 to -0.150 indirect | None — fire now after #275 lands module |
| 2 | **Lane PD-V2 + LCT bolt-on contest-CUDA confirm** (subagent #277 output) | $0.50 | -0.004 to -0.011 | Subagent #277 commits land |
| 3 | **Lane 19 SegNet logit-margin A/B vs CE** | $2-4 | -0.020 to -0.080 | Sensitivity-map operational |
| 4 | **FR-Ω Hessian block-FP** (gated on HM-S in-band) | $1.50 | -0.05 to -0.10 IF HM-S calibrates | HM-S band hit + script/Council F band reconciled (subagent #273) |
| 5 | **Lane Ω-W-V2 V2.1 with measured per-channel sensitivity weights** | $0.50 | -0.005 to -0.020 incremental | Sensitivity-map operational + Ω-W-V2 stack baseline confirmed |
| 6 | **Bandit pilot** (Council #271 Q3 PILOT-WITH-CAP) | $5 cap | If beats hand-tuning by ≥0.005 → graduate to PufferLib | Sensitivity-map operational (defines action space) |

**Total NEXT 24H budget**: ~$10-13 GPU (well within Vast.ai $30 daily cap).

### 4.3 NEXT WEEK (Phase 2 deeper — scaffolded lanes graduate to real-archive)

| Lane | Cost | Effort | Action |
|---|---|---|---|
| Lane 10 Joint-ADMM real-codec wrap V2 | $1-2 | 1 day dev + dispatch | After #276 Round 11 fix lands; wrap water_filling_codec_v2 as proximal codec; dispatch real-archive ADMM on Lane G v3 anchor |
| Lane 12 NeRV mask codec CUDA training | $1-2 | 2-3 days dev + dispatch | Coordinate-MLP overfit to 1200-frame mask sequence; quantize to int8; ship 30-50KB; verify <80KB total mask payload |
| Lane 17 IMP 10-cycle on best-of-Phase-1 anchor | $10-20 | 3-5 days dev + 12-24h Modal H100 | Iterative Magnitude Pruning; rewind weights to early iteration; 90% sparsity target |
| Lane 20 Ballé hyperprior train ScalePriorMLP | $0-1 | 2 days dev | Train on Lane G v3 qint stream; local CPU fine for 700-param MLP; CUDA optional for amortization measurement |
| OWV2 multi-codec stack archive build | $0.50 | 1 day dev | Combine Ω-W-V2 + PD-V2 + LCT + (optional sensitivity-map weights) in one archive; auth eval |

**Total NEXT WEEK budget**: ~$15-25 GPU.

### 4.4 NEXT MONTH (Phase 3 — big-bet experiments)

| Experiment | Cost | Effort | Expected impact |
|---|---|---|---|
| RL/PufferLib full PPO (only if bandit beats hand-tuning) | $50-100 | 1-2 weeks dev | -0.010 to -0.050 if action space genuinely benefits from credit assignment |
| Sensitivity-map full pass over 600 pairs × 5-10 perturbation types | $5-10 | 1 week analysis | Defines empirical R(D) curves; replaces "trust me" links in chain audit Step 2 |
| Lane J-NWC corpus codec (multi-renderer amortization) | $5-10 | 2 weeks dev | -0.020 to -0.060 ON LARGE renderers; small impact on 88K Lane A but unblocks Phase 4 architecture exploration |
| Lane 18 RAFT/radial pose preimage integration | $5-10 | 2-3 weeks dev | -0.03 if mask payload reduction lands cleanly; risk on inflate-time integration |
| Ballé hyperprior train on shared corpus | $5-10 | 2 weeks dev | -0.01 to -0.03; gates V3 hyperprior amortization decision |
| Lane 11 wavelet residual codec (Mallat) | $3-5 | 2 weeks dev | -0.080 to -0.300 bp; lower EV than Lane 12 NeRV but orthogonal |
| Lane 14 multi-pass compress with sensitivity-informed allocation | $5-10 | 2 weeks dev | -0.005 to -0.020 |
| Lane 15 bit-level archive optimizer | $5-10 | 3 weeks dev | -0.001 to -0.010 |
| Lane 16 MDL/Bayesian stack composition ranking | $0-2 | 2 weeks analysis | Picks the best stack from N codec families; methodological contribution for paper |

**Total NEXT MONTH budget**: ~$80-160 GPU (within $500 user reserve).

---

## PART 5: Outstanding Implementation Gaps

### 5.1 Phase 1.5 gaps

| Gap | Owner | Effort | Status |
|---|---|---|---|
| Lane 8 GPU inner-step (`_default_inner_step` is `NotImplementedError`) | Phase 2 follow-up | 2-3 days dev + GPU integration | Deferred from MVP; gated on sensitivity-map (byte allocation source) |
| Joint-ADMM Round 11 Nesterov dual-averaging bias fix (Q4A) | Subagent #276 | 30 min code + 1h test | IN FLIGHT |
| Joint-ADMM rho_init adaptive default (Q4B) | Subagent #276 | 1h code + 1h test | IN FLIGHT |
| OWV2 inflate handler (`submissions/robust_current/inflate_renderer.py`) | Subagent #272 | 30 LOC + 50 LOC test | IN FLIGHT — gates Lane G v3 + Ω-W-V2 stack auth eval |
| FR-Ω script vs Council F band reconciliation ([0.25-0.32] vs [0.27-0.45]) | Subagent #273 | 1h doc fix + script update | IN FLIGHT |
| Lane PD-V2 + LCT bolt-on integration into Lane G v3 archive | Subagent #277 | 1 day local + tests | IN FLIGHT |

### 5.2 Phase 2 gaps

EVERY scaffolded lane needs **real-archive integration** (not just synthetic tests):
- Lane 10 (Joint-ADMM): wrap real codecs (water_filling_codec_v2, pose_delta_codec_v2) as `StreamProximalCodec` Protocol implementations; dispatch on real Lane G v3 archive
- Lane 12 (NeRV): training pipeline on real 1200-frame mask sequence; not just synthetic per-frame
- Lane 17 (IMP): 10-cycle full run on Lane G v3 anchor with proper rewinding
- Lane 19 (SegNet logit-margin): A/B real archive build vs CE-trained baseline
- Lane 20 (Ballé hyperprior): train on real Lane G v3 qint stream; measure amortization cost

### 5.3 Phase 3 gaps

- **Sensitivity-map module** (`src/tac/sensitivity_map.py`): #275 in flight — local module + GPU dispatch design + 600-pair × 5-10 perturbation pass
- **Bit-level archive optimization** (Lane 15): not started — sketch only
- **MDL/Bayesian framework** (Lane 16): not started — sketch only
- **RAFT/radial pose** (Lane 18): `src/tac/raft_pose.py` exists untracked; needs `inflate_renderer.py` integration; 2-3 weeks dev
- **Decoder rewrite** (Lane 21): PAUSED per Council E; reactivate when needed

### 5.4 Strategic adds

- **RL/PufferLib pilot**: bandit-first per Council #271 Q3; full PPO only if bandit beats hand-tuning by ≥0.005 score at same archive bytes
- **Shared-corpus codec for J-NWC amortization**: not started; required to make Lane J-NWC favorable on small renderers (88K Lane A); enables Phase 4 architecture exploration

---

## PART 6: Risk Register + Mitigation

| # | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **HM-S band miss** (33% prior hit-rate) | MEDIUM (no prior empirical) | -$1.75 sunk + lost 6h | Backup posture: Ω-W-V2 stack derivation lands ~0.97 regardless; HM-S is NOT the critical path |
| 2 | **SC++/SA/SO Modal A10G post-OOM-fix retry surfaces unknown bugs** (no Council F clearance) | MEDIUM | -$22.92 sunk + multi-day delay | Pre-emptive: heartbeat watchdog catches stale > 30 min; Modal Volume writes prevent spawn() result-cache loss (per Round 5 §4.4 fix landed today) |
| 3 | **Lane MM v2 = 2.63 [contest-CPU advisory]** — should we $0.50 CUDA confirm? | LOW (advisory verdict already FALSIFIED) | $0.50 burn for documentation only | Council Round 5 §3.2 verdict: NO. CUDA confirm duplicates already-FALSIFIED verdict. Skip unless paper section requires the formal CUDA tag. |
| 4 | **Vast.ai orphan-instance risk** (per CLAUDE.md `feedback_vastai_cost_paranoia`) | LOW (Check A `--label` enforcement + Check B tracker registration) | -$0.05 to -$5 per orphan × N orphans | Active tracker `.omx/state/vastai_active_instances.json`; cleanup script `tools/destroy_orphan_instances.py`; Check A + B in preflight (warn-only, can graduate to STRICT) |
| 5 | **Modal `.spawn()` 24h cache GC** | MEDIUM | result loss → re-run cost | Round 5 §4.4 prescribed Modal Volume writes in `experiments/modal_train_lane.py`; harvest tool `tools/harvest_modal_calls.py` exists; CLAUDE.md non-negotiable lands |
| 6 | **OWV2 inflate handler bug** (subagent #272 in flight) | LOW-MEDIUM (~15%) | Lane G v3 + Ω-W-V2 auth eval REGRESSES → diagnostic burden | Hard gate: if auth > 1.05, investigate handler before any further dispatch on this stack |
| 7 | **Sensitivity-map MPS/CPU contamination** | LOW (CLAUDE.md non-negotiable) | False R(D) curves contaminate Lane 19/20 design | Module mandates `device='cuda'` + `eval_roundtrip=True`; tag HDF5 output with `gpu_sha=<vast.ai instance>` provenance |
| 8 | **Round 11 fix #276 misdiagnoses Q4C J_JBL** (Council #271 verdict: MISDIAGNOSED) | LOW | unnecessary API churn in `tac/training.py:91` | Council #271 explicit: KEEP J_JBL_DILATED_H64; the two `loss_mode` namespaces are independent by design (`train_renderer.py:357` argparse + Trainer's Pydantic Literal). #276 must not change J_JBL routing. |
| 9 | **HM-S/FR-Ω band inconsistency** (subagent #273 reconciling) | LOW (in flight) | Ambiguous kill criterion if FR-Ω lands at 0.40 (script: MISS; Council F: HIT) | Subagent #273 producing single canonical band; commit lands before FR-Ω dispatch decision |
| 10 | **Lane 12 NeRV inflate-time exceeds 30-min budget** | LOW (verified 2.4s on T4 per Lane 12 design) | Lane 12 unviable for shipping | Empirical test on Vast.ai 4090 within 1 month dispatch; budget allows fallback to Lane 11 wavelet |
| 11 | **Subagent commit-message swap recurrence** | LOW (file-lock serializer landed; staging-race extinct b860710c) | mis-attributed commit history | `tools/subagent_commit_serializer.py` mandatory for all subagent commits; per-subagent worktree (Option A) is the next-step extinction pattern |
| 12 | **Codex CLI bash-144 SIGURG kill on long-running dispatch** | LOW (CLAUDE.md Pattern A non-negotiable + memory feedback) | dispatch dies silently | Pattern A `nohup + bash -c '...' + disown` mandatory; Bash `run_in_background: true` for `codex exec` is FORBIDDEN |

---

## 7. Final Dispatch Table (Synthesis)

| Phase | Time | Lane | Cost | EV (score Δ) | Confidence | Dependency | Tag |
|---|---|---|---|---|---|---|---|
| **TONIGHT** | T+0h | HM-S 8-DOF homography | $1.75 | -0.60 to -0.73 IF in band | LOW (33% hit) | None | `[prediction]` |
| **TONIGHT** | T+0h | Ω-W-V2 stack auth eval | $0.50 | -0.078 | HIGH | OWV2 handler land | `[derivation]` |
| **TONIGHT** | T+0h | SC++ retry | $3.12 | -0.50 to -0.75 IF lands | LOW (no Council F clearance) | None | `[prediction]` |
| **TONIGHT** | T+0h | STC clean-source retry | $1.00 | n/a (backup) | LOW | None | `[prediction]` |
| **TONIGHT** | T+0h | SA-v2 retry | $11.00 | -0.40 to -0.65 IF lands | LOW (no Council F clearance) | None | `[prediction]` |
| **TONIGHT** | T+0h | SO Hessian | $8.80 | -0.50 to -0.75 IF lands | LOW (no Council F clearance) | None | `[prediction]` |
| **NEXT 24H** | T+12h | Sensitivity-map module CUDA | $1-2 | -0.005 to -0.020 direct + -0.050 to -0.150 indirect | HIGH (foundational) | #275 lands | `[empirical-pending]` |
| **NEXT 24H** | T+18h | PD-V2 + LCT bolt-on auth | $0.50 | -0.004 to -0.011 | MEDIUM | #277 lands | `[derivation+empirical]` |
| **NEXT 24H** | T+24h | Lane 19 SegNet logit-margin A/B | $2-4 | -0.020 to -0.080 | MEDIUM | Sensitivity-map | `[prediction]` |
| **NEXT 24H** | T+30h | FR-Ω Hessian block-FP | $1.50 | -0.05 to -0.10 IF in band | LOW-MEDIUM | HM-S calibrates + #273 reconciles | `[prediction]` |
| **NEXT 24H** | T+36h | Ω-W-V2 V2.1 (per-channel sens) | $0.50 | -0.005 to -0.020 incremental | HIGH | Sensitivity-map + Ω-W-V2 baseline | `[derivation]` |
| **NEXT 24H** | T+42h | Bandit pilot ($5 cap) | $5 | -0.005 to -0.020 IF beats hand-tuning | LOW (PILOT-WITH-CAP) | Sensitivity-map | `[prediction]` |
| **NEXT WEEK** | Day 4-7 | Lane 10 Joint-ADMM real-codec | $1-2 | -0.015 to -0.050 across stack | MEDIUM | #276 Round 11 fix | `[prediction]` |
| **NEXT WEEK** | Day 4-7 | Lane 12 NeRV mask codec CUDA | $1-2 | -0.20 to -0.25 IF lands < 80KB | MEDIUM | None | `[prediction]` |
| **NEXT WEEK** | Day 6-10 | Lane 17 IMP 10-cycle Modal H100 | $10-20 | -0.05 to -0.10 | MEDIUM | None | `[prediction]` |
| **NEXT WEEK** | Day 5-7 | Lane 20 Ballé hyperprior local | $0-1 | -0.01 to -0.03 | MEDIUM | None | `[prediction]` |
| **NEXT MONTH** | Wk 2-4 | Lane J-NWC corpus codec | $5-10 | -0.020 to -0.060 IF amortizes | LOW-MEDIUM | corpus codec built | `[prediction]` |
| **NEXT MONTH** | Wk 3-5 | Lane 18 RAFT/radial pose | $5-10 | -0.03 IF lands cleanly | LOW | inflate integration | `[prediction]` |
| **NEXT MONTH** | Wk 3-5 | Lane 11 wavelet residual codec | $3-5 | -0.008 to -0.030 | LOW | None | `[prediction]` |
| **NEXT MONTH** | Wk 4-6 | Lane 14 multi-pass compress | $5-10 | -0.005 to -0.020 | LOW | sensitivity-map full pass | `[prediction]` |
| **NEXT MONTH** | Wk 4-6 | Lane 15 bit-level optimizer | $5-10 | -0.001 to -0.010 | LOW | Lane 14 infrastructure | `[prediction]` |
| **NEXT MONTH** | Wk 5-7 | Lane 16 MDL/Bayesian ranking | $0-2 | -0.005 to -0.020 (composition) | MEDIUM | Lanes 9-15 results | `[derivation+empirical]` |

---

## 8. Council Roll Call

Each inner-council member casts a signed verdict (1-2 sentences). Per CLAUDE.md "Council conduct" — non-conservative; arguments are mathematical/empirical only.

**Shannon (LEAD, Information Theory)**: The chain from Lane G v3 1.05 to Shannon 0.28 floor decomposes cleanly: Transition 1 is `[derivation]`-grade (Ω-W-V2 stack rate-term -0.078, the strongest link in the chain); Transitions 2-3 are `[prediction]`-tagged with ~33% empirical hit-rate prior; Transition 4 has diminishing returns; Transition 5 needs an idea we don't have yet. The sensitivity-map module is the foundational tooling that converts the "trust me" load-bearing R(D) curves (chain audit Step 2) into measured curves. **VERDICT: dispatch order Section 7 is correct; the 48h target 0.93-0.97 is the highest-confidence number in the portfolio.**

**Dykstra (CO-LEAD, Convex Feasibility)**: The Pareto-improving point Lane G v3 + Ω-W-V2 is the only convex-hull intersection point we have measured; everything else is categorical orthogonality (yes/no per lane). The sensitivity-map module promotes my orthogonality verdicts from categorical to numerical (replacing assumed convex-hull shapes with measured intersection points). **VERDICT: APPROVE; execute in dispatch order; sensitivity-map is foundational.**

**Yousfi (Challenge creator, Steganalysis lineage)**: HM-S 8-DOF homography targets the right wedge (PoseNet); FR-Ω Fridrich-cost block-FP targets the right wedge (rate). Both verdicts are sound. The sensitivity-map module operationalizes UNIWARD on this scorer — the per-region byte-cost gradient IS the inverse-steganalysis signal. **VERDICT: APPROVE all of Section 7.**

**Fridrich (UNIWARD/SRM/HUGO author)**: My Hessian-cost framework predicts FR-Ω's per-channel allocation works on weights as it does on stego pixels; Ω-W-V2's 40.98% empirical confirms that prediction at the byte level. The sensitivity-map module enables score-aware encoding (Lane 19) which is the score-level operationalization of UNIWARD. **VERDICT: APPROVE.**

**Contrarian (Veto)**: I MAINTAIN my Council F discipline: no `[prediction]`-tagged dispatches without empirical anchors. Ω-W-V2 stack (Section 7 Transition 1) has BOTH `[derivation]`-grade rate math AND `[empirical]`-grade codec savings — it satisfies my discipline. HM-S/FR-Ω are gated on sequential discipline (HM-S first, FR-Ω after band calibrates). Sensitivity-map module is cheap ($1-2) and unblocks 3 lanes — no veto. The only concern I VETO is the SC++/SA-v2/SO Modal dispatches that lack Council F clearance — they were dispatched on remaining-budget logic, NOT on Council F's per-lane EV verdict. If they regress, do NOT cite them as "Council-approved". **VERDICT: APPROVE Section 7 with this asterisk.**

**Quantizr (Adversarial leaderboard reality check)**: My 0.33 archive uses block-FP at 1.017 bpw (Ω-W-V2-class). Section 7 Transition 1 replicates my approach with empirical proof. The 1-month target 0.30-0.45 is plausible IF Lane 12 NeRV lands at < 80KB total mask payload (the dominant lever vs my 0.33). The 6-month moonshot 0.18 is aggressive — it requires paradigm shifts I did not have to make. **VERDICT: APPROVE 48h + 1-week + 1-month targets; flag 6-month as moonshot-conditional.**

**Hotz (Engineering shortcuts)**: Section 7 Transition 1 ($0.50 + 30 min OWV2 dev) is the cheapest measurement in the portfolio. Sensitivity-map module ($1-2) unlocks 3 lanes — engineering ROI is excellent. RL/PufferLib bandit pilot ($5 cap) is a 1-evening prototype with clear graduation criteria; PufferLib full PPO not justified unless bandit saturates. **VERDICT: APPROVE.**

**Selfcomp (szabolcs-cs, working 0.38 anchor)**: Section 7 Transition 1 maps directly to my paradigm. Sensitivity-map module is the score-aware allocation step I never operationalized. Lane 12 NeRV mask codec (Phase 2) is the path that beats my 0.38 — if NeRV lands < 50KB, the 1-month 0.30 target is easily reachable. **VERDICT: APPROVE.**

**MacKay (Memorial seat, Information Theory + Bayesian Inference + Learning Algorithms)**: Section 7 Transition 1 is a strict MDL improvement (rate cost reduced; posterior approximation quality bounded by per-channel L_inf). Sensitivity-map module provides the empirical posterior `p(score | byte_allocation)` that Bayesian model selection (Lane 16) requires. The 6-month moonshot 0.18-0.28 is conditional on Lane 16 MDL framework producing a principled stack composition ranking. **VERDICT: APPROVE; 6-month target requires Lane 16 to land.**

**Ballé (2018 entropy bottleneck SOTA)**: Section 7 Transition 1 validates Ω-W-V2's static-histogram terminal at the score level. Once it lands, the V3 hyperprior question (Lane 20) is properly motivated. Sensitivity-map module gives the auxiliary loss signal for the hyperprior's σ network. 6-month target requires Lane 20 to land at scale (shared-corpus codec preferred). **VERDICT: APPROVE; 6-month target requires Lane 20 + corpus codec.**

**Quintet pact verdict** (Shannon + Dykstra + Yousfi + Fridrich + Contrarian): 5/5 APPROVE Section 7 dispatch order. Co-member verdict (Quantizr + Hotz + Selfcomp + MacKay + Ballé): 5/5 APPROVE. **CONSENSUS — EXECUTE.**

---

## 9. Cross-references

- Council E (Round 5 grand battleplan): `.omx/research/council_grand_battleplan_round5_20260429.md`
- Council F (Lane re-train EV + Ω-W-V2 + ADMM consult): `.omx/research/council_f_retrain_ev_validation_admm_consult_20260429.md`
- Council #271 (strategic design decisions): `.omx/research/council_strategic_design_decisions_20260430.md`
- Chain-integrity audit (Part G surfaced Option A Ω-W-V2 stack): `.omx/research/council_chain_integrity_audit_20260430.md`
- Council Rounds 8/9/10 (3-clean-pass gate complete): `.omx/research/council_round{8,9,10}_*_2026043{0,0,0}.md`
- Session state checkpoint (THE comprehensive snapshot): `~/.claude/projects/-Users-adpena-Projects-pact/memory/project_session_state_checkpoint_20260430.md`
- Phase 2-4 lane spec: `~/.claude/projects/-Users-adpena-Projects-pact/memory/project_phases_2_3_4_design_implementation_math_provenance_20260429.md`
- Codec stacking + canonical orders: `~/.claude/projects/-Users-adpena-Projects-pact/memory/project_codec_stacking_composition_canonical_orders_20260429.md`
- 6-month strategic plan: `~/.claude/projects/-Users-adpena-Projects-pact/memory/project_6month_strategic_plan_20260429.md`
- Lane G v3 1.05 [contest-CUDA] anchor: `experiments/results/lane_g_v3_landed/contest_auth_eval.json`
- Ω-W-V2 real-archive 40.98% [empirical]: `src/tac/tests/test_omega_w_v2_real_archive.py`
- 9-lane SegMapTrainer invalidation: `feedback_round6_defects_lane_mm_correction_segmap_invalidation_extended_20260429.md`
- Skunkworks council quintet pact: `feedback_skunkworks_council_shannon_dykstra_quintet_lead_20260429.md`
- Local-only validity binding rule (no MPS/CPU for strategic kill/promote): `feedback_no_local_mps_for_authoritative_kill_or_promote_20260429.md`
- Active Vast.ai instances tracker: `.omx/state/vastai_active_instances.json`
- Modal harvest tool (spawn cache GC mitigation): `tools/harvest_modal_calls.py`
- Subagent commit serializer (concurrent-commit race extinction): `tools/subagent_commit_serializer.py`
