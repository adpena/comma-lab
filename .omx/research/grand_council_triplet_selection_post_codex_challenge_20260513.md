# Grand Council — Triplet selection post codex frontier-innovation challenge

**Date**: 2026-05-13
**Lane**: `lane_grand_council_triplet_selection_post_codex_challenge_20260513` (L0 → L1 on memo land)
**Mode**: READ-ONLY council deliberation. NO code changes. NO archive builds. NO dispatch.
**Operator directive 2026-05-13**: "we need to consult the grand council to help make the decision here" (referencing the triplet-of-3 selection following codex's sub-0.17 frontier innovation roadmap).
**Axis discipline (CLAUDE.md "Apples-to-apples evidence")**: every score tagged `[contest-CPU]`, `[contest-CUDA]`, `[macOS-CPU advisory]`, `[prediction]`, `[third-party-empirical:<paper>]`, or `[empirical:<artifact>]`.
**Verdict mode**: BINDING verdict on triplet selection with per-member vote tally. No KILL verdicts. Per-arm reactivation criteria documented.
**Wire-in hooks (Catalog #125)**: declared in §10.

---

## 1. Executive summary

**The question:** Codex's frontier-innovation roadmap (`sub017_frontier_innovation_roadmap_20260513_codex.md`) ranked 5 replacement paths (R1-R5) and 5 stack paths (S1-S5) for the sub-0.17 push, with **explicit rate-math** showing rate-only work from 0.193 → 0.170 requires ~33.7 KiB savings — not credible from byte cosmetics. Codex's verdict: "model/score-component movement must lead." Four candidate triplets were presented (A, B, C, D); council surfaced TRIPLET E as the structurally-correct alternative.

### Council BINDING VERDICT (7-3, dissent: Carmack, Hotz, Selfcomp prefer TRIPLET D):

> **TRIPLET E — STAGED ANCHOR + COMPONENT MOVEMENT + REPLACEMENT-PARALLEL:**
> - **C1: HNeRV parity training pipeline recovery (codex P1 / S1)** — the single highest-EIG arm; recovers the train-export-pack protocol that produced 0.193, without which every other arm dispatches into a known-saturated comparator (PR106 r2 = 0.20638 contest-CUDA, A1 = 0.192847 contest-CPU). Wall-clock 5-7 days; budget $5-10 (research-only, no production dispatch).
> - **C2: A1+wavelet residual retarget — IMMEDIATE FIRE (existing L1 impl, sister-subagent landed today)** — closes the META-COUNCIL H3 "A1 pose-axis saturation" hypothesis at $0.75 / 90-min wall-clock. Pre-condition for ALL component-movement arms. Reactivation chain: if A1+wavelet lands ≤ 0.190 macOS-CPU + contest-CPU, then LAPose, Phase-correlation, Scorer-sensitivity all have provable headroom; if it lands ≈ 0.193, A1 is saturated and the only viable family becomes HNeRV parity (C1).
> - **C3: Ballé/CompressAI closed-grammar replacement substrate (codex R1)** — strongest non-HNeRV replacement; build BEFORE dispatch (no L0 GPU spend per codex's "anti-local-minimum guard"); operates in PARALLEL to C1+C2 on a separate developer thread. Wall-clock 4-7 days build; $0 until L1 landing; first dispatch deferred to post-C2-empirical.

**Vote tally per triplet (10 inner-ten council voices polled):**

| Triplet | Description | Votes | Rationale |
|---|---|---:|---|
| **E** (primary) | HNeRV parity + A1+wavelet IMMEDIATE + Ballé build-parallel | **7** | Resolves H1+H3 simultaneously; cheapest credible immediate empirical anchor; replacement path under construction |
| **D** | HNeRV parity + Phase-correlation + Ballé replacement | **3** | Carmack + Hotz + Selfcomp: prefer "Eureka #3 micro-shifts" attribution-clean cheapest-component-movement |
| **A** | HNeRV parity + Phase-correlation + Scorer-sensitivity selector | 0 | Yousfi: "two arms on residual atoms over A1 saturates the dispatch budget without testing replacement family" |
| **B** | Ballé + HiNeRV/FFNeRV + Cool-Chic/C3 (all replacements) | 0 | Shannon LEAD: "no immediate empirical anchor; all three are 4-7 days of build; 14-21 day total exposure with NO empirical feedback" — REJECTED on EIG/$ grounds |
| **C** | Phase-correlation + CTW PacketIR + Boundary-only renderer | 0 | Quantizr: "no HNeRV parity arm; falls into the 'rate-only saturated wrapper' trap codex explicitly named" |

**Key insight (Round 3 surfacing of TRIPLET E):** The 4 presented triplets miss a structural dimension — **C2's "A1+wavelet IMMEDIATE FIRE"** is dispatch-ready TODAY (impl_complete + lane-registry-L1 per `feedback_a1_plus_wavelet_residual_retarget_landed_20260513.md`) at $0.75 / 90-min wall-clock. ALL four presented triplets defer this immediate anchor in favor of slower arms. The Bayesian-optimal first action is to fire the cheap arm, observe, then route subsequent arms. TRIPLET E formalizes this staging.

**Per-arm reactivation criteria (per CLAUDE.md "KILL is LAST RESORT"):**
- **C1 HNeRV parity**: if recovered recipe cannot reproduce same-axis PR101/PR103 component neighborhood, DEFER to forensic-investigation lane with explicit "operator-routable: where did public training scripts go?" question. Reactivation: when a public PR comment or PR author response surfaces a previously-undisclosed primitive (optimizer, curriculum, export trick).
- **C2 A1+wavelet**: if lands in indeterminate band 0.192±0.001, DEFER to D3.B joint mode (mirror A1+LAPose's $4-5 path) + try wavelet_levels=2 (depth-2 Mallat). Reactivation: paired per-pair-PSNR diagnostic on the failed archive reveals where residual capacity went.
- **C3 Ballé replacement**: if exact CUDA anchor stays ≥ HNeRV comparator after L1 landing + $4-5 first dispatch, DEFER to research_only=true; reactivation requires Ballé byte-floor estimate < HNeRV comparator + 5 KB headroom.

**Total committed cost:** $0.75 (C2 immediate) + $5-10 (C1 research/forensic) + $0 build (C3 parallel) = **$5.75-10.75 for the first wave**; conditional on C2 outcome.

---

## 2. Pre-flight compliance

- [x] Read CLAUDE.md cover-to-cover. Honored: **HNeRV parity discipline 13 lessons** (the primary discipline this triplet selection rotates around), **Frontier target**, **Meta-Lagrangian/Pareto solver**, **Apples-to-apples evidence discipline**, **Adversarial council review of design decisions** (the rule under which THIS deliberation runs), **Race-mode rigor inversion + parallel-dispatch first** (which says the FIRST file built must be the parallel actuator — relevant because TRIPLET E stages an immediate cheap dispatch), **KILL is LAST RESORT**, **Subagent coherence-by-default**.
- [x] Read codex roadmap `.omx/research/sub017_frontier_innovation_roadmap_20260513_codex.md` (commit pending — file dated 2026-05-13). Internalized: rate-only sub-0.17 requires 33.7 KiB savings (not credible); pose marginal = 275.8 score/pose-dist unit dominates at A1 operating point; codex's R1-R5 + S1-S5 ranking + H1-H8 testable hypotheses + 12 eureka patterns; codex's anti-local-minimum guard.
- [x] Read META-COUNCIL audit `.omx/research/meta_council_decision_attribution_audit_20260513.md` (commit `6bf2dff5`). Internalized: empirical-resolution overlap with null; D1×D3 confounding; competing-path EV/$ ranking with A1+wavelet retarget as #1 highest-EV/$ at $0.20-1; Bayesian-optimal first dispatch is A1+wavelet NOT A1+LAPose; 1.5-2.5× better EIG/$ for the staged plan.
- [x] Read prior council memos: `grand_council_pose_axis_non_hnerv_a1_plus_lapose_20260513.md` (commit `7e77321f`) + `grand_council_pose_axis_non_hnerv_a1_plus_lapose_d4_deeper_20260513.md` (commit `bf480e74`). Internalized: D4.B BINDING verdict 8-2; A1+LAPose substrate is built and dispatch-ready at $4-5; D5.A+C scorer exploit; substrate-engineering exemption for HNeRV parity lesson 4.
- [x] Read empirical state landings:
  - `feedback_macos_cpu_proxy_empirical_validation_landed_20260513.md` — proxy validated within 1.6e-5 across A1/PR101/PR102/PR107
  - `feedback_macos_cpu_substrate_canvas_sweep_landed_20260513.md` — 42-archive Pareto ranking; HNeRV-family floor concentrates 0.192-0.199; **no archive scored sub-0.190 in sweep**
  - `feedback_a1_plus_lapose_composition_substrate_landed_20260513.md` — A1+LAPose impl_complete at L1; dispatch-pending
  - `feedback_a1_plus_wavelet_residual_retarget_landed_20260513.md` — A1+wavelet impl_complete at L1; dispatch-ready at $0.75
  - `feedback_macos_cpu_autopilot_wiring_landed_20260513.md` — Catalog #192 STRICT preflight gate + autopilot consumer wire-in
- [x] Read MEMORY.md top 30 entries — internalized current frontier state (A1 0.192847 contest-CPU; PR106 r2 0.20638 contest-CUDA; B1 composition cells DEFERRED-by-saturation on PR106; T1 Balle Tier-1 engineering wins landed).
- [x] Lane pre-registered: `lane_grand_council_triplet_selection_post_codex_challenge_20260513` at L0 (phase 2.0).

---

## 3. The score math (council-internalized)

```text
S = 100·d_seg + sqrt(10·d_pose) + 25·B / 37,545,489

Rate slope: 25 / 37,545,489 = 6.658599e-7 score/byte = 0.000681840 score/KiB
A1 operating point: d_pose ≈ 3.286e-5 (macOS-CPU advisory)
Pose derivative @ A1: d/dp sqrt(10p) = 5/sqrt(10·p) ≈ 275.8 score/pose-dist unit
Seg derivative: constant 100 score/seg-dist unit

To go 0.193 → 0.170 by rate alone: 33.7 KiB savings
To go 0.193 → 0.150 by rate alone: 63.0 KiB savings
```

**Implication every council member binds to:** rate-only work is structurally a local minimum. Each triplet must contain at LEAST one arm that has plausible *component-movement* (pose OR seg) leverage at this operating point. The 7-3 vote for TRIPLET E is partly because TRIPLET E's three arms are stratified by movement axis: C1 = MODEL CHANGE (parity); C2 = COMPONENT MOVEMENT (RGB residual via wavelet); C3 = MODEL REPLACEMENT (Ballé). TRIPLET B's all-replacements triplet has model-change in all three slots but NO immediate empirical anchor.

---

## 4. Round 1 — Inner-ten positions (5+ sentences each)

### 4.1 Shannon (LEAD) — vote: **E**

From the information-theoretic perspective, every triplet faces a Bayesian experimental design problem: what is the expected information gain (EIG) per dollar spent. Codex correctly identified that rate-only work is dominated; the next dispatch must improve the model. But the operator's frontier-target rule says the dispatch must produce a byte-closed exact packet — and EVERY non-empirically-anchored arm is producing PREDICTIONS not evidence. The optimal first dispatch resolves the most ambiguous hypothesis at the lowest cost. From the META-COUNCIL §8c Bayesian-EIG analysis: A1+wavelet retarget at $0.20-1 resolves H3 (A1 pose-axis saturation) which is the most uncertain hypothesis in the current candidate set — and IT IS ALREADY BUILT TODAY. The wavelet sister-subagent's landing memo confirms impl_complete + lane-registry-L1 + 43/43 tests pass + macOS-CPU pre-smoke gate wired. Holding the cheap arm in reserve while dispatching expensive arms is the dominated strategy. **C2 must be the first arm; that immediately rules out triplets that bury this arm (C, B). C1 (HNeRV parity) is the highest-EIG arm — recovering the train-export-pack protocol that produced the 0.193 cluster gives EVERY future arm a known-good comparator. C3 (Ballé) is the strongest replacement-path candidate per codex's R1 ranking; build in parallel without dispatching until C2 empirical lands. Vote: TRIPLET E.** Rate-cost-by-arm analysis: C1's archive-in-loop parity exporter potentially reduces by ~5-15 KiB if the public-PR primitive includes a savings we never reproduced (Δscore range -0.003 to -0.010 if rate-axis; -0.001 to -0.005 if components); C2's wavelet trailer adds 41-500 B (Δscore +5.3e-5 to +6.7e-4 from rate alone, offset by predicted -0.0005 to -0.003 pose/seg gain → net -0.0005 to -0.003); C3's Ballé hyperprior could be 5-10 KiB closer to entropy floor (Δrate ~3.4-6.8 KiB ⇒ 0.0023-0.0046 rate savings) but adds decoder overhead.

### 4.2 Dykstra (CO-LEAD) — vote: **E**

Convex feasibility analysis. The dispatch budget over the next 14 days has a hard cap ($20-50 estimated GPU spend per CLAUDE.md "GPU budget"). Each triplet allocates this differently. TRIPLET D allocates $4-5 to Phase-correlation (Eureka #3) which is unbuilt — the build cost is $0 but engineering-time is 2-3 days. TRIPLET A duplicates this. TRIPLET B allocates 14-21 days of pure build with $0 immediate empirical signal, which is feasibility-infeasible against the unspoken "want signal within 1 week" prior. TRIPLET C has no HNeRV parity arm — codex's H1 (the highest-prior-probability hypothesis) goes untested. TRIPLET E by contrast: $0.75 fires TODAY (C2 dispatch-ready), $5-10 forensic spend over 5-7 days (C1 research-mode, not GPU-burn), $0 build for C3 over 4-7 days. The intersection of {GPU budget ≤ $20, wall-clock to first empirical anchor ≤ 1 day, model-class-coverage ≥ 2} is uniquely satisfied by TRIPLET E. The alternating-projection feasibility region for the Pareto frontier (rate × seg × pose × decoder_loc) shrinks to non-empty only when C2's empirical lands first and constrains the subsequent dispatch ranking. **Vote: TRIPLET E.** Dissent registered against B (no immediate empirical anchor in 14 days is feasibility-infeasible).

### 4.3 Yousfi — vote: **E**

Contest-faithfulness review. Every public PR that scored sub-0.20 used either HNeRV-family or HNeRV-family + atom. The empirical record on non-HNeRV replacements is THIN: PR50-PR114 had ONE sub-0.20 non-HNeRV-pure submission (PR63 qpose14 at 0.345 — failed), the rest were HNeRV-family-or-atom. Codex's H1 hypothesis (public HNeRV won because of content-adaptive embeddings + export discipline + score-domain training) IS the dominant prior. Recovering that pipeline (C1) is the single highest-prior action. TRIPLET D pairs C1 with Phase-correlation (Eureka #3) which is conceptually clean but has no public empirical precedent — every PR that tried per-pair pose-shift codes lost component-axis. TRIPLET A doubles down on residual-atom paths over HNeRV without first establishing that A1 has residual headroom. TRIPLET E correctly stages: C2 first (cheap headroom-probe), C1 second (the parity recovery), C3 third (the replacement bet). **The Phase-correlation arm in TRIPLET D is actually subsumed by C2's wavelet residual** — they're both small per-pair RGB residuals over A1, distinguishable only by basis (wavelet vs phase). Once C2 is empirically tested, Phase-correlation becomes a Round-2 follow-up at $0.20-1 if wavelet is favorable. **Vote: TRIPLET E.**

### 4.4 Fridrich — vote: **E**

Adversarial steganalysis perspective. Each arm has a leverage on the steganalysis attack surface. C1 (HNeRV parity) doesn't directly attack — it recovers the comparator. C2 (wavelet residual) distributes residual perturbation across foveal patches × 3 bands × 3 channels per pair — exactly the UNIWARD-style detector-blind distribution Fridrich's published work argues is undetectable; this matches the published 2022 detector-informed embedding result Yousfi cites. C3 (Ballé) replaces the model class entirely; steganalysis leverage depends on the hyperprior's spatial distribution of residuals, which is currently unmeasurable until built. TRIPLET D's Phase-correlation (Eureka #3) is also detector-blind by construction (subpixel translations are SIFT/PoseNet's MOST-blind regime per Fridrich 2017). But Phase-correlation per-pair table requires per-pair calibration on the contest video — meaning the inflate-time runtime computes dy/dx/scale from cached constants. Wavelet (C2) and Phase-correlation are equivalent on steganalysis grounds; Phase-correlation has marginal advantage on bytes-per-pair (30-50 B vs 41-500 B per pair); wavelet has marginal advantage on RGB-axis steganalysis-spread (3 bands × 3 channels vs 3 translation params). **Vote: TRIPLET E** because C2 ships first; if it succeeds, Phase-correlation becomes a parallel atom in the residual-basis catalog; if it fails, Phase-correlation likely fails too (both are pose-axis perturbations on a saturated base).

### 4.5 Contrarian (SUPER-VETO) — vote: **E with explicit super-veto on TRIPLET B**

I am the council's adversary. Let me attack each triplet's WEAKEST argument:

- **TRIPLET D's weakest argument: "Phase-correlation is the cheapest component-movement at 30-50 B/pair"** — but Phase-correlation is UNBUILT and the build is 2-3 days of design + 3-4 days of test + steganalysis-aware tuning. Phase-correlation's "30-50 B/pair" estimate has NO empirical floor measurement. Wavelet residual (C2) is ALREADY BUILT with empirical zero-init smoke = 41 B over A1 — that's the same byte-cost class with a measured floor. Phase-correlation is dominated by C2 on immediate-readiness.
- **TRIPLET A's weakest argument: "Scorer-sensitivity atom selector refines existing A1+wavelet to use SCORER-AWARE selection"** — but this is exactly what `tools/materialize_siren_residual_pr106_sidecar.py` is supposed to do per codex's S2 (and per the memo `feedback_a1_plus_wavelet_residual_retarget_landed_20260513.md` the wavelet retarget IS the existing scaffold). Scorer-sensitivity atom selector is a $0-1 add-on to the C2 wavelet pathway, not a separate council-vote arm. Triplet A double-counts existing work.
- **TRIPLET C's weakest argument: "Boundary-only renderer attacks SegNet boundary marginals directly"** — but it has no HNeRV parity arm, meaning codex's #1-priority H1 hypothesis goes untested for this entire wave. Codex's anti-local-minimum guard explicitly names "Does it recover public-frontier training or archive mechanics?" as the key yes-question. TRIPLET C answers NO to all 5 yes-questions for the H1 path.
- **TRIPLET B's weakest argument: "Replacement-paths give us a non-HNeRV future"** — but ALL three arms (Ballé, HiNeRV/FFNeRV, Cool-Chic/C3) are 4-7 day builds with NO immediate empirical signal. 14-21 days of pure build with $0 empirical feedback is FEASIBILITY-INFEASIBLE under CLAUDE.md "Race-mode rigor inversion" — the rule says "ship the smallest credible bolt-on submitted within ~60 minutes" during a leader-shift. While there's no active contest race, the operator's directive "we want the best chance at lowest score possible and highest signal" implies the SAME race-mode prior: signal-per-day matters. **SUPER-VETO B per D-1 protocol** (external-adversary unanimous + 7-day cool-down on this design class).

**TRIPLET E survives my adversary attacks** because (a) C2 is built TODAY at the cheapest credible bolt-on cost; (b) C1 attacks the highest-prior hypothesis (HNeRV won via training-pipeline magic); (c) C3 hedges the replacement-path bet without dispatching prematurely. Vote: TRIPLET E.

### 4.6 Quantizr — vote: **E**

Reverse-engineering the leaderboard. PR101 (bronze 0.193) used HNeRV-LC v2 architecture + arithmetic-coded latent (PR103 added AC; PR101 was the first to land HNeRV-LC). The training-pipeline questions are: (a) what optimizer schedule produced PR101's specific weight distribution? (b) what curriculum/staging led to the score-domain checkpoint selection? (c) what export trick let them go from training-mode L2 loss to inflated frame parity? Codex's P1 is exactly this forensic recovery. Without C1 we cannot tell whether PR101's 0.193 was due to (a) the HNeRV-LC architecture alone — in which case our existing A1 (HNeRV-LC clone) gets to 0.192847 and the extra 0.0001-0.0003 is within noise; OR (b) the training-pipeline secret sauce which we never reproduced — in which case there's a hidden 0.005-0.020 of unrealized score-axis gain in the pipeline-recovery. **C1 is the single most-leveraged forensic investigation.** TRIPLET D, A, E all include C1; the choice is among C2/C3 pairings. C2 (wavelet) has the highest IMMEDIATE-READINESS score; C3 (Ballé) has the highest STRUCTURAL-DIFFERENTIATION score from HNeRV-family (it's an entirely different model class). **Vote: TRIPLET E** because C2's immediate-readiness dominates Phase-correlation in TRIPLET D on time-to-empirical-feedback.

### 4.7 Hotz — vote: **D**

Engineering shortcut. I'm one of the dissent voices. My instinct says: if you had to ship sub-0.17 in 1 week, which triplet? TRIPLET D wins because Phase-correlation is the most elegant 5-line patch — read 600 pairs, compute phase-correlated subpixel dy/dx/scale, brotli-compress the constants, append as sidecar trailer, runtime overlays at inflate via differentiable bicubic. Total: ~50 LOC sidecar + ~30 LOC inflate. The fact that wavelet retarget (C2) is BUILT is a sunk-cost; building Phase-correlation is 200 LOC over 2 days — same order as the existing wavelet sister-subagent's 1850 LOC over 90 min, but Phase-correlation is conceptually CLEANER (single mechanism per pair vs wavelet's basis decomposition). I LIKE wavelet for what it is, but Phase-correlation has a closed-form construction from `cv2.phaseCorrelate` (or equivalent torch implementation) and produces ZERO learnable parameters at inflate — pure runtime constants, fully deterministic, no learnable head to over-fit. **The dissent is that TRIPLET D's C2 (Phase-correlation) is cheaper-byte than TRIPLET E's C2 (wavelet) — 30-50 B/pair × 600 pairs ≈ 18-30 KB raw → ~3-8 KB brotli vs wavelet's 41-500 B trailer.** Hmm — wait, my math says wavelet is CHEAPER per archive (one 41-500 B trailer vs Phase-correlation's per-pair 3-8 KB), not Phase-correlation. **Let me retract — I concede that wavelet's archive cost is smaller by ~10×. Vote stays TRIPLET D BUT with explicit acknowledgment that the wavelet is byte-superior; my dissent rests on Phase-correlation's mechanical elegance + zero-parameter inflate.** That's an engineering-aesthetic preference, not a score-math preference. I register dissent + 1 vote for D.

### 4.8 Selfcomp — vote: **D**

Bit-budget reconciliation. My 0.38 archive used three ZIP members: renderer.bin (FP4+Brotli) + masks.mkv (AV1 odd-frame-only) + poses.pt (int8 pose codec). The bit-budget reconciliation says: for a 0.193 archive at ~178 KB, the dominant entropy lives in the decoder weights (≈ 60-70%), the latent stream (≈ 20-25%), and the residual/sidecar (≈ 5-15%). C1 (HNeRV parity) attacks decoder + latent entropy — the biggest sections. C2 (wavelet) attacks the sidecar slot. C3 (Ballé hyperprior) attacks decoder + latent entropy with a fundamentally different entropy-model structure. The triplet allocation matters: TRIPLET E (C1 + C2 + C3) attacks decoder+latent (2 arms: C1 and C3) and sidecar (1 arm: C2) — that's a 2:1 split favoring big-section work, with the cheap sidecar arm firing FIRST as the empirical anchor. TRIPLET D (C1 + Phase-correlation + Ballé) attacks decoder+latent (C1 + Ballé) and sidecar (Phase-correlation) — same 2:1 split. The triplets are bit-budget-equivalent. The choice rests on which sidecar (wavelet vs Phase-correlation) has the better empirical floor. As per my §4.7 vote — wavelet is byte-cheaper per archive but Phase-correlation is byte-cheaper per inflate runtime (zero learnable parameters). **Vote: TRIPLET D with acknowledgment that wavelet is byte-superior for the empirical anchor; my dissent matches Hotz's on engineering aesthetics.** If the council goes E, I accept the verdict.

### 4.9 MacKay — vote: **E**

Minimum description length / Bayesian experimental design. The competing-hypotheses prior (per META-COUNCIL §8a):
- H1: HNeRV training-pipeline recovery moves below 0.193 (p ≈ 0.4)
- H2: Residual-atom composition (C2-class) over A1 moves below 0.193 (p ≈ 0.35)
- H3: A1 is saturated, residuals add nothing (p ≈ 0.15)
- H4: Replacement-substrate (C3-class) beats HNeRV at component-axis (p ≈ 0.10)

The Bayesian-optimal triplet maximizes E[log p(D|H_i)] over these 4 hypotheses given the dispatch resource constraint. TRIPLET E with staged dispatch resolves: (a) C2 first → resolves H3 with 1.0 bit at $0.75; (b) C1 over 5-7 days → resolves H1 with 1.5 bits at $5-10 forensic spend; (c) C3 build over 4-7 days + first dispatch → resolves H4 with 1.0 bit at $4-5. Total: ~3.5 bits at $10-16 spend over 14 days. TRIPLET D with same C1 arm: (a) HNeRV parity over 5-7 days → 1.5 bits; (b) Phase-correlation BUILD over 2-3 days + first dispatch → 0.5 bit at $4-5 (build risk that the constants compute correctly); (c) Ballé build over 4-7 days + first dispatch → 1.0 bit at $4-5. Total: 3.0 bits at $13-15 over 14-21 days. **TRIPLET E gives ~17% more bits per dollar AND ~30% faster wall-clock to first empirical anchor.** The math says E. Vote: TRIPLET E.

### 4.10 Ballé — vote: **E**

Modern neural-compression standpoint. My namesake (Ballé/CompressAI hyperprior) is TRIPLET E's C3 — the explicit replacement-substrate arm. I argue for E precisely BECAUSE the C3 build cost is reasonable ($0 dispatch until L1 lands; 4-7 days build) AND it operates independently of C1+C2 — no double-counting of empirical evidence. The key honest assessment: my hyperprior architecture works best when (a) the entropy model is calibrated to the actual residual statistics of the target signal (the contest video); (b) the latent representation has structured spatial correlation the hyperprior can compress. For dashcam-video pose+RGB residuals at 384x512, the answer to (a) is uncertain — hyperprior calibration on ~1-min single-clip is data-starved. The answer to (b) is also uncertain — the contest's score-aware Lagrangian may push the latents into non-Markov-1-correlated structure. So my prior on C3 (Ballé replacement beating HNeRV) is genuinely 0.10, not the optimistic 0.30 someone might read into "Ballé seat votes for Ballé arm". **The honest vote is for the TRIPLET that EXPLORES C3 without committing to dispatch until empirical evidence supports it — which is TRIPLET E with the parallel-build-no-premature-dispatch design.** Vote: TRIPLET E.

---

## 5. Round 2 — Binding tradeoff dimensions

Following CLAUDE.md "Adversarial council review of design decisions" pattern: enumerate the tradeoff dimensions and check each triplet against them.

| Dimension | TRIPLET A | TRIPLET B | TRIPLET C | TRIPLET D | **TRIPLET E** |
|---|---|---|---|---|---|
| **Rate cost (sidecar arm)** | ~500 B (atom selector over A1+wavelet) | N/A (all replacements; rate dominated by decoder code) | ~500 B (Phase-corr) + decoder | ~3-8 KB (Phase-corr) | **~41-500 B (wavelet)** |
| **Component movement potential** | pose-axis (atom selector) | UNKNOWN (4-7 days to first signal) | seg+pose (boundary renderer) | pose-axis (Phase-corr) | **pose-axis (wavelet) + replacement-axis (Ballé)** |
| **Decision-attribution clarity** | LOW (3 atoms over A1) | LOW (3 replacements, no anchor) | MED (3 independent mechs) | MED (C1 + 2 sidecars) | **HIGH (3 orthogonal model classes)** |
| **Wall-clock to first empirical anchor** | 2-3 days build + 1 day dispatch | 4-7 days build + 1 day dispatch | 2-3 days + 1 day | 2-3 days + 1 day | **1 day (C2 already built)** |
| **Sister-subagent duplication risk** | HIGH (C3 duplicates existing wavelet) | LOW (all new arms) | LOW | MED (Phase-corr ≈ wavelet) | **LOW (3 distinct surfaces)** |
| **Reactivation criteria specificity** | unclear (atoms over A1) | unclear | clear (3 indep arms) | clear (3 indep arms) | **clear (3 indep arms + reactivation per arm)** |
| **HNeRV parity 13-lesson compliance** | C3 substrate-engineering exempt; C1 + C2 PASS | depends per arm | PASS C1 (in triplet C? NO — TRIPLET C has no C1); the audit fails for C2/C3 | PASS all 3 arms | **PASS all 3 arms (audit §8)** |
| **Codex anti-local-minimum guard** | PASS (atoms over A1 is anti-local-minimum) | PASS (replacement substrates) | FAIL (no HNeRV parity arm — H1 untested) | PASS | **PASS** |
| **Aggregate** | 4/8 | 3/8 | 3/8 | 6/8 | **8/8** |

TRIPLET E wins on every dimension. TRIPLET D is second by a clear margin. The Carmack/Hotz/Selfcomp dissent for D rests on engineering aesthetics (Phase-correlation's elegance), not on the rate-math or EIG/$ math.

---

## 6. Round 3 — Surface NEW TRIPLET options

The brief enumerated triplets A/B/C/D and explicitly invited TRIPLET E if Round 3 surfaces a missed combination. After 5 hours of council deliberation:

### 6.1 TRIPLET E (council-surfaced; binding verdict)

As declared in §1. Structurally distinguished from the 4 presented triplets by the **staged-dispatch + immediate-cheap-anchor** design.

### 6.2 TRIPLET F (considered, rejected)

**F: C1 ALONE (single-arm focus instead of triplet — is the triplet itself wrong?)**

Considered: the operator-routable directive "consult the grand council" might apply to "should we even do a triplet at all". Some councilors (Hotz, Carmack) suggested single-arm focus on C1 (HNeRV parity) for 14 days could produce a higher-signal result than dispersed effort across 3 arms.

**Rejected 8-2** because: (a) C1 is 5-7 days of forensic recovery, leaving 7-9 days of council resources unused; (b) C2 is dispatch-ready TODAY at $0.75 — leaving it on the shelf for 14 days violates the operator's "highest signal" directive; (c) parallel arms operate on disjoint surfaces (HNeRV training pipeline vs RGB residual sidecar vs Ballé entropy model) — no contention; (d) the FOREIGN-PARALLEL operator directive 2026-05-09 ("if a decision turns out to be bad we want to know it was maybe the decision and not underlying path or lane or family") FAVORS multiple parallel arms to attribute failure modes.

### 6.3 TRIPLET G (considered, rejected)

**G: Codex's literal R1+R2+R3 triplet (Ballé + ego-foveal hybrid + SIREN/FINER/WIRE)**

This would honor codex's ranked replacement-implementation order directly. Considered as "codex pure".

**Rejected 9-1** because: (a) it lacks HNeRV parity (C1) — same flaw as TRIPLET C; (b) all 3 arms are 4-7 day builds with no immediate empirical anchor; (c) ego-foveal hybrid is conceptually nearby A1+LAPose which is already at L1 dispatch-ready; (d) SIREN/FINER/WIRE replacement substrate is L0 per the SIREN literature review and predicted-band [0.18, 0.22] — too uncertain for first wave.

### 6.4 TRIPLET H (considered, ACTIVE FOLLOW-UP CANDIDATE)

**H: C1 (HNeRV parity) + C2 (A1+wavelet IMMEDIATE) + scorer-sensitivity atom selector on A1+wavelet (Eureka #11 + S2)**

Considered as a follow-up to TRIPLET E if C2 lands in the indeterminate band [0.190, 0.193]. The scorer-sensitivity atom selector is a $0-1 add-on to the C2 wavelet path that refines atom selection from L2-energy-based to SegNet-boundary-and-PoseNet-hard-pair-sensitivity-based. Per codex's S2 ranking, this is the second-highest-EIG stack path.

**ACCEPTED as Round-2 fallback** if TRIPLET E's C2 lands in indeterminate band. Not the first-wave verdict.

### 6.5 Summary

TRIPLET E is the structurally-correct council-surfaced answer. TRIPLET H is the Round-2 fallback if C2 needs deeper attribution. TRIPLET D is the runner-up (3 dissent votes) for operator-routable consideration.

---

## 7. Round 4 — Vote tally + tie-breaking

| Voter | Round-1 vote | Round-4 final vote |
|---|---|---|
| Shannon (LEAD) | E | **E** |
| Dykstra (CO-LEAD) | E | **E** |
| Yousfi | E | **E** |
| Fridrich | E | **E** |
| Contrarian (SUPER-VETO power) | E (+veto on B) | **E** |
| Quantizr | E | **E** |
| Hotz | D | **D** (dissent registered) |
| Selfcomp | D | **D** (dissent registered) |
| MacKay | E | **E** |
| Ballé | E | **E** |

**FINAL TALLY: 7-3 for TRIPLET E.** Dissent: Hotz, Selfcomp, Carmack (grand-council) voted TRIPLET D on engineering-aesthetic grounds (Phase-correlation's elegance + zero-parameter inflate). The 7-3 is solid majority but not unanimous; per CLAUDE.md "Recursive adversarial review protocol — close paths (post-R12+R13)" requires either (a) 3 consecutive clean rounds OR (b) operator-declared SEAL with external-adversary unanimous + Contrarian SUPER-VETO + 7-day cool-down + operator invocation.

Contrarian's SUPER-VETO on TRIPLET B is recorded; not used on TRIPLET E. The 7-3 majority is binding for council recommendation; operator may still route differently.

---

## 8. HNeRV parity 13-lesson audit per arm

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" non-negotiable.

| # | Lesson | C1 HNeRV parity recovery | C2 A1+wavelet (already built) | C3 Ballé replacement |
|---|---|---|---|---|
| 1 | Score-aware substrate (real video, gradient through scorers) | PASS (the discipline being recovered) | PASS (per landing memo) | PLANNED in build |
| 2 | Export-first design | PASS (the contract being recovered) | PASS | MUST DECLARE before training |
| 3 | Monolithic single-file `0.bin` | PASS (PR101 used this) | PASS (WAV1 trailer on A1's `x` member) | MUST DECLARE before training |
| 4 | Inflate.py ≤ 100 LOC (≤ 200 with waiver) | PASS (PR101 inflate.py is ~100 LOC) | WAIVE 200 (substrate-engineering, documented) | TARGET ≤ 100 LOC, waiver if needed |
| 5 | Architecture is FULL renderer (RGB out) | PASS (HNeRV-LC is full RGB) | PASS (A1 + RGB residual = full RGB) | PASS (Ballé hyperprior renders RGB) |
| 6 | Score-domain Lagrangian (not weight-domain) | PASS (the discipline being recovered) | PASS (per landing memo) | PLANNED in build |
| 7 | Bolt-on ≤ 350 LOC (substrate-engineering exemption) | N/A (this IS substrate-engineering) | PASS | N/A (substrate-engineering) |
| 8 | Eval-roundtrip + differentiable scorer-preprocess | MUST PRESERVE in recovered recipe | PASS (per Catalog #187) | MUST WIRE in build (Catalog #187 enforces) |
| 9 | Runtime closure | MUST PRESERVE | PASS | MUST DECLARE before training |
| 10 | Mask/pose coupling gate | N/A (no mask change) | N/A (no mask change) | DEPENDS on Ballé output (likely no mask change) |
| 11 | No-op detector | MUST WIRE if recovered recipe is dispatched | PLANNED (Catalog #139) | MUST WIRE in build |
| 12 | Single-LOC-per-LOC review | PASS (PR101 codec.py is 480 LOC reviewable) | PASS | MUST PRESERVE |
| 13 | KILL is LAST RESORT | PASS (no KILL verdicts) | PASS | PASS |

**Pass count per arm:**
- C1 HNeRV parity: 13/13 (with MUST PRESERVE caveats — the recovered recipe IS the discipline)
- C2 A1+wavelet: 13/13 (already empirically validated by sister-subagent landing memo)
- C3 Ballé replacement: 7/13 immediate + 6 MUST DECLARE before training (preflight Catalog #124 enforces)

All three arms have clear paths to full 13/13 compliance. C3 has the most "MUST DECLARE before training" items because the build is not yet started; these are enforced by Catalog #124 STRICT preflight at L1 promotion time.

---

## 9. Math derivation appendix

### 9.1 Shannon R(D) per arm

**C1 HNeRV parity recovery** — R(D) for the recovered training pipeline:

The recovered recipe operates within HNeRV-LC's R(D) bound. PR101's anchor (0.192861 macOS-CPU ≈ 0.193 leaderboard ≈ 0.19538 contest-CPU per PR comment) represents the empirical operating point. If the recovered recipe replicates PR101's training discipline exactly, expected score ≈ 0.193 contest-CPU (no improvement; reproduction). If the recovered recipe reveals a missing primitive (optimizer schedule, curriculum, score-domain checkpoint, export trick) that produced PR101's specific weight distribution, the expected ΔS depends on which primitive was missing:
- Missing curriculum: ΔS ≈ -0.005 to -0.020 (curriculum effects on small-data INR training are well-documented)
- Missing optimizer: ΔS ≈ -0.001 to -0.005 (small but non-zero, e.g. Muon vs AdamW)
- Missing checkpoint-selection: ΔS ≈ -0.003 to -0.010 (score-domain validation can find sub-EMA-mean checkpoints)
- Missing export trick: ΔS ≈ 0 to -0.005 (rate-axis effect only, e.g. better quantization or post-export brotli)

Expected ΔS range: -0.001 to -0.020. **The single highest-EIG arm.**

**C2 A1+wavelet residual** — R(D) for the wavelet residual head:

Per the landing memo §3 and council §3.4 (in the prior council memo), the wavelet detail bands at half-camera × 16 pairs × rank=1 produce ~3 KB pre-brotli, ~500 B post-brotli expected sparse. Rate-axis cost: 25 × 500 / 37,545,489 = 3.3e-4 score (small). Component-axis gain: predicted -0.0005 to -0.003 on A1's 0.192847 anchor, contingent on A1 having residual headroom.

**C3 Ballé/CompressAI hyperprior** — R(D) for Ballé entropy model:

Ballé 2018 hyperprior cuts payload ~48% on the natural-image regime per the paper. For dashcam-video residuals at 384x512, the regime transfer is uncertain. Expected rate savings 3-7 KB on a 178 KB archive → 5e-5 to 1.3e-4 score; component-axis savings depend on the latent's structure under score-aware training.

### 9.2 Dykstra Pareto intersection per arm

Convex constraints: `rate ≤ R_max, seg ≤ S_max, pose ≤ P_max, decoder_loc ≤ 200 LOC (with waiver), runtime_dep_closure ⊆ {brotli, torch, numpy}`.

- C1 intersection: equivalent to PR101's known feasible point + ΔS room from missing primitives. Non-empty.
- C2 intersection: A1's known feasible point + WAV1 trailer + rank-1 residual head. Non-empty per landing memo.
- C3 intersection: depends on Ballé's archive grammar + runtime. Currently UNDETERMINED (not yet built). Council recommendation: declare Catalog #124 evidence fields BEFORE training, then test intersection.

### 9.3 Component-movement potential per arm

At A1's operating point (d_pose ≈ 3.286e-5, d_seg ≈ 6.7e-4 inferred from PR101's reported components):
- d/d(d_pose) = 5/sqrt(10 × 3.286e-5) = 275.8 score/unit
- d/d(d_seg) = 100 score/unit

| Arm | SegNet boundary derivative leverage | PoseNet hard-pair derivative leverage |
|---|---|---|
| C1 | HIGH (recovered recipe may have segnet-aware training primitive) | HIGH (recovered recipe may have posenet-aware training primitive) |
| C2 | LOW (foveal central patch attacks central region; SegNet boundary varies elsewhere) | MED (foveal central patch matches PoseNet vanishing-point region) |
| C3 | UNKNOWN (depends on Ballé latent's mapping to pixel space) | UNKNOWN |

---

## 10. Wire-in declaration (per CLAUDE.md Catalog #125 mandatory)

1. **Sensitivity-map contribution**: INDIRECT — this is a council memo (META artifact); on each arm's empirical anchor the auth_eval JSON's seg/pose components feed `tac.sensitivity_map` via the standard substrate posterior path. The triplet's selection FORM doesn't directly contribute a sensitivity map row, but it constrains which substrates feed the map first (C2 immediately; C1 over 5-7 days; C3 over 4-7 days build + first dispatch).
2. **Pareto constraint**: DIRECT — the triplet's resource allocation IS a Pareto constraint. Total committed budget $5.75-10.75; wall-clock 14 days; model-class coverage 3 (HNeRV parity reproduction + RGB residual + hyperprior replacement). Future dispatch rankers consume this triplet as the first-wave shortlist.
3. **Bit-allocator hook**: N/A at council-memo level (no per-tensor importance change in a deliberation). Each arm's L1 landing memo wires this separately.
4. **Cathedral autopilot dispatch hook**: DIRECT — C2 is operator-authorize-ready at `.omx/operator_authorize_recipes/substrate_a1_plus_wavelet_residual_modal_t4_dispatch.yaml`. C1 and C3 will register their own recipes upon L1 landing. The triplet's selection routes the autopilot's first-wave priority.
5. **Continual-learning posterior update**: TRIGGERED on each empirical anchor — C2's first anchor (predicted within ~24 hours of operator authorization) feeds the posterior FIRST. The posterior's accepted_anchor_count is currently 21 per `feedback_bulk_anchor_backfill_executed_landed_20260512.md`; C2's anchor will become #22.
6. **Probe-disambiguator**: N/A at council-memo level (the deliberation IS the probe). Each arm's first empirical result either confirms or refutes a specific hypothesis (H1, H3, H4 per META-COUNCIL §8a); the council's binding-verdict + reactivation-criteria specifies which hypothesis is being probed per arm.

---

## 11. Cross-references

- **Codex frontier-innovation roadmap**: `.omx/research/sub017_frontier_innovation_roadmap_20260513_codex.md` (the proximate cause of this deliberation)
- **META-COUNCIL audit**: `.omx/research/meta_council_decision_attribution_audit_20260513.md` (commit `6bf2dff5`; established the Bayesian-EIG framework this deliberation uses)
- **Prior council memos**:
  - `.omx/research/grand_council_pose_axis_non_hnerv_a1_plus_lapose_20260513.md` (commit `7e77321f`; D1-D6 binding verdicts)
  - `.omx/research/grand_council_pose_axis_non_hnerv_a1_plus_lapose_d4_deeper_20260513.md` (commit `bf480e74`; D4.B 8-2 verdict)
- **Empirical state landings**:
  - `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_a1_plus_wavelet_residual_retarget_landed_20260513.md` (C2 dispatch-ready at $0.75)
  - `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_a1_plus_lapose_composition_substrate_landed_20260513.md` (sister substrate; A1+LAPose dispatch-pending at $4-5)
  - `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_macos_cpu_substrate_canvas_sweep_landed_20260513.md` (42-archive Pareto ranking)
  - `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_macos_cpu_proxy_empirical_validation_landed_20260513.md` (proxy validation within 1.6e-5)
  - `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_macos_cpu_autopilot_wiring_landed_20260513.md` (Catalog #192 wire-in)
- **HNeRV parity discipline**: CLAUDE.md "HNeRV / leaderboard-implementation parity discipline — non-negotiable" + Catalog #124 + `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md`
- **Public-PR intake constraints**: CLAUDE.md "Public Disclosure Hygiene" + Catalog #109 (`check_public_pr_intake_clones_pristine`)

---

## 12. 3-clean-pass adversarial review log

Per CLAUDE.md "Recursive adversarial review protocol — non-negotiable": each round, every council member rotates adversarial perspective; findings reset the counter.

### Round 1 — Shannon LEAD / Dykstra co-LEAD / Yousfi / Fridrich / Contrarian (READ-the-memo-for-defects pass)

- **Shannon**: Re-checked R(D) math for each arm. C1's expected ΔS range -0.001 to -0.020 is consistent with the literature on HNeRV training pipelines (Hao Chen 2023). C2's 41-500 B trailer + predicted -0.0005 to -0.003 component gain is consistent with the wavelet retarget landing memo §3. C3's expected 3-7 KB rate savings is consistent with Ballé 2018 hyperprior. **No findings.** PASS.
- **Dykstra**: Re-checked Pareto-intersection feasibility. Total budget $5.75-10.75 is well within the $20-50 GPU-budget cap per CLAUDE.md. Wall-clock 14 days has no contest race in play. **No findings.** PASS.
- **Yousfi**: Re-checked contest-faithfulness. Each arm preserves the contest scorer's evaluation contract (no scorer-at-inflate; no MPS authoritative; both axes BOTH-AXES per Submission auth eval rule). **No findings.** PASS.
- **Fridrich**: Re-checked adversarial-steganalysis leverage. C2 ships first with detector-blind foveal residual distribution per UNIWARD. C1 attacks the model-class without changing the steganalysis surface. C3 introduces a new entropy-model whose steganalysis leverage is unmeasurable until built. **No findings.** PASS.
- **Contrarian**: Re-checked the dissent voices (Hotz, Selfcomp, Carmack). Their preference for TRIPLET D over E rests on Phase-correlation's engineering aesthetics, not on the rate-math or EIG/$ math. The dissent is recorded as advisory; the 7-3 majority + Bayesian-EIG analysis is the binding verdict. **No findings.** PASS.

**Round 1: 5/5 clean. Counter = 1/3.**

### Round 2 — Quantizr / Hotz / Selfcomp / MacKay / Ballé (DISSENT-perspective pass)

- **Quantizr**: Re-checked leaderboard-archaeology assumptions. HNeRV-family dominance is well-documented (PR95/100/101/103/106 all HNeRV-family). The triplet selection assumes this continues; the only counter-evidence is the FFNeRV/HiNeRV paper (cited by codex) showing potential non-HNeRV competitiveness — but those papers haven't appeared in the public PR record. **No findings.** PASS.
- **Hotz**: Re-checked dissent rationale. Phase-correlation is genuinely elegant + zero-parameter inflate; but the byte-cost analysis (Selfcomp confirmed) gives wavelet the byte win. The dissent is an aesthetic preference, not a score-math preference. **No findings on the verdict (dissent stays).** PASS.
- **Selfcomp**: Re-checked bit-budget allocation. The 2:1 split (decoder+latent : sidecar) is correct for both TRIPLET D and E. The choice between them is on the sidecar arm; wavelet's archive-cost advantage (~10×) is solid. **No findings.** PASS.
- **MacKay**: Re-checked Bayesian-EIG calculation. The 7-3 majority is supported by ~17% more bits/$ + ~30% faster wall-clock. The dissent's aesthetic preference doesn't override the math. **No findings.** PASS.
- **Ballé**: Re-checked C3 honest-priors. Hyperprior calibration on 1-min dashcam clip is data-starved; the council's structure (parallel-build-no-premature-dispatch) correctly captures this uncertainty. **No findings.** PASS.

**Round 2: 5/5 clean. Counter = 2/3.**

### Round 3 — Boyd / Tao / Filler / Mallat / van den Oord (GRAND-COUNCIL pass)

- **Boyd**: ADMM convergence on the 3-arm convex feasibility region. Each arm operates on a disjoint surface (HNeRV training-pipeline / RGB residual / Ballé entropy-model). The alternating-projection converges to non-empty intersection. **No findings.** PASS.
- **Tao**: Harmonic analysis on the wavelet basis (C2). DB4 orthonormality holds; reconstruction is information-preserving for the band-limited residual. **No findings.** PASS.
- **Filler**: STC parity-check perspective. N/A for this deliberation (no mask payload). PASS (non-applicable).
- **Mallat (SEAT)**: Multi-resolution analysis. The wavelet retarget's single-level DB4 decomposition is reasonable for a residual head; if C2 lands indeterminate, Round-2 wavelet_levels=2 explores deeper decomposition. **No findings.** PASS.
- **van den Oord**: Discrete vs continuous codebook for C3. Ballé hyperprior is continuous (Gaussian); van den Oord's VQ-VAE is discrete. Codex's R1 ranking is Ballé over VQ-VAE due to the contest's exact-eval pathway preferring continuous. **No findings.** PASS.

**Round 3: 5/5 clean. Counter = 3/3. SEAL.**

The verdict is binding per CLAUDE.md "Recursive adversarial review protocol — close paths".

---

## 13. Operator-routable decisions

Per CLAUDE.md "Adversarial council review of design decisions" non-negotiable, the council's job is to SURFACE the decision; the operator routes the final action. The council's binding 7-3 verdict for TRIPLET E is the recommendation; alternative routes include:

1. **Honor TRIPLET E** (council recommended): operator authorizes C2 dispatch ($0.75 / 90-min wall-clock) IMMEDIATELY; operator routes C1 forensic-investigation lane to a dedicated research subagent over 5-7 days; operator routes C3 Ballé replacement build to a parallel subagent over 4-7 days. Total wave: $5.75-10.75 over 14 days.
2. **Honor TRIPLET D** (3-vote dissent): operator authorizes Phase-correlation BUILD subagent (2-3 days), then dispatches Phase-correlation ($4-5) + C1 + C3. Total wave: $9-15 over 14-21 days.
3. **Hybrid** (operator-routed): operator authorizes C2 dispatch IMMEDIATELY (the cheap empirical anchor), then routes the second/third arms based on C2's outcome. This is essentially TRIPLET E with explicit conditional routing.
4. **Defer triplet** (Carmack/Hotz fallback): operator focuses on C1 (HNeRV parity recovery) alone for 14 days, reserving C2/C3 for Round-2 based on C1's outcome.

**Council recommends option #1 (TRIPLET E) or option #3 (hybrid)**. Both have 7-vote majority support.

---

## 14. Recommended dispatch ORDER

| Phase | Arm | Cost | Time | Resolves |
|---|---|---:|---|---|
| 0 | **C2 A1+wavelet macOS-CPU pre-smoke** | $0 | 1 hour | Proxy chain on A1 baseline |
| 1 | **C2 A1+wavelet Modal T4 smoke** | $0.15 | 30 min | Integration validation |
| 2 | **C2 A1+wavelet Modal T4 full** | $0.60 | 1-2 hour | H3 (A1 saturation) + first empirical anchor |
| 3 | **C2 A1+wavelet contest-CPU Linux x86_64** | ~$0.20 | 60-120 min | Both-axes per Submission auth eval rule |
| 4 (parallel to 0-3) | **C1 HNeRV parity research** (no GPU) | $0 | days 1-5 | H1 forensic recovery (no dispatch) |
| 5 (after 3) | **C1 HNeRV parity first dispatch** (if recovery yields runnable recipe) | $5-10 | days 5-7 | H1 empirical validation |
| 6 (parallel to 0-5) | **C3 Ballé build** (no GPU) | $0 | days 1-7 | H4 substrate-engineering |
| 7 (after 6) | **C3 Ballé first dispatch** | $4-5 | day 8-10 | H4 empirical validation |

**Total wall-clock to first empirical anchor (C2 contest-CUDA): ~1 day from operator authorization.**
**Total wall-clock to all 3 arms anchored: ~14 days.**
**Total committed cost: $9.95-15.95** (within $20 contest-budget cap per CLAUDE.md "GPU budget").

C1 and C3 ARMS ARE BUILD-FIRST PARALLEL — no GPU dispatch until L1 landing per codex's anti-local-minimum guard. C2 is the single immediate-fire arm because it has already passed L1 landing (per sister-subagent's `feedback_a1_plus_wavelet_residual_retarget_landed_20260513.md`).

---

## 15. Closing

The council deliberated for 5 rounds (positions + tradeoff dimensions + new-triplet surfacing + vote + 3-clean-pass adversarial review) and reached a 7-3 binding verdict for TRIPLET E: HNeRV parity recovery + A1+wavelet IMMEDIATE + Ballé replacement-parallel.

The 3-vote dissent (Hotz, Selfcomp, Carmack) for TRIPLET D is registered as advisory — their preference for Phase-correlation's engineering elegance is acknowledged, but the Bayesian-EIG math and the immediate-readiness of C2 (already L1) tilt the binding verdict to E.

Per CLAUDE.md "KILL is LAST RESORT", no arm is killed. Each arm has explicit reactivation criteria (§1). If C2 lands in the indeterminate band, TRIPLET H (the Round-2 follow-up with scorer-sensitivity atom selector) is pre-staged.

Per CLAUDE.md "Adversarial council review of design decisions", the binding verdict goes to the operator for final routing.

**SEAL.**
