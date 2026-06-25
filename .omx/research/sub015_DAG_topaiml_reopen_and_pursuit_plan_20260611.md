# THE sub-0.15 DAG — top-AIML re-open + full-breadth pursuit + recursive-greenup-per-node (2026-06-11)

**Operator directive (2026-06-11, verbatim intent):** "everything we closed were like janky prototypes
and far from the top-AIML versions our project calls for; all of those findings about the orphaned and
the futuristic and everything you have suggested I want folded into the DAG and goal work; all futuristic
paths pursued too; keep digging super deep always into the math, question all interpretations and audit
multiple times similar to our recursive senior engineer review and greenup loop."

**Authority discipline (binding).** Every score figure here is `[derived]` or `[macOS advisory]` unless
tagged `[contest-CPU]`/`[contest-CUDA]`. torch-CPU evaluate.py (Linux x86_64, 600-sample) is the ONLY
leaderboard authority; macOS advisory; NO MPS. **Frontier pointer UNMOVED: 0.19109982 [contest-CPU],
177,169 B — ABOVE T_1, GOAL UNSATISFIED.** This is the structured work-graph, not a pointer move.

---

## 0. THE GOAL (root node)

Lower the EXACT score `S = 100·d_seg + sqrt(10·d_pose) + 25·archive_bytes/37,545,489` below **0.15 (T_3)**;
**T_1 = 0.19** is the floor of acceptable. Success = a NEW lower row in
`.omx/state/canonical_frontier_pointer.json` from `upstream/evaluate.py`. Frontier component split
[measured/derived]: d_seg ≈ 6.7e-4 (term 0.067), d_pose ≈ 3.4e-5 (term 0.018), rate ≈ 0.106.

## 1. THE RE-FRAME (the directive that re-opens the graph) — JANKY-PROTOTYPE → TOP-AIML

Per Catalog #307 (paradigm-vs-implementation falsification) + CLAUDE.md "Forbidden premature KILL": **every
prior "closed/falsified/DEFER" verdict is hereby re-classified as IMPLEMENTATION-LEVEL falsified on a
janky prototype — PARADIGM INTACT — and is RE-OPENED for a top-AIML re-attempt.** A closure is only valid
when ALL THREE hold:
1. **Top-AIML implementation** (SOTA-grade, not a sketch; export contract + numpy-portable inflate +
   torch-parity gate from byte zero, per HNeRV-parity discipline).
2. **A MEASURED exact-scorer row** (byte-closed archive → exact d_seg/d_pose; advisory local is the gate,
   paired contest-CPU is the verdict). Measurement-first: no closure on interpretation alone.
3. **3-clean-pass recursive senior-engineer greenup review** (CLAUDE.md "Recursive adversarial review
   protocol") that QUESTIONS ALL INTERPRETATIONS — any unresolved interpretation resets the counter.

The empirical warrant for the re-frame (this session): the d_seg plateau was re-diagnosed FOUR times in
two days (0.505 wall → EMA artifact → true plateau → under-training → capacity+broken-curriculum) — each
"wall" dissolved into an artifact. Interpretations are cheap and have been wrong repeatedly; **measured
byte-closed rows are the currency.**

## 2. THE THREE THREADS (the work graph) — all pursued in breadth

### THREAD A — Frozen-frontier score-aware ADDITIVE adapter (near-term, cheap, exact-UNTESTED)
The one orthogonal class never closed on the exact scorer: a small, spatially-conditioned, score-aware,
*additive* correction trained against the EXACT argmax-flip d_seg + official d_pose (which the frontier's
recon/proxy training never saw). Break-even [derived, agent-1]: a **1 KB adapter needs only −1.0% d_seg**
(or −7.4% d_pose). Nodes:
- **A1** 0-byte EDGE-CONDITIONED output transform (lever-G reactivation; NOT global — global cancels). `[agent 2 testing now]`
- **A2** Learned tiny-CNN score-aware POSTFILTER — operator's intuition; contract EXISTS
  (`src/tac/inflate_time_post_processing/learned_post_filter.py`); fixed-mode variants were exact-retired,
  the LEARNED-vs-exact-scorer variant was demonstrated only on PROXY/PSNR and ORPHANED → re-open top-AIML.
- **A3** VeRA seed-regen adapter (~200–600 B; agent-1 #1; best ΔS-per-byte).
- **A4** LoRA/DoRA / sparse-weight-diff / BitFit/(IA)³ (agent-1 #2–#5).
- **A5** Inverse-scorer GUIDANCE (orphaned, REUSE per `evaluator_inverse_orphan_inventory_20260609.md`):
  `scorer_inverse_decision_surface.py` (fragile boundaries / sufficient statistics) +
  `null_space_exploiter/` (byte-space invisibility) + `segnet_boundary_marginals.py` → tells A1–A4 WHERE
  to spend adapter capacity (boundary-flip pixels) and where it is free.
- Reuse: `src/tac/lora.py`, `lora_pose_v2.py`, `score_aware_loop/{trainer,live_segnet_loss}.py`.
- Gate: net ΔS < −0.0005 advisory on held-out pairs → paired contest-CPU ratification.

### THREAD B — Smaller-basis RETRAIN (the capacity wall is BASIS-SPECIFIC, not fundamental)
Reconciles the adversarial review ("base_ch=20 conv-HNeRV looks capacity-limited") with the futuristic
thesis: the param↔d_seg curve was measured ONLY on conv-HNeRV; **Cool-Chic-video reaches competitive
fidelity at ~800 synthesis params** [measured] by moving capacity into multiresolution latent grids +
a tiny autoregressive entropy model. The wall is a property of the basis we chose. PURSUE ALL (operator:
"all futuristic paths pursued too"):
- **B1** Cool-Chic / C3 score-aware codec — top of agent-3 ranking; OUR OWN DEFERRED lane (export-design
  blocker only). $0 first step: port Cool-Chic 5.0 synth+ARM to MLX, fit frame1 on n48, **measure
  param-count at the 5.6e-4 d_seg basin** (the single number that confirms/kills "basis-specific").
- **B2** Meta-learned INR init — ship base init in inflate.py (no video-derived constants → likely legal),
  store only per-video MODULATIONS → rate **0.010–0.027** [derived].
- **B3** KAN-basis decoder (learnable spline activations; fewer params at fidelity).
- **B4** Train-in-factored-form decoder (TT/Tucker/CP; NOT post-hoc — frozen weights proved full-rank).
- **B5** Hash-grid / Instant-NGP + SHACIRA latent compression.
- **B6** Information-Bottleneck score-SUFFICIENT carrier (store the minimal sufficient statistic for
  SegNet-argmax + PoseNet-6, not pixels; rate 0.016–0.043 [derived] = the intrinsic-floor neighborhood).
- **B7** Weight-superposition / polysemantic packing (speculative; interference math vs d_seg tolerance).
- **B8** Diffusion/flow micro-prior carrier (speculative; runtime-budget + legality risk; the ≤30-min /
  ≤2-dep / ≤100-LOC inflate contract is the hard wall).
- **B9** Algorithmic-information / program-synthesis witness (speculative; the V6 evaluator-equivalence-
  quotient compiler; no tractable solver yet — the K(w)≥H(w) counting bound is the governing math).
- **SHARED NEW MATH (the cross-node dependency):** a **differentiable surrogate for the argmax-flip d_seg
  set-functional** + a **quantization-aware entropy model** that survives int8/FP4A export. This is the
  genuinely new math B1/B6 + A2 all need; building it once unblocks several nodes.

### THREAD C — Capstone at the RIGHT capacity (the existing vehicle, corrected)
- **C1** Capacity verdict: bc24 + a frontier-class UNTIED (~178K) arm at n48 → does more/untied capacity
  drop the d_seg floor toward 5.6e-4? `[GPU-gated behind bc20_p192]`
- **C2** $0 muon-resume bc20_p48 +400ep logging LIVE d_seg (not EMA shadow) → asymptote verdict.
- **C3** Fix the curriculum bug (grad_clip_muon=1.0 throttles every step; smooth_disagreement RAISES d_seg).
- **C4** ONLY after a capacity verdict → paid n600 PR95-scale at frontier-class capacity (local n600 ≈
  5–6 months; needs paid GPU — few GPU-hours).

## 3. DEPENDENCIES (the DAG edges)

- A1 → (informs) A2/A3/A4 (cheapest first; 0-byte before paid-byte).
- A5 (inverse-scorer guidance) → A1/A2/A3/A4 (where to spend capacity).
- B-SHARED-MATH (differentiable argmax-flip surrogate + QA-entropy) → B1, B6, and A2's training.
- C1/C2/C3 (cheap capacity verdict) → C4 (paid n600) AND → B (if conv-HNeRV is basis-limited, B is the
  only capstone path; if not, C4 is viable).
- B1 param-at-basin measurement → decides whether sub-0.13 is reachable by basis-switch (the pivotal node).
- ALL nodes → the recursive-greenup gate (§1) before any closure.

## 4. EXECUTION DISCIPLINE (the operator's "recursive senior engineer review and greenup loop")

- **Rolling parallel pursuit:** ≥2–3 subagents at a time (operator standing directive), each implementing
  ONE node at TOP-AIML quality with search-and-familiarize-first (reuse the orphaned surfaces; do not
  rebuild). The GPU is single — GPU-bound nodes (B-retrains, C-arms) serialize; CPU/research nodes (A5,
  B-SHARED-MATH design, $0 smokes) run alongside.
- **Recursive greenup per node:** every node's verdict passes 3-clean-pass adversarial review that
  questions all interpretations (the lens that caught the 4× re-diagnosis). A closure without it is invalid.
- **Measurement-first cadence:** every cycle produces a measured byte-closed S row OR a $0 smoke with a
  falsification threshold — NOT another interpretation. (The strongest review finding: stop re-diagnosing.)
- **Deep-math always:** each node names its governing math (rate-distortion, IB, the argmax-flip
  surrogate, the K(w)≥H(w) bound, the param↔d_seg curve's architecture-dependence) and the closed-form
  bound it must beat, per CLAUDE.md "Meta-Lagrangian/Pareto solver — prefer solvable math over sweeps."

## 5. FOLDED-IN SOURCES (everything suggested, now in the graph)

- Orphaned inverse-scorer: `evaluator_inverse_orphan_inventory_20260609.md` (103 surfaces + 19
  contest_exploits; REUSE-AS-IS list) → THREAD A5 + B6.
- Futuristic 9 paths (agent 3) → THREAD B (B1–B9).
- Orthogonal adapter levers (agent 1) → THREAD A (A2–A4) + the break-even math.
- Postfilter (operator) → A2 (contract exists; re-open top-AIML vs exact scorer).
- Capstone reframe (adversarial review correction `capstone_dseg_plateau…_20260611.md`) → THREAD C + the
  basis-specific reconciliation that motivates B.
- Closed-lever proof (`frontier_pointer_move_ledger_20260610.md`) → the boundary: post-hoc re-coding is
  closed; ADDITIVE adapters (A) + RETRAINED smaller bases (B) are the open classes.

## 6b. THE ANTI-DEFERRAL / EV-READINESS GATE (operator 2026-06-11 — the leapfrog lesson)

**Source:** operator verbatim — "many deferred and retired and orphaned are causing similar signal loss
which caused us to get leapfrogged over [241]LOC when everything was sitting ready and your own research
directive had that on the path … but as deferred when it was obviously a top high-EV priority."

The May 4 2026 race postmortem is the empirical anchor: a 241-LOC silver medal shipped past us in the race
window while we held a ready stack as "deferred." **"DEFER/RETIRE/ORPHAN" is now a FORBIDDEN resting state
for any node that is (a) READY (code/contract exists or is a $0-first-step) AND (b) HIGH-EV (clears the
break-even / is on the critical path).** The bias inverts: such a node is ACTIONABLE-NOW, not parked.

Per-node mandatory tag (the gate): `{READY: yes/no | EV: high/med/low | STATUS: DOING-NOW / blocked-by-<X> /
genuinely-deferred-because-<measured-blocker>}`. A node may ONLY be `genuinely-deferred` with a NAMED,
MEASURED blocker (not "looked hard," not "we got a bad prototype result" — that re-opens per §1). If READY
∧ high-EV ∧ no measured blocker → it MUST be DOING-NOW. This gate is checked every cycle; a ready high-EV
node sitting un-launched is the bug the operator is extincting.

**The READY ∧ high-EV nodes RIGHT NOW (launched, not deferred):**
- **A2 learned score-aware postfilter** — contract EXISTS (`learned_post_filter.py`) + `lora.py` +
  `score_aware_loop/`; the LEARNED-vs-EXACT-scorer variant was orphaned on proxy/PSNR → DOING-NOW.
- **B1 Cool-Chic/C3 param-at-basin** — our deferred lane; $0 first step (port + n48 fit) → DOING-NOW.
- **A1 0-byte edge-conditioned transform** — `[agent 2 DOING-NOW]`.
- **A5 inverse-scorer guidance surfaces** — REUSE-AS-IS (orphan inventory) → feed A2/A3 immediately.

## 6. STATUS / NEXT

Pointer UNMOVED 0.191. Running: bc20_p192 (GPU, C1-scaling), c1prime (CPU, C-curriculum), agent-2
(A1 0-byte smoke + adapter feasibility — last of 3 out). When agent-2 lands → full ranked synthesis +
launch the next pursuit batch (top-AIML, recursive-greenup): the highest-EV cheap nodes first
(A1/A2 exact-scorer smoke, B1 Cool-Chic param-at-basin, B-SHARED-MATH design), GPU-bound nodes serialized
behind the capacity verdict. This memo is the canonical DAG every future tick + subagent consumes.

---

## DAG FEED 2026-06-23 (deep-math-grounded; frozen-instance-optimal; post adversarial-review-all)

**Frontier UNMOVED 0.19110 (borrowed). Adversarial review (`adversarial_review_all_results_20260623.md`, a19c109): SOLID 3 / SUSPECT 2 / OVER-CLAIMED 3 / CONTAMINATED 1.**

### SOLID (trust): recode@order-0-Shannon-floor; 384-floor 0.019 S (n600, sub-0.15 not pipeline-blocked); residency-no-op + scorer=97%/epoch (batch INERT); PR95 inflate→d_seg 6.02e-4; BUG-A muon_lr A/B.
### SUSPECT / OVER-CLAIMED / CONTAMINATED (do NOT trust as settled):
- **"0.191 = RGB ceiling" — UNDER-POWERED, NOT established.** The decisive link ("a small OWN-trained decoder can't reach sub-0.15 d_seg at the rate budget") was NEVER measured. Existence-proof cuts the OTHER way: bc20 break-even is d_seg<7.35e-4; **PR95 measures 5.6e-4 → a CONVERGED PR95-class small decoder is, on our own arithmetic, a sub-0.15 candidate.** DO NOT pivot off the RGB rung until measured.
- **reverse-engineer-prune = CONTAMINATED as a capacity claim** (pruning a co-adapted net ≠ from-scratch; 60ep KD ≈500× less than PR95's 29,650). Capacity cliff is a pruning artifact, not a capacity verdict.
- **taper "NO-GO" = SUSPECT/RETRACTED-DIRECTION:** the +18% was ge300 of a 3000 budget (stage-1/2, under-converged); **converged disk anchors FLIP the sign to −8% (taper may HELP).** Re-validate at convergence — NOT closed.
- bit-depth "dead" = OVER-CLAIM (QAT-finetune int4/5/6 never run; only PTQ + int5). closed-form 0.111/0.179 = contaminated-α + chained hypotheses (directional only). non-RGB GO = conditional (2 unmeasured factors).

### THE DECISIVE RE-VALIDATION (#1, load-bearing for 5/8 results + the whole strategy):
**CLEAN from-scratch d_seg(capacity) sweep — bc20/24/28/32 trained FROM SCRATCH (NOT pruned) via the BUG-A-corrected 8-stage curriculum to CONVERGENCE → byte-close → exact-eval.** GREEN if converged bc20/24 byte-closes <0.19 (or <0.15); RED (ceiling earned on SOLID ground) only if a fully-converged from-scratch run caps d_seg ≥ break-even with a clean ≥3-pt α fit. The never-fired run (armed since 06-11; hardened launcher + command BUILT). **Per §6b anti-deferral gate: FIRE IT; stop characterizing the wall.**

### NODE — TOPOLOGY (deep-math, frozen-instance): **RESOLVED → LIMITED (af64e924, 5eab5b78e, $0/600-frame, NEVER MPS).**
Hypothesis was: d_seg = (near-constant region adjacency) + (codim-1 boundary moving low-dim with EGO-MOTION); one ego-trajectory unifies d_seg+d_pose. **MEASURED:** partition splits CLEANLY into coarse-stable (99.3% pixels, eff-rank 4.07, top-1 mode 46%) + fine-volatile (0.72% pixels = class-1 islands, ~31/frame, eff-rank 52.9 = full-rank content-noise). The d_seg-binding flips live in the FINE islands. **UNIFICATION FALSIFIED:** ego-motion explains only R²=0.23 of the boundary (horizon-proxy R²=0.034) → d_seg and d_pose are DECOUPLED, NOT one trajectory. **No middle vertex:** drop islands → d_seg 0.0071 (12.6× frontier) S≈0.73; keep islands → ~524KB (#52) S≈0.84. The binding signal is full-rank → no $0 frozen-instance sidecar. **REINFORCES** a3061 (flip-residual rank 547) + #52 (partition 524KB) + weight-entropy floor. Reusable prior: route TRAINER capacity to the 0.72% class-1 small-component regions, not the free 99.3%. Verdict's own conclusion: **the d_seg lever belongs in TRAINING (a trained full-grid generator scored on its own frame-1), not a sidecar.**

### CONVERGENCE 2026-06-23 — THE $0/FROZEN-INSTANCE PHASE IS EXHAUSTED; sub-0.15 IS A TRAINING PROBLEM.
FOUR independent lines now agree: (1) operator compression reframe → bit/coder/mixed lever measured-closed → carrier/training; (2) adversarial-review-all → "0.191 ceiling under-powered" → fire the from-scratch sweep; (3) topology → LIMITED → trained generator not sidecar; (4) weight-entropy/qaxis → rate at Shannon floor, sub-8-bit breaks d_seg → shift D(H) via carrier/architecture. **Every post-hoc sidecar/recode/bit-trick is closed.** The only lever that touches the full-rank d_seg content-noise is a TRAINED generator = (a) FINER/WIRE spectral screen [in flight — does architecture lower d_seg(H)?] (b) the never-fired from-scratch capacity sweep [+INT4 arm]. The search has moved from frozen-instance to TRAINING.

### DAG FEED 2026-06-23e (OPERATOR VISION: our own representation/format + inflate.py interpreter/inverse-solver)
Operator: *"create our own representation or file format … bitmask but combined with magnitudes and
behaviors/ops … extremely compressible, as much packed into inflate.py as possible … inflate.py as an
interpreter to do clever/outrageous/inverse things on cpu/gpu."* = the Evaluator-Equivalent Witness Compiler
made concrete. Design: `.omx/research/custom_witness_format_inflate_interpreter_design_20260623.md`.
**ORTHOGONAL FACTORIZATION (the key):** S factors into (d_seg/d_pose = TRAINED GENERATOR) × (bytes = CUSTOM
FORMAT). The format does NOT lower d_seg; it lowers the RATE a given distortion is carried at. **EXISTENCE
PROOF (corrects my prior over-claim):** L13 already built the FORMAT's pose half — rendered pose-carrier
22.5KB, d_pose 12.66→0.006, real byte-close, round-trip-SURVIVES, **−59% vs frontier RGB rate** — but L13
d_seg=0.0068 (NOT lossless-parity, NOT sub-0.15; S≈0.79). So the format is PROVEN on smooth/low-rank parts
(pose, coarse boundary), and hits the SAME island wall on d_seg. Format = 4 layers: bitmask/contour (WHERE,
coarse rank-4 cheap) + magnitudes (HOW-MUCH, Fisher-√ band) + ops (HOW-TO-EVOLVE, warp+delta temporal-MC) +
inflate.py inverse-solver (HOW-TO-SOLVE, level-set/feasibility/NCA #55/#73/#143, CPU/GPU, render-survives) +
#99 firmware bit-pack (2.85× constant). Beats the flat-sidecar NO-GO (543KB/46%-survival) by ops+render.
**Composed projection (directional): generator d_seg~6e-4 + pose 0.006 + format rate 50-72KB → S≈0.12-0.14.**
$0 measurement spawned (add84f1): island intrinsic-dim across {pixel/DCT/contour/motion-comp/AE} → which
basis collapses m (≤13=format-compressible vs ≈53=generator-only). The format is the RATE half of sub-0.15;
the trained generator is the d_seg half — same convergence, now with the rate half quantified + half-proven.

### DAG FEED 2026-06-23f (the d_seg islands = LANE markings; 3-structure decomposition; geometry=PRIOR not sidecar)
Operator: islands are lane markings + horizon + hood, all modeled in openpilot/comma OSS. Memo
`long_thin_tail_lane_marking_codec_math_20260623`. **MEASURED (existence-proof, 3 priors):** class-1 = lane
markings (27.6 comp/frame = dashes). The "full-rank content-noise" was the wrong MODEL (pixel-linear); the
right model is thin curves on a road plane. BUT the geometric SIDECAR is NO-GO (3×): HOOD interior already
FREE (#139: 19 flips in 25% of frame, clamp saves ~0); HORIZON line cheap (213B) but band flip-matrix rank
547/600, NO-GO FINAL; LANE homography FALSIFIED (identity beats pose-warp, 1.29px). **PATTERN: geometry =
WHERE (cheap, openpilot-known); round-trip × fast-motion = WHICH-FLIP (full-rank content-noise — the
operator's "move fast + interact through round-trip expensively").** REVISED verdict: geometry is a
CAPACITY-ROUTING PRIOR (spend generator bits on lane/horizon bands; hood free) NOT a sidecar; flips need a
ROUND-TRIP-AWARE RENDERED witness + CAMERA-RES sub-pixel POLYTOPE placement (#149 set facet at 874 before
downsample; #73 Dykstra argmax-cell ∩; #55 closed-spec solver) — the one regime where flips are deterministic
not content-noise. Wavelet/curvelet (Daubechies/Candès-Donoho, "large but sparse, rate to spare") = the band
basis IF camera-res polytope reopens the flips. The lane/horizon/hood math FEEDS the generator+format build;
it does not replace it. Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-23g (DECISIVE: islands are an ~8-dim NONLINEAR manifold → GO-GENERATOR; the convergence CLOSES)
Measurement add84f1 (`island_representation_level_intrinsic_dim_20260624`, commit d3c8174c3, $0/CPU, 19
NO-FAKE tests): the class-1 island intrinsic dim across 5 levels. **LINEAR bases ALL lose** (pixel PCA k95=412,
DCT 61, Fourier-contour 29, affine-motion-comp 94 — all ≫ the Whitney m≤13 budget). **NONLINEAR m ≈ 8–13**
(AE 90%-knee=8, MLE 13.1). **NO-FAKE phase-shuffle control:** real islands 81% AE-recon @ dim-8 vs
structure-destroyed shuffle 18% (needs 32) → the low knee is REAL structure, NOT a sparsity artifact (the line
between GO-GENERATOR and WALL). **VERDICT GO-GENERATOR:** the islands are a curved ~8-dim nonlinear manifold;
a trained generator (learns the chart) captures them at ~8 latents; NO fixed basis (incl. wavelet/curvelet/
contour — all linear) can (curved manifold → secant span ≫ manifold dim). 2·8+1=17 ≤ the HNeRV 28-dim latent
→ FITS WITH ROOM. **Implication: the d_seg island wall is NOT capacity (8 dims is tiny) — it is
architecture/training (learn the curved chart + survive the round-trip).**

**THE CONVERGENCE NOW CLOSES (6 independent measured lines → sub-0.15 = a TRAINED GENERATOR):** (1) compression
bit/coder closed; (2) adversarial 0.191-ceiling under-powered; (3) topology islands full-rank LINEAR; (4)
weight-entropy at Shannon floor; (5) lane/horizon/hood geometry=WHERE, flips=content-noise; (6) **islands =
8-dim nonlinear manifold, GO-GENERATOR (shuffle-controlled).** The format = the RATE half (L13 proven −59%);
the generator = the d_seg half, now with a MEASURED target (8-dim island chart) + a concrete architecture
criterion (nonlinear high-freq chart — FINER/WIRE tests it). NEXT: FINER/WIRE verdict → from-scratch generator
with geometry-prior capacity routing + round-trip-in-loop + camera-res polytope on the lane band → byte-close
in the custom format → exact eval. Pointer UNMOVED 0.19110; the END is that exact row.

### DAG FEED 2026-06-23h (adversarial review of the sharpened design: camera-res polytope on HARD-PAINT is RED → texture-dependence wall → REINFORCES GO-GENERATOR, 7th angle)
Existence-proof check on my own sharpened-design piece (#148/#149 pincer, `curve_core_gate_RED_survival_wall_
and_the_pincer_20260618` + 2026-06-19 mechanism correction): camera-res sub-pixel boundary placement is
**RED**. geo_recon fits (→0.00106 @ mp128, below GREEN) but **realized plateaus at 0.0067** (survival_gap
0.0057, 6.3×); ~16% of boundary-band pixels flip through the roundtrip REGARDLESS of geometry fidelity OR
differentiable color/boundary-offset pre-compensation. **MECHANISM (corrected): the wall is TEXTURE-DEPENDENCE,
not the resize** — SegNet's argmax at the boundary is a function of the local TEXTURE in its receptive field,
not the boundary-pixel position; a hard-painted (flat) neighborhood → wrong margin → flips. **CORRECTION to
DAG-FEED-f/g sharpened design:** DROP "camera-res sub-pixel polytope placement on the lane band" as a
standalone hard-paint lever (RED). The polytope feasibility must be solved over the RGB NEIGHBORHOOD (texture),
not the boundary coordinate → that IS the trained generator producing the right texture. **REINFORCES
GO-GENERATOR (7th converging line):** hard-paint / geometric / sidecar all fail the texture wall; only a
trained generator that reproduces the local texture survives. The 8-dim island manifold IS the texture
manifold the generator must learn. Revised generator spec: trained RGB-output generator (produces survival-
texture) + geometry-PRIOR capacity routing (lane/horizon bands) + round-trip-in-loop + custom format (rate).
No camera-res hard-paint step. Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-24a (upstream seg-sidecar survey + adversarial correction → 8th/9th GO-GENERATOR lines)
Operator: "did any PR have useful SegNet sidecars / self-contained seg like our pose solution? search all upstream."
Survey `upstream_segnet_sidecar_selfcontained_survey_20260624` (a10d1adc, commit 99dd9a4ac) read every
inflate.py. **Asymmetry VERIFIED:** all submissions emit RGB; the scorer re-derives masks → you can't inject a
stored mask; a FLAT self-contained seg carrier (the literal pose analog) is DOMINATED (pose=6-dim-flat → 22.5KB
carrier works; seg=full-rank-linear/8-dim-nonlinear → no flat carrier). **8th GO-GENERATOR line: the HNeRV
winners (0.19–0.23) carry NO seg sidecar — seg quality is ARCHITECTURAL** (mask-conditioned renderers are
dominated: quantizr 0.33, fp4_mask_gen 0.37, qzs3_range_mask ~0.33–4.39). **ADVERSARIAL CORRECTION (existence-
proof vs our own #112):** the survey's headline "ADOPT #1 = qpose14 236-B seg-action boundary-flip sidecar" is
**the SAME lever as our Lever-D survival-selective flip coder, ALREADY MEASURED NO-GO on a converged base**
(`lever_d_nuanced_fullstack_20260612`: net-ΔS<0 needs coded-subset survival σ>σ*=b/WATERLINE=0.99/1.273=0.778;
measured best-decile σ=0.51<0.778 → realistic predictor admits NOTHING net-negative; only a cheating oracle is
GO). qpose14's 236-B win is BASE-DEPENDENT (weak 0.32 mask-renderer base → many cheap high-survival flips); on
the HNeRV frontier (d_seg 6e-4) the residual flips are the shallow/texture-wall ones (σ≈0.51) → NO-GO. Same
texture-survival wall (#149). **9th GO-GENERATOR line: "the d_seg win belongs IN TRAINING" (#112 verdict,
re-confirmed).** REAL adopt from the survey = v4_qp_aq2_roi's SegNet-guided routing → a SegNet-boundary saliency
weight on the GENERATOR's d_seg loss (in-training, the right place) + the ROI/corridor geometry-prior for
capacity routing (= the lane/horizon-band routing). qzs3 mask-coder + qpose14 sidecar = LEARN-only (dominated/
base-dependent). NET: every upstream seg approach either renders-RGB (architectural, the winners) or is a
base-dependent sidecar our good base obviates → ALL routes converge on the trained generator + in-training
saliency routing. Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-24b (FINER/WIRE spectral screen DONE → FINER is a REAL d_seg architecture win; picks the chart)
The spectral screen (a0e28b5, 3 arms × 3000ep × n100, byte-closed CPU-authority d_seg) FINISHED. **SIREN(ctrl)
best d_seg 0.001692 (~ge300 0.004754 ✓ reproduces taper-screen 0.004756 → wire-in byte-identical clean);
FINER best 0.001376 = 0.813× (−18.7%) CLEARS the 0.85× falsification bar; WIRE10 best 0.001659 = 0.98× NULL.**
VERDICT: SPECTRAL-LIMITED (partially) — FINER's variable-local-frequency sin((|x|+1)·x) is a REAL d_seg lever
at EQUAL bytes (the FIRST architecture-only positive d_seg lever this arc = the D(H) curve-shift the
convergence required); WIRE's Gabor space-localization did NOT help (frequency-adaptivity > space-localization
for the high-freq codim-1 boundary). 5-lens: spectral-bias fix (Tancik/Rahaman) — FINER extends the
coordinate-net spectrum to the lane/island high-freq edges SIREN smooths; win at convergence (ep2895),
marginal @ge300 (needs full budget). CAVEATS: n=100 proxy → advisory until n600 confirm; relative win (FINER
vs SIREN equal-everything = the valid architecture answer). **PICKS the generator chart = FINER.** Generator
spec now COMPLETE: FINER chart + geometry-PRIOR capacity routing (lane/horizon) + round-trip-in-loop +
SegNet-saliency d_seg loss (v4-adopt) + custom witness format (rate). NEXT: n600 FINER confirm → from-scratch
generator fire (local MPS free/slow vs Modal CUDA ≤$20 fast, #162-sanctioned) → byte-close → exact eval = the
END. Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-24c (ADVERSARIAL review of the FINER win: LOWER-FLOOR not faster-converger → genuine curve-shift; n600 confirm FIRED)
Sharpest adversarial challenge to FINER's −18.7%: is it a convergence-SPEED artifact (SIREN catches up with more
epochs) or a lower-FLOOR win (true D(H) curve-shift)? **MEASURED from the n100 trajectories: BOTH arms are
PLATEAUED at distinct asymptotes** — SIREN tail-slope −7.0e-8/ep (last-decile Δ −0.000023), FINER −5.7e-8/ep (Δ
−0.000010); SIREN ~0.00169 vs FINER ~0.00138. SIREN needs **~4,500 more epochs** to close the 0.00032 gap on an
already-flattening curve (won't happen). **VERDICT: FINER is a genuine LOWER-FLOOR win = D(H) curve-shift at
equal bytes, NOT merely faster** — the strongest form of the result; FINER is the sub-0.15 architecture lever,
not a training-time-only lever. 5-lens: physics — SIREN's floor is set by its fixed-ω bandwidth ceiling; FINER
raises the representable-frequency ceiling → places the high-freq boundary SIREN structurally can't; the floor
GAP = the spectral capacity SIREN lacks. TOPOLOGY PREDICTION (testable on checkpoints): FINER's floor gain
concentrates in the high-freq class-1 LANE islands, not the low-freq coarse classes → if true, weight the
generator geometry-prior toward class-1. CAVEAT: n100 floor-win; the **n600×1500 FINER-vs-SIREN confirm is now
FIRED** (durable no-orphan daemons act_screen_finer_n600 pid62958 + act_screen_siren_n600 pid62962, MPS, ~15h/
arm) to test floor-gap persistence at 6× data (prior: spectral advantage persists/grows). Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-24d (operator: research next-gen / outside-the-box / OWN nonlinearities beyond FINER)
FINER (2024) beat SIREN but is NOT the field frontier; and our target is a sharp argmax STEP/EDGE (scored on
flips through uint8 round-trip), NOT a generic PSNR image → literature PSNR-rankings DO NOT transfer (WIRE is
PSNR-respected but was NULL for us). DEEP-MATH: sinusoids (SIREN/FINER) ring on steps (Gibbs); an
edge/step-MATCHED nonlinearity may beat FINER. Research subagent ab12614a FIRED ($0 research+build, NO
training — n600 confirm owns MPS). Candidate space (2024-26): Gauss (Ramasinghe-Lucey), HOSC (hyperbolic
osc), Sinc, FLAIR (freq+locality, 2025), Fourier-Chebyshev (2026), WINNER, + **KAN/Fourier-KAN (FKAN)** =
LEARNED activation (B-spline/Fourier per edge = literal "define our own, learned"). **3 OWN designs to
implement+parity-test+stage:** (a) FINER×Gauss hybrid (FINER's winning var-freq × Gaussian envelope = WIRE
localization WITH FINER adaptivity); (b) learnable step-basis (Σ learnable shifted/scaled tanh = native
piecewise-constant for the argmax partition — most on-problem); (c) FKAN-style learnable nonlinearity.
RANK by d_seg-fit (sharp-edge @ low capacity + round-trip survival + composes-with-FINER), NOT PSNR.
Falsification bar to ADOPT over FINER: d_seg ≤ 0.813× SIREN (beat FINER). Screen STAGED (not run) to fire
when MPS frees post-n600-confirm. Wire-in: extends `activation_family.py` (siren/finer/wire infra built).
Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-24e (next-gen activation research DONE → step-native axis (hosc/step_basis) beats FINER's sinusoid on theory; 8 impl+parity-tested, staged)
Subagent ab12614a DONE (commit 97f41ed94, 8 activations, 92 NO-FAKE tests, $0, NO training). Memo
`nextgen_inr_activations_survey_and_custom_designs_20260624`. **KEY: the argmax target IS a step function;
FINER is still SINUSOIDAL (Gibbs overshoot at discontinuities → flips shallow-margin pixels). hosc=`tanh(β·sin x)`
→ square-wave step-train = native argmax shape, NO Gibbs; step_basis=`Σ aₖ·tanh(gₖ(x−cₖ))` = soft-Heaviside
sum = exact partition shape.** Ranked d_seg-fit: (1) hosc (2) step_basis [ours, +12B] (3) finer_gauss [ours,
FINER×Gauss, byte-neutral] (4) fkan [ours, learnable, +10B] (5-7) gauss/rcgauss-FLAIR/sinc (low-pass).
TOPOLOGY: step functions are the natural basis for a piecewise-constant partition → representation basis
matches target topology (deepest activation↔problem fit yet). EXISTENCE-PROOF (subagent humility, correct):
"FINER is the activation floor" is PREMATURE — WIRE was null b/c it localized the WRONG (fixed-freq) carrier;
hosc/step_basis test the UNTESTED step-shape axis. ADVERSARIAL RISK: hosc saturation (no-Gibbs) ⇒ vanishing
gradients at large β → trainability risk (finite hosc_beta knob; screen reveals). CAVEAT: learnable families
(step_basis/fkan) add ~10-12 state_dict keys → byte-closed archive of a learnable winner needs a trivial
codec sidecar (flagged, non-blocking for the screen). SEQUENCING (free/local): n600 FINER-confirm (running) →
n100 activation screen (hosc/step_basis/finer_gauss/fkan vs FINER; adopt-bar = beat FINER 0.813× SIREN) →
n600-confirm overall winner → generator fire with winning chart. The operator's "define our own" produced 2
of the top-4 + validated the step-not-sinusoid reframe. Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-24f (n600 confirm IN-PROGRESS CAUTION: FINER advantage weaker + REVERSING at n600 — challenges the "persists/grows" prior)
n600 FINER-vs-SIREN confirm at ep352/1500 (~3.5h, LIVE pid62958/62962): **FINER 0.003527 / SIREN 0.003423 =
1.030× (FINER BEHIND).** Ratio TREND: n600 ep50/100/150 = 0.968/0.959/0.951× (FINER ahead but WEAKER than
n100's 0.85×) → ep352 1.030× (REVERSED). Contrast n100: FINER held a STABLE 0.833-0.894× lead through ep100-300
→ final 0.813×. **My DAG-FEED-24c "spectral advantage persists/grows at n600" prior is CHALLENGED by the data.**
Two hypotheses, too early to call (ep352/1500, both d_seg still descending ~0.0034, NOT converged; the FLOOR
difference where n100's win lived isn't set until ~ep1500): **H1** win generalizes but emerges LATE (n100 win
was ep2895; n600 "late" ≈ ep1500) + single-seed noise; **H2** capacity-dependent artifact — at n600 (6× data,
same params) both activations are capacity-bound and can't exploit FINER's extra bandwidth → the win was an
n100 memorization-regime artifact. The early reversal leans H2 but is NOT conclusive. ACTION: let it run to
convergence (killing it discards the load-bearing floor measurement); NO over-claim either way; verdict at
~ep1500 (~11h). **CONSEQUENCE: the staged step-native activation screen (hosc/step_basis) rose in value** — if
FINER's spectral win is capacity-fragile, the step-shape axis (different from bandwidth) may generalize where
bandwidth doesn't. Existence-proof discipline: flagging my own prior as under-challenge, not defending it.
Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-24g (DEEP-MATH grounding of the n600 caution: activation choice is REGIME-dependent → predicts H2 + inverts the winner)
Memo `activation_capacity_vs_bandwidth_regime_deepmath_20260624` (commit 9fa2aeaf0). Derived from NTK +
MDL + topology + free-energy: per-frame budget = **P·b/n** → two regimes. **Bandwidth-limited (n100):**
floor set by representable frequency band → FINER's broader band wins (✓ −18.7%). **Capacity-limited
(n600/contest):** floor set by params-per-edge efficiency → FINER (sine) ties SIREN (same family); spectral
band unusable under-budget. **PREDICTION (falsifiable @ ~ep1500): n600 FINER ≈ SIREN ±5%** (= H2, predicted
NOT just observed; within-run 0.95→1.03 supports the mechanism). **THE INVERSION:** d_seg is a pointwise
argmax-at-edge = **L∞-at-edge** criterion; a step encodes an edge in **O(1) params, zero Gibbs**, a sine
needs **O(1/ε) harmonics + persistent 9% Gibbs overshoot** (which flips shallow-margin pixels = the #149
texture-survival wall). So under capacity pressure the **step-native activations (hosc/step_basis) win MORE,
not less** — opposite of FINER. PLAN CHANGES: (1) the activation screen MUST run at **n600** (capacity-limited
contest regime; n100 ranking is bandwidth-regime, does NOT transfer — n100 = cheap pre-filter only); (2)
predicted n600 ranking: hosc/step_basis > fkan > FINER≈SIREN≈finer_gauss > gauss/sinc; (3) sub-0.15 at the
contest regime is an **MDL problem** (fewest bits to represent the argmax-edge manifold) = the same principle
as the 8-dim island manifold + a topology-matched (step) basis. Falsification: if n600 converges FINER ≤0.90×
SIREN, H1 wins + capacity framing wrong. Verdict ~ep1500 (~11h). Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-24h (operator: + info-theory/signal-proc/set-theory/entropy/VCM/INR/optimal → 12-lens over-determination + NEW survival prediction)
Regime memo §7b (commit 8bea1c7bb): 7 more lenses on the argmax-edge problem, ALL converging on
step-native/partition-matched at the capacity regime (12 lenses total, over-determined): info-theory (MDL
two-part, capacity ⟺ P·b<H_boundary); **signal-proc (NEW): round-trip = LOW-PASS; sine Gibbs RINGING aliases
→ flips; steps no-ring → survive ⇒ step-native lowers the #149 survival-wall flip-fraction (~16%), not just
the d_seg floor**; set-theory (d_seg = symmetric-diff of set partitions; indicator basis natively
parametrizes membership); entropy (step lowers bits/edge → pushes capacity wall out); VCM (CEO/indirect-RD
optimal = task-boundary-matched code); INR (weights-as-code → pick most-compressible-for-step architecture);
optimal (both S terms → step basis = joint RD-optimum at capacity regime). **SCREEN ADDITION: the n600
activation screen MUST measure round-trip SURVIVAL per activation (the #149 metric), not only d_seg** — tests
the step-native-lowers-the-survival-wall prediction (first bridge between activation choice and the survival
wall). Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-24i (n600 confirm CHECKPOINT ep589 — prediction TRACKING: FINER advantage decaying −18.7%→−1.6%, ep352 reversal was transient)
n600 confirm @ ep589/1500 (LIVE): FINER d_seg 0.003106 / SIREN 0.003157 = **0.984× (FINER ahead by 1.6%)**.
Ratio trajectory: 0.95×(ep150) → 1.03×(ep352, the flagged dip) → **0.984×(ep589)** → the ep352 reversal was
training-dynamics NOISE around a small mean, NOT a sustained flip (cautious flagging vindicated). **HEADLINE:
FINER's lead COLLAPSED from −18.7% (n100) to −1.6% (n600) — ~10× decay**, exactly the capacity-regime
prediction (DAG-FEED-24g: bandwidth advantage decays as per-frame budget P·b/n tightens). H2 holding so far
(0.984× within the predicted ±5% FINER≈SIREN band). Not converged (both ~0.0031 descending); FLOOR verdict at
~ep1500 (~9h). **CONSEQUENCE: FINER is a MARGINAL lever at contest scale (−1.6%), not strong → elevates the
step-native screen** (12-lens theory: step-native wins MORE under capacity pressure, opposite of FINER's
decay). The n600 confirm caught a potential over-investment in an n100 result that doesn't transfer; the
theory already names the alternative (hosc/step_basis). Pointer UNMOVED 0.19110.

### LIVE: FINER/WIRE/SIREN architecture screen (a0e28b5, MPS, spectral k1); topology measurement (af64e924, $0).
### NEXT (ranked): (1) FIRE the from-scratch capacity sweep [THE decisive measurement] (2) FINER/WIRE verdict (3) topology af64e924 (4) k1-champion × C* × L13 witness → closed-form stack.

## DAG FEED 2026-06-23c (OPERATOR REFRAME: "int8 brotli / compression SIZE, not channels/params")

**Operator hypothesis (2026-06-23):** the binding determinant is the int8+brotli compressed SIZE, not the
raw channel/param count. **Verdict: the TARGET is right (rate = 0.106 = 55% of the frontier, the binding
term; the author DID pick bc36 under an int8+brotli byte budget), but every COMPRESSION sub-lever is
MEASURED-CLOSED on the borrowed frontier.** The operator's instinct correctly redirects away from "just
sweep channels at int8" (the structural-param RD walk, bottoms ~0.186) toward "bytes is the thing" — and
the measured answer says: on the RGB carrier the bytes CANNOT shrink without breaking d_seg.

### The complete compression-lever ledger (ALL measured, $0, byte-closed exact-eval — existence-proof grade)
| compression lever | result | source |
|---|---|---|
| better general coder (DeepCABAC/order-2 arithmetic) | EXHAUSTED — brotli-q11 6.891 vs marginal H(W) 6.884 b/param (gap 0.007, ≤0.07KB) | `decoder_weight_rate_axis_…_synthesis_20260621` |
| uniform sub-8-bit PTQ (int7/6/5/4/3) | RED — int8 is S-min over the WHOLE axis; int8→int7 saves 1.4KB for +0.044 S; d_seg+d_pose super-linear collapse swamps rate at every step | `qaxis_bitdepth_response_surface_20260623` (n48 full + n600) |
| int5 QAT-finetune (LSQ + outlier-clip) | RED — recovers d_pose −89% but d_seg only −9.5% (stays ~0.0042, 7.6× floor); CE seg-loss FLAT ep10→100 | `frontier_int5_lsq_best_shot_retest` |
| score-aware mixed/reverse-waterfill (WRQ) | MODEST + CONSTRAINED — sensitivity ~5.5× flat (2.5 bits dyn range) AND must spend MORE bits on boundary weights | synthesis §2.3 |
| latent dedup / low-rank | near-exhaustion (small / near-full-rank) | synthesis §3 |

### THE DEEP-MATH UNIFICATION (5-lens; why "bits vs channels" is the WRONG axis)
Both score-controlled terms are monotone functions of the **information content H (bits)** the decoder carries:
- **rate(H) = 25·(H/8 + overhead)/N** — Shannon: cannot store H bits in < H/8 bytes; brotli PROVES we're at the floor.
- **d_seg(H) = the rate-distortion function** — measured ~6e-4 @ H≈1.3Mbit (frontier), rising super-linearly as H drops.
So **S(H) = 100·d_seg(H) + 25·(H/8+c)/N + √(10·d_pose)** is a 1-D RD optimization. **CROSS-CHECK (the killer):
reducing H via fewer bits (int5 → d_seg 0.0026) and reducing H via fewer params (bc20 → d_seg ~0.002–0.0035)
land on the SAME D(H) curve** — H-allocation is INVARIANT; "bits vs channels" is a distinction without a
difference at the optimum, and BOTH are dominated by the d_seg(H) coupling. **Physics:** int8 weights at the
order-0 entropy floor = the 2nd law for codes (can't pack the same info into fewer bytes). **Geometry:** the
d_seg-critical capacity is the 77%-of-params early/low-res stages at the codim-1 boundary; 66.5% of flips are
<0.5 logit (shallow) → quant noise ∝2^−b flips them → bit-dropping hits exactly the fragile boundary.

### THE ESCAPE (where the operator's "bytes" instinct actually pays): SHIFT the D(H) curve, don't re-allocate H
Lower d_seg at the SAME bytes — TWO ways, both NOT post-hoc compression:
1. **Different CARRIER (task-space / L13 witness, #171, operator's own capstone):** spend H on the
   scorer-relevant manifold (argmax boundary + 6-dim pose) NOT the discarded RGB → the task-space D(H) curve is
   fundamentally lower (you stop paying to reconstruct RGB the scorer throws away). THE genuine curve-shift.
2. **Spectral ARCHITECTURE (FINER/WIRE, in flight a0e28b5):** if a high-freq activation represents the
   codim-1 boundary at lower H → curve shifts down. Measuring now.
3. **Structural weight-tie/low-rank (synthesis §2.4):** cuts rate by fewer PARAMS but only tying the
   d_seg-IRRELEVANT weights (boundary is perturbation-fragile) — the RD-walk lever, bounded ~0.186.

### ONE untested compression door (folds into the from-scratch sweep): **from-scratch co-adapted INT4 QAT.**
All bit measurements were PTQ or finetune-OF-int8; NONE trained the curriculum WITH int4 fake-quant from the
start. Capacity-cliff principle (QAT-from-scratch >> PTQ+finetune) makes it the one open variant. → when the
from-scratch sweep fires, **add an INT4-co-adapted arm** so one run settles the operator's bits-vs-channels
question with the untested variant. Does NOT change the ranked NEXT; sharpens the #1 sweep's design.

### DAG FEED 2026-06-24j (n600 confirm ep939 — REFINED: advantage INTERMEDIATE ~-5%, SIREN plateaued / FINER still descending)
ep939/1500 (LIVE, ~8.2h): FINER 0.002925 (tail-slope -6.2e-8/ep, STILL DESCENDING) / SIREN 0.003056 (tail-slope ~+3e-9, PLATEAUED) = 0.957x (FINER ahead 4.3%; extrap ep1500 ~-5%). Ratio walked 0.984x(ep589)->0.957x(ep939) as SIREN flattened + FINER kept improving (my ep589 -1.6% was mid-noise; corrected). VERDICT REFINED: FINER advantage decays with capacity but does NOT vanish: -18.7%(n100)->~-5%(n600) = INTERMEDIATE between H1(full win) and H2(parity); lands at the EDGE of the predicted +/-5% band. MECHANISM: SIREN hit its bandwidth-limited floor (plateaued ~0.00306); FINER (broader band) still finding lower-d_seg configs -> some bandwidth headroom remains at n600, ~4x less than n100. CONSEQUENCE: FINER = real-but-modest lever (keep as baseline chart); step-native screen still the high-value test (theory: steps win on EFFICIENCY which dominates at capacity-limit, beating FINER -5%). Verdict at ~ep1500 (~6h). Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-24k (OPERATOR: all implementations likely NOT optimal yet → verdicts are PROVISIONAL (sub-optimal form); optimal-form pass mandated before generator fire)
Per CLAUDE.md OPTIMAL-FORM-before-dispatch (NSCS06 v6->v7 = 44% in one cargo-cult-unwind pass): a verdict on a sub-optimal impl falsifies the IMPL not the paradigm. AUDIT (grounded) of the LIVE arms: (1) muon_lr=2e-4 vendored default in EVERY arm (wrappers pass --muon-lr-floor-fix but NOT --muon-lr; memory flags 2e-4 as ~150x too small vs working 0.03; floor-fix only fixes the cosine FLOOR keying, not base LR); (2) per-activation hyperparams are first-guess defaults (hosc_beta=4.0, omega=1.0 for ALL [SIREN textbook optimum ~30], rcgauss_rolloff=0.05, step_basis_k/fkan_k untuned) -> comparing activations at one default omega is apples-to-oranges; (3) curriculum C3 bugs (grad_clip_muon throttle + smooth_disagreement RAISES d_seg) still in active path. DEEP-MATH: comparing un-tuned hyperparams = non-Pareto points; valid comparison is between each family OPTIMUM (Pareto points). RE-LABEL: FINER -18.7%(n100)/-5%(n600), SIREN plateau, the activation ranking = ALL PROVISIONAL (sub-optimal vehicle form). ROBUSTNESS: FINER-vs-SIREN RELATIVE verdict is MOST robust (both sine-family omega=1.0 share the vehicle, shared sub-optimality partly cancels); the activation SCREEN is LEAST robust (each needs its own optimum); absolute floors are NOT optimal-form floors. PLAN CORRECTION (bounded optimal-form, not infinite tuning): before generator fire + before any load-bearing activation verdict -> optimal-form pass: (a) fix C3 curriculum bugs; (b) muon_lr value-check (2e-4 vs ~0.03); (c) small per-activation hyperparam tune (omega/hosc_beta/step_basis_k at sweet spot); THEN the n600 screen. Bound = cargo-cult-unwound + hyperparams at sweet spot + known bugs fixed. Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-24l (READY-TO-FIRE NODE: n600 step-native screen = the decisive crux test; verified flags, deterministic)
PER the convergence law (feed = a measurement design, not prose). n600 confirm ~done (ep1397/1500, FINER best 0.002916 / SIREN 0.003052 = 0.955x = −4.5%, converging to the predicted intermediate ~−5%; capacity-regime CONFIRMED). NEXT decisive measurement — tests the 12-lens prediction (step-native beats FINER at the capacity limit). FIRE when the n600 confirm frees MPS (~1h; NO contention — let it finish; durable no-orphan daemons). NEW arms only (FINER 0.002916 / SIREN 0.003052 baselines already measured): hosc(beta=4), hosc-sharp(beta=8), step_basis(k=8). EXACT verified command per arm (flags confirmed present in launch_split_by_head_basin.py): `.venv/bin/python -u experiments/launch_split_by_head_basin.py --no-split-by-head --train-device mps --device cpu --base-channels 20 --latent-dim 28 --n-pairs 600 --total-epoch-budget 1500 --seed 0 --muon-lr-floor-fix --eval-every 50 --activation <hosc|step_basis> [--hosc-beta 8 | --step-basis-k 8] --out-dir experiments/results/act_screen_<ARM>_n600_b1500` via spawn_durable_daemon.py. ADOPT-BAR: beat FINER (d_seg <= 0.002916 = <0.955x SIREN). ALSO measure round-trip SURVIVAL per arm (the signal-proc prediction: step-native lowers the #149 ~16% survival-wall flip-fraction — no Gibbs ringing to alias under the round-trip low-pass). DETERMINISM: same config as the FINER/SIREN n600 baseline (apples-to-apples; muon_lr held at the baseline value so the RELATIVE ranking is clean — a separate vehicle-level muon_lr optimal-form node addresses the absolute floor, NOT conflated here). PREDICTION (pre-registered): hosc/step_basis < FINER (step-native wins on efficiency at the capacity limit) AND lower survival-wall. Verdict feeds the generator chart choice. Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-24m (n600 confirm CONVERGED + step-native screen LIVE; launch-bug fixed; autonomy gap owned)
n600 confirm DONE (clean): FINER best 0.002915 / SIREN 0.003051 = 0.955x (−4.5% FINAL) → capacity-regime prediction CONFIRMED (−18.7% n100 → −4.5% n600; FINER modest real lever at scale). Step-native screen NOW LIVE (verified pids 65962 hosc-b4 / 65964 hosc-b8 / 65966 step_basis-k8; bc20 n600 1500ep, vs measured FINER 0.002915/SIREN 0.003051 baselines; adopt-bar beat FINER; pre-registered: step-native wins at capacity limit + lower #149 survival-wall). LAUNCH-BUG (fixed): first fire died — $COMMON shell-var through spawn_durable_daemon REMAINDER mangled args; inline-flags pattern works; NO-FAKE lesson: VERIFY launch alive before claiming. AUTONOMY GAP (operator-flagged, owned): detached daemons do NOT auto-wake the model → I only advance when prompted; true self-driving needs /loop (ScheduleWakeup) or cron. Spine discipline reaffirmed: check LIVE state (proc/mtime/results) FIRST each turn. Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-24n (/loop tick-1: step-native screen VERIFIED training; launch-error was stale attempt-1)
VERIFY-BEFORE-CLAIM resolved a contradiction: logs showed LIVE pids + growing trajectories (106 rows stage1_CE) BUT an "unrecognized args" error. Resolution: hosc_b4.log line1=attempt-1 launch, line39=attempt-1 ERROR ($COMMON-mangled), line40=inline RELAUNCH (live). So the error is the stale attempt-1 artifact in the append-log; the 3 arms (pids 65962 hosc-b4 / 65964 hosc-b8 / 65966 step_basis-k8) are attempt-2 = validly training (~ep109/1500, stage1_v328_ce). d_seg/archive_bytes None = non-eval rows (evaluated=False); d_seg lands on eval-every-50 rows (next wake). Byte-neutrality holds by construction (fixed-form activations byte-identical to siren; step_basis +12 params on ~83K = 0.01% = negligible rate) → the screen apples-to-apples vs FINER 0.002915/SIREN 0.003051 is intact. Next deep-math iteration target (when first converged eval lands): K*/β* MDL-optimality of the step-native basis matched to the ~8-dim lane-island manifold + curvelet/contour basis fit vs the argmax edges. Loop self-pacing (~1h cadence, 13h run). Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-25a (/loop tick-2: step-native screen DECISIVE @ ep~213 — hosc DIVERGES (saturation, pre-registered), step_basis STABLE; config-retire hosc, +k16 K-knee arm)
**MEASURED (byte-closed advisory; eval-every-50; all 3 arms in stage1_v328_ce = AdamW lr≈1e-3, muon_lr=null → these are PURE ACTIVATION-stage reads, Muon not yet active):**
- hosc(β=4): d_seg 0.01191→0.00841→0.00723→**0.01357** (ep202 RISING past its ep150 best; train_loss 1.48) = **DIVERGING**.
- hosc-sharp(β=8): 0.02984→0.02818→0.01969→**0.03996** (RISING; train_loss 7.15) = **DIVERGING WORSE (β-monotone)**.
- step_basis(k=8): 0.01045→0.00717→**0.00643** (monotone-descending; train_loss 0.65 = healthiest) = **STABLE**.
- baselines (n600 converged): FINER 0.002915 / SIREN 0.003051. (At ep202 comparator: FINER 0.00601, SIREN 0.00511; step_basis@ep150 0.00643 ≈ FINER-level, slightly behind SIREN — NOT yet a win, needs convergence.)
**5-LENS DEEP-MATH REVIEW of the hosc-saturation divergence (pre-registered prediction CONFIRMED):**
(math/calculus) `d/dx tanh(β·sin x) = β·cos x·sech²(β·sin x)` ≈ 0 a.e. as β grows (saturated flats), spiking only in O(1/β)-width bands at sin's zero-crossings → **vanishing gradient a.e.**; (algebra/optimization) AdamW divides by √v̂ (2nd-moment), so in the flat near-zero-grad regions it takes FULL-SIZE steps on noise → weight random-walk → d_seg rises; β=8 saturates harder than β=4 → strictly worse at every eval (the measured β-monotonicity IS the saturation signature); (signal/Fourier) tanh(β·sin) → square wave = ALL odd harmonics → maximal Gibbs/aliasing under the round-trip low-pass (the #149 wall mechanism — opposite of the hoped no-ring benefit; the no-Gibbs benefit at β=∞ is unreachable because the gradient dies first); (geometry/topology) the target is piecewise-constant (codim-1 jumps) but the OPTIMIZER cannot reach the square-wave config — the basis matches the target yet the LOSS LANDSCAPE is untrainable (saturation ⇒ flat plateaus + sharp cliffs); (info/MDL/INR) a fixed-β square-wave activation has NON-learnable slope = it cannot trade sharpness for trainability, so it is Pareto-dominated by step_basis whose **learnable finite slopes gₖ** keep the unit in the high-gradient band WHILE expressing sharp partition edges (`Σ aₖ·tanh(gₖ(x−cₖ))`). **CONCLUSION: hosc's failure is OPTIMIZER-saturation, not a paradigm refutation — the step-native PARADIGM is carried by step_basis (stable).**
**EXISTENCE-PROOF cross-check:** SIREN's textbook ω≈30 vs our ω=1.0 default does not rescue hosc — hosc's pathology is β-saturation (independent of ω); and step_basis (the trainable-slope step basis) is the EXISTENCE PROOF that a step-native unit CAN train stably (0.65 train_loss, monotone d_seg) → no terminal claim against step-native.
**DECISION (no-fake / no-premature-kill):** CONFIG-RETIRE hosc(β=4)+hosc-sharp(β=8) [stopped via spawn_durable_daemon --stop, group-kill no-orphan, VERIFIED DEAD] — measured-config-retired NOT paradigm-kill. **Reactivation criteria:** (a) β-ANNEAL (start β≈0.5, grow over training — trainable early, sharp late), (b) lower AdamW lr in the saturating regime / grad-aware init, (c) SGD/Muon instead of AdamW (no 2nd-moment full-step-on-noise). Frees MPS 3-way→1-way → step_basis(k=8) converges ~3× faster to its decisive verdict.
**ADVANCE-ONE-ITERATION (K-knee MDL, the topology/deep-math optimization this tick):** launched **step_basis(k=16)** [VERIFIED LIVE pid 81627, identical config seed=0/bc20/n600/1500ep, only --step-basis-k 16] as a clean apples-to-apples K-dose-response. RATIONALE: step_basis params/edge are tiny (3/unit: aₖ,gₖ,cₖ; k=8→16 ≈ +0.01% of 83K → rate-negligible) so K is a near-FREE distortion knob; the open question is whether MORE edge-capacity converts step-native from "≈FINER" to "beats FINER (≤0.002915)". MDL framing: K is per-unit step-count that COMPOSES across the depth-6 PixelShuffle decoder (effective edges ≫ K), so the K* knee is an empirical MDL minimum of S(K)=100·d_seg(K)+25·bytes(K)/N (bytes≈flat in K) → pure d_seg(K) search; k∈{8,16} brackets it this round (add k=4 next if k=16 wins, to localize the knee). **Both step_basis arms LIVE.**
**CRUX UPDATE (§3 of crux memo):** the d_seg wall is confirmed NOT a basis-match problem (step basis matches the partition topology) but a TRAINABILITY/optimizer problem at the boundary — sharpening §3's "TRAINED not frozen" with "and the basis must be OPTIMIZER-REACHABLE (learnable-slope, not saturating)". Pointer UNMOVED 0.19110; honest state until a converged step_basis verdict + byte-closed exact row.
