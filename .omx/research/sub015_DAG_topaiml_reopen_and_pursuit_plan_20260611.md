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

### DAG FEED 2026-06-25b (/loop tick-3: k=8 healthy-descending update + NEW deep-math iteration — CAPACITY ROUTING = adaptive-mesh-refinement / hp-FEM, the spatial-allocation half of the MDL crux)
**LIVE MEASURED (both step_basis arms LIVE, 2-way MPS, ~45s/ep):** k=8 ep202 d_seg **0.00567** (seq 0.01045→0.00717→0.00643→0.00567, monotone; train_loss 0.54 still falling) — now BETWEEN the sine baselines at the SAME epoch (FINER@ep202 0.00601, SIREN 0.00511) = step-native is in the competitive band, healthy. k=16 ep49 (first eval ep50 pending; train_loss 2.56 — expected-higher early, 2× activation params; descent-rate is the signal not the early value). NO threshold crossing; converged verdict ~ep1500.
**NEW DEEP-MATH ITERATION (step 3; complements last tick's K-knee = the per-unit edge-capacity; THIS = the SPATIAL allocation of that capacity).** Question: given fixed P params, how should the generator's representational capacity be DISTRIBUTED over the image domain Ω? 
- **MEASURED spatial structure (the prior):** d_seg-sensitivity is EXTREMELY concentrated — class-1 lane islands = 0.72% of pixels / ~27.6 components/frame carry the binding residual (codim-1 boundary band); hood interior is FREE (#139, 0-byte clamp); horizon band is NO-GO (flip rank 547/600 — high-entropy, do NOT spend capacity chasing it); interiors of all regions are cheap (locally constant). So the sensitivity field ∂d_seg/∂(capacity at x) is a SPIKE on the lane/boundary band, ~0 in interiors, NEGATIVE-EV on the horizon.
- **THE LENS (adaptive mesh refinement / hp-FEM):** representing a piecewise-analytic target with codim-1 discontinuities, the OPTIMAL approximation refines degrees-of-freedom AT the singular set and coarsens in the smooth interior — hp-FEM gives EXPONENTIAL convergence for piecewise-analytic functions ONLY with geometric refinement toward the discontinuity (uniform meshing gives algebraic O(h) at best, killed by the L∞-at-edge criterion = exactly d_seg). Translation: the generator should put O(1/h) effective capacity in the boundary band and O(1) in interiors — a NON-UNIFORM (adaptive) capacity field, NOT the uniform coordinate-MLP every prior arm used.
- **MECHANISM (how to route capacity when weights are GLOBALLY shared — the catch):** a vanilla coordinate-MLP shares all params across all pixels (no spatial routing). Three grounded routing mechanisms, ranked: (1) **coordinate-conditioned FiLM / multiplicative gating** keyed on a boundary-distance feature (cheap; gives boundary coords more effective nonlinearity) — the openpilot lane-geometry prior #138/#145 supplies the boundary-distance map at ~0 byte; (2) **frequency-adapted positional encoding** — higher Fourier-feature frequencies allocated to the boundary band (NTK: raises the learnable band exactly where the target is high-curvature) — composes with FINER/step activation; (3) **two-branch decoder** (coarse interior + fine boundary-band) with the branch mask from the geometry prior — most explicit, highest param efficiency, the hp-FEM analog. 
- **EXISTENCE-PROOF cross-check:** the routing is justified ONLY if sensitivity is spatially concentrated — and our OWN measurements prove it (hood FREE #139 + horizon NO-GO + lane-island 0.72% binding). The contrapositive existence-proof: PR95/HNeRV winners are UNIFORM coordinate decoders → they spend capacity on the cheap interiors → that is precisely the inefficiency a capacity-routed generator escapes (the curve-SHIFT, not curve-walk). No artifact already beats a boundary-routed generator (none built one), so no terminal-conclusion conflict.
- **CRUX UPDATE (§3):** the joint MDL optimum now factors into THREE matched choices: (a) topology-matched BASIS (step-native, last tick) × (b) optimizer-REACHABLE training (learnable slope, tick-2) × (c) SPATIALLY-ROUTED capacity (hp-FEM/AMR refinement at the codim-1 lane band, geometry-prior-masked, this tick). The generator design = all three composed; the activation screen settles (a)+(b), the geometry-prior routing settles (c). 
**QUEUED (no disturbance to running arms):** generator design node now specifies coordinate-conditioned-FiLM boundary routing (mechanism 1, cheapest) + frequency-adapted PE (mechanism 2) as the default; two-branch (mechanism 3) as the fallback if FiLM under-routes. Fires after the step-native screen verdict picks the activation chart. Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-25c (/loop tick-4: both arms HEALTHY-descending + k=16 PASSES health gate + NEW deep-math iteration — ROUND-TRIP-IN-LOOP = anti-aliasing, the 4th generator factor; basis & loss are the SAME principle)
**LIVE MEASURED (both step_basis arms LIVE, 2-way MPS):** k=8 ep252 d_seg **0.00485** (seq …0.00643→0.00567→0.00485, monotone; train_loss 0.46 falling) — descended below SIREN's ep202 0.00511; tracking toward the converged band (~1.7× above FINER target 0.002915, ~1250 ep to go). k=16 FIRST eval ep50 **0.00979** vs k=8@ep50 0.01045 = ~6% LOWER at the matched epoch → **k=16 PASSES the health gate (not diverging; descending)** + early hint that more edge-capacity (K↑) lowers d_seg (the K-knee hypothesis's first datapoint; ONE point, not conclusive). Decision: KEEP k=16 (health gate passed); no retire. NO threshold crossing; verdict ~ep1500.
**NEW DEEP-MATH ITERATION (step 3; the 4th matched generator factor — completes the quartet with basis/training/routing).** The contest applies a FIXED operator **R = D∘Q∘U** before SegNet: U=bicubic upsample 384→874, Q=uint8 quantize, D=bilinear downsample 874→384; d_seg = argmax-flip of SegNet(R(frame)). 
- **WHY train-through-R (eval_roundtrip, the non-negotiable, re-derived):** minimizing d_seg(generator(x)) WITHOUT R optimizes the wrong functional — R is a contraction that moves the rendered logit field before the argmax, so the un-composed proxy has a 2–11× gap to authority. The correct loss is d_seg(R(G(x))) with R DIFFERENTIABLE (differentiable bicubic U, straight-through Q, differentiable bilinear D). This is a known discipline; the NEW content is the COMPOSITION below.
- **THE UNIFICATION (signal processing — basis ⊗ loss are ONE principle):** R is a low-pass (U,D smooth) + dither (Q). For a sharp edge: a SINE/Fourier representation carries Gibbs side-lobes = high-freq energy that, under D's downsample (= sample-after-lowpass), ALIASES to unpredictable sub-pixel shifts of the edge → flips (the aliasing IS the d_seg residual on shallow-margin pixels). A STEP representation has NO side-lobes (monotone transition) → R softens the edge IN PLACE (predictable) → fewer flips. Therefore "step basis" (representation side) and "train round-trip-in-loop" (loss side) are the SAME anti-aliasing principle: the step basis minimizes the high-freq energy R aliases, and training-through-R places the (R-softened) edge at the argmax-optimal sub-pixel location. They COMPOSE multiplicatively, not additively.
- **EXISTENCE-PROOF cross-check (and a falsifiable prediction):** the #149 texture-survival wall (~16% of boundary-band pixels flip regardless of geometry/color pre-comp) was measured with SINE-FAMILY rendering ONLY → its universality is UNPROVEN for step-native. PREDICTION (pre-registered): step-native rendering lowers the survival-wall flip-fraction below 16% (less Gibbs to alias). The screen's per-arm round-trip-survival measurement (queued at convergence) IS the existence-proof test — if step-native survival ≈ 16% too, the wall is R-fundamental (operator null-space), not basis-dependent, and the lever is sub-pixel placement (#149) instead.
- **CRUX UPDATE (§3):** the joint optimum now factors into FOUR matched choices: (a) topology-matched BASIS (step-native) × (b) optimizer-REACHABLE training (learnable slope) × (c) SPATIALLY-ROUTED capacity (hp-FEM/AMR at the lane band) × (d) ROUND-TRIP-IN-LOOP loss (R differentiable; anti-aliasing, COMPOSES with the step basis). The generator = all four; the screen settles (a)+(b)+measures (d)'s survival, the geometry prior settles (c). 
**QUEUED:** generator loss spec = d_seg-surrogate ∘ R (differentiable U/Q-STE/D) + the boundary-routing of tick-3; the screen's convergence-time survival measurement decides whether (d) is basis-sensitive (step wins) or R-fundamental (pivot to sub-pixel #149). Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-25d (/loop tick-5: K-knee signal STRENGTHENS (k=16 < k=8 at BOTH ep50 & ep100) + LITERATURE GROUNDING (online authority) — step-native/discontinuity-native is SOTA-validated)
**LIVE MEASURED (both arms LIVE, 2-way MPS):** k=8 ep302 d_seg **0.00436** (seq …0.00567→0.00485→0.00436, monotone; train_loss 0.40). k=16 ep100 **0.00656** (seq 0.00979→0.00656; train_loss 1.69). **K-KNEE SIGNAL (2 matched datapoints, both favor K↑):** ep50 k16/k8 = 0.00979/0.01045 (−6.3%), ep100 = 0.00656/0.00717 (−8.5%); gap WIDENING in k=16's favor → more soft-Heaviside edges per unit LOWERS d_seg at negligible rate (the K-knee MDL prediction holding; not yet converged, not yet at the 0.002915 bar). NO threshold crossing.
**LITERATURE GROUNDING (step 3 + step 0 authority — WebSearch/WebFetch, exercised; "research literature/OSS first"):** surveyed 2024–2025 INR-for-edges SOTA.
- **"2D Neural Fields with Learned Discontinuities" (Chen et al. 2024, arXiv 2408.00771)** — EXISTENCE-PROOF for our whole paradigm: a 2D field that treats edges as potential discontinuities + optimizes their MAGNITUDES jointly (end-to-end differentiable) beats smooth INR **InstantNGP by +5–10 dB** on edge-heavy targets + beats Mumford-Shah on Chamfer. ⇒ discontinuity-native > smooth-INR on exactly our argmax-edge target is PUBLISHED, MEASURED, not just our hypothesis. Validates the step-native campaign at SOTA. CAVEAT (borrowed-substrate accounting): it is MESH-based (per-edge magnitudes = explicit storage) → RATE-COSTLY for our byte budget; we do NOT adopt the mesh. The TRANSFERABLE principle = "parametrize the discontinuity magnitude explicitly" folds into our tick-3 two-branch boundary routing: on the 0.72% lane band ONLY (where explicit params are cheap), add explicit boundary-discontinuity parameters — a HYBRID (implicit weight-shared interior + explicit-discontinuity boundary band). NEW design refinement, queued.
- **HOSC (controllable-sharpness periodic activation)** appears in the SOTA as a real tool but with the EXACT differentiability/sharpness trade we MEASURED — our tick-2 AdamW-saturation divergence (β-monotone) is the known fragility, confirming the config-retire was correct (not a paradigm error; the literature uses HOSC carefully, with annealing — matching our recorded reactivation criterion).
- **STAF (trainable sinusoidal activations, arXiv 2502.00869)** = same learnable-activation family as our step_basis/fkan; corroborates that LEARNABLE activations (not fixed-form) are the frontier — supports step_basis (learnable slopes) over fixed hosc.
**EXISTENCE-PROOF cross-check:** the +5–10 dB learned-discontinuity result is the published existence proof that the smooth-INR (PR95/HNeRV/InstantNGP) capacity is spent inefficiently on edge targets → the curve-SHIFT our step-native + discontinuity-routing campaign targets is real and SOTA-backed, not speculative. No artifact beats a discontinuity-native+rate-cheap generator at our byte budget (the mesh methods are rate-costly; nobody has done weight-shared discontinuity-native at <100KB for this task) → open frontier, no terminal-conclusion conflict.
**CRUX UPDATE (§3):** the 4-factor generator is SOTA-grounded; ADD a 5th (rate-aware) refinement to factor (c) capacity-routing: HYBRID boundary parametrization = implicit weight-shared interior + EXPLICIT discontinuity-magnitude params on the lane band only (cheap because 0.72% of pixels), drawn from the learned-discontinuities literature. Pointer UNMOVED 0.19110. Sources: arXiv 2408.00771, 2502.00869.

### DAG FEED 2026-06-25e (/loop tick-6: K-knee robust + BYTE-CLOSE PATH VERIFIED EXISTS + HONEST means/ends S-arithmetic gap)
**LIVE MEASURED:** k=8 ep352 d_seg **0.00409** (monotone, train_loss 0.36). k=16 ep150 **0.00592** = −7.9% vs k=8@ep150 0.00643. K-KNEE now 3 STABLE datapoints (ep50 −6.3%, ep100 −8.5%, ep150 −7.9% ≈ persistent −7–8%) → the extra edges buy a genuine FLOOR reduction (capacity-regime), not an early transient. NO threshold crossing.
**BYTE-CLOSE PATH VERIFIED (the rate half — answers the queued L13 iteration with MEASURED infrastructure, not prose):** the bc20 small-basis vehicle ALREADY has a parity-verified byte-close. `experiments/results/_fire_g3_basin_baseline/g3_packet_manifest.json`: archive_zip **89,244 B**, sha256 verified, `parse_back_parity_ok=True`, parity_status all-pass (build_deterministic/keys_match/latents_fixed_point), runtime_files=[inflate.py, inflate.sh, src/codec.py, src/model.py], **contest rate term = 0.0594**, d_pose=0.00034. Builder = `build_torch_vehicle_d2_archive_zip.py`; the step_basis arms ARE this exact vehicle (launch_split_by_head_basin bc20/n600). ⇒ NO new byte-close pipeline needed. CAVEAT (verified, not assumed): runtime `src/model.py` HARDCODES `torch.sin()` (SIREN, lines 45/50/51); inflate reads latent_dim/base_channels/eval_size/n_pairs from meta but NOT activation. ⇒ step_basis byte-close = the EXISTING G3 pipeline + a BOUNDED runtime add (apply step_basis Σaₖtanh(gₖ(x−cₖ)) instead of sin, read `activation` from meta; the learnable step params already live in the decoder state_dict so codec.quantize_state_dict picks them up automatically — rate delta negligible: 3 params/unit × few units ≪ 83K conv weights). De-risks THE END to a known engineering task.
**HONEST MEANS/ENDS S-ARITHMETIC (the firewall — the screen is a MEANS; does it reach the END?):** decompose 0.19110 ≈ rate 0.118 (25·~177KB/N) + 100·d_seg(~5e-4) + √(10·d_pose). bc20's rate ADVANTAGE is real (0.0594 vs 0.118 = −0.059). To BEAT 0.19110 exploiting it, bc20 needs 100·d_seg + √(10·d_pose) ≤ 0.19110 − 0.0594 = 0.132; with pose √(10·0.00034)=0.058 → **100·d_seg ≤ 0.074 → d_seg ≤ ~0.0007–0.0011** (range = pose-dependent). The activation screen's BEST plausible (FINER 0.002915; step-native K-knee maybe ~0.0026) is **~2.4–4× SHORT**. An activation buys ~7–20% (the measured K-knee), NOT 2.6×. **THEREFORE: winning the step-native screen is NECESSARY but NOT SUFFICIENT for a sub-0.19110 bc20-standalone row.** The bulk of the required d_seg reduction (0.0029 → ~0.001) must come from the OTHER generator factors — dominantly factor (c) hp-FEM CAPACITY-ROUTING (concentrate ~all capacity on the 0.72% lane band; this is the only factor with multiplicative headroom, since the residual is spatially concentrated) + factor (d) round-trip-survival. Pose also matters: bc20 d_pose=0.00034 → pose term 0.058 is ~ as large as the d_seg budget, so pose must be held/lowered too.
**CRUX UPDATE (§3 — the theory of victory, honestly):** bc20-standalone-beats-0.19110 requires d_seg ≤ ~0.001 AND pose held — the ACTIVATION (~20%) is necessary scaffolding; the DOMINANT lever is capacity-ROUTING (spatial concentration on the lane band), with round-trip-survival + pose-hold. The screen picks the activation chart; the GENERATOR (routed + round-trip + pose-FiLM) is what must close the 2.6× gap. If routing cannot close it, the fallback is composition (bc20 d_seg-core ⊕ borrowed frontier) — but that route is measured NO-GO 3× (geometric/boundary sidecar), so routing is the live bet. Pointer UNMOVED 0.19110; the honest state is that NO current measured config (incl. the best screen activation) reaches sub-0.19110 standalone — the routing factor is the crux to test next.

### DAG FEED 2026-06-25f (/loop tick-7: k=16 liveness VERIFIED + CAPACITY-ROUTING LEVERAGE made MEASURABLE — the R_cap/R_surv decomposition gates the crux)
**LIVE MEASURED:** k=8 ep402 d_seg **0.00396** (seq …0.00409→0.00396; train_loss 0.34; descent SLOWING −3.2%/50ep → approaching its floor, ~1.36× above FINER 0.002915). k=16 mid-ep200-eval, **VERIFIED HEALTHY** (ps STAT=R, 79.7% CPU, CPU-TIME advanced 90:03→90:17 in 8s wall = actively computing the contended 600-pair CPU eval; the 6-min file-silence is that expensive pass, NOT a stall). Last k=16 eval ep150 0.00592. NO threshold crossing.
**DEEP-MATH ITERATION — capacity-routing leverage is BOUNDED + MEASURABLE (resolves the tick-6 crux from speculation → a ratio):** every GT-boundary argmax flip ∈ exactly one disjoint set (set theory):
- **R_cap** = flips whose HIGH-RES (874, pre-round-trip-R) rendered argmax is ALSO wrong ⇒ the decoder failed to REPRESENT the edge ⇒ CAPACITY-limited ⇒ capacity-ROUTING (move DOF interior→boundary) CAN fix.
- **R_surv** = flips whose 874 argmax is CORRECT but the post-R (384) argmax flips ⇒ a correctly-placed edge that R=D∘Q∘U ALIASES ⇒ SURVIVAL-limited ⇒ routing CANNOT fix; only round-trip-in-loop (R-aware loss, tick-4) or sub-pixel placement (#149) helps.
**⇒ capacity-routing's LEVERAGE CEILING = R_cap / R_total.** This converts "can routing close 2.6×?" into a measurable ratio. The naive geometric upper bound (interior ~free, route ~all P to the 0.72% band → ~139× boundary-capacity) is only ACHIEVABLE on the R_cap fraction; the R_surv fraction is a HARD floor routing cannot touch.
**EXISTENCE-PROOF cross-check (and the gating prior):** #149 (COMPLETED) measured the texture-survival wall ~16% of boundary pixels flip regardless of geometry/color pre-comp — i.e. a LARGE R_surv — BUT only on SINE-family rendering. So the SINE R_surv is big (routing-resistant); the STEP-NATIVE R_surv is the OPEN gate (tick-4 prediction: step-native lowers R_surv because no Gibbs to alias). If step-native R_surv ≪ sine's 16%, routing leverage (R_cap/R_total) is high and the 2.6× gap is closable; if step-native R_surv ≈ 16%, routing is bounded and the lever shifts to round-trip-in-loop + sub-pixel #149.
**QUEUED $0 MEASUREMENT (the decisive decomposition; run when the screen frees an MPS slot — NO contention with live arms):** take the converged step_basis best-ckpt → render at 874 AND 384 → SegNet argmax at both → classify every GT-boundary flip into R_cap vs R_surv → report R_cap/R_total per arm. This single measurement (a) sizes routing's leverage ceiling, (b) tests the step-native survival prediction vs the sine 16%, (c) decides routing-first vs round-trip/sub-pixel-first for the generator build. Deterministic, ~minutes, uses the screen checkpoints.
**CRUX UPDATE (§3):** routing leverage = R_cap/R_total (measurable, queued). The generator's d_seg path = route the R_cap fraction (capacity) + round-trip-in-loop/sub-pixel the R_surv fraction — the two factors target DISJOINT flip sets, so they ADD (not redundant). The 2.6× gap closes IFF (capacity routing on R_cap) + (round-trip on R_surv) jointly reach d_seg ≤ ~0.001. Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-25g (/loop tick-8: HONEST CORRECTION — K-knee ERODES (speed-knob not floor-knob); REINFORCES capacity-as-floor-lever; k=8 entered new curriculum stage)
**LIVE MEASURED:** k=8 ep402 0.00396 last eval, now ep487 in a NEW curriculum stage (train_loss 0.34→0.63 = stage loss-form change; mid-ep452-eval, VERIFIED HEALTHY ps CPU-TIME 228:23→228:40 in 7s). k=16 ep202 **0.00556**. NO threshold crossing.
**HONEST CORRECTION (NO-FAKE / VERIFY-BEFORE-CLAIM — walks back tick-5/6):** the K-knee is NOT widening/stable — it is ERODING toward parity: k16/k8 = ep50 0.937(−6.3%) → ep100 0.915(−8.5%) → ep150 0.921(−7.9%) → **ep202 0.981(−1.9%)**. Tick-5 "gap widening" + tick-6 "genuine FLOOR reduction" were PREMATURE (read off the early, fast-convergence phase). The corrected read: **K (soft-Heaviside count per unit) is a convergence-SPEED knob, not a FLOOR knob** — more edges fit the boundary FASTER early, but the floor is the same.
**DEEP-MATH (why K is speed-not-floor — and why this REINFORCES the crux):** at the capacity-limited regime the d_seg FLOOR is set by TOTAL capacity P·b (the ~83K conv weights), not per-unit activation expressivity (K adds only 3·units tiny activation params ≪ P). More K = better-conditioned early descent + (bandwidth-regime) reach, but once the activation can express the local edges, extra K does NOT raise total capacity → the floor is unmoved. This is the SAME mechanism as FINER's n100(−18.7%)→n600(−4.5%) decay (activation edge shrinks as capacity binds). ⇒ at the contest capacity limit, ALL activations (SIREN/FINER/step/K8/K16) converge to SIMILAR floors set by P; the activation screen's value is DE-RISKING + a small bandwidth-regime edge, NOT a floor win. **The FLOOR lever is CAPACITY P and its ROUTING — exactly the tick-6 conclusion, now independently reinforced by the K-knee erosion.**
**EXISTENCE-PROOF cross-check:** consistent with (a) FINER n100→n600 decay (capacity regime), (b) the measured bc20 d_seg floor being capacity-bound (MUONJUMP plateau ~0.0021 with SIREN; the small basis as-is doesn't break it), (c) the S(H)=D(H) invariance (bits-vs-channels = same curve). No contradiction; the erosion is the predicted capacity-regime signature.
**PLANNING CONSEQUENCE (no premature kill; measurement-first):** if K is speed-not-floor, k=16's floor ≈ k=8's floor → running both to ep1500 is partly redundant. STOP-CRITERION for k=16 (frees MPS → k=8 floor verdict ~1.5× faster): if the K-knee stays ≤ −2% at ep250 AND ep300 (erosion confirmed), config-retire k=16 (reactivation: K is a speed knob, revisit only if a floor-K effect appears) and let k=8 converge solo. Do NOT stop this tick (one erosion datapoint; confirm at ep250/300 first). Keep both; re-decide next tick.
**CRUX UPDATE (§3):** the activation/K axis is now MEASURED as a speed+bandwidth knob, NOT the floor lever — the floor lever is capacity P + routing (the R_cap fraction) + round-trip-survival (the R_surv fraction). The from-scratch generator's d_seg gain comes from ROUTING + round-trip, with the activation chart (likely FINER ≈ step at the floor) as scaffolding. Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-25h (/loop tick-9: C3 CURRICULUM BUG CONFIRMED IN LIVE DATA — the smooth stage RAISES d_seg; a 3rd d_seg lever (curriculum fix); K-knee stabilized ~−2–3% → keep k=16)
**LIVE MEASURED (per-STAGE d_seg trace from k=8 — grounds the curriculum tradeoff):** stage1_v328_ce ep50→150 0.01045→0.00643 (↓), stage2_v331_softplus ep202→402 0.00567→**0.00396** (↓, the running min), **stage3_v332_smooth ep488 0.00423 (↑ ROSE +6.8% from the stage2 min)**; now stage5_c1a_l7 ep555 (train_loss 1.45). k=16 ep252 0.00471. K-knee: −6.3/−8.5/−7.9/−1.9/**−2.9%** (ep50/100/150/202/252) = STABILIZED ~−2–3%, did NOT erode <2% → stop-criterion UNMET → KEEP k=16 (no premature kill; honest refinement of tick-8: K-knee = mostly speed + a small RESIDUAL ~−2–3% floor edge).
**DEEP-MATH ITERATION (the real finding — a 3rd d_seg lever, MEASURED not theorized):** the per-stage trace CONFIRMS the C3 curriculum bug flagged in the CURRENT-STATE caveat: the `smooth_disagreement` stage (stage3) RAISES d_seg (0.00396→0.00423), and the later C1a/lambda/sigma rate-regularization stages historically trade d_seg for rate/regularity. ⇒ the 8-stage curriculum has stages that LOWER d_seg (CE, softplus, Muon) and stages that RAISE it (smooth, rate-reg). **CONSEQUENCE for the screen:** the ABSOLUTE floors at ep1500 are curriculum-bug-CONTAMINATED (inflated by smooth/rate stages); the RELATIVE ranking (step vs FINER, ALL arms share the SAME curriculum) stays VALID (shared contamination cancels). **CONSEQUENCE for the generator:** a 3rd d_seg lever = CURRICULUM FIX — skip/repair the smooth stage + de-weight the rate-reg stages + Muon-finish on d_seg. This is SEPARATE from capacity-routing and activation.
**EXISTENCE-PROOF cross-check (HYPOTHESIS, flagged NOT-same-config):** MUONJUMP SIREN reached d_seg ~0.0021 (jumped stage-5-snapshot → Muon, SKIPPING the rate-reg stages 6–7) vs full-curriculum n600 SIREN 0.003051 — CONSISTENT with "skipping rate/smooth stages lowers d_seg ~31%." BUT the MUONJUMP base run ≠ the n600-screen base run (different seed/snapshot/epochs), so this is a HYPOTHESIS, not a measurement. A clean same-config test (full-curriculum vs smooth-skipped/Muon-finished, same vehicle) is the verification — queued.
**CRUX UPDATE (§3 — now THREE measured d_seg levers + their contamination map):** the bc20 d_seg floor is shaped by (1) CAPACITY + routing (R_cap, the dominant lever, tick-6/7), (2) ROUND-TRIP survival (R_surv, tick-4), (3) CURRICULUM (skip smooth/rate + Muon-finish, tick-9, measured ~+7% from one buggy stage alone). The ACTIVATION is a 4th, smaller (speed + ~−2–3% floor). The from-scratch generator must compose ALL: capacity-routed + round-trip-in-loop + d_seg-curriculum (fixed) + best activation. The screen's relative verdict (step vs FINER) is valid; its absolute floor is NOT the achievable floor (curriculum-contaminated). Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-25i (/loop tick-10: smooth-bump is TRANSIENT (c1a recovers it, new min 0.00385) — softens tick-9; live S-arithmetic re-grounding; POSE ruled out as a lever)
**LIVE MEASURED:** k=8 ep589 stage5_c1a_l7 d_seg **0.00385** = RECOVERED below the stage2 min 0.00396 (so the stage3_smooth bump 0.00396→0.00423 was TRANSIENT; c1a brought it back to a NEW min). Now ep635 stage5_c1a (loss 1.38). k=16 ep302 stage2 **0.00417** (K-knee −4.4%; noisy band −2 to −4.4% over ep202–302). Both descending; floor verdict pending Muon stage-8 (~ep1500).
**REFINEMENT of tick-9 (honest, softens the curriculum-bug severity):** the smooth-stage d_seg bump is TRANSIENT, not permanent contamination — the next stage (c1a) recovers it AND sets a new min (net stage2→stage5: 0.00396→0.00385, still descending). So the curriculum-bug contamination of the ABSOLUTE floor is SMALLER than tick-9 implied; the bigger curriculum question remains the MUONJUMP hypothesis (do rate-reg stages 6–7 raise the FINAL floor? — still a queued same-config test).
**LIVE S-ARITHMETIC RE-GROUNDING (NO-FAKE — uses measured numbers, not estimates):** bc20 k=8 NOW = 100·0.00385 + √(10·0.0002) + 25·89244/37545489 = 0.385 + 0.045 + 0.0594 = **0.489**. Muon-floor-projected (MUONJUMP ~0.0021, not-same-config caveat) ≈ 100·0.0021+0.045+0.0594 = **0.31**. BOTH ≫ 0.19110. To beat 0.19110: 100·d_seg ≤ 0.191−0.045−0.0594 = 0.087 → **d_seg ≤ ~0.00087**. From the Muon floor ~0.0021, the 4 levers must close ~**2.4×**. This is the honest live gap.
**POSE RULED OUT AS A LEVER (closes an axis):** d_pose trajectory is DESCENDING on its own — ep50 0.0072 → ep402 0.00023 → ep488 0.00020, still trending. Pose improves FOR FREE with training (FiLM/low-rank already solve it); it is NOT a binding lever. The √(10·d_pose) term (~0.045 now, → ~0.032 if d_pose→0.0001) shrinks with training without intervention. ⇒ d_seg is the SOLE binding crux; do not spend a generator lever on pose. (Consistent with the CLAUDE.md operating-point note: pose marginal value is high at low d_pose, but here it is already being captured by training, not requiring a new lever.)
**CRUX UPDATE (§3):** unchanged structure (4 d_seg levers; pose free); the live gap is d_seg 0.0021(Muon floor)→0.00087 = ~2.4×, to be closed by capacity-routing(R_cap, dominant) + round-trip(R_surv) + curriculum-finish + activation. The screen grinds to its Muon-floor verdict (~ep1500, ~5h); the decisive MEASUREMENTS (R_cap/R_surv decomposition; same-config curriculum-fix) run when it frees a slot. Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-25j (/loop tick-11: steady descent + CURVELET edge-basis grounds the routing factor's BASIS — boundary should be DIRECTIONAL not isotropic)
**LIVE MEASURED:** k=8 stage5_c1a steadily descending 0.00385→0.00375→0.00372 (ep589/639/689; ~1.28× above FINER 0.002915), now ep707. k=16 ep402 stage2 **0.00374** = K-knee −5.6% (persistent ~−5% band over ep302–402). Both healthy, no threshold crossing; floor verdict pending Muon stage-8.
**DEEP-MATH ITERATION (curvelet edge-basis — grounds the BASIS of factor (c) routing; new, not-yet-executed):** the d_seg boundary = the lane-marking edge. openpilot models lanes as **3rd-order polynomials** (#145 cross-ref) ⇒ the boundary is a **C²-smooth curve carrying a cross-edge discontinuity** (smooth WHERE, sharp ACROSS). Nonlinear-approximation theory for "cartoon" functions (smooth regions separated by C² edges, Candès–Donoho 2004) gives a STRICT N-term-error ordering: **curvelets O(N⁻²(log N)³) ≻ wavelets O(N⁻¹) ≻ Fourier/sine O(N⁻¹ᐟ²)**. ⇒ the optimal boundary representation is ANISOTROPIC + DIRECTIONAL + multiscale (curvelet/contourlet/shearlet), aligned to the lane-curve TANGENT — NOT the isotropic pixel/step grid every prior arm uses. **Generator refinement (composes the routing sub-factors):** factor (c) routing = WHERE (hp-FEM, concentrate on the 0.72% band) × WHAT-BASIS (curvelet/directional, matched to the smooth lane curve) × SHARPNESS (step activation across the edge). The directional basis is the missing piece that makes the routed capacity efficient — an isotropic basis wastes DOF on the cross-edge direction where a 1-D step suffices, while needing many coefficients along the curve; a directional basis spends 1 coefficient along-tangent per scale.
**EXISTENCE-PROOF cross-check:** (a) curvelet optimality for cartoon/edge functions is a PROVEN theorem (Candès–Donoho), not heuristic; (b) openpilot's 3rd-order-polynomial lane model is the existence proof that the boundary IS C²-smooth (low-order) → the theorem applies; (c) consistent with the operator framing "only so many lane markings, smooth, fast-moving" (few smooth curves = sparse in a directional basis). Borrowed-substrate note: curvelets are an established transform (Candès–Donoho); our contribution is APPLYING the directional-basis-on-the-routed-band to the argmax-edge generator, not inventing the transform.
**CRUX UPDATE (§3):** factor (c) capacity-routing now has 3 sub-factors (where × basis × sharpness); the BASIS sub-factor (directional/curvelet, matched to lane-polynomial geometry) is the efficiency multiplier that helps close the ~2.4× gap. Implementation paths for "directional" in an INR generator: (i) anisotropic positional encoding (frequencies oriented to the local lane tangent from the openpilot prior), (ii) steerable/directional conv filters in the boundary branch, (iii) explicit contourlet coefficients on the boundary band (the hybrid-explicit route, arXiv 2408.00771 sister). Queued for the generator build. Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-25k (/loop tick-12: curriculum bug reproduced in BOTH arms + R operator PRECISELY GROUNDED + R_cap/R_surv script SPEC ready for convergence-time write+test)
**LIVE MEASURED:** k=8 stage5_c1a ep739 d_seg **0.00369** (slow descent 0.00372→0.00369; now ep788, loss 1.355). k=16 ep488 hit the SAME stage3_smooth bump (0.00374→0.00394) — the C3 curriculum bug is now REPRODUCED in BOTH arms (deterministic, not noise); k=16 will recover in c1a as k=8 did. K-knee at matched stage3 ep488: k16 0.00394 / k8 0.00423 = −6.9%. Both healthy; ~11h to Muon-floor verdict. NO threshold crossing.
**R OPERATOR PRECISELY GROUNDED (replaces 8 ticks of approximate description — read from src/tac/torch_vehicle/driver.py + distortion_finishing_kit.py):** the eval round-trip R = bicubic-upsample(decoder 384×512 → camera 874×1164) → uint8-STE cast → bilinear-downsample(→ SegNet input 512×384) → joint-clip. CAMERA_H,W=874,1164 (distortion_finishing_kit.py:81). KEY CLARIFICATION: R does NOT change the resolution INTO SegNet (always 512×384) — it changes the PIXEL VALUES (uint8 quantize + resize blur). The canonical scorer call = `ScorerContext.seg_pose_forward(decoded_bhwc)` (driver.py:493) where decoded_bhwc=(B,2,384,512,3) float[0,255] is POST-round-trip (caller applies R); returns (seg_logits (·,5,384,512), pose6). GT = `seg_targets_hard` (n_pairs,384,512) int64 (the cached GT argmax). d_seg = mean(argmax(seg_logits) ≠ seg_targets_hard).
**R_cap/R_surv SCRIPT SPEC (exact, ready to write+test at convergence — NON-CONTENDING then):** (1) load trained decoder+latents from the screen best-ckpt (torch_vehicle checkpoint.py); (2) bind production ScorerContext (precompute_targets → distortion_net + seg_targets_hard + GT via frame_utils.yuv420_to_rgb); (3) decoder forward → frames_raw (B,2,384,512,3)[0,255]; (4) **R_cap path:** seg_pose_forward(frames_raw NO round-trip) → argmax_sharp; **R_surv path:** seg_pose_forward(R(frames_raw)) → argmax_rt; (5) per GT-flip (argmax_rt≠GT): R_cap if argmax_sharp≠GT (decoder failed to represent), R_surv if argmax_sharp==GT (R aliased a correct edge); (6) report R_cap/R_total = routing leverage ceiling. **NO-FAKE decision:** NOT shipping an untested API-guess script now (production ScorerContext construction + checkpoint-load API need a real test, which contends with the live arms); write+`py_compile`+RUN at convergence when an MPS slot frees and the verification is non-contending. The SPEC above is the durable artifact (survives compaction; any agent can implement it).
**CRUX UPDATE (§3):** R now exactly defined (pixel-value perturbation, not resolution change) — sharpens the round-trip-survival factor (R_surv = blur+quantize-induced flips, measurable via two scorer calls). Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-25l (operator "stage something in the meantime" → 3 non-contending deliverables staged: R_cap/R_surv tool [bg agent], boundary-routing module [bg agent], + the GENERATOR BUILD SPEC [this entry])
**Context:** operator asked for research/build/stage during the ~6h-to-Muon convergence window. Constraint = NON-CONTENDING (build+synthetic-test+py_compile only; no training/real-scorer/real-checkpoint runs — those contend with the 2 live MPS arms; they run at convergence). Launched 2 background BUILD agents: (1) `experiments/measure_r_cap_r_surv.py` (the decisive routing-leverage measurement, logic synthetic-tested, real-data path written+py_compiled-not-run); (2) `src/tac/torch_vehicle/boundary_routing.py` (boundary-distance prior + BoundaryFiLM directional-routing module + anisotropic-PE research). Both verified-on-completion before any DAG claim.
**THE FROM-SCRATCH GENERATOR BUILD SPEC (deliverable #3 — the decision-gated assembly recipe the 2 modules plug into):**
- **Base vehicle:** torch_vehicle bc20 / latent28 / n600 (the PROVEN byte-closeable vehicle; G3 packet 89KB rate 0.0594 parity-verified).
- **Lever 1 — capacity-ROUTING (dominant):** wire `boundary_routing.BoundaryFiLM` into the decoder, keyed on `boundary_distance_map(seg_targets_hard)` (the ~0-byte WHERE prior). Directional/anisotropic PE (curvelet-grounded) on the boundary branch = the BASIS sub-factor.
- **Lever 2 — round-trip-in-loop:** already active in the driver train step (eval-roundtrip bicubic↑→bilinear↓→uint8-STE, driver.py:23) — CONFIRM it stays on in the generator config.
- **Lever 3 — curriculum-fix:** skip/repair stage3_smooth (measured to RAISE d_seg, recovers in c1a but wastes epochs) + ensure Muon stage-8 finishes on d_seg (the finisher). Gated on the same-config curriculum-fix verification.
- **Lever 4 — activation:** the screen winner (FINER 0.002915 vs step_basis-k16 ~−5%; decided at the Muon-floor verdict, ~ep1500).
- **Byte-close:** reuse `experiments/build_torch_vehicle_d2_archive_zip.py` (G3 pipeline) + a bounded runtime add to the packet's `src/model.py` (apply the winning activation + read it from meta; the routing FiLM params ride in the state_dict — negligible rate).
- **FIRE GATES (means/ends — do NOT burn the run unless it can reach the END):** fire the generator ONLY when ALL hold: (a) screen verdict picks the activation; (b) R_cap/R_surv shows routing has LEVERAGE (R_cap/R_total high enough that routing+round-trip can plausibly reach d_seg≤~0.00087 from the Muon floor ~0.0021 — i.e. ≤~2.4× closable); (c) the projected joint d_seg ≤ ~0.001 (else bc20-standalone CANNOT beat 0.19110 — pivot to the composition/sidecar fallback or a different vehicle, don't burn a long run chasing a means). If R_surv DOMINATES (routing can't help), the lever shifts to round-trip-in-loop + sub-pixel #149, NOT capacity-routing.
- **Exact-eval = THE END:** byte-closed archive → tac.contest_score / upstream evaluate.py (CPU authority; CUDA if available) → a real row vs 0.19110.
**STATUS:** staged, NOT fired (gated on convergence + the R_cap/R_surv verdict). Pointer UNMOVED 0.19110; this is BUILD/STAGE work, not a measured score move.

### DAG FEED 2026-06-25m (both staged deliverables VERIFIED-LANDED — parent re-tested, not just agent-reported; + usage notes for convergence)
**Both bg build agents completed + PARENT-VERIFIED (NO-FAKE: re-ran tests + spot-checked bindings myself, did not trust the reports):**
1. **`experiments/measure_r_cap_r_surv.py`** (commit 90e7e69e1) — py_compile OK; `--self-test` PASS (re-ran: all R_cap/R_surv cases + edge cases, pure numpy non-contending); round-trip API binding spot-checked REAL (driver.py:2052 bicubic↑874×1164 → bilinear↓384×512 → clamp+round-STE — exact match). Binds the AUTHORITY scorer (distortion_net CPU), not MPS. **USAGE CAVEAT (convergence-time):** point `--ckpt` at a driver checkpoint dir WITH `torch_vehicle_checkpoint_manifest.json` (the screen run-dirs have one); the bare `best/` layout raises NotImplementedError BY DESIGN (no manifest → agent refused to mis-build the architecture = correct NO-FAKE). Real-data path UNTESTED-pending-convergence (running it loads the real scorer → contends).
2. **`src/tac/torch_vehicle/boundary_routing.py`** (+ tests; commit 71bd86646) — py_compile OK; **20 pytest PASS** (re-ran myself); real-GT demo VERIFIED on `experiments/results/capstone_gt_targets_cache/gt_targets_n6.pt` → lane(class-1) frac **0.00639** (matches the codim-1 ~0.0064 measurement). Public API: `boundary_distance_map` (the ~0-byte WHERE prior, scipy EDT), `boundary_proximity_feature` (exp(−d/τ) routing key), `BoundaryFiLM` (identity-at-init, +1168 params = ~1.4% of conv weights, modulation >20× stronger on-band vs interior), `local_boundary_tangent` + `directional_positional_encoding` (the oriented/anisotropic BASIS helper, 0 archive bytes). Research grounding: AFPE (arXiv:2509.02488, anisotropic Fourier features), SASNet (arXiv:2503.09750, spatially-adaptive frequency masks = capacity routing), steerable→curvelet (Freeman-Adelson/Candès-Donoho). Borrowed-substrate accounting in the docstring (scipy EDT + AFPE/SASNet mechanisms borrowed; OURS = keying on the GT class-1 lane boundary for the argmax-edge generator). **STANDALONE — NOT yet wired into driver.py/configurable_taper_decoder.py** (the per-stage injection is the generator-build step, gated on the screen verdict).
**Net:** the dominant-lever IMPLEMENTATION (capacity-routing: WHERE prior + FiLM gate + oriented-PE basis) and the DECISIVE MEASUREMENT (R_cap/R_surv) are now real, tested, committed code — ready to fire at convergence per the FEED-2026-06-25l generator build spec + fire-gates. Live MPS arms (k=8/k=16) confirmed undisturbed throughout. Pointer UNMOVED 0.19110 (build/stage work, no measured score move).

### DAG FEED 2026-06-25n (/loop tick-15: NEW signal — K-knee may be a FLOOR effect after all (re-opens tick-8); k=8 entered rate-reg stage6)
**LIVE MEASURED:** k=8 ep1024 advanced into **stage6_lambda_sweep** (left c1a; c1a plateau was 0.00370 over ep839–989). k=16 ep689 stage5_c1a **STILL DESCENDING: 0.00357→0.00348→0.00344** (has NOT plateaued) — broke BELOW k=8's c1a plateau 0.00370 by ~7%. NO threshold crossing (0.00344 ≫ FINER 0.002915 ≫ the ~0.00087 need).
**SIGNAL (honestly hedged — re-opens the tick-8 "K=speed-not-floor" correction):** tick-8 read the early-stage K-knee erosion (−7.9%→−1.9%) as "K is a convergence-SPEED knob, not a FLOOR knob." But in c1a, k=8 PLATEAUED at 0.00370 while k=16 keeps descending past 0.00344 (~7% lower, still going) — consistent with K buying a real LOWER FLOOR, not just speed. **CAVEAT (NO over-claim):** the two arms are at different epochs/stages (k=8 in stage6, k=16 still mid-c1a un-plateaued) → NOT apples-to-apples; the early-stage erosion was real too. RECONCILIATION HYPOTHESIS: the early gap (large) was mostly SPEED; a smaller residual (~−5–7%) is a genuine FLOOR effect that only becomes visible once each arm plateaus per-stage. The CLEAN verdict is both arms' Muon-stage-8 floors at ep1500 — do not call it from mid-c1a. PLANNING: this strengthens k=16 as the likely generator activation-chart pick (more edges → lower floor at ~0 rate).
**WATCH (k=8 rate-reg stage):** stage6_lambda_sweep + stage7_sigma are the rate-regularization stages the curriculum-bug analysis (FEED 2026-06-25h) flagged as potentially d_seg-RAISING; watch k=8's stage6/7 evals for a d_seg rise, then whether Muon stage-8 recovers it (the MUONJUMP hypothesis: rate-reg stages raise the floor that Muon could otherwise reach). This is the live test of the curriculum-fix lever (#3).
**Cadence:** k=8 ~ep1024, Muon stage-8 ~ep1300 (~3h at ~42s/ep); approaching — next ticks tighten toward 1800s. Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-25o (operator Q&A synthesis — WHERE the residual is + reducibility + waterfilling + residual-aware; reconciles the floor verdicts)
Operator asked: (a) is the residual horizon vs lane vs long-tail, are we biting it; (b) small/sparse → inherently compressible → optimized waterfilling; (c) is residual-aware part of score-aware. Synthesis grounded in measured verdict files:
- **WHERE (measured):** binding residual = **class-1 LANE markings** (27.6 comp/frame, 0.72% px, drop-d_seg 0.00705). **HORIZON = NO-GO ×3 / label-noise-like** (ΔS ceiling 0.012–0.024; flips not geometry-codeable). **HOOD interior FREE** (only moving edge, 7.4% flips). **LONG TAIL = 71× over-concentration of flips at LOW GT-margin** (the ambiguous label-noise frontier). We ARE biting it (generator targets the lane band; correctly avoids horizon).
- **RECONCILED FLOOR (the good news):** `dseg_reducibility`=IRREDUCIBLE was about WHERE OUR under-capacity bc20 residual SITS (jammed at the label-noise frontier), NOT an absolute floor. `dseg_384_achievability`=CAPACITY-LIMITED (384+uint8 pipeline floor ~1.6e-4, ~11× below our 0.0021 — pipeline is NOT the wall). `label_noise_floor_RESOLUTION`: the frontier vehicle ALREADY achieves d_seg ~0.0003 (BELOW the proxy τ=0.137 label-noise floor 0.00123) = existence proof the floor is NOT a hard wall. ⇒ absolute floor ~0.00016–0.0003 < our NEED 0.00087 → **the residual IS reducible to where we need it**; bc20 is under-capacity at the boundary, not floored. (Caveat: frontier hits 0.0003 at 177KB; bc20 at 89KB reaching 0.00087 is plausible-not-guaranteed — the generator run + exact eval measures it.)
- **(b) small/sparse/compressible/waterfilling — RIGHT, with the measured caveat:** residual is low intrinsic dim (~8-dim NONLINEAR manifold, 0.72% px) = inherently compressible, BUT full-rank in every LINEAR basis (rank 53/60) → a linear sparse "store-the-flips" sidecar is NO-GO ×3; the compressibility is NONLINEAR (trained chart). Waterfilling is the correct frame for the CAPACITY ALLOCATION (fill decoder capacity to equal marginal-d_seg value at the lane band = hp-FEM routing = reverse-water-fill #157 = boundary_routing.py). The operator's phrasing NAMES the dominant lever (capacity-routing-as-waterfilling).
- **(c) residual-aware ⊂ score-aware — half-yes:** the LOSS is already residual-aware (driver SegNet argmax-flip surrogate puts gradient only on flipping/near-flipping pixels, zero on confident interior). The CAPACITY is NOT yet residual-aware (uniform param spend) — that is exactly the routing/waterfilling lever the generator adds. So: score-aware ⊇ residual-aware-LOSS (have) + residual-aware-CAPACITY (the build adds it).
- **CRUX SHARPEN (§3):** the binding residual is the lane long-tail (shallow-margin boundary flips); the absolute floor (~0.0003) is BELOW the need (0.00087) so it is reducible; the optimal attack = residual-aware-CAPACITY waterfilling (fill DOF to the 0.72% band) on top of the already-residual-aware loss, via a NONLINEAR trained chart (linear sparse sidecar is dead). All three operator framings converge on the capacity-routing lever. Pointer UNMOVED 0.19110.

### DAG FEED 2026-06-25p (operator: "ALL deep-math — algebra/topology/calculus/geometry/manifold/optimal vs FROZEN contest info" — full-lens grounding of the residual/waterfilling/residual-aware crux)
FROZEN objects (the fixed information space everything is optimized against): S=SegNet→argmax partition {C₀..C₄} of Ω=384×512; P=PoseNet (6-dim, ~solved); R=D∘Q∘U (bicubic↑874×1164 → uint8 → bilinear↓512×384, a fixed contraction); the n600 pairs; S_score=100·d_seg+√(10·d_pose)+25·bytes/N. Ground the crux in 7 lenses:
- **TOPOLOGY/MANIFOLD:** d_seg-target = piecewise-constant Ω→{0..4}; binding residual = class-1 lane = ⊔ ~27.6 tubular nbhds of centerline curves γᵢ (medial-axis M=T(γ,w), w≈const). Across n600, the lane config = a curve in config space, measured ~8-dim NONLINEAR manifold (AE-knee 8 / MLE 13). The residual is codim-1 (1-D boundary in 2-D Ω) × low-dim temporal manifold.
- **ALGEBRA/GROUP THEORY (the key new piece):** the n600 lane manifold ≈ the ORBIT {g·γ₀ : g∈G} of a template under an ~8-dim motion group G (ego-motion × lane geometry). A MOVING edge is full-rank in any FIXED linear basis (every swept pixel = a dimension → measured rank 53/60) but 8-dim under the nonlinear "where is the curve" chart = the group action. ⇒ the generator must represent g· (an O(8) smooth map), NOT store orbit points (O(n·comp)). This is the rigorous reason residual-aware CAPACITY (encode the action) ≫ residual SIDECAR (store the orbit, NO-GO ×3) and why "compressible" requires NONLINEAR.
- **CALCULUS/VARIATIONAL (waterfilling = KKT):** min_c d_seg s.t. ∫_Ω c(x)dx ≤ P. KKT optimum equalizes the marginal ∂d_seg/∂c(x)=λ (a common "water level") across the support; ∂d_seg/∂c(x) = the per-pixel d_seg-SENSITIVITY = the margin-saliency field ∂margin/∂input (#141). ⇒ "water-bucket filling" is LITERALLY: pour capacity until the saliency-weighted marginal equalizes = reverse-water-fill (#157) on the saliency map. Bucket depth(x)=saliency(x); water level=dual of the byte/param budget. NOT a metaphor — the exact optimality condition.
- **GEOMETRY (differential — ties activation⊗round-trip):** the flip set = the zero-level-set of the post-R margin field Δ(x)=logit_gt−logit_2nd. d_seg = measure of the band where R perturbs Δ across 0; band width ∝ (R-perturbation)/|∇_⊥Δ| (normal gradient). MEASURED: 71× over-concentration of flips at low |Δ| ⇒ the margin is SHALLOW (small |∇_⊥Δ|) at the boundary ⇒ tiny R-perturbation flips a wide band. LEVER: steepen |∇_⊥Δ| across the curve → narrow the flip band. A STEP-native activation produces a steep logit transition ⇒ large |∇_⊥Δ| ⇒ thin band. So lever-4 (step activation) and lever-2 (round-trip survival) are the SAME |∇_⊥Δ| mechanism, unified geometrically.
- **INFO-THEORY/MDL/indirect-RD (CEO):** the task-optimal code transmits the SUFFICIENT STATISTIC for the argmax decision = the lane-curve params (8-dim manifold coords) + margin sign — NOT pixels. Frozen-contest-optimal residual rate ≈ H(8-dim trajectory over 600 frames), heavily temporally-correlated → ~hundreds of bytes after AR coding. This IS "inherently compressible": the indirect-RD statistic is 8-dim, not |Ω|. The INR weights ARE this code (COIN/COIN++ weights-as-code).
- **SET THEORY/MEASURE:** d_seg = μ(A△B), symmetric-difference measure of the pred vs GT class-1 SETS; the residual is the thin annulus A△B around the shared boundary. Minimizing μ(A△B) = aligning the two partition boundaries; residual-aware = put representational measure on the annulus, zero elsewhere (𝟙_{annulus}).
- **OPTIMAL (the joint, vs frozen S_score):** the d_seg term dominates the controllable budget (rate fixed ~0.0594, pose free); the frozen-contest optimum = the minimum-description-length NONLINEAR chart of the 8-dim lane-orbit manifold, capacity WATERFILLED (KKT-equalized marginal = reverse-waterfill on the saliency field) to the codim-1 boundary annulus, rendered with a STEEP-margin (step) activation so R can't alias the boundary, byte-closed in the 89KB L13 packet.
**RESIDUAL-AWARE ⊂ SCORE-AWARE (formal):** score-aware = train on S∘R (the frozen scorer through the round-trip); residual-aware-LOSS = the argmax-flip surrogate weights ∝ 𝟙_{A△B} (gradient only on the annulus) — HAVE it; residual-aware-CAPACITY = c(x) ∝ KKT-waterfill(saliency) — the build ADDS it. Both are sub-cases of optimizing S∘R; the second is the missing one.
**NO-FAKE ledger:** MEASURED = 8-dim manifold, rank 53/60, 71× low-margin concentration, saliency map, floor ~0.0003<need 0.00087. DERIVED (this feed) = group-orbit algebra, waterfill=KKT-on-saliency, steep-margin=thin-band geometry, indirect-RD 8-dim statistic. No score moved; pointer UNMOVED 0.19110. These ground the generator design (boundary_routing.py = the c(x) waterfill; step activation = the |∇_⊥Δ| steepener) against the frozen objects.

### DAG FEED 2026-06-25q (operator: ESTABLISH the witness capstone as the new frontier + CHROMA lever + OPTIMAL-FORM — canonicalized, not a measured row)
**Operator directives (2026-06-25, verbatim):** *"Build the witness capstone"* + *"Chroma too"* + *"implementations are not optimal yet"* + *"Update your memories and Claude.md and establish as new frontier and focus and priority."* Done this unit (a crux-CANONICALIZATION, no score moved — pointer UNMOVED contest-CPU 0.19110):
- **CLAUDE.md** now carries §"THE CURRENT FRONTIER + FOCUS + PRIORITY — THE NON-RGB TASK-SPACE WITNESS CAPSTONE — NON-NEGOTIABLE" (committed `ddf6724ff`): the vehicle (ScoreNativeSegGenerator nonlinear coord-INR + amortized luma+chroma pose carrier + boundary_routing KKT), the trilemma (bc20 under-capacity / bc36=just-PR95 / witness=both), chroma, optimal-form, the anti-PR95-reskin NO-FAKE #7 lesson, the 4 d_seg levers.
- **Memory** CURRENT-STATE + MEMORY.md line 8 updated to the witness-capstone-as-frontier framing; pr95-reskin-fake feedback memory linked (line 14). Task #171 → in_progress.
**CHROMA deep-math (the new lever, frozen-contest-grounded):** the two scorers read DIFFERENT color spaces — **SegNet argmax on RGB**, **PoseNet on YUV6 (4 luma + 2 chroma)**. Therefore: (a) seg-frame has RGB-slack → chroma channels carry argmax-relevant signal; route capacity into chroma where it flips the codim-1 boundary annulus (𝟙_{A△B}). (b) the pose carrier MUST be luma+CHROMA — a luma-only carrier is provably lossy for PoseNet (discards 2 of 6 input planes) → the measured pose collapse (d_pose 2.67–12.66) in the first composed candidate. Chroma EXPANDS the realizable witness set on BOTH score terms. Any pre-chroma witness verdict is provisional → re-measure with chroma active.
**OPTIMAL-FORM discipline (operator):** no witness verdict (adopt/kill/"lever doesn't move d_seg") is load-bearing until per-lever hyperparams tuned to each lever's OWN optimum + chroma active + routing engaged; sub-optimal-form results LABELLED provisional (deterministic-reproducibility spine).
**Live state:** capstone-build subagent (a922483dfc636ccc3) running — a 32-pair fast loop sweeping which deep-math levers move d_seg (control reaches d_seg 0.005948@150ep, still descending). Chroma-orthogonal foundational work (characterize base levers first; chroma folds in next unit). SendMessage not wired this harness → chroma captured durably for the next unit, not injected mid-run. NOT disturbed.
**NO-FAKE:** this feed is canonicalization + a derived lever (chroma color-space asymmetry is a FROZEN-contest fact: SegNet RGB / PoseNet YUV6); no score claim, pointer UNMOVED 0.19110. The END remains a byte-closed exact-eval row below 0.19110.

### DAG FEED 2026-06-25r (operator CORRECTION: pose is SOLVED via the Quantizr-style stored-target sidecar — supersedes the FEED-q "luma+chroma pose carrier / pose collapse" framing)
**Operator (2026-06-25):** *"Pose is pretty much solved with the Quantizr style sidecar we already have built."* CORRECT — and it corrects MY framing (FEED-q + the first CLAUDE.md witness section treated pose as open and cited the "pose collapse" as a reason to build a luma+chroma pose carrier). GROUNDED in the built artifact:
- **Mechanism (frozen-contest fact):** scorer computes `d_pose = MSE(PoseNet(generated_pair)[:6], PoseNet(original_pair)[:6])` (src/tac/scorer_targets.py:9-13). The GT target is just the 6 PoseNet scalars/pair. We STORE them (`extract/save/load_posenet_targets`, 600×6×fp16 = 7.2KB raw / <5KB zlib) → supervised-condition the render to hit the exact targets (NOT self-supervised guessing). Further compressible: `src/tac/pose_from_embedding.py` MLP ~1–2KB (replaces optimized_poses.pt ~15KB); low-rank pose codec 2.7× (task #140); `openpilot_seeding.py` seed_poses.pt (N,6) ~7KB.
- **Result:** d_pose ~3.4e-5 (frontier), contribution `√(10·d_pose)` ~0.018, near-free bytes ⇒ POSE SOLVED.
- **The "pose collapse" reconciliation:** d_pose 2.67–12.66 was the **amortized-luma-CARRIER** composition (an INR that RECONSTRUCTS pose-bearing luma — task #57/#163 AmortizedLumaCarrier), a different/suboptimal approach — NOT the stored-target sidecar. Do NOT cite the collapse as a reason for a chroma pose carrier.
**Corrected witness framing (canonicalized CLAUDE.md `a71a8306b` + CURRENT-STATE memory):**
- The witness's SOLE binding controllable job is **d_seg**. Pose rides the already-built Quantizr-style stored-target sidecar (compose, don't rebuild).
- **CHROMA = a d_seg LEVER, not a pose rescue:** SegNet reads RGB so its argmax depends on chroma; seg-frame RGB-slack → route capacity into chroma where it flips the boundary annulus. (Secondary: chroma planes also feed PoseNet, but pose is on the sidecar so chroma is optimized for d_seg first.)
**NO-FAKE:** correction grounded in scorer_targets.py source + the d_pose~3.4e-5 frontier measurement; no score moved, pointer UNMOVED 0.19110. Net effect: the witness build is SIMPLER (d_seg-only) — pose is a solved compose-in, not a co-design blocker.
