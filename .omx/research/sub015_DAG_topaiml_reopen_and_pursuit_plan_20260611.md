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

### NEW NODE — TOPOLOGY (deep-math, frozen-instance): d_seg is a PARTITION-topology problem = (near-constant region adjacency/components) + (codimension-1 boundary moving low-dim with EGO-MOTION). RGB decoder wastes capacity on trivial interiors, starves the boundary (= the cliff + spectral bias). Frozen-optimal code: constant template + ego-driven boundary DEFORMATION + RENDERED sub-pixel boundary (not flat-store). UNIFICATION: one ego-trajectory drives BOTH d_seg boundary AND d_pose. $0 gate: af64e924 (topology-constant? boundary low-dim/ego-explained vs a3061 flip-residual rank). Architecture match: contour/wavelet (Daubechies).

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
