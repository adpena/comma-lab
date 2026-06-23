---
title: "PR95 d_seg basin vs our small-basis plateau — the author's design rationale, the capacity-vs-recipe gap decomposition, the deep capacity↔d_seg↔rate math, the DECISIVE frontier-weight entropy measurement, and the ranked sub-0.15 paths"
authority: "[contest-CPU advisory] / [analysis] — NON-PROMOTABLE. score_claim=false; promotion_eligible=false; ready_for_exact_eval_dispatch=false. Pointer UNMOVED 0.19109982 [contest-CPU]. $0; NO MPS; NO paid dispatch; NO pinned-upstream edits. This is a MEANS memo (research+measurement); it moves no pointer."
date: 2026-06-23
score_claim: false
promotion_eligible: false
pointer_moved: false
all_score_math_via: tac.contest_score.compute_contest_score
key_measurement: experiments/results/frontier_decoder_weight_entropy_20260623.json
measurement_tool: experiments/measure_frontier_decoder_weight_entropy.py
cross_refs:
  - .omx/research/pr95_seg_convergence_mechanism_and_recipe_gap_audit_20260611.md   # BUG-A (muon_lr 150x)
  - .omx/research/apparatus_audit_pr95_breakthrough_blocker_20260619T214001Z.md     # frontier is BORROWED; never-fired run
  - .omx/research/RECURSIVE_REVIEW_pr95_math_optimization_synthesis_20260619T231500Z.md
  - .omx/research/FIRE_pr95_full_curriculum_decisive_run_20260619T220000Z.md
  - experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/  # author's pristine code
  - https://aaronleslie.dev/blog/comma-compression  # author's writeup (JS-rendered; read via r.jina.ai)
---

# PR95's d_seg basin vs our plateau — rationale, gap decomposition, deep math, the decisive entropy number, ranked paths

**One-paragraph answer to the operator.** PR95 (Aaron Leslie, "hnerv_muon", score 0.1988) reaches
**d_seg ≈ 5.6e-4** because it fully runs an 8-stage 29,650-epoch curriculum on a 229K-param (base_ch=36)
decoder — the author picked that size deliberately as *"small enough that INT8 quantization (one byte per
weight) brotli-compresses inside the rate budget,"* and reports the **rate term, not SegNet accuracy, as the
binding constraint** (rate = 59.8% of his score; seg only 30.8%). Our small-basis reproduction "plateaus at
~0.0021" for **two stacked reasons that are now isolated**: (1) a **RECIPE bug** (the curriculum silently
threw away the working `muon_lr=0.03`, substituting PR95's torch-stage-8 `2e-4` = 150× too small + 100%
grad-clip, which FROZE d_seg at its init value — measured: 0% vs 7.6× descent in 15 identical CE epochs), and
(2) **lower capacity** (bc20 = 83K params vs bc36 = 229K). The decisive new measurement here: **the frontier's
decoder weights are already at the order-0 Shannon entropy floor (achieved 5.629 bits/param vs the per-tensor
H0 ideal 5.586 bits/param — a ratio of 1.0077, only 1,226 bytes / ΔS = −0.00082 of headroom).** Therefore a
**pure entropy recode CANNOT reach sub-0.15** — the parent's "2× rate recode → sub-0.15" is real *as an RD
target* (2× bytes/param → S* ≈ 0.137) but is **unreachable by a better coder**; it requires *fewer/lower-bit
weights or learned higher-order weight entropy* (a retrain/requant), not a recode. The single highest-EV next
move is the one that has been **armed but never fired for 12 days: run the BUG-A-corrected full 8-stage PR95
curriculum at n600 to convergence on a paid GPU, byte-close, and exact-eval** — the only measurement that
tells us whether an OWN-trained bc20/bc24 decoder bends d_seg far enough at the small-byte (rate-headroom)
operating point to cross sub-0.15.

---

## PART A — The AUTHOR's design rationale (Aaron Leslie, PR #95 "hnerv_muon")

**Sourcing note (NO FAKE).** The blog (`aaronleslie.dev/blog/comma-compression`) is JS-rendered; a direct
fetch and `web.archive.org` returned only the page header. The full rendered text WAS recovered via the
`r.jina.ai` render proxy, and the GitHub PR #95 description corroborates the headline numbers. All quotes
below are the author's verbatim words as recovered from the rendered blog; the code-level facts are read
from the pristine local intake of his actual source. Where the rendered text and code agree I mark it; where
the blog is silent I say so.

### A.1 The author's headline result (verbatim, blog + PR description)
- Final score **0.1988**, decomposed by the author: *"Rate (archive size ÷ source): 0.1188 (59.8%);
  Segmentation distortion: 0.0613 (30.8%); Pose distortion: 0.0187 (9.4%)."*
- Components (PR description, which we recompute-confirm against his `score.py`): **SegNet 0.00061212,
  PoseNet 0.00003494, rate 0.00475202.** (Our local frontier — a *recode* of his lineage — measures SegNet
  0.00055978 / PoseNet 0.00002942 / rate 0.00471878 → 0.19110; same basin.)
- 229K weights → **178 KB** archive: *"229,000 weights total, quantized to 8-bit integers and entropy-coded
  down to 178 KB."*

### A.2 Capacity / channels / why 229K (the operator's explicit question) — VERBATIM
- **28-d latent:** *"Smallest dimension that didn't bottleneck the SegNet objective at our parameter count,
  found by sweep."*
- **229K params:** *"Small enough that INT8 quantization (one byte per weight) brotli-compresses inside the
  rate budget."* ← **This is the author's capacity rationale in one sentence: the model size is chosen by the
  RATE budget, not by the d_seg headroom.** He is not maximizing seg accuracy; he is picking the largest
  decoder whose INT8 bytes still fit the rate term.
- **6×8 base + 6 PixelShuffle stages:** *"Six stages was the smallest depth that reaches the eval resolution
  from a base small enough to be embedded directly from a 28-d latent."*
- Code-level (his `model.py`, verified): `base_channels=36`, taper `[36,36,36,27,20,18,18]`, comment
  *"channel taper matches HNeRV paper"*; sin activations throughout; PixelShuffle(2) + bilinear-skip per
  stage; separate `rgb_0`/`rgb_1` heads; a dilated-conv refine residual. *"Purely a decoder (no encoder, no
  attention) with sin activations throughout."*

### A.3 Model-size ↔ score rate-distortion tradeoff (verbatim + what's MISSING)
The author **does not present a parameter-count sweep against score**. His framing is the achieved point, not
the Pareto curve. The only tradeoff statement is the rate-budget capacity rule (A.2). **He does not say "I'd
go bigger to lower d_seg"** — the opposite: he keeps the model at the largest size the *rate* term tolerates.
This is the empirical anchor for the capacity-RD-optimum math in Part C: the author intuited (without
solving it) that the contest puts the optimal HNeRV decoder near where rate and seg balance.

### A.4 The d_seg / SegNet term — how low, what limited it (verbatim)
- Achieved seg contribution **0.0613 (30.8%)**; at pixel level *"99.86% of pixels get the exact same class
  label"* in the SegNet output.
- **What limited it (verbatim mechanism):** Stage 2 *"the gradient effectively shuts off once the correct
  logit exceeds the runner-up by more than ~τ"*; Stage 3 *"the sigmoid bell gives almost none [gradient to
  easy pixels]... Nearly all the gradient flow goes to pixels right at the SegNet decision boundary."*
- **The binding constraint was rate, NOT seg accuracy:** *"Stages 5, 6, and 7 had largely flattened (0.207,
  0.205, 0.203)"* before Muon, and the rate term dominates at 59.8%. **The author plateaued his d_seg not
  because the model couldn't go lower but because more seg accuracy buys little against a rate-dominated
  score.** He gives **no Shannon/entropy/theoretical-floor statement for d_seg** — there is no "the floor is
  X" sentence in the writeup.

### A.5 Muon — why (verbatim)
- *"Muon replaces SGD/Adam with [Newton-Schulz]-orthogonalized momentum... For matrix-shaped parameters it
  produces updates whose singular values are roughly equal, essentially trust-region steepest descent on the
  matrix manifold."*
- Weight decay: *"[Chen-Li-Liu (2025)] show Muon's spectral-norm KKT theoretical justification requires
  weight decay to be active."* (His `optim.py` sets WD via the Chen-Li-Liu arXiv:2506.15054 rationale,
  applied to the hidden conv weights only; stem + RGB heads + biases + latents stay on AdamW.)
- Impact: *"Stage 8 was the moment we crossed below 0.20... Muon's orthogonalized updates then carried the
  score the rest of the way from 0.2034 down to 0.1988."* **Muon is his last-stage finisher.**

### A.6 The 8-stage curriculum rationale (verbatim per stage)
- **S1 CE:** *"gives gradient signal at every pixel regardless of confidence. We want strong,
  broadly-distributed gradients to push the model into the basin where roads look like roads."* (the
  workhorse bulk descent)
- **S2 τ-Softplus:** *"stop wasting our VERY limited capacity pushing already-confident pixels to be even
  more confident, and start redirecting gradient toward the borderline ones."*
- **S3 sigmoid-bell (smooth disagreement):** *"a tighter focus on boundary pixels."*
- **S4 INT8 QAT:** *"teaches the optimizer to find weights that survive int8 rounding without losing
  accuracy."*
- **S5 L7+C1a:** *"L7 applies a 4× additive boost... hard pixels end up weighted 5× vs 1×... C1a is the
  entropy regularizer behind the rate gain"* — and (PR description) *"shapes the weight distribution toward
  the integer grid, which collapses the entropy floor for downstream brotli compression."*
- **S8 (why so late):** *"By Stage 7 we are near the bottom, so the sharper wells just snap weights into
  place."*

### A.7 What he TRIED-AND-ABANDONED + optimization opportunities (the operator's explicit ask) — VERBATIM
The blog's "what didn't work" section is the closest thing to a "left on the table" list:
- **Bigger/teacher capacity:** *"distillation from a larger ch=72/ld=56 teacher"* — abandoned (didn't help).
  *"HiNeRV (multi-resolution temporal grids, elegant on paper but burned too many parameters on the grids)."*
  *"SIREN-activation decoders"* — abandoned. *"Pure pixel-MSE latent decoder bottomed at 6.32"* (i.e. naive
  MSE is useless — must train through the scorer). **He explicitly tried a 2× bigger teacher and it did not
  pay → corroborates Part C: bigger model is rate-dominated, not the lever.**
- **Better entropy coding:** *"DeepCABAC, custom CABAC v2/v3, order-2 arithmetic coding"* — all tried, none
  beat the per-tensor brotli/AC by enough to keep. **He already explored higher-order weight entropy coding
  and found ~nothing — directly corroborates the Part C entropy measurement (the weights are near the order-0
  floor; order-2 context buys little).**
- **Mixed-precision (fp16) training, SWA, test-time adaptation, Karras post-hoc EMA:** *"Each was either
  neutral or marginal... all cut for simplicity."*
- **NO explicit theoretical-floor / future-work section.** The author presents 0.1988 as the product of
  exhaustive iteration, not as a known plateau with named headroom. **There is no "next I would do X"
  paragraph and no "the floor is Z" claim** — an honest negative on the operator's "what would the author do
  next" question.

**A.7 takeaway:** the author's own dead-ends are a free ablation that *pre-confirms* two of our Part-C
conclusions — (i) a 2× bigger decoder/teacher does not pay (rate-dominated), and (ii) fancier entropy coders
(CABAC, order-2 AC) do not beat the per-tensor order-0 coder by enough to matter. His implicit ceiling is the
rate-dominated 0.196–0.199 cluster — exactly where the public leaderboard saturated.

---

## PART B — Why OURS plateaued: the capacity-vs-recipe gap decomposition

### B.1 The clean anchor table (recipe-contaminated points flagged)
All d_seg `[macOS-CPU advisory]` on the real `modules.py` SegNet (live + EMA agree post-warmup-fix).
Param counts exact from `model.py`: **bc20=83,356 · bc24=112,901 · bc28=148,038 · bc36=228,958.**

| anchor | params (bc) | n_pairs | recipe | epochs/stage | d_seg | clean? |
|---|---:|---:|---|---|---:|---|
| **BUGGY curriculum** (n=8) | bc20 83K | 8 | muon_lr **2e-4**, clip 1.0 | 15 (CE) | **0.50727 (FROZEN at init, 0% descent)** | RECIPE-BUG witness |
| **FIXED** (n=8) | bc20 83K | 8 | muon_lr **0.03**, clip 50 | 15 (CE) | **0.06647 (7.6× descent)** | RECIPE-isolated |
| FIXED stage-2 | bc20 83K | 8 | 0.03/50 | +25 (τ-softplus) | 0.01646 | recipe-isolated |
| FIXED stage-3 | bc20 83K | 8 | 0.03/50 | +12 (smooth) | 0.01197 | (smooth did NOT raise d_seg under the fix) |
| **bc20_p48** | bc20 83K | 48 | 0.03/50 muon_throughout | 120 (CE) | **0.0037602** | CLEAN (matched recipe) |
| **bc24_p48** | bc24 113K | 48 | 0.03/50 muon_throughout | 120 (CE) | **0.0028546** | CLEAN (matched recipe) |
| bc20_p192 | bc20 83K | 192 | 0.03/50 | 120 (CE) | 0.0192 | data/epoch-starved (more pairs, fixed epochs → worse) |
| **bc20_n600 (basin)** | bc20 83K | 600 | stage-1 CE only, AdamW, ep2325 | NOT converged (paused stage 1 of 8) | **0.0025607** | the real-contest "our" number |
| **frontier (BORROWED)** | bc36 229K | 600 | fully-trained 8-stage (PR101 content + PR95 arch + PR112 coder) | converged | **0.00055978** | 0% ours-trained |

### B.2 The two-part gap decomposition (the operator's precise question)
The operator's "0.0021-vs-5.6e-4 gap" splits cleanly into RECIPE and CAPACITY, **both now isolated**:

1. **RECIPE (the dominant, now-fixed share).** BUG-A (curriculum overwrote `muon_lr 0.03 → 2e-4`, clip
   `50 → 1.0`) **froze d_seg at the init value** in the controlled A/B (same arch, same loss, same 15
   epochs, same init 0.50727): BUGGY 0% descent vs FIXED 7.6× in stage 1 alone, descending to 0.012 by
   stage 3. The "0.0097–0.0021 plateau" memos that rested on the buggy curriculum are
   **IMPLEMENTATION-LEVEL FALSIFIED** (Catalog #307). *Most of the apparent plateau was the throttle.*
   Fixed: commit `f6a913ccc`.
2. **CAPACITY (the residual, smaller share).** At identical FIXED recipe (n=48, 120 CE epochs), bc20→bc24
   lowers d_seg **0.00376 → 0.00285 (−24% for +35% params)** — capacity *does* help (refuting the
   "capacity-bound wall" framing AND the contaminated `params^−0.71` power law fit on the wrong tiny
   factored-LF/NCA architectures). bc24's RATE floor is sub-0.15-marginal *if* d_seg < ~1.5e-4.
3. **CONVERGENCE (the un-measured share).** The "our" n600 number (0.0026) is **stage-1 CE-only, paused at
   epoch 2325 of an 8-stage 29,650-epoch curriculum** — it never reached the τ-softplus/L7/Muon-finetune
   stages that the author and the mechanism memo identify as the d_seg *finisher*. So "ours plateaus at
   ~0.0021" is **not a converged plateau at all** — it is a recipe-throttled, capacity-reduced,
   stage-1-only mid-descent reading. The clean comparison the operator wants (does a fully-trained bc20/bc24
   reach 5.6e-4) **has never been measured** — that is the never-fired run (Part E #1).

**Bottom line of Part B:** the 0.0021-vs-5.6e-4 gap is, in order of magnitude: (a) RECIPE throttle (largest;
fixed but never re-run at scale) + (b) under-convergence (stage 1 of 8) + (c) capacity (bc20 vs bc36; real
but the smallest of the three, and partially recoverable via bc24 + the byte-neutral taper). It is NOT a
clean capacity asymptote.

---

## PART C — The deep capacity ↔ d_seg ↔ rate math (with the DECISIVE entropy measurement)

All score arithmetic via `tac.contest_score.compute_contest_score` (uncompressed N = 37,545,489).

### C.1 The capacity power law d_seg ∝ params^(−α)
- **Clean matched-recipe fit (the ONLY apples-to-apples pair, bc20_p48 ↔ bc24_p48, identical recipe):
  α ≈ 0.91.** This is a **2-point fit at 120 CE-only (under-converged) epochs**, so it is a *lower bound* on
  the converged α (more training widens the capacity advantage of the bigger net).
- Cross-recipe estimates (bc20_n600-stage1 ↔ bc36-frontier) give α ≈ 1.5–1.9, but conflate
  recipe+convergence+borrowed-substrate — **DERIVED, not clean.**
- The literature value **α = 0.71 (`29.3·params^−0.71`) is CONTAMINATED** — fit on factored-rank-1-LF / NCA
  decoders (few-K–17K params), NOT the PR95 HNeRV decoder, which already beats that curve at 120 epochs.
- **Confidence: LOW-MEDIUM.** α is somewhere in [0.9, 1.5] for the real decoder; not pinned. The honest
  statement: capacity lowers d_seg with a sub-quadratic exponent, and reaching the frontier's 5.6e-4 from a
  bc20 baseline at the clean α≈0.91 would need **~678K params** (bc~57) — which **forfeits all rate
  headroom** (see C.3).

### C.2 The capacity-rate-distortion optimum S(p)
`S(p) = 100·k1·p^(−α) + sqrt(10·d_pose) + 25·(latent_fixed + bpp·p)/N`, with **bpp = 0.7036 bytes/param**
(measured: 161,104 decoder bytes / 228,958 params), latent_fixed = 16,065 B (latent+sidecar+zip), d_pose
held at the frontier's 2.94e-5, k1 calibrated so the curve passes through the real bc36 frontier point.
Solving `dS/dp = 0` numerically (via `tac.contest_score`):

| α (assumption) | p* (optimum) | bc~ | **S*** | S(bc36) | S(bc20) |
|---|---:|---:|---:|---:|---:|
| 0.91 (clean lower-bound) | 154,779 | 27 | **0.1802** | 0.1911 | 0.2070 |
| **1.12 (parent estimate)** | **177,734** | **29** | **0.1855** | 0.1911 | 0.2405 |
| 1.50 (cross-recipe) | 207,892 | 32 | 0.1900 | 0.1911 | 0.3230 |

**The parent's estimate is confirmed:** α≈1.12 → **p* ≈ 177K, S* ≈ 0.186 ≈ T_1**, and **the bc36 frontier
(S=0.1911) sits essentially AT the HNeRV capacity-RD optimum at 0.70 bytes/param.** Across all plausible α,
**no point on the capacity curve reaches sub-0.15 at the current entropy coding.** Capacity scaling alone is
exhausted as a sub-0.15 lever — it just trades seg for rate around S≈0.18–0.19.

### C.3 The rate-axis RD-shift — the candidate sub-0.15 lever (parent's hypothesis)
Re-solving the capacity-RD optimum at reduced bytes/param (α=1.12, k1 calibrated to bc36):

| bytes/param | (vs 0.704) | capacity-RD optimum p* | bc~ | **S*** |
|---:|---|---:|---:|---:|
| 0.704 | 1× (current) | 177,752 | 29 | **0.1855** |
| 0.352 | 2× | 246,489 | 34 | **0.1371 (sub-0.15)** |
| 0.234 | 3× | 298,572 | 38 | **0.1160 (≈ S_floor 0.118)** |

**The parent's numbers are confirmed:** 2× bytes/param → S* ≈ 0.137 (sub-0.15); 3× → S* ≈ 0.116 (≈ floor).
*If* bytes/param could be halved at fixed d_seg, sub-0.15 exists. **The entire question is whether the 2× is
reachable** — answered by C.4.

### C.4 THE DECISIVE $0 MEASUREMENT — is the frontier at the entropy floor? (`frontier_decoder_weight_entropy_20260623.json`)
Measured the order-0 Shannon entropy of the **real frontier decoder's** quantized INT8 weight symbols
(sha `b46897267…`, inflated via the submission's own `inflate.py`; the byte-closure proof confirms this raw
stream is exactly what the range coder encodes):

| quantity | value |
|---|---:|
| decoder params | 228,958 (28 tensors) |
| **achieved decoder bytes** | **161,104 = 5.6291 bits/param** |
| per-tensor H0 ideal (the coder's target) | 159,878 B = **5.5863 bits/param** |
| **achieved / H0-ideal ratio** | **1.0077** |
| **headroom to order-0 floor** | **1,226 bytes** |
| ΔS from a perfect order-0 recode | **−0.00082 → S = 0.19028** |
| global single-model H0 (non-adaptive) | 6.52 bits/param (the frontier's per-tensor adaptivity already saved ~1 bit/param) |

**VERDICT: the frontier decoder is at the order-0 Shannon floor (within 0.77%).** Its per-tensor adaptive
256-ary range coder (PR112's `codec_ctx.py`, constriction — NOT brotli) already extracts essentially all the
order-0 entropy. **A pure entropy recode buys ΔS = −0.00082 (→ 0.1903) — it CANNOT reach sub-0.15.**

**This refutes the "2× rate recode → sub-0.15" path *as a recode*.** The 2× target (C.3, S*≈0.137) is real
but is **not reachable by a better coder on the same INT8 symbols.** Getting bytes/param from 5.63 → ~2.8
requires one of:
- **fewer params** (smaller decoder — but C.2 shows that trades d_seg back up the capacity curve), OR
- **lower-bit weights** (INT4/INT3 + score-aware QAT — but `frontier_int5_qat_lsq_uniform_ce` already showed
  naive INT5 collapses d_seg → byte-closed S ≈ 0.47; its idealized *hold-d_seg-at-int5* projection is
  `desk_int5_hold_S = 0.153`, i.e. sub-0.15 IFF you can hold d_seg at int5 bits — which requires co-trained
  QAT, not PTQ), OR
- **learned higher-order weight entropy** (NVRC/NeuroQuant-class context models that exploit *inter-weight*
  structure the order-0 model ignores) — but the author already tried order-2 AC / DeepCABAC and found it
  marginal, so the inter-weight redundancy in this decoder appears small.

**The rate axis is near-exhausted *at fixed d_seg and fixed bits-per-weight*.** Sub-0.15 via rate REQUIRES a
joint retrain that lowers bits/param WITHOUT losing d_seg — it is a training problem, not a coding problem.

### C.5 The four lenses (geometry / fractal / calculus / physics) — tied to the numbers
1. **Geometry (codimension-1).** d_seg = (boundary-band width × boundary length) / image area — the score
   lives on the SegNet decision *boundary*, a codimension-1 set. The author's curriculum is literally a
   boundary-band-narrowing schedule (CE bulk → τ-softplus → sigmoid-bell concentrates all gradient at m=0).
   d_seg 5.6e-4 ⇒ ~0.14% of pixels flip ⇒ the residual is a thin boundary ribbon, not area error. This is
   why *capacity* helps slowly (α<1.5): adding channels sharpens the boundary band sub-linearly.
2. **Fractal.** The boundary length is capped by SegNet's **stride-2 stem at (512,384) effective
   resolution** — sub-(256,192) boundary wiggles are invisible to the scorer (CLAUDE.md SegNet blind-spot).
   So d_seg cannot be driven to 0 by texture detail below the scorer's resolution; the floor is set by the
   *coarse* boundary the scorer actually sees. This bounds how far any decoder can push d_seg and is part of
   why the cluster saturated near 5.6e-4.
3. **Calculus (the work integral).** Lowering d_seg = ∫η·(boundary flux) dt — the "work" of pushing
   flip-prone boundary pixels across margin m=0. Muon's Newton-Schulz makes the per-matrix step magnitude
   **grad-norm-independent (≈ unit/matrix)**, so the work rate is dominated by `muon_lr`, NOT the gradient
   norm. This is exactly why BUG-A's `muon_lr 2e-4` (150× too small) did **zero work** (d_seg frozen) while
   `0.03` did 7.6× — the calculus *is* the recipe bug. ∂S/∂d_seg = 100 (constant); ∂S/∂d_pose = 5/√(10·
   d_pose) = **271 at the frontier's d_pose** > 100 ⇒ at the converged operating point the pose margin is
   *more* score-sensitive per unit than seg (the RECURSIVE_REVIEW E#4 finding) — a reason the equimarginal
   seg:pose schedule is a free lever once d_pose < 2.5e-4.
4. **Physics (phase interface / free energy).** Treat the SegNet boundary as a phase interface with
   "surface tension" = the margin gradient; the τ-softplus/sigmoid losses are an interface-energy functional
   whose minimizer is the minimal-flip configuration. The RD optimum S* is the **free-energy minimum** of
   the coupled (boundary-energy + rate-entropy) system; C.2 shows that minimum sits at S≈0.185 at the
   current entropy "temperature" (bpp=0.70), and only a *colder* entropy term (lower bpp via retrain) moves
   the free-energy minimum below 0.15.

---

## PART E — Ranked sub-0.15 paths (concrete first $0/cheap test + reactivation criteria)

Ranked by P(sub-0.15) × evidence-readiness. None moves the pointer until byte-closed + `upstream/evaluate.py`.

### E.1 [HIGHEST EV] Fire the never-fired corrected full 8-stage PR95 curriculum at n600 to convergence (bc20 then bc24)
- **Why #1:** the entire "we can't beat 0.191" verdict rests on a BORROWED frontier + a recipe-throttled,
  stage-1-only "ours" run. The one measurement that has never existed: does an OWN-trained bc20/bc24 decoder,
  with the corrected `muon_lr`, run through ALL 8 stages, reach d_seg far enough below 5.6e-4 to cross
  sub-0.15 at the small-byte (rate-headroom) operating point? bc20 rate+pose floor = 0.1178; sub-0.15 needs
  d_seg < ~3.2e-4 (bc20) / ~1.5e-4 (bc24). The capacity ablation already shows bc24 at 0.00285 *at 120
  CE-only epochs* — convergence is the open variable.
- **First test ($0, days, already partly running):** the resumable daemon
  `experiments/results/torch_vehicle_full_mps_basin_bc20_n600/` — let it actually REACH stages 5–8 (it is
  paused at stage 1/ep2325). Read d_seg at the Muon-finetune stage. Monitor `decisive_fire.outer.log`.
- **Then (~$0.30):** a GPU step-time smoke (T4/A10G) to price the full curriculum.
- **Then (~$12–49, the decisive exact row):** corrected 8-stage at n600 on GPU → `build_torch_vehicle_g3_
  contest_packet.py` → CPU+CUDA `upstream/evaluate.py`.
- **Reactivation/kill:** GREEN if byte-closed S < 0.19110 (frontier shift) or < 0.15 (goal); AMBER if
  d_seg < 0.0026 but S in [0.15, 0.191]; RED (capacity wall EARNED, terminal finding re-confirmed on solid
  ground) only if the FULLY-converged 8-stage curriculum caps S ≥ frontier.

### E.2 [HIGH EV, the rate axis done RIGHT] Score-aware INT4/INT3 co-trained QAT (NOT PTQ, NOT a recode)
- **Why:** C.4 proves the rate axis needs *lower bits/param at held d_seg*, and C.3 shows 2× bits/param →
  S*≈0.137. A *recode* is dead (entropy floor); *co-trained* low-bit QAT is the live path. `desk_int5_hold_S
  = 0.153` says even INT5-held-d_seg is right at the line; INT4 co-trained (with the C1a integer-grid
  regularizer the author used, pushed harder) is the real shot.
- **First test ($0):** extend the existing `frontier_int5_qat` harness to INT4 with score-aware QAT +
  C1a-σ-sweep, warm-started from a converged E.1 basin (NOT from scratch); measure byte-closed advisory S.
  Falsifiable threshold: INT4 co-trained holds d_seg within 1.3× of the int8 basin ⇒ S → ~0.14.
- **Reactivation/kill:** kill INT4 only if co-trained QAT (not PTQ) still collapses d_seg > 2× at INT4.

### E.3 [MEDIUM EV, byte-neutral, stacks with E.1] The d_seg-aware boundary-band taper
- **Why:** the vendored taper puts ~69% of params in low-res stages SegNet's stride-2 stem discards, and
  only ~7.76K at the high-res boundary band where flips live (RECURSIVE_REVIEW #3). Reallocating to the
  boundary band is **byte-neutral (+0.05%)** and could bend d_seg materially — a free RD win that does not
  touch the rate term. Geometry/fractal lenses (C.5 #1/#2) predict the boundary band is exactly where d_seg
  residual lives.
- **First test ($0):** the `launch_bind_all_taper_ab.py --arm arm_b` A/B, KD-warm-started from the E.1
  basin, short n600 budget; measure the d_seg LATE-exponent vs the control taper.
- **Reactivation/kill:** keep if arm_b bends the d_seg exponent at equal bytes; drop if neutral.

### E.4 [MEDIUM EV, free correctness] The PR95 recipe non-optimalities the author left in
- margin-hinge seg surrogate (anneal 1.0→0.5; measured 0.643× CE residual d_seg); recover the ~70% wasted
  basin epochs via stage-transition LR restarts; equimarginal seg:pose once d_pose<2.5e-4 (∂S/∂d_pose=271);
  pose-√-outside-the-batch Jensen fix. Each is a small, free correctness lever that stacks on E.1; none is
  sub-0.15 alone but the STACK is the honest shot (RECURSIVE_REVIEW §3: modal outcome ~60% earned-0.19 wall,
  ~3% sub-0.15 — the stack is how you learn which).
- **First test ($0):** fold as additional arms in the E.3 A/B.

### E.5 [LOWER EV, paradigm] Task-space / scorer-invariant representation (do not reconstruct RGB the scorer ignores)
- **Why:** the contest is indirect rate-distortion on f(X)=(SegNet-argmax, PoseNet-6dim); all HNeRV vehicles
  pay to reconstruct RGB the scorer discards (MEMORY: "contest is indirect rate-distortion / task-space
  coding"). A representation coding only the task features could break the capacity-RD frontier entirely.
- **First test ($0):** the P-SUFF scorer-invariance gap probe (the decisive go/no-go in the task-space memo).
- **Reactivation:** only if E.1–E.4 confirm the EARNED ~0.19 wall on an own-trained decoder.

---

## 6-hook wire-in
1. **Sensitivity-map:** the per-tensor entropy table (`frontier_decoder_weight_entropy_20260623.json`) is a
   per-tensor *rate-sensitivity* map (which tensors hold the bytes: stem+blocks.0-3 = ~136K of 161K) — feeds
   the bit-allocator (hook #3). ACTIVE.
2. **Pareto constraint:** the capacity-RD optimum S(p) (C.2) + the bytes/param=0.704 measurement add the
   binding rate↔capacity Pareto edge (capacity scaling forfeits rate). ACTIVE (delta form via
   `tac.score_composition`).
3. **Bit-allocator hook:** the order-0 H0 floor (5.586 b/param) is the per-tensor allocator's hard lower
   bound — the allocator should STOP optimizing the order-0 coder (1226 B left) and redirect to INT4-QAT
   (E.2). ACTIVE.
4. **Cathedral autopilot dispatch:** E.1 is the named exact-row-feeding dispatch (the de-risked never-fired
   run). ACTIVE (op-routable).
5. **Continual-learning posterior:** the α∈[0.9,1.5] capacity exponent + bpp=0.704 + entropy-floor-ratio
   1.0077 are new empirical anchors for the capacity/rate models. ACTIVE (this memo + JSON are the anchors).
6. **Probe-disambiguator:** the entropy measurement IS the disambiguator between "rate axis has 2× headroom
   (recode)" (REFUTED) vs "rate axis needs a retrain" (CONFIRMED). ACTIVE.

## Observability surface
Every number cites a file:field — the frontier archive (sha verified `b46897267…`), its `byte_closure_proof.
json` (dec_sec_bytes 161104), `report.txt` (d_seg/d_pose), the capacity ablation `capstone_result.json`s,
the BUG-A A/B in the recipe-gap memo, and the new `experiments/results/frontier_decoder_weight_entropy_
20260623.json` (reproducible via `experiments/measure_frontier_decoder_weight_entropy.py`). Axis
`[contest-CPU advisory] / [analysis]`, score_claim=false, pointer UNMOVED 0.19109982.

## Canonical-vs-unique decision per layer
The measurement REUSES the frontier's own `inflate.py`/`codec.py` (ADOPT — the only byte-faithful way to read
the real weights) and `tac.contest_score` for ALL score math (ADOPT — the compliance bedrock). The new tool
is a thin read-only measurement over those canonical surfaces (no fork).

## NO-FAKE ledger
- **MEASURED:** frontier decoder entropy 5.629 achieved vs 5.586 H0-ideal bits/param (ratio 1.0077, the
  decisive number); capacity ablation bc20 0.00376 vs bc24 0.00285 at matched recipe; BUG-A A/B (0.507
  frozen vs 0.066, from the recipe-gap memo); the basin n600 stage-1 d_seg 0.0026; the byte budget
  (dec_sec 161104); all S values via tac.contest_score.
- **DERIVED (clearly labeled):** α∈[0.9,1.5] (2-point clean fit, under-converged → lower bound); the S(p)
  capacity-RD optima (model fit, k1 calibrated to the real frontier point); the 2×/3× RD-shift S* (the lever
  exists IFF bits/param can be halved at held d_seg — which C.4 shows needs a retrain, not a recode).
- **AUTHOR QUOTES:** verbatim from the r.jina.ai-rendered blog + PR #95 description; corroborated against
  the pristine local source. The "no theoretical-floor / no future-work section" is an honest negative, not
  an omission on my part.
- **NOT claimed:** no score moved; pointer UNMOVED 0.19110; no promotion; no exact row produced. This is a
  MEANS memo.
