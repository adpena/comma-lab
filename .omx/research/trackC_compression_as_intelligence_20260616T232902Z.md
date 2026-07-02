# Track C+ — Compression-as-Intelligence / Parameter-Free Witness (research/design, 2026-06-16)

**Authority:** DESIGN + research_only=true. Zero score claims. Every economic fact below is tagged
`[MEASURED]` (this repo's exact-`modules.py` artifacts) or `[DERIVED]` (closed-form from the contest
formula) or `[ESTIMATE]` (proposed probe target, NOT a result). No GPU, no paid dispatch, no MPS,
no edits to drivers/launchers/running jobs. The oracle is `upstream/evaluate.py` + `modules.py` +
`0.mkv`; pointer moves ONLY on Linux-x86_64 + CUDA-T4 600-sample exact eval.

**Frontier (pointer):** S = 0.19109982 [contest-CPU], 177,169 B, lane_pr110_payload_entropy_recode.
`S = 100·d_seg + sqrt(10·d_pose) + 25·bytes/37_545_489`. THE GOAL is sub-0.15 (T_3); T_1 = sub-0.19.

---

## 0. The mission framing (what Track C actually IS)

The contest is, exactly, a **Kolmogorov / MDL problem**: find the shortest compliant `archive.zip`
whose `inflate.sh` output is a *witness* inside the same frozen evaluator cells as the source video.
The "compression-as-intelligence" thesis (Cilibrasi-Vitányi NCD; the gzip-LM post; the gzip-kNN
paper) is the claim that **the best compressor of {masks, pose} IS the best model of {masks, pose}** —
so a strong general-purpose / context-mixing compressor on the RIGHT representation could pay ZERO
decoder-weight bytes and still produce an in-tolerance witness, beating the ~74-186 KB HNeRV decoder
that has to *store its model*.

This memo's job is to settle, with the repo's already-measured numbers, **whether that thesis can
cross the contest water level** — and to rank the live Track-C designs by exact-row potential.

---

## 1. Reference synthesis — what the three operator references contribute

### 1a. nathan.rs "gzip as a language model" (gzip-LM)
Mechanism: every compressor is an implicit probability model (`P(x) ≈ 2^-C(x)`); score a continuation
by `len(gzip(context + candidate))`; generate by beam search over next bytes, keeping only a short
`tail` of context visible to avoid verbatim loops. **What it contributes to us:** the *predict-next-
frame's-mask-from-history* primitive. The 600-frame argmax-mask sequence is a byte stream with
massive temporal redundancy (dashcam ego-motion is smooth); a gzip-LM-style coder predicts frame
t's partition from frames t-1..t-k as cheap back-references. **Critical caveat it self-reports:**
gzip's 32 KiB sliding window is far smaller than one 384×512 argmax frame (196,608 bytes) — so naive
gzip *cannot even see one prior frame*. This is exactly why a window-unbounded coder (lzma dict ≥ a
few frames, brotli large-window, or a true context model) is mandatory for the temporal play.

### 1b. HN thread (id 36732430) — the substantive critiques + extensions
- **Window limit is the dominant flaw** (vintermann): a coder whose context window < the object it
  predicts is structurally blind. → for us: dictionary/window MUST span ≥2-3 mask frames.
- **Alphabet mismatch** (gzip can't approach optimal code if alphabets differ): our mask alphabet is
  tiny (5 classes) → favorable, but it means the win is in the *boundary geometry*, not the symbol
  histogram (interior runs are near-free already).
- **Better compressors named:** bzip2 (BWT), lzma, **Bellard's nncp** (transformer-based lossless —
  the SOTA neural compressor), PPM, fixed-dictionary training over the corpus, digram alphabets.
- **State reuse** (save compressor state before the final byte): turns the O(N²) re-compress into
  O(N) — directly relevant if we ever do per-frame conditional coding at scale.
- **Theory grounding:** Solomonoff Universal Prior; Hinton-Van Camp "description length of the
  weights" (MDL = the contest's exact objective); the gzip-kNN ACL 2023 paper (aclanthology
  2023.findings-acl.426) and arXiv 2212.09410.

### 1c. Google Scholar profile = **Rudi Cilibrasi** (Paul Vitányi's collaborator)
This is the NCD founder. Top works: "The Google Similarity Distance" (2007, 2402 cites), "Clustering
by Compression" (2005, 1758 cites), "Normalized Information Distance" (2009). **What he contributes:**
the rigorous theory that `NCD(x,y) = (C(xy) − min(C(x),C(y))) / max(C(x),C(y))` approximates the
normalized information distance (a universal metric up to the additive Kolmogorov constant). The
operationally important corollary for us is **conditional compression**: `C(y|x) ≈ C(xy) − C(x)` —
the bytes to describe y GIVEN x are already in the codec. This is the formal engine for *conditional
coding of the mask/pose against a cheap, free-to-both-sides base* — the only way a parameter-free
carrier gets near the water level (store the residual against the base, not the object).

### 1d. The critique that bounds our expectations (Opitz 2307.15002 + efficiency analysis)
gzip-kNN's "beats BERT" was partly optimistic (test-train leakage in the top-2 tie-break) and **bag-
of-words / TF-IDF can match or beat it while being 100× faster and 2500× smaller in memory**. The
honest lesson: **compression-as-classifier is competitive when (a) data is scarce, (b) the structure
is genuinely sequential/redundant, and (c) you have NO trained model to amortize.** Our contest is a
*single video* (data-scarce by NCD standards — exactly NCD's home turf) BUT the incumbent already has
a *trained decoder that amortizes per-flip cost to zero*. That is the crux of §2.

---

## 2. The operating-point crux — why naive parameter-free LOSES, and the one door left open

This is the central, MEASURED tension. Two facts collide:

**Fact A (parameter-free is exact + cheap-per-object-to-describe but fat-to-store):** [MEASURED, #52
boundary-math seg-core]. Storing the SegNet argmax partition `L*` directly gives `d_seg = 0.0`
bit-exact (it IS the argmax), `d_pose = 0.0` (frame1 untouched). Cost under the LZMA-over-labels
baseline: **895.7 B/frame → 524.8 KB for 600 frames** = 2.96× the entire 173 KB frontier archive.
Direct-store partition for the witness = **614,430 B [MEASURED, witness probe D_direct_store]** — it
LOSES vs frontier by 3.5×. The rate term alone for a 525 KB seg carrier = 25·524800/N = **0.349**,
already 1.8× the whole current score.

**Fact B (the incumbent decoder pays ZERO bytes per flip):** [MEASURED, rate-coupling memo §"d_seg's
cheapest mechanism"]. A trained decoder amortizes a weight change across all pixels at zero per-flip
byte cost. The Class-3 flip sidecar failed at **1.525 B/flip > break-even (1.273 B/flip water level)**
precisely because a sidecar pays per-flip while a decoder pays nothing. This is *why* HNeRV wins the
seg axis: the 74-KB decoder is a lossy-but-free generator of the partition.

**The water level (the bar every Track-C byte must clear):** [DERIVED] λ* = 1.2731 B/flip =
`(100/(600·384·512)) / (25/37_545_489)`. Fixing one flip pays rent iff it costs < 1.273 bytes. The
rate price is `∂S/∂B = 25/N = 6.66e-7` score/byte.

**The one door left open — CONDITIONAL coding against a free base (Cilibrasi `C(y|x)`):** [MEASURED,
witness boundary probe n=120, the single most important Track-C number in the repo]:
- Unconditional mask-flip coding: **1.376 B/flip** (LOSES, > 1.273).
- **Conditional-on-boundary coding: 1.0227 B/flip mean, 1.0217 median, 100% of pairs under the water
  level.** `B_cost_under_break_even = true`.
- Net ΔS of fixing ALL boundary flips via this sidecar: **−0.0880 [MEASURED]** (`B_boundary_sidecar_
  net_negative_S = true`) — i.e. the conditional residual sidecar is *score-negative* (good).

That −0.088 is the entire Track-C thesis in one number: **conditional compression of the seg residual,
coded to the detector's tolerance, clears the water level where unconditional/direct storage does
not.** The verdict on the prior probe was `HYBRID_FOLD_INTO_TRAINING` (D_witness_mdl_below_frontier =
false for a *standalone* full carrier), but the *sidecar/residual* arithmetic is genuinely sub-water.
The door is: **NOT a parameter-free FULL witness, but a parameter-free RESIDUAL coded against a
cheap/free base** — the hybrid. This is also the #96 Yousfi lesson restated: non-neural was lossless
at the wrong operating point; the right operating point is *code the residual to tolerance*, not the
object losslessly.

---

## 3. Entropy budget — what a parameter-free archive would actually cost

Putting the MEASURED pieces into one archive-byte ledger (all [MEASURED] unless tagged):

| Component | Mechanism | Bytes (600 frames) | Source |
|---|---|---:|---|
| Pose trajectory (6 scalars/pair, range-coded) | direct store | **1,557** | witness probe `D_pose_traj_bytes` |
| Seg partition, direct LZMA-over-labels | direct store | 524,800 | #52 seg-core (895.7 B/fr) |
| Seg partition, conditional-on-boundary | `C(L*|boundary)` | ~543,000 | witness probe `B_total_residual_bytes_600` |
| Seg residual-vs-base, conditional sidecar | code only the flips a free base gets wrong | **24,600–64,600** | witness probe `D_conditional_mdl_band` |
| Frontier total (seg+pose+decoder) | incumbent HNeRV | 177,169 | pointer |

**The decisive comparison:** a *standalone* parameter-free seg carrier (525 KB) loses by 3×. But the
**scorer-conditional MDL band of the seg residual is 24.6–64.6 KB [MEASURED]** — *below* the 177 KB
frontier. Add pose 1.5 KB. If a cheap base (see §4) regenerates most of the partition for free, the
parameter-free RESIDUAL archive is **~26–66 KB + base-cost**, which is where sub-frontier (and the
path to sub-0.15) lives. The whole game is: **how cheap is the base, and does the residual stay in
the 25-65 KB band after the base eats most of the flips?**

The information-theoretic floor context [DERIVED, rate-coupling memo]: rate floor of the *current
representation* = 0.118; "if rate→0" distortion floor = d_seg+d_pose = 0.073; T_floor is the JOINT
R–D frontier minimum, MEASURED proxy S_floor ≈ 0.1178 rate-dominated. A 26-66 KB residual archive
has rate term 25·(26000..66000)/N = **0.0173..0.0439** — leaving a *huge* rate budget vs the
frontier's 0.118, which is exactly the zero-decoder-weights dividend the thesis predicts.

---

## 4. The candidate parameter-free witness designs (ranked by exact-row potential)

Each design: mechanism · byte/ΔS estimate · the $0 probe that validates it · the base it conditions
against. Ranked by `|predicted ΔS| / probe-cost`, steepest first.

### RANK 1 — Conditional seg-residual coder against the FREE-RENDER base (the hybrid that clears water)
**Mechanism:** take a cheap, *zero-byte-or-near-zero* base prediction of frame1's argmax (candidates:
the previous frame's mask warped by the stored pose — pose is only 1.5 KB; OR the contest's own
trivial constant/most-common-class; OR a tiny shared LUT). Code ONLY the flips the base gets wrong,
using the Cilibrasi conditional coder at the MEASURED 1.0227 B/flip (boundary-conditional, sub-water).
Pose stored direct (1,557 B). This is the parameter-free realization of the full-stack-carrier-V2
"amortized regeneration + stored residual," but with the regenerator being a *compression model*, not
a trained net — so the base costs ~0 bytes.
**Byte/ΔS [ESTIMATE, grounded]:** if a warp-base eats ~60-80% of the 884 flips/pair, residual ≈
180-350 flips/pair × 1.02 B = ~110-360 KB unconditional... BUT the conditional-MDL band already
MEASURED the *whole* residual at 24.6-64.6 KB. Target archive ≈ 26-66 KB. Predicted S if d_seg driven
to the corrected bar (~0.0011) at ~45 KB rate: rate 0.030 + d_seg 0.11 + pose ~0.017 ≈ **0.157**
[ESTIMATE] — knocking on sub-0.15 with a *zero-decoder* archive. The d_seg term dominates; the win
size is entirely how low the base+residual drives d_seg.
**$0 probe (HIGHEST EV — see §6):** measure the warp-base residual flip count: for each of 120 pairs,
warp frame(t-1) argmax by the GT pose, compare to frame(t) GT argmax, count residual flips, and code
them conditionally; report residual-flips/pair and residual-bytes/600 vs the 24.6-64.6 KB band and
vs the 177 KB frontier. Reuses the existing witness-probe harness + boundary conditional coder.

### RANK 2 — Temporal context-mixing coder on the 600-frame mask SEQUENCE (gzip-LM / nncp angle)
**Mechanism:** treat the 600 argmax frames as ONE temporal byte stream; code with a window/dict that
spans ≥2-3 frames (lzma with large `dict_size`, brotli large-window, bsc/BWT, PPMd, or cmix) so
frame t is predicted from t-1..t-k as back-references (the gzip-LM next-frame-from-history primitive,
but window-unbounded per the HN window-limit critique). The argmax sequence is near-static between
frames (smooth ego-motion) → the inter-frame delta is sparse.
**Byte/ΔS [ESTIMATE]:** the per-frame entropy (896 B LZMA, no temporal context) is an UPPER bound;
temporal context should cut it substantially (the inter-frame XOR is far sparser than the intra-frame
boundary). If temporal coding hits even 300 B/frame → 180 KB (≈ frontier, but d_seg=0 exactly). If it
hits the 170-250 B/frame target from #52 → 102-150 KB at d_seg=0. This is the *lossless* play; its
score is rate-only: 25·150000/N = 0.10 + pose 0.017 + d_seg 0 = **~0.117** [ESTIMATE] IF it reaches
150 KB lossless. Risk: lossless may not reach the band (it's the "wrong game" per Yousfi — but a
lossless d_seg=0 carrier at ≤150 KB would still beat 0.191).
**$0 probe:** measure the 600-frame argmax sequence under {lzma dict=64MB, brotli q11 lgwin=24,
PPMd via 7z, bsc, zstd --ultra -22 --long=27} as a single temporal stream AND as per-frame-delta
streams; report B/frame for each. This is the literal operator probe ("entropy of the 600-frame
argmax-mask sequence under cmix vs brotli"). Pure stdlib+CLI, no GPU.

### RANK 3 — NCD-based pose & mask prediction (Cilibrasi conditional-complexity predictor)
**Mechanism:** use `C(y|x) = C(xy) − C(x)` directly as the predictor: for pose, predict pair t's 6-dim
pose by finding the historical pair minimizing NCD to the current context and extrapolating (NCD-kNN
in pose-trajectory space); for masks, the NCD between consecutive frames quantifies the conditional
description length the coder will pay. This is the "compression-as-intelligence" predictor proper.
**Byte/ΔS [ESTIMATE]:** pose is already cheap (1,557 B direct), so NCD-pose's win is marginal on
bytes; its value is as the *entropy estimator* that tells RANK 1/2 how compressible the streams are
and as a *base predictor* (a good NCD predictor of next-frame mask = a cheaper RANK-1 base). Predicted
direct ΔS small; predicted *enabling* value high (it sharpens RANK 1's base).
**$0 probe:** compute NCD between consecutive argmax frames and between pose vectors across 120 pairs;
report the NCD distribution (low NCD ⇒ strong temporal predictability ⇒ RANK 1/2 will win).

### RANK 4 — Hybrid: TINY learned base (sub-byte) + compression-coded residual (the score-optimal blend)
**Mechanism:** the honest synthesis of the Opitz critique (pure parameter-free loses to a tiny
amortizer) + the conditional-coding win. A *minimal* learned base (a few KB block-FP / shared LUT /
a 1-2 KB FiLM-conditioned tiny synth, NOT a 74 KB HNeRV) eats the bulk of the flips for near-free
amortized cost; the compression coder handles the residual at 1.02 B/flip. This is RANK 1 with a paid
(but tiny) base instead of a zero-byte warp base. It is the bridge from "parameter-free" to "param-
minimal," and is the design most likely to actually clear sub-0.15 because the base can drive d_seg
*below* what a warp can.
**Byte/ΔS [ESTIMATE]:** base 2-8 KB + residual 25-50 KB + pose 1.5 KB ≈ 30-60 KB; if d_seg → 0.0011
bar: rate ~0.025 + d_seg 0.11 + pose 0.017 ≈ **0.152** [ESTIMATE], with clear headroom below if the
base pushes d_seg lower. NOTE: this crosses into "param-minimal," not strictly parameter-free — flag
honestly. It is the highest-ceiling design but needs (cheap) training, so it is a follow-on to the
$0 probes, not a $0 probe itself.
**$0 probe (proxy):** simulate the base as "GT-frame(t-1) argmax constant-held" (zero-param stand-in
for a tiny learned base) and measure residual; this brackets RANK 1 (warp) and RANK 4 (learned) and
tells us how much a learned base must beat a warp to be worth its bytes.

### RANK 5 (dream / DEFER) — Direct Kolmogorov/MDL program search for the witness
**Mechanism:** the contest IS "shortest program for the witness." A literal MDL/program-synthesis
search (or the evaluator null-space compiler #47 — code perturbations in the 80.67% resize-null
subspace and 22.7% zero-weight pixels that are CERTIFIED-invisible to both heads, residual==0.0) lets
us shed bytes with *provably zero distortion*. Combine with conditional residual coding.
**Byte/ΔS:** the null-space gives free bytes (any data in the invisible subspace costs nothing in
distortion), so it's a *rate-only* enabler that composes with RANK 1-4. [DERIVED] 22.7% of pixels are
amplitude-unlimited invisible. Direct program search is intractable to claim a number; DEFER the
search, but the null-space basis is a ready composer for RANK 1.
**$0 probe:** none new — #47 is LANDED + wired to the waterfiller; cite it as the rate-composer.

---

## 5. Operating-point re-attack vs #96 (the Yousfi lesson, resolved)

#96's non-neural carrier was "lossless but wrong operating point." This memo localizes WHY and HOW to
fix it:
- **Wrong:** code the *object* (partition/pose) losslessly → 525 KB, rate 0.35, loses (RANK 2's risk).
- **Right (the re-attack):** code the *residual against a free base* to the *detector's tolerance*
  (drop everything below 1.273 B/flip; ignore flips outside the boundary band; use the 1.0227 B/flip
  conditional coder) → 25-65 KB band, rate 0.017-0.044, score-negative residual (−0.088 MEASURED).
- The operating point is set by the **margin-polytope** (only ~1.4% of pixels are margin<0.5;
  boundary is a sparse 1D contour) and the **water level** (1.273 B/flip), not by lossless fidelity.
- This is inverse-steganalysis (Yousfi/Fridrich): code to the SegNet decision boundary's tolerance,
  exactly as UNIWARD weights cost by inverse local variance. Track-C's coder must be margin-aware.

---

## 6. Prioritized $0 next-probes (the actionable handoff)

All reuse existing harnesses (witness boundary probe, boundary_math seg-core, scorer_conditional_mdl
estimator, variable_level_codec); all are CPU-torch advisory / stdlib-CLI; NONE need GPU or paid
dispatch. Ranked by EV (steepest d_seg/probe-effort):

1. **[HIGHEST EV] Warp-base residual probe (validates RANK 1).** For 120 pairs: warp frame(t-1) GT
   argmax by GT pose → base prediction of frame(t) argmax; count residual flips; code conditionally
   at the existing 1.02 B/flip coder; report residual-bytes/600 vs the 24.6-64.6 KB MEASURED band and
   vs 177 KB frontier. **Falsification threshold:** if warp-base residual ≥ 90 KB, RANK 1's zero-byte
   base is insufficient → escalate to RANK 4 (tiny learned base). If ≤ 65 KB, RANK 1 is a live
   sub-frontier parameter-free carrier → build the byte-closed archive next.
2. **Temporal-sequence entropy probe (validates RANK 2 + the operator's literal ask).** Code the
   600-frame argmax sequence under {lzma dict=64MB, brotli q11 lgwin=24, 7z-PPMd, zstd --ultra-22
   --long=27, bsc} as (a) one temporal stream, (b) per-frame-delta stream. Report B/frame each.
   **Falsification:** if best temporal coder > 250 B/frame, the lossless temporal play can't reach
   the #52 target → RANK 2 is rate-bound, fold into RANK 1's base only.
3. **NCD predictability probe (validates RANK 3, sharpens RANK 1 base).** NCD between consecutive
   argmax frames + between pose vectors, 120 pairs. Report distribution. Low NCD ⇒ a learned/warp base
   will predict well ⇒ raises RANK 1/4 ceiling.
4. **Constant-held base bracket (cheap proxy for RANK 4).** Residual of "frame(t-1) constant-held"
   base — brackets warp (RANK 1) vs learned (RANK 4), quantifies the learned base's required margin.

---

## 7. research_only discipline + wire-in declaration (Catalog #125)

`research_only = true`. NO score claims; §3-4 byte/ΔS figures are [MEASURED] inputs or explicitly
[ESTIMATE] design targets, NOT results. The 6 unified-Lagrangian hooks:
1. **Sensitivity-map:** ACTIVE — the conditional-residual marginal (1.0227 B/flip, boundary-band) is a
   per-pixel seg sensitivity contribution; feeds the waterfilling allocator's seg constraint.
2. **Pareto constraint:** ACTIVE — RANK 1's 25-65 KB residual band is a new low-rate vertex on the
   R–D frontier (zero-decoder-weights regime), to be added to `tac.pareto_*`.
3. **Bit-allocator hook:** ACTIVE — the per-region/per-boundary marginals from #52 + the conditional
   coder rate are the allocator inputs at λ*=1.273.
4. **Cathedral autopilot dispatch:** N/A (research_only; no archive-deployable artifact this pass).
5. **Continual-learning posterior:** trigger on EACH $0 probe result in §6 (residual-bytes, temporal
   B/frame, NCD distribution become anchors).
6. **Probe-disambiguator:** ACTIVE — §6 probes 1+4 are the warp-vs-learned-base disambiguator
   (2 defensible interpretations: zero-byte base suffices vs needs tiny learned base).

**Cross-refs:** #52 boundary-math seg-core (the d_seg=0 partition + 1.273 water level) · #95/#96
Yousfi (partition 3-5× neural; residual-to-tolerance is the right op-point) · #47 evaluator null-space
(rate-composer, 22.7%/80.67% invisible) · witness_seg_boundary_probe_live_n120.json (the 1.0227
B/flip + −0.088 + 24.6-64.6 KB band — THE Track-C anchor) · rate_distortion_coupling (d_seg is the
linear dominant pool; rate is lossless-floored) · full_stack_carrier_v2 (the amortized-regen + stored-
residual hybrid this is the parameter-free instance of) · Cilibrasi NCD `C(y|x)=C(xy)−C(x)`.
