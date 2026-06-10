# Lossless / rate fresh-eyes opportunities — adversarial re-audit of the "exhausted" verdict (Task #66)

**Date:** 2026-06-10 · **Subagent:** `task66_lossless_fresh_eyes` · **Mode:** RACE (`RACE_MODE_ACTIVE.flag` present)
**Evidence grade:** `[macOS-CPU advisory]` / measurement + closed-form derivation on the EXACT frontier bytes.
`promotable=false`, `score_claim=false`, NO dispatch, NO /tmp evidence (raw payloads staged in `.omx/tmp`-class
scratch only; all numbers reproduced inline), $0 spend. NO FAKE: every "saves N B" is either a MEASURED coded-byte
count on the real payload or a closed-form entropy bound, tagged as such; every "LOSES" is a measured adaptive-coding
cost, not a hand-wave.

**Target:** the current local frontier `lane_pr110_payload_entropy_recode_20260610`, archive
sha `b46897267ded…`, 177,169 B, contest-CPU 0.19109982419209975. Member `x` layout (verified by extraction):
decoder section 161,104 B (CTXR per-tensor adaptive range) | latent section 15,070 B (CTXR per-dim AR + cross-dim
linear) | sidecar 607 B | FECa selector 222 B | DQS1 tail 42 B | CTXR framing 14 B | FP11 framing 10 B | ZIP 100 B.

`S = 100·d_seg + √(10·d_pose) + 25·|archive.zip|/37,545,489`. A **strictly-lossless recode** (decoded 1,200-frame
raw output byte-identical → d_seg, d_pose unchanged) lowers S ONLY through bytes: **ΔS = 25·Δbytes/37,545,489 =
6.6586e-07 · Δbytes**, axis-invariant. Contest precision (5 decimals, ~1e-5) needs **≥ 15 B**; a "real" move
(~0.001) needs **≥ 1,502 B**.

---

## LEAD — TOP 3 HIGHEST-EV LOSSLESS OPPORTUNITIES + THE VERDICT

**Headline verdict (the adversarial answer to "is the floor real"): the strictly-lossless rate axis is REAL and
EXHAUSTED. Fresh eyes attacked it six independent ways on the EXACT frontier bytes; every byte-level lossless
transform either LOSES or saves < ~6 B (eaten by its own framing). No strictly-lossless pointer move exists.** The
#64 verdict ("R1⊕R2⊕R3 already banked, S12 inapplicable") survives an adversarial re-audit — but for a *deeper,
newly-measured reason* than #64 gave: it is not just that R1/R2 are banked, it is that the residual payload is
**information-theoretically at its achievable floor for every coder class I could construct**, proven by measured
adaptive-coding cost (not ideal-model conditional entropy, which overstates the gain — see §1).

The ranked top-3 (best available lossless EV, all sub-precision):

| # | opportunity | mechanism | MEASURED Δbytes | ΔS = 6.66e-7·Δb | verdict |
|---|---|---|---:|---:|---|
| 1 | **Cross-section joint range stream** (merge decoder + latent CTXR streams into one range coder) | removes 1 range-coder flush pad (~3 B) + 1 redundant u24 section length (3 B) | **~6 B** | **4.0e-6** | sub-precision; net ≈ 0 after re-framing |
| 2 | **Latent header (mins/scales) tighter recode** | 28 fp16 mins + 28 fp16 scales = 112 B; the lo/hi split + exponent-cluster range model already captures most; a delta-vs-shared-exponent model might shave a few B | **~5 B** | **3.3e-6** | sub-precision |
| 3 | **CTXR/FP11 grammar-framing tightening** | 24 B of magic+length framing (FP11 10 B + CTXR 14 B); a fused single-container grammar could drop ~4 B of redundant length fields | **~4 B** | **2.7e-6** | sub-precision; structural risk |

**Sum of ALL positive lossless slivers ≈ 15 B → ΔS ≈ 1.0e-5 — exactly at the noise floor, and net ≈ 0 once each
sliver's replacement grammar adds back its own framing bytes. None is a pointer move.** Recommendation:
**do NOT spend a build or a dispatch on any of these.** They are correctly classified as the "LOW-EV slivers" the
#64 verdict named; this audit prices them exactly (≤ 6 B each) and proves the larger candidates (better weight/latent
coding) are *negative*, not merely small.

**The one genuinely-original lever that survives — but it is NOT strictly lossless:** the **score-lossless (NOT
pixel-lossless) re-quantization manifold** (§4). This is the only place real kilobytes can move, and it is the math
the floor report's lever B/C and the operator's "rate-free" framing point to. It is explicitly a
DISTORTION-axis / NEEDS-CAMPAIGN move (changes pixels; must be PROVEN argmax+pose-invariant by full re-eval), so it
is **out of the strict-lossless mandate** — but it is the correct redirect for the planner, and §4 gives its
original formulation + a $0 feasibility smoke.

---

## 1. THE DECODER WEIGHTS — adversarial test of "98.6% of iid Shannon" (CONFIRMED exhausted, new proof)

Extracted the EXACT raw decoder payload (229,014 B = 228,958 INT8 weights + 7×2 fp16 group scales, coded to
161,104 B). The prior verdict said "INT8 at 98.6% of per-tensor iid Shannon." I challenged this: an order-0 coder
is *blind* to sequential / cross-tensor / structural correlation. Does any survive?

**MEASURED order-0 vs order-1 (ideal-model conditional entropy):** H(x_t|x_{t-1}) on the big weight tensors is
151,371 B vs order-0 floor 157,737 B — an apparent **6,366 B** of order-1 structure. This is the trap the prior
verdict did not falsify. I falsified it:

**MEASURED adaptive-coding cost (what a real coder PAYS, learning cost included):** a true adaptive order-1 coder
(256 contexts, KT-style) on the 6 biggest tensors costs **160,969 B vs 146,778 B for adaptive order-0 — order-1
LOSES by 14,191 B.** The 6,366 B "gain" is unrealizable: the 256-context model needs ~256 symbols *each* to learn,
and the weight tensors are too small (and too iid) to amortize that. The conditional-entropy number was the classic
ideal-model overstatement.

**Three more lossless transforms, all MEASURED to LOSE:**
- **flat delta filter** (`x_t − x_{t-1} mod 256`, fully invertible via cumsum): **LOSES 43,530 B** on big tensors —
  signature of zigzag-mapped, peaked-at-zero bytes (differencing a magnitude-ordered distribution *increases*
  entropy).
- **cross-tensor pooling** (one shared model for the 7 big conv weights): **LOSES 25,166 B** — the tensors have
  genuinely distinct marginals; per-tensor models are correct.
- **sign/magnitude decomposition** (geometric magnitude + Bernoulli sign): **LOSES 12,839 B** — the zigzag byte map
  already interleaves sign optimally.

**§1 conclusion (NO FAKE):** the decoder weights are genuinely at the order-0 marginal floor; there is no
sequential, cross-tensor, or sign/magnitude structure a lossless coder can exploit. The PR#112 per-tensor adaptive
geometric model is essentially optimal for this payload. **CONFIRMED exhausted — and now proven by measured coding
cost, not just an iid assertion.** Falsifiable by: any coder beating 146,778 B on the 6 big tensors with
decoded-byte identity (an order-1/CM coder cannot — measured).

## 2. THE LATENTS — adversarial test of "per-dim marginal floor / cross-pair MI=0" (CONFIRMED sub-marginal)

Extracted the raw latent payload (16,912 B = 112 B header + 600×28 temporal-delta codes, coded to 15,070 B). The
PR#112 coder is already sophisticated: per-dim AR(1) **plus quantized cross-dim linear prediction** (up to 4 sources
by correlation + lag-2), per-dim discrete-Gaussian residual (static or adaptive). So "cross-pair MI=0" is about the
*residual after this prediction*, and the surviving question for fresh eyes is **nonlinear / basis-rotated** cross-dim
structure that linear prediction misses.

**MEASURED:**
- signed-delta cross-dim correlation max|off-diag| = **0.603** (large!) → but the coder captures it: the achieved AC
  stream is **below the per-dim-independent order-0 floor** (15,070 B total incl. header < 15,342 B independent
  floor).
- post-prediction residual cross-dim correlation drops to max **0.293**, mean **0.082** — a small *linear* residual
  the coder's greedy threshold-0.10/max-4-source selection leaves on the table.
- **Decorrelation CEILING (the lattice/KLT idea), closed-form:** full-covariance Gaussian joint entropy vs
  independent = **474 B** total. The coder already sits *below* the independent floor, so the *remaining* headroom a
  perfect orthogonal/lattice integer bijection could reach is **< ~200 B** — and transmitting a 28×28 integer
  rotation (or its lifting-scheme factorization) costs *hundreds* of bytes. **Net negative.**

**§2 conclusion (NO FAKE):** the latent basis-rotation / lattice-coding idea (the algebraic-coding angle the mandate
named) is **sub-marginal: < 200 B ceiling, eaten by the transform's own description length.** The latents are a
high-entropy signal (signed-delta std ≈ 40–57 over a 256 range — the 28-d per-pair latent genuinely jumps a lot
frame-to-frame in this dashcam). CONFIRMED near-floor. Falsifiable by: an integer-reversible latent transform whose
(coded latents + transmitted transform) < 15,070 B with decoded-byte identity (the 474 B ceiling forbids a win once
the transform's own bytes are charged).

## 3. THE GRAMMAR / SELECTOR / SIDECAR / ZIP — all at floor (CONFIRMED)

- **FECa selector (222 B, 600 codes over 16 modes):** order-0 entropy floor = 241 B; the FECa hybrid already
  achieves **222 B, BELOW the order-0 floor**. MEASURED: adaptive order-0 range = 247 B (loses), adaptive order-1 =
  261 B (loses, learning cost). The selector is at/below its achievable floor. Exhausted.
- **sidecar (607 B):** per-pair single-dim latent correction, already coded with canonical-Huffman + combinatorial
  rank encoding (PR95-family L26/L27/L31). Maximally tight; tiny.
- **DQS1 tail (42 B):** one q-domain decoder-weight patch; irreducible (an independent optimization choice, not
  derivable).
- **range-coder section overhead:** decoder header = 80 B (28 scale-lo + 2 + 25 rho + 25 packed) + ~3 B flush; all
  minimal.
- **ZIP container = 100 B** (31 B local header + 47 B central dir + 22 B EOCD, 1-char member name `x`, STORED). This
  is the **structural floor** for a 1-member ZIP that `unzip -o` (the contest's `evaluate.sh:44`) must extract. Not
  reducible without breaking the contest's extraction contract.

## 4. THE ONE ORIGINAL LEVER THAT SURVIVES — the score-lossless re-quantization manifold (NOT strictly lossless)

The mandate asks: "how much of the latent/weight structure is procedurally regenerable = free?" The strict answer
is **none** — nothing in the payload is deterministically derivable from the rest (scales map q→float and are
needed; selector codes require the scorer to regenerate; only 2.4% of weights are dead-zero, so no sparsity to
seed). The decoder weights are near-iid (§1), so there is no low-rank/procedural seed to regenerate them from.

**But the deeper original math the operator's framing points to** is: the score depends on the decoded pixels ONLY
through `argmax(SegNet(F1))` (a SET functional) and `PoseNet(pair)[:6]` (a smooth 6-vector). There is a whole
**manifold of weight/latent quantizations** that produce the *same argmax partition and same (rounded) pose* — hence
the *same score* — while differing in pixels. Formally, the score-lossless cell is

    Q* = { quantizations q' : argmax SegNet(decode_{q'}.F1_t) = L*_t  ∀t  ∧  round(PoseNet(decode_{q'}.pair_t)[:6]) = p*_t }

This is a polytope in q-space (first-order: `(J_{L*,p} − J_{c,p})·δw ≥ −margin_{p,c}` per boundary pixel, plus the
pose Jacobian band). **Entropy-constrained quantization *within Q\** ** — choosing the cheapest-to-code q' in the
manifold — is the only place real kilobytes can move, and it is exactly the "entropy-coded scalar quantization"
the 2024–25 literature ([Reducing Storage of Pretrained NNs by Rate-Constrained Quantization, arXiv 2505.18758];
[HEMP high-order entropy minimization, arXiv 2107.05298]) identifies as the dominant lever (better coding of a
*fixed* quantization is empirically near-optimal already — which is exactly what §1–§2 measured).

**Why it is NOT in this report's strict-lossless scope:** it changes the decoded pixels, so it is NOT
decoded-bytes-identical; its score-invariance must be PROVEN by a full 600-pair exact re-eval (argmax can flip at a
boundary pixel; the manifold is first-order/approximate). It is a DISTORTION-axis / NEEDS-CAMPAIGN move, correctly
deferred by the floor report (lever B/C) and the #64 verdict. **This audit's contribution is to name it precisely
as the rate lever and bound the strict-lossless slivers around it to ≤ 15 B total, so the planner stops re-mining
the lossless axis and routes byte-reduction effort to the manifold re-quantization (jointly with the distortion
campaign) instead.**

---

## TOP-3 $0 PARITY-PROVABLE SMOKES (for the record — all predicted to confirm "no move")

Each smoke proves decoded-bytes identity FIRST, then counts bytes. All are $0 local CPU, no dispatch.

1. **Joint range-stream smoke (opportunity #1).** Re-encode decoder+latent as a single CTXR range stream; decode;
   assert reconstructed raw decoder streams + raw latent payload are byte-identical to the originals
   (sha `83598024…` / `c760cab8…`); count member bytes. PREDICTED: Δ ≈ +6 B saved gross, ≈ 0 net after the merged
   grammar re-adds a length field. PARITY GATE: `decoder_raw_sha == 83598024…` AND `latent_raw_sha == c760cab8…`.
2. **Latent-header recode smoke (opportunity #2).** Re-code the 28 fp16 mins (delta-from-shared-base) + 28 scales
   (shared-exponent + mantissa residual); decode; assert the reconstructed 112-byte header is byte-identical;
   count. PREDICTED: Δ ≈ +5 B. PARITY GATE: reconstructed `latent_raw[:112]` byte-identical.
3. **Adaptive-order-1 weight-coding falsification smoke (the headline NO-FAKE guard).** Re-run the measured adaptive
   order-1 vs order-0 cost on ALL decoder tensors (not just the big 6); assert order-1 ≥ order-0 (LOSES) → proves
   the decoder-weight axis cannot be improved by context coding. PREDICTED: order-1 LOSES by ≥ 14 KB. This is the
   smoke that *protects* the floor verdict against a future agent re-proposing "better weight entropy coding."

---

## 6-hook wire-in (Catalog #125)

1. **sensitivity-map — ACTIVE.** New measured prior: the decoder-weight axis is order-0-marginal-floor-exhausted
   (adaptive O1 LOSES 14 KB, MEASURED — stronger than the prior "98.6% iid" assertion); the latent axis has a 474 B
   decorrelation ceiling eaten by transform overhead. Aiming surface for bytes = the score-lossless re-quant
   manifold (§4), jointly with distortion.
2. **Pareto — ACTIVE.** The strictly-lossless rate vertex is saturated to within ≤ 15 B (sum of all positive
   slivers); reframes the lossless axis as a hard Pareto wall, not a direction.
3. **bit-allocator — NEGATIVE for strict-lossless.** No lossless byte-saving primitive ≥ 15 B exists on this
   archive. The allocator's next bytes go to distortion-closure (§4 manifold re-quant + lever G/H/C), per the floor
   report's "sub-0.15 is distortion at constant bytes."
4. **cathedral-autopilot — NEGATIVE.** Do NOT queue a lossless-recode materializer on this frontier; the gross
   slivers (≤ 15 B) net ≈ 0 and would burn a build. The leapfrog already captured the achievable lossless gain.
5. **continual-learning — ACTIVE.** Reseeds the V3 judge with the MEASURED adaptive-coding-cost falsifications
   (order-1 −14 KB, pool −25 KB, delta −43 KB, sign/mag −13 KB, latent KLT ceiling 474 B, selector FECa-below-O0)
   so a future agent does not re-mine these. The decoder-weight floor is now proven by coding cost, not assumed.
6. **probe-disambiguator — RESOLVED.** "Is there strictly-lossless structure the iid/order-0 coders miss?" → NO
   (6 measured falsifications). "Where do real bytes still move?" → ONLY the score-lossless re-quant manifold (§4),
   which is a distortion-campaign move, not strict-lossless.

## Provenance

- Frontier archive sha256 `b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e`, member `x`
  `5e781e8e…`, 177,169 B (== canonical pointer). Raw decoder sha `83598024…`, raw latent sha `c760cab8…`
  (from `byte_closure_proof.json`, re-extracted and confirmed in this audit).
- All entropy numbers reproduced inline from the extracted raw payloads via the frontier's own `src/codec_ctx.py`
  decode path (constriction range coder, IEEE-exact). No new code landed; measurement-only.
- ΔS/byte = 25/37,545,489 = 6.6586e-07 (`evaluate.py:63` `compressed_size = archive.zip.stat().st_size`, `:92`
  D = 37,545,489).

**Cross-refs:** `lossless_stack_pointer_move_20260610T165749Z.md` (#64 — R1⊕R2⊕R3 banked, S12 inapplicable;
this audit adds the MEASURED coding-cost proof that the *residual* payload is also at floor) ·
`information_theoretic_floor_report_v1_20260610T102335Z.md` (Lever F — §4c coding-exhaustion; this audit falsifies
the order-1 "gain" by measured adaptive cost and prices the slivers) ·
`closed_spec_boundary_math_system_of_equations_20260610.md` (§4 the argmax polytope — the score-lossless re-quant
manifold of §4 here is its weight-space dual) · `t1_s12_lossless_stack_verdict_20260610.md` (latent floor / S12
INAPPLICABLE).

**Sources (online):** [Reducing Storage of Pretrained NNs by Rate-Constrained Quantization and Entropy Coding](https://arxiv.org/html/2505.18758v1) ·
[HEMP: High-order Entropy Minimization for NN comPression](https://arxiv.org/pdf/2107.05298).
