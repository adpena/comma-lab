# INFLATE-COMPUTE AS A FREE RATE LEVER — ranked menu of decode-time-compute byte savings

UTC 2026-06-11 · claude · RESEARCH + DESIGN. `[macOS-CPU advisory]` / design+derivation-only.
`promotable=false`, `score_claim=false`, no dispatch, no /tmp, no MPS, $0 local. Every byte-savings
number below is **PREDICTED** (mechanism shown), tagged `[predicted]` per NO-FAKE; none is a measured
exact-eval row. GOAL_v3 levers **A** (evaluator-equivalence quotient compiler) + **B** (score-native
decomposition) + **E** (generative micro-prior), framed as "spend the unused inflate compute to cut
`archive.zip` bytes."

    S = 100·d_seg + √(10·d_pose) + 25·B/D        D = 37,545,489   (frozen authority, evaluate.py:92)

---

## 0. THE PREMISE (operator 2026-06-11, grounded in the floor report)

The contest charges rate on **ONLY `archive.zip` bytes**, NOT on inflate compute. The official inflate
budget is **30 minutes** on a **CPU instance: 4 CPUs, 16 GB RAM** (`upstream/README.md:114`; GPU path is
a T4 16 GB but the leaderboard + the faithful local→contest axis is CPU). The current capstone inflate
(`src/tac/capstone_vq_nerv/inflate.py`) is a **single fast numpy forward** — 600 pairs × a 6-block
PixelShuffle decoder + bicubic camera upscale — measured in **seconds-to-low-minutes** on 4 CPUs.

⇒ **We are using a tiny fraction of the 30-min CPU budget.** Decode-time COMPUTE is FREE; bytes are
charged. The lever class: **trade the unused ~25+ minutes of CPU for a SMALLER archive that
reconstructs the SAME witness.**

### Why this is the RIGHT lever right now (the floor report settles it)

`information_theoretic_floor_report_v1_20260610T102335Z.md` MEASURED:
- The frontier is **RATE-bound**: rate = 25·177,169/D = **0.11797 = 100% of S_floor** at zero
  distortion. seg+pose are the recoverable 0.073 residual; the *binding wall is bytes.*
- The current achiever is **lossless-exhausted at ~176 KB** (INT8 decoder at 98.6% of per-tensor iid
  Shannon; latent at ~96.6% of its iid floor). **Better entropy coding of THIS decoder cannot go below
  0.118.**
- The ONLY door below 0.118 is a **smaller achiever / smaller amortizer** (lever A/C class shift) — an
  OPEN compression question with no proven lower bound.

**Decode-time compute is exactly how you build a smaller amortizer.** Every lever below removes bytes
from the archive by replacing STORED information with RECOMPUTED information (recurrence, refinement,
synthesis, denser codes). This is the un-exhausted axis: the floor report exhausted *coding* of the
fixed 177 KB; it did NOT exhaust *shrinking the achiever via decode compute.*

### The byte geography we are attacking (capstone archive, ~177 KB)

| section | bytes | share | what it is |
|---|---:|---:|---|
| `decoder` (merged INT8+brotli) | **~161,380** | **~91%** | decoder + per-frame FiLM weights — **THE TARGET** |
| `latent` (per-pair, LZMA-RAW) | ~10,000–15,387 | ~8% | 28-float/pair VQ-index or stored latent |
| `codebook` | small (paid once) | <1% | "free in decode" (export.py:342) |
| `pose` (fp16+brotli) | ~1,557 | ~1% | 600×6 GT pose — floor-negligible (P6 RESOLVED) |

**~91% of the archive is decoder weights.** The dominant lever MUST shrink the decoder-weight blob (or
replace it with a smaller program). Latent is secondary; pose is already at its floor.

### HARD CONSTRAINTS (every lever below satisfies all four)

1. **NO scorers at inflate** (CLAUDE.md "Strict scorer rule"): inflate.py may NOT load SegNet/PoseNet
   (~73 MB + non-compliant). Every decode-time objective is **scorer-FREE deterministic** (self-
   consistency / stored target statistic / fixed operator). Score-awareness lives at COMPRESS time
   (unlimited compute, has the scorer); inflate is a deterministic forward/iteration.
2. **numpy-portable** (substrate law): inflate runs in pure numpy + brotli/lzma (no torch, no MLX). The
   current `numpy_reference.py` already proves conv2d/pixelshuffle/bilinear/bicubic/sin/FiLM in numpy.
3. **CPU-first** (4 CPUs, 16 GB, ≤30 min): the leaderboard + faithful axis. GPU (T4) is optional
   headroom, never the target.
4. **NO FAKE**: byte savings are `[predicted]` with the removal mechanism named; admitted only after an
   exact dual-axis row proves ΔS<0 (the rent law).

### Prior art (compose / distinguish, do NOT duplicate)

- `src/tac/tto.py` — **scorer-free TTO at inflate ALREADY DOCUMENTED** (temporal-consistency /
  reconstruction / edge-preservation losses; INFLATE_TTO=0 gated; status DEFERRED, no compress-time
  loop). Lever 2 below is its rate-framed activation.
- `src/tac/procedural_replacement_surfaces.py` — seed-derived whole-section replacement matrix
  (canonical equation #26 procedural-codebook). Lever 3/5 below compose with it.
- `src/tac/renderer.py` — channel-recurrent / weight-sharing render heads (Sony approach, L1771) — prior
  art for Lever 1's weight reuse.
- This memo's angle is **"spend inflate compute to cut bytes"** — orthogonal to the carrier-design work
  (HiNeRV/SNeRV byte-pressure). It is a *decode-side* attack on the same rate term.

---

## 1. THE RANKED MENU (decode-time-compute rate levers)

Ranked by **predicted rate-Δ per unit risk** (highest-confidence, decoder-targeting, composes-with-
capstone first). All `[predicted]`.

### L1 — ITERATIVE / UNROLLED DECODE (shared-weight recurrence) — RANK 1, the core lever

**Mechanism.** Replace the 6 *independent* upsample blocks (each with its own conv+FiLM weights) with a
**single small shared block applied N times** (a recurrent/unrolled NeRV: weight-tied across upsample
stages and/or refinement iterations). The same ~C-channel conv kernel is *reused* K times at decode;
the archive stores **one** block's weights instead of six, plus a tiny per-stage FiLM/scale modulation
(cheap) to break the weight-tie symmetry. Decode cost: K× the conv FLOPs (we have 25+ free minutes).

**What it removes.** The bulk of the **decoder blob (161 KB)**. A 6-block taper
`[24,24,24,18,14,12,12]` has ~6 distinct conv weight sets; weight-tying to 1–2 shared blocks + per-
stage FiLM (≈ 2·C scalars/stage) removes ~4–5 blocks' worth of conv params.

**Predicted Δ.** Decoder conv weights dominate the 161 KB. Tying 6→2 shared blocks (conservative — keep
stem + final distinct) removes ~3–4 mid-block conv tensors. At base_ch=24, mid-block conv3×3 ≈
24·24·9 ≈ 5,184 params each; INT8+brotli ≈ ~3–4 KB each post-coding. Removing 3–4 → **−10 to −16 KB**
→ **rate −0.0067 to −0.0107 → ΔS ≈ −0.007 to −0.011 `[predicted]`**. Aggressive full weight-share to a
single iterated block could reach **−30 to −50 KB** (ΔS −0.02 to −0.033) but with real distortion risk
(needs the FiLM modulation to carry the per-stage specialization the tied weights lose).

**Composition with capstone.** Direct: the capstone decoder IS a per-block conv stack; weight-tying is a
structural edit to `numpy_decode_pair`'s block loop + the export weight dict. The render path
(pixelshuffle/bilinear/FiLM) is unchanged. **The single best fit.**

**30-min CPU feasibility.** Trivial — K iterations of the SAME conv the inflate already runs in numpy;
even K=10× the current forward is well under budget (current forward is seconds). ✓

**Scorer-free + numpy-portable.** ✓ pure forward, deterministic, numpy convs already exist. ✓

**Risk.** MEDIUM. Weight-tying reduces capacity → must verify d_seg/d_pose don't regress more than the
rate gain. The capstone's CE-only plateau (MASTER_ROADMAP §0) is a capacity-sensitivity warning: shrink
the decoder only as far as exact-eval d_seg holds. Mitigation: per-stage FiLM keeps most of the
expressivity at ~0 byte cost; sweep K and tie-depth against exact ΔS.

---

### L2 — SCORER-FREE TTO AT INFLATE (refine a tiny seed against a deterministic objective) — RANK 2

**Mechanism.** Store a **tiny seed** (low-rank/low-byte decoder OR coarse latent) plus a small
**deterministic decode-time objective**, then run a few gradient/fixed-point steps at inflate to refine
the reconstruction. The objective is **scorer-FREE** (this is the hard part — see the compliance note):
- **temporal self-consistency** (frame_{2k+1} warped to frame_{2k} via the stored pose ≈ frame_{2k}),
- **reconstruction self-consistency** (a stored *target statistic* — e.g. a downsampled/low-rank
  reference the seed must match after refinement),
- **edge/structure preservation** (Sobel self-alignment).
`tac.tto.py` ALREADY documents these exact three scorer-free losses; this lever is its rate-framed
activation (build the missing compress-time loop + un-gate INFLATE_TTO behind the deterministic path).

**What it removes.** Lets the stored seed be **smaller** than a fully-baked decoder, because decode
compute "finishes the fit." Targets the decoder blob OR the latent: a refined low-rank seed needs fewer
stored params for equal post-refinement quality.

**Predicted Δ.** Conservative (refine to recover a 20–30% smaller seed): **−5 to −15 KB** → ΔS −0.003 to
−0.010 `[predicted]`. The win scales with how much the deterministic objective can recover; bounded by
the objective's information content (you cannot recover what no stored statistic constrains).

**Composition with capstone.** Moderate. Needs (a) a differentiable-in-numpy refinement step (or a
fixed-point iteration — see L1/L4 overlap), (b) a stored target statistic the refinement descends
toward. Composes as a *post-pass* after the L1 iterated decode.

**30-min CPU feasibility.** ✓ A few (5–20) refinement steps on a small seed; numpy autodiff is awkward
but a hand-written fixed-point / closed-form proximal step avoids it (see L4). ≤ budget.

**Scorer-free + numpy-portable.** ✓ **with care** — the objective MUST be a stored deterministic target,
NEVER SegNet/PoseNet. Temporal/recon/edge self-consistency are all scorer-free. ✓ numpy.

**Risk.** MEDIUM-HIGH. (a) The no-scorer rule is easy to violate — any "refine toward what the scorer
wants" is forbidden; only stored self-consistency targets are legal. (b) Refinement can DIVERGE without
the scorer's anchor; needs the stored statistic + step-budget + restore-on-degrade guards (tto.py
already specs these). (c) Determinism across CPU/CUDA numerics must be byte-stable for the eval
roundtrip.

---

### L3 — PROCEDURAL / SUPER-RESOLUTION DETAIL SYNTHESIS (store low-res, deterministically upsample) — RANK 3

**Mechanism.** Store a **low-resolution or low-rank witness** (e.g. render at 192×256 instead of
384×512, or store rank-r factored decoder weights) + a **tiny FIXED refinement operator** (a small
learned-at-compress-time super-res conv, or a deterministic guided upsample). At inflate, deterministically
synthesize the missing high-frequency detail. Decode compute (the super-res pass) is free.

**What it removes.** If the stored witness is at half resolution, the per-pair latent + the high-res
decoder stages shrink (fewer channels needed at the coarse base). Targets BOTH decoder (drop the last
upsample stage's distinct weights → reuse the fixed SR operator) AND latent (coarse latent is smaller).

**Predicted Δ.** Dropping the finest upsample stage + a shared SR operator: **−8 to −20 KB**
→ ΔS −0.005 to −0.013 `[predicted]`. The SR operator itself costs a few KB (paid once, free in decode),
so net is the difference.

**Composition with capstone.** Good. The capstone already bicubic-upsamples 384×512→camera; this lever
adds a *learned* SR stage 192×256→384×512 (a strict generalization of the existing bicubic). Slots into
`render_all_camera_frames` before the camera upscale.

**30-min CPU feasibility.** ✓ One extra conv-based SR pass per frame; cheap. ≤ budget.

**Scorer-free + numpy-portable.** ✓ deterministic conv SR; numpy. ✓

**Risk.** MEDIUM. SR hallucination at SegNet decision boundaries could flip argmax → d_seg regression.
The floor report's seg-margin map says boundary pixels are the fragile ones; SR must preserve the
top-2-margin geometry. Mitigation: compress-time train the SR operator score-aware (it has the scorer)
so the *fixed* operator it bakes is boundary-preserving; verify exact d_seg.

---

### L4 — HEAVIER ENTROPY / ARITHMETIC-CODING UNPACK (spend decode CPU on a denser code) — RANK 4, lowest risk

**Mechanism.** Replace brotli/LZMA with a **slower but denser** decode-time coder: a context-adaptive
arithmetic/range coder with a richer context model (per-tensor, spatial, cross-tensor), or an
asymmetric numeral system (ANS) with a learned-at-compress-time frequency table. Decode is O(symbols)
but the *constant* is higher (the context model evaluates per symbol) — we have the CPU budget. The
floor report says the decoder is at 98.6% of *per-tensor iid* Shannon — but a **cross-tensor / cross-
structure context model** can exploit the residual *conditional* entropy iid coding leaves on the table.

**What it removes.** Pure coding gain on the existing decoder blob — same weights, fewer bytes. The
honest ceiling is small (iid is nearly exhausted), but cross-tensor context is un-measured.

**Predicted Δ.** Conservative (cross-tensor context recovers 2–5% of the 161 KB iid-exhausted blob):
**−3 to −8 KB** → ΔS −0.002 to −0.005 `[predicted]`. Sister-measured floor (frontier_decoder_axis_
waterfill_verdict) caps the iid gain near 1.4%; cross-tensor/structural context is the only headroom and
is bounded.

**Composition with capstone.** Excellent — it is a pure swap of the `_int8_brotli` codec for a denser
one; the render path is untouched. **Lowest-risk, smallest-but-certain.**

**30-min CPU feasibility.** ✓ Arithmetic decode of ~161 KB with a context model is milliseconds-to-
seconds even in numpy. We already have `tac.pr103_arithmetic_codec`. ≤ budget.

**Scorer-free + numpy-portable.** ✓ pure decode; existing arithmetic codec is numpy. ✓

**Risk.** LOW (correctness only — arithmetic coding determinism/CPU-CUDA byte-stability). Bounded upside
(iid nearly exhausted), so this is the *banker* lever, not the breakthrough.

---

### L5 — GENERATIVE MICRO-PRIOR / DETERMINISTIC DENOISING FROM A TINY SEED (GOAL lever E) — RANK 5, highest-ceiling/highest-risk

**Mechanism.** Store a **tiny seed** (a few hundred–few thousand bytes: a coarse latent + a small fixed
denoiser/decoder prior trained at compress time) and run a **deterministic** multi-step denoising /
generative refinement at inflate (diffusion-style but with FIXED schedule + FIXED seed → fully
deterministic, no scorer). The generative prior synthesizes the witness from the seed; decode compute
(many denoising steps) is free.

**What it removes.** Potentially the **entire decoder blob** — replace the 161 KB amortized decoder with
a ~few-KB seed + a small fixed prior (paid once). This is the literal "smaller amortizer" the floor
report names as the only door below 0.118.

**Predicted Δ.** Speculative: if a ~10–30 KB seed+prior reconstructs the witness in the cell, **−130 to
−150 KB** → rate −0.087 to −0.10 → **ΔS ≈ −0.09 to −0.10 `[predicted]`** — i.e. potentially below
S_floor 0.118. This is an OPEN compression question (Kolmogorov-uncomputable lower bound); the floor
report explicitly leaves "does any amortizer beat 177 KB" UNRESOLVED.

**Composition with capstone.** Replaces (not composes with) the capstone decoder — a class shift. The
capstone is the *base*/fallback while this is researched.

**30-min CPU feasibility.** RISK — many denoising steps × full-frame in numpy on 4 CPUs could approach
the 30-min wall for 1200 frames. Must budget: steps × frames × per-step FLOPs ≤ 30 min. A small prior
(few-K params) at 192×256 with ~20 steps is plausibly feasible; a large prior at full res is NOT.
Feasibility is a HARD gate to measure first.

**Scorer-free + numpy-portable.** ✓ fixed-schedule + fixed-seed = deterministic; numpy conv prior. ✓
(determinism + CPU-feasibility are the two real constraints).

**Risk.** HIGH. (a) Whether a tiny seed+prior lands in the evaluator cell at near-zero distortion is
unproven (the open question). (b) CPU-feasibility for 1200 frames × many steps is tight. (c) Generative
hallucination at seg boundaries → d_seg risk (same as L3, worse). Highest ceiling, lowest confidence —
the long-horizon class-shift bet.

---

## 2. RANKED MENU (one line each)

| rank | lever | removes | predicted ΔS | risk | composes w/ capstone |
|---|---|---|---|---|---|
| **L1** | **Iterative/unrolled shared-weight decode** | **decoder blob (91%)** | **−0.007…−0.033** | MED | **direct (best fit)** |
| L2 | Scorer-free TTO at inflate (refine tiny seed) | decoder/latent seed | −0.003…−0.010 | MED-HIGH | moderate (post-pass) |
| L3 | Procedural / super-res detail synthesis | decoder fine-stage + latent | −0.005…−0.013 | MED | good (extends bicubic) |
| L4 | Heavier arithmetic/context unpack | decoder blob (coding only) | −0.002…−0.005 | **LOW (banker)** | excellent (codec swap) |
| L5 | Generative micro-prior / deterministic denoise (lever E) | **entire decoder (class shift)** | **−0.09…−0.10** | **HIGH** | replaces (research) |

**Single highest-rate-Δ lever:** **L5** (generative micro-prior) — predicted −0.09 to −0.10, the only
lever that can break *below* S_floor 0.118 by being the "smaller amortizer." But it is the lowest-
confidence, CPU-feasibility-gated, open-question bet.

---

## 3. TOP 1–2 TO PROTOTYPE FIRST (best rate-Δ per unit risk)

### PROTOTYPE #1 — L1 iterative/unrolled shared-weight decode (RANK 1)

**Why first.** Targets the 91% decoder blob directly; composes natively with the capstone block loop;
CPU-trivial; scorer-free + numpy by construction; MEDIUM risk with a clean exact-eval mitigation
(per-stage FiLM keeps expressivity at ~0 byte cost). Best confidence × magnitude.

**Concrete next step.** In `numpy_reference.numpy_decode_pair` + `capstone_trainer`/`export`: add a
`tie_depth` config that shares one (or two) conv block(s) across the 6 upsample stages, with per-stage
FiLM/scale modulation as the symmetry-breaker. Compress-time: train the tied decoder score-aware (it has
the scorer); inflate-time: the numpy forward iterates the shared block K times (unchanged render path).
Sweep `tie_depth ∈ {6→2, 6→1}` and measure exact dual-axis ΔS vs the base capstone.

**Pre-registered predicted byte savings.** **−10 to −16 KB** (6→2 tie, conservative) → rate −0.0067 to
−0.0107 → **ΔS ≈ −0.007 to −0.011 `[predicted]`**; aggressive 6→1 tie: up to **−30 to −50 KB** (ΔS
−0.02…−0.033) IF d_seg/d_pose hold (the falsifiable gate).

### PROTOTYPE #2 — L4 heavier arithmetic/context unpack (the LOW-risk banker, run in parallel)

**Why second.** Lowest risk (pure codec swap, no render/capacity change), uses the existing
`tac.pr103_arithmetic_codec`, certain (if small) gain, and *stacks* with L1 (L1 shrinks the weights, L4
codes whatever remains denser). Bank a defensive −0.002…−0.005 while L1 is the offensive play.

**Concrete next step.** Swap `_int8_brotli` for a context-adaptive arithmetic coder with a cross-tensor
context model on the decoder blob; measure decode wall-clock (must be ≤ budget — trivially is) + exact
ΔS on the byte-closed archive.

**Pre-registered predicted byte savings.** **−3 to −8 KB** → rate −0.002 to −0.005 → **ΔS ≈ −0.002 to
−0.005 `[predicted]`** (bounded by the near-exhausted iid floor; cross-tensor context is the only
headroom).

**Deferred-but-named:** L5 (generative micro-prior) is the long-horizon class-shift bet that can break
sub-0.118 — gate it on a CPU-FEASIBILITY smoke FIRST (steps × 1200 frames × per-step FLOPs ≤ 30 min on
4 CPUs) before any training investment.

---

## 4. COMPLIANCE CONFIRMATION (each lever, all four constraints)

| lever | scorer-free at inflate | CPU-feasible ≤30 min (4 CPU) | numpy-portable | NO-FAKE (predicted+mechanism) |
|---|---|---|---|---|
| L1 | ✓ pure forward | ✓ K× a seconds-long conv | ✓ convs exist in numpy_reference | ✓ |
| L2 | ✓ self-consistency only (NEVER scorer) | ✓ few refine steps on small seed | ✓ (use closed-form proximal, not autodiff) | ✓ |
| L3 | ✓ deterministic SR conv | ✓ one extra conv pass/frame | ✓ generalizes existing bicubic | ✓ |
| L4 | ✓ pure decode | ✓ ms–s arithmetic decode | ✓ pr103_arithmetic_codec is numpy | ✓ |
| L5 | ✓ fixed schedule+seed = deterministic | RISK — must smoke-gate steps×frames | ✓ numpy conv prior | ✓ (speculative, tagged) |

**The no-scorer rule is the sharpest edge for L2/L5:** any decode-time objective must be a STORED
deterministic target (temporal/recon/edge self-consistency, stored statistic), NEVER a SegNet/PoseNet
forward. Score-awareness is a COMPRESS-time property baked into the fixed operator/seed; inflate only
runs deterministic recomputation.

---

## 5. WIRE-IN (Catalog #125)

1. **sensitivity-map** — ACTIVE. New prior: "decode compute is free; the un-exhausted rate axis is
   shrinking-the-achiever via decode recompute, not better coding of the fixed 177 KB." Re-ranks L1
   (weight-tie) above pure-coding levers.
2. **Pareto constraint** — ACTIVE. Each lever is a rate-axis atom admitted iff exact ΔS<0 (rent law);
   L1/L3/L5 carry a d_seg-regression Pareto wall (capacity/hallucination at seg boundaries).
3. **bit-allocator** — ACTIVE. Allocator hint: the decoder blob (91%) is the byte budget to attack;
   pose (1%) and codebook (<1%) are at floor — do not spend allocator effort there.
4. **cathedral autopilot dispatch** — N/A. Design+derivation surface; no archive bytes emitted yet
   (prototypes will emit candidate atoms into the waterfiller).
5. **continual-learning posterior** — N/A here; the prototype exact-eval rows will reseed it.
6. **probe-disambiguator** — ACTIVE. The open disambiguator (from the floor report) is "does any smaller
   amortizer beat 177 KB at near-zero distortion" — L1 (incremental) and L5 (class-shift) are the two
   complementary probes of that exact question.

## 6. CROSS-REFERENCES

`MASTER_ROADMAP_v3_to_theoretical_floor_20260609.md` (levers A/B/E; Phase 2 rate attack; the 91%
decoder blob) · `information_theoretic_floor_report_v1_20260610T102335Z.md` (RATE-bound floor 0.11797;
lossless-exhausted at 176 KB; "only door below 0.118 is a smaller amortizer" — F4/F6) ·
`src/tac/capstone_vq_nerv/{inflate.py,numpy_reference.py,export.py}` (the single-forward numpy decoder +
byte geography we attack) · `src/tac/tto.py` (scorer-free inflate-TTO prior art, INFLATE_TTO=0 gated,
DEFERRED — L2's activation) · `src/tac/procedural_replacement_surfaces.py` + canonical equation #26
(seed-derived section replacement — L3/L5 compose) · `src/tac/pr103_arithmetic_codec.py` (L4 codec) ·
`src/tac/renderer.py:1771` (channel-recurrent head — L1 weight-share prior art) ·
`upstream/README.md:114` (30-min CPU budget, 4 CPU / 16 GB) · `upstream/evaluate.py:92` (D=37,545,489).
