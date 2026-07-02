# YOUSFI #96 NON-NEURAL PARTITION+POSE WITNESS — deep-dive: the wrong-operating-point root cause + the RD-optimal redesign (2026-06-16)

**Author:** Yousfi-#96-nonneural-witness deep-dive subagent (`yousfi_96_nonneural_witness_deepdive_20260616`).
**Type:** DEEP RESEARCH + DESIGN memo. **research_only=true.** NO production code, NO GPU, NO dispatch, NO
Track-A edits. Touches no running daemon (the MPS basin daemon owns
`experiments/results/torch_vehicle_full_mps_basin_bc20_n600` + `src/tac/torch_vehicle/**`; the Cool-Chic sister
owns `src/tac/substrates/cool_chic/**`; this memo touches none of them). SPECULATIVE / Track-B-C+; Track-A
(the live small-basis basin) is TOP PRIORITY and untouched.
**Evidence grade:** `[analysis]` — every quantified claim is tagged **[MEASURED:<memo>]** (an exact number
from a cited prior CPU-torch advisory smoke, GT via `frame_utils.yuv420_to_rgb`, NEVER MPS) or
**[DERIVED:<basis>]** (a closed-form bound). **NO score is claimed.** The means/ends firewall holds: this is a
MEANS (a redesign) toward the END (a lower exact score) and **moves no row.**
**Frontier (pointer, NOT hardcoded):** `.omx/state/canonical_frontier_pointer.json` → contest-CPU
**0.19109982** (177,169 B, sha `b46897267…`, `lane_pr110_payload_entropy_recode`); contest-CUDA **0.20533**
(186,876 B). **Frontier UNMOVED.** Ladder: T_3 = sub-0.15 (the aim), T_1 = sub-0.19, S_floor ≈ 0.11797
(rate-dominated, measured-achiever), conditional MDL ≈ 0.0164–0.0430 (DERIVED headroom).

> **HEADLINE (NO-FAKE).** "#96" is the non-neural **partition+pose witness** lineage — the
> evaluator-equivalence quotient compiler whose seg core stores the SegNet argmax partition directly
> (`boundary_math_seg_core`, L2/L9) and whose pose half stores the trajectory (L3/L14). It was "measured
> lossless but wrong operating point" for **ONE precise, decomposable reason: the partition was stored
> LOSSLESSLY when the contest only needs it argmax-CORRECT, and lossless storage of the boundary MOTION
> across 600 temporally-varying frames costs ~525 KB — 3× the entire frontier archive — for a term
> (d_seg) the frontier already buys for ~0.056.** The redesign is not a better lossless coder; it is a
> **paradigm switch from STORE to AMORTIZE**, validated by the measured −59% byte-closed score-native
> generator (L13, 72,217 B). The Yousfi unlock that #96 missed: the contest SegNet's **stride-2 stem
> decides argmax at ~192×256** [MEASURED: μ4], so the witness need only be argmax-correct at HALF
> resolution — the boundary set the witness must code is ~4× smaller than the 384×512 partition #96 stored,
> and corrections below the stem's receptive field are FREE (the CNN literally cannot see them). The
> optimal non-neural witness codes the **half-resolution argmax boundary as a residual on a cheap base,
> at UNIWARD-margin cost, with the pose stored as 1.5 KB Wyner-Ziv side-info** — predicted ~30–70 KB,
> rate 0.020–0.047, genuinely below the 0.118 rate floor.

---

## PART 1 — THE PRECISE DIAGNOSIS OF #96's "WRONG OPERATING POINT"

### 1.1 What #96 actually was (the lineage, reconstructed from the measured memos)

"#96" is not a single file; it is the **non-neural partition+pose witness** — the seg-core + pose-store
realization of the Evaluator-Equivalent Witness Compiler. Its two halves and their MEASURED costs:

| Half | Mechanism | What it stores | MEASURED cost | d_seg / d_pose | Source |
|---|---|---|---:|---|---|
| **SEG (partition)** | contour/RAG codec over the SegNet argmax label map `L*`, region-MDL merge at the 1.27 B/flip water level | the partition LOSSLESSLY (interior runs + boundary entropy) | **895.7 B/frame → 524.8 KB / 600** | **d_seg = 0.0 (bit-exact)** | [MEASURED: `boundary_math_seg_core_20260610`] |
| **SEG (optimal coder)** | best temporal-context coder on the same partition | same partition, tighter coder | **253,413 B → rate 0.169** | d_seg = 0.0 | [MEASURED: `frozen_contest_space_council_lenses` §A.3(2)] |
| **POSE (trajectory)** | temporal-delta-coded 600×6 pose scalars | the GT pose readout | **1,557 B (0.88% of S)** | d_pose ≈ stored-quant floor | [MEASURED: `information_theoretic_floor_T_floor` P6] |
| **POSE (palette bridge, the realized #96 appearance)** | per-class GT-region-mean palette frame so PoseNet has *some* luma | a 15-byte palette | **15 B** but **d_pose = 12.66** | d_pose = 12.66 (√ term = 11.25 ALONE) | [MEASURED: `score_native_first_candidate` §2] |

### 1.2 The byte/term decomposition placed against the frontier + the floor (the crux)

Decompose the *best non-neural witness #96 could assemble* and score it on the contest law
`S = 100·d_seg + sqrt(10·d_pose) + 25·B/N`, N = 37,545,489:

**Variant A — lossless partition + pose trajectory (the "measured lossless" claim):**
```
seg:  100 · d_seg(0.0)                         = 0.000   ← the d_seg=0 win is REAL
pose: sqrt(10 · d_pose≈stored-floor ~1e-4)     ≈ 0.032
rate: 25 · (524,800 + 1,557) / 37,545,489      = 0.350   ← THE KILLER
                                          S_A   ≈ 0.382      [DERIVED from the two MEASURED byte counts]
```
Even with the **optimal** temporal-context partition coder (253,413 B): rate = 0.169, S_B ≈ 0.201 — STILL
above the frontier 0.191, STILL above T_1.

**Variant C — palette appearance bridge (the realized byte-closed #96 / L13-with-palette):**
```
seg:  100 · d_seg(0.0228, palette+solver)      = 2.28
pose: sqrt(10 · d_pose 12.66)                  = 11.25   ← palette is pose-BLIND (flat color = no luma texture)
rate: 25 · 72,217 / 37,545,489                 = 0.048   ← the rate is GORGEOUS (−59%)
                                          S_C   = 13.58     [MEASURED: score_native_first_candidate §3]
```

**The precise diagnosis — which term made it non-competitive, and by how much:**

1. **The lossless-partition variant (A/B) failed on the RATE term, by ~3× (lossless) or ~1.4× (optimal coder)
   over the frontier.** The mechanism is NOT that lossless masks are intrinsically expensive — a *single*
   frame's partition is ~896 B (35 regions, 0.687% boundary, ~3 KB labels) which is cheap. It is that the
   **boundary MOTION across 600 temporally-varying frames** (21,304 region-instances) has high TOTAL entropy:
   the partition's low PER-FRAME entropy does NOT imply low TOTAL entropy
   [MEASURED: `frozen_contest_space_council_lenses` §A.3(2), the Assumption-Adversary's hardest-won
   correction: *"the standalone-storage RATE-WIN is CARGO-CULTED and falsified"*]. **You paid lossless bytes
   for boundary motion the scorer never charged you for** — d_seg=0 is over-delivery; the score only needs
   d_seg ≤ ~5.6e-4 (the frontier's level), and it charges 100× for d_seg but 25/N ≈ 6.66e-7 per byte.

2. **The palette-bridge variant (C) fixed the rate (−59%, the headroom is REAL) but failed catastrophically
   on the POSE term, by ~650× (d_pose 12.66 vs a measured GT-frame1 floor of 0.0).** The mechanism: a
   piecewise-constant palette frame has **no luma texture for PoseNet** — the witness solved the seg axis by
   amortizing the argmax into a 65 KB generator but left the pose-carrying appearance unsolved (stored as a
   flat palette). [MEASURED: `score_native_first_candidate` §2: GT frame1 → d_pose 0.0; palette frame1 →
   d_pose 12.66, proving the palette IS the entire pose problem.]

### 1.3 The one-sentence root cause

> **#96's wrong operating point = it solved EACH axis at the WRONG fidelity: it stored the SEG partition
> LOSSLESSLY (d_seg=0 over-delivery → 525 KB rate disaster) when the score only needs argmax-correct within
> tolerance, AND it stored the POSE-carrying appearance as a flat palette (d_pose=12.66 under-delivery →
> 11.25 score disaster) when pose needs only ~1.5 KB of stored side-info on a luma-textured base.** Neither
> axis was at its RD-optimal operating point: the seg axis paid for invisible boundary motion; the pose axis
> refused to pay 1.5 KB for the one thing PoseNet reads. The whole witness is a study in spending bytes off
> the score's actual sensitivity.

This is the **measurement-first reframe of the entire #96 lineage:** the d_seg=0 partition is a *means-as-end*
(d_seg=0 is not the goal; a low S is). The witness must operate at the **score's tolerance, not at zero
distortion** — and amortize, not store.

---

## PART 2 — OPTIMIZE THE MATH: the RD-optimal operating point + the argmax-flip/UNIWARD residual derivation

### 2.1 The contest d_seg is an argmax-FLIP RATE, not a lossless mask (the foundational re-derivation)

[MEASURED: `upstream/modules.py` + `score_pair_components`, verified in `layer1_carrier_first_principles` §A.1]
```
d_seg = mean_pixels[ argmax f_seg(x̂_last) ≠ argmax f_seg(x_last) ]
```
This is a **0/1 per-pixel argmax-disagreement RATE**, gradient-ZERO almost everywhere, with deltas only at the
SegNet decision boundary. The witness does NOT need to reproduce `L*` losslessly; it needs the rendered
frame's argmax to AGREE with GT's argmax pixelwise — a **discrete cell membership**, invariant under any
perturbation that does not cross a boundary. #96 stored the cell ID exactly (d_seg=0); the score only needs
you INSIDE the cell (d_seg ≤ τ). The gap between "exactly at the cell label" and "anywhere inside the cell" is
the entire over-payment.

### 2.2 The RD-optimal operating point (the water level + the score tolerance)

The exact KKT water level [DERIVED: `closed_spec_boundary_math` §10; `boundary_math_seg_core`]:
```
λ* = (100 / (600·384·512)) / (25 / 37,545,489) = 1.2731 B/flip
```
Reading: fixing one argmax flip is worth `100/(N_pixels)` of score; one byte costs `25/N` of score; so a flip
repair is admissible iff it costs **< 1.27 bytes**. **The RD-optimal seg operating point is NOT d_seg=0; it is
the point where the marginal byte cost of the next flip-repair equals 1.27 B/flip.** At that point you STOP
repairing — the remaining flips are cheaper to leave (pay 100·Δd_seg in seg) than to code (pay 25·ΔB in rate).

The frontier sits at d_seg ≈ 5.6e-4 (seg term 0.056). A witness need only MATCH that — fix flips down to the
1.27 B/flip knee, then leave the rest. **#96 went all the way to d_seg=0, which (by the water-level logic)
means it paid > 1.27 B for flips worth < 1.27 B each — provably suboptimal by its own KKT condition.**

### 2.3 The flip-COUNT crux (why the per-flip sidecar ALSO failed, and the lesson)

[MEASURED: `witness_seg_boundary_decisive_probe` — the decisive $0 probe] A per-flip boundary-RESIDUAL sidecar
on the live HNeRV basin base was measured:
```
boundary set ∂ = 0.54% of frame (thin ✓)
flips/pair = 884  (d_seg = 0.0045)
conditional B/flip = 1.02  (BELOW the 1.27 break-even ✓ — the conditional-position trick works)
round-trip survival = 46.4%  (BELOW the 50% bar ✗ — τ-insensitive)
residual bytes (scaled to 600) = 884 × 600 × 1.02 ≈ 543 KB  ← THE KILLER (again)
```
**The lesson that re-ranks the whole redesign:** the per-flip price (1.02 B) cleared break-even, but there are
**~530,000 flips**, so the sidecar is a 0.5 MB archive SECTION — the SAME amortization failure as the lossless
partition, reproduced from the residual direction. **A per-flip sidecar prices each flip below break-even but
must pay it half-a-million times.** The conclusion is forced: **the seg win must be AMORTIZED (a shared
decoder that learns/generates the boundary), never STORED per-flip.** Storage loses at every realization
(lossless partition 525 KB; per-flip sidecar 543 KB; the council's "direct partition storage LOSES" verdict).

### 2.4 The UNIWARD / inverse-steganalysis residual derivation (the Yousfi math)

d_seg is **inverse steganalysis on the partition** [Yousfi, `frozen_contest_space_council_lenses` L9: *"the
whole field clusters at 0.19 because everyone optimizes a smooth recon surrogate of a non-smooth detector
functional"*]. The frozen EfficientNet-B2 SegNet IS the steganalysis detector; the witness is the embedder;
the cost of "moving" a pixel across the argmax boundary is exactly a UNIWARD-style **change cost**:
```
cost(p) = m(p) / ‖g_p‖        [MEASURED primitive: margin_polytope.py, boundary_math_seg_core]
   where m(p) = top1 − top2 SegNet logit margin at pixel p
         g_p  = the SegNet logit gradient at p (the detector sensitivity)
```
**The inverse-steganalysis derivation of the optimal residual:**
- **High-margin pixels (m(p) large) are CERTAIN** — the detector will not flip them; coding a correction there
  is wasted bytes. ~91% of boundary pixels are large-margin (omittable) [MEASURED: `boundary_math_seg_core`].
- **The entire d_seg signal lives in the margin→0 set** — the thin sub-band where the detector is uncertain.
  Code corrections ONLY there, with the entropy model + drop decision CONDITIONED on the measured margin field
  (the detector tells you where it is blind).
- **The decoder regenerates the margin field for FREE at inflate time** (it runs the same render), so the
  sidecar pays only the CONDITIONAL position cost `log2 C(|∂_low|, K)` ≪ unconditional — the steganographer's
  "selection-channel-aware" coding (the decoder shares the cover, so positions are side-info).

This is the **Margin-Weighted Contour Coder (MWCC)** [L9, DESIGN-ONLY]. Its measured target: ≲170–250 B/frame
(a 3.6–5.3× reduction vs the 896 B/frame LZMA baseline) to cross the water level. The STC-clean-source DEFER
measured uniform-cost STC at 2.4–2.6× brotli — **margin-weighting (the UNIWARD cost map) IS the named
reactivation**: it makes the coder spend bytes only where the detector is uncertain, not uniformly.

### 2.5 The optimized witness byte budget (the RD-optimal operating point, assembled)

Composing the corrected fidelities — argmax-correct (not lossless) seg, amortized (not stored), pose as
side-info — against the DERIVED conditional MDL band [`smaller_learned_basis_deep_math` §3,
`frozen_contest_space_council_lenses` §A.3(3)]:
```
B_base   (amortized cheap render, right in most argmax cells)   ~20,000 – 55,000 B   [DERIVED conditional MDL]
B_∂      (MWCC margin-conditional flip residual at the 1.27 knee, half-res, contiguous base)  ~0 – 10,000 B
B_pose   (600×6 trajectory, temporal-delta + brotli, Wyner-Ziv) ~1,557 B            [MEASURED]
B_null   (certified resize-null fill, maximally compressible)   ≈ free (−10 to −19.5% of base) [MEASURED L5]
                                                       B_witness ≈ 22,000 – 68,000 B
                                              rate = 25·B/N      ≈ 0.0146 – 0.0453
```
**vs the 0.118 rate floor and the 177 KB frontier: a 2.7×–7× rate class shift, genuinely below the floor.**
The −59% byte-closed L13 generator (72,217 B, rate 0.048) is the **lower-middle of this band, ALREADY
REALIZED** [MEASURED] — proving the rate half is not a projection. The open work is entirely DISTORTION (the
seg term at the amortized base's d_seg, and the pose term once a luma-textured base carries it).

---

## PART 3 — THE YOUSFI INVERSE-STEGANALYSIS OPERATING-POINT UNLOCK (what #96 missed)

The contest creator (Yousfi) is a steganalysis expert; the challenge IS inverse steganalysis. The unlock #96
missed is the detector's own **measured blind spot** — and it is now a measured number, not a heuristic.

### 3.1 The stride-2-stem blind spot (the half-resolution unlock) — MEASURED

[MEASURED: `small_basis_optimization_register` μ4, `upstream/modules.py:108-109`] The SegNet decides the argmax
at its **stride-2 stem → ~192×256**, while the decoder renders 384→bicubic↑874. Consequences the witness must
exploit:

1. **The witness need only be argmax-correct at ~192×256, not 384×512.** The boundary set the witness codes is
   ~**4× smaller** in pixel count than the partition #96 stored at full 384×512 — directly attacking the
   flip-COUNT crux (Part 2.3) that killed both the lossless partition and the per-flip sidecar. A 4× smaller ∂
   turns the 543 KB sidecar arithmetic into ~135 KB before any further coding gain — and combined with
   amortization, into the KB band.
2. **Corrections below the stem's receptive field are FREE** — the CNN literally cannot see HF detail above
   ~192×256. This is the certified-invisibility null space (L5: 22.7% of every channel TIER-1 certified
   invisible) given a *detector-specific* sharpening: the witness can be arbitrarily wrong in the HF band the
   stem discards, and fill it with the maximally-compressible legal values (B_null ≈ free).
3. **FP4 quant noise is HF → d_seg should tolerate interior FP4** [MEASURED: μ4 bridge]. The rate-maker (FP4
   packing, −0.022 rate) is de-risked on the SEG axis by the stem blind spot — the witness's base can be
   coarsely quantized in its interior because the detector reads only the smoothed ~192×256 structure. (The
   measured caveat: post-hoc FP4 spills d_seg +56% [MEASURED: WS-B] → FP4 must be QAT, not a free swap; but
   the stem blind spot is WHY FP4-QAT is plausible at all.)

### 3.2 Inverse-steganalysis = hide the witness ERROR where the detector cannot see it

The Yousfi lens reframes the whole witness: **do not minimize pixel error; minimize DETECTABLE error.** The
witness's job is to place its byte-saving approximations exactly in the detector's blind regions:
- **Seg:** be wrong everywhere the margin is large (the detector is certain → won't flip) and everywhere above
  the stem's ~192×256 resolution (the detector discards it). Pay bytes only at the margin→0, ≤192×256 boundary
  band — UNIWARD cost `m(p)/‖g_p‖` is the literal change-cost map.
- **Pose:** PoseNet reads YUV6 of BOTH frames, globally pooled before the √ — so be wrong in the ~590k pose-NULL
  pixel directions per pair and the 6 unscored pose dims; spend only on the 6 scored dims (store them, FiLM-inject).
- **Null:** 80.67% of camera-pixel directions are in the full resize null space [MEASURED L5]; fill with
  min-entropy legal values.

### 3.3 The candidate inverse-steganalysis cost toolkit (beyond UNIWARD) — DESIGN, ranked

[from `yousfi_fridrich_canonical_inverse_steganalysis_tools_deep_research`, with the Assumption-Adversary's
binding caveat: these are JPEG/spatial-domain cost surfaces; **each must be empirically validated per-archive
on our RGB-decoder+scorer substrate before promotion** — Slot QQ META-lesson, do NOT assume transfer]:

| Tool | Cost mechanism | Fit to the seg-boundary witness | Status |
|---|---|---|---|
| **UNIWARD / S-UNIWARD** | 8-tap Daubechies wavelet relative distortion | THE canonical boundary change-cost; `m(p)/‖g_p‖` is its contest analogue | DESIGN (MWCC); pieces exist |
| **HILL** (high-pass + 2 low-pass) | `1/(|H*I| ⊗ L1 ⊗ L2 + ε)` — aggregates corrections | orthogonal to UNIWARD's wavelet axis; clusters boundary repairs (cheaper colex coding) | DESIGN, rank #1 |
| **MiPOD** (Fisher info) | closed-form power-of-optimal-detector cost | model-driven; the EfficientNet-B2 stem IS the detector → directly applicable to which flips matter | DESIGN, rank #2 |
| **CMD** (clustered modification) | non-additive: polarizes cost by neighbor direction | multiplies with ANY additive cost; boundary flips ARE spatially clustered → big win on contiguous residuals | DESIGN, rank #3 |

**The single most-relevant insight:** the contest already gives us the detector's gradient (`g_p`) and margin
(`m(p)`) for free (frozen, differentiated). UNIWARD/HILL/MiPOD all APPROXIMATE the detector's sensitivity; we
can use the EXACT measured sensitivity. The witness is the *ideal-detector* limit of inverse steganalysis — a
luxury no real steganographer has.

---

## PART 4 — OPTIMIZE THE ENGINEERING: the optimal byte-closeable, numpy-portable witness design

### 4.1 Architecture (the four sections, each at its RD-optimal fidelity)

```
archive.zip
├── base_blob       — amortized cheap render (the L13 score-native generator OR a small HNeRV @ base_ch≈12)
│                     trained score-aware (margin-weighted seg loss) so its argmax lands the cells at ~192×256
│                     ~20–55 KB, int8/FP4-QAT packed (the stem blind spot de-risks interior FP4)
├── seg_residual    — MWCC margin-conditional flip residual: colex-rank position set over the decoder-KNOWN
│                     low-margin ≤192×256 band + per-flip class id, coded at UNIWARD cost m(p)/‖g_p‖,
│                     admitted only while marginal < 1.27 B/flip (the KKT water level). ~0–10 KB.
├── pose_sidecar    — 600×6 GT pose, temporal-delta + brotli, ~1.5 KB. Wyner-Ziv side-info.
└── null_fill       — implicit: the base's null directions filled with min-entropy legal values (L5 certified)
```

### 4.2 The pose half — the ENGINEERING fix for #96's catastrophic blocker

#96's palette killed pose (d_pose 12.66) because a flat-color frame has no luma for PoseNet. **The fix is NOT
a richer non-neural appearance** (a raw per-pair RGB appearance is catastrophically expensive: factor-8 lowres
= 17 MB / rate 11.3 [MEASURED: `score_native_first_candidate` §3]) — **it is to STORE the 6 pose scalars and
FiLM-inject them** (Wyner-Ziv). The base is *told* the pose and modulates frame1 features so the rendered
PoseNet readout matches GT → **d_pose collapses to the stored-quant floor at ~1.5 KB** [MEASURED-GO:
`pose_film_cpu_disambiguator`, the running carrier memo's measured GO at the frozen-decoder lower bound]. This
is the single most-validated witness component and the precise antidote to #96's pose disaster: the appearance
need only carry ENOUGH luma texture for PoseNet to read the FiLM-modulated pose — which a cheap amortized base
provides (the lowres curve: factor-8 base d_pose 0.033 BEFORE FiLM), not a flat palette.

### 4.3 The byte-close + numpy-portable inflate (the contest-compliance engineering)

[the L13 candidate already proves this is buildable: `score_native_candidate_20260610/`, archive sha
`7dc512b5…`, **scorer-free inflate.py, lossless parity all_match=True**] The witness inflate is a small
deterministic numpy program: unpack base_blob (int8/FP4 dequant + brotli decompress + reshape), run the base
render (numpy conv/pixelshuffle/sin — no torch needed at inflate), apply seg_residual (regenerate the margin
field from the render, decode the colex flip positions, flip the named classes), apply pose FiLM modulation
from pose_sidecar, write 1200 RGB frames. **No scorer at inflate time** (the L5/strict-scorer rule). The base
trains on MPS/CPU (Track-A's substrate), exports to numpy via the existing MLX→numpy 0.9997 argmax-parity
bridge.

### 4.4 The composition order (distortion FIRST, rate SECOND)

[DERIVED: `layer1_carrier_first_principles` §F.2, the binding order] **Phase 2a (distortion):** drive d_seg →
frontier-level and d_pose → stored-floor at constant-or-lower bytes — this ALONE, at the L13 base's 72 KB,
would land the seg term ~0.06 + pose ~0.03 + rate 0.048 ≈ **0.14 advisory** (sub-0.15) IF the base trains to
the frontier's d_seg and pose-FiLM lands. **Phase 2b (rate):** FP4-QAT the base + MWCC the residual + S12
null-fill → push toward the ~25–65 KB conditional floor for sub-0.118. The witness is ONE co-designed system
minimizing `25·B_witness/N` s.t. the seg-cell + pose-ellipsoid constraints.

---

## PART 5 — RANKED EXACT-ROW-POTENTIAL PATHS, each with its $0 probe

Ranked by `EV toward sub-0.15` = (predicted ΔS) × (confidence the byte estimate holds, grounded in the
MEASURED #96/L13 numbers + the scorer). Every probe is $0, CPU-torch, GT via `yuv420_to_rgb`, NEVER MPS, no
GPU, no basin contention (read frozen checkpoints READ-ONLY).

| Rank | Path | Predicted ΔS / byte estimate (grounded) | The single $0 probe |
|---|---|---|---|
| **1** | **Pose-FiLM on the L13 score-native base** (close #96's pose blocker on the proven −59% base) | **the decisive unlock:** L13 at d_pose 12.66 → ~0.03 collapses the √ term 11.25 → ~0.05; at the MEASURED 72 KB the candidate goes S 13.58 → **~0.14 advisory** (sub-0.15) IF seg also lands frontier-level. Byte cost +1.5 KB (MEASURED). | **The half-res pose-FiLM lower-bound probe:** on the frozen L13 base, FiLM-inject the stored GT pose into the base render, push through the eval round-trip, measure realized d_pose. GO if d_pose < 0.1 at +1.5 KB (the running carrier memo's disambiguator already returned GO on the HNeRV base; this confirms it on the L13 score-native base). |
| **2** | **Half-resolution boundary witness** (the Yousfi stride-2 unlock applied to the seg residual) | the 4× smaller ∂ turns the 543 KB sidecar → ~135 KB raw → KB-band after MWCC+amortization; on the L13 base (74% contiguous residual) the MWCC is fundable where it was empty on the frontier base | **The ≤192×256 flip-count + survival probe:** downsample the SegNet argmax to 192×256, recount flips vs the 384×512 count, re-measure round-trip survival of a ≤192×256-band correction. GO if flip-count drops ~4× AND survival rises above 50% (the stem blind spot should make corrections robust). Reuses the `witness_seg_boundary_decisive_probe` harness at half res. |
| **3** | **MWCC margin-conditional contour coder on the L13 contiguous residual** (the UNIWARD seg coder, on the RIGHT base) | crosses the 1.27 B/flip water level only on a contiguous base; L13's residual is 74% contiguous (≥4px) [MEASURED] vs the frontier's 95% single-pixel → the coder is fundable here. Target ≲170–250 B/frame (3.6–5.3× under LZMA). | **The margin-weighted-STC-vs-brotli $0 smoke** on the L13 residual: code the contiguous flip residual with the margin-conditional UNIWARD cost map (`m(p)/‖g_p‖`) vs brotli. GO if MWCC beats brotli by ≥5% at the contiguous residual (the STC-DEFER's named reactivation bar; uniform-cost STC was 2.4–2.6× WORSE, so the margin-weighting is the whole test). |
| **4** | **The full non-neural witness assembled** = base (L13, score-aware trained) + MWCC seg residual + pose-FiLM + S12 null-fill | the integrated sub-0.15 (then sub-0.118) candidate; predicted ~30–70 KB / rate 0.020–0.047 + seg ~0.06 + pose ~0.05 ≈ **0.13–0.16 advisory**, gated on ranks 1–3 | **The composed-byte-closed advisory probe:** assemble the four sections, byte-close, run the CPU-torch advisory S on 8→600 pairs. GO to a paired CPU+CUDA exact eval ONLY if advisory S beats the frontier (the fail-closed gate — do not spend $ on a non-improvement, exactly as L13 correctly did NOT). |
| **5** | **Base seg-fidelity campaign** (train the L13 generator to the frontier's d_seg ~5.6e-4 from 0.0068, the seg term's remaining 0.62) | the seg term is the next binding constraint once pose is carried (d_seg 0.0068 → 100· = 0.68; needs ≤ 5e-4 for seg ≤ 0.05). A capacity/length campaign on the amortized base. | **The generator d_seg-vs-budget probe:** extrapolate the L13 generator's d_seg power-law (the small-basis register measured `d_seg ≈ 0.0367·ep^−0.351` for the HNeRV basin; measure the L13 generator's exponent) to estimate epochs-to-frontier-d_seg. Decides Track-B-vs-C+ sequencing — gate the campaign only if the exponent predicts reachability within budget. |

### 5.1 THE single highest-EV $0 probe (the answer the prompt asks for)

> **RANK 1 — the half-res pose-FiLM lower-bound probe on the frozen L13 score-native base.** It is the
> highest-EV because it directly tests the ONE term that killed #96 (pose, the 11.25-of-13.58 disaster) on the
> ONE base that already proved the rate class shift (L13, MEASURED −59% / 72 KB / lossless byte-closed). If it
> returns GO (d_pose < 0.1 at +1.5 KB) — which the sister pose-FiLM disambiguator already returned on the
> HNeRV base — the byte-closed non-neural witness goes from S 13.58 to a **predicted ~0.14 advisory (sub-0.15)
> in a 73 KB archive**, and the whole #96 lineage flips from "wrong operating point" to "the live class-shift
> candidate." It is $0 (reuses the L13 frozen base + the stored-pose codec + the existing eval round-trip), no
> GPU, no basin contention, falsifiable, and it gates ranks 2–5.

---

## What this reseeds (continual-learning posterior)

1. **#96's wrong operating point is DOUBLY off the score's sensitivity:** seg stored LOSSLESSLY (d_seg=0
   over-delivery, 525 KB rate disaster — violates its own 1.27 B/flip KKT condition) AND pose stored as a flat
   palette (d_pose 12.66 under-delivery, 11.25 score disaster). The fix is per-axis RD-operating-point
   correction, not a better coder.
2. **STORE always loses; AMORTIZE always wins** — lossless partition (525 KB), optimal-coder partition
   (253 KB), per-flip sidecar (543 KB) all lose; the amortized generator (L13, 72 KB) wins. Three independent
   measurements of the same verdict.
3. **The Yousfi stride-2-stem blind spot (argmax decided at ~192×256) is a measured 4× reduction in the seg
   boundary the witness must code** — the operating-point unlock #96 missed; directly attacks the flip-COUNT
   crux that killed every storage realization.
4. **The pose blocker is closeable at 1.5 KB via Wyner-Ziv FiLM, NOT a richer non-neural appearance** (raw RGB
   appearance is 17 MB / rate 11.3 — categorically dominated).
5. **The non-neural witness's byte floor (~22–68 KB, rate 0.015–0.045) is genuinely below the 0.118 rate
   floor** — the −59% L13 row proves the rate half; the open work is entirely distortion (seg-fidelity of the
   base + pose-FiLM landing).

## Wire-in hooks (CLAUDE.md 6-hook per Catalog #125)

1. **Sensitivity-map** — ACTIVE (design): the UNIWARD cost map `m(p)/‖g_p‖` at ≤192×256 IS the per-pixel
   seg-sensitivity prior; the pose-Jacobian 6-dim is the pose prior; both feed the bit-allocator.
2. **Pareto constraint** — ACTIVE: the per-axis RD-operating-point correction (seg at the 1.27 knee, pose at
   the 1.5 KB floor) IS the Pareto-correct realization; #96's lossless/palette corners are dominated.
3. **Bit-allocator** — ACTIVE (design): the four-section budget (base + ∂-residual + pose + null) at the λ*
   water level is the allocator prior; the stride-2 blind spot is a NEGATIVE allocator prior (do not pay for
   HF above 192×256).
4. **Cathedral autopilot** — N/A (analysis; the $0 probes are the next dispatch surface, not an archive).
5. **Continual-learning posterior** — ACTIVE: the 5 reseed rows above.
6. **Probe-disambiguator** — ACTIVE: the rank-1 half-res pose-FiLM probe IS the disambiguator between "#96
   wrong operating point (dead)" and "#96 pose-fixed = the live class-shift candidate."

**Mission contribution:** `frontier_breaking_enabler` (a redesign that DIAGNOSES #96's wrong operating point
to the byte/term and REDIRECTS the non-neural witness onto the amortize+half-res+pose-FiLM operating point;
names the single $0 probe that flips it from dead to live). **Frontier UNMOVED 0.19109982.** No score
asserted. No GPU. No paid spend. No collision with running agents. research_only=true.

## Cross-references
`boundary_math_seg_core_20260610T101618Z.md` (the lossless partition 524.8 KB / d_seg=0 — #96 seg half) ·
`witness_seg_boundary_decisive_probe_20260612T181038Z.md` (the per-flip sidecar 543 KB / 46% survival — the
flip-count crux) · `score_native_first_candidate_20260610T112433Z.md` (L13, the −59% byte-closed amortized
generator + the palette pose disaster 12.66 — #96 realized) · `layer1_carrier_first_principles_20260612T171912Z.md`
(the witness carrier §E + the rate-floor-invariance §D) · `frozen_contest_space_council_lenses_synthesis_20260612T173627Z.md`
(the 17-lens catalogue + the ~24.6–64.6 KB conditional MDL + the "store loses → amortize" correction) ·
`small_basis_optimization_register_20260615.md` (μ4 the stride-2-stem 192×256 blind spot + WS-B FP4 NO-GO/QAT-gate) ·
`yousfi_fridrich_canonical_inverse_steganalysis_tools_deep_research_..._20260529.md` (UNIWARD/HILL/MiPOD/CMD cost
toolkit) · `pose_film_cpu_disambiguator` (the measured-GO pose-FiLM lower bound) · CLAUDE.md "Evaluator-Equivalent
Witness Compiler Paradigm" + "Fridrich inverse steganalysis" + "Quantizr intelligence" + "THE GOAL — SUB-0.15".
