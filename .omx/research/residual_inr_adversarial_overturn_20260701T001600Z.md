# Residual-INR NO-GO — adversarial overturn + the REAL n600-witness error structure (MEASURED)

- **UTC** 2026-07-01T00:16Z · **authority** `[macOS-CPU / MLX research-signal] NON-PROMOTABLE`
- **pointer UNMOVED 0.19110** · score_claim **false** · promotable **false** · ready_for_exact_eval **false**
- **Scope** CPU-only, $0, NO GPU, NO training launch, NO edit to the live n600 trainer (pid 38641 /
  sibling a1dd03403 untouched). All artifacts are NEW files under
  `experiments/results/residual_inr_adversarial_overturn_20260630T235910Z/` (a measurement script +
  render cache + report JSON + 4 PNGs) + this memo. ZERO library/trainer code changed.
- **Forward-path SELF-CHECK (faithful):** the n600 EMA-best (`levelset_witness_ema_BEST.npz`, ep125)
  forwarded through R (numpy int8-deploy fixed-point, so_iters=4) + frozen CPU-torch SegNet on the
  gt_n600 verdict subset (96 pairs, stride 6, spanning 0..570) reproduces **realized d_seg = 0.006842**
  vs the trainer's reported best **0.006771** (Δ 7.1e-5). The measurement renders the SAME bytes the
  verdict does — this is the deploy-faithful authority path, NEVER MPS.

---

## 0. HEADLINE (reframed per operator: NOT a binary; the error + the composition)

The NO-GO's TWO load-bearing claims — **(1)** "the residual is 50–86% INTERIOR (unreachable by any
GT-free annulus)" and **(2)** "unreachable_dseg bottoms out at ≈0.010, a HARD floor independent of INR
capacity" — are **BOTH artifacts of the 0.154 deterministic-warp bulk, and BOTH collapse against the
trained-witness bulk.** Against the real 0.0068 witness the residual is **94.5% boundary-localized
(only 5.5% interior of a GT class edge)**, and a GT-free annulus-override mask (dilate 4) reaches it
with **unreachable_dseg = 0.00111 < the sub-0.15 budget 0.00123 — it PASSES the geometry gate.** So the
geometry ceiling is a property of **bulk QUALITY, not the architecture**: the residual-INR compose is
geometrically viable given a good-enough bulk. The boundary-localization theorem is confirmed
empirically (interior fraction falls monotonically as the bulk improves).

But this is a COMPOSITION, not a win. The residual is dominated by the **LANE long-tail (44% of all
flips; HALF of every lane pixel flips)**, it is high-frequency / per-pair-diverse (residual-field ID
~27–38), and residual-on-the-trained-witness is a d_seg-**improvement** move (MORE bytes), not a
rate-shrink move — the rate-shrink RD question (is there a CHEAP bulk good enough?) is separate and
still open. The pointer is UNMOVED 0.19110 and nothing here moves it; this unit produces the deep
understanding of the real error + the canonical viz + the per-mechanism Δd_seg-per-byte map.

---

## 1. THE MEASURED ERROR STRUCTURE (n600 EMA-best, 96 verdict pairs)

`d_seg = 0.006842` = 129,142 flipped cells / 18,874,368 (96×384×512). SELF-DETECTED class signatures
(area / vertical-centroid) all match the CLAUDE.md canonical comma10k order — NOT hardcoded:

| class (self-detected) | GT area | vert. centroid | disagree/class | **flip-mass share** |
|---|---|---|---|---|
| 0 Road (mid-lower) | 23.3% | 0.62 | 0.83% | **28.2%** |
| **1 Lane (thin, mid)** | 0.58% | 0.59 | **51.98%** | **43.9%** |
| 2 Undrivable (top/sky) | 49.5% | 0.25 | 0.13% | 9.2% |
| 3 Movable (mid-band) | 1.17% | 0.52 | 7.41% | 12.7% |
| 4 MyCar (bottom/hood) | 25.4% | 0.87 | 0.16% | 6.0% |

**The error IS the lane long-tail.** Lane is 0.58% of the frame but 43.9% of all flips, and **half of
every lane pixel flips** (disagree 0.52). Road (28%) and Movable (13%, car edges) follow. The static
hood (MyCar) and sky (Undrivable) are essentially solved (0.16% / 0.13% disagree).

**Per-region — the residual is a codim-1 annulus, NOT interior:**

| region | flips within | interior beyond |
|---|---|---|
| GT boundary annulus r2 | **94.5%** | 5.5% |
| GT boundary annulus r4 | 96.2% | 3.8% |
| GT boundary annulus r8 | 97.5% | 2.5% |

Lane flips are **99.8% inside the r2 GT annulus** (dash-gap / dropped-interior only 0.16%) — the lane
error is thin-structure boundary instability, not big interior fill errors.

**Per-margin — the flips sit exactly on the decision boundary (and are PRIMED):**
- Realized witness margin (logit[GT]−max_other) at flips: mean **−0.62**, median −0.44 (correct cells
  +7.79). The flips are BARELY lost — a small push flips them back (the "primed" state; the θ*/RL memo's
  annulus).
- Cached GT margin at flips: mean 0.54, and **86% of flips have GT margin < 1.0, 64% < 0.5** — the
  flips concentrate on the pixels where the GT SegNet argmax is ITSELF least confident (the small-margin
  codim-1 band). This is exactly the d_seg definition made visible.

**Viz (the point):** `aggregate_flip_density_and_classmass.png` — flips trace the lane lines radiating
from the vanishing point + the horizon + the hood edge; the interior is dark. `multipane_{worst,median,
best}_pair*.png` — GT argmax | witness argmax (near-identical) | disagreement (red = the thin
lane/car-edge annulus) | realized-margin (blue confident everywhere except the thin red flip lines).

---

## 2. THE OVERTURN — bulk QUALITY, not architecture (the boundary-localization theorem, measured)

Same coverage metric as the design-refine (`measure_composition_coverage` over the bulk's OWN argmax
boundaries; interior_frac = 1 − annulus_d2_coverage), swept over bulk quality:

| bulk | bulk d_seg | interior frac (>2px of bulk bdry) | annulus override unreachable_dseg | gate (<0.00123) |
|---|---|---|---|---|
| deterministic warp k*=47 | 0.1543 | **0.855** | 0.132 (union_d4) | NO-GO |
| deterministic warp k*=3 | 0.0276 | 0.496 | 0.0100 (union_d4) | NO-GO |
| **trained witness (n600 EMA-best)** | **0.0068** | **0.224** (annulus_d2) | **0.00111 (annulus_d4) / 0.00071 (d8)** | **PASS** |

Interior fraction falls **monotonically** as the bulk improves (0.855 → 0.496 → 0.224) — the
boundary-localization theorem (perfect bulk → residual concentrates on the codim-1 boundary). Against
the good bulk the GT-free override mask reaches unreachable_dseg **below the sub-0.15 budget** at
dilate ≥4. The design-refine's "≈0.010 HARD floor independent of INR capacity" was a property of the
**0.154 deterministic bulk**, whose residual was dominated by warp-drift on the STATIC ego hood
(**MyCar residual 0.121** in the deterministic bulk vs **0.0016** for the trained witness — the hood
the warp corrupts is exactly what the witness gets right). Claim (1) and claim (2) are overturned as
structural claims.

**Full override sweep against the witness bulk** (`error_structure_report.json`):

| mask | dilate | coverage | unreachable_dseg | override_frac | gate |
|---|---|---|---|---|---|
| boundary_annulus | 2 | 0.776 | 0.00153 | 4.9% | fail |
| boundary_annulus | 4 | 0.838 | **0.00111** | 7.8% | **pass** |
| boundary_annulus | 8 | 0.896 | **0.00071** | 13.2% | **pass** |
| union | 4 | 0.838 | 0.00111 | 8.5% | pass |

---

## 3. THE ADDITIVE FORMULATION + THE HONEST NUANCE (Hole 2 + the residual-rate caveat)

**Additive compose** (`final = bulk_logits + residual_logits`, no mask) has **no unreachable region by
construction** — the mask (and thus the whole "unreachable" ceiling) was a self-imposed constraint of
the OVERRIDE formulation. So the geometry ceiling is doubly gone: (a) even the masked override reaches
below budget against a good bulk, and (b) additive removes the mask entirely.

**BUT the honest caveat (NO-FAKE):** the residual FIELD is high-frequency and per-pair-diverse. TwoNN /
MLE intrinsic dim of the (downsampled, binary) per-pair residual descriptor is **~38 / ~27 (mean 33,
Whitney 2m+1 ≈ 67)** — much HIGHER than the design-refine's 19–21 derived on the deterministic bulk.
(Caveat on the caveat: 96 points in a binary downsampled space → this is a noisy upper-ish proxy, not a
converged ID.) Read: removing the geometry ceiling does NOT make the additive residual automatically
cheap — the residual is thin high-frequency lane/edge detail that varies pair-to-pair. The residual-INR
rate is a real, open, non-trivial question — just not a *geometry* one.

---

## 4. PER-MECHANISM Δd_seg-PER-BYTE COMPOSITION (on the MEASURED error, advisory)

The flips are (a) 94.5% annulus, (b) 44% lane / 28% road / 13% movable, (c) primed (realized margin
−0.62). Each repair mechanism buys a DIFFERENT slice — they compose, they are not either/or:

| mechanism | what it buys on THIS error | byte cost | role in the composition |
|---|---|---|---|
| **θ\* levers** (Muon / directional / HOSC) | pushes the PRIMED flips (margin −0.62, near boundary) over; the primary d_seg engine | **0 (train-time)** | FIRST + FREE; bounded by the annulus/lane surrogate-gradient stall (the slow tail) |
| **deterministic-gen lane** (openpilot polynomial SDF) | the **44% lane flip-mass** IF the raster survives R; upper bound **Δd_seg 0.0030** | **0 (rule-118 free)** | huge FREE lever; R-survival unproven — the key $0 gate |
| **residual-INR override (annulus d4/d8)** | closes the residual the free levers stall on; unreachable **0.0007–0.0011 (PASSES gate)** | INR weights (residual is high-freq → not tiny) | the geometry-VIABLE closer; on the trained witness it is +bytes (d_seg move, not rate move) |
| **residual-INR additive** | no unreachable region at all | INR weights (ID ~27–38) | removes the mask ceiling; same rate caveat |
| **stored sidecar (Yousfi flip #98)** | each stored class = 1 flip; **155 KB (AC) to zero d_seg**, 77 KB to halve | ~1.2–2.6 B/flip | EXPENSIVE at bulk scale (129 K flips); efficient only for a tiny last-mile top-K |
| **quantization (int8→lower)** | int8 costs only **Δd_seg 0.00012** vs fp32; headroom to int5/4 | shrinks the ~82 KB blob | orthogonal RATE lever (the rate-shrink path) |

Store's marginal Δd_seg/byte is 4.4e-8 (AC) — constant but the flip count (129 K) makes it a last-mile
tool, not a bulk closer. The efficient bulk closers are FREE (θ* + deterministic lane); the residual
INR is the geometrically-viable finisher for what they stall on.

---

## 5. HONEST RATE ACCOUNTING (Hole 4)

The witness blob at ep125 is **81,819 B → rate 0.05448**; d_seg 0.0068 → contribution 0.684. With the
stored-pose sidecar assumption (√(10·3.4e-5)=0.0184, a BUDGET not a measured composed d_pose),
**sub-0.19 needs d_seg ≤ 0.00118, sub-0.15 needs d_seg ≤ 0.00077** — i.e. the witness needs ~6–9×
d_seg improvement (from 0.0068). residual-on-the-trained-witness is a d_seg-improvement move that ADDS
bytes; it does NOT shrink rate. The design-refine's store_rate 0.055–0.092 was for a SEPARATE cheap
deterministic keyframe bulk — calling the 82 KB witness blob "free" would double-count. The
rate-SHRINK version of the compose needs a bulk cheaper than the 82 KB witness (quantized witness, or
deterministic + small INR) that is STILL good enough for the residual to stay boundary-localized —
that is the real open RD question, distinct from the (now-overturned) geometry question.

---

## 6. WHAT IS ALIVE / THE REAL (NON-GEOMETRY) BLOCKER / NEXT STEP

- **ALIVE (overturned):** the residual-INR compose is NOT geometry-blocked. Against a good bulk the
  residual is a thin boundary annulus reachable below the sub-0.15 gate; additive removes the mask
  ceiling entirely. The design-refine's NO-GO was correct FOR the 0.154 deterministic bulk and WRONG
  as a structural claim about the architecture.
- **THE REAL BLOCKERS (measured, not geometry):**
  1. **The lane long-tail** — 44% of d_seg, half of every lane pixel, primed but stuck (realized margin
     −0.62). This is the binding residual for EVERY mechanism.
  2. **Residual rate** — the residual field is high-freq (ID ~27–38); an additive/override residual INR
     is not automatically cheap.
  3. **Rate-shrink** — residual-on-witness is +bytes; the rate win needs a cheaper good-enough bulk.
- **NEXT $0 steps (no fire):** (a) measure whether a FREE deterministic openpilot-lane raster SURVIVES
  R + SegNet argmax on the flipped lane cells — that is the 44%-of-d_seg / 0-byte lever and the single
  highest-value $0 gate; (b) the `CONDITIONAL_ON_LANE_PRIOR` residual gauge (Wyner–Ziv X−E[X|Y], Y =
  free centerline) reaches the dropped-thin-lane interior a witness-boundary annulus misses (the 16% gap
  between GT-annulus 94.5% and witness-annulus-d2 77.6%).
- **The pointer moves only** on a byte-closed `upstream/evaluate.py` row (CPU/CUDA, never MPS) < 0.19110.

---

## 7. RECURSIVE ADVERSARIAL REVIEW + means≠ends

- **Forward-path faithfulness:** PASS — d_seg 0.006842 vs 0.006771 (Δ 7e-5) on the exact verdict subset,
  int8-deploy dequant + so_iters=4 (the trainer's own path). Not a proxy.
- **NO-FAKE audit:** every number is measured through R + frozen CPU SegNet on the real gt_n600 cache;
  the residual ID is labeled a noisy upper proxy; the additive "unreachable=0" is a structural (not
  measured-magic) statement; the d_pose 0.0184 is labeled a BUDGET not a measurement; rate double-count
  called out; pointer UNMOVED; no score claim.
- **Assumption-challenge axis:** the NO-GO operated within "the deterministic bulk's residual geometry
  IS the architecture's residual geometry." The measurement FALSIFIED that shared assumption — the
  residual geometry is a function of BULK QUALITY. Surfacing this is the unit's primary value.
- **Live-run safety:** PASS — no trainer/library code touched; a future `--resume-from` on pid 38641
  picks up identical code.
- **means≠ends:** the overturn is a MEANS (it un-blocks a path + maps the error). The END is a lower
  exact score; that requires the lane long-tail closed + a byte-closed exact row. Not narrated as done.

**Artifacts:** `experiments/results/residual_inr_adversarial_overturn_20260630T235910Z/`
(`measure_witness_error_structure.py`, `error_structure_report.json`, `rendered_argmax_margin.npz`,
`aggregate_flip_density_and_classmass.png`, `multipane_{worst,median,best}_pair*.png`, `measure.log`).
