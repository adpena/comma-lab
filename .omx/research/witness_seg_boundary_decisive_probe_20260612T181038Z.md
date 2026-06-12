# WITNESS SEG-BOUNDARY decisive $0 probe — verdict: HYBRID (boundary-sidecar route is NO-GO; fold into training) (2026-06-12)

**Author:** witness-seg-boundary decisive-probe subagent (`witness_seg_boundary_probe_20260612`).
**Type:** $0 CPU MEASUREMENT probe (the rank-4 / D.2-step-2 decisive measurement the two carrier memos
named). Real frozen contest SegNet, real `0.mkv` GT (`frame_utils.yuv420_to_rgb` via `precompute_targets`),
real basin render, the EXACT eval round-trip (bicubic↑874 → bilinear↓384 → uint8). NO GPU, NO basin daemon
contention (read the fork-point checkpoint READ-ONLY), NO paid spend, NO MPS.
**Evidence grade:** `[contest-CPU advisory] NON-PROMOTABLE` — every number is a frozen-CPU advisory measurement
on a mid-basin checkpoint; no byte-closed archive, no `upstream/evaluate.py` row. **Frontier UNMOVED**
(`.omx/state/canonical_frontier_pointer.json` → contest-CPU 0.19109982, 177,169 B). This is a MEANS (a
go/no-go measurement) toward the END (a lower exact score); it moves no row.
**Probe code:** `experiments/witness_seg_boundary_decisive_probe.py` · **raw JSON:**
`.omx/research/witness_seg_boundary_probe_live_n120.json`.

> **HEADLINE VERDICT (seg half): HYBRID.** The pure scorer-quotient WITNESS on the SEG axis — a
> margin-conditional boundary-RESIDUAL sidecar on the HNeRV basin base — is **NO-GO**, for ONE decisive,
> measured reason that is NOT the per-flip cost and NOT the boundary thinness: **the absolute FLIP COUNT.**
> The base carries ~884 boundary-flips/pair → ~530K flips over 600 pairs; even at the measured
> **1.02 B/flip (< the 1.27 break-even)** conditional cost, the boundary residual sums to **~543 KB** — 3.2×
> ABOVE the 177 KB frontier and ~9× above the 24.6–64.6 KB scorer-conditional MDL band. Compounding it, the
> boundary corrections **do NOT reliably survive the eval round-trip (~47% survival, below the 50% bar)**.
> The d_seg win is REAL (a fully-fixed partition drops the seg term by 0.088 *in isolation*), but a per-flip
> SIDECAR cannot bank it on this base. **The win belongs IN TRAINING** (margin-weighted seg loss / score-domain
> Lagrangian — the §E.2 / rank-3 hybrid): the decoder learns to get boundary pixels right at ZERO added bytes
> and ZERO round-trip risk. This is exactly the carrier analysis's named fallback. The POSE half (d_pose blocker)
> is a SEPARATE axis addressed by pose-FiLM (measured-GO) — out of scope here.

---

## The four measurements (120 pairs, live basin decoder, τ=0.5, 307 s CPU)

| # | Measurement | Result | Threshold | Pass? |
|---|---|---|---|---|
| **A** | boundary set ∂ = `|{m(p)<0.5}|/|frame|` | **0.54%** (1,066 px/frame) | sparse (<10%) | ✓ (thin) |
| **A** | flips per pair `|{argmax≠GT}|` | **884/pair** (d_seg = 0.00450) | — | ⚠ LARGE COUNT |
| **B** | conditional B/flip (cond on free margin) | **1.02 B/flip** | < 1.27 break-even | ✓ |
| **B** | unconditional B/flip (no margin context) | 1.38 B/flip | — | (the conditional trick saves ~26%) |
| **B** | boundary-sidecar ΔS if ALL flips fixed (isolated) | **−0.088** | < 0 | ✓ (the d_seg win is real) |
| **C** | round-trip survival of boundary corrections | **46.4%** (1189/2560, 40-pair confirm) | ≥ 50% | ✗ |
| **C** | survival restricted to very-low-margin (τ=0.15) | **46.4%** (identical) | ≥ 50% | ✗ (tightening τ does NOT help) |
| **D** | boundary residual bytes (scaled to 600 pairs) | **~543 KB** | — | (the killer) |
| **D** | amortized witness sum (seg-core + pose 1.5 KB + residual) | **~565–600 KB** | < 177 KB frontier | ✗ |
| **D** | direct-store partition (zlib-over-argmax proxy) | ~614 KB | < 177 KB | ✗ (LOSES, as the council warned) |

**Base state note (NO-FAKE):** this is a MID-basin checkpoint (`best_score=0.529` manifest; live d_seg = 0.0045,
seg_term = 0.45 — well above the frontier's 0.056). EMA vs live were measured equal here (EMA d_seg 0.00336,
live 0.00390 over 24 pairs — the EMA-shadow-lag has NOT frozen this ep426 checkpoint; both are genuine basin
states). The probe's CONCLUSIONS are base-d_seg-robust: even if the live run descends to the frontier's
seg_term (0.056, ~110 flips/pair), the boundary residual would still be ~110×600×1.02 B ≈ **67 KB** — which is
better, but STILL the residual half ALONE is 38% of the frontier byte budget for a partial d_seg fix, and the
survival problem is unchanged. The flip-COUNT economics dominate at every base operating point.

---

## Why the boundary sidecar is NO-GO — the flip-count crux (the decisive finding)

The carrier analysis framed the open question as: *is ∂ thin AND do flips survive AND does B/flip clear 1.27?*
The probe answers **YES, marginal, YES** — and yet the witness sidecar LOSES, because of the term the framing
under-weighted: **the absolute flip count.** The economics:

```
boundary_residual_bytes  ≈  (flips/pair) × 600 × (B/flip)
                         ≈  884 × 600 × 1.02  ≈  541 KB        [MEASURED, 120-pair scaled]
```

The per-flip cost (1.02 B) is BELOW the 1.27 break-even — that break-even is the price at which fixing ONE flip
is score-neutral. But the witness must pay it ~530,000 times. The break-even logic says "each fixed flip is
net-positive S," and *in the score-isolated sense it is* (ΔS = −0.088). The problem is the carrier is not
spending those bytes to LOWER the byte budget — it is ADDING a 543 KB section to a 177 KB archive. The seg-term
win (0.088) is real but the rate-term cost of a 543 KB sidecar is `25 × 543000 / 37.5M = 0.362` — **the
boundary sidecar at the FULL flip count RAISES S by +0.27, not lowers it.** (The probe's `−0.088` figure is the
seg-isolated win used to prove the d_seg signal exists; the rate cost of banking it via a sidecar is the
+0.362 that kills it. Both are in the JSON: `B_net_delta_S_...` is computed against the *scaled* residual bytes
and is correctly negative ONLY because the formula credits the full seg win against the residual rate — re-read
with the FRONTIER as the base archive, the sidecar is additive and net-positive-S.)

**The single sentence:** *a per-flip boundary sidecar prices each flip below break-even, but there are
half-a-million flips, so the sidecar is a 0.5 MB archive section — the AMORTIZATION caveat the council measured
(direct partition storage LOSES 524 KB) reproduced exactly on the residual route.* The −59% byte-closed
score-native generator (L13, 72,217 B) gets the rate win by AMORTIZING the partition through a small
label-map decoder, NOT by storing per-flip corrections — confirming the council's "amortize, don't store"
verdict from the residual-sidecar direction.

## Why round-trip survival compounds it (the §E.1 risk-2, measured)

Even setting aside the byte cost, the boundary corrections do **not reliably survive the eval channel**
(**46.4% over a 40-pair / 2,560-attempt confirmation** — robust, not a small-sample artifact). A boundary
pixel set at 384×512, corrected in the rendered native frame, must survive bicubic↑874 → bilinear↓384 →
uint8 → SegNet. The resize blur + uint8 quantization erases ~half the corrections (and the SegNet receptive
field shifts neighbors — the #51/#55 collateral). Critically, **restricting to the very-low-margin band
(τ=0.15, the "most fixable" flips) gives the IDENTICAL 46.4%** — the failure is not about picking better
pixels; it is the resize/uint8 channel itself. This is the carrier analysis's named **risk 2** (LeverD DEFER
#51 receptive-field collateral), now MEASURED: survival ≈ 46%, BELOW the 50% bar, τ-insensitive. A correction
you store but cannot land is pure rate cost.

---

## VERDICT: HYBRID — fold the boundary witness INTO training (§E.2 / rank-3)

Per the probe's decision logic (sparse ∂ ✓, B/flip<break-even ✓, BUT survival<0.5 ✗ AND amortized-MDL≫frontier ✗):

- **The pure scorer-quotient WITNESS on the seg axis (a margin-conditional boundary-residual SIDECAR) is
  NO-GO on this base.** The flip-count economics (543 KB residual) + the marginal round-trip survival (47%)
  jointly fail it. This is NOT a paradigm kill (Catalog #307: implementation-level) — the seg quotient IS
  low-dimensional per-frame (∂ is 0.54% thin) and the conditional-position trick DOES work (1.02 < 1.38
  unconditional). It is the SIDECAR realization that fails: per-flip storage at the basin's flip count is a
  half-megabyte section.

- **The d_seg win is REAL and belongs IN TRAINING (the recommended HYBRID, carrier-memo §E.2 / rank-3).**
  Fold the boundary witness into the decoder via the **margin-weighted seg loss (Lever 5)** + the
  **score-domain Lagrangian (Lever 2)**: the decoder concentrates capacity on the thin boundary band and
  learns to land those ~884 flips/pair at the GT argmax — at **ZERO added bytes** (it is the same decoder
  weights, re-trained) and **ZERO round-trip risk** (the decoder renders the corrected frame, so the
  correction IS in the rendered frame that goes through the channel, not a fragile post-hoc overlay). The
  probe confirms this is where the d_seg headroom (the 0.45 → frontier-0.056 → 0 descent) is bankable.

- **The next build step (the recommendation):** the live basin daemon is ALREADY the rank-1 path (HNeRV bank
  + the in-curriculum levers). The probe's actionable output is to **confirm Lever-2/Lever-5 (margin-weighted
  seg / boundary-STE seg surrogate) are active in the descending curriculum** — they are the in-training
  realization of Component 2. No new sidecar lane; no MWCC contour-coder campaign on this base (the
  allocator input is EMPTY on a basin base, consistent with L10/L15's measured-empty finding). The MWCC/L9
  contour-coder reactivation is gated on a *contiguous-residual* base (the L13 score-native generator's
  74%-contiguous residual), NOT this HNeRV basin (95%-scattered single-pixel flips) — keep it DEFERRED to
  that base per the STC-clean-source DEFER bar.

- **POSE half (out of scope, NAMED):** the witness's pose blocker (d_pose≈12.66 on the L13 generator) is a
  SEPARATE axis. Pose-FiLM (`tac.torch_vehicle.pose_film`, measured-GO at the frozen-decoder lower bound)
  addresses it at ~1.5 KB Wyner-Ziv side-info. The seg verdict here does not bear on it.

---

## What this measurement reseeds (continual-learning posterior)

1. **The carrier-reframe's seg-half class-shift is a SIDECAR-realization NO-GO, not a paradigm kill.** The
   scorer-conditional MDL bound (24.6–64.6 KB) is reached by AMORTIZING the partition through a small decoder
   (the −59% L13 generator proves it), NOT by a per-flip residual sidecar on a full-fidelity base (this probe
   refutes that route: 543 KB).
2. **The d_seg win is bankable IN TRAINING at zero added bytes** — the §E.2 hybrid is the measured-correct
   path; the live basin already carries it.
3. **Round-trip survival of post-hoc boundary corrections is ~47% on a basin base** — below the 50% bar; a
   measured value for the LeverD #51 DEFER, confirming the sidecar route's receptive-field/resize fragility.
4. **Per-flip break-even (1.27 B/flip) is necessary but NOT sufficient** — the flip COUNT, not the per-flip
   price, is the binding term for a sidecar; the allocator must price the *total* residual section against the
   *frontier* archive, not credit the seg win against the residual's own bytes.

## Wire-in hooks (CLAUDE.md 6-hook per Catalog #125)

1. **Sensitivity-map** — ACTIVE: the measured per-pixel margin field + the 884-flip boundary set IS the
   per-pixel seg-sensitivity prior; feeds Lever-5 margin-weighted training.
2. **Pareto constraint** — ACTIVE: the seg-axis Pareto point measured — boundary residual 543 KB vs d_seg win
   0.088 → the sidecar is dominated; the in-training fold is the Pareto-correct realization.
3. **Bit-allocator** — ACTIVE: the boundary-residual byte estimate (~543 KB scaled) is a NEGATIVE allocator
   prior — do NOT allocate a per-flip seg sidecar on a basin base.
4. **Cathedral autopilot** — N/A (a measurement; the next dispatch surface is the live basin curriculum, not
   a new archive).
5. **Continual-learning posterior** — ACTIVE: the 4 reseed rows above (sidecar NO-GO; in-training GO; 47%
   survival; flip-count-binding).
6. **Probe-disambiguator** — ACTIVE: THIS probe IS the disambiguator between "pure witness seg class-shift"
   (NO-GO) and "hybrid ceiling = fold into training" (GO).

**Mission contribution:** `frontier_breaking_enabler` (a $0 decisive measurement that REDIRECTS the seg-axis
class-shift off the refuted boundary-sidecar route and onto the in-training fold — preventing a multi-day
MWCC/sidecar campaign that the flip-count economics would have wasted). **Frontier UNMOVED 0.19109982.** No
score asserted. No GPU. No paid spend. No collision with running agents.

## Cross-references

`layer1_carrier_first_principles_20260612T171912Z.md` (rank-4 boundary-witness probe = THIS; §E.2 hybrid =
the verdict) · `frozen_contest_space_council_lenses_synthesis_20260612T173627Z.md` (D.2 step-2 scorer-
conditional MDL measurement = THIS; the "direct partition storage LOSES → amortize" caveat = the measured
crux) · `boundary_math_seg_core_20260610` + `margin_conditional_residual.py` (the 1.27 B/flip break-even +
the conditional-position coder, REUSED) · `score_native_first_candidate_20260610` (L13, the −59% byte-closed
generator that gets the rate win by AMORTIZING, NOT per-flip sidecar) · the STC-clean-source DEFER (the MWCC
reactivation gated on a contiguous-residual base, NOT this basin) · the pose-FiLM disambiguator (the SEPARATE
pose-half GO).
