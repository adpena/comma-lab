# DMTz × Task-Aware Encoder Control → one contest rate-lever (design + $0 probe)

**Date:** 2026-07-09 · **Authority:** `[macOS-CPU advisory / DERIVED-from-GT-margin] NON-PROMOTABLE`
· **Pointer:** 0.19110 **UNMOVED** (this memo produces an UNMEASURED research candidate, not a
score claim) · **Probe:** `experiments/probe_ms_edit_sidecar_rate.py` · **Probe JSON:**
`reports/dmtz_probe_n8_20260709.json`

Every number below is labeled **MEASURED** / **DERIVED** / **INFERRED** / **ASSUMED**. No score,
frontier, promotion, or kill claim: the pointer moves only through a byte-closed n600 row via
`tools/levelset_byte_close_and_eval.py` on contest-CPU/CUDA.

---

## 1. The two papers

- **(A) DMTz** — *Multi-Tier Preservation of Discrete Morse-Smale Complexes in Error-Bounded Lossy
  Compression* (arXiv 2409.17346). Compress a scalar field lossily (SZ3/ZFP base), then apply
  **error-bounded EDITS** that restore the Morse-Smale complex: critical points via C-Loops,
  separatrix connectivity via S-Loops, edit quantization `δᵢ = ξ(1−1/2ᵐ)` storing only the integer
  `m`. Five preservation tiers; up to 6× over SZ3.
- **(B) Task-Aware Encoder Control** — *for Deep Video Compression* (arXiv 2404.04848, CVPR'24).
  VCM: encode video **for a machine task**, one frozen decoder + encoder-side control (mode
  prediction + GoP selection) → ~25% bitrate at equal task accuracy.

## 2. The thesis (both → one lever)

Our d_seg lives on the SegNet argmax **separatrix** (the class boundary). **MEASURED (prior,
`#333`/annulus):** ~97% of d_seg sits in a ~4.7%-area annulus of small-margin pixels — boundary
JITTER, not region miss. **DERIVED (prior, `#284`):** τ→0 the witness IS a discrete Morse-Smale
complex; the separatrix is its 1-skeleton. The counted archive bytes ARE the witness payload.

The synthesis is a **Morse-Smale error-bounded EDIT SIDECAR**:

> The witness INR generates a near-correct argmax **FREE** (rule-118, from the counted weights+code).
> A **COUNTED** sidecar codes only the **argmax-restoring DIFF** between the free witness partition
> and GT, restricted to the flip-prone annulus (task-aware allocation) and amortized temporally
> across pairs (GoP). This is DMTz's "edit a lossy base to restore the MS complex" mapped onto our
> witness-as-base: we do **not** re-code the whole partition (`contour_codec` already does that) —
> we code the **edit**, which is cheaper **iff the free base is already mostly right** (it is:
> d_seg≈0.005).

The distinction over our existing `tac.boundary_math.contour_codec` (which codes the *whole*
partition boundary at ~boundary-entropy via LZMA): DMTz codes the **disagreement boundary only**.
The disagreement boundary ⊆ full boundary, so in principle much cheaper — **if** the disagreement
is codable. §5 measures whether it is.

## 3. The break-even is the whole game — and it is ALREADY the canonical KKT water level

The lever is a pure rate-distortion trade. Storing one flip-restoring edit:
- **removes** `100 / N_total` score points of d_seg (`N_total = 600 frames × 384×512 = 1.180e8`),
- **adds** `bytes × 25 / 37_545_489` score points of rate.

Setting benefit = cost:

```
break-even bytes/flip = (100 / N_total) / (25 / 37_545_489) = 8.477e-7 / 6.659e-7 = 1.2731
```

**DERIVED, VERIFIED numerically — but NOT novel.** Proactive recall: this is the **exact `#157`
KKT reverse-waterfill water level λ*** already canonical in
`src/tac/canonical_equations/waterfill_annulus_through_r_store_vs_capacity_20260701.py`
(`WATER_LEVEL_BYTES_PER_FLIP = 1.2731`), and it equals the ~1.27 B/flip floor cited in
`src/tac/boundary_math/contour_codec.py`. I re-derived a known result; I cite it.

**CRITICAL correction (recovery-through-R shrinks the real bar to ~0.65 B/flip).** The 1.273 is the
*rate-only* break-even. But a stored flip must survive uint8/resize/parse-back through **R** to
actually be admitted — only a fraction `σ_eff ≈ 0.51` of stored edits change the *scored* argmax.
So the **effective admit bar is `σ_eff × 1.2731 ≈ 0.65 B/flip`** (MEASURED, `#280` Lever-D
economics). **This is the real pre-registered bar, not 1.273.** The best existing flip coder
(`#307` contour-string) already achieves **0.820 B/flip at full n600, bit-exact** — below 1.273 yet
**still NO-GO** against the 0.65 recovery bar. Any new coder must beat **0.65**, and must beat the
existing 0.820, to matter.

## 4. FREE vs COUNTED split (rule-118 firewall — binding)

| Component | rule-118 class | Why |
|---|---|---|
| witness INR forward (the base partition) | **FREE** | generic algorithm in inflate.py; the INR weights it runs are already counted |
| generic **edit-decode** algorithm (overlay `diff!=0 → target class`; DMTz C/S-Loop replay) | **FREE** | deterministic generic decoder, no video-derived table |
| curvelet/self-orient bank | **FREE** | parametric, GT-free (existing) |
| **the edit set itself** (which pixels flip → which class; the `m` integers) | **COUNTED** | irreducible video-derived residual — the diff depends on GT argmax |
| Task-Aware deep codec **weights** (mode-predictor, GoP net) | **rule-118 BLOCKED — do NOT ship** | a LEARNED video codec; its weights would be COUNTED and are large. We extract only the **principle**, never the codec. |

**The Task-Aware paper is rule-118-poison as a codec** (its whole method is learned encoder-side
nets). The only transferable thing is the **principle**: (i) allocate boundary bits by *task*
saliency (spend on pixels whose flip actually changes the scored argmax through R, skip the rest),
and (ii) **GoP selection** — code the diff on keyframe pairs and propagate, rather than per-pair.
Both are deterministic allocation policies we implement ourselves; neither ships a learned weight.

## 5. The $0 probe and its n8 result (MEASURED)

`experiments/probe_ms_edit_sidecar_rate.py`. The codability of the diff is a property of the
SegNet-argmax **boundary geometry**, fully available in the GT cache (`lstars`=GT argmax,
`margins`=GT margin field) — **no witness decode, no SegNet, ~18 MB, memory-safe alongside the live
#205 run.** The probe DERIVES a principled flip set = the `density`-fraction smallest-|margin|
pixels (our established flip locus), calibrated to a target operating d_seg, and measures **bytes
per corrected flip** for four coders: **(F)** sparse-residual floor `log2 C(N,k)+k·log2(K−1)`;
**(E)** DMTz-style raster edit-map + LZMA (exploits clustering, on the *diff*); **(T)** E +
temporal-delta (naive GoP proxy); **(P)** `contour_codec` whole-partition reference.

**MEASURED (n8 of gt_n24, `--sweep`, `reports/dmtz_probe_n8_20260709.json`):**

| density (d_seg) | flips | floor B/flip | **edit B/flip** | edit+GoP B/flip | area/perim | verdict |
|---|---|---|---|---|---|---|
| 0.0010 | 1576 | 1.672 | 3.063 | 4.530 | 0.276 | DOMINATED |
| 0.0025 | 3936 | 1.509 | 1.904 | 2.890 | 0.306 | DOMINATED |
| **0.0050** | 7864 | 1.385 | **1.254** | 1.865 | 0.355 | **GO (marginal)** |
| 0.0100 | 15728 | 1.259 | 0.755 | 1.122 | 0.444 | GO |
| 0.0200 | 31456 | 1.134 | 0.411 | 0.795 | 0.795 | GO |

Full-partition `contour_codec` reference = **886 bytes/frame** (whole boundary).

**Honest reading of the measurement (against the REAL 0.65 B/flip recovery bar, §3):**
1. **At our operating d_seg ≈ 0.005 the DMTz edit-diff-LZMA codes at 1.254 B/flip — DOMINATED.**
   It is worse than the existing `#307` contour-string coder (**0.820 B/flip at n600**), and both
   are far above the 0.65 recovery-adjusted admit bar. The naive LZMA edit map does not even reach
   the existing coder, let alone the bar. My earlier framing "marginal GO vs 1.273" was the WRONG
   bar (rate-only); against 0.65 it is a clear **DOMINATED**.
2. **Strongly density-dependent.** DOMINATED at/below ~0.005 (thin, scattered → `area/perim`
   0.28–0.35 → boundary ≈ 3× its own area → nothing to exploit); only reaches sub-0.65 at d_seg
   0.02 (0.411 B/flip). **This confirms the adversarial thesis and matches `#180`'s measured n600
   result: a thin jitter annulus is the worst case; only thick 2-D error pays** — and our error IS
   thin jitter (`#180`: partition drift is high-frequency boundary jitter, only 3.3% pose-subsumable).
3. **The naive temporal (GoP) proxy HURTS** (1.865 > 1.254 at 0.005). **MEASURED negative,
   corroborated by `#180`:** a raw frame-delta of two sparse edit maps ~doubles the nonzero support
   when flip sets don't align — and `#180` measured the same at n600 (DP vertices only 80.8%
   temporally coherent; temporal 1026 B/fr > 740 independent). A real GoP needs
   **motion-compensated** propagation (ξ-warp the previous diff), which is the hard owed piece (§9).
4. **Coding the diff does NOT beat coding the whole partition here:** 1.254 B/flip × ~983
   flips/frame ≈ 1233 B/frame > 886 B/frame `contour_codec` whole-partition. The "free base saves
   you" premise is unrealized by the LZMA diff.

**INFERRED (conservative):** the DERIVED small-margin flip set is the *thinnest* possible error, so
the probe is a lower bound on codability favorability. But even granting a real witness a somewhat
thicker residual, the existing measured n600 anchors (`#180` 444 KB @ d_seg 5.57e-4 → S≈0.37
DOMINATED; `#307` 0.820 B/flip NO-GO vs 0.65) already settle it: **flip/edit storage is rate-walled
on our error profile.**

## 6. Composition with existing lanes

- **`#180` Morse-Smale partition codec (MEASURED, `morse_smale_partition_codec_feasibility_20260626.md`):**
  the whole-partition sibling is already measured DOMINATED — 444 KB @ d_seg 5.57e-4 → S≈0.37, and
  temporal does NOT collapse it (DP verts 80.8% coherent; temporal 1026 > 740 B/fr independent). It
  loses to the witness because the witness amortizes smooth structure into shared FREE generator
  weights while MS re-specs every frame. **Its one surviving salvage role — the only live niche for
  this whole memo — is a `#180`-identified rare-class (lane-marking) residual sidecar on the
  witness, NOT a primary rate carrier.** `#311` TropNNC / `#284` supply the tropical/Laguerre view;
  `#311` is a weight-reduction lever (Δd_seg=0 through-R), a different axis, modest ΔS.
- **`#157` KKT / reverse-waterfill bit allocation:** the task-aware allocation IS a reverse-waterfill
  — allocate `m`-bits per boundary vertex by its margin-saliency (through R), spending only where
  the flip is scored. The probe's density sweep is the crude version; `#157` is the principled
  allocator.
- **`#155` level-set / fiber quotient codec:** the fiber quotient is the natural home for "code the
  boundary, interior is a free constant fill" — the edit map's 0-sentinel is exactly the quotient's
  interior class.
- **`#148`/`#241` keyframe temporal / store-nothing:** provide the ego-ξ (already coded, `xi_pose_coder`)
  to motion-compensate the GoP diff propagation (§9 owed).
- **`contour_codec` / `context_partition_codec`:** the existing whole-partition coders are the
  reference this lever must beat (886 B/frame). The edit-only framing is the delta.

## 7. Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| separatrix extraction | **ADOPT_CANONICAL** (`#180`/`contour_codec` boundary machinery) | already reversible + tested; no reason to fork |
| edit-map / overlay decode | **FORK_PRINCIPLED (thin)** | the *diff* framing (0-sentinel edit map) is new vs whole-partition; small, self-contained |
| bit allocation | **ADOPT_CANONICAL (`#157` reverse-waterfill)** | task-aware = margin-saliency waterfill; do not reinvent |
| temporal amortization | **FORK_PRINCIPLED** | motion-compensated diff via existing `xi_pose_coder` ξ — owed, not built |
| Task-Aware deep codec | **REJECT (rule-118)** | learned weights are COUNTED; principle only |

## 8. Pre-registered go/no-go bar (corrected to the recovery-adjusted admit level)

**Bar = 0.65 B/flip** (`σ_eff × 1.2731`, the recovery-through-R admit level, §3), AND must beat the
existing `#307` contour-string **0.820 B/flip** it competes with. **GO** (promote to a real
witness-diff n600 measurement) **iff** a coder — necessarily the motion-compensated GoP (§9.2) plus
task-aware allocation (§9.3), since spatial-only is already known ~0.82–0.90 B/flip — achieves
**< 0.65 B/flip on the REAL witness diff through R**. Otherwise **DOMINATED**.

**Current verdict from the n8 probe + settled n600 anchors: DOMINATED.** The raw DMTz-edit-diff is
1.254 B/flip (worse than the existing 0.820), and both exceed 0.65. Do NOT spend n600/paid budget
on this formulation. The only thing that could flip it is a motion-compensated temporal coder
crossing 0.65 — and `#180`/`#280` both measured the temporal headwind (drift is jitter, ≤3.3%
pose-subsumable), so that is a low-probability, high-effort bet, not a ready lever.

## 9. Owed follow-ups (named blockers — NOT run here)

1. **Real witness-diff probe (governed n600 follow-up):** run `levelset_byte_close_and_eval.py`
   decode to get a real witness argmax vs GT through R, feed that flip set (not the DERIVED
   small-margin proxy) into the probe core. **BLOCKER:** needs the full byte-close decode (heavy;
   the live #205 run holds most RAM — must wait / sequence). Do not launch while #205 trains.
2. **Motion-compensated GoP:** replace the naive temporal delta with a ξ-warped previous-diff
   propagation using the already-coded ego-ξ (`xi_pose_coder`). The n8 negative (§5.3) says the
   naive version is worthless; the MC version is the real test of the GoP principle.
3. **Task-aware allocation through R:** wire `#157` reverse-waterfill on margin-saliency so only
   flips that survive uint8/resize/parse-back are stored — shrinks the effective flip count to the
   clustered high-value subset (the regime where §5 shows the coder wins).

## 10. Adversarial self-review (round 1)

- **"Is this just Lever-D / `#307` / `#157` renamed?"** Largely, YES — and I nearly missed it.
  Proactive recall showed the break-even IS `#157`'s KKT λ*, the existing `#307` contour-string
  already codes flips at 0.820 B/flip (better than my edit-diff), and `#180` already measured the
  whole-partition path DOMINATED. The *only* genuinely new content is the **diff-vs-whole-partition
  framing** — and the n8 probe shows it does NOT help at our density. Honest verdict: **DOMINATED**,
  not a new lever.
- **"Where does it fail?"** Confirmed MEASURED + corroborated at n600: thin jitter annulus (our
  exact profile) is the worst case; naive GoP hurts; the diff doesn't beat whole-partition; and the
  real bar is 0.65 (recovery), which nothing here reaches. Four failure modes.
- **"Rule-118 leak?"** The Task-Aware codec is BLOCKED (§4); only the deterministic allocation
  principle is taken. No learned weight ships. The edit set is honestly COUNTED.
- **"Fake?"** No score claim; pointer UNMOVED; the n8 number is a DERIVED-structure lower bound
  explicitly labeled, not a witness measurement. The verdict is a NEGATIVE (DOMINATED), which is the
  most no-fake-honest outcome.

**verdict_scope:** DOMINATED at the **FORMULATION** level — the DMTz-edit-diff + naive-GoP coding of
the flip set is beaten by the existing `#307` contour-string (0.820 B/flip) and by the whole-memo
rate wall (real bar 0.65 B/flip; settled `#180` S≈0.37, `#307` NO-GO). The MS/separatrix
*paradigm* is NOT killed: one narrow live niche survives (the `#180` rare-class residual sidecar),
and one low-probability reactivation path remains (a motion-compensated temporal coder crossing
0.65). One failed formulation ≠ family dead — but this formulation is not worth n600/paid budget.
**Net: this unit produced an honest NEGATIVE + a lower-bound structure probe; the pointer stays at
0.19110 and the next unit should aim at d_seg via the render (make the error thinner), NOT at
storing the error.**

## 11. Triality (a DOMINATED formulation lands NO new surface — by design)

Because the verdict is **DOMINATED**, this unit deliberately does **not** add a DSL `Lever` (wiring
a dominated lever is orphan-generating anti-signal). The triality legs are:
- **DAG:** append a FEED row recording the DOMINATED verdict + the n8 anchor so the negative is not
  re-litigated (owed; this memo is the durable artifact meanwhile).
- **DSL:** no new lever (correct — dominated). If the owed §9.2 MC-GoP later crosses 0.65, THAT
  lever (not this diff-LZMA) lands as a factory.
- **equations:** no NEW equation — the governing law is the EXISTING
  `waterfill_annulus_through_r_store_vs_capacity_20260701` (λ* = 1.2731; recovery-adjusted 0.65).
  This memo's contribution is a corroborating `EmpiricalAnchor` (n8 edit-diff 1.254 B/flip > bar)
  that could be appended to that equation's anchor list — recorded as owed, not force-added here.

---

## Observability surface

*(OBSERVABILITY-ADDENDUM 2026-08-25 — APPEND-ONLY per Catalog #110/#113. This
section is an INDEX into this memo's own content per Catalog #305's 6 facets;
it adds no new claim. Facets with no counterpart in this memo say so plainly.)*

1. **Per-layer inspection** — §2 "The thesis (both -> one lever)" and §4 "FREE vs COUNTED split (rule-118 firewall — binding)" separate the lever into its free (in-`inflate.py`) and counted (in-`archive.zip`) layers, each inspectable on its own side of the firewall.
2. **Per-signal decomposition** — §3 "The break-even is the whole game — and it is ALREADY the canonical KKT water level" decomposes the lever's value against the water level; §5 reports the probe's per-term result.
3. **Run-to-run diff** — §5 "The $0 probe and its n8 result (MEASURED)" is a runnable probe (`experiments/probe_ms_edit_sidecar_rate.py`, with `--sweep`), so re-running reproduces a comparable row.
4. **Post-hoc query** — `reports/dmtz_probe_n8_20260709.json` is the retained probe output; `tools/levelset_byte_close_and_eval.py` is the byte-close path; `src/tac/boundary_math/contour_codec.py` is the codec surface.
5. **Cite-chain** — §7 "Canonical-vs-unique decision per layer" and §9 "Owed follow-ups (named blockers — NOT run here)" carry the decision and debt chain; §11 "Triality" records that a DOMINATED formulation lands no new surface by design.
6. **Counterfactual hooks** — §8 "Pre-registered go/no-go bar (corrected to the recovery-adjusted admit level)" is the pre-registered falsifier; §10 "Adversarial self-review (round 1)" is the hostile counterfactual pass; §6 "Composition with existing lanes" is the with/without-composition axis.

**Scope honesty:** §5's result is n8, not n600. It is a probe row, not a family verdict.
