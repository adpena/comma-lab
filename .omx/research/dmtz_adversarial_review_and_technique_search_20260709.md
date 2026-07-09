# DMTz edit-sidecar rate-lever — fresh-eyes adversarial review + technique search

**Date:** 2026-07-09 · **Reviewer:** fresh-eyes adversary (reviewer ≠ author of the finding under
review; did NOT write commit `2c0d9a8ba`) · **Authority:** `[macOS-CPU advisory / DERIVED +
cross-checked against MEASURED n600 anchors] NON-PROMOTABLE` · **Pointer:** 0.19110 **UNMOVED**
(everything here is MEANS; no score/promotion/kill claim — the pointer moves only through a
byte-closed n600 row via `tools/levelset_byte_close_and_eval.py`).

Under review: memo `.omx/research/dmtz_taskaware_rate_lever_design_20260709.md`, probe
`experiments/probe_ms_edit_sidecar_rate.py`, result `reports/dmtz_probe_n8_20260709.json`. Its
verdict: **DMTz Morse-Smale edit-sidecar rate-lever is DOMINATED** (1.254 B/flip DERIVED vs #307's
0.820 MEASURED vs ~0.65 admit bar).

Every number labeled MEASURED / DERIVED / INFERRED / ASSUMED. I attack my own conclusion in §4.

---

## JOB A — verdict: **CONFIRM-DOMINATED (strengthened)**, at the FORMULATION level

Default posture was to OVERTURN. I could not; the independent MEASURED evidence pushes the verdict
*further* into DOMINATED, not out of it. The decisive fact the original memo under-weighted: **#307
is already the real-witness-through-R measurement, run with a STRONGER coder than DMTz's, and it is
dominated** (`contour_string_flip_coding_n600_20260707.md`: mod32cap ep425 witness, all 600 pairs,
byte-close render + frozen CPU-torch SegNet, every frame decode-verified bit-exact → **0.8201
B/flip**). The five attack angles, each tested:

**1. DERIVED-vs-real (the proxy bias).** ASSESSED, not assumed-blocked. Memory is NOT the wall: the
#307 extraction ran at **RSS flat 2.4–2.7 GiB**, and right now free ≈ 6 GiB + ~51 GiB reclaimable
inactive (the live #205 run holds only 16.3 GiB RSS), and a durable NON-live witness snapshot
exists (`contour_string_flip_coding_n600_20260707/snapshot_ema_BEST.npz`, sha-pinned). So a real
witness-diff probe is *feasible* and touches nothing of #205. **But it is UNNECESSARY** — #307
already IS that measurement, with a better coder, dominated. Direction of the proxy bias: the
DERIVED small-|margin| flip set is the *thinnest* error → the WORST case for a coder → the probe
BIASES B/flip UP; a real (thicker) witness residual could code *lower*. That is the honest overturn
risk — and it is settled the wrong way for DMTz: the real best-coder number (0.820, contour-string)
is itself already dominated, and DMTz-LZMA-raster is a *weaker* coder than the contour-string
chain-code, so a real-witness DMTz run lands **≥ 0.820**. No overturn. (I deliberately did NOT
launch the heavy witness probe — it is already answered and would only re-confirm.)

**2. Paradigm vs implementation (#307 classification).** The probe codes an "edit-diff raster +
LZMA" — which is NOT DMTz's actual method (C-Loop/S-Loop tiered edits with magnitude quantization
`δ = ξ(1−1/2ᵐ)`, 5 preservation tiers). So the negative is not a test of DMTz-verbatim. **However,
the DMTz paradigm is INAPPLICABLE to our object by construction, not merely untested:** DMTz
compresses a **continuous scalar field** and its entire gain is storing *coarse edit magnitudes*
(few bits for `m`) instead of full residuals. Our field is a **categorical 5-class argmax** — there
is no continuous magnitude to quantize; an "edit" is a class flip, and its only content is *which
class* (already `log2(K−1)` in the probe). The `δ=ξ(1−1/2ᵐ)` tier machinery has no referent here.
The *only* transferable content is "code the disagreement boundary, not the whole partition" — which
the probe DID test, and which is dominated. Classification: **IMPLEMENTATION-dominated for the
transferable projection; paradigm INAPPLICABLE (not a premature paradigm-kill — the scalar-field
method simply does not map onto a categorical partition).**

**3. Wrong operating point / rare-class lane niche.** This was the most promising overturn lead and
it fails cleanly. The DMTz memo §6 preserved a "lane-marking residual sidecar" niche (echoing #180).
Two MEASURED reasons it does not rescue the storage lever: **(a)** the lane residual is the *worst*
confetti — #307's coherence decomposition shows 44.6% of components are singletons, mean size
3.1 px, and thin lane *dashes* ARE those singletons; a lane-only edit sidecar codes *worse* than the
global 0.820, not better. **(b)** the lane d_seg problem is ALREADY better served on a different
axis: the **openpilot analytic lane band** (render-time FREE generator, rule-118, MEASURED d_seg
**0.00087** for the lane class — memory L71/L72) restores the lane separatrix by *generating* it, not
by *storing* edits. The niche is real but it is a **DISTORTION (render) problem, not a rate/storage
problem** — which is exactly the DMTz memo's own §10 conclusion ("aim at d_seg via the render, not at
storing the error"). Reinforces DOMINATED for the edit-sidecar.

**4. Axis conflation ("make the error thinner" answers rate with distortion).** Valid *caveat*, not
an overturn. A rate lever on the residual DOES matter independently in principle (a residual d_seg
always remains). Empirically it is walled: at the current geometry every measured coder exceeds the
bar. The correct scoping — already present in the memo's `verdict_scope` — is that the negative is
**operating-point-specific**: it holds at the current *confetti* residual. This is the one honest
door left open (see §4 self-attack).

**5. Edit-count vs flip-count admit model.** The sharpest angle: DMTz's premise is that ONE
topology-critical edit corrects a CLUSTER, so bytes-per-edit can exceed 0.65 while bytes-per-flip
stays under it. **MEASURED-refuted by #307's component decomposition.** The contour-string coder
already codes connected *components* (= one edit per boundary segment spanning multiple flips) and
still floors at 0.820 because the components are tiny: mean 3.1 px, and the **per-component anchor
cost alone is 0.37 B/flip**. There is no coherent topological structure to amortize — the flips are
scattered boundary jitter, not a few large correctable features. The edit-count model gives no
leverage on *this* residual. #307's own conclusion: reaching 0.65 needs mean component size ≈ 3×
larger — **"a TRAINING outcome, not a coder trick."**

**Bar-value note (honest discrepancy, does not change the verdict):** the memo derives the admit bar
as `σ_eff·λ* ≈ 0.51·1.2731 = 0.65`. The canonical equation
`waterfill_annulus_through_r_store_vs_capacity_20260701.py` measures a through-R realization
efficiency `η_R ≈ 0.35` (only ~35% of an idealized correction survives R), which would put the bar
at **~0.45**. These are related-but-distinct efficiencies (σ_eff = fraction of stored edits that
flip the *scored* argmax; η_R = fraction of the correction *magnitude* realized through R). The bar
is somewhere in **0.45–0.65**; DMTz-LZMA (1.254) and contour-string (0.820) exceed BOTH. The
discrepancy is worth reconciling but is immaterial to this verdict.

### Job-A decisive reason (one line)
The real witness residual through R is confetti (mean 3.1 px components, 44.6% singletons); #307
already measured it at 0.820 B/flip with a *stronger* coder than DMTz's, above the 0.45–0.65 bar,
and the edit-count "one edit fixes a cluster" leverage is measured-absent — so storing the error is
rate-walled by residual *geometry*, and the fix is render-side coherence, not a better coder.

### verdict_scope
**DOMINATED at the FORMULATION level** (DMTz-scalar-field edit tiers are INAPPLICABLE to a
categorical argmax; the transferable "code the disagreement boundary" projection is IMPLEMENTATION-
dominated). The MS/separatrix *paradigm* is not killed. One narrow reactivation path survives and is
**operating-point-conditional**: if a coherence-inducing training change (the #301/#274 island arms,
or a lane-band-composed loss) makes the residual's mean component size ≈3× larger, the edit-count
storage model should be RE-MEASURED. That is a TRAINING result, not a ready lever. **Do NOT spend
n600/paid budget on any edit/flip-storage coder at the current residual geometry** — this now
includes the memo's own §9.1 "owed real witness-diff probe," which #307 has substantially
superseded.

---

## JOB B — technique queue for OUR frozen information space (ranked)

Frozen space: SegNet 5-class argmax partition (512×384) + PoseNet 6-vec on the exact 600 pairs;
~97% of d_seg on a ~4.7%-area separatrix annulus; counted bytes = witness payload; rule-118 splits
FREE generic-generator from COUNTED learned/video-derived payload. External SOTA scan (VCM /
coding-for-machines / label-map compression, July 2026) surfaced EGIC (seg-guided *RGB* compression,
ECCV'24), ICM-with-SAM edge base layers (ICIP'24), scalable base+enhancement bitstreams,
GroupedMixer / checkerboard / ELIC context entropy models, TransVFC/FCM feature codecs. **Every
learned codec in that literature ships weights → rule-118 COUNTED and large → BLOCKED as a carrier;**
only deterministic *principles* transfer. Ranked by P(beats the 0.45–0.65 admit bar OR moves d_seg)
× blast-radius:

**#1 — Render-side d_seg coherence lever (analytic-lane-band + coherence-inducing residual loss).
[HIGHEST EV × HIGHEST blast-radius; existing, not new.]** The honest answer to "optimal for our
frozen space" is NOT a storage codec — it is making the residual *thinner and more coherent* so
d_seg drops directly (removing the rate need) AND, as a bonus, satisfying #307's "3× coherence"
condition that would re-open every storage model above. The openpilot analytic lane band already
MEASURES d_seg 0.00087 on the lane class as a rule-118-FREE generator. **rule-118:** band generator
FREE; the tiny per-frame lane-trajectory coords COUNTED (~hundreds of bytes, already in
`xi_pose_coder`/#148). **Composition:** it IS the #301/#274/island arms + the analytic-band-composed
training the mod32cap baseline deliberately omitted. **$0 go/no-go bar:** measure, on the n600 GT
cache, the mean-component-size and d_seg of `witness ⊕ analytic-lane-band` vs witness-alone (the
#307 harness already computes both surfaces); GO to a render-integration arm iff band composition
drops lane-class d_seg or raises mean component size toward 3×. (Render-integration is then a TRAINING
arm, not a probe — governed, not $0.)

**#2 — Hand-built context-model entropy coder on the WHOLE argmax label map (checkerboard /
2nd-order Markov / MRF-CRF boundary context). [LOW P (~0.1) × medium blast.]** The seg-map-compression
SOTA principle, stripped of learned weights: replace `contour_codec`'s ~H0 boundary coding with a
FREE deterministic context model (a learned entropy model would be rule-118 BLOCKED). **rule-118:**
hand-coded context = FREE algorithm; the argmax bits COUNTED. **Composition:** competes with
`contour_codec` (886 B/frame) and the witness. **Why low P:** #180 already measured the whole-
partition path RD-optimal at S≈0.37 (rate is the wall); even a 1.5–2× context-coding win lands
S≈0.22–0.27, still DOMINATED by 0.19110, because the witness amortizes smooth structure into shared
FREE weights while any per-frame partition codec re-specs every frame. **$0 go/no-go bar:** build a
checkerboard/2nd-order-Markov range coder on the n600 GT argmax; GO only iff < ~450 B/frame (≈ half
`contour_codec`) — and even then cross-check against #180's S≈0.37 wall before any n600 spend.

**#3 — Through-R-quotiented sufficient-statistic codec (#155 fiber-quotient extension). [LOW P
(~0.1) × medium blast.]** Code only the argmax-relevant sufficient statistic (boundary + class),
interior = free constant fill, AND quotient the boundary by the R-operator equivalence class (store
only distinctions that survive uint8/resize/parse-back — the σ_eff/η_R idea applied at the codec, not
the flip). **rule-118:** quotient map FREE; the R-invariant boundary descriptor COUNTED.
**Composition:** #155 fiber-quotient + #157 reverse-waterfill; it is the principled *floor* of the
whole storage direction. **Why low P:** #180 already measured the geometric floor (d_seg 5.57e-4 at
444 KB → S 0.37) — the quotient can only *shrink* that, and the witness's amortization advantage is
structural. **$0 go/no-go bar:** measure the through-R-quotiented boundary entropy on n600 GT; GO
iff it beats the witness's per-pair amortized rate at equal d_seg (it almost certainly does not).

**Not-a-lever (recorded so it is not re-litigated):** learned VCM/FCM feature codecs, EGIC,
GroupedMixer/ELIC *learned* entropy models, Task-Aware deep encoder-control — all ship COUNTED
learned weights (rule-118 BLOCKED). DMTz continuous-scalar-field edit tiers — INAPPLICABLE to a
categorical argmax (Job-A angle 2).

---

## Highest-EV next $0 probe (single)
**None of the storage probes.** The single highest-EV $0 next step is to record this DOMINATED
verdict in the DAG (so DMTz/edit-sidecar is not re-litigated) and **route energy to the render-side
coherence arm (#1 above)** — where the frozen-space wall actually yields. If forced to name one $0
*measurement*: extend the #307 harness to report, per class, mean-component-size and d_seg for
`witness ⊕ analytic-lane-band`, giving a pre-registered coherence gate for the island/#301/#274
training arms (this both aims at d_seg and defines the exact condition that would re-open the storage
model — a two-birds probe). Storage-side (#2/#3) are cheap $0 confirmations at best; #180 already
walls them.

---

## §4 — Adversarial self-review (attack my own CONFIRM)
- **"Am I over-trusting #307's 0.820 as the real number?"** #307 is MEASURED n600, byte-close render
  + frozen CPU-torch SegNet, every frame decode-verified bit-exact, on a durable sha-pinned
  snapshot — the strongest evidence class we have short of an exact eval. It is a *coder-rate*
  measurement (not a score), correctly tagged NON-PROMOTABLE. I am trusting it for what it is: the
  real residual's codability with the best coder we have. That is exactly the load-bearing quantity.
- **"Could a materially better coder than contour-string exist?"** The floor decomposition says no
  at this geometry: anchors alone cost 0.37 B/flip over 142k components; even a *free* chain+class
  stream leaves you at 0.37, and the bar is 0.45–0.65. Halving the anchor cost (unlikely on 3-px
  islands) still only reaches ~0.5–0.6 — marginal, and #180's whole-partition S≈0.37 wall subsumes
  it. The residual has to change, not the coder.
- **"Is the operating-point escape real or a fig leaf?"** Real and MEASURED-conditioned: #307
  names the exact quantity (mean component size ≈3×). That is a genuine reactivation criterion, not
  a hand-wave — and it belongs to the *render/training* program, which is #1 in the Job-B queue. So
  the escape door and the highest-EV direction are the same door. Consistent.
- **"Fake?"** No score claim; pointer UNMOVED. I ran only $0 work (probe selftest = OK; memory/proc
  inspection; SOTA scan) and explicitly declined the heavy witness probe as already-answered. The
  verdict is a NEGATIVE (the most no-fake-honest outcome), scoped to FORMULATION with a named
  MEASURED reactivation condition.

## Triality
- **DAG:** owed FEED row — DMTz edit-sidecar DOMINATED (fresh-eyes CONFIRM, strengthened by #307
  real-witness 0.820 + edit-count-model measured-absent); Job-B queue #1 render-coherence lever is
  the routed direction. (This memo is the durable artifact meanwhile.)
- **DSL:** no new `Lever` (correct — dominated; wiring a dominated lever is orphan-generating). If
  the #1 analytic-band-composition arm measures a coherence/d_seg win, THAT render lever lands as a
  factory — never this diff-LZMA.
- **equations:** no NEW equation for an unmeasured candidate. The governing law is the EXISTING
  `waterfill_annulus_through_r_store_vs_capacity_20260701` (λ*=1.2731; recovery-adjusted 0.45–0.65).
  The #307 0.820 anchor + this review's edit-count-model-absent finding are corroborating
  `EmpiricalAnchor`s that could be appended to that equation's anchor list — recorded as owed, not
  force-added here (no measurement of a new candidate → no new equation).
