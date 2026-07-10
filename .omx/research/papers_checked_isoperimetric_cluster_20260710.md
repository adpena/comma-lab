# Papers-checked: isoperimetric cluster (Cartan–Hadamard + 2 arXiv links) — boundary-length / σ_cc′ theme

Date: 2026-07-10 · operator-supplied 3-link cluster · anti-re-research ledger (sister of
`papers_checked_stac_sparc_taskaware_compression_20260709`). STORES CONSULTED: MEMORY.md L55
papers-checked line · σ_cc′ per-class surface tension (#382) · contour-coding rate floor 1–1.5
bits/px (#307) · annulus law (~97% d_seg in ~4.7% annulus, #333) · Fisher-metric ↔ (−margin)
Pearson 0.978 (margin = Fisher surrogate) · Γ-limit/area-Lagrange derivation · MCF thin-lane
erasure (`dash_erasure_homogenization_v1`) · V2 originality memo (L16). Operator note: the prior
link 2311.11032 was an ID-mismatch typo — **two of these three are the same failure mode.**

## 1. arXiv 2607.06870 — OUT OF DOMAIN (ID likely a typo)
Verified from abs page: **"Phase transitions and uberholography of holographic pure-state
geometries"** — Ning Bao, Keiichiro Furuya, Jacob March. **High-energy / holography (hep-th)**:
AdS₃/CFT₂, Ryu–Takayanagi geodesics, entanglement-wedge connected/disconnected transition
(cross-ratio threshold η'/η = e^{ΔH/2}), holographic error-correcting codes, recursive
hole-punching (uberholography) with universal fractal dimension α ≈ 0.786.

**Verdict — OUT OF DOMAIN / note-only** (verdict_scope: whole paper). No isoperimetric bound, no
boundary-coding rate, no segmentation/level-set geometry. RT surfaces are minimal (area-min)
surfaces, but the paper is about entanglement-wedge phase transitions + code price/distance, not
perimeter-vs-area comparison. The α ≈ 0.786 fractal dimension is a numerical coincidence with
various α's in our notes (drift-vs-depth, holography-fractal) — **no method connection**; do not
mine it. Almost certainly the operator meant an isoperimetric/geometry preprint and the ID is a
typo, exactly like 2311.11032.

## 2. arXiv 2602.00797 — OUT OF DOMAIN for the isoperimetric theme (ID likely a typo)
Verified from abs page: **"Zero-Flow Encoders"** — Yakun Wang, Leyang Wang, Song Liu, Taiji
Suzuki (stat.ML/cs.LG; ICML 2026). **Rectified-flow representation learning**: the "zero-flow
criterion" — a rectified flow under independent coupling is zero everywhere at t=0.5 iff source =
target; used to certify conditional independence, extract sufficient statistics, learn amortized
Markov blankets + SSL latents via a simulation-free loss.

**Verdict — OUT OF DOMAIN for isoperimetry; weak-WATCH aside** (verdict_scope: whole paper for the
cluster theme). Nothing about perimeter, curvature, isoperimetric bounds, or boundary geometry —
not what the operator's Cartan–Hadamard framing points at, so it too reads as an ID mismatch.
The ONE tangential thread, recorded so it is not re-researched: the zero-flow / conditional-
independence criterion is a *sufficiency* certificate, and our witness stores a ~8-dim
sufficient statistic that amortizes the argmax — in principle a zero-flow test could certify
"payload ⟂ (everything the scorer ignores) | argmax partition." That is a generative-rep-learning
tool on a DIFFERENT axis from d_seg/rate; **not a lever, not on the isoperimetric theme.** WATCH
only if a sufficiency-certification sub-problem is ever opened. V2 untouched (generative rep
learning, not driving-recon-with-warp × codecs-for-machines).

## 3. Cartan–Hadamard conjecture (Wikipedia) — the actual isoperimetric content
Statement: in a Cartan–Hadamard manifold (complete, simply-connected, **sectional curvature ≤ 0**,
i.e. CAT(0)/Hadamard), the classical isoperimetric inequality holds *at least as strongly as
Euclidean* — for a given enclosed volume, minimal perimeter ≥ the Euclidean-sphere perimeter of the
same volume; sharp constant = the Euclidean one. Proven dims 2 (Weil), 3 (Kleiner), 4 (Croke); open
in general; a 2026 local result shows perturbing a Euclidean metric toward nonpositive curvature
raises the isoperimetric ratio.

**Verdict — CONFIRMS σ_cc′ metric-weighted-perimeter framing + ONE conditional GRAIN**
(verdict_scope: the perimeter-functional formulation of d_seg; nothing of ours killed, no
reformulation owed):

- **CONFIRMS (b):** the mathematically-canonical object for d_seg boundary cost is a *metric-
  weighted perimeter functional*, i.e. the σ_cc′-weighted (per-class-pair) boundary length measured
  in the **Fisher metric**, NOT the flat-pixel perimeter. This is already the #382 σ_cc′ / length-
  term program (Fisher ↔ −margin, 0.978). Isoperimetry names the right functional; it does not
  hand us a new one. Nothing to build.

- **THE GRAIN → #382 (σ_cc′) length-term calibration:** Cartan–Hadamard is a comparison theorem —
  in a **nonpositively-curved** metric the perimeter for a fixed enclosed area is *≥* the flat
  perimeter. Our Fisher metric is anisotropic and curvature ↔ (−margin); IF the boundary-annulus
  Fisher metric is (locally) NPC, then **flat-pixel boundary length is a LOWER bound on the true
  Fisher-metric d_seg boundary cost** — i.e. a flat-perimeter length regularizer *under-weights*
  the thin high-curvature (small-margin = Lane) boundary, the same erasure direction as MCF
  homogenization. Consumable: when the σ_cc′ length term / eikonal-length regularizer is set, weight
  it by the local Fisher/(−margin) factor (not unit pixel length), and treat the flat-length as a
  floor. This is a *sanity/calibration* grain, CONDITIONAL on a curvature-sign measurement we have
  NOT made (whether the annulus Fisher metric is NPC is untested — measure sign of sectional
  curvature of the margin-induced metric on a boundary patch before trusting the inequality
  direction). Recorded as a #382 task note, not a launched lever.

- **NEGATIVE on (a) — no rate floor** (verdict_scope: this formulation, not the paradigm):
  isoperimetry bounds *perimeter vs enclosed area*, NOT *bits vs boundary-length*. It gives no
  bits/boundary-length floor. The boundary-coding rate floor stays with contour/chain-code entropy
  (#307, 1–1.5 bits/px; ~8 bits/flip) — a different quantity. Do not re-open isoperimetry as a rate
  bound.

- **NEGATIVE on (c) — no direct consumable:** the conjecture says nothing operational about the
  ~8-dim Whitney/mod-dim embedding or about level-set/eikonal/viscosity flows in CAT(0) spaces
  (MCF-in-Hadamard theory exists elsewhere but is not in this source; not fetched, not banked).

## Cluster synthesis — what was the operator likely pointing at
The operator's instinct is correct and already ours: **the witness d_seg energy is a
boundary-length (perimeter) functional, and the mathematically-canonical form of it is a
curvature/Fisher-metric-weighted perimeter — which is exactly σ_cc′ (#382) + the length/eikonal
term.** The isoperimetric / Cartan–Hadamard lens *confirms the functional we already have* and adds
one conditional calibration grain (Fisher-weighted, not flat-pixel, length; flat length is a floor
if NPC). It does **not** yield a rate floor (that stays #307) and does **not** move the pointer.
The two arXiv IDs (2607.06870 hep-th holography; 2602.00797 Zero-Flow Encoders rep-learning) are
**both off-theme — the same ID-typo failure mode the operator already flagged for 2311.11032**; the
only on-theme source in the cluster is the Cartan–Hadamard page itself. Recommend the operator
re-supply the intended isoperimetric-geometry preprint IDs if a specific result was in mind
(candidate real work in this space: sharp isoperimetry / weighted-perimeter Γ-limits / anisotropic
mean-curvature flow — none fetched here).

## Disposition
- Ledger: this memo is the anti-re-research bank for the cluster. Proposed MEMORY.md L55 hook (do
  NOT edit MEMORY.md): `isoperim-cluster: Cartan-Hadamard CONFIRMS σ_cc′=Fisher-weighted-perimeter
  (#382) + GRAIN Fisher-not-flat length, flat=floor-IF-NPC (curvature-sign untested); NO rate floor
  (stays #307); 2607.06870=hep-th + 2602.00797=Zero-Flow-Encoders BOTH off-theme ID-typos`.
- #382 grain recorded above (task note): Fisher/(−margin)-weighted length term + curvature-sign
  measurement precondition. No new equation (no measured row of ours — prior-art/theory confirmation
  only; the metric-weighted-perimeter framing is already registered via σ_cc′).
- DAG: FEED-paper-isoperim appended (grain landed, conditional).
- means≠ends: pointer 0.19110 UNMOVED; anti-re-research + one conditional calibration grain only.
