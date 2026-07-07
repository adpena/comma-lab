# Group theory — targeted deep-math pass against the MEASURED anchors (2026-07-07)

**Agent:** deep-math research subagent (report-only; no launches, no trainer/tool edits; live #205
READ-ONLY). Task: operator directive *"is there anything we can learn or implement from group
theory"* — a per-angle adjudication in the format of
`.omx/research/mallat_balle_deepmath_review_20260707.md`, against what we already USE (tac.lie
se(3)/SE(3); store-nothing ξ = quotient by the ego one-parameter subgroup; the #287 dash-comb; the
self-orient directional basis −48%; Muon = Newton-Schulz orthogonalization; #218 class geometry;
#155 quotient codec = the DSL gauge layer; the 8=√64 parabolic finding FEED-08f).

**Sequencing (operator, council draft §23):** this review + its measured probes are T5-crucible
GATE items. Follow-up BUILDS identified here are NOT built — they are enumerated in §G as crucible
candidates with {evidence, predicted value, cost}.

**Label discipline:** MEASURED (ours, artifact cited) · THEIRS (paper cited) · DERIVED · INFERRED ·
SPECULATIVE. Attribution per `[[uniward_attribution_honest_lineage_vcm_at_heart]]`.

**Authority:** $0, advisory, MEANS. Pointer contest-CPU **0.19110 UNMOVED** (a review + two $0
byte probes move no pointer).

---

## The measured anchors reviewed against (verified in-tree before use)

| Anchor | Status | Source |
|---|---|---|
| Store-nothing ξ carrier: canonicalize-to-ground-frame, derive-H PROVEN; ξ delta-residual coder 2714 B beats 3200 B table | MEASURED | #241/#257; DAG FEED-08b |
| #287 dash-comb: max-plus translation comb, phase=ego-ξ; mechanism removes 86% of solid-band dash-gap FP at frozen ep650; render-composite net-negative → in-training A/B owed | MEASURED | DAG FEED-08c |
| Self-orient directional basis −48% d_seg ~0 bytes; = discrete shearlet (#284 Ch.5) / bandlet-class with rule-118-free flow (Mallat review A3) | MEASURED | `curvelet_directional_basis_dseg_reduction_v1` |
| Parabolic ceiling: freq_along 8 = √64 = √freq_across; 25/8 ≈ measured 3.2× along-tangent deficit; ladder probe NOW-ranked | MEASURED config + INFERRED law | FEED-08f; `parabolic_scaling_along_tangent_ceiling_v1` (PENDING) |
| Muon ≈ Stiefel (Newton-Schulz orthogonalization); −32% d_seg vs AdamW; leap/saddle reading #217 | MEASURED + THEIRS (2605.13079) | `[[muon_deep_dive_keep_and_tune]]` |
| Class order [Road, Lane, Undrivable, Movable, MyCar]; measured priors [0.232, 0.0059, 0.495, 0.0124, 0.254]; #218 Menon logit-adjust BUILT (A/B owed) | MEASURED | CLAUDE.md SegNet section; DAG #218 rows |
| #155 quotient codec = the DSL gauge layer (gauge-invariant base = scorer-equivalence quotient; gauge = cheapest fiber representative) | BUILT (framing) | DAG "DEEPEST CLOSURE" row; `witness_dsl/gauge.py` |
| Byte-close coder: per-tensor symmetric int8 + brotli q11; base stream = all params except code (ONE stream), code = second stream | SOURCE-INSPECTED | `tools/levelset_byte_close_and_eval.py:build_levelset_blob`; `lever_b_levelset_generator._int8_symmetric` |
| ep650 mod32cap EMA-best checkpoint: hosc (odd) activation, 4×96 FiLM MLP, no skips, mod-dim 32, code (1200,32) in exact temporal frame order (row = 2·pair+frame) | SOURCE-INSPECTED | ckpt `levelset_n600_witness_mod32cap_20260706T115554Z/levelset_witness_ema_BEST.npz` + trainer `_trunk` |

**New measured artifacts from THIS pass** (gitignored-durable, script beside them):
`experiments/results/group_symmetry_canonicalization_probe_20260707/`
`{measure_weight_symmetry_canonicalization.py, weight_symmetry_canonicalization_measured.json,
code_temporal_delta_measured.json}` — peak RSS < 1 GiB, foreground, READ-ONLY on the checkpoint.

---

## A. Q1 — "Code the fundamental domain, generate the orbit" as THE unifying rate principle

**THEIRS (the formal skeleton):** Kolmogorov structure function / minimal-sufficient-statistic
split (Vereshchagin–Vitányi, cs/0204037; already ledgered): K(x) ≈ K(S) + log|S|. Group-theoretic
specialization (classical MDL-of-orbits reading): when x lies on an orbit G·x₀, describe the
**generator** (the group element data / subgroup structure) + a **fundamental-domain
representative** x₀; the orbit entropy factors as H(orbit) = H(representative) + H(group
coordinates).

**Adjudication vs OURS (this IS already our architecture, now named precisely):** under rule 118
the split acquires a sharper economic form than the classical one —

> **The group ACTION (a generic deterministic algorithm: SE(3) exp, comb rasterizer, homography
> transport) ships as FREE inflate.py code; only the video-derived GROUP COORDINATES + the
> fundamental-domain content are COUNTED.**

- **Store-nothing ξ** = quotient by the ego one-parameter subgroup {exp(tξ)}: the canonical-frame
  scene is the fundamental-domain representative; the counted data is the group coordinate ξ(t)
  (MEASURED: delta-residual 2714 B, FEED-08b; derive-H PROVEN #257).
- **Dash comb** = a **frieze group** quotient, literally: a dashed lane is invariant under the
  discrete translation subgroup along the curve (frieze type p1/"hop"); the comb stores {period,
  phase, duty} = the generator description + one template = the fundamental domain (MEASURED
  mechanism: 86% FP removal, FEED-08c; in-training A/B owed).
- **#155 scorer-quotient — HONEST BOUNDARY:** the scorer-equivalence classes (all witnesses with
  the same argmax through R) are **fibers of a map, NOT group orbits** — no group acts on witness
  space whose orbits are those classes. #155 joins the unification at the QUOTIENT level
  (equivalence classes with a free canonicalizer = the DSL gauge layer), and the group case (ξ,
  comb, weight symmetries) is the sharp sub-case where the quotient has algebraic structure. A
  registered law must be stated at quotient generality or it over-claims.
- **The measured refinement from this pass (the important one):** the principle pays **only when
  the group structure creates LOW-ENTROPY COORDINATES**, not when it merely creates counting
  slack. Two new measurements demonstrate both branches:
  (i) weight-space permutation symmetry is a REAL orbit degeneracy (387 B of slack, §C) but
  canonicalization recovers ≈nothing through brotli (−8 B) — slack without low-entropy
  coordinates; (ii) the code stream's Z₂-parity × flow decomposition (§E) yields a real
  **−3,108 B** because the group-refined coordinates (per-parity temporal deltas) ARE low-entropy.

**Verdict: NOW-as-framing + canonical-equation CANDIDATE** (registration = crucible/maintenance
item, §G-C4): `rule118_orbit_coding_free_action_counted_coords_v1` — for a G-structured feature,
counted rate = H(fundamental domain) + H(video-derived G-coordinates), the action itself free;
admissible savings require the G-coordinates to be lower-entropy than the raw feature. Anchors
already measured: ξ delta_res 2714 B (FEED-08b); code parity-split −3,108 B (§E); weight-perm −8 B
(the negative branch, §C). Completing probe: the owed in-training comb A/B (n600 through-R).

---

## B. Q2 — Equivariance: steerable frames, gauge fields, and the shearlet GROUP

### B1. Is the self-orient basis a discretized steerable frame? (Cohen-Welling / E(2)-CNN)

**THEIRS:** Cohen & Welling, *Group Equivariant CNNs* (ICML 2016, arXiv:1602.07576); Weiler &
Cesa, *General E(2)-Equivariant Steerable CNNs* (NeurIPS 2019, arXiv:1911.08251): constrain
filters to irrep-structured (steerable) bases so feature maps transform predictably under a
GLOBAL group (roto-translations); the machinery's value is SHARING statistical strength across
group poses in a data DISTRIBUTION that has the symmetry. Gauge-equivariant nets (Cohen et al.,
ICML 2019) generalize to LOCAL frames on manifolds.

**Adjudication:** the self-orient basis is a **frame field** — Fourier features expressed in the
local boundary-tangent frame — i.e. gauge/bandlet-class ADAPTED coordinates, not a global
steerable representation. The steerable constraint solves a problem we do not have: we train on
ONE clip against a FROZEN scorer — there is no distribution over group poses to share filters
across, and no generalization requirement that equivariance would buy. The Mallat-review A3
adjudication already settled the economics: adaptive flow with the flow bits FREE under rule 118
(openpilot polynomial) beats any fixed equivariant frame at equal counted bytes. The
architecture-vs-canonicalization dichotomy in the equivariance literature (Kaba et al.,
arXiv:2211.09067, *Equivariance with Learned Canonicalization Functions*) independently validates
our route: canonicalize-to-ground-frame (the store-nothing ξ design) IS the canonicalization
branch, chosen over the equivariant-architecture branch — with the canonicalizer free under
rule 118. **Verdict: NO-with-reason for steerable/equivariant architectures; the citation set is
the value** (Kaba = external validation of the ξ design).

### B2. Does steerability give a cheaper along-curve modulation than the comb?

**DERIVED (small, crisp):** a finite steerable basis exists iff the feature's spectrum under the
group is finitely supported. The dash envelope is a sharp on/off periodic indicator — its Fourier
(harmonic/steerable) description needs O(#harmonics ≳ 25) coefficients per lane and reintroduces
Gibbs (the exact failure the homogenization law measured). Its ORBIT description — one template ×
the discrete translation group, {period, phase, duty} — is O(1). So the steerable literature
CONFIRMS the comb is the minimal generator for this class; it does not offer a cheaper one. This
agrees with and slightly strengthens the scattering adjudication (Mallat review A1: comb =
second-order carrier×envelope term). **Verdict: NOW-as-adjudication FOR the owed comb A/B; no new
build.**

### B3. The shearlet GROUP and the 8=√64 parabolic ceiling

**THEIRS:** the shearlet system is a genuine square-integrable group representation (Dahlke,
Kutyniok, et al.): the shearlet group = parabolic dilations A_a = diag(a, √a) ⋊ shear ⋉
translations. Curvelets are a frame construction WITHOUT a group; shearlets have one. The
α-molecule framework (Grohs–Keiper–Kutyniok–Schäfer; re-checked in LITSWEEP-REPR, still standing)
parametrizes the dilation exponent continuously: α=1/2 = parabolic (cartoon-optimal), other α =
other anisotropy laws.

**Adjudication:** the group view makes the FEED-08f finding STRUCTURAL rather than arithmetic:
within the shearlet group the along/across coupling (along ∝ √across → 8 = √64) is fixed by the
dilation SUBGROUP — no rebalancing inside the group escapes it; escaping = changing the dilation
exponent = leaving the group (per-class α; wave-atom scaling for the dash class). The freq_along
ladder probe (FEED-08f row 2, already NOW-ranked) is therefore precisely **a probe of the
dilation-group exponent α for the lane class** — the group language sharpens what the probe
discriminates but adds no new lever beyond what FEED-08f tabled. **Verdict: WATCH (framing that
sharpens the pending `parabolic_scaling_along_tangent_ceiling_v1`); no new build.**

---

## C. Q3 — Weight-space symmetries as a rate lever: MEASURED NO

**The exact symmetry group of the counted parameterization** (source-inspected: levelset `_trunk`
+ `_compose_rgb`; plain FiLM MLP, no skips; hosc = tanh(β·sin(ωu)) is ODD):

G = S₉₆⁵ (neuron permutations at the 5 hidden activation layers, each acting on incoming
rows+bias, the layer's FiLM scale+shift rows, and outgoing columns) × S₃₂ (mod space: code
columns ↔ film.weight columns) × Z₂⁴⁸⁰ (per-hidden-neuron sign flips, exact because hosc is odd —
flip incoming row+bias+FiLM-SHIFT and outgoing column; FiLM scale untouched) × S₅ (out_sdf rows +
bias + palette rows, exact for the pure render — flagged: NOT with a class-INDEXED consumer such
as lane_band `palette[lane_cls]`). Per-tensor symmetric int8 (scale = max|a|/127) is permutation-
and sign-invariant → quantization commutes with G → canonicalized nets are function-identical
(asserted numerically in the probe; max forward deviation ≤ 1.3e-3 on the 0-255 scale = fp
summation reassociation only; sign flips exactly 0.0).

**MEASURED (this pass, ep650 EMA-best through the REAL deployed coder — int8+brotli-q11,
`build_levelset_blob` accounting):**

| Quantity | Value |
|---|---|
| Baseline base stream (72,695 int8 params) | **61,842 B** brotli (85.1% of raw — near-incompressible) |
| Baseline code stream (38,400 int8) | **20,355 B** brotli |
| Theoretical symmetry slack: 5·log₂(96!) + log₂(32!) + 480 + log₂(5!) | **3,096 bits = 387 B = 0.47%** of the 82,197 B counted streams |
| perm_greedy_nn (best arm: LZ-similarity neuron ordering, all 5 layers) | **−8 B** |
| perm_lex_quant / perm_rownorm | −4 B / −2 B |
| sign_canonical / mod-space S₃₂ greedy / composed | **+72 B / +251 B / +339 B** (hurt) |
| random-group-element control | +32 B |

**Why (DERIVED):** the slack scales as O(H log H) bits while the payload scales as O(H²·8) bits —
at H=96 the ratio is ≈0.5% — and brotli's LZ matching cannot exploit row-adjacency of near-iid
int8 rows (the base stream is already only 15% compressible). The operator-prompt figure
"log₂(96!) ≈ 490 bits/layer" is confirmed (498.3 measured) — and it is negligible AND
unrecoverable through the deployed coder. PR101-L22 CONV4_STORAGE_PERMS is NOT contradicted:
that lever permutes AXIS ORDER of conv tensors (changes spatial locality structure), a different
mechanism from symmetry-orbit canonicalization.

**Verdict: NO as a rate lever (measured, both the cap and the realization).** One genuine WATCH
survives at the fleet horizon: **Git Re-Basin** (Ainsworth–Hayase–Srinivasa, arXiv:2209.04836; +
Entezari et al. arXiv:2110.06296 LMC-modulo-permutation) — permutation ALIGNMENT (not
compression) of witness weights across clips for #211 amortized-meta-init warm-starts/averaging.
Corpus-gated exactly like #211; ledgered.

---

## D. Q4 — The class simplex and S₅

- **MEASURED:** S₅ on (out_sdf rows+bias, palette rows) is an exact render symmetry (probe arm
  `class_S5_flagged`, +5 B — nothing there; slack 6.9 bits). Flag stands: exact only while no
  class-indexed consumer is active (mod32cap deploys lane-band OFF).
- **The useful direction is symmetry BREAKING, and it is already built:** the ETF/#218 target is
  the maximally S₅-symmetric class geometry (neural collapse), but the MEASURED class-frequency
  long tail [0.232, 0.0059, 0.495, 0.0124, 0.254] breaks S₅ hard — and the built #218 Menon
  logit-adjust (τ·log prior; Lane/Movable log-priors −5.13/−4.39) is exactly the group-symmetry-
  breaking correction, aimed at the un-born island mass. THEIRS (supporting): the
  minority-collapse literature (Fang et al., *Exploring Deep Neural Collapse under Class
  Imbalance*, PNAS 2021 lineage) — under imbalance the symmetric ETF is NOT the optimum, the
  prior-corrected geometry is. **Verdict: nothing new; the A/B owed on #218 is unchanged.**

---

## E. Q5 — The ego one-parameter subgroup: one confirmation, one measured surprise, one real find

- **Group-Fourier along the flow = the ξ carrier, already.** For a one-parameter subgroup the
  natural harmonic analysis along the orbit is Fourier/spline in t AFTER canonicalization — which
  is verbatim the store-nothing design (canonical frame + SE(3) B-spline + delta-residual coder).
  Nothing unexploited on the pose side.
- **Dash phase as holonomy (DERIVED, crisp, zero new bytes):** phase(t) = phase(0) +
  (1/period)·∫₀ᵗ |v(s)| ds mod 1 — the holonomy of the ego trajectory against the dash lattice.
  Since ξ(t) is ALREADY counted (2714 B) and the integrator is generic free code, the comb needs
  ONE scalar phase(0) + period + duty per lane; every per-frame phase is generated. This is the
  precise form of the "phase=ego-ξ" plan (dashgap memo's range-dependent structure) — folds into
  the comb design, no new payload.
- **MEASURED SURPRISE — the naive group prediction FAILS, its refinement PAYS:** the prediction
  "code rows are smooth along exp(tξ) → temporal-delta them" is FALSE on ep650: adjacent-row
  cosine **−0.031** vs +0.249 for random pairs; mod-256 temporal delta INFLATES the code stream
  20,355 → 33,883 B (+66%). The TRUE structure is a **Z₂(frame-parity) × flow decomposition**:
  frame0-chain cosine **0.99979** (the 600 frame0 codes are nearly ONE vector), frame1-chain
  0.723, within-pair −0.031. Coherence lives per parity isotype. Exploiting it —
  parity-deinterleave + per-dimension time-delta + column-major, all bijective byte transforms
  with asserted bit-identical decode — compresses the code stream **20,355 → 17,247 B = −3,108 B
  (−15.3%; rate term −0.00207)** through the same brotli-q11. Artifact:
  `code_temporal_delta_measured.json`. Config caveat WELDED ON: ep650 mod32cap is w_pose=0,
  lever-light — the near-constant frame0 chain may be config-specific; the lever must ship as an
  auto-pick coder FAMILY (the LBND4 pattern: try {raw, parity-split-delta-colmajor, …}, pick
  post-brotli COUNTED winner, self-describing magic), duty-to-measure per checkpoint. Deployment
  gated on inlining the decode into `_INFLATE_PY` (LBND4's honest-fail gate). **Crucible
  candidate C1 (§G) — not built here per sequencing.**
- **The design observation behind the coder (INFERRED → crucible C2):** frame0-chain cosine
  0.9998 means ≈half the code payload carries almost no per-pair information — a
  shared-frame0-code (or asymmetric mod-dim) witness design could remove those bytes at the
  SOURCE (raw ceiling ≈ −9.6 KB of the 38.4 KB raw code; realized ceiling ≈ −3-8 KB counted),
  at unknown d_seg cost. And the honest group reading: **the learned chart did not align itself
  with the group action because nothing asked it to** — if temporal coherence in code space is
  wanted (for coding OR for the flicker axis), it must be trained in (a code-smoothness /
  equivariance prior), a next-run design question (crucible C3).

---

## F. Q6 — Rhymes-only calls (recorded so they are never re-derived)

- **Wallpaper/crystallographic groups for the road plane:** the dash pattern is the 1-D FRIEZE
  case and is genuinely used (§A, §E); full 2-D wallpaper symmetry of road texture — RHYMES-ONLY
  (road texture is not 2-D periodic; no second translation generator exists to exploit).
- **Heisenberg group in the parabolic scaling:** parabolic dilations are automorphisms of the
  Heisenberg group and FIO/wave-atom theory lives there — RHYMES-ONLY for us (the actionable
  content is already carried by the α/dilation-exponent reading, §B3).
- **Renormalization (semi)group vs the τ-flow:** already adjudicated in the Mallat review A4 —
  our τ-crossover is Γ-convergence/homogenization, not an RG fixed-point theorem. RHYMES-ONLY.
- **Muon and the gauge (small genuine note, no lever):** Newton-Schulz is a polynomial in
  A·Aᵀ·…·A, hence O(n)×O(m)-equivariant: NS(P·A·Qᵀ) = P·NS(A)·Qᵀ for orthogonal (in particular
  permutation) P, Q. So Muon training COMMUTES with the S₉₆ gauge — the gauge freedom is
  preserved along training, no hidden gauge-fixing lever there; the group fact that matters
  (Muon ≈ Stiefel retraction) is already held by #217.

---

## G. Crucible candidates (enumerated, NOT built — operator sequencing, council draft §23)

| id | candidate | evidence | predicted value | cost |
|---|---|---|---|---|
| C1 | **Code-stream group-decomposition coder** (parity-deinterleave + per-dim time-delta + column-major; auto-pick family per LBND4 pattern; `_INFLATE_PY` inline gate) | MEASURED −3,108 B (−15.3% code stream), decode bit-identical asserted; `code_temporal_delta_measured.json` | rate term −0.00207 on this ckpt; family generalizes per-ckpt via auto-pick | ~half-day build + tests, $0; pure-rate (d_seg/d_pose invariant by construction) |
| C2 | **Shared-frame0-code / asymmetric mod-dim witness design** (remove the near-constant frame0 code rows at the source) | MEASURED frame0-chain cosine 0.99979 (600 rows ≈ 1 vector) | raw ceiling −9.6 KB code, realized ≈ −3-8 KB counted → rate ≤ −0.006 (UPPER bound; d_seg cost UNKNOWN) | trainer design change + a run; config-dependence risk (w_pose=0 artifact?) |
| C3 | **Code temporal-coherence prior** (train the chart to align with exp(tξ): smoothness/equivariance loss on code) | MEASURED incoherence (adjacent cosine −0.03) = the absence it would fix | SPECULATIVE: enables delta coding + possibly helps the flicker axis (44% spikes=LANE) | loss term + A/B; interacts with FiLM capacity |
| C4 | **Equations-leg registration** of `rule118_orbit_coding_free_action_counted_coords_v1` (§A) + `weight_symmetry_orbit_slack_negligible_v1` (§C, the measured negative) | both anchors measured this pass | triality consistency (measured findings must reach the equations leg) | small; NOT done here to avoid colliding with the sibling mid-edit on the registry (git status showed `canonical_equations_registry.jsonl` dirty) — an explicitly OWED maintenance item |
| — | freq_along ladder probe + in-training comb A/B | already tabled by FEED-08f / FEED-08c | — | not duplicated here |

## H. Ranked table

| # | Angle (their exact result where external) | Our measured anchor | Concrete $0 probe / lever | Verdict |
|---|---|---|---|---|
| 1 | Orbit coherence along exp(tξ) refined by frame-parity Z₂ (ours; group-representation decomposition) | Code stream 20,355 B; frame0-chain cos 0.9998; parity-split coder **−3,108 B measured, bit-identical** | Crucible C1 (coder family) + C2 (design) | **NOW** (measured; highest-EV item of this pass) |
| 2 | Orbit-coding MDL: free action / counted coordinates under rule 118 (structure-function specialization) | ξ delta_res 2714 B; comb 86%; the §C negative as the boundary case | Candidate eq `rule118_orbit_coding_free_action_counted_coords_v1` (C4); completing probe = owed comb A/B | **NOW-as-framing** |
| 3 | Steerability minimality: sharp periodic envelope = O(1) orbit description vs O(≥25) harmonics | 3.2× along-tangent deficit; comb = second-order term (Mallat A1) | None new — theory-ranks the ALREADY-OWED in-training comb A/B | **NOW-as-adjudication** |
| 4 | Holonomy phase: phase(t) = phase(0)+∫\|v\|/T mod 1 from the counted ξ | dash PHASE=ego-dist (dashgap memo); ξ carrier 2714 B | Zero new bytes — folds into comb design | **NOW-as-design-note** |
| 5 | Shearlet GROUP: along ∝ √across is fixed by the dilation subgroup; escaping = changing α = leaving the group | 8 = √64 (FEED-08f); ladder probe pending | Sharpens what the pending ladder probe discriminates (α-exponent) | **WATCH** (framing) |
| 6 | Git Re-Basin / LMC-mod-permutation (arXiv:2209.04836 / 2110.06296): align weights modulo S_H across models | #211 amortized meta-init (corpus-gated) | Alignment tool when the witness fleet corpus lands | **WATCH** (corpus-gated) |
| 7 | Weight-space symmetry slack S₉₆⁵×S₃₂×Z₂⁴⁸⁰×S₅ as free bits | **MEASURED: 387 B theoretical cap (0.47%); −8 B realized; controls +32/+72/+251 B** | Probe DONE — `weight_symmetry_canonicalization_measured.json` | **NO** (measured negative, both cap and realization) |
| 8 | E(2)/steerable/gauge-equivariant architectures (Cohen-Welling 1602.07576; Weiler-Cesa 1911.08251; Kaba 2211.09067) | One clip + frozen scorer → no distributional symmetry to share; canonicalization route already ours (store-nothing ξ) | None — Kaba = external validation of the ξ design | **NO-with-reason** (citation value only) |
| 9 | S₅ class symmetry / ETF symmetric geometry | 6.9 bits slack (measured +5 B); priors break S₅; #218 logit-adjust BUILT | Nothing beyond the owed #218 A/B | **NO** (symmetry-breaking already operationalized) |
| 10 | Wallpaper 2-D / Heisenberg / RG-semigroup readings | — | — | **RHYMES-ONLY** (recorded §F) |

**Single highest-EV NOW: row 1** — the code-stream group-decomposition coder: the only item of
this pass with a measured, bit-identical, byte-closed delta (−3,108 B, rate −0.00207), plus the
design fact behind it (frame0 codes ≈ one vector) that the T5 crucible should see before sealing
the next-run config.

## I. What this pass did NOT resolve (honest)

- Whether the frame0-code near-constancy survives a pose-ON / lever-full config (w_pose=0
  lever-light ckpt is the only one measured) — C1's auto-pick family and C2's design both carry
  this as duty-to-measure.
- Whether ANY entropy coder (not brotli) could recover more of the 387 B weight-symmetry slack —
  irrelevant in practice (0.47% cap), recorded so nobody re-opens it for the wrong reason.
- The in-training comb A/B and the freq_along ladder remain the arbiters they were; this pass
  RANKS them (rows 3, 5), it does not run them.
- Equations-leg registration (C4) is OWED — deliberately deferred to avoid colliding with a
  sibling's in-flight edit of the canonical-equations registry.

Sources (verified this pass): Cohen-Welling arXiv:1602.07576 · Weiler-Cesa arXiv:1911.08251 ·
Kaba et al. arXiv:2211.09067 · Ainsworth et al. arXiv:2209.04836 · Entezari et al.
arXiv:2110.06296 · Vereshchagin-Vitányi cs/0204037 (ledgered) · α-molecules/shearlet-group per
LITSWEEP-REPR re-check · Demanet-Ying wave atoms per FEED-08f · in-tree artifacts as cited.
