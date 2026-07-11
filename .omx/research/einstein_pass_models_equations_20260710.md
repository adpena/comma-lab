# EINSTEIN PASS over the models + equations — the physical picture, the reframe, four derived laws, ranked recommendations

**Agent:** Fable deep-math (Einstein pass) · **Date:** 2026-07-10 · **Cost:** $0, CPU-only, NO scorer
forward, cached anchors + exact arithmetic only; live run pid 88030 untouched. **Pointer 0.19108282
UNMOVED** — everything here is MEANS (laws that sharpen the model; they move no score until a
derived-law-driven arm is byte-closed through `upstream/evaluate.py`). Honesty labels throughout:
**MEASURED / DERIVED / CONJECTURED**, scope-laddered. Per `docs/operating_manual_craft_handoff.md`:
answer first, every number re-derived from primary artifacts, own-round-1 review at bottom.

**Operator question:** *"If Einstein — granddaddy of information geometry and differential geometry —
looked at this, what would he suggest we inquire more deeply into or is missing from the system, our
models and equations?"* Coordinator addenda: physical picture first; find the inquiry NOT asked;
beauty/simplicity as clue; the principle that FORCES the complexity; cash everything out as ranked,
executable rows.

**STORES CONSULTED:** flicker memo `flicker_transform_geometry_term_design_20260710.md` ·
`intake_fisher_is_loglik_curvature_at_argmax_20260710.md` ·
`intake_nielsen_many_faces_infogeom_chentsov_wasserstein_20260710.md` ·
`intake_curved_bregman_and_geometric_structures_cluster_20260710.md` · L10 GR-action memory
(`project_gr_unified_action_full_witness_architecture_20260629.md`) · the 281-equation registry
(`tools/list_canonical_equations.py --json`, full dump reviewed) · DAG FEED-flicker-term /
-flicker-reconcile / -360-force3-Wₑ / -phase-curriculum-costate · L65/L66/L67/L68/L75/L-v8 ·
`clickpolish_exact_gated_ratchet_20260710.py` (the pointer-move row).

---

## §0 THE PHYSICAL PICTURE (the gedankenexperiment — read this first)

Einstein's elevator: the freely-falling observer feels no gravity; what looked like a force was the
frame. Our elevator: **ride the ego-screw ξ.** In the ξ-comoving ground frame, the scene is MEASURED
near-static — best pixel shift (0,0) on **94.8%** of scored transitions (mean |shift| 0.053 px),
ground-plane boundary motion **97.5% coherent** [MEASURED, flicker memo §2.2 + `692489e3b`]. The
temporal flicker that looks like noise in the pixel frame is revealed, in the comoving frame, as a
**static boundary being swept by a moving measurement grid** — the fixed 291/128 resampling comb of
upstream's antialias-False bilinear-D (`modules.py:113`), phase-advancing one full cycle per camera-px
of boundary motion [MEASURED §2.1]. The flicker is not in the world. It is in the frame.

**The pixel lattice is our aether.** The whole d_seg apparatus — GT labels included — is the
ground-frame scene measured through a lattice in relative motion. GT itself carries the frame
artifact: the stride-2 spike rate 0.005318 IS the lattice-phase content of the labels [MEASURED,
flicker §2.4]. And the witness, trained by hard-CE to those labels, converged to exactly the
temporal-majority floor those artifacts define (0.00496–0.0052 ≈ 0.00532) [MEASURED §2.5].

So the equivalence-principle statement for this system, which everything below unpacks:

> **The physical content of the witness is one ξ-comoving ground-frame scene plus one trajectory ξ
> plus the deterministic measurement operator (projection ∘ sampling ∘ R). Everything pair-indexed
> is either holonomy of ξ, gauge (lattice-phase), or a genuine scene event.**

---

## §1 THE REFRAME — the inquiry that wasn't asked (coordinator charge #2)

**GENERAL COVARIANCE OF THE WITNESS (the forcing principle, coordinator charge #4):**

> *All pair-dependence of the witness must factor through (ξ, measurement operator). Any pair-indexed
> parameter NOT so factored is either a genuine scene-change event (movables, appearance events — to
> be stored as quantized reaction events) or wasted rate.*

Status: **DERIVED-as-principle, MEASURED-in-parts, CONJECTURED-as-totality.** The measured parts:
cross-pair partition variation is globally rank-8 with **95.6%** ego-coherent [MEASURED, FEED-is];
the store-nothing pose carrier collapsed 697,941 B → 1,049 B by storing ξ instead of frames
[MEASURED, `store_nothing_pose_carrier_rate_collapse_vs_dpose_v1`]; slot-relabeling discontinuities
(lane swap/birth/death) defeat every temporal model [MEASURED,
`index_permutation_discontinuity_defeats_temporal_model_v1`] — those ARE the reaction events the
principle predicts are irreducible-by-transport. The conjectured part: that the per-pair FiLM code's
residual after regressing out (ξ, measurement) is ONLY events + appearance — testable at $0 (Table B
row B1).

**What the principle FORCES (derived-not-chosen, in hindsight):** canonicalize-to-ground-frame + the
stored ξ (the covariant ontology) · the stratified per-class pose-warp (FEED-ja) · the store-nothing
pose carrier · **T1 phase_advection_consistency** (the gauge channel's transport equation — see §3.3)
· the event-sidecar treatment of movables/lane-births · per-pair code dimension collapsing toward
dim se(3)=6 + zero-modes. Every one of these was found by measurement, one at a time, over weeks.
The principle generates them all at once — that is its value: the NEXT lever should be derived from
it, not re-discovered.

**The hidden simplicity (beauty as clue, coordinator charge #3).** The three "incommensurable" score
terms are all FIRST-ORDER HOMOGENEOUS in natural displacement variables:
- d_seg = area of the symmetric difference of partitions = for small normal displacements
  ∫|δn| dℓ over the separatrix — **linear in boundary displacement** (a flat/TV/W1-type norm on the
  partition) [DERIVED; consistent with the measured "flips are uniform-ΔS",
  `resize_exploit_flip_fix_frontier_v1`];
- √(10·d_pose) = √10 · RMS twist error — **linear in ‖δξ‖** [DERIVED: MSE under a square root is an
  L2 norm];
- rate — **linear in bytes** by definition.

So S ≈ a weighted NORM on the error triple (section displacement, connection displacement,
description length) of one geometric object: a connection ξ on the time-axis with a ground-frame
section (the scene), rendered through a fixed gauge. **d_pose tests the connection; d_seg tests the
parallel-transported section rendered in the gauge; rate is the MDL of both.** The ugliness of
"three terms, two metrics" was the clue: it is one fibered object measured three ways. [DERIVED at
first order; the concavity crossover of the pose term (`score_marginal_lagrange_multipliers_v1`) is
exactly where the linearization is re-centered.]

**And the quantization (the "spectral lines"):** the seg and rate axes are ATOMIC. One pixel-pair
flip = ΔS 100/(600·196608) = 8.4771e-7 exactly; one byte = 25/37545489 = 6.6586e-7 exactly
[DERIVED-EXACT from `evaluate.py:96` + `modules.py:113`]. Their ratio is the system's fine-structure
constant: **1 flip = 1.27311 bytes.** Cross-check: the 2026-07-10 pointer move (ΔS = −1.7e-5 at
ΔB=0) is ≈ **20.05 flip quanta** — twenty pixel-pair flips, consistent with integer quantization
within the reported-component rounding [MEASURED vs DERIVED, registered as
`score_atomic_flip_byte_exchange_v1`]. Every carrier/sidecar decision reduces to: does it fix more
than 0.785 flips per byte? (The registered Lever-D 0.65 B/flip GO threshold is this exchange with a
~2× safety margin — now DERIVED, no longer a tuned constant.)

---

## §2 INQUIRY 1 — COVARIANCE-CLASS TAXONOMY (proposal + seed tagging)

The flicker result proved d_seg = covariant-boundary-error + gauge-grid-phase. Generalizing across
the registry, every equation is one of FOUR classes:

| class | definition | invariance | examples (seed tags) |
|---|---|---|---|
| **COVARIANT_LAW** | identity/theorem, frame-free | lattice, clip, monotone-logit gauge, sufficient-statistic reduction (Chentsov) | `fisher_curvature_equals_categorical_fisher_trace_caustic_v1` (exact identity) · `maslov_dequantization_bound_v1` · `independent_flicker_jitter_dseg_floor_smooth_optimal_v1` · `argmax_of_sdf_is_additively_weighted_power_diagram_v1` · `task_rd_dominates_reconstruction_rd_v1` · `multiphase_modica_mortola_perimeter_gamma_limit_v1` · `mcf_minority_erasure_inevitability_v1` · NEW: `dseg_covariant_gauge_decomposition_v1`, `island_topological_charge_conservation_v1` |
| **SCORER_FRAME_STRUCTURAL** | exact/structural property of the frozen apparatus (score law constants, scorer architecture, R chain); transfers across videos, NOT across apparatus | video/clip | `costate_lambda_marginal_ds_v1` · `contest_r_operator_mtf_allpass_to_2px_v1` · `segnet_stem_nyquist_alias_wall_v1` · `posenet_luma_chroma_sensitivity_asymmetry_v1` · `scalar_top1_top2_margin_is_exact_distance_to_flip_v1` (measured-exact on this scorer) · NEW: `score_atomic_flip_byte_exchange_v1` |
| **GAUGE_MEASUREMENT** | clip/lattice-phase-specific measured constant | none (re-measure per clip) | the 0.005318 spike rate · anisotropy 9.56:1 · churn 1.245% · δ_R 0.0196 · d0/τ/β of `dseg_stretched_exponential_anneal_trajectory_v1` · curvelet −48% · chroma flip fractions · all byte counts |
| **APPARATUS_ENGINEERING** | compute/process laws | n/a | `oom_verdict_batch_spike_peak_rss_v1` · `ema_window_pi_group_v1` · `mlx_gpu_crossprocess_nondeterminism_v1` |

**Why this matters (not bookkeeping):** covariance class = TRANSFER RULE. A COVARIANT_LAW transfers
to any clip/scorer/grid — build on it freely. A GAUGE_MEASUREMENT is a constant of THIS clip — citing
it on another vehicle is the L18 ancestor-number sin, now made structurally visible. Several
registered equations are a covariant LAW carrying a gauge CONSTANT (the spike-floor law: the theorem
transfers, 0.005318 does not) — the split must be explicit, and the new registrations model it
(`covariance_class` as a dict `{law: COVARIANT_LAW, constant: GAUGE_MEASUREMENT}`).

**Implementation (zero schema change now, field later):** `domain_of_validity` is an open Mapping by
contract → new/updated equations carry `covariance_class` as a key TODAY (the four new laws do). At
the next `CANONICAL_EQUATION_SCHEMA_VERSION` bump, promote to an optional first-class field
`covariance_class: str | Mapping | None` validated against the 4-value vocabulary
(`COVARIANCE_CLASSES` exported from `einstein_pass_covariance_laws_20260710`). Backfill of the 281
legacy rows is a mechanical sweep (Sonnet-tier) using the table above as the classification rule —
NOT done in this unit (append-only discipline; a bulk re-registration wave needs its own review).

---

## §3 INQUIRIES 2–5 — derivations, conservation laws, WFR, λ

### 3.1 DERIVE-DON'T-FIT (inquiry 2) — anchor by anchor

**(a) Fisher=margin r=0.978 — ALREADY LAW; the residual physics named.** The exact identity is
registered (`fisher_curvature_equals_categorical_fisher_trace_caustic_v1`): categorical Fisher
tr F = 1−Σp², which on the two-class annulus is ½sech²(m/2) — exactly monotone in the margin
[DERIVED, registered]. So the fit→law promotion was already done; what the Einstein pass adds is the
**residual diagnostic**: measured Spearman (0.908) < Pearson (0.978). A clean monotone nonlinearity
would give Spearman ≥ Pearson; the inversion says the residual is NOT the sech² curve shape but a
**heteroscedastic spatially-varying gain** — the pullback Jacobian term vᵀJᵀF J v (texture-dependent
input gain) [DERIVED-diagnosis, CONJECTURED-attribution]. That is precisely the gated Cramér–Rao
flip-risk thread (`intake_fisher_is_loglik_curvature…` §"one actionable thread") — the residual
physics and the gated build are the same object. No new build; the gate stands (beat exact
through-R S_R or save its cost).

**(b) Temporal-majority floor 0.005318 — PROMOTED fit→LAW.** Theorem: over a spike triple (a,b,a), a
label-constant witness errs 1 (majority), 2 (spike label), or 3 (other) times ⇒ majority optimal ⇒
floor = spike rate; first-order in spike density (adjacent-spike overlap neglected, spikes 97.7%
repairable) [DERIVED]. Registered as
`gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1` (the registration the DAG owed at
FEED-flicker-term), with the covariant-law/gauge-constant split explicit. Predicts what no prior
equation stated crisply: **every smooth-label capacity/epoch/basis sweep is bounded at 0.0053 —
NO-GO below it by theorem** — killing an entire class of would-be runs before they are launched.

**(c) Byte-vs-d_seg carrier laws — PARTIALLY DERIVED.** The exponent is derivable: d_seg is a
boundary-displacement functional, so N-term approximation of the cartoon class gives the rate
exponent (registered `shearlet_nterm_upper_bounds_task_rate_v1`, O(N⁻²log³N)); the CONSTANTS are
gauge (margin distribution of THIS scorer on THIS clip). Obstruction to full derivation: the
constant requires the flip-margin density near zero, which is an empirical field. Honest verdict:
exponent = LAW, constants = GAUGE_MEASUREMENT — re-measure per vehicle, never transfer (L18).

**(d) Anisotropy routing d_H×share×(1−static) — DERIVED at first order.** From the norm picture
(§1): per-class d_seg contribution = ∫_{∂Σ_k}|δn| dℓ ≈ (boundary Hausdorff measure d_H) ×
(flip density per unit length ∝ share/d_H) × (temporal activity 1−static), assuming the three
factors decorrelate. The routing weight is the first-order factorization of the boundary integral;
its RESIDUAL = the factor correlations (Lane is simultaneously high-share and non-static ⇒
cross-terms — the routing map under-weights Lane relative to exact accounting) [DERIVED-first-order;
residual physics = factor correlation, measurable from the existing anisotropy map `814fb1aac` at $0
when next touched].

**(e) Costate λ — already exact** (`costate_lambda_marginal_ds_v1`: score-law partials). Nothing to
derive; see 3.4 for what λ is NOT.

### 3.2 THE MISSING CONSERVATION LAWS (inquiry 3) — Noether/Bianchi/topology

**(i) Island count is a conserved topological charge [DERIVED, REGISTERED
`island_topological_charge_conservation_v1`].** β0 per class changes ONLY at degenerate critical
points of the margin field (Morse events, m=0 with ∇m=0 at an interior point); continuous
non-degenerate evolution conserves it. Perimeter-dominated descent (MCF,
`mcf_minority_erasure_inevitability_v1`) OPPOSES passage through the birth configuration ⇒
**gradient training conserves the islands-unborn state.** This is the deep reason mod32cap birthed
zero lane/movable islands [MEASURED] and why the birth levers (#300/#301/#315, one-sided Chan-Vese
area) are NECESSARY, not optional: charge must be injected by an explicit source term because the
smooth flow conserves it. Consequence with teeth: **never chase island birth with more
capacity/epochs on a smooth flow — the conservation law forbids it.** (Unborn islands carry 63.9% of
residual d_seg — this law governs the largest single residual bucket.)

**(ii) Persistence stability = the conservation law unifying dash-erasure and flicker [DERIVED,
flagged-not-registered].** The bottleneck stability theorem (d_B(Dgm f, Dgm g) ≤ ‖f−g‖_∞) says
L∞-small perturbations of the margin field can only create/destroy features of persistence ≤ the
perturbation. High-persistence structure is CONSERVED; only the low-persistence tail is volatile.
L65 lane-dash erasure (spatial persistence below one cell) and the flicker spikes (temporal
persistence = 1 scored frame, margin p50 0.555) are ONE object — the sub-cell persistence tail on
two axes — now grounded in a theorem rather than an analogy (the flicker memo §3 said this; the
stability theorem is WHY). Not registered: needs no new callable; it is the stated sister of (i).

**(iii) The storable charge: the PHASE ZERO-MODE [CONJECTURED — the carrier candidate].** T1's
advection equation fixes phase DIFFERENCES across pairs (t_{p+1} − A_ξ t_p = Δt_ξ); the solution is
determined up to **one integration constant per boundary component per trajectory-segment** (reset
at reaction events). The full winding number is nearly determined by boundary length (≈ perimeter in
lattice cells — low information, NOT a useful carrier); the ZERO-MODE is the genuinely free,
irreducible datum. Estimated size: O(components × segments) scalars per clip — order tens–hundreds
of bytes, quantizable. By the exchange law, break-even needs it to fix ≳0.785 flips/byte — the spike
channel it completes carries ~1046 px/pair, so the margin is enormous IF the phase model fits
[CONJECTURED; the T1 A/B (blink_fit_frac telemetry, pre-registered in the flicker memo) is exactly
the measurement that prices this]. Lands with T1 in build-wave #377/#386, as the carrier half.

**(iv) Bianchi analog — holonomy consistency [DERIVED, already embodied].** Pair-to-pair holonomies
must compose to the clip trajectory (integrability of the connection); the cumulative SE(3) B-spline
(`ego_motion_cumulative_se3_bspline_v1`) IS this constraint in code. Noted so the registry knows the
B-spline is the Bianchi identity, not a convenience.

### 3.3 WFR-GEODESIC UNIFICATION TEST (inquiry 4) — verdict: PARTIALLY HOLDS

Formal claim tested: witness flow = geodesic of Wasserstein–Fisher–Rao (Hellinger–Kantorovich),
interpolating transport (W) and reaction (FR).

- **HOLDS as the decomposition of the TEMPORAL structure [MEASURED support].** WFR's dynamic form
  splits evolution into transport (∂ρ+∇·(ρv)=ρg's v-term) + reaction (g-term). The scene's measured
  split is exactly this: transport generated by ξ (94.8% zero-shift, 97.5% ground-plane coherent,
  store-nothing collapse) + reaction events (class-mass creation: lane births, movable
  appearance — the index-permutation discontinuities that defeat pure-transport models). The rate
  decomposition follows: bytes(temporal) ≈ bytes(ξ) + bytes(reaction events) + bytes(phase
  zero-modes). **T1 is the WFR reaction-cost term restricted to the phase channel** — it penalizes
  apparent reaction not explained by transport. This makes T1/#360's forces the components of ONE
  WFR action rather than a grab-bag [DERIVED-as-identification].
- **FAILS as a geodesic theorem for the TRAINING dynamics [obstruction named].** The training flow
  is gradient descent of S in the Fisher pullback metric — a gradient flow, not a geodesic; and the
  metric is degenerate along the argmax gauge orbits (monotone logit rescalings), so the geodesic
  equation is not even well-posed without gauge fixing. Honest verdict: **WFR is the right metric
  for the SCENE'S time axis (and prices the temporal carriers); it is NOT the witness's training
  law.** No unification of the two objectives into one geodesic principle — instead the cleaner
  statement stands: Fisher governs distinguishability (d_seg), Wasserstein/WFR governs transport
  (ξ, temporal), and the interface discipline (Nielsen intake pt. 4, the measured
  Wasserstein-optimal/Fisher-suboptimal chart-law) remains binding.

### 3.4 λ AS THE FUNDAMENTAL CONSTANT (inquiry 5) — verdict: two constants, one inequality

**τ=ε=ħ is a genuine identity** (registered `tau_eps_hbar_one_dequantization_two_scales_v1`): one
internal scalar plays Maslov constant, interface width, temperature [DERIVED, stands]. **λ is a
genuinely fundamental object** — the exact cotangent/price vector (100, 5/√(10 d_pose), 25/N)
[DERIVED-exact, registered]. **But λ = τ would be a category error, and the identification is
REJECTED:** τ has units of logits (an internal scale of the smoothing); λ has units of score per
distortion (an external exchange rate). They live in dual spaces (state scale vs costate price).

The genuine relation is an **inequality coupling them — the semiclassical admissibility bound**
[DERIVED]: the soft action mis-prices the hard score by ≤ τ·ln K per unit annulus mass (Maslov
bound, registered), and a τ-smoothed decision is trustworthy at a flip only when the smoothing is
below the local margin scale:

> **τ_final · ln 5 ≪ m_flip,p50 = 0.65 ⟹ τ_final ≪ 0.40.**

Live schedule endpoint τ=0.05 gives τ·ln5 = 0.080 ≪ 0.65 ✓ — the hand-picked endpoint is now a
DERIVED check, with the corollary that annealing far below ~0.05 buys nothing the flip margins can
feel (diminishing returns are predicted, not suspected). Flagged as candidate
`tau_endpoint_semiclassical_admissibility_v1` — NOT registered yet (register when it SETS an
endpoint in a launched config rather than post-hoc validating one; avoids a law with no consumer).
Constraint on the costate controller: λ decides WHERE effort goes (prices), τ decides WHEN the
hard-argmax regime is trustworthy (scale) — the controller should never conflate them.

---

## §4 TABLE A — EQUATIONS: ranked, executable rows

| # | action | exact statement | disposition | what it predicts/constrains that we didn't have | verification path | near-term exact-row? |
|---|---|---|---|---|---|---|
| A1 | **REGISTER floor law** | min over smooth-label witnesses of d_seg = q_spike (majority costs 1/spike); q_spike^n600 = 0.005318 | **REGISTERED** `gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1` (DERIVED theorem + n600 anchor; law=COVARIANT, constant=GAUGE) | every smooth-label sweep is NO-GO below 0.0053 BY THEOREM — pre-kills a whole run class; the T2-alone cap is now a law, not a ledger note | already-measured n600 census; T2/T1 A/B anchors append on landing | **YES** — aims the next arm at the phase channel, the only path to the need band |
| A2 | **REGISTER atomic exchange** | ΔS_flip = 100/(600·196608), ΔS_byte = 25/37545489, exchange 1.27311 B/flip; store B bytes fixing f flips iff B < 1.273·f | **REGISTERED** `score_atomic_flip_byte_exchange_v1` (DERIVED-EXACT; SCORER_FRAME_STRUCTURAL) | every carrier/sidecar gets an exact break-even (Lever-D's 0.65 = this ×~2 margin, now derived); pointer move = 20 flip quanta (quantization confirmed) | pure arithmetic + the clickpolish row; future rows re-anchor free | **YES** — direct input to click-polish/waterfill/carrier economics feeding exact rows now |
| A3 | **REGISTER equivalence-principle split** | d_seg = d_cov (ξ-frame separatrix error) + d_gauge (lattice-phase term ≈ q_spike, present in GT); converged witness: d_cov ≈ 0 | **REGISTERED** `dseg_covariant_gauge_decomposition_v1` (DERIVED; COVARIANT_LAW) | covariant capacity is now provably WASTED at the converged witness; descent = gauge-channel fitting; forces T1 + ground-frame + code-dim collapse | anchors: 94.8% zero-shift, 97.5% coherent, blink-back 41.8%, FEED-ma 0.00086; T1 A/B is the direct test | YES (via T1's arm) |
| A4 | **REGISTER conservation law** | β̇0 = 0 without Morse events; each event ±1; perimeter descent opposes births ⇒ source terms REQUIRED | **REGISTERED** `island_topological_charge_conservation_v1` (DERIVED, Morse theory; COVARIANT_LAW) | forbids capacity/epoch-chasing island birth (63.9% of residual!); Chan-Vese/seed levers become derived-not-chosen | mod32cap zero-island outcome (measured); next birth-lever arm anchors it further | medium-term (governs the island arm design) |
| A5 | **RE-TAG covariance_class** | 4-value vocabulary (§2); `domain_of_validity["covariance_class"]` now; first-class optional field at next schema bump; backfill = mechanical sweep with §2 as the rule | proposal + 4 seed tags landed; backfill wave NOT in this unit | transfer rules become machine-visible: gauge constants can't silently cross vehicles (L18 structurally enforced) | sweep + registry validator extension | no (hygiene; prevents future fake-transfer verdicts) |
| A6 | **FLAG τ-endpoint bound** | τ_final·ln5 ≪ m_flip,p50 (=0.65) ⇒ τ_final ≪ 0.40; live 0.05 ⇒ 0.080 ✓ | FLAGGED `tau_endpoint_semiclassical_admissibility_v1` — register when it SETS a config endpoint | schedule endpoint becomes derived; predicts diminishing returns below τ≈0.05; λ≠τ category firewall for the controller | next config that derives its τ endpoint from the bound | no (long-horizon polish) |
| A7 | **FLAG WFR split** | temporal structure = WFR transport(ξ) + reaction(events) + phase zero-modes; T1 = the reaction-cost term on the phase channel; training flow is NOT a WFR geodesic (obstruction: gradient flow + gauge-degenerate metric) | FLAGGED `temporal_wfr_transport_reaction_split_v1` — register with the T1 A/B anchor (its blink_fit_frac IS the reaction-fit metric) | unifies T1/#360 forces as components of one WFR action; prices temporal carriers as bytes(ξ)+bytes(events)+bytes(zero-modes) | T1 A/B (pre-registered in flicker memo §4) | YES (with T1) |
| A8 | **Residual-physics note (no build)** | Fisher=margin residual 0.022: Spearman(0.908)<Pearson(0.978) ⇒ heteroscedastic Jacobian-pullback gain, NOT curve nonlinearity | note on the registered caustic identity; the gated Cramér–Rao thread is its build-form (gate unchanged) | names the next-order physics of our best-measured anchor; keeps the gate honest | only if the gated check ever fires (beat exact S_R or save its cost) | no |

## §5 TABLE B — SYSTEM: ranked recommendations toward the optimal system

| # | recommendation | lands where | expected effect | horizon | verify |
|---|---|---|---|---|---|
| B1 | **Adopt GENERAL COVARIANCE OF THE WITNESS as the design principle** (§1) and run its $0 audit: regress per-pair FiLM/code variance on (ξ, measurement-phase); the residual must be attributable to scene events + appearance, else it is wasted rate | design doctrine + a $0 cached-data audit script (Sonnet-tier); informs v8 #377/#386 | the next levers get DERIVED instead of rediscovered; per-pair code dim collapses toward 6+zero-modes ⇒ direct rate reduction; wasted-rate DOF become visible | **near-term** (audit is $0; v8 build-wave consumes it) | audit R² decomposition vs the rank-8 95.6% anchor; then byte-closed v8 row |
| B2 | **Store the conserved charge: the phase zero-mode carrier** (§3.2-iii) — T1 fixes phase differences; ship the per-boundary-component integration constants (tens–hundreds of bytes, quantized) as the gauge-channel completion | carrier design beside T1 in build-wave #377/#386 (trainer flag + `Lever` + carrier section TOGETHER) | completes the ONLY path below the 0.0053 floor to the need band (0.00077–0.00118); exchange law prices it: needs ≥0.785 flips/byte, spike channel offers ~1046 px/pair of headroom | **near-term exact-row path (gold)** — this + T1 is the d_seg endgame arm | T1 A/B blink_fit_frac ↑ + verdict d_seg below 0.0053, then byte-close + exact eval |
| B3 | **Island births only via source terms** (from A4): stop any plan that buys island birth with capacity/epochs; route the 63.9% island bucket exclusively through seeded/Chan-Vese arms | curriculum/config discipline (#300/#301/#315 arms); a one-line gate in the config-review checklist | prevents a provably-futile run class; concentrates the island budget on arms that can work | near-term (guides the next island arm) | the birth-lever arm's island-recall telemetry vs the conservation prediction |
| B4 | **Wire atomic units into the costate controller** (#247 SENSE/duty): rank every candidate lever by expected flips-fixed × 8.477e-7 vs bytes × 6.659e-7 (and pose via λ_pose) — one commensurable EV column for the duty-to-measure queue | `tac.witness_control.shadow_controller` duty ranking (advisory-only; actuation boundary unchanged) | the "which lever next" decision becomes a priced calculation, not judgment; consumes A2 | near-term (small advisory patch) | backtest the ranking against the ledger of measured lever ΔS rows |
| B5 | **Metric discipline codified** (from 3.3): Fisher for d_seg levers, Wasserstein/WFR for transport/temporal levers, WFR-reaction pricing for event carriers; never optimize a d_seg lever in transport metric (the measured OT-hurts-d_seg case is the canonical violation) | one paragraph in the witness DSL doc + review-checklist line | prevents Wasserstein-optimal/Fisher-suboptimal repeats (#288/#382/v8 OT surface) | standing discipline | next OT-lever A/B carries the metric declaration |
| B6 | **τ endpoint from the admissibility bound** (A6): derive, don't hand-pick, the anneal floor per config (τ_final ≪ m_flip,p50/ln5) | schedule derivation in `witness_dsl`/schedule_readback when next touched | schedule endpoint becomes provenance-laddered (DERIVED-AT-CONFIG); predicts no gain below ≈0.05 on this clip | long-horizon polish | a config whose τ endpoint cites the bound; A/B only if the bound ever binds |

**The single highest-value row the coordinator didn't ask for:** B1+B2 as one move — the covariance
principle plus its conserved-charge carrier. It converts the flicker wall (the one thing standing
between the witness and the need band) from "a term we designed" into "the gauge sector of a
principle, with its integration constants as the carrier" — and it is the near-term exact-row path.

---

## §6 HONESTY BLOCK + own-round-1 review

- $0, CPU-only, no scorer forward, no training, pid 88030 untouched. All fresh numbers are
  arithmetic or citations of already-measured n600 anchors (each traced to its memo/equation).
  **Pointer 0.19108282 UNMOVED** — four laws registered, zero score moved; the near-term-exact-row
  claims are PATHS (T1+zero-mode arm, v8 audit), not rows.
- Scope ladder: the four registrations are DERIVED laws with measured anchors (floor law
  FAMILY-level for smooth-label witnesses on this clip/scorer; exchange law exact within the
  contest apparatus; decomposition/conservation COVARIANT with clip-level constants). The reframe
  principle (§1) is CONJECTURED-as-totality with three measured pillars; its audit (B1) is the
  falsification path. WFR-geodesic: PARTIAL, obstruction named. λ=τ: REJECTED (category error),
  replaced by a derived inequality.
- Own review, adversarial pass: (1) is the 20-quanta check circular? No — the quantum is derived
  from evaluate.py constants; the pointer row was measured independently; agreement within reported
  rounding is a real consistency test, though d_seg's 5-sig-fig reporting caps its resolution at
  ~1.2 quanta. (2) Is the floor theorem exact? First-order only — adjacent-spike overlap and
  non-repairable spikes (2.3%) shift it at the few-percent level; stated in-module. (3) Is the
  island law falsifiable? Yes: a smooth-flow run that births a persistent island without a margin
  degeneracy would refute it (none observed; mod32cap is the measured instance). (4) Danger of the
  reframe: over-covariantizing — GT's gauge content must be FIT, not removed; the decomposition law
  states this explicitly (the gauge term is in the labels; equivariance, not invariance, is the
  target). (5) DSL leg: N/A-with-rationale — no new trainer flag here; the levers the laws force
  (T1/#274/Chan-Vese) are owned by their own FEEDs and the flags land in build-wave #377/#386.
