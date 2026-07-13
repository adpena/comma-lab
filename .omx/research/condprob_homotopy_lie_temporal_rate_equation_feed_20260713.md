# Canonical equation FEED — marked temporal transport rate law v1

**Date:** 2026-07-13

**Proposed equation id:** `marked_temporal_transport_rate_law_v1`

**Status:** DESIGN / equation feed; not registered

**Authority:** DERIVED-EXACT conditional information identity plus explicitly scoped geometric
conditions

**Pointer:** UNMOVED

## One-line law

Temporal witness rate is the conditional entropy of quantized Lie transport plus a marked
prediction-innovation process. Homotopy transitions are one event family; receiver-phase and chart
crossings are distinct. Phase is the complete between-event rate only when the remaining regular
residual has zero conditional entropy.

## Canonical form

Let `C_t` be public/already-decoded context, `X_t=Q(xi_t)` the receiver's quantized Lie datum,
`E_t` a full marked event with zero denoting the current regular chart, `Phi_t` receiver phase, and
`Delta_t^e` the remaining branch residual. For a receiver-valid bijective coordinate chart,

\[
\boxed{
\begin{aligned}
H(X_t,W_{t+1}|C_t)
&=H(X_t|C_t)+H(E_t|X_t,C_t)\\
&\quad+H(\Phi_t|E_t,X_t,C_t)
+H(\Delta_t^{E_t}|\Phi_t,E_t,X_t,C_t)\\
&=H(X_t|C_t)+H(E_t|X_t,C_t)\\
&\quad+\sum_e\bar p_e\{H(\Phi_t|E_t=e,X_t,C_t)
+H(\Delta_t^e|\Phi_t,E_t=e,X_t,C_t)\}.
\end{aligned}
}
\tag{Marked-temporal-rate-v1}
\]

Here `bar p_e=P(E_t=e)` is the ensemble branch mass; pointwise in decoded context use
`p_e(c)=P(E_t=e|C_t=c)` and then average over `c`. If `X_t` is already-paid side information,
remove `H(X_t|C_t)`. If `E_t` is only binary, add the marked payload term
`P(E=1)H(J_t|E=1,X_t,C_t)`.

Define the disjoint prediction-break ontology

\[
E_{\rm pred}=E_{\rm top}\ \dot\cup\
(E_{\rm chart}\setminus E_{\rm top})\ \dot\cup\
(E_R\setminus(E_{\rm top}\cup E_{\rm chart})).
\]

For class-difference fields `f_cd(x,s)`, latent continuous topology is constant while every active
zero set is regular and every junction map is transverse. A generic topology event lies on

\[
f_{cd}(x,s)=0,\qquad\nabla_xf_{cd}(x,s)=0,
\]

or a higher-junction rank failure.

Finally,

\[
H(W_{t+1}|E=0,X_t,C_t)=H(\Phi_t|E=0,X_t,C_t)
\Longleftrightarrow
H(\Delta_t^0|\Phi_t,E=0,X_t,C_t)=0.
\]

## Relation to settled laws

- Refines `weyl_evaluator_fiber_rate_law_v1`: the temporal chart is inside the exact statistic
  fiber/setoid, and a receiver-valid section remains mandatory.
- Composes with `stratified_extension_orbit_rate_law_v1`: no independence is inferred from a split
  semidirect chart; every term remains conditional.
- Refines `dseg_covariant_gauge_decomposition_v1`: receiver phase is `E_R`/gauge innovation, not
  necessarily latent topology.
- Preserves `gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1`: the n600 `0.005318`
  constant is a measured gauge-channel size on this clip, not a universal event entropy.
- Preserves `island_topological_charge_conservation_v1`: topology changes require critical events in
  the continuous field, but sampled label spikes need not be such events.

## Domain of validity

Included:

- discrete/quantized source variables or explicitly quantized continuous variables;
- fixed public receiver context and apparatus;
- a declared measurable bijection between next state and branch coordinates;
- a full event mark or an explicit separate mark payload;
- a declared topology complex/scale and regularity/transversality conditions;
- expected ideal source rate, not individual archive length.

Excluded:

- omitting `Qxi` cost without declaring side information;
- unweighted addition of per-branch conditional entropies;
- identifying latent topology events with all receiver-visible flips;
- assuming phase-only sufficiency without proving the zero residual-entropy condition;
- calling entropy a pointwise Brotli/ZIP byte count;
- pointer movement without exact receiver-consumed archive evidence.

## Evidence and epistemic status

| clause | status | boundary |
|---|---|---|
| conditional chain rule | **DERIVED-EXACT** | coordinate chart must be lossless/typed |
| topology constant off discriminant | **DERIVED** | continuous labeled fields; regularity/transversality |
| topology events subset of prediction breaks | **DERIVED under fixed-topology no-event chart** | event predicate must be preregistered |
| L85 spikes are generally receiver phase, not latent topology | **MEASURED mechanism + DERIVED distinction** | digital topology can still change at chosen pixel scale |
| between-event phase-only | **INFERRED/CONDITIONAL** | `H(Delta0|Phi,E=0,X,C)=0` unmeasured |
| exact archive saving | **UNKNOWN** | arithmetic model, tables, receiver, ZIP A/B owed |

## Falsification and reactivation

- A non-bijective phase/residual chart rejects the decomposition as a codec formulation; add the
  missing state or use a cross-entropy upper bound.
- A regular interval with a correctly declared latent topology change falsifies the discriminant
  typing/complex, not Morse theory.
- Nonzero held-out residual codelength after phase rejects phase-only sufficiency on that stratum.
- A receiver-packed conditional coder above `0.65 B/surviving flip` rejects that formulation, not
  temporal conditional coding as a family.

## Proposed consumers

- v8 event-carrier grammar and edge-centric conditional coder;
- CGauge phase/event split;
- class-pair flip waterfill and contour arithmetic coder;
- tau-rung persistence/discriminant controller;
- SE(3) pose/phase quantizer;
- cathedral stop gate for untyped event/topology/receiver claims.

## Triality note

- **Equation:** this file.
- **DAG:** `.omx/research/condprob_homotopy_lie_DAG_FEED_20260713.md`.
- **DSL:** proposed typed record in the companion memo; no shared DSL edit.

The central registry remains untouched because the mission is design-only and live siblings own
shared equation surfaces.
