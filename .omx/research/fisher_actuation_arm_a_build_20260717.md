# Fisher actuation — build-wave ARM A landing (SPEC_v10 §13.1 row 2 · §13.4 · §13.5)

**Date:** 2026-07-17 · **Branch:** `p0_build_fisher_actuation_20260717` · **Ledger:** `p0_fisher_full_leverage_20260717`
**Charter:** `.omx/tmp/build_wave_20260717/ARM_A.md` against SPEC_v10 §13 (branch `claude/p0_521_spec_v10_capstone_20260717`).
**Pointer 0.19108 UNMOVED — everything here is MEANS (levers built default-OFF + one advisory measurement), not a score row.**

## What landed (4 deliverables, each committed when its tests passed)

| # | deliverable | surface | commit | tests |
|---|---|---|---|---|
| d4 | Fisher-mass-in-annulus observable | `src/tac/witness_control/fisher_annulus.py` | 82b04059ee | 16 pass |
| d1 | `--fisher-density-weight` / `--fisher-density-source` lever | base `make_loss_fn` + levelset trainer + DSL `FisherDensityWeight` | d8349ad9be | 24 pass (shared file with d2) |
| d2 | `--head-natural-grad` / `--head-natural-grad-eps` lever | base `make_seg_logits_natural_grad_mlx` + levelset trainer + DSL `HeadNaturalGradient` | d8349ad9be | (same 24) |
| d3 | dual-metric read-back harness + THE RUN | `tools/dual_metric_readback.py` + `.omx/research/dual_metric_readback_pa_vs_seg_n96_20260717.json` | 88287bbc6b | smoke n4 + real n96 row |
| eq | equations leg | `src/tac/canonical_equations/fisher_actuation_arm_a_20260717.py` (`categorical_fisher_pseudoinverse_cotangent_precondition_v1`) | (this batch) | 10 pass |

All levers DEFAULT-OFF and byte-identical when unset (verified: `test_off_path_is_byte_identical_to_no_kwarg_path`
— loss AND grads bitwise equal to the no-kwarg path). Resume-registry: additive `__cfg_fisher_density_weight` /
`__cfg_fisher_density_source_gt` / `__cfg_head_natural_grad` / `__cfg_head_natural_grad_eps` persisted at ckpt-save +
guarded in `_resume_lever_divergences` (legacy sidecars lack the keys ⇒ no spurious divergence — verified). Both levers
FAIL-CLOSED vs `--micro-batch-pairs>1` (the batched twin does not route them; joins the #313 still-fail-closed list).

## d3 — THE MEASURED TRIPLE (the §13.5 owed row: Fisher read-back of phase_advect)

**Config:** live c2 run `levelset_n600_witness_20260717T113932Z`, `levelset_witness_ema_BEST.npz` (ep725, READ-ONLY),
armed seg base = `tau_softplus` τ=0.3 w_seg=100 (the ep300–1400 stage at ep725 per launch.sh), phase_advect as armed
(w=0.4, band 2.0, classes 0,1,2, gt_advected/interp — providers rebuilt with the trainer's own
`tac.boundary_math.phase_primitives` + gfc defaults). n96 evenly-spaced pairs, both legs.
**Axis: [macOS-MLX research-signal] / [macOS advisory] — NON-PROMOTABLE, score_claim=false, n<600 subset.**

| quantity | Euclidean (weight space, BASELINE) | Fisher (decision geometry, AUTHORITY) |
|---|---|---|
| cos(g_phase, g_seg) | **−0.1494** | **−0.1178** |
| rel-norm ‖g_phase‖/‖g_seg‖ | **0.627** | **0.478** |
| PCGrad neg-projection (of g_seg by g_phase) | 0.0236 | 104.15 (Fisher-norm units) |
| verdict (|cos|>0.05 band) | ANTAGONISTIC (mild) | ANTAGONISTIC (mild) |
| term values (mean/pair) | phase 0.0463 · seg 24.18/100=0.242 | — |

**MEASURED reading:** at ep725 the phase-advection force is (a) **large** — roughly half the armed seg force in BOTH
metrics (rel-norm 0.48–0.63; contrast weight_entropy's 0.002/6e-5 from FEED-we-conflict) — and (b) **mildly opposed**
to the seg base (cos ≈ −0.12…−0.15 in both metrics; **no Euclid↔Fisher sign flip for THIS pair** — the FEED-we-conflict
flip was the weight_entropy pair; here the two metrics AGREE, which is itself the dual-metric discipline working: the
read-back reports both and the agreement is informative). Mechanistically INFERRED (not measured): partial opposition
is the DESIGNED shrinkage tradeoff — phase pulls tie coordinates toward the ξ-advected previous-pair tie while the seg
base pulls toward the current-pair GT; they disagree exactly on straddle sites. What the row ADDS beyond design intent:
the OPPOSITION MAGNITUDE is now measured (≈12–15% of the phase force is spent against the seg descent, at ≈½ the seg
force's size) — an input for the Force-3/T1 weight re-derivation and for reading the ep750 Road-concentrated tax
(§13.2's attribution confound: this row is ep725 = post-Force-3-engage, pre-Muon-entry ep726).

**Caveats (honest):** (1) self-orient dir feats reconstructed by the DECODE-style fixed point (2 iterations from
zero-dir on the EMA weights) — the training-time orientation (last reorient ep700, feats carried forward) is not
persisted in the EMA ckpt; labeled approximation, not bit-identity. (2) n96 subset (both legs). (3) single checkpoint,
single stage. (4) Fisher leg = JVP(render-only, STE-faithful) + central-FD of the smooth SegNet — the measure-script
method (its per-pixel Fisher identity self-check is inherited by construction of the same vectorized form).

## Label ledger

* **MEASURED:** the d3 triple (table above, n96 advisory); byte-identity of the OFF paths (bitwise test); grad
  engagement of both levers when ON; forward-identity of the NG transform (loss value equal, grads different);
  16+24+10 test results.
* **DERIVED:** w(x)=sech²(m/2) as the categorical Fisher trace (registered law
  `fisher_curvature_equals_categorical_fisher_trace_caustic_v1`, ρ=0.978 measured calibration — cited, not re-measured);
  the closed-form g⁺v = v/p − mean(v/p) pseudo-inverse (new equation module, inversion-verified + cross-validated
  against `tac.information_geometry.fisher_natural_solver` at zero damping); the source choice: **model** margins are
  the Fisher-NATURAL training-force field (the pullback metric is evaluated AT the current point of the flow), **gt**
  margins are a stationary separatrix prior — both shipped, model is the default.
* **INFERRED:** the phase-vs-seg opposition mechanism (straddle-site tradeoff); the expectation that Fisher-density
  helps by concentrating budget where decisions bend (efficacy is UNMEASURED — the $0 cached-ckpt A/B is the owed gate).
* **ASSUMED:** the harness's self-orient fixed point lands close to the training-time orientation (bounded by the
  reorient-every=50 staleness + fixed-point contraction; unquantified); vjp-through-custom_function behaves identically
  inside the full trainer graph with `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` + `--safe-compile-regions hosc_activation`
  (the mini-loss test exercises value_and_grad, not the full launch config — see open items).

## Composition / antagonism vs existing levers (charter requirement)

* **FisherDensityWeight × SegFocalGamma (#301):** SAME multiplicative `seg_pixel_w` surface — they compose by
  multiplication, but DISAGREE on confidently-wrong pixels (focal (1−p_y)^γ UP-weights them; the even sech² density
  DOWN-weights them — a confidently-wrong pixel is metrically flat). Default guidance: one at a time; composing both is
  a deliberate double-concentration experiment, not a default.
* **× BoundaryDistance (#301):** boundary-distance is the θ-independent PIXEL-distance separatrix band; Fisher-gt is
  its smooth MARGIN-units analog (same intent, smoother falloff); Fisher-model additionally TRACKS the live field.
  Composing gt-source with boundary-distance double-concentrates the same annulus — prefer one; model-source adds the
  live-tracking dimension boundary-distance cannot.
* **× MarginBandSatisficing (#360, armed in c2):** complementary — satisfice CAPS the push beyond m_safe (a hinge
  target); Fisher-density reallocates the BASE loss budget. No shared surface; no sign conflict expected; joint A/B owed.
* **× l7_softplus stage (ep1401+):** l7 carries its OWN hard-pixel weight (margin<thr, mean-1); Fisher weight would
  multiply on top = double boundary-weighting at l7. If armed through l7, re-derive the joint weight (open item).
* **× SegSpikeReweight / logit-adjust:** multiplicative / orthogonal-surface respectively; no conflict identified.
* **HeadNaturalGradient × --head-offset-solver flip_median (#423, ARMED in c2):** COMPOSES — #423 is a periodic
  closed-form SOLVE of `out_sdf.bias` at checkpoints; NG preconditions the PER-STEP cotangent. Different surfaces, no
  duplicate mechanism (charter-required statement). Same relation to the #518 fork head-SOLVE (head weights per stage).
* **HeadNaturalGradient × Muon (ep726+ in c2):** OPEN antagonism risk — Muon orthogonalizes weight-space updates;
  NG reshapes the upstream cotangent; double-preconditioning is unstudied. The $0 A/B should run pre-Muon (AdamW stage)
  first.
* **Fisher-annulus observable:** observer-only; the §13.2 trigger consumer (engagement/convergence events reading
  Fisher mass, not Euclid d_seg slope) is arm-B/controller territory — this module is its sensor primitive.

## Triality legs

* **DSL:** `FisherDensityWeight` + `HeadNaturalGradient` Lever factories (curriculum_dsl); registry `completeness()`
  shows no unmapped fisher/NG flags.
* **equations:** `categorical_fisher_pseudoinverse_cotangent_precondition_v1` module landed (sister-of
  `categorical_fisher_natural_trust_region_solve_v1`, same not-yet-queryable module maturity — the JSONL registry
  append is deliberately deferred until the owed A/B lands its first empirical anchor; FORMALIZATION_PENDING:
  registering a training-efficacy law with zero training anchors would encode a verdict we have not measured. The
  Fisher-density lever itself EVALUATES the already-registered sech² law — no new law needed there).
* **DAG:** this memo is the durable artifact; the FEED row belongs to the parent's merge pass (worktree branch; the
  live DAG file is on main — noted as an open item for the merge boundary per §13.6).

## Round-1 adversarial self-review (attack my own build)

1. **Counted-but-inert check (NO-FAKE):** both levers PROVEN to bind — grads change when ON
   (`test_fisher_weight_on_changes_grads_not_budget_sign`, `test_head_ng_preserves_loss_value_changes_grads`); the
   NG transform is forward-identity by construction and its vjp verified against the explicit g matmul AND the CE
   closed form. NOT markers-without-work.
2. **Attack: does the custom vjp survive the FULL launch graph?** Untested at the real launch config
   (grouped-backward custom kernels + safe-compile regions). The mini-loss test uses plain `nn.value_and_grad`.
   MITIGATION OWED before any armed run: a 2-epoch n6 smoke at the launch flag set with `--head-natural-grad`
   asserting grads differ from the OFF run and loss values match at step 0. (Open item #1.)
3. **Attack: the lever name "head-natural-grad".** The mechanism preconditions ALL witness params through the logit
   bottleneck, not only head params. The help text + DSL notes state this explicitly ("per-step direction of every
   witness param through the logit bottleneck"); "head" refers to the space the metric lives in (SPEC §13.4(2)
   wording). Disclosed, not a fake name — but a reviewer should read the help before assuming head-only scope.
4. **Attack: harness reconstruction fidelity.** The self-orient approximation is the weakest link of the d3 row; the
   direction-level conclusion (mild antagonism, ~½ magnitude) would need the orientation error to flip gradient signs
   to be wrong — implausible but UNQUANTIFIED. A sensitivity probe (1 vs 2 vs 3 fixed-point iters) is cheap and owed
   if the row becomes load-bearing for a weight re-derivation. (Open item #2.)
5. **Attack: normalization semantics.** mean-1 renorm is per-frame (per-pair), matching the focal precedent; at λ<1
   the blend keeps min weight ≥ 1−λ (never kills interior pixels). λ>1 REFUSED (negative weights) — fail-closed
   tested.
6. **Attack: gt margins are unsigned, model margins signed.** The density is even in m, so the two sources are on the
   same scale; verified by the symmetry test.
7. **Attack: could the d3 row be a frozen/corrupted measurement (L3 confound clearance)?** The run is a read-only
   forward/backward at a fixed θ — no liveness question; term values are finite and nonzero (phase 0.046, seg 0.242);
   the non-finite guard is fail-closed; the n4 smoke and n96 run agree in sign and magnitude band (cos −0.066→−0.149
   Euclid as n grows; same ANTAGONISTIC side). Positive control: the seg self-norm ‖g_seg‖ ≫ 0 and the Fisher leg's
   per-pixel identity self-check is the same vectorized form validated in the measure script.
8. **Attack: SPEC drift.** §13.6 says every lever lands as a DSL Lever (done), default-OFF (done), live run untouched
   (only READ), merge at post-v9c2 boundary (this branch stays unmerged — parent's call). No §13 sentence contradicted
   by this landing that I can find; the DAG FEED row is deferred to the merge (stated above, not silently dropped).

## Open items (round-2 attack surfaces)

1. Launch-config NG smoke (custom vjp × grouped-backward × compile regions) before any armed run.
2. Self-orient reconstruction sensitivity probe (fixed-point iters 1/2/3) if the d3 row becomes load-bearing.
3. The $0 cached-ckpt A/Bs (§13.1 row 2 duty-to-measure): FisherDensityWeight(model) vs (gt) vs SegFocalGamma at one
   ckpt; HeadNaturalGradient pre-Muon. Registry JSONL append for the g⁺ law rides the first anchor.
4. Micro-batch twin routing for both levers (currently fail-closed).
5. l7-stage joint-weight re-derivation if Fisher-density is armed through ep1401+.
6. DAG FEED row at the merge boundary (worktree cannot append the main DAG file without clobber risk).

## STORES CONSULTED

SPEC_v10 §13 (SSoT branch) · `.omx/research/weight_entropy_gradient_conflict_n600_20260715.md` +
`experiments/measure_weight_entropy_gradient_conflict.py` (method base) · canonical equations registry
(`fisher_curvature_equals_categorical_fisher_trace_caustic_v1`, `optimal_metric_unification_v1`,
`segnet_head_rank4_linear_flipdist_v1`, `categorical_fisher_natural_trust_region_solve_v1` module) ·
`experiments/test_focal_boundary_levers.py` (lever test template) · live run `launch.sh` + BEST.npz cfg ·
`tac.boundary_math.phase_primitives` + trainer phase-advect precompute (provider reconstruction) ·
CLAUDE.md non-negotiables (EMA/byte-identity/no-invent-flags/serializer/confound L3).

## §TRAJECTORY — force-evolution map across the c2 checkpoint trajectory (operator follow-on 2026-07-17 "run fishers against full telemetry")

**All rows: n96 evenly-spaced pairs, both legs; [macOS-MLX research-signal] / [macOS advisory] NON-PROMOTABLE;
term weights as armed in c2 (phase 0.4 · satisfice 0.2 · subpix 0.3 with pa_flipmass W_e) except weight_entropy
= the λ=15 COUNTERFACTUAL (the lever is OFF in this run — this is the finish-crossover probe). Armed seg base =
tau_softplus τ=0.3, w_seg=100. Raw rows: `.omx/research/dual_metric_traj_20260717/*.json`.**

**Checkpoint inventory (honest):** the run dir preserves EMA-shadow deploy artifacts only. MEASURED states:
`seed_ep650` = ancestor mod32cap EMA BEST — bit-exactly the c2 LIVE INIT at ep651 (`--warm-start-weights-only`);
`ep725_BEST` = EMA BEST; `ep726_stage` = stageMuonStart ckpt — **MEASURED bit-identical params to ep725_BEST**
(max abs diff 0.0; metadata-only difference), so it is a REPEATABILITY CONTROL, not an independent point;
`ep800_roll` = the rolling EMA pinned at ep800 (copied before read; the file is overwritten every 25 ep).
The directive's live-weights preference could NOT be honored: no plain live-weight artifact exists — live weights
live only inside `levelset_resume_state.npz` under the `liveP__*` resume schema, which my loader does not read;
per the no-loader-adaptation constraint those states were SKIPPED, not force-fitted. Context (cited, SPEC §13.3):
EMA−live −0.00095 @ep775 — small but nonzero; the ep750/ep775 periodic saves do not exist as distinct files.
Forces phase/satisfice/subpix engaged at ep700; at seed_ep650 their rows are the PRE-ENGAGEMENT counterfactual
(what the force WOULD do at that state).

| state | term vs armed seg | Euclid cos | Fisher cos | rel-norm E | rel-norm F | verdict E/F |
|---|---|---|---|---|---|---|
| seed_ep650 (live init) | phase_advect | +0.0427 | +0.1725 | 4.76 | 1.98 | orthog/synerg |
| seed_ep650 (live init) | margin_satisfice | +0.5042 | +0.0747 | 0.0746 | 0.113 | synerg/synerg |
| seed_ep650 (live init) | subpix | +0.1366 | +0.0055 | 1.05 | 0.88 | synerg/orthog |
| seed_ep650 (live init) | weight_entropy λ15 | −0.0051 | −0.1598 | 0.271 | 0.0908 | orthog/antago |
| ep725_BEST (EMA) | phase_advect | −0.1494 | −0.1178 | 0.627 | 0.478 | antago/antago |
| ep725_BEST (EMA) | margin_satisfice | +0.1875 | −0.3208 | 0.054 | 0.0664 | synerg/antago |
| ep725_BEST (EMA) | subpix | −0.0627 | −0.4291 | 0.493 | 0.627 | antago/antago |
| ep725_BEST (EMA) | weight_entropy λ15 | +0.0219 | +0.0663 | 0.25 | 0.0893 | orthog/synerg |
| ep726_stage (≡ep725 wts) | phase_advect | −0.1494 | −0.1227 | 0.627 | 0.483 | antago/antago |
| ep726_stage (≡ep725 wts) | margin_satisfice | +0.1875 | −0.3284 | 0.054 | 0.0657 | synerg/antago |
| ep726_stage (≡ep725 wts) | subpix | −0.0627 | −0.4649 | 0.493 | 0.631 | antago/antago |
| ep726_stage (≡ep725 wts) | weight_entropy λ15 | +0.0219 | +0.0630 | 0.25 | 0.0918 | orthog/synerg |
| ep800_roll (EMA) | phase_advect | −0.0590 | −0.1158 | 0.344 | 0.289 | antago/antago |
| ep800_roll (EMA) | margin_satisfice | +0.5055 | −0.1407 | 0.0438 | 0.0652 | synerg/antago |
| ep800_roll (EMA) | subpix | +0.2156 | −0.1301 | 0.215 | 0.235 | synerg/antago |
| ep800_roll (EMA) | weight_entropy λ15 | −0.0115 | +0.0049 | 0.153 | 0.0612 | orthog/orthog |

**Fisher-leg noise floor (MEASURED, from the bit-identical ep725/ep726 pair):** the Euclid leg is
reproducible to the displayed precision; the Fisher leg (MLX-GPU JVP + FD forwards) varies by |Δcos| ≤ 0.036
(subpix −0.4291 vs −0.4649) and |Δrel| ≤ 0.005 on IDENTICAL weights — read Fisher differences smaller than
~0.04 as noise.

**Reading 1 — the phase sign flip (MEASURED).** At the pre-engagement live init the phase force is weakly
synergistic with seg (Fisher cos +0.17) and DOMINANT in size (rel-norm 4.8 E / 2.0 F — an unadapted tie field
generates an enormous would-be gradient). By the first armed state (ep725; force live since ep700) it has
flipped to mild antagonism (−0.12 F) at ≈half the seg force (0.48 F), and by ep800 its share decays further
(0.29 F) with the antagonism persisting. The flip therefore happened during the ep700–725 engagement descent
itself: once the tie field adapts, the term becomes the designed shrinkage counter-force rather than a
co-mover, and it is relaxing toward equilibrium rather than growing.

**Reading 2 — satisfice stays subdominant; subpix does NOT (and is the top dual-metric disagreement)
(MEASURED).** Satisfice holds rel-norm 0.04–0.11 in both metrics at every state — a genuinely small trim
force — but it is the strongest standing sign-flip instance: Euclid calls it synergistic (+0.19…+0.51) while
Fisher calls it antagonistic (−0.14…−0.33) at every armed state, the FEED-we-conflict lesson recurring on a
live armed force. Subpix at ep725 is a HALF-SEG-SIZED force (rel 0.49 E / 0.63 F) and the most
Fisher-antagonistic term measured (−0.43 F); by ep800 it decays to 0.22/0.24 and splits sign (+0.22 E /
−0.13 F). Its Euclid−Fisher gap (up to 0.37 in cos) is the largest of any term — decisions about Force-3
weights should be made in the Fisher column only.

**Reading 3 — the weight_entropy finish-crossover (MEASURED, counterfactual λ=15).** At the latest state
(ep800) rel-norm = 0.153 Euclid / 0.061 Fisher, vs 0.00186 Euclid at the ep25 strong-signal anchor
(FEED-we-conflict, n600) — ~80× growth, exactly the mechanistically-predicted late-regime rise. The EUCLID
reading has already crossed the ~0.1 binding threshold, but the FISHER (decision-geometry authority) reading
sits at 0.061 — below threshold with ~1.6× headroom, and orthogonal (cos +0.005). Verdict for the event-gated
compression design: arming finish-phase weight_entropy at ep800-class states is NOT yet Fisher-binding (safe
by the authority metric, borderline by Euclid); the event-gate should monitor the FISHER rel-norm against the
0.1 criterion. Non-monotonicity note: the seed state shows a HIGHER Euclid rel-norm (0.271) than ep800 —
rel-norm depends on the seg force's own magnitude (which grew as satisfice/subpix/phase re-energized the
armed base), so the crossover is not a clean monotone clock; gate on the measured ratio, not on epoch.

**Scope caveats (unchanged from §d3 + two new):** EMA-shadow states only (live-weight states skipped, above);
n96; self-orient decode-style reconstruction; single run. NEW: (a) seed-state rows use the c2-armed term
parameters against an ancestor-trained state — valid as the c2 launch counterfactual, not as an
ancestor-run property; (b) weight_entropy rows are a counterfactual force (λ=15 lever OFF in-run).
