# Launch-config decisions FLAGGED for the wire-in (#224) — 2026-07-01 drift-fix

Two MEASURED findings from the 2026-07-01 unified-flow pass are now formalized as
canonical equations (drift-fix triality DAG→equations). Each RECORDS the measured
behavior but the **launch-config choice** (activation + curriculum) is a DESIGN
decision resolved by the sibling activation-research agent + the **wire-in (#224)**.
This drift-fix pass did NOT change `tac.witness_autoconfig.proven_base` and did NOT
edit `tac/witness_dsl/gauge.py` or `curriculum_dsl.py` (consolidated at wire-in).

## Decision 1 — l7 stage: KEEP or DROP? (canonical eq: `l7_linf_sharpening_defect_in_smoothing_flow_v1`)

- **Measured (FEED-ly, n600 per-stage monitor, [macOS-CPU advisory]):** per-stage
  d_seg descends CE **0.00576** → tau_softplus **0.00417** → l7 **0.00405**. The l7
  stage moves d_seg by only **−0.00012** = the measured **d_seg-DECOUPLING**.
- **Interpretation (FEED-ly unified-flow correction):** l7 is an **L∞ sharpening**
  applied inside a **curvature-SMOOTHING** flow — it sharpens the WRONG functional,
  so it barely lowers d_seg (a DEFECT, not a lever).
- **Wire-in #224 decision:** whether the optimal-form curriculum should KEEP the l7
  stage. The measured decoupling argues the l7 budget is better spent elsewhere
  (curvelet finest-scale / margin-saliency / render-AA). Re-measure with/without l7
  on the optimal-form run before committing.

## Decision 2 — activation: step_basis vs hosc (canonical eq: `step_basis_stability_vs_hosc_saturation_v1`)

- **Measured/theory (FEED 2026-06-24e activation research + hosc_beta anneal unit test):**
  both `hosc` and `step_basis` (Σ aₖ·tanh(gₖ(x−cₖ))) are **step-native** (square-wave
  partition shape, NO Gibbs overshoot) → both fit the argmax target better than
  sinusoidal SIREN/FINER. BUT `hosc` **SATURATES gradients at large β** (trainability
  risk; finite hosc_beta knob + LINEAR/COSINE anneal, unit-tested monotone), while
  `step_basis` is trainability-STABLE. Theory d_seg-fit rank: hosc > step_basis > fkan
  > FINER≈SIREN. **n600 d_seg screen is UNMEASURED.**
- **Wire-in #224 decision:** which step-native activation `proven_base` should adopt.
  hosc is the theory-best d_seg fit but carries the β-saturation trainability risk;
  step_basis trades a little fit for stability. **Run the n600 activation screen
  (hosc/step_basis/finer_gauss/fkan) to MEASURE the d_seg rank BEFORE the launch
  choice** — do not adopt on theory alone (allergic-to-non-n600 discipline).

## How to consult these from the equations leg

```
.venv/bin/python tools/list_canonical_equations.py \
  --equation-id l7_linf_sharpening_defect_in_smoothing_flow_v1 --verbose
.venv/bin/python tools/list_canonical_equations.py \
  --equation-id step_basis_stability_vs_hosc_saturation_v1 --verbose
```
Each equation's `domain_of_validity.launch_config_decision` carries the same flag
machine-readably. DAG cite: `FEED-ly` + `FEED 2026-06-24e` in
`.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
