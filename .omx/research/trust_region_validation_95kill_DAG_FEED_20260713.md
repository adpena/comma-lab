# DAG FEED — task #454 costate trust-region validation economics

Date: 2026-07-13 UTC
Node: `FEED-454-costate-trust-region-validation-economics-20260713`
Lane: `lane_trust_region_validation_95kill_20260713`
Status: `MEASURED_FORMULATION_NO-GO`; rigorous arm `BLOCKED_MISSING_BOUND_ARTIFACTS`
Authority: `[macOS-CPU advisory; training-signal economics]`; `score_claim=false`; `pointer_moved=false`

## Edge contract

- Parents:
  - `experiments/results/yopo_first_layer_costate_probe_20260713T003635Z/receipt.json`, SHA-256 `a89585cd70b9630c90468f3a502e1efc778836cffc56ca7fb71e997fff2e6fa3`: MEASURED baseline `402` operational validation forwards, `20` operational teacher anchors, `28` measurement-only teacher calls, `48` total teacher calls, and inherited minimum costate cosines.
  - `experiments/results/segnet_validation_certificate_20260713T015633Z/receipt.json`, SHA-256 `60fe88fa1a5058d018170005890ef0720f01b31762b5e7ef0b5c7d6dc19a7d60`: MEASURED three-regime feature-margin calibration and candidate ladder.
  - `src/tac/boundary_math/segnet_validation_certificate.py`: dependency only; task #454 did not edit it.
- Producer:
  - `tools/probe_costate_trust_region_economics.py`
  - terminal receipt `experiments/results/costate_trust_region_economics_20260713T032000Z/measurement_receipt.json`, SHA-256 `60d76277ad02f0b0685fb369e8fbf9d11e4083fd5c34649528e963549d18c73e`
- Law:
  - `src/tac/canonical_equations/costate_trust_region_validation_20260713.py`
  - equation id `frozen_segnet_costate_trust_region_v1`
- DSL:
  - `src/tac/witness_dsl/costate_trust_region_policy.py`
  - radius is derived from anchor content and bound custody; there is no trainer radius flag.
- Candidate consumers:
  - #455 anchor selection and any future banked-costate controller may consume only a content-bound region with matching custody.
  - Live consumption remains `UNMEASURED_NOT_WIRED`; empirical `PROXY_REUSE` is advisory and must not be promoted to certificate or eval authority.

## Control edge

**DERIVED conditional theorem:** with first-block envelope

`q(r) = J0*r + beta*r^2/2`,

suffix pairwise-logit margin radius `rho_h`, suffix-costate Lipschitz upper bound `kappa`, renderer-VJP norm upper bound `B_R`, and a lower bound `gamma_theta` on the norm of the banked renderer gradient, admit a rigorous reuse only when

`q(r) <= rho_h` and `B_R*(J0 + beta*r)*kappa*q(r) < gamma_theta`.

This edge is for YOPO's current-prefix VJP with an exact banked suffix costate. Directly reusing a full input costate requires an additional Jacobian-drift-times-anchor-costate error term and is outside this equation's certificate scope.

The stable margin-ball inverse is

`r_margin = 2*rho_h / (J0 + sqrt(J0^2 + 2*beta*rho_h))`.

The margin/Fisher field is an O(pixels) empirical membership statistic. Its measured Pearson `0.978` association with Fisher curvature is inherited evidence, not a rigorous upper bound.

## Measured transition

- Baseline-normalized validations: `402/48 = 8.375` per total teacher call.
- New operational validation cadence: `3/3 = 1.0` per anchor.
- DERIVED reduction: `8.375x`, fraction `0.8805970149253731` (`88.05970149253731%`).
- Empirical reuse counts across `[early, boundary, late]`: `[1, 0, 0]` over `[22, 21, 21]` candidates.
- The admitted early reuse changed exact CE by `-1.1175870895385742e-08` and exact `d_seg` by `0.0`; both are fresh-shadow MEASURED.
- Rigorous arm: `BLOCKED`, because no content-bound suffix pairwise-logit/costate bounds, renderer-VJP upper bound, or projected-gradient floor artifact exists.
- Empirical arm: `NO-GO`, `verdict_scope=formulation; pair0; three sealed saved regimes; registered ladder; macOS-CPU advisory; no live trainer`, because boundary and late admit no reuse. The accepted proposal was not unsafe. The trust-region family remains open.

## Reactivation edges

1. Supply the missing rigorous bound artifacts and rerun the same terminal receipt gate; or
2. derive a different cheap input metric that admits at least one reuse in every registered regime while every accepted candidate preserves fresh exact-teacher descent and exact `d_seg`; then
3. measure sequence-integrated whole-step economics before any training-policy promotion.
