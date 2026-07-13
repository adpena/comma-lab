# DAG FEED — task #454b Jacobian-drift direct full-costate certificate

Date: 2026-07-13 UTC
Node: `FEED-454b-jacobian-drift-direct-full-costate-20260713`
Lane: `lane_jacobian_drift_certificate_95kill_20260713`
Status: `RIGOROUS_BLOCKED`; faithful and collinear measured formulations `NO-GO`
Authority: `[macOS-CPU advisory; Torch-fp32 training signal; numpy-fp32 d_seg shadow]`; `research_only=true`; `score_claim=false`; `pointer_moved=false`

## Parent edges

- Task #454 terminal receipt: `experiments/results/costate_trust_region_economics_20260713T032000Z/measurement_receipt.json`, SHA-256 `60d76277ad02f0b0685fb369e8fbf9d11e4083fd5c34649528e963549d18c73e`. Supplies MEASURED baseline counts `402/48`, DERIVED ratio `8.375`, and inherited empirical `1/64` proxy reuse; its theorem covers current-prefix VJP plus banked suffix costate only.
- Task #456 local thread control: `experiments/results/segnet_exact_forward_20260713T020000Z/receipt.json`, SHA-256 `3b04a40c7c9e656cfc417dc60f2b73781e251a21fa02689a9e78523218ad3134`. Supplies MEASURED local Torch thread count `1`; no VJP, HVP, contest-CPU, or evaluator authority transfers.
- Frozen SegNet SHA-256 `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`; GT cache SHA-256 `e3f5ce8e79374ed0b9a3f007167dd7488862b51420f0b25b7bcec7ee6865f63e`; sealed pair-0 early/boundary/late renderer checkpoints.

## Producer and evidence

- Producer: `tools/probe_jacobian_drift_certificate.py`.
- Terminal measurement: `experiments/results/jacobian_drift_certificate_20260713T034951Z/measurement_receipt.json`, SHA-256 `c1a2431ebe9df21a370748f864f2da81a5f242544051986ed341d59fe1518d48`.
- Post-run apparatus custody: `experiments/results/jacobian_drift_certificate_20260713T034951Z/postrun_hardening_receipt.json`. This is implementation/test evidence only and does not rewrite terminal measurements.
- Memo: `.omx/research/jacobian_drift_certificate_95kill_20260713.md`.

## Triality edges

- Law: `src/tac/canonical_equations/jacobian_drift_full_costate_20260713.py`, equation id `jacobian_drift_full_costate_v1`.
- DSL: `src/tac/witness_dsl/costate_trust_region_policy.py`, `DirectFullCostatePolicy`; default off, no invented flags, exact refresh fallback.
- Mechanism: `src/tac/scorer_surrogate/costate_trust_region.py`.
- DAG consumer status: research-only; no live trainer, surrogate, exact-forward, bit allocator, or autopilot actuation.
- Canonical DAG query status: `DEFERRED_MAIN_SERIALIZER_COMMIT`. The exact intent patch is `.omx/research/jacobian_drift_certificate_95kill_canonical_DAG_20260713.patch`, and `git apply --cached --check` passes against launch HEAD. This lane cannot commit; the shared hot file must be landed by main through `subagent_commit_serializer.py --patch-file`. Until then, this standalone FEED is durable generic-research state, not a `tac.corpus_query` DAG-store row.

## Control law

For `p_x=J_x^Tq_x`, `h=x-a`, and `p_hat_x=p_a+(DJ_a[h])^Tq_a`, the DERIVED whole-ball envelope is

`E(r)=(B_J L_q+L_c)r+(B_H L_q+L_H Q_a/2)r^2+(L_H L_q/2)r^3`.

With renderer-VJP upper bound `B_R` and whole-ball corrected-gradient floor `gamma_theta`, the self-adjusting strict radius is

`r*=min(sup{r<=r_cap:E(r)<gamma_theta/B_R}, r_geometry, r_cap)`.

No radius literal is admitted. Missing or stale curvature, geometry, norm, correction, descent, or tensor custody refreshes to the exact teacher.

## Measured transition

- Rigorously certified reuses: MEASURED `0`.
- Strict post-hoc sampled-ray prefixes `[early,boundary,late]`: MEASURED `[6/22,10/21,0/21]`; not balls or operational gates.
- Current CE+d_seg descent prefixes: DERIVED `[17,10,17]` from exact receipt rows.
- DERIVED from MEASURED rows: correction lowers costate error on `62/64`; corrected renderer-gradient dot is positive on `64/64`; corrected CE descends on `63/64`; corrected exact numpy-fp32 d_seg is nonworsening on `49/64`.
- Incremental HVP median: MEASURED `3.350555353972595 s`.
- Matched full through-R exact-validation median: DERIVED from MEASURED row timings `1.3124769580317661 s`.
- Optimistic HVP lower bound: DERIVED `2.552848896484399` validation-equivalents per corrected step.
- Faithful early/boundary totals at the strict prefixes: DERIVED lower bounds `16.32124488283467` and `26.430512487347823` per anchor versus baseline `8.375`; economic `NO-GO` before omitted costs.
- Collinear median unexplained displacement fractions: DERIVED `[0.9998776886940903,0.9995586817579648,0.9998358997582382]`; formulation `NO-GO`.

## Blockers and verdict scope

Rigorous blocker set: no whole-ball `Lip(DJ)` custody; no full-SegNet fixed-cell/semismooth theorem; no coercive inherited-geometry conversion; no correction numerical-error bound; no whole-ball corrected-gradient floor; no integrated proof that the bound correction bytes were derived by the trusted implementation for the bound direction.

`verdict_scope=pair0; three sealed saved regimes; one registered exact-gradient candidate ray per regime; local Torch/macOS CPU; fixed-adjoint first-order correction; matched one-step exact CE/numpy-fp32 d_seg shadows; no contiguous live window, Pose replay, contest axis, or evaluator.`

## Consumer edges

- #454 anchor-only validation remains unchanged and is not promoted.
- #455 must not use per-step faithful HVP gating; any future provider consumes only exact, content-bound anchors after its own cross-regime fidelity gate.
- #456's cheaper exact-forward control increases the economic bar; no forward/VJP equivalence is inferred.
- Unified solver/bit allocator/autopilot: `research_only=true`; no dispatch hook until rigorous bounds, integrated correction custody, and sequence economics close.

## Reactivation

Supply all rigorous artifacts, integrate correction derivation into the custody boundary, exercise the checker/DSL in a fresh source-bundled receipt, and demonstrate a contiguous cross-regime window whose complete cost is meaningfully below `8.375` while exact through-R CE/d_seg and Pose controls remain nonworsening.
