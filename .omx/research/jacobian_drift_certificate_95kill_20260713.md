# Task #454b outcome: the Jacobian-drift term is derived and locally faithful, but direct full-costate reuse is BLOCKED and its faithful HVP formulation is an economic NO-GO.

Date: 2026-07-13 UTC
Lane: `lane_jacobian_drift_certificate_95kill_20260713`
Authority: `[macOS-CPU advisory; Torch-fp32 training signal; numpy-fp32 d_seg shadow]`
`research_only=true`; `score_claim=false`; `promotion_eligible=false`; `pointer_moved=false`

## One-line outcome

**DERIVED=yes; rigorous reuse=`0` versus inherited empirical `1/64`; post-hoc sampled-ray prefixes=`[6/22, 10/21, 0/21]` for `[early,boundary,late]`; faithful drift cost is at least `2.552848896484399` matched through-R validation-equivalents per corrected step and at least `[16.32124488283467, 26.430512487347823]` per early/boundary anchor at those prefixes versus baseline `8.375`; cross-regime fresh-control-dominating descent=`NO` because late's first sampled point is one d_seg pixel worse than fresh despite improving current CE/d_seg; verdict=`BLOCKED` rigorous certificate + `NO-GO` faithful/collinear measured formulations within the scope below.**

## Verdict and scope

**DERIVED / theorem: `YES`.** The missing first-order term is

`c_a(h) = (D J_a[h])^T q_a`, where `h=x_t-x_a`.

The conditional self-adjusting radius is implemented. It is not populated with invented curvature constants.

**Rigorous direct reuse: `BLOCKED`, not falsified.** `verdict_scope=conditional direct full-input-costate certificate; frozen SegNet training-signal Jacobian; one coercive scorer-input L2 norm; current repository bound custody.` No custody-bearing whole-ball `Lip(DJ)` upper bound, fixed-activation-cell or semismooth replacement theorem, coercive margin/Fisher-to-L2 conversion, correction numerical-error bound, current corrected renderer-gradient floor, or integrated correction-derivation custody exists. Therefore the measured rigorous reuse count is `0`.

**Faithful per-step HVP formulation: `NO-GO`.** `verdict_scope=pair0; sealed early/boundary/late saved regimes; registered task-454 exact-gradient candidate ray; faithful fixed-adjoint HVP on each actual through-R displacement; matched one-step CE/numpy-fp32 d_seg shadows; macOS-arm64 CPU Torch-fp32 advisory; no live trainer.` The correction is locally faithful, but late has no fresh-control-dominating prefix and the optimistic incremental cost already exceeds the `8.375` target at the observed high-reuse early/boundary prefixes.

**One-HVP collinear formulation: `NO-GO`.** `verdict_scope=same three rays; one anchor renderer JVP plus one anchor fixed-q HVP scaled by candidate fraction.` Its median unexplained displacement fractions are `[0.9998776886940903, 0.9995586817579648, 0.9998358997582382]`; the through-R path is not linear enough for the scaled correction to inherit the faithful arm.

This is not a trust-region-family kill, not a surrogate-family kill, and not an evaluator verdict. The costate/correction remains training signal only; exact `d_seg`/`d_pose` authority is unchanged.

## STORES CONSULTED

Full `CLAUDE.md`; full `AGENTS.md`; full `docs/operating_manual_craft_handoff.md`; v7.5 §8 and v8 canonical specifications; top operator and Codex memory entries; latest sister Codex findings/session summary and Claude design/council memos; canonical frontier, lane, task, subagent, equation, probe-outcome, and DAG surfaces; task-454 source receipt and source bundle; task-455 on-policy memo/receipts; task-456 exact-forward memo and one-thread control receipt; pair-0 early/boundary/late checkpoints; frozen SegNet weights; GT cache; current mechanism/DSL/equation/probe/test bytes; terminal measurement receipt and post-run hardening receipt.

Deliberately not actuated: cloud, paid provider, GPU, MPS, protected live runs, `upstream/evaluate.py`, archive mutation, submission/pointer mutation, or sibling-owned surrogate/forward code.

## Derivation

Let `F(x)` be the frozen SegNet logits used only as a training signal, `J_x=D F(x)`, `q_x` the upstream logit adjoint, and

`p_x = J_x^T q_x`.

At anchor `a`, let `h=x-a`. Directly reusing `p_a` owes both terms in the exact identity

`p_x-p_a = (J_x^T-J_a^T)q_x + J_a^T(q_x-q_a)`.

The second term is adjoint drift, handled conditionally by task #454. For the first term, use the fixed-anchor-adjoint correction

`c_a(h) = (D J_a[h])^T q_a`

and estimator

`p_hat_x = p_a + c_tilde_a(h)`.

Assume on the complete admitted ball, in one norm and its dual:

- `||J_a|| <= B_J`;
- `||D J_a[v]|| <= B_H ||v||`;
- `||D J_z-D J_a|| <= L_H ||z-a||`, where `L_H=Lip(DJ)`;
- `||q_a|| <= Q_a` and `||q_z-q_a|| <= L_q ||z-a||`;
- `||c_tilde_a(h)-c_a(h)|| <= L_c ||h||`.

Taylor's integral remainder gives

`J_x = J_a + D J_a[h] + R_J(h)`, with `||R_J(h)|| <= (L_H/2)||h||^2`.

Adding and subtracting the anchor-adjoint correction yields

`p_x-p_hat_x = J_a^T(q_x-q_a) + (D J_a[h])^T(q_x-q_a) + R_J(h)^T q_x + (c_a-c_tilde_a)`.

For `r=||h||`, this proves the computable envelope

`E(r) = (B_J L_q + L_c)r + (B_H L_q + L_H Q_a/2)r^2 + (L_H L_q/2)r^3`.

The task-454-composed specialization retains its already-derived adjoint envelope:

`A_adj(r) = (J0+beta*r)*kappa*(J0*r + beta*r^2/2)`,

so

`E_454(r) = A_adj(r) + (L_H Q_a/2)r^2 + L_c r`.

`Lip(J)` alone bounds uncorrected Jacobian drift at first order. A corrected quadratic remainder needs `Lip(DJ)`. A point HVP, fitted slope, or power-iteration estimate is not a whole-segment operator upper bound.

### Self-adjusting safe radius

Let `K_x` be the renderer VJP with `||K_x|| <= B_R`, `g=K_x p_x`, and `g_hat=K_x p_hat_x`. Then

`g^T g_hat >= ||g_hat|| (||g_hat|| - B_R E(r))`.

If a custody-bearing bound supplies `gamma_theta <= inf_ball ||g_hat||`, strict descent follows from

`B_R E(r) < gamma_theta`.

Define

`tau_p = gamma_theta / B_R`,

`r_error = sup {r <= r_cap : E(r) < tau_p}`,

and

`r(anchor) = min(r_error, r_geometry, r_cap)`.

The implementation finds the largest representable strict cubic root by monotone bisection; equality refreshes. `r_geometry` must be rigorous and coercive in the same L2 norm. The inherited margin/Fisher RMS is a seminorm/proxy and cannot supply this leg without a proved norm conversion.

### Frozen-SegNet nonsmoothness

The frozen scorer includes activation boundaries. A classical `C^{2,1}` ball must prove the full admitted segment remains in one compatible activation cell, or replace the theorem with an explicit semismooth/generalized-Jacobian bound. No such artifact exists. This is a certificate blocker, not a reason to insert a radius literal.

## Correction choices and their authority

1. **Faithful actual-displacement HVP.** The probe detaches `q_a` and computes `(DJ_a[h])^T q_a` for the actual through-R displacement. A full CE Hessian-vector is recorded only as an adjoint-drift diagnostic; it is not mislabeled as Jacobian drift.
2. **O(pixels) proxy.** A proxy could be cheap, but sampled curvature or margin/Fisher correlation cannot become `Lip(DJ)` without an upper-bound artifact.
3. **Power iteration.** A contracted point Hessian estimate is local and ordinarily a lower estimate of an operator supremum. It does not close the ball.
4. **One-HVP collinear scaling.** `c_a(alpha*h)=alpha*c_a(h)` is algebraically valid for a truly collinear input displacement. The measured renderer/through-R displacement is not collinear with its anchor JVP, so this cheap arm is rejected.

## Measurement custody

Primary terminal receipt: `experiments/results/jacobian_drift_certificate_20260713T034951Z/measurement_receipt.json`, SHA-256 `c1a2431ebe9df21a370748f864f2da81a5f242544051986ed341d59fe1518d48`.

The run used `[macOS-CPU advisory]`, Torch `2.12.1`, Torch threads `1` selected by the source-custodied task-456 local exact-forward control, NumPy `1.26.4`, git HEAD `752c113d1db6d1dc52c1310459a33cd20fa06c9c`, frozen SegNet SHA-256 `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`, and GT-cache SHA-256 `e3f5ce8e79374ed0b9a3f007167dd7488862b51420f0b25b7bcec7ee6865f63e`.

Each regime was atomically checkpointed. The terminal receipt source-bundles the exact uncommitted launch bytes. After measurement, the checker/DSL and probe were deliberately hardened. A terminal `--resume` then refused the source drift before mutation; the primary receipt remained byte-identical. Post-run code/test custody is separate in `postrun_hardening_receipt.json`; it does not retroactively change the measurement source.

## Measured correction and descent

The strict oracle prefix below is post hoc: it sorts sampled points by L2 displacement and stops at the first point whose corrected one-step update does not satisfy current CE descent, current d_seg nonworsening, and corrected-minus-fresh d_seg `<=0`. It is not a whole ball, arbitrary-direction proof, sequential training window, or cheap pre-admission gate.

| regime | sampled points | strict fresh-control prefix | L2 prefix radius | current CE+d_seg prefix | current prefix radius | median relative costate-error reduction | corrected CE descent | corrected d_seg nonworse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| early | MEASURED `22` | MEASURED `6` | MEASURED `1.3887383937835693` | DERIVED `17` | DERIVED from MEASURED rows `103.02526092529297` | DERIVED `1.2252125577701944%` | DERIVED from MEASURED rows `22/22` | DERIVED from MEASURED rows `17/22` |
| boundary | MEASURED `21` | MEASURED `10` | MEASURED `5.644960403442383` | DERIVED `10` | DERIVED from MEASURED rows `5.644960403442383` | DERIVED `4.802897444745857%` | DERIVED from MEASURED rows `20/21` | DERIVED from MEASURED rows `14/21` |
| late | MEASURED `21` | MEASURED `0` | MEASURED `0.0` | DERIVED `17` | DERIVED from MEASURED rows `147.25413513183594` | DERIVED `4.20763668628682%` | DERIVED from MEASURED rows `21/21` | DERIVED from MEASURED rows `18/21` |

DERIVED from all `64` MEASURED sampled rows: the correction lowers costate L2 error on `62/64`; renderer-gradient dot with fresh is positive on `64/64`; the finite corrected step decreases CE on `63/64`; and exact numpy-fp32 `d_seg` is nonworsening on `49/64`.

Late's strict prefix is zero because the smallest-L2 sampled point improves current CE by `-9.74368304014206e-05` and current d_seg by `-1.0172526041666956e-05`, but is one pixel (`5.086263020833044e-06`) worse in d_seg than the fresh-gradient control. Thus “late zero” means fresh-control dominance fails, not that the corrected step increases current d_seg.

The launch receipt's fp32 cosine calculation emitted `37/64` values slightly above `1`; those raw cosine values are not used as evidence. The current probe computes cosine in fp64 and clamps roundoff for future runs.

## Economics

Inherited baseline: MEASURED `402` validation forwards and `48` total teacher forward/backward calls; DERIVED `402/48 = 8.375` validations per teacher call.

The primary receipt's model-only denominator excludes rendering and CE, so its positional `[126.92058732418636, 120.70317321385264, 122.26929480325916]` faithful totals are pessimistic diagnostics, not an apples-to-apples baseline comparison.

The conservative matched re-derivation uses the same probe's complete through-R exact one-step validation:

- MEASURED incremental fixed-q HVP median: `3.350555353972595 s`.
- DERIVED from MEASURED row timings, matched full through-R exact-validation median: `1.3124769580317661 s`.
- DERIVED optimistic lower bound: `2.552848896484399` matched validations per corrected step.
- DERIVED early cost at strict prefix `6`: at least `16.32124488283467` validation-equivalents per anchor.
- DERIVED boundary cost at strict prefix `10`: at least `26.430512487347823` validation-equivalents per anchor.

Those are lower bounds: the HVP timer starts after anchor forward/loss/adjoint setup and omits renderer projection, gate, graph retention, and exact shadows. Both exceed `8.375`, so faithful high reuse cannot win on this substrate even before omitted costs.

The collinear arm's timing would be near or below baseline, but its measured displacement residual is approximately the displacement itself and its scaled correction is worse than the uncorrected bank on `[12/22, 8/21, 3/21]` rows. Its economics are inadmissible.

## Landed apparatus and triality

- Mechanism: `src/tac/scorer_surrogate/costate_trust_region.py` — cubic error envelope, task-454 composition, strict self-adjusting radius, fixed-adjoint HVP helper, full tensor/hash custody, and fail-closed direct decisions.
- Typed DSL: `src/tac/witness_dsl/costate_trust_region_policy.py` — default-off direct mode, no radius literal, no invented trainer argv, exact-teacher fallback, and no reuse until an integrated correction-derivation authority leg exists.
- Canonical equation: `src/tac/canonical_equations/jacobian_drift_full_costate_20260713.py`, id `jacobian_drift_full_costate_v1`, with the terminal empirical anchor.
- DAG FEED: `.omx/research/jacobian_drift_certificate_95kill_DAG_FEED_20260713.md`.
- Canonical DAG intent patch: `.omx/research/jacobian_drift_certificate_95kill_canonical_DAG_20260713.patch`, verified with `git apply --cached --check` against HEAD `752c113d1db6d1dc52c1310459a33cd20fa06c9c`. The canonical `sub015_DAG` hot-file append is `DEFERRED_MAIN_SERIALIZER_COMMIT`: this sandbox lane is explicitly non-committing, and bypassing `subagent_commit_serializer.py --patch-file` would violate the hot-file contract. Until main applies that patch, `tac.corpus_query` sees the standalone FEED as generic research rather than canonical DAG state.
- Resumable probe: `tools/probe_jacobian_drift_certificate.py` — source bundle, atomic regime checkpoints, exact through-R shadows, faithful and collinear arms, and fail-closed resume custody.
- Evidence: primary terminal receipt plus `postrun_hardening_receipt.json` for post-measurement code/test custody.
- Tests: conditional theorem, strict cubic root, HVP decomposition, stale tensor/direction/certificate rejection, policy fallback, DSL contract, analytic canaries, source custody, equation anchor, and temporary locked registry population.

The post-run checker binds anchor/target scorer inputs, displacement, anchor costate, correction, corrected costate, and certificate bytes. Binding a caller-supplied correction is still not proof that the custody-hashed HVP implementation derived it. The current policy therefore fails closed until that integrated derivation leg is built and measured.

## Sibling composition

- **#454 anchor-only path:** kept intact. It remains the safe low-validation-cadence path, but its inherited empirical margin/Fisher arm has only `1/64` reuse and no rigorous bounds.
- **#455 on-policy surrogate:** its nonlinear formulation is currently scoped `NO-GO`/`NEEDS-MORE`; this result says not to use a faithful HVP at every surrogate step as its gate. A future provider must first clear non-anchor fidelity and then may consume exact anchors under the same fail-closed custody.
- **#456 exact forward:** the selected one-thread local control was reused for timing consistency. A cheaper exact forward raises the economic bar for any HVP certificate; it does not transfer contest-CPU authority and does not prove VJP/HVP parity.

No sibling-owned surrogate or forward surface was edited.

## Reactivation edges

1. Produce a content-bound whole-ball `Lip(DJ)` upper bound plus a fixed-cell or semismooth theorem for the full scorer path.
2. Produce a rigorous coercive geometry/norm-conversion artifact and a whole-ball lower bound on the corrected renderer-gradient norm.
3. Compute the correction inside a trusted custody boundary, or emit a deterministic derivation record binding anchor, direction, scorer/runtime/model, implementation, and correction bytes; then exercise checker and DSL end to end.
4. Measure a contiguous sequence-integrated training window, not one sampled ray, with exact through-R CE/d_seg and Pose controls.
5. Re-run economics against matched full validation and require the complete operational cost to be meaningfully below `8.375`.

## Pointer delta

**POINTER DELTA: UNMOVED.** No archive was created or changed, no evaluator ran, and no contest-CPU/CUDA score was claimed. This landing is a research-only certificate theorem, negative economic characterization, and fail-closed apparatus extension.
