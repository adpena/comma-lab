# Task #454 outcome: validation frequency is cut 8.375×, but the empirical region is a scoped NO-GO and the rigorous certificate remains blocked on missing bound artifacts.

Date: 2026-07-13 UTC
Lane: `lane_trust_region_validation_95kill_20260713`
Authority: `[macOS-CPU advisory; training-signal economics]`; `score_claim=false`; `promotion_eligible=false`; `rank_or_kill_eligible=false`
Review status: `self-audited; UNREVIEWED_BY_MAIN`

## Verdict

**MEASURED / empirical formulation: `NO-GO`.** `verdict_scope=formulation; pair0; sealed early, boundary, and late saved regimes; exact anchor direction; registered 64-candidate ladder; empirical margin-Fisher RMS first-block envelope; macOS-CPU advisory; no live trainer.` The cheap gate admitted `1/64` proposals: `[1, 0, 0]` in early/boundary/late. The one early acceptance preserved fresh exact-teacher CE descent and exact `d_seg`, but boundary and late admitted no reuse. This is a coverage/economics failure, not an unsafe-accept failure and not a family kill.

**DERIVED / rigorous certificate: `BLOCKED`, not falsified.** The requested first-block Jacobian Lipschitz information plus an output margin field is insufficient by itself to prove either suffix label-cell stability or exact-teacher descent. A positive rigorous certificate additionally needs content-bound suffix pairwise-logit and suffix-costate Lipschitz upper bounds, a renderer-VJP norm upper bound, and a positive projected-gradient floor. None exists as a custody-bearing bound artifact in the consulted state.

**UNMEASURED:** sequence-integrated live-trainer reuse, full-step wall-clock, cross-pair/cross-seed coverage, PoseNet response outside the inherited accepted-candidate control, contest-CPU/CUDA behavior, and archive score.

## STORES CONSULTED

Full `CLAUDE.md`; full `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; `PROGRAM.md`; `docs/vehicle_operating_system.md`; the v7.5 and v8 canonical specifications; top-10 operator memory entries; unified corpus surfaces for research/equations/memory/DAG/council/tasks/docs; current lane, subagent, active-claim, and frontier surfaces; latest sister Codex findings/session summary and Claude design/council memos; the source-custodied YOPO and validation-certificate receipts; the exact frozen CPU SegNet, saved pair-0 regimes, GT cache, and current source/test surfaces. Deliberately not actuated: paid/cloud providers, protected live runs, MPS, `upstream/evaluate.py`, archive mutation, score-pointer mutation, or the live trainer.

## Why the certificate needs more than the requested two ingredients

Let `x0` be an anchor scorer input, `h=f(x)` the first-block feature, `p(h)` the suffix costate, and `R(theta)` the renderer. This is the YOPO cut used by the inherited provider: the cheap current prefix VJP `J_f(x)^T p(h0)` is recomputed each step while the exact suffix costate `p(h0)` is banked. Assume the following are actual neighborhood upper/lower bounds, not fitted local estimates:

- `||J_f(x0)|| <= J0` and `||J_f(x)-J_f(x0)|| <= beta*||x-x0||`;
- suffix pairwise-logit bounds that induce a feature margin radius `rho_h`;
- `||p(h)-p(h0)|| <= kappa*||h-h0||`;
- `||J_R(theta)^T|| <= B_R`;
- the banked renderer-gradient norm has lower bound `gamma_theta > 0` throughout the ball.

Taylor's theorem gives the first-block envelope

`q(r) = J0*r + beta*r^2/2`.

The largest radius forced inside the suffix margin ball is inverted without quadratic cancellation:

`r_margin = 2*rho_h / (J0 + sqrt(J0^2 + 2*beta*rho_h))`.

The induced scorer-input costate error is bounded by

`E(r) = (J0 + beta*r)*kappa*q(r)`.

The renderer-projected banked direction remains a strict exact-teacher descent direction wherever

`B_R*E(r) < gamma_theta`.

Indeed, if `g_b` is the banked renderer gradient and `g` the exact gradient, then `||g-g_b|| < gamma_theta <= ||g_b||` implies `g dot g_b > 0`. Direct reuse of a full input costate is a different mechanism: its error bound also owes `||(J_f(x)-J_f(x0))^T p(h0)||`. This certificate intentionally does not cover that mechanism without the extra term.

Therefore the authoritative radius is the intersection

`r(anchor) = sup {r <= r_margin : B_R*E(r) < gamma_theta}`.

This law is implemented in `tac.scorer_surrogate.costate_trust_region`. `rigorous_upper_bound` custody can emit `CERTIFIED_REUSE`; a local fit or correlation can emit only `PROXY_REUSE`. Missing or mismatched custody fails closed to a full exact-teacher refresh.

The margin field supplies the cheap curvature statistic

`Fhat(m) = exp(-|m|)/(1 + exp(-|m|))^2`,

computed in O(pixels). The inherited `Pearson=0.978` versus Fisher curvature is **MEASURED elsewhere** and useful for a training-signal proxy, but correlation is not an upper-bound proof. The local empirical arm therefore fits `J0,beta` using two prefix-only probes—the number of probes is DERIVED from the two envelope coefficients—and checks a margin-weighted Fisher RMS displacement each step without a deep frozen-SegNet forward.

## Primary measurements and economics

Primary terminal receipt: `experiments/results/costate_trust_region_economics_20260713T032000Z/measurement_receipt.json`, SHA-256 `60d76277ad02f0b0685fb369e8fbf9d11e4083fd5c34649528e963549d18c73e`. A terminal `--resume` rechecked immutable source/input custody and preserved this SHA-256 byte-for-byte.

Baseline receipt: `experiments/results/yopo_first_layer_costate_probe_20260713T003635Z/receipt.json`, SHA-256 `a89585cd70b9630c90468f3a502e1efc778836cffc56ca7fb71e997fff2e6fa3`.

- **MEASURED baseline:** `402` operational validation forwards, `20` operational teacher forward/backward anchors, `28` measurement-only teacher forward/backward controls, `48` total teacher calls, `28` step rows.
- **MEASURED new operational path:** `3` exact anchor validations for `3` anchors = `1.0` validation per anchor. Fresh exact shadows are controls and are not charged to the proposed operational path: `1` shadow, `4` actual probe exact forwards including controls.
- **DERIVED apples-to-apples normalization:** baseline `402/48 = 8.375` validations per total teacher call versus new `3/3 = 1.0` per anchor: `8.375×` lower, reduction fraction `0.8805970149253731` = `88.05970149253731%`.
- **DERIVED secondary baseline view:** `402/20 = 20.1` validations per operational teacher anchor versus new `1.0`; reported separately because the denominator excludes the baseline's measurement-only teacher calls.
- **MEASURED inherited fidelity floor:** minimum banked/exact costate cosine `0.9998774504768612` global, `0.9998451425983044` boundary-annulus, `0.9999437090802523` renderer-gradient.

Per-regime empirical results:

| regime | candidates | derived proxy radius | accepted | fresh exact result |
|---|---:|---:|---:|---|
| early | 22 | `0.001195805008266076` margin-Fisher RMS | 1 | CE `0.014437224715948105 -> 0.01443721354007721` (`delta=-1.1175870895385742e-08`); `d_seg=0.004608154296875 -> 0.004608154296875` |
| boundary | 21 | `0.002362621224986989` margin-Fisher RMS | 0 | `NO-GO_NO_REUSE`; no exact accepted-candidate shadow owed |
| late | 21 | `0.00011931876968281038` margin-Fisher RMS | 0 | `NO-GO_NO_REUSE`; no exact accepted-candidate shadow owed |

The accepted early proposal was candidate `21`, fraction `4.76837158203125e-09`, with measured displacement `0.0009983796768615872 < 0.001195805008266076`. Its exact `d_seg` did not worsen. Its inherited Pose control did not worsen, but this probe does not claim independent full-sequence PoseNet authority.

Raw counts `402 -> 3` are intentionally not described as `134×`: the runs contain different anchor counts. The verdict uses the normalized `8.375 -> 1.0` comparison.

## Landed apparatus and triality

- Mechanism: `src/tac/scorer_surrogate/costate_trust_region.py` — conditional rigorous safe ball, empirical proxy, O(pixels) membership test, fail-closed custody, and normalized economics.
- DSL: `src/tac/witness_dsl/costate_trust_region_policy.py` — typed `anchor_only` cadence, no radius literal or invented trainer flag, full-teacher fallback, and empty `live_trainer_argv`.
- Canonical equation: `src/tac/canonical_equations/costate_trust_region_validation_20260713.py`, id `frozen_segnet_costate_trust_region_v1`.
- DAG FEED: `.omx/research/trust_region_validation_95kill_DAG_FEED_20260713.md`.
- Resumable probe: `tools/probe_costate_trust_region_economics.py` — atomic per-regime receipt checkpoints and byte-stable terminal resume.
- Tests: mechanism, DSL, probe controls, equation-to-receipt re-derivation, and append-only temporary registry population.

The sibling-sealed `onpolicy_costate.py`, `segnet_validation_certificate.py`, and `instant_projected_adjoint.py` were dependencies only and were not edited by task #454.

## Pointer delta and next admissible edge

**POINTER DELTA: UNMOVED.** The currently recorded contest-CPU defensive-bank row is `0.18804439798807521`, archive SHA-256 `196acd18e4ca10a3ab0d826436aa46014a44cba8a55eb4abf9931876cc7e98b5`, and is explicitly borrowed/non-submission custody. Task #454 produced no archive and ran no evaluator; it cannot move any pointer or evidence axis.

The next admissible #454-family action is not another radius tweak. It is either (a) construct the missing content-bound suffix/renderer bounds and exercise the rigorous arm, or (b) replace the empirical metric with a derived cheap statistic that admits reuse in all three registered regimes under the same fresh exact-shadow falsifier. Only after that should a sequence-integrated trainer measurement be wired.
