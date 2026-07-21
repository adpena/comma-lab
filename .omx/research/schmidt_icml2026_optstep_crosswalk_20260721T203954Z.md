---
schema: pact.schmidt_icml2026_optstep_crosswalk.v1
utc: 2026-07-21T20:39:54Z
lane_id: lane_schmidt_icml2026_optstep_crosswalk_20260721
research_only: true
execution_authority: false
score_claim: false
promotion_eligible: false
axis: "[research-only; source/repository/custody audit]"
pointer: "0.1910828242 [contest-CPU] UNMOVED"
main_landing_review_required: true
---

# Schmidt ICML 2026 optimizer-step crosswalk to the live witness stack

## Executive verdict

**One-line verdict:** **`NO-VERDICT_DATA_CUSTODY; ADOPT-INSTRUMENTATION-ONLY`** for a
within-event-segment optimizer-step instrument; **no adaptive step law is adopted, no backtest was
executed, and the event-first-hybrid v9/v9-CGauge stage schedule is not replaced.**

The prompt's claim that the live vehicle still uses only the hardcoded PR95 stage schedule is stale.
The current v9/v9-CGauge scheduler is an **event-first hybrid**:
`EventTriggeredCurriculum`, `WitnessNativeMorseContinuationSchedule`, `ExitEvent`,
`TauAdvanceEvent`, `BirthCompletionEvent`, and `CurriculumReanchorLevers` provide real event
machinery, while fixed backstop caps, static Muon placement, one-shot ordering, global cadence/reset
approximations, and build-owed exit/repetition branches remain explicit. The sealed #205
CE→tau→l7→Muon run is the fixed legacy control, but it is not the only fixed-clock residue. The
genuine unresolved core question is narrower: **inside one fixed event segment, with loss weights
and the incumbent optimizer state/formulation held fixed, does rescaling the fully realized
incumbent update reduce objective debt per wall-second?** Raw-GD AdGD/Polyak and stateful
Schedule-Free are adjacent optimizer-formulation forks, not answers to that held-incumbent question.

The ranked crosswalk contains **[DERIVED] 24 row-primary labels**: **[DERIVED] 1** real `ADOPT`
(probe instrumentation only), **[DERIVED] 8** `ALREADY-HAVE-BETTER`, and **[DERIVED] 15**
`N-A-WHY` rows. A row may preserve an existing partial surface while its primary disposition remains
`N-A-WHY`; the count is not a technique-level closure. `N-A-WHY` is scoped to the current
formulation/custody and never closes an optimizer family. The contest pointer remains
**0.1910828242 [contest-CPU] UNMOVED**.

### Evidence labels

- **CITED**: statement follows the linked tutorial/paper/source.
- **MEASURED**: read from a content-hashed artifact in this memo.
- **DERIVED**: arithmetic or logic is shown.
- **OBSERVED**: a named repository source/config surface was inspected; no empirical-effect claim.
- **PROPOSED**: preregistered design, not an implementation or empirical result.
- Page, task, catalog, equation, schema, epoch, and file identifiers are names, not score claims.

## Stores consulted

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; `docs/operating_manual_craft_handoff.md`; SPEC v7.5 §8;
SPEC v8; `reports/latest.md`; `.omx/state/lane_registry.json`;
`.omx/state/subagent_progress.jsonl`; the latest sister findings/session summary, v9 design, council,
and operator-directive memos; the official tutorial and its complete local text extraction; the
primary papers listed below; `src/tac/witness_dsl/curriculum_dsl.py`;
`src/tac/witness_dsl/spec_v9_cgauge.py`; `src/tac/witness_control/`;
`src/tac/optimization/`; `src/tac/canonical_equations/`; `tools/lambda_net_backtest.py`;
`tools/costate_organ_elevation_backtest.py`; `tools/n205_full_run_diagnostics.py`; the #205
trajectory/snapshot; the mod32cap launch, result, costate log, and preserved resume archives; and the
two C2 decomposition directories named in §4.

## 1. Source custody and tutorial thesis

| Object | Custody | Result |
|---|---|---|
| Official tutorial | `https://www.cs.ubc.ca/~schmidtm/Documents/2026_ICML_Tutorial.pdf`; **MEASURED** 14,225,714 bytes; SHA-256 `567617699196c7a5f6346374a89cd901421e33609390b62d12cca158e406d963`; **MEASURED** 209 pages | HTTPS stream fetched successfully on 2026-07-21. |
| Ephemeral text extraction | **MEASURED** 97,247 bytes; 3,038 lines; SHA-256 `7ac5f975af582f2f278d62663afc52bc62632356fc8a707eeb61fe20858bc2dd` | The complete extraction was read. A fresh `pdftotext` stream from the official PDF produced the same SHA-256. The transient extraction path is deliberately not a durable citation. |
| Thesis | Tutorial p. 6 | **CITED:** proof-complete PEP, coin-betting, and non-asymptotic BFGS have had little practical effect, while proof-missing Adam/AdamW/cosine/Muon changed practice. This is a governance observation, not proof that theory is useless. |
| Practical scoreboard | Tutorial pp. 8, 120 | **CITED:** Schedule-Free was the standout recent black-box optimizer result, but still requires base-LR/momentum/weight-decay choices. It removes a horizon-dependent decay schedule; it does not remove step-size selection. |

Primary sources (all accessed 2026-07-21 UTC; stable IDs and version/commit pins where available are
mirrored in the receipt):
[local/glocal smoothness](https://openreview.net/forum?id=be9PdukwEL),
[AdGD v2](https://arxiv.org/abs/1910.09529v2),
[Polyak 1963](https://www.mathnet.ru/eng/zvmmf7813),
[Polyak 1969 nonsmooth extension](https://doi.org/10.1016/0041-5553(69)90061-5),
[D-Adaptation](https://proceedings.mlr.press/v202/defazio23a.html),
[Prodigy](https://proceedings.mlr.press/v235/mishchenko24a.html),
[Schedule-Free v4](https://arxiv.org/abs/2405.15682v4),
[MiniCPM/WSD v3](https://arxiv.org/abs/2404.06395v3),
[WSD analysis v3](https://arxiv.org/abs/2410.05192v3),
[sign-gradient momentum v1](https://arxiv.org/abs/2304.13960v1),
[signSGD v3](https://arxiv.org/abs/1802.04434v3),
[Lion v4](https://arxiv.org/abs/2302.06675v4),
[coin betting v4](https://arxiv.org/abs/1602.04128v4),
[COCOB v3](https://arxiv.org/abs/1705.07795v3),
[PEP v1](https://arxiv.org/abs/1206.3209v1),
[non-asymptotic BFGS v4](https://arxiv.org/abs/2003.13607v4), and the
[official Muon implementation](https://github.com/KellerJordan/Muon) at commit
`f98f1cacc0263b04290753e32be8d498c1efc806`.

## 2. Exhaustive technical digest

The formulas below use `g_k = ∇f(w_k)`. Claims are source-scoped; none is silently promoted to the
nonconvex, changing-objective witness trainer.

| Method/result | Exact statement recovered | Assumptions and limits | Tutorial/source |
|---|---|---|---|
| GD and descent lemma | **CITED:** `w_{k+1}=w_k-α_k g_k`, and for an `L`-smooth objective, `f(w_k-αg_k) ≤ f(w_k)-α(1-αL/2)||g_k||²`. Therefore, when `g_k≠0`, `0<α<2/L` guarantees strict descent and the bound's decrease is maximized at `α=1/L`. | `L` bounds the gradient Lipschitz constant along the whole proposed segment. `||H(w_k)||` at one point is not by itself a finite-step certificate. | Tutorial pp. 27–31; Fox et al. |
| Local sharpness | **CITED:** replace the global bound by a valid local bound `L_k`; then `0<α<2/L_k` is the strict-descent range and the descent-bound optimum is `α=1/L_k`. `2/L_k` is the zero-decrease boundary, not a safe value. | The local bound must cover the accepted segment, not just the current point. | Tutorial pp. 38–42. |
| Directional quadratic step | **DERIVED:** for local quadratic model `q(α)=f_k+αg_kᵀp_k+(α²/2)p_kᵀH_kp_k`, `α_Q=-g_kᵀp_k/(p_kᵀH_kp_k)` when `g_kᵀp_k<0` and `p_kᵀH_kp_k>0`. For raw GD, `p_k=-g_k`, so `α_Q=||g_k||²/(g_kᵀH_kg_k)` and descent holds for `0<α<2α_Q` in the exact quadratic. | A negative/near-zero denominator is a curvature refusal, not license to take an unbounded step. For AdamW/Muon, use the actual preconditioned direction; the raw-GD formula is not transferable by name. | Tutorial pp. 39–46; algebra shown here. |
| Smooth-until-guilty/backtracking | **CITED:** grow or retain a candidate step until a descent-model/Armijo test fails, then shrink; resets can recover larger steps after entering a flatter region. Wolfe conditions approximate line minimization while controlling curvature. | Requires extra function/gradient evaluations. Monotone Armijo can collapse numerically on neural nets; non-monotone variants relax the acceptance envelope. | Tutorial pp. 35–36, 46, 112–114. |
| Glocal smoothness | **CITED:** globally `L`-smooth and locally `L*`-smooth near the solution, with `L*≪L`; the ideal sequence uses `1/L` in the sharp region and switches to `1/L*` in the flat region. | The neighborhood/sublevel-set definition is part of the theorem. | Tutorial pp. 43–46; Fox et al. |
| Non-uniform smoothness | **CITED:** a neural-network model is `||H(w)||≤L₀+L₁||g(w)||`; clipping/normalization controls the state-dependent curvature term. A related function-gap model motivates warmup. | This is not a measured equality for the witness loss. | Tutorial pp. 48–50. |
| Polyak step | **CITED:** `α_k=(f(w_k)-f*)/||g_k||²`; `w_{k+1}=w_k-α_kg_k`. | Classical guarantees need the stated smooth-convex conditions and true optimum/valid target; the nonsmooth/subgradient extension is Polyak 1969, not silently attributed to the 1963 smooth paper. Unknown/underestimated `f*`, nonconvexity, and event-segment objective changes can overshoot. | Tutorial p. 46; Polyak 1963 (smooth); Polyak 1969 (nonsmooth extension). |
| Malitsky–Mishchenko AdGD | **CITED:** initialize `x¹=x⁰-λ₀g⁰`, `θ₀=∞`; then `λ_k=min{sqrt(1+θ_{k-1})λ_{k-1}, ||x^k-x^{k-1}||/[2||g^k-g^{k-1}||]}`, `x^{k+1}=x^k-λ_kg^k`, `θ_k=λ_k/λ_{k-1}`. | Differentiable convex objective with locally Lipschitz gradient. The secant is along the realized raw-gradient step; it is not Fisher/margin/`nᵀHn` by identity. | Tutorial pp. 46, 68; Malitsky–Mishchenko. |
| AdaGrad-Norm | **CITED:** a representative safe parameter-free form is `α_k=α/sqrt(δ+Σ_{t≤k}||g_t||²)`; it eventually behaves like a safe fixed step under smoothness but remains sensitive to its numerator. | Does not directly estimate local sharpness. | Tutorial pp. 33–34. |
| D-Adaptation | **CITED:** estimates the unknown distance `D=||x₀-x*||` online. In its dual-averaging core, `s_{k+1}=s_k+d_kg_k`, `γ_{k+1}=1/sqrt(Σ_{i≤k}||g_i||²)`, `d̂_{k+1}=[γ_{k+1}||s_{k+1}||²-Σ_{i≤k}γ_id_i²||g_i||²]/[2||s_{k+1}||]`, `d_{k+1}=max(d_k,d̂_{k+1})`, and `x_{k+1}=x₀-γ_{k+1}s_{k+1}`. | Convex Lipschitz theory, nonempty minimizer set, and an initial lower bound `0<d₀≤D`; estimates distance-to-solution, not `L_k`. | Tutorial pp. 33, 107; Defazio–Mishchenko 2023. |
| Prodigy | **CITED:** GD form `η_k=d_k²ω_k/sqrt(d_k²G²+Σ_{i≤k}d_i²ω_i²||g_i||²)`, `x_{k+1}=x_k-η_kg_k`; `d̂_{k+1}=Σ_{i≤k}η_i⟨g_i,x₀-x_i⟩/||x_{k+1}-x₀||`, `d_{k+1}=max(d_k,d̂_{k+1})`. | Convex Lipschitz theory. The practical Adam form is heuristic and still accepts a multiplier schedule, commonly cosine. “Parameter-free” is not assumption-free or schedule-free. | Tutorial pp. 33, 107, 207; Mishchenko–Defazio 2024. |
| Coin betting / COCOB | **CITED:** KT online-learning update `w_t=(1/t)(ε+Σ_{i<t}⟨g_i,w_i⟩)Σ_{i<t}g_i` for bounded outcomes; online-to-batch averaging gives a stochastic-convex optimizer. Per coordinate, COCOB-Backprop uses `w_{t,i}=w_{1,i}+[θ_{t,i}/(L_{t,i}max(G_{t,i}+L_{t,i},αL_{t,i}))](L_{t,i}+Reward_{t,i})`, with maximum gradient `L`, absolute-gradient sum `G`, reward, and cumulative gradient `θ`. | Online convex/bounded-outcome theory. COCOB-Backprop loses the theorem and uses a default `α=100`. Neither is an explicit local-sharpness estimator. | Tutorial pp. 6, 107, 207; Orabona–Pál; Orabona–Tommasi. |
| Fixed schedules | **CITED:** for `k≥1`, textbook `α/k`; robust `α/sqrt(k)`; cosine `α_k=(α/2)(1+cos(πk/T))`; linear decay `α(1-k/T)`. | Horizon-dependent and problem-dependent. | Tutorial pp. 95–97. |
| WSD | **CITED:** for warmup end `W`, stable end `T`, run end `S`, `η_s=ηs/W` for `s<W`, `η_s=η` for `W≤s≤T`, and `η_s=η f(s-T)` for `T<s≤S`, with decreasing `f`. Stable checkpoints support cooldown branches. | Empirical LLM schedule and branch protocol, not a local-sharpness law. A reported cooldown fraction is not universal. | Tutorial pp. 57–58, 88; Hu et al. §4.2 Eq. (1) for WSD; Wen et al. for later river-valley/branch analysis. |
| PEP-optimal multistep | **CITED:** Schmidt's two-step smooth-convex example uses `α₀=2/(3L)`, `α₁=4/(3L)`; long-horizon optimized scalar sequences start near `1/(2L)` and rise near `2.2/L`. | Fixed `L`, fixed horizon, smooth-convex worst-case model; not feature-evolving dynamics. The exact tutorial numbers are cited to Schmidt, not falsely attributed as a verbatim Drori–Teboulle theorem. | Tutorial p. 59; Drori–Teboulle for PEP framework. |
| Chebyshev/silver/long steps | **CITED:** Chebyshev steps exploit known quadratic spectral interval `[μ,L]`; silver/long-step cycles deliberately include steps beyond the one-step descent threshold while preserving multistep guarantees. | Quadratic/fixed-spectrum or smooth-convex worst-case assumptions and a chosen horizon/cycle. | Tutorial pp. 59, 62, 66. |
| Barzilai–Borwein | **CITED:** with `s_k=w_k-w_{k-1}` and `y_k=g_k-g_{k-1}`, the tutorial states `α_k=s_kᵀs_k/(s_kᵀy_k)`; practical use often adds a non-monotone line search. | Positive curvature and secant reliability are not automatic in nonconvex training. The BB2 alternative is omitted because it was not recovered from the cited tutorial pages in this pass. | Tutorial pp. 63, 66. |
| SGD descent/noise phases | **CITED:** an expected descent bound has a negative gradient term and a positive `O(α²Lσ²)` noise term; large steps pay in the noise-dominated phase. Under interpolation, stochastic line search can behave more like GD. | Requires the stated variance/interpolation conditions; the witness's one-video minibatches do not automatically satisfy them. | Tutorial pp. 88–114. |
| Polyak–Ruppert averaging | **CITED:** maintain the running average of iterates; averaging reduces noise and has optimal asymptotic properties under classical stochastic approximation conditions. | Offline averaging does not alter the trajectory; online/primal averaging does. | Tutorial pp. 116–120. |
| Schedule-Free | **CITED:** the basic Schedule-Free SGD recurrence initializes `z_1=x_1`, then `y_t=(1-β)z_t+βx_t`; `z_{t+1}=z_t-γ∇f(y_t,ζ_t)`; `x_{t+1}=(1-c_{t+1})x_t+c_{t+1}z_{t+1}`, basic `c_{t+1}=1/(t+1)`. `β=0` recovers Polyak–Ruppert-style averaging and `β=1` primal averaging. | Convex theorem uses a tuned `γ`; practical Schedule-Free AdamW is a distinct formulation with additional moment, bias-correction, warmup, weighting, epsilon, and decay semantics. This crosswalk selects neither variant. | Tutorial p. 120; Defazio et al. 2024 v4. |
| Heavy ball / momentum | **CITED:** `m_{k+1}=β_km_k-α_kg_k`, `w_{k+1}=w_k+m_{k+1}`. Momentum can accelerate/dampen quadratic modes but can amplify stochastic noise. | Strong quadratic theory does not directly transfer to changing nonconvex curvature. | Tutorial pp. 128–143. |
| SignGD / sign momentum | **CITED:** SignGD is `w_{k+1}=w_k-α_k sign(g_k)`. Tutorial p. 161 separately gives sign-gradient momentum `m_{k+1}=β_km_k-α_ksign(g_k)`, `w_{k+1}=w_k+m_{k+1}`. Kunstner et al. Algorithm 1 is equivalent under a sign convention when `α` is constant (or when the momentum buffer is explicitly rescaled as `α_k` changes); it is not generally identical for an arbitrary indexed schedule. This is also not Bernstein et al.'s Signum or Lion, whose update direction takes `sign(momentum)`. Under an infinity-norm descent lemma, a safe SignGD condition is `α_k≤2||g_k||₁/L∞`. | Loses magnitude; stochastic sign can be biased/noisy. | Tutorial pp. 153–171; Kunstner et al. 2023; Bernstein et al. 2018; Chen et al. 2023. |
| Muon | **CITED:** for matrix momentum `M=UΣVᵀ`, the ideal direction `UVᵀ` is operator-norm steepest descent / sign descent on singular values. Official `muon.py` at commit `f98f1cac…` normalizes `X`, forms `A=XXᵀ`, and iterates `X←aX+(bA+cA²)X` with `(a,b,c)=(3.4445,-4.7750,2.0315)` for default `ns_steps=5`. | Hidden matrix weights only; other parameters remain on AdamW. The pinned reference explicitly says the polynomial does not converge exactly to `UVᵀ`, so “exact orthogonalization” is a false claim. | Tutorial pp. 172–175; official Muon repository at pinned commit. |
| Preconditioning / natural gradient | **CITED:** `w_{k+1}=w_k-α_kP_k^{-1}g_k`; positive-definite `P_k` changes the metric and safe step. Fisher/KFAC approximate a task-relevant metric rather than the Euclidean Hessian. | A Fisher/margin field is not automatically the training-objective directional Hessian. | Tutorial pp. 177, 185. |
| Regularized Newton | **CITED:** the second-order model with Lipschitz Hessian motivates direction `p_k=-(H_k+λ_kI)^{-1}g_k` and update `w_{k+1}=w_k+p_k`. Newton is most useful when gradient noise does not dominate. | Hessian-vector cost, indefiniteness, and stochastic noise matter; Newton need not beat GD away from its regime. | Tutorial pp. 179–181. |
| BFGS | **CITED:** `s_k=x_{k+1}-x_k`, `y_k=g_{k+1}-g_k`; inverse-Hessian update `H_{k+1}=(I-s_ky_kᵀ/(y_kᵀs_k))H_k(I-y_ks_kᵀ/(y_kᵀs_k))+s_ks_kᵀ/(y_kᵀs_k)`. | The non-asymptotic unit-step result is local and assumes strong convexity, smoothness, Lipschitz Hessian, and sufficiently close iterate/metric initialization. It does not automatically cover L-BFGS, stochastic, nonconvex, or changing loss. | Tutorial p. 6; Jin–Mokhtari 2020. |
| Sampling and variance reduction | **CITED:** random reshuffling often beats with-replacement SGD empirically; importance sampling targets heterogeneous example difficulty; variance reduction can recover large fixed steps in classic finite sums but often underperforms in deep learning, with MARS a recent large-batch exception. | Technique value depends on data/model regime and equal-compute comparison. | Tutorial pp. 191–195, 207. |
| Initialization/normalization/model scaling | **CITED:** deep networks make initialization, normalization, skip connections, and parameterization part of optimization dynamics; fixed-feature theory misses feature evolution. | Architecture-specific; not a universal step law. | Tutorial pp. 197–209. |

## 3. Ranked Pact crosswalk

Ranking is by expected value of the next **honest** action, not by novelty. `ALREADY-HAVE-BETTER`
means Pact has a more witness-native implemented surface for the scoped role; it does not assert a
positive score receipt. Counts are by each row's primary label. `N-A-WHY` negatives are scoped and
name the condition that could reopen them.

| Rank | Tutorial technique | Nearest live surface | Verdict | $0 test/build, named consumer, and falsifier |
|---:|---|---|---|---|
| 1 | AdGD / measured-local-step family | #247/#303/#426/#427/#516 observers; #305 EoS literature; #205 rolling resume-state checkpoint telemetry/names; legacy mod32cap state sidecars | **ADOPT — PROBE INSTRUMENT ONLY; LAW `NO-VERDICT_DATA_CUSTODY`** | **$0 action:** build a default-off shadow candidate-step recorder/probe on copied state with output schema `pact.within_segment_candidate_step_probe.v1`. **Consumer:** `tac.probe_outcomes_ledger.register_probe_outcome`; only a later green receipt may feed a typed `WithinSegmentAdaptiveStep`. **Falsifier:** incumbent state cannot be replayed exactly, held-out candidate-loss-per-wall does not improve, safeguards fire, or a later governed fork regresses a protected through-R facet. |
| 2 | Global/local `L`, directional `1/L_k` | #318/#320 viscous-HJ CFL, #316 normalized `nᵀHn`, Fisher/margin sensors, #430 event transitions | **N-A-WHY — `FORMULATION x UNCALIBRATED_CURVATURE_LOCUS`**; existing fields are covariates, not `H_loss` | **$0 action:** feed the existing fields to Rank 1 as covariates, never as `L_k` labels. **Consumer:** `pact.within_segment_candidate_step_probe.v1`. **Falsifier:** out-of-sample observed directional loss curvature disagrees with the surrogate. |
| 3 | Polyak step | Costate target/debt estimates; preserved segment checkpoints | **N-A-WHY — `INSTANCE x UNKNOWN_VALID_FSTAR`** | **$0 action:** enable raw-gradient Polyak in Rank 1 only with a custodied segment-valid target. **Consumer:** candidate-step probe receipt. **Falsifier:** target violation, overshoot, or worse paired candidate loss. |
| 4 | SUG / Armijo / Wolfe / exact line search | Event accept/rollback machinery; spike guard; costate trust regions | **N-A-WHY — `INSTANCE x NO_CANDIDATE_EVALUATIONS`** | **$0 action:** add non-mutating candidate-loss evaluation on the actual incumbent direction and account for wall cost. **Consumer:** candidate-step probe receipt. **Falsifier:** evaluation overhead erases decrease-per-second or non-monotone acceptance destabilizes safeguards. |
| 5 | Schedule-Free | Event-first hybrid schedule; EMA; AdamW/Muon momentum | **N-A-WHY — `FORMULATION x WITHIN_SEGMENT_UNMEASURED`** | **$0 action:** preregister state initialization/burn-in, then compare a separate Schedule-Free deterministic fork inside identical fixed event segments; do not treat it as a stateless one-step candidate or replace stage events. **Consumer:** `tac.probe_outcomes_ledger.register_probe_outcome`. **Falsifier:** no matched-wall benefit or divergent resume/EMA semantics. |
| 6 | WSD | #302/#334/#339/#430 event-first hybrid, stage checkpoints, governed rollback, #269 Muon LR anneal | **N-A-WHY — `FORMULATION x FIXED_SEGMENT_WSD_UNMEASURED`** | **$0 action:** preserve current scheduler and specify a matched fixed-segment stable/cooldown branch only if later capacity exists. **Consumer:** `WitnessNativeMorseContinuationSchedule` comparison receipt. **Falsifier:** no equal-custody wall/facet benefit over event-first continuation. |
| 7 | D-Adaptation | Costate distance/debt models and n=1 prior work #499 | **N-A-WHY — `FORMULATION x CONVEX_DISTANCE_ESTIMATOR_NOT_LOCAL_SHARPNESS`** | **$0 action:** no trainer adoption; retain only as a later probe candidate after a stable target exists. **Consumer:** candidate-step probe matrix. **Falsifier:** invalid/nonmonotone distance lower estimate or no paired benefit. |
| 8 | Prodigy | Same as Rank 7 plus current AdamW base | **N-A-WHY — `FORMULATION x PRACTICAL_ADAM_HEURISTIC_UNMEASURED`** | **$0 action:** preserve it as an unselected matched-fork candidate and do not call it parameter/schedule free. **Consumer:** candidate-step probe matrix. **Falsifier:** tuned incumbent wins at equal search/compute. |
| 9 | AdaGrad-Norm | AdamW second moments; #222 beta2-from-n and rewarm law | **N-A-WHY — `FORMULATION x NUMERATOR_STILL_TUNED`** | **$0 action:** do not add a duplicate accumulator unless Rank 1 identifies incumbent LR sensitivity as binding. **Consumer:** #222 beta2/rewarm decision surface. **Falsifier:** equal-search candidate fails to improve paired loss-per-wall. |
| 10 | Coin betting / COCOB | #499 n=1 costate-model lane; canonical real-only backtest | **N-A-WHY — `FORMULATION x ONLINE_CONVEX_ASSUMPTIONS`** | **$0 action:** retain as a conceptual candidate for n=1 uncertainty, not a trainer claim. **Consumer:** `tools/lambda_net_backtest.py` if real compatible trajectories later exist. **Falsifier:** real-only walk-forward or matched trainer fork loses the incumbent. |
| 11 | PEP / silver / long steps | #302 witness-native continuation; event accept/rollback | **N-A-WHY — `FORMULATION x FIXED_L_FIXED_HORIZON`** | **$0 action:** do not paste a worst-case scalar sequence into a feature-evolving objective; first measure segment stationarity/local-model fit. **Consumer:** candidate-step probe receipt. **Falsifier:** fixed-`L` residual or horizon assumptions fail. |
| 12 | Chebyshev / BB / spectral steps | Muon spectral finisher; #423 terminal quadratic head math | **N-A-WHY — `FORMULATION x FULL_TRUNK_SECANT_CUSTODY_ABSENT`** | **$0 action:** record `Δw,Δg` before enabling BB in Rank 1. **Consumer:** candidate-step probe receipt. **Falsifier:** nonpositive/noisy secants or no held-out benefit. |
| 13 | SignGD / sign(momentum) | implemented #469 muonh/manifold-Muon and #175 MD-Decoupling; #552 spec-only with no trainer implementation/argv consumer; #556 pending | **ALREADY-HAVE-BETTER** for the implemented geometry-specific route | **$0 action:** reuse #469/#175; do not spawn generic SignGD or imply #552/#556 live authority. **Consumer:** incumbent v9 Muon finisher. **Falsifier:** matched full-facet/wall receipt favors a precisely specified generic sign formulation. |
| 14 | Muon / Newton–Schulz matrix steepest descent | implemented incumbent Muon; #552 spec-only with no trainer implementation/argv consumer; #556 pending; #269/#270 transition laws | **ALREADY-HAVE-BETTER** for incumbent matrix weights | **$0 action:** reuse the incumbent Muon implementation and keep spec-only/pending variants separately labeled. **Consumer:** v9 Muon finisher. **Falsifier:** matched full-facet/wall-clock evaluation, not proof pedigree. |
| 15 | Heavy ball / Nesterov / momentum | AdamW trunk, incumbent Muon finisher, warm-start and moment-reset laws | **ALREADY-HAVE-BETTER** | **$0 action:** no generic momentum arm; measure only a named state/metric mismatch. **Consumer:** existing AdamW/Muon optimizer state. **Falsifier:** a matched formulation resolves that mismatch and improves loss-per-wall. |
| 16 | Adam / AdamW / cosine | incumbent trunk plus event-first hybrid curriculum, #222 beta2 gate, #175/#269 transition fixes | **ALREADY-HAVE-BETTER** | **$0 action:** retain the incumbent and sealed PR95 A/B control. **Consumer:** current v9 trainer. **Falsifier:** a matched default-off candidate beats it with equal search, resume, and facet custody. |
| 17 | Glocal/NUS/clipping/warmup | event-first continuation, gradient clipping, #318/#320 adaptive viscosity/CFL, stage rewarm | **ALREADY-HAVE-BETTER** for problem-derived safeguards | **$0 action:** reuse the existing controls while Rank 1 measures the missing objective-curvature relation. **Consumer:** current event controller. **Falsifier:** calibrated objective-local curvature shows a safer/faster accepted branch. |
| 18 | Polyak–Ruppert / gradient averaging | `src/tac/witness_control/polyak_finisher.py:PolyakTailAverager` (default-off, resumable uniform tail mean); EMA remains separate | **N-A-WHY — `FORMULATION x PR_AVERAGING_UNMEASURED`** | **$0 action:** reuse the exact existing `PolyakTailAverager`; do not equate EMA/momentum with PR averaging or claim efficacy. **Consumer:** downstream byte-close candidate selection. **Falsifier:** matched candidate receipt shows no improvement or resume parity fails. |
| 19 | Preconditioning / Fisher / natural gradient | Fisher/margin geometry, implemented #175 MD-Decoupling, argv-inert `FisherNaturalSolverPolicy`; #552 remains a separate spec-only lane | **ALREADY-HAVE-BETTER** for named metric-aware roles | **$0 action:** reuse current metric-aware surfaces without claiming Fisher=`H_loss` or #552 activation. **Consumer:** existing metric-aware optimizer/policy stack. **Falsifier:** the metric fails to predict objective directional decrease in Rank 1. |
| 20 | Regularized Newton / BFGS | #341/#423 terminal head math; `spec_c1_optimal_form_20260715.py` records #423 as not argv-reachable | **N-A-WHY — `FORMULATION x FULL_TRUNK_BFGS_ASSUMPTIONS_UNMET`**; `ALREADY-HAVE-MATH / BUILD-OWED-TRAINER-CONSUMER` for the narrow head | **$0 action:** do not claim the head solve active; reopen L-BFGS only after secant predictivity and memory/wall cost are measured. **Consumer:** future #423 trainer consumer or candidate-step probe. **Falsifier:** nonlocal/nonconvex curvature, `O(d²)` rent, or no matched benefit. |
| 21 | SPS / stochastic Polyak / stochastic line search | existing disengaged SPS instance; event rollback | **N-A-WHY — `INSTANCE x DISENGAGED_OR_INTERPOLATION_UNPROVEN`** | **$0 action:** reopen only with engagement telemetry, a real segment target, and matched batch order. **Consumer:** candidate-step probe receipt. **Falsifier:** engagement remains zero or interpolation/paired benefit fails. |
| 22 | Random reshuffling / importance sampling | deterministic batch order, `HardnessOversample`, per-class/stratum difficulty surfaces | **ALREADY-HAVE-BETTER** | **$0 action:** reuse typed hardness and exact RNG custody; hold optimizer/scorer cost equal in any A/B. **Consumer:** `HardnessOversample`. **Falsifier:** matched reshuffling/importance sampling improves protected facets per wall. |
| 23 | Variance reduction / MARS | microbatch/throughput work and n=1 renderer regime | **N-A-WHY — `FORMULATION x DEEP_LARGE_BATCH_TRANSFER_UNGROUNDED`** | **$0 action:** first measure gradient-noise decomposition. **Consumer:** candidate-step probe/noise receipt. **Falsifier:** noise, rather than curvature or receiver cost, is measured as binding. |
| 24 | Initialization / normalization / skip / scaling | SIREN `ω₀` init #178, FreSh #448, FINER++ #310, rare-class structured init #208, CGauge/normal-coordinate work | **ALREADY-HAVE-BETTER** | **$0 action:** reuse the searched non-PR95 initialization set; require a distinct basin hypothesis for any new arm. **Consumer:** current typed initialization compiler. **Falsifier:** a matched fresh-start receipt from that distinct hypothesis wins. |

### Existing non-PR95 set explicitly preserved

The crosswalk does not rediscover or replace: #175 MD-Decoupling; #469 muonh/manifold-Muon; #552
SPD-submanifold momentum (spec-only; no trainer implementation/argv consumer); pending #556
FilmPolarSPDNormalMomentum; #222 beta2/window law;
#496 low-precision M+Adam; #442 WW-PGD; #269 Muon warm-start/cold-momentum/LR anneal; #302
witness-native level-set continuation; #286 geometric tau anneal; #318/#320 adaptive viscosity; #448
FreSh; #323 island-birth homotopy; #341/#423 terminal quadratic math (#423 trainer consumer
build-owed); #217 leap residual; #383
pose-finish gate; #301 task losses; #382 anisotropic sigma; #316 StEik; #178/#310 SIREN/FINER and
step-native activation; #360 in-trunk forces; and #208 rare-class structured initialization.

## 4. D3 decisive row — what the artifacts can and cannot answer

### 4.1 Custody audit

| Evidence | What is present | What is absent | Consequence |
|---|---|---|---|
| `.omx/research/n205_full_run_trajectory_20260711.jsonl`, SHA-256 `7ee89fcc3effbd0836e8745f7a13642ee29fda41d8631e2a54c7868c7f8d4660` | **MEASURED** manifest + 579 telemetry rows; scalar loss terms, scalar `gnorm`, verdicts, event rows, and checkpoint names | No parameter vectors, gradient vectors, `Δw`, `Δg`, optimizer directions, directional HVP, loss-Fisher curvature, or candidate-step reevaluations | AdGD/BB cannot be reconstructed; Polyak needs an assumed `f*`; Armijo/Schedule-Free cannot be causally replayed. |
| `.omx/research/n205_full_run_diagnostic_snapshot_20260711.json`, SHA-256 `2e00b08e6f89eabc135b6a5e47e627dd2ea0d2e48317afc243e80aea22336b18` | **MEASURED** checkpoint paths/hashes and read-only milestone diagnosis | No per-step optimizer-state series or counterfactual state | Supports custody, not alternate-step efficacy. |
| `levelset_n600_witness_mod32cap_20260706T115554Z/levelset_train_result.json`, SHA-256 `39e0324a115829bb16924d039d211303a0c9bf24f0562b48d7c221feaa9e0d12` | **MEASURED** 41 evaluator rows across epochs 0–1000 | Evaluator cadence aliases within-segment dynamics | Can compare realized incumbent milestones only. |
| Three legacy-mod32cap preserved resume archives | **MEASURED** model, EMA, optimizer moments, LR/step, epoch, recent losses, NumPy RNG, and hardness RNG; SHA-256 `6782f64c…`, `3add1de4…`, `e02eec88…` at CE299/Muon726/TauMuon1000 | No stored gradient/HVP/candidate-loss vectors; each stamps dirty git state without a dirty-patch/source hash, resume semantic schema/stage, event ledger, or resume-vs-continuous next-state digest | State-bearing legacy-mod32cap boundary-fork candidates only. Exact deterministic fork support is **UNVERIFIED** pending byte-close reload plus incumbent one-step replay parity; current v9 custody is not inferred. |
| C2 `decomp_rows.jsonl`, `sens.json`, `slope.json` | **MEASURED** receiver-space frame/pair-side `dseg`, confidence, stratum sensitivities, and response slopes | No optimizer state or training loss-curvature series | C2 is not an optimizer trajectory. Treating it as AdGD replay data would be a type error. |
| Fisher/margin source/config surfaces | **OBSERVED** named scorer-geometry, Fisher-policy, and v9 configuration objects at their own loci | No content-hashed optimizer-trajectory row or calibration proving equality to `pᵀH_loss p` | Useful covariates; not objective-sharpness authority. |
| Normalized `nᵀHn` / PDE geometry | **DERIVED** in its named PDE/energy formulation | No derivation equating that locus to the trainer objective's directional Hessian | Preserve the PDE law; require separate empirical calibration before Rank 1 treats it as a loss-curvature covariate. |

`tools/costate_organ_elevation_backtest.py:271-272` independently records that #205 is the only
compatible costate trajectory, mod32cap lacks interval-aligned per-class debt, C2 rows were pending,
and no equivalence may be inferred. That compatibility statement is about its costate schema, not a
license to reconstruct alternate optimizer paths.

### 4.2 Identifiability result

Let `Δw_inc,k` be the fully realized incumbent weight delta, including its per-group LR ratios and
decoupled weight-decay contribution. A historical scalar row can calculate a proposed rescaling or
alternate-formulation update only if the law's inputs are present. Even then, after the first
different update,
`w'_{k+1}≠w_{k+1}` implies **[DERIVED]** subsequent `g'_{k+1}`, `p'_{k+1}`, optimizer moments,
curvature, and event time generally differ. Therefore a teacher-forced replay can check candidate
one-step predictions, safeguard activations, and local-model residual; it **cannot** prove an entire
counterfactual convergence or wall-clock trajectory.

**D3 verdict:** `NO-VERDICT_DATA_CUSTODY; ADOPT-INSTRUMENTATION-ONLY`.

- **Not GREEN:** no source-faithful AdGD, Polyak, Armijo, or Schedule-Free superiority measurement
  exists.
- **Not RED:** the optimal formulations have not been measured; family remains open.
- **GREEN only for routing:** one `$0`, fail-closed measurement task is justified.
- **No schedule replacement:** the core test rescales the incumbent update inside a fixed live event
  segment. Raw-GD and Schedule-Free are separately labeled optimizer-formulation forks.

### 4.3 Proposed `$0` fail-closed shadow candidate-step recorder/probe

**PROPOSED core scope:** fixed v9 event segment and loss weights; identical batch order, seed,
model/EMA/optimizer/RNG checkpoint, and source tensors; hold optimizer formulation, moments,
per-group LR ratios, and update semantics fixed while rescaling the full incumbent delta. It is
shadow-only and mutates neither the source run nor any archive. **Adjacent fork scope:** raw-GD-only
AdGD/Polyak and stateful Schedule-Free are separate optimizer-formulation experiments, not members
of the core held-incumbent LR test.

Per sampled update it records:

1. event/segment ID, stage objective hash, loss-weight vector, batch IDs/RNG digest, and checkpoint
   hash;
2. incumbent per-group LRs/ratios, raw `g_k`, fully realized `Δw_inc,k` at multiplier `r=1`,
   `g_kᵀΔw_inc,k`, norms, optimizer moments, decoupled-weight-decay inclusion, and wall time;
3. `Δw`, `Δg`, AdGD/BB secant denominators, and optional deterministic directional HVP
   `Δw_inc,kᵀ H_loss Δw_inc,k`, with `H_loss=∇²f_batch(w_k)` at the same pre-update
   state, batch, and differentiated objective as `g_k`;
4. source-faithful candidate steps: incumbent control; raw-gradient AdGD; raw-gradient Polyak only
   with a valid target; and, along the fully realized AdamW/Muon delta, non-monotone Armijo plus
   directional-quadratic multiplier `r_Q`;
5. actual candidate loss on copied state and the local-model residual; candidate computation time;
6. next-state digest and an explicit refusal reason when a formula's assumptions/inputs fail.

Schedule-Free is **not** evaluated as a stateless one-step candidate from an incumbent sidecar, and
no variant is selected in this pass. Basic Schedule-Free SGD owns `x_t,z_t` with `z_1=x_1` and a
gradient at `y_t`; practical Schedule-Free AdamW v4 Algorithm 1 additionally owns `v_t`,
`β_1/β_2`, warmup/bias-corrected `γ_t`, cumulative `γ_i²` weighting for `c_t`, epsilon, and
decay-at-`y` semantics. A separate fork is ineligible until one variant and its complete state,
initialization/burn-in, and additive resume schema are preregistered.

The instrument must never substitute Fisher/margin/`nᵀHn` for training-loss curvature without an
out-of-sample calibration row. AdGD stays source-faithful to raw gradient descent; the actual
AdamW/Muon delta uses the separately named directional-quadratic/line-search model.

### 4.4 Gate hierarchy

1. **Static custody gate (this pass):** **MEASURED RED** for existing counterfactual replay inputs.
2. **One-step shadow gate:** build the recorder/probe and evaluate eligible candidate losses on
   copied state from a byte-close full sidecar. A current-v9 full pre-Muon sidecar and incumbent
   one-step parity receipt are owed for the trunk question; legacy-mod32cap sidecars do not establish
   current-v9 parity.
3. **Held-out one-step gate:** select no law unless predicted decrease/calibration and observed
   loss-decrease-per-wall-second beat the tuned incumbent on held-out updates without trust-region
   violations.
4. **Short deterministic fork gate:** required for causal path effects and every Schedule-Free arm;
   resumable, per-stage checkpointed, exact same seed/batch custody, and operator GO before any
   heavy execution.
5. **Full-facet gate:** only a governed fork may test time-to-equal through-R `d_seg`, `d_pose`, rate,
   per-class nuclei/anchors, and stability. No proxy-only promotion.

**Falsifier:** candidate law is rejected at `FORMULATION x MEASURED_SEGMENT` if it fails held-out
one-step calibration, yields no positive paired objective-decrease-per-wall-second, violates the
step trust region, or worsens any protected full facet in the governed fork. That negative does not
close the adaptive-step family.

## 5. Triality and canonical-equations note

### DSL

No DSL lever lands in this research pass. Conditional future object:
`WithinSegmentAdaptiveStep`, default-off, additive resume state, scoped below the existing
event-transition controller. It may only compile after the measurement task passes.

### DAG

See `schmidt_icml2026_optstep_crosswalk_DAG_FEED_20260721T203954Z.md`.

### Equation disposition — **no law selected or registered**

```yaml
law_selection: NO_LAW_SELECTED
registered: false
conditional_candidate:
  equation_id: within_event_segment_directional_quadratic_step_model_v1
  formula:
    delta_f_hat: "r * (g^T delta_w_inc) + 0.5 * r^2 * (delta_w_inc^T H_loss delta_w_inc)"
    r_Q: "-(g^T delta_w_inc) / (delta_w_inc^T H_loss delta_w_inc)"
  inputs_and_units:
    r: dimensionless common multiplier; incumbent is r=1
    g: training-objective gradient, loss per parameter-unit
    delta_w_inc: fully realized incumbent update at r=1, parameter-unit; includes decoupled weight decay and preserves per-group LR ratios
    delta_w_inc_T_H_loss_delta_w_inc: directional training-loss curvature, loss
  measured_payoff_not_part_of_equation:
    value: "-delta_f_actual / candidate_wall_seconds"
    units: loss per second
  domain: fixed v9 event segment/loss weights and optimizer state; H_loss=nabla^2 f_batch(w_k) at the same state, batch, and objective as g; candidate segment directionally twice differentiable/model-valid; g^T delta_w_inc < 0; delta_w_inc^T H_loss delta_w_inc > 0; per-group LR ratios fixed; decoupled weight decay is inside delta_w_inc but outside H_loss unless explicitly part of f_batch
  producer_if_built: pact.within_segment_candidate_step_probe.v1
  immediate_consumer_if_measured: tac.probe_outcomes_ledger.register_probe_outcome
  conditional_downstream_consumer: tac.witness_dsl.curriculum_dsl.WithinSegmentAdaptiveStep
  registration_gate: held-out one-step model calibration plus deterministic path-level fork benefit with resume and protected-facet custody
formalization_pending_reason: current trajectory cannot identify alternate-step outcomes; no adaptive-law efficacy anchor exists
```

AdGD, raw-gradient Polyak, non-monotone Armijo, and Schedule-Free remain distinct candidate
formulations, not terms in this conditional equation. No row is appended to the canonical-equations
registry because registering an unmeasured law would launder a proposal into authority.

## 6. Six-hook disposition

| Hook | Disposition |
|---|---|
| Sensitivity map | Reuse Fisher/margin/`nᵀHn` only as named covariates; add no constant and assume no equality to the objective Hessian. |
| Pareto constraint | Wall-clock is advisory joint debt. No score constraint or pointer changes before a full-facet fork. |
| Bit allocator | `N-A-WHY — CURRENT-INSTRUMENT-DESIGN x NO-BYTE_MARGINAL`: a trainer step probe has no measured archive-byte marginal yet. |
| Cathedral/autopilot | The pending canonical task is visible; dispatch/actuation is forbidden. |
| Continual-learning posterior | No update until an empirical one-step/fork receipt exists. This memo is evidence routing, not an outcome. |
| Probe disambiguator | The adopted-build, still-pending instrument will compare mutually defensible formulations within fixed segments and must emit explicit insufficient-custody refusals. |

## 7. Governance self-critique

Schmidt's thesis is a useful criticism of this apparatus: Pact has sometimes treated derivation
completeness, typed laws, and elegant geometry as if they were evidence that an optimizer would move
the practical frontier. The repo already contains unusually rich event scheduling, Fisher/margin
geometry, PDE curvature, manifold momentum, and terminal solves, yet the decisive within-step
question is blocked by something simpler—matched gradient/direction/candidate-loss telemetry. The
correct response is not to abandon rigor; it is to make rigor pay rent earlier: instrument the
smallest causal comparison, include wall cost, preserve tuned empirical baselines, and refuse to
register a law until it wins. Proof is a guardrail and compression of evidence, not a substitute for
the evidence.

## 8. Routing and landing boundary

Exactly one canonical task is registered:

`schmidt_icml2026_optstep_crosswalk_20260721T203954Z::ADOPT_WITHIN_SEGMENT_SHADOW_CANDIDATE_STEP_PROBE`

It is pending at predicted cost **$0**, and it owns only the fail-closed shadow recorder/probe.
Adaptive laws remain unadopted; a DSL Lever, equation registration, short fork, heavy launch,
archive mutation, and pointer movement remain downstream and require their own gates/authority.

MAIN must review the complete base-to-branch diff, especially: the corrected event-first-hybrid premise;
the one-task-only adoption boundary; the D3 identifiability argument; the absence of DSL/equation
registration; all `verdict_scope` tokens; and pointer immobility.
