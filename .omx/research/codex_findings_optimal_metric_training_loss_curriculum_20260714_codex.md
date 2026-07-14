# Codex findings — optimal training-loss metric and curriculum metric anneal

**Pointer status:** `[contest-CPU Linux x86_64] 0.1910828242` UNMOVED. This landing is research/build MEANS only; no byte-closed exact archive row was produced.  
**Lane:** `lane_optimal_metric_training_loss_curriculum_20260714`  
**Containment:** $0 read-only measurements + typed DSL; no training, evaluator, paid/remote dispatch, or live-run mutation.  
**Primary receipt:** `experiments/results/optimal_metric_training_loss_curriculum_20260714/measurement_receipt.json`, SHA-256 `8e8d830013f86d775eec812ff882a597b0c63cece0fc0008e606d6f280a7f6b2`.

## Verdict first

1. **DERIVED:** the new loss geometry is not “CE plus another categorical Fisher.” CE is already the negative-entropy Bregman divergence and has Fisher curvature on the softmax quotient. The nonredundant object is the **reachable decision pullback** `G_s = J_q^T W_s J_q + lambda I`, with `J_q = D_theta[Q_s C z(R(theta))]`, and its damped inverse/pseudoinverse as the natural-gradient preconditioner.
2. **MEASURED, narrow scope:** on the preserved real-n600-source heldout-n120 records, moving the comparison from ambient RGB to the 19-D renderer pullback raised cosine alignment by `12.499998x` (round 2), `51.423882x` (round-3 linear), and `51.041743x` (round-3 RFF). This supports the reachable-geometry family, but it is a first-cut diagnostic—not the requested flip-reduction verdict.
3. **NO-VERDICT_DATA_CUSTODY:** the requested real-n600 comparison of raw-CE and Fisher/decision-quotient directions against a true finite flip-reduction direction is not measurable from retained state. Logits/probabilities, centered/winner-rival directional Jacobians, optimizer steps, and before/after perturbation outcomes were not retained or were cleanup-certified deleted. This is a custody status only: no instance, formulation, family, or paradigm is closed.
4. **DERIVED, UNMEASURED:** the #430-aligned metric schedule is `island_birth: CE/global` -> `boundary_form: centered Fisher pullback/annulus` -> `tau_sharpen_repair: winner-rival pullback + #360 constraint` -> `finish: finite through-R flip preservation`. Which metric wins each stage is not yet measured.

## 1. The loss law

Let

- `x(theta)=R(theta)` be the deterministic witness receiver;
- `z(x)` be the five-class frozen SegNet logits;
- `C=I-11^T/K` remove the softmax gauge;
- `Q_s` select the stage decision coordinates (full centered logits or the GT winner-versus-rival contrast);
- `q_s(theta)=Q_s C z(R(theta))`;
- `W_s` be a positive-semidefinite stage weight containing support, class-edge, and/or Fisher structure.

Then

`A_s(theta) = W_s(theta)^(1/2) D_theta q_s(theta)`,

`G_s(theta) = A_s(theta)^T A_s(theta) + lambda_s I`,

`L_M,s(theta;theta*) = 1/2 [q_s(theta)-q_s(theta*)]^T W_s [q_s(theta)-q_s(theta*)]`.

The corresponding trust-region step is

`u_s = argmin_u <grad L_s,u> + (1/(2 eta)) u^T G_s u`,

so `u_s = -eta G_s^dagger grad L_s` after the declared damping/restriction. Convention matters: in the already-landed similarity `h_T^T M h_S`, `M` is the **preconditioner** `G_s^dagger`, not the primal metric `G_s`.

### Fisher-Rao / Bregman / mirror-descent relation

For `psi(p)=sum_c p_c log p_c`, the Bregman divergence `D_psi(p*||p)=KL(p*||p)`. Its probability-coordinate Hessian is `diag(1/p)` on the simplex tangent; its logit-coordinate Hessian is the categorical Fisher

`F(p)=diag(p)-pp^T`,

with null direction `1`, removed by `C`. CE is precisely this negative-entropy Bregman loss against a one-hot target. Thus raw CE does **not** lack categorical Fisher curvature. What it lacks operationally in the current path is a declared reachable quotient trust region and stage-dependent support/preconditioning. Mirror descent expresses the same geometry in dual probability/logit coordinates; natural gradient expresses it by solving against `G_s` in the reachable parameter chart.

### Operational difference from CE and soft cosine

| Treatment | Loss coordinates | Update geometry | Where gradient is spent | Limitation |
|---|---|---|---|---|
| Raw through-R CE | all logits / one-hot target | incumbent Euclidean optimizer metric; CE curvature is Fisher | global pixels/classes | does not explicitly normalize by reachable quotient geometry or finite flip effect |
| Lever-2 soft cosine | angle between probability vectors | incumbent Euclidean optimizer metric | global unless separately weighted | angular/bounded proxy; probability saturation and scale removal; no explicit pullback solve |
| Reachable decision metric | centered or winner-rival quotient after actual `R` | damped `G_s^dagger` trust-region step | stage-selected annulus/active set | requires logits/Jacobians/applied-step custody and an n600 finite-effect selector |

## 2. Composition with settled levers

- **#141/#274 margin saliency:** partial metric shaping. It supplies the scalar/conformal support `a(p)` (and the #274 reachability mask) inside `W_s`. It concentrates the metric but does not define the quotient or inverse pullback solve by itself.
- **#382 `sigma_cc'`:** class-edge surface-tension anisotropy, not automatically a decision metric. It can provide a class-pair block in `W_s` only after metric closure. The current fitted instance violates one triangle inequality (`1.764344 > 0.7381986449045815 + 1.0`), so that instance is not treated as a valid distance or a Gamma-limit theorem. The family remains open via metric closure/Wulff/Finsler reformulations.
- **#360/#459 MarginBandSatisficing:** a one-sided R-headroom feasible-set hinge, not a metric. In the terminal stages it composes as an active-set constraint/projection (or barrier) around the winner-rival metric. It should not be silently absorbed into `W_s`, because doing so loses the distinct `m_safe >= delta_R` feasibility law.

The full pullback is therefore a unifying generalization, while the three settled mechanisms remain distinct factors: support (`#141`), class-edge anisotropy (`#382` after closure), and feasible-set constraint (`#360`).

## 3. $0 measured result

The sealed retained receipt `.omx/research/surrogate_vjp_fidelity_metric_remeasurement_20260714.json` has SHA-256 `c4116ff0b9af3284b00e90980f693f98be3c11b30eada0ac13bb395cf50c3753`. It binds the canonical real-n600 source cache but evaluates only the deterministic heldout split `pair_index % 5 == 0`, `n=120`.

| Formulation | raw RGB cosine | reachable 19-D `M=I` cosine | lift | reachable same-LR `eta` | Scope |
|---|---:|---:|---:|---:|---|
| round2 convex head | 0.0014157934 | 0.0176974146 | 12.499998x | 0.0023036129 | first-cut x real-n600-source heldout-n120 |
| round3 pre-SE linear | 0.0016650256 | 0.0856220782 | 51.423882x | 0.0046477802 | first-cut x real-n600-source heldout-n120 |
| round3 pre-SE RFF | 0.0016791964 | 0.0857091120 | 51.041743x | 0.0046480302 | first-cut x real-n600-source heldout-n120 |

**Asked comparison:** not computed. There is no retained `u_flip`, so substituting exact CE gradient, raw costate cosine, ordinal concordance, GT margins, or the n120 pullback as “true flip reduction” would be fake. The live #205 resume at ep250 also contains no preserved stage checkpoint set; the historical attrclean n600 run has CE/Tau/Muon/L7 weights but not the per-stage logits/Jacobians/applied-step outcomes required for the selector.

**Verdict scope:** `NO-VERDICT_DATA_CUSTODY` for the requested test. The optimal Fisher/decision/functional family remains intact. A new no-training replay from preserved checkpoints is the reformulation, not a negative conclusion.

## 4. Curriculum-varying metric

| #430 stage | Metric `M/G` | Support and composition | Gate | Evidence |
|---|---|---|---|---|
| `island_birth` | negative-entropy Bregman CE; Euclidean optimizer geometry | global/coarse; preserve capacity to enter basin | `run_start -> birth_completion` | law derived; incumbent path exists |
| `boundary_form` | centered-logit categorical Fisher pullback; damped natural step | #141/#274 margin-reachability annulus; metric-closed #382 block only | `birth_completion -> annulus_plateau` | derived; stage selector unmeasured |
| `tau_sharpen_repair` | winner-rival reachable pullback | active annulus; #360 R-headroom satisficing remains a separate constraint | `annulus_plateau -> powerlaw_meat` | derived; selector unmeasured |
| `finish` | finite applied-step/flip-preservation functional | observed flip band; Pose trust-region side condition | `powerlaw_meat -> governed_stop` | required admission authority; unmeasured |

This is the level-set coarse-to-fine flow viewed through its metric: as the boundary localizes, the relevant tangent geometry sharpens from global basin acquisition to the codimension-1 annulus, then to the active winner-rival normal, and finally to finite decision preservation. The schedule changes only at stage boundaries, matching the v7.5 operating contract.

## 5. DSL and triality

- **DSL:** `DecisionMetric`, `MetricStage`, `MetricAnneal`, and `optimal_decision_metric_anneal()` are first-class schedule primitives in `tac.witness_dsl.curriculum_dsl`. `Curriculum.metric_anneal` carries them. The objects persist law `metric_id=argmax_native_vjp_fidelity_v1` separately from per-state schema `reachable_decision_geometry_fidelity.v1` and selector schema `reachable_decision_preconditioner_selection.v1`; schema names cannot masquerade as the canonical law identifier. They emit **zero new argv**, surface a typed `TrainerSupportGap`, enforce #430 order and stage-boundary-only mutation, and refuse a measured label without receipt path, 64-hex SHA, sample count, and explicit through-R custody. `MEASURED_THROUGH_R_N600` additionally requires `n_pairs==600` and `realized_through_r is True`.
- **Equation/helper:** the sibling-owned canonical `metric_id` is `argmax_native_vjp_fidelity_v1` in `tac.canonical_equations.argmax_native_vjp_fidelity_20260714`; the shared NumPy-fp32 helper is `tac.scorer_surrogate.vjp_fidelity`. Its public composition surface is `PullbackPreconditioner`, `renderer_pullback_numpy_fp32`, `identity_pullback_preconditioner_numpy_fp32`, `categorical_fisher_preconditioner_numpy_fp32`, `margin_fisher_preconditioner_numpy_fp32`, `reachable_decision_geometry_summary`, `nullspace_reweighting_disentanglement`, and `select_preconditioner_by_through_r_agreement`. The sibling selector refuses anything but 600 complete states. This lane did not duplicate or mutate that surface. The training-loss specialization is preserved in the build spec. No new empirical equation anchor was registered for the unmeasured n600 flip selector.
- **DAG:** standalone feed `optimal_metric_training_loss_curriculum_DAG_FEED_20260714.md` records dependencies, support gap, and ranked queue.

Verification: `13 passed` new DSL tests plus `140 passed` relevant existing witness/curriculum derivation and consistency tests (`153 passed` combined); Python compile and `git diff --check` clean for owned files. The self-review additionally proved malformed stage members, unhashable names, and wrong metric objects report validation failures rather than crashing, and that receipt-schema names cannot be substituted for the law identifier. One attempted test command named two nonexistent test paths and was corrected immediately; it is not test evidence.

## 6. Ranked next actions

1. **Highest EV — full-n600, no-training finite-step selector.** Re-render exact pairs `0..599` from preserved, hashed stage checkpoints; retain logits/probabilities, centered and winner-rival JVP/VJP data, CE step, damped pullback steps, and equal-trust-region finite through-R before/after flip counts plus Pose deltas. Freeze a design split and validate without reselection on held-out pairs. **Gate:** only a custody-complete receipt may set `MEASURED_THROUGH_R_N600`. **Scope:** measurement/build; no family conclusion from a failed first formulation.
2. **Per-stage selection on CE/Tau/Muon/L7 checkpoints.** Use the same selector contract at the historical n600 stage boundaries and test whether the best metric actually changes. **Gate:** choose a different stage metric only if held-out true flip reduction improves at matched trust radius without Pose harm. **Scope:** instance x checkpoint family.
3. **Implement the trainer consumer only after 1–2.** Consume a typed sidecar/config—not free-form invented flags—persist active metric/stage/damping/solver state in every resume checkpoint, switch only at #430 boundaries, and preserve legacy byte-identical behavior when absent. **Gate:** parser, resume, deterministic replay, and no-silent-no-op tests before any launch authority.
4. **Optimal-form local short A/B after operator GO.** Compare CE vs selected metric at identical initialization/EMA/checkpoint budget and measure epochs/wall-clock to the same through-R argmax target, not proxy loss. **Gate:** n600 through-R and Pose trust region; instance verdict only.
5. **Pointer attempt last.** Only a byte-closed archive with exact contest-CPU/CUDA custody may move `0.1910828242`; the metric receipt alone never does.

## Stores consulted

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; v7.5/v8 canonical specs; `reports/latest.md`; `.omx/state/lane_registry.json`; `.omx/state/subagent_progress.jsonl`; live inbox through `2026-07-14T12:53:25Z`; #430 schedule memo; #360/#382/#141 implementations/memos; retained surrogate VJP receipt and cleanup manifests; canonical GT cache; live #205 read-only resume snapshot; historical n600 stage checkpoint inventory; sibling generic helper/equation/policy files and final API/custody handoff.

## Pointer delta honesty

No archive, scorer run, or exact evaluator row was produced. Pointer delta = `0`. The durable gain is a nonredundant law, a fail-closed typed schedule surface, an exact custody receipt, and a ranked optimal-form measurement queue.

Serializer status: content review passed, but the exact intent-manifest `--patch-file` landing failed at `git apply --cached` with `rc=128` (`unable to create temporary file` / `unable to create backing store`, sandbox). No commit SHA exists; the session summary and final handoff enumerate exact owned artifacts and exclude sibling hunks.
