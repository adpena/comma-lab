# Build spec — reachable-decision training metric and metric anneal

**Task:** operator P0 optimal metric applied to the training loss and varied by curriculum stage  
**Lane:** `lane_optimal_metric_training_loss_curriculum_20260714`  
**Authority:** DESIGN + $0-MEASURE + typed DSL only; no heavy/paid launch; no score claim  
**Pointer:** `[contest-CPU Linux x86_64] 0.1910828242` unchanged; the defensive `0.18804` object is not promoted by this work.

## Objective

Represent the training geometry as a typed, state-gated curriculum parameter without inventing trainer flags, and state the nonredundant law precisely enough that the future trainer consumer cannot confuse:

1. the categorical Fisher already implicit in cross-entropy;
2. the through-R reachable decision-quotient pullback metric; and
3. a natural-gradient preconditioner, which is the inverse/damped pseudoinverse of that pullback metric.

## Canonical law

Let `x(theta)=R(theta)`, `z(x)` be the five-class SegNet logits, `C=I-11^T/K` the centered-logit quotient, and `B_y` the winner-versus-rival decision operator. For stage `s`, define

`q_s(theta) = Q_s C z(R(theta))`,

`A_s(theta) = W_s(theta)^(1/2) D_theta q_s(theta)`,

`G_s(theta) = A_s(theta)^T A_s(theta) + lambda_s I`,

and the local through-R decision loss

`L_M,s(theta;theta*) = 1/2 ||q_s(theta)-q_s(theta*)||^2_{W_s}`.

The intrinsic simplex form uses negative entropy `psi(p)=sum_c p_c log p_c`:

`D_psi(p* || p_theta) = KL(p* || p_theta)`,

whose local logit Hessian is `F(p)=diag(p)-pp^T` on the centered quotient. The metric trust-region step is

`u_s = argmin_u <grad L_s,u> + (1/(2 eta)) u^T G_s u`,

hence `u_s = -eta G_s^dagger grad L_s` after damping/restriction. If `M` denotes the preconditioner in the already-landed similarity `h_T^T M h_S`, then `M=G_s^dagger`; it is not `G_s`. This convention must be explicit in every receipt.

The typed consumer persists canonical law `metric_id=argmax_native_vjp_fidelity_v1`. It validates per-state evidence with `state_receipt_schema=reachable_decision_geometry_fidelity.v1` and a selector with `selection_receipt_schema=reachable_decision_preconditioner_selection.v1`; schema names are not interchangeable with the canonical law identifier.

## Nonredundancy and composition

- Raw CE is already the negative-entropy Bregman divergence and has categorical-Fisher curvature. Adding a bare output-space `F^-1` label to the same CE implementation is not a new loss. The nonredundant treatment restricts the geometry through `D_theta R`, centered/winner-rival decision coordinates, support weights, and a damped trust-region solve.
- Lever-2 soft cosine is an angular probability-space proxy. It is bounded/scale-normalized but neither a renderer-pullback metric nor an explicit winner-rival trust region.
- `#141/#274` margin saliency is a scalar/conformal support factor `a(p)` and reachability mask inside `W_s`: partial metric shaping.
- `#382` `sigma_cc'` is class-edge surface-tension anisotropy, not by itself a decision metric. A fitted matrix may enter `W_s` only after metric closure; the current fitted instance violates one triangle inequality and is not silently treated as a valid distance.
- `#360/#459` MarginBandSatisficing is a one-sided R-robust feasible-set hinge, not a metric. It composes with the terminal metric as an active-set constraint/projection.

## Stage law

The schedule is state/event gated and only mutates at stage boundaries:

1. `island_birth`: through-R CE, ambient/coarse support, Euclidean optimizer geometry. Fisher preconditioning is suppressed because confident interior curvature is near-degenerate and global basin entry is still needed.
2. `boundary_form`: centered-logit categorical Fisher pulled through R, supported by margin saliency/reachability and metric-closed class-pair weights. This is a damped Gauss-Newton/natural-gradient candidate.
3. `tau_sharpen_repair`: winner-rival quotient on the active annulus, with `#360` R-headroom satisficing as a separate constraint.
4. `finish`: finite applied-step/flip-preservation metric; admission requires the actual through-R change in flip count plus Pose trust-region telemetry.

These are the exact #430 cascade stages (`island-birth -> boundary-form -> tau-sharpen+repair -> finish`). Stage entry is expressed by named sensors (`birth_completion`, `annulus_plateau`, `powerlaw_meat`, governed stop) rather than guessed epochs. The DSL emits no new argv until a trainer consumer exists; it surfaces a typed `TrainerSupportGap` instead.

## $0 evidence contract

The requested comparison is `cos(u_CE, u_flip)` versus `cos(u_M, u_flip)` on exact real pairs `0..599`, where `u_flip` is obtained from preregistered finite applied steps and actual through-R flip-count deltas. Required custody per row: logits/probabilities, exact renderer Jacobian/pullback or applied delta, CE gradient, metric step, before/after argmax, pair id, checkpoint/stage, hashes, and Pose side effect.

Current retained surrogate receipts do **not** satisfy this contract: they retain real-n600-source lineage but evaluate a fixed heldout `n=120`, and cleanup removed the logits, full costates, preconditioner, and perturbation outcomes. That is a data-custody status, not a negative verdict on the metric family.

## Owned files

- `src/tac/witness_dsl/curriculum_dsl.py`: typed `DecisionMetric`, `MetricStage`, and `MetricAnneal` primitives; optional `Curriculum.metric_anneal`; zero invented flags.
- `src/tac/tests/test_optimal_metric_training_loss_curriculum.py`: enum/order/fail-closed/support-gap/display tests.
- `.omx/research/codex_findings_optimal_metric_training_loss_curriculum_20260714_codex.md`: derivation, measured custody verdict, composition, next actions.
- `.omx/research/optimal_metric_training_loss_curriculum_DAG_FEED_20260714.md`: standalone DAG feed.

The sibling `optimal_metric_p0_build_surrogate_followons` retains ownership of the generic metric helper, centered-logit student/refit, full-n600 M-selection receipt, and generic canonical equation. This lane does not duplicate those outputs.

## Verification

1. `python -m py_compile` on touched Python files.
2. Focused pytest for the new DSL.
3. Existing curriculum DSL tests covering primitive registry/display/compile behavior.
4. `git diff --check`.
5. Own round-1 adversarial review: try invalid stage order, duplicate stage, malformed stage members/unhashable names/wrong metric objects, unknown coordinate/geometry, claimed measured selection without receipt, and invented-flag leakage.
