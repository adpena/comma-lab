# Synthetic data for the costate organ: ranked design for Task #434 (2026-07-11)

**Status:** `research_only=true` · research/design only · no training, scorer forward,
process interaction, dispatch, or score claim. **[MEASURED] Pointer 0.19108282
[contest-CPU] UNMOVED — this is MEANS work only.** Contest closed; the sub-0.15 objective
continues as the 10-year open research program.

**Claim labels:** every substantive claim below is explicitly **MEASURED** (verified from a
repo artifact, source, or primary paper record), **DERIVED** (from a named canonical equation),
or **SPECULATIVE** (a proposed design not validated here). Literature applicability to this
organ is always SPECULATIVE even when the paper metadata itself is MEASURED.

## Authorities read and what each changed

- **[MEASURED] `CLAUDE.md`:** NO-FAKE outranks all other rules; forbidden class #3 says a
  synthetic fixture used instead of `upstream/videos/0.mkv`/real authority input is not a
  canonical empirical anchor and must be `research_only=true`. THE GOAL also makes this memo
  MEANS, never pointer progress; triality requires a future implementation to join the DSL,
  DAG, and equations legs; config must compile from typed DSL rather than invented flags.
- **[MEASURED] `AGENTS.md`:** resumability is P0, settled evidence must not be re-derived,
  negatives require `verdict_scope`, every decision lists STORES CONSULTED, and the cathedral
  invariant ends each unit with pointer-delta honesty and a durable artifact. This task launches
  nothing, so no checkpoint/storage/dispatch path is activated.
- **[MEASURED] `docs/operating_manual_craft_handoff.md`:** claims must survive re-derivation
  from primary artifacts; depth belongs where silent error has the largest blast radius; labels
  travel with claims; and one must attack the conclusion before handoff. That makes real
  walk-forward generalization—not synthetic fit—the load-bearing verification.
- **[MEASURED] `costate_organ_capabilities_limits_envelope_20260711.md`:** the sealed organ has
  one trajectory/vehicle/regime sequence and eight intervals; learned MLP/GRU/DeepONet arms lose,
  fold variance spans about 30x, plateau persistence wins, and graduation needs at least three
  trajectory records. Prototype/Bregman/BSF arms were the transient-regime winners at the sealed
  snapshot, but all conclusions remain instance-scoped.
- **[MEASURED] `scorer_model_arms_430_schedule_20260711.md`:** the trajectory later grew to nine
  intervals and the whole family lost walk-forward to persistence; H gives a zero-model-error
  smoothed-argmax gradient from real cached margins, J recovers boundary susceptibility, K couples
  per-class lambdas with measured boundary adjacency and `1/sigma_cc'`, while two comma10k-prior
  formulations were inert or worse. Synthetic data must add dynamics, not merely reweight columns.
- **[DERIVED] `cgauge_master_action_20260711.py`:** the simulator must descend the relaxed action
  `S_tau = 100 D_seg + sqrt(10 D_pose) + (25/N)L_MDL`, with
  `z_p = Hol_xi(z_bar) + gauge_phase_p + events_p`; A2/A3/A6 remain audited assumptions and
  A1/A4/A5 measured inputs. Its `closed_loop_roles()` identifies training as forward
  Euler-Lagrange flow and costate lambda as the adjoint; the raw argmax is piecewise constant, so
  adjoints live on `S_tau` and campaign state, not on an invented differentiable exact score.
- **[DERIVED] `cgauge_parametrization_optima_20260711.py`:** `whitney_mod_dim(d)=2d+1+margin`
  gives 17–19 only in the SDF-like chart; `parabolic_along_tangent_allocation(nu)=sqrt(nu)`;
  the through-R bank ceiling is 128 cycles/unit; and `beta2_window(S,Tc)` returns
  `[1-1/S, 1-3/(Tc S)]`. These constrain simulated chart dimension, anisotropic bank controls,
  and optimizer-memory state rather than becoming free random knobs.
- **[DERIVED] `textured_power_diagram_20260710.py`:** the scored sufficient statistic is
  `W=(G,xi,T)`, not an untextured partition; `G` is the Laguerre/power-diagram generator set and
  `T={t_c}` is per-class scorer-legible texture. The scorer obligation matrix separates frame-0
  pose, frame-1 chroma-HF seg, and doubly priced frame-1 luma; the flip correction law uses
  `d*=D^T grad_x m|footprint` and `alpha*=mu/||P_R grad m||`, with the corrected lift still owed.
- **[DERIVED] `costate_lambda_marginal_ds_20260705.py`:** `costate_vector(d_pose)` is exactly
  `(100, 5/sqrt(10 d_pose), 25/37_545_489)`, and `chained_ds_depoch` contracts this vector with
  real state slopes. Synthetic labels must preserve this exact score-law composition while
  learning only the control-to-state response.
- **[MEASURED] `tools/lambda_net_backtest.py`:** the real CLI accepts `--run-dir`, `--seed`,
  `--skip-routing`, chunkable `--routing-folds`, chunkable `--archs`, `--no-record`, and
  `--out-dir`. It performs LOO plus deployment-faithful walk-forward against persistence,
  routing, panel, and PRISM checks. It has no synthetic-corpus flag; no such flag may be used
  until implemented and DSL-held.

**STORES CONSULTED [MEASURED]:** the two Task #426/#430 memos above; the four canonical-equation
modules above; `tac.witness_dsl.costate_agent_dsl` (`TrainingStageSpec`,
`CostateAgentProgram.validate_program()/compile()`); the real backtest CLI/help; the canonical
DAG; current lane/subagent ownership state; the Claude memory top entries; and the papers-checked
ledgers. No live-run artifact was opened or mutated.

## 1. SOTA survey (2024–2026), ranked for scalar state-to-costate trajectories

Fit score ranks usefulness for synthesizing `(state, lambda)` pairs under coupled training dynamics,
not headline generative quality on images or language.

| fit rank | method class | literature-grounded state | fit to this organ and verdict |
|---:|---|---|---|
| **1** | **Simulation-based / physics-informed generation** | **[MEASURED—literature class]** Simulation-based inference and world-model rollouts generate labeled trajectories by executing an environment model; recent optimal-design work selects simulator queries by Fisher information rather than volume ([Kurniawan et al., 2024, *Information Matching*](https://arxiv.org/abs/2411.02740)). | **[SPECULATIVE—highest fit]** We possess the action, flow geometry, score law, power-diagram representation, and class coupling. A simulator supplies causal counterfactuals outside the single observed control schedule and exposes its assumptions. Unlike a learned generator, it does not need to infer the governing law from `n=1`.
| **2** | **Dataset distillation: gradient / trajectory / distribution matching** | **[MEASURED—paper records]** Matching Training Trajectories optimizes compact synthetic data to reproduce expert parameter trajectories ([Cazenavette et al., 2022](https://arxiv.org/abs/2203.11932)); 2024 refinements address trajectory instability and fixed-horizon mismatch through Matching Convexified Trajectory ([Zhong et al., 2024](https://arxiv.org/abs/2406.19827)) and Automatic Training Trajectories ([Liu et al., 2024](https://arxiv.org/abs/2407.14245)). Latent Quantile Matching improves beyond mean-only distribution matching ([Wei et al., 2024](https://arxiv.org/abs/2406.09860)). Two 2026 audits warn that soft-label/evaluation protocol can dominate apparent distillation gains and that strong coresets remain competitive ([Dey et al., 2026](https://arxiv.org/abs/2604.18811); [Mittal et al., 2026](https://arxiv.org/abs/2606.18209)). | **[SPECULATIVE—very high fit]** Distill a large physics-generated pool into a tiny set whose gradients and short unrolled updates reproduce real-prefix plus simulator trajectories. This directly optimizes information density, but it cannot repair simulator bias; real-only validation and a selected-real-state coreset baseline remain decisive.
| **3** | **Agentic-RL / model-based rollout synthesis** | **[MEASURED—paper record]** Policy-Guided Diffusion generates entire offline-RL trajectories under a behavior distribution and guides them toward the target policy, explicitly trading on-policy relevance against model error ([Jackson et al., 2024](https://arxiv.org/abs/2404.06356)). | **[SPECULATIVE—high analogy]** Our lever schedule is the policy, the training flow is the environment, and lambda is the shadow-price field. Generate short counterfactual rollouts near real prefixes, never long free-running fantasies; the chief risk is compounding simulator error precisely where lambda is most valuable.
| **4** | **Active local data augmentation** | **[MEASURED—method class]** Classical perturbation, interpolation, mixup/manifold, noise, and symmetry-preserving augmentation remain strong when valid invariances are known; no 2024–2026 paper is needed to rename this established class. | **[SPECULATIVE—high buildability]** Perturb real #205 states only along lawful tangent directions: stage-boundary time shifts, SDF/power-generator coordinates, class weights on the simplex, learning-rate/log-scale, and lever masks. Reject arbitrary Gaussian feature noise because it violates coupled state constraints and may create impossible `(state,lambda)` pairs.
| **5** | **Distribution-matching / score-based generative augmentation** | **[MEASURED—paper records]** TabDiff uses a joint continuous-time mixed-type diffusion with feature-wise schedules ([Shi et al., 2024](https://arxiv.org/abs/2410.20626)); Time Weaver conditions diffusion time-series generation on heterogeneous static and time-varying metadata ([Narasimhan et al., 2024](https://arxiv.org/abs/2403.02682)). A 2026 forward-backward diffusion construction enforces adapted, non-anticipative sequential generation ([Cao et al., 2026](https://arxiv.org/abs/2606.06007)). Decomposed Distribution Matching targets style discrepancy and intra-class diversity in condensation ([Malakshan et al., 2024](https://arxiv.org/abs/2412.04748)). | **[SPECULATIVE—medium fit]** Useful later as a residual model over `real - physics` transition errors, conditioned on stage/class/control. Adaptedness is essential to fold isolation, but at `n=1` learning the full transition distribution is underidentified; matching marginals can erase rare transient regimes and class-edge dependence.
| **6** | **Diffusion-based trajectory synthesis** | **[MEASURED—paper records]** TabSyn performs score-based diffusion in a VAE latent space for mixed-type tables ([Zhang et al., ICLR 2024](https://arxiv.org/abs/2310.09656)); Time Weaver and Policy-Guided Diffusion show conditional time-series and controlled trajectory variants. | **[SPECULATIVE—medium/low now]** Expressive and conditionable, but data-hungry, slow, and not automatically causal. It earns a role only after multiple real trajectories exist, preferably learning simulator residuals or proposing candidates that are relabeled by the physics oracle.
| **7** | **Consistency-model synthesis** | **[MEASURED—paper record]** Improved Consistency Training makes one/few-step generation practical using direct consistency training, Pseudo-Huber loss, and revised noise/discretization schedules ([Song & Dhariwal, ICLR 2024](https://arxiv.org/abs/2310.14189)). | **[SPECULATIVE—low current fit]** Fast sampling helps only after a trustworthy diffusion/flow teacher or enough direct data exists. Here generation cost is not the bottleneck; epistemic fidelity at `n=1` is, so a consistency model accelerates the wrong stage.
| **8** | **GAN/VAE synthesis** | **[MEASURED—paper records]** CTGAN/TVAE are established mixed-tabular baselines; the 2024 PSVAE adds post-selection and minority compensation ([Shulakov, 2024](https://arxiv.org/abs/2407.13016)). Time-series GANs remain useful baselines but are vulnerable to mode collapse. | **[SPECULATIVE—low fit]** Small sample size, rare transition modes, hard physical constraints, and the need for calibrated derivatives make adversarial/latent likelihood fit brittle. A constrained VAE may compress simulator states, but not serve as the label authority.
| **9** | **Self-play / self-improvement / STaR / Self-Instruct trajectory synthesis** | **[MEASURED—paper record]** Quiet-STaR generalizes self-generated rationales and filters learning signal through future-token utility ([Zelikman et al., 2024](https://arxiv.org/abs/2403.09629)); 2025 STeP trains agents on synthetic self-reflected/corrected trajectories with partial masking ([Chen et al., 2025](https://arxiv.org/abs/2505.20023)). | **[SPECULATIVE—lowest direct fit]** The transferable principle is generate→verify→retain, not language rationales. Costate labels must come from equations/simulation or real scorer telemetry, never from an agent's self-asserted lambda. Use self-play only to propose adversarial controls and failure cases for an external oracle to label.

**[DERIVED] Ranking conclusion:** simulation plus optimal design dominates because lambda is a
derivative of a known action under a controlled flow. Pure generative modeling estimates a joint
distribution that the repo already constrains more strongly through equations; with one trajectory,
that extra flexibility is variance, not information.

## 2. Densest-signal experimental design: Fisher-optimal, disagreement-seeking queries

### Why “more random samples” is not the objective

**[DERIVED]** Let a candidate experiment `q=(x_t,u,Delta t)` produce label
`y_q=lambda_u(x_t)` and let `theta_lambda` parameterize a costate arm. Its local information is
`I_q = J_q^T Sigma_q^-1 J_q`, where `J_q = partial f_theta(x_t,u)/partial theta_lambda` and
`Sigma_q` includes simulator discrepancy plus label/numerical error. Independent random samples
from the observed occupancy concentrate in the long plateau, where the envelope measures
persistence already wins; their Fisher matrices are redundant and add little determinant in the
transient directions. D-optimal selection maximizes `log det(I_0 + sum_q w_q I_q)` (identifiability
volume); A-optimal selection minimizes `trace((I_0 + sum_q w_q I_q)^-1)` (mean parameter variance).
Both reward complementary directions rather than count.

**[SPECULATIVE] Acquisition score.** Generate a lawful candidate pool, then greedily/submodularly
select batches with:

`a(q) = Delta_logdet(I | q) + alpha Var_Rashomon[lambda(q)] + beta U_epi(q)`
`       + gamma RegimeBoundary(q) + eta NovelCoupling(q) - rho DiscrepancyRisk(q)`.

- **[SPECULATIVE] `Var_Rashomon`:** variance or action-rank disagreement across ridge,
  persistence, prototype, Bregman, BSF, scorer-prior, and per-class-v8 arms. This is the organ's
  own map of where an extra label can collapse the Rashomon set.
- **[SPECULATIVE] `U_epi`:** jackknife/bootstrap predictive variance, not raw panel activation.
  Calibrate it on real walk-forward residuals because the latest PRISM faithfulness check drifted.
- **[DERIVED] `RegimeBoundary`:** emphasize island birth/death, stage transitions, tau changes,
  Lane erosion/reversal, and onset of plateau; these are topology/flow changes under A6 rather
  than arbitrary “hard examples.”
- **[DERIVED] `NovelCoupling`:** reward under-covered `(c,c')` boundary pairs weighted by measured
  adjacency and `1/sigma_cc'`, especially Road↔Lane, while retaining all five classes.
- **[SPECULATIVE] `DiscrepancyRisk`:** distance from real-prefix support plus violation residuals
  for score law, simplex mass, power-diagram consistency, CFL/optimizer window, and action descent.

### Concrete candidate state space

**[SPECULATIVE] Seed states:** every real #205 verdict/checkpoint state available to the organ,
with chronological fold isolation. For fold `k`, the generator may see only real states up to
fold `k`; future real states remain unavailable even to calibration.

**[SPECULATIVE] Perturbations to simulate:** (1) learning rate and schedule shape within
provenance-bounded ranges; (2) one-at-a-time and sparse factorial lever on/off combinations from
the existing DSL alphabet; (3) curriculum stage lengths and transition times; (4) five-class
bulk/area/impact weights on a constrained simplex; (5) all pairwise `sigma_cc'`-coupled boundary
forces; (6) power-generator positions/weights, `xi`, gauge phase, event carriers, and `T`; (7)
optimizer-memory states including beta2 only inside the derived admissible window. Do not sample
unnamed trainer flags or unconstrained feature vectors.

**[SPECULATIVE] Prioritization:** begin with short symmetric perturbations around each real state
to estimate local directional derivatives; add boundary-crossing brackets where arms disagree;
then fill D-optimal gaps across stage x class x pair-edge x lever. Cap any one regime's share and
reserve quota for rare topology changes, so plateau abundance cannot drown birth/erosion signal.

**[SPECULATIVE] Output record:** immutable rows
`{real_prefix_id, simulator_version, equation_ids, x_t, u, dt, x_t+dt, lambda_state,
lambda_control, local_error, FIM_contribution, acquisition_terms, seed, research_only:true}`.
Every row carries epistemic provenance; no row is a score or empirical anchor.

## 3. Physics-informed generation — our unfair advantage

### Multi-class CGauge simulator

**[SPECULATIVE, DERIVED SHAPE]** Implement a deterministic NumPy-fp32 reference simulator whose
state is not a toy scalar. For classes `c in {Road,Lane,Undrivable,Movable,MyCar}`, maintain:

- level-set/SDF fields `phi_c` or equivalent Laguerre generators `(g_c,w_c)`; per-class area,
  bulk mass, Fisher/margin sensitivity, impact, perimeter, island count/persistence, and texture
  state `t_c`;
- all unordered inter-class interfaces `Gamma_cc'`, adjacency, curvature, and fitted
  `sigma_cc'`; Road↔Lane is explicitly represented, never substituted for the full graph;
- shared `xi`, lattice/gauge phase, scene events, through-R survival state, `d_pose`, archive-byte
  state, optimizer moments, stage identity/time, learning rate, lever mask, and class weights.

**[DERIVED]** Realization is the textured power diagram `W=(G,xi,T)`. Pairwise class boundaries
come from equal-power surfaces of the class generators, while `T` prevents the measured
partition-only failure. The frame/band obligation matrix constrains which simulated actuator can
affect seg, pose, or both.

**[SPECULATIVE, DERIVED SHAPE]** One step integrates a multiphase relaxed gradient flow:

`dot(phi_c) = -M_c delta/delta(phi_c) [100 D_seg,tau + E_bulk +`
`              sum_{c'<c} sigma_cc' E_interface(phi_c,phi_c') + E_area + E_birth`
`              + E_gauge(xi,phase,events) + E_texture + sqrt(10 D_pose) + 25 L_MDL/N]`.

The exact implemented terms must be resolved by canonical equation ID and source callable; a
missing term is a fail-closed build error, not zero. The simulator applies merge→diff→correct
for per-class carriers, preserves the all-class argmax partition, and integrates with adaptive
steps that satisfy the derived stability/optimizer windows. `sigma_cc'` supplies true pairwise
interaction, class bulk is `mass x sensitivity x impact`, and area/birth terms prevent a perimeter
flow from erasing minority islands.

### Labels and fidelity controls

**[DERIVED] State costates:** compute `(lambda_seg,lambda_pose,lambda_bytes)` from the exact
`costate_vector(d_pose)` and contract with the simulated state Jacobian. **[SPECULATIVE] Control
costates:** obtain `lambda_u = dS_tau/du` by a deterministic discrete adjoint or centered finite
difference through the same integrator; store both and require parity within a declared tolerance.
Do not label from a learned arm.

**[SPECULATIVE] Real-prefix calibration:** estimate only unresolved discrepancy coefficients from
the current fold's real prefix; keep canonical/clip-specific constants fixed at their provenance
rung. Use ensemble parameter draws to expose simulator epistemic uncertainty rather than silently
collapsing it to one trajectory.

**[SPECULATIVE] Conservation/validity gates per rollout:** finite action/descent residual;
class-simplex and area conservation residuals; topology accounting for births/deaths; pair-edge
symmetry; power-diagram parse-back; exact score-law recomposition; NumPy determinism; and short-
horizon one-step residual against prefix transitions. Long rollouts stop when discrepancy risk
crosses its calibrated envelope.

### Simulate → distill → train → validate loop

1. **[SPECULATIVE] SIMULATE:** from each real-prefix state, run short counterfactual controls
   chosen by the acquisition rule; produce multi-class `(state,control,next_state,lambda)` rows.
2. **[SPECULATIVE] DISTILL:** optimize a compact synthetic set to match (a) lambda-arm gradients,
   (b) short unrolled response trajectories, (c) per-regime/class/pair-edge distributions, and
   (d) Fisher coverage. Prefer convexified/adaptive-horizon trajectory matching to fixed long
   unrolls; preserve rare transitions explicitly.
3. **[SPECULATIVE] TRAIN:** fit a physics-pretrained arm on the distilled set, then SFT on the
   available real prefix with real rows upweighted and a simulator-discrepancy penalty. Ridge and
   prototype arms remain eligible; “synthetic” does not imply a neural net must win.
4. **[MEASURED interface + SPECULATIVE extension] DSL:** when implemented, add a typed
   `TrainingStageSpec(name="cgauge_fisher_synthetic_pretrain", stage="pretrain", ...)` to
   `DEFAULT_TRAINING_PIPELINE`; it remains `PROPOSED`/`BLOCKED` until a real backtest win, then
   alone may become `EXECUTED_$0` with the measured row. Corpus manifests and equation bindings
   are pydantic fields; no ad-hoc flag or file convention.
5. **[MEASURED interface + SPECULATIVE extension] BACKTEST:** register the arm in the real
   `lambda_net` architecture registry so the existing CLI can select it with `--archs`; use the
   current `--run-dir`, `--seed`, `--routing-folds`, `--no-record`, and `--out-dir`. The current
   CLI has no corpus flag, so any proposed synthetic-manifest interface is non-callable until it
   is actually implemented and DSL-held.
6. **[SPECULATIVE] VALIDATE:** only the real chronological protocol below can change status.

## 4. NO-FAKE validation gate — real trajectory authority or research-only forever

**[MEASURED, NON-NEGOTIABLE]** A synthetic-fixture result never validated against real held-out
trajectory data is CLAUDE.md forbidden fake pattern #3,
“synthetic-fixture-instead-of-real-input.” It stays `research_only=true`, cannot be a canonical
empirical anchor, cannot update architecture arbitration, and cannot be called a win. Synthetic
test performance, simulator recovery, or a synthetic-only ablation is necessary diagnostics only.

### Exact real walk-forward comparison

For every chronological real interval `k -> k+1` in the same #205 trajectory used by the sealed
capabilities envelope:

1. **[SPECULATIVE protocol] Freeze fold:** real verdicts through `k` are training/calibration;
   real verdict `k+1` is untouched test. Simulator fitting, acquisition, distillation, scaling,
   early stopping, and hyperparameter selection may use only the prefix.
2. **[SPECULATIVE protocol] Train matched arms:** compare `(a)` persistence, `(b)` current ridge,
   `(c)` current shipped prototype/Bregman/BSF arbitration, `(d)` real-prefix-only candidate, and
   `(e)` the identical candidate trained on synthetic + optional real prefix. Same features, seed,
   fold boundaries, architecture capacity, and preprocessing; synthetic contribution is the only
   treatment difference.
3. **[MEASURED CLI contract] Execute the registered architecture through
   `tools/lambda_net_backtest.py --run-dir <real-205-dir> --seed <s> --archs <registered-arms>
   --no-record --out-dir <durable-eval-dir>`. Chunk `--archs`/`--routing-folds` if needed. This
   memo does not run it and does not assert an unimplemented architecture name.
4. **[SPECULATIVE gate] Primary endpoint:** lower aggregate real walk-forward lambda MAE than
   both ridge and persistence on identical folds. Secondary gates: per-class MAE, binding AUROC
   at least the existing 0.8 floor, sign/direction accuracy, calibration of uncertainty, and no
   regression concentrated in Lane/Road coupling or topology transitions.
5. **[SPECULATIVE gate] Statistical guard:** report paired per-fold deltas and a block bootstrap
   or exact paired randomization interval; no adoption on a within-noise mean. Require improvement
   to survive leave-one-regime-out stress (transition vs plateau) and at least two deterministic
   seeds for any stochastic learner. With one trajectory, the verdict remains instance-scoped.
6. **[SPECULATIVE gate] Ablations:** physics pool random vs D-optimal; D-optimal vs
   disagreement-added; single-class vs all-class coupled; no-`sigma` vs `sigma_cc'`; no-texture vs
   `W=(G,xi,T)`; full pool vs distilled. A single-class or unconstrained-random arm cannot support
   the recommended recipe even if its mean happens to improve.
7. **[SPECULATIVE adoption] Only after the real walk-forward win:** rerun without `--no-record`,
   attach the durable artifact to the organ ledger, change the DSL stage status with its measured
   row, and re-arbitrate. Cross-trajectory graduation still waits for the existing >=3-record rule.

**[DERIVED] Why this gate is sufficient for the present claim:** the proposed benefit is better
prediction of real lambda along deployment chronology. Therefore the only direct test is future
real intervals excluded from every generation decision; synthetic fidelity metrics cannot answer
that question.

## 5. Top-3 recommended recipes — information density × fidelity × buildability

### 1. Fisher-Optimal Multi-Class CGauge Adjoint Factory — **highest EV**

**[SPECULATIVE]** Seed the full five-class simulator from real #205 prefix states, generate short
counterfactuals chosen by D/A-optimal Fisher gain plus Rashomon disagreement, and label them with
the discrete adjoint of `S_tau`; trajectory-distill the pool before real-prefix SFT. This comes
first because it uses known governing structure to escape `n=1`, spends samples exactly where the
organ is uncertain, and exposes simulator bias to a strict real walk-forward gate instead of asking
a data-hungry generator to rediscover the physics.

### 2. Real-State Disagreement Brackets + Physics-Constrained Trajectory Distillation

**[SPECULATIVE]** Avoid broad simulation initially: form lawful two-sided control brackets only
around real prefix states where arms disagree, then distill those local response segments with
adaptive/convexified trajectory matching. It is the most buildable and safest recipe, but covers
less off-policy state space and may not create enough genuinely new transient regimes.

### 3. Physics-Residual Policy-Guided Diffusion Rollouts

**[SPECULATIVE]** After multiple real trajectories accrue, fit a conditional diffusion model only
to the residual between real transitions and the CGauge simulator; guide short rollouts toward
candidate lever policies and relabel/accept them through the physics and support gates. It can
model missing stochasticity, but it is third because current `n=1` support cannot identify a
reliable residual distribution and consistency/GAN/VAE acceleration would not cure that.

## Adversarial conclusion and named risks

- **[SPECULATIVE risk]** The canonical equations may be structurally right yet quantitatively
  biased at the witness operating point; Fisher-optimal selection can then be optimally wrong.
  Cure: discrepancy ensembles, prefix-only one-step checks, short horizons, real-only adoption.
- **[SPECULATIVE risk]** Synthetic volume can overpower eight real intervals. Cure: real-prefix
  anchoring, explicit synthetic weight sweeps selected inside each training prefix, and paired
  real-only ablation.
- **[SPECULATIVE risk]** Dataset distillation can overfit the tested arm. Cure: cross-arm gradient
  objectives and evaluate ridge/prototype/learned recipients, not one architecture.
- **[SPECULATIVE risk]** D-optimality can ignore rare decision-critical regimes. Cure: combine
  log-det gain with disagreement/topology quotas and report per-regime coverage.
- **[MEASURED limit]** No proposed arm is built or validated here. `verdict_scope: design`; the
  simulator family and synthetic-data family remain open, and every result stays
  `research_only=true` until the real gate passes.

**[MEASURED] Pointer 0.19108282 [contest-CPU] UNMOVED — this memo is MEANS work only; it makes
no exact-score claim.**
