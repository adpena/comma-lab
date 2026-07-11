# The Transient Forge — SOTA-grounded synthetic-trajectory engine for the costate organ (Task #434, 2026-07-11)

**Status:** `research_only=true` · design + survey only · $0 · NO training, NO scorer forward,
NO dispatch, NO score claim · live #205 run (pid 88030) untouched (zero reads of its run dir this
pass). **[MEASURED] Pointer 0.19108282 [contest-CPU] UNMOVED — this is MEANS work.**

**Relation to the prior deliverable:** this memo EXTENDS and partially SUPERSEDES
`.omx/research/synthetic_data_for_costate_organ_supercharge_20260711.md` (the prior #434 design).
Its §2 acquisition rule, §3 multi-class CGauge simulator, §4 walk-forward NO-FAKE gate, and its
9-row generative-methods ranking are CONSUMED as-is (not re-derived). What is NEW here: (a) a
much wider frontier survey (11 domains searched, incl. the NVIDIA stack, learned-optimizer task
distributions, PFNs, UED autocurricula, weight-space learning, amortized BOED, GFlowNet-QD,
data attribution); (b) the engine is re-centered on **manufacturing transient-rich windows**
(the measured #433 blocker); (c) the three success anchors (BIRD / PDR / RQGM) are verified
against their actual papers and operationalized; (d) sibling-task reconciliations (#211, #319,
#430, #433) demanded by the coordinator; (e) the triality legs are LANDED, not promised.

**Claim labels:** every claim is **MEASURED** (verified from a repo artifact or a primary paper
record), **DERIVED** (from a registered canonical law), or **SPECULATIVE** (design not validated
here). Literature applicability to this organ is ALWAYS SPECULATIVE even when the paper itself
is MEASURED.

---

## 0. The crux framing (the anti-fake guard, stated first)

**[MEASURED, from the organ envelope + #433]** The organ predicts λ = ∂S/∂x over the **witness
training-dynamics trajectory** (verdict-cadence campaign states + lever controls), NOT over
driving pixels. It is data-starved at **n = 1 trajectory / 1 vehicle / 9–10 intervals,
plateau-dominated**; learned arms lose to persistence; the anisotropic per-class coupling
(Lane→Road 0.494, measured twice independently) is **forecast-neutral at this n because plateau
windows cannot discriminate it** — #433's named blocker is the absence of transient-rich windows.

**[DERIVED design consequence]** Therefore the synthetic data is a **rich ensemble of synthetic
(regime → λ) TRAINING TRAJECTORIES of the witness descending the cgauge master action S_τ on
0.mkv** — varied by config / seed / init / curriculum / regime-perturbation — NEVER synthetic
driving videos. A photorealistic-video pipeline would generate a data type the organ cannot
consume: CLAUDE.md NO-FAKE class #3 territory (synthetic fixture in place of the real input) at
the design level. Every NVIDIA/GAIA item below is harvested at the PATTERN level only.

**[MEASURED, operator directive 2026-07-11]** Overfitting to 0.mkv is LEGAL AND CORRECT:
deployment is the actual #205-lineage run on 0.mkv. "Cure n=1" means diversity along the
**trajectory axis** (many trajectories of the SAME video), the held-out gate holds out
**trajectories/windows**, and "generalize" means transfer across the trajectory manifold to the
live run — a new trajectory of 0.mkv the organ will advise.

---

## 1. Part A — SOTA survey (what I searched, what I found, ranked)

**Search coverage [MEASURED — what was actually searched this pass]:** (1) NVIDIA
Cosmos/Omniverse Replicator/DRIVE Sim/NuRec; (2) driving world models (Wayve GAIA-2/3);
(3) learned optimizers & task distributions (VeLO lineage); (4) amortized SBI + active
acquisition; (5) weight-space learning / model zoos; (6) GFlowNets + quality-diversity;
(7) amortized/policy Bayesian OED; (8) physics-constrained flow matching & trajectory
diffusion; (9) training-dynamics forecasting / learning-curve extrapolation; (10) influence
functions / data attribution (TRAK lineage); (11) UED / regret-based autocurricula; plus
verification fetches of the three seed anchors (BIRD 2607.08041, PDR 2510.01123, RQGM
2606.26294). **NOT searched (visible gaps, not hidden):** neuroevolution/open-endedness beyond
UED; LLM-agent text-synthesis pipelines; privacy/federated synthetic data; tabular-GAN
benchmarks beyond the prior memo's coverage; RL world-model lines (Dreamer/TD-MPC) beyond the
prior memo's policy-guided-diffusion row. The prior memo's 9-row ranking of generative-model
families (distillation, diffusion, consistency, GAN/VAE, STaR) is consumed, not repeated.

Ranked by (a) fit to the **trajectory-ensemble-on-0.mkv** problem and (b) **signal density per
sample**. Fit tiers: PATTERN (adopt the recipe shape), MACHINERY (adopt the algorithm),
DIRECT (adopt nearly as-is).

### Tier 1 — highest fit (these four shape the engine)

**A1. Learned-optimizer task distributions — VeLO lineage.** [MEASURED—literature] VeLO
(arXiv 2211.09760) meta-trained an optimizer on thousands of diverse synthetic/small
optimization TASKS (4000 TPU-months); PyLO (2506.10315) and Celo2 (2602.19142) continue the
line. [SPECULATIVE—DIRECT fit, the strongest real-world precedent] This is *exactly* our shape:
an optimization-adjacent operator (their update rule ≙ our λ-field) trained on an **ensemble of
training trajectories** rather than on one run. Two transposed lessons: (i) the task
distribution IS the product — VeLO's authors spent most effort curating it; (ii) their failure
mode (poor transfer to tasks outside the meta-distribution) maps to our sim2real risk, cured
only by the real walk-forward gate. Signal density: high — every meta-task contributes whole
trajectories of (state, control, response) triples.

**A2. Prior-data fitted networks (PFNs / TabPFN).** [MEASURED—literature, cutoff knowledge:
Hollmann et al., TabPFN; v2 published in Nature 2025] A transformer trained ENTIRELY on
millions of synthetic datasets sampled from a causal/structural prior, then applied zero-shot
to real tabular problems it has never seen. [SPECULATIVE—DIRECT fit] The purest existing proof
that "train only on synthetic tasks drawn from a physics/structure prior, deploy on the real
instance" can beat fitting the real instance directly at small n. Our physics prior is far
stronger than TabPFN's (a registered action + measured constants), and our deployment instance
is fixed (0.mkv) — both make the PFN pattern MORE applicable here, not less. The organ analogue:
pretrain λ-arms on Forge trajectories, apply to the live run with real-prefix SFT.

**A3. Unsupervised Environment Design / regret-based autocurricula.** [MEASURED—literature]
PAIRED trains a teacher to maximize protagonist-antagonist regret; ACCEL (2203.01302) evolves
levels prioritizing "simplest levels the agent cannot currently solve"; PLR (2110.02439) keeps
a replay buffer prioritized by learning potential; TRACED (2506.19997) and ATLAS (2511.12706)
are the 2025/26 refinements. [SPECULATIVE—MACHINERY fit, THE #433 unblock] Transpose: the
"level" is a witness config/curriculum/perturbation drawn from the typed DSL alphabet; the
"student" is the λ-arm ensemble; "regret" is arm disagreement + loss-vs-persistence on the
generated window. A regret-driven teacher **manufactures exactly the transient-rich windows
where the aniso coupling is discriminable** — plateau-heavy configs score low learning
potential and are not replayed. This is the acquisition upgrade over the prior memo's static
D-optimal rule: the pool itself becomes adversarially curated.

**A4. Sensitivity-constrained neural operators.** [MEASURED—literature] SC-FNO (2505.08740)
trains a Fourier neural operator jointly on solutions AND their parameter Jacobians, retaining
accuracy at low data volume; adjoint-FNO surrogates exist for inverse design (iScience 2024).
[SPECULATIVE—MACHINERY fit] Matches the envelope §7 growth spine (FNO λ-net head, rank-8,
owed > 1 trajectory). For us the Jacobian-matching term is not optional: the organ's OUTPUT is
a derivative, so the operator must be trained Sobolev-style on (state, ∂S/∂x) pairs — the Forge
supplies exactly those labels via the discrete adjoint. Signal density per sample: highest of
any surveyed method (each trajectory point carries a full gradient vector, not a scalar).

### Tier 2 — high fit (machinery adopted into specific engine stages)

**A5. Data attribution / influence functions (TRAK lineage).** [MEASURED—literature] TRAK
(Park et al. 2023) with 2025/26 theory (2602.01312) and scaling (LoGRA, LoRIF); datamodels for
robot data selection (DataMIL 2505.09603); influence-based coresets (In2Core 2408.03560).
[SPECULATIVE—MACHINERY] Use per-synthetic-window influence ON HELD-OUT REAL walk-forward skill
as the flattery detector: windows with measured negative influence are pruned. This converts
"is the synthetic data helping?" from an aggregate bet into a per-sample measurement. For
ridge/prototype arms the influence computation is exact and cheap (closed-form leave-one-out).

**A6. Amortized/policy Bayesian OED.** [MEASURED—literature] ALINE (2506.07259) jointly
amortizes inference and active acquisition; ASNPE (2412.05590) actively picks simulator
parameters; Step-DAD (2507.14057) semi-amortizes design policies; policy-gradient SBOED in
infinite dimensions (2601.05868) uses derivative-informed operator surrogates + D-optimality
rewards. [SPECULATIVE—MACHINERY, stage-2 upgrade] The prior memo's greedy `a(q)` acquisition
is the v0; once the Forge loop runs, an amortized acquisition policy trained across Forge
epochs is the literature-standard upgrade (same reward, no per-batch re-optimization).

**A7. GFlowNets + quality-diversity archives.** [MEASURED—literature] GFlowNet proportional
sampling gives intrinsic diversity; QD red-teaming (2506.07121) and MAP-Elites lineage maintain
descriptor-indexed archives with coverage metrics; 2026 GFlowNet work documents its own
mode-collapse regime. [SPECULATIVE—MACHINERY] The BIRD diversity axis needs a measurable
object: a **QD archive over regime descriptors** (stage × active-class-pair × transient-type ×
curriculum-family). Corpus admission = archive-coverage growth, not row count. GFlowNet-style
reward-proportional sampling of configs is the fallback teacher if regret-UED proves unstable.

**A8. Weight-space learning / model zoos.** [MEASURED—literature] Hyper-representations
(2110.15288, 2406.09997) learn embeddings over populations of trained networks; zoo size AND
composition strongly affect downstream quality (2504.10141); zoos harvested from public hubs
(2510.02096). [SPECULATIVE—PATTERN] Confirms empirically that *populations of training runs are
a learnable data type* and that composition (diversity) beats volume — independent support for
the BIRD-diversity design axis. Direct machinery less relevant (our states are campaign
telemetry, not raw weights).

**A9. Training-dynamics forecasting.** [MEASURED—literature] Gradient Flow Matching
(2505.20221) models weight evolution as optimizer-aware flow and extrapolates trajectories;
RLVR training shows linear-dynamics structure enabling forecast (2601.04537); learning-curve
extrapolation and "neural capacitance" (2201.04194) predict outcomes from early dynamics;
SIREN encoding-error prediction (2410.21645) forecasts INR fit quality — the closest to our
witness (an INR) and a candidate cheap outcome-labeler for #211. [SPECULATIVE—MACHINERY] Two
uses: (i) alternative arm families for the tournament (flow-matching over campaign state);
(ii) cheap surrogate rollouts for the Forge's lowest-fidelity tier.

### Tier 3 — pattern-harvest only (the NVIDIA/driving stack; explicitly NOT a video pipeline)

**A10. NVIDIA Cosmos + Physical AI Data Factory.** [MEASURED—literature/vendor] Cosmos WFMs
(Predict/Transfer/Reason, 2.5 releases; Cosmos 3 open omnimodel, June 2026) generate
photorealistic synthetic video for robotics/AV training; GTC-2026 "Physical AI Data Factory"
blueprints package generation + randomization + post-training into a flywheel.
[SPECULATIVE—PATTERN ONLY] Direct use = wrong data type (pixels; the organ consumes
trajectories). Pattern harvest, each mapped: **world model as controllable scenario generator**
→ our CGauge simulator as controllable trajectory generator; **Cosmos-Reason critic filtering
generated data** → our validity gates + influence pruning; **the data-factory flywheel**
(generate → train → deploy → mine failures → regenerate) → Forge epochs mining live-run
innovation (measured-λ − predicted-λ) as the next epoch's seed regimes.

**A11. Omniverse Replicator / DRIVE Sim / Isaac domain randomization.** [MEASURED—
literature/vendor] Replicator's core capability is programmatic domain randomization (assets,
lighting, materials, camera); NVIDIA's published sim2real case improved real AP 5%→87% by
iterative randomization refinement; NuRec provides 3DGS neural reconstruction for
sensor-faithful sim. [SPECULATIVE—PATTERN] Domain randomization ≙ **config randomization over
the typed DSL alphabet** (seeds, inits, curricula, stage lengths, lever masks, class weights) —
randomize what the organ must be invariant to, hold fixed what defines the deployment (0.mkv,
the score law). The 5→87 lesson transposed: iterate the randomization ranges against the real
walk-forward gap, don't set them once. NuRec/3DGS is the one item with a possible FUTURE direct
role — witness-side (a reconstruction prior for the carrier), not organ-side; out of scope here.

**A12. Wayve GAIA-2/GAIA-3 + counterfactual evaluation.** [MEASURED—literature] GAIA-2
(2503.20523) is a controllable multi-camera latent-diffusion driving world model; GAIA-3 (Dec
2025, 15B) is positioned for *evaluation/validation* of driving AI; CounterScene (2603.21104)
does counterfactual causal reasoning in generative world models for safety-critical closed-loop
eval. [SPECULATIVE—PATTERN] The transposable idea is **generative counterfactual EVALUATION**:
our simulator should also be the organ's counterfactual test bench ("what would λ have been
under the un-taken lever?"), which is precisely the #430 replay machinery already in-tree —
see §2.1. The ego/agent/scenario conditioning vocabulary maps to our (ξ, class-pair,
curriculum) conditioning of trajectory generation.

**A13. Physics-constrained flow matching.** [MEASURED—literature] PBFM (2506.08604) embeds PDE
residuals + algebraic constraints in the flow-matching objective with training-time unrolling;
chance-constrained FM (2509.25157); FlowTS rectified flow for time series (2411.07506).
[SPECULATIVE—low now, MACHINERY later] Upgrades the prior memo's diffusion rows: when ≥3 real
trajectories exist, a *physics-constrained* flow-matching residual generator (constraints =
score-law recomposition, simplex mass, descent residual) is the modern choice over vanilla
diffusion. Not now: at n=1 the residual distribution is unidentifiable.

**A14. Seed anchors verified.** [MEASURED] **BIRD** 2607.08041 = "An exact information theory of
generalization phase transitions in Bayesian diffusion models": exact memorization↔generalization
phase boundary in the joint (amount, diversity) plane; generation operates near the edge of
memorization; information restriction circumvents dimensionality. **PDR** 2510.01123 =
"Rethinking Thinking Tokens: LLMs as Improvement Operators": parallel diverse drafts → distill
to bounded workspace → refine, trained by unrolling the operator under verifiable reward.
**RQGM** 2606.26294 = "The Red Queen Gödel Machine": epochs with a FIXED within-epoch evaluation
criterion; the utility may change ONLY at epoch boundaries — self-improvement without evaluator
drift. All three are load-bearing in §3; applicability remains SPECULATIVE.

**[DERIVED] Survey conclusion.** The frontier's strongest, repeatedly-validated pattern for our
exact problem shape is: *train an operator on a curated distribution of synthetic TASKS/
TRAJECTORIES generated from a structure prior, curate that distribution adversarially (regret)
and information-theoretically (diversity/Fisher), value samples by measured influence, and gate
everything on real held-out transfer.* VeLO, TabPFN, and UED are three independent existence
proofs of the pattern; our advantage over all three is a REGISTERED physics prior (the master
action + measured constants) instead of a guessed one.

---

## 2. Part B — the engine: THE TRANSIENT FORGE

One sentence: **a three-tier trajectory simulator of the witness on 0.mkv, adversarially taught
(UED-regret) to manufacture the transient-rich windows the organ cannot currently see, labeled
by the exact costate law + discrete adjoint, diversity-gated by a QD archive (the BIRD axis),
distilled and consumed under a PDR/RQGM training loop whose only adoption authority is real
walk-forward-vs-persistence skill.**

### 2.1 Generator substrate — the fidelity ladder (cheapest first, all $0-capable)

- **Tier 0 — surrogate replay (ALREADY IN-TREE; consume, don't rebuild).** [MEASURED] The #430
  `schedule_backtest` machinery counterfactually replays lever policies through fitted response
  models on the real #205 trajectory, with quantified trust: self-replay MAE 0.0054 / final gap
  1.6e-4 on the walk-forward-winning state-dependent prototype model. [SPECULATIVE] The Forge's
  v0 generator is THIS machinery run over randomized policies/perturbations: thousands of cheap
  counterfactual windows per real prefix. Bias caveat: tier-0 inherits the fitted model's bias
  (it can never teach the organ dynamics the fitted model doesn't contain) — so tier-0 data may
  pretrain REPRESENTATIONS/regime coverage but adoption-relevant signal must include tier-1.
- **Tier 1 — the multi-class CGauge simulator.** [CONSUMED from the prior memo §3; DERIVED
  shape] Deterministic numpy-fp32 multiphase relaxed gradient flow on the registered action
  (per-class φ_c / Laguerre generators, all pairwise Γ_cc′ with fitted σ_cc′, ξ/gauge/events,
  optimizer-memory state, stage identity), merge→diff→correct, validity gates (descent residual,
  simplex mass, topology accounting, score-law recomposition, determinism). Every term resolved
  by canonical-equation ID, fail-closed. This is the tier that can create genuinely NEW
  transients (island birth/death, boundary formation, Lane erosion/reversal) outside the
  observed control schedule.
- **Tier 2 — short REAL witness micro-runs (owed, operator-GO).** [SPECULATIVE] The only
  bias-free trajectory source: actual witness training runs on 0.mkv with varied
  configs/seeds/curricula, launched via the governed launcher when the #205 slot frees. Even 3–5
  short (≤50-epoch) micro-runs convert every sim-vs-real question into a measurement AND feed
  the organ-ledger accrual (≥3 records) that gates learned-arch graduation. Named trigger: live
  run completes or operator grants a parallel slot; resumable + per-stage checkpoints per the
  non-negotiable.

Labels at every tier: [DERIVED] exact `costate_vector(d_pose)` contraction for state costates;
discrete adjoint / centered finite difference through the same integrator for control costates,
with declared parity tolerance (consumed from the prior memo §3). A learned distilled scorer
surrogate stays BLOCKED as a live scorer ([MEASURED] it lost to the zero-model-error arm H);
as a *label densifier* it is a candidate only after beating arm H on label fidelity — owed.

### 2.2 Generation DOF (all typed, never invented flags)

[MEASURED interfaces] The randomization alphabet is exactly what the DSLs already hold: witness
curricula (`HoscSchedule`/`Transition`/`Curriculum`, #334), the #403 curriculum-candidate pool,
lever masks from the activation ledger (17 levers), stage lengths/boundaries, per-class weights
on the simplex, seeds/inits, and regime perturbations expressible as state edits at a real
prefix (island seeding, boundary displacement, τ/ε changes within the derived admissible
windows). [SPECULATIVE] Domain-randomization discipline (A11): randomize what the organ must be
invariant to; NEVER randomize the score law, the video, or unregistered constants.

### 2.3 Transient manufacturing — the #433 unblock (the engine's reason to exist)

[SPECULATIVE, machinery from A3] A UED-style teacher curates generation:

1. **Regret signal** per generated window = λ-arm ensemble disagreement (Rashomon variance)
   + per-arm loss-vs-persistence on that window (learning potential). Plateau windows score ≈0
   and die; windows spanning island birth/death, stage transitions, and **excited Lane↔Road
   boundary dynamics** (drive the measured C_phys 0.494 coupling with asymmetric class
   perturbations so aniso ≠ iso in the DATA) score high.
2. **PLR-style replay buffer** of high-regret windows; the organ trains against the buffer, not
   the raw stream.
3. **Acquisition** = the prior memo's `a(q)` (Fisher log-det + Rashomon + regime-boundary +
   novel-coupling − discrepancy risk) with the regret term added; greedy/submodular v0,
   amortized policy (A6) as the stage-2 upgrade.
4. **Anti-Goodhart guard:** the teacher's reward is NEVER the adoption metric (real WF skill);
   it is learning-potential on synthetic folds only. Adoption authority stays with §3.

### 2.4 The BIRD gate — diversity that raises effective log(N)

[MEASURED—paper] BIRD locates a memorization↔generalization phase boundary in the joint
(amount, diversity) plane; redundant volume does not cross it. [SPECULATIVE operationalization]
- **QD archive** (A7) over regime descriptors (stage × class-pair × transient-type ×
  curriculum-family × lever-mask-class). **Corpus admission rule: a batch is admitted only if
  archive coverage grows** (new cells or improved cell-elites), not on row count.
- **Redundancy audit:** nearest-neighbor distance distribution + effective rank of the
  window-feature matrix per epoch; a shrinking effective rank at growing volume = redundant
  volume, batch rejected.
- **Memorization probe (the boundary's other side):** a probe classifier trying to identify
  WHICH source trajectory a held-out window came from; probe accuracy ≫ chance at fixed
  coverage = the corpus is memorizable → increase diversity before adding volume.

### 2.5 Training recipe — PDR × RQGM, reward = walk-forward skill

[SPECULATIVE, anchors A14]
- **PDR loop:** (i) PARALLEL: generate M diverse candidate corpora/arm-configs (different
  teacher temperatures, tiers, distillation budgets); (ii) DISTILL: compress each into a bounded
  workspace — a distilled trajectory set (trajectory/gradient matching per the prior memo §3
  step 2) + a scorecard; (iii) REFINE: retrain candidate arms on distilled + real prefix,
  conditioned on the workspace; the winner seeds the next round. Reward for any RL-tuned
  component = held-out **walk-forward skill over persistence** (anchor 1) — verifiable,
  non-gameable within an epoch.
- **RQGM epoch structure:** WITHIN an epoch the evaluation is FROZEN (fixed real folds, fixed
  persistence null, fixed MAE + binding-AUROC≥0.8 metrics, fixed seeds); the bar RATCHETS only
  BETWEEN epochs (new incumbent = last epoch's adopted arm; evaluator changes allowed only at
  the boundary, with a recorded reason in the organ ledger). This is the structural cure for
  reward-flattery/evaluator drift — the same failure class the envelope already caught once
  (LOO look-ahead flattery).
- **Influence pruning (A5):** per-window TRAK/exact-LOO influence on held-out real WF skill;
  measured-negative windows pruned each epoch. The flattery detector at sample granularity.

### 2.6 Sibling reconciliations (coordinator-required)

- **#211 (FORWARD operator; closest sibling).** [MEASURED] #211 amortizes clip → witness
  init/params; corpus-gated WATCH; the operator's overfit-to-0.mkv framing IS #211's
  contest-overfit arm. [SPECULATIVE composition] The Forge is the SHARED data engine: #211
  consumes the (config → final-params/outcome) MARGINAL of the same simulated ensemble
  (meta-init / Reptile pretraining on the 0.mkv trajectory manifold); #426 consumes the
  (state, control → λ) TRANSITIONS. One generator, two marginals — build once.
- **#319 (SimpleTES campaign_outcome_credit).** [SPECULATIVE] Simulated rollouts reach horizons
  for free → Monte-Carlo returns per window = a SECOND label channel (terminal credit)
  complementing in-run λ; feeds #319's corpus gate without new machinery.
- **#430 (schedule replay).** [MEASURED] Its replay models + self-replay-trust protocol ARE
  tier 0 (§2.1); its state-gated cascade is a priority curriculum FAMILY for generation DOF.
- **#433 (the blocker).** §2.3 is the designed cure; §3's acid test is its measurement.
- **Graduation discipline.** [DERIVED from the envelope's gate] Synthetic trajectories NEVER
  count toward the ≥3-record learned-arch graduation gate — only real trajectory records do
  (tier-2 micro-runs count; Forge output does not). Synthetic data trains; real data graduates.

---

## 3. Part C — success criterion + eval harness (first-class deliverable)

**[MEASURED harness]** `tools/lambda_net_backtest.py` — real CLI verified this pass:
`--run-dir` (required) · `--seed` · `--skip-routing` · `--routing-folds` (chunkable) · `--archs`
(chunkable subset) · `--no-record` · `--out-dir`. It already computes LOO **and**
deployment-faithful walk-forward vs persistence, binding AUROC, routing, panel, PRISM
faithfulness, and writes a durable JSON + (default) an organ-ledger record. **It has NO
synthetic-corpus flag; none may be cited until implemented and DSL-held.** The Forge-trained
arm enters as a REGISTERED architecture in `lambda_net.ARCHITECTURES` selected via `--archs`.

**The adoption gate (all six, chronological, null-relative):**
1. **Fold hygiene:** for each real interval k→k+1 of the #205 trajectory, everything the Forge
   touches (simulator calibration, teacher, distillation, hyperparameters, early stopping) sees
   only the real prefix ≤ k. Real k+1 is untouched test. No synthetic row derived from
   post-k information may exist in fold k's training set.
2. **Matched arms:** (a) persistence · (b) incumbent E_prototype_bregman · (c) same candidate
   architecture trained real-prefix-only · (d) the candidate trained Forge+real. Same features,
   seeds, folds, capacity. Synthetic contribution is the ONLY treatment difference (c vs d).
3. **Primary endpoint:** (d) beats BOTH persistence AND (b) AND (c) on aggregate real
   walk-forward λ MAE. Secondary: per-class MAE, binding AUROC ≥ 0.8, no regression
   concentrated in Lane/Road or topology transitions.
4. **Statistics:** paired per-fold deltas + block bootstrap / exact paired randomization;
   within-noise means are NOT adopted (the envelope's 30× fold-variance rule); ≥2 seeds for
   any stochastic component; verdict stays instance-scoped at 1 real trajectory.
5. **The #433 acid test:** after Forge training, re-run P (aniso) vs Q (iso ablation) — if the
   manufactured transients carry real signal, P must SEPARATE from Q on real folds
   (currently Δ=1.15e-4, within noise). Separation in the aniso direction = the engine
   delivered the discriminating data; continued neutrality = the coupling is either absent in
   reality or the simulator can't express it — both informative, neither adoptable.
6. **Sim2real report:** per-epoch (synthetic-fold skill − real-fold skill) gap, trended. A
   growing gap = the teacher is drifting into simulator-flattered regimes → tighten
   discrepancy-risk penalty; this is the Replicator 5→87 iteration loop transposed.

**Named failure modes (what would prove flattery, not signal):** (i) (d) beats (c) on synthetic
folds but not real folds → simulator-fit, reject; (ii) gains vanish when influence-pruned
windows are removed → a few flattering windows carried the mean, reject; (iii) P–Q separation
appears on synthetic but not real folds → manufactured transients are simulator artifacts;
(iv) gains appear only with `--no-record`-style protocol deviations or evaluator edits
mid-epoch → RQGM violation, void; (v) probe-classifier accuracy ≫ chance (§2.4) while WF
improves → memorization of the corpus, not generalization over the manifold — reject even
though the headline improved. **[MEASURED, NON-NEGOTIABLE]** Anything not passing the real gate
stays `research_only=true` forever (NO-FAKE class #3); this restates and inherits the prior
memo's §4 gate.

---

## 4. Part D — triality wiring (landed, not promised)

- **DAG:** FEED-434-nvidia block appended to
  `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` (this landing).
- **DSL:** one new `TrainingStageSpec(name="transient_forge_synthetic_trajectories",
  stage="pretrain", status="PROPOSED")` added to `DEFAULT_TRAINING_PIPELINE` in
  `tac.witness_dsl.costate_agent_dsl` (this landing) — the tracked, reasoned OFF state with the
  named trigger; it may become `EXECUTED_$0` only with a measured §3 row. No trainer flag is
  invented; the backtest CLI is consumed as-is.
- **Equations:** N/A this pass — no new law (design only; n=1 trajectory is below the ≥5-run
  anchor bar, consistent with the #427 seal's stance). The engine CONSUMES
  `cgauge_master_action_v1`, `costate_lambda_marginal_ds_v1`, σ_cc′ (#382), power-diagram
  (#284) — none re-derived. First candidate law WHEN the Forge runs: the measured sim2real
  transfer curve (synthetic-fold skill → real-fold skill), anchor-registered per epoch.

---

## 5. Round-1 adversarial self-review (attacking my strongest claims)

- **Strongest claim attacked — "UED-regret manufactures the #433-unblocking transients."**
  Weakness: regret-UED assumes the generator CAN express the discriminating dynamics. If the
  tier-1 simulator's Γ-interface terms are quantitatively biased exactly at the Lane–Road
  coupling, the teacher will manufacture confident, wrong transients — Fisher-optimally wrong,
  as the prior memo already warned. Defense in place: §3.5 acid test + §3.6 sim2real trend +
  discrepancy ensembles + tier-2 micro-runs as the bias-free referee. Residual honest risk:
  at 0 tier-2 runs, a null §3.5 result cannot distinguish "coupling absent" from "simulator
  can't express it" — stated in-gate, not hidden.
- **"VeLO/TabPFN prove the pattern"** — over-claim check: both operate at task-distribution
  scales (thousands–millions) we will not reach, and both had unconstrained compute. The
  transposable content is the PATTERN + the failure modes, not the scale law. Labeled
  PATTERN/SPECULATIVE accordingly.
- **Where a reviewer could catch a fake:** (a) if any future implementation cites synthetic-fold
  wins as adoption evidence — §3 forbids it; (b) if a "Forge" lands generating video frames or
  anything the organ can't consume — §0 forbids it; (c) if the DSL stage flips to EXECUTED
  without a `--archs`-registered arm and a durable backtest JSON — the DSL validator itself
  refuses (measured-row required); (d) if graduation counts synthetic records — §2.6 forbids it.
- **Weakest link overall:** tier-0's fitted-model bias is seductive because it is cheap and
  already trusted (self-replay MAE looks good) — but self-replay only bounds ON-POLICY error;
  off-policy windows are exactly where the organ needs data and exactly where tier-0 trust is
  unquantified. Mitigation: tier-0 restricted to representation pretraining + coverage; every
  adopted gain must survive with tier-0 rows influence-pruned.
- **Survey comprehensiveness:** 11 domains searched + explicit not-searched list (§1). Gaps are
  visible; the not-searched domains are judged low-fit but that judgment is itself SPECULATIVE.

## 6. What I did NOT do / owed-with-named-triggers

- Did NOT build the simulator, teacher, archive, distiller, or any arm; no code beyond the DSL
  stage stub. Trigger to build: operator GO on the Forge v0 (tier-0 + acquisition + §3 harness
  registration), $0-capable.
- Did NOT run `lambda_net_backtest.py` or any training/scorer forward; no GPU, no dispatch,
  $0 spent. Trigger: Forge v0 lands → §3 protocol on real #205 folds.
- Did NOT launch tier-2 micro-runs (compute owned by the live #205 run). Trigger: run slot
  frees or operator grants a parallel slot; these also feed the ≥3-record graduation gate.
- Did NOT register any equation anchor (below the anchor bar; design-only).
- Did NOT modify, signal, or read the run dir of pid 88030.

**[MEASURED] Pointer 0.19108282 [contest-CPU] UNMOVED — MEANS work; no score claim.**
