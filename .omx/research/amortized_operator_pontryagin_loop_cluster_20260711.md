# The Amortized-Operator Pontryagin Loop — the CGauge control organ (cluster registration + the #426 build, 2026-07-11)

**Agent:** Einstein (Fable) #426 build pass · **Cost:** $0 (local CPU; read-only on the live run;
pid 88030 verified healthy before/during/after; NO scorer forward anywhere in this pass).
**Pointer 0.19108282 [contest-CPU] UNMOVED** — everything here is MEANS (a campaign accelerator);
every number `[macOS advisory] NON-PROMOTABLE score_claim=false`.

**STORES CONSULTED:** `cgauge_master_action_and_parametrization_20260711.md` (the costate crown §2 +
knob table §4 + N1-N8) · `paper_harvest_v9cgauge_20260711.md` (ANR 2606.16303 recipe · 2512.24897
rank-8-favorable · EKI→#396) · `n205_live_telemetry_harvest_for_v9cgauge_20260711.md` (§4
binding-vs-inert = the backtest labels; §3f amber regime = the control parametrization) ·
`src/tac/witness_control/` as-built (costate_estimator tiers · ncde_trajectory DMDc · shadow
controller · producer_bridge duty queue) · `tac.subagent_contract` · lever_activation_ledger ·
canonical_equations registry (292 rows; both bound laws present) · fresh full-literature survey
(subagent, this pass — synthesis §3).

---

## 1. THE CLUSTER (operator: "identify the emergent cluster here")

**One loop, four amortized operators** — every expensive/intractable solve in the witness's
optimal-control problem becomes a learned or gradient-free operator, orchestrated by the costate
controller. This is the technical shape of the 10-year autonomous program:

| Task | Control role | The amortized solve | Status |
|---|---|---|---|
| **#211** | FORWARD operator | clip → witness-INR init/params (encoder side) | corpus-gated WATCH; de-risked by 2512.24897 (rank-8 = favorable amortization regime) |
| **#426** | **ADJOINT operator** | campaign-state → costate **λ = ∂S/∂x** as a FIELD over the design surface | **BUILT + BACKTESTED this pass** (§2) |
| **#247** | CONTROLLER / actuator | u\* = argmin H(x,u,λ) closed-form; duty ranking; agent-spawn | live (SENSE/shadow) + the #426 act-side realizes its DECIDE/ACT alphabet |
| **#396** | TERMINAL solver | gradient-free (EKI/MC) on the EXACT argmax d_seg over low-dim gauge constants | owned lever from the paper harvest; fire-decision = a #426 λ-readout |

**THE LOOP:** forward(#211) → costate(#426) → control(#247) → terminal-finish(#396) → measured
rows → RETRAIN the operators. The master action S (`cgauge_master_action_v1`) derives WHAT; these
operators are the learned HOW. λ IS `costate_lambda_marginal_ds_v1` (#247's analytic law); the
(derived-λ − measured-binding) gap is the innovation signal.

**Means/ends honesty:** #396 is the ONE member that directly touches the exact score (a terminal
finisher on a real checkpoint, byte-closed). #211/#426/#247 are the meta-loop — accelerators.
Pointer moves only through a byte-closed `upstream/evaluate.py` row.

## 2. WHAT WAS BUILT (#426 — the learned Pontryagin adjoint, as an embodied agent)

All committed, 50 tests green, ruff-F clean, containment source-scan green:

- **`tac.witness_control.lambda_net`** — the adjoint operator. SENSORS: `read_trajectory`
  (read-only run-dir parse → 6 verdict-cadence states with per-class d_seg + 148 dense per-epoch
  control rows). The costate chain is **hybrid-honest**: ∂S/∂x stays ANALYTIC (score law; per-class
  weights FITTED from verdict rows — recovered the canonical GT areas [.232/.006/.495/.012/.282]
  at residual 9e-8, a free independent check); ONLY the response operator ∂x/∂u is learned, as a
  FIELD Λ(x, φ_lever) over lever FEATURES (class/term/boundary/regularizer — derived from the
  master-action knob table) so never-fired levers get feature-structured PARTIAL-tier what-ifs.
  Four architectures: A ridge-SOLVE (DMDc-with-features, closed form) · B MLP · C GRU-over-dense-
  path · D DeepONet branch⊗trunk. B/C/D train **self-supervised through the rollout** (ANR
  2606.16303: no λ labels — the objective along the surrogate rollout is the loss, + L1(Λ)).
  Plus `rashomon_lambda_set` (jackknife SOLVE ensemble = Ensemble-SINDy-lite): per-lever sign-
  agreement; the honest uncertainty object.
- **`tac.witness_control.costate_panel`** — the diverse learned council: 6 lenses, each a distinct
  QUESTION + TOOL (flow/DMDc = "where is the flow heading" · pointwise · sequence/dense-telemetry ·
  operator-field/duty-queue · graph-precedent retrieval over the activation ledger · **spawn_agent
  = System-2**, self-activating on Rashomon action-disagreement + cross-lens dispersion). 5 routing
  modes (single-best / question-router / routing-free self-activation / component-fusion /
  evidence-shrunk stacking) + the nested-LOO `routing_benchmark` arbiter.
- **`tac.witness_control.control_alphabet`** — the ACT side: closed alphabet of SAFE ($0,
  score-neutral, executable-as-advisory-artifact) vs HEAVY actions; **HEAVY structurally returns an
  `OperatorGoTicket`** (no execute path exists); `hamiltonian_decide` = the closed-form pointwise
  u\* on the share simplex (the ANR DECIDE); `feasibility_projection` = the act-time convex
  projection = the containment governor; **`SpawnTicket`** = the bounded agent-spawn actuator
  (composes `tac.subagent_contract.standard_contract` + the INHERITED_CONTAINMENT_CLAUSE verbatim;
  harness-executed only; recursion bounded inside the advisory-$0 envelope).
- **`tac.witness_dsl.costate_agent_dsl`** — the **CostateAgent DSL**: the controller's typed
  program (pydantic, fail-closed). SensorSpec (score-neutral ⇒ default-ON with reason-required
  off) · ActuatorSpec (cost-tier typed; HEAVY must declare operator_go_gated; spawn must declare
  inherits_containment) · ExpertSpec (architecture validated against the REAL lens registry —
  never-invent) · RoutingSpec (typed mode + provenance) · EquationBinding (λ law resolved
  fail-closed against the canonical registry) · ContainmentSpec (the boundary as a typed field).
  `.compile()` → `CompiledCostateOrgan` with the REAL wiring (sense/adjoint/decide/act/spawn) —
  exercised end-to-end by the tests on the live run's telemetry. **The triality is now a closed-loop
  control system with TWO typed DSLs: witness = plant · costate = controller.**
- **`tools/lambda_net_backtest.py`** — the $0 honesty-gate CLI (tournament + routing benchmark +
  panel verdict → durable JSON under `experiments/results/costate_organ_backtests/`).

## 3. THE SURVEY → THE OPTIMAL CRUX (full-literature pass, subagent, $0)

Domains: LLM/ensemble routing+MoE (RouterBench 2403.12031 · routing-free 2604.00801 · FusionRoute
2601.05106 · RouterArena 2510.00202) · operator-learning theory (**2405.15992 curse-of-sample-
complexity** · 2603.00819 · 2512.24897) · Pontryagin-adjoint learning (**ANR 2606.16303** ·
Pontryagin Neural Operator 2401.01502 · PG-DPO 2501.12600 · **SINDy-MPC 1711.05501 + Ensemble-
SINDy**) · meta-learning/amortization (VeLO-failure 2209.11208 · 2510.11471) · agentic System-1/2
(AUQ 2601.15703 · Voyager) · uncertainty/Rashomon (**Bayesian hierarchical stacking 2101.08954** ·
Semenova-Rudin).

**THE OPTIMAL CRUX (confirmed against my hypothesis, 3 sharpenings):** *amortize the invariant
STRUCTURE analytically and spend the scarce data only on the smallest identifiable residual
object — then let capacity be admitted by EVIDENCE, not design.* Sharpenings, all implemented:
(1) self-activation must be a **partially-pooled posterior over lens competence** (2101.08954),
never a trained router and never raw train-confidence → `EVIDENCE_SHRUNK_STACKING` (MDL complexity
prior × per-channel inner-fold evidence); (2) the System-2 boundary is **Rashomon-set
action-disagreement** (sign/rank of λ), not a confidence scalar → `rashomon_lambda_set` trigger;
(3) the **identifiability ledger is a first-class output** (UNIDENTIFIED masks + duty-to-measure
as active learning) → `identified` flags + `rank_duty_to_measure`.

## 3a. INTERPRETABLE-BY-DESIGN + SELF-IMPROVING (Rudin + Schmidhuber; BUILT + BACKTESTED, 2026-07-11)

Binding directive: the organ must EXPLAIN every λ (Rudin: explanations are CONTRACTS not
retrofits) and INVENT its own next measurement + PROVE each self-change (Schmidhuber). Both BUILT,
both earn their place by measurement. Papers:
`papers_checked_prism_powerplay_godel_fno_costate_organ_20260711.md`.

- **PRISM prototype router (2607.00510) — the interpretable head (`prototype_router.py`,
  E_prototype lens).** λ = a SPARSE non-negative mixture of learned PROTOTYPES = named trajectory
  REGIMES (names DERIVED from the measured per-class d_seg signature — NO-FAKE), each anchored to a
  neighborhood of actual #205 states; the mixture weights ARE the explanation (identifiability
  ledger becomes cheap + exact — PRISM's 500× attribution). Daubechies multi-scale (coarse regime →
  fine correction) cascade. SLIM-style linear controller head (DECIDE stays readable). Prototype
  SUPPRESSION = traceable behavior removal. `PrototypeAttribution` = the OBSERVATORY row (fired
  regimes, weights, mixture-entropy uncertainty, neighborhood epochs). **MEASURED cost of
  interpretability (routing_benchmark_v2, 5-fold nested LOO): +8% scalar (0.003109 vs flow
  0.002881) but 2.3× BETTER per-class (0.0105 vs ridge 0.0237); it independently NAMED the live
  regime "lane-erosion" = the harvest §3b warning.** Interpretability is a HARD requirement (typed:
  `RoutingSpec.interpretable_head` fail-closes) and here it is also per-class-dominant.
- **Schmidhuber self-improvement (the self-improving half of the loop).** (i) **PowerPlay
  1112.5309** = the self-inventing curriculum: `powerplay_acquisition` ranks the duty-to-measure
  queue by curiosity × blast-radius / cost, where **curiosity = the innovation signal
  (derived-λ − measured-binding = compression progress / surprise)** and PowerPlay prefers the
  SIMPLEST (cheapest) not-yet-solved probe; the "does-not-break-what-works" invariant = the
  never-regress guard. (ii) **Gödel machine cs/0309048** = `GodelProofGate`: a self-change is
  admissible ONLY with a proof-of-improvement = backtest-passed AND predicted-ΔS<0 AND containment
  — the correspondence named + enforced (our backtest-IS-the-gate was always the Gödel discipline).
  (iii) **Curiosity/compression-progress** = the innovation signal IS the acquisition function.
  (iv) **LADDER ⊂ costate REAFFIRMED** (L56): LADDER = 1-channel/const-λ; island-birth = per-class
  λ homotopy (rode Movable 0.9998→0.0073 LIVE); PRISM prototypes = the compressed attributable
  trajectory memory where compression-as-intelligence meets the costate.

## 3b. THE LAB-FRONTIER SWEEP (DeepSeek/ByteDance/Qwen/Z.ai/Tencent/GDM/Meta — live, 2026-07-11)

Full rows: `papers_checked_lab_frontier_sweep_moe_agentic_rl_20260711.md`. **VERDICT: the 2026
frontier CONFIRMS all four measured conclusions** — (1) closed-form solve at small n (every
frontier stability move REMOVES learned machinery when signal is scarce: DeepSeek aux-loss-free
bias controller 2408.15664 = balance-by-CONTROLLER; GDM expert-choice 2202.09368 = balance BY
CONSTRUCTION; SAO deletes the group baseline); (2) raw self-activation fails (nobody ships it
ungated; routing-free MoE 2604.00801 itself needs bolted-on balancing); (3) evidence-shrunk
stacking = the industrial default (DeepSeek-V4's >10 specialist teachers → reverse-KL On-Policy
Distillation IS evidence-weighted panel consolidation); (4) Rashomon-gated System-2 = AlphaEvolve
in production (LLM proposes, hard evaluator gates every act). The one counter-signal — VAPO
2504.05118 (learned critic beats value-free at 5k steps × large batches) — marks the CROSSOVER
regime for a future learned λ-critic, 3+ orders beyond one trajectory. Growth ladder adopted into
the duty queue: per-lens bias starvation-guard (now, $0) → SAO-async+GSPO-sequence-clip λ updates
(n_traj≳tens) → VAPO-regime critic A/B → Engram-style telemetry→prior lookup memory.

## 4. THE BACKTEST (the NO-FAKE gate; #205 trajectory, ep2→125, 5 intervals; all advisory)

**Gate 1 — held-out forecast (LOO) vs the persistence heuristic** (what
`stage_epoch_costates` extrapolates today), scalar d_seg units:

| λ-net architecture | fMAE | heuristic | per-class MAE | verdict |
|---|---|---|---|---|
| **A ridge-SOLVE** | **0.003379** | 0.005632 | **0.0255** vs 0.1066 (**4.2×**) | **PASS — beats heuristic** |
| B MLP | 0.027076 | 0.005632 | 0.0588 | overfits (expected: 2405.15992) |
| D DeepONet | 0.008906 | 0.005632 | 0.1047 | overfits scalar; ≈heuristic per-class |
| C GRU | 0.093553 | 0.005632 | 0.1302 | data-starved (expected) |

**Gate 2 — binding alignment** (harvest §4 MEASURED labels): realized-binding AUROC 1.0 = the
magnitude heuristic's 1.0 (consistency check; labels correlate with magnitude by construction —
stated, not laundered). The field's NEW capability the heuristic lacks: per-class attribution +
never-fired what-ifs (duty ranking; top field what-if = `horizon_margin`, the deliberately-deferred
HorizonWeightedMargin — PARTIAL tier, a measurement priority not a measured effect).

**Routing benchmark (nested LOO, 5 folds, no leakage; chunked-resumable execution):**
SINGLE_BEST=flow **0.002881** (winner; =QUESTION_ROUTER) · COMPONENT_FUSION 0.003023 (best
per-class 0.085805) · EVIDENCE_SHRUNK_STACKING 0.005060 (beats the 0.005632 heuristic; the growth
form) · SELF_ACTIVATION (raw train-confidence) **0.068 FALSIFIED** (the overconfident-GRU failure
the survey's sharpening #1 predicts). Solo ablation: flow 0.002881 · pointwise 0.0413 ·
operator_field 0.0433 · sequence 0.2012. **Parsimony wins at n=5 — by measurement, not taste**
(kitchen_sink anti-pattern honored); the DSL v1 ships `mode=SINGLE_BEST` with the stacking mode as
the declared, benchmark-re-arbitrated growth form. Question/tool-coverage ablation: graph_precedent
+ spawn_agent abstain from forecast but answer questions no predictive lens can (history precedent;
open-ended deliberation) — they stay.

**Status stamps:** the A-lens forecast field on THIS trajectory = **BACKTESTED-PASS**; everything
else (learned lenses, never-fired what-ifs, cross-campaign generalization) = **SPECULATIVE-UNTIL-
BACKTESTED** carried on the artifacts themselves. verdict_scope: instance — one trajectory, one
regime; the tournament re-runs per campaign as folds accrue (`tools/lambda_net_backtest.py`).

## 5. RECONCILIATION (last-24h sweep into triality + tasks + controller)

- **#247 doc reconciled:** `tac/witness_control/__init__.py` docstring now names the loop it sits
  at the center of (+ exports the #426 surfaces). The costate crown (master action §2) is the
  controller's analytic law; #426 is its realization layer.
- **Landed & marked:** master-action pass (26abe5883) · paper harvest (33ba8be58) · SPEC_v9_cgauge ·
  live-harvest (1cf9b9843) — all committed pre-this-pass; #426 registered in the canonical task
  ledger (in_progress → completed with this commit).
- **Duty-to-measure queue (registered as `costate_organ_duty_queue_20260711`):** EKI→#396 probe ·
  UU-1 Fisher-at-witness (bounded n≤48, off the live slot) · N1/N2/N3 builds · D18 k90 ~7KB
  byte-close · T1 SEAL + n600 A/B · #299 arms (operator-GO) · UU-4/N8 canary · UU-7 cache verify ·
  learned-GNN lens + full Bayesian stacking (both owed at >1 campaign trajectory).
- **MEMORY.md re-tightened** under the 17KB cap with the cluster line added.
- **Triality legs:** DSL = `costate_agent_dsl` (NEW controller-side DSL; the witness DSL untouched —
  no trainer lever changed) · equations = EquationBinding resolves `cgauge_master_action_v1` +
  `costate_lambda_marginal_ds_v1` fail-closed (no NEW law registered: the backtest is n=1-trajectory
  calibration, below the ≥5-run anchor bar per the ncde precedent — re-open at ≥5 campaigns) ·
  DAG = FEED-426-adjoint-organ.

## 5b. EXPLORATION-POLICY ADDENDUM (NVFP4-RL / QeRL / Sol-RL fold-in, 2026-07-11)

Per `papers_checked_nvfp4rl_qerl_solrl_explore_decouple_20260711.md`: (a) **explore-cheap /
commit-high-fidelity decoupling** (Sol-RL) CONFIRMS our standing MLX-proxy→exact-authority split
and sharpens the organ's design — the λ-field/duty-queue SENSE tier is the CHEAP proposal stage
(rank many candidates), through-R/exact measurement is the reserved COMMIT stage (the #396 pool →
exact verdict and #319 K>1 emission are the same shape); (b) **quantization-noise-as-exploration**
(QeRL) = a WATCH lever for #396's EKI ensemble noise (shape/anneal it as exploration) —
SPECULATIVE-UNTIL-PROBED, rides #396's existing $0 probe; (c) HONEST CAVEAT: NVFP4's tensor-core
systems speedup is Blackwell/H100-only — NOT our M5 Max substrate; only the algorithm transfers.

## 5c. TRAINING-REGIME ADDENDUM (SAO arXiv 2607.07508 fold-in, 2026-07-11 — TIER-1)

Per `papers_checked_sao_2607_07508_single_rollout_async_20260711.md` (read live): SAO is the
paper written for our regime. (a) **Async optimization** → the organ's SENSE→DECIDE→ACT loop
never blocks on λ-training: retrain asynchronously off the critical path; adopt the SAO
stability analogue when λ-training goes online — a **trust region on Λ between refits**
(their strict double-side token-level clipping, transposed to per-component field updates) so a
stale refit cannot swing DECIDE. BUILD-OWED on the growth path. (b) **Single-rollout sampling**
(one rollout per prompt beats group-wise on generalization) is the literature-side de-risk for
training on our SINGLE #205 trajectory — conservative updates (their clipping ↔ our ridge/L1 +
evidence-shrunk pooling) are the shared load-bearing ingredient, and our measured tournament
(SOLVE beats the nets) is the same lesson measured locally. Composes with §5b's explore-cheap/
commit-exact split.

## 6. OWED THROUGH OPERATOR-GO (none of this fires autonomously)

- Any HEAVY actuation (launch/stop/config/paid-GPU) — the organ can only emit OperatorGoTickets.
- Autonomous agent-spawning in production: the SpawnTicket interface is BUILT; executing tickets
  stays with the harness/operator (inherited containment embedded verbatim in every prompt).
- #299 arms + T1 SEAL/A/B + the EKI n600 scale-up + UU-6 witness-CUDA row — each a ticket, each
  with its λ-justification field ready.

## OWN-ROUND-1 REVIEW (adversarial)

1. **Is the backtest circular?** Gate 2 partially is (labels↔magnitude) — stated in-code and here;
   Gate 1 (held-out forecast) is the discriminating test and the SOLVE genuinely beats the
   incumbent extrapolation 1.7× scalar / 4.2× per-class. The planted-dynamics unit test separately
   proves the fit recovers KNOWN causal structure (island→Movable sign) — not markers.
2. **Did the fancy nets earn their place?** As PREDICTORS, no — measured and said plainly
   (B/C/D lose; SELF_ACTIVATION falsified). As LENSES they stay only for question/tool coverage +
   the disagreement signal, and the DSL records the measured provenance for the parsimonious
   default. This is the survey's own prediction (curse of sample complexity) confirmed on our data.
3. **Identifiability laundering?** No: `identified` flags ride every field readout; co-constant
   active levers are labeled FEATURE-STRUCTURED/PARTIAL; never-fired = measurement priorities.
4. **Containment really structural?** The package source-scan test bans process tokens; HEAVY has
   no execute path; spawn returns prompts, not processes; the DSL types the boundary and its tests
   prove fail-closed on ungated HEAVY / uninherited spawn / invented tools.
5. **Scope:** one trajectory, one regime, 5 intervals. Every claim is instance-scoped; the growth
   claims (stacking, GNN, hypernet trunk) are labeled owed-at->1-campaign.

**Pointer 0.19108282 [contest-CPU] UNMOVED.**
