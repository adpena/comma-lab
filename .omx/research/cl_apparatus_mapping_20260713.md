# Continual learning belongs in the apparatus, not the witness trainer

**Date:** 2026-07-13  
**Checkpoint:** `cl_apparatus_reader`  
**Status:** ANALYSIS + $0 DESIGN ONLY; `research_only=true`; uncommitted  
**Authority:** mechanism-selection design for the research apparatus; no score, training, launch,
promotion, or shared-ledger authority  
**Git state read:** `main@41f1ff008f642975123579d4b6c8c7cc43012b29`  
**Pointer delta:** ZERO; the contest pointer is intentionally unmoved

## Executive verdict

The Harrington et al. taxonomy fits Pact's **apparatus** but does not make witness training an LLM
continual-learning problem.

| Pact knowledge surface | Verdict | One-line reason |
|---|---|---|
| `MEMORY.md` / `CLAUDE.md` + graph memory #411 + retrieval-first #346 | **GAPPED** | The graph materially improves recall and context reconstruction, but it remains mostly an opt-in/session-context affordance; it has not demonstrated competence increase under change or closed the loop into organ mechanism selection. |
| canonical equations + typed DSL + candidate pool | **MATCHED for accumulation; GAPPED for stale-fact replacement** | Append-only events, explicit supersession/domain refinement, latest-row-wins reads, and provenance are strong stable-memory machinery; reactivation criteria and cross-ID contradiction invalidation are not generally executable at read/dispatch time. |
| costate-organ backtest arms + #436 dispatch | **MATCHED only as guarded predictive selection; GAPPED as RL** | Past-only walk-forward routing, union-residual evaluation, surprise deferral, and real-only gates control noise, but one trajectory without transition-complete logged exploration cannot identify causal arm credit or a competence-increasing policy update. |

**Biggest gap:** Pact has no trajectory-level, lineage-aware **change-pattern classifier and update-
mechanism arbiter** whose outputs are evaluated on current adaptation, prior-lineage retention, and
future-lineage transfer. The stores remember a great deal, but no closed loop proves that the system
became more competent after a change rather than merely receiving better context.

**Organ selection rule:** detect `new lineage/domain` versus `within-lineage time drift` first. A new
vehicle receives retrieved/config priors only as a reversible cold start; stable repeated evidence may
refresh a lineage-conditioned distilled surrogate; RL-style credit is allowed only for selective
changes with reliable evaluator reward, target-action support, transition-complete logs, and FORE/HCM
identification. Unknown support or noisy reward routes to persistence/defer-and-collect.

**GEPA reconciliation:** **RESCOPE**, not CONFIRM. The local result says a GEPA-*style* fixed-template
candidate cycle helped neither this one trajectory nor this candidate space (`11/11 REFUSED`, `n=1`).
Harrington et al. measure a broader sequential profile—fast present-stage adaptation with forward
degradation. That profile neither rescues nor validates the local candidates. It justifies one
future, tightly gated sequential re-probe across independent run lineages with current/retention/
forward metrics; it does not justify running it now.

## 1. Fit boundary and epistemic labels

### What transfers

- **MEASURED BY HARRINGTON ET AL.:** continual learning is operationalized as increasing competence
  under change, not merely preserving or retrieving prior context. The protocol separates changes
  across domains (space) from changes over time, and compares prompt evolution, distillation,
  online RL, and context compression under one sequential evaluation frame.
- **DERIVED FOR PACT:** Pact itself is the continual learner. Its units of change are runs, vehicle
  lineages, evaluator observations, equations, DSL actions, candidate arms, and verdicts. Its
  mechanisms are knowledge-store updates and decision policies—not LLM weight updates per se.
- **DERIVED ANALOGY, NOT LITERAL DISTILLATION:** canonical equations, typed DSL records, and pool
  rows are the symbolic stable-accumulation layer. They are not SFT/SDFT weight updates.
- **DERIVED FOR PACT:** the transferable object is the protocol question: after new evidence, does the
  apparatus improve current decisions while retaining old valid competence and transferring forward?

**Numeric-label note:** the future `>=3`-lineage gate below is a **SPECULATIVE preregistered minimum**
inherited from the local organ's existing reactivation boundary, not a claim of statistical adequacy.
FORE/HCM identification and whole-run uncertainty can still refuse at any larger count.

### What does not transfer

- The paper's tasks concern LLM knowledge, question answering, tool use, prompt state, and model
  updates. A level-set witness optimizer is not an episodic-state LLM agent.
- A prompt method's failure on changing facts does not imply that `CLAUDE.md` is bad, nor does an RL
  method's success imply that the costate organ should become an online policy learner.
- FinQA/SciKE/Wikipedia/10-K outcomes do not estimate contest score, d_seg, d_pose, archive bytes,
  training stability, or run-to-run transfer in Pact.
- The public project page marked the accompanying code as **coming soon** when checked on 2026-07-13;
  the repository named in the brief could not be verified as an available code authority. This memo
  therefore uses the paper and official project results, not an unperformed code-parity audit.

**VERDICT-SCOPE:** every paper-to-Pact conclusion below is a mechanism/protocol analogy at the
apparatus layer. It is not empirical evidence about witness-training efficacy.

Primary sources: [arXiv:2607.07847](https://arxiv.org/abs/2607.07847) and the authors'
[official project page](https://anneharrington.github.io/studying-cl/).

## 2. Apparatus-as-continual-learning-system audit

### 2.1 Prompt/context surface: graph memory closes recall loss, not the learning test

| Audit item | Finding |
|---|---|
| Paper's measured failure mode | Prompt methods adapt quickly to the present stage but tend to lose forward performance; context access alone is not evidence of increased competence. In the authors' domain sequence, separate per-task prompts matter, while a shared evolving prompt can overfit transient heuristics. |
| Existing Pact mitigation | `CLAUDE.md`/`AGENTS.md` impose structural priors; `tools/corpus_query.py` performs cross-store deterministic retrieval; #411 builds a typed corpus graph with provenance, supersession, producer/consumer, blocker, and task edges; #346 adds bridge/hub/crux retrieval; the subagent contract requires retrieval-first `STORES CONSULTED`. |
| Residual gap | #346's lens path is opt-in, its measured bridges paid on only one of three probes, hub/crux results were often generic standing directives, and its governed dashboard/SENSE consumer remains owed. `costate_digest.py` exposes recall as a session-start affordance rather than automatically conditioning #436. No sequential metric distinguishes “the decision improved because the apparatus learned” from “the agent was shown more text.” |
| Concrete ticket | `cl_lineage_change_mechanism_selector_v1`: make lineage relation and change pattern explicit inputs to a governed decision record, then score current adaptation, prior retention, and forward transfer. This is staged in the candidate-row artifact, not the shared pool. |

#### Does #411 + #346 close the paper's gap?

**DERIVED verdict: NO, but it closes a meaningful prerequisite.**

It closes the “we forgot or failed to retrieve the relevant prior” class better than flat prompt
stuffing: typed edges, deterministic reconstruction, supersession chains, and recall provenance are
real apparatus. It does not close Harrington et al.'s competence criterion because:

1. Retrieval quality is not evaluated against downstream mechanism choice over a sequence of run
   changes.
2. The lensed path is not yet the default governed consumer for the organ or autopilot.
3. The store can surface mutually relevant records without deciding which record remains valid for a
   new lineage or time regime.
4. There is no three-axis CL scorecard: `adapt_current`, `retain_prior`, `transfer_forward`.

The correct disposition is to keep #411/#346 as the **prompt-prior/cold-start layer**, never treat
them as the learning layer, and require their retrieved hypotheses to pass the same empirical gates
as any other arm.

The paper's context-compression family has no clean, separate Pact analog. `costate_digest.py` and
graph reconstruction compress a large corpus into a usable decision context, so they share its
efficiency role; they do not update the underlying competence store. This is why context compression
is grouped with the context surface rather than counted as a fourth Pact learning mechanism.

### 2.2 Distillation surface: stable accumulation is real; stale-fact resistance is incomplete

| Audit item | Finding |
|---|---|
| Paper's measured failure mode | Distillation accumulates competence more stably than prompt evolution but resists selective correction: updates can be spread diffusely, new facts remain weak, and old/stable facts may be eroded. |
| Existing Pact mitigation | Canonical-equation events are append-only; explicit `domain_refined` and `deprecated` events preserve history; equation queries and the curriculum pool use latest-row-wins per stable ID; records carry domain, residual, source, empirical anchors, and reactivation criteria; the canonical posterior validator can fail closed on several stale/invalid states; a posterior-driven equation recalibrator exists. |
| Residual gap | Latest-row-wins works only when a correction uses the same stable ID. Semantically conflicting rows under different equation/candidate IDs can both look current. Most `reactivation_criteria` are human-readable custody, not executable predicates. The generic validator does not classify a record against `lineage_id × change_axis × valid_time`, and recalibration is not an automatic consequence of every qualifying result. Nothing universally blocks dispatch when a newer observation changes the old record's domain without explicitly superseding its ID. |
| Concrete ticket | `cl_executable_supersession_reactivation_guard_v1`: typed validity interval and lineage/change-pattern scope, explicit `supersedes_ids`, executable reactivation predicate, semantic-conflict set, and fail-closed read/dispatch check with forward-regression evidence. Staged as `needs-build`; no equation is minted. |

#### Does Pact structurally answer the paper's measured stale-update failure?

**DERIVED verdict: PARTIAL MATCH.** Append-only historical provenance plus explicit supersession is
better than overwriting model weights invisibly. It makes correction auditable and lets a reader
recover the old domain. But the paper's stale-fact problem is about **selective update behavior**,
not merely custody. Pact structurally matches the cure only when all of the following hold:

```text
same stable concept id
  + newer valid evidence
  + explicit domain/time/lineage relation
  + supersession or refinement event
  + consumer reads the reduced posterior
  + retention and forward checks pass
```

Today, the first, third, and fifth terms can be omitted by a producer or bypassed by a consumer.
`latest-row-wins` is therefore necessary but not sufficient.

The next guard should refuse a “current” read when two active records in the same semantic conflict
set disagree and neither supplies an executable scope relation. Reactivation should likewise be a
predicate evaluated against current context, not prose that a future agent must remember to apply.

### 2.3 RL surface: safeguards are sound, causal credit is not identified

| Audit item | Finding |
|---|---|
| Paper's measured failure mode | Online RL is strongest for selective factual updates but is noise-sensitive; noisy supervision can damage retained or forward competence. |
| Existing Pact mitigation | The organ uses chronological walk-forward folds and past-only regime labels; #436 routes transient/plateau/uncertain states and defers under meta-lambda surprise; arm evaluation uses the union residual across class outputs rather than one attractive lane; real-only backtest gates prevent synthetic adoption; FORE requires target-action occupancy support and transition-complete logging; HCM requires run-level identification and matched/randomized evidence rather than treating folds as independent runs. |
| Residual gap | #436 is a fixed, same-trajectory predictive dispatcher, not an online RL learner. The current envelope has one real trajectory, its rules were derived from that trajectory, and the regime boundary is not out-of-sample. Existing logs do not establish target-action positivity, complete `(Z,A,R,Z')` transitions, propensities, or cross-run causal support. HCM correctly gives `NO-GO` for causal campaign credit. Reward noise is additionally amplified by nonlinear score composition and coupled Seg/Pose/rate effects. |
| Concrete ticket | Extend `cl_lineage_change_mechanism_selector_v1` with a hard `RL_CREDIT_ALLOWED` predicate requiring transition-complete logs, support, reliable realized-through-R reward, union-residual safety, cross-fitted FORE, HCM whole-run uncertainty, and current/retention/forward non-regression. Until then, RL mode is structurally unreachable. |

**DERIVED verdict:** the current apparatus has the right *refusal machinery* for RL but not the data
that would make RL credit identifiable. Calling the current GEPA reflection or #436 backtest “the RL
layer” would overstate authority. They are candidate generation and guarded predictive selection.

## 3. Mechanism per change pattern for the organ

### 3.1 Two axes that must not be collapsed

Define the outer change state at decision time (t):

\[
C_t = (L_t, D_t, Q_t, O_t)
\]

where:

- (L_t \in \{\text{same-lineage},\text{related-lineage},\text{new-vehicle}\}\) is **space shift**;
- (D_t \in \{\text{stable},\text{slow-drift},\text{abrupt-selective},\text{noisy/unknown}\}\) is
  **time-change pattern**;
- (Q_t) records evaluator-reward reliability and union-residual safety;
- (O_t) records action/transition support and causal-identification readiness.

#436's existing `transient / plateau / uncertain` classifier is an **inner within-run regime**. It
must remain nested below (C_t); it cannot infer whether the run belongs to the same vehicle family,
nor whether a cross-run arm update is supported.

### 3.2 Fail-closed selection rule

```text
if lineage is new/unknown:
    mechanism = RETRIEVED_CONFIG_PRIOR_SHADOW
    shared_surrogate_update = REFUSE
    rl_credit = REFUSE
    require matched shadow evidence and lineage fingerprint

elif change is noisy/unknown or realized reward is unreliable:
    mechanism = PERSISTENCE_OR_DEFER
    collect transition-complete evidence

elif change is slow/recurrent within a supported lineage:
    mechanism = LINEAGE_CONDITIONED_DISTILLATION_REFRESH
    require chronological holdout + prior-retention + forward-transfer non-regression

elif change is abrupt/selective and reward is reliable:
    if target-action support + complete transitions + FORE + HCM gates pass:
        mechanism = RL_STYLE_CREDIT_UPDATE
    else:
        mechanism = GUARDED_ONE_SHOT_ARM_EVAL_OR_DEFER

then apply #436 inner routing:
    transient -> eligible GP/costate arms
    plateau   -> persistence/prototype family
    uncertain -> persistence/defer
```

This ordering makes the paper's mechanism taxonomy a decision rule rather than a ranking:

| Change pattern | Prompt-ish retrieved prior | Distilled surrogate refresh | RL-style credit |
|---|---|---|---|
| New vehicle / new lineage | **GO as reversible cold start only.** Retrieve related domain priors and run them in shadow; never call this learned competence. | **NO-GO initially.** After repeated, held-out, lineage-consistent evidence, fit a lineage-conditioned surrogate without overwriting other lineages. | **NO-GO.** A new lineage is out of support until actions, propensities, rewards, successors, and overlap exist. |
| Related lineage with sparse evidence | **GO with explicit source-lineage distance and uncertainty.** | **CONDITIONAL GO** only with replay across source and target lineages plus retention/forward gates. | **NO-GO** unless target-action support is independently demonstrated. |
| Slow within-run drift | Config prior becomes a baseline, not the update mechanism. | **Preferred update mechanism** when drift is recurrent and chronological holdout supports it; keep append-only surrogate versions and domain validity. | Usually unnecessary; use only if the change is selective/action-mediated and the causal gate passes. |
| Abrupt selective change, clean evaluator reward | Use priors to enumerate safe arms. | May smear the update; use only with selective residual/replay controls. | **Preferred in principle**, but only after the FORE/HCM/logged-exploration predicate passes. |
| Noisy or coupled change | Do not prompt-evolve from noise. | Conservative refresh with replay, or no update. | **REFUSE**; paper and local apparatus agree that noise sensitivity is the dominant risk. |

### 3.3 Feed into #436 without reopening settled results

Keep the current fixed policy and its honest scope:

- **MEASURED locally:** on one nine-interval trajectory, the fixed dispatcher beat the global
  single-best and persistence means, but the fold sign test was not significant and the policy and
  thresholds were derived from the same trajectory.
- **MEASURED locally:** the live-state surprise guard deferred a nominal transient classification to
  `uncertain`, demonstrating a valuable “use nothing” branch.
- **UNKNOWN:** the transient/plateau boundary generalizes to independent trajectories.

The proposed selector is an **outer gate**, not a replacement arm:

```text
lineage/change gate -> mechanism eligibility -> existing #436 regime classifier -> eligible arm
                     -> union-residual and surprise guards -> record outcome
```

Required future record fields:

```text
run_id, vehicle_lineage_id, parent_lineage_ids, change_axis, change_pattern,
state_z, action_arm, behavior_propensity, target_propensity, reward_components,
successor_state_z, union_residual, scorer_axis, archive/runtime hashes,
adapt_current, retain_prior, transfer_forward, mechanism_selected, refusal_reason
```

No new flag, DSL stage, arm, or launch is authorized by this memo.

## 4. GEPA reconciliation

### 4.1 Local and paper claims are not the same experiment

**Local MEASURED result:** `.omx/research/aniso_perclass_lambda_433_20260711.md` reports that the
second `gepa_reflection.py` cycle proposed 11 fixed candidates and all 11 were refused on the extended
same-trajectory walk-forward tournament. The local conclusion is exactly scoped: “RL/post-training
does NOT help NOW at n=1.” The implementation is explicitly GEPA-*style*: template-grounded
reflection over architecture reports, not a reproduction of Harrington et al.'s sequential benchmark
or a claim of algorithmic parity with GEPA.

**Paper MEASURED profile:** GEPA-like prompt evolution can adapt rapidly to a current domain, yet its
gains bleed forward; the authors' domain result depends on retaining separate task prompts. On noisy
temporal data it can peak and then degrade as prompt heuristics overfit.

**Reconciliation: RESCOPE.**

- It does **not CONFIRM** the local negative: the local cycle did not show the paper's initial spike,
  uses a different generator, objective, substrate, sequential unit, and sample regime.
- It does **not FALSIFY** the local negative: no paper result supplies benefit on Pact's costate
  candidates, evaluator reward, or one-trajectory arm tournament.
- It **strengthens the verdict scope**: preserve `NO HELP NOW, n=1, local GEPA-style reflection,
  current candidate space`; do not write “GEPA does not work.”
- It exposes a missing measurement: local evaluation has present/within-trajectory fit but not the
  paper's explicit prior-retention and future-lineage-transfer profile.

### 4.2 Honest re-probe condition

Stage `gepa_sequential_lineage_forward_transfer_reprobe_v1` in the reformulation queue. Do not run it
until at least three independent real run/vehicle lineages exist and the transition/authority fields
above are complete.

Compare under an equal candidate-generation/teacher-call budget:

1. shared evolving reflection policy;
2. separate policy memory per vehicle lineage;
3. retrieval-only fixed prior;
4. no-update persistence/registered #436 baseline.

At every lineage transition, report:

- `adapt_current`: held-out current-lineage union-residual/score-component debt;
- `retain_prior`: replay on every prior valid lineage;
- `transfer_forward`: frozen-policy performance on the next unseen lineage;
- proposal/refusal counts, candidate cost, reward variance, and exact authority surface.

Adoption requires current improvement **and** bounded prior/forward regression. A current spike alone
is a refusal under the apparatus CL criterion.

## 5. Concrete staged tickets

| Candidate | Status | Purpose | Promotion gate |
|---|---|---|---|
| `cl_lineage_change_mechanism_selector_v1` | `needs-build` | Outer lineage/change classifier and fail-closed mechanism eligibility for #436. | ≥3 independent real lineages; chronological current/retention/forward evaluation; union-residual safety; deterministic replay. |
| `cl_executable_supersession_reactivation_guard_v1` | `needs-build` | Make validity/supersession/reactivation executable across semantic conflict sets and consumers. | Positive stale-record refusal, legitimate reactivation, cross-ID contradiction, and legacy latest-row compatibility tests. |
| `gepa_sequential_lineage_forward_transfer_reprobe_v1` | `reformulation-queue` | Re-test only the missing sequential behavior, not re-run the settled n=1 cycle. | ≥3 real lineages; equal budget; separate-vs-shared policy arms; current/retention/forward metrics; real-only authority. |

Rows are staged through `record_candidate(..., path=<owned research JSONL>)` in
`.omx/research/cl_apparatus_mapping_candidate_rows_20260713.jsonl`. They are **not** in the shared
canonical pool and need a main-agent review/flip before becoming controller-visible.

## 6. Triality and six-hook wire-in

- **DSL:** N/A-with-reason. These are apparatus and state-evolution designs; no typed selector,
  stage, or lever exists. Inventing a flag would violate typed-DSL discipline. Each staged row carries
  this explicit N/A reason.
- **DAG:** staged FEED in `.omx/research/cl_apparatus_mapping_dag_feed_20260713.md`; no shared DAG was
  edited.
- **Equations:** N/A-with-reason. No witness-energy term or empirical law was measured. The selector
  is a fail-closed decision protocol.
- **Sensitivity map:** future mechanism outcomes must emit component-specific realized-through-R
  deltas before a candidate can influence sensitivity. No contribution from this memo.
- **Pareto constraint:** union residual and current/retention/forward non-regression are binding
  eligibility gates; no proxy-only improvement can pass.
- **Bit allocator:** N/A until an eligible arm produces exact byte and score-component effects.
- **Cathedral/autopilot:** the eventual governed consumer is the outer eligibility gate before #436;
  this memo does not wire it.
- **Continual-learning posterior:** future empirical rows must append mechanism, change pattern,
  lineage, and three-way CL metrics. No posterior update is warranted by paper evidence alone.
- **Probe-disambiguator:** the staged sequential GEPA comparison arbitrates shared versus per-lineage
  policy memory; both defensible interpretations are retained.

## 7. Falsifiers and stop rules

This design should be rejected or narrowed if any of the following occurs:

1. Independent lineages do not admit a stable, leakage-free lineage/change classifier.
2. Retrieved priors do not beat a metadata-matched random retrieval baseline on decision quality.
3. Executable supersession blocks valid historical equations more often than it prevents stale reads.
4. A distilled refresh improves current lineage but regresses any still-valid prior lineage beyond a
   preregistered tolerance.
5. FORE finds zero target-action support or HCM retains run-level causal `NO-GO`; RL credit remains
   unreachable rather than being rescued by clipping or fold multiplication.
6. The GEPA re-probe has fewer than three independent lineages, reuses the same trajectory as
   independent samples, or lacks future-lineage evaluation; keep the current n=1 refusal and stop.

## 8. Stores consulted

- `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`
- top entries of `~/.claude/projects/-Users-adpena-Projects-pact/memory/MEMORY.md`
- `.omx/state/lane_registry.json`; `.omx/state/subagent_progress.jsonl`
- latest sister findings/session summary and operator directives dated 2026-07-13
- `.omx/research/graph_memory_dag_reconstruction_20260710.md`
- `.omx/research/lens_retrieval_346_wirein_landed_20260712.md`
- `src/tac/graph_memory/{model,recall,query_tools}.py`; `tools/corpus_query.py`;
  `tools/costate_digest.py`; `src/tac/subagent_contract.py`
- `src/tac/canonical_equations/registry.py`;
  `src/tac/canonical_posterior_read_validator/__init__.py`;
  `src/tac/witness_dsl/curriculum_candidate_pool.py`
- `.omx/research/organ_regime_conditional_dispatch_436_20260711.md`;
  `src/tac/witness_control/regime_dispatch.py`
- `.omx/research/aniso_perclass_lambda_433_20260711.md`;
  `src/tac/witness_control/gepa_reflection.py`
- `.omx/research/fore_occupancy_ratio_dig_20260713.md`;
  `.omx/research/hcm_causal_attribution_dig_20260713.md`
- `.omx/research/papers_checked_costate_organ_seal_wave_20260711.md`
- Harrington et al., arXiv:2607.07847 and official project page, read 2026-07-13

## 9. Pointer-delta honesty

This unit changes no run, scorer, candidate archive, frontier pointer, canonical equation, DSL,
shared pool, shared DAG, or papers-checked ledger. It contributes a derived apparatus audit, an outer
selection rule for future #436 work, three staged candidate rows, and a precise GEPA verdict update
for main-agent review. All score effects are **UNKNOWN**.
