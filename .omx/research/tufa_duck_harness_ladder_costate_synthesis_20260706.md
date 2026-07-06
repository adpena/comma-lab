# Tufa Labs (Duck Harness + LADDER) → our costate controller — research synthesis

**Date:** 2026-07-06. **Trigger:** operator "deeply research [tufalabs.ai/research/duck-harness]
and all related and follow up and OSS ... we probably have versions of this that can be
enhanced and hardened and possibly beautiful abstractions and meta ... we have a ton of work
to do on our costate controller." Two parallel research agents (Tufa body-of-work; duck-harness
OSS) + internal inventory. This memo = the durable artifact (per "Results must become system
intelligence"; NOT chat-only).

## 1. What the two external systems ARE

**Duck Harness** (Tufa Labs, 2026-07; ARC-AGI-3 Milestone-1 winner; `github.com/Tufalabs/duck-harness`,
~40★, **NO LICENSE FILE → all-rights-reserved, adopt IDEAS not SOURCE**). An LLM-agent that plays
grid puzzles whose goal is hidden. The architecture worth stealing:
- **REPL-state**: all observations are bound as Python variables in a sandbox namespace
  (`current_frame`, `previous_frame`, `history`, `transitions`, `valid_actions`, `last_action_result`).
- **Coding-as-reasoning, single tool**: the model's ONLY tool is `python{code}` — it writes code to
  inspect state + a `segment_layer()` structured graph (objects/adjacency/containment), and acts via
  an `action(...)` primitive inside the snippet; isolated subprocess, whitelisted builtins.
- **World-model-carry, not eviction-as-memory**: durable state promoted into a compact re-injected
  "working world model" (goal model + world model, "revise immediately if `current_frame`
  contradicts it"); the raw transcript is evicted by a token-budget oldest-block dropper.
- **Action-efficiency objective**: scored on FEWEST simulator steps → reason extensively in Python
  ("free thinking") before spending one expensive `action()`. ~10× cheaper/game than GPT-5.4
  Executable-World-Models (arXiv 2605.05138). Qwen-3.6-27B-FP8, single Kaggle GPU.

**LADDER** (Tufa Labs, [arXiv:2503.00735](https://arxiv.org/abs/2503.00735), Simonds & Yoshiyama).
*"...autonomously improve ... by recursively generating and solving progressively simpler variants
of complex problems."* A **verifier-gated difficulty-gradient loop**: build an easier variant where
the hard thing is WINNABLE, learn it, recurse toward the true target; advance ONLY on
verifier-confirmed improvement; dead/unsolvable proposals absorbed via zero reward (~8% ignored).
Hard requirement: a reliable verifier + a "verifier–generator gap" (cheap to check, hard to produce).
Results: integration 1%→82% (Llama-3.2-3B); TTRL→90%. **TTRL is Tufa's, inside LADDER — NOT the
different-group arXiv:2504.16084.** Related Tufa line: MindsAI/ARC **test-time-training** philosophy
(per-task on-the-fly adaptation) — "Don't Throw the Baby Out…" (Cole & Osman), StochasticGoose
(ARC-AGI-3 preview 1st), Self-Rewarding (2505.08827), Self-distillation predictive law (2605.30070).

## 2. What WE already have (the operator's "versions of this")

- `src/tac/witness_control/` — the **θ* costate controller, Phase A** (task #303): `costate_estimator.py`
  (659 L, λ=∂S/∂x with 3 honesty tiers ANALYTIC/…), `shadow_controller.py` (409 L,
  observe→estimate→recommend→**STOP**; actuation structurally impossible). Design memo
  `.omx/research/costate_controller_design_20260705.md` frames it as the **Hamiltonian meta-layer**:
  the triality {DAG=state x(t) · DSL=control u(t) · equations=law S} completed by its 4th object, the
  costate λ = marginal-ΔS shadow price (canonical eq `costate_lambda_marginal_ds_20260705`).
- `src/tac/witness_dsl/{curriculum_dsl, gauge, campaign, powerplay}` — the DSL (the lever action-space).
- The DAG (`.omx/research/sub015_DAG_*`) = history; tools (`witness_training_time.py`, per-stage
  attribution, margin-saliency) = helper functions.
- Phase B (actuation) is DESIGN-ONLY, gated on operator GO + the Contrarian's condition (no
  horizon/projection-based argv emission until a decay-aware λ(t) is measured; rollback/stop are
  projection-free and exempt). Only TWO controls are closed-loop in-run today (bounded eikonal bump +
  early-stop arming); everything else is open-loop launch config.

## 3. The unification (the "beautiful meta")

**Our costate controller is the abstraction that SUBSUMES both external systems; they tell us how to
build its missing pieces.** The costate λ is the common currency:
- LADDER's "advance only when the verifier confirms improvement" **IS** "take the control step only
  when λ·Δx lowers S" — LADDER is a *scalar, hand-built* instance of costate-gated progression; ours
  is the *measured, multi-channel* law.
- Duck's "fewest actions" **IS** "maximize |ΔS| per paid dispatch" — action-efficiency is the costate
  controller's objective written in our scarce resource (byte-closed exact-eval / GPU dispatch, the
  <$5 Modal budget, the means/ends firewall).
- Duck's "REPL-state + coding-as-reasoning" is the **inspection/decision I/O architecture** our
  controller lacks — today it reads run.log and emits a recommendation; it is not a namespace an
  orchestrator inspects-as-objects and composes-helpers-as-code.

So: **costate controller = the law; Duck = the orchestrator I/O; LADDER = the control law for one
specific actuation (curriculum/island birth).**

## 4. Concrete ENHANCE / HARDEN / ABSTRACT plan (all → the costate controller)

1. **Campaign-REPL inspection surface (Duck pattern; highest value).** Give the (governed)
   orchestrator a namespace binding our state as typed objects — `current_run`(d_seg, per-class
   part_frac, s/epoch, loss terms, stage), `previous_run`, `history`(DAG rows), `frontier`(pointer
   0.19110), `valid_levers`(DSL flag space), `last_dispatch_result` — plus our tools as helper
   functions (`per_stage_attribution`, `margin_saliency`, `compare`, `project_S`), and ONE governed
   `emit_argv(cfg)` action that NEVER auto-fires paid GPU (CONTAINMENT). Coding-as-reasoning makes the
   emitted code the audit trace (max-observability). This is the missing surface that unifies
   DSL+DAG+equations+tools that the costate controller already reasons over.
2. **Action-efficiency as the controller's explicit objective.** Optimize *dispatches-to-a-lower-exact
   -score*, cheap n600/JSONL inspection = "free thinking" between expensive actions. Directly encodes
   the means/ends firewall: every launch justified by cheap REPL analysis first.
3. **LADDER-ize island birth (the operator's live frustration) — per-class-λ-gated difficulty
   homotopy.** THE unification of the island problem with the costate controller: reframe #300/#301/
   #315 not as "turn on an island loss" but as a **verifier-gated difficulty gradient** — make lane
   (0.6%) / movable (1.6%) *winnable* (dilate rare-class GT mask, or crop to windows where the class is
   locally dominant, or temperature-boost those logits), BIRTH them there, then **anneal the assistance
   back toward the true frozen-SegNet argmax**, gating each shrink on the **measured per-class d_seg**
   (which is exactly the costate's named per-class-λ gap #253/#255). The per-class λ IS LADDER's
   verifier signal; the difficulty-shrink is the singular-arc control. This is strictly more principled
   than naive seeding (which starves d_seg, #300) because the shrink is λ-gated, not epoch-timed.
4. **World-model-carry hardening (summary-carry, NOT eviction).** Add `campaign_world_model()` — a
   compact struct (current crux · ranked open levers · last measured deltas · ruled-out) regenerated
   from the DAG each cycle and re-injected. We keep DAG/memory as durable source of truth (anti-forget
   non-negotiable) and use Duck's *summary-carry* idea WITHOUT its eviction-as-memory (which would be
   signal-loss for us).
5. **Trace-row formalization.** Every orchestrator decision + emitted argv → a typed DAG row (Duck's
   `traces.py` machine-readable episodes), closing DAG→DSL→run→rows→equations→next-DAG.

## 5. Do-NOT-adopt (honest firewall)

- **Copy the Duck SOURCE** — no license = all-rights-reserved. Re-implement natively in `tac.witness_*`
  (also better per UNIQUE-AND-COMPLETE-PER-METHOD).
- **RL machinery (GRPO/policy-gradient/reward+format model)** — the witness is differentiable-loss GD,
  not a sampling policy. Reframing it as an RL policy to "use LADDER" is a category error. Take the
  curriculum-gating *control loop*, not the optimizer.
- **Auto-DERIVING the schedule** — LADDER offers nothing; its schedule is hand-built heuristics. Our
  first-principles τ=ε=ħ / persistence-order derivation is strictly more principled. Keep ours.
- **Self-rewarding / label-free** — its whole point is "no verifier." We HAVE the exact verifier →
  moot; treat as confirmation that verifier-gated self-improvement works, not a new tool.
- **Duck's perception stack, high-frequency action loop, goal-discovery, eviction-as-memory, full
  autonomy** — puzzle-vision-specific / assume near-free reversible actions / hidden objective / token
  budget / autonomous spend. Our actions are multi-hour multi-dollar machine-crashing-risk-gated; our
  objective is fixed (sub-0.15); durable memory is non-negotiable; heavy/paid GPU is GO-gated.

## 6. Gating / next step

Items 1, 2, 4, 5 are orchestrator-I/O + observability = shadow-safe (no actuation) → buildable without
new GPU. Item 3 (LADDER island homotopy) is a NEW loss/curriculum lever = a #205 treatment arm →
needs operator GO + governed launcher; the $0 design + a local n600 smoke of "is lane winnable under
dilation?" is shadow-safe and gates it. Phase B costate ACTUATION remains gated on operator GO + the
decay-aware λ(t) condition. Sisters: `[[project_meta_layer_above_triality_hamiltonian_control_costate_20260703]]`
+ `[[why_mod32cap_baseline_has_zero_lane_movable_islands_20260706]]` + costate design memo
`costate_controller_design_20260705.md`.
