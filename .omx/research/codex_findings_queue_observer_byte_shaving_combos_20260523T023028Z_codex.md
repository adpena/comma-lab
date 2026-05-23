# Codex Findings: Queue Observer, Runtime Profiles, And Byte-Shaving Combos

UTC: 2026-05-23T02:30:28Z

## Scope

Implemented durable infrastructure for the operator directive to stop doing
turn-by-turn serial local runs and instead load queues, watch live telemetry,
and let local CPU/MLX work feed generalized post-training byte shaving.

## Landed Surfaces

- `tools/canonicalize_local_training_runtime_profile.py`
  - Canonicalizes standalone `trainer_runtime_profile_observation.v1` and
    embedded representation-training runtime profiles.
  - Emits a normalized runtime summary and optional
    `optimizer_candidate_queue_v1` planning queue.
  - Preserves false-authority boundaries for local CPU/MLX/MPS/CUDA timing.

- `src/comma_lab/scheduler/experiment_queue_observer.py`
  - Adds a compact queue observation surface: status counts, running/queued
    steps, log tails, expected artifact validity, process matches, and
    operator commands.
  - Evaluates queue postconditions, including JSON equality and
    false-authority gates, instead of treating path existence as success.
  - Wired into `tools/experiment_queue.py observe`.

- `src/tac/optimization/byte_shaving_campaign.py`
  - Models the post-training shaving question as `(unit, operation, k)` over
    pairs, frames, byte ranges, archive sections, tensors, and packet members.
  - Emits both independent prefix ladders and bounded combination ladders.
  - Combination planning accounts for conflict sets, interaction deltas, shared
    overhead bytes, extra saved bytes, operation alternatives, and scorer-axis
    quality costs.
  - Prefix ladders now mark conflict violations and never recommend a
    conflict-invalid prefix.
  - Can derive planning-only byte-range units directly from a usable
    master-gradient anchor, preserving the diagnostic authority boundary.
  - Remains planning-only until materialization, locality/inflate checks,
    runtime proof, and exact auth eval.

- `tools/plan_byte_shaving_campaign.py`
  - Operator CLI for building JSON/Markdown byte-shaving plans from signal
    surfaces, saved-byte optimizer queues, or master-gradient archive anchors.

- `src/tac/optimizer/candidate_queue.py`
  - Ingests `byte_shaving_campaign_plan.v1` into
    `optimizer_candidate_queue_v1` as false-authority planning rows for
    prefixes and combinations.
  - Preserves source signal refs, auth-eval refs, MLX calibration refs, scorer
    response refs, selected operations, interactions, and materializer blockers.

- `tools/harvest_dqs1_local_first_result.py`
  - Refuses to overwrite existing harvest or exact-auth-request artifacts.

## Research Links Consumed

- EqR / Equilibrium Reasoners: useful as attractor-depth/restart/halting axes
  for future latent refinement smokes, not score authority.
- Zyphra EqProp/FHN: useful as a forward-only/local-learning probe family only
  after tiny paired sign-correlation and runtime-profile smokes.

Both signals are now represented as optimizer-training axes rather than chat-only
analysis: `attractor_iteration_depth`, `attractor_restart_breadth`,
`adaptive_halting_policy`, and `convergence_residual_window`.

## DQS1 Harvest

Candidate: `pairset_drop_two_r029_010_p0259_0376`

- local advisory axis: `[macOS-CPU advisory]`
- local score: `0.19203961709818362`
- archive bytes: `178558`
- archive sha256:
  `e511fc3642cfce57eee2c77034e3612fc87bed2b95f74f1feeaa475769f99b6e`
- projected contest CPU score: `0.1920291170981836`
- conservative projected contest CPU score: `0.1920321170981836`
- auth frontier pointer: `0.19202828295713675`
- eureka trigger: `false`
- recommended action: `observe_only`

The queue was rerouted to `pairset_drop_two_r028_027_p0257_0378`, and a bounded
local worker was restarted.

Second bounded local worker result:

Candidate: `pairset_drop_two_r028_027_p0257_0378`

- local advisory axis: `[macOS-CPU advisory]`
- local score: `0.19203961709818362`
- archive bytes: `178558`
- archive sha256:
  `262c84ff77599d8b2ea3c255b829e5cb32b4d6755f41e6f50d0df3ebecd19cb7`
- projected contest CPU score: `0.1920291170981836`
- conservative projected contest CPU score: `0.1920321170981836`
- auth frontier pointer: `0.19202828295713675`
- eureka trigger: `false`
- recommended action: `observe_only`

The queue was rerouted to `pairset_drop_two_r029_021_p0259_0371`.

## Verification

- `.venv/bin/python -m ruff check ...` passed.
- `.venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_local_training_runtime_profile_cli.py src/tac/tests/test_optimizer_candidate_queue.py src/tac/tests/test_dqs1_local_first_queue_builder.py`
  passed: `52 passed`.
- `.venv/bin/python tools/plan_byte_shaving_campaign.py --from-master-gradient-archive-sha 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf --output /tmp/pact_mg_byte_shaving_plan.json --repo-root . --master-gradient-low-quantile 0.001 --master-gradient-max-units 4 --max-k 4`
  wrote a planning-only master-gradient byte-shaving plan.
- `.venv/bin/python tools/build_optimizer_candidate_queue.py --source /tmp/pact_mg_byte_shaving_plan.json --output /tmp/pact_mg_byte_shaving_queue.json --repo-root . --top-k 2`
  converted that plan into an `optimizer_candidate_queue_v1` with
  `dispatch_ready=0`.

## Next Integration

1. Add MLX/local-training smoke definitions for BoostNeRV/HNeRV that emit
   runtime-profile observations automatically.
2. Teach trained candidate exporters to emit `byte_shaving_signal_surface.v1`
   from archive grammar, X-ray section entropy, master-gradient sensitivities,
   atoms, and local scorer-response observations.
3. Add a materializer adapter layer that consumes one `selected_operations`
   combo row and emits candidate archives plus locality controls.
4. Feed exact/local outcomes back into interaction priors so combo planning
   learns when operations stack constructively or interfere.
