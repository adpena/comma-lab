# Codex Findings: DQS1 Component-Marginal Feedback Cycle

- UTC: 2026-05-25T21:27:04Z
- Agent: codex
- Scope: DQS1 eureka/drop-many observation feedback, queue-owned local-first replanning
- Authority: false authority only; no score claim, promotion, rank/kill, paid dispatch, or exact-eval readiness

## Finding

The May 25 eureka/drop-many acquisition signal and DQS1 local-first observation
JSONLs were not fully default-wired into the feedback cycle. The queue could
consume explicit observation paths, and the component-marginal canonicalizer
could train from those observations, but default `latest` selection could still
resolve to stale `experiments/results/cross_family_candidate_portfolio/**`
summaries while newer DQS1-safe summaries lived under `.omx/research`. Existing
`.omx/research/dqs1_local_first_harvest_observations_*.jsonl` rows also required
manual path handoff before they influenced queue skip policy or component-model
replanning.

## Landing

- `find_latest_cross_family_action_summary(...)` now scans both
  `experiments/results` and `.omx/research`, schema-checks DQS1-safe action
  summaries, and prefers embedded/generated timestamps when present.
- `build_frontier_rate_attack_feedback_refresh(...)` now discovers DQS1
  local-first observation JSONLs from frontier artifact roots, with a default
  `.omx/research` discovery path, and reports the discovery artifact.
- `run_frontier_rate_attack_feedback_cycle.py` now canonicalizes discovered
  DQS1 observations into a pairset component-marginal action summary before
  building the next local-first queue.
- Component-marginal cycle bundles dedupe cumulative observation snapshots by
  canonical observation identity before training the planning model.

## Concrete Artifact

Generated cycle:

- `.omx/research/codex_frontier_rate_attack_component_marginal_cycle_20260525T212704Z/frontier_rate_attack_feedback_cycle.json`
- Component summary:
  `.omx/research/codex_frontier_rate_attack_component_marginal_cycle_20260525T212704Z/initial_component_marginal_refresh/action_summary.json`
- Follow-up queue:
  `.omx/research/codex_frontier_rate_attack_component_marginal_cycle_20260525T212704Z/initial_refresh/dqs1_followup_queue.json`

Observed facts:

- DQS1 observation discovery found 6 cumulative JSONL snapshots and 20 unique
  deduped local advisory observations.
- The component-marginal model is active on `macos_cpu_advisory`.
- The generated local-first queue validates cleanly with 8 experiments.
- Selected next candidates were
  `pairset_drop_one_rank010_pair0376`,
  `pairset_drop_one_rank031_pair0296`,
  `pairset_drop_one_rank027_pair0378`,
  `pairset_drop_one_rank026_pair0320`,
  `pairset_drop_one_rank022_pair0167`,
  `pairset_drop_one_rank021_pair0371`,
  `pairset_drop_one_rank020_pair0430`, and
  `pairset_drop_one_rank019_pair0151`.

## EUREKA JSON Interpretation

The latest eureka acquisition JSON produced a broad candidate pool, but its top
rankings were inherited local planning estimates rather than child empirical
score authority. The useful current result is not "drop many is proven good";
it is that observed drop-one/drop-two component deltas now train a reusable
component-marginal planning model that suppresses already-observed work and
chooses the next unobserved local candidates automatically.

Drop-two remains an interesting local advisory signal because some paired rows
showed lower advisory score than individual drop-one rows, but exact authority
is still absent. The queue is therefore correct to continue local-first
materialization/harvest rather than promote or dispatch.

## Verification

- `ruff` on touched feedback, queue, cycle, CLI, and test files: pass.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py src/tac/tests/test_dqs1_local_first_queue_builder.py -q`: 57 passed.
- Generated queue validation inside the cycle report: valid, 8 experiments.
