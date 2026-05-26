# Codex Findings - Targeted Drop-Many DQS1 Bridge - 2026-05-26T08:30:00Z

## Verdict

The targeted drop-many planner is no longer a manual leaf artifact. Its
`decoder_q_pairset_acquisition.v1` output can now be consumed directly by the
DQS1 local-first queue builder, and the operation-chain compiler queue wires
that child queue after `run_targeted_drop_many_pairset_acquisition`.

## What changed

- Added a queue adapter for `decoder_q_pairset_acquisition.v1` rows:
  acquisition rows are false-authority checked, ranked, selector-kind filtered,
  converted into `Dqs1QueueSelection`, and preserved in the
  `dqs1_selected_pairset_acquisition.v1` harvest sidecar.
- Extended `tools/build_dqs1_local_first_queue.py` with
  `--pairset-acquisition` and repeated `--selector-kind` filters so generated
  drop-many/pair-frame candidates can become bounded DQS1 queues without an
  operator rebuilding an action summary by hand.
- Extended the targeted operation-chain queue to build and validate a DQS1
  follow-up queue after pairset acquisition. The child queue path, selected
  acquisition sidecar, results root, selector allowlist, and candidate limit are
  visible in operation-chain queue metadata.
- Threaded existing DQS1 harvest observation JSONLs into the child queue command,
  so already observed candidates remain suppressed across targeted-chain
  replans instead of being re-queued manually.
- Hardened the bridge after adversarial review: direct `--pairset-acquisition`
  CLI builds no longer perform unrelated pair-frame geometry discovery, DQS1
  observation rows must match both the harvest source schema and sweep config
  before influencing bridge posteriors, same-content artifact writes are
  idempotent for worker retries after partial sidecar creation, and generated
  operation-chain child builds carry a stable `--eureka-run-id`.

## Live artifact

Built and validated with existing DQS1 harvest observations threaded into the
bridge and `VertigoDataTier` as the DQS1 workload root:

`.omx/research/frontier_rate_attack_feedback_refresh_20260526T083600Z_targeted_dqs1_followup_vertigo_observation_guard/targeted_drop_many_dqs1_followup_queue.json`

Source acquisition:

`.omx/research/frontier_rate_attack_feedback_refresh_20260526T083300Z_targeted_chain_member_inference/targeted_drop_many_pairset/targeted_drop_many_pairset_acquisition.json`

Selected queue candidates:

- `pairset_drop_many_k012_h1ecc99d178`
- `pairset_drop_many_k006_hd0960b13f2`
- `pairset_drop_many_k012_hc4a6f54207`
- `pairset_drop_many_k016_h1759043cc6`

The selected sidecar preserves the rate-savings repair budget, including
`decoder_q_pairset_rate_saved_distortion_repair_budget.v1`, so freed bytes can
be used by targeted SegNet/PoseNet repair planning without becoming score,
promotion, rank/kill, or dispatch authority.

The paired bridge artifact
`.omx/research/frontier_rate_attack_feedback_refresh_20260526T083600Z_targeted_dqs1_followup_vertigo_observation_guard/dqs1_materializer_feedback_bridge.json`
observed 2 existing DQS1 harvest candidates and recommends updating the DQS1
pairset posterior from harvest observations. This closes the sidecar-review
blocker where targeted child queues could otherwise forget local feedback and
re-queue already harvested candidates.

## Verification

- `ruff` on touched scheduler/tool/test files: passed.
- `pytest src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_frontier_rate_attack_feedback.py -q`: 82 passed.
- Focused adversarial review: no P0/P1 found; P2 side-effect discovery,
  observation-identity, and retry-idempotency findings fixed and guarded.
- Live generated queue validation: `valid=true`, 4 experiments, 28 steps.
- Queue command paths include 72 references under
  `/Volumes/VertigoDataTier/pact/experiments/results/frontier_rate_attack_feedback_20260526T083600Z_targeted_dqs1_followup_vertigo_observation_guard/dqs1_local_first_results`.
- Queue SHA-256:
  `f983942bf89fbc773c5b1b1f8765a43fe4a354d766d6a156c89663cc3afcd3f7`.

## Authority

All generated queue, selected-acquisition, and live-artifact signals remain
false-authority. They are local-first planning and materialization inputs only;
exact CPU/CUDA auth eval is still required before score, promotion, rank/kill,
or dispatch decisions.
