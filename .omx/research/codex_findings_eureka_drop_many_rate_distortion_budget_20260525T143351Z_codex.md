# Eureka Drop-Many Rate-Distortion Budget Wiring

- timestamp_utc: 2026-05-25T14:33:51Z
- agent: codex
- scope: DQS1 pairset acquisition, frontier feedback eureka hints, operator briefing, queue-owned local follow-up
- authority: local planning only

## What Changed

The local CPU eureka near-miss signal now activates a bounded beyond-drop-two
DQS1 acquisition profile instead of remaining a prose hint. The acquisition
planner emits `drop_many_beam_pairwise_interaction_waterfill` candidates with
explicit eureka provenance, rate-vs-distortion policy metadata, and queue-safe
false-authority fields.

Every candidate row now carries `distortion_repair_budget_from_rate_savings`.
When a pairset candidate saves descriptor bytes relative to the source selector,
the row records the score budget created by the rate saving, the equivalent
SegNet distortion budget at fixed PoseNet, and the PoseNet score-term budget at
fixed SegNet. This makes the "bytes saved can buy PoseNet/SegNet repair" trade
machine-readable for the next optimizer instead of chat-only intuition.

The frontier feedback eureka hint also now declares the broader rate/distortion
levels under consideration: bit, byte, packet member, tensor channel, pixel,
region, boundary, frame, pair, batch, full video, scorer axis, and receiver
runtime. This pass only makes the DQS1 drop-many local-first family executable.
The global low-impact full-pair/frame-drop probe, within-set masked/feather
probe, and inverse-scorer null-direction masked variant are preserved as
explicit blocked family requests until receiver/materializer and pair-frame
scorer-geometry bindings exist.

`tools/operator_briefing.py` now surfaces selected drop-many candidate counts
and the active eureka drop-many profile so this signal is visible in normal
operator briefing and preflight flows. It also keeps the latest eureka-active
refresh visible when a newer retention-only smoke refresh exists, so later
storage hygiene artifacts cannot hide the broader rate/distortion signal.

The shared DQS1 queue builder also now has a coherent retention command helper
for raw and MLX cache retention steps. This preserves the partner retention
execution WIP that was already present in the worktree while making the
frontier feedback refresh call path lint-clean and queue-test-covered.

## Artifacts

- acquisition plan:
  `.omx/research/codex_eureka_beyond_drop_two_acquisition_20260525T143351Z/dqs1_pairset_acquisition_eureka_drop_many.json`
- portfolio:
  `.omx/research/codex_eureka_beyond_drop_two_acquisition_20260525T143351Z/cross_family_portfolio.json`
- action summary:
  `.omx/research/codex_eureka_beyond_drop_two_acquisition_20260525T143351Z/action_summary.json`
- queue refresh:
  `.omx/research/codex_eureka_beyond_drop_two_acquisition_20260525T143351Z/frontier_refresh/feedback_refresh_report.json`
- queue:
  `.omx/research/codex_eureka_beyond_drop_two_acquisition_20260525T143351Z/frontier_refresh/dqs1_followup_queue.json`
- raw-retention execute smoke:
  `.omx/research/codex_retention_execute_smoke_20260525T151000Z/feedback_refresh_report.json`

The generated acquisition has 581 candidates, including 34 bounded drop-many
candidates. The generated follow-up queue has 8 experiments and 48 steps; 3 of
the selected local-first experiments are drop-many starts. All authority fields
remain false.

## Verification

- `ruff` on touched acquisition, feedback, briefing, and test files: passed
- `pytest src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_decoder_q_pairset_acquisition.py src/tac/tests/test_frontier_rate_attack_feedback.py src/tac/tests/test_operator_briefing.py src/tac/tests/test_all_lanes_operator_briefing_gate.py -q`: 130 passed
- `pytest src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_frontier_rate_attack_feedback.py -q`: 49 passed after retention-helper hardening
- `pytest src/tac/tests/test_operator_briefing.py -q`: 43 passed after latest-eureka refresh surfacing
- `pytest src/tac/tests/test_all_lanes_operator_briefing_gate.py -q`: 29 passed
- `tools/experiment_queue.py --queue .omx/research/codex_eureka_beyond_drop_two_acquisition_20260525T143351Z/frontier_refresh/dqs1_followup_queue.json validate`: valid, 8 experiments, 48 steps
- live operator briefing: `status=READY_LOCAL_EXECUTION`; latest refresh is
  the retention execute smoke, and latest eureka refresh remains the
  drop-many queue with `selected_drop_many=3` and
  `eureka_drop_many_counts=[3,4,6,8]`; all score/promotion/rank/dispatch/GPU
  authority fields false
- live operator briefing dispatch gate: frontier feedback passes; remaining failure is the unrelated existing L5 blocker `l5_v2_packetir_matrix_artifact_sha_mismatch`

## Residual Gap

The user is right that global no-impact or near-no-impact frame/pair drops have
not been fully explored. That should be the next canonical bridge: bind pair
component xray rows, frame-axis master-gradient decomposition, SegNet/PoseNet
score geometry, CPU/CUDA axis labels, and receiver feasibility into a
pair-frame scorer-geometry lattice that can generate queue-executable
full-drop, repair, masked, and feathered starts without false authority.
