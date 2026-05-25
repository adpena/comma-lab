# Codex Findings: Pair-Frame Geometry Feedback Queue Autowire

Date: 2026-05-25T17:31:59Z
Agent: Codex
Lane: lane_codex_pair_frame_geometry_feedback_queue_autowire_20260525
Status: implementation landed locally, pending commit serializer

## Finding

The drop-many/pair-frame stack had a real no-signal-loss gap: `tac.optimization.pair_frame_scorer_geometry_lattice` could emit `pair_frame_geometry_queue_executable_drop_request.v1` rows, but the forest-level feedback refresh only carried nearby eureka/drop-two hints as metadata. A queue-executable geometry request still required an operator or agent to manually rebind it into a DQS1 queue.

This was too leaf/manual for the current rate-attack tranche. It meant pair/frame/scorer-geometry signal could be generated, reviewed, and even marked queue-executable while remaining outside the queue-owned feedback loop.

## Change

- Added strict pair-frame geometry request validation in `src/comma_lab/scheduler/dqs1_local_first_queue.py`.
- Added `frontier_rate_attack_pair_frame_geometry_discovery.v1` discovery in `src/comma_lab/scheduler/frontier_rate_attack_feedback.py`.
- Wired discovered pair-frame geometry requests into `build_queue_from_action_summary(...)` as first-class DQS1 local-first selections.
- Added CLI support to `tools/build_frontier_rate_attack_feedback_refresh.py` and `tools/run_frontier_rate_attack_feedback_cycle.py` via `--pair-frame-geometry-lattice`.
- Added `pair_frame_geometry_discovery.json` to the standard feedback-cycle artifact family.
- Added regressions proving queue promotion and nested false-authority rejection.

## Authority

All pair-frame geometry requests remain local planning authority only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- no paid dispatch authority
- no contest CPU/CUDA score authority

Nested truthy authority inside a geometry request now fails through `FrontierRateAttackFeedbackError`, rather than leaking as an unclassified raw exception.

## Eureka Result Interpretation

The current drop-many eureka/greedy artifacts say:

- Drop-two was more interesting than drop-one as a near-boundary acquisition signal, but still advisory.
- The Build-1 pairwise interaction matrix was falsified as an arithmetic artifact because the score field was inherited from the source selector rather than child-candidate empirical evidence.
- The Build-1c greedy reducer found only one measured negative drop-one anchor and all measured K>1 sister anchors regressed versus the K=1 empirical best, so drop-many is not dead, but the current independent greedy evidence defers K>1 promotion.
- The next productive path is not manual leaf shaving: it is queue-executable pair-frame geometry starts plus true empirical K>1 anchors, with mask/feather/receiver variants still blocked until runtime materializers exist.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/dqs1_local_first_queue.py src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/comma_lab/scheduler/frontier_rate_attack_feedback_cycle.py tools/build_frontier_rate_attack_feedback_refresh.py tools/run_frontier_rate_attack_feedback_cycle.py src/tac/tests/test_frontier_rate_attack_feedback.py src/tac/tests/test_dqs1_local_first_queue_builder.py`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py::test_frontier_feedback_compiler_promotes_pair_frame_geometry_requests_to_queue src/tac/tests/test_frontier_rate_attack_feedback.py::test_frontier_feedback_pair_frame_geometry_discovery_rejects_authority_leak src/tac/tests/test_dqs1_local_first_queue_builder.py::test_dqs1_queue_builder_accepts_pair_frame_geometry_queue_requests -q`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_pair_frame_scorer_geometry_lattice.py -q`

Focused result: 56 passed.

## Remaining Work

1. Make the feedback refresh auto-locate freshly generated pair-frame lattice artifacts from the DQS1 tranche runner's output directory in production runs.
2. Generate true Build-1b empirical K>1 CPU anchors so pairwise interaction is not inferred from inherited local planner fields.
3. Add receiver/materializer support for masked and feathered variants; currently only full pairset drop requests are queue-executable.
4. Bind pair-frame geometry request outcomes back into the inverse-scorer action surface and bit allocator.
5. Keep PR95 MLX/HNeRV queue execution and export parity artifacts separate unless their receiver/runtime proof is promoted through the exact-readiness path.
