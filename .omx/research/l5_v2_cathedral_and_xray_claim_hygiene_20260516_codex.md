# L5-v2 Cathedral And Xray Claim Hygiene - 2026-05-16

## Scope

Follow-up from read-only adversarial review agents on L5-v2 Cathedral
integration and Time-Traveler xray claim hygiene.

## Fixes

- Cathedral now passes the canonical TT5L side-info consumption evidence into
  `l5_v2_dispatch_readiness`, matching the operator briefing path.
- L5-v2 readiness now exposes top-level `ready_for_exact_eval_dispatch=false`
  and `rank_or_kill_eligible=false` so the core payload is fail-closed before
  wrappers normalize it.
- Cathedral PacketIR matrix loading now verifies the pinned matrix SHA and
  top-level authority booleans before exposing any exact-eval target templates.
- FOE/foveation and predictive-coding xray modules now mark their outputs as
  planning proxies, not archive-byte or score evidence.
- C1/Z4 campaign ledgers and the Z4 substrate docstring now put primary-source
  URLs/DOIs near the relevant claims and state paired exact-eval, archive SHA,
  runtime custody, and component recomputation blockers.

## Boundary

This patch is hardening only. It does not create a new score claim, promotion
claim, or dispatch authority. L5-v2 remains blocked on empirical anchor,
paired CPU/CUDA axis plan, and architecture-lock probe evidence.

## Verification

- `.venv/bin/python -m ruff check tools/cathedral_autopilot.py src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_cathedral_autopilot.py src/tac/tests/test_l5_staircase_v2.py src/tac/xray/foveation_ego_motion.py src/tac/xray/predictive_coding_hierarchy.py src/tac/substrates/z4_cooperative_receiver_loss/__init__.py src/tac/tests/test_l5_paper_fidelity_claim_hygiene.py`
- `PYTHONPATH=src:. .venv/bin/pytest src/tac/tests/test_cathedral_autopilot.py::test_validation_queue_surfaces_l5_v2_packetir_stack_state src/tac/tests/test_cathedral_autopilot.py::test_l5_v2_validation_queue_suppresses_targets_when_matrix_blocked src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_dispatch_readiness_requires_artifact_evidence_not_booleans src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_dispatch_readiness_accepts_valid_gate_evidence src/tac/tests/test_l5_paper_fidelity_claim_hygiene.py src/tac/xray/tests/test_unified_and_codec_primitives.py -q`
