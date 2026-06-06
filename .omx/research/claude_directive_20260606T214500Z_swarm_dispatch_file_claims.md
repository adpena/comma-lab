# Claude directive — SWARM dispatch file claims (operator: "orchestrate a swarm")

UTC: 2026-06-06T21:45:00Z. Parent: claude main session. Six workflow agents in
the SHARED tree, serializer-arbitrated. Sister codex: please avoid these files
until the swarm completes (TTL ~2h).

| Agent | Claims (NEW unless noted) |
|---|---|
| swarm-A survival | `src/tac/substrates/hi_nerv/birth_survival.py`, `src/tac/substrates/hi_nerv/tests/test_birth_survival.py` |
| swarm-B composite | EDIT `src/tac/substrates/hi_nerv/mlx_renderer.py` (fit_target_region_birth_from_segnet only) + `src/tac/substrates/hi_nerv/tests/test_target_region_birth.py` (append) + `target_region_birth.py` (receipt fields) |
| swarm-C miner | `src/tac/analysis/hinerv_hard_region_miner.py`, `tools/mine_hinerv_hard_regions.py`, `src/tac/tests/test_hinerv_hard_region_miner.py` |
| swarm-D action-effect | `src/tac/analysis/action_effect.py`, `src/tac/tests/test_action_effect.py` |
| swarm-E commutator | `src/tac/analysis/action_commutator.py`, `tools/run_pr110_commutator_ledger.py`, `src/tac/tests/test_action_commutator.py` (phase 2, after D) |
| swarm-F pose-thread | EDIT `tools/run_compact_renderer_mlx_spine_runner.py` (thread pose teacher into birth-actuator callsite only) + v5 smoke artifacts on SSD + `.omx/research/hinerv_hard_birth_pose_threaded_v5_receipt_20260606.md` |

NOT claimed (codex-owned): `nerv_witness_readiness_dag.py`,
`nerv_pair_local_distortion_servo.py`, all `snerv_*`, `score_geometry.py`.
Lanes: `lane_hinerv_target_region_score_debt_smoke_20260606` (A/B/F),
`lane_receiver_replay_scorer_hard_region_miner_20260606` (C),
`lane_action_effect_thin_ir_20260606` (D),
`lane_pr110_pairwise_commutator_ledger_20260606` (E).
