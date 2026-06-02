# Codex Findings: PR95 Score-Aware Blocker Contract

UTC: 2026-06-02T03:53:00Z

## Verdict

The PR95/HNeRV compact-runner report had stale blocker metadata. A real 4-pair smoke with SegNet and PoseNet teacher bindings still emitted `pr95_segnet_posenet_network_loss_not_wired_to_mlx`, even though the MLX score-aware artifact recorded `has_real_segnet_teacher=true`, `has_real_posenet_teacher=true`, and nonzero SegNet/PoseNet distillation weights.

The runner now derives PR95 exact blockers from the artifact's score-aware metadata. If real joint scorer bindings are present, the report keeps false authority but emits the precise blocker `pr95_mlx_scoreaware_teacher_distillation_is_advisory_not_exact_contest_loss` instead of claiming the scorer loss is unwired.

## Live Smoke

- Output: `/Volumes/VertigoDataTier/pact/compact_carrier_pivots/compact_vq_pivot_30542d3214bfac78/pr95_hnerv_stage8_faithful_scoreaware_4pair_1ep_smoke_20260602T0353Z/`
- Archive bytes: `162790`
- Receiver proof: `receiver_proof_valid=true`, `runtime_consumption_proof_passed=true`
- Stale unwired blocker present: `false`
- Advisory-not-exact blocker present: `true`
- Default-zero scorer-weight blocker present: `false`
- Authority: `score_claim=false`, `ready_for_exact_eval_dispatch=false`

## Verification

- `python -m ruff check tools/run_compact_renderer_mlx_spine_runner.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py` -> passed
- `python -m pytest src/tac/tests/test_compact_renderer_mlx_spine_runner.py -q` -> 38 passed

## Next Action

Use this corrected blocker surface for the PR95 pivot launch rows. The next run should scale from this 4-pair custody smoke to a 600-pair PR95/HiNeRV/SNeRV comparison only when the runner can preserve byte-closed archive export, receiver proof, full-video MLX value replay, and exact-axis blockers without stale readiness fields.
