# Codex Findings: Tilde Aurora Swarm Integration

written_at_utc: 2026-06-04T14:34:36Z
lane_id: lane_tilde_aurora_optimizer_control_20260604
agent: codex
branch_observed: main
scope: Aurora-like MLX optimizer contract plus fail-closed campaign planner gating
score_claim: false
frontier_score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
research_only: false

## Swarm Outcome

- Euclid repaired stale lane-registry evidence for `lane_z8_symbolic_lambda_wavelet_blob_20260601`; `tools/lane_maturity.py validate` now validates 1633 lanes cleanly.
- Einstein implemented `aurora_like` as a real shared MLX optimizer path in `src/tac/substrates/_shared/mlx_score_aware/adapter.py` with tests in `src/tac/substrates/_shared/mlx_score_aware/tests/test_wave_n11_stabilizer.py`.
- Rawls wrote the false-authority SNeRV LF/TUB temporal-gate work order at `.omx/research/codex_findings_wall_attention_snerv_lf_temporal_gate_work_order_20260604T142354Z_codex.md`.
- Parent Codex reconciled the campaign planner so Aurora is no longer treated as plan-only or not integrated. It is a native MLX timing-smoke optimizer candidate, but ordinary long-campaign launch remains blocked until a local convergence/timing smoke exists.

## Aurora Authority Boundary

`aurora_like` is now runnable through the shared MLX score-aware adapter and runner argparse surfaces. It remains Pact-local and false-authority:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- no PR95 source-authority claim
- no exact CPU/CUDA auth-eval claim
- no byte-closed archive/runtime promotion claim

The planner launch blocker is intentionally narrowed to:

- `aurora_requires_local_timing_convergence_smoke`

The stale blocker below is retired because the adapter contract now exists:

- `aurora_not_integrated_with_mlx_score_aware_optimizer_contract`

`aurora_not_pr95_source_authority` remains an optimizer-control authority note, not a launch blocker for the local timing-smoke path.

## Files Advanced

- `src/tac/substrates/_shared/mlx_score_aware/adapter.py`
- `src/tac/substrates/_shared/mlx_score_aware/tests/test_wave_n11_stabilizer.py`
- `src/tac/analysis/nerv_long_training_campaign_plan.py`
- `src/tac/tests/test_nerv_long_training_campaign_plan.py`
- `.omx/state/lane_registry.json`
- `.omx/state/lane_maturity_audit.log`
- `.omx/research/codex_findings_wall_attention_snerv_lf_temporal_gate_work_order_20260604T142354Z_codex.md`

## Verification

```bash
uv run pytest src/tac/substrates/_shared/mlx_score_aware/tests/test_wave_n11_stabilizer.py -q
# 42 passed

uv run pytest src/tac/tests/test_nerv_long_training_campaign_plan.py::test_default_optimizer_kinds_cover_native_mlx_optimizer_surface src/tac/tests/test_nerv_long_training_campaign_plan.py::test_aurora_like_optimizer_row_is_native_mlx_timing_smoke_and_fail_closed -q
# 2 passed

uv run pytest src/tac/tests/test_nerv_long_training_campaign_plan.py -q
# 80 passed in 386.94s

uv run ruff check src/tac/analysis/nerv_long_training_campaign_plan.py src/tac/tests/test_nerv_long_training_campaign_plan.py src/tac/substrates/_shared/mlx_score_aware/adapter.py src/tac/substrates/_shared/mlx_score_aware/tests/test_wave_n11_stabilizer.py
# All checks passed

uv run python -m py_compile src/tac/analysis/nerv_long_training_campaign_plan.py src/tac/tests/test_nerv_long_training_campaign_plan.py src/tac/substrates/_shared/mlx_score_aware/adapter.py src/tac/substrates/_shared/mlx_score_aware/tests/test_wave_n11_stabilizer.py
# passed

uv run python tools/lane_maturity.py validate
# OK -- 1633 lane(s) validated cleanly.

uv run python tools/run_compact_renderer_mlx_spine_runner.py --help | rg -n "optimizer-kind|snerv-score-aware-long-training-optimizer"
# both surfaces list aurora_like
```

## Tiny Aurora Timing Smoke

Executed a bounded manual false-authority smoke on SSD:

```bash
uv run python tools/run_compact_renderer_mlx_spine_runner.py \
  --execute-family hi_nerv \
  --planner-row-id hi_nerv::aurora_like_timing_smoke::aurora_like \
  --allow-bounded-planner-row-timing-smoke-waiver \
  --modelsize-candidate-id manual \
  --allow-unscored-research-smoke \
  --num-pairs 1 \
  --epochs 1 \
  --batch-pairs 1 \
  --learning-rate 1e-3 \
  --optimizer-kind aurora_like \
  --hi-nerv-optimizer-policy native_optimizer \
  --compact-latent-dim 4 \
  --compact-embed-dim 4 \
  --compact-decoder-channel 4 \
  --hard-byte-ceiling 178000 \
  --segnet-distillation-weight 0 \
  --pose-distillation-weight 0 \
  --distillation-device cpu \
  --output-dir /Volumes/VertigoDataTier/pact/aurora_like_hinerv_timing_smoke_20260604T143436Z \
  --overwrite
```

Result:

- report: `/Volumes/VertigoDataTier/pact/aurora_like_hinerv_timing_smoke_20260604T143436Z/compact_renderer_mlx_spine_runner_report.json`
- mode: `executed_hi_nerv_mlx_scoreaware_and_exported`
- archive: `/Volumes/VertigoDataTier/pact/aurora_like_hinerv_timing_smoke_20260604T143436Z/hi_nerv_mlx_training/ema_archive_selection/ema/archive.zip`
- archive bytes: `29811`
- archive SHA-256: `e549a3378e2fb3bb75102273172b6d63f5f24d781259ecad328fd104469ed1fe`
- optimizer telemetry: `native_mlx_optimizer_kind_aurora_like=1.0`
- optimizer policy: `native_optimizer`
- wall-clock seconds for epoch 0: `0.21634888648986816`
- loss: `0.3303888142108917`
- output directory size: `7.6M`
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

This is intentionally not marked as contest empirical authority. It is a one-pair,
one-epoch execution proof and timing datapoint. It does not close full-video,
real-teacher, modelsize, receiver-replay, local-CPU replay, or exact CPU/CUDA
blockers.

## Next Executable Gate

Run matched false-authority timing/convergence smokes for `pact_muon_adamw` and
`muon` using the same 1-pair/1-epoch shape, then a 2-4 pair/3-epoch smoke if the
three optimizers are stable. Required output fields:

- optimizer kind
- seconds per epoch
- loss curve sanity
- SegNet/PoseNet telemetry if available
- native MLX optimizer telemetry flags
- output artifact path on SSD
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

Only after that smoke exists should the planner remove `aurora_requires_local_timing_convergence_smoke` from a candidate long-campaign queue row.
