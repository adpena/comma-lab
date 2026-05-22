# Codex Session Summary: MLX Auth Cache And Dispatch Guards

utc: 2026-05-22T11:13:44Z
agent: codex
status: landed_and_swarm_running
score_claim: false
score_claim_valid: false
promotion_eligible: false
promotable: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false

## Landed

- `d129e8ffe Guard full-mode substrate dispatch devices`
  - Added `FULL_MODE_DEVICE_CPU_BUG_CLASS` to
    `tools/audit_substrate_driver_mode_hardcode.py`.
  - Added the same blocker to `src/tac/deploy/dispatch_protocol.py` Tier 2.
  - Fixed the live Rudin full-mode Modal recipe/driver to resolve
    `RUDIN_FLOOR_DEVICE=cuda` instead of `cpu`.
  - Wrote
    `.omx/research/codex_findings_nscs06_full_mode_device_guard_20260522T105904Z_codex.md`.
- Concurrent/partner landings now on `origin/main`:
  - `98748eba4 Tighten MLX cache false authority contract`.
  - `fc70ddf17 Record decoder-q MLX tensor export dispatch`.
  - `ae8eef73b Support downloaded MLX tensor cache materialization`.

## Verification

- NSCS06/Rudin dispatch guard:
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest
    src/tac/tests/test_rudin_floor_remote_driver.py
    src/tac/tests/test_check_326_substrate_driver_consumes_trainer_mode_env_var.py
    src/tac/tests/test_dispatch_protocol_tool_scope.py` -> `66 passed`.
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_substrate_driver_mode_hardcode.py --format json`
    -> `bug_class_count=0`.
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/check_dispatch_protocol_complete.py
    --recipe .omx/operator_authorize_recipes/substrate_rudin_floor_interpretable_ml_modal_t4_dispatch.yaml`
    -> `dispatch_protocol_complete=true`.
- MLX downloaded tensor cache:
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest
    src/tac/tests/test_materialize_mlx_scorer_cache_from_auth_eval.py
    src/tac/tests/test_mlx_cache_audit.py` -> `24 passed`.
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check
    tools/materialize_mlx_scorer_cache_from_auth_eval.py
    src/tac/tests/test_materialize_mlx_scorer_cache_from_auth_eval.py
    src/tac/local_acceleration/mlx_cache_audit.py
    src/tac/local_acceleration/mlx_preprocess.py
    src/tac/tests/test_mlx_cache_audit.py` -> pass.
  - No-reinflate FEC6/PR101 cache ingestion:
    `tools/materialize_mlx_scorer_cache_from_auth_eval.py --downloaded-tensor-cache-dir
    modal_fec6_pr101_cpu_auth_tensors_20260522T060605Z/fec6_pr101_cpu_auth_tensors_20260522T060605Z/scorer_input_cache_tensors
    --tensor-volume-manifest
    experiments/results/modal_auth_eval_cpu/fec6_pr101_mlx_tensor_export_cpu_20260522T060605Z/scorer_input_cache_tensor_volume_manifest.json`
    wrote
    `experiments/results/mlx_fec6_auth_tensor_cache_local_20260522T1110Z/cache_vs_modal_cpu_auth_tensor_cache_no_reinflate_audit.json`
    with `PASS_CACHE_AUTH_EVAL_IDENTITY`.

## Swarm State

- `019e4f63-b73b-7d03-88ee-0dac1e6216e3` / Raman:
  grayscale LUT export-only checkpoint and soft train deadline worker.
- `019e4f63-d0bf-78f3-a95b-f3fb2bec197a` / Feynman:
  decoder-q auth tensor export dispatch/harvest read-only audit.
- `019e4f63-e3d6-7180-8035-66674c4bbae3` / Cicero:
  recent Markdown/roadmap/axis-label read-only audit.

## Worktree

Only known dirty tracked file after the landings is
`tools/build_hfv1_sparse_sidecar_candidate.py`, which is partner HFV WIP and
was intentionally left untouched. New no-reinflate MLX artifacts are ignored by
existing `.gitignore` rules:

- `experiments/results/*`
- `modal_*_auth_tensors_*/`

## Next Critical Path

1. Harvest Feynman's decoder-q tensor-export status and recover any completed
   Modal CPU auth tensor artifact through `tools/recover_modal_auth_eval.py`.
2. Integrate or review Raman's grayscale LUT export-only patch; then export the
   existing A100 `best.pt` without retraining if the patch is green.
3. Use the passing FEC6/PR101 auth tensor cache as the MLX transfer-calibration
   parent only under `[macOS-MLX research-signal]`, never for score promotion.
4. Keep decoder-q blocked from MLX parent calibration until it has the same
   full-sample `contest-CPU` or `contest-CUDA` auth tensor identity pass.
