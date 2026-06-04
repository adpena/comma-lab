# Codex Session Summary - 2026-06-04T1648Z

## Scope

- SNeRV/HiNeRV remained the priority stack.
- Modal is treated as exact-auth-eval-only for true frontier candidates.
- PR95 baseline/control work is local CPU/MLX only until a candidate earns the frontier exact-eval gate.

## Landed Artifacts

- Added PR95 baseline identity helper and CLI:
  - `src/tac/analysis/pr95_baseline_identity.py`
  - `tools/build_pr95_baseline_identity.py`
  - `src/tac/tests/test_pr95_baseline_identity.py`
- Bound PR95 identity, upstream `evaluate.py` priority, and Tilde/Parallax OSS policy into:
  - `src/tac/analysis/nerv_long_training_campaign_plan.py`
  - `tools/build_nerv_long_training_campaign_plan.py`
  - `src/tac/tests/test_nerv_long_training_campaign_plan.py`
- Wrote research/control artifacts:
  - `.omx/research/codex_findings_tilde_parallax_snerv_hinerv_20260604T154256Z_codex.md`
  - `.omx/research/pr95_baseline_identity_20260604Tcurrent_codex.json`
  - `.omx/research/pr95_baseline_identity_20260604Tcurrent_codex.md`
  - `.omx/research/nerv_long_training_campaign_aurora_pr95_identity_bound_20260604Tcurrent_codex.json`
  - `.omx/research/nerv_long_training_campaign_aurora_pr95_identity_bound_20260604Tcurrent_codex.md`
  - `.omx/research/nerv_long_training_campaign_aurora_pr95_identity_bound_queue_20260604Tcurrent_codex.json`

## Hashes

- PR95 identity JSON: `ce71edfc22234bda90d327540be02144228957ed95b522bc60d9247539878f3a`
- PR95 identity MD: `263ba12059235262a3d086dc7c8fd6c0bd026646639ad1d7aa03a47fffa5ae6d`
- Bound SNeRV/HiNeRV campaign JSON: `a03541d958d71d37269a56966828bae84875e914fbab1e04b7b7a1cacf5cd8fe`
- Bound SNeRV/HiNeRV campaign MD: `c149f982cd9924100013d3300593ba2f18a03ae8dee11b2542dec9cd16fe48ab`
- Bound queue JSON: `9fe70d65bc50e32468c3ce7bc9b57f2fd7458e6c63c259b2c0f333005b109d72`

## Crux Found

The prior scalar-mean SNeRV run was not rate-limited. Its epoch014799 packet was receiver-proven and byte-light:

- Archive: `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_auto_bytecap_native_rate_aware_training/checkpoint_harvest_epoch014799_ema_mlx_cache_prefilter/snerv_checkpoint_archive_bound_package/archive.zip`
- Archive bytes: `90920`
- Archive SHA-256: `78d378d0752fd1a927b44cdaef48ecebee0d9d718aa9f5a286da3127ce554fa5`
- Full-video MLX score: `101.52194306037438`
- Axis: `[macOS-MLX research-signal]`
- Average SegNet distortion: `0.6159661708275477`
- Average PoseNet distortion: `158.92011693318685`

Verdict: fit failure, especially PoseNet tail, not byte-cap failure. Exact auth eval is not justified.

## Local MLX Successor

Launched local MLX SNeRV hard-pair successor after storage waterfall preflight selected VertigoDataTier with no blockers.

- Lane: `snerv_scalarmean_hardpair_successor_20260604`
- Active job: `snerv_scalarmean_hardpair_successor_fix2_20260604Tcurrent_codex`
- PID: `45607`
- Launch manifest: `/Volumes/VertigoDataTier/pact/experiments/results/snerv_scalarmean_hardpair_successor_fix2_launch_20260604Tcurrent_codex/snerv_scalarmean_hardpair_successor_fix2_launch_manifest.json`
- Manifest SHA-256: `6a7a0dda69474c98bc1e7eef98876c7e63b2b7a761efb88b62e64ecd7d67368a`
- Startup artifact: `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_scalarmean_hardpair_successor_fix2_20260604Tcurrent_codex/compact_renderer_mlx_spine_runner_startup.json`
- Startup SHA-256: `70c9e2e21397cb7861f4093c533950992a4c69de0fb6f71695e661c62ff970ae`
- Hard-pair hitlist: `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_auto_bytecap_native_rate_aware_training/full_video_mlx_feedback_epoch014799_cache_prefilter/snerv_hard_pair_hitlist_top96.json`
- Measured SNeRV byte feedback: `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_auto_bytecap_native_rate_aware_training/checkpoint_harvest_epoch014799_ema_mlx_cache_prefilter/snerv_checkpoint_archive_export.json`

Two failed launches were terminalized and preserved:

- `snerv_scalarmean_hardpair_successor_20260604Tcurrent_codex`: refused because invented planner row id was not in the queue artifact.
- `snerv_scalarmean_hardpair_successor_fix1_20260604Tcurrent_codex`: refused because the measured epoch014799 SNeRV byte feedback was missing from the relaunch command.

Fix2 is the live corrected process. It selected the calibrated measured SNeRV row with predicted archive bytes `90920` and predicted headroom `87080`.

## Validation

- `uv run ruff check src/tac/analysis/pr95_baseline_identity.py tools/build_pr95_baseline_identity.py src/tac/tests/test_pr95_baseline_identity.py src/tac/analysis/nerv_long_training_campaign_plan.py src/tac/tests/test_nerv_long_training_campaign_plan.py tools/build_nerv_long_training_campaign_plan.py` passed.
- `uv run pytest src/tac/tests/test_pr95_baseline_identity.py src/tac/tests/test_nerv_long_training_campaign_plan.py::test_long_training_campaign_plan_binds_pr95_baseline_identity -q` passed: 4 tests.
- `uv run pytest src/tac/tests/test_nerv_long_training_campaign_plan.py src/tac/tests/test_pr95_baseline_identity.py -q` passed: 87 tests.
- `uv run python tools/lane_maturity.py validate` passed: 1641 lanes.

## Next

- Poll PID `45607` and harvest the next checkpoint export.
- If full-video MLX improves enough, run local CPU advisory replay.
- Only if a byte-closed candidate clears local gates and plausibly beats the frontier, use Modal for exact auth eval.
