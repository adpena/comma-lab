# D4 F1+F4 T0.3 Smoke Blocker 2026-05-14

Status: `blocked_before_modal_spawn`
Lane: `lane_d4_wyner_ziv_frame_0_substrate_20260514`
Scope: D4 F1+F4 recipe/driver smoke path only
Evidence axis: `[contest-CUDA]` smoke target, no score claim produced

## Recipe / Driver Delta

- F1 smoke path pinned to 50 epochs through
  `scripts/operator_authorize_substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch.sh`.
- F4 pair cap pinned to 200 pairs through
  `.omx/operator_authorize_recipes/substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch.yaml`.
- Remote D4 driver now has fail-safe direct-run defaults:
  `D4_WYNER_ZIV_FRAME_0_EPOCHS=50` and
  `D4_WYNER_ZIV_FRAME_0_MAX_PAIRS=200`.
- Remote provenance now records `max_pairs`.

## Local Verification

Commands run:

```bash
bash -n scripts/operator_authorize_substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch.sh
bash -n scripts/remote_lane_substrate_d4_wyner_ziv_frame_0.sh
.venv/bin/python -m pytest src/tac/tests/test_d4_f1_f4_smoke_recipe.py -q
.venv/bin/python -m pytest src/tac/substrates/d4_wyner_ziv_frame_0/tests/test_d4_substrate.py -q
.venv/bin/python -m pytest src/tac/tests/test_run_modal_smoke_before_full.py -q
.venv/bin/python -m pytest src/tac/tests/test_operator_authorize_canonical_tool.py src/tac/tests/test_modal_train_lane_hardening.py -q
.venv/bin/python -m pytest src/tac/tests/test_modal_train_lane_hardening.py src/tac/tests/test_modal_training_claims.py -q
.venv/bin/python tools/check_dispatch_cli_shell_hazards.py --strict
git diff --check -- .omx/operator_authorize_recipes/substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch.yaml scripts/operator_authorize_substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch.sh scripts/remote_lane_substrate_d4_wyner_ziv_frame_0.sh src/tac/tests/test_d4_f1_f4_smoke_recipe.py
```

Results:

- `test_d4_f1_f4_smoke_recipe.py`: 3 passed.
- D4 substrate tests: 65 passed.
- `test_run_modal_smoke_before_full.py`: 24 passed.
- Operator/Modal focused tests: 22 passed and 14 passed.
- Dispatch shell hazard check: pass.
- Diff whitespace check: pass.

Dry-run command:

```bash
.venv/bin/python tools/run_modal_smoke_before_full.py --recipe substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch --smoke-epochs 50 --smoke-gpu T4 --smoke-timeout-hours 1.0 --smoke-only --dry-run
```

Dry-run resolved:

- `epoch_env_var=D4_WYNER_ZIV_FRAME_0_EPOCHS`
- `smoke_validation_contract=contest_cuda_auth_eval_v1`
- `smoke_epochs=50`
- `smoke_gpu=T4`
- `timeout_hours=1.0`
- `--smoke-only` would stop after smoke.

Operator-authorize dry-run:

```bash
D4_WYNER_ZIV_FRAME_0_EPOCHS=50 D4_WYNER_ZIV_FRAME_0_MAX_PAIRS=200 MODAL_GPU=T4 .venv/bin/python tools/operator_authorize.py --recipe substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch --yes --label-suffix __smoke__50ep --timeout-hours-override 1.0 --cost-band-epochs-override 50 --cost-band-gpu-override T4 --dry-run
```

Dry-run cost band: p50 `$0.07` weak posterior, full-run reference
`modal/T4 x 2000ep fallback p50 $10.00`.

## Launch Attempt

Command:

```bash
D4_WYNER_ZIV_FRAME_0_SMOKE_ONLY=1 D4_WYNER_ZIV_FRAME_0_SMOKE_EPOCHS=50 D4_WYNER_ZIV_FRAME_0_MAX_PAIRS=200 MODAL_GPU=T4 OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1 OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1 scripts/operator_authorize_substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch.sh
```

Terminal claim rows:

- `substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch_20260514T161919Z__smoke__50ep`
  -> `failed_dispatch_rc_1`
- `substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch_20260514T161957Z__smoke__50ep`
  -> `failed_dispatch_rc_1`

No Modal call id was produced. No remote worker ran. No score artifact exists.

Failure class:

```text
Modal local upload/build race: a mounted file was modified during build.
Observed stderr: experiments/train_substrate_sabor_boundary_only_renderer.py
was modified during build process.
```

Follow-up canonical stability check:

```bash
.venv/bin/python -c "from pathlib import Path; from tac.deploy.modal.mount_manifest import REPO_ROOT, _import_trainer_module, collect_tier_required_input_files, collect_extra_mount_paths, _collect_paths_for_stability_check, verify_mount_set_mtime_stability; trainer=_import_trainer_module('experiments/train_substrate_d4_wyner_ziv_frame_0.py'); paths=_collect_paths_for_stability_check(root=REPO_ROOT, skip_structural=False, extra_dirs=(), extra_files=(), optional_dirs=(), optional_files=(), trainer_required_files=[p for _f,p in collect_tier_required_input_files(trainer)], trainer_extra_paths=collect_extra_mount_paths(trainer)); print(f'checking {len(paths)} modal mount roots for 10s stability'); verify_mount_set_mtime_stability(paths, window_seconds=10.0, max_retries=1); print('modal mount set stable for 10s')"
```

Result:

```text
tac.deploy.modal.mount_manifest.MountUploadRaceError: Modal mount set
(mtime, size) fingerprint is unstable after 1 retries (window=10.0s each).
```

Recent mounted-file writers were still active in `src/`, `experiments/`, and
`tools/`, so forcing `TAC_MODAL_MTIME_STABILITY_DISABLED=1` would risk a torn
Modal upload. Dispatch remains blocked pending mount-set quiescence.

Final live check at `2026-05-14T16:25:14Z`:

- `tools/claim_lane_dispatch.py summary --live-only --format json` reported no
  active D4 claim for `lane_d4_wyner_ziv_frame_0_substrate_20260514`.
- The same 10-second D4 Modal mount-set stability check still failed with
  `MountUploadRaceError`; dispatch remains refused until the mounted source set
  is stable.
- A sibling D4 attempt
  `substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch_20260514T162219Z__smoke__50ep`
  also closed terminally as `failed_dispatch_rc_1` plus
  `refused_dispatch_sister_subagent_mount_set_mtime_instability_attempt_3`.

## Reactivation Criteria

Re-run only after:

1. `tools/claim_lane_dispatch.py summary --live-only --format json` shows no
   active D4 claim.
2. The canonical D4 Modal mount-set stability check passes for at least 10
   seconds.
3. The same smoke command is re-run with `--smoke-only`, preserving
   `[contest-CUDA]` axis labeling and `score_claim=false` until auth eval
   produces a component-coherent CUDA result.
