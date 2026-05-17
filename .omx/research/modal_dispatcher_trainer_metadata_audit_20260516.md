# Modal dispatcher trainer metadata audit and follow-up hardening 2026-05-16

**Lane**: `lane_modal_dispatcher_trainer_metadata_audit_20260516`
**Scope**: read-only audit plus targeted fixes for Modal dispatchers that can
train or launch trainer-bearing lanes.
**Cost**: $0 local CPU only.

## Trigger

WAVE-3 fixed `experiments/modal_train_lane.py` for canonical
`scripts/remote_lane_substrate_<id>.sh` dispatches by deriving
`experiments/train_substrate_<id>.py` and staging trainer-declared payload
bytes. Follow-up audits asked whether the same metadata-consumption discipline
covered other Modal dispatch paths and whether the staged payload was preserved
in dispatch custody.

## Findings

1. `experiments/modal_train_lane.py` consumed trainer metadata for canonical
   substrate scripts, but operator recipes can also declare trainer metadata
   through `required_input_files_trainer` or `modal.cost_band_trainer`. Before
   this patch, `tools/operator_authorize.py` did not pass that explicit trainer
   module to the generic Modal launcher, so non-`remote_lane_substrate_*` lane
   scripts could silently rely on self-bootstrap and skip trainer extra mounts.

2. `experiments/modal_train_lane.py` staged trainer payload bytes but did not
   record a mount-closure manifest in `modal_metadata.json`. That made post-hoc
   custody weaker than the actual worker payload.

3. The WAVE-3 helper still warned and skipped missing trainer-declared paths in
   helper mode. For real substrate dispatch this should fail before provider
   spend. The dispatch path now passes `fail_on_missing_paths=True`.

4. `experiments/modal_component_sensitivity_shards.py` had same-line waiver
   comments on the closing lines of four `add_local_*` calls. Catalog #153
   requires the waiver on the call line. This was a pre-existing hygiene
   failure, not a trainer metadata bug.

5. `experiments/modal_t1_balle_endtoend.py` intentionally uses a static Modal
   mount manifest rather than `build_training_image`. Current coverage was OK
   by inspection, but future trainer metadata drift was latent. A pure
   source-level verifier now checks that the static manifest covers every
   statically-resolved trainer `required_input_file=True`,
   `TIER_1_EXTRA_MOUNT_PATHS`, and `MODAL_EXTRA_MOUNT_PATHS` path.

6. Archived Modal deploy helpers under `src/tac/deploy/modal/archive/` and the
   older asymmetric-warp deploy file still use ad hoc mounts. They are outside
   this targeted patch because the active generic launcher and T1 static
   manifest are now covered; a sibling guard can scan deploy-layer launchers
   separately.

## Fixes landed

- Added `--trainer-module-path` support to `experiments/modal_train_lane.py`.
- `tools/operator_authorize.py` now passes `required_input_files_trainer` or
  `modal.cost_band_trainer` into the generic Modal launcher.
- `modal_train_lane.py` fails closed if an explicit trainer module is missing,
  or if it conflicts with the substrate-derived trainer module.
- `modal_train_lane.py` dispatch now fails closed on missing
  `required_input_file=True` defaults or missing extra-mount paths.
- `modal_metadata.json` now records:
  - `trainer_module_path_resolved`
  - `trainer_metadata_source`
  - `trainer_extra_mount_payload_file_count`
  - `trainer_extra_mount_payload_total_bytes`
  - `trainer_extra_mount_payload_manifest` with `{rel_path, bytes, sha256}`
- Added `src/tac/deploy/modal/static_manifest.py` for static manifest coverage
  verification.
- `experiments/modal_t1_balle_endtoend.py` now asserts its static manifest
  covers T1 trainer metadata at import time.
- Moved `modal_component_sensitivity_shards.py` Catalog #153 waivers onto the
  same lines as their manual mount calls.

## Audit classification

| Surface | Classification | Notes |
|---|---|---|
| `experiments/modal_train_lane.py` | Fixed | Canonical substrate derivation plus explicit recipe trainer metadata now both feed payload staging. |
| `tools/operator_authorize.py` | Fixed | Recipe trainer metadata now flows to the Modal launcher. |
| `experiments/modal_t1_balle_endtoend.py` | Fixed/static-manifest verified | Static manifest retained; new verifier prevents trainer metadata drift. |
| `experiments/modal_component_sensitivity_shards.py` | Hygiene fixed | Not trainer-bearing; same-line waiver placement corrected. |
| Auth-eval / harvest / recovery / diagnostics | N/A | No trainer metadata to consume. |
| `src/tac/deploy/modal/archive/*.py` and legacy asymmetric warp deploy | Follow-up | Ad hoc deploy-layer mounts remain outside Catalog #153's `experiments/modal_*.py` scanner. |

## Verification

- `src/tac/tests/test_modal_train_lane_wave_3_trainer_module_path.py`
- `src/tac/tests/test_operator_authorize_scripts.py`
- `src/tac/tests/test_modal_static_manifest.py`
- `src/tac/tests/test_modal_t1_balle_endtoend.py`
- `src/tac/tests/test_modal_train_lane_hardening.py`
- `src/tac/tests/test_check_152_modal_mounted_input_extension.py`
- `src/tac/tests/test_check_153_modal_mount_builder.py`
- `src/tac/tests/test_check_166_modal_dispatch_head_parity.py`
- `py_compile` on touched Python files.
- `ruff --select F` on touched Python files.
- `check_modal_dispatcher_uses_canonical_mount_builder(strict=False)` -> 0.
- `check_operator_wrapper_validates_required_input_files_pre_dispatch(strict=False)` -> 0.

## Follow-up

Add a narrow deploy-layer sibling guard for active
`src/tac/deploy/modal/**/*.py` launchers that use manual mounts outside the
`experiments/modal_*.py` Catalog #153 scanner. The guard should distinguish
archived/retired deploy helpers from active `@app.function` launchers and allow
documented static-manifest coverage where appropriate.
