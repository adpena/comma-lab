# Modal Dispatcher Trainer Metadata Gap Audit - 2026-05-17

Status: read-only subagent findings preserved; no code changes from these audits yet.

Context: after the L5 v2 TT5L materialized output-identity hardening push, three read-only subagents returned Modal dispatcher metadata and mount-manifest findings. This ledger preserves the signal so it is not stranded in chat or subagent state.

## Preserved Findings

1. `experiments/modal_train_lane.py` now resolves trainer metadata for the canonical substrate path, but `_collect_trainer_extra_mount_payload()` can fail open on trainer import errors. For substrate-named scripts, that can silently recreate missing-anchor or missing-extra-mount failures before `fn.spawn()`.

2. `_derive_trainer_module_path()` returns `None` for missing or misnamed substrate trainers, and `modal_train_lane.py` treats that as legacy self-bootstrap. For substrate-named lanes, naming convention should imply trainer metadata is authoritative and missing trainer resolution should fail closed unless explicitly waived.

3. Generic recipe authorization in `tools/operator_authorize.py` builds Modal training commands without an explicit trainer module path. Non-substrate lane scripts with trainer metadata can therefore skip metadata-derived required inputs and extra mounts unless they happen to follow the canonical `remote_lane_substrate_*` naming path.

4. `modal_metadata.json` records the dispatch metadata but does not yet record the resolved trainer path, metadata source, required-input list, or per-file bytes/SHA-256 for staged extra mounts. That weakens post-hoc custody and reproducibility for trainer metadata closure.

5. `experiments/modal_t1_balle_endtoend.py` uses a static Modal mount manifest rather than trainer metadata discovery. Current coverage appears manually sufficient for its known `--pr95-parity-profile` required input, but the dispatcher does not structurally consume future `TIER_1_EXTRA_MOUNT_PATHS`, `MODAL_EXTRA_MOUNT_PATHS`, or new `required_input_file` declarations.

6. Active legacy Modal deploy helpers under `src/tac/deploy/modal/archive/` and `src/tac/deploy/modal/modal_asymmetric_warp_deploy.py` still carry manual mount logic outside the current `check_modal_dispatcher_uses_canonical_mount_builder` scan surface. Auth-eval, recovery, harvest, diagnostic, and other non-trainer Modal paths were considered N/A by the read-only audits.

7. A strict sibling guard is warranted: `check_modal_dispatcher_consumes_trainer_module_path`. Acceptance criterion: any Modal dispatcher that invokes an `experiments/train_*.py` trainer, or a lane script that invokes one, must pass `trainer_module_path=...`, use the WAVE-3 payload-staging pattern, or prove a static manifest covers `collect_tier_required_input_files()` and `collect_extra_mount_paths()` for that trainer.

## Recommended Next Actions

1. Add `--trainer-module-path` to `experiments/modal_train_lane.py` and thread `modal.cost_band_trainer` or the relevant recipe trainer field from `tools/operator_authorize.py`. Fail closed when explicit and derived trainer paths conflict.

2. Make substrate-named trainer metadata import and path resolution fail closed unless a documented waiver is present.

3. Extend `modal_metadata.json` with `trainer_module_path_resolved`, metadata source, required inputs, and `{rel_path, bytes, sha256}` for each staged extra mount.

4. Add focused regression tests for non-substrate recipe trainer metadata, import-failure fail-closed behavior, staged extra-mount manifest hashing, and static-manifest exceptions.

5. Land the sibling strict preflight gate only after the intended N/A surfaces and static-manifest exceptions are documented, so the gate blocks trainer-dispatch false authority without dragging diagnostic/recovery Modal tools into trainer-only policy.

## Subagent Sources

- `019e33fc-7dc4-7260-bd78-1cb9d67bd8c6`: fail-open and path-resolution review of `modal_train_lane.py`, plus review-gate/preflight observations.
- `019e340c-c5a5-78d1-9331-541f191fe38f`: static-manifest and canonical mount-builder review, with proposed strict gate acceptance criteria.
- `019e340d-a1bd-7a11-8126-b813e29c7780`: generic Modal training authorization and metadata-custody review across `tools/operator_authorize.py`, `experiments/modal_train_lane.py`, and `src/tac/deploy/modal`.
