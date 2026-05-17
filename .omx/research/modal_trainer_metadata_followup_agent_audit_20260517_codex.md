# Modal Trainer Metadata Follow-Up Agent Audit - 2026-05-17

## Purpose

Preserve the three read-only follow-up agent reports returned after the WAVE-3
`modal_train_lane.py` trainer metadata landing. This memo is a no-signal-loss
ledger only: the agents did not edit files, and no dispatch or score claim was
made.

## Repository State At Preservation

- Branch: `main`
- Remote: `origin` at `git@github.com:adpena/comma-lab.git`
- Local/remote head before this ledger: `0726ca04dd99dcc47db040727ad5685047ac48d7`
- Worktree state before this ledger: clean, no staged changes, no untracked
  files reported by `git ls-files --others --exclude-standard`.

## Agent Reports Preserved

### Huygens the 3rd

Primary finding: `experiments/modal_train_lane.py` still has two structural
fail-open risks around trainer metadata discovery:

- `_collect_trainer_extra_mount_payload()` can return an empty payload after
  trainer import failure. For substrate-named scripts this should fail closed
  unless explicitly waived.
- `_derive_trainer_module_path()` can return `None` for a substrate-named lane
  with a missing or misnamed trainer, and `main()` can treat that as legacy
  self-bootstrap. The lane naming convention implies trainer-side mount
  contracts should be authoritative.

Additional findings:

- `src/tac/preflight.py` has stale or confusing live-count commentary around
  the modal trainer metadata guard surface.
- Catalog #314 checkpoint scanning appears to call some complete/blocked rows
  "in-flight" and may scan all rows rather than latest-per-subagent.
- `# NO_SERIALIZER_OK:` may be too broad for absorption detection; a distinct
  `# ABSORPTION_PATTERN_OK:` waiver would better match that bug class.

Reported verification: `git diff --check`, AST parse, and focused pytest for
Catalog #314 plus the Modal WAVE-3 test passed with `62 passed`.

### Nash the 3rd

Primary finding: `experiments/modal_t1_balle_endtoend.py` dispatches a
trainer/lane path through a static Modal mount manifest rather than structural
trainer metadata discovery.

Current disposition:

- Not a current hard failure: the required PR95 parity profile is manually
  mounted, and the lane script fails if it is absent.
- Still a latent structural gap: future `TIER_1_EXTRA_MOUNT_PATHS`,
  `MODAL_EXTRA_MOUNT_PATHS`, or new `required_input_file` declarations for
  that trainer would not be consumed automatically.

Additional finding:

- `experiments/modal_component_sensitivity_shards.py` still trips the strict
  manual-mount scanner because waiver comments are on closing lines rather than
  the same `.add_local_*` call lines. This is a hygiene/preflight issue, not
  the WAVE-3 metadata class.

Recommended strict gate:

`check_modal_dispatcher_consumes_trainer_module_path`

Acceptance criterion: any Modal dispatcher that invokes an
`experiments/train_*.py` trainer, or a lane script that invokes one, must either
pass `trainer_module_path`, use the WAVE-3 payload-staging pattern, or prove a
static manifest covers `collect_tier_required_input_files()` and
`collect_extra_mount_paths()` for that trainer.

### Turing the 3rd

Primary finding: generic Modal training still needs better explicit trainer
metadata flow and mount-manifest custody in dispatch metadata.

Specific gaps:

- `tools/operator_authorize.py` builds Modal training commands with
  `--lane-script`, `--lane-id`, `--label`, GPU, timeout, env overrides, and
  `--cost-band-trainer`, but no explicit trainer module path.
- `experiments/modal_train_lane.py` derives trainer path only from
  `scripts/remote_lane_substrate_<id>.sh`, and non-substrate lane scripts can
  route through legacy self-bootstrap.
- `modal_metadata.json` records dispatch metadata but not the resolved trainer
  path, payload file list, bytes, or SHA-256s.
- Active legacy Modal deploy helpers under `src/tac/deploy/modal/` still use
  ad hoc `add_local_dir` or `add_local_file` mount logic outside the current
  scanner surface.

No-issue findings:

- Canonical substrate dispatch does consume trainer metadata through
  `experiments/train_substrate_<id>.py` discovery, `TIER_1_EXTRA_MOUNT_PATHS`,
  `MODAL_EXTRA_MOUNT_PATHS`, required input files, and worker-side payload
  materialization.
- `src/tac/deploy/modal/mount_manifest.py` remains the expected canonical
  helper; its internal `add_local_*` use is not itself a violation.

Highest-EV fixes recommended:

1. Add an explicit `--trainer-module-path` path to `experiments/modal_train_lane.py`,
   pass the recipe trainer metadata from `tools/operator_authorize.py`, and fail
   closed if explicit and derived trainer paths conflict.
2. Extend `modal_metadata.json` with `trainer_module_path_resolved`, metadata
   source, required input files, and `{rel_path, bytes, sha256}` for every
   staged extra mount.
3. Extend Catalog #153 or add a sibling guard scanning
   `src/tac/deploy/modal/**/*.py` for active `@app.function` launchers using
   manual mounts without the canonical builder, a retired stub, or a documented
   waiver.
4. Add regressions for non-substrate recipe trainer metadata, staged extra-mount
   manifest hashing, and legacy helper manual-mount detection.

## Consolidated Follow-Up Queue

1. Land `check_modal_dispatcher_consumes_trainer_module_path` as a strict or
   warn-to-strict preflight gate with fixtures for WAVE-3 payload staging,
   explicit `trainer_module_path`, and verified static manifests.
2. Make substrate-named Modal lane scripts fail closed when trainer import or
   trainer path derivation fails, unless a reviewable waiver is present.
3. Thread explicit recipe trainer metadata from `tools/operator_authorize.py` to
   `experiments/modal_train_lane.py` and reject conflicts with the derived
   trainer path.
4. Add trainer payload custody fields to `modal_metadata.json`.
5. Audit legacy Modal deploy helpers under `src/tac/deploy/modal/` for manual
   mount usage and either migrate, retire, or waive them in the scanner.
6. Tighten Catalog #314 latest-row semantics and split serializer-intent waivers
   from absorption-pattern waivers.

## Current Frontier Impact

No score claim, no provider dispatch, and no lane promotion. This memo preserves
engineering rigor findings that affect future L5/L5 v2 and non-HNeRV frontier
dispatch reliability because missing trainer metadata or incomplete Modal mount
custody can produce false negatives, silent no-op payloads, or stale-runtime
dispatches.
