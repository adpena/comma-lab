# STC v2 driver path-layer fix + Catalog #152 Wave 2 driver-path-expectation extension landed 2026-05-16

**Lane**: `lane_stc_v2_driver_fix_catalog_152_driver_path_extension_20260516`
**Operator-approved**: Option 1 (Recommended) per operator dispatch directive 2026-05-16
**Empirical anchor**: STC v2 Modal T4 dispatch `fc-01KRSVKF9VEESQY2FS33FF4WDM` rc=25 / 1.56s @ 2026-05-17T02:17:51Z
**Failure log**: `FATAL: Lane A anchor archive missing at /tmp/pact/experiments/results/lane_a_landed/archive_lane_a.zip`

## Root cause (premise verification per Catalog #229)

Wave 1's trainer-side fix (`experiments/train_substrate_stc_v2.py::TIER_1_EXTRA_MOUNT_PATHS`) at line 111 is **structurally INERT** for generic Modal dispatchers because `experiments/modal_train_lane.py:154` passes `trainer_module_path=None` to `build_training_image`:

```python
training_image = build_training_image(
    image.env({...}),
    trainer_module_path=None,  # <-- never reads trainer's TIER_1_EXTRA_MOUNT_PATHS
    optional_files=(...),
)
```

The canonical mount manifest in `src/tac/deploy/modal/mount_manifest.py::build_training_image` only consumes `TIER_1_EXTRA_MOUNT_PATHS` when `trainer_module_path` is non-None. The generic Modal dispatcher cannot name a single trainer module because it dispatches arbitrary `scripts/remote_lane_*.sh`. So the Wave 1 trainer-side declaration never propagates to the Modal image mount set.

The driver script (`scripts/remote_lane_substrate_stc_v2.sh`) computed the anchor path as `$WORKSPACE/experiments/results/lane_a_landed/archive_lane_a.zip` which Modal env-injects as `/tmp/pact/...`. The file was simply NOT present at that path on the Modal worker, the Stage 1b validation fired `exit 25` defense-in-depth.

## Fix landed (3 layers per "Bugs must be permanently fixed AND self-protected against")

### Layer 1: Driver script defensive resolution (the immediate fix)
`scripts/remote_lane_substrate_stc_v2.sh`:
- Added canonical `resolve_required_input_modal_aware` helper that probes `$WORKSPACE` → `/workspace/pact` → `/tmp/pact` candidate roots
- Replaced Stage 1b's single-path check with multi-candidate probe + diagnostic FATAL message on miss
- Updates `STC_V2_ANCHOR_ARCHIVE` to the resolved path so downstream Stage 4 (trainer subprocess) sees the actual location

### Layer 2: Sister driver fixes (audit + extincted across canvas)
`scripts/remote_lane_substrate_a1_plus_lapose.sh` + `scripts/remote_lane_substrate_a1_plus_wavelet_residual.sh`:
- Both consume `experiments/results/track4_sg_a1_t178000_20260509/submission_dir/archive.zip` (A1 archive — same Modal-IGNORED-subtree bug class)
- Added same canonical `resolve_required_input_modal_aware` helper
- Both syntax-verified (`bash -n` PASS)

### Layer 3: STRICT preflight Catalog #152 driver-path-expectation extension (the structural protection)
`src/tac/preflight.py`:
- Extended `check_operator_wrapper_validates_required_input_files_pre_dispatch` with a NEW sub-scan loop that iterates every Modal recipe (`platform: modal`) with a `lane_script: scripts/remote_lane_substrate_*.sh` field
- For each referenced driver that consumes a Modal-IGNORED `experiments/results/**` required-input file, requires ONE of:
  - (a) Canonical helper invocation (`resolve_required_input_modal_aware` / `_modal_workspace_env` / `resolve_modal_path`)
  - (b) Explicit probe of `/workspace/pact/` AND `/tmp/pact/` in proximity
  - (c) `MODAL_RUNTIME`-conditional branching
  - (d) Same-line `# DRIVER_PATH_MODAL_AWARE_OK:<rationale>` waiver (or sister `REQUIRED_INPUT_MODAL_STAGED_OK`)
- Added 5 new helper functions:
  - `_check_152_driver_has_defensive_path_resolution`
  - `_check_152_driver_has_path_waiver`
  - `_check_152_driver_references_required_input`
- Added constants `_CHECK_152_DRIVER_DEFENSIVE_PATTERNS`, `_CHECK_152_DRIVER_WAIVER_TOKENS`, `_CHECK_152_DRIVER_PATH_EXEMPT_DRIVERS`
- Violation message names the recipe + cites bug-class anchor (`fc-01KRSVKF9VEESQY2FS33FF4WDM`)

### Layer 4: CLAUDE.md Catalog #152 description Wave 2 extension
Appended Wave 2 paragraph documenting:
- Empirical anchor (call_id, timestamp, exact failure message)
- Root cause (modal_train_lane.py:154 passes trainer_module_path=None)
- The 4-option acceptance cascade
- STC v2 + a1_plus_lapose + a1_plus_wavelet_residual sister-fix landings in same commit batch
- STRICT-from-byte-one per "Strict-flip atomicity rule"

## Test coverage

`src/tac/tests/test_check_152_modal_mounted_input_extension.py` — 25 new Wave 2 tests added:
- Helper unit tests: `test_driver_has_defensive_path_resolution_*` (4 tests covering canonical helper / explicit probe / MODAL_RUNTIME conditional / negative)
- Waiver semantics: `test_driver_has_path_waiver_*` (3 tests covering rationale accepted / placeholder rejected / sister REQUIRED_INPUT_MODAL_STAGED_OK marker)
- Reference detection: `test_driver_references_required_input_*` (4 tests covering full path / basename + context / negative / empty path)
- Live-repo regression: STC v2 + a1_plus_lapose + a1_plus_wavelet_residual drivers carry defensive resolver (3 tests)
- End-to-end: zero violations live-repo regression (1 test)
- Synthetic end-to-end: undefensive flagged / defensive passes / waiver accepted / placeholder rejected / non-substrate exempt / non-Modal exempt / no-input exempt / strict-mode raises (8 tests)
- Constant pins: defensive-pattern set has canonical helper + MODAL_RUNTIME (1 test); waiver-marker tuple has both tokens (1 test)

Total: 25 new Wave 2 tests + 29 pre-existing Wave 1 tests = **54/54 tests pass**.

Main Catalog #152 test file (`test_check_152_operator_wrapper_validates_required_inputs.py`): **58/58 tests pass** — no regression.

## Live count + sister-gate cleanliness

- Catalog #152 (extended): **0 violations** post-fix
- Catalog #152 driver-path-expectation sub-check: **0 violations** post-fix
- Catalog #118 (no duplicate numbers): **0 violations**
- Catalog #159 (catalog text matches preflight strict value): **0 violations**
- Catalog #176 (strict callsites have CLAUDE.md row): **0 violations**
- Catalog #185 (LIVE_COUNT drift detection): **0 violations**

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: N/A — this is a structural preflight extension, not a substrate sensitivity primitive
2. **Pareto constraint**: ACTIVE — extending Catalog #152 strengthens the dispatch feasibility constraint; future Modal dispatches with non-defensive drivers structurally refused
3. **Bit-allocator hook**: N/A — no per-tensor importance change
4. **Cathedral autopilot dispatch hook**: ACTIVE — the operator-authorize dispatch path consults Catalog #152 (`tools/local_pre_deploy_check.py` per Catalog #243); future Wave 2 violations block dispatch
5. **Continual-learning posterior update**: N/A — no empirical anchor produced (this is gate hardening; the next STC v2 Modal smoke will produce the next anchor)
6. **Probe-disambiguator**: N/A — no 2+ defensible interpretations (the fix is mechanical)

## Cross-references

- Sister of Catalog #163 (sentinel-when-sourcing-bootstrap)
- Sister of Catalog #244 (canonical Modal NVML env block)
- Sister of Catalog #166 (Modal HEAD-parity ledger)
- Sister of Catalog #201 (sentinel files under Modal mount set)
- Sister of Catalog #153 (canonical Modal mount builder — the upstream that Wave 1 trainer-side fix relied on, but which `modal_train_lane.py` bypasses)
- Sister Wave 1 extension: bug-class anchor `fc-01KRSB76H04HM4958V2HX2JZZ4` (2026-05-16 first occurrence)
- Empirical receipts: failed dispatch metadata at `experiments/results/lane_substrate_stc_v2_modal_t4_dispatch_20260517T021720Z_modal/modal_metadata.json`
- Bug-class anchor `fc-01KRSVKF9VEESQY2FS33FF4WDM` second occurrence at 2026-05-17T02:17:51Z (this lane's anchor)

## Sister-subagent coordination notes (per Catalog #230)

PROBE-OUTCOMES bake-in (sister `ad4c862bc9f1aad12`) owns NEW gate function at next available catalog # + CLAUDE.md Catalog #292 amendment. NO file collision observed during work: this lane edited Catalog #152's existing function body + appended to its CLAUDE.md description block; PROBE-OUTCOMES edits a different gate function + different CLAUDE.md section.

The canonical serializer with `--expected-content-sha256` (POST-EDIT working-tree shas) will arbitrate any concurrent commit-time collision.

## Follow-on op-routables

1. **Re-dispatch STC v2 Modal smoke** to verify the driver fix resolves the failure mode. Use the same recipe (`substrate_stc_v2_modal_t4_dispatch.yaml`) and observe whether the file is found at one of the 3 candidate roots OR if the failure recurs with the more diagnostic message (indicating the trainer-side mount isn't reaching ANY of the 3 paths — meaning the mount is the failure, not the path expectation).

2. **Root-cause why mount manifest's `add_local_file` for `experiments/results/lane_a_landed/archive_lane_a.zip` may not be materializing on the Modal worker**, even when `trainer_module_path=None` is passed. The Wave 1 trainer fix added the path to `TIER_1_EXTRA_MOUNT_PATHS` but the dispatcher ignores it — confirm whether the operator-authorize recipe (`required_input_files` field) is somehow staging it OR whether the file is simply never mounted by the generic dispatcher.

3. **Consider auditing other drivers for the same bug class**: any `scripts/remote_lane_substrate_*.sh` that references a `default_path` under `experiments/results/**` AND dispatches to Modal will now be caught by Catalog #152's Wave 2 extension. Currently STC v2 + a1_plus_lapose + a1_plus_wavelet_residual are the 3 known violators; live count = 0 after backfill.

4. **Consider a parallel Wave 3 extension to `experiments/modal_train_lane.py` itself**: refactor the canonical dispatcher to accept a per-dispatch trainer_module_path so trainer-side TIER_1_EXTRA_MOUNT_PATHS declarations are actually consumed. This is the deeper structural fix; the current Wave 2 driver-side defensive resolver is the operator-prescribed pragmatic patch.

## Operator directive citations

- "Mirror `scripts/remote_archive_only_eval.sh` if it uses a similar pattern" — DONE (the helper structure mirrors the canonical bootstrap source-sentinel pattern)
- "the trainer-side fix is correct but the driver-side fix wasn't done — the gate caught the wrong layer" — DONE (driver-side defensive resolver + Catalog #152 driver-path-expectation extension)
- "audit + fix if found" — DONE (2 sister drivers found via the new gate's first run + fixed in same commit batch)
- "DO NOT claim a new catalog # — this is an EXTENSION of Catalog #152" — HONORED
- "live count for extended Catalog #152 post-landing (must be 0)" — VERIFIED 0

---

Lane: `lane_stc_v2_driver_fix_catalog_152_driver_path_extension_20260516`
Subagent: `stc_v2_driver_fix_20260516`
Cost: $0 (CPU smoke + bash -n syntax check + pytest)
Time: ~50 minutes
Commit: pending via canonical serializer
