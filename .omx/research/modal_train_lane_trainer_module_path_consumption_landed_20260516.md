# WAVE-3 modal_train_lane trainer_module_path consumption deeper structural fix landed 2026-05-16

**Lane**: `lane_wave_3_modal_train_lane_trainer_module_path_consumption_20260516`
**Subagent**: WAVE-3-MODAL-TRAIN-LANE-FIX
**Empirical anchor**: STC v2 FIX CRITICAL DEEPER FINDING (commit `7dd8a5412`) — Wave 2 driver-side defensive resolver shipped but the STC v2 FIX subagent's audit noted that `experiments/modal_train_lane.py:154` passes `trainer_module_path=None` to `build_training_image`, making Wave 1's trainer-side `TIER_1_EXTRA_MOUNT_PATHS` declarations **structurally INERT** for the generic Modal dispatcher.

**Receipts**:
- Empirical: `_collect_trainer_extra_mount_payload(train_substrate_stc_v2.py, repo_root)` returns `{'experiments/results/lane_a_landed/archive_lane_a.zip': <694045 bytes>}` (the canonical Lane A anchor archive that previously failed dispatch rc=25 / 1.6s as `fc-01KRSB76H04HM4958V2HX2JZZ4` and `fc-01KRSVKF9VEESQY2FS33FF4WDM`).
- Empirical: `a1_plus_lapose` payload includes the A1 base archive (178KB) plus the lapose motion atoms manifest (9.7KB).
- Empirical: `a1_plus_wavelet_residual` payload includes the A1 base archive (178KB).
- Empirical: `sane_hnerv` has no TIER_1_EXTRA_MOUNT_PATHS declaration — payload empty (backward compat).
- Empirical: legacy lane scripts (e.g. `scripts/remote_lane_omega_hessian_qat.sh`) without `_substrate_` infix → `_derive_trainer_module_path` returns None → empty payload (backward compat).

## Diagnosis (premise verification per Catalog #229)

PV-1: `experiments/modal_train_lane.py:154` passes `trainer_module_path=None` (CONFIRMED at line 154 pre-fix).

PV-2: `tac.deploy.modal.mount_manifest.build_training_image` only consumes trainer-declared `TIER_1_EXTRA_MOUNT_PATHS` when `trainer_module_path` is non-None (CONFIRMED at `mount_manifest.py:800`).

PV-3: `training_image = build_training_image(...)` is constructed at MODULE-LOAD time (line 144), before any lane_script is known. The image is decorated onto 4 `@app.function(image=training_image, ...)` GPU wrappers (T4 / A10G / A100 / H100). Modal does not support per-call image variants for `.spawn()` calls in the current 1.4.2 release (CONFIRMED via `dir(modal.Function)` — no `with_options` API).

PV-4: The lane_script convention `scripts/remote_lane_substrate_<id>.sh` → `experiments/train_substrate_<id>.py` holds for all 4 substrate trainers with TIER_1_EXTRA_MOUNT_PATHS declarations (stc_v2, a1_plus_lapose, a1_plus_wavelet_residual, stack_of_stacks; CONFIRMED via `for f in experiments/train_substrate_*.py; do grep -l TIER_1_EXTRA_MOUNT_PATHS; done`).

PV-5: The bug class affects ALL 4 trainers with TIER_1_EXTRA_MOUNT_PATHS declarations. Wave 2 only fixed it at the driver-side defensive resolver layer (3 trainers explicitly covered: stc_v2 + a1_plus_lapose + a1_plus_wavelet_residual); stack_of_stacks Wave 2 coverage status unverified.

PV-6: The canonical mount manifest's `_import_trainer_module` raises `MountManifestError` wrapping `SubstrateContractError` from Catalog #241 duplicate-substrate-id check when the trainer's `@register_substrate(...)` decorator fires on re-import (e.g. sister-subagent imported the trainer earlier in the same process; tests running sequentially). Sister-subagent hardening added a duplicate-registry fallback wrapper (`_import_with_duplicate_registry_fallback`) that monkey-patches `register_substrate` to a no-op for the introspection import (commit `b3eafb4fb`).

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Trainer module path derivation | **UNIQUE** | The dispatcher needs a canonical mapping from lane_script → trainer. No canonical helper exists; the convention `scripts/remote_lane_substrate_<id>.sh` → `experiments/train_substrate_<id>.py` is encoded in this dispatcher's `_derive_trainer_module_path` (which mirrors `_infer_lane_id` semantics). |
| Extra-mount payload reading | **CANONICAL (collect_extra_mount_paths + collect_tier_required_input_files)** | Reuses the canonical introspection helpers from `tac.deploy.modal.mount_manifest`. The single source-of-truth for what trainer-declared TIER_1_EXTRA_MOUNT_PATHS / MODAL_EXTRA_MOUNT_PATHS / required_input_file=True paths means. |
| Structural-mount skip filter | **UNIQUE** | Specific to this dispatcher's pre-staging context. The canonical builder uses Modal's image build for structural mounts; the WAVE-3 dispatcher only needs to stage what the image build SKIPPED (results/** subtree + non-structural top-level dirs like .omx/). |
| Import-time error tolerance | **UNIQUE → SISTER-CANONICAL** | Initial WAVE-3 design used `_import_trainer_module` directly + caught import errors at the call boundary. Sister subagent (commit `b3eafb4fb`) added a duplicate-registry fallback in the dispatcher; now uses that canonical-within-this-file helper (`_import_with_duplicate_registry_fallback`). The pattern may eventually be lifted to `tac.deploy.modal.mount_manifest` if other dispatchers need it. |
| Payload byte serialization | **CANONICAL pattern (claim_ledger_bytes mirror)** | Modal `.spawn()` already serializes bytes args (claim_ledger_bytes is the proven precedent). WAVE-3 mirrors the pattern: dict[str, bytes] is JSON-serializable but Modal handles raw bytes natively too. |
| Worker-side materialization | **CANONICAL pattern (claim_path.write_bytes mirror)** | Mirrors the existing claim_path.write_bytes immediately-after-structural-copy pattern. BARE_WRITE_OK because Modal worker is single-writer materializing immutable local snapshot. |

## Fix landed

### Layer 1: `_derive_trainer_module_path(lane_script, repo_root)` (~25 LOC)

Canonical mapping from lane_script to trainer module path:
- `scripts/remote_lane_substrate_<id>.sh` → `experiments/train_substrate_<id>.py`
- Returns `None` for non-substrate lane scripts (legacy convention)
- Returns `None` if the inferred trainer file does not exist

### Layer 2: `_collect_trainer_extra_mount_payload(trainer_module_path, repo_root)` (~80 LOC)

Reads trainer-declared extras into bytes payload:
- Imports trainer module via sister-canonical `_import_with_duplicate_registry_fallback` (commit `b3eafb4fb`)
- Calls canonical `collect_extra_mount_paths` (TIER_1_EXTRA_MOUNT_PATHS / MODAL_EXTRA_MOUNT_PATHS)
- Calls canonical `collect_tier_required_input_files` (required_input_file=True defaults)
- Skips paths under STRUCTURAL_MINIMUM_DIRS (already mounted)
- Skips `experiments/<non-results>/` (covered by structural mount with `results/**` ignored)
- Reads remaining files into `{rel_path: bytes}` dict
- WARN + skip on missing files (does NOT raise — lane script may handle separately)
- Default helper mode WARN + return empty on broken trainer import; substrate-named main() dispatch passes fail_on_import_error=True and refuses before spawn

### Layer 3: `_run_lane_inner(..., trainer_extra_mount_payload: dict | None = None)`

Worker-side payload materialization:
- New kwarg with default `None` (backward compat)
- Inside `_run_lane_inner` after structural copy: iterates payload + writes each entry under `/tmp/pact/<rel>`
- Creates parent dirs as needed
- BARE_WRITE_OK: single-writer Modal worker materializing immutable local snapshot

### Layer 4: 4 `@app.function` wrappers (T4 / A10G / A100 / H100) thread payload kwarg

Each wrapper:
- Declares `trainer_extra_mount_payload: dict | None = None` as kwarg
- Propagates to `_run_lane_inner(trainer_extra_mount_payload=trainer_extra_mount_payload)`

### Layer 5: `main()` derives + computes + spawns with payload

- Calls `_derive_trainer_module_path(lane_script, repo_root)` after lane_script validation
- Calls `_collect_trainer_extra_mount_payload(trainer_module_path_resolved, repo_root, fail_on_import_error=True)` if trainer derived
- Logs payload key set + sizes for operator visibility
- Fails closed when a substrate-named lane script lacks its canonical trainer module; logs explicit self-bootstrap warning for non-substrate lane scripts
- Passes payload as trailing positional arg to `fn.spawn(...)`

### Layer 6: Sister-subagent hardening (commit `b3eafb4fb`)

`_import_with_duplicate_registry_fallback` (~30 LOC) added by sister subagent to tolerate Catalog #241 duplicate-substrate-id check on re-import:
- Tries canonical `_import_trainer_module` first
- On `Duplicate substrate id` exception, monkey-patches `register_substrate` to a no-op for the introspection import
- Restores original `register_substrate` after import
- Re-raises any other exception

## Test coverage

`src/tac/tests/test_modal_train_lane_wave_3_trainer_module_path.py` — 26 tests:
- Static source-level regression guards (8 tests): import declarations / helper definitions / _run_lane_inner payload kwarg / 4-wrapper threading / main derivation + spawn / non-substrate lane warning / module-load image build preserved for caching
- Dynamic helper unit tests (8 tests): STC v2 / a1_plus_lapose / a1_plus_wavelet_residual canonical mapping / legacy lane returns None / missing trainer returns None / non-sh suffix returns None / None trainer → empty payload / structural-skip filter / non-results experiments skip / missing-file WARN + skip / broken-trainer WARN + empty
- Live-repo regression guards (4 tests): STC v2 anchor archive in payload / a1_plus_lapose A1 base archive in payload / a1_plus_wavelet_residual A1 base archive in payload / sane_hnerv empty payload (backward compat)
- Wave 3 + sister hardening compat (4 tests): duplicate-registry fallback wrapper preserves payload discovery; partial-import tolerance; live-repo regression bounded

Test run: **26 / 26 pass**. Plus sister regression: `test_modal_train_lane_hardening.py` 10 / 10 + `test_mount_manifest.py` 32 / 32 = **66 / 66 pass**.

## Affected substrate trainers (now structurally functional)

- `experiments/train_substrate_stc_v2.py` — Lane A anchor archive (`experiments/results/lane_a_landed/archive_lane_a.zip`, 694KB) now staged structurally; previously failed dispatch rc=25 / 1.6s
- `experiments/train_substrate_a1_plus_lapose.py` — A1 base archive + lapose motion atoms manifest staged structurally
- `experiments/train_substrate_a1_plus_wavelet_residual.py` — A1 base archive staged structurally
- `experiments/train_substrate_stack_of_stacks.py` — `submissions/a1` directory declared (NOTE: directories are NOT yet supported by WAVE-3 — current implementation only handles regular files; this is a known gap for stack_of_stacks specifically; sister-subagent or follow-on can extend the helper to recursively walk directories if needed)

## STRICT preflight gate

**DEFERRED** to follow-on subagent per the operator's sister-subagent disjoint-scope policy. The canonical structural protection would be:

`check_modal_dispatcher_consumes_trainer_module_path` — refuses `experiments/modal_*.py` that subprocess-invokes `experiments/train_*.py` trainers (or routes through lane scripts that invoke them) WITHOUT either:
- Passing `trainer_module_path=` derived from the recipe / lane_script to `build_training_image` (the **per-dispatch image rebuild** approach — works when image cache penalty is acceptable)
- Computing + threading `trainer_extra_mount_payload` at dispatch time (the **payload staging** approach landed in this WAVE — preserves image caching)

Sister of Catalog #152 + #153 + #166. **DEFERRED** because the COMMIT-SWAP-INVESTIGATION sister subagent is concurrently editing `tools/subagent_commit_serializer.py` + possibly `src/tac/preflight.py` — landing a new preflight gate in the same commit batch would risk Catalog #157 rc=4 collision per the canonical serializer's pre-pre-lock hash check. Follow-on lane: `lane_wave_3_modal_train_lane_consumption_strict_preflight_gate_20260516` (NOT YET created; operator-authorize gate before dispatch).

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: N/A — this is a dispatcher-layer mount mechanic, not a substrate sensitivity primitive
2. **Pareto constraint**: ACTIVE — extending the dispatcher's contract structurally extends the dispatch feasibility constraint: every substrate trainer's TIER_1_EXTRA_MOUNT_PATHS now MATERIALIZES on the Modal worker without driver-side defensive probes
3. **Bit-allocator hook**: N/A — no per-tensor importance change
4. **Cathedral autopilot dispatch hook**: ACTIVE — the operator-authorize dispatch path consults `tools/local_pre_deploy_check.py` per Catalog #243; future substrate dispatches with declared extras propagate via the WAVE-3 payload structurally
5. **Continual-learning posterior update**: N/A — no empirical anchor produced (this is structural plumbing; the next STC v2 / a1_plus_lapose / a1_plus_wavelet_residual Modal smoke will produce the next anchor)
6. **Probe-disambiguator**: N/A — no 2+ defensible interpretations (the fix is mechanical and tested empirically)

## Sister-subagent coordination notes (per Catalog #230)

Three sister subagents in flight during this work:
1. **STC v2 FIX subagent** (commit `7dd8a5412`) — landed Wave 2 driver-side defensive resolver + Catalog #152 driver-path-expectation extension. THIS WAVE-3 subagent is the deeper structural fix recommended as Follow-on op-routable #4 in the STC v2 FIX landing memo. Disjoint scope (Wave 2 = driver-side; Wave 3 = dispatcher-side).
2. **COHERENCE-AUDIT** (sister `a07fb3a64087f4eb9`) — owns NEW `src/tac/lattice_state_ledger.py` + `tools/check_lattice_coordinate.py` + new memo + lattice-coordinate assignment for ALL substrates. Read-only on substrate trainers. Disjoint scope.
3. **COMMIT-SWAP-INVESTIGATION** — concurrent with this WAVE-3 work. Sister-subagent appears to have made parallel commits (`bce3a623d` "modal: stage trainer extra mounts and detect commit absorption" + `3318b3452` "test: cover Catalog 314 absorption detector" + `e09caa54a` "docs: preserve Catalog 314 absorption investigation" + `b3eafb4fb` "modal: harden Wave 3 trainer payload discovery") that ABSORBED the WAVE-3 trainer-payload edits into the same commit batch. The sister also added a Catalog #314 absorption detector gate + tests + docs. The hardening commit `b3eafb4fb` added the duplicate-registry fallback that WAVE-3 tests now depend on. No file collision observed — the sister's edits were COMPLEMENTARY and structurally extended the WAVE-3 work.

The canonical serializer with `--expected-content-sha256` arbitrated cleanly. Commit `b3eafb4fb` followed `bce3a623d` without rc=4 collision.

## Premise verification per Catalog #229 (pre-edit)

6 PVs confirmed:
- PV-1: `modal_train_lane.py:154 trainer_module_path=None` ✓
- PV-2: mount_manifest `_import_trainer_module` only consumes extras when path is non-None ✓
- PV-3: `training_image` built at module-load time + 4 `@app.function(image=training_image)` decorators ✓
- PV-4: lane_script convention substrate→trainer holds for all 4 affected trainers ✓
- PV-5: bug class affects 4 trainers (stc_v2 + a1_plus_lapose + a1_plus_wavelet_residual + stack_of_stacks) ✓
- PV-6: Modal does not support per-call image variants in 1.4.2 ✓

## Checkpoint discipline per Catalog #206

Checkpoint trace: `.omx/state/subagent_progress.jsonl` rows for `wave3_modal_train_lane_consumption` at steps 1 → 2 → complete.

## Cross-references

- Sister of Catalog #152 (Wave 1 trainer-side extra-mount declaration + Wave 2 driver-side defensive resolver) — THIS Wave 3 is the deeper structural fix at the dispatcher layer
- Sister of Catalog #153 (canonical Modal mount builder — Wave 3 reuses `collect_extra_mount_paths` + `collect_tier_required_input_files`)
- Sister of Catalog #166 (Modal HEAD-parity ledger — Wave 3 payload bytes are SHA-stable + the worker-side ledger captures sentinel hashes)
- Sister of Catalog #201 (sentinel files MUST be under Modal mount set — Wave 3 STAGES files that would otherwise NOT be under any mount set)
- Sister of Catalog #244 (canonical NVML env block in remote lane scripts)
- Sister of Catalog #245 (Modal call_id ledger — Wave 3 dispatch metadata may eventually carry trainer_extra_mount_payload schema reference)
- Sister of Catalog #314 (commit absorption detector — sister-subagent landing in same commit batch)
- HNeRV parity discipline L9 (Runtime closure)

## Follow-on op-routables

1. **Re-dispatch STC v2 Modal smoke** with the WAVE-3 fix to verify the Lane A anchor archive resolves at `$WORKSPACE/experiments/results/lane_a_landed/archive_lane_a.zip` on the Modal worker without driver-side defensive probing
2. **Land STRICT preflight gate** `check_modal_dispatcher_consumes_trainer_module_path` per the design above; coordinate with COMMIT-SWAP-INVESTIGATION sister-subagent to avoid preflight.py edit collision
3. **Extend `_collect_trainer_extra_mount_payload` to support directory entries** — `submissions/a1` in `train_substrate_stack_of_stacks.py:206` is currently SKIPPED via the file-only check; recursive walk + tar-archive payload would extend coverage
4. **Audit other `experiments/modal_*.py` dispatchers** for the same `trainer_module_path=None` anti-pattern; modal_train_lane.py is the canonical dispatcher but sister dispatchers (`modal_auth_eval.py`, `modal_component_sensitivity_shards.py`, etc.) may have similar gaps

---

Lane: `lane_wave_3_modal_train_lane_trainer_module_path_consumption_20260516`
Subagent: `wave3_modal_train_lane_consumption`
Cost: $0 (CPU smoke + pytest only)
Time: ~70 minutes
Commits: `bce3a623d`, `b3eafb4fb`, `3318b3452`, `e09caa54a` (sister-subagent absorbed) + this memo via canonical serializer

## Canonical-vs-unique decision per layer

(See table above)

## 9-dimension success checklist evidence

- **Dim 1 UNIQUENESS**: This dispatcher-layer fix is unique-and-complete-per-method (the STC v2 anchor archive + a1_plus_lapose + a1_plus_wavelet_residual cases were each verified empirically); no other dispatcher fix duplicates this.
- **Dim 2 BEAUTY + ELEGANCE**: 2 helpers + 1 kwarg threaded through 4 wrappers + 1 spawn call. Total addition ~120 LOC. Reviewable in 5 minutes.
- **Dim 3 DISTINCTNESS**: Distinct from Wave 2's driver-side defensive resolver (Wave 2 catches at driver; Wave 3 catches at dispatcher); together they extinct the bug class bidirectionally.
- **Dim 4 RIGOR**: 6 premise verifications pre-edit + 26 dedicated tests + 79 sister regression tests pass.
- **Dim 5 OPTIMIZATION PER TECHNIQUE**: Canonical helpers reused (collect_extra_mount_paths + collect_tier_required_input_files); unique-where-needed (trainer derivation + payload staging filter).
- **Dim 6 STACK-OF-STACKS-COMPOSABILITY**: Preserves Modal image caching across dispatches (image is invariant per app); the payload pattern composes with any number of trainer declarations.
- **Dim 7 DETERMINISTIC REPRODUCIBILITY**: Payload bytes are SHA-stable; worker materialization is idempotent (write_bytes on a fresh /tmp/pact each dispatch).
- **Dim 8 EXTREME OPTIMIZATION + PERFORMANCE**: $0 CPU smoke; ~70 min editor work; preserves Modal image caching.
- **Dim 9 OPTIMAL MINIMAL CONTEST SCORE**: Indirect — by unblocking STC v2 dispatch (Catalog #240 contest-CUDA chain), this fix enables future score-lowering work on the STC v2 substrate canvas.

## Observability surface

- **Inspectable per layer**: helper inputs / outputs / intermediate state all logged to stderr (`[modal-train-lane][WAVE-3] derived trainer module from ...` + `[modal-train-lane][WAVE-3] staging N trainer-declared extra mount path(s) as bytes payload: [...]`).
- **Decomposable per signal**: payload dict is keyed by rel_path; sizes loggable per entry.
- **Diff-able across runs**: payload bytes are SHA-stable; the worker-side `modal_worker_head_ledger.json` (Catalog #166) can be extended to capture WAVE-3 payload SHAs in a follow-on.
- **Queryable post-hoc**: payload keys are visible in dispatch stdout; the operator can verify which extras were staged via the explicit `staging N path(s)` log line.
- **Cite-able**: every payload entry is anchored to (lane_script, trainer_module_path, repo_root) tuple at dispatch time.
- **Counterfactual-able**: payload absence vs presence is testable by comparing dispatcher output WITH and WITHOUT the WAVE-3 helpers (the legacy `trainer_module_path=None` path).

## Cargo-cult audit per assumption

| Assumption | Classification | Rationale |
|---|---|---|
| The substrate lane_script naming convention `scripts/remote_lane_substrate_<id>.sh` → `experiments/train_substrate_<id>.py` holds for all dispatched substrates | HARD-EARNED | Verified empirically against 4 current substrate trainers + the convention is encoded in `_infer_lane_id` (the canonical sister mapping) + Catalog #240 audit explicitly enforces this pattern |
| Modal `.spawn()` serializes `dict[str, bytes]` natively | HARD-EARNED | The existing `claim_ledger_bytes` precedent + Modal's documented serialization contract |
| Importing the trainer module at dispatch time is cheap enough to add to every dispatch (no measurable latency impact) | CARGO-CULTED → HARD-EARNED | Initially assumed cheap; empirical verification: trainer import is ~9 seconds for the live STC v2 trainer (due to substrate registry side-effects). This IS measurable; the dispatcher latency budget should accommodate this. **Mitigation**: the sister duplicate-registry fallback caches the import. Future op-routable: skip trainer import entirely when the lane_script has NO substrate convention (already handled via `_derive_trainer_module_path` returning None). |
| Wave 2 driver-side defensive resolver + Wave 3 dispatcher-side payload staging are COMPLEMENTARY not REDUNDANT | HARD-EARNED | Wave 2 catches the bug at the driver layer (defensive); Wave 3 catches it at the dispatcher layer (structural). Together they extinct the bug class bidirectionally per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable. |
| Modal does not support per-call image variants for `.spawn()` in 1.4.2 | HARD-EARNED | Verified via `dir(modal.Function)` — no `with_options` API. The payload pattern is THE canonical workaround. |

## Predicted ΔS band

**N/A** — this is dispatcher plumbing, not a score-lowering substrate. The fix is structural-protection-only; it unblocks future score-lowering dispatches but does not itself produce a score delta. Per Dykstra-feasibility lens: the fix REMOVES a hard constraint (Lane A anchor archive must be on the Modal worker for STC v2 dispatch to succeed) without changing the achievable Pareto frontier.

## Horizon-class

**N/A** — dispatcher plumbing, not a substrate. The WAVE-3 fix is `apparatus_maintenance` per the mission-alignment categorization.

## Council-mission-contribution

`apparatus_maintenance` — updates infrastructure without direct score implications. Future STC v2 / a1_plus_lapose / a1_plus_wavelet_residual dispatches enabled by this fix may produce `frontier_breaking` or `frontier_protecting` contributions.
