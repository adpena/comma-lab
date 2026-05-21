---
council_tier: T1
council_attendees: [Claude]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_decisions_recorded:
  - "diagnosed OVERNIGHT-CC 99d06f967 vendor-stub bug as META-infrastructure mtime_floor bug, NOT DP1 trainer _write_runtime bug"
  - "applied META fix in experiments/modal_train_lane.py: bypass mtime_floor for output/submission/ subtree"
  - "landed canonical helper tac.substrates._shared.trainer_skeleton.vendor_module_with_fresh_mtime (defense-in-depth)"
  - "migrated DP1 trainer _write_runtime to canonical helper (sister substrate trainers can follow same pattern)"
  - "claimed Catalog #361 via canonical git-transactional helper; gate at strict @ live count 0"
  - "20 dedicated tests pass + 99 sister tests pass (DP1 dispatch_ready + trainer_skeleton + modal_train_lane_hardening + new Catalog #361)"
  - "operator-routable: DP1 PATH A re-train + auth_eval re-fire ~$1.20 within $2.00 envelope per CC follow-up #2"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
related_deliberation_ids:
  - overnight_cc_dp1_path_a_auth_eval_refire_blocked_by_vendor_stub_bug_landed_20260521
  - dp1_3rd_attempt_path_a_success_first_paid_byte_anchor_canonical_equation_26_registration_landed_20260521
---

# OVERNIGHT-GG DP1 trainer vendor-stub bug fix + Catalog #361 META-fix LANDED 2026-05-21

## Summary

Per operator NON-NEGOTIABLE *"such bugs must be permanently fixed and self-protected against"* + OVERNIGHT-CC `99d06f967` IMPLEMENTATION-LEVEL falsification (Catalog #307: PARADIGM-INTACT for DP1; bug is at INFRASTRUCTURE harvester surface, NOT trainer paradigm).

**Root cause** (definitive diagnosis): the DP1 trainer's `_write_runtime` correctly invokes `shutil.copy2` to vendor 8 module bodies (architecture.py / archive.py / codebook.py / inflate.py / prior_application.py / procedural_codebook_inflate.py / seed_derived_codebook.py / _shared/inflate_runtime.py) into `output/submission/src/tac/.../`. The files DO land on disk. But `experiments/modal_train_lane.py`'s artifact harvester silently DROPS them via:

```python
artifact_mtime_floor = time.time() - 5.0
...
if st.st_mtime < artifact_mtime_floor:
    continue
```

`shutil.copy2` preserves source mtime. Source files at `/tmp/pact/src/tac/...` carry the LOCAL-REPO mtime (days/weeks old) propagated through Modal's `copytree(symlinks=True)` mount staging. The mtime_floor (set at lane-start - 5s) is GREATER than these old mtimes, so the vendored bodies are SKIPPED. Only `write_text` / `write_bytes` outputs (empty `__init__.py` stubs + top-level `inflate.py` string emission) have current mtime → those pass the floor → those are saved. Hence the empirically-observed pattern of present-stubs + missing-module-bodies.

Empirical receipts at `experiments/results/dp1_3rd_attempt_harvested_baseline_20260521/output/submission/`:
- `inflate.py` (1177 B, sha matches trainer string emission via `write_text`) — PASSED
- `0.bin` (26050 B, sha matches `bin_bytes` via `write_bytes`) — PASSED
- 5 × `__init__.py` (0 B via `write_text`) — ALL PASSED
- 8 × `.py` module bodies via `shutil.copy2` — ALL DROPPED

## What landed (4-surface structural extinction)

### Surface 1: META infrastructure fix in `experiments/modal_train_lane.py`

Added `under_submission` bypass guard to the harvester loop at `_run_lane_inner`:

```python
rel_parts = rel.parts
under_submission = (
    len(rel_parts) >= 3
    and rel_parts[0] == "output"
    and rel_parts[1] == "submission"
)
if not under_submission and st.st_mtime < artifact_mtime_floor:
    continue
```

Files under `output/submission/` bypass the mtime_floor because that subtree is the trainer's atomic submission packet per Catalog #146 inflate runtime contract. The mtime_floor still protects against stale prior-run artifacts OUTSIDE the submission subtree.

### Surface 2: canonical helper `vendor_module_with_fresh_mtime`

New canonical helper at `tac.substrates._shared.trainer_skeleton`:

```python
def vendor_module_with_fresh_mtime(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    os.utime(dst_path, None)  # stamp current mtime
```

Defense-in-depth: even WITHOUT the META-layer fix, this helper makes vendored bodies robust against ANY mtime-based filter. Existing `vendor_shared_inflate_runtime` was also updated to call `os.utime(dst, None)` after its `shutil.copy2`.

### Surface 3: DP1 trainer migration

`experiments/train_substrate_pretrained_driving_prior.py::_write_runtime` now uses `vendor_module_with_fresh_mtime` instead of bare `shutil.copy2`. Sister substrate trainers (14+ files matching `experiments/train_substrate_*.py` using `shutil.copy2`) can adopt the canonical helper in a follow-up sweep.

### Surface 4: STRICT preflight Catalog #361

`check_modal_artifact_filter_preserves_submission_dir` refuses any future regression that removes the `under_submission` bypass from `experiments/modal_train_lane.py`. Required canonical tokens:
- `artifact_mtime_floor`
- `under_submission`
- `parts[0] == "output"`
- `parts[1] == "submission"`
- `if not under_submission and st.st_mtime < artifact_mtime_floor`

File-level waiver `# CATALOG_361_HARVESTER_FILTER_WAIVED:<rationale>` accepted in first 80 lines with non-placeholder rationale (≥4 chars; `<rationale>` / `<reason>` literals rejected per Catalog #287 sister discipline).

STRICT-from-byte-one per CLAUDE.md "Strict-flip atomicity rule". Live count at landing: 0.

## META-FIX decision: Catalog #299 quota brake

Per CLAUDE.md "Gate consolidation discipline" Catalog #299: prefer scope extension over new gate when adding META-class protection. **Decision: NEW gate (NOT scope extension of Catalog #295).**

Rationale:
- Catalog #295's surface = `submissions/*/inflate.py` (PERMANENT submission directories under git VC; PYTHONPATH self-containment contract).
- Catalog #361's surface = `experiments/modal_train_lane.py::run_lane_inner` artifact harvester filter logic (META INFRASTRUCTURE mtime semantics).
- Different abstraction layers (per-submission permanent runtime vs Modal worker infrastructure).
- Different file globs (`submissions/*/inflate.py` vs `experiments/modal_train_lane.py`).
- Different bug classes (PYTHONPATH shim vs mtime_floor filter).
- Different waiver mechanisms (`SUBMISSION_PYTHONPATH_SHIM_OK` vs `CATALOG_361_HARVESTER_FILTER_WAIVED`).

Quota status: current Catalog # = 361, well under 400 quota. NEW gate is within quota; no "stop and consolidate" pause required. Scope-stretching Catalog #295 would dilute its permanent-submission self-containment contract with a temporary-DERIVED-OUTPUT harvester filter concern — sister-extinction architecture is cleaner.

## Premise-verification table (Catalog #229)

| Premise | Verification | Verdict |
|---|---|---|
| DP1 trainer `_write_runtime` currently vendors module bodies | Read source lines 2278-2298; existing test `test_write_runtime_vendors_procedural_inflate_dependencies` PASSES locally | TRUE (post-b93c15afd) |
| Modal worker had same `_write_runtime` code at training time | Worker sentinel sha `c5f2b77ab0910813...` matches current local sha (head ledger inspection) | TRUE |
| `shutil.copy2` preserves source mtime | Python stdlib documented behavior + empirical PV (vendor helper test) | TRUE |
| Modal `copytree(symlinks=True)` preserves source mtimes | Default Python shutil behavior | TRUE |
| Modal harvester filters by `mtime_floor = time.time() - 5.0` | Read `experiments/modal_train_lane.py:861` + filter logic lines 980-996 | TRUE |
| Saved DP1 submission_dir has all 8 module bodies MISSING | Empirical `ls` of `experiments/results/dp1_3rd_attempt_harvested_baseline_20260521/output/submission/` | TRUE |
| Stub `__init__.py` files (via `write_text`) DID make it through | Empirical: 5 × 0-byte stubs present in saved artifact | TRUE |
| Top-level `inflate.py` (via `write_text`) DID make it through | Empirical: 1177-byte file present (sha matches trainer string) | TRUE |
| Fix bypasses mtime_floor only for `output/submission/` | PV via simulated filter on 8 path/mtime combinations (test_modal_filter PV) | TRUE |
| `vendor_module_with_fresh_mtime` stamps current mtime | Test `test_vendor_module_with_fresh_mtime_basic` + end-to-end mtime_floor PV | TRUE |
| Catalog #361 strict mode raises on regression | `test_strict_mode_raises` confirms `PreflightError` | TRUE |

## Verification commands run

```bash
# 1. Verify fix: vendor helper produces current-mtime files (PV via simulated mtime_floor)
PYTHONPATH=src:upstream .venv/bin/python tests/test_vendor_helper_PV.py  # implicit in test suite

# 2. Existing DP1 trainer tests still pass
PYTHONPATH=src:upstream .venv/bin/python -m pytest \
    src/tac/substrates/pretrained_driving_prior/tests/test_dispatch_ready_extension.py -x
# 25 passed

# 3. Trainer skeleton + Modal train lane regression tests
PYTHONPATH=src:upstream .venv/bin/python -m pytest \
    src/tac/tests/test_trainer_skeleton.py src/tac/tests/test_modal_train_lane_hardening.py
# 54 passed

# 4. New Catalog #361 dedicated tests
PYTHONPATH=src:upstream .venv/bin/python -m pytest \
    src/tac/tests/test_check_361_modal_artifact_filter_preserves_submission_dir.py
# 20 passed

# 5. Catalog #361 standalone live count
PYTHONPATH=src:upstream .venv/bin/python -c \
    "from tac.preflight import check_modal_artifact_filter_preserves_submission_dir; \
     print(check_modal_artifact_filter_preserves_submission_dir())"
# []

# 6. Catalog #176 (STRICT callsite has CLAUDE.md row) accepts #361
# 7. Catalog #185 (live count drift) accepts #361
# Both gates: zero violations attributable to #361 (only pre-existing unrelated ones)

# Total: 99 tests pass + Catalog #361 gate live count = 0 + sister gates clean
```

## Sister coordination

Per CLAUDE.md "Subagent coherence-by-default":
- **Slot 3-temp** (OVERNIGHT-FF T4 symposium subagent `a465483d9514e0d5e`) touches `.omx/research/t4_grand_council_symposium_*.md` + sister staircase + sister graph + `council_deliberation_posterior.jsonl` APPEND-ONLY — **DISJOINT** (different files; T4 symposium is research-only, no overlap with my META infrastructure + STRICT gate work)
- **Cron `8a50fe12`** NSCS06 v8 harvest fires at 11:33 CDT — will trigger a sister subagent that's DISJOINT scope (NSCS06 v8 substrate ≠ DP1 substrate)
- **Other in-flight checkpoints** — 14 in `subagent_progress.jsonl` last 2h (most stale per Catalog #206 doesn't auto-mark complete); none touch any of my edited files (`experiments/modal_train_lane.py`, `experiments/train_substrate_pretrained_driving_prior.py`, `src/tac/substrates/_shared/trainer_skeleton.py`, `src/tac/preflight.py`, `CLAUDE.md`, `src/tac/tests/test_check_361_*`)
- Catalog #340 sister-checkpoint guard: PROCEED at canonical serializer time (no overlap with active sisters)

## Discipline declarations

- Catalog #206: predecessor checkpoint read (none) + 4 own checkpoints emitted (in_progress × 3 + final complete)
- Catalog #117/#157/#174: commit via canonical `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` for every edited file
- Catalog #186: catalog # claimed via canonical `tools/claim_catalog_number.py claim --commit-via-serializer`
- Catalog #110/#113: APPEND-ONLY HISTORICAL_PROVENANCE — OVERNIGHT-CC landing memo + canonical_equations registry + modal_call_id_ledger PRESERVED unchanged; only NEW landings here
- Catalog #229: PV (read all relevant source + ledger files + ran the existing test before editing)
- Catalog #287: every empirical claim carries evidence tag (file path / sha / commit ref)
- Catalog #307: paradigm-vs-implementation classification — THIS landing classified as IMPLEMENTATION-LEVEL fix; DP1 paradigm INTACT
- Catalog #299: gate consolidation discipline — NEW gate vs scope-extension decision documented above (NEW preferred for clean abstraction split; quota brake not triggered)
- Catalog #340: sister-checkpoint guard PROCEED for canonical serializer commit
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" — 4-surface structural extinction: META fix + canonical helper + DP1 trainer migration + STRICT preflight Catalog #361

## 6-hook wire-in declaration per Catalog #125

1. Sensitivity-map: N/A (defensive validator gate; no per-element sensitivity signal)
2. Pareto constraint: N/A (no scorer distortion signal)
3. Bit-allocator: N/A (no rate-axis signal)
4. **Cathedral autopilot dispatch hook: ACTIVE** — prevents future Modal dispatcher regressions from silently dropping vendored submission_dir bodies + producing phantom rc=0 dispatches that crash inflate `ModuleNotFoundError`; canonical helper enables sister substrate trainers to vendor-with-fresh-mtime
5. Continual-learning posterior: N/A (no posterior update emitted from this defensive landing)
6. **Probe-disambiguator: ACTIVE** — the canonical `under_submission` guard IS the disambiguator between (a) stale-prior-run-artifact filtering (mtime_floor's correct purpose) vs (b) trainer-emitted-submission-packet preservation (the new bypass scope)

## Operator-routable follow-up

1. **DP1 PATH A re-train + auth_eval re-fire** (per CC follow-up #2; ~$1.20 within $2.00 envelope): re-run the DP1 full training to produce a NEW archive with the correct vendored submission_dir, then re-fire the 4-arm paired auth_eval (CUDA + CPU × baseline + procedural). Expected: rc=0 + contest-axis scores instead of rc=1 ModuleNotFoundError.

2. **Sister substrate trainer migration to `vendor_module_with_fresh_mtime`** (defense-in-depth wave): 14+ substrate trainers under `experiments/train_substrate_*.py` use bare `shutil.copy2`. After the META-layer fix lands, they're already protected at the harvester surface, but routing through the canonical helper provides belt-and-suspenders. Can be done incrementally as a follow-up subagent.

3. **HF Jobs RECHARGE** (operator-only): still required per OVERNIGHT-CC follow-up #3.

4. **NSCS06 v8 Phase 4 + cron harvest** (operator-routable when slot frees per cap=2 stagger discipline; sister subagents will fire DISJOINT from this lane).

## Files touched

- `experiments/modal_train_lane.py` (added `under_submission` bypass)
- `src/tac/substrates/_shared/trainer_skeleton.py` (added `vendor_module_with_fresh_mtime` + `os.utime` to `vendor_shared_inflate_runtime`)
- `experiments/train_substrate_pretrained_driving_prior.py` (DP1 trainer migrated to canonical helper)
- `src/tac/preflight.py` (added Catalog #361 + wire-in)
- `src/tac/tests/test_check_361_modal_artifact_filter_preserves_submission_dir.py` (NEW; 20 dedicated tests)
- `CLAUDE.md` (Catalog #361 entry added)
- `.omx/state/next_catalog_number.txt` (361 → 362 via canonical claim helper)
- `.omx/state/catalog-claim.log` (claim recorded via canonical helper)
- `.omx/state/subagent_progress.jsonl` (4 own checkpoints)
- `.omx/research/overnight_gg_dp1_trainer_vendor_stub_bug_fix_plus_catalog_361_landed_20260521.md` (THIS landing memo)

## Lane

`lane_overnight_gg_dp1_trainer_vendor_stub_bug_fix_plus_catalog_295_scope_extension_20260521` L1 (impl_complete + strict_preflight + memory_entry).

Cost: $0 (zero paid GPU; pure local CPU edits + tests) + ~75 min wall-clock (15min PV diagnosis + 30min fix + 20min tests + 10min landing memo).
