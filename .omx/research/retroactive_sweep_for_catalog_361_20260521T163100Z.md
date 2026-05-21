# Retroactive sweep for Catalog #361 (2026-05-21T16:31:00Z)

Per Catalog #348 EVENT-DRIVEN RETROACTIVE VERDICT-TAINT SWEEP self-protection, every new gate must ship a sweep memo with 4 fields: bug-class symptom signature, pre-fix window, historical-KILL/DEFER/FALSIFY search results, and per-finding RE-EVAL-priority assignment.

## 1. Bug-class symptom signature

**Symptom**: Modal substrate dispatch returns rc=1 with `ModuleNotFoundError` for a `tac.substrates.<id>.<module>` import inside the saved `submission/inflate.py` runtime, despite the substrate trainer's `_write_runtime` correctly invoking `shutil.copy2` to vendor the module bodies.

**Empirical fingerprint** (from OVERNIGHT-CC 99d06f967 anchor):
1. `experiments/results/<substrate>_*/output/submission/src/tac/.../` contains ONLY empty `__init__.py` stubs (0 bytes); NO module bodies (.py files >0 bytes).
2. Top-level `output/submission/inflate.py` IS present (via `write_text`).
3. Top-level `output/submission/0.bin` IS present (via `write_bytes`).
4. Modal worker rc=1 with `ModuleNotFoundError: tac.substrates.<id>.<module>` in stderr.
5. Trainer's `_write_runtime` source code DOES invoke `shutil.copy2(substrate_src / "<module>.py", ...)`.

**Root cause**: `shutil.copy2` preserves source mtime; old local-repo mtimes propagated through Modal's `copytree(symlinks=True)` mount staging fail the `artifact_mtime_floor = time.time() - 5.0` filter in `experiments/modal_train_lane.py::run_lane_inner`.

## 2. Pre-fix window

The mtime_floor filter has existed in `experiments/modal_train_lane.py` since at least 2026-04-15 (the canonical Modal dispatcher pattern). The bug class has been latent THROUGHOUT that window for any substrate trainer using `shutil.copy2` to vendor `output/submission/src/tac/.../` module bodies.

Per `git log --oneline -- experiments/modal_train_lane.py | wc -l`: many revisions in the active window. The mtime_floor was introduced as a guard against stale prior-run artifacts in the workspace, BEFORE substrate trainers started emitting `output/submission/` packets with vendored module bodies.

**The bug class only TRIGGERS when**:
1. Substrate trainer uses `shutil.copy2` to vendor module bodies into `output/submission/src/tac/.../`
2. Modal worker uses `copytree(symlinks=True)` for mount staging (which preserves source mtimes)
3. Source files have mtimes > 5 seconds before lane start

All three conditions are present for EVERY substrate trainer that vendors via `shutil.copy2`. The 14+ substrate trainers matching `experiments/train_substrate_*.py` using `shutil.copy2` are ALL potentially affected.

## 3. Historical KILL / DEFER / FALSIFY search results

Searched git log for substrate trainer KILL / DEFER / FALSIFY verdicts in the active mtime_floor window that may have been mis-attributed to "substrate paradigm failure" when actually caused by the harvester filter:

```bash
git log --all --grep='ModuleNotFoundError' --grep='vendor' --grep='submission_dir' --oneline | head -20
```

**Findings**:

| Date | Anchor | Verdict | Re-eval priority |
|---|---|---|---|
| 2026-05-21 | OVERNIGHT-CC 99d06f967 DP1 PATH A 4-arm auth_eval rc=1 ModuleNotFoundError | IMPLEMENTATION-LEVEL (per Catalog #307) | **N/A — this IS the empirical anchor that surfaced the bug class; no prior verdict to reevaluate; DP1 paradigm INTACT and auth_eval re-fire queued per CC follow-up #2** |
| Older substrate Modal dispatch failures (pre-2026-05-21) | Most were correctly classified as recipe-driver bugs (Catalog #240) or NVML env block bugs (Catalog #244) or recipe Modal-ignored bugs (Catalog #152 WAVE-1) | N/A — different bug classes | LOW |

No prior KILL/FALSIFY/DEFER verdicts on DP1 substrate or sister substrate trainers can be attributed to THIS bug class, because:
1. OVERNIGHT-CC was the FIRST time we tried to re-fire saved DP1 auth_eval on a pre-existing submission_dir — the same problem would have manifested in any prior re-fire attempt, but no prior attempt logged the exact `ModuleNotFoundError: tac.substrates.pretrained_driving_prior.inflate` pattern.
2. The bug class is ONLY visible at AUTH_EVAL TIME (which loads the vendored submission_dir's inflate.py), not at trainer-emission time (the trainer doesn't read back its own vendor output to verify).
3. Prior substrate KILL/DEFER verdicts on DP1 (per OVERNIGHT-X1/X2/Y/Z + sister) were rooted in score-not-better-than-frontier reasoning, NOT in import failures.

## 4. Per-finding RE-EVAL-priority assignment

| Finding | RE-EVAL priority | Rationale |
|---|---|---|
| OVERNIGHT-CC DP1 PATH A 4-arm rc=1 | **HIGH** | Re-fire auth_eval per CC follow-up #2 (~$1.20 within $2.00 envelope); DP1 paradigm INTACT; expect rc=0 + contest-axis scores |
| 14+ sister substrate trainers using `shutil.copy2` | **MEDIUM** | Belt-and-suspenders: migrate to `vendor_module_with_fresh_mtime` canonical helper; META-layer fix already protects at harvester surface, but routing through canonical helper provides defense-in-depth. Can be done incrementally as a follow-up subagent sweep |
| Prior substrate Modal dispatch failures (pre-2026-05-21) | **LOW** | No evidence prior verdicts were mis-attributed; bug class only visible at auth_eval-time and prior failures were rooted in different bug classes (recipe-driver / NVML env / Modal-ignored) all correctly diagnosed at the time |
| Sister Modal-VOLUME path resolution recipes (Catalog #204 + #358 family) | **LOW** | Different bug class (path resolution `/workspace/pact/` vs `/tmp/pact/`); not affected by mtime filter |

## Cross-references

- OVERNIGHT-CC landing: `.omx/research/overnight_cc_dp1_path_a_auth_eval_refire_blocked_by_vendor_stub_bug_landed_20260521.md`
- OVERNIGHT-GG landing: `.omx/research/overnight_gg_dp1_trainer_vendor_stub_bug_fix_plus_catalog_361_landed_20260521.md`
- Catalog #361: `src/tac/preflight.py::check_modal_artifact_filter_preserves_submission_dir`
- Canonical helper: `tac.substrates._shared.trainer_skeleton.vendor_module_with_fresh_mtime`
- Catalog #295 sister (PYTHONPATH self-containment at `submissions/*/inflate.py` permanent surface)
- Catalog #166 sister (Modal worker source-parity ledger at same META infrastructure surface)
- Catalog #339 sister (silent-no-spawn extinction at dispatch-registration surface)

## Conclusion

No historical KILL / DEFER / FALSIFY verdict requires reevaluation as a direct consequence of Catalog #361 landing. The bug class was latent for many weeks but only surfaced at OVERNIGHT-CC's auth_eval re-fire attempt; that anchor IS reevaluated (DP1 paradigm restored from "auth_eval re-fire blocked by vendor-stub bug" to "auth_eval re-fire UNBLOCKED post-OVERNIGHT-GG META fix; HIGH priority").

The 14+ sister substrate trainers using `shutil.copy2` are now structurally protected at the META harvester surface; migrating them to the canonical helper is defense-in-depth, not blocker-removal.
