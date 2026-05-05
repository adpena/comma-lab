# Recovery Quarantine Signal-Loss Triage - 2026-05-05

Scope: surgical review of `.recovery_quarantine_20260505T004735Z`, stashes,
tracked deletions, orphan-pyc recovery, and live untracked recovery outputs.
This is a custody and promotion ledger, not a score ledger.

Operating decision:

- `main` is the only source-of-truth branch. Forensic trees, detached public
  clones, provider workspaces, stashes, and subagent forks are inputs only.
- No quarantine deletion is accepted until the item is one of:
  1. byte-identical to a live `main` path and represented in
     `.omx/state/signal_loss_audit_20260505T1439Z/quarantine_audit.{json,md}`;
  2. promoted to a canonical live path with focused tests; or
  3. preserved as custody-only evidence with an explicit disposition.
- Do not apply stashes wholesale. Review them as change bundles and promote
  small, tested slices.

Generated audit artifacts:

- `.omx/state/signal_loss_audit_20260505T1439Z/quarantine_audit.json`
- `.omx/state/signal_loss_audit_20260505T1439Z/quarantine_audit.md`
- `.omx/state/signal_loss_audit_20260505T1439Z/quarantine_all_files_first500.txt`
- `.omx/state/signal_loss_audit_20260505T1439Z/staged_deletions.txt`
- `.omx/state/signal_loss_audit_20260505T1439Z/staged_deletions_quarantine_match.tsv`
- `.omx/state/orphan_pyc_recovery_20260505/RECOVERY_INDEX.md`

Quarantine classification after the first canonical promotion:

- `745` total files.
- `566` byte-identical duplicates: safe to delete only after the audit manifest
  is preserved with the cleanup change.
- `87` `.recovery_spec.json` files: preserve until the matching source is
  canonical or intentionally abandoned.
- `88` incomplete recovery files: recovery stubs or raw pycdc output with
  incomplete decompilation markers. Do not promote these as source.
- `3` blocked shell recovery inputs remain: `.PREFLIGHT_DEBT` or
  `.QUARANTINED` launchers that require canonicalization before promotion.
- `0` direct promotion candidates remain after classifier hardening.
- `1` live diff remains: `docs/paper/ara/trace/events.jsonl`, a regenerated
  ARA trace/documentation surface.

Manual decisions so far:

- Promoted `experiments/repack_single_member_archive.py` from recovered pyc
  spec into a hand-rehydrated canonical tool. This is the deterministic
  x-repack utility used by PR105/PR106 public-frontier custody work.
- Added `src/tac/tests/test_repack_single_member_archive.py` to prove payload
  preservation, deterministic member rename, manifest emission, and fail-closed
  behavior on multi-member archives.
- Completed the missing package/API hygiene from a subagent fork:
  `pyproject.toml` now advertises Alpha maturity instead of Stable, and
  `src/tac/__init__.py` exposes lazy public API symbols through
  `_LAZY_PUBLIC_API`, `__getattr__`, and `__dir__` without importing torch or
  pydantic on `import tac`.
- `experiments/profile_hnerv_frontier_payloads.py` and
  `experiments/build_hnerv_frontier_scorecard.py` are live hand-rehydrations
  that intentionally differ from quarantine stubs. They compile and have been
  used to generate the PR105/PR106 payload scorecard.
- The two Modal PR95 `model.py` live diffs were parseable recovery stubs; the
  quarantine copies are raw pycdc fragments. Preserve the live stubs over the
  raw quarantine fragments. Classifier hardening now marks both as incomplete
  recovery rather than live-diff candidates.
- `docs/paper/ara/trace/events.jsonl` differs only as a regenerated trace
  surface. Treat as documentation/observability state, not code promotion.
- `scripts/remote_lane_sjkl_c067.sh` is staged as deleted but is referenced by
  SJ-KL ledgers/tests and by the recovered runbook. Do not accept this deletion
  until either a canonical replacement is present or the SJ-KL remote lane is
  explicitly retired with a tested replacement path.
- `experiments/preflight_pr91_pr92_replay_contracts.py`,
  `experiments/preflight_candidate_manifest_dispatch_readiness.py`, and their
  recovered tests were manually inspected and found to be raw/incomplete pycdc
  output, not promotable source. The audit now detects this bug class via
  `# WARNING: Decompyle incomplete`, `<NODE:`, and similar markers.
- There are `562` staged deletions, all under
  `.omx/auto_memory_snapshot_20260504T230223Z`, and every one has a matching
  byte-present path in `.recovery_quarantine_20260505T004735Z`. This makes the
  deletion set cleanup-eligible only if the quarantine manifest is preserved;
  it does not authorize deleting any non-duplicated source, script, test, doc,
  or research file.

Focused verification:

- `python -m py_compile` passed for the recovered audit/repack/profile/scorecard
  tools and `src/tac/__init__.py`.
- `pytest src/tac/tests/test_package_api_hygiene.py
  src/tac/tests/test_repack_single_member_archive.py -q` passed: `6 passed`.
- `git diff --check` passed for the touched files.

Next triage order:

1. Decide the `scripts/remote_lane_sjkl_c067.sh` deletion by comparing it
   against the current SJ-KL builder/tests and any replacement launcher.
2. Canonicalize or retire the three blocked shell recovery inputs:
   `remote_lane_pr79_segaction_search.sh.PREFLIGHT_DEBT`,
   `remote_lane_q_faithful_jointgen.sh.PREFLIGHT_DEBT`, and
   `remote_lane_sjkl_c067.sh.QUARANTINED`.
3. Extract only high-signal memory-snapshot entries into current `.omx/research`
   ledgers or public-safe docs; do not re-add stale private memory wholesale.
4. After all promoted/rejected items are represented in this ledger and audit
   JSON, quarantine duplicates may be deleted as cleanup.

## 2026-05-05 Codex continuation: no-signal-loss index restore and Yousfi recovery

Recovery-agent review found the staged cleanup was unsafe: live lane scripts
and Yousfi tests were staged as deletions/renames alongside the memory-snapshot
cleanup. I restored the index and worktree copies for:

- `scripts/remote_lane_pr79_segaction_search.sh`
- `scripts/remote_lane_q_faithful_jointgen.sh`
- `scripts/remote_lane_sjkl_c067.sh`
- `src/tac/tests/test_yousfi_3_variance_noise.py`
- `src/tac/tests/test_yousfi_5_uncertainty.py`

The `.omx/auto_memory_snapshot_20260504T230223Z` staged deletions were also
restored. Deletion is deferred until each quarantine/orphan item is represented
by either canonical main-branch source, a tracked research artifact, or an
explicit retirement note.

The ignored state audit was copied into a tracked research artifact directory:

- `.omx/research/artifacts/recovery_quarantine_signal_loss_20260505/quarantine_audit_20260505T1439Z.json`
  - SHA-256 `516d6335509fc041ca70469a7d8a1be2b0c650e182160d590a34bfa35f3d1868`
- `.omx/research/artifacts/recovery_quarantine_signal_loss_20260505/quarantine_audit_20260505T1439Z.md`
  - SHA-256 `eebc631382859b4ef70cb6b4ef651d83308a29e7e0ac1317417d0e21ee3108a2`

Recovered and hardened the Yousfi/Fridrich hidden-gem training surface:

- `tac.fridrich.variance_weighted_noise` is live and exported. It supports
  `variance`, `inverse_variance`, and `wavelet_db4` modes.
- `tac.fridrich.segnet_uncertainty_map` is live and exported.
- `tac.losses.uniward_quant_noise_loss` is live.
- `experiments/train_distill.py` no longer raises `NotImplementedError` when
  `use_variance_noise=True`; both loss paths add the weighted loss.
- `src/tac/experiments/train_renderer.py` now imports and applies
  `uniward_quant_noise_loss` under the existing `use_variance_noise` guard.
- `tac.losses.segnet_uncertainty_weighted_loss` now explicitly accepts BCHW and
  BHWC single-frame tensors. The restored orphan tests caught the old bug where
  BCHW frames were accidentally sent through an HWC pair converter, making
  width look like the channel dimension before SegNet.

Preflight hardening:

- Added `check_feature_flags_have_live_objective_effect(strict=True)` to
  `preflight_all()`.
- The check is AST-based. It rejects a feature guard that only parses/resolves
  but does not call the intended helper, does not add the weighted result to
  `loss`/`total`/`fridrich_extra`, or still raises `NotImplementedError`.
- The first protected feature is `use_variance_noise` in
  `src/tac/experiments/train_renderer.py` and `experiments/train_distill.py`.

Focused verification:

- `pytest src/tac/tests/test_yousfi_3_variance_noise.py src/tac/tests/test_yousfi_5_uncertainty.py src/tac/tests/test_yousfi_variance_uncertainty_recovery.py src/tac/tests/test_losses.py -q`
  passed: `49 passed`.
- `pytest src/tac/tests/test_remote_lane_omega_script.py src/tac/tests/test_remote_lane_omega_v2_script.py src/tac/tests/test_water_filling_codec.py src/tac/tests/test_water_filling_codec_v2.py src/tac/tests/test_omega_w_v2_real_archive.py src/tac/tests/test_joint_admm_proximal_water_filling_v2.py -q`
  passed: `170 passed`.
- `bash -n scripts/remote_lane_pr79_segaction_search.sh scripts/remote_lane_q_faithful_jointgen.sh scripts/remote_lane_sjkl_c067.sh` passed.

Updated triage order:

1. Continue quarantine/orphan recovery by canonicalizing the blocked shell
   inputs and PR95 residual/planner signal; do not delete quarantine wholesale.
2. Normalize PR106 HNeRV/frontier reports around the A++ exact replay and
   payload scorecard.
3. Harvest/close PR106 intN/OWV3/sidechannel exact-eval jobs before new GPU
   dispatch.
4. Extend the live-objective preflight pattern to other high-risk profile
   flags if the broad bug hunters find additional parsed-but-no-op controls.
