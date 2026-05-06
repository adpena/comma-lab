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

## 2026-05-05 Codex continuation: dispatch shell and submission gates

Promoted two high-signal pyc recovery stubs into canonical main-branch source
instead of restoring malformed decompiler output.

Canonicalized shell dispatch guard:

- `tools/check_dispatch_cli_shell_hazards.py`
  - SHA-256 `cab2dd453ad96daaa32df8e5596e0902671749680bae1479419e25bce9b00ee1`
- `src/tac/tests/test_dispatch_cli_shell_hazards.py`
  - SHA-256 `fcf7a123d677254d767a0de5c6e068f7a4803f8b74342fa77aa8d233ffb900dc`

Guarded bug classes:

- adjudicator-only `--required-device` / `--required-samples` accidentally
  passed to `scripts/launch_lightning_batch_job.py`;
- known typo flag `--rmote`;
- zsh-facing shell snippets that use `path` as a variable and mutate command
  lookup;
- local/macOS snippets that use GNU-only `find -printf`.

The guard is now wired into `tac.preflight.preflight_all()` through
`check_dispatch_cli_shell_hazards(strict=True)`. The default scan excludes
historical custody/result trees and passed cleanly on the live repo.

Canonicalized pre-submission compliance gate:

- `scripts/pre_submission_compliance_check.py`
  - SHA-256 `481efb3bd0aadda6fa914096e1a047fae6430660aff43873c9f1a46b4e9caf60`
- `src/tac/tests/test_pre_submission_compliance_check.py`
  - SHA-256 `54402dc3f03f592bdd55bdeb8316871222c673b6843922bf29e999c1c8a18dc8`

The new gate validates the upload surface itself and does not create score
claims. It checks required files, executable `inflate.sh`, ZIP member safety
including central/local-header agreement, duplicate members, hidden/resource
sidecars, packed-payload multiplicity, auth-eval archive identity, component
recomputed score, T4/A++ promotion stamps, runtime-tree custody, archive
manifest freshness, report custody links, terminal dispatch claim linkage, and
public supplement private-surface leaks.

New artifacts:

- `.omx/research/artifacts/recovery_quarantine_signal_loss_20260505/dispatch_cli_shell_hazards_20260505.json`
  - SHA-256 `0b4b52b1fef47409e002bdae297ff2541ded4b328c21bbcf6243f3ba298e2a6e`
  - `hazard_count=0`
- `.omx/research/artifacts/recovery_quarantine_signal_loss_20260505/quarantine_audit_after_shell_guard_20260505.json`
  - SHA-256 `99644cb35e768cf85a64b0d44346c51a8b975c05e988e8d78584fd22d015da22`
- `.omx/research/artifacts/recovery_quarantine_signal_loss_20260505/quarantine_audit_after_shell_guard_20260505.md`
  - SHA-256 `eebc631382859b4ef70cb6b4ef651d83308a29e7e0ac1317417d0e21ee3108a2`

Refreshed quarantine audit after the shell-guard promotion:

- `745` quarantined files reviewed.
- `566` are byte-duplicate memory snapshots, safe to delete only after the
  manifest/ledger state is committed.
- `88` remain incomplete recovery stubs that must not be promoted without hand
  rehydration.
- `87` recovery specs must be preserved until matching source is canonical.
- `3` recovery inputs are still blocked and need canonicalization before
  promotion.
- `1` live-diff item remains for manual review:
  `docs/paper/ara/trace/events.jsonl`.

Focused verification:

- `pytest src/tac/tests/test_dispatch_cli_shell_hazards.py
  src/tac/tests/test_preflight_meta_bugs.py::TestPreflightAllInvokesMetaBugChecks -q`
  passed: `7 passed`.
- `check_dispatch_cli_shell_hazards(strict=True, verbose=False)` returned `[]`.
- `pytest src/tac/tests/test_pre_submission_compliance_check.py -q` passed:
  `7 passed`.
- `pytest src/tac/tests/test_dispatch_cli_shell_hazards.py
  src/tac/tests/test_pre_submission_compliance_check.py
  src/tac/tests/test_preflight_meta_bugs.py::TestPreflightAllInvokesMetaBugChecks
  src/tac/tests/test_contest_auth_eval.py
  src/tac/tests/test_score_dashboard_log_fallback.py
  src/tac/tests/test_ara_compile.py
  src/tac/tests/test_predicted_vs_actual_reconciler.py -q` passed:
  `63 passed, 1 skipped`.
- `python -m py_compile` passed for touched Python tools/modules.
- `git diff --check` passed for the new gate/guard changes.

Next triage order:

1. Canonicalize PR95 residual-atom planning as a planning-only hidden-gem tool
   consuming the live PR95 packing profile.
2. Review `docs/paper/ara/trace/events.jsonl` live diff by hand before any
   delete or merge decision.
3. Continue recovering incomplete PR91/PR92 replay and packer-contract tools
   only when their wire contracts can be tested end-to-end.
4. Do not delete quarantine wholesale until all duplicate/preserve/review
   categories are committed to the manifest and this ledger.

### Adversarial review follow-up

Nash read-only review found two P0 issues in the first version of the
pre-submission gate and one live-objective wiring gap. These are now fixed:

- Dispatch-claim linkage now parses the live 8-column ledger by header name:
  `timestamp_utc | agent | lane_id | platform | instance/job_id |
  predicted_eta_utc | status | notes`. It also preserves compatibility with
  the older 6-column test fixture.
- Public hygiene scanning now handles directory paths recursively and delegates
  to `tac.preflight.check_public_release_hygiene`, with the local Modal/provider
  pattern as an extra belt-and-suspenders check.
- `preflight_all()` now calls the real
  `check_feature_flags_have_live_objective_effect(strict=True)` instead of the
  old stub.
- `tools/all_lanes_preflight.py` now runs the shell hazard guard as Gate #0,
  so the guard is visible in the normal operator readiness command.
- `AGENTS.md` and `CLAUDE.md` now document the recovered operator gates and the
  rule that high-signal tools must be wired into normal flows, not left as
  obscure one-offs.

Updated canonical hashes after the adversarial fixes:

- `AGENTS.md`
  - SHA-256 `135441384485b655d3f1cb489b8ee5ff923572b3686f508a37f4c38f79caebd0`
- `CLAUDE.md`
  - SHA-256 `a2d454ea72be726fa9a752c46d0e1606f239139ff6dbca285a0d137ff47ad110`
- `tools/all_lanes_preflight.py`
  - SHA-256 `c4d79e6fc53f3f19fe1fc22753543b07c40fdae232bc5319a921b623e7c1558b`
- `scripts/pre_submission_compliance_check.py`
  - SHA-256 `1946ac79c94113d32eea12f91ddbfc6ba2b15db0be4d62831dd7e2f542da9bec`
- `src/tac/preflight.py`
  - SHA-256 `2f1ad10bcb36d79584aa3387cb1ce542a813fdd386ae127c4b79525e2573dbd4`
- `src/tac/tests/test_pre_submission_compliance_check.py`
  - SHA-256 `7b36a7554c20ee440ed9c7105e602c0c70a52faf3d45c41f7276337607c6fde4`
- `src/tac/tests/test_preflight_meta_bugs.py`
  - SHA-256 `840088e82717df11f2250cc8e5020ce1fdba915e2fd24d855255842d3ea96aa9`

Additional verification:

- `pytest src/tac/tests/test_pre_submission_compliance_check.py
  src/tac/tests/test_dispatch_cli_shell_hazards.py
  src/tac/tests/test_preflight_meta_bugs.py::TestPreflightAllInvokesMetaBugChecks
  src/tac/tests/test_yousfi_variance_uncertainty_recovery.py -q`
  passed: `21 passed`.
- Direct strict guards returned clean:
  `check_dispatch_cli_shell_hazards(...) == []` and
  `check_feature_flags_have_live_objective_effect(...) == []`.
- `tools/all_lanes_preflight.py` passed all visible operator readiness checks:
  Gate #0 dispatch shell hazards, apogee_intN dry-run, and Lane Ω-W-V3 dry-run.
- `python -m py_compile` and `git diff --check` passed for the touched gate,
  preflight, and instruction files.

### Research-state tracking and `tac` cleanup boundary

User clarified the durable target: Claude/OMX memories, findings, state, and
other interesting research records should not disappear, but they belong to the
comma-lab/research layer rather than the clean `tac` library. The implemented
boundary is:

- `tac`: reusable codec/runtime library.
- `comma_lab`: research operations, custody, hosted supplements, provider
  state, recovery audits, and release hygiene.

New implementation:

- `src/comma_lab/research_state.py`
  - Classifies `.omx`, `.claude`, `docs`, and `reports` files into:
    `track_in_git`, `canonicalize_to_research_ledger`,
    `summarize_to_research_ledger`, `externalize_with_manifest`,
    `keep_private_local`, `delete_cache_after_manifest`, or `manual_review`.
  - Uses `git check-ignore --no-index --stdin` so raw public-intake gitlinks and
    submodule-like directories do not abort the audit.
- `tools/audit_research_state_tracking.py`
  - Thin CLI wrapper. The logic intentionally lives under `comma_lab`, not
    `tac`.
- `docs/runbooks/research_state_tracking_policy.md`
  - Records the git/external-hosting boundary.
- `.gitignore`
  - Now documents the policy explicitly: track small `.omx/research` ledgers and
    selected markdown control-plane state; canonicalize or externally host raw
    provider/job/cache/media artifacts.
- `AGENTS.md` and `CLAUDE.md`
  - Now state that `tac` must stay clean and comma-lab owns research-state
    custody.

Audit artifact:

- `.omx/research/artifacts/recovery_quarantine_signal_loss_20260505/research_state_tracking_audit_20260505.md`
- `.omx/research/artifacts/recovery_quarantine_signal_loss_20260505/research_state_tracking_audit_20260505.json`

Latest audit counts:

- total files scanned: `3715`
- `track_in_git`: `620`
- `canonicalize_to_research_ledger`: `562`
- `summarize_to_research_ledger`: `1592`
- `externalize_with_manifest`: `893`
- `keep_private_local`: `20`
- `delete_cache_after_manifest`: `28`

Important interpretation:

- Small `.omx/research` ledgers and summaries should be tracked.
- Raw `.omx/auto_memory_snapshot_*` directories are interesting, but should be
  extracted into ledgers or ARA/paper sources rather than committed wholesale.
- Raw `.omx/state/*.json`, provider logs, Modal/Lightning/Vast transcripts,
  `reports/raw`, `reports/private`, and generated public-site bundles should be
  summarized or externally hosted with manifests, not added blindly.
- The generated public site under `reports/graphs/public_site/` is classified
  as a hosted supplement build, not source. Source belongs in `docs/site` or a
  release manifest; Cloudflare/Wrangler cache remains private local state.

Focused verification:

- `pytest tests/test_comma_lab_research_state.py -q` passed: `8 passed`.
- `python -m py_compile src/comma_lab/research_state.py
  tools/audit_research_state_tracking.py tests/test_comma_lab_research_state.py`
  passed.

Follow-up cleanup boundary fix:

- Added `src/comma_lab/preflight/strict_checks.py` and
  `src/comma_lab/preflight/__init__.py`.
- This makes the ARA/paper references to
  `src/comma_lab/preflight/strict_checks.py` real while delegating to the
  existing `tac.preflight` checks during incremental migration.
- Added the documented anchors:
  `check_no_mps_fallback_default`, `check_42_train_inference_parity`,
  `check_dispatch_cli_shell_hazards`,
  `check_feature_flags_have_live_objective_effect`, and
  `check_public_release_hygiene`.
- `python -m comma_lab.preflight.strict_checks --emit-catalog` now emits a
  machine-readable catalog with the current delegated strict-check count.

Additional focused verification:

- `pytest tests/test_comma_lab_preflight_strict_checks.py
  tests/test_comma_lab_research_state.py -q` passed: `11 passed`.
- `python -m comma_lab.preflight.strict_checks --emit-catalog` emitted clean
  JSON with no runpy warning after lazy package exports.

### PR95 residual-atom recovery and reverse-engineering boundary

The PR95 residual-atom pyc recovery stubs were promoted into the canonical
library boundary instead of left as orphan scripts:

- `src/tac/pr95_hnerv.py` now owns the PR95-family single-member HNeRV archive
  wire grammar: stored `0.bin`, three length-prefixed Brotli streams,
  latent-row zigzag decode/encode, deterministic stored ZIP output, and SHA
  helpers.
- `src/tac/pr95_residual_atoms.py` now owns planning-only latent residual atom
  ledgers, exact-baseline/component-trace validation, signed atom policies, and
  candidate archive emission. Emitted candidates are explicitly not score
  claims and require exact CUDA auth eval.
- `experiments/build_pr95_hnerv_residual_atom_plan.py` is a thin CLI wrapper
  over `tac.pr95_residual_atoms`.
- `experiments/profile_pr95_hnerv_muon_packing.py` now imports the reusable
  PR95 wire helpers from `tac.pr95_hnerv` instead of carrying its own copy of
  the archive grammar.
- `reverse_engineering/README.md` and `reverse_engineering/pr95_hnerv/README.md`
  define the clean reverse-engineering surface. Curated deconstruction
  runbooks/manifests live there; raw PR clones, archives, provider transcripts,
  and large artifacts stay in ignored custody locations with ledger links.

Custody gap found during recovery: the historical PR95 exact JSON paths cited
by earlier ledgers are not present in the current live tree. The PR95 residual
planner therefore should not be run as score-bearing work until the exact JSON
is re-harvested or regenerated. This is a preserved no-signal-loss blocker, not
a reason to discard the planner.

### Reverse-engineering audit wired into preflight

The clean `reverse_engineering/` surface is now part of the normal harness
instead of a standalone convention:

- `src/comma_lab/reverse_engineering.py` classifies every file under
  `reverse_engineering/` with bytes, SHA-256, category, disposition, live-path
  comparison, target path, and reason.
- `tools/audit_reverse_engineering_tree.py` emits JSON/Markdown artifacts and
  supports `--strict --summary` for operator gates.
- `src/tac/preflight.py` now exposes
  `check_reverse_engineering_tree_curation(strict=True)` and wires it into
  `preflight_all()` immediately after the dispatch shell-hazard guard.
- `src/comma_lab/preflight/strict_checks.py` exposes the same check in the
  public strict-check catalog.
- `tools/all_lanes_preflight.py` now runs Gate #1:
  `.venv/bin/python tools/audit_reverse_engineering_tree.py --strict --summary`.

Latest reverse-engineering audit artifact:

- `.omx/research/artifacts/recovery_quarantine_signal_loss_20260505/reverse_engineering_tree_audit_20260505.json`
- `.omx/research/artifacts/recovery_quarantine_signal_loss_20260505/reverse_engineering_tree_audit_20260505.md`

Latest counts:

- total files: `748`
- strict blockers: `0`
- `track_in_git`: `3`
- `summarize_to_research_ledger`: `566`
- `preserve_until_source_disposition`: `87`
- `external_forensics_manifest_only`: `32`
- `compare_and_promote_to_tac`: `11`
- `compare_and_promote_or_ledger`: `29`
- `promote_thin_cli_or_ledger`: `17`
- `canonicalize_to_docs_or_ledger`: `3`

This intentionally permits explicit orphan recovery queues while blocking raw
artifacts, unclassified files, and manual-review leaks from the curated reverse
engineering tree. The promotion queue now gives the next cleanup tranche exact
targets rather than relying on directory spelunking.

Verification:

- `.venv/bin/python tools/audit_reverse_engineering_tree.py --repo-root . --strict
  --summary` passed with `files=748 blockers=0`.
- `.venv/bin/python tools/all_lanes_preflight.py` passed Gate #0 dispatch shell hazards,
  Gate #1 reverse-engineering curation, apogee_intN dry-run, and Lane Ω-W-V3
  dry-run.
- `pytest tests/test_comma_lab_reverse_engineering.py
  tests/test_comma_lab_preflight_strict_checks.py
  src/tac/tests/test_preflight_meta_bugs.py::TestPreflightAllInvokesMetaBugChecks
  -q` passed: `6 passed`.

Duplicate recovery cleanup:

- The audit learned to normalize pyc-recovery banner comments when comparing
  orphan files against live main. This found 16 duplicate recovery copies that
  matched main exactly after stripping the banner.
- Deleted those 16 duplicate copies from
  `reverse_engineering/orphan_pyc_recovery_20260505_codex/` after the audit
  artifact was written. Live canonical files remain in `src/tac/`,
  `experiments/`, `scripts/`, and `tools/`; recovery specs remain preserved.
- The audit now ignores `__pycache__`, `.pytest_cache`, `.mypy_cache`,
  `.ruff_cache`, `.pyc`, and `.pyo` so verification commands do not perturb
  custody counts.

Updated audit after duplicate cleanup:

- total files: `732`
- strict blockers: `0`
- `track_in_git`: `3`
- `summarize_to_research_ledger`: `566`
- `preserve_until_source_disposition`: `87`
- `external_forensics_manifest_only`: `32`
- `compare_and_promote_or_ledger`: `27`
- `promote_thin_cli_or_ledger`: `14`
- `canonicalize_to_docs_or_ledger`: `3`

Updated verification:

- `.venv/bin/python tools/audit_reverse_engineering_tree.py --repo-root . --strict
  --summary` passed with `files=732 blockers=0`.
- `.venv/bin/python tools/all_lanes_preflight.py` passed all 4 gates/lanes with the
  compact reverse-engineering gate.
- `pytest tests/test_comma_lab_reverse_engineering.py
  tests/test_comma_lab_preflight_strict_checks.py
  src/tac/tests/test_preflight_meta_bugs.py::TestPreflightAllInvokesMetaBugChecks
  -q` passed: `6 passed`.

2026-05-05 continuation: public replay preflight recovery

- Promoted recovered
  `experiments/preflight_public_replay_intake.py` back to the live tree. This
  is a static, non-score public-PR replay intake guard: it validates archive
  member safety, PR85-family `x` structure, runtime dependency manifest, and
  source-embedded-payload hazards before exact-eval dispatch.
- Promoted recovered
  `experiments/preflight_candidate_manifest_dispatch_readiness.py` back to the
  live tree. This is a fail-closed candidate manifest guard for the gap between
  builder-local readiness fields and remote exact-eval dispatch.
- Recreated
  `src/tac/tests/test_preflight_candidate_manifest_dispatch_readiness.py`
  manually because the recovered test file contained repeated
  `# WARNING: Decompyle incomplete` sections. The recovered test remains
  quarantine-only until it is deleted or summarized in the next audit pass.
- Promoted recovered
  `src/tac/tests/test_preflight_public_replay_intake.py` after focused
  verification.
- Fixed a live API drift in `src/tac/pr85_bundle.py`: `Pr85Bundle` now exposes
  derived `segment_lengths` and `segment_contracts` properties, preserving one
  canonical parser while supporting recovered public-replay tooling.
- Kept
  `reverse_engineering/orphan_pyc_recovery_20260505_codex/experiments/preflight_pr91_pr92_replay_contracts.py`
  quarantined. It compiles only with `SyntaxWarning` and contains damaged
  decompiler constructs such as `None(report_path)`, `return None(payload)`,
  and chained list indexing. This must be reimplemented from intent, not
  promoted.
- Deleted the three promoted intact orphan copies:
  `experiments/preflight_public_replay_intake.py`,
  `src/tac/tests/test_preflight_public_replay_intake.py`, and
  `experiments/preflight_candidate_manifest_dispatch_readiness.py` under the
  orphan recovery tree. Live canonical copies now own the behavior.

Focused verification:

- `.venv/bin/python -m py_compile src/tac/pr85_bundle.py
  experiments/preflight_public_replay_intake.py
  experiments/preflight_candidate_manifest_dispatch_readiness.py
  src/tac/tests/test_preflight_public_replay_intake.py
  src/tac/tests/test_preflight_candidate_manifest_dispatch_readiness.py`
  passed.
- `.venv/bin/python -m pytest src/tac/tests/test_preflight_public_replay_intake.py
  src/tac/tests/test_preflight_candidate_manifest_dispatch_readiness.py
  src/tac/tests/test_rehydrated_modules_20260505.py -q` passed:
  `41 passed, 1 warning`.
- `.venv/bin/python tools/audit_reverse_engineering_tree.py --repo-root .
  --json-out
  .omx/research/artifacts/recovery_quarantine_signal_loss_20260505/reverse_engineering_tree_audit_20260505.json
  --md-out
  .omx/research/artifacts/recovery_quarantine_signal_loss_20260505/reverse_engineering_tree_audit_20260505.md
  --strict --summary` passed with `files=730 blockers=0`.

2026-05-05 continuation: PR85/STBM1BR + PR92 RMB1 builder recovery

- Promoted recovered
  `experiments/build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.py` back to
  the live tree. This is a deterministic local candidate builder for the
  PR85-family hidden gem where only the STBM randmulti segment is replaced by
  PR92's charged RMB1 randmulti segment after decoded sparse-row parity is
  proven.
- Recreated
  `src/tac/tests/test_build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.py`
  manually because the recovered test was decompiler-damaged. The new test
  builds synthetic PR85, STBM, and PR92 `x` archives, patches the expected
  custody constants to those archives, proves decoded-randmulti parity,
  verifies STBM mask preservation, and asserts no score/remote-dispatch claim.
- Deleted the promoted intact orphan builder and the damaged orphan test copy.
- Left QH0/QM0 serializer candidates unpromoted for now. The wrapper is
  recoverable, but live `src/tac/qh0_record_serializer.py` still has
  `NotImplementedError` record-set state machines, so promoting the wrapper
  first would create a polished no-op. The next correct action is to implement
  or rehydrate the serializer core, then promote the wrapper and tests.
- Removed the older
  `experiments/build_pr85_stbm1br_rmb1_randmulti_candidate.py` orphan copy and
  its test after inspection. That file is decompiler-damaged and superseded by
  the promoted PR92-aware canonical builder; preserving it as live code would
  create duplicate lane semantics and broken helper bodies. The recovery spec
  remains as provenance.
- Reviewed and expanded the hidden-gem registry under `src/tac/hidden_gems.py`
  and `tools/list_hidden_gems.py`. The registry now surfaces the local
  high-EV partial lanes from the hidden-gem sweep: PR106 sidechannels,
  Omega-W-V3 sensitivity, PR95 residual atoms, HNeRV scorecards, RAFT radial
  pose, NeRV L2 readiness, Cool-Chic/C3, foveation/mask grammar, engineered
  correction atoms, stack contracts, and arithmetic sidecars. It remains
  read-only, provider-free, score-claim-free, and repo-relative.

Focused verification:

- `.venv/bin/python -m py_compile
  experiments/build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.py
  src/tac/tests/test_build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.py`
  passed.
- `.venv/bin/python -m pytest
  src/tac/tests/test_build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.py -q`
  passed: `1 passed`.
- `.venv/bin/python -m pytest src/tac/tests/test_hidden_gems.py -q` passed:
  `4 passed`.
- `.venv/bin/python tools/audit_reverse_engineering_tree.py --repo-root .
  --json-out
  .omx/research/artifacts/recovery_quarantine_signal_loss_20260505/reverse_engineering_tree_audit_20260505.json
  --md-out
  .omx/research/artifacts/recovery_quarantine_signal_loss_20260505/reverse_engineering_tree_audit_20260505.md
  --strict --summary` passed with `files=726 blockers=0`.
- `.venv/bin/python tools/all_lanes_preflight.py` passed Gate #0 dispatch
  shell hazards, Gate #1 reverse-engineering curation, Gate #2 hidden-gem
  registry, apogee_intN dry-run, and Omega-W-V3 dry-run.
- `ruff check` passed on the hidden-gem registry surface, all-lanes preflight,
  and the restored RMB1 test after import/lambda cleanup.
- Compared recovered PR95 HNeRV packing profiler copies against live main.
  Live `experiments/profile_pr95_hnerv_muon_packing.py` is canonical because it
  delegates shared wire parsing to `tac.pr95_hnerv`; the orphan source carried
  older embedded helper implementations. The orphan test was
  decompiler-damaged. Deleted those two orphan copies and kept the recovery
  specs as provenance.
- Promoted recovered `experiments/profile_pr94_qpose_intake.py` and
  `src/tac/tests/test_profile_pr94_qpose_intake.py`. This is a static
  byte-forensics profiler for PR94 qpose/tile-action archives; it does not run
  scorers, dispatch GPU work, or make a score claim. The profiler records
  packed segment boundaries, decoded qpose/tile-action structure, MPS-report
  invalidity for promotion, and stackability blockers.
- Deleted the promoted PR94 orphan source/test copies after live tests passed.

Additional focused verification:

- `.venv/bin/python -m pytest
  src/tac/tests/test_profile_pr95_hnerv_muon_packing.py
  src/tac/tests/test_pr95_residual_atoms.py -q` passed: `8 passed`.
- `.venv/bin/python -m pytest
  src/tac/tests/test_profile_pr94_qpose_intake.py -q` passed: `2 passed`.
- `.venv/bin/ruff check experiments/profile_pr94_qpose_intake.py
  src/tac/tests/test_profile_pr94_qpose_intake.py` passed after import-format
  cleanup.
- Promoted recovered `experiments/profile_pr97_h3_intake.py` and
  `src/tac/tests/test_profile_pr97_h3_intake.py`. This static profiler parses
  PR97 H3 single-member payloads, validates mask chunks, pose bit-packing,
  model schema byte accounting, BPGD sidecar structure, runtime static imports,
  and deterministic safe repack candidates. It remains
  `score_claim=false`.
- Deleted the promoted PR97 orphan source/test copies after live tests passed.
- `.venv/bin/python -m pytest src/tac/tests/test_profile_pr97_h3_intake.py -q`
  passed: `2 passed`.
- `.venv/bin/ruff check experiments/profile_pr97_h3_intake.py
  src/tac/tests/test_profile_pr97_h3_intake.py` passed after import-format and
  `collections.abc` cleanup.
- Promoted recovered `experiments/profile_pr95_hnerv_muon_intake.py` and wrote
  a fresh `src/tac/tests/test_profile_pr95_hnerv_muon_intake.py` because the
  recovered test was decompiler-damaged. The profiler statically accounts for
  PR95 HNeRV/Muon archive sections, decoder parameter partitions, latent bytes,
  source-tree training stage facts, score-term math from static intake, and
  immediate improvement hypotheses. It remains external/static intake only and
  not dispatchable score evidence.
- Deleted the PR95 intake orphan source/test copies after live tests passed.
- `.venv/bin/python -m pytest
  src/tac/tests/test_profile_pr95_hnerv_muon_intake.py -q` passed:
  `1 passed`.
- `.venv/bin/ruff check experiments/profile_pr95_hnerv_muon_intake.py
  src/tac/tests/test_profile_pr95_hnerv_muon_intake.py` passed after
  import-format and unused-variable cleanup.
- Promoted recovered `experiments/profile_pr75_minp_archive.py` and added
  `src/tac/tests/test_profile_pr75_minp_archive.py`. The profiler formalizes
  the current PR75/minp single-blob grammar, fixed slice plans, tile-action
  wire forms, pose codec summaries, and runtime parity probes. It is local
  reverse-engineering evidence only and not a score claim.
- Kept `experiments/profile_endgame_archive_decision.py` quarantined because
  it is only a wrapper around incomplete `tac.endgame_archive_decision`
  rehydration stubs.
- Deleted the PR75 orphan source copy after live tests passed.
- `.venv/bin/python -m pytest src/tac/tests/test_profile_pr75_minp_archive.py -q`
  passed: `3 passed`.
- `.venv/bin/ruff check experiments/profile_pr75_minp_archive.py
  src/tac/tests/test_profile_pr75_minp_archive.py` passed after mechanical
  import and lint cleanup.

Updated audit counts after this slice:

- total files: `726`
- strict blockers: `0`
- `track_in_git`: `3`
- `summarize_to_research_ledger`: `566`
- `preserve_until_source_disposition`: `87`
- `external_forensics_manifest_only`: `32`
- `compare_and_promote_or_ledger`: `25`
- `promote_thin_cli_or_ledger`: `10`
- `canonicalize_to_docs_or_ledger`: `3`

## 2026-05-05 Codex Omega-W-V3 Sensitivity Gate Greenup

Integrated the recovered/partner Ω-W-V3 strict sensitivity contract into the
main preflight surface without breaking the fast CPU smoke path:

- Reviewed `tools/dispatch_dryrun_omega_w_v3.py` strict mode. The default
  dry-run still validates wrapper syntax, PR106 archive presence, stub
  sensitivity presence, extract/repack scripts, Apogee v2 inflate adapter,
  stage-1 extraction, stage-3 byte-exact stub repack, and parser round-trip.
- Strict mode now rejects non-promotable sensitivity artifacts before GPU
  dispatch. It requires CUDA metadata, rejects stub/planning/stale markers, and
  checks `source_archive_sha256` and optional byte count against the selected
  PR106 archive.
- Added `tools/all_lanes_preflight.py --require-real-omega-sensitivity` so the
  top-level harness can be promoted from CPU smoke readiness to real
  Ω-W-V3 dispatch readiness when a CUDA sensitivity map exists. The default
  all-lanes command remains green and fast for local recovery/harness checks.

Verification:

- `.venv/bin/python -m ruff check tools/dispatch_dryrun_omega_w_v3.py
  src/tac/tests/test_dispatch_dryrun_omega_w_v3.py tools/all_lanes_preflight.py`
  passed.
- `.venv/bin/python -m pytest src/tac/tests/test_dispatch_dryrun_omega_w_v3.py
  src/tac/tests/test_lane_omega_w_v3_local_smoke.py -q` passed: `12 passed`.
- `.venv/bin/python tools/dispatch_dryrun_omega_w_v3.py -v` passed all
  `8` default checks and preserved the `164,087` byte stub invariant.
- `.venv/bin/python tools/dispatch_dryrun_omega_w_v3.py
  --require-real-sensitivity` failed closed as intended on the current CPU
  stub map: non-CUDA device, `is_stub=true`, stub tag, and missing
  `source_archive_sha256`.
- `.venv/bin/python tools/all_lanes_preflight.py` passed: Gate #0 shell/CLI,
  Gate #1 reverse-engineering curation, Gate #2 hidden-gem registry, Apogee
  intN, default Ω-W-V3, and PR106 sidechannels are all green.
- `.venv/bin/python tools/all_lanes_preflight.py
  --require-real-omega-sensitivity` failed only Ω-W-V3, which is the desired
  dispatch block until real CUDA sensitivity evidence is available.

## 2026-05-05 Codex PR95 Residual Atom Planner Hygiene

Reviewed the live PR95/HNeRV residual-atom planner that had already been
promoted out of recovery:

- `src/tac/pr95_residual_atoms.py` is the canonical reusable implementation.
  It emits planning-only residual atom ledgers, signed latent atom policies,
  and optional candidate archive builds with `score_claim=false` and
  `exact_eval_ready=false` until exact CUDA eval is run separately.
- `experiments/build_pr95_hnerv_residual_atom_plan.py` is only a thin CLI
  wrapper over `tac.pr95_residual_atoms`.
- The remaining orphan test copy was a damaged decompile and is now deleted
  from the recovery tree. The live `src/tac/tests/test_pr95_residual_atoms.py`
  is the canonical behavioral coverage.
- Fixed import/lint hygiene in the live planner instead of leaving another
  recoverable-but-sharp edge in the stack.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_pr95_residual_atoms.py -q`
  passed: `4 passed`.
- `.venv/bin/python -m ruff check src/tac/pr95_residual_atoms.py
  experiments/build_pr95_hnerv_residual_atom_plan.py
  src/tac/tests/test_pr95_residual_atoms.py` passed.
- `.venv/bin/python experiments/build_pr95_hnerv_residual_atom_plan.py --help`
  showed the canonical CLI surface and confirmed no invented flags were needed.

## 2026-05-05 Codex Hidden-Gem Readiness And QH0 Recovery

Accepted Dalton's isolated hidden-gem readiness audit and wired it into the
top-level local preflight:

- Added `src/tac/hidden_gem_readiness.py`, a deterministic audit that hashes
  hidden-gem evidence paths and integration targets, classifies local-patch
  readiness, and keeps every row `ready_for_exact_eval_dispatch=false`.
- Added `tools/audit_hidden_gem_readiness.py` and
  `docs/runbooks/hidden_gem_readiness_audit.md`.
- Added Gate #3 to `tools/all_lanes_preflight.py`: ready-for-patch hidden gems
  must have live evidence paths and live integration targets, with zero
  exact-eval dispatch readiness implied by the registry alone.

Recovered a material QH0 hidden gem from the public PR85 source rather than
leaving it as a rehydration stub:

- Implemented `tac.qh0_renderer_codec.decode_qh0_state_dict()` for the
  reviewed PR85/PR89 `QH0`/`QM0` record grammar: FP4 Conv/Embedding records,
  FP16 dense records, row-scaled int8 dense records, QH0 hi/lo FP4 split, and
  QH0 even/odd byte split.
- Replaced `tac.qh0_record_serializer` stubs with a deterministic parser and
  serializer that emits canonical QH0 and QM0 payload variants, proves decoded
  tensor parity, summarizes record byte mass, and filters byte-win candidates
  without dispatching.
- Added `src/tac/tests/test_qh0_record_serializer.py` and deleted the damaged
  recovered orphan test copy.
- Screened the recovered PR85 QH0 serializer candidate against the live PR85
  archive. Result: QH0->QM0 tensor parity is exact, but the screened Brotli/QM0
  grid did not produce a real byte win over the original QH0 model segment.
  This is a scoped `no_real_byte_win` result for that serializer transform, not
  a method-family kill.

Fixed a preflight metabug surfaced by the new all-lanes run:

- `tools/check_dispatch_cli_shell_hazards.py` now avoids flagging its own
  prose/error strings and Python-embedded remote snippets as local GNU
  `find -printf` commands. The check still catches real local shell commands
  using GNU-only `find -printf`.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_qh0_record_serializer.py
  src/tac/tests/test_rehydrated_modules_20260505.py -q` passed: `33 passed`.
- `.venv/bin/python -m ruff check src/tac/qh0_renderer_codec.py
  src/tac/qh0_record_serializer.py src/tac/tests/test_qh0_record_serializer.py`
  passed.
- `.venv/bin/python reverse_engineering/orphan_pyc_recovery_20260505_codex/
  experiments/build_pr85_qh0_serializer_candidates.py --archive
  experiments/results/public_pr85_intake_20260503_codex/archive.zip
  --replay-inflate-py reverse_engineering/orphan_pyc_recovery_20260505_codex/
  experiments/results/public_pr85_intake_20260503_codex/replay_submission/
  inflate.py --robust-current-dir submissions/robust_current --out-dir
  /tmp/pact_qh0_serializer_probe_live_runtime` completed with
  `built_candidate_count=0`, `blocker_class=no_real_byte_win`.
- `.venv/bin/python -m pytest src/tac/tests/test_hidden_gem_readiness.py
  src/tac/tests/test_hidden_gems.py -q` passed: `8 passed`.
- `.venv/bin/python tools/audit_hidden_gem_readiness.py --format json --status
  ready_for_patch --fail-if-missing-targets --fail-if-missing-evidence`
  passed: `5` ready-for-patch rows, no missing evidence, no missing targets,
  zero exact-eval dispatch-ready rows.
- `.venv/bin/python -m ruff check tools/check_dispatch_cli_shell_hazards.py`
  passed after the scanner fix.
- `.venv/bin/python tools/all_lanes_preflight.py` passed all `7` gates/lanes:
  shell/CLI hazards, reverse-engineering curation, hidden-gem registry,
  hidden-gem readiness, Apogee intN, default Ω-W-V3, and PR106 sidechannels.

## 2026-05-05 Codex PR91/PR92 Damaged-Decompile Preservation

Inspected the remaining orphan promotion queue and selected the damaged
`experiments/preflight_pr91_pr92_replay_contracts.py` recovery item. It is not
safe to promote as live source: the recovered file still contains decompiler
damage markers and impossible pycdc constructs such as `None(report_path)` and
`return None(payload)`.

Instead of promoting the damaged source, the live reverse-engineering audit now
classifies damaged pyc recovery outputs as
`preserve_until_hand_rehydration`. This keeps the intent/provenance visible in
JSON and Markdown audit output while removing these files from the promotion
queue until a hand-rehydrated implementation and focused tests exist.

Current audit effect:

- `experiments/preflight_pr91_pr92_replay_contracts.py`:
  `preserve_until_hand_rehydration`
- `src/tac/tests/test_preflight_pr91_pr92_replay_contracts.py`:
  `preserve_until_hand_rehydration`
- reverse-engineering strict blockers remain `0`

Verification:

- `.venv/bin/python -m pytest tests/test_comma_lab_reverse_engineering.py -q`
  passed: `1 passed`.
- `.venv/bin/python -m ruff check src/comma_lab/reverse_engineering.py
  tests/test_comma_lab_reverse_engineering.py` passed.
- `.venv/bin/python -m py_compile src/comma_lab/reverse_engineering.py
  tests/test_comma_lab_reverse_engineering.py` passed.
- `.venv/bin/python tools/audit_reverse_engineering_tree.py --repo-root .
  --strict --summary` passed with `files=713 blockers=0`.

## 2026-05-05 Codex Engineered-Correction Readiness Metabug Fix

Promoted the recovered engineered-correction readiness idea into a permanent
guard instead of leaving it as hidden lane-local knowledge.

Fixed bug classes:

- CLI import closure: `tools/audit_engineered_corrections.py` now adds both
  repo root and `src` to `sys.path`, so subprocess audits can import
  `experiments.precompute_gradient_corrections` from any working directory.
- Byte-cap visibility: manifest blockers such as `score_claim=true` no longer
  prevent the audit from still packing the payload and reporting
  `packed_bytes_exceed_cap_*`.
- Wire-format shape safety: the readiness guard fails closed unless sparse
  correction payloads match the current 3-channel RGB wire contract.
- Int4 semantic safety: qbits=4 payloads now fail closed when values fall
  outside the signed 4-bit `[-7, 7]` apply contract.
- Canonical unpack metadata: `unpack_sparse_corrections()` now preserves
  `top_k_pct`, so disk artifacts can round-trip through the canonical packer
  without dropping header metadata.
- Preflight coverage: `tools/all_lanes_preflight.py` now runs
  `tools/audit_engineered_corrections.py --self-test --fail-if-not-ready` as
  Gate #4. The gate proves the guard is importable and local-patch-ready while
  preserving `ready_for_exact_eval_dispatch=false`.

Updated the hidden-gem registry entry for `engineered_correction_atom_gate` so
it points at the live readiness module, CLI, and preflight integration. The
next actionable patch is no longer "add a gate"; it is to feed a real
component-trace manifest into the guarded local patch path and keep exact
dispatch blocked until charged archive bytes consume the atoms.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_engineered_correction_readiness.py
  src/tac/tests/test_lane_ec_v2_greedy.py -q` passed: `17 passed`.
- `.venv/bin/python -m ruff check experiments/precompute_gradient_corrections.py
  src/tac/engineered_correction_readiness.py
  src/tac/tests/test_engineered_correction_readiness.py
  tools/audit_engineered_corrections.py tools/all_lanes_preflight.py` passed.
- `.venv/bin/python tools/audit_engineered_corrections.py --self-test
  --max-packed-bytes 10000 --fail-if-not-ready` passed with no blockers,
  `ready_for_local_patch=true`, and `ready_for_exact_eval_dispatch=false`.
- `.venv/bin/python tools/all_lanes_preflight.py` passed all `8` gates/lanes:
  shell/CLI hazards, reverse-engineering curation, hidden-gem registry,
  hidden-gem readiness, engineered-correction readiness, Apogee intN, default
  Ω-W-V3, and PR106 sidechannels.

This remains local readiness and safety evidence only. It is not a score claim
and does not authorize GPU dispatch without a charged archive candidate and
normal dispatch claim.

## 2026-05-05 Codex Lightning Harvest Recovery

Refreshed current Lightning state after the local recovery hardening pass.
The `apogee_int4_postfix_sanity_20260505T172500Z` exact-eval job had reached
`Completed` remotely but local custody still held only `source_manifest.json`.

Harvested through the state-derived SSH path using the local
`${LIGHTNING_SSH_TARGET}` after confirming the repo doctor passed remote SSH
and supply-chain checks. The local SSH alias was not configured on this
machine and failed doctor with DNS/SSH config errors; public release surfaces
must keep the concrete Studio target in private custody only.

Harvested result:

- job: `apogee_int4_postfix_sanity_20260505T172500Z`
- evidence: A++ T4 exact CUDA custody, but A-negative scoped forensic result
- archive bytes: `109996`
- archive SHA-256:
  `3994b5fb3ecb3b06e74a10c7ceb02b5fa531b2968d7b6d25495a2769d0e4a06e`
- score: `1.4286639424744803`
- SegNet: `0.00868503`
- PoseNet: `0.02370903`
- hardware: `Tesla T4`, `n_samples=600`
- adjudication status: `REGRESSION_REVIEW_REQUIRED`, promotion ineligible
- copied files include `contest_auth_eval.json`,
  `contest_auth_eval.adjudicated.json`, `archive.zip`, `eval_provenance.json`,
  `report.txt`, `auth_eval.log`, DALI/runtime preflight artifacts, and
  supply-chain scans.

Interpretation:

- This falsifies the Apogee int4 postfix implementation as currently wired.
  It is not a broad block-FP or HNeRV-family kill.
- The byte drop was real (`109996` bytes vs PR106 baseline `186239`), but the
  component damage dominated: score delta vs baseline was
  `+1.2192072124744802`.
- The active dispatch ledger already contains a newer terminal
  `falsified_score_1.43_pareto_dominated` row for this job, which closes the
  older active claim under the dispatch-claim helper semantics.

Current frontier reminder from local A++ artifacts remains PR106 x-repack:

- score: `0.20945123680571204`
- bytes: `186231`
- SegNet: `0.00067142`
- PoseNet: `0.00003351`
- artifact:
  `experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/contest_auth_eval.adjudicated.json`

Ongoing:

- `apogee_int8_baseline_confirm_20260505T174500Z` refreshed as `Pending` on
  T4 with cost about `$0.1377`; no local result artifacts yet.

## 2026-05-05 Codex Apogee Distortion-Model Guard

The Apogee intN lane exposed a permanent bug class: parser/packer/readiness
checks can prove a byte transformation exists without proving the transformed
representation remains in the SegNet/PoseNet scorer basin. The harvested int4
result is therefore canonical only as an exact negative for that specific
archive/config, not as a frontier datapoint and not as a family kill.

Permanent fixes landed:

- `tools/dispatch_dryrun_apogee_intN.py` now fails closed by default with
  `ready_for_exact_eval_dispatch=false` unless the operator explicitly passes
  `--allow-forensic-byte-only`. The explicit mode still says not to dispatch
  as a score lane without a distortion model or exact CUDA evidence.
- `tools/apogee_intN_pareto.py` no longer emits dispatch one-liners by
  default. Its JSON rows carry `ready_for_exact_eval_dispatch=false` and
  blockers for `missing_contest_faithful_distortion_model`,
  `missing_scorer_basin_parity_gate`, and
  `byte_only_prediction_not_score_evidence`.
- `tools/all_lanes_preflight.py` treats Apogee intN as a self-protected
  forensic-only lane rather than a GPU-dispatch GO signal.
- `experiments/build_hnerv_frontier_scorecard.py` now preserves
  adjudication/promotion/regression blockers as `canonicality_blockers` and
  marks exact-negative rows as `canonical_frontier_eligible=false` when that
  metadata is present.

Verification:

- `.venv/bin/python -m pytest
  src/tac/tests/test_apogee_intN_pareto_dispatch_wiring.py
  src/tac/tests/test_build_hnerv_frontier_scorecard.py -q` passed:
  `10 passed`.
- `.venv/bin/python -m ruff check tools/dispatch_dryrun_apogee_intN.py
  tools/apogee_intN_pareto.py tools/all_lanes_preflight.py
  experiments/build_hnerv_frontier_scorecard.py
  src/tac/tests/test_apogee_intN_pareto_dispatch_wiring.py
  src/tac/tests/test_build_hnerv_frontier_scorecard.py` passed.
- `.venv/bin/python tools/dispatch_dryrun_apogee_intN.py --bits 4` now fails
  on the distortion-model gate before dispatch.
- `.venv/bin/python tools/dispatch_dryrun_apogee_intN.py --bits 4
  --allow-forensic-byte-only` passes only as forensic byte-only and prints
  `ready_for_exact_eval_dispatch=false`.

Follow-up adversarial review found one remaining P0 escape hatch:
`tools/lightning_dispatch_pr106_stack.py` could still submit `apogee_int*`
directly from predicted bands. That path is now fail-closed:

- `tools/lightning_dispatch_pr106_stack.py` requires
  `--apogee-distortion-gate-json` for any `apogee_int*` GPU dispatch. The gate
  must match the candidate archive SHA-256 and must prove one of:
  `distortion_model_status=passed`, `scorer_basin_parity_status=passed`, or
  `exact_positive_cuda_evidence=true`.
- `--allow-forensic-apogee-intN` is accepted only with `--print-only`; it can
  render commands for review but cannot stage, claim, or submit GPU work.
- `experiments/preflight_candidate_manifest_dispatch_readiness.py` now blocks
  Apogee-like or `score_affecting_payload_changed=true` manifests unless they
  carry a positive distortion/scorer-basin gate. Score-affecting payload
  changes must also record `source_archive_sha256`.
- Stale operator/paper docs now carry supersession notes marking Apogee intN
  predictions forensic/noncanonical and PR106 x-repack as a byte-identical
  custody/rate control rather than a representation advance.

Additional verification:

- `.venv/bin/python -m pytest src/tac/tests/test_lightning_dispatch_pr106_stack.py
  src/tac/tests/test_preflight_candidate_manifest_dispatch_readiness.py -q`
  passed: `19 passed`.
- `.venv/bin/python -m ruff check tools/lightning_dispatch_pr106_stack.py
  experiments/preflight_candidate_manifest_dispatch_readiness.py
  src/tac/tests/test_lightning_dispatch_pr106_stack.py
  src/tac/tests/test_preflight_candidate_manifest_dispatch_readiness.py`
  passed.
- `.venv/bin/python tools/lightning_dispatch_pr106_stack.py --lane apogee_int4
  --archive experiments/results/apogee_int4_repack_20260504_claude/apogee_int4_archive.zip
  --predicted-low 0.155 --predicted-high 0.180 --print-only` now fails before
  staging or claiming with the required Apogee distortion-gate message.

Regenerated
`experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/scorecard.{json,md}`
from the local exact JSON artifacts that still exist on disk. The refreshed
scorecard now records PR105/PR105x and PR106/PR106x as byte-identical payload
groups with 8-byte ZIP-wrapper spans and "custody/control only" readiness, so
the x-repack rows cannot be mistaken for representation improvements.

Also downgraded the Apogee intN generator/runtime/old manifests:

- `experiments/repack_pr106_with_intN_block_fp.py` now writes
  `ready_for_exact_eval_dispatch=false`, `score_affecting_payload_changed=true`,
  `source_archive_sha256`, `candidate_archive_sha256`, forensic prediction
  status, and the three distortion-gate blockers.
- Existing `experiments/results/apogee_int{4,5,6,7,8}_repack_20260504_claude/repack_metadata.json`
  files were re-stamped with the same forensic-only status and real SHA-256s.
- `submissions/apogee_intN/inflate.py` docstring now says the historical
  predicted bands are byte-only planning artifacts invalidated by exact int4
  scorer-basin collapse.

Verification:

- `.venv/bin/python -m pytest
  src/tac/tests/test_apogee_intN_pareto_dispatch_wiring.py
  src/tac/tests/test_build_hnerv_frontier_scorecard.py
  src/tac/tests/test_lightning_dispatch_pr106_stack.py
  src/tac/tests/test_preflight_candidate_manifest_dispatch_readiness.py
  src/tac/tests/test_predicted_vs_actual_reconciler.py -q` passed:
  `34 passed`.
- `.venv/bin/python -m ruff check experiments/repack_pr106_with_intN_block_fp.py
  submissions/apogee_intN/inflate.py tools/apogee_intN_pareto.py
  tools/dispatch_dryrun_apogee_intN.py
  experiments/preflight_candidate_manifest_dispatch_readiness.py` passed.
- `.venv/bin/python tools/all_lanes_preflight.py` passed all `8` checks with
  Apogee intN explicitly self-protected and not dispatch-ready.

## 2026-05-05 Codex Lightning Defaults And Reconciler Hardening

Centralized Lightning Studio environment handling so command surfaces stop
drifting between local aliases and direct Studio SSH targets.

- Added `src/tac/deploy/lightning/defaults.py` with public-safe defaults:
  user/teamspace/studio/SSH target come from environment or CLI only, while
  remote pact/tac paths remain generic Studio paths.
- Updated `scripts/lightning_repro_workspace.py`,
  `scripts/launch_lane_lightning.py`, `tools/lightning_dispatch_pr106_stack.py`,
  `tools/lightning_run.sh`, `tools/lightning_monitor.sh`, and
  `scripts/pfp16_a_plus_plus_exact_t4_eval.sh` to require an explicit
  environment/CLI target instead of embedding operator-specific provider state.
- Updated `tools/predicted_vs_actual_reconciler.py` so it no longer frames
  Apogee intN rows as "beats PR106" dispatch opportunities. It now reports the
  historical bands as forensic byte-only, shows dispatch blocked, and carries
  prediction status plus dispatch blockers in JSON output.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_lightning_repro_workspace.py
  src/tac/tests/test_lightning_dispatch_pr106_stack.py -q` passed:
  `30 passed`.
- `.venv/bin/python -m ruff check src/tac/deploy/lightning/defaults.py
  scripts/lightning_repro_workspace.py scripts/launch_lane_lightning.py
  tools/lightning_dispatch_pr106_stack.py
  src/tac/tests/test_lightning_repro_workspace.py
  src/tac/tests/test_lightning_dispatch_pr106_stack.py` passed.
- `bash -n tools/lightning_run.sh tools/lightning_monitor.sh
  scripts/pfp16_a_plus_plus_exact_t4_eval.sh` passed.
- `.venv/bin/python -m pytest src/tac/tests/test_predicted_vs_actual_reconciler.py
  -q` passed: `5 passed`.

## 2026-05-05 Codex Omega-W-V3 Stub-Dispatch Language Guard

Closed another dispatch-readiness ambiguity surfaced by the integrated
preflight. `tools/dispatch_dryrun_omega_w_v3.py` already had a strict
`--require-real-sensitivity` metadata gate, but the default stub-mode dry-run
still printed operator-facing dispatch language. That was a bug class: local
parser/repack parity can look green while the real CUDA sensitivity map is still
stub/CPU/planning-only.

Changes:

- `tools/dispatch_dryrun_omega_w_v3.py` now prints
  `ready_for_remote_cuda_dispatch=false` in default stub mode and says it is
  local smoke only.
- The same tool prints `ready_for_remote_cuda_dispatch=true` only when
  `--require-real-sensitivity` passes.
- `tools/all_lanes_preflight.py` marks the default Omega row as
  `LOCAL-SMOKE ONLY, NOT DISPATCH-READY`.
- The current strict Omega check correctly rejects the checked-in stub map:
  CPU device, `is_stub=true`, stub tag, and missing source archive SHA.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_dispatch_dryrun_omega_w_v3.py
  src/tac/tests/test_lane_omega_w_v3_local_smoke.py -q` passed:
  `12 passed`.
- `.venv/bin/python -m ruff check tools/dispatch_dryrun_omega_w_v3.py
  tools/all_lanes_preflight.py src/tac/tests/test_dispatch_dryrun_omega_w_v3.py`
  passed.
- `.venv/bin/python tools/all_lanes_preflight.py` passed all `8` checks while
  classifying Apogee as forensic-only and Omega-W-V3 as local-smoke only.

## 2026-05-05 Codex HNeRV Scorecard Routing Gate

Promoted the public HNeRV payload scorecard from a passive generated artifact
into an integrated preflight gate. This prevents stale PR105/PR106 anatomy from
driving hidden-gem follow-up work after exact replay artifacts move.

Changes:

- Added `tools/audit_hnerv_frontier_scorecard.py`.
- The audit requires `score_truth=exact_cuda_auth_eval_json`, a canonical A++
  `PR106x` row, payload/profile SHA custody, the byte-identical PR106/PR106x
  control group, and both high-value follow-up classes:
  decoder self-compression/weight-stream recoding and latent/sidecar arithmetic
  coding.
- Wired the audit into `tools/all_lanes_preflight.py` as `Gate #5`.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_audit_hnerv_frontier_scorecard.py
  -q` passed: `3 passed`.
- `.venv/bin/python -m ruff check tools/audit_hnerv_frontier_scorecard.py
  src/tac/tests/test_audit_hnerv_frontier_scorecard.py tools/all_lanes_preflight.py`
  passed.
- `.venv/bin/python tools/audit_hnerv_frontier_scorecard.py` passed:
  `4` rows, `2` payload groups, `12` follow-up targets.
- `.venv/bin/python tools/all_lanes_preflight.py` passed all `9` checks with
  the new HNeRV frontier scorecard gate included.

## 2026-05-05 Codex Audit Consolidation Start

Started canonicalizing the recovered audit/preflight ecosystem. The codebase
has many useful but script-shaped audits with duplicated repo-root handling,
SHA helpers, JSON rendering, blocker fields, and non-dispatch metadata. A broad
rename/refactor would be high churn in the current dirty shared tree, so this
tranche landed the shared primitives and migrated only the newly touched
readiness surfaces.

Changes:

- Added `src/tac/audit_contract.py` with `AuditReport` and `audit_exit_code`.
  New audits now share the same core JSON contract:
  `audit`, readiness boolean, `blockers`, `summary`, `score_claim=false`, and
  `dispatch_attempted=false`.
- Added `src/tac/repo_io.py` with deterministic `json_text`, `read_json`,
  `write_json`, `sha256_file`, and `repo_relative`.
- Migrated `tools/audit_hnerv_frontier_scorecard.py` to the shared audit
  contract and deterministic JSON helper.
- Migrated `tools/audit_hidden_gem_readiness.py` and
  `src/tac/hidden_gem_readiness.py` onto `json_text` / `sha256_file` so those
  recovered hidden-gem audits stop carrying private helper copies.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_repo_io.py
  src/tac/tests/test_audit_contract.py
  src/tac/tests/test_audit_hnerv_frontier_scorecard.py
  src/tac/tests/test_hidden_gem_readiness.py -q` passed: `12 passed`.
- `.venv/bin/python -m ruff check src/tac/repo_io.py src/tac/audit_contract.py
  tools/audit_hnerv_frontier_scorecard.py tools/audit_hidden_gem_readiness.py
  src/tac/hidden_gem_readiness.py src/tac/tests/test_repo_io.py
  src/tac/tests/test_audit_contract.py
  src/tac/tests/test_audit_hnerv_frontier_scorecard.py
  src/tac/tests/test_hidden_gem_readiness.py` passed.

Recommended next consolidation order:

1. Extract script bootstrap/path setup into one helper for tools that still
   manually edit `sys.path`.
2. Replace local `_sha256_file` / `_write_json` helpers in newly touched
   `experiments/` scripts with `tac.repo_io` as those lanes are recovered.
3. Keep historical public-submission intake trees read-only; do not churn
   vendored competitor sources for style consolidation.

## 2026-05-05 Codex Audit Consolidation Follow-Up

Migrated the next two active readiness surfaces onto the canonical audit
contract while preserving their legacy JSON fields for operator compatibility.

Changes:

- `tools/dispatch_dryrun_pr106_sidechannels.py` now emits
  `audit=pr106_sidechannel_dispatch_dryrun`,
  `ready_for_local_readiness`, `blockers`, and a compact `summary` in JSON,
  while preserving `schema`, `ok`, `checks`, `warnings`, `score_claim=false`,
  `dispatch_attempted=false`, `gpu_required=false`, and
  `provider_state_free=true`.
- `tools/audit_engineered_corrections.py` now emits
  `audit=engineered_correction_readiness`, canonical blockers/summary, and the
  existing correction-readiness fields in one payload.
- Both tools use `tac.repo_io.json_text`; the PR106 sidechannel tool also uses
  `tac.repo_io.read_json` for manifest checks.

Verification:

- `.venv/bin/python -m pytest
  src/tac/tests/test_dispatch_dryrun_pr106_sidechannels.py
  src/tac/tests/test_engineered_correction_readiness.py
  src/tac/tests/test_audit_contract.py -q` passed: `16 passed`.
- `.venv/bin/python -m ruff check tools/dispatch_dryrun_pr106_sidechannels.py
  tools/audit_engineered_corrections.py
  src/tac/tests/test_dispatch_dryrun_pr106_sidechannels.py
  src/tac/tests/test_engineered_correction_readiness.py` passed.

## 2026-05-05 Codex Tooling Duplication Inventory

Added an advisory consolidation inventory so duplicate audit/preflight helper
patterns remain visible while they are migrated incrementally.

Changes:

- Added `tools/audit_tooling_consolidation.py`.
- Added `src/tac/tests/test_audit_tooling_consolidation.py`.
- Wired the inventory into `tools/all_lanes_preflight.py` as `Gate #6`.
- Wrote the current JSON artifact to
  `.omx/research/artifacts/recovery_quarantine_signal_loss_20260505/tooling_consolidation_inventory_20260505.json`.

Current measured duplicate-pattern counts across local Python tooling:

- local SHA helpers: `67`
- local JSON dumps: `145`
- manual `sys.path` bootstraps: `282`
- manual repo-root parent probes: `338`
- manual score/dispatch metadata mentions: `356`

Interpretation: this confirms the user's concern. The right migration path is
incremental canonicalization around `tac.audit_contract`, `tac.repo_io`, and a
future bootstrap helper, not a broad style-only rewrite of vendored public
submission trees.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_audit_tooling_consolidation.py
  -q` passed: `2 passed`.
- `.venv/bin/python -m ruff check tools/audit_tooling_consolidation.py
  src/tac/tests/test_audit_tooling_consolidation.py` passed.

## 2026-05-05 Codex Tool Bootstrap Consolidation

Added a canonical bootstrap helper for root-level tools and migrated active
read-only/preflight tools away from local `sys.path` setup.

Changes:

- Added `tools/tool_bootstrap.py` with `repo_root_from_tool()` and
  `ensure_repo_imports()`.
- Migrated:
  - `tools/audit_engineered_corrections.py`
  - `tools/audit_research_state_tracking.py`
  - `tools/audit_silent_defaults.py`
  - `tools/audit_archive.py`
  - `tools/list_hidden_gems.py`
  - `tools/predicted_vs_actual_reconciler.py`
- Kept file-path import compatibility for tests that load tools directly with
  `importlib.util.spec_from_file_location`.
- `tools/audit_silent_defaults.py` now explicitly lists canonical scanned
  training entrypoints in its report header, so the report remains auditable
  even when current classifier output has no critical rows for that file.

Measured inventory deltas:

- `tooling_consolidation_inventory_20260505.json`: manual sys.path bootstraps
  `282`, manual repo-root parent probes `338`.
- `tooling_consolidation_inventory_20260505_bootstrap_r1.json`: manual
  sys.path bootstraps `277`, manual repo-root parent probes `336`.
- `tooling_consolidation_inventory_20260505_bootstrap_r2.json`: manual
  sys.path bootstraps `277`, manual repo-root parent probes `334`.

Verification:

- `.venv/bin/python -m ruff check tools/tool_bootstrap.py
  tools/audit_engineered_corrections.py tools/audit_research_state_tracking.py
  tools/audit_silent_defaults.py tools/audit_archive.py` passed.
- `.venv/bin/python -m pytest src/tac/tests/test_engineered_correction_readiness.py
  src/tac/tests/test_audit_archive.py src/tac/tests/test_silent_defaults_audit.py
  tests/test_comma_lab_research_state.py -q` passed: `30 passed`.
- `.venv/bin/python -m ruff check tools/list_hidden_gems.py
  tools/predicted_vs_actual_reconciler.py` passed.
- `.venv/bin/python -m pytest src/tac/tests/test_hidden_gems.py
  src/tac/tests/test_predicted_vs_actual_reconciler.py -q` passed:
  `9 passed`.

## 2026-05-05 Codex Audit Consolidation R3

Extended the shared helper migration to the highest-risk remaining audit
surfaces: dispatch shell/CLI hazards, quarantine recovery triage, and the
curated reverse-engineering tree audit.

Changes:

- Migrated `tools/check_dispatch_cli_shell_hazards.py` to
  `tools/tool_bootstrap.py` and `tac.repo_io.json_text/repo_relative`.
- Migrated `tools/recovery_quarantine_audit.py` to
  `tools/tool_bootstrap.py` and `tac.repo_io.json_text/sha256_file`.
- Migrated `tools/audit_reverse_engineering_tree.py` to the shared bootstrap
  helper.
- Migrated `src/comma_lab/reverse_engineering.py` to
  `tac.repo_io.json_text/repo_relative/sha256_file`.
- Wrote the updated advisory inventory to
  `.omx/research/artifacts/recovery_quarantine_signal_loss_20260505/tooling_consolidation_inventory_20260505_bootstrap_r3.json`.

Measured inventory after R3:

- local SHA helpers: `65`
- local JSON dumps: `142`
- manual `sys.path` bootstraps: `282`
- manual repo-root parent probes: `337`
- manual score/dispatch metadata mentions: `356`

Interpretation: R3 reduced local SHA and JSON helper duplication in the touched
audits. The bootstrap/repo-root counts remain a large measured backlog because
the inventory spans active `tools/`, `scripts/`, `experiments/`, `src/tac/`, and
`src/comma_lab/`; continue migrating active dispatch and experiment entrypoints
incrementally rather than normalizing vendored/public-runtime forensics.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_dispatch_cli_shell_hazards.py
  src/tac/tests/test_recovery_quarantine_audit.py
  tests/test_comma_lab_reverse_engineering.py -q` passed: `14 passed`.
- `.venv/bin/python -m ruff check tools/check_dispatch_cli_shell_hazards.py
  tools/recovery_quarantine_audit.py tools/audit_reverse_engineering_tree.py
  src/comma_lab/reverse_engineering.py` passed.

Final gate refresh after R3:

- `.venv/bin/python tools/all_lanes_preflight.py` passed:
  `ALL 10 PREFLIGHT CHECKS PASSED`.
- Final preflight inventory view scanned `1015` files and reported local SHA
  helpers `65`, local JSON dumps `142`, manual `sys.path` bootstraps `284`,
  manual repo-root parent probes `338`, and manual score/dispatch metadata
  mentions `356`.
- `.venv/bin/python -m pytest src/tac/tests/test_audit_tooling_consolidation.py
  src/tac/tests/test_dispatch_dryrun_pr106_sidechannels.py
  src/tac/tests/test_engineered_correction_readiness.py
  src/tac/tests/test_audit_contract.py src/tac/tests/test_repo_io.py
  src/tac/tests/test_audit_hnerv_frontier_scorecard.py
  src/tac/tests/test_hidden_gem_readiness.py src/tac/tests/test_hidden_gems.py
  src/tac/tests/test_predicted_vs_actual_reconciler.py
  src/tac/tests/test_audit_archive.py src/tac/tests/test_silent_defaults_audit.py
  src/tac/tests/test_dispatch_cli_shell_hazards.py
  src/tac/tests/test_recovery_quarantine_audit.py
  tests/test_comma_lab_research_state.py
  tests/test_comma_lab_reverse_engineering.py -q` passed: `74 passed`.
- Full touched-surface `ruff check` and `git diff --check` passed.

## 2026-05-05 Codex Dispatch/Builder Consolidation R4-R5

Extended consolidation into active dispatch and PR106 sidechannel builder
surfaces. These files are close to GPU-spend decisions and therefore should
share the same path, SHA, and JSON helpers as the audits.

Changes:

- Migrated dispatch/readiness tools:
  - `tools/dispatch_dryrun_apogee_intN.py`
  - `tools/dispatch_dryrun_omega_w_v3.py`
  - `tools/lightning_dispatch_pr106_stack.py`
  - `tools/predispatch_sanity.py`
- Added `tac.repo_io.json_line()` for deterministic JSONL override records.
- Preserved `tools/lightning_dispatch_pr106_stack.py::_sha256_file` as a
  backward-compatible alias to `tac.repo_io.sha256_file` because existing tests
  and helper callers referenced it directly.
- Migrated PR106 sidechannel builders to shared bootstrap/JSON helpers:
  - `experiments/build_pr106_yshift_sidechannel.py`
  - `experiments/build_pr106_lrl1_sidechannel.py`
  - `experiments/build_pr106_stacked.py`

Measured inventory after R4:

- local SHA helpers: `63`
- local JSON dumps: `142`
- manual `sys.path` bootstraps: `280`
- manual repo-root parent probes: `334`
- manual score/dispatch metadata mentions: `356`

Measured inventory after R5b:

- local SHA helpers: `63`
- local JSON dumps: `142`
- manual `sys.path` bootstraps: `280`
- manual repo-root parent probes: `331`
- manual score/dispatch metadata mentions: `356`

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_repo_io.py
  src/tac/tests/test_dispatch_cli_shell_hazards.py
  src/tac/tests/test_predispatch_sanity.py
  src/tac/tests/test_dispatch_dryrun_omega_w_v3.py
  src/tac/tests/test_apogee_intN_pareto_dispatch_wiring.py
  src/tac/tests/test_lightning_dispatch_pr106_stack.py -q` passed:
  `47 passed`.
- `.venv/bin/python -m pytest
  src/tac/tests/test_dispatch_dryrun_pr106_sidechannels.py -q` passed:
  `7 passed`.
- R5 full preflight refresh passed: `ALL 10 PREFLIGHT CHECKS PASSED`.

Final verification after the R4-R5 cleanup:

- Combined touched regression bundle passed: `106 passed`.
- Full touched-surface `ruff check` passed.
- `git diff --check` on touched paths passed.
- Final preflight remained green: `ALL 10 PREFLIGHT CHECKS PASSED`, with the
  inventory at local SHA helpers `63`, local JSON dumps `142`, manual
  `sys.path` bootstraps `280`, manual repo-root parent probes `331`, and manual
  score/dispatch metadata mentions `356`.

## 2026-05-05 Codex Submission Gate Consolidation R6

Hardened the final submission/compliance surface and kept the historical
Lightning supply-chain scanner narrow. The supply-chain scanner remains a
guardrail for the known dependency incident only; the contest work remains
archive custody, exact-eval evidence, report hygiene, and submission
correctness.

Changes:

- Migrated `scripts/pre_submission_compliance_check.py` to shared bootstrap,
  repo-relative paths, canonical JSON, JSON loading, and `tac.repo_io.sha256_file`.
- Migrated `scripts/scan_lightning_supply_chain.py` to shared bootstrap and
  canonical JSON output without broadening its mission.

Measured inventory after R6:

- local SHA helpers: `62`
- local JSON dumps: `139`
- manual `sys.path` bootstraps: `278`
- manual repo-root parent probes: `330`
- manual score/dispatch metadata mentions: `356`

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_lightning_supply_chain_scan.py
  src/tac/tests/test_pre_submission_compliance_check.py -q` passed:
  `11 passed`.
- `.venv/bin/python -m ruff check scripts/scan_lightning_supply_chain.py
  scripts/pre_submission_compliance_check.py
  src/tac/tests/test_lightning_supply_chain_scan.py
  src/tac/tests/test_pre_submission_compliance_check.py` passed.
- Full preflight refresh passed: `ALL 10 PREFLIGHT CHECKS PASSED`.

## 2026-05-05 Codex Lightning Core Consolidation R7

Moved into the central Lightning exact-eval staging/dispatch machinery in a
small verified slice.

Changes:

- Migrated `scripts/lightning_repro_workspace.py` to shared bootstrap,
  `tac.repo_io.sha256_file`, `tac.repo_io.write_json`, and canonical JSON
  printing for local diagnostics/final status. Remote embedded Python snippets
  were intentionally left unchanged because they execute in the staged Studio
  environment and are part of the remote command payload.
- Migrated selected high-traffic JSON output/write sites in
  `scripts/launch_lightning_batch_job.py` to shared `_print_json()` and
  `_write_json()` wrappers backed by `tac.repo_io.json_text/write_json`.
- Left existing command semantics intact: exact-eval, component-response,
  component-sensitivity, local/SSH harvest, artifact validation, and refresh
  status behavior remain covered by the existing tests.

Measured inventory after R7:

- local SHA helpers: `61`
- local JSON dumps: `125`
- manual `sys.path` bootstraps: `276`
- manual repo-root parent probes: `330`
- manual score/dispatch metadata mentions: `356`

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_lightning_repro_workspace.py
  src/tac/tests/test_lightning_batch_jobs.py -q` passed: `141 passed`.
- `.venv/bin/python -m ruff check scripts/lightning_repro_workspace.py
  scripts/launch_lightning_batch_job.py` passed.
- Full preflight refresh passed: `ALL 10 PREFLIGHT CHECKS PASSED`.

## 2026-05-05 Codex Legacy Lightning Lane Launcher R8

Hardened the older Lightning lane launcher wrapper without changing its
operator-facing command behavior.

Changes:

- Migrated `scripts/launch_lane_lightning.py` to shared bootstrap and
  `tac.repo_io.json_text`.
- Replaced local `json.dumps(..., indent=2)` command output with `_print_json()`
  so dispatch/status/harvest/teardown/list/probe all use canonical JSON output.

Measured inventory after R8:

- local SHA helpers: `61`
- local JSON dumps: `125`
- manual `sys.path` bootstraps: `275`
- manual repo-root parent probes: `330`
- manual score/dispatch metadata mentions: `356`

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_lightning_dispatch.py -q`
  passed: `16 passed`.
- `.venv/bin/python -m ruff check scripts/launch_lane_lightning.py` passed.
- Full preflight refresh passed: `ALL 10 PREFLIGHT CHECKS PASSED`.

## 2026-05-05 Codex Score-Custody Wrapper Consolidation R9

Moved the exact adjudication and exact-eval repro wrappers onto the shared
repo/tooling contract. This targets the recurring bug class where score-custody
scripts each hand-rolled repo-root discovery, SHA helpers, and JSON emission.

Changes:

- Migrated `scripts/adjudicate_contest_auth_eval.py` to shared bootstrap,
  `tac.repo_io.read_json`, `tac.repo_io.json_text`, and
  `tac.repo_io.sha256_file`.
- Preserved adjudicator semantics: exact archive bytes/SHA validation,
  CUDA/sample gates, T4/equivalent promotion gate, component gates,
  distillation-policy gate, and forensic-success flags remain unchanged.
- Migrated `scripts/lightning_exact_eval_repro.py` to shared bootstrap,
  `tac.repo_io.sha256_file`, `tac.repo_io.read_json`, `tac.repo_io.write_json`,
  and canonical JSON status output.
- Preserved strict parser diagnostics, including the stale `--rmote` guard, and
  kept queue/stage/submit command construction unchanged.

Measured inventory after R9:

- local SHA helpers: `59`
- local JSON dumps: `122`
- manual `sys.path` bootstraps: `275`
- manual repo-root parent probes: `330`
- manual score/dispatch metadata mentions: `356`

Generated inventory artifact:

- `.omx/state/tooling_consolidation_inventory_20260505_score_custody_r9.json`
  (local generated state; ignored by current public-release hygiene rules).

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_lightning_exact_eval_repro.py
  src/tac/tests/test_remote_auth_eval_hardening.py` passed: `53 passed`.
- `.venv/bin/python -m ruff check scripts/adjudicate_contest_auth_eval.py
  scripts/lightning_exact_eval_repro.py` passed.
- Full preflight refresh passed: `ALL 10 PREFLIGHT CHECKS PASSED`.

## 2026-05-05 Codex Bundle And Component-Response Consolidation R10

Extended the consolidation pass into the deploy-bundle builder, the Alpha-Geo
Lightning launcher, and the official component-response plan builders. This is
still harness work, not score evidence: it removes duplicated helper code from
tools that package evidence, queue exact CUDA work, and prepare official
response archives.

Changes:

- Migrated `scripts/build_pfp16_a_plus_plus_bundle.py` to shared bootstrap,
  `tac.repo_io.sha256_file`, `tac.repo_io.read_json`, and canonical JSON writes.
- Migrated `scripts/launch_lightning_alpha_geo0_pose_regen.py` to shared
  bootstrap, canonical final JSON output, and `tac.repo_io.sha256_file` for
  local artifact identities. The dispatch-claim gate and remote preflight order
  were preserved.
- Migrated `experiments/build_component_response_perturbation_plan.py`,
  `experiments/build_component_response_prediction_deltas.py`, and
  `experiments/build_component_response_plan_from_sensitivity_artifacts.py` to
  shared bootstrap and JSON/SHA file helpers.
- Fixed a live SJ-KL custody guard regression in
  `scripts/remote_lane_sjkl_c067.sh`: the delegated archive-only eval now keeps
  `eval_work` by default and records the delegated `--keep-work-dir --work-dir`
  exact-eval custody contract in executable script text.
- Preserved local canonical hash routines used for atom-set and prediction
  identity. Those hashes are scientific identifiers, so this pass did not alter
  their serialization contract.
- Left embedded Lightning remote Python snippets unchanged because they execute
  inside the remote payload rather than as repo-local helper code.

Measured inventory after R10:

- local SHA helpers: `54`
- local JSON dumps: `113`
- manual `sys.path` bootstraps: `272`
- manual repo-root parent probes: `330`
- manual score/dispatch metadata mentions: `356`

Generated inventory artifact:

- `.omx/state/tooling_consolidation_inventory_20260505_bundle_component_r10.json`
  (local generated state; ignored by current public-release hygiene rules).

Verification:

- `.venv/bin/python -m pytest
  src/tac/tests/test_build_component_response_perturbation_plan.py
  src/tac/tests/test_launch_lightning_alpha_geo0_pose_regen.py` passed:
  `19 passed`.
- Expanded focused regression after the SJ-KL fix passed:
  `.venv/bin/python -m pytest ...` on audit/repo IO, Lightning custody,
  remote auth-eval hardening, submission gates, Alpha-Geo, and
  component-response tests: `246 passed`.
- `.venv/bin/python -m ruff check
  experiments/build_component_response_perturbation_plan.py
  experiments/build_component_response_plan_from_sensitivity_artifacts.py
  experiments/build_component_response_prediction_deltas.py
  scripts/build_pfp16_a_plus_plus_bundle.py
  scripts/launch_lightning_alpha_geo0_pose_regen.py` passed.
- `.venv/bin/python -m py_compile` on the same files passed.
- Full preflight refresh passed: `ALL 10 PREFLIGHT CHECKS PASSED`.

## 2026-05-05 Codex Recovered Remote Lane Canonicalization R11

Promoted the recovered remote lane scripts from ad hoc workspace signal into a
canonical OSS-hardened preflight surface. This is harness/custody work, not
score evidence: it makes recovered lanes visible to operators and forces their
classification before any future remote dispatch.

Changes:

- Added `tools/audit_recovered_remote_lanes.py`, a provider-state-free audit
  using the shared `tac.audit_contract.AuditReport` and `tac.repo_io` JSON
  contract.
- The audit checks all recovered remote lane scripts for presence, `bash -n`
  syntax, and fail-closed custody/proxy markers:
  - `scripts/remote_lane_sjkl_c067.sh` is classified as
    `canonical_exact_eval_delegated`.
  - `scripts/remote_lane_pr79_segaction_search.sh` is classified as
    `proxy_search_only_until_exact_eval`.
  - `scripts/remote_lane_q_faithful_jointgen.sh` is classified as
    `legacy_recovered_exact_eval_runtime`.
- Added `src/tac/tests/test_audit_recovered_remote_lanes.py` with synthetic
  pass/fail contract coverage plus live CLI JSON contract coverage.
- Wired the audit into `tools/all_lanes_preflight.py` as Gate #7:
  recovered remote lane canonicalization.
- Kept `docs/runbooks/recovered_remote_lanes.md` as the operator-facing
  classification/runbook surface for these recovered scripts.

Verification:

- `.venv/bin/python -m pytest
  src/tac/tests/test_audit_recovered_remote_lanes.py
  src/tac/tests/test_recovered_remote_lane_scripts.py
  src/tac/tests/test_remote_auth_eval_hardening.py::test_live_remote_lane_scripts_avoid_fragile_auth_eval_parsers
  -q` passed: `7 passed`.
- `bash -n scripts/remote_lane_pr79_segaction_search.sh
  scripts/remote_lane_q_faithful_jointgen.sh
  scripts/remote_lane_sjkl_c067.sh` passed.
- `.venv/bin/python tools/audit_recovered_remote_lanes.py` passed:
  `recovered remote lane canonicalization: PASS (3 recovered scripts checked)`.
- `.venv/bin/python -m ruff check
  tools/audit_recovered_remote_lanes.py tools/all_lanes_preflight.py
  src/tac/tests/test_audit_recovered_remote_lanes.py
  src/tac/tests/test_recovered_remote_lane_scripts.py` passed.
- `git diff --check` on the touched recovered-lane canonicalization surfaces
  passed.
- Full preflight refresh passed: `ALL 11 PREFLIGHT CHECKS PASSED`.

### R11 adversarial bug-hunter fixes

Two xhigh read-only bug-hunter agents reviewed the recovered-lane and OSS
hardening surface. Their highest-signal findings were converted into permanent
guards where the fix was small and low-risk in this tranche.

Fixed now:

- `scripts/remote_lane_sjkl_c067.sh` now tells operators to claim the exact
  provenance lane id `lane_sjkl_c067`, not the stale shorthand `sjkl_c067`.
- `tools/audit_recovered_remote_lanes.py` now checks that SJ-KL script comments,
  provenance markers, and the runbook-visible lane id agree.
- `scripts/remote_lane_pr79_segaction_search.sh` now pins the public source
  commit (`9c93af0a5bf55cc8a03716e0f7b9babf187ad2a1`), checks out that commit
  detached after clone, records the cloned commit, pins `brotli==1.1.0`, and
  records a patch diff SHA-256 in `public_repo_lock.json`.
- `tools/audit_recovered_remote_lanes.py` now fails PR79-style public clones
  that are branch-only without a detached commit pin.
- `src/tac/audit_contract.py` now rejects metadata that tries to override
  canonical safety fields such as `score_claim`, `dispatch_attempted`,
  `blockers`, `summary`, or the readiness key. This closes a schema poisoning
  bug class in local audit tools.
- `tools/audit_tooling_consolidation.py`,
  `tools/audit_recovered_remote_lanes.py`, and
  `tools/audit_engineered_corrections.py` were adjusted to use that stricter
  audit contract without losing their top-level safety fields.
- `experiments/preflight_candidate_manifest_dispatch_readiness.py` now blocks
  pre-dispatch readiness if a manifest already records `score_claim=true`,
  `dispatch_performed=true`, or `remote_jobs_dispatched=true`; those belong in
  post-dispatch adjudication, not dispatch unlocks.
- `tools/dispatch_dryrun_pr106_sidechannels.py` now requires builder metadata
  to explicitly include `score_claim=false`; merely omitting `score_claim=true`
  is no longer sufficient.
- `experiments/build_pr106_latent_sidecar.py` now writes
  `score_claim=false`, `dispatch_attempted=false`, `promotion_eligible=false`,
  and `evidence_grade=empirical_build_only` in build metadata.

Still open from the adversarial review:

- Replace remaining raw `ZipFile.extractall(...)` surfaces with one canonical
  safe extractor and make strict preflight ban all raw extractors outside that
  helper.
- Broaden provider/CPU-score leakage scans into `submissions/` shell surfaces,
  especially stale local scripts that default to provider SSH hosts or
  `--device cpu`.
- Replace PR79 hard-coded byte slicing with a typed parser that validates total
  length, per-slice SHA-256, decompression success, and decoded magic before
  candidate archive emission.
- Add a read-only dispatch-claim check mode so no tool can print a remote
  dispatch-ready state without either an active matching claim or an explicit
  `claim_required=true` fail-closed marker.
- Add a live untracked-source audit for `tools/`, `scripts/`, `src/`,
  `experiments/`, and docs so promoted source cannot remain invisible to git or
  a dated ledger.

Additional verification after the adversarial fixes:

- `.venv/bin/python -m pytest
  src/tac/tests/test_audit_contract.py
  src/tac/tests/test_audit_tooling_consolidation.py
  src/tac/tests/test_audit_recovered_remote_lanes.py
  src/tac/tests/test_recovered_remote_lane_scripts.py
  src/tac/tests/test_preflight_candidate_manifest_dispatch_readiness.py
  src/tac/tests/test_dispatch_dryrun_pr106_sidechannels.py -q` passed:
  `29 passed`.
- `.venv/bin/python tools/audit_recovered_remote_lanes.py --format json`
  passed with all three recovered scripts present, syntax-clean, and not
  branch-only public clones.
- `.venv/bin/python tools/audit_engineered_corrections.py --self-test
  --max-packed-bytes 10000 --fail-if-not-ready` passed and preserved
  `score_claim=false`, `dispatch_attempted=false`, and
  `ready_for_exact_eval_dispatch=false`.
- `.venv/bin/python -m py_compile` passed for the touched audit, preflight,
  sidechannel, and recovered-lane files.
- `.venv/bin/python -m ruff check` passed on the touched audit/preflight/test
  files, excluding `experiments/build_pr106_latent_sidecar.py` because that
  recovered script still has pre-existing import-order/noqa cleanup debt outside
  this metadata fix.
- Full preflight refresh remained green: `ALL 11 PREFLIGHT CHECKS PASSED`.

## 2026-05-05 Codex Canonical ZIP Safety R12

Closed the P0 split-ZIP-safety bug class identified by the adversarial
bug-hunter review. Raw `ZipFile.extractall(...)` is no longer allowed in live
archive/scoring paths outside the canonical helper.

Changes:

- Added `validate_zip_member_infos(...)` and `safe_extract_zip(...)` to
  `src/tac/submission_archive.py`.
- The canonical extractor validates the entire archive before writing files:
  duplicate member names, zip-slip paths, absolute paths, hidden/resource-fork
  members, and symlink entries fail closed.
- Replaced raw extraction in:
  - `src/tac/archive_optimizer.py`
  - `src/tac/data.py`
  - `src/tac/lossless/evaluate.py`
  - `src/tac/proxy_eval.py`
  - `scripts/compress_archive.py`
  - `submissions/robust_current/runner.py`
  - `experiments/contest_eval.py`
  - `experiments/build_clean_source_stc_archive.py`
  - `experiments/build_lane_al_archive.py`
  - `experiments/build_lane_mm_archive.py`
  - `experiments/build_lane_stc_archive.py`
  - `experiments/optimize_grayscale_canvas.py`
- Hardened `src/tac/archive_optimizer.py` so zip metadata stripping validates
  input member metadata and writes deterministic output members through
  `write_deterministic_zip_member(...)`.
- Hardened `tools/audit_archive.py` so duplicate member names and invalid
  member names are explicit failures instead of disappearing through a
  name-to-size dictionary collapse.
- Added `check_no_raw_zip_extractall(...)` to `src/tac/preflight.py` and wired
  it into `preflight_all()` immediately after deterministic-ZIP checking.
- Added `src/tac/tests/test_submission_archive_safety.py` covering valid
  extraction, zip-slip/absolute/hidden/resource-fork names, duplicate names,
  and symlink members.
- Updated preflight fixture tests so intentional raw-extractor examples are
  generated dynamically and do not poison the live-tree scanner.

Verification:

- `.venv/bin/python -m pytest
  src/tac/tests/test_submission_archive_safety.py
  src/tac/tests/test_audit_archive.py
  src/tac/tests/test_preflight_meta_bugs.py::TestArchiveBuildersUseDeterministicZip
  src/tac/tests/test_preflight_meta_bugs.py::TestPreflightAllInvokesMetaBugChecks
  src/tac/tests/test_remote_lane_j_nwc_hardening.py -q` passed:
  `46 passed, 1 warning` where the warning is Python's expected duplicate-ZIP
  construction warning inside the duplicate-member regression test.
- Direct `check_no_raw_zip_extractall(strict=False, verbose=False)` returned
  `0` violations on the live tree.
- `.venv/bin/python -m py_compile` passed for the touched archive,
  extractor, preflight, runner, and experiment surfaces.
- `.venv/bin/python -m ruff check src/tac/submission_archive.py
  src/tac/archive_optimizer.py tools/audit_archive.py src/tac/data.py
  src/tac/tests/test_submission_archive_safety.py` passed.
- A broader ruff run over legacy scripts still exposes pre-existing style debt
  unrelated to this ZIP-safety fix; it was not treated as a blocker for R12.
- Full all-lanes preflight remained green: `ALL 11 PREFLIGHT CHECKS PASSED`.

Remaining from the xhigh review after R12:

- Broaden provider/CPU-score leakage scans into `submissions/` shell/Python
  surfaces.
- Replace PR79 hard-coded byte slicing with a typed parser that validates total
  length, per-slice SHA-256, decompression success, and decoded magic.
- Add live untracked-source audit for source/test/tool/doc artifacts.
- Add read-only lane-claim check mode and require it before any remote tool can
  print a dispatch-ready state.
- Decide whether `tools/audit_tooling_consolidation.py` stays explicitly
  advisory in preflight output or gains a strict high-severity threshold mode.

## 2026-05-05 Codex Submission Provider/CPU Leakage R13

Closed the stale public-helper bug class found in
`submissions/robust_current/download_and_eval.sh`: provider-specific download
defaults, disabled SSH host-key custody, and CPU scoring in a submission-facing
helper.

Changes:

- Hardened `submissions/robust_current/download_and_eval.sh`:
  - requires `LIGHTNING_TARGET`/operator remote target instead of embedding a
    provider hostname;
  - uses `BatchMode=yes` and `StrictHostKeyChecking=accept-new`;
  - changes manual score instructions to `--device cuda`;
  - makes actual scoring CUDA-only and exits with `--skip-score` guidance for
    local package-only checks.
- Added `check_no_submission_provider_or_cpu_score_leakage(...)` to
  `src/tac/preflight.py` and wired it into `preflight_all()` immediately after
  the Lightning SSH static policy guard.
- The scanner covers public/submission helper surfaces under
  `submissions/robust_current` and `submissions/pr106_stacked`; it blocks:
  provider hostnames, disabled host-key checking, `/dev/null` known-hosts, and
  score-path commands that invoke `runner.py evaluate`, `contest_auth_eval.py`,
  or `evaluate.py` with `--device cpu`/`--device mps`.
- Added focused RED/GREEN coverage in
  `src/tac/tests/test_preflight_meta_bugs.py::TestSubmissionProviderCpuScoreLeakage`.

Verification:

- `.venv/bin/python -m pytest
  src/tac/tests/test_preflight_meta_bugs.py::TestSubmissionProviderCpuScoreLeakage
  -q` passed: `2 passed`.
- `.venv/bin/python -m pytest
  src/tac/tests/test_preflight_meta_bugs.py::TestLightningSshStaticPolicy
  src/tac/tests/test_preflight_meta_bugs.py::TestSubmissionProviderCpuScoreLeakage
  -q` passed: `9 passed`.
- `bash -n submissions/robust_current/download_and_eval.sh` passed.
- `.venv/bin/python -m py_compile src/tac/preflight.py` passed.
- Direct live-tree scanner call reported
  `[submission-provider-score-leakage] OK: 21 submission helper(s) scanned`.
- `git diff --check -- src/tac/preflight.py
  src/tac/tests/test_preflight_meta_bugs.py
  submissions/robust_current/download_and_eval.sh` passed.
- `tools/all_lanes_preflight.py` remained green:
  `ALL 11 PREFLIGHT CHECKS PASSED`.

Full `preflight_all()` is still blocked by a separate pre-existing memory-file
review issue:

- `feedback_grand_council_predictor_calibration_no_arbitrariness_20260505.md`
  is missing the Grand Council, internal-consistency, and reactivation-criteria
  sections required by `check_kill_memory_files_have_council_review(...)`.

This file is outside the R13 code patch and outside Codex memory-write
permissions, so it remains a separately queued cleanup item.

## 2026-05-05 Codex PR79 Typed Payload Parser R14

Closed the PR79 hard-coded slice class in the recovered SegAction search
driver. The lane no longer builds probe/optimized archives by unchecked
`MASK_BYTES`/`MODEL_BYTES`/`POSE_BYTES` slicing.

Changes:

- Added `src/tac/pr79_segaction_payload.py` as the canonical typed parser for
  PR79/qpose14-r55 SegAction single-member `p` payloads.
- The parser supports both `P3` headered payloads and legacy headerless
  payloads, but legacy inference is validated by:
  - total payload length;
  - per-slice Brotli decompression;
  - model raw magic (`QZS3`, `QZS2`, `QZS1`, `QZC1`, `QZC2`, `QZC3`);
  - SegAction raw magic/record structure (`SG2`, `TG1`, or 4/5-byte records);
  - pose raw magic/shape (`QP1` or `uint16[*,6]`);
  - compressed and raw SHA-256 summaries for every part.
- Added deterministic `write_pr79_single_member_archive(...)`, which validates
  the candidate payload and writes only a fixed-metadata ZIP member named `p`.
- Updated `scripts/remote_lane_pr79_segaction_search.sh` so:
  - `PYTHONPATH` points at this checkout's `src`;
  - cloned public PR79 scripts are patched to import
    `parse_pr79_archive`, `parse_pr79_payload_bytes`, and
    `write_pr79_single_member_archive`;
  - public `read_packed_archive(...)` and `split_known_payload(...)` delegate
    to the typed parser;
  - proxy score payloads and final optimized payloads use
    `source_payload.replace_actions(...)` rather than raw concatenation;
  - probe and optimized archives are rebuilt through the deterministic typed
    archive writer;
  - replacement-patch targets fail closed if the upstream source shape changes.
- Strengthened `tools/audit_recovered_remote_lanes.py` and
  `src/tac/tests/test_recovered_remote_lane_scripts.py` so recovered PR79 lane
  visibility requires the typed parser/archive writer markers.
- Added `src/tac/tests/test_pr79_segaction_payload.py` for P3 parsing, legacy
  validated parsing, bad-model rejection, action replacement, and deterministic
  single-member archive writing.

Verification:

- `bash -n scripts/remote_lane_pr79_segaction_search.sh` passed.
- `.venv/bin/python -m pytest
  src/tac/tests/test_pr79_segaction_payload.py
  src/tac/tests/test_recovered_remote_lane_scripts.py -q` passed:
  `8 passed`.
- `.venv/bin/python -m py_compile
  src/tac/pr79_segaction_payload.py
  tools/audit_recovered_remote_lanes.py` passed.
- `.venv/bin/python tools/audit_recovered_remote_lanes.py --format json
  --strict` passed with zero blockers.
- `.venv/bin/python -m ruff check
  src/tac/pr79_segaction_payload.py
  src/tac/tests/test_pr79_segaction_payload.py
  tools/audit_recovered_remote_lanes.py
  src/tac/tests/test_recovered_remote_lane_scripts.py` passed.
- Applied the remote script's public-source patch logic to a temporary copy of
  the PR79 public scripts and `py_compile` passed for both patched scripts.
- `git diff --check -- src/tac/pr79_segaction_payload.py
  src/tac/tests/test_pr79_segaction_payload.py
  scripts/remote_lane_pr79_segaction_search.sh
  tools/audit_recovered_remote_lanes.py
  src/tac/tests/test_recovered_remote_lane_scripts.py` passed.
- `tools/all_lanes_preflight.py` remained green:
  `ALL 11 PREFLIGHT CHECKS PASSED`.

Important custody note:

- The PR79 recovered lane and its recovered-lane audit/test files are still
  untracked in git as of this tranche. That is not a correctness failure for
  the patch, but it reinforces the next queued item: live untracked-source
  audit and canonical tracking/ignore policy.

## 2026-05-05 Codex Untracked Source Audit R15

Added a live no-signal-loss audit for source-like untracked files. This does
not decide whether each file should be tracked, moved, or ignored; it makes the
inventory executable so source recovery cannot silently disappear behind
provider artifacts and rebuildable experiment outputs.

Changes:

- Added `tools/audit_untracked_source_artifacts.py`.
- The audit parses `git status --porcelain=v1 --untracked-files=all` and
  classifies untracked source/research paths under:
  `.omx/research`, `docs`, `experiments`, `reverse_engineering`, `scripts`,
  `src`, `submissions`, `tests`, and `tools`.
- It ignores rebuildable/generated custody surfaces by default:
  `.omx/research/artifacts`, `.omx/state`, `experiments/results`,
  `reports/raw`, and `reports/private`.
- It emits an `AuditReport` JSON contract with:
  `score_claim=false`, `dispatch_attempted=false`,
  `ready_for_no_signal_loss_canonicalization`, blockers, by-class counts, and a
  50-item sample.
- Strict mode is available for final cleanup, but all-lanes preflight wires it
  as advisory for now because the live tree intentionally contains recovered
  source pending canonical tracking/disposition.
- Added `src/tac/tests/test_audit_untracked_source_artifacts.py`.
- Wired it into `tools/all_lanes_preflight.py` as Gate #8.

Current live signal:

- Gate #8 reports `85` source-like untracked paths:
  `1` research, `2` reverse-engineering, and `82` source/tool/test/doc paths.
- This includes the recovered-lane scripts/tools/tests and several hidden-gem
  recovery modules. They should be dispositioned next as tracked canonical
  source, moved to a documented recovery queue, or explicitly ignored if
  rebuildable/private.

Verification:

- `.venv/bin/python -m pytest
  src/tac/tests/test_audit_untracked_source_artifacts.py -q` passed:
  `3 passed`.
- `.venv/bin/python -m ruff check
  tools/audit_untracked_source_artifacts.py
  src/tac/tests/test_audit_untracked_source_artifacts.py` passed.
- `.venv/bin/python -m py_compile
  tools/all_lanes_preflight.py
  tools/audit_untracked_source_artifacts.py` passed.
- `.venv/bin/python tools/all_lanes_preflight.py` remained green:
  `ALL 12 PREFLIGHT CHECKS PASSED`, with Gate #8 explicitly marked
  `ADVISORY`.

## 2026-05-05 Codex Untracked Source Disposition Template R16

Extended the untracked-source audit so the 85-file inventory can be burned down
deterministically instead of rediscovered from raw `git status`.

Changes:

- `tools/audit_untracked_source_artifacts.py` now supports disposition
  manifests with entries:
  - `path`
  - `disposition`: one of `track`, `recovery_queue`, `ignore_private`,
    `ignore_rebuildable`
  - nonempty `note`
- Added `--write-disposition-template PATH` to emit a review template from the
  live untracked-source inventory.
- Added `--disposition-manifest PATH` so strict mode can distinguish
  undispositioned source loss from files that have a documented decision.
- The audit summary now reports:
  `dispositioned_count`, `undispositioned_count`,
  `invalid_disposition_count`, and `by_disposition`.
- Added regression coverage for disposition-manifest parsing.

Generated artifact:

- `.omx/research/artifacts/recovery_quarantine_signal_loss_20260505/untracked_source_disposition_template_20260505.json`

Current result with the generated template:

- `85` source-like untracked paths are dispositioned as `track` in the template.
- `undispositioned_count=0`
- `invalid_disposition_count=0`
- strict audit with the template reports
  `ready_for_no_signal_loss_canonicalization=true`

Important caveat:

- The template is a review artifact, not a substitute for tracking. Files marked
  `track` are still untracked until the operator or a future tranche stages and
  commits them, or changes their disposition to `recovery_queue`,
  `ignore_private`, or `ignore_rebuildable` with rationale.

Verification:

- `.venv/bin/python -m pytest
  src/tac/tests/test_audit_untracked_source_artifacts.py -q` passed:
  `4 passed`.
- `.venv/bin/python -m ruff check
  tools/audit_untracked_source_artifacts.py
  src/tac/tests/test_audit_untracked_source_artifacts.py` passed.
- `.venv/bin/python tools/audit_untracked_source_artifacts.py
  --disposition-manifest
  .omx/research/artifacts/recovery_quarantine_signal_loss_20260505/untracked_source_disposition_template_20260505.json
  --strict` passed.

## 2026-05-05 Codex Reviewed Untracked Source Disposition R17

Promoted the untracked-source review from a generated template to a durable
reviewed disposition manifest and wired all-lanes preflight to consume it
automatically.

Changes:

- Added reviewed disposition file:
  `.omx/research/untracked_source_dispositions_20260505_codex.json`.
- The manifest self-dispositions itself and marks all currently live
  source-like untracked paths as `track`, with category-specific notes for:
  research ledgers, runbooks, experiments, reverse-engineering docs, remote
  lane scripts, canonical library modules, regression tests, and audit/dispatch
  tools.
- `tools/all_lanes_preflight.py` now detects this reviewed manifest and runs
  Gate #8 in strict disposition mode:
  `tools/audit_untracked_source_artifacts.py --strict --disposition-manifest
  .omx/research/untracked_source_dispositions_20260505_codex.json`.
- If a new source-like untracked file appears without a manifest row, or a
  manifest row goes stale, Gate #8 now fails instead of silently emitting an
  advisory inventory.

Current strict signal:

- `86` source-like untracked paths are live and dispositioned.
- `undispositioned_count=0`
- `invalid_disposition_count=0`
- `by_disposition={"track": 86}`

Important caveat:

- This is canonical disposition and preflight enforcement, not git staging.
  These paths are still untracked until deliberately staged/committed or moved
  to a different reviewed disposition.

Verification:

- `.venv/bin/python tools/audit_untracked_source_artifacts.py --format json
  --disposition-manifest
  .omx/research/untracked_source_dispositions_20260505_codex.json --strict`
  passed with `ready_for_no_signal_loss_canonicalization=true`.
- `.venv/bin/python -m pytest
  src/tac/tests/test_audit_untracked_source_artifacts.py -q` passed:
  `4 passed`.
- `.venv/bin/python -m py_compile tools/all_lanes_preflight.py
  tools/audit_untracked_source_artifacts.py` passed.
- `.venv/bin/python -m ruff check tools/all_lanes_preflight.py
  tools/audit_untracked_source_artifacts.py
  src/tac/tests/test_audit_untracked_source_artifacts.py` passed.
- `.venv/bin/python tools/all_lanes_preflight.py` passed:
  `ALL 12 PREFLIGHT CHECKS PASSED`, with Gate #8 reporting
  `PASSED (STRICT DISPOSITION)`.

## 2026-05-05 Codex Shadowed Index Delete Audit R18

Extended the untracked-source audit to expose a sharper source-control
failure class: same-path tracked deletions shadowed by untracked source-like
replacement files.

Why this matters:

- A file can look present in the working tree while the git index still records
  a deletion for the same path.
- If committed in that state without a deliberate index fix, recovered source
  can disappear even though local tests and preflight saw the replacement file.
- This is a no-signal-loss bug class, especially during pyc/orphan recovery and
  cross-agent canonicalization.

Changes:

- `tools/audit_untracked_source_artifacts.py` now parses all porcelain status
  records, not just `??` rows.
- The audit reports:
  - `source_like_delete_count`
  - `shadowed_index_delete_count`
  - `shadowed_index_delete_paths`
- `tools/all_lanes_preflight.py` prints
  `shadowed_index_deletes=<n>` inside Gate #8 output.
- Added parser regression coverage for deletion statuses.

Current live signal:

- `shadowed_index_delete_count=13`.
- The shadowed paths are:
  - `scripts/wave_deploy_post_apogee_int4_sanity.sh`
  - `src/tac/predictor/__init__.py`
  - `src/tac/predictor/score_band.py`
  - `src/tac/tests/test_check_calibration_provenance.py`
  - `src/tac/tests/test_check_dispatch_wrapper_stages_implemented.py`
  - `src/tac/tests/test_check_lane_smoke_signal_nontrivial.py`
  - `src/tac/tests/test_lightning_dispatch_pr106_stack.py`
  - `src/tac/tests/test_predispatch_sanity.py`
  - `src/tac/tests/test_score_band_predictor.py`
  - `tools/check_calibration_provenance.py`
  - `tools/check_dispatch_wrapper_stages_implemented.py`
  - `tools/check_lane_smoke_signal_nontrivial.py`
  - `tools/predispatch_sanity.py`

Important caveat:

- This tranche reports the index-shadow condition but does not mutate the git
  index. The next canonicalization tranche should deliberately resolve these
  13 paths by staging the recovered replacements, unstaging accidental
  deletions, or moving the replacements to a reviewed recovery queue.

Verification:

- `.venv/bin/python tools/audit_untracked_source_artifacts.py --format json
  --disposition-manifest
  .omx/research/untracked_source_dispositions_20260505_codex.json --strict`
  reported `shadowed_index_delete_count=13`.
- `.venv/bin/python -m pytest
  src/tac/tests/test_audit_untracked_source_artifacts.py -q` passed:
  `5 passed`.
- `.venv/bin/python -m py_compile tools/all_lanes_preflight.py
  tools/audit_untracked_source_artifacts.py` passed.
- `.venv/bin/python -m ruff check tools/all_lanes_preflight.py
  tools/audit_untracked_source_artifacts.py
  src/tac/tests/test_audit_untracked_source_artifacts.py` passed.
- `.venv/bin/python tools/all_lanes_preflight.py` passed:
  `ALL 12 PREFLIGHT CHECKS PASSED`, with Gate #8 now showing
  `shadowed_index_deletes=13`.

## 2026-05-05 Codex Shadowed Path Resolution R19

Resolved the same-path shadow condition instead of leaving it as an advisory
warning.

Actions:

- Staged the exact recovered replacements for the source-like paths that were
  simultaneously recorded as deleted in the index and untracked in the working
  tree.
- Removed their stale rows from
  `.omx/research/untracked_source_dispositions_20260505_codex.json` once they
  were no longer live untracked-source paths.
- Also resolved the same failure mode for
  `reports/meta_lagrangian_apogee_smoke.json` and broadened the untracked
  source audit to cover canonical `reports/` summaries while still ignoring
  `reports/raw/` and `reports/private/`.

Result:

- `shadowed_index_delete_count=0`.
- `invalid_disposition_count=0`.
- `undispositioned_count=0`.
- Live untracked source-like inventory is now `65`, all dispositioned as
  `track`.

Resolved source-like shadow paths:

- `scripts/wave_deploy_post_apogee_int4_sanity.sh`
- `src/tac/predictor/__init__.py`
- `src/tac/predictor/score_band.py`
- `src/tac/tests/test_check_calibration_provenance.py`
- `src/tac/tests/test_check_dispatch_wrapper_stages_implemented.py`
- `src/tac/tests/test_check_lane_smoke_signal_nontrivial.py`
- `src/tac/tests/test_lightning_dispatch_pr106_stack.py`
- `src/tac/tests/test_predispatch_sanity.py`
- `src/tac/tests/test_score_band_predictor.py`
- `tools/check_calibration_provenance.py`
- `tools/check_dispatch_wrapper_stages_implemented.py`
- `tools/check_lane_smoke_signal_nontrivial.py`
- `tools/predispatch_sanity.py`
- `experiments/distortion_proxy_local.py`
- `scripts/remote_lane_pr79_segaction_search.sh`
- `scripts/remote_lane_sjkl_c067.sh`
- `src/tac/optimizer/__init__.py`
- `src/tac/optimizer/meta_lagrangian.py`
- `src/tac/tests/test_meta_lagrangian.py`
- `tools/dispatch_dryrun_pr106_sidechannels.py`
- `tools/meta_lagrangian_search_cli.py`

Resolved report-summary shadow path:

- `reports/meta_lagrangian_apogee_smoke.json`

Verification:

- `.venv/bin/python tools/audit_untracked_source_artifacts.py --format json
  --disposition-manifest
  .omx/research/untracked_source_dispositions_20260505_codex.json --strict`
  passed with `shadowed_index_delete_count=0`.
- `.venv/bin/python -m pytest
  src/tac/tests/test_audit_untracked_source_artifacts.py -q` passed:
  `5 passed`.
- `.venv/bin/python -m ruff check tools/all_lanes_preflight.py
  tools/audit_untracked_source_artifacts.py
  src/tac/tests/test_audit_untracked_source_artifacts.py` passed.
- `.venv/bin/python -m py_compile tools/all_lanes_preflight.py
  tools/audit_untracked_source_artifacts.py` passed.
- `git diff --check -- tools/all_lanes_preflight.py
  tools/audit_untracked_source_artifacts.py
  src/tac/tests/test_audit_untracked_source_artifacts.py
  .omx/research/untracked_source_dispositions_20260505_codex.json
  .omx/research/recovery_quarantine_signal_loss_triage_20260505_codex.md`
  passed.
- `.venv/bin/python tools/all_lanes_preflight.py` passed:
  `ALL 12 PREFLIGHT CHECKS PASSED`, with Gate #8 reporting
  `65 source-like untracked`, `undispositioned=0`,
  `invalid_dispositions=0`, and `shadowed_index_deletes=0`.

## 2026-05-05 Codex Release-Ready Source Custody R20

Burned down the remaining reviewed `track` dispositions from live untracked
source into git-index-visible source. This is the first tranche where the
canonical source-like inventory is actually zero instead of merely
dispositioned.

Actions:

- Staged the 65 remaining source-like untracked files that had reviewed
  `track` dispositions.
- Staged the small canonical report summary
  `reports/meta_lagrangian_apogee_full_sweep.json` after broadening the audit
  to treat top-level `reports/` summaries as source-like custody surfaces.
- Updated `tools/audit_untracked_source_artifacts.py` so disposition entries
  remain valid after their files become tracked. A reviewed disposition is now
  valid when the path is either:
  - still live and source-like untracked, or
  - present in `git ls-files` as source-like tracked source.
- Added regression coverage for the resolved-tracked disposition case.
- Updated `tools/all_lanes_preflight.py` Gate #8 output to report
  `resolved_tracked_dispositions`.

Current strict signal:

- `untracked_source_like_count=0`
- `undispositioned_count=0`
- `invalid_disposition_count=0`
- `shadowed_index_delete_count=0`
- `resolved_tracked_disposition_count=65`

Remaining untracked paths after this tranche are outside the source-like audit
surface: generated `.omx/research/artifacts`, Lightning result mirrors, and
`reverse_engineering/.gitignore`.

Verification:

- `.venv/bin/python tools/audit_untracked_source_artifacts.py --format json
  --disposition-manifest
  .omx/research/untracked_source_dispositions_20260505_codex.json --strict`
  passed with `ready_for_no_signal_loss_canonicalization=true`.
- `.venv/bin/python -m pytest
  src/tac/tests/test_audit_untracked_source_artifacts.py -q` passed:
  `6 passed`.
- `.venv/bin/python -m ruff check tools/all_lanes_preflight.py
  tools/audit_untracked_source_artifacts.py
  src/tac/tests/test_audit_untracked_source_artifacts.py` passed.
- `.venv/bin/python -m py_compile tools/all_lanes_preflight.py
  tools/audit_untracked_source_artifacts.py` passed.
- `.venv/bin/python tools/all_lanes_preflight.py` passed:
  `ALL 12 PREFLIGHT CHECKS PASSED`, with Gate #8 reporting
  `0 source-like untracked`, `shadowed_index_deletes=0`, and
  `resolved_tracked_dispositions=65`.

## 2026-05-05 Codex Orphan-Recovery Deletion Triage R21

Triaged the remaining source-like deletions after canonical source staging.
These were not missing live files; they were duplicate files under
`reverse_engineering/orphan_pyc_recovery_20260505_codex/` after recovery into
canonical `experiments/`, `src/tac/`, or `tools/` paths.

Actions:

- Checked every deleted orphan-recovery path by stripping the
  `reverse_engineering/orphan_pyc_recovery_20260505_codex/` prefix and testing
  whether the canonical destination exists and is tracked.
- Restored three orphan-recovery files that did **not** have tracked canonical
  counterparts, preserving their remaining decompilation signal:
  - `reverse_engineering/orphan_pyc_recovery_20260505_codex/experiments/build_pr85_stbm1br_rmb1_randmulti_candidate.py`
  - `reverse_engineering/orphan_pyc_recovery_20260505_codex/src/tac/tests/test_build_pr85_stbm1br_rmb1_randmulti_candidate.py`
  - `reverse_engineering/orphan_pyc_recovery_20260505_codex/src/tac/tests/test_build_pr95_hnerv_residual_atom_plan.py`
- Staged deletion only for orphan-recovery duplicate files whose canonical
  destination now exists and is tracked.

Current signal:

- `source_like_delete_count=30`.
- Those 30 are staged deletions of duplicate orphan-recovery copies with
  tracked canonical counterparts.
- `shadowed_index_delete_count=0`.
- `untracked_source_like_count=0`.

Verification:

- `.venv/bin/python tools/audit_untracked_source_artifacts.py --format json
  --disposition-manifest
  .omx/research/untracked_source_dispositions_20260505_codex.json --strict`
  passed with `ready_for_no_signal_loss_canonicalization=true`.
- Manual canonical-counterpart check confirmed the restored three paths were
  the only deleted orphan-recovery files without tracked destinations.

## 2026-05-05 Codex Orphan-Recovery Canonicalization Gate R22

Added an executable audit for the duplicate orphan-recovery deletion policy.
The prior R21 manual check is now enforced by all-lanes preflight.

Changes:

- Added `tools/audit_orphan_recovery_canonicalization.py`.
- Added regression coverage in
  `src/tac/tests/test_audit_orphan_recovery_canonicalization.py`.
- Wired the audit into `tools/all_lanes_preflight.py` as Gate #9.

Gate rule:

- Any source-like deletion outside
  `reverse_engineering/orphan_pyc_recovery_20260505_codex/` is a blocker.
- Any orphan-recovery source-like deletion that is not staged cleanly is a
  blocker.
- Any orphan-recovery source-like deletion whose stripped canonical path is not
  tracked is a blocker.

Current strict signal:

- `source_like_delete_count=30`
- `canonicalized_duplicate_delete_count=30`
- `non_orphan_delete_count=0`
- `unstaged_delete_count=0`
- `missing_canonical_count=0`

Verification:

- `.venv/bin/python tools/audit_orphan_recovery_canonicalization.py --format
  json --strict` passed with
  `ready_for_orphan_recovery_cleanup=true`.
- `.venv/bin/python -m pytest
  src/tac/tests/test_audit_orphan_recovery_canonicalization.py
  src/tac/tests/test_audit_untracked_source_artifacts.py -q` passed:
  `8 passed`.
- `.venv/bin/python -m ruff check
  tools/audit_orphan_recovery_canonicalization.py
  tools/all_lanes_preflight.py
  src/tac/tests/test_audit_orphan_recovery_canonicalization.py` passed.
- `.venv/bin/python -m py_compile
  tools/audit_orphan_recovery_canonicalization.py
  tools/all_lanes_preflight.py` passed.
- `.venv/bin/python tools/all_lanes_preflight.py` passed:
  `ALL 13 PREFLIGHT CHECKS PASSED`.

## 2026-05-05 Codex Public-Release Residue Hygiene R23

Closed the generated-residue side of the no-signal-loss sweep. The previous
state had no source-like untracked files after R20/R22, but generated custody
JSON/Markdown under `.omx/research/artifacts/` and Lightning exact-eval mirrors
still appeared as untracked working-tree noise. That made it too easy to miss
real recovered source in future sweeps.

Changes:

- Updated `comma_lab.research_state.classify_relpath()` so
  `.omx/research/artifacts/**` is always classified as generated research
  artifact custody, even when the file is small text.
- Added `.gitignore` policy for generated research artifacts and local
  Lightning batch mirror files:
  - `.omx/research/artifacts/`
  - `experiments/results/lightning_batch/**/source_manifest.json`
  - exact-eval/adjudication/provenance JSONs
  - `lightning_*.json`, `lightning_dali_requirements.txt`, and mirrored
    `report.txt`
- Staged `reverse_engineering/.gitignore` as a tracked curation policy for raw
  public-PR intake payloads.
- Added regression coverage in `tests/test_comma_lab_research_state.py`.

Strict signal:

- `tools/audit_research_state_tracking.py --fail-on-untracked-trackable` now
  passes over `.omx` and `reports`.
- `git check-ignore -v` confirms current generated `.omx/research/artifacts`
  outputs and untracked Lightning mirror outputs are ignored by explicit policy.
- `git status --short --untracked-files=all | rg '^\\?\\?'` reports only
  `reverse_engineering/.gitignore` before staging.

Verification:

- `.venv/bin/python -m pytest tests/test_comma_lab_research_state.py -q`
  passed: `9 passed`.
- `.venv/bin/python -m py_compile src/comma_lab/research_state.py` passed.

## 2026-05-05 Codex HNeRV Payload-Section Manifest R24

Advanced the `hnerv_payload_scorecard_followups` hidden-gem row from prose
follow-up routing into a deterministic section manifest. This is still byte
forensics only, but it now gives future repackers exact old-section byte
ranges, hashes, entropy, and optimization roles before they are allowed to
claim that a candidate changed decoder or latent bytes.

Changes:

- Added `payload_section_manifests()` to
  `experiments/build_hnerv_frontier_scorecard.py`.
- Added stable section role classification:
  `decoder_weight_stream`, `latent_stream`, `sidecar_or_correction_stream`,
  `entropy_model_or_range_stream`, `control_or_metadata`, and
  `opaque_payload_stream`.
- Regenerated
  `experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/scorecard.json`
  and `.md` with four payload-section manifests for PR105/PR105x/PR106/PR106x.
- Tightened `tools/audit_hnerv_frontier_scorecard.py` so hidden-gem routing now
  fails closed when section manifests are absent, score-claiming, dispatching,
  missing section hashes/bytes, or missing decoder/latent roles.
- Added regression coverage in
  `src/tac/tests/test_build_hnerv_frontier_scorecard.py` and
  `src/tac/tests/test_audit_hnerv_frontier_scorecard.py`.

Current strict signal:

- `tools/audit_hnerv_frontier_scorecard.py --format json` reports
  `ready_for_hidden_gem_routing=true`, `blockers=[]`,
  `payload_section_manifest_count=4`.
- PR106 and PR106x now expose identical decoder and latent stream SHA-256s in
  the manifest, making the next repack no-op check explicit.

Verification:

- `.venv/bin/python -m pytest
  src/tac/tests/test_build_hnerv_frontier_scorecard.py
  src/tac/tests/test_audit_hnerv_frontier_scorecard.py -q` passed:
  `7 passed`.
- `.venv/bin/python -m py_compile
  experiments/build_hnerv_frontier_scorecard.py` passed.

## 2026-05-05 Codex Archive Byte Profiler Rehydration R25

Closed another recovered-source stub in the byte-analysis stack. The
rehydrated `src/tac/archive_byte_profile.py` previously exposed only
`contest_rate_term()` while core profiling functions raised
`NotImplementedError`; that left byte-level archive analysis dependent on
lane-local scripts instead of a reusable `tac` primitive.

Changes:

- Implemented `profile_archive()` as a deterministic ZIP profiler that records:
  archive bytes/SHA, rate term, member names, member compressed/uncompressed
  bytes, payload SHA-256, CRC32, compression method, top-level grouping, suffix
  grouping, and ZIP overhead.
- Implemented `invalid_archive_record()`, `build_profile_collection()`,
  `render_markdown()`, and `write_outputs()`.
- Preserved strict archive hygiene: empty names, NULs, backslashes, absolute
  paths, parent traversal, and duplicate member names fail closed.
- Added regression coverage in `src/tac/tests/test_archive_byte_profile.py`.

Evidence boundary:

- The profiler is byte-only and always records `score_claim=false`.
- It does not inflate outputs, load scorer models, or make score assertions.
- It is now safe to use as a reusable primitive for HNeRV/PR archive anatomy,
  hidden-gem byte screens, and public writeup tables.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_archive_byte_profile.py
  src/tac/tests/test_rehydrated_modules_20260505.py::test_contest_rate_term_exact
  -q` passed: `4 passed`.
- `.venv/bin/python -m py_compile src/tac/archive_byte_profile.py
  src/tac/tests/test_archive_byte_profile.py` passed.

## 2026-05-05 Codex HNeRV/Fridrich/Wavelet Stack Planning R26

Broadened the hidden-gem planning surface beyond HNeRV while keeping HNeRV as
the current public-frontier anchor.

Stackability decision:

- HNeRV remains the anchor representation for frontier-compatible archive work.
- Fridrich/UNIWARD-style detector-aware costs are most stackable as allocation
  feedback over charged HNeRV latent, sidecar, and correction atoms.
- Wavelets are most stackable as residual-basis or mask-side correction
  coordinates; they are not currently treated as a replacement for the full
  HNeRV stream.
- RAFT, telescope/foveation, SIREN, CLaDE, and SPADE-family ideas remain valid,
  but should first feed allocation fields or small charged residuals unless a
  closed runtime replacement proves exact CUDA stability.

Changes:

- Added `src/tac/hnerv_section_repack.py`, a planning-only section-repack
  target builder that consumes HNeRV payload-section manifests and emits ranked
  byte targets with old-section SHA, role, byte count, projected rate-only
  savings, and required candidate proof fields.
- Added `tools/plan_hnerv_section_repack.py`.
- Added hidden-gem rows:
  - `fridrich_inverse_steg_allocator`
  - `wavelet_residual_basis_gate`
- Generated `.omx/research/hnerv_pr106x_section_repack_plan_20260505_codex.md`
  for the PR106x anchor. The highest byte target is
  `decoder_packed_brotli` at `170278` bytes; a five-percent byte saving there
  is only about `0.005668457` score-rate term, so this lane needs either
  meaningful decoder self-compression or to combine with latent/correction
  transforms.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_hidden_gems.py
  src/tac/tests/test_hidden_gem_readiness.py
  src/tac/tests/test_hnerv_section_repack.py -q` passed: `11 passed`.
- `.venv/bin/python tools/audit_hidden_gem_readiness.py --format json`
  reports `entry_count=14`, `eligible_for_local_patch_count=7`, and
  `ready_for_exact_eval_dispatch_count=0`.
- `.venv/bin/python -m py_compile src/tac/hidden_gems.py
  src/tac/hnerv_section_repack.py tools/plan_hnerv_section_repack.py` passed.

## 2026-05-05 Codex HNeRV Candidate Diff No-Op Gate R27

Added the missing proof boundary between a section-repack target and a future
archive candidate.

Change:

- Added `audit_candidate_section_diff()` to `src/tac/hnerv_section_repack.py`.

Gate behavior:

- Requires candidate archive SHA-256.
- Requires each changed section to match a known `(label, section_name)` from
  the section-repack plan.
- Requires source section SHA-256 to match the plan.
- Requires candidate section SHA-256 to be valid and different, or candidate
  byte count to differ.
- Requires source byte count to match the plan.
- Rejects duplicate candidate section rows.
- Rejects no-op diffs.

Evidence boundary:

- A passing diff is only `ready_for_archive_preflight=true`.
- It remains `ready_for_exact_eval_dispatch=false` until archive manifest
  preflight, dispatch claim, and exact CUDA auth eval.
- It records `rate_score_delta_if_components_equal` as pure rate arithmetic,
  not as a score claim.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_hnerv_section_repack.py -q`
  passed: `4 passed`.
- `.venv/bin/python -m ruff check src/tac/hnerv_section_repack.py
  src/tac/tests/test_hnerv_section_repack.py` passed.
- `.venv/bin/python -m py_compile src/tac/hnerv_section_repack.py
  src/tac/tests/test_hnerv_section_repack.py` passed.

## 2026-05-05 Codex HNeRV Scorecard-Diff No-Op Control R28

Added a low-level byte-packing guard for public-frontier repack work.

Changes:

- Added `candidate_diff_from_scorecard_manifests()` to
  `src/tac/hnerv_section_repack.py`.
- Added `tools/audit_hnerv_section_candidate_diff.py`.
- Added CLI and unit coverage in `src/tac/tests/test_hnerv_section_repack.py`.

Current control result:

- Audited PR106 -> PR106x from the checked-in HNeRV scorecard.
- Result: `ready_for_archive_preflight=false`, `changed_section_count=0`,
  `total_byte_delta=0`.
- Blockers:
  - `candidate_section_noop:PR106:decoder_packed_brotli`
  - `candidate_section_noop:PR106:latents_and_sidecar_brotli`
  - `candidate_section_noop:PR106:packed_header_ff_len24`
  - `candidate_diff_has_no_changed_sections`

Interpretation:

- PR106x remains useful as a container/member-name control, but it is not a
  payload-section transform.
- Any aggressive low-level byte packer must beat this by changing at least one
  charged section SHA-256 and recording old/new byte counts before archive
  preflight.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_hnerv_section_repack.py -q`
  passed: `6 passed`.
- `.venv/bin/python -m ruff check src/tac/hnerv_section_repack.py
  tools/audit_hnerv_section_candidate_diff.py
  src/tac/tests/test_hnerv_section_repack.py` passed.
- `.venv/bin/python -m py_compile src/tac/hnerv_section_repack.py
  tools/audit_hnerv_section_candidate_diff.py
  src/tac/tests/test_hnerv_section_repack.py` passed.

## 2026-05-05 Codex HNeRV Low-Level Brotli Repack R29

Built the first concrete byte-changing low-level HNeRV packer for the PR106x
frontier-style payload.

Changes:

- Added `src/tac/hnerv_lowlevel_packer.py`.
- Added `tools/build_hnerv_lowlevel_repack_candidate.py`.
- Added `src/tac/tests/test_hnerv_lowlevel_packer.py`.

Contract:

- Reads strict single-member ZIP archives without assuming the member name.
- Rejects absolute paths, parent traversal, resource-fork paths, duplicate
  members, multi-member archives, CRC failures, and non-`0xff` packed HNeRV
  payloads.
- Parses the PR106-family payload as:
  - `packed_header_ff_len24`
  - `decoder_packed_brotli`
  - `latents_and_sidecar_brotli`
- Recompresses only brotli-decompressible sections.
- Emits a deterministic stored ZIP candidate only when at least one targeted
  section changes and is rate-positive by default.
- Records old/new section SHA-256s, old/new byte counts, candidate archive
  SHA-256, candidate archive bytes, and decompressed-raw equality for each
  brotli section.
- Remains `score_claim=false` and `ready_for_exact_eval_dispatch=false`.

Real PR106x byte result:

- Source archive:
  `experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/archive.zip`
- Candidate archive:
  `.omx/research/artifacts/hnerv_lowlevel_repack_20260505_codex/pr106x_hnerv_brotli_repack_candidate.zip`
- Candidate manifest:
  `.omx/research/artifacts/hnerv_lowlevel_repack_20260505_codex/pr106x_brotli_repack_manifest.json`
- Source archive bytes: `186231`
- Candidate archive bytes: `186080`
- Candidate archive SHA-256:
  `b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f`
- Accepted section:
  - `decoder_packed_brotli`: `170278 -> 170127`, `-151` bytes
- Excluded section:
  - `latents_and_sidecar_brotli`: changed under one brotli setting but did not
    shrink; excluded from candidate by the rate-positive default.
- Raw equality:
  - decoder raw: equal, `229070` bytes
  - latents raw: equal, `33712` bytes
- Candidate diff audit:
  - `ready_for_archive_preflight=true`
  - `changed_section_count=2` because the 24-bit length header also changes
  - `total_byte_delta=-151`
  - `rate_score_delta_if_components_equal=-0.000100544702`
  - blockers: `[]`

Interpretation:

- This is a small but real no-op-resistant byte opportunity.
- It is not enough to move the frontier by itself, but it proves the low-level
  packer path is wired correctly and can now be reused for more aggressive
  section transforms.
- Exact score is still unknown until the candidate passes full archive
  preflight, dispatch claim, and exact CUDA auth eval on the exact archive
  bytes.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_hnerv_lowlevel_packer.py
  src/tac/tests/test_hnerv_section_repack.py -q` passed: `11 passed`.
- `.venv/bin/python -m py_compile src/tac/hnerv_lowlevel_packer.py
  tools/build_hnerv_lowlevel_repack_candidate.py` passed.

Adversarial review hardening:

- Added decompressed-raw equality into the candidate diff itself and made the
  low-level packer call `audit_candidate_section_diff(...,
  require_raw_equivalence=True)`.
- Tightened source scorecard custody:
  - `archive_sha256` is required and must match.
  - `archive_bytes` is required and must match.
  - `zip_member` is required and must match.
  - `payload_sha256` is required and must match.
  - `member_bytes` is required and must match.
  - section count, order, `index`, `start`, `end`, bytes, and section SHA-256
    must match the parsed PR106-style payload.
- Tightened strict ZIP handling:
  - rejects archives with more than one ZIP entry, even if the extra entry is
    a directory;
  - rejects directory-only archives;
  - rejects parent traversal, absolute paths, hidden paths, backslashes, and
    resource-fork paths.
- Slugged candidate output labels before using them in file names.
- Clarified that `--fail-if-blocked` means archive-preflight readiness only,
  not exact-eval dispatch readiness.
- Added Gate #6 to `tools/all_lanes_preflight.py`, which proves the real
  PR106x low-level brotli candidate in a temporary directory while keeping
  `ready_for_exact_eval_dispatch=false`.

Updated verification:

- `.venv/bin/python -m pytest src/tac/tests/test_hnerv_lowlevel_packer.py
  src/tac/tests/test_hnerv_section_repack.py -q` passed: `12 passed`.
- `.venv/bin/python -m ruff check src/tac/hnerv_lowlevel_packer.py
  src/tac/hnerv_section_repack.py tools/build_hnerv_lowlevel_repack_candidate.py
  tools/all_lanes_preflight.py src/tac/tests/test_hnerv_lowlevel_packer.py
  src/tac/tests/test_hnerv_section_repack.py` passed.
- `.venv/bin/python -m py_compile src/tac/hnerv_lowlevel_packer.py
  src/tac/hnerv_section_repack.py tools/build_hnerv_lowlevel_repack_candidate.py
  tools/all_lanes_preflight.py` passed.
- `.venv/bin/python tools/all_lanes_preflight.py` passed: `ALL 14
  PREFLIGHT CHECKS PASSED`.

## 2026-05-05 Codex PR106x Decoder Recode And Meta-Lagrangian Wiring R30

Added a planning-only structural-recode profiler for the PR106-family HNeRV
decoder section and wired its byte atoms into a canonical meta-Lagrangian atom
ledger.

Changes:

- Added `src/tac/hnerv_decoder_recode.py`.
- Added `tools/profile_hnerv_decoder_structural_recode.py`.
- Added `src/tac/tests/test_hnerv_decoder_recode.py`.
- Added `src/tac/meta_lagrangian_allocator.py`.
- Added `tools/build_meta_lagrangian_atom_ledger.py`.
- Added `src/tac/tests/test_meta_lagrangian_allocator.py`.

Structural-recode result on real PR106x:

- Source archive:
  `experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/archive.zip`
- Profile:
  `.omx/research/artifacts/hnerv_decoder_structural_recode_20260505_codex/pr106x_decoder_recode_profile.json`
- Source decoder section bytes: `170278`
- Source decoder raw bytes: `229070`
- Quantized stream bytes: `228958`
- Scale stream bytes: `112`
- Best measured variant:
  - `brotli_q10_current_raw`
  - bytes: `170127`
  - delta: `-151`
  - raw_equal: `true`
- Dependency-free static arithmetic and canonical-Huffman probes were
  measured but lost badly:
  - global AQv1: `+18713` bytes
  - per-tensor AQv1: `+27863` bytes
  - global TFC1 Huffman: `+19175` bytes

Interpretation:

- The public comment that a categorical AC path once saved roughly this order
  of bytes is plausible, but the existing dependency-free AQv1/TFC1 coders are
  not the right implementation for this high-entropy uint8 stream.
- The only current positive byte atom remains the same lossless brotli-q10
  section recode already represented by R29.
- Larger decoder wins require a real model-aware representation change
  (block-FP/FP4/learned weight codec) with distortion evidence, not generic
  static entropy coding over the current zigzag uint8 stream.

Meta-Lagrangian integration:

- `src/tac/meta_lagrangian_allocator.py` now defines the common planning
  contract for rate atoms, hard-pair atoms, class/label atoms, pose atoms,
  foveation/geometry atoms, and openpilot-prior atoms.
- It computes:
  - exact official rate score delta from charged byte delta;
  - exact local PoseNet score-term delta around a base pose distance;
  - confidence-weighted expected SegNet/PoseNet deltas;
  - a planning-only expected total score delta.
- `tools/build_meta_lagrangian_atom_ledger.py` converts the HNeRV decoder
  recode profile into rate-only atoms and writes a ranked ledger:
  `.omx/research/artifacts/hnerv_decoder_structural_recode_20260505_codex/pr106x_decoder_recode_meta_lagrangian_ledger.json`
- Top atom:
  - `PR106x:decoder_recode:brotli_q10_current_raw`
  - byte_delta: `-151`
  - expected_total_score_delta: `-0.000100544702`
  - evidence_grade: `empirical_byte_raw_equal`
  - dispatchable: `false`

Why this matters:

- Byte opportunities are now compatible with the same allocator surface as
  hard-pair, categorical label, PoseNet/SegNet, foveation, ego-motion, and
  openpilot-policy/world-model priors.
- This keeps "water bucket filling" from becoming raw byte sorting. Every atom
  must declare bytes, expected component movement, confidence, support pairs,
  support classes, geometry priors, openpilot priors, and exact-eval blockers.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_hnerv_decoder_recode.py
  src/tac/tests/test_meta_lagrangian_allocator.py -q` passed: `7 passed`.
- `.venv/bin/python -m ruff check src/tac/hnerv_decoder_recode.py
  src/tac/meta_lagrangian_allocator.py
  tools/profile_hnerv_decoder_structural_recode.py
  tools/build_meta_lagrangian_atom_ledger.py
  src/tac/tests/test_hnerv_decoder_recode.py
  src/tac/tests/test_meta_lagrangian_allocator.py` passed.
- `.venv/bin/python -m py_compile src/tac/hnerv_decoder_recode.py
  src/tac/meta_lagrangian_allocator.py
  tools/profile_hnerv_decoder_structural_recode.py
  tools/build_meta_lagrangian_atom_ledger.py` passed.

## 2026-05-05 Codex LA-POSE Motion Atom Water-Fill Scaffold R31

Added the first concrete bridge from LA-POSE-style motion features to the
meta-Lagrangian hard-pair allocator.

Changes:

- Added `src/tac/lapose_motion_atoms.py`.
- Added `tools/build_lapose_motion_atom_manifest.py`.
- Added `src/tac/tests/test_lapose_motion_atoms.py`.

Contract:

- Input records contain:
  - `pair_index`
  - `latent_action`
  - charged byte estimate as `byte_delta` or `estimated_charged_bytes`
  - expected SegNet/PoseNet distance deltas
  - confidence
  - class support
  - geometry priors
  - openpilot priors
- The builder validates finite latent vectors, nonnegative charged byte costs,
  confidence in `[0, 1]`, class lists, and prior lists.
- It builds a deterministic sparse graph over latent-action distances using a
  connectivity-first nearest-edge construction.
- It emits atoms consumed by `tac.meta_lagrangian_allocator`, preserving
  hard-pair support, class support, geometry priors, openpilot priors, and
  exact-eval blockers.
- It remains planning-only:
  - `score_claim=false`
  - `dispatch_attempted=false`
  - `ready_for_exact_eval_dispatch=false`

Generated fixture artifact:

- Records:
  `.omx/research/artifacts/lapose_motion_atoms_20260505_codex/lapose_motion_records_fixture.json`
- Manifest:
  `.omx/research/artifacts/lapose_motion_atoms_20260505_codex/lapose_motion_atom_manifest_fixture.json`
- Fixture records use hard-pair prior pairs `67`, `75`, `127`, `285`, and
  `294`, class supports such as `2->3`/`0->3` neighborhoods, and openpilot
  priors like `ego_motion` and `yaw_rate`.
- Sparse graph:
  - nodes: `5`
  - edges: `5`
  - average degree: `2.0`
- Top planning atom under the fixture expected-score model:
  - `lapose_motion_pair:75`
  - expected_total_score_delta: `-0.012781734533`
  - byte_delta: `72`
  - class_support: `[2, 3]`
  - geometry_priors: `["foveal_lane_boundary"]`
  - openpilot_priors: `["ego_motion", "yaw_rate"]`

Interpretation:

- This is the water-bucket/Lagrangian shape we want: bytes, hard pairs,
  categories/labels, PoseNet/SegNet expected deltas, geometry, openpilot priors,
  and confidence in one atom ledger.
- The fixture component deltas are not evidence. They exist only to prove the
  pipeline shape. Real use requires exact component trace or offline target
  manifests.
- This makes LA-POSE actionable without smuggling a scorer or uncharged latent:
  the motion feature is optimizer feedback until a charged archive consumes the
  selected atoms.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_lapose_motion_atoms.py
  src/tac/tests/test_meta_lagrangian_allocator.py -q` passed: `8 passed`.
- `.venv/bin/python -m ruff check src/tac/lapose_motion_atoms.py
  tools/build_lapose_motion_atom_manifest.py
  src/tac/tests/test_lapose_motion_atoms.py` passed.
- `.venv/bin/python -m py_compile src/tac/lapose_motion_atoms.py
  tools/build_lapose_motion_atom_manifest.py` passed.

## 2026-05-05 Codex LA-POSE Evidence Bridge And Gap Postmortem R32

Added a stricter evidence-ingestion bridge for LA-POSE-style motion atoms and
recorded the public-facing postmortem for why the contest gap remained.

Code changes:

- Added `src/tac/lapose_motion_evidence.py`.
- Added `tools/build_lapose_motion_records_from_component_response.py`.
- Added `src/tac/tests/test_lapose_motion_evidence.py`.
- Extended `src/tac/lapose_motion_atoms.py` so records preserve:
  - `evidence_source_path`
  - `evidence_source_sha256`
  - `source_archive_sha256`
- Extended `src/tac/meta_lagrangian_allocator.py` so ranked atom rows preserve
  the same evidence/source-custody fields.

Evidence bridge contract:

- Consumes CUDA component-response summaries plus explicit latent-action and
  pair-opportunity records.
- Fails closed if:
  - source path is missing;
  - baseline archive SHA is missing;
  - component response is not CUDA evidence;
  - a required pair latent is missing;
  - pair opportunity mass is non-positive;
  - confidence is outside `[0, 1]`.
- Selects the best observed component-response point relative to epsilon-zero
  baseline and allocates its global byte/SegNet/PoseNet deltas across pair
  opportunities by opportunity mass.
- Uses deterministic signed-integer apportionment for byte deltas, preserving
  the exact total byte delta even for rate-saving atoms.
- Remains planning-only:
  - `score_claim=false`
  - `dispatch_attempted=false`
  - `ready_for_exact_eval_dispatch=false`

Semantic hardening found during tests:

- `byte_delta` must be signed because byte-recoding atoms can save bytes.
- `estimated_charged_bytes` remains non-negative when used as a cost estimate.
- The allocator previously dropped evidence-source fields; this is now fixed so
  every ranked atom can cite its source component-response artifact.

Writeup/site documentation:

- Added `docs/postmortem_bridge_gap_20260505.md`.
- Updated `docs/paper/04_results.md` with §4.8, "Postmortem: Why The Gap
  Remained."
- Updated `reports/graphs/final_writeup_draft.md`.
- Updated `reports/graphs/site/final_writeup_draft.md`.
- Updated `reports/graphs/writeup_outline.md`.

Postmortem summary:

- The project was not mainly short on ideas. HNeRV/NeRV, arithmetic/range
  coding, hard-pair water-filling, foveation, LA-POSE-style motion priors,
  scorer-targeted corrections, and self-compression were all in the research
  stream.
- The gap was research-to-archive conversion latency: too many ideas stayed in
  notes, partial lanes, proxy experiments, or fragile scripts instead of
  becoming byte-closed archives with exact CUDA evidence early enough.
- Future operating model: one strict promotion lane plus one risky frontier
  lane, every research note names the smallest charged archive experiment,
  public PR deconstruction runs continuously during deadline windows, and bug
  classes are treated as compiler errors before dispatch.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_lapose_motion_atoms.py
  src/tac/tests/test_lapose_motion_evidence.py
  src/tac/tests/test_meta_lagrangian_allocator.py -q` passed: `11 passed`.
- `.venv/bin/python -m ruff check src/tac/lapose_motion_atoms.py
  src/tac/lapose_motion_evidence.py src/tac/meta_lagrangian_allocator.py
  tools/build_lapose_motion_atom_manifest.py
  tools/build_lapose_motion_records_from_component_response.py
  src/tac/tests/test_lapose_motion_atoms.py
  src/tac/tests/test_lapose_motion_evidence.py` passed.
- `.venv/bin/python -m py_compile src/tac/lapose_motion_atoms.py
  src/tac/lapose_motion_evidence.py src/tac/meta_lagrangian_allocator.py
  tools/build_lapose_motion_atom_manifest.py
  tools/build_lapose_motion_records_from_component_response.py` passed.
- `git diff --check` on touched docs and LA-POSE/meta-Lagrangian files passed.

## 2026-05-05 Codex Analysis/Optimization Canonicalization R33

Canonicalized the new hard-pair, LA-POSE-lite, and meta-Lagrangian work so it
does not remain as siloed lane code.

Package structure:

- Added `src/tac/analysis/`.
- Added `src/tac/optimization/`.
- Moved canonical LA-POSE motion atom implementation to
  `src/tac/analysis/lapose_motion_atoms.py`.
- Moved component-response-to-motion-record bridge to
  `src/tac/analysis/lapose_motion_evidence.py`.
- Moved LA-POSE-lite pair metric feature builder to
  `src/tac/analysis/lapose_lite_inputs.py`.
- Moved meta-Lagrangian allocator to
  `src/tac/optimization/meta_lagrangian_allocator.py`.
- Kept compatibility wrappers at the old root import paths:
  - `src/tac/lapose_motion_atoms.py`
  - `src/tac/lapose_motion_evidence.py`
  - `src/tac/lapose_lite_inputs.py`
  - `src/tac/meta_lagrangian_allocator.py`

Durable rule added:

- `AGENTS.md` now records `src/tac/analysis/` and `src/tac/optimization/` as
  canonical package boundaries.
- `AGENTS.md` now explicitly says Lane W hard-pair weights are general CUDA
  per-pair scorer telemetry over the contest video, not Lane-W-local state, and
  should route any downstream atom family after canonicalization.
- `docs/runbooks/analysis_optimization_package_map.md` documents the same OSS
  package map for comma-ai/OSS readers.

Real non-fixture artifacts generated:

- LA-POSE-lite inputs from Lane W CUDA pair telemetry:
  `.omx/research/artifacts/lapose_motion_atoms_20260505_codex/lane_w_cuda_pair_metrics_lapose_lite_inputs.json`
  - source SHA:
    `3225b7f74742a588abc9b8cd12ecc5b9e9c9195efed65526e00aaceca48b7f33`
  - selected pairs: `30`
  - evidence grade: `empirical_cuda_pair_metric_telemetry`
  - blocker includes:
    `lane_w_pair_metrics_are_general_scorer_telemetry_not_lane_local_state`
- Component-response-allocated motion records:
  `.omx/research/artifacts/lapose_motion_atoms_20260505_codex/lane_w_component_allocated_lapose_motion_records.json`
- Ranked motion atom manifest:
  `.omx/research/artifacts/lapose_motion_atoms_20260505_codex/lane_w_component_allocated_lapose_motion_atom_manifest.json`
  - source:
    `lane_w_cuda_pair_metrics_plus_pfp16_component_response_20260505`
  - record count: `30`
  - graph average degree: `2.0`
  - top planning atoms include pairs `456`, `210`, `522`, `372`, and `454`.

Important boundary:

- This is still planning-only feedback. Pair metrics and component-response
  curves can route archive atoms but cannot claim score, rank candidates, or
  dispatch GPU work until a charged archive builder consumes the selected atoms
  and exact CUDA auth eval scores the exact bytes.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_lapose_motion_atoms.py
  src/tac/tests/test_lapose_motion_evidence.py
  src/tac/tests/test_lapose_lite_inputs.py
  src/tac/tests/test_meta_lagrangian_allocator.py -q` passed: `15 passed`.
- `.venv/bin/python -m ruff check ...` on analysis/optimization modules,
  compatibility wrappers, tools, and focused tests passed.
- `.venv/bin/python -m py_compile ...` on analysis/optimization modules,
  compatibility wrappers, and tools passed.
- Stale-import search for top-level `tac.lapose*`/`tac.meta_lagrangian*`
  imports returned no matches outside compatibility wrappers.

## 2026-05-05 Codex Distortion-Model And Evidence-Semantics Hardening R34

Harvey adversarial review found that the new analysis/optimization chain was
mostly useful but had evidence-label bugs that could recreate the Apogee intN
failure mode in subtler form: scorer-pair telemetry was being labeled as
hard-pair evidence, component-response allocation sounded per-pair empirical
when it was global-response inference, openpilot priors were emitted without a
source payload, and `max_atoms` truncated before ranking.

Fixes landed:

- `src/tac/analysis/lapose_motion_atoms.py`
  - separates `pair_support` from `hard_pair_support`;
  - preserves `hard_pair_support` only when the input record explicitly carries
    it;
  - ranks the full meta-Lagrangian ledger before `max_atoms` truncation;
  - records `source_atom_count` so truncation is auditable.
- `src/tac/analysis/lapose_motion_evidence.py`
  - relabels global component-response allocation as
    `diagnostic_cuda_global_response_allocated`;
  - records `allocation_inference=true` and
    `measurement_scope=global_component_response_allocated_to_pairs_by_opportunity_mass`;
  - emits pair support without claiming hard-pair proof.
- `src/tac/analysis/lapose_lite_inputs.py`
  - emits `scorer_pair_metric` / `pair_metric_hardness` geometry priors instead
    of `metric_hard_pair`;
  - emits openpilot priors only when the source payload supplies them.
- `src/tac/optimization/meta_lagrangian_allocator.py`
  - carries `pair_support` through the atom ledger.
- `src/tac/optimizer/meta_lagrangian.py`
  - preserves `archive_path` on `CandidateEvaluation` so CLI JSON reports keep
    exact candidate custody paths.
- `tools/dispatch_readiness_apogee_int6.py`
  - is now a fast fail-closed forensic audit. It records archive identity, then
    blocks on the missing valid distortion model and skips expensive readiness
    checks because they cannot authorize dispatch.
- `docs/paper/hard_pair_analysis.md`
  - fixes pair attribution math: the contest PoseNet term is
    `sqrt(10 * average_pose_dist)`, so per-pair allocation must use marginal or
    linearized contribution around the current average, not average per-pair
    square roots;
  - adds public-release caution for raw Lane W internals.

Regenerated real planning artifacts:

- `.omx/research/artifacts/lapose_motion_atoms_20260505_codex/lane_w_cuda_pair_metrics_lapose_lite_inputs.json`
  now has `openpilot_priors=[]` unless sourced and geometry priors
  `scorer_pair_metric`, `pair_metric_hardness`.
- `.omx/research/artifacts/lapose_motion_atoms_20260505_codex/lane_w_component_allocated_lapose_motion_records.json`
  now carries `diagnostic_cuda_global_response_allocated` and
  `allocation_inference=true`.
- `.omx/research/artifacts/lapose_motion_atoms_20260505_codex/lane_w_component_allocated_lapose_motion_atom_manifest.json`
  now carries `pair_support`, empty `hard_pair_support` for unsourced hard-pair
  proof, and `source_atom_count=30`.

New coverage:

- `src/tac/tests/test_lapose_planning_chain.py` covers the full read-only chain:
  pair metrics -> LA-POSE-lite inputs -> component-response records -> motion
  atom manifest -> meta-Lagrangian ledger.
- `src/tac/tests/test_dispatch_readiness_apogee_int6.py` asserts that Apogee
  int6 readiness cannot pass without a valid distortion model.
- Existing LA-POSE/meta-Lagrangian tests now cover rank-before-truncate,
  sourced-only openpilot priors, archive-path custody, and allocation inference.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_lapose_motion_atoms.py
  src/tac/tests/test_lapose_motion_evidence.py
  src/tac/tests/test_lapose_lite_inputs.py
  src/tac/tests/test_lapose_planning_chain.py
  src/tac/tests/test_meta_lagrangian_allocator.py
  src/tac/tests/test_meta_lagrangian.py
  src/tac/tests/test_dispatch_readiness_apogee_int6.py -q` passed:
  `32 passed`.
- `.venv/bin/python -m ruff check ...` on the touched analysis,
  optimization, meta-Lagrangian, Apogee readiness, and focused test files
  passed.
- Real Apogee int6 readiness run:
  `.venv/bin/python tools/dispatch_readiness_apogee_int6.py --json` exits `1`
  with archive SHA/bytes verified and
  `blocked: Apogee intN prediction lacks a valid distortion model`.
- Full repo integration gate:
  `.venv/bin/python tools/all_lanes_preflight.py` passed all `14` checks.
  Gate #9 reports `0` source-like untracked files, Apogee intN remains
  `forensic-only` and not exact-eval dispatch-ready, Ω-W-V3 remains local-smoke
  only, and PR106 sidechannel dry-run surfaces are intact.

Recursive adversarial review:

- Harvey found the evidence-semantics bugs fixed above.
- Spawned Erdos for Apogee/distortion-model dispatch-surface greenup.
- Spawned Darwin for read-only whole-surface adversarial review.

## R35 - Recursive Dispatch Semantics Greenup

Validated Darwin's highest-severity findings and hardened the live code paths:

- `src/tac/optimization/meta_lagrangian_allocator.py`
  - byte savings from `invalid`, `prediction`, `external`, explicitly
    non-rankable, non-raw-equal, or zero-confidence negative-byte atoms no
    longer rank ahead of valid atoms;
  - rows preserve `rankable=false` and explicit blockers such as
    `non_rankable_evidence_grade`, `raw_output_not_byte_equivalent`, and
    `byte_savings_without_trusted_equivalence` for training/diagnostic signal.
- `tools/apogee_intN_pareto.py`
  - removed the final launch-shaped command escape hatch from
    `--emit-forensic-one-liners`;
  - compatibility mode now emits non-executable withheld forensic summaries
    only, with no `launch_lane_on_vastai.py full`, `APOGEE_INTN_BITS=`,
    `--predicted-band`, or `--lane-script` text.
- `scripts/remote_lane_apogee_intN.sh`
  - fixed the Stage 0 guard comment/order regression exposed by tests so the
    first literal `provenance.json` occurrence is after required inflate/parser
    checks.
- `src/tac/tests/test_meta_lagrangian_cli_dispatch_guard.py` and
  `src/tac/tests/test_sidechannel_stack_predictor_dispatch_guard.py`
  - preserved Erdos' new guard tests as tracked source; the preflight
    untracked-source gate correctly blocked until they were staged.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_meta_lagrangian_allocator.py
  src/tac/tests/test_apogee_intN_pareto_dispatch_wiring.py
  src/tac/tests/test_predispatch_sanity.py src/tac/tests/test_meta_lagrangian.py
  src/tac/tests/test_meta_lagrangian_cli_dispatch_guard.py
  src/tac/tests/test_sidechannel_stack_predictor_dispatch_guard.py
  src/tac/tests/test_lightning_dispatch_pr106_stack.py
  src/tac/tests/test_predicted_vs_actual_reconciler.py -q` passed:
  `59 passed`.
- `.venv/bin/python -m ruff check ...` on the touched Python guard files
  passed.
- `.venv/bin/python -m py_compile ...` on the touched Python guard files
  passed.
- `bash -n scripts/remote_lane_apogee_intN.sh` passed.

## R36 - Recovered Parallel Sweep Tools Tracked And Fail-Closed

Gate #9 then surfaced two additional untracked source tools:

- `tools/parallel_dispatch_top_k.py`
- `tools/harvest_and_reseed.py`

Both are real contest-loop assets, so they were tracked rather than ignored.
They also carried the old race-window bug class: prediction-ranked candidates
could look dispatchable, and harvested scalar scores could reseed calibration
without forcing a score JSON custody path. Hardened semantics:

- `parallel_dispatch_top_k.py`
  - now describes itself as an actuator only for already exact-eval-ready
    candidates;
  - refuses any input payload marked `ready_for_exact_eval_dispatch=false`;
  - refuses candidates with missing/false readiness, prediction/proxy/forensic
    evidence semantics, existing dispatch blockers, or unverified score claims;
  - no longer falls back to `forensic_top_k`.
- `harvest_and_reseed.py`
  - requires every `[contest-CUDA]` harvested row to include a readable
    `score_json_path`;
  - verifies the scalar harvested score matches the score JSON value before
    creating an anchor;
  - records `evidence_semantics=contest_cuda_harvested_score_json_required`
    and `score_claim=false` on new anchors.

Verification:

- `.venv/bin/python -m ruff check tools/parallel_dispatch_top_k.py
  tools/harvest_and_reseed.py` passed.
- `.venv/bin/python -m py_compile tools/parallel_dispatch_top_k.py
  tools/harvest_and_reseed.py` passed.
- `git diff --check -- tools/parallel_dispatch_top_k.py
  tools/harvest_and_reseed.py` passed.

## R37 - Feedback Loop Sweep Recovered As Fail-Closed Scaffold

Gate #9 next surfaced `tools/feedback_loop_sweep.py`. It is a real closed-loop
research scaffold, but the recovered version still encoded the race-window
mistake: generated Apogee/proxy candidates could fan out into paid dispatches,
and "race mode" claimed it could drop gates.

Hardening:

- rewrote the tool contract as `rank -> dispatch-ready filter -> harvest`;
- Apogee generator marks candidates `ready_for_exact_eval_dispatch=false`,
  `evidence_semantics=byte_only_forensic`, and carries explicit dispatch
  blockers;
- added an exact-readiness filter that rejects prediction/proxy/forensic
  evidence, explicit blockers, and unverified score claims before any dispatch;
- removed the "drop gates" race-mode semantics; race mode only narrows top-K
  and budget while keeping exact-readiness gates active;
- non-dry-run now requires explicit `--allow-paid-dispatch`, after filtering.

Verification:

- `.venv/bin/python -m ruff check tools/feedback_loop_sweep.py` passed.
- `.venv/bin/python -m py_compile tools/feedback_loop_sweep.py` passed.
- `git diff --check -- tools/feedback_loop_sweep.py` passed.

## R38 - Public Evidence Drift And Release-Strict Audit Greenup

Closed the highest-risk public/release drift from Darwin's review:

- `docs/paper/04_results.md`, `docs/paper/01_introduction.md`,
  `docs/pr106_stacking_decision_table_20260504.md`, and
  `reports/graphs/site/final_writeup_draft.md`
  - separate our submitted PR100/PR107 Apogee packet from the PR106/PR106x
    public-frontier replay/control;
  - relabel Ω-W and Apogee intN sub-0.20 rows as forensic planning/local-smoke
    rather than locally verified or launch-ready;
  - preserve the historical race-window dispatch plan as process evidence, not
    current launch instruction.
- `scripts/remote_lane_omega_w_v3_pr106.sh`
  - removed the stale `NO_NVDEC_NEEDED` header;
  - removed `probe_nvdec.sh --ensure-dali`; DALI/NVDEC must be supplied by the
    canonical bootstrap rather than silently auto-installed by the wrapper.
- `tools/audit_hnerv_frontier_scorecard.py`
  - now requires `score`, `archive_bytes`, SegNet/PoseNet components,
    archive/payload/runtime SHA-256 fields, and eval artifact file custody on
    every canonical frontier row;
  - recomputes the contest formula and component contributions within stored
    JSON rounding tolerance.
- `src/comma_lab/reverse_engineering.py` and
  `tools/audit_reverse_engineering_tree.py`
  - added `--release-strict`, which blocks unresolved promotion, ledger,
    externalization, preserved-decompile, and manual-review dispositions before
    a public release bundle. Normal strict preflight remains unchanged.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_audit_hnerv_frontier_scorecard.py
  tests/test_comma_lab_reverse_engineering.py
  src/tac/tests/test_audit_recovered_remote_lanes.py -q` passed: `9 passed`.
- `.venv/bin/python -m ruff check tools/audit_hnerv_frontier_scorecard.py
  tools/audit_reverse_engineering_tree.py src/comma_lab/reverse_engineering.py
  src/tac/tests/test_audit_hnerv_frontier_scorecard.py
  tests/test_comma_lab_reverse_engineering.py` passed.
- `bash -n scripts/remote_lane_omega_w_v3_pr106.sh` passed.
- `.venv/bin/python tools/audit_hnerv_frontier_scorecard.py --format json`
  passed with zero blockers.
- `.venv/bin/python tools/audit_reverse_engineering_tree.py --summary --strict`
  passed with zero blockers.
- `.venv/bin/python tools/audit_reverse_engineering_tree.py --summary
  --release-strict` intentionally fails closed with unresolved release blockers
  until the reverse-engineering release manifest is curated.

## R39 - Reverse-Engineering Release Manifest Closure

Built the release-manifest layer that closes `--release-strict` without
weakening normal forensic custody:

- `src/comma_lab/reverse_engineering.py`
  - added `ReleaseResolutionRule` and `load_release_resolution_rules()`;
  - `release_blocking_records()` now accepts explicit release rules and keeps
    unresolved records blocking;
  - `render_json(..., release_strict=True, release_rules=[...])` records the
    number of applied rules.
- `tools/audit_reverse_engineering_tree.py`
  - added `--release-manifest`;
  - `--release-strict --release-manifest ...` resolves only records matched by
    explicit manifest rules.
- `.omx/research/reverse_engineering_release_manifest_20260505_codex.json`
  - records 10 release rules covering auto-memory snapshots, provider state,
    recovery specs, public runtime/result copies, damaged decompiles, operator
    tools, tac tests, docs/site candidates, experiment entrypoints, and
    submission runtime candidates;
  - each rule records the public-release action, ledger path, and note.
- `reverse_engineering/README.md`
  - documents the public-release manifest gate.
- `tools/all_lanes_preflight.py`
  - added Gate #11: reverse-engineering release manifest.

Verification:

- `.venv/bin/python -m pytest tests/test_comma_lab_reverse_engineering.py -q`
  passed: `1 passed`.
- `.venv/bin/python -m ruff check src/comma_lab/reverse_engineering.py
  tools/audit_reverse_engineering_tree.py tests/test_comma_lab_reverse_engineering.py`
  passed.
- `.venv/bin/python -m py_compile src/comma_lab/reverse_engineering.py
  tools/audit_reverse_engineering_tree.py tests/test_comma_lab_reverse_engineering.py`
  passed.
- `.venv/bin/python tools/audit_reverse_engineering_tree.py --summary
  --release-strict --release-manifest
  .omx/research/reverse_engineering_release_manifest_20260505_codex.json`
  passed with `files=716 blockers=0`.

## R40 - Release Index/Worktree Split Guard

Closed a publishability hazard where staged rollbacks were shadowed by the
working tree:

- Detected files whose working-tree blob still equaled `HEAD` while the index
  differed. This means local tests would read the preserved file, but a commit
  would publish a rollback. The affected split included `.omx/state/*`,
  `CLAUDE.md`, `docs/paper/07_discussion.md`, several dispatch scripts, and
  several preflight/test modules.
- Unstaged only the rollback-shaped index entries where the working tree
  already matched `HEAD`; no working-tree signal was deleted.
- Staged final working versions for the real hardening files that had both
  index and worktree edits: component-response builders, PR106 latent sidecar,
  Apogee dispatch/evidence tests, `tools/check_dispatch_cli_shell_hazards.py`,
  and `tools/predispatch_sanity.py`.
- Added `tools/audit_release_index_split.py`:
  - blocks `shadowed_staged_rollback`;
  - blocks staged provider/runtime `.omx/state` files except small markdown
    control-plane files;
  - warns, but does not fail, on unstaged private runtime state that is
    intentionally preserved locally.
- Added Gate #12 to `tools/all_lanes_preflight.py` so the release split is
  checked before any all-lanes greenup claim.
- Added `src/tac/tests/test_audit_release_index_split.py` for stable parsing and
  schema coverage.

Verification:

- `.venv/bin/python tools/audit_release_index_split.py --strict` passed with
  warnings only for unstaged private runtime state:
  `.omx/state/lightning_batch_jobs.json`,
  `.omx/state/review_tracker.duckdb`, and
  `.omx/state/review_tracker.json`.
- `.venv/bin/python -m pytest src/tac/tests/test_audit_release_index_split.py -q`
  passed: `2 passed`.
- `.venv/bin/python -m ruff check tools/audit_release_index_split.py
  src/tac/tests/test_audit_release_index_split.py tools/all_lanes_preflight.py`
  passed.
- `.venv/bin/python -m py_compile tools/audit_release_index_split.py
  tools/all_lanes_preflight.py` passed.

## R41 - Remaining Tracked Source Promotion And Custody Split

Promoted the remaining useful tracked-source hardening while preserving local
custody artifacts:

- Staged the non-raw tracked source/docs changes under `docs/`, `experiments/`,
  `scripts/`, `src/`, `submissions/`, `tests/`, and `tools/`.
- Left private/provider state unstaged:
  `.omx/state/lightning_batch_jobs.json`,
  `.omx/state/review_tracker.duckdb`, and
  `.omx/state/review_tracker.json`.
- Left generated/rebuildable result metadata and public-intake gitlinks
  unstaged under `experiments/results/`.
- Left raw orphan-recovery/public-deconstruction snapshots unstaged under
  `reverse_engineering/orphan_pyc_recovery_20260505_codex/`; canonical source
  copies are already staged or tracked elsewhere.
- Extended `tools/audit_release_index_split.py` so these local-custody classes
  are visible as warnings, not invisible residue:
  - `unstaged_private_runtime_state`;
  - `unstaged_local_custody_snapshot`.
  Strict mode still blocks staged private runtime files and staged rollback
  shadows.

Verification:

- `python3 -m py_compile` over staged Python passed: `156 files`.
- `.venv/bin/python -m pytest src/tac/tests/test_qh0_record_serializer.py
  src/tac/tests/test_submission_archive_safety.py
  src/tac/tests/test_lightning_repro_workspace.py
  src/tac/tests/test_preflight_meta_bugs.py
  src/tac/tests/test_dispatch_dryrun_omega_w_v3.py
  src/tac/tests/test_lightning_dispatch_pr106_stack.py
  src/tac/tests/test_predispatch_sanity.py
  src/tac/tests/test_audit_release_index_split.py -q` passed:
  `321 passed, 1 warning`.
- `.venv/bin/python -m ruff check --select F821` over staged Python passed.
- Full all-lanes preflight passed: `ALL 16 PREFLIGHT CHECKS PASSED`.
- Full Ruff over every staged Python was not used as a release gate in this
  tranche because it still reports broad legacy style debt (`I001`, `RUF100`,
  `RUF001`, `SIM108`, etc.) outside the safety/runtime objective. The fatal
  undefined-name check and py_compile gate passed.

## R42 - Local Custody Manifest And Quiet Release Audit

Converted the remaining release-index warnings into explicit, dated custody
documentation instead of leaving them as noisy but unexplained dirty state:

- Added `.omx/research/local_custody_release_manifest_20260505_codex.json`
  with rule IDs for provider/runtime `.omx/state`, rebuildable Apogee repack
  metadata, public-PR intake gitlinks, raw Kaggle ingest, and private orphan
  recovery snapshots.
- Extended `tools/audit_release_index_split.py` to load the manifest, match
  warning records by kind/path/prefix, downgrade documented local custody to
  `info`, and report `documented_count` in JSON summaries.
- Wired the manifest into Gate #12 in `tools/all_lanes_preflight.py` so the
  all-lanes gate remains strict on blockers while staying quiet for custody
  classes that are intentionally preserved outside the public release surface.
- Added tests for gitlink worktree-status parsing and custody-rule
  documentation.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_audit_release_index_split.py -q`
  passed: `4 passed`.
- `.venv/bin/python -m ruff check tools/audit_release_index_split.py
  tools/all_lanes_preflight.py src/tac/tests/test_audit_release_index_split.py`
  passed.
- `.venv/bin/python -m py_compile tools/audit_release_index_split.py
  tools/all_lanes_preflight.py` passed.
- `.venv/bin/python tools/audit_release_index_split.py --strict
  --local-custody-manifest
  .omx/research/local_custody_release_manifest_20260505_codex.json` passed:
  `release index split: PASS (51 documented local custody record(s))`.
- JSON audit reported `blocker_count=0`, `warning_count=0`, and
  `documented_count=51`.
- Full `.venv/bin/python tools/all_lanes_preflight.py` passed:
  `ALL 16 PREFLIGHT CHECKS PASSED`.

## R43 - Staged Public Release Hygiene Gate

Added a release-surface guard that scans only staged public files, avoiding the
false choice between publishing dirty docs and erasing local forensic custody:

- Added `tools/audit_staged_public_release_hygiene.py`.
- The guard selects staged `ACMR` paths under public surfaces such as
  `AGENTS.md`, `docs/`, `reports/graphs/`, `reports/silent_defaults.md`, and
  curated `reverse_engineering/README.md` files.
- It delegates private-surface detection to
  `tac.preflight.check_public_release_hygiene`, so local absolute paths,
  provider job surfaces, and credential-like strings are blocked before public
  publication.
- It deliberately does not scan `.omx/state`, raw ledgers, raw report custody,
  or orphan-recovery snapshots. Those are governed by the R42 local-custody
  manifest and reverse-engineering release manifest instead.
- Wired the guard into `tools/all_lanes_preflight.py` as Gate #13.
- Added `src/tac/tests/test_audit_staged_public_release_hygiene.py` for public
  path selection, private operator path detection, and placeholder allowance.

Important diagnostic:

- A broad scan of all existing `reports/graphs/` surfaces found legacy generated
  site/timeline files with local paths and raw Modal app IDs. Those files were
  not part of the staged publish surface, so R43 does not rewrite them. The next
  release-site tranche should regenerate or sanitize the complete site bundle
  before publishing those generated artifacts.

Verification:

- `.venv/bin/python -m pytest
  src/tac/tests/test_audit_staged_public_release_hygiene.py -q` passed:
  `4 passed`.
- `.venv/bin/python -m ruff check
  tools/audit_staged_public_release_hygiene.py tools/all_lanes_preflight.py
  src/tac/tests/test_audit_staged_public_release_hygiene.py` passed.
- `.venv/bin/python -m py_compile
  tools/audit_staged_public_release_hygiene.py tools/all_lanes_preflight.py`
  passed.
- `.venv/bin/python tools/audit_staged_public_release_hygiene.py --strict
  --format json` reported `public_scan_path_count=20` and
  `violation_count=0`.
- `.venv/bin/python tools/audit_staged_public_release_hygiene.py --strict`
  passed: `staged public release hygiene: PASS (20 staged public file(s)
  scanned)`.
- First full preflight correctly failed Gate #9 because the new guard/test were
  still untracked. After staging them, full `.venv/bin/python
  tools/all_lanes_preflight.py` passed: `ALL 17 PREFLIGHT CHECKS PASSED`.

## R44 - Canonical Public Site Bundle Sanitizer

Promoted the recovered public-site sanitizer back into canonical source so the
Cloudflare Pages supplement can be rebuilt without publishing private custody
surfaces:

- Restored `reports/graphs/build_public_site_bundle.py` from the
  orphan-recovery tree into its original canonical location.
- Restored `reports/graphs/test_build_public_site_bundle.py` and cleaned its
  synthetic private-surface fixtures so the staged-public hygiene scanner does
  not see test data as a real leak.
- The bundler copies `reports/graphs/site/` into ignored
  `reports/graphs/public_site/`, redacts private operator paths, Lightning
  Studio URLs, Vast SSH endpoints, Modal IDs, and credential-like assignments,
  enforces an asset-size cap, writes `public_site_manifest.json`, and then runs
  strict `check_public_release_hygiene` on the generated bundle.

Generated-bundle result:

- Command: `.venv/bin/python reports/graphs/build_public_site_bundle.py
  --source reports/graphs/site --output reports/graphs/public_site
  --max-asset-bytes 25000000`
- Output bundle: `reports/graphs/public_site/` (ignored generated artifact).
- `file_count=71`.
- `hygiene_violation_count=0`.
- `redaction_count=3350`.
- Omitted one oversized asset:
  `comparison/comparison.gif` at `114724141` bytes.
- Public hygiene recheck over `reports/graphs/public_site` scanned `72` files
  and found `0` violations.

Verification:

- `.venv/bin/python -m pytest reports/graphs/test_build_public_site_bundle.py -q`
  passed: `4 passed`.
- `.venv/bin/python -m ruff check reports/graphs/build_public_site_bundle.py
  reports/graphs/test_build_public_site_bundle.py` passed.
- `.venv/bin/python -m py_compile reports/graphs/build_public_site_bundle.py`
  passed.
- `.venv/bin/python tools/audit_staged_public_release_hygiene.py --strict
  --format json` reported `public_scan_path_count=22` and
  `violation_count=0`.
- Full `.venv/bin/python tools/all_lanes_preflight.py` passed:
  `ALL 17 PREFLIGHT CHECKS PASSED`.

## R47 - Score-Band Anchor-Role Cleanup And Final Release Gates

The remaining unstaged source after R46 was a coherent predictor/calibration
cleanup, not disposable local state:

- Tagged the canonical `lane_apogee_int8` calibration anchor as
  `anchor_role=compatibility_only` because its archive is larger than PR106
  lossless (`187731` vs `186239` bytes) and is useful only as inflate-path
  compatibility evidence.
- Updated `src/tac/predictor/score_band.py` so compatibility-only anchors are
  loaded for context but excluded from lossy curve fitting and from the
  `lossy_anchor_invalid_no_rate_savings` refusal gate.
- Updated score-band and meta-Lagrangian tests to preserve the distinction
  between fit anchors and compatibility references.
- Softened the paper method text so Int4+LZMA2 is described as a target/frontier
  lane rather than an achieved canonical result.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_score_band_predictor.py
  src/tac/tests/test_meta_lagrangian.py -q` passed: `40 passed`.
- `.venv/bin/python -m ruff check src/tac/predictor/score_band.py
  src/tac/tests/test_score_band_predictor.py src/tac/tests/test_meta_lagrangian.py`
  passed.
- `.venv/bin/python -m py_compile src/tac/predictor/score_band.py` passed.
- Strict untracked-source audit passed with `untracked_source_like_count=0` and
  `undispositioned_count=0`.
- Staged-public hygiene reported `public_scan_path_count=28` and
  `violation_count=0`.
- `git diff --cached --check` passed.
- Full `.venv/bin/python tools/all_lanes_preflight.py` passed:
  `ALL 17 PREFLIGHT CHECKS PASSED`.

Next huge tranche: split the staged mega-diff into reviewable release commits
without losing custody, then continue reducing duplicated reverse-engineering
and analysis tooling into canonical `src/tac`, `tools`, `docs/runbooks`, and
`reverse_engineering` surfaces.

## R48 - Public PR Corpus Deduplication, HF Upload, And Senior Review Fixes

The raw full public-PR intake tree is intentionally forensic and large
(`experiments/results/public_pr_intake_full`, about 17 GB). Uploading it
directly duplicated fixed contest assets and local upload state, so the upload
path now materializes a canonical deduplicated release view first:

- Added `tools/materialize_pr_archive_release_view.py`.
- The release view keeps byte-exact scored `archive.zip` files, PR metadata,
  PR bodies, provenance, logs, filtered source mirrors, README, LICENSE, and
  `OMITTED_SHARED_ASSETS.json`.
- It omits repeated reconstructable assets: `source/videos/0.mkv`,
  fixed PoseNet/SegNet weights, `.git/**`, `.cache/**`, `__pycache__`, pyc
  files, vendored `ffmpeg-new`, vendored `libSvtAv1Enc.so*`, `.DS_Store`, and
  unexpectedly large non-archive files under `source/` pending review.
- The local release view generated during review contained `6064` included
  files and `139440192` included bytes before README/LICENSE copy, while
  omitting `12209` files and `18850993539` bytes.
- The generated release view is ignored in git; the materializer and upload
  script are tracked.

HF upload:

- Initial raw upload was interrupted because it was uploading duplicated videos,
  scorer models, ffmpeg/SVT binaries, and upload cache state.
- The partial HF dataset repo was deleted and recreated through the deduplicated
  upload flow.
- Clean upload completed to
  `https://huggingface.co/datasets/adpena/comma_video_compression_challenge_pr_archive`.
- Remote verification downloaded `FETCH_SUMMARY.json`,
  `OMITTED_SHARED_ASSETS.json`, and `README.md` from the dataset. The remote
  summary reports `total_attempted=54`, `n_complete=52`, `n_with_archive=52`,
  and `n_with_source=53`; manual triage remains PR #71 and PR #24.

Senior-review fixes:

- `tools/upload_pr_archive_to_hf.sh` now uses `hf repos create --exist-ok`
  instead of swallowing a 409 through a pipeline.
- The upload script removes `SOURCE_DIR/.cache` after successful
  `hf upload-large-folder` because HF intentionally writes resumability
  metadata inside the uploaded folder.
- `docs/comma_pr_archive_dataset_card.md` was normalized to ASCII for portable
  public rendering.
- `src/tac/tests/test_preflight_meta_bugs.py` had its new and legacy
  mid-file imports consolidated so the touched file passes Ruff directly.

Kaggle decision:

- Hugging Face remains canonical because the corpus is provenance/file heavy,
  uses resumable large-folder upload, and keeps Git/LFS-backed dataset history.
- Kaggle is useful as a secondary discoverability/notebook mirror once the HF
  release view is finalized. The Kaggle mirror should upload the same
  deduplicated release view plus `dataset-metadata.json`, never the raw
  forensic intake tree.

Verification:

- `find experiments/results/public_pr_archive_release_view ...` found no
  leaked `.cache`, `.git`, repeated video, fixed scorer weights, vendored
  ffmpeg/SVT, or pycache files after regeneration.
- `.venv/bin/python tools/audit_untracked_source_artifacts.py --strict
  --disposition-manifest .omx/research/untracked_source_dispositions_20260505_codex.json
  --format json` passed with `untracked_source_like_count=0`.
- `.venv/bin/python tools/audit_staged_public_release_hygiene.py --strict
  --format json` passed with `public_scan_path_count=31` and
  `violation_count=0`.
- `git diff --cached --check` passed.
- `bash -n tools/upload_pr_archive_to_hf.sh` passed.
- `.venv/bin/python -m py_compile tools/materialize_pr_archive_release_view.py
  tools/fetch_all_public_pr_archives.py
  tools/audit_staged_public_release_hygiene.py src/tac/preflight.py` passed.
- `.venv/bin/python -m ruff check src/tac/tests/test_preflight_meta_bugs.py
  tools/materialize_pr_archive_release_view.py tools/fetch_all_public_pr_archives.py
  tools/audit_staged_public_release_hygiene.py
  src/tac/tests/test_audit_staged_public_release_hygiene.py` passed.
- `.venv/bin/python -m pytest
  src/tac/tests/test_audit_staged_public_release_hygiene.py
  src/tac/tests/test_preflight_meta_bugs.py::TestKlDivReductionCorrect::test_vendored_public_intake_tree_is_skipped
  src/tac/tests/test_preflight_meta_bugs.py::TestPreflightAllInvokesMetaBugChecks
  -q` passed: `7 passed`.

Next huge tranche: split the staged mega-diff into reviewable commits, then add
the Kaggle mirror generator (`dataset-metadata.json` + upload wrapper) using the
same deduplicated release view and no raw-custody uploads.

## R49 - Kaggle Mirror View Generator

Implemented the Kaggle secondary-mirror path without mutating the canonical HF
release view:

- Added `tools/materialize_pr_archive_kaggle_mirror.py`.
- Added `tools/upload_pr_archive_to_kaggle.sh`.
- Added `src/tac/tests/test_materialize_pr_archive_kaggle_mirror.py`.
- Updated `.gitignore` so
  `experiments/results/public_pr_archive_kaggle_mirror/` stays generated/local.
- Updated `docs/comma_pr_archive_dataset_card.md` to list the canonical HF and
  Kaggle mirror tooling.

Design:

- HF remains canonical for file/provenance-heavy custody.
- Kaggle is a secondary discoverability/notebook mirror.
- The Kaggle view consumes `experiments/results/public_pr_archive_release_view`
  and adds `dataset-metadata.json` plus `KAGGLE_MIRROR_MANIFEST.json`.
- The mirror generator skips `.cache`, `.git`, `__pycache__`, pyc files,
  `.DS_Store`, and pre-existing `dataset-metadata.json`.
- The upload wrapper materializes the HF release view first, then creates the
  Kaggle-specific view, then runs either `kaggle datasets create -p ...` or
  `kaggle datasets version -p ... -m ...`.

Local generated mirror:

- Command: `.venv/bin/python tools/materialize_pr_archive_kaggle_mirror.py --force`.
- Output: `experiments/results/public_pr_archive_kaggle_mirror/`.
- Included source-view files: `6065`.
- Included source-view bytes: `141846639`.
- Skipped source-view files: `0` because the HF release view was already clean.
- Final local mirror size: about `150M`.
- Leak scan found no `.cache`, `.git`, repeated contest video, fixed scorer
  weights, vendored ffmpeg/SVT, or pycache files.

Verification:

- `.venv/bin/python -m pytest
  src/tac/tests/test_materialize_pr_archive_kaggle_mirror.py -q` passed:
  `4 passed`.
- `.venv/bin/python -m ruff check
  tools/materialize_pr_archive_kaggle_mirror.py
  src/tac/tests/test_materialize_pr_archive_kaggle_mirror.py` passed.
- `.venv/bin/python -m py_compile
  tools/materialize_pr_archive_kaggle_mirror.py` passed.
- `bash -n tools/upload_pr_archive_to_kaggle.sh` passed.

Next huge tranche: run the full release gates again with R49 staged, then split
the staged mega-diff into reviewable commits before any public repo push.

## R45 - Leaderboard And Public-PR Intake Tool Recovery

The final release sanity pass found two untracked source tools and one
leaderboard fixture/test pair. These are useful post-contest research tools, so
they were canonicalized rather than ignored:

- Staged `tools/fetch_all_public_pr_archives.py`, which fetches public scored
  PR metadata, bodies, archives when recoverable, source checkouts, and
  provenance summaries into `experiments/results/`.
- Staged `tools/leaderboard_poll.py`, which hashes the upstream leaderboard
  score column and emits a race-mode signal when scores move.
- Staged `src/tac/tests/test_leaderboard_poll.py` and
  `src/tac/tests/fixtures/leaderboard_readme_20260505.md`, preserving the
  exact final-top-band README fixture needed for stable parser/hash tests.

Verification:

- `.venv/bin/python -m ruff check tools/fetch_all_public_pr_archives.py
  tools/leaderboard_poll.py` initially found lint debt in the recovered tools;
  `ruff --fix` repaired it, and the focused Ruff check then passed.
- `.venv/bin/python -m py_compile tools/fetch_all_public_pr_archives.py
  tools/leaderboard_poll.py` passed.
- `.venv/bin/python tools/fetch_all_public_pr_archives.py --only-prs 101,103
  --dry-run` passed and printed the intended manual fetch plan.
- `.venv/bin/python -m pytest src/tac/tests/test_leaderboard_poll.py -q`
  passed: `12 passed`.
- `.venv/bin/python -m ruff check src/tac/tests/test_leaderboard_poll.py
  tools/leaderboard_poll.py tools/fetch_all_public_pr_archives.py` passed.
- Strict untracked-source audit passed with `untracked_source_like_count=0` and
  `undispositioned_count=0`.
- Staged-public hygiene remained clean: `public_scan_path_count=22`,
  `violation_count=0`.
- Full `.venv/bin/python tools/all_lanes_preflight.py` passed:
  `ALL 17 PREFLIGHT CHECKS PASSED`.

## R46 - Optimizer Plugin, Example, And Paper-Doc Source Closure

The no-signal-loss gate exposed additional untracked source after R45. These
were real source/docs, not disposable artifacts, so they were canonicalized:

- Staged `src/tac/optimizer/sweep_plugin.py`, the generic candidate-generator
  plugin interface for closed-loop sweep workloads.
- Staged `src/tac/optimizer/generators/__init__.py` and
  `src/tac/optimizer/generators/apogee_intn.py`, the extracted Apogee intN
  concrete generator.
- Added/staged `src/tac/tests/test_optimizer_sweep_plugin.py` and staged the
  existing `src/tac/tests/test_sweep_plugin.py` coverage for registry,
  duplicate replacement, dispatch spec defaults, Apogee intN schema, and the
  synthetic plugin example.
- Staged `examples/synthetic_sweep.py`, a non-comma example showing the plugin
  pattern without contest-specific imports.
- Staged `docs/posts/cron_loop_accumulated_rigor_failure_mode.md`, preserving
  the postmortem writeup on cron-as-fan-out-cadence.
- Staged `docs/paper/00_abstract.md` and
  `docs/paper/figures/MISSING.md`, preserving paper source and arxiv-readiness
  figure TODOs.
- Updated `.gitignore` so generated local custody from leaderboard polling and
  full public-PR archive intake stays out of public source commits:
  `.omx/state/*.jsonl`, `.omx/state/RACE_MODE_ACTIVE.flag`, and
  `experiments/results/public_pr_intake_full/`.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_sweep_plugin.py
  src/tac/tests/test_optimizer_sweep_plugin.py -q` passed: `13 passed`.
- `.venv/bin/python -m pytest
  src/tac/tests/test_optimizer_sweep_plugin.py src/tac/tests/test_leaderboard_poll.py
  reports/graphs/test_build_public_site_bundle.py
  src/tac/tests/test_audit_staged_public_release_hygiene.py
  src/tac/tests/test_audit_release_index_split.py -q` passed: `26 passed`.
- Focused Ruff checks for optimizer plugin, leaderboard tools, public-site
  bundler, and tests passed.
- `.venv/bin/python examples/synthetic_sweep.py` passed and printed the
  expected synthetic generator/ranking/dispatch-spec smoke output.
- Strict untracked-source audit passed with `untracked_source_like_count=0` and
  `undispositioned_count=0`.
- Staged-public hygiene reported `public_scan_path_count=25` and
  `violation_count=0`.
- `git diff --cached --check` passed.
- Full `.venv/bin/python tools/all_lanes_preflight.py` passed:
  `ALL 17 PREFLIGHT CHECKS PASSED`.

## R50 - 2026-05-06 Dirty Nested-Custody Classification Refresh

Focused dirty-state custody refresh on `main` after the HNeRV wavelet residual
planning commit, without editing the active HNeRV sidechannel implementation or
mutating raw custody snapshots.

Current dirty-state classes:

- Eight public-PR intake gitlinks remain dirty local forensic snapshots:
  `experiments/results/public_pr100_intake_20260504_codex/source`,
  `public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source`,
  `public_pr103_intake_20260504_codex/source`,
  `public_pr105_kitchen_sink_intake_20260504_codex/source`,
  `public_pr106_belt_and_suspenders_intake_20260504_codex/source`,
  `public_pr81_qzs3_range_mask_intake_20260503_codex/repo`,
  `public_pr82_henosis_frontier_intake_20260503_codex/repo`, and
  `public_pr91_intake_20260504_worker/pr91_src/repo`.
- The common public-PR nested dirt is local modifications to recovered public
  submission training/compression files under `submissions/fp4_mask_gen`,
  `submissions/neural_inflate`, `submissions/quantizr`, and
  `submissions/svtav1_dilated_ren`; PR100/101/103/105/106/81 also have
  `submissions/ph4ntom_drv/compress.py` dirty.
- One raw Kaggle ingest gitlink remains dirty local custody at
  `reports/raw/kaggle_ingest/kaggle-dilated-h64-long1000-retry-v6-20260410T234220Z/comma_video_compression_challenge`,
  with untracked `submissions/gt_passthrough/inflate.py`,
  `submissions/gt_passthrough/inflate.sh`, and
  `submissions/gt_passthrough/report_pyav.txt` inside the nested repo.
- Thirty-four modified files remain under
  `reverse_engineering/orphan_pyc_recovery_20260505_codex/`. These are still
  private/local recovery snapshots and are covered by
  `orphan_recovery_snapshots_private`; useful source should continue to promote
  through canonical `src/tac`, `tools`, `docs`, or curated
  `reverse_engineering` paths before any quarantine cleanup.

Audit evidence:

- `.venv/bin/python tools/audit_nested_gitlink_custody.py --repo-root . --strict --local-custody-manifest .omx/research/local_custody_release_manifest_20260505_codex.json --format json`
  passed with `dirty_gitlink_count=9`, `documented_count=9`, and
  `warning_count=0`.
- `.venv/bin/python tools/audit_release_index_split.py --repo-root . --strict --local-custody-manifest .omx/research/local_custody_release_manifest_20260505_codex.json --format json`
  passed with `record_count=43`, `documented_count=43`, `blocker_count=0`, and
  `warning_count=0`.
- `.venv/bin/python tools/audit_untracked_source_artifacts.py --repo-root . --strict --format json`
  passed with `untracked_source_like_count=0` and `undispositioned_count=0`.
- `.venv/bin/python tools/audit_orphan_recovery_canonicalization.py --repo-root . --strict --format json`
  passed with `source_like_delete_count=0`, `unstaged_delete_count=0`, and
  `missing_canonical_count=0`.
- `.venv/bin/python tools/audit_reverse_engineering_tree.py --repo-root . --summary`
  passed with `files=716` and `blockers=0`.
- `.venv/bin/python tools/audit_reverse_engineering_tree.py --repo-root . --release-strict --release-manifest .omx/research/reverse_engineering_release_manifest_20260505_codex.json --summary`
  passed with `files=716` and `blockers=0`.

Conclusion: the remaining dirty state is intentionally local custody, not a
release blocker, provided release/preflight flows keep passing the explicit
local-custody manifest. No score, dispatch, or promotion claim is made from
these dirty snapshots.

## R51 - 2026-05-06 Orphan Modified-Copy Inventory Guard

Hardened `tools/audit_orphan_recovery_canonicalization.py` so the orphan
recovery tree is not only guarded at deletion time. The audit now inventories
modified source-like copies under
`reverse_engineering/orphan_pyc_recovery_20260505_codex/`, maps each one to
the canonical repo path that would result from stripping the recovery prefix,
and records whether that canonical path is already tracked.

Default behavior remains advisory for modified copies so normal preflight does
not block the working tree while the recovery queue is still being triaged.
Strict recovery mode is available with `--fail-on-shadowed-modified` and fails
closed on both classes that can lose signal:

- modified orphan copy shadows a tracked canonical file and must be diffed or
  deleted intentionally;
- modified orphan copy has no tracked canonical home and must be promoted,
  archived, or explicitly dispositioned before cleanup.

Current evidence:

- `.venv/bin/python -m pytest src/tac/tests/test_audit_orphan_recovery_canonicalization.py -q`
  passed: `3 passed`.
- `.venv/bin/python -m py_compile tools/audit_orphan_recovery_canonicalization.py`
  passed.
- `.venv/bin/python tools/audit_orphan_recovery_canonicalization.py --format json`
  passed in advisory mode with `source_like_delete_count=0`,
  `source_like_modified_count=34`, `shadowed_modified_count=3`, and
  `modified_missing_canonical_count=31`.
- `.venv/bin/python tools/audit_orphan_recovery_canonicalization.py --fail-on-shadowed-modified --format json`
  failed closed as intended with 34 blockers: 3 tracked-canonical shadows and
  31 missing-canonical modified copies.

Interpretation: the repo now has an executable no-signal-loss inventory for
the remaining orphan recovery modified copies. Next cleanup passes should use
the strict output as the queue: canonicalize or intentionally disposition the
three tracked shadows first, then decide which of the 31 missing-canonical
copies belong under `reverse_engineering/`, `tools/`, `src/tac/`, or a private
custody manifest.

## R52 - 2026-05-06 First Tracked-Shadow Canonicalization

Resolved the first strict orphan-recovery shadow class by hand-reviewing the
three modified orphan copies that mapped onto tracked canonical files:

- `reverse_engineering/orphan_pyc_recovery_20260505_codex/reports/graphs/build_public_site_bundle.py`
  mapped to `reports/graphs/build_public_site_bundle.py`.
- `reverse_engineering/orphan_pyc_recovery_20260505_codex/reports/graphs/test_build_public_site_bundle.py`
  mapped to `reports/graphs/test_build_public_site_bundle.py`.
- `reverse_engineering/orphan_pyc_recovery_20260505_codex/scripts/pre_submission_compliance_check.py`
  mapped to `scripts/pre_submission_compliance_check.py`.

Decision: keep the canonical tracked files and delete the orphan duplicates.
The orphan copies were older recovered source snapshots with pass2 comments and
without newer production/release hardening already present in canonical code.
The canonical public-site bundle keeps the private `comma-lab` URL rewrite,
public-link audit, external-path manifest redaction, `src` import bootstrap,
and tests for final manifest/link hygiene. The canonical pre-submission gate
keeps `tools.tool_bootstrap`, `tac.repo_io` JSON/SHA/path helpers, and the
current terminal-dispatch row parser.

No source signal is lost by deleting these three duplicates: their canonical
source paths are tracked, stricter, tested, and already contain the useful
recovered behavior plus later hardening. The remaining orphan modified-copy
queue should drop from `shadowed_modified_count=3` to zero after these staged
deletions, leaving only missing-canonical recovery decisions.

## R53 - 2026-05-06 Remaining Missing-Canonical Queue Classification

After the first tracked-shadow cleanup, the recovery audit reports
`shadowed_modified_count=0`, `source_like_modified_count=31`, and
`modified_missing_canonical_count=31`. These are no longer duplicates of
tracked canonical files; they require explicit promotion, archival, or
disposition decisions.

Current buckets:

- 2 root reverse-engineering tools:
  `experiments/build_pr85_qh0_serializer_candidates.py` and
  `experiments/replay_pr91_hpm1_mask.py`.
- 21 public-submission runtime/replay fragments:
  PR96/PR98/PR99 HNeRV runtime files, PR85/STBM replay runtime, PR65/PR67
  adapter inflates, PR85/PR86 inflates, PR90 qrepro codec/range/sparse
  scripts, and PR95 HNeRV inflate/model files.
- 1 renderer self-compression search artifact:
  `experiments/results/renderer_selfcompression_nextwave_worker_20260503/c101_combined_zero_search.py`.
- 2 operational scripts:
  `scripts/build_contest_submission_packet.py` and
  `scripts/q_faithful_snapshot_loop.py`.
- 4 recovered tests:
  `src/tac/tests/test_endgame_archive_decision.py`,
  `src/tac/tests/test_pr85_bundle.py`,
  `src/tac/tests/test_quantizr_torch_fp4_codec.py`, and
  `src/tac/tests/test_qzs3_postprocess_qrm1_runtime.py`.
- 1 submission runtime helper:
  `submissions/robust_current/apply_qzs3_postprocess.py`.

Recommended next promotion order:

1. Promote the four recovered tests first if they still encode useful guard
   behavior; tests are the safest signal to canonicalize and will expose any
   stale assumptions.
2. Promote or archive PR95/PR90 public runtime fragments under a clean
   `reverse_engineering/public_frontier/` path, not under `experiments/results`,
   because they are forensic reference code rather than active experiment
   outputs.
3. Diff the two operational scripts against current OSS/release flows before
   promotion; if they are stale, keep only their recovery specs and ledger
   notes.
4. Treat `apply_qzs3_postprocess.py` as high-risk because it shadows the active
   robust runtime conceptually even though its canonical path is not tracked.
   Promote only after direct runtime-parity review.

## R54 - 2026-05-06 Recovered Test And QRM1 Runtime Promotion

Promoted the recovered test/helper bucket from R53 into canonical tracked
locations:

- `src/tac/tests/test_endgame_archive_decision.py`
- `src/tac/tests/test_pr85_bundle.py`
- `src/tac/tests/test_quantizr_torch_fp4_codec.py`
- `src/tac/tests/test_qzs3_postprocess_qrm1_runtime.py`
- `submissions/robust_current/apply_qzs3_postprocess.py`

The first combined recovered-test run was red (`25 failed, 8 passed,
1 skipped`). The red state exposed three useful bug classes:

- `tac.henosis_pr82_transfer.encode_randmulti_qrm1` was still a partial
  rehydration stub while the QRM1 runtime tests needed a deterministic sparse
  row encoder. Implemented `_encode_randmulti_rows` and
  `encode_randmulti_qrm1` against the current `Pr82RandmultiGroup` contract.
- `decode_pr85_p1d1_pose_to_fp16` returned concatenated active P1D1 streams
  instead of the full `600 x 6` raw fp16 `optimized_poses.bin` contract.
  Reimplemented P1D1 delta/VLQ decode with the public PR91 semantics: dim 0
  uses `q / 512 + 20`, other dims use clipped `q / 2048`, and missing dims
  remain zero.
- `tac.quantizr_torch_fp4_codec` only decoded the canonical in-repo
  Torch-FP4 payload shape. It now also decodes the public PR63-style
  `packed_weight`/`scales_fp16` and `weight_fp16` quantized-entry shape.

Strict xfails preserve recovered but currently unimplemented/stale signal:

- `test_endgame_archive_decision.py`: the underlying
  `tac.endgame_archive_decision` module remains a partial pyc rehydration
  stub.
- `test_current_pr92_rmb1_randmulti_is_decoded_row_parity_recode`: the local
  public PR92 fixture still uses a stale RMB1 runtime shape.
- `test_repack_builder_emits_torch_fp4_archive_from_qfai`: the active repack
  builder exposes QZS3/QZS4 only; Torch-FP4 builder promotion is not wired.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_endgame_archive_decision.py
  src/tac/tests/test_pr85_bundle.py
  src/tac/tests/test_quantizr_torch_fp4_codec.py
  src/tac/tests/test_qzs3_postprocess_qrm1_runtime.py -q`
  -> `28 passed, 1 skipped, 5 xfailed`.
- `.venv/bin/python -m py_compile src/tac/henosis_pr82_transfer.py
  src/tac/pr85_bundle.py src/tac/quantizr_torch_fp4_codec.py
  submissions/robust_current/apply_qzs3_postprocess.py` passed.
- Non-strict orphan audit reports `ready_for_orphan_recovery_cleanup=true`
  with `shadowed_modified_count=0`; strict audit still fails, as intended,
  because 31 missing-canonical orphan copies remain for explicit promotion or
  archival decisions.

The PR106 yshift Lightning score-table job was checked from local state during
this tranche and remains `Running`; no harvest or claim closure was performed.

## R55 - 2026-05-06 First-Party Experiment CLI Promotion

Promoted two recovered first-party experiment wrappers out of the orphan
intake and back into their original canonical `experiments/` locations:

- `experiments/build_pr85_qh0_serializer_candidates.py`
- `experiments/replay_pr91_hpm1_mask.py`

The PR85 QH0 serializer candidate builder compiles and exposes a clean help
surface. It remains local-only byte-decision tooling: it rewrites PR85 model
segments, records candidate manifests, and does not run CUDA, dispatch, or
claim score evidence.

The PR91 HPM1 replay wrapper initially had hard imports for codec functions
that are still absent from the current partial `tac.pr91_hpm1_codec`
rehydration. That made even `--help` fail. The wrapper now imports the module
itself, exposes constants from the implemented surface, and lazily resolves
optional probe/fusion symbols only when the corresponding mode is requested.
Missing modes fail closed with an explicit message naming the missing codec
function instead of crashing at import time.

Verification:

- `.venv/bin/python -m py_compile
  experiments/build_pr85_qh0_serializer_candidates.py
  experiments/replay_pr91_hpm1_mask.py` passed.
- `.venv/bin/python experiments/build_pr85_qh0_serializer_candidates.py --help`
  passed.
- `.venv/bin/python experiments/replay_pr91_hpm1_mask.py --help` passed.
- `experiments/replay_pr91_hpm1_mask.py --fusion-plan` now reaches an
  intentional fail-closed message because
  `tac.pr91_hpm1_codec.plan_pr91_hpm1_pr85_stbm_fusion` is not currently
  implemented.

Non-strict orphan audit after this promotion reports
`modified_missing_canonical_count=26` while the two active renames are staged;
after commit the remaining queue should be the 24 public-runtime/operational
fragments that still require explicit public-frontier archival or promotion.
