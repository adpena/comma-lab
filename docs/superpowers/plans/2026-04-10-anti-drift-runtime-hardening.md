# Anti-Drift Runtime Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make promoted score/report state self-healing and boundary-hardened, then move cloud/runtime wrappers toward `tac` as the canonical semantic path.

**Architecture:** Add a canonical promoted-result state model plus deterministic projection/sync utilities under `src/comma_lab`, wire CLI commands for `state doctor`, `state sync`, and `promote`, then use those surfaces to repair the live `1.33` split-brain regression. After that, introduce thin `tac`-backed cloud entry helpers so wrappers stop owning training semantics.

**Tech Stack:** Python stdlib, existing `src/comma_lab` CLI/scheduler patterns, `src/tac`, unittest, atomic filesystem writes.

---

### Task 1: Canonical promoted-result model and projections

**Files:**
- Create: `src/comma_lab/state_models.py`
- Create: `src/comma_lab/state_sync.py`
- Test: `experiments/test_state_sync.py`

- [ ] **Step 1: Write failing tests for canonical record parsing and projection**

Create tests that:
- load a canonical promoted-result fixture
- project summary JSON
- project latest markdown snippets
- reproduce the `1.51` summary vs `1.33` ledger regression

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python3 -m unittest experiments.test_state_sync -v`
Expected: FAIL because the new model/sync utilities do not exist yet.

- [ ] **Step 3: Implement the canonical state model**

Add typed structures for:
- promoted result record
- named accounting views
- drift finding
- projection bundle

Ensure the model can be built from authoritative report fields plus provenance.
Include:
- schema version
- digests for artifact/report bindings
- explicit separation between mutable promoted pointer and append-only history identities

- [ ] **Step 4: Implement deterministic projection helpers**

Implement functions that:
- produce canonical summary JSON
- produce canonical current report copy instructions
- detect stale mirrors
- merge append-only ledgers by stable identity instead of regenerating unrelated history
- write atomically

- [ ] **Step 5: Re-run the state-sync tests**

Run: `python3 -m unittest experiments.test_state_sync -v`
Expected: PASS.

### Task 2: CLI surfaces for doctor, sync, and promote

**Files:**
- Modify: `src/comma_lab/cli.py`
- Test: `experiments/test_state_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Cover:
- `comma-lab state doctor`
- `comma-lab state sync`
- `comma-lab promote <record>`

- [ ] **Step 2: Run the CLI tests to verify they fail**

Run: `python3 -m unittest experiments.test_state_cli -v`
Expected: FAIL because these subcommands do not exist.

- [ ] **Step 3: Implement `state doctor`**

Add read-only drift reporting for:
- stale promoted summary/report
- stale latest markdown
- stale local managed-session manifests whose processes are gone

- [ ] **Step 4: Implement `state sync`**

Add the sync command to:
- read canonical promoted truth
- repair mirrored surfaces
- print drift findings before rewrite

- [ ] **Step 5: Implement `promote` boundary gate**

Add a promotion command that:
- validates authoritative report and artifact
- verifies stored digests
- syncs all mirrors
- fails hard on irreconcilable evidence

- [ ] **Step 6: Re-run the CLI tests**

Run: `python3 -m unittest experiments.test_state_cli -v`
Expected: PASS.

### Task 3: Repair the live repo drift and stale managed-session manifests

**Files:**
- Modify: `reports/raw/robust_current-current_workflow-cpu-summary.json`
- Modify: `reports/raw/robust_current-current_workflow-cpu-report.txt`
- Modify: `reports/latest.md`
- Modify: `.omx/state/current_focus.md`
- Modify: `.omx/state/next_experiments.md`
- Modify: `.omx/research/findings.md`
- Modify: `.ralph/run_log.md`
- Modify: `.omx/logs/remote_jobs/local-modal-dilated-h64-authoritative-eval.json`
- Modify: `.omx/logs/remote_jobs/local-modal-dilated-h64-proxy.json`

- [ ] **Step 1: Run `state doctor` against the live repo**

Run: `python3 -m src.comma_lab.cli state doctor --repo-root /Users/adpena/Projects/pact`
Expected: it reports the live `1.51` vs `1.33` split-brain and stale managed-session records.

- [ ] **Step 2: Run `state sync` to repair canonical mirrors**

Run: `python3 -m src.comma_lab.cli state sync --repo-root /Users/adpena/Projects/pact`
Expected: canonical summary and markdown/state docs update to `1.33`.

- [ ] **Step 3: Mark stale local managed-session manifests by observed fact**

Update managed-session manifests whose processes are gone to a terminal status such as completed/stale with notes pointing at the authoritative evidence.

- [ ] **Step 4: Re-run `state doctor`**

Run: `python3 -m src.comma_lab.cli state doctor --repo-root /Users/adpena/Projects/pact`
Expected: no canonical drift findings remain.

### Task 4: Publish/report boundary hardening

**Files:**
- Modify: `reports/graphs/build_static_site.py`
- Test: `reports/graphs/test_build_static_site.py`

- [ ] **Step 1: Write a failing publish-gate test**

Create a test where canonical promoted truth and mirrored surfaces disagree.

- [ ] **Step 2: Run the publish-gate test to verify it fails**

Run: `python3 -m unittest reports.graphs.test_build_static_site -v`
Expected: FAIL until the gate is added.

- [ ] **Step 3: Add a preflight drift check to static-site build**

Fail fast when promoted mirrors are stale relative to canonical truth.

- [ ] **Step 4: Re-run the publish tests**

Run: `python3 -m unittest reports.graphs.test_build_static_site -v`
Expected: PASS.

### Task 5: `tac`-owned cloud semantics

**Files:**
- Create: `src/tac/entrypoints.py`
- Modify: `src/tac/__init__.py`
- Modify: `experiments/train_postfilter_dilated_h64.py`
- Modify: `experiments/cloud_segnet_attack_h32_trainer.py`
- Test: `experiments/test_tac_entrypoints.py`
- Test: `experiments/test_train_postfilter_dilated_h64.py`
- Test: `experiments/test_cloud_segnet_attack_h32_trainer.py`

- [ ] **Step 1: Write failing tests for `tac`-owned trainer entry helpers**

Add tests that prove wrappers derive trainer behavior from `tac`, not embedded duplicated semantics.

- [ ] **Step 2: Run the new and existing wrapper tests to verify failures**

Run: `python3 -m unittest experiments.test_tac_entrypoints experiments.test_train_postfilter_dilated_h64 experiments.test_cloud_segnet_attack_h32_trainer -v`
Expected: FAIL until the shared entry helpers exist.

- [ ] **Step 3: Implement `src/tac/entrypoints.py`**

Add thin builder helpers for:
- standard/dilated/segnet cloud training configs
- checkpoint/output path conventions
- resume-state save/load locations
- schema version tags and compatibility adapters for emitted metadata

- [ ] **Step 4: Refactor cloud wrappers to call into `tac`**

Keep platform bootstrap local, but move training semantics and output conventions behind shared `tac` helpers.

- [ ] **Step 5: Re-run cloud wrapper tests**

Run: `python3 -m unittest experiments.test_tac_entrypoints experiments.test_train_postfilter_dilated_h64 experiments.test_cloud_segnet_attack_h32_trainer -v`
Expected: PASS.

### Task 6: Full verification and site rebuild

**Files:**
- No new files; verify modified surfaces

- [ ] **Step 1: Run state and CLI verification**

Run:
`python3 -m unittest experiments.test_state_sync experiments.test_state_cli experiments.test_scheduler_registry experiments.test_scheduler_cli -v`

- [ ] **Step 2: Run cloud/wrapper verification**

Run:
`python3 -m unittest experiments.test_tac_entrypoints experiments.test_train_postfilter_dilated_h64 experiments.test_cloud_segnet_attack_h32_trainer experiments.test_kaggle_status_sync experiments.test_kaggle_output_ingest experiments.test_kaggle_queue_tick -v`

- [ ] **Step 3: Rebuild and verify report surfaces**

Run:
`python3 reports/graphs/build_report_history.py`
`python3 reports/graphs/build_static_site.py`
`python3 reports/graphs/build_static_site.py --check`

- [ ] **Step 4: Run diff sanity checks**

Run:
`git diff --check -- docs/superpowers/specs/2026-04-10-anti-drift-runtime-design.md docs/superpowers/plans/2026-04-10-anti-drift-runtime-hardening.md src/comma_lab src/tac experiments reports .omx .ralph`

- [ ] **Step 5: Summarize final drift status**

Report:
- canonical promoted score
- remaining stale surfaces, if any
- remaining runtime wrappers not yet fully `tac`-backed
