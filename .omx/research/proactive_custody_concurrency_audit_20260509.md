# Proactive META-class custody + concurrency audit (2026-05-09)

<!-- generated_at: 2026-05-09T11:30:00Z, from_state_hash: proactive_audit_lane_proactive_custody_concurrency_audit -->

Operator-approved 2026-05-09 ("proceed with this... proactive sweep for similar
patterns elsewhere in repo"). Sister of `a44467a` (codex round-2 fix lane).

Scope:

- `src/tac/**/*.py`
- `tools/*.py`
- `experiments/*.py` and `experiments/**/*.py`
- `scripts/*.sh` and `scripts/*.py`

Exclusions:
- `experiments/results/public_pr*_intake_*/` (vendored, Check 109)
- `experiments/results/comma_lab_public_export/` (committed binary export)
- `experiments/results/vast_harvest/` (vendored uv envs)
- `upstream/` (pinned upstream snapshot)
- `.venv/`, `__pycache__/`, `node_modules/`
- `**/tests/**` (test files exercise the API legitimately)

## §1 Pattern 1 — tag-only-custody-check findings

A "tag-only-custody-check" is a code site that decides custody/promotion by
inspecting an evidence tag (or grade) string WITHOUT jointly validating
axis + hardware_substrate + required metadata.

### §1.1 Quantitative result

After exhaustive sweep against `src/tac/`, `tools/`, `experiments/`, `scripts/`:

| Severity | Count | Notes |
|---|---:|---|
| HIGH | 0 | No new sites use a tag-only predicate as a promotion gate |
| MEDIUM | 0 | All grade-string conditionals are FAIL-CLOSED (refuse promotion unless grade matches) and run alongside joint axis/SHA/auth_json validators |
| LOW | 0 | None |

**Total live count: 0.**

### §1.2 Why the count is zero

The repo's promotion paths already route through joint validators:

1. `tac.continual_learning.validate_custody_verdict` — typed `CustodyVerdict`
   with 6-class `refused_class` taxonomy (catalog #127, codex round-2 HIGH 2 fix).
2. `tac.optimization.candidate_evidence_contract.is_promotable_exact_cuda_evidence`
   — checks `evidence_grade` AND `archive_sha256` AND `score_contest_cuda`
   AND axis fields jointly via `promotable_exact_cuda_evidence_blockers`.
3. `tools/predispatch_sanity.py` — `_validate_non_proxy_readiness_evidence`
   computes `sha256_file(archive_path)` AND validates `evidence_semantics`
   AND checks `ready_for_exact_eval_dispatch` BEFORE checking
   `evidence_grade not in {"invalid", "external", "prediction"}`.

The remaining `evidence_grade` conditional sites in production code
(`tools/lightning_dispatch_pr106_stack.py:163`, `tools/predispatch_sanity.py:450`,
`experiments/build_pr85_bridge_sparse_action_candidates.py:696`,
`src/tac/uniward_delta.py:747`, `tools/build_pr107_cpu_lossy_candidate_matrix.py:512`)
are FAIL-CLOSED — they only ADD blockers when the grade doesn't match,
they never promote on a tag-membership-only basis.

### §1.3 Conclusion

Catalog #127's existing scope (`AUTHORITATIVE_TAGS` membership-checks routed
through the validator) covers the ENTIRE repo's bug-class surface. Pattern 1
is structurally extinct. No new STRICT preflight check is required for this
class beyond the existing #127.

The proposed catalog #130 (`check_no_tag_only_custody_validation`) widens
catalog #127's scope to also cover the `evidence_grade` set-membership
pattern and the `tag.startswith("[contest-...")` pattern. It lands at live
count 0 (held warn-only initially per the directive).

## §2 Pattern 2 — bare-write-on-shared-state findings

A "bare-write-on-shared-state" is a write to a `.omx/state/*.json` (or
similar shared mutable state file) WITHOUT an `fcntl.flock(LOCK_EX)` +
reload-inside-lock + unique tmp suffix + fsync + os.replace transactional
contract.

### §2.1 Quantitative result

After STRICT sweep (path-resolution variable tracking + line-local lock-context
scan):

| Severity | Pre-fix count | Post-fix count | Notes |
|---|---:|---:|---|
| HIGH | 7 | 0 | All 7 bare writes patched to canonical fcntl helpers |
| MEDIUM | 0 | 0 | — |
| LOW | 0 | 0 | — |

**Total live count post-fix: 0.**

### §2.2 Detailed findings (all FIXED in this lane)

Sorted by criticality (concurrency exposure × shared-state semantics):

| # | File:line | Shared state | Severity | Fix |
|---|---|---|---|---|
| 1 | `experiments/arch_shrink_x0.4_lightning_full.py:745` | `LIGHTNING_ACTIVE_JOBS_PATH` | HIGH | Routed through `tac.deploy.lightning.active_jobs_state.register_job` |
| 2 | `experiments/arch_shrink_x0.4_lightning_harvest.py:127` | `LIGHTNING_ACTIVE_JOBS_PATH` | HIGH | Replaced `_save_active_jobs` with `_mark_row_terminal_locked` (uses `update_active_jobs_locked`) |
| 3 | `experiments/lossy_coarsening_lightning_cuda_test.py:1154` | `LIGHTNING_ACTIVE_JOBS_PATH` | HIGH | Routed through `register_job` |
| 4 | `experiments/lossy_coarsening_lightning_harvest.py:124` | `LIGHTNING_ACTIVE_JOBS_PATH` | HIGH | Replaced `_save_active_jobs` with `_mark_row_terminal_locked` |
| 5 | `scripts/launch_lane_on_vastai.py:764` | `vastai_active_instances.json` | HIGH | Routed through `tac.vastai_tracker.register_instance` (canonical fcntl helper) |
| 6 | `src/tac/deploy/lightning/lightning_dispatch.py:220` | `lightning_active_sessions.json` | HIGH | New `_lightning_state_lock` context manager + atomic `_save_state` (unique tmp + fsync + os.replace) wrapped by `register_session` / `remove_session` |
| 7 | `scripts/verify_vast_instances.py:74` | `instance_setup_first_seen.json` | MEDIUM | Inline fcntl + reload-inside-lock + unique tmp + fsync + os.replace pattern in `_save_setup_first_seen` |

### §2.3 Concurrency exposure analysis (per finding)

#### Finding 1-4: `LIGHTNING_ACTIVE_JOBS_PATH` (4 callers)

**Bug class**: SAME META as catalog #128 (`continual_learning.save_posterior`).
Four separate experiment scripts maintained their own bare load→mutate→write
cycles against `.omx/state/lightning_active_jobs.json`:

```python
existing = json.loads(LIGHTNING_ACTIVE_JOBS_PATH.read_text())  # racey read
existing.append(record)                                         # in-process mutation
LIGHTNING_ACTIVE_JOBS_PATH.write_text(json.dumps(existing))     # racey write
```

**Real concurrency exposure**: cron-fired `arch_shrink_x0.4_lightning_full.py`
+ sister `arch_shrink_x0.4_lightning_harvest.py` invoked in the same minute
WOULD silently drop each other's row updates. The harvester reads stale state,
marks a row terminal, and writes back — losing the dispatcher's NEW row
appended in between. This is the exact MEDIUM bug class codex round-2 caught
in `continual_learning.save_posterior` (memory:
`feedback_codex_round2_custody_concurrency_fix_landed_20260509.md`).

**Canonical fix**: new module `src/tac/deploy/lightning/active_jobs_state.py`
provides:

- `_active_jobs_lock()` — fcntl context manager
- `load_active_jobs()` — read-only snapshot (safe outside lock; ``os.replace``
  semantics in writers guarantee no partial reads)
- `_save_active_jobs(rows)` — atomic write (unique tmp + fsync + os.replace);
  MUST be called inside the lock
- `update_active_jobs_locked(mutate_fn)` — locked transactional update
  (load→mutate→save inside `LOCK_EX`)
- `register_job(record)` — `update_active_jobs_locked` wrapper for
  unconditional append
- `upsert_job(record, key="job_name")` — replace-or-append by key
- `mark_job_terminal(job_name, terminal_status)` — sister-safe terminal mark

#### Finding 5: `scripts/launch_lane_on_vastai.py:register_in_tracker`

**Bug class**: identical to findings 1-4. `vastai_active_instances.json` is
written by:

- `src/tac/vastai_tracker.register_instance` (the CANONICAL writer with
  fcntl + sibling lockfile pattern)
- `scripts/launch_lane_on_vastai.py:register_in_tracker` (a duplicate
  load→mutate→write that BYPASSED the canonical writer)

If two parallel `launch_lane_on_vastai.py` invocations land at the same
moment, BOTH read empty/stale state, both append their new instance, the
last-writer wins, the OTHER instance is dropped from the tracker → the
verify_vast_instances.py orphan-cleanup code would see ONE phantom instance
that has no tracker entry, and the user pays for it indefinitely.

**Canonical fix**: route through `tac.vastai_tracker.register_instance`
(same fcntl semantics already in production for `tac.deploy.vastai.client`
calls). Minor narrow race remains between `list_instances()` (idempotency
check) and `register_instance` because the canonical helper does not
de-duplicate; idempotency is a UX nicety not a correctness invariant
(verify_vast_instances.py keys on instance_id; duplicate rows still cleanup
correctly).

#### Finding 6: `src/tac/deploy/lightning/lightning_dispatch.py`

**Bug class**: identical to findings 1-4 + 5. `LightningDispatcher._save_state`
was a bare write; `register_session` / `remove_session` did
load → mutate → save without any lock. Two parallel `dispatch_lane()` /
`tear_down()` calls would race.

**Canonical fix**: new `_lightning_state_lock` context manager (sister of
`tac.continual_learning._posterior_lock`); rewrote `_save_state` to use
unique tmp + fsync + os.replace (sister of `save_posterior`); wrapped
`register_session` and `remove_session` in `_lightning_state_lock`.

#### Finding 7: `scripts/verify_vast_instances.py:_save_setup_first_seen`

**Bug class**: same META, lower exposure. `verify_vast_instances.py` is run
by an operator on a schedule, not by the parallel-dispatch fleet. Concurrency
exposure is real but rare. Defense-in-depth fix applied: inline fcntl +
reload-inside-lock + unique tmp + fsync + os.replace pattern.

## §3 Recommended fix priority — TOP 5 (all APPLIED)

By severity × callsite-criticality:

1. **Finding 1 + 3** (dispatcher append to `LIGHTNING_ACTIVE_JOBS_PATH`):
   highest concurrency exposure (multiple sweep arms can land
   simultaneously); each lost row = invisible orphan job that we keep
   paying GPU credits for. **APPLIED via `register_job`.**
2. **Finding 2 + 4** (harvester mark-terminal on `LIGHTNING_ACTIVE_JOBS_PATH`):
   second-highest exposure (harvester races with dispatcher even in
   "single-arm" use). **APPLIED via `_mark_row_terminal_locked`.**
3. **Finding 5** (`launch_lane_on_vastai.py` Vast.ai tracker write): high
   exposure because the cleanup script's source-of-truth depends on this.
   **APPLIED via `tac.vastai_tracker.register_instance`.**
4. **Finding 6** (`lightning_dispatch.py` session state): used by the
   manual H100 path; lower-frequency but corrupting it would break
   teardown. **APPLIED via `_lightning_state_lock` + atomic `_save_state`.**
5. **Finding 7** (`verify_vast_instances.py` SETUP-first-seen): defense-in-
   depth; medium severity. **APPLIED inline.**

## §4 STRICT preflight checks proposed

Two new gates land warn-only initially per directive (Lane A → strict
pattern). Catalog numbers claimed via `tools/claim_catalog_number.py claim`
under fcntl serialization (gate #118).

### §4.1 Catalog #130 — `check_no_tag_only_custody_validation`

**Scope**: extends catalog #127 (`AUTHORITATIVE_TAGS` membership) to also
catch the `evidence_grade in {"contest-cuda", "A++", ...}` set-membership
pattern and the `tag.startswith("[contest-...")` substring-prefix pattern,
when the predicate does NOT have local joint-validator or fail-closed blocker
context.

**Detection**:

1. Scan `src/tac/**/*.py`, `tools/*.py`, `experiments/*.py` (excluding tests).
2. For each file containing one of:
   - `evidence_grade in {`
   - `evidence_grade.lower() in {`
   - `tag.startswith("[contest-CUDA"`)
   - `tag.startswith("[contest-CPU")`
   - `tag in CONTEST_TAGS` (any all-caps tag set)
3. The predicate's local line window MUST ALSO contain one of:
   - `validate_custody` / `validate_custody_verdict`
   - `posterior_update` / `posterior_update_locked`
   - `is_promotable_exact_cuda_evidence` / `promotable_exact_cuda_evidence_blockers`
   - `archive_sha256` (with a `sha256_file(...)` or hex-validation call)
   - `# CUSTODY_VALIDATOR_OK:<reason>` same-line waiver
4. Otherwise: violation.

**Canonical implementation file exclusion**: `src/tac/continual_learning.py`,
`src/tac/optimization/candidate_evidence_contract.py`,
`tools/predispatch_sanity.py` (the validators themselves).

**Held warn-only initially** per directive. Live count after adversarial
hardening: 0. Round-2 hardening explicitly removed the broad whole-file
validator-token accept so one safe helper cannot mask a later tag-only gate.

### §4.2 Catalog #131 — `check_no_bare_writes_to_shared_state`

**Scope**: refuses any code site that writes to a known shared-state path
without a real local lock context. Canonical helper calls are the preferred
replacement for a bare write, but helper names near a bare write do not waive
that write.

**Detection**:

1. Scan `src/tac/**/*.py`, `tools/*.py`, `experiments/*.py`, `scripts/*.py`
   (excluding tests).
2. Recognize shared-state path markers:
   - `.omx/state/[A-Za-z0-9_./-]+\.(json|jsonl|md|txt)`
   - Module-constant assignments matching `<NAME> = ... .omx/state/...`
3. For each file referencing a shared-state path:
   - Identify variables bound to shared-state paths via static assignment
     scan (`SHARED_VAR_RE`).
   - For each `<recv>.write_text(...)` / `.write_bytes(...)` / `os.replace(...)`
     / `open(<path>, 'w')` whose path argument resolves to a shared-state
     variable or contains a shared-state literal:
4. The line MUST satisfy ONE of:
   - The preceding local line window ALSO references a real lock/context token
     (`fcntl.flock`, `LOCK_EX`, `_posterior_lock`, `_lightning_state_lock`,
     `_active_jobs_lock`, `FileLock`).
   - Same-line `# BARE_WRITE_OK:<reason>` waiver.
5. Otherwise: violation.

**Canonical writer exclusion**: `src/tac/continual_learning.py` (catalog
#128 already covers `save_posterior`), `src/tac/vastai_tracker.py`
(canonical helper), `src/tac/deploy/lightning/lightning_dispatch.py`
(canonical helper added in this lane),
`src/tac/deploy/lightning/active_jobs_state.py` (canonical helper added
in this lane), `src/tac/deploy/azure/azure_dispatch.py` (already locked),
`tools/claim_catalog_number.py` (canonical), `tools/subagent_commit_serializer.py`
(canonical), `tools/lane_maturity.py` (uses lane-claim helper).

**Held warn-only initially** per directive. Live count after adversarial
hardening: 0. Round-2 hardening explicitly removed the broad whole-file
lock-token accept so one locked writer cannot mask a later bare shared-state
write. Follow-up adversarial hardening also removed helper-name accepts such as
`register_job` / `register_instance`; those helpers must replace the bare write,
not merely appear near it.

## §5 6-hook wire-in declaration (per CLAUDE.md "Subagent coherence-by-default")

1. **Sensitivity-map**: N/A — preflight infrastructure, not a score signal.
   The fix improves custody/concurrency fidelity for shared state that
   downstream sensitivity-map contributors and dispatch claims read; does
   not produce a new partial derivative.
2. **Pareto constraint**: N/A — no new constraint added.
3. **Bit-allocator hook**: N/A — no per-tensor importance change.
4. **Cathedral autopilot dispatch hook**: indirect — fixes unblock safe
   parallel dispatch. The locked `lightning_active_jobs.json` writer +
   `lightning_active_sessions.json` writer + `vastai_active_instances.json`
   writer are the dispatch-state surfaces the autopilot WOULD have raced
   against under heavy concurrent load.
5. **Continual-learning posterior update**: indirect — extends the same
   locked-write contract from catalog #128 (`posterior_update_locked`) to
   four additional shared-state surfaces. The custody validator (catalog
   #127) is unchanged; this lane only adds catalog #130 to widen its
   detection scope (no behavioral change).
6. **Probe-disambiguator**: N/A — fixes are deterministic (fcntl-serialized;
   no design tension between two defensible interpretations).

## §6 META-meta finding — bug-class spread

Catalog #128 caught `continual_learning.save_posterior` (1 file). The
proactive sweep here found **6 ADDITIONAL surfaces** with the same META
class:

- `LIGHTNING_ACTIVE_JOBS_PATH` (4 callers)
- `LIGHTNING_STATE` (1 file, but used in 3 methods)
- `vastai_active_instances.json` (1 caller bypassing canonical writer)
- `instance_setup_first_seen.json` (1 file)

That's a 6-7× spread of the bug class beyond the codex-found surface. The
META-meta finding: **the bug class is OPEN, not closed**, before the
proactive sweep. Catalog #128 alone narrowed-fix the symptom; catalog
#131 closes the bug class. Going forward, every new shared-state surface
will be gated at preflight time before it can ship the same META.

The pattern of "fix lands in narrow case, but META class spreads to
adjacent surfaces because each surface has its own bare load→mutate→write"
is exactly the kind of structural-debt accumulation that motivates the
"Subagent coherence-by-default" mandate. The umbrella META-class gate
(#131) is what keeps the bug class structurally extinct.

## §7 Cross-references

- Catalog #127 (`check_authoritative_tag_requires_custody_metadata`) +
  #128 (`check_continual_learning_writes_use_lock`):
  `feedback_codex_round2_custody_concurrency_fix_landed_20260509.md`
- Codex round-2 directive:
  `.omx/research/codex_review_findings_inflight_subagent_directive_20260509_round2.md`
- Codex round-1 (same META class):
  `.omx/research/codex_adversarial_review_findings_for_inflight_subagents_20260509.md`
- Substrate-vs-codec META (round 0):
  `~/.claude/projects/.../feedback_substrate_vs_codec_composition_meta_pattern_20260508.md`
- Lane registry: `.omx/state/lane_registry.json` →
  `lane_proactive_custody_concurrency_audit` (L0 SKETCH)
- Catalog state: `.omx/state/next_catalog_number.txt` (132 after two claims)

## §8 Coordination notes

### a086a57d (T19 migration + cathedral autopilot catalog)

**No pattern violations found in `tac.joint_admm_coordinator` or
`tools/cathedral_autopilot.py`.**

The audit detector scanned both files; neither references a shared-state
path that writes WITHOUT lock discipline. `cathedral_autopilot.py` does
read shared state but writes its outputs to its own per-run output
directory (not `.omx/state/`). `tac.joint_admm_coordinator` is a pure
math module with no I/O. No coordination directive needed for a086a57d.
