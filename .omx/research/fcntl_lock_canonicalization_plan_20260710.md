# fcntl-lock canonicalization plan — repo-wide scoping (2026-07-10)

Operator directive (verbatim): *"We are duplicating and hardcoding way too much."* This is the
deferred, repo-wide follow-up to `.omx/research/hardcode_duplication_audit_witness_stack_20260710.md`
finding #4, which named a canonical `tac.jsonl_store.append_locked_jsonl` helper (landed commit
`3722879fc`) and 5 known follow-up call sites, but explicitly punted the "~85-95 files touch
`fcntl.LOCK_EX` directly" repo-wide sweep as out of scope. **This memo IS that sweep. READ-ONLY —
no code was edited.**

## Executive summary — the real numbers

Raw grep for `fcntl.flock`/`import fcntl` across `src/`, `tools/`, `experiments/`: **111 files**.
Of those:

- **10 files** are inside frozen/vendored bundles (`experiments/results/comma_lab_public_export/**`,
  `experiments/results/kaggle_pr106_latent_score_table_r2_*/pact_pr106_latent_workspace/**`,
  two `experiments/results/*/register_canonical_equation_and_probe_outcome.py` one-off scripts) —
  **OUT OF SCOPE**, historical/exported snapshots, not live source per the "in-place edits to
  public PR intake clones" discipline.
- **14 files** are test files (`src/tac/tests/test_*.py`) exercising the lock modules — **OUT OF
  SCOPE for this wave** (they test call sites; they don't duplicate the lock implementation
  themselves; each will need a touch-up alongside whichever call site it covers).
- **87 files** are live, in-scope production code with **129 distinct lock-acquisition sites**
  (some files lock more than once for different purposes — e.g. a read-lock + a write-lock, or
  two independent stores in one module).

**The "85-95 duplicated append-helper" framing from the source audit is directionally right but
categorically too narrow.** Reading all 129 sites in context (not just grepping for the token)
shows the fcntl usage is **not one duplicated pattern** — it is **four structurally distinct
patterns**, only one of which is what `append_locked_jsonl` already covers:

| Pattern | Shape | Site count (sites, not files) | Migratable today? |
|---|---|---:|---|
| **A — same-file self-lock append** | `with open(p, "a") as f: flock(f); write; unlock` | **7** | YES, mechanical, to the EXISTING `append_locked_jsonl(path, row)` |
| **A′ — separate-lock-file append** | `with lock.open("a"): flock; with target.open("a"): write` | **~16** | Only after `append_locked_jsonl` grows an optional `lock_path=` parameter (small, well-scoped extension) |
| **2 — re-entrant critical-section context manager** | `_xxx_lock()` ctx-mgr: depth-counter (process-global OR thread-local) + blocking-or-`LOCK_NB`+poll+timeout, wraps an arbitrary load→mutate→save transaction | **~38** (by far the largest cluster) | NO — needs a design pass first (see below); this is the single biggest duplication cluster in the repo, bigger than the append pattern the source audit named |
| **3 — atomic-append-via-read-all+tmp+rename** | read whole file, append line, write tmp, `os.replace` | **~6** | NO — a distinct, more expensive, more "crash-safe" append variant; own sibling helper |
| **4 — atomic single-JSON-object rewrite** | `tmp_path.write_text(json.dumps(obj)); os.replace(tmp, path)` under lock | **~19** | NO — this is a REWRITE (Catalog #128/#131 atomic-write pattern), never an append; must NOT be folded into an append helper (per CLAUDE.md archival-policy note in the canonical `jsonl_store.py` docstring) |
| **5 — singleton/mutex lock (non-blocking, no retry)** | `flock(LOCK_EX\|LOCK_NB)`; raise/return-None if already held | **~3** | N/A — a different primitive entirely (process-exclusivity gate, not a JSONL writer) |
| **read-locks (`LOCK_SH`)** | shared/read lock for a concurrent-reader-safe scan | **2** | N/A — different operation (reads, not writes) |

**Bottom line for "how many files can migrate mechanically right now": 7 sites in 6 files** — the
5 already named by the source audit (`activation_ledger.py`'s 2nd inline block,
`costate_posterior.py`, `shadow_controller.py`, `campaign_repl.py`, `decode_cache.py`'s `put()`)
**plus 2 new ones this sweep found** (`tools/system_memory_governor.py::append_band_row`,
`tools/witness_memory_preflight.py::append_ledger_row`). Everything else needs either a small,
well-scoped extension to the canonical helper (Pattern A′, ~16 more sites) or a genuine design
pass before touching it (Patterns 2/3/4, ~63 more sites). The "85-95 file" figure is real as a
*count of files that touch fcntl*, but it is NOT a count of files migratable to today's helper.

## Pattern A — mechanical today (7 sites, 6 files)

All 7 use the EXACT shape `append_locked_jsonl` already implements: lock the **same file** being
written, `open(p, "a")`, `flock(LOCK_EX)`, write one JSON line, unlock. Verified by reading each
site's full function body, not just the grep hit.

| File:line | Function | fsync? | ImportError fallback? | Notes |
|---|---|---|---|---|
| `src/tac/witness_dsl/activation_ledger.py:143` | `record_activation()` | yes | yes | Named follow-up #1 in the source audit — the file already imports `append_locked_jsonl` for its OTHER function; this is the 2nd, still-inline block. |
| `src/tac/witness_control/costate_posterior.py:82` | (append) | yes | yes | Byte-identical shape to the canonical helper already. |
| `src/tac/witness_control/shadow_controller.py:619` | `write_shadow_row()` | no | yes | Docstring literally says "align with the sibling stores (costate_posterior.py)". |
| `src/tac/witness_control/campaign_repl.py:164` | `write_world_model_row()` | no | yes | Same "align with... costate_posterior.py" comment. |
| `src/tac/witness_control/decode_cache.py:87` | `put()` | no | no (bare `open`, no try/except) | `get()` at line 61 uses `LOCK_SH` — stays Pattern-B/read-lock, do NOT touch. |
| `tools/system_memory_governor.py:1712` | `append_band_row()` | no | no | **NEW finding, not in the source audit's list of 5.** |
| `tools/witness_memory_preflight.py:377` | `append_ledger_row()` | no | no | **NEW finding, not in the source audit's list of 5.** |

Migrating all 7 to `append_locked_jsonl(path, row)` is a pure behavior-preserving simplification;
2 of them (`shadow_controller.py`, `decode_cache.py`) gain an `os.fsync()` they didn't have before
(strict durability improvement, not a behavior change any caller depends on).

## Pattern A′ — mechanical after a small helper extension (~16 sites)

Same logical operation (append one JSON line) but the lock is on a **separate, stable sibling
file** (usually named `<target>.lock`), not the data file itself. This is a deliberate, sound
design choice in several of these (see `deploy/hf_jobs/job_id_ledger.py`'s docstring: *"Touch lock
file (separate from ledger) so the lock survives ledger rotation / archival without losing
exclusive serialization"* — i.e. it protects against the exact rotation race the CLAUDE.md "State
JSONL archival policy" section describes). `append_locked_jsonl` cannot express this today because
it always locks the file it writes.

**Proposed extension** (additive, backward-compatible):

```python
def append_locked_jsonl(
    p: Path, row: dict, *, sort_keys: bool = True, lock_path: Path | None = None,
) -> None:
    """... lock_path: if given, lock THIS file instead of ``p`` (rotation-survival pattern)."""
    lock_target = lock_path if lock_path is not None else p
    ...
    with open(lock_target, "a") as lockf:
        flock(lockf); try: with open(p, "a") as f: write/flush/fsync finally: unlock
```

Sites (all verified same-shape: lock separate file, append one line to target, no read-validate
step beyond a cheap dup-check or newline-separator check):

- `src/tac/cathedral_consumers/information_theoretic_floor_consumer/_posterior_store.py`
- `src/tac/deploy/hf_jobs/job_id_ledger.py`
- `src/tac/cost_band_calibration.py`
- `src/tac/council_continual_learning.py`
- `src/tac/findings_lagrangian/phase_2_ablation/ablation_framework.py`
- `src/tac/harness_failure_ledger.py`
- `src/tac/optimization/macos_cpu_advisory_signal.py`
- `src/tac/optimization/mlx_research_signal.py`
- `src/tac/optimization/mps_research_signal.py`
- `src/tac/review_counter.py` (has a `_needs_newline_separator` pre-check — cosmetic, not a
  read-validate transaction; still a clean append)
- `src/tac/session_bus/bulletin.py` (same newline-separator pre-check)
- `src/tac/substrates/pretrained_driving_prior/local_chunk_streamer.py` (`_append_jsonl_locked`,
  `"ab"` mode + fsync; its `replay_stream_log` uses `LOCK_SH` — stays Pattern-B, untouched)
- `src/tac/training/long_training_canonical.py::_append_checkpoint_retention_manifest` (single-row
  append; the SAME file's `flush()` method is a **batch**-append of N buffered rows in one lock
  scope — needs a batched variant, not a 1:1 fit, list separately below)
- `tools/extract_master_gradient_mlx.py::_append_jsonl_locked`

**Needs-design cousins, NOT drop-in even after the extension** (flagged so nobody force-fits them):
- `src/tac/optimization/mlx_dynamic_sweep_observations.py::append_observation_row` — does an
  in-lock duplicate-detection READ before appending (`_find_duplicate_observation`); this is a
  check-then-act, not a pure append. Could adopt a `pre_write_check: Callable | None` hook on the
  extended helper, but that is a real API design choice, not mechanical.
- `src/tac/training/long_training_canonical.py::flush()` — appends a LIST of buffered rows under
  one lock acquisition (batched), not "exactly one row per call." Needs a
  `append_locked_jsonl_batch(path, rows, ...)` sibling or a loop-inside-lock variant.

## Pattern 2 — the REAL biggest duplication cluster (~38 sites) — NOT mechanical, needs a design pass

This is the finding the source audit's narrower "append helper" framing missed entirely. At least
**24 distinct modules** implement a near-identical `_xxx_lock()` `@contextlib.contextmanager`:
acquire `fcntl.flock` on a dedicated lock file, guarded by an in-process re-entrancy depth counter,
wrapping an arbitrary caller-supplied load→mutate→save transaction (this is NOT a JSONL-append
primitive — the body inside the `with` block does real business logic: reload from disk, validate,
mutate, atomically rewrite). Some copies literally say "mirrors" another one in their docstring
(`canonical_equations/registry.py` → *"Mirrors Catalog #344 `_registry_lock` pattern"*;
`optimization/pair_frame_scorer_geometry_lattice_5d_canvas_populator.py` → *"Mirrors
`tac.deploy.modal.call_id_ledger._ledger_lock`"*; `tools/spawn_durable_daemon.py` → *"mirrors the
canonical active_vms_state depth-counter"*) — i.e. the repo already informally recognizes this as
ONE pattern being hand-copied, not N independent designs.

**At least 4 files (`boosting/persistence.py`, `compress_time_optimization/persistence.py`,
`inflate_time_post_processing/persistence.py`, `side_information/persistence.py`) share a
byte-identical function name (`_stage_outcomes_lock`/`_pass_outcomes_lock`/`_baker_outcomes_lock`)
AND byte-identical docstring** ("Acquire fcntl LOCK_EX on the ... lock file. Per-process advisory
lock (multiple processes serialize against each other; a single process re-entering is tracked via
depth counter so nested context managers do not deadlock)." — this is a copy-paste, not 4
independent implementations).

**Why this is NOT mechanical** — the copies are NOT identical in ways that matter for correctness:

1. **Depth-counter scope differs per file**, and at least one of these differences is a
   DELIBERATE, documented fix, not an oversight: `deploy/lightning/active_jobs_state.py` moved
   from a process-global `int` to a **thread-local** counter (`_active_jobs_lock_depth_tls`) per
   "OP-7 fix (codex chunk 5, 2026-05-15)" specifically so a DIFFERENT thread in the same process
   correctly blocks instead of silently re-entering. `probe_outcomes_ledger.py` and
   `streaming_prediction_ledger.py` also use thread-local depth; most others (`deploy/azure/
   active_vms_state.py`, `deploy/lightning/lightning_dispatch.py`, `tools/spawn_durable_daemon.py`)
   use a plain process-global `int`. Collapsing all of these onto ONE shared implementation
   requires deciding — per ledger — whether cross-thread contention is a real scenario, and a
   careless default could silently REGRESS the OP-7 fix (or over-serialize a ledger that never
   sees multi-threaded callers).
2. **Blocking vs. non-blocking+poll+timeout differs.** Roughly half acquire with a blocking
   `fcntl.flock(fd, LOCK_EX)` (waits forever); the other half use
   `fcntl.flock(fd, LOCK_EX | LOCK_NB)` in a `while True: ... except BlockingIOError: if
   time.monotonic() >= deadline: raise TimeoutError` polling loop (typically a 30s
   `LOCK_TIMEOUT_SECONDS`). A shared helper needs both modes as an explicit parameter, not a
   silent behavior change for any caller.
3. **Some are context managers (`@contextmanager`, `yield`), some return a raw `fd`/`fh` for the
   caller to close manually** (`atom/ledger.py::_acquire_lock`, `action_effect.py::
   _v1_acquire_ledger_lock`, `tools/claim_catalog_number.py`, `tools/subagent_checkpoint.py`,
   `tools/subagent_commit_serializer.py`) — different calling conventions, not just different
   variable names.

**Confirmed site list** (module :: local name; `[TL]` = thread-local depth, `[PG]` = process-global
depth, `[none]` = no depth counter shown, `[NB]` = non-blocking+poll+timeout, `[B]` = blocking):

`src/tac/autopilot_rudin_daubechies/slim_ranker.py::_slim_anchor_store_lock` [PG,B] ·
`src/tac/boosting/persistence.py::_stage_outcomes_lock` [PG,NB] ·
`src/tac/canonical_anti_patterns/registry.py` (unnamed `_registry_lock`) [PG,NB] ·
`src/tac/canonical_equations/registry.py` [PG,NB] ·
`src/tac/canonical_task_status/writer.py::_ledger_lock` [none,NB] ·
`src/tac/cathedral/verdict_ledger.py` [PG,NB] ·
`src/tac/compress_time_optimization/persistence.py::_pass_outcomes_lock` [PG,NB] ·
`src/tac/atom/ledger.py::_acquire_lock` [none,NB, returns fd] ·
`src/tac/analysis/action_effect.py::_v1_acquire_ledger_lock` [none,NB, returns fh] ·
`src/tac/codex_to_claude_inbox.py::_inbox_lock` [PG,NB] ·
`src/tac/continual_learning.py::_posterior_lock` [none,B — the ORIGINAL pattern others "mirror"] ·
`src/tac/deploy/azure/active_vms_state.py::_active_vms_lock` [PG,B] ·
`src/tac/deploy/lightning/active_jobs_state.py::_active_jobs_lock` [**TL**,B — the OP-7-fixed one] ·
`src/tac/deploy/lightning/batch_jobs.py::_state_file_lock` [none,B] ·
`src/tac/deploy/lightning/lightning_dispatch.py` (unnamed) [PG,B] ·
`src/tac/deploy/modal/call_id_ledger.py::_ledger_lock` [TL,NB] ·
`src/tac/inflate_time_post_processing/persistence.py::_pass_outcomes_lock` [PG,NB — byte-identical
to boosting/persistence.py] ·
`src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas_populator.py::_populator_lock`
[none,NB] ·
`src/tac/optimization/repair_campaign_posterior.py::_posterior_lock` [none,B] ·
`src/tac/pareto_polytope_unified_solver/solver.py::_ledger_lock` [none,B, opens lock file `"w"`
(truncates every acquire — harmless for flock but wasteful)] ·
`src/tac/per_archive_drift_posterior.py::_per_archive_lock` [none,B, returns fd] ·
`src/tac/preflight_rudin_daubechies/slim_risk_scorer.py::_preflight_anchor_store_lock` [none,B] ·
`src/tac/probe_outcomes_ledger.py::_ledger_lock` [TL,NB] ·
`src/tac/search/persistence.py::_search_outcomes_lock` [PG (mutex-protected),NB] ·
`src/tac/side_information/persistence.py::_baker_outcomes_lock` [PG,NB — byte-identical shape to
the other 3 `*_outcomes_lock` files] ·
`src/tac/streaming_prediction/streaming_prediction_ledger.py` [TL,NB] ·
`src/tac/uniward_invariant_enumerator/enumerator.py::_ledger_lock` [none,B — simplest variant] ·
`src/tac/wyner_ziv_deliverability/proof_builder.py::_proofs_lock` [none,NB] ·
`tools/spawn_durable_daemon.py::_registry_lock` [PG,B] ·
`tools/lane_maturity.py::_mutation_lock` [none,B] ·
`tools/run_nightly_catalog_gate_regression.py::_fcntl_lock` [none,B] ·
`tools/run_codex_review_for_dispatch.py::_cache_lock` [none,B] ·
`tools/prove_shell_inflate_parity.py` (inline, wraps inflate-cache business logic, not reusable
as-is) [none,B] ·
`tools/subagent_checkpoint.py::_acquire_lock` [none,NB, returns fh] ·
`tools/claim_catalog_number.py` (2 sites: the counter's own lock + an unrelated 2nd acquire helper)
[none,NB, returns fh] ·
`tools/subagent_commit_serializer.py::_acquire_lock` [none,NB, returns fh — **bootstrap-critical,
recommend excluding from any consolidation**, see Batch notes] ·
`tools/memory_blackbox.py::append_sample`'s inline lock [none,B, wraps rotate-check-then-append].

**Proposed sibling helper** (design sketch only — NOT to be built by a mechanical batch):

```python
@contextlib.contextmanager
def locked_scope(
    lock_path: Path, *, timeout: float | None = None, reentrant_scope: str = "process",
) -> Iterator[None]:
    """fcntl LOCK_EX critical section. timeout=None -> blocking; timeout=N -> LOCK_NB+poll+raise.
    reentrant_scope: "process" (global int depth) | "thread" (threading.local depth) | "none"."""
```

A migration wave here should go PER-LEDGER (audit each site's actual concurrency need — is it ever
called from multiple threads in one process? does a caller ever nest two lock acquisitions?) rather
than a blind find-replace; that per-ledger audit is exactly the kind of "needs-design" work the
source audit already flagged as out of its scope, now with a full site inventory to work from.

## Pattern 3 — atomic-append-via-read-all+tmp+rename (~6 sites)

Distinct from Pattern A/A′: instead of `open(p, "a")`, these **read the whole existing file**,
concatenate the new line, write to a `.tmp.<uuid>` file, and `os.replace()` it — under a SEPARATE
lock file. More expensive (O(file size) per append) but gives atomic-replace semantics (no
partial-line risk even without relying on POSIX `O_APPEND` atomicity). Two call sites share the
EXACT phrase *"Atomic append via tmp + rename per Catalog #131 sister discipline"* in their
docstrings:

- `src/tac/master_gradient.py::_atomic_write_append` (used by `append_anchor_locked`)
- `src/tac/cross_substrate_master_gradient_analyzer/analyzer.py::_atomic_append_jsonl`
- `src/tac/pareto_polytope_unified_solver/solver.py::_atomic_append_jsonl` (same name as above)
- `src/tac/master_gradient_mlx_pipeline.py` — **2 inline copies** (no factored helper at all;
  `_append_state_row_locked` and the manifest-write block inside `auto_schedule_...`), both
  literally `tmp.write_text(existing + line + "\n"); os.replace(tmp, target)`
- `src/tac/recursive_adversarial_review.py::append_round_locked` — calls
  `master_gradient.py`'s `_atomic_write_append` but ALSO does a locked read-validate step
  (`clean_pass_counter_for_bundle` check) INSIDE the same critical section before the write —
  this one is genuinely custom business logic, not a drop-in target even for a Pattern-3 helper.

**Proposed sibling helper**: `atomic_rewrite_append_jsonl(path, row, *, lock_path=None)` — read
existing bytes, append one serialized line, write via `NamedTemporaryFile`/`.tmp.<uuid>` + fsync +
`os.replace`. Only the first 4 non-`recursive_adversarial_review.py` sites are drop-in candidates;
even those need a decision on whether the O(n) cost is intentional (these ledgers may be small
enough that it never matters) before consolidating away the incremental-append option.

## Pattern 4 — atomic single-JSON-object rewrite (~19 sites) — explicitly NOT an append pattern

This is the Catalog #128/#131 "atomic write" contract for a **single JSON object** (not a JSONL
row), e.g. `.omx/state/canonical_frontier_pointer.json`, per-sidecar manifests, tracker files.
Shape: `tmp_path.write_text(json.dumps(payload)); os.fsync; os.replace(tmp, target)` under a
separate lock file. Confirmed sites include `canonical_frontier_pointer.py`,
`optimization/bit_allocator_end_to_end.py`, `optimization/field_equation_planner.py`,
`optimization/jacobian_fisher_importance_allocator.py`, `master_gradient_pose_vulnerability/
pose_vulnerability_map.py`, `deploy/modal/call_id_ledger.py`, `vastai_tracker.py` (load-mutate-save
of a JSON array, not a single object, but same rewrite shape), `deploy/azure/active_vms_state.py`,
`deploy/lightning/active_jobs_state.py`/`lightning_dispatch.py`/`batch_jobs.py`,
`tools/append_slot_h_cross_archive_84_cell_to_substrate_composition_matrix.py::_atomic_write_locked`,
`tools/pre_entropy_substrate_pivot_prober.py::_fcntl_locked_atomic_write`,
`tools/wyner_ziv_deliverability_prober.py::_fcntl_locked_atomic_write` (**byte-identical name AND
docstring** to the pre_entropy file — a 3rd copy exists inline in
`tools/q6_preprobe_pairwise_composition_alpha.py` without even a named helper),
`tools/extract_master_gradient_mlx.py::_locked_save_npy` (same idea for `.npy`, not JSON) and its
3rd, separate sidecar-meta lock block (which — worth flagging — writes directly via
`sidecar_path.write_text(...)` INSIDE the lock with **no tmp+rename**, i.e. it is lock-protected
but NOT crash-atomic; a genuine small latent gap, out of scope to fix here but worth a follow-up
note), `tools/archive_jsonl_state.py` (rewrites a JSONL file splitting rows into kept/archived
buckets — a real rewrite, not an append), `tools/migrate_cost_band_posterior_failed_anchors.py`
(one-off migration script, reads+rewrites the whole posterior).

**Do not fold this into an append helper** — CLAUDE.md's own state-JSONL archival-policy section
and `jsonl_store.py`'s docstring both draw this exact line (Catalog #128/#131 atomic-**write**
vs. append). A `write_json_atomic_locked(path, obj, *, lock_path=None)` sibling helper is a
plausible future consolidation target but is its own, separate design effort.

## Pattern 5 — singleton/mutex locks (not a JSONL primitive at all)

`tools/queue_fleet.py::FleetLock`, `tools/queue_supervisor.py::SupervisorLock` (near-identical:
`LOCK_EX|LOCK_NB`, raise `ExperimentQueueError` if already held — a "only one supervisor process
may run" gate, not a data writer), and `tools/memory_blackbox.py::_acquire_singleton` (same idea,
returns `None` instead of raising). Worth a shared `single_instance_lock(path) -> bool` helper
someday, but it is a THIRD, unrelated primitive — not part of this append-canonicalization effort.

## Read-locks (`LOCK_SH`) — 2 sites, stay untouched

`src/tac/witness_control/decode_cache.py::get()` and
`src/tac/substrates/pretrained_driving_prior/local_chunk_streamer.py::replay_stream_log()` both
take a shared read-lock so a concurrent appender's write doesn't tear a read. Correct as-is; no
migration action.

## Risk: preflight's Catalog #128/#131/#132/#133 gates are file-content-aware

`src/tac/preflight.py` already has 4 STRICT-flip-eligible gates policing this exact surface:
`check_continual_learning_writes_use_lock` (#128), `check_no_bare_writes_to_shared_state` (#131,
scans for `fcntl.flock`/`LOCK_EX`/named lock tokens near a write, OR checks the writing file
against a **hardcoded path allowlist** `_BARE_WRITE_CANONICAL_HELPERS`), `check_locked_writes_
preserve_deletions` (#132), and `check_no_excluded_writers_in_check_131_accept_list` (#133 — a
META-meta gate that re-verifies every file IN #131's allowlist still actually contains a real lock
pattern). **`grep -n "jsonl_store\|append_locked_jsonl" src/tac/preflight.py` returns zero hits
today** — none of these gates yet recognize the new canonical helper by name.

None of the 7 Pattern-A sites (this memo's batch 1 candidates) are in `_BARE_WRITE_CANONICAL_
HELPERS`, so they aren't relying on the allowlist path — they must currently be passing gate #131
via its window-token scan (or their specific `.omx/state`-style paths simply aren't in scope for
that scan's recognized shared-state path patterns). The two files ALREADY migrated
(`activation_ledger.py`'s first function, `curriculum_candidate_pool.py`) are a live existence
proof that swapping in `append_locked_jsonl` did not trip any STRICT gate — but this has not been
explicitly verified against #128/#131/#132/#133 in isolation, only against the general test suite.
**Any Pattern-2/3/4 consolidation touching a file IN `_BARE_WRITE_CANONICAL_HELPERS`
(`continual_learning.py`, `vastai_tracker.py`, `lightning_dispatch.py`, `active_jobs_state.py`,
`active_vms_state.py`, `call_id_ledger.py`, `job_id_ledger.py`, `probe_outcomes_ledger.py`,
`codex_to_claude_inbox.py`, `canonical_frontier_pointer.py`, `canonical_equations/registry.py`,
`cathedral/verdict_ledger.py`, `findings_lagrangian/.../ablation_framework.py`,
`deploy/vastai/client.py`, `claim_catalog_number.py`, `subagent_commit_serializer.py`,
`claim_lane_dispatch.py`, `lane_maturity.py`) MUST also update Catalog #133's re-verification
(and possibly add the new shared-helper's module path to the "delegation comment + import-from
canonical helper module" recognized list) or the gate could start flagging a false-positive on an
already-exempted file whose content changed shape.** This is exactly why Pattern 2/3/4 are
"needs-design," not mechanical — the gate co-evolution is part of the design, not an afterthought.

## Batch queue (safety-ordered)

**Sequencing note before ANY batch starts:** `git status` shows `src/tac/witness_autoconfig.py`
currently dirty (another in-flight agent), and the immediately-prior commit (`feccfa39d`) migrated
12 `tools/*` consumers onto `witness_run_artifacts.py` — a DIFFERENT axis (run-artifact filename
constants) but touching the SAME directory tree (`src/tac/witness_control/`,
`src/tac/witness_dsl/`) this plan's Batch 1 lives in. Re-check `.omx/state/active_lane_dispatch_
claims.md` and `git status` immediately before starting Batch 1 to confirm no sibling agent has
since claimed `witness_control/*` or `witness_dsl/activation_ledger.py`.

### Batch 1 — RECOMMENDED FIRST, lowest risk, highest hygiene payoff (7 sites / 6 files)

Migrate all Pattern-A sites onto the EXISTING `append_locked_jsonl(path, row)` — zero API changes
needed, behavior-preserving (2 sites gain `os.fsync`), each is a 5-10 line diff:

1. `src/tac/witness_dsl/activation_ledger.py` (`record_activation`, line ~143)
2. `src/tac/witness_control/costate_posterior.py`
3. `src/tac/witness_control/shadow_controller.py` (`write_shadow_row`)
4. `src/tac/witness_control/campaign_repl.py` (`write_world_model_row`)
5. `src/tac/witness_control/decode_cache.py` (`put()` only — leave `get()`'s `LOCK_SH` untouched)
6. `tools/system_memory_governor.py` (`append_band_row`)
7. `tools/witness_memory_preflight.py` (`append_ledger_row`)

**Test requirements**: run each file's existing test module (all 6 files have direct or indirect
test coverage per the repo's convention); run `.venv/bin/python -c "from tac.preflight import
check_continual_learning_writes_use_lock, check_no_bare_writes_to_shared_state; ..."` (or the
project's standard `preflight_all` invocation) to confirm no NEW violations appear for these 6
files specifically (none are in the `_BARE_WRITE_CANONICAL_HELPERS` allowlist, so this is the
scan-path most likely to react to a shape change). Follow the `verify-landing` skill (ruff F check
+ targeted tests + review-gate 2 clean passes + serializer commit with post-edit shas).

### Batch 2 — extend the helper, then migrate ~14 sites

1. Land the `lock_path=` optional-parameter extension to `append_locked_jsonl` in `jsonl_store.py`
   (backward-compatible default = lock the target file itself, current behavior unchanged for
   every Batch-1 caller).
2. Migrate the 14 Pattern-A′ sites listed above EXCEPT the two "needs-design cousins"
   (`mlx_dynamic_sweep_observations.py`, `training/long_training_canonical.py::flush()`).
3. **Sequencing dependency**: `deploy/hf_jobs/job_id_ledger.py`, `probe_outcomes_ledger.py`, and
   `codex_to_claude_inbox.py` are in `_BARE_WRITE_CANONICAL_HELPERS` — re-run Catalog #133 after
   touching these three specifically.
4. Test requirements: same as Batch 1, plus a dedicated test asserting `lock_path=` actually locks
   a DIFFERENT inode than the target file when given (regression guard against silently
   collapsing the rotation-survival guarantee `job_id_ledger.py`'s docstring documents).

### Batch 3 — DEFERRED, needs a design memo first (Pattern 2, ~38 sites)

Do NOT dispatch a mechanical agent at this batch. It needs its own `.omx/research/*_design_*.md`
per the "Canonical helper 6-pillar landing discipline" + "UNIQUE-AND-COMPLETE-PER-METHOD" sections
of CLAUDE.md, specifically resolving: (a) which ledgers genuinely need thread-local vs
process-global re-entrancy (audit each call site's actual thread topology, not just its current
implementation choice — the OP-7 fix on `active_jobs_state.py` is the load-bearing precedent that
must not regress); (b) blocking-vs-`LOCK_NB`+timeout as an explicit, tested parameter; (c) the
preflight Catalog #131/#133 co-evolution named above. This is the single highest LOC-reduction
opportunity in the sweep but is explicitly the wrong shape for a "batch a Sonnet agent can migrate
in one pass" per this task's own safety criterion.

### Batch 4 — DEFERRED, own design pass (Pattern 3, atomic-append-via-rewrite, ~6 sites)

New sibling helper `atomic_rewrite_append_jsonl`; needs a decision on whether the O(n)-per-append
cost is acceptable to keep as a DISTINCT primitive (probably yes — it exists because these
ledgers wanted stronger crash-atomicity than plain `O_APPEND`) vs. quietly downgrading everyone to
plain append. Do not collapse Pattern 3 into Pattern A/A′ without that explicit decision.

### Batch 5 — DEFERRED, own design pass (Pattern 4, atomic single-JSON rewrite, ~19 sites)

New sibling helper `write_json_atomic_locked`; this is the biggest of the "needs-design" clusters
by site count after Pattern 2, but every site here is ALREADY well-served by its own working
implementation (this is Catalog #128/#131's core intended pattern) — the ONLY genuine bug-shaped
finding is `tools/extract_master_gradient_mlx.py`'s sidecar-meta write block, which locks but does
NOT use tmp+rename (a real, if minor, crash-safety gap — flagging for a future targeted fix, not
folding into this consolidation).

## Recommended next action

**Run Batch 1 first.** It is the smallest, safest, most mechanical unit (7 call sites across 6
files, all byte-for-byte matching today's canonical helper's exact contract, zero API changes,
2 already proven safe by the source audit's own prior 2-file landing). It closes out the source
audit's full "follow-up 5" plus 2 new same-shape finds this sweep surfaced, and it is the only
batch that requires NO new helper code, NO preflight co-evolution, and NO per-ledger concurrency
audit before landing.

## Pointer

Exact frontier pointer **0.19108282 UNMOVED** — this is a read-only scoping memo (apparatus means),
not a score row. Feeds the next de-duplication wave; does not itself move
`.omx/state/canonical_frontier_pointer.json`.
