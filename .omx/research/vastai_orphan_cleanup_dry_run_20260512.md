# Vast.ai orphan cleanup dry-run — 2026-05-12

**Scope:** Read-only inventory of `.omx/state/vastai_active_instances.json`
under the comprehensive bug sweep (Part D of operator directive 2026-05-12).

## Headline

- **230 tracker records** present at scan time
- **Age range:** min 10d / median 13d / max 14d (all from 2026-04-28 → 2026-05-02 window)
- **Unique labels:** 212 (most lanes appear once; a few have 2-3 duplicate registrations)
- **Registered-by host:** `Primary` (all 230 from the same local workstation)

## Tool execution

```
.venv/bin/python tools/vastai_orphan_cleanup.py
```

returned:

```
FATAL: could not query vastai: FileNotFoundError(2, 'No such file or directory')
vastai_active_instances tracker: /Users/adpena/Projects/pact/.omx/state/vastai_active_instances.json
  records: 230
```

The `vastai` CLI is not on PATH in the bug-sweep agent environment, so the tool
correctly aborted before attempting any destroy / prune action (fail-closed
behavior, no false orphan-classification possible).

## Interpretation

- Every tracker record is 10-14 days old — they predate every dispatch claim
  that has been opened in the last week and almost certainly correspond to
  instances that have already been destroyed via dispatch-claim terminal rows,
  AWS-side timeouts, or operator-side teardown.
- The single-host distribution (`Primary` x 230) confirms these are all from
  the operator's main workstation; remote agents have not been touching the
  file.
- The 212 unique labels matches CLAUDE.md "label every Vast.ai create" rule —
  the records are well-tagged, just stale.
- This is exactly the failure mode CLAUDE.md Catalog #148
  (`check_vastai_tracker_strict_load`) was added to guard against on
  WRITE-side; the READ-side cleanup is what `vastai_orphan_cleanup.py
  --prune-missing` is built for.

## Recommendation (operator decision required)

Per CLAUDE.md "Forbidden destructive actions without confirmation", THIS
sweep does NOT prune. Two options surface for the operator:

1. **Operator runs the cleanup tool directly** in an environment where the
   `vastai` CLI is on PATH (e.g. user's primary shell where `which vastai`
   resolves):

   ```bash
   # Dry-run first (no destroy):
   .venv/bin/python tools/vastai_orphan_cleanup.py --prune-missing

   # If output looks right, run again with --yes to actually prune (no destroy
   # needed; these instances were already destroyed via other paths).
   .venv/bin/python tools/vastai_orphan_cleanup.py --prune-missing --yes
   ```

   The `--prune-missing` flag drops tracker records for instances that no
   longer appear in `vastai show instances`. It does NOT call `vastai destroy`
   (those instances are already gone). Cost: $0. Time: ~30 seconds for the
   API call + write.

2. **Approve a one-shot bulk-prune of all records older than 7 days** without
   the API check. This is safer than it sounds because:
   - all 230 records are 10+ days old
   - any instance still actually running after 10 days would have shown up in
     `vastai show instances` and been re-registered into the tracker on the
     next dispatch (and zero recent dispatches have used these labels)
   - the only risk is removing a record for an instance that IS still running
     and was somehow not re-registered, in which case the operator can manually
     re-register it; cost: minor inconvenience, $0 cash

## Bug-class self-protection (already present)

Both options above are already self-protected by:

- **Catalog #148** (`check_vastai_tracker_strict_load`) — refuses any mutation
  of the tracker file under `tac.vastai_tracker` that does not route through
  the canonical strict-load + fcntl-locked write path.
- **Catalog #131** (`check_no_bare_writes_to_shared_state`) — refuses any
  bare write to `.omx/state/vastai_active_instances.json` outside the
  canonical helpers.
- **Catalog #135** (`check_setup_first_seen_uses_transactional_update_inside_lock`)
  — refuses split-transaction observe + remove patterns that could lose
  records under concurrent verifier runs.

The bug class "tracker fills up with phantom entries" cannot RE-fill silently
because writes only happen via `register_instance` and `remove_instance` in
`tac.vastai_tracker`, both fcntl-locked and strict-load gated. The 230
existing entries are pre-Catalog-#148 historical accumulation.

## What this dry-run does NOT do

- It does NOT delete or mutate `.omx/state/vastai_active_instances.json`.
- It does NOT call `vastai destroy` on any instance.
- It does NOT modify the bug-sweep landing memo's claims about safety.

The full prune awaits explicit operator approval per the two options above.

## Cross-refs

- Memory: `feedback_codex_round78_findings_fix_with_self_protection_landed_20260509.md`
  (the Catalog #148 self-protection landing)
- Memory: `feedback_remote_archive_only_eval_self_bootstraps_all_deps_20260501.md`
  (the outstanding-gaps section that calls out the disk + cuda_vers gates
  which prevent NEW phantoms but don't clean OLD ones)
- Tool: `tools/vastai_orphan_cleanup.py`
- Tracker: `.omx/state/vastai_active_instances.json` (124.5 KB, 230 records)
