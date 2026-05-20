# Canonical Subagent Pre-Flight Checklist

Canonical PV (premise-verification, per Catalog #229) checklist for every
subagent BEFORE first Write. Closes empirical bug classes anchored by:

- **NERV-FAMILY-L0-BUILD stand-down 2026-05-20** (~30 min wasted duplicating
  sister commit `18b0beed6`'s ego_nerv + e_nerv + nervdc trainers because PV
  read sister NeRV trainers for the canonical pattern but did NOT run
  `git log -- <target>` BEFORE first Write).
- Sister bug classes covered by Catalog #340 sister-checkpoint guard (commit
  surface) and Catalog #314 absorption detect (post-commit surface).

This document codifies the PRE-WRITE pre-flight pattern as a canonical
subagent prompt-template recommendation (NOT a new STRICT preflight gate per
CLAUDE.md "Gate consolidation discipline" + Catalog #299 quota brake — the
right enforcement surface for PV discipline is the subagent prompt template,
not yet another gate).

## Pre-flight checklist (Catalog #229 PV)

Run **BEFORE first Write** in any subagent's PV step:

### Step 1 — Check for sister-landed commits on target files

```bash
.venv/bin/python tools/check_sister_files_recently_landed.py \
    --files <target-1> <target-2> ... \
    --lookback-hours 6 \
    --own-subagent-id <your-subagent-id>
```

**Exit codes (mirrors sister Catalog #340 helper for habit consistency):**

| Exit code | Verdict | Action |
|-----------|---------|--------|
| `0` | `PROCEED` | Safe to proceed with Writes |
| `8` | `STAND_DOWN_DUPLICATE` | Stand down + clean up; sister already landed equivalent work |
| `9` | `WAIT_AND_REASSESS` | Re-read sister landing memos first; reconsider scope |
| `2` | CLI error | Investigate (e.g. not a git repo, bad args) |

**Bug-class regression**: with this helper in place, the NERV-FAMILY-L0-BUILD
subagent would have caught sister commit `18b0beed6`'s overlap in its PV step
and stood down before writing any of the 3 duplicate trainers. The empirical
anchor (sister landed ~4.5h before duplicate dispatch) is covered by the
default 6-hour lookback window.

### Step 2 — Check for in-flight sister checkpoint conflicts at commit time

Sister of Step 1 at the COMMIT surface (covered by Catalog #340):

```bash
.venv/bin/python tools/check_sister_checkpoint_before_git_add.py \
    --files <target-1> <target-2> ... \
    --label <your-subagent-id>
```

**Step 1 + Step 2 close the multi-subagent edit/commit collision class
bidirectionally:**

- Step 1 (this doc + `check_sister_files_recently_landed.py`) — pre-WRITE,
  git-log surface, catches sisters that **already shipped equivalent work**
  in the lookback window.
- Step 2 (Catalog #340 + `check_sister_checkpoint_before_git_add.py`) —
  pre-COMMIT, in-flight-checkpoint surface, catches sisters that are
  **currently running** with overlapping `files_touched`.

The canonical `tools/subagent_commit_serializer.py` auto-invokes Step 2 via
the STRICT preflight gate Catalog #340. Step 1 is **subagent-prompt
discipline** because the gate would have to scan PV transcripts which is
structurally hard — the right surface is the prompt template.

### Step 3 — Read CLAUDE.md non-negotiables + relevant memory

Per CLAUDE.md "Subagent coherence-by-default" mandatory pre-flight (already
load-by-default):

1. Read CLAUDE.md + AGENTS.md (both files; every NON-NEGOTIABLE marker).
2. Check lane registry `.omx/state/lane_registry.json` for in-flight
   conflicts on your `lane_id`.
3. Check sibling subagents listed in parent prompt's "running in parallel
   right now" section.
4. Read latest top-of-MEMORY.md entries (at least the top 10).
5. Read all `.omx/research/*_directive_*` files dated within the last 24
   hours (operator-routed inter-subagent directives).

### Step 4 — Checkpoint immediately (Catalog #206 crash-resume protocol)

Before doing any work:

```bash
.venv/bin/python tools/subagent_checkpoint.py read \
    --subagent-id <your-subagent-id>
```

If a predecessor checkpoint exists: resume from there, don't restart from
scratch. Per Catalog #206 mandatory crash-resume protocol.

## When Step 1 returns STAND_DOWN_DUPLICATE

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable + Catalog #229
PV discipline:

1. **Do NOT proceed with Writes.** Discard any pending edits.
2. **Read the sister landing memo.** The CLI output cites the sister
   commit's short SHA + subject. Run `git show <sha>` and find the
   corresponding landing memo at
   `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_*.md`.
3. **Report the stand-down** in your final assistant message: cite the
   sister SHA, what work was duplicated, and any unique scope that the
   sister did NOT cover (if applicable).
4. **TaskStop appropriately** — the parent agent needs the stand-down
   signal to either ratify the sister's work or re-route to a genuinely
   distinct scope.

## When Step 1 returns WAIT_AND_REASSESS

Sister activity is ambiguous (touched some target files but not all). Take
these steps before deciding:

1. **Re-read the sister landing memos** for each commit cited in the CLI
   output.
2. **Reconsider your scope.** The sister may have superseded part of your
   intended scope. Update your target file list to remove what the sister
   already shipped.
3. **Re-run Step 1** with the narrowed target file list. If PROCEED:
   continue with the narrower scope. If still WAIT_AND_REASSESS or
   STAND_DOWN: escalate to TaskStop.

## When Step 1 returns PROCEED

Safe to proceed with Writes. The PROCEED verdict means:

- No sister commit touched any of your target files within the
  `--lookback-hours` window (default 6h).
- The git log surface is clean.
- Combined with Step 2 (which the canonical serializer runs at commit
  time), the multi-subagent collision bug class is structurally extinct
  for your work.

Continue with Catalog #229 PV (read full state of target files), then
proceed with Writes.

## Sister discipline references

| Surface | Catalog | Helper |
|---------|---------|--------|
| Pre-WRITE git-log | (no gate; this doc) | `tools/check_sister_files_recently_landed.py` |
| Pre-COMMIT checkpoint | #340 | `tools/check_sister_checkpoint_before_git_add.py` |
| Pre-COMMIT serializer lock | #117 / #157 / #174 | `tools/subagent_commit_serializer.py` |
| Post-COMMIT absorption detect | #314 | (preflight only) |
| Crash-resume | #206 | `tools/subagent_checkpoint.py` |
| Lane registry pre-registration | (lifecycle discipline) | `tools/lane_maturity.py` |

Memory: `feedback_wave_3_pre_write_sister_activity_check_helper_landed_20260520.md`.
Lane: `lane_wave_3_pre_write_sister_activity_check_helper_20260520`.
