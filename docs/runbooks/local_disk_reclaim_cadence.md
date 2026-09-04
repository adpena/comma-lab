# Local disk reclaim — monthly cadence

Sister of the CLAUDE.md "State JSONL archival policy" cadence (`tools/archive_jsonl_state.py`) and of the
"Local Disk, SSD Spill, Auto-Cleanup, And Provenance" non-negotiable. Docs pointer only — CLAUDE.md is
unchanged. Established by ddm_dk1 after the boot volume hit 344 MiB free and ENOSPC'd two arms mid-commit
on 2026-09-04. Full account: `.omx/research/ddm_dk1_local_disk_certify_and_move_reclaim_20260904.md`.

## Read this first: freed bytes are not free bytes

On this machine APFS Time Machine **local snapshots** pin every deleted byte until they are thinned. The
Time Machine destination is a network share, so when it is unreachable the hourly snapshots accumulate and
nothing is ever thinned automatically.

MEASURED 2026-09-04: deleting 32.97 GiB of certified-rebuildable trees moved container free space by
**+1 GiB**. Thinning 21 local snapshots then released **+65 GiB**.

So: if a reclaim's `du` delta does not show up in `df`, the bulk is not the problem.

```bash
tmutil listlocalsnapshots /          # how many are pinning deleted bytes
tmutil destinationinfo               # is the TM destination even reachable?
tmutil thinlocalsnapshots / 40000000000 1    # request ~40 GB, gentlest urgency
```

`thinlocalsnapshots` may thin more than requested — it thinned all 21 for a 40 GB request. It touches no
live file (snapshots are copies), but it does discard OS-managed recovery state, so prefer the lowest
urgency that works and say in the memo that you ran it.

## Monthly

```bash
# 1. What is actually big?
df -h /System/Volumes/Data          # the data volume; `df -h /` shows the sealed system volume
du -sh .omx/tmp/* experiments/results/* 2>/dev/null | sort -rh | head -30

# 2. Certified DELETE — trees git already holds. No copy, no destination cost.
.venv/bin/python tools/local_disk_reclaim.py --plan  --roots .omx/tmp/codex_worktrees
.venv/bin/python tools/local_disk_reclaim.py --apply --roots .omx/tmp/codex_worktrees \
    --ledger .omx/state/disk_reclaim_certs_<UTCDATE>.jsonl

# 3. Certified MOVE — everything else. Never a delete.
.venv/bin/python tools/vertigo_certify_move.py \
    --source <abs path> --source-root /Users/adpena/Projects/pact \
    --dest-root /Volumes/VertigoDataTier/pact/cold_store \
    --ledger .omx/state/disk_reclaim_certs_<UTCDATE>.jsonl \
    --category <class> --reason "<why rebuildable, or why it must be kept>" \
    --workers 6 --apply --retire-source

# 4. Reclaim the bytes you just freed.
tmutil thinlocalsnapshots / <bytes> 1
```

Step 3 is slow (multi-GiB files over Thunderbolt while arms are live); launch it detached via
`tools/launch_detached_process.py --done-receipt <name>` and wait on the receipt.

## The two classes, and why they are different

* **`git_reconstructible` → certified DELETE.** Clean `git status --porcelain` and every ref present as an
  object in the main repo. Copying these to cold store spends destination headroom to store bytes git
  already stores. The cert carries HEAD, ref count, and an executable `rebuild_command`; a `MOVED_TO.json`
  marker is left at the original path. Exercise a rebuild occasionally — an unexercised cert is a promise,
  not a receipt.
* **everything else → certified MOVE.** Run logs are *not* rebuildable; neither is uncommitted work in a
  dirty worktree. These move, never delete, whatever their age.

## Never touch

`upstream/`, `submissions/`, any `retained/` directory, any `.npz`/`.pt` GT cache,
`experiments/results/modal_auth_eval_mirror/`, `.omx/tmp/codex_runs/*.done` and `*.last.txt`, receipts,
seals, `.claude/worktrees/` and `.omx/tmp/claude_cli_worktrees/` (live agent trees), the SSD arm stores,
and anything named by a non-terminal claim inside 24 h in `.omx/state/active_lane_dispatch_claims.md` or
by a live process. `tools/local_disk_reclaim.py` enforces this list; it classifies a protected tree
without even measuring it.

## Destination budget

Waterfall is Vertigo → APDataStore → local by explicit opt-in. Check both before a large move: APDataStore
routinely sits near-full and holds live arm stores. `vertigo_certify_move.py` refuses a destination on the
source filesystem (such a "move" reclaims nothing) and enforces `--min-dest-avail-gib` (default 25).
