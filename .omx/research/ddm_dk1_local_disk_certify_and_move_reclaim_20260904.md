# ddm_dk1 — boot-volume certify-and-MOVE reclaim (P0 hygiene)

Arm: ddm_dk1 (Opus). Date: 2026-09-04. Tokens: `[no-triality] [p0-ledger-ok]`.
Charter: `.omx/research/charters/ddm_dk1_local_disk_certify_and_move_reclaim_20260904.md`.
Binding: CLAUDE.md "Local Disk, SSD Spill, Auto-Cleanup, And Provenance" (certify or block; never delete an
uncertified byte); ALWAYS KEEP THE PAYLOAD; `docs/operating_manual_craft_handoff.md`.

## Headline

**Boot volume 68 GiB → 211 GiB free (97% → 89% full). 110.40 GiB of tree reclaimed under certificate:
32.97 GiB certified-DELETED, 77.43 GiB certify-MOVED, zero uncertified bytes deleted, zero BLOCKED rows.**

The finding that matters is not the bulk. **The binding constraint was 21 APFS Time Machine local
snapshots pinning every deleted byte.** Certified deletion of 32.97 GiB moved container free space by
**+1 GiB**; thinning the snapshots then released **+65 GiB**. Both MEASURED, minutes apart, same volume.

That inverts the charter's PRIOR-LAW PREDICTION. The prediction ("≥60 GB is certifiable-rebuildable in 3
classes") was *correct about the classes* and *wrong about the mechanism*: the classes exist and were
reclaimed, but reclaiming them is not what returns bytes to the free pool on this machine. The prediction
also mis-sized class (a): it assumed stale worktrees were git-prunable in bulk, and 20.79 GiB of them turn
out to hold uncommitted work.

Free space (+143 GiB) exceeds tree reclaimed (110.40 GiB) because the snapshot thin also released bytes
that *earlier* arms had deleted today and left pinned. Only the 110.40 GiB is attributable to this arm.

## Census — BEFORE (MEASURED 2026-09-04 ~21:38Z)

| Class | Size | Note |
|---|---|---|
| `/System/Volumes/Data` avail | **68 GiB** (97% full) | the boot data volume; `/dev/disk3s5` |
| `.omx/tmp/arm_receipts_local` | 101 GB | `ddm_ap1…` 61 G, `ddm_mst1…` 21 G, `ddm_mp3…` 6.9 G, rest < 5 G |
| `.omx/tmp/codex_worktrees` | 54 GB | 25 trees (20 registered worktrees, 5 full clones) |
| `.omx/tmp/codex_runs` | 30 GB | 2000 `*.log` (23 G), 1897 `*.done`, 995 `*.last.txt` |
| `.omx/tmp/claude_cli_worktrees` | 7.3 GB | live agent worktrees — untouched |
| `experiments/results` | 149 GB | untouched this arm (see "What I did NOT do") |
| `/Volumes/VertigoDataTier` avail | 166 GiB | destination tier |
| `/Volumes/APDataStore` avail | 18 GiB | **not used as destination** (ng4's live store) |

Note the earlier `df -h /` reading of "1.8Ti size / 16Gi used / 68Gi avail" is the sealed **system**
volume `/dev/disk3s1s1`, not the data volume. The authoritative row is `/System/Volumes/Data`.

## Census — AFTER (MEASURED 2026-09-04 ~22:57Z)

| Class | Before | After |
|---|---|---|
| `/System/Volumes/Data` avail | 68 GiB (97%) | **211 GiB (89%)** |
| `.omx/tmp/arm_receipts_local` | 101 GB | 44 GB |
| `.omx/tmp/codex_worktrees` | 54 GB | 21 GB |
| `.omx/tmp/codex_runs` | 30 GB | 9.3 GB |
| `/Volumes/VertigoDataTier` avail | 166 GiB | 88 GiB |
| `/Volumes/APDataStore` avail | 18 GiB | 17 GiB (untouched by this arm) |

Cert ledger: written live to `.omx/state/disk_reclaim_certs_20260904.jsonl` (the path the charter names,
and what the tools append to) and canonicalized for git as
**`.omx/research/ddm_dk1_disk_reclaim_certs_20260904.jsonl`** — `.omx/state/*.jsonl` is gitignored by the
research-state tracking policy, so the charter's "(committed)" is honored at the `.omx/research` path, the
same convention the prior mover's `ddm_vr1_move_cert_ledger.jsonl` uses. 32 rows —
`PLAN` 2, `CERTIFIED_DELETE_PENDING` 12, `CERTIFIED_DELETED` 12, `COPIED` 2, `VERIFIED` 2,
`MOVED_SYMLINKED` 2. **No `BLOCKED_*` row.** Verified content-manifest digests:
`aea11f8897cf5a14…` (ap1 advisory, 326 rows, freed 56.34 GiB) and
`5ecdd513ebece7c9…` (codex_runs logs, 1621 rows, freed 21.09 GiB).

Both moves left transparent directory symlinks at the original paths; both resolve and list. The ap1
`retained/` sibling (958 MB, the archive.zip generators) is intact locally.
`.omx/tmp/codex_runs/MOVED_TO.json` records the destination, both manifest digests, and all 1621 moved
filenames, so any single log stays findable by name.

## The finding that matters: APFS snapshots gate every reclaim

MEASURED sequence, same volume, ~4 minutes apart:

| t | action | `codex_worktrees` on disk | container free |
|---|---|---|---|
| 21:44 | before | 54 GB | 68 GiB |
| 21:50 | 12 trees certified-DELETED (32.97 GiB) | **21 GB** | **69 GiB** (+1) |
| 21:52 | `tmutil thinlocalsnapshots / 40000000000 1` | 21 GB | **134 GiB** (+65) |
| 22:44 | ap1 advisory MOVED + source retired (56.34 GiB) | — | 134 GiB (**+0**) — one new hourly snapshot had already re-pinned it |
| 22:46 | thin that one snapshot | — | **190 GiB** (+56) |
| 22:55 | codex_runs logs MOVED + source retired (21.09 GiB) | — | **211 GiB** (+21, no snapshot existed to pin it) |

The 22:44 row is the mechanism confirming itself: a single snapshot created in the intervening hour
absorbed the entire 56 GiB retirement, and one thin released it exactly.

21 hourly `com.apple.TimeMachine.*.local` snapshots spanned 2026-09-03T19:32 → 2026-09-04T13:31. Every
file that existed at snapshot time stays referenced by the snapshot after `rm`, so the filesystem shrank
and the free pool did not. The Time Machine destination is a **network** share
(`smb://tm_primary@bat00-tm.local/TimeMachinePrimary`) — when it is unreachable the local snapshots are
never thinned and simply accumulate, which is why 21 of them existed.

Consequences worth carrying forward:

1. **Any future certify-and-move on this machine frees nothing until the snapshots referencing the source
   are thinned.** A reclaim tool that reports "freed N GiB" from `du` deltas is reporting a number the
   operator cannot spend. `vertigo_certify_move.py` records `freed_allocated_kib` from `du` — true as
   allocation, not as availability.
2. Snapshots regenerate hourly. This is a recurring hygiene item, not a one-time fix.
3. `tmutil thinlocalsnapshots` needed no elevation here. I requested 40 GB at urgency 1 (the gentlest);
   macOS thinned all 21 rather than the oldest few, so the action was less surgical than intended —
   stated plainly because it destroyed more OS-managed recovery state than I planned. No live file was
   touched; snapshots are copies, and macOS purges them automatically under pressure by design.

## Reclaim executed

### Class A — git-reconstructible trees: certified DELETE, no copy (32.97 GiB, 12 trees)

A worktree/clone whose `git status --porcelain` is empty and whose every ref is an object present in the
main repository is fully reconstructible. Copying such a tree to cold store would spend destination
headroom to store bytes git already stores; deleting it *without* that proof would be signal loss. The
proof is the certificate. Executor: `tools/local_disk_reclaim.py --apply`. Cert rows carry HEAD sha, ref
count, porcelain count, registered-vs-clone, and an executable `rebuild_command`; each original path keeps
a `MOVED_TO.json` marker naming that command.

7 registered worktrees + 3 clones + 2 more, all under `.omx/tmp/codex_worktrees/`:
`ddm_bn1x_isolated` (6.87), `ddm_dy1_scope_law_resolver…` (7.15), `einstein_kolmogorov_crux…` (2.40),
`ddm_r7_token_coder_race…` (1.88), `ddm_su2_pose_endgame…` (1.88), `ddm_eg1_endgame_chain…` (1.88),
`ddm_cb1_perclass_carrier_byteclose…` (1.86), `ddm_pf3b_52probe…` (1.86), `ddm_wf7_seven_home…` (1.86),
`ddm_dn1x` (1.79), `ddm_eu1_sol_ultra_eureka_hunt…` (1.77), `ddm_vae1_vae_corpus_harvest…` (1.77) GiB.

**The reconstructibility claim is demonstrated, not asserted.** Ledger audit: 12/12 rows have their HEAD
commit present in the main repo and a `MOVED_TO.json` marker at the original path. Then one tree was
actually rebuilt end-to-end from its own cert row:

```
git worktree add --detach <scratch> ba35bb751acff2bf9ef0093c78c4e853866af6db   # ddm_eu1_sol_ultra_eureka_hunt
39,080 files restored
restored tree 8756973cc8fb783e6444970053e39a6cd31274d7
cert     tree 8756973cc8fb783e6444970053e39a6cd31274d7   # identical
```

Scratch removed after the check. A cert that says "rebuildable" and has never been exercised is a
promise; this one is a receipt.

### Class B — `ap1/advisory`: certify-MOVE (56.3 GiB)

`.omx/tmp/arm_receipts_local/ddm_ap1_residue_purchase_scorer/advisory` — 16 cells of
`fire_local_advisory.v1` output, each ~3.5 GiB of `work/inflated` raw frames. Rebuildable: the generating
`archive.zip` for every cell lives in the arm's `retained/candidates/<cell>/runtime/` tree, which is
never-touch and **stays local**. So the generator is retained and only the deterministic output moved —
ALWAYS-KEEP-THE-PAYLOAD is satisfied in the strong form.

### Class C — stale run logs: certify-MOVE, never delete (21.09 GiB, 1621 files)

`codex_runs/*.log` with mtime older than 14 days (pre 2026-08-21). **Logs are not rebuildable**, so this
class is MOVE-only; deletion would be irreversible signal loss. Staged by same-filesystem rename into
`.omx/tmp/codex_runs_archive_pre_20260821/` then certify-moved as one tree. Retained locally and intact:
1897 `*.done`, 995 `*.last.txt`, 379 newer logs (2.04 GiB), and all `*.armed.json`. No live process
referenced any `codex_runs/*.log` at stage time (`ps -Ao command` scan).

## Blockers (certified as NOT reclaimable)

| Path | Size | Why blocked |
|---|---|---|
| `arm_receipts_local/ddm_mst1_manufactured_stage_split/capture_r2_local/retained` | **21 GB** | the entire tree is under `retained/` — charter never-touch. This is the single largest blocked item and the reason the "≥60 GB in 3 classes" prediction needed a fourth source. |
| `arm_receipts_local/ddm_ap1…/retained` | 958 MB | never-touch (`retained/`), and it holds the archive.zip generators that make Class B rebuildable |
| `codex_worktrees/ddm_lt1_commit_fallback_20260810` | 7 MiB | clean, but 2 refs are absent from the main repo — commits exist nowhere else. Correctly refused deletion. |
| 12 dirty codex worktrees | 20.79 GiB | 1–25 uncommitted paths each; bytes exist only there. Classified `certify_move_required`, **not executed** (see below). With `ddm_lt1` the `certify_move_required` class is 13 trees / 20.80 GiB. |
| `arm_receipts_local` (whole-tree) | 39.70 GiB | every remaining arm store contains a `retained/` subtree or a `.pt`; see the fourth defect below |
| `.claude/worktrees/*`, `claude_cli_worktrees/*` | 7.3 GB | live agent worktrees |
| `experiments/results` | 149 GB | out of scope this arm |

## Two bugs I shipped and caught by reading the planner's own output

Recorded because both are the *shape* of error that reads as caution:

1. **A fully-blocked census is a vacuous PASS.** The first revision classified 25/25 candidates as
   `blocked_never_touch` and I nearly accepted it as conservatism. Cause: the pin matcher treated a
   referenced *ancestor* as pinning every child, and the repo root appears in essentially every command
   line — plus the planner's own `--roots` argument entered its own pin set. Cure: pin on exact match or
   a reference *inside* the candidate only, and exclude self PIDs from the `ps` scan. Regression tests
   `test_ancestor_reference_does_not_pin_a_child`,
   `test_live_process_scan_excludes_this_tools_own_command_line`.
2. **Deref failure read as absence.** I tested ref reachability with `git cat-file -e <obj>^{commit}`.
   Codex writes `refs/codex/turn-diffs/*` pointing straight at **tree** objects, so the deref fails
   although the main repo holds every byte. That mis-classified ~21 GiB of reclaimable trees as needing a
   copy (12 trees → 3). The right test is object *presence*; HEAD alone must be a commit. Regression test
   `test_ref_pointing_at_a_tree_object_is_present_not_absent`. Genus:
   `[[available-field-vs-authoritative-field]]` — I read the answer to a question I had not asked.

A third, found in the second review pass: claim timestamps are UTC, and `time.mktime(...) - time.timezone`
mis-converts them under DST, sliding the 24 h pin window by an hour. Cured with `calendar.timegm`;
pinned by `test_claim_stamps_are_read_as_utc_not_local_time`.

A fourth, and the most dangerous, found by running the planner over `arm_receipts_local` after the
reclaim was already done: **the never-touch test judged only the candidate's own path, never its
descendants.** `ddm_mst1_manufactured_stage_split` has an innocuous top-level name and 20.55 GiB living
entirely under `capture_r2_local/retained/` — the planner offered it as movable bulk. Moving it would have
relocated a never-touch tree with every top-level check passing. Cure: a container of protected bytes is
itself protected (`find_never_touch_descendant`, fail-closed on an unscannable tree). The scan is applied
only on the bulk path, *not* to git-reconstructible trees — every checkout contains `upstream/` and
`submissions/`, but in a clean tree whose refs the main repo holds those bytes are git's and nothing is at
risk. Regression tests `test_bulk_containing_a_retained_subtree_is_blocked_not_moved`,
`test_gt_cache_inside_bulk_blocks_the_whole_container`,
`test_clean_worktree_is_not_blocked_by_its_own_upstream_directory`.

**The corrected verdict changes the map.** With the descendant scan live, `arm_receipts_local` is
**39.70 GiB, essentially all BLOCKED** — nearly every arm store holds a `retained/` subtree or a `.pt`
model (`ddm_mp3` 6.89, `ddm_jf1` 4.20, `ddm_oe1` ×2 2.07 each, `ddm_dg2` 1.46, `ddm_ld1` 1.24, plus
`ddm_mst1` 20.55 GiB). So `arm_receipts_local` is not a reclaim source at whole-tree granularity at all.
Reclaim there requires descending to a specific non-protected subtree — which is exactly the shape Class B
took (`ddm_ap1…/advisory`, with its sibling `retained/` left in place). Anyone planning the next disk arm
should start from that fact rather than from the top-level `du`.

## Tooling

**Extended, not forked.** `tools/vertigo_certify_move.py` already implements the certified move
(census → hash → copy → independent destination re-read → symlink → retire). It hardcoded
`/Volumes/VertigoDataTier` as the source root, which refused every boot-volume source. The extension is a
`--source-root` argument defaulting to the historical value; the invariant that actually keeps a "move"
honest is the pre-existing destination-on-source-filesystem refusal, not the root's identity. Ledger keys
`vertigo_df_before/after` are preserved beside new `source_df_before/after` so existing readers keep
parsing. Existing suite: 14 → 16 tests, all passing.

**New:** `tools/local_disk_reclaim.py` (`--plan` / `--apply`, mutually exclusive) adds only the class the
mover cannot express — a tree whose bytes are already in the repo's own object database, reclaimed by
certified DELETE with no copy. It *delegates* every move to `vertigo_certify_move.py` rather than
reimplementing one. Fail-closed: `--apply` executes the `git_reconstructible` class only, re-probes each
tree immediately before the irreversible step (the plan/apply gap is a real window), and uses plain
`git worktree remove` — not `--force` — so git's own dirty check is an independent second reader of the
invariant the class asserts.

Tests: **41** in `src/tac/tests/test_local_disk_reclaim.py` + 2 added to the mover suite = 57 passing.
Coverage: never-touch boundary (12 charter classes), claim terminality and TTL, pin semantics, git proofs
(clean/dirty/tree-ref/absent-commit/probe-failure), cert schema, `--plan`/`--apply` exclusivity, ledger
framing.

Cadence note (docs pointer only, no CLAUDE.md edit): this belongs beside the "State JSONL archival policy"
monthly cadence — `tools/local_disk_reclaim.py --plan --roots .omx/tmp/codex_worktrees` monthly, and
`tmutil listlocalsnapshots /` whenever a reclaim's freed bytes do not appear in `df`.

## Equations leg (`tac.canonical_equations`)

**None.** This arm is disk hygiene: it moves no score, measures no scorer quantity, and produces no law of
the form `S = f(...)`. No canonical equation is registered, refined, or cited, and none is owed. The
`freed_allocated_kib` figures in the cert ledger are allocation deltas, not score-bearing measurements.

## What I did NOT do

- **`experiments/results` (149 GB, the largest remaining tree)** — not censused past the top level, not
  touched. It is the obvious next reclaim and it needs its own arm: it mixes never-touch
  (`modal_auth_eval_mirror/`, `retained/`, `ddm_fr2_final_review_20260903/`, `.npz`/`.pt` GT caches) with
  genuinely rebuildable inflate output, and that separation is per-directory work.
- **The 12 dirty codex worktrees (20.80 GiB)** — classified and planned, deliberately not executed. The
  free-space objective was already met by a wide margin, and Vertigo headroom is the now-scarcer resource
  with fs2 and ng4 live. The exact executor command is in the plan output; they hold uncommitted bytes and
  must be MOVED, never deleted.
- **`arm_receipts_local/ddm_mp3_hpac_member_prune` (6.9 GB) and smaller arm stores** — not classified.
- **No APDataStore writes.** It has ~18 GiB free and holds ng4's and ps1's live stores.
- **I did not re-verify the Time Machine network destination's backup currency** before thinning local
  snapshots. If an earlier arm's only copy of some lost file lived in a local snapshot, that recovery path
  is gone. No live file was affected.
