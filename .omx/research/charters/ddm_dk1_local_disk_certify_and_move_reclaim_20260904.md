# CHARTER ddm_dk1 — local boot-volume certify-and-MOVE reclaim (P0 hygiene; never delete uncertified bytes)

Tokens: `[no-triality] [p0-ledger-ok]`. Owner: Opus arm. Spawned 2026-09-04 ~21:55Z (real UTC). Source: ng4's P0 report (boot volume hit
344 MiB free; ENOSPC on harness temp files; ps1 hit ENOSPC mid-commit twice). CLAUDE.md "Local Disk, SSD Spill, Auto-Cleanup, And
Provenance" binds: certify or block; move certified rebuildable bulk to the SSD tier; destructive delete only for trivial caches/build
products or explicitly certified rebuildable scratch; leave a manifest/symlink where tools still need the path; no signal loss ever.

## PRIOR-LAW PREDICTION (owed line)
ng4 measured (21:00Z): `.omx/tmp` 208 GB (`arm_receipts_local` 101 GB, `codex_worktrees` 54 GB, `codex_runs` 30 GB), `experiments/results`
149 GB. PREDICTION: ≥ 60 GB is certifiable-rebuildable in 3 classes — (a) stale codex worktrees whose HEAD commits are in the main repo
and that hold NO untracked files (git-prunable, certificate = commit sha + `git status --porcelain` empty); (b) inflated raw-frame / scorer
cache bulk under `experiments/results/*` whose producing archive + runtime sha are recorded (move to `/Volumes/VertigoDataTier/pact/cold_store/`
with a machine-readable cert); (c) `.omx/tmp/codex_runs/*.log` older than 14 days whose `.last.txt` and receipts are retained (move, never
delete). Falsifier: fewer than 30 GB certifiable → report the blocker classes with sizes; do not lower the bar.

## Objective
Free ≥ 60 GB on the boot volume (target ≥ 100 GB free) WITHOUT deleting any uncertified byte. Every move carries a cert row
(`original path, bytes, sha256 or tree hash, producing command/config where known, source archive/runtime shas where applicable,
cold-store destination, rebuildable-reason`) appended to a JSONL ledger under `.omx/state/disk_reclaim_certs_20260904.jsonl` (committed).
Symlink or `MOVED_TO.json` marker at every original path a tool may still read. Use the existing sr2 certify-and-move machinery if it
exists (`grep -rn "certify" tools/*.py | head`; `tools/archive_jsonl_state.py` for over-size state JSONLs) — extend it, never fork.

## Hard boundaries
- NEVER touch: `upstream/`, `submissions/`, the live arm stores on the SSDs, anything referenced by an ACTIVE claim in
  `.omx/state/active_lane_dispatch_claims.md` (24 h TTL) or by a live process (`pgrep -fl python | grep -v grep`), `.omx/tmp/codex_runs/*.done`,
  `*.last.txt`, receipts, seals, `experiments/results/modal_auth_eval_mirror/`, anything under a `retained/` directory, any `.npz`/`.pt` GT
  cache, `experiments/results/ddm_fr2_final_review_20260903/`.
- Live arms right now: fs2 (`/Volumes/VertigoDataTier/pact/ddm_fs2_carrier_resolve/`, CPU), ng4 (fire queued; store on APDataStore), a T4
  custody fire (ps1). Codex quota is exhausted until Sep 7 — no codex worktree is live, but VERIFY each with `git -C <wt> status --porcelain`
  and `git worktree list` before pruning; a worktree with untracked files is NOT prunable — cert-move it instead.
- Destination tiers: `/Volumes/VertigoDataTier/pact/cold_store/` (166 GiB free) first; APDataStore has ~16 GiB — do not fill it.
- Every move is `rsync -a --checksum` then verify sha/tree hash on the destination, THEN remove the source. Delete only after verification.

## Deliverables
1. Measured census (du by class, top 30 directories) BEFORE and AFTER, in the memo.
2. The cert ledger (committed) + the moves executed + `git worktree prune` receipts.
3. A `tools/local_disk_reclaim.py` extension or new tool (if none exists) with `--plan` (dry-run, prints the cert table) and `--apply`;
   fail-closed on any uncertifiable item; tests (≥12) for the classifier + cert schema; wire a monthly cadence note into the existing
   "State JSONL archival policy" section's sister list (docs pointer only, no CLAUDE.md edit).
4. Memo `.omx/research/ddm_dk1_local_disk_certify_and_move_reclaim_20260904.md` with an "Equations leg (`tac.canonical_equations`)" line
   (none expected — say so), blockers, and what you did NOT do.

## OPTIMAL FORM
Reference form = the certify-or-block rule as written; no scope reduction except the never-touch list above. Mechanism reductions: none.

## Rules that bind
NO-FAKE; ALWAYS KEEP THE PAYLOAD; commits ONLY via `tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256 <file>=<post-edit sha>`
with `[no-triality] [p0-ledger-ok]`; NO co-author trailers; .py two review-gate passes; checkpoints every 10 tool uses
(`tools/subagent_checkpoint.py --subagent-id ddm_dk1`); never invent flags; no `/tmp` evidence; long rsyncs detached via
`tools/launch_detached_process.py --done-receipt <distinct>` (foreground >3 min is reaped; launcher refuses argv with "claude"/"codex");
`docs/operating_manual_craft_handoff.md` binds. End with `fs1 S 0.14786319521362173 @ 180,022 B [contest-CUDA T4 n600]`.
