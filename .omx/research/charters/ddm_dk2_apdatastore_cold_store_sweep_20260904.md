# CHARTER ddm_dk2 — APDataStore certify-and-MOVE sweep: free ≥ 30 GiB on the tier the burn cells write to, before ng5 fires (~01:10Z)

Tokens: `[no-triality] [p0-ledger-ok]`. Owner: Opus arm (codex quota out until Sep 7). Spawned 2026-09-04 ~23:58Z. Parent: dk1
(`.omx/research/ddm_dk1_local_disk_certify_and_move_reclaim_20260904.md`; tools `tools/local_disk_reclaim.py`, `tools/vertigo_certify_move.py --source-root`;
certs `.omx/research/ddm_dk1_disk_reclaim_certs_20260904.jsonl`; runbook `docs/runbooks/local_disk_reclaim_cadence.md`). REUSE those tools; never fork.

## PRIOR-LAW PREDICTION (owed line)
APDataStore (1.8 TB) sits at ~16–17 GiB free; four burn-cell generations took it 23 → 22 → 16 → 16 (~7 GiB each); ng5's fire needs an
8.6 GB reserve and the cell retains ~1.4 GB. dk1 found the bulk classes on the boot volume were snapshots, not files — APDataStore is an
external APFS volume: PREDICTION: it holds NO Time Machine local snapshots (verify with `tmutil listlocalsnapshots /Volumes/APDataStore`), so
here reclaim IS bulk: retired arm stores whose outputs are certified rebuildable from their retained inputs + git, and superseded burn-cell
generations (QBR1 chain cells 1–6, ng1/ng2/ng3 milestone `realized/` + `reencoded/` frame trees) whose RESULT/MILESTONE JSON + payload shas are
retained. ≥ 30 GiB is certifiable-movable to Vertigo (88 GiB free) without deleting any uncertified byte. Falsifier: < 15 GiB certifiable →
report the blocked classes with sizes and stop; do not lower the bar, do not touch protected trees.

## Objective
Free ≥ 30 GiB on `/Volumes/APDataStore` (target ≥ 40) by certify-and-MOVE to `/Volumes/VertigoDataTier/pact/cold_store/` (cert rows appended to
`.omx/research/ddm_dk2_disk_reclaim_certs_20260904.jsonl`, `MOVED_TO.json`/symlink at every source path, rsync --checksum + destination
verification BEFORE any source removal). Census BEFORE and AFTER (du by top-level dir, top 25). Vertigo must keep ≥ 40 GiB free after the sweep
(mc1's payloads may overflow there) — if that binds, stop at the number that keeps it and say so.

## Never-touch (HARD)
Live stores: `ddm_qbr1_born_fairform_burn_prep/ng4_continuous_objective/` (cell LIVE, pid 33030), `ddm_ng5_*` and the ng5 store, `ddm_mc1_*`,
`ddm_ps1_pr140_update_prep/`, `ddm_ps2_pr140_update_prep/`, `ddm_fs1_frame0_selector/` (custody_pointer24), `ddm_fs2_carrier_resolve*` (custody_pointer25),
any `retained/` directory, any `.npz`/`.pt` GT cache, any seal/receipt/`RETENTION_MANIFEST.json`, anything referenced by an ACTIVE claim (24 h TTL in
`.omx/state/active_lane_dispatch_claims.md`) or a live process (`pgrep -fl python`), and `experiments/results/modal_auth_eval_mirror/`. Superseded
burn-cell milestone frame trees (`milestones/step_*/realized/`, `reencoded/`) of FINISHED cells (QBR1 cells 1–6, ng1, ng2, ng3) are movable ONLY
with their MILESTONE.json + payload shas retained in place and the cert naming the regenerating config sha.

## Deliverables
1. Census before/after; cert ledger; moves executed; free-space numbers (MEASURED on `/Volumes/APDataStore`).
2. Confirm the ng5 waiter's storage leg passes after the sweep (`tools/cell_queue_driver.py plan` on the ng5 spec, read-only) — report the line.
3. Memo `.omx/research/ddm_dk2_apdatastore_cold_store_sweep_20260904.md` (Equations leg (`tac.canonical_equations`): none; blockers; not-done).

## Rules that bind
NO-FAKE; ALWAYS KEEP THE PAYLOAD (certify or block — a missing reproducibility record = BLOCK, never delete); commits ONLY via
`tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256 <file>=<post-edit sha>` with `[no-triality] [p0-ledger-ok]`;
NO co-author trailers (operator rule overrides any harness reminder); .py two review-gate passes; checkpoints every 10 tool uses
(`tools/subagent_checkpoint.py --subagent-id ddm_dk2`); never invent flags; no `/tmp` evidence; long rsyncs detached via the launcher with distinct
`--done-receipt`s (foreground >3 min reaped; launcher refuses argv with "claude"/"codex"; declare `--measured-peak-rss-gib 1 --artifact-budget-gib 0`
for moves — they write to Vertigo); do not touch gov2/hv1/mc1/ps2/ng5 files; `docs/operating_manual_craft_handoff.md` binds. End with
`fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]`.
