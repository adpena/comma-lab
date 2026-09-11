# ddm_sr5 — restore BOTH SSD tiers above their 40 GiB fail-closed reserves by certify-or-block deletion of REBUILDABLE bulk in CLOSED stores (charter, MAIN 2026-09-11; operator full-authority GO)

## Measured state (2026-09-11 ~16:10Z)
Vertigo 38 GiB free, APDataStore 49 GiB free; both reserves are 40 GiB; usable headroom across both ≈ 7 GiB. Every cold n600
decode needs ~3.7 GB of scratch; ntb2's candidate proof already had to be routed to APDataStore; a certified tar-wrap move of
`ddm_hm1_20260810` (10.9 GiB) was correctly REFUSED by `tools/vertigo_certify_move.py` (dest would fall to 38.2 GiB). Moving
between the two tiers nets nothing (sr4 already offloaded 31 GiB of APDataStore onto Vertigo at `ddm_sr4_20260910/retained/ap_offload`).
The local disk is NOT a tier (operator 08-20: SSD = artifacts only) — do not use it; MAIN has surfaced that decision to the operator.
Survey (top of `du -sk /Volumes/VertigoDataTier/pact/*`): evidence 304.6 GiB · experiments 247.3 GiB · ddm_pk4_20260813 108.2 GiB
(cpu_authority_run 108 GiB) · cold_store 56.2 · ddm_pfs1_20260729 47.9 · ddm_qs1_20260813 40.9 · pr135_joint_solve_20260810 31.4 ·
ddm_sr4_20260910 30.7 (ap_offload; leave) · public_datasets 25.0 · hprc_projection_gap_repairs 23.8 · ddm_js5_20260812 23.3 ·
ddm_pz4_joint_target_conditioned_receiver 23.3 · ddm_mc35_20260814 17.1 · hprc_residual_transform_full600_sweep 15.9 · snerv_mistake_b 15.1.

## Deliverable: free ≥ 60 GiB on Vertigo AND keep APDataStore ≥ 45 GiB, deleting ONLY certified-rebuildable bytes
The rule (CLAUDE.md "Local Disk, SSD Spill, Auto-Cleanup, And Provenance"): destructive delete is allowed only for trivial
caches/build products or EXPLICITLY CERTIFIED rebuildable scratch; every other cleanup is move/cold-store; never delete or move a
large artifact unless a machine-readable record preserves deterministic reproducibility (original path, bytes, sha256/tree hash,
command/config/argv/env, source archive/runtime/content hashes, cold-store destination when moved, false-authority score flags, and
the reason it is rebuildable). Use the sr4 machinery (`.omx/research/ddm_sr4_ssd_reserve_reclaim_20260910.md` §CENSUS AND CERTIFIED
DELETION LIST; its producers) and `tools/vertigo_certify_move.py`; extend them, never hand-roll `rm`.
1. Census the CLOSED stores above (an arm is closed iff its memo/verdict landed and no live process, claim, or pointer receipt cites
   its payload as custody: check `.omx/state/active_lane_dispatch_claims.md`, the canonical pointer, the seal/intent receipts under
   `.omx/research/ddm_rlc5_20260910/` and `.omx/research/ddm_pc3_20260911/`, and `ps`). NEVER touch: any live arm's directory
   (ddm_pc3_*, ddm_ntb2_*, ddm_mxo2/3, ddm_rbf1 until its receipt lands, ddm_pc3_fire_move45), sealed pointer trees (ddm_rlc5_cure_on_move43,
   ddm_sj1_pass6, ddm_rp1_round2), `ddm_sr4_20260910/retained/ap_offload`, `public_datasets`, `cold_store`, or anything a pointer
   receipt names as a retained payload (the raw-identity 0.raw files of the last three moves stay).
2. For each candidate payload class (inflated raw frame trees, scorer tensor caches, decoded PNG trees, NPZ/VJP shards, duplicate
   checkpoints, provider workspaces, `.partial` files), prove rebuildability BEFORE deletion: name the exact command + inputs (archive
   sha, runtime tree digest, seed, upstream sha) that regenerates the bytes, and for at least ONE payload per class actually re-run
   it on a small slice and hash-match (the certify-or-block law is not a form to fill in). Hardlink-dedup byte-identical raws first
   (sr4's method) — that frees without deleting.
3. Write the per-file cert rows (JSONL, fcntl-locked append, schema from sr4) BEFORE each deletion; delete; verify `df`; record
   before/after in the ledger `.omx/research/ddm_sr5_certified_deletions_20260911.jsonl`. Stop at the target; do not over-reclaim.
4. Memo `.omx/research/ddm_sr5_certified_rebuildable_deletion_20260911.md`: the census table (store → class → bytes → verdict →
   cert path), freed GiB per tier, what was refused and why, the false-authority flags where a store held advisory scores.

## Boundaries
No Modal, no scorer runs, no candidate work; never edit `upstream/`, the PR tree, or any arm's files; `.py` changes only inside the
sr4/sr5 producers; serializer commits with post-edit shas, two visible review passes per .py, no co-author trailer, tags
`[no-triality] [p0-ledger-ok]`; memo needs `# FORMALIZATION_PENDING:<rationale>`. Checkpoint as `ddm_sr5`. A Git-object denial is not a stop.

## OPTIMAL FORM
- Reference form: sr4's landed certify-and-move + hardlink-dedup (memo sha recorded in your memo) and `tools/vertigo_certify_move.py`
  (its census/manifest/verify functions). Scope: the closed stores named above first, largest first. No mechanism reduction: every
  deletion carries a cert whose rebuild command was exercised at least once per class.
- Provenance pins: this charter; sr4 memo; the survey (MAIN scratchpad, copy its top-40 rows into your memo); `df` before/after.

## Prior negatives accounted (operator 2026-08-15)
- vr7 (2026-09-10): an apply step DELETED 11 moved payloads behind live redirects — hash the DESTINATION before retiring a source;
  a label is not custody (memory `moved_labels_are_not_custody_vr7_deleted_moved_payloads_behind_live_redirects_20260910`).
- sr4: "one of two separate 3,662,409,600-byte raws can release at most 3.411 GiB" — dedup math per file, not per pair.
- Local-disk reclaim frees nothing until Time Machine snapshots thin — irrelevant here (SSD tiers), noted so you do not chase it.
- ExFAT leaves AppleDouble stubs; skip `._*` in every census (landing_an_arm_bundle memory).
