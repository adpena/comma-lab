# ddm_sr3 — restore the SSD tier above the 40 GiB fallback reserve by certify-and-move / hardlink dedup (charter, MAIN 2026-09-10)

Both SSDs are under the serializer's mandatory 40 GiB fallback reserve (Vertigo 36 GiB free, APDataStore 32 GiB free at 20:35Z),
so every codex arm's rc-17 fallback bundle now fails (pr15: rc 19) and gdc2's burn enforces a live 12 GiB floor. Free ≥ 80 GiB on
Vertigo and ≥ 60 GiB on APDataStore WITHOUT deleting any uncertified byte.

## Method (binding: certify-or-block; never delete a payload; hash the DESTINATION before removing a source)
1. Measure first: `du -sg` per top-level dir on both volumes; list every retained cold n600 public raw (`0.raw`, 3,662,409,600 B)
   with sha256 — byte-identical raws (move 42 = rlc4's; move 43 = rlc5's; sj1 pass 6 parse-back = move 43's) become HARDLINKS
   with a certificate (`tools/` has the dwc1/rlc1 dedup pattern: `.omx/research/ddm_rlc1_20260910/TWIN_RAW_DEDUP_CERTIFICATE.json`;
   reuse it), never a delete. Expected ≥ 7 GiB per pair on Vertigo.
2. Expired Modal source snapshots under `.omx/tmp/modal_fire_snapshots/` (local disk, 643 MB each; retain 3 days per
   `prune_snapshots`) — prune only through the tool's own prune function; do NOT touch the run3 snapshot
   `20260910T202017Z_ddm_rlc5_first_measurement_t4_run3_20260910` (it holds a ledger row under custody).
3. Certified rebuildable bulk on Vertigo (`evidence/` 329 G, `experiments/` 248 G, `ddm_pk4_20260813` 109 G, `ddm_pfs1_20260729`
   48 G, `ddm_qs1_20260813` 41 G, `pr135_joint_solve_20260810` 32 G): for each candidate, produce a machine-readable certificate
   (original path, bytes, sha256/tree hash, producer memo + commit, rebuild command, false-authority flags) and MOVE to
   `/Volumes/VertigoDataTier/pact/cold_store/` is NOT reclaim — move OFF the full tier only to a destination with space; if no
   tier has space, the honest output is a certified DELETION LIST for the operator (memory law: MOVE labels ≠ custody; #419
   STRICT deletion guard — you may NOT delete; you produce the certified list). Prefer: inflated raw frame trees and decoded
   PNG trees that a retained archive + runtime regenerate deterministically (certify with the archive sha + runtime digest).
4. Every action appends to `.omx/research/ddm_sr3_20260910/RECLAIM_LEDGER.jsonl` (before/after `df`, per-file sha at source
   and destination, certificate path). Memo `.omx/research/ddm_sr3_ssd_reserve_reclaim_20260910.md` with the before/after
   table and the certified deletion list (if any) for the operator. Serializer commit LAST (`REVIEW_GATE_OVERRIDE=1` ok for
   non-.py); rc 17/19 is NOT a stop. Checkpoint as `ddm_sr3`.

## Boundaries
Never touch: live arm dirs (`ddm_gdc2_categorical_coolchic_k8_distill`, `ddm_rlc5_first_measurement_run3*`), sealed candidate
trees and their retained archives/raws EXCEPT to hardlink byte-identical raws with certificates, `upstream/`, the PR tree,
the repo's `.omx/state`. No deletion of any file that lacks a certificate proving deterministic rebuild; no `rm -rf`.

## OPTIMAL FORM
- Reference form: the vr8/sr2 certify-and-move machinery and the dwc1/rlc1 hardlink dedup certificates; scope = the two volumes.
- Provenance pins: `.omx/research/ddm_rlc1_20260910/TWIN_RAW_DEDUP_CERTIFICATE.json` (record sha); `docs/meta_bug_class_catalog.md`
  #419 row (record sha); `.omx/research/ddm_vr8_20260910/FE1_DISPOSITION.json` (record sha).

## Prior negatives accounted (operator 2026-08-15)
- vr7 deleted 11 moved payloads behind live redirects (27 GB) — hash the destination, never trust a MOVE label.
- cs1: local-disk reclaim frees nothing until Time Machine snapshots thin — do not spend effort on local disk.
- sr2: moves between two full tiers reclaim nothing — measure first.

Final message: before/after free GiB per volume, bytes reclaimed per method, the certified deletion list (if any), the serializer
rc, and the frontier line `composition S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (move 43)` unchanged.

<!-- # FORMALIZATION_PENDING: storage hygiene charter; no measured row -->
