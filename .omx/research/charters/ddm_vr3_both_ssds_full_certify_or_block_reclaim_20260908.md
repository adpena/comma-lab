# ddm_vr3 — both SSDs near full: certify-or-block reclaim of REBUILDABLE bulk on Vertigo (18 GiB free) so the live frontier arm can write (charter, 2026-09-08)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm · Spawned by MAIN 2026-09-08 under the operator's standing GO. Storage arm: `score_claim=false`; no scorer runs; this arm never touches archive bytes or any sealed tree's contents.

## MANDATE

Operator 20260908: *"recover and respawn and continue with all"*. MEASURED at spawn: `/Volumes/VertigoDataTier` 1.8 TiB used, **18 GiB free (100%)**; `/Volumes/APDataStore` **48 GiB free (98%)**, ExFAT; local boot 158 GiB free. The live frontier arm (ddm_sj1, pass 4) writes its shard bulk on Vertigo and will stall on ENOSPC; vr2 (`.omx/research/ddm_vr2_vertigo_reclaim_round2_20260831.md`) stopped exactly on destination capacity with a 142-row ledger of certified candidates. The waterfall's usual cure (move to the sister SSD) is closed by the sister being full. What REMAINS legal under the certify-or-block contract: (a) DELETE certified REBUILDABLE scratch — inflated raw frame dumps and parse-back renders that a retained `archive.zip` + runtime tree + decode-identity receipt reproduce deterministically (`0.raw` ≈ 3.66 GB each; there are many); (b) COMPRESS in place where sr3's keep-uncompressed carve-outs allow; (c) MOVE to APDataStore only within its 10 GiB floor. Free **≥ 60 GiB on Vertigo** by (a)+(b) first; report every byte with its certificate.

## SCOPE

1. Recall first: `tools/vertigo_certify_move.py --help`, `tools/local_disk_reclaim.py --help`, `.omx/research/ddm_vr2_vertigo_reclaim_round2_20260831.{md,jsonl}` (the 142-row ledger; "46 headroom-only rows, 288.67 GiB"), `.omx/research/ddm_sr3_ap_certify_compress_reclaim_20260826.md` (compress path + keep-uncompressed carve-outs), `.omx/research/ddm_vr1_vertigo_reclaim_20260820.md`. Reuse the ledger schema; never a hand `rm`/`mv`.
2. Census (read-only, exact): every `*.raw` / `0.raw` / `inflated/` / `parseback/` / `renders/` blob ≥ 1 GiB under `/Volumes/VertigoDataTier/pact/`, with: path, bytes, sha256 (compute; a path/size digest is NOT a hash), the retained archive + runtime tree digest that reproduces it (find the seal/receipt that names it), whether a decode-identity receipt exists (`decoded_field_matches_admitted`, `0.raw sha` in a SEAL or MODAL_REMOTE_RESULT), live references (`lsof +D`, grep of the path in `.omx/`, lane registry, seals), and the live-pointer protection (anything under the CURRENT pointer's tree `ddm_sj1_multipass_token_predistortion/candidate_pass3/` and the two sj1 seals' inputs is PROTECTED — do not touch).
3. Certify-or-block per row (the vertigo_certify_move contract): a row is DELETABLE only when (i) sha256 recorded, (ii) the reproducing archive+runtime are retained with their digests and the decode receipt's `0.raw` sha equals the file's sha, (iii) no live reference, (iv) not under a live-pointer or in-flight tree (ddm_sj1's `pass4/`, `candidate*/`, seals; ddm_gov3/scg1 write nothing there). Everything else is BLOCKED with the named reason. Write the machine-readable ledger `.omx/research/ddm_vr3_reclaim_20260908.jsonl` BEFORE any `--apply`.
4. Apply in order: compress-in-place rows (sr3 path) → delete certified rebuildable rows largest first, re-checking `lsof` immediately before each → stop at ≥ 60 GiB freed or when the certified set is exhausted. Leave a manifest/symlink at each original path if any tool still names it (vr2's "verified original-path symlink" rule). Re-run `df` after each batch and record it.
5. If the certified set < 60 GiB: report the blocked mass by reason class (the denominator) and the top 10 blocked rows with what would unblock each; do NOT lower the bar.
6. Memo `.omx/research/ddm_vr3_both_ssds_full_certify_or_block_reclaim_20260908.md`: freed GiB (before/after df), rows applied/blocked with reasons, certificates, NEXT_IF_RESUMED.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. NO scorer. Never modify, move, or delete: any `archive.zip`, any staged runtime tree, any SEAL, any `MODAL_REMOTE_RESULT.json`, anything under `ddm_sj1_multipass_token_predistortion/` (the live frontier arm is WRITING there now), `submissions/`, `experiments/results/*_codex/submission_dir/`.
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).
- Certify-or-block is the contract: NO deletion without a ledger row carrying sha256 + reproducing-artifact digests + decode receipt + reference scan. Hand-written `rm`/`mv`, local-disk spill, and citation rewrites are CLOSED (vr2 DEAD-ENDS). A row that cannot be certified is BLOCKED, kept, and reported — that is a correct outcome.
- Serializer commits w/ post-edit `--expected-content-sha256` (memo + jsonl; any `.py` change to the certifier = 2 genuine review passes). Tokens `[no-triality] [p0-ledger-ok]`. NEVER a Co-Authored-By or AI-attribution trailer.
- ALWAYS KEEP THE PAYLOAD: a deleted rebuildable file's certificate IS the payload (sha256 + bytes + reproducer). Nothing bulky is created by this arm.
- VERIFIED-AT-SOURCE LAW: the 10 GiB APDataStore floor, the 3.66 GB `0.raw` size, the sr3 carve-out list — mark `verified-at-source: <path:line>` or re-measure.
- EQUATIONS-LEG LAW: storage apparatus; memo carries `# FORMALIZATION_PENDING:storage custody — no measured score law`. Run the Catalog #344 check before the final message.
- DETACHED >30-MIN COMPUTE: hashing tens of 3.66 GB files may exceed 30 min — launch the census hasher via `.venv/bin/python tools/launch_detached_process.py --output-dir <run_dir> --done-receipt ddm_vr3_census.done --nice 10 --nice-best-effort -- <cmd>` with a resumable per-file ledger; monitor with artifact-bound loops (`until [ -f $DONE ]`, ≤ 780 s per foreground call); never `nohup`/`&`/clock waiters. Keep the hasher to ≤ 2 cores (the frontier arm's shards need the CPU; a Metal cell's host thread starves under load).
- Checkpoint discipline: `tools/subagent_checkpoint.py --subagent-id ddm_vr3 …` every ~10 tool uses; `read` first.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- vr2 stopped on DESTINATION capacity, not on certification: 46 headroom-only rows (288.67 GiB) each still need their own hash certificate; moving with APDataStore below the 10 GiB floor is closed — `.omx/research/ddm_vr2_vertigo_reclaim_round2_20260831.md` NEXT_IF_RESUMED + DEAD-ENDS.
- Local disk reclaim frees NOTHING until Time Machine snapshots thin — memory `local_disk_reclaim_frees_nothing_until_tm_snapshots_thin_20260904`; do not spend effort on the boot volume.
- APDataStore is ExFAT: AppleDouble `._*` companions corrupt staged trees and `*.pt` globs (cl3 aborted a rung on it; memory `both_ssds_usable_20260810` ExFAT lesson) — payload blobs only, never trees.
- The "reference" scan must treat historical and live readers alike (vr2 LIVE-HYPOTHESES: 93 conservative protected rows) — a conservative BLOCK is the honest result, not a failure.

## OPTIMAL FORM

- Family exemplar: `tools/vertigo_certify_move.py` at commit `6716cc9c3` and the vr2 landing (`.omx/research/ddm_vr2_vertigo_reclaim_round2_20260831.md` + its 142-row `.jsonl` ledger) — the reference form for certify-or-block storage custody with exact hashes, reference scans, and refusal rows.
- SCOPE reductions declared per row: the census is bounded to blobs ≥ 1 GiB under `/Volumes/VertigoDataTier/pact/` (SCOPE; smaller files are noise against 60 GiB). MECHANISM reductions FORBIDDEN: no path/size digests in place of sha256; no deletion without the reproducer's digest and decode receipt.
- **PRIOR-LAW PREDICTION (falsifiable):** ≥ 15 inflated `0.raw`/parse-back blobs (≥ 55 GiB) under retired candidate trees (fs1/fs2/cl2/rc1/pc1 rungs, jg5, afr1 replays) carry a SEAL or MODAL_REMOTE_RESULT naming their sha and are unreferenced by any live process → certified DELETABLE, freeing ≥ 60 GiB with compress rows. FALSIFIER: the certified set is < 20 GiB — report the blocked mass by reason and stop; do not lower the bar.

## DELIVERABLE

`.omx/research/ddm_vr3_both_ssds_full_certify_or_block_reclaim_20260908.md` + `.omx/research/ddm_vr3_reclaim_20260908.jsonl` — rows: path · bytes · sha256 · reproducer digests · decode receipt · references · verdict (DELETED / COMPRESSED / MOVED / BLOCKED:<reason>) · df before/after. Commit via the serializer. End with `sj1 S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]` (unchanged by this arm).
