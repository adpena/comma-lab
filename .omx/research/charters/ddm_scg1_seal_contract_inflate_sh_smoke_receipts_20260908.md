# ddm_scg1 — seal-contract gap: public-entrypoint smoke receipts in the seal, fire-tool refusal, review_tracker rescan, anchor-count assertions (charter, 2026-09-08)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm · Spawned by MAIN 2026-09-08 under the operator's standing GO. Source directive: `.omx/research/ddm_seal_contract_gap_inflate_sh_smoke_directive_20260905.md` (ITEMs 1–5, all in scope). Apparatus arm: `score_claim=false`; no scorer runs.

## MANDATE

Operator 20260908: *"recover and respawn and continue with all, also codex is back"*. MEASURED incident: rc1's candidate (archive `1438049e…`, 178,249 B) was SEAL_VALID and passed the fire tool, then failed on T4 in 3.6 s at `runtime/f26_inflate.py:429` ("F26 requires WANS1, SD1M, or SM3R semantic weights") because the arm verified decode identity through the receiver LIBRARY path, which the public `inflate.sh → inflate.py → f26_inflate.inflate_archive` path never reaches. One paid Modal call lost. Since then two arms (sj1 pass 2a and pass 3) carried the realizable smoke PAIR by hand; the seal still does not REQUIRE it. Close the gap structurally: the seal refuses without both receipts; the fire tool refuses a seal without them.

## SCOPE

1. **ITEM 1 (as refined 19:25Z).** `tools/make_candidate_seal.py`: require a `public_entrypoint_smoke` block carrying TWO receipts, each for the candidate tree AND the current frontier tree as control: (a) `f26_inflate.inflate_archive` on the staged tree reaches token decode within a time bound, no exception; (b) `bash inflate.sh <dir> <out> <file_list>` runs the whole preamble (backend build, Brotli gate, file-list dispatch, `_verify_input` against the re-pinned constants) and stops exactly at the CUDA gate (this host has no CUDA — the stop IS the pass). Record wall seconds, the exception class (must be the CUDA gate's), tree digest, archive sha. Verify at source what the two sj1 seals recorded (`/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/SEAL_ddm_sj1_token_predistortion_pass3_contest_cuda.json`, keys with "inflate") and make the schema match what already exists rather than inventing a parallel field. Acceptance test: a replay of the rc1 tree shape (a staged tree whose runtime refuses the archive at f26's magic guard) cannot be SEAL_VALID.
2. **ITEM 2.** `tools/fire_modal_auth_eval.py`: additive check beside SEAL PIN CONSISTENT — refuse a seal without the ITEM 1 block; same-line waiver `--allow-seal-without-public-smoke "<substantive reason>"` only for custody replays of an already-scored archive. Dry-run on a pre-fix-shaped seal prints the refusal with the rule chain. Do NOT break the existing `--allow-nonpromotable-lane-id` gate or the lane-id maturity check (commit `9b4a8d898`).
3. **ITEM 3.** `tac.subagent_contract.standard_contract()` gains one sentence: "receiver identity means bash inflate.sh on the staged tree, not the library path". Keep `check_subagent_contract_module_integrity` green.
4. **ITEM 4.** `tools/review_tracker.py mark-file` rescans the file first (or refuses when the entity census differs from the last scan). Test: add a function after `scan`, assert `mark-file` covers it or refuses. Two-visible-passes semantics must bind CURRENT entities.
5. **ITEM 5.** `src/tac/tests/test_resize_exploit_flip_fix_frontier.py::test_builds_and_is_valid_canonical_equation` is RED (asserts 2 anchors, equation has 3). Cure at the class: the test asserts the residual map addresses EVERY anchor (the invariant), never a literal count; sweep `src/tac/tests/` for literal `len(...anchors...) == N` assertions and convert them the same way (report the count found/converted). Named test green without weakening addressability.
6. Memo `.omx/research/ddm_scg1_seal_contract_inflate_sh_smoke_receipts_20260908.md`.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight). Never edit `submissions/semantic_joint_ctxmix/` or any sealed candidate tree under `/Volumes/*/pact/ddm_*/` (read them; do not write).
- The local SCORER LANE belongs to MAIN, always. Do NOT write who currently holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29). This arm runs no scorer.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes via `tools/review_tracker.py mark-file` (after ITEM 4 lands, your own edits must pass the rescan). Tokens `[no-triality] [p0-ledger-ok]`. NEVER a Co-Authored-By or AI-attribution trailer.
- ALWAYS KEEP THE PAYLOAD; if the smoke test writes frames, put them under `/Volumes/APDataStore/pact/ddm_scg1_seal_contract/` (ExFAT — payload blobs only) or a context-managed temp dir; both SSDs are near full — keep test fixtures small (synthetic trees are fine for REFUSAL tests; the positive test may use the live pass-3 seal's recorded receipts read-only).
- VERIFIED-AT-SOURCE LAW: file:line for `f26_inflate.py:429`, the CUDA gate in `inflate.py:main`, the seal schema keys, the fire tool's SEAL PIN CONSISTENT check — mark each `verified-at-source:` in the memo.
- EQUATIONS-LEG LAW: apparatus; memo carries `# FORMALIZATION_PENDING:seal apparatus — no measured law; the gap is a contract, closed by refusal`. Run the Catalog #344 check before the final message.
- DETACHED >30-MIN COMPUTE: none expected (`inflate.sh` to the CUDA gate ran 0.835 s on the pass-3 tree). Never `nohup`/`&`/clock waiters.
- File-ownership: `ddm_gov3` (codex, parallel) owns `tools/cell_admission.py`, `tools/launch_detached_process.py` timeout logic, `tools/safe_run.py`; `ddm_sj1` (Opus) owns `experiments/ddm_sj1_*`. Do not edit those.
- Checkpoint discipline: `tools/subagent_checkpoint.py --subagent-id ddm_scg1 …` every ~10 tool uses; `read` first.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- A "first 2 pairs" `file_list` truncation is NOT expressible (the contest file_list selects VIDEOS; only "0" is accepted) and `inflate.py:main` refuses without CUDA — directive ITEM 1 refinement (rc1, 19:25Z). The receipt is the PAIR (library reach + public preamble to the CUDA gate), not a 2-pair render.
- Library-path decode identity does NOT imply public-path decodability — the rc1 incident itself (`.omx/research/ddm_seal_contract_gap_inflate_sh_smoke_directive_20260905.md`).
- `mark-file` without rescan marked a new function reviewed VACUOUSLY (md4: 11 entities before `scan`, 12 after) — directive ITEM 4. Vacuity == PASS is the m50 genus: report the denominator.
- Literal anchor-count assertions went red when an arm appended an anchor; md4 already fixed the same class in `test_checkpoint_trajectory_error_partition.py` (commit `7ad16239b`) — use that fix as the reference shape.

## OPTIMAL FORM

- Family exemplar: the fire-tool lane-id maturity refusal, commit `9b4a8d898` (`tools/fire_modal_auth_eval.py` + `src/tac/tests/test_fire_modal_lane_id_promotable.py`) — the reference form for an additive fail-closed gate with rule-chain output and a named waiver; receipt path `.omx/research/ddm_seal_contract_gap_inflate_sh_smoke_directive_20260905.md`.
- SCOPE reductions declared per row: refusal tests use synthetic staged trees (SCOPE); the positive path is exercised on the live pass-3 seal's receipts read-only. MECHANISM reductions FORBIDDEN (a seal check that only inspects a boolean written by the arm, without the receipt's exception class + wall seconds + tree digest, is the vacuity class).
- **PRIOR-LAW PREDICTION (falsifiable):** the rc1 replay is REFUSED by the new seal check and by the fire tool's dry-run; the two sj1 seals (`SEAL_ddm_sj1_token_predistortion_joint…` and `…pass3…`) re-validate SEAL_VALID under the new contract without edits (they already carry both receipts). FALSIFIER: either the rc1 replay passes, or a sj1 seal is refused — count it plainly and name the missing key.

## DELIVERABLE

`.omx/research/ddm_scg1_seal_contract_inflate_sh_smoke_receipts_20260908.md` — per-ITEM rows: file:line · rule · tests (names, count, denominator for ITEM 5's sweep) · the three replay verdicts · NOT done and why · NEXT_IF_RESUMED. Commit via the serializer; register owed items with `tools/extract_canonical_tasks_from_directive.py --directive <memo> --register-all --owner ddm_scg1`. End with `sj1 S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]` (unchanged by this arm).
