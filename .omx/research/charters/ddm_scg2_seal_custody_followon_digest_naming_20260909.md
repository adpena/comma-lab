# ddm_scg2 — seal-contract follow-on: fresh structured public-smoke receipts for the two historical sj1 seals (custody replays stay fireable) + one runtime-digest definition named in every receipt (charter, 2026-09-09)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (astra, medium) · Spawned by MAIN 2026-09-09 under the operator's standing GO. Source: scg1's registered follow-on `ddm_scg1_seal_contract_inflate_sh_smoke_receipts_20260908::ITEM_7` (memo `.omx/research/ddm_scg1_seal_contract_inflate_sh_smoke_receipts_20260908.md`, landed `89f9656f3`) + MAIN's hot-state note on the two digest definitions. Apparatus: `score_claim=false`; no scorer, no Modal.

## MANDATE

Operator 20260908: *"recover and respawn and continue with all"*. scg1 made seals REQUIRE a structured `public_entrypoint_smoke` block; the two historical sj1 seals (`SEAL_ddm_sj1_token_predistortion_joint_contest_cuda.json`, `…_pass3_…`) now validate as `SEAL_PUBLIC_SMOKE_MISSING` although both archives are already scored on T4. A custody replay (re-firing an already-scored archive for the CPU axis, or a re-verification) would need the waiver every time. ITEM_7: capture both receiver legs against candidate AND current-frontier trees for those two archives and issue SUCCESSOR seals carrying the structured block — the originals are never edited (append-only custody). Second: sj1 measured that `tac.candidate_seal.measure_runtime_digest` (pass-3 tree → `2435dab86725…`) and the Modal fire manifest's `expected_runtime_tree_sha256` (same tree → `6508d184e60a…`) are two DIFFERENT digest definitions — self-consistent per tool, but a reader comparing them reads drift. Cure at the class: every receipt that names a runtime digest names WHICH definition (a `digest_definition` field with the function's dotted path), and the two tools cross-reference each other's value where both exist.

## SCOPE

1. Recall: scg1 memo (whole), `src/tac/candidate_seal.py` (`_public_smoke_problems`, `build_seal`, `validate_seal`), `tools/make_candidate_seal.py` argparse, sj1's producer `experiments/ddm_sj1_joint_admission.py public-smoke` (commit 4a1894581 — reuse it; do not write a third producer), `tools/fire_modal_auth_eval.py` (where `expected_runtime_tree_sha256` is computed — find the function), the two sj1 seals and their trees under `/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/` (READ ONLY). `tools/subagent_checkpoint.py read --subagent-id ddm_scg2` first.
2. For each historical sj1 archive: run sj1's producer with candidate = that archive's tree and frontier = the LIVE pointer tree (read `.omx/state/canonical_frontier_pointer.json` at run time); write a SUCCESSOR seal file `SEAL_<candidate>_contest_cuda_successor_20260909.json` beside the original carrying the block, `supersedes: <original seal sha>`, and the T4 row it already earned (call id, score) as `already_scored`; `validate_seal` → SEAL_VALID; the fire tool's dry-run on the successor must pass WITHOUT the waiver. Originals untouched (assert sha unchanged before/after).
3. Digest naming: add `digest_definition` (dotted path) beside every `tree_sha256`/`runtime_tree_sha256` the seal and the fire manifest emit; where the fire tool has a seal, it records BOTH digests with their definitions; a test asserts a seal→fire pair names both and that the two values differ for the same tree (the fact, recorded, not hidden). Do NOT change either digest's computation.
4. Two genuine review passes per `.py`; tests for 2 and 3; memo `.omx/research/ddm_scg2_seal_custody_followon_digest_naming_20260909.md`; mark ITEM_7 complete in the canonical task ledger with the receipt path.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire (dry-run only). Never edit the historical seals, any candidate tree, `submissions/semantic_joint_ctxmix/`, or `experiments/ddm_sj1_*` / `ddm_pc2*` / `ddm_fe1*` / `ddm_rw1*` (four Opus arms are live; you import their producers, you do not edit them).
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).
- CPU ≤ 2 procs (the smoke legs take ~100 s each at 4 threads; run them sequentially). Detached only via `tools/launch_detached_process.py … --nice 10 --nice-best-effort`; no `nohup`/`&`/clock waiters; artifact-bound waits ≤ 780 s; kill the process group on timeout (pc2's orphan lesson).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 review passes (`review_tracker mark-file` rescans now). Tokens `[no-triality] [p0-ledger-ok]`. NEVER a Co-Authored-By or AI-attribution trailer. If git object writes are refused in the sandbox, emit the serializer fallback bundle + receipt and name it (scg1's path).
- ALWAYS KEEP THE PAYLOAD (successor seals + receipts on Vertigo beside the originals; small). VERIFIED-AT-SOURCE for both digest values and the function paths. EQUATIONS-LEG: apparatus; memo carries `# FORMALIZATION_PENDING:seal custody apparatus — no measured law`; run the Catalog #344 check before the final message.
- Checkpoint discipline: `tools/subagent_checkpoint.py --subagent-id ddm_scg2 …` every ~10 tool uses.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- Library-path decode identity is not public-entrypoint proof (rc1 incident; scg1 memo) — both legs, both roles, always.
- A "first two pairs" public smoke is not expressible (file_list selects videos) — the receipt is the PAIR (token-decode reach + CUDA-gate reach).
- One timeout value used as both subprocess bound and declared `public_path_probe_seconds` self-refuses on the timeout path (pc2, cured by probing at 0.8× the bound) — reuse sj1's producer, which already does this.
- A second implementation of the contract drifts (scg1/sj1) — validate with `_public_smoke_problems`, never a re-implementation.

## OPTIMAL FORM

- Family exemplar: scg1's landing (`89f9656f3`; `src/tac/candidate_seal.py` + `src/tac/tests/test_candidate_seal.py`) and sj1's producer (4a1894581) — the reference form: structured receipts re-derived from disk, fail-closed verdicts, tests that replay the incident.
- SCOPE reductions declared per row: only the two sj1 historical seals get successors (SCOPE; older seals are not fireable custody anyway). MECHANISM reductions FORBIDDEN: no waiver-based "fix", no editing of originals, no synthetic receipts.
- **PRIOR-LAW PREDICTION (falsifiable):** both successor seals validate SEAL_VALID and pass the fire tool's dry-run without a waiver; the two digest definitions differ for every tree checked (a naming fact, not drift). FALSIFIER: a successor seal fails validation on a receipt identity (the historical tree drifted on disk) — report the drift with shas; do not repair the tree.

## DELIVERABLE

The memo, the two successor seals, the digest-naming change + tests, ITEM_7 marked complete. Commit via the serializer. End with the live frontier line.
