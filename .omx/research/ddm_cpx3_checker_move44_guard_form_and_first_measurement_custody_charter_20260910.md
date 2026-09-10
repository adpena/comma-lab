# ddm_cpx3 — compliance checker: accept move 44's CUDA-guard form under the typed CPU-refusal path, and read dispatch custody from a completed candidate_seal.v3 (charter, MAIN 2026-09-10)

swp3 (`.omx/research/ddm_swp3_pr140_swap_packet_restage_move44_20260910.md`, sha 87f1d2ee685d35f1…, landed) staged move 44's packet
(archive 04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e, 180,406 B; S 0.1372449041713402 recomputed) and
measured compliance **81/93**: (a) 7 refusals in the typed `contest_cpu_axis_refusal.v1` path landed by cpx2
(`.omx/research/ddm_cpx2_checker_cuda_only_refusal_adjudication_path_20260910.md`, sha a48f685cbd42666a…): move 44's `inflate.py` CUDA
guard text differs from move 43's (the rider cure rewrote the receiver; the guard is still reached through `inflate.sh` and the
retained move-44 CPU refusal receipt exists: `/Volumes/VertigoDataTier/pact/ddm_rlc5_move44_cpu_20260910/MODAL_REMOTE_RESULT.json`,
record `.omx/research/ddm_rlc5_packet_inputs_20260910/CPU_AXIS_ADJUDICATION.json` if swp3 wrote it, else write it in the cpx2 shape);
(b) 3 dispatch-custody refusals: the checker reads dispatch custody from a normal seal-fire receipt shape, while move 44's row was
produced by the first-measurement lifecycle (intent v5 / authorization v8 / completed `candidate_seal.v3`
`.omx/research/ddm_rlc5_20260910/SEAL_ddm_rlc2_counted_cure_move43_rlc5_contest_cuda_v3.json`, run5 call fc-01M26NNV3WR2XXXV914S8BTDR4,
ledger rows dispatched/harvested/reconciled_terminal_success); (c) import hygiene (receiver change → operator) and hosting (at publish) are NOT yours.

## Deliverable (checker-side only; code + tests in ONE serializer commit)
1. `scripts/pre_submission_compliance_check.py` (+ its helper module): the typed refusal path derives the receiver declaration
   from the STAGED tree's actual guard (parse `inflate.sh` → the Python guard it reaches; accept any guard that raises on the
   CPU path and declares `linux-nvidia-t4`), never from a hard-coded move-43 string; bind the refusal receipt to the packet's
   archive sha/size + runtime digest recomputed from the staged tree, as cpx2 did. A receipt bound to a different archive still refuses.
2. Dispatch custody: accept a completed `candidate_seal.v3` as the custody object — verify `first_measurement_receipt` bytes/sha,
   the harvested ledger row for its call id, the `reconciled_terminal_success` event, and the seal's runtime custody objects — as an
   alternative to the normal seal-fire receipt shape; refuse if any binding differs. Never invent CPU metrics; never loosen the
   normal path.
3. Tests: move-44 staged packet + real receipt PASS on the seven CPU checks and three custody checks (fixture from the real values,
   read-only); wrong sha / different guard that does NOT raise on CPU / receipt bound to move 43 → FAIL; normal path unchanged.
4. Re-run the strict check on `submissions/_staging_move44_pr140_swap/` with swp3's recorded argv (its COMPLIANCE_COMMAND.json);
   retain the output; report the count (expect 91/93: import hygiene + hosting remain). Refresh the staged README from move 44's
   facts ONLY if swp3's memo lists it as a text-only task (no receiver bytes); otherwise leave it and say so.
5. Memo `.omx/research/ddm_cpx3_checker_move44_guard_form_and_first_measurement_custody_20260910.md` with check → code → test rows,
   before/after counts. Two review passes per .py; ruff; serializer commit LAST, once; rc 17 is NOT a stop. Checkpoint as `ddm_cpx3`
   and mark it COMPLETE at the end.

## Boundaries
Do NOT edit the staged tree's receiver files, the live PR tree, `upstream/`, any `/Volumes/...` path, `src/tac/candidate_seal.py`
or any pinned pre-fire consumer; no publish, no Modal, no scorer runs. Every binding recomputed from disk.

## OPTIMAL FORM
- Reference form: cpx2's typed refusal path and the checker's existing dispatch-custody inspection (same helper, same strictness);
  scoped extensions, no new checker; run on the real staged packet.
- Provenance pins (sha256 prefixes): swp3 memo 87f1d2ee685d35f1…; cpx2 memo a48f685cbd42666a…; completed seal v3 (record sha); run5 receipt (record sha);
  move-44 CPU refusal receipt (record sha); pointer move 44 commit 99625f32f / archive 04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e.

## Prior negatives accounted (operator 2026-08-15)
- cpx1: a refusal bound to another pointer is not evidence — bind to move 44's receipt only.
- swp3 dead-end: "zero TC3 text matches is false; historical strings remain, no retracted identifiers/imports" — keep that distinction.
- r9m: runtime digests content-only; name the definition you bind.
- Catalog #249: a CUDA receipt is never CPU evidence.

Final message: check table, before/after counts, serializer rc, boundaries, and the frontier line
`composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)`.

<!-- # FORMALIZATION_PENDING: checker charter; no measured row -->
