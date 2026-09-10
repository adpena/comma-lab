# ddm_cpx2 — resume ddm_cpx1 with a CPU refusal receipt that BINDS the staged move-43 packet (charter, MAIN 2026-09-10)

cpx1 (`.omx/research/ddm_cpx1_checker_cuda_only_refusal_adjudication_path_20260910.md`, sha 5a5d83e3932cbe6a…, landed cbecea1ff) correctly
STOPPED: the retained CPU refusal bound move 42. MAIN fired the CPU axis on move 43's own bytes: Modal call
fc-01M26CHH4H0FHJ0EANBWP4WMZN, REFUSED BY DESIGN (rc 1 at 10.6 s; no contest_auth_eval.json produced), receipt
`/Volumes/VertigoDataTier/pact/ddm_sj1_pass6_cpu_20260910/MODAL_REMOTE_RESULT.json` with `expected_archive_sha256`
7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e and size 180,466; adjudication record
`.omx/research/ddm_sj1_pass6_packet_inputs_20260910/CPU_AXIS_ADJUDICATION.json` (sha d9c60343cc5a6c52…, committed). The cpx1 charter binds unchanged
(`.omx/research/ddm_cpx1_checker_cuda_only_refusal_adjudication_path_charter_20260910.md`): implement the typed
`contest_cpu_axis_refusal.v1` path in the checker, bound to the packet's REAL shas (archive sha/size; runtime digest —
note the refusal receipt's `expected_runtime_tree_sha256` is EMPTY because the container refused before computing it:
bind the runtime digest through the adjudication record + the checker's own recomputation over the staged tree, and say
so), the genuine `submission_policy_adjudication.v1` object from the packet's own passed checks, the tests, and the
re-run of `scripts/pre_submission_compliance_check.py --contest-final --strict` on `submissions/_staging_move43_pr140_swap/`
with swp2's recorded argv + the new receipt (expect the five CPU checks + the raw-policy check to PASS; import hygiene +
hosted manifest remain and are not yours).

Boundaries, OPTIMAL FORM (checker's existing CUDA inspection as the reference form; scoped extension; no scope reduction —
run on the real staged packet), and prior negatives: as cpx1's charter, plus cpx1's own finding (a refusal bound to a
different pointer is not evidence; every binding recomputed from disk). Serializer commit LAST, once; rc 17 is NOT a stop.
Two review passes per .py. Checkpoint as `ddm_cpx2`. Provenance pins: cpx1 memo sha 5a5d83e3932cbe6a…; adjudication record sha d9c60343cc5a6c52…;
pointer commit 48109233e / archive 7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e.

## OPTIMAL FORM
- Reference form: as cpx1's charter (the checker's existing CUDA-axis inspection; same helper, same strictness). No delta.
- Provenance pins: as above; swp2 COMPLIANCE_COMMAND.json (record its sha).

## Prior negatives accounted (operator 2026-08-15)
- cpx1: refusal bound to move 42 — cured by the move-43 dispatch; never relabel.
- pk1 / r9m: three runtime-digest definitions coexist — name the one you bind; content-only.
- Catalog #249 phantom-axis class: a CUDA receipt is never CPU evidence; the refusal receipt carries no metrics.

Final message: check table, before/after compliance counts, serializer rc, boundaries, and the frontier line
`composition S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (move 43)` unchanged.

<!-- # FORMALIZATION_PENDING: checker charter; no measured row -->
