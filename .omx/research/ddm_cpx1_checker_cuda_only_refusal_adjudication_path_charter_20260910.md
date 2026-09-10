# ddm_cpx1 — give the compliance checker a REAL CUDA-only refusal adjudication path (checker-side cure for 6 of swp2's 8 refusals; no receiver change, no publish) — charter, MAIN 2026-09-10

## The instance
swp2 staged the PR #140 swap on move 43's exact bytes (memo `.omx/research/ddm_swp2_pr140_swap_packet_restage_move43_20260910.md`,
sha 8e0a3ebe66895315…; blockers `.omx/research/ddm_swp2_20260910/BLOCKERS.json`, sha d1c6fa40cca0d6cd…): 85/93 PASS. Six of the eight refusals are
OUR checker's, not the packet's: `scripts/pre_submission_compliance_check.py::inspect_contest_cpu_auth_eval` (~line 1737)
parses the CPU-axis file as a normal exact eval, so a CPU-axis REFUSAL receipt (the receiver declares `linux-nvidia-t4`
and `inflate.sh` raises on the CPU path — Modal refusal call fc-01M25W34DGHHJES03531X3VDXV, record
`.omx/research/ddm_rp1_round2_packet_inputs_20260910/CPU_AXIS_ADJUDICATION.json`) fails five checks
(`contest_cpu_auth_eval_score_parseable`, `_archive_sha_matches`, `_archive_size_matches`, `_schema_metric_consistency`,
`_runtime_tree_recorded`) by MISSING metrics that must not exist; and `auth_eval_raw_promotion_policy_blockers_absent` fails
because no genuine policy adjudication object exists ("do not flip raw receipt flags"). The other two (static import hygiene
= receiver change; hosted manifest = at publish) are NOT yours.

## Deliverable (code + tests in ONE serializer commit; commit LAST)
1. A typed receipt schema `contest_cpu_axis_refusal.v1` (in the checker's canonical helper module, wherever
   `required_exact_eval_metric_blockers` lives — read the code first; do not fork a parallel parser): fields = archive
   sha256 + size, runtime-tree digest (name the definition), receiver declaration (`linux-nvidia-t4`, the exact `inflate.sh`
   guard line), the refusal Modal call id + retained receipt `{path,bytes,sha256}`, the selected axis `contest_cuda`, and
   `cpu_metrics_absent_by_design: true`. NO score/pose/seg/rate fields may appear; presence of any is a refusal.
2. `inspect_contest_cpu_auth_eval`: when `selected_axis == "contest_cuda"` AND the file validates as the refusal schema
   AND its archive sha/size and runtime digest equal the packet's, the five checks PASS as `cpu_axis_refusal_adjudicated`
   (each check records the basis); when the file is a refusal but ANY binding differs, or the axis is contest_cpu, they FAIL
   exactly as today. Never infer CPU metrics from CUDA. Keep every other check untouched.
3. `auth_eval_raw_promotion_policy_blockers_absent`: implement the "genuine policy adjudication" object the cure names —
   a checker-produced `submission_policy_adjudication.v1` written from the packet's OWN passed checks (bound to archive
   sha/size, runtime digest, the CUDA receipt, the CPU refusal receipt), consumed by that check; the raw receipt is never
   modified. If the check's contract cannot be met this way, STOP with the exact sentence.
4. Tests: refusal receipt PASSES on the staged move-43 packet's real bindings (fixture from the real values); refusal with a
   wrong sha / size / digest FAILS; refusal carrying any metric field FAILS; axis contest_cpu FAILS; the normal CPU eval path
   is unchanged (existing tests green); the adjudication object round-trips. `ruff` clean; two review passes per .py.
5. Re-run `scripts/pre_submission_compliance_check.py --contest-final --strict` on `submissions/_staging_move43_pr140_swap/`
   with the same argv swp2 recorded (`.omx/research/ddm_swp2_20260910/COMPLIANCE_COMMAND.json`) plus the new refusal
   receipt; retain the output; report the new count (expect 91/93 with import-hygiene + hosted-manifest remaining).
6. Memo `.omx/research/ddm_cpx1_checker_cuda_only_refusal_adjudication_path_20260910.md`: check → code → test table, the
   before/after counts, every boundary. Serializer commit LAST, once; rc 17 is NOT a stop (bundle; MAIN lands). Checkpoint
   as `ddm_cpx1`.

## Boundaries
Do NOT edit the staged tree, the live PR tree, `upstream/`, any `/Volumes/...` path, or any receiver file; no publish,
no Modal, no scorer runs. The refusal path must be IMPOSSIBLE to satisfy with a fabricated file: every binding is a real
sha the checker recomputes from the packet on disk.

## OPTIMAL FORM
- Reference form: the checker's existing CUDA-axis inspection (same helper, same `_add` pattern, same strictness) — a
  scoped extension, not a new checker. No scope reduction: run on the real staged packet.
- Provenance pins: swp2 memo sha 8e0a3ebe66895315…; BLOCKERS.json sha d1c6fa40cca0d6cd…; CPU adjudication record path above (record its sha);
  pointer move 43 commit 48109233e / archive 7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e.

## Prior negatives accounted (operator 2026-08-15)
- swp2 dead-end: "the existing CPU refusal JSON does not satisfy the current checker; five failures prove that" — this
  charter adds the typed path rather than relabeling the receipt.
- pk1: the checker previously read a third runtime-digest definition — name the definition you bind and record all three.
- r9m: two validators disagreeing ⇒ env-coupled digest — bind content-only digests both sides compute.
- Catalog #249 / phantom-axis class: a CUDA receipt must never be read as CPU evidence; the refusal receipt carries no metrics.

Final message: check table, before/after compliance counts, the serializer rc, every boundary, and the frontier line
`composition S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (move 43)` unchanged.

<!-- # FORMALIZATION_PENDING: checker charter; no measured row -->
