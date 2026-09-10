# ddm_cpx2 — move-43 CPU refusal adjudicated; strict packet remains 91/93

The checker cure is implemented and exercised on the real staged move-43 packet. **MEASURED [macOS-CPU checker diagnostic]: 85/93 → 91/93 PASS; strict rc 1 remains.** Exactly the six chartered failures changed to PASS. Import hygiene and the missing hosted manifest remain FAILED. The exact frontier did not move; no scorer or Modal call was launched.

## Check → code → test

| Check | Before | After | Implementation and verification |
|---|---:|---:|---|
| `contest_cpu_auth_eval_score_parseable` | FAIL | PASS | `inspect_contest_cpu_auth_eval` typed refusal branch; real move-43 positive, no CPU record or strict formula |
| `contest_cpu_auth_eval_archive_sha_matches` | FAIL | PASS | `required_contest_cpu_axis_refusal_blockers`; current archive SHA and retained archive identities; wrong SHA rejected |
| `contest_cpu_auth_eval_archive_size_matches` | FAIL | PASS | Same helper; current archive bytes 180,466; wrong size and Boolean size rejected |
| `contest_cpu_auth_eval_schema_metric_consistency` | FAIL | PASS | Closed refusal field set; no score/pose/seg/rate fields; all metric aliases, null metric fields and nested metric fields rejected |
| `contest_cpu_auth_eval_runtime_tree_recorded` | FAIL | PASS | Recomputed environment-free digest over live packet, CPU provenance, CUDA provenance; changed digest/tree rejected |
| `auth_eval_raw_promotion_policy_blockers_absent` | FAIL | PASS | Packet-produced `submission_policy_adjudication.v1`; exact serialized round-trip, unknown blockers and failed prerequisite tests |
| Remaining passing checks | 85/85 | 85/85 | Normal checker paths retained; CPU/CUDA runtime comparison now has affirmative same-packet evidence |
| Receiver import allowlist | FAIL | FAIL | Untouched receiver boundary |
| Hosted archive manifest | FAIL | FAIL | Untouched publish boundary |
| Total | **85/93** | **91/93** | Both real runs rc 1; full argv and reports retained |

Code: `src/tac/auth_eval_schema.py` (canonical schema, evidence checks, environment-free projection, policy review construction and matching); `scripts/pre_submission_compliance_check.py` (existing CPU inspector plus final packet review integration). Tests: `src/tac/tests/test_contest_cpu_axis_refusal.py`.

**Validation: 160 passed in 14.24 s, no skips**: 65 new real-packet integration cases plus 95 existing regression cases across `test_pre_submission_compliance_check.py`, `test_auth_eval_schema.py`, `test_adjudicate_contest_auth_eval_policy.py`, and `tests/test_pre_submission_compliance_frontier_score.py`. Command: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_contest_cpu_axis_refusal.py src/tac/tests/test_pre_submission_compliance_check.py src/tac/tests/test_auth_eval_schema.py src/tac/tests/test_adjudicate_contest_auth_eval_policy.py tests/test_pre_submission_compliance_frontier_score.py`. Ruff passed for all three changed Python files. `pytest_final.txt`, `ruff.txt`, and `REVIEW_PASSES.json` retain the results.

The integration fixture uses the actual retained packet and receipt values. It explicitly skips when that packet or SSD receipt is unavailable on another host; that is not a portable-positive claim. Existing normal CPU path regressions are portable and remain green. Negative tests mutate actual evidence, repin references, and check the binding/error semantics rather than merely tripping an unchanged file hash. Review found and fixed full/equal/abbreviated command shadows (the evaluator accepts argparse abbreviations), Boolean return-code aliases, metric contamination, unknown-version fallthrough, shared policy list mutation, and Boolean/numeric policy comparison. Two clean review passes per Python file were recorded with `tools/review_tracker.py`; the independent reviewer made no edits. No review override was used.

## Real custody and the three digest definitions

Archive: `7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e`, **180,466 B**, independently rehashed from `submissions/_staging_move43_pr140_swap/archive.zip`. Charter pointer commit: `48109233e`.

New refusal: `CPU_AXIS_REFUSAL.json`, schema `contest_cpu_axis_refusal.v1`, selected axis `contest_cuda`, `cpu_metrics_absent_by_design=true`. The historical check names remain, but each records `cpu_axis_refusal_adjudicated` as its explicit basis. The CPU section has `record=null`, `strict_formula=null`; no CPU score is inferred from CUDA. Selecting `contest_cpu` refuses this schema. Malformed versions, receipt references, guards, commands, metrics, or bindings fail closed.

Retained Modal call: `fc-01M26CHH4H0FHJ0EANBWP4WMZN`. Receipt `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6_cpu_20260910/MODAL_REMOTE_RESULT.json`: **46,900 B**, SHA `48071edcc7ee0a7b6a1ec93d7db9661753af0bbef93d136cd7351797e7c87c91`. Adjudication `.omx/research/ddm_sj1_pass6_packet_inputs_20260910/CPU_AXIS_ADJUDICATION.json`: **955 B**, SHA `d9c60343cc5a6c528d9b83806a562f2d5b2f0d296638cd610a23e325adaa09bc`. Other charter pins were reread and retained in `INPUT_BINDINGS.json`, including cpx1 memo `5a5d83e3932cbe6a…` and swp2 argv `890f7b0ebb728c4c…`.

The outer receipt's `expected_runtime_tree_sha256` is **empty**. It is not used as runtime proof. The retained `artifacts.provenance.json` nevertheless contains the completed 45-file manifest; the recorded CPU command also pins `--expected-runtime-files-sha256`. The adjudication binds that retained receipt by path/bytes/SHA. The checker compares its own live-file recomputation with the inner provenance and command pin. This supplies the runtime binding through the adjudication's referenced evidence, without inventing a missing outer field.

| Definition | Retained CPU | Retained CUDA | Live staged packet |
|---|---|---|---|
| `runtime_tree_sha256`, location/import dependent | `7968630f…` | `a726739a…` | `750fc1e4…` |
| `runtime_content_tree_sha256`, portable projection with import/dependency structure | `5863425f…` | `5863425f…` | `5863425f…` |
| **Chosen `contest_auth_eval.runtime_files_sha256.v1`**, environment-free files + evaluator | **`ffdbde48…`** | **`ffdbde48…`** | **`ffdbde48…`** |

The chosen digest is SHA-256 of sorted compact JSON containing only sorted `(relative_path, bytes, sha256)` file rows and the `upstream_evaluate_py` identity. Full digest: `ffdbde488ac17d3708b09d9b462aa62ef3b63d6d57e4198347995912df15e5ce`. All three manifests contain 45 runtime rows. Full digests and definitions are in `RUNTIME_BINDINGS.json`.

The source location is corrected: `inflate.sh:83` invokes `inflate.py`; `inflate.py:56` tests CUDA availability and line 57 raises the retained exception. The validator reads the launcher line, inspects the guard AST and literal exception, and compares the retained traceback. It does not claim the CUDA guard itself is in the shell script.

Trust boundary: this is consistency and custody validation against the operator-retained records and actual disk packet, not cryptographic authentication of Modal. The provider JSON has no signed call identity; the call is bound through the supplied adjudication record. Hashes cannot prove provider authenticity against replacement of every trusted record. The implementation rejects unbound/mismatched fabricated declaration files; it makes no stronger claim about a wholly rewritten trust base.

## Genuine policy review, with release debt preserved

`build_report` captures the raw CUDA and CPU refusal file references before inspection and rereads those references before policy construction, refusing changed bytes. After the full packet checks, it constructs `submission_policy_adjudication.v1` from that invocation's own checks, round-trips its serialization, and consumes that exact object for the raw-policy check. There is no CLI accepting user-supplied passed-check flags.

Only the raw evaluator's five named requests for a policy review are resolved. Unknown raw blockers or other packet failures refuse construction. The receipt binds archive SHA/size, the chosen runtime digest, the actual CUDA receipt and typed CPU refusal receipt; it lists the 90 passed non-self checks and both unresolved release checks. It excludes the raw-policy check from its own premises, avoiding circular self-approval. The two unresolved checks are explicitly permitted as **recorded review debt**, not waived release gates.

`review_complete=true`, **`release_ready=false`**, `cpu_leaderboard_reproduction_eligible=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`. The raw CUDA receipt and its flags/blockers were not modified. Both unresolved checks remain FAILED in the final strict report, so overall rc stays 1. `SUBMISSION_POLICY_ADJUDICATION.json` is an exact extraction of the checker-produced object retained inside `COMPLIANCE.json`; it grants no separate promotion authority.

## RECALL EVIDENCE

Own content recall covered all research Markdown, research JSON/JSONL arm receipts, docs/design/spec surfaces, canonical task-status and operator P0 ledgers, canonical research index and all `sub015_DAG_*` files. Queries included `cpu.{0,30}refus|submission_policy_adjudication|raw_auth_eval_does_not_verify_submission_policy_gates|runtime.{0,20}digest`, `cpu.{0,30}refus|policy.adjudicat|runtime.{0,20}digest`, and `REFUSED_BY_DESIGN|contest_cpu_axis_refusal|submission_policy_adjudication`. `tools/list_canonical_equations.py --json` returned **483 rows**, with zero matches to the recorded refusal/policy/digest query. Exact argv and retained search outputs are in `RECALL_SEARCHES.json` and `RECALL_SUPPLEMENT.json`. Graph scope is the named index/DAG, not an unqueried graph service. Lane registry was read for cpx conflicts; no cpx1/cpx2 lane entries were found. This checker-only arm owns no scorer slot or dispatch lane.

Beyond charter seeds, `ddm_g8c_gen8_compress_vendoring_20260902.md` distinguishes a lower-level local CPU decode from the refusing public entry point; it prevents treating internal CPU identity as public CPU evaluation. `canonical_submission_pipeline_specification_memo_20260526.md` and the research index reinforce separate authority axes. Source recall of `experiments/contest_auth_eval.py::_evaluation_evidence_flags` shows that raw blockers are unconditional requests for downstream policy review; `scripts/adjudicate_contest_auth_eval.py::_check_raw_promotion_policy_gate` confirms that component/hardware checks alone cannot launder them. Existing regression tests enforce that boundary. What changed: policy production occurs only after current packet checks and keeps unresolved release debt, rather than editing raw receipt flags or marking the packet approved. The live hot state supersedes stale frontier numbers in the common contract.

## Boundaries and landing

No staged/live PR tree, receiver, volume path, `upstream/`, or common-contract forbidden file was edited. No publish, remote message, Modal dispatch, scorer, GPU, training, GT decode, new score, or pointer mutation occurred. No payload was generated or discarded. Test copies are temporary test scratch; original packet bytes remain retained. No cleanup/move/delete operation was performed on retained evidence. Operator-facing evidence lives under `.omx/research/ddm_cpx2_20260910/`; provider scratch strings in existing provenance are historical content, not new consumer evidence paths.

`BOUNDARY_VERIFICATION.json` verifies **53 protected input files unchanged**, including **49 staged packet files**, and an unchanged shared Git index before serialization. Raw CUDA JSON has no diff from HEAD. Unrelated dirty work was preserved. No `git add`, stash or direct shared-index mutation was used by this arm; the serializer uses an intent patch and its private index.

All six solver hooks are N/A: no sensitivity, Pareto constraint, bit allocation, deployment actuator, empirical score posterior or solver ambiguity changed. The executable consumer is the existing compliance checker; no parallel checker or new catalog gate is introduced. The canonical schema invariants and corruption tests provide self-protection. Checkpoint identity is `ddm_cpx2`; fire orders are retained in `FIRE_ORDERS.json`.

Serializer is invoked **once, last**, with every post-edit content SHA, `--no-co-author`, `[no-triality] [p0-ledger-ok]`, and a local fallback directory honoring the volume-write prohibition. Its actual rc, commit/bundle custody and verification are in the post-attempt `SERIALIZER_STATUS.json`. An rc-17 fallback is handed to MAIN and is not called a main-branch landing.

## NEXT_IF_RESUMED

- QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer `.omx/research/ddm_cpx2_20260910/SERIALIZER_STATUS.json`; fire on rc 17 and verified bundle at harvest: land the exact checker/test/evidence commit.
- FOLDED; owner operator + MAIN; consumer `.omx/research/ddm_swp2_20260910/BLOCKERS.json`; fire when the operator selects a receiver correction: resolve the two fallback imports and obtain fresh exact evidence for any changed receiver.
- FOLDED; owner operator + MAIN; consumer `.omx/research/ddm_swp2_20260910/BLOCKERS.json`; fire after non-hosting clearance and explicit publish confirmation: produce the real hosted manifest and obtain strict PASS before release. Existing README/MANIFEST repair debt stays with that packet owner.

composition S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (move 43) unchanged.

## LIVE-HYPOTHESES

- Removing the two fallback imports may preserve normal CUDA execution because swp2's bare-venv smoke did not reach them. This is an untested receiver change, owned by MAIN; a fresh exact row is required before claiming unchanged score behavior.

## DEAD-ENDS

- INSTANCE: reusing move-42 refusal bytes for move 43 remains invalid; the new path consumes the newly retained move-43 receipt instead.
- INSTANCE: treating the empty outer runtime digest as proof is invalid; inner retained provenance and the live staged tree supply the binding.
- FORMULATION: inventing CPU metrics or selecting `contest_cpu` with a CUDA-only refusal cannot satisfy the typed path.
- FORMULATION: changing raw receipt flags or supplying a JSON of passing policy flags does not replace current packet checks.
- INSTANCE: 91/93 is not release clearance; import hygiene and hosted-manifest failures remain real blockers.

<!-- # FORMALIZATION_PENDING: scoped checker extension; no new scoring measurement or equation -->
