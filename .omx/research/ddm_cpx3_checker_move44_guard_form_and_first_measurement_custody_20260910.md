# ddm_cpx3 — move-44 checker custody complete; strict packet 91/93

**MEASURED [macOS-CPU checker diagnostic]: 81/93 → 91/93 PASS; strict rc 1 remains.** The seven CPU-refusal/policy checks and three dispatch-custody checks now pass on the actual staged move-44 packet. Import hygiene and hosting remain failed. No new exact evaluation or pointer move occurred.

The final strict invocation uses swp3's **exact recorded argv and environment**, including its staged `_packet/COMPLIANCE.json` output path. The report is also retained under `.omx/research/ddm_cpx3_20260910/COMPLIANCE.json`. Baseline output was redirected to this arm's `COMPLIANCE_BEFORE.json` to preserve it. `COMPLIANCE_COMMAND.json`, `COMPLIANCE_STATUS.json`, and stdout/stderr preserve invocation and outcome.

## Check → code → test

Paths below are repo-relative. `schema` means `src/tac/auth_eval_schema.py`; `checker` means `scripts/pre_submission_compliance_check.py`; `custody` means `scripts/pre_submission_first_measurement_custody.py`. New real-packet tests live in `src/tac/tests/test_move44_compliance_custody.py`.

| Check | Before | After | Code → executed test |
|---|---|---|---|
| `contest_cpu_auth_eval_score_parseable` | FAIL | PASS | schema `staged_cpu_refusal_receiver` → `test_real_move44_ten_checks_and_release_debt`; record and strict formula remain null |
| `contest_cpu_auth_eval_archive_sha_matches` | FAIL | PASS | schema `required_contest_cpu_axis_refusal_blockers` → wrong SHA and move-43-receipt controls |
| `contest_cpu_auth_eval_archive_size_matches` | FAIL | PASS | same helper → wrong-size control against real archive bytes |
| `contest_cpu_auth_eval_schema_metric_consistency` | FAIL | PASS | unchanged closed refusal fields → existing cpx2 metric-alias/nested-metric corruption tests |
| `contest_cpu_auth_eval_runtime_tree_recorded` | FAIL | PASS | schema live manifest and retained CPU provenance → changed guard/runtime controls |
| `contest_cpu_auth_eval_runtime_tree_matches_cuda` | FAIL | PASS | checker same-files CPU/CUDA comparison → real packet positive; CPU remains refusal-only |
| `auth_eval_raw_promotion_policy_blockers_absent` | FAIL | PASS | existing packet-produced policy adjudication → real packet positive, release-ready remains false |
| `dispatch_claim_successful_exact_eval_terminal_row` | FAIL | PASS | checker → custody completed-v3 alternative; actual call's dispatched/harvested/reconciled lifecycle; missing, duplicate, different-call and result-identity controls |
| `dispatch_claim_terminal_archive_sha_bound` | FAIL | PASS | custody hashes archive, first receipt, seal/intent/auth refs, harvested row and reconciliation → repinned archive/ref/size/score corruptions |
| `dispatch_claim_terminal_runtime_tree_sha_bound` | FAIL | PASS | custody reuses frozen runtime validator, checks both embedded custody objects and staged files → source-ref, leg, runtime digest and stale-manifest/live-file corruptions |
| `submission_runtime_imports_within_allowlist` | FAIL | FAIL | receiver boundary; both `experiments` fallbacks remain |
| `hosted_archive_manifest_supplied` | FAIL | FAIL | publish boundary; no hosted evidence manufactured |
| Other checks | 81/81 PASS | 81/81 PASS | existing normal-path regressions; the normal harvested path does not consult v3 custody |
| Total | **81/93** | **91/93** | **strict rc 1 → 1** |

**Validation: 222 passed in 16.31 s, no skips.** This comprises 62 new move-44 cases, 65 cpx2 cases, and 95 existing compliance/schema/policy/frontier cases. Full command:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_move44_compliance_custody.py src/tac/tests/test_contest_cpu_axis_refusal.py src/tac/tests/test_pre_submission_compliance_check.py src/tac/tests/test_auth_eval_schema.py src/tac/tests/test_adjudicate_contest_auth_eval_policy.py tests/test_pre_submission_compliance_frontier_score.py
```

`pytest_final.txt`, `ruff.txt`, and `REVIEW_PASSES.json` retain validation. Two clean review passes per changed Python file were marked through `tools/review_tracker.py`; no override. Earlier review defects were fixed before those final clean passes: shell-line drift, runtime dictionary path comparison, normal/retained runtime projection distinction, caught frozen-validator refusals, explicit result call/authorization identity, and harvested-score consistency.

The historical cpx2 policy fixture previously depended on the live frontier. The unmodified checker now gives that move-43 packet 89/93 because move 44 makes it a frontier regression and policy construction consequently refuses. `MOVE43_BASELINE.json` preserves that diagnosis. Only this historical test fixture now selects actual retained move-43 anchors. Production frontier checks remain unchanged and still reject the old packet today; the separate frontier regression suite passes.

## Real receiver declaration and limits

Move 44's staged shell invokes Python at **line 86**, not the refusal metadata's old line 83. Its Python guard is at line 57 and raises at line 58. `DERIVED_RECEIVER.json` records the actual declaration. The checker finds the unique shell invocation in the staged file, parses its Python target, interprets the direct `main` guard, verifies a literal T4-declaring `RuntimeError`, and matches the retained CPU traceback at the actual raise line. The launcher-line hint is descriptive metadata; file hashes and the reached traceback supply custody. No receiver file or refusal receipt was edited.

The interpreter handles declarative constants, local bindings, boolean conditions, equality/inequality, CUDA availability, and environment opt-in reads under ordinary unset opt-ins. It does not execute receiver code. Unknown expression syntax refuses; this is not a general Python program verifier. The real CPU receipt is required independently, so static guard recognition alone supplies no CPU evidence. Changed source bytes, a false/inverted/advisory-only guard, wrong archive SHA/size, or a move-43 receipt fail. The equivalent-expression test exercises parser semantics only and does not claim that altered receiver bytes inherit the original receipt.

## Completed-v3 custody and digest definitions

Resolution follows the latest matching claim's unique call into the canonical `reconciled_terminal_success` ledger event, rehashes its reconciliation, then searches **only that reconciliation's own directory** for the unique matching typed seal-link addendum. The addendum's pass-like booleans confer no authority. The completed seal reference, all referenced JSON bytes, canonical seal/intent/authorization digests, source runtime, archive, receiver pins, harvested nonce, receipt identities, packet's exact evaluator JSON, ledger event order, harvested claim text/hash, and latest claim identity are verified.

The unchanged frozen `validate_first_measurement_runtime_custody` recomputes custody on the original sealed runtime, including context/request/result/argv/provenance and both root projections. The checker compares that object with the seal and the retained timing leg, then hashes the actual staged file rows again using the checker's existing custody-file exclusions. This matters: the staged `MANIFEST.sha256` is an existing checker custody exclusion; feeding the entire staged tree directly to the pre-fire runtime validator is a different manifest definition and correctly refuses. No digest algorithm, frozen consumer, or normal path was loosened.

| Definition | Recomputed/bound value |
|---|---|
| Seal `tac.candidate_seal.measure_runtime_digest` | `f4c6c250cb1d806d6cafbc4d6cfed82f8f59d95fe432db1925637571b624c669` |
| Modal content projection | `e1e6d1252b56b66b7f7559fe0df751ffbf38559d99abf21f06f00134f685bcaf` |
| Modal normal path-tree projection, used in harvested claim | `e3d2371917920ce3dad002761203af48cfa72d98b07f82f077ef6dc0115b1891` |
| Retained worker path-tree projection, used by packet | `e44a61a1b2b2e5dab42488cdb1c33cef04e6a516b5be4ad361576c9df93506ee` |
| Environment-free files + evaluator, 48 runtime rows, CPU/CUDA/staged equality | `fb1f6295a3527fdc07b3682d7e8ed42354e14e54dbbb4317deaa9f4f27ba21d4` |

Run5 call: `fc-01M26NNV3WR2XXXV914S8BTDR4`. The seal is **16,791 B**, SHA `8fd8077f7f495325d3a56b77a54610eff481259f7e6efa62f27beadbae95bcd5`. Run5 receipt is **242,556 B**, SHA `b3ec1dd9dbe43debeeeeae075a49c1cd8b2d23507a7fc317e9884590b9c8fe46`. CPU refusal is **48,262 B**, SHA `ef39b9b7aa173f1acf3730d57b1e2dc1c661bfef381c693923c6c7480c612124`. `INPUT_BINDINGS.json` pins these and verifies swp3 `87f1d2ee685d35f1…` and cpx2 `a48f685cbd42666a…`. Pointer charter commit: `99625f32f`; archive SHA `04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e`, measured size **180,406 B**.

This is historical completion custody, not a new authorization to fire an old intent. It validates consistency against operator-retained files, not cryptographic provider authenticity against replacement of the entire trust base. No CPU metrics are invented. The policy record remains `release_ready=false`; import and hosting failures are not waived.

## RECALL EVIDENCE

Original content recall covered `.omx/research/` Markdown and JSON arm receipts, `docs/` design/SPEC/catalog surfaces, canonical task-status and operator-P0 ledgers, all named `CANONICAL_RESEARCH_INDEX*` files and `sub015_DAG_*` files. Queries were `cpu.{0,25}refus|first.measurement.{0,25}custody|dispatch.claim.{0,25}terminal`, plus `cpx3` on task ledgers. `RECALL_QUERIES.json` retains exact argv and corresponding `*_recall.txt` outputs. `tools/list_canonical_equations.py --json` returned **483 rows**; zero matches to the same content query. The lane registry was checked; no cpx3 dispatch lane was found in that scope. This checker arm owns no scorer or GPU slot.

Beyond the charter seeds, pr16's amendment requires identical custody objects in seal and timing leg and freshly read pinned sources. cust1's `RUN5_COMPLETED_SEAL_ADDENDUM.json` provides the exact seal link without broad seal hunting; its memo's ledger-event blocker is historical, because the live canonical ledger now contains the required success event. These findings changed the implementation to reuse the frozen runtime validator and resolve custody through the bound reconciliation/addendum. The design catalog's terminal-status discipline supports retaining newest-claim semantics instead of simply selecting an older passing row. Graph recall reinforces separate authority axes; it adds no alternate completion evidence. The memory registry lookup found no cpx3/checker-move44/common-contract entry in the searched scope. Current hot state and this charter supersede stale frontier prose in the common contract.

## Boundaries and landing

`BOUNDARY_VERIFICATION.json` verifies **127/128 protected input files unchanged**. The sole change is the charter-authorized final strict output `_packet/COMPLIANCE.json`; every staged shippable file remains unchanged. Staged README is left untouched: swp3 assigns its stale prose to operator/MAIN alongside the receiver decision, rather than listing a standalone text-only task for this arm. The existing move-44 adjudication was consumed, not rewritten.

No live PR tree, receiver, `/Volumes/...` file, `upstream/`, `src/tac/candidate_seal.py`, pinned pre-fire consumer, or common-contract forbidden file was edited. No publish, external message, scorer run, Modal call, training, GT decode, payload generation/discard, or retained-artifact cleanup occurred. Test copies are trivial temporary test scratch; original payloads remain retained. Operator-facing evidence is durable under this arm's research directory; scratch path strings inside existing receipts are historical provenance only. Shared Git index unchanged; unrelated dirty work preserved. No stash or direct shared-index manipulation.

`research_only=true` for scoring claims. All six solver hooks are N/A: no sensitivity, Pareto constraint, bit allocation, dispatch actuator, empirical score posterior, or solver ambiguity changed. The executable consumer is the existing release checker; invariants and corruption tests protect the change, with no new catalog gate.

Serializer is invoked **LAST, ONCE**, with an exact intent patch, post-edit SHA per selected file, `--no-co-author`, `[no-triality] [p0-ledger-ok]`, and a repo-local fallback directory. The actual return code and verified commit/bundle custody are recorded after the attempt in `SERIALIZER_STATUS.json`. rc17 requires MAIN to land the verified commit and is not main-branch success. Checkpoint identity `ddm_cpx3` is marked COMPLETE only after the checker, tests, strict report, memo and landing custody are complete. Completion means this checker charter, not release clearance or sub-0.12 achievement.

## NEXT_IF_RESUMED

- QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer `.omx/research/ddm_cpx3_20260910/SERIALIZER_STATUS.json`; fire on verified rc17 bundle at harvest: land the exact checker/test/evidence commit and verify its file hashes.
- FOLDED; owner operator + MAIN; consumer `submissions/_staging_move44_pr140_swap/_packet/BLOCKERS.json`; fire on checker harvest and operator receiver decision: resolve import hygiene and stale README, obtaining fresh exact evidence for any receiver edits.
- FOLDED; owner operator + MAIN; consumer `submissions/_staging_move44_pr140_swap/SWAP_COMMANDS.md`; fire after non-hosting clearance and explicit publish authorization: host/fetchback, obtain a real hosted manifest, rerun strict checks, then swap only with release clearance. Restage if the pointer changes.

composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)

Retained-component arithmetic rederived `0.13724490417134017`; the quoted pointer value above is its established representation. No new score measured.

## LIVE-HYPOTHESES

- Removing the two modules' `experiments` fallback imports may preserve normal receiver behavior because each already tries packaged relative imports first. This is untested on a changed receiver, owned by operator/MAIN; fresh exact evidence is required before claiming unchanged score.

## DEAD-ENDS

- INSTANCE: a move-43 receipt cannot validate move 44; archive and runtime bindings differ and corruption controls refuse it.
- FORMULATION: exact matching to the old CUDA guard text and shell line number cannot describe rewritten-but-valid receivers. Actual source parsing plus retained CPU failure is required.
- INSTANCE: feeding the staged custody MANIFEST into the pre-fire runtime digest changes the manifest definition; use the original sealed runtime for its custody proof and the existing staged-file projection for packet equality.
- FORMULATION: accepting an older harvested claim without its completed seal/reconciliation loses latest-call custody. Missing, duplicate, or different-call events and mismatched refs refuse.
- INSTANCE: 91/93 is not release clearance. The two retained failures and stale README remain with their existing owners.

<!-- # FORMALIZATION_PENDING: checker-only custody extension; no new score or scientific law -->
