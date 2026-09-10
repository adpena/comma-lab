# ddm_ffi5 — corrected first-measurement consumer amendment[3], 2026-09-10

**The exact pointer is unchanged. All three corrected consumer changes are implemented, tested, and reviewed in the working tree; MAIN must land them.** Required suites: **146 passed**. Expanded adjacent suites: **181 passed, one sandbox-blocked socket test separately recorded**. Ruff clean; two corrected review passes per each of five Python files. The single serializer attempt returned **rc 17** (Git object writes denied), but its bundle preceded discovery of MAIN's `CHARTER_CORRECTION.md` and is **obsolete; do not land it**. Final sources and `FINAL_SOURCE.patch` are the handoff.

## Controlling correction

The original charter's first diagnosis was disproved by retained bytes: run3 already passed the normal upload projection, not the intent seal hash. Its retained extraction root changed the path-dependent tree digest. MAIN then wrote `ddm_ffi5_20260910/CHARTER_CORRECTION.md` at approximately 20:50Z, replacing fix 1 with **content-only validation and preserved tree custody**. I discovered that file after the one serializer attempt and completed the replacement without a second attempt. Fixes 2 and 3 stand. The correction is retained and hashed in the final intended-file manifest.

## Finding → code → test

| Finding | Corrected implementation | Executed proof |
|---|---|---|
| Root-dependent tree hashes disagree for identical content | `tools/fire_modal_auth_eval.py::_first_measurement_main` computes existing upload-manifest digests and emits the content pin. Local `experiments/modal_auth_eval.py::main` compares both argv and context pins against actual local content before spawn. The remote wrapper and fail-closed wrapper carry it to the evaluator argv. `experiments/contest_auth_eval.py::_validate_expected_runtime_tree` checks the supplied content digest; the real path-tree digest remains in provenance. Normal calls retain their tree comparison; supplying both expectations validates both. Reuse also checks the content expectation. | `test_first_measurement_argv_and_manifest_name_both_real_digests`; `test_first_measurement_still_refuses_changed_intent_runtime_digest`; `test_content_pin_accepts_relocation_but_refuses_one_changed_byte`; `test_local_content_pin_validation_and_worker_argv` (three cases); `test_content_pin_survives_fail_closed_wrapper`; existing worker retention test now asserts forwarding. Real retained rlc5 provenance passes the content check; normal tree check reproduces run3's refusal. |
| Ledger registration uses imported snapshot defaults | `experiments/modal_auth_eval.py::main` passes the real repo cwd's ledger path **and matching `.jsonl.lock`** to the existing fail-closed registration helper. The dispatcher already uses `cwd=REPO`; it was the imported ledger module's `__file__` default that selected the snapshot. | `test_local_registration_uses_repo_ledger_when_imported_from_snapshot` (normal and first-measurement): execute the actual production call expression and real append, preserve existing row, assert repo ledger/lock and absent snapshot ledger/lock, then satisfy the real SPAWNED registration check. |
| Authorize import graph fails outside repo cwd | `tools/authorize_candidate_first_measurement.py` adds the literal `sys.path.insert(1, str(REPO))` after `src`. | `test_authorize_tool_imports_experiments_from_a_foreign_cwd`: bare subprocess without PYTHONPATH imports the real consumer and transitive evaluator dependency. No authorization invoked. |

## rlc5 digest measurements

MEASURED here [macOS-CPU advisory; scorer-free byte/provenance validation], runtime `/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/candidate_runtime`:

| Existing definition / destination | SHA-256 |
|---|---|
| `tac.candidate_seal.measure_runtime_digest` — intent identity, still independently validated | `f4c6c250cb1d806d6cafbc4d6cfed82f8f59d95fe432db1925637571b624c669` |
| `tac.decode_wall_clock.measure_t4_runtime_digest` — normal upload path projection | `e3d2371917920ce3dad002761203af48cfa72d98b07f82f077ef6dc0115b1891` |
| Actual retained worker path-tree digest | `baeb53afc8d1bb0c43bb2ce91fe1e09e845809b76fc3047bc8ed3397254155af` |
| Existing `modal_uploaded_submission_dir_runtime_manifest.runtime_content_tree_sha256` — **local and worker equal** | `e1e6d1252b56b66b7f7559fe0df751ffbf38559d99abf21f06f00134f685bcaf` |

Both sides have **48 runtime files**. Projecting the real local manifest to the recorded retained physical volume root exactly reproduces the worker tree hash and file rows. The corrected evaluator accepts that retained manifest against the independently computed local content hash. A regression changes one real source byte and recomputes the manifest: content validation refuses. Missing manifest also refuses. No digest definition was edited; no provenance hash was overwritten to imitate a match. `REAL_RUNTIME_DIGESTS.json`, `RETAINED_ROOT_DIAGNOSIS.json`, and `REAL_CONTENT_VALIDATION.json` retain the evidence.

This is source/host proof against a retained worker receipt, not a new remote positive control. Run3 remains passed=false / returncode 1, never a score. The successful completion and legacy `t4_direct` requirements remain unchanged; the existing observed-timeout branch still has its original stricter runtime-provenance requirement.

## Validation and reviews

- `.venv/bin/python -m pytest src/tac/tests/test_candidate_prefire_intent.py src/tac/tests/test_candidate_seal.py src/tac/tests/test_decode_wall_clock_t4_direct.py -q`: **146 passed**, `PYTEST_REQUIRED_FINAL.txt`.
- `.venv/bin/python -m pytest src/tac/tests/test_contest_auth_eval.py src/tac/tests/test_modal_auth_eval.py tools/tests/test_fire_modal_auth_eval_axis.py src/tac/tests/test_modal_source_snapshot.py src/tac/tests/test_fire_modal_lane_id_promotable.py -q -k 'not test_mount_ignore_still_excludes_git_tree_and_special_files'`: **181 passed, 1 deselected**, `PYTEST_ADJACENT_FINAL.txt`.
- The unfiltered run attempted the existing socket test and failed at `socket.bind` with `PermissionError: Operation not permitted` under the sandbox. It was not silently skipped or represented as passing; the initial output is in `PYTEST_CORRECTED.txt`. Its other initial failure was a new fixture missing the real upstream tree; the fixture now links that read-only tree, and the required suite passes.
- Ruff checks all five changed Python files clean (`RUFF_FINAL.txt`); `git diff --check` clean. Actual evaluator argparse accepts `--expected-runtime-content-tree-sha256` (`EVALUATOR_ARGPARSE_CHECK.json`).
- Corrected review passes 1 and 2 checked source-to-worker argument flow, changed-byte/missing-manifest negatives, independent intent identity, replay and retention preservation, normal-path behavior, ledger lock selection and foreign imports. Post-edit hashes and all ten successful `review_tracker.py mark-file` invocations are in `REVIEW_CORRECTED_1.json`, `REVIEW_CORRECTED_2.json`, and their `REVIEW_MARKS_CORRECTED_*` logs. No review override or independent-reviewer claim.

Broad fixtures substitute Git/raw transport/provider state; new digest algorithms, argv branches, local pin guard, real append/lifecycle, and foreign imports execute real code. No scorer or Modal service was invoked.

## Manifest and landing custody

`PREFIRE_IMPLEMENTATION_MANIFEST_AMENDMENT3.json` contains the 15 sorted `PREFIRE_IMPLEMENTATION_PATHS` **plus `experiments/contest_auth_eval.py`**, which the constant does not include. All 16 rows are pinned from the corrected post-review bytes; this satisfies MAIN's explicit correction without editing the contract's constant. `FREEZE_APPEND_DRAFT.json` preserves the latest frozen schema and exact scoped-risk definition and uses the required implementation-commit placeholder. MAIN must replace it only after verifying every row against committed blobs, then append amendment[3] without changing [0..2]. No claim that a corrected implementation commit already exists.

The one serializer call used every post-edit SHA, `[no-triality] [p0-ledger-ok]`, and `--no-co-author`. It refused Git object writes and produced fallback commit `011a034559af6865573d8d4b16ebf14fced71ea4` (rc 17). **That bundle is pre-correction and MUST NOT BE LANDED.** `OBSOLETE_BUNDLE.json` labels it; `SERIALIZER_STATUS.json` records the actual outcome. No second attempt was made under the once-only charter. No writes to an SSD were allowed, so the bounded source fallback used this arm's local evidence directory (about 464,714 B projected, canonical 40 GiB reserve passed); it contains no bulk dataset. Final reviewed files remain in the shared tree and are enumerated by `FINAL_INTENDED_FILES.json`; `FINAL_SOURCE.patch` provides the exact source/evidence diff for MAIN. Historical artifacts are preserved, not deleted.

## Provenance pins

| Object | SHA-256 | Bytes |
|---|---|---|
| `PREFIRE_CONTRACT_FROZEN.json` | `f2d94a58cd1abbe29bf089703a79d66c6f20253637da336ff4f8c58bf51ac85b` | 10459 |
| `ddm_pr15_ratify_fire_tool_fix_amendment1_and_first_measurement_findings_20260910.md` | `39efb49c2e0f34b9c36f7b8a858afe9e68ff7878efa4f637ef650167eedd7560` | 18114 |
| `modal_auth_eval_spawn.json` | `e07653995a4d2496b867e312eb011e61e030a20eb6bb30b62014403573877e94` | 8834 |
| `MODAL_REMOTE_RESULT.json` | `525a9bb57f5462410f321fabb57410c8e5e59bc2c57b3079176e15feb00f72f5` | 80671 |
| `modal_call_id_ledger.jsonl` | `f80d5fdc147715014b2e39e63b03fe64edd75bdf0bccc5d9d630a5ca4d7cbb50` | 1504 |

Full paths are in `BASELINE.json`. Frozen receipt has amendments[0..2] and remains unchanged. Historical implementation pins are `808da43c02a32ae328fd73c707331b9aaee2ac36` / `7a837a9de38f1be1d6a023e0a6caab0e7696a1a2`. Move 43 handoff commit is `48109233e`, archive `7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e`. Live canonical pointer state was read directly; it is not a tracked file in that commit.

## RECALL EVIDENCE

I searched the full research memo corpus by content for `runtime.tree.hash|measure_t4_runtime_digest|register_dispatched_call_id|verify_dispatch_paths`; canonical indexes and all `sub015_DAG_*` for `first.measurement|runtime.digest|dispatch.path`; design docs/SPEC for the same terms; task ledger and lane registry for `first.measurement|runtime.digest|ddm_ffi5`. Exact queries and results: `RECALL_QUERIES.json`, `research_recall.txt`, `graph_recall.txt`, `design_recall.txt`, `ledger_recall.txt`. The canonical-equations listing succeeded; no matching runtime-digest/prefire/dispatch-path term was found in its JSON scope (`equations_recall.txt`). External memory lookup found no relevant entry; no external memory fact was used.

Beyond the named seeds, the actual r9m memo (`ddm_r9m_first_contest_cpu_row_20260804/runtime_tree_hash_verification_20260804.md`) explained path projection and byte-identical file rows; hv1/gs3 dispatch records separated snapshot imports from repo cwd. An old DAG match was another vehicle's first measurement and added no rule. Design catalog entries reinforced fail-closed registration; existing task rows already owned later MAIN fire/harvest. This recall changed the investigation: inspect projection roots and module-relative defaults rather than repeat the charter's causal summary. That exposed and exactly reproduced the mismatch before MAIN's correction arrived.

## Boundaries and follow-on ownership

No Modal API/CLI, fire, live authorization/nonce reservation, timing window, n600 scorer/decode/training, or pointer move. Unit-test fixture lifecycle state is isolated. No writes to either SSD, upstream, PR/sealed trees, retained snapshot ledger/tree, gdc2's directory, frozen receipt, common-contract forbidden files, `candidate_seal.py`, or `decode_wall_clock.py`. No existing schema, digest definition or legacy `t4_direct` rule changed. The corrected first-measurement content-validation selection is exactly MAIN's later authorized replacement. No payload bytes were discarded. Unrelated work and the staged index remain preserved.

`NEXT_FIRE_ORDERS.json` holds the dispositions. The retained-root cure is **FOLDED into this completed implementation**. Any future real producer/authorize/fire/harvest remains **FOLDED into MAIN's existing lifecycle** after a new freeze and fresh committed intent/authorization; no extra dispatch order or reused run3 nonce is created.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer store `.omx/research/ddm_ffi5_20260910/FINAL_SOURCE.patch`; trigger harvest of this corrected packet:** land exact final files in one serializer commit, verify all 16 manifest rows at that commit, fill the draft placeholder, and append amendment[3]. Never land obsolete bundle `011a0345`.
- **QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer store `.omx/research/ddm_ffi5_20260910/PYTEST_CORRECTED.txt`; trigger unrestricted host at landing:** rerun the one existing sandbox-blocked Unix socket test and retain its result here.

## LIVE-HYPOTHESES

- A fresh first-measurement dispatch can now pass the observed runtime-identity failure: the actual retained worker manifest passes the corrected content check, and byte mutation refuses. Full remote execution, timing, score and completion remain untested and belong to MAIN's frozen lifecycle.

## DEAD-ENDS

- Run3 used the intent's seal digest: disproved by retained argv/request; it used the normal upload projection.
- Explicitly pinning that normal path-tree digest alone cures retention: disproved by exact retained-root reproduction; superseded by MAIN's content-validation correction.
- Trusting snapshot-module defaults for the repo ledger: closed by the actual-call regression; explicit ledger and canonical lock paths are required.
- Landing fallback `011a0345`: superseded; its implementation predates the corrected content path. Preserve it as history only.
- Weakening replay, editing earlier freeze rows, discarding retention, or claiming run3 as a score: not performed and not authorized.

composition S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (move 43) unchanged.

<!-- # FORMALIZATION_PENDING: corrected consumer implementation and scorer-free provenance validation; no new contest score -->
