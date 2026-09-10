# ddm_cpx1 — STOP: the retained CPU refusal does not bind move 43

STOP: A genuine submission_policy_adjudication.v1 cannot be produced from this packet’s passed checks because the retained CPU refusal binds a different archive and a different content-only runtime digest.

The charter's conditional STOP applies before implementation. This is a blocked handoff, **not the requested checker cure**. No parser, checker, receiver, or raw receipt was changed. The exact pointer did not move.

## Measured result

| Check group | Before | Fresh unchanged-checker run | Code / verification |
|---|---:|---:|---|
| CPU score parse, archive SHA, archive size, metric consistency, runtime recorded | 0/5 | 0/5 | `inspect_contest_cpu_auth_eval`; no refusal schema implemented because real positive bindings are unavailable |
| Raw promotion policy blockers absent | 0/1 | 0/1 | `inspect_auth_eval`; no policy adjudication produced from missing bindings |
| Runtime import allowlist | 0/1 | 0/1 | Receiver-change boundary; unchanged |
| Hosted archive manifest | 0/1 | 0/1 | Publish boundary; unchanged |
| Remaining checks | 85/85 | 85/85 | Same swp2 argv, only `--json-out` redirected to this arm's evidence directory |
| Total | **85/93** | **85/93, rc 1** | `COMPLIANCE_COMMAND.json`, `COMPLIANCE.json`, `COMPLIANCE_SUMMARY.json` |

All counts above are **MEASURED [macOS-CPU checker diagnostic]**, not scoring runs. Expected 91/93 was not obtained. The requested run with a new refusal receipt was not performed: manufacturing that receipt with these historical bindings would violate the charter.

Regression validation: **95 existing tests passed in 8.04 s**, covering `test_pre_submission_compliance_check.py`, `test_auth_eval_schema.py`, `test_adjudicate_contest_auth_eval_policy.py`, and `test_pre_submission_compliance_frontier_score.py`. Command: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_pre_submission_compliance_check.py src/tac/tests/test_auth_eval_schema.py src/tac/tests/test_adjudicate_contest_auth_eval_policy.py tests/test_pre_submission_compliance_frontier_score.py`. Ruff on the unchanged checker and canonical schema helper passed. These are baseline tests, **not tests of an implemented cure**. No `.py` files were edited, so no new Python review passes or new-schema round-trip tests are claimed.

## Binding contradiction

| Binding | Retained CPU refusal | Staged move 43 |
|---|---|---|
| Archive SHA-256 | `f111ab4259c757409e791247d33978a714ceb1cd66e50c149e2e65fbf208756f` | `7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e` |
| Archive bytes | 180,238 | 180,466 |
| `inflate.py` SHA-256 | `74cdc32cbfedd29ab6a2c3ff27b825ceeed9c518f5f298005ed1c3a4c9e8d147` | `598dbd7acb3146f6b3e3779fd82119decf5c55f302d2e1e984365ba1955e7749` |
| Environment-free `runtime_files_sha256` | `11df366af4cf87eed3d94d84c76257a6b72735bb07e88d23379906396a5e3c76` | `ffdbde488ac17d3708b09d9b462aa62ef3b63d6d57e4198347995912df15e5ce` |

The archive SHA/size were recomputed from the staged archive. All **45** retained runtime rows were compared with the staged files: **44 match; only `inflate.py` differs**. The original refusal's environment-free digest was reconstructed from its retained manifest and exactly matches its recorded value. Applying the same definition to the staged files produces the unequal digest above. This is a content mismatch, not merely a path mismatch. Evidence: `RUNTIME_COMPARISON.json` and `INPUT_BINDINGS.json`.

The three evaluator digest definitions are distinct:

1. `runtime_tree_sha256`: canonical JSON of runtime root name, full file rows, dependency roots, repo-local import manifest, and upstream evaluator identity. CPU receipt `dad46453992038bd5f8490b3b4c17c84178a7c9906de7eb08ca764a47e8b6924`; move-43 CUDA receipt `a726739a52452f8824c4eb9f98322f927b8b5d2d4c6ac1717911f61b841a9803`; local checker tree `750fc1e4e099db72d138d67db2e9d4028786531b4fbbadfe1587a5bfc8ccb8d7`. These include location-dependent structure.
2. `runtime_content_tree_sha256`: removes root/location fields from runtime files but retains dependency/import-scan structure. CPU receipt `0cf045309761acd63c98c535cf34ccb0182f1978ea339007529d7f1775aaba95`. The checker's existing portable projection for move 43 is `5863425ffe98033df7be70716563a7583473981d69349f7d49533a14764027ab`, identical for the staged tree and its CUDA receipt, unequal to the historical CPU receipt.
3. `runtime_files_sha256`: **the chosen environment-free comparison**, defined in `experiments.contest_auth_eval._runtime_dependency_manifest`: SHA-256 of sorted, compact JSON containing only `files` (sorted `relative_path`, `bytes`, `sha256` triples) and `upstream_evaluate_py`. It excludes dependency roots and import scans. It also differs, so adopting this safer definition cannot cure the instance.

The guard location in the charter is inaccurate. `inflate.sh:83` calls `python "$HERE/inflate.py" ...`. The actual condition is `inflate.py:56`, `if not torch.cuda.is_available():`; lines 57–60 raise the recorded CUDA requirement. The staged Python entry point verifies the archive at line 53 **before** that guard, using its counted archive's SHA/size pins at lines 18–19. Shared guard text alone cannot establish the full required receiver identity.

The underlying receipt is `/Volumes/VertigoDataTier/pact/ddm_rp1_round2_cpu_20260910/MODAL_REMOTE_RESULT.json`, **46,910 B**, SHA-256 `b601f731e72f4a7067a6cdea8ef7c2c783f609c8561ae926b36bb9f2bfb480fe`. Its retained logs show return code 1 and the CUDA exception; no CPU eval JSON was produced. Its Modal call identity is corroborated by the supplied adjudication and `active_lane_dispatch_claims.md`: `fc-01M25W34DGHHJES03531X3VDXV`. This arm did not contact Modal.

Provenance pins re-read and verified:

- swp2 memo: `8e0a3ebe66895315767173b224a5ee70db1cc80c14c9208f57e3a5b4e21819c4`.
- swp2 `BLOCKERS.json`: `d1c6fa40cca0d6cd81366c7f34cce3a5e3e621baa65a705a123b90f6eaaa5a82`.
- supplied `CPU_AXIS_ADJUDICATION.json`: **910 B**, `fae7d78a79552b96eb8fd5ac81e59dfdae3b7d722d0b9a20e66abed272f4925d`.
- move-43 pointer commit supplied by charter: `48109233e`; archive rehashed as above. No new score or pointer claim.

## RECALL EVIDENCE

`RECALL_SEARCHES.json` records queries and outputs. Searched all research markdown by content for `cpu.{0,30}refus|policy.adjudicat|raw_auth_eval_does_not_verify_submission_policy_gates|runtime.{0,20}digest`; all research JSON/JSONL for refusal schema/outcome/call-ID strings; canonical index and sub015 DAG; docs and research SPEC/spec paths; canonical task-status and operator P0 ledgers. The canonical equations CLI emitted **483 rows**; the recorded refusal/adjudication/runtime-digest query matched none. The raw equation dump remains retained locally with its SHA in `RECALL_SEARCHES.json`. Graph coverage is the named index/DAG files, not an unqueried graph service.

Beyond the charter seeds, pq1's `GAP_REPORT.md` ties policy adjudication to actual CPU/final-compliance evidence; the research index points back to the canonical evaluator; the r9m runtime verification documents a real host/import-scan mismatch, motivating the stronger environment-free comparison. Source recall also found the existing `scripts/adjudicate_contest_auth_eval.py::_check_raw_promotion_policy_gate`, which explicitly refuses laundering raw blockers after only component/hardware checks. Its regression suite remains green. No alternate move-43 CPU refusal was found in the recorded research JSON/JSONL query: the only matching path was the supplied move-42 adjudication. This is a bounded search result, not a claim that no such receipt exists elsewhere.

What changed: rather than treating the five missing CPU fields as an exclusively syntactic defect, verified the retained refusal's actual archive/runtime custody. The mismatch prevents the real-positive fixture and the downstream genuine policy object under the charter's equality rule. Neither re-labeling move 42 nor normalizing away an archive pin is authorized by that rule. Current hot state and September 9–10 directives were consulted; stale frontier prose in the common contract was not used as current authority.

## Boundaries and landing

No staged/live PR tree, receiver, `upstream/`, volume path, or common-contract forbidden file was edited. No publish, message to another party, Modal, scorer, GPU, training, GT decode, pointer mutation, stash, or direct staged-index mutation occurred. No payload was generated or discarded. No cleanup/delete was performed. New evidence is local and durable under `.omx/research/ddm_cpx1_20260910/`; provider scratch strings in existing receipts are provenance, never new consumer evidence paths. `BOUNDARY_VERIFICATION.json` rehashes the recorded source/packet/receipt files and confirms an unchanged index before serialization. Unrelated dirty work was preserved.

This is a metadata-only STOP landing, once and last via the serializer, with post-edit SHA-256 per file, no co-author trailer, `[no-triality] [p0-ledger-ok]`, and a local fallback directory to honor the prohibition on volume writes. Actual return code and bundle/commit custody are recorded after the attempt in `SERIALIZER_STATUS.json`. A fallback commit is not a main-branch landing. Checkpoint identity is `ddm_cpx1`; status remains blocked for the implementation. All six solver hooks are N/A: no sensitivity, Pareto constraint, allocator, dispatch implementation, posterior measurement, or solver probe changes. Consumer routing is `FIRE_ORDERS.json` plus the canonical checkpoint.

## NEXT_IF_RESUMED

- QUEUED-WITH-A-FIRE-ORDER; owner MAIN / compliance-policy owner; consumer `.omx/research/ddm_cpx1_20260910/FIRE_ORDERS.json`; fire at handoff harvest: supply matching archive/runtime refusal evidence, or explicitly amend the charter for a separately proven archive-pin-normalized refusal reuse policy, before resuming implementation. Correct the guard's source location. This order does not authorize a dispatch.
- QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer `.omx/research/ddm_cpx1_20260910/SERIALIZER_STATUS.json`; fire only on rc 17 and verified bundle custody: land the exact metadata handoff without marking the checker cure complete.

composition S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (move 43) unchanged.

## LIVE-HYPOTHESES

- An explicitly versioned reuse policy may prove the existing refusal transfers across archive-pin changes: 44/45 runtime files match and the current guard precedes decode. This is plausible, not established; it requires a rule amendment and actual comparison of the differing entry points, not an equality waiver hidden in a digest.

## DEAD-ENDS

- INSTANCE: re-labeling the retained move-42 receipt as move-43 evidence is closed by unequal archive SHA, size, and environment-free runtime digest.
- INSTANCE: fixing only environment/path normalization is insufficient; `inflate.py` bytes differ.
- INSTANCE: claiming an exact CUDA guard line in `inflate.sh` is false; the guard is in `inflate.py`.
- INSTANCE: a policy JSON containing passed-check flags cannot replace the missing bound CPU evidence. No genuine policy object or 91/93 result was produced.

<!-- # FORMALIZATION_PENDING: checker custody STOP; no implementation, new equation, or measured scoring row -->
