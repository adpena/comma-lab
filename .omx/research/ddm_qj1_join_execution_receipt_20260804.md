---
schema: ddm_qj1_join_execution_receipt.v1
date_utc: 2026-08-04
arm: ddm_qj1
charter: ".omx/tmp/codex_runs/qj1_prompt.md"
common_contract: ".omx/tmp/codex_runs/_common_contract.md"
research_only: true
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
axis: "[macOS-CPU advisory; scorer-free backlog join; no scorer forwards]"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_qj1 Join Execution Receipt

## Answer First

The join surface exists, is wired, and was rerun live:

- Code: `src/tac/followon_backlog_join.py`
- CLI: `tools/build_followon_backlog_join.py`
- Tests: `src/tac/tests/test_followon_backlog_join.py`
- Landing commit: `e311aff585` (`ddm_qj1: build follow-on backlog join [no-triality] [p0-ledger-ok]`)

Live execution output:

- JSON: `.omx/research/ddm_qj1_followon_backlog_join_20260804.json`
- Markdown: `.omx/research/ddm_qj1_followon_backlog_join_20260804.md`
- Command: `.venv/bin/python tools/build_followon_backlog_join.py --since 2026-07-18 --output-json .omx/research/ddm_qj1_followon_backlog_join_20260804.json --output-md .omx/research/ddm_qj1_followon_backlog_join_20260804.md --no-cache`
- Result: 778 dispositions; 390 queued rows all have owners; unowned queued rows = 0.

This is apparatus. It did not run a scorer, did not mutate `upstream/`, did not touch the
staged git index, and did not claim a score.

## Live Join Counts

Denominators from the live report generated 2026-08-04T16:24:23Z:

| surface | examined / declared / population | verdicts |
|---|---:|---|
| memo follow-ons | 1305 / 1305 / 7499 | ORPHANED 1, STAGED 6, UNKNOWN 141, EXECUTED 14 |
| canonical task rows | 190 / 190 / 190 | EXECUTED 18, UNKNOWN 172 |
| handoff edges | 426 / 426 / 7499 | ORPHANED 23, LIVE 125, ADVANCED 278, UNVERIFIABLE 0 |

Canary: task join passed, firing on the verified-present control and staying silent on the
verified-absent control.

## Charter Head Dispositions

The charter named #870, #879, #880, #886, and #887. The live repo task-status ledger does not
contain current rows for #879/#886/#887, so qj1 consumed them by content through the landed memos
and harvest receipts rather than by bare task id.

| head | disposition | evidence / fire order |
|---|---|---|
| #870 never-run $0 class | FIRED | `ddm_fo1_orphaned_followon_detector_20260801.md` built `tac.followon_ledger` and harvested the real wr1 receipts. Live class was much smaller than claimed: 0 ORPHANED, 4 STAGED, 84 UNKNOWN, 10 EXECUTED in its 14-day scope. Residual is retrieval/stale-never-run claims, not a large live orphan pile. |
| #879 memo UNKNOWN population | FIRED then FOLDED/QUEUED | `ddm_p1a_followon_unknown_adjudication_20260801.md` reduced 86 UNKNOWN rows to 25 not-a-follow-on, 21 already-done, and 40 real-debt rows collapsing to 29 open items. QA52 was already fired by `ddm_kl1` on 2026-07-30; QA52-b is deferred to QA57 with a named owner/condition. |
| #880 task-row join | FIRED | `ddm_p2a_task_backlog_drain_20260801.md` replayed the harness transcript and found 114 open rows, measured the refusal-receipt false-closed defect, and fixed the shipped join. Current qj1 join covers repo-visible canonical task rows and labels the missing harness mirror as a boundary, not a fake full ledger. |
| #886/#887 pre-ranked heads | FOLDED into the harvest bundle | `ddm_hv2_two_week_harvest_20260803.md` explicitly consumed "#879's 84 UNKNOWN + QA52, #886/#887's 29+18" by content; `ddm_iv1_inventory_drain_20260803.md` then fired the head bundle. No bare-id lookup was used because hv2 re-measured those ids absent from the repo ledger. |
| p1a/p2a top cost-to-falsify rows | FIRED / QUEUED-WITH-FIRE-ORDER | `ddm_iv1_inventory_drain_20260803.md` fired p1a head 1, p1a QA55/codec head, p2a T1 tier, oh1 rows 3-6, op3 #826 re-encode, and vehicle-scope owed items. Its table leaves each head with an owner or fire-condition. |
| p1a #28 Modal T4 paid row | HONESTLY-DROPPED-WITH-REASON for this run | Not fired. `ddm_iv1` could not verify a terminal staged runner in 10,235 tracked Python files, and the better-verified sibling would run a full 600-sample scorer forward. qj1 does not own the scorer slot and the common contract forbids scorer work here. |

One live delta from this qj1 rerun is `ddm_wk3_residue_disposition_20260804.md#L7`. It is a
boundary sentence ("I did not run a full evaluator"), not a follow-on request. It remains visible
in the generated join as `UNKNOWN` because changing the extractor semantics would move historical
fo1/p1a denominators; hand disposition here is HONESTLY-DROPPED-WITH-REASON.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_followon_backlog_join.py src/tac/tests/test_followon_ledger.py` -> 46 passed.
- `.venv/bin/python tools/canonical_task_status.py --validate` -> `{"rows": 522, "status": "valid"}`.
- `git show --stat --oneline e311aff585 -- src/tac/followon_backlog_join.py tools/build_followon_backlog_join.py src/tac/tests/test_followon_backlog_join.py` confirms the join surface landed in commit `e311aff585`.

Canonical task-status validation emitted historical custody warnings for older rows with uncustodied
delta-S prose and one malformed list-valued `event_notes`; validation still completed successfully.

## Boundary

The generated join is repo-visible, not the live harness TaskList. `EXECUTED` and `ADVANCED` are
candidate closure signals; hand verification is still required before closing safety-critical work.
Rows whose only honest state is artifact-undecidable exit as `QUEUED-WITH-FIRE-ORDER` with an owner,
not as completed work.

Own-vehicle frontier unchanged by qj1:
`S = 0.7541459 @ 358,084 B [macOS-CPU advisory] n600`.
