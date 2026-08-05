# NP1 Receipt - Arm Final-Message Persistence And NEXT_IF_RESUMED Surface

## Answer First

NP1 landed the record-censoring apparatus path, scorer-free. Future codex arm completions now copy the full `codex exec -o <name>.last.txt` final message byte-for-byte into `.omx/research/arm_final_messages/<name>_<utcstamp>.md`, append a locked JSONL index row at `.omx/state/codex_arm_queue.final_messages.jsonl`, and extract `NEXT_IF_RESUMED` / `NEXT-IF-RESUMED` / `Next If Resumed` blocks into `.omx/state/codex_arm_queue.next_if_resumed.jsonl`.

Mechanism choice: use the existing `.last.txt` capture rather than re-plumbing the process stream. The keeper already asks `codex exec` to write the full final message with `-o`; the smallest correct fix is a post-exit `persist-final` hook that copies that captured file, indexes it, and runs the extractor. The spawn command, model/profile law, scorer-slot law, and SSD `--add-dir` routing remain unchanged.

Readers:

| reader | where it looks |
|---|---|
| MAIN harvest ritual | `.venv/bin/python tools/codex_arm_queue.py status` now advertises `.omx/state/codex_arm_queue.next_if_resumed.jsonl`; `tools/codex_harvest_commit.py` remains the per-arm harvest/commit path and this surface is the plan-of-record input during drain. |
| costate duty queue | `tools/costate_digest.py::section_arm_next_if_resumed` reads `.omx/state/codex_arm_queue.next_if_resumed.jsonl`; the existing `section_orphaned_followons` / `tac.followon_ledger` path still scans `.omx/research` memos, including future persisted final messages. |

No scorer forwards, no `upstream/evaluate.py`, no launches, no paid dispatch, no protected files.

## Recall Evidence

| query / source | scope | finding beyond charter seed | plan impact |
|---|---|---|---|
| `np1_prompt`, `_common_contract`, `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, operating manual, `.omx/state/main_hot_state.md` | required governing files | NP1 is $0, scorer-free; od9 owns the scorer slot; live own-vehicle line is `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`. | Kept work to apparatus and status/read surfaces only. |
| `#878`, `NEXT-IF-RESUMED`, `arm final-message`, `record-censor` | `.omx/research`, `.omx/state`, DAG, queue state | `harness_tasklist_bridge_20260803.jsonl` still marks #878 pending/class open; queue row 215 names NP1; DAG repeats the 9-in-5 structural gap. | Treated the prior `09fca46f37` findings-field fix as insufficient for full final messages and NEXT blocks. |
| `.omx/research/ddm_fo1_orphaned_followon_detector_20260801.md:109-117` | source audit | FO1 measured `NEXT-IF-RESUMED` as 9 occurrences in 5 files and identified structural absence from disk. | Backfill used the pre-08-02 marker population, but emitted only parser-valid plan blocks to avoid phantom rows from audit/count mentions. |
| `.omx/research/ddm_main_friction_audit_20260802.md:25-42` | friction audit | Final-message loss is broader than the 9-in-5 marker count; killed arms lost high-value findings. | Persist the full final message, not only NEXT blocks. |
| `tools/list_canonical_equations.py --json | jq length` | canonical equations registry | 422 equations listed; no score/equation change needed for NP1. | No canonical-equation edit. |
| `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, `harness_tasklist_bridge_20260803.jsonl` | research index / graph-memory surfaces | DAG records the same structural gap; no competing current fix was found in the searched scope. | Extended the existing arm queue state, not a new independent queue. |

## Implementation

- `tools/codex_arm_queue.py`
  - Added locked append-only JSONL helpers.
  - Added `persist-final`: copies full `.last.txt` bytes into `.omx/research/arm_final_messages/`, writes the required index row `{name, rc, elapsed, path, sha256}`, then extracts NEXT blocks from the persisted final.
  - Added `extract-next`: idempotently extracts NEXT blocks from persisted finals and receipt files into `.omx/state/codex_arm_queue.next_if_resumed.jsonl`.
  - Added queue `status` visibility for both surfaces.
  - Added the keeper post-exit hook; no spawn/model/profile law lines were changed.
- `tools/costate_digest.py`
  - Added read-only `section_arm_next_if_resumed`, surfaced beside the existing orphaned-follow-on duty section.
- Tests cover title-case TC1 spelling, contract spellings, no-block negative control, idempotent rows, standalone `ddm_<arm>_...` memo name inference, final-message byte copy, final-message index row, and costate reader schema filtering.

## Positive Controls

| command | result |
|---|---|
| `extract-next --provenance positive-control --source .omx/research/ddm_tc1_20260805/TC1_RECEIPT.md --source .omx/research/ddm_au1_20260805/AU1_RECEIPT.md` | `blocks_seen=2`, `files_with_rows=2`, `written=2` on first clean rebuild; rerun wrote `0` by row-id idempotence. |
| `extract-next --provenance negative-control --source .omx/research/ddm_bn1_20260805/BN1_RECEIPT_20260805.md` | `blocks_seen=0`, `files_with_rows=0`, `written=0`; no phantom row. |
| `extract-next --provenance dogfood --source .omx/research/ddm_np1_20260805/NP1_RECEIPT.md` | `blocks_seen=1`, `files_with_rows=1`, `sources=1`, `written=1`; this receipt's own block was caught. |
| `extract-next --provenance today-receipt --source-glob ...20260805...` | `blocks_seen=35`, `files_with_rows=34`, `sources=42`, `written=32` when run after AU1, TC1, and NP1 were already present; m38 prediction `>=15` held. |

Current surface count after this receipt's dogfood row and the today sweep: 38 rows = 3 `backfilled`, 1 `dogfood`, 2 `positive-control`, 32 `today-receipt`.

## Backfill Table

| provenance | arm | source | extracted line | note |
|---|---|---|---:|---|
| `backfilled` | `dw1` | `.omx/research/ddm_dw1_qa75_distill_window_20260730.md` | 180 | Parser-valid combined LIVE/DEAD/NEXT block. |
| `backfilled` | `pj1` | `.omx/research/ddm_pj1_projection_probe_20260730.md` | 120 | Parser-valid combined LIVE/DEAD/NEXT block. |
| `backfilled` | `su2` | `.omx/research/ddm_su2_pose_endgame_program_20260730.md` | 263 | Parser-valid combined LIVE/DEAD/NEXT block. |
| bounded non-emitted | `fo1` | `.omx/research/ddm_fo1_orphaned_followon_detector_20260801.md` | 111, 117 | Audit/count evidence, not a continuation plan; not emitted. |
| bounded non-emitted | `fu1` | `.omx/research/ddm_fu1_followup_sweep_20260730.md` | 3, 18 | Store/table references to `ax1 NEXT-IF-RESUMED`; not emitted as a block because the line is not itself a resumable plan block. |

This is the honest distinction between the historical marker count and the new queue-consumable block surface. The extractor refuses to turn count/table mentions into plan rows.

## Verification

| check | result |
|---|---|
| `.venv/bin/python -m pytest src/tac/tests/test_codex_arm_queue.py` | `31 passed` |
| `.venv/bin/python -m pytest src/tac/tests/test_costate_digest_telemetry_binding.py` | `6 passed` |
| `.venv/bin/python -m compileall -q tools/codex_arm_queue.py tools/costate_digest.py src/tac/tests/test_codex_arm_queue.py src/tac/tests/test_costate_digest_telemetry_binding.py` | pass |
| `.venv/bin/python tools/codex_arm_queue.py status` | clean rc; reports watcher alive and advertises final-message/NEXT surfaces |
| `.venv/bin/python -c "import sys; sys.path.insert(0, 'tools'); import costate_digest as cd; print(cd.section_arm_next_if_resumed()[0])"` | `arm-next-if-resumed: 38 plan row(s) ... backfilled=3, dogfood=1, positive-control=2, today-receipt=32` |
| review tracker | two `mark-file --status reviewed --reviewer codex-np1` passes each for the four touched Python files |

Post-edit SHA-256:

| path | sha256 |
|---|---|
| `tools/codex_arm_queue.py` | `68b1753358b219e601182063b1b09fbc7398a32c9518d9619b005b861633bf20` |
| `tools/costate_digest.py` | `d69be161140620be136dd9a36dffa0548b2c54c88391e1c12fe206783fad1d99` |
| `src/tac/tests/test_codex_arm_queue.py` | `0b35238783a21a1a8225e35bc0a7c8a93fc6621980f72b1896bca3d714357eba` |
| `src/tac/tests/test_costate_digest_telemetry_binding.py` | `f89a60770e86204c353224995deb094d9956e50d019bc2c84110775fbf3ec414` |

## Boundaries

Measured in NP1: parser behavior over real AU1, TC1, BN1, pre-08-02 backfill files, and all 2026-08-05 receipt/NEXT files; queue status smoke; costate reader direct smoke; focused tests.

Not measured: any scorer output, any exact archive score, any full live arm completion under the new keeper after this patch. That future production event will create `.omx/research/arm_final_messages/<name>_<utcstamp>.md` plus an index row automatically.

## NEXT_IF_RESUMED

1. After the next codex arm completes under a newly generated keeper, inspect `.omx/research/arm_final_messages/<name>_<utcstamp>.md`, `.omx/state/codex_arm_queue.final_messages.jsonl`, and `.omx/state/codex_arm_queue.next_if_resumed.jsonl` together; confirm the final-message SHA matches the copied `.last.txt` bytes and that any NEXT block appears with `provenance=harvested-final`.
2. If MAIN harvest wants terminal closeout automation, make `tools/codex_harvest_commit.py` print the matching `codex_arm_queue.next_if_resumed` rows for the harvested label before requiring `--consumed-by`; keep it as a reader of this surface, not a second queue.
3. If the historical #878 marker-count audit must be represented one-row-per-mention, add a separate `marker_audit` schema; do not loosen the plan-block extractor to emit audit/count/table references as resumable work.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
