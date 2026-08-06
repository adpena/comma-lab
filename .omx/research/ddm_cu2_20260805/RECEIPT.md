# DDM CU2 Receipt: commit-custody guards (#883, #911, #914, stale checkpoints)

## Scope

Charter: `.omx/tmp/codex_runs/cu2_prompt.md`.
Common contract: `.omx/tmp/codex_runs/_common_contract.md`.

This was a scorer-free commit-custody unit. No scorer, launch, archive mutation, or pointer update
was performed.

Own-vehicle frontier line remains unchanged: `S = 0.7539807296911207 @ 357,836 B`
`[macOS-CPU advisory]` from `qo1`. Borrowed contest pointer remains unmoved at `0.19108...`.

## Result

| row | verdict | action |
|---|---|---|
| #883 serializer repair/no-stage non-empty-index hazard | FIXED | Added rc=15 staged-file-set guard. Any staged path not declared by `--files` is refused before commit, including `--no-stage` real-index repair paths. |
| #911 absorbed unauthored lines / false commit body | CORRECTED + HARDENED | Added correction memo for `06fa0ad37d`; post-commit recorded-but-not-requested files now hard rc=15. |
| #914 stale staged de1/canonical-equations index hazard | PREMISE-STALE NOW | Live main cached index empty; de1 branch/worktree still exists and its cached index is also empty. No index mutation performed. |
| stale `in_progress` checkpoints after queue landing | FIXED | Sister checkpoint guard now treats latest queue status `landed` or `dropped` as terminal, with `ddm_` alias normalization; live queue rows still block. |

## Code Changes

- `tools/subagent_commit_serializer.py`
  - Added `_staged_declared_file_set_mismatch` and rc=15 refusal for staged-but-not-declared paths.
  - The check runs for patch temp indexes, normal temp-index `git add`, and the real-index
    `--no-stage` path.
  - Post-commit attribution reconciliation is now hard on `recorded - requested`, warn-only on
    `requested - recorded`.
- `src/tac/commit_safety/sister_checkpoint_guard.py`
  - Added best-effort loading of `.omx/state/codex_arm_queue.jsonl`.
  - Latest terminal statuses `landed` and `dropped` neutralize stale in-progress checkpoints.
  - Queue name aliases cover both `fx1` and `ddm_fx1`.
- Tests:
  - `test_no_stage_refuses_staged_decoy_file_before_commit`
  - `test_no_stage_accepts_exact_declared_staged_file_set`
  - `test_landed_queue_row_neutralizes_stale_in_progress_checkpoint`
  - `test_live_queue_row_does_not_neutralize_in_progress_checkpoint`

## RECALL EVIDENCE

Sources loaded before editing:

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`,
  `.omx/state/main_hot_state.md`.
- Memory quick pass:
  - `MEMORY.md:87-96`: prior #914/de1 hazard index.
  - `MEMORY.md:103-113`: #914 no-signal-loss handling and prior empty-index fact.
  - `MEMORY.md:247-248`: queue authority and bridge/task-store distinction.

Targeted recall/provenance:

- `.omx/research/harness_tasklist_bridge_20260803.jsonl:85` records task #883 as pending:
  "two-landing: serializer repair path silently committed non-empty index".
- `.omx/research/ddm_qd1_backlog_drain_20260803.md:120` deferred #883/#911 until it was safe to
  audit without risking sibling work.
- `.omx/research/ddm_cu1_consolidation_disposition_20260803.md:70-72` names the #883 hazard:
  repair path silently committed a non-empty index and dropped sister canonical-equation rows.
- `.omx/research/ddm_rs2_orphan_resumption_20260802.md:246-249` names `06fa0ad37d` as the commit
  that absorbed `ddm_bs2` trainer wiring under an unrelated CLI-help repair.
- `git show --stat --name-status 06fa0ad37d` and `git show --numstat --format= 06fa0ad37d`
  confirmed `experiments/train_tr1_partition_renderer_mlx.py` changed by `208` insertions and
  `7` deletions inside the CLI-help commit.
- `.omx/tmp/codex_runs/codex_events.log:1535-1538` shows `ddm_de1` start/done and
  `LANDING-REVIEW-REQUIRED ddm_de1`.
- Current checks:
  - `git diff --cached --name-status` returned empty in main.
  - `git for-each-ref ... | rg de1` found `codexwt/ddm_de1_20260803T112347Z`.
  - `git -C .omx/tmp/codex_worktrees/ddm_de1_20260803T112347Z status --porcelain=v1` returned empty.
  - `git -C .omx/tmp/codex_worktrees/ddm_de1_20260803T112347Z diff --cached --name-status` returned empty.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_sister_checkpoint_guard.py src/tac/tests/test_subagent_commit_serializer.py src/tac/tests/test_serializer_file_attribution_reconcile.py src/tac/tests/test_subagent_commit_serializer_postcommit_clobber.py
```

Result: `75 passed in 6.95s`.

```bash
.venv/bin/python -m ruff check tools/subagent_commit_serializer.py src/tac/commit_safety/sister_checkpoint_guard.py src/tac/tests/test_sister_checkpoint_guard.py src/tac/tests/test_serializer_file_attribution_reconcile.py
```

Result: `All checks passed!`.

## Triality / Pointer

`[no-triality]`: apparatus and custody guards only. No DSL, DAG, or canonical-equation surface was
changed.

Pointer delta: none. This unit improves commit custody only; it is not goal progress by itself.
