# ddm_ah1 Receipt

status: COMPLETE-SCOPED
arm: ddm_ah1
date: 2026-08-07
charter: `.omx/tmp/codex_runs/ah1_prompt.md`
common_contract: `.omx/tmp/codex_runs/_common_contract.md`

## Counts First

| item | count / status |
|---|---:|
| Persisted final-message packets read in scoped corpus | 13 |
| Review findings audited | 8 |
| Review findings with named consumer or route | 8 |
| Review findings fully structurally closed in checked scope | 7 |
| Known orphan routed now | 1 |
| #254 backlog before AH1 | 145 |
| #254 backlog after AH1 | 140 |
| Targeted tests | 37 passed |
| Scorer / archive / remote launches | 0 |

## Hardening Status

| ID | status | landed surface |
|---|---|---|
| H1 | DONE | `tools/launch_detached_process.py --verify-alive-secs` default 3; immediate child exit prints rc plus log tail and returns nonzero. |
| H2 | DONE | `tools/safe_run.py --status-receipt` / `SAFE_RUN_STATUS_RECEIPT`; atomic JSON updated at spawn, every RSS sample tick, kill, and final exit with `peak_rss_observed`, `last_sample_ts`, and `kill_action`. |
| H3 | DONE | `tools/codex_arm_watch.py` emits `ARM <name> ALERT ...` for `rc!=0`, malformed rc, or `signal=` `.done` receipts. |
| H4 | DONE-PARTIAL | Five top-tier #254 entries now guard raw non-smoke training: ANR token renderer, Balle hyperprior, BlockNeRV renderer, categorical renderer, ChARM toy substrate. Warn-only backlog remains 140. |
| H5 | DONE-DESIGN | `DRIVER_RESUME_SEMANTICS_NOTE.md`; no driver code changed. |

## CQ1 Verification

Command executed:

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python` import/evaluator/provenance checker over the five CQ1 modules.

Results:

| equation | evaluator result checked | provenance anchors |
|---|---|---|
| `ddm_et4_twelfth_move_solver_carriage_split_v1` | solver split output with break-even and rate deltas | source and anchor artifacts existed |
| `ddm_et5_restricted_carriage_family_fold_v1` | folded=true, selected_count=0 | source and anchor artifacts existed |
| `ddm_rr8_stage_rc_success_contract_v1` | nonzero stage rc -> success=false | source and anchor artifacts existed |
| `ddm_rr9_mem_probe_fire_protocol_v1` | missing required receipt -> allowed=false | source and anchor artifacts existed |
| `ddm_hb1_semantic_label_incumbent_transfer_v1` | external anchor not admissible -> incumbent_stands=true | source and anchor artifacts existed |

Registry surface check:

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/list_canonical_equations.py --json | ...`

Found all five CQ1 equation IDs, missing none.

## Tests

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tools/tests/test_codex_arm_watch.py src/tac/tests/test_no_silent_failure_launch_hardening.py src/tac/tests/test_admission_coverage_gate.py`

Result: `37 passed in 7.14s`.

Additional measurement:

`check_heavy_witness_trainers_call_admission_guard(strict=False, verbose=False)` now returns 140 warnings. The first residual row is `experiments/train_cnerv_as_renderer.py`.

## Audit Artifacts

- `AUDIT_TABLE.md`: corpus classifications and RR8/RR9/RR10 chain.
- `FA1_STAGE_TRANSITION_SOFT_VELOCITY_BLEND.md`: orphan route and backtest consumer.
- `FOLLOWON_LEDGER.jsonl`: queued rows for FA1, RR10-F1 review interlock, and residual #254 backlog.
- `DRIVER_RESUME_SEMANTICS_NOTE.md`: H5 design-only note.
- `NEXT_IF_RESUMED.md`: continuation order and boundaries.

## Recall Evidence

Read/consulted before edits and claims:

- Governing files: `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`.
- Charter files: `.omx/tmp/codex_runs/ah1_prompt.md`, `.omx/tmp/codex_runs/_common_contract.md`.
- Corpus final packets/receipts: et4, et5, et6, fa1, cons1, cq1, mx1b, mx1c, mx1d, fw1, rr8, rr9, rr10.
- Targeted source surfaces: `tools/launch_detached_process.py`, `tools/safe_run.py`, `tools/codex_arm_watch.py`, #254 scanner/tests, top-tier trainer entrypoints.
- Memory quick pass: Pact queue/ownership/audit discipline and current frontier/pointer separation.

## Authority Boundary

This arm is apparatus/audit only. It did not run `upstream/evaluate.py`, did not build `archive.zip`, did not dispatch remote/GPU work, and did not claim a contest or macOS score row.

Own-vehicle frontier line remains the hot-state line observed for this arm: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.

Borrowed contest pointer remains separate and unmoved.

## Commit Attempt

Serializer command attempted with:

- `--no-co-author`
- `--triality-legs none`
- `--triality-reason "apparatus/audit hardening only [no-triality] [p0-ledger-ok]"`
- explicit `--expected-content-sha256` entries for all 17 AH1 files
- `--stdin-files` limited to the AH1 file set
- no `REVIEW_GATE_OVERRIDE`

Outcome: blocked before commit at `git add` with rc=128:

`error: unable to create temporary file: Operation not permitted`

The first failed object insert was `experiments/train_anr_token_renderer.py`. Post-failure index check:

`git diff --cached --name-status -- <AH1 file set>`

returned empty output, so no AH1 file remained staged.
