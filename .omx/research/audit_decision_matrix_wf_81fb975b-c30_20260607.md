# Audit Decision Matrix: wf_81fb975b-c30

- Created UTC: 2026-06-07T00:23:55Z
- Authority scope: audit_decision_only_no_score_authority
- Overall decision: C/E remediated; F pose plumbing and runner/source-parity perf blockers remediated, but F launch evidence remains blocked.
- Source workflow status: completed_but_partially_rate_limited, then Codex takeover audit/remediation completed locally.
- Score authority: none minted here; exact archive bytes plus upstream evaluator remain the only score authority.

## Canonical Result Inspection

- No workflow-local result/decision/matrix/summary/verdict file was present under the subagent workflow directory by filename scan.
- Outer workflow result record found: `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/89ff112f-013d-43b5-b949-2a6d43b650c3/workflows/wf_81fb975b-c30.json`
- The outer workflow record is copied into the bridge packet as raw workflow evidence, but it is partial and is not a clearance decision.

## Launch Gate Rule

No audited swarm output may feed v6 or the launch DAG unless its row below says `clear` or `remediated`, and the usual canonical gate for that family still passes. This matrix does not bypass source-forward, receiver-closed, exact replay, or custody gates.

## Target Decisions

| Target | Parsed workflow lenses | Severity counts | Decision | v6/launch DAG consumption | Required action |
|---|---:|---|---|---|---|
| A-survival | 3 | H0 M0 L5 | clear | allowed_after_normal_canonical_gates | None from this audit. LOW notes are nonblocking and no score authority is minted. |
| B-composite | 3 | H0 M3 L4 | remediated | allowed_after_remediation_commit_and_normal_canonical_gates | Require current tree or descendant containing commit 2ef6c747a; keep report-count inaccuracies out of authority claims. |
| C-miner | 3 | H0 M1 L4 | remediated | allowed_after_remediation_and_normal_canonical_gates | Require current tree with representative coverage substance validation and passing focused launch-gate/miner tests before C-miner evidence feeds v6 or launch DAG. |
| D-action-effect | 3 | H0 M2 L3 | remediated | allowed_for_current_production_consumers_after_remediation_and_normal_canonical_gates | Require current tree or descendant containing b38e800f1; do not treat batch-local exact_nonrate as contest score authority. |
| E-commutator | 0 | H0 M1 L0 | remediated | allowed_for_analysis_only_after_remediation_and_normal_canonical_gates | Do not treat commutator rows as score authority; use only analysis/planning consumption with current measured-row custody checks. |
| F-pose-v5 | 0 | H0 M1 L0 | blocked_launch_evidence | blocked_for_launch_evidence | Produce a new accepted birth/survival/coverage/action-effect ladder with receiver/parseback and exact authority before F-pose-v5 evidence can feed v6 or launch DAG. |

## Codex Takeover Remediation

### C-miner
- Remediated: launch gate now validates representative coverage substance, including `passed`, accepted/diversity thresholds, accepted outcomes, and accepted/all bucket evidence. Scalar-only forged rows block at L4.
- Verification: `uv run python -m pytest src/tac/tests/test_nerv_long_run_launch_gate.py src/tac/tests/test_hinerv_hard_region_miner.py -q` -> `37 passed in 2.57s`.

### E-commutator
- Remediated: measured commutator rows now reject normalization-scope mismatch and partial/mismatched `archive_sha256` / `payload_sha256` before computing `comm`.
- Verification: `uv run python -m pytest src/tac/tests/test_action_commutator.py -q` -> `40 passed in 0.45s`.

### F-pose-v5
- PoseNet teacher plumbing remediated in current code/tests: runner builds a `pose_dims=6` PoseNet teacher and passes it to the target-region birth actuator.
- Launch evidence remains blocked: the v5 SSD artifact records `accepted=false`, `accepted_step_count=0`, zero target support movement, `delta_score_nonrate=0.0`, and no contest CPU/CUDA/full-video exact authority.
- Verification: focused compact-runner F tests pass (`1 passed in 6.97s`; paired source-parity smoke `2 passed in 21.63s`). Whole compact-runner file hit `SystemExit: 143` after 14:13; the isolated failing test passes alone in 15.78s, so that is recorded as a runtime-budget/perf caveat, not F launch clearance.
- Runtime perf caveat remediated: source-parity binding now uses a defensive in-process cache. Verification: source-faithfulness focused tests -> `3 passed in 5.65s`.

## Source Hashes

- Workflow record SHA-256: `e6b8963ec2fe99f779a6535ab794fbde5ce2bfe8984987f569e7a4bf9cf80915`
