# Codex Findings: RATE Autopilot Authority Surface

Timestamp: 2026-05-19T00:36:47Z
Actor: codex
Task: `rate_attack_process_ach_assumption_autopilot_features`
Lane: `lane_rate_attack_process_ach_assumption_autopilot_features_20260519`

## Authority Answer

Authority is established by a custody chain, not by prose:

1. Canonical rule/pointer: the originating design memo names the task, canonical consumer, and acceptance checks.
2. Structured source state: `canonical_task_status.jsonl` records task ownership and completion.
3. Reusable implementation: `tac.contest_exploits.rate_attack_autopilot_features` builds typed rows.
4. Canonical consumer: `tools/cathedral_autopilot_autonomous_loop.py` has a dedicated loader for the feature matrix.
5. Fail-closed semantics: score, promotion, rank/kill, and exact-dispatch authority remain false.
6. Verification artifact: `reports/rate_attack_autopilot_feature_matrix_20260519T003518Z.json` plus the JSONL sidecar feed the autopilot without reading prose.

## What Landed

- New stdlib-only feature builder:
  - `src/tac/contest_exploits/rate_attack_autopilot_features.py`
  - `tools/build_rate_attack_autopilot_feature_matrix.py`
- Cathedral consumer hook:
  - `--rate-attack-feature-matrix-json`
  - `load_candidates_from_rate_attack_feature_matrix_output(...)`
- Focused tests:
  - `src/tac/contest_exploits/tests/test_rate_attack_autopilot_features.py`
- Durable artifacts:
  - `reports/rate_attack_autopilot_feature_matrix_20260519T003518Z.json`
  - `reports/rate_attack_autopilot_feature_matrix_20260519T003518Z_cathedral_autopilot_candidates.jsonl`
  - `reports/rate_attack_autopilot_feature_matrix_20260519T003518Z_autopilot_report.json`

## Artifact Summary

- Candidate rows: 9
- Source OPs: OP1 stable-orbit packet diet, OP2 tropical argmax boundary grammar, OP3 decoy mosaic residual basis
- Disconfirming-assumption coverage: 9/9
- Missing disconfirming-assumption rows: 0
- Score-claim rows: 0
- Promotion-eligible rows: 0
- Greater-than-1-dollar spend blocked by missing cheap probe: 0
- Aggregate ACH inconsistency count: 3
- Aggregate key-assumption risk count: 103

## Verification

Commands run:

```text
.venv/bin/python -m pytest src/tac/contest_exploits/tests/test_rate_attack_autopilot_features.py
.venv/bin/ruff check src/tac/contest_exploits/rate_attack_autopilot_features.py src/tac/contest_exploits/tests/test_rate_attack_autopilot_features.py tools/build_rate_attack_autopilot_feature_matrix.py tools/cathedral_autopilot_autonomous_loop.py
.venv/bin/python tools/build_rate_attack_autopilot_feature_matrix.py
.venv/bin/python tools/cathedral_autopilot_autonomous_loop.py --rate-attack-feature-matrix-json reports/rate_attack_autopilot_feature_matrix_20260519T003518Z.json --report-only --report-top-n 3 --output reports/rate_attack_autopilot_feature_matrix_20260519T003518Z_autopilot_report.json
```

Results:

- Pytest: 4 passed
- Ruff: clean after import-order fix
- Autopilot report-only: loaded and ranked 9 candidates; top 3 all preserve `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`.

## Adversarial Follow-Up Queue

Read-only xhigh adversarial review surfaced these next authority leaks:

1. P1 runtime payload custody can miss uncharged data under `src/`.
2. P1 raw-axis-only difficulty atlases can still receive Cathedral rank reward.
3. P1 Wyner-Ziv hoist can treat gradient serialization compression as archive byte savings.
4. P2 B1 advisory saliency can look like real score-gradient saliency.
5. P2 OP-SYN projector registry is looser than actual projector boundaries.
6. P2 OP1 may be over-conservative by requiring two gradient archives for same-archive planning.
7. P2 live optimal-plan CandidateRow can drop explicit planning-only blocker.
8. P2 F1/B1 should remain probe-gated until the reconciliation memo conditions are satisfied.

These are not fixed by this landing. They are preserved here as the next authority-hardening queue.

## Partner WIP Absorption

Observed stable partner WIP and absorbed it after cheap checks:

- `src/tac/master_gradient_wire_in.py`
- `src/tac/master_gradient_archive_parsers.py`
- `src/tac/tests/test_master_gradient_wire_in.py`
- `tools/build_top_1_a1_local_cpu_advisory_smoke.py`
- `tools/build_top_2_f4_summary_local_cpu_advisory_smoke.py`
- `tools/build_top_3_f5_resblock_local_cpu_advisory_smoke.py`
- partner lane-registry rows for TOP-3 local CPU and master-gradient wire-in

Realtime churn was observed first, so staging was deferred until mtimes stayed stable. After stabilization, the combined focused suite passed `py_compile`, Ruff, and 41 pytest cases across RATE features, master-gradient wire-in, and master-gradient archive parser facade tests. `tools/audit_master_gradient_wire_in_coverage.py --summary` reports 13 surfaces, 10 wired, 3 unwired, and a 47.0% -> 76.9% surface-coverage improvement; score-claim authority remains false.
