# Codex Findings - Cluster B Retroactive Sweep Gate - 2026-05-19T20:30Z

Author: Codex
Task: `codex_routing_directive_session_20260519_max_score_lowering_batch_BCEF_20260519T051028Z::CLUSTER_B`
Catalog: #348
Lane: `lane_cluster_b_retroactive_sweep_gate_catalog_348_20260519`

## What Landed

- Added `check_new_gate_landing_includes_retroactive_sweep_evidence` in
  `src/tac/preflight.py`.
- Wired it into `preflight_all()` as WARN-only, per strict-flip atomicity.
- Added focused tests in
  `src/tac/tests/test_check_348_retroactive_sweep_gate.py`.
- Added the durable Catalog #348 row to `CLAUDE.md`.
- Added the first sweep memo:
  `.omx/research/retroactive_sweep_for_catalog_348_20260519T202900Z.md`.

## Behavior

The gate scans recent git commits for newly added `def check_*` functions in
`src/tac/preflight.py`. Each new gate must resolve to a `CLAUDE.md` catalog row
and must have a matching
`.omx/research/retroactive_sweep_for_catalog_<N>_<utc>.md` memo containing:

1. bug-class symptom signature
2. pre-fix window
3. historical-KILL/DEFER/FALSIFY search results
4. per-finding RE-EVAL-priority assignment

Same-line `# RETROACTIVE_SWEEP_WAIVED:<rationale>` is accepted on the added
function definition only when the rationale is substantive and non-placeholder.

## Live Warning Baseline

The live WARN-only scan over the latest 20 commits currently reports 17
backfill gaps after the adversarial-review fix that expanded the scanner from
commit-only one-line defs to committed + staged + worktree multi-line
signatures:

- `check_no_unwaived_pyppmd_imports`
- `check_substrate_trainers_use_canonical_optimization_helpers`
- recent Catalog rows including #340, #343, #344, #346 that predate this
  evidence convention and need retroactive sweep memos or reviewed waivers

Both were intentionally left as warnings, not blockers. Strict flip should wait
for either catalog rows plus sweep memos or reviewed waivers for those recent
gate additions.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_check_348_retroactive_sweep_gate.py -q`
  - `13 passed`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check src/tac/preflight.py src/tac/tests/test_check_348_retroactive_sweep_gate.py`
  - passed
- Live WARN-only smoke:
  - `recent_commits=20`
  - `violations=17`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/canonical_task_status.py --validate`
  - valid
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/lane_maturity.py validate`
  - valid

## Authority

No score claim, promotion claim, rank claim, or dispatch claim. This is a
frontier-protecting control-plane gate that prevents stale verdicts from
silently surviving future bug-class extinctions.
