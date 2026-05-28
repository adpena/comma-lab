# Codex Findings: Repair Multi-Archive Autonomous Runner

UTC: 2026-05-28T06:00:36Z

## Scope

- `src/tac/optimization/repair_autonomous_multi_archive_runner.py`
- `tools/run_repair_autonomous_multi_archive_runner.py`
- `src/tac/optimization/repair_archive_candidate_intake.py`
- `src/tac/tests/test_repair_autonomous_multi_archive_runner.py`
- `src/tac/tests/test_repair_campaign_materialization_queue.py`
- `.omx/research/repair_multi_archive_autonomous_live_psv3_fec6_20260528T055303Z/`

## Result

The repair loop is now multi-candidate and rerunnable:

1. discover and SHA-dedupe real archive candidates;
2. compile all candidates into one five-family repair work order;
3. score and queue all selected family rows;
4. execute the byte-transform materialization queue locally;
5. build exact-handoff rows;
6. close contest-shaped submission/runtime custody per candidate;
7. rerun the exact-ready bridge with submission dirs;
8. emit the exact-axis blocker and posterior-learning counts.

Live PSV3 + FEC6 run:

- archive candidates: 2
- typed repair rows: 10
- queue-ready experiments: 10
- exact handoff candidates: 10
- archive-bound candidates: 10
- exact-ready bridge candidates: 10
- runtime content tree custody proven: 10
- posterior appended in first loop: 10
- dispatch authority: false
- terminal outcome: `strictly_better_archive_bound_candidate_exact_axis_blocked`

## Bug Class Extinguished

The live rerun exposed an overwrite-custody hole: repair archive intake used
`allow_overwrite=True` without passing expected existing artifact hashes into
the ZIP repack materializer. The materializer correctly refused the overwrite.

Fix:

- archive intake now passes expected existing output/proof SHA-256s on
  overwrite;
- regression added for deterministic rerun over the same intake output dir;
- multi-archive runner test executes two archives end to end and closes 10/10
  runtime custody rows.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/repair_autonomous_multi_archive_runner.py src/tac/optimization/repair_archive_candidate_intake.py src/tac/optimization/repair_family_exact_ready_bridge.py tools/run_repair_autonomous_multi_archive_runner.py tools/build_repair_campaign_work_order_from_archives.py src/tac/tests/test_repair_autonomous_multi_archive_runner.py src/tac/tests/test_repair_campaign_materialization_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_materialization_queue.py src/tac/tests/test_repair_autonomous_multi_archive_runner.py src/tac/tests/test_repair_family_materializers.py::test_repair_exact_ready_bridge_emits_blocked_source_queue -q`
- `.venv/bin/python tools/review_tracker.py policy-check src/tac/optimization/repair_autonomous_multi_archive_runner.py src/tac/optimization/repair_archive_candidate_intake.py tools/run_repair_autonomous_multi_archive_runner.py src/tac/tests/test_repair_autonomous_multi_archive_runner.py src/tac/tests/test_repair_campaign_materialization_queue.py`
- Recursive adversarial review bundle `3ac56fe52b63c414`: clean rounds `26b191f2471e`, `3a43bca4f31c`, `87f9c6105861`; sealed at 3.

## Remaining Exact-Axis Blocker

This is not a score claim. All rows remain blocked until contest CPU/CUDA auth
eval and lane-dispatch claim authority are present.
