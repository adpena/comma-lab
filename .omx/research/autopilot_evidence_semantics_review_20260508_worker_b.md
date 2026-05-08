# Autopilot Evidence Semantics Review - Worker B - 2026-05-08

## Scope

Reviewed the planner/evidence semantics around:

- `tools/cathedral_autopilot.py`
- `src/tac/tests/test_cathedral_autopilot_proxy_guards.py`
- `reports/cathedral_autopilot_evidence.jsonl` read-only
- `experiments/lossy_coarsening_lightning_harvest.py` read-only as the
  evidence-update producer for the new lossy-coarsening CUDA result

Main branch remained the source of truth. The evidence JSONL was not edited.

## Findings

1. Non-promotable CPU/MPS/proxy/byte-only rows were marked planning-only, but
   still replaced `predicted_archive_bytes` and could rank by score delta.
   This allowed byte-only anchors to dominate recommendations even though
   they had `score_claim=false`, `promotion_eligible=false`,
   `rank_or_kill_eligible=false`, or dispatch blockers.

2. Exact-negative catalog rows had a live first-pass guard in the working tree,
   but `lossy_coarsening_analytical` is not currently in either autopilot
   catalog. Its 2026-05-08 exact CUDA A-negative row was therefore silently
   ignored by catalog update/ranking paths.

3. The real evidence feed currently has 35 rows. A local dry run of
   `cathedral_autopilot.py evidence-update` reported 16 unknown technique
   names and 1 unknown exact-negative row. The unknown exact-negative row is
   `lossy_coarsening_analytical`.

4. The lossy-coarsening harvester still emits an initial generic
   `[contest-CUDA]` row with `contest_dispatch_verdict="completed"`. The
   scoped A-negative semantics only become machine-readable after the later
   review/evidence row adds `[contest-CUDA A-negative]`,
   `measured_config_status="measured_config_retired"`,
   `family_falsified=false`, `method_family_retired=false`, and the review
   packet path. Hardening that producer is outside this worker's write scope.

## Patch

- Added `active_ranking_blocked` semantics to `tools/cathedral_autopilot.py`.
  Non-promotable empirical anchors still preserve their byte measurements and
  audit sources, but `_rank_techniques` assigns them zero active score-delta
  and sorts them behind unblocked rows.
- Exact-negative rows now set an explicit
  `active_ranking_block_reason="exact_negative_measured_config_requires_reactivation"`.
- Added `evidence_semantics_report` to plan/evidence-update output. Unknown
  technique rows are now visible as `unknown_technique_not_ranked`; unknown
  exact-negative rows are counted separately.
- Plan notes now call out unknown technique rows, blocked ranking rows, and
  cataloged exact-negative rows.
- Added focused tests proving byte-only rows cannot dominate active ranking
  and unknown evidence rows are reported but not ranked.

## Remaining Gaps

- `lossy_coarsening_analytical` needs an explicit catalog row or a reviewed
  alias/parent-technique map if it should participate in the typed planner.
  Until then it is reported but not ranked.
- Variant rows such as `lossy_int4_quantization_gptq`,
  `lossy_int4_quantization_awq`, and cross-paradigm stack rows are still
  unknown to the autopilot catalog. The new report makes this visible, but it
  does not solve parent/variant aggregation.
- `tools/check_evidence_implementation_matches_model_spec.py` still treats
  unknown technique names as non-model-spec findings and skips them. A
  separate guard should require either catalog coverage or an explicit
  `planning_only_unknown_technique` disposition for rows appended to the
  shared evidence feed.
- `experiments/lossy_coarsening_lightning_harvest.py` should eventually emit
  the measured-config status and review-packet linkage directly when it can
  classify a CUDA result as negative; today that classification is a later
  row, which creates a short-lived ambiguous exact-CUDA record.

## Verification

- `.venv/bin/python -m py_compile tools/cathedral_autopilot.py src/tac/tests/test_cathedral_autopilot_proxy_guards.py`
- `.venv/bin/python -m ruff check tools/cathedral_autopilot.py src/tac/tests/test_cathedral_autopilot_proxy_guards.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_cathedral_autopilot_proxy_guards.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_cathedral_autopilot.py`
- `.venv/bin/python tools/cathedral_autopilot.py evidence-update --prior-evidence reports/cathedral_autopilot_evidence.jsonl --output /tmp/cathedral_autopilot_evidence_update_worker_b.json`
