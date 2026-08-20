# ddm_cons1 next if resumed

## Immediate State

- Queue harvest of clean `rc=0` FINISHED rows is done: 49 landed marks appended.
- Receipt-less live-marked rows remain: `ddm_et5`, `ddm_fw1`, `ddm_mx1c`.
- `ddm_cons1 -> landed` was appended manually after receipt creation; no `.done`
  keeper receipt was fabricated.
- Serializer boundary commit is blocked by Git object-database writability:
  `unable to create temporary file: Operation not permitted`.
- Signal-loss deterministic sample found 12 routed / 0 unrouted.

## Fire Order

1. Resume in a Git-writable environment and rerun the state boundary commit through
   `tools/subagent_commit_serializer.py` with the post-edit SHA pins recorded in
   `RECEIPT.md`.
2. If the state commit lands, separately review and commit eligible `.omx/research`
   churn through the serializer. Exclude:
   - protected `.omx/research/ddm_cr1_composition_row_827_20260801.md`;
   - bulky/live-output directories such as `.omx/research/ddm_fz3_20260804/sub_final_eval_01/`;
   - research directories containing `.py` unless the required review passes are closed
     without `REVIEW_GATE_OVERRIDE=1`.
3. Rerun `.venv/bin/python tools/consolidation_debt.py --json` and update the monitor table.
4. Do not rerun `ddm_cons1` only to obtain a keeper `.done`; use this receipt as the
   completion artifact.

## Guardrails

- Keep `REVIEW_GATE_OVERRIDE=1` only for non-Python commits.
- Never use direct Git commands to bypass the serializer.
- Do not touch the staged index outside the serializer.
- No scorer or launch work is owned by this consolidation arm.
- Keep pointer honesty: no exact row moved here.
