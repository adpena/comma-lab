# Boundary merge-queue resolver — mechanical instructions (rehearsal 20260717)

Merge order (boundary): p0_518 -> p0_328_408 -> p0_521 -> l7.

## Merges 1-3: CLEAN (ort auto-merge, zero conflicts)
- p0_518_resume_warmup_geometry: auto-merged tools/launch_witness_run.py; 8 files, no conflict.
- p0_328_408_merge_window_prep: auto-merged BOTH experiments/train_levelset_witness_realized_through_R_mlx.py
  AND tools/levelset_byte_close_and_eval.py (the warned proximity zones) — ort handled them, no conflict.
- p0_521_spec_v10_capstone: new files only, no conflict.

## Merge 4 (l7): 2 ledger conflicts ONLY — pure-additive, resolve by UNION
Trainer + all tests auto-merged CLEAN. The ONLY conflicts are two append/registry ledgers:

1. `.omx/state/lane_maturity_audit.log` (append-only JSONL):
   UNION rows — keep ALL of HEAD's appended rows AND l7's 1 row (l7_default_failloud_budget_eventlaw,
   ts 2026-07-15T15:24:58Z). Never drop either side. Every line stays valid JSON.
   l7's row is in `l7_audit_row_ADD.jsonl`.

2. `.omx/state/lane_registry.json` (structured lanes array):
   UNION lanes BY id — keep HEAD's lanes (current-main truth, incl. its 8 new lanes) and APPEND
   l7's 1 new lane. No lane is modified on both sides (both branches are purely additive vs base:
   base=1889, ours=1897 (+8), theirs=1890 (+1); intersection=1889=base). Keep HEAD top-level
   metadata (generated_at/from_state_hash/updated_at). Result = 1898 unique lanes.
   l7's lane object is in `l7_registry_lane_ADD.json`.

Validate after: `.venv/bin/python tools/lane_maturity.py validate` (must print OK).

## Real-merge note
Because both ledger conflicts are pure-additive unions, the real merge is mechanical: run the
same union. The exact bytes will differ at real-merge time (main advances), but the METHOD is
stable and the l7-only ADD pieces are frozen here. Commit the real merge THROUGH the pre-commit
hook (the rehearsal used --no-verify only because the scratch worktree has no .venv).
