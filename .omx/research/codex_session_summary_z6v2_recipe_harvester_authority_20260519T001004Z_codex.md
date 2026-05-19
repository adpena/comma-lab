# Codex Session Summary: Z6-v2 Recipe And Harvester Authority - 2026-05-19T00:10:04Z

## Scope

Closed the Z6-v2 Wave 2 mode-route and Modal harvester ledger authority tasks
from
`.omx/research/codex_routing_directive_z6v2_recipe_mode_bug_plus_harvester_ledger_gap_fix_20260518.md`.

## Findings

- ITEM_1 was already materially fixed in the current tree: the Z6-v2 Candidate
  1 recipe declares `Z6_TRAINER_MODE: "full"` and `SMOKE_ONLY: "0"` in
  `env_overrides`; the recipe/driver repair landed in `c10dec618`.
- Catalog #326 is present as the strict driver/recipe mode guard. The live
  driver-mode audit reports 48 drivers scanned and 0 bug-class drivers.
- ITEM_2 was already materially fixed in the current tree: Modal harvesters use
  the canonical call-id outcome helper, and Catalog #330 refuses harvesters that
  call `FunctionCall.from_id(...).get(...)` without a terminal ledger mirror.
- ITEM_3 is verified: the two originally flagged call IDs are terminal in
  `.omx/state/modal_call_id_ledger.jsonl`:
  `fc-01KRSVGE57MT5XSAWCGNQFQPBP` is `harvested`, and
  `fc-01KRSVKF9VEESQY2FS33FF4WDM` is `failed`.
- ITEM_4 backfill pass ran through `tools/harvest_modal_calls.py --from-ledger
  --execute --get-timeout-seconds 2.0`. The canonical ledger reported 132 total
  call IDs and 0 unharvested call IDs before execute; execute added structured
  supplemental terminal evidence for recovered Modal calls without score or
  promotion authority.
- Pascal's read-only audit found a remaining self-protection gap: the Catalog
  #326 helper could under-classify multi-key mode drivers if a dispatchable
  recipe declared neither `Z6_TRAINER_MODE` nor `SMOKE_ONLY`. The helper now
  extracts concrete mode env vars, detects branch defaults such as
  `elif [ -z "$SMOKE_ONLY" ]; then SMOKE_ONLY="1"`, and treats unsafe unknown
  defaults as a bug-class verdict.

## Verification

```bash
.venv/bin/python tools/harvest_modal_calls.py --from-ledger
.venv/bin/python tools/audit_substrate_driver_mode_hardcode.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_check_326_substrate_driver_consumes_trainer_mode_env_var.py \
  src/tac/tests/test_check_330_modal_harvester_call_id_ledger_outcome.py \
  src/tac/tests/test_z6_v2_candidate_1_wave_2_build.py
.venv/bin/python tools/harvest_modal_calls.py --from-ledger --execute --get-timeout-seconds 2.0
```

Result: 75 focused tests passed; Catalog #326 audit bug-class count is 0; Modal
call-id ledger latest view has 0 unharvested IDs after the backfill pass.

Additional self-protection verification after the multi-key classifier fix:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_check_326_substrate_driver_consumes_trainer_mode_env_var.py \
  src/tac/tests/test_check_330_modal_harvester_call_id_ledger_outcome.py \
  src/tac/tests/test_z6_v2_candidate_1_wave_2_build.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check \
  tools/audit_substrate_driver_mode_hardcode.py \
  src/tac/preflight.py \
  tools/harvest_modal_calls.py \
  src/tac/tests/test_check_326_substrate_driver_consumes_trainer_mode_env_var.py \
  src/tac/tests/test_check_330_modal_harvester_call_id_ledger_outcome.py \
  src/tac/tests/test_z6_v2_candidate_1_wave_2_build.py
```

Result: 78 focused tests passed; Ruff passed on touched gate/harvester/test
surfaces.

## Authority

This closure is infrastructure authority, not scorer authority. The new rows
are `score_claim=false`, `promotion_eligible=false`, and
`rank_or_kill_eligible=false` evidence for no-signal-loss terminal coverage.
Any archive or auth-eval artifact from these harvested calls still requires a
separate exact result-review packet before promotion, ranking, or retirement.
