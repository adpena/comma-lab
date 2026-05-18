# Codex routing directive: Z6-v2 Wave 2 recipe mode-misroute bug + Modal harvester ledger-write gap
# Date: 2026-05-18
# Empirical anchors landed THIS session via Modal harvest investigation
# Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable — every fix MUST land with sister STRICT preflight gate

## CANONICAL POINTERS (read FIRST)

1. `/Users/adpena/Projects/pact/CLAUDE.md` (full; honor NON-NEGOTIABLE markers — especially "Bugs must be permanently fixed AND self-protected against")
2. `/Users/adpena/Projects/pact/AGENTS.md` (full)
3. `.omx/state/modal_call_id_ledger.jsonl` (Bug 2 empirical receipts — Z6-v2 candidate 1 dispatches `fc-01KRW7RHFHP640BHTQ0FZM3M38` + `fc-01KRW7ZCYK5XF6MSHD24R71A46` both ran smoke=1 despite operator-authorized FULL)
4. `scripts/remote_lane_substrate_time_traveler_l5_z6.sh` (Z6 driver; correctly emits WARN per Catalog #326 — NOT the bug)
5. `.omx/operator_authorize_recipes/substrate_z6_v2_candidate_1_multi_layer_film_modal_t4_smoke_dispatch.yaml` (Bug 1 source — recipe doesn't set Z6_TRAINER_MODE)
6. `tools/harvest_modal_calls.py` (Bug 2 source — pulls artifacts but doesn't register canonical ledger event)
7. `src/tac/deploy/modal/call_id_ledger.py` (canonical helper API; `update_call_id_outcome` for `harvested` event)

## EMPIRICAL ANCHORS (THIS session 2026-05-18)

### Bug 1: Z6-v2 Wave 2 recipe mode-misroute

`fc-01KRW7RHFHP640BHTQ0FZM3M38` + `fc-01KRW7ZCYK5XF6MSHD24R71A46` BOTH harvested rc=0 / ~10s / `smoke=1 identity_predictor=false score_claim=false`. Operator authorized FULL canary (per active_lane_dispatch_claims.md `2026-05-18T00:32:06Z operator:...:full_canary`). Driver correctly emitted WARN per Catalog #326. **The recipe `env_overrides` block doesn't set `Z6_TRAINER_MODE: "full"` so driver defaults to smoke** — Catalog #326 fired its WARN but didn't refuse dispatch.

### Bug 2: Modal harvester ledger-write gap

Sister Modal harvest investigation found 2 originally-flagged unharvested call_ids (`fc-01KRSVGE57MT5XSAWCGNQFQPBP` NSCS06 v8 path b rc=0 / 20 min / 17 artifacts; `fc-01KRSVKF9VEESQY2FS33FF4WDM` STC v2 rc=25 / 1.56s). Both successfully pulled via `modal.functions.FunctionCall.from_id(cid).get()`. Then `tools/harvest_modal_calls.py` was run — it harvested MANY OTHER call_ids but **did not register `harvested`/`failed` events in `.omx/state/modal_call_id_ledger.jsonl` for the canonical ledger**. Main-Claude had to manually call `tac.deploy.modal.call_id_ledger.update_call_id_outcome(...)` to close the ledger gap.

## ITEM 1 — Fix Bug 1: Z6-v2 Wave 2 recipe MUST declare mode explicitly

### Root cause

Recipe `substrate_z6_v2_candidate_1_multi_layer_film_modal_t4_smoke_dispatch.yaml` `env_overrides` block sets Z6_LANE_ID / Z6_RECIPE_PATH / TAG / Z6_DISPATCH_INSTANCE_JOB_ID but NOT `Z6_TRAINER_MODE` or `SMOKE_ONLY`. Driver per Catalog #326 defaults to `SMOKE_ONLY=1` with WARN. Per Catalog #326 the WARN is correct behavior; the recipe is the bug.

### Fix

EITHER (a) add `Z6_TRAINER_MODE: "full"` to the recipe's `env_overrides` block (the recipe's intent is FULL canary per filename containing `t4_smoke_dispatch` is misleading — Wave 2 is the FULL training canary post-smoke-validation); OR (b) split into 2 recipes: `*_smoke_dispatch.yaml` (Z6_TRAINER_MODE=smoke) + `*_full_dispatch.yaml` (Z6_TRAINER_MODE=full). Option (b) is cleaner per HNeRV parity L4 (operator-readable in 30 sec).

### STRICT preflight gate sister (Catalog #326 extension)

Current Catalog #326 catches drivers that hardcode `--smoke`. EXTEND to catch recipes that:
- Have `dispatch_enabled: true` (or absent which defaults true)
- Use a `lane_script:` that supports the `Z6_TRAINER_MODE` / `SMOKE_ONLY` env var
- Don't set EITHER env var in their `env_overrides` block

Such recipes would silently default to smoke. STRICT gate refuses unless waived with `# RECIPE_MODE_UNDECLARED_OK:<reason>`.

## ITEM 2 — Fix Bug 2: harvester MUST register canonical ledger events

### Root cause

`tools/harvest_modal_calls.py` writes per-dispatch `_harvest_summary.json` files locally but doesn't call `tac.deploy.modal.call_id_ledger.update_call_id_outcome(...)` to register `harvested`/`failed`/`stale` events in the canonical ledger. The canonical Catalog #245 4-layer pattern requires:

1. Canonical helper (`tac.deploy.modal.call_id_ledger`) ✓ exists
2. CLI tool (`tools/harvest_modal_calls.py`) ✗ doesn't USE the canonical helper for events
3. STRICT preflight gate ✗ doesn't enforce harvester-must-write-ledger
4. Operator bypass discipline ✗ N/A

### Fix

`tools/harvest_modal_calls.py` MUST call `update_call_id_outcome()` for EVERY successfully-harvested call_id immediately after `_write_harvest_payload()` returns successfully. Event_type: `harvested` (rc=0) / `failed` (rc≠0) / `stale` (OutputExpiredError).

```python
from tac.deploy.modal.call_id_ledger import update_call_id_outcome

# After successful artifact write:
update_call_id_outcome(
    call_id=cid,
    status="harvested" if rc == 0 else "failed",
    rc=rc,
    elapsed_seconds=elapsed,
    cost_actual_usd=None,  # if available
    archive_sha256=archive_sha256_from_artifacts,
    archive_bytes=archive_bytes_from_artifacts,
    evidence_grade="advisory",  # or extract per result schema
    extra={"event_type": "harvested" if rc == 0 else "failed", "notes": "harvested by tools/harvest_modal_calls.py", "harvest_summary_path": str(summary_path)},
)
```

### STRICT preflight gate sister (NEW Catalog #)

Claim next available Catalog # via `tools/claim_catalog_number.py claim --commit-via-serializer --reason "harvester writes canonical ledger event"`. STRICT gate refuses any `tools/harvest_modal_*.py` (or sister harvester) that:
- Calls `FunctionCall.from_id(...).get()` (or equivalent)
- Does NOT call `update_call_id_outcome(...)` within ±20 lines of the get() call

Acceptance:
- Same-line waiver `# HARVESTER_LEDGER_WRITE_OK:<rationale>` for diagnostic-only harvesters
- File-level waiver `# HARVESTER_LEDGER_WRITE_OK_FILE:<rationale>` for the canonical helper itself (which IS the writer)

## ITEM 3 — Backfill the 2 originally-flagged ledger events (done in-context this session; verify)

Main-Claude manually called `update_call_id_outcome()` for:
- `fc-01KRSVGE57MT5XSAWCGNQFQPBP` → status='harvested', rc=0, elapsed=1183.31s
- `fc-01KRSVKF9VEESQY2FS33FF4WDM` → status='failed', rc=25, elapsed=1.56s

Verify in ledger; no further action needed unless audit surfaces other gaps.

## ITEM 4 — Audit for OTHER ledger gaps

After ITEM 2 fix lands, run `tools/harvest_modal_calls.py` once more (idempotent per `_already_harvested` check) to retroactively register events for any other call_ids whose artifacts exist locally but no ledger event was written. The harvester's idempotency means safe to re-run.

## DISCIPLINE

All standard discipline (Catalog #117/#157/#174 commit serializer + POST-EDIT sha; #186 catalog-claim transactionality; #206 checkpoint every ~10 tool uses; #229 premise verification; #287 [empirical:<path>] tags; #305 observability surface; #314 declare files_touched).

## SISTER SUBAGENT COORDINATION

In-flight (3-cap saturated):
- `a39ffdf80` Riemannian-Newton substrate engineering design
- `a478cbde` TT5L V2 redesign
- `ae324eabee` Cargo-cult resurrection TOP-3 symposiums

DISJOINT scope (these directive items touch `.omx/operator_authorize_recipes/substrate_z6_*.yaml` + `tools/harvest_modal_calls.py` + new STRICT preflight gate; sister subagents don't touch these files).

## CODEX EXECUTION ORDER

1. ITEM 1 Bug 1 fix (recipe `Z6_TRAINER_MODE` addition OR split) + tests
2. ITEM 1 STRICT preflight gate Catalog #326 extension + tests
3. ITEM 2 Bug 2 fix (harvester update_call_id_outcome wire-in) + tests
4. ITEM 2 STRICT preflight gate NEW Catalog # + tests + CLAUDE.md row
5. ITEM 4 re-run harvester to backfill any other ledger gaps

— Main-Claude (relayed on behalf of operator 2026-05-18)
