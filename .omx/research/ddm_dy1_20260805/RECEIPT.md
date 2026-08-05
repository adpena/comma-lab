# ddm_dy1 scope-law resolver receipt

DY1R NOTE 2026-08-05: this is the historical isolated-branch receipt from
`ddm/dy1_scope_law_resolver`. The current landed merge authority is
`.omx/research/ddm_dy1r_20260805/RECEIPT.md`; source hashes below refer to the
isolated dy1 branch state before dy2/dy1r reconciliation.

## Status

Build-only apparatus work landed in isolated clone fallback:

- Clone: `/Users/adpena/Projects/pact/.omx/tmp/codex_worktrees/ddm_dy1_scope_law_resolver_20260805T2058Z`
- Branch: `ddm/dy1_scope_law_resolver`
- Worktree add attempt failed before edits: `fatal: cannot lock ref ... Operation not permitted`
- Scorer/MLX/training/archive boundary: no scorer run, no Metal/MLX launch, no archive produced, no pointer claim.
- Main protected files remained untouched: `.omx/research/ddm_cr1_composition_row_827_20260801.md`, `.omx/research/ddm_pu2_pose_tail_floor_probe_20260803.md`, `src/tac/optimization/direct_description_carrier_compose.py`.

## Recall Evidence

Read before edits:

- Charter: `/Users/adpena/Projects/pact/.omx/tmp/codex_runs/dy1_prompt.md`
- Common contract: `/Users/adpena/Projects/pact/.omx/tmp/codex_runs/_common_contract.md`
- `PROGRAM.md`
- `CLAUDE.md` and `AGENTS.md` parity checked with `cmp -s CLAUDE.md AGENTS.md`
- `docs/operating_manual_craft_handoff.md`
- `.omx/state/main_hot_state.md`
- `.omx/research/ddm_cx1_20260805/RECEIPT.md`
- `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/cross_regime_constant_transfer_genus_finishing_stage_20260805.md`

Recall queries included:

- `rg -n "jd1_|stage_ema|realized_hold|LawRef|GuardedConstant|adaptive_eps|governance knobs|constants are poison" experiments src/tac tools .omx/research`
- `rg -n "scope law|ScopeLaw|lawref|LawRef|ticket_hash|sealed_ticket|trainer_declared_flags" src/tac/witness_dsl experiments tools`
- `tools/list_canonical_equations.py --json` attempted from the clone, then registry surfaces were inspected through the shared main `.venv`.

Plan changes from recall:

- Did not add a parallel ad hoc constant mechanism. Reused the existing `LawRef`/ticket DSL pattern and added a runtime-adjacent `scope_laws` surface.
- Did not treat JD3 constants as static LawRefs, because their valid inputs exist only at stage/window/gate scope.
- Added explicit inertness alarms so declared dynamic laws cannot sit in a ticket without runtime resolution evidence.

## Implemented Surface

New resolver:

- `src/tac/witness_dsl/scope_laws.py`
  - Defines `ScopeLaw`, `ScopeLawResolution`, typed tiers `T2_SCOPE_LAW` and `T3_LIVE_ADAPTED`.
  - Produces deterministic canonical-json `resolution_hash` rows suitable for telemetry/checkpoint metadata.
  - Exposes `validate_ticket_scope_laws`, `ticket_payload_hash`, `jd3_default_scope_law_refs`, and inertness positive-control checks.
  - Registers five JD3 laws:
    - `jd3_stage_ema_decay`
    - `jd3_realized_hold_margin`
    - `jd3_realized_hold_floor_latch`
    - `jd3_pose_retreat_bisection`
    - `jd3_max_retreats_a1_policy`

Trainer integration:

- `experiments/train_tr1_partition_renderer_mlx.py`
  - Existing JD1/JD3 derivation helpers now resolve through `scope_laws`.
  - Runtime `_resolve_scope_law` appends rows and emits `scope_law_resolution` telemetry.
  - Resolved rows are restored from checkpoint metadata and written into `jd1_pose_finish` receipt/checkpoint metadata.
  - Stage EMA entry logs and persists `stage_ema_scope_law_resolution_hash`.
  - First realized-hold latch resolves/logs floor and margin law hashes.
  - Pose-retreat and max-retreat laws are resolved after checkpoint resume restoration, avoiding pre-resume duplicate rows.

Ticket/launcher integration:

- `src/tac/witness_dsl/spec_tr1_renderer_20260728.py`
  - Optional `scope_laws` tuple on `TR1RendererProgramV1`.
  - Tickets with laws validate their schema and hash the law declarations.
  - Legacy tickets without laws keep their old argv shape and omit `scope_laws`.
- `tools/launch_tr1_run.py`
  - Refuses malformed `scope_laws`.
  - Recomputes `ticket_hash` when scope laws are present and refuses if the hash does not bind them.
- `experiments/ddm_jd1_ticket_regenerate.py`
  - JD3 v3 ticket regeneration attaches default scope-law declarations and recomputes the bound ticket hash.

Tests:

- `src/tac/witness_dsl/tests/test_scope_laws.py`
  - Determinism and hash stability.
  - JSON/resume-safe resolution rows.
  - Inertness positive-control failure for declared-but-unresolved laws.
  - Ticket hash changes when a declared law is omitted.
  - TR1 sealed-ticket argv remains unchanged when scope laws are added.

## Migration Table

| Dynamic value / decision | Tier | Scope-law / disposition | Status |
|---|---:|---|---|
| JD3 stage EMA decay from remaining stage window | T2 | `jd3_stage_ema_decay` | built |
| First realized-hold margin from gate sd / sqrt(n) | T3 | `jd3_realized_hold_margin` | built |
| First realized-hold floor latch from realized gate mean | T3 | `jd3_realized_hold_floor_latch` | built |
| Pose-pressure retreat bisection fallback | T3 | `jd3_pose_retreat_bisection` | built |
| Max retreats from A1 consecutive-refuse policy | T2 | `jd3_max_retreats_a1_policy` | built |
| Lane guard ratchet horizon | T2 | queue next scope law after v3 smoke | queued |
| Deterministic R decision surface | T2/T3 | queue runtime authority gate law, not static constant | queued |
| JD1 `w_pose` live lower/retreat arm | T3 | queue controller law after v3 smoke rows exist | queued |
| EN1 margin-weight steering | T3 | queue after JD3 v3 smoke evidence | queued |
| SL2 teacher / PE3 conditioning switches | T2/T3 | queue after v3 smoke evidence | queued |
| Governance restart / averaging / selection / gradient composition | T2/T3 | queue fire-order from m51, not solved here | queued |
| #847 NONE/no-default knobs | T2 | queue GuardedConstant-to-scope-law migration | queued |

## Hashes

Charters read from main worktree:

| Path | SHA-256 |
|---|---|
| `.omx/tmp/codex_runs/dy1_prompt.md` | `d25b4bf918e722e34f77abe31bca568e74832d5424fcdceeb479c556cc815488` |
| `.omx/tmp/codex_runs/_common_contract.md` | `eeae9e0035582e6bdd65fd837e4aa35a65e064fd09900b9c212d41ac02086771` |

Post-edit source hashes:

| Path | SHA-256 |
|---|---|
| `src/tac/witness_dsl/scope_laws.py` | `b4e023bcd2faa618c9a254501746b0127e2ababa3a7ca435b852ff5db117a4c8` |
| `src/tac/witness_dsl/tests/test_scope_laws.py` | `a417b9d64b415d07dcac6a5dcb5f4c28c3a1cd4acfc13b002758c8134bc12ee4` |
| `src/tac/witness_dsl/spec_tr1_renderer_20260728.py` | `e96fff8afd9368eec8ef6d125daf690861424b8602f52c3189b3c1a17b864668` |
| `experiments/train_tr1_partition_renderer_mlx.py` | `0d3ae955067b2dca21e7080254f79deba1c5424ec29e823b85b1d41fcea15153` |
| `experiments/ddm_jd1_ticket_regenerate.py` | `60f942aadee995ae47e2c0bc698aef1ba2c9170fdebf8b178898968f1958b10d` |
| `tools/launch_tr1_run.py` | `d56d17db0d0f7d21b80c89c8f3954b1521145d89ee523b15a1e007f9f52eccdd` |

## Verification

Commands run from the isolated clone unless noted:

```bash
python3 -m py_compile \
  src/tac/witness_dsl/scope_laws.py \
  src/tac/witness_dsl/spec_tr1_renderer_20260728.py \
  experiments/train_tr1_partition_renderer_mlx.py \
  experiments/ddm_jd1_ticket_regenerate.py \
  tools/launch_tr1_run.py \
  src/tac/witness_dsl/tests/test_scope_laws.py
git diff --check
```

Result: passed.

```bash
PYTHONPATH=$PWD/src:$PWD /Users/adpena/Projects/pact/.venv/bin/python \
  -m pytest -q src/tac/witness_dsl/tests/test_scope_laws.py \
  src/tac/tests/test_ddm_bp1_boundary_reset_race.py -k 'scope_law or jd3'
```

Result: `9 passed, 35 deselected`. MLX emitted an atexit Metal-device warning after test completion, but no MLX run was launched and the test process returned 0.

```bash
PYTHONPATH=$PWD/src:$PWD /Users/adpena/Projects/pact/.venv/bin/python \
  -m ruff check src/tac/witness_dsl/scope_laws.py \
  src/tac/witness_dsl/tests/test_scope_laws.py \
  src/tac/witness_dsl/spec_tr1_renderer_20260728.py \
  --select F,I,RUF022,UP037
```

Result: passed.

```bash
PYTHONPATH=$PWD/src:$PWD /Users/adpena/Projects/pact/.venv/bin/python \
  experiments/ddm_jd1_ticket_regenerate.py --v3 \
  --start-candidate entry_ep1336 \
  --out-ticket .omx/tmp/dy1_ticket_smoke.json
```

Result: generated ticket hash `01d43d87eda38e5805948cd80e9a6292ccdbb4b91282639bea2cb7c5ba8727c8`; verified all five expected `scope_laws` names; smoke artifact removed.

Full modified-file Ruff is not claimed clean. It still reports legacy style findings in the large trainer/launcher/ticket-regenerator files, including existing UP037/SIM/I001/B007/RUF005/RUF046/RUF100 debt. The focused new-surface check above is clean.

Review gate:

```bash
PYTHONPATH=$PWD/src:$PWD /Users/adpena/Projects/pact/.venv/bin/python \
  tools/review_tracker.py mark-file <each touched .py file> --status reviewed
```

Result: two passes completed for all six touched Python files. First pass ingested the new files with an incremental scan; second pass re-marked the same files without tracked review-state changes.

## Boundaries

- MEASURED: deterministic resolver behavior and ticket hash binding through unit tests.
- MEASURED: JD3 v3 ticket regeneration includes the five scope-law declarations and hash binds them.
- NOT MEASURED: scorer output, archive bytes from this work, contest CPU/CUDA score, or MLX training behavior.
- NOT CLAIMED: any improvement to the contest pointer.
- Own-vehicle line: this is apparatus/config-resolution work only, intended to make future JD3/v3 launches dynamically lawful and auditable.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
