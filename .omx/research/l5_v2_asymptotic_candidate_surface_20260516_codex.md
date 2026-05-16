# L5 v2 asymptotic candidate surface

**Date:** 2026-05-16
**Agent:** Codex
**Scope:** L5 v2 staircase / Cathedral autopilot / operator briefing
**Score authority:** none. Planning-only, `score_claim=false`,
`promotion_eligible=false`, `rank_or_kill_eligible=false`,
`ready_for_exact_eval_dispatch=false`, `ready_for_paid_dispatch=false`.
**Structured artifact:** `.omx/research/l5_v2_asymptotic_candidate_surface_20260516_codex.json`
(`sha256=03e513a0fc90f3aad94c24a12d435d11874138f8c0bec962a76593c4cae7b4b0`).

## Why this landed

The L5 v2 frontier had three new asymptotic-pursuit design ledgers on `main`:

- Z6/Z7/Z8 predictive-coding world-models:
  `.omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md`
- Rudin floor interpretable-ML substrate:
  `.omx/research/rudin_floor_interpretable_ml_substrate_asymptotic_pursuit_scoping_design_20260516.md`
- Tishby IB-pure primary Lagrangian substrate:
  `.omx/research/tishby_ib_pure_substrate_asymptotic_pursuit_scoping_design_20260516.md`

Those memos were durable signal, but the executable L5 v2 readiness surface did
not yet expose them as next-action candidates. That made them easy to lose
behind TT5L/PacketIR status and recreated the local-minimum failure mode where
new paradigm work exists only as prose.

## Code changes

- Added `l5_v2_asymptotic_pursuit_candidates()` in
  `src/tac/optimization/l5_staircase_v2.py`.
- Wired the candidate payload into `l5_v2_dispatch_readiness()`.
- Wired Cathedral validation queue rows in `tools/cathedral_autopilot.py` for:
  Z6/Z7/Z8, Rudin floor, and Tishby IB-pure.
- Wired `tools/operator_briefing.py` to show candidate count and a short
  next-action sample.
- Wired `tools/all_lanes_preflight.py` to fail if asymptotic candidate rows
  lose score-authority false fields, source ledgers, lane registry registration,
  or L1-build semantics.
- Emitted the machine-readable snapshot
  `.omx/research/l5_v2_asymptotic_candidate_surface_20260516_codex.json`.
- Added tests proving the rows are source-backed when ledgers exist and
  fail-closed when ledgers are absent.
- Split `ready_for_recommended_next_action` from `ready_for_l1_build` so
  source-backed ledgers do not over-authorize Rudin or Tishby before their
  ratification/probe gates.

## Registry repair

The Z6/Z7/Z8 scoping memo explicitly carried a lane id but had no lane registry
row. That violated the anti-duplication rule that a lane is registered once a
name and verdict exist. Repaired through the canonical mutator:

```bash
.venv/bin/python tools/lane_maturity.py add-lane \
  lane_time_traveler_l5_z6_z7_z8_predictive_coding_world_models_scoping_design_20260516 \
  --name "Z6/Z7/Z8 predictive-coding world-models scoping" \
  --phase 2

.venv/bin/python tools/lane_maturity.py mark \
  lane_time_traveler_l5_z6_z7_z8_predictive_coding_world_models_scoping_design_20260516 \
  --gate memory_entry \
  --evidence .omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md
```

The lane is now L1 because the `memory_entry` gate is satisfied. It remains
non-promotional: no implementation, no archive, no CPU/CUDA anchor.

The executable candidate surface also records `lane_registry_registered`,
`expected_first_artifact_status`, `ready_for_l1_build_semantics`, and
`ready_for_l1_scaffold_dispatch=false`. This prevents the common semantic
drift where "ready to start L1 scaffold work" is mistaken for "scaffold exists"
or "dispatch is allowed."

## Candidate next actions now visible

1. `z6_z7_z8_predictive_coding_world_models`
   - next action: `build_z6_l1_scaffold_first`
   - first artifacts: Z6 substrate package, trainer, and Modal T4 recipe
   - `ready_for_recommended_next_action=true`
   - `ready_for_l1_build=true`
   - blockers: L1 scaffold, identity-predictor disambiguator, paired CPU/CUDA
     anchor before score/rank authority

2. `rudin_floor_interpretable_ml_substrate`
   - next action: `ratify_and_build_rudin_k8_l1_scaffold`
   - first artifacts: Rudin substrate package, trainer, and Modal A100 recipe
   - `ready_for_recommended_next_action=true`
   - `ready_for_l1_build=false` until T3/Dykstra prerequisites are satisfied
   - blockers: T3 ratification, Dykstra feasibility, byte-mutation proof

3. `tishby_ib_pure_substrate`
   - next action:
     `run_d4_probe_and_build_variational_ib_tractability_tool`
   - first artifacts: D4 MI probe output, Variational-IB tractability checker,
     and tractability verdict JSON
   - `ready_for_recommended_next_action=true`
   - `ready_for_l1_build=false` until the D4/tractability gates are satisfied
   - blockers: D4 verdict, Variational-IB tractability, paired smoke vs ATW v2

## Verification

Focused tests:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_cathedral_autopilot.py::test_validation_queue_surfaces_l5_v2_packetir_stack_state \
  src/tac/tests/test_operator_briefing.py::test_briefing_json_composite_has_all_three_keys \
  -q
```

Result: `102 passed`.

Broader L5/Cathedral focused suite:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_cathedral_autopilot.py::test_validation_queue_surfaces_l5_v2_packetir_stack_state \
  src/tac/tests/test_operator_briefing.py \
  -q
```

Result: `121 passed`.

All-lanes/operator-briefing focused tests:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_all_lanes_operator_briefing_gate.py \
  src/tac/tests/test_operator_briefing.py \
  -q
```

Result: `45 passed`.

Lint and registry:

```bash
.venv/bin/python -m ruff check \
  src/tac/optimization/l5_staircase_v2.py \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_cathedral_autopilot.py \
  tools/all_lanes_preflight.py \
  tools/cathedral_autopilot.py \
  tools/operator_briefing.py

.venv/bin/python tools/lane_maturity.py validate
```

Results: `All checks passed!`; `OK — 770 lane(s) validated cleanly.`

Operator-visible smoke checks:

```bash
.venv/bin/python - <<'PY'
from tac.optimization.l5_staircase_v2 import l5_v2_dispatch_readiness
r = l5_v2_dispatch_readiness()
a = r["asymptotic_pursuit_candidates"]
print(a["schema"], a["candidate_count"], a["ready_for_exact_eval_dispatch"])
PY
```

Result: `l5_v2_asymptotic_pursuit_candidates_v1 3 False`.

The same payload is persisted at
`.omx/research/l5_v2_asymptotic_candidate_surface_20260516_codex.json` with
`authority=planning_only_no_score_rank_promotion_or_dispatch_authority`.

```bash
.venv/bin/python tools/cathedral_autopilot.py plan \
  --d-seg 0.00067082 --d-pose 0.0000336 --archive-bytes 185578 \
  --target-score 0.190 --output /tmp/pact_cathedral_l5_check.json
```

Result: the generated validation queue contains three
`l5_v2_asymptotic_pursuit_candidate` rows, all with
`score_claim=false` and `ready_for_exact_eval_dispatch=false`.

All-lanes operator-briefing validator:

```bash
.venv/bin/python - <<'PY'
import json
import subprocess
import sys
from tools.all_lanes_preflight import _operator_briefing_dispatch_failures
raw = subprocess.check_output(
    [sys.executable, "tools/operator_briefing.py", "--json"],
    text=True,
)
payload = json.loads(raw)
print(len(_operator_briefing_dispatch_failures(payload)))
PY
```

Result: `0`.

Syntax:

```bash
.venv/bin/python -m py_compile \
  src/tac/optimization/l5_staircase_v2.py \
  tools/cathedral_autopilot.py \
  tools/operator_briefing.py
```

Result: clean.

## Authority

This landing adds visibility and next-action custody only. It does not dispatch,
promote, rank, kill, or claim any score. Every row remains blocked until its
own byte-closed implementation, paired CPU/CUDA evidence, and adversarial
review packet exist.
