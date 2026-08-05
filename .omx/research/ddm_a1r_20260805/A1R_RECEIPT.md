# A1R receipt - realized_gate statistic repair

Date: 2026-08-05

Answer first: **repair landed, alarm registered, pointer unmoved.** The TR1
`realized_gate` now emits the legacy `realized_gate_dseg_mean` unchanged plus GD1's
repaired `realized_gate_dseg_mean_ht`, ordered per-pair ids, ordered per-pair d_seg values,
quantiles, max, sd, and a simple heavy-tail count. `A1_REALIZATION_GAP_ALARM` is now in
`src/tac/alarm_calibration.py` with weak/nonexchangeable exchangeability grading,
`block_calibration_required=true`, and a falsifier tied to the repaired producer keys.

No launches, no frozen scorer forwards, no `upstream/evaluate.py`, no n600 replay, no
threshold changes, no training-loop math changes, and no live gate-decision migration were
performed. Gate decisions that currently consume the legacy mean are left unchanged because
switching them to HT changes behavior and needs its own A/B.

## Landed changes

| File | Change |
|---|---|
| `experiments/train_tr1_partition_renderer_mlx.py` | Added `gd1_realized_gate_dseg_fields(...)`; `realized_gate(...)` emits per-pair vector, pair ids, HT mean, HT design label, q50/q90/q95, and heavy-tail count alongside the old mean. New fields are stripped from checkpoint `telemetry_tail` and remain telemetry-only. |
| `src/tac/alarm_calibration.py` | Registered `A1_REALIZATION_GAP_ALARM` in the L1 alarm registry with diagnostic-only authority, block-calibration requirement, realization-gap FDR family, and producer-key notes. |
| `src/tac/tests/test_gd1_gate_estimator.py` | Added synthetic tests for old-key back-compat, HT repair, and a heavy-tail case where the mean remains small while pair-tail fields flag. |
| `src/tac/tests/test_alarm_calibration.py` | Updated registry tests to require the repaired A1 row and queryability. |
| `src/tac/tests/test_ddm_bs3_gate_projection_kernel.py` | Pinned the new GD1 telemetry fields as checkpoint-stripped so checkpoint bytes do not silently change. |

## GD1 recall

Source: `.omx/research/ddm_gd1_undecided_defaults_audit_20260731.md`.

GD1 specified the producer repair exactly:

```text
row["realized_gate_dseg_per_pair"] = [float(d) for d in dsegs]
row["realized_gate_dseg_mean_ht"] = horvitz_thompson_mean(
    _GD1_DESIGN, {int(i): float(d) for i, d in zip(gate_ids, dsegs)})
```

It also specified the boundary: the trainer patch must be additive, leave
`realized_gate_dseg_mean` untouched, and use the same 36 renders with different weights.

## Consumer table

| Consumer | Current use | Disposition | Rationale |
|---|---|---|---|
| `a1_adjudicate` in `experiments/train_tr1_partition_renderer_mlx.py` | Stage-exit gate decision from legacy mean drop | **QUEUED, not migrated** | Behavior-changing decision logic. Charter says stop this part if the repair reveals a mean-consuming gate decision. |
| `boundary_positive_control` | Bit-exact reproduction check against parent legacy mean | Keep mean | This is a checkpoint-history comparability check, not an HT calibration site. |
| `boundary_jump_row` | Cross-boundary measurement row | **QUEUED, not migrated** | Decision/measurement semantics change if HT replaces the historic parent-tail value. |
| `gate_interval_fields` | Telemetry interval delta | Keep mean for history; HT interval is follow-on | Display/analysis aggregate over the existing series. |
| `basin_window` / `basin_entry_fires` | Handoff decision input | **QUEUED, not migrated** | Behavior-changing handoff logic. Needs A/B with logged HT before migration. |
| Gate print row | Console display | Keep mean | Display aggregate and historical operator-facing value. |
| `tools/supervise_ddm_b4s_burn4.py`, `tools/supervise_ddm_r1c_rung1.py`, `experiments/ddm_dw1_*`, `tools/ddm_gd5_ds32_window_chain.sh`, `tools/build_ddm_bp1_arm_tickets.py` | Final-gate summaries, verdict harvests, ticket metadata | Keep mean now | Historical aggregate consumers. Migrate only with the trainer decision migration so old receipts remain comparable. |
| `tools/ddm_dt1_compare_run_determinism.py` | Determinism key fixture | Keep mean | It is a deterministic legacy-key fixture, not an A1 alarm calibration consumer. |

Follow-on fired as queue disposition: **A1_HT_DECISION_MIGRATION**. Fire order: at the next safe
trainer/supervisor boundary, replay logged rows containing both mean and HT fields, produce a
mean-vs-HT decision-delta table for `a1_adjudicate`, `boundary_jump_row`, and `basin_entry_fires`,
then migrate only with a separate behavior A/B. Do not alter thresholds in that migration.

## Recall evidence

| Scope | Queries / files | Finding beyond charter seeds | Plan impact |
|---|---|---|---|
| Governing context | A1R charter, common arm contract, `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | This arm is scorer-free; `od9` owns the scorer slot; serializer and two review passes bind; own-vehicle frontier is unchanged. | Kept the landing additive and scorer-free. |
| GD1 exact spec | `rg --files .omx/research ... gd1`, `.omx/research/ddm_gd1_undecided_defaults_audit_20260731.md`, `src/tac/optimization/ddm_gd1_gate_estimator.py`, `src/tac/tests/test_gd1_gate_estimator.py` | The HT/per-pair repair already existed as pure arithmetic but was not wired into `realized_gate`. | Reused `GateDesign` and `horvitz_thompson_mean`; did not re-derive estimator. |
| AL1 block | `.omx/research/ddm_al1_20260805/AL1_RECEIPT.md`, `src/tac/alarm_calibration.py` | AL1 intentionally excluded A1 only because the producer key was missing. | Added A1 row after producer repair. |
| Full-corpus recall | `rg -n "ddm_gd1|GD1|A1_REALIZATION_GAP_ALARM|realized_gate_dseg_mean|realized_gate" ... --glob '!experiments/results/**'`; canonical equations filtered for `realized`, `realization`, `gate`, `heavy`, `tail`, `A1`, `GD1` | Existing canonical equations had realization/gate context but no conflicting A1 registry producer. Current trainer already had bs3 max/sd/per-class fields, so the live gap was narrower than the charter's source snapshot. | Avoided deleting predecessor telemetry; completed GD1 per-pair+HT gap additively. |
| Consumers | `rg -n "realized_gate_dseg_mean" experiments/train_tr1_partition_renderer_mlx.py tools/... experiments/...` | Multiple gate/decision consumers use the legacy mean. | Left behavior unchanged and queued migration with fire order. |

## Tests

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_gd1_gate_estimator.py \
  src/tac/tests/test_alarm_calibration.py \
  src/tac/tests/test_ddm_bs3_gate_projection_kernel.py -q
```

Result: `45 passed in 0.51s`.

```bash
.venv/bin/python -m ruff check \
  src/tac/alarm_calibration.py \
  src/tac/tests/test_gd1_gate_estimator.py \
  src/tac/tests/test_alarm_calibration.py \
  src/tac/tests/test_ddm_bs3_gate_projection_kernel.py
```

Result: `All checks passed!`

```bash
.venv/bin/python -m ruff check experiments/train_tr1_partition_renderer_mlx.py \
  --select F401,F821,F841,F811
.venv/bin/python -m py_compile experiments/train_tr1_partition_renderer_mlx.py
```

Result: both passed. Full-file ruff on the hot trainer still reports pre-existing style findings
outside this diff, so it was not used as the acceptance gate for this narrow repair.

Two review passes were marked with `tools/review_tracker.py` for all edited `.py` files.

## JSON

```json
{
  "schema": "ddm_a1r_receipt.v1",
  "date_utc": "2026-08-05",
  "repair_landed": true,
  "alarm_registered": true,
  "producer_keys": [
    "realized_gate_dseg_mean_ht",
    "realized_gate_dseg_per_pair",
    "realized_gate_pair_ids",
    "realized_gate_dseg_per_pair_q50",
    "realized_gate_dseg_per_pair_q90",
    "realized_gate_dseg_per_pair_q95",
    "realized_gate_dseg_per_pair_gt_2x_mean_n"
  ],
  "legacy_key_preserved": "realized_gate_dseg_mean",
  "behavior_changed": false,
  "scorer_forwards": 0,
  "launches": 0,
  "tests": {
    "pytest_focused": "45 passed",
    "ruff_non_trainer": "passed",
    "trainer_f_select": "passed",
    "trainer_py_compile": "passed"
  },
  "followons": [
    {
      "id": "A1_HT_DECISION_MIGRATION",
      "disposition": "QUEUED-WITH-FIRE-ORDER",
      "fire_order": "At the next safe trainer/supervisor boundary, replay logged mean+HT rows, produce decision-delta table for a1_adjudicate, boundary_jump_row, and basin_entry_fires, then migrate only with a separate behavior A/B and no threshold changes."
    }
  ]
}
```

## NEXT_IF_RESUMED

1. Run `A1_HT_DECISION_MIGRATION` at a safe trainer/supervisor boundary; do not hot-swap the live
   gate decision.
2. Add matched same-window/block calibration samples before allowing A1 p-values to drive any live
   alarm-fire disposition.
3. If a future full-n600 anchor carries per-pair d_seg, add the GD1 `anchored_mean` correction as a
   new key; do not replace the legacy key.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
