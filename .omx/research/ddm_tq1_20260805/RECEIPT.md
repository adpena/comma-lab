# ddm_tq1b Phase B realized acceptance receipt

Axis: `[macOS-CPU frozen-scorer advisory]`. `score_claim=false`; promotion remains MAIN-only.

## Result

- Tested moves: 12 from `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/optimal_form/tq1_phase_a_candidate_prices.jsonl`.
- Accepted moves: 8 (snap_r00_c02_L12, snap_r00_c04_L12, snap_r00_c06_L12, snap_r00_c08_L12, snap_r00_c10_L12, snap_r00_c01_L13, snap_r00_c22_L12, snap_r00_c02_L13).
- Saturation: SATURATED_ACCEPTED_PREFIX (queued 12-move Phase A price-ledger prefix only; the full 1,140-row derived menu is not claimed closed by this run).
- Baseline measured S: `0.7539814416337309` at 357836 B.
- Final held S: `0.7537933983374265` at 357837 B.
- Realized delta vs measured baseline: `delta_S=-0.000188043296`, `delta_d_seg=-0.000003153483`, `delta_d_pose=+0.000002142658`, `delta_bytes=+1`.

## Artifacts

- Receipt JSON: `/Users/adpena/Projects/pact/.omx/research/ddm_tq1_20260805/phase_b_realized_acceptance_receipt.json`
- Accepted-move ledger: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized/tq1_phase_b_accepted_move_ledger.jsonl`
- Realized measurement JSONL: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized/tq1_phase_b_realized_measurements.jsonl`
- Candidate archives: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized/candidate_archives`
- Scorer checkpoints: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized/stage_checkpoints`

## Dominated-Baseline Context

| id | S | bytes | d_seg | d_pose | scope |
|---|---:|---:|---:|---:|---|
| rt1_sb1_margin_coupled_16_12_8_4 | 1.9753490686354727 | 244436 | 0.00515854 | 0.16815221 | FORMULATION for that adaptive margin map on fz4 sub_final |
| fz4_map_repair_kill | 1.9690434 |  |  |  | FORMULATION for [16,12,8,4] map plus current F0PR1 repair |
| ed2_entropy_descent_a025 | 1.010325639339858 | 350130 | 0.004499130249023438 | 0.01071092 | FORMULATION/INSTANCE for qo1 IX2 discrete entropy step alpha=0.25 |

## Scope

queued 12-move Phase A price-ledger prefix only; the full 1,140-row derived menu is not claimed closed by this run
This is a macOS-CPU frozen-scorer advisory realization of the queued TQ1 Phase A prefix. It is not a contest-CPU/CUDA promotion row and does not move the contest pointer.

## Recall Evidence

- `.omx/tmp/codex_runs/tq1b_prompt.md`: Phase B fire order and acceptance rule.
- `.omx/tmp/codex_runs/_common_contract.md`: scorer slot, memory, serializer, and evidence constraints.
- `.omx/state/main_hot_state.md`: qo1 own-vehicle frontier and live JD6 co-tenancy.
- `tools/measure_ddm_v19_pure_priced_objective.py` and `tools/measure_ddm_v19b_joint_remeasure_stack.py`: realized receiver/scorer accounting pattern.
- `.omx/research/ddm_ng1_20260805/ng1_negative_results_audit.md` and `.omx/research/ddm_ed2_20260805/RECEIPT.md`: rt1/fz4/ed2 dominated rows.

Own-vehicle frontier line unchanged: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`. Contest pointer unchanged: `S = 0.1910828242`, borrowed/unmoved.
