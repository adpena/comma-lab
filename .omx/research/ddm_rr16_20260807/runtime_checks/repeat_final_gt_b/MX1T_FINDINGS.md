# ddm_mx1t findings

## Verdict

MX1T completed the ARM-CAP n32 checkpoint-series facet analyzer and tail-average A/B.

| field | value |
|---|---:|
| axis | [macOS-CPU advisory torch upstream SegNet] |
| score_claim | false |
| checkpoint rows | 1 |
| tail-average rows | 0 |
| step-1500 anchor expected | 0.0010732014973958333 |
| step-1500 anchor measured | 0.0010732014973958333 |
| step-1500 abs diff | 0.0 |
| latest step | 3250 |
| latest aggregate d_seg | 0.0010732014973958333 |
| latest mismatch pixels | 6752 |

Receipts JSONL: `.omx/research/ddm_rr16_20260807/runtime_checks/repeat_final_gt_b/mx1t_facets_receipts.jsonl`

## Facet Trajectory

| step | aggregate d_seg | mismatch px | near-margin mismatch <=0.1 | far-margin mismatch >0.5 | churn/current |
|---:|---:|---:|---:|---:|---:|
| 3250 | 0.001073201497 | 6752 | 0.402251 | 0.057761 | n/a |

## Tail Average A/B

| row | d_seg | delta vs final | verdict |
|---|---:|---:|---|
| final step 3250 | 0.001073201497 | 0 | baseline |

## Iteration Verdict

| question | answer | measurement basis |
|---|---|---|
| near-flip vs stuck | far_margin_stuck_not_improving | mismatch <=0.1 fraction 0.4022511848341232 -> 0.4022511848341232; >0.5 fraction 0.057760663507109004 -> 0.057760663507109004 |
| residual owner classes | {'gt_mispredicted_top': [{'class_id': 0, 'class_name': 'Road', 'gt_sites': 1466822, 'gt_mispredicted': 5449, 'gt_mispredicted_rate': 0.0037148338380526063, 'pred_sites': 1462185, 'pred_false_positive': 812, 'pred_false_positive_rate': 0.0005553332854597743}, {'class_id': 1, 'class_name': 'Lane', 'gt_sites': 37967, 'gt_mispredicted': 539, 'gt_mispredicted_rate': 0.014196539099744516, 'pred_sites': 41275, 'pred_false_positive': 3847, 'pred_false_positive_rate': 0.09320411871592973}, {'class_id': 2, 'class_name': 'Undrivable', 'gt_sites': 3114070, 'gt_mispredicted': 473, 'gt_mispredicted_rate': 0.00015189125485297377, 'pred_sites': 3114420, 'pred_false_positive': 823, 'pred_false_positive_rate': 0.0002642546605788558}], 'pred_false_positive_top': [{'class_id': 1, 'class_name': 'Lane', 'gt_sites': 37967, 'gt_mispredicted': 539, 'gt_mispredicted_rate': 0.014196539099744516, 'pred_sites': 41275, 'pred_false_positive': 3847, 'pred_false_positive_rate': 0.09320411871592973}, {'class_id': 2, 'class_name': 'Undrivable', 'gt_sites': 3114070, 'gt_mispredicted': 473, 'gt_mispredicted_rate': 0.00015189125485297377, 'pred_sites': 3114420, 'pred_false_positive': 823, 'pred_false_positive_rate': 0.0002642546605788558}, {'class_id': 0, 'class_name': 'Road', 'gt_sites': 1466822, 'gt_mispredicted': 5449, 'gt_mispredicted_rate': 0.0037148338380526063, 'pred_sites': 1462185, 'pred_false_positive': 812, 'pred_false_positive_rate': 0.0005553332854597743}]} | latest checkpoint per-class GT-mispredicted and predicted false-positive counts |
| residual owner pairs | [{'pair_id': 8, 'mismatch_pixels': 278, 'pixels': 196608, 'd_seg': 0.0014139811197916667}, {'pair_id': 188, 'mismatch_pixels': 271, 'pixels': 196608, 'd_seg': 0.0013783772786458333}, {'pair_id': 91, 'mismatch_pixels': 262, 'pixels': 196608, 'd_seg': 0.0013326009114583333}, {'pair_id': 128, 'mismatch_pixels': 247, 'pixels': 196608, 'd_seg': 0.0012563069661458333}, {'pair_id': 497, 'mismatch_pixels': 239, 'pixels': 196608, 'd_seg': 0.0012156168619791667}] | latest checkpoint per-pair d_seg vector |
| churn regime | unmeasured_single_row | median churn/current None |
| tail-average verdict | tail_average_loses_or_unavailable | best K None, delta None |
| recommended next-config delta | do_not_assume_more_steps_pay_without_new_objective_or_capacity_change | series did not show the clean near-flip/low-churn continuation signature |

## RECALL EVIDENCE

| scope | query / source | found beyond charter seeds | changed plan |
|---|---|---|---|
| Governing files | `CHARTER.md, _common_contract.md, PROGRAM.md, CLAUDE.md/AGENTS.md, docs/operating_manual_craft_handoff.md, .omx/state/main_hot_state.md, upstream/evaluate.py` | mx1t owns only the n32 CPU-torch scorer instrument; live frontier is S=0.7534578126155775 @ 357,837 B and contest pointer is borrowed/unmoved. | Kept CPU-only, score_claim=false, copied checkpoints before reading, and used mx1h step-1500 as a hard anchor. |
| Prior MX1 verdict | `torch-verdict|mx1h|d7f557bb7c|0.0010689099629720051` | MX1H already implemented strict MLX NPZ -> torch loading and CPU upstream SegNet verdict; RR14 added fail-closed NPZ/history tests. | Extended the existing loader/verdict path with torch-facets instead of adding a new loader. |
| Tail-average precedent | `git log --grep dy2 and .omx/research/ddm_dy2_20260805/RECEIPT.md` | dy2 registered jd1_plateau_tail_average_ema_v1 and documented an explicit growing-horizon tail average law for JD1. | Used a scoped post-hoc simple parameter mean for MX1 checkpoints and labeled it as this vehicle/stage only, not a general EMA verdict. |
| Canonical equations | `.venv/bin/python tools/list_canonical_equations.py --json | rg 'jd1_plateau|ema_decay|score_marginal|SegNet'` | Relevant entries include score_marginal_lagrange_multipliers_v1, ema_decay_substrate_stage_aware_v1, and jd1_plateau_tail_average_ema_v1. | No score recomputation was promoted; tail average stayed a measured A/B row under the n32 advisory axis. |
| Class order | `CLAUDE.md SegNet class table and class-order corpus search` | Canonical comma10k order is Road/Lane/Undrivable/Movable/MyCar; luma-sort is forbidden and historically wrong. | Per-class facets use that fixed class order and record the provenance in every row. |

## Boundaries

- Axis: [macOS-CPU advisory torch upstream SegNet].
- Scope: n32 ARM-CAP checkpoint-series instrument only.
- No Metal, MLX training, n600 scorer job, archive build, remote dispatch, or `upstream/evaluate.py` run.
- Live run directory was copied from before reading and otherwise kept read-only.
- Score claim is false; this is not a contest-CPU or contest-CUDA row.

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.
