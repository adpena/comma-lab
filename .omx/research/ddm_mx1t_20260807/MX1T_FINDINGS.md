# ddm_mx1t findings

## Verdict

MX1T completed the ARM-CAP n32 checkpoint-series facet analyzer and tail-average A/B.

| field | value |
|---|---:|
| axis | [macOS-CPU advisory torch upstream SegNet] |
| score_claim | false |
| checkpoint rows | 13 |
| tail-average rows | 3 |
| step-1500 anchor expected | 0.0010689099629720051 |
| step-1500 anchor measured | 0.0010689099629720051 |
| step-1500 abs diff | 0.0 |
| latest step | 3250 |
| latest aggregate d_seg | 0.0010732014973958333 |
| latest mismatch pixels | 6752 |

Receipts JSONL: `.omx/research/ddm_mx1t_20260807/mx1t_facets_receipts.jsonl`

## Facet Trajectory

| step | aggregate d_seg | mismatch px | near-margin mismatch <=0.1 | far-margin mismatch >0.5 | churn/current |
|---:|---:|---:|---:|---:|---:|
| 250 | 0.001051108042 | 6613 | 0.398004 | 0.063814 | n/a |
| 500 | 0.001066048940 | 6707 | 0.400924 | 0.063068 | 0.068883 |
| 750 | 0.001075267792 | 6765 | 0.396748 | 0.062232 | 0.060606 |
| 1000 | 0.001079877218 | 6794 | 0.400059 | 0.062261 | 0.045776 |
| 1250 | 0.001070181529 | 6733 | 0.397742 | 0.060151 | 0.044111 |
| 1500 | 0.001068909963 | 6725 | 0.400000 | 0.059331 | 0.036877 |
| 1750 | 0.001068909963 | 6725 | 0.398364 | 0.060372 | 0.040446 |
| 2000 | 0.001068751017 | 6724 | 0.400357 | 0.059488 | 0.034355 |
| 2250 | 0.001071453094 | 6741 | 0.401424 | 0.059635 | 0.029521 |
| 2500 | 0.001069068909 | 6726 | 0.402022 | 0.059025 | 0.024532 |
| 2750 | 0.001070658366 | 6736 | 0.403949 | 0.059086 | 0.020487 |
| 3000 | 0.001069704692 | 6730 | 0.404160 | 0.059138 | 0.020802 |
| 3250 | 0.001073201497 | 6752 | 0.402251 | 0.057761 | 0.019254 |

## Tail Average A/B

| row | d_seg | delta vs final | verdict |
|---|---:|---:|---|
| final step 3250 | 0.001073201497 | 0 | baseline |
| avg-K=2 | 0.001074473063 | 0.000001271566 | loses |
| avg-K=4 | 0.001072247823 | -0.000000953674 | wins |
| avg-K=8 | 0.001067320506 | -0.000005880992 | wins |

## Iteration Verdict

| question | answer | measurement basis |
|---|---|---|
| near-flip vs stuck | near_flip_fraction_rising | mismatch <=0.1 fraction 0.39800393164978076 -> 0.4022511848341232; >0.5 fraction 0.06381370028731287 -> 0.057760663507109004 |
| residual owner classes | {'gt_mispredicted_top': [{'class_id': 0, 'class_name': 'Road', 'gt_sites': 1466822, 'gt_mispredicted': 5449, 'gt_mispredicted_rate': 0.0037148338380526063, 'pred_sites': 1462185, 'pred_false_positive': 812, 'pred_false_positive_rate': 0.0005553332854597743}, {'class_id': 1, 'class_name': 'Lane', 'gt_sites': 37967, 'gt_mispredicted': 539, 'gt_mispredicted_rate': 0.014196539099744516, 'pred_sites': 41275, 'pred_false_positive': 3847, 'pred_false_positive_rate': 0.09320411871592973}, {'class_id': 2, 'class_name': 'Undrivable', 'gt_sites': 3114070, 'gt_mispredicted': 473, 'gt_mispredicted_rate': 0.00015189125485297377, 'pred_sites': 3114420, 'pred_false_positive': 823, 'pred_false_positive_rate': 0.0002642546605788558}], 'pred_false_positive_top': [{'class_id': 1, 'class_name': 'Lane', 'gt_sites': 37967, 'gt_mispredicted': 539, 'gt_mispredicted_rate': 0.014196539099744516, 'pred_sites': 41275, 'pred_false_positive': 3847, 'pred_false_positive_rate': 0.09320411871592973}, {'class_id': 2, 'class_name': 'Undrivable', 'gt_sites': 3114070, 'gt_mispredicted': 473, 'gt_mispredicted_rate': 0.00015189125485297377, 'pred_sites': 3114420, 'pred_false_positive': 823, 'pred_false_positive_rate': 0.0002642546605788558}, {'class_id': 0, 'class_name': 'Road', 'gt_sites': 1466822, 'gt_mispredicted': 5449, 'gt_mispredicted_rate': 0.0037148338380526063, 'pred_sites': 1462185, 'pred_false_positive': 812, 'pred_false_positive_rate': 0.0005553332854597743}]} | latest checkpoint per-class GT-mispredicted and predicted false-positive counts |
| residual owner pairs | [{'pair_id': 8, 'mismatch_pixels': 278, 'pixels': 196608, 'd_seg': 0.0014139811197916667}, {'pair_id': 188, 'mismatch_pixels': 271, 'pixels': 196608, 'd_seg': 0.0013783772786458333}, {'pair_id': 91, 'mismatch_pixels': 262, 'pixels': 196608, 'd_seg': 0.0013326009114583333}, {'pair_id': 128, 'mismatch_pixels': 247, 'pixels': 196608, 'd_seg': 0.0012563069661458333}, {'pair_id': 497, 'mismatch_pixels': 239, 'pixels': 196608, 'd_seg': 0.0012156168619791667}] | latest checkpoint per-pair d_seg vector |
| churn regime | low_churn_stable_residual | median churn/current 0.035615937141328075 |
| tail-average verdict | tail_average_wins_here | best K 8, delta -5.880991617838397e-06 |
| recommended next-config delta | apply_tail_average_selection_symmetrically_to_arm_cap_arm_veh_and_n120 | avg-K=8 beat final by d_seg -5.880991617838397e-06 |

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

## Serializer Status

Serializer commit was attempted with post-edit `--expected-content-sha256` for the implementation,
tests, findings, receipt JSONL, and result JSON files, with `[no-triality] [p0-ledger-ok]` and
`--no-co-author`. It failed before staging with `git_add_rc=128`: `error: unable to create
temporary file: Operation not permitted` / `failed to insert into database` for
`experiments/ddm_mx1_pr130_semantic_renderer.py`. `git diff --cached --name-only` was empty after
the failure. This is a managed-sandbox Git-write blocker, not a measurement, test, or review-gate
failure.

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.
