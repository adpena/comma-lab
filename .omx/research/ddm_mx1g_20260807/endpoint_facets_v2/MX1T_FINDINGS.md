# ddm_mx1t findings

## Verdict

MX1T completed the ARM-CAP n32 checkpoint-series facet analyzer and tail-average A/B.

| field | value |
|---|---:|
| axis | [macOS-CPU advisory torch upstream SegNet] |
| score_claim | false |
| checkpoint rows | 24 |
| tail-average rows | 3 |
| step-1500 anchor expected | 0.0010689099629720051 |
| step-1500 anchor measured | 0.0010689099629720051 |
| step-1500 abs diff | 0.0 |
| latest step | 6000 |
| latest aggregate d_seg | 0.0010890960693359375 |
| latest mismatch pixels | 6852 |

Receipts JSONL: `.omx/research/ddm_mx1g_20260807/endpoint_facets_v2/mx1t_facets_receipts.jsonl`

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
| 3500 | 0.001079877218 | 6794 | 0.402708 | 0.058287 | 0.022078 |
| 3750 | 0.001077651978 | 6780 | 0.401032 | 0.058112 | 0.016224 |
| 4000 | 0.001080195109 | 6796 | 0.402590 | 0.058564 | 0.012360 |
| 4250 | 0.001083532969 | 6817 | 0.401056 | 0.058237 | 0.014816 |
| 4500 | 0.001083850861 | 6819 | 0.400205 | 0.058220 | 0.009972 |
| 4750 | 0.001085917155 | 6832 | 0.398126 | 0.058841 | 0.008343 |
| 5000 | 0.001086235046 | 6834 | 0.398742 | 0.059116 | 0.007902 |
| 5250 | 0.001086076101 | 6833 | 0.398361 | 0.059271 | 0.005122 |
| 5500 | 0.001088142395 | 6846 | 0.399357 | 0.059451 | 0.005405 |
| 5750 | 0.001089096069 | 6852 | 0.400029 | 0.059545 | 0.003795 |
| 6000 | 0.001089096069 | 6852 | 0.399008 | 0.059107 | 0.002627 |

## Tail Average A/B

| row | d_seg | delta vs final | verdict |
|---|---:|---:|---|
| final step 6000 | 0.001089096069 | 0 | baseline |
| avg-K=2 | 0.001088460286 | -0.000000635783 | wins |
| avg-K=4 | 0.001087188721 | -0.000001907349 | wins |
| avg-K=8 | 0.001086235046 | -0.000002861023 | wins |

## Iteration Verdict

| question | answer | measurement basis |
|---|---|---|
| near-flip vs stuck | near_flip_fraction_rising | mismatch <=0.1 fraction 0.39800393164978076 -> 0.39900758902510214; >0.5 fraction 0.06381370028731287 -> 0.05910683012259194 |
| residual owner classes | {'gt_mispredicted_top': [{'class_id': 0, 'class_name': 'Road', 'gt_sites': 1466822, 'gt_mispredicted': 5552, 'gt_mispredicted_rate': 0.0037850536738609046, 'pred_sites': 1462073, 'pred_false_positive': 803, 'pred_false_positive_rate': 0.0005492201825763829}, {'class_id': 1, 'class_name': 'Lane', 'gt_sites': 37967, 'gt_mispredicted': 550, 'gt_mispredicted_rate': 0.014486264387494403, 'pred_sites': 41340, 'pred_false_positive': 3923, 'pred_false_positive_rate': 0.09489598451862603}, {'class_id': 2, 'class_name': 'Undrivable', 'gt_sites': 3114070, 'gt_mispredicted': 452, 'gt_mispredicted_rate': 0.00014514766848529418, 'pred_sites': 3114481, 'pred_false_positive': 863, 'pred_false_positive_rate': 0.0002770927162503159}], 'pred_false_positive_top': [{'class_id': 1, 'class_name': 'Lane', 'gt_sites': 37967, 'gt_mispredicted': 550, 'gt_mispredicted_rate': 0.014486264387494403, 'pred_sites': 41340, 'pred_false_positive': 3923, 'pred_false_positive_rate': 0.09489598451862603}, {'class_id': 2, 'class_name': 'Undrivable', 'gt_sites': 3114070, 'gt_mispredicted': 452, 'gt_mispredicted_rate': 0.00014514766848529418, 'pred_sites': 3114481, 'pred_false_positive': 863, 'pred_false_positive_rate': 0.0002770927162503159}, {'class_id': 0, 'class_name': 'Road', 'gt_sites': 1466822, 'gt_mispredicted': 5552, 'gt_mispredicted_rate': 0.0037850536738609046, 'pred_sites': 1462073, 'pred_false_positive': 803, 'pred_false_positive_rate': 0.0005492201825763829}]} | latest checkpoint per-class GT-mispredicted and predicted false-positive counts |
| residual owner pairs | [{'pair_id': 8, 'mismatch_pixels': 278, 'pixels': 196608, 'd_seg': 0.0014139811197916667}, {'pair_id': 188, 'mismatch_pixels': 271, 'pixels': 196608, 'd_seg': 0.0013783772786458333}, {'pair_id': 91, 'mismatch_pixels': 265, 'pixels': 196608, 'd_seg': 0.0013478597005208333}, {'pair_id': 128, 'mismatch_pixels': 250, 'pixels': 196608, 'd_seg': 0.0012715657552083333}, {'pair_id': 152, 'mismatch_pixels': 243, 'pixels': 196608, 'd_seg': 0.0012359619140625}] | latest checkpoint per-pair d_seg vector |
| churn regime | low_churn_stable_residual | median churn/current 0.020486935866983372 |
| tail-average verdict | tail_average_wins_here | best K 8, delta -2.86102294921875e-06 |
| recommended next-config delta | apply_tail_average_selection_symmetrically_to_arm_cap_arm_veh_and_n120 | avg-K=8 beat final by d_seg -2.86102294921875e-06 |

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
