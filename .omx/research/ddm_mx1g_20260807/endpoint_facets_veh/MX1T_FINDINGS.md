# ddm_mx1t findings

## Verdict

MX1T completed the ARM-VEH n32 checkpoint-series facet analyzer and tail-average A/B.
(Label corrected by MAIN 2026-08-08: the tool's findings template hardcodes "ARM-CAP"; this run's
inputs are VEH — input_cache tq1c_seg_cache.pt sha 11fd8901…, checkpoint_dir launch_arm_veh/n32_metal.
Template-hardcoded-arm-name filed as instrument debt, #977 leg.)

| field | value |
|---|---:|
| axis | [macOS-CPU advisory torch upstream SegNet] |
| score_claim | false |
| checkpoint rows | 20 |
| tail-average rows | 3 |
| step-1500 anchor expected | 0.004514535268147786 |
| step-1500 anchor measured | 0.004514535268147786 |
| step-1500 abs diff | 0.0 |
| latest step | 5000 |
| latest aggregate d_seg | 0.004509290059407552 |
| latest mismatch pixels | 28370 |

Receipts JSONL: `.omx/research/ddm_mx1g_20260807/endpoint_facets_veh/mx1t_facets_receipts.jsonl`

## Facet Trajectory

| step | aggregate d_seg | mismatch px | near-margin mismatch <=0.1 | far-margin mismatch >0.5 | churn/current |
|---:|---:|---:|---:|---:|---:|
| 250 | 0.004396756490 | 27662 | 0.126491 | 0.519847 | n/a |
| 500 | 0.004427909851 | 27858 | 0.134432 | 0.501579 | 0.073157 |
| 750 | 0.004455407461 | 28031 | 0.137526 | 0.490172 | 0.048910 |
| 1000 | 0.004555066427 | 28658 | 0.142194 | 0.476621 | 0.063054 |
| 1250 | 0.004474480947 | 28151 | 0.143938 | 0.474868 | 0.045860 |
| 1500 | 0.004514535268 | 28403 | 0.145583 | 0.466641 | 0.039573 |
| 1750 | 0.004512945811 | 28393 | 0.150178 | 0.459303 | 0.037333 |
| 2000 | 0.004528363546 | 28490 | 0.155423 | 0.453492 | 0.031695 |
| 2250 | 0.004615624746 | 29039 | 0.162058 | 0.447880 | 0.037639 |
| 2500 | 0.004673163096 | 29401 | 0.163056 | 0.442468 | 0.028979 |
| 2750 | 0.004677772522 | 29430 | 0.163133 | 0.442168 | 0.018994 |
| 3000 | 0.004633267721 | 29150 | 0.160034 | 0.443911 | 0.017907 |
| 3250 | 0.004611333211 | 29012 | 0.159555 | 0.443644 | 0.015649 |
| 3500 | 0.004578113556 | 28803 | 0.159949 | 0.443843 | 0.016977 |
| 3750 | 0.004548549652 | 28617 | 0.160324 | 0.443897 | 0.014746 |
| 4000 | 0.004536469777 | 28541 | 0.160821 | 0.443152 | 0.014015 |
| 4250 | 0.004530747732 | 28505 | 0.161516 | 0.443431 | 0.013612 |
| 4500 | 0.004515806834 | 28411 | 0.160924 | 0.443631 | 0.009996 |
| 4750 | 0.004514058431 | 28400 | 0.161092 | 0.443239 | 0.007782 |
| 5000 | 0.004509290059 | 28370 | 0.160909 | 0.443285 | 0.005710 |

## Tail Average A/B

| row | d_seg | delta vs final | verdict |
|---|---:|---:|---|
| final step 5000 | 0.004509290059 | 0 | baseline |
| avg-K=2 | 0.004510879517 | 0.000001589457 | loses |
| avg-K=4 | 0.004518349965 | 0.000009059906 | loses |
| avg-K=8 | 0.004544576009 | 0.000035285950 | loses |

## Iteration Verdict

| question | answer | measurement basis |
|---|---|---|
| near-flip vs stuck | near_flip_fraction_rising | mismatch <=0.1 fraction 0.12649121538572772 -> 0.16090941135001763; >0.5 fraction 0.5198467211336852 -> 0.44328516038068383 |
| residual owner classes | {'gt_mispredicted_top': [{'class_id': 0, 'class_name': 'Road', 'gt_sites': 1466822, 'gt_mispredicted': 14313, 'gt_mispredicted_rate': 0.009757830193438604, 'pred_sites': 1463266, 'pred_false_positive': 10757, 'pred_false_positive_rate': 0.007351363320134549}, {'class_id': 1, 'class_name': 'Lane', 'gt_sites': 37967, 'gt_mispredicted': 6794, 'gt_mispredicted_rate': 0.17894487317933996, 'pred_sites': 37060, 'pred_false_positive': 5887, 'pred_false_positive_rate': 0.15885051268213707}, {'class_id': 3, 'class_name': 'Movable', 'gt_sites': 73592, 'gt_mispredicted': 3481, 'gt_mispredicted_rate': 0.047301337101858895, 'pred_sites': 72724, 'pred_false_positive': 2613, 'pred_false_positive_rate': 0.035930366866508995}], 'pred_false_positive_top': [{'class_id': 0, 'class_name': 'Road', 'gt_sites': 1466822, 'gt_mispredicted': 14313, 'gt_mispredicted_rate': 0.009757830193438604, 'pred_sites': 1463266, 'pred_false_positive': 10757, 'pred_false_positive_rate': 0.007351363320134549}, {'class_id': 1, 'class_name': 'Lane', 'gt_sites': 37967, 'gt_mispredicted': 6794, 'gt_mispredicted_rate': 0.17894487317933996, 'pred_sites': 37060, 'pred_false_positive': 5887, 'pred_false_positive_rate': 0.15885051268213707}, {'class_id': 2, 'class_name': 'Undrivable', 'gt_sites': 3114070, 'gt_mispredicted': 3226, 'gt_mispredicted_rate': 0.0010359433153397323, 'pred_sites': 3115478, 'pred_false_positive': 4634, 'pred_false_positive_rate': 0.0014874122044835495}]} | latest checkpoint per-class GT-mispredicted and predicted false-positive counts |
| residual owner pairs | [{'pair_id': 514, 'mismatch_pixels': 1216, 'pixels': 196608, 'd_seg': 0.006184895833333333}, {'pair_id': 65, 'mismatch_pixels': 1150, 'pixels': 196608, 'd_seg': 0.005849202473958333}, {'pair_id': 577, 'mismatch_pixels': 1144, 'pixels': 196608, 'd_seg': 0.005818684895833333}, {'pair_id': 591, 'mismatch_pixels': 1087, 'pixels': 196608, 'd_seg': 0.005528767903645833}, {'pair_id': 201, 'mismatch_pixels': 1062, 'pixels': 196608, 'd_seg': 0.005401611328125}] | latest checkpoint per-pair d_seg vector |
| churn regime | low_churn_stable_residual | median churn/current 0.018994223581379546 |
| tail-average verdict | tail_average_loses_or_unavailable | best K 2, delta 1.5894571940107058e-06 |
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
- Scope: n32 ARM-VEH checkpoint-series instrument only (same "ARM-CAP" template correction as above).
- No Metal, MLX training, n600 scorer job, archive build, remote dispatch, or `upstream/evaluate.py` run.
- Live run directory was copied from before reading and otherwise kept read-only.
- Score claim is false; this is not a contest-CPU or contest-CUDA row.

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.
