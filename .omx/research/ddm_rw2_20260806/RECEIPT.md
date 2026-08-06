---
schema: ddm_rw2_receipt.v1
date_utc: 2026-08-06
arm: ddm_rw2
axis: "[mixed scorer-free + bounded n<=8 frozen-scorer advisory]"
score_claim: false
promotion_eligible: false
pointer_moved: false
tokens: [no-triality, p0-ledger-ok]
---

# RW2 Receipt

## Disposition First

| item | disposition |
|---|---|
| Trainer stop receipts | `PARTIAL_NOT_CURED`: 17 existing stop/checkpoint/event emit surfaces inventoried; 0 normalized CapStopReceipt-style rows added; R2 trainer `stopping_rule` remains `NAIVE-NAMED`. |
| Q3 three-arm A/B | `REALIZATION_CAVEAT_CURED_BUT_FOLDED`: full DK1-CVP covered 3117/3117 requested Q3 blocks; the n1 check still folded below threshold. |
| DK1 population re-grade | `SUBSET_DENOMINATOR_CURED_LOCAL_ONLY`: stratified n32 scorer-free ladder over 47,531 candidate phase blocks; CVP remains best local method; no real PoseNet/SegNet population claim. |
| Registry self-cure | `PARTIAL_CURED`: measured readable-live-Python denominator and priority-order tie-breaks added; 118/118 unaffected non-source rows hash-stable; bounded corpus scope remains. |

## Trainer Receipts

Evidence: `.omx/research/ddm_rw2_20260806/TRAINER_STOP_RECEIPT_INVENTORY.json`.

RW2 did not patch `experiments/train_levelset_witness_realized_through_R_mlx.py` or run a trainer. The source audit found existing rows for curriculum transitions, EventBackstopGate telemetry, tail/closed-loop stops, and best/stage/intra/final checkpoints, but no single normalized receipt covering every stop mode. Disposition: do not regrade the R2 trainer stopping element.

## Q3 A/B

Bounded n1, strided pair 0, `score_claim=false`, `promotion_eligible=false`.

| arm | realizer | block coverage | net Q3 flips | retained fraction | d_pose ratio | outcome |
|---|---|---:|---:|---:|---:|---|
| naive | naive-round | n/a | 45 | 0.30612244898 | 0.99994254086 | FOLDED |
| capped | dk1-cvp `max_blocks=64` | 64/3117 | -1 | -0.00680272109 | 1.00012002084 | FOLDED |
| full | dk1-cvp full requested mask | 3117/3117 | 43 | 0.29251700680 | 1.00307254820 | FOLDED |

The rw1 `max_blocks=64` caveat is now explicit in receipts: capped coverage is `PARTIAL_OR_CAPPED`, full coverage is `FULL_REQUESTED_MASK` with `coverage_form_grade=OPTIMAL-RECEIPT`. Full coverage removed the capped inversion but did not make Q3 pass.

## DK1 Re-grade

Evidence: `.omx/research/ddm_rw2_20260806/dk1_stratified_ladder_n32_scorer_free.json`.

Selection was `stratified_nonzero`, seed `20260806`, 10 strata, n32 over 47,531 candidate nonzero phase blocks. Per-stratum candidate counts were `[4748, 4791, 4788, 4402, 4462, 4709, 4439, 4871, 5216, 5105]`, with selected counts `[4, 4, 3, 3, 3, 3, 3, 3, 3, 3]`.

| method | local pose leakage mean | local pose leakage median | local seg discrepancy mean |
|---|---:|---:|---:|
| naive | 0.181765625 | 0.1808115 | 1.0514010001620446 |
| dykstra | 0.0778367984509196 | 0.07506509317355 | 0.5498632439193121 |
| cvp | 0.002622087661818458 | 0.000006873976163824788 | 0.08216367114073966 |

This cures the DK1 subset-denominator NAIVE element for local A(Dx)/D-private evidence only. `frozen_posenet_used=false`, so it is not a population PoseNet/SegNet verdict.

## Registry Self-Cure

Evidence:

- `.omx/research/ddm_rw2_20260806/vo2_registry_rebuild/ROUND_SUMMARY.json`
- `.omx/research/ddm_rw2_20260806/registry_rebuild_stability.json`

The rebuilt registry has 4,632 rows: 10 `vo1-round0`, 89 `ca1-round0`, 16 `sw1-round0`, 3 `dk1-round0`, and 4,514 `vo2-new` rows. The source denominator now records 6,247 readable live Python files, 0 decode errors, roots/exclusions, candidate rule, and token hit counts. Ordering is family priority, then verdict fanout descending, then instrument id.

Hash stability check: 118/118 unaffected non-source rows are unchanged between the original VO2 registry and the RW2 rebuild. Source-candidate rows changed from 4,512 to 4,514 because RW2 touched live source files and the denominator is intentionally live.

## NAIVE Count

Evidence: `.omx/research/ddm_rw2_20260806/NAIVE_COUNT.json`.

Same R2 23-row denominator:

| count | R2 baseline | after RW2 regrades |
|---|---:|---:|
| rows with any `NAIVE-NAMED` element | 14/23 | 13/23 |
| total `NAIVE-NAMED` elements | 26 | 21 |

The row-count improvement is only `dk1_realizer:cvp`; registry-builder source row still has bounded-corpus `subset_sampling`, and the trainer row remains uncured.

## Verification

| command | result |
|---|---|
| `.venv/bin/python -m py_compile src/tac/optimization/rw1_true_domain_instruments.py experiments/ddm_q3x_q3_convergence_measurement.py tools/measure_ddm_dk1_lattice_realizer.py tools/build_ddm_vo2_instrument_registry.py` | passed |
| `.venv/bin/python -m pytest src/tac/optimization/tests/test_rw1_true_domain_instruments.py tools/tests/test_ddm_vo2_instrument_registry.py src/tac/optimization/tests/test_lattice_native_pose_null_realizer.py` | 16 passed |
| `.venv/bin/python tools/build_ddm_vo2_instrument_registry.py --summary-only` | passed; 4,632 rows |

## Boundaries

- No `upstream/evaluate.py` run.
- No exact archive built.
- No scorer-slot ownership assumed.
- No protected files edited.
- No `/tmp` evidence persisted.
- No contest-CPU/CUDA or frontier claim.

Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer borrowed/unmoved.
