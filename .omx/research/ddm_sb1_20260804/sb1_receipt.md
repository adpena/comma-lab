# ddm_sb1 scorer batch receipt — 2026-08-04

Axis for completed scorer rows: `[macOS-CPU advisory]`. Score claims remain
`score_claim=false`, `promotion_eligible=false`; contest-CPU/CUDA pointer is
unmoved.

Baseline for deltas: fz4 `sub_final`, `S = 0.7541458627114951 @ 358,084 B`,
`d_seg = 0.00431179`, `d_pose = 0.00071459`, `rate = 0.00953734`
(`archive.zip` sha256 `ad5dd0e4fbe5b13ab53a5995a6d77cc558c25f40b63f894ea50ad336bd50fb66`).

## C — qo1 pair-bitpack

MEASURED n600 via `upstream/evaluate.py` on exact archive bytes.

Candidate:
`/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit/archive.zip`
(`d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a`,
357,836 B).

Receipt:
`/Volumes/VertigoDataTier/pact/ddm_sb1_20260804/C_qo1_pairbit_n600_eval_receipt.json`.

Result:

| field | value |
|---|---:|
| n | 600 |
| d_seg | 0.00431179 |
| d_pose | 0.00071459 |
| rate term | 0.00953073 |
| recomputed S | 0.7539807296911207 |
| delta vs baseline | -0.0001651330203744 |
| archive bytes | 357,836 |
| pose term `sqrt(10*d_pose)` | 0.084533425342 |
| R8 erosion vs 0.0845 bank | +0.000033425342 |

Verdict: **PASS / measured small rate win**. Seg and pose components matched
the fz4 baseline components exactly at reported precision; the row moved the
own-vehicle advisory baseline from `0.7541458627114951` to
`0.7539807296911207`. R8 pose-bank guard passed.

Pre-score execution corrections, not scorer rows:

- First fire inflated successfully but failed before scoring because the wrapper
  left `--videos` at `upstream/videos` while the compressed raw was not under
  `submission_dir/inflated`.
- Second fire inflated successfully but failed before scoring because
  `--videos` was pointed at the inflated raw directory; upstream expects GT
  source videos there and compressed raw under `submission_dir/inflated`.
- Correct fire inflated to
  `/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit/inflated/0.raw`
  and evaluated against GT `upstream/videos`.

## B - rt1 adaptive-margin rows on fz4 sub_final

MEASURED byte-side rebuild on actual fz4 `sub_final` tokens, then one n600 row
on the largest byte-saving candidate.

Byte receipt:
`/Volumes/VertigoDataTier/pact/ddm_sb1_20260804/B_rt1_subfinal_adaptive_byte_receipt.json`.

Selected candidate:
`/Volumes/VertigoDataTier/pact/ddm_sb1_20260804/B_rt1_margin_coupled_16_12_8_4_sub_final/archive.zip`
(`bc34987c711cdce91b33708bb98998f86832e3bc3db25ab2b4aada5e58711060`,
244,436 B).

Byte-side facts:

| field | margin [16,12,8,4] | derived fallback [16,12,8,4] |
|---|---:|---:|
| archive bytes | 244,436 | 295,582 |
| saved vs fz4 base | 113,648 B | 62,502 B |
| pre-registered net saved | 113,555 B | 62,502 B |
| pre-registered break-even `delta d_seg` | 0.0007561161342178816 | 0.0004161751628804195 |

Full n600 receipt:
`/Volumes/VertigoDataTier/pact/ddm_sb1_20260804/B_rt1_margin_16_12_8_4_n600_eval_receipt.json`.

Per-class decomposition:
`/Volumes/VertigoDataTier/pact/ddm_sb1_20260804/B_rt1_margin_16_12_8_4_per_class_seg_decomp.json`.

Result:

| field | value |
|---|---:|
| n | 600 |
| d_seg | 0.00515854 |
| d_pose | 0.16815221 |
| rate term | 0.00651040 |
| recomputed S | 1.9753490686354727 |
| delta vs baseline | +1.2212032059239775 |
| archive bytes | 244,436 |
| reported `delta d_seg` vs baseline | +0.00084675 |
| decomp `delta d_seg` | +0.0008467525906032986 |
| pre-registered bound | `< 0.0007561161342178816` |
| bound margin | +0.00009063645638541693 |
| pose term `sqrt(10*d_pose)` | 1.2967351695701015 |
| R8 erosion vs 0.0845 bank | +1.2122351695701015 |

Per-class `delta d_seg` contributors from the SegNet-only decomposition:

| class | delta errors | global `delta d_seg` |
|---|---:|---:|
| Road | +63,053 | +0.0005345069037543403 |
| Lane markings | +19,516 | +0.0001654391818576389 |
| Undrivable | +14,284 | +0.00012108696831597222 |
| Movable | +3,186 | +0.000027008056640625 |
| MyCar | -152 | -0.0000012885199652777778 |

Verdict: **FORMULATION negative** for this margin-coupled rt1 adaptive row on
the fz4 `sub_final` base. The byte saving is real, but the measured seg
collateral exceeds the pre-registered bound and the pose bank is destroyed.
Score claim remains false; contest-CPU/CUDA pointer unmoved. The derived
fallback was byte-closed but not scored because its break-even bound is tighter
and the selected margin row already failed both the seg bound and R8 guard.

## Q3 (#837) - pose-null projected reach

Typed blocker: **BLOCKED_BY_SINGLE_WRITER_INCOMPLETE_Q3_RECEIPT**.

The live board states the q3x lane is continuing under single-writer law and
explicitly says not to relaunch. Durable state found by SB1:

- `.omx/tmp/codex_runs/q3x_canonical.log` records only an n=2 strided smoke:
  retained fractions 0.30612244897959184 and 0.25075528700906347 against the
  threshold 0.9073878056818256; outcome `FOLDED`.
- The log writes the smoke JSON to `/tmp/ddm_q3x_smoke2.json`, which is not
  acceptable persisted evidence under the common contract.
- `.omx/research/ddm_q3x_q3_convergence_measurement_20260803.json` and
  `.omx/tmp/codex_runs/q3x_canonical.last.txt` were absent in the searched
  scope, so there is no durable >=32-pair receipt for SB1 to consume.

Disposition: no n600 Q3 row was run. This is a custody/ownership blocker, not a
family negative. NEXT-IF-RESUMED: wait for or recover the single-writer q3x
receipt; if it is dead, close that lane terminally first, then run the >=32
stratified/matched-base Q3 reach measurement to a durable non-`/tmp` receipt
before any n600 scorer spend.

## sq1 uncap (#935)

Typed blocker: **BLOCKED_CAP_ARTIFACT_NOT_CONVERGED**.

Consumed existing durable receipts under
`/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/`:

- `sq1_stage_n32.json` / `sq1_aggregate_n32.json`: 25-step receipt,
  `eta_net_pooled = 0.7895095948827292`, no explicit convergence stop reason.
- `sq1_stage_n32_uncap50_cw1.json` /
  `sq1_aggregate_n32_uncap50_cw1.json`: 50-step convergence-patience receipt,
  `eta_net_pooled = 0.8620042643923241`, `d_pose_after_mean =
  0.04925062965230609`, and cap census `31/32` best at the requested cap with
  `iteration_cap_best_at_cap`; `1/32` stopped as `iteration_cap_before_plateau`.

Disposition: no n600 sq1 row was run. The uncap50 evidence improves the seg
realizer but is still explicitly cap-bound, so it does not satisfy the charter's
"convergence-tested stop, not a cap" requirement and cannot produce a plausible
pre-score row under the R8 pose-bank guard. This is an instance/formulation
blocker on the current capped solve schedule, not a family kill.

NEXT-IF-RESUMED: run sq1 on the same 32 pairs with an explicit convergence stop
that is allowed to finish before the safety cap, then aggregate. Only build a
receiver-closed candidate and spend n600 if the converged receipt has plausible
`S < 0.7539807296911207` after pose-bank accounting.

## NEXT-IF-RESUMED

The only measured own-vehicle improvement in this SB1 batch is C:
`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`.

Immediate next work is not another n600 scorer spend. Close or recover the Q3
single-writer custody first, and rerun sq1 to a genuine convergence stop before
promotion. B's rt1 margin formulation is folded on this base.
