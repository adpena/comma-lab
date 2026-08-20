# CQ2 comma10k-only tiny-student sizing receipt - 2026-08-05

Status: **NO_PASS_SELECTED_BEST_PUBLIC_VAL_DIAGNOSTIC**.

Axis: `[macOS-CPU advisory / public-data tiny-student chart sizing / scorer-free]`.
`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`.

Own-vehicle baseline from hot state: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.`

## Answer First

The dataset blocker from `102d1b4fda` is resolved: comma10k is complete at sha
`6c205fe4c43cc53b2b1befafb1060d0606555027` with `9888` imgs
and `9888` masks. CQ2 trained the requested three public-data-only
student sizes and selected `25k` by the pre-registered comma10k-val rule.

## RECALL EVIDENCE

- Charter seeds read: `.omx/research/ddm_cq1_20260804/{cq1_receipt.md,NEXT_IF_RESUMED.md}` and `.omx/research/ddm_se3_20260804/se3_receipt.md`.
- Current resume receipt read: `.omx/research/ddm_cq2_20260804/{cq2_receipt.md,NEXT_IF_RESUMED.md}`; it changed the plan from blocker-only to dataset re-preflight plus training after the 2026-08-05 manifest.
- Corpus search beyond seeds found `.omx/research/ddm_rf1_20260804/RF1_RECEIPT_20260804.md`: qo1 has no legal receiver class chart, so the student remains the legal chart-source fallback and the SE3 81KB/101KB rows stay assumption-scoped until receiver closure.
- Corpus search beyond seeds found `.omx/research/ddm_nb1_20260804/nb1_receipt.md`: CQ1 GOOD-overlap and SE3 stream prices stand at their written scopes; n32 comparability is allowed but not a population verdict.
- Corpus search beyond seeds found `.omx/research/ddm_bf1_20260805/BF1_RECEIPT_20260805.md` and the 2026-08-05 per-edge directive: BF1 settled a receiver-closed lane-crop price, while cq2's 75KB lane remains relevant only if a low-dim/per-edge description can ride this public student.
- Corpus search beyond seeds found `.omx/research/distillation_smaller_student_20260610T191237Z.md`: older contest-frame student distillation was training-stability-limited and non-monotone; CQ2 therefore records explicit stop reasons and selects only by public comma10k-val metrics.

## Custody

| item | value |
|---|---:|
| dataset path | `/Volumes/VertigoDataTier/pact/public_datasets/comma10k` |
| clone manifest sha | `6c205fe4c43cc53b2b1befafb1060d0606555027` |
| manifest imgs / masks | `9888 / 9888` |
| actual imgs / masks | `9888 / 9888` |
| git HEAD | `6c205fe4c43cc53b2b1befafb1060d0606555027` |
| split | `8900 train / 988 val` |
| teacher model sha | `8208672861ad1b111dc98f3a7c54196d29875b709c7353e2dd1b7614343fb3a8` |
| eval config sha | `d260853fe0a993e23613ff38039fdce59264f5fe31f729c1fa65f8c3e5fde913` |

## Pre-Registered Selection Rule

Before any CQ1 overlap read, choose the smallest measured counted-byte student whose public comma10k-val
teacher-chart metrics satisfy: Road IoU >= `0.9`, Lane IoU >=
`0.5`, and Road/Lane mean IoU >= `0.72`.
If none pass, choose the best Road/Lane mean IoU candidate as a diagnostic non-passing frozen student;
that overlap read cannot promote the route.

## Size Curve

| size | width | params | counted B | Road IoU | Lane IoU | mean | pass | stop |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 25k | 20 | 22,805 | 23,489 | 0.000000 | 0.000000 | 0.000000 | False | safety_bound_REPORTED_max_steps |

Compression is the smallest measured real payload among int8/fp16 x Brotli q11/zlib9, written as a
decodeable tensor package and reloaded before final validation/overlap.

## Selected Student

| field | value |
|---|---:|
| selected label | `25k` |
| counted bytes | `23489` |
| payload path | `/Volumes/VertigoDataTier/pact/ddm_cq2_20260804_smoke_overlap/smoke_1step_overlap/25k_student_weights.bin` |
| payload sha256 | `f770e58d3022c46692b7c5ec0f5a7438b054f7f6c5d28ac0aab87008567297e7` |
| selected status | `NO_PASS_SELECTED_BEST_PUBLIC_VAL_DIAGNOSTIC` |

## Final CQ1 Overlap Read

Selected-student CQ1 overlap verdict: **POOR-OVERLAP**.

| metric | value |
|---|---:|
| SE3 r1 captured flips | `8670` |
| selected-student overlap numerator | `0` |
| selected-student overlap fraction | `0.0000000000` |
| GOOD threshold | `0.8` |

## Economics

| stream row | student B | stream B | total B | break-even survival |
|---|---:|---:|---:|---:|
| side_implied | 23,489 | 81,365 | 104,854 | 0.509468 |
| explicit_direction | 23,489 | 100,904 | 124,393 | 0.604405 |
| ed1_section_baseline | 23,489 | 169,149 | 192,638 | 0.935996 |

Live realizer context: se2's paint ceiling remains `0.263-0.407`, which only clears rows whose
break-even survival is below that band; sq2's solved-field eta remains the live realization candidate.
The composition verdict is MAIN's after cq2 and sq2 are both consumed.

## Boundaries

- Training and selection used only comma10k public images plus the public teacher.
- No contest SegNet/PoseNet forward was run.
- No `upstream/evaluate.py` run was performed.
- No `archive.zip` was built.
- The final overlap read used the frozen selected payload after public-val selection; it was not used to choose among candidates.
- All bulk artifacts are under `/Volumes/VertigoDataTier/pact/ddm_cq2_20260804_smoke_overlap`; no `/tmp` evidence is cited.

Own-vehicle frontier line: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.`
