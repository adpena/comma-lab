---
schema: ddm_et4_solve_within_cvp_receipt.v1
date_utc: 2026-08-06
arm: ddm_et4
axis: "[macOS-CPU frozen-scorer advisory]"
score_claim: false
promotion_eligible: false
pointer_moved: false
tokens: [no-triality, p0-ledger-ok]
---

# DDM ET4 - n600 solve-within plus CVP compose

## Answer First

No realized joint n600 row exists yet. ET4 has a resumable first-8 timing checkpoint, not a banked
candidate:

| scope | d_seg before | d_seg after | d_pose before | d_pose after | bytes | S | dS |
|---|---:|---:|---:|---:|---:|---:|---:|
| first-8 prefix only | 0.003916422526 | 0.003335316976 | 0.000543665069 | 0.000544047352 | absent | absent | absent |

The named baseline remains `S=0.7534578126155775 @ 357,837 B`, archive sha256
`b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06`.

## First-8 Timing

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_et4_solve_within_cvp_n600.py --pair-start 0 --pair-stop 8 --resume
```

Measured rows: pairs `0..7`, elapsed total `568.3400745391846 s`, mean `71.04250931739807 s/pair`.
Projected serial n600 wall from this receipt: `42625.50559043884 s` (`11.84 h`). Remaining 592 pairs
project to `42057.16551589966 s` (`11.68 h`). These are schedule estimates only, not score projections.

Completed-scope aggregate:

| field | value |
|---|---:|
| flips_before | 6160 |
| flips_after | 5246 |
| net_flip_reduction | 914 |
| label_ceiling_net_fixed | 2539 |
| eta | 0.35998424576604965 |
| pose_ratio_min | 0.9975208759595314 |
| pose_ratio_median | 0.9992154567151883 |
| pose_ratio_mean | 1.0010761399699157 |
| pose_ratio_max | 1.0110802276564776 |
| cap_stop_counts.cap_bound | 8 |
| dk1_blocks_realized | 3116 |
| dk1_exact_declared_scope_blocks | 3116 |

Per-class error delta on the first-8 prefix:

| class | delta errors |
|---|---:|
| Road | -343 |
| Lane | -195 |
| Undrivable | -68 |
| Movable | -261 |
| MyCar | -47 |

## Pose Projection Verdict

The first-8 prefix does not settle the n32 to n600 pose projection. The measured prefix pose ratio
distribution is near neutral (`min/median/mean/max = 0.9975208759595314 / 0.9992154567151883 /
1.0010761399699157 / 1.0110802276564776`), but the common contract says n=8 banks nothing and prefix
bias can invert by axis. The fire order remains: finish n600 and measure the true mean pose.

## MAIN Override Recorded Verbatim

et3 measured eta 0.3562364 (2.083x bar 0.1710048742) with pose ratios 0.8128/1.0031/1.1284
(min/med/max) and correctly withheld under its pre-registered per-pair-max guard. MAIN
override BY S-ARITHMETIC (m67 pace-vs-direction, m52 never-binary-judgment): net seg+rate
win ~= 0.3562*0.18039 - 0.0308 ~= -0.0335 S vs worst-case pose cost (max ratio applied to
ALL pairs) ~= +0.0053 S — 6x margin even at the impossible tail; the n600 MEAN pose ratio is
the real quantity and THIS RUN MEASURES IT. Subset caveat honored: the n=32 set's pose
behavior may not project (m96 axis law) — which is an argument FOR the measurement, not
against it.

## VO2 Element Grade Vector

The machine-readable VO2-shaped vector is embedded in
`.omx/research/ddm_et4_20260806/et4_solve_within_cvp_summary.json` under `vo2_element_grade_vector`.

| element | grade | note |
|---|---|---|
| initialization | OPTIMAL-RECEIPT | parent sha and bytes verified |
| proposal_step_rule | OPTIMAL-RECEIPT | SW1 c-space `delta=Nc`, no reduced/project-after form |
| stopping_rule | OPTIMAL-RECEIPT | every completed row has a CapStopReceipt; all are cap_bound |
| metric_inner_product | OPTIMAL-RECEIPT | chartered SW1 saliency-weighted score metric, no global MS4D claim |
| subset_sampling | NAIVE-NAMED | first-8 prefix timing only; n=8 banks nothing |
| realization | OPTIMAL-RECEIPT | DK1 CVP/Babai finite kept-scope realizer, no global MIQP claim |
| projection_constraint_handling | OPTIMAL-RECEIPT | pose-null basis enforced inside solve |
| tie_breaks | UNKNOWN | deterministic local order, no global tie theorem |
| seed_determinism | OPTIMAL-RECEIPT | ET4 row order and overlay codec use no RNG |
| caches_staleness | OPTIMAL-RECEIPT | completed rows verify parent and GT argmax caches |

## RECALL EVIDENCE

Sources read beyond the charter seeds:

| surface | query or command | changed plan |
|---|---|---|
| Live board | `.omx/state/main_hot_state.md` | Confirmed the live own-vehicle baseline is `S=0.7534578126155775 @ 357,837 B` on parent `b35e756829...`. |
| ET3 receipt and runner | `.omx/research/ddm_et3_20260806/RECEIPT.md`, `experiments/ddm_et3_solve_within_cvp_phase_field.py` | Found ET3 measured rows but did not persist receiver-consumable frame deltas, so ET4 had to add patch records. |
| SW1/DK1/RW2/VO2 receipts | targeted reads of their 20260806 receipt files and VO2 element ledger | Kept solve-within full form, DK1 finite kept-scope wording, CapStopReceipt handling, and the 10-element grade-vector schema. |
| Parent runtime and TQ1 path | `experiments/ddm_tq1_optimal_token_edit.py`, qo1 `sub_auto_pairbit` runtime | Reused the real parent IX2 decoder and wrapped it rather than pretending image-domain CVP corrections are token edits. |
| Submission chain | `src/tac/submission_chain.py` | ET4 stages through canonical `stage_submission`, `run_inflate`, and `run_upstream_evaluate`; custom overlay bytes get their own ledger. |
| Corpus search | `rg "et4|solve-within+CVP|DK1 CVP|phase field sidecar|image-domain correction|overlay|delta sidecar"` | No existing ET4 sidecar grammar was found in scope; added an explicit counted sparse frame_1 overlay grammar. |

## Artifacts

| path | sha256 |
|---|---|
| `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_solve_within_cvp_rows.jsonl` | `2d4f9d1be067c7b682aaf360c2ed9651cb907cfffb3cf2d9d93c91864a05e8d2` |
| `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_solve_within_cvp_summary.json` | `ab6df3e5a443279f3a898b919a9598d002ea99c422001ae869496bfa2b0c5a60` |
| `.omx/research/ddm_et4_20260806/et4_solve_within_cvp_summary.json` | `ab6df3e5a443279f3a898b919a9598d002ea99c422001ae869496bfa2b0c5a60` |

Patch records exist for pairs `0..7` under
`/Volumes/VertigoDataTier/pact/ddm_et4_20260806/patch_records/`.

## Verification

| command | result |
|---|---|
| `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile experiments/ddm_et4_overlay_codec.py experiments/ddm_et4_overlay_inflate_runner.py experiments/ddm_et4_solve_within_cvp_n600.py experiments/tests/test_ddm_et4_overlay_codec.py` | passed |
| `.venv/bin/python -m pytest experiments/tests/test_ddm_et4_overlay_codec.py -q` | 2 passed |
| `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_et4_solve_within_cvp_n600.py --pair-start 0 --pair-stop 8 --resume --prepare-only` | passed, 8 rows loaded |
| `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_et4_solve_within_cvp_n600.py --pair-start 0 --pair-stop 8 --resume` | passed, first-8 timing receipt |

## Boundaries

No n600 completion, byte-closed ET4 archive, `upstream/evaluate.py` row, dS verdict, or banked twelfth-move
candidate exists yet. No pointer promotion claim is made. The first-8 prefix is only a timing and
resume-validity receipt.

Own-vehicle frontier remains unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
