# LC1 Receipt - PE3 Ideal-Label Ceiling at n32

## Answer

LC1 measured `net_fixed = -12,884` on the NA3-derived non-prefix stratified n32. Base flips were `27,382`; ideal PE3 target-label substitution increased them to `40,266`.

Decision rule executed: PE3 target labels are NONPOSITIVE at n32, so PE3 demotes to CONDITIONING-ONLY and the TR1 learned carrier is CROWNED PRIMARY.

Axis: `[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE`. `score_claim=false`; no `upstream/evaluate.py`; no n600 authority row; contest pointer borrowed/unmoved.

Final frontier line carried from live state, not moved by LC1:

`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.`

## Measurement

LC1 reused RZ1's PE3 parse, NA3 selection, GT decode, frozen scorer, and target-label construction machinery, but intentionally did not run the forbidden solver/realizer/head-solve legs. This is the ideal-label-substitution leg only.

Command class:

```sh
.venv/bin/python - <<'PY'
# inline LC1 importer using experiments.ddm_rz1_pe3_head_solve helpers:
# extract_pe3_section, parse_pe3_components, select_pairs_from_na3,
# target_from_pe3, Scorer.seg_argmax, decode_gt_frames.
# Output: /Volumes/VertigoDataTier/pact/ddm_lc1_20260805/lc1_label_ceiling_n32.json
PY
```

The direct RZ1 script was not run because it always proceeds into `solve_margin_optimal_paint` and regional realization after target-label scoring; LC1 forbids solver, realizer, and head-solve.

Inputs verified:

| artifact | bytes | sha256 |
|---|---:|---|
| `/Volumes/VertigoDataTier/pact/ddm_pe4_20260805_r2/qo1_identity_pe4_extended_receiver/inflated/0.raw` | 3,662,409,600 | `3ce7d269a7080a4024a576694cd0ddc697099c64cd02fdd2bb879339e4b03f31` |
| `/Volumes/VertigoDataTier/pact/ddm_pe4_20260805_r2/sub_auto_pairbit_pe4_pe3_hybrid_75kb_receiver/archive.zip` | 432,428 | `3f08c7fdd1c2746fa456ef8b6d8005e850d1a3acac5665a5d08b2ef17585b5e0` |
| `.omx/research/ddm_na3_20260805/stratified_pose_selection_923.json` | 4,776 | `6f7ba6dde77500f09923ecfa7c85a7b977bda1e9f3332e416f6fe05c7bca06f7` |
| `/Volumes/VertigoDataTier/pact/ddm_lc1_20260805/lc1_label_ceiling_n32.json` | 82,232 | `d1e979759df209f40995ef9d0465024011af5c2171eb33323edba8dfaf892db6` |

PE3 parse-back:

| quantity | value |
|---|---:|
| PE3 section bytes | 74,408 |
| PE3 section sha256 | `5cc024ad32df7fedb18afb75dbed6be9c1af948dac826a1736cb1084949855c2` |
| PE3 raw bytes | 169,975 |
| PE3 raw sha256 | `beecc444dac58e7b345df3783a8b38e20c8c74e8b011ac82bd4cb02c24e697a8` |
| receiver raster sha256 | `1661535005f09a8dcd864fb54d20d18be618455bb7cf0c5801fec3c4efe83818` |
| component records | 8,644 |
| depth-conditioned curve records | 750 |
| generator-pair-bisector records | 7,894 |
| described scorer pixels, n600 | 540,058 |
| effective component/class prototype slots, n600 | 16,944 |

Selection:

| field | value |
|---|---|
| source | NA3 n120 stratified selection, seed `20260805` |
| derived seed | `20260837` |
| mode | `stratified_blocks_from_na3_n120_seeded_without_prefix` |
| pairs | `[32, 34, 47, 85, 88, 114, 163, 173, 176, 204, 217, 231, 251, 269, 293, 302, 313, 342, 348, 376, 406, 409, 422, 430, 464, 495, 510, 526, 543, 547, 574, 582]` |
| scope caveat | n32 stratified, seg-matched per NA3/m96 carry-forward at `1.0099888594483923x`; advisory routing only |

Pooled result:

| row | flips | d_seg over n32 denominator | d_seg over n600 denominator, unscaled subset numerator | net_fixed vs base |
|---|---:|---:|---:|---:|
| clean qo1 base | 27,382 | 0.004352251688639323 | 0.00023212009006076388 | 0 |
| ideal PE3 target labels | 40,266 | 0.006400108337402344 | 0.000341339111328125 | -12,884 |

Denominators:

| denominator | pixels |
|---|---:|
| n32 scorer pixels | 6,291,456 |
| n600 scorer pixels | 117,964,800 |

Mechanism totals:

| quantity | count |
|---|---:|
| PE3 band pixels in selected n32 | 28,841 |
| label-changed pixels | 14,725 |
| fixed pixels | 868 |
| introduced pixels | 13,752 |
| wrong-to-wrong changed pixels | 105 |

Every selected pair worsened:

| pair | base flips | ideal PE3 flips | net_fixed | fixed | introduced | changed label px | band px |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 781 | 1,224 | -443 | 10 | 453 | 469 | 876 |
| 34 | 742 | 1,121 | -379 | 10 | 389 | 401 | 747 |
| 47 | 762 | 1,149 | -387 | 12 | 399 | 412 | 791 |
| 85 | 790 | 1,132 | -342 | 29 | 371 | 402 | 805 |
| 88 | 1,088 | 1,506 | -418 | 65 | 483 | 555 | 1,102 |
| 114 | 833 | 1,272 | -439 | 8 | 447 | 459 | 864 |
| 163 | 942 | 1,426 | -484 | 37 | 521 | 566 | 1,065 |
| 173 | 876 | 1,358 | -482 | 20 | 502 | 523 | 1,011 |
| 176 | 847 | 1,243 | -396 | 24 | 420 | 446 | 877 |
| 204 | 646 | 903 | -257 | 28 | 285 | 315 | 642 |
| 217 | 640 | 961 | -321 | 19 | 340 | 360 | 753 |
| 231 | 807 | 1,187 | -380 | 50 | 430 | 482 | 962 |
| 251 | 922 | 1,158 | -236 | 25 | 261 | 287 | 571 |
| 269 | 707 | 1,092 | -385 | 10 | 395 | 408 | 807 |
| 293 | 938 | 1,382 | -444 | 29 | 473 | 505 | 996 |
| 302 | 758 | 1,119 | -361 | 33 | 394 | 431 | 882 |
| 313 | 660 | 922 | -262 | 25 | 287 | 313 | 570 |
| 342 | 729 | 1,094 | -365 | 13 | 378 | 392 | 738 |
| 348 | 755 | 1,132 | -377 | 18 | 395 | 417 | 770 |
| 376 | 735 | 1,227 | -492 | 12 | 504 | 517 | 991 |
| 406 | 772 | 1,153 | -381 | 16 | 397 | 417 | 806 |
| 409 | 840 | 1,283 | -443 | 16 | 459 | 479 | 933 |
| 422 | 825 | 1,205 | -380 | 18 | 398 | 422 | 845 |
| 430 | 619 | 878 | -259 | 6 | 265 | 273 | 545 |
| 464 | 818 | 1,207 | -389 | 20 | 409 | 431 | 867 |
| 495 | 944 | 1,300 | -356 | 30 | 386 | 418 | 811 |
| 510 | 1,086 | 1,553 | -467 | 63 | 530 | 604 | 1,247 |
| 526 | 887 | 1,299 | -412 | 40 | 452 | 495 | 972 |
| 543 | 1,024 | 1,561 | -537 | 42 | 579 | 627 | 1,199 |
| 547 | 1,622 | 2,348 | -726 | 72 | 798 | 875 | 1,731 |
| 574 | 994 | 1,497 | -503 | 38 | 541 | 581 | 1,165 |
| 582 | 993 | 1,374 | -381 | 30 | 411 | 443 | 900 |

## Diagnosis

The primary failure mode is Lane over-claiming into Road. Among introduced errors, PE3 target class `Lane` accounts for `5,997` introduced pixels, and the largest target-vs-GT transition is `Lane -> Road` with `5,557` introduced pixels. This is not a small boundary-offset artifact: PE3 rewrites many already-correct base Road pixels into Lane.

Introduced errors by PE3 target class:

| target class | introduced |
|---|---:|
| Lane | 5,997 |
| Road | 4,151 |
| Undrivable | 2,261 |
| Movable | 1,122 |
| MyCar | 221 |

Fixed pixels by PE3 target class:

| target class | fixed |
|---|---:|
| Movable | 263 |
| Lane | 258 |
| Road | 257 |
| Undrivable | 89 |
| MyCar | 1 |

Largest introduced transitions:

| PE3 target | GT/base class being damaged | introduced pixels |
|---|---|---:|
| Lane | Road | 5,557 |
| Road | Undrivable | 1,497 |
| Undrivable | Movable | 1,194 |
| Road | Movable | 1,154 |
| Undrivable | Road | 1,018 |
| Road | Lane | 759 |
| Road | MyCar | 741 |
| Movable | Undrivable | 560 |
| Movable | Road | 557 |
| Lane | Undrivable | 408 |

Mode attribution:

| PE3 record mode | fixed | introduced | changed labels |
|---|---:|---:|---:|
| generator_pair_bisector | 638 | 12,497 | 13,233 |
| depth_conditioned_curve | 230 | 1,255 | 1,492 |

Interpretation for conditioning:

- PE3 coverage is not label agreement. The grammar identifies many semantically relevant sites, but its WHERE/WHICH labels are miscalibrated for direct target substitution on this qo1 base.
- The useful role is as a conditioning prior for a learned carrier that can decide when to trust the grammar and when to preserve the base class, not as a frozen target table.
- Grammar-v2 should specifically attack Lane-over-Road over-claiming and the generator-pair-bisector target assignment. The depth-conditioned curve records also worsen, but they are a smaller share of the damage.

## Routing Consequences

The pre-registered nonpositive branch fires:

| consequence | LC1 disposition |
|---|---|
| (a) EU2 10K micro-student experiment fires rank-1 | FIRED-TO-RANK-1-QUEUE with EU2's own no-scorer fire order: run the `EU2-X1-10K-context-orderer` cached ordering/context experiment first; no scorer slot is authorized by EU2. |
| (b) LT1's -3,522 B PE3 recode | MOOT-FOR-SHIPPING on the PE3 target-corridor object. It remains a same-raw receiver-format idea only if a future conditioning-only stream consumes PE3 bytes. |
| (c) 114,852 B composed corridor candidate | WITHDRAWN as a shipping shape. The PE3+OD9 corridor cannot ship as a target-label correction when the ideal target-label ceiling is negative. |
| (d) BI1 birth mechanism plus 40,444 B pose carriage | ROUTED as TR1-line assets. BI1's birth seed/amplify path and the OD9/cheapdct4 pose-carriage evidence belong under TR1 learned-carrier follow-up, not PE3 target shipping. |

No positive-branch successor realizer charter was written because the falsifier did not pass. The regional median-prototype realizer remains dead from RZ1; LC1 additionally closes PE3-as-ideal-target for this routing denominator.

Verdict scope: FORMULATION/ROUTING on the NA3-derived stratified n32 PE3 ideal-label ceiling. This is not a global grammar-family kill and not a contest score claim.

## RECALL EVIDENCE

| source searched | query or source | finding beyond charter seeds | plan impact |
|---|---|---|---|
| Governing docs | `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | LC1 owns the scorer slot; exact pointer is borrowed/unmoved; advisory axes cannot be promoted; protected files and staged index must be avoided. | Ran only the bounded n32 advisory pass, edited only this receipt, and left source/protected files untouched. |
| RZ1 producer | `.omx/research/ddm_rz1_20260805/RZ1_RECEIPT.md`, `/Volumes/VertigoDataTier/pact/ddm_rz1_20260805/rz1_pe3_headsolve_smoke_n3.json` | RZ1's n3 label ceiling was -1,300 and explicitly queued an n32 label-ceiling audit before any new PE3 head-solve. | Reused RZ1 machinery but skipped solver/realizer legs. |
| NA3 selection | `.omx/research/ddm_na3_20260805/stratified_pose_selection_923.json`, `ddm_na3_receipt.md` | NA3 supplies seed-20260805 n120 stratified ids; LC1's n32 derived seed is `20260837`; no prefix. | Used RZ1 `select_pairs_from_na3` for deterministic selection. |
| Consequence receipts | `.omx/research/ddm_eu2_20260805/EU2_RECEIPT.md`, `.omx/research/ddm_lt1_20260805/LT1_RECEIPT.md`, `.omx/research/ddm_bi1_20260805/BI1_RECEIPT.md` | EU2 is a no-scorer 10K context-orderer first experiment; LT1's PE3 recode is a receiver-format byte model; BI1 built a default-OFF TR1 birth path but made no score claim. | Enumerated consequences without turning them into fake score movement. |
| Corpus search | `rg -n -I "label-ceiling|ideal PE3|CONDITIONING-ONLY|TR1 learned carrier|micro-student|114,852|bi1 birth"` over `.omx/research`, `.omx/state`, `reports`, `docs`, `src/tac`, `experiments` excluding result blobs | Found no stronger positive PE3 target-label law overriding RZ1/LC1; found matching route language in hot state and adjacent receipts. | Kept the negative as routing-scoped and did not reopen realization after a negative ideal ceiling. |
| Canonical equations | `.venv/bin/python tools/list_canonical_equations.py --json` filtered for `pe3`, `label`, `tr1`, `conditioning`, `micro-student`, `pose_null`, `trajectory` | No direct measured PE3 label-ceiling equation was present; adjacent laws concerned pose-null paint and trajectory stopping, not PE3 target agreement. | Measured the target agreement directly and did not import an unrelated law. |
| Memory registry | `rg -n "ddm_rz1|PE3|LC1|label-ceiling|conditioning-only|TR1-learned|micro-student|ddm_eu2|ddm_lt1|ddm_bi1" /Users/adpena/.codex/memories/MEMORY.md` | No direct memory entry found for LC1/RZ1/PE3 routing. | Used live repo receipts and artifacts as authority. |

## NEXT_IF_RESUMED

```json
{
  "run_id": "ddm_lc1_20260805",
  "status": "MEASURED_NONPROMOTABLE_ADVISORY",
  "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
  "score_claim": false,
  "selection": {
    "source": ".omx/research/ddm_na3_20260805/stratified_pose_selection_923.json",
    "seed": 20260805,
    "derived_seed": 20260837,
    "n": 32,
    "mode": "stratified_blocks_from_na3_n120_seeded_without_prefix",
    "seg_matched_population_factor": 1.0099888594483923
  },
  "label_ceiling_result": {
    "base_flips": 27382,
    "ideal_pe3_label_flips": 40266,
    "net_fixed": -12884,
    "base_d_seg_n32_denominator": 0.004352251688639323,
    "ideal_pe3_label_d_seg_n32_denominator": 0.006400108337402344,
    "all_32_pairs_worsened": true
  },
  "routing": {
    "pe3_as_target_labels": "FOLDED_FORMULATION_ROUTING",
    "pe3_role": "CONDITIONING_ONLY",
    "primary_line": "TR1_LEARNED_CARRIER",
    "contest_pointer": "borrowed_unmoved"
  },
  "follow_ons": {
    "eu2_x1_10k_context_orderer": {
      "disposition": "FIRED_TO_RANK_1_QUEUE",
      "fire_order": "Run cached no-scorer ordering/context byte experiment first; no scorer slot until EU2's packed-byte and cached-savings bar passes."
    },
    "lt1_pe3_minus_3522B_recode": {
      "disposition": "MOOT_FOR_SHIPPING_UNDER_PE3_TARGET_SHAPE",
      "fire_order": "Fold unless a conditioning-only PE3 receiver format needs exact raw PE3 recode integration."
    },
    "pe3_od9_114852B_corridor": {
      "disposition": "WITHDRAWN_AS_SHIPPING_SHAPE",
      "reason": "ideal PE3 target labels worsen d_seg before realization"
    },
    "bi1_birth_plus_od9_pose": {
      "disposition": "ROUTED_TO_TR1_LINE_ASSETS",
      "fire_order": "Use under TR1 learned-carrier A/B and pose-carriage integration, not PE3 target-label shipping."
    }
  },
  "bulk_artifacts": {
    "measurement_json": {
      "path": "/Volumes/VertigoDataTier/pact/ddm_lc1_20260805/lc1_label_ceiling_n32.json",
      "bytes": 82232,
      "sha256": "d1e979759df209f40995ef9d0465024011af5c2171eb33323edba8dfaf892db6"
    },
    "sha_manifest": {
      "path": "/Volumes/VertigoDataTier/pact/ddm_lc1_20260805/SHA256SUMS"
    }
  },
  "do_not_do": [
    "Do not retry PE3 target-label realization without a new positive label-ceiling audit.",
    "Do not ship the 114852 B PE3+OD9 corridor as a target-label correction shape.",
    "Do not promote this n32 advisory routing measurement to contest-CPU/CUDA authority."
  ]
}
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
