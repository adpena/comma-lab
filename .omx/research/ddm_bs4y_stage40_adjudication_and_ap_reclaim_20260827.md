# bs4y stage_40 adjudication + AP reclaim round 3 — 2026-08-27

**Author:** MAIN. **Status:** ADJUDICATED (advisory measurement, instance scope) + reclaim
plan of record. **Verdict scope:** INSTANCE (this resolved-carrier object on the dx2/gb1
lineage); consistent with the FAMILY-level sharp-optimum law measured five ways.

## 1. What happened overnight (2026-08-26 evening → 08-27 03:28Z)

The bs4y stage chain self-advanced after the ai1 SR3 compression (19:59–20:52, certificate
+ verification complete at `/Volumes/APDataStore/pact/ddm_ai1_20260809/`) freed AP space:
its storage preflight re-passed (r2, 21:21) and it ran stage_10 (exact born-small masters)
→ stage_20 (QS5 exact pair solves) → stage_30 (resolved carrier container) → **stage_40
three-way measurement (22:28)**. Final exit `bs4y_stages_1_4_r5.done` rc=0 at
2026-08-27T03:28:54Z. The "stages 1–4" plan ends at stage_40: the chain is TERMINAL, not
interrupted. All materialized payloads retained (`all_materialized_payloads_retained:
true`) — the P0 keep-the-payload law held; the retained stage_20 solve payloads are the
~60 GB that re-closed the AP storage gate.

## 2. The measurement (the decisive artifact)

`checkpoints/stage_40_three_way_measurement.json`, schema
`ddm_bs4y_stage_40_three_way_measurement.v1`, status MEASURED, axis
`[macOS-CPU advisory, seeded uniform random n=32 from n600] NON-PROMOTABLE` (n=20 measured
pairs, seeded random — NOT a prefix; score_claim false):

| leg | s (n20 advisory) | d_seg | d_pose |
|---|---|---|---|
| gb1_dx2_base (control) | 0.196344 | 0.0003265 | 0.000190 |
| born_small_stale_carrier | 4.004435 | 0.011881 | 0.727 |
| born_small_fresh_solve | 3.650975 | 0.011881 | 0.549 |

Adjudication fields: `distortion_s_delta_fresh_minus_base = +3.4546` ·
`perfect_pose_floor_distortion_s = 1.2317` (vs base distortion ≈ 0.0763) ·
`pose_gap_recovered_fraction = 0.1332`.

## 3. Adjudication

1. **The born-small RESOLVED-CARRIER object is REFUSED on distortion at instance scope.**
   Fresh QS5 compensation recovers only 13.3% of the stale-carrier pose gap, and even at
   the PERFECT-POSE floor the object's seg damage alone (+1.155 S over base; d_seg 36×
   base, identical across both carrier legs because the carrier is frame-0-only — the seg
   damage lives in the born-small BODY, not the carrier) is **28× underwater** against the
   rb1 byte-feasible rate credit (≈0.0406 S at 119,175 B) and **47× underwater** against
   bs2's own 36,858 B credit ceiling (0.0245 S).
2. **The stage_50 learned-implicit screen fails its own live-gate arithmetic.** Its gate
   was "retained stage-4 same-instrument measurements and real-byte arithmetic leave the
   learned screen live." They do not: the screen would need to remove >96% of the seg
   damage AND all pose damage to break even — against the sharp-optimum law now measured
   in the same direction by bo2 (209×), nr1 (349×), and this row (46× on total
   distortion). The bs4x/bs4y chain is CLOSED at this carrier instance.
3. **What this does NOT close:** the w96b aligned-diagonal fire (trained-renderer exact
   expected-flip margin law — a DIFFERENT object, the #1215 unentered cell) and the rb1
   four sealed trained-renderer configs (byte-feasible D56 119,175 B / F64 119,904 B, both
   ADMIT vs the 137,986 B sub-0.12 bar). Their open question is distortion under
   TRAINING, which no re-solve measurement answers. Both remain the live routes.

## 4. Storage disposition (certify-or-block, never delete)

The bs4y retained payloads (~60 GB under `retained/bs4y/stage_20/`) stay UNTOUCHED this
round: the tree is protected in the certifier (`PROTECTED_TREES`), <24 h old, and the
route's terminal record (this memo) should season before any compression adjudication.

**Reclaim round 3 instead compresses four CLOSED, >24 h-old, unprotected AP trees** via
the landed `experiments/ddm_sr3_ap_certify_compress_reclaim.py` (manifest-hash → tar.zst
→ verify → remove originals; certificate + verification receipts in-tree; measured law
~2.56× on raw-array trees), smallest-first so each rung's net gain funds the next rung's
in-tree archive scratch (AP starts at ~9.0 GiB free, abort floor 2 GiB):

| order | tree | size | closure citation |
|---|---|---|---|
| 1 | ddm_hm1_failed_source_repo_20260810 | 13 GB | ddm_hm1_model_byte_derivative_20260816.md (failed source-repo clone, reconstructable from origin) |
| 2 | ddm_tv2_tolerance_curve | 18 GB | ddm_tv2_evaluator_tolerance_curve_20260824.md (tolerance family CLOSED both ends, #1255/#1257) |
| 3 | ddm_tv1_tolerance_curve | 48 GB | ddm_tv1_evaluator_tolerance_curve_20260824.md (same closure) |
| 4 | ddm_rx2_current_mc36_label_hpac | 55 GB | ddm_rx2_mc36_label_hpac_20260814.md (MC36-label HPAC race lineage, superseded by dx2/gb1) |

Projected free after rung 3 ≈ 58 GiB → **w96b gate (33,569,378,304 B) GREEN**; after rung
4 ≈ 91 GiB → **rb1/bs4x-class gate (60,380,026,816 B) GREEN** (bs4x itself now moot per
§3). Fire order on green: w96b sealed seeds 20260815 → 20260816 per
`SEALED_FIRE_ORDER_W96B.json`, then rb1's four configs.

## 5. Cross-refs

#1304 (this chain) · #1247/#1262 (bs2/bo2) · #1215 (the 2×2 diagonal) · #1187 (nr1) ·
m124 (demand reads two ways) · m144 (gestalt: different object required) ·
`.omx/research/ddm_sr3_ap_certify_compress_reclaim_20260826.md` (the certifier's landing).
