# ddm_td1 token-drop guided surface verdict - 2026-08-04

## Scope

Task: `td1 - TOKEN-DROP AT OPTIMAL FORM (#869)`.

Baseline: `qo1 sub_auto_pairbit`, `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`, archive
`d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a`.

Scorer-slot boundary: td1 did not own the scorer slot (`sq2 [SCORER]` was live), so no SegNet/PoseNet
forward pass and no full-n600 evaluation were run. Distortion rows below are `INFERRED_NOT_A_SCORE`,
calibrated only against the measured rt1 n600 negative. Full-n600 promotion is folded for this unit.

## Measured Artifacts

- Runner: `experiments/ddm_td1_token_drop_guided_surface.py`
- Surface receipt: `.omx/research/ddm_td1_20260804/td1_surface_receipt.json`
- Cell-rung ledger: `.omx/research/ddm_td1_20260804/td1_cell_rung_ledger.jsonl` (`3072` rows)
- Raw manifest: `.omx/research/ddm_td1_20260804/td1_inflated_raw_manifest.json`
- SSD map bundle: `/Volumes/VertigoDataTier/pact/ddm_td1_20260804/td1_candidate_maps_and_fields.npz`

The live qo1 token bulk re-encoded exactly and matched `/Volumes/VertigoDataTier/pact/ddm_br1_20260803/cx1_tokens.npy`
byte-for-byte as a token lattice (`f6055f3159ce080f1d8b89dce618d7286d11bd3853d6bb0380cba9fac9801146`).

## Coder x Drop Surface

Measured bytes are real staged `archive.zip` bytes. R7 rows are real `ddm_r7_token_coder` token-frame
bytes on the same candidate token fields; the shipped candidate remains IX2.

| candidate | levels | archive B | delta B vs qo1 | rate delta S | IX2 token B | R7 smevr B | R7 brotli11 B | receiver-close |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `td1_joint_guard_16_14_12_10` | 10:384, 12:160, 14:128, 16:96 | 334,401 | -23,435 | -0.0156044046 | 317,860 | 328,859 | 395,392 | PASS rc=0, 3,662,409,600 B |
| `td1_joint_guard_16_14_12_8` | 8:384, 12:192, 14:96, 16:96 | 332,119 | -25,717 | -0.0171238947 | 315,578 | 325,037 | 393,938 | PASS rc=0, 3,662,409,600 B |
| `global_L14` | 14:768 | 334,221 | -23,615 | -0.0157242592 | 317,680 | 323,189 | 397,048 | PASS rc=0, 3,662,409,600 B |
| `td1_joint_guard_16_12_8_4` | 4:552, 8:87, 12:86, 16:43 | 246,371 | -111,465 | -0.0742199682 | 229,830 | 249,405 | 304,126 | not selected for inflate |

Base token coder controls: IX2 `341,295 B`; R7 smevr `346,478 B`; R7 brotli11 `396,442 B`.

## Projection Gate

Projection is not a score. It scales td1 cell exposure by the measured rt1 n600 negative
(`B_rt1_margin_16_12_8_4_n600_eval_receipt.json`: `d_seg = 0.00515854`, `d_pose = 0.16815221`,
`S = 1.9753490686354727`).

| candidate | projected S | projected delta vs qo1 | projected pose-term erosion | R8 |
|---|---:|---:|---:|---|
| `td1_joint_guard_16_14_12_10` | 1.3600178646 | +0.6060371350 | +0.6056866443 | FAIL |
| `td1_joint_guard_16_14_12_8` | 1.3874680519 | +0.6334873222 | +0.6340520754 | FAIL |
| `global_L14` | 1.3887461200 | +0.6347653904 | +0.5819570636 | FAIL |
| `td1_joint_guard_16_12_8_4` | 1.9667946126 | +1.2128138829 | +1.1973923067 | FAIL |

Typed verdict: `FORMULATION/INSTANCE NEGATIVE for these four token-drop maps as scorer candidates`.
The family remains open because td1 did not run the queued <=32-pair matched scorer subset, and these are
not full-n600 distortion rows.

## NEXT-IF-RESUMED

If the operator wants to challenge the rt1-calibrated pose projection, run a scorer-owned <=32-pair
matched-base subset only on `td1_joint_guard_16_14_12_10` and qo1 base first. Reject immediately if
pose-term erosion exceeds `0.005` or if projected S cannot beat `0.7539807296911207`; append a full-n600
scorer-batch spec only after that subset passes.

Own-vehicle frontier remains `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`.
