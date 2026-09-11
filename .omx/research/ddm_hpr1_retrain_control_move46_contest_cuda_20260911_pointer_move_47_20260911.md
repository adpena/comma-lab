# FORTY-SEVENTH POINTER MOVE — S 0.13654774984742127 @ 179,359 B [contest-CUDA T4 n600]: pointer move 47: S 0.13654774984742127 @ 179,359 B [contest-CUDA T4 n600] — hpr1 retrained HPAC prior in its shipped geometry (-642 B vs move 46; decoded field and raw byte-identical), normal seal inheriting move 46's measured t4_direct leg (2026-09-11)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M29166GTAZC3TA5V8W8Q54E1`. Lane `ddm_hpr1_retrain_control_move46_contest_cuda_20260911`. Modal wall 1043.6 s. Archive sha `d1fab05d69f31c90ac55173fa87072949e5ea1e069a0b7614337089b7a2a0ce9`, 179,359 B. Runtime tree `5170a79833d710a3e0fd8080d470ce8eaf3d2a7350ca2f06e5466f2e2d600c2f`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·179,359/37,545,489 | 0.11942779597303953 |
| seg 100·0.00010345 | 0.010345 |
| pose √(10·4.59e-06) | 0.00677495387438173 |
| **S** | **0.13654774984742127** |

| | ddm_ntb2_frame_even_hpac_prior_move45_first_measurement_20260911 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.1369752312953257 | 0.13654774984742127 | **-0.0004274814479044431** |
| d_seg | 0.00010345 | 0.00010345 | 0.0 |
| d_pose | 4.59e-06 | 4.59e-06 | 0.0 |
| bytes | 180,001 | 179,359 | -642 |

## Projection fidelity

Projected 0.13654774984742127; realized − projected = 0.0. conditional arithmetic from the -642 B retrained prior on move 46's field; realized minus projected within float rounding

## The mechanism

The HPAC prior (the token tail's coder model, shipped in the archive's `hpac` member and materialized identically at both ends)
descends from cl2's move-26 fit; the token field moved at move 32, so for thirteen moves the coder's model was fit to a field that
no longer existed. ddm_hpr1 retrained the mixer in its SHIPPED geometry for 60 epochs (cl2's law), warm-started from the shipped
prior itself: the `hpac` member grows 11,911 → 12,262 B (+351 B; the refitted weights are video-derived and every one is COUNTED in
the archive; no shape bit ships — the field's geometry lives in receiver code and does not move), while the RLC1 stream it
conditions falls 119,749 → 118,511 B (−1,238 B): archive 179,359 B, −887 B against move 45 and −642 B against move 46 (whose
even-rounding of the same frame embedding this retrain replaces; the two edits do not add). Output-lossless by receipt: the decoded
token field is byte-identical to the shipped field (a92e7d90…) and the cold n600 public decode is byte-identical to the pointer's
raw (2b762eba…), so d_seg and d_pose are unchanged and the only score change is the rate term. Receiver unchanged (behavior digest
equal), so the seal inherits move 46's measured T4 leg on the normal path.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.016547749847421273. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.01711995387438173: archive ≤ 154,507.3 B → **-24,851.7 B**.
- **DISTORTION corner** at held bytes 179,359: distortion ≤ 0.0005722 → **29.9× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **859.347 B under** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/VertigoDataTier/pact/ddm_hpr1_fire_move47/run1/MODAL_REMOTE_RESULT.json` (sha `e49a84cb423492840f020c6956563ad5187be7f6765ae72dd9a17849b9f3b024`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_hpr1/public/retrain_control/candidate_runtime/archive.zip` (sha `d1fab05d69f31c90ac55173fa87072949e5ea1e069a0b7614337089b7a2a0ce9`, 179,359 B).
- Seal: `/Users/adpena/Projects/pact/.omx/research/ddm_hpr1_20260911/SEAL_ddm_hpr1_retrain_control_contest_cuda.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_hpr1_retrain_control_move46_contest_cuda_20260911/custody_pointer47/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_hpr1_retrain_control_move46_contest_cuda_20260911/custody_pointer47/SEAL_ddm_hpr1_retrain_control_contest_cuda.json` (sha verified: True).

## What this does NOT claim

Not claimed: any distortion change (field and raw byte-identical); the shape rung (conv_past dilation 2 is FALSIFIED at exact bytes,
+812 B against this control; its joint-information predictor is refuted as a rung predictor on this object; verdict scope
formulation); additivity with move 46's rounding (same object; the retrain supersedes it; the rounding re-applied on top of the
retrained prior is the NEXT row, unmeasured); a T4 wall-clock claim of its own (inherited from move 46's measured leg; local
ns/symbol unchanged); the CPU axis (declaration + refusal receipt); any transfer of cl2's +0.446 capacity secant to this move
(the retrain adds 351 model bytes and buys 1,238 tail bytes — a refit, not a capacity rung).

## Next from here

Next: the even-rounding (move 46's lever) re-applied on top of the retrained prior, priced by the real coder; then hpr1's R6
(conv_a cone dilation, the cheapest unfired shape rung, a receiver change → first-measurement chain) and the temporal-axis tap sets
its atlas found positive — both measured against the RETRAINED control, never the shipped one (their predictor is refuted, so their
signs are unsupported, not merely unfired). Law carried forward: refit every model section after a field change before pricing any
structural rung on it (pc3 measured the dual: rung prices expire at a pointer move). The pose corner on this carrier is closed at
n600 (349.9 B family slack). Rate corner at this move: cap 154,507 B at held distortion; demand = archive − 154,507.

Equations leg (`tac.canonical_equations`): S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489 with d_seg, d_pose identical to move 46 (decoded field and cold n600 raw byte-identical) and B 180,001 → 179,359: ΔS = 25·(−642)/37,545,489 = −4.2748145e-4 exactly; realized 0.13654774984742127 vs projected 0.13654774984742127 (float rounding)

Own-vehicle frontier: **S 0.13654774984742127 @ 179,359 B [contest-CUDA T4 n600]**, archive sha `d1fab05d…a0ce9`.
