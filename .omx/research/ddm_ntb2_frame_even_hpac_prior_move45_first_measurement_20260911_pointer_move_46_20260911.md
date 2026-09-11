# FORTY-SIXTH POINTER MOVE — S 0.1369752312953257 @ 180,001 B [contest-CUDA T4 n600]: pointer move 46: S 0.1369752312953257 @ 180,001 B [contest-CUDA T4 n600] — ntb2 output-lossless HPAC prior rounding (-245 B; decoded field and raw byte-identical to move 45), first-measurement chain with a measured t4_direct leg of 1,140.8 s (2026-09-11)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M28Z2YVB6FA7C1QPA6HPWR6D`. Lane `ddm_ntb2_frame_even_hpac_prior_move45_first_measurement_20260911`. Modal wall 1200.4 s. Archive sha `a0de607df2ff566d4eb6fc43031179f5450f0cb622803d4ce71b987f8970d720`, 180,001 B. Runtime tree `8874e8b5e6c6bc99e7b72cae645ca3fcfb127e566a6b2d30db6bd67b478e16cc`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·180,001/37,545,489 | 0.11985527742094397 |
| seg 100·0.00010345 | 0.010345 |
| pose √(10·4.59e-06) | 0.00677495387438173 |
| **S** | **0.1369752312953257** |

| | ddm_pc3_cap1_predictor_refit_move44_contest_cuda_20260911 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.1371383667388406 | 0.1369752312953257 | **-0.00016313544351490017** |
| d_seg | 0.00010345 | 0.00010345 | 0.0 |
| d_pose | 4.59e-06 | 4.59e-06 | 0.0 |
| bytes | 180,246 | 180,001 | -245 |

## Projection fidelity

Projected 0.13697523129532568; realized − projected = 2.7755575615628914e-17. conditional arithmetic from the -245 B output-lossless prior rounding on move 45's field; realized minus projected within float rounding

## The mechanism

The HPAC section (11,911 B on move 45) is the coder's PRIOR for the token tail: a small model whose bytes are shipped in the archive
and materialized identically at both ends before any symbol is decoded, so it conditions the tail's code length without touching
a single decoded symbol. ddm_ntb2 rounded 2,320 of the prior's 4,800 frame-embedding values to the nearest even value: the HPAC
member shrinks 603 B (11,911 → 11,308) while the tail it conditions grows 358 B (119,749 → 120,107 stream), net −245 B, archive
180,246 → 180,001 B. Output-lossless by construction and by receipt: the decoded token field is byte-identical to the shipped
field, and the cold n600 public decode is byte-identical to the pointer's own raw (2b762eba…, 600 pairs, cache disabled), so
d_seg and d_pose are move 45's exactly and the only score change is the rate term, 25·(−245)/37,545,489 = −1.6314e-4. The receiver
is unchanged (behavior digest 9f6e7168… equal to move 44/45's); the move went through the first-measurement chain only because an
inherited decode-wall-clock leg cannot be inherited a second time, and its completion gives move 46 a measured T4 leg of its own.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.016975231295325716. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.01711995387438173: archive ≤ 154,507.3 B → **-25,493.7 B**.
- **DISTORTION corner** at held bytes 180,001: distortion ≤ 0.00014472 → **118.3× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **217.347 B under** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/VertigoDataTier/pact/ddm_ntb2_first_measurement/run2/MODAL_REMOTE_RESULT.json` (sha `19eac2eee6fe8b1e04bbf5f9086608f70a81c970d2b2e146e24d42197355d28c`).
- Archive: `/Volumes/APDataStore/pact/ddm_ntb2_proof45/frame_even/candidate_runtime/archive.zip` (sha `a0de607df2ff566d4eb6fc43031179f5450f0cb622803d4ce71b987f8970d720`, 180,001 B).
- Seal: `/Users/adpena/Projects/pact/.omx/research/ddm_ntb2_20260911/SEAL_ddm_ntb2_frame_even_hpac_prior_move45_contest_cuda_v3.json`.
- Second copy: `/Volumes/VertigoDataTier/pact/ddm_ntb2_frame_even_hpac_prior_move45_first_measurement_20260911/custody_pointer46/archive.zip` (sha verified: True).
- Second copy: `/Volumes/VertigoDataTier/pact/ddm_ntb2_frame_even_hpac_prior_move45_first_measurement_20260911/custody_pointer46/SEAL_ddm_ntb2_frame_even_hpac_prior_move45_contest_cuda_v3.json` (sha verified: True).

## What this does NOT claim

Not claimed: any distortion change (decoded field and raw byte-identical); a lossless-coding law beyond this prior (the ls1/ls2
atlas closed the context corner; mxo3 closed the stacking corner); frame_quad (step-4 rounding, 3,548 values) — a follow-on row,
unmeasured at n600; the row-drop treatments (+18/+26 B, lose); any additivity with hpr1's retrained-prior control (−887 B on move 45):
rounding and retraining are two edits to the SAME object (the frame embedding) and do not sum — the retrain supersedes the rounding
and is sequenced as move 47; the CPU axis (declaration + refusal receipt per the move-44/45 pattern); the local timing ratio as
authority (sequential windows differed 20.9 % on identical bytes; the matched concurrent pair, candidate 4.6 % faster, clamped the
projected fraction to 0 and the T4 leg is the measurement).

## Next from here

Next: hpr1's retrained HPAC prior (shipped geometry, 60 epochs under cl2's law; 179,359 B on move 45, −887 B, projected S 0.13654775,
receiver unchanged, output-lossless) rebases onto move 46 by normal seal inheriting this move's measured T4 leg → move 47; then the
even-rounding re-applied on top of the retrained prior if it still prices negative; then hpr1's R6 (conv_a cone dilation, the
cheapest unfired shape rung, a receiver change → first-measurement chain) and the temporal-axis tap sets its atlas found positive.
The law this move surfaced: a shipped prior fit to an older field (cl2's move-26 fit; the field moved at move 32) codes a field it
was never fit to — refit every model section after a field change. Storage: both SSD tiers above reserve after sr5; the local disk
is not a tier. pr19 (pre-written): whether an empty receiver delta may satisfy the risk gate without a local ratio, and whether
inheritance may follow a chain that shares one behavior digest — spawn only if a future rate-only move is blocked again.

Equations leg (`tac.canonical_equations`): S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489 with d_seg, d_pose identical to move 45 (decoded field and cold n600 raw byte-identical) and B 180,246 → 180,001: ΔS = 25·(−245)/37,545,489 = −1.6313544e-4 exactly; realized 0.1369752312953257 vs projected 0.13697523129532568 (float rounding); T4 inflate 1,140.8 s measured (t4_direct)

Own-vehicle frontier: **S 0.1369752312953257 @ 180,001 B [contest-CUDA T4 n600]**, archive sha `a0de607d…0d720`.
