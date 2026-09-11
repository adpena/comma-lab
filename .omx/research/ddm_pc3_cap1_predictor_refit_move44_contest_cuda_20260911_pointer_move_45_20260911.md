# FORTY-FIFTH POINTER MOVE — S 0.1371383667388406 @ 180,246 B [contest-CUDA T4 n600]: pointer move 45: S 0.1371383667388406 @ 180,246 B [contest-CUDA T4 n600] — pc3 pose-carrier predictor refit at bit-identical codes (-160 B; decode byte-identical to move 44), normal seal with inherited t4_direct through pr18's behavior digest (2026-09-11)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M28K31SPRPMJ1A9JYY9W42YB`. Lane `ddm_pc3_cap1_predictor_refit_move44_contest_cuda_20260911`. Modal wall 1293.6 s. Archive sha `145e02e21f9a1cbc8276d1ecc34f0b9ae4762afa3fea7811e5836fee770ae60a`, 180,246 B. Runtime tree `2d4dd13352b50a27f3d1b423261665c0f90e250781f07c2fefe86c46656dc967`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·180,246/37,545,489 | 0.1200184128644589 |
| seg 100·0.00010345 | 0.010345 |
| pose √(10·4.59e-06) | 0.00677495387438173 |
| **S** | **0.1371383667388406** |

| | ddm_rlc5_counted_rider_rebase_move43_first_measurement_20260910 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.1372449041713402 | 0.1371383667388406 | **-0.00010653743249958159** |
| d_seg | 0.00010345 | 0.00010345 | 0.0 |
| d_pose | 4.59e-06 | 4.59e-06 | 0.0 |
| bytes | 180,406 | 180,246 | -160 |

## Projection fidelity

Projected 0.13713836673884064; realized − projected = -2.7755575615628914e-17. conditional arithmetic from the -160 B predictor refit on move 44's raw-identical decode; realized minus projected within float rounding

## The mechanism

The shipped pose carrier (CPR1) stores an AR(1)+bias predictor whose residuals are Rice-coded. The shipped predictor was a
clipped fit: ten of its twelve biases sat at the ±16 clamp, so it was never rate-optimised. ddm_pc3 refit the predictor by an
exhaustive search over the closed legal schema at BIT-IDENTICAL codes (51,581 → 50,270 Rice bits); `decode_cap1` reconstructs the
byte-identical canonical CPR1 either way, so the decoded carrier, the token plane, and every rendered frame are unchanged. Archive
180,406 → 180,246 B (−160 B). The cold n600 public decode is byte-identical to move 44's retained raw across all 3,662,409,600 bytes,
so d_seg and d_pose are move 44's by construction; the only score change is the rate term, 25·(−160)/37,545,489 = −1.0654e-4.
The refit's fitted values live in the counted archive (rule 118: content, not code); the receiver is unchanged after pin
normalization, which is why the seal inherits move 44's t4_direct leg through the pr18 behavior digest (49 rows, equal).

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.017138366738840616. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.01711995387438173: archive ≤ 154,507.3 B → **-25,738.7 B**.
- **DISTORTION corner** at held bytes 180,246: distortion ≤ -1.8413e-05 → **inf× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **-27.653 B over** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_pc3_fire_move45/run1/MODAL_REMOTE_RESULT.json` (sha `4bcc312921382a123ccd6ffb163b03390685a771f5183435977669dd6b710b36`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_pc3_pose_carrier_curve/candidate/candidate_runtime/archive.zip` (sha `145e02e21f9a1cbc8276d1ecc34f0b9ae4762afa3fea7811e5836fee770ae60a`, 180,246 B).
- Seal: `/Users/adpena/Projects/pact/.omx/research/ddm_pc3_20260911/SEAL_ddm_pc3_cap1_predictor_refit_move44_contest_cuda.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_pc3_cap1_predictor_refit_move44_contest_cuda_20260911/custody_pointer45/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_pc3_cap1_predictor_refit_move44_contest_cuda_20260911/custody_pointer45/SEAL_ddm_pc3_cap1_predictor_refit_move44_contest_cuda.json` (sha verified: True).

## What this does NOT claim

Not claimed: any distortion change (the decode is byte-identical); any capacity gain in the pose carrier (every capacity rung —
per-dimension ÷2, global ÷2/÷4/÷8, rank ×2 — needs more d_pose improvement than the continuous-coefficient bound allows and is dead
by arithmetic at move 44's operating point); a T4 wall-clock measurement (the leg is inherited; the local timing pair was
paired-concurrent under sister load and is diagnostic only); the CPU axis (declaration + refusal receipt per the move-44 packet;
no contest-CPU row); composition with ntb2's HPAC candidate (sequenced as a rebase, not composed); any claim that the "T4 pose-print
gap" is anything but an instrument artifact of sj1's overlay re-render (90/600 pairs differ, all higher) at one-row scope.

## Next from here

Next: ntb2's output-lossless HPAC prior candidate (frame_even, −245 B on move 44) rebases onto move 45 by a section-local re-encode
against this pointer's tail plus its own cold n600 parse-back, then seals on the same normal path → move 46. pr18's behavior digest
keeps the inherit path open for every rate-only successor; the manifest is validated independently. The storage reserve (Vertigo
below 40 GiB) is the live blocker for cold decodes: certified moves only, nothing deleted without a cert row. The successor-object
family (obx2 design A) and free post-render treatment (rbf1) and the free decode-time prior (gpp1) are closed at formulation scope;
the coded-token object with lossy pre-distortion and unpriced rate levers (clipped fits, mis-rounded priors) remains the vehicle.

Equations leg (`tac.canonical_equations`): S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489 with d_seg, d_pose identical to move 44 (cold n600 raw byte-identical, 3,662,409,600 bytes) and B 180,406 → 180,246: ΔS = 25·(−160)/37,545,489 = −1.0653743e-4 exactly; realized 0.1371383667388406 vs projected 0.13713836673884064 (float rounding)

Own-vehicle frontier: **S 0.1371383667388406 @ 180,246 B [contest-CUDA T4 n600]**, archive sha `145e02e2…ae60a`.
