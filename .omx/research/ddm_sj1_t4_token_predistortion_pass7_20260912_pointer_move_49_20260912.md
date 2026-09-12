# FORTY-NINTH POINTER MOVE — S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600]: pointer move 49: S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600] — sj1 pass 7 seg-Lagrange subset (42 pairs / 69 cells / 67 tokens; +42 B; d_seg 0.00010287, d_pose 4.55e-6) on the move-48 field, normal seal inheriting move 48's measured leg (2026-09-12)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M2AZ6G51VZH39QQ0XQGPM84M`. Lane `ddm_sj1_t4_token_predistortion_pass7_20260912`. Modal wall 1161.6 s. Archive sha `73e41a6620bd4ea3aaf236eff9de46391857907527358e8eb40ded0925a1c214`, 179,153 B. Runtime tree `8f3c50266bebbca440af0690bfe0976e8096d420db3804ed72199d6a6aa64707`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·179,153/37,545,489 | 0.11929062902869636 |
| seg 100·0.00010287 | 0.010287 |
| pose √(10·4.55e-06) | 0.00674536878161602 |
| **S** | **0.13632299781031237** |

| | ddm_hpr1_comp_even_on_refit_move47_first_measurement_20260911 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.13638261682704697 | 0.13632299781031237 | **-5.9619016734596686e-05** |
| d_seg | 0.00010345 | 0.00010287 | -5.799999999999978e-07 |
| d_pose | 4.59e-06 | 4.55e-06 | -4.0000000000000464e-08 |
| bytes | 179,111 | 179,153 | +42 |

## Projection fidelity

Projected 0.1363177011168108; realized − projected = 5.296693501577465e-06. three-leg Lagrange admission on the resolved pose; realized minus projected +5.3e-6 inside the seal's declared spread; pass 7 yield 1.812 % > 1 %/pass so the family stays open

## The mechanism

Token pre-distortion pass 7 (seg-Lagrange subset) on the move-48 field. The search re-ran sj1's single-token family over all 600 pairs against the move-48 receiver and prior (the pricer rebound to move 48's own tail coder, proved by byte-identical repack of the pointer, twice) and found 221 repairable cells by 218 token moves on 130 pairs = 1.812 % of the 12,196-cell residual (above the family's 1 %/pass convergence rule). The three-leg Lagrange admission on the RESOLVED pose (carrier re-solved on the edited field; frame-0 repair inside the admission) kept 42 pairs / 69 cells / 67 tokens: seg 12,196 → 12,127 cells on the candidate's own cold parse-back decode; d_pose 4.59e-6 → 4.55e-6 (a credit: the full 130-pair field resolves to a pose COST and loses); token stream 118,896 → 118,938 B (+42 B, real encode, twins agreeing) at 5.01 bits/token against a 10.49 break-even. Receiver code byte-identical to move 48 (only archive.zip and its two pins differ), so the seal inherited move 48's MEASURED t4_direct leg (1,023 s ≤ 1,260 s ceiling) on the normal path. Exact T4 row: d_seg 0.00010287, d_pose 4.55e-6, 179,153 B; realized minus projected +5.3e-6 (the T4 pose-print class, inside the seal's 2.4e-6…1e-5 spread).

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.01632299781031238. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.01703236878161602: archive ≤ 154,638.8 B → **-24,514.2 B**.
- **DISTORTION corner** at held bytes 179,153: distortion ≤ 0.00070937 → **24.0× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **1,065.347 B under** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7_fire/run1/MODAL_REMOTE_RESULT.json` (sha `f3dff9fc37a764c8f4cf104df14c70af62ebfff1e5ebba73830607bb826319e4`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/candidate/candidate_runtime/archive.zip` (sha `73e41a6620bd4ea3aaf236eff9de46391857907527358e8eb40ded0925a1c214`, 179,153 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/SEAL_ddm_sj1_token_predistortion_pass7_contest_cuda.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_sj1_t4_token_predistortion_pass7_20260912/custody_pointer49/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_sj1_t4_token_predistortion_pass7_20260912/custody_pointer49/SEAL_ddm_sj1_token_predistortion_pass7_contest_cuda.json` (sha verified: True).

## What this does NOT claim

Not claimed: any CPU-axis score (the receiver declares linux-nvidia-t4; the CPU sibling is a declaration + typed refusal receipt, fired after this harvest). Not claimed: closure of the pre-distortion family on this field — pass 7's 1.812 % is above the 1 %/pass rule, so pass 8 is owed on the move-49 field. Not claimed: that the 154 pass-6-dropped positions re-found here are new information (F11 pre-registered them; 154/154 re-found, zero new on stale pairs). Not claimed: any renderer/realization/prior/mixer rung (all measured closed at the joint knee: ren2, rq1, dpi1, tmx1). Not claimed: a guard for the named class defect (patch_inflate_pins leaves MANIFEST.sha256 stale) — fixed for this instance only.

## Next from here

Next: (1) CPU-axis sibling of this exact archive (declaration + refusal receipt, claim-policy open, after this harvest). (2) sj1 pass 8 on the move-49 field: the family is not closed (1.812 % > 1 %); the re-render reach law says repairs consume local slack (0.41 cells/pair on repaired pairs vs 1.156 on neutral re-renders), so expect a smaller yield; same three-leg admission on the resolved pose. (3) Land the MANIFEST.sha256 guard for patch_inflate_pins (class defect named by pass 7). (4) PR #140: restage on the final pointer once the operator answers A/B (rih1's packet). Rate corner at this move: cap 154,638.8 B; demand −24,514 B at held distortion. The object stays at its measured joint knee on every non-field axis.

Equations leg (`tac.canonical_equations`): S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489: 100·0.00010287 + sqrt(4.55e-5) + 25·179,153/37,545,489 = 0.13632299781031237; Δ vs move 48 = −5.9619e-5 (seg −5.8e-5, pose −2.96e-5 from the resolved carrier, rate +2.797e-5 for +42 B); projected −6.49e-5, realized −5.96e-5, gap +5.3e-6 = the T4 pose-print class

Own-vehicle frontier: **S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600]**, archive sha `73e41a66…1c214`.
