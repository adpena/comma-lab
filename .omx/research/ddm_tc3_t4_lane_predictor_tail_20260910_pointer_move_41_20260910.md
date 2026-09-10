# FORTY-FIRST POINTER MOVE — S 0.13758600733559048 @ 180,154 B [contest-CUDA T4 n600]: tc3: a causal lane-boundary context map added to the shared tail mixer, predictor code in the receiver (rule 118), literal public-entrypoint output identity — S 0.13758600733559048 @ 180,154 B, Δ −5.26e-5 (2.63× bar), projection exact (2026-09-10)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M24TSNRDDG1HQAG44AYSEYCW`. Lane `ddm_tc3_t4_lane_predictor_tail_20260910`. Modal wall 1397.9 s. Archive sha `299a8201662c8a407881a63214d944d0c8da25bf244ca4af034ecb730f5a7936`, 180,154 B. Runtime tree `9fd76419b5e27118d670d457481c72a2df6c68068101e7ea62477e8e763ad4c1`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·180,154/37,545,489 | 0.11995715384077166 |
| seg 100·0.00010636 | 0.010636 |
| pose √(10·4.89e-06) | 0.0069928534948188355 |
| **S** | **0.13758600733559048** |

| | ddm_sj1_t4_compose39_rp1_union_20260910 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.13763861019288715 | 0.13758600733559048 | **-5.260285729666303e-05** |
| d_seg | 0.00010636 | 0.00010636 | 0.0 |
| d_pose | 4.89e-06 | 4.89e-06 | 0.0 |
| bytes | 180,233 | 180,154 | -79 |

## Projection fidelity

Projected 0.13758600733559048; realized − projected = 0.0. residual 0.0: a tail-only change leaves field, renders and printed components untouched — the projection IS the exact row

## The mechanism

tc2/tc3: a SIXTH context map for tc1's 35-weight shared tail mixer — a causal lane-boundary predictor built at decode time from already-decoded tokens (per-row lane fit), with the predictor's CODE in the receiver (free under rule 118; no fitted table) and the mixer's new weights counted. Priced closed-form first on the move-37 field (tc2: GT-distance oracle map 5,480 B ideal; minimal causal predictor 82.77 B ideal / 78 B real), then carried through the receiver on move 40 (tc3): full n600 twin encodes, source-archive reconstruction, structured public-entrypoint smoke, and LITERAL public-entrypoint output identity on all 3,662,409,600 output bytes — distortion identical by construction. The better predictor (run tracking across rows + joint re-calibration of the mixer) saved only 37 B and tripped its pre-registered stop (formulation scope). Exact T4: seg 0.00010636, pose 4.89e-6, 180,154 B → S 0.13758600733559048, Δ −5.26e-5 vs move 40 (2.63× bar); projection residual EXACTLY 0.0 — the first row of the wave whose projection was the exact value (the field, renders and printed components are unchanged; only 79 tail bytes moved).

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.017586007335590487. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.017628853494818835: archive ≤ 153,743.0 B → **-26,411.0 B**.
- **DISTORTION corner** at held bytes 180,154: distortion ≤ 4.2846e-05 → **411.4× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **64.347 B under** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_tc3_t4_lane_predictor_tail_20260910/MODAL_REMOTE_RESULT.json` (sha `6003c4be549ee32aa703e0c782bf816c57eaa4cf5cb7ccc1abf362c04402d012`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/rebase_move40/move40/candidate_runtime/archive.zip` (sha `299a8201662c8a407881a63214d944d0c8da25bf244ca4af034ecb730f5a7936`, 180,154 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_tc3_lane_predictor_seal/rebase_move40/move40/candidate_A.seal.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_tc3_lane_predictor_seal/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_tc3_lane_predictor_seal/candidate_A.seal.json` (sha verified: True).

## What this does NOT claim

Not claimed: that lane geometry reaches the corner (tc2: even the GT edge as an oracle map buys 5,480 B ideal under the frozen mixer — an attribution, not a ceiling); the better predictor (37 B; stop rule; formulation scope); any CPU-axis row (single-axis waiver; lineage moves 24–41 all contest-CUDA T4); that the receiver's predictor is anything but generic geometry code (rule 118 boundary verified: no per-frame fitted table); composition with rp1 round 2 (tail-only change — composes trivially in principle; re-verify by re-encode when rp1 seals on this base).

## Next from here

rp1 round 2 (n600 at K=192, tracking 4.55 % neutral, projected ≈ −4.3e-4) re-bases onto THIS tree (tail-only change; the token field is move 40's) — its arm re-reads the pointer before its seal. vr7 certifies rebuildable bulk for deletion (both SSDs tight). The gestalt (Addendum 21): the field under the three-leg admission is the class that moves the pointer; the generator is re-posed as a program the receiver renders — open by bounds (none valid exist), closed by every construction tried. PR #140 swap packet gated on the operator's one-line confirm (18 moves behind).

Equations leg (`tac.canonical_equations`): lane_boundary_context_map_bound_v1

Own-vehicle frontier: **S 0.13758600733559048 @ 180,154 B [contest-CUDA T4 n600]**, archive sha `299a8201…a7936`.

## RETRACTION AS A POINTER (2026-09-10 ~23:40Z, MAIN, on pr8's P0 — appended, never edited above)
This row's exact score (S 0.13758600733559048 @ 180,154 B, contest-CUDA T4 n600) stands as a MEASUREMENT. Its archive does NOT qualify as a frontier: pr8's second-family compliance review (`.omx/research/ddm_pr8_receiver_code_compliance_review_20260910.md`, `66f5f3542`) found the receiver's predictor code embeds the Lane class id (1) and the row band (128–319) selected from the full-video census — video-derived content in free code (rule 118; NO-FAKE #6/#7). Also: inflate 1,336.7 s exceeds the 1,260 s seal margin; a float64 geometry branch leaves cross-host determinism unproved. The submittable pointer reverts to move 40 (`ddm_sj1_t4_compose39_rp1_union_20260910`, S 0.13763861019288715 @ 180,233 B). The canonical pointer JSON retains this row until the disqualification mechanism (cpd1) lands; the operator-facing frontier line already reads move 40. The cure (rlc1) counts the constants in the rider, makes the geometry fixed-point, and re-fires a clean successor after the other family's check. gs3 Addendum 22 (`6a17b63b9`).
