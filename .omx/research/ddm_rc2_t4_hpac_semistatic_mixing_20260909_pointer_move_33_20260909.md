# THIRTY-THIRD POINTER MOVE — S 0.13885056455024844 @ 181,414 B [contest-CUDA T4 n600]: rc2 shared logistic depth-mixing prior on the IHS1 model rows: −231 B at zero distortion (33rd pointer move) (2026-09-09)

Tokens: `[no-triality] [p0-ledger-ok]`

## The row (exact, authority)

`upstream/evaluate.py`, Tesla T4, 600 samples, axis `contest_cuda`. Modal call `fc-01M237HB82BNSBK7X21KN62C4N`. Lane `ddm_rc2_t4_hpac_semistatic_mixing_20260909`. Modal wall 590.1 s. Archive sha `c810c2c7f72e57670dc29bde27d584b18aa82feff68b063936a61dca89cf671e`, 181,414 B. Runtime tree `efafd6ab98ef83a4b59f90bb792376f19f9c85d35b6e5ac658e8aa69326cfe63`. `passed: true`, `validation_errors: []`.

Recomputed FROM COMPONENTS (#877 — never the rounded display):

| term | value |
|---|---|
| rate 25·181,414/37,545,489 | 0.12079613612170559 |
| seg 100·0.00010913 | 0.010912999999999999 |
| pose √(10·5.1e-06) | 0.0071414284285428505 |
| **S** | **0.13885056455024844** |

| | ddm_sj1_t4_token_predistortion_pass3_20260906 (prior pointer) | this move | delta |
|---|---|---|---|
| S | 0.13900437796841966 | 0.13885056455024844 | **-0.00015381341817122252** |
| d_seg | 0.00010913 | 0.00010913 | 0.0 |
| d_pose | 5.1e-06 | 5.1e-06 | 0.0 |
| bytes | 181,645 | 181,414 | -231 |

## Projection fidelity

Projected 0.13885056455024844; realized − projected = 0.0. zero-distortion custody replay: projection = exact row to the last digit (d_seg 0.00010913, d_pose 5.1e-6 reproduced)

## The mechanism

rc2 (`.omx/research/ddm_rc2_hpac_semistatic_depth_mixing_prior_and_container_sweep_20260908.md`, commit fb39f2139) re-coded the IHS1 HPAC model rows with a COUNTED 8-byte shared logistic mixer across depths (one weight vector shared by every depth cell, adapted per symbol), on top of rc1's adaptive per-group tree coder. The field-bearing sections (semantic, carrier, token tail) are byte-identical to the move-32 pointer; only the model section and its reader changed, so d_seg and d_pose are the pointer's by construction and were reproduced exactly on T4 (0.00010913 / 5.1e-6). Rate: 181,645 → 181,414 B (−231 B); the 16-byte mixer won raw J by 5 B but lost the archive objective by 1 B. The semi-static previous-value table lost (best counted J 10,853 B, 1,210 B behind rc1) and is closed at formulation scope; container retuning alone recovered 0 B (q11 × lgwin 22/23/24 tie). All 16 designs decoded exactly, twin archives byte-identical, public-entrypoint smoke receipts present, SEAL_VALID under the scg1 contract.

## Sub-0.12 arithmetic RE-DERIVED at this move (law: binding numbers expire at every pointer move)

gap 0.018850564550248444. Exchange 25/37,545,489 = 6.658589531221714e-07 S/B.

- **RATE corner** at held distortion 0.01805442842854285: archive ≤ 153,103.9 B → **-28,310.1 B**.
- **DISTORTION corner** at held bytes 181,414: distortion ≤ -0.00079614 → **inf× reduction**.
- Zero-distortion B_max 180,218.347 B → the archive is **-1,195.653 B over** the threshold at zero distortion.

## Custody

- Harvest: `/Volumes/APDataStore/pact/ddm_rc2_t4_hpac_semistatic_mixing_20260909/MODAL_REMOTE_RESULT.json` (sha `5345c38da8bf2f37b291b4164a21f101e7b14a9fb3f04527f82600faa6e72133`).
- Archive: `/Volumes/VertigoDataTier/pact/ddm_rc2_hpac_semistatic_mixing/candidate_runtime/archive.zip` (sha `c810c2c7f72e57670dc29bde27d584b18aa82feff68b063936a61dca89cf671e`, 181,414 B).
- Seal: `/Volumes/VertigoDataTier/pact/ddm_rc2_hpac_semistatic_mixing/SEAL_ddm_rc2_hpac_semistatic_mixing_contest_cuda.json`.
- Second copy: `/Volumes/APDataStore/pact/ddm_rc2_t4_hpac_semistatic_mixing_20260909/archive.zip` (sha verified: True).
- Second copy: `/Volumes/APDataStore/pact/ddm_rc2_t4_hpac_semistatic_mixing_20260909/SEAL_ddm_rc2_hpac_semistatic_mixing_contest_cuda.json` (sha verified: True).

## What this does NOT claim

No distortion change is claimed or possible: the decoded model rows are bit-identical (restored 17,770 B IHS1 body) and every field-bearing section is unchanged; the T4 row is a custody replay of the same renders. Not claimed: the ≈953 B still between rc2's raw J and the order-1 Miller–Madow bound (a richer shared low-parameter mixer is a LIVE HYPOTHESIS, gated by rc2's own rule: only with a closed-form counted-parameter design predicting ≥ 150 B more). Not claimed: any CPU-axis number (never measured on this lineage; the PR question is operator-gated). Not claimed: composition with the three live render-side candidates (sj1 pass 4, pc2 scales, fe1 FiLM) — each re-bases onto this tree by swapping the model section; their renders do not change.

## Next from here

1. Every live arm re-bases its candidate onto this tree (model section + reader swap; field-bearing sections unchanged): sj1's pass-4 seal, pc2's held scales seal (+ ITEM 1 re-solve), fe1's FiLM candidate, rw1's fold-back base. Sequencing unchanged: sj1 pass 4 first, then pc2, then fe1, rw1 last.
2. rc3 (successor byte-coding arm, gated per rc2): a richer globally shared low-parameter mixer only if closed-form pricing predicts ≥ 150 B of the remaining ≈953 B.
3. Binding arithmetic re-derived at this move (packet): rate corner and distortion corner as written in the pointer line below.

Equations leg (`tac.canonical_equations`): tac.canonical_equations: model_section_adaptive_recode_ceiling_v1 (anchor: shared 8-B logistic mixer converts 231 B of the 1,139 B order-1 gap; semi-static tables lose 1,210 B) + coder_strength_substitutes_for_capacity_v1 (anchor appended by rc2)

Own-vehicle frontier: **S 0.13885056455024844 @ 181,414 B [contest-CUDA T4 n600]**, archive sha `c810c2c7…f671e`.
