# ET4 twelfth-move adjudication — solve-within-CVP tail through byte-close + n600 evaluate

**Date:** 2026-08-06 · **Adjudicator:** MAIN · **Axis:** [macOS-CPU advisory] · score_claim=false
**Receipt:** /Volumes/VertigoDataTier/pact/ddm_et4_20260806/byteclose_archive_receipt.json
(schema ddm_et4_byteclose_archive_receipt.v1) · evaluate report: submission/et4_evaluate_report.txt
**Custody:** archive 11,645,079 B sha256 34f0b769b67cb56c6b1a2f61519eb9303f4a7785d4edd5ccc63b986dc10d314b
(parent tq1c payload 357,729 B sha 1bd7b718… + patch stream 11,286,732 B sha 63417d6d… + metadata 490 B).
Rows-ledger dedupe custody: 751→600 keep-last, backup et4_solve_within_cvp_rows.jsonl.pre_dedup_20260806T222740Z.bak,
sha_before c0e5715c3050…, sha_after 1836954820…. Repair chain rc=0, elapsed 749 s (receipt et4_repair_final3).

## Verdict: NOT A POINTER MOVE — rate-dominated; the SOLVE leg is real, the CARRIAGE is the entire loss

| axis | baseline (tq1c, 11th move) | et4 row | Δ (S units) |
|---|---|---|---|
| d_seg | 0.004305420 | **0.00364163** | **−0.066379** (−15.4% of live d_seg) |
| d_pose | 0.000716509 | 0.00072086 | +0.000257 |
| bytes | 357,837 | 11,645,079 | +7.515711 (+11,287,242 B) |
| **S** | **0.7534578126** | **8.2030466** | **+7.449589** |

recomputed_score 8.203046586569972 (evaluate_final_score 8.2 = rounded display, per doctrine);
n600, inflate rc=0 in 255.3 s, 3,662,409,600 raw bytes out (full-population complete, dk1 blocks 234,673/234,673).

## The decomposed finding (m82: decompose every headline)

1. **The exact within-CVP solve DELIVERS on seg through the real archive path**: d_seg 0.0043054 →
   0.0036416 realized through byte-close → inflate → n600 evaluate. First measured proof at this base
   that the solver axis is not the blocker — ≈78,304 net flips fixed (6.638e-4 × 196,608 × 600).
2. **W break-even independently reproduced**: band-priced cost of those same flips = 78,304 ×
   1.27310821533 = 99,689 B → rate +0.066379 S = exactly the seg gain. (W is the break-even constant
   by construction; the receipt closes the loop to 6 decimals.)
3. **The naive carriage is 113× over break-even**: patch codec sparse_frame1_i16_delta_brotli@Q11
   spends 144.1 B/flip (11,286,732 B / 78,304 flips). Mechanism: **122.8 nonzero i16 pixel deltas per
   net flip fixed** (total_nnz 9,613,398) — the solve writes broad sparse frame corrections; almost all
   carried mass is collateral/precision, not flip-band signal. Raw 57.68 MB → brotli 11.29 MB (5.11×)
   cannot rescue a description that is wrong-domain.
4. **Doctrine confirmed** (seg ~100% solvable → gap = CARRIAGE): this is the cleanest two-sided
   measurement yet. Route: the SAME corrections re-described through band-limited / menu / #869
   adaptive-quant / per-edge (m91) carriage inside campaign #984 composition — the target is <1.27 B/flip
   on the subset that pays, not cheaper brotli on all 9.6M deltas.

## Consumers
- Campaign #984 composition (CVP exact-solve tail = a named our-lever; this row prices its ceiling
  −0.0664 S seg and its carriage bar ≤99.7 KB at break-even).
- #939 realization-half pricing (description ≠ realized correction — here realized, description unpaid).
- Costate SENSE: solver-reach law (et4: −15.4% live d_seg via within-CVP exact solve, INSTANCE scope:
  tq1c base, CVP block structure dk1 234,673 blocks).

Banked NEGATIVE with full custody; archive bytes retained on SSD (certify-or-block satisfied).
