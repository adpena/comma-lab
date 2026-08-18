# keep01 ADMITTED — the ninth pointer move, and the mass axis is now a measured curve

`verdict_scope`: **INSTANCE** for the admitted row (archive
`316d17f84817d2b10f084a71b6ca706c69411d738cdb7d5921aec9afb8c37f00` @ 177,576 B, contest-CUDA
T4, n=600). The retention-vs-mass and CPU→T4 transfer readings are the family's second and
third MEASURED points; the shapes (concavity, per-object compensation) are DERIVED-EXACT.

Receipts: `/Volumes/APDataStore/pact/ddm_sa3/keep01_compile/t4_row_r2/` —
`MODAL_REMOTE_RESULT.json` + eight embedded artifacts persisted separately per P0. Call
`fc-01M0AWSK9T0KPXK68ZG2DFPVSY`, wall 1,235 s, ~$0.16. Authority solve custody:
`/Volumes/APDataStore/pact/ddm_sa3/keep01_authority/sm3r_keep01/AUTHORITY.json` (n=600).

## 1. The row

| | value |
|---|---|
| S | **0.1571619225142182** |
| archive | 177,576 B, sha `316d17f8…` |
| d_seg | 0.00030135 |
| d_pose | 7.72e-06 |
| axis | `contest-CUDA`, Tesla T4, `gpu_t4_match=True`, n=600 |
| harness | `passed=True`, `rc=0`, `validation_errors=[]` |

Net vs the sz1 compile base (0.15771357797660338 @ 179,930 B): **−5.516555e-04, ADMITTED.**
Net vs the prior sa3 pointer (0.15765851477950737): **−4.965923e-04.** Ninth micro-campaign
pointer move. Gap to 0.15: 0.00766 → **0.00716**. Across sa3 + keep01 in one day: 7.1% of the
gap closed.

## 2. Legs, falsifiers, custody

| leg | Δ vs sz1 | share of rate credit |
|---|---:|---:|
| rate | **−1.567432e-03** (−2,354 B) | 1.000 |
| pose | +4.917765e-04 (d_pose 6.880e-06 → 7.72e-06) | 0.314 |
| seg | +5.240000e-04 (+5.24e-06 d_seg) | 0.334 |
| **net** | **−5.516555e-04** | **0.3519 retained** |

All four sealed falsifiers PASS: F1 net (158× the −3.5e-06 bar) · F2 d_pose 7.72e-06 ≤
8.773e-06 · F3 seg +5.24e-06 = 0.55× the pre-registered +9.54e-06 cap · F4 clean.
Quantization custody: `report_8dp_score_worst_case_abs_error_bound = 3.345782e-06`; the net is
**164.9× the bound**. Sign determinate by a stated margin (the #1032 cure, applied).

## 3. The two model updates this row buys

**Retention RISES with mass, harder than concavity alone predicts.** sa3 (1× mass) retained
10.47%; keep01 (3.0× mass) retained **35.19%**. The concavity table predicted ~11.2% at this
mass HOLDING compensation quality fixed — keep01's own per-object Schur solve
(residual_fraction 0.376 vs sa3's 1.0) delivered 3.36× that. The precondition law from the
morning's falsification is now measured twice: **cancellation quality is a property of the
edited OBJECT, re-solved per object, and it has so far IMPROVED with mass.**

**The CPU→T4 pose transfer is measured: 1.134× ABSOLUTE.** T4 residual d_pose 8.40e-07 vs the
CPU-solved 7.408e-07. The ABSOLUTE model under-predicted by 13.4%; the RELATIVE model (21.4×)
over-predicted by 18.9×. Successor rungs project with ABSOLUTE × ~1.13 and show both models
per the standing both-models rule.

## 4. Routing — the V-series mass ladder is the successor, and its gate is live

The keep-percent ladder is exhausted at keep01 (keep_percent=1). The remaining mass axis is
the **V-series SD1M mixed-precision ladder** (`ddm_sa1/retained`, waterfall builder,
`sd1.pack_semantic_state` bodies inside the standard rx1 container — the shipped runtime
decodes SD1M natively, `runtime/residual_archive.py:190`, compose.py parse-back PASS).
V7_V6_livepw_q2: 166,718 B = −13,212 B vs sz1 = **16.7× keep01's rate credit; the rate leg
alone (−8.80e-03 S) exceeds the entire remaining gap.** At the family's measured retention
that projects roughly −2e-03 to −3e-03 S — IF the compensation solve holds at that mass and
seg damage stays in family. The seg prior cited here: sa1 measured its quantization edits
seg-INERT (verdict_scope: formulation — semantic-tensor quantization edits on the rr4/sz1
lineage, 3 measured members; NOT a family law, and the V-series' deeper 2-3 bit depths may
leave it), and keep01's row-prune seg landed at 0.55× its pre-registered cap.

Fired: advisory n600 CPU row on V7 (pid 49826, done-receipt `v7_advisory_n600_20260818`,
archive sha `0ecf5d9c4a9b8082…` pinned, staged generation dir with the SD1M-capable runtime).
The advisory supplies (a) decode proof, (b) d_seg delta, (c) uncompensated d_pose — the three
inputs the authority solve (free Metal slot) and the rebase compile need. Chain thereafter is
the proven keep01 chain unchanged: 3h authority solve → rebase compile → seal → T4.

## 5. What is NOT settled

- V7's uncompensated pose damage is UNMEASURED; if it is large enough that required
  cancellation exceeds ~99.98% (the best achieved), the rung refuses honestly at the arithmetic
  stage before any T4 spend.
- Whether sa3's S2-quantization edit and keep01's row-prune COMPOSE is untested — both edit the
  same semantic tensors, so composition requires a joint re-solve, not a byte merge. Tracked as
  a later rung; the V-series dominates it on projected mass.
