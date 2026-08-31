# ddm_wwc1 win-win cone sweep — rate opening reproduced, scorer transfer refused

**Date:** 2026-08-31  
**Status:** `TERMINAL-BYTE-COMPLETE / NO-DUAL-AXIS-SEAL`  
**Measurement axis:** `[macOS-CPU advisory / scorer-free exact B/H/W and real RC64 re-encode]`  
**Score claim:** false; `d_seg`, `d_pose`, and net `Delta S` were not measured here  
**Retained root:** `/Volumes/VertigoDataTier/pact/ddm_wwc1_winwin_cone_sweep/`

## Result

The token-space win-win cone is real on the **rate** axis and is reproducible across multiple real
coders. It is not a demonstrated score route. Fresh source evidence predating this arm already
realized the exact edit class twice: full-union compensation failed the Pose gate, while the
pose-screened subset preserved Pose but made realized Seg worse. This falsifies the charter's
prediction that a DALI-GT benefit label transfers automatically to realized Seg benefit.

The new scorer-free work completed the missing OE1 screen and the required joint re-encode:

- all five OE1 rungs have the same exact final coding-argmax field as DX2, SHA-256
  `db498280c22c3aa1b787310e25435116911933216cae558f309f8b10baf7994e`;
- their benefit coordinates are cell-for-cell identical to FCD1's 5,268 coordinates;
- every OE1 rung saves real archive bytes, repeats byte-identically, and receiver-decodes to the
  retained benefit field exactly;
- the deduplicated FCD1 + JF2 + OE1 union contains 8,768 coordinates and produces a real 174,609 B
  JF2 archive, SHA-256
  `06cae23e83deb04a1139fbe665ff7b7a8650716e9360f73f2e0ccc05e01abe1f`;
- that archive is 4,183 B below the 178,792 B JF2 base (`Delta S_rate =
  -0.002785288000910043`), repeats byte-identically, and production-decodes to the exact retained
  union field;
- it is nevertheless **27 B larger than the already-retained JF2-only benefit archive** at
  174,582 B. The 467 FCD1/OE1-only coordinates therefore cost +27 B on the JF2 receiving body
  (`+0.4625267665952891` real bits per added edit), rather than adding their banked byte credit.

The effective frontier did not move. No scorer, Modal job, seal, READY row, or pointer mutation was
created.

## Control reproduction — first gate

The full n600 DALI-GT control ran before OE1. Two independent implementations agreed exactly.

| control population | B | H | W | denominator | disposition |
|---|---:|---:|---:|---:|---|
| raw DX2 token versus coding-argmax disagreement pool | 5,268 | 221,862 | 541 | 227,671 disagreements | reproduced |
| banked FCD1 union after selecting only B | **5,268** | **0** | **0** | 5,268 selected edits | reproduced |

The control used the retained DALI GT payload at
`/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy`, 117,964,928 B,
SHA-256 `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248`.
The earlier raw-pool-only receipt is retained under `control/`; it is superseded, not deleted. The
charter-complete two-population receipt is `control_v2/CONTROL_RESULT.json`.

## Banked union freshness and the scorer boundary

The charter's statement that FCD1 had never been realized was stale by execution time. Reading the
later source memos changed the plan from dispatching a scorer this arm did not own to completing the
remaining exact byte work and reporting the already-measured semantic boundary.

| realization | exact retained finding | disposition |
|---|---|---|
| FCD2 full 5,268-edit union with fresh in-compile compensation | `[macOS-CPU frozen-scorer advisory, DALI-lineage GT]` uncompensated `d_pose=0.0016055422933954212`; compensated `d_pose=0.00027348054805362656`; same-instrument base `0.0000063656845167356244`; 42.96168736207959x base | `INSTANCE-REFUSED-POSE-GATE`; full n600 Seg scorer not fired, so exact-union realized `d_seg` and net `Delta S` remain not measured |
| FCD3 pose-screened 4,194-edit subset | `[macOS-CPU frozen-scorer advisory]` 177,252 B, -2,940 B; `d_pose=5.8495951113985735e-6`; `d_seg` changed `0.0003474002587608993 -> 0.0003874630492646247`; net `Delta S=+0.0019433243907622244` | `INSTANCE-REFUSED-SEG-BAND` |

The FCD3 result is the falsifier named by the charter: token labels that move toward DALI GT did
not predict realized Seg benefit; their realized Seg effect was adverse in sign. FCD2 and FCD3 are
advisory-axis measurements and cannot be subtracted from the live contest-CUDA pointer.

## Per-family cone sweep

`Raw B/H/W` describes each field x model disagreement screen. `Selected cone` describes the actual
field passed to the real coder.

| family / row | raw B/H/W | selected cone B/H/W | real archive transition | real bits/edit | exact disposition |
|---|---:|---:|---:|---:|---|
| FCD1 / DX2 | 5,268 / 221,862 / 541 | 5,268 / 0 / 0 | 180,192 -> 176,436 B, **-3,756 B** | -5.703872437 | byte-admitted; semantic class refused by FCD2/FCD3 |
| JF2 k060 | 8,301 / 207,809 / 920 | 8,301 / 0 / 0 | 178,792 -> 174,582 B, **-4,210 B** | -4.057342489 | byte-admitted; MAIN scorer fire previously refused |
| LD1-induced support | 14 / 0 / 0 | 14 / 0 / 0 | 180,389 -> 180,390 B, **+1 B** | +0.571428571 | `WIN-WIN-VERIFIED-CLOSED` at family-induced scope |
| OE1 control_w0 | 5,268 / 221,862 / 541 | 5,268 / 0 / 0 | 180,368 -> 176,612 B, **-3,756 B** | -5.703872437 | byte-admitted; exact FCD1 cone, not independent semantic support |
| OE1 escape_w1 | 5,268 / 221,862 / 541 | 5,268 / 0 / 0 | 192,673 -> 188,600 B, **-4,073 B** | -6.185269552 | byte-admitted; exact FCD1 cone |
| OE1 escape_w4 | 5,268 / 221,862 / 541 | 5,268 / 0 / 0 | 192,100 -> 188,120 B, **-3,980 B** | -6.044039484 | byte-admitted; exact FCD1 cone |
| OE1 escape_w16 | 5,268 / 221,862 / 541 | 5,268 / 0 / 0 | 191,489 -> 187,616 B, **-3,873 B** | -5.881548975 | byte-admitted; exact FCD1 cone |
| OE1 escape_w64 | 5,268 / 221,862 / 541 | 5,268 / 0 / 0 | 191,186 -> 187,375 B, **-3,811 B** | -5.787395596 | byte-admitted; exact FCD1 cone |
| DG2 k040/k060 | closed by recall | closed by recall | contained in JF2 | closed by recall | duplicate physical replay not run |
| AE1 | absent | absent | absent | absent | no physical RC64 candidate or final coding-argmax object |

Every new OE1 archive has a byte-identical deterministic repeat. All five streams recovered the
same 117,964,800-byte retained benefit field, SHA-256
`7988b14811e532e751e1986a85d27aa32410e4d41b07e73ff126ed51a08d2bde`, through their own decoder
trajectory. The OE1 screen therefore closes the missing byte question without pretending it is a
new scorer-safe cone.

**Denominator:** 5 trade-space families enumerated; 3 physically screened at their available
registered scope (JF2 1/7, LD1 6/6, OE1 5/5); 1 closed by recall (DG2, contained in JF2); 1 `ABSENT`
(AE1, no physical RC64/final coding-argmax object).

## Joint re-encode

The coordinate union was applied once to JF2 k060 and then encoded once, with an independent full
repeat.

| source membership | unique coordinates |
|---|---:|
| JF2 only | 3,500 |
| FCD1 + OE1 only | 467 |
| FCD1 + JF2 + OE1 | 4,801 |
| total | **8,768** |

FCD1 and OE1 contribute the same 5,268-coordinate set. JF2 overlaps that set in 4,801 positions and
adds 3,500. All 8,768 unique coordinates change the JF2 receiving field. The retained field and
the production-decoded payload are byte-identical at SHA-256
`a43f869c10e5ff06328224afd0f18617e7e5350326835c8c863aeabff1558564`.

The joint archive proves non-additivity in the exact direction warned by the charter: the apparent
FCD1/OE1 byte credit does not stack on JF2. Relative to the JF2-only 174,582 B benefit archive, the
joint result is +27 B. It is therefore not the rank-1 byte candidate from this family set.

## Dual-axis disposition

`NO-SEAL / NO-FIRE-CURRENT-ARCHIVE`.

The joint archive is byte-real and receiver-closed, but it has no same-axis Seg/Pose row. Its
467-edit add-on is rate-dominated by JF2-only, while the same token-GT edit class has already failed
the realized semantic gate on both Pose and Seg. The evidence does not clear the admit bar for a
new two-run base/candidate scorer dispatch. MAIN's prior JF2 scorer refusal therefore remains the
consumer decision; this arm did not reopen or duplicate it.

## Custody and reproducibility

- Runner: `experiments/ddm_wwc1_winwin_cone_sweep.py`, 51,780 B, SHA-256
  `b8535f68f74288a05e430665cf032bea8e35549a89d7a4c135c2781e914919b1`.
- Manifest: `/Volumes/VertigoDataTier/pact/ddm_wwc1_winwin_cone_sweep/MANIFEST.json`, SHA-256
  `e33c9ce77dab596dc9a95afdbaddb45eafbd06dd7d2da15d62279c7e4042a378`.
- Manifest denominator: 1,662 retained artifacts, 2,094,159,396 total artifact bytes.
- Summary result: `RESULT.json`, SHA-256
  `3e40d5fcb7bdb4e490cad805f7be482923fc75e48bf7ee3bb24563e27e571167`.
- Every long replay used 20-frame receiver/encoder checkpoints. No payload was discarded or routed
  to local disk. The final Vertigo free-space observation was about 2.5 GiB, above the reserved
  1 GiB floor; AP remained an observed fallback and was not written by this arm.
- `upstream/` was not modified. No scorer or remote service was invoked.

## RECALL EVIDENCE

I searched the full `.omx/research/` corpus and retained arm receipts by content using the query
families `win-win|B/H/W|coding argmax|field-for-coder|FCD1|FCD2|FCD3|JF2|OE1|LD1|AE1|real
re-encode|compensation|pose-screened`; searched `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED
blocks, `main_hot_state.md`, and task-ledger surfaces for the same family/task identifiers; and ran
`tools/list_canonical_equations.py --json`.

Relevant registry laws were `score_marginal_lagrange_multipliers_v1`,
`pairset_component_marginal_score_decomposition_v1`, `compensated_semantic_edit_exchange_v1`,
`token_rate_model_direction_dependence_v1`, and `greedy_set_average_vs_marginal_price_v1`.

Beyond the charter's seeds, recall found:

- FCD2 and FCD3 had already realized the exact edit class and refused it on Pose and Seg,
  respectively. This removed any authority to describe the union as never scored or to transfer a
  token-label prediction into a scorer claim.
- BHW2's later MAIN adjudication had already refused the JF2 scorer fire using those realizations.
  This changed the consumer action from another dispatch request to a scorer-free completion and
  typed no-fire disposition.
- BHW1 supplied the full five-family denominator: DG2 is contained in JF2, LD1's family-induced
  support is 14 cells at +1 B, and AE1 lacks the physical object required for an honest screen.
- Live storage inspection found both SSDs mounted but nearly full. The implementation therefore
  content-addressed the five byte-identical OE1 screens and enforced phase-specific storage
  preflights with a 1 GiB post-run reserve rather than relying on the charter's older storage note.

## NEXT_IF_RESUMED

- `QUEUED-CONDITIONAL-NEW-FORMULATION` — owner: MAIN; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_wwc1_winwin_cone_sweep/followons/scorer_native/`; fire trigger:
  MAIN provides or charters a same-object scorer-native edit selector using retained realized
  Seg/Pose sensitivities plus fresh in-compile compensation, rather than DALI token-agreement
  labels. Do not fire merely because the current archive saves bytes.

## LIVE-HYPOTHESES

- A scorer-native selector could preserve the measured favorable coder direction while avoiding
  the token-label transfer failure. This is plausible because real RC64 savings reproduced across
  DX2, JF2, and five OE1 trajectories, while every observed failure occurred at the mismatch between
  token-GT agreement and realized Seg/Pose response. It is untested and requires a new formulation,
  not another sweep of the retained token-GT cone.

## DEAD-ENDS

- Re-scoring the current 5,268-cell token-GT cone as though it were untouched is closed: FCD2
  refused fresh full-union compensation at 42.961687x base Pose, and FCD3's pose-safe subset made
  realized Seg worse and net score higher.
- Treating OE1 as independent semantic confirmation is closed: all five coding-argmax fields and
  all 5,268 selected coordinates are exactly the DX2/FCD1 objects. OE1 confirms coder portability,
  not scorer safety.
- Adding the 467 FCD1/OE1-only cells to JF2 for rate is closed at this instance: the real joint
  archive is 27 B larger than JF2-only.
- Re-running DG2 separately is closed: its two rows are contained in JF2.
- Reopening LD1's family-specific cone is closed: 14 new benefit cells cost +1 B.
- Substituting AE1's pre-corrector predictor or a synthetic coder object is closed: no physical
  final RC64 candidate/coding-argmax object exists.
- Per-pair/per-frame/per-block/spatial routing, alternate lossy bodies, stale transferred Schur
  compensation, uncompensated semantic edits, RC64 reordering, and another coder-only search remain
  closed by the prior negative evidence consumed by the charter.

[contest-CUDA T4 n600] own-vehicle frontier: AFR1 — S=0.14797617125559104, archive=180,002 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25.
