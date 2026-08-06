# DDM VO1 - Verdict-Instrument Optimality Audit

**Axis:** `[scorer-free receipt/source audit]`  
**Date:** 2026-08-06  
**Authority:** routing only, `score_claim=false`, no archive, no scorer slot, no `upstream/evaluate.py`  
**Scorer slot:** not used; `ddm_et2` remains the active scorer owner under the common contract.  
**Review tags for landing:** `[no-triality] [p0-ledger-ok]`

## Answer First

VO1 found that several standing negative or midpoint verdict chains are not
optimal-form verdicts. The highest-yield defect is not a scorer result; it is
the instrument class that generated the verdict.

Top producer-fanout rows by `fanout_count * stakes_weight`:

| Rank | Producer form | Grade | Scoped fanout basis | Score |
|---:|---|---|---:|---:|
| 1 | iteration-cap/default stop gates | `NAIVE-SUSPECT` | CA1: 89 cap-default sites, 83 silent, 6 live/reopen-risk | 267 |
| 2 | project-after pose-null/projector harnesses | `NAIVE-SUSPECT` | SW1 7 seam rows plus DK1 5 lattice-native-solvable sites | 36 |
| 3 | float-first/post-hoc uint8 realizers | `NAIVE-SUSPECT` | DK1 5 source sites plus FD/V19C/PF3B dependent verdict chains | 16 |

Reopen ledger count by stakes tier:

| Stakes tier | Rows |
|---|---:|
| P0 | 4 |
| HIGH | 5 |
| MEDIUM | 3 |

Single highest-stakes reopen: `et1_phase_field_q3_projection_and_pose_finish`.
Fire order: hold scorer until `ddm_et2` releases it; wire the ET1 phase-field
target through SW1 solve-within and DK1 lattice realization scorer-free; then
run a declared non-prefix n32 Seg/Pose check; byte-close only if eta clears the
ET1 break-even after terminal pose finish.

## Scope And Method

This arm did not run scorer, train, launch, inflate, or edit upstream. It read
the charter, common contract, `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`,
`docs/operating_manual_craft_handoff.md`, the current hot state, and the
receipts/ledgers cited below. All claims here are from existing source or
receipt evidence.

Fanout counts are bounded and source-scoped. They are not a global proof that no
other verdicts exist. AU1 and probe-outcome ledgers were used as recall surfaces
only; OA1 already warns that detector counts are triage until adjudicated.

The stakes weight used for ranking is `P0=3`, `HIGH=2`, `MEDIUM=1`. Where the
count mixes source sites and verdict chains, the row says so explicitly in
`INSTRUMENT_FANOUT.jsonl`.

## Instrument Findings

### 1. Cap/Stop Gates

CA1 found 89 cap-default sites and 83 silent cap defaults. Six were live or
reopen-risk: `ddm_q3x_q3_convergence_measurement.py`,
`ddm_et1_block16_realization.py`, `ddm_sq1_pose_null_constrained_paint.py`,
`pose_frame0_inverse_solve_probe.py`, `probe_onpolicy_scorer_surrogate.py`, and
`probe_yopo_first_layer_costate.py`.

Form grade: `NAIVE-SUSPECT`. A cap-stopped row can be a floor, progress trace,
or blocker, but it is not a convergence negative unless the stop reason says
semantic convergence rather than iteration budget.

### 2. Project-After And Metric Seams

SW1 measured the same n4 slice against eta bar `0.1710048742006269`:

| Arm | eta |
|---|---:|
| project-after Euclidean | 0.0940882598 |
| project-after diagonal metric | 0.0982514571 |
| solve-within null basis | 0.3114071607 |

The null basis certificate was numerically tight (`max_abs_A_times_N`
`1.1796e-16`). The measured form defect is the seam: solving the wrong space
and projecting after the fact can invert a verdict.

Form grade: `NAIVE-SUSPECT`. Better instrument exists now: solve within the
constraint/native subspace first, then realize through the receiver.

### 3. Float-First Uint8 Realization

DK1 measured the realizer ladder on real parent frames and block16 phase
offsets:

| Realizer | mean A(Dx) leakage | mean real PoseNet dpose |
|---|---:|---:|
| naive uniform scorer-round | 0.07217925 | 4.913175500720e-09 |
| Dykstra round/project k=8 | 0.039971942048 | 2.123022215385e-10 |
| CVP/Babai kept-set enum | 0.007107145151 | 6.435523410295e-13 |

DK1 does not claim a global MIQP optimum, but it does provide a concrete
lattice-native replacement for five audited float-first sites. That is enough
to reopen rows whose verdict chain depends on naive rounding.

Form grade: `NAIVE-SUSPECT`.

### 4. Prefix, Subset, And Aggregate Gates

NA6/NA2 corrected earlier pose-family wording: the post-hoc/stored pose verdict
still stands as weak evidence, but it is not a measured family death. The old
support was contiguous prefix n8/n24 against low effective serial N, and prefix
bias was pose-axis anti-conservative. LC1 killed aggregate static PE3 target
labels at n32, not per-record positive-net filtering or conditioning-only PE3.
M5R showed top24 subset ranking could fail full n600 admission.

Form grade: `NAIVE-SUSPECT` when the subset is used as population authority;
otherwise `UNKNOWN` until the row declares its denominator.

### 5. Greedy, Fixed-Order, And Finite-Menu Compilers

V19C saturated a finite represented-coordinate inventory; PF3B found a strict
joint-distortion-improving physical edge that was rate-dominated at +860 E4
bytes; M5R/V12 were scoped to restricted/inventory-capped formulations. These
are not fake measurements. They are also not global family deaths unless a
stronger compiler, packing objective, or receiver-effective DOF accounting has
been exhausted.

Form grade: `UNKNOWN` unless a row proves the stronger compiler was tried.

## Reopens

Machine-readable rows are in `REOPEN_LEDGER.jsonl`. The P0 rows are:

| Row | Reason |
|---|---|
| `et1_phase_field_q3_projection_and_pose_finish` | priced band had live eta, but Q3 projection and terminal pose finish were unmeasured under the better instrument |
| `q3x_q3_convergence_realizer` | cap-default plus float-first source site; DK1 and CA1 provide replacements |
| `fd1_fd2_zero_accept_integer_near_margin` | FD2 found uint8 seg-realization gap, not a proven pose veto or locality death |
| `ca1_class_b_cap_stopped_rows` | six live/reopen-risk cap sites must be relabeled or rerun before reuse |

HIGH rows: `sq1_pose_null_constrained_paint`,
`q31_q3_constrained_solve`, `lc1_pe3_label_filter_regrade`,
`v19c_terminal_band_saturation`, and `rl1_road_lane_realization_half`.

MEDIUM rows: `pf3b_rg3_edge_amortized_packaging`,
`m5r_v12_g1_greedy_mdl_regrade`, and
`rv1_x6_posthoc_pose_prefix_retest`.

## Honest Non-Reopens

These rows do not get reopened on instrument form alone:

| Row/surface | Decision |
|---|---|
| RL1 byte pricing | Real serialized coder output is clean for byte-only scope; only the missing receiver half is open. |
| PF3B current edge as-is | Physical edge improves distortion, but current +860 B E4 representation is not a score mover. Reopen only packaging/composition. |
| DK1 bounded kept-set enumeration | Clean bounded local primitive; no global MIQP claim was made. |
| SW1 solve-within certificate | Clean n4 advisory instrument proof; no n600 or archive score claimed. |
| FD2 positive-control acceptor | The acceptor itself is not treated as broken; the stale part is using smooth/float proposals as realized argmax proposals. |
| LC1 aggregate PE3 target-label result | The aggregate static-label formulation stands negative for its n32 scope; only filtered/conditioning PE3 remains open. |

## Evidence Consulted

- `.omx/research/ddm_sw1_20260806/RECEIPT.md`
- `.omx/research/ddm_sw1_20260806/SEAM_METRIC_LEDGER.jsonl`
- `.omx/research/ddm_dk1_20260806/RECEIPT.md`
- `.omx/research/ddm_ca1_20260805/CA1_RECEIPT.md`
- `.omx/research/ddm_na6_20260806/RECEIPT.md`
- `.omx/research/ddm_oa1_20260805/OA1_RECEIPT.md`
- `.omx/research/ddm_lc1_20260805/LC1_RECEIPT.md`
- `.omx/research/ddm_et1_eta_on_the_priced_band_20260803.md`
- `.omx/research/ddm_fd1_family_d_gn_DAG_FEED_20260728.md`
- `.omx/research/ddm_fd2_posenull_gn_disambiguation_20260728.md`
- `.omx/research/ddm_fd2_posenull_gn_disambiguation_DAG_FEED_20260728.md`
- `.omx/research/codex_findings_ddm_v19c_correction_saturation_20260723_codex.md`
- `.omx/research/ddm_rl1_roadlane_interface_price_20260803.md`
- `.omx/research/ddm_m5r_road_frame1_fresh_solve_20260723T115443Z/receipt.json`
- `.omx/research/ddm_v12_obligation_n600_20260722T161517Z/ddm_v12_obligation_search_n600_receipt.json`
- `.omx/research/ddm_mr2_pricing_wave_merge_DAG_FEED_20260726.md`
- `.omx/research/ddm_cs1_consolidation_harvest_20260728.md`
- `.omx/research/ddm_ng1_20260805/negative_verdict_ledger.jsonl`
- `.omx/state/probe_outcomes.jsonl`
- `.omx/research/ddm_au1_20260805/au1_corrections_index.jsonl`

## Boundaries

Did measure in this arm: zero new scorer rows, zero archives, zero training
runs. Did not find in the cited receipts a fresh rerun of M5R/V12/G1 under a
stronger non-greedy compiler; that is a bounded absence over the files listed
above, not a global nonexistence claim.

No upstream files were edited. No protected files were edited. No `/tmp` path is
used as evidence.

## Frontier Status

Own-vehicle frontier remains unchanged: `S = 0.7534578126155775 @ 357,837 B`
on `[macOS-CPU advisory]` per current hot state. Contest/effective pointer is
unchanged. Goal progress was not achieved in this analysis-only VO1 arm.
