# DDM WH1 — wrong-half decomposition on the GB1 body

## Outcome

**MEASURED, full n600, `[macOS-CPU advisory / scorer-free shipped-GB1-coder instrumentation]`.**
The 113,624-byte GB1 token stream pays 111,123.343 ideal bytes (97.7996% of selected-symbol
cost) for the binary question “is the coder argmax wrong?” and 2,500.167 ideal bytes
(2.2004%) for “which class, conditional on wrong.” The WRONG branch alone is **76,470.790 B**:
227,609 wrong tokens at 2.687795 indicator bits/wrong token, plus 0.087876 which-class
bits/wrong token. The ideal selected-symbol cost is 113,623.511 B; the retained physical
payload is 113,624 B, leaving the expected 3.914-bit RC64 termination residual.

This arm did not create a model, candidate, score row, or counted byte. It instrumented a copy of
the exact GB1 runtime, decoded all 117,964,800 positions exactly against retained token custody,
and kept every 20-pair ledger/checkpoint payload. The effective frontier is unchanged.

Primary receipt: `/Volumes/APDataStore/pact/ddm_wh1_wrong_half_decomposition/measurement_v1/RESULT.json`
(170,611 B, SHA-256 `3d8adf377729d3b8e1aba49f09d39bb615015f7a6dda310be14e9e003a7c9f4d`).
Cleanup/custody manifest: `/Volumes/APDataStore/pact/ddm_wh1_wrong_half_decomposition/measurement_v1/CLEANUP_MANIFEST.json`
(17,218 B, SHA-256 `ce498aff036826867eddcd1e096df2e0b2debd1d72637a395cbf90c9a03ac6df`).
The retained store is 2.9 GiB and has 30/30 distinct stage receipts.

## Instrument and proof surface

`tools/token_wrong_half_ledger.py` reads the shipped coder’s own float32 probability rows, applies
the exact RC64 `2**31` integer-frequency quantization and winner rebalance, and retains per position:

- WRONG-indicator bits and WHICH-class-given-wrong bits;
- coder argmax, decoded token class, top1-runnerup integer-frequency margin and fixed margin bucket;
- pair/raster position and overlapping G4 geometric masks;
- complete adaptive-corrector and RC64 decoder state every 20 pairs.

Analysis retains the full G4 transition counts, packed xi-proxy membership, track/event payload,
per-stage stationarity categories, and the complete conditional-cell arrays. The copied runtime
contains the original 180,215-byte GB1 archive, SHA-256
`ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`; no sealed GB1/DX2
file or live WD3 trainer file was edited.

Implementation landings:

- `f5b4b007be` — initial retained replay and analysis instrument;
- `45cba91422` — analysis-only correction separating decoded token target from independent DALI GT;
- `5d86a411ba` — explicit moved-object input custody for the Stage-B consumer.

Each Python revision received two genuine `review_tracker.py mark-file` passes after its final edit.
Final focused verification: `py_compile` PASS, Ruff PASS, 4/4 pytest PASS, result verifier PASS.

## RECALL EVIDENCE

The recall search was broader than the charter seeds. Searches covered `.omx/research/`, canonical
research indexes, the sub-0.15 DAG/FEED surfaces, task/state ledgers, and source using the content
terms `token`, `wrong`, `indicator`, `which`, `margin`, `groupbin8`, `HPAC`, `F26`, `stationarity`,
`context ceiling`, `Lane`, and `address tax`. The canonical-equations registry was also queried in
JSON mode for token/entropy/context laws.

Beyond the charter seeds, the search found:

- `ddm_tb2_token_bit_attribution_20260823.md` and `ddm_bl1_per_position_bit_allocation_20260822.md`
  already retain the shipped replay/checkpoint pattern and exact per-symbol RC64 cost field;
- `ddm_tba1_token_bit_attribution_20260823.md` already measures the aggregate DALI-GT class join
  on DX2, including Lane’s 0.59%-area concentration, but does not split indicator from which-class;
- `ddm_hc1_hpac_calibration_reliability_20260824.md` already measures the DX2 aggregate split, so
  rebuilding calibration or reporting that aggregate as the result would duplicate settled work;
- `ddm_mi1_indicator_model_axis_20260824.md` measures the 2,162.13 B gross model-axis excess and
  the 47.4x paid-model break-even miss; GB1’s 153 B collection leaves about 2,009 B;
- `src/tac/optimization/ddm_g4_spatial_stationarity.py` supplies the exact disjoint G4 definitions
  and the explicitly non-physical metric-Pose6 G1 xi proxy;
- `ddm_ma2_merged_alphabet_lane_fold_20260824.md` and `ddm_cx3_context_axis_ceiling_20260822.md`
  keep class-merging and named replacement contexts closed at their measured scopes.

This changed the plan: WH1 reused the settled replay/checkpoint architecture, performed a fresh
20-family GB1 probability readout, treated DALI GT as an independent class tag rather than assuming
it equals the decoded token target, and reserved conditional pricing for decoder-known axes. It did
not remeasure DX2 aggregates or design a position sidecar.

## N600 decomposition

All ratios below name their numerator and denominator in `RESULT.json`.

### DALI GT class marginal

| DALI GT class | Positions / 117,964,800 | Area | WRONG-indicator bits / 611,766.317 | WRONG share | Enrichment over area |
|---|---:|---:|---:|---:|---:|
| Road | 27,407,372 | 23.2335% | 211,477.696 | 34.5684% | 1.488x |
| Lane | 690,754 | 0.5856% | 236,423.254 | **38.6460%** | **65.998x** |
| Undrivable | 58,413,067 | 49.5174% | 62,006.390 | 10.1356% | 0.205x |
| Movable | 1,460,386 | 1.2380% | 64,720.161 | 10.5792% | 8.546x |
| MyCar | 29,993,221 | 25.4256% | 37,138.816 | 6.0708% | 0.239x |

The decoded token field and DALI GT differ at 9,182 / 117,964,800 positions (0.007784%). The
primary class table is deliberately the independent DALI GT join used by the prior Lane law;
`RESULT.json` also retains the decoded-token-class table. “Wrong” itself is always defined against
the actual decoded token, never against the independent class tag.

### Integer-frequency margin marginal

| Margin bucket, bits | Positions / 117,964,800 | Area | WRONG-indicator bits | WRONG share | Enrichment |
|---|---:|---:|---:|---:|---:|
| [0, 0.25) | 52,792 | 0.0448% | 26,928.281 | 4.4017% | 98.357x |
| [0.25, 0.5) | 49,623 | 0.0421% | 26,243.713 | 4.2898% | 101.979x |
| [0.5, 1) | 97,068 | 0.0823% | 51,222.679 | 8.3729% | 101.754x |
| [1, 2) | 198,534 | 0.1683% | 99,190.213 | 16.2137% | 96.339x |
| [2, 4) | 460,531 | 0.3904% | 162,662.302 | 26.5890% | 68.107x |
| [4, 8) | 1,709,341 | 1.4490% | 159,209.055 | 26.0245% | 17.960x |
| [8, 16) | 20,643,698 | 17.4999% | 83,473.309 | 13.6446% | 0.780x |
| [16, 24) | 23,106,327 | 19.5875% | 2,836.765 | 0.4637% | 0.0237x |
| [24, 32) | 71,646,886 | 60.7358% | 0 | 0% | 0x |
| [32, inf) | 0 | 0% | 0 | 0% | n/a |

The cumulative `[0,4)` region is 858,548 / 117,964,800 positions (0.7278%) and carries
366,247.188 / 611,766.317 WRONG-indicator bits (59.8672%), an 82.258x enrichment. `[0,8)` is
2.1768% of positions and carries 85.8917% of WRONG bits.

### G4 stationarity and geometric strata

| Disjoint G4 category | Wrong positions / 227,609 | WRONG-indicator bits / 611,766.317 | WRONG share |
|---|---:|---:|---:|
| STATIC_IN_IMAGE | 197,616 | 526,386.189 | **86.0437%** |
| STATIC_IN_XI_PROXY | 93 | 254.819 | 0.0417% |
| TRANSIENT | 29,900 | 85,125.309 | 13.9147% |

The xi proxy produced 46 tracks, 93 tracked events, maximum length 3. It is a target-cache
metric-Pose6 G1 translation-only proxy, not physical BEV and not decoder-free.

| Overlapping G4 geometric stratum | Positions / 117,964,800 | WRONG-indicator bits / 611,766.317 | WRONG share | Enrichment |
|---|---:|---:|---:|---:|
| lane corridor | 746,340 | 354,993.343 | 58.0276% | 91.717x |
| movable band | 1,097,448 | 72,599.114 | 11.8671% | 12.756x |
| hood rim | 52,371 | 7,212.740 | 1.1790% | 26.557x |
| decoded-token boundaries | 2,555,705 | 607,093.985 | **99.2363%** | 45.805x |

These strata overlap and are descriptive. Lane/decoded-target/G4 tags are not automatically legal
zero-byte conditioning inputs; using their realized labels would leak the answer or require address
side information.

### Pair rank and named cells

The top ten pairs carry 17,057.642 / 611,766.317 WRONG bits (2.7883%): pairs
`0, 522, 65, 70, 517, 515, 518, 72, 74, 67`. Pair 0 is a cold-start row and is not generalized
to later pairs. Full pair rank is in `RESULT.json`.

The exact top 1,000 positions are retained at
`/Volumes/APDataStore/pact/ddm_wh1_wrong_half_decomposition/measurement_v1/TOP_CONCENTRATION_POSITIONS.json`
(314,793 B, SHA-256 `294f2f9ad4c462ca3a55277eedc3b14d07587c901428ac9425de50a884c2ff13`).
The highest single cell is pair 582, row 248, column 367: DALI-GT Lane, decoded-token MyCar,
coder argmax Road, margin bucket `[16,24)`, G4 TRANSIENT, 21.9046 WRONG-indicator bits and
2.1067 WHICH bits. This is a receipt-backed diagnostic, not a sidecar address proposal.

## Conditional-entropy-gap price

The admissible diagnostic partition is coder-argmax class × exact margin bucket × fixed 16x16
spatial tile. Those axes are known to the decoder. For each cell `c`, the uncharged oracle bound is
`n_c * h2(k_c / n_c)` and the gap is actual indicator bits minus that bound. The full arrays are
retained at `retained/conditional_oracle_cells.npz`; the top 200 cells are summarized in
`CONDITIONAL_ORACLE_CELLS.json` (73,430 B, SHA-256
`01b28eb28d6c845da30b2bcf8cae6c05edaa8ec2f594b47f2cec904673e1e749`).

| Quantity | Bits | Bytes |
|---|---:|---:|
| Actual indicator cost | 888,986.747 | 111,123.343 |
| Empirical binary entropy bound | 896,831.863 | 112,103.983 |
| Aggregate gap | **-7,845.116** | **-980.640** |
| Sum of positive cells only | 11,271.796 | 1,408.975 |
| Offsetting negative cells | -19,116.912 | -2,389.614 |

The current adaptive coder already beats this coarse fixed partition by 980.640 B. Cherry-picking
only positive cells is not a jointly realizable gain, omits model/table bytes, and at 1,408.975 B is
still below the measured ~2,009 B GB1 model-axis ceiling. Therefore this result is **CONFIRMATORY**:
the GB1 body does not expose a new class×margin×tile conditioning lever, despite the very sharp
descriptive concentration. It does not contradict the broader 2,009 B ceiling and cannot be framed
as a candidate toward the 42,229 B demand.

Verdict scope: **FORMULATION** — the declared decoder-known class × margin × fixed-tile partition on
the exact GB1 20-family object. It does not prove that every causal conditioning feature is exhausted.

## Prior-law adjudication

The falsifiable prior is confirmed, not falsified:

- DALI-GT Lane carries 38.6460% of WRONG bits on 0.5856% of positions: 65.998x enrichment, above 3x.
- The first low-margin bucket carries 4.4017% on 0.0448% of positions: 98.357x enrichment; `[0,4)`
  carries 59.8672% on 0.7278%: 82.258x.
- The near-uniform falsifier (“no class reaches 2x”) does not fire; Lane and Movable reach 65.998x
  and 8.546x.

This map is valuable for diagnosing the moved object; it is not a GB1-body rate lever by itself.

## Stage-B consumer contract

**QUEUED-WITH-A-FIRE-ORDER.** Owner: MAIN-designated Stage-B producer. Consumer store:
`/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/stage_b/`. Fire only when MAIN selects
one Stage-A seed/window and its `ddm_s1a_stage_b_fingerprint_contract.v1` binds the moved runtime,
archive, n600 receiver-consumed token field, and exact archive/stream/token hashes. Run the committed
instrument in a new subdirectory with explicit `--source-runtime`, `--archive-sha256/--archive-bytes`,
`--stream-sha256/--stream-bytes`, `--truth/--truth-sha256/--truth-bytes`, and the pinned GT/V12
custody flags. First run `--max-new-stages 1 replay` as a bounded smoke; on exact token identity,
resume `replay`, then `analyze` and `verify`. Stage B consumes the resulting map as a diagnostic on
the moved object; it may not transfer any GB1 number, G4 label, or negative verdict without that
fresh n600 replay.

## Boundaries and dead paths

- No scorer ran; no score, distortion, archive mutation, candidate, Modal dispatch, or lane claim occurred.
- Calibration remains closed by HC1; WH1 did not rebuild it.
- A position sidecar remains closed by the measured address tax; concentration is not free addressing.
- The class×margin×tile conditioning formulation is closed on GB1 by the negative aggregate gap.
- G4 xi-proxy transport is negligible here and is not physical BEV.
- DX2/GB1 aggregate token attribution is settled; successors should not rerun it without a moved object.

Own-vehicle frontier: **gb1 — S 0.14811799921260607 @ 180,215 B `[contest-CUDA T4, n600]`**;
WH1 did not move the pointer.
