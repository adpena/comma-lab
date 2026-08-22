# ddm_ws0 worldsheet grammar price

## Verdict

**The 90,000 B worldsheet conjecture is REFUTED for this grammar formulation.** On the real cached n600 partition object, the best deterministic receiver-readable lossless row is **269,921 B** and the best tolerance row is **265,930 B** at **136,839 / 117,964,800 changed cells = 0.001159998575846354 equivalent mass**. Both pre-registered falsifier clauses fire:

- lossless `269,921 >= 130,000 B`;
- tolerance `265,930 >= 110,000 B`.

The verdict scope is **FORMULATION**: horizontal row-boundary worldsheets with a counted data-induced transition-rank table, per-edge topology/source streams, ordinal spatial or prior-pair curve prediction, ULEB coordinate innovations, and the tested boundary-coordinate quantizers. It is not a global lower bound over all possible partition programs.

Measurement axis: **[macOS-CPU advisory, scorer-free n600 coder]**. No scorer or Modal job ran, no score is claimed, and the exact frontier pointer did not move.

## Measured rows

All rows contain a complete receiver envelope and all 600 pair records. Every semantic stream raced Brotli Q11, raw LZMA1, and SMEVR R7; all three coder payloads were retained. Candidate selection is the measured minimum over the registered 12-row sweep, not a claim of global MDL optimality.

| Leg | Coordinate predictor | Quantizer | Bytes | Changed cells | Equivalent mass |
|---|---:|---:|---:|---:|---:|
| lossless | minimum absolute innovation | none | 274,556 | 0 | 0 |
| **lossless** | **spatial prior row** | **none** | **269,921** | **0** | **0** |
| lossless | prior-pair temporal | none | 318,885 | 0 | 0 |
| tolerance | minimum absolute innovation | q2 | 268,911 | 136,839 | 0.001159998576 |
| **tolerance** | **spatial prior row** | **q2** | **265,930** | **136,839** | **0.001159998576** |
| tolerance | prior-pair temporal | q2 | 313,076 | 136,839 | 0.001159998576 |
| tolerance | minimum absolute innovation | q4 | 269,641 | 136,839 | 0.001159998576 |
| tolerance | spatial prior row | q4 | 268,535 | 136,839 | 0.001159998576 |
| tolerance | prior-pair temporal | q4 | 314,524 | 136,839 | 0.001159998576 |
| tolerance | minimum absolute innovation | q8 | 279,198 | 136,554 | 0.001157582601 |
| tolerance | spatial prior row | q8 | 280,317 | 136,554 | 0.001157582601 |
| tolerance | prior-pair temporal | q8 | 324,222 | 136,554 | 0.001157582601 |

The tolerance allowance saves only **3,991 B (1.4786%)** from the best lossless row. The best lossless row is **179,921 B above** the conjecture and **96,305 B (55.47%) above** pp1's 173,616 B direct-partition ceiling. The best tolerance row remains **175,930 B above** the conjecture.

## Per-stratum price

These are envelope bytes for the best rows. Event bytes include each edge's source/event record and its stream header; coordinate bytes include the corresponding innovation payload and header. Birth, death, and persistence counts are structural measurements over the lossless n600 object. Shared topology is charged once below the edge rows.

| Stratum | Births | Deaths | Persists | Lossless event B | Lossless coord B | Lossless total B | Tolerance total B |
|---|---:|---:|---:|---:|---:|---:|---:|
| Road↔Lane | 70,416 | 70,469 | 149,389 | 11,142 | 132,950 | **144,092** | **142,462** |
| Road↔Undrivable | 2,332 | 2,317 | 32,326 | 2,140 | 20,753 | 22,893 | 21,674 |
| Road↔Movable | 4,904 | 4,944 | 29,215 | 3,892 | 17,550 | 21,442 | 21,282 |
| Road↔MyCar | 3,063 | 3,064 | 11,571 | 1,827 | 14,167 | 15,994 | 15,546 |
| Lane↔Undrivable | 72 | 72 | 18 | 123 | 227 | 350 | 350 |
| Lane↔Movable | 488 | 489 | 435 | 402 | 1,024 | 1,426 | 1,443 |
| Lane↔MyCar | 369 | 369 | 49 | 247 | 820 | 1,067 | 1,050 |
| Undrivable↔Movable | 4,563 | 4,531 | 29,800 | 2,907 | 23,011 | 25,918 | 25,386 |
| Undrivable↔MyCar | 0 | 0 | 0 | 26 | 26 | 52 | 52 |
| Movable↔MyCar | 14 | 14 | 10 | 51 | 83 | 134 | 132 |
| **Shared topology/header/rank table** | — | — | — | **36,553** | — | **36,553** | **36,553** |
| **Envelope total** | — | — | — | **59,310 grammar/events** | **210,611 innovations** | **269,921** | **265,930** |

Road↔Lane alone is **144,092 B**, or **53.38%** of the best lossless object. It already exceeds the 130,000 B lossless falsifier threshold before the other nine strata are added. This is consistent with the recalled one-graph/one-hub law, but the price here is a new same-object receiver measurement rather than a transfer of that earlier statistic.

## Receiver and custody proof

The receiver parses the counted 20-byte induced transition-rank table, validates the complete 21-stream roster and ordered stream identifiers, decodes exactly 600 records per stream, reconstructs every boundary coordinate, rejects missing curve ordinals, invalid coordinate order, unconsumed edge occurrences, padding, truncation, and trailing bytes, and renders the class partition deterministically.

- Lossless parse-back: 600/600 pairs; 0 changed cells; decoded and original uint8 SHA-256 both `f2c8be94774780bda718adf337900403a8533b6ffa1352b5aae19e200a005557`.
- Tolerance parse-back: 600/600 pairs; semantic round-trip true; 136,839 changed cells, below the integer cap 136,839; decoded uint8 SHA-256 `0eaa9833a576a1277fb41eb1343385bb565a9f9a5c2d18cbc6051bd05edee412`.
- Input: `gt_n600.npz`, 5,078,017,610 B, SHA-256 `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`, member `lstars.npy`, shape `600x384x512`, dtype `int64`.
- Durable store: `/Volumes/VertigoDataTier/pact/ddm_ws0_worldsheet_grammar_price/retained/`, 1,382 files, 20 MiB at audit time. It contains 600 per-frame extraction checkpoints, 12 candidate receipts, 12 receiver envelopes, and every per-stream coder payload.
- Final result: `FINAL_RESULT.json`, 102,155 B, SHA-256 `aec00c112bbfc4bc914716b87969b0f8975ed659980ece8f5859a11ebdbe966a`.
- Best lossless envelope: 269,921 B, SHA-256 `6dcc93995d91796badf56c573608f4a3fa96b9adf1ddf3ea4d222335ab6c41ba`; receipt SHA-256 `d282294e924d0ca771028e5752b090be8ca1ab8f26b591e1deef43262d350ae9`.
- Best tolerance envelope: 265,930 B, SHA-256 `8c5dfea216ea801dc345e0e23681cf56a6722a4c4bd0c6021e5225e25af87cfe`; receipt SHA-256 `27d90661dded0d8e951417436edd2876c0ff552c23136f66d35358539c0c504b`.
- Extraction checkpoint manifest: 81,393 B, SHA-256 `d0c2a380874e0bdded25a743e752e1ea2c910a8faee3b60afca7502fb8136cdc`.
- Producer source SHA-256: `e43383df037145db9eb4081efe30a0f0771ac31bde9cd656b31a561f2e2efb4b`.

An independent post-run audit recomputed byte counts and SHA-256 for **768 / 768 retained artifacts across 12 / 12 receipts**. A second `--resume-from` execution loaded **600 / 600** frame checkpoints and re-ran every receiver verification without re-encoding. Storage preflight passed with 68,073,029,632 free bytes. `ps` and `top` visibility were sandbox-blocked, so the launch was hard-capped at eight workers: using the charter's 288% jo1 observation, nominal combined load stayed at or below 1,088%, under the 1,300% cap. The sacred jo1 run directory was not touched.

Run command:

```text
nice -n 10 env PYTHONUNBUFFERED=1 .venv/bin/python experiments/ddm_ws0_worldsheet_grammar_price.py --workers 8 --output-dir /Volumes/VertigoDataTier/pact/ddm_ws0_worldsheet_grammar_price/retained --resume-from /Volumes/VertigoDataTier/pact/ddm_ws0_worldsheet_grammar_price/retained
```

The sandbox refused the `nice` priority adjustment, but the pricing process itself completed. No artifact was moved or deleted.

## RECALL EVIDENCE

The recall preceded design and covered the full requested surfaces, not only the charter anchors:

- Research memos and receipts: content searches for `worldsheet`, `temporal transport`, `grammar induction`, `direct partition`, `contour support`, `Road hub`, `explicit curve`, `birth/death`, `prefix bias`, `173616`, `421366`, and `106465` across `.omx/research/`.
- Canonical equations: `.venv/bin/python tools/list_canonical_equations.py --json`, narrowed to `worldsheet_transport_residual_event_rate_v1`, `partition_temporal_transport_amortization_jitter_bound_v1`, and `v8_geometric_rate_decomposition_v1`.
- Research index and graph memory: searches of `.omx/research/CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_*` FEED blocks for `g1`, `pp1`, `sp1`, `m91`, `worldsheet`, `temporal`, `curve`, and `Road↔Lane`.
- Design/task surfaces: searches of the v8 specifications, `.omx/state/main_hot_state.md`, and task-ledger rows for `ddm_ws0`, `#620`, `#744`, `#1182`, and `#1187`.

The charter seeds were confirmed: g1's exact row-run prices were 59,481 B Movable, 180,701 B Lane, and 453,123 B boundary, with lower lossy knees that were candidate-set prices rather than a complete partition floor; pp1's complete direct partition was 173,616 B; sp1 explicit support was 421,366–444,394 B; and m91/pc2 identified Road as the 87.8% hub and Road↔Lane as the 49.2% dominant edge. The m88/m96 prefix warning forced the verdict onto all 600 pairs.

Three relevant findings beyond the charter seeds changed the plan:

1. `partition_temporal_transport_amortization_jitter_bound_v1` had found adjacent temporal raster transport worse than per-frame coding under a zlib proxy, but explicitly left curve-domain event coding open. That caused the same receiver object to race spatial, temporal, and minimum-innovation coordinate modes rather than assuming temporal amortization would win.
2. PE1's 106,465 B `explicit_curve_k16` row had only 0.969917 recall and no surviving complete receiver. It was retained as evidence that curve coordinates can be locally cheap, but excluded as a floor or ceiling control.
3. GV2 had formulation-closed sparse Road↔Lane token events on CP135. That prevented treating sparse event labels as geometry and kept this prototype focused on full boundary coordinates plus topology.

Within the searched corpus, I did not find a complete, receiver-closed, full-n600 worldsheet at or below 90,000 B. The search did find partial and lossy objects, so this statement is deliberately bounded to the listed stores and requirements.

## Validation and review

- Two genuine source review passes were registered for both new Python files. Pass 1 found and fixed stale-resume provenance, a non-serializable checkpoint value, missing exact-leg fail-closure, and redundant recoding. Pass 2 audited the completed receipts, receiver/accounting paths, and resume behavior without further source changes.
- `ruff check`: passed.
- `py_compile`: passed.
- Focused tests: `9 passed` in 26.12 s, including all predictor modes, real three-coder identity, exact receiver round-trip, quantization error caps, corrupt roster and padding rejection, and pickle-free checkpoints.
- Strict `check_no_measure_and_discard_payload` on the new runner: 0 findings.

## Consequence and disposition

The es1 portrait cannot continue to carry a 90,000 B partition-worldsheet assumption for this formulation. At the registered tolerance allowance it is short by **175,930 B**; at lossless it is short by **179,921 B**. That difference must be absorbed by the model/context residual or avoided by a genuinely different representation before nr1 builds a candidate.

**QUEUED-WITH-A-FIRE-ORDER:** owner `MAIN / nr1 (#1187)`; consumer store is this memo plus `/Volumes/VertigoDataTier/pact/ddm_ws0_worldsheet_grammar_price/retained/FINAL_RESULT.json`; fire trigger is the jo1 endpoint, before nr1 materializes its first archive. The action is to replace the 90,000 B conjectural row with the two measured prices and either re-balance the design or refuse the unchanged build.

**GESTALT-DELTA:** the campaign state changes from “90 KB worldsheet is the single unmeasured uncertainty” to “this receiver-closed worldsheet formulation is measured at 265.9–269.9 KB and falsified; Road↔Lane coordinate innovations are the dominant 142.5–144.1 KB obstruction, so nr1 must absorb the gap or change representation before firing.”

## LIVE-HYPOTHESES

- A persistent curve-identity grammar with explicit split/merge lineage may reduce coordinate innovations more than ordinal same-edge matching. It is plausible because g1 measured subpixel median transport residuals while this receiver used only row/transition ordinals; it remains untested as a complete n600 receiver and would need a qualitatively new object to overcome the 175,930 B gap.
- A joint parametric Road-hub generator with counted exception residuals may compress several Road-incident strata together. It is plausible because Road participates in most boundary structure and Road↔Lane dominates this measured envelope, but prior curve-relative and sparse-token formulations were negative, so neither their numbers nor this hypothesis authorize a rerun of those forms.

## DEAD-ENDS

- Prior-pair ordinal temporal prediction is closed for this formulation: it costs 318,885 B losslessly and 313,076 B at the best q2 tolerance row, both worse than spatial prediction.
- Boundary rounding at q2/q4/q8 is closed as a route to 90 KB here: the best legal tolerance row saves only 3,991 B while spending essentially the entire 0.00116 mass allowance.
- Re-racing the same 21 streams with Brotli Q11, raw LZMA1, or SMEVR R7 is closed: every stream already raced all three real coders and retained all payloads.
- The tested horizontal-row worldsheet family is closed against the registered conjecture: all three lossless predictor modes exceed 130,000 B and all nine tolerance modes exceed 110,000 B.
