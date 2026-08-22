Executed and committed as `4a49821f8f`.

Main result: NR1’s QPAIR addressing stream has a real lossless win. Generic tile-major time ordering reduces it from 52,040 B to 34,083 B with Brotli q11: **17,957 B saved**.

- RC1 tested addressing layouts: **0 B saved**
- NR1 QCTX layouts: **0 B saved**
- NR1 QEVENT split: **0 B saved**
- DX2 fixed-representation recoding: remains **0 B**, as RB1 established
- Rule-118 witnesses proved RC1 assignments and all three NR1 addressing surfaces are not receiver-derivable; only their traversal order is free.

NI1’s landed archive is 122,250 B. Applying the isolated QPAIR delta projects 104,293 B and a 0.0119568 rate-score reduction, but this is not yet an integrated archive or score.

Artifacts:

- [Evidence memo](/Users/adpena/Projects/pact/.omx/research/ddm_ad2_addressing_cost_decomposition_20260822.md)
- [Measurement script](/Users/adpena/Projects/pact/experiments/ddm_ad2_addressing_cost_decomposition.py)
- [Authoritative result](</Volumes/APDataStore/pact/ddm_ad2_addressing_cost_decomposition/measurement_v6/RESULT.json>)
- [Sealed fire order](</Volumes/APDataStore/pact/ddm_ad2_addressing_cost_decomposition/measurement_v6/SEALED_FIRE_ORDER.json>)

All 199 listed artifacts rehashed successfully. Two review passes and the payload-retention detector passed. Superseded v2–v5 trees were copy-verified to Vertigo cold storage, reclaiming 2,470,753,954 APDataStore bytes.

No scorer, Metal, Modal, or advisory evaluation ran. No live candidate or `upstream/` file changed.

OWN-VEHICLE FRONTIER: UNMOVED — AD2 S NOT MEASURED; DX2 remains S=0.14821987563243377 @ 180,368 B [contest-CUDA T4 n600].

## NEXT_IF_RESUMED

- `ad2_qpair_tile_time_receiver_integration`; disposition=`QUEUED-WITH-A-FIRE-ORDER`; owner=`MAIN assigns ddm_ni2_nr1_qpair_tile_time_receiver`; consumer store=`/Volumes/APDataStore/pact/ddm_ad2_addressing_cost_decomposition/qpair_tile_time_receiver_r1/`; fire trigger=`NI1 build_r4 scoring is terminal or MAIN explicitly forks an isolated successor, all pins revalidate, and the shared index is empty`.
- `ad2_qpair_tile_time_main_scorer`; disposition=`QUEUED-WITH-A-FIRE-ORDER`; owner=`MAIN scorer-lane dispatcher`; consumer store=`/Volumes/APDataStore/pact/ddm_ad2_addressing_cost_decomposition/qpair_tile_time_receiver_r1/harvest/`; fire trigger=`the integrated archive has repeat-identical full-RGB decode matching NI1 K32 and MAIN holds the sole n600 slot`.

## LIVE-HYPOTHESES

- Most or all 17,957 B should survive archive integration because the representation change needs no video-derived side information.
- Tile-major ordering may also improve K64, NI1’s lower-distortion fallback.
- A fixed richer causal QPAIR transform may improve on 34,083 B without introducing counted model state.

## DEAD-ENDS

- RC1 block ordering and fixed-11 packing were all worse.
- NR1 QCTX ordering and fixed-5 packing were worse.
- QEVENT address/value splitting added 16 B.
- QPAIR fixed-6 packing and pair-major block order lost to tile-time u8.
- DX2 HPAC’s entropy-model gap does not reopen RB1’s measured 0 B coder result.
- Assignment IDs, QCTX IDs, QPAIR choices, QEVENT corrections, and HPAC tables cannot legally become free receiver code.

