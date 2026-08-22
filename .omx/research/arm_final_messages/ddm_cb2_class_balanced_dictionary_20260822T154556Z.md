Executed the CB2 charter in full.

The mandatory step-2 premise was refuted, so no dictionary refit or candidate payload was permitted:

- Lane area: **0.585848%**
- Existing K=2,048 Lane capacity: **5.252197%**, or **8.965× area**
- Incremental K=256→2,048 Lane capacity: **5.974981%**
- Lane share of removed mismatches: **17.317524%**
- Lane agreement remains poor at **14.729089%**, but area-proportional capacity is not its measured cause.

RI1’s terminal n600 receipt landed during closure and was consumed read-only. It contains only aggregate env-mismatch-advisory results (`d_seg=0.01605413`, `d_pose=24.41603851`); per-pair retention failed and no per-class breakout exists.

Artifacts:

- [Memo](/Users/adpena/Projects/pact/.omx/research/ddm_cb2_class_balanced_dictionary_20260822.md)
- [Runner](/Users/adpena/Projects/pact/experiments/ddm_cb2_class_balanced_dictionary_gate.py)
- [Authoritative result](/Volumes/APDataStore/pact/ddm_cb2_class_balanced_dictionary/measurement_v5/RESULT.json), SHA-256 `5e271484…c007682`
- [Sealed fire-order](/Volumes/APDataStore/pact/ddm_cb2_class_balanced_dictionary/measurement_v5/SEALED_FIRE_ORDER.json), SHA-256 `ad75f3dc…ae2d8`
- Independent repeat matched all **16 scientific blocks**.

Commits `52ccf0feb6` and `af327aa570` are on local `main`. Two fresh review passes, policy-check, Ruff, drift refusal, and lane registration checks passed. The full developer preflight still reports eight existing repository-wide red gates, but every red gate has **zero CB2 hits**.

No scorer dispatch, Modal job, refit, candidate payload, or exact evaluation was launched. The frontier did not move.

## NEXT_IF_RESUMED

- **Disposition:** `QUEUED-WITH-FIRE-ORDER, BLOCKED`; **owner:** `MAIN scorer-lane owner`; **consumer store:** `/Volumes/APDataStore/pact/ddm_cb2_class_balanced_dictionary/measurement_v5/main_fire/`; **fire trigger:** a retained current-DX2 per-class scorer breakout exists, a differently motivated fixed-K=2,048 candidate has a complete retained receiver archive at or below 113,006 B, and MAIN owns the unique n600 scorer lane; **action:** compare baseline RC1 and that candidate through the same RI1 receiver, admitting nothing from agreement alone.

## LIVE-HYPOTHESES

- Lane failure may be allocation topology rather than total capacity: Lane appears in 1,527 codewords and receives 8.97× its area share, yet no codeword is Lane-plurality and Lane agreement is only 14.7%.
- A current-DX2 boundary-debt objective may outperform population Hamming at fixed K because Lane supplies 41.5% of mismatches despite 0.59% area.
- Ordered class-transition costs may matter more than scalar inverse-frequency weights; the ancestor G4 transition distribution and RC1 mismatch distribution differ substantially.

## DEAD-ENDS

- Re-fitting because K=2,048 allegedly tracks class area is closed: the measured allocation does not.
- Raising K is closed for this arm: K=4,096 costs 158,933 B, above the 137,986-byte ceiling.
- Treating 98.796% overall agreement as evaluator evidence is closed; Lane agreement is 14.729%, and CB2 ran no scorer.
- Treating G4 as current-DX2 sensitivity is closed; it is ancestor-v12 advisory evidence.
- Treating JG3 or task 869 as complete sensitivity corpora is closed; required DROP/Pose/byte or scorer-A/B evidence is absent.
- Inventing a uniform class prior, 19% Lane constant, or per-class compressed-byte allocation is closed because none is retained current-vehicle evidence.

**CB2 own-vehicle frontier line:** **S = 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`**; CB2 delta **0**, pointer unmoved.

