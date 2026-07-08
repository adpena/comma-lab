# SEAL v7.3 ROUND-2 SYNTHESIS (2026-07-08) — 4 lenses, tally, arbitrations, fix wave

STORES CONSULTED: seal_v73_r2_bugs (b77be61ee) · seal_v73_r2_deepmath (5e5efe042) ·
seal_v73_r2_confound (106e77b84) · seal_v73_r2_structure phase-1 (1e91081c7, BLIND-sealed
pre-memo) + phase-2 (bb5b33a0a) · SYNTHESIS_INCL + round-1 synthesis · crucible_v73_compile
memo · operator decisions (ledger, verbatim). review_status: synthesis by main orchestrator,
fresh-eyes on all four reports. Pointer 0.19110 [contest-CPU] UNMOVED — all MEANS.

## TALLY: NOT_CLEAN ×3 + REVISE-then-PROCEED ×1 (counter = 0).
## 1 BLOCKER · 5 MAJOR (1 event-conditional) · ~14 MINOR/REVISE. Fix-all policy applies.

## THE BLOCKER (deep-math; verdict_scope: formulation)
`hosc_beta_end=10.0` is a CLOCK-frame endpoint (derived so β(726)=3.18 on the linear anneal);
under the DECIDED-EVENT schedule β rides the octave fraction and FREEZES ≈10.0 = the MEASURED
forbidden fixed-high-β saturation regime (tanh(β·sin) vanishing-grad divergence). Every
β≈3.18-anchored measurement would be invalid. The constant's re-derive trigger had fired and
was not honored. FIX: event-mode hosc_beta_end ≈ 3.18 (≤4.0), derivation recorded; the GPU
bit-cert's [1,10] sweep already covers the corrected endpoint.

## ARBITRATION: the event-vs-clock re-litigations (bugs MAJOR-1 · structure R2)
Both lenses recommend clock for run-1 attribution. The OPERATOR DECISION (2026-07-08 verbatim,
ledger) is EVENT NOW, risk accepted, new baseline — BINDING; the seal does not re-open it.
What the lenses legitimately caught: the DECISION RECORD is absent from the compile memo and
the launch package (a fresh reviewer sees an unexplained contradiction). FIX: the launch
package + compile memo carry the operator override verbatim + the risk framing (not-clean
baseline, attribution via per-stage checkpoints + would-fire telemetry). NOTE the deep-math
BLOCKER was the real hazard hiding under the mode question — fixed, event mode is coherent.

## FIX WAVE (two builders, all severities, zero-unfixed precondition for round 3)
**Builder A — config/math (witness_autoconfig + typed_config + docs):**
A1 BLOCKER: event-mode hosc_beta_end → derived ≈3.18 (≤4.0) + provenance + re-derive-trigger honored.
A2 budget anchor → amortized 3.12 min/ep (startup amortized over 3000; deep-math arithmetic),
   budget ≈7.48d; reconcile the 3.65-vs-3.62 comment (bugs REVISE-4); provenance corrected.
A3 Polyak degenerate clamp start=epochs+1 (bugs MINOR-1: NOT inert, observes once — fix the
   BEHAVIOR and re-point the test at behavior not constant) + tail off-by-one 456→455 (MINOR-2).
A4 decision-record: operator event-override verbatim into compile memo + launch package;
   two-token clock-revert documented (mode AND shape; deep-math MAJOR-2).
A5 structure M1 lane-regime coherence: under DirectionalBasisRebalance(lane_offloaded), gate the
   lane-driving learned losses (persistence-recall[1], island-amplify[1]) per the FEED-07a
   two-regime law — offloaded lane rides the analytic band, learned capacity de-emphasizes lane
   texture; document the regime coupling in the lever + config.
A6 structure M2: Road per-class d_seg named PRIMARY run signal (launch package watch-list,
   already pre-registered in ledger) + register a Road-fallback lever as duty-to-measure.
A7 structure R3: assert --per-group-grad-clip present in the v7 emitted argv (test).
A8 deep-math MINOR: extend s*=ν·forfeit reactivation criterion (ν stale on the new basis).
**Builder B — code (trainer/byte-close/kernels/telemetry):**
B1 confound F1: byte-close gains the polyak arm (weights_arm ∈ {ema, live, polyak}; selection
   RECORDED with per-arm scores — the missing consumer).
B2 confound F2: D16 pool fallback LOUD (marker row on dispatch-fallback) + fingerprint/version
   gate consistent with the safe-compile sibling.
B3 confound F4 (event-conditional but v7 IS event): fire telemetry records sensor-data epoch
   alongside fire epoch.
B4 bugs REVISE-1: perf-env token-boundary compare (=1 not satisfied by =10) + test.
B5 bugs REVISE-2: document/close the pre-fire manifest-free window (stamp-always evaluated vs
   documented-hole; pick with rationale).
B6 bugs REVISE-3: decouple the closed-loop guard tests from the literal source token.
THEN: v7.4 compile verification (dry-run chain green) → ROUND 3 (same 4 lenses on the diff).
