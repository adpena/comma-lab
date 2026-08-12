# ddm_js6 — REALIZED ACCEPTANCE over ec1's event-coordinate proposals (the seg-leg join CLOSES)

The join the whole seg leg demanded is now stocked: ec1 (fb732e7579) delivered
200/200 RECEIVER-EFFECTIVE, content-distinct, representation-level proposals
(151 boundary-offset · 48 lane-program · 1 island-death) at
/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/
realized_acceptance_200 — exactly what js5's fire-order 1 and tf1's fire-order
2 named. This arm runs the EXISTING js5 pose-gated robust-improvement
acceptance loop over that store WITHOUT regenerating payloads (ec1 fire-order
1, MAIN-routed).

## MISSION
1. Verify the ec1 store state + source archive SHA (custody first; 200 unique
   proposal IDs + payload hashes already receipted — reverify, never trust).
2. Run the js5 realized-acceptance loop (experiments/
   ddm_js5_projector_distilled_conditioning.py machinery — REUSE; grep its
   argparse for the consume/acceptance path; if none exists, the SMALLEST
   adapter that feeds stored proposals into its acceptance stage, 2 review
   passes) over all 200 proposals: per-proposal realized (robust Δflips at
   δ=0.08036041259765625, pose delta on custody planes vs the 2e-6 gate,
   bytes) through the real receiver→R→uint8 chain.
3. STOP conditions per js5's F1 protocol: first useful nonzero bare admission
   (a proposal that moves robust flips at pose-pass) OR all 200 measured.
4. Emit the ACCEPTANCE TABLE: accept-rate, per-family (boundary/lane/island)
   yield, B/robust-flip vs the 1.28 economics bar, and the F1 adjudication
   js5 could not make (15/200 then; 200/200 now possible).

## BINDING LAWS
Payload P0 sha256+bytes to /Volumes/APDataStore/pact/ddm_js6_20260812/;
instrument doctrine unchanged (relative Δflips, baseline 50,389 @ batch16/
8-threads, [macOS-CPU advisory, floor 0.0131 S]; pose custody planes);
REUSE js5 loop + js4 projector + ec1 store — recall, never rebuild;
serializer --no-co-author post-edit shas [no-triality] [p0-ledger-ok];
2 review passes per .py; bounded ≤40 min per governed run; CPU only (Metal is
occupied by xi1 Leg A — do NOT touch MPS); blocked-git → commit_intent.

## FALSIFIERS
F1 (inherited, now decidable): acceptance < 5% over the 200 AND no useful
bare admission → the pose-null seg-reachable overlap is unusable even with
representation-level proposals → seg training family closed at this endpoint;
route residual seg value to se1's survival curve. F2: acceptance ≥ 5% but
B/robust-flip never trends toward 1.28 → exchange-rate ceiling, report curve.
