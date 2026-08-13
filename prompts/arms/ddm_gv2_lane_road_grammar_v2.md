# ddm_gv2 — Event grammar v2: pose-null-BY-CONSTRUCTION events on the Road↔Lane hub edge (the load-bearing seg leg)

**Charter date:** 2026-08-12 · **Owner:** codex arm (xhigh) · **Consumer:** MAIN re-fires the PROVEN vd1 n600 validator (unchanged instrument, ~$0.15/200 events, #381) on the gen-2 store.
**Parents:** ddm_vd1 run-c census (gen-1 REFUTED at n600 authority) + ddm_ec1 (gen-1 producer) + ddm_lc1 (the Lane↔Road receipts) + ddm_js4 (pose-null projector) + ddm_jo1 (+≤3 B carrier). Read `docs/operating_manual_craft_handoff.md` + CLAUDE.md/AGENTS.md first.

## Why gen-2 exists (the measured refutation to design against)

vd1 census (`/Volumes/VertigoDataTier/pact/ddm_vd1_20260812/main_harvest/results/FINAL_RESULT.json`, per-event exact T4, all 200 gen-1 events):
- Only 26/200 gain net flips; only 29/200 pass the pose budget; **intersection 5 events, +6 flips total = 5.09e-06 S** (bar was 2.16e-4; falsifier fired 42× under).
- **The binding constraint is POSE, not seg reach**: global stack budget 1.3e-7 d_pose (js7 calibration: marginal 603 S/unit at base 6.88e-6); flip-positive and pose-safe events are nearly disjoint in gen-1.
- Carrier is FREE and PROVEN (jo1: 200 events → +3 B via HP3/RC64). Selection/alphabet quality is EVERYTHING.

## The gen-2 design contract

1. **Target the hub edge**: Road↔Lane carries 49.2% of all base flips (m91, pc2); lc1 measured the failure shape — PE3 over-claimed 5,557 Lane→Road pixels (already-correct Road rewritten as Lane). Gen-2 events must fix BASE Lane/Road errors without over-claim: boundary-band-local token edits, small blast radius, sign-aware per edge (the lc1 lesson: dense per-class labels lose; sparse targeted events on the edge band are the open route).
2. **Pose-null BY CONSTRUCTION, not by post-filter**: gen-1 lost 21 of 26 flip-positive events to the pose budget. Compose the event proposal step with the js4 pose-null projector (landed; seg survives projection) and/or frame-0/Q3 placement (bo1 #889: Q3-constrained edits CANNOT create pose damage — exact kernel). Every emitted event carries a predicted pose bound; the store-level global stack prediction must be ≤ 1.3e-7 d_pose for the intended selection size.
3. **Per-event reach must be 10–100× gen-1's**: gen-1 positives averaged ~1 flip/event. Design events at the scale of boundary segments/dash cells (10–500 px), not single tokens. Optimistic eligible gain target ≥ 0.000216 S (the js7 damage scale) to justify a compose; stretch ≥ 1e-3 (the seg leg owes ≥ ~4e-3 of the sub-0.15 arithmetic).
4. **SAME STORE SCHEMA as gen-1** (`/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/realized_acceptance_200`: proposal_index.jsonl + proposals/ + state.json): the vd1 validator re-runs UNCHANGED with `--event-store` pointed at the gen-2 directory. Do not fork the validator.
5. **$0 local advisory pre-screen only** (frozen CPU-torch scorer on affected pairs is allowed as ADVISORY ranking; label `[macOS-CPU advisory]`; no authority claims; no Modal). Emit up to K=200 events ranked by predicted net S.
6. **P0 KEEP THE PAYLOAD**: retain every generated event payload + prescreen tensors under `/Volumes/VertigoDataTier/pact/ddm_gv2_20260812/`.

## Deliverable
Gen-2 event store directory (schema-compatible) + memo `.omx/research/ddm_gv2_lane_road_grammar_v2_20260812.md` with: alphabet design derivation (from the lc1/m91/pc2 receipts, not invented), per-event predicted (flips, pose bound, bytes), store-level optimistic arithmetic vs the 0.000216 bar, and the pinned MAIN re-fire command (vd1 validator, new --event-store, fresh --run-id). Serializer commits, post-edit shas.

## Falsifier (pre-registered)
If the best constructible gen-2 alphabet's OPTIMISTIC eligible gain < 0.000216 S under the honest pose-stack prediction, the sparse-event FAMILY on this base is FORMULATION-closed for the seg leg — report it plainly; the seg leg then routes to implicit edge conditioning (js1 stage-0 lineage), not to a gen-3.
