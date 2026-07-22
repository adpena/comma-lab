---
title: Codex round-1 review of Task 602 MDL member carrier preflight
utc: 2026-07-22T03:23:14Z
task: 603
feeds_task: 613
source_task: 602
lane_id: lane_ddm_mdl_member_carrier_n600_20260722
review_round: 1
main_landing_review_required: true
---

# Review disposition

`PASS_WITH_FORMULATION_BLOCKER`. The evidence supports only
`BLOCKED_602_OUTPUT_IS_NOT_A_RECEIVER_CARRIER` at `FORMULATION_OUTPUT_INTERFACE` scope. It supports no
carrier curve, membership value, uint8-loss value, Pose completeness, family negative, or score.

# Fresh re-derivation

- The full #602 receipt SHA is `b71ad6ab...` and its 64 stage files are contiguous and config-bound.
- All 64 stages say `selected_equals_canonical=true`, `changed_values=0`, and
  `selected_frame_payload=null`; the `selected_frames` directory does not exist.
- The solver and producer are SHA-bound. Their selected arrays are uint8 camera frames; a changed
  selection would be stored as an NPZ, while canonical selection reconstructs only by rereading the
  source raw.
- The target receipt named “full precision” explicitly records `plane_dtype=uint8`; it is full-
  resolution custody, not a preserved floating member state.
- #602's 77,651,017 B is `zlib-9 over uint8 camera tile bytes`; its own seed scope says raw-member
  zlib is diagnostic. D4 records `activated=false` and `n600_estimate=null`.
- The prior #603 archive is a different smooth-chart payload with measured 0.494 membership, not an
  adapter for #602 outputs.

# Attempts to falsify the blocker

1. **Use canonical source raw as the selected payload:** rejected. It is a 3.66 GB solver input, not
   a persisted coded-member description, and would silently replace the requested carrier format.
2. **Use the 78,969-byte compressed seed:** rejected. #602 explicitly says it is not a complete
   solved-pair archive and the member policy adds 416 B without removing a seed field.
3. **Wrap diagnostic zlib in ZIP:** rejected. This invents a receiver and changes the priced object;
   77,651,017 B is already 351.07x the entire 216 KiB knee at n64.
4. **Use the proven #603 receiver:** rejected as a carrier claim. Its payload is the decisive
   constant-classifier-equivalent smooth grammar, not #602's selected member.
5. **Infer n600 from n64:** rejected. #602 explicitly leaves `n600_estimate=null`, and the measured
   solver time projects beyond the delegated bounded re-derivation window.
6. **Call uint8 loss zero:** rejected. With no independent pre-uint8 state, the loss is undefined.

# Review boundary for MAIN

MAIN should confirm that absence of a coded payload—not canonical membership itself—is the blocking
interface fact; that the diagnostic bytes remain outside the #613 curve; and that no new primary
green row or canonical equation is registered. A successor is free to repair the formulation by
emitting a real payload, but it may not reinterpret this receipt as a family kill.

Pointer `0.1910828242 [contest-CPU]` unchanged. No scorer, dispatch, candidate, or provider action.
