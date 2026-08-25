# ddm_s1e_off_floor_adjudicator — build the push-button both-OFF endpoint adjudicator for the s1a stage-A burn (harness task #1270, owning memo ddm_s1_trained_renderer_diagonal_20260825.md)

## MANDATE

Operator 20260825: *"I want you to do whatever it takes and work for as long as it takes
autonomously with full authority and stand and go to accomplish frontier score lowering"*.
The s1a stage-A wd3 burn (2 OFF seeds 20260815/20260816, 65 epochs each, ~98 s/epoch) is
LIVE and reaches its both-OFF endpoint in ~1.5 h. At that boundary MAIN must adjudicate the
pre-registered fb1 renderer-corner falsifier BEFORE authorizing ON-15 (sealed
MAIN_LAUNCH_ORDER sha 708eae6a). Today that adjudication is prose arithmetic; this arm
builds it as a typed INSTRUMENT so the endpoint review is push-button, receipt-backed, and
zero-latency — the critical path loses no wall-clock at the boundary.

## SCOPE

1. Build `tools/s1a_off_floor_adjudicator.py`: read BOTH seeds' run artifacts under
   `/Volumes/APDataStore/pact/ddm_s1a_stage_a_adapter/training/off_seed_{20260815,20260816}/W96_flattened/`
   (evaluations/epoch_*_n60.json advisory rows, checkpoints, STAGE_CONTROLLER_RESULT.json
   quantization-race table) and emit per-seed × per-checkpoint typed rows:
   {bytes_shed_vs_gb1_renderer_30856B, hard_d_seg, d_pose, composed ΔS vs break-even at
   6.658e-7 S/B, falsifier_verdict}. Authority paths: the gb1 pointer body renderer block =
   30,856 B (ddm_ar1b decomposition, cited in ddm_wa1_week_audit_gestalt_toy_orphan_synergy_20260825.md);
   exchange rate 6.658e-7 S/B (ddm_tl1/tx1 derivation, hot-state POINTER_LINE).
2. bytes_shed MUST come from the REAL allocation table in STAGE_CONTROLLER_RESULT.json
   (chosen_allocation + cheap_to_shrink_ladder base_bytes), never a projected count.
   Advisory d_seg/d_pose rows are n60-subset [Darwin-mps frozen-scorer advisory] — label
   every row with that axis; the instrument NEVER emits a score claim (score_claim=false).
3. Emit the falsifier table per the charter: fb1 corner = renderer byte credit vs measured
   pose+seg damage per reachable operating point, BOTH seeds. Verdict enum:
   {CORNER_CROSSED_AT_LEAST_ONE_POINT, ENTERED_AND_REFUSED_ALL_POINTS, INCOMPLETE_DATA}.
4. Run it LIVE against seed 20260815's finished artifacts as the executed positive control
   (the seed finishes within this arm's lifetime); retain the output JSON to the arm store.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_s1e_off_floor_adjudicator/`.
- The live r7 run dirs are READ-ONLY to this arm (a live trainer owns them); read files,
  never write/move/delete anything under the training/ tree. Do NOT touch
  experiments/ddm_wd3_scorer_aware_width_distillation.py (the live trainer's builder pin
  0b976d0d0a would be invalidated by any edit — the r5/r7 birth-contract lesson).

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_fb1_sub012_feasibility_bound_20260823.md` — renderer→ZERO bytes alone lands
  S ≈ 0.12757 > 0.12: stage A output is an INPUT to stage B, never a candidate; the
  adjudicator must NOT frame any OFF row as a submission candidate.
- `ddm_w72_distortion_advisory_20260823.md` — W72 renderer rung refused 922× (S-ratio,
  corrected ladder per task #1245/#1264); pose was 65.3% of damage — the pose column is
  the decisive column, never omit it.
- `ddm_dg2_diagonal_distortion_verdict_20260824.md` — EDIT-diagonal refused 686× with pose
  93.3% of damage; per-window pose tracking is the discriminator between the trained and
  edit diagonals.
- `ddm_na2_negative_audit_20260803.md` — n60 advisory subsets carry pose-axis prefix bias
  (pose prefixes 2.54–4.21× harder); the instrument labels subset rows advisory and never
  converts them to population claims.

## OPTIMAL FORM

- Family exemplar: the distortion-aware checkpoint selector `tools/select_hpac_checkpoint.py`
  (reference implementation, commit 5624ef8bdc — argmin joint-S over per-stage checkpoints,
  distortion-protected) composed with the fb1 break-even arithmetic
  (`ddm_fb1_sub012_feasibility_bound_20260823.md` receipt).
- SCOPE reductions declared per row (n60 advisory input is what exists mid-burn — declared,
  labeled). MECHANISM reductions FORBIDDEN (no hand-typed byte counts, no omitted pose column).
- **PRIOR-LAW PREDICTION (falsifiable):** fb1's corner arithmetic + the epoch-5 advisory read
  (hard_d_seg 8.43e-4, d_pose 1.67e-2, pose contribution 0.408) predict the OFF floor's
  binding term is POSE: at every checkpoint the pose damage term will exceed the seg term in
  composed ΔS. FALSIFIER: any checkpoint on either seed where the seg term exceeds the pose
  term in the composed delta — count it plainly if it lands; it would re-rank stage-C's
  compensation priority.

## DELIVERABLE

`.omx/research/ddm_s1e_off_floor_adjudicator_20260825.md` — the instrument (committed, 2 review
passes) + the executed seed-20260815 positive-control output (typed rows, receipt path on the
arm store) + the falsifier-table schema MAIN consumes at the endpoint. Commit via the
serializer. End with the own-vehicle frontier line.
