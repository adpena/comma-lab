# ddm_mc1_motion_compensated_previous_plane — a DECODER-DERIVABLE motion-compensated previous-field plane as a new INPUT to the trained HPAC context mixer (constant-velocity from fields t−2→t−1, zero bytes), ceiling-priced in closed form first, then retrained + exact-priced through RC64 (lossless: field bit-identical ⇒ distortion held)

## MANDATE

Operator 2026-09-03: standing GO, "break walls", "synergies". THE GOAL: −42,016 B at held distortion.
Every rate rung on the token stream is measured shut for its own class: coder swaps 0 B (jt23),
categorical/positional context buckets ≤211 B held-out (`ddm_mi1_indicator_model_axis_20260824.md`), a 21-tap oracle floor of
144,167 B above the shipped 113,411 B stream (`ddm_dc1_decode_budget_conditional_coding_20260816.md`)
— which also states the MECHANISM: HPAC wins because it conditions on strictly more through
LEARNED WEIGHTS (7×7 + dilated depthwise + full previous frame + patch FiLM), "affordable only as
learned weights". The shipped mixer consumes the previous pair's field CO-LOCATED:
`submissions/semantic_joint_ctxmix/cpr1/hpac_integer.py` (7ba53d1b84) `prepare_frame_context`
one-hots `previous_raw` into `conv_past`. Between pairs the ego-vehicle moves ~0.1 s; the classes
that carry the bits — Lane (33.56% of model bits, co-located temporal IoU 0.263), Movable (IoU 0.903),
Road edges — MOVE in the image, so the co-located plane is misaligned exactly where the bits are.
Two prior negatives touch this and neither measured THIS form: xi1 (`ddm_xi1_carried_xi_inter_race_20260729.md`)
added a pose-warp context to a COUNT-BASED value model and lost +12,262 B by context dilution
(16× more contexts); d3b's `field_geometry_temporal` rows used a FROM-ZERO online log-odds mixer
(358 KB, the dc1 mechanism in reverse). DDS1's d1/d2 tested co-located previous-state BUCKETS
(0.9–4.1% overlap). Nobody has fed a motion-compensated previous plane to the TRAINED mixer. This
is the unmeasured door on the closest wall; it composes with everything (lossless, address-free,
bit-identical field ⇒ pose-null by construction).

## THE OBJECT

- Motion estimate m_t for pair t derived ONLY from already-decoded fields: constant-velocity
  extrapolation of the transform that best aligns field_{t−2} → field_{t−1} (start with a global 2-D
  translation; second candidate: a row-dependent horizontal/vertical shift for the ground plane
  — the flow of a planar road scales with image row; third: an affine). Deterministic INTEGER
  estimation (e.g., argmax over a bounded shift window of class-agreement counts) so the receiver
  reproduces it bit-exactly; pairs 0–1 fall back to co-located. ZERO archive bytes.
- MC plane = warp(field_{t−1}, m_t) as an additional one-hot input stacked with the co-located plane
  (keep both; let training weight them) into `conv_past`, or a second `conv_past_mc` branch summed
  before requantize — choose by the ceiling's per-class signal; declare which.
- Everything else unchanged: groups, scan, RC64, renderer, pose carrier. The field is bit-identical
  ⇒ d_seg/d_pose HELD; only the stream + model bytes change.

## SCOPE (ceiling → retrain → exact price; every payload retained)

1. **Closed-form ceiling FIRST (m118, /bin/zsh, minutes).** On the retained exact AFR1 field
   (`/Volumes/APDataStore/pact/ddm_jbp1_joint_batch_price/retained/exact/null/` — read the sha from
   its receipt; `/Volumes/APDataStore/pact/ddm_jbp1_joint_batch_price/retained/fields/sfp1_null_empty.u8`
   is the null overlay, NOT the field): for each candidate motion model, per class and overall,
   (a) alignment IoU of the MC plane vs the co-located plane against field_t; (b) conditional
   codelength of field_t under the SAME cross-fitted categorical family mi1 used, with contexts
   {co-located prev class} vs {co-located prev class, MC prev class} (held-out, pair-level two-fold)
   — report SCREEN bits and the ideal-coder ceiling in bytes (ceiling/8 is REFUSAL-ONLY use). Refuse
   at the ceiling if the best model's ideal saving is < 5,000 B: typed CEILING-REFUSED with numbers.
2. **Retrain the integer HPAC** with the MC plane using the existing same-schedule 60-epoch HPAC
   training + exact IHS1 pack path (`ddm_jf1_joint_field_model_refit_20260823.md`, `ddm_cl1_capacity_20260809`, custody
   `/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/` — locate the trainer entry at source;
   never invent a flag). Train from the shipped weights (warm start) with the new branch initialized
   to zero so the null behaviour is the shipped model; seeded; per-epoch checkpoints; EMA per the
   non-negotiable. Export bit-identically through `integer_model_io.py`; COUNT the model bytes.
3. **Exact price through RC64** (the JG2/RXC1 instrument; `experiments/ddm_rxc1_restartable_exact_coder.py`
   9cf2fd5d82 as the exact re-encode reference): full-state encode of the 600-pair field; receiver
   copy (under this arm's tree; the shipped packet is FROZEN) decodes it back byte-identically; two
   encodes identical. Report: stream bytes, model bytes, Δ vs 113,411 + 13,515, decode wall-clock
   delta on CPU (the MC estimator + extra branch), and the fraction of the 42,016 B demand.
4. **Decision rule (pre-registered):** receiver-closed archive ≤ 137,986 B → typed FIRE ORDER for
   MAIN (T4 confirmation row; distortion is held by construction, so it is a confirmation, not a
   search). 137,986 < archive < 180,002 B with identity → a 24th-pointer-move CANDIDATE: typed
   READY-FOR-T4 with the exact archive sha (MAIN decides the Modal buy). Otherwise REFUSED with the
   shortfall.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY; `submissions/semantic_joint_ctxmix/` READ-ONLY (live PR #140 tree) — copy
  the cpr1 receiver into this arm's tree for the MC-aware decode.
- NO scorer runs (nothing here needs one: the field is bit-identical), NO Modal, NO Metal from the
  arm; training runs on CPU torch (MPS is a gradient-only option; integer HPAC training is small).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD under `/Volumes/APDataStore/pact/ddm_mc1_motion_compensated_previous_plane/`
  (refuse if AP free < 1.5 GiB; Vertigo overflow allowed). DETACHED >30-MIN COMPUTE ONLY via
  `.venv/bin/python tools/launch_detached_process.py --output-dir <run_dir> --done-receipt <name> -- <cmd...>`.
- CLOSED-FORM-FIRST: step 1 is exact conditional-codelength arithmetic on the retained field; the
  retrain (step 2) owes its one-line reason (a context mixer is a trained object by construction).
- Rule 118: the motion estimator is a GENERIC deterministic algorithm (free); model bytes are counted.
- Decode budget: the MC estimate must be cheap (≤ a few ms/pair); report the CPU decode delta.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_xi1_carried_xi_inter_race_20260729.md` — warp context on a count-based coder: +12,262 B by
  16× context dilution. This charter feeds a LEARNED mixer, not a bucket table.
- `ddm_d3b_lossless_lane_factorization_20260826.md` — from-zero online mixers with temporal
  contexts: 358 KB (dc1 mechanism). This charter warm-starts the TRAINED weights.
- `ddm_dds1_decoder_derivable_verdict_20260901.md` + its ceiling re-adjudication — co-located
  previous-state buckets carry 0.9–4.1% of the wrong-half gain; the ceiling discipline here is theirs.
- `ddm_mi1_indicator_model_axis_20260824.md` — the model axis on existing features is drained (≤211 B); this is a NEW feature plane.
- `ddm_dc1_decode_budget_conditional_coding_20260816.md` — the 21-tap oracle floor; and its stated
  mechanism (learned receptive field) is the reason a new INPUT can move what buckets cannot.
- memory `box-retired-min-s-target-warp-family-closed-1273-bytes-per-error` — r2s closed WARP as a
  FLIP PREDICTOR (distortion axis); this is a RATE context with the field held exact — different object.
- `ddm_gb1_groupbin8_conditioning_*` — decode-scan conditioning members: −153 B; no retrain.

## OPTIMAL FORM

- Family exemplar: the shipped integer HPAC context mixer, reference
  `submissions/semantic_joint_ctxmix/cpr1/hpac_integer.py` (commit 7ba53d1b84) with its retrain path
  in `ddm_jf1_joint_field_model_refit_20260823.md` / `ddm_cl1_capacity_20260809` (receipts in `/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/`);
  ceiling exemplar `ddm_mi1_indicator_model_axis_20260824.md` (cross-fitted categorical family, held-out bytes).
- SCOPE reductions: the ceiling may use the n120 seeded random site sample DDS1 used (legal, random
  not prefix); the PRICED row is the full 600-pair field. MECHANISM reductions FORBIDDEN: no float
  model where the shipped is integer; no carried (byte-costing) motion — the estimator is decoder-
  derivable; no toy motion model (a global shift alone is the FIRST rung, not the verdict).
- **PRIOR-LAW PREDICTION (falsifiable):** dc1's learned-receptive-field mechanism predicts a
  motion-aligned previous plane saves ≥ 5,000 B of exact stream on the near-field moving classes
  (Lane/Movable) at ≤ +1,500 B of model. FALSIFIER: the closed-form ceiling < 5,000 B, or the exact
  RC64 row saves < 2,000 B net of model bytes — count it plainly; that closes the MC input at
  FORMULATION scope for the shipped receptive field.

## DELIVERABLE

`.omx/research/ddm_mc1_motion_compensated_previous_plane_20260903.md` — the ceiling table (per
motion model × per class: IoU, SCREEN bits, ideal-coder ceiling), the retrain receipt (seed, epochs,
model bytes, export sha), the exact RC64 rows (stream, model, archive, Δ, identity, decode delta),
the typed decision, RECALL EVIDENCE, NEXT_IF_RESUMED, LIVE-HYPOTHESES, DEAD-ENDS. Commit via the
serializer. Cite `docs/operating_manual_craft_handoff.md`. End with the own-vehicle frontier line.

## RE-SPAWN ADDENDUM (MAIN, 2026-09-04 23:20Z) — routed to a FABLE arm (operator: "use fable for the most crucial frontier score lowering work")
The 2026-09-03 codex spawn was stranded by the codex quota (row `live`, no process, no result; quota returns Sep 7 06:39). Nothing above
changes except the pointer numbers: the frontier is now **fs2 — S 0.14784474152757654 @ 180,023 B** (archive sha a8f3a379…0427bb6; the
fs1 → fs2 moves were pose-only, the token stream and model bytes are UNCHANGED from afr1: stream 113,411 B + model 13,515 B). Demand at
held distortion: **−41,817.8 B → archive ≤ 138,205.2 B**; exchange 6.658589531221714e-7 S/B. Decision rule §4: read "24th-pointer-move
CANDIDATE" as "26th". The identity base for the receiver copy is the fs2 fire tree
`/Volumes/VertigoDataTier/pact/ddm_fs2_carrier_resolve/fire_runtime_D_alternation/` (cpr1 receiver identical to the PR tree's). Retained
exact AFR1 field: locate via the jbp1 receipt as written above; the field is identical for fs1/fs2 (pose-only edits never touch it).
Everything else — ceiling first, warm-start retrain, RC64 exact price, payloads retained, no Modal/Metal/scorer — binds verbatim.

## ERRATUM (mc1 result, 2026-09-05 01:10Z)
The RE-SPAWN ADDENDUM's premise "stranded, no process, no result" was WRONG: the 09-03 codex spawn's screen ran (three planes, DF1 rows, best −17.1 B) with memo and receipts; only the queue row was stale. The Fable re-run measured the full ceiling (best derivable plane +159.60 B held-out vs the 5,000 B bar) → CEILING-REFUSED; the 09-03 screen stands as an independent replication (`experiments/ddm_mc1_motion_plane_ceiling_screen_20260903.py`). Memo: `ddm_mc1_motion_compensated_previous_plane_20260904.md`.
