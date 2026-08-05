# TP1 w4/w4m ep1363 boundary receipt + jd1 #366 fire record — 2026-08-05 ~17:05Z

Axis: [macOS-MLX research-signal] n36 EMA-shadow gate metric throughout — NON-PROMOTABLE, score_claim=false.
Adjudicator: `tac.optimization.trajectory_stopping.adjudicate_tail_slope` on both post-resume tails (26 gate rows each).
Full receipt: `/Volumes/VertigoDataTier/pact/ddm_tp1_20260805/w4_boundary_adjudication.json`.

## Adjudication (single variable: margin_weighted_loss, en1 #925; identical resume ep1223 / epochs 1363)

| arm | endpoint d_seg | min d_seg @ep | verdict | advisory slope |
|---|---|---|---|---|
| w4 (margin OFF) | 0.0039166 | 0.0038800 @1264 | censored_still_descending | −0.0191 S/hr |
| w4m (margin ON) | 0.0039679 | **0.0038332 @1334** | ascending_past_min | +0.0285 S/hr |

- Parent chain-min was 0.0038516 (w3 endpoint). **w4m's interior min 0.0038332 BEAT it** — seg is NOT exhausted
  (confirms the operator seg-uncapped steer); w4 OFF never beat chain-min inside this window.
- winner_by_endpoint = w4_margin_off (Δ +5.13e-5 ON-minus-OFF). Arms CROSSED twice during the window.
- **Margin A/B verdict: AMBIGUOUS** (ON wins interior-min, loses endpoint, ascends past its min). Single-variable
  discipline → the sealed jd1 loss config is KEPT unchanged; margin-weight re-A/B remains a future clean-window item.

## Routing (reasoned deviation from the pre-registered endpoint-winner rule)

Pre-registration said "jd1 regenerates vs the WINNER endpoint checkpoint." It did not anticipate crossing arms
with an interior min BELOW chain-min. Per m40 (never-weaker-state) + the operator steer
([[seg-uncapped-pose-proper-time-order-20260805]]), jd1 chains from the DEEPEST measured state:
**w4m `checkpoints/intra_seg_trunk_tau_ep01334.npz`** (d_seg 0.0038332), not the shallower OFF endpoint 0.0039166.

## RECURSIVE OPTIMIZATION PASS (operator 08-05 "ensure it is as optimal and informed as possible" + "recursive optimization pass") — CAUGHT + CURED a floor mis-calibration

The pass verified the fired config against LIVE run data (not just config text) and found a MEASURED defect:
the sealed `checkpoint_tail_ep_loss` floor source latched **1.5954** from w4m's tail — but w4m trained under
**margin-ON** loss while the jd1 child measures its seg component under **margin-OFF**. At MATCHED d_seg
(~0.0039) the parents' tail ep_loss reads 1.598 (ON) vs 0.480 (OFF): the margin weighting inflates the loss
SCALE 3.33× without changing quality. The cross-form floor was therefore ~3.3× TOO LOOSE — pose descent
could triple seg before the hinge resisted (the exact protection the #956 Fisher-anti-alignment finding
says is load-bearing). Verified at source first: the hinge itself is CORRECT (trainer `batch_loss` compares
a `pose_active=False` seg-only pass against the floor — no joint-vs-seg mismatch); only the floor's SCALE
was wrong, a cross-arm chaining side effect the seal (pre-dating the margin A/B) could not have anticipated.

**Cure (no invented constants):** killed the v1 run at ep1337 (~6 epochs, nothing of value lost; aborted
dir preserved w/ ABORT_REASON.txt), regenerated ticket v2 (hash 4a50a74bf4b2e076) with two flags changed
from sealed: `--jd1-pose-finish-start-epoch 1336` (ONE seg-only calibration epoch under the child's OWN
loss form) + `--jd1-seg-hold-floor-source last_pre_pose_epoch_loss` (floor latches from that measured
epoch). Loss form itself UNCHANGED (margin stays OFF per the AMBIGUOUS A/B — single-variable discipline).
Relaunched pid 51515.

**Cure VERIFIED live:** calibration epoch 1335 seg-only ep_loss **0.477109** (within 0.6% of the 0.480
cross-arm prediction, 3.34× below the wrong floor); engagement at ep1336 latched
`seg_hold_floor = 0.47710885961850485` via `last_pre_pose_epoch_loss`. The hinge now resists any seg
rise above the resume state's own measured loss.

## jd1 #366 fire record

- Regenerated ticket `ddm_tp1_20260805/jd1_ticket_regenerated_from_full_birth_lane_on_w4m.json`
  (v1 hash d6ca693c… superseded by **v2 hash 4a50a74bf4b2e076** after the recursive pass), regenerator
  self-check REPRODUCED the sealed EMA decay before applying the law.
- Geometry: resume ep1334 → epochs 1464 (130-ep window) · EMA decay 0.999960019990005 = 1−4/(1334×75)
  (parent-chain warm-start law, LawRef ema_decay_run_geometry_v1) · batch_pairs 4 · --max-wall-minutes 130
  (windowed continuation w/ typed recursion, not a censoring cap) · ONE seg-only calibration epoch (1335)
  → pose engages ep1336 at the #383 `post_knee` predicate ∧ start-epoch gate · seg-hold weight 0.25,
  floor self-calibrated from the child's own ep1335 seg-only loss.
- v1 launch (pid 43256, mis-calibrated floor) killed at ep1337 by the recursive pass; **v2 LIVE: pid 51515**,
  out-dir `/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/tr1_joint_pose_finish_from_full_birth_lane_on_w4m`,
  done-receipt `jd1_pose_finish_window.done` (carries child rc per the bl1/#937 cure), monitor bq3sp7ruy armed.
- Wall-cap arithmetic (informed): v1 measured ~111.6s/epoch with pose+hinge double-pass → the 130-min cap
  covers ~70 of the 130 epochs; the trainer stops with a typed `max_wall_minutes` reason + full checkpoint,
  and the recursion policy (pre-committed here) CONTINUES the window from the checkpoint per the ticket's
  recursive_encode_pass_loop while the tail is descending on either axis — a windowed continuation, never
  a censoring cap (the operator seg-uncapped steer + ca1 cap-receipt discipline).

## #956 dual-metric probe (fired FIRST on the same ep1334 state, 174s, n8 strided)

Receipt: `/Volumes/VertigoDataTier/pact/ddm_tp1_20260805/p956_class_grad_disagreement.json`.
Trunk per-class gradient cosines, off-diag: plain mean **+0.0904** vs Fisher-whitened mean **−0.0608**
(min −0.3055) → `HIGH_DISAGREEMENT_split_premise_supported` keyed on the WHITENED metric (m65 dual-metric law;
the operator's "plain cosine never optimal" correction was material — the two metrics disagree in SIGN on the
mean). Reading: in scorer geometry the per-class trunk gradients are anti-aligned on average → supports the
per-class split premise (m91 one-graph-one-hub decomposition) and validates the jd1 seg-hold floor during
in-loop pose engagement. Advisory only.
