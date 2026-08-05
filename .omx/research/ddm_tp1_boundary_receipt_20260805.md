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

## JD1 V2 ENDPOINT RECORD (2026-08-05 ~18:40Z, task #958 — the gc19 predicate consumption)

**Exit:** ep1354 (20 epochs, 1793s), rc=0, `stop_reason=a1_realization_gap_refuse` (2 consecutive
A1 alarms ep1349/1354; typed note "fd2 inherited gap signature — REROUTE, never scale"). NOT the
wall cap. Receipt: `tr1_window_receipt.json`; window ckpts entry_ep1336 / intra_ep1344 (COUPLED_
DESCENT interior min) / final_ep1354 all preserved.

**Endpoint discriminator (experiments/ddm_jd1_endpoint_verdict.py, 36 gd1-designed gate pairs,
exact jd1 loss physics; + live-vs-EMA falsifier probes n6 pose / n36 seg). Positive control
PASSED: probe EMA seg 0.0039072/0.0041096 vs gate 0.0038478/0.0041397 (within #855 basis drift).**

| basis | d_seg entry→final | d_pose entry→final | pose term √(10·d) |
|---|---|---|---|
| LIVE | 0.0035696 → 0.0059919 (+0.242 S_seg) | ~146 → 0.278016 (n6) | 38.3 → 1.67 |
| EMA (shipping) | 0.0039248 → 0.0041096 (+0.018 S_seg) | 146.479 → 76.603 (n36) | 38.27 → 27.68 |

**Predicate: P2 (pose-alive, seg-harmed) — with TWO measured controller defects:**
1. **Seg-hold guards the wrong SPACE.** Seg-only loss 0.4737 stayed BELOW the calibrated floor
   0.4771088596 the entire window (hinge never fired, correctly by its own definition) while LIVE
   realized d_seg rose 68%. Flips live at near-zero margin where the smooth surrogate barely
   moves. Cure = REALIZED-quantity hold: consume the a1 gate's realized_gate_dseg_mean (already
   computed every 5 epochs) with rollback-to-prev-gate-checkpoint + w_pose retreat on rise.
   (#888 sharpened a third time: not floor, not weight — SPACE.)
2. **EMA decay mis-scoped for a finishing stage.** decay 0.99996 (horizon ~25,000 steps, the
   parent-chain warm-start law U=resume_epoch×parent_steps) vs a 1,500-step window → the
   shipping basis absorbed ~none of the pose descent (live 0.278 vs EMA 76.6 = 308×) AND
   muffled the live seg damage 13× (gate saw +0.018 S of a live +0.242 S). Stage-scoped law:
   the SAME derivation with U = WINDOW steps (130ep×75 = 9,750 → decay 0.99959), EMA
   RE-ANCHORED at live weights at window entry.

**What the window PROVED (the #366 thesis, first realized numbers):** in-loop joint pose descent
on the live TR1 vehicle is FAST and REAL — live training-objective d_pose ~146 → 0.278 in 19
epochs, monotone (the operator's pose-descends-quickly law measured in-vehicle). What it COSTS
uncontrolled: live seg +0.242 S. What blocked promotion: both gains and damage were invisible
to the shipping basis under the mis-scoped EMA.

**Bonus finding (chain-wide):** live entry seg 0.0035696 BEATS the chain's EMA-reported minimum
0.0038332 — the whole w4m chain's telemetry (EMA basis) has understated its own live seg quality;
any future byte-close basis decision must measure BOTH bases.

**Confound GENUS (third instance in ONE window, one class):** cross-regime constant/control
transfer — (i) floor latched at margin-ON scale consumed margin-OFF (3.33×, caught pre-launch);
(ii) loss-space hold consumed against realized-space erosion; (iii) parent-chain EMA horizon
consumed in a 20-epoch finishing window. One genus: a constant/control derived in regime A,
consumed in regime B, inside the same launch. Every finishing-stage config element must be
re-derived AT THE WINDOW'S OWN SCOPE (floor: own-form calibration epoch — done; hold: own-space
realized quantity — owed; EMA: own-length horizon + re-anchor — owed).

**Routing (A1 note "REROUTE, never scale"):** jd1 v3 build = realized-quantity seg-hold +
stage-scoped re-anchored EMA + ticket regeneration + re-smoke; resume candidates adjudicated at
build time on BOTH bases (entry live-seg-best 0.00357 vs final live-pose-best 0.278/seg 0.00599 —
the od2/SL2 conditioned-base laws inform which). Byte-close does NOT fire on any v2 checkpoint
(EMA pose 76.6–146 → pose term 27.7–38.3, not shippable). Receipts:
`/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/jd1_endpoint_verdict_3ckpt.json` + probe log.

## JD3 V3 TWO-SMOKE ADJUDICATION + FULL-WINDOW FIRE (2026-08-05, MAIN)

Both v3 bounded smokes ran clean on MAIN's Metal host (8 epochs each, 0 A1 alarms, 0 rollbacks,
realized floors latched w/ derived margins, window-scoped EMA decay 0.996667 DERIVED at runtime).
Pose measured by scratch variants of `experiments/ddm_jd1_endpoint_verdict.py` (CKPTS dict +
REPO path edits only; physics untouched); receipts
`/Volumes/VertigoDataTier/pact/ddm_jd3_20260805/jd3_smoke_{entry,final}_pose_verdict.json`.
All numbers n36 gd1 hard-gate set, shipping (re-anchored window-EMA) basis,
[macOS-CPU/MLX frozen-scorer advisory], score_claim=false.

| candidate | seg start→end | pose start→end | pose_term end | joint partial-S end |
|---|---|---|---|---|
| entry_ep1336 | 0.00357→0.00732→**0.00682** (3-ep toll, then floor-held + recovering) | 151.8→**0.101** | 1.0058 | **1.688** |
| refuse_final_ep1354 | 0.00599→**0.00557** (falling) | 0.180→**0.158** | 1.2558 | 1.813 |

**WINNER: entry_ep1336.** Lower joint partial-S AND steeper endpoint slopes on BOTH axes. The
measured mechanism: pose⊗seg antagonism is a TRANSIENT OF THE UNCONDITIONED STATE (a one-time
3-epoch toll the realized hold bounds), not structural — once pose conditions, the axes descend
together (the od2 addendum-6 law, now measured in-vehicle from both directions). The fresh entry
start re-pays a bounded toll and buys the STEEP phase of pose descent (151.8→0.101 in 8 ep vs the
final candidate's decelerating 0.180→0.158).

Controller-design note (watch item, cx1 FG2 refinement): the realized floor latches at the FIRST
post-engagement gate, so pre-gate transient damage (ep1337-1339) is grandfathered into the floor.
Accepted as designed (a floor-from-resume would strangle engagement via immediate rollback); the
full window watches whether recovery continues below the grandfathered level (od2's below-baseline
recovery law predicts yes).

cx1 fire-gates at FIRE time: FG1 lane-guard ratchet — NOT TRIGGERED (Lane fell on both bases in
both smokes); FG2 realized-hold rows — SATISFIED both smokes; FG3 rollback — n/a (0 rollbacks);
FG4 deterministic-R — RECORDED DECISION: training on Metal accepted (nondeterministic GPU
accumulation, standing L70 practice) with gate renders on the deterministic mlx_cpu_fp32 stream
and byte-close/authority always CPU; FG5 v4 riders (SL2 distill · PE3 conditioning · en1
margin-weight) — held OUT of v3, queued for v4.

**FULL WINDOW FIRED** under the standing GO: ticket
`/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/jd3_ticket_v3_full_entry_cont.json`
(hash 96f7f8ff148c9994…), resume = the entry SMOKE's endpoint ckpt (never-weaker-state m40: pose
0.101 banked; snapshot custody `smoke_snapshot_ep1344/`, resume-source sha 1721cab65dd1093e…),
epochs 1345→1406 (60-epoch outer bound; typed event exits inside), wall cap 168 min, launcher dir
`full_v3_entry_cont_mainlaunch_r2` (first attempt rc=126: ticket argv lacks the interpreter —
launcher execs argv[0] directly; relaunched with `.venv/bin/python` prepended; regenerator debt:
emit the interpreter in argv).

### RECURSIVE ADVERSARIAL REVIEW — ROUND 1 (MAIN, 2026-08-05, operator-directed)

Findings (counter 0/3; fresh-eyes codex arm owns rounds 2+):
- **F1 (MEDIUM, honesty wording + instrument gap).** The committed v2 3-ckpt endpoint receipt is
  EMA-BASIS ONLY (correctly labeled `ema_basis: true`); v2 LIVE values exist only at entry/final.
  Therefore the matched-epoch v2@1344-vs-v3@1344 controller A/B is NOT derivable from existing
  data. Since v3's controllers never intervened (0 rollbacks, floor never breached), the entry
  smoke is in effect a Metal-nondeterministic RESEED of v2's epochs 1337-1344; its toll peak
  (live 0.0076 @1339) attributes to reseed noise, NOT to the controllers. CORRECTION to the
  adjudication wording above: the realized hold "STOOD GUARD (0 rollbacks; would have bounded
  further erosion)" — it did not measurably bound this window's toll. No decision changes: the
  entry-vs-final pick compared same-instrument/same-basis states and stands.
- **F2 (MEDIUM, controller design — watch item for gc20/v4).** The realized floor is STATIC
  (latched at first post-engagement gate), not ratcheting: the live window may regress from the
  current 0.0061 back up to floor 0.00712+margin without tripping the hold. A1-refuse still
  guards persistent ascent (2 consecutive alarms). A ratcheting floor (min-of-gates) is the
  candidate v4 refinement; risk = over-constraining beneficial transients — needs its own A/B,
  not a silent stack.
- **F3 (LOW).** Budget asymmetry in the adjudication (entry 8 joint-epochs vs final's lineage 27):
  the decision criterion (best available state + endpoint slopes) is unaffected, but forward
  projections from the two smokes are not budget-matched.
- **F4 (LOW, scope labeling).** The adjudication ran on the n36 designed gate sample (advisory,
  m96); training is n600; the endpoint decision + byte-close measure n600. Stated here so the
  winner-pick is never later cited as an n600 result.
Assumption-challenge axis: the operating assumption is "the n36 designed sample ranks candidates
whose differences concentrate in the hard tail" — internally valid for the pick (same set both
sides); the n600 endpoint read is the check. Measured-runnability axis: satisfied (both smokes +
the live window EXECUTED at the real config; gates emitting; peak memory unremarkable).

### RECURSIVE REVIEW — ROUND 2 (ddm_rr1 fresh-eyes, counter 0/3) + MAIN dispositions
rr1 verified all Round-1 findings (F2 QUANTIFIED: static floor + margin = 0.0073396; at ep1359
the chain could regress 0.0017823 d_seg = 0.178 S_seg on the gate basis before the hold trips —
A1 covers interval-trend only, not best-so-far) and independently reproduced the adjudication
arithmetic incl. a rate-sensitivity check (payloads NOT identical — 1,193 B telemetry delta =
0.000794 S, negligible vs the 0.125 S gap; wording corrected from "rate cancels" to "rate delta
negligible for this pick"). New findings + dispositions:
- RR1-F1 (MED): dy1's inertness control has NO production caller — declared scope laws can launch
  without fail-closed resolution checking. DISPOSITION: MERGE-BLOCKING condition added to the dy1
  merge (#961) — wire ticket scope_laws vs receipt resolved_scope_laws refusal at selection
  BEFORE or WITH the merge.
- RR1-F2 (MED): Round 1's "peak memory unremarkable" was UNVERIFIED (no peak RSS/VRAM recorded
  anywhere in the smoke/window manifests). CORRECTED to: "runnability EXECUTED; peak memory NOT
  RECORDED." Debt: launcher/receipt should emit ru_maxrss at next launcher touch.
- RR1-F3 (LOW): entry pose-verdict JSON status "running" → corrected in place (SSD metadata).
- RR1-F5 (LOW): scratch probe variants copied to the SSD receipt dir (config-only-edit claim now
  source-custodied).
Counter remains 0/3; round 3 = fresh pass after these fixes.

### MAGNITUDE-DISMISSAL COMPLIANCE NOTE (Stop-hook, 2026-08-05) — the RR1-F4 "negligible" quantified
The dismissed quantity is the candidate-selection rate delta 0.000794 S (25×1,193 B / 37,545,489).
Both relative numbers, stated explicitly: (a) vs the DECISION gap it could have flipped:
0.000794 / 0.125096 = **0.63%** — 157× too small to change the entry-vs-final pick even at full
adverse sign; (b) vs the REMAINING goal gap at the current operating point:
0.000794 / 0.5818394 = **0.136%** of the live gap to the PR130 bar. verdict_scope: INSTANCE —
this dismissal applies ONLY to the candidate-selection decision. NOTHING IS ORPHANED: the byte
axis is not a banked lever being dropped — checkpoint payload deltas are transient training
state, and the rate term re-enters IN FULL at byte-close (n600, real coder, counted archive
bytes), where 1,193 B ≡ 0.000794 S would be a real waterfill row if it persisted to export.
Same scoping applies to rr1's RR1-F4 row (the arm's phrasing "far below the partial-S gap" =
the 0.63% ratio above). The flagged RR1-F2 row is an evidence-absence correction, not a
magnitude dismissal — no ΔS was dismissed there.

### RECURSIVE REVIEW — ROUND 3 (rr1, counter 0/3) + MAIN dispositions
Round-2 dispositions RE-VERIFIED real (F5 diffs confirm probe physics unchanged; F1 correctly
gated-not-fixed; chain-sweep caveat holds). Two NEW findings:
- RR1-R3-F1 (MED): the full window continues INTO the entry-smoke dir → top-level telemetry is
  MIXED-SCOPE (smoke rows + full-window rows through ep1369+). DISPOSITION: SCOPE_MARKER.md
  written into the dir; the endpoint harvest MUST read smoke evidence from smoke_snapshot_ep1344/
  only; unique-out-dir for future continuations joins the regenerator debt (with the
  argv-interpreter fix).
- RR1-R3-F2 (LOW): smoke_start_* verdict tags are SOURCE epochs; the re-anchored saved ckpts are
  +1 (meta::epoch 1337/1356). Gate-to-gate slope arithmetic (1339→1344, 1359→1363) is unaffected;
  future tables split source_checkpoint_epoch from saved_reanchor_checkpoint_epoch.
Counter 0/3; round 4 = next fresh pass.

### RECURSIVE REVIEW — ROUND 4 (rr1, counter 0/3) + endpoint-blocking disposition
Round 4 found a new CRITICAL control-transfer issue in the fired full-v3 continuation, not in the
two-smoke adjudication. The two-smoke entry-vs-final pick still stands on `smoke_snapshot_ep1344/`
and the preserved pose verdicts. The full continuation, however, cannot be consumed as a
full-window stage-scoped-EMA v3 result as fired:

- **RR1-R4-F1 (CRITICAL): full continuation inherited the smoke EMA law.** The full ticket/manifest
  runs to `--epochs 1406` from the entry-smoke final checkpoint; at the continuation resume event
  (epoch 1346) telemetry reports `stage_ema_reanchored=true`, `active_ema_decay=0.9966666667`,
  provenance `U=1200`, and there is no second `jd1_stage_ema_reanchor` event. A 60-epoch
  continuation at 150 steps/epoch would re-derive `U=9000`, `active_ema_decay=0.9995555556`,
  warmup 4500 updates. The top-level `tr1_window_receipt.json` still records the smoke reanchor
  epoch 1337 and warmup 601 updates.
- **DISPOSITION:** do NOT byte-close, promote, or endpoint-route the full continuation as a
  full-window EMA-cured v3 result. If the owner harvests it, label it explicitly as a
  `smoke-EMA continuation` and probe live/EMA endpoint bases; otherwise regenerate/restart in a
  unique out-dir or force a fresh full-window reanchor at continuation resume. DY1/#961 must wire a
  production inertness/refuse check comparing declared full-window EMA scope to resolved active
  EMA rows before checkpoint selection.

Counter remains 0/3; round 5 = next fresh pass.

### RECURSIVE REVIEW — ROUND 4 (rr1, counter 0/3): RR1-R4-F1 CRITICAL + MAIN disposition
FINDING (verified from telemetry): the full-v3 continuation carries the SMOKE's EMA decay
0.9966666667 (U=1200 provenance, re-anchor event ep1337 only) into the U=9000 full window
(correct derived value 0.9995555556) — the resume carried `stage_ema_reanchored=true`, which
SUPPRESSED the boundary re-derivation. Cross-regime constant-transfer genus INSTANCE #4,
via a NEW mechanism (state-flag latch, not value latch); memory file updated same-turn.
DISPOSITION (binding on the endpoint boundary):
1. The window RUNS ON (dirs sacred; live basis unaffected by construction; killing discards
   banked descent — never-weaker-state). No mid-run mutation.
2. RELABEL: this run is a `smoke-EMA continuation` (fast EMA, τ≈2 ep) — NEVER cite it as
   "full-window stage-scoped EMA" evidence. The EMA-basis endpoint number is a valid fast-EMA
   measurement, honestly labeled.
3. Endpoint adjudication measures BOTH bases via the n600 endpoint probes on ACTUAL checkpoint
   weights (unaffected by the label) — the byte-close candidate is whichever measured basis wins.
4. dy1 merge condition EXTENDED (#961): scope-law resolution keys on the window's OWN geometry
   (inputs hash), never a boolean done-flag; plus rr1's production check — declared EMA scope vs
   resolved active-EMA row compared before checkpoint selection, refuse on mismatch.
5. Regenerator debt EXTENDED: continuation tickets that change window geometry MUST force a fresh
   re-anchor (clear/ignore the carried flag). Joins the argv-interpreter + unique-out-dir items.
Counter 0/3. The genus audit lesson: value-audits (cx1) cannot catch flag-latches — only
telemetry-reads at the resume row can. Round 5 = next fresh pass.

### RECURSIVE REVIEW — ROUND 5 (rr1): FIRST CLEAN PASS, counter 1/3
Zero new findings. Substantive verification (not absence-of-effort): round-4 disposition boundary
re-verified (hot state + tp1 + scope marker all label the continuation smoke-EMA; no uncaveated
full-window-EMA consumption found in current decision surfaces) + the broader resume-carried
state-flag CLASS swept (no second instance found in the searched scope). Honest boundary: dy1 +
regenerator production debts are GATED, not fixed — round 5 correctly refuses to count them
until code/launch gates enforce. Rounds 6-7 = the remaining consecutive clean passes to SEAL the
pre-endpoint state.
