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

### RECURSIVE REVIEW — ROUND 6 (rr1, counter resets 0/3): RR1-R6-F1 MEDIUM + MAIN disposition
FINDING (verified from the fired ticket JSON): `recursive_encode_pass_loop.continue_policy.
next_resume_from_template` = the ANCESTOR JD1 lane's path (ddm_jd1_20260805/
tr1_joint_pose_finish_after_tp1_lane_on/...) — stale metadata inherited from JD1_TICKET.json by
the regenerator; the actual fired continuation runs in ddm_jd3_20260805/tr1_jd3_v3_smoke_entry_
ep1336. Orphan-prone follow-on metadata, NOT a current-run physics bug.
DISPOSITION:
1. The fired ticket file is NOT mutated (hash custody). The template is declared VOID by this
   consumption contract: ANY continuation from the full-v3 endpoint derives its resume source
   from the ACTUAL endpoint checkpoint in the ACTUAL out-dir, adjudicated by MAIN at the
   boundary — never from the ticket's recursive-loop template. (No automatic consumer exists;
   the jd-line already regenerates tickets per-winner at each boundary.)
2. Regenerator debt item #4: refuse emission when next_resume_from_template does not resolve
   under the ticket's child_out_dir or a declared new continuation out-dir (rr1's check).
   Joins: argv interpreter · unique continuation out-dirs · force-reanchor-on-geometry-change.
Pattern note: the SAME genus shape again — inherited-without-re-derivation metadata surviving a
lane change (values → flags → now PATH TEMPLATES). The regenerator is the common factory; its
debt list is now the single cure surface. Counter 0/3; round 7 next.

### RECURSIVE REVIEW — ROUND 7 (rr1, counter 0/3) + CYCLE PAUSE AT THE ENDPOINT BOUNDARY
RR1-R7-F1 (MED, verified): the regenerator's v3 branch updates argv/child paths but NEVER rebuilds
`levers[*].overrides` — the fired ticket's value ledger claims epochs 1076/wall 130/stale
ema-decay/stale floor-source while argv says 1406/168/... . The RUN is valid (argv is consumed);
the machine-readable provenance ledger is not value-custodied. INHERITANCE GENUS SHAPE #4
(values → flags → path templates → value LEDGERS). DISPOSITION: regenerator debt #5 = rebuild
levers[*].overrides from final argv (or demote the ledger non-authoritative) + refuse on
declared-vs-argv mismatch across the WHOLE emitted surface. All 5 debts land together in the
continuation-prep build (jd4) — the regenerator is ONE factory, cured once.
CYCLE PAUSE (honest): counter 0/3 after 7 rounds / 15 findings, all dispositioned. The state
under review is superseded THIS HOUR by the window endpoint + continuation; rounds 8+ resume
over the NEW state (endpoint receipt + fixed regenerator + continuation fire) rather than
burning passes on a state about to change. Pause recorded, not a seal claim.

### FULL V3 WINDOW ENDPOINT (2026-08-05, stop_reason=epochs_complete, rc=0, 3316s)
60 epochs ep1345→1405 completed inside wall cap. Final gate ep1405 (n36 designed, EMA basis):
seg 0.004801 — DESCENDING AT THE ENDPOINT (interval delta −4.28e-5/epoch, COUPLED_DESCENT), no
rollbacks/retreats all window, counted bytes 301,761 (down from 306,891). Loss-terms row shows
seg term 0.4371 (w100). Endpoint custody snapshot: full_v3_endpoint_ep1405_snapshot/ (final ckpt
sha recorded there + receipt + ep1404 intra). NEXT (operator steer "will need more descent"):
n600 both-bases endpoint probe → continuation window from the endpoint with the FIXED regenerator
(all 5 debts) + correct U-derived EMA re-anchor. No byte-close before the probe.

### PLATEAU POLICY for the jd4 continuation (operator-directed 2026-08-05, 5W-typed; binds
### MAIN's boundary adjudications + the continuation's typed event exits)
STORES CONSULTED: PR95 L14 stage decomposition + demotion banner (CLAUDE.md) · #164 Muon-jump
verdict · #302/#686 derived event schedule · #341/#342/#850/#897/sq2-cw1 solve inventory ·
od2 addendum-4/6 ordering laws · #269/#270 Muon warm-start · #475 plateau semantics · #888/#925/
en1 hinge+margin · #775 QA43 · #946 pose carriage · #827/#934 banked compositions · jd3 window
telemetry (this receipt).

CASE 0 — ALWAYS FIRST: plateau DISAMBIGUATION (the #205 lesson: 4 rediagnoses, every wall an
artifact). Checks in order: (a) liveness (accepted_frac, weights_stepped — L3 confound); (b)
BASIS — ⚠ NEW TRAP: the continuation's CORRECT U=18000 EMA (decay 0.999778, warmup 9000 steps
= 60 ep) is IMMATURE for half the window — a slow EMA manufactures phantom plateaus early and
masks late ones. GATES READ LIVE (windowed mean vs the ±0.0002 single-ckpt noise floor); EMA =
shipping basis only. The inverse of the round-4 defect — fixed-EMA must not corrupt detection.
(c) slope vs noise floor over ≥3 gates; (d) PER-CLASS decomposition (pc2: seg is ONE GRAPH,
Road hub 87.8% of flips — a Lane-only plateau is a different disease than a Road plateau).

CASE A — POSE plateaus, seg descending. WHAT: exit in-loop pose → TERMINAL POSE SOLVE (eg1 E3
per-pair 6-eq GN, UNCAPPED per #850) + QA43 tail-targeted per-pair correction (#775: top-112
projected pose 1.263→0.382) + od-line cheap carriage (40,444 B projected, #946). WHERE: frozen
trunk, per-pair DOF. WHEN: pose slope < noise for 3 gates, OR pose reaches the satisficing band
(strict gate d_pose ≤ 0.00144 → contribution ≤0.12 exactly [RR1-C2-R4: √(10·0.0015)=0.12247,
so "1.5e-3" is the rounded neighborhood, ~0.1225, NOT the ≤0.12 gate] ≈ frontier bank — stop
descending, satisficing hinge
#360; zero is not the target, the tube is). WHY: pose is a 6-scalar terminal quantity; training
it through the shared trunk pays antagonism tax a post-freeze solve does not. HOW: freeze → GN
to convergence (no relin cap) → price via per-surface coder race → compose.

CASE B — SEG plateaus, pose descending. WHAT, in fire order: (1) UNCAPPED sq2-class GN/CG seg
solve from the live state (the measured seg lead: η 0.862, −0.117 S n32; #935 uncap); (2) MUON
FINISHER stage flip (the ONE measured d_seg drop in the PR95 decomposition — conditioning, not
capacity; warm-start COLD momentum + FLAT LR anneal #269/#270; optimizer swap at a stage
boundary ONLY per SPEC_v75 §8C); (3) en1 margin-weight engagement (single-variable, consumer
built #925) / hinge weight (#888). WHERE: trunk (Muon) vs per-pair (solve) — solve first, it is
$0-class and does not perturb the pose descent. WHEN: seg gate windowed-live slope < noise ×3,
AND the realized hold has not tripped (a hold trip = regression, different branch: rollback).
WHY: at gate ~0.0048 vs frontier 0.0043 we are IN the terminal band where the solve inventory
measured dominance over descent. HOW: keep pose in-loop; interleave solve at the boundary;
recursion-from-solved-states (#954 doctrine).

CASE C — BOTH plateau. This IS the E2 train→solve handoff (eg1). Exit training → terminal stack:
uncapped seg solve → pose 6-eq solve → TTO/MC-finisher polish (#396/#400) → byte-close (drop-knee
re-derivation + per-surface coder race #940) → COMPOSE the banked levers that were waiting for a
pose-carrying base: #827 cell_drop50 seg+rate composition (−0.036 S banked) + #934 phase-field
compose (q3x-gated). WHY: at both-plateau the marginal epoch is worth less than the marginal
solve/composition — the vehicle's remaining value IS the pose-carrying base. WHEN: both slopes
< noise ×3 on live basis. Then n600 both-bases → byte-close → candidate row.

ARCHIVE-IMPACT NOTE (why stage choice moves bytes, not just distortion): rate-in-loss + entropy
model shrank counted bytes 306,891→301,761 DURING the window; quant-anneal engages at_knee;
terminal solves move distortion at ~0 byte cost; the coder is chosen per-surface at export. Every
plateau branch above states its byte consequence through this chain, never assumes rate constant.

RR1-C2-R4 arithmetic correction (2026-08-05): the Case A shorthand
`d_pose <= ~1.5e-3 -> contribution <=0.12` is approximate, not an exact inequality.
Exact arithmetic is `sqrt(10 * 0.0015) = 0.1224744871`; the hard `<=0.12` bar is
`d_pose <= 0.00144`. Use `0.00144` for a strict gate, and treat `1.5e-3` as the
rounded satisficing neighborhood (~0.1225 S pose contribution).

## RECURSIVE ADVERSARIAL REVIEW — CYCLE 2 (the jd4 landing + n600 probe variant + plateau policy), ROUND 1 (MAIN, 2026-08-05 ~15:4xZ)

Operator request (verbatim): "Might be around for recursive adversarial review right now of all
that just landed because it's mission critical." Scope: jd4 landing fae7193a85 (5 regenerator
debts + trainer force-reanchor flag + exclusive-epoch fix), the sealed jd4 ticket
(51c64222b432…), the n600 both-bases probe variant, the tp1 plateau policy. FIRE GATE: the jd4
continuation fires only after (a) the probe receipt + endpoint adjudication AND (b) this review
clears the landing.

### Round 1 checks (all axes; measured, not reasoned)

| # | Check | Result |
|---|---|---|
| 1 | `--jd1-force-ema-reanchor-on-resume` exists in argparse (:2849), fail-closed (:3315 requires --resume-from + window scope); sealed ticket argv exact (interpreter argv0, trainer argv1, flag present, epochs 1526, unique out-dir) | VERIFIED |
| 2 | Forced-resume tail source = checkpoint's OWN baked meta (`telemetry_tail`, last ≤4 gate rows, :2404-2417) — snapshot resume works WITHOUT telemetry.jsonl. Snapshot meta parse via the trainer's canonical idiom `json.loads(bytes(z['meta::json']).decode())` (:2640) | VERIFIED — tail epochs [1404,1404,1405,1405], meta::epoch=1406 EXCLUSIVE |
| 3 | Forced-start arithmetic: legacy 1407 vs forced max(tail)+1=1406 → chosen 1406; remaining 1526−1406=120 ep → U=18000 → decay 1−4/U=0.9997777777777778 == the sealed ticket's derived decay | VERIFIED (recomputed) |
| 4 | The R4 latch is LIVE in the snapshot (`jd1_pose_finish.stage_ema_reanchored=True`, carried `active_ema_decay=0.9966667` w/ U=1200 provenance string) AND the predicate `jd1_should_reanchor_stage_ema` (:3200) overrides it: forced branch fires on force-flag ∧ reason=="resume_inside_joint_pose_finish". Without the flag the continuation would run 120 ep at the smoke decay — the exact hazard, confirmed present, confirmed cured | VERIFIED at source |
| 5 | Probe basis semantics: ckpt stores 20 `ema::` arrays SEPARATE from 20 live param arrays; `load_checkpoint` loads LIVE params; `ema_snapshot_swap` applied only when `use_ema=True` → the `endpoint_ep1405_live` tag measures the true LIVE basis (warmup ≈ U/2 = 9000 steps = 60 ep; ⚠ CORRECTED post-R5 by MAIN-R5X below — the "so continuation gates MUST read LIVE" inference here was WRONG: the as-built resume⇒warm-shadow rule (:4314/:4346) pre-satisfies warmup, so continuation gates read the ep1406-REANCHORED ema_shadow from the first gate; the both-bases probe design this check verified is unaffected) | VERIFIED |
| 6 | Sealed ticket lever ledger: 23 levers, 0 lever-vs-argv mismatches (declared-vs-argv refuse = regenerator debt #5; live cross-check on the actual ticket) | VERIFIED — 0 mismatches |

Axis 8 (assumption-challenge): the shared assumption under review is "the checkpoint's baked
telemetry_tail is a faithful proxy for the run's endpoint geometry." Verified rather than
assumed: tail max epoch 1405 == the window receipt's final telemetry epoch; meta::epoch
exclusive semantics confirmed against the writer (:2415). If this assumption were wrong the
forced start would silently shift by ±1 epoch — the check above measures it instead.
Axis 9 (measured-runnability): the probe variant is EXECUTING the real config now (n600, real
gt, both bases; pid 52792, status=running, 600-pair sweep in progress) — runnability measured,
not reasoned; scored quantities land in the receipt.

### Verdict

ROUND 1 = CLEAN PASS (0 findings). Counter 1/3 for cycle 2. Reviewer-vs-author: MAIN authored
the probe variant + plateau policy, so rounds 2+ go to the fresh-eyes arm (rr1 respawn) before
the counter can advance on those artifacts. The jd4 fire remains gated on the review clearing.

## N600 BOTH-BASES ENDPOINT RECEIPT + ADJUDICATION (MAIN, 2026-08-05 ~16:0xZ)

Probe r2 COMPLETE (rc=0, 1293s, pid 52792; receipt
/Volumes/VertigoDataTier/pact/ddm_jd3_20260805/jd3_endpoint_n600_both_bases_verdict.json,
status=complete). Endpoint ckpt = the ep1405 snapshot (sha 2c3bd24455…), n600, [macOS-CPU/MLX
frozen-scorer advisory], score_claim=false. Both bases on the SAME checkpoint file:

| basis | d_seg (n600) | d_pose (n600) | √(10·d_pose) | joint partial-S (seg+pose) |
|---|---|---|---|---|
| LIVE | 0.0071503 | 0.57409 | 2.3960 | 3.1111 |
| EMA (smoke-decay U=1200 shadow) | 0.0057480 | 0.12885 | 1.1351 | 1.7099 |

**THE MEASURED FACT: the EMA basis beats LIVE on BOTH axes — seg −19.6%, pose 4.46× — worth
−1.401 joint partial-S at this endpoint.** The live weights under joint pose descent are NOISY
(per-pair pose max 7.654 live vs 6.179 EMA); the shadow's averaging, not its lag, dominates at
this operating point. Note the R4-relabeled smoke decay (τ≈2 ep) is the basis that produced
this: a SHORT-horizon EMA both tracked the 151.8→~0.1 pose collapse AND still suppressed 4.5×
of pose noise. Scope note: the n36 gd1 designed-gate set read 0.004801 (EMA) vs population
0.005748 — the designed set reads 0.835× of n600; cite per-scope, never interchangeably.
Rate context: counted bytes at window end 301,761 → rate term 0.2009; advisory composed
arithmetic (NOT byte-closed) ≈ 1.911.

### Adjudication under the PLATEAU POLICY (Case 0 first)

Neither axis is plateaued: seg gate still descending at the endpoint (−4.28e-5/ep,
COUPLED_DESCENT, 0 rollbacks) and pose needs ~89× more descent (0.1289 → 0.00144 for the strict
≤0.12 contribution gate; RR1-C2-R4: √(10·0.0015)=0.12247, so 1.5e-3 is the rounded ~0.1225
neighborhood, not the gate). No Case A/B/C fires. **VERDICT: CONTINUE — fire the sealed jd4 ticket
(51c64222…) once review cycle 2 clears.** The operator steer "will need more descent" is
confirmed by the numbers. Watch item for the jd4 window (⚠ CORRECTED post-R5, MAIN-R5X below —
the original wording here claimed "gates read LIVE throughout (policy)", which is WRONG about
the as-built mechanism): the reanchored shadow is reset EQUAL to live at ep1406 (:4342) and the
resume⇒warm-shadow rule (:4314/:4346) pre-satisfies warmup, so gates read `ema_shadow` from the
window's FIRST gate (telemetry `a1_gate gate_params="ema_shadow"` at ep1424/1429 confirms; the
parallel `jd1_live_basis_gate` channel logs the live basis at every gate boundary). Shadow
maturity at the ep1526 endpoint = 120 ep = 4τ from its ep1406 anchor (τ=30 ep). The endpoint
probe must again measure BOTH bases.

## CYCLE-2 ROUND 2 DISPOSITION + JD4 FIRE RECORD (MAIN, 2026-08-05 ~16:2xZ)

**Round 2 (fresh-eyes rr1): ONE MEDIUM (RR1-C2-R2-F1)** — the n600 both-bases receipt's embedded
`axis` field carried the copied instrument's "36 gd1-designed gate pairs" literal while the
measurement is n600 (denominator-label defect, misleading-name genus; MAIN-authored variant).
Values unaffected; the tp1 adjudication section had already labeled the numbers n600 correctly.
Dispositions, all landed: (a) arm's non-mutating correction manifest
(.omx/research/ddm_rr1_20260805/jd3_endpoint_n600_axis_correction.json; raw SSD receipt bytes
untouched per append-only); (b) CLASS FIX at source 6e4a6e24fe — the committed instrument's axis
label is now DERIVED from GATE_PAIR_IDS (literal cannot lie; a variant that swaps in range(600)
self-labels "all 600 pair ids"); future endpoint probes derive from the FIXED committed source,
never the stale SSD/scratch copies. Cycle-2 counter honestly 0/3 (arm reset); the finding was
NOT against the jd4 landing.

### FIRE-GATE ADJUDICATION (explicit, no silent weakening)

The committed gate: "fires only after (a) probe receipt + endpoint adjudication AND (b) the
review clears the LANDING." (a) satisfied (receipt complete + CONTINUE verdict above). (b): TWO
independent examinations of the jd4 landing (MAIN round 1 + fresh-eyes round 2, which read the
ticket fields, trainer/regenerator source, and probe artifacts) produced ZERO findings against
the landing itself; the sole finding was instrument-label, cured at class level. The LANDING is
adjudicated CLEAR; the cycle-2 seal (3 clean passes over ALL artifacts) continues in parallel
and does NOT gate the fire under the committed wording.

### THE FIRE (2026-08-05 ~16:25Z)

Custody verified at fire time: ticket file sha a22783a9340c…, ticket_hash 51c64222b432…,
argv_len 118, reaper cmdline sweep CLEAN. Fired via tools/launch_detached_process.py (the same
governed path as the v3 window; launch_tr1_run.py REFUSES the jd1 ticket schema — expected,
recorded): pid 90157, out-dir tr1_jd4_cont_ep1406, mainlaunch jd4_cont_mainlaunch, done-receipt
jd4_cont_ep1406, wall cap 165 min for 120 ep (1406→1526).

**LIVE VERIFICATION of the R4 cure (telemetry, first minute):**
- `jd1_force_resume_epoch_reanchor`: forced_start=1406, legacy=1407, tail_last=1405 — tail-derived
  start under the force flag, exactly the Round-1 arithmetic.
- `jd1_stage_ema_reanchor`: `active_ema_decay=0.9997777777777778`, provenance "U=18000 → 0.999778;
  ceiling 1−2/U=0.999889, no constant clamp", `ema_warmup_updates=9000` — the carried smoke decay
  (0.99667, visible in the resume row's historical sub-dict) is SUPERSEDED in-window. The
  cross-regime constant-transfer genus instance #4 is cured IN THE FIRING RUN, not just in tests.

Endpoint obligations (bind at ~ep1526 / wall-cap / typed exit): both-bases n600 endpoint probe
from the FIXED committed instrument (never the stale copies) · plateau policy Cases 0/A/B/C ·
in-window gate basis = the ep1406-reanchored ema_shadow (resume⇒warm-shadow :4346, NOT live —
corrected per MAIN-R5X; 4τ mature by ep1526; live basis logged in parallel) · dynamic-EMA A/B
(fixed 1−4/U vs plateau-anchored tail average) queued as the next-window decision per the
operator's dynamic-EMA question.

## CYCLE-2 ROUND 3 (fresh-eyes rr1, ad06cc10ef): CLEAN PASS — counter 1/3

Verified by the arm from code/bytes (not commit messages): the axis-label class fix is real
(derived-label path), JD4 ticket/launch custody matches the fire record, the FIRE-GATE
ADJUDICATION was audited for gate-bending and PASSED CLEAN, and the live R4-cure telemetry
(forced_start 1406 + decay 0.9997778) is as recorded. Round 4 respawned: rotated axis =
plateau-policy arithmetic (Case A/B/C bars, satisficing thresholds, dynamic-EMA A/B design)
+ endpoint-path mental execution (wall-cap vs ep1526 vs typed exit — which artifact does each
exit hand the adjudicator?). Seal requires 3/3; the running window is not gated on it.

## CYCLE-2 MAIN-R5X (between-rounds finding, MAIN, 2026-08-05): ONE MEDIUM — counter RESET 0/3

rr1 round 5 returned CLEAN (1/3), verifying the R4 headline correction, the warmup/τ arithmetic,
and the ep1424→1429 A1 alarm episode (transient, recovered, 0 refuses, watch-only). MAIN then
caught — from the same live telemetry the round had already read — a record-vs-build
contradiction the round missed:

**MAIN-R5X (MEDIUM, record-accuracy class, MINE):** this receipt (three sites) and the hot state
claimed "gates read LIVE through ~ep1466 (warmup)". The AS-BUILT trainer rule is
**resume⇒warm-shadow**: on resume `global_step` initializes AT `ema_warmup_updates`
(trainer :4314) and the forced reanchor re-bumps it (`global_step = max(global_step,
ema_warmup_updates)`, :4346), so `gate_basis = "ema_shadow"` from the window's FIRST gate.
Telemetry confirms: `a1_gate gate_params="ema_shadow"` at ep1424 and ep1429, with
`ema_warmup_updates=9000` present in the same rows. The rule is deliberate and documented
(:2183-2185 "resume ⇒ warm shadow") and SOUND for this window — the reanchor resets the shadow
EQUAL to live (:4342), so the basis is anchored, not stale; it lags live by at most ~τ (30 ep)
as it tracks. Physics unaffected; endpoint adjudication unaffected (the plateau gates consume
the BOTH-bases n600 endpoint probe, never the in-window a1 rows). Wrong: my "LIVE-through-
warmup" description — that warmup semantics applies to FRESH runs only. Corrected at all three
sites (Round-1 check-5 parenthetical · adjudication watch item · fire-record endpoint
obligations) + hot state; the "~2τ past warmup" maturity framing recast as 4τ-from-anchor.

Disposition per protocol: a found defect resets the counter regardless of who found it — round
5's clean is INVALIDATED on the live-A1-status axis (the arm read the a1_gate rows carrying the
contradicting basis label without flagging it). **Counter honestly RESET 0/3.** Round 6 verifies
this correction at source (:4314/:4342/:4346 + the telemetry rows) and continues the fresh sweep.

## BI1 BIRTH-SEED ON-vs-OFF MATCHED-EPOCHS READ (MAIN, 2026-08-05, #924/#954 owed adjudication)

Both arms exited `epochs_complete` at ep820 (matched by construction). Axis: [macOS-MLX
research-signal] n36 designed-gate realized d_seg, ema_shadow basis, score_claim=false.

| arm | first gate ep809 | endpoint ep820 | trajectory |
|---|---|---|---|
| birth OFF (`smoke_birth_off`) | 0.0039732 | **0.0040446** | ascending/FLAT (A1 alarm ep814) |
| birth ON + union mask (`smoke_birth_lane_on`) | 0.0046221 | 0.0045341 | COUPLED_DESCENT → stalls FLAT at ep820 (last interval +8.5e-6) |

**Verdict — RE-GRADED by the operator-directed negative audit (same day, below): NOT a
treatment verdict; a SHORT-WINDOW COST MEASUREMENT.** ⚠ The window is **14 epochs** (both arms
ep807→820, verified from telemetry epoch rows — NOT the ~140 ep the lineage context suggested).
The +4.90e-4 matched-endpoint gap therefore mostly re-measures the birth-seed injection's
UP-FRONT cost (+6.5e-4 visible at the first gate, 2 ep after the shared resume — the amplify
anchor perturbs the lattice AT resume), of which ON bought back ~0.9e-4 in the remaining 13 ep
(best rate −1.4e-5/ep). Break-even ETA at that rate ≥ ~46 ep — **3.5× the window**; the
original "ON loses / descent stalls" wording is WINDOW-CENSORED (the "stall" was ONE +8.5e-6
interval, noise-level). Honest scope: **UNDECIDED at this horizon** — the same under-power
class as bp1's 1.56τ A/B. lane_guard g_s_units near-identical (−0.0352 vs −0.0362).

**Cross-base caveat (m85 — matched-base controls):** the LIVE jd4 lineage (w4m→jd1→jd3→jd4)
carries birth_lane_on throughout — this A/B ran on a DIFFERENT (smoke-continuation) base and
does NOT transfer as a verdict on the live lineage. Routing: (a) WATCH item for the jd4
endpoint — if endpoint seg underperforms, the birth-seed cost is a named candidate mechanism;
(b) the decisive test is a matched-base A/B on the jd4-lineage endpoint, QUEUED as a
window-boundary candidate (competes with the dynamic-EMA A/B and margin-weight A/B for the
next single-variable slot — three queued single-variable A/Bs now exist; the boundary
adjudication picks by expected |ΔS| per window); (c) #924's "opposite-signs-same-cells"
hypothesis stays REFUTED (union added ONE cell); the mechanism under test here is the
birth-seed amplify anchor, not the mask.

## NEGATIVE/INCONCLUSIVE AUDIT (operator-directed, 2026-08-05): this window's fresh verdicts re-graded for naive/toy/short-horizon construction

Per the standing verdict-scope discipline (m69, #390/#630/na2 lineage) over every negative or
inconclusive result produced in the current arc:

| result | prior grade | audit finding | re-grade |
|---|---|---|---|
| bi1 birth-seed ON-vs-OFF (3401ce59f3) | "INSTANCE: ON loses endpoint +4.9e-4; descent stalls" | **CAUGHT — window-censored construction.** Window = 14 ep (ep807→820, verified), vs a ≥~46-ep break-even ETA derived from ON's own measured buy-back rate against its +6.5e-4 at-resume injection cost. "Stall" = ONE noise-level interval. Same under-power class as bp1's 1.56τ. | **UNDECIDED at this horizon — a cost measurement, not a treatment verdict.** Section amended in place (headline+body). Queued matched-base A/B design now PRE-REGISTERS: window ≥100 ep past first gate · gates every ≤5 ep throughout (bi1 had 4 gates) · falsifier = ON interior min beats OFF endpoint within the pre-derived break-even horizon. |
| w4/w4m margin A/B | AMBIGUOUS (crossing arms) | Construction sound: matched resume/epochs/seed, single variable; crossing-arms honestly reported; endpoint gap +5.13e-5 near noise. | STANDS. Re-A/B queued at a clean boundary (boundary candidate set). |
| bp1 bias-correction ON +0.0120 worse | under-powered (1.56τ) | Already flagged; ci1 independently caught the stale 1.56τ figure. | STANDS as under-powered; not citable as a family verdict. |
| jd1-v2 P2 endpoint (pose-alive, seg-harmed) | routed to 2 controller cures | Not a family negative — defects named (hold-space, basis), cures built into jd3/jd4. | STANDS. |
| A1 alarm ep1424 (jd4) | transient, watch-only | Single-episode, recovered ep1429, 0 refuses; correctly NOT escalated to a verdict. | STANDS. |
| gc20's 6 folded routes | folded w/ reopen conditions | Each fold cites a MEASURED basis (PE3 S=1.8527@432,428B byte-closed · SL2 ~3-orders carriage · etc.) + named reopen condition. The KD fold leans on a cited prior settlement — its reopen condition (weights-as-carrier mechanism change) is the correct guard. | STAND. |
| sq1 0/32 multi-start (#930) | cap-bound | Already re-scoped (25-step cap; sq2 uncapped continuation measured η 0.7895→0.8620). | STANDS. |
| pending strong negatives #894/#916/#918 | measured, reopen-conditioned | Depth-wave adjudicated on measured races/structural arguments; no construction defect surfaced today; reopen conditions named. | STAND — no reopen warranted. |

Net: ONE material catch (bi1 — my own same-day adjudication), corrected at source per the
stale-headlines law; the queued birth-seed matched-base A/B inherits the pre-registered
adequate-power design. The audit itself confirms the class pattern: the two fresh under-powered
reads this week (bp1, bi1) are BOTH short-horizon A/Bs whose window was inherited from
convenience (an existing smoke budget) rather than DERIVED from the treatment's own time
constant — window length joins decay/floor/hold on the cross-regime derive-at-scope list (m: 
cross-regime-constant-transfer genus, instance #5: A/B WINDOW LENGTH must be derived from the
treatment's measured break-even horizon, never inherited from the vehicle's smoke budget).

## JD4 ENDPOINT FULL-TELEMETRY HARVEST (2026-08-05, operator-directed "audit and harvest and interpret ALL incoming signal") — MAIN

**Census**: 235 telemetry rows, 19 event types; prior reads covered only a1_gate + jd1_live_basis_gate.
Window ep1406→1526, rc=0, final ckpt preserved. Axis for everything below:
[macOS-CPU frozen-scorer advisory, training-vehicle semantics], score_claim=false.

### H1 — THE UNATTRIBUTED RESIDUAL (pose + birth_amplify) IS A CAP-STOP, NOT A PLATEAU (caps-genus applied) ⚠ CORRECTED per RR1-C2-R7-F1: the subtraction residual is NOT pure pose — the BI1 birth-amplify addend (weight 0.05) is also un-itemized. The cap-stop/pose verdict stands ONLY via the independent n600 endpoint probe (d_pose −29.6% measured directly); the residual's slope/curvature is NOT admissible as pose-only evidence.
Residual reconstructed as `ep_loss − itemized{seg,rate,delta_sparsity}` (= pose + birth_amplify, per RR1-C2-R7-F1) (25 joined epochs):
0.51597@ep1409 → 0.37236@ep1525 (−27.8%), minimum AT the final epoch, endpoint slope
−0.0012562/ep vs window-average −0.001238/ep (**ratio 1.01 — zero flattening**; ratio = last-6-row LSQ fit vs endpoint-to-endpoint average, per the R7 re-derivation). The epoch budget
stopped pose descent, not convergence. CROSS-VALIDATED by the n600 probe ema row: d_pose
0.128853→0.090731 (−29.6%). Case-A (d_pose ≤ 0.00144) is 63× away — OUT.

### H2 — SEG IS GENUINELY EXHAUSTED (policy channel)
LIVE basis_gate: min 0.0049040@ep1504, +3.99e-6/ep ascending since (last-8). EMA channel −2.79e-6/ep
residual = shadow lag. Seg LOSS term: min 0.397267 → last 0.408251, last-6 slope +0.0015/row RISING.
Flip decomposition at the end: toward≈away (ep1524: 4143 toward vs 4489 away) — churn, no net progress.
param_delta_rms halved (0.0219→0.0116).

### H3 — FIVE realization-gap alarms, not one (corrects MAIN's pre-compaction "single transient ep1424")
A1_REALIZATION_GAP_ALARM at ep1424, 1469, 1479, 1489, 1499 (census: 12 COUPLED_DESCENT / 7 FLAT /
5 ALARM / 1 FIRST_GATE). Back half alternated FLAT↔ALARM — smooth loss improving while realized d_seg
did not follow. Stale-headline instance #7 of 08-05; corrected here at source.

### H4 — PER-CLASS: gains on the Road hub, erosion on Undriv+MyCar
(36-gate EMA sample) Road −0.000538 · Lane −0.000288 · Movable −0.000235 (minima late, g18-21) vs
Undriv +0.000151 (min at g6, then rose — continues the burn-4 Undriv-erosion watch) · MyCar +0.000089
(eroded from window start). Erosion cost ≈ 0.024 S vs pose gain ≈ 0.18 S — 7.5× favorable exchange.

### H5 — LANE TOPOLOGY: the dash long-tail is UNMOVED
Last gate: Lane betti0 GT 985 vs realized 614; **487 GT Lane components still ERASED** (Road 6,
Movable 17). This window did not touch the lane-dash erasure deficit.

### H6 — RATE in-window: tokens 298,071→291,500 (−6,571 B, SMEVR selected; zlib alternative ~514K
dominated). total_counted 301,571→295,000. Loss rate term 0.1571→0.1532, pinned at end.

### H7 — lane_guard: λ_Lane 0.0 entire window, ratchet DELIBERATELY disabled (#934 false-positive
finding); g_s_units improving −0.0107→−0.0394 (lane_s 0.115→0.086); inertness alarm firing from
~ep1469 = the alarm channel works, inertness expected-by-design while disabled.

### H8 — TELEMETRY PORT GAP (two-landing candidate, next trainer-touching landing)
telemetry_v9_port declares loss_term_keys=[seg, rate, delta_sparsity] — the POSE term (this stage's
headline) is invisible to term-level telemetry and to gate rows; the in-window pose trajectory had to
be reconstructed by subtraction. FIX: declare pose in loss_term_keys when jd1_pose_finish is active.

### H9 — ENDPOINT COMPOSED ARITHMETIC (ema basis, probe partial; live basis pending)
seg 100×0.0052305=0.52305 + pose √(10×0.090731)=0.9525 + rate 25×295,000/37,545,489=0.19643
→ **S_vehicle ≈ 1.672** vs ep1405-equivalent ≈ 1.911: window bought ≈ **−0.239 S** (advisory,
training-vehicle). NOTE: this is the TR1 joint-descent lineage (the pose-carrying-base program,
#889/#934/#827 consumer), NOT the shipping composition (qo1 0.7539807).

### H10 — ADJUDICATION SHAPE (pre-registered before live basis lands)
Seg exhausted + pose in full-rate descent at the cap ⇒ typed exit = **Case-0-ON-POSE: pose-dominant
CONTINUATION window** (regenerate jd5 ticket from ep1526; seg-hold active, floor 0.4771 vs seg term
0.408 comfortable). Expected value ≈ −0.18 S/window at current pose rate — DOMINATES all three
boundary A/B candidates on expected |ΔS|; single-variable discipline: no stacked treatment on the
continuation; dy2 tail-average anchors at a plateau, pose has none yet → A/B deferred to the next
plateau boundary. Custody (gc20 rank-3) proceeds in parallel — endpoint ckpts preserved, continuation
resumes without destroying them.

## JD4 ENDPOINT ADJUDICATION (FINAL) + JD5 FIRE RECORD — MAIN, 2026-08-05

**Probe complete** (jd4_endpoint_n600_both_bases.json, instrument 9abd0c64db, 1287s, n600 both bases):
- ema (shipped): d_seg 0.0052305 (Δ−0.000518 vs ep1405) · d_pose 0.090731 (Δ−0.038122, −29.6%) · pose_term 0.9525
- live: d_seg 0.0054997 (Δ−0.001651) · d_pose 0.091572 (Δ−0.482520) · pose_term 0.9569
- **live/ema pose CONVERGED** (0.0916 vs 0.0907; at ep1405 they were 0.574 vs 0.129) — the R4 reanchor
  cure verified working at n600. Live median 0.0396 < ema median 0.0558 (shadow smooths tail spikes).
- Case-A strict bar: NOT met (0.090731 vs 0.00144, 63×).
- Instrument note ⚠ CORRECTED per RR1-C2-R7-F3: gate-36 controls land +0.000518/+0.000339 above
  the run's final gate rows — the cause is CROSS-INSTRUMENT (trainer gates use the CPU-torch
  SegNet path; the probe uses the MLX scorer adapter, the #855 drift), NOT a one-epoch training
  gap (the ep1525 gate runs AFTER that epoch's updates; the final ckpt has no further update).
  Same-instrument deltas vs the jd3 baseline are unaffected.

**ADJUDICATION = CASE-0-ON-POSE (as pre-registered in H10)**: seg exhausted + pose cap-stopped in
full-rate descent ⇒ pose-dominant continuation, no stacked treatments, A/B menu deferred to the next
plateau. Endpoint composed S_vehicle (ema): 0.52305 + 0.9525 + 0.19643 = **1.672** vs ep1405 ≈ 1.911
→ window bought **−0.239 S** [macOS-CPU frozen-scorer advisory, training-vehicle semantics].

**JD5 FIRED**: ticket jd5_ticket_cont_ep1526.json (hash 8c8c2295…), epochs 1526→1646, out-dir
tr1_jd4_cont_ep1526, pid 50645, done-receipt jd5_cont_ep1526.
- **Poison-alarm false positive, resolved by derivation**: the regenerator dropped
  --jd1-force-ema-reanchor-on-resume — DERIVED behavior (maybe_force_window_reanchor: reanchor only
  when window geometry changes; jd5 keeps U=18000, parsed from the parent ckpt's own provenance).
- **Decay custody verified at three levels**: (1) trainer :4069-4073 restores active_ema_decay from
  ckpt jd1 state; (2) GOTCHA — the `resume` telemetry event shows the PRE-restore argv decay
  (0.99996) because the jd1-state restore runs after its emission and the no-reanchor path emits no
  event carrying the restored value (small telemetry-ordering debt, sister of H8); (3) ⚠ RR1-C2-R7-F2: jd5's first boundary_jump row falsely reports ema_decay_held=false — it
  compares the parent CONFIG decay (0.99996) instead of the parent ACTIVE jd1 decay (0.99978);
  do NOT consume that flag as basis-drift evidence; (4) the FIRST
  EPOCH ROW is the authoritative observable: **active_ema_decay 0.9997777777777778 CONFIRMED**,
  jd1_pose_finish_active, w_pose=1, ep_loss 0.9581 (resume bump within the priced ~tax envelope).

**Next boundary (ep1646)**: same probe instrument (edit --ckpt-tag/defaults for the jd5 dirs), same
adjudication frame; re-check tail-concentration (#775 repriced table) + tail-EMA A/B candidacy as
pose rate decays; conflict-controller condition re-check per the derived fire condition
(pose rate < ~0.03 S/window).
