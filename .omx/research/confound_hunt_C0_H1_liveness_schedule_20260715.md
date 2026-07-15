# Confound Hunt H1 — LIVENESS + GUARDS + SCHEDULE-FIRING on the live C0 baseline

**Run:** `experiments/results/levelset_n600_witness_20260715T095030Z/` (pid 72377, C0 baseline
`v9_cgauge_ideal_mod19`, n600, 3000ep, eval-every 25). **Scope:** $0 report-only, no
training/dispatch/score-claim. **Hunter:** H1 (siblings own measurement-authority / lever-efficacy /
config-drift). **git_sha at run launch:** f0920477 (git_dirty=true). **Verdict:** surface is
LARGELY CLEAN — the L5 median-freeze bug class is CURED (not the active guard) and the schedule has
LOUD backstops on every event transition. Two residual risks worth a Phase-2 gate, one monitoring
duty, and several doc-readability traps. No finding poisons C0's eventual d_seg verdict PROVIDED the
liveness precondition (F4) is checked before the verdict is read.

Format per finding: **[signature · cited-fact · poison-scope · L1/L2/L3 fix]**

---

## CLEAN (cited proof the confound is absent)

**C1 — spike-guard is NOT the L5 median-freeze legacy.**
`--spike-guard-mode` default = `"rollback"` (`train_levelset_witness_realized_through_R_mlx.py:12851`);
launch.sh does NOT pass `--spike-guard-mode` → the live guard is `rollback`, the physics-informed cure
(single spikes ACCEPTED/stepped; sustained runaway → rollback-to-last-good + lr-cut + fresh re-arm;
bounded `max_rollbacks` then reverts to LOUD legacy skip). The L5 anchor
(`[[spike-guard-median-freeze-deadlock-ep-loss-zero-signature]]`, accepted-only median cannot re-arm →
ep_loss==0 froze runs at ep103-114) is NOT the active guard. Even the base-loop legacy path carries the
C17 escape hatch: `consec_skips >= recent_maxlen` → `spike_deadlock ALERT` → `recent.clear()` re-arm
(`train_witness_realized_through_R_mlx.py:2509-2522`) + epoch-level frozen alarm `accepted_frac <= 0.02`
→ `spike_deadlock ALERT scope=epoch` (`:2560-2572`). **CLEAN.**

**C2 — verdict/telemetry rows carry a LIVENESS stamp; a frozen run cannot be mis-read as converging.**
The verdict row is stamped with `accepted_frac`, `weights_stepped`, `accepted_batches`,
`skipped_batches`, `frozen_epoch`, plus an explicit "async verdict on a FROZEN-state epoch (all batches
skipped…)" note (`train_levelset_witness_realized_through_R_mlx.py:8597-8601`; emitter
`:3462-3487`/`:3882-3961`). This is the L5-cure fully realized at the reader surface. **CLEAN.**

**C3 — every event transition has a LOUD fail-safe backstop cap; NO silent never-fire.**
The four sensor→start transitions (muon/lane-band/chroma/temporal-screw) are `EventBackstopGate`
instances (`src/tac/witness_control/event_wirings.py:106`), driven every epoch
(`train_levelset_...:10888` muon, `:11151` screw, `:11213` chroma, `:11268` lane-band). If the wired
sensor never fires, the cap fires and emits a LOUD `cap_fired_before_event` row flagged
"FAIL-SAFE BACKSTOP FIRED … falsification-relevant (S5)" (`event_wirings.py:201-217`). All caps are
set/positive in argv (muon `--muon-start-epoch 726`, lane `--lane-band-start-epoch 500`, chroma
`--seg-chroma-boundary-start-epoch 450`, screw `--seg-temporal-screw-start-epoch 450`) → the transition
is GUARANTEED to fire by the cap epoch. A cap of `None`/`<=0` would be the silent-never-fire hole
(`:147-148`) — not present here. The `--tau-advance-mode event` controller (the ONE live event
controller under unify-tau) carries the SAME discipline: a `FAIL_SAFE_CAP` per-octave max-dwell so
"a dead / mis-calibrated sensor cannot stall the descent forever"
(`src/tac/witness_control/tau_advance.py:101-106`). **CLEAN.**

**C4 — the "l7 stage" the prompt flagged is an INERT DOC REFERENCE, not a live stage; no term silently
breaks.** Under `--seg-form-unify-tau` (set in argv), `_seg_form_for_epoch` ALWAYS returns `"unify_tau"`
for every epoch (`train_levelset_...:2596-2597`) → the discrete `l7_softplus` stage is NEVER entered
(run.log `event_curriculum_inert_under_unify`). The seg-margin-satisfice
"MASK-BY-STAGE at l7 preserves tau-anneal" note (`:6385`, `:7108`) describes an OPTIONAL config mode
(`ms_start >= l7_start`); this run sets `--seg-margin-satisfice-start-epoch 0`, so the term gates ONLY
on `ms_gate["on"]` (= `ms_start <= 1` = True from ep0) and `ms_w>0` (`:6387`) with NO `seg_form=="l7"`
dependency. So no live term keys on the dissolved l7 stage → nothing silently inert because of it.
**Answer to the prompt: there is NO live l7 stage under unify-tau; the l7 reference is inert doc.**

---

## RESIDUAL RISKS / CONFOUNDS (ranked)

**F4 [HIGH — MONITORING DUTY, the L3 precondition] · liveness UNCONFIRMED at epoch level ·**
run.log ends at `{"stage":"mem_probe","phase":"before_v0_verdict"}` (last write 04:52:48; now 05:03:52,
11 min, ZERO new lines); process is alive (pid 72377, state R, 100% CPU, RSS 25.7 GiB) but the first
verdict (ep25) has NOT landed and NO grad/epoch row has flushed. So process-liveness is CONFIRMED but
epoch-training-liveness (ep_loss>0, accepted_frac healthy, weights_stepped) is NOT YET OBSERVED. Per
`[[log_silence_is_not_a_stall_confirm_progress_before_killing_a_run]]` this is EXPECTED (n600
through-R + 1-thread CPU verdict + structured-init already burned) and NOT a stall — do NOT kill. ·
**poison-scope:** C0's eventual d_seg verdict is load-bearing ONLY if the interpreted window's rows show
`frozen_epoch=false` + `accepted_frac` above floor; a Phase-2 A/B that reads C0 d_seg without first
confirming liveness on that row would be poisoned. · **L3 fix:** apparatus-validity precondition for
ANY C0 verdict = liveness stamp present-and-healthy on the interpreted row (already emitted per C2); the
duty is to READ it, not add machinery. Positive-control sentinel present for ONE gate only (pose-finish
`canary_positive_fired:true` / `canary_negative_fired:false`, run.log `pose_finish_gate_setup`); there
is NO global positive-control for the d_seg-descent apparatus itself → L3 defense-in-depth GAP (a
known-effect d_seg canary the run must register each session), non-blocking.

**F5 [LOW — L1 GAP] · partial-freeze blind band · intermittent-skip regime raises no alarm ·**
the C17 escape hatch fires only on `consec_skips >= recent_maxlen` (20 CONSECUTIVE,
`train_witness_...:2509`) and the epoch alarm only on `accepted_frac <= 0.02` (`:2563`). An
INTERMITTENT-skip regime (e.g. accepted_frac ~0.05–0.30, skips non-consecutive so `consec_skips` keeps
resetting at `:2530`) trains SLOWLY with NO LOUD alert — `accepted_frac` IS stamped on the row (C2) but
a reader interpreting slow d_seg descent as a "converged plateau" could misattribute gradient
starvation. · **poison-scope:** could bias a plateau/convergence READ of C0 or a Phase-2 A/B if the arm
sits in the blind band; visible-but-not-loud. · **L1 fix:** add an intermediate `partial_freeze` WARN
row when `0.02 < accepted_frac < ~0.5` over an epoch (the epoch-alarm already computes `_accepted_frac`
at `:2562` — one extra branch).

**F6 [INFO — DOC-READABILITY TRAP, already LOUD] · "event curriculum inert" vs live per-transition
event gates · ** argv carries `--curriculum-event-triggered` (INERT under `--seg-form-unify-tau` —
run.log `event_curriculum_inert_under_unify`, the CE→tau→l7 boundary-firing controller is dissolved)
ALONGSIDE the FOUR live per-transition `EventBackstopGate`s (muon/lane/chroma/screw) AND the live
`--tau-advance-mode event` controller. A reader could mis-read "event curriculum INERT" as "all event
scheduling inert" — the opposite is true (the transition gates + tau-advance ARE live; C3). · **fix:**
already surfaced LOUD in run.log (`event_curriculum_inert_under_unify` note names Muon's EventBackstopGate
+ the LIVE tau-advance controller explicitly); no code change, flagged for reader awareness.

**F7 [INFO — operator-acknowledged WARNs] · schedule ordering · ** `muon_start_epoch 726 <
l7_start_epoch 800` (run.log `muon_finisher_WARN`) and `lr_anneal_epochs 1000 < epochs 3000`
(`lr_anneal_epochs_WARN`) — BOTH surfaced LOUD as WARN and operator-allowed. Under unify-tau there is
no l7 partition-forming stage (C4) so the muon<l7 placement warn is moot for partition formation. Not a
confound; already loud.

**F8 [INFO — lane never-born → cap fires on unborn lane] · lane_nucleus event predicate ·**
`lane_nucleus_event` requires `born = part_frac > min_part_frac` (`event_wirings.py:364`); at init
`part_frac[lane=1]=0.0` (run.log `structured_init`) and `--curriculum-nucleus-min-part-frac 0.0` →
`born = 0.0 > 0.0 = False`. If the lane never births (anchor `[[L2]]`/`[[L3]]`: lane not-static → init
NO-OP, historically zero lane islands), the `lane_nucleus` event NEVER fires and the lane-band
transition fires on the cap (ep500) with a LOUD `cap_fired_before_event` row — engaging the lane-band
term on an UNBORN lane. NOT silent (C3), but a firing lane-band cap is the falsification-relevant signal
that the lane never nucleated. · **poison-scope:** H3 owns whether the fired-on-cap lane-band term is
then INERT; H1 confirms only that the cap-fire is LOUD, not silent. · **fix:** already L1-loud; watch
for `cap_fired_before_event transition=lane_band` in the run.

---

## L3 verdict-clearance statement (binding)

C0's eventual d_seg verdict is admissible as load-bearing ONLY if, on the interpreted verdict row(s):
(1) `frozen_epoch == false` AND `accepted_frac` above the ~0.02 floor (liveness — emitted per C2, must
be READ per F4); (2) the stage under interpretation is ACTUALLY ACTIVE at that epoch — for a
post-ep500/450/726 read confirm the corresponding transition fired via `start_event_fired` OR
`cap_fired_before_event` (C3), NOT merely that the flag is in argv; (3) if a Phase-2 A/B reads C0 as the
CONTROL, the control arm must satisfy (1)+(2) in the SAME epoch window as the treatment. Positive-control
sentinel exists for the pose-finish gate only (F4); the d_seg-descent apparatus has no global canary —
a NOVEL confound in the d_seg path would require the F4 monitoring miss AND the absent global canary,
so treat (1)+(2) as MANDATORY reads, not assumed.

Pointer 0.19108 UNMOVED — this hunt is MEANS (apparatus validity), not a score mover.
