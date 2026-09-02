# ddm_wc2_qbr1_bug_wallclock_realization_audit — read-only bug/confound + wall-clock + realization-stage audit of the LIVE qbr1 burn stack (fixes staged at the cell boundary; fire-alarm protocol for burn-invalidating findings)

## MANDATE

Operator 20260902: *"Wall clock optimization is always on the table and there are likely more
bugs to fix and realization for further realization."*
The walls inventory (`ddm_ww1_walls_that_werent_20260902.md`) measured BUG as the single most
common wall-faker in campaign history, and the realization path has yielded three separate
times when audited (96.6% of realization flips cured as solver bugs; 78.71% of manufactured
seg error at the native render with R+uint8 net REPAIRERS). Apply that taxonomy as a FORWARD
prior to the live vehicle: the qbr1 six-cell discriminator is burning NOW (cell 1/6,
~2.02 s/step aggregate, NO per-step timing telemetry exists — measured 2026-09-02, history.jsonl
carries no timing field). A measurement-corrupting bug found during cell 1 saves five cells of
burn; a wall-clock decomposition informs the wx1 Route 1 builder's much larger training. This
arm is READ-ONLY against the live burn: it hunts, decomposes, and STAGES fixes — it never
touches pinned sources or the burn tree mid-flight.

## SCOPE

1. **BUG/CONFOUND HUNT** (fresh-eyes, report-only, the confound-hunt reference form): read the
   burn entry `experiments/ddm_qbr1_born_fairform_burn_prep.py` (sha256 prefix 0c143eb232b8f849),
   the pinned trainer/loss modules named in
   `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/authorized_configs/seed_20260902_control_native100.json`
   `source_pins` (ce1_target_margin · qbt_packet_module · qbt_no2_gate · initialized
   latents/params — verify each pin against disk first; a pin mismatch is itself a finding),
   the milestone stratified-HT S_hat estimator, and the 3-outcome adjudication tool. Seeded
   hypotheses (then sweep beyond seeds): (a) HT weight/denominator correctness in the n32
   stratified S_hat (vacuity ⇒ report the denominator, m50); (b) history field semantics —
   `seg_expected_flip_realized` vs `realized_within_class_error.Lane` vs
   `seg_expected_flip_native_interface`: which field the LOSS consumes, which the adjudicator
   reads, and whether any consumer mixes them (MAIN already conflated two of these once on
   2026-09-02 — the #1260 retained-field reading-semantics genus, carried in
   `ddm_ww1_walls_that_werent_20260902.md` §4); (c) `ema_effective_decay` ramp vs the derived
   run-geometry law; (d) tau anneal trajectory vs the sealed config; (e) chunk-boundary /
   resume identity (a resume-smoke receipt EXISTS at
   `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/resume_smoke/` — consume it,
   do not re-run); (f) pair-sampling stratification correctness vs the m88/m96 prefix-bias
   cure it claims; (g) whether milestone scoring BLOCKS the train loop (wall-clock coupling).
   Every finding typed {BURN-INVALIDATING / BOUNDARY-FIXABLE / COSMETIC}. A
   BURN-INVALIDATING finding is a FIRE-ALARM: write it to the deliverable memo AND a
   `WC2_FIRE_ALARM.json` receipt in the arm store IMMEDIATELY — do not hold it for the memo.
2. **WALL-CLOCK DECOMPOSITION**: static per-step cost decomposition from the real code
   (forward / realizer / R / scorer / loss / EMA / history IO / milestone), THEN build a
   deterministic per-stage timing harness — flag-gated, DEFAULT-OFF, byte-identical when off
   (verify by hash on the resume-smoke tree, CPU only). PREPARE, do not execute on Metal:
   the burn holds the Metal lane (ONE-Metal law); emit a typed fire order for MAIN to run
   the profiled window at a cell boundary. Deliver the projected 6-cell + Route-1-scale
   wall-clock table with named levers ranked by projected minutes saved.
3. **REALIZATION-STAGE AUDIT**: apply the measured stage split of
   `ddm_mst1_manufactured_stage_split_20260822.md` (78.71% at native render; R+uint8 net
   repairers) + the 1.157× token→argmax amplification law + the penalty≠separation lesson
   (m154: a penalty term is not a separation guarantee; gradients live through R/uint8) to
   the qbr1 realizer path: verify the CE1 expected-flip margin is computed on the SAME
   lattice/interface the realized field is scored on (the native_interface/realized field
   PAIR exists in history — verify their producer sites agree with their names); enumerate
   addressable realization losses as typed rows for the wx1 Route 1 realizer design
   (consumer: `ddm_wx1r_conditioning_family_recall_inventory_20260902.md` §8 obligations).
4. **STAGED FIXES**: fixes to PINNED sources land as patch files + fire orders in the arm
   store (never applied mid-burn); fixes to non-pinned tooling may land via the serializer
   with 2 genuine review passes.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- The local SCORER LANE belongs to MAIN, always. Do NOT write who currently holds it into a
  charter: an occupancy claim goes stale the moment that holder exits, and the arm has no way
  to learn it did (the #1210 stale-precondition genus — MEASURED 2026-08-29, when
  `ddm_bz2_bornsmall_capacity_ceiling` correctly refused to claim a capacity ceiling because
  a charter told it a since-released lane was taken). If this arm's work needs a scorer run,
  emit a typed fire order naming its trigger and let MAIN fire it; landing an honest partial
  plus a fire order is the CORRECT outcome, never a failure.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_wc2_qbr1_bug_wallclock_realization_audit/`.
- DETACHED >30-MIN COMPUTE: any single compute step projected to exceed 30 minutes MUST
  launch outside the arm session with `nohup` + `disown`, a pidfile, crash-resumable stage
  checkpoints, and a durable done-receipt. The arm MONITORS that process; a successor or
  MAIN harvests the done-receipt. An in-session multi-hour compute loop is FORBIDDEN.
- CLOSED-FORM-FIRST (operator 2026-08-31 "All upstream can be closed form"): the scoring
  chain is frozen piecewise-analytic math with every non-analytic locus exactly known —
  derive/solve against the EXACT upstream operators (atlas:
  ddm_cfa1_closed_form_atlas_20260831.md) before any fit, surrogate, or sampled estimate;
  a fitted stage owes a one-line reason the closed form was not usable.
- LIVE-BURN BOUNDARY: the burn tree
  `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/{runs,launch,authorized_configs}`
  is READ-ONLY to this arm; the pinned sources named in the sealed config's `source_pins`
  MUST NOT be edited (patches stage in the arm store). NO Metal/MPS execution of any kind
  while the burn runs — the profiling harness verifies byte-identity on CPU against the
  resume-smoke tree only. Liveness reads ARTIFACTS (history.jsonl / checkpoints /
  DONE.json), never buffered logs (the run.log 0-byte false-alarm lesson,
  `no_naive_or_toy_ever_structural_enforcement_20260813` memory).

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_ww1_walls_that_werent_20260902.md` — the 6-mechanism dissolution taxonomy (BUG ·
  TOY-SCALE/PREFIX · CONFIG · WRONG-OBJECT · STALE-CONSTANT · PATTERN-OVERFIT); this
  charter is its forward application. Its rows also carry the #1087 window-cost artifact
  (a 50-step smoke became the campaign's 4.9×-wrong cost model — do NOT extrapolate cost
  models from tiny profiled windows) and harness ledger row #1090 (per-event wall-clock
  cost is the EVAL, not the save; 70.4% of a probe run was evaluation — the PRIOR the
  profiling leg tests).
- `confound_hunt_synthesis_20260705.md` — the 18-confound hunt whose meta-confound was the
  instrument certifying a frozen run as "converging"; the L1/L2/L3 immune-system layers this
  hunt inherits (report-only, positive-control sentinels, verdict-clearance).
- `ddm_na2_negative_audit_20260803.md` — prefix bias sign-inversion by axis (pose 2.54–4.21×
  anti-conservative); the reason qbr1's stratified-HT milestone screen exists at all — the
  estimator this arm audits is itself a CURE artifact; auditing it must not silently regress
  it to a prefix read.
- `ddm_mst1_manufactured_stage_split_20260822.md` — the realization stage split (78.71%
  native-render; R+uint8 net repairers); the realization leg consumes this decomposition
  rather than re-deriving it.
- Harness-id resolution (m89 same-line rule): #1087 → `ddm_ww1_walls_that_werent_20260902.md` ·
  #1090 → `ddm_ww1_walls_that_werent_20260902.md` · #1260 → `ddm_ww1_walls_that_werent_20260902.md` §4.

## OPTIMAL FORM

- Family exemplar: the fresh-eyes confound-hunt family's landed reference form is
  `confound_hunt_synthesis_20260705.md` (18 confounds, report-only hunters over orthogonal
  surfaces, L3 verdict-clearance) — this charter is that reference form aimed at the qbr1
  stack. Provenance pins: burn entry `experiments/ddm_qbr1_born_fairform_burn_prep.py`
  sha256-prefix `0c143eb232b8f849`; walls inventory landed at commit `e79b5fef82`; recall
  inventory at commit `7994616d60`; sealed config
  `authorized_configs/seed_20260902_control_native100.json` is the pin AUTHORITY (receipt
  path: `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/authorized_configs/`).
- SCOPE reductions declared per row: profiled-window EXECUTION deferred to the cell
  boundary (read-only during burn) = SCOPE reduction; audits run on the REAL pinned sources
  and REAL live artifacts, never fixtures = no MECHANISM reduction. MECHANISM reductions
  FORBIDDEN.
- **PRIOR-LAW PREDICTION (falsifiable):** the #1090 eval-cost law (carried in
  `ddm_ww1_walls_that_werent_20260902.md` sourcing) predicts the realized-through-R scorer
  evaluation inside the loss dominates qbr1 step time — ≥50% of the 2.02 s/step aggregate.
  FALSIFIER: the per-stage timing table shows scorer+realizer stages <50% — then the
  bottleneck is elsewhere (data/EMA/IO/milestone) and the lever ranking reorders; count it
  plainly if it lands.

## DELIVERABLE

`.omx/research/ddm_wc2_qbr1_bug_wallclock_realization_audit_20260902.md` — typed rows:
findings `{finding, class ∈ BURN-INVALIDATING|BOUNDARY-FIXABLE|COSMETIC, evidence path,
staged-fix path or NONE, fire-order trigger}`; the per-stage wall-clock table (static +
harness-projected) with ranked levers; realization rows for the Route 1 consumer; the
`WC2_FIRE_ALARM.json` protocol receipt (present even if empty: `{"alarms": []}`). Commit via
the serializer. End with the own-vehicle frontier line.
