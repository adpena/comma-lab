# v7.5.2 FROM-SCRATCH PILOT — pre-registered decision record (operator-driven hold-gate 2, 2026-07-10)

**STORES CONSULTED:** `DUAL_CHAIN_BRIEF_385_20260710.md` (ADDENDUM v2 + operator GO) · owed-16 verdict
(`owed16_verdict_20260710.json`, EmpiricalAnchor `owed16_realized_transfer_measured_zero_20260710`) ·
DAG FEED-owed16-verdict (reformulation queue) · `SPEC_v75_optimal_single_trunk_20260708.md` §4/§5/§8 ·
mod32cap run dir `experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z` (run.log verdict
rows, MEASURED) · crucible_v752 compile (`tac.witness_autoconfig.compile_crucible_v752_launch_config`,
commit `3b028a374`). Pointer contest-CPU **0.19110 UNMOVED** — everything here is MEANS.

## 1. WHY (the hold's second gate — operator verbatim "We don't want to launch v7.5.2 if we know its confounded or not optimal")

The self-orient-OFF decision (owed-16 P9 RESOLVED-REFUTING) rests on **warm-start evidence only**
(verdict_scope: FORMULATION — bounded ep650→700 fine-tune from a self-orient-TRAINED parent). The
uncovered arm is FROM-SCRATCH: the parent trunk may carry internalized directional structure that
persists in the OFF ablation, so OFF-from-scratch could lag at PARTITION FORMATION (ep0–300) even though
OFF-from-warm-start is ≈0-loss. This pilot closes that formulation-scope gap BEFORE the 6–16 h launch
commits. (DAG FEED-owed16-verdict reformulation-queue item 1, now elevated to a launch gate.)

## 2. THE PILOT (config = the launch's own first 300 epochs; VERIFIED byte-close)

`compile_crucible_v752_launch_config(gt_n600, num_pairs=600, epochs=300)` — the GO'd self-orient-OFF
launch config with ONLY the epoch cap changed. **MEASURED argv diff vs the epochs-3000 launch config:
exactly 2 tokens** — `--epochs 300` (vs 3000) and `--polyak-finisher-start-epoch 301` (vs 2546; the
degenerate clamp = epochs+1 ⇒ count=0). Both Polyak starts are > 300 ⇒ **the pilot's 300 epochs are
training-dynamics-IDENTICAL to the launch's first 300 epochs** (all schedule pins are ABSOLUTE:
anneal-den 3000, tau@300, muon/pose-finish caps 726 — none rescale with --epochs). Already in-config:
`--seed 0 --eval-every 25 --verdict-pairs 0 --verdict-batch 32 --ckpt-every 25 --stage-checkpoints`
(resumability P0). DSL-validated 0 violations; parses the real trainer argparse clean.

**The RESUME-FROM-PILOT option (why the pilot costs ~nothing if it passes):** because the pilot IS the
launch's first 300 epochs, a PASS verdict lets the real launch fire as
`--config crucible_v752` (sealed epochs 3000) + `--extra-trainer-flags "--resume-from <pilot_out_dir>"`
— restoring the pilot's ep300 checkpoint and continuing to 3000, trajectory-faithful to an
uninterrupted launch (the trainer's resume contract). The pilot's wall-clock is then simply the
launch's own first ~10%, not an added cost.

## 3. PINNED GOVERNED PILOT COMMAND (fires ONLY after gate-1's machine frees; see §5)

```bash
cd /Users/adpena/Projects/pact
.venv/bin/python tools/launch_witness_run.py \
  --config crucible_v752 \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 \
  --epochs 300 \
  --out-dir experiments/results/levelset_v752_pilot_<TS>
```

(launcher stamps the explicit-epochs override note; pilot wall-clock budget DERIVES to 0.831 d; full
gate chain — memory-preflight ~24.5 GiB projected, DSL-config, schedule-provenance, safe-compile,
system-admission, throughput — runs unmodified; durable governed spawn.) Wall-clock estimate: see the
dry-start report (`experiments/results/__v752_drystart__/dry_start_report.json`) for the measured
sec/ep; mod32cap's from-scratch cadence anchor is ~116 s/ep (ep25→300 in 8.88 h, self-orient-ON) —
the OFF pilot should run at or below that; ~300 ep ⇒ roughly a 6–10 h read-out.

## 4. PRE-REGISTERED COMPARISON PROTOCOL + PASS BAND

**Reference = mod32cap's banked from-scratch n600 trajectory** (run dir above; MEASURED verdict cells,
CE stage, self-orient ON, mod-dim 32):

| ep | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 | 250 | 275 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d_seg | 0.7439 | 0.009288 | 0.007248 | 0.006354 | 0.005856 | 0.005519 | 0.005288 | 0.005121 | 0.004963 | 0.004869 | 0.004751 | 0.004682 | 0.004571 |

**HONESTY CAVEAT (binding on the verdict's scope):** mod32cap is NOT a controlled self-orient A/B
against the pilot — it differs in the WHOLE v7.5 lever stack (no counter-force / ladder-birth /
dash-comb / temporal-screw / taper / unify-τ; different curriculum pins; islands-unborn DELIBERATE).
It is the REFERENCE ANCHOR for "healthy from-scratch formation at this capacity on this GT", not an
isolation of the basis. A controlled from-scratch OFF-vs-ON matched pair remains reformulation-queue
item 1 if this pilot is ambiguous.

**Facets read (holistic, per the SPEC §5 watch-list — never the composite alone):**
1. **Composite d_seg descent** at matched epochs (the primary band, below).
2. **Descent RATE** — the log-slope over ep100→300 (mod32cap: ln(0.005856/0.004571)/200 ≈ 1.24e-3/ep).
3. **Island birth** — part_frac[lane]>0 by ep25 (the paint-then-SDF seed admission gate) and
   lane/movable part_frac approaching the Chan-Vese equilibrium band ≈1.25×GT (v752-only machinery;
   mod32cap has NO birth stack — v752 should be STRICTLY better here; a v752 lane/movable
   islands-unborn read-out is a REGRESSION vs its own design, independent of mod32cap).
4. **Per-class d_seg** — Road PRIMARY (SPEC §5: the Chan-Vese success target; run-1's 13.8× over-paint
   is the failure signature to watch), Lane/Movable vs the birth-stack expectation.
5. **d_pose** — WATCH-ONLY at ep≤300 (pose-finish gates at the muon event ≥726; pose-blind by design
   in the pilot window; mod32cap's ~104–134 band is the analog).

**PASS BAND (pre-registered):**
* **PASS → launch OFF** (optionally resume-from-pilot-ep300): pilot composite d_seg ≤ **1.15×**
  mod32cap's matched-epoch cell at ep100–300 (≤ **1.25×** at ep25–75, the noisier formation cells),
  AND log-slope(ep100→300) ≥ **0.7×** mod32cap's, AND island-birth facet healthy (lane part_frac>0 by
  ep25; no run-1-style >4× GT over-paint persisting past ep150 against the Chan-Vese equilibrium).
* **LAG → basis matters at formation:** pilot composite d_seg > the band at ≥2 consecutive matched
  cells in ep100–300, OR log-slope < 0.7× — the launch config switches to
  **self-orient-REBALANCED-early-annealed-OFF** (directional channels ON at formation with the
  freq-along-heavy allocation per the owed16v2 rebalance verdict, annealed off after the partition
  forms), compiled through the DSL as its own amendment before any launch.
* **AMBIGUOUS** (band-straddling / facet disagreement): NO launch; run the controlled from-scratch
  OFF-vs-ON matched pair (reformulation-queue item 1) before committing. One crisp verdict, then act.
* Band rationale: 1.15× ≈ the largest adjacent-cell step in the reference's ep100–300 window (so a
  within-band pilot is indistinguishable from one verdict-cadence of ordinary progress); the P2
  single-seed noise floor is unmeasured at these cells (owed) — the band is therefore deliberately
  WIDE and the LAG trigger requires 2 consecutive cells, never one.

## 5. SEQUENCING (the two pre-registered hold gates, then the launch fires with no further gates)

1. **Gate 1 — owed16v2 rebalance verdict** (in flight, `owed16v2_rebalanced_ON_20260710T114759Z`,
   pid 64206): reads out the freq-along-heavy warm-start arm vs the banked OFF trajectory. Feeds the
   LAG branch's allocation (and could independently inform the launch config).
2. **Gate 2 — this pilot:** fires AFTER the rebalance arm completes and frees the machine (~22 GiB
   pilot admits easily solo). Read-out per §4.
3. **Launch:** whichever config §4 selects, via the governed launcher; if PASS, optionally
   `--resume-from` the pilot ep300 checkpoint.

**Pointer 0.19110 UNMOVED — this record is MEANS.** Only a byte-closed `upstream/evaluate.py` n600 row
< 0.19110 moves it.

## 6. P0 FIXES FOLDED (operator directive 2026-07-10; fresh-eyes advisory `ADVISORY_v752_fresh_eyes_20260710.md`) — landed `fbeb20ae5` (+ trainer hunks absorbed into sibling co-land `acc98f2a4`, content-verified)

**Zero-effect-pre-pose_finish VERIFICATION (the directive's step-1 precondition, SOURCE-VERIFIED):**
the pilot's 300 epochs are pose-blind under BOTH gate modes, structurally:
* muon mode (pilot): `pose_finish_on = muon_fired OR ep≥726`; `muon_meat_event.fired` requires
  `nucleation_complete = ladder_arms_complete(ep, [lane 80+0+260=340, movable 60+0+200=260])` ⇒ true
  only at **ep ≥ 340** (trainer L8719-8721 + `event_wirings.ladder_arms_complete`); backstop 726.
* sigma mode (fixed launch): `cond_fired` needs the σ_min plateau detector with
  `min_points = settle(3)+hysteresis(3)−1 = 5`; σ_min points arrive at the jacobian-basin cadence
  (every 4th verdict = every 100 ep: ep0/100/200/300/400) ⇒ **earliest possible fire ~ep400**
  (`sigma_min_plateau.SigmaMinPlateauConfig` + trainer L6668 observe site); backstop 726.
⇒ pose weight = 0 through ep300 in BOTH configs; the ONLY checkpoint-content delta is the additive
`pose_finish_conditioning_gate` resume-registry key (absent-in-sidecar restores to un-fired — the
__bc_* additive pattern), which is exactly what the §7 resume-compat check confirms empirically.
CONSEQUENCE: the pilot may fire with the P0-composed config (argv ≡ launch argv modulo the 2 epoch
tokens — strictly cleaner for resume-from-pilot); the checkpoint at ep300 is behaviorally identical
either way.

**P0-1 landed:** `compile_crucible_v752_launch_config` composes `PoseFinishConditioningGate`
(`--pose-finish-engage-on sigma_min_plateau`; backstop 726 + w_pose 1.0 from the inherited pose
config); the stale absence-assertion test now asserts PRESENCE; the owed-1 "does NOT exist yet"
comment block corrected (#383 landed). **Expected-active-lever manifest** (advisory P0-1 required
gate): pinned `CRUCIBLE_V752_LAUNCH_EXPECTED_LEVERS` (9 levers) enforced fail-closed at COMPILE and
re-checked at the LAUNCHER (rc=10; runs for dry-run/dry-start/real) — built-but-not-composed extinct.
**P0-2 landed (minimum-viable):** `resolve_pose_finish_engage` (pure, unit-tested,
`tac.witness_control.sigma_min_plateau`) — the epoch backstop engages ONLY when the detector state is
NOT degenerate; DEGENERATE at the backstop ⇒ NO engage + banked-R1 selected + the loud typed row
`pose_finish_backstop_overridden_banked_r1` (once). A real conditioning fire always engages;
healthy-never-fired at backstop = the fail-safe engage (unchanged).

**OWED (ledgered, NOT launch gates per the pre-registered no-further-gates decision):**
* P0-2 full state machine ARMED→ENGAGED→ACCEPTED/REGRESSED_ROLLBACK/BANKED_COMPLETE_ARTIFACT
  (the minimum-viable guard covers the degenerate-override; rollback/accept custody is the owed end
  state). Untrusted-(canary-fail)-at-backstop currently engages via backstop — covered by the state
  machine, owed.
* advisory #3 (P0-3): banked-R1 = complete checkpoint/archive fallback (measured); an R1-dxi GRAFT
  onto an arbitrary v7.5.2 EMA is unmeasured/unimplemented — bank descriptors need the full
  compatibility-key binding.
* advisory #4 (P0-4): chroma (--seg-chroma-boundary-*) is INHERITED into launch-1 though sealed as a
  later add-back rung — a Class-B attribution confound to resolve at the ladder, not a new gate.
* advisory #5 (P0-5): amber realization (OI-5) remains open — inherited --grad-clip 1.0 defeats a
  naive preset; realize-or-waive at the P8 wall.
* advisory P1-3: pose engage mode not restart-protected (`__cfg` lacks pose_finish_engage_on;
  `_resume_lever_divergences` has no pose-mode check) — fold into the §7 resume-compat verification.

## 7. STEP-3 (pre-FULL-launch, updated per the directive): re-run the dry-start with the CORRECTED
config (expected-lever manifest green end-to-end) + confirm resume-compat from the pilot ep300
checkpoint (the composed engage-on flag + registry key must restore additive/legacy-compatible: an
absent-key pilot sidecar resumes to un-fired under the launch config).

## 8. PILOT FIRED (2026-07-10T15:41Z) — AMENDMENT: realized as the REAL config, ep0-300 read as the pilot gate

**Gate-1 CLOSED:** owed16v2 rebalanced-ON read 0.004213@ep700 vs OFF 0.004181 (0.004286 vs 0.004244
@ep675) — marginally WORSE at every cell; allocation-independence CONFIRMED; SELF-ORIENT-OFF stands
(verdict commit 40b2ed211; verdict_scope formulation — from-scratch is the uncovered arm THIS pilot tests).

**The --epochs 300 pilot config was REFUSED by the trainer's own stage-interlock validator** (verbatim:
`--muon-start-epoch (726) must be in [1, --epochs (300)]: outside the budget the Muon finisher would
NEVER engage -> a silent no-op = a FALSE 'Muon does not help' verdict`). The refusal is respected —
the sealed 3000-epoch schedule is atomic (the §2 "2-token diff" is DSL-valid but not trainer-launchable).
**Realization (strictly closer to the §2 contract):** the pilot IS the real launch —
`--config crucible_v752` at the SEALED epochs 3000, with **ep0-300 verdict cells read as the pilot
gate** per §4. PASS ⇒ the run simply CONTINUES as the launch (the resume-from-pilot option realized as
no-restart-at-all); LAG ⇒ governed stop (checkpoints preserved) + config switch per §4.

**LIVE:** run dir `experiments/results/levelset_v752_pilot_20260710T154100Z` · trainer pid 44491 (durable
daemon, safe_run rss-cap 90000 MiB) · costate shadow observer pid 45290 · dashboard :8790 auto-tracking.
Full gate chain GREEN incl. the NEW expected-active-lever manifest (9 levers) + dsl-config (157 flags,
P0-composed) + mem-preflight 24.48 GiB projected @ safe-frac 0.85 (sole workload) + admission ADMIT
(headroom 66.7 GiB) + throughput 430.7 ms + wall-clock 7.95 d ≤ 8.31 d budget. Launch defect found+fixed
in the same landing: the launcher's activation-ledger record raised PosixPath TypeError (all 9 fire
records silently dropped) — fixed (str(out_dir)) + 9 rows BACKFILLED.

**Read-out protocol unchanged (§4):** matched cells ep25..300 vs mod32cap; ep300 waiter armed
(marker `.omx/tmp/v752_pilot_ep300.marker`). Wall-clock to ep300: UNMEASURED solo OFF cadence —
mod32cap anchor ~116 s/ep (ON) ⇒ projection ~10 h (labeled projection, not measurement).

## 9. AMBER + CHROMA DECISION PACKAGE (operator elevation 2026-07-10 "Chroma rung and amber are important to pursue")

### 9a. SOURCE-VERIFIED live state (run `levelset_v752_pilot_20260710T154100Z`, ~ep0)
`--grad-clip 1.0` (explicit) · `--per-group-grad-clip` ON · NO `--stability-preset` · NO
`--pose-grad-coeff-max` (default 0 = OFF) · NO `--grad-normalize` (default none). Chroma INHERITED:
`--seg-chroma-boundary-weight 0.1 / margin-band 1.0 / start-epoch 450 / start-event annulus_plateau`.

### 9b. Mechanism findings (three defects surfaced, none blocking)
1. **The advisory's "explicit --grad-clip defeats the preset" is a DOCSTRING, not code:**
   `resolve_stability_config` implements explicit-wins ONLY for coeff-max; `preset=amber`
   unconditionally sets grad_clip 0.5. Comment/behavior mismatch (trainer L10291 + module docstring)
   — surfaced; our composition uses EXPLICIT values (no preset) and is immune either way.
2. **SPEC §1.1 lists `--grad-normalize` in amber; the BUILT `tac.witness_stability.AMBER` has NO
   grad-normalize field.** Spec-vs-built gap — NOT silently closed (we compose the 3 BUILT cures;
   grad-normalize needs its own derivation/decision if wanted).
3. **The annulus_plateau sensor is structurally DEAD at eval-every=25:** trailing dwell_windows=4
   verdict points span 75 ep < min_epochs 150 ⇒ `dwell_ok` never true ⇒ chroma AND temporal-screw
   engage ONLY at their ep450 caps (the "event-wired" claim is decorative at this cadence). Fix =
   dwell_windows 4→7 (or derive from eval_every) — OWED, not launch-blocking (caps are the designed
   fail-safe). CONSEQUENCE: the pilot window ep0-300 is structurally CLEAN of chroma + temporal-screw.

### 9c. Amber materiality (MEASURED both sides)
* **Binding side:** mod32cap gnorm > 1.0 at **998/1001 steps** (raw 18–1500+) ⇒ the clip binds at
  essentially EVERY step ⇒ grad-clip 0.5 HALVES the effective step size run-wide — amber is a
  material dynamics change wherever active, NOT a rare-event guard.
* **Inert side (ep0-300):** coeff-max 25 acts on the POSE gradient coefficient — structurally inert
  while w_pose=0 (pose-blind ≤ep300, §6 verified); per-group clip already ON. So amber's ONLY active
  component ≤ep300 is the clip halving — which would put the pilot on a different effective-LR
  trajectory than EVERY banked reference (mod32cap, owed16 arms: all clip 1.0), invalidating the §4
  pre-registered band and making a LAG unattributable (basis-matters vs clip-slower — the exact
  misattribution the pilot exists to prevent).
* **The spec's own derivation:** amber's collapse anchor is the batch=1 deep-unroll POSE-coefficient
  blowup (5/√(10·d_pose) at d_pose→0) — binding at JOINT POSE DESCENT (≥ep726) + the Muon boundary;
  AMENDMENT-4 demoted the amber×Muon crush arm to WATCH. Amber's protective value ≤ep300 ≈ nil
  (two prior clip-1.0 runs through this exact window are the measured-healthy evidence).

### 9d. THE OPTIONS
* **Option A — stop + relaunch NOW, amber composed from ep0.** Cost: ~20 min (run is at ~ep0; ZERO
  trained epochs lost — the "1.5h" is boot+ep0-verdict wall-clock). Gain: the TRUE sealed §1.1
  program for the whole run. **Risk (decisive): the ep300 pilot gate loses its comparison validity**
  (clip-0.5 trajectory vs clip-1.0-calibrated band; 998/1001-step materiality) ⇒ a LAG cannot be
  attributed ⇒ the launch-config decision the pilot feeds becomes a guess. Also clip-0.5-through-CE
  is UNMEASURED on any witness run (first-arm risk taken at the least-informative moment).
* **Option B (RECOMMENDED) — amber engages at ep300 = the CE→tau stage boundary, via stop+resume.**
  Mechanics: at the ep300 read-out (already the pilot decision point), stop cleanly + relaunch
  `--config crucible_v752` amber-composed with `--resume-from <run>` — grad_clip is an optimizer
  hyper (NOT resume-divergence-guarded; applies from the resumed epoch; VERIFIED), ep300 is a genuine
  stage boundary (tau@300 pin) so the loss/optimizer-changes-at-stage-boundaries discipline is
  satisfied exactly; the launcher's new startup-telemetry assertion verifies the runtime-resolved
  values. Cost: ONE extra boot (~15-20 min) at ep300; ZERO epochs lost. Gain: pilot comparison stays
  valid; amber active for tau/l7/Muon/pose-finish = BOTH binding moments; the amber WATCH sensor set
  (per-class d_seg slope · effective-step norm · direction-cosine) rides the first amber arm as the
  spec intended. Deviation: amber not active ep0-300 — the §9c inert-side evidence is the
  authority-bearing rationale.
* **Option C — full-run waiver.** Weakest; the spec calls amber launch-blocking; NOT recommended
  (and unnecessary — B realizes it at the binding moments for free).

**BUILD READY (either A or B): `compile_crucible_v752_launch_config(amber=True)` — landed
`95d9f429f`, 2 semantic flag diffs (grad-clip 0.5 + coeff-max 25), expected_stability manifest +
launcher startup assertion, 66 tests green.** Not wired to any launch — awaiting the operator pick.

### 9e. CHROMA VERIFICATION (P0-4; finding only — the rung build is the sibling's)
The live config INHERITS the 4 chroma flags (9a) while the SEALED program HOLDS chroma at LADDER
rung 3 with its add-back **UNMEASURED** — SPEC_v752 rung-3 verbatim: *"UNMEASURED add-back (ablation
≠ add-back — S5-N10) … MUST MEASURE add-back ΔS (constant-luma ablation FLIPS 7.54% is a WORTH, not
a GAIN)"*. **NO prior measured add-back receipt exists** (the 7.54% constant-luma figure is the
worth-indicator the advisory alludes to; it is not a receipt). ⇒ the inherited state DEVIATES from
the sealed intent. Materiality: chroma engages ONLY at the ep450 cap (9b-3: the event sensor is
dead at this cadence) ⇒ the pilot window is clean; the deviation becomes ACTIVE at ep450 — i.e. the
same ep300 stop+resume that realizes amber (Option B) can ALSO drop the 4 chroma flags to restore
the sealed hold-out (chroma weight is a loss-form key — check `__cfg` divergence handling at the
resume; if guarded, the sidecar records the drop LOUDLY, which is the correct custody). Rung-3
add-back measurement = the sibling chroma-rung builder's job; this record feeds it.
