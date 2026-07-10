# Telemetry enhancement audit — v7.5.2 / v7.5.3 / v8 levers (#404, P0 2026-07-10)

**Operator P0 (verbatim 2026-07-10):** "given our latest changes to v7.x and v8 should we do anything
to enhance our telemetry ... All of that is p0 update the triality and tasks accordingly ... Update
costate controller accordingly as well."

**Mission framing (honest):** this is MEANS. Pointer 0.19108282 [contest-CPU] is unmoved by this work;
it moves only through a byte-closed `upstream/evaluate.py` n600 row. This audit makes the levers of the
IMMINENT pointer-moving run provably BINDING-vs-INERT (the confound class that already cost us: the
inherited `--grad-clip 1.0` defeat, the spike-guard median-freeze, the SigmaMinPlateauDetector silent
2.5 h crash).

**Discipline:** docs/operating_manual_craft_handoff.md (re-derive from primary artifacts; label
MEASURED/DERIVED; a verdict from a corrupted instrument is not a verdict). Per CLAUDE.md
"'Off' is a tracked queue": read-only telemetry defaults ON. Per "Confound self-protection": these are
L1-alarm READ surfaces + instrument-liveness (L3) checks.

**HARD CONSTRAINT honored:** the sealed v7.5.2 relaunch chain
(`experiments/results/__v752_drystart_final__`, mid-dry-start) was READ-ONLY throughout. ZERO edits to
`experiments/train_levelset_witness_realized_through_R_mlx.py` or any sealed-launch-path file. All
trainer-touching telemetry below is DESIGNED + QUEUED (gated on the relaunch/resume boundary or a
stage-boundary resume), never landed hot.

## STORES CONSULTED

CLAUDE.md (§Off-is-a-tracked-queue, §Confound self-protection, §v7.5/v8 operating contract,
§Capstone trainer canonical entry point) · MEMORY.md (L31 default-off=orphaned-signal, L66 annulus,
L67 #205 CE-floor, L68 pose banked-R1, L19 T5 crucible) · `.omx/state/deferral_ledger.md` (D27b) ·
`tools/witness_observer_replay.py` (#384 owed-14 telemetry replay — read FIRST, not duplicated) ·
`tools/costate_digest.py` (existing SENSE sections: annulus, shadow, ncde, verdict_trend,
pose_conditioning_gate, duty-to-measure) · `tools/dashboard_curriculum_panel.py` +
`tools/test_dashboard_curriculum_panel.py` (SIBLING dashboard agent's uncommitted files — read-only,
NOT touched; overlap avoided) · trainer emit sites (verified by line:
`witness_stability_resolved` ~L11939, `loss_terms`/`_loss_terms_row` L3000-3043 + `LOSS_TERM_KEYS`
L2981, `confound_alarm` L8260 + kinds {adaptive_eps_INERT L9337, gnorm_hijack L9491, term_domination
L9524, spike_deadlock L9649, frozen_epoch L9754/L6992, closed_loop_frozen_stop L7193}, `verdict`
L6949/L9726 (EMA-graded, per-class fields, liveness stamp), `jacobian_basin` L6649,
`muon_finisher_switch` L8790, `pose_finish_armed` L6181 / `pose_finish_engage` L8886,
`tail_cycle_begin` L9292 / `tail_powerplay_stop` L9300 / `tail_early_stop` L9966,
`seg_chroma_boundary` L5915, `event_curriculum_inert_under_unify` startup row) ·
`src/tac/witness_control/event_wirings.py` (`start_event_fired` L191, `cap_fired_before_event` L207,
`lane_band_would_fire` L387) · `src/tac/witness_control/sigma_min_plateau.py`
(`pose_finish_conditioning_gate` L564, disengaged/backstop-override alarms L585/L627) · sealed
`launch.sh` (read-only; the flag set actually flying) · dry-start `run.log` (read-only; real-row smoke).

## Gap table (a–h)

Verdict-scope note: every INERT/BINDING verdict the new analyzers emit is **INSTANCE-scoped**
(this run, this config) — never a formulation/family kill.

| Item | EXISTS already (emit site) | GAP | Binding-vs-inert proof | Disposition | Owner |
|---|---|---|---|---|---|
| (a) event-gated curriculum decision provenance | `start_event_fired` + `cap_fired_before_event` rows carry sensor, cap, fired_by, `sensor_data_epoch`, `sensor_lag_epochs` (event_wirings.py L191/L207); `lane_band_would_fire` per verdict (implied ON by `--lane-band-start-event`, L6093); `handoff_readiness` per verdict (implied ON by `--curriculum-nucleus-guard`, L6094); `muon_finisher_switch`, `tau_advance_armed`, `pose_finish_armed/engage`, engage rows per mechanism, `resume_event_curriculum`; the `event_curriculum_inert_under_unify` LOUD row already flags that `--curriculum-event-triggered` is INERT under `--seg-form-unify-tau` in the sealed config (tau-advance-mode event is the live mechanism — NOT a gap, a documented inertness) | rows scattered across run.log; no single post-hoc QUERY surface; no per-verdict "held" (would-fire) rows for the powerlaw_meat/annulus_plateau sensors (their underlying series ARE emitted: verdict d_seg + `annulus_convergence` default-ON) | **LANDED**: `event_decision_table()` normalizes every decision row into one epoch-sorted queryable table (`tools/witness_telemetry_audit.py --section events`) | land-now DONE + low-priority queued (Q5) | this task / main agent |
| (b) amber stability stack | `witness_stability_resolved` startup row stamps effective grad_clip 0.5 / pose_grad_coeff_max 25 / per_group_grad_clip / pose_eps resolution; `loss_terms.gnorm` per row; `gnorm_hijack` alarm (L9491) | **NO clip-activation rates**: per-group pre-clip norms computed (L9444-9451) but never counted/emitted — cannot prove amber BINDING vs INERT per group (the `--grad-clip 1.0` defeat class) | **LANDED (global)**: `amber_binding()` = fraction of loss_terms rows with gnorm > effective clip → BINDING / INERT_NEVER_BINDS / SATURATED_ALWAYS_CLIPS. **QUEUED (per-group)**: Q1 below | land-now DONE (global) + Q1 queued | main agent (trainer patch at resume boundary) |
| (c) chroma rung | `seg_chroma_boundary` engage rows (L5915 + `seg_chroma_boundary_engage`); `chroma_boundary` IS in `LOSS_TERM_KEYS` (L2985) so its per-row value is already emitted; `annulus_convergence` chroma-relevant annulus series default-ON | in-run `term_domination` alarm's `_reg_keys` (L9517) = ("eikonal","length","eik_steik","boundary_distance") — chroma NOT covered; NO inert-floor alarm (engaged-but-~0 = orphaned lever) | **LANDED**: `chroma_binding()` = post-engage share of loss total → PENDING / INERT_ZERO / BINDING / DOMINATING(>40%) | land-now DONE + Q2 queued (in-run alarm parity) | main agent |
| (d) pose-finisher detector | `jacobian_basin` sensor rows default-ON (L6649: median_sigma_min, plateau_est, would_have_fired); `pose_finish_conditioning_gate` observer row per T1 (sigma_min_plateau.py L564); armed/engage rows; 4 confound alarms incl. disengaged-shipped-banked-R1; #384 owed-14 replay tool (`tools/witness_observer_replay.py`) proves fire/hold on real rows — NOT duplicated | the crash class was SILENT: a detector that dies stops emitting and nothing notices (2.5 h) | **LANDED**: `pose_gate_health()` instrument-liveness — sensor cadence vs verdict cadence → DETECTOR_STALLED when the sensor stops while verdicts advance (grace for early runs; DISABLED distinct). This is the class fix at the READ layer; the trainer's own emit path is already fail-open try/except (L6674) | land-now DONE | this task |
| (e) EMA-lag | verdict rows are EMA-graded (blob from `ema_np`); `loss_terms.pose` is live; run-1 confound CONFIRMED (verdict d_pose rises while training pose falls) | no explicit gap series; no live-weights d_pose at verdict cadence | **LANDED**: `ema_lag()` = trailing-window verdict-d_pose trend vs live pose-term trend → EMA_LAG_DIVERGING on the run-1 signature. **QUEUED**: Q3 (true live-weights d_pose every Kth verdict) | land-now DONE + Q3 queued | main agent |
| (f) solve-upon-basin readiness (D27b) | `muon_finisher_switch`; verdict d_seg series; `tail_powerplay_stop.net_marginal_s_per_ep` (MEASURED marginal); `polyak_finisher_armed`; jacobian_basin plateau est | D27b trigger ("run reaches basin/terminal band") had NO machine-readable signal | **LANDED + DEFINED**: `terminal_band_status()` — `in_basin` = muon fired AND trailing-window d_seg rel-slope < 5e-3; `terminal_band` = in_basin AND (TAIL stop OR Polyak armed); **`d27b_ready` = in_basin** → fire the D27b solve stack on the NEXT checkpoint/byte-close, never mid-run. Surfaced in the costate digest every SessionStart | land-now DONE | this task; costate controller consumes |
| (g) v8 per-class carriers | verdict rows ALREADY carry `d_seg_by_class` + `flip_share_by_class` (L6966); island_seed/island_amplify/ladder_rung rows; birth_completion rows; per-class costate λ_c capture (L6970) | v8 not flying — carrier-level attribution undefined | **PLAN (design only, lands with v8 build wave #377)**: each v8 carrier emits `{stage:"carrier_attribution", epoch, carrier_id, class_id, d_seg_class, flip_share_class, bytes_est, active}` at verdict cadence (score-neutral read of the per-class verdict it already computes) + per-class island-birth events reuse the existing island/birth rows keyed by carrier_id. Binding proof = per-carrier Δd_seg_class series vs carrier-off anchors. `telemetry_binding.py` extends with `carrier_attribution_table()` when rows exist | queued-on-v8 | v8 build wave owner |
| (h) TAIL cycle endpoints | `tail_cycle_begin` (epoch, cycle, tau, lr, reason + stage-ckpt info), `tail_powerplay_stop` (measured marginal), `tail_early_stop` (best d_seg/epoch); per-cycle stage checkpoints PRESERVED (SWA-soup inputs are the ckpts themselves) | no per-cycle endpoint STATS row | **LANDED**: `tail_cycle_endpoints()` joins cycle boundaries to nearest preceding verdicts → per-segment endpoint {d_seg, d_pose, implied_S, best-in-segment}. Q4 (explicit trainer endpoint row) is OPTIONAL — the join already answers the SWA-soup question | land-now DONE; Q4 optional | this task |

## What LANDED now (read-only, score-neutral, default-ON)

1. **`src/tac/witness_control/telemetry_binding.py`** — unit-tested analyzers (a,b,c,d,e,f,h) over
   the rows the trainer already emits; every section fail-open; schemas verified against emit sites.
2. **`tools/witness_telemetry_audit.py`** — thin CLI: `--run-dir <run> [--json] [--section X]`.
   Exercised against the REAL sealed dry-start log (read-only):
   `telem-binding: amber=UNKNOWN chroma=PENDING pose-gate=NO_SENSOR_ROWS ema-lag=UNKNOWN basin=not-yet
   events=6` (correct for a 1-verdict dry start).
3. **`tools/costate_digest.py` + `section_telemetry_binding`** — one digest line per SessionStart
   (amber/chroma/pose-gate/ema-lag verdicts + D27b basin state), bounded 1.5 MB tail read, fail-open.
   The costate controller now SEES binding-vs-inert + basin readiness without operator memory.
4. **Tests**: `src/tac/witness_control/tests/test_telemetry_binding.py` (31) +
   `src/tac/tests/test_costate_digest_telemetry_binding.py` (5); all green; ruff F clean.

## QUEUED trainer patches (gated on relaunch/resume boundary — DO NOT land hot)

All flag names verified against the trainer argparse (never-invent-flags). Each is score-neutral
observability → per the off-is-a-tracked-queue rule these default ON when they land (they read
already-materialized values; zero new compute except Q3).

- **Q1 (b) per-group clip-activation row.** At the per-group clip site (trainer ~L9444-9451, the
  `--per-group-grad-clip` branch): capture each group's pre-clip norm (returned by
  `optim.clip_grad_norm` per group), accumulate per-epoch counters, emit ONE
  `{stage:"grad_clip_activation", ep, global:{n,frac_clipped,norm_max}, per_group:{<g>:{n,frac_clipped,norm_mean,norm_max}}}`
  row per epoch adjacent to the loss_terms emit. No new flag (pure read of computed values).
  `telemetry_binding.amber_binding` then flips `per_group_rates_available=True` and per-group verdicts.
- **Q2 (c) chroma alarm parity.** Add `"chroma_boundary"` (+ the other stacked lever terms:
  `margin_saliency`, `temporal_screw`, `island_amplify`, `persistence`) to the `term_domination`
  `_reg_keys` (~L9517) AND add the inverse `term_inert` confound alarm: a lever ENGAGED (its engage
  row fired) whose share < 1e-6 for N sustained rows → LOUD. Same `_emit_confound_alarm` helper.
- **Q3 (e) live-weights verdict gap.** New flag `--verdict-live-gap-every K` (default 0 = off,
  REGISTERED lever with duty-to-measure per the activation-ledger rule — it costs one extra advisory
  inference): every Kth verdict also scores the LIVE weights and stamps `d_pose_live`/`d_seg_live`
  on the verdict row → the EMA-lag series becomes exact instead of trend-inferred.
- **Q4 (h, optional) explicit `tail_cycle_endpoint` row** at each cycle boundary (the post-hoc join
  already answers this; land only if the SWA-soup evaluation wants trainer-stamped rows).
- **Q5 (a, low) would-fire rows for powerlaw_meat + annulus_plateau sensors** per verdict under event
  mode (mirror of `lane_band_would_fire`), so held-decisions carry metric-vs-threshold explicitly.
  Underlying series already emitted; this is queryability sugar.

## The three gaps ROUTED to #404 by the dashboard sibling (DAG FEED-dashpanel, in-flight)

- **(routed-a) `ladder_birth_complete` row** — `ladder_rung` is a per-refresh progress row (fires
  ~ep1), not a birth-boundary event. QUEUED as **Q6**: emit a discrete
  `{stage:"ladder_birth_complete", epoch, class_id, r_final}` row when a ladder class's birth
  anneal completes (trainer patch, resume-gated; read of state the ladder already tracks).
- **(routed-b) `should_ship_banked_r1` emission** — **ALREADY CLOSED at the emit layer**: every
  `pose_finish_conditioning_gate` observer row carries `should_ship_banked_r1`
  (`src/tac/witness_control/sigma_min_plateau.py` L573, emitted per T1 at trainer L6670-6673). The
  panel's "—" was honest only because the live run had not yet emitted its first gate row. No patch
  needed; verify on first gate row of the relaunch.
- **(routed-c) uniform `{lever}_engage {status}` schema** — QUEUED as **Q7**: adopt one
  `{stage:"lever_engage", lever:<name>, status:"armed|fired|complete", epoch, via}` companion row
  emitted alongside each existing per-lever engage row (additive; existing stage names kept for
  back-compat). Removes per-lever stage-name coupling for every reader (dashboard, digest, analyzer).

## Queued for the DASHBOARD agent (owns tools/dashboard_server.py — NOT touched here)

- Render the `telemetry_binding` audit (the digest section's dict) as a binding-vs-inert panel row
  per lever + the D27b `d27b_ready` badge. Source: `tac.witness_control.telemetry_binding.audit_rows`
  (import, don't re-parse). The sibling's `dashboard_curriculum_panel.py` mechanism-lane states and
  this module's event table are complementary (states vs decisions) — no overlap.

## Follow-up tasks for the main agent

1. Land Q1+Q2+Q6+Q7 (+Q3 flag) at the relaunch/resume or next stage-boundary resume; wire Q3's flag
   as a DSL `Lever` factory (triality: lever→DSL same-commit).
2. Fold the (g) carrier-attribution telemetry design into the v8 build wave (#377) acceptance
   criteria (SPEC_v8 addendum).
3. D27b consumers: point the terminal-solve stack trigger check at
   `witness_telemetry_audit.py --section terminal_band --json` (`.d27b_ready`).
4. Dashboard agent task per the section above.

## Triality legs

- **DAG**: FEED-telemetry-p0 appended (sub015 DAG).
- **DSL**: N/A-with-rationale — this landing adds ZERO config levers (read-only analyzers + digest
  read; the `[no-triality-lever]` apparatus class). Q3's flag WILL be a DSL Lever when queued work lands.
- **equations**: N/A-with-rationale — no new measured law; the analyzers operationalize existing
  registered confound classes (adaptive_eps_INERT / term_domination / frozen-run / EMA-lag run-1
  anchor). A future measured EMA-lag magnitude row would mint an EmpiricalAnchor then.
