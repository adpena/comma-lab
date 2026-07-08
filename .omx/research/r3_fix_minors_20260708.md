# SEAL v7.4 ROUND-3 — MINOR fix landing (F-1 · F-2 · F-3 · DM-MINOR-1 · DM-MINOR-2)

Fixer for the 5 MINOR findings from seal round 3 (`seal_v74_r3_confound_bugs` + `seal_v74_r3_deepmath_structure`).
Fix-all policy; minimal diff (round 4 reviews only this delta). Pointer **0.19110 UNMOVED** — every item
is APPARATUS/MEANS; the END is a byte-closed n600 `upstream/evaluate.py` row < 0.19110 AFTER the run.

## STORES CONSULTED
- `seal_v74_r3_confound_bugs_20260708.md` + `seal_v74_r3_deepmath_structure_20260708.md` (the 5 findings, exact file:line).
- `SYNTHESIS_seal_v73_round2_20260708.md` (fix charter) + `LAUNCH_PACKAGE_v7_20260708.md` (watch-list surface).
- CLAUDE.md non-negotiables: NO-FAKE (tests-verify-behavior class 2) · Confound self-protection (L1/L2/L3) ·
  value-provenance ladder (no bare literals; DERIVED > MEASURED-anchor) · verdict-scope ladder ·
  measurement-first (re-derive from primary artifact) · `docs/operating_manual_craft_handoff.md` (§4 re-derive,
  §6 attack-your-own-conclusion: a fix is unreviewed new code).
- CODE read from source: trainer L7530-7605 (lane/chroma/muon engage gates), L2143-2165 (`_evt_reanchor_epoch`),
  L5631-5646 (`lane_ev`/`lane_ev_epoch` set), L6758-6763 (`_lever_epoch`) · `event_wirings.py` (`EventBackstopGate.update`,
  `sensor_lag_epochs`) · `witness_autoconfig.py` L2259-2305 (`_build_crucible_v7` base) · `curriculum_dsl.py`
  (`persistence_classes_for_basis_regime`) · `scorer_throughput_gate.py` (anchor).
- MEASURED myself: run-1 log `levelset_n600_crucible_v6_run1_20260708T095730Z/run.log` verdict `ts` timestamps
  (ep25=137.77, ep125=483.13 → r_ss(ep25→125)=3.4537); fresh `_build_crucible_v7` emit (band present /
  persistence 3 / hosc-beta-end 3.177 / budget 8.314); F-3 guard positive+negative smoke.

## PER-ITEM STATUS

### F-1 (confound, FIXED) — lane gate `sensor_lag_epochs` frame-mix
- **Frame chosen: BOTH LEVER epochs.** The lane gate FIRES on `_lever_epoch(ep)` and stores `_fired_epoch` in the
  lever (re-anchored) frame; `lane_ev_epoch` is the REAL verdict epoch. Fix: pass `sensor_data_epoch =
  _lever_epoch(lane_ev_epoch)` so both the gate's `ep` arg and its `sde` are in the lever frame.
- **Why this makes the attribution TRUE:** `_lever_epoch` is a pure additive shift (`ep + (hardcoded − fired)`),
  so `sensor_lag_epochs = _lever_epoch(ep) − _lever_epoch(sde) = ep − sde` — the shift CANCELS, yielding the true
  real-epoch verdict-cadence lag, identical in semantics to the muon/chroma gates (which fire on real `ep`). Also
  makes the persisted `_sensor_data_epoch` frame-consistent with the persisted `_fired_epoch` (both lever frame).
- **Score-neutral + additive:** telemetry-only (never read into training). Byte-identical when re-anchor OFF
  (`_lever_epoch` = identity → same value as before); `-1` "no reading yet" sentinel preserved.
- **LIVE, not latent:** v7 emits BOTH `--curriculum-reanchor-levers` AND `--lane-band-start-event`, so the
  cross-frame bug was real in the shipped config. Files: `train_levelset_witness…mlx.py:7591-7598` +
  a FRAME CONTRACT note added to `event_wirings.py::EventBackstopGate.update` docstring (documents "which frame").

### F-2 (hygiene, FIXED) — stale pre-fix dry-run artifact
- Verified the dir `experiments/results/levelset_n600_witness_20260708T173144Z/` was NEVER launched (only
  `constants_manifest.json` + `launch.sh`; no `run.log`, no checkpoints/`.npz`) and its `launch.sh:35` carried the
  BLOCKER `--hosc-beta-end 10.0`.
- Neutralized WITHOUT deleting evidence: `git mv launch.sh → launch.sh.STALE_PREFIX_BETA10_DO_NOT_LAUNCH` + added
  `STALE_DO_NOT_LAUNCH.md` explaining the pre-A1-fix emit. **NOTE:** this run dir is GITIGNORED, so the rename +
  note are LOCAL-ONLY (not in the commit) — the neutralization is on-disk (bytes preserved, script un-launchable).
- pid 63069's sacred dir is `levelset_n600_crucible_v6_run1_20260708T095730Z` (verified via `lsof` open run.log) —
  a DIFFERENT dir; untouched. No other run dir touched.

### F-3 (confound, FIXED) — A5 lane-regime / analytic-band coupling unenforced
- Fix (a): a compile-time STRUCTURAL assertion in `_build_crucible_v7` (`witness_autoconfig.py`, right at the
  regime→persistence-classes coupling site): when `_CRUCIBLE_V7_BASIS_REGIME == "lane_offloaded"` (lane dropped
  from learned recall), FAIL-LOUD if `--lane-render-band` is ABSENT from the emitted `base`. Placed at the coupling
  site → covers BOTH `derive_` and `compile_` paths. The levers never touch `--lane-render-band`, so `base` == argv.
- Fails LOUD at COMPILE (not silently at byte-close). Positive path unchanged (band IS present in the proven base).
- Tests (behavior, not constants): `test_f3_lane_offloaded_structurally_co_emits_the_analytic_band` (positive: emitted
  argv carries `--lane-render-band` under lane_offloaded) + `test_f3_guard_fails_loud_at_compile_if_band_absent`
  (negative: monkeypatch the v6 base to drop the band → `derive_crucible_v7_config` raises).

### DM-MINOR-1 (FIXED) — budget r_ss anchor window ep75→100 → ep25→125
- RE-DERIVED from the primary artifact (run-1 `run.log`): ep25=137.77 min, ep125=483.13 min → r_ss(ep25→125) =
  (483.13−137.77)/100 = **3.4537 min/ep** (the narrow ep75→100 window's 3.371 fell in a slow-adjacent-fast trough;
  per-interval slopes bounce 3.25–3.73). S(ep100) = 396.62 − 3.4537·100 = 51.25 → amortized(3000) = 3.4537 +
  51.25/3000 = **3.47**. `RUN1_MEASURED_MIN_PER_EP: 3.39 → 3.47`; budget 8.122 → **8.314 d**; refuse ceiling
  3.90 → 3.99. **Direction: strictly more conservative** (higher min/ep → tighter refuse ceiling + larger budget
  projection; never lets a too-slow run through). Updated: the constant + its provenance comment
  (`scorer_throughput_gate.py`) + the budget provenance comment (`witness_autoconfig.py`) + 3 tests pinning the old
  numbers (`test_crucible_v7_config` 3.39→3.47/8.3→8.5 · `test_wallclock_default_on_perfenv_guard` 7.0625→7.22917
  /8.12→8.31 · `test_v7_compute_exploitation` 3.39→3.47/7.3→7.4) + the LAUNCH_PACKAGE A2 fold line.

### DM-MINOR-2 (DISPOSITION recorded, NO code change) — surface-3 island-amplify [0,350] window
- The per-class-λ island-amplify (`LadderIslandHomotopy`, surface-3) self-gates on λ_lane FALLING, which needs the
  analytic band to composite (`start_epoch: 350`). Over epochs [0, 350] the homotopy grows lane islands under the
  lane_offloaded basis — the PRE-EXISTING round-2 M1 residual, BOUNDED and arguably correct (birth born-empty lane
  early, hand to the band at ep350, self-de-emphasize as λ_lane drops). Already watched via the registered
  `lane_carried` counter-arm. Recorded in the LAUNCH_PACKAGE watch-list (epochs [0,350], mechanism, counter-arm,
  jitter threshold) so the disposition is recorded-not-silent. verdict_scope: INSTANCE.

## SCOPE EXTENSION (coordinator directive, holistic-watch) — LAUNCH_PACKAGE watch-list = FULL FACET SET
Expanded the watch-list (run-1 ep125, canonical order Road0/Lane1/Undriv2/Movable3/MyCar4): Road **0.398** (blocker,
~2/3 composite) · per-class d_seg spectrum (Lane 0.039 vs band-anchor 0.00087 ~45× · Undriv 0.074 · Movable 0.0069 ·
MyCar 0.0028) · d_pose **1.90** ⇒ √(10·1.90)≈4.35 of implied_S 17.4 (POSE ALONE blocks sub-0.19; watch verdict +
byte-close 3-arm) · island-birth part_frac (POSITIVE in run-1, must not regress) · rate blob_bytes ~89–90 KB · plus
the existing jitter / would-fire / spike-guard rows + the DM-MINOR-2 window. Each = signal + threshold + response
pointer; verdict_scope note added (all INSTANCE/FORMULATION watch signals, not family kills); holistic-coupling note.

## VERIFICATION
- `ruff --select F` CLEAN on all 8 touched files; trainer `py_compile` OK.
- Affected suites GREEN: `test_crucible_v7_config` (51, incl. 2 new F-3) · `test_wallclock_default_on_perfenv_guard`
  · `test_v7_compute_exploitation` · `test_event_wirings` · `test_resume_registry` · `test_levelset_weights_arm_selection`.
- Fresh emit MEASURED: `--lane-render-band` present · `--persistence-classes 3` · `--hosc-beta-end 3.177` ·
  budget 8.314 d.

Pointer **0.19110 UNMOVED** — this landing is APPARATUS/MEANS.
