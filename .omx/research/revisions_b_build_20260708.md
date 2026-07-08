# REVISIONS-B build — T3 v7 council CONSOLIDATED REVISIONS block B (2026-07-08) [no-triality]

STORES CONSULTED: `SYNTHESIS_T3_v7_council_20260708.md` (§CONSOLIDATED REVISIONS B) ·
`position_V7_S1_shannon_rate_20260708.md` (R1 rate-aware TAIL stop) ·
`position_V7_S2_dykstra_continuation_20260708.md` (REV-A stagger) ·
`position_V7_S4_rudin_contracts_20260708.md` (R2 provenance + marginal stamp) ·
CODE: `witness_control/tail_cycles.py` · `witness_curriculum/ladder_homotopy.py` ·
`witness_dsl/{curriculum_dsl,lawref,typed_config}.py` · `witness_autoconfig.py` (crucible_v7) ·
`canonical_equations/evaluators.py` (tail_cycle_floor_v1/settle_window_v1 LawRef pattern) ·
`experiments/train_levelset_witness_realized_through_R_mlx.py` (config-validation + TAIL wire-in +
seg_form dispatch). Pointer 0.19110 [contest-CPU] UNMOVED — everything here is MEANS.

## Per-item disposition

### Item 1 — S2-REV-A LADDER↔Muon stagger invariant (LANDED, two surfaces + shared helper)
- **Shared pure helper** `tac.witness_curriculum.ladder_homotopy.ladder_muon_stagger_violation(*,
  ladder_on, lane_window, movable_window, muon_start_epoch)` + `ladder_arm_window(birth,hold,anneal)`
  — the single source of truth. Invariant: `max(arm windows) < muon_start`; returns an error STRING
  naming the violating window(s), or None (LADDER off / no Muon finisher). The `muon_start` is the
  fixed backstop CAP; on the event-armed domain the Muon event is additionally REV-B-gated on
  nucleation-complete, so the cap is the sound ceiling on BOTH domains (documented in the helper).
- **DSL surface**: `WitnessProgram.validate` consumes the helper (reads `--ladder-*` window flags +
  `--muon-start-epoch` from the flag dict) → a future config lengthening an anneal past Muon is
  refused at derive time. Verified: v7 clean (max window 340 < 726); a 780-window config fires.
- **Trainer surface**: `validate_ladder_muon_stagger_config(...)` (pure) consumes the SAME helper,
  called in the config-validation block (after the Muon guard). Raises LOUD before any GPU spend.
- **DSL docstrings**: `LadderIslandHomotopy` notes the stagger + the λ-gate provenance.

### Item 2 — S4-R2 + S1-R1 TAIL upgrades (apparatus LANDED; trainer wire-in in working tree)
- **(a) provenance (no silent literals)** — `TAIL_CONSTANT_PROVENANCE` (tail_cycles.py) machine-
  readable rows: `cycle_floor_epochs`→`tail_cycle_floor_v1` + `dwell_min`→`settle_window_v1`
  (DERIVED-AT-CONFIG, following the existing sibling LawRef-citation pattern); `tau_halving`(0.5) +
  `stop_marginal_s`(1e-4) → **HARDCODED-WITH-WAIVER** with real rationale (SGDR octave base / the
  PowerPlay attribution floor — no closed-form derivation, the honest req-T class-4). λ-gates →
  `LADDER_LAMBDA_GATE_PROVENANCE` (DERIVED-AT-CONFIG, OPEN self-documenting). Surfaced additively in
  `CrucibleV7Compiled.tail_constant_provenance` (argv byte-identical — values unchanged).
- **(b) rate-aware TAIL stop** — `TailController.step(..., byte_rows=None)`. `_cycle_net_marginal`
  = d_seg leg `100·(Δd_seg)/len` MINUS the rate leg `25·(Δbytes)/37_545_489/len` (S = 100·d_seg +
  √(10·d_pose) + 25·bytes/B). A cycle that lowers d_seg while INFLATING the counted blob has its net
  marginal cut by the rate cost → cannot read as a pure win. `byte_rows=None` ⇒ d_seg-only (BYTE-
  IDENTICAL to the pre-S1-R1 stop; existing 13 tail tests unchanged). Trainer threads `blob_bytes`
  (already computed per verdict) into `history` + into `_tail_ctrl.step(byte_rows=...)`.
- **(c) marginal numerator stamp** — `TailStep.marginal` (default NaN) carries the MEASURED net-ΔS/ep
  value at the stop (not just the `<threshold` outcome), also stamped in `reason` + the trainer's
  `tail_powerplay_stop` telemetry row (`net_marginal_s_per_ep` + `rate_aware`). Auditable post-hoc.

### Item 3 — S6-R5 event-triggered-curriculum vs dissolved CE→tau under unify (AUDITED: clean; guard + loud note)
- **Finding**: with `--seg-form-unify-tau` ON the per-epoch seg_form dispatch takes `if _unify_tau_on:`
  BEFORE `elif _evt_on:`, so `_evt_resolve_seg_form` (the thing that fires the discrete CE→tau→l7
  boundaries) is NEVER called — the event-triggered controller **cannot fire a dissolved boundary**.
  The correctness is CLEAN (no wrong-fire; the short-circuit is pre-existing + documented at the
  dispatch site). **But** v7 co-emits BOTH `--seg-form-unify-tau` AND `--curriculum-event-triggered`
  (MEASURED: both in the emitted argv), so the machinery is INERT-BUT-ARMED — a reader could mis-read
  the event flag as active.
- **Fix**: a LOUD config-validation note (`event_curriculum_inert_under_unify`) surfacing the inert
  state (orphaned-signal rule: "off"/inert must be a tracked, surfaced state). Do NOT hard-fail
  (benign) and do NOT remove the flag (charter; keeps the seal byte-stream + resume sidecar stable).
  Muon entry now has its OWN EventBackstopGate wiring; the curriculum controller's only job was the
  dissolved CE→tau→l7 boundaries.

## v7.2 diff-vs-v7 (the seal target after this landing)
- **Emitted argv: BYTE-IDENTICAL** to v7 — every sealed value (`--tail-stop-marginal-s 0.0001`,
  `--tail-tau-halving 0.5`, `--ladder-*-lambda-gate 0.0`, ...) is unchanged; this landing adds
  PROVENANCE (a manifest field) + a validation entry (stagger) + a runtime rate-leg on the TAIL stop
  + a loud R5 note. `derive_crucible_v7_config` still validates clean (stagger holds: 340 < 726).
- **New governance/validation**: the DSL `WitnessProgram.validate` now enforces the stagger invariant;
  `CrucibleV7Compiled` gains `tail_constant_provenance` (6 rows: 4 tail constants + 2 λ-gates).

## Landing state (co-edited-tree honesty)
- **COMMITTED `3563b9c9b`** (serializer, 2 review passes, post-commit `git show` verified — my blobs
  match intent, sibling additions are pure-additive with zero deletions to my lines): the apparatus
  (ladder_homotopy, tail_cycles, curriculum_dsl, witness_autoconfig) + tests (test_tail_cycles rate-
  aware additions + `test_revisions_b_stagger_tail_rate_provenance.py`).
- **WORKING TREE (uncommitted)**: the trainer surface (validate_ladder_muon_stagger_config def+call,
  byte_rows wiring, marginal print, blob_bytes-in-history ×3, R5 note). The trainer is CO-EDITED with
  the S6-R4 τ-advance sibling (their `--tau-advance-mode` hunks + untracked `tau_advance.py`; one
  adjacent-but-separable tail-step region). Per the anti-absorption discipline + the clobber warning,
  I did NOT commit the co-owned trainer (would absorb the sibling's uncommitted S6-R4). The trainer-
  dependent tests are `@_needs_trainer` skipif-guarded so the committed test file is GREEN whether or
  not the trainer commit has landed (RUN+assert when the surface is present, SKIP otherwise). The
  trainer commit is sequenced with the sibling by the coordinator.
- **Known transient (NOT my regression)**: `test_crucible_v7_config::test_governance_sensors_are_
  recognised_and_co_emitted` fails in the current WORKING TREE because a concurrent S6-R4 sibling
  added `--tau-advance-mode`/`--tau-octave-max-dwell` governance to `_crucible_v7_schedule_governance`
  mid-build (27 uncommitted insertions to witness_autoconfig AFTER my commit) without the gate's
  RECOGNISED_EVENT_SENSORS co-update. My COMMITTED witness_autoconfig has exactly the 6 governance
  keys (no tau-advance) and passed 22/22 before the sibling's edit; resolves when they finish.

## Tests: 42 mine, all green
- `test_tail_cycles.py`: 13 pre-existing (byte-identical d_seg-only path preserved) + 4 rate-aware
  (stable-bytes-no-stop, bytes-inflating-stop, marginal-stamp, byte_rows-None-byte-identical).
- `test_revisions_b_stagger_tail_rate_provenance.py`: 25 (stagger helper pos/neg/both/boundary/off/
  no-muon + DSL validate + trainer validator [guarded] + provenance rows/copy/waiver + crucible
  manifest + argv-unchanged + R5 co-emit/short-circuit/loud-note [guarded]).

means != ends: a MEANS. Only a byte-closed n600 exact row < 0.19110 from `upstream/evaluate.py`
(contest-CPU/CUDA, NEVER MPS) moves the pointer 0.19110.
