# Canonical Resumability Registry — the CLASS-fix for forgotten resume state (2026-07-08)

STORES CONSULTED (proactive recall before building):
- CLAUDE.md non-negotiables: "resumability + per-stage checkpoints MANDATORY", "deterministic
  reproducibility", "Bugs must be permanently fixed AND self-protected against", "Confound
  self-protection" (2-landing fix+gate discipline), "Subagent commits MUST use serializer".
- docs/operating_manual_craft_handoff.md (do LESS but REAL; point-fix != class-fix; attack your own
  conclusion; a fix is unreviewed new code).
- .omx/research/t5_crucible/ORCHESTRATION_LEDGER.md (crucible v7 arc; SEAL-v7-r1 MAJOR-1 = the muon
  point-fix that motivated this).
- src/tac/witness_control/event_wirings.py (the EventBackstopGate + muon_gate_* MAJOR-1 point-fix,
  commit 2b17e55f6), src/tac/witness_control/tau_advance.py (the `__ta_*` state_arrays/restore
  pattern), the trainer's `_build_resume_state_arrays` / `_load_resume_state` / `_cl_*` / `_rng_*`
  hand-written pairs.

## The bug class (why a point-fix was not enough)

The trainer's resume sidecar is an ACCRETION of hand-written `(state_arrays, restore_from_cfg)` pairs
(`_cl_*`, `tau_advance_*`, `muon_gate_*`, `_rng_*`). Adding a NEW stateful controller means remembering
to route it into `_build_resume_state_arrays` AND the resume restore — four separate hand-written pairs,
each an opportunity to forget the next one.

Empirical recurrence:
- SEAL-v7-r1 MAJOR-1: the muon event-gate's sensor-fire epoch was not persisted → a crash between the
  sensor fire and the backstop cap would restore a fresh AdamW against a Muon checkpoint. Point-fixed
  with `__mg_*` (commit 2b17e55f6). ONE gate fixed.
- Orchestrator verification 2026-07-08: the trainer instantiates THREE latching `EventBackstopGate`s
  (`_muon_gate` ~L5181, `_lane_band_gate` ~L5185, `_chroma_gate` ~L5189) but ONLY muon flowed into
  resume. On crash-resume a FIRED lane-band/chroma lever silently turned OFF until its sensor re-fired;
  the chroma annulus-plateau detector needs 4 dwell windows of fresh history → the chroma lever could
  stay dark for hundreds of epochs post-resume. A lever ON before the crash, OFF after = a config that
  never existed = resume-determinism broken.

## What was built (optimal form, not another point-fix)

1. **`src/tac/witness_control/resume_registry.py`** (new):
   - `Resumable` protocol: `state_arrays(prefix) -> dict` + `restore_from_cfg(prefix, cfg) -> bool`.
   - `ResumeRegistry`: register(name, prefix, controller) with dedupe on name+prefix + protocol check;
     `state_arrays()` merges all controllers and stamps a `__resume_registry_manifest` (ONLY when at
     least one controller wrote keys — else `{}`, byte-identical); `restore(cfg)` iterates the SAME set
     and runs completeness self-protection.
   - `TrailingSeriesResumable`: persists the bounded trailing window of a detector-history series (the
     chroma annulus-plateau history).
   - `GATE_KEY_PREFIXES = {muon: __mg_, lane_band: __lbg_, seg_chroma_boundary: __cbg_}` +
     `build_gate_resume_registry(gates, wire_sense=)` — a gate whose name has no canonical prefix
     RAISES (fail-closed at construction).
   - `ResumeIntegrityError` for the fail-closed refusal-to-silently-re-arm.

2. **`EventBackstopGate` (event_wirings.py)** gains built-in `state_arrays(prefix)` /
   `restore_from_cfg(prefix, cfg)` (generalizing the muon persistence to ANY gate). The
   `muon_gate_state_arrays` / `muon_gate_restore_from_cfg` functions are now THIN WRAPPERS delegating
   with the `__mg_` prefix — the exact same `__mg_fired_epoch` / `__mg_fired_by` keys any live sidecar
   already carries (backward compat).

3. **Trainer** (`experiments/train_levelset_witness_realized_through_R_mlx.py`), 4 minimal edits:
   - build `_resume_registry` once, right after the three gates + `_wire_sense` are constructed;
   - `_build_resume_state_arrays` takes `resume_registry=` (was `muon_gate=`) and calls
     `resume_registry.state_arrays()`;
   - the checkpoint call site passes `resume_registry=_resume_registry`;
   - the resume-restore routes through `_resume_registry.restore(resume_cfg)` (prints warnings LOUD;
     `_muon_fire_restored` read from the report — the muon finisher-resume logic is byte-identical),
     then SEEDS `band_gate["on"]` / `chroma_bnd_gate["on"]` from the restored gate fire state so the
     first post-resume epoch does not spuriously re-treat (a restored-ON gate whose lever dict still
     read OFF would trip a bogus `recent_losses.clear()` that wipes the restored spike-guard window).

## What is now persisted that wasn't (per gate)

| gate | before | after |
|---|---|---|
| muon | fired_epoch + fired_by (`__mg_*`) — MAJOR-1 | unchanged (via registry, same keys) |
| lane_band | NOTHING (bug) | fired_epoch + fired_by (`__lbg_*`) when event-mode |
| seg_chroma_boundary | NOTHING (bug) | fired_epoch + fired_by (`__cbg_*`) + trailing annulus detector window (`__cbh_*`) when event-mode |

## Unfired-gate history decision: PERSIST (not re-derive)

- **lane_band** sensor is MEMORYLESS (reads only the LATEST verdict's `lane_ev`), so an unfired gate
  needs only its fire state — nothing extra to persist.
- **chroma** annulus-plateau detector inspects only `series[-dwell_windows:]`, so persisting the
  bounded TRAILING window (`TrailingSeriesResumable`, cap 16) is EXACTLY sufficient for a bit-faithful
  re-fire of an unfired-mid-dwell plateau. Chosen PERSIST over re-derive-from-telemetry because it is
  SELF-CONTAINED (independent of the JSONL schema / verdict cadence) and matches the existing
  `__recent_losses` spike-guard bounded-window precedent.
- **muon** unfired keeps the sealed `-1` sentinel "re-arm from post-resume history" contract (unchanged).

## Completeness self-protection (the class gate)

- **Runtime (L1)**: `__resume_registry_manifest` records each persisted controller + its event flag; at
  resume, a manifest entry whose keys VANISHED → `ResumeIntegrityError` for an EVENT gate (fail-closed),
  a LOUD warning otherwise. Benign cases (legacy no-manifest sidecar, event gate added since the
  checkpoint, unfired sentinel, event-mode-OFF) warn or pass — never raise.
- **Static (L2)**: `test_every_trainer_ebgate_has_canonical_prefix_and_is_registered` scans the trainer
  source and asserts (a) every `_EBGate(name=X)` has X in `GATE_KEY_PREFIXES`, and (b) every gate
  closure-var is passed into the `build_gate_resume_registry([...])` call → a NEW gate cannot ship
  without a resume prefix AND without being routed through the registry.

**Preflight candidate note**: L2 is a plain test (sufficient for this landing). A future
`src/tac/preflight.py` gate `check_trainer_transition_gates_registered_for_resume` could lift the same
AST scan to STRICT-mode preflight (follow the existing `check_*` pattern; claim a catalog number then).
NOT claimed here.

## Legacy-compat proof (how verified)

The live run (pid 63069, v7) is CAP-ONLY (`--muon-start-epoch`, `--lane-band-start-epoch`,
`--seg-chroma-boundary-start-epoch`; NO `--*-start-event`) → all three gates event-mode OFF. Verified:
- `test_all_cap_only_registry_emits_nothing_byte_identical`: an all-cap-only registry emits `{}` with
  NO manifest → the sidecar is byte-identical; restore of an empty/legacy cfg → all-False, no warnings,
  no raise (the exact live-run resume path).
- `test_legacy_pre_registry_muon_sidecar_still_restores`: a pre-registry MAJOR-1 sidecar (bare
  `__mg_fired_epoch`/`__mg_fired_by`, no manifest) STILL restores the muon fire state → backward compat
  with every sealed event-muon sidecar (benign legacy warning only).
- 39 pre-existing `test_event_wirings.py` tests (incl. the MAJOR-1 muon round-trip) pass unchanged after
  the wrapper refactor.

## Tests (decisive named)

`src/tac/tests/test_resume_registry.py` (16): `test_crash_resume_all_gates_bit_identical_to_uninterrupted`
(THE proof — fresh gates restored via the registry are bit-identical to the uninterrupted continuation),
`test_chroma_detector_history_roundtrip_refires_identically`, `test_vanished_event_state_fails_closed`,
`test_every_trainer_ebgate_has_canonical_prefix_and_is_registered` (class gate),
`test_all_cap_only_registry_emits_nothing_byte_identical` (byte-identity),
`test_legacy_pre_registry_muon_sidecar_still_restores`. These are CODE-CORRECTNESS tests (bit-equality
of control-flow state) — NOT a score claim; no n600.

## Residual risk (named, not hand-waved)

- The registry covers the TransitionGate class (the recurrence). The OTHER controllers (`_cl_*`,
  `tau_advance_*`, `_rng_*`, `_evt_*`) remain on their own hand-written pairs — already persisted, but
  NOT yet routed through the registry, so the static class-gate does not cover them. Folding them into
  the `Resumable` protocol is a follow-up (heterogeneous signatures; deferred to avoid churning the hot
  file during the live run). A NEW controller of a NEW shape could still be forgotten; a NEW
  TransitionGate cannot.
- A half-written sidecar (one of a gate's two keys present) is not detected by the vanished-key check
  (which tests ANY prefix key). `np.savez` writes atomically so this is not observed; the restore
  fallback (`fired_by="event"`) is safe.

## Pointer

0.19110 UNMOVED. This is APPARATUS/means (resumability infrastructure), not a score-mover — only a
byte-closed n600 exact row < 0.19110 from `upstream/evaluate.py` moves the pointer.
