# R-7 FINISHERS — β2-window LR rewarmup + Polyak tail-averaging finisher  [landed]

- **UTC:** 20260708 · **Agent:** R-7 FINISHERS (Opus) · **Authority:** `[macOS advisory]` $0, NO launch, live run + pid 63069 untouched. Pointer contest-CPU **0.19110 UNMOVED** — this is APPARATUS/MEANS, not a score move.
- **Source:** T5 crucible seal-round-1 structure lens R-7 ("minor unbuilt finishers → v7.1 ledger"). Operator: "build all unbuilt items and wire and integrate and DSL and triality."
- **Commits:** `3d44fd51c` (DSL + mechanism + 16 tests) · `7790261f6` (trainer wiring, via `--patch-file` HEAD-seeded, `git show --stat` = 1 file 94+/1−, zero foreign absorption).

## STORES CONSULTED
- `.omx/research/t5_crucible/seal_v7_r1_structure_blind_20260708.md` (P1.6 cold-Muon/rewarmup law; P1.7 finisher-EMA=Polyak).
- `.omx/research/t5_crucible/SYNTHESIS_seal_v7_round1_20260708.md` (R-7 disposition → v7.1 ledger).
- `.omx/research/tail_majors_fix_20260708.md` (FIX-2: `τ_0=τ_end=0.31` ⇒ constant-τ* TURNPIKE dwell — the operating point the Polyak mean exploits).
- Code read: trainer `experiments/train_levelset_witness_realized_through_R_mlx.py` (`_stage_rewarmup_factor`, `last_boundary_epoch` main-loop set, `_do_checkpoint`, `_load_resume_state`, `_build_ema_checkpoint_arrays`, EMA update); `src/tac/witness_control/resume_registry.py` (landed sibling — `Resumable` protocol + `ResumeRegistry`); `src/tac/canonical_equations/curriculum_derivation_laws_20260705.py` (`min_rewarmup_epochs` + `rewarmup_beta2_memory_window_v1`); `margin_saliency_reachability_and_muon_finisher_20260703.py` (`muon_finisher_schedule_warmstart_and_lr_anneal_v1`); `src/tac/witness_control/tail_cycles.py` (`stop_marginal_s_lawref` DERIVED-AT-CONFIG pattern); `src/tac/witness_dsl/{curriculum_dsl,lever_registry}.py` (Lever factory + completeness).

## ARCHAEOLOGY VERDICT — finisher 1 (β2-window rewarmup) is mostly ALREADY BUILT
Grep-verified the trainer already carries the FULL mechanism:
- The re-warmup **ramp** `_stage_rewarmup_factor(ep, last_boundary_epoch, rewarmup_epochs, floor, shape)` (default-OFF at `rewarmup_epochs<=0` → returns exactly 1.0 → bit-identical).
- The **event-fired boundary**: `last_boundary_epoch = ep` set in the MAIN loop at each stage transition (line ~7490), not just on resume — so the ramp already fires on event-fired transitions.
- The three flags `--stage-transition-rewarmup-{epochs,floor,shape}` (+ `--stage-transition-reset-moments`), ALREADY DSL-**mapped** (referenced by the `PowerPlayReheat` program + the gauge helper).
- The **sizing LAW** `rewarmup_beta2_memory_window_v1` + its callable `min_rewarmup_epochs(beta2, steps_per_epoch)` = `ceil((1/(1-β2))/steps_per_epoch)`.

**⇒ What was genuinely UNBUILT: a composable `Lever` that turns the previously-HARDCODED `rewarmup_epochs` literal into a DERIVED-AT-CONFIG value.** So finisher 1 is a **DSL-only** build (NO trainer edit): `Beta2WindowRewarmup(beta2, steps_per_epoch, floor, shape)` computes the window via the existing law callable and emits the existing (mapped) flags. This is the honest scope — I did NOT re-build a mechanism that exists.

## SIZING-RULE DERIVATION (finisher 1) — DERIVED-AT-CONFIG, no bare literal
`window_epochs = ceil( (1/(1−β2)) / steps_per_epoch )` per `rewarmup_beta2_memory_window_v1` (`latex: T_rw·S_ep ≥ 1/(1−β2)`). With reset-moments ON the AdamW 2nd-moment `v` re-accumulates over `1/(1−β2)` steps; a full-LR step inside that window divides the gradient by an unconverged `v`. At β2=0.999, 75 steps/ep ⇒ `ceil(1000/75)=14` ep (the lever's derived value; tested to be the TIGHT ceil — `win−1` does NOT cover). `floor=0.1` + `shape=cosine` are the LAW's declared **PROVISIONAL** profile ("floor 0.1 underived; bound-satisfying guess"), NOT a new invention. Status of the law itself: **INFERRED_FROM_DOMAIN_LITERATURE / PROVISIONAL** — an isolated β2-sweep A/B would promote it to VERIFIED (not this build's job).

## FINISHER 2 (Polyak) — what it IS mechanically
`PolyakTailAverager` (new module `src/tac/witness_control/polyak_finisher.py`): a **uniform (Polyak/Ruppert) tail average of the LIVE iterates** over the finishing window. `observe(ep, live_weights)` folds each end-of-epoch weight set into an incremental exact uniform mean (Welford, fp64) once `ep≥start ∧ arm`. Exported every checkpoint as `levelset_witness_polyak_mlx.npz` (+ per-stage-encoded `levelset_polyak_<tag>_ep<N>.npz`) via the SAME `_build_ema_checkpoint_arrays` + on-manifold `_project_shadow_film_np` the EMA deploy npz uses — an **ADDITIONAL candidate**. It **NEVER replaces the EMA shadow** (the EMA non-negotiable stands); the byte-close/eval stop-time checklist MEASURES d_seg/d_pose/rate and picks the better candidate — that measurement is NOT this lever's job.

**Why a tail mean beats a short-horizon EMA (the math, and it's TESTED):** at the sealed constant-τ* TURNPIKE the iterates ORBIT a basin center. A fixed-horizon EMA weights recent (orbiting) iterates and still carries orbit phase; the uniform mean over the window averages the orbit out to O(1/√n) — the strictly better basin-CENTER estimate (Polyak-Ruppert). `test_polyak_tail_mean_beats_short_horizon_ema_on_orbit`: over 10 exact periods the uniform mean = center to 1e-9, the decay-0.9 EMA is >0.05 off-center (>10× the Polyak error).

## RESUME-STATE HANDLING — registry (sibling landed) + heavy sidecar, one atomic write
The resume registry sibling HAS landed. I used it:
- **Scalar sentinel** (count/start/arm) rides the `ResumeRegistry` via the `Resumable` protocol under prefix `__pta_` → the registry's completeness manifest protects it (LOUD-warn if it vanishes). The averager exposes **no** `event_mode` ⇒ registry never applies its event fail-closed to it.
- **Heavy running-mean** rides the resume sidecar under a NEW prefix `_RESUME_POLYAK_PREFIX = "polyakM__"`, routed in `_load_resume_state` to `rs["polyak"]` (ISOLATED from the model live-param restore — verified), merged into the SAME atomic `_atomic_savez` as the scalars ⇒ no cross-file desync.
- **FAIL-OPEN** (not closed): a sentinel-present-but-heavy-missing mismatch is LOUD-but-non-fatal — the EMA shadow (resume-critical) is untouched; the Polyak candidate simply restarts. Correct because Polyak is a bonus candidate, not the resume-critical artifact.
Resume round-trip is bit-faithful (continuous-run reference == restore+continue, tested + verified against the REAL trainer `_load_resume_state`/`_atomic_savez` in `test_trainer_load_resume_state_routes_polyak_and_isolates_live`).

## TRIALITY / DSL LEG (same landing)
Two `Lever` factories in `tac.witness_dsl.curriculum_dsl` — `Beta2WindowRewarmup` (reuses the mapped rewarmup flags) + `PolyakFinisher` (new `--polyak-finisher-arm`/`--polyak-finisher-start-epoch`). Both nilary ⇒ `--dsl-lever`-composable. `completeness()`: both new flags **MAPPED, 0 unmapped for the new flags, no stale**. Equations leg: I **consume** existing laws (`rewarmup_beta2_memory_window_v1`, `muon_finisher_schedule_warmstart_and_lr_anneal_v1`) + a DERIVED-AT-CONFIG provenance helper `polyak_finisher_window_provenance(stage_window, frac)` — NO new bare literal, NO new equation needed (a BUILD consuming laws, not a new measured finding). Value-provenance ladder honored.

## DEFAULT-OFF BYTE-IDENTITY — how verified
Both levers default-OFF. Structural guarantee: `--polyak-finisher-arm` absent ⇒ `_polyak = None` ⇒ every observe/heavy-merge/export/restore is guarded off ⇒ ZERO new checkpoint keys; the registry emits `{}` (no manifest) when the sentinel isn't registered; `_load_resume_state` never sees `polyakM__` keys. **CODE-CORRECTNESS check (NOT a training-benefit claim):** exercised the REAL trainer helpers (`_RESUME_POLYAK_PREFIX`, `_atomic_savez`, `_load_resume_state`) in a GPU-free round-trip — OFF path yields `rs["polyak"]=={}` (byte-identical observable); ARMED path routes the heavy mean to `rs["polyak"]` + scalar count to cfg, with live-param routing isolated + mean/count bit-faithful. No gt_n6 CPU smoke run (live run + memory contention; the None-guard makes byte-identity structural — a smoke would add no signal a training benefit could not, and I make no training-benefit claim).

## INTERACTION NOTES
- **TAIL turnpike:** the Polyak window should span the constant-τ* dwell (`tail_majors_fix` FIX-2: `τ_0=τ_end=0.31`). `polyak_finisher_window_provenance` sizes `start_epoch` from the stage window (~0.1–0.3× per the finisher law). Averaging is DECOUPLED from the TAIL controller's PowerPlay/meat-exit stop — it only reads weights, never steers.
- **Muon stage:** observe reads end-of-epoch LIVE weights AFTER the batch loop (post spike-guard rollback, which is intra-batch) ⇒ clean iterates. Independent of the AdamW→Muon geometry switch; the Polyak mean simply averages whatever iterates the finisher produces.
- **β2 rewarmup pairs with `--stage-transition-reset-moments`** (the law's premise). The lever defaults `shape=cosine` (the law's PROVISIONAL profile), distinct from the live crucible run's `linear` — they are different opt-in runs, no conflict.

## RESIDUAL RISKS (NAMED)
1. `Beta2WindowRewarmup(steps_per_epoch=75)` default matches the n600 crucible; a DIFFERENT config's real steps/ep would size the window slightly off — the operator must pass the real value (the lever notes print the arithmetic; transparent, not a bare literal).
2. `PolyakFinisher(start_epoch=0)` default arms from run START — a whole-run average, NOT a tail. This is a documented footgun: a TRUE tail needs `start_epoch` at the finisher entry (helper `polyak_finisher_window_provenance` derives it). Default 0 is honest (arm-from-start) but the operator MUST size it for the intended basin-center behavior.
3. The β2-window law is INFERRED/PROVISIONAL (no isolated A/B); `floor=0.1`/`cosine` are the law's own underived profile.
4. NOT MEASURED at n600: whether the Polyak candidate actually byte-closes BELOW the EMA shadow is a stop-time-checklist duty-to-measure, not established here. This is APPARATUS; the pointer 0.19110 is UNMOVED.
