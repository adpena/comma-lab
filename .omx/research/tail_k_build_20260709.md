# TAIL_k warm-restart refinement — BUILD LANDED (2026-07-09) [no-triality]

The post-Muon TAIL_k warm-restart stage (crucible **req L**; DRAFT_OPTIMAL_STACK_v6 §2.2e /
v5 §2.2e; §row-9 τ_k law) was DESIGNED + budget-verified but the trainer machinery was UNBUILT.
This build lands it, default-OFF (byte-identical when unset), wired so **run-1 can ADOPT it at a
stage boundary via the resume path WITHOUT touching the live process**.

## STORES CONSULTED
- CLAUDE.md (triality DSL-Lever discipline · never-invent-flags · review-gate · serializer+post-edit-sha ·
  resumability/per-stage-checkpoint non-negotiables · NO launches · anti-absorption).
- `.omx/research/t5_crucible/ORCHESTRATION_LEDGER.md` (req L/T/B/D/Q; the ν-amendment + v6.2 budget folds).
- `.omx/research/t5_crucible/DRAFT_OPTIMAL_STACK_v6_20260708.md` §2.2g/§row-9 + §2 TAIL budget lines;
  `..._v5_...md` §2.2e/§2.2f (the inherited spec body).
- `src/tac/witness_control/{trace_probes,powerlaw_exit}.py` (the ν-refit + per-cycle `powerlaw_meat_exit`
  the cycle exits consume — REUSED, not reimplemented).
- `tools/witness_tau_mq_confirm.py` (`flip_margin_quantiles` → live m_q; τ*=m_q/ln5, SC-3 live input).
- `src/tac/witness_dsl/{curriculum_dsl,lever_registry,lawref_builtins}.py` + `canonical_equations/evaluators.py`
  (Lever-factory pattern = `LrAnnealPin`; `tail_cycle_floor_v1`/`settle_window_v1` LawRefs = req-T source).
- Trainer resume path (`_resume_lever_divergences`, `_stage_tag`, muon-finisher switch, `_cl_stop_now` break).

## WHAT LANDED (2 commits)
1. **`tac.witness_control.tail_cycles`** (new, exclusive — 7d69aff57): the pure deterministic controller.
   - `next_tau(prev, cfg, live_mq)` = **τ_k = max(τ_{k−1}·halving, m_q/ln5)** clamped ≥ τ_end (§row-9);
     `live_mq=None` (default) ⇒ the halving fallback. `lr_for_tau` = **LR ∝ τ_k**. `tau_star_from_mq`=m_q/ln5.
   - `TailController.step(ep, verdict_rows, live_mq)` → `TailStep(tau, lr, begin_cycle, cycle_k, stage_tag,
     stop, reason)`. Cycle exits on **dwell(237)-gated `powerlaw_meat` exit** OR the **cycle-floor(387.09)
     fail-safe cap**; **PowerPlay stop** when a completed cycle's marginal ΔS/ep < floor; **k_max** req-B cap.
     Decide-on-previous (reads the recorded verdict trace, no lookahead); fail-safe (unfittable ⇒ NOT exhausted).
2. **Trainer + DSL wiring** (shared files — c10b09636, see co-attribution note below):
   - **8 default-OFF flags** (verified real spellings, never-invent): `--tail-cycles-max` (0=OFF=byte-identical),
     `--tail-start-epoch`, `--tail-cycle-floor-epochs` (387.09), `--tail-dwell-min` (237), `--tail-tau-halving`
     (0.5), `--tail-lr-prop-tau` (1.0), `--tail-stop-marginal-s` (1e-4), `--tail-live-mq` (fail-closed = owed
     SC-3 render). Requires `--muon-start-epoch` (fires post-Muon).
   - Loop seam (all guarded `if _tail_ctrl is not None …` ⇒ skipped when off ⇒ **byte-identical**): controller
     built after the finisher-LR setup; τ-override (over the finisher freeze) + warm-restart LR (moments
     UNTOUCHED — `opt.learning_rate` set, no re-init) each tail epoch; **per-cycle `stageTail{k}_muon`
     stage-encoded ckpt** (per-stage-checkpoint non-negotiable); `_tail_stop_now` breaks at the loop bottom
     (mirrors `_cl_stop_now` → final ckpt + result.json land normally).
   - **DSL `TailCycles(...)` Lever factory** (curriculum_dsl) — triality: `lever_registry.completeness()` now
     MAPS all 7 tail flags (were unmapped); consumes `tail_cycle_floor_v1`/`settle_window_v1` defaults (req-T).

## DEFAULTS-OFF PROOF
`--tail-cycles-max 0` (default) ⇒ `_tail_ctrl` stays `None` ⇒ every loop tail block short-circuits ⇒ the
trainer is byte-identical to pre-build. Proven by source-asserting the flag defaults + the None-guard
(`test_tail_cycles_trainer_wirein.py`), and by the controller-is-None gate. ruff F clean; 19 tests green.

## THE ADOPTION SEAM (item 3 — the delicate part)
Run-1's `launch.sh` is frozen and **resume replays frozen literals verbatim** (seal ledger). Adoption is
therefore a **NEW governed launcher invocation** resuming from run-1's last stage checkpoint with the tail
flags appended + `--epochs` extended:

    ... --resume-from <run-1>/levelset_ckpt_stage..._muon_epNNNN.npz \
        --tail-cycles-max 5 --tail-dwell-min 237 --tail-cycle-floor-epochs 387.09 \
        --tail-tau-halving 0.5 --tail-lr-prop-tau 1.0 --tail-stop-marginal-s 1e-4 --epochs <extended>

**Loader-side tolerance PROVEN**: `_resume_lever_divergences` checks only persisted `__cfg_*` keys — the
tail flags are never persisted, so a pre-tail checkpoint resumed WITH tail flags raises **NO divergence**
(the flags are a resume-time config EXTENSION, not silent drift). Test:
`test_resume_tolerates_new_tail_flags_absent_from_frozen_ckpt`. **No live-process change; adoption = operator-GO
at a stage boundary.** (Live-m_q τ*_k is the owed SC-3 render build — `--tail-live-mq` fail-closes; the sealed
τ-halving fallback is the operative default.)

## CO-ATTRIBUTION (anti-absorption, transparent)
The two SHARED files (trainer, curriculum_dsl) co-carry the **concurrent LADDER island-homotopy agent's**
additive edits (`LadderIslandHomotopy` factory + `_ladder*` trainer wiring). Committing shared files whole-file
via the serializer co-carries them; this is declared in the commit body + here (transparent co-attribution,
not silent absorption — both features additive + independent). My exclusive controller+tests landed separately
(7d69aff57). Parent/ledger reconciles attribution.

## MEANS ≠ ENDS
Machinery only; advisory. Pointer contest-CPU **0.19110 UNMOVED** — only a byte-closed n600 `evaluate.py` row
moves it. TAIL fires ~ep1100+ on run-1's timeline; it exists now so run-1 can adopt it at the FIN→TAIL boundary.
