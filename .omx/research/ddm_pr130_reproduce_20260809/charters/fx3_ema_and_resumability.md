# ddm_fx3 — CURE 4: the QAT trainer has no EMA and is not crash-resumable

**Operator 2026-08-09: "Continue with all."** This is the first cure that *improves on the borrowed
base* rather than repairing our port of it. Two of our own HIGHEST-EMPHASIS non-negotiables are
unmet by the code we are about to iterate on.

## THE FINDING (rr2, `add4e9b34d`, do not re-derive)

`RR2_SEMANTIC_LEG_AUDIT.md` verdict: the semantic leg's core is CLEAN — but the QAT trainer is
**(P0) not crash-resumable and carries no EMA**, and **(P1)** its checkpoint `config` describes an
ANCESTOR run six transformations stale, with 19 read sites and 5 propagators; the LIVE schedule exists
only at `checkpoint["result"]["config"]`. Pins: stage-07 ckpt sha `1549607db224…` (283,432 B),
stage-08 sha `3948ccfcd447…` (282,352 B).

**MAIN's severity adjudication, which you should test rather than inherit:** measured step times bound
a stage to ~17–59 min, so the *resumability* half is less severe than the multi-day rule contemplates;
the *EMA* half is duration-independent and score-relevant — CLAUDE.md requires the EMA **shadow** (not
live weights) to be the inference checkpoint, and requires per-stage checkpoints preserved under
distinct stage-encoded filenames, saved atomically, never loop-end-only. If your measurement disagrees
with my bound, say so with numbers.

## YOUR SCOPE

1. **EMA, per the canonical contract.** `tac.training.EMA` is the canonical class (float-buffer guard,
   late-bound module guard). Wire it into our lifted QAT trainer: update after every `optimizer.step()`;
   apply ONLY at eval with snapshot+restore (the canonical pattern — shadowing live weights inside the
   epoch loop freezes learning); ship the **shadow** as the inference checkpoint. Decay is NOT the flat
   0.997 by default: the DERIVED authority is LawRef `ema_decay_run_geometry_v1` (decay follows run
   geometry — steps/epoch × horizon); resolve through it, and only fall back to 0.997 where the LawRef's
   inputs are genuinely absent. **QAT caveat that binds you:** EMA changes the deployed weights inside a
   quantization-aware loop. Adopting it requires an **argmax-parity check on the deployed path** — build
   that gate and report the measured argmax delta, do not assume it is zero.
2. **Resumability + per-stage checkpoints.** `--resume-from` that continues bit-faithfully; a complete
   byte-close-loadable checkpoint at EVERY stage boundary under a DISTINCT stage-encoded filename (never
   overwrite the prior stage); periodic intra-stage saves for long stages; atomic write (tmp+rename);
   every cfg key the byte-close and resume paths need. Then PROVE it: a crash-resume smoke where a
   killed run resumes and matches the uninterrupted trajectory.
3. **Kill the stale-config hazard at the read sites.** Any consumer reading `ckpt["config"]` as
   provenance is reading an ancestor. Make the live schedule the authoritative surface, or make the
   stale one refuse. 19 read sites / 5 propagators is the population — report your swept denominator.

## OPTIMAL FORM

- **Reference form:** EMA wired per the canonical snapshot+restore contract with a LawRef-resolved
  decay AND a measured argmax-parity gate on the deployed QAT path; resumability proven by a real
  kill-and-resume trajectory match; the stale-config read sites cured or fail-closed.
- **SCOPE reductions (legal):** short runs / small step counts / CPU-only for the resume and parity
  smokes — the mechanism is what is under test, not the score. Reduced n for the smoke is fine if
  stated.
- **MECHANISM reductions (declare TOY-BRACKET):** EMA that shadows live weights during training;
  loop-end-only saving; a single overwritten checkpoint path; a resume smoke that only checks the file
  loads rather than that the trajectory matches; adopting EMA with the argmax-parity gate unrun.
- **Provenance pins:** rr2 audit `add4e9b34d`; stage-07 `1549607db224…`; stage-08 `3948ccfcd447…`;
  canonical `tac.training.EMA`; LawRef `ema_decay_run_geometry_v1`.

## NON-NEGOTIABLES

- Intake READ-ONLY — cure our lifted copy under `src/tac/pr130_lift/`, never the intake. Any body
  change to a lifted file MUST be declared in its `borrowed_substrate_accounting` header (the exact
  defect ddm_fx2 is curing in parallel — coordinate, do not collide: fx2 owns the header/test
  machinery, you own the trainer mechanism).
- MPS/MLX never score authority; `score_claim=false`; no exact-score claims from this arm.
- Resumability + per-stage checkpoints are P0 by standing operator binding — this arm exists because
  the borrowed base violates it.
- **Never consume a background job's output without asserting terminal status.**
- verdict_scope on every negative. Denominators on every count.
- Commit via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256` per file,
  tags `[no-triality] [p0-ledger-ok]`. NO Claude/AI attribution, no `Co-Authored-By`.
- **`REVIEW_GATE_OVERRIDE=1` is FORBIDDEN here — this arm edits `.py`.** Use
  `tools/review_tracker.py mark-file <file> --status reviewed` (two passes per file).

## DELIVERABLE

`.omx/research/ddm_pr130_reproduce_20260809/FX3_EMA_AND_RESUMABILITY.md` — **§1 = the measured
argmax-parity result for EMA on the deployed QAT path** (the gate that decides whether EMA ships),
then the resume-trajectory-match receipt, the stale-config sweep with denominator, ranked residuals
with falsifiers, and "could not check / why."
