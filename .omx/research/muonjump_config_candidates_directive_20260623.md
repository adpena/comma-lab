# DIRECTIVE — Muon-jump stage-8 config candidates (operator pre-authorization 2026-06-23)

**Source:** operator, 2026-06-23, during the jump-to-Muon-now execution: *"We can create a custom
config that strips out rate stuff if it is destabilizing."* Sister of the recursive-review gate
(`muonjump_stage8_recursive_review_deepmath_optimization_20260623.md`, subagent `a5fc824ba2cb08d17`).

## Binding directive for any subagent touching the Muon-jump config

The stage-8 jump config review MUST evaluate THREE candidates and recommend one (not just audit the
faithful default):

- **A — faithful stage-8:** `Muon + l7_softplus + QAT + C1a λ=0.02 + σ=0.1`. Does rate AND d_seg, but
  applies 4 simultaneous shifts (AdamW→Muon, λ 0.01→0.02, σ 0.2→0.1) the natural path eases into.
- **B — rate-stripped pure d_seg finisher (OPERATOR PRE-AUTHORIZED FALLBACK):** `Muon + l7_softplus +
  QAT KEPT + C1a λ=0 STRIPPED + σ held stable (~0.2)`. Fewest simultaneous shifts → cleanest Muon
  d_seg descent. Rate held at the current λ=0.01 equilibrium (~79.5 KB) but may drift UP as Muon moves
  weights off the brotli-friendly manifold — QUANTIFY that drift vs the cleaner/faster d_seg descent.
- **C — warm-in hybrid:** run B first (let Muon close d_seg cleanly), THEN ramp λ→0.02 for a rate
  polish. Sequential, not simultaneous.

## Hard invariants (do not violate)

1. **QAT stays in ALL three.** It is NOT "rate stuff" — it is the byte-close prerequisite. Stripping it
   makes the Muon polish die at int8 quantization (the int5 lesson). "Strip rate" = strip **C1a λ**, not QAT.
2. The rationale for B/C existing is the **simultaneous-shift instability** risk. If the deep-math/adversarial
   review finds A is stable, A is fine (it also banks rate). B/C are the fallbacks if A destabilizes the d_seg finish.
3. Determine the **implementation mechanism** for B/C (a launcher flag for stage-8 `cat_lambda`/`sigma`
   override, or a custom stage spec) so whichever config wins is launch-ready.
4. The rate prize is cheap (~0.005 of S, small bc20 basis) and **recoverable** (a later C1a pass), so a config
   that closes d_seg cleanly at a slightly higher rate is acceptable — the d_seg finish (~0.115 of S) dominates.

## Authority / discipline
`[contest-CPU advisory]` analysis only; reuse probe-2 measured data; CPU/OMP=2; DO NOT touch the live run
(pid 79893, MPS); NO FAKE; no premature kill. The live faithful run keeps running during this gate.
