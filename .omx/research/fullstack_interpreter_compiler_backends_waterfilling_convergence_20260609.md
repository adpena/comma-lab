# Full-stack interpreter/compiler — HiNeRV/SNeRV/PACT-NeRV as backends in one waterfilling currency

UTC: 2026-06-09 · claude · strategic architecture record (operator question 2026-06-09:
"are hinerv, snerv, pact-nerv ... backends ... in the final ultimate full stack
interpreter/compiler solution, and also pushing fullstack hinerv pr95-style and
fullstack snerv pr95-style all in parallel?"). Answer: YES to all three.

## The architecture (the "forest")
Per CLAUDE.md "Evaluator-Equivalent Witness Compiler Paradigm": the goal is the
shortest legal `archive.zip` whose `inflate.sh` output is a witness inside the frozen
evaluator cells. NeRV/HNeRV/HiNeRV/SNeRV are **possible witness BACKENDS, not the
goal**; the winning representation is whichever legal MIXTURE minimizes
`S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489`.

So:
- **inflate.py is the VM / interpreter.** Every archive word is an opcode it executes
  into witness frames.
- **HiNeRV, SNeRV, PACT-NeRV are BACKENDS** — each emits archive bytes (a witness
  program) the VM interprets. None is "the answer"; each is a candidate code generator.
- **The compiler layer = the evaluator-action waterfilling law** (`a350acdc3`,
  `tac.optimization.evaluator_action_waterfill`). It is the ONE currency that composes
  every backend + action atom: each is admitted iff `S(base + atom) < S(base)` by exact
  contest ΔS, ranked by value-per-byte, composed with measured commutators
  (nonlinear/noncommutative), and base-bound (anti-drift: a candidate expires when the
  base changes). This is what the sidecar incident proved we needed.

## One currency, all atom families (the unification)
`CandidateActionEvaluation` is the shared evaluator for:
- HiNeRV backend weights / precision deltas
- SNeRV LF/HF/MFU/HFR/TUB source-state actions
- PACT-NeRV selector outputs
- target-region sidecars, semantic cells, pose-compensation atoms, codec choices
Each reports `Δd_seg / Δd_pose / Δbytes / scorer_effect_survived / commutators` and is
admitted only by exact ΔS. No atom enters the archive by "it parses + applies"; it must
pay rent. This is the waterfilling law for the evaluator quotient.

## Parallel vehicles (all three active)
1. **Full-stack HiNeRV PR95-style** — UNBLOCKED 2026-06-09. The sidecar bug fix proved
   the backend SURVIVES parse-back (~11306 region wins); the destroyer was the sidecar,
   now de-conflated + gated + auto-stripped. Next: #27 full-size 600-pair PR95-style run
   → strip sidecar (lossless helper) → exact CPU/CUDA eval vs the 0.192 frontier.
2. **Full-stack SNeRV PR95-style** — parallel lane (separate commits, not blocked by
   HiNeRV). Hard blocker per CLAUDE.md: official MFU/HFR/TUB source-forward
   train/export/runtime binding + LF/HF representation collapse under real byte pressure.
   The TUB DROP_OR_REIFY source-forward proof is the gate before SNeRV source-forward
   rows can enter LoweringRace.
3. **Interpreter/compiler** — the waterfilling law (built `a350acdc3`) + inflate.py VM +
   the deterministic tie-resolution corrector (`8714cfd36`, torch-exact scorer) + the
   MLX↔torch render parity (uint8-eliminated). This is the layer that COMPOSES vehicles
   1 + 2 into a single rent-paying archive word.

## Current convergence status (2026-06-09)
- Compiler currency (waterfilling law): BUILT + tested + pushed.
- HiNeRV backend: UNBLOCKED, bug fixed + autonomous; #27 full-size run is the next step.
- SNeRV: parallel lane continues (TUB source-forward proof outstanding).
- Scorer fidelity: render drift uint8-eliminated (0px≥1/255); SegNet drift = float-order
  ties only, deterministically torch-corrected.

## DO NOT
- Do not treat any backend as "the answer" — they are code generators judged by ΔS.
- Do not admit any atom (any backend's output) without the rent law `S(base+σ)<S(base)`.
- Do not let HiNeRV and SNeRV block each other — parallel lanes, one shared currency.
- Do not promote on region-win proxies — exact paired upstream eval is the authority.
