# FEED-497 — optimal basis beyond Fourier — 2026-07-14

**Pointer status:** submittable `[contest-CPU]` **0.1910828242 UNCHANGED**; the
non-submission defensive bank **0.1880443979880752 UNCHANGED**. This FEED is
research/system intelligence, not a score row.

`research_only=true` · `$0 saved-artifact audit` · `NO training launch` ·
`NO evaluator run` · `NO paid dispatch` · `verdict_scope=FORMULATION`.

## STORES CONSULTED

`CLAUDE.md` · `AGENTS.md` · `PROGRAM.md` ·
`docs/operating_manual_craft_handoff.md` ·
`.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md` ·
`.omx/research/SPEC_v8_perclass_decomposition_20260708.md` ·
`.omx/research/owed16_verdict_20260710.json` ·
`.omx/research/owed16v2_verdict_20260710.json` · the three preserved owed-16
EMA checkpoints · `src/tac/boundary_math/lever_b_levelset_generator.py` ·
`src/tac/boundary_math/lever_b_generator.py` ·
`tools/levelset_byte_close_and_eval.py` · current task-#500 live-inbox routing.

## DAG delta

```text
historical n96 circular-GT direct-partition proxy (-48%)
  |
  +--> owed16 real-n600 through-R, bounded warm-start, seed0
  |      polar directional Fourier OFF: 0.004244, 109559 values
  |      self-oriented Fourier along8:   0.004259, 111095 values
  |      self-oriented Fourier along26:  0.004286, 111095 values
  |      => FORMULATION: OFF wins; -48% does not transfer; fresh-start OPEN
  |
  +--> source-structure re-derivation
         legacy `curvelet` atom = sin/cos(2 pi x dot k)
         paired envelope span <= 1.47e-7, no window, no translation index
         => it is global polar directional Fourier, NOT a true curvelet frame
         => genuine-frame comparison remains NO-VERDICT / unmeasured

open reformulation ladder (priority order)
  hybrid Fourier-interior + localized-curvelet-boundary
    -> true windowed curvelet
    -> compact shearlet
    -> steerable Gabor
    -> wavelet / local spline
    -> learned-frequency and metric-eigen families
```

## Triality

- **DSL:** `tac.witness_dsl.optimal_basis_20260714.BasisLeverSpec` is the typed
  basis-family config surface. It compiles the measured polar-Fourier fallback,
  the scoped self-orient reproduction, and the open SIREN/FINER surface through
  the trainer's real argparse. Every unimplemented genuinely different frame
  refuses with a typed train+inflate parity/n600-custody blocker.
- **Equation:**
  `tac.canonical_equations.optimal_basis_selection_20260714` implements
  `optimal_basis_equal_budget_through_r_v1`:

  `B*(K,A) = argmin_B d_seg(SegNet(R(G_{theta_B,B})))`

  subject to `|theta_B| <= K` and exact archive bytes `|A_B| <= A`. Missing
  archive bytes are ineligible for an archive-constrained verdict.
- **DAG:** this standalone FEED. It deliberately does not append to the shared
  hot DAG while other arms are active.

## Basis-perpendicular-metric contract (#500 handoff)

The basis owns the primal atom map `psi_i(x) -> theta_B`. Task #500 owns the
registered decision metric/pullback `G_q`. Its canonical law ID is
`argmax_native_vjp_fidelity_v1` in `tac.scorer_surrogate.vjp_fidelity`; the
state receipt schema is `reachable_decision_geometry_fidelity.v1`, the selector
schema is `reachable_decision_preconditioner_selection.v1`, and the candidate
preconditioner is `winner_rival_margin_fisher_natural`. The non-owning interface
requests `<psi_i, G_q psi_j>`; a metric-sparse family minimizes measured
through-R debt and off-diagonal Gram interference under the same parameter/byte
budget. Full-n600 metric selection remains `NO-VERDICT_DATA_CUSTODY`. No metric
schedule, helper, or selection receipt is implemented here.

## Reactivation criteria

Do **not** rerun owed16 along8 or along26 bounded warm-start. Reactivate only:

1. a fresh-start matched family isolation (distinct verdict scope), or
2. a genuinely localized frame with train/generated-inflate op parity, exactly
   matched trainable values, deterministic decode, and a real-n600 through-R
   receipt, followed by exact byte-close if it wins.

The first admissible genuine-frame experiment is a three-arm, same-seed,
same-schedule, same-`in_feat=80` comparison: global polar Fourier vs windowed
curvelet vs compact shearlet. The frame generator is rule-118 free; every
learned/video-derived coefficient remains counted. Heavy execution remains
operator-GO gated.
