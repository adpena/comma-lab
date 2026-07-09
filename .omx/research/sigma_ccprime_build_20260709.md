# σ_cc′ per-class-pair surface tension — RECONCILE + 2nd derivation (task #382, P0, 2026-07-09)

**Axis:** `[macOS-CPU advisory] NON-PROMOTABLE`. **Pointer contest-CPU 0.19110 UNMOVED — MEANS.**
$0, no GPU, #205 untouched. Operator directive: *"Pursue per-class-pair surface tension σ_cc′ as p0."*

STORES CONSULTED: `.omx/research/t5_crucible2/position_S1_deepmath_20260709.md` (§4 missing-term
analysis + σ_cc′ headline) · `experiments/train_levelset_witness_realized_through_R_mlx.py`
(`_eikonal_length_mlx`, the `--length-sigma-matrix` wiring, micro-batch twin) ·
`src/tac/boundary_math/length_sigma.py` · `src/tac/witness_dsl/curriculum_dsl.py:LengthSigma` ·
`src/tac/canonical_equations/junction_young_sigma_and_powerlaw_exit_20260707.py` ·
`experiments/results/solver_pack_20260707/junction_sigma/junction_sigma_fit.json` (the measured
primary artifact) · DAG FEED-08a (rows 9509/9513/9630/9653) · `.omx/research/length_sigma_lever_build_20260707.md`
· MEMORY.md L75 (canonical class order, MCF-erasure law), CLAUDE.md §config-orphan + §value-provenance-ladder.

## 1. Headline: σ_cc′ was ALREADY BUILT (anti-forgetting reconciliation)

Crucible-2 S1, working independently (no cross-read), flagged σ_cc′ as the "MISSING term (the
strong catch)." **It is not missing — it landed 2026-07-07 (commit 3571e5b65).** Full inventory:

| piece | location | state |
|---|---|---|
| loss generalization | `_eikonal_length_mlx(..., sigma_matrix=)` — per-interface gather σ[top1,top2] at {m=0} | default `None` = byte-identical |
| resolver + measured fit | `tac.boundary_math.length_sigma` (Young's-law preset, fail-closed) | complete |
| DSL lever | `curriculum_dsl.py:LengthSigma` → `--length-sigma-matrix` | complete |
| canonical equation | `junction_young_angle_sigma_fit_v1` | registered |
| DAG | FEED-08a | present |
| tests | `test_length_sigma_lever.py` (25) | pass |
| wiring | main loss, micro-batch twin, logging, argparse, startup fail-close | complete |

**The crucible can pick up the lever from the DSL TODAY**: `LengthSigma("fitted-20260707")`.

## 2. Crucible S1-vs-S2 answer: σ_cc′ GENERALIZES, does NOT ADD (DERIVED, code-anchored)

`L_length(σ) = mean_{m=0}( σ[top1,top2] · δ_ε(m) · |∇m| )`. σ is a per-interface MULTIPLIER on the
**same gradient channel** as the incumbent scalar length term. σ≡1 (`all-ones` → resolver returns
`None` → the pre-existing unweighted branch) recovers the incumbent **byte-identically** — a code-path
property, not multiply-by-1.0. Therefore σ_cc′ **generalizes** the length term; it composes with
**zero new loss-share confound** (unlike a competing added term). Anchor: existing regression test
`test_all_ones_matrix_bitwise_identical_to_default`. This is the direct answer to the S1/S2 tension.

## 3. The derivation law for σ

**Incumbent (principled, default treatment): Young's-angle Herring force-balance.** The frozen
scorer's triple-junction angles invert (σ_jk/sin θ_i = σ_ik/sin θ_j = σ_ij/sin θ_k) to a pairwise
tension matrix; **σ[Road-Lane]=0.377 [0.317,0.441]** excludes all-ones (uniform over-penalizes lane
boundary ~2.7× — the named lane-erasure mechanism; lowering it is the anti-erosion cure). Junction
angles ARE the Herring readout of surface tension for a frozen scorer — this is the correct σ.

**NEW (task DERIVE-half): a 2nd, independent fragility law.** `σ_cc′ = exp(−k·f_cc′)`, where
`f_cc′ = (arc≥180 sliver junctions)/(all junctions touching the pair)`, aggregated over every
triple containing (c,c′), geomean-1 gauge over observed off-diagonal pairs (unobserved → 1.0 null),
`k=1` (DERIVED gauge: one nat per unit drop-fraction). It **uses exactly the 19.3% arc≥180 slivers
the Young's-angle fit DISCARDS** (those are the flux-limited / erasure-prone regime). Producer:
`tac.boundary_math.length_sigma:derive_fragility_sigma_from_junction_fit` (deterministic; reproduces
the hardcoded `fragility-20260709` preset bit-for-bit). Selectable via `LengthSigma("fragility-20260709")`
— **the existing spec-agnostic lever consumes it; NO duplicate lever** (config-orphan avoidance).

## 4. Key measured finding (FORMALIZATION_PENDING): the two σ laws DISAGREE

| pair | Young's-angle σ | fragility σ | agree? |
|---|---|---|---|
| **Road-Lane** (flip-dominant) | **0.377** (lowered) | **1.029** (NOT lowered) | **DISAGREE** |
| Lane-Undrivable (thin sliver) | 0.738 | 0.710 | agree (both < 1) |

Road-Lane has 805 clean junctions, so its drop-fraction (0.312) is diluted → fragility does NOT
lower it, while Young's-angle strongly lowers it (angle far from Herring). **Verdict: the σ
derivation law is LOAD-BEARING — the angle-based Herring derivation is the principled default;
fragility is a second A/B arm (let-math-arbitrate, NOT an asserted improvement).** A fragility arm
that fails to lower the erasure-dominant pair is itself the evidence that abundance is a poor proxy.

## 5. Why NOT a duplicate `PerPairSurfaceTension` lever

The task template named a new `PerPairSurfaceTension` DSL factory + new flags. Building it would
DUPLICATE the existing `LengthSigma` lever / `--length-sigma-matrix` flag = the config-orphan
anti-pattern (CLAUDE.md). The disciplined move: **reuse `LengthSigma`** (it is spec-agnostic;
`LengthSigma("fragility-20260709")` routes with no invented flag) and add only the new preset +
producer. The crucible picks up either arm from the DSL/ledger.

## 6. Triality + acceptance

- **DAG:** FEED-sigma-ccprime (this build's row).
- **DSL:** `LengthSigma("fragility-20260709")` routes to `--length-sigma-matrix` (existing lever).
- **Equations:** `src/tac/canonical_equations/sigma_ccprime_generalization_20260709.py` — (A)
  `sigma_ccprime_length_generalization_v1` (VERIFIED, the generalization classification) + (B)
  `sigma_ccprime_fragility_cross_derivation_v1` (FORMALIZATION_PENDING, the disagreement finding).
  Explicit `populate_sigma_ccprime_equations` (no import-side-effect JSONL write, matching the
  solver-pack pattern).
- **Tests:** 21 new (`test_sigma_ccprime_cross_derivation.py`) + 23 existing PASS; ruff F clean.
- **Resume-registry:** N/A — σ is a config-resolved static matrix (like `fitted-20260707`), threaded
  through the existing `_len_sigma` path; no new checkpoint/persisted state.
- **Micro-batch twin:** already threads `_len_sigma` (trainer:5251) — the new preset flows for free.

**verdict_scope:** the fragility law's "does-not-lower-Road-Lane" is an INSTANCE result on this k=1
gauge + this artifact, NOT a claim the persistence/width FAMILY is dead (a width-from-GT statistic
could differ). The A/B (fitted vs fragility vs all-ones, n600 through R) is the OWED arbiter.

Pointer 0.19110 UNMOVED — every line here is MEANS until a byte-closed `upstream/evaluate.py` n600
row < 0.19110.
