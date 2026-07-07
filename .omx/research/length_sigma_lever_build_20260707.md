# σ_ij per-class-pair LENGTH-WEIGHT lever — BUILT (FEED-07x consumption path closed)

**BUILD-WAVE-2 agent F, 2026-07-07. Everything here is MEANS ([macOS-CPU advisory],
NON-PROMOTABLE); pointer contest-CPU 0.19110 UNMOVED. DAG: FEED-08a.**

Consumes: `.omx/research/solver_pack_junction_sigma_powerlaw_20260707.md` (A) +
`experiments/results/solver_pack_20260707/junction_sigma/junction_sigma_fit.json` (commit
3571e5b65) + `tac.canonical_equations.junction_young_sigma_and_powerlaw_exit_20260707`
(`junction_young_angle_sigma_fit_v1`) + viscosity hunt §7 (σ_ij = the Imbert–Monneau
flux-limiter DOF; all-ones imposes Herring 120°).

## 1. Faithfulness determination (the honest-decomposition question, answered from the code)

The trainer's length term (`_eikonal_length_mlx`) is **not per-phi**: it lives on the DECISION
MARGIN `m = φ_top1 − φ_top2` and localizes at `{m=0}` via `δ_eps(m)·|∇m|`. At each pixel the
interface present at `{m=0}` is exactly the **(argmax-top1, argmax-top2) class pair**, so a
pairwise σ_ij weights the integrand FAITHFULLY as a per-pixel gather `σ[top1, top2]`
(`mx.argsort` + `mx.take`; symmetric matrix ⇒ pair order free; argsort is a permutation ⇒ the
diagonal is never gathered). The weight is θ-piecewise-constant (indices carry no gradient) — a
data-dependent constant multiplier, the same class as the spike-reweight lever. **No per-class
row-mean projection was needed** — the per-interface decomposition the solver-pack memo feared
the loss might lack is exactly what the margin formulation already has. The EIKONAL term is
untouched (the SDF property is per-field geometry, not per-interface tension).

## 2. What landed

- **Trainer flag** `--length-sigma-matrix {all-ones | fitted-20260707 | <path.json>}`
  (`experiments/train_levelset_witness_realized_through_R_mlx.py`): default `all-ones` resolves
  to `None` ⇒ the PRE-EXISTING unweighted branch runs untouched (**byte-identical BY CODE
  PATH**); an explicit all-ones matrix through the σ branch is additionally **BITWISE identical**
  (asserted on MLX CPU, 3 seeds/shapes — length, eikonal, and the g-term). Fail-closed startup
  validation (before heavy setup). Threaded through BOTH the serial `total_loss_fn` AND the
  micro-batch twin (a keyword-binding wrapper passed as `LeverConfig.eikonal_length`, so both
  paths weight identically). Runtime shape guard (σ must be K×K) fails loud. Provenance: the
  run-config dict records `length_sigma_matrix`; the startup regularizer-magnitude log is now
  σ-aware (logs the ACTUAL trained length magnitude + the spec).
- **Canonical resolver** `src/tac/boundary_math/length_sigma.py`: `resolve_length_sigma_matrix`
  / `validate_sigma_matrix` / `load_sigma_matrix_json` / `describe_length_sigma`. The
  `fitted-20260707` preset hardcodes the 7 FULL-precision fitted values from the fit JSON
  (σ[Road-Lane]=0.3771195466360733, σ[Lane-Undriv]=0.7381986449045815,
  σ[Road-Undriv]=1.0848087450168646, σ[Road-Movable]=1.0062627915225708,
  σ[Undriv-Movable]=1.0482927871960461, σ[Lane-MyCar]=1.764344211480968,
  σ[Road-MyCar]=1.7791690170773755; provenance comments cite the JSON + commit 3571e5b65);
  NaN-unobserved pairs (Lane-Movable, Undriv-MyCar, Movable-MyCar) = the 1.0 all-ones null;
  diagonal 1.0 (never gathered). The fit tool's own JSON passes directly (dict form
  `fit.sigma_matrix_5x5` / `sigma_matrix_5x5`; NaN filled). Refusals: wrong shape /
  non-symmetric / non-positive / non-finite off-diagonal / missing file / bad JSON.
- **DSL** `LengthSigma(spec="fitted-20260707", window=0)` Lever factory in
  `tac.witness_dsl.curriculum_dsl` (exported from `tac.witness_dsl`;
  `lever_registry.lever_factories()` auto-discovers it — verified). Default spec = the fitted
  TREATMENT; `"all-ones"` is REFUSED (emitting the trainer default = a silent-no-op lever; the
  control arm is the lever's ABSENCE); malformed specs refused at factory time via the canonical
  resolver. Window/stage semantics documented in the docstring: the σ weighting reparametrizes
  the length regularizer itself — active whenever the length term is (every epoch, constant
  `--length-weight`, all curriculum stages); `window=0` = loss-geometry config change with no
  epoch budget (MuonWarmStart convention). Note: a `Lever` factory needs no `_SAMPLES` entry
  (verified against MuonWarmStart — no such coupling exists for factories).
- **Equations leg (refine-not-duplicate):** `junction_young_angle_sigma_fit_v1`
  `canonical_consumers` now name the three landed consumers
  (`tac.witness_dsl.curriculum_dsl:LengthSigma` · `tac.boundary_math.length_sigma` ·
  `experiments/train_levelset_witness_realized_through_R_mlx.py:--length-sigma-matrix`); the
  `domain_of_validity.lever` text moved PROPOSED→LANDED; a registry refinement row was appended
  (same `equation_id`, latest-row-wins) with the note that the A/B remains the OWED anchor.

## 3. Duty-to-measure (verified live after the DSL landing)

`tools/costate_digest.py` output line:

    duty-to-measure (36 owed; *=never-fired): AACoverageRender*, AdamBeta2*, AmplifyIsland*, AnalyticLaneRenderBand*, BoundaryDistance*, CacheGtSkeleton* (+30 more)

The digest truncates the alphabetical list at 6 names; the ledger query confirms membership:
`duty_to_measure()` contains **LengthSigma** and `never_fired()` contains **LengthSigma**
(state = never-fired ⇒ ranked into the costate DECIDE queue).

## 4. No frozen-ckpt probe — stated plainly

This is a **TRAINING lever**: the length term shapes gradients during training, not renders, so
there is nothing to evaluate on a frozen checkpoint ($0 probe impossible by construction). The
σ-weighted vs uniform **A/B (n600 through R, junction-local d_seg attribution)** is the
registered OWED anchor = the equation's reactivation criterion = the duty-to-measure entry.
**No training was launched** (operator-GO-gated per CONTAINMENT).

## 5. Tests + regressions

`src/tac/tests/test_length_sigma_lever.py` — 22 tests, all green:
bitwise identity at all-ones (LOAD-BEARING; 3 seeds/shapes; length+eik+g) · None-kwarg ==
default · fitted preset exact full-precision values + symmetry + null fills · cross-consistency
with the registered equation constants (<1e-3) · JSON round-trip (raw list + dict form + the
REAL fit artifact) · every refusal path (shape/symmetry/positivity/finiteness/missing
path/bad JSON/no key) · runtime wrong-shape guard · treatment changes length ONLY (eik bitwise
unchanged) + grads finite/nonzero · describe() provenance · DSL factory default/name/overrides ·
all-ones refusal · malformed-spec refusal · valid-JSON-path acceptance · never-invent-flags
(flag present in trainer argparse) · lever_registry discovery.
Regressions green: `test_levelset_micro_batch_loss` + `test_feed07b_build_levers` +
`test_loss_term_telemetry` (79) · `test_witness_campaign` + `test_witness_autoconfig` (54).
ruff `--select F` clean on all touched files. Trainer `--help` renders the flag
(guard-acknowledged parse-only smoke).

## Triality
- **DAG:** FEED-08a appended.
- **DSL:** `LengthSigma` Lever factory (flag REAL; auto-discovered; duty-to-measure registered).
- **equations:** `junction_young_angle_sigma_fit_v1` refined (consumers closed; A/B owed).
