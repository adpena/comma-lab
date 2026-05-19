# More-optimal-algorithms wire-in callsite plan

**Date:** 2026-05-19 (UTC) | **Lane:** `lane_more_optimal_algorithms_wire_in_20260518`
**Council anchor:** T3 finding #6 PROCEED at commit `6606376eb`
**Empirical anchor:** paired-comparison at commit `35c5d429f`
**Canonical shim:** `src/tac/solvers/more_optimal_algorithms.py`

## TL;DR

Per finding #6's 5 op-routables:

- **#1 FISTA → `tac.bit_allocator.allocate_bits`** — 4 substrate-relevant callsites identified; deferred to per-substrate symposium per Catalog #325.
- **#2 Frank-Wolfe → Sinkhorn callsites** — 1 callsite (`tac.losses.core.sinkhorn_w2_mask_distortion_per_pixel`) used by cathedral autopilot ranker bidirectional matching; deferred to per-substrate symposium.
- **#3 Riemannian-Newton → PQ codebook init** — 3 Lloyd-Max sites identified; deferred to per-substrate symposium.
- **#4 Canonical shim package** — **LANDED THIS WAVE** at `src/tac/solvers/more_optimal_algorithms.py`.
- **#5 Wall-clock atom emission** — **LANDED THIS WAVE** via `build_more_optimal_algorithm_wall_clock_atom` + `append_more_optimal_algorithm_wall_clock_event`.

This memo is the **callsite plan** for op-routables #1-#3. Per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" + Catalog #325 per-substrate symposium discipline, every per-substrate adoption decision MUST go through its own symposium with paired-comparison evidence — this wave delivers the **canonical drop-in surface** so each symposium can adopt or fork without re-implementing the algorithm.

## Op-routable #1: FISTA → bit-allocator callsites

Per `ALGORITHM_FISTA.consumer_callsites`:

| Callsite | Adoption verdict | Per-substrate symposium gate |
|----------|------------------|------------------------------|
| `src/tac/bit_allocator.py::allocate_bits` | UNCLEAR_NEEDS_EMPIRICAL | Whichever substrate consumes `allocate_bits` first — likely a future per-tensor PTQ trainer per the Catalog #220 substrate L1+ scaffold operational mechanism declaration. |
| `src/tac/water_filling_codec.py` | UNCLEAR_NEEDS_EMPIRICAL | Cross-substrate (PR101 / fec6 / DSnerv); each substrate's symposium per Catalog #325 evaluates whether the 1.25× speedup is worth the canonical-helper-fork risk per Catalog #290. |
| `src/tac/hessian_block_fp.py` | UNCLEAR_NEEDS_EMPIRICAL | Selfcomp-class substrates (lane G v3 / Quantizr clones). |
| `src/tac/mdl_fp4_tto.py` | UNCLEAR_NEEDS_EMPIRICAL | TTO loop is post-training; lower priority — wire-in deferred until TTO substrate symposium queues. |
| `src/tac/master_gradient_consumers.py` | UNCLEAR_NEEDS_EMPIRICAL | Master-gradient consumers run at compress time; wire-in via canonical helper `tac.master_gradient_consumers.load_optimal_plan_for_archive` would require Pareto re-derivation — defer to Lagrangian-planner sister wave. |

**Predicted speedup at adoption:** 1.25× wall-clock per call. Across a session with ~20-50 dispatched bit-allocator invocations (training + post-training), this is a ~5-10 min wall-clock savings — material for race-mode velocity per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first".

**Risk classification:** drop-in safe; byte-identical solution invariant verified at commit `35c5d429f`. No score-axis risk (same bit allocation = same archive bytes = same score).

**Canonical drop-in template** (for per-substrate-symposium adoption):

```python
from tac.solvers.more_optimal_algorithms import (
    fista_proximal_gradient,
    project_simplex,
    append_more_optimal_algorithm_wall_clock_event,
)
import time

# Replace water-filling bisection inner loop with FISTA proximal gradient.
def allocate_bits_fista(importance, budget, *, min_bits=1, max_bits=8):
    # Smooth part: -alpha * sum(importance * log(bits)) (HAWQ-style)
    def grad_smooth(bits):
        return -0.5 * importance / (bits + 1e-12)
    # Prox: project onto budget-simplex (bits >= 0, sum == budget)
    def prox(z, _step):
        clamped = project_simplex(z.clip(min_bits, max_bits), budget)
        return clamped
    t0 = time.perf_counter()
    result = fista_proximal_gradient(
        grad_smooth, prox, x0=importance, step_size=0.01,
        max_iters=200, tol=1e-8,
    )
    t1 = time.perf_counter()
    # Per finding #6 op-routable #5: emit canonical wall-clock atom
    append_more_optimal_algorithm_wall_clock_event(
        atom_id=f"fista_bit_allocator_{int(t1)}",
        algorithm_id="fista_beck_teboulle_2009",
        consumer_callsite="src/tac/bit_allocator.py::allocate_bits",
        wall_clock_seconds_canonical=...,  # measured paired baseline
        wall_clock_seconds_more_optimal=t1 - t0,
        n_iterations_canonical=64,
        n_iterations_more_optimal=result.iterations,
        invariant_check_passed=True,  # verify against canonical post hoc
        invariant_description="byte-identical bit allocation vs water-filling",
    )
    return result.x
```

## Op-routable #2: Frank-Wolfe → Sinkhorn callsites

| Callsite | Adoption verdict | Per-substrate symposium gate |
|----------|------------------|------------------------------|
| `src/tac/losses/core.py::sinkhorn_w2_mask_distortion_per_pixel` | UNCLEAR_NEEDS_EMPIRICAL | This is a training-loop loss term used by many substrate trainers; per-substrate decision required because Frank-Wolfe gives **hard K=N** selection while Sinkhorn gives **soft assignment** — different gradient signal. SIREN / NeRV / DSnerv class trainers may want hard; coordinate-MLP / Cool-Chic class may want soft. |
| Cathedral autopilot ranker bidirectional matching (callsite not yet named in canonical helper namespace) | UNCLEAR_NEEDS_EMPIRICAL | Catalog #322 sister gate already enforces autopilot consumer routing through canonical helpers — Frank-Wolfe adoption MUST land via the canonical shim with sister test pinning the K-sparsity invariant. |

**Predicted speedup at adoption:** 1.9× wall-clock per call. Cathedral autopilot ranker calls Sinkhorn-style matching dozens of times per re-ranking; adoption is **race-mode velocity-multiplier per CLAUDE.md "Race-mode rigor inversion"**.

**Risk classification:** K-sparsity contract change. Sinkhorn callers receiving soft assignments expect a doubly-stochastic matrix; Frank-Wolfe returns an extremal vertex (exactly K indicators). API-change-required at the consumer surface. Per Catalog #322 the autopilot ranker has a documented mapping for both paths.

**Canonical drop-in template:**

```python
from tac.solvers.more_optimal_algorithms import (
    frank_wolfe_kcard,
    lmo_kcardinality,
)

# Replace Sinkhorn K-ensemble selection with Frank-Wolfe K-cardinality.
def select_top_k_frank_wolfe(scores, k=8):
    def grad(x):
        return -scores  # minimize -<scores, x>
    def obj(x):
        return float(-scores @ x)
    result = frank_wolfe_kcard(grad, obj, n=len(scores), k=k, max_iters=50)
    # Exactly K members selected (sparsity invariant)
    return result.selected_indices
```

## Op-routable #3: Riemannian-Newton → PQ codebook Lloyd-Max init

| Callsite | Adoption verdict | Per-substrate symposium gate |
|----------|------------------|------------------------------|
| `src/tac/pr101_split_brotli_codec_derivers.py::derive_sidecar_codebook` | UNCLEAR_NEEDS_EMPIRICAL | PR101 substrate symposium per Catalog #325. The 16-value sidecar codebook init currently takes Lloyd-Max O(100) iterations; Riemannian-Newton would converge in O(10) iterations with machine-ε orthogonality. **But** the existing codebook is scalar (1D) — the Stiefel manifold St(n,p) formulation needs p>=1 dimensional codebook vectors. Possible adoption path: PQ codebook (p=8 subvectors, K=64 entries each). |
| `src/tac/quantization_wave/vq_codebook_quantization.py` | UNCLEAR_NEEDS_EMPIRICAL | VQ-VAE substrate symposium per Catalog #325. VQ codebooks naturally fit Stiefel (orthogonal codebook entries). |
| `src/tac/symposium_impls/carmack_hotz_strip_everything_codec.py` | UNCLEAR_NEEDS_EMPIRICAL | NSCS06 substrate symposium per Catalog #325. Per the Lloyd-Max pose quantizer at line 35-87. |

**Predicted speedup at adoption:** 1.88× wall-clock per call. PQ codebook init runs once per substrate train, ~30-60 sec savings per dispatch.

**Risk classification:** invariant contract change. Lloyd-Max returns scalar centroids; Riemannian-Newton returns orthonormal frame columns. Adoption requires the substrate's codebook to be redesigned as a Stiefel frame, NOT a drop-in for existing scalar codebooks. The machine-ε orthogonality invariant only holds if downstream consumers treat the codebook as an orthonormal frame.

**Canonical drop-in template:**

```python
from tac.solvers.more_optimal_algorithms import (
    StiefelManifold,
    riemannian_newton_step,
)

# Replace Lloyd-Max init with Riemannian-Newton on Stiefel for K=64 PQ entries
# of 8-dim subvectors.
def pq_codebook_init_riemannian(data_subvectors, n_codes=64, subvec_dim=8):
    stiefel = StiefelManifold(n=n_codes, p=subvec_dim)
    X0 = stiefel.random_point()
    def obj(X):
        return float(((data_subvectors @ X.T - data_subvectors).sum(axis=0) ** 2).sum())
    def egrad(X):
        # ... Euclidean gradient of obj
        ...
    result = riemannian_newton_step(obj, egrad, X0, max_iters=20, tol=1e-8)
    assert result.orthogonality_error < 1e-10  # invariant check
    return result.X
```

## Op-routable #4: canonical shim package — LANDED THIS WAVE

`src/tac/solvers/more_optimal_algorithms.py` (~330 LOC):

- Re-exports all 5 solvers from sister modules (FISTA + Frank-Wolfe + Sinkhorn + Riemannian-Newton-Stiefel + Numba JIT water-filling).
- Exposes `CANONICAL_ALGORITHM_REGISTRY` keyed by canonical algorithm_id; each entry is a frozen `AlgorithmCanonicalMetadata` dataclass with `paper_citation` (DOI), `canonical_module_path`, `paired_against` (canonical sister), `wall_clock_multiplier_macos_cpu_advisory`, `convergence_rate_theoretical`, `convergence_rate_paired_baseline`, `invariant_contract`, and `consumer_callsites`.
- Declares full Catalog #305 6-facet observability surface + Catalog #125 6-hook wire-in declaration.
- Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290: shim is NOT a forced canonicalization — exposes stable APIs so per-substrate symposium adoption decisions can adopt or fork.

## Op-routable #5: wall-clock atom emission — LANDED THIS WAVE

`build_more_optimal_algorithm_wall_clock_atom(...)` + `append_more_optimal_algorithm_wall_clock_event(...)` in the shim package:

- Builds a canonical `PROBE_OUTCOME` atom (per Catalog #313 verdict taxonomy) with `verdict="PROCEED"` iff wall-clock speedup >= 1.0× AND invariant_check_passed; else `verdict="OPERATOR_REVIEW_REQUIRED"` per CLAUDE.md "Forbidden premature KILL without research exhaustion" (research-defer, never kill at the algorithm layer).
- Persists via the canonical fcntl-locked `tac.atom.ledger.append_atom` → `.omx/state/atom_ledger.jsonl` per Catalog #131 / #138 / #245 sister discipline.
- Cathedral autopilot ranker can consume PROCEED rows as velocity-multiplier signal via the existing `tac.atom.ledger.query_by_min_predicted_impact` consumer surface.

## Per-substrate symposium gating per Catalog #325

Each of the per-callsite wire-in adoption decisions above MUST go through its substrate's symposium with the canonical 6-step contract:

1. Cargo-cult audit per Catalog #303 (is the canonical sister cargo-culted from prior generation? FISTA vs water-filling is HARD-EARNED; canonical was the right choice when it landed, more-optimal algorithms are NEW WIN).
2. 9-dim checklist evidence per Catalog #294.
3. Observability surface declaration per Catalog #305.
4. Sextet pact deliberation (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary).
5. Per-substrate reactivation criteria pinned per CLAUDE.md "Forbidden premature KILL".
6. Catalog #324 post-training Tier-C validation discipline.

The shim package + wall-clock atom builder + this callsite plan = canonical adoption surface so per-substrate symposiums have everything they need to adopt or fork without re-implementing.

## Cross-references

- Grand council T3 finding #6 PROCEED memo: `.omx/research/council_t3_finding_6_more_optimal_algorithms_empirical_wins_20260518.md`
- Paired-comparison commit: `35c5d429f` (commit message documents 1.25× / 1.9× / 1.88× wall-clock results)
- Findings-deliberation commit: `6606376eb` (15 council memos + 15 anchors + 15 atoms + 8 probes)
- Synthesis memo MORE-OPTIMAL ALGORITHMS appendix: `.omx/research/magic_codec_plus_water_filling_plus_lagrangian_redirection_unified_synthesis_cross_pollination_20260518.md`
- Sister callsite plan helper: `tools/empirical_solver_paired_comparison.py` (the paired-comparison runner used to validate the wall-clock multipliers)
- CLAUDE.md non-negotiables consulted: Race-mode rigor inversion + parallel-dispatch first / Substrate MUST be at OPTIMAL FORM before paid empirical dispatch / PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium / Apples-to-apples evidence discipline / Bugs must be permanently fixed AND self-protected against / UNIQUE-AND-COMPLETE-PER-METHOD operating mode / Max observability
- Catalogs consulted: #125 6-hook wire-in / #126 lane pre-registration / #131 fcntl-locked JSONL discipline / #138 strict-load discipline / #190 hardware-substrate hardcoding / #192 macOS CPU advisory non-promotion / #229 premise verification / #245 4-layer pattern / #270 dispatch optimization protocol / #287 evidence-tag discipline / #290 canonical-vs-unique decision per layer / #294 9-dim checklist / #303 cargo-cult audit / #305 observability surface / #313 probe outcomes ledger / #317 local-research-signal stamping / #322 autopilot composition_alpha provenance / #323 canonical provenance / #325 per-substrate symposium discipline
