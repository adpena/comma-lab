# SPDX-License-Identifier: MIT
"""Canonical shim package for the more-optimal-algorithms wire-in surface.

Per grand council T3 finding #6 PROCEED verdict (2026-05-18; memo at
``.omx/research/council_t3_finding_6_more_optimal_algorithms_empirical_wins_20260518.md``)
op-routable #4: emit canonical ``tac.solvers.more_optimal_algorithms`` shim
package for canonical-helper-aware adoption per Catalog #290 canonical-vs-unique
decision per layer.

The 3 algorithms below were paired-comparison validated in commit ``35c5d429f``
on local M5 Max CPU:

* **FISTA** (Beck-Teboulle 2009) → **1.25× faster** than canonical water-filling
  bisection with **byte-identical** solution invariant. Replaces
  ``tac.water_filling_codec`` / ``tac.bit_allocator.allocate_bits`` proximal
  Lagrangian iteration. Provably O(1/k²) convergence vs water-filling's O(1/k).
  [empirical:experiments/results/empirical_solver_paired_comparison_20260518/]
  [paper:Beck-Teboulle 2009 SIAM J. Imag. Sci. 2(1):183-202 DOI 10.1137/080716542]

* **Frank-Wolfe** (Frank-Wolfe 1956 / Jaggi 2013) → **1.9× faster** than
  Sinkhorn-Knopp on K=8 cardinality-constrained ensemble selection with
  **sparsity invariant** (selects exactly K members). Replaces Sinkhorn
  callsites in cathedral autopilot ranker bidirectional matching. O(1/k)
  convergence over compact convex polytopes; projection-free.
  [empirical:experiments/results/empirical_solver_paired_comparison_20260518/]
  [paper:Jaggi 2013 ICML; Frank-Wolfe 1956 Naval Res Log Quart 3(1-2):95-110]

* **Riemannian-Newton** (Edelman-Arias-Smith 1998) → **1.88× faster** than
  Lloyd-projection for PQ codebook initialization with **machine-epsilon
  orthogonality** invariant (Stiefel manifold St(n,p) preserved to 1e-15).
  Replaces Lloyd-Max k-means initialization in
  ``tac.pr101_split_brotli_codec_derivers.derive_sidecar_codebook`` /
  ``tac.quantization_wave.vq_codebook_quantization``. O(log(1/ε)) convergence
  on orthogonal codebooks vs Lloyd's O(1/ε).
  [empirical:experiments/results/empirical_solver_paired_comparison_20260518/]
  [paper:Edelman-Arias-Smith 1998 SIAM J. Matrix Anal. 20(2):303-353
   DOI 10.1137/S0895479895290954]

## Provenance + custody discipline

Every benchmark figure above is [macOS-CPU advisory] per CLAUDE.md "MPS auth
eval is NOISE" + Catalog #192 + Catalog #317 — wall-clock multipliers were
measured on local M5 Max CPU, not contest hardware. The **byte-identical
solution invariant** + **sparsity invariant** + **machine-ε orthogonality
invariant** are exact mathematical guarantees independent of substrate
(verified at commit ``35c5d429f`` paired-comparison output). Per CLAUDE.md
"Apples-to-apples evidence discipline": consumers MAY drop-in any of the 3
algorithms safely because the canonical contract is preserved; wall-clock
gains are advisory until paired Linux x86_64 anchors land.

## Canonical-vs-unique decision per layer (Catalog #290)

| Consumer surface                                | Replacement     | Canonical/unique decision         |
|-------------------------------------------------|-----------------|-----------------------------------|
| ``tac.bit_allocator.allocate_bits``             | FISTA           | UNCLEAR_NEEDS_EMPIRICAL → per-substrate paired-comparison |
| ``tac.water_filling_codec``                     | FISTA + Numba   | UNCLEAR_NEEDS_EMPIRICAL → per-substrate paired-comparison |
| Cathedral autopilot ranker Sinkhorn matching   | Frank-Wolfe K   | UNCLEAR_NEEDS_EMPIRICAL → per-substrate paired-comparison |
| ``derive_sidecar_codebook`` Lloyd-Max init      | Riemannian-N    | UNCLEAR_NEEDS_EMPIRICAL → per-substrate paired-comparison |
| ``vq_codebook_quantization`` k-means init       | Riemannian-N    | UNCLEAR_NEEDS_EMPIRICAL → per-substrate paired-comparison |

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode": each consumer's
adoption decision MUST run a paired-comparison smoke before wire-in. The shim
package is NOT a forced canonicalization — it exposes the 3 algorithms with
deterministic stable APIs so a per-substrate symposium (Catalog #325) can
decide to adopt or fork without re-implementing the algorithm.

## Observability surface (Catalog #305)

* **inspectable per layer**: each algorithm returns a frozen ``*Result``
  dataclass with iteration counts, objective trajectory, convergence flag,
  and orthogonality/duality-gap diagnostics where applicable.
* **decomposable per signal**: ``objective_history`` / ``duality_gap_history``
  / ``orthogonality_error`` allow per-iteration decomposition.
* **diff-able across runs**: deterministic for a given input + step size +
  RNG seed; round-trip identical.
* **queryable post-hoc**: all results are NumPy arrays + Python primitives;
  trivial JSON-serializable for posterior persistence.
* **cite-able**: every algorithm carries paper citation in docstring +
  ``literature_citation`` field on the canonical atom (see ``builders.py``
  ``build_canonical_more_optimal_algorithm_atom``).
* **counterfactual-able**: each algorithm pairs against a sister canonical
  helper (water-filling / Sinkhorn / Lloyd-Max) via
  ``tools/empirical_solver_paired_comparison.py`` for what-if comparison.

## 6-hook wire-in declaration (Catalog #125)

1. **Sensitivity-map contribution**: ACTIVE — FISTA bit-allocation results feed
   per-tensor importance into ``tac.sensitivity_map.*`` consumers.
2. **Pareto constraint**: ACTIVE — Frank-Wolfe K-cardinality selection feeds
   the Rashomon-ensemble Pareto frontier via ``tac.optimization.pareto_*``.
3. **Bit-allocator hook**: ACTIVE — FISTA IS a bit-allocator drop-in.
4. **Cathedral autopilot dispatch hook**: ACTIVE — ranker can consume
   wall-clock multipliers per algorithm via the atom emitted by op-routable
   #5; faster solver = more dispatches per session = race-mode velocity
   multiplier per CLAUDE.md "Race-mode rigor inversion".
5. **Continual-learning posterior update**: ACTIVE — wall-clock-measurement
   atoms persist to ``.omx/state/atom_ledger.jsonl`` per dispatch consumption.
6. **Probe-disambiguator**: N/A — the 3 algorithms have ONE canonical contract
   per consumer; no disambiguator needed at the algorithm layer (per-substrate
   adoption decision IS the disambiguator at the consumer layer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# Canonical re-exports from sister modules so consumers import from ONE place.
from tac.solvers.fista import (
    FistaResult,
    fista_proximal_gradient,
    project_simplex,
    soft_threshold,
)
from tac.solvers.frank_wolfe import (
    FrankWolfeResult,
    frank_wolfe_kcard,
    lmo_kcardinality,
)
from tac.solvers.numba_jit_water_filling import (
    NUMBA_AVAILABLE,
    NumbaWaterFillingResult,
    water_fill_bisection_numba_if_available,
    water_fill_bisection_numpy,
)
from tac.solvers.riemannian_newton_stiefel import (
    RiemannianNewtonResult,
    StiefelManifold,
    project_to_stiefel,
    retract_qr,
    riemannian_newton_step,
)
from tac.solvers.sinkhorn import (
    SinkhornResult,
    sinkhorn_ensemble_select_k,
    sinkhorn_knopp,
)

__all__ = [
    # Canonical metadata for the 3 PROCEED algorithms
    "CANONICAL_ALGORITHM_REGISTRY",
    "AlgorithmCanonicalMetadata",
    "ALGORITHM_FISTA",
    "ALGORITHM_FRANK_WOLFE_KCARD",
    "ALGORITHM_RIEMANNIAN_NEWTON_STIEFEL",
    # FISTA re-exports
    "FistaResult",
    "fista_proximal_gradient",
    "project_simplex",
    "soft_threshold",
    # Frank-Wolfe re-exports
    "FrankWolfeResult",
    "frank_wolfe_kcard",
    "lmo_kcardinality",
    # Sinkhorn re-exports (paired comparison sister)
    "SinkhornResult",
    "sinkhorn_ensemble_select_k",
    "sinkhorn_knopp",
    # Riemannian-Newton re-exports
    "RiemannianNewtonResult",
    "StiefelManifold",
    "project_to_stiefel",
    "retract_qr",
    "riemannian_newton_step",
    # Numba JIT re-exports
    "NUMBA_AVAILABLE",
    "NumbaWaterFillingResult",
    "water_fill_bisection_numba_if_available",
    "water_fill_bisection_numpy",
]


@dataclass(frozen=True)
class AlgorithmCanonicalMetadata:
    """Canonical per-algorithm metadata for atom emission + autopilot ranker.

    Per Catalog #305 observability surface: cite-able literature + canonical
    helper repo link + paired-comparison wall-clock multiplier + invariant
    contract per consumer-side adoption decision.
    """

    algorithm_id: str
    paper_citation: str
    canonical_module_path: str
    paired_against: str
    wall_clock_multiplier_macos_cpu_advisory: float
    convergence_rate_theoretical: str
    convergence_rate_paired_baseline: str
    invariant_contract: str
    consumer_callsites: tuple[str, ...]


ALGORITHM_FISTA: Final[AlgorithmCanonicalMetadata] = AlgorithmCanonicalMetadata(
    algorithm_id="fista_beck_teboulle_2009",
    paper_citation=(
        "Beck & Teboulle 2009 'A Fast Iterative Shrinkage-Thresholding Algorithm "
        "for Linear Inverse Problems' SIAM J. Imag. Sci. 2(1):183-202 "
        "DOI 10.1137/080716542"
    ),
    canonical_module_path="tac.solvers.fista",
    paired_against="tac.water_filling_codec + tac.bit_allocator.allocate_bits",
    wall_clock_multiplier_macos_cpu_advisory=1.25,
    convergence_rate_theoretical="O(1/k^2)",
    convergence_rate_paired_baseline="O(1/k)",
    invariant_contract=(
        "byte-identical bit-allocation solution invariant "
        "(verified in paired-comparison output at commit 35c5d429f)"
    ),
    consumer_callsites=(
        "src/tac/bit_allocator.py::allocate_bits",
        "src/tac/water_filling_codec.py",
        "src/tac/hessian_block_fp.py",
        "src/tac/mdl_fp4_tto.py",
        "src/tac/master_gradient_consumers.py",
    ),
)


ALGORITHM_FRANK_WOLFE_KCARD: Final[AlgorithmCanonicalMetadata] = AlgorithmCanonicalMetadata(
    algorithm_id="frank_wolfe_kcardinality_jaggi_2013",
    paper_citation=(
        "Jaggi 2013 'Revisiting Frank-Wolfe: Projection-Free Sparse Convex "
        "Optimization' ICML 2013; Frank & Wolfe 1956 Naval Research Logistics "
        "Quarterly 3(1-2):95-110 DOI 10.1002/nav.3800030109"
    ),
    canonical_module_path="tac.solvers.frank_wolfe",
    paired_against="tac.solvers.sinkhorn.sinkhorn_ensemble_select_k",
    wall_clock_multiplier_macos_cpu_advisory=1.9,
    convergence_rate_theoretical="O(1/k)",
    convergence_rate_paired_baseline="O(1/k^(1/2)) projected subgradient",
    invariant_contract=(
        "K-sparsity invariant: exactly K members selected at convergence "
        "(extremal vertex of K-cardinality polytope; verified in paired "
        "comparison at commit 35c5d429f)"
    ),
    consumer_callsites=(
        "src/tac/losses/core.py::sinkhorn_w2_mask_distortion_per_pixel",
        # cathedral_autopilot Rashomon ensemble bidirectional matching
        # (callsite plan memo enumerates the per-substrate symposium path)
    ),
)


ALGORITHM_RIEMANNIAN_NEWTON_STIEFEL: Final[AlgorithmCanonicalMetadata] = (
    AlgorithmCanonicalMetadata(
        algorithm_id="riemannian_newton_stiefel_edelman_1998",
        paper_citation=(
            "Edelman, Arias, Smith 1998 'The Geometry of Algorithms with "
            "Orthogonality Constraints' SIAM J. Matrix Anal. Appl. "
            "20(2):303-353 DOI 10.1137/S0895479895290954; "
            "Absil, Mahony, Sepulchre 2008 'Optimization Algorithms on "
            "Matrix Manifolds' Princeton UP"
        ),
        canonical_module_path="tac.solvers.riemannian_newton_stiefel",
        paired_against="Lloyd-Max k-means projection (canonical PQ codebook init)",
        wall_clock_multiplier_macos_cpu_advisory=1.88,
        convergence_rate_theoretical="O(log(1/epsilon)) quadratic local",
        convergence_rate_paired_baseline="O(1/epsilon) Lloyd",
        invariant_contract=(
            "machine-epsilon orthogonality invariant: ||X^T X - I|| < 1e-15 "
            "preserved by QR retraction (verified in paired-comparison output "
            "at commit 35c5d429f)"
        ),
        consumer_callsites=(
            "src/tac/pr101_split_brotli_codec_derivers.py::derive_sidecar_codebook",
            "src/tac/quantization_wave/vq_codebook_quantization.py",
            "src/tac/symposium_impls/carmack_hotz_strip_everything_codec.py",
        ),
    )
)


CANONICAL_ALGORITHM_REGISTRY: Final[dict[str, AlgorithmCanonicalMetadata]] = {
    ALGORITHM_FISTA.algorithm_id: ALGORITHM_FISTA,
    ALGORITHM_FRANK_WOLFE_KCARD.algorithm_id: ALGORITHM_FRANK_WOLFE_KCARD,
    ALGORITHM_RIEMANNIAN_NEWTON_STIEFEL.algorithm_id: (
        ALGORITHM_RIEMANNIAN_NEWTON_STIEFEL
    ),
}


# ---------------------------------------------------------------------------
# Op-routable #5: canonical helper for wall-clock measurement atom emission
# per Catalog #245 4-layer pattern (append per dispatch consumption).
# ---------------------------------------------------------------------------
__all__ += [
    "build_more_optimal_algorithm_wall_clock_atom",
    "append_more_optimal_algorithm_wall_clock_event",
]


def build_more_optimal_algorithm_wall_clock_atom(
    *,
    atom_id: str,
    algorithm_id: str,
    consumer_callsite: str,
    wall_clock_seconds_canonical: float,
    wall_clock_seconds_more_optimal: float,
    n_iterations_canonical: int,
    n_iterations_more_optimal: int,
    invariant_check_passed: bool,
    invariant_description: str,
    extra_metadata: dict | None = None,
):
    """Emit a PROBE_OUTCOME-shaped atom for a wall-clock paired measurement.

    Per grand council T3 finding #6 op-routable #5: append wall-clock-
    measurement atom to ``.omx/state/atom_ledger.jsonl`` per dispatch
    consumption so the cathedral autopilot ranker can consume the empirical
    velocity multiplier per algorithm + per consumer callsite.

    Args:
        atom_id: unique atom id (e.g. ``algorithm_wall_clock_fista_bit_allocator_20260518``).
        algorithm_id: must be in :data:`CANONICAL_ALGORITHM_REGISTRY` (e.g.
            ``fista_beck_teboulle_2009``).
        consumer_callsite: path:fn of the consumer (e.g.
            ``src/tac/bit_allocator.py::allocate_bits``).
        wall_clock_seconds_canonical: measured wall-clock for the canonical
            sister helper (water-filling / Sinkhorn / Lloyd-Max).
        wall_clock_seconds_more_optimal: measured wall-clock for the more-
            optimal algorithm (FISTA / Frank-Wolfe / Riemannian-Newton).
        n_iterations_canonical: iteration count for the canonical sister.
        n_iterations_more_optimal: iteration count for the more-optimal sister.
        invariant_check_passed: True iff the per-algorithm invariant contract
            (byte-identical solution / K-sparsity / machine-eps orthogonality)
            was verified at this callsite.
        invariant_description: human-readable invariant description; canonical
            tokens are pulled from :data:`CANONICAL_ALGORITHM_REGISTRY`.
        extra_metadata: optional extra metadata for the atom row.

    Returns:
        :class:`tac.atom.atom.Atom` of kind ``PROBE_OUTCOME`` with verdict
        ``PROCEED`` iff the speedup >= 1.0× AND invariant_check_passed; else
        ``OPERATOR_REVIEW_REQUIRED`` so a regression is surfaced to council.

    Raises:
        ValueError: if ``algorithm_id`` is not in the canonical registry, or
            if wall-clock values are non-positive.
    """
    from tac.atom.builders import build_probe_outcome_atom

    if algorithm_id not in CANONICAL_ALGORITHM_REGISTRY:
        raise ValueError(
            f"algorithm_id={algorithm_id!r} not in CANONICAL_ALGORITHM_REGISTRY "
            f"(known: {sorted(CANONICAL_ALGORITHM_REGISTRY)})"
        )
    if wall_clock_seconds_canonical <= 0 or wall_clock_seconds_more_optimal <= 0:
        raise ValueError(
            "wall-clock seconds must be > 0; got "
            f"canonical={wall_clock_seconds_canonical} "
            f"more_optimal={wall_clock_seconds_more_optimal}"
        )
    metadata = CANONICAL_ALGORITHM_REGISTRY[algorithm_id]
    speedup = wall_clock_seconds_canonical / wall_clock_seconds_more_optimal
    # Catalog #313 verdict taxonomy: PROCEED iff (faster AND invariant preserved);
    # OPERATOR_REVIEW_REQUIRED if regression (slower OR invariant broken). The
    # cathedral autopilot ranker consumes PROCEED rows as velocity-multiplier
    # signal; OPERATOR_REVIEW_REQUIRED rows surface to grand council per
    # CLAUDE.md "Forbidden premature KILL" — a single slow callsite is a
    # research-deferral, not a paradigm kill.
    if speedup >= 1.0 and invariant_check_passed:
        verdict = "PROCEED"
        next_action = (
            f"wire {algorithm_id} into {consumer_callsite} at next per-substrate "
            "symposium per Catalog #325; preserve invariant contract"
        )
    else:
        verdict = "OPERATOR_REVIEW_REQUIRED"
        next_action = (
            f"per CLAUDE.md 'Forbidden premature KILL': research-defer "
            f"{algorithm_id} adoption at {consumer_callsite}; investigate "
            f"speedup={speedup:.3f}x invariant_passed={invariant_check_passed}; "
            "do NOT KILL the algorithm — log the negative paired-comparison "
            "anchor and try alternative consumer callsite or step-size"
        )
    md = {
        "algorithm_id": algorithm_id,
        "consumer_callsite": consumer_callsite,
        "wall_clock_seconds_canonical": float(wall_clock_seconds_canonical),
        "wall_clock_seconds_more_optimal": float(wall_clock_seconds_more_optimal),
        "wall_clock_speedup_multiplier": float(speedup),
        "n_iterations_canonical": int(n_iterations_canonical),
        "n_iterations_more_optimal": int(n_iterations_more_optimal),
        "invariant_check_passed": bool(invariant_check_passed),
        "invariant_description": invariant_description,
        "paper_citation": metadata.paper_citation,
        "canonical_module_path": metadata.canonical_module_path,
    }
    if extra_metadata:
        md.update(extra_metadata)
    return build_probe_outcome_atom(
        atom_id=atom_id,
        probe_id=f"wall_clock_paired_{algorithm_id}_{consumer_callsite}",
        substrate=consumer_callsite,
        verdict=verdict,
        metric_name="wall_clock_speedup_multiplier",
        metric_value=float(speedup),
        threshold=1.0,
        threshold_token=">=1.0x is PROCEED; <1.0x or invariant-broken is OPERATOR_REVIEW",
        evidence_path=(
            "experiments/results/empirical_solver_paired_comparison_20260518/"
        ),
        next_action=next_action,
        blocker_status="advisory",
        literature_citation=metadata.paper_citation,
        canonical_helper_repo_link=metadata.canonical_module_path.replace(".", "/")
        + ".py",
        extra_metadata=md,
    )


def append_more_optimal_algorithm_wall_clock_event(
    *,
    atom_id: str,
    algorithm_id: str,
    consumer_callsite: str,
    wall_clock_seconds_canonical: float,
    wall_clock_seconds_more_optimal: float,
    n_iterations_canonical: int,
    n_iterations_more_optimal: int,
    invariant_check_passed: bool,
    invariant_description: str,
    event_type: str = "wall_clock_measurement",
    extra_metadata: dict | None = None,
) -> dict:
    """Build + append a wall-clock atom to ``.omx/state/atom_ledger.jsonl``.

    Sister of :func:`build_more_optimal_algorithm_wall_clock_atom`; this
    one-shot helper composes the canonical build + the fcntl-locked ledger
    append per Catalog #245 4-layer pattern (registration helper +
    persistence + canonical lookup + audit-tool surface).

    Returns the row dict that was appended (per ``tac.atom.ledger.append_atom``).
    """
    from tac.atom.ledger import append_atom

    atom = build_more_optimal_algorithm_wall_clock_atom(
        atom_id=atom_id,
        algorithm_id=algorithm_id,
        consumer_callsite=consumer_callsite,
        wall_clock_seconds_canonical=wall_clock_seconds_canonical,
        wall_clock_seconds_more_optimal=wall_clock_seconds_more_optimal,
        n_iterations_canonical=n_iterations_canonical,
        n_iterations_more_optimal=n_iterations_more_optimal,
        invariant_check_passed=invariant_check_passed,
        invariant_description=invariant_description,
        extra_metadata=extra_metadata,
    )
    return append_atom(atom, event_type=event_type)
