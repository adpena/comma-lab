# DAG FEED — V10 factor 2 bounded uint8 lattice feasibility

- FEED id: `FEED-532-v10-uint8-lattice`
- Date: 2026-07-18
- Lane: `lane_v10_uint8_lattice_20260718`
- Research-only: `true`
- Authority: local advisory build/measurement; no launch, score, rank, or promotion authority
- Pointer: `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**

## Executable producer-to-verdict path

```text
custodied target scorer plane y + exact integer numerator plane T
  -> DisjointResizeOperator (exact half-pixel rational taps)
  -> minimum-norm real preimage B(y) [initializer/diagnostic only]
  -> bounded gcd-pruned Diophantine solve per disjoint 2 x 2 RGB block
  -> uint8 camera frame + block proof/candidate-provenance diagnostics
  -> U8LF serialize -> bounded parse-back -> decoded uint8
  -> exact A residual and numerator-equality check
  -> frozen CPU Torch SegNet hard argmax
  -> advisory subset receipt
  -> inverse-solve completeness factor 2: HAVE (advisory primitive) / PARTIAL
```

The producer is `src/tac/optimization/uint8_lattice_feasibility.py`. The local
measurement consumer is `tools/measure_uint8_lattice_feasibility.py`. The
machine-readable receipt and its human-readable companion are the only verdict
consumers in this lane. No trainer, launcher, receiver, shared DAG, canonical
equation registry, or frontier pointer consumes the result.

## Exactness boundary

For each scorer pixel and channel, exact half-pixel resize supplies integer
coefficients `c` and common denominator `D`. The certified affine problem is

```text
find z in {0,...,255}^4 such that c^T z = T.
```

Suffix bounds and gcd congruences prune a finite deterministic search. A found
point is `FEASIBLE_EXACT`. Exhausting the full bounded tree is
`INFEASIBLE_EXHAUSTIVE` for that affine block only. A node-limit exit is
`NOT_FOUND_BUDGET`; it is not infeasibility. Candidate provenance is orthogonal
to proof status, so a heuristic returned after a proof/budget failure cannot
masquerade as an exact frame.

Certificate inputs are fail-closed: exact integer fields cannot be booleans or
silently coerced floats/strings, targets and tolerances must be finite, tolerance
must be non-negative, and directly constructed operator supports are checked
against re-derived canonical geometry. Search arity is capped at the four taps
the factor-2 derivation proves; the rational/float match bound is fixed from the
authoritative rational target and cannot be widened by a caller. Exact-`A`
inputs stay inside `[0,255]`, overflow/non-finite aggregate diagnostics refuse,
certificate arrays are irreversibly read-only, and serializer/parser byte caps
are symmetric.

The nonlinear winner-cell oracle is a separate hard decoded-uint8 surface.
Integer repair accepts only fresh hard-oracle improvements. Stall and cycle
statuses remain instance/budget scoped and never kill the representation
family.

Receipt resumability is also fail-closed: every completed checkpoint row is
re-derived from its frozen source, preserved stage payload, exact `A`, and
frozen CPU SegNet before reuse; edited metrics cannot be promoted by preserving
only a frame hash. The executed scorer modules must resolve to the same pinned
paths whose bytes the receipt hashes.

## Algorithm-form disposition

| form | disposition | reason |
|---|---|---|
| full frozen-SegNet MILP | rejected for this primitive | the frozen camera-to-feature map is nonlinear; an affine-head MILP would certify a different problem |
| randomized or adjacent-corner rounding | heuristic proposal only | neither enumerates the bounded integer preimage and neither can prove feasibility or infeasibility |
| lattice-Dykstra | initializer/proposal family only | nonconvex uint8 plus frozen CNN invalidates classical convex convergence claims |
| exact rational gcd-pruned bounded DFS | selected certificate form | disjoint resize supports reduce the affine problem to independent four-variable bounded Diophantine equations |

## Six-hook wire-in / explicit non-wire-in

This FEED is `research_only=true`; the absent live hooks are explicit rather
than silently orphaned.

1. **Sensitivity map:** the receipt records clip-round failure and recovery by
   class. No master-gradient row is appended because no score-authority axis or
   marginal byte intervention was measured.
2. **Pareto constraint:** the exact candidate closes Seg cells on the selected
   subset, but its raw incremental sidecar is far too large to establish a
   Pareto improvement. It is inadmissible until a receiver-closed rate point and
   Pose interaction exist.
3. **Bit allocator:** no allocator hook is activated. The sidecar bytes are a
   measured upper-bound warning, not a content-coder curve.
4. **Cathedral/autopilot:** dispatch is disabled. Full n600 must enter through
   the governor and only after MAIN adopts the primitive.
5. **Continual learning:** the typed solver, regression tests, receipt, equation
   candidate, and factor-2 matrix delta preserve the reusable signal. No
   posterior or promotion ledger is mutated by advisory evidence.
6. **Probe disambiguator:** exact DFS versus adjacent-corner search is resolved
   structurally and regression-tested with a distant feasible lattice point.
   Heuristic and exact statuses remain callable as distinct modes; no invented
   empirical arbitration is needed for certificate authority.

## Downstream debts

- full n600 frozen-SegNet replay via the governed launcher;
- PoseNet and both-frame interaction through the same `A`;
- a complete counted receiver/archive and inflate parse-back;
- measured rate benefit rather than a raw 11 MB-class sidecar;
- exact contest-CPU and contest-CUDA evaluation on identical archive bytes;
- independent MAIN landing/adoption review.

Until every debt is closed, the feed may change factor 2 only from `MISSING` to
`HAVE (advisory local primitive)` with strict certificate `PARTIAL`. It cannot
authorize V10 compilation, launch, score movement, or family closure.

## Triality

- DSL: typed module/config and measurement CLI; argv-inert outside this tool.
- DAG: this FEED.
- Equation: temporary candidate
  `bounded_uint8_resize_preimage_cell_feasibility_v1`; not registered or
  adopted.

MAIN must independently re-open the code, exact receipt, hash custody, scorer
path, factor disposition, and all remaining blockers before landing.
