# Special-q / lattice-sieve crosswalk for bounded uint8 preimages

`2026-07-20T18:24:04Z` · `research_only=true` · `$0 local` · no training/provider dispatch · no promotion authority

Authority: `codex_delegate:latticesieve_crosswalk:20260720T180654Z`  
Verdict scope: mechanism crosswalk plus one SHA-pinned frame-0 basis-reachability measurement  
Pointer delta: **none** — `0.19108 [contest-CPU]` remains unchanged

## Outcome

**ADOPT one narrow mechanism, reject the cryptanalytic analogy as an algorithm.**
Pollard/Franke-Kleinjung lattice sieving, special-q descent, and NFS relation
collection do not transfer to the current four-variable exact cell oracle or to
the scorer/coder objective. Their load-bearing economics are smoothness over a
huge sparse two-dimensional search region; ours are exact feasibility in small
dense boxes followed by a nonlinear hard oracle and non-additive byte cost.

What does transfer is ordinary integer-lattice preprocessing, and the bounded
check found a real omission in #580's integer characterization:

- **DERIVED-EXACT:** the canonical full-real-kernel atom family is a basis over
  the reals, but its integer span is generally a proper, high-index sublattice
  of the exact integer kernel of each 2x2 resize cell. “Does not enumerate
  arbitrary integer combinations” is therefore not the whole gap; some exact
  integer-kernel vectors are not integer combinations of those atoms at all.
- **MEASURED `[Darwin-arm64 CPU advisory]`:** the existing #580 frame-0 count
  reproduces exactly: `841,898 / 2,462,184 = 34.1931390993%` signed-unit-feasible
  canonical directions, including `692,712` blind coordinates.
- **MEASURED `[Darwin-arm64 CPU advisory]`:** replacing each cell's atom
  sublattice by the saturated rank-3 integer kernel and LLL-reducing its basis
  yields `1,672,680 / 2,462,184 = 67.9348090963%` signed-unit-feasible basis
  directions on the same frame. This is a **tighter lower bound for one chosen
  basis**, not the cardinality/dimension of the full bounded intersection and
  not a score or byte result.
- **NO-VERDICT on speed:** the check does not show a reduction in the measured
  `34.7690 s/frame` minimum-description selector. Reduction is a credible
  offline enumeration preconditioner; the exact coder and hard oracle still
  decide admission.

The measured tightening was sent once to the live
`r1b2_mdl_xi0_compile` inbox as authorized. No other arm was messaged.

## Regime boundary — what transfers and what does not

| Property | NFS special-q / lattice sieve | Pact uint8 preimage problem | Transfer verdict |
|---|---|---|---|
| Search set | Large 2D coefficient rectangle inside an index-`q` congruence lattice | Disjoint 2x2 cells (`4` uint8 variables, rank-3 homogeneous kernel) plus globally coupled scorer/coder selection | Structural analogy only |
| Candidate signal | Approximate accumulated `log N(p)` predicts two-sided smoothness | Exact integer numerator equality, then decoded uint8 argmax/coder/hard-oracle evidence | **N-A** |
| Sparsity | Sparse hits of many prime/factor-base sublattices | Dense small bounded boxes; every candidate is cheap enough to check locally | **N-A** |
| Descent | Factor a special ideal into smaller prime ideals until the factor base is reached | No monotone factor-base size, divisibility tree, or recursive relation identity | **N-A** |
| Useful geometry | Reduced lattice basis and efficient traversal of lattice points in a rectangle | Saturated integer-kernel basis and box-aware enumeration | **ADOPT** reduction only |
| Objective | Find enough smooth relations; approximate sieve scores are sufficient | Minimize exact counted description bytes subject to exact receiver/scorer constraints | Existing hard gates are better |

This distinction is binding. Calling residue buckets “special-q descent” would
add a name, not an algorithm: fixing a coordinate or congruence class in our
problem is useful only when an admissible lower bound proves that the excluded
classes cannot beat the current exact coder incumbent.

## Primary mechanisms and source-grounded reading

### Pollard lattice sieve

Pollard's chapter [*The lattice sieve*](https://doi.org/10.1007/BFb0091538)
is the 1993 source (pp. 43–49 in *The Development of the Number Field Sieve*).
For a special pair `(q,rho)`, the NFS search is restricted to the 2D lattice
`a = rho*b (mod q)`, a reduced basis is chosen, and factor-base sublattices mark
points whose polynomial values are likely smooth. The transferable statement is
only “reduce a lattice basis before bounded enumeration.” The prime-factor log
sieve, smoothness threshold, coprimality filter, and relation yield have no
counterpart in `HARD_ACCEPT`.

### Franke-Kleinjung line/lattice sieving

Franke and Kleinjung's primary paper
[*Continued Fractions and Lattice Sieving*](https://www.hyperelliptic.org/tanja/SHARCS/talks/FrankeKleinjung.pdf)
states the special-q lattice explicitly, reduces it (with a weighted scalar
product for skew), and searches a rectangle of `(i,j)` coefficients. Small
factor-base primes are line-sieved row by row; larger-prime lattice hits are
walked with two continued-fraction-derived vectors. The paper's actual gain is
a constant-factor/cache-friendly traversal of sparse hits, not a generic CVP
solver. It even notes that truncated Euclid is often faster than full lattice
reduction for its 2D inner case.

Our four-variable cell DFS already has exact suffix bounds and suffix-gcd
pruning. Replacing it with line/lattice sieving would add initialization and
bookkeeping without removing the exact oracle. For the rank-3 homogeneous
kernel, however, exact saturation followed by cached reduction is directly
useful preprocessing.

### Special-q descent for individual logarithms

The Joux-Lercier-Smart-Vercauteren primary paper
[*The Number Field Sieve in the Medium Prime Case*](https://iacr.org/archive/crypto2006/41170323/41170323.pdf)
and later NFS practice use special-q descent to express an out-of-factor-base
ideal through relations involving progressively smaller ideals. The recursive
measure is norm/smoothness. Pact has no corresponding monotone quantity: coder
bytes can increase after a locally shorter move, and argmax acceptance can flip
discontinuously. Coordinate-class recursion is therefore **N-A unless a Pact
specific exact lower bound is supplied**.

### CADO-NFS implementation economics

The official [CADO-NFS repository](https://github.com/cado-nfs/cado-nfs) and
its [special-q controls](https://github.com/cado-nfs/cado-nfs/blob/master/scripts/cadofactor/README.md)
confirm the operational unit: batches/ranges of prime or composite special-q's,
a large `(i,j)` sieve area, relation throughput, and later cofactorization.
CADO's parameter documentation describes a `2^I x 2^(I-1)`-scale lattice-sieve
area and memory growth by roughly 4x when `I` increases by one. Importing LAS or
CADO machinery for 196,608 independent rank-3 boxes would be category error.

### Reduction, enumeration, and CVP

The official [fplll](https://github.com/fplll/fplll) implementation provides
LLL/BKZ, Fincke-Pohst/Kannan enumeration, SVP, and Euclidean CVP. It is a useful
reference implementation, but #580's rank is only `3` per disjoint cell, so an
exact unimodular completion plus tiny LLL is simpler and auditable. The
[flatter](https://github.com/keeganryan/flatter) implementation targets very
large, high-bit-dimension reductions and is **N-A** here.

For the box itself, fixed-dimension integer programming is the correct theory:
[Lenstra 1983](https://pub.math.leidenuniv.nl/~lenstrahw/PUBLICATIONS/1983i/art.pdf)
proves polynomial solvability for fixed variable count. This supports an exact
box-enumeration/branch-and-bound route; it does not imply that a generic solver
will beat the current specialized four-variable DFS.

The rate objective is not Euclidean CVP. LLL guarantees short/nearly orthogonal
bases for Euclidean geometry, while Brotli/LZMA/archive cost is discontinuous,
context-dependent, and not even a norm. Generalized reduction exists for gauges
of symmetric convex bodies (Lovasz-Scarf,
[*The Generalized Basis Reduction Algorithm*](https://pubsonline.informs.org/doi/10.1287/moor.17.3.751)),
and deterministic approximate CVP is known for broad norm classes
([Dadush-Kun](https://ir.cwi.nl/pub/24656)), but neither result applies directly
to an entropy coder with cross-cell context. LLL is therefore a search
preconditioner; exact compressed bytes remain the objective authority.

### The Drop

The linked LeetArxiv post
[*Special-q Descent and Lattice Sieving for Individual Discrete Logarithms*](https://leetarxiv.substack.com/p/special-q-descent-and-lattice-sieving-931)
publicly exposes its title, date, and summary that it codes Pollard's 1993
paper. The courtesy continuation remained unavailable to both semantic fetch
and the available browser runtime, so no technical claim below relies on its
gated body. The primary paper, Franke-Kleinjung paper, JLSV06, and CADO-NFS
source are the evidence authorities.

## Graded mechanism-to-consumer crosswalk

Grades are deliberately consumer-specific.

| Mechanism | #532/#547 bounded cell IFF | #580 full-kernel projector/compiler | r1b2 min-description selector | #579 ERM/parallel tempering |
|---|---|---|---|---|
| Special-q descent | **N-A** — no recursive factor-base measure | **N-A** — no ideal/norm descent | **N-A (analogy only)** — residue classes need a coder-valid bound | **N-A** — not a deterministic replacement for nonlinear energy search |
| Pollard smooth-relation lattice sieve | **ALREADY-HAVE-BETTER** — exact 4-var gcd/interval DFS | **N-A** as a sieve; no factor base | **N-A** — sparse log marking does not rank exact bytes | **N-A** — approximate smoothness score cannot replace hard terminals |
| Franke-Kleinjung line/vector-pair traversal | **ALREADY-HAVE-BETTER** — direct exhaustive certificate | **N-A** for 2D traversal | **N-A** until profiling shows enumeration traversal, not coder/oracle, is binding | **N-A** |
| Exact saturation + LLL/HNF/SNF preprocessing | **ALREADY-HAVE-BETTER** for feasibility; optional ordering only | **ADOPT** — materially tighter integer reachability basis | **ADOPT-CONDITIONALLY** offline before exact coder/hard oracle; speed unmeasured | **ADOPT-CONDITIONALLY** to seed deterministic local candidates before ERM fallback |
| NFS relation-collection economics | **N-A** | **N-A** | **ALREADY-HAVE-BETTER** — registered Fisher/margin reverse-waterfill and exact score/byte break-even | **ALREADY-HAVE-BETTER** — exact energy/coder/hard-terminal accounting |
| Euclidean CVP/SVP objective | **N-A** | **ADOPT only for basis conditioning** | **N-A as final objective** — coder bytes are not L2 | **N-A as final objective** — scorer/coder energy is nonconvex and nonmetric |

## Bounded exact check — frame 0 of the real #580 fixture

### Custody and method

- Fixture:
  `/Volumes/VertigoDataTier/pact/molab_witness_machine_upstream_20260709/videos/0.mkv`
- Fixture SHA-256:
  `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`
- Decoded RGB24 frame-0 SHA-256:
  `47492a5864f0521f0ab6b129e4b172717139ba202fc3265a5220b6d1b15c24ed`
- Source operator:
  `src/tac/optimization/uint8_lattice_feasibility.py`, SHA-256
  `8f694f06bfd643a598e3f5f9ce768dd73b546f7c5133f34d436b84adea635b17`
- Repo base: `8680c8e4f6ed2fe4186be7d9e60ebb9459183cc7`
- Runtime: `macOS-26.4 arm64`; Python from the host repo venv; PyAV `17.1.0`,
  NumPy `1.26.4`, SymPy `1.14.0`.

For one cell, with exact tap numerators `(ah,bh)` and `(aw,bw)`, define

`c = (ah*aw, ah*bw, bh*aw, bh*bw)`.

Construct a unimodular `U in GL(4,Z)` by successive extended-gcd column
operations so that `c U = (g,0,0,0)`, where `g=gcd(c)`. Then the last three
columns of `U` are an exact saturated Z-basis of `ker_Z(c)`: every integer
kernel vector occurs exactly once as their integer combination. LLL-reduce the
three row vectors and count direction `b` as feasible at byte block `z` iff
`z+b` or `z-b` lies in `[0,255]^4`.

The check covered all `192 x 128 = 24,576` unique row/column tap-lattice types,
all `384 x 512` disjoint scorer cells, and all three channels. It first ran the
old atom formulas; their exact reproduction of #580's three family counts is an
orientation/control check.

### Results

| Direction family | #580 canonical feasible | Saturated-LLL feasible |
|---|---:|---:|
| Kernel basis direction 0 | `74,469` | `469,344` |
| Kernel basis direction 1 | `74,717` | `348,109` |
| Kernel basis direction 2 | `0` | `162,515` |
| Blind coordinate directions | `692,712` | `692,712` |
| **Total** | **`841,898 / 2,462,184`** | **`1,672,680 / 2,462,184`** |
| **Basis-direction lower bound** | **`34.1931390993%`** | **`67.9348090963%`** |

For the first cell,

`c = (103416,181256,182280,319480)` and `gcd(c)=8`.

The canonical atom basis is

```text
(245, 0, -139, 0)
(0, 245, 0, -139)
(22657, -12927, 39935, -22785)
```

Its integer span has index `79,346` in the saturated kernel. LLL on that atom
basis alone cannot repair the omission because LLL preserves the lattice it is
given. Saturate first; then LLL gives

```text
(-7, 7, -10, 4)
(-37, -33, 17, 21)
(-13, 48, 28, -39)
```

All three rows have exact dot product zero with `c`. The saturated basis
covolume is exactly the Euclidean norm of primitive `c/g`, while the atom-basis
covolume is `79,346` times larger; this independently certifies the index.

### Exact scope of the result

This measurement tightens a **basis-direction feasibility lower bound**. It
does not enumerate arbitrary integer combinations of the new basis, count all
points in `ker_Z(c) intersect ([0,255]^4-z)`, prove a smaller archive, or show a
runtime improvement. Those are separate owed measurements. It also does not
change `bounded_uint8_resize_preimage_cell_feasibility_v1`: the existing direct
DFS remains an exact IFF certificate for an affine target.

## Ranked mineables

1. **ADOPT now in #580/r1b2 after MAIN review — exact saturation before
   reduction.** Build the per-cell integer kernel using unimodular
   completion/SNF/HNF semantics, then LLL-reduce. Cache/derive the `24,576`
   geometry-only basis types. This is generic receiver geometry, not
   video-derived payload.
2. **Measure, do not assume — box enumeration with the saturated basis.** For
   each affine cell `z0 + B k`, enumerate `k` under the four uint8 inequalities
   using exact branch-and-bound. Compare node visits and wall time with the
   current specialized DFS at identical candidate/coder/hard-oracle budgets.
   This is the only honest route to a claim against `34.7690 s/frame`.
3. **Coder-aware r1b2 selection — reduction is not the objective.** Use the
   saturated basis to propose short legal moves, but rank/admit with exact
   Brotli/LZMA/archive delta and fresh hard evidence. Preserve the zero-weight
   fill as a control. A per-cell Euclidean nearest vector is not a
   minimum-description preimage.
4. **Deterministic local front end for #579.** Exhaust or bound the small local
   lattice first; invoke ERM/parallel tempering only for globally coupled
   `STALLED/CYCLE/BUDGET` states. This can reduce stochastic search volume, but
   it is not yet a measured replacement.
5. **Clean negative — do not port CADO LAS/special-q relation machinery.** No
   factor base, smoothness relation, huge sparse 2D region, or relation-yield
   objective exists here. Its engineering would not tighten exactness or byte
   authority.

## Triality and integration boundary

- **DSL:** N-A for this research-only crosswalk; no config or flag was added.
  A future implementation should expose an exact typed `integer_kernel_basis`
  mode, not an ad hoc “special-q” switch.
- **DAG:** `#580 exact real projector -> saturated integer cell kernel -> LLL
  search basis -> bounded uint8 candidates -> exact numerator check -> exact
  coder bytes -> fresh hard oracle -> r1b2 archive admission`.
- **Equation:** the existing
  `bounded_uint8_resize_preimage_cell_feasibility_v1` is unchanged. The new
  candidate law is: if `cU=(g,0,0,0)` with unimodular `U`, then
  `ker_Z(c)=U[:,1:4] Z^3`. It is **not registered here**; MAIN must reproduce
  the measurement and decide whether to extend #580's canonical equation.
- **Sensitivity / Pareto / bit allocator:** geometry supplies legal moves only.
  Fisher/margin ranking and the registered reverse-waterfill break-even remain
  the consumers and admission authorities.
- **Cathedral/autopilot:** no dispatch hook and no launch authority. The result
  was routed once to the already-live r1b2 consumer instead of spawning a lane.
- **Continual learning:** this memo is the durable negative/positive split:
  special-q/LAS is N-A; exact saturation plus reduction is the mineable.
- **Missing code integration:** neither this arm nor the #580 base currently
  exposes the saturated integer basis. The live r1b2 arm owns any implementation
  and must preserve exact numerator, coder, receiver-runtime, and hard-oracle
  gates.

## STORES CONSULTED

- `CLAUDE.md` and `AGENTS.md`, read completely; delegated authority file read
  and SHA-256 verified.
- `tools/graph_memory_recall.py` and the live canonical-equation registry;
  `bounded_uint8_resize_preimage_cell_feasibility_v1` and
  `resize_exploit_flip_fix_frontier_v1` were recalled before research.
- #580 committed branch `da64a5bc8e`, especially
  `.omx/research/null_compiler_full_kernel_20260720T163500Z.md` and its receipt.
- R1b committed branch through `af690d74f4`, especially
  `.omx/research/codex_findings_r1b_boundary_generator_solve_20260720_codex.md`.
- Current shared landing/delegation ledgers and the r1b2 build intent; current
  per-arm and broadcast inboxes through `2026-07-20T18:24Z`.
- The primary papers and OSS sources linked above. No live run, score pointer,
  archive, provider, training process, or submission state was mutated.

## MAIN landing review required

MAIN must independently review and reproduce: (1) the unimodular-completion
proof that the basis is saturated; (2) the old-count reproduction and all new
family counts; (3) the first-cell index/covolume calculation; (4) the strict
lower-bound-only language; (5) the non-Euclidean coder caveat; and (6) that no
special-q/LAS, speed, byte, score, or promotion claim leaked through. Any code
adoption belongs in the live #580/r1b2 reconciliation, behind exact numerator,
coder, runtime, and hard-oracle gates. This branch has no MAIN landing authority.
