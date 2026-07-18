# Codex findings — V10 factor 2 bounded uint8 lattice — 2026-07-18

## Verdict scope and pointer

Lane `lane_v10_uint8_lattice_20260718` is a local advisory SOLVE lane with
`training_bytes=0`. The authority surface is a deterministic six-pair subset,
decoded uint8, canonical `A`, and frozen CPU Torch SegNet on macOS. It is not an
archive score, Pose result, contest-axis result, n600 result, or promotion
receipt. Pointer `0.1910828242 [contest-CPU Linux x86_64]` is **UNMOVED**.

## Premise falsified before implementation

The initial `2^4` adjacent-corner proposal around one real preimage was not a
complete integer-lattice solver. Integer coordinates far from that reference
can cancel through the rational resize weights. Calling corner exhaustion
infeasibility would have been a false certificate.

The corrected form derives exact half-pixel integer taps and solves each
disjoint `2 x 2` block as a bounded Diophantine equation with suffix range and
gcd pruning. This makes exact finds and exhaustive affine-block negatives
auditable. Node-budget exits and nonlinear hard-oracle stalls remain unknown.

## Independent review findings and extinction work

The first adversarial pass was **NOT CLEAN** and reset the seal count to `0/3`.
It found:

1. a public `reference` parameter could copy a hidden source frame;
2. an inconsistent oracle could claim `satisfied=true` with a negative margin;
3. one repeated proposal could terminate a repair before another proposal won;
4. proof status and fallback candidate provenance were conflated;
5. hard-oracle unit tests used toy thresholds rather than a real decoded-uint8
   `A` plus frozen-SegNet canary;
6. public status names diverged from the frozen implementation contract.

All six were repaired. The source reference is now only an integrity assertion
equal to the internally derived minimum-norm preimage and never controls the
candidate. Proof status and fallback provenance are separate, repair proposals
exhaust admissible unseen moves before cycle termination, and inconsistent
hard flags/margins refuse. The first repaired surface passed 19 focused tests,
including the real frozen-SegNet canary, and its hash-bound n6 receipt reproduced
the metrics.

A fresh full-landing pass was also **NOT CLEAN**. It found the public exact
block solver could silently truncate non-integral coefficients/denominators or
admit NaN targets/tolerances, producing a false certificate for an undefined or
different equation. It also caught a temporary equation candidate that asserted
universal existence rather than defining feasibility conditionally. Exact
scalar inputs now reject booleans, non-integral types, non-finite values, and
negative tolerance; direct operator/support construction revalidates canonical
geometry. The equation is now an iff feasibility predicate. That then-current
surface passed 35 focused tests and reproduced the n6 metrics, but that was not
the end of the adversarial closure.

The next full pass caught one documentation-custody leak: the Markdown receipt
retained the prior run's free-space preflight value while citing the then-current
pre-wrapper receipt JSON. The value was corrected from that receipt JSON, and
the seal again reset to zero. No stale-run number is allowed merely because the
scientific metrics match.

PASS 2 and an expanded public-API audit then found adjacent numeric-custody
gaps: caller-widened or extreme tolerance, boolean/fractional repair caps,
out-of-lattice and int64-overflow inputs to exact `A`, coercible complex/string
arrays, changing oracle obligation shapes, unbounded adjacent-corner arity,
serializer/parser size asymmetry, mutable certificate arrays, non-finite
minimum-norm output, and overflowed aggregate oracle debt. Each class now fails
closed. Certificate arrays use immutable bytes backing; exact search is capped
to the four factor-2 taps before fallback allocation; float/rational agreement
uses an authoritative-target-only finite ULP bound. The hardened module SHA is
`5039902d8de5...`.

A later wrapper review was also **NOT CLEAN**. It showed that `--resume` could
reuse edited pair metrics, import a scorer module from an earlier `sys.path`
entry while hashing the pinned path, admit a byte-distinct aggregate encoding,
and preflight only the sidecar filesystem when stages lived elsewhere. The
wrapper now re-derives every completed pair from frozen source/cache, exact
`A`, deterministic solve, preserved stage, and frozen SegNet; no stored metric
field flows through. Executed `modules.py`/`frame_utils.py` paths are checked,
aggregate payload hashes bind to stages, and sidecar/stage filesystems are
preflighted independently, including an existing stage-directory mount point.
The final tool SHA is `51103ef9a97f...`; 99 focused tests pass. A final-tool
pair-90 fresh/resume smoke reproduced identical scientific/stage rows and
reported one pair re-derived with stored metric reuse false (resume receipt
SHA `141ecf637173...`).

## Independently re-derived positive evidence

Across the failed and repaired passes, independent probes re-derived exact
rational resize parity, disjoint support, thousands of tiny brute-force DFS
reachability/status comparisons, 40 sampled disjoint geometries, parser fuzz
and maximum-size roundtrip, all six decoded sidecar frames through frozen CPU
SegNet, and the resume/source-custody smoke. Those positive clears do not erase
the historical findings or count as the final three-pass seal by themselves.

The final landing receipt is SHA `665ce8ecd789...`, bound to solver
`5039902d8de5...`, tests `c4a532f7ba5c...`, and tool `51103ef9a97f...`.

## Disposition

The family remains open. Full nonlinear MILP is the wrong certificate for a
nonlinear frozen-CNN preimage; randomized/corner rounding is heuristic;
lattice-Dykstra has no applicable global convergence guarantee. Exact rational
bounded DFS is the affine certificate form, while decoded uint8 through the
frozen scorer is the separate winner-cell verdict.

No Fourier construction, paid dispatch, heavy n600 scorer pass, sacred-run
mutation, pointer mutation, or canonical equation registration occurred.

## Stores consulted

`CLAUDE.md`; `AGENTS.md`; v7.5 and v8 canonical specs; V10 source-of-truth and
completeness/factorization memos; operating manual; `PROGRAM.md`; vehicle OS;
fresh-eyes contract; latest sister findings/design/council surfaces;
`reports/latest.md`; canonical pointer, lane, subagent, probe, equation, and
dispatch state; live per-arm and broadcast inboxes; implementation, tests,
measurement tool, and frozen upstream scorer paths.

MAIN landing review is mandatory even after the branch-local three-pass seal.
