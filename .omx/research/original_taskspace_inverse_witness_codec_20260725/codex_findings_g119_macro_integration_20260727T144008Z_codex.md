# G119 macro-integration adversarial review

Date: 2026-07-27  
Axis: code review plus `[macOS-CPU structural timing/equality only]`  
Reviewed commit: `f6cbd2a6f4`  
Verdict scope: G119 implementation and its direct G110, G112, G121, V10, and
`upstream/evaluate.py` contracts.

## Authority and goal honesty

This unit did **not** run a full n600 authority evaluation, produce a promotable
archive, or move the frontier pointer. It is implementation hardening only.
The exact competitive target remains the canonical pointer's current
`effective_frontier`; this review makes no numeric frontier claim.

## Findings, ordered by severity

### P0 fixed: resume and candidate receipts self-attested physical results

The committed G119 resume path trusted persisted optimizer outputs and losses,
and the candidate reopen path trusted self-hashed `pose_mse`, archive bytes, and
archive SHA. A syntactically valid receipt could therefore claim a physical
state that had not been reproduced from the stored integer XIP2 coordinates.

The resume loader now reconstructs all 600 Y0 frames from the stored `q` and
scales, runs the exact chronological batch-16 public PoseNet path, and compares
the persisted outputs and per-pair losses. The candidate loader additionally
rematerializes the complete public archive matrix and recomputes Pose MSE,
Pose term, rate term, packet/archive hashes, and selection fields. Reopen fails
closed on any mismatch.
The composed G110 handoff additionally requires caller-supplied expected
SHA-256 values for both the final checkpoint and run receipt, preventing a
coherent rewrite of both self-described artifacts.

### P1 fixed: the archive matrix and selected-coder ABI were incomplete

G119 and the official G110 compiler hardcoded XIP2 `delta_ar`, even though the
public G110 decoder supports both raw coder `none` and `delta_ar`. G119 now
measures the complete:

`2 semantic Y1 codecs × 2 XIP2 coders × 2 outer ZIP methods = 8`

wire candidates on every materialization. It records exact bytes and SHA for
every candidate, selects the true global winner, and carries its coder through
the final strict checkpoint and run receipt. The candidate reopen accepts
either public coder and remeasures the same global selection. G119 emits a
positive selected-coder ABI closure receipt; G110 consumes this field and must
serialize that exact coder.

### P1 fixed: G119 aggregation did not exactly match upstream float32 chronology

The prior implementation used NumPy float64 residuals and population means.
Upstream evaluates each batch in Torch float32 and accumulates chronological
batch sums in float32. G119 now computes per-pair Pose loss and the n600
population mean using the same float32 Torch chronology. Final authority still
requires `upstream/evaluate.py` on the exact archive; this only closes the
internal optimizer/oracle arithmetic mismatch.

### P1 fixed: retained G115 terminal QAT could be rejected despite exact closure

The G121 retention contract permits a base public-wire obstruction to be
superseded by an exact terminal G115 QAT row. G119 previously demanded that the
base row itself be strict-open. It now accepts the terminal row only when its
measured terminal identity equals the population row's physical-stage identity,
the terminal receipt carries strict physical binding, and terminal exact `k` is
open. Otherwise it fails closed.

### P1 bounded, not solved: fixed-range XIP2 search is not global range optimization

G119 still searches integer coordinates under initializer-derived
per-dimension scales and fixed global extremum anchors. That is a legitimate
formulation, not proof of a globally cheapest XIP2 representation. Receipts now
preserve the scope and the required reactivation: enumerate deterministic
per-channel range/reanchor proposals, refit every shipped coordinate through
the same exact PoseNet loop, and arbitrate complete archive bytes.

## Exact geometry and evaluator correspondence

The reviewed implementation uses the upstream 600-pair chronology
`37 × 16 + 1 × 8`, renders `[Y0, Y1]`, uses the public G110 receiver path, and
feeds PoseNet through the upstream RGB-to-YUV6 preprocessing. The source target
is the first six frozen PoseNet outputs under the G109/G112 custody chain.
SegNet remains a last-frame Y1 obligation and is not recomputed by the Y0-only
pose refit. These correspondences are structural; only a public recursive exact
evaluation can promote a row.

## Static work and measured structural timing

G119 now emits a deterministic pre-launch work estimate:

- Pose batch forwards:
  `Q × 38 × (1 + stages × (12 + line_search_count))`.
- Complete archive materializations:
  `Q × (stages × 38 × (2 × line_search_count + 2) + 1)`.
- Each materialization performs eight public-wire archive builds.

At the smallest representable `Q=1, stages=1, line_search_count=1`, this is 532
Pose batch forwards, 153 complete archive materializations, and 1,224 outer ZIP
builds. At the configured maxima `Q=8, stages=8, line_search_count=8`, it is
48,944 Pose batch forwards, 43,784 materializations, and 350,272 outer ZIP
builds.

An actual first chronological batch from `upstream/videos/0.mkv` measured
1.3616665829904377 seconds per public PoseNet forward on this macOS CPU with
eight Torch threads. That projects the Pose-only portion to about 12.1 minutes
at the minimum and 18.5 hours at the maxima on this advisory machine. This is a
capacity warning, not a score or contest-runtime claim. The governed producer
must record same-device exact timing before choosing a launch configuration.

The public decoder's real source-frame camera warp measured a median
0.077272959 seconds here; zero displacement was bit-identical, and a nonzero
small displacement matched the repository reference exactly. A linear
600-frame projection is about 46 seconds for this operation. Again, this is
structural decoder evidence, not a 30-minute recursive evaluation proof.

## Tests and integration evidence

- Focused G119 + G110 generated-product tests: `26 passed`.
- G121 + G119 + G112 + V10 + G110 generated-product integration tests:
  `59 passed`.
- G110 generic public-receiver tests: `16 passed`.
- Ruff, `py_compile`, and `git diff --check`: passed on the final owned diff.

The tests cover upstream-equivalent float32 aggregation, the exact eight-way
wire matrix, delta-compatible selection, and terminal G115 retained-child
acceptance with physical identity binding.

## Next exact gate

1. Produce or select a real G121 retained n600 population under G120-v2 custody.
2. Run G119 through the governed resumable launcher with same-device timing and
   the complete eight-way wire matrix.
3. Verify G110 consumes the selected XIP2 coder from both strict G119 bindings
   and reproduces the selected public packet/archive bytes.
4. Compile with G110, parse back, double-decode, prove exact output-video
   custody, and run recursive `upstream/evaluate.py` on the exact archive.
5. Only that row can say whether the canonical frontier moved.

## Stores consulted

`CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, the canonical task-status and lane
surfaces, G110/G112/G119/G121 implementations and tests, the V10
selected-preimage realization, public decoder plugin, and upstream evaluator,
dataset, PoseNet, and SegNet preprocessing.
