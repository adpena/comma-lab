# G95 — population-shared conditional Pose preimage chart

Status: **frozen build specification; research-only bounded current-state gate**  
Date: 2026-07-27  
Lane: `lane_g95_population_pose_inverse_control_20260727`  
Parent dependency: committed G94 at `9e84c69b8a389337270b70fd4023a4174ef3c552`  
Competitive target: strict live recomposition from
`.omx/state/canonical_frontier_pointer.json`; no score literal is frozen here.

## Outcome and authority boundary

Build the smallest real decoder-control chart that can answer the first G95
question on the exact current G94 state:

> With final Y1 held byte-identical, can a counted, population-shared
> conditional Y0 chart drive the frozen PoseNet output toward the source target
> after exact uint8 receiver replay?

The current G94 state is executable but its G89 and G88 operands are typed
fixtures, not final learned semantic Y1 and conditional Y0. Its exact
conditioning identity is:

`7ab4829d0ecf53b973629be518cc0be575cf826f8a33eceffcb13cb00d678c9b`

Every G95 wire, fit checkpoint, replay row, and receipt MUST bind that exact
hash. Because the state is not final semantic Y1, every result is
`research_only=true`, `candidate_claim=false`, `score_claim=false`,
`promotion_eligible=false`, and `pointer_moved=false`. A useful bounded result
is reachability evidence only. It cannot transfer a Pose marginal to a future
G94 state with a different conditioning hash.

## Settled evidence and forbidden reopenings

- G91 source-Pose trajectory is initializer/factorability telemetry only. It is
  never an inverse-warp control. The current G91 seam is already infeasible
  from Seg plus rate alone (`2.8351` before Pose), and even its non-executable
  per-pair oracle over PASS plus four fixed-warp treatments has `d_pose=0.908`.
- The affine xi-to-Pose map is settled negative (`R2=-0.215`).
- Fixed rendered carrier families are settled at approximately
  `d_pose=1.486` (6D) and `1.223` (12D).
- True-depth, plane-plus-parallax, and post-hoc warp families did not cross the
  appearance wall.
- A dense/full-pixel inverse solve is a valid encode-side reachability teacher
  and a rate-dead payload. Optimized pixels, scorer weights, target Pose rows,
  and GT-derived per-pixel tables may not cross the receiver boundary.
- G95 does not run another fixed-warp arm and does not call G91 xi a solved
  control.

## Canonical-vs-unique decision

Canonical mechanisms reused:

- G94 exact pre-G88 state custody and chronological receiver order;
- frozen CPU-Torch PoseNet, exact first-six output, and target cache custody;
- differentiable YUV6 patch that is forward-equivalent to the evaluator;
- V10-style encoder-only Jacobian/VJP and damped natural-gradient mechanics;
- strict typed wire, parse/re-encode identity, section hashes, CRC, and
  deterministic NumPy receiver replay;
- automatic SSD preflight, immutable stage checkpoints, and resume.

Unique G95 mechanism:

`COPY(exact final Y1) + upsample(sum_k coefficient[p,k] * learned_basis[k])`

The learned residual basis is the SVD/Gram-pruned union of exact PoseNet VJP
rows measured against the bound Y1-conditioned state. It is shared across the
bounded pair population. Per-pair coefficients select within that shared
chart. Both basis values and coefficients are video/scorer-derived and are
COUNTED. The receiver contains only generic parse, dequantization, linear
combination, bilinear resize, clamp, and round-to-nearest-even code. No scorer,
target, or learned constant is free decoder code.

This is not PH1/ReLU merging. HOPE-style Gram/SVD pruning is allowed only on
the learned linear residual directions themselves. No V9 FiLM
`tanh(sin(...))` merge assumption is used.

## Typed G95 P-once wire

Land:

`src/tac/witness_dsl/taskspace_g95_population_pose_preimage_chart_v1.py`

The strict V1 wire is split into two counted object types. There is no
monolithic per-batch packet.

The population-global basis object owns the exact G94 product and
conditioning SHAs, the exact ordered n600 selector hash, the hash-streamed
whole structural-fixture preconditional population, the full target-table
SHA, PoseNet weights SHA, one canonical whole-state key, grid/rank, the counted
signed-int8 shared basis `[rank,h,w,3]`, and canonical positive fp32 basis
scales `[rank]`. It is counted once.

Each indexed coefficient chunk owns the exact basis-object SHA and whole-state
key, its actual preconditional-camera SHA, selected-target SHA, ordered
contiguous source-pair IDs with count in `[1,16]`, signed-int16 coefficients
`[pairs,rank]`, and positive fp32 coefficient scales `[rank]`. It contains no
basis field or basis bytes. Both object types carry section lengths/hashes,
CRC32, exact EOF, and fixed research-only truth labels.

The generic receiver accepts the exact G94 preconditional uint8 batch plus the
basis object and one chunk, verifies the basis/state/rank references and pair
IDs, copies exact Y1 into Y0, dequantizes the chart, upsamples the
low-resolution residual with a documented NumPy `align_corners=False`
bilinear reference, adds in fp32, clamps to `[0,255]`,
rounds-to-nearest-even, and emits uint8. It MUST prove:

- Y1 is byte-identical;
- deterministic double decode;
- exact output and section hashes;
- no unaddressed pair;
- no batch above 16;
- a conditioning/product/whole-state/chunk-preconditional/basis-reference
  mismatch fails closed;
- an exact production chunk stream covers selectors `0..599` once, in order,
  with no gap, overlap, rank drift, state drift, or basis duplication.

The object pair is a production-quality typed actuator section, but G88/G94 V1 do
not yet have a mode/tag for it. Do not mutate their sealed wires in this lane.
Emit a typed `RicherControlRequestV1` whose missing integration is exactly:

`G88_G94_NEW_TYPED_G95_POPULATION_CHART_MODE_AND_OUTER_ARCHIVE_RACE_OWED`

## Encode-side fitter

Land:

`tools/measure_taskspace_g95_population_pose_preimage_chart.py`

The fitter may load PoseNet weights and GT Pose targets. The receiver module
may not.

For a bounded contiguous population of at most 16 pairs:

1. reopen/reconstruct the exact committed G94 current-state fixture and assert
   the product and conditioning identities before loading the scorer;
2. decode the pre-G88 chronological state and use exact final Y1 as the fixed
   condition;
3. record both `PASS_PRECONDITIONAL_Y0` and
   `COPY_EXACT_CONDITIONAL_Y1` no-op/control Pose rows;
4. patch the evaluator YUV6 transform with its checked differentiable
   equivalent for VJP construction;
5. build shared costate directions from the exact first-six PoseNet Jacobian
   with respect to one low-resolution RGB residual grid, through bilinear
   camera upsample, uint8 STE, evaluator resize, and YUV6;
6. concatenate costates across the bounded population, perform deterministic
   SVD/Gram pruning, fix signs canonically, and quantize the retained shared
   basis to int8 before coefficient fitting;
7. fit per-pair coefficients by damped natural-gradient/LM in the frozen
   quantized basis, quantizing coefficients to int16 at every admitted replay;
8. after every treatment, serialize the exact basis object and coefficient
   chunk separately, parse both back, run the
   NumPy receiver twice, and score the exact replayed uint8 pair with frozen
   CPU PoseNet;
9. admit no proxy-only improvement. The receipt reports gradient-space
   proposal telemetry separately from exact receiver replay.

The first ladder is fixed and small:

- control: no chart;
- rank 6 at 48x64;
- rank 12 at 48x64;
- rank 24 at 48x64.

`d_pose <= 4.7366e-4` is only the derived sufficient Pose coordinate at the
current teacher / 132132-byte seam. It is not a universal pass/fail rule.
It MUST NOT stop coefficient descent or the bounded rank ladder: acceptability
is contingent on the coupled `(d_seg,d_pose,outer archive bytes,exact score)`
surface, so more Pose margin may be worth more than the additional chart bytes.
Every real row must expose that joint feasible surface
`(d_seg,d_pose,outer archive bytes,exact score)`; fields unavailable before
outer-archive integration remain explicit `null` blockers, never inferred.
If rank 24 misses, emit a typed richer-control request
with the exact residual, requested minimum next rank/grid, and formulation
scope `STATIC_SHARED_BASIS_AT_EXACT_G94_CONDITIONING_STATE`. Do not infer the
conditional-chart family dead.

The first governed run uses pair 0 only because final semantic G94 does not
exist. Pair 0 uses the same one-basis-plus-one-indexed-chunk production types.
It is a one-state reachability gate, not population-rate evidence. The exact
n600 structural-fixture stream is custody/mechanism context only and is always
labeled `NON_FINAL_SEMANTIC_Y1`; it is not a real population treatment. The
38-chunk n600 coverage proof is a type test until final-Y1 refit.
If any pair-0 rank crosses the sufficient coordinate, the next mandatory treatment is a fresh
governed contiguous population of at most 16 pairs using one shared basis. A
population miss after a pair-0 crossing is a distinct failure class and MUST
request `Y1_CONDITIONED_SHARED_GENERATOR_OR_FEATURE_MODULATED_BASIS`; it MUST
NOT be collapsed into a request for only more static rank/grid. The module and
runner remain population-shaped and require one shared basis plus
per-pair coefficients for every bounded run.

## Determinism, storage, and resumability

Configuration lives at:

`.omx/research/configs/taskspace_g95_population_pose_preimage_chart_20260727.json`

The default run root is:

`/Volumes/VertigoDataTier/pact/g95_population_pose_preimage_chart_20260727_r1`

Before launch:

- hash all inputs and scorer weights;
- hash the exact receiver module, measurement tool, and config bytes before
  scorer load;
- require the SSD tier and a configured safety reserve;
- set one seed for Python/NumPy/Torch;
- enable deterministic Torch algorithms and fixed CPU thread count;
- refuse MPS and CUDA score authority;
- refuse `/tmp`-class output paths;
- cap pair count and scorer batch size at 16.

Stages are immutable, atomic, and resume-loadable:

1. `stage_00_preflight`;
2. `stage_01_exact_g94_state`;
3. `stage_02_noop_controls`;
4. `stage_03_rank06_48x64`;
5. `stage_04_rank12_48x64`, if reached;
6. `stage_05_rank24_48x64`, if reached;
7. `stage_06_receipt`.

Every completed stage preserves its own population-basis object, coefficient
chunk, coefficients, and receiver
hashes and JSON receipt under a distinct stage path. Periodic iteration
checkpoints are atomic and include enough state to resume coefficient fitting.
Never overwrite an earlier stage. Existing complete immutable stages must be
verified and reused, not rewritten.

Every basis, iteration, and completed-rank resume surface binds one canonical
state key over ordered pair IDs, actual selected preconditional-camera SHA-256,
whole structural-fixture preconditional-population SHA-256, selected target
SHA-256, full target-table SHA-256, PoseNet weights SHA-256, G94 product
SHA-256, G94 conditioning SHA-256, config SHA-256, receiver-module SHA-256,
and measurement-tool SHA-256.
A self-consistent checkpoint from any different pair/config/code/state must
fail reuse.

## Acceptance tests

Land focused tests at:

`src/tac/witness_dsl/tests/test_taskspace_g95_population_pose_preimage_chart_v1.py`

They must cover:

- strict basis/chunk serialize/parse/re-encode identity;
- CRC, EOF, digest, shape, rank, scale, and canonical-float rejection;
- exact G94 product/conditioning/preconditional foreign-key rejection;
- pair-order and batch-16 enforcement;
- deterministic NumPy double decode;
- exact Y1 preservation;
- non-noop Y0 execution from nonzero basis/coefficients;
- round-to-nearest-even behavior and NumPy/Torch bilinear parity on a focused
  small tensor;
- one counted basis object plus indexed chunks, with no basis field in chunks;
- exact 600-row coverage via 38 chunks without basis duplication;
- wrong chunk basis/state/rank reference, gap, overlap, or reorder refusal;
- pair-selector bytes are reported separately from learned-payload bytes;
- dataclass field names are unique;
- forged wrong-state resume custody fails before scorer work;
- basis or coefficient mutation changes its counted object/output identity;
- noop/control rows remain explicit;
- `RicherControlRequestV1` is emitted after the complete ladder misses the
  labeled coordinate;
- truth labels cannot become permissive.

Required focused verification:

```text
pytest -q src/tac/witness_dsl/tests/test_taskspace_g95_population_pose_preimage_chart_v1.py
ruff check <G95 module> <G95 test> <G95 tool>
ruff format --check <G95 module> <G95 test> <G95 tool>
python -m py_compile <G95 module> <G95 tool>
```

## Receipt and observability surface

The durable receipt belongs at:

`.omx/research/original_taskspace_inverse_witness_codec_20260725/g95_population_pose_preimage_chart_receipt_20260727.json`

It records:

- exact git/head observation and dirty-tree qualifier;
- exact G94 product/archive/member and conditioning hashes;
- explicit non-final-semantic-Y1 qualifier;
- source target and PoseNet custody;
- seed, threads, hardware axis, pair IDs, and all checkpoint hashes;
- no-op, each rank treatment, exact d_pose before/after, Pose term, P-once
  basis bytes, coefficient-chunk bytes, total counted bytes, and receiver hashes;
- the labeled sufficient Pose coordinate and whether it was crossed;
- the per-rank joint feasible surface `(d_seg,d_pose,outer bytes,score)`, with
  unavailable coordinates explicit and outer-ZIP score admission owed;
- typed richer-control request if not crossed;
- missing G88/G94 typed-mode and outer-ZIP integration;
- public `inflate.sh`, full-n600, upstream exact eval, and G83 blockers;
- false-authority labels and pointer-delta honesty.
- strict live competitive-target custody, recomposed from the pointer
  constituents; the pointer is reporting state, not fit/decode state.

Six hooks:

1. sensitivity: each retained costate direction has singular value, quantized
   basis hash, pair response, and exact replay delta;
2. Pareto: only exact replay rows, never gradient proposals;
3. bit allocator: P-once basis/chunk bytes and an explicit outer-ZIP delta blocker;
4. autopilot: next action is typed G88/G94 mode integration after reachability,
   or the richer-control request after failure;
5. continual learning: receipt is append-only evidence keyed by conditioning
   hash and formulation;
6. dynamic frontier: no admission; G83 remains false until public full-n600
   same-archive closure.

## No-fake completion condition

G95 is implemented only when the typed P-once basis plus indexed chunk performs the real chart decode,
the exact receiver replay invokes frozen PoseNet on actual uint8 output, and
the receipt distinguishes proposal telemetry from replay authority. Tests
alone are not a Pose result. A missed threshold is an honest scoped negative
plus a typed richer-control request, not a fake success and not a family kill.

The bounded outcome never mutates the competitive pointer. Its receipt binds
whatever strict live target is current when the receipt is written.
