# C2 integer-plane emitter implementation specification

**Status:** FROZEN FOR BUILD, local/fixture authority only
**Lane:** `lane_c2_integer_plane_emitter_build_20260719`
**Parent:** `.omx/research/SPEC_v10_integer_plane_vehicle_20260719.md` C2
**Pointer:** `0.1910828242 [contest-CPU]` UNMOVED
**Execution:** no training, paid dispatch, score claim, promotion, or pointer authority

This is the pre-code implementation contract required by the arbitrage routing
rule. It narrows C2 into three independently owned production units plus a final
integration/review unit. It does not reopen the settled v10 design.

## Frozen evaluator and math bindings

- The frozen evaluator source is the SHA-pinned public-PR101 tree. Its
  `modules.py:130-161` composes SegNet/PoseNet into `DistortionNet`; its
  `frame_utils.py` implements Pose input as four luma samples plus 2x2-averaged
  U/V. Therefore C2 owns two independently indexed RGB planes and exposes the
  exact 2x2 Pose-visible transform; within-block chroma-only changes are never
  mislabeled as Pose-visible.
- The frozen SegNet head is SHA
  `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`.
  Its centered five rows are rank four. Sign-fixed SVD is derived from those
  bytes, not copied from a tool. The four measured singular values are
  `3.1283763256`, `2.1542713873`, `2.0247078699`, and `1.7962638357`.
  No objective divides by a singular value. All ten unordered class-pair
  differences are recovered together from the same four-dimensional span.
- Exact factor-2 numerator custody uses
  `tac.optimization.uint8_lattice_feasibility.DisjointResizeOperator` and its
  existing realize/verify wrappers. Numerator exactness is not native-f32
  resize/scorer authority. Hard Seg/Pose inference is a separate admission row.
- Structured bases, coordinate features, seeds, closed-form solves, and exact
  lattice projection are immutable inputs. Only the explicitly named quotient
  residual parameter group is learned. Every trainable array must appear in a
  capacity signature, and deleting the residual must reduce the emitter to the
  solved base. There is no learned camera-null field.
- The emitter output stage is the single surgical intervention point. It accepts
  a frozen scorer-plane base plus compact per-pair/per-plane codes and a shared
  residual head, then emits `[N,2,384,512,3]` bytes. Both plane indices are
  explicit. Expansion is deterministic, shape-parallel, and pair-independent;
  cross-pair autoregression is forbidden in C2.
- The operating contract in `docs/operating_manual_craft_handoff.md` governs
  custody, exact labels, and MAIN review.

## Unit A: core emitter, exact lattice bridge, and basis harness

**Owner files**

- `src/tac/boundary_math/integer_plane_emitter.py`
- `src/tac/boundary_math/tests/test_integer_plane_emitter.py`

**Required public surfaces**

1. A frozen geometry/contract object fixes scorer RGB NHWC geometry
   `(384,512,3)`, plane count `2`, and camera geometry `(874,1164,3)`.
2. An immutable structured state holds a float32 base, deterministic frozen
   coordinate basis, and the residual topology. A separate quotient-residual
   state contains every and only trainable array. Deterministic fresh init uses
   one recorded seed and gives independent `[pair,plane]` codes.
3. NumPy-fp32, Torch, and lazy-MLX forwards compute the same precursor and exact
   clip-round bytes. Torch reuses `tac.quantization.Uint8STE`. MLX uses
   `clip(x,0,255) + stop_gradient(round(clip)-clip)` so gradients are zero
   outside the inclusive receiver range. All entrypoints refuse wrong dtype,
   rank, geometry, non-finite values, alias/copy plane collapse when strict
   distinctness is requested, and non-integer scorer handoff.
4. The lattice bridge independently realizes and verifies every `[pair,plane]`
   scorer target. It returns camera preimages, exact numerators/denominators,
   hashes, and per-plane proof rows. No camera-plane rounding shortcut exists.
5. `SignFixedU4Basis` derives from a frozen `(5,16,3,3)` head weight only after
   SHA custody is checked. The sign rule is the first argmax of each right
   singular vector's absolute value being positive. The API exposes centered
   raw four-coordinate targets, U4 coordinates, and the deterministic 10x5
   pair-difference map. It verifies rank, reconstruction, values, signs, and
   forbids sigma normalization.
6. `FixedCapacityBasisAB` shares one emitter state/capacity signature across raw
   and U4/pair-margin objective arms. It may transform frozen-head logits but
   cannot change residual width, topology, parameter/code count, output bytes,
   or hard-oracle request. Capacity mutation is refused while basis verdict is
   unresolved.
7. A hash-bound encode-only VJP guidance record can be attached to proposal or
   trust-region metadata. It has no decoder serialization surface and cannot
   admit a candidate.
8. A deterministic RGB-pair-to-YUV6 helper exposes the frozen 2x2 Pose lattice.

**Proof floor:** at least 20 focused tests across exact NumPy/Torch bytes,
Torch gradients, lazy MLX contract/source behavior, independent planes,
shape/dtype/finiteness refusals, factor-2 numerator proofs, U4 signs/rank/ten
margins/no-division, fixed capacity, pair independence, deletion test, VJP
encode-only custody, and 2x2 Pose visibility. MLX execution is environment-
blocked when no Metal device exists; this must be reported, never inferred from
Torch.

## Unit B: typed DSL and vehicle resume stub

**Owner files**

- `src/tac/witness_dsl/integer_plane_emitter_policy.py`
- surgical factory/import patch in `src/tac/witness_dsl/curriculum_dsl.py`
- `src/tac/witness_dsl/tests/test_integer_plane_emitter_policy.py`

The policy is frozen, default-OFF, and sealed against launch, payment, scoring,
promotion, and pointer mutation. It types basis (`raw_centered` or
`sign_fixed_u4_pair_margin`), STE mode, geometry, residual width, capacity lock,
pair-parallel expansion, and the no-cross-pair-autoregression invariant.
`IntegerPlaneEmitter(*, policy=...)` is a module-level required-argument Lever
factory in `curriculum_dsl.py`, because the canonical registry AST-scans that
module. The lever is argv-inert until a later governed trainer integration;
`BASELINE.with_lever(...).validate()` must be clean and compilation must be
byte-identical to baseline. No invented trainer flag or premature direct resume
controller is allowed.

Four LawRefs resolve the measured U4 values from
`.omx/research/v10_power_diagram_frame195_diagnostic_20260718.json`, artifact
SHA `65d97194c6298a5502d0fcc792ee2fe3bf05599c69f1130d64c270dec5ec36ee`,
using the custodial identity equation and exact JSON paths. Declaration hashes,
not timestamped resolved manifests, enter the deterministic policy hash.

The standalone stage-checkpoint schema contains schema/config hashes, stage
name/index, epoch/global step/next-pair, basis/STE IDs, fixed-capacity signature,
live residual parameters, EMA shadow, optimizer state, RNG state, and hashes for
topology/discrete/event/dual state. Serialization is canonical and
tamper-refusing. Filenames are stage-encoded and non-overwriting. It documents
future `__ipe_` resume-registry hooks without registering a controller that the
trainer does not yet own.

## Unit C: fixture receipt and real-pair advisory smoke

**Owner files**

- `tools/measure_c2_integer_plane_emitter.py`
- `tools/tests/test_measure_c2_integer_plane_emitter.py`

The tool has two explicit modes:

- `fixture`: runs deterministic core parity, gradient, independence, capacity,
  and exact numerator checks and writes a small machine-readable receipt.
- `n24-advisory`: selects the frozen 24 real pair IDs while requiring at least
  six completed hard-oracle rows. It builds a structured base by exact resize
  numerator projection, applies a fresh seeded quotient residual without
  training, realizes both scorer planes through the factor-2 solver, and runs
  the frozen hard Seg/Pose oracle on the realized camera pairs. Results are
  labeled `[macOS-CPU advisory, untrained]`; no contest score, acceptance
  verdict, or pointer movement is computed.

The advisory command has two explicit local-loop twins. `numpy-torch-authority`
uses the NumPy-fp32 emitter/lattice reference and frozen CPU-Torch scorer.
`mlx-metal-iteration` keeps emitter expansion batched over the pair/plane shape,
uses the existing MLX scorer adapter
`tac.local_acceleration.mlx_scorer_adapters.load_mlx_distortion_scorer_adapter_from_upstream`,
and reuses parity-gated #212/custom-kernel surfaces where their exact operator
matches this loop. In particular it may consume grouped-convolution fast paths
and `metal_fused_r_operator` only when its R geometry/semantics match; it must
never relabel a mismatched roundtrip as the exact factor-2 lattice solve. Both
twins report setup seconds, total seconds, completed iterations, and median/p95
seconds per pair iteration. The MLX twin also reports the selected device,
fast-path availability, and exact NumPy parity state. Dev velocity is measured,
not inferred.

The smoke consumes existing hash-bound VJP custody only as encode-side proposal
metadata; it never regenerates VJPs. Primary n6 custody is
`/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/chunk_000_010_024_composed/manifest.json`
with manifest SHA
`3d1218a52ededc4b347ae94c5c2bf58d06d70dd8f530bec67bf9cab36ee00694`;
missing/mismatched custody is a blocker, not a fallback. Source frames/labels and
poses come from the SHA-pinned `gt_n600.npz` cache. The tool records source,
model, cache, VJP, code, seed, pair IDs, geometry, per-plane hashes/numerator
proofs, per-pair d_seg/d_pose, runtime/hardware axis, and the MLX execution
status. Artifacts are small and durable under `.omx/research/`; no `/tmp`
evidence and no sacred-run writes.

Current preflight found MLX importable in the project venv but array evaluation
blocked by `metal::load_device` because this execution environment exposes no
Metal device. Consequently the MLX execution parity/gradient leg must be marked
`BLOCKED_ENVIRONMENT_NO_METAL`; exact NumPy/Torch parity and all non-MLX proof
legs still run. The MLX local-loop twin remains implemented and fail-closed so
MAIN can rerun it on Metal. This is an honest C2 gate gap, not permission to
infer parity or timing.

## Integration and landing gates

1. Run all focused tests and require at least 20 collected C2 tests.
2. Run fixture receipt, then the shortest real hard-oracle smoke that completes
   at least six of the frozen n24 selection without exceeding minutes-class
   scope.
3. Write `.omx/research/c2_integer_plane_emitter_build_20260719_codex.md` with
   `MEASURED`/`DERIVED`/`BLOCKED` labels and pointer-delta honesty.
4. Perform at most five self-review rounds. Every changed Python file receives
   two clean review-tracker passes.
5. Commit only through `tools/subagent_commit_serializer.py` using each file's
   post-edit SHA-256. MAIN must review the complete base-to-head diff and keep
   promotion/training/score authority closed.
