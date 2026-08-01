# G32 specification — production n600 R10 maximum inverse fitter/compiler

Date: 2026-07-26  
Lane: `lane_g32_r10_n600_inverse_fitter_20260726`  
Authority: local build plus bounded real-input verification only; `research_only=true`  
Owner: G32 new disjoint files only; G27/G20/G22/G23/G29 files are frozen

## 1. Outcome and claim boundary

Implement a real encoder-side fitter/compiler for all nine counted physical
sections consumed by the frozen G27 R10 receiver:

1. `PAIR_INDEX`;
2. `GEOMETRY`;
3. `BASE_FEATURE`;
4. `TEXTURE`;
5. `SHOOTING_KNOT`;
6. `XIP2`;
7. `DASH1`;
8. `PULLBACK_POLYGON`; and
9. `STRATIFIED_FLOW`.

The fitter must consume the exact own-lineage G20/G22 ep725 selected archive and
frozen runtime as its base producer and `upstream/videos/0.mkv` as encoder-only
source truth. It must fit executable G27 operands by deterministic analytic or
finite inverse solving before any learned residual is admitted. It must emit
canonical bytes accepted by `parse_r10_packet`, `decode_r10_packet`, and
`build_r10_selected_solution_adapter`; a search log or parameter suggestion is
not an implementation.

This landing must not run a second full-n600 base replay, a full scorer, a
complete exact evaluation, training, remote dispatch, candidate promotion, or
pointer mutation while G28/G14 own live heavy work. A safe real n1/n2 execution
is mechanism evidence only. All d_seg, d_pose, complete candidate ZIP bytes,
and score-unit-per-byte fields remain JSON null until a later matched n600
public-receiver/scorer measurement supplies them.

## 2. Frozen custody

The implementation must fail closed unless all exact dependencies match:

| object | bytes | SHA-256 |
|---|---:|---|
| source video `upstream/videos/0.mkv` | 37,545,489 | `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9` |
| G20 selected archive | 81,027 | `8e9c7ba0fdd1fc0fdff696c639821d6e64a3110bb8744f47ae0ab3d287cd70d8` |
| selected reopened `0.bin` | 81,738 | `4789bf6b5f15272cc5f8a573f25137a9daf7e21755e81aa48a8fba84947b5634` |
| G20/G22 frozen runtime | 56,814 | `4b54d512565f7275c53f697a931dd087222a36a69495b6e536a6b65dede36224` |
| selected decoded state | — | `5485d0d94c5c834e059837e74ae5320fe9d2b526604c47008a6bfdb74144adf6` |
| G22 full-n600 selected uint8 witness | 3,662,409,600 | `8565df10cbff8f86f02233fd20ececd74857a0d3806caf278a385a4d5421dcae` |
| G22 final full-n600 receipt | 24,210 | `3a01e81abfd19a78db86e5851f1b0c453ff553c1fe7d5fad830f95bcd5ec3efd` |
| frozen G27 R10 receiver module | 65,411 | `13cd771d10c333a458c9977f8b21b916a4baf80b063bb4f849f001a6f660e11d` |

The source has exactly 1,200 frames at 20 fps, native 874×1164 yuv420p. The
selected runtime must be reopened through the selected archive member, execute
its real `_setup`/`_render_pair` producer under fixed single-threaded FP64
environment, and yield canonical ordered pairs. G32 may reproduce a selected
base range; it may not treat the receipt hash as substitute pixels.

The base realization used in the packet identity is the complete ordered
selected population. Full population hashing must be streaming and include the
same domain-separated shape prefix as G27 `realization_sha256`; never load the
7.3 GB pair tensor into RAM. Bounded n1/n2 receipts bind only their bounded base
identity and may not be relabelled n600.

## 3. New-file boundary

Implementation may create only:

- `src/tac/witness_dsl/taskspace_r10_n600_maximum_inverse_fitter.py`;
- `tools/fit_taskspace_r10_n600_maximum_inverse.py`;
- `src/tac/witness_dsl/tests/test_taskspace_r10_n600_maximum_inverse_fitter.py`;
- `tools/tests/test_fit_taskspace_r10_n600_maximum_inverse.py`;
- this G32 spec, a G32 findings memo, and a content-addressed G32 receipt/output
  directory beneath the current research root;
- G32 checkpoint, lane registry, and lane audit records through canonical tools.

Do not edit, reformat, export from, or regenerate any G27, G20, G22, G23, G29,
shared package, pointer, or candidate file. Do not commit or push.

## 4. Typed fitting API

The new library must expose explicit immutable types rather than a dict-only
optimizer:

- `R10MaximumInverseConfigV1`: seed, pair population, native dimensions,
  deterministic finite inverse schedules, chunk geometry, source/base custody,
  and authority flags. Config validation forbids NaN, unbounded decoder work,
  scorer/teacher/GT paths, implicit local spill, and full-n600 execution without
  both reviewed confirmation flags.
- `R10SectionFitV1`: one canonical section ID, exact bytes/hash and packet span,
  solver family, sufficient statistics, exact source-domain integer objective
  before/after, realized uint8 change counts, interaction parents, runtime and
  peak-memory telemetry, and nullable authority fields.
- `R10CountedStateManifestV1`: exact packet/archive/member/section identities;
  every video-specific scalar, phase, knot, selector, threshold, exception,
  polygon, flow, trajectory, or correction mapped to a counted byte span; all
  generic decoder mechanisms separately enumerated; all encoder-only evidence
  separately enumerated; no overlapping or uncovered counted bytes.
- `R10G23PhysicalGroupAdapterV1`: exact `G17PhysicalCodingGroupV1` identity,
  packet-relative and archive-relative section spans, G27 adapter identity,
  eight canonical G23 constraint mappings, and the exact G27 receiver operation.
- `R10GenericRepairContractV1`: a bounded deterministic post-R10 ABI with fixed
  generic operation ID, iteration/workspace cap, and no video-specific default.
  It is compatibility metadata, not a claim that repair has been measured.
- `R10CorrectionPlacementComparisonV1`: exact stored packet/archive bytes versus
  a generic regeneration/repair alternative. If a generic alternative cannot
  reconstruct the correction from other counted state, mark it unavailable and
  retain the stored state; never invent a zero-byte repair win.
- `R10ResidualInventoryV1`: only receiver-realized debt remaining after every
  analytic/fixed finite solve and exact interaction pass; learned residual
  admission stays false in this landing.

All canonical receipts use duplicate-key-refusing JSON reads, finite numbers,
exact field sets, sorted compact encoding, immutable write-once publication,
and content roots. Caller-attested hashes never substitute retained bytes.

## 5. Maximum inverse stages

Fit in physical receiver order with an immutable end-of-stage checkpoint after
each stage. Each stage operates over every requested pair, in ordered chunks,
and resumes only from a contiguous byte-validated prefix.

### 5.1 `PAIR_INDEX`

Compile exact canonical coordinates `0..P-1`. Record the mapping to source frame
IDs `(2p,2p+1)`. This is counted state, not a free implicit population. Any
permutation, gap, duplicate, or subset relabel refuses.

### 5.2 `GEOMETRY`

Fit the shared signed-Q20 pitch by deterministic finite global inverse solve
over the declared lattice and all pair chunks. For each lattice point, replay
the same G27 homography operation on deterministic decimated sufficient
statistics and accumulate exact integer RGB error against source frame 0.
Select by `(objective, abs(pitch_q20), pitch_q20)` and preserve every measured
point. The lattice is generic config/evidence; the selected video-specific
pitch is the counted four-byte section. A hard-coded source-tuned pitch is
forbidden.

### 5.3 `XIP2`

For every pair, solve the six-dimensional twist using coarse-to-fine deterministic
Gauss-Newton/coordinate refinement against actual source/base frame-0 pixels and
the selected geometry. Jacobians may be finite-difference encoder work; the
decoded operation remains the frozen G27 NumPy receiver. Quantize only through
the canonical XIP2 coder and re-evaluate the quantized receiver result. A zero
twist is admissible only when the fitted exact objective selects it. Preserve
the finite schedule, normal-equation condition telemetry, accepted steps, and
quantized objective; no learned pose target or Pose6 table is allowed.

### 5.4 `BASE_FEATURE`

Fit the actual G27 integer feature map. For each pair, form the frozen receiver
design columns `[1, luma-128, laplacian, centered_gradient]` from both realized
base frames after XIP2 transport and regress the source RGB residual. Factor the
4×3 coefficient matrix into one scalar feature vector and three channel weights
with deterministic rank-one SVD, canonical sign, fixed-point projection, and
bounded integer coordinate refinement against the actual uint8 receiver. Store
all seven i16 values per pair. Report rank-one residual and saturation; do not
call it scorer residual.

### 5.5 `TEXTURE` and `DASH1`

Build DASH support only from encoder-visible source/base integer residuals. Use
a fixed generic connected-support rule; every selected support/event/phase
realization that affects decode is carried in DASH1. Source-selected support is
never compiled into Python. Use an SSD-backed uint8 memmap or sparse streaming
builder so n600 labels do not coexist with full RGB tensors.

For each pair, exhaust the configured finite frequency and Q10 phase schedule,
solve the amplitude×texture-gain product analytically, choose a canonical i16
factorization, and re-evaluate through the actual integer texture carrier.
Selection order is exact integer residual then canonical operand bytes. DASH1
must set `include_xi=false`, bind external XIP2 exactly, strictly decode, and
re-emit. `TEXTURE` without DASH1 or DASH1 with another trajectory owner refuses.

### 5.6 `SHOOTING_KNOT`

Fit piecewise-linear correction knots after the per-pair base/texture solve.
Use exact endpoint constraints plus deterministic maximum-deviation refinement
and emit the complete distortion/byte Pareto family rather than hiding a tuned
knot threshold. The selected research packet is the distortion-first analytic
corner with canonical byte tie-break; later scorer/whole-ZIP pricing may select
another preserved corner. Interpolation must use the G27 half-away-from-zero
integer rule and be checked after parse-back.

### 5.7 `PULLBACK_POLYGON` and `STRATIFIED_FLOW`

Compute post-planar residual motion with deterministic integer gradients and
Lucas-Kanade normal equations. A candidate flow is legal only with an explicit
counted Q15 polygon. Polygon support is the canonical convex/bounding pullback
of the selected connected residual component; pair inclusion is itself counted
by the section record. Re-evaluate the exact G27 stratified warp and preserve
both delete and joint corners. Non-finite/singular solves become typed analytic
blockers, not guessed coefficients.

### 5.8 Interaction closure

After all nine individual fits, replay the joint packet chunkwise and perform
at least one deterministic block-coordinate refit in this order:

`GEOMETRY+XIP2 -> BASE_FEATURE -> TEXTURE+DASH1 -> SHOOTING_KNOT -> POLYGON+FLOW`.

Preserve section-deletion and pairwise parent/child interaction telemetry. The
source-domain objective is an encoder fitting coordinate only. d_seg, d_pose,
complete candidate ZIP bytes, contest score, and score-unit-per-byte remain
null until later public n600 measurement.

## 6. Counted-state and hidden-code-as-data gate

The compiled packet is the sole video-specific decoder input. The counted-state
manifest must map every active video-specific decoder value to exact packet
bytes. In particular:

- pair IDs, selected pitch, base coefficients, texture coefficients, knot
  positions/deltas, XIP2 scales/codes, DASH supports/events, polygon vertices,
  flow coefficients, repair selector, fitted threshold, per-pair exception, and
  any learned residual must be counted if active;
- generic arithmetic, fixed-point conventions, parser code, iteration structure,
  and universally fixed bounds may be free;
- encoder search traces, source frames, sufficient statistics, gradients, and
  normal equations are encoder-only evidence and must not enter packet/runtime;
- scorer tensors, GT labels, target frames, Pose6 tables, public candidate
  payloads, teacher state, and source-tuned Python literals are forbidden.

Implement an audit that refuses any active decoder claim marked video-specific
unless it has one nonempty exact counted span, and refuses any span overlap,
hash drift, dead bytes, or decoder parameter not represented in the packet.
The audit must include adversarial tests for a hidden source-selected threshold,
selector, exception table, and packaged code falsely marked generic/free.

## 7. Generic bounded decode-time repair compatibility

Expose a post-R10 repair ABI that accepts only:

```text
(realized_uint8_pairs, canonical_r10_packet, generic_iteration_budget)
    -> realized_uint8_pairs + repair receipt
```

The iteration/workspace bound is universal and fixed. Any active repair mode,
selector, threshold, exception, weight, or residual that varies with this video
must have a counted span; because frozen G27 has no repair section, G32 must keep
such repair inactive unless it is a pure deterministic function of existing
counted packet bytes. Compatibility alone is not repair evidence.

For every correction-bearing section, telemetry compares:

1. storing its exact counted bytes;
2. deleting it and regenerating exactly from other counted state, if a generic
   exact regenerator exists; and
3. deleting it and applying the bounded generic repair ABI, if executable.

Report exact packet bytes and deterministic realized uint8 deltas for available
arms. Complete candidate ZIP bytes and scorer terms remain null. When neither
generic arm reconstructs the correction, the verdict is
`STORED_COUNTED_STATE_REQUIRED`; zero-byte benefit is forbidden.

## 8. G27 and G23 closure

The final packet must pass:

- strict G27 parse/re-emit identity;
- G27 selected-solution adapter generation;
- deterministic G27 bounded decode against exact base bytes;
- physical section order exactly all nine IDs;
- exact section offsets, lengths, bit spans, CRC identities, and SHA-256;
- every G23 constraint mapped to the exact G27 operand spans;
- one deterministic STORE ZIP member containing the packet, with explicit ZIP
  member data range and complete archive bytes;
- construction of an actual `G17PhysicalCodingGroupV1` over retained exact ZIP
  and member bytes, with the G27 receiver operation and explicit logical owners.

The G32 adapter reports both packet-relative/member-relative offsets used by
G23 R10 coordinates and outer-archive offsets used by physical byte custody.
It does not edit G23 or clear G23's current receiver-consumption blocker; root
must connect the adapter into the product manifest and G29 public runtime.

## 9. Streaming, storage, resumability, cleanup

The production CLI requires `--resume-from`, `--execute-reviewed`, and exact
input paths. Exactly n600 additionally requires `--confirm-full-n600` and
`--confirm-no-live-heavy-owner`; G32 does not invoke those flags in this unit.

Storage waterfall is, in order:

1. `/Volumes/VertigoDataTier/pact`;
2. `/Volumes/APDataStore/pact`;
3. local only with explicit hidden test-only opt-in.

Preflight reserves enough space for the selected base raw witness, DASH label
memmap, stage scratch, and caller reserve. Selected base raw is physically
allocated and filled in disjoint ranges. Source frames are decoded per ordered
chunk and discarded after their stage checkpoint. Peak resident memory must be
bounded by `O(chunk_pairs*2*H*W*3)` plus declared work arrays, never n600 RGB.

Mandatory immutable stages:

```text
000_custody
010_selected_base
020_pair_index
030_geometry
040_xip2
050_base_feature
060_texture
070_shooting_knot
080_dash1
090_pullback_polygon
100_stratified_flow
110_joint_refit
120_packet_adapter
130_bounded_decode_receipt
140_cleanup_certificate
```

Every stage and chunk checkpoint is written atomically as a distinct
content-addressed file; no previous stage is overwritten. Resume validates a
contiguous stage/chunk prefix, implementation/config/input hashes, and retained
scratch ranges. Full necessary state for the next stage is present on disk.

Before removing any base raw, label memmap, source chunk, or candidate sweep,
write a machine-readable cleanup certificate with original path, bytes,
SHA-256/tree hash, exact rebuild command/config/env, source/runtime/archive
hashes, cold-store destination if moved, false-authority flags, and rebuildable
reason. Failure or incomplete proof preserves bytes and emits a blocker.

## 10. Telemetry

Per section and interaction preserve:

- exact section bytes/hash/span and complete deterministic wrapper ZIP bytes;
- source-domain integer objective before/after and realized uint8 changed
  values/pixels/L1/max delta;
- conditional `d_seg`, `d_pose`, complete candidate ZIP bytes, and score fields
  as null with their exact blocker until later measured;
- encode wall/CPU time, peak RSS/native units, chunk size, solver iterations,
  condition/singularity facts, and decoder work estimate;
- stored versus generic-regeneration/repair comparison;
- analytic exhaustion and residual inventory state;
- pointer delta false and no candidate/promotion/score claim.

No arbitrary pass/fail thresholds convert source RGB fitting into scorer
evidence. A negative is scoped to the exact formulation and objective.

## 11. Safe verification in this unit

Allowed:

- pure structural/unit tests using tiny deterministic arrays;
- one real n1/n2 native-frame mechanism run using the exact G20/G22 runtime,
  selected member, and first source frames, with SSD scratch and certified
  cleanup;
- G27 decode/adapter parse-back on that bounded packet;
- Ruff, Ruff format check, pycompile, focused pytest, and `git diff --check`.

Forbidden in this unit:

- full n600 G32 fire;
- frozen scorer imports or execution;
- `upstream/evaluate.py` invocation;
- public mux claims before G29 closes them;
- learned training or joint descent;
- commit or push.

## 12. Acceptance

The landing is acceptable only when:

1. all code/docs/tests are new disjoint G32 files;
2. the real fitter produces all nine canonical physical sections, not a search
   suggestion or copied G27 fixture;
3. exact frozen custody and source-only encoder provenance are enforced;
4. the implementation is deterministic, chunk-bounded, resumable, and has
   immutable stage checkpoints plus certify-or-block cleanup;
5. counted-state coverage is exact and hidden code-as-data attacks refuse;
6. G27 strict parse/decode/adapter and G23 physical-group compatibility pass;
7. repair comparison is honest about unavailable generic regeneration;
8. residual admission remains false and terminal joint descent remains blocked;
9. bounded real n1/n2 output changes are reported only as mechanism evidence;
10. focused tests, Ruff, format check, pycompile, lane validation, and diff check
    pass; and
11. findings report exact files/SHA-256, tests, bounded receipt, dormant n600
    fire command, blockers, and pointer delta zero.

## Stores consulted

- byte-identical `CLAUDE.md` / `AGENTS.md`, `PROGRAM.md`, and craft handoff;
- current canonical pointer and live subagent/lane state;
- G20 exact xcodec spec/receipt and G22 completed full-n600 receipt/runtime;
- frozen G27 source, runner, spec, findings, packet, and canonical receipt;
- G23 selected-solution compiler types, physical group, R10 coordinates, and
  terminal inverse-solve/joint-descent schedule;
- current project MEMORY inverse-frozen-space, realization-completeness,
  no-duplicate-data, vehicle-naming, and residual-exhaustion doctrines.
