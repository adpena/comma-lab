# G27 R10 feature / texture / luma / edge relay

Date: 2026-07-26  
Lane: `lane_g27_r10_feature_texture_relay_20260726`  
Authority: research-only implementation contract; no score claim; pointer unchanged  
Owner: G27 implementation arm; root/G23 owns selected-solution compiler integration

## 1. Outcome

Implement one real, deterministic, decoder-executable R10 cross-level relay that
turns counted video-specific operands plus an exact upstream realized-pair
dependency into actual `uint8` receiver frames. It is not an eighth semantic
stream, a Pose-vector sidecar, an opaque constraint blob, a scorer proxy, or a
digest rendered as pixels.

The relay binds and physically consumes all eight G23 R10 constraint names:

1. amplitude;
2. frequency;
3. phase;
4. contrast;
5. channel energy;
6. texture;
7. multiple shooting; and
8. frozen-feature relay.

The current unit closes executable packet/receiver mechanics and produces a
bounded real-input contract receipt. It does not price the relay scientifically:
only a changed-output full-n600 scorer run and complete archive can do that.

The relay remains compatible with maximum inverse solving. Analytic inverse
solve and generic receiver work are exhausted first; training may represent
only a counted irreducible residual left by that solve. Joint descent is not a
new carrier: it is the final exact realized-score and whole-ZIP linker pass.

## 2. Reused original production primitives

- `xi_pose_coder.py`: quantized XIP2 trajectory. R10 accepts only canonical
  `none`, `delta_ar`, or `delta_res` XIP2 spellings and re-emits them exactly.
- `warp_real_luma_frame0.py`: NumPy-fp64 SE(3) plane-homography inverse warp.
  This is the frame-0 luma/edge transport primitive.
- `dash_phase_carrier.py`: strict DASH1 chronology, event coding, counted phase
  anchors, and a decoder-visible spatial pullback for local texture.
- `stratified_depth_warp.py`: optional affine extra flow, but only beneath an
  explicitly counted polygon pullback. Flow coefficients without support refuse.
- `pdw2_spatial_receiver.py`: the nonidentifiability lesson is binding. A
  coefficient does not identify a spatial realization. R10 texture coefficients
  therefore require DASH1; stratified coefficients require counted polygons;
  base-derived luma/edge gains require an exact hash-bound upstream realized pair.

No scorer, target, teacher tensor, GT argmax, Pose6 table, full-RGB residual, or
borrowed archive is available to the decoder.

## 3. Lifecycle and identity

The packet is tied to one already-realized selected-solution population:

```text
SourceTruth(source_sha256, pair_population_sha256)
  -> upstream RealizedPair population (base_realization_sha256)
  -> counted R10 packet (packet_sha256)
  -> uint8 relayed RealizedPair population (output_sha256)
  -> research-only DecodeReceipt
```

The packet stores the three 32-byte identities and exact canonical uint16 pair
coordinates. Decode recomputes the base-population hash and validates caller
source/population identities before reading operands. A packet for another
source, population, order, shape, or base realization refuses.

`base_realization_sha256` names bytes owned and charged by the upstream
selected-solution compiler, not by R10. R10 never credits those bytes as free and
never duplicates them in its packet.

## 4. Packet syntax

Magic/version: `TSR10R1\0`, version 1. The fixed header contains mode, pair count,
height, width, source/population/base SHA-256 values, and section count. A sorted
directory names every section and records exact byte offset, length, and CRC32.
Parsing requires strict EOF, exact CRC, canonical section order, known flags, and
canonical parse/re-emit identity. The receipt additionally reports SHA-256 and
physical bit range for every section.

Sections:

| ID | Name | Required / meaning |
|---:|---|---|
| 1 | `PAIR_INDEX` | required contiguous/canonical uint16 coordinates |
| 2 | `GEOMETRY` | required counted signed-Q20 fitted pitch; fixed EON camera constants remain generic |
| 3 | `BASE_FEATURE` | required per-pair luma-bias, contrast, edge, frozen-feature, and channel-energy operands |
| 4 | `TEXTURE` | optional per-pair amplitude/frequency/phase/texture operands |
| 5 | `SHOOTING_KNOT` | optional ordered image-domain continuity knots |
| 6 | `XIP2` | required for joint frame-0 transport and DASH1 chronology |
| 7 | `DASH1` | required whenever `TEXTURE` exists; counted spatial/temporal pullback |
| 8 | `PULLBACK_POLYGON` | required whenever `STRATIFIED_FLOW` exists |
| 9 | `STRATIFIED_FLOW` | optional per-pair affine flow coefficients |

All scalar operands are fixed-width little-endian integers. Fixed-point scales
are generic ABI constants in decoder code. DASH1/XIP2 retain their own counted
self-description. No fitted default is compiled into Python.

`BASE_FEATURE` record, one per pair:

```text
luma_bias_q8:i16, contrast_q8:i16, edge_gain_q8:i16,
feature_gain_q8:i16, channel_weights_q8:i16[3]
```

`TEXTURE` record, one per pair:

```text
amplitude_q8:i16, frequency_q8:u16, phase_q10:u16,
texture_gain_q8:i16
```

`SHOOTING_KNOT` is an ordered list of unique pair coordinates with signed deltas
for luma bias, contrast, edge, amplitude, phase, and texture. Decoder interpolation
is integer piecewise-linear with an explicit half-away-from-zero quotient. The
knots are image-domain feature continuity coordinates, never Pose6 targets.

`PULLBACK_POLYGON` stores per-pair normalized-Q15 vertices. `STRATIFIED_FLOW`
stores six signed Q8 source-pixel affine coefficients per pair. Nonzero flow with
no counted polygon refuses before output is generated.

## 5. Receiver physics

### 5.1 Control branches

- Pixel/tensor alone: upstream base bytes unchanged; represented by no R10 packet.
- Relay alone: a neutral gray raster of the declared dimensions is fed to the
  same R10 operator. It is a real `uint8` counterfactual, not a candidate.
- Joint: the exact hash-bound upstream base population is fed to R10.

In joint mode XIP2 transports frame 0 with the real NumPy warp. If stratified
flow exists, the counted polygon raster and affine coefficients drive the
stratified warp. Frame 1 stays spatially fixed so Seg/Pose coupling is not hidden.

### 5.2 Global luma/edge feature relay

For each input frame, receiver computes integer BT.601 luma and a signed
four-neighbour Laplacian. Counted contrast and edge gains produce a bounded
fixed-point delta. The frozen-feature operand gates a generic local
edge-modulated feature map; it is not a stored scorer feature. Counted channel
weights project the scalar delta back to RGB. Round/clip is explicit and the
result is actual `uint8`.

### 5.3 Local texture / phase relay

DASH1 decodes tracked dash centroids, tilts, areas, births, and chronology from
counted bytes. The receiver rasterizes a compact generic oriented carrier around
each dash. Counted frequency and phase select an integer triangle wave along the
dash tangent; amplitude and texture gain scale it. DASH1 is therefore the
spatial pullback that makes those coefficients identifiable.

Texture coefficients without DASH1 refuse. DASH1 without XIP2 refuses because
its temporal advection dependency is absent. A packet operand mutation must
either change output at its named consumer or be reported as a dead/saturated
operand; no name-only consumption receipt is admitted.

## 6. Required counterfactuals

The module exposes canonical packet rewrites:

- relay-only and joint;
- phase/time permutation (TEXTURE records; population order remains fixed);
- channel-energy-preserving shuffle (permutes channel weights and proves exact
  integer sum-of-squares preservation);
- texture deletion (physically removes TEXTURE and DASH1 bytes);
- shooting-knot deletion (physically removes the section);
- adjacent shooting-knot merge (physically emits one fewer record).

Controls rebuild the packet and therefore receive new exact packet/section
identities. A zeroed retained blob is not deletion.

## 7. Telemetry, costates, and allocation

The decode receipt reports:

- exact packet bytes, section vector, and physical bit ranges;
- packet/base/output identities and deterministic double replay;
- changed pixels and L1 output delta versus the exact base;
- a mutation canary for each populated operand class;
- output hashes for required controls;
- packet-only nominal rate coordinate `25*packet_bytes/37_545_489`, explicitly
  not the complete archive rate; complete-ZIP rate remains null until rebuilt;
- scorer costates as JSON null unless two matched full-n600 complete-object
  scorer rows exist;
- strict proxy debt: full-n600 frozen CPU scorer, exact archive ZIP repricing,
  contest runtime, cross-host bit identity, and G23 placement closure;
- an ordered bit-allocation candidate list based only on physical deletion bytes,
  with value/byte null until measured;
- explicit autopilot and continual-learning hook records, both blocked from
  scientific updates until the exact dependencies above exist.

No changed-pixel count is called d_seg, d_pose, score, or scorer sensitivity.

## 8. Runner and receipt

`tools/run_taskspace_r10_feature_texture_relay.py` consumes actual video frames
from a caller path, constructs a deterministic bounded pair population and a
small counted R10 packet, double-decodes it, executes all controls, and writes a
content-addressed JSON receipt under the durable research directory. Its default
pair count is 2 and its maximum bounded-contract count is 24. This is wiring and
mechanism evidence only, never a scientific verdict.

The implementation itself supports 1..600 pairs. Any future n600 scorer run must
use the governed, storage-preflighted, resumable selected-solution runner rather
than extending this bounded contract runner into an authority path.

## 9. G23 adapter ABI

The module returns a `R10SelectedSolutionAdapterV1` with:

- the exact eight canonical G23 constraint names;
- one or more exact packet operand spans for each constraint;
- receiver operation ID;
- source/population/base/output identities;
- frame/scientific/semantic role suggestions;
- packet section identities and packet bytes; and
- live blockers.

The public receiver ABI is
`decode_r10_packet(packet_bytes, base_pairs, expected_source_sha256,
expected_pair_population_sha256) -> uint8[P,2,H,W,3]`. The adapter carries a
recursive conservative manifest of every repo-local source dependency and its
SHA-256. That manifest is not inflate custody: the blocker stays live until an
actual `inflate.py` / `inflate.sh` bundle contains the closed dependency graph,
replays these packet bytes, and runs under `upstream/evaluate.py`.

The adapter also partitions the asymmetric codec explicitly:

- unbounded encode-only compiler: source analysis, inverse solve, operand
  fit/selection, scorer/costate work, and final joint-descent linking;
- generic bounded decode VM: strict parsing, XIP2/DASH1 decode, warps,
  pullbacks, feature/texture rasterization, and uint8 realization;
- counted video statistics: every physical packet section; and
- counted learned irreducible residual: empty in this bounded object, and may be
  populated only after maximum inverse solving leaves measured debt.

The present public ABI emits exact uint8 pair populations. It does not yet mux
the contest-required video file. Runtime below 30 minutes, peak memory below
16GB, CPU/GPU parity, video emission, and an `upstream/evaluate.py` replay remain
typed promotion blockers rather than inferred from the bounded n2 run.

Root/G23 may map this descriptor into
`G17R10ConstraintCoordinateV1` and physical coding groups. G27 will not import a
second copy of G23 types and will not edit `taskspace_selected_solution_compiler.py`.

## 10. Acceptance and honest blockers

This unit is complete only when:

1. packet parse/re-emit is exact and corruption/trailing bytes refuse;
2. decode returns actual `uint8 [P,2,H,W,3]`;
3. same packet/base double replay is byte-identical;
4. identity drift refuses;
5. coefficient-only texture and flow refuse;
6. every required control physically executes;
7. section mutation canaries demonstrate receiver reachability;
8. Ruff, pycompile, and adversarial tests pass; and
9. a real-video content-addressed bounded receipt exists.

Remaining after this unit:

- `R10_FULL_N600_SCORER_AND_COMPLETE_ARCHIVE_PRICING_OWED`;
- `R10_G23_PHYSICAL_GROUP_AND_LIFECYCLE_INTEGRATION_OWED`;
- `R10_CROSS_HOST_BIT_IDENTITY_PROOF_OWED`;
- `R10_CURRENT_SELECTED_SOLUTION_FITTED_OPERANDS_OWED`;
- `R10_INFLATE_PUBLIC_RECEIVER_BUNDLE_CLOSURE_OWED`;
- `R10_MAXIMUM_INVERSE_SOLVE_AND_IRREDUCIBLE_RESIDUAL_LINK_OWED`;
- `R10_JOINT_DESCENT_EXACT_REALIZED_SCORE_ZIP_LINKER_OWED`;
- `R10_INFLATE_RUNTIME_30MIN_AND_MEMORY_16GB_PROOF_OWED`;
- `R10_CPU_GPU_DECODE_PARITY_OWED`;
- `R10_PUBLIC_VIDEO_EMISSION_AND_UPSTREAM_EVALUATE_REPLAY_OWED`;
- `R10_EXACT_CONTEST_ROW_OWED`.

## Stores consulted

- byte-identical `CLAUDE.md` / `AGENTS.md`;
- `PROGRAM.md`;
- G21 selected-solution homotopy §4.8 and compiler-gap audit;
- G24 missing-type/compiler audit §4 and §6;
- prior Pose-side feasibility verdict;
- current G23 selected-solution compiler R10 ABI;
- original XIP2, warp-real-luma, DASH1, stratified-warp, and PDW2 spatial
  receiver source implementations.
