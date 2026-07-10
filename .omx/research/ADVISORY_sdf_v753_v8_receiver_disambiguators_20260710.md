# ADVISORY — v7.5.3/v8 receiver closure and SDF disambiguators — 2026-07-10

`research_only=true`

Authority: adversarial architecture review and future proof design. This memo changes no source,
state, config, run, score pointer, or dispatch. Findings were read against `main` around
`b75ce8f8b..91dbd820f`; future work must re-derive current state.

## Answer first

Both vehicles contain useful build primitives, but neither is yet an executable contest vehicle:

- v7.5.3 can train a texture trunk or widened texture head, yet the canonical NumPy/byte-close/inflate
  receiver ignores the texture-trunk tensors. A deterministic bank intended to be free can also be
  serialized and charged.
- v8 can train class-specific output modules, yet its canonical receiver ignores those tensors and
  its fields still share a trainable pair code. The narrow head Jacobian is block-diagonal; the full
  trainable vehicle is not.

The correct next build is a receiver and proof repair, not a scored run. v7.5.3 should disambiguate
generic capacity from exact-D texture in matched final bytes. v8 should first use globally defined K
class potentials with fully class-owned trainable state and edge-weighted loss/rate ownership. A
separate E-edge codec remains held until it proves integrability or specifies a different global
labeling decoder.

## 1. Receiver identity is the first invariant

Define one decoded forward law

\[
(\phi,I_0,I_1)=D(M,q,x,t),
\]

where `M` is a versioned manifest and `q` contains exactly the receiver-consumed video-derived state.
The same law must be exercised by:

1. MLX training/export reference;
2. deterministic NumPy fp32 authority;
3. byte-close/select oracle;
4. fresh-process `inflate.py`.

For every manifest section `s`, require both directions:

\[
s\in q_{counted}\Rightarrow
\frac{\partial D}{\partial s}\ne0
\quad\text{under a deliberate test perturbation},
\]

\[
s\in q_{live,video-derived}\Rightarrow s\in q_{counted}.
\]

The first refuses dead counted bytes. The second refuses uncounted learned state. Analytic fixed banks
must be regenerated from versioned constants and construction hashes, not silently appear in either
category.

## 2. v7.5.3 audit

### 2.1 What landed

- `OutTexHidden` widens the texture head from a linear map to a one-hidden-layer MLP.
- `TextureTrunk` adds a deterministic basis and counted texture coefficients into the same
  pre-sigmoid RGB term, placed by the soft class masks.
- resume provenance keys exist for texture-trunk and widened-head architecture.
- ChromaRung is registered in the DSL ladder.

These are useful build primitives, not receiver-closed evidence.

### 2.2 P0 — receiver is texture-trunk blind

Training calls the texture trunk in `_compose_rgb`. The canonical NumPy/byte-close/inflate forwards
use `out_sdf`, `out_tex`, and `palette`; they do not consume `tex_trunk.*`. The generic archive builder
serializes non-metadata tensors, so `tex_trunk.bank_B`, `w_tex`, and `bias` can be counted even though
they do not affect decoded pixels.

The in-memory counter's fixed-bank exclusion does not cure this: `accounting_matches_canonical`
reports a discrepancy but does not refuse it. A prior example put the deterministic bank near 4.7 MB
raw / 430,878 bytes after Brotli. Regardless of exact future size, the semantic failure is binary:
counted and ignored is not a codec.

Required repair:

- explicit manifest section for texture coefficients;
- deterministic bank construction by stable ID/hash;
- one receiver computation in all four paths;
- perturbation survival through final raw;
- refusal on unknown, missing, duplicate, or unconsumed tensors;
- canonical accounting refusal, not a warning.

### 2.3 P0 — exact-D home does not exist

The current texture contribution:

- runs on both frames;
- enters before sigmoid/final RGB composition;
- is placed through soft masks/annulus power;
- is not projected into the exact local Pose-preprocess nullspace;
- has no camera-grid preimage/lift proof;
- has no fresh-raw first-six-Pose-input stability receipt.

This cannot support the claim “texture changes Seg without changing Pose.” High-frequency chroma is a
heuristic, not an exact-D theorem.

Required construction:

1. compute the fully composed frame1 RGB proposal;
2. express a local update in the exact six-atom null basis at 384x512 scorer resolution;
3. solve a bounded 2x2 camera-grid lift through the actual bilinear resize;
4. round/clip to uint8, serialize raw, parse in a fresh process;
5. prove the first six Pose inputs/outputs meet the registered equality tolerance;
6. prove a nonzero intended Seg effect;
7. set the texture contribution identically zero on frame0.

If quantization makes exact equality impossible at useful amplitude, record formulation refusal and
test the minimum-Pose-cost texture mode. Do not relabel approximate null as exact.

### 2.4 P1 — A1/A2/A3 are not matched arms

Current configuration can compose widened-head A2 and texture-trunk A3 together. The default A2
`hidden=16` adds roughly 1,600 parameters at shared hidden width 96, while the observed A3 coefficient
surface has roughly 375 trainable coefficients before archive effects. Parameter counts are not
final bytes, and the arms confound generic capacity, temporal conditioning, basis, placement, and
receiver support.

The compiler should require exactly one:

- A1: current linear texture head;
- A2: generic extra texture capacity with matched receiver/archive bytes;
- A3: exact-D SDF texture with matched receiver/archive bytes.

Match source, seed set, stage schedule, optimizer treatment, checkpoint boundaries, wall-clock/FLOP
envelope, receiver version, and final archive bytes. If exact bytes cannot be matched, measure a small
byte curve and interpolate only as advisory—not as an exact result.

### 2.5 P1 — ChromaRung is registered but not compiled

`derive_crucible_v753_config` has no chroma-addback input and does not compose `SegChromaBoundary`.
The rung is therefore a registry entry, not a named v7.5.3 experiment. It should remain OFF until:

- the receiver consumes the texture home;
- the compiler emits it as one typed, resume-persisted arm;
- the add-back coefficient path is exact-byte accounted;
- the matched through-R n600 A/B exists.

### 2.6 P1 — gauge and optimization coupling

Texture bias duplicates part of the palette home, introducing a null/gauge direction. Fix a gauge,
for example zero-mean texture over a declared basis or removal of the redundant bias. Separate heads
also do not create optimization separation while they share a trunk and latent. Report gradient
cosines/step interactions or use explicit stop-gradient/frozen controls in the matched experiment.

## 3. v7.5.3 target config design

The future typed compiler—not ad hoc flags—should produce a manifest equivalent to:

```text
vehicle = v7.5.3
geometry = shared_stem_five_potentials
texture_arm = one_of[A1_linear, A2_capacity, A3_exact_D]
texture_frame = frame1_only
texture_gauge = explicit
pose_arm = one_of[xi_control, analytic_basis, learned_basis, sdf_generator]
receiver_version = content_addressed
fixed_bank_policy = regenerate_and_hash
stage_schedule = event_driven_and_resume_persisted
selection = final_zip_exact_score
```

This is a design target, not an invented CLI surface.

## 4. v8 audit

### 4.1 What landed

`DecoupledField` supplies independent class-head tensor blocks and routes the MLX partition forward
through them when enabled. Resume keys and narrow head-parameter tests exist. This demonstrates a
useful primitive: changing `w_out[k]` can leave other output heads unchanged.

It does not demonstrate complete class-owned trainable state.

### 4.2 P0 — shared trainable code preserves theft

Each `phi_c` consumes the same trainable `self.code[code_idx]`. A loss on class `c` can update that
code and move every field. The promised invariant must cover all trainable state:

\[
\frac{\partial\phi_c}{\partial\theta_{c'}}=0,
\quad
\frac{\partial\phi_c}{\partial z_{t,c'}}=0,
\qquad c\ne c'.
\]

The minimum coherent increment uses:

\[
\phi_c(x,t)=f_c(\gamma(x,t);\theta_c,z_{t,c}),
\]

with `gamma` fixed/deterministic and every trainable decoder weight and pair latent class-owned.

Required test: take one class-only loss and a real optimizer step over all trainable parameter groups;
evaluate every other potential over a registered coordinate set before/after. A row perturbation test
is insufficient.

### 4.3 P0 — receiver is decoupled-field blind

The generic builder may serialize `decoupled_head.*`, but the canonical NumPy/inflate partition uses
the shared `out_sdf(h)`. Thus the same failure appears: counted fields can decode as the shared-head
control.

Required repair is the receiver identity invariant from section 1, including a deliberate per-class
field perturbation that survives parse-back and changes only the intended potential before argmax.

### 4.4 P0 — kill predicate has conflicting semantic surfaces

The intended benefit statistic is

\[
improvement=d_{seg}^{control}-d_{seg}^{decoupled}.
\]

Confirmation requires `improvement>delta`. The current config prose/serialization preserves the
opposite inequality (`decoupled > control + delta`) while the evaluator logic computes the benefit
direction. Tests can appear green while preserving the wrong textual contract.

Required repair:

- one stable predicate ID and normalized field names;
- derive config, evaluator, report prose, and test cases from that predicate;
- metamorphic obvious-winner/obvious-loser cases;
- refuse a manifest whose predicate ID/hash differs;
- keep mask non-inferiority and per-class/topology gates separate from aggregate improvement.

### 4.5 P0 — current analytic row is not byte closure

The existing increment-1a analytic row uses a GT-derived polynomial fit, lacks a control arm and
receiver payload, and correctly ends `REFUSED-arm-missing-or-toy` around `d_seg=0.100403`. A boolean
`measure_byte_closed_composite` cannot turn an analytic field into a codec.

Do not use that row as evidence for or against trained class isolation.

## 5. K potentials versus E edge fields

### 5.1 Recommended increment: K potentials

Use K globally defined class potentials as the receiver state. Apply edge-centric objectives to
margins

\[
m_{ij}=\phi_i-\phi_j
\]

and allocate rate to the class-owned coefficients that reduce edge debt. This preserves a trivial
global decoder `argmax_c phi_c`, makes gauge explicit, and gives exact ownership.

### 5.2 Genuine E-edge formulation

For oriented edge fields `g_ij`, require antisymmetry

\[
g_{ij}=-g_{ji}
\]

and cycle consistency. On every class cycle `c_0,...,c_r=c_0`,

\[
h_{cycle}=\sum_{k=0}^{r-1}g_{c_kc_{k+1}}=0.
\]

On the class graph, let `B` be the incidence operator. Integrable edges lie in `im(B^T)`. A Hodge
projection solves

\[
\phi^*=\arg\min_{\mathbf 1^T\phi=0}\|B^T\phi-g\|_W^2.
\]

Required receipts:

- maximum and weighted cycle holonomy;
- projection/reconstruction residual;
- fixed gauge and connected-component handling;
- post-quantization holonomy;
- partition drift between encoded edges and reconstructed potentials;
- exact receiver bytes and runtime.

If `g=B^T phi` by construction, the arm is a K-potential reparameterization. Compare entropy and
optimization behavior, but do not claim a new decoder family. If a nonzero cyclic component is kept,
specify a graph-labeling energy/solver, deterministic tie law, complexity bound, and byte-closed
receiver before any score experiment.

## 6. v8 target config design

The future typed compiler should express a single architecture decision:

```text
vehicle = v8
partition_decoder = one_of[K_class_owned_potentials, E_integrable_projection,
                           E_explicit_graph_labeler]
trainable_coordinate_basis = frozen_or_class_owned
pair_latent = class_owned
gauge = explicit
edge_loss = margins_with_measured_scorer_footprints
texture = shared_frame1_exact_D_home
pose = shared_frame0_decoder_native_home
receiver_version = content_addressed
selection = receiver_closed_exact_score
```

The first scored increment should choose `K_class_owned_potentials`. The E arms exist only as
separate disambiguators.

## 7. Unified SDF-first proof DAG

### R0 — manifest and receiver schema

- enumerate counted, generated, and forbidden state;
- stable versions/hashes;
- reject unknown/unconsumed/missing tensors;
- one forward law.

### R1 — parity and survival

- MLX versus NumPy fp32;
- byte-close oracle versus fresh inflate;
- deliberate T/B1 mutations survive through `R`;
- decoded partition and raw hashes.

### R2 — exact-D

- frame0 T is exactly zero;
- frame1 post-composition projection;
- bounded resize preimage;
- uint8/fresh-raw proof;
- intended nonzero Seg effect and no material Pose effect.

### R3 — v7.5.3 matched capacity disambiguator

- A1/A2/A3 exactly one;
- matched source/seed/stages/optimizer/compute/receiver;
- final-byte curve;
- per-class/topology/tie/Pose receipts.

### R4 — v8 isolation disambiguator

- shared-head control versus class-owned K potentials;
- one-class optimizer-step isolation;
- correct stable kill predicate;
- matched compute and final bytes;
- per-class non-inferiority and topology gates.

### R5 — optional E-edge disambiguator

- antisymmetry/gauge/holonomy;
- projection or explicit global decoder;
- post-quantization partition drift;
- matched final bytes versus K-potential arm.

### R6 — complete n600 candidate

- frame0 pose generator and frame1 G/T;
- exact archive, raw cardinality/hash, runtime/RSS;
- component scores and interactions;
- accepted/rejected selection ledger.

### R7 — exact evidence axes

- exact contest-CPU and contest-CUDA separately;
- no inference from macOS/MLX;
- canonical promotion gate only after archive identity.

## 8. Minimum receipt set

### Receiver receipt

- manifest and construction hashes;
- list of counted/generated/forbidden tensors;
- per-section perturbation survival;
- MLX/NumPy/inflate parity;
- archive and raw SHA-256;
- exact cardinality and runtime.

### Matched-arm receipt

- source/input/config/seed/checkpoint hashes;
- architecture parameter and latent ownership;
- stage schedule and resume state;
- FLOP/wall-time envelope;
- actual archive bytes and member table;
- decoded partition/raw hashes.

### Evaluator receipt

- aggregate and per-class d_seg;
- island/birth/anchor/tie-flicker diagnostics;
- exact d_pose and `sqrt(10*d_pose)`;
- rate term and total score;
- authority axis/hardware/evaluator hash.

### E-edge receipt, if applicable

- class graph/incidence/gauge;
- holonomy distribution;
- projection residual;
- quantization drift;
- deterministic global labeler identity.

## 9. Disambiguator decision table

| Question | Arm A | Arm B | Match | Decision |
|---|---|---|---|---|
| extra texture capacity or correct home? | A2 generic | A3 exact-D | final bytes/compute | exact score + components |
| fixed or learned pose basis? | analytic | learned global | final bytes/K | exact pose score |
| conditioning useful? | unconditioned P | SDF-conditioned P | parameters/bytes | exact pose + code entropy |
| full class isolation? | shared code | class-owned code | compute/bytes | one-class step + d_seg |
| K or E representation? | K potentials | integrable E | final bytes/decoder | score + holonomy/runtime |
| sparse tail useful? | uniform K | smaller K + sidecar | total bytes | hard-pair and total score |
| chroma add-back useful? | OFF | named ON arm | receiver/stages/bytes | per-class Seg + Pose |

Every negative is formulation-scoped. A failed A3 lift does not kill all texture; a failed E arm does
not kill v8 K potentials; a failed pose basis does not kill the SDF witness.

## 10. Literal dispositions

| Action | Disposition | Reason |
|---|---|---|
| v7.5.3 receiver repair | `GO — BUILD ONLY` | required prerequisite; new claimed lane |
| v7.5.3 exact-D construction | `GO — BUILD ONLY` | precise falsifiable design |
| v7.5.3 scored A/B | `HOLD` | receiver/exact-D/matching absent |
| v7.5.3 long training | `HOLD` | not a safe receiver-closed candidate |
| ChromaRung measurement | `HOLD` | not emitted by compiler/receiver |
| v8 K-potential ownership repair | `GO — BUILD ONLY` | coherent first increment |
| v8 receiver repair | `GO — BUILD ONLY` | counted state currently ignored |
| v8 increment-1a training | `HOLD` | shared code/predicate/receiver gaps |
| E-edge integrability probe | `GO — BUILD ONLY` | mathematical disambiguator |
| E-edge scored carrier | `HOLD` | global decoder unresolved |
| exact CPU/CUDA | `HOLD` | no selected receiver-closed archive |

## 11. Exact remaining blockers

1. No single manifest-aware receiver serves MLX, NumPy, byte-close, and inflate.
2. `tex_trunk.*` and `decoupled_head.*` can be counted yet ignored by decode.
3. The texture fixed bank can be charged despite the stated regenerated/free policy.
4. No exact-D frame1 construction survives the actual camera resize, uint8, and fresh raw.
5. v7.5.3 A1/A2/A3 are not mutually exclusive or final-byte matched.
6. ChromaRung is not a named compiled v7.5.3 arm.
7. v8 shares trainable pair code, invalidating complete field isolation.
8. v8 config and evaluator semantics disagree on the kill inequality.
9. The analytic v8 row is not a receiver payload or matched control.
10. K-potential versus E-edge semantics are unresolved; no E decoder receipt exists.
11. No complete n600 candidate combines repaired G/T/P/C homes.
12. No final candidate has separate exact contest-CPU and contest-CUDA custody.

## 12. Triality and future wire-in

No triality leg is changed. Future implementation units must land:

- DSL: mutually exclusive v7.5.3 arms; class-owned v8 latent/state; K/E architecture choice; fixed-
  bank policy; receiver version; resume-persisted event state;
- DAG: R0-R7 with refusal edges and exact candidate passports;
- equations: receiver consumption bijection, exact-D projection/lift, full isolation Jacobian,
  stable benefit predicate, Hodge projection/holonomy, and matched-byte decision law.

Results must feed sensitivity, Pareto, allocation, cathedral dispatch, continual learning, and
disambiguator surfaces. This advisory does not mutate them because it is research-only.

## 13. Stores consulted

- `SPEC_v75_optimal_single_trunk_20260708.md` and v7.5.3 design/build memos;
- `SPEC_v8_perclass_decomposition_20260708.md`, `t5_crucible3/SPEC_v8.1_20260709.md`, and v8 build
  receipts;
- `t5_crucible2/ADVISORY_v753_texture_trunk_fresh_eyes_20260710.md`;
- `t5_crucible3/ADVISORY_v8_fresh_eyes_20260710.md`;
- `ADVISORY_evaluator_video_geometry_20260710.md`;
- current trainer, `levelset_byte_close_and_eval.py`, archive builder, texture-trunk, decoupled-field,
  DSL, and focused test sources.
