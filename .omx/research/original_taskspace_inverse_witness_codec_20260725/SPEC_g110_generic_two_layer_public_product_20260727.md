# G110 — generic final-Y1 provider × conditional-Y0 public product

Status: public runtime and counted wire implemented, but receiver execution is
not closed. The generated-Y1 compiler accepts one self-hashed recursive G112
partition receipt, which reopens the disjoint physical semantic/initializer
children and their complete ancestry, and requires a post-G105 refit through
the exact public receiver. The train/public V10 camera-realization mismatch and
final-ZIP raw/Rice × STORE/DEFLATE arbitration are now closed in code;
promotion remains blocked on a real producer/refit, public double decode, and
exact n600 eval.

Authority: receiver/runtime/parse-back implementation only. No trained
candidate, exact evaluator row, score, promotion, or pointer movement.

The present G111/G105 producer is not frontier-ready because checkpoint
selection uses an arbitrary-scale int8 surrogate while G105 ships power-of-two
int8 weights plus power-of-two int16 biases, palette, and Y1 code. G110 does
not erase that different-semantic-Y1 producer blocker.

## Why this is the missing composition layer

G107 represented the intended conditional equation but hard-cast its semantic
base to G103, depended on an untracked outer archive helper, required every
pair to have a nonzero residual, stored two scales whose product was the only
decoder-effective quantity, and left coefficients as a raw int16 table.
Consequently it could not compose with the exact V9 program and it charged a
gauge rather than a quotient.

G110 makes the product boundary the rendered final Y1 population:

1. self-describing semantic bytes select exactly one provider;
2. both G103 and exact G105 V9 implement the same public
   `accepts_packet/parse_packet/render_scorer_y1` ABI;
3. the receiver hashes all 600 rendered Y1 planes before publishing output;
4. conditional Y0 consumes that rendered uint8 Y1, never a semantic class;
5. the conditional quotient stores one combined per-rank scale;
6. chronological int16 coefficient differences use canonical minimal-k Rice;
7. a zero pair row is legal and reproduces Y1 exactly; unused whole ranks are
   rejected;
8. the first nonzero basis value is positive, each coefficient column is
   primitive (`gcd=1`), and decoder-dead/overflowing ranks are refused;
9. the low-rank variant sends both scorer planes through exact public V10
   factor-2 realization;
10. the generated-Y1 pose variant takes the actual final V10 camera Y1 as its
    source and emits the native warped uint8 camera Y0 directly.

This is a type-level composition fix, not another local representation arm.

### Strict optional self-orient extension seam

The G110 envelope treats semantic bytes as an opaque typed provider, so a
future deterministic fixed-point self-orient V9 arm can reuse the
two-layer composition without changing current packet meaning. It must enter
as a new semantic packet magic/version, exact repository parser and canonical
re-emitter, exact public plugin filename/variant allowlist entry, and new
producer/partition custody schema. `SV9Y1V1` remains the current no-self-orient
contract and is never reinterpreted or parsed permissively. The generated-Y1
compiler remains current-G105-specific until the new provider also supplies a
typed same-semantics wire enumerator for counted-archive arbitration.

Self-orient is not the required correction: newer n600 evidence found
warm-start OFF marginally better than ON while avoiding its large memory tax.
The immediate gate is selection and scoring through exact parsed-G105
power-of-two wire quantization. Self-orient remains an optional separately
trained arm after that gate.

## Equations

For semantic provider `q` and pair `p`:

`Y1_p = q.render(P_sem, p)`.

The final-Y1 custody coordinate is:

`H_Y1 = SHA256(domain || SHA256(P_sem) || SHA256(concat_p(u16be(p), Y1_p)))`.

The conditional quotient is:

`R_p = upsample(sum_r c[p,r] s[r] B[r])`

`Y0_p = clip_uint8(round(Y1_p + R_p))`.

Only `s[r]` is stored. Any factorization
`s[r] = s_basis[r] s_coefficient[r]` is the eliminated gauge.
The residual quotient additionally fixes the sign by requiring the first
nonzero serialized basis element positive and fixes the integer scale gauge by
requiring `gcd_p(abs(c[p,r])) = 1`. A rank must have a pre-resize amplitude
bound at least `0.5` and the population bound must remain finite in float32.

Chronological coefficients use
`delta[p,r] = c[p,r] - c[p-1,r]`, zig-zag mapping, and Rice parameter

`k* = argmin_(0<=k<=15) sum_(p,r)(floor(z(delta[p,r])/2^k) + 1 + k)`.

The outer ZIP contains exactly one counted member,
`taskspace_two_layer_v1.bin`. The producer measures both deterministic legal
methods for that member and then selects across the whole product:

`(wire*, zip*) = argmin_(wire in {raw,Rice}, zip in {STORE,DEFLATE}) archive_bytes`.

The parser re-derives the method optimum for the selected packet and rejects a
byte-valid but nonminimal method or noncanonical ZIP layout.

The semantic-stage selector uses the same receiver, not a bare G105 proxy:
`build_g110_rank_zero_semantic_floor_packet` constructs a canonical rank-zero
G110 product with `Y0 == Y1`, and the typed archive-variant APIs enumerate and
parse both STORE and DEFLATE before the cross-wire minimum is chosen.
Its measured archive size is an action-level gate for deciding whether pose
refit is worth running, not a theorem-level byte lower bound for every future
generated-pose packet: whole-packet DEFLATE is non-monotone. Without an
independently proven additive byte floor, the strict impossibility kill is only
`100*d_seg >= competitive_target`.

## Corrected generated-Y1 pose product

The joint G111 checkpoint contains a useful pose initializer:

`xi_init[p] = xi_stored[p] + residual_scale * dxi[p]`.

It does **not** contain the final counted conditional operand. G105 applies its
own exact weight/code quantization and emits both canonical-within-family Y1
temporal wires for final-archive arbitration, while the public receiver
realizes the parsed scorer-grid Y1 through V10. Those operations change the
photometric source relative to the joint trainer state. Because homography
warp, rounding, and scorer preprocessing do not commute, carrying `xi_init`
unchanged would be a false transfer claim.

The admitted generated-Y1 product is therefore:

`Y1_s[p] = parse_and_render_G105(P_sem, p)`

`Y1_c[p] = V10_factor2(Y1_s[p])`

`Y0_c[p] = uint8(warp_native(Y1_c[p], H(xi_refit[p], pitch)))`.

There is no scorer-grid round after the warp and no second V10 realization of
Y0. This is the existing `WarpRealLumaFrame0Carrier.render_f0` / store-nothing
camera order specialized to the final public source. The evaluator alone
bilinear-downsamples `Y0_c` for PoseNet. A tempting
`warp -> R-down -> scorer uint8 -> V10` implementation is a different operator
and is rejected.

The typed packet magic is `G110PC01`. It stores:

- the exact G105 semantic packet;
- canonical XIP2 `delta_ar` bytes for post-refit `xi_eff[600,6]`;
- the scalar pitch; and
- the existing ordered final-Y1 population binding.

It does not store even code rows, fp64 homographies, frames, targets, or
scorers. G112 makes the G111 ownership partition physical, total, disjoint,
and recursively lineage-bound. `10_g105_semantic_child.npz` stores shared
trunk tensors plus only
`code_y1[600,mod]` and compiles through G105's direct
`compile_from_y1_state`; it contains no `code[1200,mod]`, even row, or pose
tensor/config. `20_generated_y1_pose_initializer.npz` stores the folded
encoder-only `xi_init[600,6]`, marks itself non-candidate, and requires the
post-G105 refit. The official G110 compiler accepts only the self-hashed G112
partition receipt; that opener recursively reopens both physical children and
every immutable source-node receipt back to a unique physical zero-parent cold
root. It accepts neither separately caller-selected children, a full G111
checkpoint, nor caller-supplied semantic packet bytes.

The official compile is fail-closed until
`tac.g110_post_g105_generated_y1_pose_refit_run.v1` and its strict final NPZ
exist. That receipt must bind the physical G112 partition, exact semantic
packet, rendered final-Y1 population hash, G109/Pose target custody, initializer
hash, exact source domain `parsed_g105_y1_v10_camera`, exact render order,
seed/command/Git SHA, `--resume-from`, every preserved stage checkpoint, and
the final stage. Concretely, the new receipt binds the G112 partition receipt,
both physical child SHAs, current deploy/resume/source-node receipts, current
checkpoint ID, and cold-root identity rather than taking a full G111 file or
copyable freshness marker as candidate custody.
`exact_public_receiver_in_loop=true` is mandatory. Precompile xi is
initializer/evidence only.

### G112 partition boundary

G112 commits `8dbf19b506`, `52e6c08e25`, and `da27543d7d` close the
checkpoint-shape, selected-preimage, and physical-ancestry seams. G110 consumes
`tac.g112_exact_checkpoint_partition.v2` through
`open_g112_partition_receipt`, requires
`source_chain.complete_trajectory_proven=true`, matching semantic-packet and
G109-projection identities, and reopens the bound n600 batch-16 target
aggregate. Cross-partition child mixing fails inside the G112 opener. The older
full-G111 intake remains only an encoder-side exclusion/evidence helper and is
not an official candidate-compile input.

### Adversarial P0 closure

The review found and the integrated stack now closes two structural seams:

1. G111 `generated_y1` now uses the same sparse exact V10 selected preimage as
   public G110 via `tac.v10_factor2_selected_preimage.v1`: a NumPy authority
   plus a device-resident differentiable MLX gather/mask map with scorer-round
   STE. G112's v2 initializer contract preserves and requires that schema, and
   G110 refuses an initializer without it.
2. G105 exposes typed `RAW_I16_LE` and `DELTA_RICE_BEST_K` wires, each
   canonical within its family. G110 renders both to prove identical ordered
   n600 Y1, builds the complete `{raw,Rice} × {STORE,DEFLATE}` product matrix,
   selects by exact final archive bytes with a stable typed tie-break, and
   retains all four byte/SHA/method records. This correctly handles both inner
   semantic-wire inversion and the case where STORE beats DEFLATE.
3. A first public plugin load previously wrote `__pycache__`, which made the
   exact filename allowlist reject a second decode from the same extraction.
   The loader now suppresses bytecode mutation, and a repeated-load regression
   proves the runtime tree is unchanged.

## Public runtime implementation — execution closure still owed

`submissions/robust_current/g110_two_layer_receiver` is standalone NumPy,
Python standard library, and the already-required Brotli runtime:

- `inflate.sh` invokes only its sibling `inflate.py`;
- each plugin directory has an exact filename-to-variant allowlist, so an
  injected or renamed plugin fails closed;
- semantic plugins contain the committed G103 receiver and receiver-only G105
  V9 HOSC implementation, with no repository/canonical-loader import;
- the two conditional plugins have disjoint packet magics and own strict
  envelope/Rice/XIP2 parsing, Y0 reconstruction, and the final-Y1
  population-binding check;
- repo parse-back refuses oversized compressed members before decompression,
  and producer-custody files are rejected as symlinks before resolution;
- plugin loading suppresses bytecode writes, so a first decode cannot create a
  `__pycache__` entry that invalidates the exact allowlist on a second decode;
- the V9 plugin derives its pair-invariant Fourier grid, dequantized float64
  parameters, and Y1 code once at strict parse time and reuses those immutable
  arrays across all 600 renders rather than rebuilding them per pair; the
  feature cache is bounded at 512 MiB and oversized configs fail closed;
- conditional bilinear upsampling is the canonical G95/G107
  `align_corners=False` half-pixel kernel: coordinates are clipped before
  weights are computed;
- `inflate.py` keeps output under a temporary name until all 600 Y1 frames
  close the packet binding, checks the variant-returned camera Y0 ABI, checks
  exact raw length, fsyncs, then atomically publishes. Low-rank Y0 uses
  factor-2; pose Y0 is already the exact native stored-video frame.

The runtime contains no scorer, source, target table, or video-derived value.
All learned semantic and conditional values are counted packet sections. This
tree has not yet performed a clean-extract public double decode, so G110 does
not claim receiver closure.

## Producer custody and fail-closed boundary

The repo-side compiler accepts no custody capability object. At every compile
call it content-reads and rederives:

1. G109 `tac.taskspace_v9_training_target_capsule_aggregate.v1` through its
   strict loader, including all 600 labels/margins/PoseNet-6 targets, every
   batch checkpoint, recursive G46 source custody, scorer files, and the
   portable upstream closure;
2. the committed G105 adapter source bytes plus a physical fresh G105 NPZ;
   it reconstructs the exact G109 projection from the reopened aggregate,
   verifies the checkpoint binding, recompiles the V9 packet from checkpoint
   tensors, and requires byte identity with the supplied semantic packet;
3. a self-hashed `tac.g110_fresh_conditional_y0_operand_receipt.v1`, its
   physical strict operand NPZ, and a separately sealed resumable producer-run
   receipt. The run reopens every preserved stage checkpoint, requires the
   final stage to be the operand NPZ, and binds the semantic packet, G109
   receipt, Pose target array, exact basis/scales/coefficients, seed, command,
   source Git SHA, fresh lineage, and joint-pose conditioning.

For `G110PC01`, item 3 is replaced by the physical post-G105 refit chain above.
The G111 checkpoint itself must state
`source=generated_y1`, `residual_mode=table`, native `[874,1164]`, the exact
xi-fold formula, pitch/calibration, and the pose-checkpoint contract schema.
Partial or extra `pose_carrier.*` tensors fail closed.

`G110Batch16SourcePoseCustodyV1` is only an internal result of those physical
checks. `compile_g110_two_layer_v1` has no `custody` parameter; an exact-type
`object.__new__` forgery has nowhere to enter the compile path.

G109 is committed at `1689d2025d` and its strict preflight is green. No
materialized G109 aggregate paired to a fresh G105 semantic packet has been
supplied to this compiler, and no conditional-operand producer writes the
third receipt yet. Therefore candidate compilation is deliberately blocked.
The lower-level receiver packet builder is exercised only by tests and carries
no source/candidate claim.

## Canonical-vs-unique decision per layer

- Commit serializer, review tracker, G109 strict loader, and G105 packet parser:
  adopt canonical because their content-reading custody and wire identities
  fit this product exactly.
- Outer archive and conditional wire: unique G110 implementation because G107
  depended on an untracked helper and represented a redundant scale gauge.
- Public runtime: unique complete G110 tree so recursive receiver closure does
  not rely on repository imports or mutate G108 historical evidence.
- Semantic backends: preserve each architecture-specific implementation behind
  one behavioral ABI; do not reinterpret V9 as G103.

## Triality and wire-in

- DSL: the typed G110 envelope, provider dispatcher, source/pose custody, and
  conditional-operand receipt gate.
- DAG: G103/G105 semantic packet → final-Y1 population binding → conditional
  Rice quotient → two scorer planes → V10 factor-2 → public raw.
- Equations: the quotient and temporal code equations above.

Sensitivity-map/bit-allocation contribution: the packet exposes exact semantic,
basis, scale, and Rice-stream bytes for whole-archive arbitration. Pareto
constraint: only joint `100*d_seg + sqrt(10*d_pose) + rate` exact rows may
promote it. Autopilot/continual-learning hooks remain blocked until the first
receiver-closed exact row exists; registering proxy effects would create
false authority. The G109 and conditional-receipt failures are the explicit
probe-disambiguators.

## Stores consulted

- `CLAUDE.md` / `AGENTS.md` and `PROGRAM.md`;
- current lane registry and subagent progress;
- G103 wire and committed G108 public runtime;
- frozen G105 exact V9 adapter and tests;
- frozen G107 conditional product and tests;
- committed G109 aggregate loader/spec at `1689d2025d`;
- V10 factor-2 realization implementation.

## Verification after independent adversarial P0/P1 repair

- Ruff: clean on all G110 Python files.
- built-in `compile()`: clean on all 9 G110 Python sources without creating
  volatile bytecode.
- focused suite: 29 passed in 66.07s on base dependency commit
  `a0164871e103a128af1e32ef6cabc34056944ccb`.
- public G103 frames match the committed repo receiver at pairs 0, 137, 599.
- public V9 frames match G105 at pairs 0, 17, 599.
- public conditional decode matches repo decode, including a legal zero row.
- both low-rank scorer planes pass exact V10 factor-2 verification.
- generated-Y1 public XIP2 decode and native camera warp match the repository
  fp64 NumPy authority exactly on the actual final camera-Y1 ABI.
- G105 public parse/re-emission preserves both canonical-within-family typed
  raw-int16 and best-k delta-Rice alternatives; G110 selects only after exact
  complete-archive measurement.
- one recursively verified, self-hashed G112 partition receipt is the sole
  official compile input; separately selected physical children, a full G111
  candidate input, caller-supplied semantic bytes, and hash-only child custody
  construction are blocked.
- post-G105 refit verification physically reopens its final NPZ, self-hashed
  run receipt, and preserved resumable final stage.
- ZIP parse-back is deterministic and one-member.
- oversized compressed archive members and custody-file symlinks fail closed.
- an `object.__new__` custody forgery cannot enter compilation.
- the physical conditional checkpoint/run/stage chain is reopened.
- the 2×3 boundary vector matches the independent canonical G95 kernel,
  including exact corner replication.
- sign/scale gauges, decoder-dead ranks, and finite-input overflow refuse.
- public plugin filenames and variant IDs are exact allowlists; the forbidden
  repository/scorer-import scan covers every runtime source file.
- the stable rank-zero semantic-floor APIs create and strictly parse an actual
  n600 `Y0 == Y1` G110 product and typed STORE/DEFLATE archive variants for the
  downstream stage selector.

Still not performed: a real full-n600 producer through the new exact V10
training path, a real post-G105 refit, clean-extract public double decode of
its archive, fresh archive parse-back, or `upstream/evaluate.py` n600.

Pointer delta: zero. The effective frontier remains an external target, not a
G110 result.
