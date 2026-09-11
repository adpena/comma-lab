# ddm_obx2 — edge-local implicit correction burn specification

Status: `SEALED_PENDING_EXPLICIT_OPERATOR_GO`  
Owner: MAIN / operator-designated QBF successor owner  
Consumer store: `/Volumes/VertigoDataTier/pact/ddm_obx2_edge_local_implicit_correction/`  
Score claim: false  
Training performed by ddm_obx1: none

## Decision object

Build one new QBF-family object whose counted dynamic section is a jointly trained multiresolution
edge-local feature lattice. The lattice is queried by coordinate and by the born generator's signed
interface state, and it emits a chroma-first RGB correction before the camera-size round trip. It is
not a post-hoc splice, a stored label band, or a regressor over an exact-solve carrier.

The pointer's retained decoded video is a compress-time teacher only. The receiver never sees the
pointer archive, its raw frames, the shipped token field, GT, SegNet, or PoseNet. All video-derived
weights, lattice features, quantizer metadata, and innovations are counted inside the archive.

## Why this formulation remains open

The ddm_obx1 falsifier closes `INSTANCE: exact QBT2B r10 plus decoded int8 Lane-edge correction
lattice`: 4,555,301 B and distortion 14.5660713068. It also measures the unchanged r10 physical
object at distortion 8.6267007501 over n600. That is strong evidence against an additive post-hoc
RGB lattice. It does not test a carrier co-trained with the base generator, a learned interpolation
rule, capacity reallocation out of the original latent section, or a field that can change support
during training.

The earlier BS3/HV3 closure is also respected. That route used an HG1 four-generator categorical
body, an inherited DX2 carrier, and a one-hidden-layer screen downstream of exact solved code deltas.
OBX2 uses the QBF continuous RGB generator, makes the local lattice part of the forward pass from
the first training stage, and optimizes the parsed RGB object through R. No BS3 task is reopened.

## Exact admission arithmetic

The retained QBT2B r10 packet is 106,606 B. Its four coded sections are config 488 B, model 79,760 B,
metadata 20 B, and latents 26,138 B; packet framing is 200 B. Holding the fixed portion at 80,468 B
leaves **41,532 B** for the complete replacement dynamic section under a strict 122,000 B packet
target. The old latents are replaced, not booked as an additional credit.

At 122,000 B the rate term is `0.08123479228090491`, leaving distortion `< 0.03876520771909509`
for sub-0.12. With `d_pose <= 1e-5`, the required `d_seg` is
`< 0.00028765207719095085`; with pointer-like `d_pose = 4.59e-6`, it is
`< 0.00031990253844713356`. These are hard gates, not predictions.

## Frozen input pins

| input | bytes | SHA-256 | role |
|---|---:|---|---|
| QBT2B r10 packet | 106,606 | `607abebda2708f00daab79aac7bc6839d314096e6ed5693b642525487a1019f7` | initialization and packet grammar |
| QBT2B r10 archive | 106,714 | `b26371e50696bdcdafdccbf4c629ef1119ae48aa1ac8765200a6ea2176f91830` | byte/repeat control |
| move-44 decoded raw | 3,662,409,600 | `2b762eba4a20a315c104f8447d6ea0e604f73c3d8b8b69b3fc63b0fc792d59fc` | compress-time RGB teacher |
| shipped move-44 field | 117,964,800 | `a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8` | compress-time support/diagnostic only |
| DALI GT argmax | 117,964,928 | `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248` | scorer target |
| DALI PoseNet first-six | 14,528 | `8d5cfa83df55b89493ba43b1e5386d792c836c32791666192499a089068e7eff` | scorer target |
| OBX1 terminal result | 24,379 | `ae19bb553e6810b4492d2c8df8ba5cec4f319534bf7a559010faa46f9944b81e` | parent negative and denominator control |

Any drift refuses before launch. `pass6.u8` is forbidden; only the shipped `subset6.u8` identity is
admissible.

## Required implementation before burn

1. Add a versioned packet section with explicit lattice levels, channel widths, quantizer kinds,
   entropy model, and CRC. Its decoder must reject unknown, missing, duplicate, or trailing content.
2. Implement NumPy-fp32 reference query/fusion first, then match the training backend at parity
   `>= 0.9997`. MPS is never verdict authority.
3. Make all randomness derive from one recorded seed. Deterministic repeat must reproduce packet,
   archive, decoded camera bytes, and scorer inputs.
4. Train through the exact camera-size resize, uint8 quantization surrogate, scorer resize, and
   decoded lattice values. Every admission measurement uses the parsed packet, never live tensors.
5. Race real Brotli q11, zlib 9, and LZMA2-extreme payloads per section and retain every candidate.
   Rate loss may use a differentiable estimate, but only real physical archive bytes decide gates.
6. Provide a public receiver that consumes every counted section and emits exactly 1,200 RGB frames
   at 874x1164. The receiver must not load scorers or any video-derived content outside the archive.

## Governed stages

All stages operate over the complete 600-pair population; no prefix or reduced-population verdict is
permitted. Each stage writes an atomic, byte-close-loadable EMA checkpoint with optimizer,
scheduler, quantizer, entropy-model, RNG, cursor, configuration, and source hashes. A distinct
stage-encoded checkpoint is preserved; no stage overwrites an earlier one. Long stages also save
periodic checkpoints.

1. **Stage 0 — storage and identity.** Estimate worst-case checkpoint/render/coder volume from an
   instrumented no-training forward. Require enough SSD space for two complete projected runs plus
   20 GB reserve. If neither SSD passes, refuse; no cleanup or unmanifested movement is authorized.
2. **Stage 1 — receiver parity.** Initialize the lattice to exact zero, build/repeat/parse the
   archive, and prove decoded RGB identity to the parsed zero-lattice control. Any nonzero delta is
   an implementation bug.
3. **Stage 2 — pointer-teacher warm start.** Train the base and lattice jointly against the retained
   move-44 RGB teacher through the real R path. Frame 0 receives Pose-first allocation; frame 1
   receives joint Seg/Pose allocation. Preserve the terminal EMA checkpoint even if refused.
4. **Stage 3 — scorer-cell joint descent.** Use the exact contest component formula and the canonical
   operating-point gradients, with the parsed archive in every validation. The lattice support is
   soft/dynamic in training and becomes deterministic from decoded state at inference; no shipped
   position map is allowed.
5. **Stage 4 — rate/quantization closure.** Anneal the real section budget, mixed precision, and
   entropy model while keeping all candidate payloads. Stop if the complete packet cannot reach
   `<= 122,000 B` without increasing parsed distortion beyond 0.04.
6. **Stage 5 — terminal pose finish.** Freeze the Seg trunk and solve the minimal joint-trained pose
   degrees of freedom on the parsed object. A post-hoc stored target table that the render did not
   consume is forbidden.
7. **Stage 6 — sole-lane n600 advisory.** Claim the fleet's only full scorer lane. Measure born
   control and candidate in the same frozen CPU process, recompute every component from 600 pairs,
   and retain frames, logits, argmax, pose vectors, per-pair rows, packets, archives, and repeats.
8. **Stage 7 — public timing.** Decode the exact candidate twice. Both runs must finish within
   1,260 s and emit byte-identical output. Timeouts kill the instance.

## Fire and stop rules

- **Fire trigger:** an explicit operator `GO ddm_obx2`, an active unique train/scorer claim as
  appropriate, clean source/input pins, and a passing Stage-0 storage receipt.
- **No early promotion:** sampled, MPS, proxy, live-tensor, or teacher-space improvements cannot
  admit an archive.
- **Burn admission:** parsed archive `<= 122,000 B`, n600 local-CPU advisory distortion `< 0.04`,
  public decode `<= 1,260 s`, deterministic repeat identity, and every counted section consumed.
- **Candidate-chain trigger:** the same parsed object has recomputed advisory `S < 0.12`. Only then
  may MAIN decide whether to request contest-CPU/CUDA authority.
- **Instance refusal:** either complete packet remains `> 122,000 B`, own distortion remains
  `>= 0.04`, or timing exceeds 1,260 s after the predeclared bounded tuning ladder. Refusal closes
  this seed/configuration ladder only, not all co-trained local implicit fields.
- **Hard stop:** any discarded payload, missing resume state, unpinned source, scorer at inflate,
  video-derived data in free code, denominator drift, or attempt to reuse BS3/HG1 deltas terminates
  the run without a score claim.

## Resource boundary

OBX1 authorizes no training, Modal call, paid dispatch, contest evaluation, or source mutation under
this specification. The operator GO may authorize the burn, but it does not automatically authorize
Modal or contest evaluation; those require their normal lane, budget, and authority gates.

