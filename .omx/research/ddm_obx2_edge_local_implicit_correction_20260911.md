# ddm_obx2 — edge-local implicit correction lattice: burn record

<!-- FORMALIZATION_PENDING: process/landing record for an in-flight governed burn; the canonical equations it would register are the gate arithmetic already registered under the contest objective, and no new law is claimed until the burn produces a terminal parsed row. -->

Date: 2026-09-11
Status: `IN FLIGHT — STAGES 0, 1, 7-HARNESS BUILT AND PASSING; THREE GOVERNED RUNS LIVE`
Measurement axis: `[macOS-CPU advisory]`
Score claim: false
Promotion eligible: false
Pointer moved: false
Lane: `ddm_obx2_edge_local_implicit_correction_20260911`
Custody root: `/Volumes/VertigoDataTier/pact/ddm_obx2_edge_local_implicit_correction/`

## Verdict so far

Four measured results and one derivation have already changed what this burn is about.

**1. The born object's 8.6267 is a training-coverage number, not a capacity number.**
Recomputed from the retained OBX1 per-pair rows (n600, $0, no new compute):

| population | pairs | d_seg | d_pose | distortion |
|---|---:|---:|---:|---:|
| all | 600 | 0.0632573 | 0.529448 | **8.6267** |
| the 32 pairs QBT2B r10 was trained on | 32 | 0.0030214 | 0.00110326 | **0.4072** |
| the 568 it was never trained on | 568 | 0.0666508 | 0.559214 | **9.0299** |

The untrained pairs are **22.1×** worse on Seg and **507×** worse on Pose. This is structural, not
incidental: `experiments/ddm_qbt1_qbflow_trainer.py::validate_config` REFUSES any training set other
than the sealed 32-pair selection, so the object ships 600 latent records of which 568 were never
optimized. The capacity demand at full coverage is therefore **10× on Seg and 6.9× on Pose**, not the
215× the population number suggests. Pose is the binding term even where the object is trained: its
own trained pairs give `sqrt(10 * 0.00110326) = 0.105`, already 2.6× the whole 0.04 budget.

**2. The macOS-CPU frozen-scorer harness reproduces the contest-CUDA authority row to 0.04%.**
Scoring move 44's retained decoded bytes over all 600 pairs:

| axis | d_seg | d_pose | distortion |
|---|---:|---:|---:|
| `[contest-CUDA T4 n600]` authority | 0.00010345 | 4.59e-6 | 0.0171200 |
| this harness, `[macOS-CPU advisory]` | 0.000103412 | 4.58687e-6 | **0.0171139** |

0.036% apart on distortion. Advisory rows from this instrument are tight at this operating point.
This is the denominator control the burn spec asked for.

**3. The QBF 384×512 render grid does not refuse the gate. MEASURED on n600.**
A `[0,255]` render on the QBF grid, back-projected so its camera image matches the teacher in the
scorer's own plane, is an ACHIEVABLE construction, so its distortion upper-bounds the minimum
distortion of any renderer on that grid. Complete 600-pair rungs:

| rung | d_seg | d_pose | distortion | scorer-plane RMSE | passes 0.04 |
|---|---:|---:|---:|---:|---|
| `teacher` (identity) | 0.000103412 | 4.58687e-6 | 0.0171139 | 0 | yes |
| `sp_874x1164` (control) | 0.000103412 | 4.58687e-6 | 0.0171139 | 0 | yes |
| **`sp_384x512`** | 0.000111576 | 4.52317e-5 | **0.0324253** | 0.160 | **yes** |
| `grid_384x512` (area down, bicubic up) | 0.000435232 | 0.011817 | 0.387281 | 4.544 | no |

**The render grid is cleared.** A renderer on the QBF 384×512 grid reaches distortion 0.0324 over all
600 pairs, inside the gate with 0.0076 of margin. At the 122,000 B packet gate that is
`0.0812348 + 0.0324253 = 0.1136601`, sub-0.12 by 0.0063. The cost of the grid is almost entirely
**Pose**: d_seg barely moves (1.034e-4 → 1.116e-4) while d_pose rises **9.86×**
(4.587e-6 → 4.523e-5), contributing 0.02127 of the 0.03243.

The burn's question is therefore now exact: **can 122,000 B encode a render close enough to the
`sp_384x512` field?** Not whether the architecture's output space is rich enough — it is.

Two supporting cautions, both measured on n600. The naive `grid_384x512` construction (area down,
bicubic up) scores 0.387281 — **11.9× worse** than the scorer-plane-matched render on the SAME grid,
and **261× worse on Pose** (0.011817 vs 4.52e-5) against only 3.9× on Seg. Had Stage 2a used only that
rung it would have produced a false structural refusal of the whole architecture. And the 4-pair smoke
of `sp_384x512` read 0.01475 against the n600 value of 0.0324 — **2.2× optimistic** — so no prefix
here is a verdict, in either direction.

**4. The photometric-distillation route is closed by arithmetic — the object must be a witness.**
DERIVED from three measured n600 points, and pre-registered for further confirmation by the queued
`sp384_render_noise_*` rungs. At distortion `< 0.04` with the measured `d_seg ~ 1.1e-4`
(0.011 of the budget), the Pose budget is `d_pose < 8.4e-5`. Fitting
`d_pose = 4.587e-6 + C * spRMSE^p` through `teacher` (0), `sp_384x512` (0.160, 4.523e-5) and
`grid_384x512` (4.544, 0.011817) and `sp_192x256` (8.670, 0.0391067) gives a MEASURED exponent **p = 1.7121**, `C = 9.294e-4` — the
growth is sub-quadratic, and the fit holds within **5.1%** across three decades of RMSE. The
admissible scorer-plane RMSE is **0.236 of one uint8 LSB, 0.093% of full scale** — near-lossless. No
122,000 B object encodes 1,200 frames of 384×512 at that fidelity. Registered as canonical equation
`obx2_pose_vs_scorer_plane_rmse_v1`.

The consequence is not that the burn fails. It is that **matching the teacher photometrically is the
wrong objective**: the constraint is on PoseNet's six outputs, not on the image, and the set of
pose-equivalent images is vastly larger than a neighbourhood of the teacher. Both live training runs
are therefore JOINT SCORER DESCENT, not Stage-2 teacher distillation. This is the
evaluator-equivalent-witness paradigm arriving as arithmetic rather than as doctrine.

A supporting measured fact points the same way: uniform noise applied ONLY outside the GT argmax
boundary band still drove `d_pose` to 2.82 on a 4-pair smoke. **Pose damage is not edge-local**, so a
purely edge-gated correction cannot reach the term that binds — which is why the `interface_blend`
gate (D13) exists.

**5. The first trained object: the binding term INVERTS, and the pinned pose weight was 4× wrong.**
The base-only control's epoch-20 object, byte-closed from its own checkpoint and scored on n600
through the SHIPPING torch receiver:

| quantity | value |
|---|---:|
| archive | **109,104 B** (PASSES 122,000 B, 12,896 B headroom), sha `4b622dc8c1287c38…` |
| `d_seg` | 0.02450141059 |
| `d_pose` | 0.01773646324 |
| distortion | **2.871287865** — Seg leg 2.45014, Pose leg 0.421147 |
| advisory S at its own bytes | 2.94393574 |
| gates | byte gate PASS; **distortion gate FAIL by 71.8×** |

Twenty epochs of n600 joint descent took distortion 8.6267 → 2.8713 (3.0×), with **Pose improving
29.9×** and Seg 2.6×. So **Seg now dominates**: its leg is 5.8× Pose's. That inverts the reading the
Stage-2a constructions gave, where Pose carried almost all the cost — because those constructions
started from a photometrically perfect teacher, and a trained generator does not.

The consequence is immediate and was acted on. The contest Pose term's derivative at THIS measured
operating point is `5/sqrt(10 × 0.01773646324) = 11.87234458`. Both live arms were training under a
pinned **47.67312946** — **4.015× overweight** — aiming capacity at the smaller leg. Both were stopped
at their newest checkpoints and re-opened warm-started as stage `w2` with the derived weight, their
earlier checkpoints preserved under their own names. The derivation pins its artifact, sha, receiver
and 600-pair denominator; it is registered as `obx2_pose_vs_scorer_plane_rmse_v1`.

Two honest caveats on this row. The NumPy cross-check ran on a **40-pair prefix** (d_seg 0.026926,
d_pose 0.018335, distortion 3.1208), so it is NOT a like-for-like receiver comparison against the
600-pair torch row — a flaw in how I wired the cross-check, recorded rather than papered over. And
`pose_budget_at_gate` is 0 for this object: at `d_seg` 0.0245 the Seg leg alone already exceeds the
whole 0.04 gate.

**6. The expected-flip surrogate tracks the measured `d_seg` to 1.0%.**
Stage `w2`'s first logged epoch reads `mean_expected_flip = 0.02474` against the parsed object's
MEASURED `d_seg = 0.02450141` — **1.0% apart**. So `d_seg` can be read off the training log without
spending a scoring pass, which makes the Seg leg observable every epoch instead of every ten. The
Pose surrogate is looser: `mean_pose_mse = 0.024805` against a measured `d_pose = 0.01774`, reading
**1.4× high**, which is the gap between the training-time MSE on MPS through the STE round trip and
the parsed object measured on CPU. Neither is a score; both are now logged as components.

**7. The base-only arm is CLOSED — by the store, not by my fit.**
MAIN ordered a recall before any seg cure, and the recall decided the unit. `md1`-`md4`
(`.omx/research/ddm_md1_micro_to_macro_dynamics_20260904.md`, 282 forwards over 71 checkpoints, then
8 legitimate starts, then a burned different-start cell) measured **on the QBF1 born vehicle — this
exact generator form**:

* 62.0% of the terminal `d_seg` is PERSISTENT (wrong at ≥90% of 71 checkpoints), and the persistent
  set is the SAME set across schedule, data order and start (Jaccard 0.73-0.85 against nulls of
  0.35-0.53);
* deleting every optimizer-reachable site still leaves a floor of `d_seg` **0.00174 — 12.75×** the
  accuracy corner;
* every schedule/optimizer/objective lever's **combined ceiling is 1.61×** against the **20.57×**
  needed, and in md1's own words *"the qn1 n600 realization is moot for sub-0.12 on this vehicle"*;
* md3/md4: *"the sites are scorer-hard for this generator FORM; the born accuracy corner needs a
  different GENERATOR FORM, never a start/schedule/seed"*, and the cure it names is **"a Lane carrier,
  not a loss."**

The w2 stage is an **objective lever on that exact generator form**. It is closed at FORMULATION
scope by four data-anchored prior instances. The base-only arm was stopped, checkpoints preserved.
My own pre-registered seg-slope falsifier could NOT have decided this yet — it read INDETERMINATE on
a 1.05× lever arm — which is precisely why the recall order mattered more than the fit.

What survives, and is now the only arm: **the lattice is a representation change and a counted
carrier**, which is the class md1-md4 point at. Nothing in md1-md4 closes it; its own conditional
(*"with capacity gc1 and form gf2"*) excludes exactly the thing the lattice changes.

**A correction to carry back.** The `bz2d` headline "token error amplifies ×1.157 to argmax" was
RETRACTED the same day it was measured. The store reads: *"THE RATIO DOES NOT TRANSFER"* — the ratio
moves 1.16× to 1.97× between two points, and the real relation is AFFINE,
`argmax_errors ≈ 17,241 + 1.1435 × tokens`, where **the intercept is a render-manufactured floor no
token work can remove**. For a witness that regenerates the partition rather than coding tokens, the
transferable part is the intercept's existence, not the ratio.

**8. The seg leg is 83% PARTITION error, not render floor — design A targets the smaller share.**
MAIN asked how much of the object's `d_seg` is token-level partition error, which no RGB correction
reaches, against render floor, which is what the lattice targets. The object ships no token field, so
the analogue is the generator's own internal class head — the partition it represents BEFORE the
render turns it into RGB. On a **4-pair smoke** (the n600 pass is running; a prefix is not a verdict):

| quantity | value |
|---|---:|
| `d_seg` at the scorer | 0.026754 |
| the generator's OWN partition vs GT | **0.078646** — wrong on 7.9% of pixels |
| the pointer's token plane vs GT | 0.000164 — wrong on 0.016% |
| the object's partition vs the pointer's tokens | 0.078650 |
| share of seg error where the partition was ALREADY wrong | **82.7%** |
| share that is render floor | 17.3% |

If the lattice eliminated the **entire** render floor, `d_seg` would fall 0.026754 → 0.022127, a
**1.21×** improvement against the ~60× the gate needs. On this reading design A is not aimed at the
binding term. It also matches md1-md4 without being fitted to them: *"the sites are scorer-hard for
this generator FORM."*

One nuance the numbers force, against the word "floor": the render-plus-scorer path **recovers** more
partition error than it creates. The generator's partition is wrong on 7.9% while the scorer's argmax
is wrong on 2.7% — the path repairs about two thirds. The same holds for the pointer: its token plane
is wrong on 1.64e-4 and its seg leg is 1.03e-4. So the path is a smoother in both directions, and the
decomposition measures the NET at each pixel, which is the quantity that matters.

**8b. The residual is NOT the pointer's boundary jitter — a third of it is region-level. MEASURED n600.**
The lattice arm's w2 epoch-30 object, scored on all 600 pairs through the shipping receiver:

| quantity | value |
|---|---:|
| archive | **122,778 B** — byte gate **FAIL by 778 B** |
| `d_seg` | 0.02271701389 |
| `d_pose` | 0.01138376106 |
| distortion | **2.609099686** — Seg leg 2.2717, Pose leg 0.337398; **FAIL by 65×** |

Its 2,679,808 misclassified pixels, binned by hop count to the nearest GT class change:

| distance | 0 | 1 | 2 | 3 | 4 | >4 |
|---|---:|---:|---:|---:|---:|---:|
| share | 0.377 | 0.133 | 0.070 | 0.042 | 0.033 | **0.345** |

**51.0% sits within one cell of a boundary and 34.5% sits more than four cells in.** Set that beside
the store's reading of the pointer's own residual — *"99.58% are one-pixel boundary displacements"* —
and the two objects have qualitatively different debts. The pointer's is jitter at a correct
partition; a third of this object's is a region wearing the wrong class, which no edge-local
mechanism reaches and which agrees with the 83% partition-level share above.

**9. The lattice's own bytes: the epoch-30 object is 778 B OVER the gate.**
Packet 122,668 B / archive **122,778 B** against the 122,000 B gate. The lattice section alone codes
to **16,133 B** (18,826 B raw) — above the ~15,400 B the budget left — because the trained codes are
dense: 98-99% nonzero on every level and saturating at ±127. So the correction is currently spending
13% of the packet, and by the decomposition above it is spending it on the 17% share.

**10. MAIN's rate rule, arithmetic.** Over 18 epochs (20→37) the lattice arm's surrogate moved
0.0254741 → 0.0229619: a measured **0.6089%/epoch**, already **36% below** the 0.95%/epoch MAIN set as
the continue-threshold (that 0.95% was itself measured over the stage's first 7 epochs, so the rate
has decayed). At that constant rate `d_seg` 4.0e-4 arrives at **epoch 700**, 3.5× outside the stage.
The pre-registered falsifier still reads INDETERMINATE — the 1.109× fall is under its 1.25× lever —
and MAIN's 20-epoch window is two epochs short of closing.

## What is implemented and proven

| stage | result | receipt |
|---|---|---|
| Stage 0 — storage and identity | PASS. Every frozen pin matched; two-runs-plus-reserve projection admitted against 78.8 GiB free on Vertigo. | `checkpoints/stage_00_identity.json` |
| Stage 1 — receiver parity | PASS. A zero lattice is the born object **exactly** (max abs 0.0) through the parsed packet. Encoder repeats byte-identically. torch twin vs the float64 NumPy receiver: relative-L2 parity **0.9999908**, max abs 0.00124, 0.022% of rounded uint8 values disagree. Zero-lattice packet **108,826 B**, 13,174 B under the gate. | `checkpoints/stage_01_receiver_parity.json` |
| Stage 2a — gate pricing (declared) | RUNNING, n600, 16 rungs; 3 complete (rows 2 and 3 above), including the decisive `sp_384x512`. The queued `sp384_render_noise_*` rungs confirm or refute derivation 4. | `STAGE_2A_RESULT.json` when complete |
| base-only n600 control | RUNNING, 200 epochs, MPS, joint scorer descent, lattice frozen. Loss 13.30 → 4.17 by epoch 24, then flat. Its **epoch-20 object is byte-closed: packet 108,994 B, archive 109,104 B, 12,896 B under the gate**, sha `4b622dc8c1287c38…`; its n600 distortion row is computing. | `base_only/CHECKPOINT_SCORE_*.json` |
| base+lattice n600 | RUNNING, identical config with the lattice live: the A/B that isolates the lattice's marginal contribution. | `lattice/STAGE_JOINT_RESULT.json` |
| Stage 7 — public timing harness | BUILT. Decodes the exact archive twice, requires byte-identical output inside 1,260 s. Not yet run on a candidate. | `STAGE_7_TIMING_<receiver>.json` |

Implementation: `src/tac/obx2_lattice_packet.py` (grammar + NumPy reference receiver, 41 tests),
`experiments/ddm_obx2_edge_local_implicit_correction.py` (Stages 0/2a, 21 tests),
`experiments/ddm_obx2_trainer.py` (model, packet builder, Stage 1, training stages, 10 tests).
`ruff` clean; every `.py` reviewed twice and committed through the serializer.

## Declared decisions

Every one of these is a change or an addition the burn spec left open. None is silent.

- **D1 — Stage 2a inserted.** An object-free, n600, `$0` pricing stage before the training burn, under
  the CLAUDE.md Carmack MVP-first rule. It builds, trains, and admits nothing; it measures the teacher
  ceiling and a deterministic corruption ladder so the spec's own `distortion < 0.04` gate is priced
  before days of training are spent. Results 2 and 3 above are its output.
- **D2 — scorer-plane back-projection as the grid bound.** Rungs `sp_<H>x<W>` construct the render by
  iterative back-projection against the scorer plane rather than by downsampling the camera image.
  Reason: measured, 8× (result 3). Convergence is recorded per iteration, never assumed.
- **D3 — retention policy.** The `sp_384x512` and `grid_384x512` camera payloads and the `sp_384x512`
  render are retained verbatim; every other rung records its exact deterministic recipe, master seed,
  and per-chunk SHA-256 of the materialized camera bytes, so those bytes are provably rebuildable from
  the retained teacher. Scorer outputs (argmax, pose) are retained for every rung; logits for four.
  No measured payload is reduced to a scalar.
- **D4 — the binding rate gate is the complete 122,000 B packet.** The spec's 80,468 / 41,532 split is
  its default allocation, not a separate constraint; both are reported.
- **D5 — the base is fully trainable.** All QBF parameters and latents are re-encoded into the OBX2
  packet. They are counted either way, and result 1 says the base is the under-trained part.
- **D6 — training device MPS, verdict authority NumPy.** MPS is a gradient device only. Every admission
  number comes from the NumPy reference receiver on the parsed packet plus the frozen CPU scorers.
- **D7 — the OBX2 packet is its own grammar.** New magic and version; the born packet is a pinned input
  of other live work and is never mutated. Sections 1-4 carry through, section 5 is the lattice.
- **D8 — the gate is recomputed, never shipped.** The receiver derives the edge-local gate from the
  born generator's own decoded signed-interface field. Only the single gate width is counted. No
  support map, position list, or mask is shipped (rule 118).
- **D9 — the Stage-2 distillation target is the `sp_384x512` render, not a downsample of the camera
  teacher.** Reason: measured (result 3). Distilling the naive construction would aim the generator at
  a target that cannot pass the gate.
- **D10 — the lattice sampler is eight explicit gathers, not `grid_sample`.** Reason: measured.
  `aten::grid_sampler_3d_backward` has no MPS kernel, so `grid_sample` costs the gradient device; the
  gather form is the same arithmetic in the same order as the NumPy receiver, and it takes an n600
  epoch from **32.4 min on CPU to 1.2 min on MPS (27×)**.
- **D12 — the rate term is charged on `archive.zip`, not the packet.** The builder emits the
  deterministic archive alongside the packet, proves the archive builder repeats and that its receiver
  returns the exact packet it was built from, and every advisory row's rate comes from the archive it
  would ship.
- **D13 — the `interface_blend` gate (kind 2).** The head emits a near and a far correction and blends
  them with the same recomputed gate, so the lattice keeps its edge-local half AND gains global reach.
  Reason: measured — Pose binds, and Pose damage is not edge-local. Cost: 6 more counted head outputs.
  Both gate kinds pass Stage-1 receiver parity. Implemented and ready; the first two arms run kind 1 so
  the A/B stays clean.
- **D11 — the lattice geometry is sized from a real coder result.** The first geometry's 101,760 codes
  coded at **7.11 bits/code** (90,415 B) in a real Brotli q11 / zlib 9 / LZMA2-extreme race — six times
  past budget. The default is now 15,840 codes against the ~15,400 B the packet actually leaves.

## Measured risks that are not yet closed

- **Public decode time — CURED, with a named consequence.** The portable float64 NumPy receiver costs
  **4.32 s/pair → 2,593 s for n600**, **2.06× the 1,260 s budget**, before the lattice. The same
  arithmetic in torch-CPU on the PARSED packet costs **0.311-0.557 s/pair → 187-334 s**, comfortably
  inside it, and is generic free code under rule 118. The consequence is that the two receivers are
  not interchangeable: on a trained lattice they disagree on **0.093% of rounded uint8 values**
  (max abs 0.0075 in [0,1] render space, relative-L2 parity 0.99996). Every advisory row now names its
  receiver; validation measures the shipping torch path on all 600 and keeps the portable NumPy path
  as an explicit cross-check. Which receiver ships is a Stage-7 decision, and the object must be
  scored through the one that does.
- **The governor refused two launches, and was right both times.** The first: I declared 24 GiB of peak
  RSS for the parallel lattice arm against a MEASURED training-child RSS of **1.01 GiB**; relaunched at
  8 GiB. The second: with four live processes the box was genuinely at 121.2 GiB projected against a
  116.0 GiB ceiling, so the pose ladder is QUEUED behind the scoring run rather than forced. Recorded
  because the first error was mine, and because the second is the honest state of a full machine.
- **The scorer batch was 50 pairs and peaked past 16 GiB of RSS.** Fixed: rendering still walks 50-pair
  chunks, scoring walks them in batches of 8. The two in-flight training arms carry the old code, so
  their terminal validation may still peak there; their checkpoints are saved every 10 epochs, so a
  killed validation costs a rescore, not the run.
- **MPS training is not bitwise reproducible across hosts.** The shipped artifact is deterministic and
  hashed, and the run is resumable from disk, but the training trajectory on MPS is not bit-identical.
  Declared, not hidden.
- **Rate.** At 122,000 B the object has ~15,400 B for the lattice once the born model (79,688 B) and
  latents (26,130 B) are paid for. Whether that buys the 10× Seg and 6.9× Pose that result 1 demands
  is the burn's open question.

## Live hypothesis: pose belongs to geometry, not to a photometric correction

Recorded, scoped, and NOT pursued without its falsifier firing first.

Every measurement in this burn puts Pose in the binding seat and says the damage is not edge-local:
the render grid costs 261× on Pose against 3.9× on Seg; noise applied only away from the argmax
boundary still drives `d_pose` to 2.82; and the admissible scorer-plane RMSE is 0.238 of one LSB. A
photometric correction lattice — even the blended one (D13) — is being asked to fix a term whose
error may not be photometric at all.

PoseNet estimates a six-degree-of-freedom twist from the APPARENT MOTION between a pair's two frames.
A coordinate generator renders both frames independently from one latent and a frame index; nothing in
its form makes the pair geometrically consistent. The structural alternative is to render frame 0 and
PRODUCE frame 1 by an explicit counted warp of it — the `se(3)` ego-screw the unified level-set
reading already names, where the same twist that moves the partition IS the pose. That is six counted
numbers per pair (3,600 values for the whole video), not a photometric field. It is joint, in the
forward pass, and counted — not the post-hoc stored sidecar family that earlier work found walled.

**The `$0` falsifier that must fire first**, because the hypothesis is worthless if PoseNet is not
reading geometry: perturb the teacher's frame 1 by a pure sub-pixel geometric shift at several
magnitudes and measure `d_pose` against the same photometric-RMSE budget as the existing rungs. If a
one-pixel shift moves `d_pose` far more than photometric noise of equal RMSE, Pose is geometry-bound
and the warp is the right cure. If the two responses are comparable, Pose is photometry-bound, the
warp buys nothing, and this hypothesis is closed before any of it is built.

## Custody

Retained under `/Volumes/VertigoDataTier/pact/ddm_obx2_edge_local_implicit_correction/`:
pinned inputs with SHA-256, Stage-0/1 receipts, Stage-2a per-rung chunks with per-chunk payload
facts, training checkpoints under distinct stage-encoded names, and terminal packets. Nothing was
deleted or moved. Every `/Volumes/...` source was read-only. No Modal call, no paid dispatch, no
contest evaluation, no candidate archive.

## NEXT_IF_RESUMED

```bash
# Stage 2a gate-pricing ladder (resumes per (rung, chunk) from retained checkpoints)
.venv/bin/python experiments/ddm_obx2_edge_local_implicit_correction.py stage2a --launch-authorized

# base+lattice n600 arm (identical, without --no-lattice; --gate-kind 2 for the declared blend lever)
# base-only n600 control (resume from the newest stage-encoded checkpoint)
.venv/bin/python experiments/ddm_obx2_trainer.py joint --launch-authorized --no-lattice \
  --device mps --epochs 200 --chunk-pairs 4 --learning-rate 3e-4 \
  --pose-weight-operating-point 1.1e-3 --save-every-epochs 10 --workers 8 --validate-pairs 600 \
  --output /Volumes/VertigoDataTier/pact/ddm_obx2_edge_local_implicit_correction/base_only \
  --resume-from <newest checkpoints/obx2_joint_epoch_*.pt>

# public timing on any terminal candidate
.venv/bin/python experiments/ddm_obx2_trainer.py stage7 \
  --archive <.../candidates/obx2_joint_terminal.archive.zip> --receiver torch
```

Live governed runs at hand-back: `stage2a` (pid 82411), `joint_base_only` (pid 20844),
`joint_lattice_r2` (pid 45160). Done receipts land in `.omx/tmp/codex_runs/<name>.done.done`.

The frontier is unchanged: **composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600]
(move 44)**.
