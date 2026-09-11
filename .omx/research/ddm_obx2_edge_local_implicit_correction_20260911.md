# ddm_obx2 — edge-local implicit correction lattice: burn record

<!-- FORMALIZATION_PENDING: process/landing record for an in-flight governed burn; the canonical equations it would register are the gate arithmetic already registered under the contest objective, and no new law is claimed until the burn produces a terminal parsed row. -->

Date: 2026-09-11
Status: `IN FLIGHT — STAGES 0, 1 PASSED; STAGE 2a AND THE BASE-ONLY CONTROL RUNNING`
Measurement axis: `[macOS-CPU advisory]`
Score claim: false
Promotion eligible: false
Pointer moved: false
Lane: `ddm_obx2_edge_local_implicit_correction_20260911`
Custody root: `/Volumes/VertigoDataTier/pact/ddm_obx2_edge_local_implicit_correction/`

## Verdict so far

Three measured results have already changed what this burn is about.

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

**3. The QBF 384×512 render grid does not refuse the gate — but the naive construction does.**
A `[0,255]` render on the QBF grid, back-projected so its camera image matches the teacher in the
scorer's own plane, is an ACHIEVABLE construction, so its distortion upper-bounds the minimum
distortion of any renderer on that grid. On a 4-pair smoke (not a verdict; the n600 rung is running):

| rung | distortion | camera RMSE | scorer-plane RMSE |
|---|---:|---:|---:|
| teacher (identity) | 0.01151 | 0 | 0 |
| `sp_874x1164` (control) | 0.01151 | 0 | 0 |
| **`sp_384x512`** | **0.01475** | 1.43 | 0.087 |
| `grid_384x512` (area down, bicubic up) | 0.11871 | 3.84 | — |
| `sp_192x256` | 0.69671 | 7.94 | 8.71 |

The naive area-downsample construction is **8× worse** than the scorer-plane-matched one on the same
grid. Had Stage 2a used only the naive rung it would have produced a false structural refusal of the
whole architecture. The grid cliff sits between 384 and 192: at 192×256 the back-projection stalls at
8.7 RMSE because the operator is no longer invertible there.

## What is implemented and proven

| stage | result | receipt |
|---|---|---|
| Stage 0 — storage and identity | PASS. Every frozen pin matched; two-runs-plus-reserve projection admitted against 78.8 GiB free on Vertigo. | `checkpoints/stage_00_identity.json` |
| Stage 1 — receiver parity | PASS. A zero lattice is the born object **exactly** (max abs 0.0) through the parsed packet. Encoder repeats byte-identically. torch twin vs the float64 NumPy receiver: relative-L2 parity **0.9999908**, max abs 0.00124, 0.022% of rounded uint8 values disagree. Zero-lattice packet **108,826 B**, 13,174 B under the gate. | `checkpoints/stage_01_receiver_parity.json` |
| Stage 2a — gate pricing (declared) | RUNNING, n600, 16 rungs. `teacher` rung complete (row 2 above). | `STAGE_2A_RESULT.json` when complete |
| base-only n600 control | RUNNING, 200 epochs, MPS. | `base_only/STAGE_JOINT_RESULT.json` |

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
- **D11 — the lattice geometry is sized from a real coder result.** The first geometry's 101,760 codes
  coded at **7.11 bits/code** (90,415 B) in a real Brotli q11 / zlib 9 / LZMA2-extreme race — six times
  past budget. The default is now 15,840 codes against the ~15,400 B the packet actually leaves.

## Measured risks that are not yet closed

- **Public decode time.** The portable float64 NumPy receiver costs **4.32 s/pair → 2,593 s for n600**,
  which is **2.06× the 1,260 s budget**, before the lattice. The born QBF receiver alone already
  exceeds the budget in its reference form. Named cures, none measured yet: float32, batching over
  pairs, or a generic torch-CPU receiver (all free code under rule 118). Stage 7 is a real gate and
  this is the leading candidate to fail it.
- **MPS training is not bitwise reproducible across hosts.** The shipped artifact is deterministic and
  hashed, and the run is resumable from disk, but the training trajectory on MPS is not bit-identical.
  Declared, not hidden.
- **Rate.** At 122,000 B the object has ~15,400 B for the lattice once the born model (79,688 B) and
  latents (26,130 B) are paid for. Whether that buys the 10× Seg and 6.9× Pose that result 1 demands
  is the burn's open question.

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

# base-only n600 control (resume from the newest stage-encoded checkpoint)
.venv/bin/python experiments/ddm_obx2_trainer.py joint --launch-authorized --no-lattice \
  --device mps --epochs 200 --chunk-pairs 4 --learning-rate 3e-4 \
  --pose-weight-operating-point 1.1e-3 --save-every-epochs 10 --workers 8 --validate-pairs 600 \
  --output /Volumes/VertigoDataTier/pact/ddm_obx2_edge_local_implicit_correction/base_only \
  --resume-from <newest checkpoints/obx2_joint_epoch_*.pt>
```

The frontier is unchanged: **composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600]
(move 44)**.
