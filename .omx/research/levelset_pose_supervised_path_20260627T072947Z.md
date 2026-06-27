# Level-set witness — pose-supervised path: VERIFY + derive-from-seg re-center (FEED-dx)

- UTC: 2026-06-27T07:29:47Z · git `a0b3d9c4a`
- Axis: `[macOS-CPU advisory]` / `[macOS-MLX training-gradient]` — NON-PROMOTABLE, pointer UNMOVED (contest-CPU 0.19110). NO score/frontier/promotion claim. Realized verdict = frozen CPU-torch SegNet argmax + PoseNet MSE (NEVER MPS/MLX as authority).
- Constraints honored: CPU-only $0, additive/default-off (NO edit to the trainer/generator), GPU descent (pid 72602) + lane-SDF subagent UNTOUCHED.
- Cross-refs: byte-close `tools/levelset_byte_close_and_eval.py` (FEED-dp), trainer `experiments/train_levelset_witness_realized_through_R_mlx.py`, loss `experiments/train_witness_realized_through_R_mlx.py::make_loss_fn`, `src/tac/pose_from_embedding.py`, `src/tac/nerv_pose_conditioning_bolton.py`, memory `project_posenet_rank1_discovery`.

## Q1 — is `--w-pose>0` the CORRECT store+supervise, or the COLLAPSED reconstruct? → CORRECT (verified, no wiring needed)

The level-set trainer's pose term routes through the imported `make_loss_fn`
(`train_witness_realized_through_R_mlx.py:841-847`). The mechanism is the REALIZED store+supervise,
NOT a luma-INR reconstruct:

```
f0 = render_through_R_mlx(model, cf, code0, ...)   # render frame_0 -> contest-exact R
f1 = render_through_R_mlx(model, cf, code1, ...)   # render frame_1 -> contest-exact R
yuv = _yuv6_pair_nhwc(f0, f1)                       # YUV6 from the RENDERED frames
pose = adapter.posenet(yuv)["pose"][..., :6]       # frozen PoseNet ON THE FRAMES
pose_l = mean((pose[0] - pose_tgt)**2)             # MSE vs GT PoseNet 6-target (= realized d_pose)
pose_term = sqrt(10*pose_l + eps)                  # score-domain (S's nonlinear pose term)
return w_seg*seg_l + w_pose*pose_term
```

- `pose_tgt = gt.gt_poses[pi]` = the frozen CPU-torch PoseNet first-6 scalars on the GT pair
  (`_cpu_pose_raw` → `pose[0,:half]`, half=6) = the Quantizr stored-pose GT. Loaded as
  `pose_tgts` (trainer line 452) and PASSED to the loss every step (trainer line 582). It is the
  SUPERVISION TARGET, used.
- This is realized-d_pose-toward-GT-6-targets through R → frozen PoseNet. NOT the collapsed
  amortized-luma carrier (d_pose 2.67–12.66) — that was a different module
  (`amortized_luma_carrier.py`) that RECONSTRUCTED pose from a luma INR. The `(fix g)` comment at
  trainer line 684 ("DROP pose-from-texture (the COLLAPSED amortized carrier)") refers to the
  DEFAULT `w_pose=0` making the texture head serve SegNet realism — it is NOT a statement that the
  `w_pose>0` path is collapsed. With `w_pose>0` the SAME texture/code is trained to realize pose.
- DEPLOY: there is NO pose sidecar the scorer reads. The contest scorer runs PoseNet on the
  FRAMES, so pose is carried by the trained render (the per-(pair,frame) `code`/FiLM + texture).
  `tools/levelset_byte_close_and_eval.py` (docstring 55-64) confirms: a stored 6-scalar sidecar
  does NOT lower realized d_pose; the row REQUIRES a `--w-pose>0` pose-trained render, and the tool
  reads that checkpoint and measures realized d_pose on the inflated frames. **Row pose path is
  READY at the mechanism level — no wiring required.**

## Coordinator re-center (2026-06-27): "DERIVE POSE FROM SEG", not store, not generic-w_pose-from-scratch

Reframe: pose is DERIVED from the seg partition (FEED-da asymmetry: the partition is a SUFFICIENT
STATISTIC for the 6-DOF ego-motion). Mechanism: (1) `pose_from_embedding` MLP predicts pose from
MASK/seg features (no PoseNet at inflate); (2) FiLM-condition the per-pair render on the derived
pose; (3) `w_pose>0` is ONLY the training signal that realizes the pose in pixels, in the
SegNet-null space. Analysis below — with one hard caveat that REORDERS the priorities.

### The binding fact the whole pose game turns on (re-derived, holds for BOTH framings)
`d_pose = MSE(PoseNet(generated_frames)[:6], PoseNet(gt_frames)[:6])` — the scorer reads the
**FRAMES**, never a value/sidecar. So a derived OR stored pose VALUE only matters as **render
conditioning**; lowering realized d_pose ALWAYS requires the FRAMES to be trained (`w_pose>0`) to
hit the PoseNet target. Therefore the BINDING prerequisite for a usable row is NOT "how is the pose
value obtained" but **"does training the render to realize pose RAISE d_seg (null containment)?"** —
the smoke below. This is identical for code-carried (current) and derive-from-seg conditioning.

### Rank-1 confirmed (premise test, n96, frozen CPU authority)
`project_posenet_rank1_discovery` holds on this segment: GT pose dim-0 (forward speed) mean 33.0,
std **1.02**; dims 1-5 std **≤0.038**. Pose is ~1 effective DOF/pair → the per-pair pose
conditioning is ~1 scalar. Pose RATE is near-free in every framing:
code-carried (free — the `code` is already stored for seg) · rank-1 stored scalar (1.2KB,
the rank-1 memo's OWN recommendation) · `pose_from_embedding` MLP (~9.3KB shared, 4782 params).

### Derive-from-seg VALUE-accuracy premise test (`derive_from_seg_premise_test.py`, $0 CPU)
Fit canonical `pose_from_embedding` (MaskFeatureExtractor + MLP, embedding=zeros = inflate-faithful)
on GT (seg mask PAIR → GT PoseNet 6) for n96, holdout 16:

| metric | predict-mean baseline | derive-from-seg MLP |
|---|---|---|
| 6-DOF test MSE | 0.2058 | **0.1902** (−7.5% only) |
| dim-0 test MSE | 1.233 | 1.139 |
| dim-0 test RMSE | 1.110 | **1.067** (vs GT std 1.019) |

**Finding (HONEST, INCONCLUSIVE-leaning-WEAK):** a *static* seg→pose predictor barely beats
predict-the-mean here — dim-0 RMSE (1.067) ≈ the GT std (1.019), i.e. ~no per-pair speed recovered.
Two confounds make this a LOWER bound, not a refutation: (a) masks were downsampled to 96×128 → the
frame-to-frame lane-mark displacement that encodes speed is sub-pixel and likely washed out; (b) 96
consecutive highway pairs have low speed-variance (std 1.02) so "predict mean" is already near-best.
Note the rank-1 memo ITSELF recommends STORING the dim-0 scalar (600 FP16 = 1.2KB), not deriving
it — consistent with derive-from-seg-as-static-predictor being hard. CRITICALLY: the realized
`w_pose>0` path does NOT depend on this value-prediction accuracy — it trains the FRAMES to match
the PoseNet target directly, so realized d_pose can go far below the value-MSE. A full-res
derive-from-seg retest is the fair follow-up, but it is NOT on the critical path to the row.

### FiLM conditioning site (already present)
The level-set generator ALREADY has the per-(pair,frame) FiLM conditioning site:
`lever_b_levelset_generator`/the trainer's `LevelSetRGBWitness` carry `self.code` (per-frame latent)
→ `self.film` (Linear → per-layer scale/shift) modulating the trunk. This IS where a derived pose
would FiLM-condition. `src/tac/nerv_pose_conditioning_bolton.FiLMModulator` (pose (B,6) →
(scale,shift), identity-init) is the substrate-agnostic adapter if EXPLICIT pose-conditioning
(separate from the free code) is wanted. No new conditioning machinery is needed.

## Q2 — $0 CPU null-containment smoke (does w_pose>0 lower d_pose WITHOUT raising d_seg?)

Design: n6, render-384 (faithful; the scorer always runs at 384×512 regardless of render res, so
lower render res does NOT speed it up), `--mlx-device cpu`, frozen CPU-torch realized verdict,
matched seed/epochs, control `--w-pose 0` vs treatment `--w-pose 1` (parent RGB-witness default).
Per-epoch ~1.5–2 min (MLX-CPU backward through frozen EfficientNet-B2 + FastViT is the fixed cost,
no custom-Metal accel on CPU) → bounded to 16 ep / eval-every 4.

### Result (REALIZED CPU-torch verdict; [macOS-CPU advisory]) — matched n6, render-384, seed 0

| ep | CTRL w0 d_seg | CTRL w0 d_pose | TREAT w1 d_seg | TREAT w1 d_pose | TREAT w10 d_seg | TREAT w10 d_pose |
|---|---|---|---|---|---|---|
| 0  | 0.50748 | 189.530 | 0.50748 | 189.530 | 0.50748 | 189.530 |
| 4  | 0.46801 | 189.503 | 0.46535 | 189.503 | 0.46451 | 189.499 |
| 8  | 0.43379 | 189.529 | 0.44986 | 189.539 | — | — |
| 12 | 0.59614 | 189.607 | 0.52365 | 189.611 | — | — |

**Honest interpretation (NO-FAKE; do NOT claim a descent the smoke does not show):**
1. **d_pose does NOT descend** at w_pose ∈ {1, 10} in this budget — it stays ~189.5 (even drifts up
   with d_seg instability). Cause: (a) the score-domain `sqrt(10·d_pose)` damps the pose gradient to
   ~`w_pose·5/√1900 ≈ 0.11·w_pose` at d_pose~190 (BY DESIGN — damp-then-sharpen); (b) the render must
   LEARN to emit pose-carrying frames from a cold start, a many-hundred-epoch process. Even w_pose=10
   barely nudges d_pose (189.499 @ ep4). **A short $0 CPU smoke is the WRONG instrument for a slow
   multi-hundred-epoch realized-pose descent.**
2. **d_seg in the n6/≤16-ep from-scratch regime is HIGHLY unstable** (bounces 0.43→0.60→0.52, driven
   by softmax-temp anneal + LR schedule + EMA lag, NOT the pose term). The treatment−control d_seg
   deltas (±0.01–0.07) are WITHIN this noise and SIGN-FLIP across epochs (w1 lower at ep4, higher at
   ep8, lower at ep12). => the smoke shows **NO catastrophic d_seg blow-up from pose supervision**,
   but it CANNOT decisively prove null-containment in this budget (d_seg is nowhere near the ~0.005
   boundary regime where pose-texture perturbations would actually flip the argmax).
3. **The smoke CONFIRMS:** the `w_pose>0` path runs end-to-end (render→R→PoseNet→MSE-vs-GT-6), is
   numerically stable (0 spike-skips through ep8), and does not immediately wreck d_seg.
4. **Existence proof for the descent (NOT from this smoke):** the parent RGB witness (a7660df3) with
   the IDENTICAL score-domain w_pose=1 loss descended d_pose **12.94 → 0.0009** over its full
   (~1500-ep) run (memory base_ch=20 HNeRV basin) while d_seg also descended — that full run is the
   real null-containment + descent evidence; the level-set inherits the same loss path verbatim.

**=> The decisive null-containment + d_pose-descent measurement is a FULL pose-trained descent, NOT a
short CPU smoke.** ACTIONABLE: the running GPU descent (pid 72602) uses `--w-pose 0` (d_seg
isolation) — it is NOT pose-trained, so it cannot byte-close to a usable row. A `--w-pose 1.0` GPU
descent variant (MLX-GPU, the 22-26× custom-Metal-backward path) is the instrument that will both
descend d_pose AND reveal realized null-containment at the converged d_seg operating point.

## Optimal w_pose for the row + composition
- w_pose: parent RGB-witness default `1.0` (score-domain `sqrt(10*d_pose)` already damps the early
  large-d_pose gradient so w_seg=100 keeps driving d_seg). Smoke confirms direction; the n600 row
  uses the value that lowers d_pose without raising d_seg (see trajectory).
- Composition with lane-edge (LEVER-3) + chroma: orthogonal BY CONSTRUCTION — pose supervision acts
  on PoseNet(f0,f1) (f0 is 100% SegNet-invisible; f1's sub-margin chroma/texture is RGB-slack),
  lane-edge acts on SegNet(f1) margin at GT-lane pixels. Different scorers, null-vs-partition
  directions. Joint (w_pose + lane-edge + chroma) interaction must be MEASURED in the n600 row
  (not assumed); the smoke isolates w_pose.

## Verdict for the row pose config
- `--w-pose 1.0` (the verified realized store+supervise) — READY at the mechanism level; the
  byte-close tool consumes the pose-trained checkpoint and gives the real realized d_pose.
- The reviewed n600 d_seg-isolation config (`--w-pose 0`) must flip to `--w-pose 1.0` for a usable
  row (a w_pose=0 render is pose-blind → d_pose ~190 regardless of any sidecar).
- "DERIVE pose value from seg" (`pose_from_embedding`) is an OPTIONAL deploy-side rate refinement
  (near-free either way per rank-1); as a static predictor it is WEAK here and NOT required for the
  row. If pursued, it needs (a) a full-res derivability retest and (b) a realized integration smoke
  (derive→FiLM→render→`w_pose` train→realized d_pose) — a follow-up, additive, default-off.
