# ddm_ee1 — Einstein fresh-eyes codec examination of the capstone (2026-07-28)

**Pointer honesty first: 0.1910828242 [contest-CPU] UNMOVED. Nothing in this memo moves it.**
This memo is analysis + recommendations; every claim carries an honesty label
(MEASURED / DERIVED / CONJECTURE / OPEN-QUESTION). Charter:
`scratchpad/ee1_charter.md` (operator-convened fresh-eyes arm, 5-phase method).

Phase discipline: §A below was derived and committed BEFORE any campaign material was read
(only frozen contest objects: `upstream/evaluate.py`, `upstream/modules.py`,
`upstream/frame_utils.py`, `upstream/README.md`, `videos/0.mkv` container stats).

---

## §A — The derived theory: what IS the information-theoretically optimal codec for THIS objective?

### A.0 The frozen objective, re-derived from source (all DERIVED from evaluate.py/modules.py/frame_utils.py)

```
S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489        (evaluate.py:92, :63)
```

- 600 samples = 600 non-overlapping 2-frame pairs of the 1200-frame, 20 fps, 1164×874 HEVC video.
- `d_seg` = mean over samples of per-pixel argmax disagreement of `SegNet(R(frame_1))` vs GT,
  where R = bilinear resize 874×1164→384×512 (`SegNet.preprocess_input`, modules.py:107-109;
  **only the LAST frame of each pair is seg-scored** — modules.py:108).
- `d_pose` = mean over samples of MSE on the first 6 dims of PoseNet(pair), PoseNet reading
  YUV6(R(both frames)) — 4 lossless luma phases at 192×256 + 2 box-subsampled chroma
  (frame_utils.py:51-78). Same R.
- Rate charges ONLY `archive.zip` bytes. inflate code is free (README.md:118); 30-min budget;
  no time term in S.

### A.1 The currency table (exact arithmetic; DERIVED)

| quantity | value |
|---|---|
| score per archive byte | 6.6586e-7 |
| score per single pixel argmax flip (one px, one frame) | 8.4771e-7 |
| **exchange rate: bytes per flip** | **1.2731 bytes/flip** |
| flips purchasable by 1 KB of rate | ~804 flips |
| d_pose=1e-4 → pose term | 0.0316 |
| d_pose=4e-5 → pose term | 0.0200 |
| scorer-visible fraction of pixel space | 19.33% (384·512 / 874·1164) |
| B=100 KB → rate term / bytes-per-frame | 0.0666 / 167 B/frame |
| B=150 KB → rate term / bytes-per-frame | 0.0999 / 250 B/frame |

Three structural consequences fall out immediately:

1. **The 1.27-bytes-per-flip exchange rate is THE design constant.** Any label detail whose
   coding cost exceeds ~10.2 bits per avoided flip should be ABANDONED to distortion, not coded.
   Isolated single-pixel corrections (~5–10 bits each with a good context model) sit almost
   exactly AT break-even — so the optimal codec corrects everything *more coherent than isolated
   pixels* and deliberately abandons isolated boundary jitter. The optimum is found by a
   waterfill from the lossless side: order label details by bytes-per-flip, drop until the
   marginal detail costs 1.27 B/flip.
2. **Design in scorer-native space.** Everything both nets see passes through the SAME
   R: 384×512×3. ~80.7% of camera-resolution pixel dimensions are scorer-invisible. The decoder
   should design z at 384×512 and emit any right-inverse R⁺(z) at camera size. Bilinear
   downsampling of an anti-aliased edge gives CONTINUOUS sub-pixel boundary control through
   uint8 camera-res pixels — uint8 quantization is not a binding constraint on boundary
   placement (±0.5 uint8 noise averages to ~0.2 levels through R; saturated class margins dwarf it).
3. **frame_0 of every pair is structurally seg-free** (modules.py:108) — a 589,824-DOF free
   variable whose only constraint is PoseNet. Pose control authority is essentially unlimited;
   pose information content is trivially small (600×6 scalars).

### A.2 What the codec must actually transmit (the sufficient statistic; DERIVED)

The scorer never compares images. It compares (a) two argmax PARTITIONS (5-class label maps at
384×512) and (b) two 6-vectors per pair. Therefore the codec's true payload is:

- **L*_t** — the GT label-map sequence (600 scored frames), to tolerance priced at 1.27 B/flip;
- **p*_t** — 600×6 pose scalars, to a precision set by the √ marginal (below);
- **nothing else.** Appearance, texture fidelity, human-visual quality: worth exactly zero.

The optimal codec is a **lossy label-map/partition codec + a 6-DOF trajectory codec + a
deterministic scorer-free renderer** ("witness generator") in free inflate code. The renderer's
job: paint frames whose SegNet argmax reproduces the decoded partition and whose PoseNet output
hits the decoded pose targets.

### A.3 Rate floor estimate for the partition sequence (DERIVED, order-of-magnitude)

Per-frame partition ≈ boundary curves: horizon/undrivable boundary (~512 px), hood outline
(~700 px), lane markings (thin, ~1–2 k), vehicles (~few hundred) → ℓ ≈ 2–4 k boundary px/frame.

- Naive intra chain-coding at 2–3 bits/boundary px → ~1 KB/frame → 600 KB. Dead (rate 0.4).
- **Temporal structure is the whole game**: a forward-moving camera over a quasi-static scene
  means L*_{t+1} ≈ Warp_{ξ_t}(L*_t) where ξ_t is the SAME 6-DOF ego-motion the pose term wants.
  The innovation process (what genuinely cannot be predicted) is: boundary jitter (subpixel,
  mostly abandonable at 1.27 B/flip), new content at frame edges (small), lane-dash phase
  (periodic → nearly free given geometry), independent movers (few, smooth).
- Honest floor estimate: ~20–100 bytes/frame of true innovation → **12–60 KB** partition payload
  + 2–5 KB pose + renderer params ⇒ **S_floor ≈ 0.06–0.10**. Sub-0.15 has real headroom; the
  battle is an engineering one: an actual context-modeled boundary-residual coder averaging
  ≤150–250 B/frame, composed with a renderer whose native error is small.

### A.4 The realization problem is the binding risk — and it is a RATE cost, not a wall (DERIVED)

The decoder cannot ship or run the scorers (weights would be counted; strict rule). So the
renderer is open-loop at decode time. Realization error (SegNet's argmax on painted frames vs
the intended partition) decomposes:

- **Systematic bias** (SegNet places a painted boundary 1–2 px off, consistently): FREE to fix —
  code the *control curve*, not the intended curve. The encoder (which HAS the scorers) verifies
  and pre-distorts; the coded description simply IS the pre-distorted curve. Zero extra bytes.
- **Coherent residual error** (context-dependent region flips): correctable at few bits/px —
  cheap if rare.
- **Scattered near-tie chatter**: abandon (break-even at isolated-pixel cost).

Therefore the true feasibility question is only: does there exist a class-texture library +
painting policy whose NON-systematic realization error is ≾ 2e-4 (≈ 40 px/frame)? Class
interiors are safe by margin-saturation (choose textures adversarially optimized offline, robust
to uint8+R); risk concentrates entirely on boundary-adjacent pixels. Encoder-side verification
converts any excess into a measured byte cost via the exchange rate — realization can only make
the codec *more expensive*, never *silently worse*.

### A.5 Pose: a 6-dim inverse problem per pair with overwhelming control authority (DERIVED)

- Store p*_t? No — store whatever parameters make PoseNet EMIT ≈p*_t. Parametrize each pair's
  content by the warp ξ_t (dual-use: it also drives the partition prediction) plus a small
  learned/fixed perturbation basis (8–32 dims) acting on seg-safe pixels (frame_0 entirely;
  frame_1 interiors below margin). Encoder runs Gauss-Newton against the frozen PoseNet;
  stores quantized coefficients: 600×(8–32)×(4–8 b) ≈ 2.4–19 KB.
- Marginal-balance for precision: stop refining when d(√(10·d_pose))/dB < 6.66e-7. At
  d_pose=4e-5 the pose term is 0.02; pushing to 1e-5 buys 0.01 — worth ≤ ~15 KB by the currency
  table; typically achievable for ≪ that. Equalize at the KKT point.
- OPEN-QUESTION (empirical): PoseNet on painted/synthetic canvases is out-of-distribution — the
  per-pair Jacobian conditioning determines stored-coefficient precision. With 589k free DOF in
  frame_0 alone, generic nondegeneracy strongly suggests exact hittability; conditioning is the
  only question.

### A.6 The derived optimal architecture (Phase-A blueprint)

```
archive.zip (counted)                     inflate.py (free)
─────────────────────                     ─────────────────
[1] trajectory ξ_t  ~1–3 KB       ┐       geometry engine (fl=910, ground-plane homography)
[2] partition keyframes +         ├──→    partition reconstructor (warp + residual decode)
    boundary residuals ~60–150 KB │       class-texture library (seed-generated, adversarially
[3] pose-control coeffs ~2–8 KB   │         margin-saturated, uint8/R-robust)
[4] realization corrections       │       painter: z at 384×512 → R⁺(z) at 874×1164 uint8
    (waterfilled at 1.27 B/flip)  ┘       (frame_0 = pose-only canvas; frame_1 = partition-true)
```

Byte allocation at the optimum: ~70% partition residuals · ~20% corrections · ~5% pose ·
~5% global renderer params. Predicted landing zone: S ≈ 0.10–0.18 depending almost entirely on
achieved bytes/frame of the partition residual coder and native realization fidelity —
**both are measurable cheaply and neither requires training a neural network at all.**

### A.7 Ranked design-space (Phase-A priors, to be diffed against the campaign in §C)

1. **(c) Task-space partition codec + scorer-free renderer** (above) — the derived optimum.
2. **(d) Hybrid: cheap video base as predictor/pose-carrier + partition corrections** — elegant
   (pose in-distribution for free; base = side information for the boundary coder), but a
   ~100 KB AV1 base kills thin structures (lane markings die first at low rate) → coherent
   errors needing re-synthesis anyway; likely dominated unless the base is nearly free.
3. **(b) Neural implicit (NeRV-class) full-RGB reconstruction** — spends bytes reproducing the
   80.7% scorer-invisible + appearance content; information-theoretically dominated. Its one
   virtue: realization is automatic (real-looking frames keep both nets in-distribution).
4. **(a) Classical codec + postfilter** — leaderboard-in-README evidence caps this family at
   S≈1.9–2.0; d_seg ~1e-2 at usable rates is 100× too big. Dominated.

### A.8 Falsifiable Phase-A predictions (each becomes a §C diff probe)

- P1: An honest boundary-residual coder can reach ≤250 B/frame on this video's GT partitions.
- P2: Scorer-free painting with control-curve pre-distortion achieves native d_seg ≤ 5e-4
  before corrections (else the exchange rate forces the byte cost up; measurable at n600 by the
  encoder without any new scorer capability).
- P3: Pose is exactly hittable (d_pose ≤ 1e-5) with ≤8 KB of stored per-pair coefficients on
  painted canvases.
- P4: No architecture that reconstructs full RGB appearance can beat the partition codec at
  equal bytes (dominance argument A.2); any measured counterexample falsifies A.2's sufficiency
  claim, not the objective analysis.

*(§B–§E follow after campaign reading; §A frozen as written.)*
