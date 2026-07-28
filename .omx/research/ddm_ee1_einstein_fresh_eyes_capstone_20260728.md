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

*(§A frozen as committed pre-campaign-read at 41f2affaca; §B–§E below written after.)*

---

## §B — Campaign state, with verified receipts

STORES CONSULTED: `upstream/{evaluate.py,modules.py,frame_utils.py,README.md}` + `videos/0.mkv` probe ·
MEMORY.md + memories `box_retired_…_20260728` (§1–§9) · `pose_is_a_terminal_…_20260728` ·
`pantheon_synergy_…_20260727` · `master_thesis_…_20260720` · `.omx/research/`
`council_gc5_schmidhuber_micro_macro_bridge_20260728.md` · `pr86_pr130_fullstack_intake_20260728.md` ·
`ddm_sc1_seeded_scene_carrier_20260728.md` · `ddm_ar1_archetype_codec_priced_spec_20260728.md` (price
table) · `ddm_fc1_…_20260728.md` (context/coder receipts) · `SPEC_v10_…:735` (#541 scope check) ·
`.omx/state/canonical_frontier_pointer.json` (verified live: our contest-CPU row + effective frontier
= PR130 0.172 official rank-1) · fd1 live receipts `/Volumes/VertigoDataTier/pact/ddm_fd1_20260728/
{s0_boxsolve_band_receipt.json, s2_gn_window/run_identity.json}` · grep sweeps for prior direct-lstars
coding (none found) and for #541's object (continuous plane, not tokens).

Verified load-bearing numbers (each re-read from its receipt this session):
- Pointer **0.1910828242 [contest-CPU]** UNMOVED; effective frontier = **PR130 0.172** (official
  display; bot row 0.172141 [contest-CUDA]; **no contest-CPU row exists for PR130**).
- PR130 ledger (code-verified by pi1): 191,052 B = tokens **116,980 B @ 0.00793 bpp** (≈195 B/frame)
  + int4 renderer 40,252 B + HPAC prior 20,179 B + **pose carrier 23,054 B → d_pose 2.33e-5**;
  partition leg ≈168 KB; **score is 73.9% rate**. PR86 = same lineage at 0.2736; its writeup
  independently states the witness reframe and an adversarial existence proof (d_seg 0.00 at
  unconstrained rate).
- Campaign measured walls (all n600 or receipt-cited): flip-residual **support 421,366 B LZMA**;
  sp1 contour-of-flip-support **444,394 B (worse)** → explicit-residual copy-base family
  MEASURED-CLOSED; labels **41,392 B @ H(flip|ctx)=0.325/0.345 b/flip** (solved to +0.05% of floor);
  warp-PREDICT family CLOSED (ground-ORB-H 0.9988× neutral; stratified +7.1% worse); paint-face
  base error **0.0086** (~1,690 err/frame); ws1 described base **0.024 @ 138 KB**; rp1 cells-HOLD
  (C1 min-norm preimage → uint8: d_seg 3.63e-4 = 2.39× q1 on the GT substrate; margin gap 166×);
  **fd1 stage-0 (live): cells hold on the boxsolve substrate too — c1/c0 = 1.077**, closing rp1's
  optimistic-bound caveat favorably; sc1 e_p probe: pose steering field **~2,039 B measured n600**
  (rank-1; paint base pose-dead without it) → pose leg CLOSED at ~2 KB.
- Corrected budgets: bar 0.172 ⇒ ≤ ~212 KB at solved-seg+PR130-class pose; sub-0.15 ⇒ ≤ ~179 KB.
- Live build: **fd1 = family-d GN/CG in description coordinates on the j2 engine** (706 counted
  params), stage-0 done, stage-2 GN window opening; gc5's own council capacity-flags the 706-param
  instance (Assumption-Adversary row 4 + Hotz dissent).

---

## §C — THE DIFF: derived optimum vs campaign path

Classification per charter: **(a)** our-error / scope-gap · **(b)** my-theory's-error · **(c)**
unexamined dimension (eureka candidate) · **(=)** convergence.

| # | §A element | Campaign state | Class | Finding |
|---|---|---|---|---|
| C1 | A.2 payload = partition + pose + nothing | Identical paradigm (witness doctrine; PR86 found it independently) | **=** | Triple-independent confirmation of the sufficient statistic. |
| C2 | A.1 currency 1.2731 B/flip | Endgame §3, algebraically identical | **=** | Same constant derived twice independently (here pre-campaign-read). |
| C3 | A.3 temporal innovation "20–100 B/frame via warp prediction" | THREE independent negatives: r2s warp neutral/worse · da1 XOR 1.38× worse · PR86 Table 7 temporal-diff 3.5× worse | **(b)** | **My theory's error.** Partition boundary jitter at scorer precision is temporally chaotic; temporal structure pays only as CONDITIONING CONTEXT (fc1's context includes copy argmax; PR86/130 condition on prev frame), never as prediction-then-residual. The copy-PREDICT lock is correctly scoped and NOT over-broad. |
| C4 | A.3+P1 "direct partition coding ≤250 B/frame" | **Never raced internally.** Internal attempts priced two DIFFERENT objects: (i) the continuous exact-plane (#541, ~334 KB/pair, rate-dead — not tokens); (ii) the flip-residual field vs a weak base (support 421–444 KB — dead). The GT partition (cached `lstars`) has never been fed to a real lossless/lossy contour or context coder. External anchors: **PR130 tokens 195 B/frame (learned prior)**; **ECC chain-coding: Cityscapes 2048×1024 semantic maps 2,662 B lossless = 0.0102 bpp → area-scaled ≈250 B/frame at 384×512** (§D.1). | **(c)** | **Eureka candidate #1: the direct-partition leg is unpriced internally and two external anchors put it at ~150 KB/600 frames lossless, less with the A.1 tolerance-band waterfill.** A $0 measurement closes it (§E R1). |
| C5 | A.4 realization via painting + control-curve pre-distortion | Paint face measured **159× loss** (flat/palette paint, 0.0086); crux declared = realization; live attack = fd1 GN in description coords (706 params, capacity-flagged) | **(a)+(c)** | **Scope-gap: the paint negatives were measured on ZERO-PARAMETER palettes.** The externally-measured solved form of the realization operator is a **small TRAINED partition-conditioned renderer** (PR130: 40,252 B int4 → native d_seg 2.97e-4 ≈ 58 px/frame, official rail; RF-7 says placement needs ~3 conv layers; SPADE/CLADE literature §D.3). No internal arm builds this object; sp1 §8's route (2) "upstream better-base" is READY-GATED **waiting for masks a renderer would create**. Eureka candidate #2 = build it (§E R2). |
| C6 | A.1 waterfill from lossless side | Correction-stream framing (weak base + residual-to-near-solved) dominated ALL internal pricing | **(a)** | **The correction-stream band lemma (DERIVED, new):** position cost alone bounds any correction stream below by ≈ log2(N/k)/8 B/err (uniform bound; context lowers it but fc1's measured 0.41 B/err at 0.864% density is the favorable-density case). At base error ≤~1e-3 the position floor crosses the 1.2731 water level ⇒ **correction streams are only rational in a band ≈1e-3…1e-2 base error; below it, CONCEDE dominates.** Every internal base (paint 0.0086, ws1 0.024) sat inside/above the band ⇒ correction streams looked mandatory and died of support cost; PR130's base (3e-4) sits BELOW ⇒ it ships NO correction stream. Pure arithmetic re-derives the campaign's own conclusion (base/realization quality is primary) and sharpens the spec: **the carrier must be natively ≤~1e-3, ideally ≤3e-4; sub-1e-3 correction machinery is permanently pointless.** |
| C7 | A.5 pose = cheap terminal inverse problem, 2.4–19 KB | sc1 measured ~2,039 B n600; terminal-solve staging law | **=** | Converged; my P3 confirmed by their measurement. Bonus: PR130's 23 KB pose leg is 10× our measured price — a ~21 KB budget edge. |
| C8 | A.6 blueprint byte split (~70% partition) | ar1 archetype price table (stream-1 neural column UNMEASURED; stream-2 now dead per sp1) | **=/(c)** | Same architecture class; the two UNMEASURED cells of ar1's table are exactly §E R1+R2. |
| C9 | A.7 rank: task-space > hybrid > NeRV > classical | Campaign converged (r6cal "gap=RATE"; PR130 73.9% rate) | **=** | Rate-first is the right side of the bar. Our distortion physics (E1/E2) is worth ≈21.7 KB of budget — real but secondary. |
| C10 | fd1 engine choice (charter challenge) | family-d GN family sound; 706-param instance capacity-flagged by its own council | **(c)** | **Convergence theorem (informal): family-d GN at partition-grade capacity ≡ token-grid + trained renderer in different coordinates.** Growing fd1's description space toward partition-grade DOF and training a renderer on tokens are the same object; the capacity ladder should therefore include the token+renderer parametrization as an explicit rung (§E R4), making R2 a complement to fd1, not a rival. |
| C11 | cells-hold honesty (charter challenge) | rp1 flagged its GT-substrate optimism itself; fd1 stage-0 has now measured boxsolve ratio 1.077 | **=** | Verdict was honest and the caveat is closing favorably. No challenge. |
| C12 | explicit-support death scope (charter challenge) | sp1 §8 scoped to "explicit support stream on the copy-PREDICT base" | **(a)** minor | Correctly scoped, but needs a rider so future recall does not over-read it: **direct partition coding is NOT covered by the closure** (different object, C4) and better-base concession (C6) is the third route beside implicit-carrier. |

---

## §D — Reality sweep (strongest external prior art per §C finding)

1. **Direct semantic-map coding is cheap (C4).** *"Context Adaptive Extended Chain Coding for
   Semantic Map Compression"* (arXiv:2603.03073): lossless contour coding with shared-boundary skip
   + 36-symbol extended chain code + context-adaptive arithmetic; **Cityscapes 2048×1024 semantic
   maps at 2,661.7 B avg (0.0102 bpp), beating FLIF (4,883 B) and JBIG1 (6,830 B)**; DAVIS video
   masks 367.5 B avg. Area-scaling to 384×512 ≈ **250 B/frame lossless** on HARDER (more classes)
   maps. Caveat (honest): SegNet-argmax maps are noisier than curated GT (boundary speckle/islands)
   — exactly the content the A.1 waterfill concedes at 1.27 B/flip before coding. $0 measurable.
2. **Lossy boundary coding with tolerance bands is a solved classical field (C4/C6).**
   Schuster–Katsaggelos operational-RD vertex/B-spline shape coding (ORD shape coding, 1997–2004;
   incl. *variable-width admissible control-point band* — width proportional to local trust): the
   published form of "code the curve only to the tolerance the distortion budget buys" — i.e. the
   1.2731 B/flip waterfill executed on PARTITION curves, not flip-support fragments (what sp1
   measured). MPEG-4 Part 2 binary shape coding (CAE) is the standardized ancestor.
3. **Label→image synthesis with a small consistency-only renderer is easier than the literature's
   problem (C5).** SPADE (arXiv:1903.07291) / CLADE (*Efficient Semantic Image Synthesis via
   Class-Adaptive Normalization*, arXiv:2012.04644 — comparable quality at a fraction of SPADE's
   params) solve photorealistic label→RGB; our consumer is ONE frozen SegNet (5 classes, one scene,
   874×1164), needing only argmax-consistency through R — strictly easier. Local existence proof:
   PR130's 40 KB int4 renderer at native 2.97e-4 on the official rail; PR86's RF-7 derivation
   (boundary placement decided by a 7-px window; 3 conv layers suffice) — FROZEN-SPACE candidate,
   verify on our apparatus.
4. **The whole contest objective is an instance of Video Coding for Machines (context).** MPEG VCM
   ad-hoc group (arXiv:2105.12653; surveys arXiv:2208.07313) formalizes rate–task-accuracy codecs;
   published segmentation-task BD-rate gains (up to ~98.7% vs perceptual anchors) confirm task-space
   dominance — no mechanism there is stronger than what the campaign already holds.
5. **Wyner-Ziv / dirty-paper framing (C6/A.4).** fc1's syndrome result is textbook-clean; the
   control-curve/control-token pre-distortion (solve the DESCRIPTION so the composed scorer output
   matches GT — Gelfand-Pinsker flavored) is implicitly what joint renderer+token training does; a
   post-training token re-solve (§E R7) is its cheap explicit form.

Sources: [ECC semantic map compression](https://arxiv.org/html/2603.03073) ·
[SPADE](https://arxiv.org/pdf/1903.07291) · [CLADE](https://arxiv.org/pdf/2012.04644) ·
[VCM standardization](https://arxiv.org/abs/2105.12653) ·
[task-oriented video coding survey](https://arxiv.org/pdf/2208.07313) ·
[ORD B-spline shape coding](https://www.researchgate.net/publication/274077617_Rate-Distortion_Optimal_Shape_Coding_Using_B-Spline_Snakes) ·
[variable-width tolerance band ORD](https://www.academia.edu/13395042/Variable_Width_Admissible_Control_Point_Band_for_Vertex_Based_Operational_Rate_Distortion_Optimal_Shape_Coding_Algorithms).

---

## §E — Ranked recommendations

**Top-3 (if I could only say three things):**
1. **Price the direct-partition leg NOW, at $0** — run real coders (contour/ECC-class + context-
   arith + FLIF/PNG baselines, then the tolerance-band lossy pass at the 1.2731 water level, then
   prev-frame-conditioned contexts) on the cached n600 `lstars`. Two external anchors say
   ~150 KB/600 frames lossless; nobody has the internal number; it decides whether the partition
   leg needs a learned prior at all. (unlock-current-path · DERIVED+OPEN-QUESTION)
2. **Build the trained partition→pixel renderer as the realization operator** — small (≤~64 KB
   counted), scorer-in-loop through R + uint8-STE, our-original architecture from our measured
   physics (RF-7-scale placement, E-cell margins, chroma seg-invisibility). It is the measured
   solved-form of the crux elsewhere (40 KB → 2.97e-4 official-rail) and the object sp1's
   better-base gate is waiting for. (unlock-current-path · MEASURED-external existence)
3. **Adopt the correction-stream band lemma as a spec constraint**: carriers must be natively
   ≤~1e-3 (ideally ≤3e-4); below-water correction machinery is permanently retired; fd1/renderer
   targets and acceptance gates should read this number. (substantial-change to spec discipline ·
   DERIVED)

| # | Recommendation | Scope | Label | Falsifier / FIRST measurement (costed) | Displaces / why worth it |
|---|---|---|---|---|---|
| R1 | Direct-partition coder race on cached `lstars` (lossless intra → +temporal conditioning → tolerance-band lossy at 1.2731 B/flip) | unlock-current-path | DERIVED + OPEN-QUESTION | **$0, CPU-only, no scorer slot.** Falsifier: real bytes ≥~350 KB lossless AND ≥~250 KB lossy-at-water ⇒ direct-explicit dead like flip-residual ⇒ implicit carrier confirmed sole route. Expected band from §D anchors: 90–180 KB. | Displaces nothing (fd1 keeps the slot). Closes ar1's stream-1/2 UNMEASURED cells; arbitrates classical-vs-learned for the biggest byte leg (gc5 conformance item 2 demands this race be RUN). |
| R2 | Tiny partition-conditioned renderer, scorer-in-loop (our-original; train local MPS-gradient; advisory d_seg through frozen CPU scorers; the n600 verdict + any exact row spec'd to MAIN — slot owned by fd1) | unlock-current-path | CONJECTURE with MEASURED-external existence proof | First measurement: ≤64 KB-counted renderer trained on (lstars → frames) — native d_seg through frozen scorer, n600 advisory. Falsifier: cannot reach ≤1e-3 at ≤64 KB ⇒ learned-amortized realization falsified at this scale; fd1's implicit-GN remains the only live realization family. | Converts the crux from per-instance descent into an amortized 40–64 KB object; creates the masks sp1's READY-GATED better-base contract is waiting on; complements fd1 (C10). |
| R3 | Register the correction-stream band lemma (canonical-equation candidate; acceptance-gate consumer) | substantial-change (spec discipline) | DERIVED (uniform-bound caveat stated) | $0: recompute fc1's support price at synthetic densities 3e-4/1e-3/3e-3 from cached fields — the measured curve either shows the predicted crossing near ~1e-3 or falsifies the lemma. | Retires all future sub-1e-3 correction-stream work by arithmetic instead of by repeated measurement; sharpens fd1's target spec. |
| R4 | fd1 capacity ladder: add the token-grid+renderer parametrization as an explicit capacity rung (C10 convergence) | unlock-current-path | DERIVED | The gc5 "Triple mid" capacity disambiguator, run with this rung included: if d_seg(H)/bytes at the token+renderer rung dominates the grown-706 rung, the coordinates are settled empirically. | No displacement — it is the same family in better coordinates; prevents an instance-level plateau being misread as family failure. |
| R5 | Scope riders (documentation): (i) sp1 §8 closure does not cover direct partition coding; (ii) better-base concession is the third route; (iii) copy-PREDICT lock excludes conditioning-context (already exploited by fc1) | unlock-current-path | MEASURED (scope facts) | n/a — recall-integrity fix | Prevents the next arm from reading "explicit is dead" over-broadly (the exact failure mode this arm was convened to catch). |
| R6 | Composed measured-legs arithmetic → the landing band | unlock-current-path | DERIVED | Partition ≤130 KB (R1) + renderer ≤64 KB (R2) + pose 2 KB (measured) at native ~3e-4 ⇒ S ≈ 100·3e-4 + 0.0153 + 25·(196 KB)/37.5 MB ≈ **0.176**; at 150 KB composed ≈ **0.145–0.16**. Gate: only a byte-closed `upstream/evaluate.py` row counts; name it as the exact-row target the Modal budget buys. | The honest statement of where this path lands: sub-bar with measured legs, sub-0.15 only at the optimistic corner — same shape as gc5's Q1 but with both unmeasured cells replaced by $0/cheap measurements. |
| R7 | Control-token re-solve after R2 (dirty-paper: re-optimize the coded partition through the frozen renderer+SegNet so composed output matches GT; tokens become control variables) | unlock-current-path | CONJECTURE | After R2 only: n600 advisory re-solve; falsifier: no d_seg improvement at equal bytes. | Zero-byte realization repair (A.4's pre-distortion made concrete); PR130 gets this implicitly via joint training. |
| R8 | Lane-dash symbol dictionary (JBIG2/DjVu-flavor) as one entrant inside R1's race | unlock-current-path | CONJECTURE | Included in R1 at $0; falsifier: no byte win over context-arith on the Lane class. | Lane markings are 19% of flip mass and periodic; symbol-dictionary coding is the classical exploit of periodicity. |
| R9 | Paradigm-alternative slot (charter completeness): none recommended. Borrowed-incumbent polish is operator-DEAD; full-RGB neural is dominated (A.2/A.7 + 73.9%-rate fact); the live paradigm (task-lossy ego-scene codec) is the derived optimum. | — | MEASURED+DERIVED | — | Honest negative: fresh eyes found no superior paradigm, only unpriced legs inside the current one. |

### Honest boundary
- Nothing here moves the pointer (0.1910828242 [contest-CPU] UNMOVED). All §E items are MEANS; the
  END is a byte-closed exact row.
- §A's temporal-innovation floor was WRONG (C3) — corrected by three campaign receipts; my §A floor
  estimate 0.06–0.10 was optimistic accordingly; the campaign's measured S_floor 0.11797 and my
  post-correction ≈0.13 landing arithmetic are the honest numbers.
- The band lemma (C6/R3) uses a uniform position bound; measured context coding shifts the band
  edge — the $0 curve measurement in R3 is the falsifier, not the derivation.
- R2's existence proof is external ([contest-CUDA], another team's vehicle); per intake discipline
  it enters only through the derive-or-race path — R2 IS that race, ours-original, no adopted
  bytes/weights/constants.
- No n600 scorer jobs were run by this arm (slot owned by fd1); every number cited is from receipts
  or frozen-object arithmetic.


---

## MAIN annotation (post-landing, magnitude-dismissal compliance — appended, body unmutated)

The §A waterfill passage ("deliberately abandons isolated boundary jitter … drop until the
marginal detail costs 1.27") is a PRICED concession under the registered exchange-rate law, not
an absolute-magnitude dismissal. Relative-significance grounding, both numbers stated:

- **The criterion IS per-detail ΔS arithmetic:** conceding k errors while saving B bytes has
  ΔS = k·(100/117,964,800) − B·(25/37,545,489); the waterfill drops a detail ONLY when this is
  ≤ 0 (coded cost > 1.2731 B/err = the S-breakeven, registered law, endgame memory §3 /
  ddm_r6cal). Every concession is ΔS-favorable BY CONSTRUCTION, never eyeballed.
- **Measured instance (sp1 concession table, k≥2 point):** conceded 4.5% of flips (+0.0389 seg
  term) for −59,964 B (−0.0399 rate term) → **net ΔS = −0.00058** at the support-stream level.
  Relative to the remaining competitive gap (S_current 0.19108 − bar 0.172 = 0.0191):
  **|ΔS|/gap ≈ 3.0%**, sign favorable — a paid trade, not an orphaned small-ΔS.
- verdict_scope: the "abandon jitter" clause is the waterfill's marginal rule (INSTANCE-level,
  re-evaluated per detail at the operating point), not a family kill of boundary-jitter coding;
  any future base shifts the water level and the same rule re-prices everything.
