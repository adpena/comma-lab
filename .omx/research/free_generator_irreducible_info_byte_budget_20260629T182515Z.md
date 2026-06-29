# FREE-GENERATOR irreducible-info byte budget — the MEASURED rate-half arithmetic (F4)

- **UTC:** 20260629T182515Z
- **Authority:** `[macOS advisory / research-signal]` — `score_claim=false`, `promotable=false`,
  `ready_for_exact_eval_dispatch=false`. **Pointer UNMOVED 0.19110.** This sizes the witness
  *rate-half* (the COUNTED video-derived payload); it is NOT a contest row.
- **Compute:** $0, CPU/numpy, 2.2 s. Pose from `gt_n600.npz['gt_poses']` (600×6, the EXACT d_pose
  target = first 6 of the PoseNet hydra head). Canonical scene from `gt_n96.npz['lstars']` (frozen
  CPU-torch SegNet argmax, 384×512, canonical comma10k order Road/Lane/Undriv/Movable/MyCar).
- **Tool:** `tools/measure_free_generator_byte_budget.py` (reuses `tac.boundary_math.contour_codec.
  partition_description_bytes`, `tac.contest_score.UNCOMPRESSED_SIZE_BYTES`; numpy entropy/LZMA).
- **Data:** `experiments/results/free_generator_byte_budget_20260629T182658Z/results.json`.
- **Tests:** DAG FEED-ja (free-generator framing, grok-confirmed) + FEED-iy (R2) + FEED-iw (F4 launch).
  This is the **F4** deliverable that **FEED-jb (F5 synthesis)** cites as its byte budget.

## The question (algorithmic-information framing)
The contest scores ONLY `archive.zip` bytes (`upstream/evaluate.py:63`); `inflate.py` is a FREE
deterministic interpreter (rule 118: generic algorithm free, video-derived learned content counted).
So the irreducible COUNTED info is the **Kolmogorov complexity of the witness RELATIVE TO OUR FREE
INTERPRETER** — `K_machine(witness)` = the smallest video-derived program/data the free generator
(homography + eikonal/SDF rasterizer + range decoder) needs to reproduce the argmax partition + the
pose targets. This is exactly the textbook "shortest program on a fixed universal machine" (see OSS
below); our `inflate.py` IS the fixed machine. FEED-ja decomposes `K_machine` as
`{canonical-scene descriptor + pose trajectory + per-class warp-type mask + Lane-survival residual +
~0.0008 movables}`. This memo **MEASURES** the pose + canonical terms and assembles the total with the
**CITED** residual existence proofs, then computes the MEASURED-rate sub-0.15 arithmetic.

## (a) MEASURED — pose-trajectory byte cost (the answer is YES, hundreds of bytes)
`gt_poses` (600×6): col0 = forward speed (mean 31.3, std 1.26, smooth; delta std 1.11) is the **sole
non-trivial byte cost**; cols 1–5 are near-static (std 0.007–0.036). Per quant step `q`: the d_pose
floor `mean((round_q(pose)-pose)²)` is what a render that HITS the stored quantized targets achieves;
`range_code_entropy_bytes` = order-0 entropy of the temporal deltas × N (the range/AR-code achievable,
constriction <0.1%-over regime; a context/AR model can match or beat it); `lzma` = concrete
general-purpose realized upper bound.

| q | d_pose floor | pose_term √(10·d_pose) | **range-code bytes** | LZMA bytes |
|---|---:|---:|---:|---:|
| 0.250 | 1.49e-3 | 0.122 | **310** | 700 |
| 0.125 | 6.77e-4 | 0.082 | **474** | 999 |
| 0.0625 | 2.17e-4 | 0.047 | **648** | 1,315 |
| 0.03125 | 6.34e-5 | **0.025** | **875** | 1,752 |
| 0.01563 | 1.83e-5 | 0.0135 | 1,172 | 2,117 |

**Reference:** raw fp16 = 7,200 B; the "solved" Quantizr-style sidecar is d_pose ≈ 3.4e-5 / pose_term
0.018 at ~1–5 KB. **Entropy-coding the temporal trajectory BEATS it: at solved-grade precision
(pose_term 0.025, d_pose 6.3e-5) the pose costs ~875 B (range-code) / 1.75 KB (LZMA); at the usable
threshold (pose_term 0.082, d_pose < 1e-3) it is ~474 B.** YES — hundreds of bytes. The bytes are
~entirely the forward-speed trajectory; everything else is near-free.

## (b) MEASURED — canonical-scene descriptor byte cost
- **ONE static canonical scene** (per-pixel temporal MODE partition), lossless: **480 B**.
- **BUT static-canonical (and pose-warp) is LOSSY at d_seg ≈ 0.021** (Road 0.018, Undriv 0.0085,
  MyCar 0.0027 captured; Lane 0.98 / Movable 0.40 NOT). The grok pose-warp lifts Road only +16% →
  ~0.0165 — still ~25× the frontier need. Per the eikonal memo, **lossy partition coding is
  score-DOMINATED** (break-even Δd_seg/byte = 4.0e-6; the curves trade 5–21× worse), so the naive
  "store ONE scene + warp" is NOT the rate-half.
- **The VIABLE canonical descriptor is the STRUCTURED per-class SDF manifold** (CITED, R-surviving):
  lane SDF d_seg 4.2e-4 (post-R 8e-4) ~1.5 KB/600; hood static 7.4e-4 (post-R 6.8e-4) **56 B/600**.
- **Full per-frame lossless store (the thing the free-generator AVOIDS): 416 KB → rate 0.277**
  (FEED-af's SOTA context-arith tightens this to 255 KB / rate 0.17). Either way, large.

## (c) DEEP-MATH — the assembled byte budget + irreducible-info frontier
`K_machine(witness)` COUNTED rows (MEASURED where possible, CITED otherwise):

| section | bytes / 600 | kind | carries |
|---|---:|---|---|
| pose trajectory (solved grade) | **875** | MEASURED (range/AR entropy) | d_pose 0.025 **+ Road d_seg FREE** (grok +16%) |
| canonical per-class SDF (lane+hood) | **1,556** | CITED FEED-dm/du (R-surviving) | lane 4.2e-4 / hood 7.4e-4 |
| per-class warp-type mask | **0** | FREE (class→regime dispatch) | Road→ground H, MyCar→identity, sky→rotation |
| movables residual | **750** | ESTIMATE (CITED grok GAP-1) | movable d_seg ~8e-4 |
| learned long-tail residual | VARIABLE | the witness's job (GPU unknown) | drives d_seg 0.018→ frontier 6e-4–1e-3 |

**`K_machine(witness) ≈ 3,180 B counted** (excl. the learned long-tail residual) — vs **416 KB**
lossless store = **131× smaller**. Rate term of that budget = **0.0021**.** The irreducible
counted info is *tiny*; the genuinely-new = the forward-speed pose trajectory (~875 B) + the
lane-survival/movables residual (~2.3 KB). Everything else (Road/sky/hood bulk = ~98% area, 3 of 5
classes) is the FREE pose-warp of the canonical and needs NO trained INR.

### The MEASURED-rate sub-0.15 arithmetic  `S = 100·d_seg + √(10·d_pose) + 25·bytes/N`
| scenario | d_seg | pose_term | bytes | seg | pose | rate | **S** |
|---|---:|---:|---:|---:|---:|---:|---:|
| optimistic frontier | 8e-4 | 0.025 | 3,180 | 0.080 | 0.025 | 0.0021 | **0.107** ✓ |
| conservative post-R | 1e-3 | 0.025 | 7,180 | 0.100 | 0.025 | 0.0048 | **0.130** ✓ |
| cheap-pose variant | 8e-4 | 0.047 | 2,953 | 0.080 | 0.047 | 0.0020 | **0.129** ✓ |

**Threshold (rate negligible, pose fixed): sub-0.15 ⟺ d_seg ≤ 1.23e-3; sub-0.19 ⟺ d_seg ≤ 1.63e-3.**
The cited frontier need (~6e-4–1e-3) sits **INSIDE the sub-0.15 window**. **The rate term is NOT the
constraint; sub-0.15 reduces ENTIRELY to the d_seg residual (the LEARNED long-tail + R-survival).**

## (d) COMPLIANCE — the clean free-generator boundary (rule 118)
- **FREE in inflate.py** (generic algorithm): homography warp / eikonal growth / SDF rasterizer;
  range/ANS decoder (constriction); per-class warp-type dispatch (class→{ground H, identity, rotation}).
- **COUNTED in archive.zip** (video-derived): the 6-DOF pose scalars, the per-class SDF manifold
  coords, the learned residual weights, the movable object streams.
- **FORBIDDEN**: smuggling a video-derived per-frame table/weights into inflate.py "code" (hide-data-
  in-code fake; rule 118 / NO-FAKE #6/#7).
- **Pose genericity:** COMPLIANT either way — store the 6 scalars as a tiny counted statistic
  (~875 B, measured) OR re-derive them GENERICALLY at decode (PoseNet/SfM-from-frames, free). Storing
  is the cheaper, deterministic, host-portable choice and is legally COUNTED video-derived data.

## (e) ONLINE + OSS grounding
- **K-complexity relative to a fixed machine** is the exact frame: the minimal description is the
  shortest program that outputs the object **on a fixed universal machine** (algorithmic information
  theory). Our `inflate.py` is that fixed machine; `K_machine(witness)` = the counted archive bytes.
  Recent work formalizes decoder/transformer description length converging to conditional Kolmogorov
  complexity (arXiv 2304.05366, 2509.22445) — neural-compression-with-a-free-decoder is the same idea.
- **Range/AR coding of the smooth pose trajectory:** `constriction` (bamler-lab; arXiv 2201.01741)
  provides Range Coding (queue-based, "preferable for autoregressive models") + ANS achieving **<0.1%
  over the entropy** — i.e. our measured order-0 entropy bytes (310–875 B) are realistically
  achievable, and an AR/context model over the smooth trajectory can go lower. This is the production
  path to replace LZMA (which is ~2× the entropy bound).

## Honest caveats (binding — NO-FAKE)
- **Rate is byte-MEASURED** (real LZMA + real order-0 entropy + real `partition_description_bytes`),
  NOT asserted. The d_pose floors are real `mean((quant-true)²)` on the EXACT PoseNet target.
- **The canonical-scene/pose-warp d_seg numbers are PRE-R, direct-partition** (the eikonal/grok
  proxies) — necessary, not sufficient. The structured-SDF post-R numbers (lane 8e-4, hood 6.8e-4)
  are the R-surviving existence proofs (FEED-iw confirms R is benign for wide-ramp SDFs).
- **The learned long-tail residual bytes are the genuine unknown** the GPU budget addresses; the
  budget above is the FLOOR excluding it. Its size + the achieved d_seg are what an exact byte-closed
  row must prove. This memo is a **means** (sizes the rate-half); the **end** is a byte-closed exact
  row < 0.19110.
- `[macOS advisory / research-signal]`; pointer **0.19110 UNMOVED**.

## Wire-in (Catalog #125)
- Hook #1 sensitivity-map: ACTIVE — the byte budget shows capacity must route to the lane-survival /
  movables long-tail (the only learned term), NOT to the descriptor.
- Hook #2 Pareto: ACTIVE — confirms the rate axis has large slack; d_seg is the sole binding axis.
- Hook #3 bit-allocator: ACTIVE — pose ~875 B (col0-dominated) + structured SDF ~1.5 KB + movables
  ~0.75 KB is the allocation; lossy label-space coding pruned (dominated).
- Hook #4 cathedral autopilot: N/A (advisory sizing).
- Hook #5 continual-learning: ACTIVE — this memo + DAG FEED-jc; grounds FEED-jb (F5) byte budget.
- Hook #6 probe-disambiguator: N/A.

## Sources
- https://bamler-lab.github.io/constriction/ · https://arxiv.org/abs/2201.01741 (ANS/range coding)
- https://arxiv.org/pdf/2304.05366 · https://arxiv.org/pdf/2509.22445 (Kolmogorov complexity ↔ deep learning / decoders)
