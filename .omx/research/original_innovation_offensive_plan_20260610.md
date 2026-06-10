# ORIGINAL-INNOVATION OFFENSIVE PLAN — the class-shift levers (2026-06-10)

**Subagent:** `original_innovation_research_proposal_20260610` (READ-ONLY research + proposal; this memo
is the only artifact). **Evidence grade:** `[macOS-CPU advisory]` / mechanism-only. NO score claims, NO
dispatch, `promotable=false`. Frontier read from pointer, never hardcoded.

**Operator mandate (`GOAL_standing_v3` + `innovation_mandate_and_original_directions_20260610.md`):** the
recoded-R3 hold (0.19109982, −2.6e-5 over PR#112, built FROM PR#112's codec) is a DEFENSIVE BANK, not a
submission. A SUBMISSION must be GENUINELY ORIGINAL/INNOVATIVE (a class shift) **and** meaningfully below
frontier — never a noise-margin absorb-recode. This memo ranks the offensive levers, grounds them against
the EXACT frozen evaluator + our exact assets, adds directions not yet conceived, and writes the $0
descent-smoke spec for the #1 lever.

**Frontier at write (pointer `tools/refresh_canonical_frontier.py`, refreshed 2026-06-10T09:24Z):**
contest-CPU **0.19109982** (sha `b4689726…`, the recoded-R3 defensive hold).
Score law (frozen authority, `evaluate.py:92`): **S = 100·d_seg + √(10·d_pose) + 25·B/N**, N=37,545,489.
Byte price ≈ **6.66e-7 score/byte**. T_1=sub-0.19, T_2=sub-0.17, **T_3=sub-0.15** (default aim).

---

## 0. THE EXACT-EVALUATOR REFRAME (the source of all originality — read this first)

I read `upstream/{evaluate.py, modules.py, frame_utils.py}` in full. The score's structure — not its
value — is where the class shift lives. The five load-bearing facts the whole leaderboard ignores:

1. **SegNet scores ONLY frame1's argmax, at 512×384, 5-class** (`modules.py:108` `x = x[:, -1, ...]`;
   `:109` resize to `(512,384)`; `:112` `diff = (out1.argmax(1) != out2.argmax(1)).float()`).
   d_seg = **per-pixel argmax-flip RATE** over 196,608 pixels of one frame per pair. **Frame0 is
   SegNet-invisible entirely.** The seg target is a 5-class LABEL MAP, not pixels.
2. **PoseNet scores 6-of-12 ego-motion dims, MSE, on the 2-frame YUV6 input** (`modules.py:84`
   `out[...,:h.out//2]`, h.out=12 → first 6; `:70-74` YUV6 of both frames at 512×384). The pose target
   is a **6-DOF ego-motion vector per pair** — a smooth, low-dimensional driving trajectory. B2 atlas:
   the signal is INTER-frame, Y-luma 0.964, low-band (w_equiv≈294).
3. **d_pose is a POOLED MEAN over 600 pairs BEFORE the sqrt** (`evaluate.py:90-92`). Pose is therefore
   a GLOBAL fungible budget (1:1 across pairs), and the sqrt makes the pose term ALREADY tiny
   (√(10·2.9e-5)≈0.017 of 0.192). **Pose is nearly free at the frontier operating point.**
4. **Rate counts `archive.zip` ONLY** (`evaluate.py:63`); `inflate.py`/`inflate.sh` are free real estate.
   Denominator is the contest's GT videos, fixed at 37,545,489 (`:64` rglobs `uncompressed_dir`).
5. **The public test set is ONE contiguous drive** (`0.mkv`, 1200 frames → 600 non-overlapping pairs,
   `frame_utils.py:138`). Massive INTER-pair and INTER-frame temporal redundancy.

**The reframe, made precise.** Every leaderboard entry (HNeRV/HiNeRV/ours) compresses the *RGB video* and
lets the scorer derive its two measurements. That is **pixel-native**, and it is a strictly HARDER problem
than the score asks: you pay to reconstruct 1200 full RGB frames at 1164×874 when the score reads (a) a
5-class label map on 600 frames at 512×384 and (b) a 600×6 ego-motion trajectory. The field clusters at
0.19-0.20 because it is solving the wrong (harder) problem. **The original move is to be SCORE-NATIVE:
store/synthesize only what the evaluator measures, in the representation the evaluator measures it in.**

**The budget that makes this concrete (derived, this session):**
- T_3=0.15 at frontier distortion ⇒ total byte budget **115,596 B** (vs frontier 178,495 B): must cut
  ~35% of bytes while holding distortion, OR change the decomposition so the bytes go somewhere cheaper.
- On a pixel-native carrier the ONLY place that headroom can come from is the decoder blob (90.9% of
  bytes, 99.98% of |grad|), and it is FALSIFIED-to-coarsen ×2 (no redundant precision). **The pixel-native
  axis is exhausted.** The headroom exists only OFF the pixel-native manifold.

---

## 1. LITERATURE PASS (adversarial — claims stay external until grounded against OUR evaluator)

I searched the genuinely-relevant frontier and read it against `evaluate.py`. The recurring adversarial
finding: **the published SOTA optimizes PSNR/MS-SSIM or multi-task transfer, NOT a frozen single-task
argmax+pose+bytes objective on a single overfit-authorized video.** That gap IS our originality surface.

### 1a. INR / NeRV-family video compression (the carrier frontier)
- **HiNeRV** (Kwan et al., NeurIPS 2023, v-Jan-2024, arXiv 2306.09818): hierarchical positional encodings +
  depthwise-conv/MLP/interp; **72.3% bitrate saving over HNeRV, 43.4% over DCVC on UVG — measured in
  PSNR.** First INR to beat HEVC-HM RA in MS-SSIM. *Adversarial read:* PSNR/MS-SSIM is NOT our metric.
  A 72% PSNR win can be a 0% d_seg win — argmax-flip and 6-DOF-pose are insensitive to most of the
  pixel energy HiNeRV spends bits on (the invisibility basis proves 80.67% of pixel-DOF is scorer-null).
  HiNeRV is a better PIXEL carrier; the contest rewards a better LABEL+POSE carrier. **Mechanism worth
  stealing:** the hierarchical encoding gives a deep/wide net at low param count — relevant to a SMALL
  score-native generator (direction C), NOT as a drop-in. **Class shift for THIS score? NO** (still
  pixel-native). **Cost if adopted:** a full training campaign; our `nerv_witness_readiness_dag` already
  gates it; our three NeRV vehicles share the skip-free mean-field bug (`reference_carrier_comparison`).
- **Quantized-INR / mixed-AR low-complexity INR codecs** (ICLR-2025 "On Quantizing Neural Representation";
  arXiv 2401.12587): the QAT-in-loop + autoregressive entropy model that HiNeRV explicitly did NOT do
  in-loop. *Adversarial read:* this is the entropy-coding side our PR101/PR112 lineage already
  operationalized (L20-L32). It is rate-axis polish on a pixel carrier — exhausted for us.

### 1b. Video/Image Coding for Machines (VCM) — the subfield that frames OUR EXACT problem
- **VCM / coding-for-machines** (Duan-Liu survey; Awesome_VCM list; 2024-2025 active): compress so a
  FROZEN downstream model performs, not so humans see. **This is literally the contest.** Yet no VCM
  paper targets a frozen *single* segmentation+pose head on a *single overfit* video with an *archive-byte*
  objective — they target generalization across detection/segmentation/tracking.
- **PAT-VCM** (arXiv 2604.13294, "Plug-and-play Auxiliary Tokens"): a shared baseline stream + lightweight
  task-aware auxiliary tokens that let a downstream task recover what it needs without a per-task codec.
  *Adversarial read:* the auxiliary-token idea is direction B in disguise — a tiny task-specific carrier
  on top of a cheap base. We can specialize it HARD (one task, one video) where the paper generalizes.
- **CDRE / "Embedding Compression Distortion in VCM"** (arXiv 2503.21469): put the downstream-task distortion
  INTO the coding objective. *Adversarial read:* this validates the score-aware-loss path (our AFSR-1) and,
  more importantly, validates **the seg target as a CLASSIFICATION objective, not a reconstruction one** —
  the conceptual core of direction B/C.
- **"Tell Codec What Worth Compressing" (semantically disentangled ICM, arXiv 2408.08575):** spend bits only
  on semantically-relevant regions. *Adversarial read:* our flip-map already IS this map (66,039 frontier
  flips, 91% at margin<0.5, concentrated on road/horizon band rows 171-292). The literature builds the map;
  we already MEASURED it on the exact frozen detector. **This is our unfair advantage made explicit.**

### 1c. Steganography / STC lineage (the contest's OWN theory — Yousfi = Fridrich's student)
- **Syndrome-Trellis Codes** (Filler-Judas-Fridrich, Binghamton DDE Lab — the same lab that designed this
  contest): a near-optimal convolutional code that **minimizes total embedding DISTORTION subject to a
  payload, via Viterbi over the syndrome trellis** (`Hz=m` as a min-cost trellis path). Approaches the
  rate-distortion bound of additive distortion costs. **This is the dual of our problem.** The contest is
  "minimize ARCHIVE BYTES subject to staying inside the argmax cell." STC is "minimize DISTORTION subject
  to a payload." Invert the Lagrangian and STC's coding-theoretic efficiency directly attacks the
  per-flip floor that blocked our naive seg-repair sidecar. **Class shift for THIS score? YES** — it is
  the principled coder for the seg-repair atom problem, and it is the contest's own mathematical lineage.
- **Steganographic polar codes** (arXiv 2306.15246, "errorless robust JPEG stego"): a newer near-optimal
  alternative to STC with better efficiency at high payload. *Adversarial read:* a drop-in upgrade to the
  STC coder if STC's efficiency stalls — keep in the quiver.

### 1d. INR segmentation-map compression (the direct precedent for direction B — and it is rare)
- **INR for image compression + segmentation-preserving regularizers** (ECCV-2022 2112.04267 lineage; the
  medical-imaging INR-segmentation work): MLPs compress the image while a **structural regularizer
  maximizes Dice between original and reconstructed segmentation maps.** *Adversarial read:* this is the
  CLOSEST published precedent to direction B — but it is (a) generalization-oriented, (b) Dice not
  argmax-flip-rate, (c) image not single-video, (d) no archive-byte objective, (e) reconstructs the image
  then segments, rather than synthesizing the LABEL MAP directly. **Nobody has built a carrier that stores
  the label map + ego-motion as the PRIMARY representation for a frozen single-task evaluator on an
  overfit video.** That void is the cleanest original claim available.

### 1e. Program-synthesis / procedural / demoscene compression (for direction A + "archive as program")
- The `.kkrieger` / demoscene "ship the generator + a seed, not the data" pattern (recognized in
  `grand_council_symposium_inflate_py_extreme_compression_20260518.md:64`) + classical superoptimization.
  *Adversarial read:* `evaluate.py:63` makes `inflate.py` rate-free, so any PROCEDURAL section is free
  real estate. This is the rate subsidy for a witness-PROGRAM carrier (direction A) — bounded by
  compliance-defensibility (you cannot bake video-derived content in inflate.py and call it compression),
  but unbounded for genuinely procedural structure (PE tables, generators, the YUV6 basis).

**Sources:** [HiNeRV NeurIPS 2023](https://arxiv.org/abs/2306.09818) ·
[VCM survey/list](https://github.com/lingyzhu0101/Awesome_VCM) ·
[Embedding Compression Distortion in VCM](https://arxiv.org/html/2503.21469v1) ·
[PAT-VCM](https://arxiv.org/html/2604.13294) ·
[Semantically Disentangled ICM](https://arxiv.org/pdf/2408.08575) ·
[STC (Filler-Fridrich)](http://dde.binghamton.edu/filler/pdf/fill10spie-syndrome-trellis-codes.pdf) ·
[Steganographic polar codes](https://arxiv.org/pdf/2306.15246) ·
[INR image compression](https://arxiv.org/abs/2112.04267) ·
[Mixed-AR low-complexity INR codec](https://arxiv.org/pdf/2401.12587).

---

## 2. GROUNDING THE SEED MENU A-F AGAINST OUR EVALUATOR + ASSETS

Each direction: the exact score-term it attacks, predicted ΔS band WITH derivation, the class-shift
argument, feasibility, NAMED reuse targets (from `evaluator_inverse_orphan_inventory_20260609.md` — 103
surfaces; reuse, do not rebuild), the falsifiable $0 first test.

### A. Evaluator-equivalence quotient compiler (V6 realized) — HIGHEST innovation, HIGHEST risk
- **Attacks:** ALL THREE terms by paying zero for everything outside the evaluator's equivalence cell.
- **Class-shift argument (unquestionable):** it is a NEW PROBLEM FORMULATION — task-conditioned MDL under
  a frozen oracle (a specializing superoptimizer). No leaderboard entry has this frame. The scaffold
  EXISTS: `src/tac/optimization/frozen_evaluator_contract.py` (the pluggable `FrozenEvaluator` Protocol +
  `FrozenEvaluatorContract` dataclass) + `nerv_witness_readiness_dag.py`. The pieces exist: invisibility
  basis (80.67% null), the cone/flip-map, `null_space_exploiter`, `evaluator_action_waterfill` (the
  `S(P+σ)<S(P)` admission law), `action_effect` IR.
- **Predicted ΔS band:** theoretically large (−0.04 to −0.16) but SPECULATIVE — A is the *unifying frame*,
  not a single buildable lane. Its concrete realizations ARE directions B (decomposition carrier) and D
  (STC coder) and the floor proof F. **Honest verdict: A is the THESIS; ship B/D/C as its instances.**
- **Feasibility:** weeks as a standalone vehicle; the V3 waterfiller already IS the compiler core. Reuse,
  don't rebuild. **$0 first test:** wire `frozen_evaluator_contract` to a 1-pair toy cell + confirm the
  waterfiller admits a known-good witness — a contract conformance smoke, not a frontier move.

### B. Score-native decomposition carrier — HIGH innovation, THE CLEANEST CLASS SHIFT ⭐⭐
- **Attacks:** all three at their source. Sections = the measured quantities:
  (1) **ego-motion trajectory** — 600×6 pose floats. RAW fp16 = 7,200 B; smooth-driving entropy-coded est
      **1-3 KB** (a low-dim manifold). This is the LITERAL PoseNet target, stored directly.
  (2) **semantic layout** — the frame1 5-class argmax structure, the LITERAL SegNet-scored quantity.
  (3) **minimal appearance** — only enough luma to keep PoseNet's YUV6 tube + keep the argmax off the
      flip boundary (the margin/cone budget).
- **Class-shift argument (unquestionable):** the carrier's *sections are the score terms*, not pixels. The
  field stores a 1200-frame RGB renderer; B stores a label-map generator + a trajectory + a margin sidecar.
  Fundamentally different decomposition. **The crux + the reason it is NOT dominated:** raw seg-argmax over
  600 frame1s is 34 MB (impossible) and even contour-coded is ~424 KB (the roadmap's RANK-5 "loses"
  number). BUT — and this is the original insight the roadmap missed — **the 600 seg maps are one drive's
  near-duplicate layouts, and the generator only needs argmax-MATCH, not pixel-match.** A tiny
  conditional generator g(pose_t, t) → 5-class logits, trained with a **per-pixel cross-entropy/flip loss
  (a CLASSIFICATION objective)**, amortizes the 600 maps into a small weight blob. This is strictly EASIER
  and CHEAPER than HNeRV's RGB-reconstruction objective: argmax is invariant to logit scale and to all the
  pixel energy the scorer cannot see (80.67% null). **You are training a classifier to hit a frozen
  classifier's argmax, not a renderer to match pixels.** That is the class shift, and it is exactly the
  CDRE/INR-segmentation literature's "classification objective" applied to the contest for the first time.
- **Predicted ΔS band (derived):** pose section ~1-3 KB (vs its share of the 162 KB decoder) + seg
  generator (the bet): if a small label-map INR holds d_seg at the frontier's 5.6e-4 within a ~80-120 KB
  weight blob, total bytes drop to ~100-130 KB ⇒ rate 0.119→0.067-0.087 ⇒ **ΔS ≈ −0.03 to −0.05 →
  S ≈ 0.14-0.16**. The gain is rate, paid by storing the cheap measured quantities instead of the
  expensive pixels. Derivation gate: the seg generator's bytes < (decoder bytes − pose savings) at equal
  d_seg. The pose direct-store is near-certain (1-3 KB ≪ its decoder share); the seg generator is the bet.
- **Feasibility:** real engineering, all aiming surfaces in hand. The label-map INR is a small MLP/conv
  (cheaper than HNeRV — classification, no RGB head). Inflate runs the generator → argmax → places the
  5-class map; appearance carrier fills the pose-luma tube. Drift risk: MEDIUM (the argmax objective is
  more robust to quantization than RGB — argmax tolerates logit perturbation, the invisibility-basis
  result). **NAMED reuse:** `coin_pp_implicit_neural_representation/` (INR renderer + numpy reference +
  archive + inflate — the cheapest INR vehicle we have), `siren/` (activation family + score_aware_loss),
  `segnet_boundary_marginals.py` (the per-pixel flip-boundary target), the flip map
  (`/Volumes/VertigoDataTier/pact/frontier_seg_repair_pool_*/flip_map_full/`), `ego_motion_concentration`
  + `foveation_ego_motion` (pose-trajectory priors), `differentiable_eval_roundtrip` (mandatory),
  `lf_payload_rate_distortion` (THE LAW admission), `pr101_split_brotli_codec` (archive grammar).
- **$0 first test:** see §4 (this is the #1 lever).

### C. Fresh-init score-aware NAS/training, null-space-primary — HIGH (this is the roadmap's RANK 1)
- **Attacks:** all three; the "smaller HNeRV trained score-aware" rate bet, aimed by our surfaces.
- **Class-shift argument:** the METHODOLOGY is novel (geometry-aimed, null-space-PRIMARY objective: error
  lives in certified-invisible DOF by construction, T5 of the untapped inventory). The arch family
  (HNeRV) is NOT novel. **Adversarial verdict: C is incrementally innovative, NOT unquestionably original**
  — it is still a pixel-native RGB renderer, just trained smarter. It risks the operator's exact failure
  mode ("a smaller variant of a competitor's method"). C is the SAFE frontier-protecting bet, NOT the
  offensive submission. It is already RANK 1 in the incumbent roadmap with a descent-smoke gate.
- **Predicted ΔS band:** −0.02 to −0.05 (per the roadmap), gated on the smaller arch not collapsing.
- **Relationship to B:** B's seg-generator IS a fresh-init score-aware training of a SMALLER, CLASSIFICATION
  net — B is C done score-NATIVELY. **Build B; it subsumes C's methodology with a genuinely original
  carrier.** If B's seg-generator collapses, C (the RGB fallback) is the safety net.

### D. Inverse-steganalysis-native coding — HIGH, uses the contest's OWN theory ⭐
- **Attacks:** d_seg (the argmax-cell boundary) via principled coding, beating the naive sidecar floor.
- **Class-shift argument (unquestionable):** the contest IS inverse steganalysis. STC minimizes embedding
  distortion subject to a payload; INVERT it to minimize archive bytes subject to staying inside the
  argmax cell, with **per-pixel cost = the margin/cone budget** (UNIWARD-style, but with the EXACT measured
  margin instead of a heuristic texture cost). The naive seg-repair sidecar hit a **1.525 B/flip
  position-only floor > 1.27 B/flip break-even** (`frontier_seg_repair_pool_verdict`). That floor is a
  *position-entropy* floor — and STC/polar codes are precisely the coding-theoretic tool that beats naive
  position coding by exploiting the spatial cost structure. The flip map shows the flips are NOT random
  (91% margin<0.5, banded on road/horizon) — structured cost = exactly where STC's Viterbi wins.
- **Predicted ΔS band:** the seg-repair pool is 0.056 (29% of score). The naive sidecar could not clear
  it. STC's efficiency gain over naive position coding is typically 1.5-3× at structured cost. If STC
  drops the per-flip cost below 1.27 B/flip on the structured flip set, a fraction of the 0.056 pool
  becomes reachable: **ΔS ≈ −0.01 to −0.03** (clearing 20-50% of the pool at coding-theoretic efficiency).
  Derivation gate: STC self-syndrome length on the margin-cost flip set < 1.27 B/flip break-even.
- **Adversarial caveat:** our own `lane_stc_clean_source` $0 probe (2026-05-30,
  `feedback_stc_clean_source_mask_delta_disambiguator_probe_landed_20260530.md`) found UNIFORM-cost STC
  self-syndrome 2.4-2.6× LARGER than brotli on mask-deltas. **BUT that probe used UNIFORM cost** — the
  whole point of STC is the NON-uniform margin-informed cost map, which that probe explicitly named as the
  reactivation criterion (CC#2 detector-informed cost map). D is the reactivation of that DEFER with the
  exact piece it was missing. **NAMED reuse:** `codec/syndrome_trellis_codec.py` (real Filler-Fridrich STC,
  73 tests), `uniward_delta.py` + `substrates/uniward_per_pixel_distortion/` (the margin→cost map),
  `stc_boundary_codec.py`, `segnet_boundary_marginals.py` (the margin = the cost), the flip map.
- **$0 first test:** build the margin-informed per-pixel cost map (= `exp(−margin/τ)` from the flip map),
  feed it to `ternary_stc_encode_stream` on the 66,039-flip set, measure B/flip vs the 1.27 break-even.
  KILL if margin-cost STC ≥ 1.27 B/flip (the structured cost does not help the coder). This is the #2 lever.

### E. Generative/implicit micro-prior carrier — HIGH innovation, HIGHER risk
- **Attacks:** all three via a tiny conditional generator producing evaluator-valid frames from minimal
  latents, trained to land in the cell not match pixels.
- **Class-shift argument:** genuinely novel carrier (diffusion-distilled or coordinate-INR micro-prior).
  But E is essentially B's seg-generator GENERALIZED to also produce the pose-luma appearance — i.e. **E ⊇
  B**. Building B first de-risks E (B is the seg half; E adds the appearance generator). Standalone E has
  inflate-runtime + drift risk (a diffusion model in a 30-min CPU inflate budget is dangerous).
- **Predicted ΔS band:** −0.03 to −0.06 IF the generator amortizes, but with the highest feasibility risk.
  **Verdict: E is B's natural extension; sequence B → E, do not start E cold.**

### F. Information-theoretic floor derivation — original ANALYSIS, sets T_floor
- **Attacks:** none directly — it sets the LOWER scoreboard and proves how much headroom B/C/D/E have.
- **Class-shift argument:** original publishable analysis (the minimum bits to specify a member of the
  evaluator's equivalence class of THIS video). It makes the threshold ladder principled.
- **Predicted ΔS:** indirect (aims the others). **$0 cost:** measure the entropy of (a) the 600×6 pose
  trajectory, (b) the 600 seg-argmax maps under the temporal-redundancy model, (c) the certified-null
  fraction. This is the T4 council's open lever (pose-output-entropy probe, RANK 6) GENERALIZED to all
  three terms. **NAMED reuse:** `null_space_exploiter`, `scorer_spectral_sensitivity_v2`, the B2 atlas
  JtJ spectrum, `ego_motion_concentration`. **Run F's pose+seg entropy measurement INSIDE the B smoke** —
  it is the same decode-the-frontier-latents/maps work, so F is a free byproduct of the #1 lever's smoke.

---

## 3. DIRECTIONS WE HAD NOT CONCEIVED (≥3 genuinely novel, same grounding)

### G. The ARGMAX-INVARIANCE BUDGET — train/code in the seg LOGIT-NULL, not the pixel-null ⭐ NOVEL
- **The pattern-of-patterns:** we have an invisibility basis for the *pixel→input resize* null (80.67%).
  We do NOT have the invisibility basis for the *seg LOGITS → argmax* null. d_seg = argmax-flip, so the
  scorer is invariant to ANY logit perturbation that does not cross a decision boundary — a budget
  defined by the per-pixel margin (top1−top2 logit gap). The flip map measured the BOUNDARY (where margin
  is small); the UNCONCEIVED dual is the **interior budget** (where margin is large = huge free logit
  room). A score-native seg carrier (direction B) trained with an **argmax-hinge loss** (penalize only
  when the predicted argmax would flip, zero penalty inside the margin) is fundamentally cheaper to encode
  than an L2/CE generator, because it is free to put all its representational error into the large-margin
  interior — which is ~91% of pixels (the inverse of the 91%-at-margin<0.5 flip concentration).
- **Attacks:** d_seg + rate (a hinge-trained generator needs fewer bits — it only must be right at the
  boundary). **Class shift:** the OBJECTIVE is the evaluator's own argmax-invariance, not a surrogate loss.
- **Predicted ΔS:** force-multiplier on B (−0.01 to −0.03 ON TOP of B's rate win, by shrinking the seg
  generator). **$0 first test:** compute the per-pixel margin distribution on the 600 frontier frame1s;
  measure the fraction of pixels with margin > a coding-tolerance band (the certified-free logit room).
  KILL if margins are uniformly small (no interior budget). **NAMED reuse:** `segnet_boundary_marginals`,
  `tropical_argmax_boundary_grammar.py` (the argmax decision-boundary grammar — an ORPHAN contest_exploit
  ripe for revival), `frame1_joint_safe_cone`. This is **the seg-axis analog of the resize-null basis** and
  it is the single most surprising gap (see §6).

### H. POSE-AS-A-2KB-SIDECAR + SEG-ONLY RENDERER (decompose the carrier by score-term) ⭐ NOVEL
- **The pattern:** the field stores ONE carrier that jointly produces both frames (and the scorer derives
  both terms). But the two terms are ORTHOGONAL in what they read: SegNet reads frame1-argmax only;
  PoseNet reads the inter-frame YUV6 luma. **Split the carrier:** (1) a ~2 KB pose-trajectory sidecar
  (direction B section 1) that drives an analytic warp(frame1→frame0) so PoseNet's pose is satisfied by a
  geometric warp at NEAR-ZERO stored frame0 bytes (T2 of the untapped inventory: frame0 is seg-free, so
  frame0 = warp(frame1, pose) needs only the pose to reconstruct); (2) a seg-only renderer for frame1.
  This **eliminates the frame0 representation entirely** — frame0 becomes a deterministic function of
  frame1 + the 6-DOF pose, stored in inflate.py CODE (rate-free per `evaluate.py:63`).
- **Attacks:** rate (delete ~half the per-pair frame content). **Class shift:** the carrier is decomposed
  BY SCORE-TERM ORTHOGONALITY, a decomposition no entry has. **Predicted ΔS:** if frame0 is regenerated by
  warp at the frontier's d_pose tolerance, the decoder need only represent frame1 (the seg-scored frame) +
  the pose trajectory ⇒ decoder capacity roughly halves on the frame0 axis ⇒ **ΔS ≈ −0.02 to −0.04**.
  Derivation gate: d_pose of `affine_warp(frame1, est_pose)` frame0 ≤ frontier 2.9e-5 + (rate saved)/√term.
  **$0 first test (T2's test):** for 600 frontier pairs, replace frame0 with `affine_warp(frame1, pose)`
  (zero stored frame0 bytes), measure d_pose. KILL if warp-only d_pose ≫ frontier (frame0 needs genuine
  independent content). **NAMED reuse:** `renderer.py:1096` (warp primitive),
  `nscs06_v8_chroma_lut.inflate._affine_warp_frame1_from_frame0`, `xray/foveation_ego_motion`,
  `scorer_read_surface_atoms.pose_null_projection`. **H is the structural sister of B** (B stores the
  measured quantities; H exploits that frame0's only measured quantity is pose-derivable from frame1).

### I. THE QUOTIENT-DICTIONARY CARRIER — one codebook over the 600 near-duplicate cells ⭐ NOVEL
- **The pattern-of-patterns:** the single-video structure means the 600 (seg-map, pose) cells live on a
  low-dimensional manifold (the car repeats: stopped, straight, turning). The field codes each pair
  independently. A QUOTIENT carrier stores K representative evaluator-cells (a learned dictionary of
  layout+pose archetypes) + 600 per-pair indices + sparse residuals. This is direction-A's quotient
  realized at the CELL level (not the latent level — T1 of the untapped inventory does latent dedup; I
  does EVALUATOR-CELL dedup, which is coarser and cheaper because two pairs in the SAME argmax cell need
  the SAME index, period). **The score is a quotient map; store the quotient, not the representatives.**
- **Attacks:** rate (massively, if the drive is self-similar). **Class shift:** stores the equivalence-cell
  index, the literal output of the evaluator's quotient map — the purest score-native carrier. **Predicted
  ΔS:** if 600 pairs collapse to K≈32-128 distinct cells, index cost = 600×log2(K)/8 ≈ 450-525 B + K cell
  archetypes. Could be the single largest rate win (−0.03 to −0.06) IF self-similarity is high. **$0 first
  test:** cluster the 600 frontier (seg-argmax-downsampled, pose) cells; measure distinct-cell count K at
  argmax-flip-tolerance and pose-tube tolerance. KILL if K ≈ 600 (every pair is its own cell — drive not
  self-similar). **NAMED reuse:** the T1 k-means infrastructure, `a1_specialized_inverter.py` (VQ packet),
  `pair_index_lookup_table.py` (ORPHAN — exactly the per-pair index carrier), `stable_orbit_packet_diet.py`.

### (bonus) J. CUDA-AXIS SEPARATE OPTIMIZATION as an originality surface
- Not a class shift, but worth flagging: the CUDA GT decode differs from CPU (`yuv420_to_rgb` vs NVDEC),
  and the CPU archive does NOT transfer (`cuda_axis_frontier_eval_verdict`: 0.226 > 0.205). A score-native
  carrier (B/H) re-tuned to the CUDA GT decode is a SEPARATE submission — two original artifacts from one
  method. Deprioritized vs B/D/G/H/I but a free positive externality of building B per-axis.

---

## 4. RANKING + THE #1 OFFENSIVE LEVER'S $0 DESCENT-SMOKE SPEC

### Ranking (innovation × headroom-to-T_3 × feasibility/$)

| Rank | Lever | Innovation | ΔS band (derived) | Feasibility/$ | Class-shift unquestionable? |
|---|---|---|---:|---|---|
| **1** | **B — score-native decomposition carrier (label-map generator + pose trajectory)** | **HIGHEST clean** | **−0.03 to −0.05 → S~0.14-0.16** | MEDIUM ($0 smoke→$2-5 campaign) | **YES — stores the measured quantities, classification objective** |
| **2** | **D — inverse-steganalysis margin-cost STC seg coder** | HIGH | −0.01 to −0.03 | LOW ($0 smoke, codec exists) | YES — the contest's own theory, beats the 1.27 B/flip floor |
| **3** | **H — pose-sidecar + seg-only renderer (warp-frame0)** | HIGH | −0.02 to −0.04 | LOW-MED ($0 smoke, warp exists) | YES — carrier decomposed by score-term orthogonality |
| 4 | I — quotient-dictionary cell carrier | HIGH | −0.03 to −0.06 (if self-similar) | MED | YES — stores the evaluator quotient index |
| 5 | G — argmax-invariance budget (hinge objective) | NOVEL | −0.01 to −0.03 (multiplier on B) | $0 probe | YES — the seg-logit-null basis |
| — | A (thesis frame, realized as B/D), C (RGB fallback), E (B's extension), F (floor, free in B's smoke) | — | — | — | A/E yes; C no (incremental) |

**The #1 OFFENSIVE LEVER: B — the score-native decomposition carrier.** It is the cleanest unquestionable
class shift (a carrier whose sections ARE the score terms, trained with a CLASSIFICATION objective to hit a
frozen classifier's argmax — never built for this contest), it has the largest derivable headroom toward
T_3, all aiming surfaces are in hand, and it SUBSUMES C (its seg-generator is C done score-natively),
extends to E, and stacks with D/G/H/I (D codes B's residual; G shrinks B's generator; H decomposes B's
frame0; I dedups B's cells). G/D/H/I are also $0-smokeable in parallel as B's force-multipliers.

### THE $0 DESCENT-SMOKE SPEC for lever B (ready to launch, MLX-first, $0, ~30-45 min)

**Name:** `lane_score_native_seg_generator_descent_smoke_20260610` (pre-register before any spend).
**Authority:** `[macOS-CPU advisory]` / MLX research-signal. NO score claim. The exact paired CPU+CUDA
eval is the only authority for any ΔS; this smoke only CONFIRMS-OR-KILLS the mechanism before campaign.

**The pre-registered HYPOTHESIS (H_B):** a SMALL conditional label-map generator g(pose_t, t) → 5-class
logits at 512×384, trained with an **argmax-flip (hinge/CE) objective against the frozen SegNet's argmax
on the 600 frame1s**, reaches d_seg < the frontier's seg-pool level at a weight blob SMALLER than the
seg-attributable share of the 162 KB decoder — i.e. a classification carrier is cheaper than an RGB
renderer for the seg term, because argmax-match is invariant to the 80.67% scorer-null pixel energy.

**Procedure (all $0, local, MLX-first → numpy reference):**
1. **Build the seg target ($0, reuse):** decode the 600 GT frame1s via `frame_utils.yuv420_to_rgb` (the
   CPU-correct GT — NEVER MPS), run the FROZEN `SegNet` (`modules.py`) to get the 600 ground-truth 5-class
   argmax maps at 512×384. This is the LITERAL d_seg target. (Cache to SSD `/Volumes/VertigoDataTier`.)
2. **Build the pose target ($0, reuse):** run frozen `PoseNet` on the 600 GT pairs → the 600×6 pose
   trajectory. Store it as the direct carrier section (raw fp16 = 7,200 B; measure its entropy = direction
   F's free byproduct).
3. **Train the SMALL label-map generator (MLX, ~300 ep, 16-pair smoke first then 600):** a tiny
   coordinate-INR / conv generator (reuse `coin_pp_implicit_neural_representation/mlx_renderer.py` +
   `siren/architecture.py`) with a **5-class logit head** (NOT an RGB head). Loss = per-pixel
   cross-entropy to the frozen argmax map + an **argmax-hinge term** (direction G: zero loss where the
   predicted argmax already matches with margin). Condition on (pose_t, t). QAT-in-loop. EMA. Mandatory
   `differentiable_eval_roundtrip` semantics for the resize.
4. **Measure the descent + the byte cost:** (a) exact local-CPU-torch SegNet d_seg of generator-argmax vs
   GT-argmax (advisory exact-scorer, NOT MPS); (b) the quantized generator weight blob bytes via
   `pr101_split_brotli_codec`; (c) the pose section bytes. Compose: predicted S_seg-section + S_pose-section.

**PRE-REGISTERED PREDICTION (the descent target):** by epoch ~300 on the 16-pair smoke, the generator's
d_seg DESCENDS below 0.10 (proving the label-map generator LEARNS the argmax structure — the B1-R2
mean-field bug does NOT recur for a classification objective), and the extrapolated full-600 quantized blob
is on track for < ~100 KB at d_seg approaching the frontier's 5.6e-4. (The 0.10 smoke bar is the
descent-PROOF, mirroring the roadmap RANK-1 gate; the byte/d_seg full target is the campaign goal.)

**KILL CRITERION (prove-or-pivot, pre-registered):**
- **KILL-B-MECHANISM:** if d_seg does NOT descend below 0.10 by ~300 ep (the generator cannot learn the
  argmax even with a classification objective + hinge), then the seg term is NOT cheaply generatable and B
  is FALSIFIED-at-implementation → PIVOT to D (code the seg as a margin-cost STC sidecar on the EXISTING
  frontier reconstruction, which needs no generator) and to H (delete frame0 instead, attacking rate from
  the pose side). Localizes the crux: "is the seg term generatable or only codeable?"
- **KILL-B-RATE:** if d_seg DOES descend but the quantized generator blob is NOT smaller than the seg
  share of the decoder (the classification carrier is not cheaper than the RGB carrier), then B's rate bet
  fails → PIVOT to I (the cells are not generatable-cheap but may be DEDUP-cheap) and stack G harder.
- Either KILL is a CRUX-LOCALIZING negative that redirects (outranks a non-moving positive per the goal).

**Stop/continue:** if the smoke DESCENDS past the 0.10 bar AND the byte trajectory is sub-decoder, PROMOTE
to the staged 600-pair campaign + ONE paired CPU+CUDA exact eval (~$0.6) on the byte-closed archive — the
first score-native submission candidate. If KILL, the pivot lever is already named and $0-smokeable.

**Reuse (NAMED, no rebuild — per the orphan inventory):** `coin_pp_implicit_neural_representation/*` +
`siren/*` (the INR vehicle), `modules.SegNet`/`PoseNet` (frozen targets), `frame_utils.yuv420_to_rgb`
(CPU GT), `segnet_boundary_marginals` + flip map (the seg target + margin), `tropical_argmax_boundary_grammar`
(argmax-hinge, ORPHAN revival), `ego_motion_concentration` + `foveation_ego_motion` (pose prior),
`differentiable_eval_roundtrip` (mandatory), `pr101_split_brotli_codec` (archive grammar),
`lf_payload_rate_distortion` (THE LAW admission), `null_space_exploiter` (G's logit-null when extended),
`canonical_kernels` (MLX→numpy→torch portability per Catalog #383).

---

## 5. THE TOP-3 RANKED OFFENSIVE LEVERS (the deliverable summary)

1. **B — Score-native decomposition carrier.** Predicted **ΔS −0.03 to −0.05 → S~0.14-0.16**. Class shift:
   the carrier's sections ARE the measured quantities (600×6 pose trajectory stored directly + a small
   label-map generator trained with a CLASSIFICATION/argmax-hinge objective to hit the frozen SegNet's
   argmax). No leaderboard entry stores the measured quantities; everyone stores RGB pixels and lets the
   scorer derive them. A classifier-hitting-a-frozen-classifier's-argmax is strictly cheaper than an
   RGB renderer because argmax is invariant to the 80.67% scorer-null pixel energy.
2. **D — Inverse-steganalysis margin-cost STC seg coder.** Predicted **ΔS −0.01 to −0.03**. Class shift:
   the contest IS inverse steganalysis (Yousfi = Fridrich's student); invert STC's distortion-minimization
   to minimize archive bytes subject to the argmax cell, per-pixel cost = the EXACT measured margin. Beats
   the naive seg-repair sidecar's 1.525 B/flip position-only floor by exploiting the structured cost the
   flip map measured. Reactivates the prior STC DEFER with the margin-cost map it was missing.
3. **H — Pose-sidecar + seg-only renderer (warp-frame0).** Predicted **ΔS −0.02 to −0.04**. Class shift:
   decompose the carrier BY SCORE-TERM ORTHOGONALITY — frame0 is SegNet-invisible and its only scored
   quantity (pose) is derivable from frame1 + the 6-DOF trajectory, so frame0 = warp(frame1, pose) is
   stored in rate-free inflate.py CODE, deleting ~half the per-pair frame content.

**The single most promising direction we had NOT conceived: G — the ARGMAX-INVARIANCE BUDGET (the
seg-logit-null basis).** We built an invisibility basis for the pixel→resize null (80.67%) but never built
its dual: the seg-LOGIT→argmax null, defined by the per-pixel margin. Because d_seg is an argmax-flip rate,
the scorer is INVARIANT to any logit perturbation that stays inside the margin — and 91% of pixels are in
the LARGE-margin interior (the inverse of the 91%-at-margin<0.5 flip concentration). A seg carrier trained
with an argmax-hinge loss is free to dump all its representational error into that interior, making it
fundamentally cheaper to encode than any L2/CE/RGB carrier. G is the seg-axis analog of the resize-null
basis, it is the force-multiplier that makes B's generator small, and it is a publishable original analysis
in its own right (the certified-free logit budget of a frozen argmax classifier on this video).

---

## 6. WIRE-IN (Catalog #125) + provenance

- **Hook #1 sensitivity-map:** B/D/G consume the flip map + `segnet_boundary_marginals` (the margin =
  the seg-logit-null budget); H/B consume the B2 pose atlas + `ego_motion_concentration`.
- **Hook #2 Pareto:** B/H/I move the carrier OFF the pixel-native Pareto vertex (re-decomposition, the
  only moves the exhaustion map left open); D/G are the distortion-axis coders on the new base.
- **Hook #3 bit-allocator:** B's pose-direct + seg-generator split IS the allocator; D's margin-cost STC +
  G's argmax-hinge are training/coding-time allocators; I's quotient index is the coarsest allocator.
- **Hook #4 cathedral-autopilot:** the B descent-smoke → paired-eval is the dispatch surface (same harvest
  queue as the AFSR-1 RANK 1); D/G/H/I smokes parallel-dispatch (race-mode, distinct levers).
- **Hook #5 continual-learning:** the B smoke's descent-or-not + the F entropy measurement (pose+seg+null)
  reseed the V3 judge on whether the score-native decomposition is a live class-shift axis the pixel-native
  exhaustion map could not see.
- **Hook #6 probe-disambiguator:** every lever's $0 first test IS a disambiguator (B: generatable vs
  codeable; D: structured-cost STC vs floor; G: interior logit budget exists; H: warp-frame0 d_pose; I:
  cell self-similarity K). Each resolves the lever's open question before any spend.

**Provenance:** every score-mechanism claim cites `upstream/{evaluate.py, modules.py, frame_utils.py}`
read in full (§0 line cites); every reuse target cites `evaluator_inverse_orphan_inventory_20260609.md`
(103 surfaces) + repo greps this session; every literature claim cites an external source (§1) and is
held EXTERNAL/adversarial until grounded against the evaluator. Budgets (115,596 B to T_3; 57,064 B/frame1
seg raw; 1-3 KB pose; 80.67% null; 1.525 B/flip floor) derived this session from the score law + landed
verdicts. NO score claim; `[macOS-CPU advisory]`; the exact paired CPU+CUDA eval is the only authority.

**Cross-refs:** `GOAL_standing_v3_20260610.md` (the mandate) ·
`innovation_mandate_and_original_directions_20260610.md` (the seed menu A-F this grounds + extends G-J) ·
`MASTER_ROADMAP_post_exhaustion_map_20260610.md` (RANK 1 AFSR-1 = direction C; this memo's B subsumes it;
corrects the RANK-5 "seg storage loses" framing — raw is 34 MB but the term is GENERATABLE/amortizable) ·
`untapped_technique_inventory_20260610.md` (T1=I's latent sister, T2=H's test, T5=B/G's null-training) ·
`evaluator_inverse_orphan_inventory_20260609.md` (the 103-surface reuse map — IMPORT, do not rebuild) ·
`stacking_synergy_composition_plan_20260610.md` (B/D/G/H/I stack — disjoint score-mechanisms) ·
`frontier_seg_repair_pool_verdict_20260610.md` (the 1.525 B/flip floor D beats) ·
`feedback_stc_clean_source_mask_delta_disambiguator_probe_landed_20260530.md` (the uniform-cost STC DEFER
D reactivates with margin-cost) · `src/tac/optimization/frozen_evaluator_contract.py` (direction A scaffold).
