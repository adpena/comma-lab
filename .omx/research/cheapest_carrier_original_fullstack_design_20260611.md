# The cheapest carrier — ORIGINAL full-stack design from measured scorer economics (2026-06-11)

**Operator directive (2026-06-11, verbatim intent):** "the cheapest carrier is likely not currently
elaborated in literature and we likely need to design the full stack ourselves using all lessons learned
about PoseNet and SegNet economics and contest scorer including long wiggly boundaries and residuals and
sensitivity and profiling and marginal values and everything."

**Authority:** design synthesis; every economic fact below is tagged [MEASURED] (this effort's exact-scorer
artifacts) or [DERIVED]. Frontier UNMOVED 0.19110 [contest-CPU], 177,169 B. `N=37,545,489`. This memo is the
design (a means); the build + byte-closed exact eval is the end. NO score claim here.

---

## 1. THE MEASURED ECONOMICS (the design constraints — what the carrier must obey)

Every off-the-shelf neural codec (Cool-Chic, KAN, HNeRV, hyperprior) is **scorer-AGNOSTIC**: it spends
decoder capacity ~uniformly to reconstruct ALL pixels. But the contest scorer is **wildly non-uniform**.
The cheapest carrier is the one whose every byte is allocated by the scorer's measured marginal value.
The facts the design must exploit:

1. **Rate is 62% of S** [MEASURED: 0.118 of 0.191]. The carrier is byte-minimal above all else. seg 0.056
   (d_seg 5.6e-4) + pose 0.017 (d_pose 2.94e-5) are the other 38%.
2. **Appearance ≫ partition — CONFIRMED at the tolerance** [MEASURED, B-WITNESS lossless + Yousfi tolerance
   re-audit 2026-06-11]: the Yousfi re-measure RAN the tolerance-exploiting solve B-WITNESS skipped
   (`solve_mdl_region_merge` + UNIWARD margin-weighting; tolerance sweep had ZERO prior callers). Result:
   the standalone partition store is rate **0.31–0.33 at the frontier d_seg band** (5× d_seg budget → only
   ~12% bytes); even a *perfect* STC chain-coder is ~0.16–0.22 — **no crossover with neural 0.108 exists.**
   The reason is an **amortization gap, not the coder**: the neural decoder amortizes the boundary across
   600 frames via shared weights; a per-frame partition store re-pays boundary entropy (2,782 cracks/frame)
   every frame. ⇒ the cheapest carrier IS the neural appearance basis (C0). The non-neural standalone branch
   is **CLOSED** (structural ~2× gap). The boundaries are expensive to *describe* per-frame, cheap to
   *amortize+regenerate* — exactly the neural carrier's win.

   **C1 REFINED (the Yousfi eureka):** C1 is NOT a generic boundary residual — it is the **sparse
   residual-FLIP delta**: code ONLY the pixels where the (compact, not-yet-floored) neural base's argmax
   disagrees with L*, via region-merge + `pack_sparse_delta` at UNIWARD-margin cost → bytes ∝ *residual
   flips*, not boundary cracks. On the frozen frontier this has ~0 headroom (already floored, 14-byte
   delta); its headroom is on a SMALLER base (B1 Cool-Chic) that hasn't reached the floor — nudge its few
   remaining flips cheaply instead of training all the way down. C1 composes ON C0, sequenced after the
   C0 base byte-closes (depends on B1-CLOSE #97).
3. **SegNet reacts ONLY at decision boundaries** [MEASURED: spectral atlas — SegNet broadly weak, max
   H_seg ~0.009; d_seg = argmax-flip RATE; frontier d_seg 5.6e-4 ≈ ~5 flipped px/frame]. The interior of
   each region is argmax-robust; only the thin boundary band (long wiggly contours, ~2.16% of px [MEASURED,
   B6: 4252 px/frame]) is flip-prone. ⇒ appearance fidelity matters ONLY in the boundary band.
4. **PoseNet needs DENSE texture, both frames, but only in a narrow subspace** [MEASURED: spectral atlas —
   PoseNet strongest at LOW spatial freq + HORIZONTAL orientation (H_pose 0.587 band0-horizontal); pose
   marginal 271 = 2.7× seg's 100; pose-null = low-Mahalanobis directions]. ⇒ the carrier must preserve
   low-freq-horizontal texture across both frames, but can shed pose-null detail.
5. **~80% of appearance is in the certified-invisible subspace** [DERIVED, GOAL_v3 / null_space_exploiter]:
   directions the scorer's Jacobian ignores. ⇒ spend ZERO bytes there.
6. **The surrogate is suprafloor-valid only** [MEASURED, #92: smooth_disagreement ρ=0.99 for d_seg>5e-3,
   noise at the 5.6e-4 basin]. ⇒ the training curriculum descends with the surrogate to ~5e-3, then the
   FINAL approach to the basin needs exact-d_seg-in-the-loop (boundary-band-gated) — pose uses its
   per-dim Mahalanobis tube throughout.
7. **Pose is stored, not reconstructed** [Quantizr lesson, MEASURED capstone: stored_latent holds d_pose
   ≤3e-4]. ⇒ store the 6 pose scalars/pair directly (~1KB total) + FiLM-condition the decoder.
8. **The decoder weights are ~94% of the bytes** [MEASURED, PR95 lineage]. ⇒ the rate lever is the DECODER
   capacity, and the only legal way to cut it (B2 [MEASURED]: a shipped base-init is an illegal large
   artifact) is a structurally smaller decoder — shaped by 3/4/5 above.

## 2. THE ORIGINAL PRINCIPLE — a scorer-marginal-shaped appearance carrier

**Insight (the synthesis):** a uniform 162KB decoder wastes ~80% of its bytes reconstructing
invisible-subspace appearance. The cheapest carrier allocates decoder capacity by the scorer's measured
marginal-value-per-byte (the joint P18/P19 field), so bytes land ONLY where the scorer reacts:

`capacity(region, freq, frame) ∝ marginal_value = 100·|∂d_seg/∂x| (boundary-band, frame1)
                                                   + (5/√(10·d_pose))·‖J_pose‖_{Σ⁻¹} (low-freq-horiz, both frames)`

and ≈ 0 in the invisible subspace / region interiors / pose-null. This is NOT a literature codec — it is a
decoder whose ARCHITECTURE + BIT-ALLOCATION are derived from the frozen scorer's geometry on THIS video.

## 3. THE ARCHITECTURE — three factored carriers, each scorer-shaped

A factored appearance decoder (sum of three cheap streams), NOT one uniform HNeRV:

- **(C0) Coarse global appearance** — a tiny low-freq base decoder (HNeRV-class, but a fraction of 162KB),
  shared across all pairs + per-pair latent + FiLM(pose). Carries the gist that makes SegNet roughly right
  and PoseNet's low-freq-horizontal structure. This is the bulk-but-cheap stream (low-freq = few bytes).
- **(C1) Boundary-band high-freq residual** — capacity added ONLY in the SegNet boundary band (the long
  wiggly contours where flips live). A sparse, spatially-gated residual (predicted from C0's own margin
  field — no scorer at inflate) that sharpens the argmax exactly where d_seg is binding. This is where the
  "long wiggly boundary" lesson pays: we don't *describe* the boundary (525KB), we *spend a little residual
  capacity to regenerate it sharply* (cheap, because it's only 2% of pixels and predicted, not stored
  per-pixel).
- **(C2) Pose-texture channel** — a low-freq, horizontal-biased texture across BOTH frames in the
  pose-sensitive subspace, FiLM-driven by the stored 6-pose scalars. Holds d_pose in its Mahalanobis tube;
  carries nothing in the pose-null.

Pose: 6 scalars/pair stored directly (~1KB) + FiLM. The invisible subspace gets ZERO capacity in all three.

## 3b. THE COOL-CHIC-FAMILY REALIZATION + OPTIMAL CAPACITY (operator 2026-06-11)

**Source:** operator — "could Cool-Chic or Cool-Chic-like be adapted for our own contest-custom optimal
carrier design such that it has optimal capacity?" YES — Cool-Chic is the right BACKBONE (MEASURED 17×
rate win, B1-CLOSE), and C0/C1/C2 above are its scorer-shaped realization. The capacity splits three ways:
multiresolution **latent grids** (per-frame info), the **shared synthesis net** (amortized over 600 frames),
the **ARM entropy model** (codes latents). "Optimal capacity" = allocate each by the scorer's marginal value
+ size the synth to ARGMAX-STABILITY, not pixel-perfection.

**The capacity-requirement insight (why the basin could be cheap):** under CE (pixel-perfect target) the
synth needs ~162KB conv-HNeRV-class capacity to reconstruct full appearance. Under the **margin-polytope
hinge** (correct-SIDE-of-`modules.py`-boundary target) it only needs enough to put each pixel's SegNet
logits on the right side → far less. The deep seg-profile [MEASURED] proved ~500 params is too small
(diffuse argmax-instability, post-hoc repair a wash); the optimum is between ~500 and 162KB, and the hinge
pushes it toward the small end. **The A/B/C re-fit (running) measures exactly where it sits.**

**The contest-custom optimal-capacity carrier (the design):**
- **Shared synth** sized to argmax-stability + pose-faithful texture (hinge-minimized, NOT pixel-perfect).
- **Per-frame latents** multiresolution, allocated by the joint P18/P19 field: FINE at the SegNet boundary
  band + pose-sensitive regions, COARSE in the invisible interiors (the per-frame-byte "spend where the
  scorer reacts").
- **ARM entropy** with the validated QA-entropy (#92) so coded bits == real archive bytes.
- **Stored 6-pose-scalar + FiLM** (pose tube) + the validated **1.28 B/flip flip-delta** for final-mile
  d_seg cleanup once the base is near-basin.
- **Optimal capacity = the RD optimum**: minimize bytes s.t. d_seg ≤ basin ∧ d_pose ≤ tube; the hinge +
  scorer-shaping move that optimum to the smallest argmax-stable carrier.

**HONEST risk (non-sycophantic):** if even the optimal scorer-shaped Cool-Chic-family needs near-162KB to
stabilize the argmax, the rate advantage shrinks toward conv-HNeRV and the basin is capacity-bound (→
Cool-Chic stays a rate-only win; sub-0.15 needs frontier-class capacity + PR95-scale compute). The A/B/C
re-fit decides this empirically against the exact `modules.py`/`evaluate.py`/0.mkv oracle.

## 4. THE FULL STACK

| Layer | Design (scorer-economics-shaped) |
|---|---|
| Representation | C0 coarse + C1 boundary-residual + C2 pose-texture; per-pair latent + 6 pose scalars |
| Bit-allocator | joint P18/P19 marginal-value-per-byte field (`joint_p18_p19_waterfill.py` + boundary-mass + null_space_exploiter) shapes per-region/per-freq/per-frame capacity; THE law `keep iff −ΔS_dist > 25·Δbytes/N` (`lf_payload_rate_distortion.py`) |
| Training | PR95 8-stage curriculum; loss = 100·(suprafloor smooth_disagreement → boundary-band CE → exact-d_seg-gated) + Σ per-dim-Mahalanobis pose (both frames); EMA warmup; eval-roundtrip in-loop; the #92 surrogate suprafloor + exact d_seg at the basin |
| Archive grammar | monolithic 0.bin: C0 weights (int8+brotli) + C1 residual codebook + C2 channel + per-pair latents (temporal-delta LZMA) + 6-pose scalars + the allocation map; PR95 L20-L32 entropy stack |
| Inflate | scorer-free, numpy-portable: C0(latent,FiLM(pose)) + C1 boundary-residual (self-predicted) + C2 pose-texture → RGB; ≤100 LOC, ≤2 dep, CPU+CUDA |
| Score-aware | the bit-allocator IS the score-awareness — capacity follows the frozen scorer's marginal value, measured once on 0.mkv |

## 4b. FIRMWARE-GRADE BYTE-ENCODING — the constrained-env extreme-optimization layer (operator 2026-06-11)

**Source:** operator directive — "remember the thinking about firmware and edge computing and constrained-env
extreme optimization and bit packing and shifting and clever multi-representation tricks." This is the
BYTE-ENCODING layer, distinct from the architecture (C0/C1/C2): once the carrier's content is fixed, treat
the archive like FIRMWARE — every bit counts, pack/shift at bit granularity, pick the cheapest representation
per section. It directly attacks the two MEASURED byte-gaps below.

**The measured gaps it must close:**
- **ARM-vs-real ×2.85** [MEASURED, B1-CLOSE]: the Cool-Chic byte-close used int8 weights (1 B/param) + a
  per-grid empirical freq table. Recover via: (a) **sub-byte weight packing** — int4/int2/ternary per-tensor
  where the ~500-param synth's d_seg tolerance allows (the synth is tiny + over-precise); (b) **block-FP
  self-compression** of the synth weights (`block_fp_jfg.py`, `hessian_block_fp.py`, `self_compressing_nn.py`
  — the PR56 1.017-bpw selfcomp lesson); (c) **tighter latent entropy** — temporal-delta uint8 (PR95 L25) +
  raw LZMA1 filters (L24) + a real context-model ARM instead of the coarse per-grid freq table.
- **flip-delta 187 B/flip → <2 B/flip** [MEASURED, #98]: encode the flip as **(combinatorial colex-rank
  position, PR95 L31) + (sign bit) + (few-bit margin-quantum)**, bit-packed — NOT 3 full int8 RGB channels.
  This is the in-flight cheap-encoding (#98 sub-node).

**The multi-representation per-section rule (PR95 L20–L32, reuse `pr101_split_brotli_codec_derivers.py` +
`packet_section_transform.py`):** per archive section, pick the cheapest of {split-brotli-q11 (L23/L32),
raw-LZMA1 (L24), range/arithmetic (L30), per-tensor byte-map zig/twos/off (L21) + conv storage perm (L22)}.
Monolithic 0.bin, fixed offsets in source. fp16 per-tensor scales (L29). Canonical-Huffman length-vector
rank for the sidecar (L26).

**Native bit-exact lowering (the firmware/edge end-state):** the settled byte-encoding lowers into
`runtime-rs/crates/tac-packet-compiler` (+ `qma-codec`/`residual-codec`) with golden_vectors + a Python
oracle (per the "Deterministic packet compiler" + "Native eval-time runtime discipline" non-negotiables) —
bit-identical, ≤100-LOC numpy-portable inflate as the reference, native as the speed/control layer. Payload
stays clean (no learned/video-derived constant outside archive.zip).

**Queued node (B1-PACK):** a firmware-grade byte-encoding pass on the Cool-Chic byte-close — sub-byte +
block-FP weight pack + the L20–L32 per-section multi-representation + the colex-rank flip-delta — measured
real-bytes recovery vs the ×2.85 baseline. Launch when API recovers (currently 529-overloaded). Reuse the
arsenal above; do not rebuild.

## 5. INNOVATION ACCOUNTING (NO-FAKE originality gate)

- **Ours-original:** the scorer-marginal-shaped capacity allocation (C0/C1/C2 factoring by the measured
  P18/P19 field + invisible-subspace shedding); the boundary-band self-predicted residual (regenerate, not
  describe, the wiggly contour); the pose-sensitive-subspace texture channel. None is a literature codec.
- **Borrowed (defensive substrate):** the HNeRV base block (C0), the PR95 entropy stack (grammar), the
  curriculum, the joint P18/P19 + null-space + contour infra (all CONSUMED in-repo, our prior work).
- The originality is the SHAPING by measured scorer economics; the components are reused per the
  no-duplicative-code directive.

## 6. WHY IT'S CHEAPER THAN THE FRONTIER [DERIVED — to be MEASURED]

The frontier's 162KB decoder reconstructs all pixels uniformly (~80% invisible-subspace waste). If capacity
follows marginal value, the carrier spends bytes on: the ~2% boundary band (C1) + the low-freq-horizontal
pose subspace (C2) + a coarse low-freq base (C0) — plausibly a fraction of 162KB at the same d_seg/d_pose.
Pre-registered target: decoder+latents → 60–110KB (rate 0.040–0.073) at d_seg≤8e-4, d_pose-tube held →
S ≈ 0.10–0.15. The decisive UNKNOWN is whether C1's self-predicted boundary residual reaches d_seg≤8e-4 at
the reduced C0 capacity — the build measures it.

## 7. RECURSIVE-GREENUP + BUILD PLAN (the operator's senior-engineer loop)

1. **Adversarial design review** (3-clean-pass, question-all-interpretations): does C1's self-predicted
   residual leak scorer access at inflate (illegal)? does the invisible-subspace shedding survive
   uint8/resize? does C0's reduced capacity still let SegNet re-derive the partition (the appearance≫
   partition economics assumed a 162KB carrier — does it hold at 60KB)? is the P18/P19 field stable across
   the training trajectory (it's a local tangent at one archive)?
2. **$0 measure (the gate):** on n48, ablate C0 capacity at fixed C1/C2 → the param-at-d_seg-basin curve
   for the SHAPED carrier (vs B1's uniform Cool-Chic) — does shaping reach the basin at fewer bytes?
3. **Build full-stack** (top-AIML, export+numpy-parity from byte zero) → byte-close → advisory S → if local
   sub-T_1, paired contest-CPU/CUDA exact eval (the pointer-mover).

Folds into the DAG (`sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`) as the THREAD-B capstone
carrier; B1 (Cool-Chic uniform) is the baseline this scorer-shaped design must beat; A2' (full-Jacobian
postfilter) is a refinement layer on top of C0–C2.
