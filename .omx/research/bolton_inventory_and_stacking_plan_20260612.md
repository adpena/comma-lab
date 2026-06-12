# Bolt-on inventory + stacking plan for the base_ch=20 HNeRV substrate (2026-06-12)

**Subagent:** `bolton-inventory-20260612` (READ-ONLY analysis; GPU-free; this memo is the only artifact).
**Evidence grade:** `[macOS-CPU advisory]` / mechanism + accounting only. NO score claims, NO dispatch, `promotable=false`.
**Operator ask:** comprehensive inventory of ALL orthogonal score-lowering bolt-ons that STACK on the NEW
base_ch=20 HNeRV substrate (training now, ~0.75 descending), ranked stacking plan, + promotable-to-in-curriculum shortlist.
**Frontier at audit (pointer — never hardcoded):** contest-CPU **0.19109982** (177,169 B, `lane_pr110_payload_entropy_recode_20260610`)
/ contest-CUDA **0.20533003** (186,876 B). Score law: `S = 100·d_seg + √(10·d_pose) + 25·B/N`, N=37,545,489. **Byte price ≈ 6.66e-7 score/byte.**

---

## 0. THE LOAD-BEARING STRUCTURAL FACT (read this before the table)

The current CPU frontier already **absorbed PR #112** — the operator's "idle/deferred PR112" is NOT idle anymore: it
was vendored as `tac.packet_compiler.ctx_range_coder` (MIT, attributed) and applied to OUR R3 frontier on 2026-06-10,
producing `lane_pr110_payload_entropy_recode` = the current pointer (`leapfrog_pr112_absorb_recode_verdict_20260610.md`).
So "PR110" and "PR112" are not two separate idle bolt-ons; they are **one shipped lossless rate-recode stack** on the
FP11 packet grammar. The re-activation the operator wants is **applying that whole recode stack to the base_ch=20
substrate's exported packet**, not re-discovering PR112.

**The substrate-portability axis is the whole game — AND base_ch=20 does NOT use the FP11 grammar.** The base_ch=20
vehicle is the **capstone VQ-NeRV** (`tac.capstone_vq_nerv`): its archive (`build_capstone_archive_bytes`) is a
DIFFERENT grammar — VQ-index codebook + stored per-pair latent + brotli-packed decoder weights — NOT the PR101/PR110
FP11 packet (split-brotli decoder blob + LZMA1 latent blob + FECa selector + sidecar). This is load-bearing:

- **The FP11-grammar recodes (R1/R2/R3/T1/T4/T8/T9) do NOT port for free.** PR#112's `ctx_range_coder` and
  `pr110_payload_entropy_recode` are hard-wired to the FP11 section layout (`split_fp11_member`, FECa magic, the
  PR#101 `LATENT_BLOB_LEN=15387` / split-brotli decoder constants). To reuse them on base_ch=20 you must EITHER
  (a) **re-target the coder to the capstone grammar's analogous sections** (the per-tensor adaptive range coder is
  grammar-agnostic in PRINCIPLE — it codes a weight tensor and a latent stream — but the materializer that splits/
  joins the packet must be rewritten for the capstone container), OR (b) **transcode capstone → FP11** (only if the
  capstone decoder maps cleanly onto the PR101 HNeRV decoder layout, which is NOT guaranteed for a VQ-index codebook).
  Option (a) is the realistic path: lift the *coder primitive* (`ctx_range_coder`), drop the *FP11 materializer*, and
  write a `capstone_payload_entropy_recode` that splits the capstone container, range-codes its weight + latent
  sections, and byte-closes. ~half a day of materializer work — the coder math is done.
- **The pixel/scorer-DOF bolt-ons (S12, PR98, T10, LeverD, frame0-pose-selector, residual-basis sidecars) DO port
  for free.** They operate on the decoded RGB frames / the scorer's null space / the renderer's own margin field —
  all grammar-independent. Any renderer output qualifies, capstone included.
- **The HNeRV-decoder bolt-ons (WRQ, T11) port at the WEIGHT-TENSOR level.** `score_aware_weight_requant` re-quants
  decoder TENSORS by scorer sensitivity and re-packs into the archive; it is written for "the frontier HNeRV decoder
  weights" but the mechanism is per-tensor and applies to the capstone decoder tensors once the re-pack targets the
  capstone container. T11 (channel prune + finetune) is fully substrate-agnostic at the arch level.

**So the gating integration prerequisite is NOT "export to FP11" — it is "write the capstone-grammar materializer that
the lifted `ctx_range_coder` + `score_aware_weight_requant` re-pack into."** That single materializer unlocks the
lossless-rate + weight-requant half of the stack on the capstone container. The pixel/scorer-DOF half needs no port.

---

## 1. THE CATALOG (every orthogonal bolt-on)

Columns: **Attacks** (rate / d_seg / d_pose) · **Portability** (AGNOSTIC = applies to base_ch=20 byte/pixel DOF; FP11
= needs base_ch=20→FP11 export; HNERV = needs HNeRV decoder, base_ch=20 qualifies; SPECIFIC = old-substrate-locked) ·
**Magnitude** (measured anchor where it exists; else derived band) · **Status** · **Byte-close integration point**.

### A. LOSSLESS RATE RECODES (zero distortion by construction — the free stack)

| # | Bolt-on | Attacks | Portability | Magnitude (anchor) | Status | Integration point |
|---|---|---|---|---|---|---|
| **R1** | **PR#112 per-tensor adaptive range coder on decoder weights** (`ctx_range_coder` + `pr110_payload_entropy_recode`) | rate | **FP11** (HNeRV INT8 decoder blob) | **−1,023 B** measured on R3 decoder (162,127→161,104) | **SHIPPED on frontier** (lane_pr110_payload_entropy_recode) | recodes the split-brotli decoder section of the FP11 member; lossless fail-closed gate |
| **R2** | **PR#112 per-dim AR(1)+cross-dim latent range coder** (same module, latent section) | rate | **FP11** (28-d per-pair latents) | **−317 B** measured on R3 latents (15,387→15,070) | **SHIPPED on frontier** | recodes the LZMA1 latent section of the FP11 member |
| **R3** | **FECa/FEC10-hybrid selector + framing reparameterize** (`feca_selector_reparameterize`) | rate (+ distortion via richer selector) | FP11 (selector is the frame0-perturbation menu) | **−22 B** framing/mode-table (selector spends bytes to LOWER d_seg/d_pose) | SHIPPED on frontier | selector section; verbatim-preserved by R1/R2 |
| **S12** | **resize-null preimage** (`evaluator_invisibility_basis` / `resize_null_preimage`) — fill the certified-invisible 22.7% of every channel with maximally-compressible values | rate (CERTIFIED zero distortion) | **AGNOSTIC** (pixel DOF, any renderer output) | −10 to −19.5% of *coded frame bytes* (certified, basis landed) | landed primitive; **NOT folded into the recode bundle** | postprocess on the decoded frames BEFORE entropy recode → lower-entropy input to R1/R2 (force-multiplier, test joint) |
| **HFV** | **HFV sidecar recoder** (`build_hfv_sidecar_recoder`) — RLE/identity-row collapse of the 24,016 B foveation sidecar | rate (lossless) | SPECIFIC (only if base_ch=20 ships an HFV foveation sidecar) | 97–100% identity rows → large collapse | built | only relevant if foveation sidecar is in the packet |

### B. LOSSLESS / NEAR-LOSSLESS RATE — UNBUILT cross-pair + null-space levers (the biggest untapped)

| # | Bolt-on | Attacks | Portability | Magnitude (derived) | Status | Integration point |
|---|---|---|---|---|---|---|
| **T1** | **cross-pair latent dedup / clustering** — K-codebook + per-pair index + sparse residual (single-video `0.mkv` → 600 pairs of ONE drive, near-stationary stretches → many near-duplicate latents) | rate | **FP11** (operates on the 28-d per-pair latents; HNeRV-agnostic) | **−0.0031 to −0.0061** (30–60% of the 15.4 KB latent section); LARGER than PR#112's whole win | **UNBUILT** — `untapped_technique_inventory_20260610` TOP rank; never tried (per-pair lens only) | replace per-pair LZMA/AR stream with dictionary-index codec in the FP11 latent section; ~60–90 LOC + 1 paired replay |
| **T8** | **latents projected to scorer-null before coding** (regenerate, don't just recode — push latent codes toward SegNet/PoseNet null pre-entropy) | rate | FP11 + HNERV (needs decoder Jacobian) | **−0.001 to −0.005** (below the iid floor; structurally unreachable by PR#112) | UNBUILT (MOVE 3, deferred) | project latent deltas onto resize-null preimage, then R2-code the smaller residual; compounds with T1 |
| **T9** | **global decoder-weight permutation + cross-tensor shared-model clustering** (beyond PR#101 CONV4_STORAGE_PERMS + PR#112's 4-tensor sharing) | rate (lossless) | **FP11** (HNeRV weight tensors) | **−100 to −500 B → −0.00007 to −0.00033** (small; near order-0 floor) | PARTIAL (PR#101 fixed perms exist; the SEARCH is open) | extend `shared_pmf_model` clustering on top of R1's per-tensor models |
| **T4** | **selector as order-1 Markov / RLE stream** (temporally-correlated mode IDs on a single contiguous drive) | rate | FP11 (selector) | **−50 to −100 B → −0.00003 to −0.00007** (selector is 248 B; PR#112 did order-0 only) | PARTIAL (R3 optimized the CHOICE, not the SEQUENCE entropy) | order-1 range coder on the 600 mode IDs |
| **T3** | **inflate-as-interpreter** — migrate procedural/constant sections into rate-free inflate.py code (only archive.zip is counted, `evaluate.py:63`) | rate | AGNOSTIC (compliance-bounded) | **−0.0004 to −0.0010** (small defensible sections; STANDING subsidy for procedural carriers) | RECOGNIZED, never actioned | move the sidecar/framing-constant generators into inflate.py code, ship seed in archive |

### C. WEIGHT RE-ALLOCATION (lossy on recon, INSIDE the evaluator cell — the dominant rate term)

| # | Bolt-on | Attacks | Portability | Magnitude | Status | Integration point |
|---|---|---|---|---|---|---|
| **WRQ** | **score-aware per-tensor weight RE-QUANT sweep** (`score_aware_weight_requant_sweep` + `tac.score_aware_weight_requant`) — re-allocate decoder bits by MEASURED scorer sensitivity (d_seg/d_pose tolerate far more weight error than recon does) | rate (holds d_seg/d_pose in-cell) | **HNERV** (explicitly "the frontier HNeRV decoder weights"; base_ch=20 qualifies directly) | decoder = ~91% of bytes; even modest per-tensor re-alloc is the biggest single lever after T1; magnitude pending the sweep's own exact measurement | EXACT-AUTHORITY harness BUILT (no proxy); ready to run on any HNeRV archive | re-quants tensors in q-domain → re-packs FP11/CTXR → decodes through inflate → exact d_seg/d_pose; byte-closed |
| **T11** | **structured channel pruning + survivor finetune** (zero whole channels the single-video memorizer doesn't need — DISTINCT from coarsening, which is FALSIFIED ×2) | rate | HNERV | **−0.01 to −0.02** (20–40% channel prune) IF survivors retrain to hold d_seg | RES (prune+finetune; coarsening killed but pruning is the un-killed variant) | `train_imp_cycle` IMP pruning on base_ch=20 decoder, then R1-recode survivors |

### D. DISTORTION SIDECARS (d_seg / d_pose, grammar-independent pixel/correction DOF)

| # | Bolt-on | Attacks | Portability | Magnitude | Status | Integration point |
|---|---|---|---|---|---|---|
| **LeverD** | **margin-conditional seg-repair residual + waterfill** (`boundary_math/margin_conditional_residual`) — decoder regenerates the SegNet margin field for FREE, so the sidecar only addresses the decoder-KNOWN low-margin boundary set `B`, making conditional position cost `log2 C(|B|,K)` ≪ unconditional. Waterfills flips that clear the 1.27 B/flip break-even with NET>0 (receptive-field collateral priced in) | d_seg (− if NET>0) | **AGNOSTIC** (consumes the renderer's own margin field; any HNeRV) | conditional on whether base_ch=20's flips concentrate in `B`; on a worse-trained base (more flips) the break-even is EASIER to clear than on the frozen frontier where a naive sidecar was net-negative (#51 DEFER) | BUILT (the conditional lever that un-DEFERs #51); waterfill is fail-closed on net value | correction sidecar section; sequential admit + cone-ledger debit |
| **PR98** | **decode-side channel bias postprocess** (`engineered_corrections` — frame0 R−1/B−1, frame1 G−1 to cancel scorer color-space bias) | d_seg + d_pose (systematic offset) | **AGNOSTIC** (pixel postproc, 0 archive bytes) | **−0.0001 to −0.0005** (L28 lesson, 0 bytes) | landed; bias constants are substrate-specific (re-fit per base) | 3-line inflate.py postproc; re-derive the constants on base_ch=20 render-vs-GT |
| **T10** | **CPU-GT-decode affine color correction** (per-channel/region affine fit to the EXACT `frame_utils.yuv420_to_rgb` CPU GT, beyond PR98's 3 hand-set biases) | d_seg + d_pose | AGNOSTIC | **−0.0002 to −0.001** (2nd-order beyond PR98) | RES→SB (on-host fit) | extends the PR98 postproc with a fitted affine; CPU axis only (CUDA needs its own) |
| **frame0-pose-selector** | **frame0 pose menu** (frame0 is SegNet-blind by construction, `modules.py:108`; perturb it for d_pose at ZERO d_seg cost) | d_pose only | AGNOSTIC | the FECa selector already IS this; magnitude folded into R3 | shipped (it's the selector) | the selector section; constructively offsets frame1 seg gains |

### E. RESIDUAL-BASIS SIDECARS (over the decoded outputs — a whole scaffold family)

| # | Bolt-on | Attacks | Portability | Magnitude | Status | Integration point |
|---|---|---|---|---|---|---|
| **WaveletRes / CoolChicRes / C3Res / SIRENRes / CoordMLPRes** | residual-basis scaffolds over decoded frames (Mallat scattering / Cool-Chic / C3 / SIREN / coordinate-MLP), byte-closed PR106 sidecars | d_seg/d_pose + rate tradeoff | AGNOSTIC (over decoded outputs) | per-scaffold; several at L2 dispatch-ready (paired CPU evals landed) | L1–L2 scaffolds; built over PR106 r2 outputs | residual sidecar section over base_ch=20 decoded frames |

---

## 2. THE RANKED STACKING PLAN (apply to base_ch=20 at byte-close)

**Sequencing law** (`stacking_synergy_composition_plan_20260610` + `composition_algebra_coherence_law`): admit all
mutually-orthogonal LOSSLESS rate moves as ONE batch (proof-by-construction = identical pixels = identical d_seg/d_pose,
byte savings simply sum), then sequential-admit distortion moves with re-measure + ledger debit, measure commutators only
for same-section/same-region/both-frame pairs.

**Phase 0 — CAPSTONE-GRAMMAR MATERIALIZER (the gating prerequisite).** base_ch=20 ships the capstone VQ-NeRV container,
NOT FP11. Write a `capstone_payload_entropy_recode` materializer that splits the capstone container, runs the LIFTED
`ctx_range_coder` primitive (the PR#112 coder math, grammar-agnostic) on the capstone weight + latent sections, and
re-packs `score_aware_weight_requant`'s re-quantized tensors into the capstone container, byte-closed + lossless-gated.
The coder + requant math already exist; only the capstone split/join materializer is missing. This single tool unlocks
R1/R2/T1/T8/T9 + WRQ on the capstone grammar. The pixel/scorer-DOF moves (S12/PR98/T10/LeverD/sidecars) need NO port.

**Phase 1 — THE FREE LOSSLESS RATE BATCH (one paired eval to ratify).** All orthogonal (disjoint sections, zero distortion):

| Order | Move | Section | Expected on base_ch=20 |
|---|---|---|---|
| 1 | **R1** decoder range recode | decoder weights | ~−1,000 B (scales with decoder size; base_ch=20 decoder is SMALLER so absolute B is smaller but the % holds) |
| 2 | **R2** latent range recode | latents | ~−300 B |
| 3 | **R3** selector reparameterize | selector | ~−22 B |
| 4 | **S12** resize-null preimage | frame payload | −10–19.5% of coded frame bytes (CERTIFIED 0 distortion); run BEFORE R1/R2 (force-multiplier — lower-entropy input) |
| 5 | **T1** cross-pair latent clustering | latents | **−0.0031 to −0.0061** (the dominant lossless lever; replaces R2's per-pair stream) |

Note T1 and R2 both touch the latent section — **T1 SUBSUMES R2** (clustering is a strictly stronger latent codec than
per-pair AR). Run T1; keep R2 only as the within-cluster residual coder.

**Phase 2 — WEIGHT RE-ALLOCATION (sequential, exact-authority re-measure).**

| Order | Move | Expected |
|---|---|---|
| 6 | **WRQ** score-aware per-tensor re-quant | the largest post-T1 lever (decoder = ~91% of bytes); magnitude is the sweep's own exact output |
| 7 | **T9** global perm + cluster | −100 to −500 B (small, lossless, stacks on R1/WRQ) |
| 8 | **T8** null-projected latents | −0.001 to −0.005 (compounds with T1: cluster the null-projected latents) |
| 9 | **T4** order-1 selector | −50 to −100 B (confirm-the-bound, low EV) |

**Phase 3 — DISTORTION (sequential, ledger-debited, commutators measured).**

| Order | Move | Expected | Note |
|---|---|---|---|
| 10 | **PR98 + T10** channel bias / affine GT correction | −0.0001 to −0.001 (0 archive bytes) | re-fit constants on base_ch=20; free, do early actually |
| 11 | **LeverD** margin-conditional seg-repair | conditional; EASIER to clear break-even on a less-converged base | sequential admit + cone debit; commutes with frame0-pose-selector (positive externality) |
| 12 | **T11** channel prune + finetune | −0.01 to −0.02 IF survivors hold | RES; changes the base → re-map + re-stack everything (master loop) |
| 13 | residual-basis sidecars (Wavelet/CoolChic/...) | per-scaffold | over decoded outputs |

### Expected cumulative reduction on the base_ch=20 substrate
Once base_ch=20 trains down to a competitive base (the substrate's OWN d_seg/d_pose is the dominant term until then —
**the bolt-ons cannot fix a 0.75 base; they harvest the last fraction once the base is near the frontier's distortion**),
the lossless rate stack (Phase 1) is **proof-by-construction additive**: the four orthogonal lossless moves sum. On the
frontier the equivalent stack was ~−1,381 B (R1+R2) + S12 + T1's projected **−0.003 to −0.006**. **Total expected
lossless rate reduction once base_ch=20 reaches FP11 export ≈ −0.004 to −0.007 (T1-dominated) + ~−1,400 B (R1+R2/R3)
≈ −0.005 to −0.008 on the rate axis alone**, before WRQ (potentially the largest, magnitude TBD by its exact sweep) and
before the distortion sidecars. This is the harvest that turns a base that lands NEAR the frontier into one that BEATS it.

**Honest caveat (NO FAKE):** none of this moves the score while base_ch=20 sits at 0.75. The bolt-on stack is the
**finishing kit** for a base whose intrinsic d_seg/d_pose has descended into the frontier's neighborhood. The operator's
framing ("after it's established") is exactly right — these are post-convergence harvesters, and the magnitudes above are
derived bands (`[macOS-CPU advisory]`), authoritative only after exact paired CPU+CUDA eval on the byte-closed candidate.

---

## 3. PROMOTABLE-TO-IN-CURRICULUM SHORTLIST (Phase-2 floor-chasing — training-time levers)

These bolt-ons are MORE valuable folded into the base_ch=20 training loss/arch than applied post-hoc, because they change
what the decoder must represent (compounding, not additive):

1. **T5 — train representation error INTO the certified null space** (`evaluator_invisibility_basis` as a TRAINING
   CONSTRAINT, not a postprocess). Regularize the decoder to put its error in the resize-null → certified-free error +
   lower-entropy visible residual for R1/R2 to carry. Named "the strongest synergy" (`stacking_synergy` #3); **−0.01 to
   −0.04 as a compounding term**. THE top in-curriculum promotion.
2. **WRQ as a QAT objective** — instead of post-hoc re-quant, train base_ch=20 with score-aware per-tensor bit allocation
   in the QAT loss so the decoder is byte-minimal-for-the-scorer by construction.
3. **LeverD / margin-conditional repair as a training loss** — weight the recon loss by the SegNet margin field so the
   decoder spends capacity on the fragile boundary band `B` (where flips concentrate), reducing the flip set the sidecar
   must repair. Folds the seg-repair lever into the curriculum.
4. **T2 — cheapest-frame0 synthesis** — add a warp-residual frame0 head so the decoder regenerates frame0 from
   frame1+pose (frame0 is SegNet-blind) instead of decoding a full second frame. **−0.01 to −0.03** by nearly halving the
   frame0 representation cost. Pure arch lever (`renderer.py:1096` warp primitive exists).
5. **T7 — cross-pair pose-budget reweight** — weight the pose loss by per-pair achievability (pose is pooled-mean-before-
   sqrt → fungible across pairs); flow capacity to cheap pairs. **−0.005 to −0.015** as an aiming term.

---

## 4. SUBSUMED / SUBSTRATE-SPECIFIC / DON'T-DOUBLE-COUNT (the honest guard)

- **PR112 is NOT a separate idle bolt-on** — it IS R1+R2, already shipped on the frontier. The re-activation is porting
  the recode to base_ch=20's exported packet, not re-implementing it.
- **R2 ⊂ T1** — cross-pair clustering is a strictly stronger latent codec; don't count both savings independently.
- **R3 / frame0-pose-selector** are the same FECa selector — one mechanism, don't double-count.
- **Decoder coarsening / int4-6 PTQ/QAT/GPTQ/AWQ** — FALSIFIED ×2 (knife-edge); T11 (channel PRUNING) and WRQ
  (score-aware RE-allocation, not uniform precision reduction) are the distinct un-killed variants.
- **Frame-1 seg-repair as an UNCONDITIONAL sidecar** — info-theoretically net-negative (1.525 > 1.27 B/flip). LeverD's
  margin-CONDITIONAL form is the only viable version, and even it is conditional on flip concentration.
- **HFV sidecar recoder** — only relevant if base_ch=20 ships a foveation sidecar; otherwise N/A.
- **Adding files to shrink the rate denominator** — IMPOSSIBLE (`evaluate.py:64` rglobs the contest GT, N fixed).
- **Pose dims 7-12 as transport** — killed (scorer-internal, not archive bytes).

---

## 5. 6-HOOK WIRE-IN (Catalog #125) + provenance

- **#1 sensitivity-map:** WRQ/T9 consume per-tensor |grad|; T1/T8 consume latent per-pair/per-dim sensitivity; LeverD
  consumes the SegNet margin field; T5/T2 consume the invisibility basis + B2 Y-fraction.
- **#2 Pareto:** R1/R2/R3/S12/T1/T4/T8/T9/WRQ move RATE only (orthogonal to the saturated distortion vertex);
  T5/T11/T2/LeverD/PR98/T10 are the distortion-axis moves.
- **#3 bit-allocator:** WRQ + T1 dictionary-index + T9 cluster/perm ARE bit-allocator primitives (PR95 L21–L32 family);
  T5 is a training-time allocator.
- **#4 cathedral-autopilot:** the FP11 recode bundle (R1/R2/R3/T1/T9) folds into the `byte_range_entropy_recode_chain`
  materializer + paired-eval dispatch surface (same harvest queue).
- **#5 continual-learning:** T1's k-means result + T11's prune-finetune d_seg + WRQ's exact sweep reseed the judge on
  whether cross-pair / sparsity / score-aware-requant are live axes on the new base.
- **#6 probe-disambiguator:** every UNBUILT row has a $0 local first-test (T1: k-means bytes vs 15,387 B; T8:
  null-projected latent entropy; WRQ: per-tensor finite-difference sensitivity; LeverD: |B| flip concentration;
  T11: prune-finetune knife-edge).

**Provenance:** sourced from `leapfrog_pr112_absorb_recode_verdict_20260610.md` (PR112 absorbed),
`untapped_technique_inventory_20260610.md` (T1–T11 ranked), `stacking_synergy_composition_plan_20260610.md`
(orthogonality map), `composition_algebra_coherence_law_20260610.md` (additive-stacking proof), the lane registry, and
the module headers of `pr110_payload_entropy_recode`, `score_aware_weight_requant_sweep`,
`boundary_math/margin_conditional_residual`, `build_hfv_sidecar_recoder`, `engineered_corrections`. Frontier read from
pointer. NO score claim; `[macOS-CPU advisory]`. Exact paired CPU+CUDA eval on the byte-closed base_ch=20 candidate is the
only authority for any predicted ΔS. NO FAKE: every magnitude is either a cited measured anchor or a derived band tagged
as such, and every reuse target names a real in-tree module.
