# Optimal = adaptive waterfilling at every dimension, derived from evaluate.py sensitivity (the non-arbitrariness principle)

UTC 2026-06-09 · claude · synthesis in response to the operator's rapid-fire design
questions (Q1 "optimal = adaptive quantization + waterfilling?"; Q2 "give each
precisely what it needs, in relation to all"; Q3 "are we applying ALL the lessons —
Quantizr/self-comp/Huffman/entropy/CPU-frontier?"; Q4 "uniform fp16 can't be optimal
given evaluate.py"; Q5 "adaptive + dynamic everywhere, all dimensions"; Q6
"non-arbitrariness principle"). [macOS analysis — design memo, not a score claim.]

## 0. The one-sentence answer
The operator is right on every count: **the optimum is NOT uniform anything — it is
adaptive, dynamic, sensitivity-weighted allocation at every dimension, and the law
that forbids uniform/conventional choices is the non-arbitrariness principle: every
bit, atom, gradient, and even every inter-agent hand-off must be DERIVED from the
measured marginal sensitivity of `upstream/evaluate.py` on the contest video, never
chosen by convention.** Uniform fp16 (and uniform int8) are arbitrary and therefore
provably suboptimal. fp16 in the running authority trace is only a CONTROL VARIABLE,
never the proposed fix.

## 1. The non-arbitrariness law (the spine of everything)
`S = 100·d_seg + sqrt(10·d_pose) + 25·bytes/37,545,489`. An allocation is OPTIMAL iff
every unit of cost (a bit, a parameter, a byte, a gradient step, a token of context)
is placed where its marginal `∂S/∂unit` is most negative — i.e. waterfilling. Any
allocation chosen by convention ("use fp16 everywhere", "uniform int8", "dense pixel
MSE", "send the whole transcript") spends cost where the evaluator does not look. So:

> Non-arbitrariness ⇒ adaptive/dynamic everywhere ⇒ waterfilling. They are the same
> principle stated at the level of (a justification, an architecture, an algorithm).

## 2. What evaluate.py actually rewards (the measured sensitivity map)
Verified from `upstream/modules.py` + `frame_utils.py` + the B0.5/B2 atlas this session:
- **SegNet**: reads ONLY frame1 (`x[:,-1]`), 512×384, 5-class argmax; `d_seg` = fraction
  of pixels whose argmax flips. ~95% of pixels are ROBUST (large logit margin); only
  ~4.8% are FRAGILE (boundaries, thin classes). A bit spent making a robust pixel's RGB
  more accurate buys **zero** d_seg. Sensitivity is sparse + spatial + frame1-only.
- **PoseNet**: reads BOTH frames as YUV6, normalized, FastViT-T12, first 6 of 12 pose
  dims; `d_pose` = MSE on those 6. Pose is ~96% LUMINANCE, near-full-rank across the
  video (dense carrier needed), both frames matter. Chroma precision + the 6 unscored
  dims buy ~zero d_pose. Sensitivity is dense + temporal + Y-dominant.
- **Rate**: every byte costs `25/37,545,489 ≈ 6.66e-7`. A byte that doesn't cut
  `100·d_seg + sqrt(10·d_pose)` is pure loss.

**Therefore uniform anything is suboptimal by construction** (Q4 confirmed): the optimal
codec spends high precision on the pose-Y tensors + the seg-boundary tensors and ~nothing
on chroma / robust-region / unscored-dim weights. Uniform fp16 over-pays the 95% the
evaluator ignores; uniform int8 may starve the sensitive 5%.

## 3. Optimal = adaptive/dynamic at EVERY dimension (Q1+Q5)
Waterfilling is the meta-principle; it instantiates at every level of the stack:
- **Codec bits** → per-tensor / per-channel MIXED bit-depth chosen by `∂S/∂precision`
  (PR95 L21 byte-maps, L22 perms, L23 split-brotli, L24 raw-LZMA latents, L25 temporal
  delta, L29 fp16-per-tensor-scales, L30 range/arithmetic coding, L26/L31 canonical-
  Huffman/colex sidecar, L32 brotli-q11). This IS "adaptive quantization."
- **Action atoms** → admit σ iff exact `ΔS(base+σ) < 0` (the V3 waterfiller; STEP-1
  ingest + router landed this session).
- **Training gradient** → margin-weighted seg loss on the fragile 5% + Y-dominant pose
  loss (score-aware, not uniform pixel MSE) + eval-roundtrip + EMA.
- **Latents** → temporal-delta + range-coded (dashcam temporal smoothness).
- **Inter-agent context** (the surprising 4th dimension) → each AI surface receives the
  minimal-sufficient typed artifact, not the whole transcript (see §6).

## 4. Are we applying the full arsenal? HONEST AUDIT (Q3)
Distinguish BUILT (exists in the repo) from WIRED-IN-THE-ACTIVE-PATH (the R3 pilot used it).

**BUILT + present (verified this session):**
- `src/tac/substrates/_shared/decoder_state_codec.py`: `int{2,4,6,7,8}_per_channel_axis0_fp16_scale`
  (+ `_bitpacked`, `_scale_bundled`) — adaptive per-channel, variable-bit quant family +
  a candidate-selection sweep. THE adaptive-codec arsenal EXISTS.
- `src/tac/substrates/hi_nerv/archive.py`: range coding via `constriction`
  (RangeEncoder/Decoder + Categorical) = PR95 L30. EXISTS.
- `archive_candidate.py`: temporal-delta (L25) + parseback-health (seg/pose) selection.
- The trainer: EMA 0.997 (Quantizr), eval-roundtrip-STE (L8), 8-stage PR95 curriculum
  (L14), Muon final-stage (L15), C1a coder-aware (L16), PixelShuffle+sin decoder (L18),
  per-pair latents (L19), score-aware seg/pose loss (L1/L6). WIRED into R3.

**NOT yet wired in the active R3 path (the gap the operator is pointing at):**
- The R3 export PINNED a SINGLE codec `PILOT_DECODER_CODEC = "int8_mixed"`
  (`watch_and_harvest_b1_checkpoint.py:136`). It did NOT run the sensitivity-driven
  per-tensor bit-allocation sweep across the int2..8/per-channel family. That is the
  ARBITRARY choice to replace with adaptive selection.
- The full PR95 L21-L23/L26/L31 entropy stack (per-tensor byte-maps, conv-axis perms,
  split-brotli streams, canonical-Huffman/colex sidecar) is registered as canonical
  equations but is NOT confirmed wired into the hi_nerv MLX export codec.
- **The CPU frontier (0.19199) is a SEPARATE lineage**:
  `fp11_source_brotli_recode_b7106c9bdbb8`, 178,493 bytes — a SOURCE fp11-brotli-recode,
  NOT a HiNeRV neural carrier. The technique that made our best CPU score is not being
  composed with the HiNeRV stack at all.

**Verdict on Q3: PARTIALLY.** We have the arsenal; we are NOT yet adaptively composing
it by sensitivity in the active path, and the frontier-making source-recode lineage is
orphaned from HiNeRV. **V3 is precisely the mechanism to compose all of it
non-arbitrarily** — every codec option, every PR95 lesson, the source-recode, and the
neural carrier become candidate atoms/codec-choices judged by one currency: exact ΔS.

## 5. The fp16 authority trace is a CONTROL, not the optimum (reconciles Q1/Q4 with the running job)
The detached `r3_ep250_authority_trace` re-exports the SAME ep249 checkpoint at fp16 vs
int8 through the SAME inflate+evaluate.py. fp16 is the **controlled variable** that
isolates ONE question: did uniform int8 quantization destroy seg/pose, or is the model
itself bad? It is NOT a proposal to ship fp16.
- If fp16 scores good → int8 quant is the bug → the OPTIMAL fix is the **adaptive,
  sensitivity-weighted codec that already exists** (per-tensor mixed bit-depth + range
  coding), NOT uniform fp16. The trace's verdict string now says exactly this.
- If fp16 also bad → the model/carrier failed; no codec allocation can save a carrier
  that is wrong everywhere → fork the carrier first.

This is WHY the trace runs first: **adaptive quantization can only pay off on a carrier
that is evaluator-good at high precision.** The order is non-negotiable: (1) prove the
carrier good at high precision, (2) THEN adaptive-waterfill its codec to minimum bytes.

## 6. "Give each precisely what it needs, in relation to all" (Q2) — the same law at 3 levels
- **Evaluator terms**: SegNet needs only frame1 argmax-correct on the fragile 5% (no RGB
  fidelity, no frame0, no chroma). PoseNet needs both-frame Y-dominant fidelity on 6 dims
  (no chroma, no unscored dims, no seg structure). Rate needs only rent-paying bytes.
  The JOINT constraint: frame1 is shared (seg+pose), frame0 is pose-only — so frame0 is a
  pure pose-carrier surface and frame1 must satisfy both. Allocate accordingly.
- **Codec tensors**: pose-Y tensors + seg-boundary tensors get high bits; everything else
  int8-or-lower/pruned. Each tensor gets exactly the bits its `∂S/∂precision` justifies,
  in relation to the shared byte budget (waterfilling).
- **The 3 AI surfaces** (Claude / Codex gpt-5.5 / ChatGPT 5.5 Pro): each receives the
  minimal-sufficient TYPED artifact, not the whole transcript. ChatGPT 5.5 Pro (strategy)
  ← distilled results + open decision points; Claude (execute/measure) ← current
  authoritative artifacts (exact-eval JSON, atlas, frontier pointer) + the decision tree;
  Codex (adversarial review) ← the diff + the typed rows + a scoped question. They form a
  pipeline arbitrated by typed artifacts on disk (candidate-action rows, authority trace,
  campaign decision, council posterior) — chat-only context is the orphan/arbitrary path.

## 7. The actionable consequence (what to do once the trace lands)
1. Read the authority trace verdict (model-bad vs int8-quant-bug vs bridge-bug).
2. If int8-quant-bug → make the codec ADAPTIVE: wire the per-tensor sensitivity (from the
   B2 gradient atlas) into the existing int2..8/per-channel candidate sweep → select bit
   depth per tensor by `∂S/∂precision`; add the unwired L21-L23 entropy stack as codec
   atoms; each judged by exact ΔS (non-arbitrary).
3. Compose the orphaned `fp11_source_brotli_recode` frontier lineage as a V3 atom/codec
   option so the neural carrier and the source-recode are selected jointly by ΔS.
4. NEVER pin a uniform codec again without a sensitivity justification — the
   non-arbitrariness principle is now the gate.

## Cross-refs
- `evaluator_gradient_atlas_b2_verdict_20260609.md` (pose dense/Y-dominant; seg sparse/boundary).
- `segnet_margin_field_20260609.json` (4.83% fragile support).
- CLAUDE.md "HNeRV/leaderboard binding-depth discipline" L14-L32 (the arsenal).
- CLAUDE.md "Meta-Lagrangian/Pareto solver" + "Bit-level deconstruction and entropy discipline".
- `src/tac/optimization/harvest_evidence.py` (the V3 waterfiller currency).
