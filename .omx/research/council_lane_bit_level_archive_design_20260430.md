# Council Design Review — Lane Bit-level archive optimization

**Status:** Phase A council review for Level 0 → Level 1 graduation.
**Anchor:** Lane G v3 = 1.05 [contest-CUDA]; Lane A reference archive = 694,045 bytes.
**Predicted band [prediction]:** Empirically refined per `custom_binary_container.py` audit.
- **Outer container savings (ZIP overhead):** 0–500 B on Lane A (overhead is only 328 B = 0.05%
  of archive). Carmack's quoted "50 KB" target is **unreachable** on the outer wrapper.
- **Payload-side bit-level savings (the real lane):** 1–8 KB net on Lane G v3 via:
  - Stream-interleaved Brotli on (renderer.bin + poses.pt) when the inflate path supports it
  - Deduplication of common subsequences between mask frames (most found by AV1; small residual)
  - Custom shared-prefix-table encoding for renderer FP4 codebook + Brotli dictionary
  - Bit-packing of poses.pt fields below the FP16 minimum (FP12 / FP10 if dynamic range allows)
**Cost estimate:** $0 (pure-CPU bit-level analysis; no GPU). Wall-clock: ~30 min per archive.
**Dependencies:** existing `src/tac/custom_binary_container.py` (PACT_BC_v1 wrapper),
`src/tac/archive_diet_pack.py` (Brotli + deterministic ZIP — already captures the headroom),
`src/tac/archive_optimizer.py` (existing opt routines).

## 1. Existing scaffold audit

`src/tac/custom_binary_container.py` (213 LOC) **HARD TRUTH BOMB IN DOCSTRING:**

> "Lane A's reference archive is 694,045 bytes. Its three member payloads occupy
> 693,717 compressed bytes, leaving only 328 bytes of ZIP metadata and central-directory
> structure. **That means outer-container rewrites cannot plausibly save 50 KB on this
> archive: the non-payload budget is only 0.05% of the file.**"

**This is the empirical anchor that REDIRECTS Lane Bit-level scope.** The
"Carmack-container 50 KB" target was based on flawed accounting. The actual headroom is in:
1. **Payload bit-level optimization** (NOT outer container)
2. **Cross-stream deduplication** (renderer.bin ↔ poses.pt shared substrings via Brotli dict)
3. **Sub-FP16 bit-packing** of poses.pt (if dynamic range allows FP10/FP12)
4. **Shared prefix tables** for FP4 renderer codebook + magic headers

Existing modules:
- `archive_diet_pack.py` (Subagent L's lane) — Brotli + deterministic ZIP — saves ~14.7 KB
  on Lane A. This is the existing baseline to BEAT.
- `custom_binary_container.py` — outer ZIP rewriter. Saves at most 0–500 B on Lane A. NOT
  the right scope for Lane Bit-level.
- `archive_optimizer.py` (254 LOC) — existing optimizer routines. Lane Bit-level extends.

## 2. Math foundation

### 2.1 Information-theoretic bounds on archive bytes

Per Shannon source-coding theorem: the minimum bit count to losslessly encode a stream is
its entropy `H(X) = -Σ p(x) log p(x)` per symbol. Archive byte count is bounded by:

    archive_bytes ≥ Σ_streams H(stream) / 8 + container_overhead

For Lane A:
- renderer.bin (FP4 weights, ~64 KB at Quantizr; 184 KB at Lane G v3 due to bigger model)
  has empirical entropy ~3.4–3.6 bits/symbol per Lane 20 design doc — already near floor.
- masks.mkv (AV1 monochrome) has empirical bit budget ~0.014 bpp = 14 bits per pixel pair
  per Lane 9 design doc — already near floor for the pixel-domain representation.
- poses.pt (FP16 6-DoF × 1200 frames × 2 bytes = 14.4 KB raw) — most heavily quantizable.
- container overhead: 328 B (deterministic ZIP central directory).

**Aggregate: the vast majority of archive bytes are at-or-near per-stream entropy bounds.
Bit-level wins come from CROSS-stream coding** (Brotli sharing dictionaries across
renderer/poses) and **representation changes** (FP16 → FP10/FP12 on poses).

### 2.2 The shared-prefix table (Hotz pragmatic shortcut)

Brotli supports a shared dictionary loaded BEFORE compression. If renderer.bin and poses.pt
share common 8-byte subsequences (e.g. zero runs, FP4 codebook patterns), preloading a
shared dict reduces compressed size of both. Practical savings on Lane A: ~0–2 KB
(empirically dependent on cross-stream similarity).

### 2.3 Bit-packing below FP16 (the poses.pt lane)

Lane G v3 poses.pt holds 6-DoF × 1200 frames at FP16 = 14,400 bytes. Per-dim dynamic ranges:
- dim 0 (longitudinal trans): ±0.1 typical — needs ~10 bits mantissa
- dim 1 (lateral trans): ±0.05 — needs ~9 bits
- dim 2 (vertical trans): ±0.02 — needs ~8 bits
- dims 3-5 (rotations, radians): ±0.01 — needs ~8 bits

A per-dim 8-bit quantization (with per-dim scale + offset) gives ~7,200 bytes — saves ~7 KB.
But: contest pose distortion is sensitive to numerical precision; round-off below FP12 may
move pose distortion above the 1e-4 threshold (per Lane RAFT design doc §3 Yousfi seat).
**Empirical gating required.**

### 2.4 Cross-stream deduplication (van den Oord vector-quantization analog)

If renderer.bin contains repeating FP4 codebook indices (common patterns) AND poses.pt
contains repeating quantized pose deltas, a SHARED VQ codebook across both could save
~1-3 KB. Closely related to Lane J-NWC (which trains a shared codec across renderers); this
lane handles the cross-stream sharing within a SINGLE archive.

## 3. Council deliberation

### Carmack (LEAD — channeled, raw engineering)

> "Look, my original instinct was 'rip up the ZIP container, save 50 KB.' That was wrong;
> the audit module already proved it (328 bytes of overhead, 0.05% of the file). I take
> the hit. The 50 KB target is dead.
>
> The REAL Carmack-style win here is **bit-pack the poses.pt below FP16**. 14.4 KB at
> FP16 → 7.2 KB at int8 → -7 KB net = -0.005 score. That's 14× the entire outer-container
> headroom in a 200-line patch. Do this first.
>
> Second: shared Brotli dictionary across renderer.bin + poses.pt. Train the dict at
> compress time on the concatenated streams; ship the dict (~500 B) + per-stream Brotli.
> Saves ~1-2 KB if the streams share common substrings. Cheap to implement, cheap to test.
>
> Third (deferred): rip up the ZIP container is OFF THE TABLE per the audit. Stop chasing it."
>
> **Verdict:** GREEN with REDIRECTED SCOPE. Original 50 KB target was wrong; pivot to
> payload-side bit-packing.

### Hotz (raw engineering)

> "Carmack just nailed it. The Hotz extension: **also bit-pack the poses.pt with PER-DIM
> dynamic-range adaptation**. Don't use a single global int8 quantizer; use 6 separate
> per-dim quantizers. Some dims have wider range than others; the per-dim approach saves
> another ~1-2 KB.
>
> And: don't fucking touch the outer ZIP container. The audit was right."
>
> **Verdict:** GREEN.

### Selfcomp (block-FP author)

> "Block-FP per-channel is exactly the same idea Hotz just described, applied to renderer
> weights. We could apply block-FP to poses.pt too — block-FP-1.0 bpp instead of int8 (8 bpp)
> — saves a further ~50% on the pose stream. But the dynamic range of pose dims is much
> tighter than weight matrices; might just be int8 + per-dim scale, not full block-FP.
>
> Cross-stream Brotli dict: untested in our context, may or may not work. Recommend
> empirical first."
>
> **Verdict:** GREEN with empirical gating on cross-stream Brotli.

### Boyd (channeled — convex optimization)

> "The bit-level allocation problem is a discrete optimization: for each stream
> i ∈ {renderer, masks, poses}, choose bits b_i ∈ ℤ⁺ to minimize
>     |archive| = Σ b_i + container
> subject to:
>     score(b_renderer, b_masks, b_poses) ≤ score_target
>
> ADMM-decompose into per-stream rate-distortion: each stream solves its own problem at a
> shared Lagrangian λ on the score-budget. This is a cleaner formulation of Lane 10 ADMM
> wrapped around the bit-level allocation. **Lane Bit-level Phase 2 should integrate with
> Lane 10 ADMM.**"
>
> **Verdict:** GREEN with ADMM-integration roadmap.

### Quantizr (adversarial — leader at 0.33)

> "I shipped Brotli + FP4 + KL distill. My poses.pt is already small (~10 KB at FP4 +
> Brotli — I was MORE aggressive than you on pose quantization, by the way). My
> renderer.bin is ~64 KB. Total ~80 KB content + 14 KB masks ≈ 94 KB. Container overhead
> negligible.
>
> Lane Bit-level on Lane G v3 (1.05 score, ~700 KB archive — 7× larger than mine!) has
> way more headroom than Lane Bit-level on Quantizr-class (0.33 score, ~94 KB). The audit's
> right that you can save ~7 KB on poses.pt at Lane G v3 scale. But that's a -0.005 score
> move, NOT a -0.04 move.
>
> Set realistic expectations: this lane delivers small gains, not transformative ones."
>
> **Verdict:** YELLOW. Real but modest; tag predictions accurately.

### Shannon (LEAD — information theory)

> "Per the entropy bounds I derived: the streams are ALREADY near per-stream entropy
> floors. The remaining win is in (a) sub-FP16 bit-packing where dynamic range permits
> (poses.pt = clear win) and (b) cross-stream sharing (Brotli dict = empirical TBD,
> shared VQ codebook = Lane J-NWC territory).
>
> Predict 1-8 KB net savings on Lane G v3 archive. Below 1 KB → kill; above 8 KB → likely
> a measurement bug somewhere. Tag empirical claims with explicit byte counts."
>
> **Verdict:** GREEN with [prediction:1-8KB] band.

### Contrarian

> "The custom_binary_container audit ALREADY ATE THIS LANE'S LUNCH. The 'Carmack 50 KB'
> target is dead. Why are we even doing this lane?
>
> Counter: even 7 KB on poses.pt is real. Lane Bit-level redirected to payload-side is a
> different lane than 'rip up the ZIP container.' OK if scope is honest. Council REQUIRED
> to honestly tag the predicted band."
>
> **Verdict:** GREEN conditional on honest scope tagging (NOT 50 KB target).

### MacKay (channeled — MDL grandmaster)

> "Per Lane MDL framework (just landed): every codec change here MUST be measured by
> L_total = L(M) + L(D|M). The shared-Brotli-dict shipping cost (L(M) for the dict) MUST
> be subtracted from the L(D|M) savings. If net positive, it's worth shipping; if
> net negative, drop it.
>
> The bit-packed poses.pt scheme has L(M) = ~16 bytes (per-dim quantizer scale + offset)
> and L(D|M) = ~7,200 bytes vs ~14,400 baseline. Net savings: ~7,184 bytes. Worth shipping."
>
> **Verdict:** GREEN. Lane MDL framework is the natural ranking tool here.

## 4. Decision

**Adopt:** Implement Lane Bit-level archive optimizer with **payload-side scope only**.
The outer ZIP container is OUT OF SCOPE per the empirical audit.

**Architecture:**
- `BitLevelArchiveOptimizer(archive_bytes, baseline_metric)` — orchestrator
- `PoseStreamBitPacker(per_dim_quantizer_bits, scale_offset)` — sub-FP16 packing (the main win)
- `SharedBrotliDictBuilder(streams: list[bytes])` — train shared dict at compress time
- `CrossStreamDedupAnalyzer(streams)` — find common subsequences (research-grade only)
- `audit_archive_byte_composition(archive)` — surface where the bytes ACTUALLY are
  (the analysis that already exists in custom_binary_container's docstring, made executable)

**Wire format:**
- No new outer-container format (see Carmack verdict)
- Per-stream wire format extensions:
  - `poses.pt` → optional `BLPS1` magic-byte prefix + per-dim quantizer header (16 B) +
    int8/int4 per-dim packed body
  - `renderer.bin` + `poses.pt` → optional shared Brotli dict (~500 B) shipped as
    `archive.bdict`

**Strict-scorer-rule compliance:** YES (compress-time only; no inflate-time scorer load).

**Kill criteria:**
- If empirical net savings on Lane G v3 < 1 KB after all sub-techniques: abandon the lane.
- If pose distortion regresses > 1e-4 due to bit-packing: abandon bit-packing; keep
  shared-dict Brotli only.
- If shared Brotli dict + ship-the-dict cost is net negative: drop shared-dict; keep
  bit-packed poses only.

## 5. Phase ordering (operational)

1. **Phase A** (this doc) — DONE
2. **Phase B (Level 1)** — `src/tac/bit_level_archive_optimizer.py` skeleton + 8-12 synthetic
   tests
3. **Phase C (Level 2 prep)** — empirical audit on Lane G v3 reference archive; produce
   `reports/lane_bit_level_byte_composition.json`; tag `[empirical:...]`
4. **Phase D (Level 2)** — wire `PoseStreamBitPacker` into `compress.sh` archive build;
   inflate handles `BLPS1` magic-byte
5. **Phase E (Level 3 path)** — STRICT preflight Check XX (bit-packed pose has matching
   inflate handler); 3-clean-pass adversarial review
6. **Phase F** — measure on Lane G v3 contest-CUDA; tag result `[contest-CUDA]`

## 6. Cross-references

- CLAUDE.md "Council conduct — non-negotiable" (council redirected scope; not conservative
  bias — empirical evidence forced it)
- CLAUDE.md "Auth eval EVERYWHERE"
- `feedback_production_hardened_standard_definition_20260430.md`
- `project_phases_2_3_4_design_implementation_math_provenance_20260429.md` §"Lane 15 Bit-level"
- `src/tac/custom_binary_container.py` (the audit that REDIRECTED this lane's scope; HARD
  TRUTH BOMB in module docstring is the foundational evidence)
- `src/tac/archive_diet_pack.py` (Subagent L; the existing baseline)
- `src/tac/archive_optimizer.py` (sibling lane; existing optimizer routines)
- `src/tac/mdl_bayesian_codec.py` (just-landed Lane MDL — the natural ranking tool)
- Shannon 1948 — source coding theorem
- Brotli RFC 7932 (shared dictionary support)
