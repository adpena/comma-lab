# Lever D — the margin-conditional residual coder (task #72) — PRE-REGISTRATION

**Status:** PRE-REGISTRATION (written BEFORE the conditional-entropy measurement, per the task spec's
"PRE-REGISTER FIRST" discipline). The verdict section is appended after the $0 smoke.

**Authority:** every number below is `[macOS-CPU advisory]` / `[local CPU-torch advisory]` — exact upstream
`SegNet` argmax on CPU, GT decoded via `upstream/frame_utils.yuv420_to_rgb` ONLY (NEVER PyAV rgb24 = ~100×
phantom pose; NEVER MPS). NOT the contest 600-sample harness → non-promotable per the GOAL authority ladder
unless a byte-closed candidate beats the frontier and a paired CPU+CUDA exact eval ratifies it. `$0` smoke,
≤`$1` if a confirming paired eval fires.

**Frontier (the target to beat):** contest-CPU `0.19109982419209975`, archive sha `b46897267ded…`,
**177,169 B**, `D = 37,545,489`. Components (both-axis paired, lossless-proven): `d_seg = 5.5978e-4`,
`d_pose = 2.942e-5`. Seg term `100·d_seg = 0.0560` (29% of the score = the largest single pool).

---

## 0. The lever (what task #72 builds) + how it differs from the #51 DEFER and the #55 SOLVE

Task #72 builds the **margin-conditional residual coder + waterfill selector**: a frame-1 seg-repair
sidecar whose per-flip code cost is conditioned on structure the decoder regenerates FOR FREE (it has the
renderer → it recomputes the SegNet margin field), so the sidecar stores only the CONDITIONAL residual
(among low-margin flip-prone pixels: which flipped + to which class), clustered spatially + predicted
temporally, and WATERFILLED to the subset coding below the `1.27 B/flip` break-even.

**This is the explicit reactivation lever the two prior verdicts named:**

- `#51` (`frontier_seg_repair_pool_verdict`): DEFER-pending-new-carrier. Found the position-only
  **information-theoretic floor = 1.525 B/flip** (`log2 C(196608, 110)` for K=110 scattered flips/pair) is
  ALREADY OVER the 1.27 break-even — **but that is the UNCONDITIONAL floor**. The reactivation criterion it
  named: "a class-of-flips spatially DENSE enough that position entropy drops below 1 B/flip" + the
  shared-field / per-pair carriers it falsified did NOT condition on the free margin prior. THIS lever tests
  whether conditioning beats the unconditional floor.
- `#55` (`closed_spec_boundary_solver_v1`): the polytope Gα≥b SOLVE flips targets in closed form but pays
  net-negative COLLATERAL on the frontier base (GT-snap upper bound: 594 new bad flips for 58 repaired). It
  DEFERRED the frontier-base correction and named **"lever D contour coder: if a base has contiguous
  residual, the MDL candidate's chain-code cost can be pushed below 1.27 B/flip with an STC/UNIWARD
  boundary coder, crossing the water level."**
- The MWCC design memo (`sota_plus_original_inventions`): "margin-weighting is exactly the cost-map that
  [the STC-clean-source] DEFER said was the missing reactivation lever, so this is the test of that
  hypothesis." Risk it stays above 1.27 even with margin weighting.

**Two physics levers, measured separately and honestly:**
1. **CODE COST (the rate side):** does conditioning the position+class entropy on (margin, spatial
   neighborhood, temporal previous-frame) push a fundable SUBSET below 1.27 B/flip? (the #51 unconditional
   floor was 1.525.)
2. **COLLATERAL (the distortion side):** when the stored residual is APPLIED at decode (set the stored
   pixels to a color yielding the target argmax), does it ACTUALLY drop exact d_seg by the stored subset
   with manageable net collateral? (#51 GT-snap + #55 Gα solve BOTH found net-negative collateral on the
   frontier base.) **This is the harder gate.** The #51 finding that frame-1 patches create net-new flips
   via the SegNet receptive field is the standing threat to this lever. The honest test: a SINGLE-pixel
   argmax override at the scorer grid (not a camera-res appearance patch) measured for exact net flips.

---

## 1. PRE-REGISTERED PREDICTION

**Rate side (lever 1):** Conditioning on the free margin field will lower the per-flip CODE cost meaningfully
versus the unconditional position floor, because the margin field localizes flips to the fragile boundary
band (the decoder knows WHERE flips are possible without being told), collapsing the position alphabet from
196,608 candidate pixels to the ~`margin<0.5` boundary set. **Prediction: the conditional code cost for the
densest/most-predictable subset drops below 1.27 B/flip for a NON-TRIVIAL subset** (>~5% of flips), because
the margin prior + spatial run structure + temporal persistence are real, decoder-free side information.

**Collateral side (lever 2):** This is where I predict the lever most likely DIES on the frontier base. The
#51 + #55 receptive-field findings predict that even an exact single-pixel argmax override pays net-negative
collateral. **Prediction (honest, pessimistic): the net d_seg drop after collateral will be far smaller than
the stored-flip count, and may be net-positive (worse), confirming the frontier base is at its frame-1
seg-repair floor** — UNLESS a class of flips exists whose single-pixel override is collateral-free
(isolated flips far from other boundaries). The waterfill must select on NET value (drop − collateral −
rate), not gross stored count.

**Combined prediction:** net ΔS < 0 (beats frontier) only if a subset clears BOTH the 1.27 B/flip code gate
AND has near-zero collateral. I predict this subset is small-to-empty on the frontier base (consistent with
#51's information-theoretic floor + #55's collateral DEFER), but the CONDITIONAL code-cost curve + the
exact collateral measurement are the genuine new science either way.

---

## 2. PRE-REGISTERED KILL / VERDICT CRITERION

- **NET ΔS < 0 (byte-closed, exact-decode d_seg drop − rate) → FRONTIER-CANDIDATE.** Run the confirming
  paired CPU+CUDA exact eval (~$0.3–0.6, lane-claim + HARVEST-OR-LOSE), then UPDATE the canonical frontier
  pointer. This is the only outcome that moves the pointer.
- **The conditional per-flip code cost is ≥ 1.27 B/flip for EVERY subset** → the residual is NOT worth it on
  the rate side; CONFIRM the #51 dormant verdict (1.525 B/flip unconditional ≈ confirmed) and RECORD THE
  CONDITIONAL FLOOR (the headline number: how far below 1.525 did conditioning get, and is it still ≥ 1.27?).
  DEFER (not KILL) per Forbidden-premature-KILL.
- **The conditional code cost clears 1.27 for a subset, BUT the exact-decode collateral makes net d_seg drop
  ≤ 0 (or net ΔS ≥ 0)** → the rate lever WORKS but the frontier base's collateral kills it; DEFER-to-a-
  contiguous-residual-base (the lever-B score-native generator, per #55) and record the collateral floor.
- **NO Modal dispatch** unless a byte-closed candidate shows advisory net ΔS < 0 (the local kill-gate; never
  burn ~$0.3 to confirm a non-improvement).

---

## 3. METHOD (the $0 smoke → byte-close → exact)

1. **Flip-set (REUSE the validated cache, spot-checked):** the #51 flip-map at
   `/Volumes/VertigoDataTier/pact/frontier_seg_repair_pool_20260610/flip_map_full/` (per-pair `flip_idx`,
   `flip_margin`, `flip_gtcls` for all 600 pairs; d_seg recomputed 5.5982e-4 == frontier to 5e-8). It was
   computed on archive `b7106c9bdbb8` (FP11-source-brotli-recode); the CURRENT frontier `b46897267d`
   (payload-entropy-recode) is a LOSSLESS re-encode of the SAME procedural HNeRV (decode-parity proven,
   d_seg identical) → the decoded frames + flip-set are IDENTICAL. **NO-FAKE spot-check:** render+score 2–3
   pairs on the CURRENT archive bytes and assert the flip count matches the cache before trusting it.
2. **Conditional coder (the build):** model `P(flip + target_class | margin-bin, spatial-neighborhood,
   temporal-previous-frame-flip)` and measure the realized conditional code cost (arithmetic/empirical
   entropy in bits) per flip — vs the unconditional bitmap+class cost AND vs 1.27 B/flip. The margin field
   is the decoder-free context (verify the decoder CAN regenerate it = it has the rendered frame).
3. **Waterfill:** rank flips by conditional code cost; take the subset coding BELOW 1.27 B/flip.
4. **Byte-close + exact:** store the subset residual as a sidecar; the decoder applies the single-pixel
   argmax override (measure the EXACT net d_seg drop on the real SegNet — the stored flips fixed MINUS new
   bad flips created), and the sidecar bytes. Net ΔS = −(seg drop) + 25·sidecar_bytes/D.
5. **Gate:** if advisory net ΔS < 0 → paired CPU+CUDA exact eval → pointer update. Else → DEFER + floor.

---

## 4. NO-FAKE self-commitments (verified in the verdict section)

- **Class 1 (real work on real inputs):** the stored residual must ACTUALLY fix the stored flips on the
  EXACT SegNet (decode → argmax compare), not just claim to. Measured net = fixed − new_bad.
- **Class 8 (exact authority):** rank/verdict d_seg from exact `upstream/modules.SegNet` argmax (popcount);
  GT via `yuv420_to_rgb` ONLY; NEVER MPS; the rounded final_score lies → recompute from components.
- **Conditional-beats-unconditional:** the conditional code cost must be MEASURED below the unconditional
  baseline on the REAL structured flip-set (a constant/no-op coder, or one that ignores the margin context,
  must NOT beat it — tested).
- **Waterfill cutoff at 1.27:** the selector admits iff conditional cost < 1.27 B/flip (tested; a select-all
  stub fails the discrimination test).

---

## 5. (verdict appended below after the smoke)
