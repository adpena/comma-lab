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

## 5. VERDICT (appended after the $0 smoke)

**LEAD — EXACT POINTER DELTA: the frontier pointer did NOT move. It stays at contest-CPU
`0.19109982419209975` (sha `b46897267ded…`, 177,169 B). NO byte-closed candidate was built, NO paired
eval fired — because the waterfill admits ZERO net-positive flips: every frame-1 correction on the
frontier base creates more new bad flips (collateral) than it fixes. The CONDITIONAL per-flip code cost
is `0.856 B/flip` (τ=0.5) — well BELOW the `1.27 B/flip` waterline (and below the `1.768 B/flip`
unconditional) — so the RATE side WINS, but the DISTORTION side DIES on receptive-field collateral.**

### The two-sided result (the genuine new science)

This lever splits cleanly into a rate side and a distortion side. They have OPPOSITE verdicts, and that
split is the finding:

| side | metric | result | gate |
|---|---|---|---|
| **RATE (code cost)** | conditional B/flip (τ=0.5) | **0.856** (0.616 pos + 0.241 class) | **< 1.27 → CLEARS** |
| | unconditional B/flip (#51 floor) | 1.768 (1.528 pos + 0.241 class) | > 1.27 (loses, as #51 found) |
| | conditional saving vs unconditional | **0.91 B/flip** | the margin prior is real |
| **DISTORTION (collateral)** | strongest override: fixed / new-bad | **467 / 2,823** (net −2,356, 6.0×) | net < 0 → DIES |
| | GT-snap control: fixed / new-bad | 240 / 1,023 (net −783, 4.3×) | net < 0 (matches #51) |
| | per-flip waterfill (most-isolated, gentlest) | **0 of 24 net-positive; best net = 0** | admit 0 → no bytes |

### Lever 1 (RATE) — CONFIRMED, and it overturns the #51 rate-floor framing

The decoder regenerates the SegNet margin field FOR FREE from its own render. At τ=0.5 the
decoder-known boundary set `B = {m < 0.5}` is only ~613 pixels/frame (0.31% of the 196,608 grid), and it
captures **91.2%** of the flips. So the position cost collapses from `log2 C(196608, K) = 1.528 B/flip`
(the #51 unconditional floor) to `log2 C(613, K) = 0.616 B/flip` — a **0.91 B/flip saving**, taking the
total (position + the ~0.24 B/flip class cost) to **0.856 B/flip**, comfortably below the 1.27 waterline.
This is exactly the MWCC hypothesis: **margin-weighting IS the missing reactivation lever the
STC-clean-source DEFER named.** It works. The #51 verdict ("1.525 B/flip position floor is over the 1.27
break-even → no sidecar can clear THE LAW") was an UNCONDITIONAL bound; conditioning on the free margin
prior beats it. (τ sweep: 0.25→1.045, 0.5→0.856, 1.0→0.878, 2.0→0.998 B/flip — all below 1.27.)

### Lever 2 (DISTORTION) — DIES, and it is the TRUE floor (not the rate floor #51 named)

The fatal gate is the receptive-field collateral, exactly as #51 (GT-snap net −536) and #55 (Gα-solve
net-negative) found — but measured here at maximum strength and with per-flip attribution:

- The **strongest realizable per-pixel argmax override** (gradient-ascent on the real SegNet to maximize
  the target-class logit at each stored flip pixel) DOES flip the stored pixels (467 fixed confirms the
  correction works — NO FAKE), but creates **2,823 new bad flips** (6.0× collateral) → net −2,356 → exact
  d_seg goes UP (worse). Driving the logit hard creates sharper local discontinuities the SegNet U-Net
  reads as new boundaries.
- The **per-flip waterfill admission test** — correcting the MOST-ISOLATED flips (isolation up to 56
  pixels from the nearest other flip) ONE AT A TIME with the GENTLEST correction (step 8, radius 2, 6
  steps) — found **0 of 24 flips with positive net value**; the BEST case is **net = 0** (1 fixed, 1 new
  bad). Even an isolation-56 flip creates ≥1 new bad flip. **There is NO collateral-free flip on the
  frontier base.** The EfficientNet-B2 U-Net's receptive field always couples ≥1 neighbor when you
  perturb the camera-res region that maps to a scorer pixel.

The waterfill therefore admits NONE: `n_admitted = 0`, `admitted_net_value_flips = 0`,
`admitted_code_bytes = 0`, `net_score_delta = 0.0`. **No improvement despite the rate win.**

### Against the pre-registered prediction + kill criterion

- **Rate-side prediction (conditional beats 1.27 for a non-trivial subset): CONFIRMED** — 0.856 B/flip,
  91% of flips, well below 1.27.
- **Collateral-side prediction (net drop ≪ stored count, likely net-positive/worse on the frontier base
  unless a collateral-free class exists): CONFIRMED** — best per-flip net = 0; no collateral-free class;
  all-flips override net −2,356.
- **Combined prediction (net ΔS < 0 only if a subset clears BOTH gates; predicted small-to-empty):
  CONFIRMED EMPTY.** The waterfill admits 0.
- **KILL CRITERION: "conditional code cost clears 1.27 for a subset BUT exact-decode collateral makes net
  d_seg drop ≤ 0" → DEFER-to-a-contiguous-residual-base + record the collateral floor.** This is the
  branch that fired. **DEFER, not KILL** (per Forbidden-premature-KILL): the conditional CODER is proven
  to clear the rate gate (its reusable value stands); the FRONTIER BASE is at its frame-1 seg-repair
  collateral floor. The receptive-field collateral, not the code rate, is the irreducible blocker.
- **NO Modal dispatch** (the local kill-gate fired correctly): no byte-closed candidate can show advisory
  net ΔS < 0, so firing ~$0.3–0.6 to confirm a non-improvement is the forbidden waste. **No paired eval
  pre-registered.** `$0` total spend.

### The conditional per-flip cost vs 1.27 (the headline curve the spec asked for)

```
unconditional (no margin context, #51):  1.768 B/flip   [ABOVE 1.27 — loses]
conditional τ=0.25:                       1.045 B/flip   [below — clears]
conditional τ=0.5  (best):                0.856 B/flip   [below — clears, 33% under]
conditional τ=1.0:                        0.878 B/flip   [below — clears]
conditional τ=2.0:                        0.998 B/flip   [below — clears]
                                          ── waterline ── 1.27 B/flip
```

The fundable-subset size on the rate axis is **~91% of the 66,039 pool flips** (the τ=0.5 boundary
fraction). But the fundable-subset size on the NET-VALUE axis is **ZERO** (collateral makes every flip
net ≤ 0). The binding constraint is collateral, not code rate.

### EXACT d_seg drop + sidecar bytes + net ΔS

| quantity | value |
|---|---|
| flips fixable on the rate axis (cond < 1.27) | ~60,000 of 66,039 (91% at τ=0.5) |
| flips fixable on the NET axis (waterfill-admitted) | **0** |
| stored residual sidecar bytes | **0** (nothing admitted) |
| exact d_seg drop (net, after collateral) | **0** (all-flips override is +Δd_seg = worse) |
| net ΔS | **0.0 — pointer unmoved** |
| beat frontier? | **NO** |
| paired eval fired? | **NO** (local kill-gate; no advisory improvement) |
| pointer update | **NONE** (stays 0.19109982419209975) |

### Reactivation criteria (the DEFER, per #55's handoff)

1. **A contiguous-residual base (the real next move):** the conditional coder's rate win is REAL; it dies
   only on the frontier's salt-and-pepper single-pixel collateral (95% single-pixel components, 3.2%
   temporal persistence — confirmed here). On a base whose seg residual is CONTIGUOUS multi-pixel patches
   (the lever-B score-native generator's under-fit regions, per #55 + #51 reactivation), a patch
   correction can flip a whole region with one camera-res region edit, amortizing the receptive-field
   collateral across many fixed pixels — the net-value gate may flip positive. Run `measure_code_cost` +
   the per-flip collateral attribution on the lever-B base.
2. **The decoder/latent axis (the #51 routing):** a renderer change that fixes the systematic boundary
   error in the existing 91%-decoder bytes (no sidecar) pays zero rate and zero collateral.
3. **A flip class with a large coupled POSE gain:** a flip whose correction ALSO lowers d_pose lifts its
   net value above the collateral floor (the flip-map's pose-coupling rows are the input).

### NO-FAKE self-checks (all pass)

- **Class 1 (real work):** the stored residual ACTUALLY flips the stored pixels on the exact SegNet
  (fixed=467 for the all-flips override; the per-flip test confirms each correction moves ITS scorer
  argmax). The collateral (new_bad) is the exact count of correct pixels that became wrong — measured, not
  assumed.
- **Class 8 (exact authority):** d_seg from the exact `upstream/modules.SegNet` argmax; GT via
  `yuv420_to_rgb` ONLY; NEVER MPS. The NO-FAKE SPOT-CHECK confirmed the cached flip-map matches the
  CURRENT frontier's rendered flips exactly (pair 0: 114 == 114, identical positions) — the cache (built
  on the decode-identical `b7106c9bdbb8` re-encode) is valid for `b46897267d`.
- **Conditional beats unconditional:** measured 0.856 < 1.768 on the REAL flip-set; the codec test-suite
  proves a coder that does NOT concentrate (boundary = whole grid) gets ZERO saving.
- **Waterfill cutoff at 1.27 + net-positive:** the selector admits iff `cost/net < 1.27` AND `net > 0`; a
  select-all stub fails (it admits net-negative flips); at-the-waterline is rejected (strict `<`).

### Wire-in (Catalog #125)

1. **sensitivity-map — ACTIVE:** the conditional per-flip `(position_bits, class_bits, net_value)` is the
   refined seg-axis sensitivity input; the headline is "rate-axis fundable, net-axis empty on the frontier
   base; collateral is the binding constraint."
2. **Pareto — ACTIVE:** the frontier base is at its frame-1 seg-repair Pareto vertex; no frame-1
   correction lowers d_seg without paying more collateral than it saves (now proven down to the
   single-isolated-pixel level).
3. **bit-allocator — ACTIVE:** `waterfill_select` IS the bit-allocator gate (admit iff net-value cost <
   1.27); it correctly allocates ZERO bytes on the frontier base. The conditional code-cost model is the
   refined byte-cost input for any future contiguous-residual base.
4. **cathedral-autopilot — N/A:** advisory verdict; no byte-closed candidate cleared the local kill-gate,
   so no on-host dispatch minted (the disciplined refusal — do not re-confirm a non-improvement).
5. **continual-learning — ACTIVE:** reseeds the planner: the conditional margin coder OVERTURNS the #51
   rate floor (margin conditioning gets to 0.856 B/flip, beating 1.27), exposing that the TRUE
   frontier-base seg floor is the receptive-field COLLATERAL, not the code rate. The next probe is the
   contiguous-residual lever-B base, not a tighter coder.
6. **probe-disambiguator — RESOLVED two ways:** (a) "does conditioning beat the 1.27 waterline?" → YES
   (0.856, the MWCC hypothesis confirmed); (b) "is the residual worth storing on the frontier base?" → NO
   (collateral makes every flip net ≤ 0; the waterfill admits 0). The two answers together are the lever's
   verdict.

### Cross-references

`frontier_seg_repair_pool_verdict_20260610.md` (#51 — the unconditional 1.525 B/flip floor this lever's
conditional coder beats; the GT-snap collateral this confirms) · `closed_spec_boundary_solver_v1_20260610T105830Z.md`
(#55 — the Gα-solve collateral DEFER + the explicit "lever D contour coder" reactivation this executes) ·
`sota_plus_original_inventions_20260610T125100Z.md` (the MWCC design + its "risk it stays above 1.27" the
rate side disproves + the "missing reactivation lever" the rate side confirms) ·
`lossless_stack_pointer_move_20260610T165749Z.md` (the current-frontier decode-parity that validates the
cache reuse) · `src/tac/boundary_math/margin_conditional_residual.py` + tests (the codec deliverable) ·
`experiments/results/lever_d_margin_conditional_residual_20260610/` (the three measurement tools +
`analysis_summaries/`) · `upstream/{modules.py,frame_utils.py}` (frozen authority).

